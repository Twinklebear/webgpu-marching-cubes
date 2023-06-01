import {ExclusiveScan} from "./exclusive_scan";
import {MC_CASE_TABLE} from "./mc_case_table";
import {StreamCompactIDs} from "./stream_compact_ids";
import {Volume} from "./volume";
import {compileShader} from "./util";

import computeVoxelValuesWgsl from "./compute_voxel_values.wgsl";
import markActiveVoxelsWgsl from "./mark_active_voxel.wgsl";
import computeNumVertsWgsl from "./compute_num_verts.wgsl";
import computeVerticesWgsl from "./compute_vertices.wgsl";
import {PushConstants} from "./push_constant_builder";

export class MarchingCubesResult
{
    count: number;
    buffer: GPUBuffer;

    constructor(count: number, buffer: GPUBuffer)
    {
        this.count = count;
        this.buffer = buffer;
    }
};

/* Marching Cubes execution has 5 steps
 * 1. Compute active voxels
 * 2. Stream compact active voxel IDs
 *    - Scan is done on isActive buffer to get compaction offsets
 * 3. Compute # of vertices output by active voxels
 * 4. Scan # vertices buffer to produce vertex output offsets
 * 5. Compute and output vertices
 */
export class MarchingCubes
{
    #device: GPUDevice;

    #volume: Volume;

    #exclusiveScan: ExclusiveScan;

    #streamCompactIds: StreamCompactIDs;

    // Compute pipelines for each stage of the compute 
    #markActiveVoxelPipeline: GPUComputePipeline;
    #computeNumVertsPipeline: GPUComputePipeline;
    #computeVerticesPipeline: GPUComputePipeline;

    #triCaseTable: GPUBuffer;

    #volumeInfo: GPUBuffer;

    #voxelActive: GPUBuffer;

    #volumeInfoBG: GPUBindGroup;

    #markActiveBG: GPUBindGroup;

    // Timestamp queries and query output buffer
    #timestampQuerySupport: boolean;
    #timestampQuerySet: GPUQuerySet;
    #timestampBuffer: GPUBuffer;
    #timestampReadbackBuffer: GPUBuffer;

    // Performance stats
    computeActiveVoxelsTime = 0;
    markActiveVoxelsKernelTime = -1;
    computeActiveVoxelsScanTime = 0;
    computeActiveVoxelsCompactTime = 0;

    computeVertexOffsetsTime = 0;
    computeNumVertsKernelTime = -1;
    computeVertexOffsetsScanTime = 0;

    computeVerticesTime = 0;
    computeVerticesKernelTime = -1;

    private constructor(volume: Volume, device: GPUDevice)
    {
        this.#device = device;
        this.#volume = volume;
        this.#timestampQuerySupport = device.features.has("timestamp-query");
    }

    static async create(volume: Volume, device: GPUDevice)
    {
        let mc = new MarchingCubes(volume, device);

        mc.#exclusiveScan = await ExclusiveScan.create(device);
        mc.#streamCompactIds = await StreamCompactIDs.create(device);

        // Upload the case table
        // TODO: Can optimize the size of this buffer to store each case value
        // as an int8, but since WGSL doesn't have an i8 type we then need some
        // bit unpacking in the shader to do that. Will add this after the initial
        // implementation.
        mc.#triCaseTable = device.createBuffer({
            size: MC_CASE_TABLE.byteLength,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });
        new Int32Array(mc.#triCaseTable.getMappedRange()).set(MC_CASE_TABLE);
        mc.#triCaseTable.unmap();

        mc.#volumeInfo = device.createBuffer({
            size: 8 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Uint32Array(mc.#volumeInfo.getMappedRange()).set(volume.dims);
        mc.#volumeInfo.unmap();

        // Allocate the voxel active buffer. This buffer's size is fixed for
        // the entire pipeline, we need to store a flag for each voxel if it's
        // active or not. We'll run a scan on this buffer so it also needs to be
        // aligned to the scan size.
        mc.#voxelActive = device.createBuffer({
            size: mc.#exclusiveScan.getAlignedSize(volume.dualGridNumVoxels) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Compile shaders for our compute kernels
        let markActiveVoxel = await compileShader(device,
            computeVoxelValuesWgsl + "\n" + markActiveVoxelsWgsl, "mark_active_voxel.wgsl");
        let computeNumVerts = await compileShader(device,
            computeVoxelValuesWgsl + "\n" + computeNumVertsWgsl, "compute_num_verts.wgsl");
        let computeVertices = await compileShader(device,
            computeVoxelValuesWgsl + "\n" + computeVerticesWgsl, "compute_vertices.wgsl");

        // Bind group layout for the volume parameters, shared by all pipelines in group 0
        let volumeInfoBGLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: {
                        viewDimension: "3d",
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform"
                    }
                }
            ]
        });

        mc.#volumeInfoBG = device.createBindGroup({
            layout: volumeInfoBGLayout,
            entries: [
                {
                    binding: 0,
                    resource: mc.#volume.texture.createView(),
                },
                {
                    binding: 1,
                    resource: {
                        buffer: mc.#volumeInfo,
                    }
                }
            ]
        });

        let markActiveVoxelBGLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    }
                }
            ]
        });

        mc.#markActiveBG = device.createBindGroup({
            layout: markActiveVoxelBGLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: mc.#voxelActive,
                    }
                }
            ]
        });

        let computeNumVertsBGLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    }
                }
            ]
        });

        let computeVerticesBGLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    }
                }
            ]
        });

        // Push constants BG layout
        let pushConstantsBGLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform",
                        hasDynamicOffset: true
                    }
                }
            ]
        });

        // Create pipelines
        mc.#markActiveVoxelPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout(
                {bindGroupLayouts: [volumeInfoBGLayout, markActiveVoxelBGLayout]}),
            compute: {
                module: markActiveVoxel,
                entryPoint: "main"
            }
        });


        mc.#computeNumVertsPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [
                    volumeInfoBGLayout,
                    computeNumVertsBGLayout,
                    pushConstantsBGLayout
                ]
            }),
            compute: {
                module: computeNumVerts,
                entryPoint: "main"
            }
        });

        mc.#computeVerticesPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [
                    volumeInfoBGLayout,
                    computeVerticesBGLayout,
                    pushConstantsBGLayout
                ]
            }),
            compute: {
                module: computeVertices,
                entryPoint: "main"
            }
        });

        if (mc.#timestampQuerySupport) {
            // We store 6 timestamps, for the start/end of each compute pass we run
            mc.#timestampQuerySet = device.createQuerySet({
                type: "timestamp",
                count: 6
            });
            mc.#timestampBuffer = device.createBuffer({
                size: 6 * 8,
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
            });
            mc.#timestampReadbackBuffer = device.createBuffer({
                size: mc.#timestampBuffer.size,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
        }

        return mc;
    }

    // Computes the surface for the provided isovalue, returning the number of triangles
    // in the surface and the GPUBuffer containing their vertices
    async computeSurface(isovalue: number)
    {
        this.uploadIsovalue(isovalue);

        let start = performance.now();
        let activeVoxels = await this.computeActiveVoxels();
        let end = performance.now();
        this.computeActiveVoxelsTime = end - start;
        if (activeVoxels.count == 0) {
            return new MarchingCubesResult(0, null);
        }

        start = performance.now();
        let vertexOffsets = await this.computeVertexOffsets(activeVoxels);
        end = performance.now();
        this.computeVertexOffsetsTime = end - start;
        if (vertexOffsets.count == 0) {
            return new MarchingCubesResult(0, null);
        }

        start = performance.now();
        let vertices = await this.computeVertices(activeVoxels, vertexOffsets);
        end = performance.now();
        this.computeVerticesTime = end - start;
        activeVoxels.buffer.destroy();
        vertexOffsets.buffer.destroy();

        // Map back the timestamps and get performance statistics
        if (this.#timestampQuerySupport) {
            await this.#timestampReadbackBuffer.mapAsync(GPUMapMode.READ);
            let times = new BigUint64Array(this.#timestampReadbackBuffer.getMappedRange());

            // Timestamps are in nanoseconds
            this.markActiveVoxelsKernelTime = Number(times[1] - times[0]) * 1.0e-6;
            this.computeNumVertsKernelTime = Number(times[3] - times[2]) * 1.0e-6;
            this.computeVerticesKernelTime = Number(times[5] - times[4]) * 1.0e-6;

            this.#timestampReadbackBuffer.unmap();
        }

        return new MarchingCubesResult(vertexOffsets.count, vertices);
    }

    private uploadIsovalue(isovalue: number)
    {
        let uploadIsovalue = this.#device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true
        });
        new Float32Array(uploadIsovalue.getMappedRange()).set([isovalue]);
        uploadIsovalue.unmap();

        var commandEncoder = this.#device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(uploadIsovalue, 0, this.#volumeInfo, 16, 4);
        this.#device.queue.submit([commandEncoder.finish()]);
    }

    private async computeActiveVoxels()
    {
        let dispatchSize = [
            Math.ceil(this.#volume.dualGridDims[0] / 4),
            Math.ceil(this.#volume.dualGridDims[1] / 4),
            Math.ceil(this.#volume.dualGridDims[2] / 2)
        ];

        let activeVoxelOffsets = this.#device.createBuffer({
            size: this.#voxelActive.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE
        });

        var commandEncoder = this.#device.createCommandEncoder();

        if (this.#timestampQuerySupport) {
            commandEncoder.writeTimestamp(this.#timestampQuerySet, 0);
        }
        var pass = commandEncoder.beginComputePass();

        pass.setPipeline(this.#markActiveVoxelPipeline);
        pass.setBindGroup(0, this.#volumeInfoBG);
        pass.setBindGroup(1, this.#markActiveBG);
        pass.dispatchWorkgroups(dispatchSize[0], dispatchSize[1], dispatchSize[2]);

        pass.end();
        if (this.#timestampQuerySupport) {
            commandEncoder.writeTimestamp(this.#timestampQuerySet, 1);
        }

        // Copy the active voxel info to the offsets buffer that we're going to scan,
        // since scan happens in place
        commandEncoder.copyBufferToBuffer(this.#voxelActive, 0, activeVoxelOffsets, 0, activeVoxelOffsets.size);

        this.#device.queue.submit([commandEncoder.finish()]);
        await this.#device.queue.onSubmittedWorkDone();

        let start = performance.now();
        // Scan the active voxel buffer to get offsets to output the active voxel IDs too
        let nActive = await this.#exclusiveScan.scan(activeVoxelOffsets, this.#volume.dualGridNumVoxels);
        let end = performance.now();
        this.computeActiveVoxelsScanTime = end - start;
        if (nActive == 0) {
            return new MarchingCubesResult(0, null);
        }

        let activeVoxelIDs = this.#device.createBuffer({
            size: nActive * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        start = performance.now();
        // Output the compact buffer of active voxel IDs
        await this.#streamCompactIds.compactActiveIDs(this.#voxelActive,
            activeVoxelOffsets,
            activeVoxelIDs,
            this.#volume.dualGridNumVoxels);
        end = performance.now();
        this.computeActiveVoxelsCompactTime = end - start;

        activeVoxelOffsets.destroy();

        return new MarchingCubesResult(nActive, activeVoxelIDs);
    }

    private async computeVertexOffsets(activeVoxels: MarchingCubesResult)
    {
        let vertexOffsets = this.#device.createBuffer({
            size: this.#exclusiveScan.getAlignedSize(activeVoxels.count) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });

        let bindGroup = this.#device.createBindGroup({
            layout: this.#computeNumVertsPipeline.getBindGroupLayout(1),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.#triCaseTable
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: activeVoxels.buffer,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: vertexOffsets
                    }
                }
            ]
        });

        let pushConstantsArg = new Uint32Array([activeVoxels.count]);
        let pushConstants = new PushConstants(
            this.#device, Math.ceil(activeVoxels.count / 32), pushConstantsArg.buffer);

        let pushConstantsBG = this.#device.createBindGroup({
            layout: this.#computeNumVertsPipeline.getBindGroupLayout(2),
            entries: [{
                binding: 0,
                resource: {
                    buffer: pushConstants.pushConstantsBuffer,
                    size: 12,
                }
            }]
        });

        let commandEncoder = this.#device.createCommandEncoder();
        if (this.#timestampQuerySupport) {
            commandEncoder.writeTimestamp(this.#timestampQuerySet, 2);
        }

        let pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.#computeNumVertsPipeline);
        pass.setBindGroup(0, this.#volumeInfoBG);
        pass.setBindGroup(1, bindGroup);
        for (let i = 0; i < pushConstants.numDispatches(); ++i) {
            pass.setBindGroup(2, pushConstantsBG, [i * pushConstants.stride]);
            pass.dispatchWorkgroups(pushConstants.dispatchSize(i), 1, 1);
        }
        pass.end();
        if (this.#timestampQuerySupport) {
            commandEncoder.writeTimestamp(this.#timestampQuerySet, 3);
        }
        this.#device.queue.submit([commandEncoder.finish()]);
        await this.#device.queue.onSubmittedWorkDone();

        let start = performance.now();
        let nVertices = await this.#exclusiveScan.scan(vertexOffsets, activeVoxels.count);
        let end = performance.now();
        this.computeVertexOffsetsScanTime = end - start;

        return new MarchingCubesResult(nVertices, vertexOffsets);
    }

    private async computeVertices(activeVoxels: MarchingCubesResult, vertexOffsets: MarchingCubesResult)
    {
        // We'll output a float4 per vertex
        let vertices = this.#device.createBuffer({
            size: vertexOffsets.count * 4 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC
        });

        let bindGroup = this.#device.createBindGroup({
            layout: this.#computeVerticesPipeline.getBindGroupLayout(1),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.#triCaseTable
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: activeVoxels.buffer,
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: vertexOffsets.buffer
                    }
                },
                {
                    binding: 3,
                    resource: {
                        buffer: vertices
                    }
                }
            ]
        });

        let pushConstantsArg = new Uint32Array([activeVoxels.count]);
        let pushConstants = new PushConstants(
            this.#device, Math.ceil(activeVoxels.count / 32), pushConstantsArg.buffer);

        let pushConstantsBG = this.#device.createBindGroup({
            layout: this.#computeNumVertsPipeline.getBindGroupLayout(2),
            entries: [{
                binding: 0,
                resource: {
                    buffer: pushConstants.pushConstantsBuffer,
                    size: 12,
                }
            }]
        });

        let commandEncoder = this.#device.createCommandEncoder();
        if (this.#timestampQuerySupport) {
            commandEncoder.writeTimestamp(this.#timestampQuerySet, 4);
        }

        let pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.#computeVerticesPipeline);
        pass.setBindGroup(0, this.#volumeInfoBG);
        pass.setBindGroup(1, bindGroup);
        for (let i = 0; i < pushConstants.numDispatches(); ++i) {
            pass.setBindGroup(2, pushConstantsBG, [i * pushConstants.stride]);
            pass.dispatchWorkgroups(pushConstants.dispatchSize(i), 1, 1);
        }
        pass.end();

        if (this.#timestampQuerySupport) {
            commandEncoder.writeTimestamp(this.#timestampQuerySet, 5);

            // This is our last compute pass to compute the surface, so resolve the
            // timestamp queries now as well
            commandEncoder.resolveQuerySet(this.#timestampQuerySet, 0, 6, this.#timestampBuffer, 0);
            commandEncoder.copyBufferToBuffer(this.#timestampBuffer,
                0,
                this.#timestampReadbackBuffer,
                0,
                this.#timestampBuffer.size);
        }

        this.#device.queue.submit([commandEncoder.finish()]);
        await this.#device.queue.onSubmittedWorkDone();

        return vertices;
    }
};
