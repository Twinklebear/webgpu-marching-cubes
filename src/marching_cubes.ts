import {ExclusiveScan} from "./exclusive_scan";
import {MC_CASE_TABLE} from "./mc_case_table";
import {StreamCompactIDs} from "./stream_compact_ids";
import {Volume} from "./volume";
import {compileShader} from "./util";

import computeVoxelValuesWgsl from "./compute_voxel_values.wgsl";
import markActiveVoxelsWgsl from "./mark_active_voxel.wgsl";
import computeNumVertsWgsl from "./compute_num_verts.wgsl";
import computeVerticesWgsl from "./compute_vertices.wgsl";

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

    private constructor(volume: Volume, device: GPUDevice)
    {
        this.#device = device;
        this.#volume = volume;
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

        return mc;
    }

    // Computes the surface for the provided isovalue, returning the number of triangles
    // in the surface and the GPUBuffer containing their vertices
    async computeSurface(isovalue: number)
    {
        this.uploadIsovalue(isovalue);

        await this.computeActiveVoxels();


        return [0, null];
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

        // Readback info about active voxels for validation
        let debugBuf = this.#device.createBuffer({
            size: this.#volume.dualGridNumVoxels * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        var commandEncoder = this.#device.createCommandEncoder();
        var pass = commandEncoder.beginComputePass();

        pass.setPipeline(this.#markActiveVoxelPipeline);
        pass.setBindGroup(0, this.#volumeInfoBG);
        pass.setBindGroup(1, this.#markActiveBG);
        pass.dispatchWorkgroups(dispatchSize[0], dispatchSize[1], dispatchSize[2]);

        pass.end();

        // Readback info about active voxels for validation
        commandEncoder.copyBufferToBuffer(this.#voxelActive, 0, debugBuf, 0, debugBuf.size);

        this.#device.queue.submit([commandEncoder.finish()]);
        await this.#device.queue.onSubmittedWorkDone();

        await debugBuf.mapAsync(GPUMapMode.READ);
        let activeVoxels = new Uint32Array(debugBuf.getMappedRange());
        let nActive = 0;
        for (let i = 0; i < activeVoxels.length; ++i) {
            if (activeVoxels[i]) {
                ++nActive;
            }
        }
        console.log(`# of active voxels = ${nActive} of ${this.#volume.dualGridNumVoxels}`);

        debugBuf.unmap();
    }

    private async computeVertexOffsets()
    {
    }

    private async computeVertices()
    {
    }
};
