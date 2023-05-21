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

    #exclusive_scan: ExclusiveScan;

    #stream_compact_ids: StreamCompactIDs;

    #tri_case_table: GPUBuffer;

    #volume_info: GPUBuffer;

    #voxel_active: GPUBuffer;

    // Compute pipelines for each stage of the compute 

    private constructor(volume: Volume, device: GPUDevice)
    {
        this.#device = device;
        this.#volume = volume;
    }

    static async create(volume: Volume, device: GPUDevice)
    {
        let mc = new MarchingCubes(volume, device);

        mc.#exclusive_scan = await ExclusiveScan.create(device);
        mc.#stream_compact_ids = await StreamCompactIDs.create(device);

        // Upload the case table
        // TODO: Can optimize the size of this buffer to store each case value
        // as an int8, but since WGSL doesn't have an i8 type we then need some
        // bit unpacking in the shader to do that. Will add this after the initial
        // implementation.
        mc.#tri_case_table = device.createBuffer({
            size: MC_CASE_TABLE.byteLength,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });
        new Int32Array(mc.#tri_case_table.getMappedRange()).set(MC_CASE_TABLE);
        mc.#tri_case_table.unmap();

        mc.#volume_info = device.createBuffer({
            size: 6 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Uint32Array(mc.#volume_info.getMappedRange()).set(volume.dims);
        mc.#volume_info.unmap();

        // Allocate the voxel active buffer. This buffer's size is fixed for
        // the entire pipeline, we need to store a flag for each voxel if it's
        // active or not. We'll run a scan on this buffer so it also needs to be
        // aligned to the scan size.
        mc.#voxel_active = device.createBuffer({
            size: mc.#exclusive_scan.getAlignedSize(volume.dualGridNumVoxels) * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        // Compile shaders for our compute kernels
        let markActiveVoxel = await compileShader(device,
            computeVoxelValuesWgsl + "\n" + markActiveVoxelsWgsl, "mark_active_voxel.wgsl");
        let computeNumVerts = await compileShader(device,
            computeVoxelValuesWgsl + "\n" + computeNumVertsWgsl, "compute_num_verts.wgsl");
        let computeVertices = await compileShader(device,
            computeVoxelValuesWgsl + "\n" + computeVerticesWgsl, "compute_vertices.wgsl");

        return mc;
    }

    // Computes the surface for the provided isovalue, returning the number of triangles
    // in the surface and the GPUBuffer containing their vertices
    async computeSurface(isovalue: number)
    {
        this.uploadIsovalue(isovalue);


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
        commandEncoder.copyBufferToBuffer(uploadIsovalue, 0, this.#volume_info, 16, 4);
        this.#device.queue.submit([commandEncoder.finish()]);
    }

    private async computeActiveVoxels()
    {
    }

    private async computeVertexOffsets()
    {
    }

    private async computeVertices()
    {
    }
};
