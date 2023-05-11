import {ExclusiveScan} from "./exclusive_scan";
import {MC_CASE_TABLE} from "./mc_case_table";
import {StreamCompactIDs} from "./stream_compact_ids";
import {Volume} from "./volume";

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

        // TODO: Allocate voxel count-sized buffers that
        // - is active (needs to be scan size aligned)

        return mc;
    }
};
