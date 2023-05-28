// include of compute_voxel_values.wgsl is inserted here

@group(1) @binding(0)
var<storage, read_write> voxel_active: array<u32>;

@compute @workgroup_size(4, 4, 2)
fn main(@builtin(global_invocation_id) global_id: uint3)
{
    // We might have some workgroups run for voxels out of bounds due to the
    // padding to align to the workgroup size. We also only compute for voxels
    // on the dual grid, which has dimensions of volume_dims - 1
    if (any(global_id >= volume_info.dims.xyz - uint3(1))) {
        return;
    }

    var values: array<f32, 8>;
    compute_voxel_values(global_id, &values);
    // Compute the case this falls into to see if this voxel has vertices
    var case_index = 0u;
    for (var i = 0u; i < 8u; i++) {
        if (values[i] <= volume_info.isovalue) {
            case_index |= 1u << i;
        }
    }
    let voxel_idx = global_id.x +
        (volume_info.dims.x - 1) * (global_id.y + (volume_info.dims.z - 1) * global_id.z);
    voxel_active[voxel_idx] = select(0u, 1u, case_index != 0 && case_index != MC_NUM_CASES - 1);
}

