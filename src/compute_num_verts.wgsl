// include of compute_voxel_values.wgsl is inserted here

@id(0) override WORKGROUP_SIZE: u32 = 32;

@group(1) @binding(0)
var<storage> case_table: array<i32>;

@group(1) @binding(1)
var<storage, read_write> active_voxel_ids: array<u32>;

@group(1) @binding(2)
var<storage, read_write> voxel_num_verts: array<u32>;

struct PushConstants {
    group_id_offset: u32,
    total_workgoups: u32,
    total_elements: u32
};

@group(2) @binding(0)
var<uniform> push_constants: PushConstants;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: uint3)
{
    // Skip out of bounds threads
    let work_item = global_id.x + push_constants.group_id_offset * WORKGROUP_SIZE;
    if (work_item >= push_constants.total_elements) {
        return;
    }

    let voxel_id = active_voxel_ids[work_item];
    var values: array<f32, 8>;
    compute_voxel_values(voxel_id_to_pos(voxel_id), &values);

    var case_index = 0u;
    for (var i = 0u; i < 8u; i++) {
        if (values[i] <= volume_info.isovalue) {
            case_index |= 1u << i;
        }
    }

    // There are 16 entries per-case, terminated by a -1 when the vertex
    // entries end for the given case
    var num_verts = 0u;
    for (var i = 0u; i < MC_CASE_ELEMENTS && case_table[case_index * MC_CASE_ELEMENTS + i] != -1; i++) 
    {
        num_verts++;
    }
    voxel_num_verts[work_item] = num_verts;
}

