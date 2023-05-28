// include of compute_voxel_values.wgsl is inserted here

@id(0) override WORKGROUP_SIZE: u32 = 32;

@group(1) @binding(0)
var<storage> case_table: array<i32>;

@group(1) @binding(1)
var<storage, read_write> active_voxel_ids: array<u32>;

@group(1) @binding(2)
var<storage, read_write> voxel_vertex_offsets: array<u32>;

@group(1) @binding(3)
var<storage, read_write> vertices: array<float4>;

struct PushConstants {
    group_id_offset: u32,
    total_workgoups: u32,
    total_elements: u32
};

@group(2) @binding(0)
var<uniform> push_constants: PushConstants;

const EDGE_VERTICES: array<uint2, 12> = array<uint2, 12>(
    uint2(0, 1),
    uint2(1, 2),
    uint2(2, 3),
    uint2(3, 0),
    uint2(4, 5),
    uint2(6, 5),
    uint2(6, 7),
    uint2(7, 4),
    uint2(0, 4),
    uint2(1, 5),
    uint2(2, 6),
    uint2(3, 7)
);

fn lerp_verts(va: int3, vb: int3, fa: f32, fb: f32) -> float3
{
    var t: f32 = 0.0;
    if (abs(fa - fb) >= 0.001) {
        t = (volume_info.isovalue - fa) / (fb - fa);
    }
    return mix(float3(va), float3(vb), t);
}

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

    let voxel_pos = voxel_id_to_pos(voxel_id);
    let vertex_offset = voxel_vertex_offsets[work_item];
    // Now we can finally compute and output the vertices
    for (var i = 0u; i < MC_CASE_ELEMENTS && case_table[case_index * MC_CASE_ELEMENTS + i] != -1; i++)
    {
        let edge = case_table[case_index * MC_CASE_ELEMENTS + i];
        let v0 = EDGE_VERTICES[edge].x;
        let v1 = EDGE_VERTICES[edge].y;

        // Compute the interpolated vertex for this edge within the unit cell
        var v = lerp_verts(INDEX_TO_VERTEX[v0], INDEX_TO_VERTEX[v1], values[v0], values[v1]);

        // Offset the vertex into the global volume grid
        v = v + float3(voxel_pos) + 0.5;
        vertices[vertex_offset + i] = float4(v, 1.0);
    }
}


