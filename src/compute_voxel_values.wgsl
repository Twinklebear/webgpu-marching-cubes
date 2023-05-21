alias float2 = vec2<f32>;
alias float3 = vec3<f32>;
alias float4 = vec4<f32>;
alias uint2 = vec2<u32>;
alias uint3 = vec3<u32>;
alias uint4 = vec4<u32>;
alias int2 = vec2<i32>;
alias int3 = vec3<i32>;
alias int4 = vec4<i32>;

const MC_NUM_CASES: u32 = 256;
const MC_CASE_ELEMENTS: u32 = 16;

struct VolumeInfo {
    dims: uint4,
    isovalue: f32,
};

@group(0) @binding(0)
var volume: texture_3d<f32>;

@group(0) @binding(1)
var<uniform> volume_info: VolumeInfo;

const INDEX_TO_VERTEX: array<int3, 8> = array<int3, 8>(
    int3(0, 0, 0),
    int3(1, 0, 0),
    int3(1, 1, 0),
    int3(0, 1, 0),
    int3(0, 0, 1),
    int3(1, 0, 1),
    int3(1, 1, 1),
    int3(0, 1, 1)
);


fn voxel_id_to_pos(id: u32) -> uint3
{
    return uint3(id % (volume_info.dims[0] - 1),
            (id / (volume_info.dims[0] - 1)) % (volume_info.dims[1] - 1),
            id / ((volume_info.dims[0] - 1) * (volume_info.dims[1] - 1)));
}

fn compute_voxel_values(voxel: uint3, values: ptr<function, array<f32, 8>>)
{
    for (var i = 0; i < 8; i++) {
        let p = voxel + uint3(INDEX_TO_VERTEX[i]);
        (*values)[i] = textureLoad(volume, p, 0).x;
    }
}

