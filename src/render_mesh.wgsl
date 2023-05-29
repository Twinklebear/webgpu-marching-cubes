// This type definition is just to make typing a bit easier
alias float3 = vec3<f32>;
alias float4 = vec4<f32>;

alias uint4 = vec4<u32>;

struct VertexInput {
    @location(0) position: float4,
};

struct VertexOutput {
    // This is the equivalent of gl_Position in GLSL
    @builtin(position) position: float4,
    @location(0) world_pos: float3,
};

struct ViewParams {
    proj_view: mat4x4<f32>,
    volume_dims: uint4,
};

@group(0) @binding(0)
var<uniform> params: ViewParams;

@vertex
fn vertex_main(vert: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var pos = vert.position.xyz - float3(params.volume_dims.xyz) / 2.0;
    out.position = params.proj_view * float4(pos, 1.0);
    out.world_pos = pos;
    return out;
};

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) float4 {
    let dx = dpdx(in.world_pos);
    let dy = dpdy(in.world_pos);
    let n = normalize(cross(dx, dy));
    return float4((n + 1.0) * 0.5, 1.0);
}
