// This type definition is just to make typing a bit easier
alias float4 = vec4<f32>;

struct VertexInput {
    @location(0) position: float4,
    @location(1) color: float4,
};

struct VertexOutput {
    // This is the equivalent of gl_Position in GLSL
    @builtin(position) position: float4,
    @location(0) color: float4,
};

@vertex
fn vertex_main(vert: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = vert.color;
    out.position = vert.position;
    return out;
};

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) float4 {
    return float4(in.color);
}
