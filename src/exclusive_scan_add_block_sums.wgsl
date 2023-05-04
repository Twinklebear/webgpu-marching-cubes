@group(0) @binding(0)
var<storage, read_write> vals: array<u32>;

@group(0) @binding(1)
var<storage, read_write> block_sums: array<u32>;

// Pipeline can override SCAN_BLOCK_SIZE
@id(0) override SCAN_BLOCK_SIZE: u32 = 512;

@compute @workgroup_size(SCAN_BLOCK_SIZE / 2)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>)
{
    let prev_sum = block_sums[group_id.x];
    vals[2 * global_id.x] += prev_sum;
    vals[2 * global_id.x + 1] += prev_sum;
}

