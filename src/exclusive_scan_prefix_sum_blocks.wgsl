// See https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
// Compute the prefix sum over the results from each block, this no longer
// writes out the block sums since we're scanning on the block sums
// This shader also applies the carry_in value and writes the carry_ouy

@group(0) @binding(0)
var<storage, read_write> vals: array<u32>;

struct CarryInOut {
    in: u32,
    out: u32,
}

@group(0) @binding(1)
var<storage, read_write> carry: CarryInOut;

// Pipeline can override SCAN_BLOCK_SIZE
@id(0) override SCAN_BLOCK_SIZE: u32 = 512;

var<workgroup> chunk: array<u32, SCAN_BLOCK_SIZE>;

@compute @workgroup_size(SCAN_BLOCK_SIZE / 2)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>)
{
    chunk[2 * local_id.x] = vals[2 * global_id.x];
    chunk[2 * local_id.x + 1] = vals[2 * global_id.x + 1];

    var offs = 1u;
    // Reduce step up tree
    for (var d = SCAN_BLOCK_SIZE >> 1; d > 0; d = d >> 1) {
        workgroupBarrier();
        if (local_id.x < d) {
            let a = offs * (2 * local_id.x + 1) - 1;
            let b = offs * (2 * local_id.x + 2) - 1;
            chunk[b] += chunk[a];
        }
        offs = offs << 1;
    }

    if (local_id.x == 0) {
        carry.out = chunk[SCAN_BLOCK_SIZE - 1] + carry.in;
        chunk[SCAN_BLOCK_SIZE - 1] = 0;
    }

    // Sweep down the tree to finish the scan
    for (var d = 1u; d < SCAN_BLOCK_SIZE; d = d << 1) {
        offs = offs >> 1;
        workgroupBarrier();
        if (local_id.x < d) {
            let a = offs * (2 * local_id.x + 1) - 1;
            let b = offs * (2 * local_id.x + 2) - 1;
            let tmp = chunk[a];
            chunk[a] = chunk[b];
            chunk[b] += tmp;
        }
    }

    workgroupBarrier();
    vals[2 * global_id.x] = chunk[2 * local_id.x] + carry.in;
    vals[2 * global_id.x + 1] = chunk[2 * local_id.x + 1] + carry.in;
}

