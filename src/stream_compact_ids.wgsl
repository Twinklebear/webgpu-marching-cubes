@group(0) @binding(0)
var<storage, read_write> item_active: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output_offset: array<u32>;

@group(0) @binding(2)
var<storage, read_write> output: array<u32>;

struct PushConstants {
    group_id_offset: u32,
    total_workgoups: u32,
    total_elements: u32
};

@group(1) @binding(0)
var<uniform> push_constants: PushConstants;

@id(0) override WORKGROUP_SIZE: u32 = 32;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let item_id = global_id.x + push_constants.group_id_offset * WORKGROUP_SIZE;
    // Handle out of bounds threads
    if (item_id >= push_constants.total_elements) {
        return;
    }
    // We compact down the IDs of the active elements in the buffer.
    // Active elements have non-zero values
    if (item_active[global_id.x] != 0) {
        output[output_offset[global_id.x]] = item_id;
    }
}

