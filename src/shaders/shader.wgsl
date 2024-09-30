struct Node {
    positions: vec2<f32>,
    velocities: vec2<f32>,
}

struct Uniforms {
    output_index: u32,
    chunk_size: u32,
    vertex_count: u32,
    dt: f32,
    dx: f32,
    dampening_factor: f32,
    _pad: u32,
    _pad2: u32,
}

struct InternalIndices { 
    idx: u32,
    save_location: u32,
}

@group(0)
@binding(0)
var <uniform> uniforms: Uniforms;

@group(0)
@binding(1)
var<storage, read_write> nodes: array<Node>;

@group(0)
@binding(2)
var<storage, read_write> index_buffer: InternalIndices;

@group(0)
@binding(3)
var<storage, read_write> output_buffer: array<f32>;

@group(0)
@binding(4)
var<storage, read_write> speed_buffer: array<f32>;

@group(0)
@binding(5)
var<storage, read_write> array_ptr_buffer: array<u32>;

@compute
@workgroup_size(64) 
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    let idx = global_id.x;
    var future = (array_ptr_buffer[idx] + 1) % 2;
    var current = (future + 1) % 2;

    fdm_step(idx, future, current);
    //Update the array pointer on each invocation by +1 mod passes per save
    array_ptr_buffer[idx] = (array_ptr_buffer[idx] + 1) % 2;
    
    //Swap variables
    current = future;
    future = (future + 1) % 2;

    if idx != uniforms.output_index { return; }
    if array_ptr_buffer[idx] != 1 { return; }
    let save_loc = index_buffer.save_location;
    output_buffer[save_loc] = nodes[idx].positions[current];
    index_buffer.save_location = (index_buffer.save_location + 1) % uniforms.chunk_size;

}


fn fdm_step(index: u32, future: u32, current: u32) {
    let c = speed_buffer[index];
    let dt = uniforms.dt;
    let dx = uniforms.dx;
    let dampening_factor = uniforms.dampening_factor;
    

    if (index > 0u && index < uniforms.vertex_count - 1u) {
        let a = c * ( nodes[index + 1].positions[current]
            - 2.0 * nodes[index].positions[current] + nodes[index - 1].positions[current]) / (dx * dx);

        nodes[index].velocities[future] = (nodes[index].velocities[current] + (a * dt)) * dampening_factor;
        nodes[index].positions[future] = nodes[index].positions[current] + (nodes[index].velocities[future] * dt);
    } else {
        nodes[index].positions[future] = 0.0;
        nodes[index].positions[future] = 0.0;
    }
}

