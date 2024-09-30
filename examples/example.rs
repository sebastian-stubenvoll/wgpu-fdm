// An example to demonstrate various functionalities of this crate
fn main() {
    // Obtain a new FDM-Simulation state
    let state = wgpu_fdm::State::new();
    let mut state = pollster::block_on(state).unwrap();

    // Set various simulation parameters.
    // This method calculates the CFL number and returns and error
    // if the simulation parameters are unstable.
    state
        .set_parameters(
            Some(0.00234375),  // Node distance
            Some(0.000011338), // Time steps
            Some(0.9999),      // Dampening factor
            Some(48),          // Index of the node whose displacement is saved to output
        )
        .expect("Failed to set params");

    // Set the wavespeed associated with each node.
    // This method calculates the CFL number and returns and error
    // if the simulation parameters are unstable.
    let speeds = [10000.0; 128];
    state
        .set_wavespeeds(&speeds)
        .expect("Failed to set wavespeeds");

    // Set Nodes displacements to "pluck" the string.
    // E.g. as a parabola:
    let mut displacements = [0.0; 128];
    let mid = (127 - 1) as f32 / 2.0;
    for (index, displacement) in displacements.iter_mut().enumerate() {
        let index_f32 = index as f32;
        *displacement = (1.0 - ((index_f32 - mid) / mid).powi(2)) * 0.3;
    }
    state.set_displacements(displacements).unwrap();

    // One compute pass returns CHUNK_SIZE (1024) samples.
    // We initialise and empty Vec that stores one Vec<f32> per compute pass.
    let mut result = Vec::new();

    // Let's run the simulation 43 times, resulting in just under one second
    // of audio at a 44.1 kHz sampling rate (1024 * 43 / 44100 ≈ 1)
    for _ in 0..43 {
        if let Ok(v) = state.compute() {
            result.push(v);
        }
    }

    // "Pluck" the string again by settin g displacements.
    // Pass a closure this time to generate a triangle form.
    state
        .set_displacements_from_fn(|(index, displacement)| {
            if index < 512 {
                *displacement = index as f32 / 512.0;
            } else {
                *displacement = (1024.0 - index as f32) / 512.0;
            }
        })
        .unwrap();

    // Run another 43 compute passes.
    for _ in 0..43 {
        if let Ok(v) = state.compute() {
            result.push(v);
        }
    }

    // Output the result to console.
    for l in result {
        println!("{:?}", l);
    }
}
