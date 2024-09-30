use std::{error::Error, fmt::Display, iter};

use wgpu::util::DeviceExt;

const CHUNK_SIZE: usize = 1024;
const NODE_COUNT: usize = 128;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)]
pub struct Node {
    positions: [f32; 2],
    velocities: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)]
struct FDMUniform {
    output_index: u32,
    chunk_size: u32,
    node_count: u32,
    dt: f32,
    dx: f32,
    dampening_factor: f32,
    _pad: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)]
struct InternalIndices {
    idx: u32,
    save_location: u32,
}

#[derive(Debug)]
pub enum StateError {
    AdapterRequest,
    UnstableParams,
}

impl Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateError::AdapterRequest => {
                let _ = write!(f, "Failed to obtain adapter");
            }
            StateError::UnstableParams => {
                let _ = write!(f, "Unstable parameters");
            }
        }
        Ok(())
    }
}

impl Error for StateError {}

pub struct State {
    device: wgpu::Device,
    queue: wgpu::Queue,
    fdm_uniform: FDMUniform,
    fdm_uniform_buffer: wgpu::Buffer,
    compute_bind_group: wgpu::BindGroup,
    nodes: [Node; NODE_COUNT],
    nodes_buffer: wgpu::Buffer,
    speed_arr: [f32; NODE_COUNT],
    speed_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    buffer_size: usize,
}

impl State {
    pub async fn new() -> Result<State, Box<dyn Error>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(StateError::AdapterRequest)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None,
            )
            .await?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FDM Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader.wgsl").into()),
        });

        //Uniform Buffers
        let fdm_uniform = FDMUniform {
            output_index: 80,
            chunk_size: CHUNK_SIZE as u32,
            node_count: NODE_COUNT as u32,
            dt: 0.000022676,
            dx: 0.0068125,
            dampening_factor: 0.99995,
            _pad: 0,
            _pad2: 0,
        };
        let fdm_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FDM Uniform Buffer"),
            contents: bytemuck::cast_slice(&[fdm_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let speed_arr: [f32; NODE_COUNT] = [90000.0; NODE_COUNT];

        //Node Buffers
        let nodes = [Node {
            positions: [0.0, 0.0],
            velocities: [0.0, 0.0],
        }; NODE_COUNT];

        let nodes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Node Buffer 0"),
            contents: bytemuck::cast_slice(&nodes),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let speed_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("speed buffer"),
            contents: bytemuck::cast_slice(&speed_arr),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // Buffer whose contents will eventually be copied into the CPU staging buffer
        // We store this in GPU memory for CHUNK_SIZE passes, to minimize the calls to the
        // (expensive) buffer copy operation.
        let output = [0.0f32; CHUNK_SIZE];
        let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: bytemuck::cast_slice(&[output]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        //Staging
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of_val(&output) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        //Hacky, but this allows the shader to update the ringbuffer index by itself
        let internal_indices = InternalIndices {
            idx: 2,
            save_location: 0,
        };
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: bytemuck::cast_slice(&[internal_indices]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let array_pointers = [0u32; CHUNK_SIZE];
        let array_ptr_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Array pointer buffer"),
            contents: bytemuck::cast_slice(&[array_pointers]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Compute Bind Group Layout"),
            });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fdm_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nodes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: speed_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: array_ptr_buffer.as_entire_binding(),
                },
            ],
            label: Some("Compute Bind Group Layout"),
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        //Uses the compute node buffer as storage buffer
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "compute_main",
            compilation_options: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            fdm_uniform,
            fdm_uniform_buffer,
            compute_bind_group,
            nodes,
            nodes_buffer,
            speed_arr,
            speed_buffer,
            output_buffer,
            staging_buffer,
            compute_pipeline,
            buffer_size: std::mem::size_of_val(&output),
        })
    }

    pub fn compute(&mut self) -> Result<Vec<f32>, Box<dyn Error>> {
        for _ in 0..2 {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compute Encoder"),
                });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
                compute_pass.set_pipeline(&self.compute_pipeline);
                for _ in 0..CHUNK_SIZE {
                    // Compute once
                    // The shader figures out whether this is a pass whose output should be saved
                    // or a pass that can be discarded on its own.
                    compute_pass.dispatch_workgroups(128, 1, 1);
                }
            }
            self.queue.submit(iter::once(encoder.finish()));
        }

        // Begin memory transfer to CPU
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.buffer_size as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..);
        let (tx, rx) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        // Awaits until `buffer_future` can be read from
        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            self.staging_buffer.unmap(); // Unmaps buffer from memory

            return Ok(result);
        }
        Ok(Vec::new())
    }

    pub fn set_wavespeeds(&mut self, speeds: &[f32; NODE_COUNT]) -> Result<(), Box<dyn Error>> {
        let fastest = speeds
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        if fastest * self.fdm_uniform.dt.powi(2) / self.fdm_uniform.dx.powi(2) > 1.0 {
            return Err(Box::new(StateError::UnstableParams));
        }

        self.speed_arr.copy_from_slice(speeds);
        let mut buffer = encase::StorageBuffer::new(Vec::new());
        buffer.write(&self.speed_arr)?;
        self.queue
            .write_buffer(&self.speed_buffer, 0, &buffer.into_inner());
        Ok(())
    }

    pub fn set_wavespeeds_from_fn<F>(&mut self, f: F) -> Result<(), Box<dyn Error>>
    where
        F: Fn((usize, &mut f32)),
    {
        let speeds_backup = self.speed_arr;
        self.speed_arr
            .iter_mut()
            .enumerate()
            .for_each(|(index, value)| f((index, value)));
        let fastest = self
            .speed_arr
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        if fastest * self.fdm_uniform.dt.powi(2) / self.fdm_uniform.dx.powi(2) > 1.0 {
            self.speed_arr = speeds_backup;
            return Err(Box::new(StateError::UnstableParams));
        }
        let mut buffer = encase::StorageBuffer::new(Vec::new());
        buffer.write(&self.speed_arr)?;
        self.queue
            .write_buffer(&self.speed_buffer, 0, &buffer.into_inner());

        Ok(())
    }

    pub fn set_parameters(
        &mut self,
        dx: Option<f32>,
        dt: Option<f32>,
        dampening_factor: Option<f32>,
        output_index: Option<u32>,
    ) -> Result<(), Box<dyn Error>> {
        let dx = dx.unwrap_or(self.fdm_uniform.dx);
        let dt = dt.unwrap_or(self.fdm_uniform.dt);
        let dampening_factor = dampening_factor.unwrap_or(self.fdm_uniform.dampening_factor);
        let output_index = output_index.unwrap_or(self.fdm_uniform.output_index);

        let fastest = self
            .speed_arr
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        if fastest * self.fdm_uniform.dt.powi(2) / self.fdm_uniform.dx.powi(2) > 1.0 {
            return Err(Box::new(StateError::UnstableParams));
        }

        self.fdm_uniform = FDMUniform {
            output_index,
            dx,
            dt,
            dampening_factor,
            ..self.fdm_uniform
        };

        let mut buffer = encase::UniformBuffer::new(Vec::new());
        buffer.write(&self.fdm_uniform)?;
        self.queue
            .write_buffer(&self.fdm_uniform_buffer, 0, &buffer.into_inner());

        Ok(())
    }

    pub fn set_displacements(
        &mut self,
        displacements: [f32; NODE_COUNT],
    ) -> Result<(), Box<dyn Error>> {
        //Hacky but we just set both Node arrays to the displacement?
        let new_nodes = displacements
            .iter()
            .map(|u| Node {
                positions: [*u, *u],
                velocities: [0.0, 0.0],
            })
            .collect::<Vec<Node>>();

        self.nodes.copy_from_slice(&new_nodes);

        let mut buffer = encase::StorageBuffer::new(Vec::new());
        buffer.write(&self.nodes)?;
        self.queue
            .write_buffer(&self.nodes_buffer, 0, &buffer.into_inner());
        Ok(())
    }

    pub fn set_displacements_from_fn<F>(&mut self, f: F) -> Result<(), Box<dyn Error>>
    where
        F: Fn((usize, &mut Node)),
    {
        self.nodes
            .iter_mut()
            .enumerate()
            .for_each(|(index, node)| f((index, node)));
        let mut buffer = encase::StorageBuffer::new(Vec::new());
        buffer.write(&self.nodes)?;
        self.queue
            .write_buffer(&self.nodes_buffer, 0, &buffer.into_inner());
        Ok(())
    }
}
