use std::time::Instant;

#[cfg(feature = "wgpu")]
use burn::tensor::DeviceKind;
use burn::{
    nn::loss::{MseLoss, Reduction},
    optim::{GradientsParams, Optimizer, Sgd, SgdConfig, adaptor::OptimizerAdaptor},
    prelude::*,
    tensor::{Bytes, TensorData},
};
use tynx_bench_protocol::training::{
    CachePolicy, ParameterState, SyncPolicy, TrainingCase, TrainingMode, TrainingRun,
    TrainingWorkload, cache_policy, deterministic_batch, load_training_cases, measure_training,
    model_sha256, parameter_sha256, print_training_reports, training_mlp_model,
};
use tynx_bench_protocol::{BenchResult, Threading, require_release};

#[allow(dead_code)]
mod training_mlp {
    include!(concat!(env!("OUT_DIR"), "/model/training_mlp.rs"));

    use burn::optim::GradientsParams;

    impl Model {
        pub fn trainable_from_bytes(bytes: Bytes, device: &Device) -> Self {
            let mut model = Self::from_bytes(bytes, device);
            model.constant1 = model.constant1.set_require_grad(true);
            model.constant2 = model.constant2.set_require_grad(true);
            model.constant3 = model.constant3.set_require_grad(true);
            model.constant4 = model.constant4.set_require_grad(true);
            model
        }

        pub fn gradient_names(&self, gradients: &GradientsParams) -> Vec<String> {
            let mut names = Vec::new();
            if gradients.get::<2>(self.constant1.id).is_some() {
                names.push("gemm1.weight".to_string());
            }
            if gradients.get::<1>(self.constant2.id).is_some() {
                names.push("gemm1.bias".to_string());
            }
            if gradients.get::<2>(self.constant3.id).is_some() {
                names.push("gemm2.weight".to_string());
            }
            if gradients.get::<1>(self.constant4.id).is_some() {
                names.push("gemm2.bias".to_string());
            }
            names.sort();
            names
        }

        pub fn parameter_values(&self) -> Vec<f32> {
            let mut values = Vec::new();
            values.extend(self.constant1.val().into_data().iter::<f32>());
            values.extend(self.constant2.val().into_data().iter::<f32>());
            values.extend(self.constant3.val().into_data().iter::<f32>());
            values.extend(self.constant4.val().into_data().iter::<f32>());
            values
        }
    }
}

type OptimizerType = OptimizerAdaptor<Sgd, training_mlp::Model>;

struct Workload {
    model_bytes: Vec<u8>,
    device: Device,
    model: Option<training_mlp::Model>,
    batches: Vec<(Tensor<2>, Tensor<2>)>,
    optimizer: OptimizerType,
    last_loss: Option<Tensor<1>>,
    last_gradients: Vec<String>,
    last_updated: usize,
}

impl Workload {
    fn new(case: &TrainingCase, device: Device) -> BenchResult<(Self, f64)> {
        let model_bytes =
            include_bytes!(concat!(env!("OUT_DIR"), "/model/training_mlp.bpk")).to_vec();
        let started = Instant::now();
        let model = training_mlp::Model::trainable_from_bytes(
            Bytes::from_bytes_vec(model_bytes.clone()),
            &device,
        );
        let batches = (0..case.dataset_batches)
            .map(|index| {
                let (inputs, targets) = deterministic_batch(case, index);
                (
                    Tensor::<2>::from_data(
                        TensorData::new(
                            inputs,
                            [
                                case.batch_size,
                                tynx_bench_protocol::training::INPUT_FEATURES,
                            ],
                        ),
                        &device,
                    ),
                    Tensor::<2>::from_data(
                        TensorData::new(
                            targets,
                            [
                                case.batch_size,
                                tynx_bench_protocol::training::OUTPUT_FEATURES,
                            ],
                        ),
                        &device,
                    ),
                )
            })
            .collect();
        device.sync()?;
        let prepare_ms = elapsed_ms(started);
        Ok((
            Self {
                model_bytes,
                device,
                model: Some(model),
                batches,
                optimizer: SgdConfig::new().init(),
                last_loss: None,
                last_gradients: Vec::new(),
                last_updated: 0,
            },
            prepare_ms,
        ))
    }

    fn model(&self) -> BenchResult<&training_mlp::Model> {
        self.model
            .as_ref()
            .ok_or_else(|| "Burn AOT training model was unavailable".into())
    }
}

impl TrainingWorkload for Workload {
    fn reset(&mut self) -> BenchResult<()> {
        self.model = Some(training_mlp::Model::trainable_from_bytes(
            Bytes::from_bytes_vec(self.model_bytes.clone()),
            &self.device,
        ));
        self.optimizer = SgdConfig::new().init();
        self.last_loss = None;
        self.last_gradients.clear();
        self.last_updated = 0;
        Ok(())
    }

    fn step(&mut self, batch_index: usize, mode: TrainingMode) -> BenchResult<()> {
        let (input, target) = &self.batches[batch_index % self.batches.len()];
        let prediction = self.model()?.forward(input.clone());
        let loss = MseLoss::new().forward(prediction, target.clone(), Reduction::Mean);
        self.last_gradients.clear();
        self.last_updated = 0;
        if matches!(
            mode,
            TrainingMode::ForwardBackward | TrainingMode::TrainStep
        ) {
            let gradients = GradientsParams::from_grads(loss.clone().backward(), self.model()?);
            self.last_gradients = self.model()?.gradient_names(&gradients);
            if matches!(mode, TrainingMode::TrainStep) {
                let model = self
                    .model
                    .take()
                    .ok_or("Burn AOT training model was unavailable")?;
                self.model = Some(self.optimizer.step(
                    tynx_bench_protocol::training::LEARNING_RATE,
                    model,
                    gradients,
                ));
                self.last_updated = self.last_gradients.len();
            }
        }
        self.last_loss = Some(loss);
        Ok(())
    }

    fn sync(&self) -> BenchResult<()> {
        Ok(self.device.sync()?)
    }

    fn state(&self) -> BenchResult<ParameterState> {
        let values = self.model()?.parameter_values();
        let mut parameter_sum = 0.0;
        let mut parameter_squared_sum = 0.0;
        let mut finite = true;
        for &value in &values {
            let value = f64::from(value);
            finite &= value.is_finite();
            parameter_sum += value;
            parameter_squared_sum += value * value;
        }
        let loss = self
            .last_loss
            .clone()
            .and_then(|loss| loss.into_data().iter::<f32>().next())
            .map(f64::from);
        Ok(ParameterState {
            trainable: vec![
                "gemm1.bias".to_string(),
                "gemm1.weight".to_string(),
                "gemm2.bias".to_string(),
                "gemm2.weight".to_string(),
            ],
            frozen: Vec::new(),
            gradients: self.last_gradients.clone(),
            updated_parameters: self.last_updated,
            loss,
            parameter_sum,
            parameter_l2: parameter_squared_sum.sqrt(),
            parameter_count: values.len(),
            parameter_sha256: parameter_sha256(&values),
            finite,
        })
    }
}

fn main() -> BenchResult<()> {
    require_release()?;
    let mode = TrainingMode::from_env()?;
    let sync_policy = SyncPolicy::from_env()?;
    let threading = cpu_threading()?;
    let (backend, device, device_name) = device();
    let onnx = training_mlp_model()?;
    let sha256 = model_sha256(&onnx);
    let cache = cache_policy(backend);
    let mut reports = Vec::new();
    for case in load_training_cases()? {
        let (mut workload, prepare_ms) = Workload::new(&case, device.clone())?;
        reports.push(measure_training(
            &case,
            mode,
            sync_policy,
            TrainingRun {
                engine: "burn-onnx-aot-train",
                backend,
                device: device_name.clone(),
                parse_ms: 0.0,
                prepare_ms,
                model_sha256: sha256.clone(),
                threading: threading.clone(),
                cache: CachePolicy {
                    process: cache.process.clone(),
                    autotune: cache.autotune.clone(),
                },
            },
            &mut workload,
        )?);
    }
    print_training_reports(&reports)
}

#[cfg(feature = "wgpu")]
fn cpu_threading() -> BenchResult<Option<Threading>> {
    Ok(None)
}

#[cfg(all(not(feature = "wgpu"), feature = "multithread"))]
fn cpu_threading() -> BenchResult<Option<Threading>> {
    let request = tynx_bench_protocol::thread_request()?;
    if let tynx_bench_protocol::ThreadRequest::Fixed(threads) = request {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }
    Ok(Some(
        request.report("rayon", Some(rayon::current_num_threads())),
    ))
}

#[cfg(all(not(feature = "wgpu"), not(feature = "multithread")))]
fn cpu_threading() -> BenchResult<Option<Threading>> {
    let request = tynx_bench_protocol::thread_request()?;
    if request.fixed().is_some_and(|threads| threads != 1) {
        return Err("multi-thread CPU benchmarks require the 'multithread' feature".into());
    }
    Ok(Some(request.report("serial", Some(1))))
}

#[cfg(feature = "wgpu")]
fn device() -> (&'static str, Device, Option<String>) {
    (
        "wgpu",
        Device::autodiff(Device::webgpu(DeviceKind::DefaultDevice)),
        tynx_bench_protocol::wgpu_device_name(),
    )
}

#[cfg(not(feature = "wgpu"))]
fn device() -> (&'static str, Device, Option<String>) {
    ("flex", Device::autodiff(Device::flex()), None)
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}
