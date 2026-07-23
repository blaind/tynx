use std::time::Instant;

#[cfg(feature = "wgpu")]
use burn::tensor::DeviceKind;
use burn::tensor::{Device, TensorData};
use tynx_bench_protocol::training::{
    CachePolicy, ParameterState, SyncPolicy, TrainingCase, TrainingMode, TrainingRun,
    TrainingWorkload, cache_policy, deterministic_batch, load_training_cases, measure_training,
    model_sha256, parameter_sha256, print_training_reports, training_mlp_model,
};
use tynx_bench_protocol::{BenchResult, Threading, require_release};
use tynx_core::{DynTensor, Env, Session, Value};
use tynx_train::{
    ImportedModel, InitializerNameOverrides, Sgd, TrainabilityOverrides, TrainabilityReport,
    backward, loss::mse,
};

struct Workload {
    bytes: Vec<u8>,
    device: Device,
    model: ImportedModel,
    output_name: String,
    batches: Vec<(DynTensor, DynTensor)>,
    optimizer: Sgd,
    last_loss: Option<DynTensor>,
    last_updated: usize,
}

impl Workload {
    fn new(bytes: Vec<u8>, case: &TrainingCase, device: Device) -> BenchResult<(Self, f64, f64)> {
        let started = Instant::now();
        let session = Session::from_bytes_with(&bytes, false)?;
        let parse_ms = elapsed_ms(started);

        let started = Instant::now();
        let model = imported_model(session, device.clone())?;
        let output_name = model.session().outputs()[0].name.clone();
        let batches = (0..case.dataset_batches)
            .map(|index| {
                let (inputs, targets) = deterministic_batch(case, index);
                Ok((
                    tensor(
                        inputs,
                        &[
                            case.batch_size,
                            tynx_bench_protocol::training::INPUT_FEATURES,
                        ],
                        &device,
                    )?,
                    tensor(
                        targets,
                        &[
                            case.batch_size,
                            tynx_bench_protocol::training::OUTPUT_FEATURES,
                        ],
                        &device,
                    )?,
                ))
            })
            .collect::<BenchResult<Vec<_>>>()?;
        device.sync()?;
        let prepare_ms = elapsed_ms(started);

        Ok((
            Self {
                bytes,
                device,
                model,
                output_name,
                batches,
                optimizer: Sgd::new(tynx_bench_protocol::training::LEARNING_RATE)?,
                last_loss: None,
                last_updated: 0,
            },
            parse_ms,
            prepare_ms,
        ))
    }

    fn rebuild_model(&self) -> BenchResult<ImportedModel> {
        imported_model(
            Session::from_bytes_with(&self.bytes, false)?,
            self.device.clone(),
        )
    }
}

impl TrainingWorkload for Workload {
    fn reset(&mut self) -> BenchResult<()> {
        self.model = self.rebuild_model()?;
        self.optimizer = Sgd::new(tynx_bench_protocol::training::LEARNING_RATE)?;
        self.last_loss = None;
        self.last_updated = 0;
        Ok(())
    }

    fn step(&mut self, batch_index: usize, mode: TrainingMode) -> BenchResult<()> {
        let (input, target) = &self.batches[batch_index % self.batches.len()];
        self.model.parameters().zero_grad();
        let prediction = self
            .model
            .run(Env::from([("x".to_string(), Value::Tensor(input.clone()))]))?
            .remove(&self.output_name)
            .ok_or_else(|| format!("training output '{}' was missing", self.output_name))?
            .into_tensor()?;
        let loss = mse(prediction, target.clone())?;
        self.last_updated = 0;
        if matches!(
            mode,
            TrainingMode::ForwardBackward | TrainingMode::TrainStep
        ) {
            let result = backward(&loss, self.model.parameters())?;
            if matches!(mode, TrainingMode::TrainStep) {
                self.last_updated = self.optimizer.step(self.model.parameters())?;
            }
            drop(result);
        }
        self.last_loss = Some(loss);
        Ok(())
    }

    fn sync(&self) -> BenchResult<()> {
        Ok(self.device.sync()?)
    }

    fn state(&self) -> BenchResult<ParameterState> {
        let mut trainable = Vec::new();
        let mut frozen = Vec::new();
        let mut gradients = Vec::new();
        let mut parameter_sum = 0.0;
        let mut parameter_squared_sum = 0.0;
        let mut parameter_count = 0;
        let mut finite = true;
        let mut parameter_values = Vec::new();
        for (name, slot) in self.model.parameters().named() {
            if slot.contract().trainable() {
                trainable.push(name.to_string());
            } else {
                frozen.push(name.to_string());
            }
            if slot.grad().is_some() {
                gradients.push(name.to_string());
            }
            for value in slot.value().into_data().iter::<f32>() {
                parameter_values.push(value);
                let value = f64::from(value);
                finite &= value.is_finite();
                parameter_sum += value;
                parameter_squared_sum += value * value;
                parameter_count += 1;
            }
        }
        trainable.sort();
        frozen.sort();
        gradients.sort();
        let loss = self
            .last_loss
            .clone()
            .map(|loss| -> BenchResult<f64> {
                let value = loss
                    .into_data()
                    .iter::<f32>()
                    .next()
                    .ok_or("training loss was empty")?;
                Ok(f64::from(value))
            })
            .transpose()?;
        Ok(ParameterState {
            trainable,
            frozen,
            gradients,
            updated_parameters: self.last_updated,
            loss,
            parameter_sum,
            parameter_l2: parameter_squared_sum.sqrt(),
            parameter_count,
            parameter_sha256: parameter_sha256(&parameter_values),
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
    let bytes = training_mlp_model()?;
    let sha256 = model_sha256(&bytes);
    let cache = cache_policy(backend);
    let mut reports = Vec::new();
    for case in load_training_cases()? {
        let (mut workload, parse_ms, prepare_ms) =
            Workload::new(bytes.clone(), &case, device.clone())?;
        reports.push(measure_training(
            &case,
            mode,
            sync_policy,
            TrainingRun {
                engine: "tynx-imported-train",
                backend,
                device: device_name.clone(),
                parse_ms,
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

fn tensor(values: Vec<f32>, shape: &[usize], device: &Device) -> BenchResult<DynTensor> {
    Ok(DynTensor::from_data(
        TensorData::new(values, shape.to_vec()),
        shape.len(),
        device,
    )?)
}

fn imported_model(session: Session, device: Device) -> BenchResult<ImportedModel> {
    let report = TrainabilityReport::analyze_initializers_with_names(
        session.graph(),
        &TrainabilityOverrides::new(),
        session.initializer_names(),
    );
    let mut names = InitializerNameOverrides::new();
    for initializer in report.trainable_parameters() {
        let usage = initializer
            .uses()
            .first()
            .ok_or("trainable initializer had no consumer")?;
        let suffix = match usage.input_index() {
            1 => "weight",
            2 => "bias",
            index => return Err(format!("unexpected trainable input index {index}").into()),
        };
        names.set_name(
            initializer.name(),
            format!("{}.{}", usage.node_name(), suffix),
        )?;
    }
    Ok(ImportedModel::from_session_with(
        session,
        device,
        &TrainabilityOverrides::new(),
        &names,
    )?)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initializer_override_names_survive_processed_session_round_trip() {
        let session = Session::from_bytes_with(&training_mlp_model().unwrap(), false).unwrap();
        let model = imported_model(session, Device::autodiff(Device::flex())).unwrap();
        let mut names = model
            .parameters()
            .named()
            .map(|(name, _)| name.to_string())
            .collect::<Vec<_>>();
        names.sort();

        assert_eq!(
            names,
            ["gemm1.bias", "gemm1.weight", "gemm2.bias", "gemm2.weight"]
        );
    }
}
