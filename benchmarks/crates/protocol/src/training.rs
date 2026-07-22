//! Training benchmark fixtures, measurement policy, and JSON reports.

use std::{
    env,
    hint::black_box,
    process::Command,
    time::{Duration, Instant},
};

use onnx_ir::{GraphProto, ModelProto, TensorProto, TypeProto, ValueInfoProto};
use protobuf::{Message, MessageField};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{BenchResult, Threading};

pub const INPUT_FEATURES: usize = 784;
pub const HIDDEN_FEATURES: usize = 512;
pub const OUTPUT_FEATURES: usize = 10;
pub const LEARNING_RATE: f64 = 1.0e-3;
pub const BURN_REVISION: &str = "78f10aec1ca6c6ffb1edd17a0fa131ae59ad5403";
pub const BURN_ONNX_REVISION: &str = "af2dfb43af43bf363dc2d7d858d933d86e2a65a8";

const CORRECTNESS_STEPS: usize = 5;
const WARMUP_MIN: usize = 20;
const WARMUP_MAX: usize = 200;
const WARMUP_WINDOW: usize = 10;
const WARMUP_STABLE_WINDOWS: usize = 3;
const WARMUP_TOLERANCE: f64 = 0.02;

#[derive(Clone, Debug)]
pub struct TrainingCase {
    pub id: String,
    pub batch_size: usize,
    pub dataset_batches: usize,
    pub iterations: usize,
    pub reset_interval: usize,
    pub estimated_step_flops: u64,
    pub expected_trainable: Vec<String>,
    pub expected_frozen: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Registry {
    schema: u32,
    cases: Vec<CaseSpec>,
}

#[derive(Debug, Deserialize)]
struct CaseSpec {
    id: String,
    batch_size: usize,
    dataset_batches: usize,
    iterations: usize,
    reset_interval: usize,
    estimated_step_flops: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    ForwardLoss,
    ForwardBackward,
    TrainStep,
}

impl TrainingMode {
    pub fn from_env() -> BenchResult<Self> {
        match env::var("TYNX_BENCH_TRAINING_MODE")
            .unwrap_or_else(|_| "train_step".to_string())
            .as_str()
        {
            "forward_loss" => Ok(Self::ForwardLoss),
            "forward_backward" => Ok(Self::ForwardBackward),
            "train_step" => Ok(Self::TrainStep),
            value => Err(format!(
                "TYNX_BENCH_TRAINING_MODE must be forward_loss, forward_backward, or train_step; got '{value}'"
            )
            .into()),
        }
    }

    fn expects_gradients(self) -> bool {
        matches!(self, Self::ForwardBackward | Self::TrainStep)
    }

    fn expects_updates(self) -> bool {
        matches!(self, Self::TrainStep)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncPolicy {
    EachStep,
    FinalOnly,
}

impl SyncPolicy {
    pub fn from_env() -> BenchResult<Self> {
        match env::var("TYNX_BENCH_SYNC_POLICY")
            .unwrap_or_else(|_| "each_step".to_string())
            .as_str()
        {
            "each_step" => Ok(Self::EachStep),
            "final_only" => Ok(Self::FinalOnly),
            value => Err(format!(
                "TYNX_BENCH_SYNC_POLICY must be each_step or final_only; got '{value}'"
            )
            .into()),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ParameterState {
    pub trainable: Vec<String>,
    pub frozen: Vec<String>,
    pub gradients: Vec<String>,
    pub updated_parameters: usize,
    pub loss: Option<f64>,
    pub parameter_sum: f64,
    pub parameter_l2: f64,
    pub parameter_count: usize,
    pub parameter_sha256: String,
    pub finite: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrajectoryPoint {
    pub step: usize,
    pub state: ParameterState,
}

#[derive(Clone, Debug, Serialize)]
pub struct CorrectnessReport {
    pub initial: ParameterState,
    pub trajectory: Vec<TrajectoryPoint>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WarmupReport {
    pub minimum_steps: usize,
    pub maximum_steps: usize,
    pub window: usize,
    pub tolerance: f64,
    pub required_stable_windows: usize,
    pub actual_steps: usize,
    pub converged: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CachePolicy {
    pub process: String,
    pub autotune: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Environment {
    pub revision: Option<String>,
    pub rustc: Option<String>,
    pub os: &'static str,
    pub arch: &'static str,
    pub cpu: Option<String>,
    pub driver: Option<String>,
    pub burn_revision: &'static str,
    pub burn_onnx_revision: &'static str,
    pub build_profile: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingReport {
    pub schema: u32,
    pub kind: &'static str,
    pub engine: String,
    pub backend: String,
    pub device: Option<String>,
    pub case: String,
    pub mode: TrainingMode,
    pub sync_policy: SyncPolicy,
    pub optimizer: &'static str,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub dataset_batches: usize,
    pub parameter_count: usize,
    pub model_sha256: String,
    pub cache: CachePolicy,
    pub environment: Environment,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threading: Option<Threading>,
    pub parse_ms: f64,
    pub prepare_ms: f64,
    pub cold_step_ms: f64,
    pub warmup: WarmupReport,
    pub iterations: usize,
    pub reset_interval: usize,
    pub sample_count: usize,
    pub median_step_ms: f64,
    pub p95_step_ms: f64,
    pub min_step_ms: f64,
    pub mean_step_ms: f64,
    pub samples_per_second: f64,
    pub estimated_training_gflops: f64,
    pub empty_sync_ms: f64,
    pub correctness: CorrectnessReport,
    pub final_state: ParameterState,
}

/// Engine-specific state used by the shared training measurement policy.
pub trait TrainingWorkload {
    fn reset(&mut self) -> BenchResult<()>;
    fn step(&mut self, batch_index: usize, mode: TrainingMode) -> BenchResult<()>;
    fn sync(&self) -> BenchResult<()>;
    fn state(&self) -> BenchResult<ParameterState>;
}

pub struct TrainingRun<'a> {
    pub engine: &'a str,
    pub backend: &'a str,
    pub device: Option<String>,
    pub parse_ms: f64,
    pub prepare_ms: f64,
    pub model_sha256: String,
    pub threading: Option<Threading>,
    pub cache: CachePolicy,
}

pub fn load_training_cases() -> BenchResult<Vec<TrainingCase>> {
    let registry: Registry = serde_json::from_str(include_str!("../../../training-cases.json"))?;
    if registry.schema != 1 {
        return Err(format!(
            "unsupported training benchmark registry schema {}",
            registry.schema
        )
        .into());
    }
    let requested = env::var("TYNX_BENCH_CASE")
        .ok()
        .filter(|value| !value.is_empty());
    let iterations = env_optional_usize("TYNX_BENCH_ITERATIONS")?;
    registry
        .cases
        .into_iter()
        .filter(|case| requested.as_ref().is_none_or(|value| value == &case.id))
        .map(|case| {
            let iterations = iterations.unwrap_or(case.iterations);
            if iterations == 0 || case.dataset_batches == 0 || case.reset_interval == 0 {
                return Err(
                    "training iterations, dataset batches, and reset interval must be positive"
                        .into(),
                );
            }
            Ok(TrainingCase {
                id: case.id,
                batch_size: case.batch_size,
                dataset_batches: case.dataset_batches,
                iterations,
                reset_interval: case.reset_interval,
                estimated_step_flops: case.estimated_step_flops,
                expected_trainable: vec![
                    "gemm1.bias".to_string(),
                    "gemm1.weight".to_string(),
                    "gemm2.bias".to_string(),
                    "gemm2.weight".to_string(),
                ],
                expected_frozen: Vec::new(),
            })
        })
        .collect::<BenchResult<Vec<_>>>()
        .and_then(|cases| {
            if cases.is_empty() {
                Err(format!(
                    "unknown training benchmark case '{}'",
                    requested.unwrap_or_default()
                )
                .into())
            } else {
                Ok(cases)
            }
        })
}

pub fn deterministic_batch(case: &TrainingCase, batch_index: usize) -> (Vec<f32>, Vec<f32>) {
    let input_len = case.batch_size * INPUT_FEATURES;
    let target_len = case.batch_size * OUTPUT_FEATURES;
    let inputs = (0..input_len)
        .map(|index| {
            let value = (index + batch_index * 37) % 257;
            (value as f32 - 128.0) / 256.0
        })
        .collect();
    let targets = (0..target_len)
        .map(|index| {
            let sample = index / OUTPUT_FEATURES;
            let class = index % OUTPUT_FEATURES;
            let label = (sample + batch_index * 3) % OUTPUT_FEATURES;
            if class == label { 0.75 } else { -0.25 }
        })
        .collect();
    (inputs, targets)
}

pub fn training_mlp_model() -> BenchResult<Vec<u8>> {
    let mut graph = GraphProto::new();
    graph.name = "tynx_training_mlp".to_string();
    graph.input.push(value_info("x", &[INPUT_FEATURES]));
    graph.output.push(value_info("y", &[OUTPUT_FEATURES]));
    graph.initializer = vec![
        tensor("fc1.weight", &[INPUT_FEATURES, HIDDEN_FEATURES], 17),
        tensor("fc1.bias", &[HIDDEN_FEATURES], 19),
        tensor("fc2.weight", &[HIDDEN_FEATURES, OUTPUT_FEATURES], 23),
        tensor("fc2.bias", &[OUTPUT_FEATURES], 29),
    ];
    push_node(
        &mut graph,
        "fc1",
        "Gemm",
        &["x", "fc1.weight", "fc1.bias"],
        &["hidden"],
    );
    push_node(&mut graph, "relu", "Relu", &["hidden"], &["activated"]);
    push_node(
        &mut graph,
        "fc2",
        "Gemm",
        &["activated", "fc2.weight", "fc2.bias"],
        &["y"],
    );

    let mut model = ModelProto::new();
    model.ir_version = 8;
    model.graph = MessageField::some(graph);
    model.opset_import.push(Default::default());
    model.opset_import[0].version = 13;
    Ok(model.write_to_bytes()?)
}

pub fn model_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn parameter_sha256(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

pub fn cache_policy(backend: &str) -> CachePolicy {
    CachePolicy {
        process: env::var("TYNX_BENCH_PROCESS_CACHE")
            .unwrap_or_else(|_| "fresh_process".to_string()),
        autotune: env::var("TYNX_BENCH_AUTOTUNE_CACHE").unwrap_or_else(|_| {
            if backend == "flex" {
                "not_applicable".to_string()
            } else {
                "unknown".to_string()
            }
        }),
    }
}

pub fn measure_training<W: TrainingWorkload>(
    case: &TrainingCase,
    mode: TrainingMode,
    sync_policy: SyncPolicy,
    run: TrainingRun<'_>,
    workload: &mut W,
) -> BenchResult<TrainingReport> {
    workload.sync()?;
    let started = Instant::now();
    workload.step(0, mode)?;
    workload.sync()?;
    let cold_step_ms = elapsed_ms(started);

    let correctness = correctness(case, mode, workload)?;
    workload.reset()?;
    workload.sync()?;
    let warmup = stabilize(case, mode, workload)?;
    workload.reset()?;
    workload.sync()?;

    let samples = measure_samples(case, mode, sync_policy, workload)?;
    let final_state = workload.state()?;
    validate_state(case, mode, &final_state)?;
    let empty_sync_ms = measure_empty_sync(workload)?;

    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let median_step_ms = percentile(&sorted, 0.50);
    let mean_step_ms = samples.iter().sum::<f64>() / samples.len() as f64;
    let parameter_count = correctness.initial.parameter_count;

    Ok(TrainingReport {
        schema: 1,
        kind: "training",
        engine: run.engine.to_string(),
        backend: run.backend.to_string(),
        device: run.device.or_else(|| env::var("TYNX_BENCH_DEVICE").ok()),
        case: case.id.clone(),
        mode,
        sync_policy,
        optimizer: "sgd",
        learning_rate: LEARNING_RATE,
        batch_size: case.batch_size,
        dataset_batches: case.dataset_batches,
        parameter_count,
        model_sha256: run.model_sha256,
        cache: run.cache,
        environment: environment(),
        threading: run.threading,
        parse_ms: run.parse_ms,
        prepare_ms: run.prepare_ms,
        cold_step_ms,
        warmup,
        iterations: case.iterations,
        reset_interval: case.reset_interval,
        sample_count: samples.len(),
        median_step_ms,
        p95_step_ms: percentile(&sorted, 0.95),
        min_step_ms: sorted[0],
        mean_step_ms,
        samples_per_second: case.batch_size as f64 * 1_000.0 / median_step_ms,
        estimated_training_gflops: case.estimated_step_flops as f64
            / (median_step_ms * 1_000_000.0),
        empty_sync_ms,
        correctness,
        final_state,
    })
}

pub fn print_training_reports(reports: &[TrainingReport]) -> BenchResult<()> {
    println!("{}", serde_json::to_string_pretty(reports)?);
    Ok(())
}

fn correctness<W: TrainingWorkload>(
    case: &TrainingCase,
    mode: TrainingMode,
    workload: &mut W,
) -> BenchResult<CorrectnessReport> {
    workload.reset()?;
    workload.sync()?;
    let initial = workload.state()?;
    validate_initial_state(case, &initial)?;
    let mut trajectory = Vec::with_capacity(CORRECTNESS_STEPS);
    for step in 1..=CORRECTNESS_STEPS {
        workload.step((step - 1) % case.dataset_batches, mode)?;
        workload.sync()?;
        let state = workload.state()?;
        validate_state(case, mode, &state)?;
        trajectory.push(TrajectoryPoint { step, state });
    }
    Ok(CorrectnessReport {
        initial,
        trajectory,
    })
}

fn stabilize<W: TrainingWorkload>(
    case: &TrainingCase,
    mode: TrainingMode,
    workload: &mut W,
) -> BenchResult<WarmupReport> {
    let minimum = env_optional_usize("TYNX_BENCH_WARMUP_MIN")?.unwrap_or(WARMUP_MIN);
    let maximum = env_optional_usize("TYNX_BENCH_WARMUP_MAX")?.unwrap_or(WARMUP_MAX);
    if maximum < minimum || minimum < WARMUP_WINDOW * 2 {
        return Err("warmup maximum must be >= minimum and minimum must span two windows".into());
    }
    let mut samples = Vec::with_capacity(maximum);
    let mut stable_windows = 0;
    let mut converged = false;
    for step in 0..maximum {
        let started = Instant::now();
        workload.step(step % case.dataset_batches, mode)?;
        workload.sync()?;
        samples.push(elapsed_ms(started));
        if samples.len() < minimum || samples.len() < WARMUP_WINDOW * 2 {
            continue;
        }
        let split = samples.len() - WARMUP_WINDOW;
        let previous = median(&samples[split - WARMUP_WINDOW..split]);
        let current = median(&samples[split..]);
        let change = (current - previous).abs() / previous.abs().max(f64::EPSILON);
        if change <= WARMUP_TOLERANCE {
            stable_windows += 1;
            if stable_windows >= WARMUP_STABLE_WINDOWS {
                converged = true;
                break;
            }
        } else {
            stable_windows = 0;
        }
    }
    Ok(WarmupReport {
        minimum_steps: minimum,
        maximum_steps: maximum,
        window: WARMUP_WINDOW,
        tolerance: WARMUP_TOLERANCE,
        required_stable_windows: WARMUP_STABLE_WINDOWS,
        actual_steps: samples.len(),
        converged,
    })
}

fn measure_samples<W: TrainingWorkload>(
    case: &TrainingCase,
    mode: TrainingMode,
    sync_policy: SyncPolicy,
    workload: &mut W,
) -> BenchResult<Vec<f64>> {
    let mut samples = Vec::new();
    let mut completed = 0;
    while completed < case.iterations {
        let block = case
            .reset_interval
            .min(case.iterations.saturating_sub(completed));
        workload.reset()?;
        workload.sync()?;
        match sync_policy {
            SyncPolicy::EachStep => {
                for offset in 0..block {
                    let started = Instant::now();
                    workload.step((completed + offset) % case.dataset_batches, mode)?;
                    workload.sync()?;
                    samples.push(elapsed_ms(started));
                }
            }
            SyncPolicy::FinalOnly => {
                let started = Instant::now();
                for offset in 0..block {
                    workload.step((completed + offset) % case.dataset_batches, mode)?;
                }
                workload.sync()?;
                let per_step = elapsed_ms(started) / block as f64;
                samples.push(per_step);
            }
        }
        completed += block;
    }
    if samples.is_empty() {
        return Err("training benchmark produced no timing samples".into());
    }
    black_box(&samples);
    Ok(samples)
}

fn measure_empty_sync<W: TrainingWorkload>(workload: &W) -> BenchResult<f64> {
    const SYNCS: usize = 20;
    let started = Instant::now();
    for _ in 0..SYNCS {
        workload.sync()?;
    }
    Ok(elapsed_ms(started) / SYNCS as f64)
}

fn validate_initial_state(case: &TrainingCase, state: &ParameterState) -> BenchResult<()> {
    validate_sets(case, state)?;
    if !state.gradients.is_empty() || state.updated_parameters != 0 || !state.finite {
        return Err("initial training state has gradients, updates, or non-finite values".into());
    }
    Ok(())
}

fn validate_state(
    case: &TrainingCase,
    mode: TrainingMode,
    state: &ParameterState,
) -> BenchResult<()> {
    validate_sets(case, state)?;
    if !state.finite || state.loss.is_none_or(|loss| !loss.is_finite()) {
        return Err("training state contains a non-finite parameter or loss".into());
    }
    let expected_gradients = if mode.expects_gradients() {
        &case.expected_trainable
    } else {
        &case.expected_frozen
    };
    if &state.gradients != expected_gradients {
        return Err(format!(
            "gradient set {:?} differs from expected {:?}",
            state.gradients, expected_gradients
        )
        .into());
    }
    let expected_updates = if mode.expects_updates() {
        case.expected_trainable.len()
    } else {
        0
    };
    if state.updated_parameters != expected_updates {
        return Err(format!(
            "updated {} parameters, expected {expected_updates}",
            state.updated_parameters
        )
        .into());
    }
    Ok(())
}

fn validate_sets(case: &TrainingCase, state: &ParameterState) -> BenchResult<()> {
    if state.trainable != case.expected_trainable || state.frozen != case.expected_frozen {
        return Err(format!(
            "parameter sets trainable={:?}, frozen={:?}; expected trainable={:?}, frozen={:?}",
            state.trainable, state.frozen, case.expected_trainable, case.expected_frozen
        )
        .into());
    }
    Ok(())
}

fn environment() -> Environment {
    Environment {
        revision: env::var("GITHUB_SHA").ok(),
        rustc: env::var("TYNX_BENCH_RUSTC")
            .ok()
            .or_else(|| command_output("rustc", &["--version"])),
        os: env::consts::OS,
        arch: env::consts::ARCH,
        cpu: env::var("TYNX_BENCH_CPU").ok(),
        driver: env::var("TYNX_BENCH_DRIVER").ok(),
        burn_revision: BURN_REVISION,
        burn_onnx_revision: BURN_ONNX_REVISION,
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    }
}

fn command_output(command: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new(command).args(arguments).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn push_node(
    graph: &mut GraphProto,
    name: &str,
    operator: &str,
    inputs: &[&str],
    outputs: &[&str],
) {
    graph.node.push(Default::default());
    let node = graph.node.last_mut().unwrap();
    node.name = name.to_string();
    node.op_type = operator.to_string();
    node.input = inputs.iter().map(|value| (*value).to_string()).collect();
    node.output = outputs.iter().map(|value| (*value).to_string()).collect();
}

fn value_info(name: &str, dimensions: &[usize]) -> ValueInfoProto {
    let mut value = ValueInfoProto::new();
    value.name = name.to_string();
    let mut ty = TypeProto::new();
    let tensor = ty.mut_tensor_type();
    tensor.elem_type = 1;
    let shape = tensor.shape.mut_or_insert_default();
    shape.dim.push(Default::default());
    shape.dim.last_mut().unwrap().set_dim_param("batch".into());
    for dimension in dimensions {
        shape.dim.push(Default::default());
        shape
            .dim
            .last_mut()
            .unwrap()
            .set_dim_value(*dimension as i64);
    }
    value.type_ = MessageField::some(ty);
    value
}

fn tensor(name: &str, dimensions: &[usize], seed: usize) -> TensorProto {
    let count = dimensions.iter().product();
    let values = (0..count)
        .map(|index| {
            let value = (index.wrapping_mul(seed).wrapping_add(seed * 7)) % 251;
            (value as f32 - 125.0) * 0.0001
        })
        .collect();
    let mut tensor = TensorProto::new();
    tensor.name = name.to_string();
    tensor.dims = dimensions.iter().map(|&value| value as i64).collect();
    tensor.data_type = 1;
    tensor.float_data = values;
    tensor
}

fn median(samples: &[f64]) -> f64 {
    let mut samples = samples.to_vec();
    samples.sort_by(f64::total_cmp);
    percentile(&samples, 0.5)
}

fn percentile(samples: &[f64], quantile: f64) -> f64 {
    let index = ((samples.len() as f64 * quantile).ceil() as usize)
        .saturating_sub(1)
        .min(samples.len() - 1);
    samples[index]
}

fn elapsed_ms(started: Instant) -> f64 {
    duration_ms(started.elapsed())
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn env_optional_usize(name: &str) -> BenchResult<Option<usize>> {
    match env::var(name) {
        Ok(value) if value.is_empty() => Ok(None),
        Ok(value) => {
            Ok(Some(value.parse().map_err(|error| {
                format!("invalid {name} value '{value}': {error}")
            })?))
        }
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_training_registry_and_fixture() {
        let cases = load_training_cases().unwrap();
        assert_eq!(cases.len(), 3);
        assert_eq!(cases[0].batch_size, 64);
        assert_eq!(cases[2].estimated_step_flops, 9_990_832_128);
        let model = training_mlp_model().unwrap();
        assert!(model.len() > 1_000_000);
        assert_eq!(model_sha256(&model).len(), 64);
    }

    #[test]
    fn deterministic_batches_are_stationary_and_distinct() {
        let case = &load_training_cases().unwrap()[0];
        let first = deterministic_batch(case, 0);
        let repeated = deterministic_batch(case, 0);
        let next = deterministic_batch(case, 1);
        assert_eq!(first, repeated);
        assert_ne!(first, next);
        assert_eq!(first.0.len(), 64 * INPUT_FEATURES);
        assert_eq!(first.1.len(), 64 * OUTPUT_FEATURES);
    }
}
