//! Shared benchmark cases, timing, validation, and JSON reporting.

use std::{env, error::Error, fs, hint::black_box, time::Instant};

use serde::{Deserialize, Serialize};

pub type BenchError = Box<dyn Error>;
pub type BenchResult<T> = Result<T, BenchError>;

/// Requested CPU thread configuration for a benchmark process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadRequest {
    /// Let the runtime choose its thread count.
    Auto,
    /// Use exactly this many threads.
    Fixed(usize),
}

impl ThreadRequest {
    /// Return the fixed thread count, or `None` for automatic selection.
    pub fn fixed(self) -> Option<usize> {
        match self {
            Self::Auto => None,
            Self::Fixed(threads) => Some(threads),
        }
    }

    /// Create report metadata after a runtime has been configured.
    pub fn report(self, runtime: &str, actual: Option<usize>) -> Threading {
        Threading {
            runtime: runtime.to_string(),
            mode: if matches!(self, Self::Auto) {
                "auto"
            } else {
                "fixed"
            },
            requested: self.fixed(),
            actual,
        }
    }
}

/// CPU threading metadata attached to a benchmark report.
#[derive(Clone, Debug, Serialize)]
pub struct Threading {
    /// Threading implementation used by the engine.
    pub runtime: String,
    /// Either `auto` or `fixed`.
    pub mode: &'static str,
    /// Explicitly requested thread count, if any.
    pub requested: Option<usize>,
    /// Runtime thread count when it can be queried reliably.
    pub actual: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct Input {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Case {
    pub id: String,
    pub model: String,
    pub inputs: Vec<Input>,
    pub output_shape: Vec<usize>,
    pub expected: Vec<f32>,
    pub tolerance: f32,
    pub warmup: usize,
    pub iterations: usize,
    pub estimated_flops: Option<u64>,
    pub batch_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct Registry {
    schema: u32,
    cases: Vec<CaseSpec>,
}

#[derive(Debug, Deserialize)]
struct CaseSpec {
    id: String,
    model: String,
    inputs: Vec<InputSpec>,
    expected: TensorSpec,
    tolerance: f32,
    warmup: usize,
    iterations: usize,
    estimated_flops: Option<u64>,
    batch_size: Option<usize>,
    required_feature: Option<String>,
}

#[derive(Debug, Deserialize)]
struct InputSpec {
    name: String,
    #[serde(flatten)]
    tensor: TensorSpec,
}

#[derive(Debug, Deserialize)]
struct TensorSpec {
    shape: Vec<usize>,
    data: DataSpec,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum DataSpec {
    ClampedLinear {
        start: f32,
        step: f32,
        min: f32,
        max: f32,
    },
    Constant {
        value: f32,
    },
    Explicit {
        values: Vec<f32>,
    },
    Linear {
        start: f32,
        step: f32,
    },
    RepeatedExplicit {
        values: Vec<f32>,
    },
    RepeatedLinear {
        start: f32,
        step: f32,
        period: usize,
    },
    Reference {
        path: String,
    },
    Identity,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub schema: u32,
    pub revision: Option<String>,
    pub engine: String,
    pub backend: String,
    pub device: Option<String>,
    pub case: String,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,
    pub dtype: &'static str,
    pub warmup: usize,
    pub iterations: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threading: Option<Threading>,
    pub load_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prepare_ms: Option<f64>,
    pub cold_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub min_ms: f64,
    pub mean_ms: f64,
    pub throughput_per_second: f64,
    pub batch_size: Option<usize>,
    pub throughput_items_per_second: Option<f64>,
    pub estimated_gflops: Option<f64>,
    pub output_checksum: f64,
}

impl Report {
    /// Attach the process's CPU thread configuration.
    pub fn with_threading(mut self, threading: Threading) -> Self {
        self.threading = Some(threading);
        self
    }
}

/// Read `TYNX_BENCH_THREADS` as `auto`, `single`, or a positive integer.
pub fn thread_request() -> BenchResult<ThreadRequest> {
    let value = match env::var("TYNX_BENCH_THREADS") {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => return Ok(ThreadRequest::Auto),
        Err(error) => return Err(error.into()),
    };
    parse_thread_request(&value)
}

fn parse_thread_request(value: &str) -> BenchResult<ThreadRequest> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("auto") {
        return Ok(ThreadRequest::Auto);
    }
    if value.eq_ignore_ascii_case("single") {
        return Ok(ThreadRequest::Fixed(1));
    }
    let threads = value.parse::<usize>().map_err(|_| {
        format!("TYNX_BENCH_THREADS must be 'auto', 'single', or a positive integer; got '{value}'")
    })?;
    if threads == 0 {
        return Err("TYNX_BENCH_THREADS must be greater than zero".into());
    }
    Ok(ThreadRequest::Fixed(threads))
}

pub fn load_cases() -> BenchResult<Vec<Case>> {
    let registry = registry()?;
    let requested = match env::var("TYNX_BENCH_CASE") {
        Ok(value) if !value.is_empty() => Some(value),
        Ok(_) | Err(env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    let specs = match requested {
        Some(requested) => {
            let spec = registry
                .cases
                .into_iter()
                .find(|case| case.id == requested)
                .ok_or_else(|| format!("unknown benchmark case '{requested}'"))?;
            if !feature_enabled(spec.required_feature.as_deref()) {
                return Err(format!(
                    "benchmark case '{requested}' requires the '{}' feature",
                    spec.required_feature.as_deref().unwrap_or("unknown")
                )
                .into());
            }
            vec![spec]
        }
        None => registry
            .cases
            .into_iter()
            .filter(|spec| feature_enabled(spec.required_feature.as_deref()))
            .collect(),
    };

    specs
        .into_iter()
        .map(|spec| {
            let mut case = case_from_spec(spec)?;
            case.warmup = env_usize("TYNX_BENCH_WARMUP", case.warmup)?;
            case.iterations = env_usize("TYNX_BENCH_ITERATIONS", case.iterations)?;
            if case.iterations == 0 {
                return Err("TYNX_BENCH_ITERATIONS must be greater than zero".into());
            }
            Ok(case)
        })
        .collect()
}

pub fn load_case_named(requested: &str) -> BenchResult<Case> {
    let spec = registry()?
        .cases
        .into_iter()
        .find(|case| case.id == requested)
        .ok_or_else(|| format!("unknown benchmark case '{requested}'"))?;
    case_from_spec(spec)
}

fn registry() -> BenchResult<Registry> {
    let registry: Registry = serde_json::from_str(include_str!("../../../cases.json"))?;
    if registry.schema != 2 {
        return Err(format!("unsupported benchmark registry schema {}", registry.schema).into());
    }
    Ok(registry)
}

fn case_from_spec(spec: CaseSpec) -> BenchResult<Case> {
    let inputs = spec
        .inputs
        .into_iter()
        .map(|input| {
            Ok(Input {
                name: input.name,
                values: expand(&input.tensor)?,
                shape: input.tensor.shape,
            })
        })
        .collect::<BenchResult<Vec<_>>>()?;
    let expected = expand(&spec.expected)?;

    Ok(Case {
        id: spec.id,
        model: spec.model,
        inputs,
        output_shape: spec.expected.shape,
        expected,
        tolerance: spec.tolerance,
        warmup: spec.warmup,
        iterations: spec.iterations,
        estimated_flops: spec.estimated_flops,
        batch_size: spec.batch_size,
    })
}

pub fn model_bytes(case: &Case) -> BenchResult<Vec<u8>> {
    match case.model.as_str() {
        "models/sign.onnx.hex" => decode_hex(include_str!("../../../models/sign.onnx.hex")),
        "models/matmul64.onnx.hex" => decode_hex(include_str!("../../../models/matmul64.onnx.hex")),
        "models/matmul_dynamic.onnx.hex" => {
            decode_hex(include_str!("../../../models/matmul_dynamic.onnx.hex"))
        }
        "models/matmul_add_relu_dynamic.onnx.hex" => decode_hex(include_str!(
            "../../../models/matmul_add_relu_dynamic.onnx.hex"
        )),
        "models/tiny_cnn.onnx" => Ok(include_bytes!("../../../models/tiny_cnn.onnx").to_vec()),
        "external/mobilenetv2_100_opset16.onnx" => {
            let path = env::var_os("TYNX_BENCH_MOBILENET_PATH")
                .ok_or("TYNX_BENCH_MOBILENET_PATH is required for the MobileNetV2 benchmark")?;
            Ok(fs::read(path)?)
        }
        model => Err(format!("model '{model}' is not embedded in the benchmark harness").into()),
    }
}

pub fn require_release() -> BenchResult<()> {
    if cfg!(debug_assertions) {
        return Err("benchmarks must be run with --release".into());
    }
    Ok(())
}

pub fn measure<F>(
    engine: &str,
    backend: &str,
    device: Option<String>,
    load_ms: f64,
    case: &Case,
    run: F,
) -> BenchResult<Report>
where
    F: FnMut() -> BenchResult<Vec<f32>>,
{
    measure_prepared(engine, backend, device, load_ms, None, case, run)
}

pub fn measure_prepared<F>(
    engine: &str,
    backend: &str,
    device: Option<String>,
    load_ms: f64,
    prepare_ms: Option<f64>,
    case: &Case,
    mut run: F,
) -> BenchResult<Report>
where
    F: FnMut() -> BenchResult<Vec<f32>>,
{
    let started = Instant::now();
    let cold_output = run()?;
    let cold_ms = elapsed_ms(started);
    validate(case, &cold_output)?;

    for _ in 0..case.warmup {
        black_box(run()?);
    }

    let mut samples = Vec::with_capacity(case.iterations);
    let mut last_output = cold_output;
    for _ in 0..case.iterations {
        let started = Instant::now();
        last_output = run()?;
        samples.push(elapsed_ms(started));
        black_box(&last_output);
    }
    validate(case, &last_output)?;

    samples.sort_by(f64::total_cmp);
    let median_ms = percentile(&samples, 0.50);
    let mean_ms = samples.iter().sum::<f64>() / samples.len() as f64;

    Ok(Report {
        schema: 2,
        revision: env::var("GITHUB_SHA").ok(),
        engine: engine.to_string(),
        backend: backend.to_string(),
        device: device.or_else(|| env::var("TYNX_BENCH_DEVICE").ok()),
        case: case.id.clone(),
        input_shapes: case
            .inputs
            .iter()
            .map(|input| input.shape.clone())
            .collect(),
        output_shape: case.output_shape.clone(),
        dtype: "f32",
        warmup: case.warmup,
        iterations: case.iterations,
        threading: None,
        load_ms,
        prepare_ms,
        cold_ms,
        median_ms,
        p95_ms: percentile(&samples, 0.95),
        min_ms: samples[0],
        mean_ms,
        throughput_per_second: 1_000.0 / median_ms,
        batch_size: case.batch_size,
        throughput_items_per_second: case
            .batch_size
            .map(|batch_size| batch_size as f64 * 1_000.0 / median_ms),
        estimated_gflops: case
            .estimated_flops
            .map(|flops| flops as f64 / (median_ms * 1_000_000.0)),
        output_checksum: last_output.iter().map(|&value| f64::from(value)).sum(),
    })
}

#[cfg(feature = "wgpu-device")]
pub fn wgpu_device_name() -> Option<String> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .ok()?;
    let info = adapter.get_info();
    Some(format!(
        "{} ({:?}, {:?})",
        info.name, info.backend, info.device_type
    ))
}

pub fn print_reports(reports: &[Report]) -> BenchResult<()> {
    println!("{}", serde_json::to_string_pretty(reports)?);
    Ok(())
}

fn expand(spec: &TensorSpec) -> BenchResult<Vec<f32>> {
    let len = element_count(&spec.shape)?;
    let values = match &spec.data {
        DataSpec::ClampedLinear {
            start,
            step,
            min,
            max,
        } => (0..len)
            .map(|index| (start + *step * index as f32).clamp(*min, *max))
            .collect(),
        DataSpec::Constant { value } => vec![*value; len],
        DataSpec::Explicit { values } => values.clone(),
        DataSpec::Linear { start, step } => {
            (0..len).map(|index| start + *step * index as f32).collect()
        }
        DataSpec::RepeatedExplicit { values } => {
            if values.is_empty() || !len.is_multiple_of(values.len()) {
                return Err("repeated explicit data must evenly divide the tensor shape".into());
            }
            values.iter().copied().cycle().take(len).collect()
        }
        DataSpec::RepeatedLinear {
            start,
            step,
            period,
        } => {
            if *period == 0 || !len.is_multiple_of(*period) {
                return Err("repeated linear period must evenly divide the tensor shape".into());
            }
            (0..len)
                .map(|index| start + *step * (index % period) as f32)
                .collect()
        }
        DataSpec::Reference { path } => reference_values(path)?,
        DataSpec::Identity => {
            if spec.shape.len() != 2 || spec.shape[0] != spec.shape[1] {
                return Err("identity data requires a square rank-2 tensor".into());
            }
            let size = spec.shape[0];
            let mut values = vec![0.0; len];
            for index in 0..size {
                values[index * size + index] = 1.0;
            }
            values
        }
    };
    if values.len() != len {
        return Err(format!(
            "tensor shape {:?} requires {len} values, found {}",
            spec.shape,
            values.len()
        )
        .into());
    }
    Ok(values)
}

fn feature_enabled(required: Option<&str>) -> bool {
    match required {
        None => true,
        Some("mobilenet") => cfg!(feature = "mobilenet"),
        Some(_) => false,
    }
}

fn reference_values(path: &str) -> BenchResult<Vec<f32>> {
    match path {
        "models/mobilenetv2_100_opset16.expected.json" => Ok(serde_json::from_str(include_str!(
            "../../../models/mobilenetv2_100_opset16.expected.json"
        ))?),
        path => Err(format!("unknown benchmark reference '{path}'").into()),
    }
}

fn element_count(shape: &[usize]) -> BenchResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
        .ok_or_else(|| "tensor shape element count overflowed".into())
}

fn validate(case: &Case, output: &[f32]) -> BenchResult<()> {
    if output.len() != case.expected.len() {
        return Err(format!(
            "case '{}' produced {} values, expected {}",
            case.id,
            output.len(),
            case.expected.len()
        )
        .into());
    }

    for (index, (&actual, &expected)) in output.iter().zip(&case.expected).enumerate() {
        if !actual.is_finite() || (actual - expected).abs() > case.tolerance {
            return Err(format!(
                "case '{}' output {index} was {actual}, expected {expected} +/- {}",
                case.id, case.tolerance
            )
            .into());
        }
    }
    Ok(())
}

fn percentile(samples: &[f64], quantile: f64) -> f64 {
    let index = ((samples.len() as f64 * quantile).ceil() as usize)
        .saturating_sub(1)
        .min(samples.len() - 1);
    samples[index]
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn env_usize(name: &str, default: usize) -> BenchResult<usize> {
    match env::var(name) {
        Ok(value) if value.is_empty() => Ok(default),
        Ok(value) => Ok(value
            .parse()
            .map_err(|error| format!("invalid {name} value '{value}': {error}"))?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn decode_hex(input: &str) -> BenchResult<Vec<u8>> {
    let input = input.trim();
    if !input.len().is_multiple_of(2) {
        return Err("hex model has an odd number of characters".into());
    }

    (0..input.len())
        .step_by(2)
        .map(|offset| Ok(u8::from_str_radix(&input[offset..offset + 2], 16)?))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_embedded_cases_and_models() {
        let sign = load_case_named("sign-11").unwrap();
        let matmul64 = load_case_named("matmul-64x64").unwrap();
        let matmul256 = load_case_named("matmul-256x256").unwrap();
        let matmul512 = load_case_named("matmul-512x512").unwrap();
        let matmul1024 = load_case_named("matmul-1024x1024").unwrap();
        let multi_op = load_case_named("matmul-add-relu-256x256").unwrap();
        let tiny_cnn = load_case_named("tiny-cnn-32").unwrap();
        let tiny_cnn_b128 = load_case_named("tiny-cnn-32-b128").unwrap();
        let mobilenet = load_case_named("mobilenetv2-100-b1").unwrap();

        assert_eq!(sign.inputs[0].values.len(), 11);
        assert_eq!(model_bytes(&sign).unwrap().len(), 83);
        assert_eq!(matmul64.inputs.len(), 2);
        assert_eq!(matmul64.inputs[0].values, matmul64.expected);
        assert_eq!(model_bytes(&matmul64).unwrap().len(), 122);
        assert_eq!(matmul256.output_shape, [256, 256]);
        assert_eq!(matmul256.estimated_flops, Some(33_554_432));
        assert_eq!(model_bytes(&matmul256).unwrap().len(), 124);
        assert_eq!(matmul512.output_shape, [512, 512]);
        assert_eq!(matmul512.iterations, 50);
        assert_eq!(matmul1024.output_shape, [1024, 1024]);
        assert_eq!(matmul1024.iterations, 20);
        assert_eq!(matmul1024.estimated_flops, Some(2_147_483_648));
        assert_eq!(multi_op.inputs.len(), 3);
        assert!(multi_op.inputs[2].values.iter().all(|value| *value == 0.25));
        assert_eq!(multi_op.expected[0], 0.0);
        assert!(multi_op.expected.last().unwrap() > &1.0);
        assert_eq!(model_bytes(&multi_op).unwrap().len(), 208);
        assert_eq!(tiny_cnn.inputs[0].shape, [1, 3, 32, 32]);
        assert_eq!(tiny_cnn.output_shape, [1, 10]);
        assert_eq!(tiny_cnn.estimated_flops, Some(1_032_512));
        assert_eq!(tiny_cnn.batch_size, Some(1));
        assert!((tiny_cnn.expected.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(model_bytes(&tiny_cnn).unwrap().len(), 7_242);
        assert_eq!(tiny_cnn_b128.inputs[0].values.len(), 128 * 3 * 32 * 32);
        assert_eq!(tiny_cnn_b128.expected.len(), 128 * 10);
        assert_eq!(tiny_cnn_b128.batch_size, Some(128));
        assert_eq!(tiny_cnn_b128.estimated_flops, Some(132_161_536));
        assert_eq!(mobilenet.inputs[0].shape, [1, 3, 224, 224]);
        assert_eq!(mobilenet.expected.len(), 1_000);
        assert_eq!(mobilenet.estimated_flops, Some(601_548_544));
    }

    #[test]
    fn registry_contains_all_cases() {
        assert_eq!(registry().unwrap().cases.len(), 11);
    }

    #[test]
    fn validates_output_tolerance() {
        let case = load_case_named("sign-11").unwrap();

        validate(
            &case,
            &[-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        assert!(validate(&case, &[-1.0; 11]).is_err());
    }

    #[test]
    fn parses_thread_requests() {
        assert_eq!(parse_thread_request("").unwrap(), ThreadRequest::Auto);
        assert_eq!(parse_thread_request("auto").unwrap(), ThreadRequest::Auto);
        assert_eq!(
            parse_thread_request("single").unwrap(),
            ThreadRequest::Fixed(1)
        );
        assert_eq!(parse_thread_request("8").unwrap(), ThreadRequest::Fixed(8));
        assert!(parse_thread_request("0").is_err());
        assert!(parse_thread_request("many").is_err());
    }
}
