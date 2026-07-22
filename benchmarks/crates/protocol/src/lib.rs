//! Shared benchmark cases, timing, validation, and JSON reporting.

use std::{env, error::Error, hint::black_box, time::Instant};

use serde::{Deserialize, Serialize};

pub type BenchError = Box<dyn Error>;
pub type BenchResult<T> = Result<T, BenchError>;

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
    pub load_ms: f64,
    pub cold_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub min_ms: f64,
    pub mean_ms: f64,
    pub throughput_per_second: f64,
    pub estimated_gflops: Option<f64>,
    pub output_checksum: f64,
}

pub fn load_case() -> BenchResult<Case> {
    let requested = env::var("TYNX_BENCH_CASE").unwrap_or_else(|_| "sign-11".to_string());
    let mut case = load_case_named(&requested)?;
    case.warmup = env_usize("TYNX_BENCH_WARMUP", case.warmup)?;
    case.iterations = env_usize("TYNX_BENCH_ITERATIONS", case.iterations)?;
    if case.iterations == 0 {
        return Err("TYNX_BENCH_ITERATIONS must be greater than zero".into());
    }
    Ok(case)
}

pub fn load_case_named(requested: &str) -> BenchResult<Case> {
    let registry: Registry = serde_json::from_str(include_str!("../../../cases.json"))?;
    if registry.schema != 2 {
        return Err(format!("unsupported benchmark registry schema {}", registry.schema).into());
    }

    let spec = registry
        .cases
        .into_iter()
        .find(|case| case.id == requested)
        .ok_or_else(|| format!("unknown benchmark case '{requested}'"))?;
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
        load_ms,
        cold_ms,
        median_ms,
        p95_ms: percentile(&samples, 0.95),
        min_ms: samples[0],
        mean_ms,
        throughput_per_second: 1_000.0 / median_ms,
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

pub fn print_report(report: &Report) -> BenchResult<()> {
    println!("{}", serde_json::to_string_pretty(report)?);
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
}
