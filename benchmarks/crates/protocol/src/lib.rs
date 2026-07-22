//! Shared benchmark cases, timing, validation, and JSON reporting.

use std::{env, error::Error, hint::black_box, time::Instant};

use serde::{Deserialize, Serialize};

pub type BenchError = Box<dyn Error>;
pub type BenchResult<T> = Result<T, BenchError>;

#[derive(Clone, Debug, Deserialize)]
pub struct Case {
    pub id: String,
    pub model: String,
    pub input_shape: Vec<usize>,
    pub input: Vec<f32>,
    pub expected: Vec<f32>,
    pub tolerance: f32,
    pub warmup: usize,
    pub iterations: usize,
}

#[derive(Debug, Deserialize)]
struct Registry {
    schema: u32,
    cases: Vec<Case>,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub schema: u32,
    pub revision: Option<String>,
    pub engine: String,
    pub backend: String,
    pub device: Option<String>,
    pub case: String,
    pub input_shape: Vec<usize>,
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
    pub output_checksum: f64,
}

pub fn load_case() -> BenchResult<Case> {
    let registry: Registry = serde_json::from_str(include_str!("../../../cases.json"))?;
    if registry.schema != 1 {
        return Err(format!("unsupported benchmark registry schema {}", registry.schema).into());
    }

    let requested = env::var("TYNX_BENCH_CASE").unwrap_or_else(|_| "sign-11".to_string());
    let mut case = registry
        .cases
        .into_iter()
        .find(|case| case.id == requested)
        .ok_or_else(|| format!("unknown benchmark case '{requested}'"))?;
    case.warmup = env_usize("TYNX_BENCH_WARMUP", case.warmup)?;
    case.iterations = env_usize("TYNX_BENCH_ITERATIONS", case.iterations)?;
    if case.iterations == 0 {
        return Err("TYNX_BENCH_ITERATIONS must be greater than zero".into());
    }
    Ok(case)
}

pub fn model_bytes(case: &Case) -> BenchResult<Vec<u8>> {
    match case.model.as_str() {
        "models/sign.onnx.hex" => decode_hex(include_str!("../../../models/sign.onnx.hex")),
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
        schema: 1,
        revision: env::var("GITHUB_SHA").ok(),
        engine: engine.to_string(),
        backend: backend.to_string(),
        device: env::var("TYNX_BENCH_DEVICE").ok(),
        case: case.id.clone(),
        input_shape: case.input_shape.clone(),
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
        output_checksum: last_output.iter().map(|&value| f64::from(value)).sum(),
    })
}

pub fn print_report(report: &Report) -> BenchResult<()> {
    println!("{}", serde_json::to_string_pretty(report)?);
    Ok(())
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
    fn loads_embedded_case_and_model() {
        let case = load_case().unwrap();

        assert_eq!(case.id, "sign-11");
        assert_eq!(model_bytes(&case).unwrap().len(), 83);
    }

    #[test]
    fn validates_output_tolerance() {
        let case = load_case().unwrap();

        validate(
            &case,
            &[-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        assert!(validate(&case, &[-1.0; 11]).is_err());
    }
}
