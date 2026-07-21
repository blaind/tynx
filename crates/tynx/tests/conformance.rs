//! ONNX backend conformance against the official cases vendored by Burn-ONNX.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use burn::tensor::{Device, TensorData};
use protobuf::Message;
use serde::{Deserialize, Serialize};
use tynx::onnx_ir::TensorProto;
use tynx::{Env, Session, Value};

const REGISTRY_JSON: &str = include_str!("conformance.json");

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct Source {
    repository: String,
    revision: String,
    path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Registry {
    schema: u32,
    source: Source,
    cases: BTreeMap<String, CaseStatus>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum CaseStatus {
    Pass,
    MissingData,
    ParseUnsupported,
    ReferenceUnsupported,
    ArityMismatch,
    RuntimeUnsupported,
    OutputUnsupported,
    Mismatch,
    Panic,
}

#[derive(Debug, Serialize)]
struct Observation {
    status: CaseStatus,
    detail: String,
}

#[derive(Serialize)]
struct Report {
    schema: u32,
    source: Source,
    summary: BTreeMap<CaseStatus, usize>,
    cases: BTreeMap<String, Observation>,
}

enum Reference {
    F32(Vec<f32>, Vec<usize>),
    F64(Vec<f64>, Vec<usize>),
    I32(Vec<i32>, Vec<usize>),
    I64(Vec<i64>, Vec<usize>),
    Bool(Vec<bool>, Vec<usize>),
}

impl Reference {
    fn shape(&self) -> &[usize] {
        match self {
            Self::F32(_, shape)
            | Self::F64(_, shape)
            | Self::I32(_, shape)
            | Self::I64(_, shape)
            | Self::Bool(_, shape) => shape,
        }
    }

    fn values(&self) -> Vec<f64> {
        match self {
            Self::F32(values, _) => values.iter().map(|&value| f64::from(value)).collect(),
            Self::F64(values, _) => values.clone(),
            Self::I32(values, _) => values.iter().map(|&value| f64::from(value)).collect(),
            Self::I64(values, _) => values.iter().map(|&value| value as f64).collect(),
            Self::Bool(values, _) => values.iter().map(|&value| u8::from(value) as f64).collect(),
        }
    }

    fn is_float(&self) -> bool {
        matches!(self, Self::F32(..) | Self::F64(..))
    }

    fn to_value(&self, device: &Device) -> Result<Value, String> {
        let shape = self.shape().to_vec();
        let rank = shape.len();
        let data = match self {
            Self::F32(values, _) => TensorData::new(values.clone(), shape),
            Self::F64(values, _) => TensorData::new(values.clone(), shape),
            Self::I32(values, _) => TensorData::new(values.clone(), shape),
            Self::I64(values, _) => TensorData::new(values.clone(), shape),
            Self::Bool(values, _) => TensorData::new(values.clone(), shape),
        };
        Value::from_tensor_data(data, rank, device).map_err(|error| error.to_string())
    }
}

#[test]
#[ignore = "requires the external Burn-ONNX conformance corpus"]
fn onnx_backend_conformance() {
    let registry: Registry =
        serde_json::from_str(REGISTRY_JSON).expect("valid conformance registry");
    assert_eq!(
        registry.schema, 1,
        "unsupported conformance registry schema"
    );
    assert!(
        include_str!("../Cargo.toml").contains(&registry.source.revision),
        "conformance revision must match the onnx-ir dependency revision"
    );

    let corpus = env::var_os("TYNX_ONNX_CORPUS")
        .map(PathBuf::from)
        .expect("TYNX_ONNX_CORPUS is not set; run `cargo xtask conformance fetch`");
    assert!(
        corpus.is_dir(),
        "ONNX corpus not found at {}",
        corpus.display()
    );

    let filter = env::var("TYNX_CONFORMANCE_CASE").ok();
    let bless = env::var_os("TYNX_CONFORMANCE_BLESS").is_some();
    assert!(!(bless && filter.is_some()), "cannot bless a filtered run");

    let mut directories = case_directories(&corpus).expect("read ONNX corpus");
    if let Some(case) = &filter {
        directories.retain(|path| path.file_name().is_some_and(|name| name == case.as_str()));
        assert_eq!(
            directories.len(),
            1,
            "conformance case '{case}' was not found"
        );
    }

    let device = Device::default();
    let previous_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let mut observations = BTreeMap::new();
    for directory in directories {
        let name = directory
            .file_name()
            .expect("case directory has a name")
            .to_string_lossy()
            .into_owned();
        let observation =
            match panic::catch_unwind(AssertUnwindSafe(|| run_case(&directory, &device))) {
                Ok(observation) => observation,
                Err(payload) => Observation {
                    status: CaseStatus::Panic,
                    detail: panic_message(&payload),
                },
            };
        observations.insert(name, observation);
    }
    panic::set_hook(previous_hook);

    let summary = summarize(&observations);
    print_summary(&summary, observations.len());
    write_report(&registry.source, &summary, &observations).expect("write conformance report");

    if bless {
        let path = env::var_os("TYNX_CONFORMANCE_REGISTRY")
            .map(PathBuf::from)
            .expect("TYNX_CONFORMANCE_REGISTRY is not set");
        let blessed = Registry {
            schema: registry.schema,
            source: registry.source,
            cases: observations
                .iter()
                .map(|(name, observation)| (name.clone(), observation.status))
                .collect(),
        };
        write_json(&path, &blessed).expect("write conformance registry");
        println!("updated {}", path.display());
        return;
    }

    let drift = compare(&registry.cases, &observations, filter.is_some());
    assert!(
        drift.is_empty(),
        "conformance registry drift (run `cargo xtask conformance bless` after reviewing):\n{}",
        drift.join("\n")
    );
}

fn case_directories(corpus: &Path) -> Result<Vec<PathBuf>, String> {
    let mut directories: Vec<_> = fs::read_dir(corpus)
        .map_err(|error| error.to_string())?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect();
    directories.sort();
    Ok(directories)
}

fn run_case(directory: &Path, device: &Device) -> Observation {
    let model = directory.join("model.onnx");
    let data = directory.join("test_data_set_0");
    if !model.is_file() || !data.is_dir() {
        return observed(
            CaseStatus::MissingData,
            "model or test_data_set_0 is missing",
        );
    }

    let session = match Session::from_file(&model) {
        Ok(session) => session,
        Err(error) => return observed(CaseStatus::ParseUnsupported, error),
    };
    let inputs = match load_series(&data, "input") {
        Ok(inputs) => inputs,
        Err(error) => return observed(CaseStatus::ReferenceUnsupported, error),
    };
    let outputs = match load_series(&data, "output") {
        Ok(outputs) => outputs,
        Err(error) => return observed(CaseStatus::ReferenceUnsupported, error),
    };
    if inputs.len() != session.inputs().len() || outputs.len() != session.outputs().len() {
        return observed(
            CaseStatus::ArityMismatch,
            format!(
                "reference has {}/{} inputs/outputs; model has {}/{}",
                inputs.len(),
                outputs.len(),
                session.inputs().len(),
                session.outputs().len()
            ),
        );
    }

    let mut env = Env::new();
    for (argument, reference) in session.inputs().iter().zip(&inputs) {
        let value = match reference.to_value(device) {
            Ok(value) => value,
            Err(error) => return observed(CaseStatus::ReferenceUnsupported, error),
        };
        env.insert(argument.name.clone(), value);
    }
    let actual = match session.run(device, env) {
        Ok(actual) => actual,
        Err(error) => return observed(CaseStatus::RuntimeUnsupported, error),
    };

    for (argument, reference) in session.outputs().iter().zip(&outputs) {
        let Some(value) = actual.get(&argument.name) else {
            return observed(
                CaseStatus::OutputUnsupported,
                format!("output '{}' is missing", argument.name),
            );
        };
        let Some((shape, values)) = value_data(value) else {
            return observed(
                CaseStatus::OutputUnsupported,
                format!("output '{}' cannot be materialized", argument.name),
            );
        };
        if shape != reference.shape() {
            return observed(
                CaseStatus::Mismatch,
                format!(
                    "output '{}' shape {shape:?} != {:?}",
                    argument.name,
                    reference.shape()
                ),
            );
        }
        let expected = reference.values();
        if values.len() != expected.len() {
            return observed(
                CaseStatus::Mismatch,
                format!(
                    "output '{}' length {} != {}",
                    argument.name,
                    values.len(),
                    expected.len()
                ),
            );
        }
        for (index, (&actual, &expected)) in values.iter().zip(&expected).enumerate() {
            let matches = if reference.is_float() {
                actual.is_nan() == expected.is_nan()
                    && (actual - expected).abs() <= 1e-3 + 1e-3 * expected.abs()
            } else {
                actual == expected
            };
            if !matches {
                return observed(
                    CaseStatus::Mismatch,
                    format!(
                        "output '{}' value {index}: {actual} != {expected}",
                        argument.name
                    ),
                );
            }
        }
    }

    observed(CaseStatus::Pass, "")
}

fn observed(status: CaseStatus, detail: impl ToString) -> Observation {
    Observation {
        status,
        detail: detail.to_string(),
    }
}

fn load_series(directory: &Path, prefix: &str) -> Result<Vec<Reference>, String> {
    let mut tensors = Vec::new();
    for index in 0.. {
        let path = directory.join(format!("{prefix}_{index}.pb"));
        if !path.exists() {
            break;
        }
        tensors.push(load_tensor(&path)?);
    }
    Ok(tensors)
}

fn load_tensor(path: &Path) -> Result<Reference, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let tensor = TensorProto::parse_from_bytes(&bytes)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let shape: Vec<usize> = tensor
        .dims
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension).map_err(|_| format!("negative tensor dimension {dimension}"))
        })
        .collect::<Result<_, _>>()?;
    let count = shape.iter().product();
    let raw = tensor.raw_data.as_ref();

    match tensor.data_type {
        1 => Ok(Reference::F32(
            if tensor.float_data.is_empty() {
                raw.chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    .collect()
            } else {
                tensor.float_data.clone()
            },
            shape,
        )),
        11 => Ok(Reference::F64(
            if tensor.double_data.is_empty() {
                raw.chunks_exact(8)
                    .map(|bytes| f64::from_le_bytes(bytes.try_into().expect("eight-byte chunk")))
                    .collect()
            } else {
                tensor.double_data.clone()
            },
            shape,
        )),
        6 => Ok(Reference::I32(
            if tensor.int32_data.is_empty() {
                raw.chunks_exact(4)
                    .map(|bytes| i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    .collect()
            } else {
                tensor.int32_data.clone()
            },
            shape,
        )),
        7 => Ok(Reference::I64(
            if tensor.int64_data.is_empty() {
                raw.chunks_exact(8)
                    .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("eight-byte chunk")))
                    .collect()
            } else {
                tensor.int64_data.clone()
            },
            shape,
        )),
        9 => Ok(Reference::Bool(
            if tensor.int32_data.is_empty() {
                raw.iter().take(count).map(|&value| value != 0).collect()
            } else {
                tensor.int32_data.iter().map(|&value| value != 0).collect()
            },
            shape,
        )),
        2 => Ok(Reference::I64(
            small_int(raw, &tensor.int32_data, count, 1, |bytes| {
                i64::from(bytes[0])
            }),
            shape,
        )),
        3 => Ok(Reference::I64(
            small_int(raw, &tensor.int32_data, count, 1, |bytes| {
                i64::from(bytes[0] as i8)
            }),
            shape,
        )),
        4 => Ok(Reference::I64(
            small_int(raw, &tensor.int32_data, count, 2, |bytes| {
                i64::from(u16::from_le_bytes([bytes[0], bytes[1]]))
            }),
            shape,
        )),
        5 => Ok(Reference::I64(
            small_int(raw, &tensor.int32_data, count, 2, |bytes| {
                i64::from(i16::from_le_bytes([bytes[0], bytes[1]]))
            }),
            shape,
        )),
        10 => Ok(Reference::F32(
            half_values(raw, &tensor.int32_data, half::f16::from_bits),
            shape,
        )),
        16 => Ok(Reference::F32(
            half_values(raw, &tensor.int32_data, half::bf16::from_bits),
            shape,
        )),
        data_type => Err(format!(
            "{} uses unsupported TensorProto data type {data_type}",
            path.display()
        )),
    }
}

fn small_int(
    raw: &[u8],
    typed: &[i32],
    count: usize,
    width: usize,
    decode: impl Fn(&[u8]) -> i64,
) -> Vec<i64> {
    if raw.is_empty() {
        typed.iter().map(|&value| i64::from(value)).collect()
    } else {
        raw.chunks_exact(width).take(count).map(decode).collect()
    }
}

fn half_values<T: Copy>(raw: &[u8], typed: &[i32], from_bits: impl Fn(u16) -> T) -> Vec<f32>
where
    f32: From<T>,
{
    if raw.is_empty() {
        typed
            .iter()
            .map(|&value| f32::from(from_bits(value as u16)))
            .collect()
    } else {
        raw.chunks_exact(2)
            .map(|bytes| f32::from(from_bits(u16::from_le_bytes([bytes[0], bytes[1]]))))
            .collect()
    }
}

fn value_data(value: &Value) -> Option<(Vec<usize>, Vec<f64>)> {
    match value {
        Value::Tensor(tensor) => {
            let shape = tensor.dims();
            let values = tensor.clone().into_data().iter::<f64>().collect();
            Some((shape, values))
        }
        Value::Int(tensor) => {
            let shape = tensor.dims();
            let values = tensor
                .clone()
                .into_data()
                .iter::<i64>()
                .map(|value| value as f64)
                .collect();
            Some((shape, values))
        }
        Value::Bool(tensor) => {
            let shape = tensor.dims();
            let values = tensor
                .clone()
                .into_data()
                .iter::<bool>()
                .map(|value| u8::from(value) as f64)
                .collect();
            Some((shape, values))
        }
        Value::Scalar(scalar) => Some((Vec::new(), vec![scalar.as_f64()])),
        Value::Shape(shape) => Some((
            vec![shape.len()],
            shape.iter().map(|&value| value as f64).collect(),
        )),
    }
}

fn summarize(observations: &BTreeMap<String, Observation>) -> BTreeMap<CaseStatus, usize> {
    let mut summary = BTreeMap::new();
    for observation in observations.values() {
        *summary.entry(observation.status).or_default() += 1;
    }
    summary
}

fn print_summary(summary: &BTreeMap<CaseStatus, usize>, total: usize) {
    println!("\n=== Tynx ONNX conformance ({total} cases) ===");
    for (status, count) in summary {
        println!("  {status:?}: {count}");
    }
}

fn compare(
    expected: &BTreeMap<String, CaseStatus>,
    actual: &BTreeMap<String, Observation>,
    filtered: bool,
) -> Vec<String> {
    let mut drift = Vec::new();
    for (name, observation) in actual {
        match expected.get(name) {
            Some(status) if status == &observation.status => {}
            Some(status) => drift.push(format!(
                "{name}: expected {status:?}, observed {:?} ({})",
                observation.status, observation.detail
            )),
            None => drift.push(format!(
                "{name}: missing from registry; observed {:?} ({})",
                observation.status, observation.detail
            )),
        }
    }
    if !filtered {
        for name in expected.keys() {
            if !actual.contains_key(name) {
                drift.push(format!(
                    "{name}: registered case is missing from the corpus"
                ));
            }
        }
    }
    drift
}

fn write_report(
    source: &Source,
    summary: &BTreeMap<CaseStatus, usize>,
    observations: &BTreeMap<String, Observation>,
) -> Result<(), String> {
    let Some(path) = env::var_os("TYNX_CONFORMANCE_REPORT").map(PathBuf::from) else {
        return Ok(());
    };
    let report = Report {
        schema: 1,
        source: source.clone(),
        summary: summary.clone(),
        cases: observations
            .iter()
            .map(|(name, observation)| {
                (
                    name.clone(),
                    Observation {
                        status: observation.status,
                        detail: observation.detail.clone(),
                    },
                )
            })
            .collect(),
    };
    write_json(&path, &report)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let mut json = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    json.push('\n');
    fs::write(path, json).map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}
