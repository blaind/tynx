//! ONNX backend conformance against the official cases vendored by Burn-ONNX.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use burn::tensor::{BoolStore, DType, Device, TensorData};
use half::{bf16, f16};
use protobuf::Message;
use serde::{Deserialize, Serialize};
use tynx::onnx_ir::{Node, TensorProto};
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
    F16(Vec<f16>, Vec<usize>),
    BF16(Vec<bf16>, Vec<usize>),
    F32(Vec<f32>, Vec<usize>),
    F64(Vec<f64>, Vec<usize>),
    I8(Vec<i8>, Vec<usize>),
    I16(Vec<i16>, Vec<usize>),
    I32(Vec<i32>, Vec<usize>),
    I64(Vec<i64>, Vec<usize>),
    U8(Vec<u8>, Vec<usize>),
    U16(Vec<u16>, Vec<usize>),
    U32(Vec<u32>, Vec<usize>),
    U64(Vec<u64>, Vec<usize>),
    Bool(Vec<bool>, Vec<usize>),
}

impl Reference {
    fn shape(&self) -> &[usize] {
        match self {
            Self::F16(_, shape)
            | Self::BF16(_, shape)
            | Self::F32(_, shape)
            | Self::F64(_, shape)
            | Self::I8(_, shape)
            | Self::I16(_, shape)
            | Self::I32(_, shape)
            | Self::I64(_, shape)
            | Self::U8(_, shape)
            | Self::U16(_, shape)
            | Self::U32(_, shape)
            | Self::U64(_, shape)
            | Self::Bool(_, shape) => shape,
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::F16(..) => DType::F16,
            Self::BF16(..) => DType::BF16,
            Self::F32(..) => DType::F32,
            Self::F64(..) => DType::F64,
            Self::I8(..) => DType::I8,
            Self::I16(..) => DType::I16,
            Self::I32(..) => DType::I32,
            Self::I64(..) => DType::I64,
            Self::U8(..) => DType::U8,
            Self::U16(..) => DType::U16,
            Self::U32(..) => DType::U32,
            Self::U64(..) => DType::U64,
            Self::Bool(..) => DType::Bool(BoolStore::Native),
        }
    }

    fn to_value(&self, device: &Device) -> Result<Value, String> {
        let shape = self.shape().to_vec();
        let rank = shape.len();
        let data = match self {
            Self::F16(values, _) => TensorData::new(values.clone(), shape),
            Self::BF16(values, _) => TensorData::new(values.clone(), shape),
            Self::F32(values, _) => TensorData::new(values.clone(), shape),
            Self::F64(values, _) => TensorData::new(values.clone(), shape),
            Self::I8(values, _) => TensorData::new(values.clone(), shape),
            Self::I16(values, _) => TensorData::new(values.clone(), shape),
            Self::I32(values, _) => TensorData::new(values.clone(), shape),
            Self::I64(values, _) => TensorData::new(values.clone(), shape),
            Self::U8(values, _) => TensorData::new(values.clone(), shape),
            Self::U16(values, _) => TensorData::new(values.clone(), shape),
            Self::U32(values, _) => TensorData::new(values.clone(), shape),
            Self::U64(values, _) => TensorData::new(values.clone(), shape),
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

    let stochastic = session
        .graph()
        .nodes
        .iter()
        .any(|node| matches!(node, Node::Bernoulli(_) | Node::RandomUniformLike(_)));
    for (argument, reference) in session.outputs().iter().zip(&outputs) {
        let Some(value) = actual.get(&argument.name) else {
            return observed(
                CaseStatus::OutputUnsupported,
                format!("output '{}' is missing", argument.name),
            );
        };
        let comparison = if stochastic {
            compare_binary_random_value(reference, value)
        } else {
            compare_value(reference, value)
        };
        if let Err(error) = comparison {
            return observed(
                CaseStatus::Mismatch,
                format!("output '{}': {error}", argument.name),
            );
        }
    }

    observed(CaseStatus::Pass, "")
}

fn compare_binary_random_value(reference: &Reference, value: &Value) -> Result<(), String> {
    let actual_shape = value_shape(value);
    if actual_shape != reference.shape() {
        return Err(format!("shape {actual_shape:?} != {:?}", reference.shape()));
    }

    let binary = match reference {
        Reference::F16(..) | Reference::BF16(..) | Reference::F32(..) | Reference::F64(..) => {
            float_values(value, reference.dtype())?
                .into_iter()
                .all(|value| value == 0.0 || value == 1.0)
        }
        Reference::I8(..) | Reference::I16(..) | Reference::I32(..) | Reference::I64(..) => {
            signed_values(value, reference.dtype())?
                .into_iter()
                .all(|value| value == 0 || value == 1)
        }
        Reference::U8(..) | Reference::U16(..) | Reference::U32(..) | Reference::U64(..) => {
            unsigned_values(value, reference.dtype())?
                .into_iter()
                .all(|value| value == 0 || value == 1)
        }
        Reference::Bool(..) => {
            bool_values(value, reference.dtype())?;
            true
        }
    };
    if binary {
        Ok(())
    } else {
        Err("stochastic Bernoulli output contains a value other than 0 or 1".to_string())
    }
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
        2 => Ok(Reference::U8(
            small_int(
                raw,
                &tensor.int32_data,
                count,
                1,
                |bytes| bytes[0],
                |value| value as u8,
            ),
            shape,
        )),
        3 => Ok(Reference::I8(
            small_int(
                raw,
                &tensor.int32_data,
                count,
                1,
                |bytes| bytes[0] as i8,
                |value| value as i8,
            ),
            shape,
        )),
        4 => Ok(Reference::U16(
            small_int(
                raw,
                &tensor.int32_data,
                count,
                2,
                |bytes| u16::from_le_bytes([bytes[0], bytes[1]]),
                |value| value as u16,
            ),
            shape,
        )),
        5 => Ok(Reference::I16(
            small_int(
                raw,
                &tensor.int32_data,
                count,
                2,
                |bytes| i16::from_le_bytes([bytes[0], bytes[1]]),
                |value| value as i16,
            ),
            shape,
        )),
        10 => Ok(Reference::F16(
            half_values(raw, &tensor.int32_data, f16::from_bits),
            shape,
        )),
        12 => Ok(Reference::U32(
            if tensor.uint64_data.is_empty() {
                raw.chunks_exact(4)
                    .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    .collect()
            } else {
                tensor
                    .uint64_data
                    .iter()
                    .map(|&value| value as u32)
                    .collect()
            },
            shape,
        )),
        13 => Ok(Reference::U64(
            if tensor.uint64_data.is_empty() {
                raw.chunks_exact(8)
                    .map(|bytes| u64::from_le_bytes(bytes.try_into().expect("eight-byte chunk")))
                    .collect()
            } else {
                tensor.uint64_data.clone()
            },
            shape,
        )),
        16 => Ok(Reference::BF16(
            half_values(raw, &tensor.int32_data, bf16::from_bits),
            shape,
        )),
        data_type => Err(format!(
            "{} uses unsupported TensorProto data type {data_type}",
            path.display()
        )),
    }
}

fn small_int<T>(
    raw: &[u8],
    typed: &[i32],
    count: usize,
    width: usize,
    decode_raw: impl Fn(&[u8]) -> T,
    decode_typed: impl Fn(i32) -> T,
) -> Vec<T> {
    if raw.is_empty() {
        typed.iter().copied().map(decode_typed).collect()
    } else {
        raw.chunks_exact(width)
            .take(count)
            .map(decode_raw)
            .collect()
    }
}

fn half_values<T>(raw: &[u8], typed: &[i32], from_bits: impl Fn(u16) -> T) -> Vec<T> {
    if raw.is_empty() {
        typed.iter().map(|&value| from_bits(value as u16)).collect()
    } else {
        raw.chunks_exact(2)
            .map(|bytes| from_bits(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect()
    }
}

fn compare_value(reference: &Reference, value: &Value) -> Result<(), String> {
    let actual_shape = value_shape(value);
    if actual_shape != reference.shape() {
        return Err(format!("shape {actual_shape:?} != {:?}", reference.shape()));
    }

    match reference {
        Reference::F16(expected, _) => compare_float(
            &float_values(value, reference.dtype())?,
            &expected
                .iter()
                .map(|value| value.to_f64())
                .collect::<Vec<_>>(),
        ),
        Reference::BF16(expected, _) => compare_float(
            &float_values(value, reference.dtype())?,
            &expected
                .iter()
                .map(|value| value.to_f64())
                .collect::<Vec<_>>(),
        ),
        Reference::F32(expected, _) => compare_float(
            &float_values(value, reference.dtype())?,
            &expected
                .iter()
                .map(|&value| f64::from(value))
                .collect::<Vec<_>>(),
        ),
        Reference::F64(expected, _) => {
            compare_float(&float_values(value, reference.dtype())?, expected)
        }
        Reference::I8(expected, _) => compare_exact(
            &signed_values(value, reference.dtype())?,
            &expected
                .iter()
                .map(|&value| i64::from(value))
                .collect::<Vec<_>>(),
        ),
        Reference::I16(expected, _) => compare_exact(
            &signed_values(value, reference.dtype())?,
            &expected
                .iter()
                .map(|&value| i64::from(value))
                .collect::<Vec<_>>(),
        ),
        Reference::I32(expected, _) => compare_exact(
            &signed_values(value, reference.dtype())?,
            &expected
                .iter()
                .map(|&value| i64::from(value))
                .collect::<Vec<_>>(),
        ),
        Reference::I64(expected, _) => {
            compare_exact(&signed_values(value, reference.dtype())?, expected)
        }
        Reference::U8(expected, _) => compare_exact(
            &unsigned_values(value, reference.dtype())?,
            &expected
                .iter()
                .map(|&value| u64::from(value))
                .collect::<Vec<_>>(),
        ),
        Reference::U16(expected, _) => compare_exact(
            &unsigned_values(value, reference.dtype())?,
            &expected
                .iter()
                .map(|&value| u64::from(value))
                .collect::<Vec<_>>(),
        ),
        Reference::U32(expected, _) => compare_exact(
            &unsigned_values(value, reference.dtype())?,
            &expected
                .iter()
                .map(|&value| u64::from(value))
                .collect::<Vec<_>>(),
        ),
        Reference::U64(expected, _) => {
            compare_exact(&unsigned_values(value, reference.dtype())?, expected)
        }
        Reference::Bool(expected, _) => {
            compare_exact(&bool_values(value, reference.dtype())?, expected)
        }
    }
}

fn value_shape(value: &Value) -> Vec<usize> {
    match value {
        Value::Tensor(tensor) => tensor.dims(),
        Value::Int(tensor) => tensor.dims(),
        Value::Bool(tensor) => tensor.dims(),
        Value::Scalar(_) => Vec::new(),
        Value::Shape(shape) => vec![shape.len()],
    }
}

fn float_values(value: &Value, expected_dtype: DType) -> Result<Vec<f64>, String> {
    match value {
        Value::Tensor(tensor) => {
            let data = tensor.clone().into_data();
            require_dtype(data.dtype, expected_dtype)?;
            Ok(data.iter::<f64>().collect())
        }
        Value::Scalar(tynx::Scalar::F64(value)) => Ok(vec![*value]),
        _ => Err("expected a floating-point value".to_string()),
    }
}

fn signed_values(value: &Value, expected_dtype: DType) -> Result<Vec<i64>, String> {
    match value {
        Value::Int(tensor) => {
            let data = tensor.clone().into_data();
            require_dtype(data.dtype, expected_dtype)?;
            Ok(data.iter::<i64>().collect())
        }
        Value::Scalar(tynx::Scalar::I64(value)) => Ok(vec![*value]),
        Value::Shape(shape) if expected_dtype == DType::I64 => Ok(shape.clone()),
        _ => Err("expected a signed integer value".to_string()),
    }
}

fn unsigned_values(value: &Value, expected_dtype: DType) -> Result<Vec<u64>, String> {
    match value {
        Value::Int(tensor) => {
            let data = tensor.clone().into_data();
            require_dtype(data.dtype, expected_dtype)?;
            Ok(data.iter::<u64>().collect())
        }
        Value::Scalar(tynx::Scalar::U64(value)) => Ok(vec![*value]),
        _ => Err("expected an unsigned integer value".to_string()),
    }
}

fn bool_values(value: &Value, expected_dtype: DType) -> Result<Vec<bool>, String> {
    match value {
        Value::Bool(tensor) => {
            let data = tensor.clone().into_data();
            require_dtype(data.dtype, expected_dtype)?;
            Ok(data.iter::<bool>().collect())
        }
        Value::Scalar(tynx::Scalar::Bool(value)) => Ok(vec![*value]),
        _ => Err("expected a boolean value".to_string()),
    }
}

fn require_dtype(actual: DType, expected: DType) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("dtype {actual:?} != {expected:?}"))
    }
}

fn compare_exact<T: PartialEq + std::fmt::Display>(
    actual: &[T],
    expected: &[T],
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("length {} != {}", actual.len(), expected.len()));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if actual != expected {
            return Err(format!("value {index}: {actual} != {expected}"));
        }
    }
    Ok(())
}

fn compare_float(actual: &[f64], expected: &[f64]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("length {} != {}", actual.len(), expected.len()));
    }
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if !float_matches(actual, expected) {
            return Err(format!("value {index}: {actual} != {expected}"));
        }
    }
    Ok(())
}

fn float_matches(actual: f64, expected: f64) -> bool {
    if expected.is_nan() {
        actual.is_nan()
    } else if expected.is_infinite() {
        actual == expected
    } else {
        actual.is_finite() && (actual - expected).abs() <= 1e-3 + 1e-3 * expected.abs()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compares_large_integers_exactly() {
        let device = Device::default();
        let reference = Reference::I64(vec![9_007_199_254_740_993], vec![1]);
        let actual = Value::from_tensor_data(
            TensorData::new(vec![9_007_199_254_740_992_i64], [1]),
            1,
            &device,
        )
        .unwrap();

        assert!(compare_value(&reference, &actual).is_err());
    }

    #[test]
    fn preserves_integer_input_dtypes() {
        let device = Device::default();
        let reference = Reference::U8(vec![255], vec![1]);
        let actual = reference.to_value(&device).unwrap();
        let widened =
            Value::from_tensor_data(TensorData::new(vec![255_i64], [1]), 1, &device).unwrap();

        assert!(compare_value(&reference, &actual).is_ok());
        assert!(compare_value(&reference, &widened).is_err());
    }

    #[test]
    fn preserves_half_input_dtypes() {
        let device = Device::default();
        let reference = Reference::F16(vec![f16::from_f32(1.5)], vec![1]);
        let actual = reference.to_value(&device).unwrap();
        let widened =
            Value::from_tensor_data(TensorData::new(vec![1.5_f32], [1]), 1, &device).unwrap();

        assert!(compare_value(&reference, &actual).is_ok());
        assert!(compare_value(&reference, &widened).is_err());
    }

    #[test]
    fn compares_non_finite_floats() {
        assert!(float_matches(f64::NAN, f64::NAN));
        assert!(float_matches(f64::INFINITY, f64::INFINITY));
        assert!(float_matches(f64::NEG_INFINITY, f64::NEG_INFINITY));
        assert!(!float_matches(f64::INFINITY, f64::NEG_INFINITY));
        assert!(!float_matches(0.0, f64::NAN));
    }

    #[test]
    fn validates_stochastic_binary_outputs_without_requiring_the_same_draw() {
        let device = Device::default();
        let reference = Reference::F32(vec![0.0, 1.0], vec![2]);
        let different_draw =
            Value::from_tensor_data(TensorData::new(vec![1.0_f32, 0.0], [2]), 1, &device).unwrap();
        let invalid =
            Value::from_tensor_data(TensorData::new(vec![0.5_f32, 0.0], [2]), 1, &device).unwrap();

        assert!(compare_binary_random_value(&reference, &different_draw).is_ok());
        assert!(compare_binary_random_value(&reference, &invalid).is_err());
    }
}
