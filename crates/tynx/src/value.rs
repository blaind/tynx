//! Values that flow through a model at runtime.

use burn::tensor::{DType, Device, TensorData};

use crate::error::{Result, TynxError};
use crate::tensor::{DynBool, DynInt, DynTensor};

/// A host-side scalar value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    /// A floating-point scalar.
    F64(f64),
    /// A signed integer scalar.
    I64(i64),
    /// An unsigned integer scalar.
    U64(u64),
    /// A boolean scalar.
    Bool(bool),
}

impl Scalar {
    /// Convert the scalar to `f64` for scalar tensor operations.
    pub fn as_f64(&self) -> f64 {
        match self {
            Self::F64(value) => *value,
            Self::I64(value) => *value as f64,
            Self::U64(value) => *value as f64,
            Self::Bool(value) => u8::from(*value) as f64,
        }
    }
}

/// A value flowing through a model graph.
#[derive(Debug, Clone)]
pub enum Value {
    /// An on-device floating-point tensor.
    Tensor(DynTensor),
    /// An on-device integer tensor.
    Int(DynInt),
    /// An on-device boolean tensor.
    Bool(DynBool),
    /// A host-side scalar.
    Scalar(Scalar),
    /// A host-side shape vector.
    Shape(Vec<i64>),
}

impl Value {
    /// Materialize tensor data as the corresponding runtime value.
    pub fn from_tensor_data(data: TensorData, rank: usize, device: &Device) -> Result<Self> {
        let dtype = data.dtype;

        if rank == 0 {
            return Ok(Self::Scalar(scalar_from_tensor_data(&data)?));
        }

        Ok(match dtype {
            DType::Bool(_) => Self::Bool(DynBool::from_data(data, rank, device)?),
            dtype if is_integer(dtype) => Self::Int(DynInt::from_data(data, rank, device)?),
            _ => Self::Tensor(DynTensor::from_data(data, rank, device)?),
        })
    }

    /// Borrow the value as a floating-point tensor.
    pub fn as_tensor(&self) -> Result<&DynTensor> {
        match self {
            Self::Tensor(tensor) => Ok(tensor),
            other => Err(other.type_mismatch("Tensor")),
        }
    }

    /// Consume the value as a floating-point tensor.
    pub fn into_tensor(self) -> Result<DynTensor> {
        match self {
            Self::Tensor(tensor) => Ok(tensor),
            other => Err(other.type_mismatch("Tensor")),
        }
    }

    /// Consume the value as an integer tensor.
    pub fn into_int(self) -> Result<DynInt> {
        match self {
            Self::Int(tensor) => Ok(tensor),
            other => Err(other.type_mismatch("Int tensor")),
        }
    }

    /// Consume the value as a boolean tensor.
    pub fn into_bool(self) -> Result<DynBool> {
        match self {
            Self::Bool(tensor) => Ok(tensor),
            other => Err(other.type_mismatch("Bool tensor")),
        }
    }

    fn type_mismatch(&self, expected: &str) -> TynxError {
        TynxError::TypeMismatch(format!("expected {expected}, got {}", self.kind()))
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Tensor(_) => "Tensor",
            Self::Int(_) => "Int tensor",
            Self::Bool(_) => "Bool tensor",
            Self::Scalar(_) => "Scalar",
            Self::Shape(_) => "Shape",
        }
    }
}

fn is_integer(dtype: DType) -> bool {
    dtype.is_int() || dtype.is_uint()
}

fn scalar_from_tensor_data(data: &TensorData) -> Result<Scalar> {
    let empty = || TynxError::Shape("empty rank-0 tensor".to_string());

    match data.dtype {
        DType::Bool(_) => data
            .iter::<bool>()
            .next()
            .map(Scalar::Bool)
            .ok_or_else(empty),
        dtype if dtype.is_int() => data.iter::<i64>().next().map(Scalar::I64).ok_or_else(empty),
        dtype if dtype.is_uint() => data.iter::<u64>().next().map(Scalar::U64).ok_or_else(empty),
        _ => data.iter::<f64>().next().map(Scalar::F64).ok_or_else(empty),
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Device;

    use super::*;

    #[test]
    fn converts_scalars_to_f64() {
        assert_eq!(Scalar::F64(1.5).as_f64(), 1.5);
        assert_eq!(Scalar::I64(2).as_f64(), 2.0);
        assert_eq!(Scalar::U64(3).as_f64(), 3.0);
        assert_eq!(Scalar::Bool(true).as_f64(), 1.0);
        assert_eq!(Scalar::Bool(false).as_f64(), 0.0);
    }

    #[test]
    fn reports_value_type_mismatches() {
        let error = Value::Shape(vec![2, 3]).into_tensor().unwrap_err();

        assert_eq!(
            error,
            TynxError::TypeMismatch("expected Tensor, got Shape".to_string())
        );
    }

    #[test]
    fn materializes_data_by_kind() {
        let device = Device::default();

        let float =
            Value::from_tensor_data(TensorData::new(vec![1.0_f32, 2.0], [2]), 1, &device).unwrap();
        let integer =
            Value::from_tensor_data(TensorData::new(vec![1_i64, 2], [2]), 1, &device).unwrap();
        let boolean =
            Value::from_tensor_data(TensorData::new(vec![true, false], [2]), 1, &device).unwrap();

        assert!(matches!(float, Value::Tensor(DynTensor::R1(_))));
        assert!(matches!(integer, Value::Int(DynInt::R1(_))));
        assert!(matches!(boolean, Value::Bool(DynBool::R1(_))));
    }

    #[test]
    fn materializes_rank_zero_booleans() {
        let value = Value::from_tensor_data(
            TensorData::new(vec![true], Vec::<usize>::new()),
            0,
            &Device::default(),
        )
        .unwrap();

        assert!(matches!(value, Value::Scalar(Scalar::Bool(true))));
    }

    #[test]
    fn materializes_integer_scalars_without_losing_precision() {
        let device = Device::default();
        let signed = Value::from_tensor_data(
            TensorData::new(vec![i64::MAX], Vec::<usize>::new()),
            0,
            &device,
        )
        .unwrap();
        let unsigned = Value::from_tensor_data(
            TensorData::new(vec![u64::MAX], Vec::<usize>::new()),
            0,
            &device,
        )
        .unwrap();

        assert!(matches!(signed, Value::Scalar(Scalar::I64(i64::MAX))));
        assert!(matches!(unsigned, Value::Scalar(Scalar::U64(u64::MAX))));
    }
}
