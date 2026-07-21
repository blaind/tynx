//! Values that flow through a model at runtime.

use crate::error::{Result, TynxError};
use crate::tensor::{DynBool, DynInt, DynTensor};

/// A host-side scalar value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    /// A floating-point scalar.
    F64(f64),
    /// An integer scalar.
    I64(i64),
    /// A boolean scalar.
    Bool(bool),
}

impl Scalar {
    /// Convert the scalar to `f64` for scalar tensor operations.
    pub fn as_f64(&self) -> f64 {
        match self {
            Self::F64(value) => *value,
            Self::I64(value) => *value as f64,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_scalars_to_f64() {
        assert_eq!(Scalar::F64(1.5).as_f64(), 1.5);
        assert_eq!(Scalar::I64(2).as_f64(), 2.0);
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
}
