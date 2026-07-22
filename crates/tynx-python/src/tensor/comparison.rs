//! Typed eager comparisons and boolean mask algebra.

use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyAny, PyBool},
};
use tynx_core::{DynBool, DynInt, DynTensor};

use super::data::TensorValue;
use crate::to_python_error;

#[derive(Debug, Clone, Copy)]
pub(super) enum Comparison {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

impl Comparison {
    fn symbol(self) -> &'static str {
        match self {
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
        }
    }

    fn float_tensors(self, left: DynTensor, right: DynTensor) -> tynx_core::Result<DynBool> {
        match self {
            Self::Equal => left.equal_broadcast(right),
            Self::NotEqual => left.equal_broadcast(right).map(DynBool::logical_not),
            Self::Less => left.less_broadcast(right),
            Self::LessEqual => left.less_equal_broadcast(right),
            Self::Greater => left.greater_broadcast(right),
            Self::GreaterEqual => left.greater_equal_broadcast(right),
        }
    }

    fn float_scalar(self, left: DynTensor, right: f64) -> DynBool {
        match self {
            Self::Equal => left.equal_scalar(right),
            Self::NotEqual => left.equal_scalar(right).logical_not(),
            Self::Less => left.less_scalar(right),
            Self::LessEqual => left.less_equal_scalar(right),
            Self::Greater => left.greater_scalar(right),
            Self::GreaterEqual => left.greater_equal_scalar(right),
        }
    }

    fn int_tensors(self, left: DynInt, right: DynInt) -> tynx_core::Result<DynBool> {
        match self {
            Self::Equal => left.equal_broadcast(right),
            Self::NotEqual => left.equal_broadcast(right).map(DynBool::logical_not),
            Self::Less => left.less_broadcast(right),
            Self::LessEqual => left.less_equal_broadcast(right),
            Self::Greater => left.greater_broadcast(right),
            Self::GreaterEqual => left.greater_equal_broadcast(right),
        }
    }

    fn int_scalar(self, left: DynInt, right: i64) -> DynBool {
        match self {
            Self::Equal => left.equal_scalar(right),
            Self::NotEqual => left.equal_scalar(right).logical_not(),
            Self::Less => left.less_scalar(right),
            Self::LessEqual => left.less_equal_scalar(right),
            Self::Greater => left.greater_scalar(right),
            Self::GreaterEqual => left.greater_equal_scalar(right),
        }
    }

    fn bool_tensors(self, left: DynBool, right: DynBool) -> PyResult<DynBool> {
        match self {
            Self::Equal => left
                .xor_broadcast(right)
                .map(DynBool::logical_not)
                .map_err(to_python_error),
            Self::NotEqual => left.xor_broadcast(right).map_err(to_python_error),
            _ => Err(PyTypeError::new_err(format!(
                "{} is not defined for bool Tensors",
                self.symbol()
            ))),
        }
    }

    fn bool_scalar(self, left: DynBool, right: bool) -> PyResult<DynBool> {
        match self {
            Self::Equal => Ok(if right { left } else { left.logical_not() }),
            Self::NotEqual => Ok(if right { left.logical_not() } else { left }),
            _ => Err(PyTypeError::new_err(format!(
                "{} is not defined for bool Tensors",
                self.symbol()
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum MaskOperation {
    And,
    Or,
    Xor,
}

impl MaskOperation {
    fn symbol(self) -> &'static str {
        match self {
            Self::And => "&",
            Self::Or => "|",
            Self::Xor => "^",
        }
    }
}

impl TensorValue {
    pub(super) fn compare_tensor(self, other: Self, comparison: Comparison) -> PyResult<Self> {
        let left_dtype = self.dtype_name();
        let right_dtype = other.dtype_name();
        let result = match (self, other) {
            (Self::Float(left), Self::Float(right)) => comparison
                .float_tensors(left, right)
                .map_err(to_python_error)?,
            (Self::Int(left), Self::Int(right)) => comparison
                .int_tensors(left, right)
                .map_err(to_python_error)?,
            (Self::Bool(left), Self::Bool(right)) => comparison.bool_tensors(left, right)?,
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "Tensor comparison {} requires matching dtypes, got {left_dtype} and {right_dtype}",
                    comparison.symbol()
                )));
            }
        };
        Ok(Self::Bool(result))
    }

    pub(super) fn compare_scalar(
        self,
        other: &Bound<'_, PyAny>,
        comparison: Comparison,
    ) -> PyResult<Self> {
        let result = match self {
            Self::Float(left) => comparison.float_scalar(
                left,
                other.extract::<f64>().map_err(|_| {
                    PyTypeError::new_err("float32 Tensor comparison expects a real scalar")
                })?,
            ),
            Self::Int(left) => {
                if other.is_instance_of::<PyBool>() {
                    return Err(PyTypeError::new_err(
                        "int64 Tensor comparison expects an integer scalar, not bool",
                    ));
                }
                comparison.int_scalar(
                    left,
                    other.extract::<i64>().map_err(|_| {
                        PyTypeError::new_err("int64 Tensor comparison expects an integer scalar")
                    })?,
                )
            }
            Self::Bool(left) => comparison.bool_scalar(
                left,
                other.extract::<bool>().map_err(|_| {
                    PyTypeError::new_err("bool Tensor comparison expects a bool scalar")
                })?,
            )?,
        };
        Ok(Self::Bool(result))
    }

    pub(super) fn mask_binary(self, other: Self, operation: MaskOperation) -> PyResult<Self> {
        let left_dtype = self.dtype_name();
        let right_dtype = other.dtype_name();
        let result = match (self, other) {
            (Self::Bool(left), Self::Bool(right)) => match operation {
                MaskOperation::And => left.and_broadcast(right),
                MaskOperation::Or => left.or_broadcast(right),
                MaskOperation::Xor => left.xor_broadcast(right),
            }
            .map_err(to_python_error)?,
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "Tensor mask operator {} requires bool Tensors, got {left_dtype} and {right_dtype}",
                    operation.symbol()
                )));
            }
        };
        Ok(Self::Bool(result))
    }

    pub(super) fn mask_not(self) -> PyResult<Self> {
        match self {
            Self::Bool(value) => Ok(Self::Bool(value.logical_not())),
            other => Err(PyTypeError::new_err(format!(
                "Tensor mask operator ~ requires a bool Tensor, got {}",
                other.dtype_name()
            ))),
        }
    }
}
