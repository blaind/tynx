//! Typed eager conditional selection.

use pyo3::{exceptions::PyTypeError, prelude::*};
use tynx_core::{DynBool, DynInt, DynTensor};

use super::data::TensorValue;
use crate::to_python_error;

pub(super) fn condition(value: TensorValue) -> PyResult<DynBool> {
    match value {
        TensorValue::Bool(value) => Ok(value),
        other => Err(PyTypeError::new_err(format!(
            "where condition must be a bool Tensor, got {}",
            other.dtype_name()
        ))),
    }
}

pub(super) fn scalar_like(
    template: TensorValue,
    scalar: &Bound<'_, PyAny>,
) -> PyResult<TensorValue> {
    template.scalar_like(scalar, "where branch")
}

pub(super) fn select(
    condition: DynBool,
    then: TensorValue,
    otherwise: TensorValue,
) -> PyResult<TensorValue> {
    let then_dtype = then.dtype_name();
    let otherwise_dtype = otherwise.dtype_name();
    match (then, otherwise) {
        (TensorValue::Float(then), TensorValue::Float(otherwise)) => {
            DynTensor::where_select(condition, then, otherwise)
                .map(TensorValue::Float)
                .map_err(to_python_error)
        }
        (TensorValue::Int(then), TensorValue::Int(otherwise)) => {
            DynInt::where_select(condition, then, otherwise)
                .map(TensorValue::Int)
                .map_err(to_python_error)
        }
        (TensorValue::Bool(then), TensorValue::Bool(otherwise)) => {
            DynBool::where_select(condition, then, otherwise)
                .map(TensorValue::Bool)
                .map_err(to_python_error)
        }
        _ => Err(PyTypeError::new_err(format!(
            "where branches must have matching dtypes, got {then_dtype} and {otherwise_dtype}"
        ))),
    }
}
