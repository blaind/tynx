//! Typed eager conditional selection.

use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyAny, PyBool},
};
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
    match template {
        TensorValue::Float(value) => scalar
            .extract::<f64>()
            .map(|scalar| TensorValue::Float(value.full_like(scalar)))
            .map_err(|_| PyTypeError::new_err("float32 where branch expects a real scalar")),
        TensorValue::Int(value) => {
            if scalar.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(
                    "int64 where branch expects an integer scalar, not bool",
                ));
            }
            scalar
                .extract::<i64>()
                .map(|scalar| TensorValue::Int(value.full_like(scalar)))
                .map_err(|_| PyTypeError::new_err("int64 where branch expects an integer scalar"))
        }
        TensorValue::Bool(value) => scalar
            .extract::<bool>()
            .map(|scalar| TensorValue::Bool(value.full_like(scalar)))
            .map_err(|_| PyTypeError::new_err("bool where branch expects a bool scalar")),
    }
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
