//! Typed eager indexing validation and dispatch.

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};
use tynx_core::DynInt;

use super::data::TensorValue;
use crate::to_python_error;

pub(super) fn gather_indices(
    value: TensorValue,
    data_shape: &[usize],
    dim: usize,
) -> PyResult<DynInt> {
    let indices = match value {
        TensorValue::Int(value) => value,
        other => {
            return Err(PyTypeError::new_err(format!(
                "gather index must be an int64 Tensor, got {}",
                other.dtype_name()
            )));
        }
    };
    let index_shape = indices.dims();
    if index_shape.len() != data_shape.len() {
        return Err(PyValueError::new_err(format!(
            "gather index rank {} must match input rank {}",
            index_shape.len(),
            data_shape.len()
        )));
    }
    for (axis, (&index_size, &data_size)) in index_shape.iter().zip(data_shape).enumerate() {
        if axis != dim && index_size > data_size {
            return Err(PyValueError::new_err(format!(
                "gather index size {index_size} exceeds input size {data_size} at dimension {axis}"
            )));
        }
    }

    Ok(indices)
}

pub(super) fn gather(value: TensorValue, dim: usize, indices: DynInt) -> PyResult<TensorValue> {
    match value {
        TensorValue::Float(value) => value
            .gather(dim, indices)
            .map(TensorValue::Float)
            .map_err(to_python_error),
        TensorValue::Int(value) => value
            .gather(dim, indices)
            .map(TensorValue::Int)
            .map_err(to_python_error),
        TensorValue::Bool(value) => value
            .gather(dim, indices)
            .map(TensorValue::Bool)
            .map_err(to_python_error),
    }
}
