//! Sequence-based eager tensor combination operations.

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyList, PyTuple},
};
use tynx_core::{DynBool, DynInt, DynTensor};

use super::{PyTensor, data::TensorValue, shape};
use crate::{grad_mode::is_grad_enabled, to_python_error};

fn tensor_refs<'py>(
    values: &Bound<'py, PyAny>,
    operation: &str,
) -> PyResult<Vec<PyRef<'py, PyTensor>>> {
    let items = if let Ok(tuple) = values.cast::<PyTuple>() {
        tuple.iter().collect::<Vec<_>>()
    } else if let Ok(list) = values.cast::<PyList>() {
        list.iter().collect::<Vec<_>>()
    } else {
        return Err(PyTypeError::new_err(format!(
            "{operation} expects a list or tuple of Tensors"
        )));
    };
    if items.is_empty() {
        return Err(PyValueError::new_err(format!(
            "{operation} expects a non-empty tensor sequence"
        )));
    }
    items
        .into_iter()
        .map(|item| {
            item.extract::<PyRef<'py, PyTensor>>().map_err(|_| {
                PyTypeError::new_err(format!("{operation} sequence entries must be Tensors"))
            })
        })
        .collect()
}

fn combine(values: &Bound<'_, PyAny>, dim: isize, stack: bool) -> PyResult<PyTensor> {
    let operation = if stack { "stack" } else { "cat" };
    let tensors = tensor_refs(values, operation)?;
    for tensor in &tensors {
        tensor.capture_unsupported(operation)?;
    }
    let first = tensors[0].source.value();
    if tensors
        .iter()
        .skip(1)
        .any(|tensor| tensor.source.value().dtype_name() != first.dtype_name())
    {
        return Err(PyTypeError::new_err(format!(
            "{operation} requires all tensors to have the same dtype"
        )));
    }
    let axis = shape::axis_value(dim, first.rank(), stack, operation)?;
    let tracking = is_grad_enabled();
    let sources = tensors.iter().map(|tensor| &**tensor).collect::<Vec<_>>();
    match first {
        TensorValue::Float(_) => {
            let inputs = tensors
                .iter()
                .map(|tensor| tensor.operation_float_value(tracking, operation))
                .collect::<PyResult<Vec<_>>>()?;
            let output = if stack {
                DynTensor::stack(inputs, axis)
            } else {
                DynTensor::concat(inputs, axis)
            }
            .map_err(to_python_error)?;
            Ok(if tracking {
                PyTensor::from_operation(output, &sources)
            } else {
                PyTensor::from_inner(output)
            })
        }
        TensorValue::Int(_) => {
            let inputs = tensors
                .iter()
                .map(|tensor| match tensor.source.value() {
                    TensorValue::Int(value) => Ok(value),
                    other => Err(PyTypeError::new_err(format!(
                        "{operation} requires equal dtypes, got int64 and {}",
                        other.dtype_name()
                    ))),
                })
                .collect::<PyResult<Vec<_>>>()?;
            let output = if stack {
                DynInt::stack(inputs, axis)
            } else {
                DynInt::concat(inputs, axis)
            }
            .map_err(to_python_error)?;
            Ok(PyTensor::from_int_inner(output))
        }
        TensorValue::Bool(_) => {
            let inputs = tensors
                .iter()
                .map(|tensor| match tensor.source.value() {
                    TensorValue::Bool(value) => Ok(value),
                    other => Err(PyTypeError::new_err(format!(
                        "{operation} requires equal dtypes, got bool and {}",
                        other.dtype_name()
                    ))),
                })
                .collect::<PyResult<Vec<_>>>()?;
            let output = if stack {
                DynBool::stack(inputs, axis)
            } else {
                DynBool::concat(inputs, axis)
            }
            .map_err(to_python_error)?;
            Ok(PyTensor::from_value(TensorValue::Bool(output)))
        }
    }
}

#[pyfunction(name = "cat")]
#[pyo3(signature = (tensors, dim=0))]
pub(crate) fn cat_py(tensors: &Bound<'_, PyAny>, dim: isize) -> PyResult<PyTensor> {
    combine(tensors, dim, false)
}

#[pyfunction(name = "stack")]
#[pyo3(signature = (tensors, dim=0))]
pub(crate) fn stack_py(tensors: &Bound<'_, PyAny>, dim: isize) -> PyResult<PyTensor> {
    combine(tensors, dim, true)
}
