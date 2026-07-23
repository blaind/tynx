//! Sequence-based eager tensor combination operations.

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBool, PyList, PyTuple},
};
use tynx_core::{DynBool, DynInt, DynTensor};

use super::{PyTensor, data::TensorValue, shape};
use crate::{
    capture::{record_combine, record_split},
    grad_mode::is_grad_enabled,
    to_python_error,
};

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
    let result = match first {
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
            if tracking {
                PyTensor::from_operation(output, &sources)
            } else {
                PyTensor::from_inner(output)
            }
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
            PyTensor::from_int_inner(output)
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
            PyTensor::from_value(TensorValue::Bool(output))
        }
    };
    let trace = record_combine(&sources, axis, stack)?;
    Ok(match trace {
        Some(trace) => result.with_trace(trace),
        None => result,
    })
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

fn split_sizes(spec: &Bound<'_, PyAny>, extent: usize) -> PyResult<Vec<usize>> {
    if !spec.is_instance_of::<PyBool>()
        && let Ok(size) = spec.extract::<usize>()
    {
        if size == 0 {
            return Err(PyValueError::new_err("split size must be positive"));
        }
        if extent == 0 {
            return Ok(vec![0]);
        }
        let mut sizes = vec![size; extent / size];
        let remainder = extent % size;
        if remainder != 0 {
            sizes.push(remainder);
        }
        return Ok(sizes);
    }
    let items = if let Ok(tuple) = spec.cast::<PyTuple>() {
        tuple.iter().collect::<Vec<_>>()
    } else if let Ok(list) = spec.cast::<PyList>() {
        list.iter().collect::<Vec<_>>()
    } else {
        return Err(PyTypeError::new_err(
            "split_size_or_sections must be a positive integer or sequence of integers",
        ));
    };
    items
        .into_iter()
        .map(|item| {
            if item.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(
                    "split section sizes must be integers, not bool",
                ));
            }
            item.extract::<usize>().map_err(|_| {
                PyTypeError::new_err("split section sizes must be non-negative integers")
            })
        })
        .collect()
}

pub(super) fn split_outputs(
    input: &PyTensor,
    split_size_or_sections: &Bound<'_, PyAny>,
    dim: isize,
) -> PyResult<Vec<PyTensor>> {
    let value = input.source.value();
    let axis = shape::axis_value(dim, value.rank(), false, "split")?;
    let sizes = split_sizes(split_size_or_sections, value.dims()[axis])?;
    let tracking = is_grad_enabled();
    let outputs: Vec<PyTensor> = match value {
        TensorValue::Float(_) => input
            .operation_float_value(tracking, "split")?
            .split(&sizes, axis)
            .map_err(to_python_error)
            .map(|outputs| {
                outputs
                    .into_iter()
                    .map(|output| {
                        if tracking {
                            PyTensor::from_operation(output, &[input])
                        } else {
                            PyTensor::from_inner(output)
                        }
                    })
                    .collect()
            })?,
        TensorValue::Int(value) => value
            .split(&sizes, axis)
            .map_err(to_python_error)
            .map(|outputs| outputs.into_iter().map(PyTensor::from_int_inner).collect())?,
        TensorValue::Bool(value) => {
            value
                .split(&sizes, axis)
                .map_err(to_python_error)
                .map(|outputs| {
                    outputs
                        .into_iter()
                        .map(|value| PyTensor::from_value(TensorValue::Bool(value)))
                        .collect()
                })?
        }
    };
    attach_split_traces(input, outputs, sizes, axis)
}

pub(super) fn chunk_outputs(
    input: &PyTensor,
    chunks: usize,
    dim: isize,
) -> PyResult<Vec<PyTensor>> {
    let value = input.source.value();
    let axis = shape::axis_value(dim, value.rank(), false, "chunk")?;
    let tracking = is_grad_enabled();
    let outputs: Vec<PyTensor> = match value {
        TensorValue::Float(_) => input
            .operation_float_value(tracking, "chunk")?
            .chunk(chunks, axis)
            .map_err(to_python_error)
            .map(|outputs| {
                outputs
                    .into_iter()
                    .map(|output| {
                        if tracking {
                            PyTensor::from_operation(output, &[input])
                        } else {
                            PyTensor::from_inner(output)
                        }
                    })
                    .collect()
            })?,
        TensorValue::Int(value) => value
            .chunk(chunks, axis)
            .map_err(to_python_error)
            .map(|outputs| outputs.into_iter().map(PyTensor::from_int_inner).collect())?,
        TensorValue::Bool(value) => {
            value
                .chunk(chunks, axis)
                .map_err(to_python_error)
                .map(|outputs| {
                    outputs
                        .into_iter()
                        .map(|value| PyTensor::from_value(TensorValue::Bool(value)))
                        .collect()
                })?
        }
    };
    let sizes = outputs
        .iter()
        .map(|output| output.source.value().dims()[axis])
        .collect();
    attach_split_traces(input, outputs, sizes, axis)
}

fn attach_split_traces(
    input: &PyTensor,
    mut outputs: Vec<PyTensor>,
    sizes: Vec<usize>,
    axis: usize,
) -> PyResult<Vec<PyTensor>> {
    let Some(traces) = record_split(input, sizes, axis)? else {
        return Ok(outputs);
    };
    debug_assert_eq!(outputs.len(), traces.len());
    for (output, trace) in outputs.iter_mut().zip(traces) {
        *output = output.with_trace(trace);
    }
    Ok(outputs)
}

fn output_tuple(py: Python<'_>, outputs: Vec<PyTensor>) -> PyResult<Py<PyTuple>> {
    let outputs = outputs
        .into_iter()
        .map(|output| Py::new(py, output))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, outputs)?.unbind())
}

#[pyfunction(name = "split")]
#[pyo3(signature = (input, split_size_or_sections, dim=0))]
pub(crate) fn split_py(
    py: Python<'_>,
    input: PyRef<'_, PyTensor>,
    split_size_or_sections: &Bound<'_, PyAny>,
    dim: isize,
) -> PyResult<Py<PyTuple>> {
    output_tuple(py, split_outputs(&input, split_size_or_sections, dim)?)
}

#[pyfunction(name = "chunk")]
#[pyo3(signature = (input, chunks, dim=0))]
pub(crate) fn chunk_py(
    py: Python<'_>,
    input: PyRef<'_, PyTensor>,
    chunks: usize,
    dim: isize,
) -> PyResult<Py<PyTuple>> {
    output_tuple(py, chunk_outputs(&input, chunks, dim)?)
}
