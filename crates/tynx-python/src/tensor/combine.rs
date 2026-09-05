//! Sequence-based eager tensor combination operations.

use pyo3::{
    exceptions::{PyIndexError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBool, PyList, PyTuple},
};
use tynx_core::{DynBool, DynInt, DynTensor, MAX_RANK, Slice};

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
            let tensor = item.extract::<PyRef<'py, PyTensor>>().map_err(|_| {
                PyTypeError::new_err(format!("{operation} sequence entries must be Tensors"))
            })?;
            tensor.require_owner_thread()?;
            Ok(tensor)
        })
        .collect()
}

fn combine(values: &Bound<'_, PyAny>, dim: isize, stack: bool) -> PyResult<PyTensor> {
    let operation = if stack { "stack" } else { "cat" };
    let tensors = tensor_refs(values, operation)?;
    let tensors = tensors.iter().map(|tensor| &**tensor).collect::<Vec<_>>();
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
    combine_tensors(&tensors, axis, stack, operation)
}

fn combine_tensors(
    tensors: &[&PyTensor],
    axis: usize,
    stack: bool,
    operation: &str,
) -> PyResult<PyTensor> {
    let first = tensors[0].source.value();
    let tracking = is_grad_enabled();
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
                PyTensor::from_operation(output, tensors)
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
    let trace = record_combine(tensors, axis, stack)?;
    match trace {
        Some(trace) => result.with_trace(trace),
        None => Ok(result),
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

#[pyfunction(name = "meshgrid")]
#[pyo3(signature = (*tensors, indexing=None))]
pub(crate) fn meshgrid_py(
    py: Python<'_>,
    tensors: &Bound<'_, PyTuple>,
    indexing: Option<&str>,
) -> PyResult<Py<PyTuple>> {
    let inputs = if tensors.len() == 1 {
        let candidate = tensors.get_item(0)?;
        if candidate.is_instance_of::<PyTuple>() || candidate.is_instance_of::<PyList>() {
            tensor_refs(&candidate, "meshgrid inputs")?
        } else {
            tensor_refs(tensors.as_any(), "meshgrid inputs")?
        }
    } else {
        tensor_refs(tensors.as_any(), "meshgrid inputs")?
    };
    if inputs.len() > MAX_RANK {
        return Err(PyValueError::new_err(format!(
            "meshgrid input count {} exceeds the maximum rank {MAX_RANK}",
            inputs.len()
        )));
    }
    if inputs.iter().any(|input| input.source.value().rank() != 1) {
        return Err(PyValueError::new_err(
            "meshgrid expects every input Tensor to be rank-1",
        ));
    }

    let first = inputs[0].source.value();
    if inputs
        .iter()
        .skip(1)
        .any(|input| input.source.value().dtype_name() != first.dtype_name())
    {
        return Err(PyTypeError::new_err(
            "meshgrid expects all inputs to have the same dtype",
        ));
    }
    if inputs
        .iter()
        .skip(1)
        .any(|input| input.source.value().device() != first.device())
    {
        return Err(PyValueError::new_err(
            "meshgrid expects all inputs to be on the same device",
        ));
    }

    let indexing = indexing.unwrap_or("ij");
    if !matches!(indexing, "ij" | "xy") {
        return Err(PyValueError::new_err(format!(
            "meshgrid indexing must be 'ij' or 'xy', got {indexing:?}"
        )));
    }

    let rank = inputs.len();
    let mut output_shape = inputs
        .iter()
        .map(|input| input.source.value().dims()[0])
        .collect::<Vec<_>>();
    if indexing == "xy" && rank >= 2 {
        output_shape.swap(0, 1);
    }

    let outputs = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let axis = if indexing == "xy" && index < 2 {
                1 - index
            } else {
                index
            };
            let mut view_shape = vec![1; rank];
            view_shape[axis] = input.source.value().dims()[0];
            input
                .reshape_value(view_shape)?
                .expand_value(output_shape.clone())
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, outputs)?.unbind())
}

fn roll_arguments(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i64>> {
    if !value.is_instance_of::<PyBool>()
        && let Ok(value) = value.extract::<i64>()
    {
        return Ok(vec![value]);
    }
    let items = if let Ok(tuple) = value.cast::<PyTuple>() {
        tuple.iter().collect::<Vec<_>>()
    } else if let Ok(list) = value.cast::<PyList>() {
        list.iter().collect::<Vec<_>>()
    } else {
        return Err(PyTypeError::new_err(format!(
            "roll {name} must be an int or a sequence of ints"
        )));
    };
    items
        .into_iter()
        .map(|item| {
            if item.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(format!(
                    "roll {name} must be an int or a sequence of ints"
                )));
            }
            item.extract::<i64>().map_err(|_| {
                PyTypeError::new_err(format!("roll {name} must be an int or a sequence of ints"))
            })
        })
        .collect()
}

#[pyfunction(name = "roll")]
#[pyo3(signature = (input, shifts, dims=None))]
pub(crate) fn roll_py(
    py: Python<'_>,
    input: Py<PyTensor>,
    shifts: &Bound<'_, PyAny>,
    dims: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyTensor>> {
    let shifts = roll_arguments(shifts, "shifts")?;
    let dims = dims.map(|dims| roll_arguments(dims, "dims")).transpose()?;
    if dims.as_ref().is_none_or(|dims| dims.len() != shifts.len())
        && (dims.is_some() || shifts.len() != 1)
    {
        return Err(PyValueError::new_err(
            "roll shifts and dimensions must align",
        ));
    }

    let input_ref = input.borrow(py);
    input_ref.require_owner_thread()?;
    let original_shape = input_ref.source.value().dims();
    let flattened = dims.is_none();
    let dims = dims.unwrap_or_else(|| vec![0]);
    let mut result = if flattened {
        Some(input_ref.reshape_value(vec![original_shape.iter().product()])?)
    } else {
        None
    };

    for (shift, dim) in shifts.into_iter().zip(dims) {
        let current = result.as_ref().unwrap_or(&input_ref);
        let shape = current.source.value().dims();
        let rank = i64::try_from(shape.len()).expect("tensor rank fits i64");
        if !(-rank..rank).contains(&dim) {
            return Err(PyIndexError::new_err(format!(
                "roll dimension {dim} out of range for rank {rank}"
            )));
        }
        let axis = usize::try_from(if dim < 0 { rank + dim } else { dim })
            .expect("normalized roll dimension is non-negative");
        let size = shape[axis];
        if size == 0 {
            continue;
        }
        let size_i64 = i64::try_from(size)
            .map_err(|_| PyValueError::new_err("roll dimension exceeds integer limits"))?;
        let shift = usize::try_from(shift.rem_euclid(size_i64))
            .expect("normalized roll shift is non-negative");
        if shift == 0 {
            continue;
        }

        let split = size - shift;
        let split_isize = isize::try_from(split)
            .map_err(|_| PyValueError::new_err("roll dimension exceeds index limits"))?;
        let size_isize = isize::try_from(size)
            .map_err(|_| PyValueError::new_err("roll dimension exceeds index limits"))?;
        let mut trailing_slices = vec![Slice::full(); shape.len()];
        trailing_slices[axis] = Slice::new(split_isize, Some(size_isize), 1);
        let mut trailing_shape = shape.clone();
        trailing_shape[axis] = shift;
        let trailing = current.slice_value(trailing_slices, trailing_shape)?;

        let mut leading_slices = vec![Slice::full(); shape.len()];
        leading_slices[axis] = Slice::new(0, Some(split_isize), 1);
        let mut leading_shape = shape;
        leading_shape[axis] = split;
        let leading = current.slice_value(leading_slices, leading_shape)?;
        result = Some(combine_tensors(
            &[&trailing, &leading],
            axis,
            false,
            "roll",
        )?);
    }

    if flattened {
        result = Some(
            result
                .as_ref()
                .expect("flattened roll always owns an intermediate")
                .reshape_value(original_shape)?,
        );
    }
    drop(input_ref);
    match result {
        Some(result) => Py::new(py, result),
        None => Ok(input),
    }
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
    input.require_owner_thread()?;
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
    input.require_owner_thread()?;
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
        *output = output.with_trace(trace)?;
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
