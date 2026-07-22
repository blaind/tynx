//! Typed eager indexing validation and dispatch.

use pyo3::{
    exceptions::{PyIndexError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBool, PyEllipsis, PySlice, PySliceMethods, PyTuple},
};
use tynx_core::{DynInt, Slice};

use super::data::TensorValue;
use crate::to_python_error;

pub(super) struct BasicIndex {
    pub(super) slices: Vec<Slice>,
    pub(super) output_shape: Vec<usize>,
}

enum IndexItem<'py> {
    Full,
    NewAxis,
    Value(Bound<'py, PyAny>),
}

pub(super) fn basic_index(key: &Bound<'_, PyAny>, shape: &[usize]) -> PyResult<BasicIndex> {
    let items = if let Ok(tuple) = key.cast::<PyTuple>() {
        tuple.iter().collect::<Vec<_>>()
    } else {
        vec![key.clone()]
    };
    let ellipses = items
        .iter()
        .filter(|item| item.is_instance_of::<PyEllipsis>())
        .count();
    if ellipses > 1 {
        return Err(PyIndexError::new_err(
            "an index can only have a single ellipsis",
        ));
    }
    let consuming = items
        .iter()
        .filter(|item| !item.is_instance_of::<PyEllipsis>() && !item.is_none())
        .count();
    if consuming > shape.len() {
        return Err(PyIndexError::new_err(format!(
            "too many indices for tensor of dimension {}",
            shape.len()
        )));
    }

    let mut expanded = Vec::with_capacity(shape.len() + items.len());
    for item in items {
        if item.is_instance_of::<PyEllipsis>() {
            expanded.extend((0..shape.len() - consuming).map(|_| IndexItem::Full));
        } else if item.is_none() {
            expanded.push(IndexItem::NewAxis);
        } else {
            expanded.push(IndexItem::Value(item));
        }
    }
    let consumed = expanded
        .iter()
        .filter(|item| matches!(item, IndexItem::Full | IndexItem::Value(_)))
        .count();
    expanded.extend((consumed..shape.len()).map(|_| IndexItem::Full));

    let mut slices = Vec::with_capacity(shape.len());
    let mut output_shape = Vec::with_capacity(expanded.len());
    let mut axis = 0;
    for item in expanded {
        if matches!(&item, IndexItem::NewAxis) {
            output_shape.push(1);
            continue;
        }
        let size = shape[axis];
        if matches!(&item, IndexItem::Full) {
            slices.push(Slice::full());
            output_shape.push(size);
            axis += 1;
            continue;
        }
        let IndexItem::Value(item) = item else {
            unreachable!("new axes and full slices were handled above")
        };
        if item.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "boolean and advanced tensor indexing are not supported",
            ));
        }
        if let Ok(index) = item.extract::<isize>() {
            let size_signed = isize::try_from(size)
                .map_err(|_| PyIndexError::new_err("tensor dimension exceeds index limits"))?;
            let index = if index < 0 {
                size_signed + index
            } else {
                index
            };
            if !(0..size_signed).contains(&index) {
                return Err(PyIndexError::new_err(format!(
                    "index {index} is out of bounds for dimension {axis} with size {size}"
                )));
            }
            slices.push(Slice::new(index, Some(index + 1), 1));
            axis += 1;
            continue;
        }
        if let Ok(slice) = item.cast::<PySlice>() {
            let size = isize::try_from(size)
                .map_err(|_| PyIndexError::new_err("tensor dimension exceeds index limits"))?;
            let normalized = slice.indices(size)?;
            let burn_slice = if normalized.slicelength == 0 {
                Slice::new(0, Some(0), normalized.step)
            } else if normalized.step > 0 {
                Slice::new(normalized.start, Some(normalized.stop), normalized.step)
            } else {
                let count = isize::try_from(normalized.slicelength)
                    .expect("slice length fits the indexed dimension");
                let lowest = normalized.start + (count - 1) * normalized.step;
                Slice::new(lowest, Some(normalized.start + 1), normalized.step)
            };
            slices.push(burn_slice);
            output_shape.push(normalized.slicelength);
            axis += 1;
            continue;
        }
        return Err(PyTypeError::new_err(
            "indices must be integers, slices, None, tuples of those forms, or ellipsis",
        ));
    }
    if output_shape.len() > tynx_core::MAX_RANK {
        return Err(PyIndexError::new_err(format!(
            "indexing result rank {} exceeds the maximum supported rank {}",
            output_shape.len(),
            tynx_core::MAX_RANK
        )));
    }
    if output_shape.is_empty() {
        output_shape.push(1);
    }
    Ok(BasicIndex {
        slices,
        output_shape,
    })
}

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
