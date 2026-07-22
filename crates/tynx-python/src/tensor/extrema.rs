//! Typed value-only reductions and elementwise extrema.

use pyo3::{exceptions::PyTypeError, prelude::*};
use tynx_core::DynTensor;

use super::data::TensorValue;
use crate::to_python_error;

#[derive(Debug, Clone, Copy)]
pub(super) enum Extremum {
    Minimum,
    Maximum,
}

impl Extremum {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Minimum => "minimum",
            Self::Maximum => "maximum",
        }
    }

    pub(super) fn float_pair(
        self,
        left: DynTensor,
        right: DynTensor,
    ) -> tynx_core::Result<DynTensor> {
        match self {
            Self::Minimum => left.min_broadcast(right),
            Self::Maximum => left.max_broadcast(right),
        }
    }

    pub(super) fn float_reduce(self, value: DynTensor, dims: &[usize]) -> DynTensor {
        match self {
            Self::Minimum => value.reduce_min_dims(dims),
            Self::Maximum => value.reduce_max_dims(dims),
        }
    }
}

pub(super) fn pair(
    left: TensorValue,
    right: TensorValue,
    extremum: Extremum,
) -> PyResult<TensorValue> {
    let left_dtype = left.dtype_name();
    let right_dtype = right.dtype_name();
    match (left, right) {
        (TensorValue::Float(left), TensorValue::Float(right)) => extremum
            .float_pair(left, right)
            .map(TensorValue::Float)
            .map_err(to_python_error),
        (TensorValue::Int(left), TensorValue::Int(right)) => match extremum {
            Extremum::Minimum => left.min_broadcast(right),
            Extremum::Maximum => left.max_broadcast(right),
        }
        .map(TensorValue::Int)
        .map_err(to_python_error),
        (TensorValue::Bool(left), TensorValue::Bool(right)) => match extremum {
            Extremum::Minimum => left.and_broadcast(right),
            Extremum::Maximum => left.or_broadcast(right),
        }
        .map(TensorValue::Bool)
        .map_err(to_python_error),
        _ => Err(PyTypeError::new_err(format!(
            "{} requires matching Tensor dtypes, got {left_dtype} and {right_dtype}",
            extremum.name()
        ))),
    }
}

pub(super) fn reduce(
    value: TensorValue,
    dims: &[usize],
    output_shape: Vec<usize>,
    extremum: Extremum,
) -> PyResult<TensorValue> {
    match value {
        TensorValue::Float(value) => extremum
            .float_reduce(value, dims)
            .reshape(output_shape)
            .map(TensorValue::Float)
            .map_err(to_python_error),
        TensorValue::Int(value) => match extremum {
            Extremum::Minimum => value.reduce_min_dims(dims),
            Extremum::Maximum => value.reduce_max_dims(dims),
        }
        .reshape(output_shape)
        .map(TensorValue::Int)
        .map_err(to_python_error),
        TensorValue::Bool(value) => match extremum {
            Extremum::Minimum => value.reduce_min_dims(dims),
            Extremum::Maximum => value.reduce_max_dims(dims),
        }
        .reshape(output_shape)
        .map(TensorValue::Bool)
        .map_err(to_python_error),
    }
}
