//! Python reduction argument normalization and output-shape policy.

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBool, PyTuple, PyTupleMethods},
};

pub(super) struct ReductionSpec {
    pub(super) dims: Vec<usize>,
    pub(super) output_shape: Vec<usize>,
}

impl ReductionSpec {
    pub(super) fn from_python(
        dim: Option<&Bound<'_, PyAny>>,
        input_shape: &[usize],
        keepdim: bool,
    ) -> PyResult<Self> {
        let dims = parse_dims(dim, input_shape.len())?;
        let output_shape = output_shape(input_shape, &dims, keepdim);
        Ok(Self { dims, output_shape })
    }
}

fn parse_dims(dim: Option<&Bound<'_, PyAny>>, rank: usize) -> PyResult<Vec<usize>> {
    let Some(dim) = dim else {
        return Ok((0..rank).collect());
    };

    let raw_dims = if !dim.is_instance_of::<PyBool>() {
        if let Ok(axis) = dim.extract::<isize>() {
            vec![axis]
        } else if let Ok(tuple) = dim.cast::<PyTuple>() {
            tuple
                .iter()
                .map(|axis| extract_axis(&axis))
                .collect::<PyResult<Vec<_>>>()?
        } else {
            return Err(invalid_dim_type());
        }
    } else {
        return Err(invalid_dim_type());
    };

    // PyTorch treats an empty dimension tuple as an all-axis reduction.
    let raw_dims = if raw_dims.is_empty() {
        (0..rank)
            .map(|axis| isize::try_from(axis).expect("dynamic tensor rank fits in isize"))
            .collect()
    } else {
        raw_dims
    };
    let rank = isize::try_from(rank).expect("dynamic tensor rank fits in isize");
    let mut normalized = Vec::with_capacity(raw_dims.len());
    for raw_axis in raw_dims {
        let axis = if raw_axis < 0 {
            rank + raw_axis
        } else {
            raw_axis
        };
        if !(0..rank).contains(&axis) {
            return Err(PyValueError::new_err(format!(
                "reduction dim {raw_axis} is out of range for tensor rank {rank}"
            )));
        }
        let axis = usize::try_from(axis).expect("validated reduction dim is non-negative");
        if normalized.contains(&axis) {
            return Err(PyValueError::new_err(format!(
                "reduction dim {axis} appears more than once"
            )));
        }
        normalized.push(axis);
    }
    Ok(normalized)
}

fn extract_axis(axis: &Bound<'_, PyAny>) -> PyResult<isize> {
    if axis.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "reduction dim tuple must contain only integers",
        ));
    }
    axis.extract::<isize>()
        .map_err(|_| PyTypeError::new_err("reduction dim tuple must contain only integers"))
}

fn invalid_dim_type() -> PyErr {
    PyTypeError::new_err("reduction dim must be an int, a tuple of ints, or None")
}

fn output_shape(input: &[usize], dims: &[usize], keepdim: bool) -> Vec<usize> {
    let mut output: Vec<usize> = if keepdim {
        input
            .iter()
            .enumerate()
            .map(|(axis, &size)| if dims.contains(&axis) { 1 } else { size })
            .collect()
    } else {
        input
            .iter()
            .enumerate()
            .filter_map(|(axis, &size)| (!dims.contains(&axis)).then_some(size))
            .collect()
    };
    if output.is_empty() {
        output.push(1);
    }
    output
}
