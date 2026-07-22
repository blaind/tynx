//! CPython projections for off-tape gradient transformations.

use pyo3::{prelude::*, types::PyAny};
use tynx_train::{clip_grad_norm, clip_grad_value};

use crate::{optimizer::collect_parameters, tensor::PyTensor, to_python_error};

/// Clip the aggregate L2 gradient norm and return its value before clipping.
#[pyfunction(name = "clip_grad_norm_")]
#[pyo3(signature = (parameters, max_norm, norm_type=2.0))]
pub(crate) fn clip_grad_norm_py(
    parameters: &Bound<'_, PyAny>,
    max_norm: f64,
    norm_type: f64,
) -> PyResult<PyTensor> {
    let parameters = collect_parameters(parameters, "clip_grad_norm_")?;
    clip_grad_norm(&parameters, max_norm, norm_type)
        .map(PyTensor::from_inner)
        .map_err(to_python_error)
}

/// Clamp every managed parameter gradient to a symmetric scalar range.
#[pyfunction(name = "clip_grad_value_")]
pub(crate) fn clip_grad_value_py(parameters: &Bound<'_, PyAny>, clip_value: f64) -> PyResult<()> {
    let parameters = collect_parameters(parameters, "clip_grad_value_")?;
    clip_grad_value(&parameters, clip_value)
        .map(|_| ())
        .map_err(to_python_error)
}
