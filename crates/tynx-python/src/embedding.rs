//! Narrow CPython embedding interfaces for trusted Rust integrations.

use pyo3::prelude::*;
use tynx_core::DynTensor;

use crate::tensor::PyTensor;

/// Transfer an existing Rust tensor into an ordinary Python `tynx.Tensor`.
///
/// This is an ownership handoff for trusted Rust integrations. It does not clone, detach, move,
/// read back, or otherwise transform the tensor. Consequently, device storage, autodiff tape, and
/// any external allocation retention carried by `tensor` remain attached to the Python object and
/// to Python operations derived from it.
///
/// The tensor is consumed so one Rust value can be handed to Python only once. External device
/// validation and buffer adoption must already have happened in the backend-neutral Tynx API;
/// ordinary Python callers are intentionally not given an external-buffer constructor.
pub fn wrap_external_tensor(py: Python<'_>, tensor: DynTensor) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, PyTensor::from_inner(tensor))?.into_any())
}
