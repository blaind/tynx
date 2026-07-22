//! Python bindings for Tynx.

mod capture;
mod device;
mod grad_mode;
mod gradient;
mod imported_model;
mod nn;
mod optimizer;
mod parameter;
mod random;
mod tensor;

use std::path::PathBuf;

use capture::{PyCaptureSession, PyCapturedGraph};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use tynx_core::Session;

use device::PyDevice;
use grad_mode::{PyNoGrad, is_grad_enabled_py, no_grad};
use gradient::{clip_grad_norm_py, clip_grad_value_py};
use imported_model::{PyImportedModel, PyTrainabilityReport};
use nn::conv2d_py;
use optimizer::{PyAdam, PyAdamW, PySgd};
use parameter::{PyBuffer, PyParameter};
use random::{categorical_sample_py, dropout_py, manual_seed_py, normal_sample_py};
use tensor::{PyTensor, maximum_py, minimum_py, where_py};

/// Return the process-default execution device.
#[pyfunction]
fn get_default_device() -> PyDevice {
    PyDevice::new(tynx_core::default_device())
}

/// Wait until work queued for a device is complete.
#[pyfunction(signature = (device=None))]
fn synchronize(device: Option<PyRef<'_, PyDevice>>) -> PyResult<()> {
    match device {
        Some(device) => device.sync(),
        None => PyDevice::new(tynx_core::default_device()).sync(),
    }
}

/// A parsed ONNX model.
#[pyclass(name = "Session", frozen)]
struct PySession {
    inner: Box<Session>,
}

#[pymethods]
impl PySession {
    /// Load and parse an ONNX model.
    #[new]
    #[pyo3(signature = (path, *, simplify=true))]
    fn new(path: PathBuf, simplify: bool) -> PyResult<Self> {
        let data = std::fs::read(&path).map_err(|error| {
            PyOSError::new_err(format!("could not read '{}': {error}", path.display()))
        })?;
        let session = Session::from_bytes_with(&data, simplify).map_err(to_python_error)?;

        Ok(Self {
            inner: Box::new(session),
        })
    }

    /// Names of the model's declared inputs.
    #[getter]
    fn inputs(&self) -> Vec<String> {
        self.inner
            .inputs()
            .iter()
            .map(|input| input.name.clone())
            .collect()
    }

    /// Names of the model's declared outputs.
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.inner
            .outputs()
            .iter()
            .map(|output| output.name.clone())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Session(inputs={:?}, outputs={:?})",
            self.inputs(),
            self.outputs()
        )
    }
}

pub(crate) fn to_python_error(error: tynx_core::TynxError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Native Python module for Tynx.
#[pymodule]
fn _tynx(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<PySession>()?;
    module.add_class::<PyImportedModel>()?;
    module.add_class::<PyTrainabilityReport>()?;
    module.add_class::<PyDevice>()?;
    module.add_class::<PyTensor>()?;
    module.add_class::<PyParameter>()?;
    module.add_class::<PyBuffer>()?;
    module.add_class::<PyNoGrad>()?;
    module.add_class::<PyCaptureSession>()?;
    module.add_class::<PyCapturedGraph>()?;
    module.add_class::<PySgd>()?;
    module.add_class::<PyAdam>()?;
    module.add_class::<PyAdamW>()?;
    module.add_function(wrap_pyfunction!(no_grad, module)?)?;
    module.add_function(wrap_pyfunction!(is_grad_enabled_py, module)?)?;
    module.add_function(wrap_pyfunction!(get_default_device, module)?)?;
    module.add_function(wrap_pyfunction!(synchronize, module)?)?;
    module.add_function(wrap_pyfunction!(where_py, module)?)?;
    module.add_function(wrap_pyfunction!(maximum_py, module)?)?;
    module.add_function(wrap_pyfunction!(minimum_py, module)?)?;
    module.add_function(wrap_pyfunction!(clip_grad_norm_py, module)?)?;
    module.add_function(wrap_pyfunction!(clip_grad_value_py, module)?)?;
    module.add_function(wrap_pyfunction!(manual_seed_py, module)?)?;
    module.add_function(wrap_pyfunction!(normal_sample_py, module)?)?;
    module.add_function(wrap_pyfunction!(categorical_sample_py, module)?)?;
    module.add_function(wrap_pyfunction!(dropout_py, module)?)?;
    module.add_function(wrap_pyfunction!(conv2d_py, module)?)?;
    Ok(())
}
