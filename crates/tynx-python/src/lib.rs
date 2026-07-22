//! Python bindings for Tynx.

mod grad_mode;
mod parameter;
mod tensor;

use std::path::PathBuf;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use tynx_core::Session;

use grad_mode::{PyNoGrad, is_grad_enabled_py, no_grad};
use parameter::PyParameter;
use tensor::PyTensor;

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
    module.add_class::<PyTensor>()?;
    module.add_class::<PyParameter>()?;
    module.add_class::<PyNoGrad>()?;
    module.add_function(wrap_pyfunction!(no_grad, module)?)?;
    module.add_function(wrap_pyfunction!(is_grad_enabled_py, module)?)?;
    Ok(())
}
