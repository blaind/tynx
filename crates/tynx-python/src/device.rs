//! Python projection of the process execution device.

use pyo3::{exceptions::PyRuntimeError, prelude::*};
use tynx_core::Device;

/// A Tynx execution device.
#[pyclass(name = "Device", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyDevice {
    pub(crate) inner: Box<Device>,
}

impl PyDevice {
    pub(crate) fn new(inner: Device) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }

    pub(crate) fn sync(&self) -> PyResult<()> {
        tynx_core::synchronize(&self.inner)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
}

pub(crate) fn raise_pending_device_error() -> PyResult<()> {
    match tynx_core::take_device_error() {
        Some(error) => Err(PyRuntimeError::new_err(format!(
            "asynchronous device error: {error}"
        ))),
        None => Ok(()),
    }
}

#[pymethods]
impl PyDevice {
    fn __repr__(&self) -> String {
        format!("Device({:?})", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner.as_ref() == other.inner.as_ref()
    }
}
