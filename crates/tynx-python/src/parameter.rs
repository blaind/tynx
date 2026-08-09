//! Python parameter and buffer bindings backed by stable training slots.

use pyo3::{prelude::*, types::PyAny};
use tynx_train::ParameterSlot;

use crate::{tensor::PyTensor, to_python_error};

/// A trainable tensor with stable mutable identity.
#[pyclass(name = "Parameter", extends = PyTensor, frozen)]
pub(crate) struct PyParameter;

pub(crate) fn parameter_from_slot(
    py: Python<'_>,
    slot: ParameterSlot,
) -> PyResult<Py<PyParameter>> {
    Py::new(
        py,
        PyClassInitializer::from(PyTensor::from_parameter(slot)).add_subclass(PyParameter),
    )
}

#[pymethods]
impl PyParameter {
    #[new]
    #[pyo3(signature = (data, *, name=None))]
    fn new(data: &Bound<'_, PyAny>, name: Option<String>) -> PyResult<PyClassInitializer<Self>> {
        let value = PyTensor::tensor_from_python(data)?;
        let slot = ParameterSlot::new(name, value, true).map_err(to_python_error)?;
        Ok(PyClassInitializer::from(PyTensor::from_parameter(slot)).add_subclass(Self))
    }

    /// Optional stable state-dictionary name.
    #[getter]
    fn name(slf: PyRef<'_, Self>) -> Option<String> {
        slf.as_super().parameter_name()
    }
}

/// A non-trainable tensor with stable mutable identity.
#[pyclass(name = "Buffer", extends = PyTensor, frozen)]
pub(crate) struct PyBuffer;

pub(crate) fn buffer_from_slot(py: Python<'_>, slot: ParameterSlot) -> PyResult<Py<PyBuffer>> {
    Py::new(
        py,
        PyClassInitializer::from(PyTensor::from_parameter(slot)).add_subclass(PyBuffer),
    )
}

#[pymethods]
impl PyBuffer {
    #[new]
    #[pyo3(signature = (data, *, name=None))]
    fn new(data: &Bound<'_, PyAny>, name: Option<String>) -> PyResult<PyClassInitializer<Self>> {
        let value = PyTensor::tensor_from_python(data)?;
        let slot = ParameterSlot::new(name, value, false).map_err(to_python_error)?;
        Ok(PyClassInitializer::from(PyTensor::from_parameter(slot)).add_subclass(Self))
    }

    /// Optional stable state-dictionary name.
    #[getter]
    fn name(slf: PyRef<'_, Self>) -> Option<String> {
        slf.as_super().parameter_name()
    }
}
