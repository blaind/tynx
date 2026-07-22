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
use pyo3::exceptions::{PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use tynx_core::{Device, Env, PreparedSession, Scalar, Session, Value};

use device::PyDevice;
use grad_mode::{PyNoGrad, is_grad_enabled_py, no_grad};
use gradient::{clip_grad_norm_py, clip_grad_value_py};
use imported_model::{PyImportedModel, PyTrainabilityReport};
use nn::{adaptive_avg_pool2d_py, avg_pool2d_py, conv2d_py, max_pool2d_py};
use optimizer::{PyAdam, PyAdamW, PySgd};
use parameter::{PyBuffer, PyParameter};
use random::{categorical_sample_py, dropout_py, manual_seed_py, normal_sample_py};
use tensor::{
    PyTensor, arange_py, empty_like_py, empty_py, full_like_py, full_py, maximum_py, minimum_py,
    ones_like_py, ones_py, rand_like_py, rand_py, randint_py, randn_like_py, randn_py, where_py,
    zeros_like_py, zeros_py,
};

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
    inner: Box<PreparedSession>,
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
        let device = Device::autodiff(tynx_core::default_device());
        let session = session.prepare(&device).map_err(to_python_error)?;

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

    /// Run inference with positional or ONNX-named Tensor inputs.
    #[pyo3(signature = (*args, **kwargs))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        self.execute(py, args, kwargs)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        self.execute(py, args, kwargs)
    }

    fn __repr__(&self) -> String {
        format!(
            "Session(inputs={:?}, outputs={:?})",
            self.inputs(),
            self.outputs()
        )
    }
}

impl PySession {
    fn execute<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let inputs = self.inner.inputs();
        let outputs = self.inner.outputs();
        if args.len() > inputs.len() {
            return Err(PyTypeError::new_err(format!(
                "Session expected at most {} positional inputs, got {}",
                inputs.len(),
                args.len()
            )));
        }

        let mut bound = (0..inputs.len()).map(|_| None).collect::<Vec<_>>();
        for (index, destination) in bound.iter_mut().enumerate().take(args.len()) {
            *destination = Some(args.get_item(index)?.extract::<PyRef<'py, PyTensor>>()?);
        }
        if let Some(kwargs) = kwargs {
            for (name, value) in kwargs.iter() {
                let name = name
                    .extract::<String>()
                    .map_err(|_| PyTypeError::new_err("Session input names must be strings"))?;
                let index = inputs
                    .iter()
                    .position(|input| input.name == name)
                    .ok_or_else(|| {
                        PyTypeError::new_err(format!(
                            "Session got an unexpected input {name:?}; expected {:?}",
                            self.inputs()
                        ))
                    })?;
                if bound[index].is_some() {
                    return Err(PyTypeError::new_err(format!(
                        "Session got multiple values for input {name:?}"
                    )));
                }
                bound[index] = Some(value.extract::<PyRef<'py, PyTensor>>()?);
            }
        }
        let missing = inputs
            .iter()
            .zip(&bound)
            .filter(|(_, value)| value.is_none())
            .map(|(input, _)| input.name.clone())
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(PyTypeError::new_err(format!(
                "Session is missing required inputs {missing:?}"
            )));
        }

        let mut env = Env::new();
        for (input, value) in inputs.iter().zip(&bound) {
            env.insert(
                input.name.clone(),
                value
                    .as_ref()
                    .expect("missing Session inputs were rejected above")
                    .detached_runtime_value(),
            );
        }
        let mut result = self.inner.run(env).map_err(to_python_error)?;
        let mut take_output = |name: &str| -> PyResult<Py<PyAny>> {
            let value = result
                .remove(name)
                .expect("PreparedSession returns every declared output");
            runtime_output_to_python(py, value)
        };

        if outputs.len() == 1 {
            return take_output(&outputs[0].name);
        }
        let named = PyDict::new(py);
        for output in outputs {
            named.set_item(&output.name, take_output(&output.name)?)?;
        }
        Ok(named.unbind().into_any())
    }
}

fn runtime_output_to_python(py: Python<'_>, value: Value) -> PyResult<Py<PyAny>> {
    match value {
        Value::Tensor(_) | Value::Int(_) | Value::Bool(_) => {
            Ok(Py::new(py, PyTensor::from_runtime_value(value)?)?.into_any())
        }
        Value::Scalar(Scalar::F64(value)) => Ok(value.into_pyobject(py)?.unbind().into_any()),
        Value::Scalar(Scalar::I64(value)) => Ok(value.into_pyobject(py)?.unbind().into_any()),
        Value::Scalar(Scalar::U64(value)) => Ok(value.into_pyobject(py)?.unbind().into_any()),
        Value::Scalar(Scalar::Bool(value)) => {
            Ok(value.into_pyobject(py)?.to_owned().unbind().into_any())
        }
        Value::Shape(value) => Ok(value.into_pyobject(py)?.unbind().into_any()),
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
    module.add_function(wrap_pyfunction!(max_pool2d_py, module)?)?;
    module.add_function(wrap_pyfunction!(avg_pool2d_py, module)?)?;
    module.add_function(wrap_pyfunction!(adaptive_avg_pool2d_py, module)?)?;
    module.add_function(wrap_pyfunction!(empty_py, module)?)?;
    module.add_function(wrap_pyfunction!(full_py, module)?)?;
    module.add_function(wrap_pyfunction!(zeros_py, module)?)?;
    module.add_function(wrap_pyfunction!(ones_py, module)?)?;
    module.add_function(wrap_pyfunction!(rand_py, module)?)?;
    module.add_function(wrap_pyfunction!(randn_py, module)?)?;
    module.add_function(wrap_pyfunction!(randint_py, module)?)?;
    module.add_function(wrap_pyfunction!(arange_py, module)?)?;
    module.add_function(wrap_pyfunction!(empty_like_py, module)?)?;
    module.add_function(wrap_pyfunction!(full_like_py, module)?)?;
    module.add_function(wrap_pyfunction!(zeros_like_py, module)?)?;
    module.add_function(wrap_pyfunction!(ones_like_py, module)?)?;
    module.add_function(wrap_pyfunction!(rand_like_py, module)?)?;
    module.add_function(wrap_pyfunction!(randn_like_py, module)?)?;
    Ok(())
}
