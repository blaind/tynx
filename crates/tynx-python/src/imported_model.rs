//! CPython projection of the slot-backed imported training model.

use std::{collections::HashMap, path::PathBuf};

use pyo3::{exceptions::PyOSError, prelude::*};
use tynx_core::{Device, Env, Session, Value};
use tynx_train::{ImportedModel, InitializerNameOverrides, TrainabilityOverrides};

use crate::{
    grad_mode::is_grad_enabled,
    parameter::{PyBuffer, PyParameter, buffer_from_slot, parameter_from_slot},
    tensor::PyTensor,
    to_python_error,
};

/// A callable ONNX model backed by stable trainable parameter slots.
#[pyclass(name = "ImportedModel", frozen, unsendable)]
pub(crate) struct PyImportedModel {
    inner: Box<ImportedModel>,
}

#[pymethods]
impl PyImportedModel {
    #[new]
    #[pyo3(signature = (path, *, simplify=true, initializer_names=None))]
    fn new(
        path: PathBuf,
        simplify: bool,
        initializer_names: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        let data = std::fs::read(&path).map_err(|error| {
            PyOSError::new_err(format!("could not read '{}': {error}", path.display()))
        })?;
        let session = Session::from_bytes_with(&data, simplify).map_err(to_python_error)?;
        let mut names = InitializerNameOverrides::new();
        for (report_name, state_name) in initializer_names.unwrap_or_default() {
            names
                .set_name(report_name, state_name)
                .map_err(to_python_error)?;
        }
        let device = Device::autodiff(tynx_core::default_device());
        let inner = ImportedModel::from_session_with(
            session,
            device,
            &TrainabilityOverrides::new(),
            &names,
        )
        .map_err(to_python_error)?;
        Ok(Self {
            inner: Box::new(inner),
        })
    }

    /// Run a single-input/single-output imported forward and return a normal eager Tensor.
    fn __call__(&self, input: PyRef<'_, PyTensor>) -> PyResult<PyTensor> {
        let inputs = self.inner.session().inputs();
        let outputs = self.inner.session().outputs();
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "positional imported-model calls currently require one input and one output; model declares {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        let tracking = is_grad_enabled();
        let value = input.operation_float_value(tracking, "imported model input")?;
        let env = Env::from([(inputs[0].name.clone(), Value::Tensor(value))]);
        let mut result = self
            .inner
            .run_with_tracking(env, tracking)
            .map_err(to_python_error)?;
        let output = result
            .remove(&outputs[0].name)
            .expect("ImportedModel returns every declared output")
            .into_tensor()
            .map_err(to_python_error)?;
        Ok(if tracking {
            PyTensor::from_imported_operation(
                output,
                &[&input],
                self.inner.parameters().trainable().cloned(),
            )
        } else {
            PyTensor::from_inner(output.detach())
        })
    }

    #[getter]
    fn inputs(&self) -> Vec<String> {
        self.inner
            .session()
            .inputs()
            .iter()
            .map(|input| input.name.clone())
            .collect()
    }

    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.inner
            .session()
            .outputs()
            .iter()
            .map(|output| output.name.clone())
            .collect()
    }

    fn parameters(&self, py: Python<'_>) -> PyResult<Vec<Py<PyParameter>>> {
        self.inner
            .state()
            .parameters()
            .map(|(_, slot)| parameter_from_slot(py, slot.clone()))
            .collect()
    }

    fn named_parameters(&self, py: Python<'_>) -> PyResult<Vec<(String, Py<PyParameter>)>> {
        self.inner
            .state()
            .parameters()
            .map(|(name, slot)| Ok((name.to_string(), parameter_from_slot(py, slot.clone())?)))
            .collect()
    }

    fn buffers(&self, py: Python<'_>) -> PyResult<Vec<Py<PyBuffer>>> {
        self.inner
            .state()
            .buffers()
            .map(|(_, slot)| buffer_from_slot(py, slot.clone()))
            .collect()
    }

    fn named_buffers(&self, py: Python<'_>) -> PyResult<Vec<(String, Py<PyBuffer>)>> {
        self.inner
            .state()
            .buffers()
            .map(|(name, slot)| Ok((name.to_string(), buffer_from_slot(py, slot.clone())?)))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ImportedModel(inputs={:?}, outputs={:?}, parameters={})",
            self.inputs(),
            self.outputs(),
            self.inner.parameters().trainable().count()
        )
    }
}
