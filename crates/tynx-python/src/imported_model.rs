//! CPython projection of the slot-backed imported training model.

use std::{collections::HashMap, path::PathBuf};

use pyo3::{
    exceptions::PyOSError,
    prelude::*,
    types::{PyDict, PyList},
};
use tynx_core::{Device, Env, Session, Value};
use tynx_train::{
    BackwardCapability, ImportedModel, InitializerNameOverrides, TrainabilityOverrides,
    TrainabilityReport,
};

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

/// Structured result of conservative imported-model training analysis.
#[pyclass(name = "TrainabilityReport", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyTrainabilityReport {
    inner: TrainabilityReport,
}

impl From<TrainabilityReport> for PyTrainabilityReport {
    fn from(inner: TrainabilityReport) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTrainabilityReport {
    #[getter]
    fn is_trainable(&self) -> bool {
        self.inner.is_trainable()
    }

    #[getter]
    fn selected_outputs(&self) -> Vec<String> {
        self.inner.selected_outputs().to_vec()
    }

    #[getter]
    fn trainable_parameters(&self) -> Vec<String> {
        self.inner
            .trainable_parameters()
            .map(|initializer| initializer.name().to_string())
            .collect()
    }

    #[getter]
    fn buffers(&self) -> Vec<String> {
        self.inner
            .buffers()
            .map(|initializer| initializer.name().to_string())
            .collect()
    }

    #[getter]
    fn constants(&self) -> Vec<String> {
        self.inner
            .constants()
            .map(|initializer| initializer.name().to_string())
            .collect()
    }

    #[getter]
    fn unused_parameters(&self) -> Vec<String> {
        self.inner.unused_parameters().to_vec()
    }

    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings().to_vec()
    }

    #[getter]
    fn errors(&self) -> Vec<String> {
        self.inner.errors().to_vec()
    }

    #[getter]
    fn output_parameters(&self) -> HashMap<String, Vec<String>> {
        self.inner
            .selected_outputs()
            .iter()
            .map(|output| {
                (
                    output.clone(),
                    self.inner
                        .parameters_for_output(output)
                        .unwrap_or_default()
                        .to_vec(),
                )
            })
            .collect()
    }

    #[getter]
    fn initializers(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let values = self
            .inner
            .initializers()
            .iter()
            .map(|initializer| {
                let value = PyDict::new(py);
                value.set_item("name", initializer.name())?;
                value.set_item("role", initializer.role().to_string())?;
                value.set_item("synthetic_name", initializer.has_synthetic_name())?;
                value.set_item("dtype", format!("{:?}", initializer.dtype()))?;
                value.set_item("shape", initializer.shape().map(<[usize]>::to_vec))?;
                let uses = initializer
                    .uses()
                    .iter()
                    .map(|usage| {
                        let value = PyDict::new(py);
                        value.set_item("node", usage.node_name())?;
                        value.set_item("operator", usage.operator())?;
                        value.set_item("input_index", usage.input_index())?;
                        value.set_item("proposed_role", usage.proposed_role().to_string())?;
                        value.set_item("reason", usage.reason())?;
                        Ok(value)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                value.set_item("uses", PyList::new(py, uses)?)?;
                Ok(value)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, values)?.unbind())
    }

    #[getter]
    fn backward_issues(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let values = self
            .inner
            .backward_issues()
            .iter()
            .map(|issue| {
                let value = PyDict::new(py);
                value.set_item("output", issue.output())?;
                value.set_item("node", issue.node_name())?;
                value.set_item("operator", issue.operator())?;
                value.set_item("input_index", issue.input_index())?;
                let capability = match issue.capability() {
                    BackwardCapability::Differentiable => "differentiable",
                    BackwardCapability::StopGradient(_) => "stop_gradient",
                    BackwardCapability::Unsupported(_) => "unsupported",
                };
                value.set_item("capability", capability)?;
                value.set_item("reason", issue.capability().reason())?;
                value.set_item("parameters", issue.parameters())?;
                Ok(value)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, values)?.unbind())
    }

    fn require_trainable(&self) -> PyResult<()> {
        self.inner.require_trainable().map_err(to_python_error)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "TrainabilityReport(is_trainable={}, outputs={:?})",
            self.inner.is_trainable(),
            self.inner.selected_outputs()
        )
    }
}

#[pymethods]
impl PyImportedModel {
    #[new]
    #[pyo3(signature = (path, *, simplify=true, initializer_names=None, outputs=None))]
    fn new(
        path: PathBuf,
        simplify: bool,
        initializer_names: Option<HashMap<String, String>>,
        outputs: Option<Vec<String>>,
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
        let output_names = outputs
            .as_ref()
            .map(|outputs| outputs.iter().map(String::as_str).collect::<Vec<_>>());
        let inner = ImportedModel::from_session_for_outputs_with(
            session,
            device,
            output_names.as_deref(),
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

    fn trainability_report(&self) -> PyTrainabilityReport {
        self.inner.trainability_report().clone().into()
    }

    #[pyo3(signature = (outputs=None))]
    fn require_trainable(&self, outputs: Option<Vec<String>>) -> PyResult<PyTrainabilityReport> {
        let report = match outputs {
            Some(outputs) => {
                let outputs = outputs.iter().map(String::as_str).collect::<Vec<_>>();
                TrainabilityReport::analyze_outputs(self.inner.session().graph(), &outputs)
            }
            None => TrainabilityReport::analyze_all_outputs(self.inner.session().graph()),
        };
        report.require_trainable().map_err(to_python_error)?;
        Ok(report.into())
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
