//! CPython projection of the slot-backed imported training model.

use std::{collections::HashMap, path::PathBuf, rc::Rc};

use pyo3::{
    exceptions::{PyOSError, PyTypeError},
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use tynx_capture::CapturedOperation;
use tynx_core::{Device, Env, Result, Session, Value};
use tynx_train::{
    BackwardCapability, ImportedModel, InitializerNameOverrides, TrainabilityOverrides,
    TrainabilityReport,
};

use crate::{
    capture::record_operation,
    device::{PyDevice, ensure_autodiff},
    grad_mode::is_grad_enabled,
    parameter::{PyBuffer, PyParameter, buffer_from_slot, parameter_from_slot},
    tensor::PyTensor,
    to_python_error,
};

/// A callable ONNX model backed by stable trainable parameter slots.
#[pyclass(name = "ImportedModel", frozen, unsendable)]
pub(crate) struct PyImportedModel {
    inner: Rc<ImportedModel>,
}

#[derive(Debug)]
struct CapturedImportedModel {
    inner: Rc<ImportedModel>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl CapturedOperation for CapturedImportedModel {
    fn run(&self, inputs: &[Value], tracking: bool) -> Result<Vec<Value>> {
        let mut env = Env::new();
        for (name, value) in self.inputs.iter().zip(inputs) {
            env.insert(name.clone(), value.clone());
        }
        let mut result = self.inner.run_with_tracking(env, tracking)?;
        self.outputs
            .iter()
            .map(|name| {
                result
                    .remove(name)
                    .ok_or_else(|| tynx_core::TynxError::MissingValue(name.clone()))
            })
            .collect()
    }

    fn output_count(&self) -> usize {
        self.outputs.len()
    }

    fn state_slots(&self) -> Vec<tynx_train::ParameterSlot> {
        self.inner.parameters().iter().cloned().collect()
    }
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
    #[pyo3(signature = (
        path,
        *,
        simplify=true,
        initializer_names=None,
        outputs=None,
        device=None
    ))]
    fn new(
        path: PathBuf,
        simplify: bool,
        initializer_names: Option<HashMap<String, String>>,
        outputs: Option<Vec<String>>,
        device: Option<&PyDevice>,
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
        let device = device
            .map(|device| ensure_autodiff(device.inner.as_ref().clone()))
            .unwrap_or_else(|| Device::autodiff(tynx_core::default_device()));
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
            inner: Rc::new(inner),
        })
    }

    /// Bind positional/named model inputs and return one Tensor or a named output dictionary.
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let inputs = self.inner.session().inputs();
        let outputs = self.inner.session().outputs();
        if args.len() > inputs.len() {
            return Err(PyTypeError::new_err(format!(
                "ImportedModel expected at most {} positional inputs, got {}",
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
                let name = name.extract::<String>().map_err(|_| {
                    PyTypeError::new_err("ImportedModel input names must be strings")
                })?;
                let index = inputs
                    .iter()
                    .position(|input| input.name == name)
                    .ok_or_else(|| {
                        PyTypeError::new_err(format!(
                            "ImportedModel got an unexpected input {name:?}; expected {:?}",
                            self.inputs()
                        ))
                    })?;
                if bound[index].is_some() {
                    return Err(PyTypeError::new_err(format!(
                        "ImportedModel got multiple values for input {name:?}"
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
                "ImportedModel is missing required inputs {missing:?}"
            )));
        }

        let tracking = is_grad_enabled();
        let mut env = Env::new();
        for (input, value) in inputs.iter().zip(&bound) {
            let value = value
                .as_ref()
                .expect("missing imported inputs were rejected above")
                .operation_runtime_value(tracking, "imported model input")?;
            env.insert(input.name.clone(), value);
        }
        let mut result = self
            .inner
            .run_with_tracking(env, tracking)
            .map_err(to_python_error)?;
        let sources = bound
            .iter()
            .map(|value| {
                &**value
                    .as_ref()
                    .expect("missing imported inputs were rejected above")
            })
            .collect::<Vec<_>>();
        let traces = record_operation(
            &sources,
            Rc::new(CapturedImportedModel {
                inner: self.inner.clone(),
                inputs: inputs.iter().map(|input| input.name.clone()).collect(),
                outputs: outputs.iter().map(|output| output.name.clone()).collect(),
            }),
        )?;
        let mut wrap_output = |index: usize, name: &str| -> PyResult<Py<PyTensor>> {
            let output = result
                .remove(name)
                .expect("ImportedModel returns every declared output")
                .into_tensor()
                .map_err(to_python_error)?;
            let tensor = if tracking {
                PyTensor::from_imported_operation(
                    output,
                    &sources,
                    self.inner.parameters().trainable().cloned(),
                )
            } else {
                PyTensor::from_inner(output.detach())
            };
            let tensor = match traces.as_ref().and_then(|traces| traces.get(index)) {
                Some(trace) => tensor.with_trace(trace.clone()),
                None => tensor,
            };
            Py::new(py, tensor)
        };

        if outputs.len() == 1 {
            return Ok(wrap_output(0, &outputs[0].name)?.into_any());
        }
        let named = PyDict::new(py);
        for (index, output) in outputs.iter().enumerate() {
            named.set_item(&output.name, wrap_output(index, &output.name)?)?;
        }
        Ok(named.unbind().into_any())
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

    fn state_dict(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(py
            .import("tynx.nn.state")?
            .getattr("get_state_dict")?
            .call1((slf,))?
            .unbind())
    }

    #[pyo3(signature = (state_dict, strict=true))]
    fn load_state_dict(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        state_dict: &Bound<'_, PyAny>,
        strict: bool,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("strict", strict)?;
        Ok(py
            .import("tynx.nn.state")?
            .getattr("load_state_dict")?
            .call((slf, state_dict), Some(&kwargs))?
            .unbind())
    }

    fn trainability_report(&self) -> PyTrainabilityReport {
        self.inner.trainability_report().clone().into()
    }

    #[pyo3(signature = (outputs=None))]
    fn require_trainable(&self, outputs: Option<Vec<String>>) -> PyResult<PyTrainabilityReport> {
        let outputs = outputs
            .as_ref()
            .map(|outputs| outputs.iter().map(String::as_str).collect::<Vec<_>>());
        let report = self.inner.trainability_for_outputs(outputs.as_deref());
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
