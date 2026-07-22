//! CPython optimizer projections over stable Rust parameter slots.

mod adam;

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyAnyMethods},
};
use tynx_train::{ParameterSlot, Sgd, SgdConfig};

use crate::{tensor::PyTensor, to_python_error};

pub(crate) use adam::{PyAdam, PyAdamW};

/// Stochastic gradient descent over an explicit, stable parameter list.
#[pyclass(name = "SGD", unsendable)]
pub(crate) struct PySgd {
    inner: Sgd,
    parameters: Vec<ParameterSlot>,
}

#[pymethods]
impl PySgd {
    #[new]
    #[pyo3(signature = (parameters, lr, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=false))]
    fn new(
        parameters: &Bound<'_, PyAny>,
        lr: f64,
        momentum: f64,
        dampening: f64,
        weight_decay: f64,
        nesterov: bool,
    ) -> PyResult<Self> {
        let parameters = collect_parameters(parameters, "SGD")?;
        let config = SgdConfig::new(lr)
            .with_momentum(momentum)
            .with_dampening(dampening)
            .with_weight_decay(weight_decay)
            .with_nesterov(nesterov);
        let inner = Sgd::with_config(config).map_err(to_python_error)?;
        Ok(Self { inner, parameters })
    }

    /// Clear every managed parameter's persistent gradient.
    fn zero_grad(&self) {
        zero_grad(&self.parameters);
    }

    /// Apply one serialized off-tape update without clearing gradients.
    fn step(&mut self) -> PyResult<()> {
        self.inner
            .step_slots(&self.parameters)
            .map(|_| ())
            .map_err(to_python_error)
    }

    /// Mutable learning rate shorthand.
    #[getter]
    fn lr(&self) -> f64 {
        self.inner.config().learning_rate()
    }

    #[setter]
    fn set_lr(&mut self, value: f64) -> PyResult<()> {
        self.inner.set_learning_rate(value).map_err(to_python_error)
    }

    /// Long-form alias for `lr`.
    #[getter]
    fn learning_rate(&self) -> f64 {
        self.lr()
    }

    #[setter]
    fn set_learning_rate(&mut self, value: f64) -> PyResult<()> {
        self.set_lr(value)
    }

    /// Number of unique managed parameters.
    #[getter]
    fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    /// Number of allocated momentum buffers.
    #[getter]
    fn state_size(&self) -> usize {
        self.inner.state_len()
    }

    fn __repr__(&self) -> String {
        let config = self.inner.config();
        format!(
            "SGD(lr={}, momentum={}, dampening={}, weight_decay={}, nesterov={})",
            config.learning_rate(),
            config.momentum(),
            config.dampening(),
            config.weight_decay(),
            config.is_nesterov()
        )
    }
}

fn collect_parameters(
    parameters: &Bound<'_, PyAny>,
    optimizer_name: &str,
) -> PyResult<Vec<ParameterSlot>> {
    let iterator = parameters.try_iter().map_err(|_| {
        PyTypeError::new_err(format!(
            "{optimizer_name} parameters must be an iterable of Parameter objects"
        ))
    })?;
    let mut slots: Vec<ParameterSlot> = Vec::new();
    for item in iterator {
        let item = item?;
        let tensor = item.extract::<PyRef<'_, PyTensor>>().map_err(|_| {
            PyTypeError::new_err(format!(
                "{optimizer_name} parameters must contain only Parameter objects"
            ))
        })?;
        let slot = tensor.parameter_slot().ok_or_else(|| {
            PyTypeError::new_err(format!(
                "{optimizer_name} parameters must contain only Parameter objects"
            ))
        })?;
        if !slots.iter().any(|existing| existing.id() == slot.id()) {
            slots.push(slot);
        }
    }
    if slots.is_empty() {
        return Err(PyValueError::new_err(format!(
            "{optimizer_name} requires at least one Parameter"
        )));
    }
    Ok(slots)
}

fn zero_grad(parameters: &[ParameterSlot]) {
    for parameter in parameters {
        parameter.zero_grad();
    }
}
