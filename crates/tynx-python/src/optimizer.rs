//! CPython optimizer projections over stable Rust parameter slots.

mod adam;
mod parameters;
mod state;

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyDict},
};
use tynx_train::{ParameterSlot, ParameterStore, Sgd, SgdConfig};

use crate::to_python_error;
pub(crate) use parameters::collect_parameters;

pub(crate) use adam::{PyAdam, PyAdamW};

/// Stochastic gradient descent over an explicit, stable parameter list.
#[pyclass(name = "SGD", unsendable)]
pub(crate) struct PySgd {
    inner: Sgd,
    parameters: Vec<ParameterSlot>,
    named_parameters: Option<ParameterStore>,
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
        let collected = collect_parameters(parameters, "SGD")?;
        let config = SgdConfig::new(lr)
            .with_momentum(momentum)
            .with_dampening(dampening)
            .with_weight_decay(weight_decay)
            .with_nesterov(nesterov);
        let inner = Sgd::with_config(config).map_err(to_python_error)?;
        Ok(Self {
            inner,
            parameters: collected.slots,
            named_parameters: collected.named,
        })
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

    /// Return a portable optimizer dictionary when constructed from named parameters.
    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let parameters = require_named_parameters(&self.named_parameters, "SGD")?;
        state::sgd_to_python(py, &self.inner.state_dict(parameters))
    }

    /// Restore a portable optimizer dictionary by stable parameter name.
    fn load_state_dict(&mut self, state_dict: &Bound<'_, PyAny>) -> PyResult<()> {
        let parameters = require_named_parameters(&self.named_parameters, "SGD")?;
        let state_dict = state::sgd_from_python(state_dict)?;
        self.inner
            .load_state_dict(parameters, &state_dict)
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

fn zero_grad(parameters: &[ParameterSlot]) {
    for parameter in parameters {
        parameter.zero_grad();
    }
}

fn require_named_parameters<'a>(
    parameters: &'a Option<ParameterStore>,
    optimizer_name: &str,
) -> PyResult<&'a ParameterStore> {
    parameters.as_ref().ok_or_else(|| {
        PyValueError::new_err(format!(
            "{optimizer_name}.state_dict() requires construction from model.named_parameters()"
        ))
    })
}
