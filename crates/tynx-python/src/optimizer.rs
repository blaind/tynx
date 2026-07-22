//! CPython optimizer projections over stable Rust parameter slots.

mod adam;
mod parameters;
mod state;

use std::{cell::RefCell, rc::Rc};

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyDict},
};
use tynx_capture::CapturedOptimizer;
use tynx_core::Result;
use tynx_train::{ParameterSlot, ParameterStore, Sgd, SgdConfig};

use crate::{
    capture::{record_optimizer_step, record_zero_grad},
    to_python_error,
};
pub(crate) use parameters::collect_parameters;

pub(crate) use adam::{PyAdam, PyAdamW};

/// Stochastic gradient descent over an explicit, stable parameter list.
#[pyclass(name = "SGD", unsendable)]
pub(crate) struct PySgd {
    inner: Rc<RefCell<Sgd>>,
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
            inner: Rc::new(RefCell::new(inner)),
            parameters: collected.slots,
            named_parameters: collected.named,
        })
    }

    /// Clear every managed parameter's persistent gradient.
    fn zero_grad(&self) -> PyResult<()> {
        record_zero_grad(&self.parameters)?;
        zero_grad(&self.parameters);
        Ok(())
    }

    /// Apply one serialized off-tape update without clearing gradients.
    fn step(&mut self) -> PyResult<()> {
        record_optimizer_step(Rc::new(CapturedSgd {
            inner: self.inner.clone(),
            parameters: self.parameters.clone(),
        }))?;
        self.inner
            .borrow_mut()
            .step_slots(&self.parameters)
            .map(|_| ())
            .map_err(to_python_error)
    }

    /// Return a portable optimizer dictionary when constructed from named parameters.
    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let parameters = require_named_parameters(&self.named_parameters, "SGD")?;
        let state_dict = self.inner.borrow().state_dict(parameters);
        state::sgd_to_python(py, &state_dict)
    }

    /// Restore a portable optimizer dictionary by stable parameter name.
    fn load_state_dict(&mut self, state_dict: &Bound<'_, PyAny>) -> PyResult<()> {
        let parameters = require_named_parameters(&self.named_parameters, "SGD")?;
        let state_dict = state::sgd_from_python(state_dict)?;
        self.inner
            .borrow_mut()
            .load_state_dict(parameters, &state_dict)
            .map_err(to_python_error)
    }

    /// Mutable learning rate shorthand.
    #[getter]
    fn lr(&self) -> f64 {
        self.inner.borrow().config().learning_rate()
    }

    #[setter]
    fn set_lr(&mut self, value: f64) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .set_learning_rate(value)
            .map_err(to_python_error)
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

    fn _set_config(
        &mut self,
        lr: f64,
        momentum: f64,
        dampening: f64,
        weight_decay: f64,
        nesterov: bool,
    ) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .set_config(
                SgdConfig::new(lr)
                    .with_momentum(momentum)
                    .with_dampening(dampening)
                    .with_weight_decay(weight_decay)
                    .with_nesterov(nesterov),
            )
            .map_err(to_python_error)
    }

    #[getter]
    fn momentum(&self) -> f64 {
        self.inner.borrow().config().momentum()
    }

    #[getter]
    fn dampening(&self) -> f64 {
        self.inner.borrow().config().dampening()
    }

    #[getter]
    fn weight_decay(&self) -> f64 {
        self.inner.borrow().config().weight_decay()
    }

    #[getter]
    fn nesterov(&self) -> bool {
        self.inner.borrow().config().is_nesterov()
    }

    /// Number of unique managed parameters.
    #[getter]
    fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    /// Number of allocated momentum buffers.
    #[getter]
    fn state_size(&self) -> usize {
        self.inner.borrow().state_len()
    }

    fn __repr__(&self) -> String {
        let config = self.inner.borrow().config();
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

#[derive(Debug)]
struct CapturedSgd {
    inner: Rc<RefCell<Sgd>>,
    parameters: Vec<ParameterSlot>,
}

impl CapturedOptimizer for CapturedSgd {
    fn step(&self) -> Result<()> {
        self.inner
            .borrow_mut()
            .step_slots(&self.parameters)
            .map(|_| ())
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
