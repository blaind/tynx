//! CPython Adam-family optimizers over stable Rust parameter slots.

use std::{cell::RefCell, rc::Rc};

use pyo3::{
    prelude::*,
    types::{PyAny, PyDict},
};
use tynx_capture::CapturedOptimizer;
use tynx_core::Result;
use tynx_train::{Adam, AdamConfig, AdamStateKind, AdamW, AdamWConfig, ParameterSlot};

use super::{parameters::collect_parameters, require_named_parameters, state, zero_grad};
use crate::{
    capture::{record_optimizer_step, record_zero_grad},
    to_python_error,
};

/// Adam with coupled L2 weight decay over an explicit parameter list.
#[pyclass(name = "Adam", unsendable)]
pub(crate) struct PyAdam {
    inner: Rc<RefCell<Adam>>,
    parameters: Vec<ParameterSlot>,
    named_parameters: Option<tynx_train::ParameterStore>,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (
        parameters,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
        amsgrad=false,
    ))]
    fn new(
        parameters: &Bound<'_, PyAny>,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
        amsgrad: bool,
    ) -> PyResult<Self> {
        let collected = collect_parameters(parameters, "Adam")?;
        let config = AdamConfig::new(lr)
            .with_betas(betas.0, betas.1)
            .with_epsilon(eps)
            .with_weight_decay(weight_decay)
            .with_amsgrad(amsgrad);
        let inner = Adam::with_config(config).map_err(to_python_error)?;
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
        record_optimizer_step(Rc::new(CapturedAdam {
            inner: self.inner.clone(),
            parameters: self.parameters.clone(),
        }))?;
        self.inner
            .borrow_mut()
            .step_slots(&self.parameters)
            .map(|_| ())
            .map_err(to_python_error)
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let parameters = require_named_parameters(&self.named_parameters, "Adam")?;
        let state_dict = self.inner.borrow().state_dict(parameters);
        state::adam_to_python(py, &state_dict)
    }

    fn load_state_dict(&mut self, state_dict: &Bound<'_, PyAny>) -> PyResult<()> {
        let parameters = require_named_parameters(&self.named_parameters, "Adam")?;
        let state_dict = state::adam_from_python(state_dict, AdamStateKind::Adam)?;
        self.inner
            .borrow_mut()
            .load_state_dict(parameters, &state_dict)
            .map_err(to_python_error)
    }

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
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
        amsgrad: bool,
    ) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .set_config(
                AdamConfig::new(lr)
                    .with_betas(betas.0, betas.1)
                    .with_epsilon(eps)
                    .with_weight_decay(weight_decay)
                    .with_amsgrad(amsgrad),
            )
            .map_err(to_python_error)
    }

    #[getter]
    fn betas(&self) -> (f64, f64) {
        self.inner.borrow().config().betas()
    }

    #[getter]
    fn eps(&self) -> f64 {
        self.inner.borrow().config().epsilon()
    }

    #[getter]
    fn weight_decay(&self) -> f64 {
        self.inner.borrow().config().weight_decay()
    }

    #[getter]
    fn amsgrad(&self) -> bool {
        self.inner.borrow().config().is_amsgrad()
    }

    #[getter]
    fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    #[getter]
    fn state_size(&self) -> usize {
        self.inner.borrow().state_len()
    }

    fn __repr__(&self) -> String {
        let config = self.inner.borrow().config();
        format!(
            "Adam(lr={}, betas={:?}, eps={}, weight_decay={}, amsgrad={})",
            config.learning_rate(),
            config.betas(),
            config.epsilon(),
            config.weight_decay(),
            config.is_amsgrad()
        )
    }
}

/// AdamW with decoupled weight decay over an explicit parameter list.
#[pyclass(name = "AdamW", unsendable)]
pub(crate) struct PyAdamW {
    inner: Rc<RefCell<AdamW>>,
    parameters: Vec<ParameterSlot>,
    named_parameters: Option<tynx_train::ParameterStore>,
}

#[pymethods]
impl PyAdamW {
    #[new]
    #[pyo3(signature = (
        parameters,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.01,
        amsgrad=false,
    ))]
    fn new(
        parameters: &Bound<'_, PyAny>,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
        amsgrad: bool,
    ) -> PyResult<Self> {
        let collected = collect_parameters(parameters, "AdamW")?;
        let config = AdamWConfig::new(lr)
            .with_betas(betas.0, betas.1)
            .with_epsilon(eps)
            .with_weight_decay(weight_decay)
            .with_amsgrad(amsgrad);
        let inner = AdamW::with_config(config).map_err(to_python_error)?;
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
        record_optimizer_step(Rc::new(CapturedAdamW {
            inner: self.inner.clone(),
            parameters: self.parameters.clone(),
        }))?;
        self.inner
            .borrow_mut()
            .step_slots(&self.parameters)
            .map(|_| ())
            .map_err(to_python_error)
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let parameters = require_named_parameters(&self.named_parameters, "AdamW")?;
        let state_dict = self.inner.borrow().state_dict(parameters);
        state::adam_to_python(py, &state_dict)
    }

    fn load_state_dict(&mut self, state_dict: &Bound<'_, PyAny>) -> PyResult<()> {
        let parameters = require_named_parameters(&self.named_parameters, "AdamW")?;
        let state_dict = state::adam_from_python(state_dict, AdamStateKind::AdamW)?;
        self.inner
            .borrow_mut()
            .load_state_dict(parameters, &state_dict)
            .map_err(to_python_error)
    }

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
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
        amsgrad: bool,
    ) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .set_config(
                AdamWConfig::new(lr)
                    .with_betas(betas.0, betas.1)
                    .with_epsilon(eps)
                    .with_weight_decay(weight_decay)
                    .with_amsgrad(amsgrad),
            )
            .map_err(to_python_error)
    }

    #[getter]
    fn betas(&self) -> (f64, f64) {
        self.inner.borrow().config().betas()
    }

    #[getter]
    fn eps(&self) -> f64 {
        self.inner.borrow().config().epsilon()
    }

    #[getter]
    fn weight_decay(&self) -> f64 {
        self.inner.borrow().config().weight_decay()
    }

    #[getter]
    fn amsgrad(&self) -> bool {
        self.inner.borrow().config().is_amsgrad()
    }

    #[getter]
    fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    #[getter]
    fn state_size(&self) -> usize {
        self.inner.borrow().state_len()
    }

    fn __repr__(&self) -> String {
        let config = self.inner.borrow().config();
        format!(
            "AdamW(lr={}, betas={:?}, eps={}, weight_decay={}, amsgrad={})",
            config.learning_rate(),
            config.betas(),
            config.epsilon(),
            config.weight_decay(),
            config.is_amsgrad()
        )
    }
}

#[derive(Debug)]
struct CapturedAdam {
    inner: Rc<RefCell<Adam>>,
    parameters: Vec<ParameterSlot>,
}

impl CapturedOptimizer for CapturedAdam {
    fn step(&self) -> Result<()> {
        self.inner
            .borrow_mut()
            .step_slots(&self.parameters)
            .map(|_| ())
    }
}

#[derive(Debug)]
struct CapturedAdamW {
    inner: Rc<RefCell<AdamW>>,
    parameters: Vec<ParameterSlot>,
}

impl CapturedOptimizer for CapturedAdamW {
    fn step(&self) -> Result<()> {
        self.inner
            .borrow_mut()
            .step_slots(&self.parameters)
            .map(|_| ())
    }
}
