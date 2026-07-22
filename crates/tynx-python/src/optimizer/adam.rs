//! CPython Adam-family optimizers over stable Rust parameter slots.

use pyo3::{prelude::*, types::PyAny};
use tynx_train::{Adam, AdamConfig, AdamW, AdamWConfig, ParameterSlot};

use super::{collect_parameters, zero_grad};
use crate::to_python_error;

/// Adam with coupled L2 weight decay over an explicit parameter list.
#[pyclass(name = "Adam", unsendable)]
pub(crate) struct PyAdam {
    inner: Adam,
    parameters: Vec<ParameterSlot>,
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
        let parameters = collect_parameters(parameters, "Adam")?;
        let config = AdamConfig::new(lr)
            .with_betas(betas.0, betas.1)
            .with_epsilon(eps)
            .with_weight_decay(weight_decay)
            .with_amsgrad(amsgrad);
        let inner = Adam::with_config(config).map_err(to_python_error)?;
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

    #[getter]
    fn lr(&self) -> f64 {
        self.inner.config().learning_rate()
    }

    #[setter]
    fn set_lr(&mut self, value: f64) -> PyResult<()> {
        self.inner.set_learning_rate(value).map_err(to_python_error)
    }

    #[getter]
    fn learning_rate(&self) -> f64 {
        self.lr()
    }

    #[setter]
    fn set_learning_rate(&mut self, value: f64) -> PyResult<()> {
        self.set_lr(value)
    }

    #[getter]
    fn betas(&self) -> (f64, f64) {
        self.inner.config().betas()
    }

    #[getter]
    fn eps(&self) -> f64 {
        self.inner.config().epsilon()
    }

    #[getter]
    fn weight_decay(&self) -> f64 {
        self.inner.config().weight_decay()
    }

    #[getter]
    fn amsgrad(&self) -> bool {
        self.inner.config().is_amsgrad()
    }

    #[getter]
    fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    #[getter]
    fn state_size(&self) -> usize {
        self.inner.state_len()
    }

    fn __repr__(&self) -> String {
        let config = self.inner.config();
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
    inner: AdamW,
    parameters: Vec<ParameterSlot>,
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
        let parameters = collect_parameters(parameters, "AdamW")?;
        let config = AdamWConfig::new(lr)
            .with_betas(betas.0, betas.1)
            .with_epsilon(eps)
            .with_weight_decay(weight_decay)
            .with_amsgrad(amsgrad);
        let inner = AdamW::with_config(config).map_err(to_python_error)?;
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

    #[getter]
    fn lr(&self) -> f64 {
        self.inner.config().learning_rate()
    }

    #[setter]
    fn set_lr(&mut self, value: f64) -> PyResult<()> {
        self.inner.set_learning_rate(value).map_err(to_python_error)
    }

    #[getter]
    fn learning_rate(&self) -> f64 {
        self.lr()
    }

    #[setter]
    fn set_learning_rate(&mut self, value: f64) -> PyResult<()> {
        self.set_lr(value)
    }

    #[getter]
    fn betas(&self) -> (f64, f64) {
        self.inner.config().betas()
    }

    #[getter]
    fn eps(&self) -> f64 {
        self.inner.config().epsilon()
    }

    #[getter]
    fn weight_decay(&self) -> f64 {
        self.inner.config().weight_decay()
    }

    #[getter]
    fn amsgrad(&self) -> bool {
        self.inner.config().is_amsgrad()
    }

    #[getter]
    fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    #[getter]
    fn state_size(&self) -> usize {
        self.inner.state_len()
    }

    fn __repr__(&self) -> String {
        let config = self.inner.config();
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
