//! Adam-family optimizers.

use std::collections::HashMap;

use tynx_core::{DynTensor, Result, TynxError};

use crate::{ParamId, ParameterStore};

/// Configuration shared by Adam's adaptive-moment update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamConfig {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    amsgrad: bool,
}

impl AdamConfig {
    /// Start an Adam configuration with PyTorch-compatible defaults.
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            weight_decay: 0.0,
            amsgrad: false,
        }
    }

    /// Set the first- and second-moment coefficients.
    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set the numerical-stability term added to the denominator.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set coupled L2 weight decay for Adam.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enable or disable AMSGrad's maximum second-moment tracking.
    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }

    /// Return the learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Return `(beta1, beta2)`.
    pub fn betas(&self) -> (f64, f64) {
        (self.beta1, self.beta2)
    }

    /// Return epsilon.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Return the weight-decay coefficient.
    pub fn weight_decay(&self) -> f64 {
        self.weight_decay
    }

    /// Return whether AMSGrad is enabled.
    pub fn is_amsgrad(&self) -> bool {
        self.amsgrad
    }
}

/// Configuration for AdamW's decoupled weight-decay update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamWConfig {
    inner: AdamConfig,
}

impl AdamWConfig {
    /// Start an AdamW configuration with PyTorch-compatible defaults, including weight decay 0.01.
    pub fn new(learning_rate: f64) -> Self {
        Self {
            inner: AdamConfig::new(learning_rate).with_weight_decay(0.01),
        }
    }

    /// Set the first- and second-moment coefficients.
    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.inner = self.inner.with_betas(beta1, beta2);
        self
    }

    /// Set the numerical-stability term added to the denominator.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.inner = self.inner.with_epsilon(epsilon);
        self
    }

    /// Set decoupled weight decay.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.inner = self.inner.with_weight_decay(weight_decay);
        self
    }

    /// Enable or disable AMSGrad's maximum second-moment tracking.
    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.inner = self.inner.with_amsgrad(amsgrad);
        self
    }

    /// Return the learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.inner.learning_rate()
    }

    /// Return `(beta1, beta2)`.
    pub fn betas(&self) -> (f64, f64) {
        self.inner.betas()
    }

    /// Return epsilon.
    pub fn epsilon(&self) -> f64 {
        self.inner.epsilon()
    }

    /// Return the decoupled weight-decay coefficient.
    pub fn weight_decay(&self) -> f64 {
        self.inner.weight_decay()
    }

    /// Return whether AMSGrad is enabled.
    pub fn is_amsgrad(&self) -> bool {
        self.inner.is_amsgrad()
    }
}

#[derive(Debug, Clone)]
struct AdamState {
    structure_generation: u64,
    step: u64,
    first_moment: DynTensor,
    second_moment: DynTensor,
    max_second_moment: Option<DynTensor>,
}

#[derive(Debug, Clone, Copy)]
enum WeightDecayMode {
    Coupled,
    Decoupled,
}

#[derive(Debug)]
struct AdamEngine {
    config: AdamConfig,
    state: HashMap<ParamId, AdamState>,
}

impl AdamEngine {
    fn new(config: AdamConfig) -> Result<Self> {
        validate_config(config)?;
        Ok(Self {
            config,
            state: HashMap::new(),
        })
    }

    fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        let mut next = self.config;
        next.learning_rate = learning_rate;
        validate_config(next)?;
        self.config = next;
        Ok(())
    }

    fn step_for(&self, id: ParamId) -> Option<u64> {
        self.state.get(&id).map(|state| state.step)
    }

    fn step(&mut self, parameters: &ParameterStore, mode: WeightDecayMode) -> Result<usize> {
        let mut next_state = self.state.clone();
        let mut updates = Vec::new();

        for parameter in parameters.trainable() {
            let Some(gradient) = parameter.grad() else {
                continue;
            };
            let structure_generation = parameter.structure_generation();
            let weight = parameter.value().inner();
            let mut direction = gradient;
            if matches!(mode, WeightDecayMode::Coupled) && self.config.weight_decay != 0.0 {
                direction =
                    direction.add_broadcast(weight.clone().mul_scalar(self.config.weight_decay))?;
            }

            let previous = next_state
                .get(&parameter.id())
                .filter(|state| state.structure_generation == structure_generation);
            let (previous_first, previous_second, previous_max, previous_step) = match previous {
                Some(state) => (
                    state.first_moment.clone(),
                    state.second_moment.clone(),
                    state.max_second_moment.clone(),
                    state.step,
                ),
                None => (
                    zeros_like(&weight),
                    zeros_like(&weight),
                    self.config.amsgrad.then(|| zeros_like(&weight)),
                    0,
                ),
            };
            let step = previous_step.checked_add(1).ok_or_else(|| {
                TynxError::TypeMismatch(format!(
                    "Adam step counter exhausted for parameter {}",
                    parameter.id().get()
                ))
            })?;

            let first_moment = previous_first
                .mul_scalar(self.config.beta1)
                .add_broadcast(direction.clone().mul_scalar(1.0 - self.config.beta1))?;
            let squared = direction.clone().mul_broadcast(direction)?;
            let second_moment = previous_second
                .mul_scalar(self.config.beta2)
                .add_broadcast(squared.mul_scalar(1.0 - self.config.beta2))?;

            let (denominator_moment, max_second_moment) = if self.config.amsgrad {
                let previous_max = previous_max.ok_or_else(|| {
                    TynxError::TypeMismatch(format!(
                        "AMSGrad maximum moment is missing for parameter {}",
                        parameter.id().get()
                    ))
                })?;
                let maximum = previous_max.max_broadcast(second_moment.clone())?;
                (maximum.clone(), Some(maximum))
            } else {
                (second_moment.clone(), None)
            };

            let bias_correction1 = 1.0 - self.config.beta1.powf(step as f64);
            let bias_correction2 = 1.0 - self.config.beta2.powf(step as f64);
            let first_hat = first_moment.clone().div_scalar(bias_correction1);
            let second_hat = denominator_moment.div_scalar(bias_correction2);
            let adaptive_update = first_hat
                .div_broadcast(second_hat.sqrt().add_scalar(self.config.epsilon))?
                .mul_scalar(self.config.learning_rate);

            let decayed_weight =
                if matches!(mode, WeightDecayMode::Decoupled) && self.config.weight_decay != 0.0 {
                    weight.mul_scalar(1.0 - self.config.learning_rate * self.config.weight_decay)
                } else {
                    weight
                };
            let updated = decayed_weight.sub_broadcast(adaptive_update)?.to_autodiff();
            updates.push((parameter.clone(), updated));
            next_state.insert(
                parameter.id(),
                AdamState {
                    structure_generation,
                    step,
                    first_moment,
                    second_moment,
                    max_second_moment,
                },
            );
        }

        let updated_count = updates.len();
        for (parameter, value) in updates {
            parameter.replace_value(value)?;
        }
        self.state = next_state;
        Ok(updated_count)
    }
}

/// Adam with coupled L2 weight decay.
#[derive(Debug)]
pub struct Adam {
    engine: AdamEngine,
}

impl Adam {
    /// Create Adam with the given learning rate and standard defaults.
    pub fn new(learning_rate: f64) -> Result<Self> {
        Self::with_config(AdamConfig::new(learning_rate))
    }

    /// Create Adam from a complete configuration.
    pub fn with_config(config: AdamConfig) -> Result<Self> {
        Ok(Self {
            engine: AdamEngine::new(config)?,
        })
    }

    /// Return the current configuration.
    pub fn config(&self) -> AdamConfig {
        self.engine.config
    }

    /// Change the learning rate without discarding moment state.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        self.engine.set_learning_rate(learning_rate)
    }

    /// Return the number of parameter-state entries.
    pub fn state_len(&self) -> usize {
        self.engine.state.len()
    }

    /// Return the current step counter for a parameter identity.
    pub fn step_for(&self, id: ParamId) -> Option<u64> {
        self.engine.step_for(id)
    }

    /// Update every unique trainable slot that currently has a gradient.
    pub fn step(&mut self, parameters: &ParameterStore) -> Result<usize> {
        self.engine.step(parameters, WeightDecayMode::Coupled)
    }
}

/// AdamW with decoupled weight decay.
#[derive(Debug)]
pub struct AdamW {
    engine: AdamEngine,
}

impl AdamW {
    /// Create AdamW with standard defaults, including weight decay 0.01.
    pub fn new(learning_rate: f64) -> Result<Self> {
        Self::with_config(AdamWConfig::new(learning_rate))
    }

    /// Create AdamW from a complete configuration.
    pub fn with_config(config: AdamWConfig) -> Result<Self> {
        Ok(Self {
            engine: AdamEngine::new(config.inner)?,
        })
    }

    /// Return the current configuration.
    pub fn config(&self) -> AdamWConfig {
        AdamWConfig {
            inner: self.engine.config,
        }
    }

    /// Change the learning rate without discarding moment state.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        self.engine.set_learning_rate(learning_rate)
    }

    /// Return the number of parameter-state entries.
    pub fn state_len(&self) -> usize {
        self.engine.state.len()
    }

    /// Return the current step counter for a parameter identity.
    pub fn step_for(&self, id: ParamId) -> Option<u64> {
        self.engine.step_for(id)
    }

    /// Update every unique trainable slot that currently has a gradient.
    pub fn step(&mut self, parameters: &ParameterStore) -> Result<usize> {
        self.engine.step(parameters, WeightDecayMode::Decoupled)
    }
}

fn zeros_like(tensor: &DynTensor) -> DynTensor {
    tensor.clone().mul_scalar(0.0)
}

fn validate_config(config: AdamConfig) -> Result<()> {
    validate_nonnegative_finite("learning rate", config.learning_rate)?;
    validate_beta("beta1", config.beta1)?;
    validate_beta("beta2", config.beta2)?;
    validate_nonnegative_finite("epsilon", config.epsilon)?;
    validate_nonnegative_finite("weight decay", config.weight_decay)?;
    Ok(())
}

fn validate_beta(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        return Err(TynxError::TypeMismatch(format!(
            "Adam {name} must be finite and in [0, 1), got {value}"
        )));
    }
    Ok(())
}

fn validate_nonnegative_finite(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(TynxError::TypeMismatch(format!(
            "Adam {name} must be finite and non-negative, got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, TensorData};

    use super::*;
    use crate::{ParameterSlot, backward, loss::mse};

    fn tensor(values: Vec<f32>, dims: &[usize], device: &Device) -> DynTensor {
        DynTensor::from_data(TensorData::new(values, dims.to_vec()), dims.len(), device).unwrap()
    }

    fn scalar_parameter(
        name: &str,
        value: f32,
        device: &Device,
    ) -> (ParameterSlot, ParameterStore) {
        let parameter = ParameterSlot::new(None, tensor(vec![value], &[1], device), true).unwrap();
        let mut store = ParameterStore::new();
        store.insert(name, parameter.clone()).unwrap();
        (parameter, store)
    }

    fn set_gradient(parameter: &ParameterSlot, store: &ParameterStore, multiplier: f64) {
        store.zero_grad();
        let loss = parameter.read().mul_scalar(multiplier);
        backward(&loss, store).unwrap();
    }

    fn scalar_value(parameter: &ParameterSlot) -> f32 {
        parameter.value().into_data().iter::<f32>().next().unwrap()
    }

    #[test]
    fn exposes_expected_defaults_and_validates_configuration() {
        let adam = Adam::new(0.001).unwrap();
        assert_eq!(adam.config().betas(), (0.9, 0.999));
        assert_eq!(adam.config().epsilon(), 1.0e-8);
        assert_eq!(adam.config().weight_decay(), 0.0);

        let adamw = AdamW::new(0.001).unwrap();
        assert_eq!(adamw.config().weight_decay(), 0.01);

        assert!(Adam::new(-0.1).is_err());
        assert!(Adam::with_config(AdamConfig::new(0.1).with_betas(1.0, 0.9)).is_err());
        assert!(Adam::with_config(AdamConfig::new(0.1).with_betas(0.9, f64::NAN)).is_err());
        assert!(Adam::with_config(AdamConfig::new(0.1).with_epsilon(-1.0)).is_err());
        assert!(
            AdamW::with_config(AdamWConfig::new(0.1).with_weight_decay(f64::INFINITY)).is_err()
        );
    }

    #[test]
    fn adam_bias_correction_matches_constant_gradient_reference() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = scalar_parameter("weight", 2.0, &device);
        let mut adam = Adam::new(0.1).unwrap();

        for expected in [1.9, 1.8] {
            set_gradient(&parameter, &store, 2.0);
            assert_eq!(adam.step(&store).unwrap(), 1);
            assert!((scalar_value(&parameter) - expected).abs() < 1.0e-6);
        }

        assert_eq!(adam.step_for(parameter.id()), Some(2));
        assert_eq!(adam.state_len(), 1);
        assert!(parameter.grad().is_some());
        assert_eq!(parameter.structure_generation(), 0);
    }

    #[test]
    fn adam_and_adamw_apply_distinct_weight_decay() {
        let device = Device::autodiff(Device::default());
        let (adam_parameter, adam_store) = scalar_parameter("weight", 2.0, &device);
        let (adamw_parameter, adamw_store) = scalar_parameter("weight", 2.0, &device);
        set_gradient(&adam_parameter, &adam_store, 1.0);
        set_gradient(&adamw_parameter, &adamw_store, 1.0);
        let mut adam = Adam::with_config(
            AdamConfig::new(0.1)
                .with_betas(0.0, 0.0)
                .with_epsilon(0.0)
                .with_weight_decay(0.5),
        )
        .unwrap();
        let mut adamw = AdamW::with_config(
            AdamWConfig::new(0.1)
                .with_betas(0.0, 0.0)
                .with_epsilon(0.0)
                .with_weight_decay(0.5),
        )
        .unwrap();

        adam.step(&adam_store).unwrap();
        adamw.step(&adamw_store).unwrap();

        assert!((scalar_value(&adam_parameter) - 1.9).abs() < 1.0e-6);
        assert!((scalar_value(&adamw_parameter) - 1.8).abs() < 1.0e-6);
    }

    #[test]
    fn amsgrad_uses_the_largest_second_moment() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = scalar_parameter("weight", 2.0, &device);
        let mut adam = Adam::with_config(
            AdamConfig::new(0.1)
                .with_betas(0.0, 0.0)
                .with_epsilon(0.0)
                .with_amsgrad(true),
        )
        .unwrap();

        set_gradient(&parameter, &store, 2.0);
        adam.step(&store).unwrap();
        set_gradient(&parameter, &store, 0.5);
        adam.step(&store).unwrap();

        assert!((scalar_value(&parameter) - 1.875).abs() < 1.0e-6);
    }

    #[test]
    fn structural_rebind_resets_moments_and_step() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = scalar_parameter("weight", 2.0, &device);
        let mut adam = Adam::new(0.1).unwrap();

        set_gradient(&parameter, &store, 2.0);
        adam.step(&store).unwrap();
        parameter
            .rebind(tensor(vec![2.0], &[1], &device), true)
            .unwrap();
        set_gradient(&parameter, &store, 2.0);
        adam.step(&store).unwrap();

        assert_eq!(adam.step_for(parameter.id()), Some(1));
        assert!((scalar_value(&parameter) - 1.9).abs() < 1.0e-6);
    }

    #[test]
    fn mutable_learning_rate_retains_state() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = scalar_parameter("weight", 2.0, &device);
        let mut adamw = AdamW::new(0.1).unwrap();
        set_gradient(&parameter, &store, 1.0);
        adamw.step(&store).unwrap();

        adamw.set_learning_rate(0.01).unwrap();
        assert_eq!(adamw.config().learning_rate(), 0.01);
        assert_eq!(adamw.step_for(parameter.id()), Some(1));
        assert!(adamw.set_learning_rate(-1.0).is_err());
        assert_eq!(adamw.config().learning_rate(), 0.01);
    }

    #[test]
    fn adam_learns_a_rust_authored_linear_model() {
        let device = Device::autodiff(Device::default());
        let weight = ParameterSlot::new(None, tensor(vec![0.0], &[1], &device), true).unwrap();
        let bias = ParameterSlot::new(None, tensor(vec![0.0], &[1], &device), true).unwrap();
        let mut store = ParameterStore::new();
        store.insert("weight", weight.clone()).unwrap();
        store.insert("bias", bias.clone()).unwrap();
        let input = tensor(vec![-1.0, 0.0, 1.0, 2.0], &[4], &device);
        let target = tensor(vec![-1.0, 1.0, 3.0, 5.0], &[4], &device);
        let mut adam = Adam::new(0.05).unwrap();

        for _ in 0..300 {
            store.zero_grad();
            let prediction = weight
                .read()
                .mul_broadcast(input.clone())
                .unwrap()
                .add_broadcast(bias.read())
                .unwrap();
            let loss = mse(prediction, target.clone()).unwrap();
            backward(&loss, &store).unwrap();
            assert_eq!(adam.step(&store).unwrap(), 2);
        }

        assert!((scalar_value(&weight) - 2.0).abs() < 1.0e-3);
        assert!((scalar_value(&bias) - 1.0).abs() < 1.0e-3);
        assert_eq!(adam.step_for(weight.id()), Some(300));
        assert_eq!(adam.step_for(bias.id()), Some(300));
        assert_eq!(weight.structure_generation(), 0);
        assert_eq!(bias.structure_generation(), 0);
    }
}
