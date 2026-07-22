//! Optimizers over stable parameter slots.

mod adam;

pub use adam::{Adam, AdamConfig, AdamW, AdamWConfig};

use std::collections::HashMap;

use tynx_core::{DynTensor, Result, TynxError};

use crate::{ParamId, ParameterStore};

/// Configuration for stochastic gradient descent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgdConfig {
    learning_rate: f64,
    momentum: f64,
    dampening: f64,
    weight_decay: f64,
    nesterov: bool,
}

impl SgdConfig {
    /// Start an SGD configuration with the given learning rate.
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }

    /// Set the momentum coefficient.
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set momentum dampening.
    pub fn with_dampening(mut self, dampening: f64) -> Self {
        self.dampening = dampening;
        self
    }

    /// Set coupled L2 weight decay.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enable or disable Nesterov momentum.
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Return the learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Return the momentum coefficient.
    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    /// Return momentum dampening.
    pub fn dampening(&self) -> f64 {
        self.dampening
    }

    /// Return coupled L2 weight decay.
    pub fn weight_decay(&self) -> f64 {
        self.weight_decay
    }

    /// Return whether Nesterov momentum is enabled.
    pub fn is_nesterov(&self) -> bool {
        self.nesterov
    }
}

/// Stochastic gradient descent with optional momentum, dampening, weight decay, and Nesterov.
///
/// Updates run entirely on inner non-autodiff tensors. `step` reads persistent gradients but does
/// not clear them; call [`ParameterStore::zero_grad`] explicitly at the desired accumulation
/// boundary.
#[derive(Debug)]
pub struct Sgd {
    config: SgdConfig,
    momentum_buffers: HashMap<ParamId, (u64, DynTensor)>,
}

impl Sgd {
    /// Create plain SGD with the given learning rate.
    pub fn new(learning_rate: f64) -> Result<Self> {
        Self::with_config(SgdConfig::new(learning_rate))
    }

    /// Create SGD from a complete configuration.
    pub fn with_config(config: SgdConfig) -> Result<Self> {
        validate_config(config)?;
        Ok(Self {
            config,
            momentum_buffers: HashMap::new(),
        })
    }

    /// Return the current configuration.
    pub fn config(&self) -> SgdConfig {
        self.config
    }

    /// Change the learning rate without discarding momentum state.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        let mut next = self.config;
        next.learning_rate = learning_rate;
        validate_config(next)?;
        self.config = next;
        Ok(())
    }

    /// Return the number of parameter momentum buffers.
    pub fn state_len(&self) -> usize {
        self.momentum_buffers.len()
    }

    /// Update every unique trainable slot that currently has a gradient.
    ///
    /// Returns the number of updated slots. Frozen slots and parameters without gradients are
    /// skipped. All update tensors are computed before publication, providing one serialized
    /// multi-parameter commit in the initial single-threaded runtime.
    pub fn step(&mut self, parameters: &ParameterStore) -> Result<usize> {
        let mut next_momentum = self.momentum_buffers.clone();
        let mut updates = Vec::new();

        for parameter in parameters.trainable() {
            let Some(gradient) = parameter.grad() else {
                continue;
            };
            let structure_generation = parameter.structure_generation();
            let weight = parameter.value().inner();
            let mut direction = if self.config.weight_decay == 0.0 {
                gradient
            } else {
                gradient.add_broadcast(weight.clone().mul_scalar(self.config.weight_decay))?
            };

            if self.config.momentum != 0.0 {
                let buffer = match next_momentum.get(&parameter.id()) {
                    Some((generation, previous)) if *generation == structure_generation => previous
                        .clone()
                        .mul_scalar(self.config.momentum)
                        .add_broadcast(direction.clone().mul_scalar(1.0 - self.config.dampening))?,
                    _ => direction.clone(),
                };
                next_momentum.insert(parameter.id(), (structure_generation, buffer.clone()));
                direction = if self.config.nesterov {
                    direction.add_broadcast(buffer.mul_scalar(self.config.momentum))?
                } else {
                    buffer
                };
            }

            let updated = weight
                .sub_broadcast(direction.mul_scalar(self.config.learning_rate))?
                .to_autodiff();
            updates.push((parameter.clone(), updated));
        }

        let updated_count = updates.len();
        for (parameter, value) in updates {
            parameter.replace_value(value)?;
        }
        self.momentum_buffers = next_momentum;
        Ok(updated_count)
    }
}

fn validate_config(config: SgdConfig) -> Result<()> {
    validate_nonnegative_finite("learning rate", config.learning_rate)?;
    validate_nonnegative_finite("momentum", config.momentum)?;
    validate_nonnegative_finite("dampening", config.dampening)?;
    validate_nonnegative_finite("weight decay", config.weight_decay)?;
    if config.nesterov && (config.momentum <= 0.0 || config.dampening != 0.0) {
        return Err(TynxError::TypeMismatch(
            "Nesterov SGD requires positive momentum and zero dampening".to_string(),
        ));
    }
    Ok(())
}

fn validate_nonnegative_finite(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(TynxError::TypeMismatch(format!(
            "SGD {name} must be finite and non-negative, got {value}"
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
        let leaf = parameter.read();
        let loss = leaf.mul_scalar(multiplier);
        backward(&loss, store).unwrap();
    }

    fn values(tensor: DynTensor) -> Vec<f32> {
        tensor.into_data().iter::<f32>().collect()
    }

    #[test]
    fn plain_sgd_updates_off_tape_and_preserves_gradient() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = scalar_parameter("weight", 2.0, &device);
        set_gradient(&parameter, &store, 3.0);
        let mut sgd = Sgd::new(0.1).unwrap();

        assert_eq!(sgd.step(&store).unwrap(), 1);

        assert_eq!(values(parameter.value()), [1.7]);
        assert!(!parameter.value().is_require_grad());
        assert_eq!(values(parameter.grad().unwrap()), [3.0]);
        assert_eq!(parameter.value_generation(), 1);
        assert_eq!(parameter.structure_generation(), 0);
        assert_eq!(sgd.state_len(), 0);
    }

    #[test]
    fn applies_coupled_weight_decay() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = scalar_parameter("weight", 2.0, &device);
        set_gradient(&parameter, &store, 3.0);
        let mut sgd = Sgd::with_config(SgdConfig::new(0.1).with_weight_decay(0.5)).unwrap();

        sgd.step(&store).unwrap();

        assert_eq!(values(parameter.value()), [1.6]);
    }

    #[test]
    fn momentum_matches_pytorch_first_and_later_step_semantics() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = scalar_parameter("weight", 2.0, &device);
        let mut sgd =
            Sgd::with_config(SgdConfig::new(0.1).with_momentum(0.9).with_dampening(0.5)).unwrap();

        set_gradient(&parameter, &store, 2.0);
        sgd.step(&store).unwrap();
        assert_eq!(values(parameter.value()), [1.8]);

        store.zero_grad();
        set_gradient(&parameter, &store, 2.0);
        sgd.step(&store).unwrap();

        let value = values(parameter.value())[0];
        assert!((value - 1.52).abs() < 1.0e-6);
        assert_eq!(sgd.state_len(), 1);
    }

    #[test]
    fn nesterov_uses_lookahead_direction() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = scalar_parameter("weight", 2.0, &device);
        set_gradient(&parameter, &store, 2.0);
        let mut sgd =
            Sgd::with_config(SgdConfig::new(0.1).with_momentum(0.9).with_nesterov(true)).unwrap();

        sgd.step(&store).unwrap();

        let value = values(parameter.value())[0];
        assert!((value - 1.62).abs() < 1.0e-6);
    }

    #[test]
    fn skips_missing_gradients_frozen_slots_and_tied_aliases() {
        let device = Device::autodiff(Device::default());
        let active = ParameterSlot::new(None, tensor(vec![2.0], &[1], &device), true).unwrap();
        let missing = ParameterSlot::new(None, tensor(vec![4.0], &[1], &device), true).unwrap();
        let frozen = ParameterSlot::new(None, tensor(vec![6.0], &[1], &device), false).unwrap();
        let mut store = ParameterStore::new();
        store.insert("active", active.clone()).unwrap();
        store.insert("active_alias", active.clone()).unwrap();
        store.insert("missing", missing.clone()).unwrap();
        store.insert("frozen", frozen.clone()).unwrap();
        set_gradient(&active, &store, 1.0);
        let mut sgd = Sgd::new(0.1).unwrap();

        assert_eq!(sgd.step(&store).unwrap(), 1);
        assert_eq!(values(active.value()), [1.9]);
        assert_eq!(values(missing.value()), [4.0]);
        assert_eq!(values(frozen.value()), [6.0]);
    }

    #[test]
    fn validates_configuration_and_mutable_learning_rate() {
        assert!(Sgd::new(-0.1).is_err());
        assert!(Sgd::new(f64::NAN).is_err());
        assert!(Sgd::with_config(SgdConfig::new(0.1).with_momentum(-0.1)).is_err());
        assert!(Sgd::with_config(SgdConfig::new(0.1).with_weight_decay(f64::INFINITY)).is_err());
        assert!(Sgd::with_config(SgdConfig::new(0.1).with_nesterov(true)).is_err());
        assert!(
            Sgd::with_config(
                SgdConfig::new(0.1)
                    .with_momentum(0.9)
                    .with_dampening(0.1)
                    .with_nesterov(true)
            )
            .is_err()
        );

        let mut sgd = Sgd::new(0.1).unwrap();
        sgd.set_learning_rate(0.01).unwrap();
        assert_eq!(sgd.config().learning_rate(), 0.01);
        assert!(sgd.set_learning_rate(-1.0).is_err());
        assert_eq!(sgd.config().learning_rate(), 0.01);
    }

    #[test]
    fn structural_changes_reset_momentum_for_that_slot() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = scalar_parameter("weight", 2.0, &device);
        let mut sgd = Sgd::with_config(SgdConfig::new(0.1).with_momentum(0.9)).unwrap();

        set_gradient(&parameter, &store, 2.0);
        sgd.step(&store).unwrap();
        parameter
            .rebind(tensor(vec![2.0], &[1], &device), true)
            .unwrap();
        store.zero_grad();
        set_gradient(&parameter, &store, 2.0);
        sgd.step(&store).unwrap();

        assert_eq!(values(parameter.value()), [1.8]);
        assert_eq!(sgd.state_len(), 1);
    }

    #[test]
    fn learns_a_rust_authored_linear_model() {
        let device = Device::autodiff(Device::default());
        let weight = ParameterSlot::new(None, tensor(vec![0.0], &[1], &device), true).unwrap();
        let bias = ParameterSlot::new(None, tensor(vec![0.0], &[1], &device), true).unwrap();
        let mut store = ParameterStore::new();
        store.insert("weight", weight.clone()).unwrap();
        store.insert("bias", bias.clone()).unwrap();
        let input = tensor(vec![-1.0, 0.0, 1.0, 2.0], &[4], &device);
        let target = tensor(vec![-1.0, 1.0, 3.0, 5.0], &[4], &device);
        let mut sgd = Sgd::new(0.1).unwrap();

        for _ in 0..100 {
            store.zero_grad();
            let prediction = weight
                .read()
                .mul_broadcast(input.clone())
                .unwrap()
                .add_broadcast(bias.read())
                .unwrap();
            let loss = mse(prediction, target.clone()).unwrap();
            backward(&loss, &store).unwrap();
            assert_eq!(sgd.step(&store).unwrap(), 2);
        }

        let learned_weight = values(weight.value())[0];
        let learned_bias = values(bias.value())[0];
        assert!((learned_weight - 2.0).abs() < 1.0e-3);
        assert!((learned_bias - 1.0).abs() < 1.0e-3);
        assert_eq!(weight.value_generation(), 100);
        assert_eq!(bias.value_generation(), 100);
        assert_eq!(weight.structure_generation(), 0);
        assert_eq!(bias.structure_generation(), 0);
    }
}
