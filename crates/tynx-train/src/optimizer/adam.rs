//! Adam-family optimizers.

use std::collections::{BTreeMap, HashMap, HashSet};

use tynx_core::{DynTensor, Result, TynxError};

use crate::{ParamId, ParameterSlot, ParameterStore};

use super::{
    trainable_by_name, validate_parameter_name_match, validate_state_names, validate_state_tensor,
};

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

/// Portable adaptive-moment state for one Adam or AdamW parameter.
///
/// Process-local parameter identity and structure generation are excluded so a checkpoint can be
/// rebound to a freshly constructed model after names are resolved by the caller.
#[derive(Debug, Clone)]
pub struct AdamParameterState {
    step: u64,
    first_moment: DynTensor,
    second_moment: DynTensor,
    max_second_moment: Option<DynTensor>,
}

impl AdamParameterState {
    /// Construct a portable Adam-family state entry.
    pub fn new(
        step: u64,
        first_moment: DynTensor,
        second_moment: DynTensor,
        max_second_moment: Option<DynTensor>,
    ) -> Self {
        Self {
            step,
            first_moment: first_moment.detach(),
            second_moment: second_moment.detach(),
            max_second_moment: max_second_moment.map(DynTensor::detach),
        }
    }

    /// Return the completed update count for this parameter.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Return the detached first moment.
    pub fn first_moment(&self) -> DynTensor {
        self.first_moment.clone()
    }

    /// Return the detached second moment.
    pub fn second_moment(&self) -> DynTensor {
        self.second_moment.clone()
    }

    /// Return the detached AMSGrad maximum second moment, when present.
    pub fn max_second_moment(&self) -> Option<DynTensor> {
        self.max_second_moment.clone()
    }
}

/// Name-keyed, portable Adam-family configuration and adaptive state.
#[derive(Debug, Clone)]
pub struct AdamStateDict {
    kind: AdamStateKind,
    config: AdamConfig,
    parameter_names: Vec<String>,
    state: BTreeMap<String, AdamParameterState>,
}

/// Adam-family update rule recorded in a portable state dictionary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdamStateKind {
    /// Coupled L2 weight decay used by Adam.
    Adam,
    /// Decoupled weight decay used by AdamW.
    AdamW,
}

impl AdamStateDict {
    /// Construct and structurally validate a portable Adam-family state dictionary.
    pub fn new(
        kind: AdamStateKind,
        config: AdamConfig,
        parameter_names: Vec<String>,
        state: BTreeMap<String, AdamParameterState>,
    ) -> Result<Self> {
        validate_config(config)?;
        let parameter_names = validate_state_names(parameter_names, state.keys(), "Adam")?;
        for (name, parameter_state) in &state {
            if parameter_state.step == 0 {
                return Err(TynxError::TypeMismatch(format!(
                    "Adam state step for '{name}' must be positive"
                )));
            }
            if config.amsgrad != parameter_state.max_second_moment.is_some() {
                return Err(TynxError::TypeMismatch(format!(
                    "Adam state maximum second moment for '{name}' must match amsgrad={} configuration",
                    config.amsgrad
                )));
            }
        }
        Ok(Self {
            kind,
            config,
            parameter_names,
            state,
        })
    }

    /// Return whether the payload belongs to Adam or AdamW.
    pub fn kind(&self) -> AdamStateKind {
        self.kind
    }

    /// Return the serialized Adam-family configuration.
    pub fn config(&self) -> AdamConfig {
        self.config
    }

    /// Return stable parameter names in lexical order, including entries without adaptive state.
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    /// Iterate allocated adaptive state by stable parameter name.
    pub fn state(&self) -> impl ExactSizeIterator<Item = (&str, &AdamParameterState)> {
        self.state
            .iter()
            .map(|(name, state)| (name.as_str(), state))
    }
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
        self.set_config(next)
    }

    fn set_config(&mut self, config: AdamConfig) -> Result<()> {
        validate_config(config)?;
        if config.amsgrad != self.config.amsgrad {
            for state in self.state.values_mut() {
                state.max_second_moment =
                    config.amsgrad.then(|| state.second_moment.clone().detach());
            }
        }
        self.config = config;
        Ok(())
    }

    fn step_for(&self, id: ParamId) -> Option<u64> {
        self.state.get(&id).map(|state| state.step)
    }

    fn state_for(&self, parameter: &ParameterSlot) -> Option<AdamParameterState> {
        self.state
            .get(&parameter.id())
            .filter(|state| state.structure_generation == parameter.structure_generation())
            .map(|state| {
                AdamParameterState::new(
                    state.step,
                    state.first_moment.clone().detach(),
                    state.second_moment.clone().detach(),
                    state.max_second_moment.clone().map(DynTensor::detach),
                )
            })
    }

    fn state_dict(&self, parameters: &ParameterStore, kind: AdamStateKind) -> AdamStateDict {
        let named = parameters
            .named()
            .filter(|(_, parameter)| parameter.contract().trainable())
            .collect::<Vec<_>>();
        let parameter_names = named.iter().map(|(name, _)| (*name).to_string()).collect();
        let state = named
            .into_iter()
            .filter_map(|(name, parameter)| {
                self.state_for(parameter)
                    .map(|state| (name.to_string(), state))
            })
            .collect();
        AdamStateDict::new(kind, self.config, parameter_names, state)
            .expect("live Adam configuration and ParameterStore names are valid")
    }

    fn load_state_dict(
        &mut self,
        parameters: &ParameterStore,
        state_dict: &AdamStateDict,
        expected_kind: AdamStateKind,
    ) -> Result<()> {
        if state_dict.kind != expected_kind {
            return Err(TynxError::TypeMismatch(format!(
                "cannot load {:?} state into {:?} optimizer",
                state_dict.kind, expected_kind
            )));
        }
        let current = trainable_by_name(parameters);
        validate_parameter_name_match(
            current.keys().copied(),
            &state_dict.parameter_names,
            "Adam",
        )?;
        let states = current
            .into_iter()
            .map(|(name, parameter)| (parameter.clone(), state_dict.state.get(name).cloned()))
            .collect::<Vec<_>>();
        let mut replacement = Self::new(state_dict.config)?;
        replacement.replace_slot_states(&states)?;
        *self = replacement;
        Ok(())
    }

    fn replace_slot_states(
        &mut self,
        states: &[(ParameterSlot, Option<AdamParameterState>)],
    ) -> Result<()> {
        let mut seen = HashSet::new();
        let mut replacement = HashMap::new();
        for (parameter, state) in states {
            if !seen.insert(parameter.id()) {
                return Err(TynxError::TypeMismatch(format!(
                    "duplicate Adam state destination for parameter {}",
                    parameter.id().get()
                )));
            }
            let Some(state) = state else {
                continue;
            };
            if state.step == 0 {
                return Err(TynxError::TypeMismatch(
                    "Adam state step must be positive".to_string(),
                ));
            }
            validate_state_tensor("Adam first moment", &state.first_moment, parameter)?;
            validate_state_tensor("Adam second moment", &state.second_moment, parameter)?;
            if let Some(maximum) = &state.max_second_moment {
                validate_state_tensor("Adam maximum second moment", maximum, parameter)?;
            }
            if self.config.amsgrad != state.max_second_moment.is_some() {
                return Err(TynxError::TypeMismatch(format!(
                    "Adam state maximum second moment presence must match amsgrad={} configuration",
                    self.config.amsgrad
                )));
            }
            replacement.insert(
                parameter.id(),
                AdamState {
                    structure_generation: parameter.structure_generation(),
                    step: state.step,
                    first_moment: state.first_moment.clone().detach(),
                    second_moment: state.second_moment.clone().detach(),
                    max_second_moment: state.max_second_moment.clone().map(DynTensor::detach),
                },
            );
        }
        self.state = replacement;
        Ok(())
    }

    fn step_parameters<'a>(
        &mut self,
        parameters: impl IntoIterator<Item = &'a ParameterSlot>,
        mode: WeightDecayMode,
    ) -> Result<usize> {
        let mut next_state = self.state.clone();
        let mut updates = Vec::new();

        for parameter in parameters {
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
        ParameterSlot::replace_values_atomic(&updates)?;
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

    /// Replace the live parameter-group configuration while retaining compatible moments.
    pub fn set_config(&mut self, config: AdamConfig) -> Result<()> {
        self.engine.set_config(config)
    }

    /// Return the number of parameter-state entries.
    pub fn state_len(&self) -> usize {
        self.engine.state.len()
    }

    /// Return the current step counter for a parameter identity.
    pub fn step_for(&self, id: ParamId) -> Option<u64> {
        self.engine.step_for(id)
    }

    /// Snapshot the current-generation adaptive state for one parameter.
    pub fn state_for(&self, parameter: &ParameterSlot) -> Option<AdamParameterState> {
        self.engine.state_for(parameter)
    }

    /// Export configuration and adaptive state by canonical parameter name.
    pub fn state_dict(&self, parameters: &ParameterStore) -> AdamStateDict {
        self.engine.state_dict(parameters, AdamStateKind::Adam)
    }

    /// Restore configuration and adaptive state by canonical name onto fresh runtime slots.
    pub fn load_state_dict(
        &mut self,
        parameters: &ParameterStore,
        state_dict: &AdamStateDict,
    ) -> Result<()> {
        self.engine
            .load_state_dict(parameters, state_dict, AdamStateKind::Adam)
    }

    /// Atomically replace all adaptive state for a runtime parameter list.
    pub fn replace_slot_states(
        &mut self,
        states: &[(ParameterSlot, Option<AdamParameterState>)],
    ) -> Result<()> {
        self.engine.replace_slot_states(states)
    }

    /// Update every unique trainable slot that currently has a gradient.
    pub fn step(&mut self, parameters: &ParameterStore) -> Result<usize> {
        self.engine
            .step_parameters(parameters.trainable(), WeightDecayMode::Coupled)
    }

    /// Update a runtime list of parameter slots without requiring persisted state names.
    ///
    /// Repeated/tied identities are updated once in first-occurrence order. Frozen slots and slots
    /// without gradients are skipped, matching [`Self::step`].
    pub fn step_slots(&mut self, parameters: &[ParameterSlot]) -> Result<usize> {
        self.engine
            .step_parameters(unique_trainable(parameters), WeightDecayMode::Coupled)
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

    /// Replace the live parameter-group configuration while retaining compatible moments.
    pub fn set_config(&mut self, config: AdamWConfig) -> Result<()> {
        self.engine.set_config(config.inner)
    }

    /// Return the number of parameter-state entries.
    pub fn state_len(&self) -> usize {
        self.engine.state.len()
    }

    /// Return the current step counter for a parameter identity.
    pub fn step_for(&self, id: ParamId) -> Option<u64> {
        self.engine.step_for(id)
    }

    /// Snapshot the current-generation adaptive state for one parameter.
    pub fn state_for(&self, parameter: &ParameterSlot) -> Option<AdamParameterState> {
        self.engine.state_for(parameter)
    }

    /// Export configuration and adaptive state by canonical parameter name.
    pub fn state_dict(&self, parameters: &ParameterStore) -> AdamStateDict {
        self.engine.state_dict(parameters, AdamStateKind::AdamW)
    }

    /// Restore configuration and adaptive state by canonical name onto fresh runtime slots.
    pub fn load_state_dict(
        &mut self,
        parameters: &ParameterStore,
        state_dict: &AdamStateDict,
    ) -> Result<()> {
        self.engine
            .load_state_dict(parameters, state_dict, AdamStateKind::AdamW)
    }

    /// Atomically replace all adaptive state for a runtime parameter list.
    pub fn replace_slot_states(
        &mut self,
        states: &[(ParameterSlot, Option<AdamParameterState>)],
    ) -> Result<()> {
        self.engine.replace_slot_states(states)
    }

    /// Update every unique trainable slot that currently has a gradient.
    pub fn step(&mut self, parameters: &ParameterStore) -> Result<usize> {
        self.engine
            .step_parameters(parameters.trainable(), WeightDecayMode::Decoupled)
    }

    /// Update a runtime list of parameter slots without requiring persisted state names.
    ///
    /// Repeated/tied identities are updated once in first-occurrence order. Frozen slots and slots
    /// without gradients are skipped, matching [`Self::step`].
    pub fn step_slots(&mut self, parameters: &[ParameterSlot]) -> Result<usize> {
        self.engine
            .step_parameters(unique_trainable(parameters), WeightDecayMode::Decoupled)
    }
}

fn unique_trainable(parameters: &[ParameterSlot]) -> Vec<&ParameterSlot> {
    let mut seen = HashSet::new();
    parameters
        .iter()
        .filter(|parameter| parameter.contract().trainable() && seen.insert(parameter.id()))
        .collect()
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
    fn adaptive_snapshot_rebinds_to_a_fresh_slot_and_resumes_exactly() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = scalar_parameter("weight", 2.0, &device);
        let config = AdamConfig::new(0.1).with_amsgrad(true);
        let mut baseline = Adam::with_config(config).unwrap();
        set_gradient(&parameter, &store, 2.0);
        baseline.step(&store).unwrap();
        let snapshot = baseline.state_for(&parameter).unwrap();
        assert_eq!(snapshot.step(), 1);
        assert!(snapshot.max_second_moment().is_some());

        let (resumed_parameter, resumed_store) =
            scalar_parameter("weight", scalar_value(&parameter), &device);
        let mut resumed = Adam::with_config(config).unwrap();
        resumed
            .replace_slot_states(&[(resumed_parameter.clone(), Some(snapshot))])
            .unwrap();

        set_gradient(&parameter, &store, 2.0);
        baseline.step(&store).unwrap();
        set_gradient(&resumed_parameter, &resumed_store, 2.0);
        resumed.step(&resumed_store).unwrap();

        assert!((scalar_value(&resumed_parameter) - scalar_value(&parameter)).abs() < 1.0e-7);
        assert_eq!(resumed.step_for(resumed_parameter.id()), Some(2));
    }

    #[test]
    fn named_adam_state_restores_config_on_a_fresh_parameter_identity() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = scalar_parameter("policy.weight", 2.0, &device);
        let config = AdamConfig::new(0.03)
            .with_betas(0.8, 0.95)
            .with_epsilon(1.0e-6)
            .with_amsgrad(true);
        let mut baseline = Adam::with_config(config).unwrap();
        set_gradient(&parameter, &store, 2.0);
        baseline.step(&store).unwrap();
        let state_dict = baseline.state_dict(&store);
        assert_eq!(state_dict.kind(), AdamStateKind::Adam);
        assert_eq!(state_dict.parameter_names(), ["policy.weight"]);
        assert_eq!(state_dict.state().len(), 1);

        let mut wrong_kind = AdamW::new(0.4).unwrap();
        assert!(wrong_kind.load_state_dict(&store, &state_dict).is_err());
        assert_eq!(wrong_kind.config().learning_rate(), 0.4);

        let (resumed_parameter, resumed_store) =
            scalar_parameter("policy.weight", scalar_value(&parameter), &device);
        let mut resumed = Adam::new(0.9).unwrap();
        resumed
            .load_state_dict(&resumed_store, &state_dict)
            .unwrap();
        assert_eq!(resumed.config(), config);

        set_gradient(&parameter, &store, -1.5);
        baseline.step(&store).unwrap();
        set_gradient(&resumed_parameter, &resumed_store, -1.5);
        resumed.step(&resumed_store).unwrap();
        assert!((scalar_value(&resumed_parameter) - scalar_value(&parameter)).abs() < 1.0e-7);
    }

    #[test]
    fn adaptive_state_load_is_atomic_and_configuration_checked() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = scalar_parameter("weight", 2.0, &device);
        let mut adam = Adam::new(0.1).unwrap();
        set_gradient(&parameter, &store, 2.0);
        adam.step(&store).unwrap();
        let original = adam.state_for(&parameter).unwrap();
        let incompatible = AdamParameterState::new(
            1,
            original.first_moment(),
            original.second_moment(),
            Some(original.second_moment()),
        );

        assert!(
            adam.replace_slot_states(&[(parameter.clone(), Some(incompatible))])
                .is_err()
        );
        assert_eq!(adam.state_for(&parameter).unwrap().step(), 1);

        let zero_step =
            AdamParameterState::new(0, original.first_moment(), original.second_moment(), None);
        assert!(
            adam.replace_slot_states(&[(parameter.clone(), Some(zero_step))])
                .is_err()
        );
        assert_eq!(adam.state_for(&parameter).unwrap().step(), 1);
    }

    #[test]
    fn slot_list_updates_tied_identity_once_without_requiring_a_name() {
        let device = Device::autodiff(Device::default());
        let parameter = ParameterSlot::new(None, tensor(vec![2.0], &[1], &device), true).unwrap();
        let store = {
            let mut store = ParameterStore::new();
            store.insert("weight", parameter.clone()).unwrap();
            store
        };
        set_gradient(&parameter, &store, 2.0);
        let mut adam = Adam::new(0.1).unwrap();

        assert_eq!(
            adam.step_slots(&[parameter.clone(), parameter.clone()])
                .unwrap(),
            1
        );
        assert!((scalar_value(&parameter) - 1.9).abs() < 1.0e-6);
        assert_eq!(adam.state_len(), 1);
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
