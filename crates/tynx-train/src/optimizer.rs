//! Optimizers over stable parameter slots.

mod adam;

pub use adam::{
    Adam, AdamConfig, AdamParameterState, AdamStateDict, AdamStateKind, AdamW, AdamWConfig,
};

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use tynx_core::{DynTensor, Result, TynxError};

use crate::{ParamId, ParameterSlot, ParameterStore};

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

/// Portable momentum state for one SGD parameter.
///
/// Runtime parameter identity and structure generation are deliberately excluded. Loading binds
/// the detached buffer to a destination slot's current identity and structural contract.
#[derive(Debug, Clone)]
pub struct SgdParameterState {
    momentum_buffer: DynTensor,
}

impl SgdParameterState {
    /// Construct a portable state entry from a momentum buffer.
    pub fn new(momentum_buffer: DynTensor) -> Self {
        Self {
            momentum_buffer: momentum_buffer.detach(),
        }
    }

    /// Return the detached momentum buffer.
    pub fn momentum_buffer(&self) -> DynTensor {
        self.momentum_buffer.clone()
    }
}

/// Name-keyed, portable SGD configuration and momentum state.
#[derive(Debug, Clone)]
pub struct SgdStateDict {
    config: SgdConfig,
    parameter_names: Vec<String>,
    state: BTreeMap<String, SgdParameterState>,
}

impl SgdStateDict {
    /// Construct and structurally validate a portable SGD state dictionary.
    pub fn new(
        config: SgdConfig,
        parameter_names: Vec<String>,
        state: BTreeMap<String, SgdParameterState>,
    ) -> Result<Self> {
        validate_config(config)?;
        let parameter_names = validate_state_names(parameter_names, state.keys(), "SGD")?;
        if config.momentum == 0.0 && !state.is_empty() {
            return Err(TynxError::TypeMismatch(
                "SGD without momentum cannot contain momentum state".to_string(),
            ));
        }
        Ok(Self {
            config,
            parameter_names,
            state,
        })
    }

    /// Return the serialized optimizer configuration.
    pub fn config(&self) -> SgdConfig {
        self.config
    }

    /// Return stable parameter names in lexical order, including entries without momentum state.
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    /// Iterate allocated momentum state by stable parameter name.
    pub fn state(&self) -> impl ExactSizeIterator<Item = (&str, &SgdParameterState)> {
        self.state
            .iter()
            .map(|(name, state)| (name.as_str(), state))
    }
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
        self.set_config(next)
    }

    /// Replace the live parameter-group configuration without discarding compatible momentum.
    pub fn set_config(&mut self, config: SgdConfig) -> Result<()> {
        validate_config(config)?;
        if config.momentum == 0.0 {
            self.momentum_buffers.clear();
        }
        self.config = config;
        Ok(())
    }

    /// Return the number of parameter momentum buffers.
    pub fn state_len(&self) -> usize {
        self.momentum_buffers.len()
    }

    /// Snapshot the current-generation momentum state for one parameter.
    pub fn state_for(&self, parameter: &ParameterSlot) -> Option<SgdParameterState> {
        self.momentum_buffers
            .get(&parameter.id())
            .filter(|(generation, _)| *generation == parameter.structure_generation())
            .map(|(_, momentum_buffer)| SgdParameterState::new(momentum_buffer.clone().detach()))
    }

    /// Export configuration and current-generation momentum state by canonical parameter name.
    pub fn state_dict(&self, parameters: &ParameterStore) -> SgdStateDict {
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
        SgdStateDict::new(self.config, parameter_names, state)
            .expect("live SGD configuration and ParameterStore names are valid")
    }

    /// Restore configuration and momentum state by canonical name onto fresh runtime slots.
    pub fn load_state_dict(
        &mut self,
        parameters: &ParameterStore,
        state_dict: &SgdStateDict,
    ) -> Result<()> {
        let current = trainable_by_name(parameters);
        validate_parameter_name_match(current.keys().copied(), &state_dict.parameter_names, "SGD")?;
        let states = current
            .into_iter()
            .map(|(name, parameter)| (parameter.clone(), state_dict.state.get(name).cloned()))
            .collect::<Vec<_>>();
        let mut replacement = Self::with_config(state_dict.config)?;
        replacement.replace_slot_states(&states)?;
        *self = replacement;
        Ok(())
    }

    /// Atomically replace all SGD state for a runtime parameter list.
    ///
    /// Each `Some` entry is rebound to the destination slot's current identity and structure
    /// generation. `None` represents a parameter with no allocated momentum state. Duplicate
    /// destination identities are rejected rather than silently selecting one entry.
    pub fn replace_slot_states(
        &mut self,
        states: &[(ParameterSlot, Option<SgdParameterState>)],
    ) -> Result<()> {
        let mut seen = HashSet::new();
        let mut replacement = HashMap::new();
        for (parameter, state) in states {
            if !seen.insert(parameter.id()) {
                return Err(TynxError::TypeMismatch(format!(
                    "duplicate SGD state destination for parameter {}",
                    parameter.id().get()
                )));
            }
            let Some(state) = state else {
                continue;
            };
            validate_state_tensor("SGD momentum buffer", &state.momentum_buffer, parameter)?;
            replacement.insert(
                parameter.id(),
                (
                    parameter.structure_generation(),
                    state.momentum_buffer.clone().detach(),
                ),
            );
        }
        self.momentum_buffers = replacement;
        Ok(())
    }

    /// Update every unique trainable slot that currently has a gradient.
    ///
    /// Returns the number of updated slots. Frozen slots and parameters without gradients are
    /// skipped. All update tensors are computed before publication, providing one serialized
    /// multi-parameter commit in the initial single-threaded runtime.
    pub fn step(&mut self, parameters: &ParameterStore) -> Result<usize> {
        self.step_parameters(parameters.trainable())
    }

    /// Update a runtime list of parameter slots without requiring persisted state names.
    ///
    /// Repeated/tied identities are updated once in first-occurrence order. Frozen slots and slots
    /// without gradients are skipped, matching [`Self::step`].
    pub fn step_slots(&mut self, parameters: &[ParameterSlot]) -> Result<usize> {
        let mut seen = HashSet::new();
        let unique = parameters
            .iter()
            .filter(|parameter| parameter.contract().trainable() && seen.insert(parameter.id()))
            .collect::<Vec<_>>();
        self.step_parameters(unique)
    }

    fn step_parameters<'a>(
        &mut self,
        parameters: impl IntoIterator<Item = &'a ParameterSlot>,
    ) -> Result<usize> {
        let mut next_momentum = self.momentum_buffers.clone();
        let mut updates = Vec::new();

        for parameter in parameters {
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
        ParameterSlot::replace_values_atomic(&updates)?;
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

fn validate_state_tensor(label: &str, tensor: &DynTensor, parameter: &ParameterSlot) -> Result<()> {
    let contract = parameter.contract();
    if tensor.dims() != contract.shape()
        || tensor.dtype() != contract.dtype()
        || tensor.device() != *contract.device()
    {
        return Err(TynxError::TypeMismatch(format!(
            "{label} shape {:?}, dtype {:?}, and device {:?} do not match parameter {} contract shape {:?}, dtype {:?}, and device {:?}",
            tensor.dims(),
            tensor.dtype(),
            tensor.device(),
            parameter.id().get(),
            contract.shape(),
            contract.dtype(),
            contract.device(),
        )));
    }
    Ok(())
}

fn validate_state_names<'a>(
    parameter_names: Vec<String>,
    state_names: impl IntoIterator<Item = &'a String>,
    optimizer: &str,
) -> Result<Vec<String>> {
    let mut names = BTreeSet::new();
    for name in parameter_names {
        if name.trim().is_empty() {
            return Err(TynxError::TypeMismatch(format!(
                "{optimizer} parameter state name cannot be empty"
            )));
        }
        if !names.insert(name.clone()) {
            return Err(TynxError::TypeMismatch(format!(
                "duplicate {optimizer} parameter state name '{name}'"
            )));
        }
    }
    for name in state_names {
        if !names.contains(name) {
            return Err(TynxError::TypeMismatch(format!(
                "{optimizer} state for unknown parameter name '{name}'"
            )));
        }
    }
    Ok(names.into_iter().collect())
}

fn trainable_by_name(parameters: &ParameterStore) -> BTreeMap<&str, &ParameterSlot> {
    parameters
        .named()
        .filter(|(_, parameter)| parameter.contract().trainable())
        .collect()
}

fn validate_parameter_name_match<'a>(
    current_names: impl IntoIterator<Item = &'a str>,
    saved_names: &[String],
    optimizer: &str,
) -> Result<()> {
    let current = current_names.into_iter().collect::<BTreeSet<_>>();
    let saved = saved_names
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let missing = current.difference(&saved).copied().collect::<Vec<_>>();
    let unexpected = saved.difference(&current).copied().collect::<Vec<_>>();
    if !missing.is_empty() || !unexpected.is_empty() {
        return Err(TynxError::TypeMismatch(format!(
            "{optimizer} parameter names do not match: missing={missing:?}, unexpected={unexpected:?}"
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
    fn momentum_snapshot_rebinds_to_a_fresh_slot_and_resumes_exactly() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = scalar_parameter("weight", 2.0, &device);
        let config = SgdConfig::new(0.1).with_momentum(0.9);
        let mut baseline = Sgd::with_config(config).unwrap();
        set_gradient(&parameter, &store, 2.0);
        baseline.step(&store).unwrap();
        let snapshot = baseline.state_for(&parameter).unwrap();

        let (resumed_parameter, resumed_store) =
            scalar_parameter("weight", values(parameter.value())[0], &device);
        let mut resumed = Sgd::with_config(config).unwrap();
        resumed
            .replace_slot_states(&[(resumed_parameter.clone(), Some(snapshot))])
            .unwrap();

        store.zero_grad();
        set_gradient(&parameter, &store, 2.0);
        baseline.step(&store).unwrap();
        set_gradient(&resumed_parameter, &resumed_store, 2.0);
        resumed.step(&resumed_store).unwrap();

        assert_eq!(values(resumed_parameter.value()), values(parameter.value()));
        assert_eq!(resumed.state_len(), 1);
    }

    #[test]
    fn named_sgd_state_restores_config_and_slots_independent_of_discovery_order() {
        let device = Device::autodiff(Device::default());
        let weight = ParameterSlot::new(None, tensor(vec![2.0], &[1], &device), true).unwrap();
        let bias = ParameterSlot::new(None, tensor(vec![1.0], &[1], &device), true).unwrap();
        let mut store = ParameterStore::new();
        store.insert("weight", weight.clone()).unwrap();
        store.insert("bias", bias.clone()).unwrap();
        let config = SgdConfig::new(0.1).with_momentum(0.9);
        let mut baseline = Sgd::with_config(config).unwrap();
        set_gradient(&weight, &store, 2.0);
        set_gradient(&bias, &store, 3.0);
        baseline.step(&store).unwrap();
        let state_dict = baseline.state_dict(&store);
        assert_eq!(state_dict.parameter_names(), ["bias", "weight"]);
        assert_eq!(state_dict.state().len(), 2);

        let resumed_weight =
            ParameterSlot::new(None, weight.value(), true).expect("weight contract remains valid");
        let resumed_bias =
            ParameterSlot::new(None, bias.value(), true).expect("bias contract remains valid");
        let mut resumed_store = ParameterStore::new();
        resumed_store.insert("bias", resumed_bias.clone()).unwrap();
        resumed_store
            .insert("weight", resumed_weight.clone())
            .unwrap();
        let mut resumed = Sgd::new(0.8).unwrap();
        resumed
            .load_state_dict(&resumed_store, &state_dict)
            .unwrap();
        assert_eq!(resumed.config(), config);

        store.zero_grad();
        set_gradient(&weight, &store, 2.0);
        set_gradient(&bias, &store, 3.0);
        baseline.step(&store).unwrap();
        set_gradient(&resumed_weight, &resumed_store, 2.0);
        set_gradient(&resumed_bias, &resumed_store, 3.0);
        resumed.step(&resumed_store).unwrap();

        assert_eq!(values(resumed_weight.value()), values(weight.value()));
        assert_eq!(values(resumed_bias.value()), values(bias.value()));

        let (wrong_parameter, wrong_store) = scalar_parameter("other", 0.0, &device);
        let mut unchanged = Sgd::new(0.7).unwrap();
        assert!(
            unchanged
                .load_state_dict(&wrong_store, &state_dict)
                .is_err()
        );
        assert_eq!(unchanged.config().learning_rate(), 0.7);
        assert_eq!(wrong_parameter.value_generation(), 0);
    }

    #[test]
    fn momentum_state_load_validates_before_replacement_and_ignores_stale_state() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = scalar_parameter("weight", 2.0, &device);
        set_gradient(&parameter, &store, 2.0);
        let mut sgd = Sgd::with_config(SgdConfig::new(0.1).with_momentum(0.9)).unwrap();
        sgd.step(&store).unwrap();
        let original = sgd.state_for(&parameter).unwrap();
        let invalid = SgdParameterState::new(tensor(vec![1.0, 2.0], &[2], &device));

        assert!(
            sgd.replace_slot_states(&[(parameter.clone(), Some(invalid))])
                .is_err()
        );
        assert_eq!(
            values(sgd.state_for(&parameter).unwrap().momentum_buffer()),
            values(original.momentum_buffer())
        );

        parameter
            .rebind(tensor(vec![1.8], &[1], &device), true)
            .unwrap();
        assert!(sgd.state_for(&parameter).is_none());
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
    fn slot_list_updates_tied_identity_once_without_requiring_a_name() {
        let device = Device::autodiff(Device::default());
        let parameter = ParameterSlot::new(None, tensor(vec![2.0], &[1], &device), true).unwrap();
        let store = {
            let mut store = ParameterStore::new();
            store.insert("weight", parameter.clone()).unwrap();
            store
        };
        set_gradient(&parameter, &store, 3.0);
        let mut sgd = Sgd::new(0.1).unwrap();

        assert_eq!(
            sgd.step_slots(&[parameter.clone(), parameter.clone()])
                .unwrap(),
            1
        );
        assert_eq!(values(parameter.value()), [1.7]);
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
