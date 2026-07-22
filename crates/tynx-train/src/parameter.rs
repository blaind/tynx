//! Stable parameter identities and versioned tensor storage.

use std::{
    cell::RefCell,
    hash::{Hash, Hasher},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

use burn::tensor::{DType, Device};
use tynx_core::{DynTensor, Gradients, Result, TynxError};

static NEXT_PARAM_ID: AtomicU64 = AtomicU64::new(1);

/// Process-local identity of a parameter slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamId(u64);

impl ParamId {
    /// Return the numeric process-local identity.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// The structural contract used to validate updates and guard captured graphs.
#[derive(Debug, Clone)]
pub struct ParameterContract {
    shape: Vec<usize>,
    dtype: DType,
    device: Device,
    trainable: bool,
}

impl PartialEq for ParameterContract {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.dtype == other.dtype
            && self.device == other.device
            && self.device.is_autodiff() == other.device.is_autodiff()
            && self.trainable == other.trainable
    }
}

impl Eq for ParameterContract {}

impl ParameterContract {
    /// Return the parameter shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the parameter element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Return the parameter device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Return whether reads produce autodiff leaves.
    pub fn trainable(&self) -> bool {
        self.trainable
    }
}

#[derive(Debug)]
struct ParameterState {
    name: Option<String>,
    value: DynTensor,
    contract: ParameterContract,
    value_generation: u64,
    structure_generation: u64,
    leaf: Option<(u64, DynTensor)>,
    grad: Option<DynTensor>,
}

/// A stable, cloneable parameter identity whose current tensor can change in place.
///
/// The initial facade is intentionally single-threaded. Clones share one `Rc<RefCell<_>>`, which
/// matches the planned unsendable CPython ownership model. A compatible value update advances only
/// `value_generation`; explicit rebinding or a trainability change also advances
/// `structure_generation`.
#[derive(Debug, Clone)]
pub struct ParameterSlot {
    id: ParamId,
    state: Rc<RefCell<ParameterState>>,
}

impl ParameterSlot {
    /// Create a slot from a tensor value.
    ///
    /// Trainable v1 parameters must be f32 tensors on an autodiff-enabled device.
    pub fn new(name: Option<String>, value: DynTensor, trainable: bool) -> Result<Self> {
        let contract = contract_for(&value, trainable);
        validate_trainability(&contract)?;
        let value = off_tape(value, &contract.device);
        Ok(Self {
            id: next_param_id(),
            state: Rc::new(RefCell::new(ParameterState {
                name,
                value,
                contract,
                value_generation: 0,
                structure_generation: 0,
                leaf: None,
                grad: None,
            })),
        })
    }

    /// Return this slot's stable process-local identity.
    pub fn id(&self) -> ParamId {
        self.id
    }

    /// Return the optional state/checkpoint name.
    pub fn name(&self) -> Option<String> {
        self.state.borrow().name.clone()
    }

    pub(crate) fn set_name(&self, name: String) {
        self.state.borrow_mut().name = Some(name);
    }

    /// Return the current structural contract.
    pub fn contract(&self) -> ParameterContract {
        self.state.borrow().contract.clone()
    }

    /// Return the current value generation.
    pub fn value_generation(&self) -> u64 {
        self.state.borrow().value_generation
    }

    /// Return the current structural generation.
    pub fn structure_generation(&self) -> u64 {
        self.state.borrow().structure_generation
    }

    /// Read the current value without attaching it to an autodiff tape.
    pub fn value(&self) -> DynTensor {
        self.state.borrow().value.clone()
    }

    /// Return the accumulated gradient without clearing it.
    pub fn grad(&self) -> Option<DynTensor> {
        self.state.borrow().grad.clone()
    }

    /// Clear the accumulated gradient.
    pub fn zero_grad(&self) {
        self.state.borrow_mut().grad = None;
    }

    /// Read the tensor used by a forward pass.
    ///
    /// Trainable slots create one leaf on the first read of each value generation and reuse that
    /// exact leaf for subsequent reads. Frozen slots return the off-tape value.
    pub fn read(&self) -> DynTensor {
        let mut state = self.state.borrow_mut();
        if !state.contract.trainable {
            return state.value.clone();
        }
        if let Some((generation, leaf)) = &state.leaf
            && *generation == state.value_generation
        {
            return leaf.clone();
        }
        let leaf = state.value.clone().require_grad();
        state.leaf = Some((state.value_generation, leaf.clone()));
        leaf
    }

    /// Replace the tensor after an ordinary optimizer-style value update.
    ///
    /// Shape, dtype, device, and trainability remain unchanged. The replacement is detached before
    /// publication, the value generation advances, and the next read creates the new leaf.
    pub fn replace_value(&self, value: DynTensor) -> Result<()> {
        let mut state = self.state.borrow_mut();
        let next_contract = contract_for(&value, state.contract.trainable);
        if next_contract != state.contract {
            return Err(contract_mismatch(&state.contract, &next_contract));
        }
        state.value = off_tape(value, &state.contract.device);
        state.value_generation = next_generation(state.value_generation, "value")?;
        state.leaf = None;
        Ok(())
    }

    /// Explicitly replace the slot's structural contract and value.
    ///
    /// This is the path for shape, dtype, device, or role changes. Both generations advance so
    /// eager reads select a new leaf and captured graphs can invalidate their structural guard.
    pub fn rebind(&self, value: DynTensor, trainable: bool) -> Result<()> {
        let next_contract = contract_for(&value, trainable);
        validate_trainability(&next_contract)?;
        let value = off_tape(value, &next_contract.device);
        let mut state = self.state.borrow_mut();
        state.value = value;
        state.contract = next_contract;
        state.value_generation = next_generation(state.value_generation, "value")?;
        state.structure_generation = next_generation(state.structure_generation, "structure")?;
        state.leaf = None;
        state.grad = None;
        Ok(())
    }

    /// Freeze or unfreeze this slot while retaining its current tensor value.
    pub fn set_trainable(&self, trainable: bool) -> Result<()> {
        let mut state = self.state.borrow_mut();
        if state.contract.trainable == trainable {
            return Ok(());
        }
        let mut next_contract = state.contract.clone();
        next_contract.trainable = trainable;
        validate_trainability(&next_contract)?;
        state.contract = next_contract;
        state.value_generation = next_generation(state.value_generation, "value")?;
        state.structure_generation = next_generation(state.structure_generation, "structure")?;
        state.leaf = None;
        Ok(())
    }

    pub(crate) fn accumulate_from(&self, gradients: &Gradients) -> Result<bool> {
        let leaf = {
            let state = self.state.borrow();
            match &state.leaf {
                Some((generation, leaf)) if *generation == state.value_generation => leaf.clone(),
                _ => return Ok(false),
            }
        };
        let Some(gradient) = leaf.grad(gradients) else {
            return Ok(false);
        };

        let mut state = self.state.borrow_mut();
        let gradient_device = tensor_device(&gradient);
        if gradient.dims() != state.contract.shape
            || gradient.dtype() != state.contract.dtype
            || gradient_device != state.contract.device
        {
            return Err(TynxError::TypeMismatch(format!(
                "gradient shape {:?}, dtype {:?}, and device {gradient_device:?} do not match parameter contract {:?}",
                gradient.dims(),
                gradient.dtype(),
                state.contract,
            )));
        }
        let gradient = off_tape(gradient, &gradient_device);
        let accumulated = match &state.grad {
            Some(current) => current.clone().add_broadcast(gradient)?,
            None => gradient,
        };
        let accumulated_device = tensor_device(&accumulated);
        state.grad = Some(off_tape(accumulated, &accumulated_device));
        Ok(true)
    }
}

impl PartialEq for ParameterSlot {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ParameterSlot {}

impl Hash for ParameterSlot {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

fn next_param_id() -> ParamId {
    ParamId(NEXT_PARAM_ID.fetch_add(1, Ordering::Relaxed))
}

fn contract_for(value: &DynTensor, trainable: bool) -> ParameterContract {
    ParameterContract {
        shape: value.dims(),
        dtype: value.dtype(),
        device: tensor_device(value),
        trainable,
    }
}

fn tensor_device(value: &DynTensor) -> Device {
    match value {
        DynTensor::R1(tensor) => tensor.device(),
        DynTensor::R2(tensor) => tensor.device(),
        DynTensor::R3(tensor) => tensor.device(),
        DynTensor::R4(tensor) => tensor.device(),
        DynTensor::R5(tensor) => tensor.device(),
        DynTensor::R6(tensor) => tensor.device(),
    }
}

fn off_tape(value: DynTensor, device: &Device) -> DynTensor {
    if device.is_autodiff() {
        value.detach()
    } else {
        value
    }
}

fn validate_trainability(contract: &ParameterContract) -> Result<()> {
    if !contract.trainable {
        return Ok(());
    }
    if contract.dtype != DType::F32 {
        return Err(TynxError::TypeMismatch(format!(
            "trainable parameters must use f32 in v1, got {:?}",
            contract.dtype
        )));
    }
    if !contract.device.is_autodiff() {
        return Err(TynxError::TypeMismatch(
            "trainable parameters require an autodiff-enabled device".to_string(),
        ));
    }
    Ok(())
}

fn contract_mismatch(current: &ParameterContract, next: &ParameterContract) -> TynxError {
    TynxError::TypeMismatch(format!(
        "parameter value update changed its structural contract from {current:?} to {next:?}; use rebind for an explicit structural change"
    ))
}

fn next_generation(current: u64, kind: &str) -> Result<u64> {
    current
        .checked_add(1)
        .ok_or_else(|| TynxError::TypeMismatch(format!("parameter {kind} generation exhausted")))
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;

    fn tensor(values: Vec<f32>, dims: &[usize], device: &Device) -> DynTensor {
        DynTensor::from_data(TensorData::new(values, dims.to_vec()), dims.len(), device).unwrap()
    }

    #[test]
    fn clones_share_stable_identity_and_state() {
        let device = Device::autodiff(Device::default());
        let slot = ParameterSlot::new(
            Some("weight".to_string()),
            tensor(vec![1.0, 2.0], &[2], &device),
            true,
        )
        .unwrap();
        let alias = slot.clone();

        assert_eq!(slot, alias);
        assert_eq!(slot.id(), alias.id());
        assert_eq!(slot.name().as_deref(), Some("weight"));
        alias
            .replace_value(tensor(vec![3.0, 4.0], &[2], &device))
            .unwrap();
        assert_eq!(slot.value_generation(), 1);
        assert_eq!(slot.structure_generation(), 0);
        assert_eq!(
            slot.value().into_data().iter::<f32>().collect::<Vec<_>>(),
            [3.0, 4.0]
        );
    }

    #[test]
    fn reuses_one_leaf_for_every_read_of_a_value_generation() {
        let device = Device::autodiff(Device::default());
        let slot = ParameterSlot::new(None, tensor(vec![1.0, 2.0], &[2], &device), true).unwrap();
        let first = slot.read();
        let second = slot.read();
        let first_loss = first
            .clone()
            .mul_broadcast(first.clone())
            .unwrap()
            .mean_dims(&[0]);
        let second_loss = second
            .clone()
            .mul_broadcast(second)
            .unwrap()
            .mean_dims(&[0]);
        let loss = first_loss.add_broadcast(second_loss).unwrap();

        let gradients = loss.backward();
        let gradient = first.grad(&gradients).unwrap();

        assert_eq!(
            gradient.into_data().iter::<f32>().collect::<Vec<_>>(),
            [2.0, 4.0]
        );
    }

    #[test]
    fn compatible_updates_advance_only_the_value_generation() {
        let device = Device::autodiff(Device::default());
        let slot = ParameterSlot::new(None, tensor(vec![1.0, 2.0], &[2], &device), true).unwrap();
        let old_leaf = slot.read();
        let old_loss = old_leaf
            .clone()
            .mul_broadcast(old_leaf.clone())
            .unwrap()
            .mean_dims(&[0]);

        let attached_update = old_leaf
            .clone()
            .mul_broadcast(tensor(vec![3.0, 2.0], &[2], &device))
            .unwrap();
        slot.replace_value(attached_update).unwrap();
        let new_leaf = slot.read();

        assert_eq!(slot.value_generation(), 1);
        assert_eq!(slot.structure_generation(), 0);
        assert!(!slot.value().is_require_grad());
        assert!(new_leaf.is_require_grad());
        assert_eq!(
            new_leaf.into_data().iter::<f32>().collect::<Vec<_>>(),
            [3.0, 4.0]
        );

        let gradients = old_loss.backward();
        assert_eq!(
            old_leaf
                .grad(&gradients)
                .unwrap()
                .into_data()
                .iter::<f32>()
                .collect::<Vec<_>>(),
            [1.0, 2.0]
        );
    }

    #[test]
    fn incompatible_value_updates_require_explicit_rebinding() {
        let device = Device::autodiff(Device::default());
        let slot = ParameterSlot::new(None, tensor(vec![1.0, 2.0], &[2], &device), true).unwrap();

        let error = slot
            .replace_value(tensor(vec![1.0, 2.0], &[1, 2], &device))
            .unwrap_err();
        assert!(error.to_string().contains("use rebind"));
        assert_eq!(slot.value_generation(), 0);
        assert_eq!(slot.structure_generation(), 0);

        slot.rebind(tensor(vec![1.0, 2.0], &[1, 2], &device), true)
            .unwrap();
        assert_eq!(slot.value_generation(), 1);
        assert_eq!(slot.structure_generation(), 1);
        assert_eq!(slot.contract().shape(), [1, 2]);
    }

    #[test]
    fn device_autodiff_capability_is_part_of_the_contract() {
        let inference_device = Device::default();
        let training_device = Device::autodiff(inference_device.clone());
        let slot = ParameterSlot::new(None, tensor(vec![1.0, 2.0], &[2], &inference_device), false)
            .unwrap();

        let error = slot
            .replace_value(tensor(vec![3.0, 4.0], &[2], &training_device))
            .unwrap_err();

        assert!(error.to_string().contains("use rebind"));
        assert_eq!(slot.value_generation(), 0);
        assert_eq!(slot.structure_generation(), 0);
    }

    #[test]
    fn freeze_and_unfreeze_are_structural_changes() {
        let device = Device::autodiff(Device::default());
        let slot = ParameterSlot::new(None, tensor(vec![1.0, 2.0], &[2], &device), true).unwrap();

        slot.set_trainable(false).unwrap();
        assert!(!slot.read().is_require_grad());
        assert_eq!(slot.value_generation(), 1);
        assert_eq!(slot.structure_generation(), 1);

        slot.set_trainable(true).unwrap();
        assert!(slot.read().is_require_grad());
        assert_eq!(slot.value_generation(), 2);
        assert_eq!(slot.structure_generation(), 2);
    }

    #[test]
    fn trainable_slots_require_f32_and_an_autodiff_device() {
        let inference_device = Device::default();
        let error = ParameterSlot::new(None, tensor(vec![1.0, 2.0], &[2], &inference_device), true)
            .unwrap_err();
        assert!(error.to_string().contains("autodiff-enabled device"));

        let training_device = Device::autodiff(Device::default());
        let f64 = DynTensor::from_data(
            TensorData::new(vec![1.0_f64, 2.0], [2]),
            1,
            &training_device,
        )
        .unwrap();
        let error = ParameterSlot::new(None, f64, true).unwrap_err();
        assert!(error.to_string().contains("must use f32"));
    }
}
