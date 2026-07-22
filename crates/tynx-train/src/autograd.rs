//! Explicit Rust bridge from Burn backward results to persistent parameter gradients.

use std::panic::{AssertUnwindSafe, catch_unwind};

use tynx_core::{DynTensor, Gradients, Result, TynxError};

use crate::ParameterStore;

/// The raw Burn gradients and parameter-population summary from one backward pass.
pub struct BackwardResult {
    gradients: Gradients,
    parameters_with_grad: usize,
}

impl BackwardResult {
    /// Return the raw gradients container for inspecting non-parameter tensor gradients.
    pub fn gradients(&self) -> &Gradients {
        &self.gradients
    }

    /// Consume this result and return the raw gradients container.
    pub fn into_gradients(self) -> Gradients {
        self.gradients
    }

    /// Return how many unique parameter slots received a gradient.
    pub fn parameters_with_grad(&self) -> usize {
        self.parameters_with_grad
    }
}

/// Run backward and accumulate gradients into the store's current parameter leaves.
///
/// The loss must contain exactly one element and still participate in an autodiff graph. This
/// explicit first-milestone bridge follows the normal `forward -> backward -> step` lifecycle: it
/// resolves the current leaf cached by each slot, so callers must not replace parameter values
/// between forward and backward.
pub fn backward(loss: &DynTensor, parameters: &ParameterStore) -> Result<BackwardResult> {
    let numel = loss.dims().into_iter().try_fold(1_usize, |count, dim| {
        count
            .checked_mul(dim)
            .ok_or_else(|| TynxError::Shape("loss element count overflowed usize".to_string()))
    })?;
    if numel != 1 {
        return Err(TynxError::Shape(format!(
            "backward without an explicit gradient requires a one-element loss, got shape {:?}",
            loss.dims()
        )));
    }
    // This pinned Burn revision exposes leaf `require_grad` but not the underlying tensor's
    // `is_tracked` state. Derived tracked losses and detached losses both report false, while Burn
    // panics for the latter. Contain that backend precondition at this Rust facade boundary until
    // a public graph-membership query is available.
    let gradients = catch_unwind(AssertUnwindSafe(|| loss.backward())).map_err(|_| {
        TynxError::TypeMismatch(
            "backward requires a loss attached to an autodiff graph".to_string(),
        )
    })?;
    let mut parameters_with_grad = 0;
    for parameter in parameters.trainable() {
        if parameter.accumulate_from(&gradients)? {
            parameters_with_grad += 1;
        }
    }

    Ok(BackwardResult {
        gradients,
        parameters_with_grad,
    })
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, TensorData};

    use super::*;
    use crate::ParameterSlot;

    fn tensor(values: Vec<f32>, dims: &[usize], device: &Device) -> DynTensor {
        DynTensor::from_data(TensorData::new(values, dims.to_vec()), dims.len(), device).unwrap()
    }

    fn store_with(name: &str, parameter: ParameterSlot) -> ParameterStore {
        let mut store = ParameterStore::new();
        store.insert(name, parameter).unwrap();
        store
    }

    #[test]
    fn populates_every_participating_parameter_gradient() {
        let device = Device::autodiff(Device::default());
        let weight = ParameterSlot::new(None, tensor(vec![2.0], &[1], &device), true).unwrap();
        let bias = ParameterSlot::new(None, tensor(vec![1.0], &[1], &device), true).unwrap();
        let unused = ParameterSlot::new(None, tensor(vec![9.0], &[1], &device), true).unwrap();
        let mut store = ParameterStore::new();
        store.insert("weight", weight.clone()).unwrap();
        store.insert("bias", bias.clone()).unwrap();
        store.insert("unused", unused.clone()).unwrap();

        let input = tensor(vec![3.0], &[1], &device).require_grad();
        let prediction = weight
            .read()
            .mul_broadcast(input.clone())
            .unwrap()
            .add_broadcast(bias.read())
            .unwrap();
        let result = backward(&prediction, &store).unwrap();

        assert_eq!(result.parameters_with_grad(), 2);
        assert_eq!(
            weight
                .grad()
                .unwrap()
                .into_data()
                .iter::<f32>()
                .collect::<Vec<_>>(),
            [3.0]
        );
        assert_eq!(
            bias.grad()
                .unwrap()
                .into_data()
                .iter::<f32>()
                .collect::<Vec<_>>(),
            [1.0]
        );
        assert!(unused.grad().is_none());
        assert_eq!(
            input
                .grad(result.gradients())
                .unwrap()
                .into_data()
                .iter::<f32>()
                .collect::<Vec<_>>(),
            [2.0]
        );
    }

    #[test]
    fn repeated_backward_accumulates_until_zeroed() {
        let device = Device::autodiff(Device::default());
        let parameter =
            ParameterSlot::new(None, tensor(vec![1.0, 2.0], &[2], &device), true).unwrap();
        let store = store_with("weight", parameter.clone());

        for _ in 0..2 {
            let leaf = parameter.read();
            let loss = leaf.clone().mul_broadcast(leaf).unwrap().mean_dims(&[0]);
            backward(&loss, &store).unwrap();
        }

        assert_eq!(
            parameter
                .grad()
                .unwrap()
                .into_data()
                .iter::<f32>()
                .collect::<Vec<_>>(),
            [2.0, 4.0]
        );

        store.zero_grad();
        assert!(parameter.grad().is_none());
    }

    #[test]
    fn value_replacement_does_not_clear_gradients() {
        let device = Device::autodiff(Device::default());
        let parameter = ParameterSlot::new(None, tensor(vec![2.0], &[1], &device), true).unwrap();
        let store = store_with("weight", parameter.clone());
        let leaf = parameter.read();
        let loss = leaf.clone().mul_broadcast(leaf).unwrap();
        backward(&loss, &store).unwrap();

        parameter
            .replace_value(tensor(vec![1.5], &[1], &device))
            .unwrap();

        assert_eq!(
            parameter
                .grad()
                .unwrap()
                .into_data()
                .iter::<f32>()
                .collect::<Vec<_>>(),
            [4.0]
        );
    }

    #[test]
    fn rejects_non_scalar_and_detached_losses() {
        let device = Device::autodiff(Device::default());
        let parameter =
            ParameterSlot::new(None, tensor(vec![1.0, 2.0], &[2], &device), true).unwrap();
        let store = store_with("weight", parameter.clone());

        let error = backward(&parameter.read(), &store).err().unwrap();
        assert!(error.to_string().contains("one-element loss"));

        let detached = parameter.read().mean_dims(&[0]).detach();
        let error = backward(&detached, &store).err().unwrap();
        assert!(error.to_string().contains("attached to an autodiff graph"));
    }

    #[test]
    fn tied_aliases_accumulate_only_once() {
        let device = Device::autodiff(Device::default());
        let parameter = ParameterSlot::new(None, tensor(vec![2.0], &[1], &device), true).unwrap();
        let mut store = ParameterStore::new();
        store.insert("left.weight", parameter.clone()).unwrap();
        store.insert("right.weight", parameter.clone()).unwrap();
        let loss = parameter.read().add_broadcast(parameter.read()).unwrap();

        let result = backward(&loss, &store).unwrap();

        assert_eq!(result.parameters_with_grad(), 1);
        assert_eq!(
            parameter
                .grad()
                .unwrap()
                .into_data()
                .iter::<f32>()
                .collect::<Vec<_>>(),
            [2.0]
        );
    }
}
