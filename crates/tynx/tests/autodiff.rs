#![cfg(feature = "training")]

use burn::tensor::{Device, TensorData};
use tynx::DynTensor;

fn tensor(values: Vec<f32>, dims: &[usize], device: &Device) -> DynTensor {
    DynTensor::from_data(TensorData::new(values, dims.to_vec()), dims.len(), device).unwrap()
}

#[test]
fn differentiates_dynamic_tensor_operations() {
    let device = Device::autodiff(Device::default());
    let parameter = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device).require_grad();
    let squared = parameter.clone().mul_broadcast(parameter.clone()).unwrap();
    let loss = squared.mean_dims(&[0, 1]).reshape(vec![1]).unwrap();

    let mut gradients = loss.backward();

    let gradient = parameter.grad(&gradients).unwrap();
    assert_eq!(gradient.dims(), [2, 2]);
    assert_eq!(
        gradient.into_data().iter::<f32>().collect::<Vec<_>>(),
        [0.5, 1.0, 1.5, 2.0]
    );
    assert!(parameter.grad_remove(&mut gradients).is_some());
    assert!(parameter.grad_remove(&mut gradients).is_none());
}

#[test]
fn detach_and_inner_remove_tracking_before_releafing() {
    let device = Device::autodiff(Device::default());
    let leaf = tensor(vec![1.0, 2.0], &[2], &device).require_grad();

    assert!(leaf.is_require_grad());
    assert!(!leaf.clone().detach().is_require_grad());

    let inner = leaf.inner();
    assert!(!inner.is_require_grad());

    let next_leaf = inner.to_autodiff().require_grad();
    assert!(next_leaf.is_require_grad());
}
