//! Off-tape gradient transformations over stable parameter slots.

use std::collections::HashSet;

use tynx_core::{DynTensor, Result, TynxError};

use crate::ParameterSlot;

const NORM_EPSILON: f64 = 1.0e-6;

/// Clip the aggregate L2 gradient norm and return its value before clipping.
///
/// Tied parameter identities contribute once. Parameters without gradients are skipped. The
/// returned tensor has shape `(1,)` and remains on the parameters' device.
pub fn clip_grad_norm(
    parameters: &[ParameterSlot],
    max_norm: f64,
    norm_type: f64,
) -> Result<DynTensor> {
    validate_nonnegative_finite("maximum norm", max_norm)?;
    if norm_type != 2.0 {
        return Err(TynxError::TypeMismatch(format!(
            "gradient clipping currently supports only norm_type=2.0, got {norm_type}"
        )));
    }
    let unique = unique_parameters(parameters);
    let mut gradients = Vec::new();
    for parameter in &unique {
        if let Some(gradient) = parameter.grad() {
            gradients.push((parameter, gradient));
        }
    }

    let mut total_squared = if let Some((_, gradient)) = gradients.first() {
        sum_all(gradient.clone().powi_scalar(2))?
    } else {
        let template = unique.first().ok_or_else(|| {
            TynxError::TypeMismatch("gradient clipping requires at least one parameter".to_string())
        })?;
        sum_all(template.value().mul_scalar(0.0))?
    };
    for (_, gradient) in gradients.iter().skip(1) {
        let squared = sum_all(gradient.clone().powi_scalar(2))?;
        total_squared = total_squared.add_broadcast(squared)?;
    }
    let total_norm = total_squared.sqrt();

    if !gradients.is_empty() {
        let denominator = total_norm.clone().add_scalar(NORM_EPSILON);
        let coefficient = denominator
            .clone()
            .full_like(max_norm)
            .div_broadcast(denominator)?
            .min_broadcast(total_norm.clone().full_like(1.0))?;
        for (parameter, gradient) in gradients {
            parameter.replace_grad(gradient.mul_broadcast(coefficient.clone())?)?;
        }
    }

    Ok(total_norm)
}

/// Clamp every gradient element to `[-clip_value, clip_value]`.
///
/// Returns the number of unique parameter gradients changed.
pub fn clip_grad_value(parameters: &[ParameterSlot], clip_value: f64) -> Result<usize> {
    validate_nonnegative_finite("clip value", clip_value)?;
    let mut clipped = 0;
    for parameter in unique_parameters(parameters) {
        let Some(gradient) = parameter.grad() else {
            continue;
        };
        parameter.replace_grad(gradient.clip(Some(-clip_value), Some(clip_value)))?;
        clipped += 1;
    }
    Ok(clipped)
}

fn unique_parameters(parameters: &[ParameterSlot]) -> Vec<&ParameterSlot> {
    let mut seen = HashSet::new();
    parameters
        .iter()
        .filter(|parameter| parameter.contract().trainable() && seen.insert(parameter.id()))
        .collect()
}

fn sum_all(value: DynTensor) -> Result<DynTensor> {
    let dims = (0..value.rank()).collect::<Vec<_>>();
    value.sum_dims(&dims).reshape(vec![1])
}

fn validate_nonnegative_finite(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(TynxError::TypeMismatch(format!(
            "gradient clipping {name} must be finite and non-negative, got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, TensorData};

    use super::*;
    use crate::{ParameterStore, backward};

    fn tensor(values: Vec<f32>, device: &Device) -> DynTensor {
        DynTensor::from_data(
            TensorData::new(values.clone(), vec![values.len()]),
            1,
            device,
        )
        .unwrap()
    }

    fn values(value: DynTensor) -> Vec<f32> {
        value.into_data().iter::<f32>().collect()
    }

    fn parameter(values: Vec<f32>, device: &Device) -> (ParameterSlot, ParameterStore) {
        let parameter = ParameterSlot::new(None, tensor(values, device), true).unwrap();
        let mut store = ParameterStore::new();
        store.insert("weight", parameter.clone()).unwrap();
        (parameter, store)
    }

    #[test]
    fn norm_clipping_returns_pre_clip_norm_and_scales_once() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = parameter(vec![3.0, 4.0], &device);
        let loss = parameter
            .read()
            .clone()
            .mul_broadcast(parameter.read())
            .unwrap()
            .sum_dims(&[0]);
        backward(&loss, &store).unwrap();

        let norm = clip_grad_norm(&[parameter.clone(), parameter.clone()], 5.0, 2.0).unwrap();

        assert!((values(norm)[0] - 10.0).abs() < 1.0e-6);
        let gradient = values(parameter.grad().unwrap());
        assert!((gradient[0] - 3.0).abs() < 1.0e-5);
        assert!((gradient[1] - 4.0).abs() < 1.0e-5);
    }

    #[test]
    fn value_clipping_skips_missing_gradients_and_validates_arguments() {
        let device = Device::autodiff(Device::default());
        let (parameter, store) = parameter(vec![-3.0, 4.0], &device);
        let missing = ParameterSlot::new(None, tensor(vec![1.0], &device), true).unwrap();
        let loss = parameter
            .read()
            .clone()
            .mul_broadcast(parameter.read())
            .unwrap()
            .sum_dims(&[0]);
        backward(&loss, &store).unwrap();

        assert_eq!(
            clip_grad_value(&[parameter.clone(), parameter.clone(), missing], 2.0).unwrap(),
            1
        );
        assert_eq!(values(parameter.grad().unwrap()), [-2.0, 2.0]);
        assert!(clip_grad_value(std::slice::from_ref(&parameter), -1.0).is_err());
        assert!(clip_grad_norm(&[parameter], 1.0, 1.0).is_err());
    }

    #[test]
    fn norm_without_gradients_is_zero() {
        let device = Device::autodiff(Device::default());
        let (parameter, _) = parameter(vec![3.0, 4.0], &device);

        let norm = clip_grad_norm(&[parameter], 1.0, 2.0).unwrap();

        assert_eq!(values(norm), [0.0]);
    }
}
