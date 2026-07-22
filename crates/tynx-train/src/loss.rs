//! Differentiable loss functions composed from Tynx tensor operations.

use tynx_core::{DynTensor, Result, TynxError};

/// Return the mean squared error as a one-element rank-1 tensor.
///
/// Prediction and target shapes must match exactly. The returned tensor remains on the device and
/// is built entirely from ordinary Tynx operations, so it retains the autodiff tape when its inputs
/// require gradients.
pub fn mse(prediction: DynTensor, target: DynTensor) -> Result<DynTensor> {
    let prediction_dims = prediction.dims();
    let target_dims = target.dims();
    if prediction_dims != target_dims {
        return Err(TynxError::Shape(format!(
            "MSE prediction shape {prediction_dims:?} differs from target shape {target_dims:?}"
        )));
    }

    let reduction_dims: Vec<usize> = (0..prediction_dims.len()).collect();
    let difference = prediction.sub_broadcast(target)?;
    let squared = difference.clone().mul_broadcast(difference)?;
    squared.mean_dims(&reduction_dims).reshape(vec![1])
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, TensorData};
    use tynx_core::DynTensor;

    use super::*;

    fn tensor(values: Vec<f32>, dims: &[usize], device: &Device) -> DynTensor {
        DynTensor::from_data(TensorData::new(values, dims.to_vec()), dims.len(), device).unwrap()
    }

    #[test]
    fn computes_mean_squared_error_as_one_element_tensor() {
        let device = Device::default();
        let prediction = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device);
        let target = tensor(vec![1.0, 4.0, 2.0, 3.0], &[2, 2], &device);

        let loss = mse(prediction, target).unwrap();

        assert_eq!(loss.dims(), [1]);
        let values = loss.into_data().iter::<f32>().collect::<Vec<_>>();
        assert_eq!(values.len(), 1);
        assert!((values[0] - 1.5).abs() < 1.0e-6);
    }

    #[test]
    fn rejects_shape_broadcasting() {
        let device = Device::default();
        let prediction = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &device);
        let target = tensor(vec![1.0, 2.0], &[2, 1], &device);

        let error = mse(prediction, target).unwrap_err();

        assert_eq!(
            error,
            TynxError::Shape(
                "MSE prediction shape [2, 2] differs from target shape [2, 1]".to_string()
            )
        );
    }
}
