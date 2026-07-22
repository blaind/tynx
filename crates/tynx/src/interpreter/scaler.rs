//! ONNX-ML Scaler execution.

use burn::tensor::{DType, Device, TensorData};
use onnx_ir::node::scaler::ScalerNode;

use super::{Env, resolve};
use crate::{DynTensor, Result, TynxError, Value};

pub(super) fn scaler(node: &ScalerNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let mut tensor = match input {
        Value::Tensor(tensor) => tensor.cast(DType::F32),
        Value::Int(tensor) => tensor.to_float(DType::F32),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "Scaler requires a floating-point or integer tensor, got {other:?}"
            )));
        }
    };

    let dims = tensor.dims();
    if let Some(offset) = &node.config.offset {
        tensor = tensor.sub_broadcast(feature_values(offset, &dims, "offset", device)?)?;
    }
    if let Some(scale) = &node.config.scale {
        tensor = tensor.mul_broadcast(feature_values(scale, &dims, "scale", device)?)?;
    }

    Ok(vec![Value::Tensor(tensor)])
}

fn feature_values(
    values: &[f32],
    input_dims: &[usize],
    name: &str,
    device: &Device,
) -> Result<DynTensor> {
    let features = input_dims.last().copied().ok_or_else(|| {
        TynxError::Shape("Scaler requires a tensor with at least one dimension".to_string())
    })?;
    if values.len() != 1 && values.len() != features {
        return Err(TynxError::Shape(format!(
            "Scaler {name} has {} values, expected 1 or {features}",
            values.len()
        )));
    }

    let mut dims = vec![1; input_dims.len()];
    let Some(last) = dims.last_mut() else {
        return Err(TynxError::Shape(
            "Scaler requires a tensor with at least one dimension".to_string(),
        ));
    };
    *last = values.len();
    DynTensor::from_data(TensorData::new(values.to_vec(), [values.len()]), 1, device)?.reshape(dims)
}

#[cfg(test)]
mod tests {
    use onnx_ir::node::scaler::{ScalerConfig, ScalerNodeBuilder};

    use super::*;

    #[test]
    fn scales_integer_input_per_feature() {
        let node = ScalerNodeBuilder::new("scaler")
            .input_tensor("input", 2, DType::I64)
            .output_tensor("output", 2, DType::F32)
            .config(ScalerConfig::new(
                Some(vec![2.0, 3.0]),
                Some(vec![1.0, 2.0]),
            ))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "input".to_string(),
            Value::from_tensor_data(TensorData::new(vec![1_i64, 2, 3, 4], [2, 2]), 2, &device)
                .unwrap(),
        );

        let output = scaler(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();

        assert_eq!(output.dtype(), DType::F32);
        output.into_data().assert_eq(
            &TensorData::new(vec![0.0_f32, 0.0, 4.0, 6.0], [2, 2]),
            false,
        );
    }

    #[test]
    fn broadcasts_a_single_scale_value() {
        let node = ScalerNodeBuilder::new("scaler")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(ScalerConfig::new(Some(vec![0.5]), None))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "input".to_string(),
            Value::from_tensor_data(TensorData::new(vec![2.0_f32, 4.0], [1, 2]), 2, &device)
                .unwrap(),
        );

        let output = scaler(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();

        output
            .into_data()
            .assert_eq(&TensorData::new(vec![1.0_f32, 2.0], [1, 2]), false);
    }

    #[test]
    fn rejects_an_incompatible_feature_count() {
        let node = ScalerNodeBuilder::new("scaler")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(ScalerConfig::new(Some(vec![1.0, 2.0, 3.0]), None))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "input".to_string(),
            Value::from_tensor_data(TensorData::new(vec![0.0_f32; 4], [2, 2]), 2, &device).unwrap(),
        );

        let error = scaler(&node, &env, &device).unwrap_err();

        assert_eq!(
            error,
            TynxError::Shape("Scaler scale has 3 values, expected 1 or 2".to_string())
        );
    }
}
