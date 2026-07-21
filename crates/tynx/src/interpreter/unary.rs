//! Rank-preserving unary operators.

use burn::tensor::Device;
use onnx_ir::node::{exp::ExpNode, relu::ReluNode, sigmoid::SigmoidNode, tanh::TanhNode};

use super::{Env, resolve};
use crate::{Result, Value};

pub(super) fn relu(node: &ReluNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.relu())])
}

pub(super) fn sigmoid(node: &SigmoidNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.sigmoid())])
}

pub(super) fn tanh(node: &TanhNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.tanh())])
}

pub(super) fn exp(node: &ExpNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.exp())])
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::{
            exp::ExpNodeBuilder, relu::ReluNodeBuilder, sigmoid::SigmoidNodeBuilder,
            tanh::TanhNodeBuilder,
        },
    };

    use super::*;

    #[test]
    fn clamps_negative_values() {
        let node = ReluNodeBuilder::new("relu")
            .input_tensor("x", 2, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .build();
        let device = Device::default();
        let input = Value::from_tensor_data(
            TensorData::new(vec![-2.0_f32, 0.0, 3.0, -4.0], [2, 2]),
            2,
            &device,
        )
        .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = relu(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data();

        assert_eq!(
            output.iter::<f32>().collect::<Vec<_>>(),
            [0.0, 0.0, 3.0, 0.0]
        );
    }

    #[test]
    fn sigmoid_maps_values_to_zero_one_range() {
        let node = SigmoidNodeBuilder::new("sigmoid")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![0.0_f32, 1.098_612_3], [2]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = sigmoid(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn tanh_maps_values_to_minus_one_one_range() {
        let node = TanhNodeBuilder::new("tanh")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-1.0_f32, 0.0, 1.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = tanh(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert!((output[0] + 0.761_594_2).abs() < 1e-6);
        assert_eq!(output[1], 0.0);
        assert!((output[2] - 0.761_594_2).abs() < 1e-6);
    }

    #[test]
    fn exponentiates_values() {
        let node = ExpNodeBuilder::new("exp")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![0.0_f32, 1.0], [2]), 1, &device).unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = exp(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output[0], 1.0);
        assert!((output[1] - std::f32::consts::E).abs() < 1e-6);
    }
}
