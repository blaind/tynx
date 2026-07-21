//! Rank-preserving unary operators.

use burn::tensor::Device;
use onnx_ir::node::{
    abs::AbsNode, acos::AcosNode, acosh::AcoshNode, asin::AsinNode, asinh::AsinhNode, cos::CosNode,
    cosh::CoshNode, exp::ExpNode, log::LogNode, neg::NegNode, relu::ReluNode, sigmoid::SigmoidNode,
    sin::SinNode, sinh::SinhNode, sqrt::SqrtNode, tan::TanNode, tanh::TanhNode,
};

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

pub(super) fn log(node: &LogNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.log())])
}

pub(super) fn sqrt(node: &SqrtNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.sqrt())])
}

pub(super) fn abs(node: &AbsNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.abs())])
}

pub(super) fn neg(node: &NegNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.negated())])
}

pub(super) fn sin(node: &SinNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.sin())])
}

pub(super) fn cos(node: &CosNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.cos())])
}

pub(super) fn tan(node: &TanNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.tan())])
}

pub(super) fn cosh(node: &CoshNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.cosh())])
}

pub(super) fn sinh(node: &SinhNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.sinh())])
}

pub(super) fn acos(node: &AcosNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.acos())])
}

pub(super) fn acosh(node: &AcoshNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.acosh())])
}

pub(super) fn asin(node: &AsinNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.asin())])
}

pub(super) fn asinh(node: &AsinhNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.asinh())])
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::{
            abs::AbsNodeBuilder, acos::AcosNodeBuilder, acosh::AcoshNodeBuilder,
            asin::AsinNodeBuilder, asinh::AsinhNodeBuilder, cos::CosNodeBuilder,
            cosh::CoshNodeBuilder, exp::ExpNodeBuilder, log::LogNodeBuilder, neg::NegNodeBuilder,
            relu::ReluNodeBuilder, sigmoid::SigmoidNodeBuilder, sin::SinNodeBuilder,
            sinh::SinhNodeBuilder, sqrt::SqrtNodeBuilder, tan::TanNodeBuilder,
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

    #[test]
    fn takes_natural_logarithms() {
        let node = LogNodeBuilder::new("log")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input = Value::from_tensor_data(
            TensorData::new(vec![1.0_f32, std::f32::consts::E], [2]),
            1,
            &device,
        )
        .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = log(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output[0], 0.0);
        assert!((output[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn takes_square_roots() {
        let node = SqrtNodeBuilder::new("sqrt")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![0.0_f32, 4.0, 9.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = sqrt(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [0.0, 2.0, 3.0]);
    }

    #[test]
    fn takes_absolute_values() {
        let node = AbsNodeBuilder::new("abs")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-2.0_f32, 0.0, 3.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = abs(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [2.0, 0.0, 3.0]);
    }

    #[test]
    fn negates_values() {
        let node = NegNodeBuilder::new("neg")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-2.0_f32, 0.0, 3.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = neg(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [2.0, 0.0, -3.0]);
    }

    #[test]
    fn takes_sines() {
        let node = SinNodeBuilder::new("sin")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input = Value::from_tensor_data(
            TensorData::new(
                vec![
                    0.0_f32,
                    std::f32::consts::FRAC_PI_2,
                    -std::f32::consts::FRAC_PI_2,
                ],
                [3],
            ),
            1,
            &device,
        )
        .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = sin(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [0.0, 1.0, -1.0]);
    }

    #[test]
    fn takes_cosines() {
        let node = CosNodeBuilder::new("cos")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input = Value::from_tensor_data(
            TensorData::new(
                vec![0.0_f32, std::f32::consts::PI, std::f32::consts::TAU],
                [3],
            ),
            1,
            &device,
        )
        .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = cos(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        for (actual, expected) in output.into_iter().zip([1.0, -1.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn takes_tangents() {
        let node = TanNodeBuilder::new("tan")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input = Value::from_tensor_data(
            TensorData::new(
                vec![
                    -std::f32::consts::FRAC_PI_4,
                    0.0_f32,
                    std::f32::consts::FRAC_PI_4,
                ],
                [3],
            ),
            1,
            &device,
        )
        .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = tan(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        for (actual, expected) in output.into_iter().zip([-1.0, 0.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn takes_hyperbolic_cosines() {
        let node = CoshNodeBuilder::new("cosh")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![0.0_f32, 1.0, -1.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = cosh(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        for (actual, expected) in output.into_iter().zip([1.0, 1.543_080_6, 1.543_080_6]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn takes_hyperbolic_sines() {
        let node = SinhNodeBuilder::new("sinh")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![0.0_f32, 1.0, -1.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = sinh(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        for (actual, expected) in output.into_iter().zip([0.0, 1.175_201_2, -1.175_201_2]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn takes_inverse_cosines() {
        let node = AcosNodeBuilder::new("acos")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![1.0_f32, 0.0, -1.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = acos(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        for (actual, expected) in
            output
                .into_iter()
                .zip([0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI])
        {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn takes_inverse_hyperbolic_cosines() {
        let node = AcoshNodeBuilder::new("acosh")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![1.0_f32, 2.0], [2]), 1, &device).unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = acosh(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        for (actual, expected) in output.into_iter().zip([0.0, 1.316_958]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn takes_inverse_sines() {
        let node = AsinNodeBuilder::new("asin")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-1.0_f32, 0.0, 1.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = asin(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        for (actual, expected) in output.into_iter().zip([
            -std::f32::consts::FRAC_PI_2,
            0.0,
            std::f32::consts::FRAC_PI_2,
        ]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn takes_inverse_hyperbolic_sines() {
        let node = AsinhNodeBuilder::new("asinh")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-1.0_f32, 0.0, 1.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = asinh(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        for (actual, expected) in output.into_iter().zip([-0.881_373_6, 0.0, 0.881_373_6]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
