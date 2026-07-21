//! Rank-preserving unary operators.

use burn::tensor::Device;
use onnx_ir::node::{
    abs::AbsNode, acos::AcosNode, acosh::AcoshNode, asin::AsinNode, asinh::AsinhNode,
    atan::AtanNode, atanh::AtanhNode, ceil::CeilNode, celu::CeluNode, cos::CosNode, cosh::CoshNode,
    elu::EluNode, erf::ErfNode, exp::ExpNode, floor::FloorNode, hard_sigmoid::HardSigmoidNode,
    leaky_relu::LeakyReluNode, log::LogNode, mish::MishNode, neg::NegNode,
    reciprocal::ReciprocalNode, relu::ReluNode, round::RoundNode, selu::SeluNode,
    sigmoid::SigmoidNode, sign::SignNode, sin::SinNode, sinh::SinhNode, softplus::SoftplusNode,
    softsign::SoftsignNode, sqrt::SqrtNode, tan::TanNode, tanh::TanhNode,
    thresholded_relu::ThresholdedReluNode,
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

pub(super) fn atan(node: &AtanNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.atan())])
}

pub(super) fn atanh(node: &AtanhNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.atanh())])
}

pub(super) fn erf(node: &ErfNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.erf())])
}

pub(super) fn ceil(node: &CeilNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.ceil())])
}

pub(super) fn floor(node: &FloorNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.floor())])
}

pub(super) fn round(node: &RoundNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.round())])
}

pub(super) fn reciprocal(node: &ReciprocalNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.reciprocal())])
}

pub(super) fn sign(node: &SignNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.sign())])
}

pub(super) fn softplus(node: &SoftplusNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.softplus())])
}

pub(super) fn elu(node: &EluNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.elu(node.config.alpha))])
}

pub(super) fn leaky_relu(node: &LeakyReluNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.leaky_relu(node.config.alpha))])
}

pub(super) fn selu(node: &SeluNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(
        input.selu(node.config.alpha, node.config.gamma),
    )])
}

pub(super) fn softsign(node: &SoftsignNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.softsign())])
}

pub(super) fn hard_sigmoid(
    node: &HardSigmoidNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(
        input.hard_sigmoid(node.config.alpha, node.config.beta),
    )])
}

pub(super) fn thresholded_relu(
    node: &ThresholdedReluNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(
        input.thresholded_relu(node.config.alpha),
    )])
}

pub(super) fn celu(node: &CeluNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.celu(node.config.alpha))])
}

pub(super) fn mish(node: &MishNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.mish())])
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::{
            abs::AbsNodeBuilder,
            acos::AcosNodeBuilder,
            acosh::AcoshNodeBuilder,
            asin::AsinNodeBuilder,
            asinh::AsinhNodeBuilder,
            atan::AtanNodeBuilder,
            atanh::AtanhNodeBuilder,
            ceil::CeilNodeBuilder,
            celu::{CeluConfig, CeluNodeBuilder},
            cos::CosNodeBuilder,
            cosh::CoshNodeBuilder,
            elu::{EluConfig, EluNodeBuilder},
            erf::ErfNodeBuilder,
            exp::ExpNodeBuilder,
            floor::FloorNodeBuilder,
            hard_sigmoid::{HardSigmoidConfig, HardSigmoidNodeBuilder},
            leaky_relu::{LeakyReluConfig, LeakyReluNodeBuilder},
            log::LogNodeBuilder,
            mish::MishNodeBuilder,
            neg::NegNodeBuilder,
            reciprocal::ReciprocalNodeBuilder,
            relu::ReluNodeBuilder,
            round::RoundNodeBuilder,
            selu::{SeluConfig, SeluNodeBuilder},
            sigmoid::SigmoidNodeBuilder,
            sign::SignNodeBuilder,
            sin::SinNodeBuilder,
            sinh::SinhNodeBuilder,
            softplus::SoftplusNodeBuilder,
            softsign::SoftsignNodeBuilder,
            sqrt::SqrtNodeBuilder,
            tan::TanNodeBuilder,
            tanh::TanhNodeBuilder,
            thresholded_relu::{ThresholdedReluConfig, ThresholdedReluNodeBuilder},
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

    #[test]
    fn takes_inverse_tangents() {
        let node = AtanNodeBuilder::new("atan")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-1.0_f32, 0.0, 1.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = atan(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        for (actual, expected) in output.into_iter().zip([
            -std::f32::consts::FRAC_PI_4,
            0.0,
            std::f32::consts::FRAC_PI_4,
        ]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn takes_inverse_hyperbolic_tangents() {
        let node = AtanhNodeBuilder::new("atanh")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-0.5_f32, 0.0, 0.5], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = atanh(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        for (actual, expected) in output.into_iter().zip([-0.549_306_15, 0.0, 0.549_306_15]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn applies_error_function() {
        let node = ErfNodeBuilder::new("erf")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-1.0_f32, 0.0, 1.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = erf(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        for (actual, expected) in output.into_iter().zip([-0.842_700_8, 0.0, 0.842_700_8]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn rounds_toward_positive_infinity() {
        let node = CeilNodeBuilder::new("ceil")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-1.2_f32, 0.0, 1.2], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = ceil(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [-1.0, 0.0, 2.0]);
    }

    #[test]
    fn rounds_toward_negative_infinity() {
        let node = FloorNodeBuilder::new("floor")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-1.2_f32, 0.0, 1.2], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = floor(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [-2.0, 0.0, 1.0]);
    }

    #[test]
    fn rounds_halfway_values_to_even() {
        let node = RoundNodeBuilder::new("round")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input = Value::from_tensor_data(
            TensorData::new(vec![-2.5_f32, -1.5, -0.5, 0.5, 1.5, 2.5], [6]),
            1,
            &device,
        )
        .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = round(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [-2.0, -2.0, 0.0, 0.0, 2.0, 2.0]);
    }

    #[test]
    fn takes_reciprocals() {
        let node = ReciprocalNodeBuilder::new("reciprocal")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-2.0_f32, 0.5, 4.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = reciprocal(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [-0.5, 2.0, 0.25]);
    }

    #[test]
    fn takes_signs() {
        let node = SignNodeBuilder::new("sign")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-3.0_f32, 0.0, 4.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = sign(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [-1.0, 0.0, 1.0]);
    }

    #[test]
    fn applies_softplus() {
        let node = SoftplusNodeBuilder::new("softplus")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-1.0_f32, 0.0, 1.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = softplus(&node, &env, &device)
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
                .zip([0.313_261_7, std::f32::consts::LN_2, 1.313_261_6])
        {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn applies_elu_with_alpha() {
        let node = EluNodeBuilder::new("elu")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .config(EluConfig::new(2.0))
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-1.0_f32, 0.0, 1.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = elu(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert!((output[0] + 1.264_241_1).abs() < 1e-6);
        assert_eq!(output[1], 0.0);
        assert_eq!(output[2], 1.0);
    }

    #[test]
    fn applies_leaky_relu_with_alpha() {
        let node = LeakyReluNodeBuilder::new("leaky_relu")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .config(LeakyReluConfig::new(0.2))
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-2.0_f32, 0.0, 3.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = leaky_relu(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [-0.4, 0.0, 3.0]);
    }

    #[test]
    fn applies_selu_with_custom_coefficients() {
        let node = SeluNodeBuilder::new("selu")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .config(SeluConfig::new(2.0, 3.0))
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-1.0_f32, 0.0, 1.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = selu(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert!((output[0] + 3.792_723_4).abs() < 1e-6);
        assert_eq!(output[1], 0.0);
        assert_eq!(output[2], 3.0);
    }

    #[test]
    fn applies_softsign() {
        let node = SoftsignNodeBuilder::new("softsign")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-2.0_f32, 0.0, 2.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = softsign(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        for (actual, expected) in output.into_iter().zip([-2.0 / 3.0, 0.0, 2.0 / 3.0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn applies_hard_sigmoid_with_custom_coefficients() {
        let node = HardSigmoidNodeBuilder::new("hard_sigmoid")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .config(HardSigmoidConfig::new(0.25, 0.5))
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-4.0_f32, 0.0, 4.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = hard_sigmoid(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [0.0, 0.5, 1.0]);
    }

    #[test]
    fn applies_thresholded_relu_with_custom_alpha() {
        let node = ThresholdedReluNodeBuilder::new("thresholded_relu")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .config(ThresholdedReluConfig::new(0.5))
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-1.0_f32, 0.5, 2.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = thresholded_relu(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [0.0, 0.0, 2.0]);
    }

    #[test]
    fn applies_celu_with_custom_alpha() {
        let node = CeluNodeBuilder::new("celu")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .config(CeluConfig::new(2.0))
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-2.0_f32, 0.0, 1.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = celu(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert!((output[0] + 1.264_241_1).abs() < 1e-6);
        assert_eq!(output[1], 0.0);
        assert_eq!(output[2], 1.0);
    }

    #[test]
    fn applies_mish() {
        let node = MishNodeBuilder::new("mish")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let input =
            Value::from_tensor_data(TensorData::new(vec![-1.0_f32, 0.0, 1.0], [3]), 1, &device)
                .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = mish(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        for (actual, expected) in output.into_iter().zip([-0.303_401_47, 0.0, 0.865_098_36]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
