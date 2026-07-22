//! Softmax-family operator execution.

use burn::tensor::Device;
use onnx_ir::node::{log_softmax::LogSoftmaxNode, softmax::SoftmaxNode};

use super::{Env, resolve};
use crate::{Result, Value};

pub(super) fn softmax(node: &SoftmaxNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.softmax(node.config.axis))])
}

pub(super) fn log_softmax(node: &LogSoftmaxNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.log_softmax(node.config.axis))])
}

#[cfg(test)]
mod tests {
    use burn::tensor::{TensorData, Tolerance};
    use onnx_ir::{
        DType,
        node::{
            log_softmax::{LogSoftmaxConfig, LogSoftmaxNodeBuilder},
            softmax::{SoftmaxConfig, SoftmaxNodeBuilder},
        },
    };

    use super::*;
    use crate::DynTensor;

    #[test]
    fn applies_softmax_along_selected_axis() {
        let node = SoftmaxNodeBuilder::new("softmax")
            .input_tensor("x", 2, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .config(SoftmaxConfig::new(0))
            .build();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::Tensor(
                DynTensor::from_data(
                    TensorData::new(vec![0.0_f32, 0.0, 1.0, 2.0], [2, 2]),
                    2,
                    &Device::default(),
                )
                .unwrap(),
            ),
        );

        let output = softmax(&node, &env, &Device::default()).unwrap();
        let Value::Tensor(output) = output.into_iter().next().unwrap() else {
            panic!("expected tensor output");
        };
        output.into_data().assert_approx_eq(
            &TensorData::new(
                vec![0.268_941_43_f32, 0.119_202_92, 0.731_058_6, 0.880_797],
                [2, 2],
            ),
            Tolerance::<f32>::absolute(1e-5),
        );
    }

    #[test]
    fn applies_log_softmax_stably() {
        let node = LogSoftmaxNodeBuilder::new("log_softmax")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .config(LogSoftmaxConfig::new(0))
            .build();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::Tensor(
                DynTensor::from_data(
                    TensorData::new(vec![10_000.0_f32, 10_001.0], [2]),
                    1,
                    &Device::default(),
                )
                .unwrap(),
            ),
        );

        let output = log_softmax(&node, &env, &Device::default()).unwrap();
        let Value::Tensor(output) = output.into_iter().next().unwrap() else {
            panic!("expected tensor output");
        };
        output.into_data().assert_approx_eq(
            &TensorData::new(vec![-1.313_261_6_f32, -0.313_261_66], [2]),
            Tolerance::<f32>::absolute(1e-4),
        );
    }
}
