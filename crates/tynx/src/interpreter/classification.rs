//! Element-wise ONNX numeric classification operators.

use burn::tensor::Device;
use onnx_ir::node::is_inf::IsInfNode;

use super::{Env, resolve};
use crate::{Result, Scalar, TynxError, Value};

pub(super) fn is_inf(node: &IsInfNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let output = match input {
        Value::Tensor(tensor) => Value::Bool(
            tensor.is_inf_signed(node.config.detect_negative, node.config.detect_positive),
        ),
        Value::Scalar(Scalar::F64(value)) => Value::Scalar(Scalar::Bool(
            value.is_infinite()
                && ((value.is_sign_negative() && node.config.detect_negative)
                    || (value.is_sign_positive() && node.config.detect_positive)),
        )),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "IsInf expects a floating-point tensor, got {other:?}"
            )));
        }
    };
    Ok(vec![output])
}

#[cfg(test)]
mod tests {
    use burn::tensor::{BoolStore, TensorData};
    use onnx_ir::{
        DType,
        node::is_inf::{IsInfConfig, IsInfNodeBuilder},
    };

    use super::*;

    #[test]
    fn detects_only_positive_infinity() {
        let node = IsInfNodeBuilder::new("is_inf")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::Bool(BoolStore::Native))
            .config(IsInfConfig {
                detect_negative: false,
                detect_positive: true,
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(vec![f32::NEG_INFINITY, -1.0, f32::INFINITY, f32::NAN], [4]),
                1,
                &device,
            )
            .unwrap(),
        );

        let output = is_inf(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_bool()
            .unwrap()
            .into_data()
            .iter::<bool>()
            .collect::<Vec<_>>();

        assert_eq!(output, [false, false, true, false]);
    }
}
