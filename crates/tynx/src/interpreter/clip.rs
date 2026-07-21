//! ONNX Clip execution.

use burn::tensor::Device;
use onnx_ir::node::clip::{ClipInput, ClipNode};

use super::{Env, resolve};
use crate::{Result, Scalar, TynxError, Value};

pub(super) fn clip(node: &ClipNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let min = bound(&node.config.min, node, env, device)?;
    let max = bound(&node.config.max, node, env, device)?;
    let input = resolve::first(env, &node.name, &node.inputs, device)?;

    let output = match input {
        Value::Tensor(tensor) => Value::Tensor(tensor.clip(
            min.map(|value| value.as_f64()),
            max.map(|value| value.as_f64()),
        )),
        Value::Int(tensor) => Value::Int(tensor.clip(min, max)),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "Clip expects a numeric tensor, got {other:?}"
            )));
        }
    };

    Ok(vec![output])
}

fn bound(
    input: &Option<ClipInput>,
    node: &ClipNode,
    env: &Env,
    device: &Device,
) -> Result<Option<Scalar>> {
    match input {
        None => Ok(None),
        Some(ClipInput::Static(value)) => Ok(Some(Scalar::F64(*value))),
        Some(ClipInput::Runtime(reference)) => Ok(Some(
            resolve::at(env, &node.name, &node.inputs, reference.input_index, device)?
                .into_scalar()?,
        )),
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::clip::{ClipConfig, ClipInput, ClipNodeBuilder},
    };

    use super::*;

    #[test]
    fn clips_float_values_to_static_bounds() {
        let node = ClipNodeBuilder::new("clip")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .config(ClipConfig {
                min: Some(ClipInput::Static(-1.0)),
                max: Some(ClipInput::Static(1.0)),
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".to_string(),
            Value::from_tensor_data(TensorData::new(vec![-2.0_f32, 0.0, 3.0], [3]), 1, &device)
                .unwrap(),
        );

        let output = clip(&node, &env, &device)
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
}
