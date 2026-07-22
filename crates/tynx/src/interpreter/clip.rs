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
        Value::Scalar(Scalar::F64(value)) => Value::Scalar(Scalar::F64(
            value
                .max(min.map(|bound| bound.as_f64()).unwrap_or(f64::NEG_INFINITY))
                .min(max.map(|bound| bound.as_f64()).unwrap_or(f64::INFINITY)),
        )),
        Value::Scalar(Scalar::I64(value)) => Value::Scalar(Scalar::I64(
            value
                .max(min.map(scalar_as_i64).unwrap_or(i64::MIN))
                .min(max.map(scalar_as_i64).unwrap_or(i64::MAX)),
        )),
        Value::Scalar(Scalar::U64(value)) => Value::Scalar(Scalar::U64(
            value
                .max(min.map(scalar_as_u64).unwrap_or(u64::MIN))
                .min(max.map(scalar_as_u64).unwrap_or(u64::MAX)),
        )),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "Clip expects a numeric tensor, got {other:?}"
            )));
        }
    };

    Ok(vec![output])
}

fn scalar_as_i64(value: Scalar) -> i64 {
    match value {
        Scalar::I64(value) => value,
        Scalar::U64(value) => value as i64,
        Scalar::F64(value) => value as i64,
        Scalar::Bool(value) => i64::from(value),
    }
}

fn scalar_as_u64(value: Scalar) -> u64 {
    match value {
        Scalar::I64(value) => value as u64,
        Scalar::U64(value) => value,
        Scalar::F64(value) => value as u64,
        Scalar::Bool(value) => u64::from(value),
    }
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
