//! ONNX Range execution.

use burn::tensor::{Device, TensorData};
use onnx_ir::node::range::{RangeInput, RangeNode};

use super::{Env, resolve};
use crate::{DynInt, DynTensor, Result, Scalar, TynxError, Value};

pub(super) fn range(node: &RangeNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let start = input_value(&node.name, &node.inputs, &node.config.start, env, device)?;
    let limit = input_value(&node.name, &node.inputs, &node.config.limit, env, device)?;
    let delta = input_value(&node.name, &node.inputs, &node.config.delta, env, device)?;
    if delta == 0.0 {
        return Err(TynxError::Shape("Range delta must not be zero".to_string()));
    }

    let mut values = Vec::new();
    let mut current = start;
    while if delta > 0.0 {
        current < limit
    } else {
        current > limit
    } {
        values.push(current);
        current += delta;
    }

    let dtype = node.outputs[0].ty.elem_type();
    let length = values.len();
    let output = if dtype.is_float() {
        Value::Tensor(
            DynTensor::from_data(TensorData::new(values, [length]), 1, device)?.cast(dtype),
        )
    } else if dtype.is_int() || dtype.is_uint() {
        let values = values
            .into_iter()
            .map(|value| value as i64)
            .collect::<Vec<_>>();
        Value::Int(DynInt::from_data(TensorData::new(values, [length]), 1, device)?.cast(dtype))
    } else {
        return Err(TynxError::TypeMismatch(format!(
            "Range output dtype {dtype:?} is unsupported"
        )));
    };

    Ok(vec![output])
}

fn input_value(
    node_name: &str,
    inputs: &[onnx_ir::Argument],
    input: &RangeInput,
    env: &Env,
    device: &Device,
) -> Result<f64> {
    match input {
        RangeInput::Static(value) => Ok(*value as f64),
        RangeInput::Runtime(reference) => {
            let value = resolve::at(env, node_name, inputs, reference.input_index, device)?;
            match value {
                Value::Scalar(Scalar::F64(value)) => Ok(value),
                Value::Scalar(Scalar::I64(value)) => Ok(value as f64),
                Value::Scalar(Scalar::U64(value)) => Ok(value as f64),
                Value::Tensor(value) if value.dims().iter().product::<usize>() == 1 => value
                    .into_data()
                    .iter::<f64>()
                    .next()
                    .ok_or_else(|| TynxError::Shape("empty Range input".to_string())),
                Value::Int(value) if value.dims().iter().product::<usize>() == 1 => value
                    .into_data()
                    .iter::<i64>()
                    .next()
                    .map(|value| value as f64)
                    .ok_or_else(|| TynxError::Shape("empty Range input".to_string())),
                other => Err(TynxError::TypeMismatch(format!(
                    "Range input must be scalar, got {other:?}"
                ))),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_descending_integer_range() {
        let node = onnx_ir::node::range::RangeNodeBuilder::new("range")
            .input_scalar("start", onnx_ir::DType::I64)
            .input_scalar("limit", onnx_ir::DType::I64)
            .input_scalar("delta", onnx_ir::DType::I64)
            .output_tensor("output", 1, onnx_ir::DType::I64)
            .config(onnx_ir::node::range::RangeConfig::new(
                RangeInput::Static(5),
                RangeInput::Static(0),
                RangeInput::Static(-2),
            ))
            .build();

        let output = range(&node, &Env::new(), &Device::default())
            .unwrap()
            .pop()
            .unwrap()
            .into_int()
            .unwrap()
            .into_data()
            .iter::<i64>()
            .collect::<Vec<_>>();

        assert_eq!(output, [5, 3, 1]);
    }
}
