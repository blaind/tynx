//! Numeric Cast and CastLike execution.

use burn::tensor::{DType, Device, TensorData};
use onnx_ir::{
    ir::Argument,
    node::{cast::CastNode, cast_like::CastLikeNode},
};

use super::{Env, resolve};
use crate::{Result, Scalar, TynxError, Value};

pub(super) fn cast(node: &CastNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    cast_input(&node.name, &node.inputs, node.config.to, env, device)
}

pub(super) fn cast_like(node: &CastLikeNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    cast_input(&node.name, &node.inputs, node.config.to, env, device)
}

fn cast_input(
    name: &str,
    inputs: &[Argument],
    target: DType,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::first(env, name, inputs, device)?;
    Ok(vec![cast_value(input, target, device)?])
}

fn cast_value(input: Value, target: DType, device: &Device) -> Result<Value> {
    if target.is_float() {
        return Ok(match input {
            Value::Tensor(tensor) => Value::Tensor(tensor.cast(target)),
            Value::Int(tensor) => Value::Tensor(tensor.to_float(target)),
            Value::Bool(tensor) => Value::Tensor(tensor.to_float(target)),
            Value::Scalar(scalar) => Value::Scalar(Scalar::F64(scalar.as_f64())),
            Value::Shape(values) => {
                let rank = values.len();
                let value = Value::from_tensor_data(TensorData::new(values, [rank]), 1, device)?;
                cast_value(value, target, device)?
            }
        });
    }

    if target.is_int() || target.is_uint() {
        return Ok(match input {
            Value::Tensor(tensor) => Value::Int(tensor.to_int(target)),
            Value::Int(tensor) => Value::Int(tensor.cast(target)),
            Value::Bool(tensor) => Value::Int(tensor.to_int(target)),
            Value::Scalar(scalar) if target.is_uint() => {
                Value::Scalar(Scalar::U64(scalar.as_f64() as u64))
            }
            Value::Scalar(scalar) => Value::Scalar(Scalar::I64(scalar.as_f64() as i64)),
            Value::Shape(values) => Value::Shape(values),
        });
    }

    if target.is_bool() {
        return Ok(match input {
            Value::Tensor(tensor) => Value::Bool(tensor.to_bool()),
            Value::Int(tensor) => Value::Bool(tensor.to_bool()),
            Value::Bool(tensor) => Value::Bool(tensor),
            Value::Scalar(scalar) => Value::Scalar(Scalar::Bool(scalar.as_f64() != 0.0)),
            Value::Shape(values) => {
                let rank = values.len();
                let value = Value::from_tensor_data(TensorData::new(values, [rank]), 1, device)?;
                cast_value(value, target, device)?
            }
        });
    }

    Err(TynxError::TypeMismatch(format!(
        "casting to dtype {target:?} is unsupported"
    )))
}

#[cfg(test)]
mod tests {
    use burn::tensor::{BoolStore, TensorData};
    use onnx_ir::{
        DType,
        node::{
            cast::{CastConfig, CastNodeBuilder},
            cast_like::{CastLikeConfig, CastLikeNodeBuilder},
        },
    };

    use super::*;
    use crate::DynTensor;

    #[test]
    fn casts_float_tensor_to_explicit_integer_dtype() {
        let node = CastNodeBuilder::new("cast")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::I16)
            .config(CastConfig::new(DType::I16))
            .build();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::Tensor(
                DynTensor::from_data(
                    TensorData::new(vec![-2.8_f32, 0.0, 3.9], [3]),
                    1,
                    &Device::default(),
                )
                .unwrap(),
            ),
        );

        let output = cast(&node, &env, &Device::default()).unwrap();
        let Value::Int(output) = output.into_iter().next().unwrap() else {
            panic!("expected integer tensor output");
        };
        assert_eq!(output.dtype(), DType::I16);
        assert_eq!(
            output.into_data().iter::<i64>().collect::<Vec<_>>(),
            [-2, 0, 3]
        );
    }

    #[test]
    fn cast_like_uses_configured_target_dtype() {
        let node = CastLikeNodeBuilder::new("cast_like")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::Bool(BoolStore::Native))
            .config(CastLikeConfig::new(
                DType::Bool(BoolStore::Native),
                None,
                None,
            ))
            .build();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::Tensor(
                DynTensor::from_data(
                    TensorData::new(vec![0.0_f32, -1.0, 2.0], [3]),
                    1,
                    &Device::default(),
                )
                .unwrap(),
            ),
        );

        let output = cast_like(&node, &env, &Device::default()).unwrap();
        let Value::Bool(output) = output.into_iter().next().unwrap() else {
            panic!("expected boolean tensor output");
        };
        assert_eq!(
            output.into_data().iter::<bool>().collect::<Vec<_>>(),
            [false, true, true]
        );
    }
}
