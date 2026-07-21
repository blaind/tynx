//! Element-wise ONNX logical execution.

use burn::tensor::Device;
use onnx_ir::node::{and::AndNode, not::NotNode, or::OrNode, xor::XorNode};

use super::{Env, resolve};
use crate::{Result, TynxError, Value};

pub(super) fn and(node: &AndNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    binary(&node.name, &node.inputs, env, device, |left, right| {
        left.and_broadcast(right)
    })
}

pub(super) fn or(node: &OrNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    binary(&node.name, &node.inputs, env, device, |left, right| {
        left.or_broadcast(right)
    })
}

pub(super) fn xor(node: &XorNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    binary(&node.name, &node.inputs, env, device, |left, right| {
        left.xor_broadcast(right)
    })
}

pub(super) fn not(node: &NotNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    match input {
        Value::Bool(input) => Ok(vec![Value::Bool(input.logical_not())]),
        other => Err(TynxError::TypeMismatch(format!(
            "Not expects a boolean tensor, got {other:?}"
        ))),
    }
}

fn binary(
    node_name: &str,
    inputs: &[onnx_ir::ir::Argument],
    env: &Env,
    device: &Device,
    operation: fn(crate::DynBool, crate::DynBool) -> Result<crate::DynBool>,
) -> Result<Vec<Value>> {
    let left = resolve::at(env, node_name, inputs, 0, device)?;
    let right = resolve::at(env, node_name, inputs, 1, device)?;
    match (left, right) {
        (Value::Bool(left), Value::Bool(right)) => Ok(vec![Value::Bool(operation(left, right)?)]),
        (left, right) => Err(TynxError::TypeMismatch(format!(
            "logical inputs must be boolean tensors, got {left:?} and {right:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{BoolStore, TensorData};
    use onnx_ir::{DType, node::and::AndNodeBuilder};

    use super::*;

    #[test]
    fn applies_logical_and_with_broadcasting() {
        let bool_dtype = DType::Bool(BoolStore::Native);
        let node = AndNodeBuilder::new("and")
            .input_tensor("left", 2, bool_dtype)
            .input_tensor("right", 1, bool_dtype)
            .output_tensor("output", 2, bool_dtype)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "left".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![true, true, false, false], [2, 2]),
                2,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "right".to_string(),
            Value::from_tensor_data(TensorData::new(vec![true, false], [2]), 1, &device).unwrap(),
        );

        let output = and(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_bool()
            .unwrap()
            .into_data()
            .iter::<bool>()
            .collect::<Vec<_>>();

        assert_eq!(output, [true, false, false, false]);
    }
}
