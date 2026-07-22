//! ONNX integer and modulo operator execution.

use burn::tensor::Device;
use onnx_ir::{
    ir::Argument,
    node::{
        bitshift::{BitShiftNode, Direction},
        bitwiseand::BitwiseAndNode,
        bitwisenot::BitwiseNotNode,
        bitwiseor::BitwiseOrNode,
        bitwisexor::BitwiseXorNode,
        modulo::ModNode,
    },
};

use super::{Env, resolve};
use crate::{DynInt, Result, TynxError, Value};

pub(super) fn modulo(node: &ModNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let left = resolve::at(env, &node.name, &node.inputs, 0, device)?;
    let right = resolve::at(env, &node.name, &node.inputs, 1, device)?;
    let output = match (left, right) {
        (Value::Tensor(left), Value::Tensor(right)) => {
            Value::Tensor(left.modulo_broadcast(right, node.config.fmod)?)
        }
        (Value::Int(left), Value::Int(right)) => {
            Value::Int(left.modulo_broadcast(right, node.config.fmod)?)
        }
        (left, right) => {
            return Err(TynxError::TypeMismatch(format!(
                "Mod operands must have matching numeric tensor kinds, got {left:?} and {right:?}"
            )));
        }
    };
    Ok(vec![output])
}

pub(super) fn bitwise_and(node: &BitwiseAndNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    bitwise_binary(
        &node.name,
        &node.inputs,
        env,
        device,
        DynInt::bitwise_and_broadcast,
    )
}

pub(super) fn bitwise_or(node: &BitwiseOrNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    bitwise_binary(
        &node.name,
        &node.inputs,
        env,
        device,
        DynInt::bitwise_or_broadcast,
    )
}

pub(super) fn bitwise_xor(node: &BitwiseXorNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    bitwise_binary(
        &node.name,
        &node.inputs,
        env,
        device,
        DynInt::bitwise_xor_broadcast,
    )
}

pub(super) fn bitwise_not(node: &BitwiseNotNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_int()?;
    Ok(vec![Value::Int(input.bitwise_not())])
}

pub(super) fn bitshift(node: &BitShiftNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let operation = match node.config.direction {
        Direction::Left => DynInt::bitwise_left_shift_broadcast,
        Direction::Right => DynInt::bitwise_right_shift_broadcast,
    };
    bitwise_binary(&node.name, &node.inputs, env, device, operation)
}

fn bitwise_binary(
    node_name: &str,
    inputs: &[Argument],
    env: &Env,
    device: &Device,
    operation: impl FnOnce(DynInt, DynInt) -> Result<DynInt>,
) -> Result<Vec<Value>> {
    let left = resolve::at(env, node_name, inputs, 0, device)?.into_int()?;
    let right = resolve::at(env, node_name, inputs, 1, device)?.into_int()?;
    Ok(vec![Value::Int(operation(left, right)?)])
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;

    #[test]
    fn floor_mod_follows_the_divisor_sign() {
        let device = Device::default();
        let left = DynInt::from_data(TensorData::new(vec![-3_i64, 3], [2]), 1, &device).unwrap();
        let right = DynInt::from_data(TensorData::new(vec![2_i64, -2], [2]), 1, &device).unwrap();

        let output = left
            .modulo_broadcast(right, false)
            .unwrap()
            .into_data()
            .iter::<i64>()
            .collect::<Vec<_>>();

        assert_eq!(output, [1, -1]);
    }
}
