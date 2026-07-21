//! Element-wise Max, Min, Mean, and Sum execution.

use burn::tensor::Device;
use onnx_ir::{
    ir::Argument,
    node::{max::MaxNode, mean::MeanNode, min::MinNode, sum::SumNode},
};

use super::{Env, resolve};
use crate::{Result, Scalar, TynxError, Value};

pub(super) fn max(node: &MaxNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    fold(node.name.as_str(), &node.inputs, env, device, max_pair)
}

pub(super) fn min(node: &MinNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    fold(node.name.as_str(), &node.inputs, env, device, min_pair)
}

pub(super) fn sum(node: &SumNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    fold(node.name.as_str(), &node.inputs, env, device, add_pair)
}

pub(super) fn mean(node: &MeanNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let count = node.inputs.len();
    let mut output = fold(node.name.as_str(), &node.inputs, env, device, add_pair)?
        .pop()
        .ok_or_else(|| TynxError::Shape("Mean has no inputs".to_string()))?;

    output = match output {
        Value::Tensor(tensor) => Value::Tensor(tensor.div_scalar(count as f64)),
        Value::Scalar(Scalar::F64(value)) => Value::Scalar(Scalar::F64(value / count as f64)),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "Mean expects floating-point inputs, got {other:?}"
            )));
        }
    };

    Ok(vec![output])
}

fn fold(
    node_name: &str,
    inputs: &[Argument],
    env: &Env,
    device: &Device,
    operation: fn(Value, Value) -> Result<Value>,
) -> Result<Vec<Value>> {
    if inputs.is_empty() {
        return Err(TynxError::Shape(format!(
            "node '{node_name}' has no inputs"
        )));
    }
    let mut output = resolve::at(env, node_name, inputs, 0, device)?;

    for index in 1..inputs.len() {
        let input = resolve::at(env, node_name, inputs, index, device)?;
        output = operation(output, input)?;
    }

    Ok(vec![output])
}

fn add_pair(left: Value, right: Value) -> Result<Value> {
    match (left, right) {
        (Value::Tensor(left), Value::Tensor(right)) => {
            Ok(Value::Tensor(left.add_broadcast(right)?))
        }
        (Value::Int(left), Value::Int(right)) => Ok(Value::Int(left.add_broadcast(right)?)),
        (Value::Scalar(left), Value::Scalar(right)) => add_scalars(left, right).map(Value::Scalar),
        (left, right) => Err(pair_mismatch("Sum/Mean", left, right)),
    }
}

fn max_pair(left: Value, right: Value) -> Result<Value> {
    match (left, right) {
        (Value::Tensor(left), Value::Tensor(right)) => {
            Ok(Value::Tensor(left.max_broadcast(right)?))
        }
        (Value::Int(left), Value::Int(right)) => Ok(Value::Int(left.max_broadcast(right)?)),
        (Value::Scalar(left), Value::Scalar(right)) => max_scalars(left, right).map(Value::Scalar),
        (left, right) => Err(pair_mismatch("Max", left, right)),
    }
}

fn min_pair(left: Value, right: Value) -> Result<Value> {
    match (left, right) {
        (Value::Tensor(left), Value::Tensor(right)) => {
            Ok(Value::Tensor(left.min_broadcast(right)?))
        }
        (Value::Int(left), Value::Int(right)) => Ok(Value::Int(left.min_broadcast(right)?)),
        (Value::Scalar(left), Value::Scalar(right)) => min_scalars(left, right).map(Value::Scalar),
        (left, right) => Err(pair_mismatch("Min", left, right)),
    }
}

fn add_scalars(left: Scalar, right: Scalar) -> Result<Scalar> {
    match (left, right) {
        (Scalar::F64(left), Scalar::F64(right)) => Ok(Scalar::F64(left + right)),
        (Scalar::I64(left), Scalar::I64(right)) => Ok(Scalar::I64(left + right)),
        (Scalar::U64(left), Scalar::U64(right)) => Ok(Scalar::U64(left + right)),
        (left, right) => Err(scalar_mismatch(left, right)),
    }
}

fn max_scalars(left: Scalar, right: Scalar) -> Result<Scalar> {
    match (left, right) {
        (Scalar::F64(left), Scalar::F64(right)) => Ok(Scalar::F64(left.max(right))),
        (Scalar::I64(left), Scalar::I64(right)) => Ok(Scalar::I64(left.max(right))),
        (Scalar::U64(left), Scalar::U64(right)) => Ok(Scalar::U64(left.max(right))),
        (left, right) => Err(scalar_mismatch(left, right)),
    }
}

fn min_scalars(left: Scalar, right: Scalar) -> Result<Scalar> {
    match (left, right) {
        (Scalar::F64(left), Scalar::F64(right)) => Ok(Scalar::F64(left.min(right))),
        (Scalar::I64(left), Scalar::I64(right)) => Ok(Scalar::I64(left.min(right))),
        (Scalar::U64(left), Scalar::U64(right)) => Ok(Scalar::U64(left.min(right))),
        (left, right) => Err(scalar_mismatch(left, right)),
    }
}

fn pair_mismatch(operator: &str, left: Value, right: Value) -> TynxError {
    TynxError::TypeMismatch(format!(
        "{operator} inputs must have matching numeric kinds, got {left:?} and {right:?}"
    ))
}

fn scalar_mismatch(left: Scalar, right: Scalar) -> TynxError {
    TynxError::TypeMismatch(format!(
        "scalar inputs must have matching numeric kinds, got {left:?} and {right:?}"
    ))
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::{
            max::MaxNodeBuilder, mean::MeanNodeBuilder, min::MinNodeBuilder, sum::SumNodeBuilder,
        },
    };

    use super::*;

    fn float_env(device: &Device) -> Env {
        let mut env = Env::new();
        env.insert(
            "a".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 4.0, 3.0, 2.0], [2, 2]),
                2,
                device,
            )
            .unwrap(),
        );
        env.insert(
            "b".to_string(),
            Value::from_tensor_data(TensorData::new(vec![2.0_f32, 3.0], [2]), 1, device).unwrap(),
        );
        env
    }

    fn floats(output: Value) -> Vec<f32> {
        output
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect()
    }

    #[test]
    fn takes_elementwise_maximum_with_broadcasting() {
        let node = MaxNodeBuilder::new("max")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 1, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .build();
        let device = Device::default();

        let output = max(&node, &float_env(&device), &device)
            .unwrap()
            .pop()
            .unwrap();

        assert_eq!(floats(output), [2.0, 4.0, 3.0, 3.0]);
    }

    #[test]
    fn takes_elementwise_minimum_with_broadcasting() {
        let node = MinNodeBuilder::new("min")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 1, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .build();
        let device = Device::default();

        let output = min(&node, &float_env(&device), &device)
            .unwrap()
            .pop()
            .unwrap();

        assert_eq!(floats(output), [1.0, 3.0, 2.0, 2.0]);
    }

    #[test]
    fn averages_inputs_with_broadcasting() {
        let node = MeanNodeBuilder::new("mean")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 1, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .build();
        let device = Device::default();

        let output = mean(&node, &float_env(&device), &device)
            .unwrap()
            .pop()
            .unwrap();

        assert_eq!(floats(output), [1.5, 3.5, 2.5, 2.5]);
    }

    #[test]
    fn sums_inputs_with_broadcasting() {
        let node = SumNodeBuilder::new("sum")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 1, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .build();
        let device = Device::default();

        let output = sum(&node, &float_env(&device), &device)
            .unwrap()
            .pop()
            .unwrap();

        assert_eq!(floats(output), [3.0, 7.0, 5.0, 5.0]);
    }
}
