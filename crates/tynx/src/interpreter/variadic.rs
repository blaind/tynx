//! Element-wise Max, Min, Mean, and Sum execution.

use burn::tensor::Device;
use onnx_ir::{
    ir::Argument,
    node::{max::MaxNode, mean::MeanNode, min::MinNode, sum::SumNode},
};

use super::{Env, binary, resolve};
use crate::{DynInt, DynTensor, Result, Scalar, TynxError, Value};

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
    operation: fn(Value, Value, &Device) -> Result<Value>,
) -> Result<Vec<Value>> {
    if inputs.is_empty() {
        return Err(TynxError::Shape(format!(
            "node '{node_name}' has no inputs"
        )));
    }
    let mut output = resolve::at(env, node_name, inputs, 0, device)?;

    for index in 1..inputs.len() {
        let input = resolve::at(env, node_name, inputs, index, device)?;
        output = operation(output, input, device)?;
    }

    Ok(vec![output])
}

fn add_pair(left: Value, right: Value, device: &Device) -> Result<Value> {
    numeric_pair(left, right, VariadicOp::Add, "Sum/Mean", device)
}

fn max_pair(left: Value, right: Value, device: &Device) -> Result<Value> {
    numeric_pair(left, right, VariadicOp::Max, "Max", device)
}

fn min_pair(left: Value, right: Value, device: &Device) -> Result<Value> {
    numeric_pair(left, right, VariadicOp::Min, "Min", device)
}

#[derive(Debug, Clone, Copy)]
enum VariadicOp {
    Add,
    Max,
    Min,
}

fn numeric_pair(
    left: Value,
    right: Value,
    operation: VariadicOp,
    name: &str,
    device: &Device,
) -> Result<Value> {
    match (left, right) {
        (Value::Tensor(left), Value::Tensor(right)) => {
            float_pair(left, right, operation).map(Value::Tensor)
        }
        (Value::Tensor(left), Value::Scalar(right)) => {
            let dtype = left.dtype();
            float_pair(
                left,
                DynTensor::full(&[1], right.as_f64(), device, dtype)?,
                operation,
            )
            .map(Value::Tensor)
        }
        (Value::Scalar(left), Value::Tensor(right)) => {
            let dtype = right.dtype();
            float_pair(
                DynTensor::full(&[1], left.as_f64(), device, dtype)?,
                right,
                operation,
            )
            .map(Value::Tensor)
        }
        (Value::Int(left), Value::Int(right)) => int_pair(left, right, operation).map(Value::Int),
        (Value::Int(left), Value::Scalar(right)) => {
            let dtype = left.dtype();
            int_pair(
                left,
                binary::scalar_int_tensor(right, dtype, device)?,
                operation,
            )
            .map(Value::Int)
        }
        (Value::Scalar(left), Value::Int(right)) => {
            let dtype = right.dtype();
            int_pair(
                binary::scalar_int_tensor(left, dtype, device)?,
                right,
                operation,
            )
            .map(Value::Int)
        }
        (Value::Scalar(left), Value::Scalar(right)) => {
            scalar_pair(left, right, operation).map(Value::Scalar)
        }
        (Value::Shape(left), Value::Shape(right)) => {
            shape_pair(left, right, operation).map(Value::Shape)
        }
        (Value::Shape(left), Value::Int(right)) => {
            int_pair(binary::shape_tensor(left, device)?, right, operation).map(Value::Int)
        }
        (Value::Int(left), Value::Shape(right)) => {
            int_pair(left, binary::shape_tensor(right, device)?, operation).map(Value::Int)
        }
        (left, right) => Err(pair_mismatch(name, left, right)),
    }
}

fn float_pair(left: DynTensor, right: DynTensor, operation: VariadicOp) -> Result<DynTensor> {
    match operation {
        VariadicOp::Add => left.add_broadcast(right),
        VariadicOp::Max => left.max_broadcast(right),
        VariadicOp::Min => left.min_broadcast(right),
    }
}

fn int_pair(left: DynInt, right: DynInt, operation: VariadicOp) -> Result<DynInt> {
    match operation {
        VariadicOp::Add => left.add_broadcast(right),
        VariadicOp::Max => left.max_broadcast(right),
        VariadicOp::Min => left.min_broadcast(right),
    }
}

fn scalar_pair(left: Scalar, right: Scalar, operation: VariadicOp) -> Result<Scalar> {
    match operation {
        VariadicOp::Add => add_scalars(left, right),
        VariadicOp::Max => max_scalars(left, right),
        VariadicOp::Min => min_scalars(left, right),
    }
}

fn shape_pair(left: Vec<i64>, right: Vec<i64>, operation: VariadicOp) -> Result<Vec<i64>> {
    let output_len = left.len().max(right.len());
    if left.len() != output_len && left.len() != 1 || right.len() != output_len && right.len() != 1
    {
        return Err(TynxError::Shape(format!(
            "shape operands cannot broadcast lengths {} and {}",
            left.len(),
            right.len()
        )));
    }
    Ok((0..output_len)
        .map(|index| {
            let left = left[if left.len() == 1 { 0 } else { index }];
            let right = right[if right.len() == 1 { 0 } else { index }];
            match operation {
                VariadicOp::Add => left + right,
                VariadicOp::Max => left.max(right),
                VariadicOp::Min => left.min(right),
            }
        })
        .collect())
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

    #[test]
    fn takes_maximum_of_tensor_and_scalar() {
        let node = MaxNodeBuilder::new("max")
            .input_tensor("a", 1, DType::F32)
            .input_scalar("floor", DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "a".into(),
            Value::from_tensor_data(TensorData::new(vec![-2.0_f32, 1.0, 4.0], [3]), 1, &device)
                .unwrap(),
        );
        env.insert("floor".into(), Value::Scalar(Scalar::F64(0.5)));

        let output = max(&node, &env, &device).unwrap().pop().unwrap();

        assert_eq!(floats(output), [0.5, 1.0, 4.0]);
    }

    #[test]
    fn takes_maximum_of_shape_and_integer_tensor() {
        let node = MaxNodeBuilder::new("max")
            .input_shape("shape", 3)
            .input_tensor("limit", 1, DType::I64)
            .output_tensor("y", 1, DType::I64)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert("shape".into(), Value::Shape(vec![20, 8, 3]));
        env.insert(
            "limit".into(),
            Value::from_tensor_data(TensorData::new(vec![0_i64, 10, 5], [3]), 1, &device).unwrap(),
        );

        let output = max(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_int()
            .unwrap();

        assert_eq!(
            output.into_data().iter::<i64>().collect::<Vec<_>>(),
            [20, 10, 5]
        );
    }
}
