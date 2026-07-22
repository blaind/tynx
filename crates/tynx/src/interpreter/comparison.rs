//! Element-wise ONNX comparison execution.

use burn::tensor::{DType, Device, TensorData};
use onnx_ir::node::comparison::{
    EqualNode, GreaterNode, GreaterOrEqualNode, LessNode, LessOrEqualNode,
};

use super::{Env, resolve};
use crate::{DynInt, DynTensor, Result, Scalar, TynxError, Value};

#[derive(Debug, Clone, Copy)]
enum Comparison {
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
}

pub(super) fn equal(node: &EqualNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    compare(&node.name, &node.inputs, env, device, Comparison::Equal)
}

pub(super) fn greater(node: &GreaterNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    compare(&node.name, &node.inputs, env, device, Comparison::Greater)
}

pub(super) fn greater_or_equal(
    node: &GreaterOrEqualNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    compare(
        &node.name,
        &node.inputs,
        env,
        device,
        Comparison::GreaterOrEqual,
    )
}

pub(super) fn less(node: &LessNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    compare(&node.name, &node.inputs, env, device, Comparison::Less)
}

pub(super) fn less_or_equal(
    node: &LessOrEqualNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    compare(
        &node.name,
        &node.inputs,
        env,
        device,
        Comparison::LessOrEqual,
    )
}

fn compare(
    node_name: &str,
    inputs: &[onnx_ir::ir::Argument],
    env: &Env,
    device: &Device,
    comparison: Comparison,
) -> Result<Vec<Value>> {
    let left = resolve::at(env, node_name, inputs, 0, device)?;
    let right = resolve::at(env, node_name, inputs, 1, device)?;

    let output = match (left, right) {
        (Value::Tensor(left), Value::Tensor(right)) => Value::Bool(match comparison {
            Comparison::Equal => left.equal_broadcast(right)?,
            Comparison::Greater => left.greater_broadcast(right)?,
            Comparison::GreaterOrEqual => left.greater_equal_broadcast(right)?,
            Comparison::Less => left.less_broadcast(right)?,
            Comparison::LessOrEqual => left.less_equal_broadcast(right)?,
        }),
        (Value::Tensor(left), Value::Scalar(right)) => {
            let dtype = left.dtype();
            compare_floats(
                left,
                DynTensor::full(&[1], right.as_f64(), device, dtype)?,
                comparison,
            )?
        }
        (Value::Scalar(left), Value::Tensor(right)) => {
            let dtype = right.dtype();
            compare_floats(
                DynTensor::full(&[1], left.as_f64(), device, dtype)?,
                right,
                comparison,
            )?
        }
        (Value::Int(left), Value::Int(right)) => Value::Bool(match comparison {
            Comparison::Equal => left.equal_broadcast(right)?,
            Comparison::Greater => left.greater_broadcast(right)?,
            Comparison::GreaterOrEqual => left.greater_equal_broadcast(right)?,
            Comparison::Less => left.less_broadcast(right)?,
            Comparison::LessOrEqual => left.less_equal_broadcast(right)?,
        }),
        (Value::Int(left), Value::Scalar(right)) => {
            let dtype = left.dtype();
            compare_ints(left, scalar_int_tensor(right, dtype, device)?, comparison)?
        }
        (Value::Scalar(left), Value::Int(right)) => {
            let dtype = right.dtype();
            compare_ints(scalar_int_tensor(left, dtype, device)?, right, comparison)?
        }
        (Value::Scalar(left), Value::Scalar(right)) => {
            Value::Scalar(Scalar::Bool(compare_scalars(left, right, comparison)?))
        }
        (Value::Shape(left), Value::Shape(right)) => {
            if left.len() != right.len() {
                return Err(TynxError::Shape(format!(
                    "comparison shape operands have different lengths: {} and {}",
                    left.len(),
                    right.len()
                )));
            }
            let values = left
                .into_iter()
                .zip(right)
                .map(|(left, right)| {
                    compare_scalars(Scalar::I64(left), Scalar::I64(right), comparison)
                })
                .collect::<Result<Vec<_>>>()?;
            Value::from_tensor_data(TensorData::new(values.clone(), [values.len()]), 1, device)?
        }
        (Value::Shape(left), Value::Int(right)) => {
            compare_ints(shape_tensor(left, device)?, right, comparison)?
        }
        (Value::Int(left), Value::Shape(right)) => {
            compare_ints(left, shape_tensor(right, device)?, comparison)?
        }
        (left, right) => {
            return Err(TynxError::TypeMismatch(format!(
                "comparison inputs must have matching numeric kinds, got {left:?} and {right:?}"
            )));
        }
    };

    Ok(vec![output])
}

fn compare_floats(left: DynTensor, right: DynTensor, comparison: Comparison) -> Result<Value> {
    Ok(Value::Bool(match comparison {
        Comparison::Equal => left.equal_broadcast(right)?,
        Comparison::Greater => left.greater_broadcast(right)?,
        Comparison::GreaterOrEqual => left.greater_equal_broadcast(right)?,
        Comparison::Less => left.less_broadcast(right)?,
        Comparison::LessOrEqual => left.less_equal_broadcast(right)?,
    }))
}

fn compare_ints(left: DynInt, right: DynInt, comparison: Comparison) -> Result<Value> {
    Ok(Value::Bool(match comparison {
        Comparison::Equal => left.equal_broadcast(right)?,
        Comparison::Greater => left.greater_broadcast(right)?,
        Comparison::GreaterOrEqual => left.greater_equal_broadcast(right)?,
        Comparison::Less => left.less_broadcast(right)?,
        Comparison::LessOrEqual => left.less_equal_broadcast(right)?,
    }))
}

fn shape_tensor(values: Vec<i64>, device: &Device) -> Result<DynInt> {
    let length = values.len();
    DynInt::from_data(TensorData::new(values, [length]), 1, device)
}

fn scalar_int_tensor(scalar: Scalar, dtype: DType, device: &Device) -> Result<DynInt> {
    let data = if dtype.is_uint() {
        TensorData::new(vec![scalar.as_f64() as u64], [1])
    } else {
        TensorData::new(vec![scalar.as_f64() as i64], [1])
    };
    Ok(DynInt::from_data(data, 1, device)?.cast(dtype))
}

fn compare_scalars(left: Scalar, right: Scalar, comparison: Comparison) -> Result<bool> {
    macro_rules! compare {
        ($left:expr, $right:expr) => {
            Ok(match comparison {
                Comparison::Equal => $left == $right,
                Comparison::Greater => $left > $right,
                Comparison::GreaterOrEqual => $left >= $right,
                Comparison::Less => $left < $right,
                Comparison::LessOrEqual => $left <= $right,
            })
        };
    }

    match (left, right) {
        (Scalar::F64(left), Scalar::F64(right)) => compare!(left, right),
        (Scalar::I64(left), Scalar::I64(right)) => compare!(left, right),
        (Scalar::U64(left), Scalar::U64(right)) => compare!(left, right),
        (Scalar::Bool(left), Scalar::Bool(right)) if matches!(comparison, Comparison::Equal) => {
            Ok(left == right)
        }
        (left, right) => Err(TynxError::TypeMismatch(format!(
            "comparison scalar kinds differ: {left:?} and {right:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{BoolStore, TensorData};
    use onnx_ir::{DType, node::comparison::LessNodeBuilder};

    use super::*;

    #[test]
    fn compares_integer_tensors_with_broadcasting() {
        let node = LessNodeBuilder::new("less")
            .input_tensor("left", 2, DType::I32)
            .input_tensor("right", 1, DType::I32)
            .output_tensor("output", 2, DType::Bool(BoolStore::Native))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "left".to_string(),
            Value::from_tensor_data(TensorData::new(vec![1_i32, 4, 3, 2], [2, 2]), 2, &device)
                .unwrap(),
        );
        env.insert(
            "right".to_string(),
            Value::from_tensor_data(TensorData::new(vec![2_i32, 3], [2]), 1, &device).unwrap(),
        );

        let output = less(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_bool()
            .unwrap()
            .into_data()
            .iter::<bool>()
            .collect::<Vec<_>>();

        assert_eq!(output, [true, false, false, true]);
    }

    #[test]
    fn compares_tensor_against_scalar() {
        let node = LessNodeBuilder::new("less")
            .input_tensor("left", 1, DType::F32)
            .input_scalar("right", DType::F32)
            .output_tensor("output", 1, DType::Bool(BoolStore::Native))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "left".to_string(),
            Value::from_tensor_data(TensorData::new(vec![1.0_f32, 3.0], [2]), 1, &device).unwrap(),
        );
        env.insert("right".to_string(), Value::Scalar(Scalar::F64(2.0)));

        let output = less(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_bool()
            .unwrap()
            .into_data()
            .iter::<bool>()
            .collect::<Vec<_>>();

        assert_eq!(output, [true, false]);
    }
}
