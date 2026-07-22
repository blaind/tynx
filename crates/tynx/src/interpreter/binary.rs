//! Element-wise binary operators.

use burn::tensor::{DType, Device, TensorData};
use onnx_ir::node::{
    arithmetic::{AddNode, DivNode, MulNode, SubNode},
    prelu::PReluNode,
};

use super::{Env, resolve};
use crate::{DynInt, DynTensor, Result, Scalar, TynxError, Value};

pub(super) fn add(node: &AddNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    numeric_binary(&node.name, &node.inputs, env, device, BinaryOp::Add)
}

pub(super) fn sub(node: &SubNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    numeric_binary(&node.name, &node.inputs, env, device, BinaryOp::Sub)
}

pub(super) fn mul(node: &MulNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    numeric_binary(&node.name, &node.inputs, env, device, BinaryOp::Mul)
}

pub(super) fn div(node: &DivNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    numeric_binary(&node.name, &node.inputs, env, device, BinaryOp::Div)
}

#[derive(Debug, Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

fn numeric_binary(
    node_name: &str,
    inputs: &[onnx_ir::Argument],
    env: &Env,
    device: &Device,
    operation: BinaryOp,
) -> Result<Vec<Value>> {
    let left = resolve::at(env, node_name, inputs, 0, device)?;
    let right = resolve::at(env, node_name, inputs, 1, device)?;
    let output = match (left, right) {
        (Value::Tensor(left), Value::Tensor(right)) => {
            Value::Tensor(float_binary(left, right, operation)?)
        }
        (Value::Tensor(left), Value::Scalar(right)) => {
            let dtype = left.dtype();
            let right = DynTensor::full(&[1], right.as_f64(), device, dtype)?;
            Value::Tensor(float_binary(left, right, operation)?)
        }
        (Value::Scalar(left), Value::Tensor(right)) => {
            let dtype = right.dtype();
            let left = DynTensor::full(&[1], left.as_f64(), device, dtype)?;
            Value::Tensor(float_binary(left, right, operation)?)
        }
        (Value::Int(left), Value::Int(right)) => Value::Int(int_binary(left, right, operation)?),
        (Value::Int(left), Value::Scalar(right)) => {
            let right = scalar_int_tensor(right, left.dtype(), device)?;
            Value::Int(int_binary(left, right, operation)?)
        }
        (Value::Scalar(left), Value::Int(right)) => {
            let left = scalar_int_tensor(left, right.dtype(), device)?;
            Value::Int(int_binary(left, right, operation)?)
        }
        (Value::Scalar(left), Value::Scalar(right)) => {
            Value::Scalar(scalar_binary(left, right, operation)?)
        }
        (Value::Shape(left), Value::Shape(right)) => {
            Value::Shape(shape_binary(left, right, operation)?)
        }
        (Value::Shape(left), Value::Int(right)) => {
            Value::Int(int_binary(shape_tensor(left, device)?, right, operation)?)
        }
        (Value::Int(left), Value::Shape(right)) => {
            Value::Int(int_binary(left, shape_tensor(right, device)?, operation)?)
        }
        (left, right) => {
            return Err(TynxError::TypeMismatch(format!(
                "numeric operands must have matching tensor kinds, got {left:?} and {right:?}"
            )));
        }
    };
    Ok(vec![output])
}

fn float_binary(left: DynTensor, right: DynTensor, operation: BinaryOp) -> Result<DynTensor> {
    match operation {
        BinaryOp::Add => left.add_broadcast(right),
        BinaryOp::Sub => left.sub_broadcast(right),
        BinaryOp::Mul => left.mul_broadcast(right),
        BinaryOp::Div => left.div_broadcast(right),
    }
}

fn int_binary(left: DynInt, right: DynInt, operation: BinaryOp) -> Result<DynInt> {
    match operation {
        BinaryOp::Add => left.add_broadcast(right),
        BinaryOp::Sub => left.sub_broadcast(right),
        BinaryOp::Mul => left.mul_broadcast(right),
        BinaryOp::Div => left.div_broadcast(right),
    }
}

fn scalar_binary(left: Scalar, right: Scalar, operation: BinaryOp) -> Result<Scalar> {
    macro_rules! apply {
        ($left:expr, $right:expr, $kind:ident) => {{
            let value = match operation {
                BinaryOp::Add => $left + $right,
                BinaryOp::Sub => $left - $right,
                BinaryOp::Mul => $left * $right,
                BinaryOp::Div => $left / $right,
            };
            Ok(Scalar::$kind(value))
        }};
    }
    match (left, right) {
        (Scalar::F64(left), Scalar::F64(right)) => apply!(left, right, F64),
        (Scalar::I64(left), Scalar::I64(right)) => apply!(left, right, I64),
        (Scalar::U64(left), Scalar::U64(right)) => apply!(left, right, U64),
        (left, right) => Err(scalar_kind_error(left, right)),
    }
}

fn scalar_kind_error(left: Scalar, right: Scalar) -> TynxError {
    TynxError::TypeMismatch(format!(
        "numeric scalar kinds differ: {left:?} and {right:?}"
    ))
}

fn shape_binary(left: Vec<i64>, right: Vec<i64>, operation: BinaryOp) -> Result<Vec<i64>> {
    if left.len() != right.len() {
        return Err(TynxError::Shape(format!(
            "shape operands have different lengths: {} and {}",
            left.len(),
            right.len()
        )));
    }
    left.into_iter()
        .zip(right)
        .map(|(left, right)| {
            match scalar_binary(Scalar::I64(left), Scalar::I64(right), operation)? {
                Scalar::I64(value) => Ok(value),
                _ => unreachable!("shape arithmetic preserves signed integers"),
            }
        })
        .collect()
}

fn shape_tensor(values: Vec<i64>, device: &Device) -> Result<DynInt> {
    let length = values.len();
    DynInt::from_data(TensorData::new(values, [length]), 1, device)
}

fn scalar_int_tensor(scalar: Scalar, dtype: DType, device: &Device) -> Result<DynInt> {
    let data = if dtype.is_uint() {
        let value = match scalar {
            Scalar::U64(value) => value,
            Scalar::I64(value) => value as u64,
            Scalar::F64(value) => value as u64,
            Scalar::Bool(value) => u64::from(value),
        };
        TensorData::new(vec![value], [1])
    } else {
        let value = match scalar {
            Scalar::U64(value) => value as i64,
            Scalar::I64(value) => value,
            Scalar::F64(value) => value as i64,
            Scalar::Bool(value) => i64::from(value),
        };
        TensorData::new(vec![value], [1])
    };
    Ok(DynInt::from_data(data, 1, device)?.cast(dtype))
}

pub(super) fn prelu(node: &PReluNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?;
    let slope = resolve::at(env, &node.name, &node.inputs, 1, device)?.into_tensor()?;

    Ok(vec![Value::Tensor(input.prelu(slope)?)])
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::{
            arithmetic::{AddNodeBuilder, DivNodeBuilder, MulNodeBuilder, SubNodeBuilder},
            prelu::PReluNodeBuilder,
        },
    };

    use super::*;

    #[test]
    fn adds_with_multidirectional_broadcasting() {
        let node = AddNodeBuilder::new("add")
            .input_tensor("matrix", 2, DType::F32)
            .input_tensor("row", 1, DType::F32)
            .output_tensor("sum", 2, DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "matrix".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]),
                2,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "row".to_string(),
            Value::from_tensor_data(TensorData::new(vec![10.0_f32, 20.0], [2]), 1, &device)
                .unwrap(),
        );

        let output = add(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn subtracts_in_operand_order() {
        let node = SubNodeBuilder::new("sub")
            .input_tensor("matrix", 2, DType::F32)
            .input_tensor("row", 1, DType::F32)
            .output_tensor("difference", 2, DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "matrix".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![10.0_f32, 20.0, 30.0, 40.0], [2, 2]),
                2,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "row".to_string(),
            Value::from_tensor_data(TensorData::new(vec![1.0_f32, 2.0], [2]), 1, &device).unwrap(),
        );

        let output = sub(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [9.0, 18.0, 29.0, 38.0]);
    }

    #[test]
    fn multiplies_with_multidirectional_broadcasting() {
        let node = MulNodeBuilder::new("mul")
            .input_tensor("matrix", 2, DType::F32)
            .input_tensor("row", 1, DType::F32)
            .output_tensor("product", 2, DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "matrix".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]),
                2,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "row".to_string(),
            Value::from_tensor_data(TensorData::new(vec![10.0_f32, 20.0], [2]), 1, &device)
                .unwrap(),
        );

        let output = mul(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [10.0, 40.0, 30.0, 80.0]);
    }

    #[test]
    fn divides_in_operand_order_with_broadcasting() {
        let node = DivNodeBuilder::new("div")
            .input_tensor("matrix", 2, DType::F32)
            .input_tensor("row", 1, DType::F32)
            .output_tensor("quotient", 2, DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "matrix".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![10.0_f32, 40.0, 30.0, 80.0], [2, 2]),
                2,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "row".to_string(),
            Value::from_tensor_data(TensorData::new(vec![10.0_f32, 20.0], [2]), 1, &device)
                .unwrap(),
        );

        let output = div(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn applies_prelu_with_unidirectional_broadcasting() {
        let node = PReluNodeBuilder::new("prelu")
            .input_tensor("x", 2, DType::F32)
            .input_tensor("slope", 1, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![-1.0_f32, -2.0, -3.0, -4.0], [2, 2]),
                2,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "slope".to_string(),
            Value::from_tensor_data(TensorData::new(vec![0.1_f32, 0.2], [2]), 1, &device).unwrap(),
        );

        let output = prelu(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [-0.1, -0.4, -0.3, -0.8]);
    }
}
