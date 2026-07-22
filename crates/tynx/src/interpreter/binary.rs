//! Element-wise binary operators.

use burn::tensor::Device;
use onnx_ir::node::{
    arithmetic::{AddNode, DivNode, MulNode, SubNode},
    prelu::PReluNode,
};

use super::{Env, resolve};
use crate::{DynInt, DynTensor, Result, Scalar, TynxError, Value};

pub(super) fn add(node: &AddNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    numeric_binary(
        &node.name,
        &node.inputs,
        env,
        device,
        DynTensor::add_broadcast,
        DynInt::add_broadcast,
        add_scalars,
    )
}

pub(super) fn sub(node: &SubNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    numeric_binary(
        &node.name,
        &node.inputs,
        env,
        device,
        DynTensor::sub_broadcast,
        DynInt::sub_broadcast,
        sub_scalars,
    )
}

pub(super) fn mul(node: &MulNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    numeric_binary(
        &node.name,
        &node.inputs,
        env,
        device,
        DynTensor::mul_broadcast,
        DynInt::mul_broadcast,
        mul_scalars,
    )
}

pub(super) fn div(node: &DivNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    numeric_binary(
        &node.name,
        &node.inputs,
        env,
        device,
        DynTensor::div_broadcast,
        DynInt::div_broadcast,
        div_scalars,
    )
}

fn numeric_binary(
    node_name: &str,
    inputs: &[onnx_ir::Argument],
    env: &Env,
    device: &Device,
    float_op: impl FnOnce(DynTensor, DynTensor) -> Result<DynTensor>,
    int_op: impl FnOnce(DynInt, DynInt) -> Result<DynInt>,
    scalar_op: fn(Scalar, Scalar) -> Result<Scalar>,
) -> Result<Vec<Value>> {
    let left = resolve::at(env, node_name, inputs, 0, device)?;
    let right = resolve::at(env, node_name, inputs, 1, device)?;
    let output = match (left, right) {
        (Value::Tensor(left), Value::Tensor(right)) => Value::Tensor(float_op(left, right)?),
        (Value::Int(left), Value::Int(right)) => Value::Int(int_op(left, right)?),
        (Value::Scalar(left), Value::Scalar(right)) => Value::Scalar(scalar_op(left, right)?),
        (Value::Shape(left), Value::Shape(right)) => {
            Value::Shape(shape_binary(left, right, scalar_op)?)
        }
        (left, right) => {
            return Err(TynxError::TypeMismatch(format!(
                "numeric operands must have matching tensor kinds, got {left:?} and {right:?}"
            )));
        }
    };
    Ok(vec![output])
}

fn add_scalars(left: Scalar, right: Scalar) -> Result<Scalar> {
    scalar_arithmetic(left, right, |a, b| a + b, |a, b| a + b, |a, b| a + b)
}

fn sub_scalars(left: Scalar, right: Scalar) -> Result<Scalar> {
    scalar_arithmetic(left, right, |a, b| a - b, |a, b| a - b, |a, b| a - b)
}

fn mul_scalars(left: Scalar, right: Scalar) -> Result<Scalar> {
    scalar_arithmetic(left, right, |a, b| a * b, |a, b| a * b, |a, b| a * b)
}

fn div_scalars(left: Scalar, right: Scalar) -> Result<Scalar> {
    match (left, right) {
        (Scalar::F64(left), Scalar::F64(right)) => Ok(Scalar::F64(left / right)),
        (Scalar::I64(left), Scalar::I64(right)) => left
            .checked_div(right)
            .map(Scalar::I64)
            .ok_or_else(|| TynxError::Shape("invalid signed integer division".to_string())),
        (Scalar::U64(left), Scalar::U64(right)) => left
            .checked_div(right)
            .map(Scalar::U64)
            .ok_or_else(|| TynxError::Shape("integer division by zero".to_string())),
        (left, right) => Err(scalar_kind_error(left, right)),
    }
}

fn scalar_arithmetic(
    left: Scalar,
    right: Scalar,
    float_op: fn(f64, f64) -> f64,
    signed_op: fn(i64, i64) -> i64,
    unsigned_op: fn(u64, u64) -> u64,
) -> Result<Scalar> {
    match (left, right) {
        (Scalar::F64(left), Scalar::F64(right)) => Ok(Scalar::F64(float_op(left, right))),
        (Scalar::I64(left), Scalar::I64(right)) => Ok(Scalar::I64(signed_op(left, right))),
        (Scalar::U64(left), Scalar::U64(right)) => Ok(Scalar::U64(unsigned_op(left, right))),
        (left, right) => Err(scalar_kind_error(left, right)),
    }
}

fn scalar_kind_error(left: Scalar, right: Scalar) -> TynxError {
    TynxError::TypeMismatch(format!(
        "numeric scalar kinds differ: {left:?} and {right:?}"
    ))
}

fn shape_binary(
    left: Vec<i64>,
    right: Vec<i64>,
    operation: fn(Scalar, Scalar) -> Result<Scalar>,
) -> Result<Vec<i64>> {
    if left.len() != right.len() {
        return Err(TynxError::Shape(format!(
            "shape operands have different lengths: {} and {}",
            left.len(),
            right.len()
        )));
    }
    left.into_iter()
        .zip(right)
        .map(
            |(left, right)| match operation(Scalar::I64(left), Scalar::I64(right))? {
                Scalar::I64(value) => Ok(value),
                _ => unreachable!("shape arithmetic preserves signed integers"),
            },
        )
        .collect()
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
