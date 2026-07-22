//! ONNX matrix multiplication execution.

use burn::tensor::{DType, Device, linalg::det as burn_det};
use onnx_ir::node::{
    det::DetNode, gemm::GemmNode, matmul::MatMulNode, matmulinteger::MatMulIntegerNode,
};

use super::{Env, resolve, shape};
use crate::{DynInt, DynTensor, Result, Scalar, TynxError, Value};

pub(super) fn matmul(node: &MatMulNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let left = resolve::at(env, &node.name, &node.inputs, 0, device)?;
    let right = resolve::at(env, &node.name, &node.inputs, 1, device)?;
    Ok(vec![matmul_values(left, right, device)?])
}

pub(super) fn det(node: &DetNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    let output = match input {
        DynTensor::R1(_) => {
            return Err(TynxError::Shape(
                "Det requires a tensor with rank at least 2".to_string(),
            ));
        }
        DynTensor::R2(tensor) => {
            let [rows, columns] = tensor.dims();
            let determinant = burn_det::<3, 2, 1>(tensor.reshape([1, rows, columns]));
            Value::from_tensor_data(determinant.into_data(), 0, device)?
        }
        DynTensor::R3(tensor) => Value::Tensor(DynTensor::R1(burn_det::<3, 2, 1>(tensor))),
        DynTensor::R4(tensor) => Value::Tensor(DynTensor::R2(burn_det::<4, 3, 2>(tensor))),
        DynTensor::R5(tensor) => Value::Tensor(DynTensor::R3(burn_det::<5, 4, 3>(tensor))),
        DynTensor::R6(tensor) => Value::Tensor(DynTensor::R4(burn_det::<6, 5, 4>(tensor))),
    };
    Ok(vec![output])
}

pub(super) fn gemm(node: &GemmNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let mut left = resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?;
    let mut right = resolve::at(env, &node.name, &node.inputs, 1, device)?.into_tensor()?;
    if left.rank() != 2 || right.rank() != 2 {
        return Err(TynxError::Shape(format!(
            "Gemm requires rank-2 A and B, got {} and {}",
            left.rank(),
            right.rank()
        )));
    }
    if node.config.trans_a != 0 {
        left = left.permute(vec![1, 0])?;
    }
    if node.config.trans_b != 0 {
        right = right.permute(vec![1, 0])?;
    }

    validate_matrix_dims(&left.dims(), &right.dims())?;
    let dtype = left.dtype();
    let mut output = left
        .matmul(right.cast(dtype))?
        .mul_scalar(node.config.alpha as f64);

    if node.inputs.get(2).is_some_and(|input| !input.is_optional()) {
        let bias = resolve::at(env, &node.name, &node.inputs, 2, device)?;
        output = match bias {
            Value::Tensor(bias) => {
                output.add_broadcast(bias.cast(dtype).mul_scalar(node.config.beta as f64))?
            }
            Value::Scalar(bias) => output.add_scalar(node.config.beta as f64 * bias.as_f64()),
            other => {
                return Err(TynxError::TypeMismatch(format!(
                    "Gemm bias must be a float tensor or scalar, got {other:?}"
                )));
            }
        };
    }

    Ok(vec![Value::Tensor(output)])
}

pub(super) fn matmul_integer(
    node: &MatMulIntegerNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let left = resolve::at(env, &node.name, &node.inputs, 0, device)?
        .into_int()?
        .cast(DType::I32);
    let right = resolve::at(env, &node.name, &node.inputs, 1, device)?
        .into_int()?
        .cast(DType::I32);
    let left_zero = optional_zero_point(node, env, device, 2)?;
    let right_zero = optional_zero_point(node, env, device, 3)?;

    Ok(vec![matmul_values(
        Value::Int(left.sub_scalar(left_zero)),
        Value::Int(right.sub_scalar(right_zero)),
        device,
    )?])
}

fn optional_zero_point(
    node: &MatMulIntegerNode,
    env: &Env,
    device: &Device,
    index: usize,
) -> Result<i64> {
    if !node
        .inputs
        .get(index)
        .is_some_and(|input| !input.is_optional())
    {
        return Ok(0);
    }
    let value = resolve::at(env, &node.name, &node.inputs, index, device)?;
    match value {
        Value::Scalar(Scalar::I64(value)) => Ok(value),
        Value::Scalar(Scalar::U64(value)) => i64::try_from(value)
            .map_err(|_| TynxError::Shape(format!("zero point {value} exceeds i64"))),
        Value::Int(tensor) if tensor.dims().iter().product::<usize>() == 1 => tensor
            .into_data()
            .iter::<i64>()
            .next()
            .ok_or_else(|| TynxError::Shape("empty zero-point tensor".to_string())),
        other => Err(TynxError::TypeMismatch(format!(
            "MatMulInteger zero point must be scalar, got {other:?}"
        ))),
    }
}

pub(super) fn matmul_values(left: Value, right: Value, device: &Device) -> Result<Value> {
    match (left, right) {
        (Value::Tensor(left), Value::Tensor(right)) => {
            let dtype = left.dtype();
            let (left, right, left_vector, right_vector) = prepare_float(left, right.cast(dtype))?;
            let output = Value::Tensor(left.matmul(right)?);
            finish_matmul(output, left_vector, right_vector, device)
        }
        (Value::Int(left), Value::Int(right)) => {
            let dtype = left.dtype();
            let (left, right, left_vector, right_vector) = prepare_int(left, right.cast(dtype))?;
            let output = Value::Int(left.matmul(right)?);
            finish_matmul(output, left_vector, right_vector, device)
        }
        (left, right) => Err(TynxError::TypeMismatch(format!(
            "MatMul inputs must have matching numeric tensor kinds, got {left:?} and {right:?}"
        ))),
    }
}

fn prepare_float(
    mut left: DynTensor,
    mut right: DynTensor,
) -> Result<(DynTensor, DynTensor, bool, bool)> {
    let left_vector = left.rank() == 1;
    let right_vector = right.rank() == 1;
    if left_vector {
        let length = left.dims()[0];
        left = left.reshape(vec![1, length])?;
    }
    if right_vector {
        let length = right.dims()[0];
        right = right.reshape(vec![length, 1])?;
    }
    let rank = left.rank().max(right.rank());
    left = left.to_rank(rank)?;
    right = right.to_rank(rank)?;
    validate_matrix_dims(&left.dims(), &right.dims())?;
    Ok((left, right, left_vector, right_vector))
}

fn prepare_int(mut left: DynInt, mut right: DynInt) -> Result<(DynInt, DynInt, bool, bool)> {
    let left_vector = left.rank() == 1;
    let right_vector = right.rank() == 1;
    if left_vector {
        let length = left.dims()[0];
        left = left.reshape(vec![1, length])?;
    }
    if right_vector {
        let length = right.dims()[0];
        right = right.reshape(vec![length, 1])?;
    }
    let rank = left.rank().max(right.rank());
    left = left.to_rank(rank)?;
    right = right.to_rank(rank)?;
    validate_matrix_dims(&left.dims(), &right.dims())?;
    Ok((left, right, left_vector, right_vector))
}

fn validate_matrix_dims(left: &[usize], right: &[usize]) -> Result<()> {
    if left.len() != right.len() || left.len() < 2 {
        return Err(TynxError::Shape(format!(
            "MatMul requires matching ranks >= 2, got {left:?} and {right:?}"
        )));
    }
    if left[left.len() - 1] != right[right.len() - 2] {
        return Err(TynxError::Shape(format!(
            "MatMul inner dimensions differ: {} and {}",
            left[left.len() - 1],
            right[right.len() - 2]
        )));
    }
    for (&left, &right) in left[..left.len() - 2].iter().zip(&right[..right.len() - 2]) {
        if left != right && left != 1 && right != 1 {
            return Err(TynxError::Shape(format!(
                "MatMul batch dimensions are not broadcastable: {left} and {right}"
            )));
        }
    }
    Ok(())
}

fn finish_matmul(
    output: Value,
    left_vector: bool,
    right_vector: bool,
    device: &Device,
) -> Result<Value> {
    let mut dims = shape::value_dims(&output);
    if left_vector {
        let axis = dims.len() - 2;
        dims.remove(axis);
    }
    if right_vector {
        dims.pop();
    }
    shape::reshape_value(output, dims, device)
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::{det::DetNodeBuilder, matmul::MatMulNodeBuilder},
    };

    use super::*;

    #[test]
    fn multiplies_two_matrices() {
        let node = MatMulNodeBuilder::new("matmul")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "a".into(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
                2,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "b".into(),
            Value::from_tensor_data(
                TensorData::new(vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2]),
                2,
                &device,
            )
            .unwrap(),
        );

        let output = matmul(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn computes_a_matrix_determinant() {
        let node = DetNodeBuilder::new("det")
            .input_tensor("x", 2, DType::F32)
            .output_scalar("y", DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]),
                2,
                &device,
            )
            .unwrap(),
        );

        let output = det(&node, &env, &device).unwrap();

        assert!(matches!(
            output.as_slice(),
            [Value::Scalar(Scalar::F64(value))] if (*value + 2.0).abs() < 1e-6
        ));
    }
}
