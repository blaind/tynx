//! ONNX matrix multiplication execution.

use burn::tensor::{DType, Device, linalg::det as burn_det};
use onnx_ir::node::{
    det::DetNode, gemm::GemmNode, linear::LinearNode, matmul::MatMulNode,
    matmulinteger::MatMulIntegerNode,
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
    let dims = input.dims();
    if dims.len() < 2 {
        return Err(TynxError::Shape(
            "Det requires a tensor with rank at least 2".to_string(),
        ));
    }
    if dims[dims.len() - 2] != dims[dims.len() - 1] {
        return Err(TynxError::Shape(format!(
            "Det requires square matrices, got trailing dimensions {} and {}",
            dims[dims.len() - 2],
            dims[dims.len() - 1]
        )));
    }
    let output = match input {
        DynTensor::R1(_) => return Err(TynxError::Shape("Det rank invariant".to_string())),
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

pub(super) fn linear(node: &LinearNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?;
    let mut weight = resolve::at(env, &node.name, &node.inputs, 1, device)?.into_tensor()?;
    if node.config.transpose_weight {
        weight = weight.permute(vec![1, 0])?;
    }

    let dtype = input.dtype();
    let mut output = matmul_values(
        Value::Tensor(input),
        Value::Tensor(weight.cast(dtype)),
        device,
    )?
    .into_tensor()?;

    if node.inputs.get(2).is_some_and(|input| !input.is_optional()) {
        let bias = resolve::at(env, &node.name, &node.inputs, 2, device)?;
        output = match bias {
            Value::Tensor(bias) => output.add_broadcast(bias.cast(dtype))?,
            Value::Scalar(bias) => output.add_scalar(bias.as_f64()),
            other => {
                return Err(TynxError::TypeMismatch(format!(
                    "Linear bias must be a float tensor or scalar, got {other:?}"
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
    let left = subtract_zero_point(left, left_zero, 2)?;
    let right = subtract_zero_point(right, right_zero, 1)?;

    Ok(vec![matmul_values(
        Value::Int(left),
        Value::Int(right),
        device,
    )?])
}

enum MatMulZeroPoint {
    Scalar(i64),
    Tensor(Box<DynInt>),
}

fn optional_zero_point(
    node: &MatMulIntegerNode,
    env: &Env,
    device: &Device,
    index: usize,
) -> Result<Option<MatMulZeroPoint>> {
    if !node
        .inputs
        .get(index)
        .is_some_and(|input| !input.is_optional())
    {
        return Ok(None);
    }
    let value = resolve::at(env, &node.name, &node.inputs, index, device)?;
    match value {
        Value::Scalar(Scalar::I64(value)) => Ok(Some(MatMulZeroPoint::Scalar(value))),
        Value::Scalar(Scalar::U64(value)) => i64::try_from(value)
            .map(MatMulZeroPoint::Scalar)
            .map(Some)
            .map_err(|_| TynxError::Shape(format!("zero point {value} exceeds i64"))),
        Value::Int(tensor) if tensor.dims().iter().product::<usize>() == 1 => {
            let value = tensor
                .into_data()
                .iter::<i64>()
                .next()
                .ok_or_else(|| TynxError::Shape("empty zero-point tensor".to_string()))?;
            Ok(Some(MatMulZeroPoint::Scalar(value)))
        }
        Value::Int(tensor) => Ok(Some(MatMulZeroPoint::Tensor(Box::new(
            tensor.cast(DType::I32),
        )))),
        other => Err(TynxError::TypeMismatch(format!(
            "MatMulInteger zero point must be an integer scalar or tensor, got {other:?}"
        ))),
    }
}

fn subtract_zero_point(
    input: DynInt,
    zero_point: Option<MatMulZeroPoint>,
    matrix_axis_from_end: usize,
) -> Result<DynInt> {
    let Some(zero_point) = zero_point else {
        return Ok(input);
    };
    let mut zero_point = match zero_point {
        MatMulZeroPoint::Scalar(zero_point) => return Ok(input.sub_scalar(zero_point)),
        MatMulZeroPoint::Tensor(zero_point) => *zero_point,
    };

    let input_dims = input.dims();
    let rank = input_dims.len();
    let matrix_axis = rank.checked_sub(matrix_axis_from_end).ok_or_else(|| {
        TynxError::Shape(format!(
            "MatMulInteger input rank {rank} is too small for a matrix"
        ))
    })?;
    let zero_dims = zero_point.dims();
    if zero_dims.len() == 1 {
        if zero_dims[0] != input_dims[matrix_axis] {
            return Err(TynxError::Shape(format!(
                "MatMulInteger zero point length {} does not match matrix dimension {}",
                zero_dims[0], input_dims[matrix_axis]
            )));
        }
        let mut broadcast_dims = vec![1; rank];
        broadcast_dims[matrix_axis] = zero_dims[0];
        zero_point = zero_point.reshape(broadcast_dims)?;
    } else if zero_dims.len() == rank {
        let singleton_axis = if matrix_axis_from_end == 2 {
            rank - 1
        } else {
            rank - 2
        };
        if zero_dims[singleton_axis] != 1 {
            return Err(TynxError::Shape(format!(
                "MatMulInteger zero point dimension {singleton_axis} must be 1, got {}",
                zero_dims[singleton_axis]
            )));
        }
    } else {
        return Err(TynxError::Shape(format!(
            "MatMulInteger zero point rank {} must be 1 or match input rank {rank}",
            zero_dims.len()
        )));
    }

    input.sub_broadcast(zero_point)
}

pub fn matmul_values(left: Value, right: Value, device: &Device) -> Result<Value> {
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
        ir::{ArgType, Argument, TensorType},
        node::{
            det::DetNodeBuilder,
            linear::{LinearConfig, LinearNode},
            matmul::MatMulNodeBuilder,
            matmulinteger::MatMulIntegerNodeBuilder,
        },
    };

    use super::*;

    #[test]
    fn executes_linear_with_transposed_weight_and_bias() {
        let tensor = |name, shape: Vec<usize>| {
            Argument::new(
                name,
                ArgType::Tensor(TensorType::new_known(DType::F32, shape)),
            )
        };
        let node = LinearNode {
            name: "linear".into(),
            inputs: vec![
                tensor("input", vec![2, 2]),
                tensor("weight", vec![3, 2]),
                tensor("bias", vec![3]),
            ],
            outputs: vec![tensor("output", vec![2, 3])],
            config: LinearConfig::new(true),
        };
        let device = Device::default();
        let mut env = Env::new();
        for (name, values, shape) in [
            ("input", vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]),
            ("weight", vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]),
            ("bias", vec![0.5_f32, 1.0, -1.0], vec![3]),
        ] {
            env.insert(
                name.into(),
                Value::from_tensor_data(
                    TensorData::new(values, shape.clone()),
                    shape.len(),
                    &device,
                )
                .unwrap(),
            );
        }

        let output = linear(&node, &env, &device)
            .unwrap()
            .remove(0)
            .into_tensor()
            .unwrap();

        assert_eq!(
            output.into_data().to_vec::<f32>().unwrap(),
            vec![1.5, 3.0, 2.0, 3.5, 5.0, 6.0]
        );
    }

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
    fn broadcasts_batched_matrix_dimensions() {
        let node = MatMulNodeBuilder::new("matmul")
            .input_tensor("a", 4, DType::F32)
            .input_tensor("b", 4, DType::F32)
            .output_tensor("y", 4, DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "a".into(),
            Value::from_tensor_data(TensorData::new(vec![1.0_f32; 12], [2, 1, 2, 3]), 4, &device)
                .unwrap(),
        );
        env.insert(
            "b".into(),
            Value::from_tensor_data(TensorData::new(vec![1.0_f32; 24], [1, 4, 3, 2]), 4, &device)
                .unwrap(),
        );

        let output = matmul(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();

        assert_eq!(output.dims(), [2, 4, 2, 2]);
        assert_eq!(
            output.into_data().iter::<f32>().collect::<Vec<_>>(),
            [3.0; 32]
        );
    }

    #[test]
    fn vector_dot_product_returns_a_scalar() {
        let node = MatMulNodeBuilder::new("matmul")
            .input_tensor("a", 1, DType::F32)
            .input_tensor("b", 1, DType::F32)
            .output_scalar("y", DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "a".into(),
            Value::from_tensor_data(TensorData::new(vec![1.0_f32, 2.0, 3.0], [3]), 1, &device)
                .unwrap(),
        );
        env.insert(
            "b".into(),
            Value::from_tensor_data(TensorData::new(vec![4.0_f32, 5.0, 6.0], [3]), 1, &device)
                .unwrap(),
        );

        let output = matmul(&node, &env, &device).unwrap();

        assert!(matches!(
            output.as_slice(),
            [Value::Scalar(Scalar::F64(value))] if (*value - 32.0).abs() < 1e-6
        ));
    }

    #[test]
    fn matmul_integer_broadcasts_row_and_column_zero_points() {
        let node = MatMulIntegerNodeBuilder::new("matmul_integer")
            .input_tensor("a", 2, DType::U8)
            .input_tensor("b", 2, DType::U8)
            .input_tensor("a_zero", 1, DType::U8)
            .input_tensor("b_zero", 1, DType::U8)
            .output_tensor("y", 2, DType::I32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        for (name, values, dims, rank) in [
            ("a", vec![2_u8, 3, 4, 5], vec![2, 2], 2),
            ("b", vec![6_u8, 7, 8, 9], vec![2, 2], 2),
            ("a_zero", vec![1_u8, 2], vec![2], 1),
            ("b_zero", vec![3_u8, 4], vec![2], 1),
        ] {
            env.insert(
                name.into(),
                Value::from_tensor_data(TensorData::new(values, dims), rank, &device).unwrap(),
            );
        }

        let output = matmul_integer(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_int()
            .unwrap()
            .into_data()
            .iter::<i32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [13, 13, 21, 21]);
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

    #[test]
    fn rejects_non_square_determinants_without_panicking() {
        let node = DetNodeBuilder::new("det")
            .input_tensor("x", 2, DType::F32)
            .output_scalar("y", DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
                2,
                &device,
            )
            .unwrap(),
        );

        let error = det(&node, &env, &device).unwrap_err();

        assert_eq!(
            error,
            TynxError::Shape(
                "Det requires square matrices, got trailing dimensions 2 and 3".to_string()
            )
        );
    }
}
