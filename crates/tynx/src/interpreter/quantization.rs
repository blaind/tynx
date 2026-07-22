//! Linear quantization operator execution.

use burn::tensor::{DType, Device};
use onnx_ir::node::{
    dequantize_linear::DequantizeLinearNode, qlinear_matmul::QLinearMatMulNode,
    quantize_linear::QuantizeLinearNode,
};

use super::{Env, matrix, resolve};
use crate::{DynInt, DynTensor, Result, TynxError, Value};

pub(super) fn dequantize_linear(
    node: &DequantizeLinearNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::at(env, &node.name, &node.inputs, 0, device)?.into_int()?;
    let scale = resolve::at(env, &node.name, &node.inputs, 1, device)?;
    let zero = optional_input(&node.name, &node.inputs, 2, env, device)?;
    let dtype = node.outputs[0].ty.elem_type();

    Ok(vec![Value::Tensor(dequantize(
        input,
        scale,
        zero,
        node.config.axis,
        dtype,
    )?)])
}

pub(super) fn quantize_linear(
    node: &QuantizeLinearNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?;
    let scale = resolve::at(env, &node.name, &node.inputs, 1, device)?;
    let zero = optional_input(&node.name, &node.inputs, 2, env, device)?;
    let dtype = node.outputs[0].ty.elem_type();

    Ok(vec![Value::Int(quantize(
        input,
        scale,
        zero,
        node.config.axis,
        dtype,
        true,
    )?)])
}

pub(super) fn qlinear_matmul(
    node: &QLinearMatMulNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let a = dequantize(
        resolve::at(env, &node.name, &node.inputs, 0, device)?.into_int()?,
        resolve::at(env, &node.name, &node.inputs, 1, device)?,
        Some(resolve::at(env, &node.name, &node.inputs, 2, device)?),
        None,
        DType::F32,
    )?;
    let b = dequantize(
        resolve::at(env, &node.name, &node.inputs, 3, device)?.into_int()?,
        resolve::at(env, &node.name, &node.inputs, 4, device)?,
        Some(resolve::at(env, &node.name, &node.inputs, 5, device)?),
        None,
        DType::F32,
    )?;
    let product =
        matrix::matmul_values(Value::Tensor(a), Value::Tensor(b), device)?.into_tensor()?;
    let output = quantize(
        product,
        resolve::at(env, &node.name, &node.inputs, 6, device)?,
        Some(resolve::at(env, &node.name, &node.inputs, 7, device)?),
        None,
        node.outputs[0].ty.elem_type(),
        false,
    )?;

    Ok(vec![Value::Int(output)])
}

fn dequantize(
    input: DynInt,
    scale: Value,
    zero: Option<Value>,
    axis: Option<i64>,
    dtype: DType,
) -> Result<DynTensor> {
    let rank = input.rank();
    let mut output = input.to_float(dtype);
    if let Some(zero) = zero {
        output = match zero {
            Value::Scalar(zero) => output.sub_scalar(zero.as_f64()),
            Value::Int(zero) => {
                let zero = reshape_parameter(zero.to_float(dtype), rank, axis)?;
                output.sub_broadcast(zero)?
            }
            other => return Err(quantization_type_error("zero point", &other)),
        };
    }

    match scale {
        Value::Scalar(scale) => Ok(output.mul_scalar(scale.as_f64())),
        Value::Tensor(scale) => {
            let scale = reshape_parameter(scale.cast(dtype), rank, axis)?;
            output.mul_broadcast(scale)
        }
        other => Err(quantization_type_error("scale", &other)),
    }
}

fn quantize(
    input: DynTensor,
    scale: Value,
    zero: Option<Value>,
    axis: Option<i64>,
    dtype: DType,
    saturate: bool,
) -> Result<DynInt> {
    let rank = input.rank();
    let calculation_dtype = input.dtype();
    let mut output = match scale {
        Value::Scalar(scale) => input.div_scalar(scale.as_f64()),
        Value::Tensor(scale) => {
            let scale = reshape_parameter(scale.cast(calculation_dtype), rank, axis)?;
            input.div_broadcast(scale)?
        }
        other => return Err(quantization_type_error("scale", &other)),
    }
    .round();

    if let Some(zero) = zero {
        output = match zero {
            Value::Scalar(zero) => output.add_scalar(zero.as_f64()),
            Value::Int(zero) => {
                let zero = reshape_parameter(zero.to_float(calculation_dtype), rank, axis)?;
                output.add_broadcast(zero)?
            }
            other => return Err(quantization_type_error("zero point", &other)),
        };
    }

    if saturate {
        let (min, max) = quantized_range(dtype)?;
        Ok(output.clip(Some(min), Some(max)).to_int(dtype))
    } else {
        Ok(output.to_int(dtype))
    }
}

fn reshape_parameter(
    parameter: DynTensor,
    input_rank: usize,
    axis: Option<i64>,
) -> Result<DynTensor> {
    if parameter.rank() != 1 || input_rank == 1 {
        return Ok(parameter);
    }
    let axis = normalize_axis(axis.unwrap_or(1), input_rank)?;
    let mut dims = vec![1; input_rank];
    dims[axis] = parameter.dims()[0];
    parameter.reshape(dims)
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize> {
    let rank_i64 = rank as i64;
    let axis = if axis < 0 { axis + rank_i64 } else { axis };
    if !(0..rank_i64).contains(&axis) {
        return Err(TynxError::Shape(format!(
            "quantization axis {axis} is out of range for rank {rank}"
        )));
    }
    Ok(axis as usize)
}

fn quantized_range(dtype: DType) -> Result<(f64, f64)> {
    match dtype {
        DType::U8 => Ok((u8::MIN as f64, u8::MAX as f64)),
        DType::I8 => Ok((i8::MIN as f64, i8::MAX as f64)),
        DType::U16 => Ok((u16::MIN as f64, u16::MAX as f64)),
        DType::I16 => Ok((i16::MIN as f64, i16::MAX as f64)),
        _ => Err(TynxError::TypeMismatch(format!(
            "unsupported quantized output dtype {dtype:?}"
        ))),
    }
}

fn optional_input(
    node_name: &str,
    inputs: &[onnx_ir::Argument],
    index: usize,
    env: &Env,
    device: &Device,
) -> Result<Option<Value>> {
    if !inputs.get(index).is_some_and(|input| !input.is_optional()) {
        return Ok(None);
    }
    resolve::at(env, node_name, inputs, index, device).map(Some)
}

fn quantization_type_error(name: &str, value: &Value) -> TynxError {
    TynxError::TypeMismatch(format!(
        "quantization {name} has unsupported value {value:?}"
    ))
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;
    use crate::Scalar;

    #[test]
    fn quantizes_with_ties_to_even_and_saturation() {
        let device = Device::default();
        let input = DynTensor::from_data(
            TensorData::new(vec![-1000.0_f32, 0.5, 1.5, 1000.0], [4]),
            1,
            &device,
        )
        .unwrap();

        let output = quantize(
            input,
            Value::Scalar(Scalar::F64(1.0)),
            None,
            None,
            DType::U8,
            true,
        )
        .unwrap();

        assert_eq!(
            output.into_data().iter::<u64>().collect::<Vec<_>>(),
            [0, 0, 2, 255]
        );
    }

    #[test]
    fn rounds_before_adding_the_zero_point() {
        let device = Device::default();
        let input = DynTensor::from_data(TensorData::new(vec![0.5_f32], [1]), 1, &device).unwrap();

        let output = quantize(
            input,
            Value::Scalar(Scalar::F64((5.0_f32 / 255.0_f32) as f64)),
            Some(Value::Scalar(Scalar::U64(153))),
            None,
            DType::U8,
            true,
        )
        .unwrap();

        assert_eq!(output.into_data().iter::<u64>().collect::<Vec<_>>(), [179]);
    }
}
