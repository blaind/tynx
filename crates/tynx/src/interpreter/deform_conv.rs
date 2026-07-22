//! ONNX DeformConv execution.

use burn::tensor::{Device, Tensor, module::deform_conv2d, ops::DeformConvOptions};
use onnx_ir::{ir::Argument, node::deform_conv::DeformConvNode};

use super::{Env, resolve};
use crate::{DynTensor, Result, TynxError, Value};

pub(super) fn deform_conv(node: &DeformConvNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    validate_nonzero(node)?;
    let input = rank4(resolve::at(env, &node.name, &node.inputs, 0, device)?, 0)?;
    let dtype = input.dtype();
    let weight = rank4(resolve::at(env, &node.name, &node.inputs, 1, device)?, 1)?.cast(dtype);
    let offset = rank4(resolve::at(env, &node.name, &node.inputs, 2, device)?, 2)?.cast(dtype);
    let bias = optional_rank1(env, &node.inputs, 3, device)?.map(|value| value.cast(dtype));
    let mask = optional_rank4(env, &node.inputs, 4, device)?.map(|value| value.cast(dtype));
    let [batch, input_channels, input_height, input_width] = input.dims();
    let [
        output_channels,
        weight_channels,
        kernel_height,
        kernel_width,
    ] = weight.dims();
    if [kernel_height, kernel_width] != node.config.kernel_size {
        return Err(TynxError::Shape(format!(
            "DeformConv weight kernel [{kernel_height}, {kernel_width}] differs from configured {:?}",
            node.config.kernel_size
        )));
    }
    if input_channels % node.config.groups != 0
        || output_channels % node.config.groups != 0
        || weight_channels != input_channels / node.config.groups
    {
        return Err(TynxError::Shape(format!(
            "DeformConv channels {input_channels}/{output_channels}/{weight_channels} are incompatible with {} weight groups",
            node.config.groups
        )));
    }
    let (top, left, bottom, right) = node.config.padding.as_tuple();
    if top != bottom || left != right {
        return Err(TynxError::UnsupportedOp(
            "DeformConv asymmetric padding".to_string(),
        ));
    }
    let output_height = output_size(
        input_height,
        kernel_height,
        node.config.stride[0],
        node.config.dilation[0],
        top,
        bottom,
    )?;
    let output_width = output_size(
        input_width,
        kernel_width,
        node.config.stride[1],
        node.config.dilation[1],
        left,
        right,
    )?;
    let expected_offset_channels = 2 * node.config.offset_groups * kernel_height * kernel_width;
    if offset.dims() != [batch, expected_offset_channels, output_height, output_width] {
        return Err(TynxError::Shape(format!(
            "DeformConv offset has shape {:?}, expected [{batch}, {expected_offset_channels}, {output_height}, {output_width}]",
            offset.dims()
        )));
    }
    if input_channels % node.config.offset_groups != 0 {
        return Err(TynxError::Shape(format!(
            "DeformConv input channels {input_channels} are not divisible by {} offset groups",
            node.config.offset_groups
        )));
    }
    if let Some(mask) = &mask {
        let expected_mask_channels = node.config.offset_groups * kernel_height * kernel_width;
        if mask.dims() != [batch, expected_mask_channels, output_height, output_width] {
            return Err(TynxError::Shape(format!(
                "DeformConv mask has shape {:?}, expected [{batch}, {expected_mask_channels}, {output_height}, {output_width}]",
                mask.dims()
            )));
        }
    }
    if let Some(bias) = &bias
        && bias.dims() != [output_channels]
    {
        return Err(TynxError::Shape(format!(
            "DeformConv bias has shape {:?}, expected [{output_channels}]",
            bias.dims()
        )));
    }
    let options = DeformConvOptions::new(
        node.config.stride,
        [top, left],
        node.config.dilation,
        node.config.groups,
        node.config.offset_groups,
    );
    let output = deform_conv2d(input, offset, weight, mask, bias, options);
    Ok(vec![Value::Tensor(DynTensor::R4(output))])
}

fn rank4(value: Value, index: usize) -> Result<Tensor<4>> {
    match value.into_tensor()? {
        DynTensor::R4(value) => Ok(value),
        value => Err(TynxError::Shape(format!(
            "DeformConv input {index} must have rank 4, got rank {}",
            value.rank()
        ))),
    }
}

fn optional_rank4(
    env: &Env,
    inputs: &[Argument],
    index: usize,
    device: &Device,
) -> Result<Option<Tensor<4>>> {
    let Some(input) = inputs.get(index).filter(|input| !input.is_optional()) else {
        return Ok(None);
    };
    rank4(resolve::input(env, input, device)?, index).map(Some)
}

fn optional_rank1(
    env: &Env,
    inputs: &[Argument],
    index: usize,
    device: &Device,
) -> Result<Option<Tensor<1>>> {
    let Some(input) = inputs.get(index).filter(|input| !input.is_optional()) else {
        return Ok(None);
    };
    match resolve::input(env, input, device)?.into_tensor()? {
        DynTensor::R1(value) => Ok(Some(value)),
        value => Err(TynxError::Shape(format!(
            "DeformConv input {index} must have rank 1, got rank {}",
            value.rank()
        ))),
    }
}

fn validate_nonzero(node: &DeformConvNode) -> Result<()> {
    if node.config.stride.contains(&0)
        || node.config.dilation.contains(&0)
        || node.config.groups == 0
        || node.config.offset_groups == 0
    {
        return Err(TynxError::Shape(
            "DeformConv strides, dilations, and group counts must be non-zero".to_string(),
        ));
    }
    Ok(())
}

fn output_size(
    input: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    before: usize,
    after: usize,
) -> Result<usize> {
    let effective_kernel = kernel
        .checked_sub(1)
        .and_then(|value| value.checked_mul(dilation))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| TynxError::Shape("DeformConv kernel extent overflow".to_string()))?;
    let padded = input
        .checked_add(before)
        .and_then(|value| value.checked_add(after))
        .ok_or_else(|| TynxError::Shape("DeformConv padded size overflow".to_string()))?;
    if padded < effective_kernel {
        return Err(TynxError::Shape(format!(
            "DeformConv effective kernel {effective_kernel} exceeds padded input {padded}"
        )));
    }
    Ok((padded - effective_kernel) / stride + 1)
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::{
            deform_conv::{DeformConvConfig, DeformConvNodeBuilder},
            padding::PaddingConfig2d,
        },
    };

    use super::*;

    #[test]
    fn applies_a_one_by_one_deformable_convolution() {
        let node = DeformConvNodeBuilder::new("deform_conv")
            .input_tensor("x", 4, DType::F32)
            .input_tensor("w", 4, DType::F32)
            .input_tensor("offset", 4, DType::F32)
            .output_tensor("y", 4, DType::F32)
            .config(DeformConvConfig {
                kernel_size: [1, 1],
                stride: [1, 1],
                padding: PaddingConfig2d::Valid,
                dilation: [1, 1],
                groups: 1,
                offset_groups: 1,
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
                4,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "w".into(),
            Value::from_tensor_data(TensorData::new(vec![2.0_f32], [1, 1, 1, 1]), 4, &device)
                .unwrap(),
        );
        env.insert(
            "offset".into(),
            Value::from_tensor_data(TensorData::new(vec![0.0_f32; 8], [1, 2, 2, 2]), 4, &device)
                .unwrap(),
        );

        let output = deform_conv(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [2.0, 4.0, 6.0, 8.0]);
    }
}
