//! ONNX spatial layout rearrangement operators.

use burn::tensor::Device;
use onnx_ir::node::{
    depth_to_space::{DepthToSpaceMode, DepthToSpaceNode},
    space_to_depth::SpaceToDepthNode,
};

use super::{Env, resolve, shape};
use crate::{Result, TynxError, Value};

pub(super) fn depth_to_space(
    node: &DepthToSpaceNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let [batch, channels_in, height, width] = rank_four(&input, "DepthToSpace")?;
    let block = node.config.block_size;
    let block_squared = block
        .checked_mul(block)
        .filter(|&value| value != 0)
        .ok_or_else(|| TynxError::Shape("DepthToSpace block size is invalid".to_string()))?;
    if !channels_in.is_multiple_of(block_squared) {
        return Err(TynxError::Shape(format!(
            "DepthToSpace channels {channels_in} are not divisible by block size squared {block_squared}"
        )));
    }
    let channels = channels_in / block_squared;
    let output_height = height
        .checked_mul(block)
        .ok_or_else(|| TynxError::Shape("DepthToSpace height overflow".to_string()))?;
    let output_width = width
        .checked_mul(block)
        .ok_or_else(|| TynxError::Shape("DepthToSpace width overflow".to_string()))?;

    let (expanded, axes) = match node.config.mode {
        DepthToSpaceMode::Dcr => (
            vec![batch, block, block, channels, height, width],
            vec![0, 3, 4, 1, 5, 2],
        ),
        DepthToSpaceMode::Crd => (
            vec![batch, channels, block, block, height, width],
            vec![0, 1, 4, 2, 5, 3],
        ),
    };
    let output = shape::reshape_value(input, expanded, device)?;
    let output = permute(output, axes)?;
    let output = shape::reshape_value(
        output,
        vec![batch, channels, output_height, output_width],
        device,
    )?;
    Ok(vec![output])
}

pub(super) fn space_to_depth(
    node: &SpaceToDepthNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let [batch, channels, height, width] = rank_four(&input, "SpaceToDepth")?;
    let block = node.config.block_size;
    if block == 0 || !height.is_multiple_of(block) || !width.is_multiple_of(block) {
        return Err(TynxError::Shape(format!(
            "SpaceToDepth spatial shape [{height}, {width}] is not divisible by block size {block}"
        )));
    }
    let output_height = height / block;
    let output_width = width / block;
    let output_channels = channels
        .checked_mul(block)
        .and_then(|value| value.checked_mul(block))
        .ok_or_else(|| TynxError::Shape("SpaceToDepth channel overflow".to_string()))?;

    let output = shape::reshape_value(
        input,
        vec![batch, channels, output_height, block, output_width, block],
        device,
    )?;
    let output = permute(output, vec![0, 3, 5, 1, 2, 4])?;
    let output = shape::reshape_value(
        output,
        vec![batch, output_channels, output_height, output_width],
        device,
    )?;
    Ok(vec![output])
}

fn rank_four(value: &Value, operator: &str) -> Result<[usize; 4]> {
    shape::value_dims(value)
        .try_into()
        .map_err(|dims: Vec<usize>| {
            TynxError::Shape(format!(
                "{operator} expects rank 4, got rank {}",
                dims.len()
            ))
        })
}

fn permute(value: Value, axes: Vec<usize>) -> Result<Value> {
    match value {
        Value::Tensor(tensor) => Ok(Value::Tensor(tensor.permute(axes)?)),
        Value::Int(tensor) => Ok(Value::Int(tensor.permute(axes)?)),
        Value::Bool(tensor) => Ok(Value::Bool(tensor.permute(axes)?)),
        other => Err(TynxError::TypeMismatch(format!(
            "spatial layout operator expects a tensor, got {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::depth_to_space::{DepthToSpaceConfig, DepthToSpaceMode, DepthToSpaceNodeBuilder},
    };

    use super::*;

    #[test]
    fn rearranges_depth_in_crd_order() {
        let node = DepthToSpaceNodeBuilder::new("depth_to_space")
            .input_tensor("x", 4, DType::F32)
            .output_tensor("y", 4, DType::F32)
            .config(DepthToSpaceConfig {
                mode: DepthToSpaceMode::Crd,
                block_size: 2,
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(
                    (0..8).map(|value| value as f32).collect::<Vec<_>>(),
                    [1, 8, 1, 1],
                ),
                4,
                &device,
            )
            .unwrap(),
        );

        let output = depth_to_space(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();

        assert_eq!(output.dims(), [1, 2, 2, 2]);
        assert_eq!(
            output.into_data().iter::<f32>().collect::<Vec<_>>(),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
    }
}
