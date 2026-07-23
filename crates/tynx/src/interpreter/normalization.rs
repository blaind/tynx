//! ONNX normalization operator execution.

use burn::tensor::{DType, Device, Slice};
use onnx_ir::{
    ir::Argument,
    node::{
        batch_norm::{BatchNormConfig, BatchNormalizationNode},
        group_norm::GroupNormalizationNode,
        instance_norm::InstanceNormalizationNode,
        layer_norm::LayerNormalizationNode,
        lp_normalization::LpNormalizationNode,
        lrn::LrnNode,
        mean_variance_normalization::MeanVarianceNormalizationNode,
    },
};

use super::{Env, resolve};
use crate::{DynTensor, Result, TynxError, Value};

pub(super) fn batch_normalization(
    node: &BatchNormalizationNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    if node.outputs.len() != 1 {
        return Err(TynxError::UnsupportedOp(
            "BatchNormalization training outputs".to_string(),
        ));
    }

    let input = resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?;
    let rank = input.rank();
    let channels = channels(&input, "BatchNormalization", 2)?;
    let dtype = input.dtype();
    let parameters = ChannelParameters {
        node_name: &node.name,
        inputs: &node.inputs,
        channels,
        input_rank: rank,
        dtype,
        env,
        device,
    };
    let scale = parameters.get(1)?;
    let bias = parameters.get(2)?;
    let mean = parameters.get(3)?;
    let variance = parameters.get(4)?;
    let epsilon = match &node.config {
        BatchNormConfig::Static(config) => config.epsilon,
        BatchNormConfig::Runtime(config) => config.epsilon,
    };

    let output = input
        .sub_broadcast(mean)?
        .div_broadcast(variance.add_scalar(epsilon).sqrt())?
        .mul_broadcast(scale)?
        .add_broadcast(bias)?;
    Ok(vec![Value::Tensor(output)])
}

pub(super) fn instance_normalization(
    node: &InstanceNormalizationNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?;
    let rank = input.rank();
    let channels = channels(&input, "InstanceNormalization", 3)?;
    let dtype = input.dtype();
    let parameters = ChannelParameters {
        node_name: &node.name,
        inputs: &node.inputs,
        channels,
        input_rank: rank,
        dtype,
        env,
        device,
    };
    let scale = parameters.get(1)?;
    let bias = parameters.get(2)?;
    let spatial_axes = (2..rank).collect::<Vec<_>>();
    let mean = input.clone().mean_dims(&spatial_axes);
    let centered = input.sub_broadcast(mean)?;
    let variance = centered.clone().powi_scalar(2).mean_dims(&spatial_axes);

    let output = centered
        .div_broadcast(variance.add_scalar(node.config.epsilon).sqrt())?
        .mul_broadcast(scale)?
        .add_broadcast(bias)?;
    Ok(vec![Value::Tensor(output)])
}

pub(super) fn layer_normalization(
    node: &LayerNormalizationNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?;
    let scale = resolve::at(env, &node.name, &node.inputs, 1, device)?.into_tensor()?;
    let bias = node
        .inputs
        .get(2)
        .filter(|input| !input.is_optional())
        .map(|bias| resolve::input_at(env, bias, 2, device)?.into_tensor())
        .transpose()?;
    layer_normalization_values(
        input,
        scale,
        bias,
        node.config.epsilon,
        node.config.full_precision,
        node.outputs.len(),
    )
}

/// Execute ONNX LayerNormalization from already-resolved tensor values.
///
/// This is shared with imported training execution so scale and bias can come
/// from live parameter slots without duplicating ONNX normalization semantics.
pub fn layer_normalization_values(
    input: DynTensor,
    scale: DynTensor,
    bias: Option<DynTensor>,
    epsilon: f64,
    full_precision: bool,
    output_count: usize,
) -> Result<Vec<Value>> {
    if !(1..=3).contains(&output_count) {
        return Err(TynxError::Shape(format!(
            "LayerNormalization expects between 1 and 3 outputs, got {output_count}"
        )));
    }
    let output_dtype = input.dtype();
    let normalized_rank = scale.rank();
    if normalized_rank > input.rank() {
        return Err(TynxError::Shape(format!(
            "LayerNormalization scale rank {normalized_rank} exceeds input rank {}",
            input.rank()
        )));
    }
    let expected = &input.dims()[input.rank() - normalized_rank..];
    if scale.dims() != expected {
        return Err(TynxError::Shape(format!(
            "LayerNormalization scale shape {:?} does not match normalized shape {expected:?}",
            scale.dims()
        )));
    }
    let compute_dtype = if full_precision && matches!(output_dtype, DType::F16 | DType::BF16) {
        DType::F32
    } else {
        output_dtype
    };
    let input = input.cast(compute_dtype);
    let input_rank = input.rank();
    let axes = (input_rank - normalized_rank..input_rank).collect::<Vec<_>>();
    let mean = input.clone().mean_dims(&axes);
    let centered = input.sub_broadcast(mean.clone())?;
    let variance = centered.clone().powi_scalar(2).mean_dims(&axes);
    let inv_std = variance.add_scalar(epsilon).powf_scalar(-0.5);
    let scale = scale.cast(compute_dtype).to_rank(input_rank)?;
    let mut output = centered
        .mul_broadcast(inv_std.clone())?
        .mul_broadcast(scale)?;
    if let Some(bias) = bias {
        let bias = bias.cast(compute_dtype).to_rank(input_rank)?;
        output = output.add_broadcast(bias)?;
    }

    let mut outputs = vec![Value::Tensor(output.cast(output_dtype))];
    if output_count > 1 {
        outputs.push(Value::Tensor(mean));
    }
    if output_count > 2 {
        outputs.push(Value::Tensor(inv_std));
    }
    Ok(outputs)
}

pub(super) fn group_normalization(
    node: &GroupNormalizationNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?;
    let dims = input.dims();
    let channels = channels(&input, "GroupNormalization", 3)?;
    if node.config.num_groups == 0 || !channels.is_multiple_of(node.config.num_groups) {
        return Err(TynxError::Shape(format!(
            "GroupNormalization channels {channels} are not divisible by {} groups",
            node.config.num_groups
        )));
    }
    if input.rank() >= crate::MAX_RANK {
        return Err(TynxError::UnsupportedOp(format!(
            "GroupNormalization rank {} requires an internal rank {} tensor",
            input.rank(),
            input.rank() + 1
        )));
    }
    let output_dtype = input.dtype();
    let compute_dtype =
        if node.config.full_precision && matches!(output_dtype, DType::F16 | DType::BF16) {
            DType::F32
        } else {
            output_dtype
        };
    let mut grouped_dims = Vec::with_capacity(dims.len() + 1);
    grouped_dims.push(dims[0]);
    grouped_dims.push(node.config.num_groups);
    grouped_dims.push(channels / node.config.num_groups);
    grouped_dims.extend_from_slice(&dims[2..]);
    let grouped = input.cast(compute_dtype).reshape(grouped_dims)?;
    let axes = (2..grouped.rank()).collect::<Vec<_>>();
    let mean = grouped.clone().mean_dims(&axes);
    let centered = grouped.sub_broadcast(mean)?;
    let variance = centered.clone().powi_scalar(2).mean_dims(&axes);
    let normalized = centered
        .div_broadcast(variance.add_scalar(node.config.epsilon).sqrt())?
        .reshape(dims.clone())?;
    let parameters = ChannelParameters {
        node_name: &node.name,
        inputs: &node.inputs,
        channels,
        input_rank: dims.len(),
        dtype: compute_dtype,
        env,
        device,
    };
    let output = normalized
        .mul_broadcast(parameters.get(1)?)?
        .add_broadcast(parameters.get(2)?)?
        .cast(output_dtype);
    Ok(vec![Value::Tensor(output)])
}

pub(super) fn lrn(node: &LrnNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    let channel_count = channels(&input, "LRN", 3)?;
    let size = usize::try_from(node.config.size)
        .map_err(|_| TynxError::Shape(format!("LRN size {} is invalid", node.config.size)))?;
    let squares = input.clone().powi_scalar(2);
    let left = (size - 1) / 2;
    let right = size / 2;
    let mut channel_sums = Vec::with_capacity(channel_count);
    for channel in 0..channel_count {
        let start = channel.saturating_sub(left);
        let end = (channel + right + 1).min(channel_count);
        let mut slices = vec![Slice::full(); input.rank()];
        slices[1] = Slice::new(start as isize, Some(end as isize), 1);
        channel_sums.push(squares.clone().slice(&slices).sum_dims(&[1]));
    }
    let local_sum = DynTensor::concat(channel_sums, 1)?;
    let scale = local_sum
        .mul_scalar(node.config.alpha as f64 / size as f64)
        .add_scalar(node.config.bias as f64)
        .powf_scalar(node.config.beta as f64);
    Ok(vec![Value::Tensor(input.div_broadcast(scale)?)])
}

pub(super) fn lp_normalization(
    node: &LpNormalizationNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    let norm = match node.config.p {
        1 => input.clone().abs().sum_dims(&[node.config.axis]),
        2 => input
            .clone()
            .powi_scalar(2)
            .sum_dims(&[node.config.axis])
            .sqrt(),
        p => {
            return Err(TynxError::UnsupportedOp(format!("LpNormalization p={p}")));
        }
    };
    Ok(vec![Value::Tensor(input.div_broadcast(norm)?)])
}

pub(super) fn mean_variance_normalization(
    node: &MeanVarianceNormalizationNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    let mean = input.clone().mean_dims(&node.config.axes);
    let centered = input.sub_broadcast(mean)?;
    let variance = centered.clone().powi_scalar(2).mean_dims(&node.config.axes);
    Ok(vec![Value::Tensor(
        centered.div_broadcast(variance.sqrt())?,
    )])
}

struct ChannelParameters<'a> {
    node_name: &'a str,
    inputs: &'a [Argument],
    channels: usize,
    input_rank: usize,
    dtype: DType,
    env: &'a Env,
    device: &'a Device,
}

impl ChannelParameters<'_> {
    fn get(&self, index: usize) -> Result<DynTensor> {
        let parameter = resolve::at(self.env, self.node_name, self.inputs, index, self.device)?
            .into_tensor()?;
        if parameter.dims() != [self.channels] {
            return Err(TynxError::Shape(format!(
                "normalization parameter {} has shape {:?}; expected [{}]",
                self.inputs[index].name,
                parameter.dims(),
                self.channels
            )));
        }
        let mut dims = vec![1; self.input_rank];
        dims[1] = self.channels;
        parameter.cast(self.dtype).reshape(dims)
    }
}

fn channels(input: &DynTensor, operator: &str, minimum_rank: usize) -> Result<usize> {
    if input.rank() < minimum_rank {
        return Err(TynxError::Shape(format!(
            "{operator} requires rank >= {minimum_rank}, got {}",
            input.rank()
        )));
    }
    Ok(input.dims()[1])
}

#[cfg(test)]
mod tests {
    use burn::tensor::{TensorData, Tolerance};
    use onnx_ir::{
        DType,
        node::{
            batch_norm::{BatchNormConfig, BatchNormRuntimeConfig, BatchNormalizationNodeBuilder},
            instance_norm::{InstanceNormConfig, InstanceNormalizationNodeBuilder},
            layer_norm::{LayerNormConfig, LayerNormalizationNodeBuilder},
            lp_normalization::{LpNormalizationConfig, LpNormalizationNodeBuilder},
            mean_variance_normalization::{
                MeanVarianceNormalizationConfig, MeanVarianceNormalizationNodeBuilder,
            },
        },
    };

    use super::*;

    #[test]
    fn batch_norm_uses_running_statistics_and_affine_parameters() {
        let node = BatchNormalizationNodeBuilder::new("batch_norm")
            .input_tensor("x", 3, DType::F32)
            .input_tensor("scale", 1, DType::F32)
            .input_tensor("bias", 1, DType::F32)
            .input_tensor("mean", 1, DType::F32)
            .input_tensor("variance", 1, DType::F32)
            .output_tensor("y", 3, DType::F32)
            .config(BatchNormConfig::Runtime(BatchNormRuntimeConfig::new(
                0.0, 0.9,
            )))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        insert(&mut env, "x", vec![1.0, 3.0, 2.0, 6.0], [1, 2, 2], &device);
        insert_1d(&mut env, "scale", vec![2.0, 0.5], &device);
        insert_1d(&mut env, "bias", vec![1.0, -1.0], &device);
        insert_1d(&mut env, "mean", vec![1.0, 2.0], &device);
        insert_1d(&mut env, "variance", vec![4.0, 1.0], &device);

        let output = batch_normalization(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data();

        output.assert_eq(
            &TensorData::new(vec![1.0_f32, 3.0, -1.0, 1.0], [1, 2, 2]),
            false,
        );
    }

    #[test]
    fn lp_normalization_supports_l1() {
        let node = LpNormalizationNodeBuilder::new("lp_normalization")
            .input_tensor("x", 2, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .config(LpNormalizationConfig { axis: 1, p: 1 })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(vec![3.0_f32, 4.0, 1.0, -1.0], [2, 2]),
                2,
                &device,
            )
            .unwrap(),
        );

        let output = lp_normalization(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data();

        output.assert_approx_eq::<f32>(
            &TensorData::new(vec![3.0_f32 / 7.0, 4.0 / 7.0, 0.5, -0.5], [2, 2]),
            Tolerance::default(),
        );
    }

    #[test]
    fn mean_variance_normalization_uses_selected_axes() {
        let node = MeanVarianceNormalizationNodeBuilder::new("mvn")
            .input_tensor("x", 2, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .config(MeanVarianceNormalizationConfig { axes: vec![1] })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(TensorData::new(vec![1.0_f32, 2.0, 3.0], [1, 3]), 2, &device)
                .unwrap(),
        );

        let output = mean_variance_normalization(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data();

        let scale = 1.5_f32.sqrt();
        output.assert_approx_eq::<f32>(
            &TensorData::new(vec![-scale, 0.0, scale], [1, 3]),
            Tolerance::default(),
        );
    }

    #[test]
    fn instance_norm_uses_custom_epsilon() {
        let epsilon = 1e-3;
        let node = InstanceNormalizationNodeBuilder::new("instance_norm")
            .input_tensor("x", 3, DType::F32)
            .input_tensor("scale", 1, DType::F32)
            .input_tensor("bias", 1, DType::F32)
            .output_tensor("y", 3, DType::F32)
            .config(InstanceNormConfig::new(epsilon))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        insert(&mut env, "x", vec![1.0, 3.0, 2.0, 6.0], [1, 2, 2], &device);
        insert_1d(&mut env, "scale", vec![2.0, 0.5], &device);
        insert_1d(&mut env, "bias", vec![1.0, -1.0], &device);
        let first = 1.0 / f32::sqrt(1.0 + epsilon as f32);
        let second = 2.0 / f32::sqrt(4.0 + epsilon as f32);

        let output = instance_normalization(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data();

        output.assert_approx_eq(
            &TensorData::new(
                vec![
                    1.0 - 2.0 * first,
                    1.0 + 2.0 * first,
                    -1.0 - 0.5 * second,
                    -1.0 + 0.5 * second,
                ],
                [1, 2, 2],
            ),
            Tolerance::<f32>::absolute(1e-5),
        );
    }

    #[test]
    fn rejects_wrong_channel_parameter_length() {
        let device = Device::default();
        let mut env = Env::new();
        insert_1d(&mut env, "scale", vec![1.0], &device);
        let inputs = vec![
            Argument::new(
                "x",
                onnx_ir::ArgType::Tensor(onnx_ir::TensorType::new(DType::F32, 3, None)),
            ),
            Argument::new(
                "scale",
                onnx_ir::ArgType::Tensor(onnx_ir::TensorType::new(DType::F32, 1, None)),
            ),
        ];

        let parameters = ChannelParameters {
            node_name: "norm",
            inputs: &inputs,
            channels: 2,
            input_rank: 3,
            dtype: DType::F32,
            env: &env,
            device: &device,
        };
        assert!(parameters.get(1).is_err());
    }

    #[test]
    fn layer_norm_normalizes_the_scale_shape_axes() {
        let node = LayerNormalizationNodeBuilder::new("layer_norm")
            .input_tensor("x", 2, DType::F32)
            .input_tensor("scale", 1, DType::F32)
            .input_tensor("bias", 1, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .config(LayerNormConfig::new(0.0, false))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        insert(&mut env, "x", vec![1.0, 2.0], [1, 2], &device);
        insert_1d(&mut env, "scale", vec![1.0, 1.0], &device);
        insert_1d(&mut env, "bias", vec![0.0, 0.0], &device);

        let output = layer_normalization(&node, &env, &device)
            .unwrap()
            .remove(0)
            .into_tensor()
            .unwrap()
            .into_data();

        output.assert_approx_eq(
            &TensorData::new(vec![-1.0_f32, 1.0], [1, 2]),
            Tolerance::<f32>::absolute(1e-6),
        );
    }

    fn insert<const D: usize>(
        env: &mut Env,
        name: &str,
        values: Vec<f32>,
        dims: [usize; D],
        device: &Device,
    ) {
        env.insert(
            name.to_string(),
            Value::from_tensor_data(TensorData::new(values, dims), D, device).unwrap(),
        );
    }

    fn insert_1d(env: &mut Env, name: &str, values: Vec<f32>, device: &Device) {
        let len = values.len();
        insert(env, name, values, [len], device);
    }
}
