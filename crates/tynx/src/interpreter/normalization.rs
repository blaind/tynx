//! Batch and instance normalization execution.

use burn::tensor::{DType, Device};
use onnx_ir::{
    ir::Argument,
    node::{
        batch_norm::{BatchNormConfig, BatchNormalizationNode},
        instance_norm::InstanceNormalizationNode,
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
