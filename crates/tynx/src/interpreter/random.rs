//! Random tensor and Bernoulli operator execution.

use std::collections::VecDeque;

use burn::tensor::{Device, Distribution};
use onnx_ir::{
    ModelProto, Node,
    ir::OnnxGraph,
    node::{
        bernoulli::BernoulliNode,
        random::{RandomNormalNode, RandomUniformNode},
        random_like::{RandomNormalLikeNode, RandomUniformLikeNode},
    },
};
use protobuf::Message;

use super::{Env, resolve, shape};
use crate::{DynTensor, Result, TynxError, Value};

const SEED_PREFIX: &str = "__tynx_random_seed_";

pub(super) fn preserve_attributes(data: &[u8], graph: &mut OnnxGraph) -> Result<()> {
    let model =
        ModelProto::parse_from_bytes(data).map_err(|error| TynxError::Parse(error.to_string()))?;
    let Some(raw_graph) = model.graph.as_ref() else {
        return Ok(());
    };
    let mut bernoulli = seeds(raw_graph, "Bernoulli");
    let mut normal = seeds(raw_graph, "RandomNormal");
    let mut normal_like = seeds(raw_graph, "RandomNormalLike");
    let mut uniform = seeds(raw_graph, "RandomUniform");
    let mut uniform_like = seeds(raw_graph, "RandomUniformLike");

    for node in &mut graph.nodes {
        let (name, seed) = match node {
            Node::Bernoulli(node) => (&mut node.name, bernoulli.pop_front()),
            Node::RandomNormal(node) => (&mut node.name, normal.pop_front()),
            Node::RandomNormalLike(node) => (&mut node.name, normal_like.pop_front()),
            Node::RandomUniform(node) => (&mut node.name, uniform.pop_front()),
            Node::RandomUniformLike(node) => (&mut node.name, uniform_like.pop_front()),
            _ => continue,
        };
        if let Some(Some(seed)) = seed {
            *name = format!("{SEED_PREFIX}{seed}:{name}");
        }
    }
    Ok(())
}

fn seeds(graph: &onnx_ir::GraphProto, operator: &str) -> VecDeque<Option<u64>> {
    graph
        .node
        .iter()
        .filter(|node| node.op_type == operator)
        .map(|node| {
            node.attribute
                .iter()
                .find(|attribute| attribute.name == "seed")
                .map(|attribute| u64::from(attribute.f.to_bits()))
        })
        .collect()
}

pub(super) fn bernoulli(node: &BernoulliNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    seed_device(&node.name, device);
    let probabilities = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    let random = DynTensor::random(
        &probabilities.dims(),
        Distribution::Default,
        device,
        probabilities.dtype(),
    )?;
    let samples = random.less_broadcast(probabilities)?;
    let dtype = node.outputs[0].ty.elem_type();
    let output = if dtype.is_float() {
        Value::Tensor(samples.to_float(dtype))
    } else if dtype.is_int() || dtype.is_uint() {
        Value::Int(samples.to_int(dtype))
    } else if dtype.is_bool() {
        Value::Bool(samples)
    } else {
        return Err(TynxError::TypeMismatch(format!(
            "Bernoulli output dtype {dtype:?} is unsupported"
        )));
    };
    Ok(vec![output])
}

pub(super) fn random_normal(node: &RandomNormalNode, device: &Device) -> Result<Vec<Value>> {
    seed_device(&node.name, device);
    validate_normal(node.config.mean, node.config.scale, "RandomNormal")?;
    let output = DynTensor::random(
        &node.config.shape,
        Distribution::Normal(node.config.mean, node.config.scale),
        device,
        node.outputs[0].ty.elem_type(),
    )?;
    Ok(vec![Value::Tensor(output)])
}

pub(super) fn random_normal_like(
    node: &RandomNormalLikeNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    seed_device(&node.name, device);
    validate_normal(node.config.mean, node.config.scale, "RandomNormalLike")?;
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let output = DynTensor::random(
        &shape::value_dims(&input),
        Distribution::Normal(node.config.mean, node.config.scale),
        device,
        node.outputs[0].ty.elem_type(),
    )?;
    Ok(vec![Value::Tensor(output)])
}

pub(super) fn random_uniform(node: &RandomUniformNode, device: &Device) -> Result<Vec<Value>> {
    seed_device(&node.name, device);
    validate_uniform(node.config.low, node.config.high, "RandomUniform")?;
    let output = DynTensor::random(
        &node.config.shape,
        Distribution::Uniform(node.config.low, node.config.high),
        device,
        node.outputs[0].ty.elem_type(),
    )?;
    Ok(vec![Value::Tensor(output)])
}

pub(super) fn random_uniform_like(
    node: &RandomUniformLikeNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    seed_device(&node.name, device);
    validate_uniform(node.config.low, node.config.high, "RandomUniformLike")?;
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let output = DynTensor::random(
        &shape::value_dims(&input),
        Distribution::Uniform(node.config.low, node.config.high),
        device,
        node.outputs[0].ty.elem_type(),
    )?;
    Ok(vec![Value::Tensor(output)])
}

fn validate_normal(mean: f64, scale: f64, operator: &str) -> Result<()> {
    if !mean.is_finite() || !scale.is_finite() || scale <= 0.0 {
        return Err(TynxError::Shape(format!(
            "{operator} requires a finite mean and positive finite scale, got {mean} and {scale}"
        )));
    }
    Ok(())
}

fn validate_uniform(low: f64, high: f64, operator: &str) -> Result<()> {
    if !low.is_finite() || !high.is_finite() || low >= high {
        return Err(TynxError::Shape(format!(
            "{operator} requires finite low < high, got {low} and {high}"
        )));
    }
    Ok(())
}

fn seed_device(name: &str, device: &Device) {
    if let Some(seed) = encoded_seed(name) {
        device.seed(seed);
    }
}

fn encoded_seed(name: &str) -> Option<u64> {
    name.strip_prefix(SEED_PREFIX)
        .and_then(|name| name.split_once(':'))
        .and_then(|(seed, _)| seed.parse().ok())
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::{
            bernoulli::BernoulliNodeBuilder,
            random::{
                RandomNormalConfig, RandomNormalNodeBuilder, RandomUniformConfig,
                RandomUniformNodeBuilder,
            },
            random_like::{RandomNormalLikeConfig, RandomNormalLikeNodeBuilder},
        },
    };

    use super::*;

    #[test]
    fn bernoulli_samples_are_binary() {
        let node = BernoulliNodeBuilder::new(format!("{SEED_PREFIX}42:bernoulli"))
            .input_tensor("probabilities", 1, DType::F32)
            .output_tensor("samples", 1, DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "probabilities".into(),
            Value::from_tensor_data(TensorData::new(vec![0.5_f32; 32], [32]), 1, &device).unwrap(),
        );

        let first = bernoulli(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();
        assert!(first.iter().all(|value| matches!(value, 0.0 | 1.0)));
    }

    #[test]
    fn random_normal_uses_configured_shape_and_dtype() {
        let node = RandomNormalNodeBuilder::new(format!("{SEED_PREFIX}42:normal"))
            .output_tensor("samples", 2, DType::F64)
            .config(RandomNormalConfig {
                mean: 2.0,
                scale: 0.5,
                shape: vec![2, 3],
            })
            .build();
        let device = Device::default();

        let first = random_normal(&node, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();
        assert_eq!(first.dims(), vec![2, 3]);
        assert_eq!(first.dtype(), DType::F64);
        assert!(first.into_data().iter::<f64>().all(f64::is_finite));
    }

    #[test]
    fn random_normal_like_uses_input_shape_and_output_dtype() {
        let node = RandomNormalLikeNodeBuilder::new("normal_like")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("samples", 2, DType::F64)
            .config(RandomNormalLikeConfig {
                mean: 0.0,
                scale: 1.0,
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "input".into(),
            Value::from_tensor_data(TensorData::new(vec![0.0_f32; 6], [3, 2]), 2, &device).unwrap(),
        );

        let output = random_normal_like(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();

        assert_eq!(output.dims(), vec![3, 2]);
        assert_eq!(output.dtype(), DType::F64);
    }

    #[test]
    fn random_uniform_rejects_an_invalid_range() {
        let node = RandomUniformNodeBuilder::new("uniform")
            .output_tensor("samples", 1, DType::F32)
            .config(RandomUniformConfig {
                low: 1.0,
                high: 1.0,
                shape: vec![4],
            })
            .build();

        let error = random_uniform(&node, &Device::default()).unwrap_err();

        assert_eq!(
            error,
            TynxError::Shape("RandomUniform requires finite low < high, got 1 and 1".to_string())
        );
    }

    #[test]
    fn extracts_an_encoded_seed_from_a_node_name() {
        assert_eq!(encoded_seed("__tynx_random_seed_42:normal"), Some(42));
        assert_eq!(encoded_seed("normal"), None);
    }
}
