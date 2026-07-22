//! Random tensor and Bernoulli operator execution.

use std::collections::VecDeque;

use burn::tensor::{Device, Distribution};
use onnx_ir::{
    ModelProto, Node,
    ir::OnnxGraph,
    node::{bernoulli::BernoulliNode, random_like::RandomUniformLikeNode},
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
    let mut uniform_like = seeds(raw_graph, "RandomUniformLike");

    for node in &mut graph.nodes {
        let (name, seed) = match node {
            Node::Bernoulli(node) => (&mut node.name, bernoulli.pop_front()),
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

pub(super) fn random_uniform_like(
    node: &RandomUniformLikeNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    seed_device(&node.name, device);
    if !node.config.low.is_finite()
        || !node.config.high.is_finite()
        || node.config.low >= node.config.high
    {
        return Err(TynxError::Shape(format!(
            "RandomUniformLike requires finite low < high, got {} and {}",
            node.config.low, node.config.high
        )));
    }
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let output = DynTensor::random(
        &shape::value_dims(&input),
        Distribution::Uniform(node.config.low, node.config.high),
        device,
        node.outputs[0].ty.elem_type(),
    )?;
    Ok(vec![Value::Tensor(output)])
}

fn seed_device(name: &str, device: &Device) {
    if let Some(seed) = name
        .strip_prefix(SEED_PREFIX)
        .and_then(|name| name.split_once(':'))
        .and_then(|(seed, _)| seed.parse().ok())
    {
        device.seed(seed);
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{DType, node::bernoulli::BernoulliNodeBuilder};

    use super::*;

    #[test]
    fn seeded_bernoulli_is_reproducible_and_binary() {
        let node = BernoulliNodeBuilder::new(format!("{SEED_PREFIX}0:bernoulli"))
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
        let second = bernoulli(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(first, second);
        assert!(first.iter().all(|value| matches!(value, 0.0 | 1.0)));
    }
}
