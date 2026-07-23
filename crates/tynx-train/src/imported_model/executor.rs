//! Internal graph execution with live parameter-slot resolution.

use burn::tensor::Device;
use tynx_core::onnx_ir::{
    Node,
    ir::{ArgType, Argument, OnnxGraph},
    node::{
        batch_norm::{BatchNormConfig, BatchNormalizationNode},
        conv2d::Conv2dNode,
        gather::GatherNode,
        gemm::GemmNode,
        layer_norm::LayerNormalizationNode,
        linear::LinearNode,
        matmul::MatMulNode,
    },
};
use tynx_core::{
    DynTensor, Env, Result, Session, TynxError, Value, execute, execute_onnx_gather,
    execute_onnx_layer_normalization, resolve_onnx_padding2d,
};

use crate::ImportedState;

/// Validate that every materialized initializer is consumed by a slot-aware operator path.
pub(super) fn validate(graph: &OnnxGraph, state: &ImportedState) -> Result<()> {
    for (node_index, node) in graph.nodes.iter().enumerate() {
        for (input_index, input) in node.inputs().iter().enumerate() {
            if state
                .slot_for_input(node_index, input_index, input)
                .is_none()
            {
                continue;
            }
            let supported = match node {
                Node::Linear(_) | Node::Gemm(_) => matches!(input_index, 1 | 2),
                Node::MatMul(_) => matches!(input_index, 0 | 1),
                Node::BatchNormalization(_) => matches!(input_index, 1..=4),
                Node::Conv2d(_) => matches!(input_index, 0..=2),
                Node::Gather(_) => input_index == 0,
                Node::LayerNormalization(_) => matches!(input_index, 0..=2),
                _ => false,
            };
            if !supported {
                return Err(TynxError::UnsupportedOp(format!(
                    "slot-backed execution for {} node '{}' input {} is not implemented",
                    node_kind(node),
                    node.name(),
                    input_index
                )));
            }
        }
    }
    Ok(())
}

/// Execute a graph, specializing only nodes that need live slot-backed state.
pub(super) fn run(
    session: &Session,
    state: &ImportedState,
    device: &Device,
    mut env: Env,
    tracking: bool,
) -> Result<Env> {
    for (node_index, node) in session.graph().nodes.iter().enumerate() {
        let values = match node {
            Node::Linear(node) => execute_linear(node, node_index, &env, state, device, tracking),
            Node::Gemm(node) => execute_gemm(node, node_index, &env, state, device, tracking),
            Node::MatMul(node) => execute_matmul(node, node_index, &env, state, device, tracking),
            Node::BatchNormalization(node) => {
                execute_batch_normalization(node, node_index, &env, state, device, tracking)
            }
            Node::Conv2d(node) => execute_conv2d(node, node_index, &env, state, device, tracking),
            Node::Gather(node) => execute_gather(node, node_index, &env, state, device, tracking),
            Node::LayerNormalization(node) => {
                execute_layer_normalization(node, node_index, &env, state, device, tracking)
            }
            _ => execute(node, &env, device),
        }?;
        if values.len() != node.outputs().len() {
            return Err(TynxError::Shape(format!(
                "node '{}' returned {} values for {} outputs",
                node.name(),
                values.len(),
                node.outputs().len()
            )));
        }
        for (output, value) in node.outputs().iter().zip(values) {
            env.insert(output.name.clone(), value);
        }
    }

    session.collect_outputs(&env)
}

fn execute_layer_normalization(
    node: &LayerNormalizationNode,
    node_index: usize,
    env: &Env,
    state: &ImportedState,
    device: &Device,
    tracking: bool,
) -> Result<Vec<Value>> {
    let input = resolve_tensor(node, node_index, 0, env, state, device, tracking)?;
    let scale = resolve_tensor(node, node_index, 1, env, state, device, tracking)?;
    let bias = node
        .inputs
        .get(2)
        .filter(|input| !input.is_optional())
        .map(|_| resolve_tensor(node, node_index, 2, env, state, device, tracking))
        .transpose()?;
    execute_onnx_layer_normalization(
        input,
        scale,
        bias,
        node.config.epsilon,
        node.config.full_precision,
        node.outputs.len(),
    )
}

fn execute_gather(
    node: &GatherNode,
    node_index: usize,
    env: &Env,
    state: &ImportedState,
    device: &Device,
    tracking: bool,
) -> Result<Vec<Value>> {
    let data = resolve_value(node, node_index, 0, env, state, device, tracking)?;
    let indices = resolve_value(node, node_index, 1, env, state, device, tracking)?;
    execute_onnx_gather(data, indices, node.config.axis, device).map(|value| vec![value])
}

fn execute_conv2d(
    node: &Conv2dNode,
    node_index: usize,
    env: &Env,
    state: &ImportedState,
    device: &Device,
    tracking: bool,
) -> Result<Vec<Value>> {
    let input = resolve_tensor(node, node_index, 0, env, state, device, tracking)?;
    let weight = resolve_tensor(node, node_index, 1, env, state, device, tracking)?;
    let bias = node
        .inputs
        .get(2)
        .filter(|input| !input.is_optional())
        .map(|_| resolve_tensor(node, node_index, 2, env, state, device, tracking))
        .transpose()?;
    let shape = input.dims();
    if shape.len() != 4 {
        return Err(TynxError::Shape(format!(
            "training Conv2d node '{}' requires rank-4 NCHW input, got shape {shape:?}",
            node.name
        )));
    }
    let [(top, bottom), (left, right)] = resolve_onnx_padding2d(
        [shape[2], shape[3]],
        node.config.kernel_size,
        node.config.stride,
        node.config.dilation,
        &node.config.padding,
        &node.config.auto_pad,
        false,
    );
    let output = input.conv2d_padded(
        weight,
        bias,
        node.config.stride,
        [[top, bottom], [left, right]],
        node.config.dilation,
        node.config.groups,
    )?;
    Ok(vec![Value::Tensor(output)])
}

fn execute_batch_normalization(
    node: &BatchNormalizationNode,
    node_index: usize,
    env: &Env,
    state: &ImportedState,
    device: &Device,
    tracking: bool,
) -> Result<Vec<Value>> {
    if node.outputs.len() != 1 {
        return Err(TynxError::UnsupportedOp(
            "BatchNormalization training outputs".to_string(),
        ));
    }
    let input = resolve_tensor(node, node_index, 0, env, state, device, tracking)?;
    let shape = input.dims();
    let channels = shape.get(1).copied().ok_or_else(|| {
        TynxError::Shape(format!(
            "training BatchNormalization node '{}' requires rank at least 2, got shape {shape:?}",
            node.name
        ))
    })?;
    let scale = batch_norm_parameter(
        node,
        node_index,
        1,
        channels,
        input.rank(),
        env,
        state,
        device,
        tracking,
    )?;
    let bias = batch_norm_parameter(
        node,
        node_index,
        2,
        channels,
        input.rank(),
        env,
        state,
        device,
        tracking,
    )?;
    let mean = batch_norm_parameter(
        node,
        node_index,
        3,
        channels,
        input.rank(),
        env,
        state,
        device,
        false,
    )?;
    let variance = batch_norm_parameter(
        node,
        node_index,
        4,
        channels,
        input.rank(),
        env,
        state,
        device,
        false,
    )?;
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

#[allow(clippy::too_many_arguments)]
fn batch_norm_parameter(
    node: &BatchNormalizationNode,
    node_index: usize,
    input_index: usize,
    channels: usize,
    input_rank: usize,
    env: &Env,
    state: &ImportedState,
    device: &Device,
    tracking: bool,
) -> Result<DynTensor> {
    let parameter = resolve_tensor(node, node_index, input_index, env, state, device, tracking)?;
    if parameter.dims() != [channels] {
        return Err(TynxError::Shape(format!(
            "training BatchNormalization node '{}' input {input_index} expects [{channels}], got {:?}",
            node.name,
            parameter.dims()
        )));
    }
    let mut shape = vec![1; input_rank];
    shape[1] = channels;
    parameter.reshape(shape)
}

fn execute_linear(
    node: &LinearNode,
    node_index: usize,
    env: &Env,
    state: &ImportedState,
    device: &Device,
    tracking: bool,
) -> Result<Vec<Value>> {
    let input = resolve_tensor(node, node_index, 0, env, state, device, tracking)?;
    let mut weight = resolve_tensor(node, node_index, 1, env, state, device, tracking)?;
    if node.config.transpose_weight {
        weight = weight.permute(vec![1, 0])?;
    }
    let mut output = rank2_matmul(&node.name, input, weight)?;
    if node.inputs.get(2).is_some_and(|input| !input.is_optional()) {
        output = add_bias(
            output,
            resolve_value(node, node_index, 2, env, state, device, tracking)?,
            1.0,
        )?;
    }
    Ok(vec![Value::Tensor(output)])
}

fn execute_gemm(
    node: &GemmNode,
    node_index: usize,
    env: &Env,
    state: &ImportedState,
    device: &Device,
    tracking: bool,
) -> Result<Vec<Value>> {
    let mut left = resolve_tensor(node, node_index, 0, env, state, device, tracking)?;
    let mut right = resolve_tensor(node, node_index, 1, env, state, device, tracking)?;
    if node.config.trans_a != 0 {
        left = left.permute(vec![1, 0])?;
    }
    if node.config.trans_b != 0 {
        right = right.permute(vec![1, 0])?;
    }
    let mut output = rank2_matmul(&node.name, left, right)?.mul_scalar(node.config.alpha as f64);
    if node.inputs.get(2).is_some_and(|input| !input.is_optional()) {
        output = add_bias(
            output,
            resolve_value(node, node_index, 2, env, state, device, tracking)?,
            node.config.beta as f64,
        )?;
    }
    Ok(vec![Value::Tensor(output)])
}

fn execute_matmul(
    node: &MatMulNode,
    node_index: usize,
    env: &Env,
    state: &ImportedState,
    device: &Device,
    tracking: bool,
) -> Result<Vec<Value>> {
    let left = resolve_tensor(node, node_index, 0, env, state, device, tracking)?;
    let right = resolve_tensor(node, node_index, 1, env, state, device, tracking)?;
    Ok(vec![Value::Tensor(rank2_matmul(&node.name, left, right)?)])
}

fn resolve_tensor<N: NodeInputs>(
    node: &N,
    node_index: usize,
    input_index: usize,
    env: &Env,
    state: &ImportedState,
    device: &Device,
    tracking: bool,
) -> Result<DynTensor> {
    resolve_value(node, node_index, input_index, env, state, device, tracking)?.into_tensor()
}

fn resolve_value<N: NodeInputs>(
    node: &N,
    node_index: usize,
    input_index: usize,
    env: &Env,
    state: &ImportedState,
    device: &Device,
    tracking: bool,
) -> Result<Value> {
    let input = node.inputs().get(input_index).ok_or_else(|| {
        TynxError::Shape(format!(
            "node '{}' has no input at index {input_index}",
            node.name()
        ))
    })?;
    if let Some(slot) = state.slot_for_input(node_index, input_index, input) {
        return Ok(Value::Tensor(if tracking {
            slot.read()
        } else {
            slot.value()
        }));
    }
    if let Some(value) = env.get(&input.name) {
        return Ok(value.clone());
    }
    if let Some(data) = input.value() {
        if matches!(input.ty, ArgType::Shape(_)) {
            return Ok(Value::Shape(data.iter::<i64>().collect()));
        }
        let rank = data.shape.len();
        return Value::from_tensor_data(data, rank, device);
    }
    let name = if input.name.is_empty() {
        "<static/absent>"
    } else {
        &input.name
    };
    Err(TynxError::MissingValue(name.to_string()))
}

fn rank2_matmul(node_name: &str, left: DynTensor, right: DynTensor) -> Result<DynTensor> {
    if left.rank() != 2 || right.rank() != 2 {
        return Err(TynxError::Shape(format!(
            "training node '{node_name}' currently requires rank-2 matrix inputs, got ranks {} and {}",
            left.rank(),
            right.rank()
        )));
    }
    left.matmul(right)
}

fn add_bias(output: DynTensor, bias: Value, scale: f64) -> Result<DynTensor> {
    match bias {
        Value::Tensor(bias) => output.add_broadcast(bias.mul_scalar(scale)),
        Value::Scalar(bias) => Ok(output.add_scalar(scale * bias.as_f64())),
        other => Err(TynxError::TypeMismatch(format!(
            "Linear/Gemm bias must be a float tensor or scalar, got {other:?}"
        ))),
    }
}

fn node_kind(node: &Node) -> String {
    format!("{node:?}")
        .split(['(', '{'])
        .next()
        .unwrap_or("Unknown")
        .to_string()
}

trait NodeInputs {
    fn name(&self) -> &str;
    fn inputs(&self) -> &[Argument];
}

macro_rules! impl_node_inputs {
    ($($type:ty),+ $(,)?) => {
        $(
            impl NodeInputs for $type {
                fn name(&self) -> &str {
                    &self.name
                }

                fn inputs(&self) -> &[Argument] {
                    &self.inputs
                }
            }
        )+
    };
}

impl_node_inputs!(
    LinearNode,
    GemmNode,
    MatMulNode,
    BatchNormalizationNode,
    Conv2dNode,
    GatherNode,
    LayerNormalizationNode,
);
