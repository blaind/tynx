//! ONNX reduction operator execution.

use std::collections::{HashMap, VecDeque};

use burn::tensor::Device;
use onnx_ir::{
    ModelProto, Node,
    ir::{Argument, OnnxGraph},
    node::reduce::{
        ReduceConfig, ReduceL1Node, ReduceL2Node, ReduceLogSumExpNode, ReduceLogSumNode,
        ReduceMaxNode, ReduceMeanNode, ReduceMinNode, ReduceProdNode, ReduceSumNode,
        ReduceSumSquareNode,
    },
};
use protobuf::Message;

use super::{Env, resolve, shape};
use crate::{DynTensor, Result, TynxError, Value};

const NOOP_EMPTY_PREFIX: &str = "__tynx_reduce_noop_empty__";
type ReductionFlags = HashMap<(String, String), VecDeque<bool>>;

pub(super) fn preserve_attributes(data: &[u8], graph: &mut OnnxGraph) -> Result<()> {
    let mut flags = reduction_noop_flags(data)?;
    for node in &mut graph.nodes {
        match node {
            Node::ReduceL1(node) => mark_reduction("ReduceL1", node, &mut flags),
            Node::ReduceL2(node) => mark_reduction("ReduceL2", node, &mut flags),
            Node::ReduceLogSum(node) => mark_reduction("ReduceLogSum", node, &mut flags),
            Node::ReduceLogSumExp(node) => mark_reduction("ReduceLogSumExp", node, &mut flags),
            Node::ReduceMax(node) => mark_reduction("ReduceMax", node, &mut flags),
            Node::ReduceMean(node) => mark_reduction("ReduceMean", node, &mut flags),
            Node::ReduceMin(node) => mark_reduction("ReduceMin", node, &mut flags),
            Node::ReduceProd(node) => mark_reduction("ReduceProd", node, &mut flags),
            Node::ReduceSum(node) => mark_reduction("ReduceSum", node, &mut flags),
            Node::ReduceSumSquare(node) => mark_reduction("ReduceSumSquare", node, &mut flags),
            _ => {}
        }
    }
    Ok(())
}

fn reduction_noop_flags(data: &[u8]) -> Result<ReductionFlags> {
    let model =
        ModelProto::parse_from_bytes(data).map_err(|error| TynxError::Parse(error.to_string()))?;
    let Some(graph) = model.graph.as_ref() else {
        return Ok(HashMap::new());
    };
    let mut flags = HashMap::<_, VecDeque<_>>::new();

    for node in &graph.node {
        if node.op_type.starts_with("Reduce") {
            let Some(input) = node.input.first() else {
                continue;
            };
            let noop = node
                .attribute
                .iter()
                .any(|attribute| attribute.name == "noop_with_empty_axes" && attribute.i == 1);
            flags
                .entry((node.op_type.clone(), input.clone()))
                .or_default()
                .push_back(noop);
        }
    }
    Ok(flags)
}

fn mark_reduction<N>(operator: &str, node: &mut N, flags: &mut ReductionFlags)
where
    N: ReductionNode,
{
    let Some(input) = node.inputs().first() else {
        return;
    };
    let key = (operator.to_string(), input.name.clone());
    if flags.get_mut(&key).and_then(VecDeque::pop_front) == Some(true) {
        *node.name_mut() = format!("{NOOP_EMPTY_PREFIX}{}", node.name());
    }
}

trait ReductionNode {
    fn name(&self) -> &str;
    fn name_mut(&mut self) -> &mut String;
    fn inputs(&self) -> &[Argument];
}

macro_rules! impl_reduction_node {
    ($($type:ty),+ $(,)?) => {
        $(
            impl ReductionNode for $type {
                fn name(&self) -> &str { &self.name }
                fn name_mut(&mut self) -> &mut String { &mut self.name }
                fn inputs(&self) -> &[Argument] { &self.inputs }
            }
        )+
    };
}

impl_reduction_node!(
    ReduceL1Node,
    ReduceL2Node,
    ReduceLogSumNode,
    ReduceLogSumExpNode,
    ReduceMaxNode,
    ReduceMeanNode,
    ReduceMinNode,
    ReduceProdNode,
    ReduceSumNode,
    ReduceSumSquareNode,
);

#[derive(Debug, Clone, Copy)]
enum Operation {
    L1,
    L2,
    LogSum,
    LogSumExp,
    Max,
    Mean,
    Min,
    Prod,
    Sum,
    SumSquare,
}

macro_rules! reduce_entry {
    ($function:ident, $node:ty, $operation:ident) => {
        pub(super) fn $function(node: &$node, env: &Env, device: &Device) -> Result<Vec<Value>> {
            reduce(
                &node.name,
                &node.inputs,
                &node.config,
                Operation::$operation,
                env,
                device,
            )
        }
    };
}

reduce_entry!(reduce_l1, ReduceL1Node, L1);
reduce_entry!(reduce_l2, ReduceL2Node, L2);
reduce_entry!(reduce_log_sum, ReduceLogSumNode, LogSum);
reduce_entry!(reduce_log_sum_exp, ReduceLogSumExpNode, LogSumExp);
reduce_entry!(reduce_max, ReduceMaxNode, Max);
reduce_entry!(reduce_mean, ReduceMeanNode, Mean);
reduce_entry!(reduce_min, ReduceMinNode, Min);
reduce_entry!(reduce_prod, ReduceProdNode, Prod);
reduce_entry!(reduce_sum, ReduceSumNode, Sum);
reduce_entry!(reduce_sum_square, ReduceSumSquareNode, SumSquare);

fn reduce(
    node_name: &str,
    inputs: &[Argument],
    config: &ReduceConfig,
    operation: Operation,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::first(env, node_name, inputs, device)?;
    let input_dims = shape::value_dims(&input);
    let axes = if config.dims.is_empty()
        && inputs
            .get(1)
            .is_some_and(|argument| !argument.is_optional())
    {
        let raw_axes = shape::value_to_i64s(resolve::at(env, node_name, inputs, 1, device)?)?;
        if raw_axes.is_empty() {
            if node_name.starts_with(NOOP_EMPTY_PREFIX) {
                return Ok(vec![input]);
            }
            (0..input_dims.len()).collect()
        } else {
            normalize_axes(&raw_axes, input_dims.len())?
        }
    } else {
        axes(config, input_dims.len())?
    };

    let reduced_dims = input_dims
        .iter()
        .enumerate()
        .map(|(axis, &dim)| if axes.contains(&axis) { 1 } else { dim })
        .collect::<Vec<_>>();
    let has_empty_reduced_axis = axes.iter().any(|&axis| input_dims[axis] == 0);

    let reduced = match input {
        Value::Tensor(tensor) => {
            let dtype = tensor.dtype();
            Value::Tensor(match operation {
                Operation::Max if has_empty_reduced_axis => {
                    DynTensor::full(&reduced_dims, f64::NEG_INFINITY, device, dtype)?
                }
                Operation::Min if has_empty_reduced_axis => {
                    DynTensor::full(&reduced_dims, f64::INFINITY, device, dtype)?
                }
                Operation::L1 => tensor.abs().sum_dims(&axes),
                Operation::L2 => tensor.powi_scalar(2).sum_dims(&axes).sqrt(),
                Operation::LogSum => tensor.sum_dims(&axes).log(),
                Operation::LogSumExp => tensor.exp().sum_dims(&axes).log(),
                Operation::Max => tensor.reduce_max_dims(&axes),
                Operation::Mean => tensor.mean_dims(&axes),
                Operation::Min => tensor.reduce_min_dims(&axes),
                Operation::Prod => tensor.prod_dims(&axes),
                Operation::Sum => tensor.sum_dims(&axes),
                Operation::SumSquare => tensor.powi_scalar(2).sum_dims(&axes),
            })
        }
        Value::Int(tensor) => Value::Int(match operation {
            Operation::L1 => tensor.abs().sum_dims(&axes),
            Operation::Max => tensor.reduce_max_dims(&axes),
            Operation::Min => tensor.reduce_min_dims(&axes),
            Operation::Prod => tensor.prod_dims(&axes),
            Operation::Sum => tensor.sum_dims(&axes),
            Operation::SumSquare => tensor.powi_scalar(2).sum_dims(&axes),
            other => {
                return Err(TynxError::TypeMismatch(format!(
                    "{other:?} reduction requires a floating-point tensor"
                )));
            }
        }),
        Value::Bool(tensor) => Value::Bool(match operation {
            Operation::Max => tensor.reduce_max_dims(&axes),
            Operation::Min => tensor.reduce_min_dims(&axes),
            other => {
                return Err(TynxError::TypeMismatch(format!(
                    "{other:?} reduction does not support boolean tensors"
                )));
            }
        }),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "reduction expects a tensor, got {other:?}"
            )));
        }
    };

    let output_dims = if config.keepdims {
        input_dims
            .iter()
            .enumerate()
            .map(|(axis, &dim)| if axes.contains(&axis) { 1 } else { dim })
            .collect()
    } else {
        input_dims
            .into_iter()
            .enumerate()
            .filter_map(|(axis, dim)| (!axes.contains(&axis)).then_some(dim))
            .collect()
    };

    Ok(vec![shape::reshape_value(reduced, output_dims, device)?])
}

fn axes(config: &ReduceConfig, rank: usize) -> Result<Vec<usize>> {
    let axes = if config.dims.is_empty() {
        (0..rank).collect::<Vec<_>>()
    } else {
        config.dims.clone()
    };
    let mut seen = vec![false; rank];
    for &axis in &axes {
        let Some(slot) = seen.get_mut(axis) else {
            return Err(TynxError::Shape(format!(
                "reduction axis {axis} is out of range for rank {rank}"
            )));
        };
        if *slot {
            return Err(TynxError::Shape(format!(
                "reduction axis {axis} appears more than once"
            )));
        }
        *slot = true;
    }
    Ok(axes)
}

fn normalize_axes(raw_axes: &[i64], rank: usize) -> Result<Vec<usize>> {
    let rank = i64::try_from(rank)
        .map_err(|_| TynxError::Shape("reduction rank exceeds i64".to_string()))?;
    let dims = raw_axes
        .iter()
        .map(|&axis| if axis < 0 { axis + rank } else { axis })
        .map(|axis| {
            usize::try_from(axis)
                .map_err(|_| TynxError::Shape(format!("invalid reduction axis {axis}")))
        })
        .collect::<Result<Vec<_>>>()?;
    axes(
        &ReduceConfig {
            dims,
            keepdims: false,
        },
        rank as usize,
    )
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::reduce::{ReduceConfig, ReduceSumNodeBuilder},
    };

    use super::*;

    #[test]
    fn sums_selected_axes_without_keepdims() {
        let node = ReduceSumNodeBuilder::new("reduce")
            .input_tensor("x", 3, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .config(ReduceConfig {
                dims: vec![0, 2],
                keepdims: false,
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(
                    (1..=8).map(|value| value as f32).collect::<Vec<_>>(),
                    [2, 2, 2],
                ),
                3,
                &device,
            )
            .unwrap(),
        );

        let output = reduce_sum(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();

        assert_eq!(output.dims(), [2]);
        assert_eq!(
            output.into_data().iter::<f32>().collect::<Vec<_>>(),
            [14.0, 22.0]
        );
    }
}
