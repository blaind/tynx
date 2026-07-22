//! ONNX convolution execution.

use std::collections::{HashMap, HashSet, VecDeque};

use burn::tensor::{
    DType, Device, Tensor,
    module::{
        conv_transpose1d as burn_conv_transpose1d, conv_transpose2d as burn_conv_transpose2d,
        conv_transpose3d as burn_conv_transpose3d, conv1d as burn_conv1d, conv2d as burn_conv2d,
        conv3d as burn_conv3d,
    },
    ops::{ConvOptions, ConvTransposeOptions, PadMode, PaddedConvOptions},
};
use onnx_ir::{
    ModelProto, Node, TensorProto, ValueInfoProto,
    ir::{Argument, OnnxGraph},
    node::{
        conv_transpose1d::ConvTranspose1dNode, conv_transpose2d::ConvTranspose2dNode,
        conv_transpose3d::ConvTranspose3dNode, conv1d::Conv1dNode, conv2d::Conv2dNode,
        conv3d::Conv3dNode,
    },
};
use protobuf::Message;

use super::{
    Env, resolve,
    spatial::{padding1d, padding2d, padding3d, rank3, rank4, rank5},
};
use crate::{DynTensor, Result, TynxError, Value};

// The pinned onnx-ir revision requires Conv weight data while extracting the
// kernel shape. Supply an allocation-free placeholder for parsing, then restore
// the original dynamic argument before the session can execute the graph.
pub(super) fn prepare_model(data: &[u8]) -> Result<(Vec<u8>, bool)> {
    let mut model =
        ModelProto::parse_from_bytes(data).map_err(|error| TynxError::Parse(error.to_string()))?;
    let Some(graph) = model.graph.as_mut() else {
        return Ok((Vec::new(), false));
    };
    let original_initializers = graph
        .initializer
        .iter()
        .map(|initializer| initializer.name.clone())
        .collect::<HashSet<_>>();
    let dynamic_weights = graph
        .node
        .iter()
        .filter(|node| matches!(node.op_type.as_str(), "Conv" | "ConvTranspose"))
        .filter_map(|node| node.input.get(1))
        .filter(|name| !original_initializers.contains(*name))
        .cloned()
        .collect::<HashSet<_>>();

    let mut injected = false;
    for name in dynamic_weights {
        let Some(value_info) = graph.input.iter().find(|value| value.name == name) else {
            continue;
        };
        let Some(initializer) = dummy_initializer(value_info) else {
            continue;
        };
        graph.initializer.push(initializer);
        injected = true;
    }

    if !injected {
        return Ok((Vec::new(), false));
    }
    let bytes = model
        .write_to_bytes()
        .map_err(|error| TynxError::Parse(error.to_string()))?;
    Ok((bytes, true))
}

pub(super) fn restore_dynamic_inputs(data: &[u8], graph: &mut OnnxGraph) -> Result<()> {
    let model =
        ModelProto::parse_from_bytes(data).map_err(|error| TynxError::Parse(error.to_string()))?;
    let raw_graph = &model.graph;
    let initializer_names = raw_graph
        .initializer
        .iter()
        .map(|initializer| initializer.name.as_str())
        .collect::<HashSet<_>>();
    let runtime_inputs = raw_graph
        .input
        .iter()
        .filter(|input| !initializer_names.contains(input.name.as_str()))
        .map(|input| {
            Argument::try_from(input.clone())
                .map(|argument| (input.name.clone(), argument))
                .map_err(|error| TynxError::Parse(format!("{error:?}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let arguments = runtime_inputs
        .iter()
        .map(|(_, argument)| (argument.name.clone(), argument.clone()))
        .collect::<HashMap<_, _>>();
    let raw_arguments = runtime_inputs.iter().cloned().collect::<HashMap<_, _>>();
    graph.inputs = runtime_inputs
        .into_iter()
        .map(|(_, argument)| argument)
        .collect();

    let mut weights = VecDeque::<Option<String>>::new();
    for node in &raw_graph.node {
        if !matches!(node.op_type.as_str(), "Conv" | "ConvTranspose") || node.input.len() < 2 {
            continue;
        }
        let weight = &node.input[1];
        let dynamic = if initializer_names.contains(weight.as_str()) {
            None
        } else {
            raw_arguments
                .get(weight)
                .map(|argument| argument.name.clone())
        };
        weights.push_back(dynamic);
    }

    for node in &mut graph.nodes {
        match node {
            Node::Conv1d(node) => restore_weight(node, &arguments, &mut weights),
            Node::Conv2d(node) => restore_weight(node, &arguments, &mut weights),
            Node::Conv3d(node) => restore_weight(node, &arguments, &mut weights),
            Node::ConvTranspose1d(node) => restore_weight(node, &arguments, &mut weights),
            Node::ConvTranspose2d(node) => restore_weight(node, &arguments, &mut weights),
            Node::ConvTranspose3d(node) => restore_weight(node, &arguments, &mut weights),
            _ => {}
        }
    }
    Ok(())
}

fn dummy_initializer(value_info: &ValueInfoProto) -> Option<TensorProto> {
    let tensor_type = value_info.type_.as_ref()?.tensor_type();
    let shape = tensor_type.shape.as_ref()?;
    let mut dims = shape
        .dim
        .iter()
        .map(|dimension| dimension.has_dim_value().then(|| dimension.dim_value()))
        .collect::<Option<Vec<_>>>()?;
    if dims.iter().any(|dimension| *dimension <= 0)
        || dims.len() < 2
        || !matches!(tensor_type.elem_type, 1 | 10 | 11 | 16)
    {
        return None;
    }
    // Preserve the rank, output channels, and kernel dimensions needed by
    // onnx-ir while keeping the temporary initializer allocation-free.
    dims[1] = 0;
    let mut tensor = TensorProto::new();
    tensor.name = value_info.name.clone();
    tensor.dims = dims;
    tensor.data_type = tensor_type.elem_type;
    Some(tensor)
}

fn restore_weight<N: ConvNode>(
    node: &mut N,
    arguments: &HashMap<String, Argument>,
    weights: &mut VecDeque<Option<String>>,
) {
    let Some(Some(weight_name)) = weights.pop_front() else {
        return;
    };
    if let Some(argument) = arguments.get(&weight_name) {
        node.inputs_mut()[1] = argument.clone();
    }
}

trait ConvNode {
    fn inputs_mut(&mut self) -> &mut [Argument];
}

macro_rules! impl_conv_node {
    ($($type:ty),+ $(,)?) => {
        $(
            impl ConvNode for $type {
                fn inputs_mut(&mut self) -> &mut [Argument] { &mut self.inputs }
            }
        )+
    };
}

impl_conv_node!(
    Conv1dNode,
    Conv2dNode,
    Conv3dNode,
    ConvTranspose1dNode,
    ConvTranspose2dNode,
    ConvTranspose3dNode,
);

pub(super) fn conv1d(node: &Conv1dNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = rank3(resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?)?;
    let dtype = input.dtype();
    let weight = rank3(
        resolve::at(env, &node.name, &node.inputs, 1, device)?
            .into_tensor()?
            .cast(dtype),
    )?;
    let bias = bias(&node.name, &node.inputs, env, device, dtype)?;
    let dims = input.dims();
    let [(left, right)] = padding1d(
        dims[2],
        node.config.kernel_size,
        node.config.stride,
        node.config.dilation,
        &node.config.padding,
        &node.config.auto_pad,
        false,
    );
    let options = PaddedConvOptions::asymmetric(
        [node.config.stride],
        [left],
        [right],
        [node.config.dilation],
        node.config.groups,
    );
    Ok(vec![Value::Tensor(DynTensor::R3(burn_conv1d(
        input, weight, bias, options,
    )))])
}

pub(super) fn conv2d(node: &Conv2dNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = rank4(resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?)?;
    let dtype = input.dtype();
    let weight = rank4(
        resolve::at(env, &node.name, &node.inputs, 1, device)?
            .into_tensor()?
            .cast(dtype),
    )?;
    let bias = bias(&node.name, &node.inputs, env, device, dtype)?;
    let dims = input.dims();
    let [(top, bottom), (left, right)] = padding2d(
        [dims[2], dims[3]],
        node.config.kernel_size,
        node.config.stride,
        node.config.dilation,
        &node.config.padding,
        &node.config.auto_pad,
        false,
    );
    let options = PaddedConvOptions::asymmetric(
        node.config.stride,
        [top, left],
        [bottom, right],
        node.config.dilation,
        node.config.groups,
    );
    Ok(vec![Value::Tensor(DynTensor::R4(burn_conv2d(
        input, weight, bias, options,
    )))])
}

pub(super) fn conv3d(node: &Conv3dNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = rank5(resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?)?;
    let dtype = input.dtype();
    let weight = rank5(
        resolve::at(env, &node.name, &node.inputs, 1, device)?
            .into_tensor()?
            .cast(dtype),
    )?;
    let bias = bias(&node.name, &node.inputs, env, device, dtype)?;
    let dims = input.dims();
    let padding = padding3d(
        [dims[2], dims[3], dims[4]],
        node.config.kernel_size,
        node.config.stride,
        node.config.dilation,
        &node.config.padding,
        &node.config.auto_pad,
        false,
    );
    let input = input.pad(padding, PadMode::Constant(0.0));
    let options = ConvOptions::new(
        node.config.stride,
        [0; 3],
        node.config.dilation,
        node.config.groups,
    );
    Ok(vec![Value::Tensor(DynTensor::R5(burn_conv3d(
        input, weight, bias, options,
    )))])
}

pub(super) fn conv_transpose1d(
    node: &ConvTranspose1dNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = rank3(resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?)?;
    let dtype = input.dtype();
    let weight = rank3(
        resolve::at(env, &node.name, &node.inputs, 1, device)?
            .into_tensor()?
            .cast(dtype),
    )?;
    let bias = bias(&node.name, &node.inputs, env, device, dtype)?;
    let options = ConvTransposeOptions::new(
        [node.config.stride],
        [node.config.padding],
        [node.config.padding_out],
        [node.config.dilation],
        node.config.groups,
    );
    Ok(vec![Value::Tensor(DynTensor::R3(burn_conv_transpose1d(
        input, weight, bias, options,
    )))])
}

pub(super) fn conv_transpose2d(
    node: &ConvTranspose2dNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = rank4(resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?)?;
    let dtype = input.dtype();
    let weight = rank4(
        resolve::at(env, &node.name, &node.inputs, 1, device)?
            .into_tensor()?
            .cast(dtype),
    )?;
    let bias = bias(&node.name, &node.inputs, env, device, dtype)?;
    let options = ConvTransposeOptions::new(
        node.config.stride,
        node.config.padding,
        node.config.padding_out,
        node.config.dilation,
        node.config.groups,
    );
    Ok(vec![Value::Tensor(DynTensor::R4(burn_conv_transpose2d(
        input, weight, bias, options,
    )))])
}

pub(super) fn conv_transpose3d(
    node: &ConvTranspose3dNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = rank5(resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?)?;
    let dtype = input.dtype();
    let weight = rank5(
        resolve::at(env, &node.name, &node.inputs, 1, device)?
            .into_tensor()?
            .cast(dtype),
    )?;
    let bias = bias(&node.name, &node.inputs, env, device, dtype)?;
    let options = ConvTransposeOptions::new(
        node.config.stride,
        node.config.padding,
        node.config.padding_out,
        node.config.dilation,
        node.config.groups,
    );
    Ok(vec![Value::Tensor(DynTensor::R5(burn_conv_transpose3d(
        input, weight, bias, options,
    )))])
}

fn bias(
    node_name: &str,
    inputs: &[Argument],
    env: &Env,
    device: &Device,
    dtype: DType,
) -> Result<Option<Tensor<1>>> {
    if !inputs.get(2).is_some_and(|input| !input.is_optional()) {
        return Ok(None);
    }
    match resolve::at(env, node_name, inputs, 2, device)?
        .into_tensor()?
        .cast(dtype)
    {
        DynTensor::R1(tensor) => Ok(Some(tensor)),
        tensor => Err(TynxError::Shape(format!(
            "convolution bias must have rank 1, got rank {}",
            tensor.rank()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::{
            conv_transpose1d::{ConvTranspose1dConfig, ConvTranspose1dNodeBuilder},
            conv_transpose2d::{ConvTranspose2dConfig, ConvTranspose2dNodeBuilder},
            conv_transpose3d::{ConvTranspose3dConfig, ConvTranspose3dNodeBuilder},
            conv2d::{Conv2dConfig, Conv2dNodeBuilder},
            padding::{AutoPad, PaddingConfig2d},
        },
    };

    use super::*;

    #[test]
    fn convolves_with_asymmetric_padding() {
        let node = Conv2dNodeBuilder::new("conv")
            .input_tensor("x", 4, DType::F32)
            .input_tensor("weight", 4, DType::F32)
            .output_tensor("y", 4, DType::F32)
            .config(Conv2dConfig::new(
                [2, 2],
                [1, 1],
                PaddingConfig2d::Explicit(1, 0, 0, 1),
                [1, 1],
                1,
                AutoPad::NotSet,
            ))
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
            "weight".into(),
            Value::from_tensor_data(TensorData::new(vec![1.0_f32; 4], [1, 1, 2, 2]), 4, &device)
                .unwrap(),
        );

        let output = conv2d(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [3.0, 2.0, 10.0, 6.0]);
    }

    #[test]
    fn transposed_convolution_expands_one_dimension() {
        let node = ConvTranspose1dNodeBuilder::new("conv_transpose")
            .input_tensor("x", 3, DType::F32)
            .input_tensor("weight", 3, DType::F32)
            .output_tensor("y", 3, DType::F32)
            .config(ConvTranspose1dConfig::new(2, 1, 1, 1, 0, 0))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(TensorData::new(vec![2.0_f32], [1, 1, 1]), 3, &device).unwrap(),
        );
        env.insert(
            "weight".into(),
            Value::from_tensor_data(TensorData::new(vec![1.0_f32; 2], [1, 1, 2]), 3, &device)
                .unwrap(),
        );

        let output = conv_transpose1d(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [2.0, 2.0]);
    }

    #[test]
    fn transposed_convolution_expands_two_dimensions() {
        let node = ConvTranspose2dNodeBuilder::new("conv_transpose")
            .input_tensor("x", 4, DType::F32)
            .input_tensor("weight", 4, DType::F32)
            .output_tensor("y", 4, DType::F32)
            .config(ConvTranspose2dConfig::new(
                [2, 2],
                [1, 1],
                [1, 1],
                [0, 0],
                [0, 0],
                1,
            ))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(TensorData::new(vec![2.0_f32], [1, 1, 1, 1]), 4, &device)
                .unwrap(),
        );
        env.insert(
            "weight".into(),
            Value::from_tensor_data(TensorData::new(vec![1.0_f32; 4], [1, 1, 2, 2]), 4, &device)
                .unwrap(),
        );

        let output = conv_transpose2d(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [2.0; 4]);
    }

    #[test]
    fn transposed_convolution_expands_three_dimensions() {
        let node = ConvTranspose3dNodeBuilder::new("conv_transpose")
            .input_tensor("x", 5, DType::F32)
            .input_tensor("weight", 5, DType::F32)
            .output_tensor("y", 5, DType::F32)
            .config(ConvTranspose3dConfig::new(
                [2, 2, 2],
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 0],
                [0, 0, 0],
                1,
            ))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(TensorData::new(vec![2.0_f32], [1, 1, 1, 1, 1]), 5, &device)
                .unwrap(),
        );
        env.insert(
            "weight".into(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32; 8], [1, 1, 2, 2, 2]),
                5,
                &device,
            )
            .unwrap(),
        );

        let output = conv_transpose3d(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [2.0; 8]);
    }
}
