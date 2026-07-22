//! ONNX pooling execution.

use std::collections::{HashMap, VecDeque};

use burn::tensor::{
    DType, Device, Int, Tensor,
    module::{
        conv1d as burn_conv1d, conv2d as burn_conv2d, conv3d as burn_conv3d,
        max_pool1d as burn_max_pool1d, max_pool1d_with_indices, max_pool2d as burn_max_pool2d,
        max_pool2d_with_indices,
    },
    ops::{ConvOptions, PadMode},
};
use onnx_ir::{
    ModelProto, Node,
    ir::{Argument, OnnxGraph},
    node::{
        avg_pool1d::AveragePool1dNode, avg_pool2d::AveragePool2dNode,
        avg_pool3d::AveragePool3dNode, global_avg_pool::GlobalAveragePoolNode,
        lp_pool1d::LpPool1dNode, lp_pool2d::LpPool2dNode, max_pool1d::MaxPool1dNode,
        max_pool2d::MaxPool2dNode, max_pool3d::MaxPool3dNode, unsupported::GlobalMaxPoolNode,
    },
};
use protobuf::Message;

use super::{
    Env, resolve,
    spatial::{padding1d, padding2d, padding3d, rank3, rank4, rank5},
};
use crate::{DynInt, DynTensor, Result, TynxError, Value};

const COLUMN_MAJOR_PREFIX: &str = "__tynx_maxpool_column_major__";

pub(super) fn preserve_attributes(data: &[u8], graph: &mut OnnxGraph) -> Result<()> {
    let model =
        ModelProto::parse_from_bytes(data).map_err(|error| TynxError::Parse(error.to_string()))?;
    let Some(raw_graph) = model.graph.as_ref() else {
        return Ok(());
    };
    let mut flags = HashMap::<String, VecDeque<bool>>::new();
    for node in &raw_graph.node {
        if node.op_type != "MaxPool" {
            continue;
        }
        let Some(input) = node.input.first() else {
            continue;
        };
        let column_major = node
            .attribute
            .iter()
            .any(|attribute| attribute.name == "storage_order" && attribute.i == 1);
        flags
            .entry(input.clone())
            .or_default()
            .push_back(column_major);
    }

    for node in &mut graph.nodes {
        match node {
            Node::MaxPool1d(node) => mark_storage_order(node, &mut flags),
            Node::MaxPool2d(node) => mark_storage_order(node, &mut flags),
            Node::MaxPool3d(node) => mark_storage_order(node, &mut flags),
            _ => {}
        }
    }
    Ok(())
}

fn mark_storage_order<N: PoolNode>(node: &mut N, flags: &mut HashMap<String, VecDeque<bool>>) {
    let Some(input) = node.inputs().first() else {
        return;
    };
    if flags.get_mut(&input.name).and_then(VecDeque::pop_front) == Some(true) {
        *node.name_mut() = format!("{COLUMN_MAJOR_PREFIX}{}", node.name());
    }
}

trait PoolNode {
    fn name(&self) -> &str;
    fn name_mut(&mut self) -> &mut String;
    fn inputs(&self) -> &[Argument];
}

macro_rules! impl_pool_node {
    ($($type:ty),+ $(,)?) => {
        $(
            impl PoolNode for $type {
                fn name(&self) -> &str { &self.name }
                fn name_mut(&mut self) -> &mut String { &mut self.name }
                fn inputs(&self) -> &[Argument] { &self.inputs }
            }
        )+
    };
}

impl_pool_node!(MaxPool1dNode, MaxPool2dNode, MaxPool3dNode);

pub(super) fn average_pool1d(
    node: &AveragePool1dNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = rank3(resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?)?;
    let dims = input.dims();
    let padding = padding1d(
        dims[2],
        node.config.kernel_size,
        node.config.stride,
        node.config.dilation,
        &node.config.padding,
        &node.config.auto_pad,
        node.config.ceil_mode,
    );
    let output = grouped_average_pool1d(
        input,
        node.config.kernel_size,
        node.config.stride,
        node.config.dilation,
        padding,
        padding1d(
            dims[2],
            node.config.kernel_size,
            node.config.stride,
            node.config.dilation,
            &node.config.padding,
            &node.config.auto_pad,
            false,
        ),
        node.config.count_include_pad,
    );
    Ok(vec![Value::Tensor(DynTensor::R3(output))])
}

pub(super) fn average_pool2d(
    node: &AveragePool2dNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = rank4(resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?)?;
    let dims = input.dims();
    let padding = padding2d(
        [dims[2], dims[3]],
        node.config.kernel_size,
        node.config.strides,
        node.config.dilation,
        &node.config.padding,
        &node.config.auto_pad,
        node.config.ceil_mode,
    );
    let output = grouped_average_pool2d(
        input,
        node.config.kernel_size,
        node.config.strides,
        node.config.dilation,
        padding,
        padding2d(
            [dims[2], dims[3]],
            node.config.kernel_size,
            node.config.strides,
            node.config.dilation,
            &node.config.padding,
            &node.config.auto_pad,
            false,
        ),
        node.config.count_include_pad,
    );
    Ok(vec![Value::Tensor(DynTensor::R4(output))])
}

pub(super) fn average_pool3d(
    node: &AveragePool3dNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = rank5(resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?)?;
    let dims = input.dims();
    let padding = padding3d(
        [dims[2], dims[3], dims[4]],
        node.config.kernel_size,
        node.config.strides,
        node.config.dilation,
        &node.config.padding,
        &node.config.auto_pad,
        node.config.ceil_mode,
    );
    let output = grouped_average_pool3d(
        input,
        node.config.kernel_size,
        node.config.strides,
        node.config.dilation,
        padding,
        padding3d(
            [dims[2], dims[3], dims[4]],
            node.config.kernel_size,
            node.config.strides,
            node.config.dilation,
            &node.config.padding,
            &node.config.auto_pad,
            false,
        ),
        node.config.count_include_pad,
    );
    Ok(vec![Value::Tensor(DynTensor::R5(output))])
}

pub(super) fn max_pool1d(node: &MaxPool1dNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let (input, integer_dtype) = float_for_max_pool(input)?;
    let input = rank3(input)?;
    let dims = input.dims();
    if node.outputs.len() > 1 {
        let [(left, right)] = padding1d(
            dims[2],
            node.config.kernel_size,
            node.config.stride,
            node.config.dilation,
            &node.config.padding,
            &node.config.auto_pad,
            false,
        );
        if left != right {
            return Err(TynxError::UnsupportedOp(
                "MaxPool indices with asymmetric padding".to_string(),
            ));
        }
        let (output, indices) = max_pool1d_with_indices(
            input,
            node.config.kernel_size,
            node.config.stride,
            left,
            node.config.dilation,
            node.config.ceil_mode,
        );
        let [batch, channels, length] = dims;
        let indices_device = indices.device();
        let offsets =
            Tensor::<1, Int>::arange(0..(batch * channels) as i64, (&indices_device, DType::I64))
                .reshape([batch, channels, 1])
                .mul_scalar(length as i64);
        let indices = indices.cast(DType::I64).add(offsets);
        return Ok(vec![
            restore_max_pool(DynTensor::R3(output), integer_dtype),
            Value::Int(DynInt::R3(indices.cast(DType::I64))),
        ]);
    }
    let padding = padding1d(
        dims[2],
        node.config.kernel_size,
        node.config.stride,
        node.config.dilation,
        &node.config.padding,
        &node.config.auto_pad,
        node.config.ceil_mode,
    );
    let output = burn_max_pool1d(
        input.pad(padding, PadMode::Constant(f32::NEG_INFINITY)),
        node.config.kernel_size,
        node.config.stride,
        0,
        node.config.dilation,
        false,
    );
    Ok(vec![restore_max_pool(DynTensor::R3(output), integer_dtype)])
}

pub(super) fn max_pool2d(node: &MaxPool2dNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let (input, integer_dtype) = float_for_max_pool(input)?;
    let input = rank4(input)?;
    let dims = input.dims();
    if node.outputs.len() > 1 {
        let [(top, bottom), (left, right)] = padding2d(
            [dims[2], dims[3]],
            node.config.kernel_size,
            node.config.strides,
            node.config.dilation,
            &node.config.padding,
            &node.config.auto_pad,
            false,
        );
        if top != bottom || left != right {
            return Err(TynxError::UnsupportedOp(
                "MaxPool indices with asymmetric padding".to_string(),
            ));
        }
        let (output, indices) = max_pool2d_with_indices(
            input,
            node.config.kernel_size,
            node.config.strides,
            [top, left],
            node.config.dilation,
            node.config.ceil_mode,
        );
        let [batch, channels, height, width] = dims;
        let indices_device = indices.device();
        let offsets =
            Tensor::<1, Int>::arange(0..(batch * channels) as i64, (&indices_device, DType::I64))
                .reshape([batch, channels, 1, 1])
                .mul_scalar((height * width) as i64);
        let indices = indices.cast(DType::I64);
        let indices = if node.name.starts_with(COLUMN_MAJOR_PREFIX) {
            let row = indices.clone().div_scalar(width as i64);
            let column = indices.remainder_scalar(width as i64);
            column.mul_scalar(height as i64).add(row)
        } else {
            indices
        };
        let indices = indices.add(offsets);
        return Ok(vec![
            restore_max_pool(DynTensor::R4(output), integer_dtype),
            Value::Int(DynInt::R4(indices.cast(DType::I64))),
        ]);
    }
    let padding = padding2d(
        [dims[2], dims[3]],
        node.config.kernel_size,
        node.config.strides,
        node.config.dilation,
        &node.config.padding,
        &node.config.auto_pad,
        node.config.ceil_mode,
    );
    let output = burn_max_pool2d(
        input.pad(padding, PadMode::Constant(f32::NEG_INFINITY)),
        node.config.kernel_size,
        node.config.strides,
        [0; 2],
        node.config.dilation,
        false,
    );
    Ok(vec![restore_max_pool(DynTensor::R4(output), integer_dtype)])
}

pub(super) fn max_pool3d(node: &MaxPool3dNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    reject_indices(&node.outputs)?;
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let (input, integer_dtype) = float_for_max_pool(input)?;
    let input = rank5(input)?;
    let dims = input.dims();
    let padding = padding3d(
        [dims[2], dims[3], dims[4]],
        node.config.kernel_size,
        node.config.strides,
        node.config.dilation,
        &node.config.padding,
        &node.config.auto_pad,
        node.config.ceil_mode,
    );
    let input = input.pad(padding, PadMode::Constant(f32::NEG_INFINITY));
    let [batch, channels, depth, height, width] = input.dims();
    let spatial = input.reshape([batch * channels * depth, 1, height, width]);
    let spatial = burn_max_pool2d(
        spatial,
        [node.config.kernel_size[1], node.config.kernel_size[2]],
        [node.config.strides[1], node.config.strides[2]],
        [0; 2],
        [node.config.dilation[1], node.config.dilation[2]],
        false,
    );
    let [_, _, out_height, out_width] = spatial.dims();
    let depth_lines = spatial
        .reshape([batch, channels, depth, out_height, out_width])
        .permute([0, 1, 3, 4, 2])
        .reshape([batch * channels * out_height * out_width, 1, depth]);
    let depth_lines = burn_max_pool1d(
        depth_lines,
        node.config.kernel_size[0],
        node.config.strides[0],
        0,
        node.config.dilation[0],
        false,
    );
    let out_depth = depth_lines.dims()[2];
    let output = depth_lines
        .reshape([batch, channels, out_height, out_width, out_depth])
        .permute([0, 1, 4, 2, 3]);
    Ok(vec![restore_max_pool(DynTensor::R5(output), integer_dtype)])
}

pub(super) fn global_average_pool(
    node: &GlobalAveragePoolNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    let axes = (2..input.rank()).collect::<Vec<_>>();
    Ok(vec![Value::Tensor(input.mean_dims(&axes))])
}

pub(super) fn global_max_pool(
    node: &GlobalMaxPoolNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let output = match input {
        Value::Tensor(tensor) => {
            let axes = (2..tensor.rank()).collect::<Vec<_>>();
            Value::Tensor(tensor.reduce_max_dims(&axes))
        }
        Value::Int(tensor) => {
            let axes = (2..tensor.rank()).collect::<Vec<_>>();
            Value::Int(tensor.reduce_max_dims(&axes))
        }
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "GlobalMaxPool requires a numeric tensor, got {other:?}"
            )));
        }
    };
    Ok(vec![output])
}

pub(super) fn lp_pool1d(node: &LpPool1dNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = rank3(resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?)?;
    let dims = input.dims();
    let padding = padding1d(
        dims[2],
        node.config.kernel_size,
        node.config.stride,
        node.config.dilation,
        &node.config.padding,
        &node.config.auto_pad,
        node.config.ceil_mode,
    );
    let p = node.config.p as f64;
    let sum = grouped_sum_pool1d(
        input.abs().powf_scalar(p),
        node.config.kernel_size,
        node.config.stride,
        node.config.dilation,
        padding,
    );
    Ok(vec![Value::Tensor(DynTensor::R3(sum.powf_scalar(1.0 / p)))])
}

pub(super) fn lp_pool2d(node: &LpPool2dNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = rank4(resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?)?;
    let dims = input.dims();
    let padding = padding2d(
        [dims[2], dims[3]],
        node.config.kernel_size,
        node.config.strides,
        node.config.dilation,
        &node.config.padding,
        &node.config.auto_pad,
        node.config.ceil_mode,
    );
    let p = node.config.p as f64;
    let sum = grouped_sum_pool2d(
        input.abs().powf_scalar(p),
        node.config.kernel_size,
        node.config.strides,
        node.config.dilation,
        padding,
    );
    Ok(vec![Value::Tensor(DynTensor::R4(sum.powf_scalar(1.0 / p)))])
}

fn grouped_sum_pool1d(
    input: Tensor<3>,
    kernel: usize,
    stride: usize,
    dilation: usize,
    padding: [(usize, usize); 1],
) -> Tensor<3> {
    let [_, channels, _] = input.dims();
    let device = input.device();
    let dtype = input.dtype();
    burn_conv1d(
        input.pad(padding, PadMode::Constant(0.0)),
        Tensor::<3>::ones([channels, 1, kernel], (&device, dtype)),
        None,
        ConvOptions::new([stride], [0], [dilation], channels),
    )
}

fn grouped_sum_pool2d(
    input: Tensor<4>,
    kernel: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    padding: [(usize, usize); 2],
) -> Tensor<4> {
    let [_, channels, _, _] = input.dims();
    let device = input.device();
    let dtype = input.dtype();
    burn_conv2d(
        input.pad(padding, PadMode::Constant(0.0)),
        Tensor::<4>::ones([channels, 1, kernel[0], kernel[1]], (&device, dtype)),
        None,
        ConvOptions::new(stride, [0; 2], dilation, channels),
    )
}

fn grouped_average_pool1d(
    input: Tensor<3>,
    kernel: usize,
    stride: usize,
    dilation: usize,
    padding: [(usize, usize); 1],
    included_padding: [(usize, usize); 1],
    count_include_pad: bool,
) -> Tensor<3> {
    let [_, channels, _] = input.dims();
    let device = input.device();
    let dtype = input.dtype();
    let mask = average_pool_mask(
        input.ones_like(),
        padding,
        included_padding,
        count_include_pad,
    );
    let weights = Tensor::<3>::ones([channels, 1, kernel], (&device, dtype));
    let options = ConvOptions::new([stride], [0], [dilation], channels);
    let sum = burn_conv1d(
        input.pad(padding, PadMode::Constant(0.0)),
        weights.clone(),
        None,
        options.clone(),
    );
    if let Some(mask) = mask {
        let count = burn_conv1d(mask, weights, None, options);
        sum.div(count)
    } else {
        sum.div_scalar(kernel as f64)
    }
}

fn grouped_average_pool2d(
    input: Tensor<4>,
    kernel: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    padding: [(usize, usize); 2],
    included_padding: [(usize, usize); 2],
    count_include_pad: bool,
) -> Tensor<4> {
    let [_, channels, _, _] = input.dims();
    let device = input.device();
    let dtype = input.dtype();
    let mask = average_pool_mask(
        input.ones_like(),
        padding,
        included_padding,
        count_include_pad,
    );
    let weights = Tensor::<4>::ones([channels, 1, kernel[0], kernel[1]], (&device, dtype));
    let options = ConvOptions::new(stride, [0; 2], dilation, channels);
    let sum = burn_conv2d(
        input.pad(padding, PadMode::Constant(0.0)),
        weights.clone(),
        None,
        options.clone(),
    );
    if let Some(mask) = mask {
        let count = burn_conv2d(mask, weights, None, options);
        sum.div(count)
    } else {
        sum.div_scalar(kernel.iter().product::<usize>() as f64)
    }
}

fn grouped_average_pool3d(
    input: Tensor<5>,
    kernel: [usize; 3],
    stride: [usize; 3],
    dilation: [usize; 3],
    padding: [(usize, usize); 3],
    included_padding: [(usize, usize); 3],
    count_include_pad: bool,
) -> Tensor<5> {
    let [_, channels, _, _, _] = input.dims();
    let device = input.device();
    let dtype = input.dtype();
    let mask = average_pool_mask(
        input.ones_like(),
        padding,
        included_padding,
        count_include_pad,
    );
    let weights = Tensor::<5>::ones(
        [channels, 1, kernel[0], kernel[1], kernel[2]],
        (&device, dtype),
    );
    let options = ConvOptions::new(stride, [0; 3], dilation, channels);
    let sum = burn_conv3d(
        input.pad(padding, PadMode::Constant(0.0)),
        weights.clone(),
        None,
        options.clone(),
    );
    if let Some(mask) = mask {
        let count = burn_conv3d(mask, weights, None, options);
        sum.div(count)
    } else {
        sum.div_scalar(kernel.iter().product::<usize>() as f64)
    }
}

fn average_pool_mask<const D: usize, const N: usize>(
    ones: Tensor<D>,
    padding: [(usize, usize); N],
    included_padding: [(usize, usize); N],
    count_include_pad: bool,
) -> Option<Tensor<D>> {
    if !count_include_pad {
        return Some(ones.pad(padding, PadMode::Constant(0.0)));
    }
    if padding == included_padding {
        return None;
    }

    let ceil_padding: [(usize, usize); N] = core::array::from_fn(|axis| {
        (
            padding[axis].0.saturating_sub(included_padding[axis].0),
            padding[axis].1.saturating_sub(included_padding[axis].1),
        )
    });
    Some(
        ones.pad(included_padding, PadMode::Constant(1.0))
            .pad(ceil_padding, PadMode::Constant(0.0)),
    )
}

fn reject_indices(outputs: &[Argument]) -> Result<()> {
    if outputs.len() > 1 {
        Err(TynxError::UnsupportedOp(
            "MaxPool indices output".to_string(),
        ))
    } else {
        Ok(())
    }
}

fn float_for_max_pool(value: Value) -> Result<(DynTensor, Option<DType>)> {
    match value {
        Value::Tensor(tensor) => Ok((tensor, None)),
        Value::Int(tensor) => {
            let dtype = tensor.dtype();
            Ok((tensor.to_float(DType::F32), Some(dtype)))
        }
        other => Err(TynxError::TypeMismatch(format!(
            "MaxPool requires a numeric tensor, got {other:?}"
        ))),
    }
}

fn restore_max_pool(tensor: DynTensor, integer_dtype: Option<DType>) -> Value {
    match integer_dtype {
        Some(dtype) => Value::Int(tensor.to_int(dtype)),
        None => Value::Tensor(tensor),
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::{
            lp_pool1d::{LpPool1dConfig, LpPool1dNodeBuilder},
            max_pool3d::{MaxPool3dConfig, MaxPool3dNodeBuilder},
            padding::{AutoPad, PaddingConfig1d, PaddingConfig3d},
        },
    };

    use super::*;

    #[test]
    fn average_pool_supports_dilation() {
        let device = Device::default();
        let input = Tensor::<4>::from_data(
            TensorData::new(
                vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [1, 1, 3, 3],
            ),
            (&device, DType::F32),
        );

        let output = grouped_average_pool2d(
            input,
            [2, 2],
            [1, 1],
            [2, 2],
            [(0, 0), (0, 0)],
            [(0, 0), (0, 0)],
            false,
        )
        .into_data()
        .iter::<f32>()
        .collect::<Vec<_>>();

        assert_eq!(output, [5.0]);
    }

    #[test]
    fn lp_pool_sums_powered_windows() {
        let node = LpPool1dNodeBuilder::new("lp_pool")
            .input_tensor("x", 3, DType::F32)
            .output_tensor("y", 3, DType::F32)
            .config(LpPool1dConfig::new(
                2,
                2,
                PaddingConfig1d::Valid,
                1,
                false,
                AutoPad::NotSet,
                2,
            ))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 1, 4]),
                3,
                &device,
            )
            .unwrap(),
        );

        let output = lp_pool1d(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert!((output[0] - 5.0_f32.sqrt()).abs() < 1e-6);
        assert_eq!(output[1], 5.0);
    }

    #[test]
    fn max_pool3d_combines_all_spatial_dimensions() {
        let node = MaxPool3dNodeBuilder::new("pool")
            .input_tensor("x", 5, DType::F32)
            .output_tensor("y", 5, DType::F32)
            .config(MaxPool3dConfig::new(
                [2, 2, 2],
                [1, 1, 1],
                PaddingConfig3d::Valid,
                [1, 1, 1],
                false,
                AutoPad::NotSet,
            ))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(
                    vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    [1, 1, 2, 2, 2],
                ),
                5,
                &device,
            )
            .unwrap(),
        );

        let output = max_pool3d(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [8.0]);
    }
}
