//! ONNX multi-head attention execution.

use burn::tensor::{Device, TensorData};
use onnx_ir::node::attention::{AttentionNode, AttentionQkMatmulOutputMode as QkOutputMode};

use super::{Env, matrix, resolve};
use crate::{DynBool, DynTensor, Result, TynxError, Value};

pub(super) fn attention(node: &AttentionNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let query_input = resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?;
    let key_input = resolve::at(env, &node.name, &node.inputs, 1, device)?.into_tensor()?;
    let value_input = resolve::at(env, &node.name, &node.inputs, 2, device)?.into_tensor()?;
    let rank3 = query_input.rank() == 3;

    let (query, q_heads) = prepare_query(query_input, node.config.q_num_heads)?;
    let (mut key, kv_heads) = prepare_kv(key_input, node.config.kv_num_heads, "key")?;
    let (mut value, value_heads) = prepare_kv(value_input, node.config.kv_num_heads, "value")?;
    if kv_heads != value_heads {
        return Err(TynxError::Shape(format!(
            "Attention key and value head counts differ: {kv_heads} and {value_heads}"
        )));
    }

    if has_input(node, 4) {
        let past_key = resolve::at(env, &node.name, &node.inputs, 4, device)?.into_tensor()?;
        let past_value = resolve::at(env, &node.name, &node.inputs, 5, device)?.into_tensor()?;
        key = DynTensor::concat(vec![past_key, key], 2)?;
        value = DynTensor::concat(vec![past_value, value], 2)?;
    }
    let present_key = key.clone();
    let present_value = value.clone();

    key = repeat_heads(key, q_heads, kv_heads)?;
    value = repeat_heads(value, q_heads, kv_heads)?;

    let query_dims = query.dims();
    let key_dims = key.dims();
    let value_dims = value.dims();
    if query_dims[0] != key_dims[0]
        || key_dims[0] != value_dims[0]
        || query_dims[3] != key_dims[3]
        || key_dims[2] != value_dims[2]
    {
        return Err(TynxError::Shape(format!(
            "Attention has incompatible Q/K/V shapes: {query_dims:?}, {key_dims:?}, {value_dims:?}"
        )));
    }

    let q_length = query_dims[2];
    let kv_length = key_dims[2];
    let scale = node
        .config
        .scale
        .unwrap_or(1.0 / (query_dims[3] as f64).sqrt());
    let sqrt_scale = scale.sqrt();
    let transposed_key = key.permute(vec![0, 1, 3, 2])?;
    let mut scores = matrix::matmul_values(
        Value::Tensor(query.mul_scalar(sqrt_scale)),
        Value::Tensor(transposed_key.mul_scalar(sqrt_scale)),
        device,
    )?
    .into_tensor()?;
    let qk_matmul = scores.clone();

    if has_input(node, 3) {
        let mask = resolve::at(env, &node.name, &node.inputs, 3, device)?;
        scores = apply_attention_mask(scores, mask, kv_length, device)?;
    }
    if node.config.is_causal {
        let bias = causal_bias(q_length, kv_length, scores.dtype(), device)?;
        scores = scores.add_broadcast(bias)?;
    }
    if has_input(node, 6) {
        let lengths = resolve::at(env, &node.name, &node.inputs, 6, device)?.into_int()?;
        let bias = nonpadding_bias(lengths, query_dims[0], kv_length, scores.dtype(), device)?;
        scores = scores.add_broadcast(bias)?;
    }
    let qk_with_mask = scores.clone();

    if node.config.softcap > 0.0 {
        scores = scores
            .div_scalar(node.config.softcap)
            .tanh()
            .mul_scalar(node.config.softcap);
    }
    let qk_after_softcap = scores.clone();
    let probabilities = scores.softmax(3);
    let qk_after_softmax = probabilities.clone();
    let output = matrix::matmul_values(Value::Tensor(probabilities), Value::Tensor(value), device)?
        .into_tensor()?;
    let output = if rank3 {
        collapse_heads(output)?
    } else {
        output
    };

    let mut outputs = vec![Value::Tensor(output)];
    if node.outputs.len() >= 3 {
        outputs.push(Value::Tensor(present_key));
        outputs.push(Value::Tensor(present_value));
    }
    if node.outputs.len() >= 4 {
        let qk = match node.config.qk_matmul_output_mode {
            QkOutputMode::Matmul => qk_matmul,
            QkOutputMode::MatmulPlusAttentionMask => qk_with_mask,
            QkOutputMode::MatmulAfterSoftcap => qk_after_softcap,
            QkOutputMode::MatmulAfterSoftmax => qk_after_softmax,
        };
        outputs.push(Value::Tensor(qk));
    }

    Ok(outputs)
}

fn prepare_query(input: DynTensor, configured_heads: Option<usize>) -> Result<(DynTensor, usize)> {
    if input.rank() == 4 {
        let heads = input.dims()[1];
        return Ok((input, heads));
    }
    prepare_rank3(input, configured_heads, "query")
}

fn prepare_kv(
    input: DynTensor,
    configured_heads: Option<usize>,
    name: &str,
) -> Result<(DynTensor, usize)> {
    if input.rank() == 4 {
        let heads = input.dims()[1];
        return Ok((input, heads));
    }
    prepare_rank3(input, configured_heads, name)
}

fn prepare_rank3(
    input: DynTensor,
    configured_heads: Option<usize>,
    name: &str,
) -> Result<(DynTensor, usize)> {
    if input.rank() != 3 {
        return Err(TynxError::Shape(format!(
            "Attention {name} must have rank 3 or 4, got {}",
            input.rank()
        )));
    }
    let heads = configured_heads.ok_or_else(|| {
        TynxError::Shape(format!("Attention rank-3 {name} is missing its head count"))
    })?;
    let dims = input.dims();
    if heads == 0 || !dims[2].is_multiple_of(heads) {
        return Err(TynxError::Shape(format!(
            "Attention {name} hidden size {} is not divisible by {heads} heads",
            dims[2]
        )));
    }
    let head_size = dims[2] / heads;
    Ok((
        input
            .reshape(vec![dims[0], dims[1], heads, head_size])?
            .permute(vec![0, 2, 1, 3])?,
        heads,
    ))
}

fn repeat_heads(input: DynTensor, q_heads: usize, kv_heads: usize) -> Result<DynTensor> {
    if q_heads == kv_heads {
        return Ok(input);
    }
    if kv_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
        return Err(TynxError::Shape(format!(
            "Attention query heads {q_heads} are not divisible by KV heads {kv_heads}"
        )));
    }
    let groups = q_heads / kv_heads;
    Ok(input.repeat(&[1, groups, 1, 1]))
}

fn collapse_heads(input: DynTensor) -> Result<DynTensor> {
    let dims = input.dims();
    input
        .permute(vec![0, 2, 1, 3])?
        .reshape(vec![dims[0], dims[2], dims[1] * dims[3]])
}

fn apply_attention_mask(
    scores: DynTensor,
    mask: Value,
    kv_length: usize,
    device: &Device,
) -> Result<DynTensor> {
    let dtype = scores.dtype();
    match mask {
        Value::Tensor(mask) => {
            let mask = normalize_float_mask(mask, kv_length, device)?;
            scores.add_broadcast(mask.cast(dtype))
        }
        Value::Bool(mask) => {
            let mask = normalize_bool_mask(mask, kv_length, device)?;
            let blocked = DynTensor::full(&[1, 1, 1, 1], f64::NEG_INFINITY, device, dtype)?;
            DynTensor::where_select(mask, scores, blocked)
        }
        Value::Int(mask) => {
            let mask = normalize_bool_mask(mask.to_bool(), kv_length, device)?;
            let blocked = DynTensor::full(&[1, 1, 1, 1], f64::NEG_INFINITY, device, dtype)?;
            DynTensor::where_select(mask, scores, blocked)
        }
        other => Err(TynxError::TypeMismatch(format!(
            "Attention mask must be a tensor, got {other:?}"
        ))),
    }
}

fn normalize_float_mask(mask: DynTensor, kv_length: usize, device: &Device) -> Result<DynTensor> {
    let dtype = mask.dtype();
    let mask = normalize_mask_rank(mask)?;
    let dims = mask.dims();
    if dims[3] > kv_length {
        return Err(TynxError::Shape(format!(
            "Attention mask length {} exceeds key length {kv_length}",
            dims[3]
        )));
    }
    if dims[3] == kv_length {
        return Ok(mask);
    }
    let padding = DynTensor::full(
        &[dims[0], dims[1], dims[2], kv_length - dims[3]],
        f64::NEG_INFINITY,
        device,
        dtype,
    )?;
    DynTensor::concat(vec![mask, padding], 3)
}

fn normalize_bool_mask(mask: DynBool, kv_length: usize, device: &Device) -> Result<DynBool> {
    let mask = normalize_mask_rank(mask)?;
    let dims = mask.dims();
    if dims[3] > kv_length {
        return Err(TynxError::Shape(format!(
            "Attention mask length {} exceeds key length {kv_length}",
            dims[3]
        )));
    }
    if dims[3] == kv_length {
        return Ok(mask);
    }
    let padding = DynBool::full(
        &[dims[0], dims[1], dims[2], kv_length - dims[3]],
        false,
        device,
    )?;
    DynBool::concat(vec![mask, padding], 3)
}

trait AttentionMask: Sized {
    fn rank(&self) -> usize;
    fn dims(&self) -> Vec<usize>;
    fn reshape(self, dims: Vec<usize>) -> Result<Self>;
}

impl AttentionMask for DynTensor {
    fn rank(&self) -> usize {
        DynTensor::rank(self)
    }
    fn dims(&self) -> Vec<usize> {
        DynTensor::dims(self)
    }
    fn reshape(self, dims: Vec<usize>) -> Result<Self> {
        DynTensor::reshape(self, dims)
    }
}

impl AttentionMask for DynBool {
    fn rank(&self) -> usize {
        DynBool::rank(self)
    }
    fn dims(&self) -> Vec<usize> {
        DynBool::dims(self)
    }
    fn reshape(self, dims: Vec<usize>) -> Result<Self> {
        DynBool::reshape(self, dims)
    }
}

fn normalize_mask_rank<T: AttentionMask>(mask: T) -> Result<T> {
    let dims = mask.dims();
    match mask.rank() {
        2 => mask.reshape(vec![1, 1, dims[0], dims[1]]),
        3 => mask.reshape(vec![dims[0], 1, dims[1], dims[2]]),
        4 => Ok(mask),
        rank => Err(TynxError::Shape(format!(
            "Attention mask must have rank 2, 3, or 4, got {rank}"
        ))),
    }
}

fn causal_bias(
    q_length: usize,
    kv_length: usize,
    dtype: burn::tensor::DType,
    device: &Device,
) -> Result<DynTensor> {
    let mut values = Vec::with_capacity(q_length * kv_length);
    for row in 0..q_length {
        for column in 0..kv_length {
            values.push(if column <= row {
                0.0_f32
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    Ok(DynTensor::from_data(
        TensorData::new(values, [1, 1, q_length, kv_length]),
        4,
        device,
    )?
    .cast(dtype))
}

fn nonpadding_bias(
    lengths: crate::DynInt,
    batch_size: usize,
    kv_length: usize,
    dtype: burn::tensor::DType,
    device: &Device,
) -> Result<DynTensor> {
    let lengths = lengths.into_data().iter::<i64>().collect::<Vec<_>>();
    if lengths.len() != batch_size {
        return Err(TynxError::Shape(format!(
            "Attention nonpad_kv_seqlen has {} values for batch size {batch_size}",
            lengths.len()
        )));
    }
    let mut values = Vec::with_capacity(batch_size * kv_length);
    for length in lengths {
        let length = usize::try_from(length).map_err(|_| {
            TynxError::Shape(format!("Attention non-padding length {length} is negative"))
        })?;
        if length > kv_length {
            return Err(TynxError::Shape(format!(
                "Attention non-padding length {length} exceeds key length {kv_length}"
            )));
        }
        for column in 0..kv_length {
            values.push(if column < length {
                0.0_f32
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    Ok(DynTensor::from_data(
        TensorData::new(values, [batch_size, 1, 1, kv_length]),
        4,
        device,
    )?
    .cast(dtype))
}

fn has_input(node: &AttentionNode, index: usize) -> bool {
    node.inputs
        .get(index)
        .is_some_and(|input| !input.is_optional())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_query_heads_repeat_in_onnx_order() {
        let device = Device::default();
        let input = DynTensor::from_data(
            TensorData::new(vec![10.0_f32, 20.0], [1, 2, 1, 1]),
            4,
            &device,
        )
        .unwrap();

        let output = repeat_heads(input, 4, 2)
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [10.0, 20.0, 10.0, 20.0]);
    }

    #[test]
    fn causal_attention_uses_upper_left_alignment() {
        let device = Device::default();
        let output = causal_bias(2, 3, burn::tensor::DType::F32, &device)
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output[0], 0.0);
        assert!(output[1].is_infinite() && output[1].is_sign_negative());
        assert!(output[2].is_infinite() && output[2].is_sign_negative());
        assert_eq!(&output[3..], &[0.0, 0.0, f32::NEG_INFINITY]);
    }
}
