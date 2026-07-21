//! ONNX shape-changing operator execution.

use std::collections::HashSet;

use burn::tensor::{Device, TensorData};
use onnx_ir::node::{
    flatten::FlattenNode,
    reshape::{ReshapeInput, ReshapeNode},
    squeeze::{SqueezeInput, SqueezeNode},
    transpose::TransposeNode,
    unsqueeze::{UnsqueezeConfig, UnsqueezeNode},
};

use super::{Env, resolve};
use crate::{Result, Scalar, TynxError, Value};

pub(super) fn reshape(node: &ReshapeNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let input_dims = value_dims(&input);
    let requested = match &node.config.shape {
        ReshapeInput::Static(shape) => shape.clone(),
        ReshapeInput::Runtime(reference) => value_to_i64s(resolve::at(
            env,
            &node.name,
            &node.inputs,
            reference.input_index,
            device,
        )?)?,
    };
    let dims = reshape_dims(&input_dims, &requested)?;

    Ok(vec![reshape_value(input, dims, device)?])
}

pub(super) fn flatten(node: &FlattenNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let dims = value_dims(&input);
    if node.config.axis > dims.len() {
        return Err(TynxError::Shape(format!(
            "Flatten axis {} exceeds rank {}",
            node.config.axis,
            dims.len()
        )));
    }
    let left = checked_product(&dims[..node.config.axis])?;
    let right = checked_product(&dims[node.config.axis..])?;

    Ok(vec![reshape_value(input, vec![left, right], device)?])
}

pub(super) fn transpose(node: &TransposeNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let rank = value_dims(&input).len();
    let axes = permutation(&node.config.perm, rank)?;
    let output = match input {
        Value::Tensor(tensor) => Value::Tensor(tensor.permute(axes)?),
        Value::Int(tensor) => Value::Int(tensor.permute(axes)?),
        Value::Bool(tensor) => Value::Bool(tensor.permute(axes)?),
        Value::Scalar(value) if axes.is_empty() => Value::Scalar(value),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "Transpose expects a tensor, got {other:?}"
            )));
        }
    };

    Ok(vec![output])
}

pub(super) fn squeeze(node: &SqueezeNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let input_dims = value_dims(&input);
    let axes = match &node.config.axes {
        None => input_dims
            .iter()
            .enumerate()
            .filter_map(|(axis, &dim)| (dim == 1).then_some(axis))
            .collect(),
        Some(SqueezeInput::Static(axes)) => normalize_axes(axes, input_dims.len())?,
        Some(SqueezeInput::Runtime(reference)) => normalize_axes(
            &value_to_i64s(resolve::at(
                env,
                &node.name,
                &node.inputs,
                reference.input_index,
                device,
            )?)?,
            input_dims.len(),
        )?,
    };
    for &axis in &axes {
        if input_dims[axis] != 1 {
            return Err(TynxError::Shape(format!(
                "Squeeze axis {axis} has dimension {}, expected 1",
                input_dims[axis]
            )));
        }
    }
    let dims = input_dims
        .into_iter()
        .enumerate()
        .filter_map(|(axis, dim)| (!axes.contains(&axis)).then_some(dim))
        .collect();

    Ok(vec![reshape_value(input, dims, device)?])
}

pub(super) fn unsqueeze(node: &UnsqueezeNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let mut dims = value_dims(&input);
    let raw_axes = match &node.config {
        UnsqueezeConfig::Static(axes) => axes.clone(),
        UnsqueezeConfig::Runtime(reference) => value_to_i64s(resolve::at(
            env,
            &node.name,
            &node.inputs,
            reference.input_index,
            device,
        )?)?,
    };
    let output_rank = dims
        .len()
        .checked_add(raw_axes.len())
        .ok_or_else(|| TynxError::Shape("Unsqueeze output rank overflowed usize".to_string()))?;
    let axes = normalize_axes(&raw_axes, output_rank)?;
    for axis in axes {
        dims.insert(axis, 1);
    }

    Ok(vec![reshape_value(input, dims, device)?])
}

fn reshape_value(value: Value, dims: Vec<usize>, device: &Device) -> Result<Value> {
    let old_count = checked_product(&value_dims(&value))?;
    let new_count = checked_product(&dims)?;
    if old_count != new_count {
        return Err(TynxError::Shape(format!(
            "cannot reshape {old_count} elements into {dims:?} ({new_count} elements)"
        )));
    }

    if dims.is_empty() {
        return match value {
            Value::Tensor(tensor) => Value::from_tensor_data(tensor.into_data(), 0, device),
            Value::Int(tensor) => Value::from_tensor_data(tensor.into_data(), 0, device),
            Value::Bool(tensor) => Value::from_tensor_data(tensor.into_data(), 0, device),
            Value::Scalar(scalar) => Ok(Value::Scalar(scalar)),
            other => Err(TynxError::TypeMismatch(format!(
                "shape operation expects a tensor, got {other:?}"
            ))),
        };
    }

    if dims.contains(&0) {
        let data = match value {
            Value::Tensor(tensor) => tensor.into_data(),
            Value::Int(tensor) => tensor.into_data(),
            Value::Bool(tensor) => tensor.into_data(),
            other => {
                return Err(TynxError::TypeMismatch(format!(
                    "empty shape operation expects a tensor, got {other:?}"
                )));
            }
        };
        let data = TensorData::from_bytes(data.bytes, dims.clone(), data.dtype);
        return Value::from_tensor_data(data, dims.len(), device);
    }

    match value {
        Value::Tensor(tensor) => Ok(Value::Tensor(tensor.reshape(dims)?)),
        Value::Int(tensor) => Ok(Value::Int(tensor.reshape(dims)?)),
        Value::Bool(tensor) => Ok(Value::Bool(tensor.reshape(dims)?)),
        Value::Scalar(scalar) => {
            Value::from_tensor_data(scalar_data(scalar, &dims), dims.len(), device)
        }
        other => Err(TynxError::TypeMismatch(format!(
            "shape operation expects a tensor, got {other:?}"
        ))),
    }
}

fn scalar_data(scalar: Scalar, dims: &[usize]) -> TensorData {
    match scalar {
        Scalar::F64(value) => TensorData::new(vec![value], dims),
        Scalar::I64(value) => TensorData::new(vec![value], dims),
        Scalar::U64(value) => TensorData::new(vec![value], dims),
        Scalar::Bool(value) => TensorData::new(vec![value], dims),
    }
}

fn value_dims(value: &Value) -> Vec<usize> {
    match value {
        Value::Tensor(tensor) => tensor.dims(),
        Value::Int(tensor) => tensor.dims(),
        Value::Bool(tensor) => tensor.dims(),
        Value::Scalar(_) => Vec::new(),
        Value::Shape(shape) => vec![shape.len()],
    }
}

fn value_to_i64s(value: Value) -> Result<Vec<i64>> {
    match value {
        Value::Shape(values) => Ok(values),
        Value::Int(tensor) => Ok(tensor.into_data().iter::<i64>().collect()),
        Value::Scalar(Scalar::I64(value)) => Ok(vec![value]),
        Value::Scalar(Scalar::U64(value)) => i64::try_from(value)
            .map(|value| vec![value])
            .map_err(|_| TynxError::Shape(format!("shape value {value} exceeds i64"))),
        other => Err(TynxError::TypeMismatch(format!(
            "shape or axes must be integer values, got {other:?}"
        ))),
    }
}

fn reshape_dims(input: &[usize], requested: &[i64]) -> Result<Vec<usize>> {
    let input_count = checked_product(input)?;
    let mut inferred = None;
    let mut dims = Vec::with_capacity(requested.len());
    for (axis, &dim) in requested.iter().enumerate() {
        match dim {
            -1 if inferred.replace(axis).is_none() => dims.push(1),
            -1 => {
                return Err(TynxError::Shape(
                    "Reshape has more than one -1 dimension".into(),
                ));
            }
            0 => dims.push(*input.get(axis).ok_or_else(|| {
                TynxError::Shape(format!(
                    "Reshape dimension {axis} cannot copy absent input axis"
                ))
            })?),
            dim if dim > 0 => {
                dims.push(usize::try_from(dim).map_err(|_| {
                    TynxError::Shape(format!("Reshape dimension {dim} exceeds usize"))
                })?)
            }
            dim => return Err(TynxError::Shape(format!("invalid Reshape dimension {dim}"))),
        }
    }

    let known = checked_product(&dims)?;
    if let Some(axis) = inferred {
        if known == 0 || !input_count.is_multiple_of(known) {
            return Err(TynxError::Shape(format!(
                "cannot infer Reshape dimension from {input_count} elements and known product {known}"
            )));
        }
        dims[axis] = input_count / known;
    }
    if checked_product(&dims)? != input_count
        && input_count == 0
        && inferred.is_none()
        && requested.contains(&0)
    {
        for (axis, &dim) in requested.iter().enumerate() {
            if dim == 0 {
                dims[axis] = 0;
            }
        }
    }
    if checked_product(&dims)? != input_count {
        return Err(TynxError::Shape(format!(
            "Reshape target {dims:?} does not contain {input_count} elements"
        )));
    }
    Ok(dims)
}

fn normalize_axes(raw_axes: &[i64], rank: usize) -> Result<Vec<usize>> {
    let rank_i64 =
        i64::try_from(rank).map_err(|_| TynxError::Shape(format!("rank {rank} exceeds i64")))?;
    let mut axes = Vec::with_capacity(raw_axes.len());
    let mut seen = HashSet::with_capacity(raw_axes.len());
    for &raw in raw_axes {
        let axis = if raw < 0 { raw + rank_i64 } else { raw };
        if !(0..rank_i64).contains(&axis) {
            return Err(TynxError::Shape(format!(
                "axis {raw} is out of range for rank {rank}"
            )));
        }
        let axis = axis as usize;
        if !seen.insert(axis) {
            return Err(TynxError::Shape(format!("duplicate axis {raw}")));
        }
        axes.push(axis);
    }
    axes.sort_unstable();
    Ok(axes)
}

fn permutation(raw_axes: &[i64], rank: usize) -> Result<Vec<usize>> {
    if raw_axes.len() != rank {
        return Err(TynxError::Shape(format!(
            "Transpose permutation has {} axes for rank {rank}",
            raw_axes.len()
        )));
    }
    normalize_axes(raw_axes, rank).and_then(|_| {
        raw_axes
            .iter()
            .map(|&axis| {
                usize::try_from(axis)
                    .map_err(|_| TynxError::Shape(format!("invalid Transpose axis {axis}")))
            })
            .collect()
    })
}

fn checked_product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1_usize, |count, &dim| {
        count
            .checked_mul(dim)
            .ok_or_else(|| TynxError::Shape(format!("shape product overflow for {dims:?}")))
    })
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::flatten::{FlattenConfig, FlattenNodeBuilder},
    };

    use super::*;

    #[test]
    fn flattens_around_the_configured_axis() {
        let node = FlattenNodeBuilder::new("flatten")
            .input_tensor("x", 3, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .config(FlattenConfig { axis: 1 })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(
                    (0..24).map(|value| value as f32).collect::<Vec<_>>(),
                    [2, 3, 4],
                ),
                3,
                &device,
            )
            .unwrap(),
        );

        let output = flatten(&node, &env, &device).unwrap().pop().unwrap();

        assert_eq!(value_dims(&output), [2, 12]);
    }

    #[test]
    fn resolves_reshape_copy_and_inferred_dimensions() {
        assert_eq!(reshape_dims(&[2, 3, 4], &[0, -1]).unwrap(), [2, 12]);
    }

    #[test]
    fn normalizes_negative_axes_and_rejects_duplicates() {
        assert_eq!(normalize_axes(&[-1, 0], 3).unwrap(), [0, 2]);
        assert!(normalize_axes(&[1, -2], 3).is_err());
    }
}
