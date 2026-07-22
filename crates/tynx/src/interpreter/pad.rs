//! ONNX Pad execution.

use std::collections::HashSet;

use burn::tensor::{Device, Slice};
use onnx_ir::node::pad::{AxesInput, ConstantValueInput, PadInput, PadMode, PadNode};

use super::{Env, resolve, shape};
use crate::{DynBool, DynInt, DynTensor, Result, Scalar, TynxError, Value};

pub(super) fn pad(node: &PadNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let dims = shape::value_dims(&input);
    let pads = resolve_pads(node, env, device, dims.len())?;
    if pads.iter().all(|pair| *pair == (0, 0)) {
        return Ok(vec![input]);
    }

    let output = match node.config.mode {
        PadMode::Constant => {
            let fill = resolve_constant(node, env, device)?;
            constant_pad(input, &dims, &pads, fill, device)?
        }
        PadMode::Reflect => nonconstant_pad(input, &pads, true)?,
        PadMode::Edge => nonconstant_pad(input, &pads, false)?,
    };
    Ok(vec![output])
}

fn resolve_pads(
    node: &PadNode,
    env: &Env,
    device: &Device,
    rank: usize,
) -> Result<Vec<(usize, usize)>> {
    match &node.config.pads {
        PadInput::Static(pads) => {
            if pads.len() != rank {
                return Err(TynxError::Shape(format!(
                    "Pad has {} dimension pairs for rank {rank}",
                    pads.len()
                )));
            }
            Ok(pads.clone())
        }
        PadInput::Runtime { input, axes } => {
            let raw = shape::value_to_i64s(resolve::at(
                env,
                &node.name,
                &node.inputs,
                input.input_index,
                device,
            )?)?;
            let axes = match axes {
                None => (0..rank).collect::<Vec<_>>(),
                Some(AxesInput::Static(axes)) => axes.clone(),
                Some(AxesInput::Runtime(reference)) => normalize_axes(
                    shape::value_to_i64s(resolve::at(
                        env,
                        &node.name,
                        &node.inputs,
                        reference.input_index,
                        device,
                    )?)?,
                    rank,
                )?,
            };
            if raw.len() != axes.len() * 2 {
                return Err(TynxError::Shape(format!(
                    "Pad values length {} does not equal twice the axes length {}",
                    raw.len(),
                    axes.len()
                )));
            }
            let values = raw
                .into_iter()
                .map(|value| {
                    usize::try_from(value).map_err(|_| {
                        TynxError::Shape(format!("negative Pad value {value} is unsupported"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let count = axes.len();
            let mut pads = vec![(0, 0); rank];
            for (index, axis) in axes.into_iter().enumerate() {
                pads[axis] = (values[index], values[count + index]);
            }
            Ok(pads)
        }
    }
}

fn normalize_axes(axes: Vec<i64>, rank: usize) -> Result<Vec<usize>> {
    let mut seen = HashSet::new();
    axes.into_iter()
        .map(|axis| {
            let normalized = if axis < 0 { axis + rank as i64 } else { axis };
            if !(0..rank as i64).contains(&normalized) {
                return Err(TynxError::Shape(format!(
                    "Pad axis {axis} is outside rank {rank}"
                )));
            }
            let normalized = normalized as usize;
            if !seen.insert(normalized) {
                return Err(TynxError::Shape(format!(
                    "Pad axis {normalized} is specified more than once"
                )));
            }
            Ok(normalized)
        })
        .collect()
}

fn resolve_constant(node: &PadNode, env: &Env, device: &Device) -> Result<Scalar> {
    match &node.config.constant_value {
        ConstantValueInput::Static(value) => Ok(Scalar::F64(*value as f64)),
        ConstantValueInput::Runtime(reference) => scalar_value(
            resolve::at(env, &node.name, &node.inputs, reference.input_index, device)?,
            device,
        ),
    }
}

fn scalar_value(value: Value, device: &Device) -> Result<Scalar> {
    match value {
        Value::Scalar(value) => Ok(value),
        Value::Tensor(tensor) => scalar_from_data(tensor.into_data(), device),
        Value::Int(tensor) => scalar_from_data(tensor.into_data(), device),
        Value::Bool(tensor) => scalar_from_data(tensor.into_data(), device),
        other => Err(TynxError::TypeMismatch(format!(
            "Pad constant must be scalar, got {other:?}"
        ))),
    }
}

fn scalar_from_data(data: burn::tensor::TensorData, device: &Device) -> Result<Scalar> {
    Value::from_tensor_data(data, 0, device)?.into_scalar()
}

fn constant_pad(
    input: Value,
    dims: &[usize],
    pads: &[(usize, usize)],
    fill: Scalar,
    device: &Device,
) -> Result<Value> {
    let output_dims = dims
        .iter()
        .zip(pads)
        .map(|(&dim, &(before, after))| {
            dim.checked_add(before)
                .and_then(|dim| dim.checked_add(after))
                .ok_or_else(|| TynxError::Shape("padded dimension overflowed usize".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let slices = dims
        .iter()
        .zip(pads)
        .map(|(&dim, &(before, _))| Slice::new(before as isize, Some((before + dim) as isize), 1))
        .collect::<Vec<_>>();

    match input {
        Value::Tensor(input) => {
            let dtype = input.dtype();
            Ok(Value::Tensor(
                DynTensor::full(&output_dims, fill.as_f64(), device, dtype)?
                    .slice_assign(&slices, input)?,
            ))
        }
        Value::Int(input) => {
            let dtype = input.dtype();
            Ok(Value::Int(
                DynInt::full(&output_dims, scalar_i64(fill)?, device, dtype)?
                    .slice_assign(&slices, input)?,
            ))
        }
        Value::Bool(input) => Ok(Value::Bool(
            DynBool::full(&output_dims, fill.as_f64() != 0.0, device)?
                .slice_assign(&slices, input)?,
        )),
        other => Err(TynxError::TypeMismatch(format!(
            "Pad expects a tensor, got {other:?}"
        ))),
    }
}

fn scalar_i64(value: Scalar) -> Result<i64> {
    match value {
        Scalar::I64(value) => Ok(value),
        Scalar::U64(value) => i64::try_from(value)
            .map_err(|_| TynxError::Shape(format!("Pad value {value} exceeds i64"))),
        Scalar::F64(value) => Ok(value as i64),
        Scalar::Bool(value) => Ok(i64::from(value)),
    }
}

fn nonconstant_pad(input: Value, pads: &[(usize, usize)], reflect: bool) -> Result<Value> {
    macro_rules! apply {
        ($tensor:expr, $kind:ident) => {{
            let mut tensor = $tensor;
            for (axis, &(before, after)) in pads.iter().enumerate() {
                if before == 0 && after == 0 {
                    continue;
                }
                let size = tensor.dims()[axis];
                if size == 0 {
                    return Err(TynxError::Shape(format!(
                        "cannot pad empty dimension {axis}"
                    )));
                }
                if reflect && (before >= size || after >= size) {
                    return Err(TynxError::Shape(format!(
                        "reflect padding ({before}, {after}) must be smaller than dimension {axis} size {size}"
                    )));
                }
                let mut parts = Vec::with_capacity(3);
                if before > 0 {
                    let mut slices = vec![Slice::full(); tensor.rank()];
                    let (start, end) = if reflect { (1, before + 1) } else { (0, 1) };
                    slices[axis] = Slice::new(start as isize, Some(end as isize), 1);
                    let mut edge = tensor.clone().slice(&slices);
                    if reflect {
                        edge = edge.flip_dim(axis);
                    } else {
                        let mut repeats = vec![1; tensor.rank()];
                        repeats[axis] = before;
                        edge = edge.repeat(&repeats);
                    }
                    parts.push(edge);
                }
                parts.push(tensor.clone());
                if after > 0 {
                    let mut slices = vec![Slice::full(); tensor.rank()];
                    let (start, end) = if reflect {
                        (size - after - 1, size - 1)
                    } else {
                        (size - 1, size)
                    };
                    slices[axis] = Slice::new(start as isize, Some(end as isize), 1);
                    let mut edge = tensor.slice(&slices);
                    if reflect {
                        edge = edge.flip_dim(axis);
                    } else {
                        let mut repeats = vec![1; edge.rank()];
                        repeats[axis] = after;
                        edge = edge.repeat(&repeats);
                    }
                    parts.push(edge);
                }
                tensor = $kind::concat(parts, axis)?;
            }
            tensor
        }};
    }

    Ok(match input {
        Value::Tensor(tensor) => Value::Tensor(apply!(tensor, DynTensor)),
        Value::Int(tensor) => Value::Int(apply!(tensor, DynInt)),
        Value::Bool(tensor) => Value::Bool(apply!(tensor, DynBool)),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "Pad expects a tensor, got {other:?}"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;

    #[test]
    fn normalizes_runtime_negative_axes() {
        assert_eq!(normalize_axes(vec![-1, 0], 3).unwrap(), [2, 0]);
        assert!(normalize_axes(vec![0, -3], 3).is_err());
    }

    #[test]
    fn edge_pad_repeats_boundary_values() {
        let device = Device::default();
        let input = Value::Tensor(
            DynTensor::from_data(TensorData::new(vec![1.0_f32, 2.0], [2]), 1, &device).unwrap(),
        );

        let output = nonconstant_pad(input, &[(2, 1)], false).unwrap();
        let Value::Tensor(output) = output else {
            panic!("expected float tensor")
        };

        assert_eq!(
            output.into_data().iter::<f32>().collect::<Vec<_>>(),
            [1.0, 1.0, 1.0, 2.0, 2.0]
        );
    }
}
