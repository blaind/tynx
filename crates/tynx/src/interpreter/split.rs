//! ONNX Split execution.

use burn::tensor::{Device, Slice};
use onnx_ir::node::split::{SplitNode, SplitSizesInput};

use super::{Env, resolve, shape};
use crate::{Result, TynxError, Value};

pub(super) fn split(node: &SplitNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let dims = shape::value_dims(&input);
    let axis_size = *dims.get(node.config.axis).ok_or_else(|| {
        TynxError::Shape(format!(
            "Split axis {} is outside rank {}",
            node.config.axis,
            dims.len()
        ))
    })?;
    let sizes = split_sizes(node, env, device, axis_size)?;

    if sizes.len() != node.outputs.len() {
        return Err(TynxError::Shape(format!(
            "Split produced {} parts for {} outputs",
            sizes.len(),
            node.outputs.len()
        )));
    }
    let total = sizes.iter().try_fold(0_usize, |total, size| {
        total
            .checked_add(*size)
            .ok_or_else(|| TynxError::Shape("Split sizes overflow usize".to_string()))
    })?;
    if total != axis_size {
        return Err(TynxError::Shape(format!(
            "Split sizes total {total}, expected axis size {axis_size}"
        )));
    }

    let mut offset = 0;
    sizes
        .into_iter()
        .map(|size| {
            let end = offset + size;
            let output = slice_part(input.clone(), &dims, node.config.axis, offset, end)?;
            offset = end;
            Ok(output)
        })
        .collect()
}

fn split_sizes(
    node: &SplitNode,
    env: &Env,
    device: &Device,
    axis_size: usize,
) -> Result<Vec<usize>> {
    if let Some(sizes) = &node.config.split_sizes {
        return match sizes {
            SplitSizesInput::Static(sizes) => Ok(sizes.clone()),
            SplitSizesInput::Runtime(reference) => shape::value_to_i64s(resolve::at(
                env,
                &node.name,
                &node.inputs,
                reference.input_index,
                device,
            )?)?
            .into_iter()
            .map(|size| {
                usize::try_from(size).map_err(|_| {
                    TynxError::Shape(format!("Split size must be non-negative, got {size}"))
                })
            })
            .collect(),
        };
    }

    if let Some(size) = node.config.split_size {
        if size == 0 {
            return Err(TynxError::Shape(
                "Split size must be greater than zero".to_string(),
            ));
        }
        let mut remaining = axis_size;
        let mut sizes = Vec::new();
        while remaining > 0 {
            let part = remaining.min(size);
            sizes.push(part);
            remaining -= part;
        }
        return Ok(sizes);
    }

    let count = node.config.num_outputs.unwrap_or(node.outputs.len());
    if count == 0 {
        return Err(TynxError::Shape(
            "Split output count must be greater than zero".to_string(),
        ));
    }
    let chunk = axis_size.div_ceil(count);
    let mut remaining = axis_size;
    Ok((0..count)
        .map(|_| {
            let size = chunk.min(remaining);
            remaining = remaining.saturating_sub(chunk);
            size
        })
        .collect())
}

fn slice_part(
    input: Value,
    dims: &[usize],
    axis: usize,
    start: usize,
    end: usize,
) -> Result<Value> {
    let start = isize::try_from(start)
        .map_err(|_| TynxError::Shape(format!("Split offset {start} exceeds isize")))?;
    let end = isize::try_from(end)
        .map_err(|_| TynxError::Shape(format!("Split offset {end} exceeds isize")))?;
    let mut slices = vec![Slice::full(); dims.len()];
    slices[axis] = Slice::new(start, Some(end), 1);

    match input {
        Value::Tensor(tensor) => Ok(Value::Tensor(tensor.slice(&slices))),
        Value::Int(tensor) => Ok(Value::Int(tensor.slice(&slices))),
        Value::Bool(tensor) => Ok(Value::Bool(tensor.slice(&slices))),
        Value::Shape(values) if axis == 0 => {
            Ok(Value::Shape(values[start as usize..end as usize].to_vec()))
        }
        other => Err(TynxError::TypeMismatch(format!(
            "Split expects a tensor or shape, got {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::split::{SplitConfig, SplitNodeBuilder, SplitSizesInput},
    };

    use super::*;

    #[test]
    fn splits_with_uneven_static_sizes() {
        let node = SplitNodeBuilder::new("split")
            .input_tensor("x", 2, DType::F32)
            .output_tensor("a", 2, DType::F32)
            .output_tensor("b", 2, DType::F32)
            .output_tensor("c", 2, DType::F32)
            .config(SplitConfig {
                axis: 1,
                split_size: None,
                split_sizes: Some(SplitSizesInput::Static(vec![1, 2, 1])),
                num_outputs: None,
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new((0..8).map(|value| value as f32).collect::<Vec<_>>(), [2, 4]),
                2,
                &device,
            )
            .unwrap(),
        );

        let outputs = split(&node, &env, &device).unwrap();
        let dims = outputs.iter().map(shape::value_dims).collect::<Vec<_>>();

        assert_eq!(dims, [vec![2, 1], vec![2, 2], vec![2, 1]]);
        assert_eq!(
            outputs[1]
                .clone()
                .into_tensor()
                .unwrap()
                .into_data()
                .iter::<f32>()
                .collect::<Vec<_>>(),
            [1.0, 2.0, 5.0, 6.0]
        );
    }
}
