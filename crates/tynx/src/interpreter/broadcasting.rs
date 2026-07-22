//! Expand and Tile execution.

use burn::tensor::{Device, TensorData};
use onnx_ir::node::{
    expand::{ExpandConfig, ExpandNode},
    tile::{TileInput, TileNode},
};

use super::{Env, resolve, shape};
use crate::{Result, TynxError, Value};

pub(super) fn expand(node: &ExpandNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let requested = match &node.config {
        ExpandConfig::Static(shape) => shape.clone(),
        ExpandConfig::Runtime(reference) => shape::value_to_i64s(resolve::at(
            env,
            &node.name,
            &node.inputs,
            reference.input_index,
            device,
        )?)?,
    };
    let output_dims = expand_dims(&shape::value_dims(&input), &requested)?;
    Ok(vec![expand_value(input, output_dims, device)?])
}

pub(super) fn tile(node: &TileNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let repeats = match &node.config.repeats {
        TileInput::Static(repeats) => repeats.clone(),
        TileInput::Runtime(reference) => shape::value_to_i64s(resolve::at(
            env,
            &node.name,
            &node.inputs,
            reference.input_index,
            device,
        )?)?
        .into_iter()
        .map(|repeat| {
            usize::try_from(repeat).map_err(|_| {
                TynxError::Shape(format!("Tile repeat must be non-negative, got {repeat}"))
            })
        })
        .collect::<Result<Vec<_>>>()?,
    };
    if repeats.len() != shape::value_dims(&input).len() {
        return Err(TynxError::Shape(format!(
            "Tile has {} repeats for rank {}",
            repeats.len(),
            shape::value_dims(&input).len()
        )));
    }

    Ok(vec![match input {
        Value::Tensor(tensor) => Value::Tensor(tensor.repeat(&repeats)),
        Value::Int(tensor) => Value::Int(tensor.repeat(&repeats)),
        Value::Bool(tensor) => Value::Bool(tensor.repeat(&repeats)),
        Value::Scalar(scalar) if repeats.is_empty() => Value::Scalar(scalar),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "Tile expects a tensor, got {other:?}"
            )));
        }
    }])
}

fn expand_dims(input: &[usize], requested: &[i64]) -> Result<Vec<usize>> {
    let rank = input.len().max(requested.len());
    let mut output = Vec::with_capacity(rank);
    for axis in 0..rank {
        let input_dim = input
            .get(axis.wrapping_sub(rank - input.len()))
            .copied()
            .unwrap_or(1);
        let requested_dim = requested
            .get(axis.wrapping_sub(rank - requested.len()))
            .copied()
            .unwrap_or(1);
        let requested_dim = usize::try_from(requested_dim).map_err(|_| {
            TynxError::Shape(format!(
                "Expand dimension must be non-negative, got {requested_dim}"
            ))
        })?;
        output.push(match (input_dim, requested_dim) {
            (input, requested) if input == requested => input,
            (1, requested) => requested,
            (input, 1) => input,
            _ => {
                return Err(TynxError::Shape(format!(
                    "cannot expand input shape {input:?} to {requested:?}"
                )));
            }
        });
    }
    Ok(output)
}

fn expand_value(input: Value, dims: Vec<usize>, device: &Device) -> Result<Value> {
    if dims.is_empty() {
        return shape::reshape_value(input, dims, device);
    }
    let rank = dims.len();
    Ok(match input {
        Value::Tensor(tensor) => Value::Tensor(tensor.to_rank(rank)?.expand(&dims)?),
        Value::Int(tensor) => Value::Int(tensor.to_rank(rank)?.expand(&dims)?),
        Value::Bool(tensor) => Value::Bool(tensor.to_rank(rank)?.expand(&dims)?),
        Value::Scalar(scalar) => {
            let tensor = shape::reshape_value(Value::Scalar(scalar), vec![1; rank], device)?;
            return expand_value(tensor, dims, device);
        }
        Value::Shape(values) => {
            let len = values.len();
            let tensor = Value::from_tensor_data(TensorData::new(values, [len]), 1, device)?;
            return expand_value(tensor, dims, device);
        }
    })
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::{
            expand::{ExpandConfig, ExpandNodeBuilder},
            tile::{TileConfig, TileInput, TileNodeBuilder},
        },
    };

    use super::*;

    #[test]
    fn expands_with_leading_rank_promotion() {
        let node = ExpandNodeBuilder::new("expand")
            .input_tensor("x", 2, DType::F32)
            .input_shape("shape", 3)
            .output_tensor("y", 3, DType::F32)
            .config(ExpandConfig::Static(vec![2, 3, 4]))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 4]),
                2,
                &device,
            )
            .unwrap(),
        );

        let output = expand(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();

        assert_eq!(output.dims(), [2, 3, 4]);
    }

    #[test]
    fn tiles_each_axis() {
        let node = TileNodeBuilder::new("tile")
            .input_tensor("x", 2, DType::I64)
            .input_shape("repeats", 2)
            .output_tensor("y", 2, DType::I64)
            .config(TileConfig::new(TileInput::Static(vec![2, 3])))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(TensorData::new(vec![1_i64, 2], [1, 2]), 2, &device).unwrap(),
        );

        let output = tile(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_int()
            .unwrap();

        assert_eq!(output.dims(), [2, 6]);
        assert_eq!(
            output.into_data().iter::<i64>().collect::<Vec<_>>(),
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        );
    }

    #[test]
    fn rejects_incompatible_expansion() {
        assert!(expand_dims(&[2, 3], &[2, 4]).is_err());
    }
}
