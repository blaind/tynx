//! ONNX scatter operator execution.

use burn::tensor::{Device, IndexingUpdateOp, TensorData};
use onnx_ir::{
    ModelProto,
    node::{
        scatter_elements::{ScatterElementsNode, ScatterElementsReduction},
        scatter_nd::{ScatterNDNode, ScatterNDReduction},
    },
};
use protobuf::Message;

use super::{Env, gather, resolve, shape};
use crate::{Result, TynxError, Value};

// Scatter was renamed to ScatterElements in opset 11 without changing the
// update semantics used by the legacy operator. Rewrite it before onnx-ir
// parsing so the typed ScatterElements processor retains the axis attribute.
pub(super) fn prepare_model(data: &[u8]) -> Result<(Vec<u8>, bool)> {
    let mut model =
        ModelProto::parse_from_bytes(data).map_err(|error| TynxError::Parse(error.to_string()))?;
    let Some(graph) = model.graph.as_mut() else {
        return Ok((Vec::new(), false));
    };
    let mut changed = false;
    for node in &mut graph.node {
        if node.op_type == "Scatter" {
            node.op_type = "ScatterElements".to_string();
            changed = true;
        }
    }
    if !changed {
        return Ok((Vec::new(), false));
    }
    for opset in &mut model.opset_import {
        if (opset.domain.is_empty() || opset.domain == "ai.onnx") && opset.version < 11 {
            opset.version = 11;
        }
    }
    model
        .write_to_bytes()
        .map(|bytes| (bytes, true))
        .map_err(|error| TynxError::Parse(error.to_string()))
}

pub(super) fn scatter_elements(
    node: &ScatterElementsNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let data = resolve::first(env, &node.name, &node.inputs, device)?;
    let indices = resolve::at(env, &node.name, &node.inputs, 1, device)?;
    let updates = resolve::at(env, &node.name, &node.inputs, 2, device)?;
    let data_dims = shape::value_dims(&data);
    let axis_size = *data_dims.get(node.config.axis).ok_or_else(|| {
        TynxError::Shape(format!(
            "ScatterElements axis {} is outside rank {}",
            node.config.axis,
            data_dims.len()
        ))
    })?;
    let indices_dims = shape::value_dims(&indices);
    let indices = gather::indices_tensor(indices, device)?.normalize_indices(axis_size)?;
    let update = scatter_elements_update(&node.config.reduction);
    let coordinates =
        element_coordinates(indices, &indices_dims, node.config.axis, &data_dims, device)?;
    let update_count = checked_product(&indices_dims)?;

    let output = match (data, updates) {
        (Value::Tensor(data), Value::Tensor(updates)) => Value::Tensor(data.scatter_nd(
            coordinates,
            updates.reshape(vec![update_count])?,
            update,
        )?),
        (Value::Int(data), Value::Int(updates)) => Value::Int(data.scatter_nd(
            coordinates,
            updates.reshape(vec![update_count])?,
            update,
        )?),
        (Value::Bool(data), Value::Bool(updates)) => Value::Bool(data.scatter_nd(
            coordinates,
            updates.reshape(vec![update_count])?,
            update,
        )?),
        (data, updates) => {
            return Err(TynxError::TypeMismatch(format!(
                "ScatterElements data and updates must have matching tensor kinds, got {data:?} and {updates:?}"
            )));
        }
    };
    Ok(vec![output])
}

fn element_coordinates(
    indices: crate::DynInt,
    indices_dims: &[usize],
    scatter_axis: usize,
    data_dims: &[usize],
    device: &Device,
) -> Result<crate::DynInt> {
    let rank = indices_dims.len();
    if rank == 0 || rank >= crate::MAX_RANK {
        return Err(TynxError::UnsupportedOp(format!(
            "ScatterElements rank {rank} coordinate construction"
        )));
    }
    let dtype = indices.dtype();
    let mut coordinates = Vec::with_capacity(rank);
    for axis in 0..rank {
        let coordinate = if axis == scatter_axis {
            indices.clone()
        } else {
            let size = indices_dims[axis];
            if size > data_dims[axis] {
                return Err(TynxError::Shape(format!(
                    "ScatterElements indices dimension {axis} has size {size}, exceeding data size {}",
                    data_dims[axis]
                )));
            }
            let mut base_dims = vec![1; rank];
            base_dims[axis] = size;
            crate::DynInt::from_data(
                TensorData::new((0..size as i64).collect::<Vec<_>>(), [size]),
                1,
                device,
            )?
            .cast(dtype)
            .reshape(base_dims)?
            .expand(indices_dims)?
        };
        let mut coordinate_dims = indices_dims.to_vec();
        coordinate_dims.push(1);
        coordinates.push(coordinate.reshape(coordinate_dims)?);
    }
    let coordinates = crate::DynInt::concat(coordinates, rank)?;
    coordinates.reshape(vec![checked_product(indices_dims)?, rank])
}

fn checked_product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1_usize, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or_else(|| TynxError::Shape("ScatterElements size overflowed usize".into()))
    })
}

pub(super) fn scatter_nd(node: &ScatterNDNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let data = resolve::first(env, &node.name, &node.inputs, device)?;
    let indices = resolve::at(env, &node.name, &node.inputs, 1, device)?;
    let updates = resolve::at(env, &node.name, &node.inputs, 2, device)?;
    let data_dims = shape::value_dims(&data);
    let indices_dims = shape::value_dims(&indices);
    let tuple_size = *indices_dims
        .last()
        .ok_or_else(|| TynxError::Shape("ScatterND indices must have rank at least 1".into()))?;
    if tuple_size > data_dims.len() {
        return Err(TynxError::Shape(format!(
            "ScatterND tuple size {tuple_size} exceeds data rank {}",
            data_dims.len()
        )));
    }
    let indices = gather::normalize_coordinate_tuples(
        gather::indices_tensor(indices, device)?,
        &data_dims,
        tuple_size,
    )?;
    let update = scatter_nd_update(&node.config.reduction);

    let output = match (data, updates) {
        (Value::Tensor(data), Value::Tensor(updates)) => {
            Value::Tensor(data.scatter_nd(indices, updates, update)?)
        }
        (Value::Int(data), Value::Int(updates)) => {
            Value::Int(data.scatter_nd(indices, updates, update)?)
        }
        (Value::Bool(data), Value::Bool(updates)) => {
            Value::Bool(data.scatter_nd(indices, updates, update)?)
        }
        (data, updates) => {
            return Err(TynxError::TypeMismatch(format!(
                "ScatterND data and updates must have matching tensor kinds, got {data:?} and {updates:?}"
            )));
        }
    };
    Ok(vec![output])
}

fn scatter_elements_update(reduction: &ScatterElementsReduction) -> IndexingUpdateOp {
    match reduction {
        ScatterElementsReduction::None => IndexingUpdateOp::Assign,
        ScatterElementsReduction::Add => IndexingUpdateOp::Add,
        ScatterElementsReduction::Mul => IndexingUpdateOp::Mul,
        ScatterElementsReduction::Min => IndexingUpdateOp::Min,
        ScatterElementsReduction::Max => IndexingUpdateOp::Max,
    }
}

fn scatter_nd_update(reduction: &ScatterNDReduction) -> IndexingUpdateOp {
    match reduction {
        ScatterNDReduction::None => IndexingUpdateOp::Assign,
        ScatterNDReduction::Add => IndexingUpdateOp::Add,
        ScatterNDReduction::Mul => IndexingUpdateOp::Mul,
        ScatterNDReduction::Min => IndexingUpdateOp::Min,
        ScatterNDReduction::Max => IndexingUpdateOp::Max,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_all_scatter_reductions() {
        assert_eq!(
            scatter_elements_update(&ScatterElementsReduction::None),
            IndexingUpdateOp::Assign
        );
        assert_eq!(
            scatter_elements_update(&ScatterElementsReduction::Add),
            IndexingUpdateOp::Add
        );
        assert_eq!(
            scatter_elements_update(&ScatterElementsReduction::Mul),
            IndexingUpdateOp::Mul
        );
        assert_eq!(
            scatter_elements_update(&ScatterElementsReduction::Min),
            IndexingUpdateOp::Min
        );
        assert_eq!(
            scatter_elements_update(&ScatterElementsReduction::Max),
            IndexingUpdateOp::Max
        );
    }
}
