//! ONNX gather operator execution.

use burn::tensor::{Device, Slice, TensorData};
use onnx_ir::node::{
    gather::GatherNode, gather_elements::GatherElementsNode, gathernd::GatherNDNode,
};

use super::{Env, resolve, shape};
use crate::{DynInt, Result, Scalar, TynxError, Value};

pub(super) fn gather(node: &GatherNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let data = resolve::first(env, &node.name, &node.inputs, device)?;
    let indices = resolve::at(env, &node.name, &node.inputs, 1, device)?;
    let indices_dims = shape::value_dims(&indices);

    if let Value::Shape(values) = data {
        if node.config.axis != 0 {
            return Err(TynxError::Shape("Shape Gather only supports axis 0".into()));
        }
        let gathered = gather_shape(&values, shape::value_to_i64s(indices)?)?;
        return Ok(vec![if indices_dims.is_empty() {
            Value::Scalar(Scalar::I64(gathered[0]))
        } else {
            Value::Shape(gathered)
        }]);
    }

    let data_dims = shape::value_dims(&data);
    let axis_size = *data_dims.get(node.config.axis).ok_or_else(|| {
        TynxError::Shape(format!(
            "Gather axis {} is outside rank {}",
            node.config.axis,
            data_dims.len()
        ))
    })?;
    let indices = indices_tensor(indices, device)?
        .reshape(vec![indices_dims.iter().product::<usize>().max(1)])?
        .normalize_indices(axis_size)?;
    let selected = match data {
        Value::Tensor(data) => Value::Tensor(data.select(node.config.axis, indices)?),
        Value::Int(data) => Value::Int(data.select(node.config.axis, indices)?),
        Value::Bool(data) => Value::Bool(data.select(node.config.axis, indices)?),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "Gather expects tensor data, got {other:?}"
            )));
        }
    };

    let mut output_dims = data_dims[..node.config.axis].to_vec();
    output_dims.extend(indices_dims);
    output_dims.extend_from_slice(&data_dims[node.config.axis + 1..]);
    Ok(vec![shape::reshape_value(selected, output_dims, device)?])
}

pub(super) fn gather_elements(
    node: &GatherElementsNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let data = resolve::first(env, &node.name, &node.inputs, device)?;
    let indices = resolve::at(env, &node.name, &node.inputs, 1, device)?;
    let data_dims = shape::value_dims(&data);
    let indices_dims = shape::value_dims(&indices);
    if data_dims.len() != indices_dims.len() {
        return Err(TynxError::Shape(format!(
            "GatherElements data rank {} differs from indices rank {}",
            data_dims.len(),
            indices_dims.len()
        )));
    }
    let axis_size = *data_dims.get(node.config.axis).ok_or_else(|| {
        TynxError::Shape(format!(
            "GatherElements axis {} is outside rank {}",
            node.config.axis,
            data_dims.len()
        ))
    })?;
    let indices = indices_tensor(indices, device)?.normalize_indices(axis_size)?;
    let output = match data {
        Value::Tensor(data) => Value::Tensor(data.gather(node.config.axis, indices)?),
        Value::Int(data) => Value::Int(data.gather(node.config.axis, indices)?),
        Value::Bool(data) => Value::Bool(data.gather(node.config.axis, indices)?),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "GatherElements expects tensor data, got {other:?}"
            )));
        }
    };
    Ok(vec![output])
}

pub(super) fn gather_nd(node: &GatherNDNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let data = resolve::first(env, &node.name, &node.inputs, device)?;
    let indices = resolve::at(env, &node.name, &node.inputs, 1, device)?;
    let data_dims = shape::value_dims(&data);
    let indices_dims = shape::value_dims(&indices);
    let indices_rank = indices_dims.len();
    let tuple_size = *indices_dims
        .last()
        .ok_or_else(|| TynxError::Shape("GatherND indices must have rank at least 1".into()))?;
    if node.config.batch_dims + tuple_size > data_dims.len() {
        return Err(TynxError::Shape(format!(
            "GatherND tuple size {tuple_size} with {} batch dimensions exceeds data rank {}",
            node.config.batch_dims,
            data_dims.len()
        )));
    }
    for axis in 0..node.config.batch_dims {
        if indices_dims[axis] != data_dims[axis] {
            return Err(TynxError::Shape(format!(
                "GatherND batch dimension {axis} differs: data {}, indices {}",
                data_dims[axis], indices_dims[axis]
            )));
        }
    }

    let mut indices = indices_tensor(indices, device)?;
    if node.config.batch_dims > 0 {
        indices =
            prepend_batch_coordinates(indices, &indices_dims, node.config.batch_dims, device)?;
    }
    indices =
        normalize_coordinate_tuples(indices, &data_dims, node.config.batch_dims + tuple_size)?;

    let output_rank = indices_rank + data_dims.len() - tuple_size - 1 - node.config.batch_dims;
    if output_rank == 0 {
        return Err(TynxError::UnsupportedOp(
            "GatherND with scalar output".into(),
        ));
    }
    let output = match data {
        Value::Tensor(data) => Value::Tensor(data.gather_nd(indices, output_rank)?),
        Value::Int(data) => Value::Int(data.gather_nd(indices, output_rank)?),
        Value::Bool(data) => Value::Bool(data.gather_nd(indices, output_rank)?),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "GatherND expects tensor data, got {other:?}"
            )));
        }
    };
    Ok(vec![output])
}

fn indices_tensor(value: Value, device: &Device) -> Result<DynInt> {
    match value {
        Value::Int(indices) => Ok(indices),
        Value::Shape(values) => {
            DynInt::from_data(TensorData::new(values.clone(), [values.len()]), 1, device)
        }
        Value::Scalar(Scalar::I64(index)) => {
            DynInt::from_data(TensorData::new(vec![index], [1]), 1, device)
        }
        Value::Scalar(Scalar::U64(index)) => {
            DynInt::from_data(TensorData::new(vec![index], [1]), 1, device)
        }
        other => Err(TynxError::TypeMismatch(format!(
            "indices must be integer values, got {other:?}"
        ))),
    }
}

fn gather_shape(values: &[i64], indices: Vec<i64>) -> Result<Vec<i64>> {
    let size = values.len() as i64;
    indices
        .into_iter()
        .map(|index| {
            let index = if index < 0 { index + size } else { index };
            values.get(index as usize).copied().ok_or_else(|| {
                TynxError::Shape(format!("Gather index {index} is outside dimension {size}"))
            })
        })
        .collect()
}

fn prepend_batch_coordinates(
    indices: DynInt,
    indices_dims: &[usize],
    batch_dims: usize,
    device: &Device,
) -> Result<DynInt> {
    let rank = indices_dims.len();
    let dtype = indices.dtype();
    let mut parts = Vec::with_capacity(batch_dims + 1);
    for axis in 0..batch_dims {
        let size = indices_dims[axis];
        let coordinate = DynInt::from_data(
            TensorData::new((0..size as i64).collect::<Vec<_>>(), [size]),
            1,
            device,
        )?
        .cast(dtype);
        let mut base_dims = vec![1; rank];
        base_dims[axis] = size;
        let mut expanded_dims = indices_dims.to_vec();
        expanded_dims[rank - 1] = 1;
        parts.push(coordinate.reshape(base_dims)?.expand(&expanded_dims)?);
    }
    parts.push(indices);
    DynInt::concat(parts, rank - 1)
}

fn normalize_coordinate_tuples(
    indices: DynInt,
    data_dims: &[usize],
    tuple_size: usize,
) -> Result<DynInt> {
    let rank = indices.rank();
    let last_axis = rank - 1;
    let mut coordinates = Vec::with_capacity(tuple_size);
    for (coordinate, &size) in data_dims.iter().take(tuple_size).enumerate() {
        let mut slices = vec![Slice::full(); rank];
        slices[last_axis] = Slice::new(coordinate as isize, Some(coordinate as isize + 1), 1);
        coordinates.push(indices.clone().slice(&slices).normalize_indices(size)?);
    }
    DynInt::concat(coordinates, last_axis)
}

#[cfg(test)]
mod tests {
    use burn::tensor::DType;

    use super::*;

    #[test]
    fn gathers_negative_shape_indices() {
        assert_eq!(gather_shape(&[2, 3, 5], vec![-1, 0]).unwrap(), [5, 2]);
    }

    #[test]
    fn gathers_elements_along_an_axis() {
        let device = Device::default();
        let data = crate::DynTensor::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]),
            2,
            &device,
        )
        .unwrap();
        let indices =
            DynInt::from_data(TensorData::new(vec![1_i64, 0, 0, 1], [2, 2]), 2, &device).unwrap();

        let output = data.gather(1, indices).unwrap().into_data();

        assert_eq!(output.dtype, DType::F32);
        assert_eq!(
            output.iter::<f32>().collect::<Vec<_>>(),
            [2.0, 1.0, 3.0, 4.0]
        );
    }
}
