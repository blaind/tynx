//! ONNX selection and index-producing operator execution.

use burn::tensor::{DType, Device, TensorData};
use onnx_ir::node::{
    nonzero::NonZeroNode,
    one_hot::{OneHotDepthInput, OneHotNode, OneHotValuesInput},
    topk::{TopKInput, TopKNode},
};

use super::{Env, gather, resolve, shape};
use crate::{DynBool, DynInt, DynTensor, Result, Scalar, TynxError, Value};

pub(super) fn topk(node: &TopKNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let dims = shape::value_dims(&input);
    let axis_size = *dims.get(node.config.axis).ok_or_else(|| {
        TynxError::Shape(format!(
            "TopK axis {} is outside rank {}",
            node.config.axis,
            dims.len()
        ))
    })?;
    let k = match &node.config.k {
        TopKInput::Static(k) => *k,
        TopKInput::Runtime(reference) => {
            let values = shape::value_to_i64s(resolve::at(
                env,
                &node.name,
                &node.inputs,
                reference.input_index,
                device,
            )?)?;
            if values.len() != 1 {
                return Err(TynxError::Shape(format!(
                    "TopK k must have one value, got {}",
                    values.len()
                )));
            }
            usize::try_from(values[0]).map_err(|_| {
                TynxError::Shape(format!("TopK k must be positive, got {}", values[0]))
            })?
        }
    };
    if k == 0 || k > axis_size {
        return Err(TynxError::Shape(format!(
            "TopK k must be between 1 and axis size {axis_size}, got {k}"
        )));
    }

    let (values, indices) = match input {
        Value::Tensor(tensor) => {
            let (values, indices) = tensor.topk(k, node.config.axis);
            (Value::Tensor(values), indices)
        }
        Value::Int(tensor) => {
            let (values, indices) = tensor.topk(k, node.config.axis);
            (Value::Int(values), indices)
        }
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "TopK expects a numeric tensor, got {other:?}"
            )));
        }
    };
    Ok(vec![values, Value::Int(indices)])
}

pub(super) fn one_hot(node: &OneHotNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let indices_value = resolve::first(env, &node.name, &node.inputs, device)?;
    let indices_dims = shape::value_dims(&indices_value);
    let depth = match &node.config.depth {
        OneHotDepthInput::Static(depth) => *depth,
        OneHotDepthInput::Runtime(reference) => {
            let value = resolve::at(env, &node.name, &node.inputs, reference.input_index, device)?;
            one_hot_depth(value)?
        }
    };
    if depth == 0 {
        return Err(TynxError::Shape("OneHot depth must be positive".into()));
    }
    let output_rank = indices_dims.len() + 1;
    let axis = normalize_one_hot_axis(node.config.axis, output_rank)?;
    let (off, on) = match &node.config.values {
        OneHotValuesInput::Static(values) => (values[0] as f64, values[1] as f64),
        OneHotValuesInput::Runtime(reference) => value_pair(resolve::at(
            env,
            &node.name,
            &node.inputs,
            reference.input_index,
            device,
        )?)?,
    };
    let dtype = node.outputs[0].ty.elem_type();

    let indices = one_hot_indices(indices_value, device)?.normalize_indices(depth)?;
    let mut output_dims = indices_dims.clone();
    output_dims.insert(axis, depth);
    let mut indices_view_dims = indices_dims;
    indices_view_dims.insert(axis, 1);
    let indices = indices.reshape(indices_view_dims)?.expand(&output_dims)?;

    let classes = DynInt::from_data(
        TensorData::new((0..depth as i64).collect::<Vec<_>>(), [depth]),
        1,
        device,
    )?
    .cast(indices.dtype());
    let mut class_dims = vec![1; output_rank];
    class_dims[axis] = depth;
    let classes = classes.reshape(class_dims)?.expand(&output_dims)?;
    let selected = indices.equal_broadcast(classes)?;

    let output = if dtype.is_float() {
        Value::Tensor(DynTensor::where_select(
            selected,
            DynTensor::full(&output_dims, on, device, dtype)?,
            DynTensor::full(&output_dims, off, device, dtype)?,
        )?)
    } else if dtype.is_int() || dtype.is_uint() {
        Value::Int(DynInt::where_select(
            selected,
            DynInt::full(&output_dims, on as i64, device, dtype)?,
            DynInt::full(&output_dims, off as i64, device, dtype)?,
        )?)
    } else if dtype.is_bool() {
        Value::Bool(DynBool::where_select(
            selected,
            DynBool::full(&output_dims, on != 0.0, device)?,
            DynBool::full(&output_dims, off != 0.0, device)?,
        )?)
    } else {
        return Err(TynxError::TypeMismatch(format!(
            "OneHot output dtype {dtype:?} is unsupported"
        )));
    };
    Ok(vec![output])
}

pub(super) fn nonzero(node: &NonZeroNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let mask = match input {
        Value::Tensor(tensor) => tensor.to_bool(),
        Value::Int(tensor) => tensor.to_bool(),
        Value::Bool(tensor) => tensor,
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "NonZero expects a tensor, got {other:?}"
            )));
        }
    };
    Ok(vec![Value::Int(mask.nonzero())])
}

fn normalize_one_hot_axis(axis: i64, output_rank: usize) -> Result<usize> {
    let output_rank = output_rank as i64;
    let axis = if axis < 0 { axis + output_rank } else { axis };
    if !(0..output_rank).contains(&axis) {
        return Err(TynxError::Shape(format!(
            "OneHot axis {axis} is outside output rank {output_rank}"
        )));
    }
    Ok(axis as usize)
}

fn one_hot_indices(value: Value, device: &Device) -> Result<DynInt> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.to_int(DType::I64)),
        Value::Scalar(Scalar::F64(value)) if value.is_finite() => {
            DynInt::from_data(TensorData::new(vec![value as i64], [1]), 1, device)
        }
        value => gather::indices_tensor(value, device),
    }
}

fn one_hot_depth(value: Value) -> Result<usize> {
    let depth = match value {
        Value::Scalar(Scalar::I64(value)) => value,
        Value::Scalar(Scalar::U64(value)) => i64::try_from(value)
            .map_err(|_| TynxError::Shape(format!("OneHot depth {value} exceeds i64")))?,
        Value::Scalar(Scalar::F64(value)) if value.is_finite() && value.fract() == 0.0 => {
            value as i64
        }
        value => {
            let values = shape::value_to_i64s(value)?;
            if values.len() != 1 {
                return Err(TynxError::Shape(format!(
                    "OneHot depth must have one value, got {}",
                    values.len()
                )));
            }
            values[0]
        }
    };
    usize::try_from(depth)
        .map_err(|_| TynxError::Shape(format!("OneHot depth must be non-negative, got {depth}")))
}

fn value_pair(value: Value) -> Result<(f64, f64)> {
    let data = match value {
        Value::Tensor(tensor) => tensor.into_data(),
        Value::Int(tensor) => tensor.into_data(),
        Value::Bool(tensor) => tensor.into_data(),
        Value::Shape(values) if values.len() == 2 => {
            return Ok((values[0] as f64, values[1] as f64));
        }
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "OneHot values must be a tensor, got {other:?}"
            )));
        }
    };
    let values = data.iter::<f64>().collect::<Vec<_>>();
    match values.as_slice() {
        [off, on] => Ok((*off, *on)),
        _ => Err(TynxError::Shape(format!(
            "OneHot values must contain two elements, got {}",
            values.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_negative_one_hot_axes() {
        assert_eq!(normalize_one_hot_axis(-1, 3).unwrap(), 2);
        assert_eq!(normalize_one_hot_axis(-3, 3).unwrap(), 0);
        assert!(normalize_one_hot_axis(-4, 3).is_err());
    }

    #[test]
    fn extracts_runtime_value_pair() {
        assert_eq!(value_pair(Value::Shape(vec![-1, 2])).unwrap(), (-1.0, 2.0));
    }
}
