//! Concat operator execution.

use burn::tensor::{DType, Device, TensorData};
use onnx_ir::{ir::ArgType, node::concat::ConcatNode};

use super::{Env, cast, resolve, shape};
use crate::{Result, Scalar, TynxError, Value};

pub(super) fn concat(node: &ConcatNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let values = node
        .inputs
        .iter()
        .enumerate()
        .map(|(index, input)| resolve::input_at(env, input, index, device))
        .collect::<Result<Vec<_>>>()?;

    if matches!(
        node.outputs.first().map(|output| &output.ty),
        Some(ArgType::Shape(_))
    ) {
        let output = values
            .into_iter()
            .map(shape::value_to_i64s)
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        return Ok(vec![Value::Shape(output)]);
    }

    let target = node
        .outputs
        .first()
        .map(|output| output.ty.elem_type())
        .ok_or_else(|| TynxError::Shape("Concat has no output".to_string()))?;
    let values = values
        .into_iter()
        .map(|value| scalar_to_tensor(value, target, device))
        .collect::<Result<Vec<_>>>()?;
    validate_shapes(&values, node.config.axis)?;

    let output = if target.is_float() {
        let tensors = values
            .into_iter()
            .map(|value| cast::cast_value(value, target, device)?.into_tensor())
            .collect::<Result<Vec<_>>>()?;
        Value::Tensor(crate::DynTensor::concat(tensors, node.config.axis)?)
    } else if target.is_int() || target.is_uint() {
        let tensors = values
            .into_iter()
            .map(|value| cast::cast_value(value, target, device)?.into_int())
            .collect::<Result<Vec<_>>>()?;
        Value::Int(crate::DynInt::concat(tensors, node.config.axis)?)
    } else if target.is_bool() {
        let tensors = values
            .into_iter()
            .map(|value| cast::cast_value(value, target, device)?.into_bool())
            .collect::<Result<Vec<_>>>()?;
        Value::Bool(crate::DynBool::concat(tensors, node.config.axis)?)
    } else {
        return Err(TynxError::TypeMismatch(format!(
            "Concat dtype {target:?} is unsupported"
        )));
    };

    Ok(vec![output])
}

fn scalar_to_tensor(value: Value, target: DType, device: &Device) -> Result<Value> {
    let Value::Scalar(scalar) = value else {
        return Ok(value);
    };
    let data = match scalar {
        Scalar::F64(value) => TensorData::new(vec![value], [1]),
        Scalar::I64(value) => TensorData::new(vec![value], [1]),
        Scalar::U64(value) => TensorData::new(vec![value], [1]),
        Scalar::Bool(value) => TensorData::new(vec![value], [1]),
    };
    cast::cast_value(Value::from_tensor_data(data, 1, device)?, target, device)
}

fn validate_shapes(values: &[Value], axis: usize) -> Result<()> {
    let Some(first) = values.first() else {
        return Err(TynxError::Shape(
            "Concat requires at least one input".to_string(),
        ));
    };
    let expected = shape::value_dims(first);
    if axis >= expected.len() {
        return Err(TynxError::Shape(format!(
            "Concat axis {axis} is out of range for rank {}",
            expected.len()
        )));
    }
    for dims in values.iter().skip(1).map(shape::value_dims) {
        if dims.len() != expected.len()
            || dims
                .iter()
                .zip(&expected)
                .enumerate()
                .any(|(dim, (actual, expected))| dim != axis && actual != expected)
        {
            return Err(TynxError::Shape(format!(
                "Concat input shape {dims:?} is incompatible with {expected:?} on axis {axis}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::concat::{ConcatConfig, ConcatNodeBuilder},
    };

    use super::*;
    use crate::DynTensor;

    #[test]
    fn concatenates_tensors_along_nonzero_axis() {
        let node = ConcatNodeBuilder::new("concat")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .config(ConcatConfig { axis: 1 })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        for (name, values) in [("a", vec![1.0_f32, 2.0]), ("b", vec![3.0, 4.0])] {
            env.insert(
                name.into(),
                Value::Tensor(
                    DynTensor::from_data(TensorData::new(values, [2, 1]), 2, &device).unwrap(),
                ),
            );
        }

        let output = concat(&node, &env, &device).unwrap();
        let Value::Tensor(output) = output.into_iter().next().unwrap() else {
            panic!("expected tensor output");
        };
        assert_eq!(output.dims(), [2, 2]);
        assert_eq!(
            output.into_data().iter::<f32>().collect::<Vec<_>>(),
            [1.0, 3.0, 2.0, 4.0]
        );
    }

    #[test]
    fn rejects_incompatible_non_axis_dimensions() {
        let values = [
            Value::Tensor(
                DynTensor::from_data(
                    TensorData::new(vec![0.0_f32; 6], [2, 3]),
                    2,
                    &Device::default(),
                )
                .unwrap(),
            ),
            Value::Tensor(
                DynTensor::from_data(
                    TensorData::new(vec![0.0_f32; 8], [4, 2]),
                    2,
                    &Device::default(),
                )
                .unwrap(),
            ),
        ];

        assert!(validate_shapes(&values, 1).is_err());
    }
}
