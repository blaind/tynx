//! ONNX-ML Imputer execution.

use burn::tensor::{DType, Device, TensorData};
use onnx_ir::node::imputer::ImputerNode;

use super::{Env, resolve};
use crate::{DynInt, DynTensor, Result, Scalar, TynxError, Value};

pub(super) fn imputer(node: &ImputerNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let output = match input {
        Value::Tensor(tensor) => impute_float_tensor(node, tensor, device)?,
        Value::Int(tensor) => impute_int_tensor(node, tensor, device)?,
        Value::Scalar(scalar) => impute_scalar(node, scalar),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "Imputer requires a floating-point or integer value, got {other:?}"
            )));
        }
    };
    Ok(vec![output])
}

fn impute_float_tensor(node: &ImputerNode, tensor: DynTensor, device: &Device) -> Result<Value> {
    let Some(values) = node.config.imputed_value_floats.as_deref() else {
        return Ok(Value::Tensor(tensor));
    };
    let dims = tensor.dims();
    let dtype = tensor.dtype();
    let replaced = node.config.replaced_value_float.unwrap_or(f32::NAN) as f64;
    let mask = if replaced.is_nan() {
        tensor.clone().is_nan()
    } else {
        tensor
            .clone()
            .equal_broadcast(DynTensor::full(&[1], replaced, device, dtype)?)?
    };
    let replacements = float_replacements(values, &dims, dtype, device)?;
    Ok(Value::Tensor(DynTensor::where_select(
        mask,
        replacements,
        tensor,
    )?))
}

fn impute_int_tensor(node: &ImputerNode, tensor: DynInt, device: &Device) -> Result<Value> {
    let Some(values) = node.config.imputed_value_int64s.as_deref() else {
        return Ok(Value::Int(tensor));
    };
    let dims = tensor.dims();
    let dtype = tensor.dtype();
    let replaced = node.config.replaced_value_int64.unwrap_or(0);
    let mask = tensor
        .clone()
        .equal_broadcast(DynInt::full(&[1], replaced, device, dtype)?)?;
    let replacements = int_replacements(values, &dims, dtype, device)?;
    Ok(Value::Int(DynInt::where_select(
        mask,
        replacements,
        tensor,
    )?))
}

fn impute_scalar(node: &ImputerNode, scalar: Scalar) -> Value {
    match scalar {
        Scalar::F64(value) => {
            let Some(&replacement) = node
                .config
                .imputed_value_floats
                .as_deref()
                .and_then(|values| values.first())
            else {
                return Value::Scalar(scalar);
            };
            let replaced = node.config.replaced_value_float.unwrap_or(f32::NAN) as f64;
            if (replaced.is_nan() && value.is_nan()) || value == replaced {
                Value::Scalar(Scalar::F64(f64::from(replacement)))
            } else {
                Value::Scalar(scalar)
            }
        }
        Scalar::I64(value) => {
            let Some(&replacement) = node
                .config
                .imputed_value_int64s
                .as_deref()
                .and_then(|values| values.first())
            else {
                return Value::Scalar(scalar);
            };
            if value == node.config.replaced_value_int64.unwrap_or(0) {
                Value::Scalar(Scalar::I64(replacement))
            } else {
                Value::Scalar(scalar)
            }
        }
        Scalar::U64(_) | Scalar::Bool(_) => Value::Scalar(scalar),
    }
}

fn float_replacements(
    values: &[f32],
    input_dims: &[usize],
    dtype: DType,
    device: &Device,
) -> Result<DynTensor> {
    validate_feature_count(values.len(), input_dims)?;
    if values.len() == 1 {
        return DynTensor::full(&[1], f64::from(values[0]), device, dtype);
    }
    let tensor = DynTensor::from_data(
        TensorData::new(
            values.iter().map(|&value| f64::from(value)).collect(),
            [values.len()],
        ),
        1,
        device,
    )?
    .cast(dtype);
    tensor.reshape(feature_shape(input_dims.len(), values.len()))
}

fn int_replacements(
    values: &[i64],
    input_dims: &[usize],
    dtype: DType,
    device: &Device,
) -> Result<DynInt> {
    validate_feature_count(values.len(), input_dims)?;
    if values.len() == 1 {
        return DynInt::full(&[1], values[0], device, dtype);
    }
    DynInt::from_data(TensorData::new(values.to_vec(), [values.len()]), 1, device)?
        .cast(dtype)
        .reshape(feature_shape(input_dims.len(), values.len()))
}

fn validate_feature_count(count: usize, input_dims: &[usize]) -> Result<()> {
    let features = input_dims.last().copied().ok_or_else(|| {
        TynxError::Shape("Imputer requires a tensor with at least one dimension".to_string())
    })?;
    if count != 1 && count != features {
        return Err(TynxError::Shape(format!(
            "Imputer has {count} replacement values, expected 1 or {features}"
        )));
    }
    Ok(())
}

fn feature_shape(rank: usize, features: usize) -> Vec<usize> {
    let mut dims = vec![1; rank];
    if let Some(last) = dims.last_mut() {
        *last = features;
    }
    dims
}

#[cfg(test)]
mod tests {
    use onnx_ir::node::imputer::{ImputerConfig, ImputerNodeBuilder};

    use super::*;

    #[test]
    fn replaces_nan_values_per_feature() {
        let node = ImputerNodeBuilder::new("imputer")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(ImputerConfig::new(
                Some(vec![1.0, 2.0, 3.0]),
                None,
                None,
                None,
            ))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "input".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![f32::NAN, 5.0, f32::NAN, 7.0, f32::NAN, 9.0], [2, 3]),
                2,
                &device,
            )
            .unwrap(),
        );

        let output = imputer(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();

        output.into_data().assert_eq(
            &TensorData::new(vec![1.0_f32, 5.0, 3.0, 7.0, 2.0, 9.0], [2, 3]),
            false,
        );
    }

    #[test]
    fn replaces_default_integer_sentinel_and_preserves_dtype() {
        let node = ImputerNodeBuilder::new("imputer")
            .input_tensor("input", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .config(ImputerConfig::new(None, Some(vec![9]), None, None))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "input".to_string(),
            Value::from_tensor_data(TensorData::new(vec![0_i32, 1, 2, 0], [2, 2]), 2, &device)
                .unwrap(),
        );

        let output = imputer(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_int()
            .unwrap();

        assert_eq!(output.dtype(), DType::I32);
        output
            .into_data()
            .assert_eq(&TensorData::new(vec![9_i32, 1, 2, 9], [2, 2]), false);
    }

    #[test]
    fn replaces_an_explicit_float_scalar_sentinel() {
        let node = ImputerNodeBuilder::new("imputer")
            .input_scalar("input", DType::F32)
            .output_scalar("output", DType::F32)
            .config(ImputerConfig::new(
                Some(vec![1.5]),
                None,
                Some(-999.0),
                None,
            ))
            .build();
        let mut env = Env::new();
        env.insert("input".to_string(), Value::Scalar(Scalar::F64(-999.0)));

        let output = imputer(&node, &env, &Device::default())
            .unwrap()
            .pop()
            .unwrap();

        assert!(matches!(output, Value::Scalar(Scalar::F64(1.5))));
    }

    #[test]
    fn rejects_an_incompatible_replacement_count() {
        let node = ImputerNodeBuilder::new("imputer")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(ImputerConfig::new(
                Some(vec![1.0, 2.0, 3.0]),
                None,
                None,
                None,
            ))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "input".to_string(),
            Value::from_tensor_data(TensorData::new(vec![0.0_f32; 4], [2, 2]), 2, &device).unwrap(),
        );

        let error = imputer(&node, &env, &device).unwrap_err();

        assert_eq!(
            error,
            TynxError::Shape("Imputer has 3 replacement values, expected 1 or 2".to_string())
        );
    }
}
