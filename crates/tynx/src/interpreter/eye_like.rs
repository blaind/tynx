//! ONNX EyeLike execution.

use burn::tensor::{Bool, Device, Tensor};
use onnx_ir::{DType, node::eye_like::EyeLikeNode};

use super::{Env, resolve, shape};
use crate::{DynBool, Result, TynxError, Value};

pub(super) fn eye_like(node: &EyeLikeNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let dims = shape::value_dims(&input);
    let [rows, columns] = dims.as_slice() else {
        return Err(TynxError::Shape(format!(
            "EyeLike expects rank 2, got rank {}",
            dims.len()
        )));
    };
    let input_dtype = value_dtype(&input)?;
    let output_dtype = node.config.dtype.unwrap_or(input_dtype);
    let mask = DynBool::R2(
        Tensor::<2, Bool>::diag_mask([*rows, *columns], node.config.k, device).bool_not(),
    );
    let output = if output_dtype.is_float() {
        Value::Tensor(mask.to_float(output_dtype))
    } else if output_dtype.is_int() || output_dtype.is_uint() {
        Value::Int(mask.to_int(output_dtype))
    } else if output_dtype.is_bool() {
        Value::Bool(mask)
    } else {
        return Err(TynxError::TypeMismatch(format!(
            "EyeLike output dtype {output_dtype:?} is unsupported"
        )));
    };
    Ok(vec![output])
}

fn value_dtype(value: &Value) -> Result<DType> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.dtype()),
        Value::Int(tensor) => Ok(tensor.dtype()),
        Value::Bool(_) => Ok(DType::Bool(burn::tensor::BoolStore::Native)),
        other => Err(TynxError::TypeMismatch(format!(
            "EyeLike expects a tensor, got {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::eye_like::{EyeLikeConfig, EyeLikeNodeBuilder},
    };

    use super::*;

    #[test]
    fn creates_an_offset_integer_diagonal() {
        let node = EyeLikeNodeBuilder::new("eye_like")
            .input_tensor("x", 2, DType::F32)
            .output_tensor("y", 2, DType::I32)
            .config(EyeLikeConfig {
                dtype: Some(DType::I32),
                k: -1,
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(TensorData::new(vec![0.0_f32; 12], [3, 4]), 2, &device)
                .unwrap(),
        );

        let output = eye_like(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_int()
            .unwrap();

        assert_eq!(output.dtype(), DType::I32);
        assert_eq!(
            output.into_data().iter::<i64>().collect::<Vec<_>>(),
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        );
    }
}
