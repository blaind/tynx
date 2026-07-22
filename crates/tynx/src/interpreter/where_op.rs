//! ONNX Where execution.

use burn::tensor::{Device, TensorData};
use onnx_ir::node::where_op::WhereNode;

use super::{Env, resolve};
use crate::{DynBool, DynInt, DynTensor, Result, Scalar, TynxError, Value};

pub(super) fn where_op(node: &WhereNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let condition = resolve::at(env, &node.name, &node.inputs, 0, device)?.into_bool()?;
    let then = resolve::at(env, &node.name, &node.inputs, 1, device)?;
    let otherwise = resolve::at(env, &node.name, &node.inputs, 2, device)?;

    let output = match (then, otherwise) {
        (Value::Tensor(then), Value::Tensor(otherwise)) => {
            Value::Tensor(DynTensor::where_select(condition, then, otherwise)?)
        }
        (Value::Tensor(then), Value::Scalar(otherwise)) => {
            let dtype = then.dtype();
            Value::Tensor(DynTensor::where_select(
                condition,
                then,
                DynTensor::full(&[1], otherwise.as_f64(), device, dtype)?,
            )?)
        }
        (Value::Scalar(then), Value::Tensor(otherwise)) => {
            let dtype = otherwise.dtype();
            Value::Tensor(DynTensor::where_select(
                condition,
                DynTensor::full(&[1], then.as_f64(), device, dtype)?,
                otherwise,
            )?)
        }
        (Value::Int(then), Value::Int(otherwise)) => {
            Value::Int(DynInt::where_select(condition, then, otherwise)?)
        }
        (Value::Int(then), Value::Scalar(otherwise)) => {
            let dtype = then.dtype();
            Value::Int(DynInt::where_select(
                condition,
                then,
                scalar_int_tensor(otherwise, dtype, device)?,
            )?)
        }
        (Value::Scalar(then), Value::Int(otherwise)) => {
            let dtype = otherwise.dtype();
            Value::Int(DynInt::where_select(
                condition,
                scalar_int_tensor(then, dtype, device)?,
                otherwise,
            )?)
        }
        (Value::Shape(then), Value::Int(otherwise)) => Value::Int(DynInt::where_select(
            condition,
            shape_tensor(then, device)?,
            otherwise,
        )?),
        (Value::Int(then), Value::Shape(otherwise)) => Value::Int(DynInt::where_select(
            condition,
            then,
            shape_tensor(otherwise, device)?,
        )?),
        (Value::Bool(then), Value::Bool(otherwise)) => {
            Value::Bool(DynBool::where_select(condition, then, otherwise)?)
        }
        (then, otherwise) => {
            return Err(TynxError::TypeMismatch(format!(
                "Where branches must have matching tensor kinds, got {then:?} and {otherwise:?}"
            )));
        }
    };

    Ok(vec![output])
}

fn shape_tensor(values: Vec<i64>, device: &Device) -> Result<DynInt> {
    let length = values.len();
    DynInt::from_data(TensorData::new(values, [length]), 1, device)
}

fn scalar_int_tensor(
    scalar: Scalar,
    dtype: burn::tensor::DType,
    device: &Device,
) -> Result<DynInt> {
    let data = if dtype.is_uint() {
        TensorData::new(vec![scalar.as_f64() as u64], [1])
    } else {
        TensorData::new(vec![scalar.as_f64() as i64], [1])
    };
    Ok(DynInt::from_data(data, 1, device)?.cast(dtype))
}

#[cfg(test)]
mod tests {
    use burn::tensor::{BoolStore, TensorData};
    use onnx_ir::{DType, node::where_op::WhereNodeBuilder};

    use super::*;

    #[test]
    fn selects_values_with_broadcasting() {
        let node = WhereNodeBuilder::new("where")
            .input_tensor("condition", 1, DType::Bool(BoolStore::Native))
            .input_tensor("then", 2, DType::F32)
            .input_tensor("otherwise", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "condition".to_string(),
            Value::from_tensor_data(TensorData::new(vec![true, false], [2]), 1, &device).unwrap(),
        );
        env.insert(
            "then".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]),
                2,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "otherwise".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![10.0_f32, 20.0, 30.0, 40.0], [2, 2]),
                2,
                &device,
            )
            .unwrap(),
        );

        let output = where_op(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    fn selects_tensor_and_scalar_branches() {
        let node = WhereNodeBuilder::new("where")
            .input_tensor("condition", 1, DType::Bool(BoolStore::Native))
            .input_tensor("then", 1, DType::F32)
            .input_scalar("otherwise", DType::F32)
            .output_tensor("output", 1, DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "condition".to_string(),
            Value::from_tensor_data(TensorData::new(vec![true, false], [2]), 1, &device).unwrap(),
        );
        env.insert(
            "then".to_string(),
            Value::from_tensor_data(TensorData::new(vec![1.0_f32, 2.0], [2]), 1, &device).unwrap(),
        );
        env.insert("otherwise".to_string(), Value::Scalar(Scalar::F64(10.0)));

        let output = where_op(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [1.0, 10.0]);
    }
}
