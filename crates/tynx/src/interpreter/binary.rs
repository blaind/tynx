//! Element-wise binary operators.

use burn::tensor::Device;
use onnx_ir::node::arithmetic::{AddNode, SubNode};

use super::{Env, resolve};
use crate::{Result, Value};

pub(super) fn add(node: &AddNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let left = resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?;
    let right = resolve::at(env, &node.name, &node.inputs, 1, device)?.into_tensor()?;

    Ok(vec![Value::Tensor(left.add_broadcast(right)?)])
}

pub(super) fn sub(node: &SubNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let left = resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?;
    let right = resolve::at(env, &node.name, &node.inputs, 1, device)?.into_tensor()?;

    Ok(vec![Value::Tensor(left.sub_broadcast(right)?)])
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::arithmetic::{AddNodeBuilder, SubNodeBuilder},
    };

    use super::*;

    #[test]
    fn adds_with_multidirectional_broadcasting() {
        let node = AddNodeBuilder::new("add")
            .input_tensor("matrix", 2, DType::F32)
            .input_tensor("row", 1, DType::F32)
            .output_tensor("sum", 2, DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "matrix".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]),
                2,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "row".to_string(),
            Value::from_tensor_data(TensorData::new(vec![10.0_f32, 20.0], [2]), 1, &device)
                .unwrap(),
        );

        let output = add(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn subtracts_in_operand_order() {
        let node = SubNodeBuilder::new("sub")
            .input_tensor("matrix", 2, DType::F32)
            .input_tensor("row", 1, DType::F32)
            .output_tensor("difference", 2, DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "matrix".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![10.0_f32, 20.0, 30.0, 40.0], [2, 2]),
                2,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "row".to_string(),
            Value::from_tensor_data(TensorData::new(vec![1.0_f32, 2.0], [2]), 1, &device).unwrap(),
        );

        let output = sub(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [9.0, 18.0, 29.0, 38.0]);
    }
}
