//! Rank-preserving unary operators.

use burn::tensor::Device;
use onnx_ir::node::relu::ReluNode;

use super::{Env, resolve};
use crate::{Result, Value};

pub(super) fn relu(node: &ReluNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(input.relu())])
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{DType, node::relu::ReluNodeBuilder};

    use super::*;

    #[test]
    fn clamps_negative_values() {
        let node = ReluNodeBuilder::new("relu")
            .input_tensor("x", 2, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .build();
        let device = Device::default();
        let input = Value::from_tensor_data(
            TensorData::new(vec![-2.0_f32, 0.0, 3.0, -4.0], [2, 2]),
            2,
            &device,
        )
        .unwrap();
        let mut env = Env::new();
        env.insert("x".to_string(), input);

        let output = relu(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data();

        assert_eq!(
            output.iter::<f32>().collect::<Vec<_>>(),
            [0.0, 0.0, 3.0, 0.0]
        );
    }
}
