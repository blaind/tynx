//! Inference-mode Dropout execution.

use burn::tensor::Device;
use onnx_ir::node::dropout::DropoutNode;

use super::{Env, resolve, shape};
use crate::{DynBool, Result, Scalar, Value};

pub(super) fn dropout(node: &DropoutNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let output = resolve::first(env, &node.name, &node.inputs, device)?;
    if node.outputs.len() == 1 {
        return Ok(vec![output]);
    }

    let dims = shape::value_dims(&output);
    let mask = if dims.is_empty() {
        Value::Scalar(Scalar::Bool(true))
    } else {
        Value::Bool(DynBool::full(&dims, true, device)?)
    };
    Ok(vec![output, mask])
}

#[cfg(test)]
mod tests {
    use burn::tensor::{BoolStore, TensorData};
    use onnx_ir::{
        DType,
        node::dropout::{DropoutConfig, DropoutInput, DropoutNodeBuilder},
    };

    use super::*;

    #[test]
    fn inference_returns_input_and_all_true_mask() {
        let node = DropoutNodeBuilder::new("dropout")
            .input_tensor("x", 2, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .output_tensor("mask", 2, DType::Bool(BoolStore::Native))
            .config(DropoutConfig::new(DropoutInput::Static(0.75)))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(vec![0.0_f32, 2.0, -3.0, 4.0], [2, 2]),
                2,
                &device,
            )
            .unwrap(),
        );

        let outputs = dropout(&node, &env, &device).unwrap();
        let Value::Tensor(output) = outputs[0].clone() else {
            panic!("expected tensor output");
        };
        let Value::Bool(mask) = outputs[1].clone() else {
            panic!("expected mask output");
        };

        assert_eq!(
            output.into_data().iter::<f32>().collect::<Vec<_>>(),
            [0.0, 2.0, -3.0, 4.0]
        );
        assert_eq!(
            mask.into_data().iter::<bool>().collect::<Vec<_>>(),
            [true; 4]
        );
    }
}
