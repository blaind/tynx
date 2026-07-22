//! Inference-mode Dropout execution.

use burn::tensor::Device;
use onnx_ir::node::dropout::{DropoutInput, DropoutNode};

use super::{Env, resolve, shape};
use crate::{DynBool, Result, Scalar, TynxError, Value};

pub(super) fn dropout(node: &DropoutNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    if training_mode(node, env, device)? && ratio(node, env, device)? != 0.0 {
        return Err(TynxError::UnsupportedOp(
            "Dropout training mode".to_string(),
        ));
    }
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

fn training_mode(node: &DropoutNode, env: &Env, device: &Device) -> Result<bool> {
    if !node.inputs.get(2).is_some_and(|input| !input.is_optional()) {
        return Ok(false);
    }
    match resolve::at(env, &node.name, &node.inputs, 2, device)? {
        Value::Scalar(Scalar::Bool(value)) => Ok(value),
        other => Err(TynxError::TypeMismatch(format!(
            "Dropout training_mode must be a boolean scalar, got {other:?}"
        ))),
    }
}

fn ratio(node: &DropoutNode, env: &Env, device: &Device) -> Result<f64> {
    match &node.config.prob {
        DropoutInput::Static(value) => Ok(*value),
        DropoutInput::Runtime(input) => {
            match resolve::at(env, &node.name, &node.inputs, input.input_index, device)? {
                Value::Scalar(value) => Ok(value.as_f64()),
                other => Err(TynxError::TypeMismatch(format!(
                    "Dropout ratio must be a numeric scalar, got {other:?}"
                ))),
            }
        }
    }
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

    #[test]
    fn rejects_nonzero_training_dropout() {
        let node = DropoutNodeBuilder::new("dropout")
            .input_tensor("x", 1, DType::F32)
            .input_scalar("ratio", DType::F32)
            .input_scalar("training", DType::Bool(BoolStore::Native))
            .output_tensor("y", 1, DType::F32)
            .config(DropoutConfig::new(DropoutInput::Static(0.5)))
            .build();
        let mut env = Env::new();
        env.insert("training".into(), Value::Scalar(Scalar::Bool(true)));

        let error = dropout(&node, &env, &Device::default()).unwrap_err();

        assert_eq!(
            error,
            TynxError::UnsupportedOp("Dropout training mode".to_string())
        );
    }
}
