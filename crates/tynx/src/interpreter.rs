//! Runtime dispatch for individual ONNX nodes.

use std::collections::HashMap;

use burn::tensor::Device;
use onnx_ir::ir::{Argument, Node};

use crate::{Result, TynxError, Value};

/// Values available to nodes, keyed by ONNX argument name.
pub type Env = HashMap<String, Value>;

/// Execute one ONNX node using values from the runtime environment.
pub fn execute(node: &Node, env: &Env, device: &Device) -> Result<Vec<Value>> {
    match node {
        Node::Identity(_) => Ok(vec![resolve(env, first_input(node)?, device)?]),
        _ => Err(TynxError::UnsupportedOp(format!("node '{}'", node.name()))),
    }
}

fn first_input(node: &Node) -> Result<&Argument> {
    node.inputs()
        .first()
        .ok_or_else(|| TynxError::Shape(format!("node '{}' has no input", node.name())))
}

fn resolve(env: &Env, argument: &Argument, device: &Device) -> Result<Value> {
    if let Some(value) = env.get(&argument.name) {
        return Ok(value.clone());
    }
    if let Some(data) = argument.value() {
        return Value::from_tensor_data(data, argument.ty.rank(), device);
    }

    let name = if argument.name.is_empty() {
        "<static/absent>"
    } else {
        &argument.name
    };
    Err(TynxError::MissingValue(name.to_string()))
}

#[cfg(test)]
mod tests {
    use onnx_ir::{DType, Node, node::identity::IdentityNodeBuilder};

    use super::*;
    use crate::Scalar;

    #[test]
    fn identity_returns_its_input() {
        let node = Node::Identity(
            IdentityNodeBuilder::new("identity")
                .input_scalar("x", DType::I64)
                .output_scalar("y", DType::I64)
                .build(),
        );
        let mut env = Env::new();
        env.insert("x".to_string(), Value::Scalar(Scalar::I64(42)));

        let outputs = execute(&node, &env, &Device::default()).unwrap();

        assert!(matches!(
            outputs.as_slice(),
            [Value::Scalar(Scalar::I64(42))]
        ));
    }

    #[test]
    fn reports_a_missing_identity_input() {
        let node = Node::Identity(
            IdentityNodeBuilder::new("identity")
                .input_scalar("missing", DType::I64)
                .output_scalar("y", DType::I64)
                .build(),
        );

        let error = execute(&node, &Env::new(), &Device::default()).unwrap_err();

        assert_eq!(error, TynxError::MissingValue("missing".to_string()));
    }

    #[test]
    fn identity_materializes_a_static_input() {
        let node = Node::Identity(
            IdentityNodeBuilder::new("identity")
                .input_const_i64("constant", 42)
                .output_scalar("y", DType::I64)
                .build(),
        );

        let outputs = execute(&node, &Env::new(), &Device::default()).unwrap();

        assert!(matches!(
            outputs.as_slice(),
            [Value::Scalar(Scalar::I64(42))]
        ));
    }
}
