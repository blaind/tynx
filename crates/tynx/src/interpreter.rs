//! Runtime dispatch for individual ONNX nodes.

use std::collections::HashMap;

use onnx_ir::ir::{Argument, Node};

use crate::{Result, TynxError, Value};

/// Values available to nodes, keyed by ONNX argument name.
pub type Env = HashMap<String, Value>;

/// Execute one ONNX node using values from the runtime environment.
pub fn execute(node: &Node, env: &Env) -> Result<Vec<Value>> {
    match node {
        Node::Identity(_) => Ok(vec![lookup(env, first_input(node)?)?.clone()]),
        _ => Err(TynxError::UnsupportedOp(format!("node '{}'", node.name()))),
    }
}

fn first_input(node: &Node) -> Result<&Argument> {
    node.inputs()
        .first()
        .ok_or_else(|| TynxError::Shape(format!("node '{}' has no input", node.name())))
}

fn lookup<'a>(env: &'a Env, argument: &Argument) -> Result<&'a Value> {
    env.get(&argument.name)
        .ok_or_else(|| TynxError::MissingValue(argument.name.clone()))
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

        let outputs = execute(&node, &env).unwrap();

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

        let error = execute(&node, &Env::new()).unwrap_err();

        assert_eq!(error, TynxError::MissingValue("missing".to_string()));
    }
}
