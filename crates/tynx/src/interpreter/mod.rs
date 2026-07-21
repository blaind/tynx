//! Runtime dispatch for individual ONNX nodes.

mod binary;
mod resolve;
mod unary;

use std::collections::HashMap;

use burn::tensor::Device;
use onnx_ir::ir::Node;

use crate::{Result, TynxError, Value};

/// Values available to nodes, keyed by ONNX argument name.
pub type Env = HashMap<String, Value>;

/// Execute one ONNX node using values from the runtime environment.
pub fn execute(node: &Node, env: &Env, device: &Device) -> Result<Vec<Value>> {
    match node {
        Node::Add(node) => binary::add(node, env, device),
        Node::Identity(node) => Ok(vec![resolve::first(env, &node.name, &node.inputs, device)?]),
        Node::Mul(node) => binary::mul(node, env, device),
        Node::Relu(node) => unary::relu(node, env, device),
        Node::Sigmoid(node) => unary::sigmoid(node, env, device),
        Node::Sub(node) => binary::sub(node, env, device),
        _ => Err(TynxError::UnsupportedOp(format!("node '{}'", node.name()))),
    }
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
}
