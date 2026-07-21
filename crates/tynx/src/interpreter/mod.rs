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
        Node::Abs(node) => unary::abs(node, env, device),
        Node::Acos(node) => unary::acos(node, env, device),
        Node::Acosh(node) => unary::acosh(node, env, device),
        Node::Add(node) => binary::add(node, env, device),
        Node::Asin(node) => unary::asin(node, env, device),
        Node::Asinh(node) => unary::asinh(node, env, device),
        Node::Atan(node) => unary::atan(node, env, device),
        Node::Atanh(node) => unary::atanh(node, env, device),
        Node::Ceil(node) => unary::ceil(node, env, device),
        Node::Cos(node) => unary::cos(node, env, device),
        Node::Cosh(node) => unary::cosh(node, env, device),
        Node::Div(node) => binary::div(node, env, device),
        Node::Erf(node) => unary::erf(node, env, device),
        Node::Exp(node) => unary::exp(node, env, device),
        Node::Identity(node) => Ok(vec![resolve::first(env, &node.name, &node.inputs, device)?]),
        Node::Log(node) => unary::log(node, env, device),
        Node::Mul(node) => binary::mul(node, env, device),
        Node::Neg(node) => unary::neg(node, env, device),
        Node::Relu(node) => unary::relu(node, env, device),
        Node::Sigmoid(node) => unary::sigmoid(node, env, device),
        Node::Sin(node) => unary::sin(node, env, device),
        Node::Sinh(node) => unary::sinh(node, env, device),
        Node::Sqrt(node) => unary::sqrt(node, env, device),
        Node::Sub(node) => binary::sub(node, env, device),
        Node::Tan(node) => unary::tan(node, env, device),
        Node::Tanh(node) => unary::tanh(node, env, device),
        _ => Err(TynxError::UnsupportedOp(operator_kind(node))),
    }
}

fn operator_kind(node: &Node) -> String {
    node.to_string()
        .split_whitespace()
        .next()
        .unwrap_or("unknown")
        .to_string()
}

#[cfg(test)]
mod tests {
    use onnx_ir::{
        DType, Node,
        node::{floor::FloorNodeBuilder, identity::IdentityNodeBuilder},
    };

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
    fn unsupported_errors_name_the_operator() {
        let node = Node::Floor(
            FloorNodeBuilder::new("")
                .input_tensor("x", 1, DType::F32)
                .output_tensor("y", 1, DType::F32)
                .build(),
        );

        let error = execute(&node, &Env::new(), &Device::default()).unwrap_err();

        assert_eq!(error, TynxError::UnsupportedOp("Floor".to_string()));
    }
}
