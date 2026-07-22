//! Loading and inspecting ONNX models.

use std::{fs, path::Path};

use burn::tensor::Device;
use onnx_ir::OnnxGraphBuilder;
use onnx_ir::ir::{Argument, OnnxGraph};

use crate::{Env, Result, TynxError, execute};

/// A parsed ONNX model.
#[derive(Debug, Clone)]
pub struct Session {
    graph: OnnxGraph,
}

impl Session {
    /// Load a model from a file and simplify its graph.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_file_with(path, true)
    }

    /// Load a model from a file with optional graph simplification.
    pub fn from_file_with(path: impl AsRef<Path>, simplify: bool) -> Result<Self> {
        let data = fs::read(path).map_err(|error| TynxError::Parse(error.to_string()))?;
        Self::from_bytes_with(&data, simplify)
    }

    /// Load a model from bytes and simplify its graph.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Self::from_bytes_with(data, true)
    }

    /// Load a model from bytes with optional graph simplification.
    pub fn from_bytes_with(data: &[u8], simplify: bool) -> Result<Self> {
        let (prepared, changed) = crate::interpreter::prepare_model(data)?;
        let parse_data = if changed { prepared.as_slice() } else { data };
        let mut graph = OnnxGraphBuilder::new()
            .simplify(simplify)
            .parse_bytes(parse_data)?;
        if changed {
            crate::interpreter::restore_dynamic_conv_inputs(data, &mut graph)?;
        }
        crate::interpreter::preserve_attributes(data, &mut graph)?;

        Ok(Self { graph })
    }

    /// Return the parsed graph.
    pub fn graph(&self) -> &OnnxGraph {
        &self.graph
    }

    /// Return the model's declared inputs.
    pub fn inputs(&self) -> &[Argument] {
        &self.graph.inputs
    }

    /// Return the model's declared outputs.
    pub fn outputs(&self) -> &[Argument] {
        &self.graph.outputs
    }

    /// Run inference and return the graph outputs by name.
    pub fn run(&self, device: &Device, mut env: Env) -> Result<Env> {
        for node in &self.graph.nodes {
            let values = execute(node, &env, device)?;
            if values.len() != node.outputs().len() {
                return Err(TynxError::Shape(format!(
                    "node '{}' returned {} values for {} outputs",
                    node.name(),
                    values.len(),
                    node.outputs().len()
                )));
            }

            for (output, value) in node.outputs().iter().zip(values) {
                env.insert(output.name.clone(), value);
            }
        }

        self.graph
            .outputs
            .iter()
            .map(|output| {
                let value = env
                    .get(&output.name)
                    .cloned()
                    .ok_or_else(|| TynxError::MissingValue(output.name.clone()))?;
                Ok((output.name.clone(), value))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use onnx_ir::{DType, Node, node::identity::IdentityNodeBuilder};

    use crate::{Scalar, TynxError, Value};

    use super::*;

    #[test]
    fn reports_invalid_model_bytes() {
        let error = Session::from_bytes(b"not an ONNX model").unwrap_err();

        assert!(matches!(error, TynxError::Parse(_)));
    }

    #[test]
    fn runs_an_identity_graph() {
        let identity = IdentityNodeBuilder::new("identity")
            .input_scalar("x", DType::I64)
            .output_scalar("y", DType::I64)
            .build();
        let mut graph = OnnxGraph::default();
        graph.inputs = identity.inputs.clone();
        graph.outputs = identity.outputs.clone();
        graph.nodes.push(Node::Identity(identity));
        let session = Session { graph };
        let mut inputs = Env::new();
        inputs.insert("x".to_string(), Value::Scalar(Scalar::I64(42)));

        let outputs = session.run(&Device::default(), inputs).unwrap();

        assert!(matches!(
            outputs.get("y"),
            Some(Value::Scalar(Scalar::I64(42)))
        ));
    }
}
