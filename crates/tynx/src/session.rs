//! Loading and inspecting ONNX models.

use std::path::Path;

use onnx_ir::OnnxGraphBuilder;
use onnx_ir::ir::{Argument, OnnxGraph};

use crate::error::Result;

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
        let graph = OnnxGraphBuilder::new()
            .simplify(simplify)
            .parse_file(path)?;

        Ok(Self { graph })
    }

    /// Load a model from bytes and simplify its graph.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Self::from_bytes_with(data, true)
    }

    /// Load a model from bytes with optional graph simplification.
    pub fn from_bytes_with(data: &[u8], simplify: bool) -> Result<Self> {
        let graph = OnnxGraphBuilder::new()
            .simplify(simplify)
            .parse_bytes(data)?;

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
}

#[cfg(test)]
mod tests {
    use crate::TynxError;

    use super::*;

    #[test]
    fn reports_invalid_model_bytes() {
        let error = Session::from_bytes(b"not an ONNX model").unwrap_err();

        assert!(matches!(error, TynxError::Parse(_)));
    }
}
