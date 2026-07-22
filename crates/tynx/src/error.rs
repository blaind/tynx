//! Errors returned while loading and running models.

/// Errors returned by Tynx.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TynxError {
    /// An ONNX model could not be parsed or type-checked.
    #[error("failed to parse or type-check ONNX: {0}")]
    Parse(String),

    /// The model contains an operator that Tynx does not support.
    #[error("unsupported operator: {0}")]
    UnsupportedOp(String),

    /// A tensor rank exceeds the maximum supported rank.
    #[error("tensor rank {rank} exceeds the maximum supported rank {max}")]
    RankOverflow {
        /// Requested tensor rank.
        rank: usize,
        /// Highest rank implemented by the runtime.
        max: usize,
    },

    /// A tensor could not be promoted to the requested rank.
    #[error("cannot promote rank {from} to {to}")]
    RankPromote { from: usize, to: usize },

    /// A named value was not present in the runtime environment.
    #[error("value '{0}' not found in environment")]
    MissingValue(String),

    /// A prepared session received a tensor allocated on another device.
    #[error("input '{name}' is on {actual}, but the prepared session uses {expected}")]
    DeviceMismatch {
        /// Input name.
        name: String,
        /// Prepared device.
        expected: String,
        /// Input tensor device.
        actual: String,
    },

    /// A value had an unexpected type.
    #[error("type mismatch: {0}")]
    TypeMismatch(String),

    /// A shape or index computation was invalid.
    #[error("shape error: {0}")]
    Shape(String),
}

impl From<onnx_ir::Error> for TynxError {
    fn from(error: onnx_ir::Error) -> Self {
        Self::Parse(error.to_string())
    }
}

/// A result returned by Tynx.
pub type Result<T> = std::result::Result<T, TynxError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_context() {
        let error = TynxError::RankPromote { from: 4, to: 2 };

        assert_eq!(error.to_string(), "cannot promote rank 4 to 2");

        let error = TynxError::RankOverflow { rank: 7, max: 6 };
        assert_eq!(
            error.to_string(),
            "tensor rank 7 exceeds the maximum supported rank 6"
        );
    }

    #[test]
    fn converts_onnx_errors() {
        let error = TynxError::from(onnx_ir::Error::MissingOpsetVersion);

        assert_eq!(
            error,
            TynxError::Parse(
                "ONNX model must specify opset version for default domain".to_string()
            )
        );
    }
}
