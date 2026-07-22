//! Stable identities for values embedded in a processed ONNX graph.

use onnx_ir::ir::{Argument, ValueSource};

/// Stable identity for an initializer in a processed ONNX graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum InitializerId {
    /// A named constant value.
    Named(String),
    /// A lifted static tensor identified by the graph's data ID.
    Static(usize),
    /// An unnamed constant scoped to its first consumer position.
    Unnamed {
        /// Index of the consuming node.
        node_index: usize,
        /// Input position within the consuming node.
        input_index: usize,
    },
}

impl InitializerId {
    /// Identify an embedded graph input, or return `None` for runtime and optional inputs.
    pub fn from_argument(input: &Argument, node_index: usize, input_index: usize) -> Option<Self> {
        match input.value_source {
            ValueSource::Static(data_id) => Some(Self::Static(data_id)),
            ValueSource::Constant if !input.name.is_empty() => {
                Some(Self::Named(input.name.clone()))
            }
            ValueSource::Constant => Some(Self::Unnamed {
                node_index,
                input_index,
            }),
            ValueSource::Dynamic | ValueSource::Optional => None,
        }
    }
}

pub(crate) fn env_key(id: &InitializerId) -> String {
    match id {
        InitializerId::Named(name) => format!("\0tynx:constant:{name}"),
        InitializerId::Static(data_id) => format!("\0tynx:static:{data_id}"),
        InitializerId::Unnamed { input_index, .. } => {
            format!("\0tynx:current-constant:{input_index}")
        }
    }
}

#[cfg(test)]
mod tests {
    use onnx_ir::{ArgType, DType};

    use super::*;

    #[test]
    fn identifies_static_and_constant_values() {
        let mut static_input = Argument::new("", ArgType::ScalarNative(DType::I64));
        static_input.value_source = ValueSource::Static(7);
        let mut constant = Argument::new("weight", ArgType::ScalarNative(DType::I64));
        constant.value_source = ValueSource::Constant;

        assert_eq!(
            InitializerId::from_argument(&static_input, 3, 1),
            Some(InitializerId::Static(7))
        );
        assert_eq!(
            InitializerId::from_argument(&constant, 3, 2),
            Some(InitializerId::Named("weight".to_string()))
        );
    }
}
