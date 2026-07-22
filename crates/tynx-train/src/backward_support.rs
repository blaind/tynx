//! Conservative ONNX backward-capability registry.

use tynx_core::onnx_ir::{Node, ir::ArgType};

/// Backward behavior for one operator input relative to one output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackwardCapability {
    /// Gradients can propagate through this input.
    Differentiable,
    /// This input deliberately terminates gradient propagation.
    StopGradient(&'static str),
    /// Tynx has no validated backward contract for this input yet.
    Unsupported(&'static str),
}

impl BackwardCapability {
    /// Return whether gradients can propagate through this input.
    pub fn is_differentiable(self) -> bool {
        matches!(self, Self::Differentiable)
    }

    /// Return the restriction reason for a blocked input.
    pub fn reason(self) -> Option<&'static str> {
        match self {
            Self::Differentiable => None,
            Self::StopGradient(reason) | Self::Unsupported(reason) => Some(reason),
        }
    }
}

/// Explicit registry for the backward surface validated by imported-model training.
///
/// Inference support is intentionally not used as a proxy for this registry. Operators are added
/// here only after their data-input gradient behavior is understood and covered by training tests.
#[derive(Debug, Clone, Copy, Default)]
pub struct BackwardSupportRegistry;

impl BackwardSupportRegistry {
    /// Return the backward capability of `input_index` for `output_index`.
    pub fn input_capability(
        node: &Node,
        output_index: usize,
        input_index: usize,
    ) -> BackwardCapability {
        let Some(input) = node.inputs().get(input_index) else {
            return BackwardCapability::Unsupported("input position does not exist");
        };
        if output_index >= node.outputs().len() {
            return BackwardCapability::Unsupported("output position does not exist");
        }

        let differentiable = BackwardCapability::Differentiable;
        let metadata = BackwardCapability::StopGradient(
            "integer, boolean, shape, or semantic metadata is not differentiable",
        );
        let unsupported = BackwardCapability::Unsupported(
            "operator input has no validated Tynx backward contract",
        );

        if !argument_is_float(&input.ty) {
            return metadata;
        }
        if !argument_is_float(&node.outputs()[output_index].ty) {
            return BackwardCapability::StopGradient(
                "integer, boolean, or shape-valued output is not differentiable",
            );
        }

        match node {
            // Elementwise arithmetic and smooth unary operations.
            Node::Add(_) | Node::Sub(_) | Node::Mul(_) | Node::Div(_) => differentiable,
            Node::Neg(_)
            | Node::Abs(_)
            | Node::Pow(_)
            | Node::Reciprocal(_)
            | Node::Sqrt(_)
            | Node::Exp(_)
            | Node::Log(_)
            | Node::Erf(_)
            | Node::Sin(_)
            | Node::Cos(_)
            | Node::Tan(_)
            | Node::Sinh(_)
            | Node::Cosh(_)
            | Node::Tanh(_)
            | Node::Relu(_)
            | Node::Sigmoid(_)
            | Node::Softmax(_)
            | Node::LogSoftmax(_)
            | Node::LeakyRelu(_)
            | Node::Elu(_)
            | Node::Selu(_)
            | Node::Celu(_)
            | Node::Gelu(_)
            | Node::Mish(_)
            | Node::Softplus(_)
            | Node::Softsign(_)
            | Node::HardSigmoid(_)
            | Node::HardSwish(_)
            | Node::Identity(_)
                if input_index == 0 =>
            {
                differentiable
            }
            Node::PRelu(_) if matches!(input_index, 0 | 1) => differentiable,

            // Selection and reduction. Conditions, axes, and indices remain metadata.
            Node::Where(_) if matches!(input_index, 1 | 2) => differentiable,
            Node::ReduceMax(_)
            | Node::ReduceMin(_)
            | Node::ReduceMean(_)
            | Node::ReduceSum(_)
            | Node::ReduceProd(_)
            | Node::ReduceL1(_)
            | Node::ReduceL2(_)
            | Node::ReduceLogSum(_)
            | Node::ReduceLogSumExp(_)
            | Node::ReduceSumSquare(_)
                if input_index == 0 =>
            {
                differentiable
            }
            Node::Max(_) | Node::Min(_) | Node::Mean(_) | Node::Sum(_) => differentiable,

            // Shape-preserving/rearrangement data inputs.
            Node::Concat(_) => differentiable,
            Node::Expand(_)
            | Node::Flatten(_)
            | Node::Pad(_)
            | Node::Reshape(_)
            | Node::Slice(_)
            | Node::Split(_)
            | Node::Squeeze(_)
            | Node::Tile(_)
            | Node::Transpose(_)
            | Node::Unsqueeze(_)
            | Node::DepthToSpace(_)
            | Node::SpaceToDepth(_)
                if input_index == 0 =>
            {
                differentiable
            }
            Node::Gather(_) | Node::GatherElements(_) | Node::GatherND(_) if input_index == 0 => {
                differentiable
            }

            // Dense and convolutional layers.
            Node::MatMul(_) | Node::Gemm(_) | Node::Linear(_) => differentiable,
            Node::Conv1d(_)
            | Node::Conv2d(_)
            | Node::Conv3d(_)
            | Node::ConvTranspose1d(_)
            | Node::ConvTranspose2d(_)
            | Node::ConvTranspose3d(_)
                if matches!(input_index, 0..=2) =>
            {
                differentiable
            }

            // Pooling and fixed-stat normalization.
            Node::AveragePool1d(_)
            | Node::AveragePool2d(_)
            | Node::AveragePool3d(_)
            | Node::MaxPool1d(_)
            | Node::MaxPool2d(_)
            | Node::MaxPool3d(_)
            | Node::GlobalAveragePool(_)
                if input_index == 0 =>
            {
                differentiable
            }
            Node::BatchNormalization(_) if matches!(input_index, 0..=2) => differentiable,
            Node::BatchNormalization(_) if matches!(input_index, 3 | 4) => {
                BackwardCapability::StopGradient("fixed BatchNorm running statistics are buffers")
            }
            Node::InstanceNormalization(_)
            | Node::LayerNormalization(_)
            | Node::GroupNormalization(_)
                if matches!(input_index, 0..=2) =>
            {
                differentiable
            }

            // Inference-mode Dropout is identity. A supplied training-mode input is not accepted
            // by the v1 trainable-import contract, even when it is only known at runtime.
            Node::Dropout(node) if input_index == 0 && dropout_is_inference_only(node) => {
                differentiable
            }
            Node::Dropout(_) if input_index == 0 => BackwardCapability::Unsupported(
                "Dropout training_mode must be absent or statically false",
            ),

            // Discrete operators explicitly terminate gradients instead of masquerading as
            // unsupported implementation work.
            Node::Ceil(_)
            | Node::Floor(_)
            | Node::Round(_)
            | Node::Sign(_)
            | Node::Equal(_)
            | Node::Greater(_)
            | Node::GreaterOrEqual(_)
            | Node::Less(_)
            | Node::LessOrEqual(_)
            | Node::ArgMax(_)
            | Node::ArgMin(_)
            | Node::Shape(_)
            | Node::Size(_)
            | Node::NonZero(_)
            | Node::TopK(_)
                if input_index == 0 =>
            {
                BackwardCapability::StopGradient(
                    "operator is discrete or shape-valued and deliberately stops gradients",
                )
            }
            _ => unsupported,
        }
    }
}

fn argument_is_float(argument: &ArgType) -> bool {
    match argument {
        ArgType::ScalarTensor(dtype) | ArgType::ScalarNative(dtype) => dtype.is_float(),
        ArgType::Tensor(tensor) => tensor.dtype.is_float(),
        ArgType::Shape(_) => false,
    }
}

fn dropout_is_inference_only(node: &tynx_core::onnx_ir::node::dropout::DropoutNode) -> bool {
    let Some(training_mode) = node.inputs.get(2) else {
        return true;
    };
    if training_mode.is_optional() {
        return true;
    }
    training_mode
        .value()
        .and_then(|value| {
            value
                .as_slice::<bool>()
                .ok()
                .and_then(|values| values.first().copied())
        })
        .is_some_and(|training| !training)
}
