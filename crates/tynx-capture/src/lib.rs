#![forbid(unsafe_code)]

//! Binding-neutral graph capture and replay.
//!
//! This crate owns the small runtime IR used by every future frontend. It intentionally contains
//! no Python callbacks: a frontend records tensor inputs, stable parameter-slot reads, and Tynx
//! operations once, then [`Graph::run`] replays the complete graph in Rust.

use std::collections::HashSet;

use tynx_core::{DType, Device, DynTensor, Result, TynxError};
use tynx_train::{ParamId, ParameterContract, ParameterSlot};

/// Identifier of a value produced by one graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(usize);

impl ValueId {
    /// Return the zero-based node index.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Exact tensor signature used by the first capture cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSignature {
    shape: Vec<usize>,
    dtype: DType,
    device: Device,
}

impl TensorSignature {
    /// Capture the shape, dtype, and device of a tensor.
    pub fn from_tensor(tensor: &DynTensor) -> Self {
        Self {
            shape: tensor.dims(),
            dtype: tensor.dtype(),
            device: tensor.device(),
        }
    }

    /// Return the exact captured shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the captured element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Return the captured execution device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    fn validate(&self, tensor: &DynTensor, label: &str) -> Result<()> {
        let actual = Self::from_tensor(tensor);
        if actual != *self {
            return Err(TynxError::TypeMismatch(format!(
                "captured {label} expected shape {:?}, dtype {:?}, and device {:?}, got shape {:?}, dtype {:?}, and device {:?}",
                self.shape, self.dtype, self.device, actual.shape, actual.dtype, actual.device,
            )));
        }
        Ok(())
    }
}

/// Unary operations supported by the initial runtime IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnaryOp {
    /// Rectified linear activation.
    Relu,
    /// Logistic activation.
    Sigmoid,
    /// Hyperbolic tangent activation.
    Tanh,
    /// Natural exponential.
    Exp,
    /// Natural logarithm.
    Log,
    /// Elementwise square root.
    Sqrt,
    /// Gaussian error linear unit.
    Gelu,
    /// Reshape without changing element order.
    Reshape(Vec<usize>),
    /// Permute all tensor axes.
    Permute(Vec<usize>),
}

/// Binary operations supported by the initial runtime IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// Broadcasted addition.
    Add,
    /// Broadcasted subtraction.
    Subtract,
    /// Broadcasted multiplication.
    Multiply,
    /// Broadcasted division.
    Divide,
    /// Matrix multiplication.
    Matmul,
}

#[derive(Debug, Clone)]
enum Node {
    Input(usize),
    Parameter(ParameterGuard),
    Unary {
        op: UnaryOp,
        input: ValueId,
    },
    Binary {
        op: BinaryOp,
        left: ValueId,
        right: ValueId,
    },
}

#[derive(Debug, Clone)]
struct ParameterGuard {
    slot: ParameterSlot,
    contract: ParameterContract,
    structure_generation: u64,
}

impl ParameterGuard {
    fn validate(&self) -> Result<()> {
        let actual_generation = self.slot.structure_generation();
        let actual_contract = self.slot.contract();
        if actual_generation != self.structure_generation || actual_contract != self.contract {
            return Err(TynxError::TypeMismatch(format!(
                "captured parameter {} changed structure: expected generation {} and contract {:?}, got generation {} and contract {:?}",
                self.slot.id().get(),
                self.structure_generation,
                self.contract,
                actual_generation,
                actual_contract,
            )));
        }
        Ok(())
    }
}

/// Builder used by a frontend while executing its first eager call.
#[derive(Debug, Default)]
pub struct GraphBuilder {
    nodes: Vec<Node>,
    input_signatures: Vec<TensorSignature>,
}

impl GraphBuilder {
    /// Create an empty graph builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one exact-signature runtime input.
    pub fn input(&mut self, tensor: &DynTensor) -> ValueId {
        let input = self.input_signatures.len();
        self.input_signatures
            .push(TensorSignature::from_tensor(tensor));
        self.push(Node::Input(input))
    }

    /// Record a read from a stable parameter slot.
    pub fn parameter(&mut self, slot: ParameterSlot) -> ValueId {
        let guard = ParameterGuard {
            contract: slot.contract(),
            structure_generation: slot.structure_generation(),
            slot,
        };
        self.push(Node::Parameter(guard))
    }

    /// Record a unary operation.
    pub fn unary(&mut self, op: UnaryOp, input: ValueId) -> Result<ValueId> {
        self.require_value(input)?;
        Ok(self.push(Node::Unary { op, input }))
    }

    /// Record a binary operation.
    pub fn binary(&mut self, op: BinaryOp, left: ValueId, right: ValueId) -> Result<ValueId> {
        self.require_value(left)?;
        self.require_value(right)?;
        Ok(self.push(Node::Binary { op, left, right }))
    }

    /// Finish the graph with one or more output values.
    pub fn finish(self, outputs: Vec<ValueId>) -> Result<Graph> {
        if outputs.is_empty() {
            return Err(TynxError::TypeMismatch(
                "a captured graph must have at least one output".to_string(),
            ));
        }
        for output in &outputs {
            self.require_value(*output)?;
        }
        Ok(Graph {
            nodes: self.nodes,
            input_signatures: self.input_signatures,
            outputs,
        })
    }

    fn push(&mut self, node: Node) -> ValueId {
        let id = ValueId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    fn require_value(&self, value: ValueId) -> Result<()> {
        if value.index() >= self.nodes.len() {
            return Err(TynxError::MissingValue(format!(
                "captured node {}",
                value.index()
            )));
        }
        Ok(())
    }
}

/// Immutable captured graph executable wholly in Rust.
#[derive(Debug, Clone)]
pub struct Graph {
    nodes: Vec<Node>,
    input_signatures: Vec<TensorSignature>,
    outputs: Vec<ValueId>,
}

impl Graph {
    /// Return exact input signatures in call order.
    pub fn input_signatures(&self) -> &[TensorSignature] {
        &self.input_signatures
    }

    /// Return the number of recorded operation and source nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return stable parameter slots read by this graph, deduplicated in first-use order.
    pub fn parameters(&self) -> Vec<ParameterSlot> {
        let mut seen = HashSet::<ParamId>::new();
        self.nodes
            .iter()
            .filter_map(|node| match node {
                Node::Parameter(guard) if seen.insert(guard.slot.id()) => Some(guard.slot.clone()),
                _ => None,
            })
            .collect()
    }

    /// Replay the graph with exact-signature inputs.
    ///
    /// When `tracking` is true, parameter nodes read the current generation's autodiff leaf.
    /// Ordinary optimizer value updates do not invalidate the graph; structural changes do.
    pub fn run(&self, inputs: &[DynTensor], tracking: bool) -> Result<Vec<DynTensor>> {
        if inputs.len() != self.input_signatures.len() {
            return Err(TynxError::TypeMismatch(format!(
                "captured graph expected {} inputs, got {}",
                self.input_signatures.len(),
                inputs.len()
            )));
        }
        for (index, (signature, input)) in
            self.input_signatures.iter().zip(inputs.iter()).enumerate()
        {
            signature.validate(input, &format!("input {index}"))?;
        }

        let mut values = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            let value = match node {
                Node::Input(input) => inputs[*input].clone(),
                Node::Parameter(guard) => {
                    guard.validate()?;
                    if tracking {
                        guard.slot.read()
                    } else {
                        guard.slot.value()
                    }
                }
                Node::Unary { op, input } => execute_unary(op, value(&values, *input)?)?,
                Node::Binary { op, left, right } => {
                    execute_binary(*op, value(&values, *left)?, value(&values, *right)?)?
                }
            };
            debug_assert_eq!(values.len(), index);
            values.push(value);
        }

        self.outputs
            .iter()
            .map(|output| value(&values, *output))
            .collect()
    }
}

fn value(values: &[DynTensor], id: ValueId) -> Result<DynTensor> {
    values
        .get(id.index())
        .cloned()
        .ok_or_else(|| TynxError::MissingValue(format!("captured node {}", id.index())))
}

fn execute_unary(op: &UnaryOp, input: DynTensor) -> Result<DynTensor> {
    match op {
        UnaryOp::Relu => Ok(input.relu()),
        UnaryOp::Sigmoid => Ok(input.sigmoid()),
        UnaryOp::Tanh => Ok(input.tanh()),
        UnaryOp::Exp => Ok(input.exp()),
        UnaryOp::Log => Ok(input.log()),
        UnaryOp::Sqrt => Ok(input.sqrt()),
        UnaryOp::Gelu => Ok(input.gelu()),
        UnaryOp::Reshape(shape) => input.reshape(shape.clone()),
        UnaryOp::Permute(axes) => input.permute(axes.clone()),
    }
}

fn execute_binary(op: BinaryOp, left: DynTensor, right: DynTensor) -> Result<DynTensor> {
    match op {
        BinaryOp::Add => left.add_broadcast(right),
        BinaryOp::Subtract => left.sub_broadcast(right),
        BinaryOp::Multiply => left.mul_broadcast(right),
        BinaryOp::Divide => left.div_broadcast(right),
        BinaryOp::Matmul => left.matmul(right),
    }
}
