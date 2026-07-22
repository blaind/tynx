#![forbid(unsafe_code)]

//! Binding-neutral graph capture and replay.
//!
//! This crate owns the small runtime IR used by every future frontend. It intentionally contains
//! no Python callbacks: a frontend records tensor inputs, stable parameter-slot reads, and Tynx
//! operations once, then [`Graph::run`] replays the complete graph in Rust.

use std::{collections::HashSet, fmt::Debug, rc::Rc};

use tynx_core::{DType, Device, Distribution, DynTensor, Result, TynxError, Value};
use tynx_train::{ParamId, ParameterContract, ParameterSlot, backward_slots};

/// Rust-only optimizer action retained by a captured whole-step program.
///
/// Implementations may share native optimizer state with a language binding, but replay never
/// invokes a Python callback.
pub trait CapturedOptimizer: Debug {
    /// Apply one update using gradients already accumulated in the retained parameter slots.
    fn step(&self) -> Result<()>;
}

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

    /// Capture the shape, dtype, and device of any device tensor value.
    pub fn from_value(value: &Value) -> Result<Self> {
        let signature = match value {
            Value::Tensor(tensor) => Self {
                shape: tensor.dims(),
                dtype: tensor.dtype(),
                device: tensor.device(),
            },
            Value::Int(tensor) => Self {
                shape: tensor.dims(),
                dtype: tensor.dtype(),
                device: tensor.device(),
            },
            Value::Bool(tensor) => Self {
                shape: tensor.dims(),
                dtype: tensor.dtype(),
                device: tensor.device(),
            },
            Value::Scalar(_) | Value::Shape(_) => {
                return Err(TynxError::TypeMismatch(
                    "capture inputs must be device tensors".to_string(),
                ));
            }
        };
        Ok(signature)
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

    fn validate_value(&self, value: &Value, label: &str) -> Result<()> {
        let actual = Self::from_value(value)?;
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
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    /// Arithmetic negation.
    Negate,
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
    /// Add a trace-time scalar constant.
    AddScalar(f64),
    /// Subtract a trace-time scalar constant.
    SubtractScalar(f64),
    /// Subtract the tensor from a trace-time scalar constant.
    ReverseSubtractScalar(f64),
    /// Multiply by a trace-time scalar constant.
    MultiplyScalar(f64),
    /// Divide by a trace-time scalar constant.
    DivideScalar(f64),
    /// Divide a trace-time scalar constant by the tensor.
    ReverseDivideScalar(f64),
    /// Raise the tensor to a trace-time scalar exponent.
    PowerScalar(f64),
    /// Raise a trace-time scalar base to the tensor.
    ReversePowerScalar(f64),
    /// Numerically stable log-softmax along one axis.
    LogSoftmax(usize),
    /// Clamp to optional trace-time scalar bounds.
    Clamp {
        /// Optional lower bound.
        min: Option<f64>,
        /// Optional upper bound.
        max: Option<f64>,
    },
    /// Reshape without changing element order.
    Reshape(Vec<usize>),
    /// Permute all tensor axes.
    Permute(Vec<usize>),
    /// Sum selected axes and apply the binding-visible output shape.
    Sum {
        /// Reduced axes.
        dims: Vec<usize>,
        /// Final shape after keepdim/rank-one-floor handling.
        output_shape: Vec<usize>,
    },
    /// Average selected axes and apply the binding-visible output shape.
    Mean {
        /// Reduced axes.
        dims: Vec<usize>,
        /// Final shape after keepdim/rank-one-floor handling.
        output_shape: Vec<usize>,
    },
    /// Training-mode Dropout driven by the device's advancing native RNG stream.
    Dropout(f64),
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
    /// Elementwise floating-point exponentiation.
    Power,
    /// Elementwise minimum.
    Minimum,
    /// Elementwise maximum.
    Maximum,
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
    Gather {
        input: ValueId,
        dim: usize,
        indices: ValueId,
    },
}

#[derive(Debug, Clone)]
enum Effect {
    ZeroGrad(Vec<ParameterSlot>),
    Backward {
        loss: ValueId,
        parameters: Vec<ParameterSlot>,
    },
    OptimizerStep(Rc<dyn CapturedOptimizer>),
}

#[derive(Debug, Clone)]
struct PositionedEffect {
    before_node: usize,
    effect: Effect,
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
    effects: Vec<PositionedEffect>,
}

impl GraphBuilder {
    /// Create an empty graph builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one exact-signature runtime input.
    pub fn input(&mut self, tensor: &DynTensor) -> ValueId {
        self.input_value(&Value::Tensor(tensor.clone()))
            .expect("floating-point tensors are valid capture inputs")
    }

    /// Record one exact-signature runtime tensor input of any supported dtype.
    pub fn input_value(&mut self, value: &Value) -> Result<ValueId> {
        let input = self.input_signatures.len();
        self.input_signatures
            .push(TensorSignature::from_value(value)?);
        Ok(self.push(Node::Input(input)))
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

    /// Record differentiable floating-point gather with integer indices.
    pub fn gather(&mut self, input: ValueId, dim: usize, indices: ValueId) -> Result<ValueId> {
        self.require_value(input)?;
        self.require_value(indices)?;
        Ok(self.push(Node::Gather {
            input,
            dim,
            indices,
        }))
    }

    /// Record gradient clearing at the current execution position.
    pub fn zero_grad(&mut self, parameters: Vec<ParameterSlot>) {
        self.push_effect(Effect::ZeroGrad(parameters));
    }

    /// Record reverse-mode autodiff from a one-element loss at the current execution position.
    pub fn backward(&mut self, loss: ValueId, parameters: Vec<ParameterSlot>) -> Result<()> {
        self.require_value(loss)?;
        self.push_effect(Effect::Backward { loss, parameters });
        Ok(())
    }

    /// Record a native optimizer update at the current execution position.
    pub fn optimizer_step(&mut self, optimizer: Rc<dyn CapturedOptimizer>) {
        self.push_effect(Effect::OptimizerStep(optimizer));
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
            effects: self.effects,
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

    fn push_effect(&mut self, effect: Effect) {
        self.effects.push(PositionedEffect {
            before_node: self.nodes.len(),
            effect,
        });
    }
}

/// Immutable captured graph executable wholly in Rust.
#[derive(Debug, Clone)]
pub struct Graph {
    nodes: Vec<Node>,
    input_signatures: Vec<TensorSignature>,
    outputs: Vec<ValueId>,
    effects: Vec<PositionedEffect>,
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
        let inputs = inputs
            .iter()
            .cloned()
            .map(Value::Tensor)
            .collect::<Vec<_>>();
        self.run_values(&inputs, tracking)?
            .into_iter()
            .map(Value::into_tensor)
            .collect()
    }

    /// Replay the graph with exact-signature float, integer, or boolean tensor inputs.
    pub fn run_values(&self, inputs: &[Value], tracking: bool) -> Result<Vec<Value>> {
        self.validate_value_inputs(inputs)?;

        let mut values = Vec::with_capacity(self.nodes.len());
        let mut effects = self.effects.iter().peekable();
        for (index, node) in self.nodes.iter().enumerate() {
            while effects
                .peek()
                .is_some_and(|effect| effect.before_node == index)
            {
                execute_effect(&effects.next().expect("peeked effect").effect, &values)?;
            }
            let value = match node {
                Node::Input(input) => inputs[*input].clone(),
                Node::Parameter(guard) => {
                    guard.validate()?;
                    if tracking {
                        Value::Tensor(guard.slot.read())
                    } else {
                        Value::Tensor(guard.slot.value())
                    }
                }
                Node::Unary { op, input } => execute_unary(op, value(&values, *input)?)?,
                Node::Binary { op, left, right } => {
                    execute_binary(*op, value(&values, *left)?, value(&values, *right)?)?
                }
                Node::Gather {
                    input,
                    dim,
                    indices,
                } => execute_gather(value(&values, *input)?, *dim, value(&values, *indices)?)?,
            };
            debug_assert_eq!(values.len(), index);
            values.push(value);
        }
        for effect in effects {
            debug_assert_eq!(effect.before_node, self.nodes.len());
            execute_effect(&effect.effect, &values)?;
        }

        self.outputs
            .iter()
            .map(|output| value(&values, *output))
            .collect()
    }

    /// Validate exact input signatures and stable parameter structures without executing nodes.
    pub fn validate_inputs(&self, inputs: &[DynTensor]) -> Result<()> {
        let inputs = inputs
            .iter()
            .cloned()
            .map(Value::Tensor)
            .collect::<Vec<_>>();
        self.validate_value_inputs(&inputs)
    }

    /// Validate exact typed input signatures and stable parameter structures without execution.
    pub fn validate_value_inputs(&self, inputs: &[Value]) -> Result<()> {
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
            signature.validate_value(input, &format!("input {index}"))?;
        }
        for node in &self.nodes {
            if let Node::Parameter(guard) = node {
                guard.validate()?;
            }
        }
        Ok(())
    }
}

fn execute_effect(effect: &Effect, values: &[Value]) -> Result<()> {
    match effect {
        Effect::ZeroGrad(parameters) => {
            for parameter in parameters {
                parameter.zero_grad();
            }
            Ok(())
        }
        Effect::Backward { loss, parameters } => {
            let loss = value(values, *loss)?.into_tensor()?;
            backward_slots(&loss, parameters).map(|_| ())
        }
        Effect::OptimizerStep(optimizer) => optimizer.step(),
    }
}

fn value(values: &[Value], id: ValueId) -> Result<Value> {
    values
        .get(id.index())
        .cloned()
        .ok_or_else(|| TynxError::MissingValue(format!("captured node {}", id.index())))
}

fn execute_unary(op: &UnaryOp, input: Value) -> Result<Value> {
    if let UnaryOp::Reshape(shape) = op {
        return match input {
            Value::Tensor(value) => value.reshape(shape.clone()).map(Value::Tensor),
            Value::Int(value) => value.reshape(shape.clone()).map(Value::Int),
            Value::Bool(value) => value.reshape(shape.clone()).map(Value::Bool),
            Value::Scalar(_) | Value::Shape(_) => Err(TynxError::TypeMismatch(
                "captured reshape requires a device tensor".to_string(),
            )),
        };
    }
    let input = input.into_tensor()?;
    let output = match op {
        UnaryOp::Negate => Ok(input.negated()),
        UnaryOp::Relu => Ok(input.relu()),
        UnaryOp::Sigmoid => Ok(input.sigmoid()),
        UnaryOp::Tanh => Ok(input.tanh()),
        UnaryOp::Exp => Ok(input.exp()),
        UnaryOp::Log => Ok(input.log()),
        UnaryOp::Sqrt => Ok(input.sqrt()),
        UnaryOp::Gelu => Ok(input.gelu()),
        UnaryOp::AddScalar(value) => Ok(input.add_scalar(*value)),
        UnaryOp::SubtractScalar(value) => Ok(input.sub_scalar(*value)),
        UnaryOp::ReverseSubtractScalar(value) => Ok(input.negated().add_scalar(*value)),
        UnaryOp::MultiplyScalar(value) => Ok(input.mul_scalar(*value)),
        UnaryOp::DivideScalar(value) => Ok(input.div_scalar(*value)),
        UnaryOp::ReverseDivideScalar(value) => Ok(input.reciprocal().mul_scalar(*value)),
        UnaryOp::PowerScalar(value) => Ok(input.powf_scalar(*value)),
        UnaryOp::ReversePowerScalar(value) => input.clone().full_like(*value).powf_broadcast(input),
        UnaryOp::LogSoftmax(dim) => Ok(input.log_softmax(*dim)),
        UnaryOp::Clamp { min, max } => Ok(input.clip(*min, *max)),
        UnaryOp::Reshape(shape) => input.reshape(shape.clone()),
        UnaryOp::Permute(axes) => input.permute(axes.clone()),
        UnaryOp::Sum { dims, output_shape } => input.sum_dims(dims).reshape(output_shape.clone()),
        UnaryOp::Mean { dims, output_shape } => input.mean_dims(dims).reshape(output_shape.clone()),
        UnaryOp::Dropout(probability) => {
            if *probability == 1.0 {
                Ok(input.mul_scalar(0.0))
            } else if *probability == 0.0 {
                Ok(input)
            } else {
                let mask = DynTensor::random(
                    &input.dims(),
                    Distribution::Bernoulli(1.0 - probability),
                    &input.device(),
                    DType::F32,
                )?;
                Ok(input
                    .mul_broadcast(mask)?
                    .mul_scalar(1.0 / (1.0 - probability)))
            }
        }
    }?;
    Ok(Value::Tensor(output))
}

fn execute_binary(op: BinaryOp, left: Value, right: Value) -> Result<Value> {
    let left = left.into_tensor()?;
    let right = right.into_tensor()?;
    let output = match op {
        BinaryOp::Add => left.add_broadcast(right),
        BinaryOp::Subtract => left.sub_broadcast(right),
        BinaryOp::Multiply => left.mul_broadcast(right),
        BinaryOp::Divide => left.div_broadcast(right),
        BinaryOp::Matmul => left.matmul(right),
        BinaryOp::Power => left.powf_broadcast(right),
        BinaryOp::Minimum => left.min_broadcast(right),
        BinaryOp::Maximum => left.max_broadcast(right),
    }?;
    Ok(Value::Tensor(output))
}

fn execute_gather(input: Value, dim: usize, indices: Value) -> Result<Value> {
    let input = input.into_tensor()?;
    let indices = indices.into_int()?;
    input.gather(dim, indices).map(Value::Tensor)
}
