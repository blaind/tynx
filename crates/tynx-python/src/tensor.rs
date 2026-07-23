//! Eager CPython tensor projection over the binding-neutral Rust tensor facade.

mod combine;
mod comparison;
mod data;
mod extrema;
mod factory;
mod indexing;
mod reduction;
mod selection;
mod shape;

use std::{
    cell::{Cell, RefCell},
    panic::{AssertUnwindSafe, catch_unwind},
    rc::{Rc, Weak},
};

use pyo3::{
    exceptions::{PyIndexError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBool, PyList, PyTuple},
};
use tynx_capture::{BinaryOp, UnaryOp};
use tynx_core::{Device, DynInt, DynTensor, Gradients, Slice, TensorData, Value};
use tynx_train::ParameterSlot;

use crate::{
    capture::{
        TraceValue, record_backward, record_binary, record_gather, record_index_select,
        record_unary, record_unsupported,
    },
    device::{PyDevice, ensure_autodiff, raise_pending_device_error},
    grad_mode::is_grad_enabled,
    to_python_error,
};
pub(crate) use combine::{cat_py, chunk_py, split_py, stack_py};
use comparison::{Comparison, MaskOperation};
pub(crate) use data::IntBounds;
use data::TensorValue;
use extrema::Extremum;
pub(crate) use factory::{
    arange_py, empty_like_py, empty_py, full_like_py, full_py, ones_like_py, ones_py, rand_like_py,
    rand_py, randint_py, randn_like_py, randn_py, zeros_like_py, zeros_py,
};
use reduction::ReductionSpec;

/// Eager device tensor with optional floating-point autodiff state.
///
/// Burn-owned tensor state stays in a Rust heap allocation and the initial binding is explicitly
/// unsendable. Operations return new tensors and delegate numerical semantics to `DynTensor`.
#[pyclass(name = "Tensor", frozen, unsendable, subclass)]
pub(crate) struct PyTensor {
    source: TensorSource,
    int_bounds: Option<Rc<RefCell<Option<IntBounds>>>>,
    targets: Vec<GradTarget>,
    leaf: Option<Rc<LeafState>>,
    backward_graphs: Vec<Rc<BackwardGraph>>,
    trace: Option<TraceValue>,
}

#[derive(Debug, Default)]
struct BackwardGraph {
    consumed: Cell<bool>,
}

#[derive(Debug)]
enum TensorSource {
    Owned(Box<TensorValue>),
    Parameter(ParameterSlot),
}

impl TensorSource {
    fn value(&self) -> TensorValue {
        match self {
            Self::Owned(value) => value.as_ref().clone(),
            Self::Parameter(slot) => TensorValue::Float(slot.value()),
        }
    }

    fn operation_input(&self, tracking: bool, operation: &str) -> PyResult<DynTensor> {
        match self {
            Self::Owned(value) if tracking => value.as_ref().clone().float(operation),
            Self::Owned(value) => value.as_ref().clone().detach().float(operation),
            Self::Parameter(slot) if tracking => Ok(slot.read()),
            Self::Parameter(slot) => Ok(slot.value()),
        }
    }
}

#[derive(Debug, Clone)]
enum GradTarget {
    Tensor(Weak<LeafState>),
    Parameter {
        slot: ParameterSlot,
        generation: Option<u64>,
        generation_conflict: bool,
    },
}

impl GradTarget {
    fn validate_generation(&self) -> PyResult<()> {
        if let Self::Parameter {
            slot,
            generation_conflict: true,
            ..
        } = self
        {
            return Err(PyRuntimeError::new_err(format!(
                "parameter {} was used at multiple value generations in one graph; rebuild the graph before backward()",
                slot.name()
                    .unwrap_or_else(|| format!("#{}", slot.id().get()))
            )));
        }
        if let Self::Parameter {
            slot,
            generation: Some(expected),
            ..
        } = self
            && *expected != slot.value_generation()
        {
            return Err(PyRuntimeError::new_err(format!(
                "parameter {} was modified after the forward pass; rebuild the graph before backward()",
                slot.name()
                    .unwrap_or_else(|| format!("#{}", slot.id().get()))
            )));
        }
        Ok(())
    }

    fn same_identity(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Tensor(left), Self::Tensor(right)) => left.ptr_eq(right),
            (Self::Parameter { slot: left, .. }, Self::Parameter { slot: right, .. }) => {
                left.id() == right.id()
            }
            _ => false,
        }
    }

    fn merge_same_identity(&mut self, other: &Self) -> bool {
        if !self.same_identity(other) {
            return false;
        }
        if let (
            Self::Parameter {
                generation: left,
                generation_conflict,
                ..
            },
            Self::Parameter {
                slot,
                generation: right,
                generation_conflict: right_conflict,
            },
        ) = (self, other)
        {
            let left = left.get_or_insert_with(|| slot.value_generation());
            let right = right.unwrap_or_else(|| slot.value_generation());
            *generation_conflict |= *right_conflict || *left != right;
        }
        true
    }

    fn accumulate(&self, gradients: &Gradients) -> tynx_core::Result<()> {
        match self {
            Self::Tensor(leaf) => {
                if let Some(leaf) = leaf.upgrade() {
                    leaf.accumulate(gradients)?;
                }
            }
            Self::Parameter {
                slot,
                generation,
                generation_conflict,
            } => {
                if *generation_conflict {
                    return Err(tynx_core::TynxError::TypeMismatch(format!(
                        "parameter {} was used at multiple value generations in one graph; rebuild the graph before backward()",
                        slot.name()
                            .unwrap_or_else(|| format!("#{}", slot.id().get()))
                    )));
                }
                if generation.is_some_and(|expected| expected != slot.value_generation()) {
                    return Err(tynx_core::TynxError::TypeMismatch(format!(
                        "parameter {} was modified after the forward pass; rebuild the graph before backward()",
                        slot.name()
                            .unwrap_or_else(|| format!("#{}", slot.id().get()))
                    )));
                }
                slot.accumulate_grad(gradients)?;
            }
        }
        Ok(())
    }

    fn for_operation(&self) -> Self {
        match self {
            Self::Tensor(leaf) => Self::Tensor(leaf.clone()),
            Self::Parameter {
                slot,
                generation,
                generation_conflict,
            } => Self::Parameter {
                slot: slot.clone(),
                generation: Some(generation.unwrap_or_else(|| slot.value_generation())),
                generation_conflict: *generation_conflict,
            },
        }
    }

    fn mark_tape_consumed(&self) {
        if let Self::Tensor(leaf) = self
            && let Some(leaf) = leaf.upgrade()
        {
            leaf.tape_consumed.set(true);
        }
    }
}

#[derive(Debug)]
struct LeafState {
    tensor: RefCell<DynTensor>,
    tape_consumed: Cell<bool>,
    grad: RefCell<Option<DynTensor>>,
}

impl LeafState {
    fn operation_input(&self) -> DynTensor {
        if self.tape_consumed.replace(false) {
            let fresh = self.tensor.borrow().clone().detach().require_grad();
            *self.tensor.borrow_mut() = fresh;
        }
        self.tensor.borrow().clone()
    }

    fn accumulate(&self, gradients: &Gradients) -> tynx_core::Result<()> {
        let Some(gradient) = self.tensor.borrow().grad(gradients) else {
            return Ok(());
        };
        let gradient = gradient.detach();
        let mut current = self.grad.borrow_mut();
        *current = Some(match current.take() {
            Some(previous) => previous.add_broadcast(gradient)?,
            None => gradient,
        });
        Ok(())
    }
}

impl PyTensor {
    fn reject_in_place(&self, operation: &str) -> PyResult<()> {
        Err(PyRuntimeError::new_err(format!(
            "in-place arithmetic ({operation}) is not supported; use an explicit assignment or an optimizer step"
        )))
    }

    pub(crate) fn from_inner(inner: DynTensor) -> Self {
        Self {
            source: TensorSource::Owned(Box::new(TensorValue::Float(inner))),
            int_bounds: None,
            targets: Vec::new(),
            leaf: None,
            backward_graphs: Vec::new(),
            trace: None,
        }
    }

    pub(crate) fn from_int_inner(inner: DynInt) -> Self {
        Self::from_value(TensorValue::Int(inner))
    }

    pub(crate) fn from_int_inner_with_bounds(inner: DynInt, bounds: IntBounds) -> Self {
        Self::from_value_with_int_bounds(TensorValue::Int(inner), Some(bounds))
    }

    fn from_value(value: TensorValue) -> Self {
        Self::from_value_with_int_bounds(value, None)
    }

    fn from_value_with_int_bounds(value: TensorValue, bounds: Option<IntBounds>) -> Self {
        let int_bounds =
            matches!(value, TensorValue::Int(_)).then(|| Rc::new(RefCell::new(bounds)));
        Self {
            source: TensorSource::Owned(Box::new(value)),
            int_bounds,
            targets: Vec::new(),
            leaf: None,
            backward_graphs: Vec::new(),
            trace: None,
        }
    }

    fn from_leaf(inner: DynTensor) -> Self {
        let inner = inner.require_grad();
        let leaf = Rc::new(LeafState {
            tensor: RefCell::new(inner.clone()),
            tape_consumed: Cell::new(false),
            grad: RefCell::new(None),
        });
        Self {
            source: TensorSource::Owned(Box::new(TensorValue::Float(inner))),
            int_bounds: None,
            targets: vec![GradTarget::Tensor(Rc::downgrade(&leaf))],
            leaf: Some(leaf),
            backward_graphs: Vec::new(),
            trace: None,
        }
    }

    pub(crate) fn from_parameter(slot: ParameterSlot) -> Self {
        let targets = if slot.contract().trainable() {
            vec![GradTarget::Parameter {
                slot: slot.clone(),
                generation: None,
                generation_conflict: false,
            }]
        } else {
            Vec::new()
        };
        Self {
            source: TensorSource::Parameter(slot.clone()),
            int_bounds: None,
            targets,
            leaf: None,
            backward_graphs: Vec::new(),
            trace: None,
        }
    }

    pub(crate) fn from_operation(inner: DynTensor, sources: &[&Self]) -> Self {
        let mut targets: Vec<GradTarget> = Vec::new();
        let mut backward_graphs: Vec<Rc<BackwardGraph>> = Vec::new();
        for source in sources {
            for target in &source.targets {
                if let Some(existing) = targets
                    .iter_mut()
                    .find(|existing| existing.same_identity(target))
                {
                    existing.merge_same_identity(&target.for_operation());
                } else {
                    targets.push(target.for_operation());
                }
            }
            for graph in &source.backward_graphs {
                if !backward_graphs
                    .iter()
                    .any(|existing| Rc::ptr_eq(existing, graph))
                {
                    backward_graphs.push(graph.clone());
                }
            }
        }
        backward_graphs.push(Rc::new(BackwardGraph::default()));
        Self {
            source: TensorSource::Owned(Box::new(TensorValue::Float(inner))),
            int_bounds: None,
            targets,
            leaf: None,
            backward_graphs,
            trace: None,
        }
    }

    pub(crate) fn from_imported_operation(
        inner: DynTensor,
        sources: &[&Self],
        parameters: impl IntoIterator<Item = ParameterSlot>,
    ) -> Self {
        let mut output = Self::from_operation(inner, sources);
        for parameter in parameters {
            let generation = parameter.value_generation();
            let target = GradTarget::Parameter {
                slot: parameter,
                generation: Some(generation),
                generation_conflict: false,
            };
            if let Some(existing) = output
                .targets
                .iter_mut()
                .find(|existing| existing.same_identity(&target))
            {
                existing.merge_same_identity(&target);
            } else {
                output.targets.push(target);
            }
        }
        output
    }

    fn operation_input(&self, tracking: bool, operation: &str) -> PyResult<DynTensor> {
        if tracking && let Some(leaf) = &self.leaf {
            return Ok(leaf.operation_input());
        }
        self.source.operation_input(tracking, operation)
    }

    fn binary(
        &self,
        other: &Self,
        capture_op: BinaryOp,
        operation: impl FnOnce(DynTensor, DynTensor) -> tynx_core::Result<DynTensor>,
    ) -> PyResult<Self> {
        let tracking = is_grad_enabled();
        let left = self.operation_input(tracking, "arithmetic")?;
        let right = other.operation_input(tracking, "arithmetic")?;
        let inner = operation(left, right).map_err(to_python_error)?;
        let mut result = if tracking {
            Self::from_operation(inner, &[self, other])
        } else {
            Self::from_inner(inner)
        };
        result.trace = record_binary(self, other, capture_op)?;
        Ok(result)
    }

    fn arithmetic(
        &self,
        other: &Bound<'_, PyAny>,
        capture_op: BinaryOp,
        tensor_operation: impl FnOnce(DynTensor, DynTensor) -> tynx_core::Result<DynTensor>,
        scalar_operation: impl FnOnce(DynTensor, f64) -> DynTensor,
    ) -> PyResult<Self> {
        if let Ok(other) = other.extract::<PyRef<'_, Self>>() {
            return self.binary(&other, capture_op, tensor_operation);
        }
        let scalar = extract_scalar_operand(other)?;
        let capture_op = match capture_op {
            BinaryOp::Add => UnaryOp::AddScalar(scalar),
            BinaryOp::Subtract => UnaryOp::SubtractScalar(scalar),
            BinaryOp::Multiply => UnaryOp::MultiplyScalar(scalar),
            BinaryOp::Divide => UnaryOp::DivideScalar(scalar),
            BinaryOp::Power => UnaryOp::PowerScalar(scalar),
            BinaryOp::Matmul
            | BinaryOp::Minimum
            | BinaryOp::Maximum
            | BinaryOp::NormalSample { .. } => {
                unreachable!("these operations do not use scalar arithmetic")
            }
        };
        self.unary_captured(capture_op, |input| Ok(scalar_operation(input, scalar)))
    }

    fn unary(
        &self,
        operation: impl FnOnce(DynTensor) -> tynx_core::Result<DynTensor>,
    ) -> PyResult<Self> {
        let tracking = is_grad_enabled();
        let input = self.operation_input(tracking, "operation")?;
        let inner = operation(input).map_err(to_python_error)?;
        let mut result = if tracking {
            Self::from_operation(inner, &[self])
        } else {
            Self::from_inner(inner)
        };
        record_unsupported(self, "this tensor operation")?;
        result.trace = None;
        Ok(result)
    }

    fn unary_captured(
        &self,
        capture_op: UnaryOp,
        operation: impl FnOnce(DynTensor) -> tynx_core::Result<DynTensor>,
    ) -> PyResult<Self> {
        let tracking = is_grad_enabled();
        let input = self.operation_input(tracking, "operation")?;
        let inner = operation(input).map_err(to_python_error)?;
        let mut result = if tracking {
            Self::from_operation(inner, &[self])
        } else {
            Self::from_inner(inner)
        };
        result.trace = record_unary(self, capture_op)?;
        Ok(result)
    }

    fn compare(&self, other: &Bound<'_, PyAny>, comparison: Comparison) -> PyResult<Self> {
        self.capture_unsupported("tensor comparisons")?;
        let left = self.source.value().detach();
        let value = if let Ok(other) = other.extract::<PyRef<'_, Self>>() {
            left.compare_tensor(other.source.value().detach(), comparison)?
        } else {
            left.compare_scalar(other, comparison)?
        };
        Ok(Self::from_value(value))
    }

    fn mask_binary(&self, other: &Self, operation: MaskOperation) -> PyResult<Self> {
        self.source
            .value()
            .detach()
            .mask_binary(other.source.value().detach(), operation)
            .map(Self::from_value)
    }

    fn where_operands(
        condition: &Self,
        then_tensor: Option<&Self>,
        then_scalar: Option<&Bound<'_, PyAny>>,
        otherwise_tensor: Option<&Self>,
        otherwise_scalar: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let template = then_tensor.or(otherwise_tensor).ok_or_else(|| {
            PyTypeError::new_err("where requires at least one Tensor branch to infer dtype/device")
        })?;
        let template = template.source.value().detach();
        let tracking = is_grad_enabled();

        let branch_value =
            |tensor: Option<&Self>, scalar: Option<&Bound<'_, PyAny>>| -> PyResult<TensorValue> {
                if let Some(tensor) = tensor {
                    return match tensor.source.value() {
                        TensorValue::Float(_) => tensor
                            .operation_input(tracking, "where")
                            .map(TensorValue::Float),
                        value => Ok(value.detach()),
                    };
                }
                selection::scalar_like(
                    template.clone(),
                    scalar.expect("a where branch is either a Tensor or a scalar"),
                )
            };

        let then = branch_value(then_tensor, then_scalar)?;
        let otherwise = branch_value(otherwise_tensor, otherwise_scalar)?;
        let condition = selection::condition(condition.source.value().detach())?;
        let result = selection::select(condition, then, otherwise)?;

        match result {
            TensorValue::Float(inner) if tracking => {
                let mut sources = Vec::with_capacity(2);
                if let Some(source) = then_tensor {
                    sources.push(source);
                }
                if let Some(source) = otherwise_tensor {
                    sources.push(source);
                }
                Ok(Self::from_operation(inner, &sources))
            }
            value => Ok(Self::from_value(value.detach())),
        }
    }

    fn where_from_python(
        condition: &Self,
        then: &Bound<'_, PyAny>,
        otherwise: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let then_tensor = then.extract::<PyRef<'_, Self>>().ok();
        let otherwise_tensor = otherwise.extract::<PyRef<'_, Self>>().ok();
        Self::where_operands(
            condition,
            then_tensor.as_deref(),
            then_tensor.is_none().then_some(then),
            otherwise_tensor.as_deref(),
            otherwise_tensor.is_none().then_some(otherwise),
        )
    }

    fn gather_impl(&self, dim: usize, index: &Self) -> PyResult<Self> {
        let input_shape = self.source.value().dims();
        let indices = indexing::gather_indices(index.source.value().detach(), &input_shape, dim)?;
        index.validate_index_bounds(input_shape[dim], dim, "gather")?;
        let tracking = is_grad_enabled();
        match self.source.value() {
            TensorValue::Float(_) => {
                let input = self.operation_input(tracking, "gather")?;
                let inner = input.gather(dim, indices).map_err(to_python_error)?;
                let mut output = if tracking {
                    Self::from_operation(inner, &[self])
                } else {
                    Self::from_inner(inner)
                };
                output.trace = record_gather(self, dim, input_shape[dim], index)?;
                Ok(output)
            }
            value => indexing::gather(value.detach(), dim, indices).map(Self::from_value),
        }
    }

    fn elementwise_extreme(&self, other: &Bound<'_, PyAny>, extremum: Extremum) -> PyResult<Self> {
        let tracking = is_grad_enabled();
        let other_tensor = other.extract::<PyRef<'_, Self>>().ok();
        if matches!(self.source.value(), TensorValue::Float(_))
            && let Some(other) = other_tensor.as_deref()
            && matches!(other.source.value(), TensorValue::Float(_))
        {
            let capture_op = match extremum {
                Extremum::Minimum => BinaryOp::Minimum,
                Extremum::Maximum => BinaryOp::Maximum,
            };
            return self.binary(other, capture_op, |left, right| {
                extremum.float_pair(left, right)
            });
        }
        let left = match self.source.value() {
            TensorValue::Float(_) => self
                .operation_input(tracking, extremum.name())
                .map(TensorValue::Float)?,
            value => value.detach(),
        };
        let right = if let Some(other) = other_tensor.as_deref() {
            match other.source.value() {
                TensorValue::Float(_) => other
                    .operation_input(tracking, extremum.name())
                    .map(TensorValue::Float)?,
                value => value.detach(),
            }
        } else {
            self.source
                .value()
                .detach()
                .scalar_like(other, extremum.name())?
        };
        let result = extrema::pair(left, right, extremum)?;
        match result {
            TensorValue::Float(inner) if tracking => {
                let mut sources = vec![self];
                if let Some(other) = other_tensor.as_deref() {
                    sources.push(other);
                }
                Ok(Self::from_operation(inner, &sources))
            }
            value => Ok(Self::from_value(value.detach())),
        }
    }

    pub(crate) fn tensor_from_python(data: &Bound<'_, PyAny>) -> PyResult<DynTensor> {
        if let Ok(tensor) = data.extract::<PyRef<'_, Self>>() {
            return tensor.detached_float_value("Parameter/Buffer construction");
        }
        let device = Device::autodiff(tynx_core::default_device());
        TensorValue::float_from_python(data, &device)
    }

    pub(crate) fn parameter_name(&self) -> Option<String> {
        match &self.source {
            TensorSource::Parameter(slot) => slot.name(),
            TensorSource::Owned(_) => None,
        }
    }

    pub(crate) fn parameter_slot(&self) -> Option<ParameterSlot> {
        match &self.source {
            TensorSource::Parameter(slot) => Some(slot.clone()),
            TensorSource::Owned(_) => None,
        }
    }

    pub(crate) fn trace(&self) -> Option<&TraceValue> {
        self.trace.as_ref()
    }

    pub(crate) fn with_trace(&self, trace: TraceValue) -> Self {
        Self {
            source: TensorSource::Owned(Box::new(self.source.value())),
            int_bounds: self.int_bounds.clone(),
            targets: self.targets.clone(),
            leaf: self.leaf.clone(),
            backward_graphs: self.backward_graphs.clone(),
            trace: Some(trace),
        }
    }

    pub(crate) fn without_trace(&self) -> Self {
        Self {
            source: TensorSource::Owned(Box::new(self.source.value())),
            int_bounds: self.int_bounds.clone(),
            targets: self.targets.clone(),
            leaf: self.leaf.clone(),
            backward_graphs: self.backward_graphs.clone(),
            trace: None,
        }
    }

    pub(crate) fn with_recorded_unary(
        mut self,
        source: &Self,
        operation: UnaryOp,
    ) -> PyResult<Self> {
        self.trace = record_unary(source, operation)?;
        Ok(self)
    }

    pub(crate) fn with_recorded_binary(
        mut self,
        left: &Self,
        right: &Self,
        operation: BinaryOp,
    ) -> PyResult<Self> {
        self.trace = record_binary(left, right, operation)?;
        Ok(self)
    }

    pub(crate) fn with_inherited_int_bounds(mut self, source: &Self) -> Self {
        if self.int_bounds.is_some() {
            self.int_bounds = source.int_bounds.clone();
        }
        self
    }

    pub(crate) fn validate_index_bounds(
        &self,
        size: usize,
        dim: usize,
        operation: &str,
    ) -> PyResult<()> {
        let size = i64::try_from(size).map_err(|_| {
            PyValueError::new_err(format!(
                "{operation} dimension {dim} exceeds the supported index range"
            ))
        })?;
        let cache = self.int_bounds.as_ref().ok_or_else(|| {
            PyTypeError::new_err(format!("{operation} indices must be an int64 Tensor"))
        })?;
        let bounds = if let Some(bounds) = *cache.borrow() {
            bounds
        } else {
            let indices = match self.source.value().detach() {
                TensorValue::Int(indices) => indices,
                _ => unreachable!("an integer bounds cache is attached only to int64 tensors"),
            };
            let values = indices.into_data().iter::<i64>().collect::<Vec<_>>();
            raise_pending_device_error()?;
            let bounds = IntBounds::from_values(&values);
            *cache.borrow_mut() = Some(bounds);
            bounds
        };
        let IntBounds::Range { min, max } = bounds else {
            return Ok(());
        };
        let invalid = if min < 0 {
            Some(min)
        } else if max >= size {
            Some(max)
        } else {
            None
        };
        if let Some(index) = invalid {
            return Err(PyIndexError::new_err(format!(
                "{operation} index {index} is out of bounds for dimension {dim} with size {size}"
            )));
        }
        Ok(())
    }

    pub(crate) fn detached_float_value(&self, operation: &str) -> PyResult<DynTensor> {
        self.source.value().detach().float(operation)
    }

    pub(crate) fn detached_runtime_value(&self) -> Value {
        self.source.value().detach().into_runtime()
    }

    pub(crate) fn operation_runtime_value(
        &self,
        tracking: bool,
        operation: &str,
    ) -> PyResult<Value> {
        match self.source.value() {
            TensorValue::Float(_) => self.operation_input(tracking, operation).map(Value::Tensor),
            value => Ok(value.detach().into_runtime()),
        }
    }

    pub(crate) fn from_runtime_value(value: Value) -> PyResult<Self> {
        match value {
            Value::Tensor(value) => Ok(Self::from_inner(value)),
            Value::Int(value) => Ok(Self::from_int_inner(value)),
            Value::Bool(value) => Ok(Self::from_value(TensorValue::Bool(value))),
            Value::Scalar(_) | Value::Shape(_) => Err(PyTypeError::new_err(
                "captured graph output must be a device tensor",
            )),
        }
    }

    pub(crate) fn operation_float_value(
        &self,
        tracking: bool,
        operation: &str,
    ) -> PyResult<DynTensor> {
        self.operation_input(tracking, operation)
    }

    pub(crate) fn capture_unsupported(&self, reason: &str) -> PyResult<()> {
        record_unsupported(self, reason)
    }
}

#[pymethods]
impl PyTensor {
    /// Construct a typed tensor from a scalar or rectangular nested list/tuple.
    #[new]
    #[pyo3(signature = (data, *, dtype=None, device=None, requires_grad=false))]
    fn new(
        data: &Bound<'_, PyAny>,
        dtype: Option<&str>,
        device: Option<PyRef<'_, PyDevice>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let (value, bounds, inherited_bounds) =
            if let Ok(tensor) = data.extract::<PyRef<'_, Self>>() {
                let value = tensor.source.value().detach();
                let target =
                    device.map_or_else(|| value.device(), |device| device.inner.as_ref().clone());
                let target_dtype = dtype.unwrap_or(value.dtype_name());
                let preserve_bounds = value.dtype_name() == "int64" && target_dtype == "int64";
                (
                    value
                        .cast(target_dtype)?
                        .move_to_device(&ensure_autodiff(target)),
                    None,
                    preserve_bounds.then(|| tensor.int_bounds.clone()).flatten(),
                )
            } else {
                let device = ensure_autodiff(
                    device
                        .map(|device| device.inner.as_ref().clone())
                        .unwrap_or_else(tynx_core::default_device),
                );
                let (value, bounds) = TensorValue::from_python(data, dtype, &device)?;
                (value, bounds, None)
            };
        if requires_grad {
            return Ok(Self::from_leaf(value.float("requires_grad=True")?));
        }
        let mut output = Self::from_value_with_int_bounds(value, bounds);
        if inherited_bounds.is_some() {
            output.int_bounds = inherited_bounds;
        }
        Ok(output)
    }

    /// Tensor dimensions as a Python tuple.
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.source.value().dims())?.unbind())
    }

    /// Number of tensor dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.source.value().rank()
    }

    /// Number of tensor elements.
    #[getter]
    fn numel(&self) -> usize {
        self.source.value().dims().into_iter().product()
    }

    fn __len__(&self) -> usize {
        self.source.value().dims()[0]
    }

    /// Select values using basic indices or a one-dimensional first-axis advanced index.
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(index) = key.extract::<PyRef<'_, Self>>() {
            return self.advanced_getitem(&index);
        }
        if let Ok(indices) = key.cast::<PyList>() {
            self.capture_unsupported("Tensor.__getitem__ advanced indexing")?;
            let value = self.source.value();
            let size = value.dims()[0];
            let size_signed = isize::try_from(size)
                .map_err(|_| PyIndexError::new_err("tensor dimension exceeds index limits"))?;
            let mut values = Vec::with_capacity(indices.len());
            for item in indices.iter() {
                if item.is_instance_of::<PyBool>() {
                    return Err(PyTypeError::new_err(
                        "advanced index lists must contain only integers",
                    ));
                }
                let original = item.extract::<isize>().map_err(|_| {
                    PyTypeError::new_err("advanced index lists must contain only integers")
                })?;
                let normalized = if original < 0 {
                    size_signed + original
                } else {
                    original
                };
                if !(0..size_signed).contains(&normalized) {
                    return Err(PyIndexError::new_err(format!(
                        "index {original} is out of bounds for dimension 0 with size {size}"
                    )));
                }
                values.push(normalized as i64);
            }
            let count = values.len();
            let index = if count == 0 {
                DynInt::empty(&[0], &value.device(), tynx_core::DType::I64)
            } else {
                DynInt::from_data(TensorData::new(values, [count]), 1, &value.device())
            }
            .map_err(to_python_error)?;
            return self.select_indices(0, index, "Tensor.__getitem__", None);
        }
        let value = self.source.value();
        let spec = indexing::basic_index(key, &value.dims())?;
        let capture_op = UnaryOp::Slice {
            slices: spec.slices.clone(),
            output_shape: spec.output_shape.clone(),
        };
        let tracking = is_grad_enabled();
        let mut output = match value {
            TensorValue::Float(_) => {
                let output = self
                    .operation_input(tracking, "Tensor.__getitem__")?
                    .slice(&spec.slices)
                    .reshape(spec.output_shape.clone())
                    .map_err(to_python_error)?;
                if tracking {
                    Self::from_operation(output, &[self])
                } else {
                    Self::from_inner(output)
                }
            }
            TensorValue::Int(value) => value
                .slice(&spec.slices)
                .reshape(spec.output_shape.clone())
                .map(Self::from_int_inner)
                .map_err(to_python_error)?,
            TensorValue::Bool(value) => value
                .slice(&spec.slices)
                .reshape(spec.output_shape)
                .map(|value| Self::from_value(TensorValue::Bool(value)))
                .map_err(to_python_error)?,
        };
        output.trace = record_unary(self, capture_op)?;
        Ok(output)
    }

    /// Split into ordinary tensor results along one dimension.
    #[pyo3(signature = (split_size_or_sections, dim=0))]
    fn split(
        &self,
        py: Python<'_>,
        split_size_or_sections: &Bound<'_, PyAny>,
        dim: isize,
    ) -> PyResult<Py<PyTuple>> {
        let outputs = combine::split_outputs(self, split_size_or_sections, dim)?;
        let outputs = outputs
            .into_iter()
            .map(|output| Py::new(py, output))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, outputs)?.unbind())
    }

    /// Divide into at most `chunks` ordinary tensor results.
    #[pyo3(signature = (chunks, dim=0))]
    fn chunk(&self, py: Python<'_>, chunks: usize, dim: isize) -> PyResult<Py<PyTuple>> {
        let outputs = combine::chunk_outputs(self, chunks, dim)?;
        let outputs = outputs
            .into_iter()
            .map(|output| Py::new(py, output))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, outputs)?.unbind())
    }

    fn __bool__(&self, py: Python<'_>) -> PyResult<bool> {
        self.capture_unsupported("tensor-dependent Python control flow")?;
        if self.numel() != 1 {
            return Err(PyValueError::new_err(format!(
                "the truth value of a Tensor with shape {:?} is ambiguous",
                self.source.value().dims()
            )));
        }
        let value = self.source.value().item(py)?;
        raise_pending_device_error()?;
        value.bind(py).is_truthy()
    }

    /// Element dtype.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.source.value().dtype_name()
    }

    /// Execution device used by this tensor.
    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice::new(self.source.value().device())
    }

    /// Whether this tensor participates in an autodiff graph.
    #[getter]
    fn requires_grad(&self) -> bool {
        !self.targets.is_empty()
    }

    /// Whether this object is a user-created autodiff leaf.
    #[getter]
    fn is_leaf(&self) -> bool {
        self.leaf.is_some() || matches!(self.source, TensorSource::Parameter(_))
    }

    /// Return the accumulated gradient for a leaf tensor.
    #[getter]
    fn grad(&self) -> Option<Self> {
        let gradient = match &self.source {
            TensorSource::Parameter(slot) => slot.grad(),
            TensorSource::Owned(_) => self
                .leaf
                .as_ref()
                .and_then(|leaf| leaf.grad.borrow().clone()),
        };
        gradient.map(Self::from_inner)
    }

    /// Copy tensor values to nested Python lists.
    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.capture_unsupported("Tensor.tolist() host reads")?;
        let result = self.source.value().tolist(py);
        raise_pending_device_error()?;
        result
    }

    /// Copy tensor values to a NumPy array with the matching dtype and shape.
    fn numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.capture_unsupported("Tensor.numpy() host reads")?;
        let result = self.source.value().numpy(py);
        raise_pending_device_error()?;
        result
    }

    /// Copy a one-element tensor to a Python scalar.
    fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.capture_unsupported("Tensor.item() host reads")?;
        if self.numel() != 1 {
            return Err(PyValueError::new_err(format!(
                "item() requires a one-element tensor, got shape {:?}",
                self.source.value().dims()
            )));
        }
        let result = self.source.value().item(py);
        raise_pending_device_error()?;
        result
    }

    /// Return an off-tape tensor sharing the current numerical value.
    fn detach(&self) -> Self {
        Self::from_value(self.source.value().detach()).with_inherited_int_bounds(self)
    }

    /// Cast tensor values to one of Tynx's supported dtypes.
    fn cast(&self, dtype: &str) -> PyResult<Self> {
        self.to(None, Some(dtype))
    }

    /// Move or cast a tensor without a host staging path.
    #[pyo3(signature = (device=None, *, dtype=None))]
    fn to(&self, device: Option<PyRef<'_, PyDevice>>, dtype: Option<&str>) -> PyResult<Self> {
        self.capture_unsupported("Tensor.to()/cast()")?;
        let current = self.source.value();
        let dtype = dtype.unwrap_or(current.dtype_name());
        let target_device = ensure_autodiff(
            device.map_or_else(|| current.device(), |device| device.inner.as_ref().clone()),
        );
        let current_device = current.device();
        if current_device.clone().inner() != target_device.clone().inner() {
            return Err(PyNotImplementedError::new_err(format!(
                "Tensor.to() cannot move tensors between backends ({current_device:?} to {target_device:?}); create the tensor on the target device instead"
            )));
        }
        let tracking = is_grad_enabled();
        if matches!(current, TensorValue::Float(_)) && dtype == "float32" {
            let output = self
                .operation_input(tracking, "Tensor.to")?
                .cast(tynx_core::DType::F32)
                .to_device(&target_device);
            return Ok(if tracking {
                Self::from_operation(output, &[self])
            } else {
                Self::from_inner(output)
            });
        }
        let preserve_bounds = current.dtype_name() == "int64" && dtype == "int64";
        let output = current.detach().cast(dtype)?.move_to_device(&target_device);
        let output = Self::from_value(output);
        Ok(if preserve_bounds {
            output.with_inherited_int_bounds(self)
        } else {
            output
        })
    }

    /// Replace stable Parameter/Buffer state from a compatible tensor without changing identity.
    fn copy_(&self, source: PyRef<'_, Self>) -> PyResult<()> {
        source.capture_unsupported("state mutation through Tensor.copy_()")?;
        let slot = self.parameter_slot().ok_or_else(|| {
            PyTypeError::new_err("copy_ target must be a stable Parameter or Buffer")
        })?;
        let value = source.source.value().detach().float("copy_")?;
        slot.replace_value(value).map_err(to_python_error)
    }

    /// Clear this leaf tensor's accumulated gradient.
    fn zero_grad(&self) {
        match &self.source {
            TensorSource::Parameter(slot) => slot.zero_grad(),
            TensorSource::Owned(_) => {
                if let Some(leaf) = &self.leaf {
                    *leaf.grad.borrow_mut() = None;
                }
            }
        }
    }

    /// Run reverse-mode autodiff, optionally seeded by a matching tensor.
    #[pyo3(signature = (gradient=None))]
    fn backward(&self, gradient: Option<PyRef<'_, Self>>) -> PyResult<()> {
        if gradient.is_some() {
            self.capture_unsupported("backward with an explicit gradient")?;
        } else {
            let parameters = self
                .targets
                .iter()
                .filter_map(|target| match target {
                    GradTarget::Parameter { slot, .. } => Some(slot.clone()),
                    GradTarget::Tensor(_) => None,
                })
                .collect();
            record_backward(self, parameters)?;
        }
        if gradient.is_none() && self.numel() != 1 {
            return Err(PyValueError::new_err(format!(
                "backward() without an explicit gradient requires a one-element tensor, got shape {:?}",
                self.source.value().dims()
            )));
        }
        if !self.requires_grad() {
            return Err(PyValueError::new_err(
                "backward() requires a tensor attached to an autodiff graph",
            ));
        }
        if self
            .backward_graphs
            .iter()
            .any(|graph| graph.consumed.get())
        {
            return Err(PyValueError::new_err(
                "backward() graph was already freed by a previous backward() call",
            ));
        }
        for target in &self.targets {
            target.validate_generation()?;
        }
        let output = self.operation_input(true, "backward")?;
        let root = match gradient {
            Some(gradient) => {
                let seed = gradient.source.value().float("backward gradient")?;
                if seed.dims() != output.dims() {
                    return Err(PyValueError::new_err(format!(
                        "backward() gradient shape {:?} does not match output shape {:?}",
                        seed.dims(),
                        output.dims()
                    )));
                }
                let dims = (0..output.rank()).collect::<Vec<_>>();
                output
                    .mul_broadcast(seed.detach())
                    .map_err(to_python_error)?
                    .sum_dims(&dims)
                    .reshape(vec![1])
                    .map_err(to_python_error)?
            }
            None => output,
        };
        for graph in &self.backward_graphs {
            graph.consumed.set(true);
        }
        let gradients = catch_unwind(AssertUnwindSafe(|| root.backward())).map_err(|_| {
            PyValueError::new_err(
                "backward() could not traverse the autodiff graph; it may already have been freed",
            )
        })?;
        for target in &self.targets {
            target.accumulate(&gradients).map_err(to_python_error)?;
            target.mark_tape_consumed();
        }
        raise_pending_device_error()?;
        Ok(())
    }

    /// Sum values over all, one, or several dimensions.
    #[pyo3(signature = (dim=None, keepdim=false))]
    fn sum(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool) -> PyResult<Self> {
        self.reduce(dim, keepdim, true)
    }

    /// Average values over all, one, or several dimensions.
    #[pyo3(signature = (dim=None, keepdim=false))]
    fn mean(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool) -> PyResult<Self> {
        self.reduce(dim, keepdim, false)
    }

    /// Return value-only maxima over all, one, or several dimensions.
    #[pyo3(signature = (dim=None, keepdim=false))]
    fn max(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool) -> PyResult<Self> {
        self.reduce_extreme(dim, keepdim, Extremum::Maximum)
    }

    /// Return value-only minima over all, one, or several dimensions.
    #[pyo3(signature = (dim=None, keepdim=false))]
    fn min(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool) -> PyResult<Self> {
        self.reduce_extreme(dim, keepdim, Extremum::Minimum)
    }

    /// Return indices of the first maximum over all values or one dimension.
    #[pyo3(signature = (dim=None, keepdim=false))]
    fn argmax(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool) -> PyResult<Self> {
        self.reduce_arg_extreme(dim, keepdim, true)
    }

    /// Return indices of the first minimum over all values or one dimension.
    #[pyo3(signature = (dim=None, keepdim=false))]
    fn argmin(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool) -> PyResult<Self> {
        self.reduce_arg_extreme(dim, keepdim, false)
    }

    /// Sort values along one dimension and return values plus source indices.
    #[pyo3(signature = (dim=-1, descending=false, stable=false))]
    fn sort(&self, dim: isize, descending: bool, stable: bool) -> PyResult<(Self, Self)> {
        if stable {
            return Err(PyNotImplementedError::new_err(
                "sort(stable=True) is not supported by the current backends",
            ));
        }
        let dim = shape::axis_value(dim, self.ndim(), false, "sort")?;
        self.ordered(dim, descending, None)
    }

    /// Return indices that sort values along one dimension.
    #[pyo3(signature = (dim=-1, descending=false, stable=false))]
    fn argsort(&self, dim: isize, descending: bool, stable: bool) -> PyResult<Self> {
        if stable {
            return Err(PyNotImplementedError::new_err(
                "argsort(stable=True) is not supported by the current backends",
            ));
        }
        let dim = shape::axis_value(dim, self.ndim(), false, "argsort")?;
        self.ordered(dim, descending, None)
            .map(|(_, indices)| indices)
    }

    /// Return the `k` largest or smallest values and their source indices.
    #[pyo3(signature = (k, dim=None, largest=true, sorted=true))]
    fn topk(
        &self,
        k: usize,
        dim: Option<isize>,
        largest: bool,
        sorted: bool,
    ) -> PyResult<(Self, Self)> {
        let dim = shape::axis_value(dim.unwrap_or(-1), self.ndim(), false, "topk")?;
        let size = self.source.value().dims()[dim];
        if k > size {
            return Err(PyValueError::new_err(format!(
                "topk k={k} exceeds dimension {dim} with size {size}"
            )));
        }
        let _ = sorted;
        self.ordered(dim, largest, Some(k))
    }

    /// Return coordinates of nonzero values, optionally grouped by dimension.
    #[pyo3(signature = (*, as_tuple=false))]
    fn nonzero(&self, py: Python<'_>, as_tuple: bool) -> PyResult<Py<PyAny>> {
        self.capture_unsupported("Tensor.nonzero() dynamic-shape query")?;
        let coordinates = match self.source.value().detach() {
            TensorValue::Float(value) => value.equal_scalar(0.0).logical_not().nonzero(),
            TensorValue::Int(value) => value.equal_scalar(0).logical_not().nonzero(),
            TensorValue::Bool(value) => value.nonzero(),
        };
        let dimensions = coordinates.dims();
        let rank = dimensions[0];
        let count = dimensions[1];
        if !as_tuple {
            let output = coordinates.permute(vec![1, 0]).map_err(to_python_error)?;
            return Ok(Py::new(py, Self::from_int_inner(output))?.into_any());
        }
        let mut outputs = Vec::with_capacity(rank);
        for axis in 0..rank {
            let output = if count == 0 {
                DynInt::empty(&[0], &coordinates.device(), tynx_core::DType::I64)
                    .map_err(to_python_error)?
            } else {
                coordinates
                    .clone()
                    .slice(&[
                        Slice::new(axis as isize, Some(axis as isize + 1), 1),
                        Slice::full(),
                    ])
                    .reshape(vec![count])
                    .map_err(to_python_error)?
            };
            outputs.push(Py::new(py, Self::from_int_inner(output))?);
        }
        Ok(PyTuple::new(py, outputs)?.unbind().into_any())
    }

    /// Take the elementwise maximum with a broadcastable tensor or scalar.
    fn maximum(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.elementwise_extreme(other, Extremum::Maximum)
    }

    /// Take the elementwise minimum with a broadcastable tensor or scalar.
    fn minimum(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.elementwise_extreme(other, Extremum::Minimum)
    }

    /// Apply rectified linear activation element-wise.
    fn relu(&self) -> PyResult<Self> {
        self.unary_captured(UnaryOp::Relu, |input| Ok(input.relu()))
    }

    /// Apply logistic sigmoid activation element-wise.
    fn sigmoid(&self) -> PyResult<Self> {
        self.unary_captured(UnaryOp::Sigmoid, |input| Ok(input.sigmoid()))
    }

    /// Apply hyperbolic tangent element-wise.
    fn tanh(&self) -> PyResult<Self> {
        self.unary_captured(UnaryOp::Tanh, |input| Ok(input.tanh()))
    }

    /// Apply the exponential function element-wise.
    fn exp(&self) -> PyResult<Self> {
        self.unary_captured(UnaryOp::Exp, |input| Ok(input.exp()))
    }

    /// Apply the natural logarithm element-wise.
    fn log(&self) -> PyResult<Self> {
        self.unary_captured(UnaryOp::Log, |input| Ok(input.log()))
    }

    /// Apply the square root element-wise.
    fn sqrt(&self) -> PyResult<Self> {
        self.unary_captured(UnaryOp::Sqrt, |input| Ok(input.sqrt()))
    }

    /// Apply Gaussian error linear unit activation element-wise.
    fn gelu(&self) -> PyResult<Self> {
        self.unary_captured(UnaryOp::Gelu, |input| Ok(input.gelu()))
    }

    /// Normalize values into probabilities along one dimension.
    fn softmax(&self, dim: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dim = shape::axis(dim, self.ndim(), false, "softmax")?;
        self.unary_captured(UnaryOp::Softmax(dim), |input| Ok(input.softmax(dim)))
    }

    /// Apply numerically stable log-softmax along one dimension.
    fn log_softmax(&self, dim: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dim = shape::axis(dim, self.ndim(), false, "log_softmax")?;
        self.unary_captured(UnaryOp::LogSoftmax(dim), |input| Ok(input.log_softmax(dim)))
    }

    /// Clamp values to optional scalar bounds.
    #[pyo3(signature = (min=None, max=None))]
    fn clamp(&self, min: Option<f64>, max: Option<f64>) -> PyResult<Self> {
        self.clip_bounds(min, max)
    }

    /// Alias for `clamp`.
    #[pyo3(signature = (min=None, max=None))]
    fn clip(&self, min: Option<f64>, max: Option<f64>) -> PyResult<Self> {
        self.clip_bounds(min, max)
    }

    /// Return a tensor with the same values and a new shape.
    #[pyo3(signature = (*shape))]
    fn reshape(&self, shape: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let output = shape::reshape(shape, self.numel())?;
        self.reshape_value(output)
    }

    /// PyTorch-compatible alias for `reshape`.
    #[pyo3(signature = (*shape))]
    fn view(&self, shape: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let output = shape::reshape(shape, self.numel())?;
        self.reshape_value(output)
    }

    /// Return a layout-safe tensor with the same values.
    fn contiguous(&self) -> PyResult<Self> {
        self.reshape_value(self.source.value().dims())
    }

    /// Broadcast singleton dimensions to a larger shape.
    #[pyo3(signature = (*shape))]
    fn expand(&self, shape: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let output = shape::expand(shape, &self.source.value().dims())?;
        match self.source.value() {
            TensorValue::Float(_) => self
                .unary_captured(UnaryOp::Expand(output.clone()), move |input| {
                    input.to_rank(output.len())?.expand(&output)
                }),
            value => {
                let expanded = match value.detach() {
                    TensorValue::Int(input) => TensorValue::Int(
                        input
                            .to_rank(output.len())
                            .and_then(|input| input.expand(&output))
                            .map_err(to_python_error)?,
                    ),
                    TensorValue::Bool(input) => TensorValue::Bool(
                        input
                            .to_rank(output.len())
                            .and_then(|input| input.expand(&output))
                            .map_err(to_python_error)?,
                    ),
                    TensorValue::Float(_) => unreachable!("float expansion handled above"),
                };
                let mut result = Self::from_value(expanded).with_inherited_int_bounds(self);
                result.trace = record_unary(self, UnaryOp::Expand(output))?;
                Ok(result)
            }
        }
    }

    /// Materialize repetitions along each dimension.
    #[pyo3(signature = (*repeats))]
    fn repeat(&self, repeats: &Bound<'_, PyTuple>) -> PyResult<Self> {
        self.capture_unsupported("Tensor.repeat")?;
        let repeats = shape::repeat(repeats, self.ndim())?;
        let tracking = is_grad_enabled();
        match self.source.value() {
            TensorValue::Float(_) => {
                let output = self
                    .operation_input(tracking, "Tensor.repeat")?
                    .to_rank(repeats.len())
                    .map_err(to_python_error)?
                    .repeat(&repeats);
                Ok(if tracking {
                    Self::from_operation(output, &[self])
                } else {
                    Self::from_inner(output)
                })
            }
            TensorValue::Int(input) => Ok(Self::from_int_inner(
                input
                    .to_rank(repeats.len())
                    .map_err(to_python_error)?
                    .repeat(&repeats),
            )
            .with_inherited_int_bounds(self)),
            TensorValue::Bool(input) => Ok(Self::from_value(TensorValue::Bool(
                input
                    .to_rank(repeats.len())
                    .map_err(to_python_error)?
                    .repeat(&repeats),
            ))),
        }
    }

    /// Flatten a contiguous range of dimensions.
    #[pyo3(signature = (start_dim=0, end_dim=-1))]
    fn flatten(&self, start_dim: isize, end_dim: isize) -> PyResult<Self> {
        let input_shape = self.source.value().dims();
        let start = shape::axis_value(start_dim, input_shape.len(), false, "flatten start_dim")?;
        let end = shape::axis_value(end_dim, input_shape.len(), false, "flatten end_dim")?;
        let output = shape::flatten(&input_shape, start, end)?;
        self.reshape_value(output)
    }

    /// Swap two tensor dimensions.
    fn transpose(&self, dim0: &Bound<'_, PyAny>, dim1: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rank = self.ndim();
        let dim0 = shape::axis(dim0, rank, false, "transpose")?;
        let dim1 = shape::axis(dim1, rank, false, "transpose")?;
        let mut axes = (0..rank).collect::<Vec<_>>();
        axes.swap(dim0, dim1);
        self.unary_captured(UnaryOp::Permute(axes.clone()), move |input| {
            input.permute(axes)
        })
    }

    /// Matrix transpose shorthand.
    #[getter(T)]
    fn matrix_transpose(&self) -> PyResult<Self> {
        if self.ndim() != 2 {
            return Err(PyValueError::new_err(format!(
                "Tensor.T requires a rank-2 matrix, got shape {:?}",
                self.source.value().dims()
            )));
        }
        let axes = vec![1, 0];
        self.unary_captured(UnaryOp::Permute(axes.clone()), move |input| {
            input.permute(axes)
        })
    }

    /// Reorder all tensor dimensions.
    #[pyo3(signature = (*dims))]
    fn permute(&self, dims: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let axes = shape::permutation(dims, self.ndim())?;
        self.unary_captured(UnaryOp::Permute(axes.clone()), move |input| {
            input.permute(axes)
        })
    }

    /// Remove singleton dimensions, or one selected singleton dimension.
    #[pyo3(signature = (dim=None))]
    fn squeeze(&self, dim: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let input_shape = self.source.value().dims();
        let dim = dim
            .map(|dim| shape::axis(dim, input_shape.len(), false, "squeeze"))
            .transpose()?;
        let output = shape::squeeze(&input_shape, dim);
        self.reshape_value(output)
    }

    /// Insert a singleton dimension.
    fn unsqueeze(&self, dim: &Bound<'_, PyAny>) -> PyResult<Self> {
        let input_shape = self.source.value().dims();
        let dim = shape::axis(dim, input_shape.len(), true, "unsqueeze")?;
        let output = shape::unsqueeze(&input_shape, dim)?;
        self.reshape_value(output)
    }

    /// Select values from this tensor and another branch using a boolean condition.
    #[pyo3(name = "where")]
    fn where_(&self, condition: PyRef<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let other_tensor = other.extract::<PyRef<'_, Self>>().ok();
        Self::where_operands(
            &condition,
            Some(self),
            None,
            other_tensor.as_deref(),
            other_tensor.is_none().then_some(other),
        )
    }

    /// Gather values along one dimension using an in-bounds, same-rank int64 index tensor.
    fn gather(&self, dim: &Bound<'_, PyAny>, index: PyRef<'_, Self>) -> PyResult<Self> {
        let dim = shape::axis(dim, self.ndim(), false, "gather")?;
        self.gather_impl(dim, &index)
    }

    /// Select whole slices along one dimension using a one-dimensional int64 tensor.
    fn index_select(&self, dim: isize, index: PyRef<'_, Self>) -> PyResult<Self> {
        let dim = shape::axis_value(dim, self.ndim(), false, "index_select")?;
        let input = self.source.value();
        let input_device = input.device();
        let checked_index = match index.source.value() {
            TensorValue::Int(index) if index.rank() == 1 => index,
            TensorValue::Int(index) => {
                return Err(PyValueError::new_err(format!(
                    "index_select index must be one-dimensional, got rank {}",
                    index.rank()
                )));
            }
            other => {
                return Err(PyTypeError::new_err(format!(
                    "index_select index must be an int64 Tensor, got {}",
                    other.dtype_name()
                )));
            }
        };
        ensure_index_device(&input_device, &checked_index.device(), "index_select")?;
        let checked_index = checked_index
            .checked_select_indices(input.dims()[dim], false)
            .map_err(index_error)?;
        self.select_indices(
            dim,
            checked_index,
            "Tensor.index_select",
            Some((&index, false)),
        )
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(
            other,
            BinaryOp::Add,
            DynTensor::add_broadcast,
            DynTensor::add_scalar,
        )
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(
            other,
            BinaryOp::Add,
            DynTensor::add_broadcast,
            DynTensor::add_scalar,
        )
    }

    fn __iadd__(&self, _other: &Bound<'_, PyAny>) -> PyResult<()> {
        self.reject_in_place("+=")
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(
            other,
            BinaryOp::Subtract,
            DynTensor::sub_broadcast,
            DynTensor::sub_scalar,
        )
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let scalar = extract_scalar_operand(other)?;
        self.unary_captured(UnaryOp::ReverseSubtractScalar(scalar), |input| {
            Ok(input.negated().add_scalar(scalar))
        })
    }

    fn __isub__(&self, _other: &Bound<'_, PyAny>) -> PyResult<()> {
        self.reject_in_place("-=")
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(
            other,
            BinaryOp::Multiply,
            DynTensor::mul_broadcast,
            DynTensor::mul_scalar,
        )
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(
            other,
            BinaryOp::Multiply,
            DynTensor::mul_broadcast,
            DynTensor::mul_scalar,
        )
    }

    fn __imul__(&self, _other: &Bound<'_, PyAny>) -> PyResult<()> {
        self.reject_in_place("*=")
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(
            other,
            BinaryOp::Divide,
            DynTensor::div_broadcast,
            DynTensor::div_scalar,
        )
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let scalar = extract_scalar_operand(other)?;
        self.unary_captured(UnaryOp::ReverseDivideScalar(scalar), |input| {
            Ok(input.reciprocal().mul_scalar(scalar))
        })
    }

    fn __itruediv__(&self, _other: &Bound<'_, PyAny>) -> PyResult<()> {
        self.reject_in_place("/=")
    }

    fn __matmul__(&self, other: PyRef<'_, Self>) -> PyResult<Self> {
        let left_shape = self.source.value().dims();
        let right_shape = other.source.value().dims();
        match (left_shape.as_slice(), right_shape.as_slice()) {
            ([left], [right]) => {
                let left = *left;
                let right = *right;
                self.binary(&other, BinaryOp::Matmul, move |left_value, right_value| {
                    left_value
                        .reshape(vec![1, left])?
                        .matmul(right_value.reshape(vec![right, 1])?)?
                        .reshape(vec![1])
                })
            }
            ([rows, inner], [right]) => {
                let rows = *rows;
                let _inner = *inner;
                let right = *right;
                self.binary(&other, BinaryOp::Matmul, move |left_value, right_value| {
                    left_value
                        .matmul(right_value.reshape(vec![right, 1])?)?
                        .reshape(vec![rows])
                })
            }
            ([left], [inner, columns]) => {
                let left = *left;
                let _inner = *inner;
                let columns = *columns;
                self.binary(&other, BinaryOp::Matmul, move |left_value, right_value| {
                    left_value
                        .reshape(vec![1, left])?
                        .matmul(right_value)?
                        .reshape(vec![columns])
                })
            }
            _ => self.binary(&other, BinaryOp::Matmul, DynTensor::matmul),
        }
    }

    fn __imatmul__(&self, _other: &Bound<'_, PyAny>) -> PyResult<()> {
        self.reject_in_place("@=")
    }

    fn __neg__(&self) -> PyResult<Self> {
        self.unary_captured(UnaryOp::Negate, |input| Ok(input.negated()))
    }

    fn abs(&self) -> PyResult<Self> {
        match self.source.value() {
            TensorValue::Float(_) => self.unary(|input| Ok(input.abs())),
            TensorValue::Int(value) => Ok(Self::from_value(TensorValue::Int(value.abs()))),
            TensorValue::Bool(_) => {
                Err(PyTypeError::new_err("abs() does not support bool Tensors"))
            }
        }
    }

    fn __abs__(&self) -> PyResult<Self> {
        self.abs()
    }

    fn __pow__(
        &self,
        exponent: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if modulo.is_some_and(|value| !value.is_none()) {
            return Err(PyTypeError::new_err(
                "Tensor power does not support a modulo argument",
            ));
        }
        self.arithmetic(
            exponent,
            BinaryOp::Power,
            DynTensor::powf_broadcast,
            |base, exponent| base.powf_scalar(exponent),
        )
    }

    fn __rpow__(
        &self,
        base: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if modulo.is_some_and(|value| !value.is_none()) {
            return Err(PyTypeError::new_err(
                "Tensor power does not support a modulo argument",
            ));
        }
        let base = extract_scalar_operand(base)?;
        self.unary_captured(UnaryOp::ReversePowerScalar(base), move |exponent| {
            exponent.clone().full_like(base).powf_broadcast(exponent)
        })
    }

    fn __ipow__(
        &self,
        _other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.reject_in_place("**=")
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.compare(other, Comparison::Equal)
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.compare(other, Comparison::NotEqual)
    }

    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.compare(other, Comparison::Less)
    }

    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.compare(other, Comparison::LessEqual)
    }

    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.compare(other, Comparison::Greater)
    }

    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.compare(other, Comparison::GreaterEqual)
    }

    fn __and__(&self, other: PyRef<'_, Self>) -> PyResult<Self> {
        self.mask_binary(&other, MaskOperation::And)
    }

    fn __or__(&self, other: PyRef<'_, Self>) -> PyResult<Self> {
        self.mask_binary(&other, MaskOperation::Or)
    }

    fn __xor__(&self, other: PyRef<'_, Self>) -> PyResult<Self> {
        self.mask_binary(&other, MaskOperation::Xor)
    }

    fn __invert__(&self) -> PyResult<Self> {
        self.source
            .value()
            .detach()
            .mask_not()
            .map(Self::from_value)
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={}, requires_grad={})",
            self.source.value().dims().as_slice(),
            self.dtype(),
            self.requires_grad()
        )
    }
}

impl PyTensor {
    fn advanced_getitem(&self, index_tensor: &Self) -> PyResult<Self> {
        let input = self.source.value();
        let size = input.dims()[0];
        let input_device = input.device();
        let (index, trace_index) = match index_tensor.source.value() {
            TensorValue::Int(index) if index.rank() == 1 => {
                ensure_index_device(&input_device, &index.device(), "advanced indexing")?;
                (
                    index
                        .checked_select_indices(size, true)
                        .map_err(index_error)?,
                    Some((index_tensor, true)),
                )
            }
            TensorValue::Int(index) => {
                return Err(PyIndexError::new_err(format!(
                    "only one-dimensional integer tensor indices are supported, got rank {}",
                    index.rank()
                )));
            }
            TensorValue::Bool(mask) if mask.rank() == 1 && mask.dims()[0] == size => {
                self.capture_unsupported("boolean advanced indexing with a dynamic output shape")?;
                ensure_index_device(&input_device, &mask.device(), "advanced indexing")?;
                let coordinates = mask.nonzero();
                let count = coordinates.dims()[1];
                let indices = if count == 0 {
                    DynInt::empty(&[0], &coordinates.device(), tynx_core::DType::I64)
                        .map_err(to_python_error)?
                } else {
                    coordinates.reshape(vec![count]).map_err(to_python_error)?
                };
                (indices, None)
            }
            TensorValue::Bool(mask) if mask.rank() != 1 => {
                return Err(PyIndexError::new_err(format!(
                    "only one-dimensional boolean tensor indices are supported, got rank {}",
                    mask.rank()
                )));
            }
            TensorValue::Bool(mask) => {
                return Err(PyIndexError::new_err(format!(
                    "boolean index length {} does not match first dimension size {size}",
                    mask.dims()[0]
                )));
            }
            other => {
                return Err(PyIndexError::new_err(format!(
                    "advanced tensor indices must have int64 or bool dtype, got {}",
                    other.dtype_name()
                )));
            }
        };
        self.select_indices(0, index, "Tensor.__getitem__", trace_index)
    }

    fn select_indices(
        &self,
        dim: usize,
        index: DynInt,
        operation: &str,
        trace_index: Option<(&Self, bool)>,
    ) -> PyResult<Self> {
        let tracking = is_grad_enabled();
        let mut output = match self.source.value() {
            TensorValue::Float(_) => {
                let output = self
                    .operation_input(tracking, operation)?
                    .select(dim, index)
                    .map_err(to_python_error)?;
                if tracking {
                    Self::from_operation(output, &[self])
                } else {
                    Self::from_inner(output)
                }
            }
            TensorValue::Int(value) => value
                .select(dim, index)
                .map(Self::from_int_inner)
                .map_err(to_python_error)?,
            TensorValue::Bool(value) => value
                .select(dim, index)
                .map(|value| Self::from_value(TensorValue::Bool(value)))
                .map_err(to_python_error)?,
        };
        if let Some((index, allow_negative)) = trace_index {
            output.trace = record_index_select(self, dim, index, allow_negative)?;
        }
        Ok(output)
    }

    fn ordered(&self, dim: usize, descending: bool, k: Option<usize>) -> PyResult<(Self, Self)> {
        self.capture_unsupported("Tensor ordering")?;
        let tracking = is_grad_enabled();
        let index_bounds = IntBounds::Range {
            min: 0,
            max: self.source.value().dims()[dim].saturating_sub(1) as i64,
        };
        match self.source.value() {
            TensorValue::Float(_) => {
                let input = self.operation_input(tracking, "Tensor ordering")?;
                let (values, indices) = match k {
                    Some(k) => input.topk_ordered(k, dim, descending),
                    None => input.sort_with_indices(dim, descending),
                };
                let values = if tracking {
                    Self::from_operation(values, &[self])
                } else {
                    Self::from_inner(values)
                };
                Ok((
                    values,
                    Self::from_int_inner_with_bounds(indices, index_bounds),
                ))
            }
            TensorValue::Int(input) => {
                let (values, indices) = match k {
                    Some(k) => input.topk_ordered(k, dim, descending),
                    None => input.sort_with_indices(dim, descending),
                };
                Ok((
                    Self::from_int_inner(values),
                    Self::from_int_inner_with_bounds(indices, index_bounds),
                ))
            }
            TensorValue::Bool(_) => Err(PyTypeError::new_err(
                "sort, argsort, and topk do not support bool Tensors",
            )),
        }
    }

    fn reshape_value(&self, output: Vec<usize>) -> PyResult<Self> {
        match self.source.value() {
            TensorValue::Float(_) => self
                .unary_captured(UnaryOp::Reshape(output.clone()), move |input| {
                    input.reshape(output)
                }),
            value => {
                let mut result = Self::from_value(value.reshape(output.clone())?)
                    .with_inherited_int_bounds(self);
                result.trace = record_unary(self, UnaryOp::Reshape(output))?;
                Ok(result)
            }
        }
    }

    fn clip_bounds(&self, min: Option<f64>, max: Option<f64>) -> PyResult<Self> {
        if min.is_none() && max.is_none() {
            return Err(PyValueError::new_err(
                "clamp requires at least one of min or max",
            ));
        }
        self.unary_captured(
            UnaryOp::Clamp { min, max },
            |input| Ok(input.clip(min, max)),
        )
    }

    fn reduce(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool, sum: bool) -> PyResult<Self> {
        let input_shape = self.source.value().dims();
        let spec = ReductionSpec::from_python(dim, &input_shape, keepdim)?;
        let capture_op = if sum {
            UnaryOp::Sum {
                dims: spec.dims.clone(),
                output_shape: spec.output_shape.clone(),
            }
        } else {
            UnaryOp::Mean {
                dims: spec.dims.clone(),
                output_shape: spec.output_shape.clone(),
            }
        };
        self.unary_captured(capture_op, move |input| {
            let reduced = if sum {
                input.sum_dims(&spec.dims)
            } else {
                input.mean_dims(&spec.dims)
            };
            reduced.reshape(spec.output_shape)
        })
    }

    fn reduce_extreme(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        keepdim: bool,
        extremum: Extremum,
    ) -> PyResult<Self> {
        let input_shape = self.source.value().dims();
        let spec = ReductionSpec::from_python(dim, &input_shape, keepdim)?;
        let reduce_all = dim.is_none() && input_shape.len() > 1;
        let reduction_dims = if reduce_all {
            vec![0]
        } else {
            spec.dims.clone()
        };
        let tracking = is_grad_enabled();
        match self.source.value() {
            TensorValue::Float(_) => {
                let mut input = self.operation_input(tracking, extremum.name())?;
                if reduce_all {
                    input = input.reshape(vec![self.numel()]).map_err(to_python_error)?;
                }
                let inner = extremum
                    .float_reduce(input, &reduction_dims)
                    .and_then(|value| value.reshape(spec.output_shape))
                    .map_err(to_python_error)?;
                Ok(if tracking {
                    Self::from_operation(inner, &[self])
                } else {
                    Self::from_inner(inner)
                })
            }
            value => {
                let mut value = value.detach();
                if reduce_all {
                    value = value.reshape(vec![self.numel()])?;
                }
                extrema::reduce(value, &reduction_dims, spec.output_shape, extremum)
                    .map(Self::from_value)
            }
        }
    }

    fn reduce_arg_extreme(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        keepdim: bool,
        maximum: bool,
    ) -> PyResult<Self> {
        let input_shape = self.source.value().dims();
        let operation = if maximum { "argmax" } else { "argmin" };
        let (value, axis, output_shape) = match dim {
            Some(dim) => {
                let axis = shape::axis(dim, input_shape.len(), false, operation)?;
                let mut output_shape = input_shape.clone();
                if keepdim {
                    output_shape[axis] = 1;
                } else {
                    output_shape.remove(axis);
                    if output_shape.is_empty() {
                        output_shape.push(1);
                    }
                }
                (self.source.value().detach(), axis, output_shape)
            }
            None => {
                let output_shape = if keepdim {
                    vec![1; input_shape.len()]
                } else {
                    vec![1]
                };
                (
                    self.source.value().detach().reshape(vec![self.numel()])?,
                    0,
                    output_shape,
                )
            }
        };
        let size = value.dims()[axis];
        let bounds = if size == 0 {
            IntBounds::Empty
        } else {
            IntBounds::Range {
                min: 0,
                max: size.saturating_sub(1) as i64,
            }
        };
        extrema::arg(value, axis, output_shape, maximum)
            .map(|value| Self::from_value_with_int_bounds(value, Some(bounds)))
    }
}

fn extract_scalar_operand(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    value.extract::<f64>().map_err(|_| {
        PyTypeError::new_err("Tensor arithmetic expects another Tensor or a real number")
    })
}

fn ensure_index_device(left: &Device, right: &Device, operation: &str) -> PyResult<()> {
    if left == right {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{operation} requires input and index tensors on the same device, got {left:?} and {right:?}"
    )))
}

fn index_error(error: tynx_core::TynxError) -> PyErr {
    PyIndexError::new_err(error.to_string())
}

/// Select values from two tensor/scalar branches using a boolean tensor condition.
#[pyfunction(name = "where")]
pub(crate) fn where_py(
    condition: PyRef<'_, PyTensor>,
    input: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
) -> PyResult<PyTensor> {
    PyTensor::where_from_python(&condition, input, other)
}

#[pyfunction(name = "maximum")]
pub(crate) fn maximum_py(
    input: PyRef<'_, PyTensor>,
    other: &Bound<'_, PyAny>,
) -> PyResult<PyTensor> {
    input.elementwise_extreme(other, Extremum::Maximum)
}

#[pyfunction(name = "minimum")]
pub(crate) fn minimum_py(
    input: PyRef<'_, PyTensor>,
    other: &Bound<'_, PyAny>,
) -> PyResult<PyTensor> {
    input.elementwise_extreme(other, Extremum::Minimum)
}

#[pyfunction(name = "sort", signature = (input, dim=-1, descending=false, stable=false))]
pub(crate) fn sort_py(
    input: PyRef<'_, PyTensor>,
    dim: isize,
    descending: bool,
    stable: bool,
) -> PyResult<(PyTensor, PyTensor)> {
    input.sort(dim, descending, stable)
}

#[pyfunction(name = "argsort", signature = (input, dim=-1, descending=false, stable=false))]
pub(crate) fn argsort_py(
    input: PyRef<'_, PyTensor>,
    dim: isize,
    descending: bool,
    stable: bool,
) -> PyResult<PyTensor> {
    input.argsort(dim, descending, stable)
}

#[pyfunction(name = "topk", signature = (input, k, dim=None, largest=true, sorted=true))]
pub(crate) fn topk_py(
    input: PyRef<'_, PyTensor>,
    k: usize,
    dim: Option<isize>,
    largest: bool,
    sorted: bool,
) -> PyResult<(PyTensor, PyTensor)> {
    input.topk(k, dim, largest, sorted)
}

#[pyfunction(name = "nonzero", signature = (input, *, as_tuple=false))]
pub(crate) fn nonzero_py(
    py: Python<'_>,
    input: PyRef<'_, PyTensor>,
    as_tuple: bool,
) -> PyResult<Py<PyAny>> {
    input.nonzero(py, as_tuple)
}

#[pyfunction(name = "index_select")]
pub(crate) fn index_select_py(
    input: PyRef<'_, PyTensor>,
    dim: isize,
    index: PyRef<'_, PyTensor>,
) -> PyResult<PyTensor> {
    input.index_select(dim, index)
}
