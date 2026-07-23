//! CPython tracing projection over the binding-neutral capture runtime.

use std::{
    cell::RefCell,
    collections::HashMap,
    rc::{Rc, Weak},
};

use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyTuple};
use tynx_capture::{
    BinaryOp, CapturedOperation, CapturedOptimizer, Graph, GraphBuilder, UnaryOp, ValueId,
};
use tynx_train::{ParamId, ParameterSlot};

use crate::{grad_mode::is_grad_enabled, tensor::PyTensor, to_python_error};

thread_local! {
    static ACTIVE_CAPTURE: RefCell<Option<Weak<CaptureState>>> = const { RefCell::new(None) };
}

#[derive(Debug)]
struct CaptureInner {
    builder: Option<GraphBuilder>,
    parameters: HashMap<ParamId, ValueId>,
    value_inputs: HashMap<ValueId, usize>,
    next_input: usize,
    index_constraints: Vec<IndexConstraint>,
    failure: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct IndexConstraint {
    input: usize,
    size: usize,
    dim: usize,
}

/// Shared state carried by tensors produced during one first-call trace.
#[derive(Debug)]
pub(crate) struct CaptureState {
    fullgraph: bool,
    inner: RefCell<CaptureInner>,
}

impl CaptureState {
    fn new(fullgraph: bool) -> Rc<Self> {
        Rc::new(Self {
            fullgraph,
            inner: RefCell::new(CaptureInner {
                builder: Some(GraphBuilder::new()),
                parameters: HashMap::new(),
                value_inputs: HashMap::new(),
                next_input: 0,
                index_constraints: Vec::new(),
                failure: None,
            }),
        })
    }

    fn input(&self, tensor: &PyTensor) -> PyResult<ValueId> {
        let value = tensor.detached_runtime_value();
        let value =
            self.with_builder(|builder| builder.input_value(&value).map_err(to_python_error))?;
        let mut inner = self.inner.borrow_mut();
        let input = inner.next_input;
        inner.next_input += 1;
        inner.value_inputs.insert(value, input);
        Ok(value)
    }

    fn parameter(&self, slot: ParameterSlot) -> PyResult<ValueId> {
        let mut inner = self.inner.borrow_mut();
        if let Some(value) = inner.parameters.get(&slot.id()).copied() {
            return Ok(value);
        }
        let builder = inner.builder.as_mut().ok_or_else(capture_finished)?;
        let value = builder.parameter(slot.clone());
        inner.parameters.insert(slot.id(), value);
        Ok(value)
    }

    fn unary(&self, op: UnaryOp, input: ValueId) -> PyResult<ValueId> {
        let preserves_indices = matches!(
            op,
            UnaryOp::Reshape(_) | UnaryOp::Expand(_) | UnaryOp::Permute(_)
        );
        let origin = preserves_indices
            .then(|| self.inner.borrow().value_inputs.get(&input).copied())
            .flatten();
        let value =
            self.with_builder(|builder| builder.unary(op, input).map_err(to_python_error))?;
        if let Some(origin) = origin {
            self.inner.borrow_mut().value_inputs.insert(value, origin);
        }
        Ok(value)
    }

    fn binary(&self, op: BinaryOp, left: ValueId, right: ValueId) -> PyResult<ValueId> {
        self.with_builder(|builder| builder.binary(op, left, right).map_err(to_python_error))
    }

    fn gather(
        &self,
        input: ValueId,
        dim: usize,
        size: usize,
        indices: ValueId,
    ) -> PyResult<ValueId> {
        let origin = self.inner.borrow().value_inputs.get(&indices).copied();
        let value = self.with_builder(|builder| {
            if origin.is_some() {
                builder.gather(input, dim, indices)
            } else {
                builder.gather_checked(input, dim, indices)
            }
            .map_err(to_python_error)
        })?;
        if let Some(origin) = origin {
            let constraint = IndexConstraint {
                input: origin,
                size,
                dim,
            };
            let mut inner = self.inner.borrow_mut();
            if !inner.index_constraints.iter().any(|existing| {
                existing.input == constraint.input
                    && existing.size == constraint.size
                    && existing.dim == constraint.dim
            }) {
                inner.index_constraints.push(constraint);
            }
        }
        Ok(value)
    }

    fn index_select(
        &self,
        input: ValueId,
        dim: usize,
        indices: ValueId,
        allow_negative: bool,
    ) -> PyResult<ValueId> {
        self.with_builder(|builder| {
            builder
                .index_select(input, dim, indices, allow_negative)
                .map_err(to_python_error)
        })
    }

    fn operation(
        &self,
        operation: Rc<dyn CapturedOperation>,
        inputs: Vec<ValueId>,
    ) -> PyResult<Vec<ValueId>> {
        self.with_builder(|builder| {
            builder
                .operation(operation, inputs)
                .map_err(to_python_error)
        })
    }

    fn zero_grad(&self, parameters: Vec<ParameterSlot>) -> PyResult<()> {
        self.with_builder(|builder| {
            builder.zero_grad(parameters);
            Ok(())
        })
    }

    fn backward(&self, loss: ValueId, parameters: Vec<ParameterSlot>) -> PyResult<()> {
        self.with_builder(|builder| builder.backward(loss, parameters).map_err(to_python_error))
    }

    fn optimizer_step(&self, optimizer: Rc<dyn CapturedOptimizer>) -> PyResult<()> {
        self.with_builder(|builder| {
            builder.optimizer_step(optimizer);
            Ok(())
        })
    }

    fn with_builder<T>(
        &self,
        operation: impl FnOnce(&mut GraphBuilder) -> PyResult<T>,
    ) -> PyResult<T> {
        let mut inner = self.inner.borrow_mut();
        let builder = inner.builder.as_mut().ok_or_else(capture_finished)?;
        operation(builder)
    }

    fn unsupported(&self, reason: impl Into<String>) -> PyResult<()> {
        let reason = reason.into();
        if self.fullgraph {
            return Err(PyRuntimeError::new_err(format!(
                "tynx.compile(fullgraph=True) cannot capture {reason}"
            )));
        }
        let mut inner = self.inner.borrow_mut();
        inner.failure.get_or_insert(reason);
        Ok(())
    }

    fn finish(&self, outputs: Vec<ValueId>) -> PyResult<Option<(Graph, Vec<IndexConstraint>)>> {
        let mut inner = self.inner.borrow_mut();
        if let Some(reason) = inner.failure.take() {
            inner.builder = None;
            return if self.fullgraph {
                Err(PyRuntimeError::new_err(format!(
                    "tynx.compile(fullgraph=True) cannot capture {reason}"
                )))
            } else {
                Ok(None)
            };
        }
        let builder = inner.builder.take().ok_or_else(capture_finished)?;
        let constraints = std::mem::take(&mut inner.index_constraints);
        builder
            .finish(outputs)
            .map(|graph| Some((graph, constraints)))
            .map_err(to_python_error)
    }

    fn abort(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.builder = None;
        inner.parameters.clear();
        inner.value_inputs.clear();
        inner.index_constraints.clear();
        inner.failure = None;
    }
}

fn capture_finished() -> PyErr {
    PyRuntimeError::new_err("capture session has already finished")
}

/// Trace metadata carried alongside an eager tensor value.
#[derive(Debug, Clone)]
pub(crate) struct TraceValue {
    state: Rc<CaptureState>,
    value: ValueId,
}

impl TraceValue {
    pub(crate) fn unary(&self, op: UnaryOp) -> PyResult<Option<Self>> {
        self.state.unary(op, self.value).map(|value| {
            Some(Self {
                state: self.state.clone(),
                value,
            })
        })
    }

    pub(crate) fn unsupported(&self, reason: &str) -> PyResult<()> {
        self.state.unsupported(reason)
    }
}

fn activate(state: &Rc<CaptureState>) -> PyResult<()> {
    ACTIVE_CAPTURE.with(|active| {
        let mut active = active.borrow_mut();
        if let Some(existing) = active.as_ref().and_then(Weak::upgrade)
            && !Rc::ptr_eq(&existing, state)
        {
            return Err(PyRuntimeError::new_err(
                "nested tynx.compile tracing sessions are not supported",
            ));
        }
        *active = Some(Rc::downgrade(state));
        Ok(())
    })
}

fn deactivate(state: &Rc<CaptureState>) {
    ACTIVE_CAPTURE.with(|active| {
        let mut active = active.borrow_mut();
        if active
            .as_ref()
            .and_then(Weak::upgrade)
            .is_some_and(|current| Rc::ptr_eq(&current, state))
        {
            *active = None;
        }
    });
}

fn active() -> Option<Rc<CaptureState>> {
    ACTIVE_CAPTURE.with(|active| active.borrow().as_ref().and_then(Weak::upgrade))
}

fn trace_for(tensor: &PyTensor) -> PyResult<Option<TraceValue>> {
    if let Some(trace) = tensor.trace() {
        return Ok(Some(trace.clone()));
    }
    let (Some(state), Some(slot)) = (active(), tensor.parameter_slot()) else {
        return Ok(None);
    };
    let value = state.parameter(slot)?;
    Ok(Some(TraceValue { state, value }))
}

pub(crate) fn record_unary(tensor: &PyTensor, op: UnaryOp) -> PyResult<Option<TraceValue>> {
    trace_for(tensor)?
        .map(|trace| trace.unary(op))
        .transpose()
        .map(Option::flatten)
}

pub(crate) fn record_unsupported(tensor: &PyTensor, reason: &str) -> PyResult<()> {
    if let Some(trace) = trace_for(tensor)? {
        trace.unsupported(reason)?;
    }
    Ok(())
}

pub(crate) fn record_zero_grad(parameters: &[ParameterSlot]) -> PyResult<()> {
    if let Some(state) = active() {
        state.zero_grad(parameters.to_vec())?;
    }
    Ok(())
}

pub(crate) fn record_backward(loss: &PyTensor, parameters: Vec<ParameterSlot>) -> PyResult<()> {
    let Some(active) = active() else {
        return Ok(());
    };
    let Some(trace) = trace_for(loss)? else {
        return active.unsupported("backward from a loss disconnected from captured inputs");
    };
    if !Rc::ptr_eq(&active, &trace.state) {
        return active.unsupported("backward from another capture session");
    }
    active.backward(trace.value, parameters)
}

pub(crate) fn record_optimizer_step(optimizer: Rc<dyn CapturedOptimizer>) -> PyResult<()> {
    if let Some(state) = active() {
        state.optimizer_step(optimizer)?;
    }
    Ok(())
}

pub(crate) fn record_binary(
    left: &PyTensor,
    right: &PyTensor,
    op: BinaryOp,
) -> PyResult<Option<TraceValue>> {
    let left_trace = trace_for(left)?;
    let right_trace = trace_for(right)?;
    let state = match (&left_trace, &right_trace) {
        (None, None) => return Ok(None),
        (Some(trace), None) => {
            trace.state.unsupported(
                "a closed-over Tensor value; pass changing tensors as function inputs",
            )?;
            return Ok(None);
        }
        (None, Some(trace)) => {
            trace.state.unsupported(
                "a closed-over Tensor value; pass changing tensors as function inputs",
            )?;
            return Ok(None);
        }
        (Some(left), Some(right)) if Rc::ptr_eq(&left.state, &right.state) => left.state.clone(),
        (Some(left), Some(_)) => {
            left.state
                .unsupported("tensors from different capture sessions")?;
            return Ok(None);
        }
    };
    let left = left_trace.expect("matched as a traced value").value;
    let right = right_trace.expect("matched as a traced value").value;
    state
        .binary(op, left, right)
        .map(|value| Some(TraceValue { state, value }))
}

pub(crate) fn record_gather(
    input: &PyTensor,
    dim: usize,
    size: usize,
    indices: &PyTensor,
) -> PyResult<Option<TraceValue>> {
    let input_trace = trace_for(input)?;
    let index_trace = trace_for(indices)?;
    let state = match (&input_trace, &index_trace) {
        (None, None) => return Ok(None),
        (Some(trace), None) | (None, Some(trace)) => {
            trace.state.unsupported(
                "a closed-over Tensor value; pass changing tensors as function inputs",
            )?;
            return Ok(None);
        }
        (Some(input), Some(index)) if Rc::ptr_eq(&input.state, &index.state) => input.state.clone(),
        (Some(input), Some(_)) => {
            input
                .state
                .unsupported("tensors from different capture sessions")?;
            return Ok(None);
        }
    };
    let input = input_trace.expect("matched as a traced value").value;
    let indices = index_trace.expect("matched as a traced value").value;
    state
        .gather(input, dim, size, indices)
        .map(|value| Some(TraceValue { state, value }))
}

pub(crate) fn record_index_select(
    input: &PyTensor,
    dim: usize,
    indices: &PyTensor,
    allow_negative: bool,
) -> PyResult<Option<TraceValue>> {
    let input_trace = trace_for(input)?;
    let index_trace = trace_for(indices)?;
    let state = match (&input_trace, &index_trace) {
        (None, None) => return Ok(None),
        (Some(trace), None) | (None, Some(trace)) => {
            trace.state.unsupported(
                "a closed-over Tensor value; pass changing tensors as function inputs",
            )?;
            return Ok(None);
        }
        (Some(input), Some(index)) if Rc::ptr_eq(&input.state, &index.state) => input.state.clone(),
        (Some(input), Some(_)) => {
            input
                .state
                .unsupported("tensors from different capture sessions")?;
            return Ok(None);
        }
    };
    let input = input_trace.expect("matched as a traced value").value;
    let indices = index_trace.expect("matched as a traced value").value;
    state
        .index_select(input, dim, indices, allow_negative)
        .map(|value| Some(TraceValue { state, value }))
}

pub(crate) fn record_operation(
    inputs: &[&PyTensor],
    operation: Rc<dyn CapturedOperation>,
) -> PyResult<Option<Vec<TraceValue>>> {
    let traces = inputs
        .iter()
        .map(|input| trace_for(input))
        .collect::<PyResult<Vec<_>>>()?;
    let Some(state) = traces
        .iter()
        .flatten()
        .next()
        .map(|trace| trace.state.clone())
    else {
        return Ok(None);
    };
    if traces.iter().any(Option::is_none) {
        state
            .unsupported("a closed-over Tensor value; pass changing tensors as function inputs")?;
        return Ok(None);
    }
    if traces
        .iter()
        .flatten()
        .any(|trace| !Rc::ptr_eq(&state, &trace.state))
    {
        state.unsupported("tensors from different capture sessions")?;
        return Ok(None);
    }
    let inputs = traces
        .into_iter()
        .map(|trace| trace.expect("all operation inputs were traced").value)
        .collect();
    let outputs = state.operation(operation, inputs)?;
    Ok(Some(
        outputs
            .into_iter()
            .map(|value| TraceValue {
                state: state.clone(),
                value,
            })
            .collect(),
    ))
}

/// Native first-call trace session used by the public Python decorator.
#[pyclass(name = "_CaptureSession", unsendable)]
pub(crate) struct PyCaptureSession {
    state: Rc<CaptureState>,
}

#[pymethods]
impl PyCaptureSession {
    #[new]
    #[pyo3(signature = (*, fullgraph=false))]
    fn new(fullgraph: bool) -> Self {
        Self {
            state: CaptureState::new(fullgraph),
        }
    }

    fn input(&self, tensor: PyRef<'_, PyTensor>) -> PyResult<PyTensor> {
        if tensor.parameter_slot().is_some() {
            return Err(PyRuntimeError::new_err(
                "Parameter objects cannot be compile inputs; close over stable model parameters",
            ));
        }
        activate(&self.state)?;
        let value = self.state.input(&tensor)?;
        Ok(tensor.with_trace(TraceValue {
            state: self.state.clone(),
            value,
        }))
    }

    fn finish(&self, outputs: &Bound<'_, PyTuple>) -> PyResult<Option<PyCapturedGraph>> {
        deactivate(&self.state);
        let mut values = Vec::with_capacity(outputs.len());
        for output in outputs.iter() {
            let output = output.extract::<PyRef<'_, PyTensor>>()?;
            let Some(trace) = output.trace() else {
                self.state
                    .unsupported("a function output disconnected from its Tensor inputs")?;
                return Ok(None);
            };
            if !Rc::ptr_eq(&self.state, &trace.state) {
                self.state
                    .unsupported("an output from another capture session")?;
                return Ok(None);
            }
            values.push(trace.value);
        }
        self.state
            .finish(values)
            .map(|graph| graph.map(|(graph, constraints)| PyCapturedGraph::new(graph, constraints)))
    }

    fn release(&self, output: PyRef<'_, PyTensor>) -> PyTensor {
        output.without_trace()
    }

    fn abort(&self) {
        deactivate(&self.state);
        self.state.abort();
    }
}

impl Drop for PyCaptureSession {
    fn drop(&mut self) {
        deactivate(&self.state);
    }
}

/// Immutable native graph replay object.
#[pyclass(name = "_CapturedGraph", frozen, unsendable)]
pub(crate) struct PyCapturedGraph {
    graph: Graph,
    parameters: Vec<ParameterSlot>,
    index_constraints: Vec<IndexConstraint>,
}

impl PyCapturedGraph {
    fn new(graph: Graph, index_constraints: Vec<IndexConstraint>) -> Self {
        let parameters = graph.parameters();
        Self {
            graph,
            parameters,
            index_constraints,
        }
    }

    fn inputs<'py>(
        inputs: &'py Bound<'py, PyTuple>,
        operation: &str,
    ) -> PyResult<Vec<PyRef<'py, PyTensor>>> {
        inputs
            .iter()
            .map(|input| {
                input.extract::<PyRef<'py, PyTensor>>().map_err(|_| {
                    PyRuntimeError::new_err(format!(
                        "{operation} accepts only Tensor positional arguments"
                    ))
                })
            })
            .collect()
    }
}

#[pymethods]
impl PyCapturedGraph {
    #[getter]
    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    #[getter]
    fn structure_valid(&self) -> bool {
        self.graph.validate_parameters().is_ok()
    }

    #[pyo3(signature = (*inputs))]
    fn matches(&self, inputs: &Bound<'_, PyTuple>) -> PyResult<bool> {
        let inputs = Self::inputs(inputs, "captured graph matching")?;
        let values = inputs
            .iter()
            .map(|input| input.detached_runtime_value())
            .collect::<Vec<_>>();
        Ok(self.graph.validate_value_inputs(&values).is_ok())
    }

    #[pyo3(signature = (*inputs))]
    fn __call__(&self, py: Python<'_>, inputs: &Bound<'_, PyTuple>) -> PyResult<Py<PyTuple>> {
        let inputs = Self::inputs(inputs, "captured graph replay")?;
        for constraint in &self.index_constraints {
            inputs[constraint.input].validate_index_bounds(
                constraint.size,
                constraint.dim,
                "gather",
            )?;
        }
        let tracking = is_grad_enabled();
        let values = inputs
            .iter()
            .map(|input| input.operation_runtime_value(tracking, "captured graph replay"))
            .collect::<PyResult<Vec<_>>>()?;
        let outputs = self
            .graph
            .run_values(&values, tracking)
            .map_err(to_python_error)?;
        let sources = inputs.iter().map(|input| &**input).collect::<Vec<_>>();
        let outputs = outputs
            .into_iter()
            .zip(self.graph.output_differentiability())
            .map(|(output, differentiable)| {
                let tensor = match output {
                    tynx_core::Value::Tensor(output) if tracking && *differentiable => {
                        PyTensor::from_imported_operation(
                            output,
                            &sources,
                            self.parameters.iter().cloned(),
                        )
                    }
                    output => PyTensor::from_runtime_value(output)?,
                };
                Py::new(py, tensor)
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, outputs).map(Bound::unbind)
    }
}
