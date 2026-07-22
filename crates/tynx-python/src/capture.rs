//! CPython tracing projection over the binding-neutral capture runtime.

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyTuple};
use tynx_capture::{BinaryOp, Graph, GraphBuilder, UnaryOp, ValueId};
use tynx_train::{ParamId, ParameterSlot};

use crate::{grad_mode::is_grad_enabled, tensor::PyTensor, to_python_error};

#[derive(Debug)]
struct CaptureInner {
    builder: Option<GraphBuilder>,
    parameters: HashMap<ParamId, ValueId>,
    failure: Option<String>,
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
                failure: None,
            }),
        })
    }

    fn input(&self, tensor: &PyTensor) -> PyResult<ValueId> {
        let value = tensor.detached_float_value("compile input")?;
        self.with_builder(|builder| Ok(builder.input(&value)))
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
        self.with_builder(|builder| builder.unary(op, input).map_err(to_python_error))
    }

    fn binary(&self, op: BinaryOp, left: ValueId, right: ValueId) -> PyResult<ValueId> {
        self.with_builder(|builder| builder.binary(op, left, right).map_err(to_python_error))
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

    fn finish(&self, output: ValueId) -> PyResult<Option<Graph>> {
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
        builder
            .finish(vec![output])
            .map(Some)
            .map_err(to_python_error)
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

    pub(crate) fn binary(&self, other: &PyTensor, op: BinaryOp) -> PyResult<Option<Self>> {
        let right = if let Some(other_trace) = other.trace() {
            if !Rc::ptr_eq(&self.state, &other_trace.state) {
                self.state
                    .unsupported("tensors from different capture sessions")?;
                return Ok(None);
            }
            other_trace.value
        } else if let Some(slot) = other.parameter_slot() {
            self.state.parameter(slot)?
        } else {
            self.state.unsupported(
                "a closed-over Tensor value; pass changing tensors as function inputs",
            )?;
            return Ok(None);
        };
        self.state.binary(op, self.value, right).map(|value| {
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

pub(crate) fn record_binary(
    left: &PyTensor,
    right: &PyTensor,
    op: BinaryOp,
) -> PyResult<Option<TraceValue>> {
    if let Some(left_trace) = left.trace() {
        return left_trace.binary(right, op);
    }
    let Some(right_trace) = right.trace() else {
        return Ok(None);
    };
    let Some(slot) = left.parameter_slot() else {
        right_trace
            .state
            .unsupported("a closed-over Tensor value; pass changing tensors as function inputs")?;
        return Ok(None);
    };
    let left = right_trace.state.parameter(slot)?;
    right_trace
        .state
        .binary(op, left, right_trace.value)
        .map(|value| {
            Some(TraceValue {
                state: right_trace.state.clone(),
                value,
            })
        })
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
        let value = self.state.input(&tensor)?;
        Ok(tensor.with_trace(TraceValue {
            state: self.state.clone(),
            value,
        }))
    }

    fn finish(&self, output: PyRef<'_, PyTensor>) -> PyResult<Option<PyCapturedGraph>> {
        let Some(trace) = output.trace() else {
            self.state
                .unsupported("a function whose output is not connected to its Tensor inputs")?;
            return Ok(None);
        };
        if !Rc::ptr_eq(&self.state, &trace.state) {
            self.state
                .unsupported("an output from another capture session")?;
            return Ok(None);
        }
        self.state
            .finish(trace.value)
            .map(|graph| graph.map(PyCapturedGraph::new))
    }

    fn release(&self, output: PyRef<'_, PyTensor>) -> PyTensor {
        output.without_trace()
    }
}

/// Immutable native graph replay object.
#[pyclass(name = "_CapturedGraph", frozen, unsendable)]
pub(crate) struct PyCapturedGraph {
    graph: Graph,
    parameters: Vec<ParameterSlot>,
}

impl PyCapturedGraph {
    fn new(graph: Graph) -> Self {
        let parameters = graph.parameters();
        Self { graph, parameters }
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

    #[pyo3(signature = (*inputs))]
    fn matches(&self, inputs: &Bound<'_, PyTuple>) -> PyResult<bool> {
        let inputs = Self::inputs(inputs, "captured graph matching")?;
        let values = inputs
            .iter()
            .map(|input| input.detached_float_value("captured graph matching"))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(self.graph.validate_inputs(&values).is_ok())
    }

    #[pyo3(signature = (*inputs))]
    fn __call__(&self, inputs: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
        let inputs = Self::inputs(inputs, "captured graph replay")?;
        let tracking = is_grad_enabled();
        let values = inputs
            .iter()
            .map(|input| input.operation_float_value(tracking, "captured graph replay"))
            .collect::<PyResult<Vec<_>>>()?;
        let mut outputs = self.graph.run(&values, tracking).map_err(to_python_error)?;
        let output = outputs
            .pop()
            .expect("capture graphs are constructed with one output");
        if tracking {
            let sources = inputs.iter().map(|input| &**input).collect::<Vec<_>>();
            Ok(PyTensor::from_imported_operation(
                output,
                &sources,
                self.parameters.iter().cloned(),
            ))
        } else {
            Ok(PyTensor::from_inner(output))
        }
    }
}
