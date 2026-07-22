//! Eager CPython tensor projection over the binding-neutral Rust tensor facade.

mod data;
mod reduction;
mod shape;

use std::{
    cell::RefCell,
    panic::{AssertUnwindSafe, catch_unwind},
    rc::{Rc, Weak},
};

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyTuple},
};
use tynx_core::{Device, DynTensor, Gradients};
use tynx_train::ParameterSlot;

use crate::{grad_mode::is_grad_enabled, to_python_error};
use data::TensorValue;
use reduction::ReductionSpec;

/// Eager device tensor with optional floating-point autodiff state.
///
/// Burn-owned tensor state stays in a Rust heap allocation and the initial binding is explicitly
/// unsendable. Operations return new tensors and delegate numerical semantics to `DynTensor`.
#[pyclass(name = "Tensor", frozen, unsendable, subclass)]
pub(crate) struct PyTensor {
    source: TensorSource,
    targets: Vec<GradTarget>,
    leaf: Option<Rc<LeafState>>,
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
    Parameter(ParameterSlot),
}

impl GradTarget {
    fn same_identity(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Tensor(left), Self::Tensor(right)) => left.ptr_eq(right),
            (Self::Parameter(left), Self::Parameter(right)) => left.id() == right.id(),
            _ => false,
        }
    }

    fn accumulate(&self, gradients: &Gradients) -> tynx_core::Result<()> {
        match self {
            Self::Tensor(leaf) => {
                if let Some(leaf) = leaf.upgrade() {
                    leaf.accumulate(gradients)?;
                }
            }
            Self::Parameter(slot) => {
                slot.accumulate_grad(gradients)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct LeafState {
    tensor: DynTensor,
    grad: RefCell<Option<DynTensor>>,
}

impl LeafState {
    fn accumulate(&self, gradients: &Gradients) -> tynx_core::Result<()> {
        let Some(gradient) = self.tensor.grad(gradients) else {
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
    fn from_inner(inner: DynTensor) -> Self {
        Self {
            source: TensorSource::Owned(Box::new(TensorValue::Float(inner))),
            targets: Vec::new(),
            leaf: None,
        }
    }

    fn from_value(value: TensorValue) -> Self {
        Self {
            source: TensorSource::Owned(Box::new(value)),
            targets: Vec::new(),
            leaf: None,
        }
    }

    fn from_leaf(inner: DynTensor) -> Self {
        let inner = inner.require_grad();
        let leaf = Rc::new(LeafState {
            tensor: inner.clone(),
            grad: RefCell::new(None),
        });
        Self {
            source: TensorSource::Owned(Box::new(TensorValue::Float(inner))),
            targets: vec![GradTarget::Tensor(Rc::downgrade(&leaf))],
            leaf: Some(leaf),
        }
    }

    pub(crate) fn from_parameter(slot: ParameterSlot) -> Self {
        Self {
            source: TensorSource::Parameter(slot.clone()),
            targets: vec![GradTarget::Parameter(slot)],
            leaf: None,
        }
    }

    fn from_operation(inner: DynTensor, sources: &[&Self]) -> Self {
        let mut targets: Vec<GradTarget> = Vec::new();
        for source in sources {
            for target in &source.targets {
                if !targets
                    .iter()
                    .any(|existing| existing.same_identity(target))
                {
                    targets.push(target.clone());
                }
            }
        }
        Self {
            source: TensorSource::Owned(Box::new(TensorValue::Float(inner))),
            targets,
            leaf: None,
        }
    }

    fn binary(
        &self,
        other: &Self,
        operation: impl FnOnce(DynTensor, DynTensor) -> tynx_core::Result<DynTensor>,
    ) -> PyResult<Self> {
        let tracking = is_grad_enabled();
        let left = self.source.operation_input(tracking, "arithmetic")?;
        let right = other.source.operation_input(tracking, "arithmetic")?;
        let inner = operation(left, right).map_err(to_python_error)?;
        Ok(if tracking {
            Self::from_operation(inner, &[self, other])
        } else {
            Self::from_inner(inner)
        })
    }

    fn arithmetic(
        &self,
        other: &Bound<'_, PyAny>,
        tensor_operation: impl FnOnce(DynTensor, DynTensor) -> tynx_core::Result<DynTensor>,
        scalar_operation: impl FnOnce(DynTensor, f64) -> DynTensor,
    ) -> PyResult<Self> {
        if let Ok(other) = other.extract::<PyRef<'_, Self>>() {
            return self.binary(&other, tensor_operation);
        }
        let scalar = extract_scalar_operand(other)?;
        self.unary(|input| Ok(scalar_operation(input, scalar)))
    }

    fn unary(
        &self,
        operation: impl FnOnce(DynTensor) -> tynx_core::Result<DynTensor>,
    ) -> PyResult<Self> {
        let tracking = is_grad_enabled();
        let input = self.source.operation_input(tracking, "operation")?;
        let inner = operation(input).map_err(to_python_error)?;
        Ok(if tracking {
            Self::from_operation(inner, &[self])
        } else {
            Self::from_inner(inner)
        })
    }

    pub(crate) fn tensor_from_python(data: &Bound<'_, PyAny>) -> PyResult<DynTensor> {
        let device = Device::autodiff(Device::default());
        TensorValue::float_from_python(data, &device)
    }

    pub(crate) fn parameter_name(&self) -> Option<String> {
        match &self.source {
            TensorSource::Parameter(slot) => slot.name(),
            TensorSource::Owned(_) => None,
        }
    }
}

#[pymethods]
impl PyTensor {
    /// Construct a typed tensor from a scalar or rectangular nested list/tuple.
    #[new]
    #[pyo3(signature = (data, *, dtype="float32", requires_grad=false))]
    fn new(data: &Bound<'_, PyAny>, dtype: &str, requires_grad: bool) -> PyResult<Self> {
        let device = Device::autodiff(Device::default());
        let value = TensorValue::from_python(data, dtype, &device)?;
        if requires_grad {
            return Ok(Self::from_leaf(value.float("requires_grad=True")?));
        }
        Ok(Self::from_value(value))
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

    /// Element dtype.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.source.value().dtype_name()
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
        self.source.value().tolist(py)
    }

    /// Copy a one-element tensor to a Python scalar.
    fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.numel() != 1 {
            return Err(PyValueError::new_err(format!(
                "item() requires a one-element tensor, got shape {:?}",
                self.source.value().dims()
            )));
        }
        self.source.value().item(py)
    }

    /// Return an off-tape tensor sharing the current numerical value.
    fn detach(&self) -> Self {
        Self::from_value(self.source.value().detach())
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
        let output = self.source.operation_input(true, "backward")?;
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
        let gradients = catch_unwind(AssertUnwindSafe(|| root.backward())).map_err(|_| {
            PyValueError::new_err("backward() could not traverse the autodiff graph")
        })?;
        for target in &self.targets {
            target.accumulate(&gradients).map_err(to_python_error)?;
        }
        Ok(())
    }

    /// Sum values over all, one, or several dimensions.
    #[pyo3(signature = (dim=None, keepdim=false))]
    fn sum(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool) -> PyResult<Self> {
        self.reduce(dim, keepdim, DynTensor::sum_dims)
    }

    /// Average values over all, one, or several dimensions.
    #[pyo3(signature = (dim=None, keepdim=false))]
    fn mean(&self, dim: Option<&Bound<'_, PyAny>>, keepdim: bool) -> PyResult<Self> {
        self.reduce(dim, keepdim, DynTensor::mean_dims)
    }

    /// Apply rectified linear activation element-wise.
    fn relu(&self) -> PyResult<Self> {
        self.unary(|input| Ok(input.relu()))
    }

    /// Apply logistic sigmoid activation element-wise.
    fn sigmoid(&self) -> PyResult<Self> {
        self.unary(|input| Ok(input.sigmoid()))
    }

    /// Apply hyperbolic tangent element-wise.
    fn tanh(&self) -> PyResult<Self> {
        self.unary(|input| Ok(input.tanh()))
    }

    /// Apply the exponential function element-wise.
    fn exp(&self) -> PyResult<Self> {
        self.unary(|input| Ok(input.exp()))
    }

    /// Apply the natural logarithm element-wise.
    fn log(&self) -> PyResult<Self> {
        self.unary(|input| Ok(input.log()))
    }

    /// Apply the square root element-wise.
    fn sqrt(&self) -> PyResult<Self> {
        self.unary(|input| Ok(input.sqrt()))
    }

    /// Apply Gaussian error linear unit activation element-wise.
    fn gelu(&self) -> PyResult<Self> {
        self.unary(|input| Ok(input.gelu()))
    }

    /// Normalize values into probabilities along one dimension.
    fn softmax(&self, dim: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dim = shape::axis(dim, self.ndim(), false, "softmax")?;
        self.unary(|input| Ok(input.softmax(dim)))
    }

    /// Apply numerically stable log-softmax along one dimension.
    fn log_softmax(&self, dim: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dim = shape::axis(dim, self.ndim(), false, "log_softmax")?;
        self.unary(|input| Ok(input.log_softmax(dim)))
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
        self.unary(move |input| input.reshape(output))
    }

    /// Flatten a contiguous range of dimensions.
    #[pyo3(signature = (start_dim=0, end_dim=-1))]
    fn flatten(&self, start_dim: isize, end_dim: isize) -> PyResult<Self> {
        let input_shape = self.source.value().dims();
        let start = shape::axis_value(start_dim, input_shape.len(), false, "flatten start_dim")?;
        let end = shape::axis_value(end_dim, input_shape.len(), false, "flatten end_dim")?;
        let output = shape::flatten(&input_shape, start, end)?;
        self.unary(move |input| input.reshape(output))
    }

    /// Swap two tensor dimensions.
    fn transpose(&self, dim0: &Bound<'_, PyAny>, dim1: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rank = self.ndim();
        let dim0 = shape::axis(dim0, rank, false, "transpose")?;
        let dim1 = shape::axis(dim1, rank, false, "transpose")?;
        let mut axes = (0..rank).collect::<Vec<_>>();
        axes.swap(dim0, dim1);
        self.unary(move |input| input.permute(axes))
    }

    /// Reorder all tensor dimensions.
    #[pyo3(signature = (*dims))]
    fn permute(&self, dims: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let axes = shape::permutation(dims, self.ndim())?;
        self.unary(move |input| input.permute(axes))
    }

    /// Remove singleton dimensions, or one selected singleton dimension.
    #[pyo3(signature = (dim=None))]
    fn squeeze(&self, dim: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let input_shape = self.source.value().dims();
        let dim = dim
            .map(|dim| shape::axis(dim, input_shape.len(), false, "squeeze"))
            .transpose()?;
        let output = shape::squeeze(&input_shape, dim);
        self.unary(move |input| input.reshape(output))
    }

    /// Insert a singleton dimension.
    fn unsqueeze(&self, dim: &Bound<'_, PyAny>) -> PyResult<Self> {
        let input_shape = self.source.value().dims();
        let dim = shape::axis(dim, input_shape.len(), true, "unsqueeze")?;
        let output = shape::unsqueeze(&input_shape, dim)?;
        self.unary(move |input| input.reshape(output))
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(other, DynTensor::add_broadcast, DynTensor::add_scalar)
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(other, DynTensor::add_broadcast, DynTensor::add_scalar)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(other, DynTensor::sub_broadcast, DynTensor::sub_scalar)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let scalar = extract_scalar_operand(other)?;
        self.unary(|input| Ok(input.negated().add_scalar(scalar)))
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(other, DynTensor::mul_broadcast, DynTensor::mul_scalar)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(other, DynTensor::mul_broadcast, DynTensor::mul_scalar)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.arithmetic(other, DynTensor::div_broadcast, DynTensor::div_scalar)
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let scalar = extract_scalar_operand(other)?;
        self.unary(|input| Ok(input.reciprocal().mul_scalar(scalar)))
    }

    fn __matmul__(&self, other: PyRef<'_, Self>) -> PyResult<Self> {
        self.binary(&other, DynTensor::matmul)
    }

    fn __neg__(&self) -> PyResult<Self> {
        self.unary(|input| Ok(input.negated()))
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
    fn clip_bounds(&self, min: Option<f64>, max: Option<f64>) -> PyResult<Self> {
        if min.is_none() && max.is_none() {
            return Err(PyValueError::new_err(
                "clamp requires at least one of min or max",
            ));
        }
        self.unary(|input| Ok(input.clip(min, max)))
    }

    fn reduce(
        &self,
        dim: Option<&Bound<'_, PyAny>>,
        keepdim: bool,
        operation: impl FnOnce(DynTensor, &[usize]) -> DynTensor,
    ) -> PyResult<Self> {
        let input_shape = self.source.value().dims();
        let spec = ReductionSpec::from_python(dim, &input_shape, keepdim)?;
        self.unary(move |input| operation(input, &spec.dims).reshape(spec.output_shape))
    }
}

fn extract_scalar_operand(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    value.extract::<f64>().map_err(|_| {
        PyTypeError::new_err("Tensor arithmetic expects another Tensor or a real number")
    })
}
