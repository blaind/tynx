//! Eager CPython tensor projection over the binding-neutral Rust tensor facade.

use std::{
    cell::RefCell,
    panic::{AssertUnwindSafe, catch_unwind},
    rc::{Rc, Weak},
};

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyList, PyListMethods, PyTuple, PyTupleMethods},
};
use tynx_core::{Device, DynTensor, Gradients, TensorData};

use crate::to_python_error;

/// Eager floating-point tensor.
///
/// Burn-owned tensor state stays in a Rust heap allocation and the initial binding is explicitly
/// unsendable. Operations return new tensors and delegate numerical semantics to `DynTensor`.
#[pyclass(name = "Tensor", frozen, unsendable)]
pub(crate) struct PyTensor {
    inner: Box<DynTensor>,
    leaves: Vec<Weak<LeafState>>,
    leaf: Option<Rc<LeafState>>,
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
            inner: Box::new(inner),
            leaves: Vec::new(),
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
            inner: Box::new(inner),
            leaves: vec![Rc::downgrade(&leaf)],
            leaf: Some(leaf),
        }
    }

    fn from_operation(inner: DynTensor, sources: &[&Self]) -> Self {
        let mut leaves: Vec<Weak<LeafState>> = Vec::new();
        for source in sources {
            for leaf in &source.leaves {
                if !leaves.iter().any(|existing| existing.ptr_eq(leaf)) {
                    leaves.push(leaf.clone());
                }
            }
        }
        Self {
            inner: Box::new(inner),
            leaves,
            leaf: None,
        }
    }

    fn binary(
        &self,
        other: &Self,
        operation: impl FnOnce(DynTensor, DynTensor) -> tynx_core::Result<DynTensor>,
    ) -> PyResult<Self> {
        let inner = operation(self.inner.as_ref().clone(), other.inner.as_ref().clone())
            .map_err(to_python_error)?;
        Ok(Self::from_operation(inner, &[self, other]))
    }
}

#[pymethods]
impl PyTensor {
    /// Construct an f32 tensor from a scalar or rectangular nested list/tuple.
    #[new]
    #[pyo3(signature = (data, *, requires_grad=false))]
    fn new(data: &Bound<'_, PyAny>, requires_grad: bool) -> PyResult<Self> {
        let mut values = Vec::new();
        let mut shape = parse_value(data, &mut values)?;
        if shape.is_empty() {
            shape.push(1);
        }
        let device = Device::autodiff(Device::default());
        let tensor =
            DynTensor::from_data(TensorData::new(values, shape.clone()), shape.len(), &device)
                .map_err(to_python_error)?;
        Ok(if requires_grad {
            Self::from_leaf(tensor)
        } else {
            Self::from_inner(tensor)
        })
    }

    /// Tensor dimensions as a Python tuple.
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.inner.dims())?.unbind())
    }

    /// Number of tensor dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.rank()
    }

    /// Number of tensor elements.
    #[getter]
    fn numel(&self) -> usize {
        self.inner.dims().into_iter().product()
    }

    /// Element dtype. The initial eager constructor is f32-only.
    #[getter]
    fn dtype(&self) -> &'static str {
        "float32"
    }

    /// Whether this tensor participates in an autodiff graph.
    #[getter]
    fn requires_grad(&self) -> bool {
        !self.leaves.is_empty()
    }

    /// Whether this object is a user-created autodiff leaf.
    #[getter]
    fn is_leaf(&self) -> bool {
        self.leaf.is_some()
    }

    /// Return the accumulated gradient for a leaf tensor.
    #[getter]
    fn grad(&self) -> Option<Self> {
        self.leaf
            .as_ref()
            .and_then(|leaf| leaf.grad.borrow().clone())
            .map(Self::from_inner)
    }

    /// Copy tensor values to nested Python lists.
    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let shape = self.inner.dims();
        let values = self
            .inner
            .as_ref()
            .clone()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();
        nested_list(py, &values, &shape)
    }

    /// Copy a one-element tensor to a Python float.
    fn item(&self) -> PyResult<f32> {
        if self.numel() != 1 {
            return Err(PyValueError::new_err(format!(
                "item() requires a one-element tensor, got shape {:?}",
                self.inner.dims()
            )));
        }
        Ok(self
            .inner
            .as_ref()
            .clone()
            .into_data()
            .iter::<f32>()
            .next()
            .expect("one-element tensor data must contain one value"))
    }

    /// Return an off-tape tensor sharing the current numerical value.
    fn detach(&self) -> Self {
        Self::from_inner(self.inner.as_ref().clone().detach())
    }

    /// Clear this leaf tensor's accumulated gradient.
    fn zero_grad(&self) {
        if let Some(leaf) = &self.leaf {
            *leaf.grad.borrow_mut() = None;
        }
    }

    /// Run reverse-mode autodiff from a one-element tensor.
    fn backward(&self) -> PyResult<()> {
        if self.numel() != 1 {
            return Err(PyValueError::new_err(format!(
                "backward() requires a one-element tensor, got shape {:?}",
                self.inner.dims()
            )));
        }
        if !self.requires_grad() {
            return Err(PyValueError::new_err(
                "backward() requires a tensor attached to an autodiff graph",
            ));
        }
        let gradients = catch_unwind(AssertUnwindSafe(|| self.inner.backward())).map_err(|_| {
            PyValueError::new_err("backward() could not traverse the autodiff graph")
        })?;
        for leaf in &self.leaves {
            if let Some(leaf) = leaf.upgrade() {
                leaf.accumulate(&gradients).map_err(to_python_error)?;
            }
        }
        Ok(())
    }

    /// Reduce all dimensions to the v1 one-element `(1,)` tensor shape.
    fn mean(&self) -> PyResult<Self> {
        let dims = (0..self.inner.rank()).collect::<Vec<_>>();
        let inner = self
            .inner
            .as_ref()
            .clone()
            .mean_dims(&dims)
            .reshape(vec![1])
            .map_err(to_python_error)?;
        Ok(Self::from_operation(inner, &[self]))
    }

    fn __add__(&self, other: PyRef<'_, Self>) -> PyResult<Self> {
        self.binary(&other, DynTensor::add_broadcast)
    }

    fn __sub__(&self, other: PyRef<'_, Self>) -> PyResult<Self> {
        self.binary(&other, DynTensor::sub_broadcast)
    }

    fn __mul__(&self, other: PyRef<'_, Self>) -> PyResult<Self> {
        self.binary(&other, DynTensor::mul_broadcast)
    }

    fn __truediv__(&self, other: PyRef<'_, Self>) -> PyResult<Self> {
        self.binary(&other, DynTensor::div_broadcast)
    }

    fn __matmul__(&self, other: PyRef<'_, Self>) -> PyResult<Self> {
        self.binary(&other, DynTensor::matmul)
    }

    fn __neg__(&self) -> Self {
        Self::from_operation(self.inner.as_ref().clone().negated(), &[self])
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype=float32, requires_grad={})",
            self.inner.dims().as_slice(),
            self.requires_grad()
        )
    }
}

fn parse_value(value: &Bound<'_, PyAny>, values: &mut Vec<f32>) -> PyResult<Vec<usize>> {
    if let Ok(number) = value.extract::<f32>() {
        values.push(number);
        return Ok(Vec::new());
    }
    if let Ok(list) = value.cast::<PyList>() {
        return parse_sequence(list.iter(), values);
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return parse_sequence(tuple.iter(), values);
    }
    Err(PyValueError::new_err(
        "Tensor data must be a number or a rectangular nested list/tuple of numbers",
    ))
}

fn parse_sequence<'py>(
    items: impl Iterator<Item = Bound<'py, PyAny>>,
    values: &mut Vec<f32>,
) -> PyResult<Vec<usize>> {
    let mut count = 0;
    let mut child_shape: Option<Vec<usize>> = None;
    for item in items {
        let shape = parse_value(&item, values)?;
        if let Some(expected) = &child_shape {
            if expected != &shape {
                return Err(PyValueError::new_err(format!(
                    "Tensor data is ragged: expected nested shape {expected:?}, got {shape:?}"
                )));
            }
        } else {
            child_shape = Some(shape);
        }
        count += 1;
    }
    let Some(child_shape) = child_shape else {
        return Err(PyValueError::new_err(
            "Tensor data cannot contain an empty list/tuple in v1",
        ));
    };
    let mut shape = Vec::with_capacity(child_shape.len() + 1);
    shape.push(count);
    shape.extend(child_shape);
    Ok(shape)
}

fn nested_list(py: Python<'_>, values: &[f32], shape: &[usize]) -> PyResult<Py<PyAny>> {
    if shape.len() == 1 {
        return Ok(PyList::new(py, values.iter().copied())?.into_any().unbind());
    }
    let stride = shape[1..].iter().product::<usize>();
    let children = values
        .chunks_exact(stride)
        .map(|chunk| nested_list(py, chunk, &shape[1..]))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, children)?.into_any().unbind())
}
