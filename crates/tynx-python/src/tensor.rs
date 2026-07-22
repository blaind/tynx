//! Eager CPython tensor projection over the binding-neutral Rust tensor facade.

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyList, PyListMethods, PyTuple, PyTupleMethods},
};
use tynx_core::{Device, DynTensor, TensorData};

use crate::to_python_error;

/// Eager floating-point tensor.
///
/// Burn-owned tensor state stays in a Rust heap allocation and the initial binding is explicitly
/// unsendable. Operations return new tensors and delegate numerical semantics to `DynTensor`.
#[pyclass(name = "Tensor", frozen, unsendable)]
pub(crate) struct PyTensor {
    inner: Box<DynTensor>,
}

impl PyTensor {
    fn from_inner(inner: DynTensor) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }

    fn binary(
        &self,
        other: &Self,
        operation: impl FnOnce(DynTensor, DynTensor) -> tynx_core::Result<DynTensor>,
    ) -> PyResult<Self> {
        operation(self.inner.as_ref().clone(), other.inner.as_ref().clone())
            .map(Self::from_inner)
            .map_err(to_python_error)
    }
}

#[pymethods]
impl PyTensor {
    /// Construct an f32 tensor from a scalar or rectangular nested list/tuple.
    #[new]
    fn new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut values = Vec::new();
        let mut shape = parse_value(data, &mut values)?;
        if shape.is_empty() {
            shape.push(1);
        }
        let device = Device::autodiff(Device::default());
        let tensor =
            DynTensor::from_data(TensorData::new(values, shape.clone()), shape.len(), &device)
                .map_err(to_python_error)?;
        Ok(Self::from_inner(tensor))
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
        Self::from_inner(self.inner.as_ref().clone().negated())
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype=float32)",
            self.inner.dims().as_slice()
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
