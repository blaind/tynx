//! Python tensor dtype parsing, typed device storage, and host conversion.

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBool, PyList, PyListMethods, PyRange, PyTuple, PyTupleMethods},
};
use tynx_core::{Device, DynBool, DynInt, DynTensor, TensorData};

use crate::to_python_error;

#[derive(Debug, Clone)]
pub(super) enum TensorValue {
    Float(DynTensor),
    Int(DynInt),
    Bool(DynBool),
}

impl TensorValue {
    pub(super) fn from_python(
        data: &Bound<'_, PyAny>,
        dtype: &str,
        device: &Device,
    ) -> PyResult<Self> {
        match dtype {
            "float32" => {
                let (values, shape) = parse(data, "float32", |value| value.extract::<f32>())?;
                DynTensor::from_data(TensorData::new(values, shape.clone()), shape.len(), device)
                    .map(Self::Float)
                    .map_err(to_python_error)
            }
            "int64" => {
                let (values, shape) = parse(data, "int64", |value| {
                    if value.is_instance_of::<PyBool>() {
                        return Err(PyTypeError::new_err(
                            "int64 Tensor data must contain integers, not booleans",
                        ));
                    }
                    value.extract::<i64>()
                })?;
                DynInt::from_data(TensorData::new(values, shape.clone()), shape.len(), device)
                    .map(Self::Int)
                    .map_err(to_python_error)
            }
            "bool" => {
                let (values, shape) = parse(data, "bool", |value| value.extract::<bool>())?;
                DynBool::from_data(TensorData::new(values, shape.clone()), shape.len(), device)
                    .map(Self::Bool)
                    .map_err(to_python_error)
            }
            other => Err(PyValueError::new_err(format!(
                "unsupported Tensor dtype {other:?}; expected 'float32', 'int64', or 'bool'"
            ))),
        }
    }

    pub(super) fn float_from_python(
        data: &Bound<'_, PyAny>,
        device: &Device,
    ) -> PyResult<DynTensor> {
        match Self::from_python(data, "float32", device)? {
            Self::Float(value) => Ok(value),
            _ => unreachable!("the float32 parser always creates a floating-point tensor"),
        }
    }

    pub(super) fn float(self, operation: &str) -> PyResult<DynTensor> {
        match self {
            Self::Float(value) => Ok(value),
            other => Err(PyTypeError::new_err(format!(
                "{operation} requires a float32 Tensor, got {}",
                other.dtype_name()
            ))),
        }
    }

    pub(super) fn scalar_like(self, scalar: &Bound<'_, PyAny>, context: &str) -> PyResult<Self> {
        match self {
            Self::Float(value) => scalar
                .extract::<f64>()
                .map(|scalar| Self::Float(value.full_like(scalar)))
                .map_err(|_| {
                    PyTypeError::new_err(format!("float32 {context} expects a real scalar"))
                }),
            Self::Int(value) => {
                if scalar.is_instance_of::<PyBool>() {
                    return Err(PyTypeError::new_err(format!(
                        "int64 {context} expects an integer scalar, not bool"
                    )));
                }
                scalar
                    .extract::<i64>()
                    .map(|scalar| Self::Int(value.full_like(scalar)))
                    .map_err(|_| {
                        PyTypeError::new_err(format!("int64 {context} expects an integer scalar"))
                    })
            }
            Self::Bool(value) => scalar
                .extract::<bool>()
                .map(|scalar| Self::Bool(value.full_like(scalar)))
                .map_err(|_| PyTypeError::new_err(format!("bool {context} expects a bool scalar"))),
        }
    }

    pub(super) fn detach(self) -> Self {
        match self {
            Self::Float(value) => Self::Float(value.detach()),
            other => other,
        }
    }

    pub(super) fn reshape(self, dims: Vec<usize>) -> PyResult<Self> {
        match self {
            Self::Float(value) => value
                .reshape(dims)
                .map(Self::Float)
                .map_err(to_python_error),
            Self::Int(value) => value.reshape(dims).map(Self::Int).map_err(to_python_error),
            Self::Bool(value) => value.reshape(dims).map(Self::Bool).map_err(to_python_error),
        }
    }

    pub(super) fn dims(&self) -> Vec<usize> {
        match self {
            Self::Float(value) => value.dims(),
            Self::Int(value) => value.dims(),
            Self::Bool(value) => value.dims(),
        }
    }

    pub(super) fn rank(&self) -> usize {
        match self {
            Self::Float(value) => value.rank(),
            Self::Int(value) => value.rank(),
            Self::Bool(value) => value.rank(),
        }
    }

    pub(super) fn dtype_name(&self) -> &'static str {
        match self {
            Self::Float(_) => "float32",
            Self::Int(_) => "int64",
            Self::Bool(_) => "bool",
        }
    }

    pub(super) fn tolist(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let shape = self.dims();
        match self {
            Self::Float(value) => {
                let values = value.into_data().iter::<f32>().collect::<Vec<_>>();
                nested_list(py, &values, &shape)
            }
            Self::Int(value) => {
                let values = value.into_data().iter::<i64>().collect::<Vec<_>>();
                nested_list(py, &values, &shape)
            }
            Self::Bool(value) => {
                let values = value.into_data().iter::<bool>().collect::<Vec<_>>();
                nested_list(py, &values, &shape)
            }
        }
    }

    pub(super) fn item(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Float(value) => Ok(value
                .into_data()
                .iter::<f32>()
                .next()
                .expect("one-element tensor data must contain one value")
                .into_pyobject(py)
                .expect("f32 conversion is infallible")
                .into_any()
                .unbind()),
            Self::Int(value) => Ok(value
                .into_data()
                .iter::<i64>()
                .next()
                .expect("one-element tensor data must contain one value")
                .into_pyobject(py)
                .expect("i64 conversion is infallible")
                .into_any()
                .unbind()),
            Self::Bool(value) => Ok(value
                .into_data()
                .iter::<bool>()
                .next()
                .expect("one-element tensor data must contain one value")
                .into_pyobject(py)
                .expect("bool conversion is infallible")
                .to_owned()
                .into_any()
                .unbind()),
        }
    }
}

fn parse<T>(
    value: &Bound<'_, PyAny>,
    dtype: &str,
    extract: impl Copy + Fn(&Bound<'_, PyAny>) -> PyResult<T>,
) -> PyResult<(Vec<T>, Vec<usize>)> {
    let mut values = Vec::new();
    let mut shape = parse_value(value, &mut values, dtype, extract)?;
    if shape.is_empty() {
        shape.push(1);
    }
    Ok((values, shape))
}

fn parse_value<T>(
    value: &Bound<'_, PyAny>,
    values: &mut Vec<T>,
    dtype: &str,
    extract: impl Copy + Fn(&Bound<'_, PyAny>) -> PyResult<T>,
) -> PyResult<Vec<usize>> {
    if !value.is_instance_of::<PyList>()
        && !value.is_instance_of::<PyTuple>()
        && !value.is_instance_of::<PyRange>()
    {
        values.push(extract(value).map_err(|_| {
            PyTypeError::new_err(format!(
                "{dtype} Tensor data must contain compatible scalar values"
            ))
        })?);
        return Ok(Vec::new());
    }
    if let Ok(list) = value.cast::<PyList>() {
        return parse_sequence(list.iter(), values, dtype, extract);
    }
    if let Ok(range) = value.cast::<PyRange>() {
        let items = range.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        return parse_sequence(items.into_iter(), values, dtype, extract);
    }
    let tuple = value.cast::<PyTuple>()?;
    parse_sequence(tuple.iter(), values, dtype, extract)
}

fn parse_sequence<'py, T>(
    items: impl Iterator<Item = Bound<'py, PyAny>>,
    values: &mut Vec<T>,
    dtype: &str,
    extract: impl Copy + Fn(&Bound<'_, PyAny>) -> PyResult<T>,
) -> PyResult<Vec<usize>> {
    let mut count = 0;
    let mut child_shape: Option<Vec<usize>> = None;
    for item in items {
        let shape = parse_value(&item, values, dtype, extract)?;
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

fn nested_list<'py, T>(py: Python<'py>, values: &[T], shape: &[usize]) -> PyResult<Py<PyAny>>
where
    T: Copy + IntoPyObject<'py>,
    <T as IntoPyObject<'py>>::Error: Into<PyErr>,
{
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
