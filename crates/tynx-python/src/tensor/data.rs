//! Python tensor dtype parsing, typed device storage, and host conversion.

use numpy::{PyArrayDyn, PyReadonlyArrayDyn, ndarray::ArrayD};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBool, PyList, PyListMethods, PyRange, PyTuple, PyTupleMethods},
};
use tynx_core::{BoolStore, DType, Device, DynBool, DynInt, DynTensor, TensorData, Value};

use crate::to_python_error;

use super::factory::validate_allocation;

#[derive(Debug, Clone)]
pub(super) enum TensorValue {
    Float(DynTensor),
    Int(DynInt),
    Bool(DynBool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntBounds {
    Empty,
    Range { min: i64, max: i64 },
}

impl IntBounds {
    pub(crate) fn from_values(values: &[i64]) -> Self {
        let Some((&first, remaining)) = values.split_first() else {
            return Self::Empty;
        };
        let (min, max) = remaining.iter().fold((first, first), |(min, max), &value| {
            (min.min(value), max.max(value))
        });
        Self::Range { min, max }
    }
}

impl TensorValue {
    pub(super) fn into_runtime(self) -> Value {
        match self {
            Self::Float(value) => Value::Tensor(value),
            Self::Int(value) => Value::Int(value),
            Self::Bool(value) => Value::Bool(value),
        }
    }

    pub(super) fn from_python(
        data: &Bound<'_, PyAny>,
        dtype: Option<&str>,
        device: &Device,
    ) -> PyResult<(Self, Option<IntBounds>)> {
        if let Some(value) = Self::from_numpy(data, dtype, device)? {
            return Ok(value);
        }
        let dtype = dtype.unwrap_or("float32");
        match dtype {
            "float32" => {
                let (values, shape) = parse(data, "float32", |value| value.extract::<f32>())?;
                validate_allocation(&shape, DType::F32, device)?;
                DynTensor::from_data(TensorData::new(values, shape.clone()), shape.len(), device)
                    .map(Self::Float)
                    .map(|value| (value, None))
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
                let bounds = IntBounds::from_values(&values);
                validate_allocation(&shape, DType::I64, device)?;
                DynInt::from_data(TensorData::new(values, shape.clone()), shape.len(), device)
                    .map(Self::Int)
                    .map(|value| (value, Some(bounds)))
                    .map_err(to_python_error)
            }
            "bool" => {
                let (values, shape) = parse(data, "bool", |value| value.extract::<bool>())?;
                validate_allocation(&shape, DType::Bool(BoolStore::U32), device)?;
                bool_from_data(values, shape, device)
                    .map(Self::Bool)
                    .map(|value| (value, None))
                    .map_err(to_python_error)
            }
            other => Err(PyValueError::new_err(format!(
                "unsupported Tensor dtype {other:?}; expected 'float32', 'int64', or 'bool'"
            ))),
        }
    }

    fn from_numpy(
        data: &Bound<'_, PyAny>,
        dtype: Option<&str>,
        device: &Device,
    ) -> PyResult<Option<(Self, Option<IntBounds>)>> {
        if !data.hasattr("__array_interface__")? {
            return Ok(None);
        }
        data.py().import("numpy")?;

        macro_rules! extract {
            ($element:ty, $dtype:literal, $kind:ident, $constructor:ident) => {
                if let Ok(array) = data.extract::<PyReadonlyArrayDyn<'_, $element>>() {
                    if let Some(requested) = dtype
                        && requested != $dtype
                    {
                        return Err(PyTypeError::new_err(format!(
                            "NumPy array dtype {} must match requested Tensor dtype {requested}",
                            $dtype
                        )));
                    }
                    let array = array.as_array();
                    let mut shape = array.shape().to_vec();
                    if shape.is_empty() {
                        shape.push(1);
                    }
                    if array.is_empty() {
                        return Err(PyValueError::new_err(
                            "Tensor data cannot contain an empty NumPy array",
                        ));
                    }
                    validate_allocation(&shape, DType::F32, device)?;
                    let values = array.iter().copied().collect::<Vec<_>>();
                    return $constructor::from_data(
                        TensorData::new(values, shape.clone()),
                        shape.len(),
                        device,
                    )
                    .map(Self::$kind)
                    .map(|value| (value, None))
                    .map(Some)
                    .map_err(to_python_error);
                }
            };
        }

        extract!(f32, "float32", Float, DynTensor);
        if let Ok(array) = data.extract::<PyReadonlyArrayDyn<'_, f64>>() {
            if let Some(requested) = dtype
                && requested != "float32"
            {
                return Err(PyTypeError::new_err(format!(
                    "NumPy float64 data normalizes to float32 and cannot match requested Tensor dtype {requested}"
                )));
            }
            let array = array.as_array();
            let mut shape = array.shape().to_vec();
            if shape.is_empty() {
                shape.push(1);
            }
            if array.is_empty() {
                return Err(PyValueError::new_err(
                    "Tensor data cannot contain an empty NumPy array",
                ));
            }
            validate_allocation(&shape, DType::F32, device)?;
            let values = array.iter().map(|value| *value as f32).collect::<Vec<_>>();
            return DynTensor::from_data(
                TensorData::new(values, shape.clone()),
                shape.len(),
                device,
            )
            .map(Self::Float)
            .map(|value| (value, None))
            .map(Some)
            .map_err(to_python_error);
        }
        if let Ok(array) = data.extract::<PyReadonlyArrayDyn<'_, i64>>() {
            if let Some(requested) = dtype
                && requested != "int64"
            {
                return Err(PyTypeError::new_err(format!(
                    "NumPy array dtype int64 must match requested Tensor dtype {requested}"
                )));
            }
            let array = array.as_array();
            let mut shape = array.shape().to_vec();
            if shape.is_empty() {
                shape.push(1);
            }
            if array.is_empty() {
                return Err(PyValueError::new_err(
                    "Tensor data cannot contain an empty NumPy array",
                ));
            }
            validate_allocation(&shape, DType::I64, device)?;
            let values = array.iter().copied().collect::<Vec<_>>();
            let bounds = IntBounds::from_values(&values);
            return DynInt::from_data(TensorData::new(values, shape.clone()), shape.len(), device)
                .map(Self::Int)
                .map(|value| (value, Some(bounds)))
                .map(Some)
                .map_err(to_python_error);
        }
        if let Ok(array) = data.extract::<PyReadonlyArrayDyn<'_, i32>>() {
            if let Some(requested) = dtype
                && requested != "int64"
            {
                return Err(PyTypeError::new_err(format!(
                    "NumPy int32 data normalizes to int64 and cannot match requested Tensor dtype {requested}"
                )));
            }
            let array = array.as_array();
            let mut shape = array.shape().to_vec();
            if shape.is_empty() {
                shape.push(1);
            }
            if array.is_empty() {
                return Err(PyValueError::new_err(
                    "Tensor data cannot contain an empty NumPy array",
                ));
            }
            validate_allocation(&shape, DType::I64, device)?;
            let values = array
                .iter()
                .map(|value| i64::from(*value))
                .collect::<Vec<_>>();
            let bounds = IntBounds::from_values(&values);
            return DynInt::from_data(TensorData::new(values, shape.clone()), shape.len(), device)
                .map(Self::Int)
                .map(|value| (value, Some(bounds)))
                .map(Some)
                .map_err(to_python_error);
        }
        if let Ok(array) = data.extract::<PyReadonlyArrayDyn<'_, bool>>() {
            if let Some(requested) = dtype
                && requested != "bool"
            {
                return Err(PyTypeError::new_err(format!(
                    "NumPy array dtype bool must match requested Tensor dtype {requested}"
                )));
            }
            let array = array.as_array();
            let mut shape = array.shape().to_vec();
            if shape.is_empty() {
                shape.push(1);
            }
            if array.is_empty() {
                return Err(PyValueError::new_err(
                    "Tensor data cannot contain an empty NumPy array",
                ));
            }
            validate_allocation(&shape, DType::Bool(BoolStore::U32), device)?;
            let values = array.iter().copied().collect::<Vec<_>>();
            return bool_from_data(values, shape, device)
                .map(Self::Bool)
                .map(|value| (value, None))
                .map(Some)
                .map_err(to_python_error);
        }
        match dtype {
            Some(dtype) => Err(PyTypeError::new_err(format!(
                "NumPy array dtype must match requested Tensor dtype {dtype}"
            ))),
            None => Err(PyTypeError::new_err(
                "unsupported NumPy dtype; expected float32, float64, int32, int64, or bool",
            )),
        }
    }

    pub(super) fn float_from_python(
        data: &Bound<'_, PyAny>,
        device: &Device,
    ) -> PyResult<DynTensor> {
        match Self::from_python(data, Some("float32"), device)?.0 {
            Self::Float(value) => Ok(value),
            _ => unreachable!("the float32 parser always creates a floating-point tensor"),
        }
    }

    pub(super) fn float(self, operation: &str) -> PyResult<DynTensor> {
        match self {
            Self::Float(value) => Ok(value),
            other => Err(PyTypeError::new_err(format!(
                "{operation} requires a float32 Tensor, got {}; convert it with .cast(\"float32\")",
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

    pub(super) fn cast(self, dtype: &str) -> PyResult<Self> {
        match (self, dtype) {
            (Self::Float(value), "float32") => Ok(Self::Float(value.cast(tynx_core::DType::F32))),
            (Self::Float(value), "int64") => Ok(Self::Int(value.to_int(tynx_core::DType::I64))),
            (Self::Float(value), "bool") => Ok(Self::Bool(value.to_bool())),
            (Self::Int(value), "float32") => Ok(Self::Float(value.to_float(tynx_core::DType::F32))),
            (Self::Int(value), "int64") => Ok(Self::Int(value.cast(tynx_core::DType::I64))),
            (Self::Int(value), "bool") => Ok(Self::Bool(value.to_bool())),
            (Self::Bool(value), "float32") => {
                Ok(Self::Float(value.to_float(tynx_core::DType::F32)))
            }
            (Self::Bool(value), "int64") => Ok(Self::Int(value.to_int(tynx_core::DType::I64))),
            (Self::Bool(value), "bool") => Ok(Self::Bool(value)),
            (_, other) => Err(PyValueError::new_err(format!(
                "unsupported Tensor dtype {other:?}; expected 'float32', 'int64', or 'bool'"
            ))),
        }
    }

    pub(super) fn move_to_device(self, device: &Device) -> Self {
        match self {
            Self::Float(value) => Self::Float(value.to_device(device)),
            Self::Int(value) => Self::Int(value.to_device(device)),
            Self::Bool(value) => Self::Bool(value.to_device(device)),
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

    pub(super) fn device(&self) -> Device {
        match self {
            Self::Float(value) => value.device(),
            Self::Int(value) => value.device(),
            Self::Bool(value) => value.device(),
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

    pub(super) fn numpy(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        py.import("numpy")?;
        let shape = self.dims();
        macro_rules! convert {
            ($value:expr, $element:ty) => {{
                let values = $value.into_data().iter::<$element>().collect::<Vec<_>>();
                let array = ArrayD::from_shape_vec(shape, values)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
                Ok(PyArrayDyn::<$element>::from_owned_array(py, array)
                    .into_any()
                    .unbind())
            }};
        }
        match self {
            Self::Float(value) => convert!(value, f32),
            Self::Int(value) => convert!(value, i64),
            Self::Bool(value) => convert!(value, bool),
        }
    }
}

fn bool_from_data(
    values: Vec<bool>,
    shape: Vec<usize>,
    device: &Device,
) -> tynx_core::Result<DynBool> {
    let values = values.into_iter().map(i64::from).collect::<Vec<_>>();
    let rank = shape.len();
    DynInt::from_data(TensorData::new(values, shape), rank, device)
        .map(|tensor| tensor.equal_scalar(1))
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
    if shape[0] == 0 {
        return Ok(PyList::empty(py).into_any().unbind());
    }
    if shape.len() == 1 {
        return Ok(PyList::new(py, values.iter().copied())?.into_any().unbind());
    }
    let stride = shape[1..].iter().product::<usize>();
    if stride == 0 {
        let children = (0..shape[0])
            .map(|_| nested_list(py, &values[0..0], &shape[1..]))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(PyList::new(py, children)?.into_any().unbind());
    }
    let children = values
        .chunks_exact(stride)
        .map(|chunk| nested_list(py, chunk, &shape[1..]))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, children)?.into_any().unbind())
}
