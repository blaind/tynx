//! Native tensor factories exposed through the CPython projection.

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBool, PyTuple},
};
use tynx_core::{DType, Device, Distribution, DynBool, DynInt, DynTensor, MAX_RANK};

use super::{
    PyTensor,
    data::{IntBounds, TensorValue},
};
use crate::{
    device::{PyDevice, ensure_autodiff},
    to_python_error,
};

fn validate_shape(shape: Vec<usize>) -> PyResult<Vec<usize>> {
    if shape.is_empty() {
        return Err(PyValueError::new_err(
            "factory shape must contain at least one dimension because rank-zero tensors are not supported",
        ));
    }
    if shape.len() > MAX_RANK {
        return Err(PyValueError::new_err(format!(
            "factory rank {} exceeds the maximum rank {MAX_RANK}",
            shape.len()
        )));
    }
    Ok(shape)
}

fn shape_value(value: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(dimension) = value.extract::<usize>() {
        return Ok(vec![dimension]);
    }
    value.extract::<Vec<usize>>().map_err(|_| {
        PyTypeError::new_err(
            "factory shape must be an integer, a sequence of integers, or positional integers",
        )
    })
}

fn shape_args(values: &Bound<'_, PyTuple>) -> PyResult<Vec<usize>> {
    if values.len() == 1 {
        return shape_value(&values.get_item(0)?);
    }
    values.extract::<Vec<usize>>().map_err(|_| {
        PyTypeError::new_err(
            "factory shape must be an integer, a sequence of integers, or positional integers",
        )
    })
}

fn select_device(device: Option<PyRef<'_, PyDevice>>) -> Device {
    let device = device
        .map(|device| device.inner.as_ref().clone())
        .unwrap_or_else(tynx_core::default_device);
    ensure_autodiff(device)
}

fn finish(value: TensorValue, requires_grad: bool) -> PyResult<PyTensor> {
    if requires_grad {
        return value.float("requires_grad=True").map(PyTensor::from_leaf);
    }
    Ok(PyTensor::from_value(value))
}

fn empty_value(shape: &[usize], dtype: &str, device: &Device) -> PyResult<TensorValue> {
    match dtype {
        "float32" => DynTensor::empty(shape, device, DType::F32).map(TensorValue::Float),
        "int64" => DynInt::empty(shape, device, DType::I64).map(TensorValue::Int),
        "bool" => DynBool::empty(shape, device).map(TensorValue::Bool),
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported Tensor dtype {other:?}; expected 'float32', 'int64', or 'bool'"
            )));
        }
    }
    .map_err(to_python_error)
}

fn full_value(
    shape: &[usize],
    fill_value: &Bound<'_, PyAny>,
    dtype: &str,
    device: &Device,
) -> PyResult<TensorValue> {
    match dtype {
        "float32" => fill_value
            .extract::<f64>()
            .map_err(|_| PyTypeError::new_err("float32 full fill_value must be a real number"))
            .and_then(|value| {
                DynTensor::full(shape, value, device, DType::F32)
                    .map(TensorValue::Float)
                    .map_err(to_python_error)
            }),
        "int64" => {
            if fill_value.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(
                    "int64 full fill_value must be an integer, not bool",
                ));
            }
            fill_value
                .extract::<i64>()
                .map_err(|_| PyTypeError::new_err("int64 full fill_value must be an integer"))
                .and_then(|value| {
                    DynInt::full(shape, value, device, DType::I64)
                        .map(TensorValue::Int)
                        .map_err(to_python_error)
                })
        }
        "bool" => {
            let value = fill_value.extract::<bool>().or_else(|_| {
                fill_value
                    .extract::<f64>()
                    .map(|value| value != 0.0)
                    .map_err(|_| {
                        PyTypeError::new_err("bool full fill_value must be a bool or real number")
                    })
            })?;
            DynBool::full(shape, value, device)
                .map(TensorValue::Bool)
                .map_err(to_python_error)
        }
        other => Err(PyValueError::new_err(format!(
            "unsupported Tensor dtype {other:?}; expected 'float32', 'int64', or 'bool'"
        ))),
    }
}

#[pyfunction(name = "empty")]
#[pyo3(signature = (*shape, dtype="float32", device=None, requires_grad=false))]
pub(crate) fn empty_py(
    shape: &Bound<'_, PyTuple>,
    dtype: &str,
    device: Option<PyRef<'_, PyDevice>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let shape = validate_shape(shape_args(shape)?)?;
    let device = select_device(device);
    finish(empty_value(&shape, dtype, &device)?, requires_grad)
}

#[pyfunction(name = "full")]
#[pyo3(signature = (shape, fill_value, *, dtype="float32", device=None, requires_grad=false))]
pub(crate) fn full_py(
    shape: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: &str,
    device: Option<PyRef<'_, PyDevice>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let shape = validate_shape(shape_value(shape)?)?;
    let device = select_device(device);
    finish(
        full_value(&shape, fill_value, dtype, &device)?,
        requires_grad,
    )
}

#[pyfunction(name = "zeros")]
#[pyo3(signature = (*shape, dtype="float32", device=None, requires_grad=false))]
pub(crate) fn zeros_py(
    py: Python<'_>,
    shape: &Bound<'_, PyTuple>,
    dtype: &str,
    device: Option<PyRef<'_, PyDevice>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let shape = validate_shape(shape_args(shape)?)?;
    let device = select_device(device);
    let zero = 0_i64.into_pyobject(py)?.into_any();
    finish(full_value(&shape, &zero, dtype, &device)?, requires_grad)
}

#[pyfunction(name = "ones")]
#[pyo3(signature = (*shape, dtype="float32", device=None, requires_grad=false))]
pub(crate) fn ones_py(
    py: Python<'_>,
    shape: &Bound<'_, PyTuple>,
    dtype: &str,
    device: Option<PyRef<'_, PyDevice>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let shape = validate_shape(shape_args(shape)?)?;
    let device = select_device(device);
    let one = 1_i64.into_pyobject(py)?.into_any();
    finish(full_value(&shape, &one, dtype, &device)?, requires_grad)
}

fn random_float(
    shape: Vec<usize>,
    distribution: Distribution,
    device: Option<PyRef<'_, PyDevice>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let shape = validate_shape(shape)?;
    let device = select_device(device);
    let value = DynTensor::random(&shape, distribution, &device, DType::F32)
        .map(TensorValue::Float)
        .map_err(to_python_error)?;
    finish(value, requires_grad)
}

#[pyfunction(name = "rand")]
#[pyo3(signature = (*shape, dtype="float32", device=None, requires_grad=false))]
pub(crate) fn rand_py(
    shape: &Bound<'_, PyTuple>,
    dtype: &str,
    device: Option<PyRef<'_, PyDevice>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    if dtype != "float32" {
        return Err(PyTypeError::new_err("rand supports only dtype='float32'"));
    }
    random_float(
        shape_args(shape)?,
        Distribution::Default,
        device,
        requires_grad,
    )
}

#[pyfunction(name = "randn")]
#[pyo3(signature = (*shape, dtype="float32", device=None, requires_grad=false))]
pub(crate) fn randn_py(
    shape: &Bound<'_, PyTuple>,
    dtype: &str,
    device: Option<PyRef<'_, PyDevice>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    if dtype != "float32" {
        return Err(PyTypeError::new_err("randn supports only dtype='float32'"));
    }
    random_float(
        shape_args(shape)?,
        Distribution::Normal(0.0, 1.0),
        device,
        requires_grad,
    )
}

#[pyfunction(name = "randint")]
#[pyo3(signature = (low, high, shape, *, dtype="int64", device=None, requires_grad=false))]
pub(crate) fn randint_py(
    low: i64,
    high: i64,
    shape: &Bound<'_, PyAny>,
    dtype: &str,
    device: Option<PyRef<'_, PyDevice>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    if low >= high {
        return Err(PyValueError::new_err(format!(
            "randint requires low < high, got {low} and {high}"
        )));
    }
    if dtype != "int64" || requires_grad {
        return Err(PyTypeError::new_err(
            "randint supports only dtype='int64' and requires_grad=False",
        ));
    }
    let shape = validate_shape(shape_value(shape)?)?;
    let device = select_device(device);
    let bounds = if shape.iter().product::<usize>() == 0 {
        IntBounds::Empty
    } else {
        IntBounds::Range {
            min: low,
            max: high - 1,
        }
    };
    DynInt::random(
        &shape,
        Distribution::Uniform(low as f64, high as f64),
        &device,
        DType::I64,
    )
    .map(|value| PyTensor::from_int_inner_with_bounds(value, bounds))
    .map_err(to_python_error)
}

#[pyfunction(name = "arange")]
#[pyo3(signature = (start, end=None, step=1, *, dtype=None, device=None, requires_grad=false))]
pub(crate) fn arange_py(
    start: i64,
    end: Option<i64>,
    step: i64,
    dtype: Option<&str>,
    device: Option<PyRef<'_, PyDevice>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let (start, end) = end.map_or((0, start), |end| (start, end));
    let dtype = dtype.unwrap_or("int64");
    let device = select_device(device);
    let values = DynInt::arange(start, end, step, &device, DType::I64).map_err(to_python_error)?;
    match dtype {
        "int64" if !requires_grad => Ok(PyTensor::from_int_inner(values)),
        "int64" => Err(PyTypeError::new_err(
            "int64 arange does not support requires_grad=True",
        )),
        "float32" => finish(
            TensorValue::Float(values.to_float(DType::F32)),
            requires_grad,
        ),
        other => Err(PyValueError::new_err(format!(
            "arange supports dtype='int64' or 'float32', got {other:?}"
        ))),
    }
}

fn like_spec(
    input: &PyTensor,
    dtype: Option<&str>,
    device: Option<PyRef<'_, PyDevice>>,
) -> (Vec<usize>, String, Device) {
    let value = input.source.value();
    let shape = value.dims();
    let dtype = dtype.unwrap_or(value.dtype_name()).to_string();
    let device = device.map_or_else(
        || ensure_autodiff(value.device()),
        |device| ensure_autodiff(device.inner.as_ref().clone()),
    );
    (shape, dtype, device)
}

macro_rules! like_full {
    ($name:ident, $python_name:literal, $value:expr) => {
        #[pyfunction(name = $python_name)]
        #[pyo3(signature = (input, *, dtype=None, device=None, requires_grad=false))]
        pub(crate) fn $name(
            py: Python<'_>,
            input: PyRef<'_, PyTensor>,
            dtype: Option<&str>,
            device: Option<PyRef<'_, PyDevice>>,
            requires_grad: bool,
        ) -> PyResult<PyTensor> {
            let (shape, dtype, device) = like_spec(&input, dtype, device);
            let value = ($value).into_pyobject(py)?.into_any();
            finish(full_value(&shape, &value, &dtype, &device)?, requires_grad)
        }
    };
}

like_full!(zeros_like_py, "zeros_like", 0_i64);
like_full!(ones_like_py, "ones_like", 1_i64);

#[pyfunction(name = "empty_like")]
#[pyo3(signature = (input, *, dtype=None, device=None, requires_grad=false))]
pub(crate) fn empty_like_py(
    input: PyRef<'_, PyTensor>,
    dtype: Option<&str>,
    device: Option<PyRef<'_, PyDevice>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let (shape, dtype, device) = like_spec(&input, dtype, device);
    finish(empty_value(&shape, &dtype, &device)?, requires_grad)
}

#[pyfunction(name = "full_like")]
#[pyo3(signature = (input, fill_value, *, dtype=None, device=None, requires_grad=false))]
pub(crate) fn full_like_py(
    input: PyRef<'_, PyTensor>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&str>,
    device: Option<PyRef<'_, PyDevice>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let (shape, dtype, device) = like_spec(&input, dtype, device);
    finish(
        full_value(&shape, fill_value, &dtype, &device)?,
        requires_grad,
    )
}

#[pyfunction(name = "rand_like")]
#[pyo3(signature = (input, *, dtype=None, device=None, requires_grad=false))]
pub(crate) fn rand_like_py(
    input: PyRef<'_, PyTensor>,
    dtype: Option<&str>,
    device: Option<PyRef<'_, PyDevice>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let (shape, dtype, device) = like_spec(&input, dtype, device);
    if dtype != "float32" {
        return Err(PyTypeError::new_err(
            "rand_like supports only dtype='float32'",
        ));
    }
    let value = DynTensor::random(&shape, Distribution::Default, &device, DType::F32)
        .map(TensorValue::Float)
        .map_err(to_python_error)?;
    finish(value, requires_grad)
}

#[pyfunction(name = "randn_like")]
#[pyo3(signature = (input, *, dtype=None, device=None, requires_grad=false))]
pub(crate) fn randn_like_py(
    input: PyRef<'_, PyTensor>,
    dtype: Option<&str>,
    device: Option<PyRef<'_, PyDevice>>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let (shape, dtype, device) = like_spec(&input, dtype, device);
    if dtype != "float32" {
        return Err(PyTypeError::new_err(
            "randn_like supports only dtype='float32'",
        ));
    }
    let value = DynTensor::random(&shape, Distribution::Normal(0.0, 1.0), &device, DType::F32)
        .map(TensorValue::Float)
        .map_err(to_python_error)?;
    finish(value, requires_grad)
}
