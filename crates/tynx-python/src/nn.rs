//! Native differentiable neural-network operations used by Python layers and functionals.

use pyo3::{exceptions::PyTypeError, prelude::*};
use tynx_capture::UnaryOp;
use tynx_core::Value;

use crate::{
    capture::record_conv2d, grad_mode::is_grad_enabled, tensor::PyTensor, to_python_error,
};

#[pyfunction(name = "_conv2d")]
#[pyo3(signature = (input, weight, bias, stride, padding, dilation, groups))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_py(
    input: PyRef<'_, PyTensor>,
    weight: PyRef<'_, PyTensor>,
    bias: Option<PyRef<'_, PyTensor>>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> PyResult<PyTensor> {
    let tracking = is_grad_enabled();
    let input_value = input.operation_float_value(tracking, "conv2d")?;
    let weight_value = weight.operation_float_value(tracking, "conv2d")?;
    let bias_value = bias
        .as_ref()
        .map(|value| value.operation_float_value(tracking, "conv2d"))
        .transpose()?;
    let output = input_value
        .conv2d(
            weight_value,
            bias_value,
            [stride.0, stride.1],
            [padding.0, padding.1],
            [dilation.0, dilation.1],
            groups,
        )
        .map_err(to_python_error)?;

    let mut sources = vec![&*input, &*weight];
    if let Some(bias) = &bias {
        sources.push(bias);
    }
    let result = if tracking {
        PyTensor::from_operation(output, &sources)
    } else {
        PyTensor::from_inner(output)
    };
    let trace = record_conv2d(
        &input,
        &weight,
        bias.as_deref(),
        [stride.0, stride.1],
        [padding.0, padding.1],
        [dilation.0, dilation.1],
        groups,
    )?;
    Ok(match trace {
        Some(trace) => result.with_trace(trace),
        None => result,
    })
}

#[pyfunction(name = "_max_pool2d")]
#[pyo3(signature = (input, kernel_size, stride, padding, dilation, ceil_mode))]
pub(crate) fn max_pool2d_py(
    input: PyRef<'_, PyTensor>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    ceil_mode: bool,
) -> PyResult<PyTensor> {
    let tracking = is_grad_enabled();
    let output = input
        .operation_float_value(tracking, "max_pool2d")?
        .max_pool2d(
            [kernel_size.0, kernel_size.1],
            [stride.0, stride.1],
            [padding.0, padding.1],
            [dilation.0, dilation.1],
            ceil_mode,
        )
        .map_err(to_python_error)?;
    let result = if tracking {
        PyTensor::from_operation(output, &[&input])
    } else {
        PyTensor::from_inner(output)
    };
    result.with_recorded_unary(
        &input,
        UnaryOp::MaxPool2d {
            kernel_size: [kernel_size.0, kernel_size.1],
            stride: [stride.0, stride.1],
            padding: [padding.0, padding.1],
            dilation: [dilation.0, dilation.1],
            ceil_mode,
        },
    )
}

#[pyfunction(name = "_avg_pool2d")]
#[pyo3(signature = (input, kernel_size, stride, padding, ceil_mode, count_include_pad))]
pub(crate) fn avg_pool2d_py(
    input: PyRef<'_, PyTensor>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    ceil_mode: bool,
    count_include_pad: bool,
) -> PyResult<PyTensor> {
    let tracking = is_grad_enabled();
    let output = input
        .operation_float_value(tracking, "avg_pool2d")?
        .avg_pool2d(
            [kernel_size.0, kernel_size.1],
            [stride.0, stride.1],
            [padding.0, padding.1],
            count_include_pad,
            ceil_mode,
        )
        .map_err(to_python_error)?;
    let result = if tracking {
        PyTensor::from_operation(output, &[&input])
    } else {
        PyTensor::from_inner(output)
    };
    result.with_recorded_unary(
        &input,
        UnaryOp::AvgPool2d {
            kernel_size: [kernel_size.0, kernel_size.1],
            stride: [stride.0, stride.1],
            padding: [padding.0, padding.1],
            count_include_pad,
            ceil_mode,
        },
    )
}

#[pyfunction(name = "_adaptive_avg_pool2d")]
pub(crate) fn adaptive_avg_pool2d_py(
    input: PyRef<'_, PyTensor>,
    output_size: (usize, usize),
) -> PyResult<PyTensor> {
    let tracking = is_grad_enabled();
    let output = input
        .operation_float_value(tracking, "adaptive_avg_pool2d")?
        .adaptive_avg_pool2d([output_size.0, output_size.1])
        .map_err(to_python_error)?;
    let result = if tracking {
        PyTensor::from_operation(output, &[&input])
    } else {
        PyTensor::from_inner(output)
    };
    result.with_recorded_unary(
        &input,
        UnaryOp::AdaptiveAvgPool2d {
            output_size: [output_size.0, output_size.1],
        },
    )
}

#[pyfunction(name = "_embedding")]
pub(crate) fn embedding_py(
    input: PyRef<'_, PyTensor>,
    weight: PyRef<'_, PyTensor>,
    padding_idx: Option<usize>,
) -> PyResult<PyTensor> {
    input.capture_unsupported("embedding")?;
    weight.capture_unsupported("embedding")?;
    let weight_shape = weight.detached_float_value("embedding")?.dims();
    if weight_shape.len() != 2 {
        return Err(PyTypeError::new_err(format!(
            "embedding weight must be rank-2 float32, got shape {weight_shape:?}"
        )));
    }
    input.validate_index_bounds(weight_shape[0], 0, "embedding")?;
    let indices = match input.detached_runtime_value() {
        Value::Int(indices) => indices,
        value => {
            return Err(PyTypeError::new_err(format!(
                "embedding input must be an int64 Tensor, got {value:?}"
            )));
        }
    };
    let tracking = is_grad_enabled();
    let output = weight
        .operation_float_value(tracking, "embedding")?
        .embedding(indices, padding_idx)
        .map_err(to_python_error)?;
    Ok(if tracking {
        PyTensor::from_operation(output, &[&weight])
    } else {
        PyTensor::from_inner(output)
    })
}
