//! Native differentiable neural-network operations used by Python layers and functionals.

use pyo3::prelude::*;

use crate::{grad_mode::is_grad_enabled, tensor::PyTensor, to_python_error};

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

    if !tracking {
        return Ok(PyTensor::from_inner(output));
    }
    let mut sources = vec![&*input, &*weight];
    if let Some(bias) = &bias {
        sources.push(bias);
    }
    Ok(PyTensor::from_operation(output, &sources))
}
