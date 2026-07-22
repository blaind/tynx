//! Native random-state and sampling operations shared by Python distributions and layers.

use pyo3::{exceptions::PyValueError, prelude::*};
use tynx_capture::UnaryOp;
use tynx_core::{DType, Device, Distribution, DynTensor};

use crate::{grad_mode::is_grad_enabled, tensor::PyTensor, to_python_error};

#[pyfunction(name = "manual_seed")]
pub(crate) fn manual_seed_py(seed: u64) {
    Device::autodiff(tynx_core::default_device()).seed(seed);
}

#[pyfunction(name = "_normal_sample")]
#[pyo3(signature = (loc, scale, seed=None))]
pub(crate) fn normal_sample_py(
    loc: PyRef<'_, PyTensor>,
    scale: PyRef<'_, PyTensor>,
    seed: Option<u64>,
) -> PyResult<PyTensor> {
    loc.capture_unsupported("random Normal sampling")?;
    scale.capture_unsupported("random Normal sampling")?;
    let loc = loc.detached_float_value("Normal.sample")?;
    let scale = scale.detached_float_value("Normal.sample")?;
    let device = loc.device();
    if let Some(seed) = seed {
        device.seed(seed);
    }
    let dims = loc.dims();
    let noise = DynTensor::random(&dims, Distribution::Normal(0.0, 1.0), &device, DType::F32)
        .map_err(to_python_error)?;
    let sample = loc
        .add_broadcast(scale.mul_broadcast(noise).map_err(to_python_error)?)
        .map_err(to_python_error)?;
    Ok(PyTensor::from_inner(sample.detach()))
}

#[pyfunction(name = "_categorical_sample")]
#[pyo3(signature = (logits, seed=None))]
pub(crate) fn categorical_sample_py(
    logits: PyRef<'_, PyTensor>,
    seed: Option<u64>,
) -> PyResult<PyTensor> {
    logits.capture_unsupported("random Categorical sampling")?;
    let logits = logits.detached_float_value("Categorical.sample")?;
    let rank = logits.rank();
    if rank == 0 {
        return Err(PyValueError::new_err(
            "Categorical logits must have at least one dimension",
        ));
    }
    let categories = logits.dims()[rank - 1];
    if categories == 0 {
        return Err(PyValueError::new_err(
            "Categorical logits must contain at least one category",
        ));
    }
    let device = logits.device();
    if let Some(seed) = seed {
        device.seed(seed);
    }
    let dims = logits.dims();
    let uniform = DynTensor::random(
        &dims,
        Distribution::Uniform(1.0e-7, 1.0 - 1.0e-7),
        &device,
        DType::F32,
    )
    .map_err(to_python_error)?;
    let gumbel = uniform.log().mul_scalar(-1.0).log().mul_scalar(-1.0);
    let perturbed = logits.add_broadcast(gumbel).map_err(to_python_error)?;
    let indices = perturbed.arg_extreme(rank - 1, true, false);
    let mut output_shape = dims[..rank - 1].to_vec();
    if output_shape.is_empty() {
        output_shape.push(1);
    }
    Ok(PyTensor::from_int_inner(
        indices.reshape(output_shape).map_err(to_python_error)?,
    ))
}

#[pyfunction(name = "_dropout")]
pub(crate) fn dropout_py(input: PyRef<'_, PyTensor>, probability: f64) -> PyResult<PyTensor> {
    if !(0.0..=1.0).contains(&probability) {
        return Err(PyValueError::new_err(format!(
            "dropout probability must be between 0 and 1, got {probability}"
        )));
    }

    let tracking = is_grad_enabled();
    let value = input.operation_float_value(tracking, "dropout")?;
    let output = if probability == 1.0 {
        value.mul_scalar(0.0)
    } else if probability == 0.0 {
        value
    } else {
        let mask = DynTensor::random(
            &value.dims(),
            Distribution::Bernoulli(1.0 - probability),
            &value.device(),
            DType::F32,
        )
        .map_err(to_python_error)?;
        value
            .mul_broadcast(mask)
            .map_err(to_python_error)?
            .mul_scalar(1.0 / (1.0 - probability))
    };

    let output = if tracking {
        PyTensor::from_operation(output, &[&input])
    } else {
        PyTensor::from_inner(output)
    };
    output.with_recorded_unary(&input, UnaryOp::Dropout(probability))
}
