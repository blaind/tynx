//! Ordinary-Python optimizer state dictionaries over Rust-owned payloads.

use std::collections::BTreeMap;

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyDict, PyDictMethods, PyList},
};
use tynx_train::{
    AdamConfig, AdamParameterState, AdamStateDict, AdamStateKind, SgdConfig, SgdParameterState,
    SgdStateDict,
};

use crate::{tensor::PyTensor, to_python_error};

const STATE_VERSION: u64 = 1;

pub(super) fn sgd_to_python(py: Python<'_>, state: &SgdStateDict) -> PyResult<Py<PyDict>> {
    let result = state_header(py, "SGD", state.parameter_names())?;
    let config = PyDict::new(py);
    let values = state.config();
    config.set_item("lr", values.learning_rate())?;
    config.set_item("momentum", values.momentum())?;
    config.set_item("dampening", values.dampening())?;
    config.set_item("weight_decay", values.weight_decay())?;
    config.set_item("nesterov", values.is_nesterov())?;
    result.set_item("config", config)?;

    let entries = PyDict::new(py);
    for (name, parameter_state) in state.state() {
        let entry = PyDict::new(py);
        entry.set_item(
            "momentum_buffer",
            Py::new(py, PyTensor::from_inner(parameter_state.momentum_buffer()))?,
        )?;
        entries.set_item(name, entry)?;
    }
    result.set_item("state", entries)?;
    Ok(result.unbind())
}

pub(super) fn sgd_from_python(value: &Bound<'_, PyAny>) -> PyResult<SgdStateDict> {
    let state = state_dictionary(value, "SGD")?;
    let config = required_dictionary(&state, "config")?;
    let config = SgdConfig::new(required_extract(&config, "lr")?)
        .with_momentum(required_extract(&config, "momentum")?)
        .with_dampening(required_extract(&config, "dampening")?)
        .with_weight_decay(required_extract(&config, "weight_decay")?)
        .with_nesterov(required_extract(&config, "nesterov")?);
    let names = required_extract(&state, "parameter_names")?;
    let entries = required_dictionary(&state, "state")?;
    let mut native = BTreeMap::new();
    for (name, entry) in entries.iter() {
        let name = name.extract::<String>().map_err(|_| {
            PyTypeError::new_err("SGD state keys must be stable parameter-name strings")
        })?;
        let entry = entry.cast::<PyDict>().map_err(|_| {
            PyTypeError::new_err(format!("SGD state for {name:?} must be a dictionary"))
        })?;
        native.insert(
            name,
            SgdParameterState::new(required_tensor(entry, "momentum_buffer")?),
        );
    }
    SgdStateDict::new(config, names, native).map_err(to_python_error)
}

pub(super) fn adam_to_python(py: Python<'_>, state: &AdamStateDict) -> PyResult<Py<PyDict>> {
    let optimizer = match state.kind() {
        AdamStateKind::Adam => "Adam",
        AdamStateKind::AdamW => "AdamW",
    };
    let result = state_header(py, optimizer, state.parameter_names())?;
    let config = PyDict::new(py);
    let values = state.config();
    config.set_item("lr", values.learning_rate())?;
    config.set_item("betas", values.betas())?;
    config.set_item("eps", values.epsilon())?;
    config.set_item("weight_decay", values.weight_decay())?;
    config.set_item("amsgrad", values.is_amsgrad())?;
    result.set_item("config", config)?;

    let entries = PyDict::new(py);
    for (name, parameter_state) in state.state() {
        let entry = PyDict::new(py);
        entry.set_item("step", parameter_state.step())?;
        entry.set_item(
            "exp_avg",
            Py::new(py, PyTensor::from_inner(parameter_state.first_moment()))?,
        )?;
        entry.set_item(
            "exp_avg_sq",
            Py::new(py, PyTensor::from_inner(parameter_state.second_moment()))?,
        )?;
        if let Some(maximum) = parameter_state.max_second_moment() {
            entry.set_item(
                "max_exp_avg_sq",
                Py::new(py, PyTensor::from_inner(maximum))?,
            )?;
        }
        entries.set_item(name, entry)?;
    }
    result.set_item("state", entries)?;
    Ok(result.unbind())
}

pub(super) fn adam_from_python(
    value: &Bound<'_, PyAny>,
    expected_kind: AdamStateKind,
) -> PyResult<AdamStateDict> {
    let optimizer = match expected_kind {
        AdamStateKind::Adam => "Adam",
        AdamStateKind::AdamW => "AdamW",
    };
    let state = state_dictionary(value, optimizer)?;
    let config = required_dictionary(&state, "config")?;
    let betas: (f64, f64) = required_extract(&config, "betas")?;
    let config = AdamConfig::new(required_extract(&config, "lr")?)
        .with_betas(betas.0, betas.1)
        .with_epsilon(required_extract(&config, "eps")?)
        .with_weight_decay(required_extract(&config, "weight_decay")?)
        .with_amsgrad(required_extract(&config, "amsgrad")?);
    let names = required_extract(&state, "parameter_names")?;
    let entries = required_dictionary(&state, "state")?;
    let mut native = BTreeMap::new();
    for (name, entry) in entries.iter() {
        let name = name.extract::<String>().map_err(|_| {
            PyTypeError::new_err(format!(
                "{optimizer} state keys must be stable parameter-name strings"
            ))
        })?;
        let entry = entry.cast::<PyDict>().map_err(|_| {
            PyTypeError::new_err(format!(
                "{optimizer} state for {name:?} must be a dictionary"
            ))
        })?;
        let maximum = if config.is_amsgrad() {
            Some(required_tensor(entry, "max_exp_avg_sq")?)
        } else {
            None
        };
        native.insert(
            name,
            AdamParameterState::new(
                required_extract(entry, "step")?,
                required_tensor(entry, "exp_avg")?,
                required_tensor(entry, "exp_avg_sq")?,
                maximum,
            ),
        );
    }
    AdamStateDict::new(expected_kind, config, names, native).map_err(to_python_error)
}

fn state_header<'py>(
    py: Python<'py>,
    optimizer: &str,
    parameter_names: &[String],
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("version", STATE_VERSION)?;
    result.set_item("optimizer", optimizer)?;
    result.set_item("parameter_names", PyList::new(py, parameter_names)?)?;
    Ok(result)
}

fn state_dictionary<'py>(
    value: &Bound<'py, PyAny>,
    expected_optimizer: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let state = value.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{expected_optimizer} state_dict must be a dictionary"
        ))
    })?;
    let version: u64 = required_extract(state, "version")?;
    if version != STATE_VERSION {
        return Err(PyValueError::new_err(format!(
            "unsupported optimizer state_dict version {version}; expected {STATE_VERSION}"
        )));
    }
    let optimizer: String = required_extract(state, "optimizer")?;
    if optimizer != expected_optimizer {
        return Err(PyValueError::new_err(format!(
            "cannot load {optimizer} state into {expected_optimizer} optimizer"
        )));
    }
    Ok(state.clone())
}

fn required_dictionary<'py>(
    dictionary: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Bound<'py, PyDict>> {
    required(dictionary, key)?
        .cast_into::<PyDict>()
        .map_err(|_| PyTypeError::new_err(format!("state_dict field {key:?} must be a dictionary")))
}

fn required_extract<'py, T>(dictionary: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    T: for<'a> FromPyObject<'a, 'py>,
{
    required(dictionary, key)?
        .extract::<T>()
        .map_err(|_| PyTypeError::new_err(format!("invalid state_dict field {key:?}")))
}

fn required_tensor(dictionary: &Bound<'_, PyDict>, key: &str) -> PyResult<tynx_core::DynTensor> {
    let value = required(dictionary, key)?;
    let tensor = value.extract::<PyRef<'_, PyTensor>>().map_err(|_| {
        PyTypeError::new_err(format!("state_dict field {key:?} must be a Tynx Tensor"))
    })?;
    tensor.detached_float_value("optimizer state")
}

fn required<'py>(dictionary: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dictionary
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("state_dict is missing field {key:?}")))
}
