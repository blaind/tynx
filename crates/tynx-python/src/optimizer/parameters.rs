//! Python optimizer parameter normalization.

use std::collections::HashSet;

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyAnyMethods, PyTuple, PyTupleMethods},
};
use tynx_train::{ParameterSlot, ParameterStore};

use crate::{tensor::PyTensor, to_python_error};

pub(crate) struct CollectedParameters {
    pub(crate) slots: Vec<ParameterSlot>,
    pub(crate) named: Option<ParameterStore>,
}

pub(crate) fn collect_parameters(
    parameters: &Bound<'_, PyAny>,
    optimizer_name: &str,
) -> PyResult<CollectedParameters> {
    let iterator = parameters.try_iter().map_err(|_| {
        PyTypeError::new_err(format!(
            "{optimizer_name} parameters must be an iterable of Parameter objects or (name, Parameter) pairs"
        ))
    })?;
    let mut slots: Vec<ParameterSlot> = Vec::new();
    let mut named_entries: Vec<(String, ParameterSlot)> = Vec::new();
    let mut named_input: Option<bool> = None;
    for item in iterator {
        let item = item?;
        let pair = item.cast::<PyTuple>().ok();
        let is_named = pair.is_some();
        if named_input.is_some_and(|expected| expected != is_named) {
            return Err(PyTypeError::new_err(format!(
                "{optimizer_name} parameters cannot mix Parameter objects and named pairs"
            )));
        }
        named_input = Some(is_named);

        let (name, parameter) = match pair {
            Some(pair) => {
                if pair.len() != 2 {
                    return Err(PyTypeError::new_err(format!(
                        "{optimizer_name} named parameters must be (name, Parameter) pairs"
                    )));
                }
                let name = pair.get_item(0)?.extract::<String>().map_err(|_| {
                    PyTypeError::new_err(format!(
                        "{optimizer_name} parameter names must be strings"
                    ))
                })?;
                let parameter = extract_parameter(&pair.get_item(1)?, optimizer_name)?;
                (Some(name), parameter)
            }
            None => (None, extract_parameter(&item, optimizer_name)?),
        };
        if !slots.iter().any(|existing| existing.id() == parameter.id()) {
            slots.push(parameter.clone());
        }
        if let Some(name) = name {
            named_entries.push((name, parameter));
        }
    }
    if slots.is_empty() {
        return Err(PyValueError::new_err(format!(
            "{optimizer_name} requires at least one Parameter"
        )));
    }

    let named = if named_input == Some(true) {
        let mut store = ParameterStore::new();
        for (name, parameter) in named_entries {
            store.insert(name, parameter).map_err(to_python_error)?;
        }
        Some(store)
    } else {
        let inferred = slots
            .iter()
            .map(ParameterSlot::name)
            .collect::<Option<Vec<_>>>();
        match inferred {
            Some(names)
                if names.iter().all(|name| !name.is_empty())
                    && names.iter().collect::<HashSet<_>>().len() == names.len() =>
            {
                let mut store = ParameterStore::new();
                for (name, parameter) in names.into_iter().zip(&slots) {
                    store
                        .insert(name, parameter.clone())
                        .map_err(to_python_error)?;
                }
                Some(store)
            }
            _ => None,
        }
    };
    Ok(CollectedParameters { slots, named })
}

fn extract_parameter(value: &Bound<'_, PyAny>, optimizer_name: &str) -> PyResult<ParameterSlot> {
    let tensor = value.extract::<PyRef<'_, PyTensor>>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{optimizer_name} parameters must contain only Parameter objects"
        ))
    })?;
    let slot = tensor.parameter_slot().ok_or_else(|| {
        PyTypeError::new_err(format!(
            "{optimizer_name} parameters must contain only Parameter objects"
        ))
    })?;
    if !slot.contract().trainable() {
        return Err(PyTypeError::new_err(format!(
            "{optimizer_name} parameters must contain only Parameter objects"
        )));
    }
    Ok(slot)
}
