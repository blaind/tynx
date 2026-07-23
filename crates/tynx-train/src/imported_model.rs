//! Slot-backed eager execution for imported ONNX training models.

mod executor;

use burn::tensor::Device;
use tynx_core::onnx_ir::ir::{ArgType, Argument, TensorType};
use tynx_core::{Env, Result, Session, TynxError, Value};

use crate::{
    ImportedState, InitializerNameOverrides, ParameterStore, TrainabilityOverrides,
    TrainabilityReport,
};

/// Imported ONNX graph with stable mutable parameter/buffer slots.
///
/// The public model owns lifecycle and state. Operator dispatch and slot resolution live in the
/// internal executor so expanding graph coverage does not expand this API type.
#[derive(Debug)]
pub struct ImportedModel {
    session: Session,
    state: ImportedState,
    trainability: TrainabilityReport,
    device: Device,
}

impl ImportedModel {
    /// Build a model using automatic initializer roles and preserved stable names.
    pub fn from_session(session: Session, device: Device) -> Result<Self> {
        Self::from_session_with(
            session,
            device,
            &TrainabilityOverrides::new(),
            &InitializerNameOverrides::new(),
        )
    }

    /// Build a model using explicit role and stable-name overrides.
    pub fn from_session_with(
        session: Session,
        device: Device,
        role_overrides: &TrainabilityOverrides,
        name_overrides: &InitializerNameOverrides,
    ) -> Result<Self> {
        Self::from_session_for_outputs_with(session, device, None, role_overrides, name_overrides)
    }

    /// Build a model after validating either every output or an explicit output subset.
    pub fn from_session_for_outputs_with(
        session: Session,
        device: Device,
        outputs: Option<&[&str]>,
        role_overrides: &TrainabilityOverrides,
        name_overrides: &InitializerNameOverrides,
    ) -> Result<Self> {
        let trainability = analyze_session_outputs(&session, outputs, role_overrides);
        trainability.require_trainable()?;
        let state = ImportedState::materialize_with_names(
            session.graph(),
            &device,
            role_overrides,
            name_overrides,
            session.initializer_names(),
        )?;
        executor::validate(session.graph(), &state)?;
        Ok(Self {
            session,
            state,
            trainability,
            device,
        })
    }

    /// Return the immutable parsed inference session.
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Return the runtime device used for inputs, state, and outputs.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Return the output-specific report validated during construction.
    pub fn trainability_report(&self) -> &TrainabilityReport {
        &self.trainability
    }

    /// Re-run output-specific trainability analysis using declared ONNX output names.
    pub fn trainability_for_outputs(&self, outputs: Option<&[&str]>) -> TrainabilityReport {
        analyze_session_outputs(&self.session, outputs, &TrainabilityOverrides::new())
    }

    /// Return imported parameter and buffer state.
    pub fn state(&self) -> &ImportedState {
        &self.state
    }

    /// Return the optimizer/backward-facing parameter store.
    pub fn parameters(&self) -> &ParameterStore {
        self.state.store()
    }

    /// Run one eager forward with current slot values and return declared graph outputs.
    pub fn run(&self, env: Env) -> Result<Env> {
        self.run_with_tracking(env, true)
    }

    /// Run one eager forward, choosing whether parameters participate in autodiff.
    pub fn run_with_tracking(&self, mut env: Env, tracking: bool) -> Result<Env> {
        validate_inputs(&self.session, &self.device, &env)?;
        self.session.internalize_inputs(&mut env)?;
        executor::run(&self.session, &self.state, &self.device, env, tracking)
    }
}

fn validate_inputs(session: &Session, device: &Device, env: &Env) -> Result<()> {
    for input in session.inputs() {
        let value = env
            .get(&input.name)
            .ok_or_else(|| TynxError::MissingValue(input.name.clone()))?;
        validate_input(input, device, value)?;
    }
    Ok(())
}

fn validate_input(input: &Argument, device: &Device, value: &Value) -> Result<()> {
    match &input.ty {
        ArgType::Tensor(expected) => validate_tensor_input(&input.name, expected, device, value),
        ArgType::ScalarTensor(dtype) => validate_tensor_input(
            &input.name,
            &TensorType::new(*dtype, 1, Some(vec![Some(1)])),
            device,
            value,
        ),
        ArgType::ScalarNative(_) if matches!(value, Value::Scalar(_)) => Ok(()),
        ArgType::Shape(rank) => match value {
            Value::Shape(shape) if shape.len() == *rank => Ok(()),
            Value::Shape(shape) => Err(TynxError::Shape(format!(
                "imported input '{}' expects a shape of length {rank}, got length {}",
                input.name,
                shape.len()
            ))),
            other => Err(TynxError::TypeMismatch(format!(
                "imported input '{}' expects {}, got {other:?}",
                input.name, input.ty
            ))),
        },
        _ => Err(TynxError::TypeMismatch(format!(
            "imported input '{}' expects {}, got {value:?}",
            input.name, input.ty
        ))),
    }
}

fn validate_tensor_input(
    name: &str,
    expected: &TensorType,
    device: &Device,
    value: &Value,
) -> Result<()> {
    let (actual_dtype, actual_shape, actual_device) = match value {
        Value::Tensor(tensor) => (tensor.dtype(), tensor.dims(), tensor.device()),
        Value::Int(tensor) => (tensor.dtype(), tensor.dims(), tensor.device()),
        Value::Bool(tensor) => (tensor.dtype(), tensor.dims(), tensor.device()),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "imported input '{name}' expects a tensor, got {other:?}"
            )));
        }
    };
    if actual_dtype != expected.dtype {
        return Err(TynxError::TypeMismatch(format!(
            "imported input '{name}' expects dtype {:?}, got {actual_dtype:?}",
            expected.dtype
        )));
    }
    let compatible_device = match value {
        Value::Tensor(_) => {
            actual_device == *device && actual_device.is_autodiff() == device.is_autodiff()
        }
        Value::Int(_) | Value::Bool(_) => {
            actual_device.clone().inner() == device.clone().inner()
        }
        Value::Scalar(_) | Value::Shape(_) => unreachable!("tensor inputs were matched above"),
    };
    if !compatible_device {
        return Err(TynxError::DeviceMismatch {
            name: name.to_string(),
            expected: format!("{device:?}"),
            actual: format!("{actual_device:?}"),
        });
    }
    if actual_shape.len() != expected.rank {
        return Err(TynxError::Shape(format!(
            "imported input '{name}' expects rank {}, got shape {actual_shape:?}",
            expected.rank
        )));
    }
    if let Some(expected_shape) = &expected.static_shape {
        for (axis, (expected_dimension, actual_dimension)) in
            expected_shape.iter().zip(&actual_shape).enumerate()
        {
            if let Some(expected_dimension) = expected_dimension
                && actual_dimension != expected_dimension
            {
                return Err(TynxError::Shape(format!(
                    "imported input '{name}' expects shape {}, got {actual_shape:?}; dimension {axis} must be {expected_dimension}",
                    display_shape(expected_shape)
                )));
            }
        }
    }
    Ok(())
}

fn display_shape(shape: &[Option<usize>]) -> String {
    let dimensions = shape
        .iter()
        .map(|dimension| dimension.map_or_else(|| "?".to_string(), |value| value.to_string()))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{dimensions}]")
}

fn analyze_session_outputs(
    session: &Session,
    outputs: Option<&[&str]>,
    role_overrides: &TrainabilityOverrides,
) -> TrainabilityReport {
    let internal_outputs = outputs.map(|outputs| {
        outputs
            .iter()
            .map(|output| {
                session
                    .internal_output_name(output)
                    .unwrap_or(output)
                    .to_string()
            })
            .collect::<Vec<_>>()
    });
    let mut report = match &internal_outputs {
        Some(outputs) => {
            let outputs = outputs.iter().map(String::as_str).collect::<Vec<_>>();
            TrainabilityReport::analyze_outputs_with_names(
                session.graph(),
                &outputs,
                role_overrides,
                session.initializer_names(),
            )
        }
        None => TrainabilityReport::analyze_all_outputs_with_names(
            session.graph(),
            role_overrides,
            session.initializer_names(),
        ),
    };
    let internal_to_public = session
        .output_name_mapping()
        .map(|(public, internal)| (internal.to_string(), public.to_string()))
        .collect::<std::collections::HashMap<_, _>>();
    report.remap_outputs(&internal_to_public);
    report
}

#[cfg(test)]
mod tests;
