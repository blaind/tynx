//! Slot-backed eager execution for imported ONNX training models.

mod executor;

use burn::tensor::Device;
use tynx_core::{Env, Result, Session};

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
        let state = ImportedState::materialize_with(
            session.graph(),
            &device,
            role_overrides,
            name_overrides,
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
    pub fn run_with_tracking(&self, env: Env, tracking: bool) -> Result<Env> {
        executor::run(&self.session, &self.state, &self.device, env, tracking)
    }
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
            TrainabilityReport::analyze_outputs_with(session.graph(), &outputs, role_overrides)
        }
        None => TrainabilityReport::analyze_all_outputs_with(session.graph(), role_overrides),
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
