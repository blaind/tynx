#![forbid(unsafe_code)]

//! Training utilities for Tynx.
//!
//! This crate builds losses and optimizers from the same dynamic tensor operations used by the
//! ONNX interpreter. Autodiff integration remains feature-gated in the Tynx core.

pub mod autograd;
pub mod backward_support;
pub mod imported_model;
pub mod imported_state;
pub mod loss;
pub mod optimizer;
pub mod parameter;
pub mod store;
pub mod trainability;

pub use autograd::{BackwardResult, backward};
pub use backward_support::{BackwardCapability, BackwardSupportRegistry};
pub use imported_model::ImportedModel;
pub use imported_state::{ImportedState, InitializerNameOverrides};
pub use optimizer::{Adam, AdamConfig, AdamW, AdamWConfig, Sgd, SgdConfig};
pub use parameter::{ParamId, ParameterContract, ParameterSlot};
pub use store::ParameterStore;
pub use trainability::{
    BackwardPathIssue, InitializerId, InitializerReport, InitializerRole, InitializerUse,
    TrainabilityOverrides, TrainabilityReport,
};
