#![forbid(unsafe_code)]

//! Training utilities for Tynx.
//!
//! This crate builds losses and optimizers from the same dynamic tensor operations used by the
//! ONNX interpreter. Autodiff integration remains feature-gated in the Tynx core.

pub mod loss;
pub mod parameter;
pub mod store;

pub use parameter::{ParamId, ParameterContract, ParameterSlot};
pub use store::ParameterStore;
