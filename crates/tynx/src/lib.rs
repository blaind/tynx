#![forbid(unsafe_code)]

mod error;
mod tensor;

pub use error::{Result, TynxError};
pub use tensor::{DynBool, DynInt, DynTensor, MAX_RANK};
