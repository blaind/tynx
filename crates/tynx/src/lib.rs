#![forbid(unsafe_code)]

mod error;
mod tensor;
mod value;

pub use error::{Result, TynxError};
pub use tensor::{DynBool, DynInt, DynTensor, MAX_RANK};
pub use value::{Scalar, Value};
