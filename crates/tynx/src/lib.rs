#![forbid(unsafe_code)]

mod error;
mod session;
mod tensor;
mod value;

pub use error::{Result, TynxError};
pub use session::Session;
pub use tensor::{DynBool, DynInt, DynTensor, MAX_RANK};
pub use value::{Scalar, Value};

pub use onnx_ir;
