#![forbid(unsafe_code)]

mod device;
mod error;
mod interpreter;
mod session;
mod tensor;
mod value;

pub use device::default_device;
pub use error::{Result, TynxError};
pub use interpreter::{Env, execute};
pub use session::Session;
pub use tensor::{DynBool, DynInt, DynTensor, MAX_RANK};
pub use value::{Scalar, Value};

pub use burn::tensor::{DType, Device, TensorData};

#[cfg(feature = "training")]
pub use burn::tensor::Gradients;

pub use onnx_ir;
