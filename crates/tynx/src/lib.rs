#![forbid(unsafe_code)]

mod device;
mod error;
mod external;
mod initializer;
mod interpreter;
mod session;
mod tensor;
mod value;

pub use device::{allocation_size_limit, default_device, synchronize, take_device_error};
pub use error::{Result, TynxError};
pub use external::{
    AcquiredExternalTensorDescriptor, DeviceContextCapability, ExternalAccess, ExternalBufferLease,
    ExternalBufferUsage, ExternalSubmission, ExternalTensorDescriptor, ExternalTensorRetention,
    SubmissionToken, ValidatedExternalTensorDescriptor,
};
pub use initializer::InitializerId;
pub use interpreter::binary::prelu_values as execute_onnx_prelu;
pub use interpreter::convolution::{
    conv_transpose1d_values as execute_onnx_conv_transpose1d,
    conv_transpose2d_values as execute_onnx_conv_transpose2d,
    conv_transpose3d_values as execute_onnx_conv_transpose3d, conv1d_values as execute_onnx_conv1d,
    conv3d_values as execute_onnx_conv3d,
};
pub use interpreter::gather::gather_values as execute_onnx_gather;
pub use interpreter::matrix::matmul_values as execute_onnx_matmul;
pub use interpreter::normalization::{
    group_normalization_values as execute_onnx_group_normalization,
    instance_normalization_values as execute_onnx_instance_normalization,
    layer_normalization_values as execute_onnx_layer_normalization,
};
pub use interpreter::spatial::padding2d as resolve_onnx_padding2d;
pub use interpreter::{Env, execute};
pub use session::{PreparedSession, Session};
pub use tensor::{DynBool, DynInt, DynTensor, MAX_RANK};
pub use value::{Scalar, Value};

pub use burn::tensor::{BoolStore, DType, Device, Distribution, Slice, TensorData};

#[cfg(feature = "training")]
pub use burn::tensor::Gradients;

pub use onnx_ir;
