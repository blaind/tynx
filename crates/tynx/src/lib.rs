#![forbid(unsafe_code)]

mod device;
mod error;
mod external;
#[cfg(all(feature = "external-wgpu", any(feature = "wgpu", feature = "vulkan")))]
mod external_wgpu;
mod initializer;
mod interpreter;
mod session;
mod tensor;
mod value;

#[cfg(feature = "training")]
pub use burn::tensor::Gradients;
pub use burn::tensor::{BoolStore, DType, Device, Distribution, Slice, TensorData};
pub use device::{
    allocation_size_limit, default_device, synchronize, synchronize_initialized_default_device,
    take_device_error,
};
pub use error::{Result, TynxError};
pub use external::{
    AcquiredExternalTensorDescriptor, DeviceContextCapability, ExternalAccess, ExternalBufferLease,
    ExternalBufferUsage, ExternalSubmission, ExternalTensorDescriptor, ExternalTensorRetention,
    SubmissionToken, ValidatedExternalTensorDescriptor,
};
#[cfg(all(feature = "external-wgpu", any(feature = "wgpu", feature = "vulkan")))]
pub use external_wgpu::ExternalWgpuContext;
pub use initializer::InitializerId;
pub use interpreter::{
    Env,
    binary::prelu_values as execute_onnx_prelu,
    convolution::{
        conv_transpose1d_values as execute_onnx_conv_transpose1d,
        conv_transpose2d_values as execute_onnx_conv_transpose2d,
        conv_transpose3d_values as execute_onnx_conv_transpose3d,
        conv1d_values as execute_onnx_conv1d, conv3d_values as execute_onnx_conv3d,
    },
    execute,
    gather::gather_values as execute_onnx_gather,
    matrix::matmul_values as execute_onnx_matmul,
    normalization::{
        group_normalization_values as execute_onnx_group_normalization,
        instance_normalization_values as execute_onnx_instance_normalization,
        layer_normalization_values as execute_onnx_layer_normalization,
    },
    spatial::padding2d as resolve_onnx_padding2d,
};
pub use onnx_ir;
pub use session::{PreparedSession, Session};
pub use tensor::{DynBool, DynInt, DynTensor, MAX_RANK};
pub use value::{Scalar, Value};
