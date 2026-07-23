//! Zero-copy adoption of externally owned WGPU storage.

use std::{
    any::Any,
    fmt::{Debug, Formatter},
    sync::Arc,
};

#[cfg(feature = "training")]
use burn::backend::{Autodiff, AutodiffBackend};
use burn::prelude::{DeviceKind, Shape, Tensor};
use cubecl::{
    Runtime,
    wgpu::{RuntimeOptions, WgpuCompiler, WgpuDevice, WgpuResource, WgpuRuntime, WgpuSetup},
};

#[cfg(feature = "wgpu")]
use cubecl::wgpu::AutoCompiler;
#[cfg(feature = "vulkan")]
use cubecl::wgpu::SpirvCompiler;

use crate::{
    AcquiredExternalTensorDescriptor, Device, DeviceContextCapability, DynTensor,
    ExternalBufferLease, ExternalBufferUsage, Result, TynxError,
};

#[derive(Debug, Clone, Copy)]
enum Compiler {
    #[cfg(feature = "wgpu")]
    Automatic,
    #[cfg(feature = "vulkan")]
    Vulkan,
}

/// An engine-owned WGPU buffer paired with its logical allocation lease.
///
/// Cloning a [`wgpu::Buffer`] keeps the GPU object alive but does not stop an engine allocator
/// from recycling the logical slot. The private owner is therefore retained until CubeCL releases
/// every managed handle and queued binding that can access the buffer.
struct ExternalWgpuBuffer {
    buffer: wgpu::Buffer,
    _owner: Arc<dyn Any + Send + Sync>,
}

impl Debug for ExternalWgpuBuffer {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExternalWgpuBuffer")
            .field("buffer", &self.buffer)
            .finish_non_exhaustive()
    }
}

/// A Tynx execution context initialized from an engine-owned WGPU device and queue.
///
/// The context capability must be cloned into every external buffer lease, tensor descriptor, and
/// producer submission token submitted to this context. Adoption registers a view of the original
/// WGPU buffer with CubeCL; it does not allocate a staging buffer or copy tensor data.
#[derive(Debug, Clone)]
pub struct ExternalWgpuContext {
    capability: DeviceContextCapability,
    runtime_device: WgpuDevice,
    device: Device,
    compiler: Compiler,
    binding_alignment: u64,
}

impl ExternalWgpuContext {
    /// Initialize the automatic WGPU compiler on an existing engine device and queue.
    ///
    /// Create at most one Tynx context for each engine device generation. Other runtimes may use
    /// the same underlying device and queue through their own CubeCL runtime identities.
    #[cfg(feature = "wgpu")]
    pub fn from_wgpu_setup(setup: WgpuSetup, options: RuntimeOptions) -> Result<Self> {
        let binding_alignment =
            u64::from(setup.device.limits().min_storage_buffer_offset_alignment);
        let runtime_device = cubecl::wgpu::init_device(setup, options);
        let device = Device::wgpu(existing_device_kind(&runtime_device)?);
        Ok(Self {
            capability: DeviceContextCapability::new(),
            runtime_device,
            device,
            compiler: Compiler::Automatic,
            binding_alignment,
        })
    }

    /// Initialize the SPIR-V compiler on an existing engine device and queue.
    ///
    /// Create at most one Tynx context for each engine device generation. Other runtimes may use
    /// the same underlying device and queue through their own CubeCL runtime identities.
    #[cfg(feature = "vulkan")]
    pub fn from_vulkan_setup(setup: WgpuSetup, options: RuntimeOptions) -> Result<Self> {
        let binding_alignment =
            u64::from(setup.device.limits().min_storage_buffer_offset_alignment);
        let runtime_device =
            cubecl::wgpu::init_device_with_compiler::<SpirvCompiler>(setup, options);
        let device = Device::vulkan(existing_device_kind(&runtime_device)?);
        Ok(Self {
            capability: DeviceContextCapability::new(),
            runtime_device,
            device,
            compiler: Compiler::Vulkan,
            binding_alignment,
        })
    }

    /// Return the opaque identity required by descriptors targeting this context.
    pub fn capability(&self) -> DeviceContextCapability {
        self.capability.clone()
    }

    /// Return the inference device backed by the shared engine context.
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Bind an engine buffer and its logical allocation owner to this context.
    ///
    /// The returned lease derives its byte length, storage usage, and binding alignment from the
    /// actual WGPU objects. The owner can release an engine read lease from its `Drop`
    /// implementation; Tynx keeps it alive through asynchronous execution and autodiff.
    pub fn lease_buffer<R>(
        &self,
        buffer: wgpu::Buffer,
        owner: Arc<R>,
    ) -> Result<ExternalBufferLease>
    where
        R: Any + Send + Sync,
    {
        let byte_length = buffer.size();
        let usage = if buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
            ExternalBufferUsage::Storage
        } else {
            ExternalBufferUsage::NonStorage
        };
        ExternalBufferLease::new(
            self.capability(),
            Arc::new(ExternalWgpuBuffer {
                buffer,
                _owner: owner,
            }),
            byte_length,
            self.binding_alignment.max(4),
            usage,
        )
    }

    /// Return the autodiff device backed by the shared engine context.
    #[cfg(feature = "training")]
    pub fn training_device(&self) -> Device {
        self.device.clone().autodiff()
    }

    /// Adopt an acquired descriptor as a read-only inference tensor without copying.
    pub fn adopt_f32(&self, descriptor: AcquiredExternalTensorDescriptor) -> Result<DynTensor> {
        self.ensure_context(&descriptor)?;
        match self.compiler {
            #[cfg(feature = "wgpu")]
            Compiler::Automatic => {
                let (tensor, shape) =
                    register_external::<AutoCompiler>(descriptor, self.runtime_device.clone())?;
                into_dyn_inference::<burn::backend::Wgpu>(tensor, &shape)
            }
            #[cfg(feature = "vulkan")]
            Compiler::Vulkan => {
                let (tensor, shape) =
                    register_external::<SpirvCompiler>(descriptor, self.runtime_device.clone())?;
                into_dyn_inference::<burn::backend::Vulkan>(tensor, &shape)
            }
        }
    }

    /// Adopt an acquired descriptor as a read-only tensor on the autodiff backend.
    ///
    /// The external input does not require its own gradient, but operations that combine it with
    /// trainable parameters are recorded normally.
    #[cfg(feature = "training")]
    pub fn adopt_f32_training(
        &self,
        descriptor: AcquiredExternalTensorDescriptor,
    ) -> Result<DynTensor> {
        self.ensure_context(&descriptor)?;
        match self.compiler {
            #[cfg(feature = "wgpu")]
            Compiler::Automatic => {
                let (tensor, shape) =
                    register_external::<AutoCompiler>(descriptor, self.runtime_device.clone())?;
                into_dyn_training::<burn::backend::Wgpu>(tensor, &shape)
            }
            #[cfg(feature = "vulkan")]
            Compiler::Vulkan => {
                let (tensor, shape) =
                    register_external::<SpirvCompiler>(descriptor, self.runtime_device.clone())?;
                into_dyn_training::<burn::backend::Vulkan>(tensor, &shape)
            }
        }
    }

    fn ensure_context(&self, descriptor: &AcquiredExternalTensorDescriptor) -> Result<()> {
        if descriptor.belongs_to(&self.capability) {
            return Ok(());
        }
        Err(external_error(
            "external tensor was validated for a different WGPU context",
        ))
    }
}

fn existing_device_kind(device: &WgpuDevice) -> Result<DeviceKind> {
    match device {
        WgpuDevice::Existing(id) => Ok(DeviceKind::Existing(*id)),
        other => Err(external_error(format!(
            "CubeCL external-device initialization returned unexpected device {other:?}"
        ))),
    }
}

fn register_external<C: WgpuCompiler>(
    descriptor: AcquiredExternalTensorDescriptor,
    device: WgpuDevice,
) -> Result<(burn::backend::wgpu::CubeTensor<WgpuRuntime<C>>, Vec<usize>)> {
    let buffer = descriptor
        .buffer()
        .resource::<ExternalWgpuBuffer>()
        .ok_or_else(|| {
            external_error("external buffer was not created by ExternalWgpuContext::lease_buffer")
        })?;
    let shape = descriptor.shape().to_vec();
    let client = WgpuRuntime::<C>::client(&device);
    let resource = WgpuResource::from_external_buffer_with_retention(
        buffer.buffer.clone(),
        descriptor.offset_bytes(),
        descriptor.length_bytes(),
        Arc::new(descriptor.retain()),
    );
    let handle = client
        .register_external(resource)
        .map_err(|error| external_error(format!("CubeCL rejected external buffer: {error}")))?;
    let tensor = burn::backend::wgpu::CubeTensor::new_contiguous(
        client,
        device,
        Shape::from(shape.clone()),
        handle,
        descriptor.dtype(),
    );
    Ok((tensor, shape))
}

fn into_dyn_inference<B>(tensor: B::FloatTensorPrimitive, shape: &[usize]) -> Result<DynTensor>
where
    B: burn::backend::Backend,
    burn::backend::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    match shape.len() {
        1 => Ok(DynTensor::R1(Tensor::<1>::from_primitive::<B>(tensor))),
        2 => Ok(DynTensor::R2(Tensor::<2>::from_primitive::<B>(tensor))),
        3 => Ok(DynTensor::R3(Tensor::<3>::from_primitive::<B>(tensor))),
        4 => Ok(DynTensor::R4(Tensor::<4>::from_primitive::<B>(tensor))),
        5 => Ok(DynTensor::R5(Tensor::<5>::from_primitive::<B>(tensor))),
        6 => Ok(DynTensor::R6(Tensor::<6>::from_primitive::<B>(tensor))),
        rank => Err(TynxError::RankOverflow {
            rank,
            max: crate::MAX_RANK,
        }),
    }
}

#[cfg(feature = "training")]
fn into_dyn_training<B>(tensor: B::FloatTensorPrimitive, shape: &[usize]) -> Result<DynTensor>
where
    B: burn::backend::Backend,
    Autodiff<B>: burn::backend::Backend,
    burn::backend::DispatchTensor: burn::backend::DispatchKindConversion<Autodiff<B>>,
{
    let tensor = <Autodiff<B> as AutodiffBackend>::from_inner(tensor);
    match shape.len() {
        1 => Ok(DynTensor::R1(Tensor::<1>::from_primitive::<Autodiff<B>>(
            tensor,
        ))),
        2 => Ok(DynTensor::R2(Tensor::<2>::from_primitive::<Autodiff<B>>(
            tensor,
        ))),
        3 => Ok(DynTensor::R3(Tensor::<3>::from_primitive::<Autodiff<B>>(
            tensor,
        ))),
        4 => Ok(DynTensor::R4(Tensor::<4>::from_primitive::<Autodiff<B>>(
            tensor,
        ))),
        5 => Ok(DynTensor::R5(Tensor::<5>::from_primitive::<Autodiff<B>>(
            tensor,
        ))),
        6 => Ok(DynTensor::R6(Tensor::<6>::from_primitive::<Autodiff<B>>(
            tensor,
        ))),
        rank => Err(TynxError::RankOverflow {
            rank,
            max: crate::MAX_RANK,
        }),
    }
}

fn external_error(message: impl Into<String>) -> TynxError {
    TynxError::ExternalTensor(message.into())
}

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use std::sync::{
        Weak,
        atomic::{AtomicUsize, Ordering},
    };

    use burn::backend::Wgpu;

    use super::*;
    use crate::{
        DType, ExternalAccess, ExternalSubmission, ExternalTensorDescriptor, SubmissionToken,
    };

    #[derive(Debug)]
    struct Owner;

    #[derive(Debug)]
    struct RecyclingOwner {
        releases: Arc<AtomicUsize>,
    }

    impl Drop for RecyclingOwner {
        fn drop(&mut self) {
            self.releases.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct Submission {
        calls: Arc<AtomicUsize>,
    }

    impl ExternalSubmission for Submission {
        fn ensure_visible(&self) -> Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn noop_setup() -> WgpuSetup {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::NOOP,
            backend_options: wgpu::BackendOptions {
                noop: wgpu::NoopBackendOptions {
                    enable: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let backend = adapter.get_info().backend;
        WgpuSetup {
            instance,
            adapter,
            device,
            queue,
            backend,
        }
    }

    fn executing_setup() -> Option<WgpuSetup> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("Tynx external WGPU execution test"),
                required_features: adapter
                    .features()
                    .difference(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            }),
        )
        .ok()?;
        let backend = adapter.get_info().backend;
        Some(WgpuSetup {
            instance,
            adapter,
            device,
            queue,
            backend,
        })
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    fn descriptor(
        context: &ExternalWgpuContext,
        buffer: wgpu::Buffer,
    ) -> (
        AcquiredExternalTensorDescriptor,
        wgpu::Buffer,
        Weak<Owner>,
        Arc<AtomicUsize>,
    ) {
        let owner = Arc::new(Owner);
        let owner_weak = Arc::downgrade(&owner);
        let lease = context.lease_buffer(buffer.clone(), owner).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let capability = context.capability();
        let token = SubmissionToken::new(
            capability.clone(),
            Submission {
                calls: calls.clone(),
            },
        );
        let descriptor = ExternalTensorDescriptor::new(
            capability.clone(),
            lease,
            0,
            16,
            vec![4],
            DType::F32,
            vec![1],
            ExternalAccess::ReadOnly,
            token,
        )
        .validate_read_only_dense(&capability)
        .unwrap()
        .acquire()
        .unwrap();
        (descriptor, buffer, owner_weak, calls)
    }

    fn context_and_buffer() -> (ExternalWgpuContext, wgpu::Buffer) {
        let setup = noop_setup();
        let buffer = setup.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tynx external adoption test"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (
            ExternalWgpuContext::from_wgpu_setup(setup, RuntimeOptions::default()).unwrap(),
            buffer,
        )
    }

    fn executing_context_and_buffer() -> Option<(ExternalWgpuContext, wgpu::Buffer, wgpu::Queue)> {
        let setup = executing_setup()?;
        let queue = setup.queue.clone();
        let buffer = setup.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tynx external execution test"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let context =
            ExternalWgpuContext::from_wgpu_setup(setup, RuntimeOptions::default()).unwrap();
        Some((context, buffer, queue))
    }

    #[test]
    fn adopts_the_original_buffer_and_retains_its_logical_owner() {
        let (context, buffer) = context_and_buffer();
        let (descriptor, buffer, owner, calls) = descriptor(&context, buffer);

        let adopted = context.adopt_f32(descriptor).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(adopted.device(), context.device());
        assert!(owner.upgrade().is_some());
        let primitive = match adopted {
            DynTensor::R1(tensor) => tensor.try_into_primitive::<Wgpu>().unwrap(),
            _ => panic!("expected rank-1 external tensor"),
        };
        let resource = primitive
            .client
            .get_resource(primitive.handle.clone())
            .unwrap();
        assert_eq!(resource.resource().buffer, buffer);
        assert_eq!(resource.resource().offset, 0);
        assert_eq!(resource.resource().size, 16);
    }

    #[test]
    fn executes_a_reduction_over_the_original_buffer_contents() {
        let Some((context, buffer, queue)) = executing_context_and_buffer() else {
            eprintln!("skipping external WGPU execution test: no executing adapter");
            return;
        };
        queue.write_buffer(&buffer, 0, &f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
        let (descriptor, _, _, calls) = descriptor(&context, buffer);

        let mean = context
            .adopt_f32(descriptor)
            .unwrap()
            .mean_dims(&[0])
            .reshape(vec![1])
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(mean, [2.5]);
    }

    #[cfg(feature = "training")]
    #[test]
    fn adopts_external_inputs_on_the_autodiff_backend_without_requiring_input_gradients() {
        let (context, buffer) = context_and_buffer();
        let (descriptor, _, owner, _) = descriptor(&context, buffer);

        let adopted = context.adopt_f32_training(descriptor).unwrap();

        assert_eq!(adopted.device(), context.training_device());
        assert!(!adopted.is_require_grad());
        assert!(owner.upgrade().is_some());
        match adopted {
            DynTensor::R1(tensor) => {
                tensor.try_into_primitive::<Autodiff<Wgpu>>().unwrap();
            }
            _ => panic!("expected rank-1 external tensor"),
        }
    }

    #[cfg(feature = "training")]
    #[test]
    fn retains_the_engine_slot_through_external_forward_and_backward() {
        let Some((context, buffer, queue)) = executing_context_and_buffer() else {
            eprintln!("skipping external WGPU autodiff test: no executing adapter");
            return;
        };
        queue.write_buffer(&buffer, 0, &f32_bytes(&[1.0, 2.0, 3.0, 4.0]));

        let releases = Arc::new(AtomicUsize::new(0));
        let owner = Arc::new(RecyclingOwner {
            releases: releases.clone(),
        });
        let owner_weak = Arc::downgrade(&owner);
        let lease = context.lease_buffer(buffer, owner).unwrap();
        let capability = context.capability();
        let calls = Arc::new(AtomicUsize::new(0));
        let descriptor = ExternalTensorDescriptor::new(
            capability.clone(),
            lease,
            0,
            16,
            vec![4],
            DType::F32,
            vec![1],
            ExternalAccess::ReadOnly,
            SubmissionToken::new(
                capability.clone(),
                Submission {
                    calls: calls.clone(),
                },
            ),
        )
        .validate_read_only_dense(&capability)
        .unwrap()
        .acquire()
        .unwrap();

        let input = context.adopt_f32_training(descriptor).unwrap();
        let parameter = DynTensor::from_data(
            crate::TensorData::new(vec![2.0_f32, 2.0, 2.0, 2.0], [4]),
            1,
            &context.training_device(),
        )
        .unwrap()
        .require_grad();
        let loss = input
            .clone()
            .mul_broadcast(parameter.clone())
            .unwrap()
            .mean_dims(&[0])
            .reshape(vec![1])
            .unwrap();
        drop(input);

        assert!(owner_weak.upgrade().is_some());
        assert_eq!(releases.load(Ordering::SeqCst), 0);
        let gradients = loss.backward();
        crate::synchronize(&context.training_device()).unwrap();
        assert_eq!(
            parameter
                .grad(&gradients)
                .unwrap()
                .into_data()
                .iter::<f32>()
                .collect::<Vec<_>>(),
            [0.25, 0.5, 0.75, 1.0]
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(owner_weak.upgrade().is_some());
        assert_eq!(releases.load(Ordering::SeqCst), 0);

        drop(gradients);
        drop(loss);
        drop(parameter);
        crate::synchronize(&context.training_device()).unwrap();
        assert!(owner_weak.upgrade().is_none());
        assert_eq!(releases.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn rejects_a_descriptor_validated_for_another_context() {
        let (context, buffer) = context_and_buffer();
        let (descriptor, _, _, _) = descriptor(&context, buffer);
        let wrong_context = ExternalWgpuContext {
            capability: DeviceContextCapability::new(),
            ..context.clone()
        };

        let error = wrong_context.adopt_f32(descriptor).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("validated for a different WGPU context")
        );
    }
}
