//! Process-default device selection with GPU-to-CPU fallback.

use std::sync::Mutex;

use crate::error::{Result, TynxError};

static LAST_DEVICE_ERROR: Mutex<Option<String>> = Mutex::new(None);

#[cfg(any(
    test,
    all(
        any(feature = "wgpu", feature = "vulkan"),
        feature = "flex",
        not(target_family = "wasm")
    )
))]
fn record_device_error(message: String) {
    if let Ok(mut error) = LAST_DEVICE_ERROR.lock() {
        // Preserve the first failure: later validation errors may only be consequences of it.
        if error.is_none() {
            *error = Some(message);
        }
    }
}

/// Take the first pending asynchronous device error, if one was reported.
///
/// Reading clears the stored error. This is primarily useful for GPU memory pressure, which wgpu
/// can report from a device thread after the Python or Rust call that queued the allocation has
/// returned.
pub fn take_device_error() -> Option<String> {
    LAST_DEVICE_ERROR.lock().ok()?.take()
}

/// Wait for queued work and report any pending asynchronous device error.
pub fn synchronize(device: &burn::tensor::Device) -> Result<()> {
    let synchronized = device.sync();
    if let Some(error) = take_device_error() {
        return Err(TynxError::AsynchronousDevice(error));
    }
    synchronized.map_err(|error| TynxError::DeviceSynchronization(error.to_string()))
}

#[cfg(all(
    any(feature = "wgpu", feature = "vulkan"),
    feature = "flex",
    not(target_family = "wasm")
))]
mod probe {
    use std::sync::{Arc, OnceLock};

    use super::record_device_error;

    static DEFAULT_DEVICE_USABLE: OnceLock<bool> = OnceLock::new();
    static DEFAULT_BUFFER_SIZE_LIMIT: OnceLock<u64> = OnceLock::new();
    #[cfg(all(test, feature = "vulkan"))]
    static CONFIGURED_SETUP: OnceLock<burn::backend::wgpu::WgpuSetup> = OnceLock::new();

    pub(super) fn default_device_usable() -> bool {
        *DEFAULT_DEVICE_USABLE.get_or_init(|| {
            let usable = adapter_available();
            if !usable {
                let message =
                    "tynx: no usable GPU device found; falling back to the Flex CPU backend";
                log::warn!("{message}");
                // Also print directly: Python and CLI users rarely have a log backend
                // installed, and silent CPU fallback would misrepresent performance.
                eprintln!("{message}");
            }
            usable
        })
    }

    pub(super) fn configure_default_device() {
        static CONFIGURED: OnceLock<()> = OnceLock::new();
        CONFIGURED.get_or_init(|| {
            let device = burn::backend::wgpu::WgpuDevice::default();
            #[cfg(feature = "vulkan")]
            let setup = burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
                &device,
                Default::default(),
            );
            #[cfg(all(not(feature = "vulkan"), feature = "wgpu"))]
            let setup = burn::backend::wgpu::init_setup::<
                burn::backend::wgpu::graphics::AutoGraphicsApi,
            >(&device, Default::default());

            let limits = setup.device.limits();
            let buffer_size_limit = limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size);
            let _ = DEFAULT_BUFFER_SIZE_LIMIT.set(buffer_size_limit);

            setup.device.on_uncaptured_error(Arc::new(|error| {
                record_device_error(error.to_string());
                log::error!("asynchronous wgpu device error: {error}");
            }));

            #[cfg(all(test, feature = "vulkan"))]
            let _ = CONFIGURED_SETUP.set(setup);
        });
    }

    pub(super) fn default_buffer_size_limit() -> Option<u64> {
        DEFAULT_BUFFER_SIZE_LIMIT.get().copied()
    }

    #[cfg(all(test, feature = "vulkan"))]
    pub(super) fn configured_setup() -> &'static burn::backend::wgpu::WgpuSetup {
        CONFIGURED_SETUP
            .get()
            .expect("default_device must initialize the CubeCL wgpu setup first")
    }

    /// Ask wgpu for the adapter the GPU device server would request, without entering the
    /// server's infallible init path. Mirrors cubecl's `AutoGraphicsApi` backend choice
    /// (Metal on macOS, Vulkan elsewhere) so the probe and the real init agree.
    fn adapter_available() -> bool {
        let backends = if cfg!(target_os = "macos") {
            wgpu::Backends::METAL
        } else {
            wgpu::Backends::VULKAN
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .is_ok()
    }
}

/// The process-default execution device, verified usable.
///
/// With a GPU backend compiled in, [`Device::default`](burn::tensor::Device::default) prefers
/// the GPU but aborts deep in the device server when no adapter exists (headless hosts,
/// containers, CI). This asks wgpu for the same adapter once per process and falls back to the
/// Flex CPU device with a warning instead of crashing.
pub fn default_device() -> burn::tensor::Device {
    #[cfg(all(
        any(feature = "wgpu", feature = "vulkan"),
        feature = "flex",
        not(target_family = "wasm")
    ))]
    {
        if !probe::default_device_usable() {
            return burn::tensor::Device::flex();
        }
        probe::configure_default_device();
    }
    burn::tensor::Device::default()
}

/// Return the maximum single storage allocation for the selected default GPU.
///
/// CPU devices and builds without the WGPU-backed default return `None`. This limit catches
/// statically impossible allocations before CubeCL dispatch; it is not an estimate of currently
/// available device memory.
pub fn allocation_size_limit(device: &burn::tensor::Device) -> Option<u64> {
    #[cfg(all(
        any(feature = "wgpu", feature = "vulkan"),
        feature = "flex",
        not(target_family = "wasm")
    ))]
    {
        let limit = probe::default_buffer_size_limit()?;
        if device == &burn::tensor::Device::default() {
            return Some(limit);
        }
    }
    let _ = device;
    None
}

#[cfg(test)]
mod tests {
    use super::{default_device, record_device_error, synchronize, take_device_error};
    use crate::TynxError;

    #[cfg(any(feature = "wgpu", feature = "vulkan"))]
    trait FusionBackendMarker {}

    #[cfg(any(feature = "wgpu", feature = "vulkan"))]
    impl<B: burn_fusion::FusionBackend> FusionBackendMarker for burn_fusion::Fusion<B> {}

    #[cfg(any(feature = "wgpu", feature = "vulkan"))]
    fn assert_fusion_backend<B: FusionBackendMarker>() {}

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_backend_is_fusion_wrapped() {
        assert_fusion_backend::<burn::backend::wgpu::Wgpu>();
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_backend_is_fusion_wrapped() {
        assert_fusion_backend::<burn::backend::wgpu::Vulkan>();
    }

    #[test]
    fn default_device_never_panics() {
        let device = default_device();
        let tensor = burn::tensor::Tensor::<1>::from_data(
            burn::tensor::TensorData::new(vec![2.0_f32], [1]),
            (&device, burn::tensor::DType::F32),
        );
        assert_eq!(tensor.into_data().to_vec::<f32>().unwrap(), vec![2.0]);
    }

    #[test]
    fn asynchronous_device_errors_can_be_observed_once() {
        let _ = take_device_error();
        record_device_error("Out of Memory".to_string());
        record_device_error("cascading validation error".to_string());

        assert_eq!(
            synchronize(&default_device()),
            Err(TynxError::AsynchronousDevice("Out of Memory".to_string()))
        );
        assert_eq!(take_device_error(), None);
    }

    #[cfg(all(feature = "vulkan", not(target_family = "wasm")))]
    #[test]
    fn allocation_limit_matches_configured_vulkan_storage_limits() {
        let device = default_device();
        let limits = super::probe::configured_setup().device.limits();
        let expected = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);
        assert_eq!(super::allocation_size_limit(&device), Some(expected));
    }

    #[cfg(all(feature = "vulkan", not(target_family = "wasm")))]
    #[test]
    #[ignore = "requires a dedicated Vulkan GPU and deliberately exhausts its memory"]
    fn real_vulkan_oom_is_reported_and_the_device_recovers() {
        const ALLOCATION_BYTES: u64 = 256 * 1024 * 1024;
        const MAX_ALLOCATIONS: usize = 512;

        assert!(
            super::probe::default_device_usable(),
            "the OOM acceptance test requires a usable Vulkan GPU"
        );
        let burn_device = default_device();
        let setup = super::probe::configured_setup();
        let adapter = setup.adapter.get_info();
        assert_eq!(
            adapter.device_type,
            wgpu::DeviceType::DiscreteGpu,
            "refusing to exhaust memory on non-discrete adapter {:?}",
            adapter.name
        );

        let _ = take_device_error();
        let allocation_size = ALLOCATION_BYTES.min(setup.device.limits().max_buffer_size);
        let mut buffers = Vec::new();
        for _ in 0..MAX_ALLOCATIONS {
            buffers.push(setup.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tynx forced OOM acceptance allocation"),
                size: allocation_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
            setup.device.poll(wgpu::PollType::Poll).unwrap();
            if super::LAST_DEVICE_ERROR
                .lock()
                .expect("device error mutex must not be poisoned")
                .is_some()
            {
                break;
            }
        }

        assert_eq!(
            synchronize(&burn_device),
            Err(TynxError::AsynchronousDevice("Out of Memory".to_string()))
        );

        drop(buffers);
        setup
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let tensor = burn::tensor::Tensor::<1>::from_data(
            burn::tensor::TensorData::new(vec![2.0_f32], [1]),
            (&burn_device, burn::tensor::DType::F32),
        );
        assert_eq!(tensor.into_data().to_vec::<f32>().unwrap(), vec![2.0]);
        assert_eq!(take_device_error(), None);
    }
}
