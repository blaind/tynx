//! Process-default device selection with GPU-to-CPU fallback.

use std::sync::Mutex;

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

#[cfg(all(
    any(feature = "wgpu", feature = "vulkan"),
    feature = "flex",
    not(target_family = "wasm")
))]
mod probe {
    use std::sync::{Arc, OnceLock};

    use super::record_device_error;

    static DEFAULT_DEVICE_USABLE: OnceLock<bool> = OnceLock::new();

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

            setup.device.on_uncaptured_error(Arc::new(|error| {
                record_device_error(error.to_string());
                log::error!("asynchronous wgpu device error: {error}");
            }));
        });
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
            ..wgpu::RequestAdapterOptions::default()
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

#[cfg(test)]
mod tests {
    use super::{default_device, record_device_error, take_device_error};

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

        assert_eq!(take_device_error().as_deref(), Some("Out of Memory"));
        assert_eq!(take_device_error(), None);
    }
}
