use std::sync::{Arc, Mutex, Weak};

use _tynx::wrap_external_tensor;
use pyo3::{prelude::*, types::PyDict};
use tynx_core::{
    DType, ExternalAccess, ExternalSubmission, ExternalTensorDescriptor, ExternalWgpuContext,
    Result, SubmissionToken,
};

#[derive(Debug)]
struct AllocationOwner;

struct Submitted;

static PYTHON_TEST: Mutex<()> = Mutex::new(());

impl ExternalSubmission for Submitted {
    fn ensure_visible(&self) -> Result<()> {
        Ok(())
    }
}

fn noop_context() -> (ExternalWgpuContext, wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions { enable: true },
            ..Default::default()
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .unwrap();
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let context =
        ExternalWgpuContext::from_wgpu_handles(instance, adapter, device.clone(), queue.clone())
            .unwrap();
    (context, device, queue)
}

fn executing_context() -> Option<(ExternalWgpuContext, wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(
        adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Tynx Python external embedding test"),
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
    let context =
        ExternalWgpuContext::from_wgpu_handles(instance, adapter, device.clone(), queue.clone())
            .ok()?;
    Some((context, device, queue))
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn descriptor(
    context: &ExternalWgpuContext,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (
    tynx_core::AcquiredExternalTensorDescriptor,
    Weak<AllocationOwner>,
) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Tynx Python external tensor"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &f32_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let owner = Arc::new(AllocationOwner);
    let owner_weak = Arc::downgrade(&owner);
    let lease = context.lease_buffer(buffer, owner).unwrap();
    let capability = context.capability();
    let descriptor = ExternalTensorDescriptor::new(
        capability.clone(),
        lease,
        0,
        16,
        vec![4],
        DType::F32,
        vec![1],
        ExternalAccess::ReadOnly,
        SubmissionToken::new(capability.clone(), Submitted),
    )
    .validate_read_only_dense(&capability)
    .unwrap()
    .acquire()
    .unwrap();
    (descriptor, owner_weak)
}

#[test]
fn wraps_an_adopted_inference_tensor_without_changing_its_storage() {
    let _serial = PYTHON_TEST.lock().unwrap();
    Python::initialize();
    let (context, device, queue) = noop_context();
    let (descriptor, owner) = descriptor(&context, &device, &queue);
    let external = context.adopt_f32(descriptor).unwrap();

    let tensor = Python::attach(|py| -> PyResult<_> {
        let tensor = wrap_external_tensor(py, external)?;
        let value = tensor.bind(py);
        assert_eq!(value.get_type().name()?.extract::<String>()?, "Tensor");
        assert_eq!(value.getattr("shape")?.extract::<Vec<usize>>()?, [4]);
        assert_eq!(value.getattr("dtype")?.extract::<String>()?, "float32");
        Ok(tensor)
    })
    .unwrap();

    assert!(owner.upgrade().is_some());
    Python::attach(|_| drop(tensor));
    context.reclaim_unused_external_buffers().unwrap();
}

#[test]
fn executes_python_inference_over_an_adopted_tensor() {
    let _serial = PYTHON_TEST.lock().unwrap();
    Python::initialize();
    let Some((context, device, queue)) = executing_context() else {
        eprintln!("skipping Python external inference test: no executing WGPU adapter");
        return;
    };
    let (descriptor, owner) = descriptor(&context, &device, &queue);
    let external = context.adopt_f32(descriptor).unwrap();

    let tensor = Python::attach(|py| -> PyResult<_> {
        let tensor = wrap_external_tensor(py, external)?;
        let value = tensor.bind(py);
        assert_eq!(
            value
                .call_method0("mean")?
                .call_method0("item")?
                .extract::<f64>()?,
            2.5
        );
        Ok(tensor)
    })
    .unwrap();

    assert!(owner.upgrade().is_some());
    Python::attach(|_| drop(tensor));
    context.reclaim_unused_external_buffers().unwrap();
    assert!(owner.upgrade().is_none());
}

#[test]
fn wraps_an_adopted_autodiff_tensor_as_a_python_constant() {
    let _serial = PYTHON_TEST.lock().unwrap();
    Python::initialize();
    let (context, device, queue) = noop_context();
    let (descriptor, owner) = descriptor(&context, &device, &queue);
    let external = context.adopt_f32_training(descriptor).unwrap();

    let tensor = Python::attach(|py| -> PyResult<_> {
        let tensor = wrap_external_tensor(py, external)?;
        let value = tensor.bind(py);
        assert_eq!(value.get_type().name()?.extract::<String>()?, "Tensor");
        assert_eq!(value.getattr("shape")?.extract::<Vec<usize>>()?, [4]);
        assert!(!value.getattr("requires_grad")?.extract::<bool>()?);
        Ok(tensor)
    })
    .unwrap();

    assert!(owner.upgrade().is_some());
    Python::attach(|_| drop(tensor));
    context.reclaim_unused_external_buffers().unwrap();
}

#[test]
fn pending_python_tensor_is_drop_only_after_context_retirement() {
    let _serial = PYTHON_TEST.lock().unwrap();
    Python::initialize();
    let (context, device, queue) = noop_context();
    let (descriptor, owner) = descriptor(&context, &device, &queue);
    let external = context.adopt_f32_training(descriptor).unwrap();
    let tensor = Python::attach(|py| wrap_external_tensor(py, external)).unwrap();

    context.retire();
    assert!(!context.capability().is_active());
    Python::attach(|py| {
        assert_eq!(
            tensor
                .bind(py)
                .getattr("shape")
                .unwrap()
                .extract::<Vec<usize>>()
                .unwrap(),
            [4]
        );
    });
    assert!(owner.upgrade().is_some());

    let rejected = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rejected retired-context allocation"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    assert!(
        context
            .lease_buffer(rejected, Arc::new(AllocationOwner))
            .unwrap_err()
            .to_string()
            .contains("generation is retired")
    );

    Python::attach(|_| drop(tensor));
    context.reclaim_unused_external_buffers().unwrap();
}

#[test]
fn backward_retains_an_external_input_after_its_python_wrapper_drops() {
    let _serial = PYTHON_TEST.lock().unwrap();
    Python::initialize();
    let Some((context, device, queue)) = executing_context() else {
        eprintln!("skipping Python external autodiff test: no executing WGPU adapter");
        return;
    };
    let (descriptor, owner) = descriptor(&context, &device, &queue);
    let external = context.adopt_f32_training(descriptor).unwrap();

    let (input, parameter, loss) = Python::attach(|py| -> PyResult<_> {
        let input = wrap_external_tensor(py, external)?;
        let value = input.bind(py);
        let kwargs = PyDict::new(py);
        kwargs.set_item("device", value.getattr("device")?)?;
        kwargs.set_item("requires_grad", true)?;
        let parameter = value
            .get_type()
            .call((vec![2.0_f32; 4],), Some(&kwargs))?
            .unbind();
        let loss = value
            .call_method1("__mul__", (parameter.bind(py),))?
            .call_method0("mean")?
            .unbind();
        Ok((input, parameter, loss))
    })
    .unwrap();

    Python::attach(|_| drop(input));
    context.reclaim_unused_external_buffers().unwrap();
    assert!(
        owner.upgrade().is_some(),
        "the autodiff tape released the external allocation before backward"
    );

    Python::attach(|py| -> PyResult<()> {
        loss.bind(py).call_method0("backward")?;
        assert_eq!(
            parameter
                .bind(py)
                .getattr("grad")?
                .call_method0("tolist")?
                .extract::<Vec<f64>>()?,
            [0.25, 0.5, 0.75, 1.0]
        );
        Ok(())
    })
    .unwrap();

    Python::attach(|_| {
        drop(loss);
        drop(parameter);
    });
    context.reclaim_unused_external_buffers().unwrap();
    assert!(owner.upgrade().is_none());
}
