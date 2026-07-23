#![cfg(feature = "training")]

use std::sync::{
    Arc, Weak,
    atomic::{AtomicUsize, Ordering},
};

use tynx::{
    DType, Device, DeviceContextCapability, DynTensor, ExternalAccess, ExternalBufferLease,
    ExternalBufferUsage, ExternalSubmission, ExternalTensorDescriptor, ExternalTensorRetention,
    Result, SubmissionToken, TensorData, synchronize,
};

#[derive(Debug)]
struct MockBuffer {
    drops: Arc<AtomicUsize>,
}

impl Drop for MockBuffer {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct MockSubmission {
    calls: Arc<AtomicUsize>,
}

impl ExternalSubmission for MockSubmission {
    fn ensure_visible(&self) -> Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct MockBorrowedTensor {
    tensor: DynTensor,
    retention: ExternalTensorRetention,
}

impl MockBorrowedTensor {
    fn forward(&self) -> MockAutodiffTape {
        let squared = self
            .tensor
            .clone()
            .mul_broadcast(self.tensor.clone())
            .unwrap();
        let loss = squared.mean_dims(&[0]).reshape(vec![1]).unwrap();
        MockAutodiffTape {
            loss,
            _retention: self.retention.retain_for_tape(),
        }
    }
}

struct MockAutodiffTape {
    loss: DynTensor,
    _retention: ExternalTensorRetention,
}

impl MockAutodiffTape {
    fn backward(&self) -> tynx::Gradients {
        self.loss.clone().backward()
    }
}

#[test]
fn lease_survives_forward_backward_synchronization_and_input_drop() {
    let context = DeviceContextCapability::new();
    let drops = Arc::new(AtomicUsize::new(0));
    let resource = Arc::new(MockBuffer {
        drops: drops.clone(),
    });
    let weak_resource: Weak<MockBuffer> = Arc::downgrade(&resource);
    let buffer = ExternalBufferLease::new(
        context.clone(),
        resource.clone(),
        64,
        4,
        ExternalBufferUsage::Storage,
    )
    .unwrap();
    drop(resource);

    let submissions = Arc::new(AtomicUsize::new(0));
    let token = SubmissionToken::new(
        context.clone(),
        MockSubmission {
            calls: submissions.clone(),
        },
    );
    let acquired = ExternalTensorDescriptor::new(
        context.clone(),
        buffer,
        0,
        16,
        vec![4],
        DType::F32,
        vec![1],
        ExternalAccess::ReadOnly,
        token,
    )
    .validate_read_only_dense(&context)
    .unwrap()
    .acquire()
    .unwrap();
    assert_eq!(submissions.load(Ordering::SeqCst), 1);

    let device = Device::autodiff(Device::default());
    let leaf = DynTensor::from_data(
        TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [4]),
        1,
        &device,
    )
    .unwrap()
    .require_grad();
    let borrowed = MockBorrowedTensor {
        tensor: leaf.clone(),
        retention: acquired.retain(),
    };
    drop(acquired);
    let tape = borrowed.forward();
    drop(borrowed);

    assert!(weak_resource.upgrade().is_some());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    synchronize(&device).unwrap();
    assert!(weak_resource.upgrade().is_some());

    let gradients = tape.backward();
    synchronize(&device).unwrap();
    assert_eq!(
        leaf.grad(&gradients)
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>(),
        [0.5, 1.0, 1.5, 2.0]
    );
    assert!(weak_resource.upgrade().is_some());
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    drop(gradients);
    drop(tape);
    assert!(weak_resource.upgrade().is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
