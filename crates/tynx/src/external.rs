//! Binding-neutral contracts for adopting externally owned device tensors.
//!
//! Validation is backend-neutral and never stages or copies data. Backend adapters consume an
//! acquired descriptor only after checking its opaque device/queue capability, then retain its
//! lease through asynchronous work and any autodiff tape that may read the forward activation.

use std::{
    any::Any,
    fmt::{Debug, Formatter},
    sync::Arc,
};

use crate::{DType, MAX_RANK, Result, TynxError};

/// Opaque identity for one engine-owned device and queue context.
///
/// Equality is capability identity, not a caller-provided name or raw handle. A trusted engine
/// integration creates one capability when it initializes Tynx and clones it into every external
/// buffer and submission token associated with that device and queue.
#[derive(Clone)]
pub struct DeviceContextCapability {
    identity: Arc<()>,
}

impl DeviceContextCapability {
    /// Create a fresh opaque context capability.
    pub fn new() -> Self {
        Self {
            identity: Arc::new(()),
        }
    }

    /// Return whether two capabilities identify the same device and queue context.
    pub fn same_context(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.identity, &other.identity)
    }
}

impl Default for DeviceContextCapability {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for DeviceContextCapability {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceContextCapability")
            .finish_non_exhaustive()
    }
}

/// Access requested by an external tensor view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalAccess {
    /// Tynx may only read the external allocation.
    ReadOnly,
    /// Tynx may read and write the external allocation.
    ReadWrite,
}

/// Relevant usage declared by the external allocation owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalBufferUsage {
    /// The allocation may be bound as device storage.
    Storage,
    /// The allocation is not valid device storage.
    NonStorage,
}

/// Retained ownership of an externally allocated device buffer.
///
/// The resource is type-erased at the Tynx boundary and can be recovered by the trusted backend
/// adapter with [`Self::resource`]. Cloning this value retains the engine allocation.
#[derive(Clone)]
pub struct ExternalBufferLease {
    context: DeviceContextCapability,
    resource: Arc<dyn Any + Send + Sync>,
    byte_length: u64,
    alignment: u64,
    usage: ExternalBufferUsage,
}

impl ExternalBufferLease {
    /// Wrap an externally owned allocation and its immutable storage metadata.
    pub fn new<R>(
        context: DeviceContextCapability,
        resource: Arc<R>,
        byte_length: u64,
        alignment: u64,
        usage: ExternalBufferUsage,
    ) -> Result<Self>
    where
        R: Any + Send + Sync,
    {
        if byte_length == 0 {
            return Err(external_error(
                "external buffer byte length must be positive",
            ));
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(external_error(format!(
                "external buffer alignment must be a positive power of two, got {alignment}"
            )));
        }
        Ok(Self {
            context,
            resource,
            byte_length,
            alignment,
            usage,
        })
    }

    /// Recover a typed clone of the trusted integration's resource.
    pub fn resource<R>(&self) -> Option<Arc<R>>
    where
        R: Any + Send + Sync,
    {
        self.resource.clone().downcast().ok()
    }

    /// Return the allocation's total byte length.
    pub fn byte_length(&self) -> u64 {
        self.byte_length
    }

    /// Return the required byte-offset alignment.
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Return the allocation usage declared by its owner.
    pub fn usage(&self) -> ExternalBufferUsage {
        self.usage
    }
}

impl Debug for ExternalBufferLease {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExternalBufferLease")
            .field("context", &self.context)
            .field("byte_length", &self.byte_length)
            .field("alignment", &self.alignment)
            .field("usage", &self.usage)
            .finish_non_exhaustive()
    }
}

/// Producer ordering action carried by an opaque submission token.
pub trait ExternalSubmission: Send + Sync {
    /// Make the producer's writes visible before the consumer queues Tynx work.
    ///
    /// An engine integration may encode a queue dependency rather than blocking the host.
    fn ensure_visible(&self) -> Result<()>;
}

/// Opaque producer submission token tied to one device and queue context.
#[derive(Clone)]
pub struct SubmissionToken {
    context: DeviceContextCapability,
    submission: Arc<dyn ExternalSubmission>,
}

impl SubmissionToken {
    /// Wrap a producer ordering action from a trusted engine integration.
    pub fn new<S>(context: DeviceContextCapability, submission: S) -> Self
    where
        S: ExternalSubmission + 'static,
    {
        Self {
            context,
            submission: Arc::new(submission),
        }
    }

    fn ensure_visible(&self) -> Result<()> {
        self.submission.ensure_visible()
    }
}

impl Debug for SubmissionToken {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SubmissionToken")
            .field("context", &self.context)
            .finish_non_exhaustive()
    }
}

/// Description of one externally owned tensor view.
#[derive(Debug, Clone)]
pub struct ExternalTensorDescriptor {
    context: DeviceContextCapability,
    buffer: ExternalBufferLease,
    offset_bytes: u64,
    length_bytes: u64,
    shape: Vec<usize>,
    dtype: DType,
    strides_elements: Vec<usize>,
    access: ExternalAccess,
    producer_token: SubmissionToken,
}

impl ExternalTensorDescriptor {
    /// Create an unvalidated external tensor descriptor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: DeviceContextCapability,
        buffer: ExternalBufferLease,
        offset_bytes: u64,
        length_bytes: u64,
        shape: Vec<usize>,
        dtype: DType,
        strides_elements: Vec<usize>,
        access: ExternalAccess,
        producer_token: SubmissionToken,
    ) -> Self {
        Self {
            context,
            buffer,
            offset_bytes,
            length_bytes,
            shape,
            dtype,
            strides_elements,
            access,
            producer_token,
        }
    }

    /// Validate the initial read-only, dense-contiguous f32 adoption contract.
    ///
    /// Validation is side-effect free and performs no allocation, backend registration, readback,
    /// or copy. The expected capability must be the exact one used to initialize the future
    /// consumer backend.
    pub fn validate_read_only_dense(
        self,
        expected_context: &DeviceContextCapability,
    ) -> Result<ValidatedExternalTensorDescriptor> {
        if !self.context.same_context(expected_context) {
            return Err(external_error(
                "external tensor belongs to a different device/queue context",
            ));
        }
        if !self.buffer.context.same_context(&self.context) {
            return Err(external_error(
                "external buffer lease belongs to a different device/queue context",
            ));
        }
        if !self.producer_token.context.same_context(&self.context) {
            return Err(external_error(
                "producer submission token belongs to a different device/queue context",
            ));
        }
        if self.buffer.usage != ExternalBufferUsage::Storage {
            return Err(external_error(
                "external buffer is not declared for device storage use",
            ));
        }
        if self.access != ExternalAccess::ReadOnly {
            return Err(external_error(
                "external tensor adoption currently supports read-only access",
            ));
        }
        if self.dtype != DType::F32 {
            return Err(external_error(format!(
                "external tensor adoption currently supports f32, got {}",
                self.dtype.name()
            )));
        }
        if self.shape.is_empty() {
            return Err(external_error(
                "external tensor shape must have at least one dimension",
            ));
        }
        if self.shape.len() > MAX_RANK {
            return Err(TynxError::RankOverflow {
                rank: self.shape.len(),
                max: MAX_RANK,
            });
        }
        if self.shape.contains(&0) {
            return Err(external_error(format!(
                "external tensor dimensions must be positive, got {:?}",
                self.shape
            )));
        }
        let expected_strides = dense_strides(&self.shape)?;
        if self.strides_elements != expected_strides {
            return Err(external_error(format!(
                "external tensor must be dense-contiguous with strides {expected_strides:?}, got {:?}",
                self.strides_elements
            )));
        }
        let elements = self.shape.iter().try_fold(1_usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or_else(|| external_error("external tensor element count overflowed usize"))
        })?;
        let element_bytes = u64::try_from(self.dtype.size())
            .map_err(|_| external_error("external tensor element size overflowed u64"))?;
        let expected_length = u64::try_from(elements)
            .ok()
            .and_then(|elements| elements.checked_mul(element_bytes))
            .ok_or_else(|| external_error("external tensor byte length overflowed u64"))?;
        if self.length_bytes != expected_length {
            return Err(external_error(format!(
                "external tensor byte length must be exactly {expected_length}, got {}",
                self.length_bytes
            )));
        }
        if !self.offset_bytes.is_multiple_of(self.buffer.alignment) {
            return Err(external_error(format!(
                "external tensor byte offset {} is not aligned to {}",
                self.offset_bytes, self.buffer.alignment
            )));
        }
        if !self.offset_bytes.is_multiple_of(element_bytes)
            || !self.length_bytes.is_multiple_of(element_bytes)
        {
            return Err(external_error(format!(
                "external tensor byte range must be aligned to f32 element size {element_bytes}"
            )));
        }
        let end = self
            .offset_bytes
            .checked_add(self.length_bytes)
            .ok_or_else(|| external_error("external tensor byte range overflowed u64"))?;
        if end > self.buffer.byte_length {
            return Err(external_error(format!(
                "external tensor byte range [{}, {end}) exceeds buffer length {}",
                self.offset_bytes, self.buffer.byte_length
            )));
        }
        Ok(ValidatedExternalTensorDescriptor { descriptor: self })
    }
}

/// A descriptor whose context, range, dtype, shape, layout, usage, and access were validated.
#[derive(Debug)]
pub struct ValidatedExternalTensorDescriptor {
    descriptor: ExternalTensorDescriptor,
}

impl ValidatedExternalTensorDescriptor {
    /// Apply producer-before-consumer ordering and retain the descriptor for backend adoption.
    pub fn acquire(self) -> Result<AcquiredExternalTensorDescriptor> {
        self.descriptor.producer_token.ensure_visible()?;
        Ok(AcquiredExternalTensorDescriptor {
            descriptor: self.descriptor,
        })
    }
}

/// Validated external tensor metadata after producer ordering has been established.
///
/// This still is not a Tynx tensor. A backend adapter must retain this object or an
/// [`ExternalTensorRetention`] clone in the adopted storage and every autodiff tape that may
/// read the forward activation.
#[derive(Debug)]
pub struct AcquiredExternalTensorDescriptor {
    descriptor: ExternalTensorDescriptor,
}

impl AcquiredExternalTensorDescriptor {
    #[cfg(any(feature = "wgpu", feature = "vulkan"))]
    pub(crate) fn belongs_to(&self, expected_context: &DeviceContextCapability) -> bool {
        self.descriptor.context.same_context(expected_context)
    }

    /// Return the type-erased external buffer lease.
    pub fn buffer(&self) -> &ExternalBufferLease {
        &self.descriptor.buffer
    }

    /// Return the view's byte offset.
    pub fn offset_bytes(&self) -> u64 {
        self.descriptor.offset_bytes
    }

    /// Return the exact byte length of the tensor view.
    pub fn length_bytes(&self) -> u64 {
        self.descriptor.length_bytes
    }

    /// Return the tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.descriptor.shape
    }

    /// Return the tensor element type.
    pub fn dtype(&self) -> DType {
        self.descriptor.dtype
    }

    /// Return dense element strides.
    pub fn strides_elements(&self) -> &[usize] {
        &self.descriptor.strides_elements
    }

    /// Clone a sidecar that retains the external allocation for an operation or autodiff tape.
    pub fn retain(&self) -> ExternalTensorRetention {
        ExternalTensorRetention {
            buffer: self.descriptor.buffer.clone(),
        }
    }
}

/// Cloneable sidecar retaining an external allocation through asynchronous work and autodiff.
#[derive(Clone)]
pub struct ExternalTensorRetention {
    buffer: ExternalBufferLease,
}

impl ExternalTensorRetention {
    /// Clone the retention sidecar for a forward output or autodiff tape.
    pub fn retain_for_tape(&self) -> Self {
        self.clone()
    }
}

impl Debug for ExternalTensorRetention {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExternalTensorRetention")
            .field("buffer", &self.buffer)
            .finish_non_exhaustive()
    }
}

fn dense_strides(shape: &[usize]) -> Result<Vec<usize>> {
    let mut strides = vec![1; shape.len()];
    let mut stride = 1_usize;
    for (index, dimension) in shape.iter().enumerate().rev() {
        strides[index] = stride;
        stride = stride
            .checked_mul(*dimension)
            .ok_or_else(|| external_error("external tensor stride computation overflowed usize"))?;
    }
    Ok(strides)
}

fn external_error(message: impl Into<String>) -> TynxError {
    TynxError::ExternalTensor(message.into())
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    #[derive(Debug)]
    struct Resource;

    struct Submission {
        calls: Arc<AtomicUsize>,
        fail: bool,
    }

    impl ExternalSubmission for Submission {
        fn ensure_visible(&self) -> Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                Err(external_error("producer submission failed"))
            } else {
                Ok(())
            }
        }
    }

    fn descriptor() -> (ExternalTensorDescriptor, Arc<AtomicUsize>) {
        let context = DeviceContextCapability::new();
        let buffer = ExternalBufferLease::new(
            context.clone(),
            Arc::new(Resource),
            256,
            4,
            ExternalBufferUsage::Storage,
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let token = SubmissionToken::new(
            context.clone(),
            Submission {
                calls: calls.clone(),
                fail: false,
            },
        );
        (
            ExternalTensorDescriptor::new(
                context,
                buffer,
                16,
                24,
                vec![2, 3],
                DType::F32,
                vec![3, 1],
                ExternalAccess::ReadOnly,
                token,
            ),
            calls,
        )
    }

    #[test]
    fn validates_and_acquires_a_dense_read_only_descriptor() {
        let (descriptor, calls) = descriptor();
        let expected = descriptor.context.clone();

        let acquired = descriptor
            .validate_read_only_dense(&expected)
            .unwrap()
            .acquire()
            .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(acquired.offset_bytes(), 16);
        assert_eq!(acquired.length_bytes(), 24);
        assert_eq!(acquired.shape(), [2, 3]);
        assert_eq!(acquired.dtype(), DType::F32);
        assert_eq!(acquired.strides_elements(), [3, 1]);
        assert!(acquired.buffer().resource::<Resource>().is_some());
    }

    #[test]
    fn rejects_invalid_external_buffer_metadata() {
        let context = DeviceContextCapability::new();
        let resource = Arc::new(Resource);

        let zero_length = ExternalBufferLease::new(
            context.clone(),
            resource.clone(),
            0,
            4,
            ExternalBufferUsage::Storage,
        )
        .unwrap_err();
        assert!(
            zero_length
                .to_string()
                .contains("byte length must be positive")
        );

        let bad_alignment =
            ExternalBufferLease::new(context, resource, 16, 3, ExternalBufferUsage::Storage)
                .unwrap_err();
        assert!(bad_alignment.to_string().contains("positive power of two"));
    }

    #[test]
    fn rejects_context_usage_access_dtype_and_layout_mismatches() {
        let (descriptor, _) = descriptor();
        let wrong_context = DeviceContextCapability::new();
        assert!(
            descriptor
                .clone()
                .validate_read_only_dense(&wrong_context)
                .unwrap_err()
                .to_string()
                .contains("different device/queue context")
        );

        let expected = descriptor.context.clone();
        let mut wrong_buffer_context = descriptor.clone();
        wrong_buffer_context.buffer.context = wrong_context.clone();
        assert!(
            wrong_buffer_context
                .validate_read_only_dense(&expected)
                .unwrap_err()
                .to_string()
                .contains("buffer lease")
        );

        let mut wrong_token_context = descriptor.clone();
        wrong_token_context.producer_token.context = wrong_context;
        assert!(
            wrong_token_context
                .validate_read_only_dense(&expected)
                .unwrap_err()
                .to_string()
                .contains("submission token")
        );

        let mut non_storage = descriptor.clone();
        non_storage.buffer.usage = ExternalBufferUsage::NonStorage;
        assert!(
            non_storage
                .validate_read_only_dense(&expected)
                .unwrap_err()
                .to_string()
                .contains("storage use")
        );

        let mut writable = descriptor.clone();
        writable.access = ExternalAccess::ReadWrite;
        assert!(
            writable
                .validate_read_only_dense(&expected)
                .unwrap_err()
                .to_string()
                .contains("read-only")
        );

        let mut integer = descriptor.clone();
        integer.dtype = DType::I64;
        assert!(
            integer
                .validate_read_only_dense(&expected)
                .unwrap_err()
                .to_string()
                .contains("supports f32")
        );

        let mut strided = descriptor;
        strided.strides_elements = vec![1, 2];
        assert!(
            strided
                .validate_read_only_dense(&expected)
                .unwrap_err()
                .to_string()
                .contains("dense-contiguous")
        );
    }

    #[test]
    fn rejects_invalid_shape_alignment_and_byte_ranges() {
        let (descriptor, _) = descriptor();
        let expected = descriptor.context.clone();

        let mut empty_shape = descriptor.clone();
        empty_shape.shape.clear();
        empty_shape.strides_elements.clear();
        assert!(
            empty_shape
                .validate_read_only_dense(&expected)
                .unwrap_err()
                .to_string()
                .contains("at least one dimension")
        );

        let mut zero_dimension = descriptor.clone();
        zero_dimension.shape = vec![2, 0];
        zero_dimension.strides_elements = vec![0, 1];
        zero_dimension.length_bytes = 0;
        assert!(
            zero_dimension
                .validate_read_only_dense(&expected)
                .unwrap_err()
                .to_string()
                .contains("dimensions must be positive")
        );

        let mut excessive_rank = descriptor.clone();
        excessive_rank.shape = vec![1; MAX_RANK + 1];
        excessive_rank.strides_elements = vec![1; MAX_RANK + 1];
        let error = excessive_rank
            .validate_read_only_dense(&expected)
            .unwrap_err();
        assert!(matches!(error, TynxError::RankOverflow { .. }));

        let mut wrong_length = descriptor.clone();
        wrong_length.length_bytes = 20;
        assert!(
            wrong_length
                .validate_read_only_dense(&expected)
                .unwrap_err()
                .to_string()
                .contains("exactly 24")
        );

        let mut unaligned = descriptor.clone();
        unaligned.offset_bytes = 18;
        assert!(
            unaligned
                .validate_read_only_dense(&expected)
                .unwrap_err()
                .to_string()
                .contains("not aligned")
        );

        let mut outside = descriptor;
        outside.offset_bytes = 240;
        assert!(
            outside
                .validate_read_only_dense(&expected)
                .unwrap_err()
                .to_string()
                .contains("exceeds buffer length")
        );
    }

    #[test]
    fn propagates_producer_ordering_failures_without_acquiring() {
        let (mut descriptor, calls) = descriptor();
        let expected = descriptor.context.clone();
        descriptor.producer_token = SubmissionToken::new(
            expected.clone(),
            Submission {
                calls: calls.clone(),
                fail: true,
            },
        );

        let error = descriptor
            .validate_read_only_dense(&expected)
            .unwrap()
            .acquire()
            .unwrap_err();

        assert!(error.to_string().contains("producer submission failed"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
