//! Rank-erased tensor containers used by the runtime.
//!
//! The public operations on [`DynTensor`], [`DynInt`], and [`DynBool`] are Tynx's shared
//! numerical boundary. Frontends translate their own conventions at the edge—ONNX attributes in
//! the interpreter and Python arguments in the CPython facade—then delegate device computation to
//! these methods. Keeping the numerical implementation here prevents eager and imported models
//! from developing separate semantics.

use burn::tensor::{
    Bool, DType, Device, Distribution, IndexingUpdateOp, Int, Slice, Tensor, TensorData,
    activation,
    module::{
        adaptive_avg_pool2d as burn_adaptive_avg_pool2d, avg_pool2d as burn_avg_pool2d,
        conv2d as burn_conv2d, max_pool2d as burn_max_pool2d,
    },
    ops::ConvOptions,
};

use crate::error::{Result, TynxError};

/// Highest tensor rank represented by Tynx.
pub const MAX_RANK: usize = 6;

fn rank_overflow(rank: usize) -> TynxError {
    TynxError::RankOverflow {
        rank,
        max: MAX_RANK,
    }
}

fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
    let rank = left.len().max(right.len());
    let mut output = Vec::with_capacity(rank);

    for offset in 0..rank {
        let left_dim = left
            .len()
            .checked_sub(offset + 1)
            .map_or(1, |index| left[index]);
        let right_dim = right
            .len()
            .checked_sub(offset + 1)
            .map_or(1, |index| right[index]);

        if left_dim != right_dim && left_dim != 1 && right_dim != 1 {
            let axis = rank - offset - 1;
            return Err(TynxError::Shape(format!(
                "cannot broadcast shapes {left:?} and {right:?}: dimensions {left_dim} and {right_dim} conflict at axis {axis}"
            )));
        }
        output.push(left_dim.max(right_dim));
    }

    output.reverse();
    Ok(output)
}

fn validate_matmul_shapes(left: &[usize], right: &[usize]) -> Result<()> {
    if left.len() != right.len() || left.len() < 2 {
        return Err(TynxError::Shape(format!(
            "matmul requires matching ranks >= 2, got {} and {}",
            left.len(),
            right.len()
        )));
    }

    let left_inner = left[left.len() - 1];
    let right_inner = right[right.len() - 2];
    if left_inner != right_inner {
        return Err(TynxError::Shape(format!(
            "matmul inner dimensions must match for shapes {left:?} and {right:?}, got {left_inner} and {right_inner}"
        )));
    }

    broadcast_shape(&left[..left.len() - 2], &right[..right.len() - 2])?;
    Ok(())
}

fn validate_pool2d(
    input: &[usize],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    operation: &str,
) -> Result<()> {
    if input.len() != 4 {
        return Err(TynxError::Shape(format!(
            "{operation} requires rank-4 NCHW input, got {input:?}"
        )));
    }
    if kernel.contains(&0) || stride.contains(&0) || dilation.contains(&0) {
        return Err(TynxError::Shape(format!(
            "{operation} kernel, stride, and dilation must contain positive integers"
        )));
    }
    for axis in 0..2 {
        if padding[axis] > kernel[axis] / 2 {
            return Err(TynxError::Shape(format!(
                "{operation} padding {} exceeds half the kernel size {} at spatial axis {axis}",
                padding[axis], kernel[axis]
            )));
        }
        let effective_kernel = dilation[axis]
            .checked_mul(kernel[axis] - 1)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| TynxError::Shape(format!("{operation} kernel extent overflowed")))?;
        let padded_input = padding[axis]
            .checked_mul(2)
            .and_then(|value| input[axis + 2].checked_add(value))
            .ok_or_else(|| TynxError::Shape(format!("{operation} padded extent overflowed")))?;
        if padded_input < effective_kernel {
            return Err(TynxError::Shape(format!(
                "{operation} kernel extent {effective_kernel} exceeds padded input extent {padded_input} at spatial axis {axis}"
            )));
        }
    }
    Ok(())
}

/// A floating-point tensor with a runtime rank.
#[derive(Debug, Clone)]
pub enum DynTensor {
    R1(Tensor<1>),
    R2(Tensor<2>),
    R3(Tensor<3>),
    R4(Tensor<4>),
    R5(Tensor<5>),
    R6(Tensor<6>),
}

/// An integer tensor with a runtime rank.
#[derive(Debug, Clone)]
pub enum DynInt {
    R1(Tensor<1, Int>),
    R2(Tensor<2, Int>),
    R3(Tensor<3, Int>),
    R4(Tensor<4, Int>),
    R5(Tensor<5, Int>),
    R6(Tensor<6, Int>),
}

/// A boolean tensor with a runtime rank.
#[derive(Debug, Clone)]
pub enum DynBool {
    R1(Tensor<1, Bool>),
    R2(Tensor<2, Bool>),
    R3(Tensor<3, Bool>),
    R4(Tensor<4, Bool>),
    R5(Tensor<5, Bool>),
    R6(Tensor<6, Bool>),
}

macro_rules! map_float {
    ($tensor:expr, |$value:ident| $body:expr) => {
        match $tensor {
            DynTensor::R1($value) => DynTensor::R1($body),
            DynTensor::R2($value) => DynTensor::R2($body),
            DynTensor::R3($value) => DynTensor::R3($body),
            DynTensor::R4($value) => DynTensor::R4($body),
            DynTensor::R5($value) => DynTensor::R5($body),
            DynTensor::R6($value) => DynTensor::R6($body),
        }
    };
}

macro_rules! map_int {
    ($tensor:expr, |$value:ident| $body:expr) => {
        match $tensor {
            DynInt::R1($value) => DynInt::R1($body),
            DynInt::R2($value) => DynInt::R2($body),
            DynInt::R3($value) => DynInt::R3($body),
            DynInt::R4($value) => DynInt::R4($body),
            DynInt::R5($value) => DynInt::R5($body),
            DynInt::R6($value) => DynInt::R6($body),
        }
    };
}

macro_rules! map_bool {
    ($tensor:expr, |$value:ident| $body:expr) => {
        match $tensor {
            DynBool::R1($value) => DynBool::R1($body),
            DynBool::R2($value) => DynBool::R2($body),
            DynBool::R3($value) => DynBool::R3($body),
            DynBool::R4($value) => DynBool::R4($body),
            DynBool::R5($value) => DynBool::R5($body),
            DynBool::R6($value) => DynBool::R6($body),
        }
    };
}

macro_rules! map_float_bool {
    ($tensor:expr, |$value:ident| $body:expr) => {
        match $tensor {
            DynTensor::R1($value) => DynBool::R1($body),
            DynTensor::R2($value) => DynBool::R2($body),
            DynTensor::R3($value) => DynBool::R3($body),
            DynTensor::R4($value) => DynBool::R4($body),
            DynTensor::R5($value) => DynBool::R5($body),
            DynTensor::R6($value) => DynBool::R6($body),
        }
    };
}

macro_rules! map_int_bool {
    ($tensor:expr, |$value:ident| $body:expr) => {
        match $tensor {
            DynInt::R1($value) => DynBool::R1($body),
            DynInt::R2($value) => DynBool::R2($body),
            DynInt::R3($value) => DynBool::R3($body),
            DynInt::R4($value) => DynBool::R4($body),
            DynInt::R5($value) => DynBool::R5($body),
            DynInt::R6($value) => DynBool::R6($body),
        }
    };
}

macro_rules! reshape_dyn {
    ($tensor:expr, $dims:expr, $kind:ident) => {
        match $dims.as_slice() {
            [d0] => $kind::R1($tensor.reshape([*d0])),
            [d0, d1] => $kind::R2($tensor.reshape([*d0, *d1])),
            [d0, d1, d2] => $kind::R3($tensor.reshape([*d0, *d1, *d2])),
            [d0, d1, d2, d3] => $kind::R4($tensor.reshape([*d0, *d1, *d2, *d3])),
            [d0, d1, d2, d3, d4] => $kind::R5($tensor.reshape([*d0, *d1, *d2, *d3, *d4])),
            [d0, d1, d2, d3, d4, d5] => $kind::R6($tensor.reshape([*d0, *d1, *d2, *d3, *d4, *d5])),
            _ => return Err(rank_overflow($dims.len())),
        }
    };
}

macro_rules! zip_float {
    ($left:expr, $right:expr, |$a:ident, $b:ident| $body:expr) => {
        match ($left, $right) {
            (DynTensor::R1($a), DynTensor::R1($b)) => DynTensor::R1($body),
            (DynTensor::R2($a), DynTensor::R2($b)) => DynTensor::R2($body),
            (DynTensor::R3($a), DynTensor::R3($b)) => DynTensor::R3($body),
            (DynTensor::R4($a), DynTensor::R4($b)) => DynTensor::R4($body),
            (DynTensor::R5($a), DynTensor::R5($b)) => DynTensor::R5($body),
            (DynTensor::R6($a), DynTensor::R6($b)) => DynTensor::R6($body),
            _ => unreachable!("tensor ranks were promoted before the operation"),
        }
    };
}

macro_rules! zip_int {
    ($left:expr, $right:expr, |$a:ident, $b:ident| $body:expr) => {
        match ($left, $right) {
            (DynInt::R1($a), DynInt::R1($b)) => DynInt::R1($body),
            (DynInt::R2($a), DynInt::R2($b)) => DynInt::R2($body),
            (DynInt::R3($a), DynInt::R3($b)) => DynInt::R3($body),
            (DynInt::R4($a), DynInt::R4($b)) => DynInt::R4($body),
            (DynInt::R5($a), DynInt::R5($b)) => DynInt::R5($body),
            (DynInt::R6($a), DynInt::R6($b)) => DynInt::R6($body),
            _ => unreachable!("tensor ranks were promoted before the operation"),
        }
    };
}

macro_rules! zip_float_bool {
    ($left:expr, $right:expr, |$a:ident, $b:ident| $body:expr) => {
        match ($left, $right) {
            (DynTensor::R1($a), DynTensor::R1($b)) => DynBool::R1($body),
            (DynTensor::R2($a), DynTensor::R2($b)) => DynBool::R2($body),
            (DynTensor::R3($a), DynTensor::R3($b)) => DynBool::R3($body),
            (DynTensor::R4($a), DynTensor::R4($b)) => DynBool::R4($body),
            (DynTensor::R5($a), DynTensor::R5($b)) => DynBool::R5($body),
            (DynTensor::R6($a), DynTensor::R6($b)) => DynBool::R6($body),
            _ => unreachable!("tensor ranks were promoted before the operation"),
        }
    };
}

macro_rules! zip_int_bool {
    ($left:expr, $right:expr, |$a:ident, $b:ident| $body:expr) => {
        match ($left, $right) {
            (DynInt::R1($a), DynInt::R1($b)) => DynBool::R1($body),
            (DynInt::R2($a), DynInt::R2($b)) => DynBool::R2($body),
            (DynInt::R3($a), DynInt::R3($b)) => DynBool::R3($body),
            (DynInt::R4($a), DynInt::R4($b)) => DynBool::R4($body),
            (DynInt::R5($a), DynInt::R5($b)) => DynBool::R5($body),
            (DynInt::R6($a), DynInt::R6($b)) => DynBool::R6($body),
            _ => unreachable!("tensor ranks were promoted before the operation"),
        }
    };
}

macro_rules! zip_bool {
    ($left:expr, $right:expr, |$a:ident, $b:ident| $body:expr) => {
        match ($left, $right) {
            (DynBool::R1($a), DynBool::R1($b)) => DynBool::R1($body),
            (DynBool::R2($a), DynBool::R2($b)) => DynBool::R2($body),
            (DynBool::R3($a), DynBool::R3($b)) => DynBool::R3($body),
            (DynBool::R4($a), DynBool::R4($b)) => DynBool::R4($body),
            (DynBool::R5($a), DynBool::R5($b)) => DynBool::R5($body),
            (DynBool::R6($a), DynBool::R6($b)) => DynBool::R6($body),
            _ => unreachable!("tensor ranks were promoted before the operation"),
        }
    };
}

macro_rules! where_float {
    ($condition:expr, $then:expr, $otherwise:expr) => {
        match ($condition, $then, $otherwise) {
            (DynBool::R1(condition), DynTensor::R1(then), DynTensor::R1(otherwise)) => {
                DynTensor::R1(otherwise.mask_where(condition, then))
            }
            (DynBool::R2(condition), DynTensor::R2(then), DynTensor::R2(otherwise)) => {
                DynTensor::R2(otherwise.mask_where(condition, then))
            }
            (DynBool::R3(condition), DynTensor::R3(then), DynTensor::R3(otherwise)) => {
                DynTensor::R3(otherwise.mask_where(condition, then))
            }
            (DynBool::R4(condition), DynTensor::R4(then), DynTensor::R4(otherwise)) => {
                DynTensor::R4(otherwise.mask_where(condition, then))
            }
            (DynBool::R5(condition), DynTensor::R5(then), DynTensor::R5(otherwise)) => {
                DynTensor::R5(otherwise.mask_where(condition, then))
            }
            (DynBool::R6(condition), DynTensor::R6(then), DynTensor::R6(otherwise)) => {
                DynTensor::R6(otherwise.mask_where(condition, then))
            }
            _ => unreachable!("Where operands were promoted to the same rank"),
        }
    };
}

macro_rules! where_int {
    ($condition:expr, $then:expr, $otherwise:expr) => {
        match ($condition, $then, $otherwise) {
            (DynBool::R1(condition), DynInt::R1(then), DynInt::R1(otherwise)) => {
                DynInt::R1(otherwise.mask_where(condition, then))
            }
            (DynBool::R2(condition), DynInt::R2(then), DynInt::R2(otherwise)) => {
                DynInt::R2(otherwise.mask_where(condition, then))
            }
            (DynBool::R3(condition), DynInt::R3(then), DynInt::R3(otherwise)) => {
                DynInt::R3(otherwise.mask_where(condition, then))
            }
            (DynBool::R4(condition), DynInt::R4(then), DynInt::R4(otherwise)) => {
                DynInt::R4(otherwise.mask_where(condition, then))
            }
            (DynBool::R5(condition), DynInt::R5(then), DynInt::R5(otherwise)) => {
                DynInt::R5(otherwise.mask_where(condition, then))
            }
            (DynBool::R6(condition), DynInt::R6(then), DynInt::R6(otherwise)) => {
                DynInt::R6(otherwise.mask_where(condition, then))
            }
            _ => unreachable!("Where operands were promoted to the same rank"),
        }
    };
}

macro_rules! where_bool {
    ($condition:expr, $then:expr, $otherwise:expr) => {
        match ($condition, $then, $otherwise) {
            (DynBool::R1(condition), DynBool::R1(then), DynBool::R1(otherwise)) => {
                DynBool::R1(otherwise.mask_where(condition, then))
            }
            (DynBool::R2(condition), DynBool::R2(then), DynBool::R2(otherwise)) => {
                DynBool::R2(otherwise.mask_where(condition, then))
            }
            (DynBool::R3(condition), DynBool::R3(then), DynBool::R3(otherwise)) => {
                DynBool::R3(otherwise.mask_where(condition, then))
            }
            (DynBool::R4(condition), DynBool::R4(then), DynBool::R4(otherwise)) => {
                DynBool::R4(otherwise.mask_where(condition, then))
            }
            (DynBool::R5(condition), DynBool::R5(then), DynBool::R5(otherwise)) => {
                DynBool::R5(otherwise.mask_where(condition, then))
            }
            (DynBool::R6(condition), DynBool::R6(then), DynBool::R6(otherwise)) => {
                DynBool::R6(otherwise.mask_where(condition, then))
            }
            _ => unreachable!("Where operands were promoted to the same rank"),
        }
    };
}

macro_rules! impl_metadata {
    ($name:ident) => {
        impl $name {
            /// Return the tensor's rank.
            pub fn rank(&self) -> usize {
                match self {
                    Self::R1(_) => 1,
                    Self::R2(_) => 2,
                    Self::R3(_) => 3,
                    Self::R4(_) => 4,
                    Self::R5(_) => 5,
                    Self::R6(_) => 6,
                }
            }

            /// Return the tensor's dimensions.
            pub fn dims(&self) -> Vec<usize> {
                match self {
                    Self::R1(tensor) => tensor.dims().to_vec(),
                    Self::R2(tensor) => tensor.dims().to_vec(),
                    Self::R3(tensor) => tensor.dims().to_vec(),
                    Self::R4(tensor) => tensor.dims().to_vec(),
                    Self::R5(tensor) => tensor.dims().to_vec(),
                    Self::R6(tensor) => tensor.dims().to_vec(),
                }
            }

            /// Return the device that owns the tensor.
            pub fn device(&self) -> Device {
                match self {
                    Self::R1(tensor) => tensor.device(),
                    Self::R2(tensor) => tensor.device(),
                    Self::R3(tensor) => tensor.device(),
                    Self::R4(tensor) => tensor.device(),
                    Self::R5(tensor) => tensor.device(),
                    Self::R6(tensor) => tensor.device(),
                }
            }

            /// Move the tensor to another runtime device while preserving rank and dtype.
            pub fn to_device(self, device: &Device) -> Self {
                match self {
                    Self::R1(tensor) => Self::R1(tensor.to_device(device)),
                    Self::R2(tensor) => Self::R2(tensor.to_device(device)),
                    Self::R3(tensor) => Self::R3(tensor.to_device(device)),
                    Self::R4(tensor) => Self::R4(tensor.to_device(device)),
                    Self::R5(tensor) => Self::R5(tensor.to_device(device)),
                    Self::R6(tensor) => Self::R6(tensor.to_device(device)),
                }
            }

            /// Create a rank-erased tensor from host data.
            pub fn from_data(data: TensorData, rank: usize, device: &Device) -> Result<Self> {
                let dtype = data.dtype;
                Ok(match rank {
                    1 => Self::R1(Tensor::from_data(data, (device, dtype))),
                    2 => Self::R2(Tensor::from_data(data, (device, dtype))),
                    3 => Self::R3(Tensor::from_data(data, (device, dtype))),
                    4 => Self::R4(Tensor::from_data(data, (device, dtype))),
                    5 => Self::R5(Tensor::from_data(data, (device, dtype))),
                    6 => Self::R6(Tensor::from_data(data, (device, dtype))),
                    0 => {
                        return Err(TynxError::UnsupportedOp(
                            "rank-0 tensor must be represented as a scalar".to_string(),
                        ));
                    }
                    rank => return Err(rank_overflow(rank)),
                })
            }

            /// Read the tensor back into host data.
            pub fn into_data(self) -> TensorData {
                match self {
                    Self::R1(tensor) => tensor.into_data(),
                    Self::R2(tensor) => tensor.into_data(),
                    Self::R3(tensor) => tensor.into_data(),
                    Self::R4(tensor) => tensor.into_data(),
                    Self::R5(tensor) => tensor.into_data(),
                    Self::R6(tensor) => tensor.into_data(),
                }
            }
        }
    };
}

impl_metadata!(DynTensor);
impl_metadata!(DynInt);
impl_metadata!(DynBool);

#[cfg(feature = "training")]
impl DynTensor {
    /// Return whether this tensor is an autodiff leaf that requires gradients.
    pub fn is_require_grad(&self) -> bool {
        match self {
            Self::R1(tensor) => tensor.is_require_grad(),
            Self::R2(tensor) => tensor.is_require_grad(),
            Self::R3(tensor) => tensor.is_require_grad(),
            Self::R4(tensor) => tensor.is_require_grad(),
            Self::R5(tensor) => tensor.is_require_grad(),
            Self::R6(tensor) => tensor.is_require_grad(),
        }
    }

    /// Mark this tensor as a differentiable leaf.
    pub fn require_grad(self) -> Self {
        map_float!(self, |tensor| tensor.require_grad())
    }

    /// Stop autodiff tracking while keeping the tensor on its current device/backend.
    pub fn detach(self) -> Self {
        map_float!(self, |tensor| tensor.detach().set_require_grad(false))
    }

    /// Return the underlying non-autodiff tensor for off-tape updates.
    pub fn inner(self) -> Self {
        map_float!(self, |tensor| tensor.inner())
    }

    /// Lift an underlying tensor back onto the autodiff backend.
    pub fn to_autodiff(self) -> Self {
        map_float!(self, |tensor| Tensor::from_inner(tensor))
    }

    /// Run reverse-mode autodiff from this tensor.
    pub fn backward(&self) -> crate::Gradients {
        match self {
            Self::R1(tensor) => tensor.backward(),
            Self::R2(tensor) => tensor.backward(),
            Self::R3(tensor) => tensor.backward(),
            Self::R4(tensor) => tensor.backward(),
            Self::R5(tensor) => tensor.backward(),
            Self::R6(tensor) => tensor.backward(),
        }
    }

    /// Clone this tensor's gradient from a backward result without removing it.
    pub fn grad(&self, gradients: &crate::Gradients) -> Option<Self> {
        match self {
            Self::R1(tensor) => tensor.grad(gradients).map(Self::R1),
            Self::R2(tensor) => tensor.grad(gradients).map(Self::R2),
            Self::R3(tensor) => tensor.grad(gradients).map(Self::R3),
            Self::R4(tensor) => tensor.grad(gradients).map(Self::R4),
            Self::R5(tensor) => tensor.grad(gradients).map(Self::R5),
            Self::R6(tensor) => tensor.grad(gradients).map(Self::R6),
        }
    }

    /// Remove this tensor's gradient from a backward result.
    pub fn grad_remove(&self, gradients: &mut crate::Gradients) -> Option<Self> {
        match self {
            Self::R1(tensor) => tensor.grad_remove(gradients).map(Self::R1),
            Self::R2(tensor) => tensor.grad_remove(gradients).map(Self::R2),
            Self::R3(tensor) => tensor.grad_remove(gradients).map(Self::R3),
            Self::R4(tensor) => tensor.grad_remove(gradients).map(Self::R4),
            Self::R5(tensor) => tensor.grad_remove(gradients).map(Self::R5),
            Self::R6(tensor) => tensor.grad_remove(gradients).map(Self::R6),
        }
    }
}

macro_rules! impl_concat {
    ($kind:ident) => {
        impl $kind {
            /// Concatenate tensors of the same rank along one dimension.
            pub fn concat(tensors: Vec<Self>, dim: usize) -> Result<Self> {
                let Some(first) = tensors.first() else {
                    return Err(TynxError::Shape(
                        "concatenation requires at least one tensor".to_string(),
                    ));
                };
                let rank = first.rank();
                if dim >= rank {
                    return Err(TynxError::Shape(format!(
                        "concatenation axis {dim} is out of range for rank {rank}"
                    )));
                }
                if tensors.iter().any(|tensor| tensor.rank() != rank) {
                    return Err(TynxError::Shape(
                        "concatenation requires tensors with equal ranks".to_string(),
                    ));
                }
                let first_dims = first.dims();
                let first_dtype = first.dtype();
                let first_device = first.device();
                for tensor in tensors.iter().skip(1) {
                    if tensor.dtype() != first_dtype {
                        return Err(TynxError::Shape(
                            "concatenation requires tensors with equal dtypes".to_string(),
                        ));
                    }
                    if tensor.device() != first_device {
                        return Err(TynxError::Shape(
                            "concatenation requires tensors on the same device".to_string(),
                        ));
                    }
                    let dims = tensor.dims();
                    if dims
                        .iter()
                        .zip(&first_dims)
                        .enumerate()
                        .any(|(axis, (left, right))| axis != dim && left != right)
                    {
                        return Err(TynxError::Shape(format!(
                            "concatenation shapes differ outside axis {dim}: {first_dims:?} and {dims:?}"
                        )));
                    }
                }

                Ok(match rank {
                    1 => Self::R1(Tensor::cat(
                        tensors
                            .into_iter()
                            .map(|tensor| match tensor {
                                Self::R1(tensor) => tensor,
                                _ => unreachable!("tensor ranks were validated"),
                            })
                            .collect(),
                        dim,
                    )),
                    2 => Self::R2(Tensor::cat(
                        tensors
                            .into_iter()
                            .map(|tensor| match tensor {
                                Self::R2(tensor) => tensor,
                                _ => unreachable!("tensor ranks were validated"),
                            })
                            .collect(),
                        dim,
                    )),
                    3 => Self::R3(Tensor::cat(
                        tensors
                            .into_iter()
                            .map(|tensor| match tensor {
                                Self::R3(tensor) => tensor,
                                _ => unreachable!("tensor ranks were validated"),
                            })
                            .collect(),
                        dim,
                    )),
                    4 => Self::R4(Tensor::cat(
                        tensors
                            .into_iter()
                            .map(|tensor| match tensor {
                                Self::R4(tensor) => tensor,
                                _ => unreachable!("tensor ranks were validated"),
                            })
                            .collect(),
                        dim,
                    )),
                    5 => Self::R5(Tensor::cat(
                        tensors
                            .into_iter()
                            .map(|tensor| match tensor {
                                Self::R5(tensor) => tensor,
                                _ => unreachable!("tensor ranks were validated"),
                            })
                            .collect(),
                        dim,
                    )),
                    6 => Self::R6(Tensor::cat(
                        tensors
                            .into_iter()
                            .map(|tensor| match tensor {
                                Self::R6(tensor) => tensor,
                                _ => unreachable!("tensor ranks were validated"),
                            })
                            .collect(),
                        dim,
                    )),
                    _ => return Err(rank_overflow(rank)),
                })
            }

            /// Stack equal-shaped tensors along a new dimension.
            pub fn stack(tensors: Vec<Self>, dim: usize) -> Result<Self> {
                let Some(first) = tensors.first() else {
                    return Err(TynxError::Shape(
                        "stack requires at least one tensor".to_string(),
                    ));
                };
                let rank = first.rank();
                if rank == MAX_RANK {
                    return Err(rank_overflow(rank + 1));
                }
                if dim > rank {
                    return Err(TynxError::Shape(format!(
                        "stack axis {dim} is out of range for rank {rank}"
                    )));
                }
                let shape = first.dims();
                if tensors.iter().any(|tensor| tensor.dims() != shape) {
                    return Err(TynxError::Shape(
                        "stack requires tensors with equal shapes".to_string(),
                    ));
                }
                let expanded = tensors
                    .into_iter()
                    .map(|tensor| {
                        let mut dims = shape.clone();
                        dims.insert(dim, 1);
                        tensor.reshape(dims)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Self::concat(expanded, dim)
            }
        }
    };
}

macro_rules! select_dyn {
    ($tensor:expr, $dim:expr, $indices:expr, $kind:ident) => {
        match $indices {
            DynInt::R1(indices) => match $tensor {
                $kind::R1(tensor) => $kind::R1(tensor.select($dim, indices)),
                $kind::R2(tensor) => $kind::R2(tensor.select($dim, indices)),
                $kind::R3(tensor) => $kind::R3(tensor.select($dim, indices)),
                $kind::R4(tensor) => $kind::R4(tensor.select($dim, indices)),
                $kind::R5(tensor) => $kind::R5(tensor.select($dim, indices)),
                $kind::R6(tensor) => $kind::R6(tensor.select($dim, indices)),
            },
            _ => return Err(TynxError::Shape("select indices must be rank 1".into())),
        }
    };
}

macro_rules! gather_dyn {
    ($tensor:expr, $dim:expr, $indices:expr, $kind:ident) => {
        match ($tensor, $indices) {
            ($kind::R1(tensor), DynInt::R1(indices)) => $kind::R1(tensor.gather($dim, indices)),
            ($kind::R2(tensor), DynInt::R2(indices)) => $kind::R2(tensor.gather($dim, indices)),
            ($kind::R3(tensor), DynInt::R3(indices)) => $kind::R3(tensor.gather($dim, indices)),
            ($kind::R4(tensor), DynInt::R4(indices)) => $kind::R4(tensor.gather($dim, indices)),
            ($kind::R5(tensor), DynInt::R5(indices)) => $kind::R5(tensor.gather($dim, indices)),
            ($kind::R6(tensor), DynInt::R6(indices)) => $kind::R6(tensor.gather($dim, indices)),
            (tensor, indices) => {
                return Err(TynxError::Shape(format!(
                    "gather ranks differ: data {}, indices {}",
                    tensor.rank(),
                    indices.rank()
                )));
            }
        }
    };
}

fn gather_slices(data: &[usize], indices: &[usize], dim: usize) -> Result<Vec<Slice>> {
    if data.len() != indices.len() {
        return Err(TynxError::Shape(format!(
            "gather ranks differ: data {}, indices {}",
            data.len(),
            indices.len()
        )));
    }
    if dim >= data.len() {
        return Err(TynxError::Shape(format!(
            "gather dimension {dim} is outside rank {}",
            data.len()
        )));
    }

    data.iter()
        .zip(indices)
        .enumerate()
        .map(|(axis, (&data_size, &index_size))| {
            if axis == dim {
                return Ok(Slice::full());
            }
            if index_size > data_size {
                return Err(TynxError::Shape(format!(
                    "gather index size {index_size} exceeds data size {data_size} at dimension {axis}"
                )));
            }
            let end = isize::try_from(index_size).map_err(|_| {
                TynxError::Shape(format!(
                    "gather index size at dimension {axis} exceeds platform limits"
                ))
            })?;
            Ok(Slice::new(0, Some(end), 1))
        })
        .collect()
}

macro_rules! slice_assign_dyn {
    ($tensor:expr, $slices:expr, $values:expr, $kind:ident) => {
        match ($tensor, $values) {
            ($kind::R1(tensor), $kind::R1(values)) => {
                $kind::R1(tensor.slice_assign($slices, values))
            }
            ($kind::R2(tensor), $kind::R2(values)) => {
                $kind::R2(tensor.slice_assign($slices, values))
            }
            ($kind::R3(tensor), $kind::R3(values)) => {
                $kind::R3(tensor.slice_assign($slices, values))
            }
            ($kind::R4(tensor), $kind::R4(values)) => {
                $kind::R4(tensor.slice_assign($slices, values))
            }
            ($kind::R5(tensor), $kind::R5(values)) => {
                $kind::R5(tensor.slice_assign($slices, values))
            }
            ($kind::R6(tensor), $kind::R6(values)) => {
                $kind::R6(tensor.slice_assign($slices, values))
            }
            (tensor, values) => {
                return Err(TynxError::Shape(format!(
                    "slice assignment ranks differ: destination {}, values {}",
                    tensor.rank(),
                    values.rank()
                )));
            }
        }
    };
}

macro_rules! gather_nd_output {
    ($tensor:expr, $indices:expr, $rank:expr, $kind:ident) => {
        match $rank {
            1 => $kind::R1($tensor.gather_nd($indices)),
            2 => $kind::R2($tensor.gather_nd($indices)),
            3 => $kind::R3($tensor.gather_nd($indices)),
            4 => $kind::R4($tensor.gather_nd($indices)),
            5 => $kind::R5($tensor.gather_nd($indices)),
            6 => $kind::R6($tensor.gather_nd($indices)),
            rank => return Err(rank_overflow(rank)),
        }
    };
}

macro_rules! gather_nd_indices {
    ($tensor:expr, $indices:expr, $rank:expr, $kind:ident) => {
        match $indices {
            DynInt::R1(indices) => gather_nd_output!($tensor, indices, $rank, $kind),
            DynInt::R2(indices) => gather_nd_output!($tensor, indices, $rank, $kind),
            DynInt::R3(indices) => gather_nd_output!($tensor, indices, $rank, $kind),
            DynInt::R4(indices) => gather_nd_output!($tensor, indices, $rank, $kind),
            DynInt::R5(indices) => gather_nd_output!($tensor, indices, $rank, $kind),
            DynInt::R6(indices) => gather_nd_output!($tensor, indices, $rank, $kind),
        }
    };
}

macro_rules! gather_nd_dyn {
    ($tensor:expr, $indices:expr, $rank:expr, $kind:ident) => {
        match $tensor {
            $kind::R1(tensor) => gather_nd_indices!(tensor, $indices, $rank, $kind),
            $kind::R2(tensor) => gather_nd_indices!(tensor, $indices, $rank, $kind),
            $kind::R3(tensor) => gather_nd_indices!(tensor, $indices, $rank, $kind),
            $kind::R4(tensor) => gather_nd_indices!(tensor, $indices, $rank, $kind),
            $kind::R5(tensor) => gather_nd_indices!(tensor, $indices, $rank, $kind),
            $kind::R6(tensor) => gather_nd_indices!(tensor, $indices, $rank, $kind),
        }
    };
}

macro_rules! scatter_dyn {
    ($tensor:expr, $dim:expr, $indices:expr, $values:expr, $update:expr, $kind:ident) => {
        match ($tensor, $indices, $values) {
            ($kind::R1(tensor), DynInt::R1(indices), $kind::R1(values)) => {
                $kind::R1(tensor.scatter($dim, indices, values, $update))
            }
            ($kind::R2(tensor), DynInt::R2(indices), $kind::R2(values)) => {
                $kind::R2(tensor.scatter($dim, indices, values, $update))
            }
            ($kind::R3(tensor), DynInt::R3(indices), $kind::R3(values)) => {
                $kind::R3(tensor.scatter($dim, indices, values, $update))
            }
            ($kind::R4(tensor), DynInt::R4(indices), $kind::R4(values)) => {
                $kind::R4(tensor.scatter($dim, indices, values, $update))
            }
            ($kind::R5(tensor), DynInt::R5(indices), $kind::R5(values)) => {
                $kind::R5(tensor.scatter($dim, indices, values, $update))
            }
            ($kind::R6(tensor), DynInt::R6(indices), $kind::R6(values)) => {
                $kind::R6(tensor.scatter($dim, indices, values, $update))
            }
            (tensor, indices, values) => {
                return Err(TynxError::Shape(format!(
                    "scatter ranks differ: data {}, indices {}, updates {}",
                    tensor.rank(),
                    indices.rank(),
                    values.rank()
                )));
            }
        }
    };
}

macro_rules! scatter_nd_values {
    ($tensor:expr, $indices:expr, $values:expr, $update:expr, $kind:ident, $variant:ident) => {
        match $values {
            $kind::R1(values) => $kind::$variant($tensor.scatter_nd($indices, values, $update)),
            $kind::R2(values) => $kind::$variant($tensor.scatter_nd($indices, values, $update)),
            $kind::R3(values) => $kind::$variant($tensor.scatter_nd($indices, values, $update)),
            $kind::R4(values) => $kind::$variant($tensor.scatter_nd($indices, values, $update)),
            $kind::R5(values) => $kind::$variant($tensor.scatter_nd($indices, values, $update)),
            $kind::R6(values) => $kind::$variant($tensor.scatter_nd($indices, values, $update)),
        }
    };
}

macro_rules! scatter_nd_indices {
    ($tensor:expr, $indices:expr, $values:expr, $update:expr, $kind:ident, $variant:ident) => {
        match $indices {
            DynInt::R1(indices) => {
                scatter_nd_values!($tensor, indices, $values, $update, $kind, $variant)
            }
            DynInt::R2(indices) => {
                scatter_nd_values!($tensor, indices, $values, $update, $kind, $variant)
            }
            DynInt::R3(indices) => {
                scatter_nd_values!($tensor, indices, $values, $update, $kind, $variant)
            }
            DynInt::R4(indices) => {
                scatter_nd_values!($tensor, indices, $values, $update, $kind, $variant)
            }
            DynInt::R5(indices) => {
                scatter_nd_values!($tensor, indices, $values, $update, $kind, $variant)
            }
            DynInt::R6(indices) => {
                scatter_nd_values!($tensor, indices, $values, $update, $kind, $variant)
            }
        }
    };
}

macro_rules! scatter_nd_dyn {
    ($tensor:expr, $indices:expr, $values:expr, $update:expr, $kind:ident) => {
        match $tensor {
            $kind::R1(tensor) => {
                scatter_nd_indices!(tensor, $indices, $values, $update, $kind, R1)
            }
            $kind::R2(tensor) => {
                scatter_nd_indices!(tensor, $indices, $values, $update, $kind, R2)
            }
            $kind::R3(tensor) => {
                scatter_nd_indices!(tensor, $indices, $values, $update, $kind, R3)
            }
            $kind::R4(tensor) => {
                scatter_nd_indices!(tensor, $indices, $values, $update, $kind, R4)
            }
            $kind::R5(tensor) => {
                scatter_nd_indices!(tensor, $indices, $values, $update, $kind, R5)
            }
            $kind::R6(tensor) => {
                scatter_nd_indices!(tensor, $indices, $values, $update, $kind, R6)
            }
        }
    };
}

impl_concat!(DynTensor);
impl_concat!(DynInt);
impl_concat!(DynBool);

impl DynTensor {
    /// Create an uninitialized floating-point tensor with an explicit shape and dtype.
    pub fn empty(dims: &[usize], device: &Device, dtype: DType) -> Result<Self> {
        Ok(match dims {
            [d0] => Self::R1(Tensor::<1>::empty([*d0], (device, dtype))),
            [d0, d1] => Self::R2(Tensor::<2>::empty([*d0, *d1], (device, dtype))),
            [d0, d1, d2] => Self::R3(Tensor::<3>::empty([*d0, *d1, *d2], (device, dtype))),
            [d0, d1, d2, d3] => Self::R4(Tensor::<4>::empty([*d0, *d1, *d2, *d3], (device, dtype))),
            [d0, d1, d2, d3, d4] => Self::R5(Tensor::<5>::empty(
                [*d0, *d1, *d2, *d3, *d4],
                (device, dtype),
            )),
            [d0, d1, d2, d3, d4, d5] => Self::R6(Tensor::<6>::empty(
                [*d0, *d1, *d2, *d3, *d4, *d5],
                (device, dtype),
            )),
            _ => return Err(rank_overflow(dims.len())),
        })
    }

    /// Apply an NCHW two-dimensional convolution with symmetric padding.
    pub fn conv2d(
        self,
        weight: Self,
        bias: Option<Self>,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Self> {
        let input_shape = self.dims();
        let weight_shape = weight.dims();
        if input_shape.len() != 4 || weight_shape.len() != 4 {
            return Err(TynxError::Shape(format!(
                "conv2d requires rank-4 NCHW input and weight, got {input_shape:?} and {weight_shape:?}"
            )));
        }
        if groups == 0 {
            return Err(TynxError::Shape(
                "conv2d groups must be a positive integer".to_string(),
            ));
        }
        if stride.contains(&0) || dilation.contains(&0) {
            return Err(TynxError::Shape(
                "conv2d stride and dilation must contain positive integers".to_string(),
            ));
        }
        let input_channels = input_shape[1];
        let output_channels = weight_shape[0];
        if !input_channels.is_multiple_of(groups) || !output_channels.is_multiple_of(groups) {
            return Err(TynxError::Shape(format!(
                "conv2d input channels ({input_channels}) and output channels ({output_channels}) must be divisible by groups ({groups})"
            )));
        }
        if weight_shape[1] != input_channels / groups {
            return Err(TynxError::Shape(format!(
                "conv2d expected weight input channels {}, got shape {weight_shape:?}",
                input_channels / groups
            )));
        }
        for axis in 0..2 {
            let kernel = weight_shape[axis + 2];
            if kernel == 0 {
                return Err(TynxError::Shape(
                    "conv2d kernel dimensions must be positive".to_string(),
                ));
            }
            let effective_kernel = dilation[axis]
                .checked_mul(kernel - 1)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| TynxError::Shape("conv2d kernel extent overflowed".to_string()))?;
            let padded_input = padding[axis]
                .checked_mul(2)
                .and_then(|value| input_shape[axis + 2].checked_add(value))
                .ok_or_else(|| TynxError::Shape("conv2d padded extent overflowed".to_string()))?;
            if padded_input < effective_kernel {
                return Err(TynxError::Shape(format!(
                    "conv2d kernel extent {effective_kernel} exceeds padded input extent {padded_input} at spatial axis {axis}"
                )));
            }
        }
        let input_dtype = self.dtype();
        if input_dtype != weight.dtype() {
            return Err(TynxError::Shape(format!(
                "conv2d input and weight dtypes must match, got {:?} and {:?}",
                input_dtype,
                weight.dtype()
            )));
        }

        let bias = match bias {
            Some(Self::R1(bias))
                if bias.dims()[0] == output_channels && bias.dtype() == input_dtype =>
            {
                Some(bias)
            }
            Some(bias) => {
                return Err(TynxError::Shape(format!(
                    "conv2d bias must have shape [{output_channels}] and dtype {input_dtype:?}, got {:?} and {:?}",
                    bias.dims(),
                    bias.dtype()
                )));
            }
            None => None,
        };
        match (self, weight) {
            (Self::R4(input), Self::R4(weight)) => Ok(Self::R4(burn_conv2d(
                input,
                weight,
                bias,
                ConvOptions::new(stride, padding, dilation, groups),
            ))),
            _ => unreachable!("conv2d ranks were validated before dispatch"),
        }
    }

    /// Apply rank-4 NCHW max pooling.
    pub fn max_pool2d(
        self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> Result<Self> {
        validate_pool2d(
            &self.dims(),
            kernel_size,
            stride,
            padding,
            dilation,
            "max_pool2d",
        )?;
        match self {
            Self::R4(input) => Ok(Self::R4(burn_max_pool2d(
                input,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            ))),
            _ => unreachable!("pooling rank was validated before dispatch"),
        }
    }

    /// Apply rank-4 NCHW average pooling.
    pub fn avg_pool2d(
        self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> Result<Self> {
        validate_pool2d(
            &self.dims(),
            kernel_size,
            stride,
            padding,
            [1, 1],
            "avg_pool2d",
        )?;
        match self {
            Self::R4(input) => Ok(Self::R4(burn_avg_pool2d(
                input,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            ))),
            _ => unreachable!("pooling rank was validated before dispatch"),
        }
    }

    /// Apply rank-4 NCHW adaptive average pooling.
    pub fn adaptive_avg_pool2d(self, output_size: [usize; 2]) -> Result<Self> {
        let dims = self.dims();
        if dims.len() != 4 {
            return Err(TynxError::Shape(format!(
                "adaptive_avg_pool2d requires rank-4 NCHW input, got {dims:?}"
            )));
        }
        if output_size.contains(&0) {
            return Err(TynxError::Shape(
                "adaptive_avg_pool2d output dimensions must be positive".to_string(),
            ));
        }
        match self {
            Self::R4(input) => Ok(Self::R4(burn_adaptive_avg_pool2d(input, output_size))),
            _ => unreachable!("pooling rank was validated before dispatch"),
        }
    }

    /// Create a random floating-point tensor with an explicit shape and dtype.
    pub fn random(
        dims: &[usize],
        distribution: Distribution,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Ok(match dims {
            [d0] => Self::R1(Tensor::<1>::random([*d0], distribution, (device, dtype))),
            [d0, d1] => Self::R2(Tensor::<2>::random(
                [*d0, *d1],
                distribution,
                (device, dtype),
            )),
            [d0, d1, d2] => Self::R3(Tensor::<3>::random(
                [*d0, *d1, *d2],
                distribution,
                (device, dtype),
            )),
            [d0, d1, d2, d3] => Self::R4(Tensor::<4>::random(
                [*d0, *d1, *d2, *d3],
                distribution,
                (device, dtype),
            )),
            [d0, d1, d2, d3, d4] => Self::R5(Tensor::<5>::random(
                [*d0, *d1, *d2, *d3, *d4],
                distribution,
                (device, dtype),
            )),
            [d0, d1, d2, d3, d4, d5] => Self::R6(Tensor::<6>::random(
                [*d0, *d1, *d2, *d3, *d4, *d5],
                distribution,
                (device, dtype),
            )),
            _ => return Err(rank_overflow(dims.len())),
        })
    }

    /// Create a floating-point tensor filled with one value and an explicit dtype.
    pub fn full(dims: &[usize], value: f64, device: &Device, dtype: DType) -> Result<Self> {
        Ok(match dims {
            [d0] => Self::R1(Tensor::<1>::full([*d0], value, (device, dtype))),
            [d0, d1] => Self::R2(Tensor::<2>::full([*d0, *d1], value, (device, dtype))),
            [d0, d1, d2] => Self::R3(Tensor::<3>::full([*d0, *d1, *d2], value, (device, dtype))),
            [d0, d1, d2, d3] => Self::R4(Tensor::<4>::full(
                [*d0, *d1, *d2, *d3],
                value,
                (device, dtype),
            )),
            [d0, d1, d2, d3, d4] => Self::R5(Tensor::<5>::full(
                [*d0, *d1, *d2, *d3, *d4],
                value,
                (device, dtype),
            )),
            [d0, d1, d2, d3, d4, d5] => Self::R6(Tensor::<6>::full(
                [*d0, *d1, *d2, *d3, *d4, *d5],
                value,
                (device, dtype),
            )),
            _ => return Err(rank_overflow(dims.len())),
        })
    }

    /// Reshape the tensor while preserving its elements and dtype.
    pub fn reshape(self, dims: Vec<usize>) -> Result<Self> {
        Ok(match self {
            Self::R1(tensor) => reshape_dyn!(tensor, dims, DynTensor),
            Self::R2(tensor) => reshape_dyn!(tensor, dims, DynTensor),
            Self::R3(tensor) => reshape_dyn!(tensor, dims, DynTensor),
            Self::R4(tensor) => reshape_dyn!(tensor, dims, DynTensor),
            Self::R5(tensor) => reshape_dyn!(tensor, dims, DynTensor),
            Self::R6(tensor) => reshape_dyn!(tensor, dims, DynTensor),
        })
    }

    /// Expand singleton dimensions to the requested shape.
    pub fn expand(self, dims: &[usize]) -> Result<Self> {
        if dims.len() != self.rank() {
            return Err(TynxError::Shape(format!(
                "expand shape has rank {}, expected {}",
                dims.len(),
                self.rank()
            )));
        }
        Ok(match self {
            Self::R1(tensor) => Self::R1(tensor.expand([dims[0] as i64])),
            Self::R2(tensor) => Self::R2(tensor.expand([dims[0] as i64, dims[1] as i64])),
            Self::R3(tensor) => {
                Self::R3(tensor.expand([dims[0] as i64, dims[1] as i64, dims[2] as i64]))
            }
            Self::R4(tensor) => Self::R4(tensor.expand([
                dims[0] as i64,
                dims[1] as i64,
                dims[2] as i64,
                dims[3] as i64,
            ])),
            Self::R5(tensor) => Self::R5(tensor.expand([
                dims[0] as i64,
                dims[1] as i64,
                dims[2] as i64,
                dims[3] as i64,
                dims[4] as i64,
            ])),
            Self::R6(tensor) => Self::R6(tensor.expand([
                dims[0] as i64,
                dims[1] as i64,
                dims[2] as i64,
                dims[3] as i64,
                dims[4] as i64,
                dims[5] as i64,
            ])),
        })
    }

    /// Tile each dimension by the corresponding repeat count.
    pub fn repeat(self, repeats: &[usize]) -> Self {
        map_float!(self, |tensor| tensor.repeat(repeats))
    }

    /// Slice the tensor with one slice per dimension.
    pub fn slice(self, slices: &[Slice]) -> Self {
        map_float!(self, |tensor| tensor.slice(slices))
    }

    /// Assign a same-rank tensor into a slice.
    pub fn slice_assign(self, slices: &[Slice], values: Self) -> Result<Self> {
        Ok(slice_assign_dyn!(self, slices, values, DynTensor))
    }

    /// Reverse one dimension.
    pub fn flip_dim(self, dim: usize) -> Self {
        map_float!(self, |tensor| tensor.flip([dim as isize]))
    }

    /// Select slices along one dimension using flattened indices.
    pub fn select(self, dim: usize, indices: DynInt) -> Result<Self> {
        Ok(select_dyn!(self, dim, indices, DynTensor))
    }

    /// Gather elements from a same-rank index tensor.
    pub fn gather(self, dim: usize, indices: DynInt) -> Result<Self> {
        let slices = gather_slices(&self.dims(), &indices.dims(), dim)?;
        Ok(gather_dyn!(self.slice(&slices), dim, indices, DynTensor))
    }

    /// Gather slices using tuples stored in the last index dimension.
    pub fn gather_nd(self, indices: DynInt, output_rank: usize) -> Result<Self> {
        Ok(gather_nd_dyn!(self, indices, output_rank, DynTensor))
    }

    /// Scatter same-rank updates along one dimension.
    pub fn scatter(
        self,
        dim: usize,
        indices: DynInt,
        values: Self,
        update: IndexingUpdateOp,
    ) -> Result<Self> {
        Ok(scatter_dyn!(self, dim, indices, values, update, DynTensor))
    }

    /// Scatter updates using tuples stored in the last index dimension.
    pub fn scatter_nd(
        self,
        indices: DynInt,
        values: Self,
        update: IndexingUpdateOp,
    ) -> Result<Self> {
        Ok(scatter_nd_dyn!(self, indices, values, update, DynTensor))
    }

    /// Compute a cumulative sum with ONNX exclusive and reverse semantics.
    pub fn cumsum(self, dim: usize, exclusive: bool, reverse: bool) -> Self {
        map_float!(self, |tensor| {
            let input = if reverse {
                tensor.flip([dim as isize])
            } else {
                tensor
            };
            let output = input.clone().cumsum(dim);
            let output = if exclusive { output.sub(input) } else { output };
            if reverse {
                output.flip([dim as isize])
            } else {
                output
            }
        })
    }

    /// Return the largest values and their indices along one dimension.
    pub fn topk(self, k: usize, dim: usize) -> (Self, DynInt) {
        match self {
            Self::R1(tensor) => {
                let (values, indices) = tensor.topk_with_indices(k, dim);
                (Self::R1(values), DynInt::R1(indices.cast(DType::I64)))
            }
            Self::R2(tensor) => {
                let (values, indices) = tensor.topk_with_indices(k, dim);
                (Self::R2(values), DynInt::R2(indices.cast(DType::I64)))
            }
            Self::R3(tensor) => {
                let (values, indices) = tensor.topk_with_indices(k, dim);
                (Self::R3(values), DynInt::R3(indices.cast(DType::I64)))
            }
            Self::R4(tensor) => {
                let (values, indices) = tensor.topk_with_indices(k, dim);
                (Self::R4(values), DynInt::R4(indices.cast(DType::I64)))
            }
            Self::R5(tensor) => {
                let (values, indices) = tensor.topk_with_indices(k, dim);
                (Self::R5(values), DynInt::R5(indices.cast(DType::I64)))
            }
            Self::R6(tensor) => {
                let (values, indices) = tensor.topk_with_indices(k, dim);
                (Self::R6(values), DynInt::R6(indices.cast(DType::I64)))
            }
        }
    }

    /// Permute the tensor dimensions.
    pub fn permute(self, axes: Vec<usize>) -> Result<Self> {
        if axes.len() != self.rank() {
            return Err(TynxError::Shape(format!(
                "permutation has {} axes for rank {}",
                axes.len(),
                self.rank()
            )));
        }
        Ok(match self {
            Self::R1(tensor) => Self::R1(tensor.permute([axes[0]])),
            Self::R2(tensor) => Self::R2(tensor.permute([axes[0], axes[1]])),
            Self::R3(tensor) => Self::R3(tensor.permute([axes[0], axes[1], axes[2]])),
            Self::R4(tensor) => Self::R4(tensor.permute([axes[0], axes[1], axes[2], axes[3]])),
            Self::R5(tensor) => {
                Self::R5(tensor.permute([axes[0], axes[1], axes[2], axes[3], axes[4]]))
            }
            Self::R6(tensor) => {
                Self::R6(tensor.permute([axes[0], axes[1], axes[2], axes[3], axes[4], axes[5]]))
            }
        })
    }

    /// Promote the tensor by adding leading singleton dimensions.
    pub fn to_rank(self, target: usize) -> Result<Self> {
        let current = self.rank();
        if current == target {
            return Ok(self);
        }
        if current > target || target > MAX_RANK {
            return Err(TynxError::RankPromote {
                from: current,
                to: target,
            });
        }

        Ok(match (self, target) {
            (Self::R1(tensor), 2) => Self::R2(tensor.unsqueeze()),
            (Self::R1(tensor), 3) => Self::R3(tensor.unsqueeze()),
            (Self::R1(tensor), 4) => Self::R4(tensor.unsqueeze()),
            (Self::R1(tensor), 5) => Self::R5(tensor.unsqueeze()),
            (Self::R1(tensor), 6) => Self::R6(tensor.unsqueeze()),
            (Self::R2(tensor), 3) => Self::R3(tensor.unsqueeze()),
            (Self::R2(tensor), 4) => Self::R4(tensor.unsqueeze()),
            (Self::R2(tensor), 5) => Self::R5(tensor.unsqueeze()),
            (Self::R2(tensor), 6) => Self::R6(tensor.unsqueeze()),
            (Self::R3(tensor), 4) => Self::R4(tensor.unsqueeze()),
            (Self::R3(tensor), 5) => Self::R5(tensor.unsqueeze()),
            (Self::R3(tensor), 6) => Self::R6(tensor.unsqueeze()),
            (Self::R4(tensor), 5) => Self::R5(tensor.unsqueeze()),
            (Self::R4(tensor), 6) => Self::R6(tensor.unsqueeze()),
            (Self::R5(tensor), 6) => Self::R6(tensor.unsqueeze()),
            (_, target) => return Err(rank_overflow(target)),
        })
    }

    /// Add two tensors using ONNX-style multidirectional broadcasting.
    pub fn add_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;

        Ok(zip_float!(left, right, |left, right| left.add(right)))
    }

    /// Subtract two tensors using ONNX-style multidirectional broadcasting.
    pub fn sub_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;

        Ok(zip_float!(left, right, |left, right| left.sub(right)))
    }

    /// Multiply two tensors using ONNX-style multidirectional broadcasting.
    pub fn mul_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;

        Ok(zip_float!(left, right, |left, right| left.mul(right)))
    }

    /// Divide two tensors using ONNX-style multidirectional broadcasting.
    pub fn div_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;

        Ok(zip_float!(left, right, |left, right| left.div(right)))
    }

    /// Apply floating-point modulo semantics with multidirectional broadcasting.
    pub fn modulo_broadcast(self, other: Self, fmod: bool) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_float!(left, right, |left, right| {
            if fmod {
                left.fmod(right)
            } else {
                let remainder = left.remainder(right.clone());
                remainder.add(right.clone()).remainder(right)
            }
        }))
    }

    /// Take the element-wise maximum with multidirectional broadcasting.
    pub fn max_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_float!(left, right, |left, right| left.max_pair(right)))
    }

    /// Take the element-wise minimum with multidirectional broadcasting.
    pub fn min_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_float!(left, right, |left, right| left.min_pair(right)))
    }

    /// Divide every element by a scalar.
    pub fn div_scalar(self, divisor: f64) -> Self {
        map_float!(self, |tensor| tensor.div_scalar(divisor))
    }

    /// Multiply every element by a scalar.
    pub fn mul_scalar(self, multiplier: f64) -> Self {
        map_float!(self, |tensor| tensor.mul_scalar(multiplier))
    }

    /// Add a scalar to every element.
    pub fn add_scalar(self, value: f64) -> Self {
        map_float!(self, |tensor| tensor.add_scalar(value))
    }

    /// Subtract a scalar from every element.
    pub fn sub_scalar(self, value: f64) -> Self {
        map_float!(self, |tensor| tensor.sub_scalar(value))
    }

    /// Create a tensor with the same shape, device, and dtype filled with one scalar.
    pub fn full_like(self, value: f64) -> Self {
        map_float!(self, |tensor| tensor.full_like(value))
    }

    /// Multiply matrices or batches of matrices with matching runtime ranks.
    pub fn matmul(self, other: Self) -> Result<Self> {
        validate_matmul_shapes(&self.dims(), &other.dims())?;
        Ok(match (self, other) {
            (Self::R2(left), Self::R2(right)) => Self::R2(left.matmul(right)),
            (Self::R3(left), Self::R3(right)) => Self::R3(left.matmul(right)),
            (Self::R4(left), Self::R4(right)) => Self::R4(left.matmul(right)),
            (Self::R5(left), Self::R5(right)) => Self::R5(left.matmul(right)),
            (Self::R6(left), Self::R6(right)) => Self::R6(left.matmul(right)),
            (left, right) => {
                return Err(TynxError::Shape(format!(
                    "matmul requires matching ranks >= 2, got {} and {}",
                    left.rank(),
                    right.rank()
                )));
            }
        })
    }

    /// Sum elements along dimensions while retaining singleton dimensions.
    pub fn sum_dims(self, dims: &[usize]) -> Self {
        map_float!(self, |tensor| tensor.sum_dims(dims))
    }

    /// Average elements along dimensions while retaining singleton dimensions.
    pub fn mean_dims(self, dims: &[usize]) -> Self {
        map_float!(self, |tensor| tensor.mean_dims(dims))
    }

    /// Multiply elements along dimensions while retaining singleton dimensions.
    pub fn prod_dims(self, dims: &[usize]) -> Self {
        map_float!(self, |tensor| tensor.prod_dims(dims))
    }

    /// Take the maximum along dimensions while retaining singleton dimensions.
    pub fn reduce_max_dims(self, dims: &[usize]) -> Self {
        map_float!(self, |tensor| tensor.max_dims(dims))
    }

    /// Take the minimum along dimensions while retaining singleton dimensions.
    pub fn reduce_min_dims(self, dims: &[usize]) -> Self {
        map_float!(self, |tensor| tensor.min_dims(dims))
    }

    /// Take the maximum while propagating NaN if any reduced element is NaN.
    pub fn reduce_max_dims_nan(self, dims: &[usize]) -> Result<Self> {
        let nan_mask = self.clone().is_nan().reduce_max_dims(dims);
        let reduced = self.reduce_max_dims(dims);
        Self::where_select(nan_mask, reduced.clone().full_like(f64::NAN), reduced)
    }

    /// Take the minimum while propagating NaN if any reduced element is NaN.
    pub fn reduce_min_dims_nan(self, dims: &[usize]) -> Result<Self> {
        let nan_mask = self.clone().is_nan().reduce_max_dims(dims);
        let reduced = self.reduce_min_dims(dims);
        Self::where_select(nan_mask, reduced.clone().full_like(f64::NAN), reduced)
    }

    /// Compare two tensors for equality with multidirectional broadcasting.
    pub fn equal_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_float_bool!(left, right, |left, right| left.equal(right)))
    }

    /// Compare every element with a scalar for equality.
    pub fn equal_scalar(self, other: f64) -> DynBool {
        map_float_bool!(self, |tensor| tensor.equal_elem(other))
    }

    /// Compare two tensors using greater-than with multidirectional broadcasting.
    pub fn greater_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_float_bool!(left, right, |left, right| left.greater(right)))
    }

    /// Compare every element with a scalar using greater-than.
    pub fn greater_scalar(self, other: f64) -> DynBool {
        map_float_bool!(self, |tensor| tensor.greater_elem(other))
    }

    /// Compare two tensors using greater-or-equal with multidirectional broadcasting.
    pub fn greater_equal_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_float_bool!(left, right, |left, right| left.greater_equal(right)))
    }

    /// Compare every element with a scalar using greater-or-equal.
    pub fn greater_equal_scalar(self, other: f64) -> DynBool {
        map_float_bool!(self, |tensor| tensor.greater_equal_elem(other))
    }

    /// Compare two tensors using less-than with multidirectional broadcasting.
    pub fn less_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_float_bool!(left, right, |left, right| left.lower(right)))
    }

    /// Compare every element with a scalar using less-than.
    pub fn less_scalar(self, other: f64) -> DynBool {
        map_float_bool!(self, |tensor| tensor.lower_elem(other))
    }

    /// Compare two tensors using less-or-equal with multidirectional broadcasting.
    pub fn less_equal_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_float_bool!(left, right, |left, right| left.lower_equal(right)))
    }

    /// Compare every element with a scalar using less-or-equal.
    pub fn less_equal_scalar(self, other: f64) -> DynBool {
        map_float_bool!(self, |tensor| tensor.lower_equal_elem(other))
    }

    /// Apply parametric rectified linear unit with a broadcastable slope tensor.
    pub fn prelu(self, slope: Self) -> Result<Self> {
        let input_dims = self.dims();
        let slope_dims = slope.dims();
        let input_rank = self.rank();
        if slope.rank() > input_rank {
            return Err(TynxError::Shape(format!(
                "PRelu slope rank {} exceeds input rank {input_rank}",
                slope.rank()
            )));
        }
        if broadcast_shape(&input_dims, &slope_dims)? != input_dims {
            return Err(TynxError::Shape(format!(
                "PRelu slope shape {slope_dims:?} cannot broadcast to input shape {input_dims:?}"
            )));
        }
        let slope = slope.to_rank(input_rank)?;

        Ok(zip_float!(self, slope, |input, slope| {
            input
                .clone()
                .clamp_min(0.0)
                .add(slope.mul(input.clamp_max(0.0)))
        }))
    }

    /// Raise every element to a broadcastable floating-point exponent.
    pub fn powf_broadcast(self, exponent: Self) -> Result<Self> {
        let dtype = self.dtype();
        let (base, exponent) = Self::broadcast_pair(self, exponent.cast(dtype))?;
        Ok(zip_float!(base, exponent, |base, exponent| base.powf(exponent)))
    }

    /// Raise every element to a floating-point scalar exponent.
    pub fn powf_scalar(self, exponent: f64) -> Self {
        map_float!(self, |tensor| tensor.powf_scalar(exponent))
    }

    /// Raise every element to an integer scalar exponent.
    pub fn powi_scalar(self, exponent: i64) -> Self {
        map_float!(self, |tensor| tensor.powi_scalar(exponent))
    }

    /// Return the tensor's element type.
    pub fn dtype(&self) -> DType {
        match self {
            Self::R1(tensor) => tensor.dtype(),
            Self::R2(tensor) => tensor.dtype(),
            Self::R3(tensor) => tensor.dtype(),
            Self::R4(tensor) => tensor.dtype(),
            Self::R5(tensor) => tensor.dtype(),
            Self::R6(tensor) => tensor.dtype(),
        }
    }

    /// Cast the tensor while preserving its rank.
    pub fn cast(self, dtype: DType) -> Self {
        map_float!(self, |tensor| tensor.cast(dtype))
    }

    /// Convert the tensor to an integer tensor with an explicit dtype.
    pub fn to_int(self, dtype: DType) -> DynInt {
        match self {
            Self::R1(tensor) => DynInt::R1(tensor.int().cast(dtype)),
            Self::R2(tensor) => DynInt::R2(tensor.int().cast(dtype)),
            Self::R3(tensor) => DynInt::R3(tensor.int().cast(dtype)),
            Self::R4(tensor) => DynInt::R4(tensor.int().cast(dtype)),
            Self::R5(tensor) => DynInt::R5(tensor.int().cast(dtype)),
            Self::R6(tensor) => DynInt::R6(tensor.int().cast(dtype)),
        }
    }

    /// Convert the tensor to booleans using nonzero truth semantics.
    pub fn to_bool(self) -> DynBool {
        match self {
            Self::R1(tensor) => DynBool::R1(tensor.bool()),
            Self::R2(tensor) => DynBool::R2(tensor.bool()),
            Self::R3(tensor) => DynBool::R3(tensor.bool()),
            Self::R4(tensor) => DynBool::R4(tensor.bool()),
            Self::R5(tensor) => DynBool::R5(tensor.bool()),
            Self::R6(tensor) => DynBool::R6(tensor.bool()),
        }
    }

    fn broadcast_pair(left: Self, right: Self) -> Result<(Self, Self)> {
        broadcast_shape(&left.dims(), &right.dims())?;
        let rank = left.rank().max(right.rank());
        Ok((left.to_rank(rank)?, right.to_rank(rank)?))
    }

    /// Apply rectified linear unit element-wise.
    pub fn relu(self) -> Self {
        map_float!(self, |tensor| activation::relu(tensor))
    }

    /// Apply the sigmoid function element-wise.
    pub fn sigmoid(self) -> Self {
        map_float!(self, |tensor| activation::sigmoid(tensor))
    }

    /// Apply the hyperbolic tangent function element-wise.
    pub fn tanh(self) -> Self {
        map_float!(self, |tensor| activation::tanh(tensor))
    }

    /// Apply the exponential function element-wise.
    pub fn exp(self) -> Self {
        map_float!(self, |tensor| tensor.exp())
    }

    /// Apply the natural logarithm element-wise.
    pub fn log(self) -> Self {
        map_float!(self, |tensor| tensor.log())
    }

    /// Apply the square root function element-wise.
    pub fn sqrt(self) -> Self {
        map_float!(self, |tensor| tensor.sqrt())
    }

    /// Apply the absolute value function element-wise.
    pub fn abs(self) -> Self {
        map_float!(self, |tensor| tensor.abs())
    }

    /// Negate each tensor element.
    pub fn negated(self) -> Self {
        map_float!(self, |tensor| tensor.neg())
    }

    /// Apply the sine function element-wise.
    pub fn sin(self) -> Self {
        map_float!(self, |tensor| tensor.sin())
    }

    /// Apply the cosine function element-wise.
    pub fn cos(self) -> Self {
        map_float!(self, |tensor| tensor.cos())
    }

    /// Apply the tangent function element-wise.
    pub fn tan(self) -> Self {
        map_float!(self, |tensor| tensor.tan())
    }

    /// Apply the hyperbolic cosine function element-wise.
    pub fn cosh(self) -> Self {
        map_float!(self, |tensor| tensor.cosh())
    }

    /// Apply the hyperbolic sine function element-wise.
    pub fn sinh(self) -> Self {
        map_float!(self, |tensor| tensor.sinh())
    }

    /// Apply the inverse cosine function element-wise.
    pub fn acos(self) -> Self {
        map_float!(self, |tensor| tensor.acos())
    }

    /// Apply the inverse hyperbolic cosine function element-wise.
    pub fn acosh(self) -> Self {
        map_float!(self, |tensor| tensor.acosh())
    }

    /// Apply the inverse sine function element-wise.
    pub fn asin(self) -> Self {
        map_float!(self, |tensor| tensor.asin())
    }

    /// Apply the inverse hyperbolic sine function element-wise.
    pub fn asinh(self) -> Self {
        map_float!(self, |tensor| tensor.asinh())
    }

    /// Apply the inverse tangent function element-wise.
    pub fn atan(self) -> Self {
        map_float!(self, |tensor| tensor.atan())
    }

    /// Apply the inverse hyperbolic tangent function element-wise.
    pub fn atanh(self) -> Self {
        map_float!(self, |tensor| tensor.atanh())
    }

    /// Apply the error function element-wise.
    pub fn erf(self) -> Self {
        map_float!(self, |tensor| tensor.erf())
    }

    /// Round each element toward positive infinity.
    pub fn ceil(self) -> Self {
        map_float!(self, |tensor| tensor.ceil())
    }

    /// Round each element toward negative infinity.
    pub fn floor(self) -> Self {
        map_float!(self, |tensor| tensor.floor())
    }

    /// Round each element to the nearest integer, with ties to even.
    pub fn round(self) -> Self {
        map_float!(self, |tensor| tensor.round())
    }

    /// Replace each element with its multiplicative inverse.
    pub fn reciprocal(self) -> Self {
        map_float!(self, |tensor| tensor.recip())
    }

    /// Return the sign of each tensor element.
    pub fn sign(self) -> Self {
        map_float!(self, |tensor| tensor.sign())
    }

    /// Test each element for infinity, optionally filtering by sign.
    pub fn is_inf_signed(self, detect_negative: bool, detect_positive: bool) -> DynBool {
        macro_rules! classify {
            ($tensor:expr, $variant:path) => {{
                let tensor = $tensor;
                let infinite = tensor.clone().is_inf();
                let output = match (detect_negative, detect_positive) {
                    (true, true) => infinite,
                    (true, false) => infinite.bool_and(tensor.lower_elem(0.0)),
                    (false, true) => infinite.bool_and(tensor.greater_elem(0.0)),
                    (false, false) => infinite.clone().bool_and(infinite.bool_not()),
                };
                $variant(output)
            }};
        }

        match self {
            Self::R1(tensor) => classify!(tensor, DynBool::R1),
            Self::R2(tensor) => classify!(tensor, DynBool::R2),
            Self::R3(tensor) => classify!(tensor, DynBool::R3),
            Self::R4(tensor) => classify!(tensor, DynBool::R4),
            Self::R5(tensor) => classify!(tensor, DynBool::R5),
            Self::R6(tensor) => classify!(tensor, DynBool::R6),
        }
    }

    /// Test each element for NaN.
    pub fn is_nan(self) -> DynBool {
        match self {
            Self::R1(tensor) => DynBool::R1(tensor.is_nan()),
            Self::R2(tensor) => DynBool::R2(tensor.is_nan()),
            Self::R3(tensor) => DynBool::R3(tensor.is_nan()),
            Self::R4(tensor) => DynBool::R4(tensor.is_nan()),
            Self::R5(tensor) => DynBool::R5(tensor.is_nan()),
            Self::R6(tensor) => DynBool::R6(tensor.is_nan()),
        }
    }

    /// Apply the softplus function element-wise.
    pub fn softplus(self) -> Self {
        map_float!(self, |tensor| activation::softplus(tensor, 1.0))
    }

    /// Apply the exponential linear unit function element-wise.
    pub fn elu(self, alpha: f64) -> Self {
        map_float!(self, |tensor| activation::elu(tensor, alpha))
    }

    /// Apply the leaky rectified linear unit function element-wise.
    pub fn leaky_relu(self, alpha: f64) -> Self {
        map_float!(self, |tensor| activation::leaky_relu(tensor, alpha))
    }

    /// Apply the scaled exponential linear unit function element-wise.
    pub fn selu(self, alpha: f64, gamma: f64) -> Self {
        map_float!(self, |tensor| activation::elu(tensor, alpha)
            .mul_scalar(gamma))
    }

    /// Apply the softsign function element-wise.
    pub fn softsign(self) -> Self {
        map_float!(self, |tensor| activation::softsign(tensor))
    }

    /// Apply the hard sigmoid function element-wise.
    pub fn hard_sigmoid(self, alpha: f64, beta: f64) -> Self {
        map_float!(self, |tensor| activation::hard_sigmoid(tensor, alpha, beta))
    }

    /// Apply thresholded rectified linear unit element-wise.
    pub fn thresholded_relu(self, alpha: f64) -> Self {
        map_float!(self, |tensor| activation::thresholded_relu(tensor, alpha))
    }

    /// Apply the continuously differentiable exponential linear unit function element-wise.
    pub fn celu(self, alpha: f64) -> Self {
        map_float!(self, |tensor| activation::celu(tensor, alpha))
    }

    /// Apply element-wise shrinkage outside the configured dead zone.
    pub fn shrink(self, lambda: f64, bias: f64) -> Self {
        map_float!(self, |tensor| activation::shrink(tensor, lambda, bias))
    }

    /// Apply the Swish activation with an explicit alpha coefficient.
    pub fn swish(self, alpha: f64) -> Self {
        map_float!(self, |tensor| {
            let sigmoid = activation::sigmoid(tensor.clone().mul_scalar(alpha));
            tensor.mul(sigmoid)
        })
    }

    /// Apply the Mish activation function element-wise.
    pub fn mish(self) -> Self {
        map_float!(self, |tensor| activation::mish(tensor))
    }

    /// Apply the hard swish activation function element-wise.
    pub fn hard_swish(self) -> Self {
        map_float!(self, |tensor| activation::hard_swish(tensor))
    }

    /// Apply the Gaussian error linear unit function element-wise.
    pub fn gelu(self) -> Self {
        map_float!(self, |tensor| activation::gelu(tensor))
    }

    /// Apply softmax along one dimension.
    pub fn softmax(self, dim: usize) -> Self {
        map_float!(self, |tensor| activation::softmax(tensor, dim))
    }

    /// Apply log-softmax along one dimension.
    pub fn log_softmax(self, dim: usize) -> Self {
        map_float!(self, |tensor| activation::log_softmax(tensor, dim))
    }

    /// Return indices of extrema along one dimension using ONNX tie semantics.
    pub fn arg_extreme(self, dim: usize, maximum: bool, select_last: bool) -> DynInt {
        macro_rules! apply {
            ($tensor:expr) => {{
                let tensor = $tensor;
                let nan_mask = tensor.clone().is_nan();
                let nan_value = if maximum {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                };
                let tensor = tensor
                    .clone()
                    .mask_where(nan_mask, tensor.full_like(nan_value));
                let axis_size = tensor.dims()[dim] as i64;
                let indices = if maximum {
                    if select_last {
                        tensor
                            .flip([dim as isize])
                            .argmax(dim)
                            .mul_scalar(-1_i64)
                            .add_scalar(axis_size - 1)
                    } else {
                        tensor.argmax(dim)
                    }
                } else if select_last {
                    tensor
                        .flip([dim as isize])
                        .argmin(dim)
                        .mul_scalar(-1_i64)
                        .add_scalar(axis_size - 1)
                } else {
                    tensor.argmin(dim)
                };
                indices.cast(DType::I64)
            }};
        }

        match self {
            Self::R1(tensor) => DynInt::R1(apply!(tensor)),
            Self::R2(tensor) => DynInt::R2(apply!(tensor)),
            Self::R3(tensor) => DynInt::R3(apply!(tensor)),
            Self::R4(tensor) => DynInt::R4(apply!(tensor)),
            Self::R5(tensor) => DynInt::R5(apply!(tensor)),
            Self::R6(tensor) => DynInt::R6(apply!(tensor)),
        }
    }

    /// Clamp every element to the optional lower and upper bounds.
    pub fn clip(self, min: Option<f64>, max: Option<f64>) -> Self {
        match (min, max) {
            (Some(min), Some(max)) => {
                map_float!(self, |tensor| tensor.clamp_min(min).clamp_max(max))
            }
            (Some(min), None) => map_float!(self, |tensor| tensor.clamp_min(min)),
            (None, Some(max)) => map_float!(self, |tensor| tensor.clamp_max(max)),
            (None, None) => self,
        }
    }

    /// Select elements from two tensors using a broadcastable boolean condition.
    pub fn where_select(condition: DynBool, then: Self, otherwise: Self) -> Result<Self> {
        let output_shape = broadcast_shape(&condition.dims(), &then.dims())?;
        broadcast_shape(&output_shape, &otherwise.dims())?;
        let rank = condition.rank().max(then.rank()).max(otherwise.rank());
        let dtype = then.dtype();
        Ok(where_float!(
            condition.to_rank(rank)?,
            then.to_rank(rank)?,
            otherwise.cast(dtype).to_rank(rank)?
        ))
    }
}

impl DynInt {
    /// Create an uninitialized integer tensor with an explicit shape and dtype.
    pub fn empty(dims: &[usize], device: &Device, dtype: DType) -> Result<Self> {
        Ok(match dims {
            [d0] => Self::R1(Tensor::<1, Int>::empty([*d0], (device, dtype))),
            [d0, d1] => Self::R2(Tensor::<2, Int>::empty([*d0, *d1], (device, dtype))),
            [d0, d1, d2] => Self::R3(Tensor::<3, Int>::empty([*d0, *d1, *d2], (device, dtype))),
            [d0, d1, d2, d3] => Self::R4(Tensor::<4, Int>::empty(
                [*d0, *d1, *d2, *d3],
                (device, dtype),
            )),
            [d0, d1, d2, d3, d4] => Self::R5(Tensor::<5, Int>::empty(
                [*d0, *d1, *d2, *d3, *d4],
                (device, dtype),
            )),
            [d0, d1, d2, d3, d4, d5] => Self::R6(Tensor::<6, Int>::empty(
                [*d0, *d1, *d2, *d3, *d4, *d5],
                (device, dtype),
            )),
            _ => return Err(rank_overflow(dims.len())),
        })
    }

    /// Create a random integer tensor with an explicit shape, distribution, and dtype.
    pub fn random(
        dims: &[usize],
        distribution: Distribution,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Ok(match dims {
            [d0] => Self::R1(Tensor::<1, Int>::random(
                [*d0],
                distribution,
                (device, dtype),
            )),
            [d0, d1] => Self::R2(Tensor::<2, Int>::random(
                [*d0, *d1],
                distribution,
                (device, dtype),
            )),
            [d0, d1, d2] => Self::R3(Tensor::<3, Int>::random(
                [*d0, *d1, *d2],
                distribution,
                (device, dtype),
            )),
            [d0, d1, d2, d3] => Self::R4(Tensor::<4, Int>::random(
                [*d0, *d1, *d2, *d3],
                distribution,
                (device, dtype),
            )),
            [d0, d1, d2, d3, d4] => Self::R5(Tensor::<5, Int>::random(
                [*d0, *d1, *d2, *d3, *d4],
                distribution,
                (device, dtype),
            )),
            [d0, d1, d2, d3, d4, d5] => Self::R6(Tensor::<6, Int>::random(
                [*d0, *d1, *d2, *d3, *d4, *d5],
                distribution,
                (device, dtype),
            )),
            _ => return Err(rank_overflow(dims.len())),
        })
    }

    /// Create a rank-one integer range with a nonzero signed step.
    pub fn arange(start: i64, end: i64, step: i64, device: &Device, dtype: DType) -> Result<Self> {
        if step == 0 {
            return Err(TynxError::Shape("arange step must be nonzero".to_string()));
        }
        let distance = if step > 0 {
            end.saturating_sub(start).max(0)
        } else {
            start.saturating_sub(end).max(0)
        };
        let magnitude = step.unsigned_abs();
        let count = (distance as u64).div_ceil(magnitude);
        let count = i64::try_from(count)
            .map_err(|_| TynxError::Shape("arange output is too large".to_string()))?;
        Ok(Self::R1(
            Tensor::<1, Int>::arange(0..count, (device, dtype))
                .mul_scalar(step)
                .add_scalar(start),
        ))
    }

    /// Create an integer tensor filled with one value and an explicit dtype.
    pub fn full(dims: &[usize], value: i64, device: &Device, dtype: DType) -> Result<Self> {
        Ok(match dims {
            [d0] => Self::R1(Tensor::<1, Int>::full([*d0], value, (device, dtype))),
            [d0, d1] => Self::R2(Tensor::<2, Int>::full([*d0, *d1], value, (device, dtype))),
            [d0, d1, d2] => Self::R3(Tensor::<3, Int>::full(
                [*d0, *d1, *d2],
                value,
                (device, dtype),
            )),
            [d0, d1, d2, d3] => Self::R4(Tensor::<4, Int>::full(
                [*d0, *d1, *d2, *d3],
                value,
                (device, dtype),
            )),
            [d0, d1, d2, d3, d4] => Self::R5(Tensor::<5, Int>::full(
                [*d0, *d1, *d2, *d3, *d4],
                value,
                (device, dtype),
            )),
            [d0, d1, d2, d3, d4, d5] => Self::R6(Tensor::<6, Int>::full(
                [*d0, *d1, *d2, *d3, *d4, *d5],
                value,
                (device, dtype),
            )),
            _ => return Err(rank_overflow(dims.len())),
        })
    }

    /// Reshape the tensor while preserving its elements and dtype.
    pub fn reshape(self, dims: Vec<usize>) -> Result<Self> {
        Ok(match self {
            Self::R1(tensor) => reshape_dyn!(tensor, dims, DynInt),
            Self::R2(tensor) => reshape_dyn!(tensor, dims, DynInt),
            Self::R3(tensor) => reshape_dyn!(tensor, dims, DynInt),
            Self::R4(tensor) => reshape_dyn!(tensor, dims, DynInt),
            Self::R5(tensor) => reshape_dyn!(tensor, dims, DynInt),
            Self::R6(tensor) => reshape_dyn!(tensor, dims, DynInt),
        })
    }

    /// Expand singleton dimensions to the requested shape.
    pub fn expand(self, dims: &[usize]) -> Result<Self> {
        if dims.len() != self.rank() {
            return Err(TynxError::Shape(format!(
                "expand shape has rank {}, expected {}",
                dims.len(),
                self.rank()
            )));
        }
        Ok(match self {
            Self::R1(tensor) => Self::R1(tensor.expand([dims[0] as i64])),
            Self::R2(tensor) => Self::R2(tensor.expand([dims[0] as i64, dims[1] as i64])),
            Self::R3(tensor) => {
                Self::R3(tensor.expand([dims[0] as i64, dims[1] as i64, dims[2] as i64]))
            }
            Self::R4(tensor) => Self::R4(tensor.expand([
                dims[0] as i64,
                dims[1] as i64,
                dims[2] as i64,
                dims[3] as i64,
            ])),
            Self::R5(tensor) => Self::R5(tensor.expand([
                dims[0] as i64,
                dims[1] as i64,
                dims[2] as i64,
                dims[3] as i64,
                dims[4] as i64,
            ])),
            Self::R6(tensor) => Self::R6(tensor.expand([
                dims[0] as i64,
                dims[1] as i64,
                dims[2] as i64,
                dims[3] as i64,
                dims[4] as i64,
                dims[5] as i64,
            ])),
        })
    }

    /// Tile each dimension by the corresponding repeat count.
    pub fn repeat(self, repeats: &[usize]) -> Self {
        map_int!(self, |tensor| tensor.repeat(repeats))
    }

    /// Slice the tensor with one slice per dimension.
    pub fn slice(self, slices: &[Slice]) -> Self {
        map_int!(self, |tensor| tensor.slice(slices))
    }

    /// Assign a same-rank tensor into a slice.
    pub fn slice_assign(self, slices: &[Slice], values: Self) -> Result<Self> {
        Ok(slice_assign_dyn!(self, slices, values, DynInt))
    }

    /// Reverse one dimension.
    pub fn flip_dim(self, dim: usize) -> Self {
        map_int!(self, |tensor| tensor.flip([dim as isize]))
    }

    /// Map negative indices into the corresponding positive dimension indices.
    pub fn normalize_indices(self, size: usize) -> Result<Self> {
        if size == 0 {
            return Err(TynxError::Shape("cannot index an empty dimension".into()));
        }
        let size = i64::try_from(size)
            .map_err(|_| TynxError::Shape("indexed dimension exceeds i64".into()))?;
        Ok(map_int!(self, |tensor| {
            let negative = tensor.clone().lower_elem(0);
            tensor.clone().mask_where(negative, tensor.add_scalar(size))
        }))
    }

    /// Select slices along one dimension using flattened indices.
    pub fn select(self, dim: usize, indices: DynInt) -> Result<Self> {
        Ok(select_dyn!(self, dim, indices, DynInt))
    }

    /// Gather elements from a same-rank index tensor.
    pub fn gather(self, dim: usize, indices: DynInt) -> Result<Self> {
        let slices = gather_slices(&self.dims(), &indices.dims(), dim)?;
        Ok(gather_dyn!(self.slice(&slices), dim, indices, DynInt))
    }

    /// Gather slices using tuples stored in the last index dimension.
    pub fn gather_nd(self, indices: DynInt, output_rank: usize) -> Result<Self> {
        Ok(gather_nd_dyn!(self, indices, output_rank, DynInt))
    }

    /// Scatter same-rank updates along one dimension.
    pub fn scatter(
        self,
        dim: usize,
        indices: DynInt,
        values: Self,
        update: IndexingUpdateOp,
    ) -> Result<Self> {
        Ok(scatter_dyn!(self, dim, indices, values, update, DynInt))
    }

    /// Scatter updates using tuples stored in the last index dimension.
    pub fn scatter_nd(
        self,
        indices: DynInt,
        values: Self,
        update: IndexingUpdateOp,
    ) -> Result<Self> {
        Ok(scatter_nd_dyn!(self, indices, values, update, DynInt))
    }

    /// Compute a cumulative sum with ONNX exclusive and reverse semantics.
    pub fn cumsum(self, dim: usize, exclusive: bool, reverse: bool) -> Self {
        map_int!(self, |tensor| {
            let input = if reverse {
                tensor.flip([dim as isize])
            } else {
                tensor
            };
            let output = input.clone().cumsum(dim);
            let output = if exclusive { output.sub(input) } else { output };
            if reverse {
                output.flip([dim as isize])
            } else {
                output
            }
        })
    }

    /// Return the largest values and their indices along one dimension.
    pub fn topk(self, k: usize, dim: usize) -> (Self, DynInt) {
        match self {
            Self::R1(tensor) => {
                let (values, indices) = tensor.topk_with_indices(k, dim);
                (Self::R1(values), Self::R1(indices.cast(DType::I64)))
            }
            Self::R2(tensor) => {
                let (values, indices) = tensor.topk_with_indices(k, dim);
                (Self::R2(values), Self::R2(indices.cast(DType::I64)))
            }
            Self::R3(tensor) => {
                let (values, indices) = tensor.topk_with_indices(k, dim);
                (Self::R3(values), Self::R3(indices.cast(DType::I64)))
            }
            Self::R4(tensor) => {
                let (values, indices) = tensor.topk_with_indices(k, dim);
                (Self::R4(values), Self::R4(indices.cast(DType::I64)))
            }
            Self::R5(tensor) => {
                let (values, indices) = tensor.topk_with_indices(k, dim);
                (Self::R5(values), Self::R5(indices.cast(DType::I64)))
            }
            Self::R6(tensor) => {
                let (values, indices) = tensor.topk_with_indices(k, dim);
                (Self::R6(values), Self::R6(indices.cast(DType::I64)))
            }
        }
    }

    /// Permute the tensor dimensions.
    pub fn permute(self, axes: Vec<usize>) -> Result<Self> {
        if axes.len() != self.rank() {
            return Err(TynxError::Shape(format!(
                "permutation has {} axes for rank {}",
                axes.len(),
                self.rank()
            )));
        }
        Ok(match self {
            Self::R1(tensor) => Self::R1(tensor.permute([axes[0]])),
            Self::R2(tensor) => Self::R2(tensor.permute([axes[0], axes[1]])),
            Self::R3(tensor) => Self::R3(tensor.permute([axes[0], axes[1], axes[2]])),
            Self::R4(tensor) => Self::R4(tensor.permute([axes[0], axes[1], axes[2], axes[3]])),
            Self::R5(tensor) => {
                Self::R5(tensor.permute([axes[0], axes[1], axes[2], axes[3], axes[4]]))
            }
            Self::R6(tensor) => {
                Self::R6(tensor.permute([axes[0], axes[1], axes[2], axes[3], axes[4], axes[5]]))
            }
        })
    }

    /// Promote the tensor by adding leading singleton dimensions.
    pub fn to_rank(self, target: usize) -> Result<Self> {
        let current = self.rank();
        if current == target {
            return Ok(self);
        }
        if current > target || target > MAX_RANK {
            return Err(TynxError::RankPromote {
                from: current,
                to: target,
            });
        }

        Ok(match (self, target) {
            (Self::R1(tensor), 2) => Self::R2(tensor.unsqueeze()),
            (Self::R1(tensor), 3) => Self::R3(tensor.unsqueeze()),
            (Self::R1(tensor), 4) => Self::R4(tensor.unsqueeze()),
            (Self::R1(tensor), 5) => Self::R5(tensor.unsqueeze()),
            (Self::R1(tensor), 6) => Self::R6(tensor.unsqueeze()),
            (Self::R2(tensor), 3) => Self::R3(tensor.unsqueeze()),
            (Self::R2(tensor), 4) => Self::R4(tensor.unsqueeze()),
            (Self::R2(tensor), 5) => Self::R5(tensor.unsqueeze()),
            (Self::R2(tensor), 6) => Self::R6(tensor.unsqueeze()),
            (Self::R3(tensor), 4) => Self::R4(tensor.unsqueeze()),
            (Self::R3(tensor), 5) => Self::R5(tensor.unsqueeze()),
            (Self::R3(tensor), 6) => Self::R6(tensor.unsqueeze()),
            (Self::R4(tensor), 5) => Self::R5(tensor.unsqueeze()),
            (Self::R4(tensor), 6) => Self::R6(tensor.unsqueeze()),
            (Self::R5(tensor), 6) => Self::R6(tensor.unsqueeze()),
            (_, target) => return Err(rank_overflow(target)),
        })
    }

    /// Return the tensor's element type.
    pub fn dtype(&self) -> burn::tensor::DType {
        match self {
            Self::R1(tensor) => tensor.dtype(),
            Self::R2(tensor) => tensor.dtype(),
            Self::R3(tensor) => tensor.dtype(),
            Self::R4(tensor) => tensor.dtype(),
            Self::R5(tensor) => tensor.dtype(),
            Self::R6(tensor) => tensor.dtype(),
        }
    }

    /// Cast the tensor while preserving its rank.
    pub fn cast(self, dtype: DType) -> Self {
        map_int!(self, |tensor| tensor.cast(dtype))
    }

    /// Convert the tensor to a floating-point tensor with an explicit dtype.
    pub fn to_float(self, dtype: DType) -> DynTensor {
        match self {
            Self::R1(tensor) => DynTensor::R1(tensor.float().cast(dtype)),
            Self::R2(tensor) => DynTensor::R2(tensor.float().cast(dtype)),
            Self::R3(tensor) => DynTensor::R3(tensor.float().cast(dtype)),
            Self::R4(tensor) => DynTensor::R4(tensor.float().cast(dtype)),
            Self::R5(tensor) => DynTensor::R5(tensor.float().cast(dtype)),
            Self::R6(tensor) => DynTensor::R6(tensor.float().cast(dtype)),
        }
    }

    /// Convert the tensor to booleans using nonzero truth semantics.
    pub fn to_bool(self) -> DynBool {
        match self {
            Self::R1(tensor) => DynBool::R1(tensor.bool()),
            Self::R2(tensor) => DynBool::R2(tensor.bool()),
            Self::R3(tensor) => DynBool::R3(tensor.bool()),
            Self::R4(tensor) => DynBool::R4(tensor.bool()),
            Self::R5(tensor) => DynBool::R5(tensor.bool()),
            Self::R6(tensor) => DynBool::R6(tensor.bool()),
        }
    }

    /// Return indices of extrema along one dimension using ONNX tie semantics.
    pub fn arg_extreme(self, dim: usize, maximum: bool, select_last: bool) -> Self {
        macro_rules! apply {
            ($tensor:expr) => {{
                let tensor = $tensor;
                let axis_size = tensor.dims()[dim] as i64;
                let indices = if maximum {
                    if select_last {
                        tensor
                            .flip([dim as isize])
                            .argmax(dim)
                            .mul_scalar(-1_i64)
                            .add_scalar(axis_size - 1)
                    } else {
                        tensor.argmax(dim)
                    }
                } else if select_last {
                    tensor
                        .flip([dim as isize])
                        .argmin(dim)
                        .mul_scalar(-1_i64)
                        .add_scalar(axis_size - 1)
                } else {
                    tensor.argmin(dim)
                };
                indices.cast(DType::I64)
            }};
        }

        match self {
            Self::R1(tensor) => Self::R1(apply!(tensor)),
            Self::R2(tensor) => Self::R2(apply!(tensor)),
            Self::R3(tensor) => Self::R3(apply!(tensor)),
            Self::R4(tensor) => Self::R4(apply!(tensor)),
            Self::R5(tensor) => Self::R5(apply!(tensor)),
            Self::R6(tensor) => Self::R6(apply!(tensor)),
        }
    }

    /// Raise every element to a broadcastable integer exponent.
    pub fn powi_broadcast(self, exponent: Self) -> Result<Self> {
        let dtype = self.dtype();
        let (base, exponent) = Self::broadcast_pair(self, exponent.cast(dtype))?;
        Ok(zip_int!(base, exponent, |base, exponent| base.powi(exponent)))
    }

    /// Raise every element to an integer scalar exponent.
    pub fn powi_scalar(self, exponent: i64) -> Self {
        map_int!(self, |tensor| tensor.powi_scalar(exponent))
    }

    /// Take the absolute value of each integer element.
    pub fn abs(self) -> Self {
        map_int!(self, |tensor| tensor.abs())
    }

    /// Subtract a signed scalar from every integer element.
    pub fn sub_scalar(self, value: i64) -> Self {
        map_int!(self, |tensor| tensor.sub_scalar(value))
    }

    /// Add a signed scalar to every integer element.
    pub fn add_scalar(self, value: i64) -> Self {
        map_int!(self, |tensor| tensor.add_scalar(value))
    }

    /// Multiply every integer element by a signed scalar.
    pub fn mul_scalar(self, value: i64) -> Self {
        map_int!(self, |tensor| tensor.mul_scalar(value))
    }

    /// Create an integer tensor with the same shape, device, and dtype filled with one scalar.
    pub fn full_like(self, value: i64) -> Self {
        map_int!(self, |tensor| tensor.full_like(value))
    }

    /// Multiply matrices or batches of matrices with matching runtime ranks.
    pub fn matmul(self, other: Self) -> Result<Self> {
        validate_matmul_shapes(&self.dims(), &other.dims())?;
        Ok(match (self, other) {
            (Self::R2(left), Self::R2(right)) => Self::R2(left.matmul(right)),
            (Self::R3(left), Self::R3(right)) => Self::R3(left.matmul(right)),
            (Self::R4(left), Self::R4(right)) => Self::R4(left.matmul(right)),
            (Self::R5(left), Self::R5(right)) => Self::R5(left.matmul(right)),
            (Self::R6(left), Self::R6(right)) => Self::R6(left.matmul(right)),
            (left, right) => {
                return Err(TynxError::Shape(format!(
                    "matmul requires matching ranks >= 2, got {} and {}",
                    left.rank(),
                    right.rank()
                )));
            }
        })
    }

    /// Add two integer tensors with multidirectional broadcasting.
    pub fn add_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int!(left, right, |left, right| left.add(right)))
    }

    /// Subtract two integer tensors with multidirectional broadcasting.
    pub fn sub_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int!(left, right, |left, right| left.sub(right)))
    }

    /// Multiply two integer tensors with multidirectional broadcasting.
    pub fn mul_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int!(left, right, |left, right| left.mul(right)))
    }

    /// Divide two integer tensors with multidirectional broadcasting.
    pub fn div_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int!(left, right, |left, right| left.div(right)))
    }

    /// Apply ONNX modulo or fmod semantics with multidirectional broadcasting.
    pub fn modulo_broadcast(self, other: Self, fmod: bool) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int!(left, right, |left, right| {
            if fmod {
                let quotient = left.clone().div(right.clone());
                left.sub(quotient.mul(right))
            } else {
                let remainder = left.remainder(right.clone());
                remainder.add(right.clone()).remainder(right)
            }
        }))
    }

    /// Apply bitwise AND with multidirectional broadcasting.
    pub fn bitwise_and_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int!(left, right, |left, right| left.bitwise_and(right)))
    }

    /// Apply bitwise OR with multidirectional broadcasting.
    pub fn bitwise_or_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int!(left, right, |left, right| left.bitwise_or(right)))
    }

    /// Apply bitwise XOR with multidirectional broadcasting.
    pub fn bitwise_xor_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int!(left, right, |left, right| left.bitwise_xor(right)))
    }

    /// Apply bitwise NOT.
    pub fn bitwise_not(self) -> Self {
        map_int!(self, |tensor| tensor.bitwise_not())
    }

    /// Shift left with multidirectional broadcasting.
    pub fn bitwise_left_shift_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int!(left, right, |left, right| left.bitwise_left_shift(right)))
    }

    /// Shift right with multidirectional broadcasting.
    pub fn bitwise_right_shift_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int!(left, right, |left, right| left.bitwise_right_shift(right)))
    }

    /// Sum elements along dimensions while retaining singleton dimensions.
    pub fn sum_dims(self, dims: &[usize]) -> Self {
        map_int!(self, |tensor| tensor.sum_dims(dims))
    }

    /// Multiply elements along dimensions while retaining singleton dimensions.
    pub fn prod_dims(self, dims: &[usize]) -> Self {
        map_int!(self, |tensor| tensor.prod_dims(dims))
    }

    /// Take the maximum along dimensions while retaining singleton dimensions.
    pub fn reduce_max_dims(self, dims: &[usize]) -> Self {
        map_int!(self, |tensor| tensor.max_dims(dims))
    }

    /// Take the minimum along dimensions while retaining singleton dimensions.
    pub fn reduce_min_dims(self, dims: &[usize]) -> Self {
        map_int!(self, |tensor| tensor.min_dims(dims))
    }

    /// Take the element-wise maximum with multidirectional broadcasting.
    pub fn max_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int!(left, right, |left, right| left.max_pair(right)))
    }

    /// Take the element-wise minimum with multidirectional broadcasting.
    pub fn min_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int!(left, right, |left, right| left.min_pair(right)))
    }

    /// Compare two integer tensors for equality with multidirectional broadcasting.
    pub fn equal_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int_bool!(left, right, |left, right| left.equal(right)))
    }

    /// Compare every element with an integer scalar for equality.
    pub fn equal_scalar(self, other: i64) -> DynBool {
        map_int_bool!(self, |tensor| tensor.equal_elem(other))
    }

    /// Compare two integer tensors using greater-than with multidirectional broadcasting.
    pub fn greater_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int_bool!(left, right, |left, right| left.greater(right)))
    }

    /// Compare every element with an integer scalar using greater-than.
    pub fn greater_scalar(self, other: i64) -> DynBool {
        map_int_bool!(self, |tensor| tensor.greater_elem(other))
    }

    /// Compare two integer tensors using greater-or-equal with multidirectional broadcasting.
    pub fn greater_equal_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int_bool!(left, right, |left, right| left.greater_equal(right)))
    }

    /// Compare every element with an integer scalar using greater-or-equal.
    pub fn greater_equal_scalar(self, other: i64) -> DynBool {
        map_int_bool!(self, |tensor| tensor.greater_equal_elem(other))
    }

    /// Compare two integer tensors using less-than with multidirectional broadcasting.
    pub fn less_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int_bool!(left, right, |left, right| left.lower(right)))
    }

    /// Compare every element with an integer scalar using less-than.
    pub fn less_scalar(self, other: i64) -> DynBool {
        map_int_bool!(self, |tensor| tensor.lower_elem(other))
    }

    /// Compare two integer tensors using less-or-equal with multidirectional broadcasting.
    pub fn less_equal_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int_bool!(left, right, |left, right| left.lower_equal(right)))
    }

    /// Compare every element with an integer scalar using less-or-equal.
    pub fn less_equal_scalar(self, other: i64) -> DynBool {
        map_int_bool!(self, |tensor| tensor.lower_equal_elem(other))
    }

    fn broadcast_pair(left: Self, right: Self) -> Result<(Self, Self)> {
        broadcast_shape(&left.dims(), &right.dims())?;
        let rank = left.rank().max(right.rank());
        Ok((left.to_rank(rank)?, right.to_rank(rank)?))
    }

    /// Clamp every element to the optional lower and upper bounds.
    pub fn clip(self, min: Option<crate::Scalar>, max: Option<crate::Scalar>) -> Self {
        if self.dtype().is_uint() {
            let min = min.map(scalar_as_u64);
            let max = max.map(scalar_as_u64);
            match (min, max) {
                (Some(min), Some(max)) => {
                    map_int!(self, |tensor| tensor.clamp_min(min).clamp_max(max))
                }
                (Some(min), None) => map_int!(self, |tensor| tensor.clamp_min(min)),
                (None, Some(max)) => map_int!(self, |tensor| tensor.clamp_max(max)),
                (None, None) => self,
            }
        } else {
            let min = min.map(scalar_as_i64);
            let max = max.map(scalar_as_i64);
            match (min, max) {
                (Some(min), Some(max)) => {
                    map_int!(self, |tensor| tensor.clamp_min(min).clamp_max(max))
                }
                (Some(min), None) => map_int!(self, |tensor| tensor.clamp_min(min)),
                (None, Some(max)) => map_int!(self, |tensor| tensor.clamp_max(max)),
                (None, None) => self,
            }
        }
    }

    /// Select elements from two integer tensors using a broadcastable boolean condition.
    pub fn where_select(condition: DynBool, then: Self, otherwise: Self) -> Result<Self> {
        let output_shape = broadcast_shape(&condition.dims(), &then.dims())?;
        broadcast_shape(&output_shape, &otherwise.dims())?;
        let rank = condition.rank().max(then.rank()).max(otherwise.rank());
        let dtype = then.dtype();
        Ok(where_int!(
            condition.to_rank(rank)?,
            then.to_rank(rank)?,
            otherwise.cast(dtype).to_rank(rank)?
        ))
    }
}

fn scalar_as_i64(value: crate::Scalar) -> i64 {
    match value {
        crate::Scalar::F64(value) => value as i64,
        crate::Scalar::I64(value) => value,
        crate::Scalar::U64(value) => value.min(i64::MAX as u64) as i64,
        crate::Scalar::Bool(value) => i64::from(value),
    }
}

fn scalar_as_u64(value: crate::Scalar) -> u64 {
    match value {
        crate::Scalar::F64(value) => value as u64,
        crate::Scalar::I64(value) => value.max(0) as u64,
        crate::Scalar::U64(value) => value,
        crate::Scalar::Bool(value) => u64::from(value),
    }
}

impl DynBool {
    /// Create an uninitialized boolean tensor with an explicit shape.
    pub fn empty(dims: &[usize], device: &Device) -> Result<Self> {
        let dtype = DType::Bool(burn::tensor::BoolStore::Native);
        Ok(match dims {
            [d0] => Self::R1(Tensor::<1, Bool>::empty([*d0], (device, dtype))),
            [d0, d1] => Self::R2(Tensor::<2, Bool>::empty([*d0, *d1], (device, dtype))),
            [d0, d1, d2] => Self::R3(Tensor::<3, Bool>::empty([*d0, *d1, *d2], (device, dtype))),
            [d0, d1, d2, d3] => Self::R4(Tensor::<4, Bool>::empty(
                [*d0, *d1, *d2, *d3],
                (device, dtype),
            )),
            [d0, d1, d2, d3, d4] => Self::R5(Tensor::<5, Bool>::empty(
                [*d0, *d1, *d2, *d3, *d4],
                (device, dtype),
            )),
            [d0, d1, d2, d3, d4, d5] => Self::R6(Tensor::<6, Bool>::empty(
                [*d0, *d1, *d2, *d3, *d4, *d5],
                (device, dtype),
            )),
            _ => return Err(rank_overflow(dims.len())),
        })
    }

    /// Return the tensor's boolean storage type.
    pub fn dtype(&self) -> DType {
        match self {
            Self::R1(tensor) => tensor.dtype(),
            Self::R2(tensor) => tensor.dtype(),
            Self::R3(tensor) => tensor.dtype(),
            Self::R4(tensor) => tensor.dtype(),
            Self::R5(tensor) => tensor.dtype(),
            Self::R6(tensor) => tensor.dtype(),
        }
    }

    /// Create a boolean tensor filled with one value.
    pub fn full(dims: &[usize], value: bool, device: &Device) -> Result<Self> {
        let dtype = DType::Bool(burn::tensor::BoolStore::Native);
        Ok(match dims {
            [d0] => Self::R1(Tensor::<1, Bool>::full([*d0], value, (device, dtype))),
            [d0, d1] => Self::R2(Tensor::<2, Bool>::full([*d0, *d1], value, (device, dtype))),
            [d0, d1, d2] => Self::R3(Tensor::<3, Bool>::full(
                [*d0, *d1, *d2],
                value,
                (device, dtype),
            )),
            [d0, d1, d2, d3] => Self::R4(Tensor::<4, Bool>::full(
                [*d0, *d1, *d2, *d3],
                value,
                (device, dtype),
            )),
            [d0, d1, d2, d3, d4] => Self::R5(Tensor::<5, Bool>::full(
                [*d0, *d1, *d2, *d3, *d4],
                value,
                (device, dtype),
            )),
            [d0, d1, d2, d3, d4, d5] => Self::R6(Tensor::<6, Bool>::full(
                [*d0, *d1, *d2, *d3, *d4, *d5],
                value,
                (device, dtype),
            )),
            _ => return Err(rank_overflow(dims.len())),
        })
    }

    /// Return coordinates of all true elements in row-major order.
    pub fn nonzero(self) -> DynInt {
        let coordinates = match self {
            Self::R1(tensor) => tensor.nonzero(),
            Self::R2(tensor) => tensor.nonzero(),
            Self::R3(tensor) => tensor.nonzero(),
            Self::R4(tensor) => tensor.nonzero(),
            Self::R5(tensor) => tensor.nonzero(),
            Self::R6(tensor) => tensor.nonzero(),
        };
        DynInt::R2(Tensor::stack(coordinates, 0).cast(DType::I64))
    }

    /// Reshape the tensor while preserving its elements and dtype.
    pub fn reshape(self, dims: Vec<usize>) -> Result<Self> {
        Ok(match self {
            Self::R1(tensor) => reshape_dyn!(tensor, dims, DynBool),
            Self::R2(tensor) => reshape_dyn!(tensor, dims, DynBool),
            Self::R3(tensor) => reshape_dyn!(tensor, dims, DynBool),
            Self::R4(tensor) => reshape_dyn!(tensor, dims, DynBool),
            Self::R5(tensor) => reshape_dyn!(tensor, dims, DynBool),
            Self::R6(tensor) => reshape_dyn!(tensor, dims, DynBool),
        })
    }

    /// Expand singleton dimensions to the requested shape.
    pub fn expand(self, dims: &[usize]) -> Result<Self> {
        if dims.len() != self.rank() {
            return Err(TynxError::Shape(format!(
                "expand shape has rank {}, expected {}",
                dims.len(),
                self.rank()
            )));
        }
        Ok(match self {
            Self::R1(tensor) => Self::R1(tensor.expand([dims[0] as i64])),
            Self::R2(tensor) => Self::R2(tensor.expand([dims[0] as i64, dims[1] as i64])),
            Self::R3(tensor) => {
                Self::R3(tensor.expand([dims[0] as i64, dims[1] as i64, dims[2] as i64]))
            }
            Self::R4(tensor) => Self::R4(tensor.expand([
                dims[0] as i64,
                dims[1] as i64,
                dims[2] as i64,
                dims[3] as i64,
            ])),
            Self::R5(tensor) => Self::R5(tensor.expand([
                dims[0] as i64,
                dims[1] as i64,
                dims[2] as i64,
                dims[3] as i64,
                dims[4] as i64,
            ])),
            Self::R6(tensor) => Self::R6(tensor.expand([
                dims[0] as i64,
                dims[1] as i64,
                dims[2] as i64,
                dims[3] as i64,
                dims[4] as i64,
                dims[5] as i64,
            ])),
        })
    }

    /// Tile each dimension by the corresponding repeat count.
    pub fn repeat(self, repeats: &[usize]) -> Self {
        map_bool!(self, |tensor| tensor.repeat(repeats))
    }

    /// Slice the tensor with one slice per dimension.
    pub fn slice(self, slices: &[Slice]) -> Self {
        map_bool!(self, |tensor| tensor.slice(slices))
    }

    /// Assign a same-rank tensor into a slice.
    pub fn slice_assign(self, slices: &[Slice], values: Self) -> Result<Self> {
        Ok(slice_assign_dyn!(self, slices, values, DynBool))
    }

    /// Reverse one dimension.
    pub fn flip_dim(self, dim: usize) -> Self {
        map_bool!(self, |tensor| tensor.flip([dim as isize]))
    }

    /// Select slices along one dimension using flattened indices.
    pub fn select(self, dim: usize, indices: DynInt) -> Result<Self> {
        Ok(select_dyn!(self, dim, indices, DynBool))
    }

    /// Gather elements from a same-rank index tensor.
    pub fn gather(self, dim: usize, indices: DynInt) -> Result<Self> {
        let slices = gather_slices(&self.dims(), &indices.dims(), dim)?;
        Ok(gather_dyn!(self.slice(&slices), dim, indices, DynBool))
    }

    /// Gather slices using tuples stored in the last index dimension.
    pub fn gather_nd(self, indices: DynInt, output_rank: usize) -> Result<Self> {
        Ok(gather_nd_dyn!(self, indices, output_rank, DynBool))
    }

    /// Scatter same-rank updates along one dimension.
    pub fn scatter(
        self,
        dim: usize,
        indices: DynInt,
        values: Self,
        update: IndexingUpdateOp,
    ) -> Result<Self> {
        Ok(scatter_dyn!(self, dim, indices, values, update, DynBool))
    }

    /// Scatter updates using tuples stored in the last index dimension.
    pub fn scatter_nd(
        self,
        indices: DynInt,
        values: Self,
        update: IndexingUpdateOp,
    ) -> Result<Self> {
        Ok(scatter_nd_dyn!(self, indices, values, update, DynBool))
    }

    /// Permute the tensor dimensions.
    pub fn permute(self, axes: Vec<usize>) -> Result<Self> {
        if axes.len() != self.rank() {
            return Err(TynxError::Shape(format!(
                "permutation has {} axes for rank {}",
                axes.len(),
                self.rank()
            )));
        }
        Ok(match self {
            Self::R1(tensor) => Self::R1(tensor.permute([axes[0]])),
            Self::R2(tensor) => Self::R2(tensor.permute([axes[0], axes[1]])),
            Self::R3(tensor) => Self::R3(tensor.permute([axes[0], axes[1], axes[2]])),
            Self::R4(tensor) => Self::R4(tensor.permute([axes[0], axes[1], axes[2], axes[3]])),
            Self::R5(tensor) => {
                Self::R5(tensor.permute([axes[0], axes[1], axes[2], axes[3], axes[4]]))
            }
            Self::R6(tensor) => {
                Self::R6(tensor.permute([axes[0], axes[1], axes[2], axes[3], axes[4], axes[5]]))
            }
        })
    }

    /// Promote the tensor by adding leading singleton dimensions.
    pub fn to_rank(self, target: usize) -> Result<Self> {
        let current = self.rank();
        if current == target {
            return Ok(self);
        }
        if current > target || target > MAX_RANK {
            return Err(TynxError::RankPromote {
                from: current,
                to: target,
            });
        }

        Ok(match (self, target) {
            (Self::R1(tensor), 2) => Self::R2(tensor.unsqueeze()),
            (Self::R1(tensor), 3) => Self::R3(tensor.unsqueeze()),
            (Self::R1(tensor), 4) => Self::R4(tensor.unsqueeze()),
            (Self::R1(tensor), 5) => Self::R5(tensor.unsqueeze()),
            (Self::R1(tensor), 6) => Self::R6(tensor.unsqueeze()),
            (Self::R2(tensor), 3) => Self::R3(tensor.unsqueeze()),
            (Self::R2(tensor), 4) => Self::R4(tensor.unsqueeze()),
            (Self::R2(tensor), 5) => Self::R5(tensor.unsqueeze()),
            (Self::R2(tensor), 6) => Self::R6(tensor.unsqueeze()),
            (Self::R3(tensor), 4) => Self::R4(tensor.unsqueeze()),
            (Self::R3(tensor), 5) => Self::R5(tensor.unsqueeze()),
            (Self::R3(tensor), 6) => Self::R6(tensor.unsqueeze()),
            (Self::R4(tensor), 5) => Self::R5(tensor.unsqueeze()),
            (Self::R4(tensor), 6) => Self::R6(tensor.unsqueeze()),
            (Self::R5(tensor), 6) => Self::R6(tensor.unsqueeze()),
            (_, target) => return Err(rank_overflow(target)),
        })
    }

    /// Apply logical AND with multidirectional broadcasting.
    pub fn and_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_bool!(left, right, |left, right| left.bool_and(right)))
    }

    /// Apply logical OR with multidirectional broadcasting.
    pub fn or_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_bool!(left, right, |left, right| left.bool_or(right)))
    }

    /// Apply logical XOR with multidirectional broadcasting.
    pub fn xor_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_bool!(left, right, |left, right| left.bool_xor(right)))
    }

    /// Apply logical NOT element-wise.
    pub fn logical_not(self) -> Self {
        map_bool!(self, |tensor| tensor.bool_not())
    }

    /// Create a boolean tensor with the same shape and device filled with one scalar.
    pub fn full_like(self, value: bool) -> Self {
        map_bool!(self, |tensor| tensor.full_like(value))
    }

    /// Convert the boolean tensor to an integer tensor with an explicit dtype.
    pub fn to_int(self, dtype: DType) -> DynInt {
        match self {
            Self::R1(tensor) => DynInt::R1(tensor.int().cast(dtype)),
            Self::R2(tensor) => DynInt::R2(tensor.int().cast(dtype)),
            Self::R3(tensor) => DynInt::R3(tensor.int().cast(dtype)),
            Self::R4(tensor) => DynInt::R4(tensor.int().cast(dtype)),
            Self::R5(tensor) => DynInt::R5(tensor.int().cast(dtype)),
            Self::R6(tensor) => DynInt::R6(tensor.int().cast(dtype)),
        }
    }

    /// Convert the boolean tensor to a floating-point tensor with an explicit dtype.
    pub fn to_float(self, dtype: DType) -> DynTensor {
        match self {
            Self::R1(tensor) => DynTensor::R1(tensor.float().cast(dtype)),
            Self::R2(tensor) => DynTensor::R2(tensor.float().cast(dtype)),
            Self::R3(tensor) => DynTensor::R3(tensor.float().cast(dtype)),
            Self::R4(tensor) => DynTensor::R4(tensor.float().cast(dtype)),
            Self::R5(tensor) => DynTensor::R5(tensor.float().cast(dtype)),
            Self::R6(tensor) => DynTensor::R6(tensor.float().cast(dtype)),
        }
    }

    /// Take the maximum along dimensions while retaining singleton dimensions.
    pub fn reduce_max_dims(self, dims: &[usize]) -> Self {
        // Boolean values are exact in f32, avoiding WGPU's non-portable i64 reduction shader.
        map_bool!(self, |tensor| tensor
            .float()
            .cast(DType::F32)
            .max_dims(dims)
            .bool())
    }

    /// Take the minimum along dimensions while retaining singleton dimensions.
    pub fn reduce_min_dims(self, dims: &[usize]) -> Self {
        // Boolean values are exact in f32, avoiding WGPU's non-portable i64 reduction shader.
        map_bool!(self, |tensor| tensor
            .float()
            .cast(DType::F32)
            .min_dims(dims)
            .bool())
    }

    /// Select elements from two boolean tensors using a broadcastable condition.
    pub fn where_select(condition: Self, then: Self, otherwise: Self) -> Result<Self> {
        let output_shape = broadcast_shape(&condition.dims(), &then.dims())?;
        broadcast_shape(&output_shape, &otherwise.dims())?;
        let rank = condition.rank().max(then.rank()).max(otherwise.rank());
        Ok(where_bool!(
            condition.to_rank(rank)?,
            then.to_rank(rank)?,
            otherwise.to_rank(rank)?
        ))
    }

    fn broadcast_pair(left: Self, right: Self) -> Result<(Self, Self)> {
        broadcast_shape(&left.dims(), &right.dims())?;
        let rank = left.rank().max(right.rank());
        Ok((left.to_rank(rank)?, right.to_rank(rank)?))
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Device;

    use super::*;

    #[test]
    fn reports_runtime_metadata() {
        let tensor = Tensor::<2>::zeros([2, 3], &Device::default());
        let tensor = DynTensor::R2(tensor);

        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.dims(), vec![2, 3]);
    }

    #[test]
    fn promotes_rank_with_leading_singleton_dimensions() {
        let tensor = DynTensor::R1(Tensor::<1>::zeros([3], &Device::default()));

        let promoted = tensor.to_rank(3).unwrap();

        assert_eq!(promoted.dims(), vec![1, 1, 3]);
    }

    #[test]
    fn rejects_rank_demotions() {
        let tensor = DynTensor::R2(Tensor::<2>::zeros([2, 3], &Device::default()));

        let error = tensor.to_rank(1).unwrap_err();

        assert_eq!(error, TynxError::RankPromote { from: 2, to: 1 });
    }

    #[test]
    fn rejects_incompatible_broadcast_shapes_before_dispatch() {
        let device = Device::default();
        let left = DynTensor::R2(Tensor::<2>::zeros([2, 3], &device));
        let right = DynTensor::R2(Tensor::<2>::zeros([4, 5], &device));

        let error = left.add_broadcast(right).unwrap_err();

        assert_eq!(
            error,
            TynxError::Shape(
                "cannot broadcast shapes [2, 3] and [4, 5]: dimensions 3 and 5 conflict at axis 1"
                    .to_string()
            )
        );
    }

    #[test]
    fn rejects_incompatible_matmul_inner_dimensions_before_dispatch() {
        let device = Device::default();
        let left = DynTensor::R2(Tensor::<2>::zeros([1, 2], &device));
        let right = DynTensor::R2(Tensor::<2>::zeros([1, 2], &device));

        let error = left.matmul(right).unwrap_err();

        assert_eq!(
            error,
            TynxError::Shape(
                "matmul inner dimensions must match for shapes [1, 2] and [1, 2], got 2 and 1"
                    .to_string()
            )
        );
    }

    #[test]
    fn rejects_incompatible_where_shapes_before_dispatch() {
        let device = Device::default();
        let condition = DynBool::R2(Tensor::<2, Bool>::zeros([2, 3], &device));
        let then = DynTensor::R2(Tensor::<2>::zeros([2, 3], &device));
        let otherwise = DynTensor::R2(Tensor::<2>::zeros([4, 5], &device));

        let error = DynTensor::where_select(condition, then, otherwise).unwrap_err();

        assert!(
            matches!(error, TynxError::Shape(message) if message.contains("cannot broadcast shapes"))
        );
    }

    #[test]
    fn gather_accepts_smaller_non_index_dimensions() {
        let device = Device::default();
        let data = DynTensor::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
            2,
            &device,
        )
        .unwrap();
        let indices =
            DynInt::from_data(TensorData::new(vec![2_i64, 0], [1, 2]), 2, &device).unwrap();

        let output = data.gather(1, indices).unwrap();

        assert_eq!(output.dims(), vec![1, 2]);
        assert_eq!(
            output.into_data().iter::<f32>().collect::<Vec<_>>(),
            [3.0, 1.0]
        );
    }

    #[test]
    fn nan_propagating_extrema_do_not_hide_nan_values() {
        let device = Device::default();
        let data = TensorData::new(vec![1.0_f32, f32::NAN, 3.0], [3]);

        let minimum = DynTensor::from_data(data.clone(), 1, &device)
            .unwrap()
            .reduce_min_dims_nan(&[0])
            .unwrap()
            .into_data()
            .iter::<f32>()
            .next()
            .unwrap();
        let maximum = DynTensor::from_data(data, 1, &device)
            .unwrap()
            .reduce_max_dims_nan(&[0])
            .unwrap()
            .into_data()
            .iter::<f32>()
            .next()
            .unwrap();

        assert!(minimum.is_nan());
        assert!(maximum.is_nan());
    }
}
