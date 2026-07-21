//! Rank-erased tensor containers used by the runtime.

use burn::tensor::{Bool, Device, Int, Tensor, TensorData, activation};

use crate::error::{Result, TynxError};

/// Highest tensor rank represented by Tynx.
pub const MAX_RANK: usize = 6;

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
    ($tensor:expr, $operation:expr) => {
        match $tensor {
            DynTensor::R1(tensor) => DynTensor::R1(($operation)(tensor)),
            DynTensor::R2(tensor) => DynTensor::R2(($operation)(tensor)),
            DynTensor::R3(tensor) => DynTensor::R3(($operation)(tensor)),
            DynTensor::R4(tensor) => DynTensor::R4(($operation)(tensor)),
            DynTensor::R5(tensor) => DynTensor::R5(($operation)(tensor)),
            DynTensor::R6(tensor) => DynTensor::R6(($operation)(tensor)),
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
                    rank => return Err(TynxError::RankOverflow(rank)),
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

impl DynTensor {
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
            (_, target) => return Err(TynxError::RankOverflow(target)),
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

    fn broadcast_pair(left: Self, right: Self) -> Result<(Self, Self)> {
        let rank = left.rank().max(right.rank());
        Ok((left.to_rank(rank)?, right.to_rank(rank)?))
    }

    /// Apply rectified linear unit element-wise.
    pub fn relu(self) -> Self {
        map_float!(self, activation::relu)
    }

    /// Apply the sigmoid function element-wise.
    pub fn sigmoid(self) -> Self {
        map_float!(self, activation::sigmoid)
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
}
