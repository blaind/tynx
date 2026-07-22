//! Rank-erased tensor containers used by the runtime.

use burn::tensor::{Bool, DType, Device, Int, Slice, Tensor, TensorData, activation};

use crate::error::{Result, TynxError};

/// Highest tensor rank represented by Tynx.
pub const MAX_RANK: usize = 6;

fn rank_overflow(rank: usize) -> TynxError {
    TynxError::RankOverflow {
        rank,
        max: MAX_RANK,
    }
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

impl_concat!(DynTensor);
impl_concat!(DynInt);
impl_concat!(DynBool);

impl DynTensor {
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

    /// Select slices along one dimension using flattened indices.
    pub fn select(self, dim: usize, indices: DynInt) -> Result<Self> {
        Ok(select_dyn!(self, dim, indices, DynTensor))
    }

    /// Gather elements from a same-rank index tensor.
    pub fn gather(self, dim: usize, indices: DynInt) -> Result<Self> {
        Ok(gather_dyn!(self, dim, indices, DynTensor))
    }

    /// Gather slices using tuples stored in the last index dimension.
    pub fn gather_nd(self, indices: DynInt, output_rank: usize) -> Result<Self> {
        Ok(gather_nd_dyn!(self, indices, output_rank, DynTensor))
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

    /// Multiply matrices or batches of matrices with matching runtime ranks.
    pub fn matmul(self, other: Self) -> Result<Self> {
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

    /// Compare two tensors for equality with multidirectional broadcasting.
    pub fn equal_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_float_bool!(left, right, |left, right| left.equal(right)))
    }

    /// Compare two tensors using greater-than with multidirectional broadcasting.
    pub fn greater_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_float_bool!(left, right, |left, right| left.greater(right)))
    }

    /// Compare two tensors using greater-or-equal with multidirectional broadcasting.
    pub fn greater_equal_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_float_bool!(left, right, |left, right| left.greater_equal(right)))
    }

    /// Compare two tensors using less-than with multidirectional broadcasting.
    pub fn less_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_float_bool!(left, right, |left, right| left.lower(right)))
    }

    /// Compare two tensors using less-or-equal with multidirectional broadcasting.
    pub fn less_equal_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_float_bool!(left, right, |left, right| left.lower_equal(right)))
    }

    /// Apply parametric rectified linear unit with a broadcastable slope tensor.
    pub fn prelu(self, slope: Self) -> Result<Self> {
        let input_rank = self.rank();
        if slope.rank() > input_rank {
            return Err(TynxError::Shape(format!(
                "PRelu slope rank {} exceeds input rank {input_rank}",
                slope.rank()
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
        Ok(gather_dyn!(self, dim, indices, DynInt))
    }

    /// Gather slices using tuples stored in the last index dimension.
    pub fn gather_nd(self, indices: DynInt, output_rank: usize) -> Result<Self> {
        Ok(gather_nd_dyn!(self, indices, output_rank, DynInt))
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
        let rank = self.rank().max(exponent.rank());
        let base = self.to_rank(rank)?;
        let exponent = exponent.cast(dtype).to_rank(rank)?;
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

    /// Multiply matrices or batches of matrices with matching runtime ranks.
    pub fn matmul(self, other: Self) -> Result<Self> {
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

    /// Compare two integer tensors using greater-than with multidirectional broadcasting.
    pub fn greater_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int_bool!(left, right, |left, right| left.greater(right)))
    }

    /// Compare two integer tensors using greater-or-equal with multidirectional broadcasting.
    pub fn greater_equal_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int_bool!(left, right, |left, right| left.greater_equal(right)))
    }

    /// Compare two integer tensors using less-than with multidirectional broadcasting.
    pub fn less_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int_bool!(left, right, |left, right| left.lower(right)))
    }

    /// Compare two integer tensors using less-or-equal with multidirectional broadcasting.
    pub fn less_equal_broadcast(self, other: Self) -> Result<DynBool> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int_bool!(left, right, |left, right| left.lower_equal(right)))
    }

    fn broadcast_pair(left: Self, right: Self) -> Result<(Self, Self)> {
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

    /// Select slices along one dimension using flattened indices.
    pub fn select(self, dim: usize, indices: DynInt) -> Result<Self> {
        Ok(select_dyn!(self, dim, indices, DynBool))
    }

    /// Gather elements from a same-rank index tensor.
    pub fn gather(self, dim: usize, indices: DynInt) -> Result<Self> {
        Ok(gather_dyn!(self, dim, indices, DynBool))
    }

    /// Gather slices using tuples stored in the last index dimension.
    pub fn gather_nd(self, indices: DynInt, output_rank: usize) -> Result<Self> {
        Ok(gather_nd_dyn!(self, indices, output_rank, DynBool))
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
        map_bool!(self, |tensor| tensor
            .int()
            .cast(DType::I64)
            .max_dims(dims)
            .bool())
    }

    /// Take the minimum along dimensions while retaining singleton dimensions.
    pub fn reduce_min_dims(self, dims: &[usize]) -> Self {
        map_bool!(self, |tensor| tensor
            .int()
            .cast(DType::I64)
            .min_dims(dims)
            .bool())
    }

    /// Select elements from two boolean tensors using a broadcastable condition.
    pub fn where_select(condition: Self, then: Self, otherwise: Self) -> Result<Self> {
        let rank = condition.rank().max(then.rank()).max(otherwise.rank());
        Ok(where_bool!(
            condition.to_rank(rank)?,
            then.to_rank(rank)?,
            otherwise.to_rank(rank)?
        ))
    }

    fn broadcast_pair(left: Self, right: Self) -> Result<(Self, Self)> {
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
}
