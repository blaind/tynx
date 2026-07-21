//! Rank-erased tensor containers used by the runtime.

use burn::tensor::{Bool, DType, Device, Int, Tensor, TensorData, activation};

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
}

impl DynInt {
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

    /// Add two integer tensors with multidirectional broadcasting.
    pub fn add_broadcast(self, other: Self) -> Result<Self> {
        let (left, right) = Self::broadcast_pair(self, other)?;
        Ok(zip_int!(left, right, |left, right| left.add(right)))
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
