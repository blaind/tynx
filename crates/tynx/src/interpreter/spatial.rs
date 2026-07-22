//! Shared spatial shape and padding helpers.

use burn::tensor::Tensor;
use onnx_ir::node::padding::{AutoPad, PaddingConfig1d, PaddingConfig2d, PaddingConfig3d};

use crate::{DynTensor, Result, TynxError};

pub(super) fn padding1d(
    input: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    padding: &PaddingConfig1d,
    auto_pad: &AutoPad,
    ceil_mode: bool,
) -> [(usize, usize); 1] {
    [resolve_axis(
        input,
        kernel,
        stride,
        dilation,
        padding.as_tuple(),
        auto_pad,
        ceil_mode,
    )]
}

pub(super) fn padding2d(
    input: [usize; 2],
    kernel: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    padding: &PaddingConfig2d,
    auto_pad: &AutoPad,
    ceil_mode: bool,
) -> [(usize, usize); 2] {
    let (top, left, bottom, right) = padding.as_tuple();
    [
        resolve_axis(
            input[0],
            kernel[0],
            stride[0],
            dilation[0],
            (top, bottom),
            auto_pad,
            ceil_mode,
        ),
        resolve_axis(
            input[1],
            kernel[1],
            stride[1],
            dilation[1],
            (left, right),
            auto_pad,
            ceil_mode,
        ),
    ]
}

pub(super) fn padding3d(
    input: [usize; 3],
    kernel: [usize; 3],
    stride: [usize; 3],
    dilation: [usize; 3],
    padding: &PaddingConfig3d,
    auto_pad: &AutoPad,
    ceil_mode: bool,
) -> [(usize, usize); 3] {
    let (front, top, left, back, bottom, right) = padding.as_tuple();
    [
        resolve_axis(
            input[0],
            kernel[0],
            stride[0],
            dilation[0],
            (front, back),
            auto_pad,
            ceil_mode,
        ),
        resolve_axis(
            input[1],
            kernel[1],
            stride[1],
            dilation[1],
            (top, bottom),
            auto_pad,
            ceil_mode,
        ),
        resolve_axis(
            input[2],
            kernel[2],
            stride[2],
            dilation[2],
            (left, right),
            auto_pad,
            ceil_mode,
        ),
    ]
}

fn resolve_axis(
    input: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    explicit: (usize, usize),
    auto_pad: &AutoPad,
    ceil_mode: bool,
) -> (usize, usize) {
    let effective_kernel = kernel
        .saturating_sub(1)
        .saturating_mul(dilation)
        .saturating_add(1);
    let (start, mut end) = match auto_pad {
        AutoPad::NotSet => explicit,
        AutoPad::Valid => (0, 0),
        AutoPad::SameUpper | AutoPad::SameLower => {
            let output = input.div_ceil(stride);
            let total = output
                .saturating_sub(1)
                .saturating_mul(stride)
                .saturating_add(effective_kernel)
                .saturating_sub(input);
            let small = total / 2;
            let large = total - small;
            if matches!(auto_pad, AutoPad::SameLower) {
                (large, small)
            } else {
                (small, large)
            }
        }
    };

    if ceil_mode {
        let padded = input.saturating_add(start).saturating_add(end);
        if padded >= effective_kernel {
            let numerator = padded - effective_kernel;
            let mut output = numerator.div_ceil(stride) + 1;
            if output > 0 && (output - 1).saturating_mul(stride) >= input.saturating_add(start) {
                output -= 1;
            }
            let required = output
                .saturating_sub(1)
                .saturating_mul(stride)
                .saturating_add(effective_kernel);
            end = end.saturating_add(required.saturating_sub(padded));
        }
    }
    (start, end)
}

pub(super) fn rank3(tensor: DynTensor) -> Result<Tensor<3>> {
    match tensor {
        DynTensor::R3(tensor) => Ok(tensor),
        tensor => Err(TynxError::Shape(format!(
            "operation requires rank 3, got rank {}",
            tensor.rank()
        ))),
    }
}

pub(super) fn rank4(tensor: DynTensor) -> Result<Tensor<4>> {
    match tensor {
        DynTensor::R4(tensor) => Ok(tensor),
        tensor => Err(TynxError::Shape(format!(
            "operation requires rank 4, got rank {}",
            tensor.rank()
        ))),
    }
}

pub(super) fn rank5(tensor: DynTensor) -> Result<Tensor<5>> {
    match tensor {
        DynTensor::R5(tensor) => Ok(tensor),
        tensor => Err(TynxError::Shape(format!(
            "operation requires rank 5, got rank {}",
            tensor.rank()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_padding_places_extra_cell_at_the_expected_end() {
        assert_eq!(
            resolve_axis(4, 3, 2, 1, (0, 0), &AutoPad::SameUpper, false),
            (0, 1)
        );
        assert_eq!(
            resolve_axis(4, 3, 2, 1, (0, 0), &AutoPad::SameLower, false),
            (1, 0)
        );
    }

    #[test]
    fn ceil_mode_extends_only_the_trailing_padding() {
        assert_eq!(
            resolve_axis(4, 3, 2, 1, (0, 0), &AutoPad::NotSet, true),
            (0, 1)
        );
    }
}
