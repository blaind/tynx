//! ONNX GridSample execution.

use burn::tensor::{
    Device, Slice, Tensor, TensorData,
    ops::{GridSampleOptions, GridSamplePaddingMode as BurnPaddingMode, InterpolateMode},
};
use onnx_ir::node::grid_sample::{
    GridSampleMode, GridSampleNode, GridSamplePaddingMode as OnnxPaddingMode,
};

use super::{Env, resolve};
use crate::{DynTensor, Result, TynxError, Value};

pub(super) fn grid_sample(node: &GridSampleNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?;
    let grid = resolve::at(env, &node.name, &node.inputs, 1, device)?.into_tensor()?;

    let input_dims = input.dims();
    let grid_dims = grid.dims();
    let input_dtype = input.dtype();
    let input = match input {
        DynTensor::R4(tensor) => tensor,
        tensor => {
            return Err(TynxError::UnsupportedOp(format!(
                "GridSample rank {} (only 2D spatial sampling is supported)",
                tensor.rank()
            )));
        }
    };
    let grid = match grid.cast(input_dtype) {
        DynTensor::R4(tensor) => tensor,
        tensor => {
            return Err(TynxError::Shape(format!(
                "GridSample grid must have rank 4, got rank {}",
                tensor.rank()
            )));
        }
    };

    if grid_dims[3] != 2 {
        return Err(TynxError::Shape(format!(
            "GridSample 2D grid last dimension must be 2, got {}",
            grid_dims[3]
        )));
    }
    if input_dims[0] != grid_dims[0] {
        return Err(TynxError::Shape(format!(
            "GridSample batch dimensions differ: {} and {}",
            input_dims[0], grid_dims[0]
        )));
    }

    if matches!(node.config.mode, GridSampleMode::Bicubic) {
        return Ok(vec![Value::Tensor(bicubic_grid_sample(
            input,
            grid,
            &node.config.padding_mode,
            node.config.align_corners,
            device,
        )?)]);
    }

    let (mode, grid) = match node.config.mode {
        GridSampleMode::Bilinear => (InterpolateMode::Bilinear, grid),
        GridSampleMode::Nearest => (
            InterpolateMode::Nearest,
            nearest_ties_to_even_grid(
                grid,
                input_dims[2],
                input_dims[3],
                node.config.align_corners,
            ),
        ),
        GridSampleMode::Bicubic => {
            return Err(TynxError::UnsupportedOp(
                "GridSample bicubic dispatch invariant".to_string(),
            ));
        }
    };
    let padding_mode = match node.config.padding_mode {
        OnnxPaddingMode::Zeros => BurnPaddingMode::Zeros,
        OnnxPaddingMode::Border => BurnPaddingMode::Border,
        OnnxPaddingMode::Reflection => BurnPaddingMode::Reflection,
    };
    let options = GridSampleOptions::new(mode)
        .with_padding_mode(padding_mode)
        .with_align_corners(node.config.align_corners);

    Ok(vec![Value::Tensor(DynTensor::R4(
        input.grid_sample_2d(grid, options),
    ))])
}

fn bicubic_grid_sample(
    input: Tensor<4>,
    grid: Tensor<4>,
    padding: &OnnxPaddingMode,
    align_corners: bool,
    device: &Device,
) -> Result<DynTensor> {
    let [batch, channels, input_height, input_width] = input.dims();
    let [_, output_height, output_width, _] = grid.dims();
    if input_height == 0 || input_width == 0 {
        return Err(TynxError::Shape(
            "GridSample input spatial dimensions must be positive".into(),
        ));
    }
    let dtype = input.dtype();
    let input = input.into_data().iter::<f64>().collect::<Vec<_>>();
    let grid = grid.into_data().iter::<f64>().collect::<Vec<_>>();
    let mut output = Vec::with_capacity(batch * channels * output_height * output_width);

    for n in 0..batch {
        for c in 0..channels {
            for output_y in 0..output_height {
                for output_x in 0..output_width {
                    let grid_offset =
                        ((n * output_height + output_y) * output_width + output_x) * 2;
                    let source_x = denormalize(grid[grid_offset], input_width, align_corners);
                    let source_y = denormalize(grid[grid_offset + 1], input_height, align_corners);
                    let base_x = source_x.floor() as i64;
                    let base_y = source_y.floor() as i64;
                    let mut value = 0.0;

                    for y_tap in -1..=2 {
                        let y = base_y + y_tap;
                        let y_weight = cubic_weight(source_y - y as f64);
                        for x_tap in -1..=2 {
                            let x = base_x + x_tap;
                            let x_weight = cubic_weight(source_x - x as f64);
                            value += sample_value(
                                &input,
                                n,
                                c,
                                y,
                                x,
                                [channels, input_height, input_width],
                                padding,
                                align_corners,
                            ) * y_weight
                                * x_weight;
                        }
                    }
                    output.push(value);
                }
            }
        }
    }

    Ok(DynTensor::from_data(
        TensorData::new(output, [batch, channels, output_height, output_width]),
        4,
        device,
    )?
    .cast(dtype))
}

fn denormalize(coordinate: f64, size: usize, align_corners: bool) -> f64 {
    if align_corners {
        (coordinate + 1.0) * (size.saturating_sub(1)) as f64 / 2.0
    } else {
        ((coordinate + 1.0) * size as f64 - 1.0) / 2.0
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_value(
    input: &[f64],
    batch: usize,
    channel: usize,
    y: i64,
    x: i64,
    dims: [usize; 3],
    padding: &OnnxPaddingMode,
    align_corners: bool,
) -> f64 {
    let [channels, height, width] = dims;
    let Some(y) = padded_index(y, height, padding, align_corners) else {
        return 0.0;
    };
    let Some(x) = padded_index(x, width, padding, align_corners) else {
        return 0.0;
    };
    input[((batch * channels + channel) * height + y) * width + x]
}

fn padded_index(
    index: i64,
    size: usize,
    padding: &OnnxPaddingMode,
    align_corners: bool,
) -> Option<usize> {
    match padding {
        OnnxPaddingMode::Zeros => usize::try_from(index).ok().filter(|index| *index < size),
        OnnxPaddingMode::Border => Some(index.clamp(0, size as i64 - 1) as usize),
        OnnxPaddingMode::Reflection => {
            let (low, high) = if align_corners {
                (0.0, size.saturating_sub(1) as f64)
            } else {
                (-0.5, size as f64 - 0.5)
            };
            Some(reflect(index as f64, low, high).clamp(0.0, (size - 1) as f64) as usize)
        }
    }
}

fn reflect(coordinate: f64, low: f64, high: f64) -> f64 {
    let span = high - low;
    if span == 0.0 {
        return 0.0;
    }
    let distance = (coordinate - low).abs();
    let flips = (distance / span).floor() as i64;
    let remainder = distance - flips as f64 * span;
    if flips % 2 == 0 {
        low + remainder
    } else {
        high - remainder
    }
}

fn cubic_weight(distance: f64) -> f64 {
    const COEFFICIENT: f64 = -0.75;
    let distance = distance.abs();
    if distance <= 1.0 {
        (COEFFICIENT + 2.0) * distance.powi(3) - (COEFFICIENT + 3.0) * distance.powi(2) + 1.0
    } else if distance < 2.0 {
        COEFFICIENT * distance.powi(3) - 5.0 * COEFFICIENT * distance.powi(2)
            + 8.0 * COEFFICIENT * distance
            - 4.0 * COEFFICIENT
    } else {
        0.0
    }
}

fn nearest_ties_to_even_grid(
    grid: Tensor<4>,
    input_height: usize,
    input_width: usize,
    align_corners: bool,
) -> Tensor<4> {
    let prefix = [Slice::full(), Slice::full(), Slice::full()];
    let x = grid
        .clone()
        .slice([prefix[0], prefix[1], prefix[2], Slice::new(0, Some(1), 1)]);
    let y = grid.slice([prefix[0], prefix[1], prefix[2], Slice::new(1, Some(2), 1)]);
    Tensor::cat(
        vec![
            snap_coordinate(x, input_width, align_corners),
            snap_coordinate(y, input_height, align_corners),
        ],
        3,
    )
}

fn snap_coordinate(coordinate: Tensor<4>, size: usize, align_corners: bool) -> Tensor<4> {
    if align_corners {
        if size == 1 {
            return coordinate.mul_scalar(0.0);
        }
        let scale = (size - 1) as f64 / 2.0;
        coordinate
            .add_scalar(1.0)
            .mul_scalar(scale)
            .round()
            .mul_scalar(2.0 / (size - 1) as f64)
            .sub_scalar(1.0)
    } else {
        let size = size as f64;
        coordinate
            .add_scalar(1.0)
            .mul_scalar(size / 2.0)
            .sub_scalar(0.5)
            .round()
            .add_scalar(0.5)
            .mul_scalar(2.0 / size)
            .sub_scalar(1.0)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::grid_sample::{GridSampleConfig, GridSampleNodeBuilder},
    };

    use super::*;

    #[test]
    fn samples_a_bilinear_center_point() {
        let node = GridSampleNodeBuilder::new("grid_sample")
            .input_tensor("x", 4, DType::F32)
            .input_tensor("grid", 4, DType::F32)
            .output_tensor("y", 4, DType::F32)
            .config(GridSampleConfig {
                mode: GridSampleMode::Bilinear,
                padding_mode: OnnxPaddingMode::Border,
                align_corners: true,
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
                4,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "grid".into(),
            Value::from_tensor_data(
                TensorData::new(vec![0.0_f32, 0.0], [1, 1, 1, 2]),
                4,
                &device,
            )
            .unwrap(),
        );

        let output = grid_sample(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [2.5]);
    }

    #[test]
    fn nearest_halfway_coordinates_round_to_even() {
        let node = GridSampleNodeBuilder::new("grid_sample")
            .input_tensor("x", 4, DType::F32)
            .input_tensor("grid", 4, DType::F32)
            .output_tensor("y", 4, DType::F32)
            .config(GridSampleConfig {
                mode: GridSampleMode::Nearest,
                align_corners: true,
                ..Default::default()
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
                4,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "grid".into(),
            Value::from_tensor_data(
                TensorData::new(vec![0.0_f32, 0.0], [1, 1, 1, 2]),
                4,
                &device,
            )
            .unwrap(),
        );

        let output = grid_sample(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [1.0]);
    }

    #[test]
    fn samples_a_bicubic_integer_coordinate() {
        let node = GridSampleNodeBuilder::new("grid_sample")
            .input_tensor("x", 4, DType::F32)
            .input_tensor("grid", 4, DType::F32)
            .output_tensor("y", 4, DType::F32)
            .config(GridSampleConfig {
                mode: GridSampleMode::Bicubic,
                padding_mode: OnnxPaddingMode::Border,
                align_corners: true,
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(TensorData::new(vec![1.0_f32], [1, 1, 1, 1]), 4, &device)
                .unwrap(),
        );
        env.insert(
            "grid".into(),
            Value::from_tensor_data(TensorData::new(vec![0.0_f32; 2], [1, 1, 1, 2]), 4, &device)
                .unwrap(),
        );

        let output = grid_sample(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [1.0]);
    }
}
