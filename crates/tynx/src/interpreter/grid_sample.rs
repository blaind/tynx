//! ONNX GridSample execution.

use burn::tensor::{
    Device, Slice, Tensor,
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
                "GridSample bicubic interpolation".to_string(),
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
    fn rejects_bicubic_without_panicking() {
        let node = GridSampleNodeBuilder::new("grid_sample")
            .input_tensor("x", 4, DType::F32)
            .input_tensor("grid", 4, DType::F32)
            .output_tensor("y", 4, DType::F32)
            .config(GridSampleConfig {
                mode: GridSampleMode::Bicubic,
                ..Default::default()
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

        let error = grid_sample(&node, &env, &device).unwrap_err();

        assert_eq!(
            error,
            TynxError::UnsupportedOp("GridSample bicubic interpolation".to_string())
        );
    }
}
