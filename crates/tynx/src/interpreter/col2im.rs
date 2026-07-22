//! Col2Im execution through indexed scatter-add.

use burn::tensor::{DType, Device, IndexingUpdateOp, Slice, TensorData};
use onnx_ir::node::col2im::{Col2ImConfig, Col2ImNode};

use super::{Env, resolve};
use crate::{DynInt, DynTensor, Result, TynxError, Value};

pub(super) fn col2im(node: &Col2ImNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    let input_dims = input.dims();
    if input_dims.len() != 3 {
        return Err(TynxError::Shape(format!(
            "Col2Im requires rank-3 [batch, channels*blocks, windows] input, got rank {}",
            input_dims.len()
        )));
    }
    let geometry = Geometry::new(&node.config)?;
    if input_dims[1] % geometry.block_product != 0 {
        return Err(TynxError::Shape(format!(
            "Col2Im input channel dimension {} is not divisible by block size {}",
            input_dims[1], geometry.block_product
        )));
    }
    if input_dims[2] != geometry.total_windows {
        return Err(TynxError::Shape(format!(
            "Col2Im input has {} windows, expected {}",
            input_dims[2], geometry.total_windows
        )));
    }

    let batch = input_dims[0];
    let channels = input_dims[1] / geometry.block_product;
    let flattened = geometry
        .block_product
        .checked_mul(geometry.total_windows)
        .ok_or_else(|| TynxError::Shape("Col2Im flattened input size overflows".to_string()))?;
    let input = input.reshape(vec![batch, channels, flattened])?;
    let dtype = input.dtype();
    let canvas = DynTensor::full(&[batch, channels, geometry.padded_size], 0.0, device, dtype)?;
    let indices = DynInt::from_data(TensorData::new(geometry.indices()?, [flattened]), 1, device)?
        .cast(DType::I64)
        .reshape(vec![1, 1, flattened])?
        .expand(&[batch, channels, flattened])?;
    let output = canvas.scatter(2, indices, input, IndexingUpdateOp::Add)?;

    let mut padded_shape = vec![batch, channels];
    padded_shape.extend_from_slice(&geometry.padded_dims);
    let output = output.reshape(padded_shape)?;
    let output = if geometry.has_padding() {
        let mut slices = vec![Slice::full(), Slice::full()];
        for axis in 0..geometry.spatial_rank {
            let start = geometry.pads_begin[axis];
            let end = start + node.config.image_shape[axis];
            slices.push(Slice::new(start as isize, Some(end as isize), 1));
        }
        output.slice(&slices)
    } else {
        output
    };
    Ok(vec![Value::Tensor(output)])
}

struct Geometry<'a> {
    config: &'a Col2ImConfig,
    spatial_rank: usize,
    pads_begin: Vec<usize>,
    pads_end: Vec<usize>,
    padded_dims: Vec<usize>,
    output_counts: Vec<usize>,
    block_product: usize,
    total_windows: usize,
    padded_size: usize,
}

impl<'a> Geometry<'a> {
    fn new(config: &'a Col2ImConfig) -> Result<Self> {
        let spatial_rank = config.image_shape.len();
        if !(1..=2).contains(&spatial_rank) {
            return Err(TynxError::UnsupportedOp(format!(
                "Col2Im with {spatial_rank} spatial dimensions"
            )));
        }
        if config.block_shape.len() != spatial_rank
            || config.dilations.len() != spatial_rank
            || config.strides.len() != spatial_rank
            || config.pads.len() != spatial_rank * 2
        {
            return Err(TynxError::Shape(
                "Col2Im image, block, dilation, stride, and pad ranks are inconsistent".to_string(),
            ));
        }
        if config.image_shape.contains(&0)
            || config.block_shape.contains(&0)
            || config.dilations.contains(&0)
            || config.strides.contains(&0)
        {
            return Err(TynxError::Shape(
                "Col2Im image, block, dilation, and stride values must be positive".to_string(),
            ));
        }

        let pads_begin = config.pads[..spatial_rank].to_vec();
        let pads_end = config.pads[spatial_rank..].to_vec();
        let mut padded_dims = Vec::with_capacity(spatial_rank);
        let mut output_counts = Vec::with_capacity(spatial_rank);
        for axis in 0..spatial_rank {
            let padded = config.image_shape[axis]
                .checked_add(pads_begin[axis])
                .and_then(|value| value.checked_add(pads_end[axis]))
                .ok_or_else(|| TynxError::Shape("Col2Im padded size overflows".to_string()))?;
            let effective = config.dilations[axis]
                .checked_mul(config.block_shape[axis] - 1)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| TynxError::Shape("Col2Im effective block overflows".to_string()))?;
            if effective > padded {
                return Err(TynxError::Shape(format!(
                    "Col2Im effective block {effective} exceeds padded dimension {padded} on axis {axis}"
                )));
            }
            padded_dims.push(padded);
            output_counts.push((padded - effective) / config.strides[axis] + 1);
        }

        Ok(Self {
            config,
            spatial_rank,
            pads_begin,
            pads_end,
            block_product: checked_product(&config.block_shape, "block")?,
            total_windows: checked_product(&output_counts, "window")?,
            padded_size: checked_product(&padded_dims, "padded output")?,
            padded_dims,
            output_counts,
        })
    }

    fn has_padding(&self) -> bool {
        self.pads_begin
            .iter()
            .chain(&self.pads_end)
            .any(|&pad| pad != 0)
    }

    fn indices(&self) -> Result<Vec<i64>> {
        let mut indices = Vec::with_capacity(self.block_product * self.total_windows);
        for block in 0..self.block_product {
            let block_coords = unravel(block, &self.config.block_shape);
            for window in 0..self.total_windows {
                let window_coords = unravel(window, &self.output_counts);
                let mut index = 0usize;
                for axis in 0..self.spatial_rank {
                    let coordinate = window_coords[axis]
                        .checked_mul(self.config.strides[axis])
                        .and_then(|value| {
                            block_coords[axis]
                                .checked_mul(self.config.dilations[axis])
                                .and_then(|block| value.checked_add(block))
                        })
                        .ok_or_else(|| {
                            TynxError::Shape("Col2Im output index overflows".to_string())
                        })?;
                    index = index
                        .checked_mul(self.padded_dims[axis])
                        .and_then(|value| value.checked_add(coordinate))
                        .ok_or_else(|| {
                            TynxError::Shape("Col2Im flattened index overflows".to_string())
                        })?;
                }
                indices.push(
                    i64::try_from(index).map_err(|_| {
                        TynxError::Shape("Col2Im index exceeds i64 range".to_string())
                    })?,
                );
            }
        }
        Ok(indices)
    }
}

fn checked_product(values: &[usize], name: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |product, &value| {
        product
            .checked_mul(value)
            .ok_or_else(|| TynxError::Shape(format!("Col2Im {name} size overflows")))
    })
}

fn unravel(mut index: usize, dims: &[usize]) -> Vec<usize> {
    let mut coordinates = vec![0; dims.len()];
    for axis in (0..dims.len()).rev() {
        coordinates[axis] = index % dims[axis];
        index /= dims[axis];
    }
    coordinates
}

#[cfg(test)]
mod tests {
    use onnx_ir::node::col2im::{Col2ImConfig, Col2ImNodeBuilder};

    use super::*;

    #[test]
    fn folds_overlapping_one_dimensional_blocks() {
        let node = Col2ImNodeBuilder::new("col2im")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(Col2ImConfig::new(
                vec![4],
                vec![2],
                vec![1],
                vec![0, 0],
                vec![1],
            ))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "input".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3]),
                3,
                &device,
            )
            .unwrap(),
        );

        let output = col2im(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();

        output.into_data().assert_eq(
            &TensorData::new(vec![1.0_f32, 6.0, 8.0, 6.0], [1, 1, 4]),
            false,
        );
    }

    #[test]
    fn crops_padding_after_scatter() {
        let node = Col2ImNodeBuilder::new("col2im")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(Col2ImConfig::new(
                vec![3],
                vec![1],
                vec![1],
                vec![1, 1],
                vec![1],
            ))
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "input".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![10.0_f32, 1.0, 2.0, 3.0, 20.0], [1, 1, 5]),
                3,
                &device,
            )
            .unwrap(),
        );

        let output = col2im(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();

        output
            .into_data()
            .assert_eq(&TensorData::new(vec![1.0_f32, 2.0, 3.0], [1, 1, 3]), false);
    }
}
