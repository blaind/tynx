//! ONNX Resize execution for NCHW tensors.

use burn::tensor::{Device, TensorData};
use onnx_ir::node::resize::{
    CoordinateTransformMode, NearestMode, ResizeMode, ResizeNode, ResizeScales, ResizeSizes,
};

use super::{Env, resolve, shape};
use crate::{DynInt, DynTensor, Result, TynxError, Value};

pub(super) fn resize(node: &ResizeNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let dims = shape::value_dims(&input);
    if dims.len() != 4 {
        return Err(TynxError::UnsupportedOp(format!(
            "Resize rank {} (only NCHW rank 4 is supported)",
            dims.len()
        )));
    }
    let (output_size, scales) = resolve_geometry(node, env, device, &dims)?;
    let crop_roi = resolve_crop_roi(node, env, device)?;

    let output = match node.config.mode {
        ResizeMode::Nearest => {
            nearest_resize(node, input, &dims, output_size, scales, crop_roi, device)?
        }
        ResizeMode::Linear | ResizeMode::Cubic => match input {
            Value::Tensor(tensor) => Value::Tensor(interpolate_2d(
                tensor,
                &dims,
                output_size,
                scales,
                crop_roi,
                node,
                device,
            )?),
            other => {
                return Err(TynxError::TypeMismatch(format!(
                    "linear and cubic Resize require a float tensor, got {other:?}"
                )));
            }
        },
    };
    Ok(vec![output])
}

fn resolve_geometry(
    node: &ResizeNode,
    env: &Env,
    device: &Device,
    dims: &[usize],
) -> Result<([usize; 2], [f64; 2])> {
    if let Some(sizes) = &node.config.sizes {
        let sizes = match sizes {
            ResizeSizes::Static(sizes) => sizes.clone(),
            ResizeSizes::Runtime(reference) => shape::value_to_i64s(resolve::at(
                env,
                &node.name,
                &node.inputs,
                reference.input_index,
                device,
            )?)?
            .into_iter()
            .map(|size| {
                usize::try_from(size).map_err(|_| {
                    TynxError::Shape(format!("Resize size must be non-negative, got {size}"))
                })
            })
            .collect::<Result<Vec<_>>>()?,
        };
        let spatial = spatial_values(&sizes, "sizes")?;
        if spatial[0] == 0 || spatial[1] == 0 {
            return Err(TynxError::Shape(
                "Resize output sizes must be positive".into(),
            ));
        }
        return Ok((
            spatial,
            [
                spatial[0] as f64 / dims[2] as f64,
                spatial[1] as f64 / dims[3] as f64,
            ],
        ));
    }

    let scales = match node.config.scales.as_ref() {
        Some(ResizeScales::Static(scales)) => scales.clone(),
        Some(ResizeScales::Runtime(reference)) => float_values(
            resolve::at(env, &node.name, &node.inputs, reference.input_index, device)?,
            "scales",
        )?,
        None => return Err(TynxError::Shape("Resize requires scales or sizes".into())),
    };
    let scales = spatial_values(&scales, "scales")?;
    if !scales.iter().all(|scale| scale.is_finite() && *scale > 0.0) {
        return Err(TynxError::Shape(format!(
            "Resize scales must be finite and positive, got {scales:?}"
        )));
    }
    let scales = [scales[0] as f64, scales[1] as f64];
    Ok((
        [
            (dims[2] as f64 * scales[0]).floor() as usize,
            (dims[3] as f64 * scales[1]).floor() as usize,
        ],
        scales,
    ))
}

fn spatial_values<T: Copy>(values: &[T], name: &str) -> Result<[T; 2]> {
    match values {
        [height, width] => Ok([*height, *width]),
        [_, _, height, width] => Ok([*height, *width]),
        _ => Err(TynxError::Shape(format!(
            "Resize {name} must have 2 spatial or 4 NCHW values, got {}",
            values.len()
        ))),
    }
}

fn float_values(value: Value, name: &str) -> Result<Vec<f32>> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.into_data().iter::<f32>().collect()),
        Value::Shape(values) => Ok(values.into_iter().map(|value| value as f32).collect()),
        other => Err(TynxError::TypeMismatch(format!(
            "Resize {name} must be a float tensor, got {other:?}"
        ))),
    }
}

#[derive(Debug, Clone, Copy)]
struct CropRoi {
    height: [f64; 2],
    width: [f64; 2],
}

impl CropRoi {
    fn for_axis(self, axis: usize) -> Option<[f64; 2]> {
        match axis {
            2 => Some(self.height),
            3 => Some(self.width),
            _ => None,
        }
    }
}

fn resolve_crop_roi(node: &ResizeNode, env: &Env, device: &Device) -> Result<Option<CropRoi>> {
    if !matches!(
        node.config.coordinate_transformation_mode,
        CoordinateTransformMode::TfCropAndResize
    ) {
        return Ok(None);
    }
    if node.config.extrapolation_value != 0.0 {
        return Err(TynxError::UnsupportedOp(
            "Resize tf_crop_and_resize with nonzero extrapolation".into(),
        ));
    }
    let roi = float_values(
        resolve::at(env, &node.name, &node.inputs, 1, device)?,
        "roi",
    )?;
    let [
        n_start,
        c_start,
        h_start,
        w_start,
        n_end,
        c_end,
        h_end,
        w_end,
    ] = roi.as_slice()
    else {
        return Err(TynxError::Shape(format!(
            "Resize tf_crop_and_resize ROI must have 8 NCHW values, got {}",
            roi.len()
        )));
    };
    if !roi.iter().all(|value| value.is_finite()) {
        return Err(TynxError::Shape(
            "Resize tf_crop_and_resize ROI must be finite".into(),
        ));
    }
    if (*n_start, *c_start, *n_end, *c_end) != (0.0, 0.0, 1.0, 1.0) {
        return Err(TynxError::UnsupportedOp(
            "Resize tf_crop_and_resize over batch or channel axes".into(),
        ));
    }
    Ok(Some(CropRoi {
        height: [f64::from(*h_start), f64::from(*h_end)],
        width: [f64::from(*w_start), f64::from(*w_end)],
    }))
}

fn nearest_resize(
    node: &ResizeNode,
    input: Value,
    input_dims: &[usize],
    output_size: [usize; 2],
    scales: [f64; 2],
    crop_roi: Option<CropRoi>,
    device: &Device,
) -> Result<Value> {
    let height = nearest_indices(
        input_dims[2],
        output_size[0],
        scales[0],
        &node.config.coordinate_transformation_mode,
        &node.config.nearest_mode,
        crop_roi.and_then(|roi| roi.for_axis(2)),
    )?;
    let width = nearest_indices(
        input_dims[3],
        output_size[1],
        scales[1],
        &node.config.coordinate_transformation_mode,
        &node.config.nearest_mode,
        crop_roi.and_then(|roi| roi.for_axis(3)),
    )?;
    let height = DynInt::from_data(TensorData::new(height, [output_size[0]]), 1, device)?;
    let width = DynInt::from_data(TensorData::new(width, [output_size[1]]), 1, device)?;

    match input {
        Value::Tensor(tensor) => Ok(Value::Tensor(tensor.select(2, height)?.select(3, width)?)),
        Value::Int(tensor) => Ok(Value::Int(tensor.select(2, height)?.select(3, width)?)),
        Value::Bool(tensor) => Ok(Value::Bool(tensor.select(2, height)?.select(3, width)?)),
        other => Err(TynxError::TypeMismatch(format!(
            "Resize expects a tensor, got {other:?}"
        ))),
    }
}

fn nearest_indices(
    input_size: usize,
    output_size: usize,
    scale: f64,
    transform: &CoordinateTransformMode,
    nearest: &NearestMode,
    crop_roi: Option<[f64; 2]>,
) -> Result<Vec<i64>> {
    if input_size == 0 || output_size == 0 {
        return Err(TynxError::Shape(
            "Resize dimensions must be positive".into(),
        ));
    }
    (0..output_size)
        .map(|output| {
            let source =
                source_coordinate(output, input_size, output_size, scale, transform, crop_roi)?;
            if crop_roi.is_some() && !(0.0..=(input_size - 1) as f64).contains(&source) {
                return Err(TynxError::UnsupportedOp(
                    "nearest Resize tf_crop_and_resize extrapolation".into(),
                ));
            }
            let index = match nearest {
                NearestMode::RoundPreferFloor => (source - 0.5).ceil(),
                NearestMode::RoundPreferCeil => (source + 0.5).floor(),
                NearestMode::Floor => source.floor(),
                NearestMode::Ceil => source.ceil(),
            };
            Ok(index.clamp(0.0, (input_size - 1) as f64) as i64)
        })
        .collect()
}

fn interpolate_2d(
    tensor: DynTensor,
    input_dims: &[usize],
    output_size: [usize; 2],
    scales: [f64; 2],
    crop_roi: Option<CropRoi>,
    node: &ResizeNode,
    device: &Device,
) -> Result<DynTensor> {
    let tensor = interpolate_axis(
        tensor,
        2,
        input_dims[2],
        output_size[0],
        scales[0],
        crop_roi.and_then(|roi| roi.for_axis(2)),
        node,
        device,
    )?;
    interpolate_axis(
        tensor,
        3,
        input_dims[3],
        output_size[1],
        scales[1],
        crop_roi.and_then(|roi| roi.for_axis(3)),
        node,
        device,
    )
}

#[allow(clippy::too_many_arguments)]
fn interpolate_axis(
    tensor: DynTensor,
    axis: usize,
    input_size: usize,
    output_size: usize,
    scale: f64,
    crop_roi: Option<[f64; 2]>,
    node: &ResizeNode,
    device: &Device,
) -> Result<DynTensor> {
    let tap_count = match node.config.mode {
        ResizeMode::Linear => 2,
        ResizeMode::Cubic => 4,
        ResizeMode::Nearest => unreachable!(),
    };
    let mut tap_indices = vec![Vec::with_capacity(output_size); tap_count];
    let mut tap_weights = vec![Vec::with_capacity(output_size); tap_count];
    for output in 0..output_size {
        let source = source_coordinate(
            output,
            input_size,
            output_size,
            scale,
            &node.config.coordinate_transformation_mode,
            crop_roi,
        )?;
        let extrapolated =
            crop_roi.is_some() && !(0.0..=(input_size.saturating_sub(1)) as f64).contains(&source);
        let base = source.floor() as i64;
        if tap_count == 2 {
            let fraction = source - source.floor();
            tap_indices[0].push(clamp_index(base, input_size));
            tap_indices[1].push(clamp_index(base + 1, input_size));
            tap_weights[0].push(if extrapolated {
                0.0
            } else {
                (1.0 - fraction) as f32
            });
            tap_weights[1].push(if extrapolated { 0.0 } else { fraction as f32 });
        } else {
            for tap in 0..4 {
                let index = base + tap as i64 - 1;
                tap_indices[tap].push(clamp_index(index, input_size));
                tap_weights[tap].push(if extrapolated {
                    0.0
                } else {
                    cubic_weight(source - index as f64, node.config.cubic_coeff_a as f64) as f32
                });
            }
        }
    }

    let dtype = tensor.dtype();
    let mut output: Option<DynTensor> = None;
    for (indices, weights) in tap_indices.into_iter().zip(tap_weights) {
        let indices = DynInt::from_data(TensorData::new(indices, [output_size]), 1, device)?;
        let selected = tensor.clone().select(axis, indices)?;
        let mut weight_dims = vec![1; 4];
        weight_dims[axis] = output_size;
        let weights = DynTensor::from_data(TensorData::new(weights, [output_size]), 1, device)?
            .cast(dtype)
            .reshape(weight_dims)?;
        let weighted = selected.mul_broadcast(weights)?;
        output = Some(match output {
            Some(accumulated) => accumulated.add_broadcast(weighted)?,
            None => weighted,
        });
    }
    output.ok_or_else(|| TynxError::Shape("Resize interpolation has no taps".into()))
}

fn source_coordinate(
    output: usize,
    input_size: usize,
    output_size: usize,
    scale: f64,
    transform: &CoordinateTransformMode,
    crop_roi: Option<[f64; 2]>,
) -> Result<f64> {
    let output = output as f64;
    Ok(match transform {
        CoordinateTransformMode::HalfPixel => (output + 0.5) / scale - 0.5,
        CoordinateTransformMode::PytorchHalfPixel => {
            if output_size > 1 {
                (output + 0.5) / scale - 0.5
            } else {
                0.0
            }
        }
        CoordinateTransformMode::AlignCorners => {
            let nominal_output_size = input_size as f64 * scale;
            if output_size == 1 || nominal_output_size <= 1.0 {
                0.0
            } else {
                output * (input_size - 1) as f64 / (nominal_output_size - 1.0)
            }
        }
        CoordinateTransformMode::Asymmetric => output / scale,
        CoordinateTransformMode::TfHalfPixelForNn => (output + 0.5) / scale,
        CoordinateTransformMode::TfCropAndResize => {
            let [start, end] = crop_roi.ok_or_else(|| {
                TynxError::Shape("Resize tf_crop_and_resize is missing ROI values".into())
            })?;
            let last = input_size.saturating_sub(1) as f64;
            if output_size > 1 {
                start * last + output * (end - start) * last / (output_size - 1) as f64
            } else {
                0.5 * (start + end) * last
            }
        }
    })
}

fn clamp_index(index: i64, size: usize) -> i64 {
    index.clamp(0, size as i64 - 1)
}

fn cubic_weight(distance: f64, coefficient: f64) -> f64 {
    let distance = distance.abs();
    if distance <= 1.0 {
        (coefficient + 2.0) * distance.powi(3) - (coefficient + 3.0) * distance.powi(2) + 1.0
    } else if distance < 2.0 {
        coefficient * distance.powi(3) - 5.0 * coefficient * distance.powi(2)
            + 8.0 * coefficient * distance
            - 4.0 * coefficient
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use onnx_ir::{
        DType,
        ir::RuntimeInputRef,
        node::resize::{ResizeConfig, ResizeNodeBuilder},
    };

    use super::*;

    #[test]
    fn linearly_resizes_a_spatial_crop() {
        let node = ResizeNodeBuilder::new("resize")
            .input_tensor("x", 4, DType::F32)
            .input_tensor("roi", 1, DType::F32)
            .input_tensor("sizes", 1, DType::I64)
            .output_tensor("y", 4, DType::F32)
            .config(ResizeConfig {
                mode: ResizeMode::Linear,
                sizes: Some(ResizeSizes::Runtime(RuntimeInputRef {
                    name: "sizes".into(),
                    input_index: 2,
                })),
                coordinate_transformation_mode: CoordinateTransformMode::TfCropAndResize,
                ..Default::default()
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(
                    (1..=16).map(|value| value as f32).collect::<Vec<_>>(),
                    [1, 1, 4, 4],
                ),
                4,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "roi".into(),
            Value::from_tensor_data(
                TensorData::new(vec![0.0_f32, 0.0, 0.4, 0.6, 1.0, 1.0, 0.6, 0.8], [8]),
                1,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "sizes".into(),
            Value::from_tensor_data(TensorData::new(vec![1_i64, 1, 3, 3], [4]), 1, &device)
                .unwrap(),
        );

        let output = resize(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();

        assert_eq!(output.dims(), [1, 1, 3, 3]);
        let output = output.into_data().iter::<f32>().collect::<Vec<_>>();
        let expected = [7.6_f32, 7.9, 8.2, 8.8, 9.1, 9.4, 10.0, 10.3, 10.6];
        assert!(
            output
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() < 1e-5)
        );
    }

    #[test]
    fn nearest_asymmetric_floor_repeats_source_cells() {
        assert_eq!(
            nearest_indices(
                2,
                4,
                2.0,
                &CoordinateTransformMode::Asymmetric,
                &NearestMode::Floor,
                None,
            )
            .unwrap(),
            [0, 0, 1, 1]
        );
    }

    #[test]
    fn nearest_align_corners_reaches_both_edges() {
        assert_eq!(
            nearest_indices(
                3,
                5,
                5.0 / 3.0,
                &CoordinateTransformMode::AlignCorners,
                &NearestMode::RoundPreferFloor,
                None,
            )
            .unwrap(),
            [0, 0, 1, 1, 2]
        );
    }

    #[test]
    fn cubic_weights_interpolate_and_sum_to_one() {
        let weights = (-1..=2)
            .map(|index| cubic_weight(0.25 - index as f64, -0.75))
            .sum::<f64>();
        assert!((weights - 1.0).abs() < 1e-12);
    }
}
