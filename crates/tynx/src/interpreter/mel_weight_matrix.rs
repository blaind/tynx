//! ONNX MelWeightMatrix execution.

use burn::tensor::{Device, TensorData};
use onnx_ir::node::mel_weight_matrix::MelWeightMatrixNode;

use super::{Env, resolve};
use crate::{DynTensor, Result, Scalar, TynxError, Value};

pub(super) fn mel_weight_matrix(
    node: &MelWeightMatrixNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let num_mel_bins = integer_input(node, env, 0, device, "num_mel_bins")?;
    let dft_length = integer_input(node, env, 1, device, "dft_length")?;
    let sample_rate = integer_input(node, env, 2, device, "sample_rate")?;
    let lower_hertz = float_input(node, env, 3, device, "lower_edge_hertz")?;
    let upper_hertz = float_input(node, env, 4, device, "upper_edge_hertz")?;
    if num_mel_bins < 0 || dft_length < 0 || sample_rate <= 0 {
        return Err(TynxError::Shape(format!(
            "MelWeightMatrix requires non-negative bin/DFT counts and a positive sample rate, got {num_mel_bins}, {dft_length}, {sample_rate}"
        )));
    }
    if !lower_hertz.is_finite()
        || !upper_hertz.is_finite()
        || lower_hertz < 0.0
        || upper_hertz <= lower_hertz
    {
        return Err(TynxError::Shape(format!(
            "MelWeightMatrix requires 0 <= lower_edge_hertz < upper_edge_hertz, got {lower_hertz} and {upper_hertz}"
        )));
    }
    let num_mel_bins = usize::try_from(num_mel_bins)
        .map_err(|_| TynxError::Shape("num_mel_bins exceeds usize".to_string()))?;
    let spectrogram_bins_i64 = dft_length
        .checked_div(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| TynxError::Shape("spectrogram bin count overflow".to_string()))?;
    let spectrogram_bins = usize::try_from(spectrogram_bins_i64)
        .map_err(|_| TynxError::Shape("spectrogram bin count exceeds usize".to_string()))?;
    let edge_count = num_mel_bins
        .checked_add(2)
        .ok_or_else(|| TynxError::Shape("mel edge count overflow".to_string()))?;
    let low_mel = 2595.0_f64 * (1.0 + lower_hertz / 700.0).log10();
    let high_mel = 2595.0_f64 * (1.0 + upper_hertz / 700.0).log10();
    let mel_step = (high_mel - low_mel) / edge_count as f64;
    let dft_scale = dft_length
        .checked_add(1)
        .ok_or_else(|| TynxError::Shape("DFT length overflow".to_string()))?
        as f64
        / sample_rate as f64;
    let edges = (0..edge_count)
        .map(|index| {
            let mel = low_mel + index as f64 * mel_step;
            let hertz = 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0);
            (dft_scale * hertz).floor() as i64
        })
        .collect::<Vec<_>>();
    let element_count = spectrogram_bins
        .checked_mul(num_mel_bins)
        .ok_or_else(|| TynxError::Shape("mel matrix element count overflow".to_string()))?;
    let mut values = vec![0.0_f32; element_count];
    for mel_bin in 0..num_mel_bins {
        let lower = edges[mel_bin];
        let center = edges[mel_bin + 1];
        let upper = edges[mel_bin + 2];
        let rise = center - lower;
        if rise == 0 {
            if let Ok(center) = usize::try_from(center)
                && center < spectrogram_bins
            {
                values[center * num_mel_bins + mel_bin] = 1.0;
            }
        } else if rise > 0 {
            let start = usize::try_from(lower.max(0)).unwrap_or(usize::MAX);
            let end = center.clamp(0, spectrogram_bins.saturating_sub(1) as i64) as usize;
            if start <= end && center >= 0 {
                for frequency_bin in start..=end {
                    values[frequency_bin * num_mel_bins + mel_bin] =
                        (frequency_bin as f64 - lower as f64) as f32 / rise as f32;
                }
            }
        }
        let fall = upper - center;
        if fall > 0 {
            let start = usize::try_from(center.max(0)).unwrap_or(usize::MAX);
            let end = upper.clamp(0, spectrogram_bins as i64) as usize;
            if start < end && upper > 0 {
                for frequency_bin in start..end {
                    values[frequency_bin * num_mel_bins + mel_bin] =
                        (upper as f64 - frequency_bin as f64) as f32 / fall as f32;
                }
            }
        }
    }
    let output = DynTensor::from_data(
        TensorData::new(values, [spectrogram_bins, num_mel_bins]),
        2,
        device,
    )?
    .cast(node.config.output_dtype);
    Ok(vec![Value::Tensor(output)])
}

fn integer_input(
    node: &MelWeightMatrixNode,
    env: &Env,
    index: usize,
    device: &Device,
    name: &str,
) -> Result<i64> {
    let value = resolve::at(env, &node.name, &node.inputs, index, device)?;
    match value {
        Value::Scalar(Scalar::I64(value)) => Ok(value),
        Value::Scalar(Scalar::U64(value)) => {
            i64::try_from(value).map_err(|_| TynxError::Shape(format!("{name} exceeds i64")))
        }
        Value::Int(value) => one(value.into_data().iter::<i64>(), name),
        other => Err(TynxError::TypeMismatch(format!(
            "{name} must be an integer scalar, got {other:?}"
        ))),
    }
}

fn float_input(
    node: &MelWeightMatrixNode,
    env: &Env,
    index: usize,
    device: &Device,
    name: &str,
) -> Result<f64> {
    let value = resolve::at(env, &node.name, &node.inputs, index, device)?;
    match value {
        Value::Scalar(Scalar::F64(value)) => Ok(value),
        Value::Tensor(value) => one(value.into_data().iter::<f64>(), name),
        other => Err(TynxError::TypeMismatch(format!(
            "{name} must be a floating-point scalar, got {other:?}"
        ))),
    }
}

fn one<T>(mut values: impl Iterator<Item = T>, name: &str) -> Result<T> {
    let value = values
        .next()
        .ok_or_else(|| TynxError::Shape(format!("{name} is empty")))?;
    if values.next().is_some() {
        return Err(TynxError::Shape(format!(
            "{name} must contain exactly one value"
        )));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use onnx_ir::{
        DType,
        node::mel_weight_matrix::{MelWeightMatrixConfig, MelWeightMatrixNodeBuilder},
    };

    use super::*;

    #[test]
    fn creates_an_explicitly_typed_filter_bank() {
        let node = MelWeightMatrixNodeBuilder::new("mel")
            .input_scalar("mel_bins", DType::I64)
            .input_scalar("dft", DType::I64)
            .input_scalar("rate", DType::I64)
            .input_scalar("lower", DType::F32)
            .input_scalar("upper", DType::F32)
            .output_tensor("weights", 2, DType::F64)
            .config(MelWeightMatrixConfig {
                output_dtype: DType::F64,
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert("mel_bins".into(), Value::Scalar(Scalar::I64(4)));
        env.insert("dft".into(), Value::Scalar(Scalar::I64(16)));
        env.insert("rate".into(), Value::Scalar(Scalar::I64(16_000)));
        env.insert("lower".into(), Value::Scalar(Scalar::F64(0.0)));
        env.insert("upper".into(), Value::Scalar(Scalar::F64(8_000.0)));

        let output = mel_weight_matrix(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();

        assert_eq!(output.dims(), [9, 4]);
        assert_eq!(output.dtype(), DType::F64);
        assert!(output.into_data().iter::<f64>().any(|value| value > 0.0));
    }
}
