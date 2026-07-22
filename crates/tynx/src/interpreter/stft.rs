//! Short-time Fourier transform execution.

use burn::tensor::{DType, Device, Tensor, TensorData, signal::StftOptions, signal::stft};
use onnx_ir::node::stft::StftNode;

use super::{Env, resolve};
use crate::{DynTensor, Result, TynxError, Value};

pub(super) fn stft_node(node: &StftNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let signal = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    let dtype = signal.dtype();
    let signal = signal_2d(signal)?;
    let [batch, signal_length] = signal.dims();
    validate_config(node, signal_length)?;

    let window = if node.config.has_window {
        let window = resolve::at(env, &node.name, &node.inputs, 2, device)?.into_tensor()?;
        match window.cast(dtype) {
            DynTensor::R1(window) if window.dims()[0] == node.config.frame_length => Some(window),
            DynTensor::R1(window) => {
                return Err(TynxError::Shape(format!(
                    "STFT window length {} does not match frame length {}",
                    window.dims()[0],
                    node.config.frame_length
                )));
            }
            window => {
                return Err(TynxError::Shape(format!(
                    "STFT window must have rank 1, got rank {}",
                    window.rank()
                )));
            }
        }
    } else {
        None
    };

    if signal_length < node.config.frame_length {
        let frequencies = frequency_count(node);
        return Ok(vec![Value::Tensor(DynTensor::full(
            &[batch, 0, frequencies, 2],
            0.0,
            device,
            dtype,
        )?)]);
    }

    let output = if node.config.frame_length.is_power_of_two() {
        DynTensor::R4(stft(
            signal,
            window,
            StftOptions {
                n_fft: node.config.frame_length,
                hop_length: node.config.frame_step,
                win_length: None,
                center: false,
                onesided: node.config.onesided,
            },
        ))
    } else {
        matrix_dft(signal, window, node, device, dtype)?
    };
    Ok(vec![Value::Tensor(output)])
}

fn signal_2d(signal: DynTensor) -> Result<Tensor<2>> {
    match signal {
        DynTensor::R2(signal) => Ok(signal),
        DynTensor::R3(signal) => {
            let [batch, length, complex] = signal.dims();
            if complex != 1 {
                return Err(TynxError::UnsupportedOp(format!(
                    "STFT complex input with trailing dimension {complex}"
                )));
            }
            Ok(signal.reshape([batch, length]))
        }
        signal => Err(TynxError::Shape(format!(
            "STFT signal must have rank 2 or rank 3, got rank {}",
            signal.rank()
        ))),
    }
}

fn validate_config(node: &StftNode, _signal_length: usize) -> Result<()> {
    if node.config.frame_length == 0 || node.config.frame_step == 0 {
        return Err(TynxError::Shape(
            "STFT frame length and step must be positive".to_string(),
        ));
    }
    if node.config.frame_step > node.config.frame_length {
        return Err(TynxError::Shape(format!(
            "STFT frame step {} exceeds frame length {}",
            node.config.frame_step, node.config.frame_length
        )));
    }
    Ok(())
}

fn frequency_count(node: &StftNode) -> usize {
    if node.config.onesided {
        node.config.frame_length / 2 + 1
    } else {
        node.config.frame_length
    }
}

fn matrix_dft(
    signal: Tensor<2>,
    window: Option<Tensor<1>>,
    node: &StftNode,
    device: &Device,
    output_dtype: DType,
) -> Result<DynTensor> {
    let n_fft = node.config.frame_length;
    let n_freqs = frequency_count(node);
    let frames = signal.unfold(1, n_fft, node.config.frame_step);
    let window = window.unwrap_or_else(|| Tensor::<1>::ones([n_fft], (device, output_dtype)));
    let windowed = frames.mul(window.reshape([1, 1, n_fft])).cast(DType::F64);
    let [batch, n_frames, _] = windowed.dims();

    let mut real = Vec::with_capacity(n_freqs * n_fft);
    let mut imaginary = Vec::with_capacity(n_freqs * n_fft);
    for frequency in 0..n_freqs {
        for sample in 0..n_fft {
            let angle =
                2.0 * core::f64::consts::PI * frequency as f64 * sample as f64 / n_fft as f64;
            real.push(angle.cos());
            imaginary.push(-angle.sin());
        }
    }
    let real = Tensor::<2>::from_data(
        TensorData::new(real, [n_freqs, n_fft]),
        (device, DType::F64),
    )
    .transpose();
    let imaginary = Tensor::<2>::from_data(
        TensorData::new(imaginary, [n_freqs, n_fft]),
        (device, DType::F64),
    )
    .transpose();
    let flattened = windowed.reshape([batch * n_frames, n_fft]);
    let real = flattened
        .clone()
        .matmul(real)
        .reshape([batch, n_frames, n_freqs])
        .cast(output_dtype);
    let imaginary = flattened
        .matmul(imaginary)
        .reshape([batch, n_frames, n_freqs])
        .cast(output_dtype);
    Ok(DynTensor::R4(Tensor::stack(vec![real, imaginary], 3)))
}

#[cfg(test)]
mod tests {
    use onnx_ir::node::stft::{StftConfig, StftNodeBuilder};

    use super::*;

    fn node(frame_length: usize) -> StftNode {
        StftNodeBuilder::new("stft")
            .input_tensor("signal", 2, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(StftConfig {
                onesided: true,
                frame_step: frame_length,
                frame_length,
                has_window: false,
            })
            .build()
    }

    fn impulse(length: usize, device: &Device) -> Env {
        let mut values = vec![0.0_f32; length];
        values[0] = 1.0;
        let mut env = Env::new();
        env.insert(
            "signal".to_string(),
            Value::from_tensor_data(TensorData::new(values, [1, length]), 2, device).unwrap(),
        );
        env
    }

    #[test]
    fn transforms_an_impulse_with_the_fft_path() {
        let device = Device::default();
        let output = stft_node(&node(4), &impulse(4, &device), &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();

        output.into_data().assert_approx_eq(
            &TensorData::new(vec![1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0], [1, 1, 3, 2]),
            burn::tensor::Tolerance::<f32>::default(),
        );
    }

    #[test]
    fn transforms_an_impulse_with_the_matrix_path() {
        let device = Device::default();
        let output = stft_node(&node(3), &impulse(3, &device), &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();

        output.into_data().assert_approx_eq(
            &TensorData::new(vec![1.0_f32, 0.0, 1.0, 0.0], [1, 1, 2, 2]),
            burn::tensor::Tolerance::<f32>::default(),
        );
    }
}
