//! ONNX signal window generation.

use burn::tensor::{Device, signal};
use onnx_ir::{
    DType,
    ir::Argument,
    node::{
        blackman_window::BlackmanWindowNode,
        hamming_window::HammingWindowNode,
        hann_window::{HannWindowNode, WindowSize},
    },
};

use super::{Env, resolve, shape};
use crate::{DynTensor, Result, TynxError, Value};

pub(super) fn hann_window(node: &HannWindowNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    generate(
        &node.name,
        &node.inputs,
        &node.config.size,
        node.config.periodic,
        node.config.output_dtype,
        WindowKind::Hann,
        env,
        device,
    )
}

pub(super) fn hamming_window(
    node: &HammingWindowNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    generate(
        &node.name,
        &node.inputs,
        &node.config.size,
        node.config.periodic,
        node.config.output_dtype,
        WindowKind::Hamming,
        env,
        device,
    )
}

pub(super) fn blackman_window(
    node: &BlackmanWindowNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    generate(
        &node.name,
        &node.inputs,
        &node.config.size,
        node.config.periodic,
        node.config.output_dtype,
        WindowKind::Blackman,
        env,
        device,
    )
}

#[derive(Debug, Clone, Copy)]
enum WindowKind {
    Hann,
    Hamming,
    Blackman,
}

#[allow(clippy::too_many_arguments)]
fn generate(
    node_name: &str,
    inputs: &[Argument],
    size: &WindowSize,
    periodic: bool,
    dtype: DType,
    kind: WindowKind,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let size = resolve_size(node_name, inputs, size, env, device)?;
    i64::try_from(size).map_err(|_| TynxError::Shape(format!("window size {size} exceeds i64")))?;
    let output = match kind {
        WindowKind::Hann => signal::hann_window(size, periodic, (device, dtype)),
        WindowKind::Hamming => signal::hamming_window(size, periodic, (device, dtype)),
        WindowKind::Blackman => signal::blackman_window(size, periodic, (device, dtype)),
    };
    Ok(vec![Value::Tensor(DynTensor::R1(output))])
}

fn resolve_size(
    node_name: &str,
    inputs: &[Argument],
    size: &WindowSize,
    env: &Env,
    device: &Device,
) -> Result<usize> {
    match size {
        WindowSize::Static(size) => Ok(*size),
        WindowSize::Runtime(reference) => {
            let values = shape::value_to_i64s(resolve::at(
                env,
                node_name,
                inputs,
                reference.input_index,
                device,
            )?)?;
            let value = values.first().copied().ok_or_else(|| {
                TynxError::Shape("window size input must contain one value".to_string())
            })?;
            usize::try_from(value).map_err(|_| {
                TynxError::Shape(format!("window size must be non-negative, got {value}"))
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::hamming_window::{HammingWindowConfig, HammingWindowNodeBuilder, WindowSize},
    };

    use super::*;

    #[test]
    fn creates_a_symmetric_hamming_window() {
        let node = HammingWindowNodeBuilder::new("hamming")
            .output_tensor("y", 1, DType::F32)
            .config(HammingWindowConfig {
                periodic: false,
                output_dtype: DType::F32,
                size: WindowSize::Static(5),
            })
            .build();

        let output = hamming_window(&node, &Env::new(), &Device::default())
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data();

        output.assert_approx_eq::<f32>(
            &TensorData::new(
                vec![0.0869565_f32, 0.54347825, 1.0, 0.54347825, 0.0869565],
                [5],
            ),
            burn::tensor::Tolerance::default(),
        );
    }
}
