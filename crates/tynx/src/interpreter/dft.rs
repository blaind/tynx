//! Forward real-input ONNX DFT execution.

use burn::tensor::{Device, Slice, TensorData};
use onnx_ir::node::dft::DftNode;

use super::{Env, resolve};
use crate::{DynTensor, Result, TynxError, Value};

pub(super) fn dft(node: &DftNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    if node.config.inverse || !node.config.is_real_input {
        return Err(TynxError::UnsupportedOp(
            "DFT inverse or complex-input transform".to_string(),
        ));
    }
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    Ok(vec![Value::Tensor(forward_real(
        input,
        node.config.axis,
        node.config.dft_length,
        node.config.onesided,
        device,
    )?)])
}

fn forward_real(
    input: DynTensor,
    axis: usize,
    dft_length: Option<usize>,
    onesided: bool,
    device: &Device,
) -> Result<DynTensor> {
    let dtype = input.dtype();
    let input_dims = input.dims();
    if input_dims.len() < 2 || input_dims.last() != Some(&1) {
        return Err(TynxError::Shape(format!(
            "DFT real input must have rank >= 2 and trailing dimension 1, got {input_dims:?}"
        )));
    }
    let signal_dims = input_dims[..input_dims.len() - 1].to_vec();
    if axis >= signal_dims.len() {
        return Err(TynxError::Shape(format!(
            "DFT axis {axis} is outside signal rank {}",
            signal_dims.len()
        )));
    }
    let rank = signal_dims.len();
    let mut signal = input.reshape(signal_dims.clone())?;
    let input_length = signal_dims[axis];
    let length = dft_length.unwrap_or(input_length).max(1);

    if length < input_length {
        let end = isize::try_from(length)
            .map_err(|_| TynxError::Shape(format!("DFT length {length} exceeds isize")))?;
        let mut slices = vec![Slice::full(); rank];
        slices[axis] = Slice::new(0, Some(end), 1);
        signal = signal.slice(&slices);
    } else if length > input_length {
        let mut padding_dims = signal_dims.clone();
        padding_dims[axis] = length - input_length;
        let zeros = DynTensor::full(&padding_dims, 0.0, device, dtype)?;
        signal = DynTensor::concat(vec![signal, zeros], axis)?;
    }

    let frequencies = if onesided { length / 2 + 1 } else { length };
    let matrix_elements = length
        .checked_mul(frequencies)
        .ok_or_else(|| TynxError::Shape("DFT matrix size overflow".to_string()))?;
    let mut real_matrix = Vec::with_capacity(matrix_elements);
    let mut imaginary_matrix = Vec::with_capacity(matrix_elements);
    for sample in 0..length {
        for frequency in 0..frequencies {
            let angle =
                2.0 * core::f64::consts::PI * frequency as f64 * sample as f64 / length as f64;
            real_matrix.push(angle.cos());
            imaginary_matrix.push(-angle.sin());
        }
    }
    let real_matrix = DynTensor::from_data(
        TensorData::new(real_matrix, [length, frequencies]),
        2,
        device,
    )?
    .cast(dtype);
    let imaginary_matrix = DynTensor::from_data(
        TensorData::new(imaginary_matrix, [length, frequencies]),
        2,
        device,
    )?
    .cast(dtype);

    let permutation = (0..rank)
        .filter(|&candidate| candidate != axis)
        .chain(core::iter::once(axis))
        .collect::<Vec<_>>();
    let permuted = signal.permute(permutation.clone())?;
    let permuted_dims = permuted.dims();
    let leading = permuted_dims[..rank - 1].iter().product();
    let flattened = permuted.reshape(vec![leading, length])?;
    let mut inverse = vec![0; rank];
    for (position, &original_axis) in permutation.iter().enumerate() {
        inverse[original_axis] = position;
    }
    let contract = |matrix: DynTensor, flattened: DynTensor| -> Result<DynTensor> {
        let output = flattened.matmul(matrix)?;
        let mut dims = permuted_dims.clone();
        dims[rank - 1] = frequencies;
        output.reshape(dims)?.permute(inverse.clone())
    };
    let real = contract(real_matrix, flattened.clone())?;
    let imaginary = contract(imaginary_matrix, flattened)?;

    let mut component_dims = real.dims();
    component_dims.push(1);
    let real = real.reshape(component_dims.clone())?;
    let imaginary = imaginary.reshape(component_dims)?;
    DynTensor::concat(vec![real, imaginary], rank)
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::dft::{DftConfig, DftNodeBuilder},
    };

    use super::*;

    #[test]
    fn transforms_a_real_impulse() {
        let node = DftNodeBuilder::new("dft")
            .input_tensor("x", 3, DType::F32)
            .output_tensor("y", 3, DType::F32)
            .config(DftConfig {
                inverse: false,
                onesided: false,
                axis: 1,
                dft_length: None,
                is_real_input: true,
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 0.0, 0.0, 0.0], [1, 4, 1]),
                3,
                &device,
            )
            .unwrap(),
        );

        let output = dft(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap();

        assert_eq!(output.dims(), [1, 4, 2]);
        assert_eq!(
            output.into_data().iter::<f32>().collect::<Vec<_>>(),
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        );
    }
}
