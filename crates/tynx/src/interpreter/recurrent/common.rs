use burn::tensor::{DType, Device, Tensor, TensorData};
use onnx_ir::ir::Argument;

use crate::{DynTensor, Result, Scalar, TynxError, Value};

use super::super::{Env, resolve};

pub(super) fn required_rank3(
    env: &Env,
    node_name: &str,
    inputs: &[Argument],
    index: usize,
    device: &Device,
) -> Result<Tensor<3>> {
    let tensor = resolve::at(env, node_name, inputs, index, device)?.into_tensor()?;
    match tensor {
        DynTensor::R3(tensor) => Ok(tensor),
        tensor => Err(TynxError::Shape(format!(
            "recurrent input {index} must have rank 3, got rank {}",
            tensor.rank()
        ))),
    }
}

pub(super) fn optional_rank2(
    env: &Env,
    inputs: &[Argument],
    index: usize,
    device: &Device,
) -> Result<Option<Tensor<2>>> {
    let Some(input) = inputs.get(index).filter(|input| !input.is_optional()) else {
        return Ok(None);
    };
    match resolve::input(env, input, device)?.into_tensor()? {
        DynTensor::R2(tensor) => Ok(Some(tensor)),
        tensor => Err(TynxError::Shape(format!(
            "recurrent input {index} must have rank 2, got rank {}",
            tensor.rank()
        ))),
    }
}

pub(super) fn optional_rank3(
    env: &Env,
    inputs: &[Argument],
    index: usize,
    device: &Device,
) -> Result<Option<Tensor<3>>> {
    let Some(input) = inputs.get(index).filter(|input| !input.is_optional()) else {
        return Ok(None);
    };
    match resolve::input(env, input, device)?.into_tensor()? {
        DynTensor::R3(tensor) => Ok(Some(tensor)),
        tensor => Err(TynxError::Shape(format!(
            "recurrent input {index} must have rank 3, got rank {}",
            tensor.rank()
        ))),
    }
}

pub(super) fn sequence_lengths(
    env: &Env,
    inputs: &[Argument],
    batch: usize,
    sequence: usize,
    device: &Device,
) -> Result<Option<Vec<usize>>> {
    let Some(input) = inputs.get(4).filter(|input| !input.is_optional()) else {
        return Ok(None);
    };
    let raw = match resolve::input(env, input, device)? {
        Value::Int(tensor) => tensor.into_data().iter::<i64>().collect(),
        Value::Shape(values) => values,
        Value::Scalar(Scalar::I64(value)) => vec![value],
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "sequence_lens must be integer values, got {other:?}"
            )));
        }
    };
    if raw.len() != batch {
        return Err(TynxError::Shape(format!(
            "sequence_lens has {} values for batch size {batch}",
            raw.len()
        )));
    }
    raw.into_iter()
        .map(|length| {
            let length = usize::try_from(length).map_err(|_| {
                TynxError::Shape(format!("sequence length {length} must be non-negative"))
            })?;
            if length > sequence {
                return Err(TynxError::Shape(format!(
                    "sequence length {length} exceeds input length {sequence}"
                )));
            }
            Ok(length)
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

pub(super) fn time_major(input: Tensor<3>, batch_first: bool) -> Tensor<3> {
    if batch_first {
        input.permute([1, 0, 2])
    } else {
        input
    }
}

pub(super) fn direction_major(state: Tensor<3>, batch_first: bool) -> Tensor<3> {
    if batch_first {
        state.permute([1, 0, 2])
    } else {
        state
    }
}

pub(super) fn initial_state(
    state: Option<Tensor<3>>,
    batch: usize,
    hidden: usize,
    dtype: DType,
    device: &Device,
    name: &str,
) -> Result<Tensor<2>> {
    match state {
        Some(state) => {
            if state.dims() != [1, batch, hidden] {
                return Err(TynxError::Shape(format!(
                    "{name} has shape {:?}, expected [1, {batch}, {hidden}]",
                    state.dims()
                )));
            }
            Ok(state.cast(dtype).reshape([batch, hidden]))
        }
        None => Ok(Tensor::<2>::zeros([batch, hidden], (device, dtype))),
    }
}

pub(super) fn validate_sequence_length(operation: &str, sequence: usize) -> Result<()> {
    if sequence == 0 {
        return Err(TynxError::Shape(format!(
            "{operation} does not support an empty sequence"
        )));
    }
    Ok(())
}

pub(super) fn output_layout(
    sequence: Tensor<4>,
    hidden: Tensor<3>,
    batch_first: bool,
) -> (DynTensor, DynTensor) {
    if batch_first {
        (
            DynTensor::R4(sequence.permute([2, 0, 1, 3])),
            DynTensor::R3(hidden.permute([1, 0, 2])),
        )
    } else {
        (DynTensor::R4(sequence), DynTensor::R3(hidden))
    }
}

pub(super) fn output_layout_lstm(
    sequence: Tensor<4>,
    hidden: Tensor<3>,
    cell: Tensor<3>,
    batch_first: bool,
) -> (DynTensor, DynTensor, DynTensor) {
    let (sequence, hidden) = output_layout(sequence, hidden, batch_first);
    let cell = if batch_first {
        cell.permute([1, 0, 2])
    } else {
        cell
    };
    (sequence, hidden, DynTensor::R3(cell))
}

pub(super) fn active_mask(
    lengths: Option<&[usize]>,
    timestep: usize,
    batch: usize,
    dtype: DType,
    device: &Device,
) -> Option<Tensor<2>> {
    lengths.map(|lengths| {
        let values = lengths
            .iter()
            .map(|length| if timestep < *length { 1.0_f32 } else { 0.0 })
            .collect::<Vec<_>>();
        Tensor::<2>::from_data(TensorData::new(values, [batch, 1]), (device, DType::F32))
            .cast(dtype)
    })
}

pub(super) fn masked_state(
    candidate: Tensor<2>,
    previous: Tensor<2>,
    mask: Option<Tensor<2>>,
) -> (Tensor<2>, Tensor<2>) {
    match mask {
        Some(mask) => {
            let active = candidate.clone().mul(mask.clone());
            let state = active
                .clone()
                .add(previous.mul(mask.mul_scalar(-1.0).add_scalar(1.0)));
            (state, active)
        }
        None => (candidate.clone(), candidate),
    }
}

pub(super) fn clip<const D: usize>(tensor: Tensor<D>, threshold: Option<f32>) -> Tensor<D> {
    match threshold {
        Some(threshold) => tensor.clamp(-(threshold as f64), threshold as f64),
        None => tensor,
    }
}

pub(super) fn validate_single_direction(
    operation: &str,
    weight_dirs: usize,
    recurrent_dirs: usize,
) -> Result<()> {
    if weight_dirs != 1 || recurrent_dirs != 1 {
        return Err(TynxError::UnsupportedOp(format!(
            "bidirectional {operation}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_batch_first_inputs_and_outputs() {
        let device = Device::default();
        let input = Tensor::<3>::zeros([2, 3, 4], (&device, DType::F32));
        let sequence = Tensor::<4>::zeros([3, 1, 2, 5], (&device, DType::F32));
        let hidden = Tensor::<3>::zeros([1, 2, 5], (&device, DType::F32));

        let input = time_major(input, true);
        let (sequence, hidden) = output_layout(sequence, hidden, true);

        assert_eq!(input.dims(), [3, 2, 4]);
        assert_eq!(sequence.dims(), [2, 3, 1, 5]);
        assert_eq!(hidden.dims(), [2, 1, 5]);
    }
}
