//! ONNX CumSum execution.

use burn::tensor::Device;
use onnx_ir::node::cumsum::{CumSumAxis, CumSumNode};

use super::{Env, resolve, shape};
use crate::{Result, TynxError, Value};

pub(super) fn cumsum(node: &CumSumNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let rank = shape::value_dims(&input).len();
    let axis = match &node.config.axis {
        CumSumAxis::Static(axis) => *axis,
        CumSumAxis::Runtime(reference) => {
            let values = shape::value_to_i64s(resolve::at(
                env,
                &node.name,
                &node.inputs,
                reference.input_index,
                device,
            )?)?;
            if values.len() != 1 {
                return Err(TynxError::Shape(format!(
                    "CumSum axis must have one value, got {}",
                    values.len()
                )));
            }
            normalize_axis(values[0], rank)?
        }
    };
    if axis >= rank {
        return Err(TynxError::Shape(format!(
            "CumSum axis {axis} is outside rank {rank}"
        )));
    }

    let output = match input {
        Value::Tensor(tensor) => {
            Value::Tensor(tensor.cumsum(axis, node.config.exclusive, node.config.reverse))
        }
        Value::Int(tensor) => {
            Value::Int(tensor.cumsum(axis, node.config.exclusive, node.config.reverse))
        }
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "CumSum expects a numeric tensor, got {other:?}"
            )));
        }
    };
    Ok(vec![output])
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize> {
    let rank = rank as i64;
    let axis = if axis < 0 { axis + rank } else { axis };
    if !(0..rank).contains(&axis) {
        return Err(TynxError::Shape(format!(
            "CumSum axis {axis} is outside rank {rank}"
        )));
    }
    Ok(axis as usize)
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;
    use crate::DynTensor;

    #[test]
    fn computes_reverse_exclusive_sum() {
        let device = Device::default();
        let tensor =
            DynTensor::from_data(TensorData::new(vec![1.0_f32, 2.0, 3.0], [3]), 1, &device)
                .unwrap();

        let output = tensor.cumsum(0, true, true).into_data();

        assert_eq!(output.iter::<f32>().collect::<Vec<_>>(), [5.0, 3.0, 0.0]);
    }
}
