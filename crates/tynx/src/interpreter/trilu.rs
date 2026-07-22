//! Upper and lower triangular tensor execution.

use burn::tensor::{Device, TensorData};
use onnx_ir::node::trilu::TriluNode;

use super::{Env, resolve};
use crate::{DynBool, DynInt, DynTensor, Result, Scalar, TynxError, Value};

pub(super) fn trilu(node: &TriluNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let diagonal = diagonal(node, env, device)?;
    let dims = match &input {
        Value::Tensor(tensor) => tensor.dims(),
        Value::Int(tensor) => tensor.dims(),
        Value::Bool(tensor) => tensor.dims(),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "Trilu expects a tensor, got {other:?}"
            )));
        }
    };
    if dims.len() < 2 {
        return Err(TynxError::Shape(format!(
            "Trilu requires rank >= 2, got {}",
            dims.len()
        )));
    }
    let mask = triangular_mask(&dims, node.config.upper, diagonal, device)?;
    let output = match input {
        Value::Tensor(input) => {
            let zeros = DynTensor::full(&dims, 0.0, device, input.dtype())?;
            Value::Tensor(DynTensor::where_select(mask, input, zeros)?)
        }
        Value::Int(input) => {
            let zeros = DynInt::full(&dims, 0, device, input.dtype())?;
            Value::Int(DynInt::where_select(mask, input, zeros)?)
        }
        Value::Bool(input) => {
            let zeros = DynBool::full(&dims, false, device)?;
            Value::Bool(DynBool::where_select(mask, input, zeros)?)
        }
        _ => unreachable!("input kind was validated"),
    };
    Ok(vec![output])
}

fn diagonal(node: &TriluNode, env: &Env, device: &Device) -> Result<i64> {
    if !node.inputs.get(1).is_some_and(|input| !input.is_optional()) {
        return Ok(node.config.diagonal);
    }
    match resolve::at(env, &node.name, &node.inputs, 1, device)? {
        Value::Scalar(Scalar::I64(value)) => Ok(value),
        Value::Scalar(Scalar::U64(value)) => i64::try_from(value)
            .map_err(|_| TynxError::Shape(format!("Trilu diagonal {value} exceeds i64"))),
        Value::Int(value) if value.dims().iter().product::<usize>() == 1 => value
            .into_data()
            .iter::<i64>()
            .next()
            .ok_or_else(|| TynxError::Shape("empty Trilu diagonal".to_string())),
        other => Err(TynxError::TypeMismatch(format!(
            "Trilu diagonal must be an integer scalar, got {other:?}"
        ))),
    }
}

fn triangular_mask(dims: &[usize], upper: bool, diagonal: i64, device: &Device) -> Result<DynBool> {
    let rows = dims[dims.len() - 2];
    let columns = dims[dims.len() - 1];
    let count = dims.iter().product();
    let values = (0..count)
        .map(|index| {
            let row = (index / columns) % rows;
            let column = index % columns;
            if upper {
                column as i64 >= row as i64 + diagonal
            } else {
                column as i64 <= row as i64 + diagonal
            }
        })
        .collect::<Vec<_>>();
    DynBool::from_data(TensorData::new(values, dims.to_vec()), dims.len(), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_offset_lower_triangle() {
        let output = triangular_mask(&[3, 3], false, -1, &Device::default())
            .unwrap()
            .into_data()
            .iter::<bool>()
            .collect::<Vec<_>>();

        assert_eq!(
            output,
            [false, false, false, true, false, false, true, true, false]
        );
    }
}
