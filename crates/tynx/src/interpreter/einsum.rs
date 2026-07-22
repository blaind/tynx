//! Bounded ONNX Einsum execution through reductions and batched matrix multiplication.

use burn::tensor::Device;
use onnx_ir::node::einsum::{EinsumNode, ParsedEinsum};

use super::{Env, resolve, shape};
use crate::{DynTensor, Result, TynxError, Value};

pub(super) fn einsum(node: &EinsumNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    if node.inputs.len() != 2 {
        return Err(TynxError::UnsupportedOp(format!(
            "Einsum with {} operands",
            node.inputs.len()
        )));
    }
    let parsed = ParsedEinsum::parse(&node.config.equation).map_err(|error| {
        TynxError::UnsupportedOp(format!("Einsum '{}': {error}", node.config.equation))
    })?;
    let left = resolve::at(env, &node.name, &node.inputs, 0, device)?.into_tensor()?;
    let right = resolve::at(env, &node.name, &node.inputs, 1, device)?.into_tensor()?;
    let output = execute_two(left, right, &parsed)?;
    if parsed.output.is_empty() {
        return Ok(vec![shape::reshape_value(
            Value::Tensor(output),
            Vec::new(),
            device,
        )?]);
    }
    Ok(vec![Value::Tensor(output)])
}

fn label_permutation(current: &[char], desired: &[char]) -> Result<Vec<usize>> {
    desired
        .iter()
        .map(|desired| {
            current
                .iter()
                .position(|current| current == desired)
                .ok_or_else(|| TynxError::Shape(format!("Einsum axis '{desired}' was not found")))
        })
        .collect()
}

fn sum_out(
    mut tensor: DynTensor,
    labels: &[char],
    reduced: &[char],
) -> Result<(DynTensor, Vec<char>)> {
    let mut labels = labels.to_vec();
    for reduced_label in reduced {
        if let Some(axis) = labels.iter().position(|label| label == reduced_label) {
            tensor = tensor.sum_dims(&[axis]);
            let mut dims = tensor.dims();
            dims.remove(axis);
            if dims.is_empty() {
                dims.push(1);
            }
            tensor = tensor.reshape(dims)?;
            labels.remove(axis);
        }
    }
    Ok((tensor, labels))
}

fn checked_product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1_usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| TynxError::Shape(format!("Einsum shape product overflow: {dims:?}")))
    })
}

fn execute_two(left: DynTensor, right: DynTensor, parsed: &ParsedEinsum) -> Result<DynTensor> {
    let batch = parsed.batch_axes();
    let contraction = parsed.contraction_axes();
    let free_left = parsed.free_lhs_axes();
    let free_right = parsed.free_rhs_axes();
    let (left, left_labels) = sum_out(left, &parsed.lhs, &parsed.reduced_lhs_axes())?;
    let (right, right_labels) = sum_out(right, &parsed.rhs, &parsed.reduced_rhs_axes())?;

    let left_order = batch
        .iter()
        .chain(&free_left)
        .chain(&contraction)
        .copied()
        .collect::<Vec<_>>();
    let right_order = batch
        .iter()
        .chain(&contraction)
        .chain(&free_right)
        .copied()
        .collect::<Vec<_>>();
    let left = left.permute(label_permutation(&left_labels, &left_order)?)?;
    let right = right.permute(label_permutation(&right_labels, &right_order)?)?;

    let batch_count = batch.len();
    let free_left_count = free_left.len();
    let contraction_count = contraction.len();
    let free_right_count = free_right.len();
    let left_dims = left.dims();
    let right_dims = right.dims();
    let batch_dims = left_dims[..batch_count].to_vec();
    let free_left_dims = left_dims[batch_count..batch_count + free_left_count].to_vec();
    let free_right_dims = right_dims
        [batch_count + contraction_count..batch_count + contraction_count + free_right_count]
        .to_vec();
    let batch_size = checked_product(&batch_dims)?;
    let rows = checked_product(&free_left_dims)?;
    let shared = checked_product(&left_dims[batch_count + free_left_count..])?;
    let columns = checked_product(&free_right_dims)?;

    let left = left.reshape(vec![batch_size, rows, shared])?;
    let right = right.reshape(vec![batch_size, shared, columns])?;
    let output = left.matmul(right)?;
    let mut output_dims = batch_dims;
    output_dims.extend(free_left_dims);
    output_dims.extend(free_right_dims);
    if output_dims.is_empty() {
        return output.reshape(vec![1]);
    }
    let output_labels = batch
        .iter()
        .chain(&free_left)
        .chain(&free_right)
        .copied()
        .collect::<Vec<_>>();
    output
        .reshape(output_dims)?
        .permute(label_permutation(&output_labels, &parsed.output)?)
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::einsum::{EinsumConfig, EinsumNodeBuilder},
    };

    use super::*;
    use crate::Scalar;

    #[test]
    fn contracts_an_inner_product_to_a_scalar() {
        let node = EinsumNodeBuilder::new("einsum")
            .input_tensor("left", 1, DType::F32)
            .input_tensor("right", 1, DType::F32)
            .output_scalar_tensor("output", DType::F32)
            .config(EinsumConfig {
                equation: "i,i->".to_string(),
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "left".into(),
            Value::from_tensor_data(TensorData::new(vec![1.0_f32, 2.0, 3.0], [3]), 1, &device)
                .unwrap(),
        );
        env.insert(
            "right".into(),
            Value::from_tensor_data(TensorData::new(vec![4.0_f32, 5.0, 6.0], [3]), 1, &device)
                .unwrap(),
        );

        let output = einsum(&node, &env, &device).unwrap();

        assert!(matches!(
            output.as_slice(),
            [Value::Scalar(Scalar::F64(value))] if (*value - 32.0).abs() < 1e-6
        ));
    }
}
