//! ONNX Hardmax execution.

use burn::tensor::{Device, IndexingUpdateOp};
use onnx_ir::node::hardmax::HardmaxNode;

use super::{Env, resolve};
use crate::{Result, TynxError, Value};

pub(super) fn hardmax(node: &HardmaxNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?.into_tensor()?;
    let dims = input.dims();
    if node.config.axis >= dims.len() {
        return Err(TynxError::Shape(format!(
            "Hardmax axis {} is outside rank {}",
            node.config.axis,
            dims.len()
        )));
    }
    if dims[node.config.axis] == 0 {
        return Err(TynxError::Shape(format!(
            "Hardmax cannot select from empty axis {}",
            node.config.axis
        )));
    }

    let dtype = input.dtype();
    let indices = input.clone().arg_extreme(node.config.axis, true, false);
    let ones = indices
        .clone()
        .to_float(dtype)
        .mul_scalar(0.0)
        .add_scalar(1.0);
    let output =
        input
            .mul_scalar(0.0)
            .scatter(node.config.axis, indices, ones, IndexingUpdateOp::Add)?;
    Ok(vec![Value::Tensor(output)])
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::hardmax::{HardmaxConfig, HardmaxNodeBuilder},
    };

    use super::*;

    #[test]
    fn selects_the_first_tied_maximum() {
        let node = HardmaxNodeBuilder::new("hardmax")
            .input_tensor("x", 2, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .config(HardmaxConfig { axis: 1 })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::from_tensor_data(
                TensorData::new(vec![1.0_f32, 3.0, 3.0, 4.0, 2.0, 0.0], [2, 3]),
                2,
                &device,
            )
            .unwrap(),
        );

        let output = hardmax(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    }
}
