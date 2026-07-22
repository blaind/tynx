//! ArgMax and ArgMin execution.

use burn::tensor::Device;
use onnx_ir::{
    ir::Argument,
    node::{argmax::ArgMaxNode, argmin::ArgMinNode},
};

use super::{Env, resolve, shape};
use crate::{Result, TynxError, Value};

pub(super) fn argmax(node: &ArgMaxNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    arg_extreme(
        &node.name,
        &node.inputs,
        node.config.axis,
        node.config.keepdims,
        node.config.select_last_index,
        true,
        env,
        device,
    )
}

pub(super) fn argmin(node: &ArgMinNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    arg_extreme(
        &node.name,
        &node.inputs,
        node.config.axis,
        node.config.keepdims,
        node.config.select_last_index,
        false,
        env,
        device,
    )
}

#[allow(clippy::too_many_arguments)]
fn arg_extreme(
    name: &str,
    inputs: &[Argument],
    axis: usize,
    keepdims: bool,
    select_last: bool,
    maximum: bool,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::first(env, name, inputs, device)?;
    let mut output_dims = shape::value_dims(&input);
    if axis >= output_dims.len() {
        return Err(TynxError::Shape(format!(
            "axis {axis} is out of range for rank {}",
            output_dims.len()
        )));
    }
    if output_dims[axis] == 0 {
        return Err(TynxError::Shape(format!(
            "cannot select an index from empty axis {axis}"
        )));
    }

    let output = match input {
        Value::Tensor(tensor) => Value::Int(tensor.arg_extreme(axis, maximum, select_last)),
        Value::Int(tensor) => Value::Int(tensor.arg_extreme(axis, maximum, select_last)),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "ArgMax/ArgMin expects a numeric tensor, got {other:?}"
            )));
        }
    };

    output_dims[axis] = 1;
    if !keepdims {
        output_dims.remove(axis);
    }
    Ok(vec![shape::reshape_value(output, output_dims, device)?])
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::{
            argmax::{ArgMaxConfig, ArgMaxNodeBuilder},
            argmin::{ArgMinConfig, ArgMinNodeBuilder},
        },
    };

    use super::*;
    use crate::{DynTensor, Scalar};

    #[test]
    fn argmax_selects_last_tied_index() {
        let node = ArgMaxNodeBuilder::new("argmax")
            .input_tensor("x", 2, DType::F32)
            .output_tensor("y", 2, DType::I64)
            .config(ArgMaxConfig::new(1, true, true))
            .build();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::Tensor(
                DynTensor::from_data(
                    TensorData::new(vec![4.0_f32, 4.0, 1.0, 2.0, 3.0, 3.0], [2, 3]),
                    2,
                    &Device::default(),
                )
                .unwrap(),
            ),
        );

        let output = argmax(&node, &env, &Device::default()).unwrap();
        let Value::Int(output) = output.into_iter().next().unwrap() else {
            panic!("expected integer tensor output");
        };
        assert_eq!(output.into_data().iter::<i64>().collect::<Vec<_>>(), [1, 2]);
    }

    #[test]
    fn argmin_without_keepdims_can_return_a_scalar() {
        let node = ArgMinNodeBuilder::new("argmin")
            .input_tensor("x", 1, DType::F32)
            .output_scalar_tensor("y", DType::I64)
            .config(ArgMinConfig::new(0, false, false))
            .build();
        let mut env = Env::new();
        env.insert(
            "x".into(),
            Value::Tensor(
                DynTensor::from_data(
                    TensorData::new(vec![2.0_f32, -1.0, 3.0], [3]),
                    1,
                    &Device::default(),
                )
                .unwrap(),
            ),
        );

        let output = argmin(&node, &env, &Device::default()).unwrap();
        assert!(matches!(output.as_slice(), [Value::Scalar(Scalar::I64(1))]));
    }
}
