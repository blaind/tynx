//! ONNX Pow execution.

use burn::tensor::{DType, Device};
use onnx_ir::node::pow::PowNode;

use super::{Env, resolve};
use crate::{Result, Scalar, TynxError, Value};

pub(super) fn pow(node: &PowNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let base = resolve::at(env, &node.name, &node.inputs, 0, device)?;
    let exponent = resolve::at(env, &node.name, &node.inputs, 1, device)?;

    let output = match (base, exponent) {
        (Value::Tensor(base), Value::Tensor(exponent)) => {
            Value::Tensor(base.powf_broadcast(exponent)?)
        }
        (Value::Tensor(base), Value::Int(exponent)) => {
            let dtype = base.dtype();
            Value::Tensor(base.powf_broadcast(exponent.to_float(dtype))?)
        }
        (Value::Tensor(base), Value::Scalar(exponent)) => Value::Tensor(match exponent {
            Scalar::F64(exponent) => base.powf_scalar(exponent),
            exponent => base.powi_scalar(scalar_as_i64(exponent)),
        }),
        (Value::Int(base), Value::Int(exponent)) => Value::Int(base.powi_broadcast(exponent)?),
        (Value::Int(base), Value::Tensor(exponent)) => {
            let output_dtype = base.dtype();
            let compute_dtype = exponent.dtype();
            Value::Int(
                base.to_float(compute_dtype)
                    .powf_broadcast(exponent)?
                    .to_int(output_dtype),
            )
        }
        (Value::Int(base), Value::Scalar(exponent)) => match exponent {
            Scalar::F64(exponent) => {
                let output_dtype = base.dtype();
                Value::Int(
                    base.to_float(DType::F64)
                        .powf_scalar(exponent)
                        .to_int(output_dtype),
                )
            }
            exponent => Value::Int(base.powi_scalar(scalar_as_i64(exponent))),
        },
        (Value::Scalar(base), Value::Scalar(exponent)) => {
            Value::Scalar(pow_scalars(base, exponent))
        }
        (base, exponent) => {
            return Err(TynxError::TypeMismatch(format!(
                "unsupported Pow operands: {base:?} and {exponent:?}"
            )));
        }
    };

    Ok(vec![output])
}

fn scalar_as_i64(value: Scalar) -> i64 {
    match value {
        Scalar::F64(value) => value as i64,
        Scalar::I64(value) => value,
        Scalar::U64(value) => value.min(i64::MAX as u64) as i64,
        Scalar::Bool(value) => i64::from(value),
    }
}

fn pow_scalars(base: Scalar, exponent: Scalar) -> Scalar {
    let exponent = exponent.as_f64();
    match base {
        Scalar::F64(base) => Scalar::F64(base.powf(exponent)),
        Scalar::I64(base) => Scalar::I64((base as f64).powf(exponent) as i64),
        Scalar::U64(base) => Scalar::U64((base as f64).powf(exponent) as u64),
        Scalar::Bool(base) => Scalar::Bool(base && exponent != 0.0),
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{DType, node::pow::PowNodeBuilder};

    use super::*;

    #[test]
    fn raises_float_tensor_to_broadcast_exponents() {
        let node = PowNodeBuilder::new("pow")
            .input_tensor("base", 2, DType::F32)
            .input_tensor("exponent", 1, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let device = Device::default();
        let mut env = Env::new();
        env.insert(
            "base".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![2.0_f32, 3.0, 4.0, 5.0], [2, 2]),
                2,
                &device,
            )
            .unwrap(),
        );
        env.insert(
            "exponent".to_string(),
            Value::from_tensor_data(TensorData::new(vec![2.0_f32, 3.0], [2]), 1, &device).unwrap(),
        );

        let output = pow(&node, &env, &device)
            .unwrap()
            .pop()
            .unwrap()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .collect::<Vec<_>>();

        assert_eq!(output, [4.0, 27.0, 16.0, 125.0]);
    }
}
