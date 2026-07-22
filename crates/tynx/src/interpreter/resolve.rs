//! Resolution of runtime and embedded ONNX inputs.

use burn::tensor::Device;
use onnx_ir::ir::{ArgType, Argument};

use super::Env;
use crate::{InitializerId, Result, TynxError, Value, initializer::env_key};

pub(super) fn first(
    env: &Env,
    node_name: &str,
    inputs: &[Argument],
    device: &Device,
) -> Result<Value> {
    at(env, node_name, inputs, 0, device)
}

pub(super) fn at(
    env: &Env,
    node_name: &str,
    inputs: &[Argument],
    index: usize,
    device: &Device,
) -> Result<Value> {
    let argument = inputs.get(index).ok_or_else(|| {
        TynxError::Shape(format!("node '{node_name}' has no input at index {index}"))
    })?;

    input_at(env, argument, index, device)
}

pub(super) fn input_at(
    env: &Env,
    argument: &Argument,
    input_index: usize,
    device: &Device,
) -> Result<Value> {
    input_impl(env, argument, input_index, device)
}

fn input_impl(
    env: &Env,
    argument: &Argument,
    input_index: usize,
    device: &Device,
) -> Result<Value> {
    if let Some(value) = env.get(&argument.name) {
        return Ok(value.clone());
    }
    let initializer = InitializerId::from_argument(argument, 0, input_index);
    if let Some(value) = initializer.and_then(|id| env.get(&env_key(&id))) {
        return Ok(value.clone());
    }
    if let Some(data) = argument.value() {
        return materialize(argument, data, device);
    }

    let name = if argument.name.is_empty() {
        "<static/absent>"
    } else {
        &argument.name
    };
    Err(TynxError::MissingValue(name.to_string()))
}

pub(crate) fn materialize(
    argument: &Argument,
    data: burn::tensor::TensorData,
    device: &Device,
) -> Result<Value> {
    if matches!(argument.ty, ArgType::Shape(_)) {
        return Ok(Value::Shape(data.iter::<i64>().collect()));
    }
    let rank = data.shape.len();
    Value::from_tensor_data(data, rank, device)
}

#[cfg(test)]
mod tests {
    use onnx_ir::{ArgType, DType};

    use super::*;
    use crate::Scalar;

    #[test]
    fn reports_a_missing_runtime_input() {
        let argument = Argument::new("missing", ArgType::ScalarNative(DType::I64));

        let error = input_at(&Env::new(), &argument, 0, &Device::default()).unwrap_err();

        assert_eq!(error, TynxError::MissingValue("missing".to_string()));
    }

    #[test]
    fn materializes_a_static_input() {
        let argument = Argument::from_const_i64("constant", 42);

        let value = input_at(&Env::new(), &argument, 0, &Device::default()).unwrap();

        assert!(matches!(value, Value::Scalar(Scalar::I64(42))));
    }
}
