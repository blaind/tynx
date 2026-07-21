//! Resolution of runtime and embedded ONNX inputs.

use burn::tensor::Device;
use onnx_ir::ir::Argument;

use super::Env;
use crate::{Result, TynxError, Value};

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

    input(env, argument, device)
}

pub(super) fn input(env: &Env, argument: &Argument, device: &Device) -> Result<Value> {
    if let Some(value) = env.get(&argument.name) {
        return Ok(value.clone());
    }
    if let Some(data) = argument.value() {
        return Value::from_tensor_data(data, argument.ty.rank(), device);
    }

    let name = if argument.name.is_empty() {
        "<static/absent>"
    } else {
        &argument.name
    };
    Err(TynxError::MissingValue(name.to_string()))
}

#[cfg(test)]
mod tests {
    use onnx_ir::{ArgType, DType};

    use super::*;
    use crate::Scalar;

    #[test]
    fn reports_a_missing_runtime_input() {
        let argument = Argument::new("missing", ArgType::ScalarNative(DType::I64));

        let error = input(&Env::new(), &argument, &Device::default()).unwrap_err();

        assert_eq!(error, TynxError::MissingValue("missing".to_string()));
    }

    #[test]
    fn materializes_a_static_input() {
        let argument = Argument::from_const_i64("constant", 42);

        let value = input(&Env::new(), &argument, &Device::default()).unwrap();

        assert!(matches!(value, Value::Scalar(Scalar::I64(42))));
    }
}
