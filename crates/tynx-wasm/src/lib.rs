//! WebAssembly bindings for Tynx.

#[cfg(all(feature = "webgpu", target_family = "wasm"))]
use burn::tensor::DeviceKind;
use burn::tensor::{Device, TensorData};
use js_sys::Error;
use tynx_core::{DynTensor, Env, Session, Value};
use wasm_bindgen::prelude::*;

/// Install a panic hook that reports Rust panics in the browser console.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// A parsed ONNX model running on a browser-compatible backend.
#[wasm_bindgen(js_name = Session)]
pub struct WasmSession {
    inner: Session,
    device: Device,
}

#[wasm_bindgen(js_class = Session)]
impl WasmSession {
    /// Parse an ONNX model and use the CPU WASM backend.
    #[wasm_bindgen(constructor)]
    pub fn new(model: &[u8]) -> Result<WasmSession, JsValue> {
        Ok(Self {
            inner: Session::from_bytes(model).map_err(js_error)?,
            device: Device::flex(),
        })
    }

    /// Parse an ONNX model and use the browser WebGPU backend.
    #[cfg(all(feature = "webgpu", target_family = "wasm"))]
    #[wasm_bindgen(js_name = withWebGpu)]
    pub async fn with_webgpu(model: &[u8]) -> Result<WasmSession, JsValue> {
        Ok(Self {
            inner: Session::from_bytes(model).map_err(js_error)?,
            device: Device::wgpu_async(DeviceKind::DefaultDevice).await,
        })
    }

    /// Names of the model's declared inputs.
    #[wasm_bindgen(getter)]
    pub fn inputs(&self) -> Vec<String> {
        self.inner
            .inputs()
            .iter()
            .map(|input| input.name.clone())
            .collect()
    }

    /// Names of the model's declared outputs.
    #[wasm_bindgen(getter)]
    pub fn outputs(&self) -> Vec<String> {
        self.inner
            .outputs()
            .iter()
            .map(|output| output.name.clone())
            .collect()
    }

    /// Run a single-input, single-output floating-point model.
    pub async fn run(&self, input: Vec<f32>, shape: Vec<u32>) -> Result<Vec<f32>, JsValue> {
        let shape = checked_shape(&shape, input.len())?;
        let input_name = only_name(self.inner.inputs(), "input")?;
        let output_name = only_name(self.inner.outputs(), "output")?;
        let rank = shape.len();

        let mut env = Env::new();
        env.insert(
            input_name,
            Value::from_tensor_data(TensorData::new(input, shape), rank, &self.device)
                .map_err(js_error)?,
        );

        let mut outputs = self.inner.run(&self.device, env).map_err(js_error)?;
        let output = outputs
            .remove(&output_name)
            .ok_or_else(|| js_error(format!("model did not produce output '{output_name}'")))?;

        let Value::Tensor(output) = output else {
            return Err(js_error("model output is not a floating-point tensor"));
        };
        let data = tensor_data(output).await?;
        data.convert::<f32>().to_vec::<f32>().map_err(js_error)
    }
}

fn checked_shape(shape: &[u32], elements: usize) -> Result<Vec<usize>, JsValue> {
    let shape: Vec<usize> = shape.iter().map(|&dimension| dimension as usize).collect();
    let expected = shape
        .iter()
        .try_fold(1_usize, |size, &dimension| size.checked_mul(dimension))
        .ok_or_else(|| js_error("input shape overflows usize"))?;

    if expected != elements {
        return Err(js_error(format!(
            "input shape contains {expected} elements, but received {elements}"
        )));
    }
    Ok(shape)
}

fn only_name(
    arguments: &[tynx_core::onnx_ir::ir::Argument],
    kind: &str,
) -> Result<String, JsValue> {
    match arguments {
        [argument] => Ok(argument.name.clone()),
        _ => Err(js_error(format!(
            "the WASM scaffold requires exactly one model {kind}"
        ))),
    }
}

async fn tensor_data(tensor: DynTensor) -> Result<TensorData, JsValue> {
    let result = match tensor {
        DynTensor::R1(tensor) => tensor.into_data_async().await,
        DynTensor::R2(tensor) => tensor.into_data_async().await,
        DynTensor::R3(tensor) => tensor.into_data_async().await,
        DynTensor::R4(tensor) => tensor.into_data_async().await,
        DynTensor::R5(tensor) => tensor.into_data_async().await,
        DynTensor::R6(tensor) => tensor.into_data_async().await,
    };
    result.map_err(js_error)
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    Error::new(&error.to_string()).into()
}
