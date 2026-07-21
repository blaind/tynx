#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

mod common;

use tynx_wasm::WasmSession;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn runs_onnx_on_swiftshader_webgpu() {
    let session = WasmSession::with_webgpu(common::SIGN_MODEL)
        .await
        .expect("parse Sign model");
    let output = session
        .run(vec![-2.0, 0.0, 3.0], vec![3])
        .await
        .expect("run Sign model");

    assert_eq!(output, [-1.0, 0.0, 1.0]);
}
