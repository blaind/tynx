use std::{env, fs, path::PathBuf};

use burn_onnx::ModelGen;

fn main() {
    generate("sign", include_str!("../../models/sign.onnx.hex"));
    generate("matmul64", include_str!("../../models/matmul64.onnx.hex"));

    println!("cargo:rerun-if-changed=../../models/sign.onnx.hex");
    println!("cargo:rerun-if-changed=../../models/matmul64.onnx.hex");
    println!("cargo:rerun-if-changed=build.rs");
}

fn generate(name: &str, model_hex: &str) {
    let model = decode_hex(model_hex.trim());
    let model_path =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set")).join(format!("{name}.onnx"));
    fs::write(&model_path, model).expect("write generated ONNX fixture");

    ModelGen::new()
        .input(model_path.to_str().expect("UTF-8 model path"))
        .out_dir("model/")
        .run_from_script();
}

fn decode_hex(input: &str) -> Vec<u8> {
    assert!(input.len().is_multiple_of(2), "model hex has an odd length");
    (0..input.len())
        .step_by(2)
        .map(|offset| u8::from_str_radix(&input[offset..offset + 2], 16).expect("valid model hex"))
        .collect()
}
