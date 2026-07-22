use std::{env, fs, path::PathBuf};

use burn_onnx::ModelGen;

fn main() {
    let model_hex = include_str!("../../models/sign.onnx.hex").trim();
    let model = decode_hex(model_hex);
    let model_path =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set")).join("sign.onnx");
    fs::write(&model_path, model).expect("write generated ONNX fixture");

    println!("cargo:rerun-if-changed=../../models/sign.onnx.hex");
    println!("cargo:rerun-if-changed=build.rs");
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
