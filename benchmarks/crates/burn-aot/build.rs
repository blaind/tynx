use std::{env, fs, path::PathBuf};

use burn_onnx::ModelGen;

fn main() {
    generate_hex("sign", include_str!("../../models/sign.onnx.hex"));
    generate_hex("matmul64", include_str!("../../models/matmul64.onnx.hex"));
    generate_hex(
        "matmul_dynamic",
        include_str!("../../models/matmul_dynamic.onnx.hex"),
    );
    generate_hex(
        "matmul_add_relu_dynamic",
        include_str!("../../models/matmul_add_relu_dynamic.onnx.hex"),
    );
    generate("tiny_cnn", include_bytes!("../../models/tiny_cnn.onnx"));

    println!("cargo:rerun-if-changed=../../models/sign.onnx.hex");
    println!("cargo:rerun-if-changed=../../models/matmul64.onnx.hex");
    println!("cargo:rerun-if-changed=../../models/matmul_dynamic.onnx.hex");
    println!("cargo:rerun-if-changed=../../models/matmul_add_relu_dynamic.onnx.hex");
    println!("cargo:rerun-if-changed=../../models/tiny_cnn.onnx");
    println!("cargo:rerun-if-changed=build.rs");
}

fn generate_hex(name: &str, model_hex: &str) {
    generate(name, &decode_hex(model_hex.trim()));
}

fn generate(name: &str, model: &[u8]) {
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
