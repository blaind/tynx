use std::{env, fs, path::PathBuf};

use burn_onnx::ModelGen;

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let model_path = out_dir.join("training_mlp.onnx");
    fs::write(
        &model_path,
        tynx_bench_protocol::training::training_mlp_model().expect("generate training MLP fixture"),
    )
    .expect("write training MLP fixture");
    ModelGen::new()
        .input(model_path.to_str().expect("UTF-8 model path"))
        .out_dir("model/")
        .run_from_script();
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../training-cases.json");
}
