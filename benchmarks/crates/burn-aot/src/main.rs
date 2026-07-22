use std::time::Instant;

use burn::{prelude::*, tensor::TensorData};
use tynx_bench_protocol::{BenchResult, load_case, measure, print_report, require_release};

#[allow(dead_code)]
mod sign {
    include!(concat!(env!("OUT_DIR"), "/model/sign.rs"));
}

fn main() -> BenchResult<()> {
    require_release()?;
    let case = load_case()?;
    if case.input_shape.len() != 1 {
        return Err("the scaffolded AOT model expects a rank-1 input".into());
    }
    let (backend, device) = device();

    let started = Instant::now();
    let model = sign::Model::new(&device);
    let load_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let report = measure("burn-onnx-aot", backend, load_ms, &case, || {
        let input = Tensor::<1>::from_data(
            TensorData::new(case.input.clone(), case.input_shape.clone()),
            &device,
        );
        let output = model.forward(input).into_data();
        Ok(output.convert::<f32>().to_vec::<f32>()?)
    })?;
    print_report(&report)
}

#[cfg(feature = "wgpu")]
fn device() -> (&'static str, Device) {
    ("wgpu", Device::webgpu(DeviceKind::DefaultDevice))
}

#[cfg(not(feature = "wgpu"))]
fn device() -> (&'static str, Device) {
    ("flex", Device::flex())
}
