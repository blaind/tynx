use std::time::Instant;

use burn::{prelude::*, tensor::TensorData};
use tynx_bench_protocol::{BenchResult, load_case, measure, print_report, require_release};

#[allow(dead_code)]
mod sign {
    include!(concat!(env!("OUT_DIR"), "/model/sign.rs"));
}
#[allow(dead_code)]
mod matmul64 {
    include!(concat!(env!("OUT_DIR"), "/model/matmul64.rs"));
}

fn main() -> BenchResult<()> {
    require_release()?;
    let case = load_case()?;
    let (backend, device, device_name) = device();

    match case.id.as_str() {
        "sign-11" => run_sign(case, backend, device, device_name),
        "matmul-64x64" => run_matmul(case, backend, device, device_name),
        id => Err(format!("burn AOT runner does not embed case '{id}'").into()),
    }
}

fn run_sign(
    case: tynx_bench_protocol::Case,
    backend: &str,
    device: Device,
    device_name: Option<String>,
) -> BenchResult<()> {
    let input = &case.inputs[0];
    if input.shape.len() != 1 {
        return Err("the sign AOT model expects a rank-1 input".into());
    }

    let started = Instant::now();
    let model = sign::Model::new(&device);
    let load_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let report = measure(
        "burn-onnx-aot",
        backend,
        device_name,
        load_ms,
        &case,
        || {
            let input = Tensor::<1>::from_data(
                TensorData::new(input.values.clone(), input.shape.clone()),
                &device,
            );
            let output = model.forward(input).into_data();
            Ok(output.convert::<f32>().to_vec::<f32>()?)
        },
    )?;
    print_report(&report)
}

fn run_matmul(
    case: tynx_bench_protocol::Case,
    backend: &str,
    device: Device,
    device_name: Option<String>,
) -> BenchResult<()> {
    let [lhs, rhs] = case.inputs.as_slice() else {
        return Err("the MatMul AOT model expects two inputs".into());
    };
    if lhs.shape.len() != 2 || rhs.shape.len() != 2 {
        return Err("the MatMul AOT model expects rank-2 inputs".into());
    }

    let started = Instant::now();
    let model = matmul64::Model::new(&device);
    let load_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let report = measure(
        "burn-onnx-aot",
        backend,
        device_name,
        load_ms,
        &case,
        || {
            let lhs = Tensor::<2>::from_data(
                TensorData::new(lhs.values.clone(), lhs.shape.clone()),
                &device,
            );
            let rhs = Tensor::<2>::from_data(
                TensorData::new(rhs.values.clone(), rhs.shape.clone()),
                &device,
            );
            let output = model.forward(lhs, rhs).into_data();
            Ok(output.convert::<f32>().to_vec::<f32>()?)
        },
    )?;
    print_report(&report)
}

#[cfg(feature = "wgpu")]
fn device() -> (&'static str, Device, Option<String>) {
    use tynx_bench_protocol::wgpu_device_name;

    (
        "wgpu",
        Device::webgpu(DeviceKind::DefaultDevice),
        wgpu_device_name(),
    )
}

#[cfg(not(feature = "wgpu"))]
fn device() -> (&'static str, Device, Option<String>) {
    ("flex", Device::flex(), None)
}
