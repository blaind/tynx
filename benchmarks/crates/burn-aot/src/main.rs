use std::time::Instant;

use burn::{
    prelude::*,
    tensor::{Bytes, TensorData},
};
use tynx_bench_protocol::{
    BenchResult, Case, Report, load_cases, measure, print_reports, require_release,
};

#[allow(dead_code)]
mod sign {
    include!(concat!(env!("OUT_DIR"), "/model/sign.rs"));
}
#[allow(dead_code)]
mod matmul64 {
    include!(concat!(env!("OUT_DIR"), "/model/matmul64.rs"));
}
#[allow(dead_code)]
mod matmul_dynamic {
    include!(concat!(env!("OUT_DIR"), "/model/matmul_dynamic.rs"));
}
#[allow(dead_code)]
mod matmul_add_relu_dynamic {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/matmul_add_relu_dynamic.rs"
    ));
}
#[allow(dead_code)]
mod tiny_cnn {
    include!(concat!(env!("OUT_DIR"), "/model/tiny_cnn.rs"));
}
#[cfg(feature = "mobilenet")]
#[allow(dead_code)]
mod mobilenetv2 {
    include!(concat!(env!("OUT_DIR"), "/model/mobilenetv2.rs"));
}

fn main() -> BenchResult<()> {
    require_release()?;
    let cases = load_cases()?;
    let (backend, device, device_name) = device();
    let reports = cases
        .iter()
        .map(|case| run_case(case, backend, &device, device_name.clone()))
        .collect::<BenchResult<Vec<_>>>()?;
    print_reports(&reports)
}

fn run_case(
    case: &Case,
    backend: &str,
    device: &Device,
    device_name: Option<String>,
) -> BenchResult<Report> {
    match case.model.as_str() {
        "models/sign.onnx.hex" => run_sign(case, backend, device, device_name),
        "models/matmul64.onnx.hex" => run_matmul64(case, backend, device, device_name),
        "models/matmul_dynamic.onnx.hex" => run_matmul_dynamic(case, backend, device, device_name),
        "models/matmul_add_relu_dynamic.onnx.hex" => {
            run_matmul_add_relu(case, backend, device, device_name)
        }
        "models/tiny_cnn.onnx" => run_tiny_cnn(case, backend, device, device_name),
        #[cfg(feature = "mobilenet")]
        "external/mobilenetv2_100_opset16.onnx" => {
            run_mobilenet(case, backend, device, device_name)
        }
        model => Err(format!("burn AOT runner does not embed model '{model}'").into()),
    }
}

fn run_sign(
    case: &Case,
    backend: &str,
    device: &Device,
    device_name: Option<String>,
) -> BenchResult<Report> {
    let input = &case.inputs[0];
    if input.shape.len() != 1 {
        return Err("the sign AOT model expects a rank-1 input".into());
    }

    let started = Instant::now();
    let model = sign::Model::new(device);
    let load_ms = started.elapsed().as_secs_f64() * 1_000.0;

    measure("burn-onnx-aot", backend, device_name, load_ms, case, || {
        let input = Tensor::<1>::from_data(
            TensorData::new(input.values.clone(), input.shape.clone()),
            device,
        );
        let output = model.forward(input).into_data();
        Ok(output.convert::<f32>().to_vec::<f32>()?)
    })
}

fn run_matmul64(
    case: &Case,
    backend: &str,
    device: &Device,
    device_name: Option<String>,
) -> BenchResult<Report> {
    let started = Instant::now();
    let model = matmul64::Model::new(device);
    let load_ms = started.elapsed().as_secs_f64() * 1_000.0;
    run_matmul(case, backend, device, device_name, load_ms, |lhs, rhs| {
        model.forward(lhs, rhs)
    })
}

fn run_matmul_dynamic(
    case: &Case,
    backend: &str,
    device: &Device,
    device_name: Option<String>,
) -> BenchResult<Report> {
    let started = Instant::now();
    let model = matmul_dynamic::Model::new(device);
    let load_ms = started.elapsed().as_secs_f64() * 1_000.0;
    run_matmul(case, backend, device, device_name, load_ms, |lhs, rhs| {
        model.forward(lhs, rhs)
    })
}

fn run_matmul<F>(
    case: &Case,
    backend: &str,
    device: &Device,
    device_name: Option<String>,
    load_ms: f64,
    mut forward: F,
) -> BenchResult<Report>
where
    F: FnMut(Tensor<2>, Tensor<2>) -> Tensor<2>,
{
    let [lhs, rhs] = case.inputs.as_slice() else {
        return Err("the MatMul AOT model expects two inputs".into());
    };
    if lhs.shape.len() != 2 || rhs.shape.len() != 2 {
        return Err("the MatMul AOT model expects rank-2 inputs".into());
    }

    measure("burn-onnx-aot", backend, device_name, load_ms, case, || {
        let lhs = Tensor::<2>::from_data(
            TensorData::new(lhs.values.clone(), lhs.shape.clone()),
            device,
        );
        let rhs = Tensor::<2>::from_data(
            TensorData::new(rhs.values.clone(), rhs.shape.clone()),
            device,
        );
        let output = forward(lhs, rhs).into_data();
        Ok(output.convert::<f32>().to_vec::<f32>()?)
    })
}

fn run_matmul_add_relu(
    case: &Case,
    backend: &str,
    device: &Device,
    device_name: Option<String>,
) -> BenchResult<Report> {
    let [lhs, rhs, bias] = case.inputs.as_slice() else {
        return Err("the MatMul-Add-ReLU AOT model expects three inputs".into());
    };
    if lhs.shape.len() != 2 || rhs.shape.len() != 2 || bias.shape.len() != 1 {
        return Err(
            "the MatMul-Add-ReLU AOT model expects rank-2, rank-2, and rank-1 inputs".into(),
        );
    }

    let started = Instant::now();
    let model = matmul_add_relu_dynamic::Model::new(device);
    let load_ms = started.elapsed().as_secs_f64() * 1_000.0;

    measure("burn-onnx-aot", backend, device_name, load_ms, case, || {
        let lhs = Tensor::<2>::from_data(
            TensorData::new(lhs.values.clone(), lhs.shape.clone()),
            device,
        );
        let rhs = Tensor::<2>::from_data(
            TensorData::new(rhs.values.clone(), rhs.shape.clone()),
            device,
        );
        let bias = Tensor::<1>::from_data(
            TensorData::new(bias.values.clone(), bias.shape.clone()),
            device,
        );
        let output = model.forward(lhs, rhs, bias).into_data();
        Ok(output.convert::<f32>().to_vec::<f32>()?)
    })
}

fn run_tiny_cnn(
    case: &Case,
    backend: &str,
    device: &Device,
    device_name: Option<String>,
) -> BenchResult<Report> {
    let [input] = case.inputs.as_slice() else {
        return Err("the tiny CNN AOT model expects one input".into());
    };
    if input.shape.len() != 4 {
        return Err("the tiny CNN AOT model expects a rank-4 input".into());
    }

    let started = Instant::now();
    let model = tiny_cnn::Model::from_bytes(
        Bytes::from_bytes_vec(
            include_bytes!(concat!(env!("OUT_DIR"), "/model/tiny_cnn.bpk")).to_vec(),
        ),
        device,
    );
    let load_ms = started.elapsed().as_secs_f64() * 1_000.0;

    measure("burn-onnx-aot", backend, device_name, load_ms, case, || {
        let input = Tensor::<4>::from_data(
            TensorData::new(input.values.clone(), input.shape.clone()),
            device,
        );
        let output = model.forward(input).into_data();
        Ok(output.convert::<f32>().to_vec::<f32>()?)
    })
}

#[cfg(feature = "mobilenet")]
fn run_mobilenet(
    case: &Case,
    backend: &str,
    device: &Device,
    device_name: Option<String>,
) -> BenchResult<Report> {
    let [input] = case.inputs.as_slice() else {
        return Err("the MobileNetV2 AOT model expects one input".into());
    };
    if input.shape.len() != 4 {
        return Err("the MobileNetV2 AOT model expects a rank-4 input".into());
    }

    let started = Instant::now();
    let model = mobilenetv2::Model::from_bytes(
        Bytes::from_bytes_vec(
            include_bytes!(concat!(env!("OUT_DIR"), "/model/mobilenetv2.bpk")).to_vec(),
        ),
        device,
    );
    let load_ms = started.elapsed().as_secs_f64() * 1_000.0;

    measure("burn-onnx-aot", backend, device_name, load_ms, case, || {
        let input = Tensor::<4>::from_data(
            TensorData::new(input.values.clone(), input.shape.clone()),
            device,
        );
        let output = model.forward(input).into_data();
        Ok(output.convert::<f32>().to_vec::<f32>()?)
    })
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
