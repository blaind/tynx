use std::time::Instant;

use burn::tensor::{Device, TensorData};
use tynx::{Env, Session, Value};
use tynx_bench_protocol::{
    BenchResult, load_case, measure, model_bytes, print_report, require_release,
};

fn main() -> BenchResult<()> {
    require_release()?;
    let case = load_case()?;
    let bytes = model_bytes(&case)?;
    let (backend, device) = device();

    let started = Instant::now();
    let session = Session::from_bytes(&bytes)?;
    let load_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let input_name = session.inputs()[0].name.clone();
    let output_name = session.outputs()[0].name.clone();

    let report = measure("tynx", backend, load_ms, &case, || {
        let mut inputs = Env::new();
        inputs.insert(
            input_name.clone(),
            Value::from_tensor_data(
                TensorData::new(case.input.clone(), case.input_shape.clone()),
                case.input_shape.len(),
                &device,
            )?,
        );
        let mut outputs = session.run(&device, inputs)?;
        let output = outputs
            .remove(&output_name)
            .ok_or_else(|| format!("Tynx did not produce output '{output_name}'"))?
            .into_tensor()?
            .into_data();
        Ok(output.convert::<f32>().to_vec::<f32>()?)
    })?;
    print_report(&report)
}

#[cfg(feature = "wgpu")]
fn device() -> (&'static str, Device) {
    use burn::tensor::DeviceKind;

    ("wgpu", Device::wgpu(DeviceKind::DefaultDevice))
}

#[cfg(not(feature = "wgpu"))]
fn device() -> (&'static str, Device) {
    ("flex", Device::flex())
}
