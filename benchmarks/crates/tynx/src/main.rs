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
    let (backend, device, device_name) = device();

    let started = Instant::now();
    let session = Session::from_bytes(&bytes)?;
    let load_ms = started.elapsed().as_secs_f64() * 1_000.0;
    if session.inputs().len() != case.inputs.len() {
        return Err(format!(
            "model requires {} inputs, case '{}' provides {}",
            session.inputs().len(),
            case.id,
            case.inputs.len()
        )
        .into());
    }
    let output_name = session.outputs()[0].name.clone();

    let report = measure("tynx", backend, device_name.clone(), load_ms, &case, || {
        let mut inputs = Env::new();
        for (model_input, input) in session.inputs().iter().zip(&case.inputs) {
            if model_input.name != input.name {
                return Err(format!(
                    "model input '{}' does not match case input '{}'",
                    model_input.name, input.name
                )
                .into());
            }
            inputs.insert(
                input.name.clone(),
                Value::from_tensor_data(
                    TensorData::new(input.values.clone(), input.shape.clone()),
                    input.shape.len(),
                    &device,
                )?,
            );
        }
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
fn device() -> (&'static str, Device, Option<String>) {
    use burn::tensor::DeviceKind;
    use tynx_bench_protocol::wgpu_device_name;

    (
        "wgpu",
        Device::wgpu(DeviceKind::DefaultDevice),
        wgpu_device_name(),
    )
}

#[cfg(not(feature = "wgpu"))]
fn device() -> (&'static str, Device, Option<String>) {
    ("flex", Device::flex(), None)
}
