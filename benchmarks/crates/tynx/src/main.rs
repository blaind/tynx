use std::time::Instant;

use burn::tensor::{Device, TensorData};
use tynx::{Env, Session, Value};
use tynx_bench_protocol::{
    BenchResult, Case, Report, Threading, load_cases, measure_prepared, model_bytes, print_reports,
    require_release,
};

fn main() -> BenchResult<()> {
    require_release()?;
    let cases = load_cases()?;
    let threading = cpu_threading()?;
    let (backend, device, device_name) = device();
    let reports = cases
        .iter()
        .map(|case| {
            run_case(
                case,
                backend,
                &device,
                device_name.clone(),
                threading.as_ref(),
            )
        })
        .collect::<BenchResult<Vec<_>>>()?;
    print_reports(&reports)
}

fn run_case(
    case: &Case,
    backend: &str,
    device: &Device,
    device_name: Option<String>,
    threading: Option<&Threading>,
) -> BenchResult<Report> {
    let bytes = model_bytes(case)?;

    let started = Instant::now();
    let session = Session::from_bytes(&bytes)?;
    let load_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let started = Instant::now();
    let session = session.prepare(device)?;
    let prepare_ms = started.elapsed().as_secs_f64() * 1_000.0;
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

    let report = measure_prepared(
        "tynx",
        backend,
        device_name,
        load_ms,
        Some(prepare_ms),
        case,
        || {
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
                        device,
                    )?,
                );
            }
            let mut outputs = session.run(inputs)?;
            let output = outputs
                .remove(&output_name)
                .ok_or_else(|| format!("Tynx did not produce output '{output_name}'"))?
                .into_tensor()?
                .into_data();
            Ok(output.convert::<f32>().to_vec::<f32>()?)
        },
    )?;
    Ok(match threading {
        Some(threading) => report.with_threading(threading.clone()),
        None => report,
    })
}

#[cfg(feature = "wgpu")]
fn cpu_threading() -> BenchResult<Option<Threading>> {
    Ok(None)
}

#[cfg(all(not(feature = "wgpu"), feature = "multithread"))]
fn cpu_threading() -> BenchResult<Option<Threading>> {
    let request = tynx_bench_protocol::thread_request()?;
    if let tynx_bench_protocol::ThreadRequest::Fixed(threads) = request {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }
    let actual = rayon::current_num_threads();
    Ok(Some(request.report("rayon", Some(actual))))
}

#[cfg(all(not(feature = "wgpu"), not(feature = "multithread")))]
fn cpu_threading() -> BenchResult<Option<Threading>> {
    let request = tynx_bench_protocol::thread_request()?;
    if request.fixed().is_some_and(|threads| threads != 1) {
        return Err("multi-thread CPU benchmarks require the 'multithread' feature".into());
    }
    Ok(Some(request.report("serial", Some(1))))
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
