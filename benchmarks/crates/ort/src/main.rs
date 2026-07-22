#[cfg(all(feature = "cpu", feature = "cuda"))]
compile_error!("select either the cpu or cuda feature, not both");
#[cfg(not(any(feature = "cpu", feature = "cuda")))]
compile_error!("select either the cpu or cuda feature");

use std::time::Instant;

use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::{Shape, TensorRef},
};
use tynx_bench_protocol::{
    BenchResult, Case, Report, ThreadRequest, load_cases, measure, model_bytes, print_reports,
    require_release,
};

fn main() -> BenchResult<()> {
    require_release()?;
    let cases = load_cases()?;
    let threading = cpu_threading()?;
    let reports = cases
        .iter()
        .map(|case| run_case(case, threading))
        .collect::<BenchResult<Vec<_>>>()?;
    print_reports(&reports)
}

fn run_case(case: &Case, threading: Option<ThreadRequest>) -> BenchResult<Report> {
    let bytes = model_bytes(case)?;

    let started = Instant::now();
    let mut builder = Session::builder()?.with_optimization_level(GraphOptimizationLevel::All)?;
    if let Some(threads) = threading.and_then(ThreadRequest::fixed) {
        builder = builder.with_intra_threads(threads)?;
    }
    #[cfg(feature = "cuda")]
    let mut builder =
        builder.with_execution_providers([ep::CUDA::default().build().error_on_failure()])?;
    #[cfg(feature = "cpu")]
    let mut builder =
        builder.with_execution_providers([ep::CPU::default().build().error_on_failure()])?;
    let mut session = builder.commit_from_memory(&bytes)?;
    let load_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let report = measure("onnxruntime", backend(), None, load_ms, case, || {
        let inputs = case
            .inputs
            .iter()
            .map(|input| {
                let shape = Shape::new(input.shape.iter().map(|&dimension| dimension as i64));
                Ok((
                    input.name.as_str(),
                    TensorRef::from_array_view((shape, input.values.as_slice()))?,
                ))
            })
            .collect::<ort::Result<Vec<_>>>()?;
        let outputs = session.run(inputs)?;
        Ok(outputs[0].try_extract_tensor::<f32>()?.1.to_vec())
    })?;
    Ok(match threading {
        Some(request) => report.with_threading(request.report("onnxruntime", request.fixed())),
        None => report,
    })
}

#[cfg(feature = "cpu")]
fn cpu_threading() -> BenchResult<Option<ThreadRequest>> {
    Ok(Some(tynx_bench_protocol::thread_request()?))
}

#[cfg(feature = "cuda")]
fn cpu_threading() -> BenchResult<Option<ThreadRequest>> {
    Ok(None)
}

#[cfg(feature = "cuda")]
fn backend() -> &'static str {
    "cuda"
}

#[cfg(feature = "cpu")]
fn backend() -> &'static str {
    "cpu"
}
