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
    BenchResult, load_case, measure, model_bytes, print_report, require_release,
};

fn main() -> BenchResult<()> {
    require_release()?;
    let case = load_case()?;
    let bytes = model_bytes(&case)?;

    let started = Instant::now();
    let builder = Session::builder()?.with_optimization_level(GraphOptimizationLevel::All)?;
    #[cfg(feature = "cuda")]
    let mut builder =
        builder.with_execution_providers([ep::CUDA::default().build().error_on_failure()])?;
    #[cfg(feature = "cpu")]
    let mut builder =
        builder.with_execution_providers([ep::CPU::default().build().error_on_failure()])?;
    let mut session = builder.commit_from_memory(&bytes)?;
    let load_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let shape = Shape::new(case.input_shape.iter().map(|&dimension| dimension as i64));

    let report = measure("onnxruntime", backend(), load_ms, &case, || {
        let input = TensorRef::from_array_view((shape.clone(), case.input.as_slice()))?;
        let outputs = session.run(ort::inputs![input])?;
        Ok(outputs[0].try_extract_tensor::<f32>()?.1.to_vec())
    })?;
    print_report(&report)
}

#[cfg(feature = "cuda")]
fn backend() -> &'static str {
    "cuda"
}

#[cfg(feature = "cpu")]
fn backend() -> &'static str {
    "cpu"
}
