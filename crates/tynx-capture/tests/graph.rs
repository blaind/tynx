use burn::tensor::{DType, Device, TensorData};
use std::{cell::RefCell, rc::Rc};

use tynx_capture::{BinaryOp, CapturedOptimizer, GraphBuilder, UnaryOp};
use tynx_core::{DynTensor, Result};
use tynx_train::{ParameterSlot, ParameterStore, Sgd, backward};

fn tensor(data: &[f32], shape: [usize; 2], device: &Device) -> DynTensor {
    DynTensor::from_data(
        TensorData::new(data.to_vec(), shape).convert::<f32>(),
        2,
        device,
    )
    .unwrap()
}

fn values(tensor: DynTensor) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn replays_linear_relu_graph_and_tracks_current_slots() {
    let device = Device::autodiff(Device::default());
    let input = tensor(&[1.0, -2.0, 3.0, 4.0], [2, 2], &device).require_grad();
    let weight = ParameterSlot::new(
        Some("weight".to_string()),
        tensor(&[2.0, 0.0, 0.0, 1.0], [2, 2], &device),
        true,
    )
    .unwrap();
    let bias = ParameterSlot::new(
        Some("bias".to_string()),
        tensor(&[1.0, -1.0], [1, 2], &device),
        true,
    )
    .unwrap();

    let mut builder = GraphBuilder::new();
    let x = builder.input(&input);
    let w = builder.parameter(weight.clone());
    let b = builder.parameter(bias.clone());
    let affine = builder.binary(BinaryOp::Matmul, x, w).unwrap();
    let affine = builder.binary(BinaryOp::Add, affine, b).unwrap();
    let output = builder.unary(UnaryOp::Relu, affine).unwrap();
    let graph = builder.finish(vec![output]).unwrap();

    assert_eq!(graph.node_count(), 6);
    assert_eq!(graph.parameters().len(), 2);
    let output = graph.run(&[input], true).unwrap().remove(0);
    assert_eq!(values(output.clone()), vec![3.0, 0.0, 7.0, 3.0]);

    let mut parameters = ParameterStore::new();
    parameters.insert("weight", weight.clone()).unwrap();
    parameters.insert("bias", bias.clone()).unwrap();
    let loss = output.sum_dims(&[0, 1]).reshape(vec![1]).unwrap();
    backward(&loss, &parameters).unwrap();
    let mut optimizer = Sgd::new(0.1).unwrap();
    optimizer.step(&parameters).unwrap();

    let next_input = tensor(&[1.0, -2.0, 3.0, 4.0], [2, 2], &device);
    let next = graph.run(&[next_input], false).unwrap().remove(0);
    assert_ne!(values(next), vec![3.0, 0.0, 7.0, 3.0]);
}

#[test]
fn exact_input_signature_rejects_shape_changes() {
    let device = Device::autodiff(Device::default());
    let input = tensor(&[1.0, 2.0, 3.0, 4.0], [2, 2], &device);
    let mut builder = GraphBuilder::new();
    let output = builder.input(&input);
    let graph = builder.finish(vec![output]).unwrap();

    let changed = DynTensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [1, 4]).convert::<f32>(),
        2,
        &device,
    )
    .unwrap();
    let error = graph.run(&[changed], false).unwrap_err();
    assert!(error.to_string().contains("expected shape [2, 2]"));
    assert!(error.to_string().contains("got shape [1, 4]"));
}

#[test]
fn structural_parameter_changes_invalidate_replay() {
    let device = Device::autodiff(Device::default());
    let input = tensor(&[1.0, 2.0, 3.0, 4.0], [2, 2], &device);
    let parameter = ParameterSlot::new(
        Some("weight".to_string()),
        tensor(&[1.0, 0.0, 0.0, 1.0], [2, 2], &device),
        true,
    )
    .unwrap();
    let mut builder = GraphBuilder::new();
    let x = builder.input(&input);
    let w = builder.parameter(parameter.clone());
    let output = builder.binary(BinaryOp::Matmul, x, w).unwrap();
    let graph = builder.finish(vec![output]).unwrap();

    parameter.set_trainable(false).unwrap();
    let error = graph.run(&[input], false).unwrap_err();
    assert!(error.to_string().contains("changed structure"));
}

#[test]
fn signatures_include_dtype_and_autodiff_device_capability() {
    let device = Device::autodiff(Device::default());
    let input = tensor(&[1.0, 2.0, 3.0, 4.0], [2, 2], &device);
    let signature = tynx_capture::TensorSignature::from_tensor(&input);

    assert_eq!(signature.shape(), [2, 2]);
    assert_eq!(signature.dtype(), DType::F32);
    assert!(signature.device().is_autodiff());
}

#[derive(Debug)]
struct SharedSgd {
    inner: RefCell<Sgd>,
    parameters: Vec<ParameterSlot>,
}

impl CapturedOptimizer for SharedSgd {
    fn step(&self) -> Result<()> {
        self.inner
            .borrow_mut()
            .step_slots(&self.parameters)
            .map(|_| ())
    }
}

#[test]
fn replays_ordered_zero_backward_and_optimizer_effects() {
    let device = Device::autodiff(Device::default());
    let input = tensor(&[2.0], [1, 1], &device);
    let parameter = ParameterSlot::new(
        Some("weight".to_string()),
        tensor(&[1.0], [1, 1], &device),
        true,
    )
    .unwrap();
    let optimizer: Rc<dyn CapturedOptimizer> = Rc::new(SharedSgd {
        inner: RefCell::new(Sgd::new(0.1).unwrap()),
        parameters: vec![parameter.clone()],
    });

    let mut builder = GraphBuilder::new();
    let x = builder.input(&input);
    builder.zero_grad(vec![parameter.clone()]);
    let weight = builder.parameter(parameter.clone());
    let prediction = builder.binary(BinaryOp::Multiply, x, weight).unwrap();
    let squared = builder
        .binary(BinaryOp::Multiply, prediction, prediction)
        .unwrap();
    let loss = builder
        .unary(
            UnaryOp::Mean {
                dims: vec![0, 1],
                output_shape: vec![1],
            },
            squared,
        )
        .unwrap();
    builder.backward(loss, vec![parameter.clone()]).unwrap();
    builder.optimizer_step(optimizer);
    let graph = builder.finish(vec![loss]).unwrap();

    let first_loss = graph
        .run(std::slice::from_ref(&input), true)
        .unwrap()
        .remove(0);
    assert_eq!(values(first_loss), vec![4.0]);
    assert!((values(parameter.value())[0] - 0.2).abs() < 1.0e-6);

    let second_loss = graph
        .run(std::slice::from_ref(&input), true)
        .unwrap()
        .remove(0);
    assert!((values(second_loss)[0] - 0.16).abs() < 1.0e-6);
    assert!((values(parameter.value())[0] - 0.04).abs() < 1.0e-6);
}
