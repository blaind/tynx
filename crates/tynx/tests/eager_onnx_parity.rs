//! Parity between eager tensor operations and equivalent imported ONNX nodes.

use tynx::onnx_ir::{
    DType, Node,
    node::{
        arithmetic::{AddNodeBuilder, MulNodeBuilder, SubNodeBuilder},
        matmul::MatMulNodeBuilder,
        reduce::{ReduceConfig, ReduceMeanNodeBuilder},
        relu::ReluNodeBuilder,
    },
};
use tynx::{Device, DynTensor, Env, TensorData, Value, execute};

fn tensor(values: Vec<f32>, dims: &[usize], device: &Device) -> DynTensor {
    DynTensor::from_data(TensorData::new(values, dims.to_vec()), dims.len(), device).unwrap()
}

fn assert_tensor_close(actual: DynTensor, expected: DynTensor) {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.into_data().to_vec::<f32>().unwrap();
    let expected = expected.into_data().to_vec::<f32>().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "element {index} differs: eager={expected}, ONNX={actual}"
        );
    }
}

#[test]
fn eager_mlp_slice_matches_equivalent_onnx_nodes() {
    let device = Device::default();
    let input = tensor(vec![1.0, -2.0, 0.5, 3.0], &[2, 2], &device);
    let weight = tensor(vec![2.0, -1.0, 0.5, 3.0, -2.0, 1.0], &[2, 3], &device);
    let bias = tensor(vec![0.25, -0.5, 1.0], &[3], &device);

    let eager_matmul = input.clone().matmul(weight.clone()).unwrap();
    let eager_add = eager_matmul.clone().add_broadcast(bias.clone()).unwrap();
    let eager_relu = eager_add.clone().relu();

    let matmul = Node::MatMul(
        MatMulNodeBuilder::new("matmul")
            .input_tensor("input", 2, DType::F32)
            .input_tensor("weight", 2, DType::F32)
            .output_tensor("projected", 2, DType::F32)
            .build(),
    );
    let add = Node::Add(
        AddNodeBuilder::new("add_bias")
            .input_tensor("projected", 2, DType::F32)
            .input_tensor("bias", 1, DType::F32)
            .output_tensor("biased", 2, DType::F32)
            .build(),
    );
    let relu = Node::Relu(
        ReluNodeBuilder::new("relu")
            .input_tensor("biased", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build(),
    );

    let mut env = Env::from([
        ("input".to_string(), Value::Tensor(input)),
        ("weight".to_string(), Value::Tensor(weight)),
        ("bias".to_string(), Value::Tensor(bias)),
    ]);

    let projected = execute(&matmul, &env, &device)
        .unwrap()
        .pop()
        .unwrap()
        .into_tensor()
        .unwrap();
    assert_tensor_close(projected.clone(), eager_matmul);
    env.insert("projected".to_string(), Value::Tensor(projected));

    let biased = execute(&add, &env, &device)
        .unwrap()
        .pop()
        .unwrap()
        .into_tensor()
        .unwrap();
    assert_tensor_close(biased.clone(), eager_add);
    env.insert("biased".to_string(), Value::Tensor(biased));

    let output = execute(&relu, &env, &device)
        .unwrap()
        .pop()
        .unwrap()
        .into_tensor()
        .unwrap();
    assert_tensor_close(output, eager_relu);
}

#[test]
fn eager_mse_primitives_match_equivalent_onnx_nodes() {
    let device = Device::default();
    let prediction = tensor(vec![1.0, 3.0, -1.0, 2.0], &[2, 2], &device);
    let target = tensor(vec![0.0, 1.0, 1.0, 3.0], &[2, 2], &device);

    let eager_difference = prediction.clone().sub_broadcast(target.clone()).unwrap();
    let eager_squared = eager_difference
        .clone()
        .mul_broadcast(eager_difference.clone())
        .unwrap();
    let eager_mean = eager_squared.clone().mean_dims(&[0, 1]);

    let sub = Node::Sub(
        SubNodeBuilder::new("difference")
            .input_tensor("prediction", 2, DType::F32)
            .input_tensor("target", 2, DType::F32)
            .output_tensor("difference", 2, DType::F32)
            .build(),
    );
    let mul = Node::Mul(
        MulNodeBuilder::new("square")
            .input_tensor("difference", 2, DType::F32)
            .input_tensor("difference", 2, DType::F32)
            .output_tensor("squared", 2, DType::F32)
            .build(),
    );
    let mean = Node::ReduceMean(
        ReduceMeanNodeBuilder::new("mean")
            .input_tensor("squared", 2, DType::F32)
            .output_tensor("loss", 2, DType::F32)
            .config(ReduceConfig {
                dims: vec![0, 1],
                keepdims: true,
            })
            .build(),
    );

    let mut env = Env::from([
        ("prediction".to_string(), Value::Tensor(prediction)),
        ("target".to_string(), Value::Tensor(target)),
    ]);

    let difference = execute(&sub, &env, &device)
        .unwrap()
        .pop()
        .unwrap()
        .into_tensor()
        .unwrap();
    assert_tensor_close(difference.clone(), eager_difference);
    env.insert("difference".to_string(), Value::Tensor(difference));

    let squared = execute(&mul, &env, &device)
        .unwrap()
        .pop()
        .unwrap()
        .into_tensor()
        .unwrap();
    assert_tensor_close(squared.clone(), eager_squared);
    env.insert("squared".to_string(), Value::Tensor(squared));

    let mean = execute(&mean, &env, &device)
        .unwrap()
        .pop()
        .unwrap()
        .into_tensor()
        .unwrap();
    assert_tensor_close(mean, eager_mean);
}
