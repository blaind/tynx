use burn::tensor::{Device, TensorData};
use protobuf::{Message, MessageField};
use tynx_core::onnx_ir::{GraphProto, ModelProto, TensorProto, TypeProto, ValueInfoProto};
use tynx_core::{DynTensor, Env, Session, TynxError, Value};

use super::ImportedModel;
use crate::{
    InitializerNameOverrides, Sgd, TrainabilityOverrides, TrainabilityReport, backward, loss::mse,
};

fn tensor(values: Vec<f32>, dims: &[usize], device: &Device) -> DynTensor {
    DynTensor::from_data(TensorData::new(values, dims.to_vec()), dims.len(), device).unwrap()
}

fn env_with_input(input: DynTensor) -> Env {
    Env::from([("x".to_string(), Value::Tensor(input))])
}

fn stable_names(session: &Session, prefix: &str) -> InitializerNameOverrides {
    let report = TrainabilityReport::analyze_initializers_with_names(
        session.graph(),
        &TrainabilityOverrides::new(),
        session.initializer_names(),
    );
    let mut names = InitializerNameOverrides::new();
    for initializer in report.trainable_parameters() {
        let suffix = match initializer.uses()[0].input_index() {
            1 => "weight",
            2 => "bias",
            index => panic!("unexpected parameter input {index}"),
        };
        names
            .set_name(initializer.name(), format!("{prefix}.{suffix}"))
            .unwrap();
    }
    names
}

fn batch_norm_names(session: &Session) -> InitializerNameOverrides {
    let report = TrainabilityReport::analyze_initializers_with_names(
        session.graph(),
        &TrainabilityOverrides::new(),
        session.initializer_names(),
    );
    let mut names = InitializerNameOverrides::new();
    for initializer in report.initializers() {
        let suffix = match initializer.uses()[0].input_index() {
            1 => "weight",
            2 => "bias",
            3 => "running_mean",
            4 => "running_var",
            index => panic!("unexpected BatchNormalization state input {index}"),
        };
        names
            .set_name(initializer.name(), format!("norm.{suffix}"))
            .unwrap();
    }
    names
}

#[test]
fn imported_gemm_learns_and_next_forward_reads_updated_slots() {
    let session = Session::from_bytes_with(&gemm_model_bytes(), false).unwrap();
    let names = stable_names(&session, "head");
    let device = Device::autodiff(Device::default());
    let model = ImportedModel::from_session_with(
        session,
        device.clone(),
        &TrainabilityOverrides::new(),
        &names,
    )
    .unwrap();
    let output_name = model.session().outputs()[0].name.clone();
    let input = tensor(vec![-1.0, 0.0, 1.0, 2.0], &[4, 1], &device);
    let target = tensor(vec![-1.0, 1.0, 3.0, 5.0], &[4, 1], &device);
    let before = model
        .run(env_with_input(input.clone()))
        .unwrap()
        .remove(&output_name)
        .unwrap()
        .into_tensor()
        .unwrap()
        .into_data()
        .iter::<f32>()
        .collect::<Vec<_>>();
    assert_eq!(before, [0.0; 4]);

    let mut optimizer = Sgd::new(0.1).unwrap();
    for _ in 0..100 {
        model.parameters().zero_grad();
        let prediction = model
            .run(env_with_input(input.clone()))
            .unwrap()
            .remove(&output_name)
            .unwrap()
            .into_tensor()
            .unwrap();
        let loss = mse(prediction, target.clone()).unwrap();
        let result = backward(&loss, model.parameters()).unwrap();
        assert_eq!(result.parameters_with_grad(), 2);
        assert_eq!(optimizer.step(model.parameters()).unwrap(), 2);
    }

    let weight = model.parameters().get_by_name("head.weight").unwrap();
    let bias = model.parameters().get_by_name("head.bias").unwrap();
    let learned_weight = weight.value().into_data().iter::<f32>().next().unwrap();
    let learned_bias = bias.value().into_data().iter::<f32>().next().unwrap();
    assert!((learned_weight - 2.0).abs() < 1.0e-3);
    assert!((learned_bias - 1.0).abs() < 1.0e-3);
    assert_eq!(weight.value_generation(), 100);
    assert_eq!(bias.value_generation(), 100);

    let after = model
        .run(env_with_input(input.clone()))
        .unwrap()
        .remove(&output_name)
        .unwrap()
        .into_tensor()
        .unwrap()
        .into_data()
        .iter::<f32>()
        .collect::<Vec<_>>();
    for (actual, expected) in after.iter().zip([-1.0, 1.0, 3.0, 5.0]) {
        assert!((actual - expected).abs() < 1.0e-3);
    }

    let frozen_inference = model
        .session()
        .run(&device, env_with_input(input))
        .unwrap()
        .remove(&output_name)
        .unwrap()
        .into_tensor()
        .unwrap()
        .into_data()
        .iter::<f32>()
        .collect::<Vec<_>>();
    assert_eq!(frozen_inference, [0.0; 4]);
}

#[test]
fn imported_simplified_model_preserves_declared_output_name() {
    let session = Session::from_bytes(&gemm_model_bytes()).unwrap();
    assert_eq!(session.outputs()[0].name, "y");
    assert_ne!(session.graph().outputs[0].name, "y");

    let device = Device::autodiff(Device::default());
    let model = ImportedModel::from_session(session, device.clone()).unwrap();
    let internal_output = model.session().graph().outputs[0].name.as_str();
    let report = model.trainability_report();
    assert_eq!(report.selected_outputs(), ["y"]);
    assert!(report.parameters_for_output("y").is_some());
    assert!(report.parameters_for_output(internal_output).is_none());

    let selected = model.trainability_for_outputs(Some(&["y"]));
    assert_eq!(selected.selected_outputs(), ["y"]);
    selected.require_trainable().unwrap();

    let outputs = model
        .run(env_with_input(tensor(
            vec![2.0, 3.0, 4.0, 5.0],
            &[4, 1],
            &device,
        )))
        .unwrap();

    assert_eq!(outputs.keys().collect::<Vec<_>>(), ["y"]);
}

#[test]
fn repeated_imported_steps_remain_generation_local() {
    const STEPS: u64 = 2_048;

    let session = Session::from_bytes_with(&gemm_model_bytes(), false).unwrap();
    let names = stable_names(&session, "head");
    let device = Device::autodiff(Device::default());
    let model = ImportedModel::from_session_with(
        session,
        device.clone(),
        &TrainabilityOverrides::new(),
        &names,
    )
    .unwrap();
    let output_name = model.session().outputs()[0].name.clone();
    let input = tensor(vec![-1.0, 0.0, 1.0, 2.0], &[4, 1], &device);
    let target = tensor(vec![-1.0, 1.0, 3.0, 5.0], &[4, 1], &device);
    let mut optimizer = Sgd::new(0.0).unwrap();

    for generation in 1..=STEPS {
        model.parameters().zero_grad();
        let prediction = model
            .run(env_with_input(input.clone()))
            .unwrap()
            .remove(&output_name)
            .unwrap()
            .into_tensor()
            .unwrap();
        let loss = mse(prediction, target.clone()).unwrap();
        let result = backward(&loss, model.parameters()).unwrap();
        assert_eq!(result.parameters_with_grad(), 2);
        drop(result);
        drop(loss);

        let weight = model.parameters().get_by_name("head.weight").unwrap();
        let bias = model.parameters().get_by_name("head.bias").unwrap();
        assert_eq!(values(weight.grad().unwrap()), [-7.0]);
        assert_eq!(values(bias.grad().unwrap()), [-4.0]);
        assert_eq!(optimizer.step(model.parameters()).unwrap(), 2);
        assert_eq!(weight.value_generation(), generation);
        assert_eq!(bias.value_generation(), generation);
        assert_eq!(weight.structure_generation(), 0);
        assert_eq!(bias.structure_generation(), 0);
    }

    assert_eq!(
        values(
            model
                .parameters()
                .get_by_name("head.weight")
                .unwrap()
                .value()
        ),
        [0.0]
    );
    assert_eq!(
        values(model.parameters().get_by_name("head.bias").unwrap().value()),
        [0.0]
    );
}

#[test]
fn unsupported_slot_consumer_is_rejected_before_forward() {
    let session = Session::from_bytes_with(&conv_model_bytes(), false).unwrap();
    let names = stable_names(&session, "conv");

    let error = ImportedModel::from_session_with(
        session,
        Device::autodiff(Device::default()),
        &TrainabilityOverrides::new(),
        &names,
    )
    .unwrap_err();

    assert!(matches!(error, TynxError::UnsupportedOp(_)));
    assert!(error.to_string().contains("slot-backed execution"));
    assert!(error.to_string().contains("Conv2d"));
}

#[test]
fn imported_model_accepts_only_declared_dynamic_dimensions() {
    let session = Session::from_bytes_with(&dynamic_batch_gemm_model_bytes(), false).unwrap();
    let names = stable_names(&session, "head");
    let device = Device::autodiff(Device::default());
    let model = ImportedModel::from_session_with(
        session,
        device.clone(),
        &TrainabilityOverrides::new(),
        &names,
    )
    .unwrap();

    for batch in [1, 3, 7] {
        let output = model
            .run(env_with_input(tensor(
                vec![1.0; batch],
                &[batch, 1],
                &device,
            )))
            .unwrap()
            .remove("y")
            .unwrap()
            .into_tensor()
            .unwrap();
        assert_eq!(output.dims(), [batch, 1]);
    }

    let error = model
        .run(env_with_input(tensor(vec![1.0; 6], &[3, 2], &device)))
        .unwrap_err();
    assert!(error.to_string().contains("expects shape [?, 1]"));
    assert!(error.to_string().contains("dimension 1 must be 1"));

    let error = model
        .run(env_with_input(tensor(vec![1.0; 3], &[3], &device)))
        .unwrap_err();
    assert!(error.to_string().contains("expects rank 2"));
}

#[test]
fn imported_model_rejects_drift_in_fixed_batch_dimensions() {
    let session = Session::from_bytes_with(&gemm_model_bytes(), false).unwrap();
    let names = stable_names(&session, "head");
    let device = Device::autodiff(Device::default());
    let model = ImportedModel::from_session_with(
        session,
        device.clone(),
        &TrainabilityOverrides::new(),
        &names,
    )
    .unwrap();

    let error = model
        .run(env_with_input(tensor(vec![1.0; 2], &[2, 1], &device)))
        .unwrap_err();
    assert!(error.to_string().contains("expects shape [4, 1]"));
    assert!(error.to_string().contains("dimension 0 must be 4"));
}

#[test]
fn imported_batch_norm_trains_affine_state_with_frozen_statistics() {
    let session = Session::from_bytes_with(&batch_norm_model_bytes(), false).unwrap();
    let names = batch_norm_names(&session);
    let device = Device::autodiff(Device::default());
    let model = ImportedModel::from_session_with(
        session,
        device.clone(),
        &TrainabilityOverrides::new(),
        &names,
    )
    .unwrap();
    let input = tensor(vec![3.0, 8.0, 5.0, 11.0], &[2, 2], &device);
    let mut output = model.run(env_with_input(input.clone())).unwrap();
    let output = output.remove("y").unwrap().into_tensor().unwrap();
    assert_eq!(values(output.clone()), [2.5, 5.0, 4.5, 8.0]);

    let running_mean = model.parameters().get_by_name("norm.running_mean").unwrap();
    let running_var = model.parameters().get_by_name("norm.running_var").unwrap();
    let mean_before = values(running_mean.value());
    let variance_before = values(running_var.value());

    backward(&output.sum_dims(&[0, 1]), model.parameters()).unwrap();
    let weight = model.parameters().get_by_name("norm.weight").unwrap();
    let bias = model.parameters().get_by_name("norm.bias").unwrap();
    assert_eq!(values(weight.grad().unwrap()), [3.0, 5.0]);
    assert_eq!(values(bias.grad().unwrap()), [2.0, 2.0]);
    assert_eq!(values(running_mean.value()), mean_before);
    assert_eq!(values(running_var.value()), variance_before);

    let mut optimizer = Sgd::new(0.1).unwrap();
    assert_eq!(optimizer.step(model.parameters()).unwrap(), 2);
    assert_eq!(values(running_mean.value()), mean_before);
    assert_eq!(values(running_var.value()), variance_before);
}

#[test]
fn imported_dropout_is_inactive_or_rejected_before_training() {
    let device = Device::autodiff(Device::default());
    let inactive_session = Session::from_bytes(&dropout_model_bytes(false)).unwrap();
    let inactive_names = stable_names(&inactive_session, "head");
    let inactive = ImportedModel::from_session_with(
        inactive_session,
        device.clone(),
        &TrainabilityOverrides::new(),
        &inactive_names,
    )
    .unwrap();
    let input = tensor(vec![2.0, 3.0], &[2, 1], &device);
    let output = inactive
        .run(env_with_input(input))
        .unwrap()
        .remove("y")
        .unwrap()
        .into_tensor()
        .unwrap();
    backward(&output.sum_dims(&[0, 1]), inactive.parameters()).unwrap();
    assert!(
        inactive
            .parameters()
            .get_by_name("head.weight")
            .unwrap()
            .grad()
            .is_some()
    );

    let active_session = Session::from_bytes(&dropout_model_bytes(true)).unwrap();
    let active_names = stable_names(&active_session, "head");
    let error = ImportedModel::from_session_with(
        active_session,
        device,
        &TrainabilityOverrides::new(),
        &active_names,
    )
    .unwrap_err();
    assert!(error.to_string().contains("Dropout"));
    assert!(error.to_string().contains("training_mode"));
}

fn gemm_model_bytes() -> Vec<u8> {
    model_bytes(
        "Gemm",
        &[4, 1],
        &[4, 1],
        vec![
            tensor_proto("weight", &[1, 1], &[0.0]),
            tensor_proto("bias", &[1], &[0.0]),
        ],
    )
}

fn dynamic_batch_gemm_model_bytes() -> Vec<u8> {
    model_bytes_with_shapes(
        "Gemm",
        &[None, Some(1)],
        &[None, Some(1)],
        vec![
            tensor_proto("weight", &[1, 1], &[0.0]),
            tensor_proto("bias", &[1], &[0.0]),
        ],
    )
}

fn batch_norm_model_bytes() -> Vec<u8> {
    let mut graph = GraphProto::new();
    graph.name = "imported_batch_norm_test".to_string();
    graph
        .input
        .push(value_info_with_shape("x", &[Some(2), Some(2)]));
    graph
        .output
        .push(value_info_with_shape("y", &[Some(2), Some(2)]));
    graph.node.push(Default::default());
    let node = graph.node.last_mut().unwrap();
    node.name = "norm".to_string();
    node.op_type = "BatchNormalization".to_string();
    node.input = vec![
        "x".to_string(),
        "scale".to_string(),
        "bias".to_string(),
        "mean".to_string(),
        "variance".to_string(),
    ];
    node.output = vec!["y".to_string()];
    graph.initializer = vec![
        tensor_proto("scale", &[2], &[2.0, 3.0]),
        tensor_proto("bias", &[2], &[0.5, -1.0]),
        tensor_proto("mean", &[2], &[1.0, 2.0]),
        tensor_proto("variance", &[2], &[4.0, 9.0]),
    ];

    finish_model(graph, 13)
}

fn dropout_model_bytes(training: bool) -> Vec<u8> {
    let mut graph = GraphProto::new();
    graph.name = "imported_dropout_test".to_string();
    graph
        .input
        .push(value_info_with_shape("x", &[None, Some(1)]));
    graph
        .output
        .push(value_info_with_shape("y", &[None, Some(1)]));

    graph.node.push(Default::default());
    let gemm = graph.node.last_mut().unwrap();
    gemm.name = "head".to_string();
    gemm.op_type = "Gemm".to_string();
    gemm.input = vec!["x".to_string(), "weight".to_string(), "bias".to_string()];
    gemm.output = vec!["hidden".to_string()];

    graph.node.push(Default::default());
    let dropout = graph.node.last_mut().unwrap();
    dropout.name = "dropout".to_string();
    dropout.op_type = "Dropout".to_string();
    dropout.input = vec![
        "hidden".to_string(),
        "ratio".to_string(),
        "training".to_string(),
    ];
    dropout.output = vec!["y".to_string(), "mask".to_string()];

    graph.initializer = vec![
        tensor_proto("weight", &[1, 1], &[1.0]),
        tensor_proto("bias", &[1], &[0.0]),
        tensor_proto("ratio", &[], &[0.5]),
        bool_tensor_proto("training", training),
    ];

    finish_model(graph, 13)
}

fn conv_model_bytes() -> Vec<u8> {
    model_bytes(
        "Conv",
        &[1, 1, 2, 2],
        &[1, 1, 2, 2],
        vec![
            tensor_proto("weight", &[1, 1, 1, 1], &[1.0]),
            tensor_proto("bias", &[1], &[0.0]),
        ],
    )
}

fn model_bytes(
    operator: &str,
    input_shape: &[usize],
    output_shape: &[usize],
    initializers: Vec<TensorProto>,
) -> Vec<u8> {
    let input_shape = input_shape.iter().copied().map(Some).collect::<Vec<_>>();
    let output_shape = output_shape.iter().copied().map(Some).collect::<Vec<_>>();
    model_bytes_with_shapes(operator, &input_shape, &output_shape, initializers)
}

fn model_bytes_with_shapes(
    operator: &str,
    input_shape: &[Option<usize>],
    output_shape: &[Option<usize>],
    initializers: Vec<TensorProto>,
) -> Vec<u8> {
    let mut graph = GraphProto::new();
    graph.name = "imported_training_test".to_string();
    graph.input.push(value_info_with_shape("x", input_shape));
    graph.output.push(value_info_with_shape("y", output_shape));
    graph.node.push(Default::default());
    let node = graph.node.last_mut().unwrap();
    node.name = "layer".to_string();
    node.op_type = operator.to_string();
    node.input = vec!["x".to_string(), "weight".to_string(), "bias".to_string()];
    node.output = vec!["y".to_string()];
    graph.initializer = initializers;

    finish_model(graph, 13)
}

fn finish_model(graph: GraphProto, opset: i64) -> Vec<u8> {
    let mut model = ModelProto::new();
    model.ir_version = 8;
    model.graph = MessageField::some(graph);
    model.opset_import.push(Default::default());
    model.opset_import[0].version = opset;
    model.write_to_bytes().unwrap()
}

fn value_info_with_shape(name: &str, dimensions: &[Option<usize>]) -> ValueInfoProto {
    let mut value = ValueInfoProto::new();
    value.name = name.to_string();
    let mut ty = TypeProto::new();
    let tensor = ty.mut_tensor_type();
    tensor.elem_type = 1;
    let shape = tensor.shape.mut_or_insert_default();
    for dimension in dimensions {
        shape.dim.push(Default::default());
        let output = shape.dim.last_mut().unwrap();
        match dimension {
            Some(dimension) => output.set_dim_value(*dimension as i64),
            None => output.set_dim_param("batch".to_string()),
        }
    }
    value.type_ = MessageField::some(ty);
    value
}

fn tensor_proto(name: &str, dimensions: &[usize], values: &[f32]) -> TensorProto {
    let mut tensor = TensorProto::new();
    tensor.name = name.to_string();
    tensor.dims = dimensions
        .iter()
        .map(|dimension| *dimension as i64)
        .collect();
    tensor.data_type = 1;
    tensor.float_data = values.to_vec();
    tensor
}

fn bool_tensor_proto(name: &str, value: bool) -> TensorProto {
    let mut tensor = TensorProto::new();
    tensor.name = name.to_string();
    tensor.data_type = 9;
    tensor.int32_data = vec![i32::from(value)];
    tensor
}

fn values(tensor: DynTensor) -> Vec<f32> {
    tensor.into_data().iter::<f32>().collect()
}
