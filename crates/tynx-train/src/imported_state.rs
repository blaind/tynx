//! Materialization of classified ONNX initializers into stable parameter slots.

use std::collections::{HashMap, HashSet};

use burn::tensor::{DType, Device, TensorData};
use tynx_core::onnx_ir::ir::{Argument, OnnxGraph};
use tynx_core::{DynTensor, Result, TynxError};

use crate::{
    InitializerId, InitializerRole, ParamId, ParameterSlot, ParameterStore, TrainabilityOverrides,
    TrainabilityReport,
};

/// Stable checkpoint-name replacements for processed initializer names.
///
/// Tynx refuses to persist generated `__static#N` names for genuinely synthetic graph values, so
/// callers may supply a stable replacement for those values before materialization.
#[derive(Debug, Clone, Default)]
pub struct InitializerNameOverrides {
    names: HashMap<String, String>,
}

impl InitializerNameOverrides {
    /// Create an empty name-override set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace one report name with a stable state/checkpoint name.
    pub fn set_name(
        &mut self,
        report_name: impl Into<String>,
        state_name: impl Into<String>,
    ) -> Result<()> {
        let report_name = report_name.into();
        let state_name = state_name.into();
        if report_name.trim().is_empty() {
            return Err(TynxError::TypeMismatch(
                "initializer report name cannot be empty".to_string(),
            ));
        }
        if state_name.trim().is_empty() {
            return Err(TynxError::TypeMismatch(
                "initializer state name cannot be empty".to_string(),
            ));
        }
        self.names.insert(report_name, state_name);
        Ok(())
    }
}

/// Stable runtime state materialized from one processed ONNX graph.
#[derive(Debug)]
pub struct ImportedState {
    store: ParameterStore,
    initializer_slots: HashMap<InitializerId, ParamId>,
    initializer_roles: HashMap<InitializerId, InitializerRole>,
    report: TrainabilityReport,
}

impl ImportedState {
    /// Classify and materialize graph state using automatic roles and preserved names.
    pub fn materialize(graph: &OnnxGraph, device: &Device) -> Result<Self> {
        Self::materialize_with(
            graph,
            device,
            &TrainabilityOverrides::new(),
            &InitializerNameOverrides::new(),
        )
    }

    /// Classify and materialize graph state with explicit role and stable-name overrides.
    pub fn materialize_with(
        graph: &OnnxGraph,
        device: &Device,
        role_overrides: &TrainabilityOverrides,
        name_overrides: &InitializerNameOverrides,
    ) -> Result<Self> {
        Self::materialize_with_names(
            graph,
            device,
            role_overrides,
            name_overrides,
            &HashMap::new(),
        )
    }

    /// Classify and materialize graph state using preserved ONNX initializer provenance.
    pub fn materialize_with_names(
        graph: &OnnxGraph,
        device: &Device,
        role_overrides: &TrainabilityOverrides,
        name_overrides: &InitializerNameOverrides,
        stable_names: &HashMap<InitializerId, String>,
    ) -> Result<Self> {
        let report = TrainabilityReport::analyze_initializers_with_names(
            graph,
            role_overrides,
            stable_names,
        );
        let mut values = HashMap::<InitializerId, TensorData>::new();
        for (node_index, node) in graph.nodes.iter().enumerate() {
            for (input_index, input) in node.inputs().iter().enumerate() {
                let Some(id) = InitializerId::from_argument(input, node_index, input_index) else {
                    continue;
                };
                if values.contains_key(&id) {
                    continue;
                }
                if let Some(value) = input.value() {
                    values.insert(id, value);
                }
            }
        }
        materialize_report(report, device, name_overrides, |id| values.get(id).cloned())
    }

    /// Return the role/classification report used to construct this state.
    pub fn report(&self) -> &TrainabilityReport {
        &self.report
    }

    /// Return all materialized parameters and buffers.
    pub fn store(&self) -> &ParameterStore {
        &self.store
    }

    /// Consume this imported state and return its generic parameter store.
    pub fn into_store(self) -> ParameterStore {
        self.store
    }

    /// Return the stable slot associated with one processed initializer identity.
    pub fn slot_for_initializer(&self, id: &InitializerId) -> Option<&ParameterSlot> {
        self.initializer_slots
            .get(id)
            .and_then(|slot_id| self.store.get(*slot_id))
    }

    /// Resolve an embedded node input to its materialized slot.
    pub fn slot_for_input(
        &self,
        node_index: usize,
        input_index: usize,
        input: &Argument,
    ) -> Option<&ParameterSlot> {
        let id = InitializerId::from_argument(input, node_index, input_index)?;
        self.slot_for_initializer(&id)
    }

    /// Return the classified role for a materialized initializer.
    pub fn role_for_initializer(&self, id: &InitializerId) -> Option<InitializerRole> {
        self.initializer_roles.get(id).copied()
    }

    /// Iterate named trainable parameter slots.
    pub fn parameters(&self) -> impl Iterator<Item = (&str, &ParameterSlot)> {
        self.store
            .named()
            .filter(|(_, slot)| slot.contract().trainable())
    }

    /// Iterate named fixed buffer slots.
    pub fn buffers(&self) -> impl Iterator<Item = (&str, &ParameterSlot)> {
        self.store
            .named()
            .filter(|(_, slot)| !slot.contract().trainable())
    }
}

fn materialize_report(
    report: TrainabilityReport,
    device: &Device,
    name_overrides: &InitializerNameOverrides,
    mut load: impl FnMut(&InitializerId) -> Option<TensorData>,
) -> Result<ImportedState> {
    if !report.roles_ready() {
        let reason = report.errors().first().cloned().unwrap_or_else(|| {
            "one or more initializer roles are ambiguous; provide explicit role overrides"
                .to_string()
        });
        return Err(TynxError::TypeMismatch(format!(
            "cannot materialize imported state: {reason}"
        )));
    }

    let mut store = ParameterStore::new();
    let mut initializer_slots = HashMap::new();
    let mut initializer_roles = HashMap::new();
    let mut used_name_overrides = HashSet::new();

    for initializer in report.initializers() {
        let role = initializer.role();
        if role == InitializerRole::Constant {
            continue;
        }
        if role == InitializerRole::Ambiguous {
            return Err(TynxError::TypeMismatch(format!(
                "initializer '{}' has an ambiguous state role",
                initializer.name()
            )));
        }

        let state_name = match name_overrides.names.get(initializer.name()) {
            Some(name) => {
                used_name_overrides.insert(initializer.name().to_string());
                name.clone()
            }
            None if initializer.has_synthetic_name() => {
                return Err(TynxError::TypeMismatch(format!(
                    "initializer '{}' has no stable ONNX provenance; provide a stable name override before materialization",
                    initializer.name()
                )));
            }
            None => initializer.name().to_string(),
        };

        let value = load(initializer.id()).ok_or_else(|| {
            TynxError::MissingValue(format!(
                "initializer '{}' ({:?}) has no embedded tensor data",
                initializer.name(),
                initializer.id()
            ))
        })?;
        validate_tensor_metadata(initializer, &value)?;
        let rank = value.shape.len();
        if !(1..=6).contains(&rank) {
            return Err(TynxError::Shape(format!(
                "initializer '{state_name}' has rank {rank}; imported state currently supports ranks 1 through 6"
            )));
        }
        if value.dtype != DType::F32 {
            return Err(TynxError::TypeMismatch(format!(
                "initializer '{state_name}' has dtype {:?}; v1 materialized training parameters and buffers must use f32",
                value.dtype
            )));
        }

        let tensor = DynTensor::from_data(value, rank, device)?;
        let trainable = role == InitializerRole::Parameter;
        let slot = ParameterSlot::new(Some(state_name.clone()), tensor, trainable)?;
        let slot_id = store.insert(state_name, slot)?;
        initializer_slots.insert(initializer.id().clone(), slot_id);
        initializer_roles.insert(initializer.id().clone(), role);
    }

    for report_name in name_overrides.names.keys() {
        if !used_name_overrides.contains(report_name) {
            return Err(TynxError::TypeMismatch(format!(
                "initializer name override '{report_name}' did not match a materialized parameter or buffer"
            )));
        }
    }

    Ok(ImportedState {
        store,
        initializer_slots,
        initializer_roles,
        report,
    })
}

fn validate_tensor_metadata(
    initializer: &crate::InitializerReport,
    value: &TensorData,
) -> Result<()> {
    if initializer.dtype() != Some(value.dtype) {
        return Err(TynxError::TypeMismatch(format!(
            "initializer '{}' report dtype {:?} differs from embedded dtype {:?}",
            initializer.name(),
            initializer.dtype(),
            value.dtype
        )));
    }
    let embedded_shape = value.shape.iter().copied().collect::<Vec<_>>();
    if let Some(shape) = initializer.shape()
        && shape != embedded_shape
    {
        return Err(TynxError::Shape(format!(
            "initializer '{}' report shape {:?} differs from embedded shape {:?}",
            initializer.name(),
            shape,
            embedded_shape
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use protobuf::{Message, MessageField};
    use tynx_core::onnx_ir::{
        DType, GraphProto, ModelProto, Node, TensorProto, TypeProto, ValueInfoProto,
        ir::{ArgType, TensorType, ValueSource},
        node::{
            batch_norm::{BatchNormConfig, BatchNormRuntimeConfig, BatchNormalizationNode},
            linear::{LinearConfig, LinearNode},
        },
    };

    use super::*;

    fn dynamic(name: &str, shape: &[usize]) -> Argument {
        Argument::new(
            name,
            ArgType::Tensor(TensorType::new_known(DType::F32, shape.to_vec())),
        )
    }

    fn constant(name: &str, shape: &[usize]) -> Argument {
        let mut argument = dynamic(name, shape);
        argument.value_source = ValueSource::Constant;
        argument
    }

    fn static_tensor(data_id: usize, shape: &[usize]) -> Argument {
        let mut argument = dynamic("", shape);
        argument.value_source = ValueSource::Static(data_id);
        argument
    }

    fn linear(name: &str, weight: Argument, bias: Argument) -> Node {
        Node::Linear(LinearNode {
            name: name.to_string(),
            inputs: vec![dynamic("x", &[1, 2]), weight, bias],
            outputs: vec![dynamic(&format!("{name}.output"), &[1, 2])],
            config: LinearConfig::new(false),
        })
    }

    fn graph(nodes: Vec<Node>) -> OnnxGraph {
        let mut graph = OnnxGraph::default();
        graph.nodes = nodes;
        graph
    }

    fn data(values: Vec<f32>, shape: &[usize]) -> TensorData {
        TensorData::new(values, shape.to_vec())
    }

    #[test]
    fn materializes_parameters_and_buffers_with_stable_roles() {
        let graph = graph(vec![
            linear(
                "encoder",
                constant("encoder.weight", &[2, 2]),
                constant("encoder.bias", &[2]),
            ),
            Node::BatchNormalization(BatchNormalizationNode {
                name: "norm".to_string(),
                inputs: vec![
                    dynamic("encoder.output", &[1, 2]),
                    constant("norm.scale", &[2]),
                    constant("norm.bias", &[2]),
                    constant("norm.mean", &[2]),
                    constant("norm.variance", &[2]),
                ],
                outputs: vec![dynamic("output", &[1, 2])],
                config: BatchNormConfig::Runtime(BatchNormRuntimeConfig::new(1.0e-5, 0.9)),
            }),
        ]);
        let report = TrainabilityReport::analyze_initializers(&graph);
        let values = HashMap::from([
            (
                InitializerId::Named("encoder.weight".to_string()),
                data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]),
            ),
            (
                InitializerId::Named("encoder.bias".to_string()),
                data(vec![0.1, 0.2], &[2]),
            ),
            (
                InitializerId::Named("norm.scale".to_string()),
                data(vec![1.0, 1.0], &[2]),
            ),
            (
                InitializerId::Named("norm.bias".to_string()),
                data(vec![0.0, 0.0], &[2]),
            ),
            (
                InitializerId::Named("norm.mean".to_string()),
                data(vec![0.5, 0.5], &[2]),
            ),
            (
                InitializerId::Named("norm.variance".to_string()),
                data(vec![2.0, 2.0], &[2]),
            ),
        ]);
        let device = Device::autodiff(Device::default());

        let state = materialize_report(report, &device, &InitializerNameOverrides::new(), |id| {
            values.get(id).cloned()
        })
        .unwrap();

        assert_eq!(state.store().len(), 6);
        assert_eq!(
            state.parameters().map(|(name, _)| name).collect::<Vec<_>>(),
            ["encoder.weight", "encoder.bias", "norm.scale", "norm.bias"]
        );
        assert_eq!(
            state.buffers().map(|(name, _)| name).collect::<Vec<_>>(),
            ["norm.mean", "norm.variance"]
        );
        let weight_id = InitializerId::Named("encoder.weight".to_string());
        let weight = state.slot_for_initializer(&weight_id).unwrap();
        assert!(weight.contract().trainable());
        assert_eq!(
            weight.value().into_data().iter::<f32>().collect::<Vec<_>>(),
            [1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            state.role_for_initializer(&InitializerId::Named("norm.mean".to_string())),
            Some(InitializerRole::Buffer)
        );
        assert_eq!(
            state
                .slot_for_input(0, 1, &graph.nodes[0].inputs()[1])
                .unwrap()
                .id(),
            weight.id()
        );
    }

    #[test]
    fn shared_initializer_materializes_once_and_resolves_from_every_use() {
        let shared = constant("shared.weight", &[2, 2]);
        let graph = graph(vec![
            linear("first", shared.clone(), constant("first.bias", &[2])),
            linear("second", shared, constant("second.bias", &[2])),
        ]);
        let report = TrainabilityReport::analyze_initializers(&graph);
        let values = HashMap::from([
            (
                InitializerId::Named("shared.weight".to_string()),
                data(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]),
            ),
            (
                InitializerId::Named("first.bias".to_string()),
                data(vec![0.0, 0.0], &[2]),
            ),
            (
                InitializerId::Named("second.bias".to_string()),
                data(vec![0.0, 0.0], &[2]),
            ),
        ]);
        let state = materialize_report(
            report,
            &Device::autodiff(Device::default()),
            &InitializerNameOverrides::new(),
            |id| values.get(id).cloned(),
        )
        .unwrap();

        assert_eq!(state.store().len(), 3);
        assert_eq!(
            state
                .slot_for_input(0, 1, &graph.nodes[0].inputs()[1])
                .unwrap()
                .id(),
            state
                .slot_for_input(1, 1, &graph.nodes[1].inputs()[1])
                .unwrap()
                .id()
        );
    }

    #[test]
    fn synthetic_processed_name_requires_stable_override() {
        let graph = graph(vec![linear(
            "encoder",
            static_tensor(7, &[2, 2]),
            constant("encoder.bias", &[2]),
        )]);
        let report = TrainabilityReport::analyze_initializers(&graph);
        let values = HashMap::from([
            (
                InitializerId::Static(7),
                data(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]),
            ),
            (
                InitializerId::Named("encoder.bias".to_string()),
                data(vec![0.0, 0.0], &[2]),
            ),
        ]);
        let device = Device::autodiff(Device::default());

        let error = materialize_report(
            report.clone(),
            &device,
            &InitializerNameOverrides::new(),
            |id| values.get(id).cloned(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("stable name override"));

        let mut names = InitializerNameOverrides::new();
        names.set_name("__static#7", "encoder.weight").unwrap();
        let state =
            materialize_report(report, &device, &names, |id| values.get(id).cloned()).unwrap();

        assert!(state.store().get_by_name("encoder.weight").is_some());
        assert!(state.store().get_by_name("__static#7").is_none());
    }

    #[test]
    fn public_graph_materializer_reports_missing_embedded_data() {
        let graph = graph(vec![linear(
            "encoder",
            constant("encoder.weight", &[2, 2]),
            constant("encoder.bias", &[2]),
        )]);

        let error =
            ImportedState::materialize(&graph, &Device::autodiff(Device::default())).unwrap_err();

        assert!(matches!(error, TynxError::MissingValue(_)));
        assert!(error.to_string().contains("encoder.weight"));
    }

    #[test]
    fn trainable_materialization_requires_autodiff_device() {
        let graph = graph(vec![linear(
            "encoder",
            constant("encoder.weight", &[2, 2]),
            constant("encoder.bias", &[2]),
        )]);
        let report = TrainabilityReport::analyze_initializers(&graph);
        let values = HashMap::from([
            (
                InitializerId::Named("encoder.weight".to_string()),
                data(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]),
            ),
            (
                InitializerId::Named("encoder.bias".to_string()),
                data(vec![0.0, 0.0], &[2]),
            ),
        ]);

        let error = materialize_report(
            report,
            &Device::default(),
            &InitializerNameOverrides::new(),
            |id| values.get(id).cloned(),
        )
        .unwrap_err();

        assert!(error.to_string().contains("autodiff-enabled device"));
    }

    #[test]
    fn unused_name_override_is_rejected() {
        let graph = graph(vec![linear(
            "encoder",
            constant("encoder.weight", &[2, 2]),
            constant("encoder.bias", &[2]),
        )]);
        let report = TrainabilityReport::analyze_initializers(&graph);
        let values = HashMap::from([
            (
                InitializerId::Named("encoder.weight".to_string()),
                data(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]),
            ),
            (
                InitializerId::Named("encoder.bias".to_string()),
                data(vec![0.0, 0.0], &[2]),
            ),
        ]);
        let mut names = InitializerNameOverrides::new();
        names.set_name("missing", "renamed").unwrap();

        let error =
            materialize_report(report, &Device::autodiff(Device::default()), &names, |id| {
                values.get(id).cloned()
            })
            .unwrap_err();

        assert!(error.to_string().contains("did not match"));
    }

    #[test]
    fn materializes_tensor_data_from_a_parsed_onnx_graph() {
        let bytes = gemm_model_bytes();
        let session = tynx_core::Session::from_bytes_with(&bytes, false).unwrap();
        let report = TrainabilityReport::analyze_initializers(session.graph());
        let mut names = InitializerNameOverrides::new();
        for initializer in report.trainable_parameters() {
            let state_name = match initializer.uses()[0].input_index() {
                1 => "head.weight",
                2 => "head.bias",
                index => panic!("unexpected Gemm parameter input {index}"),
            };
            names.set_name(initializer.name(), state_name).unwrap();
        }

        let state = ImportedState::materialize_with(
            session.graph(),
            &Device::autodiff(Device::default()),
            &TrainabilityOverrides::new(),
            &names,
        )
        .unwrap();

        assert_eq!(state.store().len(), 2);
        assert_eq!(
            state
                .store()
                .get_by_name("head.weight")
                .unwrap()
                .value()
                .into_data()
                .iter::<f32>()
                .collect::<Vec<_>>(),
            [1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            state
                .store()
                .get_by_name("head.bias")
                .unwrap()
                .value()
                .into_data()
                .iter::<f32>()
                .collect::<Vec<_>>(),
            [0.25, -0.5]
        );
    }

    fn gemm_model_bytes() -> Vec<u8> {
        let mut graph = GraphProto::new();
        graph.name = "materialization_test".to_string();
        graph.input.push(value_info("x", &[1, 2]));
        graph.output.push(value_info("y", &[1, 2]));

        graph.node.push(Default::default());
        let node = graph.node.last_mut().unwrap();
        node.name = "head".to_string();
        node.op_type = "Gemm".to_string();
        node.input = vec!["x".to_string(), "weight".to_string(), "bias".to_string()];
        node.output = vec!["y".to_string()];

        graph
            .initializer
            .push(tensor_proto("weight", &[2, 2], &[1.0, 2.0, 3.0, 4.0]));
        graph
            .initializer
            .push(tensor_proto("bias", &[2], &[0.25, -0.5]));

        let mut model = ModelProto::new();
        model.ir_version = 8;
        model.graph = MessageField::some(graph);
        model.opset_import.push(Default::default());
        model.opset_import[0].version = 13;
        model.write_to_bytes().unwrap()
    }

    fn value_info(name: &str, dimensions: &[usize]) -> ValueInfoProto {
        let mut value = ValueInfoProto::new();
        value.name = name.to_string();
        let mut ty = TypeProto::new();
        let tensor = ty.mut_tensor_type();
        tensor.elem_type = 1;
        let shape = tensor.shape.mut_or_insert_default();
        for dimension in dimensions {
            shape.dim.push(Default::default());
            shape
                .dim
                .last_mut()
                .unwrap()
                .set_dim_value(*dimension as i64);
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
}
