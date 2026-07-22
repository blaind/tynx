//! Loading, preparing, and inspecting ONNX models.

use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
    sync::Arc,
};

use burn::tensor::Device;
use onnx_ir::OnnxGraphBuilder;
use onnx_ir::ir::{Argument, Node, OnnxGraph};

use crate::{
    Env, InitializerId, Result, TynxError, Value, execute, initializer::env_key,
    interpreter::resolve,
};

/// A parsed ONNX model.
#[derive(Debug, Clone)]
pub struct Session {
    graph: Arc<OnnxGraph>,
}

/// A parsed model whose embedded values have been materialized on one device.
///
/// Preparing is the intended path for repeated inference. [`Session::run`] remains a
/// deliberately uncached convenience for one-shot execution.
#[derive(Debug, Clone)]
pub struct PreparedSession {
    session: Session,
    device: Device,
    initializers: HashMap<InitializerId, Value>,
    plan: ExecutionPlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SlotId(usize);

#[derive(Debug, Clone, Default)]
struct NodePlan {
    dead_slots: Vec<SlotId>,
    unnamed_initializers: Vec<(usize, InitializerId)>,
}

#[derive(Debug, Clone, Default)]
struct ExecutionPlan {
    slot_names: Vec<String>,
    nodes: Vec<NodePlan>,
}

impl Session {
    /// Load a model from a file and simplify its graph.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_file_with(path, true)
    }

    /// Load a model from a file with optional graph simplification.
    pub fn from_file_with(path: impl AsRef<Path>, simplify: bool) -> Result<Self> {
        let data = fs::read(path).map_err(|error| TynxError::Parse(error.to_string()))?;
        Self::from_bytes_with(&data, simplify)
    }

    /// Load a model from bytes and simplify its graph.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Self::from_bytes_with(data, true)
    }

    /// Load a model from bytes with optional graph simplification.
    pub fn from_bytes_with(data: &[u8], simplify: bool) -> Result<Self> {
        let (prepared, changed) = crate::interpreter::prepare_model(data)?;
        let parse_data = if changed { prepared.as_slice() } else { data };
        let mut graph = OnnxGraphBuilder::new()
            .simplify(simplify)
            .parse_bytes(parse_data)?;
        if changed {
            crate::interpreter::restore_dynamic_conv_inputs(data, &mut graph)?;
        }
        crate::interpreter::preserve_attributes(data, &mut graph)?;

        Ok(Self {
            graph: Arc::new(graph),
        })
    }

    /// Return the parsed graph.
    pub fn graph(&self) -> &OnnxGraph {
        &self.graph
    }

    /// Return the model's declared inputs.
    pub fn inputs(&self) -> &[Argument] {
        &self.graph.inputs
    }

    /// Return the model's declared outputs.
    pub fn outputs(&self) -> &[Argument] {
        &self.graph.outputs
    }

    /// Materialize all unique embedded graph values on `device` for repeated inference.
    pub fn prepare(&self, device: &Device) -> Result<PreparedSession> {
        let mut initializers = HashMap::new();
        for (node_index, node) in self.graph.nodes.iter().enumerate() {
            for (input_index, input) in node.inputs().iter().enumerate() {
                let Some(id) = InitializerId::from_argument(input, node_index, input_index) else {
                    continue;
                };
                if initializers.contains_key(&id) {
                    continue;
                }
                if let Some(data) = input.value() {
                    initializers.insert(id, resolve::materialize(input, data, device)?);
                }
            }
        }

        Ok(PreparedSession {
            session: self.clone(),
            device: device.clone(),
            initializers,
            plan: ExecutionPlan::new(&self.graph),
        })
    }

    /// Run one-shot inference and return the graph outputs by name.
    ///
    /// Embedded values are materialized for every call. Use [`Session::prepare`] when
    /// running a model more than once.
    pub fn run(&self, device: &Device, mut env: Env) -> Result<Env> {
        run_graph(&self.graph, &mut env, device)?;
        collect_outputs(&self.graph, &env)
    }
}

impl PreparedSession {
    /// Return the parsed session shared by this prepared state.
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Return the parsed graph.
    pub fn graph(&self) -> &OnnxGraph {
        self.session.graph()
    }

    /// Return the model's declared inputs.
    pub fn inputs(&self) -> &[Argument] {
        self.session.inputs()
    }

    /// Return the model's declared outputs.
    pub fn outputs(&self) -> &[Argument] {
        self.session.outputs()
    }

    /// Return the exact device used to prepare this state.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Return the number of unique embedded values materialized during preparation.
    pub fn initializer_count(&self) -> usize {
        self.initializers.len()
    }

    /// Run inference using cached embedded values.
    pub fn run(&self, mut env: Env) -> Result<Env> {
        self.validate_input_devices(&env)?;
        for (id, value) in &self.initializers {
            if !matches!(id, InitializerId::Unnamed { .. }) {
                env.insert(env_key(id), value.clone());
            }
        }
        run_graph_prepared(
            &self.session.graph,
            &mut env,
            &self.device,
            &self.initializers,
            &self.plan,
        )?;
        collect_outputs(&self.session.graph, &env)
    }

    fn validate_input_devices(&self, env: &Env) -> Result<()> {
        for (name, value) in env {
            let Some(actual) = value.device() else {
                continue;
            };
            if actual != self.device || actual.is_autodiff() != self.device.is_autodiff() {
                return Err(TynxError::DeviceMismatch {
                    name: name.clone(),
                    expected: format!("{:?}", self.device),
                    actual: format!("{actual:?}"),
                });
            }
        }
        Ok(())
    }
}

impl ExecutionPlan {
    fn new(graph: &OnnxGraph) -> Self {
        let mut slot_ids = HashMap::<String, SlotId>::new();
        let mut slot_names = Vec::new();
        let mut slot = |name: &str| {
            if name.is_empty() {
                return None;
            }
            Some(*slot_ids.entry(name.to_string()).or_insert_with(|| {
                let id = SlotId(slot_names.len());
                slot_names.push(name.to_string());
                id
            }))
        };

        for input in &graph.inputs {
            slot(&input.name);
        }
        for node in &graph.nodes {
            for input in node.inputs() {
                slot(&input.name);
            }
            for output in node.outputs() {
                slot(&output.name);
            }
        }
        for output in &graph.outputs {
            slot(&output.name);
        }

        let mut nodes = vec![NodePlan::default(); graph.nodes.len()];
        for (node_index, node) in graph.nodes.iter().enumerate() {
            for (input_index, input) in node.inputs().iter().enumerate() {
                if let Some(id @ InitializerId::Unnamed { .. }) =
                    InitializerId::from_argument(input, node_index, input_index)
                {
                    nodes[node_index]
                        .unnamed_initializers
                        .push((input_index, id));
                }
            }
        }

        // Control-flow subgraphs inherit values from their outer environment. Until
        // subgraph captures have explicit slots, retain the conservative whole env.
        let has_control_flow = graph
            .nodes
            .iter()
            .any(|node| matches!(node, Node::If(_) | Node::Loop(_) | Node::Scan(_)));
        if !has_control_flow {
            let graph_outputs = graph
                .outputs
                .iter()
                .map(|output| output.name.as_str())
                .collect::<HashSet<_>>();
            let mut last_use = HashMap::<SlotId, usize>::new();
            for (node_index, node) in graph.nodes.iter().enumerate() {
                for input in node.inputs() {
                    if let Some(id) = slot_ids.get(&input.name) {
                        last_use.insert(*id, node_index);
                    }
                }
            }
            for (id, node_index) in last_use {
                let name = &slot_names[id.0];
                if !graph_outputs.contains(name.as_str())
                    && !graph.nodes[node_index]
                        .outputs()
                        .iter()
                        .any(|output| output.name == *name)
                {
                    nodes[node_index].dead_slots.push(id);
                }
            }
        }

        Self { slot_names, nodes }
    }
}

fn collect_outputs(graph: &OnnxGraph, env: &Env) -> Result<Env> {
    graph
        .outputs
        .iter()
        .map(|output| {
            let value = env
                .get(&output.name)
                .cloned()
                .ok_or_else(|| TynxError::MissingValue(output.name.clone()))?;
            Ok((output.name.clone(), value))
        })
        .collect()
}

fn run_graph_prepared(
    graph: &OnnxGraph,
    env: &mut Env,
    device: &Device,
    initializers: &HashMap<InitializerId, Value>,
    plan: &ExecutionPlan,
) -> Result<()> {
    for (node_index, node) in graph.nodes.iter().enumerate() {
        let node_plan = &plan.nodes[node_index];
        for (input_index, id) in &node_plan.unnamed_initializers {
            if let Some(value) = initializers.get(id) {
                env.insert(
                    env_key(&InitializerId::Unnamed {
                        node_index,
                        input_index: *input_index,
                    }),
                    value.clone(),
                );
            }
        }

        execute_and_insert(node, env, device)?;

        for (input_index, _) in &node_plan.unnamed_initializers {
            env.remove(&env_key(&InitializerId::Unnamed {
                node_index,
                input_index: *input_index,
            }));
        }
        for slot in &node_plan.dead_slots {
            env.remove(&plan.slot_names[slot.0]);
        }
    }
    Ok(())
}

pub(crate) fn run_graph(graph: &OnnxGraph, env: &mut Env, device: &Device) -> Result<()> {
    for node in &graph.nodes {
        execute_and_insert(node, env, device)?;
    }
    Ok(())
}

fn execute_and_insert(node: &Node, env: &mut Env, device: &Device) -> Result<()> {
    let values = execute(node, env, device)?;
    if values.len() != node.outputs().len() {
        return Err(TynxError::Shape(format!(
            "node '{}' returned {} values for {} outputs",
            node.name(),
            values.len(),
            node.outputs().len()
        )));
    }

    for (output, value) in node.outputs().iter().zip(values) {
        env.insert(output.name.clone(), value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use onnx_ir::{DType, Node, node::identity::IdentityNodeBuilder};

    use crate::{Scalar, TynxError, Value};

    use super::*;

    #[test]
    fn reports_invalid_model_bytes() {
        let error = Session::from_bytes(b"not an ONNX model").unwrap_err();

        assert!(matches!(error, TynxError::Parse(_)));
    }

    #[test]
    fn runs_an_identity_graph() {
        let identity = IdentityNodeBuilder::new("identity")
            .input_scalar("x", DType::I64)
            .output_scalar("y", DType::I64)
            .build();
        let mut graph = OnnxGraph::default();
        graph.inputs = identity.inputs.clone();
        graph.outputs = identity.outputs.clone();
        graph.nodes.push(Node::Identity(identity));
        let session = Session {
            graph: Arc::new(graph),
        };
        let mut inputs = Env::new();
        inputs.insert("x".to_string(), Value::Scalar(Scalar::I64(42)));

        let outputs = session.run(&Device::default(), inputs).unwrap();

        assert!(matches!(
            outputs.get("y"),
            Some(Value::Scalar(Scalar::I64(42)))
        ));
    }

    #[test]
    fn prepares_and_reuses_an_embedded_value() {
        let mut identity = IdentityNodeBuilder::new("identity")
            .input_scalar("unused", DType::I64)
            .output_scalar("y", DType::I64)
            .build();
        identity.inputs[0] = Argument::from_const_i64("constant", 42);
        let mut graph = OnnxGraph::default();
        graph.outputs = identity.outputs.clone();
        graph.nodes.push(Node::Identity(identity));
        let session = Session {
            graph: Arc::new(graph),
        };
        let prepared = session.prepare(&Device::default()).unwrap();

        assert_eq!(prepared.initializer_count(), 1);
        for _ in 0..2 {
            let outputs = prepared.run(Env::new()).unwrap();
            assert!(matches!(
                outputs.get("y"),
                Some(Value::Scalar(Scalar::I64(42)))
            ));
        }
    }

    #[test]
    fn prepared_session_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<PreparedSession>();
    }

    #[test]
    fn prepared_session_can_run_concurrently() {
        let mut identity = IdentityNodeBuilder::new("identity")
            .input_scalar("unused", DType::I64)
            .output_scalar("y", DType::I64)
            .build();
        identity.inputs[0] = Argument::from_const_i64("constant", 42);
        let mut graph = OnnxGraph::default();
        graph.outputs = identity.outputs.clone();
        graph.nodes.push(Node::Identity(identity));
        let session = Session {
            graph: Arc::new(graph),
        };
        let prepared = Arc::new(session.prepare(&Device::default()).unwrap());

        let threads = (0..2)
            .map(|_| {
                let prepared = prepared.clone();
                std::thread::spawn(move || prepared.run(Env::new()).unwrap())
            })
            .collect::<Vec<_>>();
        for thread in threads {
            let outputs = thread.join().unwrap();
            assert!(matches!(
                outputs.get("y"),
                Some(Value::Scalar(Scalar::I64(42)))
            ));
        }
    }

    #[test]
    #[cfg(feature = "training")]
    fn prepared_session_rejects_an_input_with_different_capabilities() {
        let identity = IdentityNodeBuilder::new("identity")
            .input_tensor("x", 1, DType::F32)
            .output_tensor("y", 1, DType::F32)
            .build();
        let mut graph = OnnxGraph::default();
        graph.inputs = identity.inputs.clone();
        graph.outputs = identity.outputs.clone();
        graph.nodes.push(Node::Identity(identity));
        let session = Session {
            graph: Arc::new(graph),
        };
        let prepared = session.prepare(&Device::default()).unwrap();
        let mut inputs = Env::new();
        inputs.insert(
            "x".to_string(),
            Value::from_tensor_data(
                burn::tensor::TensorData::new(vec![1.0_f32], [1]),
                1,
                &Device::default().autodiff(),
            )
            .unwrap(),
        );

        assert!(matches!(
            prepared.run(inputs),
            Err(TynxError::DeviceMismatch { .. })
        ));
    }

    #[test]
    fn execution_plan_records_last_uses_as_slots() {
        let first = IdentityNodeBuilder::new("first")
            .input_scalar("x", DType::I64)
            .output_scalar("middle", DType::I64)
            .build();
        let second = IdentityNodeBuilder::new("second")
            .input_scalar("middle", DType::I64)
            .output_scalar("y", DType::I64)
            .build();
        let mut graph = OnnxGraph::default();
        graph.inputs = first.inputs.clone();
        graph.outputs = second.outputs.clone();
        graph.nodes.push(Node::Identity(first));
        graph.nodes.push(Node::Identity(second));

        let plan = ExecutionPlan::new(&graph);
        let dead_after_first = plan.nodes[0]
            .dead_slots
            .iter()
            .map(|slot| plan.slot_names[slot.0].as_str())
            .collect::<Vec<_>>();
        let dead_after_second = plan.nodes[1]
            .dead_slots
            .iter()
            .map(|slot| plan.slot_names[slot.0].as_str())
            .collect::<Vec<_>>();

        assert_eq!(dead_after_first, ["x"]);
        assert_eq!(dead_after_second, ["middle"]);
    }
}
