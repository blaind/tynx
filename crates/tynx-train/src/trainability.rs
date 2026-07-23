//! Conservative ONNX initializer-role classification.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    fmt,
};

pub use tynx_core::InitializerId;
use tynx_core::onnx_ir::{
    DType, Node,
    ir::{ArgType, Argument, OnnxGraph, ValueSource},
};
use tynx_core::{Result, TynxError};

use crate::backward_support::{BackwardCapability, BackwardSupportRegistry};

/// Semantic state role assigned to an ONNX initializer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum InitializerRole {
    /// Optimizer-visible trainable state.
    Parameter,
    /// Persistent model state that is not optimizer-visible.
    Buffer,
    /// Fixed graph data.
    Constant,
    /// Conflicting consumer roles require an explicit override.
    Ambiguous,
}

impl fmt::Display for InitializerRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parameter => formatter.write_str("parameter"),
            Self::Buffer => formatter.write_str("buffer"),
            Self::Constant => formatter.write_str("constant"),
            Self::Ambiguous => formatter.write_str("ambiguous"),
        }
    }
}

/// One consumer role that contributed to an initializer's classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitializerUse {
    node_name: String,
    operator: &'static str,
    input_index: usize,
    proposed_role: InitializerRole,
    reason: &'static str,
}

impl InitializerUse {
    /// Return the consuming node name.
    pub fn node_name(&self) -> &str {
        &self.node_name
    }

    /// Return the consuming operator name.
    pub fn operator(&self) -> &str {
        self.operator
    }

    /// Return the input position at that node.
    pub fn input_index(&self) -> usize {
        self.input_index
    }

    /// Return the role proposed by this consumer position.
    pub fn proposed_role(&self) -> InitializerRole {
        self.proposed_role
    }

    /// Return the semantic reason for that proposal.
    pub fn reason(&self) -> &str {
        self.reason
    }
}

/// Classification and metadata for one unique initializer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitializerReport {
    id: InitializerId,
    name: String,
    synthetic_name: bool,
    role: InitializerRole,
    dtype: Option<DType>,
    shape: Option<Vec<usize>>,
    uses: Vec<InitializerUse>,
}

impl InitializerReport {
    /// Return the processed-graph initializer identity.
    pub fn id(&self) -> &InitializerId {
        &self.id
    }

    /// Return the checkpoint/report name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return whether the name is a generated fallback rather than preserved ONNX provenance.
    pub fn has_synthetic_name(&self) -> bool {
        self.synthetic_name
    }

    /// Return the classified role.
    pub fn role(&self) -> InitializerRole {
        self.role
    }

    /// Return the element type when this initializer has tensor/scalar tensor data.
    pub fn dtype(&self) -> Option<DType> {
        self.dtype
    }

    /// Return the statically known shape, when available.
    pub fn shape(&self) -> Option<&[usize]> {
        self.shape.as_deref()
    }

    /// Return every consumer use that contributed to classification.
    pub fn uses(&self) -> &[InitializerUse] {
        &self.uses
    }
}

/// Exact-name role overrides applied after automatic consumer-role analysis.
#[derive(Debug, Clone, Default)]
pub struct TrainabilityOverrides {
    roles: HashMap<String, InitializerRole>,
}

impl TrainabilityOverrides {
    /// Create an empty override set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Assign an explicit non-ambiguous role to one report name.
    pub fn set_role(&mut self, name: impl Into<String>, role: InitializerRole) -> Result<()> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(TynxError::TypeMismatch(
                "initializer override name cannot be empty".to_string(),
            ));
        }
        if role == InitializerRole::Ambiguous {
            return Err(TynxError::TypeMismatch(
                "initializer override must choose parameter, buffer, or constant".to_string(),
            ));
        }
        self.roles.insert(name, role);
        Ok(())
    }
}

/// One blocked edge on a selected parameter-to-output backward slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackwardPathIssue {
    output: String,
    node_name: String,
    operator: String,
    input_index: usize,
    capability: BackwardCapability,
    parameters: Vec<String>,
}

impl BackwardPathIssue {
    /// Return the requested model output whose backward slice contains this edge.
    pub fn output(&self) -> &str {
        &self.output
    }

    /// Return the node name.
    pub fn node_name(&self) -> &str {
        &self.node_name
    }

    /// Return the operator name.
    pub fn operator(&self) -> &str {
        &self.operator
    }

    /// Return the blocked input position.
    pub fn input_index(&self) -> usize {
        self.input_index
    }

    /// Return whether the edge is deliberately stopped or lacks backward support.
    pub fn capability(&self) -> BackwardCapability {
        self.capability
    }

    /// Return trainable parameters upstream of this blocked edge.
    pub fn parameters(&self) -> &[String] {
        &self.parameters
    }
}

/// Structured imported-model role and output-specific trainability report.
#[derive(Debug, Clone, Default)]
pub struct TrainabilityReport {
    initializers: Vec<InitializerReport>,
    role_errors: bool,
    output_analysis_complete: bool,
    selected_outputs: Vec<String>,
    output_parameters: BTreeMap<String, Vec<String>>,
    backward_issues: Vec<BackwardPathIssue>,
    unused_parameters: Vec<String>,
    warnings: Vec<String>,
    errors: Vec<String>,
}

impl TrainabilityReport {
    /// Classify every initializer consumed by the processed graph.
    pub fn analyze_initializers(graph: &OnnxGraph) -> Self {
        Self::analyze_initializers_with(graph, &TrainabilityOverrides::new())
    }

    /// Classify every initializer and apply exact-name role overrides.
    pub fn analyze_initializers_with(graph: &OnnxGraph, overrides: &TrainabilityOverrides) -> Self {
        Self::analyze_initializers_with_names(graph, overrides, &HashMap::new())
    }

    /// Classify initializers using stable ONNX provenance captured before graph processing.
    pub fn analyze_initializers_with_names(
        graph: &OnnxGraph,
        overrides: &TrainabilityOverrides,
        stable_names: &HashMap<InitializerId, String>,
    ) -> Self {
        analyze(graph, overrides, stable_names)
    }

    /// Analyze every declared graph output using automatic initializer roles.
    pub fn analyze_all_outputs(graph: &OnnxGraph) -> Self {
        Self::analyze_all_outputs_with(graph, &TrainabilityOverrides::new())
    }

    /// Analyze every declared graph output using explicit initializer-role overrides.
    pub fn analyze_all_outputs_with(graph: &OnnxGraph, overrides: &TrainabilityOverrides) -> Self {
        Self::analyze_all_outputs_with_names(graph, overrides, &HashMap::new())
    }

    /// Analyze every output using stable ONNX initializer provenance.
    pub fn analyze_all_outputs_with_names(
        graph: &OnnxGraph,
        overrides: &TrainabilityOverrides,
        stable_names: &HashMap<InitializerId, String>,
    ) -> Self {
        let outputs = graph
            .outputs
            .iter()
            .map(|output| output.name.as_str())
            .collect::<Vec<_>>();
        Self::analyze_outputs_with_names(graph, &outputs, overrides, stable_names)
    }

    /// Analyze only the named declared graph outputs using automatic initializer roles.
    pub fn analyze_outputs(graph: &OnnxGraph, outputs: &[&str]) -> Self {
        Self::analyze_outputs_with(graph, outputs, &TrainabilityOverrides::new())
    }

    /// Analyze only the named declared graph outputs using explicit initializer-role overrides.
    pub fn analyze_outputs_with(
        graph: &OnnxGraph,
        outputs: &[&str],
        overrides: &TrainabilityOverrides,
    ) -> Self {
        Self::analyze_outputs_with_names(graph, outputs, overrides, &HashMap::new())
    }

    /// Analyze selected outputs using stable ONNX initializer provenance.
    pub fn analyze_outputs_with_names(
        graph: &OnnxGraph,
        outputs: &[&str],
        overrides: &TrainabilityOverrides,
        stable_names: &HashMap<InitializerId, String>,
    ) -> Self {
        let mut report = analyze(graph, overrides, stable_names);
        analyze_backward_paths(graph, outputs, &mut report);
        report
    }

    pub(crate) fn remap_outputs(&mut self, internal_to_public: &HashMap<String, String>) {
        let public_name = |name: &str| {
            internal_to_public
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.to_string())
        };

        for output in &mut self.selected_outputs {
            *output = public_name(output);
        }
        self.output_parameters = std::mem::take(&mut self.output_parameters)
            .into_iter()
            .map(|(output, parameters)| (public_name(&output), parameters))
            .collect();
        for issue in &mut self.backward_issues {
            issue.output = public_name(&issue.output);
        }
        for message in self.warnings.iter_mut().chain(self.errors.iter_mut()) {
            for (internal, public) in internal_to_public {
                if internal != public {
                    *message = message.replace(
                        &format!("requested output '{internal}'"),
                        &format!("requested output '{public}'"),
                    );
                }
            }
        }
    }

    /// Return initializer reports in first-consumer graph order.
    pub fn initializers(&self) -> &[InitializerReport] {
        &self.initializers
    }

    /// Iterate automatically or explicitly selected trainable parameters.
    pub fn trainable_parameters(&self) -> impl Iterator<Item = &InitializerReport> {
        self.initializers
            .iter()
            .filter(|initializer| initializer.role == InitializerRole::Parameter)
    }

    /// Iterate persistent non-trainable buffers.
    pub fn buffers(&self) -> impl Iterator<Item = &InitializerReport> {
        self.initializers
            .iter()
            .filter(|initializer| initializer.role == InitializerRole::Buffer)
    }

    /// Iterate fixed constants.
    pub fn constants(&self) -> impl Iterator<Item = &InitializerReport> {
        self.initializers
            .iter()
            .filter(|initializer| initializer.role == InitializerRole::Constant)
    }

    /// Return whether output-specific backward analysis has run.
    pub fn has_output_analysis(&self) -> bool {
        self.output_analysis_complete
    }

    /// Return requested outputs in caller order, with duplicates removed.
    pub fn selected_outputs(&self) -> &[String] {
        &self.selected_outputs
    }

    /// Return trainable parameters that structurally influence one requested output.
    pub fn parameters_for_output(&self, output: &str) -> Option<&[String]> {
        self.output_parameters.get(output).map(Vec::as_slice)
    }

    /// Return blocked edges found only on requested parameter-to-output slices.
    pub fn backward_issues(&self) -> &[BackwardPathIssue] {
        &self.backward_issues
    }

    /// Return selected trainable parameters unused by every requested output.
    pub fn unused_parameters(&self) -> &[String] {
        &self.unused_parameters
    }

    /// Return visible conservative/ambiguity/provenance warnings.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Return errors that prevent a valid role or requested-output training contract.
    pub fn errors(&self) -> &[String] {
        &self.errors
    }

    /// Return whether role classification is unambiguous and compatible with v1 f32 training.
    pub fn roles_ready(&self) -> bool {
        !self.role_errors
            && self
                .initializers
                .iter()
                .all(|initializer| initializer.role != InitializerRole::Ambiguous)
    }

    /// Return whether requested outputs have fully supported paths to their influencing parameters.
    pub fn is_trainable(&self) -> bool {
        self.output_analysis_complete
            && self.roles_ready()
            && self.errors.is_empty()
            && self.backward_issues.is_empty()
    }

    /// Reject an incomplete or unsupported requested-output training contract.
    pub fn require_trainable(&self) -> Result<()> {
        if !self.output_analysis_complete {
            return Err(TynxError::TypeMismatch(
                "output-specific trainability analysis has not run".to_string(),
            ));
        }
        if let Some(error) = self.errors.first() {
            return Err(TynxError::TypeMismatch(format!(
                "model is not trainable: {error}"
            )));
        }
        if self
            .initializers
            .iter()
            .any(|initializer| initializer.role == InitializerRole::Ambiguous)
        {
            return Err(TynxError::TypeMismatch(
                "model is not trainable: one or more initializer roles are ambiguous".to_string(),
            ));
        }
        if let Some(issue) = self.backward_issues.first() {
            return Err(TynxError::UnsupportedOp(format!(
                "{} '{}' input {} blocks gradients to output '{}': {}",
                issue.operator,
                issue.node_name,
                issue.input_index,
                issue.output,
                issue
                    .capability
                    .reason()
                    .unwrap_or("unknown backward restriction")
            )));
        }
        Ok(())
    }
}

impl fmt::Display for TrainabilityReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_role_section(
            formatter,
            "Trainable parameters",
            self.trainable_parameters(),
        )?;
        write_role_section(formatter, "Fixed buffers", self.buffers())?;
        write_role_section(formatter, "Constants", self.constants())?;
        write_role_section(
            formatter,
            "Ambiguous initializers",
            self.initializers
                .iter()
                .filter(|initializer| initializer.role == InitializerRole::Ambiguous),
        )?;
        if self.output_analysis_complete {
            writeln!(formatter, "Requested outputs:")?;
            for output in &self.selected_outputs {
                let parameters = self
                    .parameters_for_output(output)
                    .unwrap_or_default()
                    .join(", ");
                writeln!(formatter, "  {output}: [{parameters}]")?;
            }
            writeln!(formatter, "Blocked backward paths:")?;
            for issue in &self.backward_issues {
                writeln!(
                    formatter,
                    "  {}:{} input {} -> {} ({}) [{}]",
                    issue.operator,
                    issue.node_name,
                    issue.input_index,
                    issue.output,
                    issue.capability.reason().unwrap_or("unknown restriction"),
                    issue.parameters.join(", ")
                )?;
            }
            writeln!(formatter, "Unused parameters:")?;
            for parameter in &self.unused_parameters {
                writeln!(formatter, "  {parameter}")?;
            }
        }
        write_message_section(formatter, "Warnings", &self.warnings)?;
        write_message_section(formatter, "Errors", &self.errors)
    }
}

#[derive(Debug)]
struct WorkEntry {
    id: InitializerId,
    name: String,
    synthetic_name: bool,
    dtype: Option<DType>,
    shape: Option<Vec<usize>>,
    uses: Vec<(InitializerUse, bool)>,
}

#[derive(Debug, Clone, Copy)]
struct RoleProposal {
    role: InitializerRole,
    reason: &'static str,
    recognized: bool,
}

fn analyze(
    graph: &OnnxGraph,
    overrides: &TrainabilityOverrides,
    stable_names: &HashMap<InitializerId, String>,
) -> TrainabilityReport {
    let mut entries = Vec::<WorkEntry>::new();
    let mut by_id = HashMap::<InitializerId, usize>::new();

    for (node_index, node) in graph.nodes.iter().enumerate() {
        for (input_index, input) in node.inputs().iter().enumerate() {
            if !matches!(
                input.value_source,
                ValueSource::Static(_) | ValueSource::Constant
            ) {
                continue;
            }
            let id = initializer_id(input, node_index, input_index);
            let (candidate_name, synthetic_name) = initializer_name(input, &id, stable_names);
            let entry_index = match by_id.get(&id).copied() {
                Some(index) => index,
                None => {
                    let index = entries.len();
                    entries.push(WorkEntry {
                        id: id.clone(),
                        name: candidate_name.clone(),
                        synthetic_name,
                        dtype: argument_dtype(input),
                        shape: argument_shape(input),
                        uses: Vec::new(),
                    });
                    by_id.insert(id.clone(), index);
                    index
                }
            };
            let entry = &mut entries[entry_index];
            if entry.synthetic_name && !synthetic_name {
                entry.name = candidate_name;
                entry.synthetic_name = false;
            }
            let proposal = role_proposal(node, input_index);
            entry.uses.push((
                InitializerUse {
                    node_name: node.name().to_string(),
                    operator: operator_name(node),
                    input_index,
                    proposed_role: proposal.role,
                    reason: proposal.reason,
                },
                proposal.recognized,
            ));
        }
    }

    let mut report = TrainabilityReport::default();
    let mut used_overrides = HashSet::new();
    for entry in entries {
        let roles = entry
            .uses
            .iter()
            .map(|(usage, _)| usage.proposed_role)
            .collect::<BTreeSet<_>>();
        let automatic_role = match roles.iter().copied().next() {
            Some(role) if roles.len() == 1 => role,
            Some(_) => InitializerRole::Ambiguous,
            None => InitializerRole::Constant,
        };
        let role = match overrides.roles.get(&entry.name).copied() {
            Some(role) => {
                used_overrides.insert(entry.name.clone());
                if role != automatic_role {
                    report.warnings.push(format!(
                        "initializer '{}' automatic role {automatic_role} was overridden as {role}",
                        entry.name
                    ));
                }
                role
            }
            None => automatic_role,
        };

        if entry.synthetic_name {
            report.warnings.push(format!(
                "initializer '{}' has no preserved ONNX name; checkpoint identity needs a stable provenance mapping",
                entry.name
            ));
        }
        if automatic_role == InitializerRole::Ambiguous && role == InitializerRole::Ambiguous {
            let uses = entry
                .uses
                .iter()
                .map(|(usage, _)| {
                    format!(
                        "{}:{} input {} -> {}",
                        usage.operator, usage.node_name, usage.input_index, usage.proposed_role
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            report.warnings.push(format!(
                "initializer '{}' has conflicting consumer roles ({uses}); add an explicit override",
                entry.name
            ));
        }
        if !overrides.roles.contains_key(&entry.name) {
            for (usage, recognized) in &entry.uses {
                if !recognized {
                    report.warnings.push(format!(
                        "initializer '{}' remains constant because {} input {} has no registered trainable role",
                        entry.name, usage.operator, usage.input_index
                    ));
                }
            }
        }
        if role == InitializerRole::Parameter && entry.dtype != Some(DType::F32) {
            report.role_errors = true;
            report.errors.push(format!(
                "initializer '{}' is a trainable parameter with dtype {:?}; v1 training requires f32",
                entry.name, entry.dtype
            ));
        }

        report.initializers.push(InitializerReport {
            id: entry.id,
            name: entry.name,
            synthetic_name: entry.synthetic_name,
            role,
            dtype: entry.dtype,
            shape: entry.shape,
            uses: entry.uses.into_iter().map(|(usage, _)| usage).collect(),
        });
    }

    for name in overrides.roles.keys() {
        if !used_overrides.contains(name) {
            report.warnings.push(format!(
                "initializer override '{name}' did not match any processed graph initializer"
            ));
        }
    }
    report
}

fn analyze_backward_paths(graph: &OnnxGraph, outputs: &[&str], report: &mut TrainabilityReport) {
    report.output_analysis_complete = true;

    let declared_outputs = graph
        .outputs
        .iter()
        .map(|output| output.name.as_str())
        .collect::<HashSet<_>>();
    let mut selected = HashSet::new();
    for output in outputs {
        if output.is_empty() {
            report
                .errors
                .push("requested output name cannot be empty".to_string());
        } else if !declared_outputs.contains(output) {
            report.errors.push(format!(
                "requested output '{output}' is not a declared graph output"
            ));
        } else if selected.insert((*output).to_string()) {
            report.selected_outputs.push((*output).to_string());
        }
    }
    if outputs.is_empty() {
        report.errors.push(
            "at least one graph output must be selected for trainability analysis".to_string(),
        );
    }

    let parameter_names = report
        .trainable_parameters()
        .map(|parameter| (parameter.id.clone(), parameter.name.clone()))
        .collect::<HashMap<_, _>>();
    let all_parameters = parameter_names.values().cloned().collect::<BTreeSet<_>>();

    let mut value_parameters = HashMap::<String, BTreeSet<String>>::new();
    for (node_index, node) in graph.nodes.iter().enumerate() {
        let mut parameters = BTreeSet::new();
        for (input_index, input) in node.inputs().iter().enumerate() {
            parameters.extend(parameters_for_argument(
                node_index,
                input_index,
                input,
                &parameter_names,
                &value_parameters,
            ));
        }
        for output in node.outputs() {
            if !output.name.is_empty() {
                value_parameters.insert(output.name.clone(), parameters.clone());
            }
        }
    }

    let producers = graph
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(node_index, node)| {
            node.outputs()
                .iter()
                .enumerate()
                .filter(|(_, output)| !output.name.is_empty())
                .map(move |(output_index, output)| {
                    (output.name.clone(), (node_index, output_index))
                })
        })
        .collect::<HashMap<_, _>>();

    let mut used_parameters = BTreeSet::new();
    for output in report.selected_outputs.clone() {
        let influencing = value_parameters.get(&output).cloned().unwrap_or_default();
        used_parameters.extend(influencing.iter().cloned());
        report
            .output_parameters
            .insert(output.clone(), influencing.iter().cloned().collect());
        if influencing.is_empty() {
            report.warnings.push(format!(
                "requested output '{output}' is not influenced by any selected trainable parameter"
            ));
            continue;
        }

        let mut queue = VecDeque::from([output.clone()]);
        let mut visited_values = HashSet::new();
        while let Some(value_name) = queue.pop_front() {
            if !visited_values.insert(value_name.clone()) {
                continue;
            }
            let Some(&(node_index, output_index)) = producers.get(&value_name) else {
                continue;
            };
            let node = &graph.nodes[node_index];
            for (input_index, input) in node.inputs().iter().enumerate() {
                let upstream = parameters_for_argument(
                    node_index,
                    input_index,
                    input,
                    &parameter_names,
                    &value_parameters,
                )
                .intersection(&influencing)
                .cloned()
                .collect::<BTreeSet<_>>();
                if upstream.is_empty() {
                    continue;
                }

                let capability =
                    BackwardSupportRegistry::input_capability(node, output_index, input_index);
                if capability.is_differentiable() {
                    if matches!(input.value_source, ValueSource::Dynamic) && !input.name.is_empty()
                    {
                        queue.push_back(input.name.clone());
                    }
                } else {
                    let issue = BackwardPathIssue {
                        output: output.clone(),
                        node_name: node.name().to_string(),
                        operator: node_kind_name(node),
                        input_index,
                        capability,
                        parameters: upstream.into_iter().collect(),
                    };
                    if !report.backward_issues.contains(&issue) {
                        report.backward_issues.push(issue);
                    }
                }
            }
        }
    }

    report.unused_parameters = all_parameters
        .difference(&used_parameters)
        .cloned()
        .collect();
    for parameter in &report.unused_parameters {
        report.warnings.push(format!(
            "selected trainable parameter '{parameter}' is unused by the requested outputs"
        ));
    }
}

fn parameters_for_argument(
    node_index: usize,
    input_index: usize,
    input: &Argument,
    parameter_names: &HashMap<InitializerId, String>,
    value_parameters: &HashMap<String, BTreeSet<String>>,
) -> BTreeSet<String> {
    if matches!(
        input.value_source,
        ValueSource::Static(_) | ValueSource::Constant
    ) {
        let id = initializer_id(input, node_index, input_index);
        if let Some(parameter) = parameter_names.get(&id) {
            return BTreeSet::from([parameter.clone()]);
        }
    }
    value_parameters
        .get(&input.name)
        .cloned()
        .unwrap_or_default()
}

fn initializer_id(input: &Argument, node_index: usize, input_index: usize) -> InitializerId {
    InitializerId::from_argument(input, node_index, input_index)
        .unwrap_or_else(|| unreachable!("initializer_id is called only for embedded inputs"))
}

fn initializer_name(
    input: &Argument,
    id: &InitializerId,
    stable_names: &HashMap<InitializerId, String>,
) -> (String, bool) {
    if let Some(name) = stable_names.get(id) {
        return (name.clone(), false);
    }
    if !input.name.is_empty() {
        return (input.name.clone(), false);
    }
    match id {
        InitializerId::Named(name) => (name.clone(), false),
        InitializerId::Static(data_id) => (format!("__static#{data_id}"), true),
        InitializerId::Unnamed {
            node_index,
            input_index,
        } => (format!("__unnamed#{node_index}:{input_index}"), true),
    }
}

fn argument_dtype(argument: &Argument) -> Option<DType> {
    match &argument.ty {
        ArgType::ScalarTensor(dtype) | ArgType::ScalarNative(dtype) => Some(*dtype),
        ArgType::Tensor(tensor) => Some(tensor.dtype),
        ArgType::Shape(_) => None,
    }
}

fn argument_shape(argument: &Argument) -> Option<Vec<usize>> {
    argument
        .ty
        .static_shape_known()
        .or_else(|| argument.value().map(|value| value.shape.to_vec()))
}

fn role_proposal(node: &Node, input_index: usize) -> RoleProposal {
    let parameter = RoleProposal {
        role: InitializerRole::Parameter,
        reason: "trainable weight, bias, or affine term",
        recognized: true,
    };
    let buffer = RoleProposal {
        role: InitializerRole::Buffer,
        reason: "persistent normalization statistic",
        recognized: true,
    };
    let constant = RoleProposal {
        role: InitializerRole::Constant,
        reason: "semantic metadata or fixed operator input",
        recognized: true,
    };

    match node {
        Node::Linear(_) | Node::Gemm(_) if matches!(input_index, 1 | 2) => parameter,
        Node::MatMul(_) if matches!(input_index, 0 | 1) => parameter,
        Node::Conv1d(_)
        | Node::Conv2d(_)
        | Node::Conv3d(_)
        | Node::ConvTranspose1d(_)
        | Node::ConvTranspose2d(_)
        | Node::ConvTranspose3d(_)
            if matches!(input_index, 1 | 2) =>
        {
            parameter
        }
        Node::BatchNormalization(_) if matches!(input_index, 1 | 2) => parameter,
        Node::BatchNormalization(_) if matches!(input_index, 3 | 4) => buffer,
        Node::InstanceNormalization(_)
        | Node::LayerNormalization(_)
        | Node::GroupNormalization(_)
            if matches!(input_index, 1 | 2) =>
        {
            parameter
        }
        Node::PRelu(_) if input_index == 1 => parameter,
        Node::Gather(_) | Node::GatherElements(_) | Node::GatherND(_) if input_index == 0 => {
            parameter
        }
        Node::Reshape(_)
        | Node::Expand(_)
        | Node::Tile(_)
        | Node::Squeeze(_)
        | Node::Unsqueeze(_)
            if input_index == 1 =>
        {
            constant
        }
        Node::Gather(_) | Node::GatherElements(_) | Node::GatherND(_) if input_index == 1 => {
            constant
        }
        Node::Slice(_) | Node::Resize(_) | Node::Pad(_) if input_index >= 1 => constant,
        Node::Clip(_) if matches!(input_index, 1 | 2) => constant,
        Node::OneHot(_) if matches!(input_index, 1 | 2) => constant,
        Node::TopK(_) | Node::CumSum(_) | Node::Trilu(_) if input_index == 1 => constant,
        Node::Range(_) | Node::ConstantOfShape(_) => constant,
        _ => RoleProposal {
            role: InitializerRole::Constant,
            reason: "no trainable initializer role is registered for this consumer input",
            recognized: false,
        },
    }
}

fn operator_name(node: &Node) -> &'static str {
    match node {
        Node::Linear(_) => "Linear",
        Node::Gemm(_) => "Gemm",
        Node::MatMul(_) => "MatMul",
        Node::Conv1d(_) => "Conv1d",
        Node::Conv2d(_) => "Conv2d",
        Node::Conv3d(_) => "Conv3d",
        Node::ConvTranspose1d(_) => "ConvTranspose1d",
        Node::ConvTranspose2d(_) => "ConvTranspose2d",
        Node::ConvTranspose3d(_) => "ConvTranspose3d",
        Node::BatchNormalization(_) => "BatchNormalization",
        Node::InstanceNormalization(_) => "InstanceNormalization",
        Node::LayerNormalization(_) => "LayerNormalization",
        Node::GroupNormalization(_) => "GroupNormalization",
        Node::PRelu(_) => "PRelu",
        Node::Gather(_) => "Gather",
        Node::GatherElements(_) => "GatherElements",
        Node::GatherND(_) => "GatherND",
        Node::Reshape(_) => "Reshape",
        Node::Expand(_) => "Expand",
        Node::Tile(_) => "Tile",
        Node::Squeeze(_) => "Squeeze",
        Node::Unsqueeze(_) => "Unsqueeze",
        Node::Slice(_) => "Slice",
        Node::Resize(_) => "Resize",
        Node::Pad(_) => "Pad",
        Node::Clip(_) => "Clip",
        Node::OneHot(_) => "OneHot",
        Node::TopK(_) => "TopK",
        Node::CumSum(_) => "CumSum",
        Node::Trilu(_) => "Trilu",
        Node::Range(_) => "Range",
        Node::ConstantOfShape(_) => "ConstantOfShape",
        Node::Identity(_) => "Identity",
        _ => "Other",
    }
}

fn node_kind_name(node: &Node) -> String {
    let registered = operator_name(node);
    if registered != "Other" {
        return registered.to_string();
    }
    let debug = format!("{node:?}");
    debug
        .split(['(', '{'])
        .next()
        .unwrap_or("Unknown")
        .to_string()
}

fn write_role_section<'a>(
    formatter: &mut fmt::Formatter<'_>,
    heading: &str,
    initializers: impl Iterator<Item = &'a InitializerReport>,
) -> fmt::Result {
    writeln!(formatter, "{heading}:")?;
    for initializer in initializers {
        writeln!(
            formatter,
            "  {} {:?} {:?}",
            initializer.name, initializer.dtype, initializer.shape
        )?;
    }
    Ok(())
}

fn write_message_section(
    formatter: &mut fmt::Formatter<'_>,
    heading: &str,
    messages: &[String],
) -> fmt::Result {
    writeln!(formatter, "{heading}:")?;
    for message in messages {
        writeln!(formatter, "  {message}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use tynx_core::onnx_ir::{
        ir::{TensorType, ValueSource},
        node::{
            batch_norm::{BatchNormConfig, BatchNormRuntimeConfig, BatchNormalizationNode},
            ceil::CeilNode,
            clip::{ClipConfig, ClipInput, ClipNode},
            det::DetNode,
            gather::{GatherConfig, GatherNode},
            identity::IdentityNode,
            linear::{LinearConfig, LinearNode},
            relu::ReluNode,
        },
    };

    use super::*;

    fn dynamic(name: &str, dtype: DType, shape: &[usize]) -> Argument {
        Argument::new(
            name,
            ArgType::Tensor(TensorType::new_known(dtype, shape.to_vec())),
        )
    }

    fn static_tensor(data_id: usize, dtype: DType, shape: &[usize]) -> Argument {
        let mut argument = dynamic("", dtype, shape);
        argument.value_source = ValueSource::Static(data_id);
        argument
    }

    fn named_constant(name: &str, dtype: DType, shape: &[usize]) -> Argument {
        let mut argument = dynamic(name, dtype, shape);
        argument.value_source = ValueSource::Constant;
        argument
    }

    fn linear(weight: Argument, bias: Argument) -> Node {
        linear_named("linear", "x", "y", weight, bias)
    }

    fn linear_named(
        name: &str,
        input: &str,
        output: &str,
        weight: Argument,
        bias: Argument,
    ) -> Node {
        Node::Linear(LinearNode {
            name: name.to_string(),
            inputs: vec![dynamic(input, DType::F32, &[1, 2]), weight, bias],
            outputs: vec![dynamic(output, DType::F32, &[1, 2])],
            config: LinearConfig::new(false),
        })
    }

    fn batch_norm(scale: Argument, bias: Argument, mean: Argument, variance: Argument) -> Node {
        Node::BatchNormalization(BatchNormalizationNode {
            name: "batch_norm".to_string(),
            inputs: vec![
                dynamic("x", DType::F32, &[1, 2, 2]),
                scale,
                bias,
                mean,
                variance,
            ],
            outputs: vec![dynamic("y", DType::F32, &[1, 2, 2])],
            config: BatchNormConfig::Runtime(BatchNormRuntimeConfig::new(1.0e-5, 0.9)),
        })
    }

    fn graph(nodes: Vec<Node>) -> OnnxGraph {
        let mut graph = OnnxGraph::default();
        graph.nodes = nodes;
        graph
    }

    fn graph_with_outputs(nodes: Vec<Node>, outputs: &[&str]) -> OnnxGraph {
        let mut graph = graph(nodes);
        graph.outputs = outputs
            .iter()
            .map(|name| dynamic(name, DType::F32, &[1, 2]))
            .collect();
        graph
    }

    #[test]
    fn classifies_linear_weights_and_biases_as_parameters() {
        let graph = graph(vec![linear(
            named_constant("encoder.weight", DType::F32, &[2, 2]),
            named_constant("encoder.bias", DType::F32, &[2]),
        )]);

        let report = TrainabilityReport::analyze_initializers(&graph);

        assert!(report.roles_ready());
        assert_eq!(
            report
                .trainable_parameters()
                .map(InitializerReport::name)
                .collect::<Vec<_>>(),
            ["encoder.weight", "encoder.bias"]
        );
        assert!(report.buffers().next().is_none());
        assert!(report.errors().is_empty());
    }

    #[test]
    fn separates_batch_norm_affine_parameters_and_running_buffers() {
        let graph = graph(vec![batch_norm(
            named_constant("bn.scale", DType::F32, &[2]),
            named_constant("bn.bias", DType::F32, &[2]),
            named_constant("bn.mean", DType::F32, &[2]),
            named_constant("bn.variance", DType::F32, &[2]),
        )]);

        let report = TrainabilityReport::analyze_initializers(&graph);

        assert_eq!(
            report
                .trainable_parameters()
                .map(InitializerReport::name)
                .collect::<Vec<_>>(),
            ["bn.scale", "bn.bias"]
        );
        assert_eq!(
            report
                .buffers()
                .map(InitializerReport::name)
                .collect::<Vec<_>>(),
            ["bn.mean", "bn.variance"]
        );
    }

    #[test]
    fn reports_conflicting_shared_initializer_roles() {
        let shared = named_constant("shared", DType::F32, &[2]);
        let graph = graph(vec![
            linear(shared.clone(), named_constant("bias", DType::F32, &[2])),
            batch_norm(
                named_constant("scale", DType::F32, &[2]),
                named_constant("bn_bias", DType::F32, &[2]),
                shared,
                named_constant("variance", DType::F32, &[2]),
            ),
        ]);

        let report = TrainabilityReport::analyze_initializers(&graph);
        let shared = report
            .initializers()
            .iter()
            .find(|initializer| initializer.name() == "shared")
            .unwrap();

        assert_eq!(shared.role(), InitializerRole::Ambiguous);
        assert!(!report.roles_ready());
        assert!(
            report
                .warnings()
                .iter()
                .any(|warning| warning.contains("conflicting"))
        );
    }

    #[test]
    fn exact_override_resolves_ambiguity_and_unknown_overrides_warn() {
        let shared = named_constant("shared", DType::F32, &[2]);
        let graph = graph(vec![
            linear(shared.clone(), named_constant("bias", DType::F32, &[2])),
            batch_norm(
                named_constant("scale", DType::F32, &[2]),
                named_constant("bn_bias", DType::F32, &[2]),
                shared,
                named_constant("variance", DType::F32, &[2]),
            ),
        ]);
        let mut overrides = TrainabilityOverrides::new();
        overrides
            .set_role("shared", InitializerRole::Parameter)
            .unwrap();
        overrides
            .set_role("missing", InitializerRole::Constant)
            .unwrap();

        let report = TrainabilityReport::analyze_initializers_with(&graph, &overrides);

        assert!(report.roles_ready());
        assert_eq!(
            report
                .initializers()
                .iter()
                .find(|initializer| initializer.name() == "shared")
                .unwrap()
                .role(),
            InitializerRole::Parameter
        );
        assert!(
            report
                .warnings()
                .iter()
                .any(|warning| warning.contains("missing"))
        );
    }

    #[test]
    fn rejects_non_f32_trainable_parameters() {
        let graph = graph(vec![linear(
            named_constant("weight", DType::F16, &[2, 2]),
            named_constant("bias", DType::F32, &[2]),
        )]);

        let report = TrainabilityReport::analyze_initializers(&graph);

        assert!(!report.roles_ready());
        assert!(
            report
                .errors()
                .iter()
                .any(|error| error.contains("requires f32"))
        );
    }

    #[test]
    fn unknown_float_initializer_stays_constant_with_warning() {
        let graph = graph(vec![Node::Identity(IdentityNode {
            name: "identity".to_string(),
            inputs: vec![named_constant("table", DType::F32, &[2])],
            outputs: vec![dynamic("y", DType::F32, &[2])],
        })]);

        let report = TrainabilityReport::analyze_initializers(&graph);

        assert_eq!(report.constants().next().unwrap().name(), "table");
        assert!(
            report
                .warnings()
                .iter()
                .any(|warning| warning.contains("Identity"))
        );
    }

    #[test]
    fn lifted_static_ids_are_deduplicated_and_report_provenance_warning() {
        let weight = static_tensor(7, DType::F32, &[2, 2]);
        let graph = graph(vec![
            linear(weight.clone(), static_tensor(8, DType::F32, &[2])),
            linear(weight, static_tensor(9, DType::F32, &[2])),
        ]);

        let report = TrainabilityReport::analyze_initializers(&graph);

        assert_eq!(report.initializers().len(), 3);
        let weight = report
            .initializers()
            .iter()
            .find(|initializer| initializer.id() == &InitializerId::Static(7))
            .unwrap();
        assert_eq!(weight.uses().len(), 2);
        assert!(weight.has_synthetic_name());
        assert!(
            report
                .warnings()
                .iter()
                .any(|warning| warning.contains("provenance"))
        );
    }

    #[test]
    fn display_has_user_facing_sections() {
        let graph = graph(vec![linear(
            named_constant("weight", DType::F32, &[2, 2]),
            named_constant("bias", DType::F32, &[2]),
        )]);

        let display = TrainabilityReport::analyze_initializers(&graph).to_string();

        assert!(display.contains("Trainable parameters:"));
        assert!(display.contains("Fixed buffers:"));
        assert!(display.contains("Warnings:"));
        assert!(display.contains("weight"));
    }

    #[test]
    fn registry_distinguishes_gather_data_from_indices() {
        let node = Node::Gather(GatherNode {
            name: "gather".to_string(),
            inputs: vec![
                dynamic("data", DType::F32, &[4, 2]),
                dynamic("indices", DType::I64, &[1]),
            ],
            outputs: vec![dynamic("selected", DType::F32, &[1, 2])],
            config: GatherConfig::new(0),
        });

        assert_eq!(
            BackwardSupportRegistry::input_capability(&node, 0, 0),
            BackwardCapability::Differentiable
        );
        assert!(matches!(
            BackwardSupportRegistry::input_capability(&node, 0, 1),
            BackwardCapability::StopGradient(_)
        ));
    }

    #[test]
    fn registry_accepts_clip_data_but_not_runtime_bounds() {
        let node = Node::Clip(ClipNode {
            name: "relu6".to_string(),
            inputs: vec![
                dynamic("data", DType::F32, &[2, 2]),
                dynamic("minimum", DType::F32, &[1]),
                dynamic("maximum", DType::F32, &[1]),
            ],
            outputs: vec![dynamic("clipped", DType::F32, &[2, 2])],
            config: ClipConfig::new(Some(ClipInput::Static(0.0)), Some(ClipInput::Static(6.0))),
        });

        assert_eq!(
            BackwardSupportRegistry::input_capability(&node, 0, 0),
            BackwardCapability::Differentiable
        );
        assert!(matches!(
            BackwardSupportRegistry::input_capability(&node, 0, 1),
            BackwardCapability::Unsupported(_)
        ));
        assert!(matches!(
            BackwardSupportRegistry::input_capability(&node, 0, 2),
            BackwardCapability::Unsupported(_)
        ));
    }

    #[test]
    fn supported_slice_reports_influencing_and_unused_parameters() {
        let graph = two_branch_graph();

        let report = TrainabilityReport::analyze_outputs(&graph, &["policy"]);

        assert!(report.is_trainable());
        assert_eq!(
            report.parameters_for_output("policy").unwrap(),
            ["policy.bias", "policy.weight"]
        );
        assert!(report.backward_issues().is_empty());
        assert_eq!(
            report.unused_parameters(),
            ["diagnostic.bias", "diagnostic.weight"]
        );
        report.require_trainable().unwrap();
    }

    #[test]
    fn irrelevant_stop_gradient_branch_does_not_block_selected_output() {
        let graph = two_branch_graph();

        let policy = TrainabilityReport::analyze_outputs(&graph, &["policy"]);
        let diagnostic = TrainabilityReport::analyze_outputs(&graph, &["diagnostic"]);

        assert!(policy.is_trainable());
        assert!(!diagnostic.is_trainable());
        assert_eq!(diagnostic.backward_issues().len(), 1);
        let issue = &diagnostic.backward_issues()[0];
        assert_eq!(issue.operator(), "Ceil");
        assert_eq!(issue.node_name(), "diagnostic_round");
        assert!(matches!(
            issue.capability(),
            BackwardCapability::StopGradient(_)
        ));
        assert_eq!(issue.parameters(), ["diagnostic.bias", "diagnostic.weight"]);
        assert!(matches!(
            diagnostic.require_trainable(),
            Err(TynxError::UnsupportedOp(_))
        ));
    }

    #[test]
    fn unregistered_backward_operator_is_actionable() {
        let nodes = vec![
            linear_named(
                "matrix",
                "x",
                "matrix_value",
                named_constant("matrix.weight", DType::F32, &[2, 2]),
                named_constant("matrix.bias", DType::F32, &[2]),
            ),
            Node::Det(DetNode {
                name: "determinant".to_string(),
                inputs: vec![dynamic("matrix_value", DType::F32, &[1, 2])],
                outputs: vec![dynamic("score", DType::F32, &[1])],
            }),
        ];
        let graph = graph_with_outputs(nodes, &["score"]);

        let report = TrainabilityReport::analyze_all_outputs(&graph);

        assert!(!report.is_trainable());
        let issue = &report.backward_issues()[0];
        assert_eq!(issue.operator(), "Det");
        assert!(matches!(
            issue.capability(),
            BackwardCapability::Unsupported(_)
        ));
        assert!(report.to_string().contains("determinant"));
    }

    #[test]
    fn unknown_requested_output_fails_before_execution() {
        let graph = two_branch_graph();

        let report = TrainabilityReport::analyze_outputs(&graph, &["missing"]);

        assert!(!report.is_trainable());
        assert!(report.roles_ready());
        assert!(report.errors()[0].contains("not a declared graph output"));
        assert!(matches!(
            report.require_trainable(),
            Err(TynxError::TypeMismatch(_))
        ));
    }

    fn two_branch_graph() -> OnnxGraph {
        let nodes = vec![
            linear_named(
                "policy_linear",
                "x",
                "policy_hidden",
                named_constant("policy.weight", DType::F32, &[2, 2]),
                named_constant("policy.bias", DType::F32, &[2]),
            ),
            Node::Relu(ReluNode {
                name: "policy_relu".to_string(),
                inputs: vec![dynamic("policy_hidden", DType::F32, &[1, 2])],
                outputs: vec![dynamic("policy", DType::F32, &[1, 2])],
            }),
            linear_named(
                "diagnostic_linear",
                "x",
                "diagnostic_hidden",
                named_constant("diagnostic.weight", DType::F32, &[2, 2]),
                named_constant("diagnostic.bias", DType::F32, &[2]),
            ),
            Node::Ceil(CeilNode {
                name: "diagnostic_round".to_string(),
                inputs: vec![dynamic("diagnostic_hidden", DType::F32, &[1, 2])],
                outputs: vec![dynamic("diagnostic", DType::F32, &[1, 2])],
            }),
        ];
        graph_with_outputs(nodes, &["policy", "diagnostic"])
    }
}
