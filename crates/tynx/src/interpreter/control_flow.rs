//! ONNX If, Loop, and Scan execution.

use burn::tensor::{Device, Slice};
use onnx_ir::{
    ir::{Argument, OnnxGraph},
    node::{if_node::IfNode, loop_node::LoopNode, scan_node::ScanNode},
};

use super::{Env, resolve, shape};
use crate::{DynBool, DynInt, DynTensor, Result, Scalar, TynxError, Value, session::run_graph};

pub(super) fn if_node(node: &IfNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let condition = condition(resolve::first(env, &node.name, &node.inputs, device)?)?;
    let branch = if condition {
        &node.config.then_branch
    } else {
        &node.config.else_branch
    };
    let mut branch_env = env.clone();
    bind_scope(
        env,
        &mut branch_env,
        &node.inputs,
        1,
        &node.config.scope_ref_names,
        device,
    )?;
    run_graph(branch, &mut branch_env, device)?;
    graph_outputs(branch, &branch_env)
}

pub(super) fn loop_node(node: &LoopNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let body = &node.config.body;
    if body.inputs.len() < 2 {
        return Err(TynxError::Shape(
            "Loop body requires iteration and condition inputs".to_string(),
        ));
    }
    let carried_count = body.inputs.len() - 2;
    if body.outputs.len() < carried_count + 1 {
        return Err(TynxError::Shape(
            "Loop body has too few outputs for its carried values".to_string(),
        ));
    }
    let scan_count = body.outputs.len() - carried_count - 1;
    let max_trip = optional_i64(env, node.inputs.first(), 0, device)?;
    let mut keep_going = match node.inputs.get(1).filter(|input| !input.is_optional()) {
        Some(input) => condition(resolve::input_at(env, input, 1, device)?)?,
        None => true,
    };
    let mut carried = (0..carried_count)
        .map(|index| resolve::at(env, &node.name, &node.inputs, index + 2, device))
        .collect::<Result<Vec<_>>>()?;
    let mut scans = vec![Vec::new(); scan_count];
    let mut iteration = 0_i64;

    while keep_going && max_trip.is_none_or(|limit| iteration < limit) {
        if iteration >= 1_000_000 {
            return Err(TynxError::Shape(
                "Loop exceeded the runtime limit of 1000000 iterations".to_string(),
            ));
        }
        let mut body_env = env.clone();
        bind_scope(
            env,
            &mut body_env,
            &node.inputs,
            2 + carried_count,
            &node.config.scope_ref_names,
            device,
        )?;
        body_env.insert(
            body.inputs[0].name.clone(),
            Value::Scalar(Scalar::I64(iteration)),
        );
        body_env.insert(
            body.inputs[1].name.clone(),
            Value::Scalar(Scalar::Bool(keep_going)),
        );
        for (input, value) in body.inputs[2..].iter().zip(&carried) {
            body_env.insert(input.name.clone(), value.clone());
        }
        run_graph(body, &mut body_env, device)?;

        keep_going = condition(output(&body_env, &body.outputs[0])?)?;
        for (index, value) in carried.iter_mut().enumerate() {
            *value = output(&body_env, &body.outputs[index + 1])?;
        }
        for (index, values) in scans.iter_mut().enumerate() {
            values.push(output(&body_env, &body.outputs[1 + carried_count + index])?);
        }
        iteration += 1;
    }

    let mut values = carried;
    for scan in scans {
        values.push(stack(scan, 0, device)?);
    }
    Ok(values)
}

pub(super) fn scan(node: &ScanNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let config = &node.config;
    let body = &config.body;
    let scan_input_count = usize::try_from(config.num_scan_inputs)
        .map_err(|_| TynxError::Shape("Scan num_scan_inputs must be non-negative".to_string()))?;
    if body.inputs.len() < scan_input_count {
        return Err(TynxError::Shape(
            "Scan body has fewer inputs than num_scan_inputs".to_string(),
        ));
    }
    let state_count = body.inputs.len() - scan_input_count;
    let leading = node
        .inputs
        .len()
        .saturating_sub(state_count + scan_input_count + config.scope_ref_names.len());
    if leading > 1 {
        return Err(TynxError::UnsupportedOp(
            "Scan has an unsupported leading-input layout".to_string(),
        ));
    }

    let state = (0..state_count)
        .map(|index| resolve::at(env, &node.name, &node.inputs, leading + index, device))
        .collect::<Result<Vec<_>>>()?;
    let sequences = (0..scan_input_count)
        .map(|index| {
            resolve::at(
                env,
                &node.name,
                &node.inputs,
                leading + state_count + index,
                device,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    if leading == 0 {
        return scan_core(node, env, state, sequences, state_count, leading, device);
    }

    let batch = sequences
        .first()
        .or(state.first())
        .and_then(|value| shape::value_dims(value).first().copied())
        .ok_or_else(|| {
            TynxError::Shape("Scan has no batched state or sequence input".to_string())
        })?;
    let mut batched_outputs = vec![Vec::new(); body.outputs.len()];
    for batch_index in 0..batch {
        let batch_state = state
            .iter()
            .map(|value| scan_element(value, 0, batch_index, device))
            .collect::<Result<Vec<_>>>()?;
        let batch_sequences = sequences
            .iter()
            .map(|value| scan_element(value, 0, batch_index, device))
            .collect::<Result<Vec<_>>>()?;
        let outputs = scan_core(
            node,
            env,
            batch_state,
            batch_sequences,
            state_count,
            leading,
            device,
        )?;
        for (values, value) in batched_outputs.iter_mut().zip(outputs) {
            values.push(value);
        }
    }
    batched_outputs
        .into_iter()
        .map(|values| stack(values, 0, device))
        .collect()
}

fn scan_core(
    node: &ScanNode,
    env: &Env,
    mut state: Vec<Value>,
    sequences: Vec<Value>,
    state_count: usize,
    leading: usize,
    device: &Device,
) -> Result<Vec<Value>> {
    let config = &node.config;
    let body = &config.body;
    let scan_output_count = body.outputs.len().saturating_sub(state_count);
    let mut outputs = vec![Vec::new(); scan_output_count];
    let mut axes = Vec::with_capacity(sequences.len());
    for (index, sequence) in sequences.iter().enumerate() {
        axes.push(normalize_axis(
            config.scan_input_axes.get(index).copied().unwrap_or(0),
            shape::value_dims(sequence).len(),
        )?);
    }
    let iterations = sequences
        .first()
        .map(|sequence| shape::value_dims(sequence)[axes[0]])
        .unwrap_or(0);
    for (sequence, axis) in sequences.iter().zip(&axes) {
        if shape::value_dims(sequence)[*axis] != iterations {
            return Err(TynxError::Shape(
                "Scan inputs have different sequence lengths".to_string(),
            ));
        }
    }

    for iteration in 0..iterations {
        let mut body_env = env.clone();
        bind_scope(
            env,
            &mut body_env,
            &node.inputs,
            leading + state_count + sequences.len(),
            &config.scope_ref_names,
            device,
        )?;
        for (input, value) in body.inputs.iter().take(state_count).zip(&state) {
            body_env.insert(input.name.clone(), value.clone());
        }
        for (index, sequence) in sequences.iter().enumerate() {
            let sequence_index = if config.scan_input_directions.get(index) == Some(&1) {
                iterations - iteration - 1
            } else {
                iteration
            };
            body_env.insert(
                body.inputs[state_count + index].name.clone(),
                scan_element(sequence, axes[index], sequence_index, device)?,
            );
        }
        run_graph(body, &mut body_env, device)?;
        for (index, value) in state.iter_mut().enumerate() {
            *value = output(&body_env, &body.outputs[index])?;
        }
        for (index, values) in outputs.iter_mut().enumerate() {
            values.push(output(&body_env, &body.outputs[state_count + index])?);
        }
    }

    let mut values = state;
    for (index, mut scan_values) in outputs.into_iter().enumerate() {
        if config.scan_output_directions.get(index) == Some(&1) {
            scan_values.reverse();
        }
        let output_rank = scan_values
            .first()
            .map(|value| shape::value_dims(value).len() + 1)
            .unwrap_or(1);
        let axis = normalize_axis(
            config.scan_output_axes.get(index).copied().unwrap_or(0),
            output_rank,
        )?;
        values.push(stack(scan_values, axis, device)?);
    }
    Ok(values)
}

fn bind_scope(
    outer: &Env,
    inner: &mut Env,
    inputs: &[Argument],
    offset: usize,
    names: &[String],
    device: &Device,
) -> Result<()> {
    for (index, name) in names.iter().enumerate() {
        if let Some(input) = inputs.get(offset + index)
            && let Ok(value) = resolve::input_at(outer, input, offset + index, device)
        {
            inner.insert(name.clone(), value);
        }
    }
    Ok(())
}

fn optional_i64(
    env: &Env,
    input: Option<&Argument>,
    input_index: usize,
    device: &Device,
) -> Result<Option<i64>> {
    let Some(input) = input.filter(|input| !input.is_optional()) else {
        return Ok(None);
    };
    let value = resolve::input_at(env, input, input_index, device)?;
    let value = match value {
        Value::Scalar(Scalar::I64(value)) => value,
        Value::Scalar(Scalar::U64(value)) => i64::try_from(value)
            .map_err(|_| TynxError::Shape("Loop trip count exceeds i64".to_string()))?,
        Value::Int(value) => value
            .into_data()
            .iter::<i64>()
            .next()
            .ok_or_else(|| TynxError::Shape("Loop trip count is empty".to_string()))?,
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "Loop trip count must be an integer scalar, got {other:?}"
            )));
        }
    };
    Ok(Some(value.max(0)))
}

fn condition(value: Value) -> Result<bool> {
    match value {
        Value::Scalar(Scalar::Bool(value)) => Ok(value),
        Value::Bool(value) => value
            .into_data()
            .iter::<bool>()
            .next()
            .ok_or_else(|| TynxError::Shape("condition tensor is empty".to_string())),
        Value::Scalar(value) => Ok(value.as_f64() != 0.0),
        Value::Int(value) => Ok(value
            .into_data()
            .iter::<i64>()
            .next()
            .is_some_and(|v| v != 0)),
        Value::Tensor(value) => Ok(value
            .into_data()
            .iter::<f64>()
            .next()
            .is_some_and(|v| v != 0.0)),
        Value::Shape(value) => Ok(value.first().is_some_and(|value| *value != 0)),
    }
}

fn graph_outputs(graph: &OnnxGraph, env: &Env) -> Result<Vec<Value>> {
    graph
        .outputs
        .iter()
        .map(|argument| output(env, argument))
        .collect()
}

fn output(env: &Env, argument: &Argument) -> Result<Value> {
    env.get(&argument.name)
        .cloned()
        .ok_or_else(|| TynxError::MissingValue(argument.name.clone()))
}

fn scan_element(value: &Value, axis: usize, index: usize, device: &Device) -> Result<Value> {
    let mut slices = vec![Slice::full(); shape::value_dims(value).len()];
    let start = isize::try_from(index)
        .map_err(|_| TynxError::Shape("Scan index exceeds isize".to_string()))?;
    let end = start
        .checked_add(1)
        .ok_or_else(|| TynxError::Shape("Scan index overflow".to_string()))?;
    slices[axis] = Slice::new(start, Some(end), 1);
    let sliced = match value.clone() {
        Value::Tensor(value) => Value::Tensor(value.slice(&slices)),
        Value::Int(value) => Value::Int(value.slice(&slices)),
        Value::Bool(value) => Value::Bool(value.slice(&slices)),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "Scan input must be a tensor, got {other:?}"
            )));
        }
    };
    let mut dims = shape::value_dims(&sliced);
    dims.remove(axis);
    shape::reshape_value(sliced, dims, device)
}

fn stack(values: Vec<Value>, axis: usize, device: &Device) -> Result<Value> {
    if values.is_empty() {
        return Err(TynxError::Shape(
            "cannot materialize an empty control-flow scan output".to_string(),
        ));
    }
    let expanded = values
        .into_iter()
        .map(|value| {
            let mut dims = shape::value_dims(&value);
            if axis > dims.len() {
                return Err(TynxError::Shape(format!(
                    "scan output axis {axis} exceeds rank {}",
                    dims.len() + 1
                )));
            }
            dims.insert(axis, 1);
            shape::reshape_value(value, dims, device)
        })
        .collect::<Result<Vec<_>>>()?;
    concat(expanded, axis)
}

fn concat(values: Vec<Value>, axis: usize) -> Result<Value> {
    match values.first() {
        Some(Value::Tensor(_)) => values
            .into_iter()
            .map(Value::into_tensor)
            .collect::<Result<Vec<DynTensor>>>()
            .and_then(|values| DynTensor::concat(values, axis))
            .map(Value::Tensor),
        Some(Value::Int(_)) => values
            .into_iter()
            .map(Value::into_int)
            .collect::<Result<Vec<DynInt>>>()
            .and_then(|values| DynInt::concat(values, axis))
            .map(Value::Int),
        Some(Value::Bool(_)) => values
            .into_iter()
            .map(Value::into_bool)
            .collect::<Result<Vec<DynBool>>>()
            .and_then(|values| DynBool::concat(values, axis))
            .map(Value::Bool),
        _ => Err(TynxError::TypeMismatch(
            "scan outputs must have one consistent tensor type".to_string(),
        )),
    }
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize> {
    let rank_i64 =
        i64::try_from(rank).map_err(|_| TynxError::Shape("tensor rank exceeds i64".to_string()))?;
    let axis = if axis < 0 { axis + rank_i64 } else { axis };
    if !(0..rank_i64).contains(&axis) {
        return Err(TynxError::Shape(format!(
            "control-flow scan axis {axis} is outside rank {rank}"
        )));
    }
    Ok(axis as usize)
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;

    #[test]
    fn reads_boolean_scalar_and_tensor_conditions() {
        let device = Device::default();
        assert!(condition(Value::Scalar(Scalar::Bool(true))).unwrap());
        assert!(
            !condition(
                Value::from_tensor_data(TensorData::new(vec![false], [1]), 1, &device).unwrap()
            )
            .unwrap()
        );
    }

    #[test]
    fn stacks_scan_values_on_a_new_axis() {
        let device = Device::default();
        let values = [1.0_f32, 2.0]
            .into_iter()
            .map(|value| {
                Value::from_tensor_data(TensorData::new(vec![value], [1]), 1, &device).unwrap()
            })
            .collect();

        let output = stack(values, 0, &device).unwrap().into_tensor().unwrap();

        assert_eq!(output.dims(), [2, 1]);
        assert_eq!(
            output.into_data().iter::<f32>().collect::<Vec<_>>(),
            [1.0, 2.0]
        );
    }
}
