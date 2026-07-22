//! ONNX Slice execution.

use std::collections::HashSet;

use burn::tensor::{Device, Slice};
use onnx_ir::node::slice::{SliceInput, SliceNode};

use super::{Env, resolve, shape};
use crate::{Result, TynxError, Value};

pub(super) fn slice(node: &SliceNode, env: &Env, device: &Device) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let dims = shape::value_dims(&input);
    let starts = resolve_input(&node.config.starts, node, env, device)?;
    let ends = resolve_input(&node.config.ends, node, env, device)?;
    let axes = match &node.config.axes {
        Some(value) => resolve_input(value, node, env, device)?,
        None => (0..starts.len()).map(|axis| axis as i64).collect(),
    };
    let steps = match &node.config.steps {
        Some(value) => resolve_input(value, node, env, device)?,
        None => vec![1; starts.len()],
    };

    if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != steps.len() {
        return Err(TynxError::Shape(format!(
            "Slice starts, ends, axes, and steps must have equal lengths, got {}, {}, {}, and {}",
            starts.len(),
            ends.len(),
            axes.len(),
            steps.len()
        )));
    }

    let mut slices = vec![Slice::full(); dims.len()];
    let mut seen = HashSet::new();
    for (((start, end), axis), step) in starts.into_iter().zip(ends).zip(axes).zip(steps) {
        let axis = normalize_axis(axis, dims.len())?;
        if !seen.insert(axis) {
            return Err(TynxError::Shape(format!(
                "Slice axis {axis} is specified more than once"
            )));
        }
        slices[axis] = normalize_slice(start, end, step, dims[axis])?;
    }

    let output = match input {
        Value::Tensor(tensor) => Value::Tensor(tensor.slice(&slices)),
        Value::Int(tensor) => Value::Int(tensor.slice(&slices)),
        Value::Bool(tensor) => Value::Bool(tensor.slice(&slices)),
        Value::Shape(values) if slices.len() == 1 => {
            let indices = slices[0]
                .into_iter()
                .map(|index| index as usize)
                .collect::<Vec<_>>();
            Value::Shape(indices.into_iter().map(|index| values[index]).collect())
        }
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "Slice expects a tensor or shape, got {other:?}"
            )));
        }
    };
    Ok(vec![output])
}

fn resolve_input(
    input: &SliceInput,
    node: &SliceNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<i64>> {
    match input {
        SliceInput::Static(values) => Ok(values.clone()),
        SliceInput::Runtime(reference) => shape::value_to_i64s(resolve::at(
            env,
            &node.name,
            &node.inputs,
            reference.input_index,
            device,
        )?),
    }
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize> {
    let rank = rank as i64;
    let axis = if axis < 0 { axis + rank } else { axis };
    if !(0..rank).contains(&axis) {
        return Err(TynxError::Shape(format!(
            "Slice axis {axis} is outside rank {rank}"
        )));
    }
    Ok(axis as usize)
}

fn normalize_slice(start: i64, end: i64, step: i64, size: usize) -> Result<Slice> {
    if step == 0 {
        return Err(TynxError::Shape("Slice step must not be zero".into()));
    }
    let size =
        i64::try_from(size).map_err(|_| TynxError::Shape("Slice dimension exceeds i64".into()))?;

    if step > 0 {
        let start = normalize_forward(start, size);
        let end = normalize_forward(end, size);
        return make_slice(start, end, step);
    }

    if size == 0 {
        return Ok(Slice::new(0, Some(0), -1));
    }
    let start = normalize_reverse(start, size, false);
    let end = normalize_reverse(end, size, end == i64::MIN);
    let distance = start - end;
    if distance <= 0 {
        return Ok(Slice::new(0, Some(0), -1));
    }
    let count = (distance + step.unsigned_abs() as i64 - 1) / step.unsigned_abs() as i64;
    let lowest = start + (count - 1) * step;
    make_slice(lowest, start + 1, step)
}

fn normalize_forward(index: i64, size: i64) -> i64 {
    if index < 0 {
        index.saturating_add(size).clamp(0, size)
    } else {
        index.min(size)
    }
}

fn normalize_reverse(index: i64, size: i64, minimum_sentinel: bool) -> i64 {
    if minimum_sentinel {
        -1
    } else if index < 0 {
        index.saturating_add(size).clamp(-1, size - 1)
    } else {
        index.min(size - 1)
    }
}

fn make_slice(start: i64, end: i64, step: i64) -> Result<Slice> {
    let start = isize::try_from(start)
        .map_err(|_| TynxError::Shape(format!("Slice start {start} exceeds isize")))?;
    let end = isize::try_from(end)
        .map_err(|_| TynxError::Shape(format!("Slice end {end} exceeds isize")))?;
    let step = isize::try_from(step)
        .map_err(|_| TynxError::Shape(format!("Slice step {step} exceeds isize")))?;
    Ok(Slice::new(start, Some(end), step))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_reverse_slice_to_burn_bounds() {
        assert_eq!(
            normalize_slice(4, i64::MIN, -2, 5).unwrap(),
            Slice::new(0, Some(5), -2)
        );
    }

    #[test]
    fn clamps_forward_bounds() {
        assert_eq!(
            normalize_slice(-20, i64::MAX, 2, 5).unwrap(),
            Slice::new(0, Some(5), 2)
        );
    }

    #[test]
    fn rejects_zero_step() {
        assert!(normalize_slice(0, 1, 0, 5).is_err());
    }
}
