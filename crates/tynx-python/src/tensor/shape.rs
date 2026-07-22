//! Python shape argument normalization for eager tensor movement operations.

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBool, PyList, PyListMethods, PyTuple, PyTupleMethods},
};
use tynx_core::MAX_RANK;

pub(super) fn reshape(args: &Bound<'_, PyTuple>, numel: usize) -> PyResult<Vec<usize>> {
    let requested = variadic_dims(args, "reshape shape")?;
    if requested.is_empty() {
        return Err(PyValueError::new_err(
            "reshape requires at least one dimension because rank-zero tensors are not supported",
        ));
    }
    if requested.len() > MAX_RANK {
        return Err(PyValueError::new_err(format!(
            "reshape rank {} exceeds the maximum rank {MAX_RANK}",
            requested.len()
        )));
    }

    let mut inferred = None;
    let mut known_product = 1usize;
    let mut output = Vec::with_capacity(requested.len());
    for (index, dim) in requested.into_iter().enumerate() {
        if dim == -1 {
            if inferred.replace(index).is_some() {
                return Err(PyValueError::new_err(
                    "reshape can infer at most one dimension with -1",
                ));
            }
            output.push(1);
        } else if dim <= 0 {
            return Err(PyValueError::new_err(format!(
                "reshape dimensions must be positive or -1, got {dim}"
            )));
        } else {
            let dim = usize::try_from(dim).expect("positive shape dimension fits usize");
            known_product = known_product.checked_mul(dim).ok_or_else(|| {
                PyValueError::new_err("reshape dimension product exceeds platform limits")
            })?;
            output.push(dim);
        }
    }

    if let Some(index) = inferred {
        if known_product == 0 || !numel.is_multiple_of(known_product) {
            return Err(size_mismatch(numel, &output));
        }
        output[index] = numel / known_product;
    } else if known_product != numel {
        return Err(size_mismatch(numel, &output));
    }
    Ok(output)
}

pub(super) fn expand(args: &Bound<'_, PyTuple>, input: &[usize]) -> PyResult<Vec<usize>> {
    let requested = variadic_dims(args, "expand shape")?;
    if requested.len() < input.len() {
        return Err(PyValueError::new_err(format!(
            "expand shape has rank {}, but input rank is {}",
            requested.len(),
            input.len()
        )));
    }
    if requested.len() > MAX_RANK {
        return Err(PyValueError::new_err(format!(
            "expand rank {} exceeds the maximum rank {MAX_RANK}",
            requested.len()
        )));
    }

    let output_rank = requested.len();
    let leading_dims = output_rank - input.len();
    requested
        .into_iter()
        .enumerate()
        .map(|(axis, requested)| {
            let leading = axis < leading_dims;
            let actual = if leading {
                1
            } else {
                input[axis - leading_dims]
            };
            if requested == -1 {
                if leading {
                    return Err(PyValueError::new_err(
                        "expand cannot use -1 for a new leading dimension",
                    ));
                }
                return Ok(actual);
            }
            if requested < 0 {
                return Err(PyValueError::new_err(format!(
                    "expand dimensions must be non-negative or -1, got {requested}"
                )));
            }
            let requested =
                usize::try_from(requested).expect("non-negative expand dimension fits usize");
            if actual != 1 && actual != requested {
                return Err(PyValueError::new_err(format!(
                    "cannot expand dimension of size {actual} to {requested}"
                )));
            }
            Ok(requested)
        })
        .collect()
}

pub(super) fn repeat(args: &Bound<'_, PyTuple>, rank: usize) -> PyResult<Vec<usize>> {
    let repeats = variadic_dims(args, "repeat counts")?;
    if repeats.len() < rank {
        return Err(PyValueError::new_err(format!(
            "repeat has {} counts for tensor rank {rank}",
            repeats.len()
        )));
    }
    if repeats.len() > MAX_RANK {
        return Err(PyValueError::new_err(format!(
            "repeat rank {} exceeds the maximum rank {MAX_RANK}",
            repeats.len()
        )));
    }
    repeats
        .into_iter()
        .map(|repeat| {
            usize::try_from(repeat).map_err(|_| {
                PyValueError::new_err(format!(
                    "repeat counts must be non-negative integers, got {repeat}"
                ))
            })
        })
        .collect()
}

pub(super) fn permutation(args: &Bound<'_, PyTuple>, rank: usize) -> PyResult<Vec<usize>> {
    let requested = variadic_dims(args, "permutation")?;
    if requested.len() != rank {
        return Err(PyValueError::new_err(format!(
            "permutation has {} dimensions for tensor rank {rank}",
            requested.len()
        )));
    }
    normalize_unique(requested, rank, "permutation")
}

pub(super) fn axis(
    value: &Bound<'_, PyAny>,
    rank: usize,
    insertion: bool,
    operation: &str,
) -> PyResult<usize> {
    let raw = extract_dim(value, operation)?;
    axis_value(raw, rank, insertion, operation)
}

pub(super) fn axis_value(
    raw: isize,
    rank: usize,
    insertion: bool,
    operation: &str,
) -> PyResult<usize> {
    let upper = if insertion { rank + 1 } else { rank };
    let upper_signed = isize::try_from(upper).expect("dynamic tensor rank fits isize");
    let normalized = if raw < 0 { upper_signed + raw } else { raw };
    if !(0..upper_signed).contains(&normalized) {
        return Err(PyValueError::new_err(format!(
            "{operation} dimension {raw} is out of range for tensor rank {rank}"
        )));
    }
    Ok(usize::try_from(normalized).expect("validated dimension is non-negative"))
}

pub(super) fn flatten(input: &[usize], start: usize, end: usize) -> PyResult<Vec<usize>> {
    if start > end {
        return Err(PyValueError::new_err(format!(
            "flatten start_dim {start} cannot come after end_dim {end}"
        )));
    }
    let flattened = input[start..=end]
        .iter()
        .try_fold(1usize, |product, &size| product.checked_mul(size))
        .ok_or_else(|| {
            PyValueError::new_err("flatten dimension product exceeds platform limits")
        })?;
    let mut output = Vec::with_capacity(input.len() - (end - start));
    output.extend_from_slice(&input[..start]);
    output.push(flattened);
    output.extend_from_slice(&input[end + 1..]);
    Ok(output)
}

pub(super) fn squeeze(input: &[usize], dim: Option<usize>) -> Vec<usize> {
    let mut output = input
        .iter()
        .enumerate()
        .filter_map(|(axis, &size)| {
            let remove = size == 1 && dim.is_none_or(|selected| selected == axis);
            (!remove).then_some(size)
        })
        .collect::<Vec<_>>();
    if output.is_empty() {
        output.push(1);
    }
    output
}

pub(super) fn unsqueeze(input: &[usize], dim: usize) -> PyResult<Vec<usize>> {
    if input.len() == MAX_RANK {
        return Err(PyValueError::new_err(format!(
            "unsqueeze would exceed the maximum rank {MAX_RANK}"
        )));
    }
    let mut output = input.to_vec();
    output.insert(dim, 1);
    Ok(output)
}

fn variadic_dims(args: &Bound<'_, PyTuple>, argument: &str) -> PyResult<Vec<isize>> {
    if args.len() == 1 {
        let first = args.get_item(0)?;
        if let Ok(tuple) = first.cast::<PyTuple>() {
            return tuple
                .iter()
                .map(|value| extract_dim(&value, argument))
                .collect();
        }
        if let Ok(list) = first.cast::<PyList>() {
            return list
                .iter()
                .map(|value| extract_dim(&value, argument))
                .collect();
        }
    }
    args.iter()
        .map(|value| extract_dim(&value, argument))
        .collect()
}

fn extract_dim(value: &Bound<'_, PyAny>, argument: &str) -> PyResult<isize> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "{argument} dimensions must be integers"
        )));
    }
    value
        .extract::<isize>()
        .map_err(|_| PyTypeError::new_err(format!("{argument} dimensions must be integers")))
}

fn normalize_unique(raw: Vec<isize>, rank: usize, operation: &str) -> PyResult<Vec<usize>> {
    let rank_signed = isize::try_from(rank).expect("dynamic tensor rank fits isize");
    let mut normalized = Vec::with_capacity(raw.len());
    for dim in raw {
        let axis = if dim < 0 { rank_signed + dim } else { dim };
        if !(0..rank_signed).contains(&axis) {
            return Err(PyValueError::new_err(format!(
                "{operation} dimension {dim} is out of range for tensor rank {rank}"
            )));
        }
        let axis = usize::try_from(axis).expect("validated dimension is non-negative");
        if normalized.contains(&axis) {
            return Err(PyValueError::new_err(format!(
                "{operation} dimension {axis} appears more than once"
            )));
        }
        normalized.push(axis);
    }
    Ok(normalized)
}

fn size_mismatch(numel: usize, shape: &[usize]) -> PyErr {
    PyValueError::new_err(format!(
        "cannot reshape a tensor with {numel} elements to shape {shape:?}"
    ))
}
