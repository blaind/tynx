use burn::tensor::{Device, Tensor, activation};
use onnx_ir::node::gru::{GruActivationFunction, GruDirection, GruNode};

use super::{Env, common};
use crate::{Result, TynxError, Value};

pub(in crate::interpreter) fn gru(
    node: &GruNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = common::required_rank3(env, &node.name, &node.inputs, 0, device)?;
    let weight = common::required_rank3(env, &node.name, &node.inputs, 1, device)?;
    let recurrent = common::required_rank3(env, &node.name, &node.inputs, 2, device)?;
    let bias = common::optional_rank2(env, &node.inputs, 3, device)?;
    let initial = common::optional_rank3(env, &node.inputs, 5, device)?;
    let input = common::time_major(input, node.config.batch_first);
    let initial = initial.map(|value| common::direction_major(value, node.config.batch_first));
    let [sequence, batch, input_size] = input.dims();
    common::validate_sequence_length("GRU", sequence)?;
    let lengths = common::sequence_lengths(env, &node.inputs, batch, sequence, device)?;
    let reverse = match node.config.direction {
        GruDirection::Forward => false,
        GruDirection::Reverse => true,
        GruDirection::Bidirectional => {
            return Err(TynxError::UnsupportedOp("bidirectional GRU".to_string()));
        }
    };
    let hidden = node.config.hidden_size;
    common::validate_single_direction("GRU", weight.dims()[0], recurrent.dims()[0])?;
    if weight.dims() != [1, 3 * hidden, input_size] || recurrent.dims() != [1, 3 * hidden, hidden] {
        return Err(TynxError::Shape(format!(
            "GRU weights have shapes {:?} and {:?}, expected [1, {}, {input_size}] and [1, {}, {hidden}]",
            weight.dims(),
            recurrent.dims(),
            3 * hidden,
            3 * hidden
        )));
    }
    let dtype = input.dtype();
    let weight = weight.cast(dtype).reshape([3 * hidden, input_size]);
    let recurrent = recurrent.cast(dtype).reshape([3 * hidden, hidden]);
    let (input_bias, recurrent_bias) = match bias {
        Some(bias) => {
            if bias.dims() != [1, 6 * hidden] {
                return Err(TynxError::Shape(format!(
                    "GRU bias has shape {:?}, expected [1, {}]",
                    bias.dims(),
                    6 * hidden
                )));
            }
            let bias = bias.cast(dtype).reshape([6 * hidden]);
            (
                bias.clone()
                    .narrow(0, 0, 3 * hidden)
                    .reshape([1, 3 * hidden]),
                bias.narrow(0, 3 * hidden, 3 * hidden)
                    .reshape([1, 3 * hidden]),
            )
        }
        None => (
            Tensor::<2>::zeros([1, 3 * hidden], (device, dtype)),
            Tensor::<2>::zeros([1, 3 * hidden], (device, dtype)),
        ),
    };
    let projected = input
        .reshape([sequence * batch, input_size])
        .matmul(weight.transpose())
        .reshape([sequence, batch, 3 * hidden])
        .add(input_bias.reshape([1, 1, 3 * hidden]));
    let mut state = common::initial_state(initial, batch, hidden, dtype, device, "GRU initial_h")?;
    let recurrent = recurrent.transpose();
    let zero = Tensor::<3>::zeros([1, batch, hidden], (device, dtype));
    let mut outputs = vec![zero; sequence];
    for step in 0..sequence {
        let timestep = if reverse { sequence - step - 1 } else { step };
        let projected = projected
            .clone()
            .narrow(0, timestep, 1)
            .reshape([batch, 3 * hidden]);
        let input_z = projected.clone().narrow(1, 0, hidden);
        let input_r = projected.clone().narrow(1, hidden, hidden);
        let input_h = projected.narrow(1, 2 * hidden, hidden);
        let hidden_projection = state
            .clone()
            .matmul(recurrent.clone())
            .add(recurrent_bias.clone());
        let hidden_z = hidden_projection.clone().narrow(1, 0, hidden);
        let hidden_r = hidden_projection.narrow(1, hidden, hidden);
        let update = activate(
            common::clip(input_z.add(hidden_z), node.config.clip),
            node.config.gate_activation,
        )?;
        let reset = activate(
            common::clip(input_r.add(hidden_r), node.config.clip),
            node.config.gate_activation,
        )?;
        let candidate_projection = if node.config.linear_before_reset {
            let projection = state
                .clone()
                .matmul(recurrent.clone())
                .narrow(1, 2 * hidden, hidden)
                .add(recurrent_bias.clone().narrow(1, 2 * hidden, hidden));
            input_h.add(reset.mul(projection))
        } else {
            let projection = reset
                .mul(state.clone())
                .matmul(recurrent.clone())
                .narrow(1, 2 * hidden, hidden)
                .add(recurrent_bias.clone().narrow(1, 2 * hidden, hidden));
            input_h.add(projection)
        };
        let candidate = activate(
            common::clip(candidate_projection, node.config.clip),
            node.config.hidden_activation,
        )?;
        let next = update
            .clone()
            .mul(state.clone())
            .add(update.mul_scalar(-1.0).add_scalar(1.0).mul(candidate));
        let mask = common::active_mask(lengths.as_deref(), timestep, batch, dtype, device);
        let (next, output) = common::masked_state(next, state, mask);
        state = next;
        outputs[timestep] = output.reshape([1, batch, hidden]);
    }
    let sequence_output = Tensor::cat(outputs, 0).reshape([sequence, 1, batch, hidden]);
    let hidden_output = state.reshape([1, batch, hidden]);
    let (sequence_output, hidden_output) =
        common::output_layout(sequence_output, hidden_output, node.config.batch_first);
    let mut outputs = vec![Value::Tensor(sequence_output), Value::Tensor(hidden_output)];
    outputs.truncate(node.outputs.len());
    Ok(outputs)
}

fn activate(tensor: Tensor<2>, kind: GruActivationFunction) -> Result<Tensor<2>> {
    Ok(match kind {
        GruActivationFunction::Sigmoid => activation::sigmoid(tensor),
        GruActivationFunction::Tanh => tensor.tanh(),
        GruActivationFunction::Relu => activation::relu(tensor),
        other => {
            return Err(TynxError::UnsupportedOp(format!(
                "GRU activation {other:?}"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::gru::{GruConfig, GruNodeBuilder},
    };

    use super::*;

    #[test]
    fn zero_weights_keep_zero_hidden_state() {
        let node = GruNodeBuilder::new("gru")
            .input_tensor("x", 3, DType::F32)
            .input_tensor("w", 3, DType::F32)
            .input_tensor("r", 3, DType::F32)
            .output_tensor("y", 4, DType::F32)
            .output_tensor("yh", 3, DType::F32)
            .config(GruConfig {
                input_size: 1,
                hidden_size: 1,
                direction: GruDirection::Forward,
                has_bias: false,
                has_initial_h: false,
                batch_first: false,
                clip: None,
                linear_before_reset: false,
                gate_activation: GruActivationFunction::Sigmoid,
                hidden_activation: GruActivationFunction::Tanh,
                activation_alpha: None,
                activation_beta: None,
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        for (name, size) in [("x", 1), ("w", 3), ("r", 3)] {
            env.insert(
                name.into(),
                Value::from_tensor_data(
                    TensorData::new(vec![0.0_f32; size], [1, size, 1]),
                    3,
                    &device,
                )
                .unwrap(),
            );
        }

        let output = gru(&node, &env, &device).unwrap();

        assert_eq!(
            output[1]
                .clone()
                .into_tensor()
                .unwrap()
                .into_data()
                .iter::<f32>()
                .collect::<Vec<_>>(),
            [0.0]
        );
    }
}
