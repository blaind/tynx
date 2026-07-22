use burn::tensor::{Device, Tensor, activation};
use onnx_ir::node::lstm::{LstmActivationFunction, LstmDirection, LstmNode};

use super::{Env, common};
use crate::{Result, TynxError, Value};

pub(in crate::interpreter) fn lstm(
    node: &LstmNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    if node.config.has_peephole || node.inputs.get(7).is_some_and(|input| !input.is_optional()) {
        return Err(TynxError::UnsupportedOp("LSTM peepholes".to_string()));
    }
    let input = common::required_rank3(env, &node.name, &node.inputs, 0, device)?;
    let weight = common::required_rank3(env, &node.name, &node.inputs, 1, device)?;
    let recurrent = common::required_rank3(env, &node.name, &node.inputs, 2, device)?;
    let bias = common::optional_rank2(env, &node.inputs, 3, device)?;
    let initial_hidden = common::optional_rank3(env, &node.inputs, 5, device)?;
    let initial_cell = common::optional_rank3(env, &node.inputs, 6, device)?;
    let input = common::time_major(input, node.config.batch_first);
    let initial_hidden =
        initial_hidden.map(|value| common::direction_major(value, node.config.batch_first));
    let initial_cell =
        initial_cell.map(|value| common::direction_major(value, node.config.batch_first));
    let [sequence, batch, input_size] = input.dims();
    common::validate_sequence_length("LSTM", sequence)?;
    let lengths = common::sequence_lengths(env, &node.inputs, batch, sequence, device)?;
    let reverse = match node.config.direction {
        LstmDirection::Forward => false,
        LstmDirection::Reverse => true,
        LstmDirection::Bidirectional => {
            return Err(TynxError::UnsupportedOp("bidirectional LSTM".to_string()));
        }
    };
    let hidden = node.config.hidden_size;
    common::validate_single_direction("LSTM", weight.dims()[0], recurrent.dims()[0])?;
    if weight.dims() != [1, 4 * hidden, input_size] || recurrent.dims() != [1, 4 * hidden, hidden] {
        return Err(TynxError::Shape(format!(
            "LSTM weights have shapes {:?} and {:?}, expected [1, {}, {input_size}] and [1, {}, {hidden}]",
            weight.dims(),
            recurrent.dims(),
            4 * hidden,
            4 * hidden
        )));
    }
    let dtype = input.dtype();
    let weight = weight.cast(dtype).reshape([4 * hidden, input_size]);
    let recurrent = recurrent.cast(dtype).reshape([4 * hidden, hidden]);
    let (input_bias, recurrent_bias) = match bias {
        Some(bias) => {
            if bias.dims() != [1, 8 * hidden] {
                return Err(TynxError::Shape(format!(
                    "LSTM bias has shape {:?}, expected [1, {}]",
                    bias.dims(),
                    8 * hidden
                )));
            }
            let bias = bias.cast(dtype).reshape([8 * hidden]);
            (
                bias.clone()
                    .narrow(0, 0, 4 * hidden)
                    .reshape([1, 4 * hidden]),
                bias.narrow(0, 4 * hidden, 4 * hidden)
                    .reshape([1, 4 * hidden]),
            )
        }
        None => (
            Tensor::<2>::zeros([1, 4 * hidden], (device, dtype)),
            Tensor::<2>::zeros([1, 4 * hidden], (device, dtype)),
        ),
    };
    let projected = input
        .reshape([sequence * batch, input_size])
        .matmul(weight.transpose())
        .reshape([sequence, batch, 4 * hidden])
        .add(input_bias.reshape([1, 1, 4 * hidden]));
    let mut hidden_state = common::initial_state(
        initial_hidden,
        batch,
        hidden,
        dtype,
        device,
        "LSTM initial_h",
    )?;
    let mut cell_state =
        common::initial_state(initial_cell, batch, hidden, dtype, device, "LSTM initial_c")?;
    let recurrent = recurrent.transpose();
    let zero = Tensor::<3>::zeros([1, batch, hidden], (device, dtype));
    let mut outputs = vec![zero; sequence];
    for step in 0..sequence {
        let timestep = if reverse { sequence - step - 1 } else { step };
        let gates = projected
            .clone()
            .narrow(0, timestep, 1)
            .reshape([batch, 4 * hidden])
            .add(hidden_state.clone().matmul(recurrent.clone()))
            .add(recurrent_bias.clone());
        let input_gate = activate(
            common::clip(gates.clone().narrow(1, 0, hidden), node.config.clip),
            node.config.gate_activation,
        )?;
        let output_gate = activate(
            common::clip(gates.clone().narrow(1, hidden, hidden), node.config.clip),
            node.config.gate_activation,
        )?;
        let forget_gate = if node.config.input_forget {
            input_gate.clone().mul_scalar(-1.0).add_scalar(1.0)
        } else {
            activate(
                common::clip(
                    gates.clone().narrow(1, 2 * hidden, hidden),
                    node.config.clip,
                ),
                node.config.gate_activation,
            )?
        };
        let candidate = activate(
            common::clip(gates.narrow(1, 3 * hidden, hidden), node.config.clip),
            node.config.cell_activation,
        )?;
        let next_cell = forget_gate
            .mul(cell_state.clone())
            .add(input_gate.mul(candidate));
        let next_hidden = output_gate.mul(activate(
            common::clip(next_cell.clone(), node.config.clip),
            node.config.hidden_activation,
        )?);
        let mask = common::active_mask(lengths.as_deref(), timestep, batch, dtype, device);
        let (next_hidden, output) = common::masked_state(next_hidden, hidden_state, mask.clone());
        let (next_cell, _) = common::masked_state(next_cell, cell_state, mask);
        hidden_state = next_hidden;
        cell_state = next_cell;
        outputs[timestep] = output.reshape([1, batch, hidden]);
    }
    let sequence_output = Tensor::cat(outputs, 0).reshape([sequence, 1, batch, hidden]);
    let hidden_output = hidden_state.reshape([1, batch, hidden]);
    let cell_output = cell_state.reshape([1, batch, hidden]);
    let (sequence_output, hidden_output, cell_output) = common::output_layout_lstm(
        sequence_output,
        hidden_output,
        cell_output,
        node.config.batch_first,
    );
    let mut outputs = vec![
        Value::Tensor(sequence_output),
        Value::Tensor(hidden_output),
        Value::Tensor(cell_output),
    ];
    outputs.truncate(node.outputs.len());
    Ok(outputs)
}

fn activate(tensor: Tensor<2>, kind: LstmActivationFunction) -> Result<Tensor<2>> {
    Ok(match kind {
        LstmActivationFunction::Sigmoid => activation::sigmoid(tensor),
        LstmActivationFunction::Tanh => tensor.tanh(),
        LstmActivationFunction::Relu => activation::relu(tensor),
        other => {
            return Err(TynxError::UnsupportedOp(format!(
                "LSTM activation {other:?}"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::lstm::{LstmConfig, LstmNodeBuilder},
    };

    use super::*;

    #[test]
    fn zero_weights_keep_zero_cell_and_hidden_states() {
        let node = LstmNodeBuilder::new("lstm")
            .input_tensor("x", 3, DType::F32)
            .input_tensor("w", 3, DType::F32)
            .input_tensor("r", 3, DType::F32)
            .output_tensor("y", 4, DType::F32)
            .output_tensor("yh", 3, DType::F32)
            .output_tensor("yc", 3, DType::F32)
            .config(LstmConfig {
                input_size: 1,
                hidden_size: 1,
                direction: LstmDirection::Forward,
                has_bias: false,
                has_initial_h: false,
                has_initial_c: false,
                has_peephole: false,
                batch_first: false,
                clip: None,
                input_forget: false,
                gate_activation: LstmActivationFunction::Sigmoid,
                cell_activation: LstmActivationFunction::Tanh,
                hidden_activation: LstmActivationFunction::Tanh,
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        for (name, size) in [("x", 1), ("w", 4), ("r", 4)] {
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

        let output = lstm(&node, &env, &device).unwrap();

        for value in &output[1..] {
            assert_eq!(
                value
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
}
