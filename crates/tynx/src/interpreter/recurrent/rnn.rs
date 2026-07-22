use burn::tensor::{Device, Tensor, activation};
use onnx_ir::node::rnn::{RnnActivationFunction, RnnDirection, RnnNode};

use super::{Env, common};
use crate::{Result, TynxError, Value};

pub(in crate::interpreter) fn rnn(
    node: &RnnNode,
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
    common::validate_sequence_length("RNN", sequence)?;
    let lengths = common::sequence_lengths(env, &node.inputs, batch, sequence, device)?;
    let reverse = match node.config.direction {
        RnnDirection::Forward => false,
        RnnDirection::Reverse => true,
        RnnDirection::Bidirectional => {
            return Err(TynxError::UnsupportedOp("bidirectional RNN".to_string()));
        }
    };
    let hidden = node.config.hidden_size;
    common::validate_single_direction("RNN", weight.dims()[0], recurrent.dims()[0])?;
    if weight.dims() != [1, hidden, input_size] || recurrent.dims() != [1, hidden, hidden] {
        return Err(TynxError::Shape(format!(
            "RNN weights have shapes {:?} and {:?}, expected [1, {hidden}, {input_size}] and [1, {hidden}, {hidden}]",
            weight.dims(),
            recurrent.dims()
        )));
    }
    let dtype = input.dtype();
    let weight = weight.cast(dtype).reshape([hidden, input_size]);
    let recurrent = recurrent.cast(dtype).reshape([hidden, hidden]);
    let bias = match bias {
        Some(bias) => {
            if bias.dims() != [1, 2 * hidden] {
                return Err(TynxError::Shape(format!(
                    "RNN bias has shape {:?}, expected [1, {}]",
                    bias.dims(),
                    2 * hidden
                )));
            }
            let bias = bias.cast(dtype).reshape([2 * hidden]);
            bias.clone()
                .narrow(0, 0, hidden)
                .add(bias.narrow(0, hidden, hidden))
                .reshape([1, hidden])
        }
        None => Tensor::<2>::zeros([1, hidden], (device, dtype)),
    };
    let projected = input
        .reshape([sequence * batch, input_size])
        .matmul(weight.transpose())
        .reshape([sequence, batch, hidden])
        .add(bias.reshape([1, 1, hidden]));
    let mut state = common::initial_state(initial, batch, hidden, dtype, device, "RNN initial_h")?;
    let recurrent = recurrent.transpose();
    let zero = Tensor::<3>::zeros([1, batch, hidden], (device, dtype));
    let mut outputs = vec![zero; sequence];
    for step in 0..sequence {
        let timestep = if reverse { sequence - step - 1 } else { step };
        let input = projected
            .clone()
            .narrow(0, timestep, 1)
            .reshape([batch, hidden]);
        let candidate = activate(
            common::clip(
                input.add(state.clone().matmul(recurrent.clone())),
                node.config.clip,
            ),
            node.config.hidden_activation,
        )?;
        let mask = common::active_mask(lengths.as_deref(), timestep, batch, dtype, device);
        let (next, output) = common::masked_state(candidate, state, mask);
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

fn activate(tensor: Tensor<2>, kind: RnnActivationFunction) -> Result<Tensor<2>> {
    Ok(match kind {
        RnnActivationFunction::Tanh => tensor.tanh(),
        RnnActivationFunction::Relu => activation::relu(tensor),
        RnnActivationFunction::Sigmoid => activation::sigmoid(tensor),
        other => {
            return Err(TynxError::UnsupportedOp(format!(
                "RNN activation {other:?}"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use onnx_ir::{
        DType,
        node::rnn::{RnnConfig, RnnNodeBuilder},
    };

    use super::*;

    #[test]
    fn runs_a_single_tanh_step() {
        let node = RnnNodeBuilder::new("rnn")
            .input_tensor("x", 3, DType::F32)
            .input_tensor("w", 3, DType::F32)
            .input_tensor("r", 3, DType::F32)
            .output_tensor("y", 4, DType::F32)
            .output_tensor("yh", 3, DType::F32)
            .config(RnnConfig {
                input_size: 1,
                hidden_size: 1,
                direction: RnnDirection::Forward,
                has_bias: false,
                has_initial_h: false,
                batch_first: false,
                clip: None,
                hidden_activation: RnnActivationFunction::Tanh,
            })
            .build();
        let device = Device::default();
        let mut env = Env::new();
        for (name, value) in [("x", 1.0_f32), ("w", 1.0), ("r", 0.0)] {
            env.insert(
                name.into(),
                Value::from_tensor_data(TensorData::new(vec![value], [1, 1, 1]), 3, &device)
                    .unwrap(),
            );
        }

        let output = rnn(&node, &env, &device).unwrap();

        let value = output[0]
            .clone()
            .into_tensor()
            .unwrap()
            .into_data()
            .iter::<f32>()
            .next()
            .unwrap();
        assert!((value - 1.0_f32.tanh()).abs() < 1e-6);
    }
}
