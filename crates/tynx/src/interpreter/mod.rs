//! Runtime dispatch for individual ONNX nodes.

mod binary;
mod broadcasting;
mod cast;
mod clip;
mod comparison;
mod concat;
mod convolution;
mod cumsum;
mod dropout;
mod extrema;
mod gather;
mod logical;
mod matrix;
mod normalization;
mod pooling;
mod pow;
mod reduction;
mod resolve;
mod scatter;
mod selection;
mod shape;
mod slice;
mod softmax;
mod spatial;
mod unary;
mod variadic;
mod where_op;

use std::collections::HashMap;

use burn::tensor::Device;
use onnx_ir::ir::{Node, OnnxGraph};

use crate::{Result, TynxError, Value};

/// Values available to nodes, keyed by ONNX argument name.
pub type Env = HashMap<String, Value>;

pub(crate) fn prepare_model(data: &[u8]) -> Result<(Vec<u8>, bool)> {
    convolution::prepare_model(data)
}

pub(crate) fn restore_dynamic_conv_inputs(data: &[u8], graph: &mut OnnxGraph) -> Result<()> {
    convolution::restore_dynamic_inputs(data, graph)
}

pub(crate) fn preserve_attributes(data: &[u8], graph: &mut OnnxGraph) -> Result<()> {
    reduction::preserve_attributes(data, graph)?;
    pooling::preserve_attributes(data, graph)
}

/// Execute one ONNX node using values from the runtime environment.
pub fn execute(node: &Node, env: &Env, device: &Device) -> Result<Vec<Value>> {
    match node {
        Node::Abs(node) => unary::abs(node, env, device),
        Node::Acos(node) => unary::acos(node, env, device),
        Node::Acosh(node) => unary::acosh(node, env, device),
        Node::Add(node) => binary::add(node, env, device),
        Node::And(node) => logical::and(node, env, device),
        Node::ArgMax(node) => extrema::argmax(node, env, device),
        Node::ArgMin(node) => extrema::argmin(node, env, device),
        Node::Asin(node) => unary::asin(node, env, device),
        Node::Asinh(node) => unary::asinh(node, env, device),
        Node::Atan(node) => unary::atan(node, env, device),
        Node::Atanh(node) => unary::atanh(node, env, device),
        Node::AveragePool1d(node) => pooling::average_pool1d(node, env, device),
        Node::AveragePool2d(node) => pooling::average_pool2d(node, env, device),
        Node::AveragePool3d(node) => pooling::average_pool3d(node, env, device),
        Node::BatchNormalization(node) => normalization::batch_normalization(node, env, device),
        Node::Cast(node) => cast::cast(node, env, device),
        Node::CastLike(node) => cast::cast_like(node, env, device),
        Node::Ceil(node) => unary::ceil(node, env, device),
        Node::Celu(node) => unary::celu(node, env, device),
        Node::Clip(node) => clip::clip(node, env, device),
        Node::Concat(node) => concat::concat(node, env, device),
        Node::ConstantOfShape(node) => shape::constant_of_shape(node, env, device),
        Node::Conv1d(node) => convolution::conv1d(node, env, device),
        Node::Conv2d(node) => convolution::conv2d(node, env, device),
        Node::Conv3d(node) => convolution::conv3d(node, env, device),
        Node::Cos(node) => unary::cos(node, env, device),
        Node::Cosh(node) => unary::cosh(node, env, device),
        Node::CumSum(node) => cumsum::cumsum(node, env, device),
        Node::Div(node) => binary::div(node, env, device),
        Node::Dropout(node) => dropout::dropout(node, env, device),
        Node::Elu(node) => unary::elu(node, env, device),
        Node::Equal(node) => comparison::equal(node, env, device),
        Node::Erf(node) => unary::erf(node, env, device),
        Node::Exp(node) => unary::exp(node, env, device),
        Node::Expand(node) => broadcasting::expand(node, env, device),
        Node::Floor(node) => unary::floor(node, env, device),
        Node::Flatten(node) => shape::flatten(node, env, device),
        Node::Gelu(node) => unary::gelu(node, env, device),
        Node::Gather(node) => gather::gather(node, env, device),
        Node::GatherElements(node) => gather::gather_elements(node, env, device),
        Node::GatherND(node) => gather::gather_nd(node, env, device),
        Node::GlobalAveragePool(node) => pooling::global_average_pool(node, env, device),
        Node::GlobalMaxPool(node) => pooling::global_max_pool(node, env, device),
        Node::Greater(node) => comparison::greater(node, env, device),
        Node::GreaterOrEqual(node) => comparison::greater_or_equal(node, env, device),
        Node::HardSigmoid(node) => unary::hard_sigmoid(node, env, device),
        Node::HardSwish(node) => unary::hard_swish(node, env, device),
        Node::Identity(node) => Ok(vec![resolve::first(env, &node.name, &node.inputs, device)?]),
        Node::InstanceNormalization(node) => {
            normalization::instance_normalization(node, env, device)
        }
        Node::LeakyRelu(node) => unary::leaky_relu(node, env, device),
        Node::Less(node) => comparison::less(node, env, device),
        Node::LessOrEqual(node) => comparison::less_or_equal(node, env, device),
        Node::Log(node) => unary::log(node, env, device),
        Node::LogSoftmax(node) => softmax::log_softmax(node, env, device),
        Node::Gemm(node) => matrix::gemm(node, env, device),
        Node::Max(node) => variadic::max(node, env, device),
        Node::MatMul(node) => matrix::matmul(node, env, device),
        Node::MatMulInteger(node) => matrix::matmul_integer(node, env, device),
        Node::MaxPool1d(node) => pooling::max_pool1d(node, env, device),
        Node::MaxPool2d(node) => pooling::max_pool2d(node, env, device),
        Node::MaxPool3d(node) => pooling::max_pool3d(node, env, device),
        Node::Mean(node) => variadic::mean(node, env, device),
        Node::Min(node) => variadic::min(node, env, device),
        Node::Mish(node) => unary::mish(node, env, device),
        Node::Mul(node) => binary::mul(node, env, device),
        Node::Neg(node) => unary::neg(node, env, device),
        Node::Not(node) => logical::not(node, env, device),
        Node::NonZero(node) => selection::nonzero(node, env, device),
        Node::OneHot(node) => selection::one_hot(node, env, device),
        Node::Or(node) => logical::or(node, env, device),
        Node::PRelu(node) => binary::prelu(node, env, device),
        Node::Pow(node) => pow::pow(node, env, device),
        Node::Reciprocal(node) => unary::reciprocal(node, env, device),
        Node::ReduceL1(node) => reduction::reduce_l1(node, env, device),
        Node::ReduceL2(node) => reduction::reduce_l2(node, env, device),
        Node::ReduceLogSum(node) => reduction::reduce_log_sum(node, env, device),
        Node::ReduceLogSumExp(node) => reduction::reduce_log_sum_exp(node, env, device),
        Node::ReduceMax(node) => reduction::reduce_max(node, env, device),
        Node::ReduceMean(node) => reduction::reduce_mean(node, env, device),
        Node::ReduceMin(node) => reduction::reduce_min(node, env, device),
        Node::ReduceProd(node) => reduction::reduce_prod(node, env, device),
        Node::ReduceSum(node) => reduction::reduce_sum(node, env, device),
        Node::ReduceSumSquare(node) => reduction::reduce_sum_square(node, env, device),
        Node::Reshape(node) => shape::reshape(node, env, device),
        Node::Relu(node) => unary::relu(node, env, device),
        Node::Round(node) => unary::round(node, env, device),
        Node::Selu(node) => unary::selu(node, env, device),
        Node::ScatterElements(node) => scatter::scatter_elements(node, env, device),
        Node::ScatterND(node) => scatter::scatter_nd(node, env, device),
        Node::Shape(node) => shape::shape_of(node, env, device),
        Node::Sigmoid(node) => unary::sigmoid(node, env, device),
        Node::Sign(node) => unary::sign(node, env, device),
        Node::Sin(node) => unary::sin(node, env, device),
        Node::Sinh(node) => unary::sinh(node, env, device),
        Node::Slice(node) => slice::slice(node, env, device),
        Node::Softplus(node) => unary::softplus(node, env, device),
        Node::Softmax(node) => softmax::softmax(node, env, device),
        Node::Softsign(node) => unary::softsign(node, env, device),
        Node::Sqrt(node) => unary::sqrt(node, env, device),
        Node::Squeeze(node) => shape::squeeze(node, env, device),
        Node::Sub(node) => binary::sub(node, env, device),
        Node::Sum(node) => variadic::sum(node, env, device),
        Node::Tan(node) => unary::tan(node, env, device),
        Node::Tanh(node) => unary::tanh(node, env, device),
        Node::ThresholdedRelu(node) => unary::thresholded_relu(node, env, device),
        Node::Tile(node) => broadcasting::tile(node, env, device),
        Node::TopK(node) => selection::topk(node, env, device),
        Node::Transpose(node) => shape::transpose(node, env, device),
        Node::Unsqueeze(node) => shape::unsqueeze(node, env, device),
        Node::Where(node) => where_op::where_op(node, env, device),
        Node::Xor(node) => logical::xor(node, env, device),
        _ => Err(TynxError::UnsupportedOp(operator_kind(node))),
    }
}

fn operator_kind(node: &Node) -> String {
    node.to_string()
        .split_whitespace()
        .next()
        .unwrap_or("unknown")
        .to_string()
}

#[cfg(test)]
mod tests {
    use onnx_ir::{
        BoolStore, DType, Node,
        node::{identity::IdentityNodeBuilder, is_nan::IsNaNNodeBuilder},
    };

    use super::*;
    use crate::Scalar;

    #[test]
    fn identity_returns_its_input() {
        let node = Node::Identity(
            IdentityNodeBuilder::new("identity")
                .input_scalar("x", DType::I64)
                .output_scalar("y", DType::I64)
                .build(),
        );
        let mut env = Env::new();
        env.insert("x".to_string(), Value::Scalar(Scalar::I64(42)));

        let outputs = execute(&node, &env, &Device::default()).unwrap();

        assert!(matches!(
            outputs.as_slice(),
            [Value::Scalar(Scalar::I64(42))]
        ));
    }

    #[test]
    fn unsupported_errors_name_the_operator() {
        let node = Node::IsNaN(
            IsNaNNodeBuilder::new("")
                .input_tensor("x", 1, DType::F32)
                .output_tensor("y", 1, DType::Bool(BoolStore::Native))
                .build(),
        );

        let error = execute(&node, &Env::new(), &Device::default()).unwrap_err();

        assert_eq!(error, TynxError::UnsupportedOp("IsNaN".to_string()));
    }
}
