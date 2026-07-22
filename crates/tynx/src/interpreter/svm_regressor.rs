//! ONNX-ML SVMRegressor execution.

use burn::tensor::{DType, Device, TensorData};
use onnx_ir::node::svmregressor::{
    SVMKernelType, SVMPostTransform, SVMRegressorConfig, SVMRegressorNode,
};

use super::{Env, resolve};
use crate::{DynTensor, Result, TynxError, Value};

pub(super) fn svm_regressor(
    node: &SVMRegressorNode,
    env: &Env,
    device: &Device,
) -> Result<Vec<Value>> {
    let input = resolve::first(env, &node.name, &node.inputs, device)?;
    let input = match input {
        Value::Tensor(tensor) => tensor.cast(DType::F32),
        Value::Int(tensor) => tensor.to_float(DType::F32),
        other => {
            return Err(TynxError::TypeMismatch(format!(
                "SVMRegressor requires a floating-point or integer tensor, got {other:?}"
            )));
        }
    };
    let input_dims = input.dims();
    if input_dims.len() != 2 {
        return Err(TynxError::Shape(format!(
            "SVMRegressor requires rank-2 [batch, features] input, got rank {}",
            input_dims.len()
        )));
    }

    let mut prediction = match svm_parameters(&node.config, input_dims[1], device)? {
        Some((coefficients, support_vectors)) => {
            let kernel = kernel_values(input, support_vectors, &node.config)?;
            kernel.matmul(coefficients)?.add_scalar(rho(&node.config)?)
        }
        None => {
            if input_dims[1] != 1 {
                return Err(TynxError::Shape(format!(
                    "SVMRegressor without support vectors requires one input feature, got {}",
                    input_dims[1]
                )));
            }
            input
        }
    };

    prediction = match node.config.post_transform {
        SVMPostTransform::None => prediction,
        SVMPostTransform::Logistic | SVMPostTransform::SoftmaxZero => prediction.sigmoid(),
        SVMPostTransform::Softmax => {
            return Err(TynxError::UnsupportedOp(
                "SVMRegressor SOFTMAX post-transform".to_string(),
            ));
        }
        SVMPostTransform::Probit => {
            return Err(TynxError::UnsupportedOp(
                "SVMRegressor PROBIT post-transform".to_string(),
            ));
        }
    };

    Ok(vec![Value::Tensor(
        prediction.reshape(vec![input_dims[0]])?,
    )])
}

fn svm_parameters(
    config: &SVMRegressorConfig,
    input_features: usize,
    device: &Device,
) -> Result<Option<(DynTensor, DynTensor)>> {
    let (Some(coefficients), Some(support_vectors)) =
        (&config.coefficients, &config.support_vectors)
    else {
        if config.coefficients.is_some() || config.support_vectors.is_some() {
            return Err(TynxError::Shape(
                "SVMRegressor coefficients and support_vectors must be provided together"
                    .to_string(),
            ));
        }
        return Ok(None);
    };
    let n_supports = config
        .n_supports
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0)
        .ok_or_else(|| {
            TynxError::Shape(
                "SVMRegressor requires a positive n_supports with model parameters".to_string(),
            )
        })?;
    if coefficients.len() != n_supports {
        return Err(TynxError::Shape(format!(
            "SVMRegressor has {} coefficients for {n_supports} support vectors",
            coefficients.len()
        )));
    }
    if support_vectors.len() != n_supports * input_features {
        return Err(TynxError::Shape(format!(
            "SVMRegressor has {} support-vector values, expected {}",
            support_vectors.len(),
            n_supports * input_features
        )));
    }
    if let Some(features) = config.n_features
        && features != input_features
    {
        return Err(TynxError::Shape(format!(
            "SVMRegressor expects {features} features, got {input_features}"
        )));
    }

    let coefficients = DynTensor::from_data(
        TensorData::new(coefficients.clone(), [n_supports, 1]),
        2,
        device,
    )?;
    let support_vectors = DynTensor::from_data(
        TensorData::new(support_vectors.clone(), [n_supports, input_features]),
        2,
        device,
    )?;
    Ok(Some((coefficients, support_vectors)))
}

fn kernel_values(
    input: DynTensor,
    support_vectors: DynTensor,
    config: &SVMRegressorConfig,
) -> Result<DynTensor> {
    let support_transposed = support_vectors.clone().permute(vec![1, 0])?;
    match config.kernel_type {
        SVMKernelType::Linear => input.matmul(support_transposed),
        SVMKernelType::Rbf => {
            let gamma = kernel_param(config, 0, "gamma")?;
            let dot = input.clone().matmul(support_transposed)?;
            let input_norms = input.powf_scalar(2.0).sum_dims(&[1]);
            let support_norms = support_vectors
                .powf_scalar(2.0)
                .sum_dims(&[1])
                .permute(vec![1, 0])?;
            input_norms
                .add_broadcast(support_norms)?
                .sub_broadcast(dot.mul_scalar(2.0))
                .map(|distance| distance.mul_scalar(-gamma).exp())
        }
        SVMKernelType::Poly => {
            let gamma = kernel_param(config, 0, "gamma")?;
            let coef0 = kernel_param_or(config, 1, 0.0);
            let degree = kernel_param_or(config, 2, 0.0);
            Ok(input
                .matmul(support_transposed)?
                .mul_scalar(gamma)
                .add_scalar(coef0)
                .powf_scalar(degree))
        }
        SVMKernelType::Sigmoid => {
            let gamma = kernel_param(config, 0, "gamma")?;
            let coef0 = kernel_param_or(config, 1, 0.0);
            Ok(input
                .matmul(support_transposed)?
                .mul_scalar(gamma)
                .add_scalar(coef0)
                .tanh())
        }
    }
}

fn kernel_param(config: &SVMRegressorConfig, index: usize, name: &str) -> Result<f64> {
    config
        .kernel_params
        .as_ref()
        .and_then(|params| params.get(index))
        .copied()
        .map(f64::from)
        .ok_or_else(|| TynxError::Shape(format!("SVMRegressor kernel is missing {name}")))
}

fn kernel_param_or(config: &SVMRegressorConfig, index: usize, default: f64) -> f64 {
    config
        .kernel_params
        .as_ref()
        .and_then(|params| params.get(index))
        .copied()
        .map(f64::from)
        .unwrap_or(default)
}

fn rho(config: &SVMRegressorConfig) -> Result<f64> {
    let values = config.rho.as_deref().unwrap_or(&[]);
    match values {
        [] => Ok(0.0),
        [value] => Ok(f64::from(*value)),
        _ => Err(TynxError::Shape(format!(
            "SVMRegressor expects one rho value, got {}",
            values.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use onnx_ir::node::svmregressor::{SVMRegressorConfig, SVMRegressorNodeBuilder};

    use super::*;

    fn node(kernel: SVMKernelType, post_transform: SVMPostTransform) -> SVMRegressorNode {
        SVMRegressorNodeBuilder::new("svm")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(SVMRegressorConfig::new(
                Some(vec![1.0, -0.5]),
                match kernel {
                    SVMKernelType::Linear => None,
                    SVMKernelType::Rbf => Some(vec![0.5]),
                    SVMKernelType::Poly => Some(vec![1.0, 0.0, 2.0]),
                    SVMKernelType::Sigmoid => Some(vec![0.5, 0.0]),
                },
                kernel,
                Some(2),
                Some(2),
                None,
                post_transform,
                Some(vec![0.25]),
                Some(vec![1.0, 0.0, 0.0, 1.0]),
            ))
            .build()
    }

    fn input(device: &Device) -> Env {
        let mut env = Env::new();
        env.insert(
            "input".to_string(),
            Value::from_tensor_data(
                TensorData::new(vec![2.0_f32, 4.0, 1.0, 3.0], [2, 2]),
                2,
                device,
            )
            .unwrap(),
        );
        env
    }

    #[test]
    fn executes_a_linear_regressor() {
        let device = Device::default();
        let output = svm_regressor(
            &node(SVMKernelType::Linear, SVMPostTransform::None),
            &input(&device),
            &device,
        )
        .unwrap()
        .pop()
        .unwrap()
        .into_tensor()
        .unwrap();

        output
            .into_data()
            .assert_eq(&TensorData::new(vec![0.25_f32, -0.25], [2]), false);
    }

    #[test]
    fn executes_an_rbf_regressor() {
        let device = Device::default();
        let output = svm_regressor(
            &node(SVMKernelType::Rbf, SVMPostTransform::None),
            &input(&device),
            &device,
        )
        .unwrap()
        .pop()
        .unwrap()
        .into_tensor()
        .unwrap()
        .into_data()
        .iter::<f32>()
        .collect::<Vec<_>>();

        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn applies_the_logistic_post_transform() {
        let device = Device::default();
        let output = svm_regressor(
            &node(SVMKernelType::Linear, SVMPostTransform::Logistic),
            &input(&device),
            &device,
        )
        .unwrap()
        .pop()
        .unwrap()
        .into_tensor()
        .unwrap()
        .into_data()
        .iter::<f32>()
        .collect::<Vec<_>>();

        assert!(output.iter().all(|value| (0.0..=1.0).contains(value)));
    }
}
