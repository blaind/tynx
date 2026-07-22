#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_SHA256="c1793982c0504e1808e7d0d99d4cc5972de35137d6b5e8492573ecb72b2e241f"
readonly MODEL_URL="https://media.githubusercontent.com/media/onnx/models/d55d2baeb0d6641643d5295a4f42b545fcf9d9e2/Computer_Vision/mobilenetv2_100_Opset16_timm/mobilenetv2_100_Opset16.onnx"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_dir="${TYNX_BENCH_MODEL_DIR:-${repo_root}/.cache/benchmark-models}"
model_path="${model_dir}/mobilenetv2_100_opset16.onnx"

if [[ -f "${model_path}" ]] && printf '%s  %s\n' "${MODEL_SHA256}" "${model_path}" | sha256sum --check --status; then
  printf '%s\n' "${model_path}"
  exit 0
fi

mkdir -p "${model_dir}"
download_path="${model_path}.download"
trap 'rm -f "${download_path}"' EXIT
curl --fail --location --retry 3 "${MODEL_URL}" --output "${download_path}"
printf '%s  %s\n' "${MODEL_SHA256}" "${download_path}" | sha256sum --check --status
mv "${download_path}" "${model_path}"
trap - EXIT
printf '%s\n' "${model_path}"
