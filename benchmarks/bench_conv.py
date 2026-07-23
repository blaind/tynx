#!/usr/bin/env python3
"""Materializing Conv2d throughput comparison for Tynx and cuDNN."""

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable
from typing import Any

import tynx

SHAPES = ((64, 56), (128, 28), (256, 14), (512, 7))


def measure(
    chain: Callable[[], Any],
    chain_length: int,
    warmup: int,
    samples: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        chain().item()
    timings = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        sentinel = chain()
        sentinel.item()
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0 / chain_length)
    median_ms = statistics.median(timings)
    return {
        "median_ms": median_ms,
        "min_ms": min(timings),
        "p95_ms": sorted(timings)[math.ceil(len(timings) * 0.95) - 1],
        "samples_ms": timings,
    }


def with_tflops(result: dict[str, Any], flops: int) -> dict[str, Any]:
    result["tflops"] = flops / (result["median_ms"] / 1000.0) / 1.0e12
    return result


def tynx_case(
    channels: int,
    spatial: int,
    batch: int,
    chain_length: int,
    warmup: int,
    samples: int,
) -> dict[str, Any]:
    input = tynx.ones(batch, channels, spatial, spatial)
    weight = tynx.ones(channels, channels, 3, 3) * (1.0 / (channels * 9))

    def chain() -> tynx.Tensor:
        output = input
        for _ in range(chain_length):
            output = tynx.nn.functional.conv2d(output, weight, padding=1)
        return output.sum()

    flops = 2 * batch * spatial * spatial * channels * channels * 9
    return with_tflops(measure(chain, chain_length, warmup, samples), flops)


def explicit_im2col_conv2d(
    input: tynx.Tensor,
    weight: tynx.Tensor,
    zero_columns: tynx.Tensor,
    zero_rows: tynx.Tensor,
) -> tynx.Tensor:
    """Lower a k3/s1/p1 NCHW convolution to explicit im2col + matmul."""
    batch, channels, height, width = input.shape
    output_channels = weight.shape[0]
    padded_width = tynx.cat((zero_columns, input, zero_columns), dim=3)
    padded = tynx.cat((zero_rows, padded_width, zero_rows), dim=2)
    patches = [
        padded[:, :, kernel_y : kernel_y + height, kernel_x : kernel_x + width]
        for kernel_y in range(3)
        for kernel_x in range(3)
    ]
    columns = (
        tynx.stack(patches, dim=2)
        .permute(0, 3, 4, 1, 2)
        .reshape(batch * height * width, channels * 9)
    )
    kernels = weight.reshape(output_channels, channels * 9).transpose(0, 1)
    return (
        (columns @ kernels)
        .reshape(batch, height, width, output_channels)
        .permute(0, 3, 1, 2)
    )


def tynx_im2col_case(
    channels: int,
    spatial: int,
    batch: int,
    chain_length: int,
    warmup: int,
    samples: int,
) -> dict[str, Any]:
    input = tynx.ones(batch, channels, spatial, spatial)
    weight = tynx.ones(channels, channels, 3, 3) * (1.0 / (channels * 9))
    zero_columns = tynx.zeros(batch, channels, spatial, 1)
    zero_rows = tynx.zeros(batch, channels, 1, spatial + 2)
    expected = tynx.nn.functional.conv2d(input, weight, padding=1)
    actual = explicit_im2col_conv2d(input, weight, zero_columns, zero_rows)
    max_abs_error = (expected - actual).abs().max().item()

    def chain() -> tynx.Tensor:
        output = input
        for _ in range(chain_length):
            output = explicit_im2col_conv2d(output, weight, zero_columns, zero_rows)
        return output.sum()

    flops = 2 * batch * spatial * spatial * channels * channels * 9
    result = with_tflops(measure(chain, chain_length, warmup, samples), flops)
    result["max_abs_error"] = max_abs_error
    return result


def torch_cases(
    channels: int,
    spatial: int,
    batch: int,
    chain_length: int,
    warmup: int,
    samples: int,
) -> dict[str, Any] | None:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    flops = 2 * batch * spatial * spatial * channels * channels * 9

    def run(dtype: Any, tf32: bool, channels_last: bool) -> dict[str, Any]:
        torch.backends.cudnn.allow_tf32 = tf32
        input = torch.ones(
            (batch, channels, spatial, spatial), device="cuda", dtype=dtype
        )
        weight = torch.full(
            (channels, channels, 3, 3),
            1.0 / (channels * 9),
            device="cuda",
            dtype=dtype,
        )
        if channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
            weight = weight.contiguous(memory_format=torch.channels_last)

        @torch.no_grad()
        def chain() -> Any:
            output = input
            for _ in range(chain_length):
                output = functional.conv2d(output, weight, padding=1)
            return output.sum()

        return with_tflops(measure(chain, chain_length, warmup, samples), flops)

    return {
        "f32": run(torch.float32, False, False),
        "tf32": run(torch.float32, True, False),
        "fp16_channels_last": run(torch.float16, True, True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--chain-length", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if min(args.batch, args.chain_length, args.warmup, args.samples) <= 0:
        parser.error("batch, chain-length, warmup, and samples must be positive")

    cases = []
    for channels, spatial in SHAPES:
        cases.append(
            {
                "shape": [args.batch, channels, spatial, spatial],
                "kernel": [3, 3],
                "stride": [1, 1],
                "padding": [1, 1],
                "tynx": tynx_case(
                    channels,
                    spatial,
                    args.batch,
                    args.chain_length,
                    args.warmup,
                    args.samples,
                ),
                "tynx_explicit_im2col": tynx_im2col_case(
                    channels,
                    spatial,
                    args.batch,
                    args.chain_length,
                    args.warmup,
                    args.samples,
                ),
                "torch": torch_cases(
                    channels,
                    spatial,
                    args.batch,
                    args.chain_length,
                    args.warmup,
                    args.samples,
                ),
            }
        )
    print(
        json.dumps(
            {
                "schema": 1,
                "case": "conv2d-k3-s1-p1-dependent-chain",
                "device": str(tynx.get_default_device()),
                "chain_length": args.chain_length,
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
