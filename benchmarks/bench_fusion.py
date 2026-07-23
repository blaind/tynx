#!/usr/bin/env python3
"""Materializing fusion throughput probe for dependent elementwise chains."""

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable
from typing import Any

import tynx


def measure(
    operation: Callable[[], tynx.Tensor], warmup: int, samples: int
) -> dict[str, Any]:
    for _ in range(warmup):
        operation().item()
    timings = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        operation().item()
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    ordered = sorted(timings)
    return {
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "p95_ms": ordered[math.ceil(samples * 0.95) - 1],
        "p99_ms": ordered[math.ceil(samples * 0.99) - 1],
        "samples_ms": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--binary-ops", type=int, default=20)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if min(args.elements, args.warmup, args.samples, args.binary_ops) <= 0:
        parser.error("all numeric arguments must be positive")

    input = tynx.ones(args.elements)
    other = tynx.ones(args.elements)

    def scalar_sum() -> tynx.Tensor:
        return input.sum()

    def unary_chain() -> tynx.Tensor:
        output = input.tanh().sigmoid() * 1.5 + 0.5
        return (output.abs() + 0.1).log().sqrt().sum()

    def binary_chain() -> tynx.Tensor:
        output = input
        for _ in range(args.binary_ops):
            output = output + other
        return output.sum()

    reduction = measure(scalar_sum, args.warmup, args.samples)
    unary = measure(unary_chain, args.warmup, args.samples)
    binary = measure(binary_chain, args.warmup, args.samples)
    unary_compute_ms = max(0.0, unary["median_ms"] - reduction["median_ms"])
    binary_compute_ms = max(0.0, binary["median_ms"] - reduction["median_ms"])
    result = {
        "schema": 1,
        "case": "materialized-dependent-fusion",
        "device": str(tynx.get_default_device()),
        "elements": args.elements,
        "unary_ops": 8,
        "binary_ops": args.binary_ops,
        "scalar_sum": reduction,
        "unary_chain": {
            **unary,
            "estimated_compute_ms": unary_compute_ms,
            "estimated_gb_per_second_per_op": (
                args.elements * 4 * 2 * 8 / (unary_compute_ms / 1000) / 1.0e9
                if unary_compute_ms
                else None
            ),
        },
        "binary_chain": {
            **binary,
            "estimated_compute_ms": binary_compute_ms,
        },
        "acceptance": {
            "unary_compute_median_ms_max": 1.0,
            "unary_compute_pass": unary_compute_ms <= 1.0,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.check and not result["acceptance"]["unary_compute_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
