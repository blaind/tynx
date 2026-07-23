#!/usr/bin/env python3
"""Lazy-safe GPU regression sweep across reduction sizes and graph depth."""

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable
from typing import Any

import tynx


DEFAULT_SIZES = (1 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024)


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
        "p99_ms": ordered[math.ceil(samples * 0.99) - 1],
        "samples_ms": timings,
    }


def reduction_case(elements: int, warmup: int, samples: int) -> dict[str, Any]:
    input = (
        tynx.cat((tynx.ones(elements - 1), tynx.full(1, 2.0)))
        if elements > 1
        else tynx.full(1, 2.0)
    )
    return {
        "elements": elements,
        "sum": measure(input.sum, warmup, samples),
        "max": measure(input.max, warmup, samples),
        "argmax": measure(input.argmax, warmup, samples),
        "argmax_expected": elements - 1,
        "argmax_observed": input.argmax().item(),
    }


def graph_construction_case(operations: int) -> dict[str, Any]:
    input = tynx.ones(1)
    started = time.perf_counter_ns()
    output = input
    for _ in range(operations):
        output = output + 1.0
    construction_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    materialized = output.item()
    return {
        "operations": operations,
        "construction_ms": construction_ms,
        "materialized_value": materialized,
        "expected_value": operations + 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--graph-ops", type=int, default=5000)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if min(*args.sizes, args.warmup, args.samples, args.graph_ops) <= 0:
        parser.error("all numeric arguments must be positive")

    reductions = [
        reduction_case(elements, args.warmup, args.samples) for elements in args.sizes
    ]
    largest = max(reductions, key=lambda case: case["elements"])
    graph = graph_construction_case(args.graph_ops)
    acceptance = {
        "largest_sum_median_ms_max": 0.4,
        "largest_argmax_median_ms_max": 2.0,
        "graph_construction_ms_max": 1000.0,
        "sum_pass": largest["sum"]["median_ms"] <= 0.4,
        "argmax_pass": largest["argmax"]["median_ms"] <= 2.0,
        "argmax_correct": largest["argmax_observed"] == largest["argmax_expected"],
        "graph_pass": graph["construction_ms"] <= 1000.0
        and graph["materialized_value"] == graph["expected_value"],
    }
    print(
        json.dumps(
            {
                "schema": 1,
                "case": "lazy-safe-regression-sweep",
                "device": str(tynx.get_default_device()),
                "reductions": reductions,
                "graph_construction": graph,
                "acceptance": acceptance,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.check and not (
        acceptance["sum_pass"]
        and acceptance["argmax_pass"]
        and acceptance["argmax_correct"]
        and acceptance["graph_pass"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
