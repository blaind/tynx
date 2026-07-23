#!/usr/bin/env python3
"""Materializing scalar- and row-reduction performance probe."""

import argparse
import json
import math
import os
import statistics
import time
from collections.abc import Callable
from typing import Any

import tynx


def measure(
    function: Callable[[], Any],
    expected: float | int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        value = function().item()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        value = function().item()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    if isinstance(expected, int):
        if value != expected:
            raise RuntimeError(f"reduction returned {value}, expected {expected}")
    elif abs(value - expected) > max(1.0e-5, abs(expected) * 1.0e-6):
        raise RuntimeError(f"reduction returned {value}, expected {expected}")
    median_ms = statistics.median(samples)
    return {
        "median_ms": median_ms,
        "min_ms": min(samples),
        "p95_ms": sorted(samples)[math.ceil(len(samples) * 0.95) - 1],
        "p99_ms": sorted(samples)[math.ceil(len(samples) * 0.99) - 1],
        "samples_ms": samples,
    }


def add_bandwidth(result: dict[str, Any], elements: int) -> dict[str, Any]:
    result["effective_gbps"] = elements * 4 / (result["median_ms"] / 1000.0) / 1.0e9
    return result


def tynx_results(elements: int, warmup: int, iterations: int) -> dict[str, Any]:
    vector = tynx.ones(elements)
    matrix = tynx.ones(4096, 4096)
    results = {
        "sum_scalar": add_bandwidth(
            measure(vector.sum, float(elements), warmup, iterations), elements
        ),
        "mean_scalar": add_bandwidth(
            measure(vector.mean, 1.0, warmup, iterations), elements
        ),
        "max_scalar": add_bandwidth(
            measure(vector.max, 1.0, warmup, iterations), elements
        ),
        "argmax_scalar": add_bandwidth(
            measure(vector.argmax, 0, warmup, iterations), elements
        ),
    }
    row = measure(
        lambda: matrix.sum(dim=-1).sum(),
        float(4096 * 4096),
        warmup,
        iterations,
    )
    results["sum_rows_4096x4096"] = add_bandwidth(row, 4096 * 4096)
    return results


def torch_results(elements: int, warmup: int, iterations: int) -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    vector = torch.ones(elements, device="cuda")
    matrix = torch.ones((4096, 4096), device="cuda")

    def timed(function: Callable[[], Any]) -> Any:
        output = function()
        torch.cuda.synchronize()
        return output

    results = {
        "sum_scalar": add_bandwidth(
            measure(lambda: timed(vector.sum), float(elements), warmup, iterations),
            elements,
        ),
        "mean_scalar": add_bandwidth(
            measure(lambda: timed(vector.mean), 1.0, warmup, iterations), elements
        ),
        "max_scalar": add_bandwidth(
            measure(lambda: timed(vector.max), 1.0, warmup, iterations), elements
        ),
        "argmax_scalar": add_bandwidth(
            measure(lambda: timed(vector.argmax), 0, warmup, iterations), elements
        ),
    }
    row = measure(
        lambda: timed(matrix.sum(dim=-1).sum()),
        float(4096 * 4096),
        warmup,
        iterations,
    )
    results["sum_rows_4096x4096"] = add_bandwidth(row, 4096 * 4096)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=64_000_000)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--cpu",
        type=int,
        help="pin the process to one Linux CPU to remove scheduler-induced readback modes",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero unless the materialized reduction acceptance targets pass",
    )
    args = parser.parse_args()
    if min(args.elements, args.warmup, args.iterations) <= 0:
        parser.error("elements, warmup, and iterations must be positive")
    if args.cpu is not None:
        if not hasattr(os, "sched_setaffinity"):
            parser.error("--cpu is only supported on platforms with sched_setaffinity")
        available = os.sched_getaffinity(0)
        if args.cpu not in available:
            parser.error(
                f"CPU {args.cpu} is unavailable; choose one of {sorted(available)}"
            )
        os.sched_setaffinity(0, {args.cpu})

    tynx_report = tynx_results(args.elements, args.warmup, args.iterations)
    targets = {
        "sum_scalar_median_ms_max": 0.4,
        "sum_scalar_p99_ms_max": 1.0,
        "argmax_scalar_median_ms_max": 2.0,
    }
    checks = {
        "sum_scalar_median": tynx_report["sum_scalar"]["median_ms"]
        <= targets["sum_scalar_median_ms_max"],
        "sum_scalar_p99": tynx_report["sum_scalar"]["p99_ms"]
        <= targets["sum_scalar_p99_ms_max"],
        "argmax_scalar_median": tynx_report["argmax_scalar"]["median_ms"]
        <= targets["argmax_scalar_median_ms_max"],
    }
    report = {
        "schema": 2,
        "case": "materialized-reductions",
        "elements": args.elements,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "device": str(tynx.get_default_device()),
        "tynx": tynx_report,
        "torch": torch_results(args.elements, args.warmup, args.iterations),
        "acceptance": {
            "targets": targets,
            "checks": checks,
            "passed": all(checks.values()),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.check and not report["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
