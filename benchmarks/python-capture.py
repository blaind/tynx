#!/usr/bin/env python3
"""Compare eager Python/PyO3 dispatch with native Tynx graph replay."""

import argparse
import json
import statistics
import time
from collections.abc import Callable

import tynx


def operation_heavy(input: tynx.Tensor, depth: int) -> tynx.Tensor:
    output = input
    for _ in range(depth):
        output = output.relu()
    return output


def measure(
    function: Callable[[tynx.Tensor], tynx.Tensor],
    input: tynx.Tensor,
    iterations: int,
    repeats: int,
) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            function(input)
        tynx.synchronize()
        elapsed = time.perf_counter_ns() - start
        samples.append(elapsed / iterations / 1_000.0)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.depth, args.size, args.warmup, args.iterations, args.repeats) <= 0:
        parser.error("all numeric arguments must be positive")

    input = tynx.Tensor([float(index) - args.size / 2 for index in range(args.size)])

    def eager(value: tynx.Tensor) -> tynx.Tensor:
        return operation_heavy(value, args.depth)

    compiled = tynx.compile(eager, fullgraph=True)
    compiled(input)
    for _ in range(args.warmup):
        eager(input)
        compiled(input)
    tynx.synchronize()

    eager_us = measure(eager, input, args.iterations, args.repeats)
    compiled_us = measure(compiled, input, args.iterations, args.repeats)
    eager_median = statistics.median(eager_us)
    compiled_median = statistics.median(compiled_us)
    print(
        json.dumps(
            {
                "schema": 1,
                "case": "python-operation-heavy-relu",
                "backend": str(input.device),
                "depth": args.depth,
                "size": args.size,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "ir_nodes": compiled.node_counts[0],
                "eager_median_us": eager_median,
                "compiled_median_us": compiled_median,
                "speedup": eager_median / compiled_median,
                "eager_samples_us": eager_us,
                "compiled_samples_us": compiled_us,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
