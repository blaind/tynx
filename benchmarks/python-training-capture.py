#!/usr/bin/env python3
"""Compare eager Python training steps with native whole-step capture replay."""

import argparse
import json
import statistics
import time
from collections.abc import Callable

import tynx


def build_step(
    features: int, hidden: int, outputs: int
) -> Callable[[tynx.Tensor, tynx.Tensor], tynx.Tensor]:
    tynx.manual_seed(20260722)
    model = tynx.nn.Sequential(
        tynx.nn.Linear(features, hidden),
        tynx.nn.ReLU(),
        tynx.nn.Linear(hidden, outputs),
    )
    optimizer = tynx.optim.SGD(model.parameters(), lr=0.001)

    def step(input: tynx.Tensor, target: tynx.Tensor) -> tynx.Tensor:
        optimizer.zero_grad()
        loss = tynx.nn.functional.mse_loss(model(input), target)
        loss.backward()
        optimizer.step()
        return loss

    return step


def measure(
    function: Callable[[tynx.Tensor, tynx.Tensor], tynx.Tensor],
    input: tynx.Tensor,
    target: tynx.Tensor,
    iterations: int,
    repeats: int,
) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            function(input, target)
        tynx.synchronize()
        elapsed = time.perf_counter_ns() - start
        samples.append(elapsed / iterations / 1_000.0)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--outputs", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if (
        min(
            args.batch,
            args.features,
            args.hidden,
            args.outputs,
            args.warmup,
            args.iterations,
            args.repeats,
        )
        <= 0
    ):
        parser.error("all numeric arguments must be positive")

    input = tynx.Tensor(
        [
            [
                ((row * args.features + column) % 31 - 15) / 16.0
                for column in range(args.features)
            ]
            for row in range(args.batch)
        ]
    )
    target = tynx.Tensor(
        [
            [
                ((row * args.outputs + column) % 13 - 6) / 8.0
                for column in range(args.outputs)
            ]
            for row in range(args.batch)
        ]
    )
    eager = build_step(args.features, args.hidden, args.outputs)
    captured = tynx.compile(
        build_step(args.features, args.hidden, args.outputs),
        fullgraph=True,
    )
    captured(input, target)
    for _ in range(args.warmup):
        eager(input, target)
        captured(input, target)
    tynx.synchronize()

    eager_us = measure(eager, input, target, args.iterations, args.repeats)
    captured_us = measure(captured, input, target, args.iterations, args.repeats)
    eager_median = statistics.median(eager_us)
    captured_median = statistics.median(captured_us)
    print(
        json.dumps(
            {
                "schema": 1,
                "case": "python-whole-step-mlp-sgd",
                "backend": str(input.device),
                "batch": args.batch,
                "features": args.features,
                "hidden": args.hidden,
                "outputs": args.outputs,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "ir_nodes": captured.node_counts[0],
                "eager_median_us": eager_median,
                "captured_median_us": captured_median,
                "speedup": eager_median / captured_median,
                "eager_samples_us": eager_us,
                "captured_samples_us": captured_us,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
