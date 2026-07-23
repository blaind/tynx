#!/usr/bin/env python3
"""Show the difference between lazy graph construction and forced execution."""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from typing import Any

import tynx


def measure(
    operation: Callable[[], None],
    samples: int,
    cleanup: Callable[[], None] | None = None,
) -> dict[str, Any]:
    timings = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        operation()
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
        if cleanup is not None:
            cleanup()
    return {
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "samples_ms": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--chain-length", type=int, default=20)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if min(args.elements, args.chain_length, args.samples) <= 0:
        parser.error("all arguments must be positive")

    input = tynx.ones(args.elements)

    def construct_unread() -> None:
        for _ in range(args.chain_length):
            output = input + 1.0
            del output

    def construct_and_materialize() -> None:
        for _ in range(args.chain_length):
            (input + 1.0).sum().item()

    for _ in range(3):
        construct_unread()
        tynx.synchronize()
        construct_and_materialize()

    print(
        json.dumps(
            {
                "schema": 1,
                "case": "lazy-materialization",
                "device": str(tynx.get_default_device()),
                "elements": args.elements,
                "chain_length": args.chain_length,
                "unread_independent_adds": measure(
                    construct_unread, args.samples, tynx.synchronize
                ),
                "materialized_each_scalar_sentinel": measure(
                    construct_and_materialize, args.samples
                ),
                "warning": (
                    "Unread timing measures graph construction and disposal, not tensor "
                    "throughput; use the materialized result for performance claims."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
