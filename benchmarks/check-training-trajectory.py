#!/usr/bin/env python3
"""Compare Tynx and Burn-AOT training benchmark correctness reports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ValidationError(Exception):
    """A benchmark report failed cross-engine validation."""


def mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{location}: expected an object, got {type(value).__name__}")
    return value


def sequence(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{location}: expected an array, got {type(value).__name__}")
    return value


def field(value: Mapping[str, Any], name: str, location: str) -> Any:
    if name not in value:
        raise ValidationError(f"{location}: missing field {name!r}")
    return value[name]


def number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{location}: expected a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{location}: expected a finite number, got {value!r}")
    return result


def equal(left: Any, right: Any, location: str) -> None:
    if left != right:
        raise ValidationError(f"{location}: {left!r} != {right!r}")


def close(
    left: Any,
    right: Any,
    location: str,
    *,
    absolute: float,
    relative: float = 0.0,
) -> None:
    left_number = number(left, f"{location} (Tynx)")
    right_number = number(right, f"{location} (Burn-AOT)")
    tolerance = absolute + relative * max(abs(left_number), abs(right_number))
    difference = abs(left_number - right_number)
    if difference > tolerance:
        raise ValidationError(
            f"{location}: {left_number!r} != {right_number!r} "
            f"(difference {difference:.8g}, tolerance {tolerance:.8g})"
        )


def load_reports(path: Path) -> dict[str, Mapping[str, Any]]:
    try:
        with path.open(encoding="utf-8") as stream:
            reports = sequence(json.load(stream), str(path))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"{path}: {error}") from error

    by_case: dict[str, Mapping[str, Any]] = {}
    for index, raw_report in enumerate(reports):
        location = f"{path}[{index}]"
        report = mapping(raw_report, location)
        case = field(report, "case", location)
        if not isinstance(case, str) or not case:
            raise ValidationError(f"{location}.case: expected a non-empty string")
        if case in by_case:
            raise ValidationError(f"{path}: duplicate case {case!r}")
        by_case[case] = report
    if not by_case:
        raise ValidationError(f"{path}: report array is empty")
    return by_case


def trajectory_by_step(report: Mapping[str, Any], location: str) -> dict[int, Mapping[str, Any]]:
    correctness = mapping(field(report, "correctness", location), f"{location}.correctness")
    trajectory = sequence(
        field(correctness, "trajectory", f"{location}.correctness"),
        f"{location}.correctness.trajectory",
    )
    by_step: dict[int, Mapping[str, Any]] = {}
    for index, raw_point in enumerate(trajectory):
        point_location = f"{location}.correctness.trajectory[{index}]"
        point = mapping(raw_point, point_location)
        step = field(point, "step", point_location)
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValidationError(f"{point_location}.step: expected a positive integer")
        if step in by_step:
            raise ValidationError(f"{location}: duplicate trajectory step {step}")
        by_step[step] = mapping(field(point, "state", point_location), f"{point_location}.state")
    return by_step


def initial_state(report: Mapping[str, Any], location: str) -> Mapping[str, Any]:
    correctness = mapping(field(report, "correctness", location), f"{location}.correctness")
    return mapping(
        field(correctness, "initial", f"{location}.correctness"),
        f"{location}.correctness.initial",
    )


def compare_state_shape(
    tynx: Mapping[str, Any], burn: Mapping[str, Any], location: str
) -> None:
    for name in (
        "trainable",
        "frozen",
        "gradients",
        "updated_parameters",
        "parameter_count",
    ):
        equal(field(tynx, name, location), field(burn, name, location), f"{location}.{name}")
    if field(tynx, "finite", location) is not True:
        raise ValidationError(f"{location}.finite: Tynx state is not finite")
    if field(burn, "finite", location) is not True:
        raise ValidationError(f"{location}.finite: Burn-AOT state is not finite")


def compare_case(
    case: str,
    tynx: Mapping[str, Any],
    burn: Mapping[str, Any],
    mode: str,
) -> None:
    location = f"case {case!r}"
    for name in (
        "model_sha256",
        "backend",
        "mode",
        "sync_policy",
        "batch_size",
        "dataset_batches",
        "learning_rate",
    ):
        equal(field(tynx, name, location), field(burn, name, location), f"{location}.{name}")

    tynx_initial = initial_state(tynx, f"{location} Tynx")
    burn_initial = initial_state(burn, f"{location} Burn-AOT")
    compare_state_shape(tynx_initial, burn_initial, f"{location}.initial")
    equal(
        field(tynx_initial, "parameter_sha256", location),
        field(burn_initial, "parameter_sha256", location),
        f"{location}.initial.parameter_sha256",
    )

    tynx_trajectory = trajectory_by_step(tynx, f"{location} Tynx")
    burn_trajectory = trajectory_by_step(burn, f"{location} Burn-AOT")
    equal(set(tynx_trajectory), set(burn_trajectory), f"{location}.trajectory steps")
    if not tynx_trajectory:
        raise ValidationError(f"{location}.trajectory: no correctness steps")

    for step in sorted(tynx_trajectory):
        step_location = f"{location}.trajectory step {step}"
        tynx_state = tynx_trajectory[step]
        burn_state = burn_trajectory[step]
        compare_state_shape(tynx_state, burn_state, step_location)
        if mode == "cpu":
            close(
                field(tynx_state, "loss", step_location),
                field(burn_state, "loss", step_location),
                f"{step_location}.loss",
                absolute=0.0001,
            )
            equal(
                field(tynx_state, "parameter_sha256", step_location),
                field(burn_state, "parameter_sha256", step_location),
                f"{step_location}.parameter_sha256",
            )
        else:
            close(
                field(tynx_state, "loss", step_location),
                field(burn_state, "loss", step_location),
                f"{step_location}.loss",
                absolute=0.0001,
                relative=0.0001,
            )
            close(
                field(tynx_state, "parameter_sum", step_location),
                field(burn_state, "parameter_sum", step_location),
                f"{step_location}.parameter_sum",
                absolute=0.0001,
                relative=0.0001,
            )
            close(
                field(tynx_state, "parameter_l2", step_location),
                field(burn_state, "parameter_l2", step_location),
                f"{step_location}.parameter_l2",
                absolute=0.00001,
                relative=0.00001,
            )


def check(mode: str, tynx_path: Path, burn_path: Path) -> int:
    tynx_reports = load_reports(tynx_path)
    burn_reports = load_reports(burn_path)
    equal(set(tynx_reports), set(burn_reports), "benchmark cases")
    for case in sorted(tynx_reports):
        compare_case(case, tynx_reports[case], burn_reports[case], mode)
    print(f"{mode.upper()} training trajectories match for {len(tynx_reports)} case(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("tynx", type=Path, help="Tynx benchmark JSON report")
    parser.add_argument("burn_aot", type=Path, help="Burn-AOT benchmark JSON report")
    arguments = parser.parse_args()
    try:
        return check(arguments.mode, arguments.tynx, arguments.burn_aot)
    except ValidationError as error:
        print(f"training trajectory mismatch: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
