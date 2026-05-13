#!/usr/bin/env python3
"""
Check XRoboToolkit teleoperation pickle logs for common data quality problems.

Examples:
    python3 scripts/misc/check_teleop_log_health.py logs/x2_upper_body_hardware
    python3 scripts/misc/check_teleop_log_health.py logs/x2_upper_body_hardware/teleop_log_*.pkl --decode-images

The script exits with:
    0 when no ERROR is found
    1 when any ERROR is found, or when --strict-warnings is used and WARN exists
"""

from __future__ import annotations

import argparse
import glob
import math
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import cv2
except ImportError:  # Image decoding is optional unless --decode-images is used.
    cv2 = None


DEFAULT_REQUIRED_KEYS = ("timestamp", "arm_state", "arm_command", "image")
NUMERIC_DICT_KEYS = (
    "arm_state",
    "arm_velocity",
    "arm_command",
    "head_state",
    "head_velocity",
    "head_command",
    "hand_command",
    "hand_trigger_raw",
)


@dataclass
class Issue:
    level: str
    message: str


@dataclass
class LogReport:
    path: Path
    entry_count: int = 0
    duration_s: float | None = None
    mean_hz: float | None = None
    min_dt_s: float | None = None
    max_dt_s: float | None = None
    issues: list[Issue] = field(default_factory=list)
    numeric_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    motion_summary: dict[str, float] = field(default_factory=dict)
    cameras: list[str] = field(default_factory=list)
    mostly_static: bool = False
    mostly_static_reason: str = ""

    def add(self, level: str, message: str) -> None:
        self.issues.append(Issue(level, message))

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.level == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.level == "WARN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Log .pkl file(s), directories, or glob patterns. Directories are searched recursively.",
    )
    parser.add_argument("--required-key", action="append", dest="required_keys", help="Required top-level key.")
    parser.add_argument("--min-entries", type=int, default=2, help="Minimum valid entries required per log.")
    parser.add_argument("--min-hz", type=float, default=1.0, help="Warn if average logging frequency is below this.")
    parser.add_argument("--max-hz", type=float, default=120.0, help="Warn if average logging frequency is above this.")
    parser.add_argument("--max-dt", type=float, default=0.5, help="Warn if adjacent timestamp gap is above this many seconds.")
    parser.add_argument("--joint-limit", type=float, default=2 * math.pi, help="Warn if joint position/command abs value exceeds this.")
    parser.add_argument("--velocity-limit", type=float, default=20.0, help="Warn if joint velocity abs value exceeds this.")
    parser.add_argument("--max-jump", type=float, default=1.0, help="Warn if adjacent joint position/command jump exceeds this.")
    parser.add_argument(
        "--static-action-travel",
        type=float,
        default=0.5,
        help="Mark a log as mostly static if total arm_command joint travel is below this.",
    )
    parser.add_argument(
        "--static-state-travel",
        type=float,
        default=0.5,
        help="Mark a log as mostly static if total arm_state joint travel is below this.",
    )
    parser.add_argument(
        "--static-command-range",
        type=float,
        default=0.2,
        help="Mark a log as mostly static if max per-joint arm_command range is below this.",
    )
    parser.add_argument(
        "--static-state-range",
        type=float,
        default=0.2,
        help="Mark a log as mostly static if max per-joint arm_state range is below this.",
    )
    parser.add_argument("--decode-images", action="store_true", help="Decode image bytes/arrays and validate dimensions.")
    parser.add_argument("--strict-warnings", action="store_true", help="Return exit code 1 when WARN issues exist.")
    parser.add_argument("--max-issues", type=int, default=20, help="Maximum issues printed per log.")
    return parser.parse_args()


def expand_paths(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        raw_text = str(raw_path)
        matches = [Path(match) for match in sorted(glob.glob(raw_text))] if any(ch in raw_text for ch in "*?[]") else [raw_path]
        for path in matches:
            if path.is_dir():
                files.extend(sorted(path.rglob("*.pkl")))
            elif path.is_file():
                files.append(path)
            else:
                print(f"WARN: path does not exist or is not readable: {path}", file=sys.stderr)
    return sorted(dict.fromkeys(path.resolve() for path in files))


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and np.isfinite(float(value))


def flatten_numeric_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        if is_finite_number(item):
            result[str(key)] = float(item)
        elif isinstance(item, np.ndarray) and np.issubdtype(item.dtype, np.number):
            arr = np.asarray(item, dtype=float).reshape(-1)
            for idx, number in enumerate(arr):
                result[f"{key}[{idx}]"] = float(number)
        elif isinstance(item, (list, tuple)) and all(is_finite_number(x) for x in item):
            for idx, number in enumerate(item):
                result[f"{key}[{idx}]"] = float(number)
    return result


def decode_image(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, dict) and "raw" in raw:
        return decode_raw_image_dict(raw)
    if isinstance(raw, bytes):
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed; cannot decode image bytes")
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if isinstance(raw, np.ndarray):
        return raw
    return None


def raw_encoding_to_dtype_channels(encoding: str):
    normalized = str(encoding).lower()
    if normalized in {"bgr8", "rgb8"}:
        return np.uint8, 3
    if normalized in {"mono8", "8uc1"}:
        return np.uint8, 1
    if normalized in {"mono16", "16uc1"}:
        return np.uint16, 1
    if normalized == "32fc1":
        return np.float32, 1
    return None, None


def decode_raw_image_dict(raw: dict) -> np.ndarray | None:
    dtype, channels = raw_encoding_to_dtype_channels(raw.get("encoding", ""))
    if dtype is None:
        return None
    width = int(raw["width"])
    height = int(raw["height"])
    itemsize = np.dtype(dtype).itemsize
    step = int(raw.get("step") or width * channels * itemsize)
    row_items = step // itemsize
    image = np.frombuffer(raw["raw"], dtype=dtype, count=height * row_items)
    if bool(raw.get("is_bigendian", False)) != (sys.byteorder == "big"):
        image = image.byteswap()
    image = image.reshape((height, row_items))
    image = image[:, : width * channels]
    if channels > 1:
        image = image.reshape((height, width, channels))
    else:
        image = image.reshape((height, width))
    return image


def collect_image_streams(image_value: Any) -> dict[str, Any]:
    streams: dict[str, Any] = {}
    if not isinstance(image_value, dict):
        return streams
    for camera_name, camera_data in image_value.items():
        if isinstance(camera_data, dict):
            for stream_name, raw in camera_data.items():
                streams[f"{camera_name}.{stream_name}"] = raw
        else:
            streams[str(camera_name)] = camera_data
    return streams


def check_log(path: Path, args: argparse.Namespace) -> LogReport:
    report = LogReport(path=path)
    try:
        data = load_pickle(path)
    except Exception as exc:
        report.add("ERROR", f"failed to load pickle: {exc}")
        return report

    if not isinstance(data, list):
        report.add("ERROR", f"top-level object must be list, got {type(data).__name__}")
        return report
    report.entry_count = len(data)
    if len(data) < args.min_entries:
        report.add("ERROR", f"too few entries: {len(data)} < {args.min_entries}")
        return report
    if not all(isinstance(entry, dict) for entry in data):
        bad = [idx for idx, entry in enumerate(data) if not isinstance(entry, dict)][:5]
        report.add("ERROR", f"all entries must be dict; bad indices: {bad}")
        return report

    required_keys = tuple(args.required_keys or DEFAULT_REQUIRED_KEYS)
    check_required_keys(data, required_keys, report)
    check_key_consistency(data, report)
    check_timestamps(data, args, report)
    check_numeric_fields(data, args, report)
    check_motion_content(data, args, report)
    check_images(data, args, report)
    return report


def check_required_keys(data: list[dict], required_keys: tuple[str, ...], report: LogReport) -> None:
    for key in required_keys:
        missing = [idx for idx, entry in enumerate(data) if key not in entry]
        if missing:
            report.add("ERROR", f"required key '{key}' missing in {len(missing)} entries; first indices: {missing[:5]}")


def check_key_consistency(data: list[dict], report: LogReport) -> None:
    expected = set(data[0].keys())
    mismatches = []
    for idx, entry in enumerate(data[1:], start=1):
        keys = set(entry.keys())
        if keys != expected:
            mismatches.append((idx, sorted(expected - keys), sorted(keys - expected)))
    if mismatches:
        details = "; ".join(f"#{idx} missing={missing} extra={extra}" for idx, missing, extra in mismatches[:3])
        report.add("ERROR", f"top-level keys are inconsistent in {len(mismatches)} entries: {details}")

    for key in NUMERIC_DICT_KEYS:
        first_dict = next((entry.get(key) for entry in data if isinstance(entry.get(key), dict)), None)
        if first_dict is None:
            continue
        expected_subkeys = set(first_dict.keys())
        bad = []
        for idx, entry in enumerate(data):
            value = entry.get(key)
            if value is None:
                continue
            if not isinstance(value, dict):
                bad.append((idx, "not dict"))
            elif set(value.keys()) != expected_subkeys:
                bad.append((idx, "subkeys changed"))
        if bad:
            report.add("ERROR", f"field '{key}' has inconsistent structure in {len(bad)} entries; first: {bad[:5]}")


def check_timestamps(data: list[dict], args: argparse.Namespace, report: LogReport) -> None:
    timestamps = []
    invalid = []
    for idx, entry in enumerate(data):
        value = entry.get("timestamp")
        if is_finite_number(value):
            timestamps.append(float(value))
        else:
            invalid.append(idx)
    if invalid:
        report.add("ERROR", f"timestamp is missing or non-finite in {len(invalid)} entries; first indices: {invalid[:5]}")
    if len(timestamps) < 2:
        return

    arr = np.asarray(timestamps, dtype=float)
    dt = np.diff(arr)
    non_positive = np.where(dt <= 0)[0]
    if len(non_positive):
        report.add("ERROR", f"timestamp is not strictly increasing at {len(non_positive)} locations; first indices: {non_positive[:5].tolist()}")

    positive_dt = dt[dt > 0]
    if not len(positive_dt):
        return
    report.duration_s = float(arr[-1] - arr[0])
    report.mean_hz = float((len(arr) - 1) / report.duration_s) if report.duration_s > 0 else None
    report.min_dt_s = float(positive_dt.min())
    report.max_dt_s = float(positive_dt.max())

    if report.mean_hz is not None and report.mean_hz < args.min_hz:
        report.add("WARN", f"average frequency is low: {report.mean_hz:.2f} Hz < {args.min_hz:.2f} Hz")
    if report.mean_hz is not None and report.mean_hz > args.max_hz:
        report.add("WARN", f"average frequency is high: {report.mean_hz:.2f} Hz > {args.max_hz:.2f} Hz")
    if report.max_dt_s > args.max_dt:
        report.add("WARN", f"large timestamp gap: max dt {report.max_dt_s:.3f}s > {args.max_dt:.3f}s")


def check_numeric_fields(data: list[dict], args: argparse.Namespace, report: LogReport) -> None:
    values_by_field: dict[str, list[dict[str, float]]] = defaultdict(list)
    invalid_by_field: dict[str, list[int]] = defaultdict(list)

    for idx, entry in enumerate(data):
        for key in NUMERIC_DICT_KEYS:
            if key not in entry:
                continue
            value = entry[key]
            if value is None:
                continue
            if not isinstance(value, dict):
                invalid_by_field[key].append(idx)
                continue
            flat = flatten_numeric_dict(value)
            if len(flat) != len(value):
                non_numeric = [name for name, item in value.items() if name not in flat and not is_finite_number(item)]
                if non_numeric:
                    invalid_by_field[key].append(idx)
            if flat:
                values_by_field[key].append(flat)

    for key, indices in invalid_by_field.items():
        report.add("ERROR", f"field '{key}' contains non-numeric/non-finite values in {len(indices)} entries; first indices: {indices[:5]}")

    for key, rows in values_by_field.items():
        all_names = sorted({name for row in rows for name in row})
        if not all_names:
            continue
        matrix = np.asarray([[row.get(name, np.nan) for name in all_names] for row in rows], dtype=float)
        finite_values = matrix[np.isfinite(matrix)]
        if finite_values.size == 0:
            continue
        report.numeric_summary[key] = {
            "min": float(finite_values.min()),
            "max": float(finite_values.max()),
            "mean": float(finite_values.mean()),
        }

        limit = args.velocity_limit if "velocity" in key else args.joint_limit
        if key in ("hand_command", "hand_trigger_raw"):
            low_bad = np.argwhere(matrix < -1e-6)
            high_bad = np.argwhere(matrix > 1.0 + 1e-6)
            if low_bad.size or high_bad.size:
                report.add("WARN", f"field '{key}' has values outside [0, 1]")
        elif np.nanmax(np.abs(matrix)) > limit:
            report.add("WARN", f"field '{key}' abs max {np.nanmax(np.abs(matrix)):.3f} exceeds limit {limit:.3f}")

        if key.endswith("state") or key.endswith("command"):
            jumps = np.abs(np.diff(matrix, axis=0))
            if jumps.size and np.nanmax(jumps) > args.max_jump:
                row, col = np.unravel_index(np.nanargmax(jumps), jumps.shape)
                report.add("WARN", f"field '{key}' jump {jumps[row, col]:.3f} at entry {row}->{row + 1}, joint '{all_names[col]}'")


def check_images(data: list[dict], args: argparse.Namespace, report: LogReport) -> None:
    first_image = next((entry.get("image") for entry in data if isinstance(entry.get("image"), dict)), None)
    if first_image is None:
        report.add("WARN", "no image dict found")
        return

    expected_cameras = set(first_image.keys())
    report.cameras = sorted(str(name) for name in expected_cameras)
    bad_structure = []
    missing_cameras = []
    stream_shapes: dict[str, tuple[int, ...]] = {}
    bad_decode = []
    empty_streams = []

    for idx, entry in enumerate(data):
        image = entry.get("image")
        if not isinstance(image, dict):
            bad_structure.append(idx)
            continue
        cameras = set(image.keys())
        if cameras != expected_cameras:
            missing_cameras.append(idx)

        for stream_name, raw in collect_image_streams(image).items():
            if raw is None:
                continue
            if isinstance(raw, bytes) and len(raw) == 0:
                empty_streams.append((idx, stream_name))
                continue
            if not args.decode_images:
                continue
            try:
                img = decode_image(raw)
            except Exception:
                bad_decode.append((idx, stream_name))
                continue
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                bad_decode.append((idx, stream_name))
                continue
            shape = tuple(int(x) for x in img.shape)
            previous_shape = stream_shapes.setdefault(stream_name, shape)
            if previous_shape != shape:
                report.add("WARN", f"image stream '{stream_name}' shape changed: {previous_shape} -> {shape} at entry {idx}")

    if bad_structure:
        report.add("ERROR", f"image field is not dict in {len(bad_structure)} entries; first indices: {bad_structure[:5]}")
    if missing_cameras:
        report.add("ERROR", f"camera keys are inconsistent in {len(missing_cameras)} entries; first indices: {missing_cameras[:5]}")
    if empty_streams:
        report.add("ERROR", f"empty image byte streams found; first: {empty_streams[:5]}")
    if bad_decode:
        report.add("ERROR", f"image decode failed; first: {bad_decode[:5]}")


def numeric_matrix(data: list[dict], key: str) -> tuple[list[str], np.ndarray] | None:
    names = None
    rows = []
    for entry in data:
        flat = flatten_numeric_dict(entry.get(key))
        if not flat:
            continue
        if names is None:
            names = sorted(flat)
        if set(flat) != set(names):
            continue
        rows.append([flat[name] for name in names])
    if names is None or len(rows) < 2:
        return None
    return names, np.asarray(rows, dtype=float)


def matrix_motion_stats(matrix: np.ndarray) -> dict[str, float]:
    diffs = np.abs(np.diff(matrix, axis=0))
    ranges = np.nanmax(matrix, axis=0) - np.nanmin(matrix, axis=0)
    return {
        "travel": float(np.nansum(diffs)),
        "mean_step": float(np.nanmean(diffs)) if diffs.size else 0.0,
        "max_step": float(np.nanmax(diffs)) if diffs.size else 0.0,
        "max_range": float(np.nanmax(ranges)) if ranges.size else 0.0,
        "mean_range": float(np.nanmean(ranges)) if ranges.size else 0.0,
    }


def check_motion_content(data: list[dict], args: argparse.Namespace, report: LogReport) -> None:
    command_result = numeric_matrix(data, "arm_command")
    state_result = numeric_matrix(data, "arm_state")
    if command_result is None or state_result is None:
        return

    _, command_matrix = command_result
    _, state_matrix = state_result
    command_stats = matrix_motion_stats(command_matrix)
    state_stats = matrix_motion_stats(state_matrix)
    report.motion_summary = {
        "arm_command_travel": command_stats["travel"],
        "arm_command_max_range": command_stats["max_range"],
        "arm_command_max_step": command_stats["max_step"],
        "arm_state_travel": state_stats["travel"],
        "arm_state_max_range": state_stats["max_range"],
        "arm_state_max_step": state_stats["max_step"],
    }

    static_by_travel = (
        command_stats["travel"] < args.static_action_travel
        and state_stats["travel"] < args.static_state_travel
    )
    static_by_range = (
        command_stats["max_range"] < args.static_command_range
        and state_stats["max_range"] < args.static_state_range
    )
    if static_by_travel or static_by_range:
        report.mostly_static = True
        report.mostly_static_reason = (
            f"arm_command travel={command_stats['travel']:.4f}, max_range={command_stats['max_range']:.4f}; "
            f"arm_state travel={state_stats['travel']:.4f}, max_range={state_stats['max_range']:.4f}"
        )


def print_report(report: LogReport, max_issues: int) -> None:
    status = "FAIL" if report.error_count else "OK"
    print(f"\n[{status}] {report.path}")
    print(f"  entries: {report.entry_count}")
    if report.duration_s is not None:
        print(
            "  timing: "
            f"duration={report.duration_s:.3f}s, mean={report.mean_hz:.2f}Hz, "
            f"dt=[{report.min_dt_s:.4f}, {report.max_dt_s:.4f}]s"
        )
    if report.cameras:
        print(f"  cameras: {', '.join(report.cameras)}")
    for key, stats in sorted(report.numeric_summary.items()):
        print(f"  {key}: min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}")
    if report.motion_summary:
        print(
            "  motion: "
            f"arm_command travel={report.motion_summary['arm_command_travel']:.4f}, "
            f"range={report.motion_summary['arm_command_max_range']:.4f}; "
            f"arm_state travel={report.motion_summary['arm_state_travel']:.4f}, "
            f"range={report.motion_summary['arm_state_max_range']:.4f}"
        )
    if report.mostly_static:
        print(f"  observation: mostly static ({report.mostly_static_reason})")

    print(f"  issues: {report.error_count} error(s), {report.warning_count} warning(s)")
    for issue in report.issues[:max_issues]:
        print(f"    {issue.level}: {issue.message}")
    if len(report.issues) > max_issues:
        print(f"    ... {len(report.issues) - max_issues} more issue(s)")


def main() -> int:
    args = parse_args()
    log_files = expand_paths(args.paths)
    if not log_files:
        print("ERROR: no .pkl log files found", file=sys.stderr)
        return 1

    reports = [check_log(path, args) for path in log_files]
    for report in reports:
        print_report(report, args.max_issues)

    total_errors = sum(report.error_count for report in reports)
    total_warnings = sum(report.warning_count for report in reports)
    print(f"\nChecked {len(reports)} file(s): {total_errors} error(s), {total_warnings} warning(s)")
    static_reports = [report for report in reports if report.mostly_static]
    if static_reports:
        print("\nMostly static logs:")
        for report in static_reports:
            print(f"  {report.path.name}: {report.mostly_static_reason}")
    else:
        print("\nMostly static logs: none")

    if total_errors:
        return 1
    if args.strict_warnings and total_warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
