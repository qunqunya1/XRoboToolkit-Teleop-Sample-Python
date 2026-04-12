#!/usr/bin/env python3

"""Quick diagnostic for XRoboToolkit/Pico connectivity.

This script checks three layers:
1. Can the bundled XR SDK shared libraries be loaded?
2. Can the SDK connect to the local XRoboToolkit PC Service?
3. Is live Pico/XR data actually arriving (poses, buttons, timestamps)?
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _preload_sdk_library(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "dependencies" / "XRoboToolkit-PC-Service-Pybind" / "lib" / "libPXREARobotSDK.so",
        repo_root
        / "dependencies"
        / "XRoboToolkit-PC-Service-Pybind"
        / "build"
        / "lib.linux-x86_64-cpython-310"
        / "libPXREARobotSDK.so",
    ]
    for lib_path in candidates:
        if lib_path.exists():
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            return lib_path
    return None


def _import_sdk(repo_root: Path):
    vendor_dir = repo_root / ".vendor"
    if vendor_dir.exists():
        sys.path.insert(0, str(vendor_dir))
    return __import__("xrobotoolkit_sdk")


def _is_nonzero_sequence(values) -> bool:
    return any(abs(float(v)) > 1e-9 for v in values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether Pico/XR data is reaching the teleop stack.")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to collect.")
    parser.add_argument("--interval", type=float, default=0.3, help="Seconds between samples.")
    args = parser.parse_args()

    repo_root = _find_repo_root()
    print(f"repo_root={repo_root}")

    preloaded = _preload_sdk_library(repo_root)
    if preloaded is None:
        print("ERROR: libPXREARobotSDK.so was not found under dependencies/.")
        return 2
    print(f"preloaded_sdk_lib={preloaded}")

    try:
        xrt = _import_sdk(repo_root)
    except Exception as exc:
        print(f"ERROR: failed to import xrobotoolkit_sdk: {type(exc).__name__}: {exc}")
        return 2

    print(f"sdk_module={getattr(xrt, '__file__', '<unknown>')}")

    try:
        xrt.init()
    except Exception as exc:
        print(f"ERROR: SDK init failed: {type(exc).__name__}: {exc}")
        return 3

    print("sdk_init=OK")

    samples = []
    try:
        for idx in range(args.samples):
            sample = {
                "left_pose": xrt.get_left_controller_pose(),
                "right_pose": xrt.get_right_controller_pose(),
                "head_pose": xrt.get_headset_pose(),
                "left_grip": xrt.get_left_grip(),
                "right_grip": xrt.get_right_grip(),
                "left_trigger": xrt.get_left_trigger(),
                "right_trigger": xrt.get_right_trigger(),
                "timestamp_ns": xrt.get_time_stamp_ns(),
                "motion_trackers": xrt.num_motion_data_available(),
            }
            samples.append(sample)
            print(f"sample[{idx}]={sample}")
            time.sleep(args.interval)
    finally:
        xrt.close()
        print("sdk_close=OK")

    any_timestamp = any(int(sample["timestamp_ns"]) > 0 for sample in samples)
    any_pose = any(
        _is_nonzero_sequence(sample["left_pose"])
        or _is_nonzero_sequence(sample["right_pose"])
        or _is_nonzero_sequence(sample["head_pose"])
        for sample in samples
    )
    any_input = any(
        float(sample["left_grip"]) > 0.0
        or float(sample["right_grip"]) > 0.0
        or float(sample["left_trigger"]) > 0.0
        or float(sample["right_trigger"]) > 0.0
        for sample in samples
    )
    any_motion_tracker = any(int(sample["motion_trackers"]) > 0 for sample in samples)

    print("\nSummary:")
    print(f"  timestamp_active={any_timestamp}")
    print(f"  pose_nonzero={any_pose}")
    print(f"  controller_input_nonzero={any_input}")
    print(f"  motion_tracker_present={any_motion_tracker}")

    if any_timestamp and any_pose:
        print("PASS: SDK is connected and live XR pose data is arriving.")
        if not any_input:
            print("NOTE: No trigger/grip activity was observed during sampling.")
        return 0

    print("FAIL: SDK connected, but live Pico/XR data does not look valid yet.")
    print("Hints:")
    print("  1. Make sure XRoboToolkit PC Service is running.")
    print("  2. Make sure the Pico headset/controllers are connected to the service.")
    print("  3. Open the headset-side app/streaming pipeline expected by the service.")
    print("  4. Press and hold a grip button while sampling; this teleop only activates an arm when grip > 0.9.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
