#!/usr/bin/env python3
"""
Replay collected X2 upper-body data back to the real robot through ROS2 command topics.

This script is intentionally dry-run by default. It only publishes commands when
--execute-hardware is passed.

Examples:
  # Dry-run a raw log.
  python3 scripts/hardware/replay_x2_collected_data_on_hardware.py logs/x2_upper_body_hardware/teleop_log_*.pkl

  # Move hardware. Start with low speed and keep one hand near the e-stop.
  python3 scripts/hardware/replay_x2_collected_data_on_hardware.py \
    logs/x2_upper_body_hardware/teleop_log_YYYYMMDD_HHMMSS_1.pkl \
    --execute-hardware \
    --move-to-start \
    --speed 0.5

  # Replay a converted LeRobot v3 dataset directory.
  python3 scripts/hardware/replay_x2_collected_data_on_hardware.py \
    datasets/x2_hardware_lerobot_v3 \
    --execute-hardware \
    --move-to-start
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_ARM_STATE_TOPIC = "/aima/hal/joint/arm/state"
DEFAULT_ARM_COMMAND_TOPIC = "/aima/hal/joint/arm/command"
DEFAULT_HEAD_STATE_TOPIC = "/aima/hal/joint/head/state"
DEFAULT_HEAD_COMMAND_TOPIC = "/aima/hal/joint/head/command"
DEFAULT_HAND_COMMAND_TOPIC = "/aima/hal/joint/hand/command"


ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]

HEAD_JOINT_NAMES = ["head_yaw_joint", "head_pitch_joint"]
HAND_NAMES = ["left_hand", "right_hand"]


@dataclass
class ReplayTrajectory:
    source: Path
    mode: str
    fps: float
    timestamps: list[float]
    arm_targets: list[dict[str, float]]
    head_targets: list[dict[str, float]]
    hand_targets: list[dict[str, float]]

    @property
    def length(self) -> int:
        return len(self.arm_targets)


@dataclass(frozen=True)
class JointCommandSpec:
    lower_limit: float
    upper_limit: float
    kp: float
    kd: float


DEFAULT_HARDWARE_COMMAND_SPECS = {
    "left_shoulder_pitch_joint": JointCommandSpec(-3.08, 2.04, 40.0, 4.0),
    "left_shoulder_roll_joint": JointCommandSpec(-0.061, 2.993, 40.0, 4.0),
    "left_shoulder_yaw_joint": JointCommandSpec(-2.556, 2.556, 40.0, 4.0),
    "left_elbow_joint": JointCommandSpec(-2.3556, 0.0, 40.0, 4.0),
    "left_wrist_yaw_joint": JointCommandSpec(-2.556, 2.556, 20.0, 2.0),
    "left_wrist_pitch_joint": JointCommandSpec(-0.558, 0.558, 20.0, 2.0),
    "left_wrist_roll_joint": JointCommandSpec(-1.571, 0.724, 20.0, 2.0),
    "right_shoulder_pitch_joint": JointCommandSpec(-3.08, 2.04, 40.0, 4.0),
    "right_shoulder_roll_joint": JointCommandSpec(-2.993, 0.061, 40.0, 4.0),
    "right_shoulder_yaw_joint": JointCommandSpec(-2.556, 2.556, 40.0, 4.0),
    "right_elbow_joint": JointCommandSpec(-2.3556, 0.0, 40.0, 4.0),
    "right_wrist_yaw_joint": JointCommandSpec(-2.556, 2.556, 20.0, 2.0),
    "right_wrist_pitch_joint": JointCommandSpec(-0.558, 0.558, 20.0, 2.0),
    "right_wrist_roll_joint": JointCommandSpec(-0.724, 1.571, 20.0, 2.0),
    "head_yaw_joint": JointCommandSpec(-0.366, 0.366, 20.0, 2.0),
    "head_pitch_joint": JointCommandSpec(-0.3838, 0.3838, 20.0, 2.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="Raw .pkl log(s), directories, glob patterns, or a LeRobot dataset dir.")
    parser.add_argument("--execute-hardware", action="store_true", help="Actually publish ROS2 commands to the real robot.")
    parser.add_argument("--source-field", choices=("arm_command", "arm_state", "action", "observation.state"), default="arm_command", help="Preferred replay source. Raw logs use arm_command/arm_state; LeRobot uses action/observation.state.")
    parser.add_argument("--fps", type=float, default=None, help="Override replay FPS. Defaults to timestamps or dataset metadata.")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier.")
    parser.add_argument("--start-index", type=int, default=0, help="First frame to replay.")
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive end frame. Defaults to the trajectory end.")
    parser.add_argument("--loop", action="store_true", help="Loop playback until Ctrl+C.")
    parser.add_argument("--include-head", action="store_true", help="Replay head_command/head_state when available.")
    parser.add_argument(
        "--include-hands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay left_hand/right_hand when available. Use --no-include-hands to disable.",
    )
    parser.add_argument("--move-to-start", action="store_true", help="Gradually move from current robot state to the first replay frame before playback.")
    parser.add_argument("--move-to-start-duration-s", type=float, default=3.0, help="Duration for --move-to-start.")
    parser.add_argument("--max-initial-error-rad", type=float, default=0.35, help="Abort hardware replay if the first frame is farther than this and --move-to-start is not set.")
    parser.add_argument("--max-step-rad", type=float, default=0.08, help="Maximum per-joint command step between publish cycles.")
    parser.add_argument("--initial-state-timeout-s", type=float, default=10.0, help="Timeout waiting for arm state topic.")
    parser.add_argument("--arm-state-topic", default=DEFAULT_ARM_STATE_TOPIC)
    parser.add_argument("--arm-command-topic", default=DEFAULT_ARM_COMMAND_TOPIC)
    parser.add_argument("--head-state-topic", default=DEFAULT_HEAD_STATE_TOPIC)
    parser.add_argument("--head-command-topic", default=DEFAULT_HEAD_COMMAND_TOPIC)
    parser.add_argument("--hand-command-topic", default=DEFAULT_HAND_COMMAND_TOPIC)
    return parser.parse_args()


def is_lerobot_dir(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def expand_paths(paths: list[Path]) -> list[Path]:
    expanded = []
    for raw in paths:
        text = str(raw)
        matches = [Path(match) for match in sorted(glob.glob(text))] if any(ch in text for ch in "*?[]") else [raw]
        for path in matches:
            if path.is_dir() and is_lerobot_dir(path):
                expanded.append(path)
            elif path.is_dir():
                expanded.extend(sorted(path.rglob("*.pkl")))
            elif path.is_file():
                expanded.append(path)
            else:
                print(f"WARN: path does not exist or is not readable: {path}", file=sys.stderr)
    return sorted(dict.fromkeys(path.resolve() for path in expanded))


def derive_fps(timestamps: list[float], fallback: float = 15.0) -> float:
    if len(timestamps) < 2:
        return fallback
    dt = np.diff(np.asarray(timestamps, dtype=np.float64))
    dt = dt[dt > 1.0e-6]
    return float(1.0 / np.median(dt)) if dt.size else fallback


def finite_numeric_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for name, item in value.items():
        if isinstance(item, (int, float, np.number)) and np.isfinite(float(item)):
            result[str(name)] = float(item)
    return result


def load_raw_logs(paths: list[Path], args: argparse.Namespace) -> ReplayTrajectory:
    entries = []
    for path in paths:
        with path.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} top-level object must be a list.")
        entries.extend(entry for entry in data if isinstance(entry, dict))
    if not entries:
        raise ValueError("No valid log entries found.")

    source_field = args.source_field if args.source_field in ("arm_command", "arm_state") else "arm_command"
    arm_targets = []
    head_targets = []
    hand_targets = []
    timestamps = []
    for index, entry in enumerate(entries):
        arm = finite_numeric_dict(entry.get(source_field))
        if not arm and source_field == "arm_command":
            arm = finite_numeric_dict(entry.get("arm_state"))
        arm = {name: arm[name] for name in ARM_JOINT_NAMES if name in arm}
        if len(arm) != len(ARM_JOINT_NAMES):
            continue
        arm_targets.append(arm)

        head_key = "head_command" if source_field == "arm_command" else "head_state"
        head = finite_numeric_dict(entry.get(head_key))
        head_targets.append({name: head[name] for name in HEAD_JOINT_NAMES if name in head})

        hand = finite_numeric_dict(entry.get("hand_command"))
        hand_targets.append({name: hand[name] for name in HAND_NAMES if name in hand})
        timestamps.append(float(entry.get("timestamp", index)))

    if not arm_targets:
        raise ValueError(f"No complete arm targets found from '{source_field}'.")
    return ReplayTrajectory(
        source=paths[0] if len(paths) == 1 else paths[0].parent,
        mode=f"raw-pkl:{source_field}",
        fps=args.fps or derive_fps(timestamps),
        timestamps=timestamps,
        arm_targets=arm_targets,
        head_targets=head_targets,
        hand_targets=hand_targets,
    )


def strip_pos_suffix(name: str) -> str:
    return name[:-4] if name.endswith(".pos") else name


def flatten_column(values: Any) -> list[list[float]]:
    rows = []
    for item in values or []:
        if isinstance(item, np.ndarray):
            rows.append([float(x) for x in item.reshape(-1)])
        elif isinstance(item, (list, tuple)):
            rows.append([float(x) for x in item])
        else:
            rows.append([float(item)])
    return rows


def load_lerobot_dataset(path: Path, args: argparse.Namespace) -> ReplayTrajectory:
    if pq is None:
        raise SystemExit(
            "Missing dependency: pyarrow is required to read LeRobot parquet data.\n"
            "Install it in the active environment with:\n"
            "  PYTHONNOUSERSITE=1 python3 -m pip install pyarrow"
        )
    info = json.loads((path / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info.get("features", {})
    source_field = args.source_field if args.source_field in ("action", "observation.state") else "action"
    feature = features.get(source_field, {})
    names = [strip_pos_suffix(str(name)) for name in (feature.get("names") or [])]
    if not names:
        raise ValueError(f"Dataset feature '{source_field}' has no names in meta/info.json.")

    data_files = sorted((path / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise ValueError(f"No parquet files found under {path / 'data'}")
    tables = [pq.read_table(file_path) for file_path in data_files]
    table = tables[0] if len(tables) == 1 else __import__("pyarrow").concat_tables(tables)
    columns = table.to_pydict()
    rows = flatten_column(columns[source_field])
    timestamps = [row[0] if row else float(index) for index, row in enumerate(flatten_column(columns.get("timestamp", [])))]
    if not timestamps:
        timestamps = [float(index) for index in range(len(rows))]

    arm_targets = []
    hand_targets = []
    for values in rows:
        row = {name: float(values[idx]) for idx, name in enumerate(names) if idx < len(values)}
        arm = {name: row[name] for name in ARM_JOINT_NAMES if name in row}
        if len(arm) != len(ARM_JOINT_NAMES):
            continue
        arm_targets.append(arm)
        hand_targets.append({name: row[name] for name in HAND_NAMES if name in row})

    if not arm_targets:
        raise ValueError(f"No complete arm targets found from dataset feature '{source_field}'.")
    return ReplayTrajectory(
        source=path,
        mode=f"lerobot-v3:{source_field}",
        fps=args.fps or float(info.get("fps") or derive_fps(timestamps)),
        timestamps=timestamps[: len(arm_targets)],
        arm_targets=arm_targets,
        head_targets=[{} for _ in arm_targets],
        hand_targets=hand_targets,
    )


def clip_targets(targets: dict[str, float]) -> dict[str, float]:
    clipped = {}
    for joint_name, value in targets.items():
        spec = DEFAULT_HARDWARE_COMMAND_SPECS[joint_name]
        clipped[joint_name] = float(np.clip(value, spec.lower_limit, spec.upper_limit))
    return clipped


def limit_step(targets: dict[str, float], previous: dict[str, float], max_step: float) -> dict[str, float]:
    if max_step <= 0.0:
        return dict(targets)
    return {
        joint_name: float(np.clip(target, previous[joint_name] - max_step, previous[joint_name] + max_step))
        for joint_name, target in targets.items()
    }


def max_abs_error(a: dict[str, float], b: dict[str, float]) -> float:
    shared = [name for name in a if name in b]
    return max((abs(float(a[name]) - float(b[name])) for name in shared), default=0.0)


def publish_arm(interface, targets: dict[str, float], prev_targets: dict[str, float], dt: float) -> None:
    velocities = {name: (targets[name] - prev_targets[name]) / max(dt, 1.0e-6) for name in targets}
    interface.publish_command(
        joint_targets=targets,
        joint_velocities=velocities,
        command_specs=DEFAULT_HARDWARE_COMMAND_SPECS,
    )


def publish_head(interface, targets: dict[str, float], prev_targets: dict[str, float], dt: float) -> None:
    velocities = {name: (targets[name] - prev_targets.get(name, targets[name])) / max(dt, 1.0e-6) for name in targets}
    interface.publish_command(
        joint_targets=targets,
        joint_velocities=velocities,
        command_specs=DEFAULT_HARDWARE_COMMAND_SPECS,
    )


def setup_ros_interfaces(args: argparse.Namespace):
    from xrobotoolkit_teleop.hardware.x2_ros2_teleop_controller import (
        MultiThreadedExecutor,
        Ros2HandCommandInterface,
        Ros2JointGroupInterface,
        _ROS2_IMPORT_ERROR,
        rclpy,
    )

    if _ROS2_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ROS2 dependencies are unavailable. Please source /opt/ros/humble/setup.bash "
            "and ros2_ws/install/setup.bash before running hardware replay."
        ) from _ROS2_IMPORT_ERROR
    if not rclpy.ok():
        rclpy.init(args=None)

    arm_interface = Ros2JointGroupInterface(
        node_name="x2_collected_data_replay_arm",
        state_topic=args.arm_state_topic,
        command_topic=args.arm_command_topic,
        joint_names=ARM_JOINT_NAMES,
    )
    head_interface = Ros2JointGroupInterface(
        node_name="x2_collected_data_replay_head",
        state_topic=args.head_state_topic,
        command_topic=args.head_command_topic,
        joint_names=HEAD_JOINT_NAMES,
        enable_state_subscription=False,
    )
    hand_interface = Ros2HandCommandInterface(
        node_name="x2_collected_data_replay_hand",
        command_topic=args.hand_command_topic,
    )
    executor = MultiThreadedExecutor()
    executor.add_node(arm_interface)
    executor.add_node(head_interface)
    executor.add_node(hand_interface)
    spin_thread = threading.Thread(
        target=executor.spin,
        name="x2_collected_data_replay_ros2_spin",
        daemon=True,
    )
    spin_thread.start()
    return rclpy, executor, spin_thread, arm_interface, head_interface, hand_interface


def cleanup_ros(rclpy_module, executor, spin_thread, *interfaces) -> None:
    if executor is not None:
        for interface in interfaces:
            if interface is not None:
                try:
                    executor.remove_node(interface)
                except Exception:
                    pass
        executor.shutdown()
    if spin_thread is not None and spin_thread.is_alive():
        spin_thread.join(timeout=2.0)
    for interface in interfaces:
        if interface is not None:
            interface.destroy_node()
    if rclpy_module is not None and rclpy_module.ok():
        rclpy_module.shutdown()


def move_to_start(arm_interface, start_targets: dict[str, float], duration_s: float, fps: float, max_step: float) -> dict[str, float]:
    current = arm_interface.get_joint_positions()
    current = {name: current.get(name, 0.0) for name in ARM_JOINT_NAMES}
    steps = max(1, int(max(duration_s, 0.1) * max(fps, 1.0)))
    dt = 1.0 / max(fps, 1.0)
    previous = dict(current)
    print(f"Moving to first replay frame over {duration_s:.2f}s ({steps} steps)...")
    for step in range(1, steps + 1):
        alpha = step / steps
        target = {name: current[name] + alpha * (start_targets[name] - current[name]) for name in ARM_JOINT_NAMES}
        target = limit_step(clip_targets(target), previous, max_step)
        publish_arm(arm_interface, target, previous, dt)
        previous = target
        time.sleep(dt)
    return previous


def dry_run_summary(traj: ReplayTrajectory, args: argparse.Namespace) -> None:
    start = max(0, int(args.start_index))
    end = min(traj.length, int(args.end_index) if args.end_index is not None else traj.length)
    arm_matrix = np.asarray([[frame[name] for name in ARM_JOINT_NAMES] for frame in traj.arm_targets[start:end]], dtype=float)
    diffs = np.abs(np.diff(arm_matrix, axis=0))
    ranges = np.nanmax(arm_matrix, axis=0) - np.nanmin(arm_matrix, axis=0)
    print(f"\nReplay source: {traj.source}")
    print(f"  mode: {traj.mode}")
    print(f"  frames: {start}:{end} ({end - start})")
    print(f"  fps: {traj.fps:.3f}, speed: {args.speed:.3f}x")
    print(f"  execute_hardware: {args.execute_hardware}")
    print(f"  arm max_range={float(np.nanmax(ranges)):.4f}, max_step={float(np.nanmax(diffs)) if diffs.size else 0.0:.4f}")
    print(f"  include_head={args.include_head}, include_hands={args.include_hands}")
    hand_frames = sum(1 for hand in traj.hand_targets[start:end] if any(name in hand for name in HAND_NAMES))
    if args.include_hands:
        print(f"  hand command frames: {hand_frames}/{end - start}")
        if hand_frames == 0:
            print(
                "  WARN: no left_hand/right_hand targets found. "
                "Raw .pkl replay needs hand_command; LeRobot replay needs action names left_hand/right_hand."
            )


def replay_hardware(traj: ReplayTrajectory, args: argparse.Namespace) -> None:
    start = max(0, int(args.start_index))
    end = min(traj.length, int(args.end_index) if args.end_index is not None else traj.length)
    if start >= end:
        raise ValueError(f"Invalid replay range: {start}:{end}")

    rclpy_module = executor = spin_thread = arm_interface = head_interface = hand_interface = None
    try:
        rclpy_module, executor, spin_thread, arm_interface, head_interface, hand_interface = setup_ros_interfaces(args)
        print("Waiting for current arm state...")
        if not arm_interface.wait_for_state(args.initial_state_timeout_s):
            raise RuntimeError(
                f"Timed out waiting for arm state topic: {args.arm_state_topic}\n"
                "Please check that the robot state publisher is running and that the topic name is correct:\n"
                f"  ros2 topic echo --once {args.arm_state_topic}\n"
                "If your robot uses another topic, pass it with --arm-state-topic."
            )

        first_targets = clip_targets(traj.arm_targets[start])
        current = arm_interface.get_joint_positions()
        initial_error = max_abs_error(first_targets, current)
        if initial_error > args.max_initial_error_rad and not args.move_to_start:
            raise RuntimeError(
                f"First replay frame is far from current arm state: {initial_error:.3f} rad. "
                "Use --move-to-start if this is expected."
            )

        replay_fps = max(traj.fps * args.speed, 1.0e-6)
        dt = 1.0 / replay_fps
        previous_arm = (
            move_to_start(
                arm_interface=arm_interface,
                start_targets=first_targets,
                duration_s=args.move_to_start_duration_s,
                fps=replay_fps,
                max_step=args.max_step_rad,
            )
            if args.move_to_start
            else {name: current.get(name, first_targets[name]) for name in ARM_JOINT_NAMES}
        )
        previous_head = {name: 0.0 for name in HEAD_JOINT_NAMES}

        print("Starting hardware replay. Press Ctrl+C to stop.")
        while True:
            for index in range(start, end):
                target = limit_step(clip_targets(traj.arm_targets[index]), previous_arm, args.max_step_rad)
                publish_arm(arm_interface, target, previous_arm, dt)
                previous_arm = target

                if args.include_head and traj.head_targets[index]:
                    head_target = clip_targets({name: traj.head_targets[index][name] for name in HEAD_JOINT_NAMES if name in traj.head_targets[index]})
                    if len(head_target) == len(HEAD_JOINT_NAMES):
                        head_target = limit_step(head_target, previous_head, args.max_step_rad)
                        publish_head(head_interface, head_target, previous_head, dt)
                        previous_head = head_target

                if args.include_hands and traj.hand_targets[index]:
                    hand = traj.hand_targets[index]
                    hand_interface.publish_command(
                        left_position=float(np.clip(hand.get("left_hand", 1.0), 0.0, 1.0)),
                        right_position=float(np.clip(hand.get("right_hand", 1.0), 0.0, 1.0)),
                    )

                time.sleep(dt)
            if not args.loop:
                break
        print("Hardware replay finished.")
    except KeyboardInterrupt:
        print("\nHardware replay interrupted.")
    finally:
        cleanup_ros(rclpy_module, executor, spin_thread, arm_interface, head_interface, hand_interface)


def main() -> int:
    args = parse_args()
    paths = expand_paths(args.paths)
    if not paths:
        print("ERROR: no input files or datasets found", file=sys.stderr)
        return 1
    if len(paths) == 1 and is_lerobot_dir(paths[0]):
        traj = load_lerobot_dataset(paths[0], args)
    else:
        pkl_paths = [path for path in paths if path.suffix == ".pkl"]
        if not pkl_paths:
            print("ERROR: expected .pkl log files or a LeRobot dataset directory", file=sys.stderr)
            return 1
        traj = load_raw_logs(pkl_paths, args)

    dry_run_summary(traj, args)
    if not args.execute_hardware:
        print("\nDry-run only. Add --execute-hardware to publish commands to the real robot.")
        return 0
    replay_hardware(traj, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
