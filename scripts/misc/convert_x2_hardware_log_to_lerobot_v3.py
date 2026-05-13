#!/usr/bin/env python3
"""
Convert X2 hardware teleop logs (.pkl) into a LeRobot v3-style dataset layout.

This converter is intentionally opinionated for the current X2 hardware logger:
- Uses arm state only for `observation.state` by default.
- Uses arm command plus hand command for `action` by default.
- Excludes head state/command unless `--include-head` is enabled.
- Exports camera frames as MP4 files under `videos/<camera>/chunk-000/file-000.mp4`.
- Writes a single data shard and a single episode metadata shard.

Expected input log entry format:
  {
    "timestamp": float,
    "arm_state": {joint_name: position, ...},
    "arm_command": {joint_name: position, ...},
    "head_state": {joint_name: position, ...},      # optional
    "head_command": {joint_name: position, ...},    # optional
    "hand_command": {"left_hand": position, "right_hand": position},  # optional
    "image": {
      "head_front": {"color": <jpg-bytes-or-numpy-array>, "depth": ...},
      ...
    }
  }
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Sequence

import cv2
import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset
except ImportError:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset
    except ImportError:
        _LeRobotDataset = None


DEFAULT_HAND_COMMAND_NAMES = ["left_hand", "right_hand"]
DEFAULT_HAND_COMMAND_VALUE = 1.0


def _load_log(path: Path) -> list[dict]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Log file {path} is empty or invalid.")
    return data


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray([float(v) for v in values], dtype=np.float32)


def _decode_image(raw):
    if raw is None:
        return None
    if isinstance(raw, dict) and "raw" in raw:
        return _decode_raw_image_dict(raw)
    if isinstance(raw, bytes):
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode JPG image bytes.")
        return img

    img = np.asarray(raw)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    raise ValueError(f"Unsupported image format with shape {img.shape}.")


def _raw_encoding_to_dtype_channels(encoding: str):
    normalized = str(encoding).lower()
    if normalized in {"bgr8", "rgb8"}:
        return np.uint8, 3
    if normalized in {"mono8", "8uc1"}:
        return np.uint8, 1
    if normalized in {"mono16", "16uc1"}:
        return np.uint16, 1
    if normalized == "32fc1":
        return np.float32, 1
    raise ValueError(f"Unsupported raw image encoding: {encoding}")


def _decode_raw_image_dict(raw: dict):
    dtype, channels = _raw_encoding_to_dtype_channels(raw.get("encoding", ""))
    width = int(raw["width"])
    height = int(raw["height"])
    step = int(raw.get("step") or width * channels * np.dtype(dtype).itemsize)
    itemsize = np.dtype(dtype).itemsize
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
    if str(raw.get("encoding", "")).lower() == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _first_camera_frame(entry: dict, camera_name: str):
    image_dict = entry.get("image", {})
    if not isinstance(image_dict, dict):
        return None
    camera_data = image_dict.get(camera_name)
    if camera_data is None:
        return None
    if isinstance(camera_data, dict):
        return _decode_image(camera_data.get("color"))
    return _decode_image(camera_data)


def _discover_joint_names(entries: Sequence[dict], key: str) -> list[str]:
    for entry in entries:
        data = entry.get(key)
        if isinstance(data, dict) and data:
            return list(data.keys())
    return []


def _hand_command_values(entry: dict, hand_names: Sequence[str]) -> list[float]:
    hand_command = entry.get("hand_command", {})
    if not isinstance(hand_command, dict):
        hand_command = {}
    hand_trigger_raw = entry.get("hand_trigger_raw", {})
    if not isinstance(hand_trigger_raw, dict):
        hand_trigger_raw = {}
    return [
        float(hand_command.get(name, hand_trigger_raw.get(name, DEFAULT_HAND_COMMAND_VALUE)))
        for name in hand_names
    ]


def _has_hand_command(entry: dict, hand_names: Sequence[str] = DEFAULT_HAND_COMMAND_NAMES) -> bool:
    for key in ("hand_command", "hand_trigger_raw"):
        value = entry.get(key)
        if isinstance(value, dict) and any(name in value for name in hand_names):
            return True
    return False


def _feature_stats(array: np.ndarray) -> dict:
    return {
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
    }


def _build_episode_table(episodes_meta: list[dict]) -> pa.Table:
    columns = {
        "episode_index": pa.array([item["episode_index"] for item in episodes_meta], type=pa.int64()),
        "tasks": pa.array([item["tasks"] for item in episodes_meta], type=pa.list_(pa.string())),
        "length": pa.array([item["length"] for item in episodes_meta], type=pa.int64()),
        "data_chunk_index": pa.array([item["data_chunk_index"] for item in episodes_meta], type=pa.int64()),
        "data_file_index": pa.array([item["data_file_index"] for item in episodes_meta], type=pa.int64()),
        "dataset_from_index": pa.array([item["dataset_from_index"] for item in episodes_meta], type=pa.int64()),
        "dataset_to_index": pa.array([item["dataset_to_index"] for item in episodes_meta], type=pa.int64()),
    }

    video_keys = sorted(
        {
            key[len("video_") : -len("_chunk_index")]
            for item in episodes_meta
            for key in item.keys()
            if key.startswith("video_") and key.endswith("_chunk_index")
        }
    )
    for video_key in video_keys:
        columns[f"video_{video_key}_chunk_index"] = pa.array(
            [item.get(f"video_{video_key}_chunk_index", -1) for item in episodes_meta],
            type=pa.int64(),
        )
        columns[f"video_{video_key}_file_index"] = pa.array(
            [item.get(f"video_{video_key}_file_index", -1) for item in episodes_meta],
            type=pa.int64(),
        )
        columns[f"video_{video_key}_from_index"] = pa.array(
            [item.get(f"video_{video_key}_from_index", -1) for item in episodes_meta],
            type=pa.int64(),
        )
        columns[f"video_{video_key}_to_index"] = pa.array(
            [item.get(f"video_{video_key}_to_index", -1) for item in episodes_meta],
            type=pa.int64(),
        )
    return pa.table(columns)


def _build_tasks_table(task_names: Sequence[str]) -> pa.Table:
    return pa.table(
        {
            "task_index": pa.array(list(range(len(task_names))), type=pa.int64()),
            "task": pa.array(list(task_names), type=pa.string()),
        }
    )


def _build_feature_spec(
    obs_names: Sequence[str],
    action_names: Sequence[str],
    camera_names: Sequence[str],
    video_shapes: dict[str, tuple[int, int]],
) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(obs_names),),
            "names": [f"{name}.pos" for name in obs_names],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": [f"{name}.pos" for name in action_names],
        },
        "timestamp": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
    }
    for camera_name in camera_names:
        video_shape = video_shapes.get(camera_name)
        if video_shape is None:
            continue
        height, width = video_shape
        features[f"observation.images.{camera_name}"] = {
            "dtype": "image",
            "shape": (int(height), int(width), 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _try_convert_with_lerobot_sdk(
    log_files: Sequence[Path],
    output_dir: Path,
    task_name: str,
    fps: float,
    repo_id: str,
    camera_names: Sequence[str],
    include_head: bool,
    include_hands: bool,
) -> bool:
    if _LeRobotDataset is None:
        return False

    first_entry = None
    obs_names = None
    action_names = None
    video_shapes: dict[str, tuple[int, int]] = {}
    for log_file in log_files:
        entries = _load_log(log_file)
        arm_state_names = _discover_joint_names(entries, "arm_state")
        arm_command_names = _discover_joint_names(entries, "arm_command")
        head_state_names = _discover_joint_names(entries, "head_state") if include_head else []
        head_command_names = _discover_joint_names(entries, "head_command") if include_head else []
        hand_command_names = list(DEFAULT_HAND_COMMAND_NAMES) if include_hands else []
        if not arm_state_names or not arm_command_names:
            continue
        obs_names = list(arm_state_names) + list(head_state_names)
        action_names = list(arm_command_names) + list(head_command_names) + hand_command_names
        for entry in entries:
            for camera_name in camera_names:
                if camera_name in video_shapes:
                    continue
                frame = _first_camera_frame(entry, camera_name)
                if frame is not None:
                    video_shapes[camera_name] = tuple(frame.shape[:2])
            if len(video_shapes) == len(camera_names):
                break
        first_entry = entries[0]
        break

    if first_entry is None or obs_names is None or action_names is None:
        raise ValueError("No frames available for SDK export.")
    image_keys = {
        camera_name: f"observation.images.{camera_name}"
        for camera_name in camera_names
        if camera_name in video_shapes
    }

    dataset = _LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=float(fps),
        robot_type="x2_upper_body_hardware",
        features=_build_feature_spec(
            obs_names=obs_names,
            action_names=action_names,
            camera_names=camera_names,
            video_shapes=video_shapes,
        ),
        use_videos=bool(image_keys),
    )

    try:
        for log_file in log_files:
            entries = _load_log(log_file)
            arm_state_names = _discover_joint_names(entries, "arm_state")
            arm_command_names = _discover_joint_names(entries, "arm_command")
            head_state_names = _discover_joint_names(entries, "head_state") if include_head else []
            head_command_names = _discover_joint_names(entries, "head_command") if include_head else []
            hand_command_names = list(DEFAULT_HAND_COMMAND_NAMES) if include_hands else []
            current_obs_names = list(arm_state_names) + list(head_state_names)
            current_action_names = list(arm_command_names) + list(head_command_names) + hand_command_names
            if current_obs_names != list(obs_names) or current_action_names != list(action_names):
                raise ValueError(
                    f"Inconsistent joint layout in {log_file}.\n"
                    f"Expected obs={obs_names}, action={action_names}\n"
                    f"Got obs={current_obs_names}, action={current_action_names}"
                )

            for frame_index, entry in enumerate(entries):
                arm_state = entry.get("arm_state", {})
                arm_command = entry.get("arm_command", {})
                head_state = entry.get("head_state", {}) if include_head else {}
                head_command = entry.get("head_command", {}) if include_head else {}
                obs = _as_float_array(
                    [arm_state[name] for name in arm_state_names]
                    + [head_state[name] for name in head_state_names]
                )
                action = _as_float_array(
                    [arm_command[name] for name in arm_command_names]
                    + [head_command[name] for name in head_command_names]
                    + _hand_command_values(entry, DEFAULT_HAND_COMMAND_NAMES if include_hands else [])
                )
                timestamp = float(entry.get("timestamp", frame_index))
                sdk_frame = {
                    "observation.state": obs,
                    "action": action,
                    "timestamp": np.asarray([timestamp], dtype=np.float32),
                    "task": task_name,
                }
                for camera_name, image_key in image_keys.items():
                    frame = _first_camera_frame(entry, camera_name)
                    if frame is not None:
                        sdk_frame[image_key] = frame
                dataset.add_frame(sdk_frame)
            try:
                dataset.save_episode(task=task_name)
            except TypeError:
                dataset.save_episode()
        finalize = getattr(dataset, "finalize", None)
        if callable(finalize):
            finalize()
    except Exception:
        close = getattr(dataset, "finalize", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        raise

    return True


def _write_mp4(video_path: Path, frames: Sequence[np.ndarray], fps: float) -> tuple[int, int]:
    if not frames:
        raise ValueError(f"No frames to write for {video_path}")
    height, width = frames[0].shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()
    return height, width


def _open_video_writer(video_path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    height, width = frame.shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")
    return writer


def _make_parquet_row_group(columns: Dict[str, list], obs_dim: int, action_dim: int) -> pa.Table:
    timestamps_array = np.asarray(columns["timestamp"], dtype=np.float32).reshape(-1, 1)
    obs_array = np.stack(columns["observation.state"], axis=0)
    action_array = np.stack(columns["action"], axis=0)
    return pa.table(
        {
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(obs_array.reshape(-1).tolist(), type=pa.float32()),
                list_size=obs_dim,
            ),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(action_array.reshape(-1).tolist(), type=pa.float32()),
                list_size=action_dim,
            ),
            "timestamp": pa.FixedSizeListArray.from_arrays(
                pa.array(timestamps_array.reshape(-1).tolist(), type=pa.float32()),
                list_size=1,
            ),
            "frame_index": pa.array(columns["frame_index"], type=pa.int64()),
            "episode_index": pa.array(columns["episode_index"], type=pa.int64()),
            "index": pa.array(columns["index"], type=pa.int64()),
            "task_index": pa.array(columns["task_index"], type=pa.int64()),
            "task": pa.array(columns["task"], type=pa.string()),
            "next.done": pa.array(columns["next.done"], type=pa.bool_()),
        }
    )


def _scan_logs(
    log_files: Sequence[Path],
    camera_names: Sequence[str],
    include_head: bool,
    include_hands: bool,
    fps: float | None,
) -> dict:
    obs_names = None
    action_names = None
    resolved_fps = fps
    video_shapes: dict[str, tuple[int, int]] = {}
    total_frames = 0
    total_camera_frames = {camera_name: 0 for camera_name in camera_names}
    total_hand_frames = 0
    episodes_meta = []
    global_index = 0
    fps_deltas: list[float] = []

    for episode_index, log_file in enumerate(log_files):
        entries = _load_log(log_file)
        arm_state_names = _discover_joint_names(entries, "arm_state")
        arm_command_names = _discover_joint_names(entries, "arm_command")
        head_state_names = _discover_joint_names(entries, "head_state") if include_head else []
        head_command_names = _discover_joint_names(entries, "head_command") if include_head else []
        hand_command_names = list(DEFAULT_HAND_COMMAND_NAMES) if include_hands else []
        if not arm_state_names or not arm_command_names:
            raise ValueError(f"{log_file} is missing arm_state/arm_command data.")

        current_obs_names = list(arm_state_names) + list(head_state_names)
        current_action_names = list(arm_command_names) + list(head_command_names) + hand_command_names
        if obs_names is None:
            obs_names = current_obs_names
            action_names = current_action_names
        elif obs_names != current_obs_names or action_names != current_action_names:
            raise ValueError(
                f"Inconsistent joint layout in {log_file}.\n"
                f"Expected obs={obs_names}, action={action_names}\n"
                f"Got obs={current_obs_names}, action={current_action_names}"
            )

        episode_start_index = global_index
        episode_camera_start_indices = dict(total_camera_frames)
        episode_timestamps = []

        for frame_index, entry in enumerate(entries):
            timestamp = float(entry.get("timestamp", frame_index))
            episode_timestamps.append(timestamp)
            if include_hands and _has_hand_command(entry):
                total_hand_frames += 1
            for camera_name in camera_names:
                frame = _first_camera_frame(entry, camera_name)
                if frame is not None:
                    total_camera_frames[camera_name] += 1
                    if camera_name not in video_shapes:
                        video_shapes[camera_name] = tuple(frame.shape[:2])
            global_index += 1

        if resolved_fps is None and len(episode_timestamps) >= 2:
            deltas = np.diff(np.asarray(episode_timestamps, dtype=np.float64))
            deltas = deltas[deltas > 1.0e-6]
            if deltas.size > 0:
                fps_deltas.extend(deltas.tolist())

        episode_meta = {
            "episode_index": episode_index,
            "tasks": ["__TASK_PLACEHOLDER__"],
            "length": len(entries),
            "data_chunk_index": 0,
            "data_file_index": 0,
            "dataset_from_index": episode_start_index,
            "dataset_to_index": global_index,
        }
        for camera_name in camera_names:
            if total_camera_frames[camera_name] <= episode_camera_start_indices[camera_name]:
                continue
            episode_meta[f"video_{camera_name}_chunk_index"] = 0
            episode_meta[f"video_{camera_name}_file_index"] = 0
            episode_meta[f"video_{camera_name}_from_index"] = episode_camera_start_indices[camera_name]
            episode_meta[f"video_{camera_name}_to_index"] = total_camera_frames[camera_name]
        episodes_meta.append(episode_meta)
        total_frames += len(entries)

    if obs_names is None or action_names is None or total_frames == 0:
        raise ValueError("No valid frames found.")
    if resolved_fps is None:
        resolved_fps = float(1.0 / np.median(np.asarray(fps_deltas, dtype=np.float64))) if fps_deltas else 30.0

    return {
        "obs_names": obs_names,
        "action_names": action_names,
        "resolved_fps": resolved_fps,
        "video_shapes": video_shapes,
        "total_frames": total_frames,
        "total_camera_frames": total_camera_frames,
        "total_hand_frames": total_hand_frames,
        "episodes_meta": episodes_meta,
    }


def convert(
    log_files: Sequence[Path],
    output_dir: Path,
    camera_names: Sequence[str],
    task_name: str,
    fps: float | None,
    include_head: bool,
    include_hands: bool,
    repo_id: str | None,
) -> None:
    if pa is None or pq is None:
        raise SystemExit(
            "Missing dependency: pyarrow\n"
            "Install with: python3 -m pip install --user pyarrow"
        )

    scan = _scan_logs(
        log_files=log_files,
        camera_names=camera_names,
        include_head=include_head,
        include_hands=include_hands,
        fps=fps,
    )
    obs_names = scan["obs_names"]
    action_names = scan["action_names"]
    resolved_fps = scan["resolved_fps"]
    episodes_meta = scan["episodes_meta"]
    video_shapes = scan["video_shapes"]
    total_frames = scan["total_frames"]
    total_camera_frames = scan["total_camera_frames"]
    total_hand_frames = scan["total_hand_frames"]

    print(
        "Action layout: "
        f"{len(action_names)} dim -> {', '.join(action_names)}"
    )
    if include_hands:
        print(
            "Hand action export: enabled "
            f"({DEFAULT_HAND_COMMAND_NAMES}; source frames with hand data: {total_hand_frames}/{total_frames})"
        )
        if total_hand_frames == 0:
            print(
                "Warning: no hand_command/hand_trigger_raw values were found. "
                f"{DEFAULT_HAND_COMMAND_NAMES} will be exported with default value {DEFAULT_HAND_COMMAND_VALUE}."
            )
    else:
        print("Hand action export: disabled")

    resolved_repo_id = repo_id or output_dir.name
    sdk_error = None
    if _LeRobotDataset is not None:
        try:
            if output_dir.exists():
                for required_path in ("data", "meta", "videos"):
                    if (output_dir / required_path).exists():
                        raise ValueError(
                            f"Output directory {output_dir} already contains dataset files. "
                            "Please choose a new --output-dir or clear the old dataset first."
                        )
            exported = _try_convert_with_lerobot_sdk(
                log_files=log_files,
                output_dir=output_dir,
                task_name=task_name,
                fps=float(resolved_fps),
                repo_id=resolved_repo_id,
                camera_names=camera_names,
                include_head=include_head,
                include_hands=include_hands,
            )
            if exported:
                print("\nLeRobot dataset written with official SDK.")
                print(f"Output: {output_dir}")
                print(f"Frames: {total_frames} | Episodes: {len(log_files)} | FPS: {resolved_fps:.3f}")
                print(f"Action dim: {len(action_names)}")
                return
        except Exception as exc:
            sdk_error = exc
            print(f"Warning: official lerobot SDK export failed, falling back to manual export: {exc}")

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data" / "chunk-000"
    meta_dir = output_dir / "meta"
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "file-000.parquet"
    parquet_writer = None
    video_writers: dict[str, cv2.VideoWriter] = {}
    video_sizes: dict[str, tuple[int, int]] = {}
    row_group_size = 2048
    row_buffer = {
        "observation.state": [],
        "action": [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
        "task": [],
        "next.done": [],
    }
    obs_sum = None
    obs_sq_sum = None
    obs_min = None
    obs_max = None
    action_sum = None
    action_sq_sum = None
    action_min = None
    action_max = None
    total_written = 0
    global_index = 0

    try:
        for episode_index, log_file in enumerate(log_files):
            entries = _load_log(log_file)
            arm_state_names = _discover_joint_names(entries, "arm_state")
            arm_command_names = _discover_joint_names(entries, "arm_command")
            head_state_names = _discover_joint_names(entries, "head_state") if include_head else []
            head_command_names = _discover_joint_names(entries, "head_command") if include_head else []

            for frame_index, entry in enumerate(entries):
                arm_state = entry.get("arm_state", {})
                arm_command = entry.get("arm_command", {})
                head_state = entry.get("head_state", {}) if include_head else {}
                head_command = entry.get("head_command", {}) if include_head else {}
                hand_command_names = list(DEFAULT_HAND_COMMAND_NAMES) if include_hands else []

                obs = _as_float_array(
                    [arm_state[name] for name in arm_state_names]
                    + [head_state[name] for name in head_state_names]
                )
                action = _as_float_array(
                    [arm_command[name] for name in arm_command_names]
                    + [head_command[name] for name in head_command_names]
                    + _hand_command_values(entry, hand_command_names)
                )
                timestamp = float(entry.get("timestamp", frame_index))

                row_buffer["observation.state"].append(obs)
                row_buffer["action"].append(action)
                row_buffer["timestamp"].append(timestamp)
                row_buffer["frame_index"].append(frame_index)
                row_buffer["episode_index"].append(episode_index)
                row_buffer["index"].append(global_index)
                row_buffer["task_index"].append(0)
                row_buffer["task"].append(task_name)
                row_buffer["next.done"].append(frame_index == len(entries) - 1)

                obs64 = obs.astype(np.float64)
                action64 = action.astype(np.float64)
                if obs_sum is None:
                    obs_sum = np.zeros_like(obs64)
                    obs_sq_sum = np.zeros_like(obs64)
                    obs_min = obs64.copy()
                    obs_max = obs64.copy()
                    action_sum = np.zeros_like(action64)
                    action_sq_sum = np.zeros_like(action64)
                    action_min = action64.copy()
                    action_max = action64.copy()
                obs_sum += obs64
                obs_sq_sum += obs64 * obs64
                obs_min = np.minimum(obs_min, obs64)
                obs_max = np.maximum(obs_max, obs64)
                action_sum += action64
                action_sq_sum += action64 * action64
                action_min = np.minimum(action_min, action64)
                action_max = np.maximum(action_max, action64)

                for camera_name in camera_names:
                    frame = _first_camera_frame(entry, camera_name)
                    if frame is None:
                        continue
                    if camera_name not in video_writers:
                        video_writers[camera_name] = _open_video_writer(
                            output_dir / "videos" / camera_name / "chunk-000" / "file-000.mp4",
                            frame,
                            fps=resolved_fps,
                        )
                        video_sizes[camera_name] = frame.shape[:2]
                    else:
                        video_height, video_width = video_sizes[camera_name]
                        if frame.shape[:2] != (video_height, video_width):
                            frame = cv2.resize(frame, (video_width, video_height))
                    video_writers[camera_name].write(frame)

                global_index += 1
                total_written += 1

                if len(row_buffer["timestamp"]) >= row_group_size:
                    table = _make_parquet_row_group(
                        row_buffer,
                        obs_dim=len(obs_names),
                        action_dim=len(action_names),
                    )
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(parquet_path, table.schema)
                    parquet_writer.write_table(table)
                    for key in row_buffer:
                        row_buffer[key].clear()

            print(f"Loaded {log_file} ({len(entries)} frames)")

        if row_buffer["timestamp"]:
            table = _make_parquet_row_group(
                row_buffer,
                obs_dim=len(obs_names),
                action_dim=len(action_names),
            )
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(parquet_path, table.schema)
            parquet_writer.write_table(table)
            for key in row_buffer:
                row_buffer[key].clear()
    finally:
        if parquet_writer is not None:
            parquet_writer.close()
        for video_writer in video_writers.values():
            video_writer.release()

    for episode_meta in episodes_meta:
        episode_meta["tasks"] = [task_name]

    pq.write_table(_build_episode_table(episodes_meta), episodes_dir / "file-000.parquet")
    pq.write_table(_build_tasks_table([task_name]), meta_dir / "tasks.parquet")

    info = {
        "codebase_version": "v3.0",
        "robot_type": "x2_upper_body_hardware",
        "total_episodes": len(log_files),
        "total_frames": int(total_frames),
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": float(resolved_fps),
        "splits": {"train": f"0:{len(log_files)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [int(len(action_names))],
                "names": [f"{name}.pos" for name in action_names],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [int(len(obs_names))],
                "names": [f"{name}.pos" for name in obs_names],
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "task": {
                "dtype": "string",
                "shape": [1],
                "names": None,
            },
        },
    }
    for camera_name in camera_names:
        if total_camera_frames.get(camera_name, 0) <= 0 or camera_name not in video_sizes:
            continue
        video_height, video_width = video_sizes[camera_name]
        info["features"][f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": [int(video_height), int(video_width), 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": float(resolved_fps),
                "video.codec": "mp4v",
            },
        }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    obs_mean = obs_sum / total_written
    action_mean = action_sum / total_written
    obs_var = np.maximum(obs_sq_sum / total_written - obs_mean * obs_mean, 0.0)
    action_var = np.maximum(action_sq_sum / total_written - action_mean * action_mean, 0.0)
    stats = {
        "observation.state": {
            "mean": obs_mean.tolist(),
            "std": np.sqrt(obs_var).tolist(),
            "min": obs_min.tolist(),
            "max": obs_max.tolist(),
        },
        "action": {
            "mean": action_mean.tolist(),
            "std": np.sqrt(action_var).tolist(),
            "min": action_min.tolist(),
            "max": action_max.tolist(),
        },
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nLeRobot v3-style dataset written to: {output_dir}")
    print(f"Frames: {total_frames} | Episodes: {len(log_files)} | FPS: {resolved_fps:.3f}")
    print(f"Action dim: {len(action_names)}")
    if include_hands:
        print(f"Hand action dims: {', '.join(DEFAULT_HAND_COMMAND_NAMES)}")
    for camera_name in camera_names:
        camera_frame_count = total_camera_frames.get(camera_name, 0)
        if camera_frame_count > 0:
            print(f"Camera video exported: {camera_name} ({camera_frame_count} frames)")
        else:
            print(f"No camera frames found for camera '{camera_name}'.")
    if sdk_error is not None:
        print(f"Fallback reason: {sdk_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_paths",
        nargs="+",
        help="One or more hardware teleop .pkl files, or directories containing them.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/x2_hardware_lerobot_v3",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--camera-name",
        default=None,
        help="Backward-compatible single camera key inside entry['image']. Ignored when --camera-names is set.",
    )
    parser.add_argument(
        "--camera-names",
        default=None,
        help="Comma-separated camera keys inside entry['image'].",
    )
    parser.add_argument(
        "--task",
        default="x2_hardware_teleop",
        help="Task/language instruction stored in metadata and frame data.",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Language instruction alias for --task. If set, overrides --task.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override dataset/video FPS. Defaults to timestamp-derived FPS or 30.",
    )
    parser.add_argument(
        "--include-head",
        action="store_true",
        help="Include head_state/head_command in observation.state/action.",
    )
    parser.add_argument(
        "--include-hands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append left_hand/right_hand to action from hand_command or hand_trigger_raw. "
            "Enabled by default; use --no-include-hands to disable."
        ),
    )
    parser.add_argument(
        "--exclude-hands",
        action="store_true",
        help="Backward-compatible alias for --no-include-hands.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Preferred repo_id when official lerobot SDK export is available. Defaults to output dir name.",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.log_paths]
    log_files: list[Path] = []
    for path in paths:
        if path.is_dir():
            log_files.extend(sorted(path.glob("*.pkl")))
        elif path.is_file():
            log_files.append(path)
    log_files = sorted(log_files)
    if not log_files:
        raise SystemExit("No .pkl log files found.")

    camera_names_arg = args.camera_names or args.camera_name or "head_front,right_wrist,left_wrist"
    camera_names = [name.strip() for name in camera_names_arg.split(",") if name.strip()]
    if not camera_names:
        raise SystemExit("No camera names configured.")
    include_hands = bool(args.include_hands) and not bool(args.exclude_hands)

    convert(
        log_files=log_files,
        output_dir=Path(args.output_dir),
        camera_names=camera_names,
        task_name=args.instruction or args.task,
        fps=args.fps,
        include_head=args.include_head,
        include_hands=include_hands,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
