#!/usr/bin/env python3
"""
Convert X2 hardware teleop logs (.pkl) into a LeRobot v3-style dataset layout.

This converter is intentionally opinionated for the current X2 hardware logger:
- Uses arm state only for `observation.state` by default.
- Uses arm command only for `action` by default.
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
    camera_name: str,
    video_shape: tuple[int, int] | None,
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
    if video_shape is not None:
        height, width = video_shape
        features[f"observation.images.{camera_name}"] = {
            "dtype": "image",
            "shape": (int(height), int(width), 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _try_convert_with_lerobot_sdk(
    frames_by_episode: Sequence[dict],
    output_dir: Path,
    task_name: str,
    fps: float,
    repo_id: str,
) -> bool:
    if _LeRobotDataset is None:
        return False

    first_frame = None
    for episode in frames_by_episode:
        if episode["frames"]:
            first_frame = episode["frames"][0]
            break
    if first_frame is None:
        raise ValueError("No frames available for SDK export.")

    image_key = next((key for key in first_frame.keys() if key.startswith("observation.images.")), None)
    video_shape = None
    if image_key is not None and first_frame[image_key] is not None:
        video_shape = tuple(first_frame[image_key].shape[:2])

    dataset = _LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=float(fps),
        robot_type="x2_upper_body_hardware",
        features=_build_feature_spec(
            obs_names=first_frame["observation.state_names"],
            action_names=first_frame["action_names"],
            camera_name=image_key.split(".", 2)[2] if image_key else "head_front",
            video_shape=video_shape,
        ),
        use_videos=image_key is not None,
    )

    try:
        for episode in frames_by_episode:
            for frame in episode["frames"]:
                sdk_frame = {
                    "observation.state": frame["observation.state"],
                    "action": frame["action"],
                    "timestamp": np.asarray([frame["timestamp"]], dtype=np.float32),
                    "task": task_name,
                }
                if image_key is not None and image_key in frame and frame[image_key] is not None:
                    sdk_frame[image_key] = frame[image_key]
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


def convert(
    log_files: Sequence[Path],
    output_dir: Path,
    camera_name: str,
    task_name: str,
    fps: float | None,
    include_head: bool,
    repo_id: str | None,
) -> None:
    if pa is None or pq is None:
        raise SystemExit(
            "Missing dependency: pyarrow\n"
            "Install with: python3 -m pip install --user pyarrow"
        )

    all_obs = []
    all_actions = []
    all_timestamps = []
    all_episode_indices = []
    all_frame_indices = []
    all_global_indices = []
    all_task_indices = []
    all_done = []
    episodes_meta = []
    camera_frames = []
    frames_by_episode = []

    global_index = 0
    obs_names = None
    action_names = None
    resolved_fps = fps
    video_height = None
    video_width = None

    for episode_index, log_file in enumerate(log_files):
        entries = _load_log(log_file)
        arm_state_names = _discover_joint_names(entries, "arm_state")
        arm_command_names = _discover_joint_names(entries, "arm_command")
        head_state_names = _discover_joint_names(entries, "head_state") if include_head else []
        head_command_names = _discover_joint_names(entries, "head_command") if include_head else []
        if not arm_state_names or not arm_command_names:
            raise ValueError(f"{log_file} is missing arm_state/arm_command data.")

        current_obs_names = list(arm_state_names) + list(head_state_names)
        current_action_names = list(arm_command_names) + list(head_command_names)
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
        episode_camera_start_index = len(camera_frames)
        episode_timestamps = []
        episode_frames = []

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
            )
            timestamp = float(entry.get("timestamp", frame_index))

            all_obs.append(obs)
            all_actions.append(action)
            all_timestamps.append(timestamp)
            all_episode_indices.append(episode_index)
            all_frame_indices.append(frame_index)
            all_global_indices.append(global_index)
            all_task_indices.append(0)
            all_done.append(frame_index == len(entries) - 1)
            episode_timestamps.append(timestamp)

            frame = _first_camera_frame(entry, camera_name)
            if frame is not None:
                camera_frames.append(frame)
                if video_height is None or video_width is None:
                    video_height, video_width = frame.shape[:2]

            sdk_frame = {
                "observation.state": obs.copy(),
                "action": action.copy(),
                "timestamp": timestamp,
                "observation.state_names": list(current_obs_names),
                "action_names": list(current_action_names),
            }
            if frame is not None:
                sdk_frame[f"observation.images.{camera_name}"] = frame.copy()
            episode_frames.append(sdk_frame)

            global_index += 1

        if resolved_fps is None and len(episode_timestamps) >= 2:
            deltas = np.diff(np.asarray(episode_timestamps, dtype=np.float64))
            deltas = deltas[deltas > 1.0e-6]
            if deltas.size > 0:
                resolved_fps = float(1.0 / np.median(deltas))

        episode_meta = {
            "episode_index": episode_index,
            "tasks": [task_name],
            "length": len(entries),
            "data_chunk_index": 0,
            "data_file_index": 0,
            "dataset_from_index": episode_start_index,
            "dataset_to_index": global_index,
        }
        if len(camera_frames) > episode_camera_start_index:
            episode_meta[f"video_{camera_name}_chunk_index"] = 0
            episode_meta[f"video_{camera_name}_file_index"] = 0
            episode_meta[f"video_{camera_name}_from_index"] = episode_camera_start_index
            episode_meta[f"video_{camera_name}_to_index"] = len(camera_frames)
        episodes_meta.append(episode_meta)
        frames_by_episode.append({"frames": episode_frames})
        print(f"Loaded {log_file} ({len(entries)} frames)")

    if not all_obs:
        raise ValueError("No valid frames found.")
    if resolved_fps is None:
        resolved_fps = 30.0

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
                frames_by_episode=frames_by_episode,
                output_dir=output_dir,
                task_name=task_name,
                fps=float(resolved_fps),
                repo_id=resolved_repo_id,
            )
            if exported:
                print("\nLeRobot dataset written with official SDK.")
                print(f"Output: {output_dir}")
                print(f"Frames: {len(all_obs)} | Episodes: {len(log_files)} | FPS: {resolved_fps:.3f}")
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

    obs_array = np.stack(all_obs, axis=0)
    action_array = np.stack(all_actions, axis=0)
    timestamps_array = np.asarray(all_timestamps, dtype=np.float32).reshape(-1, 1)

    data_columns = {
        "observation.state": pa.FixedSizeListArray.from_arrays(
            pa.array(obs_array.reshape(-1).tolist(), type=pa.float32()),
            list_size=obs_array.shape[1],
        ),
        "action": pa.FixedSizeListArray.from_arrays(
            pa.array(action_array.reshape(-1).tolist(), type=pa.float32()),
            list_size=action_array.shape[1],
        ),
        "timestamp": pa.FixedSizeListArray.from_arrays(
            pa.array(timestamps_array.reshape(-1).tolist(), type=pa.float32()),
            list_size=1,
        ),
        "frame_index": pa.array(all_frame_indices, type=pa.int64()),
        "episode_index": pa.array(all_episode_indices, type=pa.int64()),
        "index": pa.array(all_global_indices, type=pa.int64()),
        "task_index": pa.array(all_task_indices, type=pa.int64()),
        "next.done": pa.array(all_done, type=pa.bool_()),
    }
    pq.write_table(pa.table(data_columns), data_dir / "file-000.parquet")

    if camera_frames:
        video_height, video_width = _write_mp4(
            output_dir / "videos" / camera_name / "chunk-000" / "file-000.mp4",
            camera_frames,
            fps=resolved_fps,
        )

    pq.write_table(_build_episode_table(episodes_meta), episodes_dir / "file-000.parquet")
    pq.write_table(_build_tasks_table([task_name]), meta_dir / "tasks.parquet")

    info = {
        "codebase_version": "v3.0",
        "robot_type": "x2_upper_body_hardware",
        "total_episodes": len(log_files),
        "total_frames": int(obs_array.shape[0]),
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
                "shape": [int(action_array.shape[1])],
                "names": [f"{name}.pos" for name in action_names],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [int(obs_array.shape[1])],
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
        },
    }
    if camera_frames and video_height is not None and video_width is not None:
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

    stats = {
        "observation.state": _feature_stats(obs_array),
        "action": _feature_stats(action_array),
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nLeRobot v3-style dataset written to: {output_dir}")
    print(f"Frames: {obs_array.shape[0]} | Episodes: {len(log_files)} | FPS: {resolved_fps:.3f}")
    if camera_frames:
        print(f"Camera video exported: {camera_name} ({len(camera_frames)} frames)")
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
        default="head_front",
        help="Camera key inside entry['image'].",
    )
    parser.add_argument(
        "--task",
        default="x2_hardware_teleop",
        help="Task description stored in metadata.",
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

    convert(
        log_files=log_files,
        output_dir=Path(args.output_dir),
        camera_name=args.camera_name,
        task_name=args.task,
        fps=args.fps,
        include_head=args.include_head,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
