#!/usr/bin/env python3
"""
Convert x2_ultra simulation teleop logs (.pkl) to a LeRobot-style dataset directory.

Input log format:
  - list[dict], each entry includes timestamp, qpos, and optional qpos_des.
Output layout:
  - <output_dir>/data/chunk-000/episode_000000.parquet
  - <output_dir>/meta/info.json
  - <output_dir>/meta/episodes.jsonl
  - <output_dir>/meta/tasks.jsonl
  - <output_dir>/meta/stats.json
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pyarrow\n"
        "Install with: python3 -m pip install --user pyarrow"
    ) from exc


def _load_log(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Log file {path} is empty or invalid.")
    return data


def _extract_state_action(log_entries):
    states = []
    actions = []
    timestamps = []
    for entry in log_entries:
        qpos = np.asarray(entry["qpos"], dtype=np.float32)
        state = qpos[7:] if qpos.shape[0] > 7 else qpos
        states.append(state)
        timestamps.append(float(entry.get("timestamp", 0.0)))

        if "qpos_des" in entry:
            qpos_des = np.asarray(entry["qpos_des"], dtype=np.float32)
            action = qpos_des[7:] if qpos_des.shape[0] > 7 else qpos_des
        else:
            action = state.copy()
        actions.append(action)

    state_dim = states[0].shape[0]
    action_dim = actions[0].shape[0]
    if any(s.shape[0] != state_dim for s in states):
        raise ValueError("Inconsistent state dimension in log.")
    if any(a.shape[0] != action_dim for a in actions):
        raise ValueError("Inconsistent action dimension in log.")

    return np.stack(states), np.stack(actions), np.asarray(timestamps, dtype=np.float64)


def _write_episode_parquet(
    out_file: Path,
    episode_index: int,
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
):
    n = states.shape[0]
    frame_index = np.arange(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    done[-1] = True

    state_list = pa.FixedSizeListArray.from_arrays(
        pa.array(states.reshape(-1).tolist(), type=pa.float32()),
        list_size=states.shape[1],
    )
    action_list = pa.FixedSizeListArray.from_arrays(
        pa.array(actions.reshape(-1).tolist(), type=pa.float32()),
        list_size=actions.shape[1],
    )

    table = pa.table(
        {
            "episode_index": pa.array([episode_index] * n, type=pa.int64()),
            "frame_index": pa.array(frame_index.tolist(), type=pa.int64()),
            "timestamp": pa.array(timestamps.tolist(), type=pa.float64()),
            "task_index": pa.array([0] * n, type=pa.int64()),
            "observation.state": state_list,
            "action": action_list,
            "next.done": pa.array(done.tolist(), type=pa.bool_()),
        }
    )
    pq.write_table(table, out_file)


def _compute_stats(states: np.ndarray, actions: np.ndarray):
    return {
        "observation.state": {
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).tolist(),
            "min": states.min(axis=0).tolist(),
            "max": states.max(axis=0).tolist(),
        },
        "action": {
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
        },
    }


def convert(log_files: list[Path], output_dir: Path):
    data_dir = output_dir / "data" / "chunk-000"
    meta_dir = output_dir / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    all_states = []
    all_actions = []
    episodes_meta = []

    for ep_idx, log_file in enumerate(log_files):
        entries = _load_log(log_file)
        states, actions, timestamps = _extract_state_action(entries)
        out_file = data_dir / f"episode_{ep_idx:06d}.parquet"
        _write_episode_parquet(out_file, ep_idx, states, actions, timestamps)

        all_states.append(states)
        all_actions.append(actions)
        episodes_meta.append(
            {
                "episode_index": ep_idx,
                "length": int(states.shape[0]),
                "source_log": str(log_file),
                "task_index": 0,
            }
        )

        print(f"Converted {log_file} -> {out_file} ({states.shape[0]} frames)")

    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    info = {
        "codebase_version": "xrobotoolkit_teleop_x2_keyboard",
        "robot_type": "x2_ultra",
        "total_episodes": len(log_files),
        "total_frames": int(all_states.shape[0]),
        "features": {
            "observation.state_dim": int(all_states.shape[1]),
            "action_dim": int(all_actions.shape[1]),
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": "x2_keyboard_teleop"}) + "\n")

    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for item in episodes_meta:
            f.write(json.dumps(item) + "\n")

    stats = _compute_stats(all_states, all_actions)
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nLeRobot-style dataset written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_paths",
        nargs="+",
        help="One or more teleop log .pkl files (or directories containing .pkl).",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/x2_ultra_keyboard_lerobot",
        help="Output dataset directory.",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.log_paths]
    log_files = []
    for p in paths:
        if p.is_dir():
            log_files.extend(sorted(p.glob("*.pkl")))
        elif p.is_file():
            log_files.append(p)
    log_files = sorted(log_files)
    if not log_files:
        raise SystemExit("No .pkl log files found.")

    convert(log_files, Path(args.output_dir))


if __name__ == "__main__":
    main()
