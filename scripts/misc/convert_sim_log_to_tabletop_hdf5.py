#!/usr/bin/env python3
"""Convert XRoboToolkit simulation .pkl logs to Tabletop-Sim-style HDF5 episodes."""

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np


def _load_log(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Log file {path} is empty or invalid.")
    return data


def _decode_image(raw):
    if isinstance(raw, bytes):
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError(
                "Decoding JPG image bytes requires OpenCV-compatible environment. "
                "Please record with camera_jpg_quality=0 to store raw uint8 frames."
            ) from exc

        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode JPG image bytes.")
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        return img.astype(np.uint8)
    arr = np.asarray(raw)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Invalid image shape: {arr.shape}")
    return arr[:, :, :3].astype(np.uint8)


def _extract_episode_arrays(entries, camera_name: str, instruction: str):
    qpos = []
    qvel = []
    action_joint = []
    images_front = []
    timestamps = []

    for i, entry in enumerate(entries):
        if camera_name:
            image_dict = entry.get("image", {})
            if camera_name not in image_dict:
                continue
            img = _decode_image(image_dict[camera_name])
        else:
            image_dict = entry.get("image", {})
            if not image_dict:
                continue
            first_key = next(iter(image_dict.keys()))
            img = _decode_image(image_dict[first_key])

        q = np.asarray(entry["qpos"], dtype=np.float32)
        qd = np.asarray(entry["qvel"], dtype=np.float32)
        q_des = np.asarray(entry.get("qpos_des", q), dtype=np.float32)

        qpos.append(q)
        qvel.append(qd)
        action_joint.append(q_des)
        images_front.append(img)
        timestamps.append(float(entry.get("timestamp", i)))

    if len(qpos) == 0:
        raise ValueError("No valid image frames found in log entries.")

    qpos = np.stack(qpos)
    qvel = np.stack(qvel)
    action_joint = np.stack(action_joint)
    images_front = np.stack(images_front)

    # Tabletop-style placeholders for fields not present in XR logs.
    env_state = np.zeros((len(qpos), 1), dtype=np.float32)
    instructions = np.asarray([instruction] * len(qpos), dtype=object)

    return {
        "qpos": qpos,
        "qvel": qvel,
        "action_joint": action_joint,
        "front": images_front,
        "env_state": env_state,
        "language_instruction": instructions,
        "timestamp": np.asarray(timestamps, dtype=np.float64),
    }


def _write_hdf5(out_file: Path, arrays: dict):
    n = arrays["qpos"].shape[0]
    h, w, c = arrays["front"].shape[1:]
    if c != 3:
        raise ValueError("Expected RGB image with 3 channels.")

    with h5py.File(out_file, "w", rdcc_nbytes=1024**2 * 2) as root:
        obs = root.create_group("observations")
        states = obs.create_group("states")
        images = obs.create_group("images")
        actions = root.create_group("actions")

        states.create_dataset("qpos", data=arrays["qpos"])
        states.create_dataset("qvel", data=arrays["qvel"])
        states.create_dataset("env_state", data=arrays["env_state"])
        states.create_dataset("language_instruction", data=arrays["language_instruction"], dtype=h5py.string_dtype("utf-8"))
        states.create_dataset("timestamp", data=arrays["timestamp"])

        images.create_dataset("front", data=arrays["front"], dtype="uint8", chunks=(1, h, w, 3))
        actions.create_dataset("joint_pos", data=arrays["action_joint"])



def convert(log_files: list[Path], output_dir: Path, camera_name: str, instruction: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx, log_file in enumerate(log_files):
        entries = _load_log(log_file)
        arrays = _extract_episode_arrays(entries, camera_name=camera_name, instruction=instruction)
        out_path = output_dir / f"episode_{ep_idx}.hdf5"
        _write_hdf5(out_path, arrays)
        print(f"Converted {log_file} -> {out_path} ({arrays['qpos'].shape[0]} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_paths", nargs="+", help="One or more .pkl files or directories containing .pkl logs.")
    parser.add_argument("--output-dir", default="datasets/x2_tabletop_hdf5", help="Output directory for HDF5 episodes.")
    parser.add_argument("--camera-name", default="rgbd_head_front_camera", help="Camera name in logged entry['image'].")
    parser.add_argument("--instruction", default="Teleoperate x2 upper body in simulation", help="Instruction string stored per frame.")
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

    convert(log_files, Path(args.output_dir), camera_name=args.camera_name, instruction=args.instruction)


if __name__ == "__main__":
    main()
