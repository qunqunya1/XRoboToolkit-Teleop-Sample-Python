#!/usr/bin/env python3
"""
Play back collected XRoboToolkit teleoperation data for quick quality checks.

Supported inputs:
  - Raw teleop pickle logs: logs/<robot>/teleop_log_*.pkl
  - LeRobot v3-style dataset directories: datasets/<name>/

Examples:
  python3 scripts/misc/play_collected_data.py logs/x2_upper_body_hardware/teleop_log_*.pkl
  python3 scripts/misc/play_collected_data.py datasets/x2_hardware_lerobot_v3
  python3 scripts/misc/play_collected_data.py datasets/x2_hardware_lerobot_v3 --no-display
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np


def configure_qt_font_dir() -> None:
    """Avoid noisy OpenCV Qt font warnings when cv2's bundled fonts are absent."""
    if os.environ.get("QT_QPA_FONTDIR"):
        return
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation2"),
        Path("/usr/share/fonts/truetype"),
        Path("/usr/share/fonts"),
    )
    for candidate in candidates:
        if candidate.is_dir():
            os.environ["QT_QPA_FONTDIR"] = str(candidate)
            return


configure_qt_font_dir()

import cv2

try:
    import pyarrow.parquet as pq
except ImportError:  # Optional unless reading LeRobot parquet files.
    pq = None

try:
    from scripts.misc.check_teleop_log_health import check_log, print_report
except ImportError:  # pragma: no cover - helper may not be importable outside repo root
    check_log = None
    print_report = None


STATE_KEYS = ("observation.state", "arm_state")
ACTION_KEYS = ("action", "arm_command")


@dataclass
class VideoInfo:
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0


@dataclass
class PlaybackData:
    source: Path
    mode: str
    fps: float
    timestamps: list[float]
    states: np.ndarray | None
    actions: np.ndarray | None
    state_names: list[str]
    action_names: list[str]
    frames: list[dict[str, np.ndarray]] | None = None
    video_paths: dict[str, Path] | None = None
    video_info: dict[str, VideoInfo] | None = None

    @property
    def length(self) -> int:
        if self.timestamps:
            return len(self.timestamps)
        if self.states is not None:
            return int(self.states.shape[0])
        if self.actions is not None:
            return int(self.actions.shape[0])
        if self.frames is not None:
            return len(self.frames)
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help=".pkl logs, directories, glob patterns, or a LeRobot dataset dir.")
    parser.add_argument("--camera-names", default=None, help="Comma-separated camera names to show. Defaults to all cameras found.")
    parser.add_argument("--fps", type=float, default=None, help="Override playback FPS. Defaults to timestamps or dataset metadata.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--fit-window", action="store_true", help="Scale camera images to fit --max-width/--max-height. Default keeps original video pixels.")
    parser.add_argument("--max-width", type=int, default=1280, help="Maximum mosaic width when --fit-window is used.")
    parser.add_argument("--max-height", type=int, default=900, help="Maximum mosaic height when --fit-window is used.")
    parser.add_argument("--trail", type=int, default=180, help="Number of recent samples shown in curves.")
    parser.add_argument("--no-display", action="store_true", help="Print quality summary and validate readable frames without opening a window.")
    parser.add_argument("--decode-images", action="store_true", help="Decode every raw log image during health check.")
    return parser.parse_args()


def expand_paths(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
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


def is_lerobot_dir(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def health_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        required_keys=None,
        min_entries=2,
        min_hz=1.0,
        max_hz=120.0,
        max_dt=0.5,
        joint_limit=2 * math.pi,
        velocity_limit=20.0,
        max_jump=1.0,
        static_action_travel=0.5,
        static_state_travel=0.5,
        static_command_range=0.2,
        static_state_range=0.2,
        decode_images=args.decode_images,
        strict_warnings=False,
        max_issues=20,
    )


def decode_image(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, dict) and "raw" in raw:
        return decode_raw_image_dict(raw)
    if isinstance(raw, bytes):
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    img = np.asarray(raw)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.shape[2] == 3:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            return img
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
    if str(raw.get("encoding", "")).lower() == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def extract_camera_frame(image_value: Any, camera_name: str) -> np.ndarray | None:
    if not isinstance(image_value, dict):
        return None
    camera_data = image_value.get(camera_name)
    if isinstance(camera_data, dict):
        raw = camera_data.get("color")
        if raw is None:
            raw = next(iter(camera_data.values()), None)
        return decode_image(raw)
    return decode_image(camera_data)


def sorted_numeric_values(value: Any) -> tuple[list[str], list[float]]:
    if not isinstance(value, dict):
        return [], []
    names = []
    numbers = []
    for name in sorted(value):
        item = value[name]
        if isinstance(item, (int, float, np.number)) and np.isfinite(float(item)):
            names.append(str(name))
            numbers.append(float(item))
    return names, numbers


def derive_fps(timestamps: list[float], fallback: float = 30.0) -> float:
    if len(timestamps) < 2:
        return fallback
    dt = np.diff(np.asarray(timestamps, dtype=np.float64))
    dt = dt[dt > 1.0e-6]
    return float(1.0 / np.median(dt)) if dt.size else fallback


def load_raw_logs(paths: list[Path], args: argparse.Namespace) -> PlaybackData:
    entries: list[dict] = []
    source = paths[0] if len(paths) == 1 else paths[0].parent
    for path in paths:
        if check_log is not None and print_report is not None:
            report = check_log(path, health_args(args))
            print_report(report, max_issues=20)
        with path.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} top-level object must be a list.")
        entries.extend(entry for entry in data if isinstance(entry, dict))
    if not entries:
        raise ValueError("No valid log entries found.")

    camera_names = resolve_raw_camera_names(entries, args.camera_names)
    timestamps = [float(entry.get("timestamp", idx)) for idx, entry in enumerate(entries)]
    state_names, action_names = [], []
    state_rows, action_rows = [], []
    frames: list[dict[str, np.ndarray]] = []

    for entry in entries:
        current_frames = {}
        for camera_name in camera_names:
            frame = extract_camera_frame(entry.get("image"), camera_name)
            if frame is not None:
                current_frames[camera_name] = frame
        frames.append(current_frames)

        names, values = sorted_numeric_values(entry.get("arm_state"))
        if values:
            if not state_names:
                state_names = names
            if names == state_names:
                state_rows.append(values)
        names, values = sorted_numeric_values(entry.get("arm_command"))
        if values:
            if not action_names:
                action_names = names
            if names == action_names:
                action_rows.append(values)

    return PlaybackData(
        source=source,
        mode="raw-pkl",
        fps=args.fps or derive_fps(timestamps),
        timestamps=timestamps,
        states=np.asarray(state_rows, dtype=np.float32) if state_rows else None,
        actions=np.asarray(action_rows, dtype=np.float32) if action_rows else None,
        state_names=state_names,
        action_names=action_names,
        frames=frames,
        video_info=infer_frame_video_info(frames, args.fps or derive_fps(timestamps)),
    )


def resolve_raw_camera_names(entries: list[dict], camera_names_arg: str | None) -> list[str]:
    if camera_names_arg:
        return [name.strip() for name in camera_names_arg.split(",") if name.strip()]
    for entry in entries:
        image = entry.get("image")
        if isinstance(image, dict) and image:
            return sorted(str(name) for name in image.keys())
    return []


def load_lerobot_dataset(path: Path, args: argparse.Namespace) -> PlaybackData:
    if pq is None:
        raise SystemExit(
            "Missing dependency: pyarrow is required to read LeRobot parquet data.\n"
            "Install it in the active environment with:\n"
            "  PYTHONNOUSERSITE=1 python3 -m pip install pyarrow"
        )
    info_path = path / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    camera_names = resolve_lerobot_camera_names(features, args.camera_names)
    data_files = sorted((path / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise ValueError(f"No parquet files found under {path / 'data'}")

    tables = [pq.read_table(file_path) for file_path in data_files]
    table = tables[0] if len(tables) == 1 else __import__("pyarrow").concat_tables(tables)
    columns = table.to_pydict()
    timestamps = flatten_column(columns.get("timestamp"))
    if not timestamps:
        timestamps = [float(idx) for idx in range(table.num_rows)]

    states = column_to_matrix(columns.get("observation.state"))
    actions = column_to_matrix(columns.get("action"))
    video_paths = {}
    video_info = {}
    for camera_name in camera_names:
        matches = sorted((path / "videos" / camera_name).glob("chunk-*/file-*.mp4"))
        if matches:
            video_paths[camera_name] = matches[0]
            video_info[camera_name] = read_video_info(matches[0])

    print_lerobot_summary(path, info, table.num_rows, video_paths)
    return PlaybackData(
        source=path,
        mode="lerobot-v3",
        fps=args.fps or float(info.get("fps") or derive_fps(timestamps)),
        timestamps=[float(x) for x in timestamps],
        states=states,
        actions=actions,
        state_names=feature_names(features, "observation.state"),
        action_names=feature_names(features, "action"),
        video_paths=video_paths,
        video_info=video_info,
    )


def read_video_info(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        return VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def infer_frame_video_info(frames: list[dict[str, np.ndarray]], fps: float) -> dict[str, VideoInfo]:
    info: dict[str, VideoInfo] = {}
    counts: dict[str, int] = {}
    for frame_set in frames:
        for camera_name, frame in frame_set.items():
            counts[camera_name] = counts.get(camera_name, 0) + 1
            if camera_name not in info:
                height, width = frame.shape[:2]
                info[camera_name] = VideoInfo(width=width, height=height, fps=fps, frame_count=0)
    for camera_name, count in counts.items():
        info[camera_name].frame_count = count
    return info


def resolve_lerobot_camera_names(features: dict, camera_names_arg: str | None) -> list[str]:
    if camera_names_arg:
        return [name.strip() for name in camera_names_arg.split(",") if name.strip()]
    prefix = "observation.images."
    return sorted(key[len(prefix) :] for key, spec in features.items() if key.startswith(prefix) and spec.get("dtype") in {"video", "image"})


def flatten_column(values: Any) -> list[float]:
    if values is None:
        return []
    result = []
    for item in values:
        if isinstance(item, (list, tuple, np.ndarray)):
            result.append(float(item[0]) if len(item) else 0.0)
        else:
            result.append(float(item))
    return result


def column_to_matrix(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    rows = []
    for item in values:
        if isinstance(item, np.ndarray):
            rows.append(item.astype(np.float32).reshape(-1))
        elif isinstance(item, (list, tuple)):
            rows.append(np.asarray(item, dtype=np.float32).reshape(-1))
        else:
            rows.append(np.asarray([item], dtype=np.float32))
    return np.stack(rows, axis=0) if rows else None


def feature_names(features: dict, key: str) -> list[str]:
    names = features.get(key, {}).get("names") or []
    return [str(name) for name in names]


def print_lerobot_summary(path: Path, info: dict, row_count: int, video_paths: dict[str, Path]) -> None:
    print(f"\n[OK] {path}")
    print(f"  mode: lerobot-v3")
    print(f"  rows: {row_count}")
    print(f"  episodes: {info.get('total_episodes', 'unknown')}")
    print(f"  fps: {float(info.get('fps', 0.0)):.3f}")
    if video_paths:
        print(f"  videos: {', '.join(sorted(video_paths))}")
    else:
        print("  videos: none found")


def print_quality_summary(data: PlaybackData) -> None:
    print(f"\nPlayback source: {data.source}")
    print(f"  mode: {data.mode}")
    print(f"  frames: {data.length}")
    print(f"  fps: {data.fps:.3f}")
    print_video_info(data.video_info)
    print_numeric_quality("state", data.states, data.state_names)
    print_numeric_quality("action", data.actions, data.action_names)


def print_video_info(video_info: dict[str, VideoInfo] | None) -> None:
    if not video_info:
        print("  video info: none")
        return
    print("  video info:")
    for camera_name, info in sorted(video_info.items()):
        resolution = f"{info.width}x{info.height}" if info.width and info.height else "unknown"
        fps = f"{info.fps:.3f}" if info.fps > 0 else "unknown"
        print(f"    {camera_name}: {resolution}, fps={fps}, frames={info.frame_count}")


def print_numeric_quality(label: str, matrix: np.ndarray | None, names: list[str]) -> None:
    if matrix is None or matrix.size == 0:
        print(f"  {label}: none")
        return
    finite = np.isfinite(matrix)
    if not finite.all():
        print(f"  {label}: {matrix.shape}, non-finite values: {int((~finite).sum())}")
    else:
        print(f"  {label}: {matrix.shape}, finite")
    diffs = np.abs(np.diff(matrix, axis=0))
    ranges = np.nanmax(matrix, axis=0) - np.nanmin(matrix, axis=0)
    max_step = float(np.nanmax(diffs)) if diffs.size else 0.0
    max_range = float(np.nanmax(ranges)) if ranges.size else 0.0
    moving = int(np.sum(ranges > 1.0e-4))
    print(f"    moving dims: {moving}/{matrix.shape[1]}, max_range={max_range:.4f}, max_step={max_step:.4f}")
    if names:
        top_indices = np.argsort(ranges)[-min(3, len(ranges)) :][::-1]
        top = ", ".join(f"{names[idx]}:{ranges[idx]:.3f}" for idx in top_indices)
        print(f"    largest ranges: {top}")


def validate_readable_frames(data: PlaybackData) -> None:
    if data.frames is not None:
        readable = sum(1 for frame_set in data.frames if frame_set)
        print(f"  readable image timesteps: {readable}/{len(data.frames)}")
        return
    if not data.video_paths:
        print("  readable videos: none")
        return
    for camera_name, video_path in sorted(data.video_paths.items()):
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        shape = "unknown" if frame is None else f"{frame.shape[1]}x{frame.shape[0]}"
        status = "OK" if ok else "FAIL"
        print(f"  video {camera_name}: {status}, frames={total}, shape={shape}, fps={fps:.3f}, path={video_path}")


def make_mosaic(
    frames: dict[str, np.ndarray],
    max_width: int,
    max_height: int,
    fit_window: bool,
) -> np.ndarray:
    if not frames:
        return np.zeros((240, 426, 3), dtype=np.uint8)
    tiles = [ensure_bgr_uint8(frame) for _, frame in sorted(frames.items())]
    cols = min(2, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))

    if fit_window:
        tile_w = max(1, max_width // cols)
        tile_h = max(1, max_height // (rows + 1))
        tiles = [resize_to_box(tile, tile_w, tile_h) for tile in tiles]

    cell_w = max(tile.shape[1] for tile in tiles)
    cell_h = max(tile.shape[0] for tile in tiles)
    padded = [pad_to_box(tile, cell_w, cell_h) for tile in tiles]
    blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    while len(padded) < rows * cols:
        padded.append(blank.copy())
    row_images = [np.hstack(padded[row * cols : (row + 1) * cols]) for row in range(rows)]
    return np.vstack(row_images)


def ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    frame = image
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def pad_to_box(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - image.shape[0]) // 2
    x = (width - image.shape[1]) // 2
    canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
    return canvas


def resize_to_box(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    resized = cv2.resize(image, new_size)
    canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    y = (max_height - resized.shape[0]) // 2
    x = (max_width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def plot_curves(data: PlaybackData, index: int, width: int, height: int, trail: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 25, dtype=np.uint8)
    split = height // 2
    draw_matrix_curve(canvas[:split], data.states, index, trail, "state", (80, 200, 255))
    draw_matrix_curve(canvas[split:], data.actions, index, trail, "action", (120, 255, 120))
    cv2.line(canvas, (0, split), (width, split), (70, 70, 70), 1)
    return canvas


def draw_matrix_curve(canvas: np.ndarray, matrix: np.ndarray | None, index: int, trail: int, label: str, color: tuple[int, int, int]) -> None:
    height, width = canvas.shape[:2]
    cv2.putText(canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if matrix is None or matrix.size == 0:
        cv2.putText(canvas, "no data", (110, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
        return
    end = min(index + 1, matrix.shape[0])
    start = max(0, end - trail)
    segment = matrix[start:end]
    if segment.shape[0] < 2:
        return
    dims = min(segment.shape[1], 12)
    ymin = float(np.nanmin(segment[:, :dims]))
    ymax = float(np.nanmax(segment[:, :dims]))
    if abs(ymax - ymin) < 1.0e-6:
        ymax = ymin + 1.0
    for dim in range(dims):
        values = segment[:, dim]
        xs = np.linspace(0, width - 1, len(values)).astype(np.int32)
        ys = (height - 12 - (values - ymin) / (ymax - ymin) * (height - 36)).astype(np.int32)
        dim_color = tuple(int(c * (0.45 + 0.55 * (dim + 1) / dims)) for c in color)
        points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], False, dim_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"range [{ymin:.2f}, {ymax:.2f}]", (110, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)


def make_status_bar(data: PlaybackData, frames: dict[str, np.ndarray], index: int, width: int, speed: float) -> np.ndarray:
    lines = [
        f"{index + 1}/{data.length}  t={data.timestamps[index]:.3f}s  playback={data.fps:.2f}fps  speed={speed:.2f}x"
    ]
    camera_parts = []
    for camera_name in sorted(frames):
        info = (data.video_info or {}).get(camera_name)
        if info is None:
            height, frame_width = frames[camera_name].shape[:2]
            camera_parts.append(f"{camera_name}: {frame_width}x{height}")
            continue
        resolution = f"{info.width}x{info.height}" if info.width and info.height else "unknown"
        fps = f"{info.fps:.2f}fps" if info.fps > 0 else "fps unknown"
        camera_parts.append(f"{camera_name}: {resolution} @ {fps}")
    if camera_parts:
        lines.append("  |  ".join(camera_parts))

    bar_height = 28 * len(lines) + 12
    bar = np.full((bar_height, width, 3), 18, dtype=np.uint8)
    for line_idx, line in enumerate(lines):
        cv2.putText(
            bar,
            line,
            (12, 24 + line_idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return bar


def frame_source(data: PlaybackData):
    if data.frames is not None:
        for frames in data.frames:
            yield frames
        return
    captures = {name: cv2.VideoCapture(str(path)) for name, path in (data.video_paths or {}).items()}
    try:
        for _ in range(data.length):
            frames = {}
            for name, cap in captures.items():
                ok, frame = cap.read()
                if ok:
                    frames[name] = frame
            yield frames
    finally:
        for cap in captures.values():
            cap.release()


def play(data: PlaybackData, args: argparse.Namespace) -> None:
    print("\nControls: space pause/resume, n step, q/esc quit")
    delay = max(1.0 / max(data.fps * args.speed, 1.0e-6), 0.001)
    paused = False
    window_name = "XRoboToolkit data playback"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cached_frames = list(frame_source(data))
    index = 0
    while index < data.length:
        frames = cached_frames[index] if index < len(cached_frames) else {}
        mosaic_height = max(240, args.max_height - 220)
        mosaic = make_mosaic(frames, args.max_width, mosaic_height, args.fit_window)
        status = make_status_bar(data, frames, index, mosaic.shape[1], args.speed)
        curves = plot_curves(data, index, mosaic.shape[1], 220, args.trail)
        composed = np.vstack([mosaic, status, curves])
        cv2.imshow(window_name, composed)

        start = time.time()
        while True:
            key = cv2.waitKey(30 if paused else 1) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyWindow(window_name)
                return
            if key == ord(" "):
                paused = not paused
            if key == ord("n"):
                index += 1
                break
            if not paused and time.time() - start >= delay:
                index += 1
                break
    cv2.destroyWindow(window_name)


def main() -> int:
    args = parse_args()
    paths = expand_paths(args.paths)
    if not paths:
        print("ERROR: no input files or datasets found", file=sys.stderr)
        return 1

    if len(paths) == 1 and is_lerobot_dir(paths[0]):
        data = load_lerobot_dataset(paths[0], args)
    else:
        pkl_paths = [path for path in paths if path.suffix == ".pkl"]
        if not pkl_paths:
            print("ERROR: expected .pkl log files or a LeRobot dataset directory", file=sys.stderr)
            return 1
        data = load_raw_logs(pkl_paths, args)

    print_quality_summary(data)
    validate_readable_frames(data)
    if args.no_display:
        return 0
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        print("ERROR: DISPLAY is not set. Use --no-display for a non-GUI quality check.", file=sys.stderr)
        return 1
    play(data, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
