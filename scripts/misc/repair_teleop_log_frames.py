#!/usr/bin/env python3
"""
Remove invalid frames from XRoboToolkit teleoperation pickle logs.

By default this script only reports what it would change. Use --write to save
repaired logs. In-place writes create *.bak backups by default; pass
--no-backup if you do not need backups.
Empty or unreadable pickle files are removed in --write mode unless --keep-bad-files
is set.

Examples:
    python3 scripts/misc/repair_teleop_log_frames.py logs/x2_upper_body_hardware
    python3 scripts/misc/repair_teleop_log_frames.py logs/x2_upper_body_hardware --write
    python3 scripts/misc/repair_teleop_log_frames.py logs/x2_upper_body_hardware --write --no-backup
    python3 scripts/misc/repair_teleop_log_frames.py bad.pkl --write --output-dir repaired_logs
"""

from __future__ import annotations

import argparse
import glob
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REQUIRED_KEYS = ("timestamp", "arm_state", "arm_command", "image")
DEFAULT_CONSISTENT_DICT_KEYS = (
    "arm_state",
    "arm_velocity",
    "arm_command",
    "head_state",
    "head_velocity",
    "head_command",
    "hand_command",
    "hand_trigger_raw",
    "image",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="Log .pkl file(s), directories, or glob patterns.")
    parser.add_argument("--write", action="store_true", help="Actually write repaired logs. Default is dry-run.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write repaired logs into this directory instead of editing files in-place.",
    )
    parser.add_argument(
        "--backup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create *.bak files for in-place writes. Enabled by default; use --no-backup to disable.",
    )
    parser.add_argument(
        "--required-key",
        action="append",
        dest="required_keys",
        help="Required top-level key. Can be passed multiple times.",
    )
    parser.add_argument(
        "--consistent-key",
        action="append",
        dest="consistent_keys",
        help="Dict field whose subkeys must stay consistent. Can be passed multiple times.",
    )
    parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=2,
        help="Refuse to write a repaired log with fewer valid frames.",
    )
    parser.add_argument(
        "--keep-bad-files",
        action="store_true",
        help="Do not delete empty or unreadable pickle files in --write mode.",
    )
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


def save_pickle(path: Path, data: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(data, f)


def find_reference_subkeys(entries: list[dict], key: str) -> set[str] | None:
    for entry in entries:
        value = entry.get(key)
        if isinstance(value, dict) and value:
            return set(value.keys())
    return None


def find_bad_frames(
    entries: list[dict],
    required_keys: tuple[str, ...],
    consistent_keys: tuple[str, ...],
) -> dict[int, list[str]]:
    bad: dict[int, list[str]] = {}
    reference_top_keys = set(entries[0].keys())
    reference_subkeys = {
        key: subkeys
        for key in consistent_keys
        if (subkeys := find_reference_subkeys(entries, key)) is not None
    }

    for idx, entry in enumerate(entries):
        reasons: list[str] = []
        missing_required = [key for key in required_keys if key not in entry]
        if missing_required:
            reasons.append(f"missing required keys {missing_required}")

        if set(entry.keys()) != reference_top_keys:
            missing = sorted(reference_top_keys - set(entry.keys()))
            extra = sorted(set(entry.keys()) - reference_top_keys)
            reasons.append(f"top-level keys changed missing={missing} extra={extra}")

        for key, expected_subkeys in reference_subkeys.items():
            value = entry.get(key)
            if not isinstance(value, dict):
                reasons.append(f"{key} is not dict")
                continue
            current_subkeys = set(value.keys())
            if current_subkeys != expected_subkeys:
                missing = sorted(expected_subkeys - current_subkeys)
                extra = sorted(current_subkeys - expected_subkeys)
                reasons.append(f"{key} subkeys changed missing={missing} extra={extra}")

        if reasons:
            bad[idx] = reasons

    return bad


def destination_for(path: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return path
    return output_dir / path.name


def backup_file(path: Path, args: argparse.Namespace) -> Path | None:
    if not args.backup:
        return None
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"      backup: {backup}")
    else:
        print(f"      backup exists: {backup}")
    return backup


def remove_bad_file(path: Path, args: argparse.Namespace, reason: str) -> tuple[int, int, bool]:
    print(f"[BAD] {path}: {reason}")
    if not args.write:
        return 0, 1, False
    if args.keep_bad_files:
        print("      kept because --keep-bad-files is set")
        return 0, 1, False
    if args.output_dir is not None:
        print("      not removed because --output-dir is set")
        return 0, 1, False
    backup_file(path, args)
    path.unlink()
    print(f"      removed: {path}")
    return 0, 1, True


def repair_file(path: Path, args: argparse.Namespace) -> tuple[int, int, bool]:
    required_keys = tuple(args.required_keys or DEFAULT_REQUIRED_KEYS)
    consistent_keys = tuple(args.consistent_keys or DEFAULT_CONSISTENT_DICT_KEYS)

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        print(f"[SKIP] {path}: failed to stat file: {exc}")
        return 0, 0, False
    if file_size == 0:
        return remove_bad_file(path, args, "empty pickle file")

    try:
        data = load_pickle(path)
    except Exception as exc:
        return remove_bad_file(path, args, f"failed to load pickle: {exc}")

    if not isinstance(data, list) or not data or not all(isinstance(entry, dict) for entry in data):
        return remove_bad_file(path, args, f"expected non-empty list[dict], got {type(data).__name__}")

    bad_frames = find_bad_frames(data, required_keys, consistent_keys)
    if not bad_frames:
        print(f"[OK] {path}: {len(data)} frame(s), no bad frames")
        return len(data), 0, False

    repaired = [entry for idx, entry in enumerate(data) if idx not in bad_frames]
    bad_preview = ", ".join(str(idx) for idx in list(bad_frames)[:10])
    print(f"[FIX] {path}: remove {len(bad_frames)}/{len(data)} frame(s): {bad_preview}")
    for idx, reasons in list(bad_frames.items())[:5]:
        print(f"      frame {idx}: {'; '.join(reasons)}")
    if len(bad_frames) > 5:
        print(f"      ... {len(bad_frames) - 5} more bad frame(s)")

    if len(repaired) < args.min_valid_frames:
        print(f"      refused: repaired log would have only {len(repaired)} valid frame(s)")
        return len(data), len(bad_frames), False

    if not args.write:
        return len(data), len(bad_frames), False

    dst = destination_for(path, args.output_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst == path:
        backup_file(path, args)

    save_pickle(dst, repaired)
    print(f"      wrote: {dst} ({len(repaired)} frame(s))")
    return len(data), len(bad_frames), True


def main() -> int:
    args = parse_args()
    files = expand_paths(args.paths)
    if not files:
        print("ERROR: no .pkl files found", file=sys.stderr)
        return 1

    total_files = 0
    total_frames = 0
    total_bad_frames = 0
    written_files = 0
    for path in files:
        frames, bad_frames, wrote = repair_file(path, args)
        total_files += 1
        total_frames += frames
        total_bad_frames += bad_frames
        written_files += int(wrote)

    mode = "WRITE" if args.write else "DRY-RUN"
    print(
        f"\n{mode}: checked {total_files} file(s), "
        f"{total_frames} frame(s), found {total_bad_frames} bad frame(s), wrote {written_files} file(s)"
    )
    if not args.write and total_bad_frames:
        print("Run again with --write to apply repairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
