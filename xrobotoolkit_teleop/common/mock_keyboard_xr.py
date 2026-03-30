import atexit
import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass

import meshcat.transformations as tf
import numpy as np


@dataclass
class _ControlStep:
    position: float = 0.01
    rotation_deg: float = 3.0


class MockKeyboardXR:
    """
    Keyboard-driven mock for XR inputs.
    It exposes a subset of the XR SDK behavior used by this project.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._running = True
        self._steps = _ControlStep()
        self._selected_source = "right_controller"
        self._exclusive_grip = os.environ.get("XR_MOCK_EXCLUSIVE_GRIP", "1").strip() not in {"0", "false", "False"}

        self._poses = {
            "left_controller": np.array([0.30, 0.20, 1.00, 0.0, 0.0, 0.0, 1.0], dtype=float),
            "right_controller": np.array([0.30, -0.20, 1.00, 0.0, 0.0, 0.0, 1.0], dtype=float),
            "headset": np.array([0.0, 0.0, 1.55, 0.0, 0.0, 0.0, 1.0], dtype=float),
        }
        auto_grip = os.environ.get("XR_MOCK_AUTO_GRIP", "0").strip() not in {"0", "false", "False"}
        self._key_values = {
            "left_trigger": 0.0,
            "right_trigger": 0.0,
            "left_grip": 1.0 if auto_grip else 0.0,
            "right_grip": 1.0 if auto_grip else 0.0,
        }
        self._joysticks = {
            "left": [0.0, 0.0],
            "right": [0.0, 0.0],
        }
        self._button_deadline = {}
        self._tty_mode = os.name == "posix"
        self._tty_stream = None
        self._tty_fd = None
        self._tty_old_settings = None

        if self._tty_mode:
            try:
                # Prefer controlling terminal directly for better compatibility.
                self._tty_stream = open("/dev/tty", "rb", buffering=0)
                self._tty_fd = self._tty_stream.fileno()
            except OSError:
                if sys.stdin.isatty():
                    self._tty_fd = sys.stdin.fileno()
                else:
                    self._tty_mode = False

        if self._tty_mode:
            self._thread = threading.Thread(target=self._keyboard_loop, daemon=True)
            self._thread.start()
            atexit.register(self._restore_tty_if_needed)
            self._print_help()
            self._print_status()
        else:
            self._thread = None
            print("Mock keyboard XR enabled, but stdin is not a TTY. Input will stay static.")

    def _print_help(self):
        print(
            "\n[Mock Keyboard XR]\n"
            "Select source: 1=left_controller, 2=right_controller, 3=headset\n"
            "Tip:          press 1/2 to select which arm to control (exclusive grip on by default)\n"
            "Move source:   W/S(+/-X), A/D(+/-Y), R/F(+/-Z)\n"
            "Rotate source: I/K(+/-pitch), J/L(+/-yaw), U/O(+/-roll)\n"
            "Grip toggle:   G (selected controller)\n"
            "Trigger:       T increase, Y decrease (selected controller)\n"
            "Buttons:       B press B, P press right_axis_click\n"
            "Step tuning:   ] increase step, [ decrease step\n"
            "Help:          H\n"
        )

    def _print_status(self):
        with self._lock:
            src = self._selected_source
            pose = self._poses[src]
            left_grip = self._key_values["left_grip"]
            right_grip = self._key_values["right_grip"]
            left_trigger = self._key_values["left_trigger"]
            right_trigger = self._key_values["right_trigger"]
            print(
                f"[MockXR] src={src} "
                f"pos=({pose[0]:+.3f},{pose[1]:+.3f},{pose[2]:+.3f}) "
                f"grip(L/R)=({left_grip:.1f}/{right_grip:.1f}) "
                f"trigger(L/R)=({left_trigger:.1f}/{right_trigger:.1f}) "
                f"step=({self._steps.position:.3f}m,{self._steps.rotation_deg:.1f}deg)"
            )

    def _pulse_button(self, name: str, duration: float = 0.15):
        self._button_deadline[name] = time.time() + duration

    def _selected_controller_prefix(self) -> str | None:
        if self._selected_source == "left_controller":
            return "left"
        if self._selected_source == "right_controller":
            return "right"
        return None

    def _apply_position_delta(self, dx: float, dy: float, dz: float):
        pose = self._poses[self._selected_source]
        pose[:3] += np.array([dx, dy, dz], dtype=float)

    def _apply_rotation_delta(self, roll: float, pitch: float, yaw: float):
        pose = self._poses[self._selected_source]
        q_curr = np.array([pose[6], pose[3], pose[4], pose[5]], dtype=float)  # wxyz

        q_roll = tf.quaternion_about_axis(roll, [1, 0, 0])
        q_pitch = tf.quaternion_about_axis(pitch, [0, 1, 0])
        q_yaw = tf.quaternion_about_axis(yaw, [0, 0, 1])
        q_delta = tf.quaternion_multiply(tf.quaternion_multiply(q_yaw, q_pitch), q_roll)
        q_next = tf.quaternion_multiply(q_delta, q_curr)
        q_next = q_next / np.linalg.norm(q_next)

        pose[3] = q_next[1]
        pose[4] = q_next[2]
        pose[5] = q_next[3]
        pose[6] = q_next[0]

    def _apply_key(self, key: str):
        pos_step = self._steps.position
        rot_step = np.deg2rad(self._steps.rotation_deg)

        with self._lock:
            if key == "1":
                self._selected_source = "left_controller"
                if self._exclusive_grip:
                    self._key_values["left_grip"] = 1.0
                    self._key_values["right_grip"] = 0.0
                return
            if key == "2":
                self._selected_source = "right_controller"
                if self._exclusive_grip:
                    self._key_values["left_grip"] = 0.0
                    self._key_values["right_grip"] = 1.0
                return
            if key == "3":
                self._selected_source = "headset"
                return

            if key == "w":
                self._apply_position_delta(pos_step, 0.0, 0.0)
            elif key == "s":
                self._apply_position_delta(-pos_step, 0.0, 0.0)
            elif key == "a":
                self._apply_position_delta(0.0, pos_step, 0.0)
            elif key == "d":
                self._apply_position_delta(0.0, -pos_step, 0.0)
            elif key == "r":
                self._apply_position_delta(0.0, 0.0, pos_step)
            elif key == "f":
                self._apply_position_delta(0.0, 0.0, -pos_step)
            elif key == "i":
                self._apply_rotation_delta(0.0, rot_step, 0.0)
            elif key == "k":
                self._apply_rotation_delta(0.0, -rot_step, 0.0)
            elif key == "j":
                self._apply_rotation_delta(0.0, 0.0, rot_step)
            elif key == "l":
                self._apply_rotation_delta(0.0, 0.0, -rot_step)
            elif key == "u":
                self._apply_rotation_delta(rot_step, 0.0, 0.0)
            elif key == "o":
                self._apply_rotation_delta(-rot_step, 0.0, 0.0)
            elif key == "g":
                prefix = self._selected_controller_prefix()
                if prefix is not None:
                    grip_key = f"{prefix}_grip"
                    self._key_values[grip_key] = 0.0 if self._key_values[grip_key] > 0.5 else 1.0
            elif key == "t":
                prefix = self._selected_controller_prefix()
                if prefix is not None:
                    trigger_key = f"{prefix}_trigger"
                    self._key_values[trigger_key] = min(1.0, self._key_values[trigger_key] + 0.1)
            elif key == "y":
                prefix = self._selected_controller_prefix()
                if prefix is not None:
                    trigger_key = f"{prefix}_trigger"
                    self._key_values[trigger_key] = max(0.0, self._key_values[trigger_key] - 0.1)
            elif key == "b":
                self._pulse_button("B")
            elif key == "p":
                self._pulse_button("right_axis_click")
            elif key == "]":
                self._steps.position = min(0.10, self._steps.position + 0.005)
                self._steps.rotation_deg = min(20.0, self._steps.rotation_deg + 1.0)
            elif key == "[":
                self._steps.position = max(0.001, self._steps.position - 0.005)
                self._steps.rotation_deg = max(0.5, self._steps.rotation_deg - 1.0)
            elif key == "h":
                self._print_help()

        if key in {
            "1", "2", "3",
            "w", "a", "s", "d", "r", "f",
            "i", "j", "k", "l", "u", "o",
            "g", "t", "y", "[", "]",
        }:
            self._print_status()

    def _restore_tty_if_needed(self):
        if self._tty_fd is None or self._tty_old_settings is None:
            return
        try:
            termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, self._tty_old_settings)
        except termios.error:
            pass

    def _keyboard_loop(self):
        fd = self._tty_fd
        old_settings = termios.tcgetattr(fd)
        self._tty_old_settings = old_settings
        tty.setcbreak(fd)
        try:
            while self._running:
                ready, _, _ = select.select([fd], [], [], 0.05)
                if not ready:
                    continue
                key = os.read(fd, 1).decode(errors="ignore")
                if not key:
                    continue
                self._apply_key(key.lower())
        finally:
            self._restore_tty_if_needed()

    def get_pose_by_name(self, name: str) -> np.ndarray:
        if name not in self._poses:
            raise ValueError(f"Invalid name: {name}")
        with self._lock:
            return self._poses[name].copy()

    def get_key_value_by_name(self, name: str) -> float:
        if name not in self._key_values:
            raise ValueError(f"Invalid key name: {name}")
        with self._lock:
            return float(self._key_values[name])

    def get_button_state_by_name(self, name: str) -> bool:
        with self._lock:
            deadline = self._button_deadline.get(name, 0.0)
            return time.time() < deadline

    def get_timestamp_ns(self) -> int:
        return time.time_ns()

    def get_hand_tracking_state(self, hand: str):
        _ = hand
        return None

    def get_joystick_state(self, controller: str) -> list[float]:
        if controller not in self._joysticks:
            raise ValueError(f"Invalid controller: {controller}")
        with self._lock:
            return list(self._joysticks[controller])

    def get_motion_tracker_data(self) -> dict:
        return {}

    def get_body_tracking_data(self):
        return None

    def close(self):
        self._running = False
        self._restore_tty_if_needed()
        if self._tty_stream is not None:
            try:
                self._tty_stream.close()
            except OSError:
                pass
