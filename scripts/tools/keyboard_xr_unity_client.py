#!/usr/bin/env python3
"""
Keyboard-driven XR simulator client.

This script mimics the Unity XR client enough for XRoboToolkit-PC-Service to
forward controller state through xrobotoolkit_sdk into the existing teleop
pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import socket
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field


PACKET_HEAD = 0x3F
PACKET_TAIL = 0xA5
CMD_CONNECT = 0x19
CMD_HEARTBEAT = 0x23
CMD_SEND_VERSION = 0x6C
CMD_FUNCTION = 0x6D

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 63901
DEFAULT_DEVICE_SN = "KEYBOARD_SIM"
DEFAULT_SEND_HZ = 60.0
DEFAULT_HEARTBEAT_SEC = 10.0


@dataclass
class PoseState:
    position: list[float]
    quat_xyzw: list[float]


@dataclass
class ControllerState:
    pose: PoseState
    trigger: float = 0.0
    grip: float = 0.0
    menu_button: bool = False
    axis_x: float = 0.0
    axis_y: float = 0.0
    axis_click: bool = False
    primary_button: bool = False
    secondary_button: bool = False
    pulse_deadlines: dict[str, float] = field(default_factory=dict)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_quat_wxyz(quat: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in quat))
    if norm <= 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    return [v / norm for v in quat]


def quat_about_axis(angle_rad: float, axis: tuple[float, float, float]) -> list[float]:
    ax, ay, az = axis
    half = angle_rad * 0.5
    s = math.sin(half)
    return normalize_quat_wxyz([math.cos(half), ax * s, ay * s, az * s])


def quat_multiply(lhs: list[float], rhs: list[float]) -> list[float]:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


class KeyboardXRUnityClient:
    def __init__(
        self,
        host: str,
        port: int,
        device_sn: str,
        send_hz: float,
        pos_step: float,
        rot_step_deg: float,
    ) -> None:
        self.host = host
        self.port = port
        self.device_sn = device_sn
        self.send_hz = max(1.0, send_hz)
        self.send_period = 1.0 / self.send_hz
        self.heartbeat_period = DEFAULT_HEARTBEAT_SEC
        self.pos_step = pos_step
        self.rot_step_rad = math.radians(rot_step_deg)

        self._lock = threading.Lock()
        self._running = True
        self._selected = "right"
        self._socket: socket.socket | None = None
        self._connected = False
        self._tty_stream = None
        self._tty_fd: int | None = None
        self._tty_old = None
        self._last_status_time = 0.0

        self.controllers = {
            "left": ControllerState(
                pose=PoseState(position=[0.30, 0.20, 1.00], quat_xyzw=[0.0, 0.0, 0.0, 1.0]),
            ),
            "right": ControllerState(
                pose=PoseState(position=[0.30, -0.20, 1.00], quat_xyzw=[0.0, 0.0, 0.0, 1.0]),
            ),
        }

    def run(self) -> None:
        self._setup_tty()
        self._print_help()
        self._print_status(force=True)

        sender = threading.Thread(target=self._sender_loop, daemon=True)
        keyboard = threading.Thread(target=self._keyboard_loop, daemon=True)
        sender.start()
        keyboard.start()

        try:
            while self._running:
                time.sleep(0.2)
        except KeyboardInterrupt:
            self._running = False
        finally:
            self.close()
            sender.join(timeout=1.0)
            keyboard.join(timeout=1.0)

    def close(self) -> None:
        self._running = False
        self._restore_tty()
        self._disconnect()

    def _setup_tty(self) -> None:
        if os.name != "posix":
            raise RuntimeError("keyboard_xr_unity_client.py currently requires a POSIX terminal.")

        try:
            self._tty_stream = open("/dev/tty", "rb", buffering=0)
            self._tty_fd = self._tty_stream.fileno()
        except OSError:
            if sys.stdin.isatty():
                self._tty_fd = sys.stdin.fileno()
            else:
                raise RuntimeError("stdin is not a TTY; run this script in an interactive terminal.")

        self._tty_old = termios.tcgetattr(self._tty_fd)
        tty.setcbreak(self._tty_fd)

    def _restore_tty(self) -> None:
        if self._tty_fd is not None and self._tty_old is not None:
            try:
                termios.tcsetattr(self._tty_fd, termios.TCSADRAIN, self._tty_old)
            except termios.error:
                pass
        if self._tty_stream is not None:
            try:
                self._tty_stream.close()
            except OSError:
                pass
            self._tty_stream = None

    def _connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        sock.connect((self.host, self.port))
        sock.settimeout(None)
        self._socket = sock
        self._connected = True
        print(f"[keyboard-xr] connected to {self.host}:{self.port}")
        self._send_packet(CMD_CONNECT, f"{self.device_sn}|-1".encode("utf-8"))
        self._send_packet(CMD_SEND_VERSION, f"{self.device_sn}|1.0|keyboard-sim".encode("utf-8"))

    def _disconnect(self) -> None:
        self._connected = False
        sock = self._socket
        self._socket = None
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def _send_packet(self, cmd: int, payload: bytes) -> None:
        if self._socket is None:
            raise ConnectionError("socket is not connected")
        data = bytearray()
        data.append(PACKET_HEAD)
        data.append(cmd & 0xFF)
        data.extend(len(payload).to_bytes(4, byteorder="little", signed=False))
        data.extend(payload)
        timestamp_ms = int(time.time() * 1000)
        data.extend(timestamp_ms.to_bytes(8, byteorder="little", signed=False))
        data.append(PACKET_TAIL)
        self._socket.sendall(data)

    def _sender_loop(self) -> None:
        next_heartbeat = 0.0
        while self._running:
            if not self._connected:
                try:
                    self._connect()
                    next_heartbeat = 0.0
                except OSError as exc:
                    print(f"[keyboard-xr] connect failed: {exc}")
                    time.sleep(1.0)
                    continue

            now = time.time()
            try:
                if now >= next_heartbeat:
                    self._send_packet(CMD_HEARTBEAT, self.device_sn.encode("utf-8"))
                    next_heartbeat = now + self.heartbeat_period
                payload = self._build_tracking_payload()
                outer = {"functionName": "Tracking", "value": json.dumps(payload, separators=(",", ":"))}
                self._send_packet(CMD_FUNCTION, json.dumps(outer, separators=(",", ":")).encode("utf-8"))
            except OSError as exc:
                print(f"[keyboard-xr] send failed: {exc}")
                self._disconnect()
                time.sleep(1.0)
                continue

            time.sleep(self.send_period)

    def _keyboard_loop(self) -> None:
        assert self._tty_fd is not None
        while self._running:
            ready, _, _ = select.select([self._tty_fd], [], [], 0.05)
            if not ready:
                continue
            raw = os.read(self._tty_fd, 1)
            if not raw:
                continue
            key = raw.decode(errors="ignore")
            if not key:
                continue
            self._apply_key(key)

    def _pulse_button(self, state: ControllerState, button: str, duration: float = 0.15) -> None:
        state.pulse_deadlines[button] = time.time() + duration

    def _set_button_now(self, state: ControllerState, name: str) -> None:
        active = time.time() < state.pulse_deadlines.get(name, 0.0)
        setattr(state, name, active)

    def _refresh_pulses(self) -> None:
        with self._lock:
            for controller in self.controllers.values():
                for name in ("menu_button", "axis_click", "primary_button", "secondary_button"):
                    self._set_button_now(controller, name)

    def _apply_rotation(self, controller: ControllerState, roll: float, pitch: float, yaw: float) -> None:
        curr_xyzw = controller.pose.quat_xyzw
        curr_wxyz = [curr_xyzw[3], curr_xyzw[0], curr_xyzw[1], curr_xyzw[2]]
        delta = quat_multiply(
            quat_multiply(quat_about_axis(yaw, (0.0, 0.0, 1.0)), quat_about_axis(pitch, (0.0, 1.0, 0.0))),
            quat_about_axis(roll, (1.0, 0.0, 0.0)),
        )
        next_wxyz = normalize_quat_wxyz(quat_multiply(delta, curr_wxyz))
        controller.pose.quat_xyzw = [next_wxyz[1], next_wxyz[2], next_wxyz[3], next_wxyz[0]]

    def _apply_key(self, key: str) -> None:
        key = key.lower()
        with self._lock:
            if key == "q":
                self._running = False
                return
            if key == "1":
                self._selected = "left"
                self._print_status(force=True)
                return
            if key == "2":
                self._selected = "right"
                self._print_status(force=True)
                return

            state = self.controllers[self._selected]

            if key == "w":
                state.pose.position[0] += self.pos_step
            elif key == "s":
                state.pose.position[0] -= self.pos_step
            elif key == "a":
                state.pose.position[1] += self.pos_step
            elif key == "d":
                state.pose.position[1] -= self.pos_step
            elif key == "r":
                state.pose.position[2] += self.pos_step
            elif key == "f":
                state.pose.position[2] -= self.pos_step
            elif key == "i":
                self._apply_rotation(state, 0.0, self.rot_step_rad, 0.0)
            elif key == "k":
                self._apply_rotation(state, 0.0, -self.rot_step_rad, 0.0)
            elif key == "j":
                self._apply_rotation(state, 0.0, 0.0, self.rot_step_rad)
            elif key == "l":
                self._apply_rotation(state, 0.0, 0.0, -self.rot_step_rad)
            elif key == "u":
                self._apply_rotation(state, self.rot_step_rad, 0.0, 0.0)
            elif key == "o":
                self._apply_rotation(state, -self.rot_step_rad, 0.0, 0.0)
            elif key == "g":
                state.grip = 0.0 if state.grip > 0.5 else 1.0
            elif key == "t":
                state.trigger = clamp(state.trigger + 0.1, 0.0, 1.0)
            elif key == "y":
                state.trigger = clamp(state.trigger - 0.1, 0.0, 1.0)
            elif key == "8":
                state.axis_y = clamp(state.axis_y + 0.25, -1.0, 1.0)
            elif key == "5":
                state.axis_y = clamp(state.axis_y - 0.25, -1.0, 1.0)
            elif key == "4":
                state.axis_x = clamp(state.axis_x - 0.25, -1.0, 1.0)
            elif key == "6":
                state.axis_x = clamp(state.axis_x + 0.25, -1.0, 1.0)
            elif key == "0":
                state.axis_x = 0.0
                state.axis_y = 0.0
            elif key == "m":
                self._pulse_button(state, "menu_button")
            elif key == "c":
                self._pulse_button(state, "axis_click")
            elif key == "v":
                self._pulse_button(state, "primary_button")
            elif key == "b":
                self._pulse_button(state, "secondary_button")
            elif key == "h":
                self._print_help()
            else:
                return

        self._print_status()

    def _controller_json(self, state: ControllerState) -> dict[str, object]:
        self._set_button_now(state, "menu_button")
        self._set_button_now(state, "axis_click")
        self._set_button_now(state, "primary_button")
        self._set_button_now(state, "secondary_button")
        pose = state.pose.position + state.pose.quat_xyzw
        pose_str = ",".join(f"{value:.6f}" for value in pose)
        return {
            "pose": pose_str,
            "trigger": round(state.trigger, 4),
            "grip": round(state.grip, 4),
            "menuButton": state.menu_button,
            "axisX": round(state.axis_x, 4),
            "axisY": round(state.axis_y, 4),
            "axisClick": state.axis_click,
            "primaryButton": state.primary_button,
            "secondaryButton": state.secondary_button,
        }

    def _build_tracking_payload(self) -> dict[str, object]:
        self._refresh_pulses()
        with self._lock:
            return {
                "Controller": {
                    "left": self._controller_json(self.controllers["left"]),
                    "right": self._controller_json(self.controllers["right"]),
                },
                "timeStampNs": time.time_ns(),
                "Input": 0,
            }

    def _print_help(self) -> None:
        print(
            "\n[keyboard-xr] controls\n"
            "  select: 1=left, 2=right\n"
            "  move:   w/s x, a/d y, r/f z\n"
            "  rotate: i/k pitch, j/l yaw, u/o roll\n"
            "  grip:   g toggle selected grip\n"
            "  trig:   t/y +/- selected trigger\n"
            "  axis:   8/5 y, 4/6 x, 0 reset axis\n"
            "  pulse:  m menu, c axis click, v primary, b secondary\n"
            "  misc:   h help, q quit\n"
        )

    def _print_status(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_status_time < 0.05:
            return
        self._last_status_time = now
        with self._lock:
            left = self.controllers["left"]
            right = self.controllers["right"]
            print(
                "[keyboard-xr] "
                f"selected={self._selected} "
                f"Lpos=({left.pose.position[0]:+.3f},{left.pose.position[1]:+.3f},{left.pose.position[2]:+.3f}) "
                f"Rpos=({right.pose.position[0]:+.3f},{right.pose.position[1]:+.3f},{right.pose.position[2]:+.3f}) "
                f"grip(L/R)=({left.grip:.1f}/{right.grip:.1f}) "
                f"trigger(L/R)=({left.trigger:.1f}/{right.trigger:.1f}) "
                f"axis(L)=({left.axis_x:+.2f},{left.axis_y:+.2f}) "
                f"axis(R)=({right.axis_x:+.2f},{right.axis_y:+.2f})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keyboard-driven XR simulator client for XRoboToolkit-PC-Service."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="PC service host. Default: %(default)s")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="PC service TCP port. Default: %(default)s")
    parser.add_argument(
        "--device-sn",
        default=DEFAULT_DEVICE_SN,
        help="Simulated device serial number used during handshake. Default: %(default)s",
    )
    parser.add_argument("--send-hz", type=float, default=DEFAULT_SEND_HZ, help="Tracking send rate. Default: %(default)s")
    parser.add_argument(
        "--pos-step",
        type=float,
        default=0.01,
        help="Position increment in meters per keypress. Default: %(default)s",
    )
    parser.add_argument(
        "--rot-step-deg",
        type=float,
        default=3.0,
        help="Rotation increment in degrees per keypress. Default: %(default)s",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        KeyboardXRUnityClient(
            host=args.host,
            port=args.port,
            device_sn=args.device_sn,
            send_hz=args.send_hz,
            pos_step=args.pos_step,
            rot_step_deg=args.rot_step_deg,
        ).run()
    except RuntimeError as exc:
        print(f"[keyboard-xr] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
