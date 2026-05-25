#!/usr/bin/env python3
"""Tk slider teleop node that publishes X2 dual-arm position commands."""

from __future__ import annotations

import signal
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base_control.x2_bimanual_gravity_compensation import (  # noqa: E402
    BIMANUAL_ARM_JOINTS,
)


@dataclass(frozen=True)
class JointCommandSpec:
    lower: float
    upper: float
    kp: float
    kd: float


DEFAULT_HARDWARE_COMMAND_SPECS: Dict[str, JointCommandSpec] = {
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


class X2BimanualSliderPositionPublisher(Node):
    """Publish upstream position references and track target/current error."""

    def __init__(self) -> None:
        super().__init__("x2_bimanual_slider_position_publisher")
        self.declare_parameter("state_topic", "/aima/hal/joint/arm/state")
        self.declare_parameter("command_topic", "/upper_body/teleop_joint_states")
        self.declare_parameter("publish_period_sec", 0.02)

        self.joint_names = list(BIMANUAL_ARM_JOINTS)
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)
        self.lock = threading.Lock()
        self.current_positions: Dict[str, float] = {name: 0.0 for name in self.joint_names}
        self.current_velocities: Dict[str, float] = {name: 0.0 for name in self.joint_names}
        self.target_positions: Dict[str, float] = {name: 0.0 for name in self.joint_names}
        self.state_received = False
        self.targets_initialized = False
        self.publish_enabled = True
        self.command_sequence = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(JointStateArray, self.state_topic, self._on_joint_states, qos)
        self.pub_cmd = self.create_publisher(JointCommandArray, self.command_topic, qos)
        self.create_timer(
            max(0.005, float(self.get_parameter("publish_period_sec").value)),
            self._publish_commands,
        )

        self.get_logger().info("X2 bimanual slider position publisher started.")
        self.get_logger().info(f"State topic: {self.state_topic}")
        self.get_logger().info(f"Command topic: {self.command_topic}")

    def _on_joint_states(self, msg: JointStateArray) -> None:
        with self.lock:
            for joint in msg.joints:
                if joint.name not in self.current_positions:
                    continue
                self.current_positions[joint.name] = float(joint.position)
                self.current_velocities[joint.name] = float(joint.velocity)

            self.state_received = True
            if not self.targets_initialized:
                for name in self.joint_names:
                    self.target_positions[name] = self.current_positions[name]
                self.targets_initialized = True

    def set_target_position(self, joint_name: str, position: float) -> None:
        spec = DEFAULT_HARDWARE_COMMAND_SPECS[joint_name]
        clipped = min(max(float(position), spec.lower), spec.upper)
        with self.lock:
            self.target_positions[joint_name] = clipped

    def set_publish_enabled(self, enabled: bool) -> None:
        with self.lock:
            self.publish_enabled = bool(enabled)

    def sync_targets_to_current(self) -> Dict[str, float]:
        with self.lock:
            for name in self.joint_names:
                self.target_positions[name] = self.current_positions[name]
            self.targets_initialized = True
            return dict(self.target_positions)

    def snapshot(self) -> tuple[Dict[str, float], Dict[str, float], bool, bool, bool]:
        with self.lock:
            return (
                dict(self.target_positions),
                dict(self.current_positions),
                self.state_received,
                self.targets_initialized,
                self.publish_enabled,
            )

    def _publish_commands(self) -> None:
        with self.lock:
            if not self.state_received or not self.targets_initialized or not self.publish_enabled:
                return
            targets = dict(self.target_positions)

        msg = JointCommandArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.meas_stamp = msg.header.stamp
        msg.header.sequence = self.command_sequence
        self.command_sequence += 1
        for joint_name in self.joint_names:
            spec = DEFAULT_HARDWARE_COMMAND_SPECS[joint_name]
            cmd = JointCommand()
            cmd.name = joint_name
            cmd.position = float(targets[joint_name])
            cmd.velocity = 0.0
            cmd.effort = 0.0
            cmd.stiffness = spec.kp
            cmd.damping = spec.kd
            msg.joints.append(cmd)
        self.pub_cmd.publish(msg)


class SliderGui:
    """Small Tk UI for commanding each arm joint and reading position error."""

    def __init__(self, node: X2BimanualSliderPositionPublisher) -> None:
        self.node = node
        self.root = tk.Tk()
        self.root.title("X2 bimanual joint position sliders")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.updating_sliders = False

        self.enabled_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Waiting for joint state...")
        self.target_labels: Dict[str, tk.StringVar] = {}
        self.current_labels: Dict[str, tk.StringVar] = {}
        self.error_labels: Dict[str, tk.StringVar] = {}
        self.sliders: Dict[str, tk.Scale] = {}

        self._build()
        self.root.after(100, self._refresh)

    def _build(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        toolbar = ttk.Frame(main)
        toolbar.grid(row=0, column=0, columnspan=6, sticky="ew", pady=(0, 8))
        ttk.Checkbutton(
            toolbar,
            text="publish",
            variable=self.enabled_var,
            command=self._on_enabled_changed,
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(toolbar, text="sync to current", command=self._sync_to_current).grid(
            row=0, column=1, padx=(8, 0)
        )
        ttk.Label(toolbar, textvariable=self.status_var).grid(row=0, column=2, padx=(12, 0))

        headers = ("joint", "target", "slider", "current", "error")
        for col, text in enumerate(headers):
            ttk.Label(main, text=text).grid(row=1, column=col, sticky="w", padx=4)

        for row, joint_name in enumerate(self.node.joint_names, start=2):
            spec = DEFAULT_HARDWARE_COMMAND_SPECS[joint_name]
            target_var = tk.StringVar(value="0.000")
            current_var = tk.StringVar(value="0.000")
            error_var = tk.StringVar(value="0.000")
            self.target_labels[joint_name] = target_var
            self.current_labels[joint_name] = current_var
            self.error_labels[joint_name] = error_var

            ttk.Label(main, text=joint_name).grid(row=row, column=0, sticky="w", padx=4)
            ttk.Label(main, textvariable=target_var, width=8).grid(row=row, column=1, padx=4)
            slider = tk.Scale(
                main,
                from_=spec.lower,
                to=spec.upper,
                resolution=0.001,
                orient=tk.HORIZONTAL,
                length=360,
                showvalue=False,
                command=lambda value, name=joint_name: self._on_slider_changed(name, value),
            )
            slider.grid(row=row, column=2, sticky="ew", padx=4)
            self.sliders[joint_name] = slider
            ttk.Label(main, textvariable=current_var, width=8).grid(row=row, column=3, padx=4)
            ttk.Label(main, textvariable=error_var, width=8).grid(row=row, column=4, padx=4)

        main.columnconfigure(2, weight=1)

    def _on_slider_changed(self, joint_name: str, value: str) -> None:
        if self.updating_sliders:
            return
        self.node.set_target_position(joint_name, float(value))

    def _on_enabled_changed(self) -> None:
        self.node.set_publish_enabled(bool(self.enabled_var.get()))

    def _sync_to_current(self) -> None:
        targets = self.node.sync_targets_to_current()
        self._set_sliders(targets)

    def _set_sliders(self, targets: Dict[str, float]) -> None:
        self.updating_sliders = True
        try:
            for joint_name, position in targets.items():
                self.sliders[joint_name].set(position)
        finally:
            self.updating_sliders = False

    def _refresh(self) -> None:
        targets, currents, state_received, targets_initialized, publish_enabled = self.node.snapshot()
        if targets_initialized:
            self._set_sliders(targets)

        for joint_name in self.node.joint_names:
            target = targets[joint_name]
            current = currents[joint_name]
            error = target - current
            self.target_labels[joint_name].set(f"{target: .3f}")
            self.current_labels[joint_name].set(f"{current: .3f}")
            self.error_labels[joint_name].set(f"{error: .3f}")

        self.enabled_var.set(publish_enabled)
        if not state_received:
            self.status_var.set("Waiting for joint state...")
        elif publish_enabled:
            self.status_var.set("Publishing position commands")
        else:
            self.status_var.set("Publishing paused")
        self.root.after(100, self._refresh)

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self.node.set_publish_enabled(False)
        self.root.quit()
        self.root.destroy()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = X2BimanualSliderPositionPublisher()
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    def handle_signal(sig, frame):
        del sig, frame
        gui.close()

    gui = SliderGui(node)
    signal.signal(signal.SIGINT, handle_signal)
    try:
        gui.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        ros_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
