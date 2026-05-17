from pathlib import Path
import sys
import threading
import time

import tyro


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from xrobotoolkit_teleop.hardware.x2_ros2_teleop_controller import (
    DEFAULT_ARM_COMMAND_TOPIC,
    DEFAULT_ARM_STATE_TOPIC,
    DEFAULT_HEAD_COMMAND_TOPIC,
    DEFAULT_HEAD_STATE_TOPIC,
    DEFAULT_HARDWARE_COMMAND_SPECS,
    DEFAULT_MODEL_TO_HARDWARE_JOINT_MAP,
    ARM_MODEL_JOINT_NAMES,
    HEAD_MODEL_JOINT_NAMES,
    JointCommandSpec,
    MultiThreadedExecutor,
    Ros2JointGroupInterface,
    _ROS2_IMPORT_ERROR,
    rclpy,
)


DEFAULT_LEG_STATE_TOPIC = "/aima/hal/joint/leg/state"
DEFAULT_LEG_COMMAND_TOPIC = "/aima/hal/joint/leg/command"
DEFAULT_WAIST_STATE_TOPIC = "/aima/hal/joint/waist/state"
DEFAULT_WAIST_COMMAND_TOPIC = "/aima/hal/joint/waist/command"

LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

WAIST_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
]

LOWER_BODY_COMMAND_SPECS = {
    "left_hip_pitch_joint": JointCommandSpec(-2.4871, 2.4871, 80.0, 4.0),
    "left_hip_roll_joint": JointCommandSpec(-0.12217, 2.9059, 40.0, 4.0),
    "left_hip_yaw_joint": JointCommandSpec(-1.6842, 3.4296, 30.0, 3.0),
    "left_knee_joint": JointCommandSpec(0.026179, 2.1206, 80.0, 8.0),
    "left_ankle_pitch_joint": JointCommandSpec(-0.80285, 0.45378, 40.0, 4.0),
    "left_ankle_roll_joint": JointCommandSpec(-0.2618, 0.2618, 20.0, 2.0),
    "right_hip_pitch_joint": JointCommandSpec(-2.4871, 2.4871, 80.0, 4.0),
    "right_hip_roll_joint": JointCommandSpec(-2.9059, 0.12217, 40.0, 4.0),
    "right_hip_yaw_joint": JointCommandSpec(-3.4296, 1.6842, 30.0, 3.0),
    "right_knee_joint": JointCommandSpec(0.026179, 2.1206, 80.0, 8.0),
    "right_ankle_pitch_joint": JointCommandSpec(-0.80285, 0.45378, 40.0, 4.0),
    "right_ankle_roll_joint": JointCommandSpec(-0.2618, 0.2618, 20.0, 2.0),
    "waist_yaw_joint": JointCommandSpec(-3.4296, 2.3824, 60.0, 4.0),
    "waist_pitch_joint": JointCommandSpec(-0.17453, 0.17453, 160.0, 4.0),
    "waist_roll_joint": JointCommandSpec(-0.48869, 0.48869, 60.0, 4.0),
}


def _build_joint_name_lists():
    arm_joint_names = [DEFAULT_MODEL_TO_HARDWARE_JOINT_MAP[name] for name in ARM_MODEL_JOINT_NAMES]
    head_joint_names = [DEFAULT_MODEL_TO_HARDWARE_JOINT_MAP[name] for name in HEAD_MODEL_JOINT_NAMES]
    return arm_joint_names, head_joint_names


def _interpolate_targets(
    start_targets: dict[str, float],
    end_targets: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    alpha = float(min(max(alpha, 0.0), 1.0))
    return {
        joint_name: (1.0 - alpha) * start_targets[joint_name] + alpha * end_targets[joint_name]
        for joint_name in start_targets
    }


def _constant_velocities(
    start_targets: dict[str, float],
    end_targets: dict[str, float],
    duration_s: float,
) -> dict[str, float]:
    if duration_s <= 0.0:
        return {joint_name: 0.0 for joint_name in start_targets}
    return {
        joint_name: (end_targets[joint_name] - start_targets[joint_name]) / duration_s
        for joint_name in start_targets
    }


def _clip_targets(
    targets: dict[str, float],
    command_specs: dict[str, JointCommandSpec],
) -> dict[str, float]:
    clipped = {}
    for joint_name, target in targets.items():
        spec = command_specs[joint_name]
        clipped[joint_name] = float(min(max(target, spec.lower_limit), spec.upper_limit))
    return clipped


def _build_zero_targets(
    group_name: str,
    joint_names: list[str],
    command_specs: dict[str, JointCommandSpec],
) -> dict[str, float]:
    targets = {joint_name: 0.0 for joint_name in joint_names}
    if group_name == "arm":
        #targets["left_shoulder_yaw_joint"] = 0.5
        targets["left_shoulder_yaw_joint"] = 0.1
        targets["left_elbow_joint"] = -1.8
        #targets["right_shoulder_yaw_joint"] = -0.5
        targets["right_shoulder_yaw_joint"] = -0.1
        targets["right_elbow_joint"] = -1.8
    return _clip_targets(targets, command_specs)


def main(
    arm_state_topic: str = DEFAULT_ARM_STATE_TOPIC,
    arm_command_topic: str = DEFAULT_ARM_COMMAND_TOPIC,
    head_state_topic: str = DEFAULT_HEAD_STATE_TOPIC,
    head_command_topic: str = DEFAULT_HEAD_COMMAND_TOPIC,
    leg_state_topic: str = DEFAULT_LEG_STATE_TOPIC,
    leg_command_topic: str = DEFAULT_LEG_COMMAND_TOPIC,
    waist_state_topic: str = DEFAULT_WAIST_STATE_TOPIC,
    waist_command_topic: str = DEFAULT_WAIST_COMMAND_TOPIC,
    include_lower_body: bool = True,
    move_duration_s: float = 3.0,
    hold_duration_s: float = 0.5,
    publish_rate_hz: float = 100.0,
    initial_state_timeout_s: float = 10.0,
):
    """Move the X2 joints to the zero pose before teleoperation."""

    if _ROS2_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ROS2 dependencies are unavailable. Please source your ROS2 workspace "
            "so rclpy and aimdk_msgs can be imported."
        ) from _ROS2_IMPORT_ERROR

    arm_joint_names, head_joint_names = _build_joint_name_lists()
    command_specs = DEFAULT_HARDWARE_COMMAND_SPECS.copy()
    command_specs.update(LOWER_BODY_COMMAND_SPECS)

    if not rclpy.ok():
        rclpy.init(args=None)

    group_configs = [
        {
            "group_name": "arm",
            "node_name": "x2_arm_zero_reset_interface",
            "state_topic": arm_state_topic,
            "command_topic": arm_command_topic,
            "joint_names": arm_joint_names,
        },
        {
            "group_name": "head",
            "node_name": "x2_head_zero_reset_interface",
            "state_topic": head_state_topic,
            "command_topic": head_command_topic,
            "joint_names": head_joint_names,
        },
    ]
    if include_lower_body:
        group_configs.extend(
            [
                {
                    "group_name": "leg",
                    "node_name": "x2_leg_zero_reset_interface",
                    "state_topic": leg_state_topic,
                    "command_topic": leg_command_topic,
                    "joint_names": LEG_JOINT_NAMES,
                },
                {
                    "group_name": "waist",
                    "node_name": "x2_waist_zero_reset_interface",
                    "state_topic": waist_state_topic,
                    "command_topic": waist_command_topic,
                    "joint_names": WAIST_JOINT_NAMES,
                },
            ]
        )

    interfaces = {}
    for group in group_configs:
        interfaces[group["group_name"]] = Ros2JointGroupInterface(
            node_name=group["node_name"],
            state_topic=group["state_topic"],
            command_topic=group["command_topic"],
            joint_names=group["joint_names"],
        )

    executor = MultiThreadedExecutor()
    for interface in interfaces.values():
        executor.add_node(interface)
    spin_thread = threading.Thread(target=executor.spin, name="x2_zero_reset_spin", daemon=True)
    spin_thread.start()

    try:
        print("Waiting for initial ROS2 joint states...")
        for group in group_configs:
            interface = interfaces[group["group_name"]]
            if not interface.wait_for_state(initial_state_timeout_s):
                raise RuntimeError(
                    f"Timed out waiting for {group['group_name']} joint state topic: {group['state_topic']}"
                )

        start_targets = {
            group["group_name"]: interfaces[group["group_name"]].get_joint_positions()
            for group in group_configs
        }
        zero_targets = {
            group["group_name"]: _build_zero_targets(
                group["group_name"],
                group["joint_names"],
                command_specs,
            )
            for group in group_configs
        }
        velocities = {
            group["group_name"]: _constant_velocities(
                start_targets[group["group_name"]],
                zero_targets[group["group_name"]],
                move_duration_s,
            )
            for group in group_configs
        }

        print("Moving X2 joints to zero pose...")
        for group in group_configs:
            print(f"  {group['group_name']} joints: {group['joint_names']}")
        print(f"  duration: {move_duration_s:.2f}s, publish_rate: {publish_rate_hz:.1f}Hz")
        if include_lower_body:
            print("  lower body: enabled (legs + waist will also move to zero)")

        start_time = time.time()
        publish_period = 1.0 / max(publish_rate_hz, 1e-6)

        while True:
            elapsed = time.time() - start_time
            alpha = 1.0 if move_duration_s <= 0.0 else min(elapsed / move_duration_s, 1.0)

            for group in group_configs:
                group_name = group["group_name"]
                group_targets = _interpolate_targets(
                    start_targets[group_name],
                    zero_targets[group_name],
                    alpha,
                )
                interfaces[group_name].publish_command(
                    joint_targets=group_targets,
                    joint_velocities=velocities[group_name] if alpha < 1.0 else None,
                    command_specs=command_specs,
                )

            if alpha >= 1.0:
                break
            time.sleep(publish_period)

        hold_end_time = time.time() + max(0.0, hold_duration_s)
        zero_velocities = {
            group["group_name"]: {joint_name: 0.0 for joint_name in group["joint_names"]}
            for group in group_configs
        }
        while time.time() < hold_end_time:
            for group in group_configs:
                group_name = group["group_name"]
                interfaces[group_name].publish_command(
                    joint_targets=zero_targets[group_name],
                    joint_velocities=zero_velocities[group_name],
                    command_specs=command_specs,
                )
            time.sleep(publish_period)

        print("X2 has been moved to the zero pose.")
    finally:
        executor.shutdown()
        for interface in interfaces.values():
            interface.destroy_node()
        if spin_thread.is_alive():
            spin_thread.join(timeout=2.0)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
