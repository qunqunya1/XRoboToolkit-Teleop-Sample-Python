from pathlib import Path
import sys

import tyro

def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from xrobotoolkit_teleop.hardware.x2_ros2_teleop_controller import (
    DEFAULT_ARM_COMMAND_TOPIC,
    DEFAULT_ARM_STATE_TOPIC,
    DEFAULT_HAND_COMMAND_TOPIC,
    DEFAULT_HEAD_COMMAND_TOPIC,
    DEFAULT_HEAD_STATE_TOPIC,
    DEFAULT_X2_CAMERA_COLOR_TOPICS,
    DEFAULT_X2_CAMERA_DEPTH_TOPICS,
    DEFAULT_X2_MANIPULATOR_CONFIG,
    DEFAULT_X2_UPPER_BODY_URDF_PATH,
    X2Ros2TeleopController,
)


def main(
    robot_urdf_path: str = DEFAULT_X2_UPPER_BODY_URDF_PATH,
    scale_factor: float = 1.2,
    enable_log_data: bool = True,
    visualize_placo: bool = False,
    control_rate_hz: int = 30,
    log_freq: float = 30.0,
    log_dir: str = "logs/x2_hardware_logs",
    validate_log_before_save: bool = True,
    decode_images_on_log_validate: bool = True,
    enable_camera: bool = True,
    camera_fps: int = 30,
    show_camera_window: bool = False,
    camera_color_topics: str = DEFAULT_X2_CAMERA_COLOR_TOPICS,
    camera_depth_topics: str = DEFAULT_X2_CAMERA_DEPTH_TOPICS,
    camera_width: int = 1280,
    camera_height: int = 720,
    camera_enable_depth: bool = False,
    camera_enable_compression: bool = True,
    camera_jpg_quality: int = 85,
    camera_raw_passthrough_for_logging: bool = False,
    enable_head_tracking: bool = False,
    enable_head_state_feedback: bool = False,
    head_yaw_scale: float = 1.0,
    head_pitch_scale: float = 1.0,
    arm_state_topic: str = DEFAULT_ARM_STATE_TOPIC,
    arm_command_topic: str = DEFAULT_ARM_COMMAND_TOPIC,
    head_state_topic: str = DEFAULT_HEAD_STATE_TOPIC,
    head_command_topic: str = DEFAULT_HEAD_COMMAND_TOPIC,
    hand_command_topic: str = DEFAULT_HAND_COMMAND_TOPIC,
    initial_state_timeout_s: float = 10.0,
    software_estop_button: str = "right_menu_button",
    software_estop_hold_s: float = 0.5,
    max_arm_joint_step_rad: float = 1.0,
    max_head_joint_step_rad: float = 0.05,
    debug_print_targets: bool = False,
    debug_print_hz: float = 2.0,
    enable_ruckig_smoothing: bool = False,
    arm_ruckig_max_velocity: float = 1.2,
    arm_ruckig_max_acceleration: float = 4.0,
    arm_ruckig_max_jerk: float = 40.0,
    head_ruckig_max_velocity: float = 0.8,
    head_ruckig_max_acceleration: float = 2.5,
    head_ruckig_max_jerk: float = 25.0,
):
    """Run X2 upper-body teleoperation on hardware via ROS2 joint topics."""

    controller = X2Ros2TeleopController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=DEFAULT_X2_MANIPULATOR_CONFIG,
        scale_factor=scale_factor,
        enable_log_data=enable_log_data,
        visualize_placo=visualize_placo,
        control_rate_hz=control_rate_hz,
        log_freq=log_freq,
        log_dir=log_dir,
        validate_log_before_save=validate_log_before_save,
        decode_images_on_log_validate=decode_images_on_log_validate,
        enable_camera=enable_camera,
        camera_fps=camera_fps,
        show_camera_window=show_camera_window,
        camera_color_topics=camera_color_topics,
        camera_depth_topics=camera_depth_topics,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_enable_depth=camera_enable_depth,
        camera_enable_compression=camera_enable_compression,
        camera_jpg_quality=camera_jpg_quality,
        camera_raw_passthrough_for_logging=camera_raw_passthrough_for_logging,
        enable_head_tracking=enable_head_tracking,
        enable_head_state_feedback=enable_head_state_feedback,
        head_yaw_scale=head_yaw_scale,
        head_pitch_scale=head_pitch_scale,
        arm_state_topic=arm_state_topic,
        arm_command_topic=arm_command_topic,
        head_state_topic=head_state_topic,
        head_command_topic=head_command_topic,
        hand_command_topic=hand_command_topic,
        initial_state_timeout_s=initial_state_timeout_s,
        software_estop_button=software_estop_button,
        software_estop_hold_s=software_estop_hold_s,
        max_arm_joint_step_rad=max_arm_joint_step_rad,
        max_head_joint_step_rad=max_head_joint_step_rad,
        debug_print_targets=debug_print_targets,
        debug_print_hz=debug_print_hz,
        enable_ruckig_smoothing=enable_ruckig_smoothing,
        arm_ruckig_max_velocity=arm_ruckig_max_velocity,
        arm_ruckig_max_acceleration=arm_ruckig_max_acceleration,
        arm_ruckig_max_jerk=arm_ruckig_max_jerk,
        head_ruckig_max_velocity=head_ruckig_max_velocity,
        head_ruckig_max_acceleration=head_ruckig_max_acceleration,
        head_ruckig_max_jerk=head_ruckig_max_jerk,
    )
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
