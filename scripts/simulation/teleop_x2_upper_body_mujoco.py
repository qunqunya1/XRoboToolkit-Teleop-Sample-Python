from pathlib import Path
import sys

import tyro

def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import MujocoTeleopController


def main(
    xml_path: str = str(_find_repo_root() / "X2_URDF" / "scene_upper_body_position.xml"),
    robot_urdf_path: str = str(_find_repo_root() / "X2_URDF" / "x2_upper_body_no_waist.urdf"),
    scale_factor: float = 1.0,
    visualize_placo: bool = True,
    left_palm_offset_xyz: tuple[float, float, float] = (0.00, -0.0, -0.1),
    right_palm_offset_xyz: tuple[float, float, float] = (0.00, 0.0, -0.1),
    left_tracker_serial: str = "",
    right_tracker_serial: str = "",
    control_profile: str = "balanced",
    sim_steps_per_control: int = 6,
    enable_log_data: bool = False,
    log_dir: str = "logs/x2_upper_body_sim",
    log_freq: float = 30.0,
    enable_camera_log: bool = False,
    camera_names: str = "rgbd_head_front_camera",
    camera_width: int = 640,
    camera_height: int = 480,
    camera_log_freq: float = 10.0,
    camera_jpg_quality: int = 0,
):
    """Run X2 upper-body teleoperation in MuJoCo using XR controllers."""

    config = {
        "left_arm": {
            "link_name": "left_wrist_roll_link",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "activation_on_frames": 1,
            "activation_off_frames": 4,
            "control_point_offset_xyz": list(left_palm_offset_xyz),
            "input_linear_deadband_m": 0.003,
            "input_angular_deadband_rad": 0.04,
            "input_position_alpha": 0.35,
            "input_rotation_alpha": 0.25,
            "max_target_linear_step_m": 0.03,
            "max_target_angular_step_rad": 0.35,
            "vis_target": "left_target",
        },
        "right_arm": {
            "link_name": "right_wrist_roll_link",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "activation_on_frames": 1,
            "activation_off_frames": 4,
            "control_point_offset_xyz": list(right_palm_offset_xyz),
            "input_linear_deadband_m": 0.003,
            "input_angular_deadband_rad": 0.04,
            "input_position_alpha": 0.35,
            "input_rotation_alpha": 0.25,
            "max_target_linear_step_m": 0.03,
            "max_target_angular_step_rad": 0.35,
            "vis_target": "right_target",
        },
    }

    # Optional motion trackers (unitree_g1-style elbow tracking extension).
    if left_tracker_serial:
        config["left_arm"]["motion_tracker"] = {
            "serial": left_tracker_serial,
            "link_target": "left_elbow_link",
        }
    if right_tracker_serial:
        config["right_arm"]["motion_tracker"] = {
            "serial": right_tracker_serial,
            "link_target": "right_elbow_link",
        }

    profile_table = {
        "stable": {"scale_factor": 1.0, "joints_regularization_weight": 3e-4, "sim_steps_per_control": 4},
        "balanced": {"scale_factor": 1.2, "joints_regularization_weight": 1e-4, "sim_steps_per_control": 6},
        "fast": {"scale_factor": 1.4, "joints_regularization_weight": 5e-5, "sim_steps_per_control": 8},
    }
    if control_profile not in profile_table:
        valid = ", ".join(profile_table.keys())
        raise ValueError(f"Invalid control_profile={control_profile!r}. Choose one of: {valid}")

    profile = profile_table[control_profile]
    scale_factor = profile["scale_factor"] if scale_factor <= 0 else scale_factor
    sim_steps_per_control = max(profile["sim_steps_per_control"], int(sim_steps_per_control))

    parsed_camera_names = [c.strip() for c in camera_names.split(",") if c.strip()]
    if enable_camera_log and not parsed_camera_names:
        raise ValueError("enable_camera_log=True but no camera names are provided.")

    if enable_camera_log and not enable_log_data:
        print("Camera logging requested. Enabling data logging automatically.")
        enable_log_data = True

    # Lock lower-body and waist joints at current startup pose.
    static_joint_targets = {
        "left_hip_pitch_joint": None,
        "left_hip_roll_joint": None,
        "left_hip_yaw_joint": None,
        "left_knee_joint": None,
        "left_ankle_pitch_joint": None,
        "left_ankle_roll_joint": None,
        "right_hip_pitch_joint": None,
        "right_hip_roll_joint": None,
        "right_hip_yaw_joint": None,
        "right_knee_joint": None,
        "right_ankle_pitch_joint": None,
        "right_ankle_roll_joint": None,
        "waist_yaw_joint": None,
        "waist_pitch_joint": None,
        "waist_roll_joint": None,
    }

    controller = MujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        visualize_placo=visualize_placo,
        lock_floating_base=True,
        static_joint_targets=static_joint_targets,
        hard_lock_static_joints=True,
        # x2_upper_body_no_waist.urdf is torso-rooted; offset aligns IK world with MuJoCo world.
        ik_world_offset=[0.0, 0.0, 0.8350706],
        sim_steps_per_control=sim_steps_per_control,
        enable_log_data=enable_log_data,
        log_dir=log_dir,
        log_freq=log_freq,
        enable_camera_log=enable_camera_log,
        camera_names=parsed_camera_names,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_log_freq=camera_log_freq,
        camera_jpg_quality=camera_jpg_quality,
    )

    joints_task = controller.solver.add_joints_task()
    default_joints = {
        "left_shoulder_pitch_joint": 0.3,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": -1.0,
        "left_wrist_yaw_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "right_shoulder_pitch_joint": 0.3,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": -1.0,
        "right_wrist_yaw_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "head_yaw_joint": 0.0,
        "head_pitch_joint": 0.0,
    }
    joints_task.set_joints(default_joints)
    joints_task.configure("joints_regularization", "soft", profile["joints_regularization_weight"])

    print("Starting X2 upper-body teleoperation in MuJoCo...")
    print("Control mapping:")
    print(f"  - Left controller -> Left arm (left_wrist_roll_link + palm offset {left_palm_offset_xyz})")
    print(f"  - Right controller -> Right arm (right_wrist_roll_link + palm offset {right_palm_offset_xyz})")
    print("  - Hold grip buttons to activate arm control")
    print(f"  - control profile: {control_profile}, sim_steps_per_control: {sim_steps_per_control}")
    print(f"  - data logging: {enable_log_data}, log_dir: {log_dir}, log_freq: {log_freq}")
    if enable_camera_log:
        print(
            "  - camera logging: "
            f"names={parsed_camera_names}, size=({camera_width}x{camera_height}), "
            f"freq={camera_log_freq}Hz, jpg_quality={camera_jpg_quality}"
        )
    if left_tracker_serial or right_tracker_serial:
        print("  - Motion tracker enabled for elbow auxiliary tracking")

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
