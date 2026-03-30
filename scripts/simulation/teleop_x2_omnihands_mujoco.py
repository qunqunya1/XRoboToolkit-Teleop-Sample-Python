from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import tyro

from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import MujocoTeleopController


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_revolute_joint_limits(urdf_path: str) -> dict[str, tuple[float, float]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    limits: dict[str, tuple[float, float]] = {}
    # Parse namespace-agnostic URDF tags.
    for joint in root.iter():
        if not joint.tag.endswith("joint"):
            continue
        if joint.get("type") != "revolute":
            continue
        name = joint.get("name")
        limit = None
        for child in joint:
            if child.tag.endswith("limit"):
                limit = child
                break
        if name is None or limit is None:
            continue
        lower = float(limit.get("lower", "0"))
        upper = float(limit.get("upper", "0"))
        limits[name] = (lower, upper)
    return limits


def _make_abs_mesh_urdf(urdf_path: str) -> str:
    """
    Create a temporary URDF with mesh filenames rewritten to absolute paths,
    so parsers that do not resolve relative paths against the URDF directory
    (e.g. some placo setups) can still load meshes.
    """
    urdf_file = Path(urdf_path).resolve()
    urdf_dir = urdf_file.parent

    tree = ET.parse(str(urdf_file))
    root = tree.getroot()
    modified = False

    for elem in root.iter():
        if not elem.tag.endswith("mesh"):
            continue
        filename = elem.get("filename")
        if not filename:
            continue
        mesh_path = Path(filename)
        if mesh_path.is_absolute():
            continue
        abs_mesh = (urdf_dir / mesh_path).resolve()
        elem.set("filename", str(abs_mesh))
        modified = True

    if not modified:
        return str(urdf_file)

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_abs_mesh.urdf",
        prefix="x2_omnihands_",
        delete=False,
    )
    with tmp:
        tree.write(tmp.name, encoding="unicode")
    return tmp.name


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _open_close_from_limits(lower: float, upper: float) -> tuple[float, float]:
    # "Open" is chosen as the bound closer to neutral angle 0.
    if abs(lower) <= abs(upper):
        return lower, upper
    return upper, lower


def _build_omnihand_driver_config(
    joint_limits: dict[str, tuple[float, float]],
    side_prefix: str,
    trigger_name: str,
) -> dict:
    def pick_joint(primary: str, fallback: str) -> str:
        if primary in joint_limits:
            return primary
        if fallback in joint_limits:
            print(
                f"Info: '{primary}' is not revolute in URDF, "
                f"fallback to '{fallback}' for trigger driving."
            )
            return fallback
        raise ValueError(f"Required omnihand joint '{primary}' not found in URDF.")

    active_joint_names = [
        f"{side_prefix}thumb_mcp_joint",
        pick_joint(f"{side_prefix}index_abad_joint", f"{side_prefix}index_pip_joint"),
        pick_joint(f"{side_prefix}middle_abad_joint", f"{side_prefix}middle_pip_joint"),
        pick_joint(f"{side_prefix}ring_abad_joint", f"{side_prefix}ring_pip_joint"),
        pick_joint(f"{side_prefix}pinky_abad_joint", f"{side_prefix}pinky_pip_joint"),
    ]

    open_pos: list[float] = []
    close_pos: list[float] = []
    for joint_name in active_joint_names:
        if joint_name not in joint_limits:
            raise ValueError(f"Required omnihand joint '{joint_name}' not found in URDF.")
        lower, upper = joint_limits[joint_name]
        open_val, close_val = _open_close_from_limits(lower, upper)
        open_pos.append(open_val)
        close_pos.append(close_val)

    thumb_abad_joint = f"{side_prefix}thumb_abad_joint"
    if thumb_abad_joint not in joint_limits:
        raise ValueError(f"Required omnihand joint '{thumb_abad_joint}' not found in URDF.")

    _, thumb_abad_upper = joint_limits[thumb_abad_joint]

    return {
        "type": "parallel",
        "gripper_trigger": trigger_name,
        "joint_names": active_joint_names,
        "open_pos": open_pos,
        "close_pos": close_pos,
        # Only update hand target when arm teleop is active.
        "active_only": True,
        "thumb_abad_joint": thumb_abad_joint,
        "thumb_abad_default": thumb_abad_upper,
    }


def main(
    xml_path: str = str(_find_repo_root() / "X2_URDF" / "scene_upper_body_omnihands_position.xml"),
    robot_urdf_path: str = str(_find_repo_root() / "X2_with_omnihands_URDF" / "x2_ultra_with_omnihands.urdf"),
    ik_urdf_path: str = str(_find_repo_root() / "X2_URDF" / "x2_upper_body_no_waist.urdf"),
    scale_factor: float = 1.0,
    visualize_placo: bool = False,
    left_tracker_serial: str = "",
    right_tracker_serial: str = "",
    control_profile: str = "low_latency",
    sim_steps_per_control: int = 0,
    thumb_abad_angle: float | None = None,
    allow_missing_hand_joints: bool = False,
    enable_log_data: bool = False,
    log_dir: str = "logs/x2_omnihands_upper_body_sim",
    log_freq: float = 30.0,
    enable_camera_log: bool = False,
    camera_names: str = "rgbd_head_front_camera",
    camera_width: int = 640,
    camera_height: int = 480,
    camera_log_freq: float = 10.0,
    camera_jpg_quality: int = 0,
):
    """Run X2 upper-body teleoperation with omnihands trigger capture in MuJoCo."""

    # Use omnihands URDF only for hand joint mapping.
    joint_limits = _load_revolute_joint_limits(robot_urdf_path)
    # Use a dedicated IK URDF for placo (default: upper-body without hand chain).
    placo_urdf_path = _make_abs_mesh_urdf(ik_urdf_path)
    left_gripper = _build_omnihand_driver_config(joint_limits, "L_", "left_trigger")
    right_gripper = _build_omnihand_driver_config(joint_limits, "R_", "right_trigger")

    if thumb_abad_angle is None:
        left_thumb_abad = float(left_gripper["thumb_abad_default"])
        right_thumb_abad = float(right_gripper["thumb_abad_default"])
    else:
        l_lo, l_hi = joint_limits[left_gripper["thumb_abad_joint"]]
        r_lo, r_hi = joint_limits[right_gripper["thumb_abad_joint"]]
        left_thumb_abad = _clamp(float(thumb_abad_angle), l_lo, l_hi)
        right_thumb_abad = _clamp(float(thumb_abad_angle), r_lo, r_hi)

    left_gripper["thumb_abad_target"] = left_thumb_abad
    right_gripper["thumb_abad_target"] = right_thumb_abad

    config = {
        "left_arm": {
            "link_name": "left_wrist_roll_link",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "vis_target": "left_target",
            "gripper_config": left_gripper,
        },
        "right_arm": {
            "link_name": "right_wrist_roll_link",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "vis_target": "right_target",
            "gripper_config": right_gripper,
        },
    }

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
        "low_latency": {"scale_factor": 1.2, "joints_regularization_weight": 5e-5, "sim_steps_per_control": 1},
    }
    if control_profile not in profile_table:
        valid = ", ".join(profile_table.keys())
        raise ValueError(f"Invalid control_profile={control_profile!r}. Choose one of: {valid}")

    profile = profile_table[control_profile]
    scale_factor = profile["scale_factor"] if scale_factor <= 0 else scale_factor
    sim_steps_per_control = profile["sim_steps_per_control"] if sim_steps_per_control <= 0 else int(sim_steps_per_control)

    parsed_camera_names = [c.strip() for c in camera_names.split(",") if c.strip()]
    if enable_camera_log and not parsed_camera_names:
        raise ValueError("enable_camera_log=True but no camera names are provided.")

    if enable_camera_log and not enable_log_data:
        print("Camera logging requested. Enabling data logging automatically.")
        enable_log_data = True

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
        left_gripper["thumb_abad_joint"]: left_thumb_abad,
        right_gripper["thumb_abad_joint"]: right_thumb_abad,
    }

    controller = MujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=placo_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        visualize_placo=visualize_placo,
        lock_floating_base=True,
        static_joint_targets=static_joint_targets,
        hard_lock_static_joints=True,
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
        allow_missing_gripper_joints=allow_missing_hand_joints,
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

    print("Starting X2 upper-body + omnihands teleoperation in MuJoCo...")
    print("Control mapping:")
    print("  - Left controller -> Left arm + left omnihand (left trigger)")
    print("  - Right controller -> Right arm + right omnihand (right trigger)")
    print("  - Hold grip buttons to activate arm control and hand trigger capture")
    print(f"  - control profile: {control_profile}, sim_steps_per_control: {sim_steps_per_control}")
    print(f"  - thumb_abad fixed angle: left={left_thumb_abad:.4f}, right={right_thumb_abad:.4f}")
    print(f"  - allow_missing_hand_joints: {allow_missing_hand_joints}")
    print(f"  - ik_urdf_path: {ik_urdf_path}")
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
