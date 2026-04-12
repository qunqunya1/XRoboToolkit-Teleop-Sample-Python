from pathlib import Path
import sys
import time
from typing import Dict, Iterable

import mujoco
import numpy as np
import tyro
from mujoco import viewer as mj_viewer


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from xrobotoolkit_teleop.hardware.x2_ros2_teleop_controller import (  # noqa: E402
    DEFAULT_ARM_STATE_TOPIC,
    DEFAULT_HEAD_STATE_TOPIC,
)
from xrobotoolkit_teleop.utils.mujoco_utils import (  # noqa: E402
    calc_mujoco_ctrl_from_qpos,
    set_mujoco_joint_pos_by_name,
)

try:
    import rclpy
    from aimdk_msgs.msg import JointStateArray
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
except ImportError as exc:  # pragma: no cover - depends on ROS2 runtime
    rclpy = None
    JointStateArray = None
    Node = object
    _ROS2_IMPORT_ERROR = exc
else:
    _ROS2_IMPORT_ERROR = None


ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]

HEAD_JOINT_NAMES = [
    "head_yaw_joint",
    "head_pitch_joint",
]

STATIC_JOINT_NAMES = [
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
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
]


if rclpy is not None:
    STATE_SUB_QOS = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )
else:
    STATE_SUB_QOS = None


class X2StateMirrorNode(Node):
    def __init__(
        self,
        arm_state_topic: str,
        head_state_topic: str,
        tracked_joints: Iterable[str],
    ):
        super().__init__("x2_state_mirror")
        self.tracked_joints = set(tracked_joints)
        self.joint_positions: Dict[str, float] = {}
        self.joint_velocities: Dict[str, float] = {}
        self.arm_msg_count = 0
        self.head_msg_count = 0
        self.last_update_time = 0.0

        self._arm_sub = self.create_subscription(
            JointStateArray,
            arm_state_topic,
            self._arm_callback,
            STATE_SUB_QOS,
        )
        self._head_sub = self.create_subscription(
            JointStateArray,
            head_state_topic,
            self._head_callback,
            STATE_SUB_QOS,
        )

    @staticmethod
    def _iter_state_entries(msg):
        for attr_name in ("joints", "joints_state", "states"):
            if hasattr(msg, attr_name):
                entries = getattr(msg, attr_name)
                if entries is not None:
                    return entries
        return []

    def _update_from_msg(self, msg) -> int:
        count = 0
        for joint in self._iter_state_entries(msg):
            joint_name = getattr(joint, "name", "")
            if joint_name not in self.tracked_joints:
                continue
            self.joint_positions[joint_name] = float(getattr(joint, "position", 0.0))
            self.joint_velocities[joint_name] = float(getattr(joint, "velocity", 0.0))
            count += 1
        if count > 0:
            self.last_update_time = time.time()
        return count

    def _arm_callback(self, msg):
        updated = self._update_from_msg(msg)
        if updated > 0:
            self.arm_msg_count += 1

    def _head_callback(self, msg):
        updated = self._update_from_msg(msg)
        if updated > 0:
            self.head_msg_count += 1


def _infer_actuator_modes(mj_model: mujoco.MjModel) -> list[bool]:
    actuator_mode_direct_pos: list[bool] = []
    for i in range(mj_model.nu):
        is_simple_motor = (
            mj_model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED
            and mj_model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE
        )
        actuator_mode_direct_pos.append(not is_simple_motor)
    return actuator_mode_direct_pos


def _build_ctrl_from_qpos(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    qpos_desired: np.ndarray,
    actuator_mode_direct_pos: list[bool],
    motor_servo_kp: float,
    motor_servo_kd: float,
) -> np.ndarray:
    ctrl = np.zeros(mj_model.nu)
    for i in range(mj_model.nu):
        joint_id = mj_model.actuator_trnid[i, 0]
        if joint_id < 0:
            continue

        qpos_addr = mj_model.jnt_qposadr[joint_id]
        qvel_addr = mj_model.jnt_dofadr[joint_id]
        q_des = qpos_desired[qpos_addr]
        q_cur = mj_data.qpos[qpos_addr]
        qd_cur = mj_data.qvel[qvel_addr]

        if actuator_mode_direct_pos[i]:
            u = q_des
        else:
            u = motor_servo_kp * (q_des - q_cur) - motor_servo_kd * qd_cur

        if mj_model.actuator_ctrllimited[i]:
            umin, umax = mj_model.actuator_ctrlrange[i]
            u = float(np.clip(u, umin, umax))
        ctrl[i] = u
    return ctrl


def main(
    xml_path: str = str(_find_repo_root() / "X2_URDF" / "scene_upper_body_position.xml"),
    arm_state_topic: str = DEFAULT_ARM_STATE_TOPIC,
    head_state_topic: str = DEFAULT_HEAD_STATE_TOPIC,
    sim_steps_per_cycle: int = 6,
    viewer_distance: float = 2.0,
    viewer_azimuth: float = 0.0,
    viewer_elevation: float = -50.0,
    viewer_lookat_x: float = 0.2,
    viewer_lookat_y: float = 0.0,
    viewer_lookat_z: float = 0.0,
    print_hz: float = 1.0,
    warn_if_stale_after_s: float = 1.0,
    motor_servo_kp: float = 40.0,
    motor_servo_kd: float = 4.0,
):
    """Mirror X2 hardware state topics into MuJoCo."""

    if _ROS2_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ROS2 dependencies are unavailable. Please source your ROS2 workspace "
            "so rclpy and aimdk_msgs can be imported."
        ) from _ROS2_IMPORT_ERROR

    if not rclpy.ok():
        rclpy.init(args=None)

    tracked_joints = ARM_JOINT_NAMES + HEAD_JOINT_NAMES
    node = X2StateMirrorNode(
        arm_state_topic=arm_state_topic,
        head_state_topic=head_state_topic,
        tracked_joints=tracked_joints,
    )

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    actuator_mode_direct_pos = _infer_actuator_modes(mj_model)

    mujoco.mj_resetData(mj_model, mj_data)
    home_key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key_id != -1:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, home_key_id)
    else:
        print("Warning: keyframe 'home' not found. Using MuJoCo default reset pose.")
    mujoco.mj_forward(mj_model, mj_data)

    locked_static_qpos = {}
    for joint_name in STATIC_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            continue
        qpos_addr = mj_model.jnt_qposadr[joint_id]
        locked_static_qpos[joint_name] = float(mj_data.qpos[qpos_addr])

    qpos_desired = mj_data.qpos.copy()
    if mj_model.nu == len(qpos_desired):
        mj_data.ctrl[:] = calc_mujoco_ctrl_from_qpos(mj_model, qpos_desired)

    print("Starting X2 hardware-state mirror in MuJoCo...")
    print(f"  arm_state_topic: {arm_state_topic}")
    print(f"  head_state_topic: {head_state_topic}")
    print(f"  tracked joints: {len(tracked_joints)}")
    print(f"  sim_steps_per_cycle: {sim_steps_per_cycle}")

    last_print_time = 0.0
    stale_warned = False

    try:
        with mj_viewer.launch_passive(mj_model, mj_data) as viewer:
            viewer.cam.azimuth = viewer_azimuth
            viewer.cam.elevation = viewer_elevation
            viewer.cam.distance = viewer_distance
            viewer.cam.lookat = [viewer_lookat_x, viewer_lookat_y, viewer_lookat_z]

            while True:
                rclpy.spin_once(node, timeout_sec=0.0)

                qpos_desired[:] = mj_data.qpos
                for joint_name, joint_value in node.joint_positions.items():
                    set_mujoco_joint_pos_by_name(mj_model, qpos_desired, joint_name, joint_value)
                for joint_name, joint_value in locked_static_qpos.items():
                    set_mujoco_joint_pos_by_name(mj_model, qpos_desired, joint_name, joint_value)

                ctrl = _build_ctrl_from_qpos(
                    mj_model=mj_model,
                    mj_data=mj_data,
                    qpos_desired=qpos_desired,
                    actuator_mode_direct_pos=actuator_mode_direct_pos,
                    motor_servo_kp=motor_servo_kp,
                    motor_servo_kd=motor_servo_kd,
                )
                mj_data.ctrl[:] = ctrl

                for _ in range(max(1, int(sim_steps_per_cycle))):
                    mujoco.mj_step(mj_model, mj_data)
                    for joint_name, joint_value in locked_static_qpos.items():
                        set_mujoco_joint_pos_by_name(mj_model, mj_data.qpos, joint_name, joint_value)
                    mujoco.mj_forward(mj_model, mj_data)

                now = time.time()
                if print_hz > 0.0 and (now - last_print_time) >= (1.0 / print_hz):
                    last_print_time = now
                    ordered = ", ".join(
                        f"{joint_name}={node.joint_positions[joint_name]:+.3f}"
                        for joint_name in tracked_joints
                        if joint_name in node.joint_positions
                    )
                    print(
                        "State mirror status: "
                        f"arm_msgs={node.arm_msg_count}, head_msgs={node.head_msg_count}, "
                        f"tracked={len(node.joint_positions)}, joints=[{ordered}]"
                    )

                is_stale = (
                    warn_if_stale_after_s > 0.0
                    and node.last_update_time > 0.0
                    and (now - node.last_update_time) > warn_if_stale_after_s
                )
                if is_stale and not stale_warned:
                    print(
                        "Warning: state topics look stale. "
                        f"No fresh state seen for {now - node.last_update_time:.2f}s."
                    )
                    stale_warned = True
                elif not is_stale:
                    stale_warned = False

                viewer.sync()
    except KeyboardInterrupt:
        print("\nState mirror stopped.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
