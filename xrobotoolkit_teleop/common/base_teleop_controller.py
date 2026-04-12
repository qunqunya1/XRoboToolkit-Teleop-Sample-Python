import abc
import threading
import webbrowser
from typing import Any, Dict

import meshcat.transformations as tf
import numpy as np
import placo
from placo_utils.visualization import (
    frame_viz,
    robot_frame_viz,
    robot_viz,
)

from xrobotoolkit_teleop.common.data_logger import DataLogger
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.geometry import (
    apply_delta_pose,
    quat_diff_as_angle_axis,
)
from xrobotoolkit_teleop.utils.parallel_gripper_utils import (
    calc_parallel_gripper_position,
)


class BaseTeleopController(abc.ABC):
    def __init__(
        self,
        robot_urdf_path: str,
        manipulator_config: Dict[str, Dict[str, Any]],
        floating_base: bool,
        R_headset_world: np.ndarray,
        scale_factor: float,
        q_init: np.ndarray,
        dt: float,
        enable_log_data: bool = False,
        log_dir: str = "logs",
        log_freq: float = 50,
    ):
        self.robot_urdf_path = robot_urdf_path
        self.manipulator_config = manipulator_config
        self.floating_base = floating_base
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.q_init = q_init
        self.dt = dt
        self.xr_client = XrClient()

        self.enable_log_data = enable_log_data
        self.log_dir = log_dir
        self.log_freq = log_freq
        if enable_log_data:
            self.data_logger = DataLogger(log_dir=log_dir)

        # Initial poses
        self.ref_ee_xyz = {name: None for name in manipulator_config.keys()}
        self.ref_ee_quat = {name: None for name in manipulator_config.keys()}
        self.ref_controller_xyz = {name: None for name in manipulator_config.keys()}
        self.ref_controller_quat = {name: None for name in manipulator_config.keys()}
        self.filtered_controller_xyz = {name: None for name in manipulator_config.keys()}
        self.filtered_controller_quat = {name: None for name in manipulator_config.keys()}
        self.effector_task = {}
        self.effector_control_mode = {}  # Store control mode for each end effector
        self.active = {}
        self._activation_true_counts = {name: 0 for name in manipulator_config.keys()}
        self._activation_false_counts = {name: 0 for name in manipulator_config.keys()}
        self.gripper_pos_target = {}
        self.gripper_trigger_value = {}
        self.gripper_trigger_when_active = {}

        # Motion tracker support
        self.motion_tracker_task = {}
        self.ref_tracker_xyz = {}  # Store initial tracker positions
        self.ref_robot_xyz = {}  # Store initial robot end-effector positions
        for name, config in self.manipulator_config.items():
            if "gripper_config" in config:
                gripper_config = config["gripper_config"]
                self.gripper_pos_target[name] = {
                    joint_name: joint_pos
                    for joint_name, joint_pos in zip(gripper_config["joint_names"], gripper_config["open_pos"])
                }
                self.gripper_trigger_value[name] = 0.0
                self.gripper_trigger_when_active[name] = None

        self._stop_event = threading.Event()

        self._robot_setup()
        self._placo_setup()

    def _get_current_target_control_pose(
        self,
        name: str,
        config: Dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        if name not in self.effector_task:
            return self._get_controlled_pose(config)

        control_mode = self.effector_control_mode.get(name, "pose")
        if control_mode == "position":
            link_xyz = np.array(self.effector_task[name].target_world, dtype=float).copy()
            _, link_quat = self._get_link_pose(config["link_name"])
        else:
            link_target = np.array(self.effector_task[name].T_world_frame, dtype=float).copy()
            link_xyz = link_target[:3, 3]
            link_quat = tf.quaternion_from_matrix(link_target)

        return self._link_pose_to_control_pose(config, link_xyz, link_quat)

    def _get_control_point_offset(self, config: Dict[str, Any]) -> np.ndarray:
        offset = config.get("control_point_offset_xyz", [0.0, 0.0, 0.0])
        offset_array = np.array(offset, dtype=float)
        if offset_array.shape != (3,):
            raise ValueError(
                f"control_point_offset_xyz for {config.get('link_name', '<unknown>')} must have shape (3,)."
            )
        return offset_array

    def _link_pose_to_control_pose(
        self,
        config: Dict[str, Any],
        link_xyz: np.ndarray,
        link_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        offset_xyz = self._get_control_point_offset(config)
        if np.allclose(offset_xyz, 0.0):
            return link_xyz, link_quat

        link_frame = tf.quaternion_matrix(link_quat)
        link_frame[:3, 3] = link_xyz
        control_xyz = link_xyz + link_frame[:3, :3] @ offset_xyz
        return control_xyz, link_quat

    def _get_controlled_pose(self, config: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Return the user-facing control point pose.

        By default this is the controlled link pose itself. When
        `control_point_offset_xyz` is provided in the config, the control point
        is translated from the link origin in the link's local frame.
        """
        link_xyz, link_quat = self._get_link_pose(config["link_name"])
        return self._link_pose_to_control_pose(config, link_xyz, link_quat)

    def _control_pose_to_link_pose(
        self,
        config: Dict[str, Any],
        control_xyz: np.ndarray,
        control_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a desired control-point pose back to the controlled link pose."""
        offset_xyz = self._get_control_point_offset(config)
        if np.allclose(offset_xyz, 0.0):
            return control_xyz, control_quat

        control_frame = tf.quaternion_matrix(control_quat)
        control_frame[:3, 3] = control_xyz
        link_xyz = control_xyz - control_frame[:3, :3] @ offset_xyz
        return link_xyz, control_quat

    def _apply_target_step_limits(
        self,
        name: str,
        config: Dict[str, Any],
        desired_xyz: np.ndarray,
        desired_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        prev_xyz, prev_quat = self._get_current_target_control_pose(name, config)

        limited_xyz = np.array(desired_xyz, dtype=float).copy()
        limited_quat = np.array(desired_quat, dtype=float).copy()

        max_linear_step = float(config.get("max_target_linear_step_m", 0.0) or 0.0)
        if max_linear_step > 0.0:
            delta_xyz = limited_xyz - prev_xyz
            delta_norm = float(np.linalg.norm(delta_xyz))
            if delta_norm > max_linear_step:
                limited_xyz = prev_xyz + delta_xyz * (max_linear_step / delta_norm)

        max_angular_step = float(config.get("max_target_angular_step_rad", 0.0) or 0.0)
        if max_angular_step > 0.0:
            delta_rot = quat_diff_as_angle_axis(prev_quat, limited_quat)
            delta_angle = float(np.linalg.norm(delta_rot))
            if delta_angle > max_angular_step:
                axis = delta_rot / max(delta_angle, 1.0e-9)
                limited_quat = tf.quaternion_multiply(
                    tf.quaternion_about_axis(max_angular_step, axis),
                    prev_quat,
                )

        return limited_xyz, limited_quat

    def _get_control_signal_value(self, trigger_spec) -> float:
        """Return the strongest activation value for a trigger spec.

        `trigger_spec` can be a single XR input name, a sequence of names, or
        `None` to indicate always-active control.
        """
        if trigger_spec is None:
            return 1.0

        if isinstance(trigger_spec, str):
            trigger_names = [trigger_spec]
        else:
            trigger_names = list(trigger_spec)

        max_value = 0.0
        for trigger_name in trigger_names:
            max_value = max(max_value, float(self.xr_client.get_key_value_by_name(trigger_name)))
        return max_value

    def _is_control_active(self, config: Dict[str, Any]) -> bool:
        activation_mode = str(config.get("activation_mode", "analog")).strip().lower()
        if activation_mode in {"always_on", "always", "on"}:
            return True

        threshold = float(config.get("activation_threshold", 0.9))
        signal_value = self._get_control_signal_value(config.get("control_trigger"))
        return signal_value > threshold

    def _update_control_activation(self, src_name: str, config: Dict[str, Any]) -> bool:
        raw_active = self._is_control_active(config)
        prev_active = bool(self.active.get(src_name, False))

        activation_on_frames = max(1, int(config.get("activation_on_frames", 1)))
        activation_off_frames = max(1, int(config.get("activation_off_frames", 1)))

        if raw_active:
            self._activation_true_counts[src_name] += 1
            self._activation_false_counts[src_name] = 0
            if prev_active:
                return True
            return self._activation_true_counts[src_name] >= activation_on_frames

        self._activation_false_counts[src_name] += 1
        self._activation_true_counts[src_name] = 0
        if not prev_active:
            return False
        return self._activation_false_counts[src_name] < activation_off_frames

    def _process_xr_pose(self, xr_pose, src_name):
        """Process the current XR controller pose."""
        # Get position and orientation
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = [
            xr_pose[6],  # w
            xr_pose[3],  # x
            xr_pose[4],  # y
            xr_pose[5],  # z
        ]

        controller_xyz = self.R_headset_world @ controller_xyz

        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat),
            tf.quaternion_conjugate(R_quat),
        )

        config = self.manipulator_config[src_name]
        controller_xyz, controller_quat = self._filter_controller_pose(
            src_name,
            config,
            controller_xyz,
            np.array(controller_quat, dtype=float),
        )

        if self.ref_controller_xyz[src_name] is None:
            self.ref_controller_xyz[src_name] = controller_xyz
            self.ref_controller_quat[src_name] = controller_quat

            delta_xyz = np.zeros(3)
            delta_rot = np.array([0.0, 0.0, 0.0])
        else:
            delta_xyz = (controller_xyz - self.ref_controller_xyz[src_name]) * self.scale_factor
            delta_rot = quat_diff_as_angle_axis(self.ref_controller_quat[src_name], controller_quat)

        return delta_xyz, delta_rot

    @staticmethod
    def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
        quat = np.array(quat, dtype=float, copy=True)
        norm = float(np.linalg.norm(quat))
        if norm <= 1.0e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        quat /= norm
        if quat[0] < 0.0:
            quat = -quat
        return quat

    def _slerp_quaternion(self, q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        q0 = self._normalize_quaternion(q0)
        q1 = self._normalize_quaternion(q1)

        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        if dot > 0.9995:
            blended = q0 + alpha * (q1 - q0)
            return self._normalize_quaternion(blended)

        theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        sin_theta_0 = float(np.sin(theta_0))
        if sin_theta_0 < 1.0e-9:
            return q0

        theta = theta_0 * alpha
        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        return self._normalize_quaternion((s0 * q0) + (s1 * q1))

    def _filter_controller_pose(
        self,
        src_name: str,
        config: Dict[str, Any],
        controller_xyz: np.ndarray,
        controller_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        prev_xyz = self.filtered_controller_xyz.get(src_name)
        prev_quat = self.filtered_controller_quat.get(src_name)
        if prev_xyz is None or prev_quat is None:
            filtered_xyz = np.array(controller_xyz, dtype=float, copy=True)
            filtered_quat = self._normalize_quaternion(controller_quat)
            self.filtered_controller_xyz[src_name] = filtered_xyz
            self.filtered_controller_quat[src_name] = filtered_quat
            return filtered_xyz, filtered_quat

        linear_deadband = float(config.get("input_linear_deadband_m", 0.0) or 0.0)
        position_alpha = float(np.clip(config.get("input_position_alpha", 1.0), 0.0, 1.0))
        linear_delta = np.array(controller_xyz, dtype=float) - prev_xyz
        linear_norm = float(np.linalg.norm(linear_delta))
        if linear_norm <= linear_deadband:
            filtered_xyz = prev_xyz.copy()
        else:
            effective_xyz = np.array(controller_xyz, dtype=float, copy=True)
            if linear_deadband > 0.0 and linear_norm > 1.0e-9:
                effective_xyz = prev_xyz + linear_delta * ((linear_norm - linear_deadband) / linear_norm)
            filtered_xyz = prev_xyz + position_alpha * (effective_xyz - prev_xyz)

        angular_deadband = float(config.get("input_angular_deadband_rad", 0.0) or 0.0)
        rotation_alpha = float(np.clip(config.get("input_rotation_alpha", 1.0), 0.0, 1.0))
        controller_quat = self._normalize_quaternion(controller_quat)
        delta_rot = quat_diff_as_angle_axis(prev_quat, controller_quat)
        delta_angle = float(np.linalg.norm(delta_rot))
        if delta_angle <= angular_deadband:
            filtered_quat = prev_quat.copy()
        else:
            effective_quat = controller_quat
            if angular_deadband > 0.0 and delta_angle > 1.0e-9:
                axis = delta_rot / delta_angle
                effective_quat = tf.quaternion_multiply(
                    tf.quaternion_about_axis(delta_angle - angular_deadband, axis),
                    prev_quat,
                )
            filtered_quat = self._slerp_quaternion(prev_quat, effective_quat, rotation_alpha)

        self.filtered_controller_xyz[src_name] = filtered_xyz
        self.filtered_controller_quat[src_name] = filtered_quat
        return filtered_xyz, filtered_quat

    def _placo_setup(self):
        """Set up the placo inverse kinematics solver."""
        self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)
        print("Joint names in the Placo model:")
        for joint_name in self.placo_robot.model.names:
            print(f"  {joint_name}")

        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = self.dt
        # self.solver.add_kinetic_energy_regularization_task(1e-6)

        # Set initial configuration
        if self.q_init is not None:
            if self.floating_base:
                self.placo_robot.state.q = self.q_init.copy()
            else:
                self.solver.mask_fbase(True)
                self.placo_robot.state.q[7:] = self.q_init.copy()
        else:
            if not self.floating_base:
                self.solver.mask_fbase(True)
            self.placo_robot.state.q[:7] = np.array([0, 0, 0, 0, 0, 0, 1])  # Identity quaternion for base

        self.placo_robot.update_kinematics()

        # Set up end effector tasks
        for name, config in self.manipulator_config.items():
            # Get control mode (default to "pose" for backward compatibility)
            control_mode = config.get("control_mode", "pose")
            self.effector_control_mode[name] = control_mode
            
            ee_xyz, ee_quat = self._get_link_pose(config["link_name"])
            
            if control_mode == "position":
                # Position-only control
                self.effector_task[name] = self.solver.add_position_task(config["link_name"], ee_xyz)
                print(f"Created position task for {name} -> {config['link_name']}")
            else:
                # Full pose control (default)
                ee_target = tf.quaternion_matrix(ee_quat)
                ee_target[:3, 3] = ee_xyz
                self.effector_task[name] = self.solver.add_frame_task(config["link_name"], ee_target)
                print(f"Created pose task for {name} -> {config['link_name']}")
            
            task_weight = config.get("task_weight", 1.0)
            self.effector_task[name].configure(name, "soft", task_weight)
            manipulability = self.solver.add_manipulability_task(config["link_name"], "both", 1.0)
            manipulability_weight = config.get("manipulability_weight", 1e-2)
            manipulability.configure("manipulability", "soft", manipulability_weight)

            # Set up motion tracker tasks if configured (position only)
            if "motion_tracker" in config:
                tracker_config = config["motion_tracker"]
                link_target = tracker_config["link_target"]

                # Get current position of the target link
                target_xyz, _ = self._get_link_pose(link_target)

                # Create position task for motion tracker target (xyz only)
                tracker_task_name = f"{name}_tracker"
                self.motion_tracker_task[name] = self.solver.add_position_task(link_target, target_xyz)
                self.motion_tracker_task[name].configure(tracker_task_name, "soft", 1.0)

                print(f"Motion tracker position task created for {name} -> {link_target}")

        self.placo_robot.update_kinematics()

    def _update_ik(self):
        """
        This is the core IK logic block. It reads from XR, updates Placo tasks,
        and solves the kinematics.
        """
        self._update_robot_state()
        self.placo_robot.update_kinematics()

        for src_name, config in self.manipulator_config.items():
            self.active[src_name] = self._update_control_activation(src_name, config)

            if self.active[src_name]:
                if self.ref_ee_xyz[src_name] is None:
                    print(f"{src_name} is activated.")
                    self.ref_ee_xyz[src_name], self.ref_ee_quat[src_name] = self._get_controlled_pose(config)

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                delta_xyz, delta_rot = self._process_xr_pose(xr_pose, src_name)

                if self.effector_control_mode[src_name] == "position":
                    # Position-only control: only apply position delta
                    target_xyz = self.ref_ee_xyz[src_name] + delta_xyz
                    _, curr_control_quat = self._get_controlled_pose(config)
                    link_target_xyz, _ = self._control_pose_to_link_pose(
                        config,
                        target_xyz,
                        curr_control_quat,
                    )
                    self.effector_task[src_name].target_world = link_target_xyz
                else:
                    # Full pose control: apply both position and orientation deltas
                    target_xyz, target_quat = apply_delta_pose(
                        self.ref_ee_xyz[src_name],
                        self.ref_ee_quat[src_name],
                        delta_xyz,
                        delta_rot,
                    )
                    target_xyz, target_quat = self._apply_target_step_limits(
                        src_name,
                        config,
                        target_xyz,
                        target_quat,
                    )
                    link_target_xyz, link_target_quat = self._control_pose_to_link_pose(
                        config,
                        target_xyz,
                        target_quat,
                    )
                    target_pose = tf.quaternion_matrix(link_target_quat)
                    target_pose[:3, 3] = link_target_xyz
                    self.effector_task[src_name].T_world_frame = target_pose
            else:
                if self.ref_ee_xyz[src_name] is not None:
                    print(f"{src_name} is deactivated.")
                    self.ref_ee_xyz[src_name] = None
                    self.ref_controller_xyz[src_name] = None
                    self.ref_controller_quat[src_name] = None
                    self.filtered_controller_xyz[src_name] = None
                    self.filtered_controller_quat[src_name] = None
                # Hold inactive end-effector at current pose to avoid cross-arm drift.
                curr_xyz, curr_quat = self._get_link_pose(config["link_name"])
                if self.effector_control_mode[src_name] == "position":
                    self.effector_task[src_name].target_world = curr_xyz
                else:
                    curr_pose = tf.quaternion_matrix(curr_quat)
                    curr_pose[:3, 3] = curr_xyz
                    self.effector_task[src_name].T_world_frame = curr_pose

        # Process motion tracker data
        self._update_motion_tracker_tasks()

        try:
            self.solver.solve(True)
            #print(self.placo_robot.state.q)

        except RuntimeError as e:
            print(f"IK solver failed: {e}")
            # Recover by resetting task targets and teleop references.
            self.sync_end_effector_poses_to_placo_tasks()
            for name in self.manipulator_config.keys():
                self.ref_ee_xyz[name] = None
                self.ref_ee_quat[name] = None
                self.ref_controller_xyz[name] = None
                self.ref_controller_quat[name] = None
                self.filtered_controller_xyz[name] = None
                self.filtered_controller_quat[name] = None

    def _update_motion_tracker_tasks(self):
        """Process motion tracker data and update corresponding Placo tasks."""
        motion_tracker_data = self.xr_client.get_motion_tracker_data()

        for src_name, config in self.manipulator_config.items():
            # Skip if no motion tracker configured for this end effector
            if "motion_tracker" not in config:
                continue

            # Skip if main controller is not active
            if not self.active.get(src_name, False):
                # Reset motion tracker references when controller is inactive
                if src_name in self.ref_tracker_xyz:
                    del self.ref_tracker_xyz[src_name]
                    del self.ref_robot_xyz[src_name]
                continue

            tracker_config = config["motion_tracker"]
            serial = tracker_config["serial"]

            # Skip if this tracker is not available
            if serial not in motion_tracker_data:
                continue

            # Get motion tracker pose
            tracker_pose = motion_tracker_data[serial]["pose"]
            tracker_xyz = self.R_headset_world @ np.array(tracker_pose[:3])

            # Initialize reference positions on first detection
            if src_name not in self.ref_tracker_xyz:
                self.ref_tracker_xyz[src_name] = tracker_xyz.copy()
                # Get current robot end-effector position as baseline
                robot_xyz, _ = self._get_link_pose(config["motion_tracker"]["link_target"])
                self.ref_robot_xyz[src_name] = robot_xyz.copy()
                continue

            # Calculate movement delta from tracker's initial position
            tracker_delta = tracker_xyz - self.ref_tracker_xyz[src_name]

            # Apply scaled tracker movement to robot's initial position
            final_target_xyz = self.ref_robot_xyz[src_name] + tracker_delta * self.scale_factor

            # Update motion tracker task target position
            if src_name in self.motion_tracker_task:
                self.motion_tracker_task[src_name].target_world = final_target_xyz

    def _init_placo_viz(self):
        self.placo_vis = robot_viz(self.placo_robot)
        webbrowser.open(self.placo_vis.viewer.url())
        self.placo_vis.display(self.placo_robot.state.q)
        for name, config in self.manipulator_config.items():
            robot_frame_viz(self.placo_robot, config["link_name"])
            
            # Show appropriate visualization based on control mode
            if self.effector_control_mode[name] == "position":
                # Create a frame matrix for position-only visualization
                target_frame = np.eye(4)
                target_frame[:3, 3] = self.effector_task[name].target_world
                frame_viz(f"vis_target_{name}", target_frame)
            else:
                # Full pose visualization
                link_target = self.effector_task[name].T_world_frame
                link_xyz = link_target[:3, 3].copy()
                link_quat = tf.quaternion_from_matrix(link_target)
                control_xyz, control_quat = self._link_pose_to_control_pose(config, link_xyz, link_quat)
                control_target = tf.quaternion_matrix(control_quat)
                control_target[:3, 3] = control_xyz
                frame_viz(f"vis_target_{name}", control_target)

            # Visualize motion tracker target if configured
            if "motion_tracker" in config and name in self.motion_tracker_task:
                link_target = config["motion_tracker"]["link_target"]
                robot_frame_viz(self.placo_robot, link_target)
                # Create a frame matrix for visualization
                tracker_frame = np.eye(4)
                tracker_frame[:3, 3] = self.motion_tracker_task[name].target_world
                frame_viz(f"vis_tracker_{name}", tracker_frame)

    def _update_placo_viz(self):
        self.placo_vis.display(self.placo_robot.state.q)
        for name, config in self.manipulator_config.items():
            robot_frame_viz(self.placo_robot, config["link_name"])
            
            # Show appropriate visualization based on control mode
            if self.effector_control_mode[name] == "position":
                # Create a frame matrix for position-only visualization
                target_frame = np.eye(4)
                target_frame[:3, 3] = self.effector_task[name].target_world
                frame_viz(f"vis_target_{name}", target_frame)
            else:
                # Full pose visualization
                link_target = self.effector_task[name].T_world_frame
                link_xyz = link_target[:3, 3].copy()
                link_quat = tf.quaternion_from_matrix(link_target)
                control_xyz, control_quat = self._link_pose_to_control_pose(config, link_xyz, link_quat)
                control_target = tf.quaternion_matrix(control_quat)
                control_target[:3, 3] = control_xyz
                frame_viz(f"vis_target_{name}", control_target)

            # Update motion tracker target visualization if configured
            if "motion_tracker" in config and name in self.motion_tracker_task:
                link_target = config["motion_tracker"]["link_target"]
                robot_frame_viz(self.placo_robot, link_target)
                # Create a frame matrix for visualization
                tracker_frame = np.eye(4)
                tracker_frame[:3, 3] = self.motion_tracker_task[name].target_world
                frame_viz(f"vis_tracker_{name}", tracker_frame)

    def sync_end_effector_poses_to_placo_tasks(self):
        """
        Syncs the current end effector link poses to their corresponding placo tasks.
        This is useful for initializing or resetting task targets to current robot state.
        """
        for name, config in self.manipulator_config.items():
            # Get current link pose
            ee_xyz, ee_quat = self._get_link_pose(config["link_name"])
            
            # Update the corresponding placo task
            if self.effector_control_mode[name] == "position":
                # Position-only control: update target position
                self.effector_task[name].target_world = ee_xyz
            else:
                # Full pose control: update target pose
                ee_target = tf.quaternion_matrix(ee_quat)
                ee_target[:3, 3] = ee_xyz
                self.effector_task[name].T_world_frame = ee_target
            
            print(f"Synced {name} end effector pose to placo task: {config['link_name']}")

    def _update_gripper_target(self):
        for gripper_name in self.manipulator_config.keys():
            if "gripper_config" not in self.manipulator_config[gripper_name]:
                continue

            gripper_config = self.manipulator_config[gripper_name]["gripper_config"]
            gripper_type = gripper_config["type"]
            trigger_value = self.xr_client.get_key_value_by_name(gripper_config["gripper_trigger"])
            self.gripper_trigger_value[gripper_name] = float(trigger_value)

            active_only = bool(gripper_config.get("active_only", False))
            if active_only and not self.active.get(gripper_name, False):
                self.gripper_trigger_when_active[gripper_name] = None
                continue
            self.gripper_trigger_when_active[gripper_name] = float(trigger_value)

            if gripper_type == "parallel":
                for joint_name, open_pos, close_pos in zip(
                    gripper_config["joint_names"],
                    gripper_config["open_pos"],
                    gripper_config["close_pos"],
                ):
                    # Calculate the target position based on the trigger value
                    gripper_pos = calc_parallel_gripper_position(open_pos, close_pos, trigger_value)
                    self.gripper_pos_target[gripper_name][joint_name] = gripper_pos
            else:
                # TODO: add dexterous hand support
                raise ValueError(f"Unsupported gripper type: {gripper_type}")

    def _log_data(self):
        """
        Logs the current state of the robot, including joint positions, end effector poses,
        and any other relevant data
        """
        if self.enable_log_data:
            raise NotImplementedError

    # ---------------------------------------------------------
    # --- Abstract Methods (to be implemented by subclasses) ---
    # ---------------------------------------------------------

    @abc.abstractmethod
    def _robot_setup(self):
        """Initializes the specific backend (connects to robot, starts sim, etc.)."""
        raise NotImplementedError

    @abc.abstractmethod
    def _update_robot_state(self):
        """Reads the current joint states from the robot/sim and updates self.placo_robot.state.q."""
        raise NotImplementedError

    @abc.abstractmethod
    def _send_command(self):
        """Sends the calculated target joint positions from self.placo_robot.state.q to the robot/sim."""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_link_pose(self, link_name):
        """Gets the current world pose for a given link name."""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        """
        The main entry point. Subclasses must implement this to define their
        execution model (single-threaded or multi-threaded).
        """
        raise NotImplementedError
