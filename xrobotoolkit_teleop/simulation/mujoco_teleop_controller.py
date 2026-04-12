import time
from typing import Any, Dict


import mujoco
import numpy as np
from meshcat import transformations as tf
from mujoco import viewer as mj_viewer

from xrobotoolkit_teleop.common.base_teleop_controller import BaseTeleopController
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
)
from xrobotoolkit_teleop.utils.mujoco_utils import (
    calc_mujoco_ctrl_from_qpos,
    calc_mujoco_qpos_from_placo_q,
    calc_placo_q_from_mujoco_qpos,
    set_mujoco_joint_pos_by_name,
)


class MujocoTeleopController(BaseTeleopController):
    def __init__(
        self,
        xml_path: str,
        robot_urdf_path: str,
        manipulator_config: Dict[str, Dict[str, Any]],
        floating_base=False,
        R_headset_world=R_HEADSET_TO_WORLD,
        visualize_placo=False,
        scale_factor=1.0,
        dt=0.01,
        mj_qpos_init=None,
        enable_log_data: bool = False,
        log_dir: str = "logs/simulation",
        log_freq: float = 50,
        lock_floating_base: bool = False,
        static_joint_targets: Dict[str, float] | None = None,
        hard_lock_static_joints: bool = True,
        freeze_inactive_effectors: bool = False,
        inactive_effector_joint_groups: Dict[str, list[str]] | None = None,
        motor_servo_kp: float = 40.0,
        motor_servo_kd: float = 4.0,
        ik_world_offset: np.ndarray | None = None,
        sim_steps_per_control: int = 1,
        enable_camera_log: bool = False,
        camera_names: list[str] | None = None,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_log_freq: float | None = None,
        camera_jpg_quality: int = 0,
        allow_missing_gripper_joints: bool = False,
    ):
        self.visualize_placo = visualize_placo
        self.xml_path = xml_path
        self.mj_qpos_init = mj_qpos_init
        self._start_time = 0.0
        self._is_logging = False
        self._prev_b_button_state = False
        self._next_log_time = 0.0
        self._last_qpos_desired = None
        self.lock_floating_base = lock_floating_base
        self.static_joint_targets = static_joint_targets or {}
        self.hard_lock_static_joints = hard_lock_static_joints
        self.freeze_inactive_effectors = freeze_inactive_effectors
        self.inactive_effector_joint_groups = inactive_effector_joint_groups or {}
        self._floating_base_qpos_ref = None
        self.motor_servo_kp = motor_servo_kp
        self.motor_servo_kd = motor_servo_kd
        self._actuator_mode_direct_pos: list[bool] = []
        self.sim_steps_per_control = max(1, int(sim_steps_per_control))
        self.enable_camera_log = enable_camera_log
        self.camera_names = list(camera_names) if camera_names else []
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.camera_jpg_quality = int(camera_jpg_quality)
        self.camera_log_freq = float(camera_log_freq) if camera_log_freq else float(log_freq)
        self._next_camera_log_time = 0.0
        self._camera_renderers: dict[str, mujoco.Renderer] = {}
        self.allow_missing_gripper_joints = bool(allow_missing_gripper_joints)
        self._warned_missing_gripper_joints: set[str] = set()
        if ik_world_offset is None:
            self.ik_world_offset = np.zeros(3)
        else:
            self.ik_world_offset = np.array(ik_world_offset, dtype=float).copy()

        # To be initialized later
        self.mj_model = None
        self.mj_data = None
        self.target_mocap_idx = {name: -1 for name in manipulator_config.keys()}

        super().__init__(
            robot_urdf_path,
            manipulator_config,
            floating_base,
            R_headset_world,
            scale_factor,
            q_init=None,
            dt=dt,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
        )

        if visualize_placo:
            self._init_placo_viz()

    def _robot_setup(self):
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        print("Joint names in the Mujoco model:")
        for i in range(self.mj_model.njnt):
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(f"  {joint_name}")
        self._infer_actuator_modes()

        # Configure scene lighting
        self.mj_model.vis.headlight.ambient = [0.4, 0.4, 0.4]
        self.mj_model.vis.headlight.diffuse = [0.8, 0.8, 0.8]
        self.mj_model.vis.headlight.specular = [0.6, 0.6, 0.6]

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.mj_qpos_init is None:
            home_key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if home_key_id != -1:
                mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, home_key_id)
            else:
                print("Warning: keyframe 'home' not found. Using MuJoCo default reset pose.")
        else:
            self.mj_data.qpos[:] = self.mj_qpos_init
            self.mj_data.ctrl[:] = calc_mujoco_ctrl_from_qpos(self.mj_model, self.mj_qpos_init)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._capture_static_references()
        self._enforce_static_constraints()

        if self.enable_camera_log:
            if not self.camera_names:
                for i in range(self.mj_model.ncam):
                    cam_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                    if cam_name:
                        self.camera_names.append(cam_name)

            for cam_name in self.camera_names:
                cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                if cam_id == -1:
                    raise ValueError(f"Camera '{cam_name}' not found in MuJoCo model.")
                self._camera_renderers[cam_name] = mujoco.Renderer(
                    self.mj_model,
                    height=self.camera_height,
                    width=self.camera_width,
                )
            print(
                f"Simulation camera logging enabled: cameras={self.camera_names}, "
                f"size=({self.camera_width}x{self.camera_height}), "
                f"freq={self.camera_log_freq}Hz, jpg_quality={self.camera_jpg_quality}"
            )

        # setup mocap target
        for name, config in self.manipulator_config.items():
            if "vis_target" not in config:
                print(f"Warning: 'vis_target' not found in config for {name}. Skipping mocap setup.")
                continue
            vis_target = config["vis_target"]
            mocap_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, vis_target)
            if mocap_id == -1:
                raise ValueError(f"Mocap body '{vis_target}' not found in the model.")

            if self.mj_model.body_mocapid[mocap_id] == -1:
                raise ValueError(f"Body '{self.vis_target}' is not configured for mocap.")
            else:
                self.target_mocap_idx[name] = self.mj_model.body_mocapid[mocap_id]

            print(f"Mocap ID for '{vis_target}' body: {self.target_mocap_idx[name]}")

    def _infer_actuator_modes(self):
        """
        Infer actuator control mode.
        - direct position mode: ctrl is desired joint position.
        - motor torque mode: ctrl is torque/force, so we run an internal PD servo.
        """
        self._actuator_mode_direct_pos = []
        direct_pos_count = 0
        motor_pd_count = 0
        for i in range(self.mj_model.nu):
            # "motor" actuators are typically (gain=fixed, bias=none).
            is_simple_motor = (
                self.mj_model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED
                and self.mj_model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE
            )
            use_direct_pos = not is_simple_motor
            self._actuator_mode_direct_pos.append(use_direct_pos)
            if use_direct_pos:
                direct_pos_count += 1
            else:
                motor_pd_count += 1
        print(
            f"Actuator control mode: direct_position={direct_pos_count}, "
            f"motor_pd_servo={motor_pd_count} (kp={self.motor_servo_kp}, kd={self.motor_servo_kd})"
        )

    def _capture_static_references(self):
        if self.lock_floating_base:
            self._floating_base_qpos_ref = self.mj_data.qpos[:7].copy()

        # If caller passes None for a joint target, keep current value as lock target.
        for joint_name, target in list(self.static_joint_targets.items()):
            if target is None:
                joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id == -1:
                    continue
                qpos_addr = self.mj_model.jnt_qposadr[joint_id]
                self.static_joint_targets[joint_name] = float(self.mj_data.qpos[qpos_addr])

    def _zero_joint_velocity(self, joint_name: str):
        joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            return
        dof_addr = self.mj_model.jnt_dofadr[joint_id]
        jnt_type = self.mj_model.jnt_type[joint_id]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            self.mj_data.qvel[dof_addr : dof_addr + 6] = 0.0
        elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
            self.mj_data.qvel[dof_addr : dof_addr + 3] = 0.0
        else:
            self.mj_data.qvel[dof_addr] = 0.0

    def _enforce_static_constraints(self):
        if self.lock_floating_base and self._floating_base_qpos_ref is not None:
            self.mj_data.qpos[:7] = self._floating_base_qpos_ref
            # floating_base_joint is free-joint name in x2_ultra
            self._zero_joint_velocity("floating_base_joint")

        if self.hard_lock_static_joints:
            for joint_name, target in self.static_joint_targets.items():
                if target is None:
                    continue
                set_mujoco_joint_pos_by_name(self.mj_model, self.mj_data.qpos, joint_name, float(target))
                self._zero_joint_velocity(joint_name)

        if self.lock_floating_base or (self.hard_lock_static_joints and self.static_joint_targets):
            mujoco.mj_forward(self.mj_model, self.mj_data)

    def _get_joint_position_by_name(self, joint_name: str) -> float | None:
        joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            return None
        qpos_addr = self.mj_model.jnt_qposadr[joint_id]
        if qpos_addr >= len(self.mj_data.qpos):
            return None
        return float(self.mj_data.qpos[qpos_addr])

    def _send_command(self):
        qpos_desired = calc_mujoco_qpos_from_placo_q(
            self.mj_model,
            self.placo_robot,
            self.placo_robot.state.q,
            floating_base=self.floating_base,
            strict_joint_mapping=not self.allow_missing_gripper_joints,
        )

        if self.freeze_inactive_effectors and self.inactive_effector_joint_groups:
            # Hard-hold inactive manipulators at current joint states to avoid cross-arm drift.
            for effector_name, joint_names in self.inactive_effector_joint_groups.items():
                if self.active.get(effector_name, False):
                    continue
                for joint_name in joint_names:
                    joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    if joint_id == -1:
                        continue
                    qpos_addr = self.mj_model.jnt_qposadr[joint_id]
                    qpos_desired[qpos_addr] = self.mj_data.qpos[qpos_addr]

        # Keep selected joints static by commanding fixed targets through the actuator path.
        for joint_name, target in self.static_joint_targets.items():
            if target is None:
                continue
            set_mujoco_joint_pos_by_name(self.mj_model, qpos_desired, joint_name, float(target))

        for gripper_name, gripper_target in self.gripper_pos_target.items():
            for joint_name, joint_pos in gripper_target.items():
                success = set_mujoco_joint_pos_by_name(
                    self.mj_model,
                    qpos_desired,
                    joint_name,
                    joint_pos,
                )
                if not success:
                    if self.allow_missing_gripper_joints:
                        if joint_name not in self._warned_missing_gripper_joints:
                            print(
                                f"Warning: gripper joint '{joint_name}' is not in MuJoCo model. "
                                "Skipping this joint."
                            )
                            self._warned_missing_gripper_joints.add(joint_name)
                        continue
                    raise ValueError(f"Joint '{joint_name}' not found in MuJoCo model.")

        self._last_qpos_desired = qpos_desired.copy()

        ctrl = np.zeros(self.mj_model.nu)
        for i in range(self.mj_model.nu):
            joint_id = self.mj_model.actuator_trnid[i, 0]
            if joint_id < 0:
                continue

            qpos_addr = self.mj_model.jnt_qposadr[joint_id]
            qvel_addr = self.mj_model.jnt_dofadr[joint_id]

            q_des = qpos_desired[qpos_addr]
            q_cur = self.mj_data.qpos[qpos_addr]
            qd_cur = self.mj_data.qvel[qvel_addr]

            if self._actuator_mode_direct_pos[i]:
                u = q_des
            else:
                # PD in joint space for torque/force motor actuators.
                u = self.motor_servo_kp * (q_des - q_cur) - self.motor_servo_kd * qd_cur

            if self.mj_model.actuator_ctrllimited[i]:
                umin, umax = self.mj_model.actuator_ctrlrange[i]
                u = float(np.clip(u, umin, umax))
            ctrl[i] = u

        # Use in-place assignment for MuJoCo control buffers. Some bindings / versions
        # don't reliably apply replacing the property object wholesale.
        self.mj_data.ctrl[:] = ctrl

        if self.visualize_placo:
            self._update_placo_viz()

    def _check_logging_button(self):
        if not self.enable_log_data:
            return

        b_button_state = self.xr_client.get_button_state_by_name("B")
        right_axis_click = self.xr_client.get_button_state_by_name("right_axis_click")

        if b_button_state and not self._prev_b_button_state:
            self._is_logging = not self._is_logging
            if self._is_logging:
                self._next_log_time = 0.0
                print("--- Started simulation data logging ---")
            else:
                print("--- Stopped simulation data logging. Saving data... ---")
                self.data_logger.save()
                self.data_logger.reset()

        if right_axis_click and self._is_logging:
            print("--- Stopped simulation data logging. Discarding data... ---")
            self.data_logger.reset()
            self._is_logging = False

        self._prev_b_button_state = b_button_state

    def _log_data(self):
        if not self.enable_log_data or not self._is_logging:
            return

        now = time.time()
        elapsed = now - self._start_time
        if elapsed < self._next_log_time:
            return
        self._next_log_time = elapsed + (1.0 / self.log_freq)

        entry = {
            "timestamp": elapsed,
            "qpos": self.mj_data.qpos.copy(),
            "qvel": self.mj_data.qvel.copy(),
            "ctrl": self.mj_data.ctrl.copy(),
        }
        if self._last_qpos_desired is not None:
            entry["qpos_des"] = self._last_qpos_desired.copy()

        # Record controller state to keep the teleop input trajectory.
        entry["xr"] = {
            "left_controller_pose": self.xr_client.get_pose_by_name("left_controller"),
            "right_controller_pose": self.xr_client.get_pose_by_name("right_controller"),
            "headset_pose": self.xr_client.get_pose_by_name("headset"),
            "left_grip": self.xr_client.get_key_value_by_name("left_grip"),
            "right_grip": self.xr_client.get_key_value_by_name("right_grip"),
            "left_trigger": self.xr_client.get_key_value_by_name("left_trigger"),
            "right_trigger": self.xr_client.get_key_value_by_name("right_trigger"),
        }
        if self.gripper_pos_target:
            hand_control = {}
            for gripper_name, joint_targets in self.gripper_pos_target.items():
                gripper_cfg = self.manipulator_config.get(gripper_name, {}).get("gripper_config", {})
                driver_joint_names = list(gripper_cfg.get("joint_names", []))
                hand_control[gripper_name] = {
                    "active": bool(self.active.get(gripper_name, False)),
                    "trigger_raw": self.gripper_trigger_value.get(gripper_name),
                    "trigger_when_active": self.gripper_trigger_when_active.get(gripper_name),
                    "driver_joint_targets": {
                        joint_name: joint_targets[joint_name]
                        for joint_name in driver_joint_names
                        if joint_name in joint_targets
                    },
                    "preset_mode": self.gripper_preset_mode.get(gripper_name),
                    "selected_preset_index": self.gripper_preset_selected_index.get(gripper_name),
                    "applied_preset_index": self.gripper_preset_applied_index.get(gripper_name),
                    "preset_joint_targets": (
                        dict(
                            self.gripper_preset_targets.get(gripper_name, [])[self.gripper_preset_selected_index[gripper_name]]
                        )
                        if (
                            gripper_name in self.gripper_preset_selected_index
                            and self.gripper_preset_targets.get(gripper_name)
                        )
                        else {}
                    ),
                    "preset_ratio_values": (
                        dict(
                            self.gripper_preset_ratio_values.get(gripper_name, [])[self.gripper_preset_selected_index[gripper_name]]
                        )
                        if (
                            gripper_name in self.gripper_preset_selected_index
                            and self.gripper_preset_ratio_values.get(gripper_name)
                        )
                        else {}
                    ),
                }
            entry["hand_control"] = hand_control
        image_dict = self._capture_camera_frames(elapsed)
        if image_dict:
            entry["image"] = image_dict
            if self.camera_jpg_quality > 0:
                entry["image_encoding"] = "jpg"

        self.data_logger.add_entry(entry)

    def _capture_camera_frames(self, elapsed: float) -> dict[str, np.ndarray | bytes]:
        if (not self.enable_camera_log) or (not self._camera_renderers):
            return {}

        if elapsed < self._next_camera_log_time:
            return {}
        self._next_camera_log_time = elapsed + (1.0 / self.camera_log_freq)

        frames: dict[str, np.ndarray | bytes] = {}
        for cam_name, renderer in self._camera_renderers.items():
            renderer.update_scene(self.mj_data, camera=cam_name)
            rgb = np.ascontiguousarray(renderer.render()[:, :, :3].astype(np.uint8))
            if self.camera_jpg_quality > 0:
                try:
                    from xrobotoolkit_teleop.utils.image_utils import compress_image_to_jpg
                except Exception as exc:
                    raise RuntimeError(
                        "JPG camera logging requires OpenCV-compatible environment. "
                        "Set camera_jpg_quality=0 to store raw uint8 frames."
                    ) from exc
                frames[cam_name] = compress_image_to_jpg(rgb, quality=self.camera_jpg_quality)
            else:
                frames[cam_name] = rgb
        return frames

    def _close_camera_renderers(self):
        for renderer in self._camera_renderers.values():
            close_fn = getattr(renderer, "close", None)
            if callable(close_fn):
                close_fn()
        self._camera_renderers = {}

    def _update_robot_state(self):
        mj_qpos = self.mj_data.qpos.copy()
        self.placo_robot.state.q = calc_placo_q_from_mujoco_qpos(
            self.mj_model,
            self.placo_robot,
            mj_qpos,
            floating_base=self.floating_base,
        )
        self.placo_robot.update_kinematics()

    def _update_mocap_target(self):
        for name, task in self.effector_task.items():
            if self.effector_control_mode.get(name, "pose") == "position":
                T_world_target = tf.identity_matrix()
                T_world_target[:3, 3] = task.target_world
            else:
                link_target = task.T_world_frame
                config = self.manipulator_config[name]
                link_xyz = link_target[:3, 3].copy()
                link_quat = tf.quaternion_from_matrix(link_target)
                control_xyz, control_quat = self._link_pose_to_control_pose(config, link_xyz, link_quat)
                T_world_target = tf.quaternion_matrix(control_quat)
                T_world_target[:3, 3] = control_xyz
            mocap_idx = self.target_mocap_idx.get(name)
            if mocap_idx is not None and mocap_idx != -1:
                self.mj_data.mocap_pos[mocap_idx] = T_world_target[:3, 3] + self.ik_world_offset
                self.mj_data.mocap_quat[mocap_idx] = tf.quaternion_from_matrix(T_world_target)

    def _get_link_pose(self, ee_name):
        """Get the end effector position and orientation."""
        ee_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        if ee_id == -1:
            raise ValueError(f"End effector body '{ee_name}' not found in the model.")

        ee_xyz = self.mj_data.xpos[ee_id].copy() - self.ik_world_offset
        ee_quat = self.mj_data.xquat[ee_id].copy()

        return ee_xyz, ee_quat

    def run(self):
        with mj_viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            # Set up viewer camera
            viewer.cam.azimuth = 0
            viewer.cam.elevation = -50
            viewer.cam.distance = 2.0
            viewer.cam.lookat = [0.2, 0, 0]
            self._start_time = time.time()

            while not self._stop_event.is_set():
                try:
                    self._check_logging_button()
                    self._enforce_static_constraints()
                    self._update_robot_state()
                    self._update_ik()
                    self._update_gripper_target()
                    self._update_mocap_target()
                    self._send_command()
                    self._log_data()

                    # Advance simulation with optional substeps per control cycle.
                    for _ in range(self.sim_steps_per_control):
                        mujoco.mj_step(self.mj_model, self.mj_data)
                        self._enforce_static_constraints()
                    viewer.sync()
                except KeyboardInterrupt:
                    print("\nTeleoperation stopped.")
                    self._stop_event.set()

            if self.enable_log_data and self._is_logging:
                print("--- Saving remaining simulation data before exit... ---")
                self.data_logger.save()
                self.data_logger.reset()

        self._close_camera_renderers()
