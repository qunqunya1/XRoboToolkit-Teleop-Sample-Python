#!/usr/bin/env python3
"""Generate MuJoCo XMLs for X2 omnihands upper-body teleop scene."""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import xml.etree.ElementTree as ET

import mujoco


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_revolute_limits(urdf_path: Path) -> dict[str, tuple[float, float, float]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    limits: dict[str, tuple[float, float, float]] = {}
    for joint in root.iter():
        if not joint.tag.endswith("joint") or joint.get("type") != "revolute":
            continue
        name = joint.get("name")
        lim = None
        for c in joint:
            if c.tag.endswith("limit"):
                lim = c
                break
        if name is None or lim is None:
            continue
        lower = float(lim.get("lower", "0"))
        upper = float(lim.get("upper", "0"))
        effort = float(lim.get("effort", "2.0"))
        limits[name] = (lower, upper, effort)
    return limits


def _actuator_table(joint_limits: dict[str, tuple[float, float, float]]) -> list[dict[str, float | str]]:
    table: list[dict[str, float | str]] = []
    # Keep upper-body arm/head control behavior close to existing x2_upper_body_position config.
    fixed = [
        ("head_yaw_joint", 80, 8, -2.6, 2.6),
        ("head_pitch_joint", 80, 8, -0.6, 0.6),
        ("left_shoulder_pitch_joint", 800, 60, -36, 36),
        ("left_shoulder_roll_joint", 800, 60, -36, 36),
        ("left_shoulder_yaw_joint", 650, 45, -24, 24),
        ("left_elbow_joint", 650, 45, -24, 24),
        ("left_wrist_yaw_joint", 300, 20, -24, 24),
        ("left_wrist_pitch_joint", 250, 18, -4.8, 4.8),
        ("left_wrist_roll_joint", 250, 18, -4.8, 4.8),
        ("right_shoulder_pitch_joint", 800, 60, -36, 36),
        ("right_shoulder_roll_joint", 800, 60, -36, 36),
        ("right_shoulder_yaw_joint", 650, 45, -24, 24),
        ("right_elbow_joint", 650, 45, -24, 24),
        ("right_wrist_yaw_joint", 300, 20, -24, 24),
        ("right_wrist_pitch_joint", 250, 18, -4.8, 4.8),
        ("right_wrist_roll_joint", 250, 18, -4.8, 4.8),
    ]
    for joint, kp, kv, fmin, fmax in fixed:
        if joint not in joint_limits:
            continue
        lower, upper, _ = joint_limits[joint]
        table.append(
            {
                "name": f"pos_{joint}",
                "joint": joint,
                "kp": kp,
                "kv": kv,
                "fmin": fmin,
                "fmax": fmax,
                "lower": lower,
                "upper": upper,
            }
        )

    # 10 active omnihand driver joints (as agreed).
    hand_joints = [
        "L_thumb_mcp_joint",
        "L_index_abad_joint",
        "L_middle_abad_joint",
        "L_ring_abad_joint",
        "L_pinky_abad_joint",
        "R_thumb_mcp_joint",
        "R_index_abad_joint",
        "R_middle_abad_joint",
        "R_ring_abad_joint",
        "R_pinky_abad_joint",
    ]
    for joint in hand_joints:
        if joint not in joint_limits:
            continue
        lower, upper, effort = joint_limits[joint]
        f = max(0.8, float(effort))
        table.append(
            {
                "name": f"pos_{joint}",
                "joint": joint,
                "kp": 120.0,
                "kv": 10.0,
                "fmin": -f,
                "fmax": f,
                "lower": lower,
                "upper": upper,
            }
        )
    return table


def _write_scene_xml(scene_xml: Path, model_xml: Path, joint_limits: dict[str, tuple[float, float, float]]):
    model_rel = model_xml.name
    lines: list[str] = []
    lines.append('<mujoco model="x2 scene upper-body omnihands position">')
    lines.append(f'  <include file="{model_rel}"/>')
    lines.append("")
    lines.append('  <statistic center="0 0 0.6" extent="1.3"/>')
    lines.append("")
    lines.append("  <visual>")
    lines.append('    <headlight diffuse="0.5 0.5 0.5" ambient="0.2 0.2 0.2" specular="0.9 0.9 0.9"/>')
    lines.append('    <rgba haze="0.15 0.25 0.35 1"/>')
    lines.append('    <global azimuth="200" elevation="-20"/>')
    lines.append("  </visual>")
    lines.append("")
    lines.append("  <asset>")
    lines.append('    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>')
    lines.append('    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.3 0.52 0.63" rgb2="0.3 0.52 0.63" markrgb="1 1 1" width="300" height="300"/>')
    lines.append('    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>')
    lines.append("  </asset>")
    lines.append("")
    lines.append("  <worldbody>")
    lines.append('    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>')
    lines.append('    <camera name="rgbd_head_front_camera" mode="targetbodycom" target="head_pitch_link" pos="0.5 0 1.2"/>')
    lines.append('    <body name="right_target" pos="0.35 -0.2 1.0" quat="1 0 0 0" mocap="true">')
    lines.append('      <geom name="right_target_x_shaft" type="cylinder" size="0.005 0.05" pos="0.05 0 0" zaxis="1 0 0" rgba="1 0 0 1" contype="0" conaffinity="0" group="1"/>')
    lines.append('      <geom name="right_target_y_shaft" type="cylinder" size="0.005 0.05" pos="0 0.05 0" zaxis="0 1 0" rgba="0 1 0 1" contype="0" conaffinity="0" group="1"/>')
    lines.append('      <geom name="right_target_z_shaft" type="cylinder" size="0.005 0.05" pos="0 0 0.05" zaxis="0 0 1" rgba="0 0 1 1" contype="0" conaffinity="0" group="1"/>')
    lines.append('      <site type="sphere" size="0.01" rgba="0 0 1 1" group="1"/>')
    lines.append("    </body>")
    lines.append('    <body name="left_target" pos="0.35 0.2 1.0" quat="1 0 0 0" mocap="true">')
    lines.append('      <geom name="left_target_x_shaft" type="cylinder" size="0.005 0.05" pos="0.05 0 0" zaxis="1 0 0" rgba="1 0 0 1" contype="0" conaffinity="0" group="1"/>')
    lines.append('      <geom name="left_target_y_shaft" type="cylinder" size="0.005 0.05" pos="0 0.05 0" zaxis="0 1 0" rgba="0 1 0 1" contype="0" conaffinity="0" group="1"/>')
    lines.append('      <geom name="left_target_z_shaft" type="cylinder" size="0.005 0.05" pos="0 0 0.05" zaxis="0 0 1" rgba="0 0 1 1" contype="0" conaffinity="0" group="1"/>')
    lines.append('      <site type="sphere" size="0.01" rgba="0 0 1 1" group="1"/>')
    lines.append("    </body>")
    lines.append("  </worldbody>")
    lines.append("")
    lines.append("  <actuator>")
    for cfg in _actuator_table(joint_limits):
        lines.append(
            f'    <position name="{cfg["name"]}" joint="{cfg["joint"]}" '
            f'ctrlrange="{cfg["lower"]} {cfg["upper"]}" '
            f'kp="{cfg["kp"]}" kv="{cfg["kv"]}" '
            f'forcerange="{cfg["fmin"]} {cfg["fmax"]}"/>'
        )
    lines.append("  </actuator>")
    lines.append("</mujoco>")
    scene_xml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(urdf_path: Path, model_xml_path: Path, scene_xml_path: Path):
    model_xml_path.parent.mkdir(parents=True, exist_ok=True)
    scene_xml_path.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    mujoco.mj_saveLastXML(str(model_xml_path), model)
    # Fix mesh directory so the generated XML can resolve omnihand meshes.
    model_tree = ET.parse(model_xml_path)
    model_root = model_tree.getroot()
    compiler = model_root.find("compiler")
    if compiler is not None:
        mesh_dir_abs = (urdf_path.parent / "meshes").resolve()
        relpath = os.path.relpath(mesh_dir_abs, model_xml_path.parent.resolve()).replace("\\", "/")
        compiler.set("meshdir", relpath if relpath.endswith("/") else f"{relpath}/")
        model_tree.write(model_xml_path, encoding="utf-8", xml_declaration=True)

    limits = _parse_revolute_limits(urdf_path)
    _write_scene_xml(scene_xml_path, model_xml_path, limits)

    print(f"Generated model XML: {model_xml_path}")
    print(f"Generated scene XML: {scene_xml_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate X2 omnihands MuJoCo model+scene XML.")
    parser.add_argument(
        "--urdf-path",
        default=str(_repo_root() / "X2_with_omnihands_URDF" / "x2_ultra_with_omnihands.urdf"),
    )
    parser.add_argument(
        "--model-xml-path",
        default=str(_repo_root() / "X2_URDF" / "x2_ultra_with_omnihands_generated.xml"),
    )
    parser.add_argument(
        "--scene-xml-path",
        default=str(_repo_root() / "X2_URDF" / "scene_upper_body_omnihands_position.xml"),
    )
    args = parser.parse_args()

    generate(Path(args.urdf_path), Path(args.model_xml_path), Path(args.scene_xml_path))


if __name__ == "__main__":
    main()
