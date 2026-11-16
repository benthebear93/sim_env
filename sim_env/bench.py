"""Configuration, frame conventions and a hardware-free scene loader."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


def load_config(path=None):
    return json.loads(Path(path or ROOT / "configs/bench.json").read_text())


def transform(xyz=(0, 0, 0), rotation=None):
    out = np.eye(4)
    out[:3, 3] = xyz
    if rotation is not None:
        out[:3, :3] = rotation
    return out


def origin(element):
    if element is None:
        return np.eye(4)
    xyz = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    return transform(xyz, Rotation.from_euler("xyz", rpy).as_matrix())


def urdf_fk(q):
    """Nominal URDF world -> tool0; independent of MuJoCo compilation."""
    source = ET.parse(ROOT / "assets/ur5e/source/ur5e.urdf").getroot()
    angles = dict(zip(JOINT_NAMES, q))
    frames = {"world": np.eye(4)}
    pending = list(source.findall("joint"))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in frames:
                continue
            local = origin(joint.find("origin"))
            if joint.get("type") == "revolute":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                local = local @ transform(rotation=Rotation.from_rotvec(
                    axis * angles[joint.get("name")]).as_matrix())
            frames[joint.find("child").get("link")] = frames[parent] @ local
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("URDF contains a disconnected/cyclic joint tree")
    return frames["tool0"]


def frame_transforms(config):
    calibration = json.loads((ROOT / config["calibration_file"]).read_text())["transform"]
    robot = config["robot"]
    position = np.array(calibration["translation_m"], dtype=float)
    position[:2] -= np.array(robot["world_origin_in_calibration_xy_mm"]) / 1000
    position[2] = config["table"]["top_z_m"] + robot["ur_base_above_table_m"]
    table_from_ur = transform(position, calibration["rotation_matrix"])
    rotation = Rotation.from_euler("z", robot["model_to_ur_base_yaw_deg"], degrees=True).as_matrix()
    offset = np.zeros(3)
    if robot["fit_model_translation"]:
        nominal = urdf_fk(np.deg2rad(robot["reference_q_deg"]))
        offset = np.array(robot["reference_tool0_ur_base_mm"]) / 1000 - rotation @ nominal[:3, 3]
    return table_from_ur, transform(offset, rotation)


def grid_hole_position(ix, iy, config=None):
    """One-based grid: x increases from min-x; y decreases from max-y."""
    config = config or load_config()
    table = config["table"]
    counts = np.rint((np.array(table["max_xy_m"]) - table["min_xy_m"]) /
                     table["hole_spacing_m"]).astype(int)
    if not (1 <= ix <= counts[0] and 1 <= iy <= counts[1]):
        raise ValueError(f"Grid indices must be in [1,{counts[0]}] x [1,{counts[1]}]")
    return np.array([
        table["min_xy_m"][0] + table["hole_edge_offset_m"] + (ix - 1) * table["hole_spacing_m"],
        table["max_xy_m"][1] - table["hole_edge_offset_m"] - (iy - 1) * table["hole_spacing_m"],
        table["top_z_m"],
    ])


def load_scene(path=None):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(path or ROOT / "models/bench.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    mujoco.mj_forward(model, data)
    return model, data


def set_gripper_opening(model, data, opening_m):
    """Set the symmetric jaw inner-gap target in meters."""
    maximum = 2 * model.jnt_range[model.joint("wsg_left_slide").id, 1]
    if not 0 <= opening_m <= maximum:
        raise ValueError(f"Opening must be between 0 and {maximum} m")
    for name in ("wsg_left", "wsg_right"):
        data.ctrl[model.actuator(name).id] = opening_m / 2
