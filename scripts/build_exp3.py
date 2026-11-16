#!/usr/bin/env python3
"""Assemble the dual-UR10e/HX5-D20 scene from the local asset library."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim_env.bench import JOINT_NAMES


def numbers(values):
    return " ".join(f"{float(value):.12g}" for value in np.ravel(values))


def quaternion(rpy_deg):
    q = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_quat()
    return numbers(q[[3, 0, 1, 2]])


def save_xml(root, path):
    ET.indent(root, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="unicode")
    with path.open("a") as stream:
        stream.write("\n")


def prefix_identifiers(element, prefix):
    """Prefix object definitions/references while retaining file paths."""
    attributes = {"name", "class", "childclass", "mesh", "material", "joint",
                  "body1", "body2", "joint1", "joint2", "site", "target"}
    for node in element.iter():
        for attr in attributes:
            if attr in node.attrib:
                node.set(attr, prefix + node.get(attr))


def include_fragment(root, name, elements):
    fragment = ET.Element("mujoco")
    for element in elements:
        fragment.append(element)
    save_xml(fragment, ROOT / "models/exp3_components" / f"{name}.xml")
    ET.SubElement(root, "include", file=f"exp3_components/{name}.xml")


def add_tables(world, assets, config):
    # Reuse the exact visual/collision tiles and complete perforated table
    # generated for Setup 1. No changes are made to those shared assets.
    requested = json.loads((ROOT / config["table_template"]).read_text())["table"]
    generated = json.loads((ROOT / "models/assembly.json").read_text())["config"]["table"]
    if requested != generated:
        raise ValueError("The table template changed; run scripts/build_scene.py before building Setup 3.")
    source_assets = ET.parse(ROOT / "models/components/assets.xml").getroot()
    for mesh in source_assets.findall("asset/mesh"):
        if mesh.get("name", "").startswith("table_"):
            assets.append(copy.deepcopy(mesh))
    template = ET.parse(ROOT / "models/components/table.xml").find("worldbody/body")
    for side, color in [("left", "0.32 0.35 0.38 1"), ("right", "0.30 0.33 0.36 1")]:
        body = copy.deepcopy(template)
        # Mesh assets are shared; only names need a per-table namespace.
        for node in body.iter():
            if "name" in node.attrib:
                node.set("name", side + "_" + node.get("name"))
            if node.tag == "geom" and node.get("group") == "2":
                node.set("rgba", color)
        body.set("pos", numbers(config["tables"][f"{side}_origin_m"]))
        world.append(body)
    # A visual seam identifies the touching plates without adding a collision
    # wall or changing the through-hole geometry.
    right_origin = np.asarray(config["tables"]["right_origin_m"])
    table_min, table_max = np.asarray(requested["min_xy_m"]), np.asarray(requested["max_xy_m"])
    seam_position = right_origin + [table_min[0], (table_min[1]+table_max[1])/2, requested["top_z_m"]+.00015]
    ET.SubElement(world, "geom", name="table_join_marker", type="box",
                  pos=numbers(seam_position), size=numbers([.0006,(table_max[1]-table_min[1])/2,.00015]),
                  rgba="0.12 0.14 0.16 1", contype="0", conaffinity="0", group="2")


def add_arm(world, assets, defaults, actuators, contacts, config, side):
    source_path = ROOT / "assets/ur10e/source/ur10e.xml"
    source = ET.parse(source_path).getroot()
    # One shared, namespaced asset/default set supports both arm instances.
    if side == "left":
        default = copy.deepcopy(source.find("default"))
        prefix_identifiers(default, "ur10e_")
        defaults.extend(default)
        for item in source.find("asset"):
            item = copy.deepcopy(item)
            if item.tag == "mesh":
                item.set("name", item.get("name", Path(item.get("file")).stem))
                item.set("file", "../assets/ur10e/source/assets/" + item.get("file"))
            prefix_identifiers(item, "ur10e_")
            assets.append(item)
    body = copy.deepcopy(source.find("worldbody/body"))
    for node in body.iter():
        if "name" in node.attrib:
            node.set("name", side + "_" + node.get("name"))
        for attr in ("class", "childclass", "mesh", "material"):
            if attr in node.attrib:
                node.set(attr, "ur10e_" + node.get(attr))
    placement = config["robots"][side]
    if placement["hand"] != side:
        raise ValueError(f"Setup 3 requires the matching {side} HX5-D20 hand.")
    wrapper = ET.SubElement(world, "body", name=f"{side}_mount",
                            pos=numbers(placement["base_position_m"]),
                            quat=quaternion([0, 0, placement["mount_yaw_deg"]]))
    wrapper.append(body)
    ET.SubElement(wrapper, "site", name=f"{side}_base_frame", size="0.004", group="4")
    height = placement["base_position_m"][2]
    ET.SubElement(wrapper, "geom", name=f"{side}_base_spacer", type="cylinder",
                  pos=numbers([0, 0, -height/2]),
                  size=numbers([config["mount"]["spacer_radius_m"], height/2]),
                  rgba="0.19 0.20 0.22 1", contype="0", conaffinity="0", group="2")
    for actuator in source.find("actuator"):
        item = copy.deepcopy(actuator)
        item.set("name", side + "_" + item.get("name"))
        item.set("joint", side + "_" + item.get("joint"))
        item.set("class", "ur10e_" + item.get("class"))
        actuators.append(item)
    wrist = body.find(f".//body[@name='{side}_wrist_3_link']")
    site = wrist.find(f"site[@name='{side}_attachment_site']")
    site.set("pos", numbers([0, config["mount"]["tool0_wrist3_y_m"], 0]))
    flange = ET.SubElement(wrist, "body", name=f"{side}_flange_frame",
                           pos=site.get("pos"), quat=site.get("quat"))
    ET.SubElement(flange, "site", name=f"{side}_flange", size="0.004", group="4")
    adapter = config["mount"]
    thickness = adapter["adapter_thickness_m"]
    ET.SubElement(flange, "geom", name=f"{side}_hand_adapter", type="cylinder",
                  size=numbers([adapter["adapter_radius_m"], thickness/2]),
                  pos=numbers([0, 0, thickness/2]), rgba="0.34 0.36 0.39 1",
                  density=str(adapter["adapter_density_kg_m3"]))
    hand_mount = ET.SubElement(flange, "body", name=f"{side}_hand_mount",
                               pos=numbers([0, 0, thickness]),
                               quat=quaternion(config["hand_mount_rpy_deg"]))
    ET.SubElement(hand_mount, "site", name=f"{side}_hand_base_frame", size="0.004", group="4")
    hand_path = ROOT / f"assets/robotis_hx5_d20/mujoco/hx5_d20_rev2/hx5_d20_{side}.xml"
    hand = ET.parse(hand_path).getroot()
    defaults.extend(copy.deepcopy(hand.find("default")))
    for item in hand.find("asset"):
        item = copy.deepcopy(item)
        item.set("file", "../assets/robotis_hx5_d20/meshes/hx5_d20_rev2/" + item.get("file"))
        assets.append(item)
    hand_mount.append(copy.deepcopy(hand.find("worldbody/body")))
    actuators.extend(copy.deepcopy(hand.find("actuator")))
    contacts.extend(copy.deepcopy(hand.find("contact")))
    # No URDF base calibration is transferred from the distinct UR5e bench.


def solve_ready_pose(scene_path, config):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(scene_path.resolve()))
    data = mujoco.MjData(model)
    solutions = {}
    for side in ("left", "right"):
        placement = config["robots"][side]
        ids = [model.joint(f"{side}_{name}").id for name in JOINT_NAMES]
        addresses = model.jnt_qposadr[ids]
        target_pos = np.array(placement["flange_target_position_m"])
        target_rot = Rotation.from_euler("xyz", placement["flange_target_rpy_deg"], degrees=True).as_matrix()
        seed = np.deg2rad(placement["ik_seed_deg"])
        site_id = model.site(f"{side}_flange").id

        def residual(q):
            data.qpos[addresses] = q
            mujoco.mj_kinematics(model, data)
            position_error = data.site_xpos[site_id] - target_pos
            rotation_error = Rotation.from_matrix(target_rot.T @ data.site_xmat[site_id].reshape(3,3)).as_rotvec()
            return np.r_[5 * position_error, rotation_error]

        fit = least_squares(residual, seed, bounds=model.jnt_range[ids].T,
                            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=500)
        error = residual(fit.x)
        if np.linalg.norm(error) > 1e-6:
            raise RuntimeError(f"{side} ready-pose IK did not converge: {error}")
        solutions[side] = fit.x.tolist()
    for aid in range(model.nu):
        joint_id = model.actuator_trnid[aid, 0]
        data.ctrl[aid] = data.qpos[model.jnt_qposadr[joint_id]]
    mujoco.mj_forward(model, data)
    collisions = [{"geom1": model.geom(c.geom1).name, "geom2": model.geom(c.geom2).name,
                   "distance_m": float(c.dist)} for c in data.contact if c.dist < -1e-5]
    if collisions:
        raise RuntimeError(f"Ready pose has penetrating contacts: {collisions[:12]}")
    return data.qpos.copy(), data.ctrl.copy(), solutions


def build(config_path=None):
    config = json.loads(Path(config_path or ROOT / "configs/exp3.json").read_text())
    root = ET.Element("mujoco", model="exp3_dual_ur10e_hx5_d20")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", fusestatic="false")
    ET.SubElement(root, "option", timestep=str(config["timestep_s"]), integrator="implicitfast",
                  cone="elliptic", impratio="10", gravity="0 0 -9.81")
    ET.SubElement(root, "statistic", center="1.2 -0.4 0.35", extent="2.8")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1800", offheight="1200")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.65 0.65 0.65", specular="0.15 0.15 0.15")
    ET.SubElement(visual, "quality", shadowsize="4096")
    defaults = ET.Element("default")
    ET.SubElement(defaults, "geom", friction="0.8 0.005 0.0001")
    assets, table_world, robot_world = ET.Element("asset"), ET.Element("worldbody"), ET.Element("worldbody")
    ET.SubElement(assets, "texture", type="skybox", builtin="gradient", rgb1="0.22 0.25 0.29",
                  rgb2="0.06 0.08 0.1", width="512", height="3072")
    actuators, contacts = ET.Element("actuator"), ET.Element("contact")
    ET.SubElement(table_world, "light", pos="1.2 -0.2 2.5", dir="0 0 -1", diffuse="0.7 0.7 0.7")
    add_tables(table_world, assets, config)
    for side in ("left", "right"):
        add_arm(robot_world, assets, defaults, actuators, contacts, config, side)
    for name, element in [("defaults", defaults), ("assets", assets), ("tables", table_world),
                           ("robots", robot_world), ("actuators", actuators), ("contacts", contacts)]:
        include_fragment(root, name, [element])
    output = ROOT / "models/exp3.xml"
    save_xml(root, output)
    qpos, ctrl, solved = solve_ready_pose(output, config)
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="ready", qpos=numbers(qpos), ctrl=numbers(ctrl))
    save_xml(root, output)
    manifest = {"config": config, "ready_arm_q_rad": solved,
                "ready_arm_q_deg": {side: np.rad2deg(q).tolist() for side,q in solved.items()},
                "nq": len(qpos), "nu": len(ctrl), "total_through_holes": 768,
                "installation_status": "Nominal layout; no measured Exp 3 base calibration applied"}
    (ROOT / "models/exp3_assembly.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {output}: {len(qpos)} DOF, 2 tables, 768 through-holes")
    print("Ready arm joints [deg]:", manifest["ready_arm_q_deg"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    build(parser.parse_args().config)
