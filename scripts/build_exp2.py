#!/usr/bin/env python3
"""Add the short custom fingers and dual GelSight Minis to the Exp 1 bench."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim_env.bench import load_scene


def numbers(values):
    return " ".join(f"{float(v):.12g}" for v in np.ravel(values))


def quaternion(rotation):
    q = Rotation.from_matrix(rotation).as_quat()
    return numbers(q[[3, 0, 1, 2]])


def save_xml(root, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    with path.open("a") as stream:
        stream.write("\n")


def sensor_mount(config, sign, side, half_body_depth):
    """Sensor frame: +x outward, +z toward the WSG; gap refers to gel apexes."""
    mount, grip, sensor = config["mount"], config["gripper"], config["sensor"]
    rotation = (Rotation.from_euler("z", 0 if sign == 1 else 180, degrees=True).as_matrix()
                @ Rotation.from_euler("y", mount[f"{side}_sensor_tilt_deg"], degrees=True).as_matrix())
    mount_point = np.array(mount["finger_mount_point_sensor_m"])
    apex_x = sensor["gel_geometry"]["apex_sensor_m"][0]
    center = np.array([sign * (grip["surface_gap_at_reference_m"]/2 - apex_x),
                       0, -(half_body_depth + mount["carriage_extension_m"] + mount_point[2])])
    jaw = center + rotation @ mount_point - [sign*grip["reference_opening_m"]/2, 0, 0]
    return rotation, jaw, -rotation @ mount_point


def prepare_meshes(config):
    directory = ROOT / "assets/wsg50_gelsight"
    mesh = trimesh.load_mesh(directory / "source/WSG50_GELSIGHT_fin.stl")
    np.testing.assert_allclose(mesh.bounds, [[0, -46, 0], [29, 15, 30]], atol=.001)
    registration = config["finger_source_to_sensor"]
    matrix = np.eye(4)
    matrix[:3, :3] = registration["scale_to_m"] * np.array(registration["rotation_matrix"])
    matrix[:3, 3] = registration["translation_m"]
    mesh.apply_transform(matrix)
    (directory / "meshes").mkdir(exist_ok=True)
    mesh.export(directory / "meshes/custom_finger.obj")
    gel_config = config["sensor"]["gel_geometry"]
    gel = trimesh.load_mesh(ROOT / gel_config["source_stl"])
    rotation = np.array(gel_config["source_to_sensor_rotation"])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    if not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError("Gel registration must be a proper rotation.")
    gel.vertices = (gel.vertices @ rotation.T + gel_config["source_to_sensor_translation_mm"]) * gel_config["scale_to_m"]
    np.testing.assert_allclose(gel.bounds, gel_config["bounds_sensor_m"], atol=2e-9)
    if not gel.is_watertight or not gel.is_winding_consistent or gel.volume <= 0:
        raise ValueError("Curved gel must be a closed, consistently oriented solid.")
    gel.export(directory / "meshes/gel_curved_sensor_m.obj")
    hull = ConvexHull(gel.vertices)
    gel_report = dict(source=gel_config["source_stl"], bounds_sensor_m=gel.bounds.tolist(),
                      vertices=len(gel.vertices), triangles=len(gel.faces), watertight=bool(gel.is_watertight),
                      volume_m3=float(gel.volume), convex_hull_volume_m3=float(hull.volume),
                      convex_hull_relative_volume_excess=float(hull.volume/gel.volume-1))
    return matrix, mesh.bounds.tolist(), gel_report


def build(config_path=None):
    config = json.loads(Path(config_path or ROOT / "configs/exp2.json").read_text())
    base = json.loads((ROOT / config["base_config"]).read_text())
    base_assembly = json.loads((ROOT / "models/assembly.json").read_text())
    if base != base_assembly["config"]:
        raise ValueError("Exp 1 configuration changed; run scripts/build_scene.py first.")
    grip, sensor = config["gripper"], config["sensor"]
    maximum = base["gripper"]["maximum_opening_m"]
    if not 0 <= grip["opening_m"] <= maximum:
        raise ValueError("Initial WSG opening exceeds travel limits.")
    registration, finger_bounds, gel_report = prepare_meshes(config)
    components = ROOT / "models/exp2_components"

    root = ET.parse(ROOT / "models/bench.xml").getroot()
    root.set("model", "exp2_ur5e_wsg50_dual_gelsight")
    for include in root.findall("include"):
        if include.get("file") == "components/robot.xml":
            include.set("file", "exp2_components/robot.xml")
    ET.SubElement(root, "include", file="exp2_components/assets.xml")
    robot = ET.parse(ROOT / "models/components/robot.xml").getroot()
    tool = robot.find(".//body[@name='urdf_tool0']")
    for include in tool.findall("include"):
        include.set("file", "wsg50_gelsight.xml" if include.get("file") == "wsg50.xml"
                    else "../components/" + include.get("file"))
    save_xml(robot, components / "robot.xml")

    assets = ET.Element("mujoco")
    asset = ET.SubElement(assets, "asset")
    for name, relative in [("gs_custom_finger", "meshes/custom_finger.obj"),
                           ("gs_case", "source/case.obj"), ("gs_back_cover", "source/back_cover.obj"),
                           ("gs_curved_gel", "meshes/gel_curved_sensor_m.obj")]:
        ET.SubElement(asset, "mesh", name=name, file="../assets/wsg50_gelsight/" + relative)
    ET.SubElement(asset, "material", name="gs_gel_material", rgba=".12 .68 .65 1",
                  specular=".6", shininess=".5")
    save_xml(assets, components / "assets.xml")

    gripper = ET.Element("mujoco")
    # Preserve Exp 1's body, flange adapter and calibration exactly. Only its
    # two simplified box fingers and TCP are replaced by the sensor assembly.
    old_gripper = ET.parse(ROOT / "models/components/wsg50.xml").getroot()
    gripper.append(copy.deepcopy(old_gripper.find("body[@name='wsg50']")))
    half_depth = base["gripper"]["body_size_sdf_m"][1] / 2
    center_z = base["flange_adapter"]["thickness_m"] + half_depth
    mount_frame = ET.SubElement(gripper, "body", name="gelsight_wsg_frame",
                               pos=numbers([0, 0, center_z]), quat="0 1 0 0")
    mount_frames, surface_centers = {}, []
    for side, sign in [("left", -1), ("right", 1)]:
        rotation, jaw_position, sensor_position = sensor_mount(config, sign, side, half_depth)
        jaw = ET.SubElement(mount_frame, "body", name=f"wsg_{side}_finger", pos=numbers(jaw_position))
        # The same named actuators and travel limits are shared with Exp 1.
        original = old_gripper.find(f"body[@name='wsg_{side}_finger']/joint")
        jaw.append(copy.deepcopy(original))
        ET.SubElement(jaw, "geom", name=f"wsg_{side}_carriage", type="box", size=".01 .015 .007",
                      pos="0 0 .007", rgba=".18 .2 .23 1", mass=str(sensor["carriage_mass_kg"]))
        body = ET.SubElement(jaw, "body", name=f"gelsight_{side}", pos=numbers(sensor_position),
                             quat=quaternion(rotation))
        ET.SubElement(body, "inertial", pos=numbers(sensor["assembly_com_sensor_m"]),
                      mass=str(sensor["sensor_mass_kg"] + sensor["finger_mass_kg"]),
                      diaginertia=numbers(sensor["assembly_diaginertia_kg_m2"]))
        for part, mesh, color in [("finger", "gs_custom_finger", ".12 .12 .14 1"),
                                  ("case", "gs_case", ".08 .09 .1 1"),
                                  ("back_cover", "gs_back_cover", ".14 .15 .16 1")]:
            ET.SubElement(body, "geom", name=f"gelsight_{side}_{part}", type="mesh", mesh=mesh,
                          rgba=color, contype="0", conaffinity="0", group="2", mass="0")
        # MuJoCo uses the convex hull for mesh collisions. The short bracket's
        # pockets/bolt holes are visual details, not resolved collision cavities.
        ET.SubElement(body, "geom", name=f"gelsight_{side}_finger_collision", type="mesh",
                      mesh="gs_custom_finger", group="3", mass="0")
        ET.SubElement(body, "geom", name=f"gelsight_{side}_housing_collision", type="box",
                      pos=".014 0 0", size=".011875 .014 .016002", group="3", mass="0")
        ET.SubElement(body, "geom", name=f"gelsight_{side}_gel", type="mesh",
                      mesh="gs_curved_gel", material="gs_gel_material", mass="0")
        surface = sensor["gel_geometry"]["apex_sensor_m"]
        for name in [f"gelsight_{side}_surface", f"wsg_{side}_inner"]:
            ET.SubElement(body, "site", name=name, pos=numbers(surface), size=".001", group="4")
        surface_centers.append(jaw_position + [sign*grip["opening_m"]/2, 0, 0]
                               + sensor_position + rotation @ surface)
        mount_frames[side] = dict(jaw_position_wsg_m=jaw_position.tolist(),
                                 sensor_position_jaw_m=sensor_position.tolist(),
                                 sensor_rotation_jaw=rotation.tolist())
    tcp = np.diag([1, -1, -1]) @ np.mean(surface_centers, axis=0) + [0, 0, center_z]
    ET.SubElement(gripper, "site", name="tcp", pos=numbers(tcp), size=".004", group="4")
    save_xml(gripper, components / "wsg50_gelsight.xml")
    key = root.find("keyframe/key[@name='ready']")
    ready = np.r_[np.deg2rad(base["robot"]["initial_q_deg"]), np.repeat(grip["opening_m"]/2, 2)]
    key.set("qpos", numbers(ready))
    key.set("ctrl", numbers(ready))
    output = ROOT / "models/exp2.xml"
    save_xml(root, output)
    model, data = load_scene(output)
    penetrations = [(model.geom(c.geom1).name, model.geom(c.geom2).name, float(c.dist))
                    for c in data.contact if c.dist < -1e-5]
    if penetrations:
        raise RuntimeError(f"Initial assembly has penetrating contacts: {penetrations[:10]}")
    manifest = dict(config=config, base_config=base, finger_source_to_sensor=registration.tolist(),
                    finger_bounds_sensor_m=finger_bounds, curved_gel=gel_report,
                    mounts=mount_frames, tcp_tool0_m=tcp.tolist(),
                    table_from_ur_base=base_assembly["table_from_ur_base"],
                    ur_base_from_urdf_model=base_assembly["ur_base_from_urdf_model"],
                    nq=model.nq, nu=model.nu,
                    calibration_status="Exp 1 robot/table calibration; nominal GelSight mounting and opening zero",
                    contact_model="Curved CAD gel meshes with convex-hull collision; housing boxes; convex hulls of custom fingers; no optical or continuum gel model")
    (ROOT / "models/exp2_assembly.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {output}: {model.nq} DOF, two short custom fingers and two GelSight Minis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    build(parser.parse_args().config)
