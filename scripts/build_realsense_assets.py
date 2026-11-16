#!/usr/bin/env python3
"""Convert the pinned D405/D455 description subset into standalone MuJoCo assets.

This handles the numeric properties and fixed joints in these two pinned
macros; it is not a general Xacro processor. Nominal extrinsics are enabled,
mesh visuals are selected, and the optional USB plug is omitted.
"""
from __future__ import annotations

import ast
import json
import operator
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/realsense"
SOURCE = ASSETS / "source/realsense2_description"
XACRO = "{http://ros.org/wiki/xacro}"


def expression(value, properties):
    """Evaluate only arithmetic and already resolved numeric properties."""
    binary = {ast.Add: operator.add, ast.Sub: operator.sub,
              ast.Mult: operator.mul, ast.Div: operator.truediv}
    unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            return properties[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            return unary[type(node.op)](visit(node.operand))
        raise ValueError(f"Unsupported numeric expression: {value}")
    return float(visit(ast.parse(value, mode="eval").body))


def expand(value, properties, name):
    value = value.replace("${name}", name).replace("${parent}", "world")
    return re.sub(r"\$\{([^}]+)\}", lambda m: f"{expression(m[1], properties):.12g}", value)


def numbers(values):
    return " ".join(f"{float(v):.12g}" for v in np.ravel(values))


def pose(origin, properties, name):
    xyz = np.fromstring(expand(origin.get("xyz", "0 0 0"), properties, name), sep=" ") if origin is not None else np.zeros(3)
    rpy = np.fromstring(expand(origin.get("rpy", "0 0 0"), properties, name), sep=" ") if origin is not None else np.zeros(3)
    rotation = Rotation.from_euler("xyz", rpy)
    quat = rotation.as_quat()[[3, 0, 1, 2]]
    return xyz, rotation.as_matrix(), {"pos": numbers(xyz), "quat": numbers(quat)}


def save(root, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    with path.open("a") as stream:
        stream.write("\n")


def build(name):
    macro = ET.parse(SOURCE / f"urdf/_{name}.urdf.xacro").getroot().find(XACRO + "macro")
    properties = {}
    for item in macro.iter(XACRO + "property"):
        properties[item.get("name")] = expression(expand(item.get("value"), properties, name), properties)
    links = {expand(link.get("name"), properties, name): link for link in macro.iter("link")}
    folder = ASSETS / name
    root = ET.Element("mujoco", model=f"realsense_{name}")
    root.append(ET.Comment("Generated from the pinned RealSense URDF: nominal extrinsics, no plug. Inertia is a uniform-box approximation."))
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", fusestatic="false")
    ET.SubElement(root, "include", file="assets.xml")
    world = ET.SubElement(root, "worldbody")
    body_fragment = ET.Element("mujoco")
    bodies = {"world": body_fragment}
    resolved = {}
    pending = list(macro.iter("joint"))
    while pending:
        for joint in pending[:]:
            if joint.get("type") != "fixed":
                raise ValueError("Camera descriptions must contain only fixed joints.")
            parent = expand(joint.find("parent").get("link"), properties, name)
            child = expand(joint.find("child").get("link"), properties, name)
            if parent not in bodies:
                continue
            xyz, rotation, attrs = pose(joint.find("origin"), properties, name)
            body = ET.SubElement(bodies[parent], "body", name=child, **attrs)
            ET.SubElement(body, "site", name=child + "_site", size=".001", group="4")
            bodies[child] = body
            resolved[child] = dict(parent=parent, translation_m=xyz.tolist(), rotation_matrix=rotation.tolist())
            pending.remove(joint)
            break
        else:
            raise ValueError("Disconnected camera frame tree.")
    link = links[name + "_link"]
    body = bodies[name + "_link"]
    visual = link.find("visual")
    visual_xyz, visual_rotation, visual_attrs = pose(visual.find(".//origin"), properties, name)
    mesh = visual.find(".//mesh")
    scale = np.fromstring(mesh.get("scale"), sep=" ")
    ET.SubElement(body, "geom", name=name + "_visual", type="mesh", mesh=name + "_mesh",
                  rgba=".55 .57 .59 1", group="2", contype="0", conaffinity="0", mass="0", **visual_attrs)
    collision = link.find("collision")
    collision_xyz, _, collision_attrs = pose(collision.find("origin"), properties, name)
    dimensions = np.fromstring(expand(collision.find("geometry/box").get("size"), properties, name), sep=" ")
    ET.SubElement(body, "geom", name=name + "_collision", type="box", size=numbers(dimensions/2),
                  group="3", mass="0", **collision_attrs)
    mass = float(link.find("inertial/mass").get("value"))
    inertia = mass/12 * np.array([dimensions[1]**2+dimensions[2]**2,
                                 dimensions[0]**2+dimensions[2]**2,
                                 dimensions[0]**2+dimensions[1]**2])
    ET.SubElement(body, "inertial", pos=numbers(collision_xyz), mass=str(mass), diaginertia=numbers(inertia))
    save(body_fragment, folder / "body.xml")
    ET.SubElement(world, "include", file="body.xml")
    save(root, folder / f"{name}.xml")
    asset_fragment = ET.Element("mujoco")
    assets = ET.SubElement(asset_fragment, "asset")
    mesh_file = "../source/realsense2_description/meshes/" + Path(mesh.get("filename")).name
    ET.SubElement(assets, "mesh", name=name + "_mesh", file=mesh_file, scale=numbers(scale))
    save(asset_fragment, folder / "assets.xml")
    scene = ET.Element("mujoco", model=f"realsense_{name}_preview")
    ET.SubElement(scene, "include", file=f"{name}.xml")
    ET.SubElement(scene, "statistic", center=numbers([0, 0, dimensions[2]/2]), extent=str(max(dimensions)*2.5))
    display = ET.SubElement(scene, "visual")
    ET.SubElement(display, "headlight", ambient=".4 .4 .4", diffuse=".7 .7 .7")
    ET.SubElement(display, "global", offwidth="1200", offheight="900")
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(world, "light", pos=".2 -.2 .5", dir="-.2 .2 -.5")
    ET.SubElement(world, "geom", name="preview_floor", type="plane", size=".3 .3 .001",
                  pos="0 0 -.0002", rgba=".16 .18 .2 1", contype="0", conaffinity="0")
    save(scene, folder / "scene.xml")
    raw_mesh = trimesh.load_mesh(folder / mesh_file)
    report = dict(model=name.upper(), root_body=name + "_bottom_screw_frame", frames=resolved,
                  source_mesh_units="millimeters", mesh_scale_to_m=scale.tolist(),
                  source_mesh_bounds_mm=raw_mesh.bounds.tolist(),
                  visual_origin_link_m=visual_xyz.tolist(), visual_rotation_link=visual_rotation.tolist(),
                  collision_box_dimensions_xyz_m=dimensions.tolist(),
                  mass_kg=mass, inertia_diagonal_kg_m2=inertia.tolist(),
                  dynamics_status="Upstream placeholder mass; uniform-box inertia computed here. Not measured camera dynamics.",
                  extrinsics_status="Nominal upstream fixed frames, not per-device calibrated extrinsics.",
                  camera_model_status="Geometry and frame sites only; no optical intrinsics, depth noise or hardware stream simulation.")
    (folder / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Built {name}: {len(resolved)} fixed frames, {len(raw_mesh.faces)} visual triangles")


if __name__ == "__main__":
    for camera in ("d405", "d455"):
        build(camera)
