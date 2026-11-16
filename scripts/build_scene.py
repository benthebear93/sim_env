#!/usr/bin/env python3
"""Regenerate portable MJCF and meshes from the vendored source assets."""
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
from sim_env.bench import JOINT_NAMES, frame_transforms, grid_hole_position, load_config, origin


def numbers(values):
    return " ".join(f"{float(value):.12g}" for value in np.ravel(values))


def pose_attributes(matrix):
    xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return {"pos": numbers(matrix[:3, 3]), "quat": numbers(xyzw[[3, 0, 1, 2]])}


def save_xml(element, path):
    ET.indent(element, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(element).write(path, encoding="unicode", xml_declaration=False)
    with path.open("a") as stream:
        stream.write("\n")


def write_obj(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write("# Generated in meters by scripts/build_scene.py\n")
        for vertex in vertices:
            stream.write("v " + numbers(vertex) + "\n")
        for face in faces:
            stream.write("f " + " ".join(str(int(i) + 1) for i in face) + "\n")


def mesh_asset(assets, name, path):
    ET.SubElement(assets, "mesh", name=name, file="../" + str(path.relative_to(ROOT)))


def convert_visuals(assets):
    """Preserve DAE scene transforms and per-part diffuse colors."""
    result = {}
    out_dir = ROOT / "assets/ur5e/meshes"
    for source in sorted((ROOT / "assets/ur5e/source/meshes/visual").glob("*.dae")):
        scene = trimesh.load(source, force="scene", process=False)
        parts = []
        for index, node in enumerate(sorted(scene.graph.nodes_geometry)):
            matrix, geom_name = scene.graph[node]
            mesh = scene.geometry[geom_name].copy()
            mesh.apply_transform(matrix)
            name = f"ur5e_{source.stem}_{index}"
            path = out_dir / f"{source.stem}_{index}.obj"
            write_obj(path, mesh.vertices, mesh.faces)
            mesh_asset(assets, name, path)
            material = getattr(mesh.visual, "material", None)
            diffuse = getattr(material, "baseColorFactor", None)
            if diffuse is None:
                diffuse = getattr(material, "diffuse", [170, 180, 190, 255])
            rgba = np.asarray(diffuse, dtype=float) / 255
            rgba[3] = 1
            parts.append((name, numbers(rgba)))
        result[source.name] = parts
    for source in sorted((ROOT / "assets/wsg50/source").glob("*.gltf")):
        loaded = trimesh.load(source, force="scene", process=False).dump(concatenate=True)
        loaded.vertices = loaded.vertices @ Rotation.from_euler("x", 90, degrees=True).as_matrix().T
        path = ROOT / "assets/wsg50/meshes" / (source.stem + ".obj")
        write_obj(path, loaded.vertices, loaded.faces)
        if source.stem == "wsg_body":
            mesh_asset(assets, "wsg_body_mesh", path)
    return result


def add_table(world, assets, config):
    table = config["table"]
    lo, hi = np.array(table["min_xy_m"]), np.array(table["max_xy_m"])
    pitch, edge, radius = (table[k] for k in ("hole_spacing_m", "hole_edge_offset_m", "hole_radius_m"))
    count = (hi - lo) / pitch
    if not np.allclose(count, np.rint(count)) or not np.isclose(edge, pitch / 2):
        raise ValueError("Tiled table requires integer pitch counts and edge_offset = spacing / 2")
    count = np.rint(count).astype(int)
    n, depth = int(table["hole_segments"]), table["thickness_m"]
    if n < 8 or n % 8 or not 0 < radius < pitch / 2 or depth <= 0:
        raise ValueError("Use positive thickness, 0 < radius < spacing/2 and segments divisible by 8")
    angles = np.arange(n) * 2 * np.pi / n
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    inner = radius * unit
    outer = unit * (pitch / 2 / np.max(np.abs(unit), axis=1))[:, None]
    vertices, faces = [], []
    for i in range(n):
        j = (i + 1) % n
        polygon = np.array([inner[i], outer[i], outer[j], inner[j]])
        wedge = np.vstack([np.column_stack([polygon, np.zeros(4)]),
                           np.column_stack([polygon, np.full(4, -depth)])])
        hull = ConvexHull(wedge)
        path = ROOT / "assets/table/meshes" / f"wedge_{i:02d}.obj"
        # Collision vertices are convex; orient every triangle outward.
        hull_faces = hull.simplices.copy()
        for face in hull_faces:
            a, b, c = wedge[face]
            if np.dot(np.cross(b-a, c-a), (a+b+c)/3-wedge.mean(axis=0)) < 0:
                face[1], face[2] = face[2], face[1]
        write_obj(path, wedge, hull_faces)
        mesh_asset(assets, f"table_wedge_{i}", path)
        start = len(vertices)
        vertices.extend(wedge)
        # Top, bottom, hole wall, and outer tile wall. No radial internal faces.
        local_faces = [[0,1,2],[0,2,3],[4,6,5],[4,7,6],
                       [0,3,7],[0,7,4],[1,5,6],[1,6,2]]
        faces.extend([[start+k for k in face] for face in local_faces])
    visual_path = ROOT / "assets/table/meshes/tile.obj"
    write_obj(visual_path, vertices, faces)
    mesh_asset(assets, "table_tile_visual", visual_path)
    body = ET.SubElement(world, "body", name="table")
    holes = []
    for ix in range(1, count[0]+1):
        for iy in range(1, count[1]+1):
            center = grid_hole_position(ix, iy, config)
            tag = f"{ix:02d}_{iy:02d}"
            ET.SubElement(body, "geom", name=f"table_visual_{tag}", type="mesh", mesh="table_tile_visual",
                          pos=numbers(center), rgba="0.32 0.35 0.38 1", contype="0", conaffinity="0", group="2")
            for i in range(n):
                ET.SubElement(body, "geom", name=f"table_collision_{tag}_{i:02d}", type="mesh",
                              mesh=f"table_wedge_{i}", pos=numbers(center), group="3", rgba="0.5 0.5 0.5 1")
            ET.SubElement(body, "site", name=f"table_hole_{tag}", pos=numbers(center), size="0.001", group="4")
            holes.append({"grid": [ix, iy], "site": f"table_hole_{tag}", "position_m": center.tolist()})
    ET.SubElement(body, "site", name="table_origin", pos=numbers([0,0,table["top_z_m"]]), size="0.003", group="4")
    return holes


def add_robot(world, assets, actuators, config, visuals):
    source = ET.parse(ROOT / "assets/ur5e/source/ur5e.urdf").getroot()
    links = {x.get("name"): x for x in source.findall("link")}
    table_from_ur, ur_from_model = frame_transforms(config)
    ur = ET.SubElement(world, "body", name="ur_base", **pose_attributes(table_from_ur))
    ET.SubElement(ur, "site", name="ur_base_frame", size="0.004", group="4")
    model_body = ET.SubElement(ur, "body", name="urdf_model", **pose_attributes(ur_from_model))
    # Visual spacer only: actual mounting-plate CAD is unavailable.
    nominal_base = table_from_ur @ ur_from_model
    height = nominal_base[2, 3] - config["table"]["top_z_m"]
    # Keep the bottom on the tabletop even when the fitted model origin has
    # an additional z offset. The small base tilt is not modeled in this spacer.
    spacer_position = nominal_base[:3, 3].copy()
    spacer_position[2] -= height / 2
    ET.SubElement(world, "geom", name="robot_mount_spacer", type="cylinder",
                  size=numbers([config["robot"]["mount_spacer_radius_m"], height/2]),
                  pos=numbers(spacer_position), rgba="0.18 0.19 0.21 1", contype="0", conaffinity="0", group="2")
    bodies = {"world": model_body}
    pending = list(source.findall("joint"))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in bodies:
                continue
            name = joint.find("child").get("link")
            # The source has a separate controller-base dummy link named base.
            body = ET.SubElement(bodies[parent], "body", name="urdf_"+name,
                                 **pose_attributes(origin(joint.find("origin"))))
            bodies[name] = body
            link = links[name]
            inertial = link.find("inertial")
            if inertial is not None:
                tensor = inertial.find("inertia")
                mat = np.array([[float(tensor.get("ixx")),float(tensor.get("ixy")),float(tensor.get("ixz"))],
                                [float(tensor.get("ixy")),float(tensor.get("iyy")),float(tensor.get("iyz"))],
                                [float(tensor.get("ixz")),float(tensor.get("iyz")),float(tensor.get("izz"))]])
                frame = origin(inertial.find("origin"))
                mat = frame[:3,:3] @ mat @ frame[:3,:3].T
                ET.SubElement(body,"inertial",pos=numbers(frame[:3,3]),mass=inertial.find("mass").get("value"),
                              fullinertia=numbers([mat[0,0],mat[1,1],mat[2,2],mat[0,1],mat[0,2],mat[1,2]]))
            if joint.get("type") == "revolute":
                limit = joint.find("limit")
                joint_name = joint.get("name")
                ET.SubElement(body,"joint",name=joint_name,type="hinge",axis=joint.find("axis").get("xyz"),
                              range=limit.get("lower")+" "+limit.get("upper"), damping="2", armature="0.1")
                kp, kv = (2000, 100) if joint_name in JOINT_NAMES[:3] else (500, 30)
                effort=limit.get("effort")
                ET.SubElement(actuators,"general",name=joint_name.removesuffix("_joint"),joint=joint_name,
                              gaintype="fixed",biastype="affine",gainprm=str(kp),biasprm=f"0 {-kp} {-kv}",
                              ctrlrange=limit.get("lower")+" "+limit.get("upper"),forcerange=f"-{effort} {effort}")
            for visual in link.findall("visual"):
                filename = Path(visual.find("geometry/mesh").get("filename")).name
                for mesh_name, rgba in visuals[filename]:
                    ET.SubElement(body,"geom",type="mesh",mesh=mesh_name,rgba=rgba,
                                  contype="0",conaffinity="0",group="2",**pose_attributes(origin(visual.find("origin"))))
            for i, collision in enumerate(link.findall("collision")):
                filename = Path(collision.find("geometry/mesh").get("filename")).name
                mesh_name="ur_collision_"+name+str(i)
                mesh_asset(assets,mesh_name,ROOT/"assets/ur5e/source/meshes/collision"/filename)
                ET.SubElement(body,"geom",name=mesh_name,type="mesh",mesh=mesh_name,group="3",
                              **pose_attributes(origin(collision.find("origin"))))
            pending.remove(joint)
            progressed=True
        if not progressed:
            raise ValueError("Disconnected URDF tree")
    ET.SubElement(bodies["tool0"],"site",name="flange",size="0.004",group="4")
    add_gripper(bodies["tool0"],actuators,config)
    return table_from_ur,ur_from_model


def add_gripper(tool, actuators, config):
    adapter, grip = config["flange_adapter"], config["gripper"]
    radius, thick = adapter["radius_m"],adapter["thickness_m"]
    ET.SubElement(tool,"geom",name="flange_adapter",type="cylinder",size=numbers([radius,thick/2]),
                  pos=numbers([0,0,thick/2]),rgba="0.32 0.34 0.37 1",density=str(adapter["density_kg_m3"]))
    half = np.array(grip["body_size_sdf_m"])/2
    model_z=thick-grip["body_link_y_m"]+half[1]
    wsg=ET.SubElement(tool,"body",name="wsg50",pos=numbers([0,0,model_z]),quat="0.707106781187 0.707106781187 0 0")
    position=numbers([0,grip["body_link_y_m"],0])
    ET.SubElement(wsg,"geom",name="wsg_body_visual",type="mesh",mesh="wsg_body_mesh",pos=position,
                  contype="0",conaffinity="0",group="2",rgba="0.15 0.16 0.18 1",mass="0")
    ET.SubElement(wsg,"geom",name="wsg_body_collision",type="box",size=numbers(half),pos=position,
                  group="3",mass=str(grip["body_mass_kg"]))
    finger_size=np.array(grip["finger_size_tool0_m"])
    front_z=thick+2*half[1]
    for side,sign in [("left",-1),("right",1)]:
        finger=ET.SubElement(tool,"body",name=f"wsg_{side}_finger",
                             pos=numbers([sign*finger_size[0]/2,0,front_z+finger_size[2]/2]))
        ET.SubElement(finger,"joint",name=f"wsg_{side}_slide",type="slide",axis=f"{sign} 0 0",
                      range=numbers([0,grip["maximum_opening_m"]/2]),damping="5",armature="0.01")
        ET.SubElement(finger,"geom",name=f"wsg_{side}_pad",type="box",size=numbers(finger_size/2),
                      rgba="0.08 0.09 0.10 1",mass=str(grip["finger_mass_kg"]))
        ET.SubElement(finger,"site",name=f"wsg_{side}_inner",pos=numbers([-sign*finger_size[0]/2,0,0]),size="0.002",group="4")
        ET.SubElement(actuators,"general",name=f"wsg_{side}",joint=f"wsg_{side}_slide",
                      gaintype="fixed",biastype="affine",gainprm="1000",biasprm="0 -1000 -15",
                      ctrlrange=numbers([0,grip["maximum_opening_m"]/2]),forcerange="-40 40")
    ET.SubElement(tool,"site",name="tcp",pos=numbers(grip["tcp_from_tool0_m"]),size="0.004",group="4")


def build(config_path=None):
    config=load_config(config_path)
    root=ET.Element("mujoco",model="calibrated_ur5e_wsg50_bench")
    ET.SubElement(root,"compiler",angle="radian",autolimits="true",inertiafromgeom="auto",fusestatic="false")
    ET.SubElement(root,"option",timestep=str(config["simulation"]["timestep_s"]),integrator="implicitfast",gravity="0 0 -9.81")
    ET.SubElement(root,"size",njmax="2000",nconmax="2000")
    default=ET.SubElement(root,"default")
    ET.SubElement(default,"geom",friction=numbers(config["simulation"]["friction"]),condim="3")
    visual=ET.SubElement(root,"visual")
    ET.SubElement(visual,"global",offwidth="1600",offheight="1200")
    ET.SubElement(visual,"headlight",ambient="0.35 0.35 0.35",diffuse="0.6 0.6 0.6",specular="0.2 0.2 0.2")
    ET.SubElement(visual,"quality",shadowsize="2048")
    assets=ET.SubElement(root,"asset")
    ET.SubElement(assets,"texture",type="skybox",builtin="gradient",rgb1="0.22 0.25 0.29",
                  rgb2="0.06 0.08 0.10",width="512",height="3072")
    world=ET.SubElement(root,"worldbody")
    ET.SubElement(world,"light",pos="0.4 -0.4 2",dir="0 0 -1",diffuse="0.7 0.7 0.7",castshadow="true")
    actuators=ET.SubElement(root,"actuator")
    visuals=convert_visuals(assets)
    holes=add_table(world,assets,config)
    table_from_ur,ur_from_model=add_robot(world,assets,actuators,config,visuals)
    contact=ET.SubElement(root,"contact")
    # Fixed mount interfaces and adjacent URDF links must not generate self contacts.
    chain=["base_link_inertia","shoulder_link","upper_arm_link","forearm_link","wrist_1_link","wrist_2_link","wrist_3_link"]
    for a,b in zip(chain,chain[1:]):
        ET.SubElement(contact,"exclude",body1="urdf_"+a,body2="urdf_"+b)
    for body in ["wsg_left_finger","wsg_right_finger"]:
        ET.SubElement(contact,"exclude",body1="wsg50",body2=body)
        ET.SubElement(contact,"exclude",body1="urdf_tool0",body2=body)
    keyframe=ET.SubElement(root,"keyframe")
    ready=np.r_[np.deg2rad(config["robot"]["initial_q_deg"]),np.repeat(config["gripper"]["opening_m"]/2,2)]
    ET.SubElement(keyframe,"key",name="ready",qpos=numbers(ready),ctrl=numbers(ready))
    output=ROOT/"models/bench.xml"
    # Components are complete MJCF fragments, not alternate sources of configuration.
    components=ROOT/"models/components"
    tool=world.find(".//body[@name='urdf_tool0']")
    adapter_fragment,gripper_fragment=ET.Element("mujoco"),ET.Element("mujoco")
    for child in list(tool):
        if child.get("name")=="flange":
            continue
        tool.remove(child)
        (adapter_fragment if child.get("name")=="flange_adapter" else gripper_fragment).append(child)
    save_xml(adapter_fragment,components/"flange_adapter.xml")
    save_xml(gripper_fragment,components/"wsg50.xml")
    ET.SubElement(tool,"include",file="flange_adapter.xml")
    ET.SubElement(tool,"include",file="wsg50.xml")
    for name,tag in [("assets","asset"),("actuators","actuator"),("contacts","contact")]:
        fragment=ET.Element("mujoco")
        child=copy.deepcopy(root.find(tag))
        fragment.append(child)
        save_xml(fragment,components/f"{name}.xml")
    for name,body in [("table",world.find("body[@name='table']")),("robot",world.find("body[@name='ur_base']"))]:
        fragment=ET.Element("mujoco")
        ET.SubElement(fragment,"worldbody").append(copy.deepcopy(body))
        save_xml(fragment,components/f"{name}.xml")
    # MuJoCo resolves mesh paths against the main XML (models/bench.xml).
    # Only include-file paths are resolved relative to the including XML.
    for tag in ("asset", "actuator", "contact"):
        root.remove(root.find(tag))
    for body in list(world.findall("body")):
        world.remove(body)
    for name in ("assets", "table", "robot", "actuators", "contacts"):
        ET.SubElement(root, "include", file=f"components/{name}.xml")
    save_xml(root, output)
    manifest={"config":config,"table_from_ur_base":table_from_ur.tolist(),
              "ur_base_from_urdf_model":ur_from_model.tolist(),"holes":holes,
              "joint_order":list(JOINT_NAMES)+["wsg_left_slide","wsg_right_slide"],
              "table_collision":f"{config['table']['hole_segments']} convex prisms per tile; polygonal through-holes; no solid collision slab"}
    (ROOT/"models/assembly.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Built {output}: {len(holes)} through-holes, {len(holes)*config['table']['hole_segments']} table collision geoms")
    print("URDF model -> UR base translation [mm]:",ur_from_model[:3,3]*1000)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path)
    build(parser.parse_args().config)
