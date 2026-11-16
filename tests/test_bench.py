"""Geometry, transform, portability and simulation regression checks."""
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from sim_env.bench import (
    JOINT_NAMES, ROOT, frame_transforms, grid_hole_position, load_config,
    load_scene, set_gripper_opening, urdf_fk,
)


@pytest.fixture(scope="module")
def model():
    return load_scene()[0]


def ready(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    mujoco.mj_forward(model, data)
    return data


@pytest.mark.parametrize("degrees", [
    [0,0,0,0,0,0],
    [-35.56,-101.41,-71.46,-88.19,88.47,-169.95],
    [-3.83,-108.92,-98.15,-61.33,90.44,-137.63],
])
def test_fk_matches_source_urdf(model, degrees):
    data = ready(model)
    q = np.deg2rad(degrees)
    for name, angle in zip(JOINT_NAMES, q):
        data.joint(name).qpos[0] = angle
    mujoco.mj_forward(model, data)
    table_from_ur, ur_from_model = frame_transforms(load_config())
    expected = table_from_ur @ ur_from_model @ urdf_fk(q)
    np.testing.assert_allclose(data.site("flange").xpos, expected[:3,3], atol=1e-10)
    np.testing.assert_allclose(data.site("flange").xmat.reshape(3,3), expected[:3,:3], atol=1e-10)


def test_measured_reference_and_tcp(model):
    config = load_config()
    data = ready(model)
    data.qpos[:6] = np.deg2rad(config["robot"]["reference_q_deg"])
    mujoco.mj_forward(model, data)
    table_from_ur, _ = frame_transforms(config)
    flange_ur = np.linalg.inv(table_from_ur) @ np.r_[data.site("flange").xpos, 1]
    np.testing.assert_allclose(flange_ur[:3], np.array(config["robot"]["reference_tool0_ur_base_mm"])/1000, atol=1e-10)
    delta = data.site("tcp").xpos - data.site("flange").xpos
    rotation = data.site("flange").xmat.reshape(3,3)
    np.testing.assert_allclose(rotation.T @ delta, config["gripper"]["tcp_from_tool0_m"], atol=1e-10)


def test_all_table_holes_are_open_and_grid_matches(model):
    data = ready(model)
    # Ray tests against only collision group 3. Exclude robot by checking a
    # segment confined to the tabletop rather than rays from above the robot.
    group = np.zeros(6, dtype=np.uint8)
    group[3] = 1
    gid = np.zeros(1, dtype=np.int32)
    for ix in range(1,25):
        for iy in range(1,17):
            point = grid_hole_position(ix, iy)
            np.testing.assert_allclose(data.site(f"table_hole_{ix:02d}_{iy:02d}").xpos, point, atol=1e-10)
            distance = mujoco.mj_ray(model, data, point+[0,0,0.0001], np.array([0.,0.,-1.]), group, 1, -1, gid)
            assert distance < 0 or distance > 0.036, (ix,iy,distance)
    np.testing.assert_allclose(grid_hole_position(13,9), [0.625,-0.425,0])
    point = grid_hole_position(13,9) + [0.015,0,0.01]
    distance = mujoco.mj_ray(model,data,point,np.array([0.,0.,-1.]),group,1,-1,gid)
    assert abs(distance-0.01) < 1e-7


def test_probe_contacts_table_but_passes_hole():
    # Actual collision test, because render/ray success alone cannot rule out
    # a single convex hull incorrectly filling a non-convex mesh's holes.
    def expanded(path):
        tree = ET.parse(path).getroot()
        def visit(parent):
            for child in list(parent):
                if child.tag == "include":
                    fragment = expanded(path.parent/child.get("file"))
                    index = list(parent).index(child)
                    parent.remove(child)
                    for offset, item in enumerate(fragment):
                        parent.insert(index+offset, item)
                else:
                    visit(child)
        visit(tree)
        return tree
    xml = expanded(ROOT/"models/bench.xml")
    for mesh in xml.findall("asset/mesh"):
        mesh.set("file",str((ROOT/"models"/mesh.get("file")).resolve()))
    xml.remove(xml.find("keyframe"))
    world = xml.find("worldbody")
    probe = ET.SubElement(world,"body",name="probe")
    ET.SubElement(probe,"freejoint",name="probe_free")
    ET.SubElement(probe,"geom",name="probe_sphere",type="sphere",size="0.001",mass="0.01")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding="unicode"))
    data = mujoco.MjData(model)
    pid = model.geom("probe_sphere").id
    center = grid_hole_position(13,9)
    def contacts_at(xyz):
        data.joint("probe_free").qpos[:] = np.r_[xyz,[1,0,0,0]]
        mujoco.mj_forward(model,data)
        return [c for c in data.contact if pid in (c.geom1,c.geom2)]
    for z in np.linspace(0.003,-0.038,15):
        assert not contacts_at(center+[0,0,z])
    assert contacts_at(center+[0.015,0,0.0005])
    assert contacts_at(center+[0.0075,0,-0.015])


@pytest.mark.parametrize("opening", [0.0,0.05,0.11])
def test_jaw_gap(model, opening):
    data = ready(model)
    set_gripper_opening(model,data,opening)
    for side in ("left","right"):
        data.joint(f"wsg_{side}_slide").qpos[0] = opening/2
    mujoco.mj_forward(model,data)
    distance = np.linalg.norm(data.site("wsg_left_inner").xpos-data.site("wsg_right_inner").xpos)
    assert abs(distance-opening)<1e-10


def test_dynamics_and_open_close(model):
    data=ready(model)
    assert data.ncon==0
    for target in (0.05,0.025,0.09):
        set_gripper_opening(model,data,target)
        for _ in range(500):
            mujoco.mj_step(model,data)
        assert np.isfinite(data.qpos).all()
        assert not data.warning.number.any()
        assert data.ncon==0
        np.testing.assert_allclose(data.qpos[-2:],target/2,atol=0.0001)
    assert np.max(np.abs(data.qpos[:6]-model.key_qpos[0,:6]))<0.1


def test_assets_portable(tmp_path):
    shutil.copytree(ROOT/"models",tmp_path/"models")
    # source assets are included as collision meshes; no external repository is needed.
    shutil.copytree(ROOT/"assets",tmp_path/"assets")
    model=mujoco.MjModel.from_xml_path(str(tmp_path/"models/bench.xml"))
    assert model.nq==8


def test_source_hashes():
    import hashlib
    for entry in json.loads((ROOT/"assets/sources.json").read_text())["files"]:
        assert hashlib.sha256((ROOT/entry["path"]).read_bytes()).hexdigest()==entry["sha256"]
