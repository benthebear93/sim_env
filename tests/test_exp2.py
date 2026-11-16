"""Geometry, inherited calibration and contact checks for dual GelSight fingers."""
import json
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import trimesh

from sim_env.bench import ROOT, load_scene, set_gripper_opening


@pytest.fixture(scope="module")
def scene():
    return load_scene(ROOT / "models/exp2.xml")


def test_exp1_calibration_and_arm_are_preserved(scene):
    model, data = scene
    original, reference = load_scene()
    assert (model.nq, model.nu) == (8, 8)
    for name in ["ur_base", "urdf_tool0", "wsg50", "table"]:
        np.testing.assert_allclose(data.body(name).xpos, reference.body(name).xpos, atol=1e-12)
        np.testing.assert_allclose(data.body(name).xmat, reference.body(name).xmat, atol=1e-12)
    np.testing.assert_allclose(model.key_qpos[0, :6], original.key_qpos[0, :6])
    assert model.geom("wsg_body_collision").size.tolist() == original.geom("wsg_body_collision").size.tolist()
    assert not any(c.dist < -1e-5 for c in data.contact)


def test_short_finger_units_and_mating_plane():
    mesh = trimesh.load_mesh(ROOT / "assets/wsg50_gelsight/meshes/custom_finger.obj")
    np.testing.assert_allclose(mesh.bounds, [[-.000125, -.015, -.020], [.028875, .015, .041]], atol=1e-8)
    report = json.loads((ROOT / "models/exp2_assembly.json").read_text())
    mapped = np.array(report["finger_source_to_sensor"]) @ [26, -26, 15, 1]
    cover = trimesh.load_mesh(ROOT / "assets/wsg50_gelsight/source/back_cover.obj")
    np.testing.assert_allclose(mapped[:3], [cover.bounds[1, 0], 0, 0], atol=1e-8)


@pytest.mark.parametrize("opening", [0.0, .05, .11])
def test_opposing_gel_faces_and_opening(scene, opening):
    model, _ = scene
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    for side in ("left", "right"):
        data.joint(f"wsg_{side}_slide").qpos[0] = opening/2
    mujoco.mj_forward(model, data)
    left, right = [data.site(f"gelsight_{s}_surface") for s in ("left", "right")]
    left_normal = -left.xmat.reshape(3, 3)[:, 0]
    right_normal = -right.xmat.reshape(3, 3)[:, 0]
    np.testing.assert_allclose(left_normal, -right_normal, atol=1e-12)
    np.testing.assert_allclose(right.xpos-left.xpos, opening*left_normal, atol=1e-12)
    np.testing.assert_allclose(data.site("tcp").xpos, (left.xpos+right.xpos)/2, atol=1e-12)
    assert not any(c.dist < -1e-5 for c in data.contact)


def test_dynamic_open_close(scene):
    model, _ = scene
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    for opening in (.05, .02, .08, 0.):
        set_gripper_opening(model, data, opening)
        for _ in range(750):
            mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all()
        assert not data.warning.number.any()
        assert not any(c.dist < -.001 for c in data.contact)
        np.testing.assert_allclose(data.qpos[-2:], opening/2, atol=1e-5)
    assert np.max(np.abs(data.qpos[:6]-model.key_qpos[0, :6])) < .1


def test_probe_contacts_exposed_gel_faces(scene):
    def expanded(path):
        root = ET.parse(path).getroot()
        def visit(parent):
            for item in list(parent):
                if item.tag == "include":
                    fragment = expanded(path.parent / item.get("file"))
                    index = list(parent).index(item)
                    parent.remove(item)
                    for offset, child in enumerate(fragment):
                        parent.insert(index+offset, child)
                else:
                    visit(item)
        visit(root)
        return root
    xml = expanded(ROOT / "models/exp2.xml")
    for mesh in xml.findall("asset/mesh"):
        mesh.set("file", str((ROOT / "models" / mesh.get("file")).resolve()))
    xml.remove(xml.find("keyframe"))
    probe = ET.SubElement(xml.find("worldbody"), "body", name="probe")
    ET.SubElement(probe, "freejoint", name="probe_free")
    ET.SubElement(probe, "geom", name="probe_sphere", type="sphere", size=".001", mass=".01")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    source_model, initial = scene
    for j in range(source_model.njnt):
        name = source_model.joint(j).name
        data.joint(name).qpos[:] = initial.joint(name).qpos
    sphere_id = model.geom("probe_sphere").id
    for side in ("left", "right"):
        face = initial.site(f"gelsight_{side}_surface")
        inward = -face.xmat.reshape(3, 3)[:, 0]
        data.joint("probe_free").qpos[:] = np.r_[face.xpos + .0005*inward, [1, 0, 0, 0]]
        mujoco.mj_forward(model, data)
        others = [c.geom2 if c.geom1 == sphere_id else c.geom1 for c in data.contact
                  if sphere_id in (c.geom1, c.geom2)]
        assert others and all(g == model.geom(f"gelsight_{side}_gel").id for g in others)
    data.joint("probe_free").qpos[:] = np.r_[initial.site("tcp").xpos, [1, 0, 0, 0]]
    mujoco.mj_forward(model, data)
    assert not any(sphere_id in (c.geom1, c.geom2) for c in data.contact)


def test_exp2_portability(tmp_path):
    shutil.copytree(ROOT / "models", tmp_path / "models")
    shutil.copytree(ROOT / "assets", tmp_path / "assets")
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "models/exp2.xml"))
    assert model.nq == 8
