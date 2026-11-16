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
    # Housing mounts are unchanged: the curved CAD apex is 0.25 mm farther
    # back on each side than the former flat pad face.
    np.testing.assert_allclose(right.xpos-left.xpos, (opening+.0005)*left_normal, atol=1e-12)
    np.testing.assert_allclose(data.site("tcp").xpos, (left.xpos+right.xpos)/2, atol=1e-12)
    assert not any(c.dist < -1e-5 for c in data.contact)


def cad_front_x(mesh, y, z):
    """Intersect the original triangles along x, without using a convex hull."""
    triangles = mesh.triangles
    origin = triangles[:, 0, 1:]
    a, b = triangles[:, 1, 1:] - origin, triangles[:, 2, 1:] - origin
    d = np.array([y, z]) - origin
    determinant = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]
    valid = np.abs(determinant) > 1e-18
    u = np.divide(d[:, 0]*b[:, 1]-d[:, 1]*b[:, 0], determinant,
                  out=np.full_like(determinant, np.nan), where=valid)
    v = np.divide(a[:, 0]*d[:, 1]-a[:, 1]*d[:, 0], determinant,
                  out=np.full_like(determinant, np.nan), where=valid)
    inside = valid & (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9)
    x = triangles[:, 0, 0] + u*(triangles[:, 1, 0]-triangles[:, 0, 0]) + v*(triangles[:, 2, 0]-triangles[:, 0, 0])
    return float(x[inside].min())


def test_curved_cad_registration_and_housing_alignment(scene):
    mesh = trimesh.load_mesh(ROOT / "assets/wsg50_gelsight/meshes/gel_curved_sensor_m.obj")
    reference = trimesh.load_mesh(ROOT / "assets/wsg50_gelsight/source/curved/gel_curved_sensor_m.obj")
    np.testing.assert_allclose(mesh.vertices, reference.vertices, atol=1e-8)
    assert mesh.is_watertight and mesh.is_winding_consistent
    np.testing.assert_allclose(mesh.bounds, [[-.001875, -.010375, -.012625], [.002125, .010375, .012625]], atol=2e-9)
    case = trimesh.load_mesh(ROOT / "assets/wsg50_gelsight/source/case.obj")
    assert abs(mesh.bounds[1, 0]-case.bounds[0, 0]) < 1e-9
    assert cad_front_x(mesh, .008, 0)-cad_front_x(mesh, 0, 0) > .00025
    assert cad_front_x(mesh, 0, .01)-cad_front_x(mesh, 0, 0) > .00025
    model, data = scene
    for side in ("left", "right"):
        geom = model.geom(f"gelsight_{side}_gel")
        assert geom.type[0] == mujoco.mjtGeom.mjGEOM_MESH
        assert geom.dataid[0] == model.mesh("gs_curved_gel").id
        # The sensor's back remains 29.25 mm from the central plane at ready.
        center = data.body(f"gelsight_{side}").xpos
        assert abs(np.linalg.norm(center-data.site("tcp").xpos)-.027125) < 1e-12


def test_native_contact_tracks_curved_front():
    path = ROOT / "assets/wsg50_gelsight/meshes/gel_curved_sensor_m.obj"
    mesh = trimesh.load_mesh(path)
    # A 50 micrometer sphere locates the compiled MuJoCo contact surface.
    # This also detects accidental replacement by a flat box or flat hull.
    radius = .00005
    xml = f'''<mujoco><asset><mesh name="gel" file="{path}"/></asset>
      <worldbody><geom name="gel" type="mesh" mesh="gel"/>
      <body><freejoint name="probe"/><geom name="sphere" type="sphere" size="{radius}" mass=".001"/></body>
      </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    positions = [(0, 0), (.008, 0), (-.008, 0), (0, .01), (0, -.01), (.008, .01)]
    measured = []
    for y, z in positions:
        front = cad_front_x(mesh, y, z)
        lo, hi = front-2*radius, front+radius
        for _ in range(22):
            x = (lo+hi)/2
            data.joint("probe").qpos[:] = [x, y, z, 1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            if any(c.dist < 0 for c in data.contact):
                hi = x
            else:
                lo = x
        measured.append((lo+hi)/2+radius)
        assert abs(measured[-1]-front) < .00002
    assert measured[1]-measured[0] > .00025
    assert measured[3]-measured[0] > .00025


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
