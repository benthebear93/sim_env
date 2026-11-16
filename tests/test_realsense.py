"""Checks for scale, mounting frames and portable RealSense assets."""
import shutil

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from sim_env.bench import ROOT

LIBRARY = ROOT / "assets/realsense"
PARAMETERS = [
    ("d405", [.01085, .009, .021], [.0038, -.009, 0], [.023, .042, .042], .018),
    ("d455", [.01115, .0475, .0145], [.00465, -.0475, 0], [.026, .124, .029], .095),
]


@pytest.mark.parametrize("name,link_pos,visual_pos,dimensions,baseline", PARAMETERS)
def test_scale_and_nominal_frames(name, link_pos, visual_pos, dimensions, baseline):
    model = mujoco.MjModel.from_xml_path(str(LIBRARY / name / f"{name}.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert (model.nq, model.nu) == (0, 0)
    np.testing.assert_allclose(data.body(name+"_bottom_screw_frame").xpos, [0, 0, 0])
    np.testing.assert_allclose(data.body(name+"_link").xpos, link_pos, atol=1e-12)
    np.testing.assert_allclose(data.body(name+"_infra2_frame").xpos-data.body(name+"_infra1_frame").xpos,
                               [0, -baseline, 0], atol=1e-12)
    for kind in ("depth", "infra1", "infra2", "color"):
        rotation = data.body(f"{name}_{kind}_optical_frame").xmat.reshape(3, 3)
        np.testing.assert_allclose(rotation, [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], atol=1e-10)
    collision = model.geom(name+"_collision")
    np.testing.assert_allclose(collision.size*2, dimensions, atol=1e-12)
    assert model.geom(name+"_visual").contype[0] == 0
    assert collision.contype[0] != 0
    # Verify compiled vertex transforms, including MuJoCo's mesh recentering,
    # against the source URDF visual pose and millimeter-to-meter conversion.
    source = trimesh.load_mesh(LIBRARY / f"source/realsense2_description/meshes/{name}.stl")
    rotation = Rotation.from_euler("xyz", [np.pi/2, 0, np.pi/2]).as_matrix()
    expected = source.vertices*.001 @ rotation.T + visual_pos + np.array(link_pos)
    mesh = model.mesh(name+"_mesh")
    vertices = model.mesh_vert[mesh.vertadr[0]:mesh.vertadr[0]+mesh.vertnum[0]]
    geom = data.geom(name+"_visual")
    actual = vertices @ geom.xmat.reshape(3, 3).T + geom.xpos
    np.testing.assert_allclose([actual.min(0), actual.max(0)], [expected.min(0), expected.max(0)], atol=1e-7)


def test_camera_assets_portable(tmp_path):
    copied = tmp_path / "realsense"
    shutil.copytree(LIBRARY, copied)
    for name in ("d405", "d455"):
        model = mujoco.MjModel.from_xml_path(str(copied / name / "scene.xml"))
        assert model.body(name+"_bottom_screw_frame").id > 0


@pytest.mark.skipif(tuple(int(x) for x in mujoco.__version__.split(".")[:2]) < (3, 13),
                    reason="Model/attach composition is validated with MuJoCo 3.13+.")
def test_two_camera_instances_attach_at_mounts():
    xml = f'''<mujoco><asset>
      <model name="d405_asset" file="{LIBRARY}/d405/d405.xml"/>
      <model name="d455_asset" file="{LIBRARY}/d455/d455.xml"/></asset>
      <worldbody><body name="near_mount" pos=".1 0 .2">
      <attach model="d405_asset" body="d405_bottom_screw_frame" prefix="near_"/></body>
      <body name="far_mount" pos="-.1 0 .2">
      <attach model="d455_asset" body="d455_bottom_screw_frame" prefix="far_"/></body></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.body("near_d405_bottom_screw_frame").xpos, [.1, 0, .2])
    np.testing.assert_allclose(data.body("far_d455_bottom_screw_frame").xpos, [-.1, 0, .2])
