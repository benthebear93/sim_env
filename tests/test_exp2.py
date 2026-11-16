"""Layout and kinematic checks for the dual UR10e / HX5-D20 assembly."""
import json
import shutil

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim_env.bench import JOINT_NAMES, ROOT, load_scene

pytestmark = pytest.mark.skipif(
    tuple(int(x) for x in mujoco.__version__.split(".")[:2]) < (3, 13),
    reason="Run Setup 2 validation with the modern MuJoCo environment (3.13+).",
)


@pytest.fixture(scope="module")
def scene():
    return load_scene(ROOT / "models/exp2.xml")


def test_layout_and_hand_assignment(scene):
    model, data = scene
    assert (model.nq, model.nv, model.nu) == (52, 52, 52)
    np.testing.assert_allclose(data.body("left_table").xpos, [0,0,0])
    np.testing.assert_allclose(data.body("right_table").xpos, [1.2,0,0])
    np.testing.assert_allclose(data.site("left_base_frame").xpos, [.175,-.625,.041])
    np.testing.assert_allclose(data.site("right_base_frame").xpos, [2.225,-.625,.041])
    for side in ("left", "right"):
        hand = model.body(f"hx5_d20_{side}_base").id
        assert model.body(model.body_parentid[hand]).name == f"{side}_hand_mount"
        np.testing.assert_allclose(data.body(hand).xpos, data.site(f"{side}_hand_base_frame").xpos)
        expected = np.array([0,0,.020])
        flange = data.site(f"{side}_flange")
        delta = flange.xmat.reshape(3,3).T @ (data.body(hand).xpos-flange.xpos)
        np.testing.assert_allclose(delta, expected, atol=1e-10)


@pytest.mark.parametrize("side", ["left", "right"])
def test_ur10e_fk_matches_source_and_ready_target(scene, side):
    model, data = scene
    config = json.loads((ROOT / "configs/exp2.json").read_text())
    nominal = mujoco.MjModel.from_xml_path(str(ROOT / "assets/ur10e/source/ur10e.xml"))
    reference = mujoco.MjData(nominal)
    for name in JOINT_NAMES:
        reference.joint(name).qpos[0] = data.joint(f"{side}_{name}").qpos[0]
    mujoco.mj_forward(nominal, reference)
    source_site = reference.site("attachment_site")
    source_rotation = source_site.xmat.reshape(3,3)
    # The copied model's attachment marker is inside its flange. The
    # assembly corrects the marker's wrist3-y offset from 0.1 to 0.11655 m.
    delta = config["mount"]["tool0_wrist3_y_m"] - .1
    source_position = source_site.xpos + source_rotation @ [0,0,delta]
    mount = config["robots"][side]
    rotation = Rotation.from_euler("z", mount["mount_yaw_deg"], degrees=True).as_matrix()
    site = data.site(f"{side}_flange")
    np.testing.assert_allclose(site.xpos, rotation @ source_position + mount["base_position_m"], atol=1e-10)
    np.testing.assert_allclose(site.xmat.reshape(3,3), rotation @ source_rotation, atol=1e-10)
    np.testing.assert_allclose(site.xpos, mount["flange_target_position_m"], atol=1e-8)


def test_both_tables_have_open_holes(scene):
    model, data = scene
    group = np.array([0,0,0,1,0,0], dtype=np.uint8)
    hit = np.zeros(1, dtype=np.int32)
    for side in ("left", "right"):
        for ix in range(1,25):
            for iy in range(1,17):
                point = data.site(f"{side}_table_hole_{ix:02d}_{iy:02d}").xpos
                distance = mujoco.mj_ray(model, data, point+[0,0,.0001], np.array([0.,0.,-1.]), group, 1, -1, hit)
                assert distance < 0 or distance > .036
    # Table extents meet exactly at x=1.2; the collision surface also exists
    # immediately to either side of the seam.
    for x in (1.199, 1.201):
        distance = mujoco.mj_ray(model, data, np.array([x,-.4,.01]), np.array([0.,0.,-1.]), group, 1, -1, hit)
        assert abs(distance-.01)<1e-7


def test_initial_contacts_and_dynamic_hold(scene):
    model, initial = scene
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    mujoco.mj_forward(model, data)
    assert not any(c.dist < -1e-5 for c in data.contact)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert not data.warning.number.any()
    assert not any(c.dist < -.001 for c in data.contact)
    for side in ("left", "right"):
        for name in JOINT_NAMES:
            assert abs(data.joint(f"{side}_{name}").qpos[0]-initial.joint(f"{side}_{name}").qpos[0]) < .1


def test_exp2_portability(tmp_path):
    shutil.copytree(ROOT/"models", tmp_path/"models")
    shutil.copytree(ROOT/"assets", tmp_path/"assets")
    model = mujoco.MjModel.from_xml_path(str(tmp_path/"models/exp2.xml"))
    assert model.nq == 52
