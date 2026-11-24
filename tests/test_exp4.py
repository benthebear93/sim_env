"""Verify the single right-arm scene preserves Exp 3's geometry and controls."""
import json
import shutil

import mujoco
import numpy as np
import pytest

from sim_env.bench import JOINT_NAMES, ROOT, load_scene

pytestmark = pytest.mark.skipif(
    tuple(int(x) for x in mujoco.__version__.split(".")[:2]) < (3, 13),
    reason="Run Setup 4 validation with MuJoCo 3.13+.",
)


@pytest.fixture(scope="module")
def scene():
    return load_scene(ROOT / "models/exp4.xml")


def test_only_right_robot_and_exp3_geometry_preserved(scene):
    model, data = scene
    original, reference = load_scene(ROOT / "models/exp3.xml")
    config = json.loads((ROOT / "configs/exp4.json").read_text())
    placement = config["robots"]["right"]
    # Compare source kinematics at Exp 4's relocated mount and solved joints.
    original.body_pos[original.body("right_mount").id] = placement["base_position_m"]
    assert (model.nq, model.nv, model.nu) == (26, 26, 26)
    for name in ("left_mount", "left_hand_mount", "hx5_d20_left_base"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) == -1
    for jid in range(model.njnt):
        name = model.joint(jid).name
        assert name.startswith(("right_", "finger_r_"))
        reference.joint(name).qpos[:] = data.joint(name).qpos
        np.testing.assert_allclose(model.joint(name).range, original.joint(name).range)
    mujoco.mj_forward(original, reference)
    for aid in range(model.nu):
        name = model.actuator(aid).name
        assert name.startswith(("right_", "finger_r_"))
        source_id = original.actuator(name).id
        joint_name = model.joint(model.actuator_trnid[aid, 0]).name
        np.testing.assert_allclose(data.ctrl[aid], data.joint(joint_name).qpos[0], atol=1e-10)
        assert joint_name == original.joint(original.actuator_trnid[source_id, 0]).name
        for field in ("gainprm", "biasprm", "ctrlrange", "forcerange"):
            np.testing.assert_allclose(getattr(model, f"actuator_{field}")[aid],
                                       getattr(original, f"actuator_{field}")[source_id])
    # Compare every retained body and site, including both tables, their hole
    # frames, the complete arm, the adapter, and every finger link.
    for bid in range(model.nbody):
        name = model.body(bid).name
        np.testing.assert_allclose(data.body(bid).xpos, reference.body(name).xpos, atol=1e-10)
        np.testing.assert_allclose(data.body(bid).xmat, reference.body(name).xmat, atol=1e-10)
    for sid in range(model.nsite):
        name = model.site(sid).name
        np.testing.assert_allclose(data.site(sid).xpos, reference.site(name).xpos, atol=1e-10)
        np.testing.assert_allclose(data.site(sid).xmat, reference.site(name).xmat, atol=1e-10)
    assert sum("table_hole_" in model.site(sid).name for sid in range(model.nsite)) == 768
    np.testing.assert_allclose(data.site("right_base_frame").xpos, placement["base_position_m"], atol=1e-10)
    # Local indices 2 and 3 from the front-right correspond to global
    # hole-site indices 23/22 in x and 15/14 in y.
    hole_midpoint = 0.5 * (data.site("right_table_hole_22_14").xpos
                           + data.site("right_table_hole_23_15").xpos)
    np.testing.assert_allclose(data.site("right_base_frame").xpos,
                               hole_midpoint + [0, 0, .041], atol=1e-10)
    np.testing.assert_allclose(data.site("right_flange").xpos, [1.45, -.20, .64], atol=1e-8)


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
    for name in JOINT_NAMES:
        assert abs(data.joint(f"right_{name}").qpos[0] - initial.joint(f"right_{name}").qpos[0]) < .1


def test_exp4_portability(tmp_path):
    (tmp_path / "models").mkdir()
    shutil.copy2(ROOT / "models/exp4.xml", tmp_path / "models/exp4.xml")
    shutil.copytree(ROOT / "models/exp4_components", tmp_path / "models/exp4_components")
    shutil.copytree(ROOT / "assets", tmp_path / "assets")
    model, data = load_scene(tmp_path / "models/exp4.xml")
    assert (model.nq, model.nu) == (26, 26)
    np.testing.assert_allclose(data.site("right_flange").xpos, [1.45, -.20, .64], atol=1e-8)
