# ROBOTIS HX5-D20 hand assets — left and right

Official **HX5-D20 rev2** models from
[ROBOTIS-GIT/robotis_hand](https://github.com/ROBOTIS-GIT/robotis_hand), stored as
standalone assets for future experimental setups.

- Pinned commit: `7730b76bbac51786b122023a7ab2afc17afc7d12`.
- License: Apache-2.0; the original [LICENSE](LICENSE) is included.
- Provenance: [source.json](source.json) records upstream paths, Git blob IDs,
  SHA-256 hashes, and file sizes for all 32 copied files.
- Upstream model contents are unmodified. The original directory layout for
  mesh and MJCF references is preserved.
- Official reference: [ROBOTIS simulation documentation](https://docs.robotis.com/docs/systems/hx5_d20/simulation/).

## Model entry points

| Model | MJCF model | Preview scene | URDF/Xacro macro |
|---|---|---|---|
| Left | [hx5_d20_left.xml](mujoco/hx5_d20_rev2/hx5_d20_left.xml) | [scene_left.xml](mujoco/hx5_d20_rev2/scene_left.xml) | [hx5_d20_left.urdf.xacro](urdf/hx5_d20_rev2/hx5_d20_left.urdf.xacro) |
| Right | [hx5_d20_right.xml](mujoco/hx5_d20_rev2/hx5_d20_right.xml) | [scene_right.xml](mujoco/hx5_d20_rev2/scene_right.xml) | [hx5_d20_right.urdf.xacro](urdf/hx5_d20_rev2/hx5_d20_right.urdf.xacro) |
| Both | [hx5_d20.xml](mujoco/hx5_d20_rev2/hx5_d20.xml) | [scene.xml](mujoco/hx5_d20_rev2/scene.xml) | Use the two macros with their respective prefixes |

Each hand has 20 revolute joints and 20 position actuators. The combined model
has 40 joints and 40 actuators. Left and right meshes are stored separately in
`meshes/hx5_d20_rev2/hx5_d20_left/` and `hx5_d20_right/`.
Each side includes two convex palm collision meshes in its `collision/` directory.

STL vertices are in **millimeters**. The supplied MJCF and Xacro files apply
`scale="0.001 0.001 0.001"` to convert them to meters. Joint positions and
inertial offsets are in meters, and joint angles are in radians.

The attachment bodies are `hx5_d20_left_base` and `hx5_d20_right_base`.
Joint names are `finger_l_joint1` through `finger_l_joint20` and
`finger_r_joint1` through `finger_r_joint20`.
The combined preview places their base origins at `[0, 0.18, 0]` and
`[0, -0.18, 0] m`; these are preview placements, not calibrated bench mounts.

## MuJoCo compatibility and viewing

All three preview scenes were loaded and stepped successfully with
**MuJoCo 3.13.0**. The bench's older system MuJoCo 3.1.5 cannot load these
unmodified upstream models: it rejects the position actuator's `dampratio`
attribute. The combined model also uses the newer MJCF `model`/`attach`
composition mechanism. Use an environment with MuJoCo 3.13.0 for the validated
configuration; no compatibility edits have been applied to the upstream XML.

From the repository root, in that environment:

```bash
python -m mujoco.viewer --mjcf="$PWD/assets/robotis_hx5_d20/mujoco/hx5_d20_rev2/scene.xml"
python -m mujoco.viewer --mjcf="$PWD/assets/robotis_hx5_d20/mujoco/hx5_d20_rev2/scene_left.xml"
python -m mujoco.viewer --mjcf="$PWD/assets/robotis_hx5_d20/mujoco/hx5_d20_rev2/scene_right.xml"
```

The existing local interpreter used for validation is
`/home/haegu/DifferentiablePatch/.venv-ss/bin/python`.
The model files themselves contain no dependency on that interpreter path.

```python
from pathlib import Path
import mujoco

scene_path = Path(
    "assets/robotis_hx5_d20/mujoco/hx5_d20_rev2/scene_left.xml"
).resolve()
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
```

Visual meshes use geom group 2; collision meshes use group 3. Hide group 3
when inspecting the visual model. The upstream actuator gains and torque
limits are simulation approximations, not identified hardware control gains.
Use absolute scene paths so nested model attachments resolve consistently.

## URDF reuse and validation

The supplied `.urdf.xacro` files are geometry/kinematics macros, not expanded
standalone URDF files. Instantiate them using Xacro with a `prefix` argument.
Their mesh URIs retain the upstream `package://robotis_hand_description/`
prefix, which must be resolved by a ROS package installation or rewritten by
the importing tool. ROS launch, Gazebo, and hardware controller wrappers are
not part of this asset subset. `upstream_package.xml` is retained as metadata.

Validation checked the copied files against the pinned upstream Git blob IDs,
loaded all three scenes, stepped each for 100 steps without numerical warnings,
and reloaded them after copying this entire asset directory to a different
location. The validation report is generated at
`outputs/hx5_validation.json` in the repository root.
