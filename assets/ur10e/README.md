# Universal Robots UR10e assets

Original MuJoCo Menagerie UR10e MJCF and OBJ meshes are retained in
[`source/`](source/README.md), with the upstream [BSD-3-Clause license](source/LICENSE).
They were copied from a local Menagerie archive; its upstream commit is unknown.
[`source.json`](source.json) records the source paths and SHA-256 hashes.

The standalone model is [`source/ur10e.xml`](source/ur10e.xml): six revolute joints
and six position actuators, with geometry in meters and angles in radians.
The source base body has a 90° rotation about z. Visual geoms use group 2;
collision geoms use group 3.

Exp 2 imports this model twice, prefixes identifiers with `left_` and `right_`,
and adds installation transforms and hand adapters in generated MJCF. Source
files remain unchanged. The assembly moves `attachment_site` from wrist-3
local y = 0.100 m to 0.11655 m, placing the hand adapter at the nominal flange
face. That offset comes from the retained Universal Robots
[`default_kinematics.yaml`](reference/default_kinematics.yaml), whose
[license](reference/LICENSE) is included. It is not measured calibration.

Edit [`configs/exp2.json`](../../configs/exp2.json) to change installation poses,
adapter geometry, or the flange offset, then run `python scripts/build_exp2.py`.
See [Setup 2](../../docs/setup_2.md) for the assembled scene and environment.
