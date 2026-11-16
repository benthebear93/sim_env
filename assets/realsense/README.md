# RealSense D455 and D405

Reusable camera geometry and nominal frames from the official
[RealSense ROS description](https://github.com/realsenseai/realsense-ros/tree/960306819f45006ca3f903aa046bed32c509679a/realsense2_description).
These assets are stored independently of the experiment scenes.

| Camera | MuJoCo model | Preview | Original URDF macro | Nominal housing W × H × D |
|---|---|---|---|---|
| D455 | [d455.xml](d455/d455.xml) | [scene.xml](d455/scene.xml) | [_d455.urdf.xacro](source/realsense2_description/urdf/_d455.urdf.xacro) | 124 × 29 × 26 mm |
| D405 | [d405.xml](d405/d405.xml) | [scene.xml](d405/scene.xml) | [_d405.urdf.xacro](source/realsense2_description/urdf/_d405.urdf.xacro) | 42 × 42 × 23 mm |

From the repository root:

```bash
python -m mujoco.viewer --mjcf="$PWD/assets/realsense/d455/scene.xml"
python -m mujoco.viewer --mjcf="$PWD/assets/realsense/d405/scene.xml"
python scripts/build_realsense_assets.py
```

Standalone models support MuJoCo 3.1.5 and newer. Visual geometry uses group 2,
collision boxes group 3, and frame sites group 4. Hide groups 3 and 4 when
viewing the housing. The upstream STL models are untextured; no lens colors or
camera textures are inferred.

## Frames and reuse

The root body is `d455_bottom_screw_frame` or `d405_bottom_screw_frame`, at the
upstream bottom screw mounting origin. Body axes are +x forward, +y left, +z up.
Depth, infra1, infra2, color, and optical frames retain the upstream names
with the camera model as prefix. Each frame also has a site named
`<frame_name>_site`. Optical axes are +x right, +y down, +z forward.
D455 also retains the nominal IMU frame tree.

The nominal stereo baselines are 95 mm for D455 and 18 mm for D405. Frames are
URDF nominal values; per-device calibrated intrinsics and extrinsics are not
included. These assets do not implement RGB/depth streams, stereo matching,
depth noise, or an IMU measurement model.

For MuJoCo 3.13+, attach a model at its screw mount with a unique instance prefix.
For example, in a scene XML placed at the repository root:

```xml
<mujoco>
  <asset>
    <model name="camera_asset" file="assets/realsense/d455/d455.xml"/>
  </asset>
  <worldbody>
    <body name="camera_mount" pos="0.3 0 0.4">
      <attach model="camera_asset" body="d455_bottom_screw_frame" prefix="front_"/>
    </body>
  </worldbody>
</mujoco>
```

Use an absolute path when loading the outer scene. Update the installation
body's `pos`/`quat` for the actual mount. Each camera directory also contains
`assets.xml`, `body.xml`, and `metadata.json` for inspection or manual assembly.
Mesh paths are relative to the camera model's main XML; keep the `source/`
directory beside `d405/` and `d455/` when moving the library.

## Source and approximations

- Pinned upstream commit: `960306819f45006ca3f903aa046bed32c509679a`.
- [source.json](source.json) records SHA-256, Git blob SHA, and size for all
  13 original files. Original meshes, Xacro dependencies, package metadata,
  [Apache-2.0 license](source/LICENSE), and [NOTICE](source/NOTICE.md) are retained.
- STL coordinates are millimeters; MJCF applies the upstream `0.001` scale
  and visual mounting transform. Generated XML positions are meters.
- The builder enables nominal extrinsics and omits the optional USB plug.
  Plug meshes and its macro are retained for later use. The original Xacro
  files keep their ROS package references and require ROS/Xacro for direct use.
- The source explicitly labels its inertial values as unreliable. Generated
  models retain its **0.072 kg placeholder mass for each camera**, but replace
  the inertia with a uniform-box estimate at the collision box center. These
  are simulation assumptions, not measured D455/D405 dynamics.

Validation checks scale, compiled visual transforms, mount/optical frames,
nominal stereo baselines, model composition, relocation, and source hashes.
