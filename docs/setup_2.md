# Experimental Setup 2

Exp 1's **UR5e, 20 mm flange adapter, WSG-50, and Siegmund welding table** with
two short **`WSG50_GELSIGHT_fin.stl` fingers and GelSight Mini sensors with curved gels** from
`diffpatch_ex`. The robot/table calibration and initial arm pose are unchanged.

![Experimental Setup 2](images/exp2.png)

![Custom fingers and dual GelSight Mini sensors with curved gels](images/exp2_tool.png)

```bash
python scripts/view_scene.py --setup 2
python scripts/view_scene.py --setup 2 --view tool
python scripts/view_scene.py --setup 2 --step
python scripts/build_exp2.py
python scripts/view_scene.py --setup 2 --render outputs/exp2.png --software
python scripts/view_scene.py --setup 2 --view tool --render outputs/exp2_tool.png --software
```

Edit [`configs/exp2.json`](../configs/exp2.json) for the sensor mounting,
opening-to-gap reference, assumed masses, and camera. Rebuild Exp 1 first if
[`configs/bench.json`](../configs/bench.json) changes. The scene is
[`models/exp2.xml`](../models/exp2.xml); resolved transforms and provenance of
the base calibration are recorded in
[`models/exp2_assembly.json`](../models/exp2_assembly.json).

The finger STL is interpreted in millimeters and registered to the sensor
using the short-pocket transform in `diffpatch_ex.mujoco_system`. The Mini case
and back cover meshes are already in meters. See the
[asset documentation](../assets/wsg50_gelsight/README.md) for sources and frames.

Both sensors face inward with zero tilt. The 14 mm carriage extension and
finger mating transform are inherited assembly assumptions; Exp 2 does not
import recording-specific camera corrections, sensor tilt, or fitted materials.
The nominal ready opening is 50 mm. Each slide coordinate is half the WSG
opening; the default gives a **50.5 mm gap between the curved gel apexes**.
The previous finger/housing poses are retained: the new apex recedes 0.25 mm
per side compared with the former flat box. This nominal mapping is
uncalibrated and can be changed using `reference_opening_m` and
`surface_gap_at_reference_m`. For parallel sensors:
`apex_gap = opening + surface_gap_at_reference − reference_opening`.
Here `opening` is the WSG register and the two reference parameters define
its nominal offset from the closest-point gap.

The scene has eight joints and eight actuators. `gelsight_left_surface` and
`gelsight_right_surface` mark the inward gel apexes; `tcp` is their nominal
midpoint, 147.5 mm along tool0 +z. The flange and WSG body are shared with Exp 1.

Gel geometry uses the **FEATS GelSight Mini CAD**, with a maximum thickness of
**4.0 mm** and footprint **20.75 × 25.25 mm**. The back is fixed at sensor
x = +2.125 mm and the curved apex is x = −1.875 mm. At 8 mm lateral offset,
the front recedes about 0.301 mm; at 10 mm vertical offset, about 0.272 mm.
The exact registered mesh is used for rendering. MuJoCo uses its convex hull
for collision; six sphere-probe samples verify contact within 20 micrometers
of the CAD front. The hull's volume exceeds the source mesh by about 0.0274%.
This is nominal published CAD, not a scan of the individual gel.

The housing uses a box proxy and each finger uses a convex collision hull.
Bracket pockets and bolt holes are visual details in this collision
approximation. Masses, inertia, and contact parameters are simulation
assumptions. Curved geometry does not add optical tactile rendering or
continuum gel deformation.

Validation covers unchanged Exp 1 base/flange transforms, CAD units and mating
planes, opposing sensor normals, apex gaps over the travel range, curved CAD
profiles and native sphere contacts, dynamic opening/closing, source hashes,
and relocation.
