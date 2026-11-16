# WSG-50 custom fingers and GelSight Mini

Copied from `diffpatch_ex` for [Experimental Setup 2](../../docs/setup_2.md).
[`source.json`](source.json) lists exact local source paths and SHA-256 hashes.
No external checkout or recorded experiment is needed to load or rebuild the scene.

| File | Use and units |
|---|---|
| `source/WSG50_GELSIGHT_fin.stl` | Short custom finger; source bounds [0, −46, 0] to [29, 15, 30], interpreted as mm |
| `source/case.obj`, `source/back_cover.obj` | GelSight Mini housing meshes in meters, expressed in the sensor frame |
| `meshes/custom_finger.obj` | Generated finger mesh in meters and the same sensor frame |
| `source/mujoco_system_reference.py` | Unmodified source reference for pocket registration and mounting; never imported by the builder |
| `source/setup_reference.json` | Original hardware setup metadata and unknown installation parameters |
| `source/geometry.json`, `source/source.json` | Original upstream geometry/provenance metadata |

The source metadata also describes an older, longer adapter. Exp 2 uses only
the **short `WSG50_GELSIGHT_fin.stl`** and the case/back cover meshes; the old
adapter geometry and its collision bounds are not imported.

The sensor frame has its origin at the gel center, +x pointing outward through
the housing, and +z pointing toward the WSG body. Its inward contact face is
x = −2.125 mm. The source finger coordinates transform as:

```text
x_sensor = 0.001 x_stl − 0.000125
y_sensor = −0.001 z_stl + 0.015
z_sensor = 0.001 y_stl + 0.026
```

STL coordinates are interpreted as millimeters; sensor coordinates above are
meters. The pocket point [26, −26, 15] mm maps to [25.875, 0, 0] mm, matching
the sensor back plane. This is a geometry-based mating transform, not a measured
installation. The source file is retained unchanged.

The GelSight repository and WSG licenses are retained in
[`source/GELSIGHT_REPOSITORY_LICENSE`](source/GELSIGHT_REPOSITORY_LICENSE) and
[`source/WSG_LICENSE`](source/WSG_LICENSE). No separate license was supplied for
the local custom finger CAD. See the retained provenance for upstream CAD links.

Regenerate with `python scripts/build_exp2.py`. Tune installation in
[`configs/exp2.json`](../../configs/exp2.json); do not edit source meshes.
