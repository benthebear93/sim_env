# UR5e Base Calibration

This directory stores the bench base calibration imported from
`/home/haegu/ur_controller`.

## Frame Convention

`ur5e_base_calibration.json` defines `T_world_from_ur_base`.

For a point measured in the UR RTDE base frame:

```text
p_world = R_world_from_ur_base * p_ur_base + t_world_from_ur_base
```

For a world/table target that must be sent to the UR controller:

```text
p_ur_base = R_world_from_ur_base^T * (p_world - t_world_from_ur_base)
```

The transform values are in meters. The source sample points are kept in
millimeters because that is how the original calibration script stored them.

## Source

- Source repo: `/home/haegu/ur_controller`
- Source point files:
  - `calibration/robot_base_p.pkl`
  - `calibration/world_base_p_prime.pkl`
- Source computation: `calibration/calibration_from_data.py`
- Matching constants in source: `config.py` `UR5E_BASE_POS` and
  `UR5E_BASE_ROT`

The original source also had a `UR5E_BASE_QUAT` constant, but its component
ordering/signs do not reconstruct the stored rotation matrix. Use the rotation
matrix in this directory as the primary calibration value.
