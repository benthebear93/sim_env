# Simulation Environments

Reusable MuJoCo scenes and robot assets for manipulation experiments.

| Experiment | Setup | Scene |
|---|---|---|
| [Exp 1](docs/setup_1.md) | UR5e, flange adapter, and WSG-50 on a Siegmund welding table | <img src="docs/images/exp1.png" alt="Experimental Setup 1" width="480"> |
| [Exp 2](docs/setup_2.md) | Exp 1 with custom WSG-50 fingers and dual GelSight Mini sensors | <img src="docs/images/exp2.png" alt="Experimental Setup 2" width="480"> |
| [Exp 3](docs/setup_3.md) | Dual UR10e with left/right HX5-D20 hands on two joined Siegmund welding tables | <img src="docs/images/exp3.png" alt="Experimental Setup 3" width="480"> |

```bash
python3 scripts/view_scene.py
python3 scripts/view_scene.py --setup 2
python3 scripts/view_scene.py --setup 3  # MuJoCo 3.13+
```
