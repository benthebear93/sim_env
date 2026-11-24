#!/usr/bin/env python3
"""Open the bench in MuJoCo, or render PNGs without connecting to hardware."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=int, choices=[1, 2, 3, 4], default=1,
                        help="Experimental setup to load (default: 1)")
    parser.add_argument("--render", type=Path, help="Save a PNG and exit")
    parser.add_argument("--view", choices=["overview", "tool", "top"], default="overview")
    parser.add_argument("--step", action="store_true", help="Run dynamics in the interactive viewer")
    parser.add_argument("--software", action="store_true", help="Use Mesa software EGL rendering")
    args = parser.parse_args()
    if args.render:
        os.environ.setdefault("MUJOCO_GL", "egl")
    if args.software:
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["EGL_PLATFORM"] = "surfaceless"
    import mujoco
    from sim_env.bench import load_scene
    scene_path = ROOT / "models" / ("bench.xml" if args.setup == 1 else f"exp{args.setup}.xml")
    try:
        model, data = load_scene(scene_path)
    except ValueError as error:
        if args.setup in (3, 4) and "Schema violation" in str(error):
            raise SystemExit(
                f"Setup {args.setup} includes upstream HX5-D20 actuators requiring newer MuJoCo. "
                f"Use the MuJoCo 3.13+ environment described in docs/setup_{args.setup}.md.\n"
                + str(error)
            ) from error
        raise
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    if args.setup in (3, 4):
        scene_config = json.loads((ROOT / f"configs/exp{args.setup}.json").read_text())
        camera_config = scene_config["camera"]
        camera.lookat[:] = camera_config["lookat"]
        camera.distance = camera_config["distance"]
        camera.azimuth = camera_config["azimuth"]
        camera.elevation = camera_config["elevation"]
        if args.view == "top":
            camera.lookat[:] = [1.2, -0.4, 0]
            camera.distance, camera.azimuth, camera.elevation = 3.3, 90, -89.9
        elif args.view == "tool":
            camera.lookat[:] = sum(data.site(f"{side}_flange").xpos for side in scene_config["robots"]) / len(scene_config["robots"])
            camera.distance = 1.25 if len(scene_config["robots"]) == 2 else 0.65
            camera.azimuth, camera.elevation = 90, -20
    elif args.view == "tool":
        camera.lookat[:] = data.site("tcp" if args.setup == 2 else "flange").xpos
        camera.distance, camera.azimuth, camera.elevation = 0.55, 140, -18
        if args.setup == 2:
            camera.lookat[:] = 0.75 * data.site("tcp").xpos + 0.25 * data.site("flange").xpos
            camera.distance = 0.30
    elif args.view == "top":
        camera.lookat[:] = [0.6, -0.4, 0]
        camera.distance, camera.azimuth, camera.elevation = 1.55, 90, -89.9
    else:
        camera.lookat[:] = [0.53, -0.37, 0.18]
        camera.distance, camera.azimuth, camera.elevation = 1.95, 125, -28
        if args.setup == 2:
            config = json.loads((ROOT / "configs/exp2.json").read_text())["camera"]
            camera.lookat[:] = config["lookat"]
            camera.distance, camera.azimuth, camera.elevation = config["distance"], config["azimuth"], config["elevation"]
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    if args.render:
        from PIL import Image
        args.render.parent.mkdir(parents=True, exist_ok=True)
        with mujoco.Renderer(model, height=1000, width=1400) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=options)
            Image.fromarray(renderer.render()).save(args.render)
        print(args.render.resolve())
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth = camera.azimuth
            viewer.cam.elevation = camera.elevation
            viewer.opt.geomgroup[3] = 0
            viewer.opt.sitegroup[:] = 0
            while viewer.is_running():
                start = time.perf_counter()
                if args.step:
                    mujoco.mj_step(model, data)
                else:
                    mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(max(0, (model.opt.timestep if args.step else 1/60) - (time.perf_counter()-start)))


if __name__ == "__main__":
    main()
