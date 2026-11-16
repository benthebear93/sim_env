"""Articulated UR5e / WSG50 / two GelSight Minis / free rod assembly.

Mechanical geometry is audited before any material fitting. The original
camera calibration and centered flange mounting are preserved by default.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .replay import Session, pose_matrix
from .real2sim import numbers, quaternion, camera_times

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT/"assets/real2sim_system"
ROBOT_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k,v in attrs.items()})


def transform(position, rotation):
    value = np.eye(4)
    value[:3,3], value[:3,:3] = position, rotation
    return value


def initial_object(session, base_camera=None):
    rows = [r for r in session.rows["ArUco"] if r["data"].get("valid") and 0 <= r["t_s"] < 1]
    if len(rows) < 3:
        raise ValueError("Need valid initial marker frames")
    measured = np.array([r["data"]["pose"]["T_camera_marker"] for r in rows])
    cm = transform(np.median(measured[:,:3,3], axis=0), Rotation.from_matrix(measured[:,:3,:3]).mean().as_matrix())
    bm = np.array(session.asset_json("hardware/rod_contact_v02/parameters.json")["T_body_marker_mm"],float)
    bm[:3,3] *= .001
    return (session.base_camera() if base_camera is None else base_camera) @ cm @ np.linalg.inv(bm), bm


def sensor_mount_geometry(config, width, sign, tilt_deg):
    """Return sensor rotation and jaw/sensor translations, preserving the chosen gel center."""
    rotation = (Rotation.from_euler("z", 0 if sign == 1 else 180, degrees=True).as_matrix()
                @ Rotation.from_euler("y", tilt_deg, degrees=True).as_matrix())
    center = np.array([sign*(config["jaw_surface_gap_at_recorded_width_m"]/2+.002125), 0.,
                       -(.03625+config["carriage_extension_m"]+.041)])
    mount_point = np.array([.014375, 0., .041])
    jaw = center + rotation @ mount_point - np.array([sign*width/2, 0., 0.])
    return rotation, jaw, -rotation @ mount_point


def fit_robot_base(session, source):
    """Fit robot coordinate registration using robot feedback only, training poses only."""
    model = mujoco.MjModel.from_xml_path(str(source)); data = mujoco.MjData(model)
    sid = model.site("attachment_site").id
    moves = [r["t_s"] for r in session.rows["events"] if r["kind"] == "command_started" and r["data"].get("action") == "move"]
    split = moves[-1] if len(moves) >= 2 else session.duration*.67
    source_poses, target_poses, train = [], [], []
    for row in session.rows["UR5e"]:
        data.qpos[:] = row["data"]["joint_positions_rad"]
        mujoco.mj_forward(model,data)
        source_poses.append(transform(data.site_xpos[sid],data.site_xmat[sid].reshape(3,3)))
        target_poses.append(pose_matrix(row["data"]["tcp_pose_m_rad"]))
        train.append(row["t_s"] < split)
    source_poses, target_poses, train = np.array(source_poses),np.array(target_poses),np.array(train)
    r = Rotation.from_matrix(target_poses[train,:3,:3] @ source_poses[train,:3,:3].transpose(0,2,1)).mean().as_matrix()
    p = np.mean(target_poses[train,:3,3] - source_poses[train,:3,3] @ r.T,axis=0)
    correction = transform(p,r)
    predicted = correction @ source_poses
    position_error = np.linalg.norm(predicted[:,:3,3]-target_poses[:,:3,3],axis=1)*1000
    angle_error = np.rad2deg(Rotation.from_matrix(predicted[:,:3,:3].transpose(0,2,1) @ target_poses[:,:3,:3]).magnitude())
    report = dict(T_recorded_base_source_world=correction.tolist(), split_s=split,
                  position_rms_mm=float(np.sqrt(np.mean(position_error**2))),
                  position_max_mm=float(position_error.max()),
                  rotation_rms_deg=float(np.sqrt(np.mean(angle_error**2))),
                  method="constant root-frame registration; no object observations used",
                  note="Nominal Menagerie kinematics, not the robot's factory-calibrated DH parameters")
    return correction, report


def default_config():
    return dict(schema_version=1, timestep_s=.0005,
                camera_translation_correction_base_m=[0.,0.,0.],
                camera_correction_status="original recording calibration; unmodified",
                adapter_thickness_m=.02, adapter_radius_m=.04,
                flange_wsg_center_offset_xy_m=[0.,0.],
                flange_wsg_yaw_deg=0., carriage_extension_m=.014,
                jaw_servo_kp_N_m=200000., jaw_servo_kv_Ns_m=300.,
                jaw_force_limit_N=5.,
                jaw_surface_gap_at_recorded_width_m=.030,
                positive_sensor_tilt_deg=-3., negative_sensor_tilt_deg=0.,
                object_mass_kg=.03, finger_mass_kg=.025, sensor_mass_kg=.035,
                friction=.6, torsional_friction_m=.0001,
                normal_stiffness_s2=20000., normal_damping_s=150.,
                user_confirmed=dict(centered_direct_flange_wsg_mount=True,
                                    adapter_thickness_mm=20, material="Prusa PLA", infill_percent=5,
                                    upper_only_contact_on_observed_sensor=True),
                unmeasured=["adapter diameter and mass", "WSG yaw about flange normal",
                            "carriage extension", "surface gap versus WSG register",
                            "sensor/finger tilt", "printed object mass/COM/inertia",
                            "finger/sensor mass", "contact coefficients", "WSG force limit per-jaw convention"],
                notes=["Both fingers carry a GelSight Mini; recording is one sensor only.",
                       "14 mm carriage extension and -3 degree tilt are explicit assembly hypotheses, not measurements.",
                       "Camera calibration remains as recorded; no object-centering correction is silently applied.",
                       "5 N per-jaw actuator cap is a hypothesis from the GUI force setting; recorded GetForce is zero throughout."])


def inertia_element(body, mesh, mass):
    inertia = mesh.moment_inertia/mesh.volume*mass
    values, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:,0] *= -1
    add(body,"inertial",mass=mass,pos=numbers(mesh.center_mass),diaginertia=numbers(values),quat=numbers(quaternion(axes)))


def prepare_session_assets(session, output):
    dest = output/"assets"
    dest.mkdir(parents=True, exist_ok=True)
    for source, target in [(ASSETS/"wsg_body.obj",dest/"wsg_body.obj"),
                           (ASSETS/"gelsight/case.obj",dest/"case.obj"),
                           (ASSETS/"gelsight/back_cover.obj",dest/"back_cover.obj")]:
        shutil.copy2(source,target)
    for path in (ASSETS/"ur5e/assets").glob("*.obj"):
        shutil.copy2(path,dest/path.name)
    for source in [ASSETS/"ur5e/LICENSE",ASSETS/"gelsight/WSG_LICENSE",ASSETS/"gelsight/GELSIGHT_REPOSITORY_LICENSE"]:
        shutil.copy2(source,dest/("UR5E_LICENSE" if source.name=="LICENSE" else source.name))
    copied = {}
    for suffix,name in [("hardware/WSG50_GELSIGHT_fin.stl","finger_source.stl"),
                        ("hardware/rod_contact_v02/rod_cylinder_v02.stl","rod_source.stl")]:
        asset = next(a for a in session.manifest["assets"] if a["path"].endswith(suffix))
        path = session.file(asset["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError("Recorded CAD checksum mismatch")
        shutil.copy2(path,dest/name); copied[name] = asset
    finger = trimesh.load_mesh(dest/"finger_source.stl")
    np.testing.assert_allclose(finger.bounds,[[0,-46,0],[29,15,30]],atol=.001)
    # Current (shorter) finger pocket: x=26, y=-26, z=15 mm.
    # Match the sensor back plane x=25.875 mm; never substitute the old finfin STL.
    tf = np.eye(4)
    tf[:3,:3] = .001*np.array([[1,0,0],[0,0,-1],[0,1,0]])
    tf[:3,3] = [-.000125,.015,.026]
    finger.apply_transform(tf); finger.export(dest/"custom_finger.obj")
    rod = trimesh.load_mesh(dest/"rod_source.stl"); rod.apply_scale(.001); rod.export(dest/"rod.obj")
    cut = rod.section(plane_origin=[0,0,.07],plane_normal=[0,0,1]).vertices
    upper = trimesh.convex.convex_hull(np.vstack([rod.vertices[rod.vertices[:,2]>=.07],cut]))
    upper.export(dest/"rod_upper.obj")
    return rod, dict(recorded_assets=copied,T_sensor_finger_source=tf.tolist(),
                     finger_source_bounds_mm=[[0,-46,0],[29,15,30]],
                     finger_mount_status="inferred current-pocket mating; fasteners not measured",
                     gel_dimensions_m=[.00425,.02075,.02525],
                     upper_collision_bounds_m=upper.bounds.tolist(),
                     licenses=["UR5E_LICENSE","WSG_LICENSE","GELSIGHT_REPOSITORY_LICENSE"])


def build(session_path, output, config=None):
    session, output = Session(session_path),Path(output)
    if output.resolve().is_relative_to(session.path.resolve()):
        raise ValueError("Keep model outputs outside original recording")
    output.mkdir(parents=True,exist_ok=True)
    cfg = default_config() if config is None else copy.deepcopy(config)
    if cfg["flange_wsg_center_offset_xy_m"] != [0.,0.]:
        raise ValueError("User confirmed centered direct mount: do not fit an eccentric WSG mount")
    source = ASSETS/"ur5e/ur5e.xml"
    correction, fk_report = fit_robot_base(session, source)
    bc = session.base_camera().copy()
    bc[:3,3] += cfg.get("camera_translation_correction_base_m",[0.,0.,0.])
    object_pose, body_marker = initial_object(session,bc)
    rod, cad_report = prepare_session_assets(session,output)
    root = ET.parse(source).getroot()
    root.set("model","UR5e WSG50 dual GelSight actual-parts assembly")
    compiler = root.find("compiler"); compiler.set("meshdir","assets")
    option = root.find("option")
    option.attrib.update(dict(timestep=str(cfg["timestep_s"]),gravity="0 0 -9.81",cone="elliptic",solver="Newton",iterations="100",tolerance="1e-10"))
    option.set("noslip_iterations",str(cfg.get("noslip_iterations",0)))
    visual = add(root,"visual"); add(visual,"global",offwidth="1280",offheight="720")
    add(visual,"headlight",ambient=".5 .5 .5")
    root.remove(root.find("keyframe"))
    defaults = root.find("default")
    system = add(defaults,"default",**{"class":"system"})
    add(system,"geom",contype=0,conaffinity=0,density=0)
    add(system,"joint",limited="false",damping=0,armature=0)
    world = root.find("worldbody")
    base = world.find("body")
    original_r = Rotation.from_quat(np.fromstring(base.get("quat"),sep=" ")[[1,2,3,0]]).as_matrix()
    base.set("quat",numbers(quaternion(correction[:3,:3] @ original_r)))
    base.set("pos",numbers(correction[:3,3]))
    # Arm collisions remain available visually, but only hand/rod environment
    # contacts are enabled for this fitting scene; no self-collision shortcuts on rod.
    for geom in world.iter("geom"):
        geom.set("contype","0");geom.set("conaffinity","0")
    assets = root.find("asset")
    for name,file in [("wsg_mesh","wsg_body.obj"),("custom_finger_mesh","custom_finger.obj"),
                      ("gs_case","case.obj"),("gs_back","back_cover.obj"),("rod_visual","rod.obj"),("rod_upper","rod_upper.obj")]:
        add(assets,"mesh",name=name,file=file)
    wrist = root.find(".//body[@name='wrist_3_link']")
    site = wrist.find("site[@name='attachment_site']")
    flange = add(wrist,"body",name="ur_flange",pos=site.get("pos"),quat=site.get("quat"),childclass="system")
    add(flange,"geom",name="adapter_plate",type="cylinder",size=numbers([cfg["adapter_radius_m"],cfg["adapter_thickness_m"]/2]),
        pos=numbers([0,0,cfg["adapter_thickness_m"]/2]),rgba=".62 .65 .68 1",mass=".15")
    mount_r = Rotation.from_euler("z",cfg["flange_wsg_yaw_deg"],degrees=True).as_matrix() @ np.diag([1.,-1.,-1.])
    wsg = add(flange,"body",name="wsg50",pos=numbers([0,0,cfg["adapter_thickness_m"]+.03625]),quat=numbers(quaternion(mount_r)))
    add(wsg,"inertial",pos="0 0 0",mass=".988882",diaginertia=".00064 .0022 .00196")
    add(wsg,"geom",type="mesh",mesh="wsg_mesh",rgba=".3 .32 .35 1")
    add(wsg,"geom",name="wsg_collision",type="box",size=".073 .025 .03625",rgba="0 0 0 0",group=3,contype=4,conaffinity=1)
    width = session.rows["WSG50"][0]["data"]["opening_mm"]*.001
    actuator = root.find("actuator")
    for element in list(actuator):
        actuator.remove(element)
    for j,joint in enumerate(ROBOT_JOINTS):
        add(actuator,"motor",name=joint+"_motor",joint=joint,ctrlrange=("-150 150" if j<3 else "-28 28"),ctrllimited="true")
    # WSG opening register is an input; its unknown zero relative to custom gel
    # surfaces is a geometry parameter explicitly distinguished from joint travel.
    for sign,name,tilt in [(1,"positive",cfg["positive_sensor_tilt_deg"]),(-1,"negative",cfg["negative_sensor_tilt_deg"])]:
        r, jaw_p, sensor_p = sensor_mount_geometry(cfg,width,sign,tilt)
        jaw = add(wsg,"body",name=name+"_jaw",pos=numbers(jaw_p))
        add(jaw,"joint",name=name+"_jaw_slide",type="slide",axis=numbers([sign,0,0]),range="0 .055",limited="true",damping="4",armature=".01")
        add(jaw,"geom",type="box",size=".01 .015 .007",pos="0 0 .007",rgba=".18 .2 .23 1",mass=".05")
        sensor = add(jaw,"body",name=name+"_gelsight",pos=numbers(sensor_p),quat=numbers(quaternion(r)))
        add(sensor,"inertial",pos=".014 0 .014",mass=cfg["sensor_mass_kg"]+cfg["finger_mass_kg"],diaginertia=".00002 .00003 .00002")
        add(sensor,"geom",name=name+"_custom_finger",type="mesh",mesh="custom_finger_mesh",rgba=".12 .12 .14 1")
        add(sensor,"geom",name=name+"_case",type="mesh",mesh="gs_case",rgba=".08 .09 .1 1")
        add(sensor,"geom",type="mesh",mesh="gs_back",rgba=".14 .15 .16 1")
        add(sensor,"geom",name=name+"_gel",type="box",size=".002125 .010375 .012625",rgba=".12 .68 .65 .8",contype=2,conaffinity=1)
        add(sensor,"geom",name=name+"_housing_collision",type="box",pos=".016 0 0",size=".009875 .014 .016",rgba="0 0 0 0",group=3,contype=4,conaffinity=1)
        add(sensor,"site",name=name+"_gel_surface",pos="-.002125 0 0",size=".001",rgba="1 .3 .2 1")
        force=cfg.get("jaw_force_limit_N",5.)
        add(actuator,"position",name=name+"_jaw_drive",joint=name+"_jaw_slide",
            kp=cfg.get("jaw_servo_kp_N_m",200000.),kv=cfg.get("jaw_servo_kv_Ns_m",300.),
            forcerange=numbers([-force,force]),forcelimited="true")
    rod_body = add(world,"body",name="object",pos=numbers(object_pose[:3,3]),quat=numbers(quaternion(object_pose[:3,:3])),childclass="system")
    add(rod_body,"freejoint",name="object_free")
    inertia_element(rod_body,rod,cfg["object_mass_kg"])
    add(rod_body,"geom",type="mesh",mesh="rod_visual",rgba=".2 .2 .22 1")
    add(rod_body,"geom",name="rod_contact",type="mesh",mesh="rod_upper",rgba="0 0 0 0",group=3,contype=1,conaffinity=6)
    add(rod_body,"geom",name="plate_contact",type="box",size=".03 .002 .03",pos="0 0 .03",rgba="0 0 0 0",group=3,contype=1,conaffinity=6)
    add(rod_body,"geom",name="neck_contact",type="box",size=".009 .005 .013",pos="0 .005 .065",rgba="0 0 0 0",group=3,contype=1,conaffinity=6)
    add(rod_body,"site",name="object_marker",pos=numbers(body_marker[:3,3]),quat=numbers(quaternion(body_marker[:3,:3])),size=".002",rgba="1 .5 0 1")
    # Native marker artwork, six-by-six ArUco pattern and white paper margin.
    pattern = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),0,6)
    add(rod_body,"geom",type="box",size=".027 .0001 .027",pos="0 -.0022 .03",rgba="1 1 1 1")
    for iy,ix in zip(*np.where(pattern==0)):
        add(rod_body,"geom",type="box",size=numbers([.04/12,.00005,.04/12]),
            pos=numbers([(ix-2.5)*.04/6,-.00235,.03+(2.5-iy)*.04/6]),rgba=".01 .01 .01 1")
    contact = add(root,"contact")
    for side in ["positive","negative"]:
        add(contact,"pair",geom1="rod_contact",geom2=side+"_gel",condim="4",
            friction=numbers([cfg["friction"],cfg["friction"],cfg["torsional_friction_m"],0,0]),
            solref=numbers([-cfg["normal_stiffness_s2"],-cfg["normal_damping_s"]]),solimp=".9 .95 .001 .5 2",solreffriction=".002 1")
    camera = session.rows["ArUco"][0]["data"]["camera"]
    add(world,"camera",name="recorded_camera",pos=numbers(bc[:3,3]),quat=numbers(quaternion(bc[:3,:3]@np.diag([1.,-1.,-1.]))),
        fovy=np.rad2deg(2*np.arctan(camera["image_size"][1]/(2*camera["matrix"][1][1]))))
    add(world,"geom",name="table",type="plane",size="1 1 .01",rgba=".35 .38 .42 1",contype=4,conaffinity=1)
    tree = ET.ElementTree(root); ET.indent(tree)
    path = output/"system.xml"; tree.write(path,encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    for i,joint in enumerate(ROBOT_JOINTS):
        data.qpos[model.joint(joint).qposadr[0]] = session.rows["UR5e"][0]["data"]["joint_positions_rad"][i]
    for side in ["positive","negative"]:
        data.qpos[model.joint(side+"_jaw_slide").qposadr[0]] = width/2
    mujoco.mj_forward(model,data)
    q_initial = data.qpos.copy()
    keyframe = add(root,"keyframe");add(keyframe,"key",name="recorded_initial",qpos=numbers(q_initial))
    ET.indent(tree);tree.write(path,encoding="unicode")
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
    audit = assembly_audit(model,data,session,cfg)
    report = dict(model="full articulated robot + gripper + two sensor/finger assemblies + free object",
                  stage="geometry audit before material fitting",config=cfg,robot_registration=fk_report,cad=cad_report,
                  assembly_audit=audit,T_base_camera=bc.tolist(),initial_qpos=q_initial.tolist(),
                  no_object_constraints=bool(model.neq==0), nq=model.nq,nv=model.nv,nu=model.nu,
                  parameter_fit_started=False)
    (output/"assembly.json").write_text(json.dumps(report,indent=2)+"\n")
    (output/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    return model,data,report


def assembly_audit(model,data,session,config):
    oid=model.body("object").id
    contact_rows=[]
    for c in data.contact:
        contact_rows.append(dict(geom1=model.geom(c.geom1).name,geom2=model.geom(c.geom2).name,
                                 distance_mm=float(c.dist*1000),position_base_m=c.pos.tolist()))
    flange=model.body("ur_flange").id;wsg=model.body("wsg50").id
    rod_axis=data.xpos[oid]+data.xmat[oid].reshape(3,3)@np.array([0,0,.14])
    wg=transform(data.xpos[wsg],data.xmat[wsg].reshape(3,3))
    point=(np.linalg.inv(wg)@np.r_[rod_axis,1])[:3]
    distances={side:float(mujoco.mj_geomDistance(model,data,model.geom("rod_contact").id,model.geom(side+"_gel").id,.1,None))*1000 for side in ["positive","negative"]}
    initial_penetration=min([r["distance_mm"] for r in contact_rows]+[0])
    both=all(distances[s]<0 for s in distances)
    severe=initial_penetration < -2
    return dict(flange_position_base_m=data.xpos[flange].tolist(),
                rod_axis_at_z140mm_in_wsg_m=point.tolist(),
                pad_rod_signed_distances_mm=distances,initial_contacts=contact_rows,
                bilateral_contact=both,excessive_penetration=severe,
                ready_for_contact_fit=bool(both and not severe),
                note="A missing pad contact or >2 mm penetration blocks friction fitting; resolve assembly/calibration first.")


def render_assembly(model,data,session_path,output):
    from .frame_io import load_frame
    session=Session(session_path);output=Path(output)
    renderer=mujoco.Renderer(model,height=720,width=1280)
    opt=mujoco.MjvOption();opt.geomgroup[3]=0
    try:
        renderer.update_scene(data,camera="recorded_camera",scene_option=opt)
        sim=renderer.render().copy();cv2.imwrite(str(output/"camera_view.png"),cv2.cvtColor(sim,cv2.COLOR_RGB2BGR))
        row=next(r for r in session.rows["ArUco"] if r["t_s"]>=0)
        real=load_frame(session.file(row["data"]["raw_image"]))
        panels=np.hstack([real,cv2.cvtColor(sim,cv2.COLOR_RGB2BGR)])
        cv2.imwrite(str(output/"real_and_system.jpg"),panels)
        camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:]=[.15,-.25,.3];camera.distance=1.3;camera.azimuth=135;camera.elevation=-20
        renderer.update_scene(data,camera=camera,scene_option=opt)
        cv2.imwrite(str(output/"whole_system.png"),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
        camera.lookat[:]=data.xpos[model.body("wsg50").id]+[0,0,-.06]
        camera.distance=.46;camera.azimuth=55;camera.elevation=-5
        renderer.update_scene(data,camera=camera,scene_option=opt)
        cv2.imwrite(str(output/"gripper_closeup.png"),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
    finally:
        renderer.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("session",type=Path);p.add_argument("--output",type=Path)
    p.add_argument("--config",type=Path);p.add_argument("--render",action="store_true")
    a=p.parse_args(argv)
    output=a.output or ROOT/"output/full_real2sim"/a.session.name
    cfg=json.loads(a.config.read_text()) if a.config else None
    model,data,report=build(a.session,output,cfg)
    if a.render:render_assembly(model,data,a.session,output)
    print(json.dumps(dict(output=str(output),robot=report["robot_registration"],audit=report["assembly_audit"]),indent=2))


if __name__=="__main__":main()
