from argparse import ArgumentParser
import importlib.util
from pathlib import Path
import threading
import time
import tkinter as tk

import mujoco
import mujoco.viewer
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import torch
from torch import nn
import zarr


BASE = Path(__file__).resolve().parent
REPOSITORY = BASE.parents[1]
MOVEMENT_DATA = REPOSITORY / "data" / "movement_data" / "Movement_Data"
VALIDATION_DATA = REPOSITORY / "data" / "validation" / "Validation"
GOAL_HZ = 40
TARGET_INTERVAL_STEPS = 1
OMEGA = 60.0
ZETA = 0.5
USE_SAVED_GAINS = True
BINO_SMOOTH = (5, 5, 5, 0, 0)  # t-2, t-1, t, t+1, t+2
VEL_ORG = True
STD_MULTIPLIER = 1
RESIDUAL_STD = REPOSITORY / "configs" / "Residual_Std.npy"
CHECKPOINT = REPOSITORY / "models" / "Goal_Refinement_v5_batch1480.pt"
NORMALIZATION = REPOSITORY / "configs" / "Goal_Refinement_Normalization.npz"
CHECKPOINT_SMOOTH = 15
_CHECKPOINT_CONTEXT = None
_PLANNER_CONTEXT = None
REGIONS = (
    tuple(f"right_{x}" for x in ("hip_flex", "hip_abd", "hip_rot", "knee_flex", "ankle_flex", "ankle_inv", "toes_flex")),
    tuple(f"left_{x}" for x in ("hip_flex", "hip_abd", "hip_rot", "knee_flex", "ankle_flex", "ankle_inv", "toes_flex")),
    tuple(f"{part}_{axis}" for part in ("lumbar_lower", "lumbar_upper", "thoracic_lower", "thoracic_upper") for axis in ("flex", "lat", "axial")),
    ("right_shoulder_flex", "right_shoulder_abd", "right_shoulder_rot", "right_elbow_flex", "right_forearm_roll", "right_wrist_flex", "right_wrist_dev", "right_thumb_flex", "right_fingers_flex"),
    ("left_shoulder_flex", "left_shoulder_abd", "left_shoulder_rot", "left_elbow_flex", "left_forearm_roll", "left_wrist_flex", "left_wrist_dev", "left_thumb_flex", "left_fingers_flex"),
    ("neck_flex", "neck_lat", "neck_axial"))

if len(BINO_SMOOTH) != 5 or sum(BINO_SMOOTH) == 0:
    raise ValueError("BINO_SMOOTH must contain five weights with a nonzero sum")


def make_spline(qpos, joint_qpos, velocity=None):
    angles = qpos[:, joint_qpos]
    if velocity is None:
        velocity = np.empty_like(angles)
        velocity[1:-1] = GOAL_HZ / 2 * (angles[2:] - angles[:-2])
        velocity[0] = GOAL_HZ * (angles[1] - angles[0])
        velocity[-1] = GOAL_HZ * (angles[-1] - angles[-2])
    return CubicHermiteSpline(np.arange(len(angles)) / GOAL_HZ, angles, velocity)


def show_target(model, data, scene):
    scene.ngeom = 0
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], model.geom_type[i], model.geom_size[i],
                            data.geom_xpos[i], data.geom_xmat[i],
                            np.array((0.1, 0.3, 1.0, 0.35), dtype=np.float32))
        scene.ngeom += 1


def alter_goals(qpos, start, count, joint_qpos, ranges, std, rng, correction, accumulate):
    altered, original = qpos.copy(), qpos[start:start + count].copy()
    noise = rng.normal(size=(count, len(std)-3)) * std[3:] * STD_MULTIPLIER
    if accumulate:
        noise = np.cumsum(noise, axis=0)
    else:
        noise[[0, -1]] = 0
    padded = np.pad(noise, ((2, 2), (0, 0)))
    noise = sum(weight*padded[i:i+count] for i, weight in enumerate(BINO_SMOOTH)) / sum(BINO_SMOOTH)
    if accumulate:
        noise *= np.sin(np.linspace(0, np.pi, count))[:, None]
    noise[[0, -1]] = 0
    original[:, joint_qpos] = np.clip(original[:, joint_qpos] + noise, ranges[:, 0], ranges[:, 1])
    original = correction._collision_correct(original)
    original[[0, -1]] = qpos[[start, start + count - 1]]
    altered[start:start + count] = original
    return altered


class CheckpointActor(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.current = nn.Sequential(nn.Linear(476, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        self.goal = nn.Sequential(nn.Linear(263, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU())
        self.future = nn.LSTM(64, 256, batch_first=True)
        self.main = nn.LSTM(576, 512, 2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, output))
        self.register_buffer("action_std", torch.empty(output))

    def step(self, current, goal, future, hidden):
        value = torch.cat((self.current(current), future, goal), -1)[:, None]
        output, hidden = self.main(value, hidden)
        return self.head(output[:, 0]), hidden


def checkpoint_targets(model, live, target_data, frames_qpos, frames_qvel, joint_qpos,
                       joint_dofs, names, kp, kd):
    global _CHECKPOINT_CONTEXT
    if _CHECKPOINT_CONTEXT is None:
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        if saved.get("schema") != "goal_residual_online_v5": raise ValueError("Checkpoint schema is incompatible with this viewer")
        region = int(saved["exploring_region"])
        if region not in range(1, len(REGIONS) + 1): raise ValueError("Checkpoint has an invalid exploring region")
        policy_names = sum(REGIONS[:region], ())
        try: policy_joint = np.array([np.flatnonzero(names == name).item() for name in policy_names])
        except ValueError as error: raise ValueError("Checkpoint exploring-region names do not match actuator order") from error
        if len(np.unique(policy_joint)) != len(policy_joint): raise ValueError("Checkpoint exploring regions contain duplicate actuators")
        actor = CheckpointActor(2 * len(policy_joint)); actor.load_state_dict(saved["actor"], strict=True); actor.eval()
        with np.load(NORMALIZATION) as norm:
            if int(norm["schema"]) != 3 or not np.array_equal(norm["kp"], kp) or not np.array_equal(norm["kd"], kd):
                raise ValueError("Checkpoint normalization is incompatible with the PD gains")
            cm, cs, gm, gs = [norm[key].astype(np.float32) for key in ("current_mean", "current_std", "goal_mean", "goal_std")]
        if [x.shape for x in (cm, cs, gm, gs)] != [(476,), (476,), (263,), (263,)] or np.any(cs <= 0) or np.any(gs <= 0):
            raise ValueError("Malformed checkpoint normalization")
        _CHECKPOINT_CONTEXT = actor, policy_joint, cm, cs, gm, gs
    actor, policy_joint, cm, cs, gm, gs = _CHECKPOINT_CONTEXT

    horizon = len(frames_qpos) - 1
    qpos, qvel = frames_qpos[:, :model.nq], frames_qvel[:, :model.nv]
    bodies = frames_qpos[:, model.nq:].reshape(horizon + 1, model.nbody - 1, 3)
    body_velocity = frames_qvel[:, model.nv:].reshape(horizon + 1, model.nbody - 1, 3)
    mass = model.body_mass[1:].copy(); mass /= mass.sum()
    floor = model.geom("floor").id
    feet = tuple(model.geom(f"{side}_{part}").id for side in ("left", "right") for part in ("foot", "toes_chunk"))

    def contacts(data):
        flags = np.zeros(2, np.float32)
        for contact in data.contact:
            a, b = contact.geom1, contact.geom2
            if (a == floor and b in feet[:2]) or (b == floor and a in feet[:2]): flags[0] = 1
            if (a == floor and b in feet[2:]) or (b == floor and a in feet[2:]): flags[1] = 1
        return flags

    def state(p, v, xyz, xyz_velocity, foot_contact):
        w, x, y, z = p[3:7]
        rotation = np.array((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y),
                             2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)))
        relative = xyz.copy(); relative[:, :2] -= p[:2]
        com = (xyz * mass[:, None]).sum(0); com_velocity = (xyz_velocity * mass[:, None]).sum(0)
        return np.r_[p[:2]-qpos[0, :2], p[2], relative[:, :2].ravel(), xyz[:, 2], v[:3],
                     (xyz_velocity-v[:3]).ravel(), rotation, v[3:6], p[joint_qpos], v[joint_dofs],
                     foot_contact, com-p[:3], com_velocity]

    goal_rows = []
    for frame in range(1, horizon + 1):
        target_data.qpos[:], target_data.qvel[:] = qpos[frame], qvel[frame]
        mujoco.mj_forward(model, target_data)
        goal_rows.append(np.r_[state(qpos[frame], qvel[frame], bodies[frame], body_velocity[frame], contacts(target_data)),
                               (horizon-frame)/horizon, frame/horizon])
    goals = torch.from_numpy((np.asarray(goal_rows, np.float32)-gm)/gs)[None]
    with torch.inference_mode():
        embedded = actor.goal(goals)
        reverse_future, _ = actor.future(torch.flip(embedded, (1,)))
        future = torch.flip(reverse_future, (1,))

    mujoco.mj_resetData(model, live); live.qpos[:], live.qvel[:] = qpos[0], qvel[0]
    mujoco.mj_forward(model, live)
    altered, altered_velocity = qpos.copy(), qvel.copy()
    previous = np.zeros(2*len(names), np.float32)
    history = np.zeros((CHECKPOINT_SMOOTH, 2*len(policy_joint)), np.float32)
    mean_carry = np.zeros(2*len(policy_joint), np.float32)
    previous_body = live.xipos[1:].copy(); current_body_velocity = body_velocity[0].copy()
    hidden = None; mass_matrix = np.empty((model.nv, model.nv)); dt = 1/GOAL_HZ
    limits = model.jnt_range[model.actuator_trnid[:, 0]]
    with torch.inference_mode():
        for t in range(horizon):
            live_xyz = live.xipos[1:].copy()
            raw = np.r_[state(live.qpos, live.qvel, live_xyz, current_body_velocity, contacts(live)),
                        (live_xyz-bodies[t]).ravel(), (horizon-t)/horizon, t/horizon,
                        live.qfrc_bias[joint_dofs]-live.qfrc_passive[joint_dofs], previous]
            current = torch.from_numpy(((raw.astype(np.float32)-cm)/cs))[None]
            increment, hidden = actor.step(current, embedded[:, t], future[:, t], hidden)
            mean_carry += increment[0].numpy(); raw_action = mean_carry.copy(); forced = t == horizon-1
            if forced: raw_action[:] = 0
            history[:-1] = history[1:]; history[-1] = raw_action
            filtered = history.mean(0) * np.sin(np.pi*t/max(horizon-1, 1))
            if forced: filtered[:] = 0
            pos, vel = np.split(filtered, 2)
            position = np.zeros(len(names), np.float32); velocity = np.zeros(len(names), np.float32)
            position[policy_joint], velocity[policy_joint] = pos, vel
            p1 = np.clip(qpos[t+1, joint_qpos] + position, limits[:, 0], limits[:, 1])
            effective = np.r_[p1-qpos[t+1, joint_qpos], velocity].astype(np.float32)
            altered[t+1, joint_qpos] = p1
            altered_velocity[t+1, joint_dofs] += velocity
            p0 = qpos[t, joint_qpos] + previous[:len(names)]
            v0 = qvel[t, joint_dofs] + previous[len(names):]
            v1 = qvel[t+1, joint_dofs] + velocity
            for substep in range(1, round(1/(GOAL_HZ*model.opt.timestep))+1):
                u = substep * model.opt.timestep / dt; u2, u3 = u*u, u*u*u
                target = ((2*u3-3*u2+1)*p0 + (u3-2*u2+u)*dt*v0 + (-2*u3+3*u2)*p1 + (u3-u2)*dt*v1)
                target_velocity = ((6*u2-6*u)/dt*p0 + (3*u2-4*u+1)*v0 + (-6*u2+6*u)/dt*p1 + (3*u2-2*u)*v1)
                acceleration = ((12*u-6)*(p0-p1)/dt**2 + (6*u-4)*v0/dt + (6*u-2)*v1/dt)
                mujoco.mj_forward(model, live); mujoco.mj_fullM(model, live, mass_matrix)
                live.ctrl[:] = (live.qfrc_bias[joint_dofs]-live.qfrc_passive[joint_dofs] + kp*target + kd*target_velocity
                                + mass_matrix[np.ix_(joint_dofs, joint_dofs)] @ acceleration)
                mujoco.mj_step(model, live)
            mujoco.mj_forward(model, live)
            current_xyz = live.xipos[1:].copy(); current_body_velocity = (current_xyz-previous_body)*GOAL_HZ
            previous_body, previous = current_xyz, effective
    return altered, altered_velocity


def planner_targets(model, target_data, frames_qpos, frames_qvel, joint_qpos, joint_dofs, names):
    global _PLANNER_CONTEXT
    if _PLANNER_CONTEXT is None:
        spec = importlib.util.spec_from_file_location("goal_planner", REPOSITORY / "src" / "trajectory" / "Goal_Planner.py")
        planner = importlib.util.module_from_spec(spec); spec.loader.exec_module(planner)
        saved = torch.load(planner.CHECKPOINT, map_location="cpu", weights_only=False)
        if saved.get("schema") != 7: raise ValueError("Goal-planner checkpoint is incompatible")
        weights = saved["model"]; planner.ENCODER_WIDTH = weights["encoder.0.weight"].shape[0]
        planner.DECODER_WIDTH = weights["decoder.weight_hh_l0"].shape[1]
        with np.load(planner.NORMALIZATION) as norm:
            if (int(norm["schema"]) != 7 or int(norm["smooth"]) != planner.SMOOTH
                    or not np.array_equal(norm["joint_names"], names)):
                raise ValueError("Goal-planner normalization is incompatible with the viewer")
            stats = {key:norm[key].astype(np.float32) for key in ("state_mean", "state_std", "path_mean",
                "path_std", "duration_mean", "duration_std", "time_mean", "time_std")}
        if ([stats[x].shape for x in stats] != [(planner.STATE_SIZE,), (planner.STATE_SIZE,),
                (planner.PATH_SIZE,), (planner.PATH_SIZE,), (1,), (1,), (2,), (2,)]
                or np.any(stats["state_std"] <= 0) or np.any(stats["path_std"] <= 0)
                or np.any(stats["duration_std"] <= 0) or np.any(stats["time_std"] <= 0)):
            raise ValueError("Malformed goal-planner normalization")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor = planner.Planner(stats).to(device); actor.load_state_dict(weights, strict=True); actor.eval()
        _PLANNER_CONTEXT = planner, actor, stats, device
    planner, actor, stats, device = _PLANNER_CONTEXT
    horizon = len(frames_qpos)-1
    if not 1 <= horizon <= planner.MAX_STEPS+1: raise ValueError("Planner duration is outside its supported range")
    if (frames_qpos.shape[1] != model.nq+3*(model.nbody-1)
            or frames_qvel.shape[1] != model.nv+3*(model.nbody-1)):
        raise ValueError("Movement body data is incompatible with the planner")
    qpos, qvel = frames_qpos[:, :model.nq], frames_qvel[:, :model.nv]
    if horizon == 1: return qpos.copy()
    bodies = frames_qpos[:, model.nq:].reshape(horizon+1, model.nbody-1, 3)
    body_velocity = frames_qvel[:, model.nv:].reshape(horizon+1, model.nbody-1, 3)
    mass = model.body_mass[1:].copy(); mass /= mass.sum()
    floor = model.geom("floor").id
    feet = tuple(model.geom(f"{side}_{part}").id for side in ("left", "right") for part in ("foot", "toes_chunk"))

    def state(frame):
        target_data.qpos[:], target_data.qvel[:] = qpos[frame], qvel[frame]; mujoco.mj_forward(model, target_data)
        contact = np.zeros(2, np.float32)
        for item in target_data.contact:
            for side, geoms in enumerate((feet[:2], feet[2:])):
                if (item.geom1 == floor and item.geom2 in geoms) or (item.geom2 == floor and item.geom1 in geoms): contact[side] = 1
        p, v, xyz, xyz_velocity = qpos[frame], qvel[frame], bodies[frame], body_velocity[frame]
        relative = xyz.copy(); relative[:, :2] -= p[:2]; quaternion = p[3:7]/max(np.linalg.norm(p[3:7]), 1e-12)
        if quaternion[0] < 0: quaternion = -quaternion
        com, com_velocity = (xyz*mass[:, None]).sum(0), (xyz_velocity*mass[:, None]).sum(0)
        return np.r_[p[:2]-qpos[0, :2], p[2], relative[:, :2].ravel(), xyz[:, 2], v[:3],
                     (xyz_velocity-v[:3]).ravel(), quaternion, v[3:6], p[joint_qpos], v[joint_dofs],
                     contact, com-p[:3], com_velocity].astype(np.float32)

    path = np.c_[qpos[:, :3], qpos[:, 3:7], qpos[:, joint_qpos]].astype(np.float32)
    path[:, :2] -= path[0, :2]; quaternion = path[:, 3:7]
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-12)
    if quaternion[0, 0] < 0: quaternion *= -1
    quaternion[1:] *= np.cumprod(np.where((quaternion[1:]*quaternion[:-1]).sum(1) < 0, -1., 1.))[:, None]
    sm, ss, pm, ps = (stats[x] for x in ("state_mean", "state_std", "path_mean", "path_std"))
    step = np.arange(1, planner.MAX_STEPS+1); raw_time = np.stack((np.maximum(horizon-step, 0)/horizon, step/horizon), -1)
    inputs = ((state(0)-sm)/ss, (state(horizon)-sm)/ss,
              (np.array((horizon/GOAL_HZ,), np.float32)-stats["duration_mean"])/stats["duration_std"],
              (path[0]-pm)/ps, (path[-1]-pm)/ps, (raw_time-stats["time_mean"])/stats["time_std"])
    tensors = [torch.from_numpy(x[None].astype(np.float32)).to(device) for x in inputs]
    with torch.no_grad(): predicted = (actor(*tensors)[0].cpu().numpy()*ps+pm)[:horizon-1]
    result = qpos.copy(); result[1:horizon, :3] = predicted[:, :3]; result[1:horizon, :2] += qpos[0, :2]
    result[1:horizon, 3:7] = predicted[:, 3:7]/np.maximum(np.linalg.norm(predicted[:, 3:7], axis=1, keepdims=True), 1e-12)
    result[1:horizon, joint_qpos] = predicted[:, 7:]
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument("--duration", type=float, default=1.0)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--target_only", action="store_true")
    modes.add_argument("--checkpoint_target_only", action="store_true")
    modes.add_argument("--checkpoint_compare", action="store_true")
    modes.add_argument("--checkpoint_pd", action="store_true")
    modes.add_argument("--planner_compare", action="store_true")
    modes.add_argument("--planner_validation", action="store_true")
    modes.add_argument("--neutral", action="store_true")
    modes.add_argument("--compare", action="store_true")
    modes.add_argument("--altered", action="store_true")
    modes.add_argument("--accumulate", action="store_true")
    modes.add_argument("--compare_accumulate", action="store_true")
    modes.add_argument("--kinematic", action="store_true")
    parser.add_argument("--specific", nargs=3, metavar=("CATEGORY", "SCENE", "START"))
    parser.add_argument("--slowdown", type=float, default=1.0)
    args = parser.parse_args()
    accumulated = args.accumulate or args.compare_accumulate
    compared = args.compare or args.compare_accumulate
    altered_mode = args.altered or accumulated
    checkpoint_mode = args.checkpoint_target_only or args.checkpoint_compare or args.checkpoint_pd
    validation_mode = args.planner_validation
    planner_mode = args.planner_compare or validation_mode
    if args.neutral and args.specific:
        parser.error("--neutral cannot be combined with --specific")
    if validation_mode and args.specific:
        parser.error("--planner_validation uses the stored validation windows, not --specific")
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be positive and finite")
    if not np.isfinite(args.slowdown) or args.slowdown < 1:
        parser.error("--slowdown must be finite and at least 1")

    model = mujoco.MjModel.from_xml_path(str(REPOSITORY / "configs" / "human_body_47.xml"))
    if args.kinematic:
        spec = mujoco.MjSpec.from_file(str(REPOSITORY / "configs" / "human_body_47.xml"))
        for i, joint in enumerate(model.actuator_trnid[:, 0]):
            spec.add_equality(name=f"kinematic_{i}", type=mujoco.mjtEq.mjEQ_JOINT,
                              name1=model.joint(int(joint)).name, data=[0]*11, solref=(.005, 1))
        model = spec.compile()
    data = mujoco.MjData(model)
    if args.neutral:
        mujoco.mj_forward(model, data)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = data.subtree_com[1]
            while viewer.is_running():
                viewer.sync()
                time.sleep(args.slowdown * model.opt.timestep)
        return
    target_data = mujoco.MjData(model)
    goal_steps = round(1 / (GOAL_HZ * model.opt.timestep))
    if (not np.isclose(goal_steps*model.opt.timestep, 1/GOAL_HZ)
            or (not planner_mode and (not isinstance(TARGET_INTERVAL_STEPS, int)
                or TARGET_INTERVAL_STEPS <= 0 or goal_steps % TARGET_INTERVAL_STEPS))):
        raise ValueError("TARGET_INTERVAL_STEPS must divide the MuJoCo steps per 40 Hz frame")
    if not planner_mode and not USE_SAVED_GAINS and (not np.isfinite(OMEGA) or OMEGA <= 0
                                                     or not np.isfinite(ZETA) or ZETA < 0):
        raise ValueError("OMEGA must be positive and ZETA must be nonnegative")
    joints = model.actuator_trnid[:, 0]
    joint_qpos, joint_dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    if (checkpoint_mode or planner_mode) and (model.nq, model.nv, model.nu, model.nbody) != (54, 53, 47, 25):
        raise ValueError("Checkpoint model dimensions do not match the viewer")
    if args.kinematic and (model.neq != len(joints)
            or not np.array_equal(model.eq_obj1id, joints)
            or not np.all(model.eq_type == mujoco.mjtEq.mjEQ_JOINT)):
        raise ValueError("Kinematic constraint joint indexing does not match the actuators")
    correction = residual_std = None
    if altered_mode or compared:
        residual_std = np.load(RESIDUAL_STD)
        if (residual_std.shape != (3 + len(joints),)
                or not np.all(np.isfinite(residual_std) & (residual_std > 0))
                or not np.isfinite(STD_MULTIPLIER) or STD_MULTIPLIER < 0):
            raise ValueError("Residual_Std.npy is incompatible with the model")
        spec = importlib.util.spec_from_file_location("floor_collision", REPOSITORY / "src" / "data_processing" / "Frame_interpolation+Floor_Correction.py")
        correction = importlib.util.module_from_spec(spec); spec.loader.exec_module(correction); correction._worker_init()
        if not np.array_equal(np.sort(correction.HINGE_QPOS), np.sort(joint_qpos)):
            raise ValueError("Collision correction joint indexing does not match the viewer")
    names = np.array([model.joint(int(joint)).name for joint in joints])
    if not planner_mode:
        gains = np.load(REPOSITORY / "configs" / "PD_Gains.npz")
        if not np.array_equal(gains["joint_names"], names): raise ValueError("PD gain joint indexing does not match the model")
        inertia = gains["inertia"]
        if (inertia.shape != names.shape or not np.all(np.isfinite(inertia) & (inertia > 0))
                or not np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                or not np.allclose(model.actuator_gear[:, 0], 1) or not np.allclose(model.actuator_gear[:, 1:], 0)
                or not np.all(model.actuator_ctrllimited)):
            raise ValueError("PD gains or actuator indexing are incompatible with the model")
        kp, kd = ((gains["kp"], gains["kd"]) if USE_SAVED_GAINS else
                  (inertia*OMEGA**2, 2*ZETA*OMEGA*inertia))
        if checkpoint_mode: kp, kd = kp.astype("f4"), kd.astype("f4")
        if kp.shape != names.shape or kd.shape != names.shape or not np.all(np.isfinite(kp) & (kp >= 0)) or not np.all(np.isfinite(kd) & (kd >= 0)):
            raise ValueError("PD gains are incompatible with the model")
        mass_matrix = np.empty((model.nv, model.nv)); model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        limits = model.actuator_ctrlrange.copy()
        if args.kinematic: model.actuator_gainprm[:, 0] = 0; model.actuator_biasprm[:] = 0
        else:
            model.actuator_ctrllimited[:] = 0; model.actuator_forcelimited[:] = 1; model.actuator_forcerange[:] = limits
            model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE; model.actuator_biasprm[:] = 0
            model.actuator_biasprm[:, 1], model.actuator_biasprm[:, 2] = -kp, -kd

    root = zarr.open_group(MOVEMENT_DATA, mode="r")
    span = round(args.duration*GOAL_HZ) if planner_mode else max(1, round(args.duration/model.opt.timestep))*model.opt.timestep*GOAL_HZ
    steps = int(span*goal_steps) if planner_mode else max(1, round(args.duration/model.opt.timestep))
    records, counts = [], []
    if validation_mode:
        validation = np.load(VALIDATION_DATA / "Long_Term_Validation_Positions.npy")
        with np.load(VALIDATION_DATA / "Standing_Long_Term_Validation.npz") as saved:
            standing, standing_steps = saved["endpoints"][:, 0].copy(), saved["total_physics_steps"].copy()
        standing_horizons = np.ceil(standing_steps/goal_steps).astype(int)
        if (validation.shape != (2400, 4) or standing.shape != (600, model.nq+model.nv+6*(model.nbody-1))
                or standing_steps.shape != (600,) or np.any(standing_steps <= 0)):
            raise ValueError("Planner validation files are malformed")
    else:
        for category in sorted(root.group_keys()):
            for scene in sorted(root[category].group_keys(), key=int):
                count = int(np.floor(root[category][scene]["qpos"].shape[0] - span))
                if count > 0: records.append((category, scene)); counts.append(count)
        cumulative = np.cumsum(counts)
        if not len(cumulative): raise ValueError("--duration exceeds every movement scene")
    specific = None
    if args.specific:
        category, scene, start = args.specific
        try:
            start = int(start)
            count = int(np.floor(root[category][scene]["qpos"].shape[0] - span))
        except (KeyError, ValueError):
            parser.error("--specific category, scene, or integer start is invalid")
        if not 0 <= start < count:
            parser.error("--specific start plus --duration exceeds the scene")
        specific = category, scene, start
    rng = np.random.default_rng()
    paused = threading.Event()

    def controls():
        root = tk.Tk()
        root.title("Simulation")
        button = tk.Button(root, text="Stop", width=8)
        button.config(command=lambda: (paused.clear(), button.config(text="Stop"))
                      if paused.is_set() else (paused.set(), button.config(text="Start")))
        button.pack()
        root.protocol("WM_DELETE_WINDOW", lambda: (paused.clear(), root.destroy()))
        root.mainloop()

    threading.Thread(target=controls, daemon=True).start()

    with mujoco.viewer.launch_passive(model, target_data if args.checkpoint_target_only else data) as viewer:
        while viewer.is_running():
            if paused.is_set():
                viewer.sync()
                time.sleep(0.01)
                continue
            if validation_mode:
                selected = int(rng.integers(3000))
                if selected < len(validation):
                    code, scene, start, end = map(int, validation[selected])
                    category = {1:"Common_actions", 2:"Extreme_unusual_actions", 3:"Walking_Running", 4:"Jumping"}[code]
                    group = root[category][str(scene)]; stored_qpos, stored_qvel = np.asarray(group["qpos"]), np.asarray(group["qvel"])
                    if start < 0 or end >= len(stored_qpos) or end <= start: raise ValueError("Invalid movement validation window")
                    count, steps, label = end-start+1, (end-start)*goal_steps, f"{category} {scene} {start}-{end}"
                else:
                    index = selected-len(validation); frame, horizon = standing[index], standing_horizons[index]
                    split = model.nq+3*(model.nbody-1)
                    stored_qpos = np.repeat(frame[None, :split], horizon+1, 0)
                    stored_qvel = np.repeat(frame[None, split:], horizon+1, 0)
                    qpos, start, count, steps = stored_qpos[:, :model.nq], 0, horizon+1, horizon*goal_steps
                    label = f"Standing validation {index} duration={horizon/GOAL_HZ:g}s"
            elif specific:
                category, scene, start = specific
            else:
                selected = int(rng.integers(cumulative[-1]))
                index = int(np.searchsorted(cumulative, selected, side="right"))
                start = selected - (cumulative[index - 1] if index else 0)
                category, scene = records[index]
            if not validation_mode:
                group = root[category][scene]
                stored_qpos, stored_qvel = np.asarray(group["qpos"]), np.asarray(group["qvel"])
                qpos = stored_qpos[:, :model.nq]
                count = int(np.ceil(span-1e-12))+1
            elif selected < len(validation):
                qpos = stored_qpos[:, :model.nq]
            if planner_mode:
                altered = qpos.copy(); altered[start:start+count] = planner_targets(model, target_data,
                    stored_qpos[start:start+count], stored_qvel[start:start+count], joint_qpos, joint_dofs, names)
            elif checkpoint_mode:
                altered, altered_qvel = qpos.copy(), stored_qvel[:, :model.nv].copy()
                target_qpos, target_qvel = checkpoint_targets(model, data, target_data,
                    stored_qpos[start:start+count], stored_qvel[start:start+count],
                    joint_qpos, joint_dofs, names, kp, kd)
                altered[start:start+count], altered_qvel[start:start+count] = target_qpos, target_qvel
            else:
                altered = (alter_goals(qpos, start, count, joint_qpos,
                                   model.jnt_range[joints], residual_std, rng, correction, accumulated)
                           if altered_mode or compared else qpos)
            mujoco.mj_resetData(model, data)
            if not planner_mode and not args.target_only and not compared and not args.checkpoint_target_only and not args.checkpoint_compare:
                qvel = stored_qvel[:, :model.nv]
                control_qpos = altered if altered_mode or args.checkpoint_pd else qpos
                spline = make_spline(control_qpos, joint_qpos,
                                     altered_qvel[:, joint_dofs] if args.checkpoint_pd else
                                     qvel[:, joint_dofs] if VEL_ORG else None)
                sample_steps = np.minimum(
                    np.arange(TARGET_INTERVAL_STEPS, steps + TARGET_INTERVAL_STEPS,
                              TARGET_INTERVAL_STEPS), steps)
                sample_times = start / GOAL_HZ + sample_steps * model.opt.timestep
                desired = np.repeat(spline(sample_times), TARGET_INTERVAL_STEPS, axis=0)[:steps]
                desired_velocity = np.repeat(
                    spline(sample_times, 1), TARGET_INTERVAL_STEPS, axis=0)[:steps]
                if args.checkpoint_pd:
                    desired_acceleration = np.repeat(spline(sample_times, 2), TARGET_INTERVAL_STEPS, axis=0)[:steps]
                else:
                    transition = np.arange(start, start + (steps + goal_steps - 1) // goal_steps)
                    dt = 1 / GOAL_HZ
                    knot_velocity = qvel[transition][:, joint_dofs] if VEL_ORG else spline(transition / GOAL_HZ, 1)
                    acceleration = 2 * (control_qpos[transition + 1][:, joint_qpos]
                                        - control_qpos[transition][:, joint_qpos]
                                        - knot_velocity * dt) / dt ** 2
                    desired_acceleration = np.repeat(acceleration, goal_steps, axis=0)[:steps]
                data.qpos[:], data.qvel[:] = qpos[start], qvel[start]
            print(label if validation_mode else f"{category} {scene} {start}", flush=True)
            for step in range(1, steps + 1):
                while viewer.is_running() and paused.is_set():
                    viewer.sync()
                    time.sleep(0.01)
                if not viewer.is_running():
                    return
                begin = time.perf_counter()
                goal = start + (step + goal_steps - 1) // goal_steps
                if planner_mode:
                    data.qpos[:], data.qvel[:] = qpos[goal], 0
                    target_data.qpos[:], target_data.qvel[:] = altered[goal], 0
                    mujoco.mj_forward(model, data); mujoco.mj_forward(model, target_data)
                    show_target(model, target_data, viewer.user_scn); viewer.cam.lookat[:] = target_data.subtree_com[1]
                elif args.checkpoint_target_only:
                    target_data.qpos[:], target_data.qvel[:] = altered[goal], 0
                    mujoco.mj_forward(model, target_data)
                    viewer.user_scn.ngeom = 0
                    viewer.cam.lookat[:] = target_data.subtree_com[1]
                elif args.checkpoint_compare:
                    data.qpos[:], data.qvel[:] = qpos[goal], 0
                    target_data.qpos[:], target_data.qvel[:] = altered[goal], 0
                    mujoco.mj_forward(model, data); mujoco.mj_forward(model, target_data)
                    show_target(model, target_data, viewer.user_scn)
                    viewer.cam.lookat[:] = target_data.subtree_com[1]
                elif args.target_only:
                    data.qpos[:], data.qvel[:] = qpos[goal], 0
                    mujoco.mj_forward(model, data)
                    viewer.user_scn.ngeom = 0
                    viewer.cam.lookat[:] = data.subtree_com[1]
                elif compared:
                    data.qpos[:], data.qvel[:] = qpos[goal], 0
                    target_data.qpos[:], target_data.qvel[:] = altered[goal], 0
                    mujoco.mj_forward(model, data); mujoco.mj_forward(model, target_data)
                    show_target(model, target_data, viewer.user_scn)
                    viewer.cam.lookat[:] = target_data.subtree_com[1]
                else:
                    target, target_velocity = desired[step - 1], desired_velocity[step - 1]
                    if args.kinematic:
                        model.eq_data[:, 0] = target; data.ctrl[:] = 0
                        mujoco.mj_step(model, data); mujoco.mj_forward(model, data)
                    else:
                        mujoco.mj_forward(model, data)
                        mujoco.mj_fullM(model, data, mass_matrix)
                        data.ctrl[:] = (data.qfrc_bias[joint_dofs] - data.qfrc_passive[joint_dofs]
                                        + kp * target + kd * target_velocity
                                        + mass_matrix[np.ix_(joint_dofs, joint_dofs)]
                                        @ desired_acceleration[step - 1])
                        mujoco.mj_step(model, data)
                    target_data.qpos[:], target_data.qvel[:] = control_qpos[goal], 0
                    mujoco.mj_forward(model, target_data)
                    show_target(model, target_data, viewer.user_scn)
                    viewer.cam.lookat[:] = target_data.subtree_com[1]
                viewer.sync()
                if not viewer.is_running():
                    return
                time.sleep(max(0, args.slowdown * model.opt.timestep
                               - (time.perf_counter() - begin)))


if __name__ == "__main__":
    main()
