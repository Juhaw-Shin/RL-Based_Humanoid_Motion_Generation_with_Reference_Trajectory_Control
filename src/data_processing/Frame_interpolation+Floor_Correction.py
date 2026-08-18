"""Retarget CMU/AIST++ motion to the 47-actuator MuJoCo body and save 40-FPS Zarr."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import gc, os, pickle, sys, warnings

import mujoco
import numpy as np
import zarr
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
XML = REPOSITORY / "configs" / "human_body_47.xml"
BODY_BASIS = np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0.0)))  # SMPL: left/up/forward -> XML: y/z/x
FLOOR_PERCENTILE = 20
CONTACT_THRESHOLD = 0.055
SEVERE_PENETRATION_THRESHOLD = -0.15
TRANSITION_FRAMES = 3
SHOULDER_VELOCITY_WEIGHT = 0.003
SHOULDER_ACCELERATION_WEIGHT = 0.012
SHOULDER_MOTION_SCALE = np.deg2rad(10.0)
SHOULDER_MIN_STEP_LIMIT = np.deg2rad(5.0)
SHOULDER_STEP_LIMIT_RATIO = 1.5
SHOULDER_MAX_NFEV = 25
COLLISION_TOLERANCE = 0.001
COLLISION_STRIDE = 4
COLLISION_PADDING = 4
COLLISION_REFINEMENTS = 3
COLLISION_POSE_PASSES = 5
COLLISION_MAX_STEP = np.deg2rad(2.0)
COLLISION_MAX_CORRECTION = np.deg2rad(10.0)
COLLISION_MAX_TEMPORAL_STEP = np.deg2rad(3.0)
FLOOR_PENETRATION_TOLERANCE = 0.001
FLOOR_MAX_LIFT_STEP = 0.010
STABILITY_FILTER = True
GROUND_HEIGHT_THRESHOLD = 0.025
COM_SPEED_THRESHOLD = 0.01
STABILITY_DURATION = 0.3
BOS_OUTSIDE_MARGIN = 0.03
MODEL = DATA = JOINT_NAMES = JOINT_RANGES = HINGE_QPOS = HINGE_DOFS = FLOOR_GEOM = None
COLLISION_WEIGHTS = None


# Load one MuJoCo model per worker and reuse it for every assigned scene.
def _worker_init():
    global MODEL, DATA, JOINT_NAMES, JOINT_RANGES, HINGE_QPOS, HINGE_DOFS, FLOOR_GEOM
    global COLLISION_WEIGHTS
    MODEL = mujoco.MjModel.from_xml_path(str(XML))
    DATA = mujoco.MjData(MODEL)
    joint_ids = np.arange(1, MODEL.njnt)
    joint_ids = joint_ids[np.argsort(MODEL.jnt_qposadr[joint_ids])]
    JOINT_NAMES = [MODEL.joint(int(i)).name for i in joint_ids]
    JOINT_RANGES = MODEL.jnt_range[joint_ids].copy()
    HINGE_QPOS = MODEL.jnt_qposadr[joint_ids]
    HINGE_DOFS = MODEL.jnt_dofadr[joint_ids]
    FLOOR_GEOM = MODEL.geom("floor").id
    COLLISION_WEIGHTS = np.array([
        4.0 if name.startswith(("lumbar", "thoracic", "neck")) else
        2.0 if any(part in name for part in ("hip", "knee", "ankle", "toes")) else 1.0
        for name in JOINT_NAMES
    ])


def _self_depths(qpos):
    depths = np.zeros(len(qpos))
    for i, pose in enumerate(qpos):
        DATA.qpos[:] = pose[:MODEL.nq]
        mujoco.mj_forward(MODEL, DATA)
        depths[i] = max((-float(c.dist) for c in DATA.contact[:DATA.ncon]
                         if MODEL.geom_bodyid[c.geom1] and MODEL.geom_bodyid[c.geom2]),
                        default=0.0)
    return depths


def _floor_depths(qpos):
    depths = np.zeros(len(qpos))
    for i, pose in enumerate(qpos):
        DATA.qpos[:] = pose[:MODEL.nq]
        mujoco.mj_forward(MODEL, DATA)
        depths[i] = max((-float(c.dist) for c in DATA.contact[:DATA.ncon]
                         if FLOOR_GEOM in (c.geom1, c.geom2)), default=0.0)
    return depths


def _lift_from_floor(qpos):
    depth = _floor_depths(qpos)
    lift = np.maximum(0.0, depth - FLOOR_PENETRATION_TOLERANCE)
    for i in range(1, len(lift)):
        lift[i] = max(lift[i], lift[i - 1] - FLOOR_MAX_LIFT_STEP)
    for i in range(len(lift) - 2, -1, -1):
        lift[i] = max(lift[i], lift[i + 1] - FLOOR_MAX_LIFT_STEP)
    qpos[:, 2] += lift
    return depth, lift


def _repair_collision_pose(original, initial, enabled=None):
    enabled = np.arange(len(HINGE_QPOS)) if enabled is None else np.asarray(enabled, dtype=int)
    qpos_index, dof_index, weights = HINGE_QPOS[enabled], HINGE_DOFS[enabled], COLLISION_WEIGHTS[enabled]
    q = initial.copy()
    for _ in range(COLLISION_POSE_PASSES):
        DATA.qpos[:] = q
        mujoco.mj_forward(MODEL, DATA)
        jacobian = DATA.efc_J.reshape(DATA.nefc, MODEL.nv)
        update = np.zeros(len(enabled))
        active = False
        for contact in DATA.contact[:DATA.ncon]:
            if (contact.dist >= -COLLISION_TOLERANCE or
                    not MODEL.geom_bodyid[contact.geom1] or
                    not MODEL.geom_bodyid[contact.geom2] or contact.efc_address < 0):
                continue
            row = jacobian[contact.efc_address, dof_index]
            direction = row / weights
            denominator = row @ direction
            if denominator > 1e-10:
                update += direction * (-COLLISION_TOLERANCE - contact.dist) / denominator
                active = True
        if not active:
            break
        update = np.clip(update, -COLLISION_MAX_STEP, COLLISION_MAX_STEP)
        q[qpos_index] = np.clip(
            q[qpos_index] + update,
            np.maximum(JOINT_RANGES[enabled, 0], original[qpos_index] - COLLISION_MAX_CORRECTION),
            np.minimum(JOINT_RANGES[enabled, 1], original[qpos_index] + COLLISION_MAX_CORRECTION),
        )
    return q


def _collision_correct(qpos, return_info=False, enabled=None, floor=True, lock_ends=False):
    enabled = np.arange(len(HINGE_QPOS)) if enabled is None else np.asarray(enabled, dtype=int)
    qpos_index, ranges = HINGE_QPOS[enabled], JOINT_RANGES[enabled]
    original = np.asarray(qpos, dtype=np.float64)
    corrected = original.copy()
    before = _self_depths(original)
    infected = np.flatnonzero(before > COLLISION_TOLERANCE)
    windows = []
    if len(infected):
        runs = np.split(infected, np.flatnonzero(np.diff(infected) > 3) + 1)
        for run in runs:
            window = [max(0, int(run[0]) - COLLISION_PADDING),
                      min(len(original) - 1, int(run[-1]) + COLLISION_PADDING)]
            if windows and window[0] <= windows[-1][1]:
                windows[-1][1] = max(windows[-1][1], window[1])
            else:
                windows.append(window)

    for start, end in windows:
        controls = {start: np.zeros(len(enabled)), end: np.zeros(len(enabled))}
        if start + 1 < end and before[start:start + 2].max() <= COLLISION_TOLERANCE:
            controls[start + 1] = np.zeros(len(enabled))
        if end - 1 > start and before[end - 1:end + 1].max() <= COLLISION_TOLERANCE:
            controls[end - 1] = np.zeros(len(enabled))
        bad = infected[(infected >= start) & (infected <= end)]
        selected = np.unique(np.r_[bad[::COLLISION_STRIDE], bad[np.argmax(before[bad])]])
        if lock_ends: selected = selected[(selected > 0) & (selected < len(original)-1)]
        previous_delta = np.zeros(len(enabled))
        for frame in selected:
            initial = original[frame].copy()
            initial[qpos_index] += previous_delta
            pose = _repair_collision_pose(original[frame], initial, enabled)
            controls[int(frame)] = pose[qpos_index] - original[frame, qpos_index]
            previous_delta = controls[int(frame)]

        for _ in range(COLLISION_REFINEMENTS + 1):
            keys = np.array(sorted(controls))
            values = np.array([controls[int(key)] for key in keys])
            frames = np.arange(start, end + 1)
            delta = PchipInterpolator(keys, values, axis=0)(frames)
            for i in range(1, len(delta)):
                delta[i] = np.clip(delta[i], delta[i - 1] - COLLISION_MAX_TEMPORAL_STEP,
                                   delta[i - 1] + COLLISION_MAX_TEMPORAL_STEP)
            for i in range(len(delta) - 2, -1, -1):
                delta[i] = np.clip(delta[i], delta[i + 1] - COLLISION_MAX_TEMPORAL_STEP,
                                   delta[i + 1] + COLLISION_MAX_TEMPORAL_STEP)
            corrected[start:end + 1, qpos_index] = np.clip(
                original[start:end + 1, qpos_index] + delta,
                np.maximum(ranges[:, 0], original[start:end + 1, qpos_index] - COLLISION_MAX_CORRECTION),
                np.minimum(ranges[:, 1], original[start:end + 1, qpos_index] + COLLISION_MAX_CORRECTION),
            )
            residual = _self_depths(corrected[start:end + 1])
            bad_local = np.flatnonzero(residual > COLLISION_TOLERANCE)
            if not len(bad_local):
                break
            candidates = bad_local[::COLLISION_STRIDE]
            candidates = np.unique(np.r_[candidates, bad_local[np.argmax(residual[bad_local])]])
            if lock_ends: candidates = candidates[((start+candidates) > 0) & ((start+candidates) < len(original)-1)]
            for local in candidates:
                frame = start + int(local)
                pose = _repair_collision_pose(original[frame], corrected[frame], enabled)
                controls[frame] = pose[qpos_index] - original[frame, qpos_index]

    floor_before, lift = _lift_from_floor(corrected) if floor else (np.zeros(len(corrected)), np.zeros(len(corrected)))
    if not return_info:
        return corrected
    info = {"before": before, "after": _self_depths(corrected), "windows": len(windows),
            "floor_before": floor_before, "floor_after": _floor_depths(corrected),
            "root_lift": lift}
    return corrected, info


# Fit one bounded shoulder path directly to the source rotations.  Temporal terms
# select a continuous path without switching between discrete Euler branches.
def _continuous_shoulder_angles(local, limits, abduction_sign):
    if len(local) == 0:
        return np.empty((0, 3), dtype=np.float64)

    principal = local.as_euler("YXZ")
    principal[:, 1] *= abduction_sign
    lower, upper = limits[:, 0], limits[:, 1]
    result = np.empty_like(principal)

    def orientation_error(q, target):
        physical = q.copy()
        physical[1] *= abduction_sign
        return (Rotation.from_euler("YXZ", physical).inv() * target).as_rotvec()

    def solve(residual, initial, solve_lower=lower, solve_upper=upper):
        fit = least_squares(
            residual, np.clip(initial, solve_lower, solve_upper),
            bounds=(solve_lower, solve_upper),
            ftol=1e-7, xtol=1e-7, gtol=1e-7, max_nfev=SHOULDER_MAX_NFEV,
        )
        if not fit.success:
            fit = least_squares(
                residual, fit.x, bounds=(solve_lower, solve_upper),
                ftol=1e-7, xtol=1e-7, gtol=1e-7,
                max_nfev=4 * SHOULDER_MAX_NFEV,
            )
        if not fit.success or not np.all(np.isfinite(fit.x)):
            raise RuntimeError(f"shoulder fit failed: {fit.message}")
        return fit.x

    result[0] = solve(lambda q: orientation_error(q, local[0]), principal[0])
    previous_previous = result[0]
    previous = result[0]
    for i in range(1, len(local)):
        source_step = (local[i - 1].inv() * local[i]).magnitude()
        step_limit = max(SHOULDER_MIN_STEP_LIMIT,
                         SHOULDER_STEP_LIMIT_RATIO * source_step)
        solve_lower = np.maximum(lower, previous - step_limit)
        solve_upper = np.minimum(upper, previous + step_limit)
        predicted = np.clip(2 * previous - previous_previous,
                            solve_lower, solve_upper)
        temporal_scale = 1.0 / (1.0 + (source_step / SHOULDER_MOTION_SCALE) ** 2)
        velocity_scale = np.sqrt(SHOULDER_VELOCITY_WEIGHT * temporal_scale)
        acceleration_scale = np.sqrt(SHOULDER_ACCELERATION_WEIGHT * temporal_scale)

        def residual(q):
            return np.concatenate((
                orientation_error(q, local[i]),
                velocity_scale * (q - previous),
                acceleration_scale * (q - predicted),
            ))

        result[i] = solve(residual, predicted, solve_lower, solve_upper)
        previous_previous, previous = previous, result[i]
    return result


# Compose source rotations, align SMPL and XML rest axes, and extract the 47 XML hinges.
def _map_pose(pose, trans, kind, scaling=1.0):
    p = np.asarray(pose, dtype=np.float64).reshape(len(pose), -1, 3)
    B = BODY_BASIS

    def yxz(rotation, basis=None):
        matrix = B @ rotation.as_matrix() @ B.T
        if basis is not None:
            matrix = basis.T @ matrix @ basis
        return Rotation.from_matrix(matrix).as_euler("YXZ")

    def twist(rotation, axis):
        q = rotation.as_quat()
        angle = 2 * np.arctan2(q[:, :3] @ axis, q[:, 3])
        return (angle + np.pi) % (2 * np.pi) - np.pi

    angles = {}
    spine = [Rotation.from_rotvec(p[:, 3]) ** 0.5,
             Rotation.from_rotvec(p[:, 3]) ** 0.5,
             Rotation.from_rotvec(p[:, 6]), Rotation.from_rotvec(p[:, 9])]
    for prefix, rotation in zip(
        ("lumbar_lower", "lumbar_upper", "thoracic_lower", "thoracic_upper"), spine
    ):
        e = yxz(rotation)
        angles.update({f"{prefix}_flex": e[:, 0], f"{prefix}_lat": e[:, 1],
                       f"{prefix}_axial": e[:, 2]})

    e = yxz(Rotation.from_rotvec(p[:, 12]) * Rotation.from_rotvec(p[:, 15]))
    angles.update(neck_flex=e[:, 0], neck_lat=e[:, 1], neck_axial=e[:, 2])

    shoulder_drop = np.arctan2(0.297641, 0.090194)
    for side, collar, shoulder, elbow, wrist, sign in (
        ("left", 13, 16, 18, 20, 1), ("right", 14, 17, 19, 21, -1)
    ):
        source = Rotation.from_rotvec(p[:, collar]) * Rotation.from_rotvec(p[:, shoulder])
        neutral = Rotation.from_rotvec((0, 0, -sign * shoulder_drop))
        flex_axis = np.array((0, -0.957024, -sign * 0.290007))
        rot_axis = np.array((0, sign * 0.290007, -0.957024))
        basis = np.column_stack(((1, 0, 0), flex_axis, rot_axis))
        local = Rotation.from_matrix(
            basis.T @ (B @ (source * neutral.inv()).as_matrix() @ B.T) @ basis
        )
        names = [f"{side}_shoulder_flex", f"{side}_shoulder_abd", f"{side}_shoulder_rot"]
        limits = np.array([MODEL.jnt_range[MODEL.joint(name).id] for name in names])
        e = _continuous_shoulder_angles(local, limits, sign)
        elbow_rotation = Rotation.from_rotvec(p[:, elbow])
        roll_axis, elbow_flex_axis = np.array((sign, 0, 0)), np.array((0, -sign, 0))
        roll = twist(elbow_rotation, roll_axis)
        flex = twist(elbow_rotation * Rotation.from_rotvec(roll[:, None] * roll_axis).inv(),
                     elbow_flex_axis)
        wrist_angles = Rotation.from_rotvec(p[:, wrist]).as_euler("XYZ")
        angles.update({
            f"{side}_shoulder_flex": e[:, 0],
            f"{side}_shoulder_abd": e[:, 1],
            f"{side}_shoulder_rot": e[:, 2],
            f"{side}_elbow_flex": flex,
            f"{side}_forearm_roll": (roll + sign * wrist_angles[:, 0] + np.pi)
                                    % (2 * np.pi) - np.pi,
            f"{side}_wrist_flex": -sign * wrist_angles[:, 1],
            f"{side}_wrist_dev": sign * wrist_angles[:, 2],
            f"{side}_thumb_flex": np.zeros(len(p)),
            f"{side}_fingers_flex": np.zeros(len(p)),
        })

    for side, hip, knee, ankle, foot, sign in (
        ("left", 1, 4, 7, 10, 1), ("right", 2, 5, 8, 11, -1)
    ):
        h, a, toe = (yxz(Rotation.from_rotvec(p[:, j])) for j in (hip, ankle, foot))
        knee_rotation = Rotation.from_matrix(
            B @ Rotation.from_rotvec(p[:, knee]).as_matrix() @ B.T
        )
        angles.update({
            f"{side}_hip_flex": -h[:, 0], f"{side}_hip_abd": sign * h[:, 1],
            f"{side}_hip_rot": h[:, 2],
            f"{side}_knee_flex": twist(knee_rotation, np.array((0, 1, 0))),
            f"{side}_ankle_flex": a[:, 0], f"{side}_ankle_inv": sign * a[:, 1],
            f"{side}_toes_flex": toe[:, 0],
        })

    hinges = np.column_stack([angles[name] for name in JOINT_NAMES])
    hinges = np.clip(hinges, JOINT_RANGES[:, 0], JOINT_RANGES[:, 1])

    root = Rotation.from_rotvec(p[:, 0]).as_matrix()
    root = root @ B.T if kind == "CMU" else B @ root @ B.T
    quat = Rotation.from_matrix(root).as_quat()[:, (3, 0, 1, 2)]  # MuJoCo WXYZ
    for i in range(1, len(quat)):
        if quat[i] @ quat[i - 1] < 0:
            quat[i] *= -1
    xyz = np.asarray(trans, dtype=np.float64)
    if kind == "motions":
        xyz = (B @ (xyz / float(scaling)).T).T
    return np.column_stack((xyz, quat, hinges))


# Set toes neutral, align the 10th-lowest ankle frame, then limit downward ankle motion.
def _floor_correct(qpos):
    sides = ("left", "right")
    ankle_ids = [MODEL.joint(f"{side}_ankle_flex").id for side in sides]
    flex_q = [MODEL.jnt_qposadr[joint] for joint in ankle_ids]
    inv_q = [MODEL.jnt_qposadr[MODEL.joint(f"{side}_ankle_inv").id] for side in sides]
    toe_q = [MODEL.jnt_qposadr[MODEL.joint(f"{side}_toes_flex").id] for side in sides]
    qpos[:, toe_q] = 0
    heights = np.empty((len(qpos), 2))
    for i, frame in enumerate(qpos):
        DATA.qpos[:] = frame
        mujoco.mj_forward(MODEL, DATA)
        heights[i] = DATA.xanchor[ankle_ids, 2]
    shift = np.percentile(heights.min(axis=1), FLOOR_PERCENTILE) - 0.09
    qpos[:, 2] -= shift
    heights -= shift

    # Upright XML geometry: toe reaches 0.195 m forward and 0.09 m downward.
    flex_r, flex_phase = np.hypot(0.195, 0.09), np.arctan2(0.09, 0.195)
    flex_limit = np.deg2rad(50)
    for side in range(2):
        h = heights[:, side]
        maximum = np.where(
            h <= 0.09, 0,
            np.where(h >= 0.195 * np.sin(flex_limit) + 0.09 * np.cos(flex_limit),
                     flex_limit, np.arcsin(np.clip(h / flex_r, -1, 1)) - flex_phase))
        qpos[:, flex_q[side]] = np.minimum(qpos[:, flex_q[side]], maximum)

        # The foot extends 0.048 m sideways and 0.09 m downward from the ankle.
        inv_r, inv_phase = np.hypot(0.048, 0.09), np.arctan2(0.09, 0.048)
        maximum = np.where(
            h <= 0.09, 0,
            np.where(h >= inv_r, np.inf,
                     np.arcsin(np.clip(h / inv_r, -1, 1)) - inv_phase))
        qpos[:, inv_q[side]] = np.clip(qpos[:, inv_q[side]], -maximum, maximum)

    return qpos


# Calculate MuJoCo-compatible linear, angular, and hinge velocities at the source FPS.
def _velocities(qpos, fps):
    qvel = np.empty((len(qpos), MODEL.nv), dtype=np.float64)
    if len(qpos) == 1:
        qvel[0] = 0
        return qvel
    dt = 1.0 / fps
    mujoco.mj_differentiatePos(MODEL, qvel[0], dt, qpos[0], qpos[1])
    for i in range(1, len(qpos) - 1):
        mujoco.mj_differentiatePos(MODEL, qvel[i], 2 * dt, qpos[i - 1], qpos[i + 1])
    mujoco.mj_differentiatePos(MODEL, qvel[-1], dt, qpos[-2], qpos[-1])
    return qvel


# Resample translations/hinges with their velocities and root orientation with quaternion SLERP.
def _interpolate(qpos, qvel, fps):
    old_t = np.arange(len(qpos)) / fps
    count = max(1, round(len(qpos) * 40 / fps))
    new_t = np.minimum(np.arange(count) / 40, old_t[-1])
    if len(qpos) == 1:
        return qpos.copy(), qvel.copy()
    out = np.empty((count, MODEL.nq))
    for q, v in zip(range(3), range(3)):
        out[:, q] = CubicHermiteSpline(old_t, qpos[:, q], qvel[:, v])(new_t)
    rotation = Rotation.from_quat(qpos[:, (4, 5, 6, 3)])
    out[:, 3:7] = Slerp(old_t, rotation)(new_t).as_quat()[:, (3, 0, 1, 2)]
    for q, v in zip(range(7, MODEL.nq), range(6, MODEL.nv)):
        values = np.unwrap(qpos[:, q])
        out[:, q] = CubicHermiteSpline(old_t, values, qvel[:, v])(new_t)
    out[:, 7:] = np.clip(out[:, 7:], JOINT_RANGES[:, 0], JOINT_RANGES[:, 1])
    velocity = np.column_stack([
        np.interp(new_t, old_t, qvel[:, i]) for i in range(MODEL.nv)
    ])
    return out, velocity


def interpolation_60(qpos, qvel):
    return _interpolate(qpos, qvel, 60)


def interpolation_120(qpos, qvel):
    return _interpolate(qpos, qvel, 120)


# Convert one independent scene; ProcessPoolExecutor runs this calculation concurrently.
def _process_scene(task):
    kind, path = task
    if kind == "motions":
        with open(path, "rb") as file:
            source = pickle.load(file)
        pose, trans = source["smpl_poses"], source["smpl_trans"]
        fps, subject, scaling = 60, None, float(source["smpl_scaling"][0])
    else:
        with np.load(path) as source:
            fps, subject = round(float(source["mocap_frame_rate"])), Path(path).parent.name
            pose, trans = source["poses"], source["trans"]
        scaling = 1.0
    if fps not in (60, 120):
        raise ValueError(f"{path}: unsupported {fps} FPS")
    pose, trans = np.asarray(pose), np.asarray(trans)
    if pose.ndim != 2 or pose.shape[0] == 0 or pose.shape[1] < 66 \
            or pose.shape[1] % 3 or trans.shape != (len(pose), 3):
        raise ValueError(f"invalid source shapes pose={pose.shape}, trans={trans.shape}")
    if not np.isfinite(pose).all() or not np.isfinite(trans).all() \
            or not np.isfinite(scaling) or scaling == 0:
        raise ValueError("non-finite source values or invalid scaling")
    qpos = _map_pose(pose, trans, kind, scaling)
    qpos = _floor_correct(qpos)
    qvel = _velocities(qpos, fps)
    qpos, qvel = (interpolation_60 if fps == 60 else interpolation_120)(qpos, qvel)
    if len(qpos) > 2:
        smooth = qpos.copy()
        smooth[1:-1, :3] = (qpos[:-2, :3] + qpos[1:-1, :3] + qpos[2:, :3]) / 3
        quat = qpos[:-2, 3:7] + qpos[1:-1, 3:7] + qpos[2:, 3:7]
        smooth[1:-1, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        smooth[1:-1, 7:] = (qpos[:-2, 7:] + qpos[1:-1, 7:] + qpos[2:, 7:]) / 3
        qpos = smooth
    feet = [[MODEL.geom(f"{side}_{part}").id for part in ("foot", "toes_chunk")]
            for side in ("left", "right")]
    geoms = feet[0] + feet[1]
    lowest = np.empty(len(qpos))
    for i, frame in enumerate(qpos):
        DATA.qpos[:] = frame
        mujoco.mj_forward(MODEL, DATA)
        lowest[i] = min(DATA.geom_xpos[g, 2] -
                        np.abs(DATA.geom_xmat[g].reshape(3, 3)[2]) @ MODEL.geom_size[g]
                        for g in geoms)
    if lowest.min() < SEVERE_PENETRATION_THRESHOLD:
        return kind, subject, Path(path).stem, None, None
    contact = lowest <= CONTACT_THRESHOLD
    weight = contact.astype(float)
    for distance in range(1, TRANSITION_FRAMES + 1):
        blend = (TRANSITION_FRAMES + 1 - distance) / (TRANSITION_FRAMES + 1)
        weight[distance:] = np.maximum(weight[distance:], contact[:-distance] * blend)
        weight[:-distance] = np.maximum(weight[:-distance], contact[distance:] * blend)
    qpos[:, 2] -= weight * lowest
    qpos = _collision_correct(qpos)
    qvel = _velocities(qpos, 40)

    centers = np.empty((len(qpos), (MODEL.nbody - 1) * 3))
    com = np.empty((len(qpos), 3))
    unstable_support = np.zeros(len(qpos), dtype=bool)
    signs = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                      [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
    for i, frame in enumerate(qpos):
        DATA.qpos[:] = frame
        mujoco.mj_forward(MODEL, DATA)
        centers[i] = DATA.xipos[1:].ravel()
        if STABILITY_FILTER:
            com[i] = DATA.subtree_com[1]
            grounded = [foot for foot in feet if min(
                DATA.geom_xpos[g, 2] - np.abs(DATA.geom_xmat[g].reshape(3, 3)[2]) @ MODEL.geom_size[g]
                for g in foot) <= GROUND_HEIGHT_THRESHOLD]
            if not grounded:
                unstable_support[i] = True
            else:
                points = np.concatenate([
                    (DATA.geom_xpos[g] + (DATA.geom_xmat[g].reshape(3, 3) @
                     (signs * MODEL.geom_size[g]).T).T)[:, :2]
                    for foot in grounded for g in foot])
                hull = ConvexHull(points)
                polygon = points[hull.vertices]
                if np.any(hull.equations[:, :2] @ com[i, :2] + hull.equations[:, 2] > 0):
                    edges = np.roll(polygon, -1, axis=0) - polygon
                    t = np.clip(np.sum((com[i, :2] - polygon) * edges, axis=1) /
                                np.sum(edges * edges, axis=1), 0, 1)
                    unstable_support[i] = np.min(np.linalg.norm(
                        com[i, :2] - polygon - t[:, None] * edges, axis=1)) > BOS_OUTSIDE_MARGIN
    if STABILITY_FILTER:
        speed = (np.linalg.norm(np.gradient(com, 1 / 40, axis=0), axis=1)
                 if len(qpos) > 1 else np.zeros(1))
        bad = unstable_support & (speed < COM_SPEED_THRESHOLD)
        frames = max(1, round(STABILITY_DURATION * 40))
        if len(bad) >= frames and np.convolve(bad, np.ones(frames, dtype=int), "valid").max() == frames:
            raise ValueError(f"unstable support maintained for {STABILITY_DURATION:g} s")
    center_vel = (np.gradient(centers, 1 / 40, axis=0) if len(qpos) > 1
                  else np.zeros_like(centers))
    qpos = np.column_stack((qpos, centers))
    qvel = np.column_stack((qvel, center_vel))
    return kind, subject, Path(path).stem, qpos.astype("f4"), qvel.astype("f4")


# A bad source scene must not abort conversion of every later scene.
def _process(task):
    kind, path = task
    subject = None if kind == "motions" else Path(path).parent.name
    scene = Path(path).stem
    try:
        result = _process_scene(task)
        qpos, qvel = result[3:5]
        reason = (f"minimum foot height below {SEVERE_PENETRATION_THRESHOLD:g} m"
                  if qpos is None else None)
        if qpos is not None:
            centers = 3 * (MODEL.nbody - 1)
            expected = (MODEL.nq + centers, MODEL.nv + centers)
            if qpos.ndim != 2 or qvel.ndim != 2 or qpos.shape[0] == 0:
                raise ValueError(f"invalid output shapes qpos={qpos.shape}, qvel={qvel.shape}")
            if qpos.shape[0] != qvel.shape[0] or qpos.shape[1] != expected[0] \
                    or qvel.shape[1] != expected[1]:
                raise ValueError(f"incompatible output shapes qpos={qpos.shape}, qvel={qvel.shape}")
            if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                raise ValueError("non-finite converted values")
            if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-4):
                raise ValueError("invalid root quaternion")
        return (*result, reason)
    except Exception as error:
        return kind, subject, scene, None, None, f"{type(error).__name__}: {error}"


# Write one scene at a time so the complete datasets never occupy memory together.
def _write(root, parts, qpos, qvel):
    group = root
    for part in parts:
        group = group.require_group(part)
    group.create_array("qpos", data=qpos, chunks=(min(1024, len(qpos)), qpos.shape[1]))
    group.create_array("qvel", data=qvel, chunks=(min(1024, len(qvel)), qvel.shape[1]))


def _complete(root, subject, scene):
    try:
        qpos, qvel = root[subject][scene]["qpos"], root[subject][scene]["qvel"]
        return qpos.ndim == qvel.ndim == 2 and qpos.shape[0] > 0 \
            and qpos.shape[0] == qvel.shape[0]
    except KeyError:
        return False


def main():
    warnings.filterwarnings("ignore", message="Gimbal lock detected")
    options = set(sys.argv[1:])
    if not options <= {"--restart", "--retry-skipped"}:
        raise SystemExit("Options: --restart | --retry-skipped")
    cmu = sorted((REPOSITORY / "data" / "raw_cmu" / "CMU").glob("*/*_stageii.npz"))
    if not cmu:
        raise FileNotFoundError("No CMU files found; output was not overwritten")
    root = zarr.open_group(REPOSITORY / "data" / "corrected_cmu" / "CMU_corrected",
                           mode="w" if "--restart" in options else "a")
    saved_skips = {} if options & {"--restart", "--retry-skipped"} \
        else dict(root.attrs.get("skipped_scenes", {}))
    tasks, complete, checkpoint_skips = [], 0, 0
    for path in cmu:
        subject, scene = path.parent.name, path.stem
        key = f"{subject}/{scene}"
        if _complete(root, subject, scene):
            complete += 1
            saved_skips.pop(key, None)
        elif key in saved_skips:
            checkpoint_skips += 1
        else:
            if subject in root and scene in root[subject]:
                del root[subject][scene]  # remove a write interrupted between qpos and qvel
            tasks.append(("CMU", str(path)))
    root.attrs["skipped_scenes"] = saved_skips
    print(f"Checkpoint: complete={complete}, skipped={checkpoint_skips}, "
          f"remaining={len(tasks)}", flush=True)
    workers = min(8, os.cpu_count() or 1)
    map_options = {"chunksize": 1}
    if sys.version_info >= (3, 14):
        map_options["buffersize"] = workers * 2
    progress_interval, skipped = max(1, len(tasks) // 100), checkpoint_skips
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        for done, result in enumerate(
                executor.map(_process, tasks, **map_options), 1):
            kind, subject, scene, qpos, qvel, reason = result
            key = f"{subject}/{scene}"
            if qpos is None:
                skipped += 1
                saved_skips[key] = reason
                root.attrs["skipped_scenes"] = saved_skips
                location = f"{kind}/{scene}" if subject is None else f"{kind}/{subject}/{scene}"
                print(f"Skipped {location}: {reason}", flush=True)
            else:
                _write(root, (scene,) if subject is None else (subject, scene), qpos, qvel)
                if saved_skips.pop(key, None) is not None:
                    root.attrs["skipped_scenes"] = saved_skips
                del qpos, qvel
                gc.collect()
            if done % progress_interval == 0 or done == len(tasks):
                finished = complete + checkpoint_skips + done
                print(f"Conversion: {100 * finished / len(cmu):6.2f}% "
                      f"({finished}/{len(cmu)}, skipped={skipped})", flush=True)

if __name__ == "__main__":
    main()


# Original implementation instructions

#Import needed modules such as zarr, gc, mujoco, numpy

#First:Map each scene's data to useful form:

#Make a loop for each of the motions file's pkl
#Within the loop, we will create the variable "motions" with the key at being the file's name
#Loop: Call the pkl file, for each pkl file: make empty motions variable. within the motions variable, make another loop over frame of the pkl that delete the angles that are useless and order the useful anlges within each frame as the order the XML file has on it's acuator(after converting to quatornion). approximate Toe flex and forearm roll for the givien values(foot and elbow twist rotations and leave thumb have no actions.) (If some acgles are over the angle that is allowed, clip it.)
#Then, also within each front line of each frame(inside the frame loop), add xyz translation
#If you done it for each of the frame, then save it as a numpy array with 2 dimensions.
#continue this for all files within the motions.
#then it is done for motions folder

#Again, create a file-unit loop for each of the files within the CMU(this file has like this: 1,2,3,4,5 and within it, each scenes.)
# within the loop, create a variable CMU with keys being thesae two: first key is 1,2,3,4,5 and the second is the specific file name within it.
#  call each of the CMU file within the file loop and proccess the same angular and translational value within each of hte CMU keys.
#However, this time, in front of the CMU variable's array of values(of each frame), add the frame rate it has was recorded on.


#Second: floor correction

#Make a def lowest(model, data) that when given the data, it changes the mujoco model's state and the do mj.forward then measures the foot and toes's lowest value(in respect of hiehgt). This should consider the shape of hte foot and toes. Then we have four value: left and right of toe, of foot
#Then fromt eh 4 values, pick the lowest one then output it.

#For motion variable, make a simular loop of file and within each frame and for  each of the frame, change a mujoco's model's angular and translational data and do lowest(model, data). This gives the 1 dimentional numpy array output per file(scene)
#for each of the file, collect the 10 lowest value's index and height(each of different variable neamed motions_lowest_index and motions_lowest_height, ordered the same). Then, do median(motions_lowest_height) then the output of index and value, then
# add that median height value to all of the scene's frame of y(height) exapt the  5 first index we have collected and for the special 5 index's frame, add the corresponding lowest height value to the y.
#repeat for each of the file(scenes).

#For the CMu variable, do the same.

#Third: frame interpolation

#make interpolation_60 and interpolation_120 function that can interpolate the 60 frame values and 120 fps values to 40 fps, while preserving all the velocities(can be derieved) and positional values
#For motions folder, do the interpolation for each of the files for 60 frame
#Fro CMU folder, do the interpolation for each of the files for the value that is embbeded at the first index, which is hte fps for each of hte recording Can be 60 or 120.

#After all these steps, save the files in the zarr form while preserving the architecture of hte folder. For example, fo rthe CMU, it is stored as 1,2,3,4,5 and then for each them, have the file names, and for the motions, it is just pure files within the folder. WRit eht folder name as CMU_corrected and motions_corrected.




#   1. Should each output frame be full MuJoCo qpos: root XYZ, root quaternion WXYZ, then 47 hinge angles—54 values total?

#yes

#   2. The XML uses Z as height, but your instruction says correct Y. Should floor correction modify root Z?

#Yes. The height

#   3. Should all output angles be stored in radians? MuJoCo runtime uses radians even though the XML ranges are written in degrees.

#Yes.

#   4. Should AIST++ smpl_scaling be applied to smpl_trans? Its translations and CMU translations use different scales/coordinate systems.

#Yes

#   5. For the two thumb actuators, should motions files store zero while CMU uses its real thumb joints, keeping 47 angles in both?

#Yes.

#   6. For the four XML spine sections, should I use our agreed mapping: split spine1 across both lumbar sections, then map spine2/3 to lower/upper thoracic?

#Yes.

#   7. CMU contains neutral_stagei.npz. Should those be skipped and only *_stageii.npz processed?

# Yes

#   8. Floor correction says collect 10 lowest frames but specially correct only the first 5. Is that intentional? Also, should the correction be root_z -= lowest_height so the lowest geometry becomes zero?

#Yes. What I said was to pick a 5 values that are lower or equal tot he median(because this is defenition of median). Also yes. that is the right correction.

#   9. Per-frame special corrections can cause sudden vertical jumps. Do you want one constant floor offset per scene, or a smoothly varying correction?

#No, just keep special corrections.

#   10. “Preserve velocities”: should Zarr store both qpos and calculated qvel, or only 40-FPS positions whose timing preserves motion speed?

#Yes. It should be calculated through original frame(60 or 120) using frame before and after. so for frame i, do i-1 and i+1 then divide by the time.

#   11. Should FPS be stored once as Zarr metadata rather than repeated as the first value of every CMU frame?

#Just don't keep fps record. All are 40fps.

#   12. If corrected output folders already exist, should they be overwritten or skipped?

# overwritten




#All of the things are run within this file. When clicked Run without debugging.
