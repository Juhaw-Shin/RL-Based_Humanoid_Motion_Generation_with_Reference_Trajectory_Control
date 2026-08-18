"""Online recurrent PPO goal-residual learning with fixed PD gains and MJWarp."""
import argparse, csv, math, os, socket, time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import warp as wp
import zarr

# ARTIFACT CONTRACT
# These paths are one training state: cache/normalization/checkpoint must match the
# XML, gains, residual Std, and feature ordering. Reusing only some can silently
# change the policy's input or action meaning even when tensor shapes still match.
ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DATA_PATH = REPOSITORY / "data" / "movement_data" / "Movement_Data"
VALIDATION_PATH = REPOSITORY / "data" / "validation" / "Validation"
XML_PATH = REPOSITORY / "configs" / "human_body_47.xml"
GAINS_PATH = REPOSITORY / "configs" / "PD_Gains.npz"
STD_PATH = REPOSITORY / "configs" / "Residual_Std.npy"
CACHE_DATA, CACHE_META = REPOSITORY / "data" / "cache" / "Movement_Data_Cache.npy", REPOSITORY / "data" / "cache" / "Movement_Data_Cache_Metadata.npz"
NORMALIZATION_PATH = REPOSITORY / "configs" / "Goal_Refinement_Normalization.npz"
CHECKPOINT_PATH = REPOSITORY / "models" / "Goal_Refinement_v5_batch1480.pt"
TRAIN_CSV = REPOSITORY / "results" / "Goal_Refinement_Training.csv"
VALID_CSV = REPOSITORY / "results" / "Goal_Refinement_Validation.csv"
TRACKING_PNG = REPOSITORY / "results" / "Goal_Refinement_Tracking.png"
RESIDUAL_PNG = REPOSITORY / "results" / "Goal_Refinement_Residual_Feasibility.png"
HEALTH_PNG = REPOSITORY / "results" / "Goal_Refinement_Learning_Health.png"
PERFORMANCE_REPORT = REPOSITORY / "results" / "Goal_Refinement_Performance.txt"
REWARD_CALIBRATION_PATH = REPOSITORY / "configs" / "Goal_Refinement_Reward_Calibration.npz"
REWARD_REPORT = REPOSITORY / "docs" / "REWARD_METRICS.md"

# LEARNING CONTRACT
# Dimensions encode exact qpos/qvel/body/actuator order, not merely array sizes.
# Changes to action filtering, Std, PD dynamics, or reward definitions invalidate
# calibrated cost scales; changes to recurrent inputs also invalidate checkpoints.
CRITIC_ONLY = False
EXPLORING_REGION = 6
CUDA_PHYSICS_DEBUG = False
TRANSITION_CUDA_GRAPH = True
SMOOTH = 15  # uniform causal average over current and preceding raw residuals
STD_MULTIPLIER = 0.7
ENVIRONMENT_COUNT, COLLECTION_TRANSITIONS = 450, 150_000
PRINT_INTERVAL, VALID_INTERVAL, CHECKPOINT_INTERVAL, PNG_INTERVAL = 1, 10, 20, 100
ACTOR_LR, CRITIC_LR, GRADIENT_CLIP = 2e-5, 2e-4, 1.0
PPO_EPOCHS, PPO_MINIBATCHES, PPO_CLIP = 5, 8, .2
PENETRATION_GRADIENT_FRACTION = .05
SUPERVISED_PENETRATION = True  # False skips correction labels and their entire backward path
COLLISION_PASSES, COLLISION_SMOOTH_RADIUS = 5, 4
COLLISION_TOLERANCE = .001
COLLISION_MAX_STEP, COLLISION_MAX_CORRECTION = math.radians(2), math.radians(10)
COLLISION_MAX_TEMPORAL_STEP = math.radians(3)
GAMMA, GAE_LAMBDA, VALUE_COEFFICIENT = 0.99, 0.99, 1.0
GAE_SETTINGS = (GAMMA, GAE_LAMBDA)
DURATION_STD, DURATION_MIN, DURATION_MAX = 2.0, 0.05, 4
ACTION_HZ, PHYSICS_HZ, PHYSICS_DT = 40, 400, .0025
SUBSTEPS, MAX_HORIZON = PHYSICS_HZ // ACTION_HZ, round(DURATION_MAX * ACTION_HZ)
SEED, NJMAX, NCONMAX = 12, 330, 64
FRAME_SIZE, QPOS_SIZE, QVEL_SIZE = 251, 126, 125
NQ, NV, NBODY, NU, ACTION_SIZE = 54, 53, 24, 47, 94
STATE_SIZE, CURRENT_SIZE, GOAL_SIZE = 261, 476, 263
CATEGORIES = ("Common_actions", "Extreme_unusual_actions", "Jumping", "Walking_Running")
SOURCES = ("Standing",) + CATEGORIES
REGIONS = (
    tuple(f"right_{x}" for x in ("hip_flex", "hip_abd", "hip_rot", "knee_flex", "ankle_flex", "ankle_inv", "toes_flex")),
    tuple(f"left_{x}" for x in ("hip_flex", "hip_abd", "hip_rot", "knee_flex", "ankle_flex", "ankle_inv", "toes_flex")),
    tuple(f"{part}_{axis}" for part in ("lumbar_lower", "lumbar_upper", "thoracic_lower", "thoracic_upper")
          for axis in ("flex", "lat", "axial")),
    ("right_shoulder_flex", "right_shoulder_abd", "right_shoulder_rot", "right_elbow_flex",
     "right_forearm_roll", "right_wrist_flex", "right_wrist_dev", "right_thumb_flex", "right_fingers_flex"),
    ("left_shoulder_flex", "left_shoulder_abd", "left_shoulder_rot", "left_elbow_flex",
     "left_forearm_roll", "left_wrist_flex", "left_wrist_dev", "left_thumb_flex", "left_fingers_flex"),
    ("neck_flex", "neck_lat", "neck_axial"))
if EXPLORING_REGION not in range(1, len(REGIONS)+1): raise ValueError("EXPLORING_REGION must be 1 through 6")
POLICY_JOINT_NAMES = sum(REGIONS[:EXPLORING_REGION], ())
POLICY_NU, POLICY_SIZE = len(POLICY_JOINT_NAMES), 2*len(POLICY_JOINT_NAMES)
STATIONARY = ("left_thumb_flex", "left_fingers_flex", "right_thumb_flex",
              "right_fingers_flex", "left_toes_flex", "right_toes_flex")
# These fallback coefficients predate the current smoothing; --recalibrate measures
# authoritative values for the selected SMOOTH setting.
REWARD_WEIGHTS = np.array((.50, .25, .0000, 0.0, 0.15, .1, 0.), np.float32)
DIFFERENCE_POSITION_MEAN, DIFFERENCE_VELOCITY_MEAN = .001274222576, .036871570497
# Fall is a fixed once-per-rollout cost, deliberately outside continuous reward calibration.
FALL_COST = 25.0
FALL_HEIGHT_DROP, FALL_TILT_HEIGHT_DROP, FALL_ANGLE = .35, .20, math.radians(45)
FALL_SETTINGS = (FALL_COST, FALL_HEIGHT_DROP, FALL_TILT_HEIGHT_DROP, FALL_ANGLE)
EPS = 1e-8

if not isinstance(SMOOTH, int) or SMOOTH < 1: raise ValueError("SMOOTH must be a positive integer")
if not (0 <= GAMMA <= 1 and 0 <= GAE_LAMBDA <= 1): raise ValueError("GAMMA and GAE_LAMBDA must be within [0, 1]")

RNG = np.random.default_rng(SEED)
DEVICE = torch.device("cuda")
RANK, WORLD_SIZE, PROFILE, PERFORMANCE = 0, 1, {}, False


def primary(): return RANK == 0


# PERFORMANCE ATTRIBUTION
# CUDA synchronization makes section timings meaningful but is intentionally absent
# in normal training because it would serialize otherwise asynchronous GPU work.
@contextmanager
def measured(name):
    if not PERFORMANCE:
        yield; return
    torch.cuda.synchronize(); start = time.perf_counter()
    yield
    torch.cuda.synchronize(); PROFILE[name] = PROFILE.get(name, 0.) + time.perf_counter() - start


# CONTACT OBSERVATION AND PENETRATION COST
# Foot/toe flags are policy observations; depth is a reward diagnostic. The 1 mm
# cutoff must remain identical to reward calibration or penetration scale changes.
@wp.kernel
def _contacts(nacon: wp.array(dtype=wp.int32), geom: wp.array(dtype=wp.vec2i),
              worldid: wp.array(dtype=wp.int32), dist: wp.array(dtype=wp.float32),
              floor: int, left_foot: int, left_toe: int, right_foot: int, right_toe: int,
              flags: wp.array2d(dtype=wp.int32), depth: wp.array(dtype=wp.float32),
              self_depth: wp.array(dtype=wp.float32)):
    i = wp.tid()
    if i < nacon[0]:
        pair, world, d = geom[i], worldid[i], dist[i]
        a, b = pair[0], pair[1]
        if (a == floor and (b == left_foot or b == left_toe)) or \
           (b == floor and (a == left_foot or a == left_toe)):
            flags[world, 0] = 1
        if (a == floor and (b == right_foot or b == right_toe)) or \
           (b == floor and (a == right_foot or a == right_toe)):
            flags[world, 1] = 1
        if d < -0.001:
            wp.atomic_max(depth, world, -d)
            if a != floor and b != floor: wp.atomic_max(self_depth, world, -d)


# GPU SELF-COLLISION LABEL SOLVER
# Target forward already builds this sparse Jacobian. Static-world contacts are
# excluded through geom body IDs, matching the former CPU correction semantics.
@wp.kernel
def _collision_scale(nacon: wp.array(dtype=wp.int32), geom: wp.array(dtype=wp.vec2i),
        worldid: wp.array(dtype=wp.int32), dist: wp.array(dtype=wp.float32),
        efc_address: wp.array2d(dtype=wp.int32), geom_bodyid: wp.array(dtype=wp.int32),
        rownnz: wp.array2d(dtype=wp.int32), rowadr: wp.array2d(dtype=wp.int32),
        colind: wp.array3d(dtype=wp.int32), jacobian: wp.array3d(dtype=wp.float32),
        dofs: wp.array(dtype=wp.int32), weights: wp.array(dtype=wp.float32),
        active: wp.array(dtype=wp.int32), scale: wp.array(dtype=wp.float32)):
    contact = wp.tid(); scale[contact] = 0.0
    if contact < nacon[0]:
        pair, world = geom[contact], worldid[contact]
        if active[world] != 0 and geom_bodyid[pair[0]] != 0 and geom_bodyid[pair[1]] != 0 \
                and dist[contact] < -COLLISION_TOLERANCE:
            row = efc_address[contact, 0]
            if row >= 0:
                start, count = rowadr[world, row], rownnz[world, row]
                denominator = float(0.0)
                for joint in range(dofs.shape[0]):
                    value = float(0.0)
                    for entry in range(count):
                        address = start + entry
                        if colind[world, 0, address] == dofs[joint]: value = jacobian[world, 0, address]
                    denominator += value * value / weights[joint]
                if denominator > 1.0e-10:
                    scale[contact] = (-COLLISION_TOLERANCE-dist[contact]) / denominator


@wp.kernel
def _collision_delta(worldid: wp.array(dtype=wp.int32), efc_address: wp.array2d(dtype=wp.int32),
        rownnz: wp.array2d(dtype=wp.int32), rowadr: wp.array2d(dtype=wp.int32),
        colind: wp.array3d(dtype=wp.int32), jacobian: wp.array3d(dtype=wp.float32),
        dofs: wp.array(dtype=wp.int32), weights: wp.array(dtype=wp.float32),
        scale: wp.array(dtype=wp.float32), delta: wp.array2d(dtype=wp.float32)):
    contact, joint = wp.tid(); factor = scale[contact]
    if factor != 0.0:
        world, row = worldid[contact], efc_address[contact, 0]
        start, count = rowadr[world, row], rownnz[world, row]
        value = float(0.0)
        for entry in range(count):
            address = start + entry
            if colind[world, 0, address] == dofs[joint]: value = jacobian[world, 0, address]
        wp.atomic_add(delta, world, joint, value / weights[joint] * factor)


@wp.kernel
def _apply_collision_delta(qpos: wp.array2d(dtype=wp.float32),
        original: wp.array2d(dtype=wp.float32), qpos_index: wp.array(dtype=wp.int32),
        lower: wp.array(dtype=wp.float32), upper: wp.array(dtype=wp.float32),
        active: wp.array(dtype=wp.int32), delta: wp.array2d(dtype=wp.float32)):
    world, joint = wp.tid()
    if active[world] != 0:
        index = qpos_index[joint]
        low = wp.max(lower[joint], original[world, index]-COLLISION_MAX_CORRECTION)
        high = wp.min(upper[joint], original[world, index]+COLLISION_MAX_CORRECTION)
        qpos[world, index] = wp.clamp(qpos[world, index]
            + wp.clamp(delta[world, joint], -COLLISION_MAX_STEP, COLLISION_MAX_STEP), low, high)


@wp.kernel
def _reject_worse_collision(qpos: wp.array2d(dtype=wp.float32),
        original: wp.array2d(dtype=wp.float32), qpos_index: wp.array(dtype=wp.int32),
        before: wp.array(dtype=wp.float32), after: wp.array(dtype=wp.float32)):
    world = wp.tid()
    if after[world] >= before[world]:
        for joint in range(qpos_index.shape[0]):
            qpos[world, qpos_index[joint]] = original[world, qpos_index[joint]]
        after[world] = before[world]


# FEEDFORWARD GENERALIZED ACCELERATION
# Hermite acceleration exists only in actuator coordinates; unused generalized
# coordinates must stay zero before M(q)*alpha or root forces would be requested.
@wp.kernel
def _set_alpha(acceleration: wp.array3d(dtype=wp.float32), jd: wp.array(dtype=wp.int32),
               step: int, alpha: wp.array2d(dtype=wp.float32)):
    world, actuator = wp.tid()
    alpha[world, jd[actuator]] = acceleration[world, step, actuator]


# ACTUATOR TORQUE REQUEST
# ctrl stores feedforward plus positive PD targets. MuJoCo's affine actuator bias
# supplies -kp*q-kd*qvel, allowing implicitfast to integrate damping implicitly.
# Saturation/utilization must use the resulting total request, not ctrl alone.
@wp.kernel
def _pd(qpos: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32),
        bias: wp.array2d(dtype=wp.float32), passive: wp.array2d(dtype=wp.float32),
        mass_alpha: wp.array2d(dtype=wp.float32), target: wp.array3d(dtype=wp.float32),
        target_vel: wp.array3d(dtype=wp.float32), jq: wp.array(dtype=wp.int32),
        jd: wp.array(dtype=wp.int32), kp: wp.array(dtype=wp.float32),
        kd: wp.array(dtype=wp.float32), low: wp.array(dtype=wp.float32),
        high: wp.array(dtype=wp.float32), active: wp.array(dtype=wp.int32), step: int,
        ctrl: wp.array2d(dtype=wp.float32), saturated: wp.array(dtype=wp.int32),
        utilization: wp.array(dtype=wp.float32)):
    world, actuator = wp.tid()
    q, v = jq[actuator], jd[actuator]
    if active[world] == 0:
        ctrl[world, actuator] = kp[actuator]*qpos[world, q] + kd[actuator]*qvel[world, v]
    else:
        base = (bias[world, v] - passive[world, v] + mass_alpha[world, v]
                + kp[actuator]*target[world, step, actuator]
                + kd[actuator]*target_vel[world, step, actuator])
        request = base - kp[actuator]*qpos[world, q] - kd[actuator]*qvel[world, v]
        clipped = wp.clamp(request, low[actuator], high[actuator])
        ctrl[world, actuator] = base
        if request != clipped:
            wp.atomic_add(saturated, world, 1)
        denominator = wp.where(clipped >= 0.0, high[actuator], -low[actuator])
        wp.atomic_add(utilization, world, wp.abs(clipped) / denominator)


# NETWORK INITIALIZATION
# Actor output weights start at zero but each active output bias starts uniformly
# within +/-15 unscaled reference Std, deliberately giving a poor nonzero curriculum
# start. The critic uses ordinary initialization because it must fit varied returns.
def initialize(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LSTM):
            for name, value in layer.named_parameters():
                if "weight_ih" in name:
                    for gate in value.chunk(4): nn.init.xavier_uniform_(gate)
                elif "weight_hh" in name:
                    for gate in value.chunk(4): nn.init.orthogonal_(gate)
                else:
                    nn.init.zeros_(value)
                    if "bias_ih" in name: value.data[value.numel() // 4:value.numel() // 2] = 1


# FUTURE-GOAL CONTEXT
# Reverse recurrence lets the representation at t summarize G(t+1) through the
# rollout end. Packing is essential: padding must never become invented future goals.
def _reverse_context(encoder, lstm, goals, lengths):
    embedded = encoder(goals)
    t = goals.shape[1]; index = (lengths[:, None] - 1 - torch.arange(t, device=goals.device)).clamp_min(0)
    reverse = embedded.gather(1, index[..., None].expand(-1, -1, embedded.shape[-1]))
    packed = pack_padded_sequence(reverse, lengths.cpu(), batch_first=True, enforce_sorted=False)
    output, _ = lstm(packed)
    output = pad_packed_sequence(output, batch_first=True, total_length=t)[0]
    return embedded, output.gather(1, index[..., None].expand(-1, -1, output.shape[-1]))


# ACTOR / CRITIC
# They share information structure but not parameters or capacity. step() is used
# during live simulation; forward() must reproduce the same sequence for PPO updates.
# Only cumulative-region outputs exist, so inactive regions receive no sampled action
# and no output-row gradient; shared feature/recurrent layers still learn normally.
class RecurrentPolicy(nn.Module):
    def __init__(self, critic=False, initial_bias=None):
        super().__init__(); cur, future, main = ((128, 128, 256) if critic else (256, 256, 512))
        self.current = nn.Sequential(nn.Linear(CURRENT_SIZE, cur), nn.SiLU(),
            nn.Linear(cur, cur), nn.SiLU(), nn.Linear(cur, cur), nn.SiLU())
        self.goal = nn.Sequential(nn.Linear(GOAL_SIZE, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU())
        self.future = nn.LSTM(64, future, batch_first=True)
        self.main = nn.LSTM(cur + future + 64, main, 2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(main, main // 2), nn.SiLU(),
                                  nn.Linear(main // 2, 1 if critic else POLICY_SIZE))
        self.critic = critic; initialize(self)
        if not critic:
            if initial_bias is None or initial_bias.shape != (POLICY_SIZE,): raise ValueError("Actor initial bias mismatch")
            nn.init.zeros_(self.head[-1].weight)
            with torch.no_grad(): self.head[-1].bias.uniform_(-1, 1).mul_(initial_bias)

    def prepare(self, goals, lengths):
        return _reverse_context(self.goal, self.future, goals, lengths)

    def step(self, current, goal_embedding, future, hidden=None):
        value = torch.cat((self.current(current), future, goal_embedding), -1)[:, None]
        output, hidden = self.main(value, hidden)
        return self.head(output[:, 0]), hidden

    def forward(self, current, goals, lengths):
        embedded, future = self.prepare(goals, lengths)
        value = torch.cat((self.current(current), future, embedded), -1)
        packed = pack_padded_sequence(value, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.main(packed)
        output = pad_packed_sequence(output, batch_first=True, total_length=current.shape[1])[0]
        result = self.head(output)
        return result[..., 0] if self.critic else result


# LEARNING FEATURES
# Quaternion 6D avoids the q/-q discontinuity. XY is rollout-relative while Z stays
# absolute; changing either convention changes what balance and floor height mean.
def quaternion_6d(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y),
                        2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), -1)


def model_state(qpos, qvel, bodies, body_velocity, contacts, start_xy, jq, jd, mass):
    root = qpos[..., :3]; rel_body = bodies.clone(); rel_body[..., :2] -= root[..., None, :2]
    relative_velocity = body_velocity - qvel[..., None, :3]
    com = (bodies * mass.view(*((1,) * (bodies.ndim - 2)), -1, 1)).sum(-2)
    com_velocity = (body_velocity * mass.view(*((1,) * (body_velocity.ndim - 2)), -1, 1)).sum(-2)
    return torch.cat((root[..., :2] - start_xy, root[..., 2:3], rel_body[..., :2].flatten(-2),
        bodies[..., 2].flatten(-1), qvel[..., :3], relative_velocity.flatten(-2),
        quaternion_6d(qpos[..., 3:7]), qvel[..., 3:6], qpos[..., jq], qvel[..., jd],
        contacts.float(), com - root, com_velocity), -1)


# VARIABLE HORIZON
# Source and duration are independent. Rounding happens at 40 Hz so every sampled
# horizon maps exactly onto whole policy decisions and ten physics substeps each.
def duration_frames(rng):
    seconds = np.clip(abs(rng.normal(0., DURATION_STD)), DURATION_MIN, DURATION_MAX)
    return int(np.clip(np.rint(seconds * ACTION_HZ), DURATION_MIN * ACTION_HZ, MAX_HORIZON))


# 40 HZ GOAL -> 400 HZ CONTROL TARGET
# Position, velocity, and acceleration come from the same Hermite polynomial.
# Replacing any endpoint velocity with live velocity breaks trajectory consistency.
def hermite(p0, p1, v0, v1, u):
    u2, u3, dt = u*u, u*u*u, 1/ACTION_HZ
    position = ((2*u3-3*u2+1)*p0 + (u3-2*u2+u)*dt*v0
                + (-2*u3+3*u2)*p1 + (u3-u2)*dt*v1)
    velocity = ((6*u2-6*u)/dt*p0 + (3*u2-4*u+1)*v0
                + (-6*u2+6*u)/dt*p1 + (3*u2-2*u)*v1)
    acceleration = ((12*u-6)*(p0-p1)/dt**2 + (6*u-4)*v0/dt + (6*u-2)*v1/dt)
    return position, velocity, acceleration


# CAUSAL RESIDUAL FILTER
# Sampling/PPO remain unsmoothed. Only the combined raw sample sent to physics is a
# uniform current-and-past average; zero history gives an unambiguous rollout boundary.
def smooth_residual(history, current_raw, forced):
    history = torch.cat((history[:, 1:], current_raw[:, None]), 1)
    filtered = history.mean(1)
    return torch.where(forced[..., None], torch.zeros_like(filtered), filtered), history


# Symmetric triangular filtering spreads a collision correction to nearby labels;
# bidirectional step limiting keeps the label continuous and exact-zero at endpoints.
def smooth_collision(delta, lengths):
    radius = COLLISION_SMOOTH_RADIUS
    weights = torch.cat((torch.arange(1, radius+2, device=DEVICE),
                         torch.arange(radius, 0, -1, device=DEVICE))).to(delta)
    kernel = (weights/weights.sum()).view(1, 1, -1).repeat(delta.shape[-1], 1, 1)
    result = nn.functional.conv1d(nn.functional.pad(delta.transpose(1, 2), (radius, radius)),
                                  kernel, groups=delta.shape[-1]).transpose(1, 2)
    frame = torch.arange(delta.shape[1], device=DEVICE)[None]
    internal = (frame > 0) & (frame < lengths[:, None]-1)
    result = torch.where(internal[..., None], result, torch.zeros_like(result))
    for t in range(1, result.shape[1]):
        result[:, t] = torch.where(internal[:, t, None], torch.clamp(result[:, t],
            result[:, t-1]-COLLISION_MAX_TEMPORAL_STEP,
            result[:, t-1]+COLLISION_MAX_TEMPORAL_STEP), torch.zeros_like(result[:, t]))
    for t in range(result.shape[1]-2, -1, -1):
        result[:, t] = torch.where(internal[:, t, None], torch.clamp(result[:, t],
            result[:, t+1]-COLLISION_MAX_TEMPORAL_STEP,
            result[:, t+1]+COLLISION_MAX_TEMPORAL_STEP), torch.zeros_like(result[:, t]))
    return result


# TRAIN / VALIDATION SEPARATION
# A training start is blocked whenever its complete horizon would overlap a protected
# span. Excluding only the start frame would leak validation motion into training.
def validation_exclusions():
    names = {1:CATEGORIES[0], 2:CATEGORIES[1], 3:CATEGORIES[3], 4:CATEGORIES[2]}; blocked = {}
    for code, scene, center in np.load(VALIDATION_PATH / "Validation_Positions.npy"):
        if int(code) == 4: continue
        blocked.setdefault((names[int(code)], int(scene)), []).append((int(center)-2, int(center)+2))
    for code, scene, start, end in np.load(VALIDATION_PATH / "Long_Term_Validation_Positions.npy"):
        if int(code) == 4: continue
        blocked.setdefault((names[int(code)], int(scene)), []).append((int(start), int(end)))
    return blocked


# DATASET MANIFEST AND RAM CACHE
# The manifest binds each flat cache offset to category/scene/length. Its validation
# prevents using a structurally valid but stale cache after Movement_Data is rebuilt.
def _manifest(verbose=True):
    store = zarr.open_group(DATA_PATH, mode="r"); records, total = [], 0
    if verbose: print("Reading Movement_Data scene index...", flush=True)
    keys = [(category, key) for category in CATEGORIES
            for key in sorted(store[category].group_keys(), key=int)]
    next_progress = 5
    if verbose: print(f"Scanning Movement_Data metadata: 0/{len(keys)}", flush=True)
    for done, (category, key) in enumerate(keys, 1):
        group = store[category][key]; length = int(group["qpos"].shape[0])
        if group["qpos"].shape != (length, QPOS_SIZE) or group["qvel"].shape != (length, QVEL_SIZE):
            raise ValueError(f"Malformed Movement_Data/{category}/{key}")
        records.append((category, int(key), total, length)); total += length
        if verbose and 100*done/len(keys) >= next_progress:
            print(f"Scanning Movement_Data metadata: {next_progress}%", flush=True); next_progress += 5
    return store, records, total


def load_cache(verbose=True):
    store, records, total = _manifest(verbose); valid = False
    manifest = np.array([(CATEGORIES.index(category), scene, length)
                         for category, scene, _, length in records], np.int32)
    if CACHE_DATA.is_file() and CACHE_META.is_file():
        try:
            with np.load(CACHE_META) as meta:
                valid = (int(meta["version"]) == 2 and tuple(meta["shape"]) == (total, FRAME_SIZE)
                         and np.array_equal(meta["manifest"], manifest))
            cached = np.load(CACHE_DATA, mmap_mode="r")
            valid &= cached.shape == (total, FRAME_SIZE) and cached.dtype == np.float32
            del cached
        except (KeyError, OSError, ValueError): valid = False
    if not valid:
        temporary = CACHE_DATA.with_suffix(".temporary.npy")
        if verbose: print(f"Building {total:,}-frame movement cache...", flush=True)
        cached = np.lib.format.open_memmap(temporary, "w+", np.float32, (total, FRAME_SIZE))
        next_progress = 5
        for category, scene, offset, length in records:
            group = store[category][str(scene)]
            cached[offset:offset+length, :QPOS_SIZE] = group["qpos"][:]
            cached[offset:offset+length, QPOS_SIZE:] = group["qvel"][:]
            progress = 100*(offset+length)/total
            if verbose and progress >= next_progress:
                print(f"Building movement cache: {progress:.0f}%", flush=True)
                next_progress = 5*(int(progress//5)+1)
        cached.flush(); del cached; os.replace(temporary, CACHE_DATA)
        np.savez(CACHE_META, version=2, shape=np.array((total, FRAME_SIZE)), manifest=manifest)
    if verbose: print("Loading movement cache into CPU RAM...", flush=True)
    data = np.load(CACHE_DATA)
    if verbose: print("Movement cache loaded.", flush=True)
    standing = np.load(DATA_PATH / "Standing_Pose_Pool.npy")
    if standing.ndim != 2 or standing.shape[1] != FRAME_SIZE: raise ValueError("Malformed standing pool")
    by_source = {source: [] for source in CATEGORIES}
    for record in records: by_source[record[0]].append(record)
    return data, standing, by_source


# ROLLOUT SAMPLING
# Sources are uniform, then scenes are uniform within a source; this deliberately
# prevents long scenes/categories from dominating. Standing is a full fifth source.
# Padding rows have horizon 1 only to fill fixed MJWarp world count and are masked.
class Sampler:
    def __init__(self, data, standing, scenes, standing_validation=None):
        self.data, self.standing, self.scenes = data, standing, scenes
        self.standing_validation = standing_validation
        self.blocked = validation_exclusions(); self.valid = {}
        self.lookup = {(record[0], record[1]):record for values in scenes.values() for record in values}

    def add_validation(self, specs):
        for source, scene, start, horizon in specs:
            if source not in ("Standing", "Jumping"):
                self.blocked.setdefault((source, scene), []).append((start, start+horizon))
        self.valid.clear()

    def one(self, rng, source=None, horizon=None, exclude=True):
        source = SOURCES[int(rng.integers(len(SOURCES)))] if source is None else source
        horizon = duration_frames(rng) if horizon is None else horizon
        if source == "Standing": return source, int(rng.integers(len(self.standing))), 0, horizon
        key = (source, horizon, exclude)
        if key not in self.valid:
            choices = []
            for record in self.scenes[source]:
                maximum = record[3]-horizon-1
                if maximum < 0: continue
                forbidden = [] if not exclude else [(max(0, a-horizon), min(maximum, b))
                    for a, b in self.blocked.get((source, record[1]), ()) if a-horizon <= maximum and b >= 0]
                merged = []
                for begin, end in sorted(forbidden):
                    if merged and begin <= merged[-1][1]+1: merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                    else: merged.append((begin, end))
                segments, cursor = [], 0
                for begin, end in merged:
                    if cursor < begin: segments.append((cursor, begin-1))
                    cursor = end+1
                if cursor <= maximum: segments.append((cursor, maximum))
                count = sum(end-begin+1 for begin, end in segments)
                if count: choices.append((record, segments, count))
            self.valid[key] = choices
        choices = self.valid[key]
        if not choices: raise ValueError(f"No valid {source} scene supports {horizon} transitions")
        record, segments, count = choices[int(rng.integers(len(choices)))]
        choice = int(rng.integers(count))
        for begin, end in segments:
            width = end-begin+1
            if choice < width: return source, record[1], begin+choice, horizon
            choice -= width
        raise AssertionError("valid-start selection failed")

    def collect(self, transitions, rng=None):
        if rng is None: rng = RNG
        result, count = [], 0
        while count < transitions:
            spec = self.one(rng); result.append(spec); count += spec[3]
        return result

    def frames(self, specs):
        maximum = max(x[3] for x in specs); output = np.empty((ENVIRONMENT_COUNT, maximum+1, FRAME_SIZE), np.float32)
        lengths = np.zeros(ENVIRONMENT_COUNT, np.int64)
        for row, (source, scene, start, horizon) in enumerate(specs):
            lengths[row] = horizon
            if source == "Standing":
                frame = self.standing[scene] if scene >= 0 else self.standing_validation[-scene-1]
                output[row, :horizon+1] = frame
            else:
                record = self.lookup[(source, scene)]
                output[row, :horizon+1] = self.data[record[2]+start:record[2]+start+horizon+1]
            output[row, horizon+1:] = output[row, horizon]
        for row in range(len(specs), ENVIRONMENT_COUNT):
            output[row] = self.standing[0]; lengths[row] = 1
        return output, lengths


# MUJOCO CONTROL MODEL
# Actuator name order is checked before loading gains. ctrl limits become force limits
# because native affine bias adds feedback after ctrl; ctrl-only clipping would allow
# the actual torque to exceed the intended directional range.
def configure_model():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    if (model.nq, model.nv, model.nu, model.nbody) != (NQ, NV, NU, NBODY+1):
        raise ValueError("Unexpected MuJoCo model dimensions")
    if not math.isclose(model.opt.timestep, PHYSICS_DT): raise ValueError("XML timestep must be 0.0025")
    joints = model.actuator_trnid[:, 0].astype(np.int32)
    jq, jd = model.jnt_qposadr[joints].astype(np.int32), model.jnt_dofadr[joints].astype(np.int32)
    names = np.array([model.joint(int(j)).name for j in joints])
    with np.load(GAINS_PATH) as gain:
        if not np.array_equal(names, gain["joint_names"]): raise ValueError("PD gain order mismatch")
        kp, kd = gain["kp"].astype("f4"), gain["kd"].astype("f4")
    force = model.actuator_ctrlrange.astype("f4").copy()
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.actuator_ctrllimited[:] = 0; model.actuator_forcelimited[:] = 1
    model.actuator_forcerange[:] = force
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE; model.actuator_biasprm[:] = 0
    model.actuator_biasprm[:, 1], model.actuator_biasprm[:, 2] = -kp, -kd
    mass = model.body_mass[1:].astype("f4"); mass /= mass.sum()
    foot = [model.geom(f"{side}_{part}").id for side in ("left", "right") for part in ("foot", "toes_chunk")]
    return model, jq, jd, names, kp, kd, force, mass, (model.geom("floor").id, *foot)


# GPU PHYSICS RUNTIME
# One object owns live worlds and separate kinematic target worlds. Target FK must not
# mutate live state; live qfrc/M/contact must be recomputed at every 400 Hz substep.
# Torch/Warp buffers are shared views, so ordering across their CUDA streams is vital.
class Runtime:
    def __init__(self, model, jq, jd, kp, kd, force, geoms, policy_joint, names):
        self.model, self.wmodel = model, mjw.put_model(model)
        self.data = mjw.make_data(model, nworld=ENVIRONMENT_COUNT, nconmax=NCONMAX, njmax=NJMAX)
        self.target_data = mjw.make_data(model, nworld=ENVIRONMENT_COUNT, nconmax=NCONMAX, njmax=NJMAX)
        if SUPERVISED_PENETRATION and self.target_data.efc.J.shape[1] != 1:
            raise ValueError("GPU collision correction requires MJWarp sparse Jacobians")
        self.jq = torch.as_tensor(jq, dtype=torch.long, device=DEVICE)
        self.jd = torch.as_tensor(jd, dtype=torch.long, device=DEVICE)
        self.wp_jq = wp.array(jq, dtype=wp.int32, device="cuda")
        self.wp_jd = wp.array(jd, dtype=wp.int32, device="cuda")
        self.wp_kp = wp.array(kp, dtype=wp.float32, device="cuda")
        self.wp_kd = wp.array(kd, dtype=wp.float32, device="cuda")
        self.wp_low = wp.array(force[:, 0], dtype=wp.float32, device="cuda")
        self.wp_high = wp.array(force[:, 1], dtype=wp.float32, device="cuda")
        self.geoms = tuple(int(x) for x in geoms)
        policy_joint = np.asarray(policy_joint, np.int32)
        if SUPERVISED_PENETRATION:
            collision_qpos, collision_dof = np.asarray(jq)[policy_joint], np.asarray(jd)[policy_joint]
            collision_range = model.jnt_range[model.actuator_trnid[policy_joint, 0]].astype("f4")
            selected_names = np.asarray(names)[policy_joint]
            weights = np.array([4. if name.startswith(("lumbar", "thoracic", "neck")) else
                2. if any(part in name for part in ("hip", "knee", "ankle", "toes")) else 1.
                for name in selected_names], np.float32)
            self.wp_geom_bodyid = wp.array(model.geom_bodyid.astype(np.int32), dtype=wp.int32, device="cuda")
            self.wp_collision_qpos = wp.array(collision_qpos, dtype=wp.int32, device="cuda")
            self.wp_collision_dof = wp.array(collision_dof, dtype=wp.int32, device="cuda")
            self.wp_collision_low = wp.array(collision_range[:, 0], dtype=wp.float32, device="cuda")
            self.wp_collision_high = wp.array(collision_range[:, 1], dtype=wp.float32, device="cuda")
            self.wp_collision_weights = wp.array(weights, dtype=wp.float32, device="cuda")
        self.initial_qpos = torch.empty(ENVIRONMENT_COUNT, NQ, device=DEVICE)
        self.initial_qvel = torch.empty(ENVIRONMENT_COUNT, NV, device=DEVICE)
        self.target_qpos = torch.empty_like(self.initial_qpos); self.target_qvel = torch.empty_like(self.initial_qvel)
        self.targets = torch.empty(ENVIRONMENT_COUNT, SUBSTEPS, NU, device=DEVICE)
        self.target_velocities = torch.empty_like(self.targets)
        self.accelerations = torch.empty_like(self.targets)
        self.alpha = torch.zeros(ENVIRONMENT_COUNT, NV, device=DEVICE)
        self.mass_alpha = torch.empty_like(self.alpha)
        self.active = torch.zeros(ENVIRONMENT_COUNT, dtype=torch.int32, device=DEVICE)
        self.saturated = torch.zeros_like(self.active)
        self.utilization = torch.zeros(ENVIRONMENT_COUNT, device=DEVICE)
        self.contact = torch.zeros(ENVIRONMENT_COUNT, 2, dtype=torch.int32, device=DEVICE)
        self.depth = torch.zeros(ENVIRONMENT_COUNT, device=DEVICE)
        self.self_depth = torch.zeros_like(self.depth)
        collision_names = ()
        if SUPERVISED_PENETRATION:
            self.collision_original = torch.empty_like(self.initial_qpos)
            self.collision_active = torch.zeros_like(self.active)
            self.collision_delta = torch.zeros(ENVIRONMENT_COUNT, POLICY_NU, device=DEVICE)
            self.collision_scale = torch.zeros(self.target_data.contact.dist.shape[0], device=DEVICE)
            self.collision_before = torch.zeros_like(self.depth); self.collision_after = torch.zeros_like(self.depth)
            collision_names = ("collision_original", "collision_active", "collision_delta",
                               "collision_scale", "collision_before", "collision_after")
        for name in ("initial_qpos", "initial_qvel", "target_qpos", "target_qvel", "targets",
                     "target_velocities", "accelerations", "alpha", "mass_alpha", "active", "saturated",
                     "utilization", "contact", "depth", "self_depth") + collision_names:
            setattr(self, "wp_" + name, wp.from_torch(getattr(self, name)))
        self.qpos = wp.to_torch(self.data.qpos); self.qvel = wp.to_torch(self.data.qvel)
        self.xipos = wp.to_torch(self.data.xipos)
        self.cvel = wp.to_torch(self.data.cvel); self.subtree_com = wp.to_torch(self.data.subtree_com)
        self.bias = wp.to_torch(self.data.qfrc_bias); self.passive = wp.to_torch(self.data.qfrc_passive)
        self.ctrl = wp.to_torch(self.data.ctrl); self.nefc = wp.to_torch(self.data.nefc)
        self.nacon = wp.to_torch(self.data.nacon); self.overflow = wp.to_torch(self.data.overflow)
        self.solver_niter = wp.to_torch(self.data.solver_niter)
        self.target_xipos = wp.to_torch(self.target_data.xipos)
        self.target_state_qpos = wp.to_torch(self.target_data.qpos)
        self.target_cvel = wp.to_torch(self.target_data.cvel)
        self.target_subtree_com = wp.to_torch(self.target_data.subtree_com)
        self.torch_stream = torch.cuda.Stream(device=DEVICE)
        self.stream = wp.stream_from_torch(self.torch_stream)
        self.debug, self.debug_specs, self.debug_transition, self.debug_last = False, (), -1, {}
        wp.set_stream(self.stream); self._capture(); self.debug = CUDA_PHYSICS_DEBUG
        if self.debug:
            wp.config.verify_cuda = True
            print("CUDA_PHYSICS_DEBUG: transition graph disabled; every physics stage is synchronized.", flush=True)
        elif not TRANSITION_CUDA_GRAPH:
            print("Transition CUDA graph disabled; physics stages launch directly and asynchronously.", flush=True)

    # Contact arrays are reused by captured graphs; stale flags would make contact
    # observations and penetration rewards persist after contact has ended.
    def _clear_contacts(self):
        self.wp_contact.zero_(); self.wp_depth.zero_(); self.wp_self_depth.zero_()

    def _contact_kernel(self, data):
        wp.launch(_contacts, dim=data.contact.dist.shape[0], inputs=[data.nacon, data.contact.geom,
            data.contact.worldid, data.contact.dist, *self.geoms, self.wp_contact, self.wp_depth,
            self.wp_self_depth])

    def _reset_ops(self):
        mjw.reset_data(self.wmodel, self.data)
        wp.copy(self.data.qpos, self.wp_initial_qpos); wp.copy(self.data.qvel, self.wp_initial_qvel)
        self._clear_contacts(); mjw.forward(self.wmodel, self.data); self._contact_kernel(self.data)

    def _target_ops(self):
        wp.copy(self.target_data.qpos, self.wp_target_qpos); wp.copy(self.target_data.qvel, self.wp_target_qvel)
        self._clear_contacts(); mjw.forward(self.wmodel, self.target_data); self._contact_kernel(self.target_data)

    def _correction_ops(self):
        wp.copy(self.target_data.qpos, self.wp_target_qpos); wp.copy(self.target_data.qvel, self.wp_target_qvel)
        wp.copy(self.wp_collision_original, self.wp_target_qpos)
        contacts = self.target_data.contact.dist.shape[0]
        for collision_pass in range(COLLISION_PASSES):
            self._clear_contacts(); mjw.forward(self.wmodel, self.target_data)
            if collision_pass == 0:
                self._contact_kernel(self.target_data); wp.copy(self.wp_collision_before, self.wp_self_depth)
            wp.launch(_collision_scale, dim=contacts, inputs=[self.target_data.nacon,
                self.target_data.contact.geom, self.target_data.contact.worldid,
                self.target_data.contact.dist, self.target_data.contact.efc_address,
                self.wp_geom_bodyid, self.target_data.efc.J_rownnz, self.target_data.efc.J_rowadr,
                self.target_data.efc.J_colind, self.target_data.efc.J, self.wp_collision_dof,
                self.wp_collision_weights, self.wp_collision_active, self.wp_collision_scale])
            self.wp_collision_delta.zero_()
            wp.launch(_collision_delta, dim=(contacts, POLICY_NU), inputs=[
                self.target_data.contact.worldid, self.target_data.contact.efc_address,
                self.target_data.efc.J_rownnz, self.target_data.efc.J_rowadr,
                self.target_data.efc.J_colind, self.target_data.efc.J, self.wp_collision_dof,
                self.wp_collision_weights, self.wp_collision_scale, self.wp_collision_delta])
            wp.launch(_apply_collision_delta, dim=(ENVIRONMENT_COUNT, POLICY_NU), inputs=[
                self.target_data.qpos, self.wp_collision_original, self.wp_collision_qpos,
                self.wp_collision_low, self.wp_collision_high, self.wp_collision_active,
                self.wp_collision_delta])
        self._clear_contacts(); mjw.forward(self.wmodel, self.target_data); self._contact_kernel(self.target_data)
        wp.copy(self.wp_collision_after, self.wp_self_depth)
        wp.launch(_reject_worse_collision, dim=ENVIRONMENT_COUNT, inputs=[self.target_data.qpos,
            self.wp_collision_original, self.wp_collision_qpos, self.wp_collision_before,
            self.wp_collision_after])

    def debug_context(self, specs, transition):
        self.debug_specs, self.debug_transition = specs, transition

    def _debug_message(self, operation, step, world=None):
        spec = self.debug_specs[world] if world is not None and world < len(self.debug_specs) else None
        context = (f"source={spec[0]} scene={spec[1]} start={spec[2]}" if spec else "source=unknown")
        values = " ".join(f"{key}={value}" for key, value in self.debug_last.items())
        return (f"CUDA PHYSICS DEBUG FAILURE: operation={operation} world={world} "
                f"transition={self.debug_transition} substep={step} {context}\n{values}")

    def _debug_check(self, operation, step):
        wp.synchronize_stream(self.stream)
        arrays = {"qpos":self.qpos, "qvel":self.qvel, "target":self.targets,
                  "target_vel":self.target_velocities, "acceleration":self.accelerations,
                  "mass_alpha":self.mass_alpha, "bias":self.bias, "passive":self.passive,
                  "ctrl":self.ctrl}
        bad_world, maxima = None, {}
        for name, value in arrays.items():
            finite = torch.isfinite(value); valid = value[finite]
            maxima[name] = f"{float(valid.abs().max()) if valid.numel() else math.inf:.6g}"
            invalid = ~finite.reshape(ENVIRONMENT_COUNT, -1).all(1)
            if bad_world is None and invalid.any(): bad_world = int(invalid.nonzero()[0])
        overflow = self.overflow.detach().cpu().numpy(); nefc = self.nefc.detach().cpu().numpy()
        nacon, solver = int(self.nacon[0]), int(self.solver_niter.max())
        overflow_world = np.flatnonzero(overflow)
        if bad_world is None and len(overflow_world): bad_world = int(overflow_world[0])
        if bad_world is None and np.max(nefc) > self.data.njmax: bad_world = int(np.argmax(nefc))
        flags = ((1,"NEFC"),(2,"NJMAX_NNZ"),(4,"BROADPHASE"),(8,"NARROWPHASE"),
                 (16,"CCD"),(32,"HFIELD"),(64,"CONTACT_MATCH"),(128,"NVMAX"),(256,"EPA_HORIZON"))
        bits = int(overflow[bad_world]) if bad_world is not None else 0
        self.debug_last = {"max_nefc":int(np.max(nefc)), "njmax":self.data.njmax,
            "contacts":nacon, "naconmax":self.data.naconmax, "solver_iterations":solver,
            "overflow":"|".join(name for bit, name in flags if bits & bit) or "none", **maxima}
        if bad_world is not None or nacon > self.data.naconmax:
            raise RuntimeError(self._debug_message(operation, step, bad_world))

    def _debug_call(self, operation, step, function):
        try:
            function(); self._debug_check(operation, step)
        except Exception as error:
            if isinstance(error, RuntimeError) and str(error).startswith("CUDA PHYSICS DEBUG FAILURE"):
                raise
            raise RuntimeError(self._debug_message(operation, step)) from error

    # This is the physical meaning of one 40 Hz action: ten live-feedback steps.
    # M(q)*alpha and bias/passive terms are refreshed, while Hermite targets are fixed
    # by the two altered knots. Holding torque across these steps would be incorrect.
    def _transition_ops(self):
        self.wp_saturated.zero_(); self.wp_utilization.zero_()
        for step in range(SUBSTEPS):
            forward = lambda: mjw.forward(self.wmodel, self.data)
            set_alpha = lambda: wp.launch(_set_alpha, dim=(ENVIRONMENT_COUNT, NU),
                inputs=[self.wp_accelerations, self.wp_jd, step, self.wp_alpha])
            mul_m = lambda: mjw.mul_m(self.wmodel, self.data, self.wp_mass_alpha, self.wp_alpha)
            pd = lambda: wp.launch(_pd, dim=(ENVIRONMENT_COUNT, NU), inputs=[self.data.qpos, self.data.qvel,
                self.data.qfrc_bias, self.data.qfrc_passive, self.wp_mass_alpha, self.wp_targets,
                self.wp_target_velocities, self.wp_jq, self.wp_jd, self.wp_kp, self.wp_kd,
                self.wp_low, self.wp_high, self.wp_active, step, self.data.ctrl,
                self.wp_saturated, self.wp_utilization])
            advance = lambda: mjw.step(self.wmodel, self.data)
            if self.debug:
                for name, function in (("mjw.forward",forward),("set_alpha",set_alpha),
                        ("mjw.mul_m",mul_m),("pd",pd),("mjw.step",advance)):
                    self._debug_call(name, step, function)
            else:
                forward(); set_alpha(); mul_m(); pd(); advance()
        if self.debug:
            self._debug_call("clear_contacts", SUBSTEPS, self._clear_contacts)
            self._debug_call("final_mjw.forward", SUBSTEPS, lambda: mjw.forward(self.wmodel, self.data))
            self._debug_call("contact_kernel", SUBSTEPS, lambda: self._contact_kernel(self.data))
        else:
            self._clear_contacts(); mjw.forward(self.wmodel, self.data); self._contact_kernel(self.data)

    # Capture freezes operation topology, not buffer values. Every tensor copied into
    # these buffers must retain shape/dtype/address or graph replay becomes invalid.
    def _capture(self):
        self.initial_qpos.copy_(torch.as_tensor(self.model.qpos0, device=DEVICE).expand_as(self.initial_qpos))
        self.initial_qvel.zero_(); self.target_qpos.copy_(self.initial_qpos); self.target_qvel.zero_()
        self.targets.zero_(); self.target_velocities.zero_(); self.accelerations.zero_()
        self.active.zero_(); self.alpha.zero_()
        if SUPERVISED_PENETRATION: self.collision_active.zero_()
        self.stream.wait_stream(wp.stream_from_torch(torch.cuda.current_stream()))
        self._reset_ops(); self._target_ops()
        if SUPERVISED_PENETRATION: self._correction_ops()
        self._transition_ops(); wp.synchronize_stream(self.stream)
        with wp.ScopedCapture(device="cuda", stream=self.stream, force_module_load=False) as capture:
            self._reset_ops()
        self.reset_graph = capture.graph
        with wp.ScopedCapture(device="cuda", stream=self.stream, force_module_load=False) as capture:
            self._target_ops()
        self.target_graph = capture.graph
        self.correction_graph = None
        if SUPERVISED_PENETRATION:
            with wp.ScopedCapture(device="cuda", stream=self.stream, force_module_load=False) as capture:
                self._correction_ops()
            self.correction_graph = capture.graph
        self.transition_graph = None
        if TRANSITION_CUDA_GRAPH and not CUDA_PHYSICS_DEBUG:
            with wp.ScopedCapture(device="cuda", stream=self.stream, force_module_load=False) as capture:
                self._transition_ops()
            self.transition_graph = capture.graph

    # Explicit cross-stream waits prevent Torch from reading a partially completed
    # Warp rollout or Warp from consuming inputs still being written by Torch.
    def _launch(self, graph):
        self.stream.wait_stream(wp.stream_from_torch(torch.cuda.current_stream()))
        if graph is None:
            with wp.ScopedStream(self.stream): self._transition_ops()
        else: wp.capture_launch(graph, stream=self.stream)
        torch.cuda.current_stream().wait_stream(self.torch_stream)

    def reset(self, qpos, qvel):
        self.initial_qpos.copy_(qpos); self.initial_qvel.copy_(qvel); self._launch(self.reset_graph)

    def target(self, qpos, qvel):
        self.target_qpos.copy_(qpos); self.target_qvel.copy_(qvel); self._launch(self.target_graph)
        body_velocity = (self.target_cvel[:, 1:, 3:] + torch.linalg.cross(
            self.target_cvel[:, 1:, :3], self.target_xipos[:, 1:]
            - self.target_subtree_com[:, 1:2], dim=-1))
        return self.target_xipos[:, 1:].clone(), body_velocity.clone(), self.contact.clone(), self.depth.clone()

    def collision_correct(self, qpos, qvel, active):
        self.target_qpos.copy_(qpos); self.target_qvel.copy_(qvel); self.collision_active.copy_(active.int())
        self._launch(self.correction_graph)
        return (self.target_state_qpos.clone(), self.collision_before.clone(),
                self.collision_after.clone())

    # Endpoint velocities are altered policy goals, not the simulated live qvel.
    # The returned saturation/utilization are averages over all actuators/substeps.
    def transition(self, p0, p1, v0, v1, active):
        u = torch.arange(1, SUBSTEPS+1, device=DEVICE, dtype=p0.dtype).view(1, -1, 1) / SUBSTEPS
        target, velocity, acceleration = hermite(p0[:, None], p1[:, None], v0[:, None], v1[:, None], u)
        self.targets.copy_(target); self.target_velocities.copy_(velocity); self.accelerations.copy_(acceleration)
        self.alpha.zero_(); self.active.copy_(active.int())
        if self.debug:
            self.stream.wait_stream(wp.stream_from_torch(torch.cuda.current_stream()))
            self._debug_check("transition_inputs", -1)
        self._launch(self.transition_graph)
        return (self.xipos[:, 1:].clone(), self.qpos[:, :7].clone(),
                self.saturated.float() / (SUBSTEPS*NU), self.utilization / (SUBSTEPS*NU))


# IMMUTABLE FEATURE NORMALIZATION
# Statistics describe zero-residual physics, while previous-residual channels use
# action Std by construction. Updating these statistics during PPO would make old
# rollout log-probabilities and checkpoint behavior nonstationary.
class Normalizer:
    def __init__(self, current_mean, current_std, goal_mean, goal_std):
        self.current_mean = torch.as_tensor(current_mean, device=DEVICE)
        self.current_std = torch.as_tensor(current_std, device=DEVICE)
        self.goal_mean = torch.as_tensor(goal_mean, device=DEVICE)
        self.goal_std = torch.as_tensor(goal_std, device=DEVICE)

    def current(self, value): return (value - self.current_mean) / self.current_std
    def goal(self, value): return (value - self.goal_mean) / self.goal_std


# ORIGINAL-GOAL FEATURES
# Stored body XYZ/velocity is the reference motion, while contact is recomputed from
# original qpos/qvel under the current XML. Time features distinguish identical poses
# that require different actions because different rollout time remains.
def raw_goal_features(frames, contacts, lengths, jq, jd, mass):
    qpos, qvel = frames[..., :NQ], frames[..., QPOS_SIZE:QPOS_SIZE+NV]
    bodies = frames[..., NQ:QPOS_SIZE].reshape(*frames.shape[:2], NBODY, 3)
    body_velocity = frames[..., QPOS_SIZE+NV:].reshape(*frames.shape[:2], NBODY, 3)
    state = model_state(qpos, qvel, bodies, body_velocity, contacts,
                        qpos[:, :1, :2], jq, jd, mass)
    t = torch.arange(frames.shape[1], device=frames.device)[None]
    time_feature = torch.stack(((lengths[:, None]-t)/lengths[:, None], t/lengths[:, None]), -1)
    return torch.cat((state, time_feature), -1)


# LIVE-CURRENT FEATURES
# The actor sees physical live state, all-body error to the unaltered reference,
# current generalized bias/passive load, and previous effective residual. Feeding raw
# pre-clipped residual here would tell the network an action physics never received.
def raw_current_feature(runtime, original_bodies, body_velocity, contact, start_xy,
                        goal_index, lengths, previous_residual, jq, jd, mass):
    state = model_state(runtime.qpos, runtime.qvel, runtime.xipos[:, 1:], body_velocity,
                        contact, start_xy, jq, jd, mass)
    error = (runtime.xipos[:, 1:] - original_bodies).flatten(1)
    time_feature = torch.stack(((lengths-goal_index)/lengths,
                                torch.full_like(lengths, goal_index)/lengths), -1)
    actuator = runtime.bias[:, jd] - runtime.passive[:, jd]
    return torch.cat((state, error, time_feature, actuator, previous_residual), -1)


# FIXED VALIDATION BANK
# Motion and standing spans are protected before training sampling begins. Validation
# is deterministic actor mean, so changes measure policy drift rather than fresh noise.
def load_validation(sampler):
    names = {1:CATEGORIES[0], 2:CATEGORIES[1], 3:CATEGORIES[3], 4:CATEGORIES[2]}
    positions = np.load(VALIDATION_PATH / "Long_Term_Validation_Positions.npy")
    if positions.shape != (2400, 4): raise ValueError("Malformed long-term validation positions")
    specs = [(names[int(code)], int(scene), int(start), int(end-start)) for code, scene, start, end in positions]
    with np.load(VALIDATION_PATH / "Standing_Long_Term_Validation.npz") as saved:
        if saved["endpoints"].shape[0] != 600 or saved["total_physics_steps"].shape != (600,):
            raise ValueError("Malformed standing validation")
        sampler.standing_validation = saved["endpoints"][:, 0].copy()
        horizons = np.ceil(saved["total_physics_steps"] / SUBSTEPS).astype(int)
    specs += [("Standing", -index-1, 0, int(horizon)) for index, horizon in enumerate(horizons)]
    sampler.add_validation(specs); return specs


# FIXED-WORLD PACKING
# Sorting by horizon groups similar lengths and reduces masked simulation after short
# rollouts end. Record order is irrelevant because recurrent state resets per rollout.
def chunk_specs(specs):
    ordered = sorted(specs, key=lambda x: x[3])
    return [ordered[i:i+ENVIRONMENT_COUNT] for i in range(0, len(ordered), ENVIRONMENT_COUNT)]


@torch.no_grad()
def rollout_chunk(specs, sampler, runtime, actor, critic, normalizer, action_std,
                  joint_low, joint_high, policy_joint, jq, jd, mass, deterministic=False,
                  normalization=False, calibration=False):
    # REFERENCE PRECOMPUTATION
    # Keep stored body trajectories for tracking, but recompute original target body
    # velocities/contact from qpos/qvel so altered and original FK share one XML.
    frames_np, lengths_np = sampler.frames(specs)
    frames = torch.from_numpy(frames_np).to(DEVICE); lengths = torch.from_numpy(lengths_np).to(DEVICE)
    maximum = int(lengths.max()); qpos = frames[..., :NQ]; qvel = frames[..., QPOS_SIZE:QPOS_SIZE+NV]
    original_bodies = frames[..., NQ:QPOS_SIZE].reshape(ENVIRONMENT_COUNT, maximum+1, NBODY, 3)
    original_body_velocity = frames[..., QPOS_SIZE+NV:].reshape(ENVIRONMENT_COUNT, maximum+1, NBODY, 3)
    goal_contacts = torch.zeros(ENVIRONMENT_COUNT, maximum+1, 2, dtype=torch.int32, device=DEVICE)
    goal_body_velocity = torch.empty_like(original_body_velocity)
    for t in range(maximum+1):
        _, goal_body_velocity[:, t], goal_contacts[:, t], _ = runtime.target(qpos[:, t], qvel[:, t])
    raw_goals = raw_goal_features(frames, goal_contacts, lengths, jq, jd, mass)
    next_goals = raw_goals[:, 1:]
    # Actor and critic future encoders are separate. Their reverse contexts are built
    # once because original goals never change during an online rollout.
    if normalizer:
        next_goals = normalizer.goal(next_goals)
        actor_goal, actor_future = actor.prepare(next_goals, lengths)
        critic_goal, critic_future = critic.prepare(next_goals, lengths)
    # CAUSAL ROLLOUT STATE
    # Live physics starts at the recorded qpos/qvel. effective histories drive
    # controller knot continuity/smoothness; raw history drives the causal filter.
    # Actor outputs and stochastic innovations have separate zero-initialized carries.
    runtime.reset(qpos[:, 0], qvel[:, 0])
    previous_body = runtime.xipos[:, 1:].clone()
    current_body_velocity = original_body_velocity[:, 0].clone()
    previous_previous = torch.zeros(ENVIRONMENT_COUNT, ACTION_SIZE, device=DEVICE)
    previous = torch.zeros_like(previous_previous)
    raw_history = torch.zeros(ENVIRONMENT_COUNT, SMOOTH, POLICY_SIZE, device=DEVICE)
    mean_history = torch.zeros_like(raw_history) if SUPERVISED_PENETRATION else None
    mean_carry = torch.zeros(ENVIRONMENT_COUNT, POLICY_SIZE, device=DEVICE)
    noise_carry = torch.zeros(ENVIRONMENT_COUNT, POLICY_SIZE, device=DEVICE)
    fallen = torch.zeros(ENVIRONMENT_COUNT, dtype=torch.bool, device=DEVICE)
    actor_hidden = critic_hidden = None
    current_list, action_list, logp_list, value_list, effective_list, component_list = [], [], [], [], [], []
    altered_tracking, clip_list, penetration_list, fall_list = [], [], [], []
    goal_position_list, goal_velocity_list = [], []
    raw_current_list, mean_joint_list, correction_list, collision_before_list, base_joint_list = [], [], [], [], []
    for t in range(maximum):
        # Padded worlds are inactive. The last real action is forced to zero so a
        # rollout cannot finish with an unobserved residual discontinuity.
        active = t < lengths
        forced = (t == lengths-1) | ~active
        # OBSERVE, VALUE, AND SAMPLE
        # The actor emits an increment. Its mean carry and the sampled-noise carry
        # accumulate independently from zero before filtering and the sine envelope.
        raw_current = raw_current_feature(runtime, original_bodies[:, t], current_body_velocity,
            runtime.contact, qpos[:, 0, :2], t, lengths, previous, jq, jd, mass)
        raw_current_list.append(raw_current)
        if normalizer:
            current = normalizer.current(raw_current); current_list.append(current)
            mean_increment, actor_hidden = actor.step(current, actor_goal[:, t], actor_future[:, t], actor_hidden)
            value, critic_hidden = critic.step(current, critic_goal[:, t], critic_future[:, t], critic_hidden)
            value = value[:, 0]
            mean_carry = mean_carry + mean_increment
            mean = mean_carry
            noise_offset = torch.zeros_like(noise_carry) if deterministic else noise_carry
            distribution = torch.distributions.Normal(mean + noise_offset, actor.action_std)
            raw_action = mean if deterministic else distribution.sample()
            if not deterministic: noise_carry = raw_action - mean
            logp = distribution.log_prob(raw_action).sum(-1)
        elif calibration:
            std = torch.cat((action_std[:NU][policy_joint], action_std[NU:][policy_joint]))
            noise_offset = noise_carry
            raw_action = noise_offset + torch.randn_like(noise_carry)*std
            noise_carry = raw_action
            logp = value = torch.zeros(ENVIRONMENT_COUNT, device=DEVICE)
        else:
            noise_offset = torch.zeros_like(noise_carry)
            raw_action = torch.zeros(ENVIRONMENT_COUNT, POLICY_SIZE, device=DEVICE)
            logp = value = torch.zeros(ENVIRONMENT_COUNT, device=DEVICE)
        # ACTION TRANSFORM
        # Score the raw action, then apply causal smoothing and the sine boundary.
        # Only cumulative-region outputs are scattered into the full residual; every
        # inactive position/velocity slot is exact zero and has no policy distribution.
        # Position is clipped by XML joint range; velocity deliberately remains free.
        # effective records what survived clipping and is the only residual used by
        # physics continuity, observations, smoothness, and subsequent rewards.
        raw_action = torch.where(forced[:, None], torch.zeros_like(raw_action), raw_action)
        mean = torch.where(forced[:, None], torch.zeros_like(mean), mean) if normalizer else raw_action
        noise_offset = torch.where(forced[:, None], torch.zeros_like(noise_offset), noise_offset)
        logp = torch.where(forced, torch.zeros_like(logp), logp)
        filtered_policy, raw_history = smooth_residual(raw_history, raw_action, forced)
        envelope = torch.sin(math.pi*t/(lengths-1).clamp_min(1))[:, None]
        filtered_policy *= envelope
        if SUPERVISED_PENETRATION and normalizer and not deterministic:
            filtered_mean, mean_history = smooth_residual(mean_history, mean, forced)
            mean_position = (filtered_mean*envelope)[:, :POLICY_NU]
            mean_joint = torch.maximum(torch.minimum(qpos[:, t+1, jq][:, policy_joint] + mean_position,
                joint_high[policy_joint]), joint_low[policy_joint])
            mean_qpos = qpos[:, t+1].clone(); mean_qpos[:, jq[policy_joint]] = mean_joint
            corrected_qpos, collision_before, _ = runtime.collision_correct(
                mean_qpos, qvel[:, t+1], active & ~forced)
            mean_joint_list.append(mean_joint)
            correction_list.append(corrected_qpos[:, jq[policy_joint]]-mean_joint)
            collision_before_list.append(collision_before)
            base_joint_list.append(qpos[:, t+1, jq][:, policy_joint])
        policy_position, policy_velocity = filtered_policy.split(POLICY_NU, -1)
        position_residual = torch.zeros(ENVIRONMENT_COUNT, NU, device=DEVICE)
        velocity_residual = torch.zeros_like(position_residual)
        position_residual[:, policy_joint] = policy_position
        velocity_residual[:, policy_joint] = policy_velocity
        raw_target = qpos[:, t+1, jq] + position_residual
        altered_joint = torch.maximum(torch.minimum(raw_target, joint_high), joint_low)
        effective_position = altered_joint - qpos[:, t+1, jq]
        effective = torch.cat((effective_position, velocity_residual), -1)
        effective = torch.where(active[:, None], effective, torch.zeros_like(effective))
        altered_qpos = qpos[:, t+1].clone(); altered_qpos[:, jq] = altered_joint
        altered_qvel = qvel[:, t+1].clone(); altered_qvel[:, jd] += velocity_residual
        # ALTERED TARGET AND LIVE TRANSITION
        # Target FK measures what was proposed; live transition measures what limited
        # actuators/contact actually achieved. p0/v0 use the previous altered knot,
        # while p1/v1 use this one, keeping Hermite interpolation continuous.
        altered_bodies, altered_body_velocity, _, depth = runtime.target(altered_qpos, altered_qvel)
        p0 = qpos[:, t, jq] + previous[:, :NU]; p1 = altered_joint
        v0 = qvel[:, t, jd] + previous[:, NU:]; v1 = altered_qvel[:, jd]
        if CUDA_PHYSICS_DEBUG: runtime.debug_context(specs, t)
        actual_bodies, actual_root, saturation, utilization = runtime.transition(p0, p1, v0, v1, active)
        # REWARD COMPONENTS
        # Tracking compares live motion to original motion. Goal difference constrains
        # how far the policy rewrites that reference. Smoothness acts on the applied
        # normalized 94-vector; torque terms penalize physical infeasibility. Target
        # penetration is diagnostic only; self-collision learning is supervised.
        # These raw scales are coupled to the calibrated coefficients above.
        actual_velocity = (actual_bodies - previous_body) * ACTION_HZ
        reference_velocity = (original_bodies[:, t+1] - original_bodies[:, t]) * ACTION_HZ
        position_distance = torch.linalg.vector_norm(actual_bodies-original_bodies[:, t+1], dim=-1).mean(-1)
        velocity_distance = torch.linalg.vector_norm(actual_velocity-reference_velocity, dim=-1).mean(-1)
        goal_position_distance = torch.linalg.vector_norm(
            altered_bodies-original_bodies[:, t+1], dim=-1).mean(-1)
        goal_velocity_distance = torch.linalg.vector_norm(
            altered_body_velocity-goal_body_velocity[:, t+1], dim=-1).mean(-1)
        goal_difference = (.75*goal_position_distance.square()/DIFFERENCE_POSITION_MEAN
                           + .25*goal_velocity_distance.square()/DIFFERENCE_VELOCITY_MEAN)
        smooth = (((effective-2*previous+previous_previous)/action_std).square().sum(-1)
                  if t >= 2 else torch.zeros(ENVIRONMENT_COUNT, device=DEVICE))
        depth = torch.where(depth > .001, depth, torch.zeros_like(depth))
        depth = torch.where(forced, torch.zeros_like(depth), depth)
        actual_quaternion = actual_root[:, 3:7] / torch.linalg.vector_norm(
            actual_root[:, 3:7], dim=-1, keepdim=True).clamp_min(EPS)
        goal_quaternion = qpos[:, t+1, 3:7] / torch.linalg.vector_norm(
            qpos[:, t+1, 3:7], dim=-1, keepdim=True).clamp_min(EPS)
        root_angle = 2*torch.acos(torch.abs((actual_quaternion*goal_quaternion).sum(-1)).clamp_max(1))
        height_drop = qpos[:, t+1, 2] - actual_root[:, 2]
        invalid_root = ~torch.isfinite(actual_root).all(-1)
        fall = active & ~fallen & (invalid_root | (height_drop >= FALL_HEIGHT_DROP)
            | ((height_drop >= FALL_TILT_HEIGHT_DROP) & (root_angle >= FALL_ANGLE)))
        fallen |= fall
        components = torch.stack((position_distance.square(), velocity_distance.square(),
            goal_difference, smooth, saturation, utilization, depth), -1)
        components *= active[:, None]
        # Store raw sampled actions/log-probabilities for PPO, not filtered actions.
        # Diagnostics retain effective outcomes so policy statistics are not confused
        # with joint clipping, torque clipping, or target-vs-live tracking failure.
        if normalizer:
            action_list.append(raw_action); logp_list.append(logp); value_list.append(value)
            effective_list.append(effective); component_list.append(components)
        if normalizer:
            goal_position_list.append(goal_position_distance)
            goal_velocity_list.append(goal_velocity_distance)
            altered_tracking.append(torch.linalg.vector_norm(actual_bodies-altered_bodies, dim=-1).mean(-1))
            clip_list.append((raw_target != altered_joint).float().mean(-1)*active)
            penetration_list.append((depth > 0).float()*active)
        elif calibration:
            component_list.append(components); goal_position_list.append(goal_position_distance)
            goal_velocity_list.append(goal_velocity_distance)
        if normalizer or calibration: fall_list.append(fall.float())
        # Advance all carriers only after this transition and reward are complete.
        # Reordering these assignments shifts residuals/body velocity by one goal.
        previous_previous, previous = previous, effective
        previous_body, current_body_velocity = actual_bodies, actual_velocity
    # CPU ROLLOUT RECORDS
    # Slice each padded world back to its true horizon before returns/minibatches.
    # Reward is negative cost; accidentally changing this sign reverses PPO learning.
    if normalization:
        return torch.stack(raw_current_list, 1).cpu(), raw_goals[:, 1:].cpu()
    if calibration:
        components = torch.stack(component_list, 1).cpu(); falls = torch.stack(fall_list, 1).cpu()
        goal_positions = torch.stack(goal_position_list, 1).cpu()
        goal_velocities = torch.stack(goal_velocity_list, 1).cpu()
        return [{"components":components[row, :spec[3]], "goal_position":goal_positions[row, :spec[3]],
                 "goal_velocity":goal_velocities[row, :spec[3]], "fall":falls[row, :spec[3]]}
                for row, spec in enumerate(specs)]
    current = torch.stack(current_list, 1).cpu(); action = torch.stack(action_list, 1).cpu()
    logp, values = torch.stack(logp_list, 1).cpu(), torch.stack(value_list, 1).cpu()
    effective, components = torch.stack(effective_list, 1).cpu(), torch.stack(component_list, 1).cpu()
    altered_tracking = torch.stack(altered_tracking, 1).cpu(); clips = torch.stack(clip_list, 1).cpu()
    penetrations = torch.stack(penetration_list, 1).cpu()
    falls = torch.stack(fall_list, 1).cpu()
    goal_positions = torch.stack(goal_position_list, 1).cpu()
    goal_velocities = torch.stack(goal_velocity_list, 1).cpu()
    rewards = -(components * torch.as_tensor(REWARD_COEFFICIENTS)).sum(-1) - FALL_COST*falls
    next_goals = next_goals.cpu()
    collision_target = collision_mask = base_joint = None
    if SUPERVISED_PENETRATION and not deterministic:
        mean_joint = torch.stack(mean_joint_list, 1)
        correction = smooth_collision(torch.stack(correction_list, 1), lengths)
        corrected_joint = torch.maximum(torch.minimum(mean_joint+correction,
            joint_high[policy_joint]), joint_low[policy_joint])
        corrected_depth = []
        for t in range(maximum):
            corrected_qpos = qpos[:, t+1].clone(); corrected_qpos[:, jq[policy_joint]] = corrected_joint[:, t]
            runtime.target(corrected_qpos, qvel[:, t+1]); corrected_depth.append(runtime.self_depth.clone())
        collision_before, corrected_depth = torch.stack(collision_before_list, 1), torch.stack(corrected_depth, 1)
        base_joint = torch.stack(base_joint_list, 1)
        collision_target = corrected_joint-base_joint
        collision_mask = ((corrected_joint-mean_joint).abs().amax(-1) > 1e-7) \
            & (corrected_depth < collision_before-1e-6)
        base_joint, collision_target, collision_mask = base_joint.cpu(), collision_target.cpu(), collision_mask.cpu()
    records = []
    for row, spec in enumerate(specs):
        h = spec[3]
        record = {"current":current[row, :h], "goals":next_goals[row, :h],
            "action":action[row, :h], "old_logp":logp[row, :h], "value":values[row, :h],
            "effective":effective[row, :h], "components":components[row, :h],
            "reward":rewards[row, :h], "altered_tracking":altered_tracking[row, :h],
            "goal_position":goal_positions[row, :h], "goal_velocity":goal_velocities[row, :h],
            "clip":clips[row, :h], "penetration":penetrations[row, :h], "fall":falls[row, :h]}
        if collision_target is not None:
            record.update(base_joint=base_joint[row, :h], collision_target=collision_target[row, :h],
                          collision_mask=collision_mask[row, :h])
        records.append(record)
    return records


def build_normalizer(sampler, runtime, policy_joint, jq, jd, mass, names, action_std, joint_low, joint_high, kp, kd):
    # PD gains affect calibrated live-state features. Residual Std affects only the
    # explicitly constructed previous-residual channels, which can be updated exactly.
    if NORMALIZATION_PATH.is_file():
        with np.load(NORMALIZATION_PATH) as saved:
            schema = int(saved["schema"]); old_std = saved["action_std"]
            if not np.array_equal(saved["kp"], kp) or not np.array_equal(saved["kd"], kd):
                raise ValueError("Incompatible normalization PD gains")
            values = [saved[key].copy() for key in ("current_mean", "current_std", "goal_mean", "goal_std")]
        migrated = schema == 2 and old_std.shape == (NU,) and values[0].shape == (CURRENT_SIZE-NU,)
        if migrated:
            values[0] = np.r_[values[0], np.zeros(NU, np.float32)]
            values[1] = np.r_[values[1], action_std[NU:].cpu().numpy()]
        elif schema != 3: raise ValueError("Incompatible normalization schema")
        if [x.shape for x in values] != [(CURRENT_SIZE,), (CURRENT_SIZE,), (GOAL_SIZE,), (GOAL_SIZE,)] \
                or not all(np.isfinite(x).all() for x in values) or np.any(values[1] <= 0) or np.any(values[3] <= 0):
            raise ValueError("Malformed feature normalization")
        current_std = action_std.detach().cpu().numpy()
        changed = migrated or not np.array_equal(old_std, current_std)
        values[0][-ACTION_SIZE:] = 0; values[1][-ACTION_SIZE:] = current_std
        if changed:
            np.savez(NORMALIZATION_PATH, schema=3, action_std=current_std, kp=kp, kd=kd,
                     current_mean=values[0], current_std=values[1], goal_mean=values[2], goal_std=values[3])
            print("Updated normalization residual channels for current Residual_Std.npy.", flush=True)
        return Normalizer(*values)
    # Calibration uses zero residuals so normalization describes the baseline task,
    # not one random initial policy. Each source contributes equally, including Standing.
    print("Calibrating immutable feature normalization with 1,000 zero-residual one-second rollouts...", flush=True)
    rng = np.random.default_rng(SEED+2)
    specs = [sampler.one(rng, source, 40) for source in SOURCES for _ in range(200)]
    current_values, goal_values = [], []
    for chunk in chunk_specs(specs):
        current, goals = rollout_chunk(chunk, sampler, runtime, None, None, None, action_std,
            joint_low, joint_high, policy_joint, jq, jd, mass, normalization=True)
        rows = len(chunk); mask = torch.arange(current.shape[1])[None] < torch.tensor([x[3] for x in chunk])[:, None]
        current_values.append(current[:rows][mask]); goal_values.append(goals[:rows][mask])
    current = torch.cat(current_values).double(); goals = torch.cat(goal_values).double()
    current_mean, current_std = current.mean(0), current.std(0, correction=0)
    goal_mean, goal_std = goals.mean(0), goals.std(0, correction=0)
    # Previous-residual features have a known zero mean/action scale. Only explicitly
    # stationary joint-angle channels get the 0.02-degree floor; global clipping would
    # erase real differences between naturally low-variance and high-variance features.
    action_std = action_std.detach().cpu().double()
    current_mean[-ACTION_SIZE:] = 0; current_std[-ACTION_SIZE:] = action_std
    angle_start = 2+1+48+24+3+72+6+3
    for name in STATIONARY:
        index = int(np.where(names == name)[0][0])
        current_std[angle_start+index] = goal_std[angle_start+index] = math.radians(.02)
    if not all(torch.isfinite(x).all() for x in (current_mean, current_std, goal_mean, goal_std)) \
            or torch.any(current_std <= 0) or torch.any(goal_std <= 0):
        raise ValueError("A measured feature normalization Std is zero or nonfinite")
    values = [x.float().numpy() for x in (current_mean, current_std, goal_mean, goal_std)]
    np.savez(NORMALIZATION_PATH, schema=3, action_std=action_std.cpu().numpy(), kp=kp, kd=kd,
             current_mean=values[0], current_std=values[1], goal_mean=values[2], goal_std=values[3])
    return Normalizer(*values)


# REWARD CALIBRATION
# This is a separate zero-mean exploration measurement, never a training batch.
# Each rollout contributes equally regardless of sampled duration.
def load_reward_calibration(action_std, kp, kd):
    global REWARD_COEFFICIENTS, DIFFERENCE_POSITION_MEAN, DIFFERENCE_VELOCITY_MEAN
    if not REWARD_CALIBRATION_PATH.is_file(): return False
    settings = np.array((DURATION_STD, DURATION_MIN, DURATION_MAX, ACTION_HZ,
                         EXPLORING_REGION, STD_MULTIPLIER), np.float64)
    with np.load(REWARD_CALIBRATION_PATH) as saved:
        schema = int(saved["schema"])
        compatible = (schema in (2, 3) and np.array_equal(saved["settings"], settings)
            and int(saved["smooth"]) == SMOOTH
            and np.array_equal(saved["weights"], REWARD_WEIGHTS)
            and np.array_equal(saved["action_std"], action_std.cpu().numpy())
            and np.array_equal(saved["kp"], kp) and np.array_equal(saved["kd"], kd)
            and (schema == 2 or np.array_equal(saved["fall_settings"], FALL_SETTINGS)))
        if not compatible: return False
        coefficients = saved["coefficients"]
        difference = saved["difference_means"]
    if coefficients.shape != (7,) or difference.shape != (2,) or not np.isfinite(coefficients).all() \
            or not np.isfinite(difference).all() or np.any(difference <= 0):
        raise ValueError("Malformed reward calibration")
    REWARD_COEFFICIENTS = coefficients.astype(np.float32)
    DIFFERENCE_POSITION_MEAN, DIFFERENCE_VELOCITY_MEAN = map(float, difference)
    print("Loaded reward calibration.", flush=True)
    return True


def recalibrate_rewards(sampler, runtime, action_std, joint_low, joint_high,
                        policy_joint, jq, jd, mass, kp, kd):
    global REWARD_COEFFICIENTS, DIFFERENCE_POSITION_MEAN, DIFFERENCE_VELOCITY_MEAN
    rng = np.random.default_rng(SEED+3+RANK); torch.manual_seed(SEED+3+RANK); torch.cuda.manual_seed(SEED+3+RANK)
    specs = [sampler.one(rng, source) for source in SOURCES for _ in range(RANK, 1000, WORLD_SIZE)]
    records, chunks = [], chunk_specs(specs)
    if primary(): print("Recalibrating rewards with 5,000 current-duration rollouts...", flush=True)
    for index, chunk in enumerate(chunks, 1):
        records.extend(rollout_chunk(chunk, sampler, runtime, None, None, None, action_std,
            joint_low, joint_high, policy_joint, jq, jd, mass, calibration=True))
        if primary(): print(f"Reward recalibration: {100*index/len(chunks):.0f}%", flush=True)
    costs = np.stack([record["components"].numpy().mean(0) for record in records])
    position = np.array([record["goal_position"].square().mean().item() for record in records])
    velocity = np.array([record["goal_velocity"].square().mean().item() for record in records])
    falls = np.array([record["fall"].any().item() for record in records], np.float32)
    if WORLD_SIZE > 1:
        received = [None] * WORLD_SIZE if primary() else None
        dist.gather_object((costs, position, velocity, falls), received, dst=0)
        if not primary(): return
        costs = np.concatenate([value[0] for value in received])
        position = np.concatenate([value[1] for value in received])
        velocity = np.concatenate([value[2] for value in received])
        falls = np.concatenate([value[3] for value in received])
    difference = np.array((position.mean(), velocity.mean()), np.float64)
    costs[:, 2] = .75*position/difference[0] + .25*velocity/difference[1]
    means, stds = costs.mean(0), costs.std(0)
    scaled = np.divide(REWARD_WEIGHTS, stds, out=np.zeros(7), where=(REWARD_WEIGHTS > 0) & (stds > 0))
    denominator = scaled @ means
    if not np.isfinite(costs).all() or not np.isfinite(denominator) or denominator <= 0 \
            or np.any((REWARD_WEIGHTS > 0) & (stds <= 0)):
        raise ValueError("Reward recalibration produced zero or nonfinite statistics")
    coefficients = scaled / denominator
    settings = np.array((DURATION_STD, DURATION_MIN, DURATION_MAX, ACTION_HZ,
                         EXPLORING_REGION, STD_MULTIPLIER), np.float64)
    np.savez(REWARD_CALIBRATION_PATH, schema=3, settings=settings, smooth=SMOOTH,
        weights=REWARD_WEIGHTS, fall_settings=np.asarray(FALL_SETTINGS),
        action_std=action_std.cpu().numpy(), kp=kp, kd=kd,
        difference_means=difference, means=means, stds=stds, coefficients=coefficients)
    REWARD_COEFFICIENTS = coefficients.astype(np.float32)
    DIFFERENCE_POSITION_MEAN, DIFFERENCE_VELOCITY_MEAN = map(float, difference)
    labels = ("Tracking position squared", "Tracking velocity squared", "Altered-goal difference",
              "Residual second difference", "Torque saturation fraction",
              "Directional torque utilization", "Penetration depth")
    lines = ["Current reward calibration", "==========================", "",
        f"rollouts={len(costs)} (1,000 per source); duration seconds=clip(abs(N(0,{DURATION_STD})),"
        f" {DURATION_MIN}, {DURATION_MAX}); SMOOTH={SMOOTH}; exploring_region={EXPLORING_REGION}",
        "Zero actor mean; cumulative Gaussian exploration; equal weight per rollout.",
        f"Fall: fixed +{FALL_COST:g} once/run; root drop >= {FALL_HEIGHT_DROP:g} m, or drop >= "
        f"{FALL_TILT_HEIGHT_DROP:g} m with root error >= {math.degrees(FALL_ANGLE):g} deg; calibration fall rate={falls.mean():.6g}.", "",
        f"Altered-goal position squared mean: {difference[0]:.12g} m^2",
        f"Altered-goal velocity squared mean: {difference[1]:.12g} (m/s)^2", "",
        "Cost                              Mean          Std   Coefficient", "-------------------------------- ------------ ------------ ------------"]
    lines += [f"{name:<32} {mean:12.6g} {std:12.6g} {coefficient:12.6g}"
              for name, mean, std, coefficient in zip(labels, means, stds, coefficients)]
    REWARD_REPORT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Saved {REWARD_CALIBRATION_PATH.name} and {REWARD_REPORT}.", flush=True)


# MULTI-CHUNK COLLECTION
# One logical PPO batch can exceed GPU world count. Chunks are concatenated only after
# every rollout is complete; recurrent trajectories are never split across chunks.
def collect(specs, sampler, runtime, actor, critic, normalizer, action_std,
            joint_low, joint_high, policy_joint, jq, jd, mass, deterministic=False):
    result = []
    for chunk in chunk_specs(specs):
        result.extend(rollout_chunk(chunk, sampler, runtime, actor, critic, normalizer, action_std,
            joint_low, joint_high, policy_joint, jq, jd, mass, deterministic))
    return result


# RETURN AND ADVANTAGE
# Backward GAE uses the critic at the next sampled state and zero bootstrap at the
# chosen terminal horizon. The critic target is the matching lambda-return V + A.
def finish_returns(records):
    for record in records:
        reward, value = record["reward"], record["value"]
        advantage = torch.empty_like(reward)
        carry = next_value = torch.zeros((), dtype=reward.dtype, device=reward.device)
        for t in range(len(reward)-1, -1, -1):
            delta = reward[t] + GAMMA * next_value - value[t]
            carry = delta + GAMMA * GAE_LAMBDA * carry; advantage[t] = carry
            next_value = value[t]
        record["advantage"] = advantage; record["return"] = advantage + value


# FRAME-BALANCED RECURRENT MINIBATCHES
# Whole rollouts stay intact for LSTM semantics; balancing by transition count avoids
# long horizons dominating one optimizer step merely through padding shape.
def minibatches(records, count):
    shuffled = list(records); RNG.shuffle(shuffled)
    bins, sizes = [[] for _ in range(count)], np.zeros(count, np.int64)
    for record in sorted(shuffled, key=lambda x: len(x["reward"]), reverse=True):
        index = int(sizes.argmin()); bins[index].append(record); sizes[index] += len(record["reward"])
    RNG.shuffle(bins)
    return bins


def padded(records, key):
    return nn.utils.rnn.pad_sequence([x[key] for x in records], batch_first=True).to(DEVICE)


def distributed_scale(count):
    if WORLD_SIZE == 1: return 1.
    total = torch.tensor(float(count), device=DEVICE); dist.all_reduce(total)
    return 0. if total == 0 else WORLD_SIZE * count / float(total)


# PPO OPTIMIZATION
# Critic and actor receive separate backward/clip/step operations. valid masks remove
# padding, and actor_valid additionally removes the forced terminal-zero action.
def update_policy(records, actor, critic, actor_optimizer, critic_optimizer, position_std,
                  policy_joint, joint_low, joint_high):
    actor_gradients, critic_gradients, actor_losses, critic_losses, collision_losses, clip_fractions = [], [], [], [], [], []
    policy = actor.module if WORLD_SIZE > 1 else actor; parameters = tuple(policy.parameters())
    batch_count = torch.tensor(min(PPO_MINIBATCHES, len(records)), device=DEVICE)
    if WORLD_SIZE > 1: dist.all_reduce(batch_count, op=dist.ReduceOp.MIN)
    batch_count = int(batch_count.item())
    for _ in range(PPO_EPOCHS):
        for batch in minibatches(records, batch_count):
            lengths = torch.tensor([len(x["reward"]) for x in batch], device=DEVICE)
            current, goals = padded(batch, "current"), padded(batch, "goals")
            returns, advantages = padded(batch, "return"), padded(batch, "advantage")
            old_logp, actions = padded(batch, "old_logp"), padded(batch, "action")
            valid = torch.arange(current.shape[1], device=DEVICE)[None] < lengths[:, None]
            critic_optimizer.zero_grad(set_to_none=True)
            prediction = critic(current, goals, lengths)
            critic_loss = ((prediction-returns)[valid].square()).mean() * VALUE_COEFFICIENT
            (critic_loss*distributed_scale(valid.sum())).backward()
            critic_gradients.append(float(nn.utils.clip_grad_norm_(critic.parameters(), GRADIENT_CLIP)))
            critic_optimizer.step(); critic_losses.append(float(critic_loss.detach()))
            if not CRITIC_ONLY:
                actor_optimizer.zero_grad(set_to_none=True)
                mean_increment = policy(current, goals, lengths); std = policy.action_std
                mean = mean_increment.cumsum(1)
                # Exact conditional likelihood with both carries: previous noise is
                # previous sampled action minus previous accumulated actor mean.
                noise_offset = torch.cat((torch.zeros_like(mean[:, :1]),
                    actions[:, :-1]-mean[:, :-1]), 1)
                logp = torch.distributions.Normal(mean + noise_offset, std).log_prob(actions).sum(-1)
                actor_valid = valid & (torch.arange(current.shape[1], device=DEVICE)[None] != lengths[:, None]-1)
                ratio = (logp[actor_valid]-old_logp[actor_valid]).exp()
                clip_fractions.append(float(((ratio < 1-PPO_CLIP) | (ratio > 1+PPO_CLIP)).float().mean()))
                advantage = advantages[actor_valid].detach()
                # PPO clipping limits policy-ratio movement, while gradient clipping
                # below limits optimizer magnitude; they guard different instabilities.
                objective = torch.minimum(ratio*advantage, ratio.clamp(1-PPO_CLIP, 1+PPO_CLIP)*advantage)
                actor_loss = -objective.mean(); rl_scale = distributed_scale(actor_valid.sum())
                rl_grad = torch.autograd.grad(actor_loss*rl_scale, parameters,
                                              retain_graph=SUPERVISED_PENETRATION)
                if SUPERVISED_PENETRATION:
                    raw_mean = torch.where(actor_valid[..., None], mean, torch.zeros_like(mean))
                    filtered = nn.functional.avg_pool1d(nn.functional.pad(raw_mean.transpose(1, 2),
                        (SMOOTH-1, 0)), SMOOTH, stride=1).transpose(1, 2)
                    t = torch.arange(mean.shape[1], device=DEVICE)[None]
                    filtered *= torch.sin(math.pi*t/(lengths[:, None]-1).clamp_min(1))[..., None]
                    base = padded(batch, "base_joint"); target = padded(batch, "collision_target")
                    predicted = torch.maximum(torch.minimum(base+filtered[..., :POLICY_NU],
                        joint_high[policy_joint][None, None]), joint_low[policy_joint][None, None])-base
                    collision_valid = padded(batch, "collision_mask").bool() & actor_valid
                    collision_loss = (((predicted-target)/position_std)**2)[collision_valid].mean() \
                        if collision_valid.any() else predicted.sum()*0
                    collision_scale = distributed_scale(collision_valid.sum())
                    pen_grad = torch.autograd.grad(collision_loss*collision_scale, parameters, allow_unused=True)
                    pen_grad = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(parameters, pen_grad))
                    if WORLD_SIZE > 1:
                        for gradient in (*rl_grad, *pen_grad): dist.all_reduce(gradient); gradient.div_(WORLD_SIZE)
                    rl_norm = torch.sqrt(sum(g.square().sum() for g in rl_grad))
                    pen_norm = torch.sqrt(sum(g.square().sum() for g in pen_grad))
                    if pen_norm > EPS and rl_norm > EPS:
                        gradients = tuple(rl_norm*((1-PENETRATION_GRADIENT_FRACTION)*r/rl_norm
                            + PENETRATION_GRADIENT_FRACTION*p/pen_norm) for r, p in zip(rl_grad, pen_grad))
                    elif pen_norm > EPS: gradients = pen_grad
                    else: gradients = rl_grad
                    collision_losses.append(float((collision_loss*collision_scale).detach()))
                else:
                    if WORLD_SIZE > 1:
                        for gradient in rl_grad: dist.all_reduce(gradient); gradient.div_(WORLD_SIZE)
                    gradients = rl_grad
                for parameter, gradient in zip(parameters, gradients): parameter.grad = gradient
                actor_gradients.append(float(nn.utils.clip_grad_norm_(actor.parameters(), GRADIENT_CLIP)))
                actor_optimizer.step(); actor_losses.append(float(actor_loss.detach()))
    result = {"actor_loss":np.mean(actor_losses) if actor_losses else 0.,
            "critic_loss":np.mean(critic_losses), "actor_grad":np.mean(actor_gradients) if actor_gradients else 0.,
            "critic_grad":np.mean(critic_gradients),
            "penetration_loss":np.mean(collision_losses) if collision_losses else 0.,
            "ppo_clip_rate":np.mean(clip_fractions) if clip_fractions else 0.}
    if WORLD_SIZE > 1:
        values = torch.tensor(list(result.values()), dtype=torch.float64, device=DEVICE)
        dist.all_reduce(values); values /= WORLD_SIZE
        result = dict(zip(result, values.cpu().tolist()))
    return result


# TRAINING DIAGNOSTICS
# Keep original tracking, proposed-goal deviation, actuator feasibility, and critic
# fit separate. A lower total cost alone cannot reveal reward-component domination.
def summarize(records):
    components = torch.cat([x["components"] for x in records]).numpy()
    returns = torch.cat([x["return"] for x in records]).numpy()
    values = torch.cat([x["value"] for x in records]).numpy()
    distances = np.sqrt(np.maximum(components[:, :2], 0))
    internal = torch.cat([x["components"][:-1] for x in records]).numpy()
    smooth = torch.cat([x["components"][1:, 3] for x in records]).numpy()
    goal_position = torch.cat([x["goal_position"][:-1] for x in records]).numpy()
    goal_velocity = torch.cat([x["goal_velocity"][:-1] for x in records]).numpy()
    clips = torch.cat([x["clip"][:-1] for x in records]).numpy()
    penetrations = torch.cat([x["penetration"][:-1] for x in records]).numpy()
    altered = torch.cat([x["altered_tracking"] for x in records]).numpy(); error = returns-values
    fall_events = sum(float(record["fall"].any()) for record in records)
    stats = np.r_[components.sum(0), len(components), internal[:, 2].sum(), len(internal),
        internal[:, 6].sum(), len(internal), smooth.sum(), len(smooth),
        np.square(goal_position).sum(), len(goal_position), np.square(goal_velocity).sum(), len(goal_velocity),
        clips.sum(), len(clips), penetrations.sum(), len(penetrations), altered.sum(), len(altered),
        returns.sum(), np.square(returns).sum(), error.sum(), np.square(error).sum(), len(returns),
        fall_events, len(records)]
    if WORLD_SIZE > 1:
        reduced = torch.as_tensor(stats, dtype=torch.float64, device=DEVICE); dist.all_reduce(reduced)
        stats = reduced.cpu().numpy()
        local = torch.as_tensor(distances, device=DEVICE); size = torch.tensor(len(local), device=DEVICE)
        sizes = [torch.empty_like(size) for _ in range(WORLD_SIZE)]; dist.all_gather(sizes, size)
        maximum = max(map(int, sizes)); padded = torch.zeros(maximum, 2, device=DEVICE); padded[:len(local)] = local
        gathered = [torch.empty_like(padded) for _ in range(WORLD_SIZE)]; dist.all_gather(gathered, padded)
        distances = torch.cat([value[:int(size)] for value, size in zip(gathered, sizes)])
        p95 = torch.quantile(distances, .95, dim=0).cpu().numpy()
    else: p95 = np.quantile(distances, .95, axis=0)
    csum, count = stats[:7], stats[7]; rmean, emean = stats[24]/stats[28], stats[26]/stats[28]
    variance = stats[25]/stats[28]-rmean*rmean; error_variance = stats[27]/stats[28]-emean*emean
    result = {"cost":float((csum @ REWARD_COEFFICIENTS+FALL_COST*stats[29])/count),
        "pos_rmse":float(np.sqrt(csum[0]/count)), "pos_p95":float(p95[0]),
        "vel_rmse":float(np.sqrt(csum[1]/count)), "vel_p95":float(p95[1]),
        "goal_diff_rmse":float(np.sqrt(stats[14]/stats[15])),
        "goal_velocity_diff_rmse":float(np.sqrt(stats[16]/stats[17])),
        "goal_combined":float(stats[8]/stats[9]), "smooth":float(stats[12]/stats[13]),
        "saturation":float(csum[4]/count), "torque_util":float(csum[5]/count),
        "penetration_depth":float(stats[10]/stats[11]), "joint_clip":float(stats[18]/stats[19]),
        "penetration":float(stats[20]/stats[21]), "altered_track":float(stats[22]/stats[23]),
        "fall_rate":float(stats[29]/stats[30]), "weighted_fall":float(FALL_COST*stats[29]/count),
        "explained_variance":float(1-error_variance/variance) if variance > EPS else 0.}
    for index, name in enumerate(("pos", "vel", "goal", "smooth", "sat", "torque", "penetration")):
        result[f"weighted_{name}"] = float(csum[index]*REWARD_COEFFICIENTS[index]/count)
    return result


# APPEND-ONLY NUMERICAL HISTORY
# Atomically extend an old header when new diagnostics appear; old rows get blanks.
def append_csv(path, row):
    exists = path.is_file(); fields = list(row)
    if exists:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file); old_fields, rows = reader.fieldnames or [], list(reader)
        fields = list(dict.fromkeys(old_fields+fields))
        if fields != old_fields:
            temporary = path.with_suffix(".temporary.csv")
            with temporary.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
                writer.writerows({key:value for key, value in old.items() if key is not None} for old in rows)
            os.replace(temporary, path)
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not exists: writer.writeheader()
        writer.writerow(row)


# HUMAN-READABLE LEARNING CURVES
# Training and deterministic validation share axes so stochastic improvement is not
# mistaken for generalization. Plots intentionally derive only from saved history.
def plots(history):
    if not history: return
    x = [r["batch"] for r in history]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, keys, title in zip(axes.flat,
        (("pos_rmse", "validation_pos_rmse"), ("pos_p95", "validation_pos_p95"),
         ("vel_rmse", "validation_vel_rmse"), ("vel_p95", "validation_vel_p95")),
        ("Position RMSE (m)", "Position P95 (m)", "Velocity RMSE (m/s)", "Velocity P95 (m/s)")):
        for key in keys: axis.plot(x, [r.get(key, np.nan) for r in history], label=key)
        axis.set_title(title); axis.grid(alpha=.3); axis.legend()
    figure.tight_layout(); figure.savefig(TRACKING_PNG, dpi=130); plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    keys = (("goal_diff_rmse",), ("smooth",), ("joint_clip",), ("saturation",), ("torque_util",),
            ("penetration", "validation_penetration", "fall_rate", "validation_fall_rate"))
    for axis, names, title in zip(axes.flat, keys,
        ("Goal difference RMSE (m)", "Residual second difference", "Joint clipping", "Saturation", "Torque utilization", "Penetration / fall rate")):
        for name in names: axis.plot(x, [r.get(name, np.nan) for r in history], label=name)
        axis.set_title(title); axis.grid(alpha=.3)
        if len(names) > 1: axis.legend()
    figure.tight_layout(); figure.savefig(RESIDUAL_PNG, dpi=130); plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, key, title in zip(axes.flat, ("cost", "critic_loss", "explained_variance", "critic_grad"),
        ("Total cost", "Critic loss", "Critic explained variance", "Gradient norms")):
        axis.plot(x, [r[key] for r in history], label=key)
        if key == "cost":
            for item in ("pos", "vel", "goal", "smooth", "torque", "fall"):
                axis.plot(x, [r.get(f"weighted_{item}", np.nan) for r in history], label=item)
        if key == "critic_grad": axis.plot(x, [r["actor_grad"] for r in history], label="actor_grad")
        axis.set_title(title); axis.grid(alpha=.3); axis.legend()
    figure.tight_layout(); figure.savefig(HEALTH_PNG, dpi=130); plt.close(figure)


# ATOMIC TRAINING STATE
# The temporary-then-replace write prevents a partial checkpoint. Model, optimizers,
# RNGs, and metric history must advance together for a reproducible resume.
def checkpoint(batch, actor, critic, actor_optimizer, critic_optimizer, history):
    torch.cuda.synchronize()
    state = {"rng":RNG.bit_generator.state, "torch_rng":torch.get_rng_state(),
             "cuda_rng":torch.cuda.get_rng_state()}
    states = [None] * WORLD_SIZE if primary() else None
    if WORLD_SIZE > 1: dist.gather_object(state, states, dst=0)
    else: states = [state]
    if not primary(): return
    print(f"checkpoint batch={batch}: CUDA synchronization passed", flush=True)
    payload = {"schema":"goal_residual_online_v5", "exploring_region":EXPLORING_REGION,
        "fall_settings":FALL_SETTINGS, "gae_settings":GAE_SETTINGS, "batch":batch, "history":history,
        "actor":actor.state_dict(), "critic":critic.state_dict(),
        "actor_optimizer":actor_optimizer.state_dict(), "critic_optimizer":critic_optimizer.state_dict(),
        "rng":states[0]["rng"], "torch_rng":states[0]["torch_rng"],
        "cuda_rng":states[0]["cuda_rng"], "rng_states":states}
    temporary = CHECKPOINT_PATH.with_suffix(".temporary.pt")
    torch.save(payload, temporary); os.replace(temporary, CHECKPOINT_PATH)
    print(f"checkpoint batch={batch}: saved", flush=True)


# STRICT RESUME
# Schema/shape checks reject semantically incompatible policies. Current residual Std,
# learning rates, fall settings, and GAE settings are authoritative. Optimizer moments are
# retained; any forward region expansion preserves old output rows but resets actor moments.
# An interrupted partial batch is never represented as a completed history entry.
def load_checkpoint(actor, critic, actor_optimizer, critic_optimizer):
    global RNG
    if not CHECKPOINT_PATH.is_file() or PERFORMANCE:
        RNG = np.random.default_rng(SEED+RANK); torch.manual_seed(SEED+RANK); torch.cuda.manual_seed(SEED+RANK)
        return 0, []
    saved = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if saved.get("schema") != "goal_residual_online_v5":
        raise ValueError("The existing checkpoint belongs to the replaced prototype; move it aside to start this project")
    saved_fall = saved.get("fall_settings")
    if primary() and saved_fall is None:
        print("Checkpoint predates fall loss; current fall settings become authoritative now.", flush=True)
    elif primary() and not np.array_equal(saved_fall, FALL_SETTINGS):
        print("Checkpoint fall settings differ; current top constants are authoritative.", flush=True)
    saved_gae = saved.get("gae_settings")
    if primary() and saved_gae is None:
        print("Checkpoint predates true GAE metadata; current gamma/lambda become authoritative now.", flush=True)
    elif primary() and not np.array_equal(saved_gae, GAE_SETTINGS):
        print("Checkpoint GAE settings differ; current top constants are authoritative.", flush=True)
    saved_region = int(saved["exploring_region"])
    if saved_region not in range(1, len(REGIONS)+1): raise ValueError("Checkpoint has an invalid exploring region")
    transition = saved_region < EXPLORING_REGION
    if saved_region != EXPLORING_REGION and not transition:
        raise ValueError("Checkpoint region cannot be newer than the current exploring region")
    if transition:
        old_n, new_n = sum(map(len, REGIONS[:saved_region])), POLICY_NU
        state = actor.state_dict(); special = {"head.2.weight", "head.2.bias", "action_std"}
        for key, value in saved["actor"].items():
            if key not in special:
                if key not in state or state[key].shape != value.shape: raise ValueError("Actor checkpoint shape mismatch")
                state[key] = value
        for key in ("head.2.weight", "head.2.bias"):
            source, target = saved["actor"][key], state[key].clone()
            target[:old_n] = source[:old_n]
            target[new_n:new_n+old_n] = source[old_n:]
            state[key] = target
        actor.load_state_dict(state)
    else:
        actor_state = dict(saved["actor"]); actor_state["action_std"] = actor.action_std
        actor.load_state_dict(actor_state); actor_optimizer.load_state_dict(saved["actor_optimizer"])
    critic.load_state_dict(saved["critic"]); critic_optimizer.load_state_dict(saved["critic_optimizer"])
    for group in actor_optimizer.param_groups: group["lr"] = ACTOR_LR
    for group in critic_optimizer.param_groups: group["lr"] = CRITIC_LR
    states = saved.get("rng_states")
    if states is not None and RANK < len(states): state = states[RANK]
    elif RANK == 0: state = saved
    else:
        RNG = np.random.default_rng(SEED+RANK); torch.manual_seed(SEED+RANK); torch.cuda.manual_seed(SEED+RANK)
        state = None
    if state is not None:
        RNG.bit_generator.state = state["rng"]
        torch.set_rng_state(torch.as_tensor(state["torch_rng"], dtype=torch.uint8, device="cpu").contiguous())
        torch.cuda.set_rng_state(torch.as_tensor(state["cuda_rng"], dtype=torch.uint8, device="cpu").contiguous())
    batch, history = int(saved["batch"]), saved.get("history", [])
    completed = int(history[-1]["batch"]) if history else 0
    if batch == completed + 1:
        steps = {int(state["step"].item()) for state in saved["critic_optimizer"]["state"].values()
                 if "step" in state}
        expected = len(history) * PPO_EPOCHS * PPO_MINIBATCHES
        if steps and steps != {expected}:
            raise ValueError("Interrupted checkpoint contains a partial critic update")
        if primary(): print(f"Ignoring unfinished saved batch {batch}; resuming completed batch {completed}.", flush=True)
        batch = completed
    elif batch != completed:
        raise ValueError("Checkpoint batch differs from completed training history")
    if transition and primary(): print(f"Expanded exploring region {saved_region} -> {EXPLORING_REGION}; actor optimizer reset.", flush=True)
    if primary(): print(f"Resumed batch {batch}.", flush=True)
    return batch, history


# OPTIONAL END-TO-END PROFILING
# Construction/warmup are excluded so the report describes steady-state training,
# not one-time graph compilation or cache cost.
def performance_report(batches):
    total = sum(PROFILE.values()); lines = ["Goal residual online PPO performance",
        f"measured_batches={batches}", f"collection_transitions_at_least={COLLECTION_TRANSITIONS}",
        "cache, normalization calibration, model construction, graph capture, and warm-up are excluded", ""]
    for name, elapsed in PROFILE.items(): lines.append(f"{name}: {1000*elapsed/batches:.3f} ms/batch, {100*elapsed/total:.2f}%")
    lines += ["", f"categorized_total={1000*total/batches:.3f} ms/batch",
              f"valid_transitions_per_second_at_least={COLLECTION_TRANSITIONS*batches/total:.3f}"]
    PERFORMANCE_REPORT.write_text("\n".join(lines)+"\n", encoding="utf-8"); print("\n".join(lines), flush=True)


# TRAINING LIFECYCLE
# Setup locks data/model/statistics before any PPO update. Each batch then performs
# sample -> physical rollout -> terminal GAE -> PPO -> diagnostics -> optional fixed
# validation -> atomic persistence. KeyboardInterrupt saves only completed batches.
def arguments():
    parser = argparse.ArgumentParser(); parser.add_argument("--stop", type=int)
    parser.add_argument("--recalibrate", action="store_true")
    args = parser.parse_args()
    if args.stop is not None and args.stop < 0: parser.error("--stop must be nonnegative")
    if args.recalibrate and args.stop is not None: parser.error("--recalibrate cannot be combined with --stop")
    return args


def main(performance=False):
    global PERFORMANCE, PROFILE
    args = arguments(); PERFORMANCE = performance
    # CUDA is not optional: CPU PyTorch cannot validate MJWarp graph/stream semantics.
    if not torch.cuda.is_available() or not wp.get_cuda_devices(): raise RuntimeError("CUDA PyTorch and NVIDIA Warp are required")
    torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
    model, jq_np, jd_np, names, kp, kd, force, mass_np, geoms = configure_model()
    if WORLD_SIZE > 1 and not primary():
        dist.barrier(); data, standing, scenes = load_cache(False)
    else:
        data, standing, scenes = load_cache(primary())
        if WORLD_SIZE > 1: dist.barrier()
    sampler = Sampler(data, standing, scenes)
    validation = load_validation(sampler)[RANK::WORLD_SIZE]
    jq = torch.as_tensor(jq_np, dtype=torch.long, device=DEVICE)
    jd = torch.as_tensor(jd_np, dtype=torch.long, device=DEVICE)
    mass = torch.as_tensor(mass_np, device=DEVICE)
    try: policy_joint_np = np.array([np.flatnonzero(names == name).item() for name in POLICY_JOINT_NAMES])
    except ValueError as error: raise ValueError("Exploring-region joint names do not match actuator order") from error
    if len(np.unique(policy_joint_np)) != POLICY_NU: raise ValueError("Exploring regions contain duplicate actuators")
    policy_joint = torch.as_tensor(policy_joint_np, dtype=torch.long, device=DEVICE)
    position_std = torch.as_tensor(np.load(STD_PATH)[3:], dtype=torch.float32, device=DEVICE)
    action_std = STD_MULTIPLIER * torch.cat((position_std, 5*position_std))
    policy_reference_std = torch.cat((position_std[policy_joint], 5*position_std[policy_joint]))
    policy_std = STD_MULTIPLIER * policy_reference_std
    if action_std.shape != (ACTION_SIZE,) or not torch.all(action_std > 0): raise ValueError("Invalid fixed residual Std")
    joint_range = torch.as_tensor(model.jnt_range[model.actuator_trnid[:, 0]], dtype=torch.float32, device=DEVICE)
    joint_low, joint_high = joint_range.unbind(-1)
    print(f"GPU {RANK}: preparing MJWarp runtime for {ENVIRONMENT_COUNT} worlds...", flush=True)
    runtime = Runtime(model, jq_np, jd_np, kp, kd, force, geoms, policy_joint_np, names)
    print(f"GPU {RANK}: MJWarp runtime ready.", flush=True)
    if args.recalibrate:
        recalibrate_rewards(sampler, runtime, action_std, joint_low, joint_high,
                            policy_joint, jq, jd, mass, kp, kd)
        return
    if not load_reward_calibration(action_std, kp, kd):
        if CHECKPOINT_PATH.is_file():
            raise ValueError("Reward calibration is incompatible with the checkpoint; run with --recalibrate only after discarding it")
        if primary(): print("Fresh start has no compatible reward calibration; recalibrating once...", flush=True)
        recalibrate_rewards(sampler, runtime, action_std, joint_low, joint_high,
                            policy_joint, jq, jd, mass, kp, kd)
        if WORLD_SIZE > 1: dist.barrier()
        if not primary() and not load_reward_calibration(action_std, kp, kd):
            raise RuntimeError("Primary GPU did not save a compatible reward calibration")
    if WORLD_SIZE > 1:
        if primary():
            normalizer = build_normalizer(sampler, runtime, policy_joint, jq, jd, mass, names, action_std,
                                          joint_low, joint_high, kp, kd)
        dist.barrier()
        if not primary():
            normalizer = build_normalizer(sampler, runtime, policy_joint, jq, jd, mass, names, action_std,
                                          joint_low, joint_high, kp, kd)
    else:
        normalizer = build_normalizer(sampler, runtime, policy_joint, jq, jd, mass, names, action_std,
                                      joint_low, joint_high, kp, kd)
    if primary(): print("Feature normalization ready.", flush=True)
    actor = RecurrentPolicy(initial_bias=0*policy_reference_std.cpu()).to(DEVICE)
    critic = RecurrentPolicy(critic=True).to(DEVICE)
    actor.register_buffer("action_std", policy_std.clone())
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
    batch, history = load_checkpoint(actor, critic, actor_optimizer, critic_optimizer)
    actor_train = DDP(actor, device_ids=[RANK], output_device=RANK) if WORLD_SIZE > 1 else actor
    critic_train = DDP(critic, device_ids=[RANK], output_device=RANK) if WORLD_SIZE > 1 else critic
    if args.stop is not None and batch >= args.stop:
        checkpoint(batch, actor, critic, actor_optimizer, critic_optimizer, history); return
    measured_batches = 0; batch_complete = True
    try:
        while args.stop is None or batch < args.stop:
            # Collect at least the requested transitions using complete variable-length
            # rollouts; the final rollout may intentionally exceed the target count.
            started = time.perf_counter(); next_batch = batch + 1; batch_complete = False
            with measured("01_sampling"):
                specs = sampler.collect(COLLECTION_TRANSITIONS // WORLD_SIZE)
            with measured("02_online_mjwarp_collection"):
                records = collect(specs, sampler, runtime, actor, critic, normalizer, action_std,
                                  joint_low, joint_high, policy_joint, jq, jd, mass)
                finish_returns(records)
            with measured("03_ppo_five_epochs"):
                learning = update_policy(records, actor_train, critic_train, actor_optimizer, critic_optimizer,
                                         policy_std[:POLICY_NU], policy_joint, joint_low, joint_high)
            with measured("04_metrics"):
                metrics = summarize(records); metrics.update(learning); metrics["batch"] = next_batch
            if next_batch % VALID_INTERVAL == 0:
                # Validation uses actor mean only. It measures the learned goal rewrite,
                # not exploration noise, against spans excluded from training.
                with measured("05_fixed_validation"):
                    validation_records = collect(validation, sampler, runtime, actor, critic, normalizer,
                        action_std, joint_low, joint_high, policy_joint, jq, jd, mass, deterministic=True)
                    finish_returns(validation_records); valid = summarize(validation_records)
                    metrics.update({"validation_"+k:v for k, v in valid.items()})
                    if not PERFORMANCE and primary(): append_csv(VALID_CSV, {"batch":next_batch, **valid})
            elapsed = torch.tensor(time.perf_counter()-started, device=DEVICE)
            if WORLD_SIZE > 1: dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            metrics["batch_seconds"] = float(elapsed)
            history.append(metrics)
            batch = next_batch; batch_complete = True
            if not PERFORMANCE and primary():
                append_csv(TRAIN_CSV, {key:value for key, value in metrics.items()
                                       if not key.startswith("validation_")})
            if batch % PRINT_INTERVAL == 0 and primary():
                deterministic_penetration = (f"{100*metrics['validation_penetration']:.2f}%"
                    if "validation_penetration" in metrics else "n/a")
                deterministic_fall = (f"{100*metrics['validation_fall_rate']:.2f}%"
                    if "validation_fall_rate" in metrics else "n/a")
                print(f"batch={batch} cost={metrics['cost']:.5g} pos={metrics['pos_rmse']:.5g}m "
                    f"vel={metrics['vel_rmse']:.5g}m/s goal_diff={metrics['goal_diff_rmse']:.5g}m "
                    f"goal_vel_diff={metrics['goal_velocity_diff_rmse']:.5g}m/s "
                    f"smooth={metrics['smooth']:.4g} sat={100*metrics['saturation']:.2f}% "
                    f"torque_util={100*metrics['torque_util']:.2f}% joint_clip={100*metrics['joint_clip']:.2f}% "
                    f"penetration_loss={metrics['penetration_loss']:.5g} fall={100*metrics['fall_rate']:.2f}% "
                    f"det_fall={deterministic_fall} det_penetration={deterministic_penetration} "
                    f"ppo_clip={100*metrics['ppo_clip_rate']:.2f}% "
                    f"actor_grad={metrics['actor_grad']:.4g} "
                    f"critic_grad={metrics['critic_grad']:.4g} time={metrics['batch_seconds']:.2f}s "
                    f"critic_ev={metrics['explained_variance']:.4g}", flush=True)
            if not PERFORMANCE and batch % CHECKPOINT_INTERVAL == 0:
                checkpoint(batch, actor, critic, actor_optimizer, critic_optimizer, history)
            if not PERFORMANCE and batch % PNG_INTERVAL == 0 and primary(): plots(history)
            if PERFORMANCE:
                # First batch warms kernels/allocators; only three later batches enter
                # the steady-state attribution report.
                if batch == 1: PROFILE.clear()
                else:
                    measured_batches += 1
                    if measured_batches == 3: break
    except KeyboardInterrupt:
        if not batch_complete:
            print("Interrupted during an unfinished batch; keeping the previous on-disk checkpoint.", flush=True)
            return
        print("Interrupted between batches; saving the completed state.", flush=True)
    if PERFORMANCE:
        if primary(): performance_report(measured_batches)
    else:
        checkpoint(batch, actor, critic, actor_optimizer, critic_optimizer, history)
        if primary(): plots(history)


def _worker(rank, world_size, init_method, performance=False):
    global RANK, WORLD_SIZE, DEVICE, RNG
    RANK, WORLD_SIZE = rank, world_size; DEVICE = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank); wp.set_device(wp.get_cuda_device(rank)); RNG = np.random.default_rng(SEED+rank)
    if world_size > 1: dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size)
    try: main(performance)
    finally:
        if world_size > 1: dist.destroy_process_group()


def launch(performance=False):
    if "-h" in os.sys.argv or "--help" in os.sys.argv: arguments(); return
    count = torch.cuda.device_count()
    if count < 1: raise RuntimeError("CUDA PyTorch and NVIDIA Warp are required")
    if count == 1: _worker(0, 1, None, performance); return
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0)); port = connection.getsockname()[1]
    print(f"Launching {count} GPUs with {COLLECTION_TRANSITIONS//count:,} transitions per GPU.", flush=True)
    mp.spawn(_worker, args=(count, f"tcp://127.0.0.1:{port}", performance), nprocs=count, join=True)


if __name__ == "__main__": launch()
