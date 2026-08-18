"""Autoregressive supervised planner between two recorded motion goals."""
import argparse, csv, os, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from torch import nn

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
CACHE = REPOSITORY / "data" / "cache" / "Movement_Data_Cache.npy"
META = REPOSITORY / "data" / "cache" / "Movement_Data_Cache_Metadata.npz"
STANDING = REPOSITORY / "data" / "movement_data" / "Movement_Data" / "Standing_Pose_Pool.npy"
XML = REPOSITORY / "configs" / "human_body_47.xml"
GAINS = REPOSITORY / "configs" / "PD_Gains.npz"
NORMALIZATION = REPOSITORY / "configs" / "Goal_Planner_Normalization.npz"
CHECKPOINT = REPOSITORY / "models" / "Goal_Planner_latest.pt"
METRICS_CSV, METRICS_PNG = REPOSITORY / "results" / "Goal_Planner_Metrics_1024.csv", REPOSITORY / "results" / "Goal_Planner_Metrics_1024.png"
VALIDATION = REPOSITORY / "data" / "validation" / "Validation"

SEED, HZ = 12, 40
DURATION_STD, DURATION_MIN, DURATION_MAX = 1., .25, 2.
ANGLE_STD_FLOOR = float(np.deg2rad(.02))
MAX_STEPS = round(DURATION_MAX*HZ)-1
SMOOTH, CORRECT_BEFORE_LOSS = 10, False
CORRECTION_PASSES = 10
CORRECTION_TOLERANCE = .001
CORRECTION_MAX_STEP, CORRECTION_MAX_TOTAL = float(np.deg2rad(3.)), float(np.deg2rad(30.))
BATCH_SIZE, LEARNING_RATE = 512, 1e-4
ENCODER_WIDTH, DECODER_WIDTH = 1024, 512
NORMALIZATION_SAMPLES, LOSS_CALIBRATION_BATCHES = 10_000, 10
PRINT_INTERVAL, VALID_INTERVAL, CHECKPOINT_INTERVAL = 1000000, 10, 100
NQ, NV, NBODY, NU, FRAME_SIZE, STATE_SIZE, PATH_SIZE = 54, 53, 24, 47, 251, 259, 54
ROOT_XYZ, ROOT_QUAT, JOINT, STATE_JOINT = slice(0, 3), slice(3, 7), slice(7, 54), slice(157, 204)
LOSS_WEIGHTS = torch.tensor((.40, .20, .10, .30))  # XY, Z, root orientation, joints
METRIC_FIELDS = ("batch", "split", "loss", "xy_normalized", "z_normalized",
    "root_rotation_normalized", "joints_normalized", "xy_rmse_m", "z_rmse_m",
    "root_rotation_rmse_deg", "joint_rmse_deg", "gradient_norm", "time_seconds", "valid_frames")
if CORRECTION_PASSES % 2: raise ValueError("CORRECTION_PASSES must be even for staged correction")


@wp.kernel
def _foot_contacts(nacon: wp.array(dtype=wp.int32), geom: wp.array(dtype=wp.vec2i),
                   worldid: wp.array(dtype=wp.int32), floor: int, left_foot: int,
                   left_toe: int, right_foot: int, right_toe: int,
                   flags: wp.array2d(dtype=wp.int32)):
    contact = wp.tid()
    if contact < nacon[0]:
        pair, world = geom[contact], worldid[contact]
        a, b = pair[0], pair[1]
        if (a == floor and (b == left_foot or b == left_toe)) or \
           (b == floor and (a == left_foot or a == left_toe)):
            flags[world, 0] = 1
        if (a == floor and (b == right_foot or b == right_toe)) or \
           (b == floor and (a == right_foot or a == right_toe)):
            flags[world, 1] = 1


@wp.kernel
def _depths(nacon: wp.array(dtype=wp.int32), geom: wp.array(dtype=wp.vec2i),
            worldid: wp.array(dtype=wp.int32), dist: wp.array(dtype=wp.float32),
            bodyid: wp.array(dtype=wp.int32), floor: int, active: wp.array(dtype=wp.int32),
            self_depth: wp.array(dtype=wp.float32), floor_depth: wp.array(dtype=wp.float32)):
    contact = wp.tid()
    if contact < nacon[0]:
        pair, world, depth = geom[contact], worldid[contact], -dist[contact]
        if active[world] != 0 and depth > CORRECTION_TOLERANCE:
            if bodyid[pair[0]] != 0 and bodyid[pair[1]] != 0:
                wp.atomic_max(self_depth, world, depth)
            if pair[0] == floor or pair[1] == floor:
                wp.atomic_max(floor_depth, world, depth)


@wp.kernel
def _collision_scale(nacon: wp.array(dtype=wp.int32), geom: wp.array(dtype=wp.vec2i),
        worldid: wp.array(dtype=wp.int32), dist: wp.array(dtype=wp.float32),
        efc_address: wp.array2d(dtype=wp.int32), bodyid: wp.array(dtype=wp.int32),
        rownnz: wp.array2d(dtype=wp.int32), rowadr: wp.array2d(dtype=wp.int32),
        colind: wp.array3d(dtype=wp.int32), jacobian: wp.array3d(dtype=wp.float32),
        dofs: wp.array(dtype=wp.int32), weights: wp.array(dtype=wp.float32),
        active: wp.array(dtype=wp.int32), scale: wp.array(dtype=wp.float32)):
    contact = wp.tid(); scale[contact] = 0.0
    if contact < nacon[0]:
        pair, world = geom[contact], worldid[contact]
        if active[world] != 0 and bodyid[pair[0]] != 0 and bodyid[pair[1]] != 0 \
                and dist[contact] < -CORRECTION_TOLERANCE:
            row = efc_address[contact, 0]
            if row >= 0:
                start, count = rowadr[world, row], rownnz[world, row]; denominator = float(0.0)
                for joint in range(dofs.shape[0]):
                    value = float(0.0)
                    for entry in range(count):
                        address = start+entry
                        if colind[world, 0, address] == dofs[joint]: value = jacobian[world, 0, address]
                    denominator += value*value/weights[joint]
                if denominator > 1.0e-10:
                    scale[contact] = (-CORRECTION_TOLERANCE-dist[contact])/denominator


@wp.kernel
def _collision_delta(worldid: wp.array(dtype=wp.int32), efc_address: wp.array2d(dtype=wp.int32),
        rownnz: wp.array2d(dtype=wp.int32), rowadr: wp.array2d(dtype=wp.int32),
        colind: wp.array3d(dtype=wp.int32), jacobian: wp.array3d(dtype=wp.float32),
        dofs: wp.array(dtype=wp.int32), weights: wp.array(dtype=wp.float32),
        scale: wp.array(dtype=wp.float32), delta: wp.array2d(dtype=wp.float32)):
    contact, joint = wp.tid(); factor = scale[contact]
    if factor != 0.0:
        world, row = worldid[contact], efc_address[contact, 0]
        start, count = rowadr[world, row], rownnz[world, row]; value = float(0.0)
        for entry in range(count):
            address = start+entry
            if colind[world, 0, address] == dofs[joint]: value = jacobian[world, 0, address]
        wp.atomic_add(delta, world, joint, value/weights[joint]*factor)


@wp.kernel
def _apply_collision(qpos: wp.array2d(dtype=wp.float32), original: wp.array2d(dtype=wp.float32),
        indices: wp.array(dtype=wp.int32), lower: wp.array(dtype=wp.float32),
        upper: wp.array(dtype=wp.float32), active: wp.array(dtype=wp.int32),
        delta: wp.array2d(dtype=wp.float32)):
    world, joint = wp.tid()
    if active[world] != 0:
        index = indices[joint]
        low = wp.max(lower[joint], original[world, index]-CORRECTION_MAX_TOTAL)
        high = wp.min(upper[joint], original[world, index]+CORRECTION_MAX_TOTAL)
        qpos[world, index] = wp.clamp(qpos[world, index]
            + wp.clamp(delta[world, joint], -CORRECTION_MAX_STEP, CORRECTION_MAX_STEP), low, high)


@wp.kernel
def _reject_worse(qpos: wp.array2d(dtype=wp.float32), original: wp.array2d(dtype=wp.float32),
        indices: wp.array(dtype=wp.int32), before: wp.array(dtype=wp.float32),
        after: wp.array(dtype=wp.float32)):
    world = wp.tid()
    if after[world] >= before[world]:
        for joint in range(indices.shape[0]): qpos[world, indices[joint]] = original[world, indices[joint]]


@wp.kernel
def _record_overflow(overflow: wp.array(dtype=wp.int32), seen: wp.array(dtype=wp.int32)):
    world = wp.tid()
    if overflow[world] != 0: wp.atomic_max(seen, 0, 1)


class ForwardRuntime:
    def __init__(self, model, floor, feet, device, jq, jd, names):
        worlds = 2*BATCH_SIZE
        self.device, self.wmodel = device, mjw.put_model(model)
        self.data = mjw.make_data(model, nworld=worlds, nconmax=64, njmax=330)
        self.input_qpos = torch.empty(worlds, NQ, device=device)
        self.input_qvel = torch.empty(worlds, NV, device=device)
        self.flags = torch.zeros(worlds, 2, dtype=torch.int32, device=device)
        self.overflow = wp.to_torch(self.data.overflow)
        self.default_qpos = torch.as_tensor(model.qpos0, dtype=torch.float32, device=device).expand_as(self.input_qpos)
        self.wp_qpos, self.wp_qvel, self.wp_flags = map(wp.from_torch,
            (self.input_qpos, self.input_qvel, self.flags))
        self.geoms = (int(floor), *(int(x) for side in feet for x in side))
        self.torch_stream = torch.cuda.Stream(device=device)
        self.stream = wp.stream_from_torch(self.torch_stream)
        if CORRECT_BEFORE_LOSS:
            if self.data.efc.J.shape[1] != 1: raise ValueError("GPU correction requires sparse MJWarp Jacobians")
            ranges = model.jnt_range[model.actuator_trnid[:, 0]].astype("f4")
            weights = np.array([4. if name.startswith(("lumbar", "thoracic", "neck")) else
                2. if any(part in name for part in ("hip", "knee", "ankle", "toes")) else 1.
                for name in names], np.float32)
            self.original = torch.empty_like(self.input_qpos); self.active = torch.zeros(worlds, dtype=torch.int32, device=device)
            self.delta = torch.zeros(worlds, NU, device=device)
            contacts = self.data.contact.dist.shape[0]
            self.scale = torch.zeros(contacts, device=device); self.before = torch.zeros(worlds, device=device)
            self.after = torch.zeros_like(self.before); self.floor_depth = torch.zeros_like(self.before)
            self.overflow_seen = torch.zeros(1, dtype=torch.int32, device=device)
            for name in ("original", "active", "delta", "scale", "before", "after", "floor_depth", "overflow_seen"):
                setattr(self, "wp_"+name, wp.from_torch(getattr(self, name)))
            self.state_qpos = wp.to_torch(self.data.qpos)
            self.wp_bodyid = wp.array(model.geom_bodyid.astype(np.int32), dtype=wp.int32, device="cuda")
            self.wp_jq = wp.array(jq, dtype=wp.int32, device="cuda"); self.wp_jd = wp.array(jd, dtype=wp.int32, device="cuda")
            self.wp_low = wp.array(ranges[:, 0], dtype=wp.float32, device="cuda")
            self.wp_high = wp.array(ranges[:, 1], dtype=wp.float32, device="cuda")
            self.wp_weights = wp.array(weights, dtype=wp.float32, device="cuda")
            self._capture_correction()

    def _forward(self):
        mjw.forward(self.wmodel, self.data)
        wp.launch(_record_overflow, dim=len(self.input_qpos), inputs=[self.data.overflow, self.wp_overflow_seen])

    def _measure(self):
        self.wp_after.zero_(); self.wp_floor_depth.zero_()
        wp.launch(_depths, dim=self.data.contact.dist.shape[0], inputs=[self.data.nacon,
            self.data.contact.geom, self.data.contact.worldid, self.data.contact.dist,
            self.wp_bodyid, self.geoms[0], self.wp_active, self.wp_after, self.wp_floor_depth])

    def _detect_ops(self):
        wp.copy(self.data.qpos, self.wp_qpos); wp.copy(self.data.qvel, self.wp_qvel)
        self._forward(); self._measure()

    def _five_pass_ops(self):
        wp.copy(self.data.qpos, self.wp_qpos); wp.copy(self.data.qvel, self.wp_qvel); self._forward()
        contacts, capacity = self.data.contact.dist.shape[0], len(self.input_qpos)
        for _ in range(CORRECTION_PASSES//2):
            wp.launch(_collision_scale, dim=contacts, inputs=[self.data.nacon,
                self.data.contact.geom, self.data.contact.worldid, self.data.contact.dist,
                self.data.contact.efc_address, self.wp_bodyid, self.data.efc.J_rownnz,
                self.data.efc.J_rowadr, self.data.efc.J_colind, self.data.efc.J,
                self.wp_jd, self.wp_weights, self.wp_active, self.wp_scale])
            self.wp_delta.zero_(); wp.launch(_collision_delta, dim=(contacts, NU), inputs=[
                self.data.contact.worldid, self.data.contact.efc_address, self.data.efc.J_rownnz,
                self.data.efc.J_rowadr, self.data.efc.J_colind, self.data.efc.J,
                self.wp_jd, self.wp_weights, self.wp_scale, self.wp_delta])
            wp.launch(_apply_collision, dim=(capacity, NU), inputs=[self.data.qpos,
                self.wp_original, self.wp_jq, self.wp_low, self.wp_high,
                self.wp_active, self.wp_delta]); self._forward()
        self._measure()

    def _finalize_ops(self):
        wp.launch(_reject_worse, dim=len(self.input_qpos), inputs=[self.data.qpos,
            self.wp_original, self.wp_jq, self.wp_before, self.wp_after])
        self._forward(); self._measure()

    def _capture_correction(self):
        self.input_qpos.copy_(self.default_qpos); self.input_qvel.zero_(); self.original.copy_(self.default_qpos)
        self.active.zero_(); self.before.zero_(); self.after.zero_(); self.overflow_seen.zero_()
        self.stream.wait_stream(wp.stream_from_torch(torch.cuda.current_stream(self.device)))
        with wp.ScopedStream(self.stream): self._detect_ops(); self._five_pass_ops(); self._finalize_ops()
        wp.synchronize_stream(self.stream)
        graphs = []
        for operation in (self._detect_ops, self._five_pass_ops, self._finalize_ops):
            with wp.ScopedCapture(device="cuda", stream=self.stream, force_module_load=False) as capture: operation()
            graphs.append(capture.graph)
        self.detect_graph, self.five_graph, self.finalize_graph = graphs
        self.overflow_seen.zero_()

    def _launch(self, graph):
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(wp.stream_from_torch(current)); wp.capture_launch(graph, stream=self.stream)
        current.wait_stream(self.torch_stream)

    def contacts(self, qpos, qvel):
        count = len(qpos)
        if count > len(self.input_qpos): raise ValueError("MJWarp endpoint batch exceeds its fixed capacity")
        self.input_qpos.copy_(self.default_qpos); self.input_qvel.zero_()
        self.input_qpos[:count].copy_(torch.as_tensor(qpos, device=self.device))
        self.input_qvel[:count].copy_(torch.as_tensor(qvel, device=self.device))
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(wp.stream_from_torch(current))
        with wp.ScopedStream(self.stream):
            wp.copy(self.data.qpos, self.wp_qpos); wp.copy(self.data.qvel, self.wp_qvel)
            self.wp_flags.zero_(); mjw.forward(self.wmodel, self.data)
            wp.launch(_foot_contacts, dim=self.data.contact.geom.shape[0], inputs=[self.data.nacon,
                self.data.contact.geom, self.data.contact.worldid, *self.geoms, self.wp_flags])
        current.wait_stream(self.torch_stream)
        flags, overflow = self.flags[:count].cpu().numpy(), self.overflow[:count].cpu().numpy()
        if np.any(overflow): raise RuntimeError(f"MJWarp endpoint forward overflow: flags={np.unique(overflow)}")
        return flags.astype(np.float32)

    def _batched(self, graph, qpos, original=None, before=None, after=None):
        result, depth, floor = torch.empty_like(qpos), torch.empty(len(qpos), device=self.device), torch.empty(len(qpos), device=self.device)
        capacity = len(self.input_qpos)
        for start in range(0, len(qpos), capacity):
            end = min(start+capacity, len(qpos)); count = end-start
            self.input_qpos.copy_(self.default_qpos); self.input_qvel.zero_(); self.original.copy_(self.default_qpos)
            self.active.zero_(); self.before.zero_(); self.after.zero_()
            self.input_qpos[:count].copy_(qpos[start:end]); self.active[:count] = 1
            if original is not None: self.original[:count].copy_(original[start:end])
            if before is not None: self.before[:count].copy_(before[start:end])
            if after is not None: self.after[:count].copy_(after[start:end])
            self._launch(graph); result[start:end].copy_(self.state_qpos[:count])
            depth[start:end].copy_(self.after[:count]); floor[start:end].copy_(self.floor_depth[:count])
        return result, depth, floor

    def correct(self, qpos, active):
        if not CORRECT_BEFORE_LOSS: return qpos
        self.overflow_seen.zero_(); valid = active.nonzero().flatten(); original_valid = qpos[valid]
        _, initial_depth, final_floor = self._batched(self.detect_graph, original_valid)
        corrected_valid = original_valid.clone(); impacted = (initial_depth > 0).nonzero().flatten()
        if impacted.numel():
            original = original_valid[impacted]; before = initial_depth[impacted]
            stage, residual, floor = self._batched(self.five_graph, original, original, before)
            corrected_valid[impacted] = stage; final_floor[impacted] = floor
            unresolved = (residual > 0).nonzero().flatten()
            if unresolved.numel():
                stage2, residual2, _ = self._batched(self.five_graph, stage[unresolved],
                    original[unresolved], before[unresolved])
                final, _, floor2 = self._batched(self.finalize_graph, stage2,
                    original[unresolved], before[unresolved], residual2)
                locations = impacted[unresolved]; corrected_valid[locations] = final; final_floor[locations] = floor2
        corrected_valid[:, 2] += torch.clamp(final_floor-CORRECTION_TOLERANCE, min=0)
        corrected = qpos.clone(); corrected[valid] = corrected_valid
        if self.overflow_seen.item(): raise RuntimeError("MJWarp correction overflow")
        return corrected


class MotionData:
    def __init__(self, device):
        if not CACHE.is_file() or not META.is_file():
            raise FileNotFoundError("Run Goal_Refinement_Learning.py once to build the movement cache")
        self.frames = np.load(CACHE, mmap_mode="r")
        with np.load(META) as saved:
            if int(saved["version"]) != 2 or tuple(saved["shape"]) != self.frames.shape:
                raise ValueError("Movement cache metadata is incompatible")
            manifest = saved["manifest"]
        if self.frames.shape[1:] != (FRAME_SIZE,) or manifest.ndim != 2 or manifest.shape[1] != 3:
            raise ValueError("Malformed movement cache")
        self.records, offset = [[] for _ in range(4)], 0
        for category, scene, length in manifest:
            self.records[int(category)].append((int(scene), offset, int(length))); offset += int(length)
        if offset != len(self.frames) or any(not records for records in self.records):
            raise ValueError("Movement cache manifest does not match its frames")
        self.standing = np.load(STANDING)
        if self.standing.ndim != 2 or self.standing.shape[1] != FRAME_SIZE:
            raise ValueError("Malformed standing-pose pool")
        names = {1:0, 2:1, 3:3, 4:2}; self.blocked, self.valid = {}, {}
        for code, scene, center in np.load(VALIDATION / "Validation_Positions.npy"):
            if int(code) != 4: self.blocked.setdefault((names[int(code)], int(scene)), []).append((int(center)-2, int(center)+2))
        for code, scene, start, end in np.load(VALIDATION / "Long_Term_Validation_Positions.npy"):
            if int(code) != 4: self.blocked.setdefault((names[int(code)], int(scene)), []).append((int(start), int(end)))
        lookup = {(category, scene):(offset, length) for category, records in enumerate(self.records)
                  for scene, offset, length in records}
        self.validation = []
        positions = np.load(VALIDATION / "Long_Term_Validation_Positions.npy")
        if positions.shape != (2400, 4): raise ValueError("Malformed long-term validation positions")
        for code, scene, start, end in positions:
            category = names[int(code)]; offset, length = lookup[(category, int(scene))]
            start, end = int(start), int(end)
            if start < 0 or end >= length or end <= start or end-start > MAX_STEPS:
                raise ValueError("Invalid planner validation span")
            self.validation.append(np.asarray(self.frames[offset+start:offset+end+1]).copy())
        with np.load(VALIDATION / "Standing_Long_Term_Validation.npz") as saved:
            standing, steps = saved["endpoints"][:, 0], saved["total_physics_steps"]
        horizons = np.ceil(steps/(400/HZ)).astype(np.int64)
        if standing.shape != (600, FRAME_SIZE) or steps.shape != (600,) or np.any(horizons > MAX_STEPS):
            raise ValueError("Malformed standing planner validation")
        self.validation += [np.repeat(frame[None], horizon+1, 0) for frame, horizon in zip(standing, horizons)]
        if len(self.validation) != 3000: raise ValueError("Planner validation must contain 3,000 trajectories")

        self.model = mujoco.MjModel.from_xml_path(str(XML))
        if (self.model.nq, self.model.nv, self.model.nu, self.model.nbody) != (NQ, NV, NU, NBODY+1):
            raise ValueError("Unexpected MuJoCo model dimensions")
        joints = self.model.actuator_trnid[:, 0]
        if len(np.unique(joints)) != NU or not np.all(self.model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Planner requires 47 unique hinge-joint actuators")
        free = int(self.model.body_jntadr[1])
        if (self.model.njnt != NU+1 or self.model.body_jntnum[1] != 1
                or self.model.jnt_type[free] != mujoco.mjtJoint.mjJNT_FREE
                or set(map(int, joints)) != set(range(self.model.njnt))-{free}):
            raise ValueError("Differentiable FK requires one root free joint and exactly 47 actuated hinges")
        self.jq, self.jd = self.model.jnt_qposadr[joints], self.model.jnt_dofadr[joints]
        self.names = np.array([self.model.joint(int(joint)).name for joint in joints])
        with np.load(GAINS) as saved:
            if not np.array_equal(saved["joint_names"], self.names):
                raise ValueError("PD inertia joint order does not match actuator order")
            inertia = saved["inertia"].astype(np.float32)
        if inertia.shape != (NU,) or not np.all(np.isfinite(inertia) & (inertia > 0)):
            raise ValueError("Malformed actuator inertia")
        self.inertia = inertia / inertia.sum()
        mass = self.model.body_mass[1:].astype(np.float32)
        self.mass = mass / mass.sum()
        self.floor = self.model.geom("floor").id
        self.feet = ((self.model.geom("left_foot").id, self.model.geom("left_toes_chunk").id),
                     (self.model.geom("right_foot").id, self.model.geom("right_toes_chunk").id))
        self.forward = ForwardRuntime(self.model, self.floor, self.feet, device,
                                      self.jq.astype(np.int32), self.jd.astype(np.int32), self.names)
        self.device = device; self.joint_output = np.full(self.model.njnt, -1, np.int32)
        self.joint_output[joints] = np.arange(NU)
        for name in ("body_pos", "body_quat", "body_ipos", "jnt_pos", "jnt_axis", "qpos0"):
            setattr(self, "fk_"+name, torch.as_tensor(getattr(self.model, name), dtype=torch.float32, device=device))

    def sample(self, count, rng):
        horizons = np.clip(np.rint(np.clip(abs(rng.normal(0, DURATION_STD, count)),
            DURATION_MIN, DURATION_MAX)*HZ), DURATION_MIN*HZ, DURATION_MAX*HZ).astype(np.int64)
        result = np.empty((count, MAX_STEPS+2, FRAME_SIZE), np.float32)
        for row, horizon in enumerate(horizons):
            source = int(rng.integers(5))
            if source == 4:
                result[row, :horizon+1] = self.standing[int(rng.integers(len(self.standing)))]
            else:
                key = source, horizon
                if key not in self.valid:
                    choices = []
                    for scene, offset, length in self.records[source]:
                        maximum = length-horizon-1
                        if maximum < 0: continue
                        forbidden = [(max(0, a-horizon), min(maximum, b))
                            for a, b in self.blocked.get((source, scene), ()) if a-horizon <= maximum and b >= 0]
                        merged = []
                        for begin, end in sorted(forbidden):
                            if merged and begin <= merged[-1][1]+1: merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                            else: merged.append((begin, end))
                        segments, cursor = [], 0
                        for begin, end in merged:
                            if cursor < begin: segments.append((cursor, begin-1))
                            cursor = end+1
                        if cursor <= maximum: segments.append((cursor, maximum))
                        available = sum(end-begin+1 for begin, end in segments)
                        if available: choices.append((offset, segments, available))
                    self.valid[key] = choices
                choices = self.valid[key]
                if not choices: raise ValueError(f"Source {source} has no scene supporting {horizon} transitions")
                offset, segments, available = choices[int(rng.integers(len(choices)))]; choice = int(rng.integers(available))
                for begin, end in segments:
                    width = end-begin+1
                    if choice < width: start = begin+choice; break
                    choice -= width
                result[row, :horizon+1] = self.frames[offset+start:offset+start+horizon+1]
            result[row, horizon+1:] = result[row, horizon]
        return result, horizons

    def validation_batches(self):
        for start in range(0, len(self.validation), BATCH_SIZE):
            records = self.validation[start:start+BATCH_SIZE]
            frames = np.empty((BATCH_SIZE, MAX_STEPS+2, FRAME_SIZE), np.float32); frames[:] = self.standing[0]
            horizons = np.ones(BATCH_SIZE, np.int64)
            for row, record in enumerate(records):
                horizons[row] = len(record)-1; frames[row, :len(record)] = record; frames[row, len(record):] = record[-1]
            yield frames, horizons

    def state(self, frame, start_xy, contacts):
        qpos, qvel = frame[:, :NQ], frame[:, 126:126+NV]
        bodies = frame[:, NQ:126].reshape(-1, NBODY, 3)
        body_velocity = frame[:, 126+NV:].reshape(-1, NBODY, 3)
        root = qpos[:, :3]; relative = bodies.copy(); relative[:, :, :2] -= root[:, None, :2]
        relative_velocity = body_velocity-qvel[:, None, :3]
        rotation = qpos[:, 3:7].copy(); rotation /= np.maximum(np.linalg.norm(rotation, axis=-1, keepdims=True), 1e-12)
        rotation *= np.where(rotation[:, :1] < 0, -1., 1.)
        com = (bodies*self.mass[None, :, None]).sum(1)
        com_velocity = (body_velocity*self.mass[None, :, None]).sum(1)
        value = np.concatenate((root[:, :2]-start_xy, root[:, 2:3], relative[:, :, :2].reshape(-1, 48),
            bodies[:, :, 2], qvel[:, :3], relative_velocity.reshape(-1, 72), rotation, qvel[:, 3:6],
            qpos[:, self.jq], qvel[:, self.jd], contacts, com-root, com_velocity), -1)
        if value.shape[1] != STATE_SIZE: raise AssertionError("Endpoint state size changed")
        return value.astype(np.float32)

    def endpoints(self, frames, horizons):
        rows = np.arange(len(frames)); start, end = frames[:, 0], frames[rows, horizons]
        qpos = np.concatenate((start[:, :NQ], end[:, :NQ]))
        qvel = np.concatenate((start[:, 126:126+NV], end[:, 126:126+NV]))
        contacts = self.forward.contacts(qpos, qvel); count = len(start); start_xy = start[:, :2]
        return self.state(start, start_xy, contacts[:count]), self.state(end, start_xy, contacts[count:])

    def path(self, frames):
        qpos = frames[:, :, :NQ]
        root = qpos[:, :, :3].copy(); root[:, :, :2] -= root[:, :1, :2]
        quaternion = qpos[:, :, 3:7].copy()
        quaternion /= np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12)
        quaternion *= np.where(quaternion[:, :1, :1] < 0, -1., 1.)
        flips = np.where((quaternion[:, 1:]*quaternion[:, :-1]).sum(-1) < 0, -1., 1.)
        quaternion[:, 1:] *= np.cumprod(flips, axis=1)[..., None]
        return np.concatenate((root, quaternion, frames[:, :, self.jq]), -1).astype(np.float32)

    def fk(self, path):
        root_pos, root_quat = path[..., ROOT_XYZ], normalize_quaternion(path[..., ROOT_QUAT])
        positions, quaternions, centers = [None]*self.model.nbody, [None]*self.model.nbody, []
        positions[0], quaternions[0] = torch.zeros_like(root_pos), torch.zeros_like(root_quat)
        quaternions[0][..., 0] = 1; positions[1], quaternions[1] = root_pos, root_quat
        centers.append(root_pos+quat_rotate(root_quat, self.fk_body_ipos[1]))
        for body in range(2, self.model.nbody):
            parent = int(self.model.body_parentid[body]); parent_pos, parent_quat = positions[parent], quaternions[parent]
            pos = parent_pos+quat_rotate(parent_quat, self.fk_body_pos[body])
            quat = quat_multiply(parent_quat, self.fk_body_quat[body])
            address, number = int(self.model.body_jntadr[body]), int(self.model.body_jntnum[body])
            for joint in range(address, address+number):
                output = int(self.joint_output[joint]); anchor = pos+quat_rotate(quat, self.fk_jnt_pos[joint])
                angle = path[..., JOINT][..., output]-self.fk_qpos0[int(self.model.jnt_qposadr[joint])]
                quat = quat_multiply(quat, axis_angle_quaternion(self.fk_jnt_axis[joint], angle))
                pos = anchor-quat_rotate(quat, self.fk_jnt_pos[joint])
            quat = nn.functional.normalize(quat, dim=-1); positions[body], quaternions[body] = pos, quat
            centers.append(pos+quat_rotate(quat, self.fk_body_ipos[body]))
        return torch.stack(centers, -2)

    def validate_fk(self):
        indices = np.linspace(0, len(self.frames)-1, min(64, len(self.frames)), dtype=np.int64)
        frames = np.asarray(self.frames[indices], dtype=np.float32)[:, None]
        predicted = self.fk(torch.from_numpy(self.path(frames)[:, 0]).to(self.device))
        target = frames[:, 0, NQ:126].reshape(-1, NBODY, 3).copy()
        target[..., :2] -= frames[:, 0, None, :2]
        target = torch.from_numpy(target).to(self.device)
        maximum = float((predicted-target).abs().max())
        if maximum > 5e-4: raise RuntimeError(f"Differentiable FK disagrees with cached MuJoCo bodies: max={maximum:.6g} m")


def normalize_quaternion(value):
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True); identity = torch.zeros_like(value)
    identity[..., 0] = 1
    return torch.where(norm > 1e-6, value/norm.clamp_min(1e-6), identity)


def quat_multiply(first, second):
    w1, xyz1, w2, xyz2 = first[..., :1], first[..., 1:], second[..., :1], second[..., 1:]
    return torch.cat((w1*w2-(xyz1*xyz2).sum(-1, keepdim=True),
        w1*xyz2+w2*xyz1+torch.linalg.cross(xyz1, xyz2.expand_as(xyz1), dim=-1)), -1)


def quat_rotate(quaternion, value):
    xyz = quaternion[..., 1:]; cross = torch.linalg.cross(xyz, value.expand_as(xyz), dim=-1)
    return value+2*(quaternion[..., :1]*cross+torch.linalg.cross(xyz, cross, dim=-1))


def axis_angle_quaternion(axis, angle):
    half = angle/2
    return torch.cat((torch.cos(half)[..., None], axis.expand(*angle.shape, 3)*torch.sin(half)[..., None]), -1)


class Moments:
    def __init__(self, size): self.sum = np.zeros(size, np.float64); self.square = np.zeros(size, np.float64); self.count = 0
    def add(self, values):
        values = values.reshape(-1, values.shape[-1]).astype(np.float64)
        self.sum += values.sum(0); self.square += (values*values).sum(0); self.count += len(values)
    def result(self):
        mean = self.sum/self.count; std = np.sqrt(np.maximum(self.square/self.count-mean*mean, 0))
        return mean.astype(np.float32), np.where(std > 1e-7, std, 1).astype(np.float32)


def feature_statistics(data):
    rng = np.random.default_rng(SEED+1); state, path = Moments(STATE_SIZE), Moments(PATH_SIZE)
    duration, timing = Moments(1), Moments(2)
    done = 0
    while done < NORMALIZATION_SAMPLES:
        count = min(BATCH_SIZE, NORMALIZATION_SAMPLES-done); frames, horizons = data.sample(count, rng)
        begin, end = data.endpoints(frames, horizons); state.add(begin); state.add(end)
        values = data.path(frames); mask = np.arange(values.shape[1])[None] <= horizons[:, None]
        step = np.arange(1, MAX_STEPS+1)[None]; valid = step < horizons[:, None]
        times = np.stack(((horizons[:, None]-step)/horizons[:, None], step/horizons[:, None]), -1)
        path.add(values[mask]); duration.add((horizons/HZ)[:, None]); timing.add(times[valid]); done += count
    sm, ss = state.result(); pm, ps = path.result(); dm, ds = duration.result(); tm, ts = timing.result()
    ss[STATE_JOINT] = np.maximum(ss[STATE_JOINT], ANGLE_STD_FLOOR)
    ps[JOINT] = np.maximum(ps[JOINT], ANGLE_STD_FLOOR)
    return {"state_mean":sm, "state_std":ss, "path_mean":pm, "path_std":ps,
            "duration_mean":dm, "duration_std":ds, "time_mean":tm, "time_std":ts}


class Planner(nn.Module):
    def __init__(self, stats):
        super().__init__()
        layers = []
        for source in (2*STATE_SIZE+1,)+(ENCODER_WIDTH,)*3: layers += [nn.Linear(source, ENCODER_WIDTH), nn.SiLU()]
        self.encoder = nn.Sequential(*layers)
        self.hidden, self.cell = (nn.Linear(ENCODER_WIDTH, 2*DECODER_WIDTH) for _ in range(2))
        self.decoder = nn.LSTM(2*PATH_SIZE+2, DECODER_WIDTH, 2, batch_first=True)
        self.output = nn.Sequential(nn.Linear(DECODER_WIDTH, DECODER_WIDTH), nn.SiLU(), nn.Linear(DECODER_WIDTH, PATH_SIZE))
        self.register_buffer("quaternion_mean", torch.from_numpy(stats["path_mean"][ROOT_QUAT]))
        self.register_buffer("quaternion_std", torch.from_numpy(stats["path_std"][ROOT_QUAT]))
        self.register_buffer("quaternion_identity", torch.tensor((1, 0, 0, 0), dtype=torch.float32), persistent=False)

    def forward(self, start, end, duration, first, last, timing):
        context = self.encoder(torch.cat((start, end, duration), -1)); batch = len(start)
        hidden = self.hidden(context).view(batch, 2, DECODER_WIDTH).transpose(0, 1).contiguous()
        cell = self.cell(context).view(batch, 2, DECODER_WIDTH).transpose(0, 1).contiguous()
        value, result = first, []; history = first[:, None].expand(-1, SMOOTH, -1).clone()
        for step in range(MAX_STEPS):
            difference = last-value
            current_quaternion = normalize_quaternion(value[..., ROOT_QUAT]*self.quaternion_std+self.quaternion_mean)
            last_quaternion = normalize_quaternion(last[..., ROOT_QUAT]*self.quaternion_std+self.quaternion_mean)
            relative = quat_multiply(torch.cat((current_quaternion[..., :1], -current_quaternion[..., 1:]), -1),
                                     last_quaternion)
            relative = torch.where(relative[..., :1] < 0, -relative, relative)
            relative = relative-self.quaternion_identity
            remaining = torch.cat((difference[..., ROOT_XYZ], relative, difference[..., JOINT]), -1)
            decoder_input = torch.cat((value, timing[:, step], remaining), -1)
            raw, (hidden, cell) = self.decoder(decoder_input[:, None], (hidden, cell))
            raw = self.output(raw[:, 0]); history = torch.cat((history[:, 1:], raw[:, None]), 1)
            reference = normalize_quaternion(value[..., ROOT_QUAT]*self.quaternion_std+self.quaternion_mean)[:, None]
            value = history.mean(1); quaternion = normalize_quaternion(
                history[..., ROOT_QUAT]*self.quaternion_std+self.quaternion_mean)
            quaternion = torch.where((quaternion*reference).sum(-1, keepdim=True) < 0, -quaternion, quaternion)
            quaternion = (normalize_quaternion(quaternion.mean(1))-self.quaternion_mean)/self.quaternion_std
            value = torch.cat((value[..., ROOT_XYZ], quaternion, value[..., JOINT]), -1); result.append(value)
        return torch.stack(result, 1)


def tensors(frames, horizons, data, stats, device):
    start, end = data.endpoints(frames, horizons); path = data.path(frames)
    bodies = frames[:, :, NQ:126].reshape(len(frames), -1, NBODY, 3).copy()
    bodies[..., :2] -= frames[:, :1, None, :2]
    bodies = torch.from_numpy(bodies).to(device)
    sm, ss = stats["state_mean"], stats["state_std"]; pm, ps = stats["path_mean"], stats["path_std"]
    start = torch.from_numpy((start-sm)/ss).to(device); end = torch.from_numpy((end-sm)/ss).to(device)
    duration = torch.from_numpy(((horizons[:, None]/HZ-stats["duration_mean"])/stats["duration_std"])
                                .astype(np.float32)).to(device)
    path = torch.from_numpy(path).to(device); pm, ps = torch.as_tensor(pm, device=device), torch.as_tensor(ps, device=device)
    first, last = (path[:, 0]-pm)/ps, (path[torch.arange(len(path), device=device),
        torch.as_tensor(horizons, device=device)]-pm)/ps
    step = np.arange(1, MAX_STEPS+1)[None]; raw_time = np.stack((
        np.maximum(horizons[:, None]-step, 0)/horizons[:, None], step/horizons[:, None]), -1)
    timing = torch.from_numpy(((raw_time-stats["time_mean"])/stats["time_std"]).astype(np.float32)).to(device)
    return start, end, duration, first, last, timing, path, bodies


def component_loss(prediction, target, target_bodies, horizons, data):
    device = prediction.device; steps = prediction.shape[1]
    error = prediction-target[:, 1:steps+1]
    mask = torch.arange(steps, device=device)[None] < torch.as_tensor(horizons-1, device=device)[:, None]
    root, angle = error[..., ROOT_XYZ], error[..., JOINT]
    xyz = data.fk(prediction)-target_bodies[:, 1:steps+1]
    mass = torch.as_tensor(data.mass, device=device); inertia = torch.as_tensor(data.inertia, device=device)
    root_xy, root_z = root[..., :2].square().mean(-1), root[..., 2].square()
    body_xy = (xyz[..., :2].square().mean(-1)*mass).sum(-1)
    body_z = (xyz[..., 2].square()*mass).sum(-1)
    predicted_quaternion = normalize_quaternion(prediction[..., ROOT_QUAT])
    target_quaternion = normalize_quaternion(target[:, 1:steps+1, ROOT_QUAT])
    dot = (predicted_quaternion*target_quaternion).sum(-1).abs().clamp(max=1)
    orientation = torch.clamp(1-dot.square(), min=0); joint_squared = (angle.square()*inertia).sum(-1)
    components = torch.stack(tuple(value[mask].mean() for value in
        (root_xy, body_xy, root_z, body_z, orientation, joint_squared)))
    report = torch.stack(tuple(value[mask].mean() for value in ((root_xy+body_xy)/2,
        (root_z+body_z)/2, (2*torch.acos(dot)).square(), joint_squared))).detach()
    return components, report


def normalized_loss(components, scales):
    value = components/scales
    return torch.stack(((value[0]+value[1])/2, (value[2]+value[3])/2, value[4], value[5]))


def readable_metrics(report):
    return torch.sqrt(torch.clamp(report, min=0))*report.new_tensor((1, 1, 180/np.pi, 180/np.pi))


def correct_prediction(prediction, horizons, data):
    if not CORRECT_BEFORE_LOSS: return prediction
    flat = prediction.detach().reshape(-1, PATH_SIZE); count = len(flat)
    qpos = data.forward.default_qpos[0].expand(count, -1).clone()
    qpos[:, :3] = flat[:, ROOT_XYZ]; qpos[:, 3:7] = normalize_quaternion(flat[:, ROOT_QUAT])
    joints = torch.as_tensor(data.jq, dtype=torch.long, device=prediction.device)
    qpos[:, joints] = flat[:, JOINT]
    active = (torch.arange(MAX_STEPS, device=prediction.device)[None]
              < torch.as_tensor(horizons-1, device=prediction.device)[:, None]).flatten()
    corrected_qpos = data.forward.correct(qpos, active)
    corrected = flat.clone(); corrected[:, ROOT_XYZ] = corrected_qpos[:, :3]
    corrected[:, JOINT] = corrected_qpos[:, joints]
    corrected = corrected.view_as(prediction)
    return prediction+(corrected-prediction.detach())


@torch.no_grad()
def loss_scales(model, data, stats, device):
    rng = np.random.default_rng(SEED+2); total, frames_count = torch.zeros(6, device=device), 0
    pm, ps = (torch.as_tensor(stats[key], device=device) for key in ("path_mean", "path_std"))
    for _ in range(LOSS_CALIBRATION_BATCHES):
        frames, horizons = data.sample(BATCH_SIZE, rng); values = tensors(frames, horizons, data, stats, device)
        prediction = correct_prediction(model(*values[:6])*ps+pm, horizons, data)
        count = int((horizons-1).sum()); components, _ = component_loss(prediction, values[6], values[7], horizons, data)
        total += components*count
        frames_count += count
    result = (total/frames_count).cpu().numpy()
    if not np.all(np.isfinite(result) & (result > 0)): raise ValueError("Initial loss calibration failed")
    return result


@torch.no_grad()
def validation_loss(model, data, stats, scales, device):
    total, report_total, count = torch.zeros(6, device=device), torch.zeros(4, device=device), 0
    pm, ps = (torch.as_tensor(stats[key], device=device) for key in ("path_mean", "path_std"))
    for frames, horizons in data.validation_batches():
        values = tensors(frames, horizons, data, stats, device)
        prediction = correct_prediction(model(*values[:6])*ps+pm, horizons, data)
        valid = int((horizons-1).sum()); components, report = component_loss(prediction, values[6], values[7], horizons, data)
        total += components*valid; report_total += report*valid
        count += valid
    return normalized_loss(total/count, scales), readable_metrics(report_total/count), count


def save_normalization(stats, scales, data):
    temporary = NORMALIZATION.with_suffix(".temporary.npz")
    np.savez(temporary, schema=7, joint_names=data.names, smooth=SMOOTH,
             correction=CORRECT_BEFORE_LOSS, loss_scales=scales, **stats)
    os.replace(temporary, NORMALIZATION)


def checkpoint(batch, model, optimizer, rng):
    temporary = CHECKPOINT.with_suffix(".temporary.pt")
    torch.save({"schema":7, "batch":batch, "model":model.state_dict(),
        "optimizer":optimizer.state_dict(), "numpy_rng":rng.bit_generator.state,
        "torch_rng":torch.get_rng_state(), "cuda_rng":torch.cuda.get_rng_state() if torch.cuda.is_available() else None}, temporary)
    os.replace(temporary, CHECKPOINT)


def metric_history(batch):
    rows = []
    if METRICS_CSV.is_file():
        with METRICS_CSV.open(newline="") as file:
            reader = csv.DictReader(file)
            if tuple(reader.fieldnames or ()) != METRIC_FIELDS: raise ValueError("Goal-planner metrics CSV is incompatible")
            rows = [row for row in reader if int(row["batch"]) <= batch]
    temporary = METRICS_CSV.with_suffix(".temporary.csv")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, METRICS_CSV); return rows


def record_metric(rows, batch, split, loss, normalized, readable, frames, gradient="", elapsed=""):
    values = [float(x) for x in normalized]; physical = [float(x) for x in readable]
    row = dict(zip(METRIC_FIELDS, (batch, split, float(loss), *values, *physical,
        gradient, elapsed, int(frames))))
    with METRICS_CSV.open("a", newline="") as file: csv.DictWriter(file, fieldnames=METRIC_FIELDS).writerow(row)
    rows.append({key:str(value) for key, value in row.items()})


def plot_metrics(rows):
    settings = (("loss", "Normalized loss"), ("xy_rmse_m", "XY RMSE (m)"),
        ("z_rmse_m", "Z RMSE (m)"), ("root_rotation_rmse_deg", "Root rotation RMSE (deg)"),
        ("joint_rmse_deg", "Joint RMSE (deg)"), ("gradient_norm", "Gradient norm"))
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, (key, title) in zip(axes.flat, settings):
        for split, style in (("train", "-"), ("validation", "o")):
            selected = [row for row in rows if row["split"] == split and row[key] != ""]
            if selected: axis.plot([int(row["batch"]) for row in selected],
                [float(row[key]) for row in selected], style, markersize=3, linewidth=.8, label=split)
        axis.set_title(title); axis.set_xlabel("Batch"); axis.grid(alpha=.25)
        if key != "gradient_norm": axis.legend()
    figure.tight_layout(); temporary = METRICS_PNG.with_suffix(".temporary.png")
    figure.savefig(temporary, dpi=140); plt.close(figure); os.replace(temporary, METRICS_PNG)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--stop", type=int); args = parser.parse_args()
    if args.stop is not None and args.stop < 0: parser.error("--stop must be nonnegative")
    if not torch.cuda.is_available(): raise RuntimeError("Goal Planner requires CUDA for batched MJWarp forward")
    torch.cuda.set_device(0); wp.set_device(wp.get_cuda_device(0)); device = torch.device("cuda")
    np.random.seed(SEED); torch.manual_seed(SEED); data = MotionData(device); data.validate_fk()
    if NORMALIZATION.is_file():
        with np.load(NORMALIZATION) as saved:
            if (int(saved["schema"]) != 7 or int(saved["smooth"]) != SMOOTH
                    or bool(saved["correction"]) != CORRECT_BEFORE_LOSS
                    or not np.array_equal(saved["joint_names"], data.names)):
                raise ValueError("Goal-planner normalization is incompatible")
            stats = {key:saved[key].copy() for key in ("state_mean", "state_std", "path_mean", "path_std",
                                                        "duration_mean", "duration_std", "time_mean", "time_std")}
            scales = saved["loss_scales"].copy()
        expected = {"state_mean":(STATE_SIZE,), "state_std":(STATE_SIZE,),
                    "path_mean":(PATH_SIZE,), "path_std":(PATH_SIZE,),
                    "duration_mean":(1,), "duration_std":(1,), "time_mean":(2,), "time_std":(2,)}
        if any(stats[key].shape != shape or not np.all(np.isfinite(stats[key]))
               for key, shape in expected.items()) or np.shape(scales) != (6,) or not np.all(np.isfinite(scales) & (scales > 0)):
            raise ValueError("Malformed goal-planner normalization")
    else:
        print("Measuring fixed input normalization...", flush=True); stats = feature_statistics(data); scales = None
    model = Planner(stats).to(device)
    if scales is None:
        print("Calibrating initial root/body/rotation/joint losses...", flush=True)
        scales = loss_scales(model, data, stats, device); save_normalization(stats, scales, data)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    batch, rng = 0, np.random.default_rng(SEED+3)
    if CHECKPOINT.is_file():
        saved = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        if saved.get("schema") != 7: raise ValueError("Goal-planner checkpoint is incompatible")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups: group["lr"] = LEARNING_RATE
        batch = int(saved["batch"]); rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved["cuda_rng"] is not None: torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
    history = metric_history(batch)
    sample = (torch.zeros(BATCH_SIZE, STATE_SIZE, device=device),
              torch.zeros(BATCH_SIZE, STATE_SIZE, device=device),
              torch.zeros(BATCH_SIZE, 1, device=device),
              torch.zeros(BATCH_SIZE, PATH_SIZE, device=device),
              torch.zeros(BATCH_SIZE, PATH_SIZE, device=device),
              torch.zeros(BATCH_SIZE, MAX_STEPS, 2, device=device))
    torch.cuda.synchronize(); graphed_model = torch.cuda.make_graphed_callables(model, sample)
    optimizer.zero_grad(set_to_none=True)
    weights, scales = LOSS_WEIGHTS.to(device), torch.as_tensor(scales, device=device)
    try:
        while args.stop is None or batch < args.stop:
            torch.cuda.synchronize(); started = time.perf_counter()
            frames, horizons = data.sample(BATCH_SIZE, rng); values = tensors(frames, horizons, data, stats, device)
            pm, ps = (torch.as_tensor(stats[key], device=device) for key in ("path_mean", "path_std"))
            prediction = correct_prediction(graphed_model(*values[:6])*ps+pm, horizons, data)
            components, report = component_loss(prediction, values[6], values[7], horizons, data)
            normalized = normalized_loss(components, scales); loss = (weights*normalized).sum()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0)); optimizer.step()
            torch.cuda.synchronize(); elapsed = time.perf_counter()-started; batch += 1
            readable, valid = readable_metrics(report), int((horizons-1).sum())
            record_metric(history, batch, "train", loss.detach(), normalized.detach(), readable, valid, gradient, elapsed)
            if batch % PRINT_INTERVAL == 0:
                print(f"train batch={batch} loss={loss.item():.6g} xy={readable[0]:.5g}m "
                      f"z={readable[1]:.5g}m root_rot={readable[2]:.5g}deg "
                      f"joints={readable[3]:.5g}deg grad={gradient:.5g} time={elapsed:.3f}s", flush=True)
            if batch % VALID_INTERVAL == 0:
                value, physical, valid = validation_loss(graphed_model, data, stats, scales, device)
                validation_value = (weights*value).sum()
                record_metric(history, batch, "validation", validation_value, value, physical, valid)
                print(f"valid batch={batch} loss={validation_value.item():.6g} xy={physical[0]:.5g}m "
                      f"z={physical[1]:.5g}m root_rot={physical[2]:.5g}deg "
                      f"joints={physical[3]:.5g}deg frames={valid}", flush=True)
                plot_metrics(history)
            if batch % CHECKPOINT_INTERVAL == 0: checkpoint(batch, model, optimizer, rng)
    except KeyboardInterrupt: pass
    if history: plot_metrics(history)
    checkpoint(batch, model, optimizer, rng)


if __name__ == "__main__": main()
