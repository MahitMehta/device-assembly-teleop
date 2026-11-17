"""Pose helpers shared by the dataset, the policy and the sim (numpy, no torch)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

OPEN_M = 0.085   # gripper stroke (m)

# Optional proprioceptive inputs (config `state`), per observed frame, all in ~[-1, 1]:
#   gripper_width  measured jaw gap
#   gravity_tool   unit "down" vector in the tool0 frame: tilt (roll/pitch) without yaw, so the
#                  policy knows how far it is from vertical but must get yaw from the images
#   ee_rot6d       full tool0 orientation in base (6D rotation), yaw included
#   wrench         the UR's wrist force/torque (tool0_controller frame), as WRENCH_FILTER describes:
#                  mean of the last 0.1 s of the 500 Hz stream (removes the arm's own inertia
#                  spikes), minus its median over the 0.5 s before the episode/run starts; scaled
#                  so the demos' insertion pushes (~10-40 N, ~1-4 N m) land around +/-1
# Not roll/pitch/yaw angles: with the tool pointing down roll sits at +/-180 deg and wraps.
STATE_DIMS = {"gripper_width": 1, "gravity_tool": 3, "ee_rot6d": 6, "wrench": 6}
WRENCH_FILTER = {"window_s": 0.1, "zero_window_s": 0.5}   # the robot node must do exactly this too
WRENCH_SCALE = np.array([30.0, 30.0, 30.0, 3.0, 3.0, 3.0])   # N, N m -> ~[-1, 1]
WRENCH_CLIP = 2.0


def state_dim(keys) -> int:
    return sum(STATE_DIMS[k] for k in keys)


def state_vector(tool0: np.ndarray, gripper_width: float, keys, wrench: np.ndarray | None = None) -> np.ndarray:
    """tool0 (4,4) measured pose in base, gripper_width measured jaw gap (m), wrench (6,) filtered and
    zeroed force (N) + torque (N m) -> (state_dim,) in ~[-1, 1]."""
    parts = []
    for k in keys:
        if k == "gripper_width":
            parts.append([float(np.clip(gripper_width, 0.0, OPEN_M)) / OPEN_M * 2 - 1])
        elif k == "gravity_tool":
            parts.append(np.asarray(tool0)[:3, :3].T @ np.array([0.0, 0.0, -1.0]))
        elif k == "ee_rot6d":
            parts.append(mat_to_rot6d(np.asarray(tool0)[:3, :3]))
        elif k == "wrench":
            if wrench is None:
                raise ValueError("the wrench state input needs the measured force/torque")
            parts.append(np.clip(np.asarray(wrench, np.float64) / WRENCH_SCALE, -WRENCH_CLIP, WRENCH_CLIP))
        else:
            raise ValueError(f"unknown state input {k!r}; known: {list(STATE_DIMS)}")
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)


def pose7_to_mat(p: np.ndarray) -> np.ndarray:
    """(..., 7) xyz + quat wxyz -> (..., 4, 4)."""
    p = np.asarray(p, dtype=np.float64)
    T = np.zeros(p.shape[:-1] + (4, 4))
    T[..., :3, :3] = R.from_quat(p.reshape(-1, 7)[:, [4, 5, 6, 3]]).as_matrix().reshape(p.shape[:-1] + (3, 3))
    T[..., :3, 3] = p[..., :3]
    T[..., 3, 3] = 1.0
    return T


# Nominal UR5e kinematics (UR's published DH), the same as teleop's quest_teleop/utils/core.py `fk`.
# Datasets use this for the measured tool0 pose (convert_rosbags --pose-source nominal_fk), not the
# driver's /tcp_pose_broadcaster/pose: the driver reports the controller's factory-calibrated
# kinematics, while teleop commanded (target_pose) and the robot node executes through the nominal
# ones. Mixing them made "hold still" a constant ~1.2 mm / 0.3 deg action on the lab arm (different
# on every arm), which the robot then executed every plan. One model end to end: still = zero, on any UR5e.
UR5E_A = (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0)
UR5E_D = (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996)
UR5E_ALPHA = (np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0)
UR5E_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def ur5e_fk(q: np.ndarray) -> np.ndarray:
    """(..., 6) joint angles -> (..., 4, 4) tool0 in `base`, nominal UR5e kinematics."""
    q = np.asarray(q, dtype=np.float64)
    T = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
    for i in range(6):
        ct, st = np.cos(q[..., i]), np.sin(q[..., i])
        ca, sa = np.cos(UR5E_ALPHA[i]), np.sin(UR5E_ALPHA[i])
        A = np.zeros(q.shape[:-1] + (4, 4))
        A[..., 0, 0], A[..., 0, 1], A[..., 0, 2], A[..., 0, 3] = ct, -st * ca, st * sa, UR5E_A[i] * ct
        A[..., 1, 0], A[..., 1, 1], A[..., 1, 2], A[..., 1, 3] = st, ct * ca, -ct * sa, UR5E_A[i] * st
        A[..., 2, 1], A[..., 2, 2], A[..., 2, 3] = sa, ca, UR5E_D[i]
        A[..., 3, 3] = 1.0
        T = T @ A
    return T


def mat_to_pose7(T: np.ndarray) -> np.ndarray:
    q = R.from_matrix(T[..., :3, :3].reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(T.shape[:-2] + (4,))
    return np.concatenate([T[..., :3, 3], q], -1)


def mat_to_rot6d(Rm: np.ndarray) -> np.ndarray:
    """First two columns of R, flattened column-wise (Zhou et al. 2019)."""
    return np.concatenate([Rm[..., :, 0], Rm[..., :, 1]], -1)


def rot6d_to_mat(r: np.ndarray) -> np.ndarray:
    a, b = r[..., :3], r[..., 3:6]
    x = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (x * b).sum(-1, keepdims=True) * x
    y = b / np.linalg.norm(b, axis=-1, keepdims=True)
    z = np.cross(x, y)
    return np.stack([x, y, z], -1)


def poses_to_action(anchor: np.ndarray, targets: np.ndarray, grip: np.ndarray, space: str) -> np.ndarray:
    """anchor (4,4) current tool0; targets (H,4,4) future tool0 targets; grip (H,) -> (H, 10).

    rel_ee: targets in the current tool0 frame (UMI-style; needs no proprioception as input, only
            the current pose to execute). abs_ee: targets in `base`."""
    T = np.linalg.inv(anchor) @ targets if space == "rel_ee" else targets
    return np.concatenate([T[:, :3, 3], mat_to_rot6d(T[:, :3, :3]), grip[:, None]], -1).astype(np.float32)


def action_to_poses(anchor: np.ndarray, action: np.ndarray, space: str) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of poses_to_action: (H,10) -> tool0 targets in base (H,4,4), gripper opening (H,)."""
    T = np.tile(np.eye(4), (len(action), 1, 1))
    T[:, :3, :3] = rot6d_to_mat(action[:, 3:9])
    T[:, :3, 3] = action[:, :3]
    if space == "rel_ee":
        T = anchor @ T
    return T, action[:, 9]
