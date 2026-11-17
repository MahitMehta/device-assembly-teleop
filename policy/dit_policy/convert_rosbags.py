"""Converts teleop rosbags (teleop/scripts/record.sh) into the flat dataset the DiT trains on.

Pure Python (the `rosbags` library), so it runs in any container without ROS:

    python -m dit_policy.convert_rosbags /data/teleop_demos/clear-piece out/clear_piece_v2

Everything is resampled at --hz on the bag's receive clock, which all topics share (the camera's own
header stamps come from the laptop's clock). Output directory:

    images.npy        (N, H, W, 3) uint8   wrist colour, resized
    tcp_pose.npy      (N, 7) float32       measured tool0 pose in `base`: xyz + quat (wxyz). With
                                           --pose-source nominal_fk (default) it is geometry.ur5e_fk of
                                           /joint_states, the kinematics teleop commanded through and
                                           the robot node executes with; with `driver` it is
                                           /tcp_pose_broadcaster/pose (the controller's calibrated
                                           kinematics: older datasets, see geometry.ur5e_fk)
    driver_tcp_pose.npy (N, 7) float32     /tcp_pose_broadcaster/pose, kept for reference
    target_pose.npy   (N, 7) float32       teleop's commanded tool0 pose (/teleop/target_pose)
    gripper_cmd.npy   (N,)   float32       last commanded finger opening (m; 0 closed, 0.085 open)
    gripper_width.npy (N,)   float32       measured finger opening (m)
    wrench.npy        (N, 6) float32       wrist force (N) + torque (N m) in tool0_controller, filtered
                                           and zeroed per episode as geometry.WRENCH_FILTER says
    episode_ends.npy  (E,)   int64         exclusive end index of each episode
    meta.json                              source bags, rate, camera intrinsics, trimming

Actions are not baked in: the dataset turns poses into whatever action space the config asks for.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation, Slerp

from .geometry import UR5E_JOINTS, WRENCH_FILTER, mat_to_pose7, ur5e_fk

IMG = "/camera/camera/color/image_raw/compressed"
INFO = "/camera/camera/color/camera_info"
TCP = "/tcp_pose_broadcaster/pose"
TARGET = "/teleop/target_pose"
GRIP_CMD = "/teleop/gripper_command"
GRIP_POS = "/teleop/gripper_position"
WRENCH = "/force_torque_sensor_broadcaster/wrench"
JOINTS = "/joint_states"
OPEN_M = 0.085
WRENCH_WINDOW_S, WRENCH_ZERO_S = WRENCH_FILTER["window_s"], WRENCH_FILTER["zero_window_s"]


def _pose(msg) -> list[float]:
    p, q = msg.pose.position, msg.pose.orientation
    return [p.x, p.y, p.z, q.w, q.x, q.y, q.z]


def _interp_poses(t_src: np.ndarray, poses: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear in position, slerp in rotation; clamps outside the source span."""
    t = np.clip(t, t_src[0], t_src[-1])
    xyz = np.stack([np.interp(t, t_src, poses[:, i]) for i in range(3)], 1)
    rot = Slerp(t_src, Rotation.from_quat(poses[:, [4, 5, 6, 3]]))(t).as_quat()  # xyzw
    return np.concatenate([xyz, rot[:, [3, 0, 1, 2]]], 1)


def repair_target_quats(tgt_t: np.ndarray, tgt: np.ndarray, tcp_t: np.ndarray, tcp: np.ndarray,
                        eps: float = 0.02) -> tuple[np.ndarray, int]:
    """Fixes the sign errors in teleop's published orientation (poses: xyz + quat wxyz).

    teleop_node._quat takes each of x, y, z's sign from R[2,1]-R[1,2] etc. (= 4·w·q_i). Near a 180°
    rotation (gripper pointing down: w ≈ 0) those are ~0 and their signs are noise, so the published
    rotation jumps by up to 180°. The magnitudes are right, so: for each sample, among the sign
    choices of the components whose |w·q_i| < eps, keep the one closest to the measured tool0
    orientation at that time. Reliable signs are never touched. Returns (fixed, n_changed)."""
    out = tgt.copy()
    ci = np.clip(np.searchsorted(tcp_t, tgt_t), 0, len(tcp_t) - 1)
    ref = tcp[ci, 3:]                                   # wxyz
    q = tgt[:, 3:]
    amb = np.abs(q[:, :1] * q[:, 1:]) < eps             # (N, 3) ambiguous x/y/z signs
    signs = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)], float)
    best = np.abs(np.sum(q * ref, 1))                   # |<q, ref>| = cos(half-angle)
    changed = np.zeros(len(q), bool)
    for s in signs[1:]:
        ok = np.all(amb | (s > 0), axis=1)               # flips only where ambiguous
        cand = np.concatenate([q[:, :1], q[:, 1:] * s], 1)
        d = np.abs(np.sum(cand * ref, 1))
        better = ok & (d > best + 1e-9)
        out[better, 3:] = cand[better]; best[better] = d[better]; changed |= better
    return out, int(changed.sum())


def _hold(t_src: np.ndarray, vals: np.ndarray, t: np.ndarray, before: float) -> np.ndarray:
    """Zero-order hold: the last value at or before each t."""
    if len(t_src) == 0:
        return np.full(len(t), before, np.float32)
    i = np.searchsorted(t_src, t, side="right") - 1
    return np.where(i >= 0, vals[np.maximum(i, 0)], before).astype(np.float32)


def wrench_at(t_src: np.ndarray, w: np.ndarray, grid: np.ndarray, window: float = WRENCH_WINDOW_S,
              zero: float = WRENCH_ZERO_S) -> np.ndarray:
    """Causal filter of the 500 Hz wrench at each grid time: the mean of the samples in
    (t - window, t], minus the median over the `zero` s before the first grid time (the arm is
    still and touching nothing there, as when a policy run starts). Zeros if there's no wrench."""
    if len(t_src) < 2:
        return np.zeros((len(grid), 6), np.float32)
    c = np.vstack([np.zeros((1, 6)), np.cumsum(w, 0)])
    hi = np.searchsorted(t_src, grid, side="right")
    lo = np.searchsorted(t_src, grid - window, side="right")
    n = np.maximum(hi - lo, 1)[:, None]
    mean = (c[hi] - c[lo]) / n
    last = w[np.clip(hi - 1, 0, len(w) - 1)]
    mean = np.where((hi - lo)[:, None] > 0, mean, last)          # a gap in the stream: hold the last sample
    z = (t_src > grid[0] - zero) & (t_src <= grid[0])
    bias = np.median(w[z], 0) if z.any() else w[np.clip(hi[0] - 1, 0, len(w) - 1)]
    return (mean - bias).astype(np.float32)


def read_bag(path: pathlib.Path, hz: float, size: tuple[int, int], still_speed: float,
             pose_source: str = "nominal_fk"):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    streams = {k: ([], []) for k in (IMG, TCP, TARGET, GRIP_CMD, GRIP_POS, WRENCH, JOINTS)}
    info = None
    with AnyReader([path], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in streams or c.topic == INFO]
        for conn, t_ns, raw in reader.messages(connections=conns):
            t = t_ns * 1e-9
            if conn.topic == INFO:
                if info is None:
                    m = reader.deserialize(raw, conn.msgtype)
                    info = {"K": np.asarray(m.k).tolist(), "width": int(m.width), "height": int(m.height)}
                continue
            if conn.topic == IMG:  # keep the JPEG bytes; decode only the frames we keep
                m = reader.deserialize(raw, conn.msgtype)
                val = bytes(m.data)
            else:
                m = reader.deserialize(raw, conn.msgtype)
                if conn.topic in (TCP, TARGET):
                    val = _pose(m)
                elif conn.topic == WRENCH:
                    f, q = m.wrench.force, m.wrench.torque
                    val = [f.x, f.y, f.z, q.x, q.y, q.z]
                elif conn.topic == JOINTS:
                    index = {n: i for i, n in enumerate(m.name)}
                    if not all(n in index for n in UR5E_JOINTS):
                        continue
                    val = [float(m.position[index[n]]) for n in UR5E_JOINTS]
                else:
                    val = float(m.data)
            streams[conn.topic][0].append(t)
            streams[conn.topic][1].append(val)

    ts = {k: np.asarray(v[0]) for k, v in streams.items()}
    for k in (IMG, TCP, TARGET) + ((JOINTS,) if pose_source == "nominal_fk" else ()):
        if len(ts[k]) < 2:
            raise ValueError(f"{path.name}: no {k}")

    # Trim the idle head and tail: keep from the first to the last moment the commanded pose moves.
    tgt_t, tgt = ts[TARGET], np.asarray(streams[TARGET][1])
    tgt, n_fixed = repair_target_quats(tgt_t, tgt, ts[TCP], np.asarray(streams[TCP][1]))
    speed = np.linalg.norm(np.diff(tgt[:, :3], axis=0), axis=1) / np.maximum(np.diff(tgt_t), 1e-4)
    moving = np.flatnonzero(speed > still_speed)
    t0 = max(ts[IMG][0], ts[TCP][0], tgt_t[moving[0]] - 0.5 if len(moving) else tgt_t[0])
    t1 = min(ts[IMG][-1], ts[TCP][-1], tgt_t[moving[-1] + 1] + 1.0 if len(moving) else tgt_t[-1])
    grid = np.arange(t0, t1, 1.0 / hz)

    # Nearest camera frame at or before each grid time (never a future frame).
    idx = np.clip(np.searchsorted(ts[IMG], grid, side="right") - 1, 0, len(ts[IMG]) - 1)
    imgs = np.empty((len(grid), size[0], size[1], 3), np.uint8)
    for j, i in enumerate(idx):
        bgr = cv2.imdecode(np.frombuffer(streams[IMG][1][i], np.uint8), cv2.IMREAD_COLOR)
        imgs[j] = cv2.resize(bgr, (size[1], size[0]), interpolation=cv2.INTER_AREA)[:, :, ::-1]

    driver_pose = _interp_poses(ts[TCP], np.asarray(streams[TCP][1]), grid).astype(np.float32)
    if pose_source == "nominal_fk":
        # Joint angles at each grid time (linear; 500 Hz source), then nominal kinematics.
        jt, jq = ts[JOINTS], np.asarray(streams[JOINTS][1], np.float64)
        g = np.clip(grid, jt[0], jt[-1])
        q = np.stack([np.interp(g, jt, jq[:, i]) for i in range(6)], 1)
        tcp_pose = mat_to_pose7(ur5e_fk(q)).astype(np.float32)
    else:
        tcp_pose = driver_pose
    out = {
        "images": imgs,
        "tcp_pose": tcp_pose,
        "driver_tcp_pose": driver_pose,
        "target_pose": _interp_poses(tgt_t, tgt, grid).astype(np.float32),
        "gripper_cmd": _hold(ts[GRIP_CMD], np.asarray(streams[GRIP_CMD][1]), grid, OPEN_M),
        "gripper_width": _hold(ts[GRIP_POS], np.asarray(streams[GRIP_POS][1]), grid, OPEN_M),
        "wrench": wrench_at(ts[WRENCH], np.asarray(streams[WRENCH][1], np.float64).reshape(-1, 6), grid),
    }
    stats = {"bag": path.name, "frames": len(grid), "seconds": round(t1 - t0, 2),
             "trimmed_head_s": round(t0 - ts[IMG][0], 2), "trimmed_tail_s": round(ts[IMG][-1] - t1, 2),
             "camera_frame_gap_max_s": round(float(np.max(np.diff(ts[IMG]))), 3),
             "target_quat_signs_fixed": n_fixed, "wrench_samples": int(len(ts[WRENCH]))}
    if pose_source == "nominal_fk":   # how far the two kinematics are apart in this take
        dp = np.linalg.norm(tcp_pose[:, :3] - driver_pose[:, :3], axis=1) * 1000
        stats["nominal_vs_driver_mm_median"] = round(float(np.median(dp)), 2)
    return out, info, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=pathlib.Path, help="Directory of bag directories, e.g. /data/teleop_demos/clear-piece")
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--skip-first", type=int, default=0, help="Drop the earliest N bags (sorted by name = time)")
    ap.add_argument("--exclude", nargs="*", default=[], help="Bag names to drop")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--still-speed", type=float, default=0.005, help="m/s below which the arm counts as idle")
    ap.add_argument("--pose-source", choices=("nominal_fk", "driver"), default="nominal_fk",
                    help="Measured tool0 pose for tcp_pose.npy: nominal kinematics of /joint_states (default; the "
                         "model teleop commanded with and the robot runs on) or the driver's calibrated tcp_pose")
    a = ap.parse_args()

    bags = sorted(p for p in a.src.iterdir() if (p / "metadata.yaml").exists())[a.skip_first:]
    bags = [b for b in bags if b.name not in a.exclude]
    a.out.mkdir(parents=True, exist_ok=True)
    keys = ["images", "tcp_pose", "driver_tcp_pose", "target_pose", "gripper_cmd", "gripper_width", "wrench"]
    parts, ends, stats, info = {k: [] for k in keys}, [], [], None
    for b in bags:
        try:
            ep, ep_info, st = read_bag(b, a.hz, (a.height, a.width), a.still_speed, a.pose_source)
        except Exception as e:  # one broken take shouldn't sink the dataset; it's listed in meta.json
            print(f"SKIP {b.name}: {e}")
            stats.append({"bag": b.name, "error": str(e)})
            continue
        info = info or ep_info
        for k in keys:
            parts[k].append(ep[k])
        ends.append((ends[-1] if ends else 0) + st["frames"])
        stats.append(st)
        print(f"{b.name}: {st['frames']} frames, {st['seconds']} s")

    for k in keys:
        np.save(a.out / f"{k}.npy", np.concatenate(parts[k]))
    np.save(a.out / "episode_ends.npy", np.asarray(ends, np.int64))
    if info:  # intrinsics scaled to the stored image size
        sx, sy = a.width / info["width"], a.height / info["height"]
        K = np.asarray(info["K"]).reshape(3, 3) * [[sx], [sy], [1]]
        info = {**info, "K_stored": K.tolist()}
    meta = {"source": str(a.src), "hz": a.hz, "image_hw": [a.height, a.width], "camera_info": info,
            "pose_frame": "tool0 in base, xyz + quat wxyz", "pose_source": a.pose_source, "episodes": stats}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(ends)} episodes, {ends[-1] if ends else 0} frames -> {a.out}")


if __name__ == "__main__":
    main()
