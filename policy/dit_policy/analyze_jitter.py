"""Measures shake in the teleop demos from the raw bags (not the 10 Hz dataset, which can't show
anything above 5 Hz). Read-only; prints one JSON line per bag and a summary.

    python -m dit_policy.analyze_jitter /data/teleop_demos/clear-piece

Per bag, for the commanded (`/teleop/target_pose`) and measured (`/tcp_pose_broadcaster/pose`) tool0:
  rate_hz             native message rate
  jit_mm@fc           RMS of (signal - zero-phase low-pass at fc Hz), position, while moving
  jit_deg@fc          same for rotation (angle of the residual rotation)
  band_frac           share of velocity power in 0-1 / 1-3 / 3-6 / 6+ Hz (Welch)
split by phase: `approach` = until the gripper is first commanded closed, `rest` = after.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.signal import butter, sosfiltfilt, welch
from scipy.spatial.transform import Rotation

from .convert_rosbags import repair_target_quats

TCP = "/tcp_pose_broadcaster/pose"
TARGET = "/teleop/target_pose"
GRIP_CMD = "/teleop/gripper_command"
FS = 100.0                # analysis rate (uniform resample)
CUTOFFS = (1.0, 2.0, 3.0)
BANDS = ((0, 1), (1, 3), (3, 6), (6, 50))


def read(path: pathlib.Path, repair: bool = True):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    s = {k: ([], []) for k in (TCP, TARGET, GRIP_CMD)}
    with AnyReader([path], default_typestore=ts) as r:
        conns = [c for c in r.connections if c.topic in s]
        for c, t, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            if c.topic == GRIP_CMD:
                v = float(m.data)
            else:
                p, q = m.pose.position, m.pose.orientation
                v = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
            s[c.topic][0].append(t * 1e-9); s[c.topic][1].append(v)
    s = {k: (np.asarray(a), np.asarray(b)) for k, (a, b) in s.items()}
    if repair and len(s[TARGET][0]) and len(s[TCP][0]):  # same fix as the converter (it uses wxyz)
        wxyz = lambda x: x[:, [0, 1, 2, 6, 3, 4, 5]]
        fixed, _ = repair_target_quats(s[TARGET][0], wxyz(s[TARGET][1]), s[TCP][0], wxyz(s[TCP][1]))
        s[TARGET] = (s[TARGET][0], fixed[:, [0, 1, 2, 4, 5, 6, 3]])
    return s


def uniform(t, x, grid):
    """Linear resample; rotations via unwrapped rotation vectors relative to the first sample."""
    t, keep = np.unique(t, return_index=True); x = x[keep]
    xyz = np.stack([np.interp(grid, t, x[:, i]) for i in range(3)], 1)
    R = Rotation.from_quat(x[:, 3:])
    rv = (R[0].inv() * R).as_rotvec()
    rv = np.stack([np.interp(grid, t, rv[:, i]) for i in range(3)], 1)
    return xyz, R[0] * Rotation.from_rotvec(rv)


def metrics(xyz, rot, moving):
    out = {}
    rv = (rot[0].inv() * rot).as_rotvec()
    for fc in CUTOFFS:
        sos = butter(2, fc, fs=FS, output="sos")
        lp = sosfiltfilt(sos, xyz, axis=0)
        res = np.linalg.norm(xyz - lp, axis=1)[moving]
        out[f"jit_mm@{fc:g}"] = round(float(np.sqrt(np.mean(res ** 2))) * 1000, 2) if len(res) else None
        lpr = rot[0] * Rotation.from_rotvec(sosfiltfilt(sos, rv, axis=0))
        ang = np.degrees((lpr.inv() * rot).magnitude())[moving]
        out[f"jit_deg@{fc:g}"] = round(float(np.sqrt(np.mean(ang ** 2))), 3) if len(ang) else None
    v = np.gradient(xyz, 1 / FS, axis=0)
    if moving.sum() > 256:
        f, p = welch(v[moving], fs=FS, nperseg=256, axis=0); p = p.sum(1)
        tot = p.sum()
        out["band_frac"] = [round(float(p[(f >= a) & (f < b)].sum() / tot), 3) for a, b in BANDS]
    return out


def analyze(path: pathlib.Path, still: float, repair: bool = True):
    s = read(path, repair)
    tt, tx = s[TARGET]
    if len(tt) < 10 or len(s[TCP][0]) < 10:
        return None
    t0, t1 = max(tt[0], s[TCP][0][0]), min(tt[-1], s[TCP][0][-1])
    grid = np.arange(t0, t1, 1 / FS)
    gt, gv = s[GRIP_CMD]
    t_close = t1
    if len(gv):
        ch = np.flatnonzero(np.abs(gv - gv[0]) > 0.2 * (np.ptp(gv) or 1))
        if len(ch): t_close = gt[ch[0]]
    rec = {"bag": path.name, "dur_s": round(t1 - t0, 1), "t_close_s": round(t_close - t0, 1)}
    for name, topic in (("target", TARGET), ("tcp", TCP)):
        t, x = s[topic]
        rec[f"{name}_rate_hz"] = round(len(t) / (t[-1] - t[0]), 1)
        xyz, rot = uniform(t, x, grid)
        sp = np.linalg.norm(sosfiltfilt(butter(2, 1.0, fs=FS, output="sos"), np.gradient(xyz, 1 / FS, axis=0), axis=0), axis=1)
        mov = sp > still
        for ph, mask in (("approach", grid < t_close), ("rest", grid >= t_close)):
            rec[f"{name}_{ph}"] = metrics(xyz, rot, mov & mask)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--skip-first", type=int, default=0)
    ap.add_argument("--still-speed", type=float, default=0.005)
    ap.add_argument("--no-repair", action="store_true", help="analyze the raw (sign-glitched) target orientation")
    a = ap.parse_args()
    bags = sorted(p for p in pathlib.Path(a.root).iterdir() if (p / "metadata.yaml").exists())[a.skip_first:]  # like the converter: subfolders (failures/) ignored
    recs = []
    for b in bags:
        try:
            r = analyze(b, a.still_speed, not a.no_repair)
        except Exception as e:  # keep going; report the bag
            r = {"bag": b.name, "error": repr(e)}
        if r: print(json.dumps(r), flush=True); recs.append(r)
    ok = [r for r in recs if "error" not in r]
    summ = {}
    for key in ("target_approach", "target_rest", "tcp_approach", "tcp_rest"):
        for m in ("jit_mm@2", "jit_deg@2", "jit_mm@1", "jit_mm@3"):
            v = np.array([r[key][m] for r in ok if r[key].get(m) is not None])
            if len(v): summ[f"{key}.{m}"] = {"median": round(float(np.median(v)), 2), "p90": round(float(np.percentile(v, 90)), 2), "max": round(float(v.max()), 2)}
        bf = np.array([r[key]["band_frac"] for r in ok if "band_frac" in r[key]])
        if len(bf): summ[f"{key}.band_frac_median"] = np.round(np.median(bf, 0), 3).tolist()
    worst = sorted(ok, key=lambda r: -(r["target_approach"].get("jit_mm@2") or 0))[:6]
    summ["worst_approach_target@2"] = [(r["bag"], r["target_approach"].get("jit_mm@2")) for r in worst]
    print("SUMMARY " + json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
