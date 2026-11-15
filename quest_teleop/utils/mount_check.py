"""Is the wrist camera mounted as it was for the demos? Measured from the gripper's left finger.

The camera clamps round the wrist, so a re-mount can leave it turned a few degrees about the
wrist axis. That turns the whole image by about the same angle, fingers included (they are fixed
to the tool, not the camera). In the demos the open left finger is a dark blob at the bottom of
the image whose position and angle varied by under 0.3 deg over 45 demos, so the change in its
angle measures the mount turn directly.

Reference: config/wrist_mount_reference.json, made from a dataset by
    python3 -m quest_teleop.utils.mount_check <dataset_dir> [--write]
It needs the gripper open (as at the start of a demo) over the dark mat. Images are the stored
size the policy sees (256x192, resized from 640x480 with INTER_AREA).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

REFERENCE_PATH = pathlib.Path(__file__).resolve().parents[2] / "config" / "wrist_mount_reference.json"
ROW0 = 120          # the finger is in the bottom 72 rows
DARK = 45           # grey level: the finger is black, the mat ~100
MIN_AREA, MAX_AREA = 0.6, 1.6   # x the reference blob area


@dataclass
class FingerPose:
    x: float        # centroid, px
    y: float
    angle: float    # blob orientation, deg (image coords)
    area: int       # px


def finger_pose(frames_rgb: list[np.ndarray] | np.ndarray, near: tuple[float, float] | None = None,
                ref_area: float | None = None) -> FingerPose | None:
    """frames: a few consecutive (192, 256, 3) uint8 RGB frames with the arm still; the per-pixel
    minimum keeps the finger and suppresses mat texture. Picks the dark blob nearest `near`
    (the reference centroid) and returns None if there is none of a plausible size."""
    import cv2
    g = np.min([np.asarray(f, np.float32).mean(-1) for f in frames_rgb], axis=0)[ROW0:]
    n, lab, st, cen = cv2.connectedComponentsWithStats((g < DARK).astype(np.uint8))
    best = None
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        if ref_area and not (MIN_AREA * ref_area <= area <= MAX_AREA * ref_area):
            continue
        if not ref_area and area < 200:
            continue
        d = 0.0 if near is None else float(np.hypot(cen[i][0] - near[0], cen[i][1] + ROW0 - near[1]))
        if best is None or d < best[0]:
            best = (d, i, area)
    if best is None or (near is not None and best[0] > 40):
        return None
    _, i, area = best
    mu = cv2.moments((lab == i).astype(np.uint8), binaryImage=True)
    angle = 0.5 * np.degrees(np.arctan2(2 * mu["mu11"], mu["mu20"] - mu["mu02"]))
    return FingerPose(float(cen[i][0]), float(cen[i][1] + ROW0), float(angle), area)


def load_reference(path: str | pathlib.Path = REFERENCE_PATH) -> dict | None:
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def mount_offset_deg(pose: FingerPose, ref: dict) -> float:
    """How far the mount is turned from the demos, deg. Positive = the image is turned
    counter-clockwise (OpenCV's positive rotation) relative to the demos."""
    d = ref["finger_angle_deg"] - pose.angle   # blob angles are measured with y down: CCW lowers them
    return float((d + 90.0) % 180.0 - 90.0)


def check(frames_rgb, ref: dict, tolerance_deg: float) -> tuple[bool | None, str]:
    """(within tolerance, message); None when the finger can't be found."""
    pose = finger_pose(frames_rgb, near=tuple(ref["finger_xy"]), ref_area=ref["finger_area"])
    if pose is None:
        return None, ("camera mount: couldn't find the left finger (gripper open, over the mat, "
                      "nothing dark next to it?)")
    off = mount_offset_deg(pose, ref)
    dx, dy = pose.x - ref["finger_xy"][0], pose.y - ref["finger_xy"][1]
    ok = abs(off) <= tolerance_deg
    msg = (f"camera mount turned {off:+.1f} deg from the demos (policy tolerates +/-{tolerance_deg:g}); "
           f"finger moved ({dx:+.1f}, {dy:+.1f}) px")
    return ok, msg


def main() -> None:
    """Measure the reference from a dataset: the first open-gripper frames of every episode."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", type=pathlib.Path, help="convert_rosbags.py output (images.npy, gripper_width.npy, episode_ends.npy)")
    ap.add_argument("--write", action="store_true", help=f"write {REFERENCE_PATH}")
    a = ap.parse_args()
    img = np.load(a.dataset / "images.npy", mmap_mode="r")
    width = np.load(a.dataset / "gripper_width.npy")
    ends = np.load(a.dataset / "episode_ends.npy")
    poses = []
    for s, e in zip(np.r_[0, ends[:-1]], ends):
        idx = np.where(width[s:e] > 0.08)[0][:20:4] + s
        p = finger_pose([img[i] for i in idx])
        if p is not None:
            poses.append(p)
    arr = np.array([[p.x, p.y, p.angle, p.area] for p in poses])
    ref = {"finger_xy": [round(float(np.median(arr[:, 0])), 2), round(float(np.median(arr[:, 1])), 2)],
           "finger_angle_deg": round(float(np.median(arr[:, 2])), 2),
           "finger_area": int(np.median(arr[:, 3])),
           "spread_deg": round(float(arr[:, 2].std()), 2),
           "episodes": len(poses), "source": str(a.dataset)}
    print(json.dumps(ref, indent=2))
    if a.write:
        REFERENCE_PATH.write_text(json.dumps(ref, indent=2) + "\n")
        print(f"wrote {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
