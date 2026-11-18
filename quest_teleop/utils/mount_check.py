"""Is the wrist camera mounted as it was for the demos? Measured from the gripper's left finger.

The camera clamps round the wrist, so a re-mount can leave it turned a few degrees about the
wrist axis. That turns the whole image by about the same angle, fingers included (they are fixed
to the tool, not the camera), so the turn of the open left finger in the image measures the mount
turn directly.

How: a template of the finger (its pixels and a thin ring of whatever is round it, from the median
of the demos' first open-gripper frames) is turned and shifted over the live image, and the angle
where it fits best (masked zero-mean normalised correlation) is the mount turn. Normalising makes
it indifferent to brightness and contrast, and the finger's outline dominates the fit, so it does not
depend on what the arm is over. (The first version thresholded dark pixels and took the blob's
angle: over a darker mat or with different lighting the blob changed shape and read ~3 deg off.)

Reference: config/wrist_mount_reference.json and config/wrist_mount_finger.npz, made from a dataset by
    python3 -m quest_teleop.utils.mount_check <dataset_dir> [--write]
Images are the stored size the policy sees (256x192, resized from 640x480 with INTER_AREA). Only
building the reference needs OpenCV; checking is numpy only.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

REFERENCE_PATH = pathlib.Path(__file__).resolve().parents[2] / "config" / "wrist_mount_reference.json"
TEMPLATE_PATH = REFERENCE_PATH.with_name("wrist_mount_finger.npz")
ROW0 = 120          # the finger is in the bottom 72 rows
DARK = 45           # grey level: the finger is black, the mat ~100 (only for finding it in the reference)
MIN_AREA, MAX_AREA = 0.6, 1.6   # x the reference blob area
RING = 3            # px of surroundings kept round the finger in the template
MAX_DEG = 8.0       # turns searched, +/- deg
SEARCH_PX = 12      # shifts searched, +/- px (a turn about the wrist axis also shifts the finger a little)
MIN_SCORE = 0.6     # a worse best fit means the finger isn't there to see (closed, covered, dark)
MAX_Y = 177         # template rows above this only: the finger runs off the bottom of the image, and a
                    # template reaching the edge loses part of itself when the finger sits lower (as live,
                    # 7 px), which let a turn trade for a slide (3 deg errors)


@dataclass
class FingerPose:
    x: float        # centroid, px
    y: float
    angle: float    # blob orientation, deg (image coords)
    area: int       # px


@dataclass
class FingerMatch:
    angle: float    # deg the image is turned from the reference, + = counter-clockwise (OpenCV's positive)
    dx: float       # px the finger moved
    dy: float
    score: float    # fit, -1..1


# ---- the reference: finding the finger by thresholding, in the demos' own frames
def _finger_blob(g: np.ndarray, near=None, ref_area=None):
    """g: grey image (full size). The dark blob below ROW0 nearest `near`: (FingerPose, mask) or None."""
    import cv2
    sub = g[ROW0:]
    n, lab, st, cen = cv2.connectedComponentsWithStats((sub < DARK).astype(np.uint8))
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
    blob = (lab == i).astype(np.uint8)
    mu = cv2.moments(blob, binaryImage=True)
    angle = 0.5 * np.degrees(np.arctan2(2 * mu["mu11"], mu["mu20"] - mu["mu02"]))
    mask = np.zeros(g.shape, bool)
    mask[ROW0:] = blob > 0
    return FingerPose(float(cen[i][0]), float(cen[i][1] + ROW0), float(angle), area), mask


def finger_pose(frames_rgb, near=None, ref_area=None) -> FingerPose | None:
    """Threshold-and-blob measure (the first version), used to build the reference."""
    g = np.min([np.asarray(f, np.float32).mean(-1) for f in frames_rgb], axis=0)
    found = _finger_blob(g, near, ref_area)
    return None if found is None else found[0]


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            src = mask[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
            out[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)] |= src
    return out


def make_template(grey: np.ndarray, finger: np.ndarray, ring: int = RING, max_y: int = MAX_Y) -> dict:
    """The finger's pixels and a `ring` px band round it, from a grey reference image, above row max_y."""
    keep = _dilate(finger, ring)
    keep[max_y + 1:] = False
    finger = finger.copy()
    finger[max_y + 1:] = False
    ys, xs = np.nonzero(keep)
    fy, fx = np.nonzero(finger)
    return {"points": np.c_[xs, ys].astype(np.float64), "values": grey[ys, xs].astype(np.float64),
            "center": np.array([fx.mean(), fy.mean()])}


# ---- measuring: the template turned and shifted over the live image
def grey_of(frames_rgb) -> np.ndarray:
    """Median grey of a few frames with the arm still (less noise than one)."""
    return np.median([np.asarray(f, np.float64).mean(-1) for f in frames_rgb], axis=0)


def _sample(g: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear samples of g at (x, y); NaN outside the image."""
    h, w = g.shape
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    inside = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0, y0 = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    v = (g[y0, x0] * (1 - fx) * (1 - fy) + g[y0, x0 + 1] * fx * (1 - fy)
         + g[y0 + 1, x0] * (1 - fx) * fy + g[y0 + 1, x0 + 1] * fx * fy)
    return np.where(inside, v, np.nan)


def _zncc(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Zero-mean normalised correlation of template values t (n,) with each row of v (k, n),
    over the samples inside the image; -1 when under 80% of them are."""
    ok = ~np.isnan(v)
    w = ok.astype(np.float64)
    v = np.where(ok, v, 0.0)
    n = w.sum(1)
    tc = (t[None] - (w * t).sum(1, keepdims=True) / np.maximum(n, 1)[:, None]) * w
    vc = (v - v.sum(1, keepdims=True) / np.maximum(n, 1)[:, None]) * w
    s = (tc * vc).sum(1) / np.sqrt((tc ** 2).sum(1) * (vc ** 2).sum(1) + 1e-9)
    return np.where(n >= 0.8 * len(t), s, -1.0)


def _scores(g, tpl, angles, shifts):
    """Best fit over `shifts` (k, 2) for each angle: (scores (a,), best shift index (a,))."""
    p, c, t = tpl["points"], tpl["center"], tpl["values"]
    best, at = np.empty(len(angles)), np.empty(len(angles), int)
    for i, a in enumerate(np.radians(angles)):
        # where a reference pixel lands when the image turns by a about c (cv2.getRotationMatrix2D)
        ca, sa = np.cos(a), np.sin(a)
        d = p - c
        q = c + np.c_[ca * d[:, 0] + sa * d[:, 1], -sa * d[:, 0] + ca * d[:, 1]]
        v = _sample(g, q[None, :, 0] + shifts[:, :1], q[None, :, 1] + shifts[:, 1:])
        s = _zncc(t, v)
        at[i] = int(np.argmax(s))
        best[i] = s[at[i]]
    return best, at


def match(grey: np.ndarray, tpl: dict, max_deg: float = MAX_DEG, search_px: int = SEARCH_PX) -> FingerMatch:
    """Coarse search (1 deg, 1 px), then fine (0.1 deg, 0.25 px) round the best, then a parabola."""
    r = np.arange(-search_px, search_px + 1, dtype=np.float64)
    shifts = np.stack(np.meshgrid(r, r), -1).reshape(-1, 2)
    coarse = np.arange(-max_deg, max_deg + 1e-9, 1.0)
    s, at = _scores(grey, tpl, coarse, shifts)
    i = int(np.argmax(s))
    f = np.arange(-1.0, 1.0 + 1e-9, 0.25)
    fine_shifts = shifts[at[i]] + np.stack(np.meshgrid(f, f), -1).reshape(-1, 2)
    fine = coarse[i] + np.arange(-1.0, 1.0 + 1e-9, 0.1)
    s, at = _scores(grey, tpl, fine, fine_shifts)
    j = int(np.argmax(s))
    angle = fine[j]
    if 0 < j < len(fine) - 1:
        den = s[j - 1] - 2 * s[j] + s[j + 1]
        if den < 0:
            angle += 0.1 * 0.5 * (s[j - 1] - s[j + 1]) / den
    dx, dy = fine_shifts[at[j]]
    return FingerMatch(float(angle), float(dx), float(dy), float(s[j]))


# ---- reference files and the check
def load_reference(path: str | pathlib.Path = REFERENCE_PATH) -> dict | None:
    p = pathlib.Path(path)
    if not p.exists():
        return None
    ref = json.loads(p.read_text())
    tp = p.with_name(ref.get("template", TEMPLATE_PATH.name))
    if tp.exists():
        with np.load(tp) as z:
            ref["tpl"] = {k: z[k] for k in ("points", "values", "center")}
    return ref


def mount_offset_deg(pose: FingerPose, ref: dict) -> float:
    """Threshold version: how far the mount is turned from the demos, deg. Positive = the image is
    turned counter-clockwise (OpenCV's positive rotation) relative to the demos."""
    d = ref["finger_angle_deg"] - pose.angle   # blob angles are measured with y down: CCW lowers them
    return float((d + 90.0) % 180.0 - 90.0)


def check(frames_rgb, ref: dict, tolerance_deg: float) -> tuple[bool | None, str]:
    """(within tolerance, message); None when the finger can't be seen."""
    if "tpl" not in ref:
        return None, f"camera mount: no finger template ({TEMPLATE_PATH.name}); rebuild the reference with --write"
    m = match(grey_of(frames_rgb), ref["tpl"])
    if m.score < MIN_SCORE:
        return None, (f"camera mount: couldn't see the left finger (best fit {m.score:.2f}; gripper open, "
                      f"nothing covering it?)")
    ok = abs(m.angle) <= tolerance_deg
    return ok, (f"camera mount turned {m.angle:+.1f} deg from the demos (policy tolerates +/-{tolerance_deg:g}); "
                f"finger moved ({m.dx:+.1f}, {m.dy:+.1f}) px, fit {m.score:.2f}")


def main() -> None:
    """Build the reference from a dataset: the first open-gripper frames of every episode."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", type=pathlib.Path, help="convert_rosbags.py output (images.npy, gripper_width.npy, episode_ends.npy)")
    ap.add_argument("--write", action="store_true", help=f"write {REFERENCE_PATH.name} and {TEMPLATE_PATH.name}")
    a = ap.parse_args()
    img = np.load(a.dataset / "images.npy", mmap_mode="r")
    width = np.load(a.dataset / "gripper_width.npy")
    ends = np.load(a.dataset / "episode_ends.npy")
    greys = []
    for s, e in zip(np.r_[0, ends[:-1]], ends):
        idx = np.where(width[s:e] > 0.08)[0][:20:4] + s
        if len(idx):
            greys.append(grey_of([img[i] for i in idx]))
    g = np.median(greys, axis=0)   # the finger is the same in all; what's round it averages out
    found = _finger_blob(g)
    if found is None:
        raise SystemExit("no finger found in the median open-gripper image")
    pose, mask = found
    tpl = make_template(g, mask)
    per_episode = np.array([match(x, tpl).angle for x in greys])
    ref = {"finger_xy": [round(pose.x, 2), round(pose.y, 2)], "finger_angle_deg": round(pose.angle, 2),
           "finger_area": int(pose.area), "template": TEMPLATE_PATH.name, "template_pixels": int(len(tpl["values"])),
           "spread_deg": round(float(per_episode.std()), 3), "episodes": len(greys), "source": str(a.dataset)}
    print(json.dumps(ref, indent=2))
    print("per-episode turn (deg):", np.round(per_episode, 2).tolist())
    if a.write:
        REFERENCE_PATH.write_text(json.dumps(ref, indent=2) + "\n")
        np.savez_compressed(TEMPLATE_PATH, **tpl)
        print(f"wrote {REFERENCE_PATH} and {TEMPLATE_PATH}")


if __name__ == "__main__":
    main()
