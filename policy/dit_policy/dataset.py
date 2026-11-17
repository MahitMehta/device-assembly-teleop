"""Training samples from a converted dataset (convert_rosbags.py, or the sim's collect_demos.py).

One sample = the last `obs_horizon` wrist images up to step t, optionally the robot state at those
frames (config `state`: gripper width, tool orientation, wrist wrench; see geometry.state_vector), and the next
`action_horizon` actions from t. Indices past an episode's ends are clamped, i.e. the first
frame is repeated before the start and the last action is held after the end.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import OPEN_M, pose7_to_mat, poses_to_action, state_vector


def load_arrays(root: str | pathlib.Path, mmap: bool = True) -> dict:
    root = pathlib.Path(root)
    mode = "r" if mmap else None
    d = {k: np.load(root / f"{k}.npy", mmap_mode=mode) for k in
         ("images", "tcp_pose", "target_pose", "gripper_cmd", "episode_ends")}
    for k in ("gripper_width", "wrench"):
        if (root / f"{k}.npy").exists():
            d[k] = np.load(root / f"{k}.npy", mmap_mode=mode)
    d["meta"] = json.loads((root / "meta.json").read_text())
    return d


class Normalizer:
    """Maps each action dim to [-1, 1]. Positions use dataset min/max (with a margin); rot6d is
    already in [-1, 1]; the gripper uses its physical range."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray):
        self.lo, self.hi = np.asarray(lo, np.float32), np.asarray(hi, np.float32)

    @classmethod
    def fit(cls, actions: np.ndarray, margin: float = 0.05) -> "Normalizer":
        lo, hi = actions[:, :3].min(0), actions[:, :3].max(0)
        pad = (hi - lo) * margin + 1e-4
        lo3, hi3 = lo - pad, hi + pad
        return cls(np.r_[lo3, -np.ones(6), 0.0], np.r_[hi3, np.ones(6), OPEN_M])

    def normalize(self, a):
        lo, hi = (torch.as_tensor(self.lo, device=a.device), torch.as_tensor(self.hi, device=a.device)) \
            if isinstance(a, torch.Tensor) else (self.lo, self.hi)
        return (a - lo) / (hi - lo) * 2 - 1

    def unnormalize(self, a):
        lo, hi = (torch.as_tensor(self.lo, device=a.device), torch.as_tensor(self.hi, device=a.device)) \
            if isinstance(a, torch.Tensor) else (self.lo, self.hi)
        return (a + 1) / 2 * (hi - lo) + lo

    def state_dict(self) -> dict:
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}

    @classmethod
    def from_state_dict(cls, s: dict) -> "Normalizer":
        return cls(s["lo"], s["hi"])


class DemoDataset(Dataset):
    def __init__(self, root: str, obs_horizon: int, action_horizon: int, action_space: str,
                 episodes: list[int] | None = None, crop_hw: tuple[int, int] | None = None,
                 train: bool = True, in_memory: bool = True, state_keys=(), mount_rotation: dict | None = None):
        d = load_arrays(root, mmap=not in_memory)
        self.images = np.ascontiguousarray(d["images"]) if in_memory else d["images"]
        ends = d["episode_ends"]
        starts = np.r_[0, ends[:-1]]
        episodes = list(range(len(ends))) if episodes is None else episodes
        self.To, self.H, self.space = obs_horizon, action_horizon, action_space
        self.crop_hw, self.train = crop_hw, train
        # Wrist-mount turn augmentation (training only; see _rotate). None or deg 0 = off.
        mr = mount_rotation or {}
        self.rot_deg = float(mr.get("deg", 0.0)) if train else 0.0
        self.rot_pivot = np.asarray(mr.get("pivot_xy", (0.0, 0.0)), np.float64)
        self.rot_jitter = float(mr.get("pivot_jitter_px", 0.0))

        # Precompute every sample's action chunk once: poses are tiny next to images.
        tcp, tgt = pose7_to_mat(np.asarray(d["tcp_pose"])), pose7_to_mat(np.asarray(d["target_pose"]))
        grip = np.asarray(d["gripper_cmd"], np.float32)
        self.state_keys = tuple(state_keys)
        self.state = None
        if self.state_keys:
            for k in ("gripper_width", "wrench"):
                if k in self.state_keys and k not in d:
                    raise FileNotFoundError(f"{root}/{k}.npy needed for the {k} state input (re-run convert_rosbags.py)")
            width = np.asarray(d.get("gripper_width", np.zeros(len(tcp))), np.float32)
            wrench = np.asarray(d.get("wrench", np.zeros((len(tcp), 6))), np.float32)
            self.state = np.stack([state_vector(tcp[i], width[i], self.state_keys, wrench[i]) for i in range(len(tcp))])
        self.index, acts = [], []
        for e in episodes:
            s, t_end = int(starts[e]), int(ends[e])
            for t in range(s, t_end):
                fut = np.minimum(np.arange(t, t + self.H), t_end - 1)
                acts.append(poses_to_action(tcp[t], tgt[fut], grip[fut], action_space))
                self.index.append((t, s))
        self.actions = np.stack(acts)  # (N, H, 10)
        self.episodes = episodes
        self.ep_len = np.array([int(ends[e] - starts[e]) for e in episodes])
        self.sample_ep = np.repeat(np.arange(len(episodes)), self.ep_len)  # position in `episodes`

    def sample_weights(self, power: float) -> np.ndarray:
        """Per-sample weights that favour shorter (faster) demos: each episode's total share of the
        batches is proportional to len**-power, split evenly over its frames. power=0 gives every
        episode the same share; plain uniform sampling over frames corresponds to power=-1."""
        share = self.ep_len.astype(np.float64) ** -power
        share /= share.sum()
        return (share / self.ep_len)[self.sample_ep]

    def __len__(self) -> int:
        return len(self.index)

    def _rotate(self, imgs: np.ndarray) -> np.ndarray:
        """Simulates the camera mount clamped a few degrees round the wrist. The clamp can only turn
        about the wrist (tool z) axis; for this camera (~15 deg off that axis, ~7 cm from it) a turn
        of a degrees rotates the image by ~a degrees about the pixel where the tool axis meets the
        working surface: ~(165, 110) in the stored 256x192 image, within 1-2 px over the image.
        That pixel moves with distance and the exact tilt, hence the jitter. Same angle and pivot
        for every frame of the sample (the mount doesn't move within an episode); the action label
        is unchanged, since turning the camera moves nothing in the world."""
        if not self.rot_deg:
            return imgs
        import cv2
        ang = np.random.uniform(-self.rot_deg, self.rot_deg)
        piv = self.rot_pivot + np.random.uniform(-self.rot_jitter, self.rot_jitter, 2)
        M = cv2.getRotationMatrix2D((float(piv[0]), float(piv[1])), ang, 1.0)
        h, w = imgs.shape[1:3]
        return np.stack([cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                         for im in imgs])

    def _crop(self, imgs: np.ndarray) -> np.ndarray:
        if self.crop_hw is None:
            return imgs
        h, w = imgs.shape[1:3]
        ch, cw = self.crop_hw
        if self.train:
            y, x = np.random.randint(0, h - ch + 1), np.random.randint(0, w - cw + 1)
        else:
            y, x = (h - ch) // 2, (w - cw) // 2
        return imgs[:, y:y + ch, x:x + cw]

    def __getitem__(self, i: int) -> dict:
        t, s = self.index[i]
        obs_idx = np.maximum(np.arange(t - self.To + 1, t + 1), s)
        imgs = self._crop(self._rotate(np.asarray(self.images[obs_idx])))  # (To, h, w, 3) uint8
        out = {"images": torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2),  # (To,3,h,w) uint8
               "actions": torch.from_numpy(self.actions[i])}
        if self.state is not None:
            out["state"] = torch.from_numpy(self.state[obs_idx])  # (To, Ds), same frames as the images
        return out
