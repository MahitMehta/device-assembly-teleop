"""Inference wrapper: checkpoint in, tool0 target poses out. Used by the sim evaluator (and, later,
a robot node). Keeps the last `obs_horizon` frames itself.

    pol = Policy.load("runs/x/final.pt")
    pol.reset()
    targets, grip = pol.act(image_hwc_uint8, current_tool0_4x4, gripper_width_m)   # (n_exec, 4, 4), (n_exec,)

The width is used only by policies trained with state inputs (config `state`); image-only
checkpoints ignore it.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .dataset import Normalizer
from .geometry import action_to_poses, state_vector
from .model import DiTPolicy
from .train import make_schedulers, sample_actions


class Policy:
    def __init__(self, ckpt: dict, device: str | None = None, use_ema: bool = True):
        self.cfg = ckpt["cfg"]
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = DiTPolicy({**self.cfg, "model": {**self.cfg["model"], "pretrained_encoder": False}})
        self.model.load_state_dict(ckpt["ema" if use_ema else "model"])
        self.model.to(self.dev).eval()
        self.norm = Normalizer.from_state_dict(ckpt["normalizer"])
        _, self.ddim = make_schedulers(self.cfg)
        self.To, self.space = self.cfg["obs_horizon"], self.cfg["action_space"]
        self.n_exec = self.cfg["exec_horizon"]
        self.crop = self.cfg["crop_hw"]
        self.state_keys = tuple(self.cfg.get("state") or ())
        self.frames: deque = deque(maxlen=self.To)
        self.states: deque = deque(maxlen=self.To)

    @classmethod
    def load(cls, path: str, **kw) -> "Policy":
        return cls(torch.load(path, map_location="cpu", weights_only=False), **kw)

    def reset(self) -> None:
        self.frames.clear()
        self.states.clear()

    def _center_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]; ch, cw = self.crop
        y, x = (h - ch) // 2, (w - cw) // 2
        return img[y:y + ch, x:x + cw]

    def observe(self, image: np.ndarray, tool0: np.ndarray | None = None, gripper_width: float | None = None,
                wrench: np.ndarray | None = None) -> None:
        """image: (H, W, 3) uint8 RGB at the dataset's stored size; tool0 (4,4) the measured pose in
        base, gripper_width the measured jaw gap (m) and wrench (6,) the filtered, zeroed wrist
        force/torque (geometry.WRENCH_FILTER) when the image was taken (state policies only)."""
        self.frames.append(self._center_crop(image))
        if self.state_keys:
            if tool0 is None or gripper_width is None:
                raise ValueError(f"this policy needs the robot state ({', '.join(self.state_keys)}) with each image")
            self.states.append(state_vector(tool0, gripper_width, self.state_keys, wrench))
        while len(self.frames) < self.To:  # first call: repeat, as training clamps at episode start
            self.frames.appendleft(self.frames[0])
            if self.state_keys:
                self.states.appendleft(self.states[0])

    @torch.no_grad()
    def plan(self) -> np.ndarray:
        imgs = torch.from_numpy(np.stack(self.frames)).permute(0, 3, 1, 2)[None].to(self.dev)
        st = torch.from_numpy(np.stack(self.states))[None].to(self.dev) if self.state_keys else None
        with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            a = sample_actions(self.model, self.ddim, imgs, self.cfg["diffusion"]["inference_steps"], state=st).float()
        return self.norm.unnormalize(a)[0].cpu().numpy()  # (H, 10)

    def act(self, image: np.ndarray, tool0: np.ndarray, gripper_width: float | None = None,
            wrench: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.observe(image, tool0, gripper_width, wrench)
        chunk = self.plan()[: self.n_exec]
        return action_to_poses(tool0, chunk, self.space)
