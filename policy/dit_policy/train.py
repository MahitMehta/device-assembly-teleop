"""Trains the DiT policy (images, plus the robot state inputs listed in the config's `state`).

    python -m dit_policy.train --config configs/clear_piece_dit.yaml --data /path/to/dataset --out runs/x
    python -m dit_policy.train ... --set train.steps=2000 model.depth=4     # dotted overrides

Resumable: the run directory's latest.pt is picked up automatically, so a preempted or timed-out
Slurm job can simply be resubmitted (the sbatch script also requeues itself on SIGUSR1).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pathlib
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import DemoDataset, Normalizer
from .model import DiTPolicy


def set_dotted(cfg: dict, key: str, value: str) -> None:
    *path, last = key.split(".")
    for p in path:
        cfg = cfg[p]
    cfg[last] = yaml.safe_load(value)


def make_schedulers(cfg: dict):
    d = cfg["diffusion"]
    kw = dict(num_train_timesteps=d["train_steps"], beta_schedule=d["beta_schedule"],
              clip_sample=True, prediction_type=d["prediction_type"])
    return DDPMScheduler(**kw), DDIMScheduler(**kw, set_alpha_to_one=True, steps_offset=0)


@torch.no_grad()
def sample_actions(model: DiTPolicy, sched: DDIMScheduler, images: torch.Tensor, steps: int,
                   generator: torch.Generator | None = None, state: torch.Tensor | None = None) -> torch.Tensor:
    """images (B, To, 3, h, w) uint8, state (B, To, Ds) or None -> normalized action chunk (B, H, Da)."""
    tok, pooled = model.encode(images, state)
    x = torch.randn(images.shape[0], model.H, model.Da, device=images.device, generator=generator)
    sched.set_timesteps(steps)
    for k in sched.timesteps:
        kk = torch.full((x.shape[0],), int(k), device=x.device, dtype=torch.long)
        x = sched.step(model.denoiser(x, kk, tok, pooled), k, x).prev_sample
    return x


class ColorJitter:
    """Per-sample brightness/contrast/saturation jitter on GPU (uint8 in, uint8 out). The real
    camera's exposure and white balance drifted between sessions, so this matters."""

    def __init__(self, b: float, c: float, s: float):
        self.b, self.c, self.s = b, c, s

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        f = x.float() / 255.0
        r = lambda a: (1 + (torch.rand(B, 1, 1, 1, 1, device=x.device) * 2 - 1) * a)
        f = f * r(self.b)
        mean = f.mean(dim=(1, 2, 3, 4), keepdim=True)
        f = (f - mean) * r(self.c) + mean
        gray = (0.299 * f[:, :, 0:1] + 0.587 * f[:, :, 1:2] + 0.114 * f[:, :, 2:3])
        f = (f - gray) * r(self.s) + gray
        return (f.clamp(0, 1) * 255).to(torch.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="Dataset directory (convert_rosbags.py / collect_demos.py output)")
    ap.add_argument("--out", required=True, help="Run directory; resumed if it has latest.pt")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        set_dotted(cfg, *kv.split("=", 1))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tc = cfg["train"]
    torch.manual_seed(tc["seed"]); np.random.seed(tc["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    # Episode split: the last `val_episodes` (by recording time) are held out, or, with
    # `train_shortest: N`, train on the N shortest demos and validate on all the others.
    ends = np.load(pathlib.Path(a.data) / "episode_ends.npy")
    n_eps = len(ends)
    if tc.get("train_shortest"):
        lens = np.diff(np.r_[0, ends])
        order = np.argsort(lens, kind="stable")
        tr_eps = sorted(order[:tc["train_shortest"]].tolist())
        va_eps = sorted(order[tc["train_shortest"]:].tolist())
        print(f"train on the {len(tr_eps)} shortest demos ({lens[tr_eps].min()}-{lens[tr_eps].max()} frames): {tr_eps}")
    else:
        n_val = tc["val_episodes"]
        tr_eps, va_eps = list(range(n_eps - n_val)), list(range(n_eps - n_val, n_eps))
    n_val = len(va_eps)
    common = dict(obs_horizon=cfg["obs_horizon"], action_horizon=cfg["action_horizon"],
                  action_space=cfg["action_space"], crop_hw=tuple(cfg["crop_hw"]), state_keys=cfg.get("state") or ())
    ds = DemoDataset(a.data, episodes=tr_eps, train=True, mount_rotation=tc.get("mount_rotation"), **common)
    va = DemoDataset(a.data, episodes=va_eps, train=False, **common) if n_val else None
    norm = Normalizer.fit(ds.actions.reshape(-1, ds.actions.shape[-1]))
    print(f"train {len(ds)} samples / {len(tr_eps)} episodes, val {len(va) if va else 0} samples / {n_val} episodes")

    # Optionally favour shorter demos (see DemoDataset.sample_weights); null = uniform over frames.
    power = tc.get("episode_weight_power")
    sampler = None
    if power is not None:
        w = ds.sample_weights(float(power))
        sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=len(ds), replacement=True,
                                        generator=torch.Generator().manual_seed(tc["seed"]))
        share = np.bincount(ds.sample_ep, weights=w)
        print(f"episode weighting power={power}: shortest demo ({ds.ep_len.min()} frames) gets "
              f"{share[ds.ep_len.argmin()] / share[ds.ep_len.argmax()]:.2f}x the batches of the longest ({ds.ep_len.max()})")
    dl = DataLoader(ds, batch_size=tc["batch_size"], shuffle=sampler is None, sampler=sampler, num_workers=tc["num_workers"],
                    pin_memory=True, drop_last=True, persistent_workers=tc["num_workers"] > 0)
    model = DiTPolicy(cfg).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    enc_params = list(model.encoder.trunk.parameters())
    enc_ids = {id(p) for p in enc_params}
    other = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": other, "lr": tc["lr"]},
                             {"params": enc_params, "lr": tc["lr"] * tc["encoder_lr_scale"]}],
                            weight_decay=tc["weight_decay"], betas=(0.9, 0.95))
    lr_at = lambda s: min(1.0, (s + 1) / tc["warmup_steps"]) * 0.5 * (1 + math.cos(math.pi * min(s / tc["steps"], 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    ddpm, ddim = make_schedulers(cfg)
    jitter = ColorJitter(*tc["color_jitter"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {n_params / 1e6:.1f} M params on {dev}")

    step = 0
    ckpt_path = out / "latest.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); step = ck["step"]
        print(f"resumed from step {step}")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    def save(path: pathlib.Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "cfg": cfg, "normalizer": norm.state_dict()}, tmp)
        os.replace(tmp, path)  # atomic: a kill mid-save never leaves a broken latest.pt

    stop = {"flag": False}
    signal.signal(signal.SIGUSR1, lambda *_: stop.update(flag=True))  # Slurm's pre-timeout warning
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    @torch.no_grad()
    def validate() -> dict:
        if not va:
            return {}
        g = torch.Generator(device=dev).manual_seed(0)
        idx = np.linspace(0, len(va) - 1, min(len(va), tc["val_samples"])).astype(int)
        errs = []
        for i in range(0, len(idx), 64):
            b = [va[j] for j in idx[i:i + 64]]
            imgs = torch.stack([x["images"] for x in b]).to(dev)
            gt = torch.stack([x["actions"] for x in b]).to(dev)
            st = torch.stack([x["state"] for x in b]).to(dev) if "state" in b[0] else None
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                pred = sample_actions(ema, ddim, imgs, cfg["diffusion"]["inference_steps"], g, st).float()
            errs.append((norm.unnormalize(pred) - gt).cpu())
        e = torch.cat(errs)
        return {"val_pos_mm_first": float(e[:, 0, :3].norm(dim=-1).mean() * 1000),
                "val_pos_mm_chunk": float(e[:, :, :3].norm(dim=-1).mean() * 1000),
                "val_grip_mm": float(e[:, :, 9].abs().mean() * 1000)}

    model.train()
    t0, losses, log = time.time(), [], open(out / "log.jsonl", "a")
    while step < tc["steps"] and not stop["flag"]:
        for batch in dl:
            imgs = jitter(batch["images"].to(dev, non_blocking=True))
            act = norm.normalize(batch["actions"].to(dev, non_blocking=True))
            st = None
            if "state" in batch:   # noise, so the policy can't lean on exact state values
                st = batch["state"].to(dev, non_blocking=True)
                st = st + torch.randn_like(st) * tc.get("state_noise_std", 0.0)
            noise = torch.randn_like(act)
            k = torch.randint(0, ddpm.config.num_train_timesteps, (act.shape[0],), device=dev)
            noisy = ddpm.add_noise(act, noise, k)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                pred = model(imgs, noisy, k, st)
            target = noise if cfg["diffusion"]["prediction_type"] == "epsilon" else act
            loss = F.mse_loss(pred.float(), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step(); sched.step(); step += 1
            decay = min(tc["ema_decay"], (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            losses.append(loss.item())

            if step % tc["log_every"] == 0:
                rec = {"step": step, "loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0],
                       "steps_per_s": tc["log_every"] / (time.time() - t0)}
                losses, t0 = [], time.time()
                if step % tc["val_every"] == 0:
                    model.eval(); rec |= validate(); model.train()
                print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            if step % tc["save_every"] == 0:
                save(ckpt_path)
            if step % tc["keep_every"] == 0:
                save(out / f"step_{step:07d}.pt")
            if step >= tc["steps"] or stop["flag"]:
                break
    save(ckpt_path)
    if step >= tc["steps"]:
        save(out / "final.pt")
        print("done")
    else:
        print(f"stopped at step {step}; resubmit to resume")


if __name__ == "__main__":
    main()
