"""Image-conditioned Diffusion Transformer (DiT) policy.

    wrist images (To frames) --ResNet-18--> spatial tokens ------------------+
    robot state (To frames, optional) --MLP--> one token per frame ---------+
                                     \--> pooled feature --+                 | cross-attention
    diffusion step k --sinusoid-MLP-------------------------+--> adaLN-Zero  |
    noisy action chunk (H x 10) --linear + pos-emb--> [DiT block x L] <------+--> predicted noise

By default only images condition the policy. Config `state` adds proprioception per frame
(geometry.STATE_DIMS: gripper width, tilt, orientation; never position or joint angles, which
would let it memorise where the fixture was). The action chunk is predicted as DDPM noise
(epsilon) and sampled with DDIM.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .geometry import state_dim

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _replace_bn_with_gn(m: nn.Module, groups: int = 16) -> nn.Module:
    """BatchNorm statistics drift with small, highly correlated robot batches (Diffusion Policy, Chi
    et al. 2023); GroupNorm keeps the pretrained affine parameters and avoids that."""
    for name, child in m.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(groups, child.num_features)
            gn.weight.data.copy_(child.weight.data); gn.bias.data.copy_(child.bias.data)
            setattr(m, name, gn)
        else:
            _replace_bn_with_gn(child, groups)
    return m


class ImageEncoder(nn.Module):
    def __init__(self, d_model: int, obs_horizon: int, pretrained: bool = True, group_norm: bool = True,
                 token_hw: tuple[int, int] = (6, 8)):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        if group_norm:
            _replace_bn_with_gn(r)
        self.trunk = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.proj = nn.Conv2d(512, d_model, 1)
        self.pool_hw = token_hw  # adaptive pool so any crop size gives the same token grid
        self.pos = nn.Parameter(torch.zeros(1, 1, token_hw[0] * token_hw[1], d_model))
        self.frame = nn.Parameter(torch.zeros(1, obs_horizon, 1, d_model))
        nn.init.normal_(self.pos, std=0.02); nn.init.normal_(self.frame, std=0.02)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """images (B, To, 3, h, w) float in [0,1] -> tokens (B, To*P, d), pooled (B, d)."""
        B, To = images.shape[:2]
        x = (images.flatten(0, 1) - self.mean) / self.std
        f = F.adaptive_avg_pool2d(self.proj(self.trunk(x)), self.pool_hw)  # (B*To, d, th, tw)
        tok = f.flatten(2).transpose(1, 2).reshape(B, To, -1, f.shape[1]) + self.pos + self.frame
        tok = tok.flatten(1, 2)
        return tok, tok.mean(1)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Self-attention over the action chunk, cross-attention to image tokens, MLP. Each sub-layer
    is adaLN-Zero modulated by the conditioning vector c (Peebles & Xie 2023)."""

    def __init__(self, d: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.xattn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp_ratio)), nn.GELU(approximate="tanh"),
                                 nn.Dropout(dropout), nn.Linear(int(d * mlp_ratio), d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 9 * d))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, mem, c):
        s1, sc1, g1, s2, sc2, g2, s3, sc3, g3 = self.ada(c).chunk(9, -1)
        h = modulate(self.n1(x), s1, sc1)
        x = x + g1.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = modulate(self.n2(x), s2, sc2)
        x = x + g2.unsqueeze(1) * self.xattn(h, mem, mem, need_weights=False)[0]
        h = modulate(self.n3(x), s3, sc3)
        return x + g3.unsqueeze(1) * self.mlp(h)


class DiTDenoiser(nn.Module):
    def __init__(self, action_dim: int, horizon: int, d: int, depth: int, heads: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(action_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, horizon, d)); nn.init.normal_(self.pos, std=0.02)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.c_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.mem_norm = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([DiTBlock(d, heads, dropout=dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, action_dim)
        for m in (self.out_ada[1], self.out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        self.d = d

    def forward(self, noisy, k, tokens, pooled):
        c = self.t_mlp(timestep_embedding(k, self.d)) + self.c_mlp(pooled)
        x = self.inp(noisy) + self.pos
        mem = self.mem_norm(tokens)
        for b in self.blocks:
            x = b(x, mem, c)
        shift, scale = self.out_ada(c).chunk(2, -1)
        return self.out(modulate(self.out_norm(x), shift, scale))


class DiTPolicy(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.cfg = cfg
        self.To, self.H, self.Da = cfg["obs_horizon"], cfg["action_horizon"], cfg["action_dim"]
        self.encoder = ImageEncoder(m["d_model"], self.To, pretrained=m["pretrained_encoder"],
                                    group_norm=m["group_norm"], token_hw=tuple(m["token_hw"]))
        self.denoiser = DiTDenoiser(self.Da, self.H, m["d_model"], m["depth"], m["heads"], m["dropout"])
        self.Ds = state_dim(cfg.get("state") or [])
        if self.Ds:
            d = m["d_model"]
            self.state_enc = nn.Sequential(nn.Linear(self.Ds, d), nn.SiLU(), nn.Linear(d, d))
            self.state_frame = nn.Parameter(torch.zeros(1, self.To, d)); nn.init.normal_(self.state_frame, std=0.02)

    def encode(self, images_uint8: torch.Tensor, state: torch.Tensor | None = None):
        """images (B, To, 3, h, w) uint8, state (B, To, Ds) -> memory tokens (B, N, d), pooled (B, d)."""
        tok, pooled = self.encoder(images_uint8.float() / 255.0)
        if self.Ds:
            if state is None:
                raise ValueError("this policy was trained with state inputs: pass state")
            st = self.state_enc(state.to(tok.dtype)) + self.state_frame   # (B, To, d)
            tok = torch.cat([tok, st], 1)
            pooled = pooled + st.mean(1)
        return tok, pooled

    def forward(self, images_uint8, noisy, k, state=None):
        tok, pooled = self.encode(images_uint8, state)
        return self.denoiser(noisy, k, tok, pooled)
