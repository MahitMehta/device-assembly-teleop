"""Runs a trained image-only DiT policy (DeviceAssembly/policy) on wrist images. No ROS.

A deployed policy is a directory, made by scripts/deploy_policy.sh:

    <dir>/final.pt        checkpoint (weights, config, action normalizer)
    <dir>/src/dit_policy  the policy code it was trained with (so code and weights always match)
    <dir>/meta.json       the training dataset's meta.json (stored image size), optional

The pieces that need no torch (preprocessing, waypoint timing, gripper hysteresis) are here too,
so they can be tested without a GPU.
"""
from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

DEFAULT_IMAGE_HW = (192, 256)   # what convert_rosbags.py stores; meta.json overrides it


def image_hw(policy_dir: str | pathlib.Path) -> tuple[int, int]:
    meta = pathlib.Path(policy_dir) / "meta.json"
    if meta.exists():
        return tuple(json.loads(meta.read_text())["image_hw"])
    return DEFAULT_IMAGE_HW


def preprocess(jpeg: bytes, hw: tuple[int, int]) -> np.ndarray:
    """JPEG from /camera/camera/color/image_raw/compressed -> (H, W, 3) uint8 RGB at the stored size.
    Exactly as convert_rosbags.py made the training images (INTER_AREA from 640x480, then BGR->RGB)."""
    import cv2
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode the camera image")
    return np.ascontiguousarray(cv2.resize(bgr, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)[:, :, ::-1])


@dataclass
class Chunk:
    """A planned chunk: tool0 targets in base at t0, t0 + dt, ..., and the commanded opening."""
    t0: float
    dt: float
    poses: np.ndarray   # (H, 4, 4)
    grip: np.ndarray    # (H,) m, 0 = closed

    def end(self) -> float:
        return self.t0 + self.dt * (len(self.poses) - 1)

    def at(self, t: float) -> tuple[np.ndarray, float]:
        """Target at time t: linear in position, slerp in rotation, held at either end."""
        s = float(np.clip((t - self.t0) / self.dt, 0.0, len(self.poses) - 1))
        i = min(int(s), len(self.poses) - 2) if len(self.poses) > 1 else 0
        f = s - i
        if len(self.poses) == 1 or f <= 0.0:
            return self.poses[i].copy(), float(self.grip[i])
        a, b = self.poses[i], self.poses[i + 1]
        out = np.eye(4)
        out[:3, 3] = (1 - f) * a[:3, 3] + f * b[:3, 3]
        out[:3, :3] = _slerp(a[:3, :3], b[:3, :3], f)
        return out, float(self.grip[i] if f < 0.5 else self.grip[i + 1])


def _slerp(Ra: np.ndarray, Rb: np.ndarray, f: float) -> np.ndarray:
    from quest_teleop.utils.core import matrix_from_rotvec, rotvec_from_matrix
    return matrix_from_rotvec(f * rotvec_from_matrix(Rb @ Ra.T)) @ Ra


class GripperHysteresis:
    """Commanded opening (m) -> closed/open, with a dead band so a prediction near the middle
    doesn't flicker the gripper, and a minimum hold: after a change the state stays for
    `min_hold_s`, so the fingers actually close (or open) before the next plan can undo it.
    Successive plans often disagree about the exact moment to grasp; without the hold one plan's
    "close" is overturned by the next plan's "open" a few ms later, the jaws never move, the
    camera never sees a grasp, and the policy hovers over the part. None until the policy first
    asks to close: commanding open at startup would drop anything held (as in teleop)."""

    def __init__(self, close_below: float = 0.030, open_above: float = 0.055, min_hold_s: float = 1.0):
        self.close_below, self.open_above, self.min_hold_s = close_below, open_above, min_hold_s
        self.closed: bool | None = None
        self._changed_at = -float("inf")

    def update(self, opening: float, t: float = 0.0) -> bool | None:
        """Returns the new state when it changes, else None. t: seconds (monotonic) for the hold."""
        if self.closed is not None and t - self._changed_at < self.min_hold_s:
            return None
        if self.closed:
            want = opening < self.open_above
        else:
            want = opening < self.close_below
        if self.closed is None and not want:
            return None
        if want == self.closed:
            return None
        self.closed = want
        self._changed_at = t
        return want


class DiTRunner:
    """Loads a deployed policy directory and plans chunks. Needs torch."""

    def __init__(self, policy_dir: str | pathlib.Path, checkpoint: str = "final.pt", device: str | None = None):
        d = pathlib.Path(policy_dir)
        src = d / "src"
        if not (src / "dit_policy").is_dir():
            raise FileNotFoundError(f"{src}/dit_policy missing: deploy with scripts/deploy_policy.sh")
        sys.path.insert(0, str(src))
        from dit_policy.geometry import action_to_poses
        from dit_policy.policy import Policy
        self._to_poses = action_to_poses
        self.policy = Policy.load(str(d / checkpoint), device=device)
        self.hw = image_hw(d)
        self.dt = 1.0 / 10.0   # the dataset's rate (convert_rosbags --hz)
        self.n_exec = self.policy.n_exec
        self.horizon = self.policy.cfg["action_horizon"]

    def reset(self) -> None:
        self.policy.reset()

    def observe(self, rgb: np.ndarray) -> None:
        self.policy.observe(rgb)

    def plan(self, tool0: np.ndarray, t_obs: float) -> Chunk:
        """Plan from the frames observed so far; tool0 is the measured pose when the last one was taken."""
        action = self.policy.plan()   # (H, 10), unnormalized
        poses, grip = self._to_poses(tool0, action, self.policy.space)
        return Chunk(t0=t_obs, dt=self.dt, poses=poses, grip=np.asarray(grip, float))

    def warmup(self) -> None:
        """One plan on a blank image, so CUDA kernels compile before the arm moves."""
        self.observe(np.zeros((*self.hw, 3), np.uint8))
        self.policy.plan()
        self.reset()


def _planner_main(conn, policy_dir: str, checkpoint: str, device: str | None) -> None:
    """Child process: owns torch and the policy, so planning never competes with the ROS
    callbacks and control loop for the interpreter lock. Messages:
      ("observe", jpeg)      decode, resize, append to the frame history (no reply)
      ("plan", tool0, t)     reply ("chunk", Chunk, ms)
      ("reset",)             clear the frame history (no reply)
    On start it replies ("ready", info) after a warm-up plan, or ("error", text).
    Ctrl-C is for the parent: it closes the pipe, which ends this process."""
    import signal
    import time
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        r = DiTRunner(policy_dir, checkpoint, device)
        t0 = time.monotonic(); r.warmup()
        conn.send(("ready", {"device": str(r.policy.dev), "hw": r.hw, "horizon": r.horizon, "n_exec": r.n_exec,
                             "dt": r.dt, "warmup_ms": (time.monotonic() - t0) * 1000}))
    except Exception as e:  # report, don't hang the parent
        conn.send(("error", repr(e)))
        return
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] == "observe":
            try:
                r.observe(preprocess(msg[1], r.hw))
            except ValueError:
                pass   # an undecodable frame: keep the history as it was
        elif msg[0] == "plan":
            t0 = time.monotonic()
            chunk = r.plan(msg[1], msg[2])
            conn.send(("chunk", chunk, (time.monotonic() - t0) * 1000))
        elif msg[0] == "reset":
            r.reset()
        elif msg[0] == "quit":
            return


class PlannerProcess:
    """DiTRunner in a child process (see _planner_main). Not thread-safe: use from one thread."""

    def __init__(self, policy_dir: str, checkpoint: str = "final.pt", device: str | None = None, timeout: float = 300.0):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")   # a fresh interpreter: no forked ROS or CUDA state
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_planner_main, args=(child, str(policy_dir), checkpoint, device), daemon=True)
        self._proc.start()
        if not self._conn.poll(timeout):
            raise RuntimeError("policy process did not start in time")
        kind, info = self._conn.recv()
        if kind != "ready":
            raise RuntimeError(f"policy process failed: {info}")
        self.info = info
        self.dt, self.n_exec, self.horizon, self.hw = info["dt"], info["n_exec"], info["horizon"], tuple(info["hw"])

    def observe(self, jpeg: bytes) -> None:
        self._conn.send(("observe", jpeg))

    def reset(self) -> None:
        self._conn.send(("reset",))

    def plan(self, tool0: np.ndarray, t_obs: float, timeout: float = 5.0) -> tuple[Chunk, float]:
        self._conn.send(("plan", tool0, t_obs))
        if not self._conn.poll(timeout):
            raise RuntimeError(f"no plan within {timeout:.0f} s")
        _, chunk, ms = self._conn.recv()
        return chunk, ms

    def close(self) -> None:
        try:
            self._conn.send(("quit",))
        except (BrokenPipeError, OSError):
            pass
        self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()


class AutoStop:
    """Decides that the task looks done: the gripper has closed and opened again in this run
    (the part was released), and since then every plan has moved the tool less than `mm` over
    the part of it that runs, for `seconds`. The policy has no success signal of its own; after
    releasing, a demo-trained policy tends to hover, which is what this looks for."""

    def __init__(self, seconds: float = 2.0, mm: float = 3.0):
        self.seconds, self.mm = seconds, mm
        self.reset()

    def reset(self) -> None:
        self.closed_once = self.released = False
        self._still_since = None

    def gripper(self, closed: bool) -> None:
        if closed:
            self.closed_once = True
        elif self.closed_once:
            self.released = True

    def plan(self, chunk: Chunk, n_exec: int, t: float) -> bool:
        """Feed each new plan; True once the task looks done."""
        if not self.seconds or not self.released:
            self._still_since = None
            return False
        p = chunk.poses[:n_exec, :3, 3]
        span_mm = float(np.max(np.linalg.norm(p - p[0], axis=1))) * 1000
        if span_mm >= self.mm:
            self._still_since = None
            return False
        if self._still_since is None:
            self._still_since = t
        return t - self._still_since >= self.seconds
