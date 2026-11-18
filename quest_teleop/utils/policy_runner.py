"""Runs a trained image-only DiT policy (DeviceAssembly/policy) on wrist images. No ROS.

A deployed policy is a directory, made by scripts/deploy_policy.sh:

    <dir>/final.pt        checkpoint (weights, config, action normalizer)
    <dir>/src/dit_policy  the policy code it was trained with (so code and weights always match)
    <dir>/meta.json       the training dataset's meta.json (stored image size), optional

Policies trained with state inputs (config `state`) also need the measured tool0 pose, jaw gap and,
for `wrench`, the filtered wrist force/torque with each image; image-only ones ignore them.

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


def pose_source(policy_dir: str | pathlib.Path) -> str:
    """Which measured tool0 the policy was trained on (its dataset's meta.json "pose_source"):
    "nominal_fk" (core.fk of the joints) or "driver" (/tcp_pose_broadcaster/pose). Datasets from
    before the field existed used the driver's pose."""
    meta = pathlib.Path(policy_dir) / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text()).get("pose_source", "driver")
    return "driver"


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


def tilt_of(R: np.ndarray) -> tuple[float, np.ndarray]:
    """Tilt of a tool0 orientation (base frame) from pointing straight down: (deg, direction), the
    direction being where the fingertips lean, as a unit (x, y) in the tool's own frame
    (zeros when vertical)."""
    z = R[:, 2]
    deg = float(np.degrees(np.arccos(np.clip(-z[2], -1.0, 1.0))))
    lean = (R.T @ np.array([z[0], z[1], 0.0]))[:2]   # horizontal part of the tool axis, in tool x/y
    n = np.linalg.norm(lean)
    return deg, (lean / n if n > 1e-9 else np.zeros(2))


class ContactCompliance:
    """Admittance for placing a held piece, as an offset on the policy's position targets (125 Hz,
    on top of the stiff position servo, like the PBIC layer of arXiv 2603.15152 but for flat landings).

    Uses the UR's wrist wrench (tool0_controller frame), low-passed and zeroed at the start of a run:
      - Flat landing, piece partly on the rim: the vertical reaction acts off the tool axis, so the
        torque says where. For a force F at a point r (sensor frame), tau = r x F, so with the contact
        `contact_z` m along tool z from where the UR reports torque:
            r_x = (r_z F_x - tau_y) / F_z,   r_y = (tau_x + r_z F_y) / F_z
        (unchanged if the sensor's sign convention is flipped). Once |F_z| > min_fz and |r| > min_r,
        the tool slides away from r, toward the hole, at `speed` m/s.
        Press test on the arm (2026-11-16): the sideways readings during an upward press are mostly
        sensor error (UR quotes ~3.5 N), and r_z times them swamps the torque, so contact_z defaults to
        0 (r = (-tau_y, tau_x) / F_z). Then a press under the front/back edge read 27 mm forward/back
        for a ~8 mm radius piece: right direction, ~3x the size, hence min_r 8 mm. Direction only.
      - Chamfers and side walls: sideways force beyond `force_deadband` slides it along that force at
        `force_gain` m/s per N (the force on the tool: pushes on the held piece gave way the right way).
      - Pushing harder than `push_limit` N along tool z: `pushing` is set, and the node stops descending.
    The offset is in base x, y, capped at `max_offset` m."""

    def __init__(self, speed: float = 0.0, contact_z: float = 0.0, min_fz: float = 3.0, min_r: float = 0.008,
                 max_offset: float = 0.003, force_gain: float = 0.001, force_deadband: float = 2.0,
                 push_limit: float = 12.0, filter_s: float = 0.05, edge_max_side: float = 0.3,
                 edge_dir: float = -1.0):
        self.speed, self.contact_z, self.min_fz, self.min_r = speed, contact_z, min_fz, min_r
        self.max_offset, self.force_gain, self.force_deadband = max_offset, force_gain, force_deadband
        self.push_limit, self.filter_s = push_limit, filter_s
        self.edge_max_side = edge_max_side   # torque (edge) path only while sideways force < this x |F_z|
        self.edge_dir = edge_dir             # -1: slide away from the touching edge, +1: toward it
        self.reset()

    def reset(self, keep_zero: bool = False) -> None:
        """Start of a run, arm still and touching nothing: the next reading is the zero. keep_zero
        (gripper closed, maybe pressing on the part): keep the zero taken before, so a run started
        mid-press doesn't count the press as nothing (it let one run push ~24 N)."""
        if not (keep_zero and getattr(self, "zero", None) is not None):
            self.zero = None
        self.w = np.zeros(6)
        self.offset = np.zeros(2)
        self.contact = None     # r (m, tool x, y) while touching
        self.pushing = False
        self.v_base = np.zeros(2)   # the slide this tick (m/s, base x, y)
        self.total = np.zeros(2)    # everything slid while holding this piece (base x, y), capped at max_offset
        self.path = ""              # which term is sliding: "force", "edge" or ""

    def release(self) -> None:
        """The piece was let go: the next one starts from no correction."""
        self.offset[:] = 0.0
        self.total[:] = 0.0

    def filter(self, wrench: np.ndarray, dt: float) -> None:
        wrench = np.asarray(wrench, float)
        if self.zero is None:
            self.zero = wrench.copy()
        a = dt / (self.filter_s + dt)
        self.w += a * (wrench - self.zero - self.w)
        fz = self.w[2]
        self.pushing = abs(fz) > self.push_limit

    def step(self, R: np.ndarray, dt: float, extra: np.ndarray | None = None) -> np.ndarray:
        """Move the offset for one tick (tool0 orientation R in base); `extra` adds a slide (m/s, base
        x, y), e.g. SpiralSearch's. Returns the offset (base x, y)."""
        F, T = self.w[:3], self.w[3:]
        v = np.zeros(2)
        self.contact = None
        self.path = ""
        side = np.linalg.norm(F[:2])
        # The edge estimate only holds for a nearly vertical push: real sideways contact force acts
        # ~0.15-0.19 m down the tool and its torque swamps the off-centre push (first --compliance on run).
        if abs(F[2]) > self.min_fz and side < self.edge_max_side * abs(F[2]):
            rz = self.contact_z
            r = np.array([(rz * F[0] - T[1]) / F[2], (T[0] + rz * F[1]) / F[2]])
            self.contact = r
            n = np.linalg.norm(r)
            if n > self.min_r:
                v += self.edge_dir * self.speed * r / n
                self.path = "edge"
        if self.force_gain and side > self.force_deadband:
            v += self.force_gain * (side - self.force_deadband) * F[:2] / side
            self.path = "force"
        self.v_base = R[:2, :2] @ v
        if extra is not None:
            self.v_base = self.v_base + extra
        total = self.total + self.v_base * dt
        n = np.linalg.norm(total)
        if n > self.max_offset:            # the cap is on the whole correction, across plans
            total *= self.max_offset / n
        self.offset += total - self.total
        self.total = total
        return self.offset

    def rebase(self, at_plan: np.ndarray) -> None:
        """A new plan starts from the measured pose, which already includes the offset applied when
        its frame was taken: keep only what was added since."""
        self.offset = self.offset - at_plan


class SpiralSearch:
    """Feel for the hole when the held piece sits flat on the rim: pressing, no sideways force to
    follow (it read < 2 N, and pointed the wrong way, in the rim landings), and no downward progress
    for `stall_s`. Then it spirals outward from there (radius growing `pitch` per turn, `speed` m/s
    along the path, up to `max_r`) while holding the push near `force` N, until the piece drops
    `drop` m below where it stalled (found) or the spiral reaches max_r (failed). One search per held
    piece. It gives a slide velocity for ContactCompliance.step, so replans keep what it moved."""

    def __init__(self, pitch: float = 0.001, speed: float = 0.005, max_r: float = 0.004, force: float = 6.0,
                 stall_s: float = 0.5, drop: float = 0.001, min_fz: float = 3.0, side_max: float = 2.0,
                 z_speed: float = 0.001, progress: float = 0.0003):
        self.pitch, self.speed, self.max_r, self.force = pitch, speed, max_r, force
        self.stall_s, self.drop, self.min_fz, self.side_max = stall_s, drop, min_fz, side_max
        self.z_speed, self.progress = z_speed, progress
        self.reset()

    def reset(self) -> None:
        self.state = "idle"     # idle -> searching -> found | failed
        self.theta = self.r = 0.0
        self.z_stall = None
        self._window = None     # (time, height) since the push started without progress

    def update(self, t: float, dt: float, fz: float, side: float, z: float) -> np.ndarray:
        """One tick: |F_z| fz and sideways force side (N, filtered and zeroed), measured height z.
        Returns the slide velocity (m/s, base x, y)."""
        fz = abs(fz)
        if self.state == "idle":
            if fz > self.min_fz and side < self.side_max:
                if self._window is None or z < self._window[1] - self.progress:
                    self._window = (t, z)               # (still) going down: restart the clock
                elif t - self._window[0] >= self.stall_s:
                    self.state, self.z_stall, self.theta, self.r = "searching", z, 0.0, 0.0
            else:
                self._window = None
            return np.zeros(2)
        if self.state != "searching":
            return np.zeros(2)
        if z < self.z_stall - self.drop:
            self.state = "found"
            return np.zeros(2)
        if fz < self.min_fz and z > self.z_stall + 0.002:    # the policy lifted away: start over
            self.reset()
            return np.zeros(2)
        theta = self.theta + self.speed * dt / max(self.r, 0.0005)
        r = self.pitch * theta / (2 * np.pi)
        if r > self.max_r:
            self.state = "failed"
            return np.zeros(2)
        step = r * np.array([np.cos(theta), np.sin(theta)]) - self.r * np.array([np.cos(self.theta), np.sin(self.theta)])
        self.theta, self.r = theta, r
        return step / dt

    def z_target(self, target_z: float, fz: float, dt: float) -> float:
        """While searching: rise if pushing harder than `force`, sink if under half of it."""
        fz = abs(fz)
        if fz > self.force:
            return target_z + self.z_speed * dt
        if fz < 0.5 * self.force:
            return target_z - self.z_speed * dt
        return target_z


def contact_descent(desired_z: float, target_z: float, fz: float, min_fz: float, pushing: bool,
                    push_z: float | None, dt: float, descent: float = 0.002, backoff: float = 0.0005,
                    backoff_speed: float = 0.005) -> tuple[float, float | None, str]:
    """Height for the next target while placing (base z, m). target_z: the current target.
    - Pushing past the limit: rise at `backoff_speed` to `backoff` above where it started pushing,
      then hold there (the policy may still lift higher).
    - Touching (|fz| > min_fz): descend no faster than `descent` m/s, so sideways corrections have time.
    Returns (z, the height where pushing started or None, state for the log)."""
    if pushing:
        push_z = target_z if push_z is None else push_z
        return max(desired_z, min(target_z + backoff_speed * dt, push_z + backoff)), push_z, "PUSH-BACKOFF"
    if abs(fz) > min_fz:
        return max(desired_z, target_z - descent * dt), None, "contact-slow"
    return desired_z, None, ""


FORCE_MODE_AXES = ("x", "y", "z", "rx", "ry", "rz")


def force_mode_settings(axes: str, push: float) -> tuple[list[bool], list[float]]:
    """--fm-axes ("x,y", "x,y,z", ...) and --fm-push (N) -> UR force_mode selection vector and wrench,
    in the base frame. Compliant axes get zero force (follow whatever pushes), except z, which pushes
    down with `push` N (the wrench is what the robot applies to its surroundings: down = -z)."""
    names = [a.strip().lower() for a in axes.split(",") if a.strip()]
    bad = [a for a in names if a not in FORCE_MODE_AXES]
    if bad or not names:
        raise ValueError(f"force mode axes must be some of {','.join(FORCE_MODE_AXES)}, got {axes!r}")
    selection = [a in names for a in FORCE_MODE_AXES]
    wrench = [0.0] * 6
    if selection[2]:
        wrench[2] = -abs(push)
    return selection, wrench


def force_mode_override(desired: np.ndarray, measured: np.ndarray, selection: list[bool]) -> np.ndarray:
    """While in force mode the arm ignores the commanded pose along compliant axes, so command the
    measured pose there: the command stays on the arm (no tracking-error hold) and nothing jumps when
    force mode ends. Rotations come from the measured pose if any rotational axis is compliant."""
    out = desired.copy()
    for i in range(3):
        if selection[i]:
            out[i, 3] = measured[i, 3]
    if any(selection[3:]):
        out[:3, :3] = measured[:3, :3]
    return out


class ForceModeGate:
    """When to be in UR force mode: holding the piece (gripper closed) and the plan going down.
      - Enters when the plan's target `lookahead` s ahead is more than `enter` m below the current one
        (and, with min_fz > 0, the piece is already touching: |F_z| over it).
      - Leaves at once when the gripper opens (releasing in force mode would push on the fingers), or
        when the plan rises more than `exit` m over the lookahead (lifting out).
      - Between the two (holding still), it stays as it is."""

    def __init__(self, enter: float = 0.001, exit: float = 0.001, lookahead: float = 0.3, min_fz: float = 0.0):
        self.enter, self.exit, self.lookahead, self.min_fz = enter, exit, lookahead, min_fz
        self.reset()

    def reset(self) -> None:
        self.engaged = False
        self.reason = ""

    def planned_dz(self, chunk: "Chunk", now: float) -> float:
        """How far the plan goes up (+) or down (-) over the next `lookahead` s (m)."""
        return float(chunk.at(now + self.lookahead)[0][2, 3] - chunk.at(now)[0][2, 3])

    def update(self, closed: bool, dz: float, fz: float = 0.0) -> bool | None:
        """Returns the new state when it changes, else None; `reason` says why."""
        if self.engaged:
            if not closed:
                return self._set(False, "gripper opened")
            if dz > self.exit:
                return self._set(False, f"plan lifting {dz * 1000:.1f} mm")
            return None
        if closed and dz < -self.enter and abs(fz) >= self.min_fz:
            return self._set(True, f"holding, plan descending {-dz * 1000:.1f} mm")
        return None

    def _set(self, on: bool, reason: str) -> bool:
        self.engaged, self.reason = on, reason
        return on


class GripperHysteresis:
    """Commanded opening (m) -> closed/open, with a dead band so a prediction near the middle
    doesn't flicker the gripper. None until the policy first asks to close: commanding open at
    startup would drop anything held (as in teleop). Once closed it stays closed for at least
    `min_closed_s`, even if the prediction jumps straight back open, so a one-plan blip still grips."""

    def __init__(self, close_below: float = 0.030, open_above: float = 0.055, min_closed_s: float = 0.0):
        self.close_below, self.open_above, self.min_closed_s = close_below, open_above, min_closed_s
        self.closed: bool | None = None
        self.closed_at: float | None = None

    def update(self, opening: float, now: float | None = None) -> bool | None:
        """Returns the new state when it changes, else None. `now` (s, monotonic) times the hold."""
        if self.closed:
            want = opening < self.open_above
            if not want and now is not None and self.closed_at is not None and now - self.closed_at < self.min_closed_s:
                return None   # too soon after closing: hold
        else:
            want = opening < self.close_below
        if self.closed is None and not want:
            return None
        if want == self.closed:
            return None
        self.closed = want
        self.closed_at = now if want else None
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
        self.state_keys = tuple(getattr(self.policy, "state_keys", ()))

    def reset(self) -> None:
        self.policy.reset()

    def observe(self, rgb: np.ndarray, tool0: np.ndarray | None = None, gripper_width: float | None = None,
                wrench: np.ndarray | None = None) -> None:
        if "wrench" in self.state_keys:
            self.policy.observe(rgb, tool0, gripper_width, wrench)
        elif self.state_keys:   # policies from before the wrench input take three arguments
            self.policy.observe(rgb, tool0, gripper_width)
        else:
            self.policy.observe(rgb)

    def plan(self, tool0: np.ndarray, t_obs: float) -> Chunk:
        """Plan from the frames observed so far; tool0 is the measured pose when the last one was taken."""
        action = self.policy.plan()   # (H, 10), unnormalized
        poses, grip = self._to_poses(tool0, action, self.policy.space)
        return Chunk(t0=t_obs, dt=self.dt, poses=poses, grip=np.asarray(grip, float))

    def warmup(self) -> None:
        """One plan on a blank image, so CUDA kernels compile before the arm moves."""
        self.observe(np.zeros((*self.hw, 3), np.uint8), np.eye(4), 0.085, np.zeros(6))
        self.policy.plan()
        self.reset()


def _planner_main(conn, policy_dir: str, checkpoint: str, device: str | None) -> None:
    """Child process: owns torch and the policy, so planning never competes with the ROS
    callbacks and control loop for the interpreter lock. Messages:
      ("observe", jpeg, tool0, width, wrench)  decode, resize, append to the history (no reply)
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
                             "dt": r.dt, "state": list(r.state_keys), "warmup_ms": (time.monotonic() - t0) * 1000,
                             "mount_rotation_deg": float(((r.policy.cfg.get("train") or {}).get("mount_rotation") or {}).get("deg", 0.0))}))
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
                r.observe(preprocess(msg[1], r.hw), *msg[2:])
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
        self.state_keys = tuple(info.get("state", ()))
        self.mount_rotation_deg = float(info.get("mount_rotation_deg", 0.0))   # trained-in mount-turn augmentation

    def observe(self, jpeg: bytes, tool0: np.ndarray | None = None, gripper_width: float | None = None,
                wrench: np.ndarray | None = None) -> None:
        self._conn.send(("observe", jpeg, tool0, gripper_width, wrench))

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


class WrenchFilter:
    """The wrist wrench as policies with a `wrench` input were trained on (dit_policy.geometry.
    WRENCH_FILTER, convert_rosbags.wrench_at): the mean of the last 0.1 s of samples, minus the
    median over the 0.5 s before the run started, when the arm was still and touching nothing.
    Thread-safe: add() from the 500 Hz subscription, the rest from the policy thread."""

    WINDOW_S, ZERO_S = 0.1, 0.5

    def __init__(self, keep_s: float = 1.0):
        import collections
        import threading
        self._buf = collections.deque()
        self._lock = threading.Lock()
        self.keep_s = keep_s
        self.bias = None

    def add(self, t: float, w) -> None:
        with self._lock:
            self._buf.append((t, np.asarray(w, float)))
            while self._buf and self._buf[0][0] < t - self.keep_s:
                self._buf.popleft()

    def _since(self, t0: float) -> np.ndarray:
        with self._lock:
            return np.array([w for t, w in self._buf if t > t0]).reshape(-1, 6)

    def zero(self, now: float) -> bool:
        """Take the bias from the last ZERO_S s. False if there are no samples."""
        w = self._since(now - self.ZERO_S)
        if not len(w):
            return False
        self.bias = np.median(w, 0)
        return True

    def value(self, now: float) -> np.ndarray | None:
        """Filtered, zeroed wrench (6,), or None if nothing arrived in the last WINDOW_S s (or never zeroed)."""
        w = self._since(now - self.WINDOW_S)
        if not len(w) or self.bias is None:
            return None
        return w.mean(0) - self.bias


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
