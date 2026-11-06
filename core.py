"""ROS-free teleop logic: UR5e kinematics, clutch mapping, and motion limits."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Nominal UR5e DH, ordered shoulder_pan .. wrist_3. Control is differential, so
# small differences from the arm's factory calibration do not accumulate.
_A = (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0)
_D = (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996)
_ALPHA = (np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0)

JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


def _dh(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _chain(q: np.ndarray) -> list[np.ndarray]:
    frames, T = [np.eye(4)], np.eye(4)
    for i in range(6):
        T = T @ _dh(_A[i], _ALPHA[i], _D[i], q[i])
        frames.append(T)
    return frames


def fk(q: np.ndarray, tcp_offset: np.ndarray | None = None) -> np.ndarray:
    """Pose of the TCP (or tool0 with no offset) in the UR `base` frame."""
    T = _chain(np.asarray(q, dtype=float))[-1]
    return T if tcp_offset is None else T @ tcp_offset


def jacobian(q: np.ndarray, tcp_offset: np.ndarray | None = None) -> np.ndarray:
    frames = _chain(np.asarray(q, dtype=float))
    end = frames[-1] if tcp_offset is None else frames[-1] @ tcp_offset
    J = np.zeros((6, 6))
    for i in range(6):
        z, p = frames[i][:3, 2], frames[i][:3, 3]
        J[:3, i] = np.cross(z, end[:3, 3] - p)
        J[3:, i] = z
    return J


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    # Quaternion route: stable at both 0 and pi, where the acos formula is not.
    t = np.trace(R)
    q = np.array([
        t,
        R[0, 0] - R[1, 1] - R[2, 2],
        R[1, 1] - R[0, 0] - R[2, 2],
        R[2, 2] - R[0, 0] - R[1, 1],
    ])
    k = int(np.argmax(q))
    if k == 0:
        w, v = 1 + t, np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    else:
        i, j, l = k - 1, k % 3, (k + 1) % 3
        v = np.zeros(3)
        v[i] = 1 + q[k]
        v[j] = R[j, i] + R[i, j]
        v[l] = R[l, i] + R[i, l]
        w = R[l, j] - R[j, l]
    n = np.sqrt(w * w + v @ v)
    w, v = w / n, v / n
    if w < 0:
        w, v = -w, -v
    return 2 * np.arctan2(np.linalg.norm(v), w) * (v / max(np.linalg.norm(v), 1e-12))


def matrix_from_rotvec(r: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    x, y, z = r / th
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def operator_to_base(head: np.ndarray, robot_forward_yaw: float) -> np.ndarray:
    """Rotation from Quest world (y up) to the UR base frame (z up).

    "Forward" is the way the headset faces, ignoring tilt, and it maps to the robot's
    chosen forward direction.
    """
    R = head[:3, :3]
    up = np.array([0.0, 1.0, 0.0])
    look = -R[:, 2]
    forward = look - (look @ up) * up
    if np.linalg.norm(forward) < 0.3:
        # Looking almost straight down or up: the top of the headset still points level.
        top = R[:, 1] if look[1] < 0 else -R[:, 1]
        forward = top - (top @ up) * up
    forward = forward / np.linalg.norm(forward)
    left = np.cross(up, forward)
    quest_axes = np.column_stack([forward, left, up])
    c, s = np.cos(robot_forward_yaw), np.sin(robot_forward_yaw)
    base_axes = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return base_axes @ quest_axes.T


def jog_velocity(jog, speed: float, deadband: float = 0.15) -> np.ndarray:
    """Turn (stick x, stick y, in/out) into a tool-frame velocity.

    The dead zone is per axis so a nudge moves along one axis only, and the squared stick
    response gives fine control near the centre. In/out comes from buttons, so it has one
    speed: half of full stick. Stick up is tool +x and stick right is tool +y: looking along
    the tool's +z, that keeps the stick's layout instead of mirroring it.
    """
    def stick(v: float) -> float:
        size = (min(abs(v), 1.0) - deadband) / (1.0 - deadband)
        return float(np.sign(v)) * size ** 2 if size > 0 else 0.0

    x, y, z = jog
    return speed * np.array([stick(y), stick(x), 0.5 * float(np.clip(z, -1.0, 1.0))])


@dataclass
class Limits:
    # Deliberately slow for first runs; raise only after testing.
    max_linear_speed: float = 0.10      # m/s
    max_angular_speed: float = 0.50     # rad/s
    max_joint_speed: float = 0.50       # rad/s
    joint_limit: float = 2 * np.pi - 0.05  # rad; UR5e wrists can reach +/-2pi
    elbow_limit: float = np.pi - 0.05   # rad
    box_half_extent: float = 0.40       # m, around the TCP pose at startup
    min_z: float | None = None          # m in the base frame; lowest allowed TCP height
    max_controller_jump: float = 0.10   # m between frames: treated as a glitch
    max_controller_jump_rot: float = 0.60  # rad between frames


@dataclass
class Mapping:
    # Direction in the UR base frame (rad from +x) that counts as the robot's forward.
    # Fixed by the setup, so it does not change when the operator moves.
    robot_forward_yaw: float = np.pi  # base -x, measured for this setup
    # Used only when no headset pose is supplied (tests and scripted sources).
    operator_to_base: np.ndarray = field(default_factory=lambda: np.eye(3))
    position_scale: float = 0.5
    rotation_scale: float = 1.0
    tcp_offset: np.ndarray = field(default_factory=lambda: np.eye(4))


class TeleopCore:
    """Turns controller poses into rate-limited joint targets.

    Motion happens only while the clutch is held. Each engagement re-references
    both poses, so gripping never makes the arm jump.
    """

    def __init__(self, q0, mapping: Mapping | None = None, limits: Limits | None = None):
        self.map = mapping or Mapping()
        self.lim = limits or Limits()
        self.q_cmd = np.array(q0, dtype=float)
        self.target = fk(self.q_cmd, self.map.tcp_offset)
        self.box_center = self.target[:3, 3].copy()
        self.engaged = False
        self.latched_glitch = False
        self.status = "idle"
        self._ctrl0 = self._tcp0 = self._last_ctrl = None
        self._rotation = self.map.operator_to_base

    def resync(self, q_actual) -> None:
        """Adopt the measured joints and drop the clutch (used after any fault)."""
        self.q_cmd = np.array(q_actual, dtype=float)
        self.target = fk(self.q_cmd, self.map.tcp_offset)
        self.engaged = False
        self._last_ctrl = None

    def update(self, ctrl: np.ndarray | None, clutch: bool, dt: float, head: np.ndarray | None = None) -> np.ndarray:
        if ctrl is None or not clutch:
            self.engaged = False
            self.latched_glitch = False if not clutch else self.latched_glitch
            self.status = "tracking lost" if clutch else "idle"
            self._last_ctrl = None
            self.target = fk(self.q_cmd, self.map.tcp_offset)
            return self.q_cmd

        if self.latched_glitch:
            self.status = "glitch: release and re-grip"
            return self.q_cmd

        if self._is_jump(ctrl):
            self.engaged = False
            self.latched_glitch = True
            self.status = "glitch: release and re-grip"
            self.target = fk(self.q_cmd, self.map.tcp_offset)
            return self.q_cmd
        self._last_ctrl = ctrl.copy()

        if not self.engaged:
            self.engaged = True
            self._ctrl0 = ctrl.copy()
            # Heading is fixed for the whole grip, so looking around mid-motion changes nothing.
            if head is not None:
                self._rotation = operator_to_base(head, self.map.robot_forward_yaw)
            self._tcp0 = fk(self.q_cmd, self.map.tcp_offset)
            self.target = self._tcp0.copy()
            self.status = "engaged"
            return self.q_cmd

        self.status = "engaged"  # _desired and _ik_step may replace this with why motion stopped
        self.target = self._limited_target(self._desired(ctrl), dt)
        self.q_cmd = self._ik_step(self.target, dt)
        return self.q_cmd

    def jog(self, velocity_tool: np.ndarray, dt: float) -> np.ndarray:
        """Move along the tool's own axes, holding its orientation. Same limits as hand motion."""
        self.engaged = False
        self.latched_glitch = False
        self._last_ctrl = None
        self.status = "jogging"
        velocity = np.asarray(velocity_tool, dtype=float)
        n = np.linalg.norm(velocity)
        if n > self.lim.max_linear_speed:
            velocity = velocity * self.lim.max_linear_speed / n
        out = self.target.copy()
        out[:3, 3] = self._clip_to_workspace(out[:3, 3] + out[:3, :3] @ velocity * dt, out[2, 3])
        self.target = out
        self.q_cmd = self._ik_step(out, dt)
        return self.q_cmd

    def grip_motion(self) -> tuple[float, float, float]:
        """Target motion since the grip was pressed, as (forward, left, up) of the robot."""
        d = self.target[:3, 3] - self._tcp0[:3, 3]
        c, s = np.cos(self.map.robot_forward_yaw), np.sin(self.map.robot_forward_yaw)
        return float(d @ [c, s, 0.0]), float(d @ [-s, c, 0.0]), float(d[2])

    def _is_jump(self, ctrl: np.ndarray) -> bool:
        if self._last_ctrl is None:
            return False
        dp = np.linalg.norm(ctrl[:3, 3] - self._last_ctrl[:3, 3])
        dr = np.linalg.norm(rotvec_from_matrix(ctrl[:3, :3] @ self._last_ctrl[:3, :3].T))
        return dp > self.lim.max_controller_jump or dr > self.lim.max_controller_jump_rot

    def _desired(self, ctrl: np.ndarray) -> np.ndarray:
        R = self._rotation
        out = np.eye(4)
        out[:3, 3] = self._tcp0[:3, 3] + self.map.position_scale * R @ (ctrl[:3, 3] - self._ctrl0[:3, 3])
        rel = rotvec_from_matrix(ctrl[:3, :3] @ self._ctrl0[:3, :3].T)
        out[:3, :3] = matrix_from_rotvec(self.map.rotation_scale * R @ rel) @ self._tcp0[:3, :3]
        out[:3, 3] = self._clip_to_workspace(out[:3, 3], self._tcp0[2, 3])
        return out

    def _clip_to_workspace(self, position: np.ndarray, start_z: float) -> np.ndarray:
        half = self.lim.box_half_extent
        low, high = self.box_center - half, self.box_center + half
        if self.lim.min_z is not None:
            # Motion starting below the floor may stay there; it must not lift by itself.
            low[2] = max(low[2], min(self.lim.min_z, start_z))
        clipped = np.clip(position, low, high)
        if not np.array_equal(clipped, position):
            self.status = "workspace limit"
        return clipped

    def _limited_target(self, desired: np.ndarray, dt: float) -> np.ndarray:
        out = self.target.copy()
        dp = desired[:3, 3] - out[:3, 3]
        n = np.linalg.norm(dp)
        out[:3, 3] += dp * min(1.0, self.lim.max_linear_speed * dt / max(n, 1e-12))
        dr = rotvec_from_matrix(desired[:3, :3] @ out[:3, :3].T)
        n = np.linalg.norm(dr)
        out[:3, :3] = matrix_from_rotvec(dr * min(1.0, self.lim.max_angular_speed * dt / max(n, 1e-12))) @ out[:3, :3]
        return out

    def _ik_step(self, target: np.ndarray, dt: float) -> np.ndarray:
        q, off = self.q_cmd.copy(), self.map.tcp_offset
        for _ in range(4):
            cur = fk(q, off)
            err = np.concatenate([target[:3, 3] - cur[:3, 3], rotvec_from_matrix(target[:3, :3] @ cur[:3, :3].T)])
            if np.linalg.norm(err) < 1e-5:
                break
            J = jacobian(q, off)
            q = q + J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err)
        step = np.clip(q - self.q_cmd, -self.lim.max_joint_speed * dt, self.lim.max_joint_speed * dt)
        next_q = self.q_cmd + step
        limit = np.full(6, self.lim.joint_limit)
        limit[2] = self.lim.elbow_limit
        if np.any(np.abs(next_q) > limit):
            self.status = "joint limit"
            self.target = fk(self.q_cmd, off)
            return self.q_cmd
        return next_q
