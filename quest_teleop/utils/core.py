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


def jog_velocity(jog, speed: float, yaw_speed: float, deadband: float = 0.15) -> np.ndarray:
    """Turn (right stick x, right stick y, in/out, left stick x) into
    (forward m/s, yaw rad/s, in/out m/s, sideways m/s).

    Right stick up/down drives forward/back along the way the gripper faces, and right stick
    left/right drives it level along its x axis. Left stick right turns it clockwise seen from
    above. The dead zone is per axis so a nudge does one thing only, and the squared stick
    response gives fine control near the centre. In/out comes from buttons, so it has one
    speed: half of full stick.
    """
    def stick(v: float) -> float:
        size = (min(abs(v), 1.0) - deadband) / (1.0 - deadband)
        return float(np.sign(v)) * size ** 2 if size > 0 else 0.0

    x, y, z, turn = jog
    return np.array([speed * stick(y), -yaw_speed * stick(turn), 0.5 * speed * float(np.clip(z, -1.0, 1.0)),
                     speed * stick(x)])


def heading(R: np.ndarray, axis: int = 1) -> np.ndarray | None:
    """The way the gripper faces, flat on the table: tool +y with its vertical part removed
    (axis=0: tool +x, the sideways direction, likewise).

    None when that axis is nearly vertical, so the direction is not defined.
    """
    flat = np.array([R[0, axis], R[1, axis], 0.0])
    n = np.linalg.norm(flat)
    return flat / n if n > 0.2 else None


def pointing_down(R: np.ndarray) -> np.ndarray:
    """The orientation with tool +z straight down (normal to the table) closest to R.

    It turns about the horizontal axis only, so the yaw stays as close to R's as possible.
    """
    z, down = R[:, 2], np.array([0.0, 0.0, -1.0])
    axis = np.cross(z, down)
    angle = np.arctan2(np.linalg.norm(axis), z @ down)
    if np.linalg.norm(axis) < 1e-9:
        if z @ down > 0:
            return R.copy()
        axis = R[:, 0]  # pointing straight up: any horizontal axis turns it down
    return matrix_from_rotvec(angle * axis / np.linalg.norm(axis)) @ R


def yaw_of(rotvec: np.ndarray) -> float:
    """The part of a base-frame rotation that turns about vertical (swing-twist split)."""
    th = np.linalg.norm(rotvec)
    if th < 1e-12:
        return 0.0
    w, z = np.cos(th / 2), np.sin(th / 2) * rotvec[2] / th
    return 2 * np.arctan2(z, w)


def gripper_points(length: float | None) -> np.ndarray:
    """Points on the gripper in the flange (tool0) frame, used to keep it above the floor.

    A box around the 2F-85 that does not depend on which way its fingers open: the
    housing (15 cm square at the flange and 11 cm out) and the fingertip plane (12 cm
    square at `length`, wider than the jaws at full opening). None: the flange centre only.
    """
    if not length:
        return np.zeros((1, 3))
    pts = [(0.0, 0.0, 0.0), (0.0, 0.0, length)]
    for z, half in ((0.0, 0.075), (min(0.11, length), 0.075), (length, 0.06)):
        pts += [(sx * half, sy * half, z) for sx in (-1, 1) for sy in (-1, 1)]
    return np.array(pts)


@dataclass
class Limits:
    # Deliberately slow for first runs; raise only after testing.
    max_linear_speed: float = 0.10      # m/s
    max_angular_speed: float = 0.50     # rad/s
    max_joint_speed: float = 0.50       # rad/s
    joint_limit: float = 2 * np.pi - 0.05  # rad; UR5e wrists can reach +/-2pi
    elbow_limit: float = np.pi - 0.05   # rad
    box_half_extent: float = 0.40       # m, around the TCP pose at startup
    min_z: float | None = None          # m in the base frame; nothing on the arm or gripper goes lower
    gripper_length: float | None = None  # m from flange to fingertips; None: check the flange only
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
        self.tool_down = False
        self._aligning = False
        # Gripper points relative to the TCP, which sits tcp_z out along the flange's z.
        self._grip_pts = gripper_points(self.lim.gripper_length) - np.array([0.0, 0.0, self.map.tcp_offset[2, 3]])
        self._was_click = False

    def click(self, pressed: bool) -> bool | None:
        """Feed the stick-click state every tick. Each new press toggles tool-down mode.

        Returns the new mode when this press toggled it, else None.
        """
        toggled = pressed and not self._was_click
        self._was_click = pressed
        if not toggled:
            return None
        self.set_tool_down(not self.tool_down)
        return self.tool_down

    def set_tool_down(self, on: bool) -> None:
        """Keep the tool pointing straight down (normal to the table), or stop doing so.

        While on, the tool turns down by itself at the angular speed limit whenever the
        grip is released and the controller is live, then holds there. Hand tilt is ignored
        but hand twist about vertical still turns the tool. Toggling drops the clutch so the
        next update re-references both poses and the arm does not jump.
        """
        self.tool_down = bool(on)
        self.engaged = False
        self._aligning = False
        self._last_ctrl = None

    def resync(self, q_actual) -> None:
        """Adopt the measured joints and drop the clutch (used after any fault)."""
        self.q_cmd = np.array(q_actual, dtype=float)
        self.target = fk(self.q_cmd, self.map.tcp_offset)
        self.engaged = False
        self._aligning = False
        self._last_ctrl = None

    def update(self, ctrl: np.ndarray | None, clutch: bool, dt: float, head: np.ndarray | None = None) -> np.ndarray:
        if ctrl is None or not clutch:
            self.engaged = False
            self.latched_glitch = False if not clutch else self.latched_glitch
            self._last_ctrl = None
            if self.tool_down and ctrl is not None:
                return self._align(dt)
            self._aligning = False
            self.status = "tracking lost" if clutch else "idle"
            self.target = fk(self.q_cmd, self.map.tcp_offset)
            return self.q_cmd
        self._aligning = False

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
        # The speed limit can leave the orientation short of the one _desired clipped for.
        self.target[:3, 3] = self._clip_to_workspace(self.target[:3, 3], self.target[:3, :3], self._tcp0)
        self.q_cmd = self._ik_step(self.target, dt)
        return self.q_cmd

    def jog(self, command: np.ndarray, dt: float) -> np.ndarray:
        """Fine motion from jog_velocity's (forward, yaw rate, in/out, sideways).

        Forward runs level along the gripper's heading, sideways level along its x axis, yaw
        turns it about vertical through the TCP, and in/out runs along the tool's z axis.
        Same limits as hand motion.
        """
        self.engaged = False
        self.latched_glitch = False
        self._aligning = False
        self._last_ctrl = None
        self.status = "jogging"
        forward, yaw_rate, inout, sideways = (float(v) for v in command)
        out = self.target.copy()
        R = out[:3, :3]
        ahead, side = heading(R), heading(R, axis=0)
        if ahead is None and forward:
            forward = 0.0
            self.status = "no heading: gripper facing up or down along tool y"
        if side is None and sideways:
            sideways = 0.0
            self.status = "no sideways direction: tool x points up or down"
        velocity = ((forward * ahead if forward else np.zeros(3)) + (sideways * side if sideways else np.zeros(3))
                    + inout * R[:, 2])
        n = np.linalg.norm(velocity)
        if n > self.lim.max_linear_speed:
            velocity = velocity * self.lim.max_linear_speed / n
        start = self.target
        yaw_rate = float(np.clip(yaw_rate, -self.lim.max_angular_speed, self.lim.max_angular_speed))
        out[:3, :3] = matrix_from_rotvec(np.array([0.0, 0.0, yaw_rate * dt])) @ R
        if self.tool_down:
            out = self._limited_rotation(out, pointing_down(out[:3, :3]), dt)
        out[:3, 3] = self._clip_to_workspace(out[:3, 3] + velocity * dt, out[:3, :3], start)
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
        turn = self.map.rotation_scale * R @ rotvec_from_matrix(ctrl[:3, :3] @ self._ctrl0[:3, :3].T)
        if self.tool_down:
            # Tilt stays down; only the hand's twist about vertical turns the tool.
            out[:3, :3] = matrix_from_rotvec(np.array([0.0, 0.0, yaw_of(turn)])) @ pointing_down(self._tcp0[:3, :3])
        else:
            out[:3, :3] = matrix_from_rotvec(turn) @ self._tcp0[:3, :3]
        out[:3, 3] = self._clip_to_workspace(out[:3, 3], out[:3, :3], self._tcp0)
        return out

    def _align(self, dt: float) -> np.ndarray:
        """Turn toward pointing down in place, keeping the current yaw."""
        if not self._aligning:
            self._aligning = True
            self.target = fk(self.q_cmd, self.map.tcp_offset)
        goal = pointing_down(self.target[:3, :3])
        if np.linalg.norm(rotvec_from_matrix(goal @ self.target[:3, :3].T)) < 1e-4:
            self.status = "idle"
            return self.q_cmd
        self.status = "turning tool down"
        start = self.target
        self.target = self._limited_rotation(self.target, goal, dt)
        # Turning can swing the fingertips down; rise as needed to keep them off the floor.
        self.target[:3, 3] = self._clip_to_workspace(self.target[:3, 3], self.target[:3, :3], start)
        self.q_cmd = self._ik_step(self.target, dt)
        return self.q_cmd

    def _lowest(self, R: np.ndarray) -> float:
        """Height of the gripper's lowest point relative to the TCP, for orientation R."""
        return float(np.min(self._grip_pts @ R[2, :]))

    def _body_low(self, q: np.ndarray) -> float:
        """Lowest point of the arm (elbow and wrist joints) and gripper, for joints q."""
        frames = _chain(q)
        flange_pts = frames[6][:3, :3] @ (self._grip_pts + [0.0, 0.0, self.map.tcp_offset[2, 3]]).T
        joints = min(f[2, 3] for f in frames[2:6])
        return float(min(joints, np.min(flange_pts[2] + frames[6][2, 3])))

    def _clip_to_workspace(self, position: np.ndarray, R: np.ndarray, start: np.ndarray) -> np.ndarray:
        """Keep the TCP in the box, and every gripper point (for orientation R) above min_z."""
        half = self.lim.box_half_extent
        low, high = self.box_center - half, self.box_center + half
        if self.lim.min_z is not None:
            # Motion starting below the floor may stay there; it must not go lower.
            floor = min(self.lim.min_z, start[2, 3] + self._lowest(start[:3, :3]))
            low[2] = max(low[2], floor - self._lowest(R))
            high[2] = max(high[2], low[2])
        clipped = np.clip(position, low, high)
        if not np.array_equal(clipped, position):
            self.status = "workspace limit"
        return clipped

    def _limited_target(self, desired: np.ndarray, dt: float) -> np.ndarray:
        out = self.target.copy()
        dp = desired[:3, 3] - out[:3, 3]
        n = np.linalg.norm(dp)
        out[:3, 3] += dp * min(1.0, self.lim.max_linear_speed * dt / max(n, 1e-12))
        return self._limited_rotation(out, desired[:3, :3], dt)

    def _limited_rotation(self, pose: np.ndarray, R_goal: np.ndarray, dt: float) -> np.ndarray:
        out = pose.copy()
        dr = rotvec_from_matrix(R_goal @ out[:3, :3].T)
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
        if self.lim.min_z is not None:
            # The target keeps the gripper up; this catches the elbow and wrist too.
            floor = min(self.lim.min_z, self._body_low(self.q_cmd))
            if self._body_low(next_q) < floor - 0.003:
                self.status = "floor limit"
                self.target = fk(self.q_cmd, off)
                return self.q_cmd
        return next_q
