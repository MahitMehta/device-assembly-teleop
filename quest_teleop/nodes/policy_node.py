#!/usr/bin/env python3
"""Runs a trained DiT policy (wrist images, plus any state inputs it was trained with) on the
UR5e, through the same driver, controller, safety holds and motion limits as teleop_node.

Dry-run is the default: it plans and logs and publishes /policy/target_pose, but sends nothing to
the arm or gripper. This node never switches controllers.

--force-mode on (live): while the gripper holds the piece and the plan goes down, the arm is put in
UR force mode (force_mode_controller, activate it first) and gives way to contact along --fm-axes;
it leaves force mode when the gripper opens, the plan lifts, or the run stops for any reason.

Enter starts a run, Enter again pauses it (the arm holds where it is), B (while paused, live only)
moves the arm back to the home pose (--home-joints), Ctrl-C quits. Each run
starts the policy fresh (no frames from before the pause). A run pauses by itself after
--max-run-s, on any hold, and when the task looks done (--auto-stop-s: the gripper has closed and
reopened, and the policy has planned almost no motion since).

How the policy is driven (it was trained at 10 Hz):
  - every 0.1 s the newest wrist frame goes into the policy's frame history;
  - every exec_horizon (8) frames it plans a 16-step chunk of tool0 targets, relative to the tool0
    pose measured with that frame. Planning runs on its own thread; meanwhile the arm keeps
    following the previous chunk, whose later waypoints cover the gap, so it never stops to think;
  - the control loop (125 Hz) interpolates the chunk's waypoints and moves toward them through
    TeleopCore.follow: the same speed, workspace box, floor and joint limits as teleop.
"""
from __future__ import annotations

import argparse
import math
import select
import sys
import termios
import tty
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from std_srvs.srv import Trigger
from ur_dashboard_msgs.msg import RobotMode, SafetyMode
from ur_msgs.srv import SetForceMode
from controller_manager_msgs.srv import ListControllers

from quest_teleop.nodes.teleop_node import (CONTROLLER, JOINT_STALE_S, MAX_TRACKING_ERROR, ROBOT_RUNNING,
                                            SAFETY_NORMAL, preflight, set_payload, zero_ft)
from quest_teleop.utils.core import JOINT_NAMES, Limits, Mapping, TeleopCore, fk
from quest_teleop.utils.gripper import STROKE_M, GripperWorker, URCapGripper
from quest_teleop.utils import mount_check
from quest_teleop.utils.policy_runner import AutoStop, GripperHysteresis, WrenchFilter, preprocess, pose_source, tilt_of, ContactCompliance, SpiralSearch, contact_descent
from quest_teleop.utils.policy_runner import ForceModeGate, force_mode_override, force_mode_settings
from scipy.spatial.transform import Rotation

IMAGE_TOPIC = "/camera/camera/color/image_raw/compressed"
FORCE_MODE_CONTROLLER = "force_mode_controller"
FORCE_MODE_CALL_S = 1.0   # a start/stop_force_mode call slower than this pauses the run and is reported
# Jaw gap guesses for a gripper_width policy input when the gripper reports nothing (--no-gripper):
# in the demos the jaws read 17 mm closed on the clear piece and 84 mm open.
GUESS_CLOSED_M, GUESS_OPEN_M = 0.017, 0.084
# Median joints (rad) at the start of the 45 clear-piece demos (/data/teleop_demos/clear-piece): tool
# straight down, TCP ~35 cm above the base. Per joint the demos spread 4-7 deg (wrist 3: 16 deg).
DEMO_START_Q = (3.496, -1.307, -2.135, -1.305, 1.571, 0.247)
# Where B returns to (2026-11-18): the arm's pose then, flange at (-0.089, 0.423, 0.410) m in base,
# with the 4.0 deg tilt the driver measured taken out, so the driver's tool0 z points straight down.
HOME_Q = (2.0926, -1.3938, -1.9324, -1.3872, 1.5768, 0.4831)


class PolicyNode(Node):
    def __init__(self, args, runner):
        super().__init__("dit_policy")
        self.args, self.runner = args, runner
        self.lock = threading.Lock()
        self.q_actual, self.joint_time = None, 0.0
        self.tcp_pose, self.tcp_time = None, 0.0   # the driver's measured tool0 (the controller's calibrated kinematics)
        self._anchor_logged = False
        self.robot_mode = self.safety_mode = self.speed_scaling = None
        self.force = self.torque = self.force_baseline = None
        # Wrist force/torque as a policy input (`wrench` in the policy's state): filtered and zeroed
        # per run exactly as the training data was (WrenchFilter).
        self.uses_wrench = "wrench" in runner.state_keys
        self.wrench = WrenchFilter()
        self.wrench_in = None   # the last value given to the policy
        self.jpeg, self.image_time = None, 0.0
        self.image_seq = 0
        self.mount_ref = mount_check.load_reference()
        # The tool0 the policy plans from must be the one its dataset measured (see _anchor).
        self.anchor = args.anchor if args.anchor != "auto" else (
            "fk" if pose_source(args.policy_dir) == "nominal_fk" else "driver")
        # Mount turn the policy should cope with: what it was trained with, else ~1 deg (only the crop shifts).
        self.mount_tolerance = (args.mount_tolerance_deg if args.mount_tolerance_deg is not None
                                else runner.mount_rotation_deg or 1.0)
        self.core = None
        self.running = False          # a run is in progress (Enter toggles)
        self.run_start = 0.0
        self.fault = None
        self.chunk = None             # the Chunk being followed
        self.plans = 0
        self.plan_ms = []
        self.opening = self.min_opening = None   # policy's predicted jaw opening (m): now, and lowest this run
        self.homing = False           # moving back to --home-joints (B), between runs
        self.plan_tilt = None         # (deg, lean) the most tilted pose the latest plan asks for
        self.plan_side = None         # mm: the latest plan's largest sideways move (~0 when holding still)
        self.comp = ContactCompliance(speed=0.0 if args.comp_edge == "off" else args.comp_speed_mm_s / 1000,
                                      edge_dir=1.0 if args.comp_edge == "toward" else -1.0, contact_z=args.comp_contact_z,
                                      min_fz=args.comp_min_fz, min_r=args.comp_min_r_mm / 1000,
                                      max_offset=args.comp_max_mm / 1000, force_gain=args.comp_force_gain / 1000,
                                      force_deadband=args.comp_force_deadband, push_limit=args.comp_push_limit,
                                      edge_max_side=args.comp_edge_max_side)
        self._comp_chunk = None       # the chunk the offset was last rebased to
        self.search = SpiralSearch(pitch=args.search_pitch_mm / 1000, speed=args.search_speed_mm_s / 1000,
                                   max_r=args.search_max_mm / 1000, force=args.search_force,
                                   stall_s=args.search_stall_s, drop=args.search_drop_mm / 1000,
                                   min_fz=args.comp_min_fz, side_max=args.comp_force_deadband)
        self._open_since = None       # when the gripper last opened (the offset survives a brief flicker)
        self._push_z = None           # target height when pushing past --comp-push-limit began
        self._comp_state = ""
        # UR force mode while placing (--force-mode). fm_want is what the gate asks for; _fm_pump sends
        # it to the arm, one call at a time, and fm_active is what the arm confirmed.
        self.fm_selection, self.fm_wrench = force_mode_settings(args.fm_axes, args.fm_push)
        self.fm_gate = ForceModeGate(enter=args.fm_enter_mm / 1000, exit=args.fm_exit_mm / 1000,
                                     lookahead=args.fm_lookahead_s, min_fz=args.fm_min_fz)
        self.fm_live = args.live and args.force_mode == "on"
        self.fm_want = self.fm_active = False
        self.fm_busy = None           # (turning on?, sent at) while a call is in flight
        self.fm_retry_at = 0.0
        self.fm_start = None          # measured tool0 position when the arm confirmed force mode
        self.home_q = np.array(args.home_joints, dtype=float)
        self._resync = False
        self._recenter = False
        self.grip = GripperHysteresis(args.grip_close_below, args.grip_open_above, args.grip_min_closed_s)
        self.auto_stop = AutoStop(args.auto_stop_s, args.auto_stop_mm)
        self.last_log = 0.0
        self.mapping = Mapping(tcp_offset=np.eye(4))   # targets are tool0 poses, as in training
        self.limits = Limits(max_linear_speed=args.max_linear_speed, max_angular_speed=args.max_angular_speed,
                             max_joint_speed=args.max_joint_speed, box_half_extent=args.box_half_extent,
                             min_z=args.min_z, gripper_length=args.gripper_length)

        sensors, control = MutuallyExclusiveCallbackGroup(), MutuallyExclusiveCallbackGroup()
        self.create_subscription(JointState, "/joint_states", self._on_joints, qos_profile_sensor_data, callback_group=sensors)
        self.create_subscription(PoseStamped, "/tcp_pose_broadcaster/pose", self._on_tcp, qos_profile_sensor_data, callback_group=sensors)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(RobotMode, "/io_and_status_controller/robot_mode", lambda m: setattr(self, "robot_mode", m.mode), latched)
        self.create_subscription(SafetyMode, "/io_and_status_controller/safety_mode", lambda m: setattr(self, "safety_mode", m.mode), latched)
        self.create_subscription(Float64, "/speed_scaling_state_broadcaster/speed_scaling", lambda m: setattr(self, "speed_scaling", m.data), 10)
        self.create_subscription(WrenchStamped, "/force_torque_sensor_broadcaster/wrench", self._on_wrench, qos_profile_sensor_data, callback_group=sensors)
        self.create_subscription(CompressedImage, IMAGE_TOPIC, self._on_image, qos_profile_sensor_data, callback_group=sensors)

        self.cmd_pub = self.create_publisher(Float64MultiArray, f"/{CONTROLLER}/commands", 1) if args.live else None
        self.target_pub = self.create_publisher(PoseStamped, "/policy/target_pose", 10)
        self.status_pub = self.create_publisher(String, "/policy/status", 10)
        self.running_pub = self.create_publisher(Bool, "/policy/running", 10)
        self.grip_cmd_pub = self.create_publisher(Float64, "/policy/gripper_command", 10)
        self.opening_pub = self.create_publisher(Float64, "/policy/predicted_opening", 10)
        self.gripper = None
        if args.live and not args.no_gripper:
            self.gripper = GripperWorker(URCapGripper(args.gripper_ip, sid=args.gripper_sid),
                                         args.gripper_speed, args.gripper_force, self.get_logger())
        if self.fm_live:
            calls = ReentrantCallbackGroup()   # answered while the control loop keeps running
            self.fm_start_cli = self.create_client(SetForceMode, f"/{FORCE_MODE_CONTROLLER}/start_force_mode", callback_group=calls)
            self.fm_stop_cli = self.create_client(Trigger, f"/{FORCE_MODE_CONTROLLER}/stop_force_mode", callback_group=calls)
        self.timer = self.create_timer(1.0 / args.rate, self._tick, callback_group=control)
        self._stop = threading.Event()
        self._thinker = threading.Thread(target=self._policy_loop, daemon=True)

    # ---- inputs
    def _on_tcp(self, msg: PoseStamped) -> None:
        p, o = msg.pose.position, msg.pose.orientation
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
        T[:3, 3] = [p.x, p.y, p.z]
        with self.lock:
            self.tcp_pose, self.tcp_time = T, time.monotonic()

    def _anchor(self, q: np.ndarray) -> np.ndarray | None:
        """tool0 for the policy (plan anchor and tilt input). The demos' actions are
        inv(driver's tcp_pose) @ teleop's command, and the two come from different kinematics (the
        controller's calibrated model vs core.fk's nominal one, ~1.2 mm and 0.3 deg apart here), so a
        still arm reads as a constant small action. Anchored on fk(q) instead, that action moves the
        arm every plan: the steady creep and tilt. The anchor must match the dataset: 'driver' for those
        (the older ones, tcp_pose from the driver), 'fk' for datasets converted with --pose-source
        nominal_fk, where a still arm is a zero action on any UR5e. --anchor auto reads meta.json."""
        if self.anchor == "fk":
            return fk(q)
        with self.lock:
            T, age = self.tcp_pose, time.monotonic() - self.tcp_time
        if T is None or age > 0.1:
            return None
        if not self._anchor_logged:
            self._anchor_logged = True
            F = fk(q)
            d = F[:3, :3].T @ (T[:3, 3] - F[:3, 3]) * 1000
            ang = np.degrees(np.linalg.norm(Rotation.from_matrix(F[:3, :3].T @ T[:3, :3]).as_rotvec()))
            self.get_logger().info(f"Policy anchor: driver tcp_pose, {np.linalg.norm(d):.1f} mm ({d[0]:+.1f},{d[1]:+.1f},{d[2]:+.1f}) "
                                   f"and {ang:.2f} deg from the nominal fk(q)")
        return T

    def _on_joints(self, msg: JointState) -> None:
        index = {n: i for i, n in enumerate(msg.name)}
        if not all(n in index for n in JOINT_NAMES):
            return
        with self.lock:
            self.q_actual = np.array([msg.position[index[n]] for n in JOINT_NAMES])
            self.joint_time = time.monotonic()

    def _on_wrench(self, msg) -> None:
        f, t = msg.wrench.force, msg.wrench.torque
        self.force, self.torque = np.array([f.x, f.y, f.z]), np.array([t.x, t.y, t.z])
        if self.uses_wrench:
            self.wrench.add(time.monotonic(), [f.x, f.y, f.z, t.x, t.y, t.z])

    def _on_image(self, msg: CompressedImage) -> None:
        with self.lock:   # keep only the newest; decode at 10 Hz in the policy thread
            self.jpeg, self.image_time = bytes(msg.data), time.monotonic()
            self.image_seq += 1

    # ---- run control
    def toggle(self) -> None:
        if self.running:
            self._pause("paused (Enter)")
        elif self.fault:
            self.get_logger().warning(f"Not starting: {self.fault}. Fix it, then Enter again.")
            self.fault = None   # the next Enter tries again; the gate re-checks every tick
        elif self.core is None:
            self.get_logger().warning("Not starting: no joint states yet")
        elif not self._mount_ok():
            pass
        elif not self._zero_wrench():
            pass
        else:
            with self.lock:
                self.chunk = None
            self._resync = True        # done by the control loop, which owns self.core
            self._recenter = True      # the workspace box follows each run's start pose
            self.run_start = time.monotonic()
            self.opening = self.min_opening = None
            self.homing = False
            self.plan_tilt = None
            self.plan_side = None
            # Zeroed from the first reading of the run (arm still, touching nothing), unless holding the
            # piece, which may be pressing on the part: then keep the earlier zero.
            self.comp.reset(keep_zero=bool(self.grip.closed))
            self.search.reset()
            self._comp_chunk = None
            self._open_since = self._push_z = None
            self._comp_state = ""
            self.running = True
            self.get_logger().info(f"[{'LIVE' if self.args.live else 'DRY-RUN'}] run started (Enter pauses)")

    def home(self) -> None:
        """B: between runs, move the arm back to --home-joints (the home pose). B again stops."""
        log = self.get_logger()
        if self.homing:
            self.homing = False
            log.info("Return to start stopped (B); the arm holds here.")
        elif self.running:
            log.warning("Pause the run (Enter) before returning to the start pose.")
        elif not self.args.live:
            log.warning("Return to start needs --live (a dry run sends nothing to the arm).")
        elif self.core is None:
            log.warning("No joint states yet.")
        else:
            with self.lock:
                q = None if self.q_actual is None else self.q_actual.copy()
            reason = None if q is None else self._gate(q)
            if q is None or reason:
                log.warning(f"Not moving: {reason or 'no joint states'}.")
                return
            self.fault = None
            far = np.max(np.abs(self.home_q - q))
            if far > self.args.home_max_move:
                log.error(f"Not moving: a joint would turn {np.degrees(far):.0f} deg to reach the start pose "
                          f"(--home-max-move {np.degrees(self.args.home_max_move):.0f} deg). Jog it closer first.")
                return
            self._resync = True        # start from where the arm actually is
            self.homing = True
            log.info(f"Returning to the start pose at {self.args.home_speed:.2f} rad/s (B stops, Enter starts a run).")

    def _home_step(self, dt: float) -> np.ndarray:
        """One control tick toward --home-joints: all joints arrive together, the fastest at
        --home-speed, never lower than the floor allows (same check as TeleopCore)."""
        c, log = self.core, self.get_logger()
        d = self.home_q - c.q_cmd
        n = float(np.max(np.abs(d)))
        if n < 1e-4:
            self.homing = False
            log.info("At the start pose. Press Enter to start a run.")
            return c.q_cmd
        next_q = c.q_cmd + d * min(1.0, self.args.home_speed * dt / n)
        if c.lim.min_z is not None:
            floor = min(c.lim.min_z, c._body_low(c.q_cmd))
            if c._body_low(next_q) < floor - 0.003:
                self.homing = False
                log.error("Return to start stopped: the path would take the arm or gripper below the floor. Jog it up first.")
                return c.q_cmd
        c.q_cmd = next_q
        c.target = fk(next_q, c.map.tcp_offset)
        return next_q

    def _recent_frames(self, n: int = 5, timeout: float = 1.5) -> list[np.ndarray]:
        """The next n distinct camera frames, at the size the policy sees."""
        frames, last, t_end = [], None, time.monotonic() + timeout
        while len(frames) < n and time.monotonic() < t_end:
            with self.lock:
                jpeg, seq = self.jpeg, self.image_seq
            if jpeg is not None and seq != last:
                last = seq
                try:
                    frames.append(preprocess(jpeg, self.runner.hw))
                except ValueError:
                    pass
            time.sleep(0.02)
        return frames

    def _mount_ok(self) -> bool:
        """Before a run: is the camera mount turned within what the policy tolerates? Measured from
        the open left finger (utils/mount_check.py). A live run doesn't start if it's out of range;
        if it can't be measured (gripper closed, finger not found) it only warns."""
        if self.args.skip_mount_check or self.mount_ref is None:
            return True
        log = self.get_logger()
        state = self.gripper.state if self.gripper is not None else None
        if self.grip.closed or (state is not None and state[0] < 0.08):
            log.warning("camera mount: not checked (gripper closed; the reference is with it open)")
            return True
        frames = self._recent_frames()
        if len(frames) < 2:
            log.warning("camera mount: not checked (no camera frames)")
            return True
        ok, msg = mount_check.check(frames, self.mount_ref, self.mount_tolerance)
        if ok is None:
            log.warning(msg)
            return True
        if ok:
            log.info(msg)
            return True
        if self.args.live:
            log.error(f"Not starting: {msg}. Turn the mount back (or deploy a policy trained with a wider "
                      f"mount_rotation), or pass --skip-mount-check.")
            return False
        log.warning(f"{msg} (dry run: starting anyway)")
        return True

    def _zero_wrench(self) -> bool:
        """Before a run of a wrench policy: take the force/torque zero from the last 0.5 s, as each
        demo was zeroed over the still, untouched 0.5 s before it started. With the gripper closed
        (resuming mid-task, maybe holding the piece or touching the fixture) it keeps the previous
        zero instead. False (don't start) if there has never been a reading."""
        if not self.uses_wrench:
            return True
        log = self.get_logger()
        if self.grip.closed:
            if self.wrench.bias is not None:
                log.info("force/torque input: gripper closed, keeping the zero from the previous run")
                return True
            log.warning("force/torque input: zeroing with the gripper closed; start with it open and nothing touching")
        if not self.wrench.zero(time.monotonic()):
            log.error("Not starting: no /force_torque_sensor_broadcaster/wrench readings (the policy needs them)")
            return False
        b = self.wrench.bias
        log.info(f"force/torque input zeroed: offset {np.linalg.norm(b[:3]):.1f} N, {np.linalg.norm(b[3:]):.2f} N m "
                 f"(keep the arm still and touching nothing when starting)")
        return True

    def _pause(self, why: str) -> None:
        if self.running:
            self.get_logger().info(f"Run stopped: {why}{self._readout()}")
        self.running = False
        with self.lock:
            self.chunk = None
        self._resync = True            # done by the control loop, which owns self.core
        self.comp.reset(keep_zero=bool(self.grip.closed))   # log while paused: re-zero unless holding
        self.search.reset()
        if self.fm_gate.engaged and self.args.force_mode != "off":
            self.get_logger().info(f"force mode OFF (run stopped){'' if self.fm_live else ' [not sent]'}")
        self.fm_gate.reset()
        self.fm_want = False           # _fm_pump (every tick, holds included) ends it on the arm

    # ---- the policy, at 10 Hz on its own thread
    def _policy_loop(self) -> None:
        dt = self.runner.dt
        k, next_t, was_running = 0, time.monotonic(), False
        while not self._stop.is_set():
            next_t += dt
            time.sleep(max(0.0, next_t - time.monotonic()))
            if time.monotonic() - next_t > dt:   # fell behind (a slow plan): don't try to catch up
                next_t = time.monotonic()
            if not self.running:
                was_running = False
                continue
            if not was_running:          # a new run: no frames from before it
                self.runner.reset()
                self.auto_stop.reset()
                if self.grip.closed:       # left closed by the previous run: opening counts as a release
                    self.auto_stop.gripper(True)
                k, was_running = 0, True
            with self.lock:
                jpeg, age, q = self.jpeg, time.monotonic() - self.image_time, self.q_actual
            if jpeg is None or age > self.args.max_image_age:
                self._pause(f"camera image {'missing' if jpeg is None else f'{age * 1000:.0f} ms old'}")
                continue
            t_obs, tool0_fk = time.monotonic(), fk(q)
            tool0 = self._anchor(q)
            if tool0 is None:
                self._pause("no /tcp_pose_broadcaster/pose in the last 0.1 s (needed for --anchor driver)")
                continue
            comp_at_obs = self.comp.offset.copy()   # already in the measured pose (--compliance on)
            wrench = None
            if self.uses_wrench:
                wrench = self.wrench.value(t_obs)
                if wrench is None:
                    self._pause("no force/torque readings in the last 0.1 s")
                    continue
                self.wrench_in = wrench
            self.runner.observe(jpeg, tool0, self._jaw_gap(), wrench)   # decoded and resized in the policy process
            if k % self.runner.n_exec == 0:
                try:
                    chunk, ms = self.runner.plan(tool0, t_obs)
                except RuntimeError as e:
                    self._pause(str(e))
                    continue
                self.plan_ms = (self.plan_ms + [ms])[-50:]
                if not self.running:      # paused while planning: drop it
                    continue
                # What the policy asked for: the most tilted pose in the part of the plan that runs.
                self.plan_tilt = max((tilt_of(T[:3, :3]) for T in chunk.poses[:self.runner.n_exec]), key=lambda t: t[0])
                p = chunk.poses[:self.runner.n_exec, :2, 3] - tool0_fk[:2, 3]
                self.plan_side = float(np.max(np.linalg.norm(p, axis=1))) * 1000
                chunk.comp_at_obs = comp_at_obs
                with self.lock:
                    self.chunk = chunk
                    self.plans += 1
                self._publish_target(chunk.poses[0])
                if self.auto_stop.plan(chunk, self.runner.n_exec, time.monotonic()):
                    self._pause(f"task looks done: part released and the policy has planned under "
                                f"{self.args.auto_stop_mm:g} mm of motion for {self.args.auto_stop_s:g} s")
            k += 1

    def _jaw_gap(self) -> float:
        """Measured jaw gap (m) for policies with a gripper_width input: the gripper's own reading,
        as recorded in the demos (/teleop/gripper_position). With no reading (--no-gripper, or not
        connected yet) it guesses from the last command: the demos' gap when holding the piece, or open."""
        state = self.gripper.state if self.gripper is not None else None
        if state is not None:
            return float(state[0])
        return GUESS_CLOSED_M if self.grip.closed else GUESS_OPEN_M

    # ---- the control loop
    def _gate(self, q_actual) -> str | None:
        """Same holds as teleop_node._gate."""
        if time.monotonic() - self.joint_time > JOINT_STALE_S:
            return "joint states stale"
        if self.robot_mode != ROBOT_RUNNING:
            return f"robot mode {self.robot_mode}"
        if self.safety_mode != SAFETY_NORMAL:
            return f"safety mode {self.safety_mode}"
        if not self.speed_scaling:
            return "speed scaling is 0: program paused or not running"
        if np.max(np.abs(self.core.q_cmd - q_actual)) > MAX_TRACKING_ERROR:
            return "commanded joints too far from measured joints"
        if self.force is not None:
            if np.linalg.norm(self.force) > self.args.max_force_abs:
                return "absolute force limit exceeded"
            if self.force_baseline is not None and np.linalg.norm(self.force - self.force_baseline) > self.args.max_force_delta:
                return "contact force limit exceeded"
        return None

    def _tick(self) -> None:
        with self.lock:
            q_actual = None if self.q_actual is None else self.q_actual.copy()
            chunk = self.chunk
        if q_actual is None:
            return
        if self.core is None:
            self.core = TeleopCore(q_actual, self.mapping, self.limits)
            self.get_logger().info("Initialized from measured joints. Press Enter to start a run.")
        if self._resync:   # run started or stopped: continue from where the arm actually is
            self._resync = False
            self.core.resync(q_actual)
        if self._recenter:
            self._recenter = False
            self.core.box_center = self.core.target[:3, 3].copy()
        self._fm_pump()

        if self.args.live:
            reason = self._gate(q_actual)
            if reason:
                if reason != self.fault:
                    self.get_logger().error(f"HOLD: {reason}.{self._readout()}")
                self.fault = reason
                self._pause(reason)
                if self.homing:
                    self.homing = False
                    self.get_logger().error("Return to start stopped by the hold.")
                self.force_baseline = None
                self._status(f"hold: {reason}")
                return
        if self.running and time.monotonic() - self.run_start > self.args.max_run_s:
            self._pause(f"--max-run-s {self.args.max_run_s:.0f} reached")

        q_cmd = self.core.q_cmd
        if self.running and chunk is not None:
            if self.force_baseline is None and self.force is not None:
                self.force_baseline = self.force.copy()
            desired, opening = chunk.at(time.monotonic())
            desired = self._comply(desired, q_actual, chunk)
            desired = self._force_mode(desired, q_actual, chunk)
            q_cmd = self.core.follow(desired, 1.0 / self.args.rate)
            self.opening = float(opening)
            self.min_opening = self.opening if self.min_opening is None else min(self.min_opening, self.opening)
            self.opening_pub.publish(Float64(data=self.opening))
            changed = self.grip.update(opening, time.monotonic())
            if changed is not None:
                self.get_logger().info(f"gripper -> {'close' if changed else 'open'} (policy opening {opening * 1000:.0f} mm)")
                self.grip_cmd_pub.publish(Float64(data=0.0 if changed else STROKE_M))
                self.auto_stop.gripper(changed)
                if self.gripper is not None:
                    self.gripper.command(changed)
            status = self.core.status
            if status == "workspace limit":
                status += f" ({self.core.limit_detail})"
        elif self.running:
            status = "waiting for the first plan"
        elif self.homing:
            q_cmd = self._home_step(1.0 / self.args.rate)
            status = "returning to start" if self.homing else "paused"
        else:
            self.force_baseline = None
            status = "paused"
            if self.args.compliance == "log" and self.force is not None and self.torque is not None:
                # Press test: what the layer would do, with the arm still. Re-zeroed at each pause.
                self.comp.filter(np.r_[self.force, self.torque], 1.0 / self.args.rate)
                self.comp.step(fk(q_actual)[:3, :3], 1.0 / self.args.rate)
        if self.cmd_pub is not None:
            self.cmd_pub.publish(Float64MultiArray(data=[float(v) for v in q_cmd]))
        self._status(status)

    def _comply(self, desired: np.ndarray, q_actual: np.ndarray, chunk) -> np.ndarray:
        """--compliance: while holding the piece, slide the target along sideways contact force and
        away from the edge the wrist torque points to; descend slowly while touching; back off a little
        when pushing past --comp-push-limit. 'log' computes and logs the same but leaves the target alone."""
        mode, dt = self.args.compliance, 1.0 / self.args.rate
        if mode == "off" or self.force is None or self.torque is None:
            return desired
        self.comp.filter(np.r_[self.force, self.torque], dt)
        now = time.monotonic()
        if mode == "on" and chunk is not self._comp_chunk:
            self._comp_chunk = chunk
            self.comp.rebase(getattr(chunk, "comp_at_obs", np.zeros(2)))
        if self.grip.closed:
            self._open_since = None
        elif self._open_since is None:
            self._open_since = now
        # A brief open (the policy flickering) keeps the offset; a real release drops it.
        keep = self.grip.closed or (self._open_since is not None and now - self._open_since < self.args.comp_keep_s)
        if not keep:
            self.comp.release()
            self.search.reset()
            self._push_z, self._comp_state = None, ""
            return desired
        measured = fk(q_actual)
        if self.grip.closed and not self._fm_on():   # in force mode the arm itself gives way: no slide or search
            extra = None
            if self.args.search:
                w = self.comp.w
                extra = self.search.update(now, dt, w[2], float(np.linalg.norm(w[:2])), measured[2, 3])
            self.comp.step(measured[:3, :3], dt, extra)
        if mode != "on":
            return desired
        out = desired.copy()
        out[:2, 3] += self.comp.offset
        if self.search.state == "searching":
            self._push_z = None
            out[2, 3] = max(out[2, 3], self.search.z_target(self.core.target[2, 3], self.comp.w[2], dt))
            self._comp_state = f"SEARCH r={self.search.r * 1000:.1f}mm"
        else:
            out[2, 3], self._push_z, self._comp_state = contact_descent(
                out[2, 3], self.core.target[2, 3], self.comp.w[2], self.comp.min_fz, self.comp.pushing, self._push_z, dt,
                descent=self.args.comp_descent_mm_s / 1000, backoff=self.args.comp_backoff_mm / 1000)
            if self.search.state in ("found", "failed"):
                self._comp_state = (self._comp_state + f" search-{self.search.state}").strip()
        return out

    # ---- UR force mode while placing (--force-mode)
    def _fm_on(self) -> bool:
        """The arm is, or may be, in force mode (asked for, confirmed, or a call in flight)."""
        return self.fm_live and (self.fm_want or self.fm_active or self.fm_busy is not None)

    def _force_mode(self, desired: np.ndarray, q_actual: np.ndarray, chunk) -> np.ndarray:
        """Enter force mode when holding the piece and the plan descends, leave when the gripper opens
        or the plan lifts (ForceModeGate). While in it, command the measured pose along the compliant
        axes (the arm ignores the command there). 'log' only reports the switches."""
        if self.args.force_mode == "off":
            return desired
        if self.args.compliance == "off" and self.force is not None and self.torque is not None:
            self.comp.filter(np.r_[self.force, self.torque], 1.0 / self.args.rate)   # zeroed F_z for --fm-min-fz
        fz = self.comp.w[2] if self.comp.zero is not None else 0.0
        now = time.monotonic()
        changed = self.fm_gate.update(bool(self.grip.closed), self.fm_gate.planned_dz(chunk, now), fz)
        if changed is not None:
            self.get_logger().info(f"force mode {'ON' if changed else 'OFF'} ({self.fm_gate.reason})"
                                   f"{'' if self.fm_live else ' [not sent: --force-mode log or dry run]'}")
            self.fm_want = changed
        if not self._fm_on():
            return desired
        measured = fk(q_actual)
        if self.fm_start is not None:
            moved = np.linalg.norm((measured[:3, 3] - self.fm_start)[self.fm_selection[:3]])
            if moved > self.args.fm_max_mm / 1000:
                self._pause(f"force mode moved the tool {moved * 1000:.1f} mm (--fm-max-mm {self.args.fm_max_mm:g})")
                return desired
        return force_mode_override(desired, measured, self.fm_selection)

    def _fm_request(self) -> SetForceMode.Request:
        a, req = self.args, SetForceMode.Request()
        req.task_frame.header.frame_id = "base"   # the arm's base: x, y horizontal, z up
        req.task_frame.pose.orientation.w = 1.0
        (req.selection_vector_x, req.selection_vector_y, req.selection_vector_z,
         req.selection_vector_rx, req.selection_vector_ry, req.selection_vector_rz) = (bool(v) for v in self.fm_selection)
        w = self.fm_wrench
        req.wrench.force.x, req.wrench.force.y, req.wrench.force.z = (float(v) for v in w[:3])
        req.wrench.torque.x, req.wrench.torque.y, req.wrench.torque.z = (float(v) for v in w[3:])
        req.type = SetForceMode.Request.NO_TRANSFORM
        req.speed_limits.linear.x = req.speed_limits.linear.y = req.speed_limits.linear.z = float(a.fm_speed)
        req.speed_limits.angular.x = req.speed_limits.angular.y = req.speed_limits.angular.z = float(a.fm_rot_speed)
        req.damping_factor = float(a.fm_damping)
        req.gain_scaling = float(a.fm_gain_scaling)
        return req

    def _fm_pump(self) -> None:
        """Every tick (holds too): send the arm what fm_want says, one call at a time."""
        if not self.fm_live:
            return
        now = time.monotonic()
        if self.fm_busy is not None:
            on, sent = self.fm_busy
            if now - sent > FORCE_MODE_CALL_S and now >= self.fm_retry_at:
                self.fm_retry_at = now + 1.0
                self.get_logger().error(f"{'start' if on else 'stop'}_force_mode has not answered for {now - sent:.1f} s")
                self._pause("force mode call not answered")
            return
        if self.fm_want == self.fm_active or now < self.fm_retry_at:
            return
        on = self.fm_want
        fut = (self.fm_start_cli.call_async(self._fm_request()) if on
               else self.fm_stop_cli.call_async(Trigger.Request()))
        self.fm_busy = (on, now)
        fut.add_done_callback(self._fm_done)

    def _fm_done(self, fut) -> None:
        on, sent = self.fm_busy
        try:
            ok = bool(fut.result().success)
        except Exception as e:   # noqa: BLE001 - reported, and the pump retries
            ok = False
            self.get_logger().error(f"{'start' if on else 'stop'}_force_mode failed: {e}")
        ms = (time.monotonic() - sent) * 1000
        if ok:
            if on:
                with self.lock:
                    q = self.q_actual
                self.fm_start = fk(q)[:3, 3].copy()
            else:
                self.fm_start = None
            self.fm_active = on
            self.get_logger().info(f"arm {'in' if on else 'out of'} force mode ({ms:.0f} ms)")
        else:
            self.fm_retry_at = time.monotonic() + 0.5
            if on:
                self.get_logger().error(f"Could not start force mode ({ms:.0f} ms); is {FORCE_MODE_CONTROLLER} active?")
                self._pause("could not start force mode")
            else:
                self.get_logger().error(f"Could not stop force mode ({ms:.0f} ms); retrying. Stop the program on the pendant if it persists.")
        self.fm_busy = None

    def stop_force_mode(self, executor, timeout: float = 2.0) -> None:
        """On exit: end force mode on the arm, whatever this node thinks its state is."""
        if not self.fm_live or not rclpy.ok():
            return
        fut = self.fm_stop_cli.call_async(Trigger.Request())
        executor.spin_until_future_complete(fut, timeout_sec=timeout)
        ok = fut.done() and fut.result() is not None and fut.result().success
        print("Force mode stopped." if ok else "Could NOT confirm force mode stopped: stop the program on the pendant.", flush=True)

    # ---- outputs
    def _status(self, status: str) -> None:
        self.status_pub.publish(String(data=status))
        self.running_pub.publish(Bool(data=self.running))
        now = time.monotonic()
        if now - self.last_log > 1.0:
            self.last_log = now
            mode = "LIVE" if self.args.live else "DRY-RUN"
            extra = ""
            if self.running:
                extra = f" t={now - self.run_start:.0f}s plans={self.plans}"
                if self.plan_ms:
                    extra += f" plan={np.median(self.plan_ms):.0f}ms"
                with self.lock:
                    age = max(0.0, time.monotonic() - self.image_time)
                extra += f" image_age={age * 1000:.0f}ms"
                if self.plan_tilt is not None:
                    with self.lock:
                        q = self.q_actual
                    deg, lean = tilt_of(fk(q)[:3, :3])
                    pdeg, plean = self.plan_tilt
                    extra += (f" tilt={deg:.1f}deg lean=({lean[0]:+.2f},{lean[1]:+.2f})"
                              f" planned={pdeg:.1f}deg lean=({plean[0]:+.2f},{plean[1]:+.2f})")
                if self.args.compliance != "off" and self.grip.closed:
                    extra += self._comp_text()
                if self.args.force_mode != "off" and (self.grip.closed or self.fm_gate.engaged):
                    extra += f" fm[{self.args.force_mode}]={'ON' if self.fm_gate.engaged else 'off'}"
                    if self.fm_live:
                        extra += f" arm={'in' if self.fm_active else 'out'}{' (call pending)' if self.fm_busy else ''}"
                    if self.fm_start is not None:
                        with self.lock:
                            q = self.q_actual
                        d = (fk(q)[:3, 3] - self.fm_start) * 1000
                        extra += f" fm_moved=({d[0]:+.1f},{d[1]:+.1f},{d[2]:+.1f})mm"
                if self.plan_side is not None:
                    extra += f" side={self.plan_side:.1f}mm"
                if self.opening is not None:
                    extra += (f" opening={self.opening * 1000:.0f}mm (min {self.min_opening * 1000:.0f},"
                              f" closes <{self.args.grip_close_below * 1000:.0f})")
                if self.wrench_in is not None:
                    extra += f" ft_in={np.linalg.norm(self.wrench_in[:3]):.1f}N"
            if not self.running and self.args.compliance == "log":
                extra += self._comp_text()
            self.get_logger().info(f"[{mode}] {status}{extra} |{self._readout()}")

    def _comp_text(self) -> str:
        """The compliance layer in robot terms: fwd = the robot's forward (Mapping.robot_forward_yaw),
        left, up. push = filtered, zeroed force on the tool; slide = where it is moving the target."""
        c = self.comp
        with self.lock:
            q = self.q_actual
        R = fk(q)[:3, :3]
        yaw = self.mapping.robot_forward_yaw
        fwd, left = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
        Fb = R @ c.w[:3]
        out = (f" comp[{self.args.compliance}] push fwd={Fb[:2] @ fwd:+.1f} left={Fb[:2] @ left:+.1f} up={Fb[2]:+.1f}N"
               f" (tool F=({c.w[0]:+.1f},{c.w[1]:+.1f},{c.w[2]:+.1f})N tau=({c.w[3]:+.2f},{c.w[4]:+.2f},{c.w[5]:+.2f})Nm)")
        if c.contact is not None:
            rb = R[:2, :2] @ c.contact
            out += f" contact fwd={rb @ fwd * 1000:+.0f} left={rb @ left * 1000:+.0f}mm"
        out += (f" slide[{c.path or '-'}] fwd={c.v_base @ fwd * 1000:+.1f} left={c.v_base @ left * 1000:+.1f}mm/s"
                f" slid fwd={c.total @ fwd * 1000:+.1f} left={c.total @ left * 1000:+.1f}mm")
        if self._comp_state:
            out += f" {self._comp_state}"
        elif c.pushing:
            out += " PUSHING"
        return out

    def _publish_target(self, T: np.ndarray) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in T[:3, 3])
        x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = float(x), float(y), float(z), float(w)
        self.target_pub.publish(msg)

    def _readout(self) -> str:
        out = ""
        if self.force is not None:
            out += f" force={np.linalg.norm(self.force):.1f}N"
            if self.force_baseline is not None:
                out += f" change={np.linalg.norm(self.force - self.force_baseline):.1f}N"
        if self.grip.closed is not None:
            out += f" gripper={'closed' if self.grip.closed else 'open'}"
        state = self.gripper.state if self.gripper is not None else None
        if state is not None:
            out += f" jaws at {state[0] * 1000:.0f}mm{' on object' if state[1] else ''}"
        return out

    def start_thinking(self) -> None:
        self._thinker.start()

    def stop_thinking(self) -> None:
        self._stop.set()


def force_mode_preflight(node: PolicyNode, timeout: float = 5.0) -> str | None:
    """--force-mode on: force_mode_controller must be active (the driver loads it inactive) and its
    services reachable. Like preflight, this never switches controllers."""
    lc = node.create_client(ListControllers, "/controller_manager/list_controllers")
    if not lc.wait_for_service(timeout_sec=timeout):
        return "controller_manager is not reachable"
    fut = lc.call_async(ListControllers.Request())
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
    if not fut.done():
        return "list_controllers timed out"
    state = {c.name: c.state for c in fut.result().controller}.get(FORCE_MODE_CONTROLLER, "missing")
    if state != "active":
        return (f"{FORCE_MODE_CONTROLLER} is {state}. Activate it (it runs alongside forward_position_controller): "
                f"docker compose run --rm ros ros2 control switch_controllers --activate {FORCE_MODE_CONTROLLER}")
    for cli in (node.fm_start_cli, node.fm_stop_cli):
        if not cli.wait_for_service(timeout_sec=timeout):
            return f"{cli.srv_name} is not reachable"
    selected = ",".join(a for a, s in zip(("x", "y", "z", "rx", "ry", "rz"), node.fm_selection) if s)
    node.get_logger().info(f"Force mode ready: compliant in {selected} (base frame)"
                           + (f", pushing {node.args.fm_push:g} N down" if node.fm_selection[2] else "")
                           + f", up to {node.args.fm_speed * 1000:.0f} mm/s")
    return None


def keyboard(node: PolicyNode, stop: threading.Event) -> None:
    """Enter toggles the run; B returns to the start pose. Single keys on a terminal (main() puts it in
    cbreak mode), else whole lines ("b" + Enter). Polls stdin so Ctrl-C still reaches the main thread."""
    tty_in = sys.stdin.isatty()
    while not stop.is_set():
        r, _, _ = select.select([sys.stdin], [], [], 0.2)
        if not r:
            continue
        key = sys.stdin.read(1) if tty_in else sys.stdin.readline()
        if not key:   # end of input
            return
        if key.strip().lower() == "b":
            node.home()
        elif key in ("\n", "\r") or not tty_in:
            node.toggle()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("policy_dir", help="Deployed policy directory (scripts/deploy_policy.sh), e.g. /data/policies/real_v1")
    p.add_argument("--checkpoint", default="final.pt", help="File in policy_dir (e.g. step_0080000.pt)")
    p.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    p.add_argument("--live", action="store_true", help="Actually command the arm and gripper")
    p.add_argument("--no-ft-zero", action="store_true")
    p.add_argument("--payload-mass", type=float, default=1.080)
    p.add_argument("--payload-cog", type=float, nargs=3, default=[-0.009, -0.009, 0.066], metavar=("X", "Y", "Z"))
    p.add_argument("--rate", type=float, default=125.0)
    p.add_argument("--max-run-s", type=float, default=60.0, help="A run pauses by itself after this long (s)")
    p.add_argument("--auto-stop-s", type=float, default=2.0,
                   help="After the gripper has closed and reopened, pause once the policy has planned under --auto-stop-mm of motion for this long (s); 0 disables")
    p.add_argument("--auto-stop-mm", type=float, default=3.0, help="mm: see --auto-stop-s")
    p.add_argument("--max-image-age", type=float, default=0.3, help="s; an older newest frame pauses the run")
    # Slower than teleop's defaults for first runs; the demos were teleoperated at up to 0.20 m/s.
    p.add_argument("--max-linear-speed", type=float, default=0.10, help="m/s")
    p.add_argument("--max-angular-speed", type=float, default=0.5, help="rad/s")
    p.add_argument("--max-joint-speed", type=float, default=0.5, help="rad/s")
    p.add_argument("--max-force-delta", type=float, default=30.0, help="N above the value when the run started")
    p.add_argument("--max-force-abs", type=float, default=60.0, help="N")
    p.add_argument("--box-half-extent", type=float, default=0.35, help="m the tool may move from where the run started, per axis (the demos went up to 30 cm)")
    p.add_argument("--min-z", type=float, default=0.01)
    p.add_argument("--anchor", choices=("auto", "driver", "fk"), default="auto",
                   help="tool0 the policy plans from; it must match how its dataset measured tool0. auto (default): "
                        "from the policy's meta.json pose_source: nominal_fk -> fk (core.fk of the joints), else "
                        "driver (/tcp_pose_broadcaster/pose, the older datasets; fk there creeps ~1.2 mm/plan)")
    p.add_argument("--compliance", choices=("off", "log", "on"), default="off",
                   help="While holding the piece: slide away from where the wrist torque says it touches, and stop "
                        "descending past --comp-push-limit. 'log' only computes and logs it")
    p.add_argument("--comp-edge", choices=("off", "toward", "away"), default="off",
                   help="On a near-vertical push, slide toward (or away from) the edge the wrist torque says is "
                        "touching. 'toward': front edge touching -> forward. Off by default: unreliable whenever "
                        "there is real sideways force (it is gated off then by --comp-edge-max-side)")
    p.add_argument("--comp-speed-mm-s", type=float, default=2.0, help="mm/s of that edge slide")
    p.add_argument("--comp-contact-z", type=float, default=0.0,
                   help="m from the flange to where sideways contact force acts, for the contact estimate; 0 (default) "
                        "ignores the sideways part, which the press test showed is mostly sensor error")
    p.add_argument("--comp-min-fz", type=float, default=3.0, help="N along tool z before it counts as touching")
    p.add_argument("--comp-min-r-mm", type=float, default=8.0,
                   help="contact estimate this close to the tool axis (mm) counts as centred (it reads ~3x the real offset)")
    p.add_argument("--comp-max-mm", type=float, default=6.0,
                   help="largest total sideways correction per held piece, sliding and search together (mm)")
    p.add_argument("--search", action=argparse.BooleanOptionalAction, default=True,
                   help="With --compliance: spiral to find the hole when the piece sits on the rim (pressing, "
                        "no sideways force, no progress)")
    p.add_argument("--search-pitch-mm", type=float, default=1.0, help="spiral radius growth per turn (mm)")
    p.add_argument("--search-speed-mm-s", type=float, default=5.0, help="speed along the spiral (mm/s)")
    p.add_argument("--search-max-mm", type=float, default=4.0, help="largest spiral radius (mm)")
    p.add_argument("--search-force", type=float, default=6.0, help="push held while searching (N)")
    p.add_argument("--search-stall-s", type=float, default=0.5, help="s pressing without progress before it searches")
    p.add_argument("--search-drop-mm", type=float, default=1.0, help="drop below the stall height that counts as found (mm)")
    p.add_argument("--comp-edge-max-side", type=float, default=0.3,
                   help="--comp-edge acts only while sideways force is under this fraction of the push")
    p.add_argument("--comp-force-gain", type=float, default=1.0,
                   help="mm/s per N of sideways contact force beyond the dead band (0 disables)")
    p.add_argument("--comp-force-deadband", type=float, default=2.0, help="N of sideways force ignored")
    p.add_argument("--comp-push-limit", type=float, default=12.0, help="N along tool z past which it backs off")
    p.add_argument("--comp-backoff-mm", type=float, default=0.5, help="how far it rises when pushing past the limit")
    p.add_argument("--comp-descent-mm-s", type=float, default=2.0, help="fastest descent while touching (mm/s)")
    p.add_argument("--comp-keep-s", type=float, default=1.0,
                   help="s the gripper may open (policy flicker) before the sideways offset is dropped")
    p.add_argument("--force-mode", choices=("off", "log", "on"), default="off",
                   help="UR force mode while placing: entered when the gripper holds the piece and the plan descends, "
                        "left when it opens or the plan lifts. 'on' (with --live) needs force_mode_controller active; "
                        "'log' only reports when it would switch")
    p.add_argument("--fm-axes", default="x,y",
                   help="compliant axes in the base frame, some of x,y,z,rx,ry,rz. x,y: slides along contact, "
                        "follows the plan in z. With z: pushes down with --fm-push instead of following the plan's height")
    p.add_argument("--fm-push", type=float, default=5.0, help="N pushed down when z is compliant")
    p.add_argument("--fm-speed", type=float, default=0.02, help="m/s: fastest motion along compliant axes")
    p.add_argument("--fm-rot-speed", type=float, default=0.1, help="rad/s: fastest motion about compliant axes")
    p.add_argument("--fm-damping", type=float, default=0.025, help="UR force mode damping, 0-1 (1 stops quickly when the force goes)")
    p.add_argument("--fm-gain-scaling", type=float, default=0.5, help="UR force mode gain scaling, 0-2 (over 1 can be unstable on hard contact)")
    p.add_argument("--fm-enter-mm", type=float, default=1.0, help="enter when the plan goes down more than this over --fm-lookahead-s")
    p.add_argument("--fm-exit-mm", type=float, default=1.0, help="leave when the plan goes up more than this over --fm-lookahead-s")
    p.add_argument("--fm-lookahead-s", type=float, default=0.3, help="s of the plan looked at to tell descending from lifting")
    p.add_argument("--fm-min-fz", type=float, default=3.0,
                   help="N: also require this much (zeroed) push along tool z before entering, so the plan can still "
                        "steer sideways in free air; 0 enters on descent alone")
    p.add_argument("--fm-max-mm", type=float, default=8.0,
                   help="mm the tool may move along compliant axes in one stretch of force mode before the run pauses")
    p.add_argument("--home-joints", type=float, nargs=6, default=list(HOME_Q), metavar="Q",
                   help="Joints (rad) that B returns to between runs; default: HOME_Q, tool straight down. "
                        f"The clear-piece demos' median start: {' '.join(map(str, DEMO_START_Q))}")
    p.add_argument("--home-speed", type=float, default=0.3, help="rad/s of the fastest joint when returning to start")
    p.add_argument("--home-max-move", type=float, default=1.5,
                   help="rad; B refuses if any joint would turn more than this (e.g. the pan wrapped the other way)")
    p.add_argument("--gripper-length", type=float, default=0.18)
    p.add_argument("--grip-close-below", type=float, default=0.030, help="Predicted opening (m) that closes the gripper")
    p.add_argument("--grip-open-above", type=float, default=0.055, help="...and that opens it again")
    p.add_argument("--grip-min-closed-s", type=float, default=0.25,
                   help="once closed, the gripper stays closed at least this long (s), even if the policy's "
                        "predicted opening jumps straight back (e.g. to 85 mm)")
    p.add_argument("--gripper-ip", default="10.168.4.251")
    p.add_argument("--gripper-sid", type=int, default=3)
    p.add_argument("--gripper-speed", type=int, default=50)
    p.add_argument("--gripper-force", type=int, default=0)
    p.add_argument("--no-gripper", action="store_true")
    p.add_argument("--mount-tolerance-deg", type=float, default=None,
                   help="Largest camera-mount turn (deg, measured from the gripper finger) a live run may start with; "
                        "default: the policy's trained mount_rotation, else 1")
    p.add_argument("--skip-mount-check", action="store_true", help="Start runs without checking the camera mount")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from quest_teleop.utils.policy_runner import PlannerProcess
    # torch runs in its own process, so planning never waits on this one's interpreter lock.
    runner = PlannerProcess(args.policy_dir, args.checkpoint, args.device)
    i = runner.info
    print(f"Policy {args.policy_dir}/{args.checkpoint} on {i['device']}: images {runner.hw}, plans {runner.horizon} steps, "
          f"runs {runner.n_exec} before replanning; warm-up plan {i['warmup_ms']:.0f} ms", flush=True)
    if args.skip_mount_check:
        print("Camera mount check: off (--skip-mount-check)", flush=True)
    elif mount_check.load_reference() is None:
        print(f"Camera mount check: off (no {mount_check.REFERENCE_PATH})", flush=True)
    else:
        tol = args.mount_tolerance_deg if args.mount_tolerance_deg is not None else runner.mount_rotation_deg or 1.0
        print(f"Camera mount check before each run: within +/-{tol:g} deg of the demos "
              f"({'trained with' if runner.mount_rotation_deg else 'no'} mount-rotation augmentation)", flush=True)
    from quest_teleop.utils.policy_runner import pose_source
    src = pose_source(args.policy_dir)
    anchor = args.anchor if args.anchor != "auto" else ("fk" if src == "nominal_fk" else "driver")
    print(f"Policy anchor: {anchor} (dataset pose_source: {src}{'' if args.anchor == 'auto' else ', set by --anchor'})", flush=True)
    print(f"Inputs: wrist images{''.join(', ' + k for k in runner.state_keys)}"
          + (" (jaw gap guessed from the commands: no gripper connection without --live, or --no-gripper)"
             if "gripper_width" in runner.state_keys and (args.no_gripper or not args.live) else ""),
          flush=True)

    if args.force_mode == "on" and args.live:
        # Ctrl-C as a plain KeyboardInterrupt, so ROS is still up to take the arm out of force mode.
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.SIGTERM)
    else:
        rclpy.init()
    node = PolicyNode(args, runner)
    node.timer.cancel()
    if args.live:
        error = (preflight(node)
                 or (force_mode_preflight(node) if args.force_mode == "on" else None)
                 or (None if args.payload_mass <= 0 else set_payload(node, args.payload_mass, args.payload_cog))
                 or (None if args.no_ft_zero else zero_ft(node)))
        if error:
            node.get_logger().error(f"Refusing to go live: {error}")
            node.destroy_node(); rclpy.shutdown()
            return 1
    if node.gripper is not None:
        node.gripper.start()
    node.timer.reset()
    node.start_thinking()
    stop = threading.Event()
    term = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
    if term is not None:
        tty.setcbreak(sys.stdin)   # single keys, no Enter needed for B; Ctrl-C still works
    threading.Thread(target=keyboard, args=(node, stop), daemon=True).start()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception:
        if rclpy.ok():   # not a SIGTERM-triggered shutdown: a real error
            raise
    finally:
        stop.set()
        if term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, term)
        node.stop_thinking()
        node.stop_force_mode(executor)
        runner.close()
        if node.gripper is not None:
            node.gripper.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
