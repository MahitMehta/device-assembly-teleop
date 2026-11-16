#!/usr/bin/env python3
"""Runs a trained DiT policy (wrist images, plus any state inputs it was trained with) on the
UR5e, through the same driver, controller, safety holds and motion limits as teleop_node.

Dry-run is the default: it plans and logs and publishes /policy/target_pose, but sends nothing to
the arm or gripper. This node never switches controllers.

Enter starts a run, Enter again pauses it (the arm holds where it is), B (while paused, live only)
moves the arm back to the demos' start pose (--home-joints), Ctrl-C quits. --tool-down keeps the
planned poses pointing straight down. Each run
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
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from ur_dashboard_msgs.msg import RobotMode, SafetyMode

from quest_teleop.nodes.teleop_node import (CONTROLLER, JOINT_STALE_S, MAX_TRACKING_ERROR, ROBOT_RUNNING,
                                            SAFETY_NORMAL, preflight, set_payload, zero_ft)
from quest_teleop.utils.core import JOINT_NAMES, Limits, Mapping, TeleopCore, fk
from quest_teleop.utils.gripper import STROKE_M, GripperWorker, URCapGripper
from quest_teleop.utils import mount_check
from quest_teleop.utils.policy_runner import AutoStop, GripperHysteresis, WrenchFilter, preprocess, tilt_of, tool_down, xy_deadband
from scipy.spatial.transform import Rotation

IMAGE_TOPIC = "/camera/camera/color/image_raw/compressed"
# Jaw gap guesses for a gripper_width policy input when the gripper reports nothing (--no-gripper):
# in the demos the jaws read 17 mm closed on the clear piece and 84 mm open.
GUESS_CLOSED_M, GUESS_OPEN_M = 0.017, 0.084
# Median joints (rad) at the start of the 45 clear-piece demos (/data/teleop_demos/clear-piece): tool
# straight down, TCP ~35 cm above the base. Per joint the demos spread 4-7 deg (wrist 3: 16 deg).
DEMO_START_Q = (3.496, -1.307, -2.135, -1.305, 1.571, 0.247)


class PolicyNode(Node):
    def __init__(self, args, runner):
        super().__init__("dit_policy")
        self.args, self.runner = args, runner
        self.lock = threading.Lock()
        self.q_actual, self.joint_time = None, 0.0
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
        self.plan_tilt = None         # (deg, lean) the latest plan asked for, before --tool-down
        self.plan_side = None         # (mm, dropped) the latest plan's sideways move, and if --hold-xy-deadband-mm dropped it
        self.home_q = np.array(args.home_joints, dtype=float)
        self._resync = False
        self._recenter = False
        self.grip = GripperHysteresis(args.grip_close_below, args.grip_open_above)
        self.auto_stop = AutoStop(args.auto_stop_s, args.auto_stop_mm)
        self.last_log = 0.0
        self.mapping = Mapping(tcp_offset=np.eye(4))   # targets are tool0 poses, as in training
        self.limits = Limits(max_linear_speed=args.max_linear_speed, max_angular_speed=args.max_angular_speed,
                             max_joint_speed=args.max_joint_speed, box_half_extent=args.box_half_extent,
                             min_z=args.min_z, gripper_length=args.gripper_length)

        sensors, control = MutuallyExclusiveCallbackGroup(), MutuallyExclusiveCallbackGroup()
        self.create_subscription(JointState, "/joint_states", self._on_joints, qos_profile_sensor_data, callback_group=sensors)
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
        self.timer = self.create_timer(1.0 / args.rate, self._tick, callback_group=control)
        self._stop = threading.Event()
        self._thinker = threading.Thread(target=self._policy_loop, daemon=True)

    # ---- inputs
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
            self.running = True
            self.get_logger().info(f"[{'LIVE' if self.args.live else 'DRY-RUN'}] run started (Enter pauses)")

    def home(self) -> None:
        """B: between runs, move the arm back to --home-joints (the demos' start pose). B again stops."""
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
            t_obs, tool0 = time.monotonic(), fk(q)
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
                if self.args.tool_down:
                    chunk = tool_down(chunk)
                if self.grip.closed:      # holding the piece: no creeping sideways on the way down
                    chunk, side, dropped = xy_deadband(chunk, tool0, self.runner.n_exec, self.args.hold_xy_deadband_mm)
                    self.plan_side = (side, dropped)
                else:
                    self.plan_side = None
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
            q_cmd = self.core.follow(desired, 1.0 / self.args.rate)
            self.opening = float(opening)
            self.min_opening = self.opening if self.min_opening is None else min(self.min_opening, self.opening)
            self.opening_pub.publish(Float64(data=self.opening))
            changed = self.grip.update(opening)
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
        if self.cmd_pub is not None:
            self.cmd_pub.publish(Float64MultiArray(data=[float(v) for v in q_cmd]))
        self._status(status)

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
                              f" planned={pdeg:.1f}deg lean=({plean[0]:+.2f},{plean[1]:+.2f})"
                              + (" [tool-down]" if self.args.tool_down else ""))
                if self.plan_side is not None:
                    extra += f" side={self.plan_side[0]:.1f}mm{' (dropped)' if self.plan_side[1] else ''}"
                if self.opening is not None:
                    extra += (f" opening={self.opening * 1000:.0f}mm (min {self.min_opening * 1000:.0f},"
                              f" closes <{self.args.grip_close_below * 1000:.0f})")
                if self.wrench_in is not None:
                    extra += f" ft_in={np.linalg.norm(self.wrench_in[:3]):.1f}N"
            self.get_logger().info(f"[{mode}] {status}{extra} |{self._readout()}")

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
    p.add_argument("--payload-mass", type=float, default=0.925)
    p.add_argument("--payload-cog", type=float, nargs=3, default=[0.0, 0.0, 0.058], metavar=("X", "Y", "Z"))
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
    p.add_argument("--tool-down", action="store_true",
                   help="Keep every planned pose pointing straight down (only the predicted yaw is used), as in the "
                        "clear-piece demos (all within 0.3 deg of vertical at the grasp)")
    p.add_argument("--hold-xy-deadband-mm", type=float, default=2.0,
                   help="While holding the piece, a mostly-downward plan that moves sideways less than this (mm) goes "
                        "straight down instead (demos: last 3 cm of the placing descent median 0.1 mm sideways); 0 disables")
    p.add_argument("--home-joints", type=float, nargs=6, default=list(DEMO_START_Q), metavar="Q",
                   help="Joints (rad) that B returns to between runs; default: median start of the clear-piece demos")
    p.add_argument("--home-speed", type=float, default=0.3, help="rad/s of the fastest joint when returning to start")
    p.add_argument("--home-max-move", type=float, default=1.5,
                   help="rad; B refuses if any joint would turn more than this (e.g. the pan wrapped the other way)")
    p.add_argument("--gripper-length", type=float, default=0.18)
    p.add_argument("--grip-close-below", type=float, default=0.030, help="Predicted opening (m) that closes the gripper")
    p.add_argument("--grip-open-above", type=float, default=0.055, help="...and that opens it again")
    p.add_argument("--gripper-ip", default="10.168.4.208")
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
    print(f"Inputs: wrist images{''.join(', ' + k for k in runner.state_keys)}"
          + (" (jaw gap guessed from the commands: no gripper connection without --live, or --no-gripper)"
             if "gripper_width" in runner.state_keys and (args.no_gripper or not args.live) else ""),
          flush=True)

    rclpy.init()
    node = PolicyNode(args, runner)
    node.timer.cancel()
    if args.live:
        error = (preflight(node)
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
        runner.close()
        if node.gripper is not None:
            node.gripper.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
