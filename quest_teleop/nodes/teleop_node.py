#!/usr/bin/env python3
"""Quest 3 controller teleop for the UR5e through the running ur_robot_driver.

Dry-run is the default: it computes motion but sends nothing to the arm or gripper.
This node never switches controllers. Activate forward_position_controller yourself.

Controls (right controller): hold grip = move, trigger = gripper,
and with the grip released: stick up/down = forward/back along the way the gripper faces, stick left/right = turn it about
vertical, A/B = fine in/out along the tool's z axis.
Click the stick to toggle tool-down mode: the gripper turns to point straight down and stays
that way; twisting your hand still turns it about vertical. Any hold turns the mode off.
"""
from __future__ import annotations

import argparse
import math
import threading
import time

import numpy as np
import rclpy
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rcl_interfaces.srv import GetParameters
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from ur_dashboard_msgs.msg import RobotMode, SafetyMode

from quest_teleop.utils.quest_link import DEFAULT_PORT, NO_JOG, UdpSource, quest_sample
from quest_teleop.utils.core import JOINT_NAMES, Limits, Mapping, TeleopCore, jog_velocity
from quest_teleop.utils.gripper import STROKE_M, GripperWorker, URCapGripper

ROBOT_RUNNING, SAFETY_NORMAL = 7, 1
JOINT_STALE_S = 0.05
QUEST_STALE_S = 0.25
MAX_TRACKING_ERROR = 0.10   # rad between commanded and measured joints
CONTROLLER = "forward_position_controller"
TRIGGER_CLOSE = 0.5         # trigger depth that closes the gripper
TRIGGER_OPEN = 0.3          # ...and below which it opens again


class QuestSource:
    def __init__(self, ip: str | None):
        from oculus_reader.reader import OculusReader
        self._reader = OculusReader(ip_address=ip)
        self._last, self._changed = None, 0.0

    def read(self):
        """Return (pose or None, clutch, trigger, head, jog, click). Identical poses mean a frozen stream."""
        transforms, buttons = self._reader.get_transformations_and_buttons()
        pose, clutch, trigger, head, jog, click = quest_sample(transforms or {}, buttons or {})
        now = time.monotonic()
        if pose is not None and (self._last is None or not np.array_equal(pose, self._last)):
            self._last, self._changed = pose.copy(), now
        if pose is None or now - self._changed > QUEST_STALE_S:
            pose = None
        return pose, clutch, trigger, head, jog, click


class FakeSource:
    """Scripted motion for testing without a headset: hold, move, release, re-grip."""

    def __init__(self, amplitude: float = 0.03, jog: bool = False):
        self._t0 = time.monotonic()
        self._amplitude = amplitude
        self._jog = jog

    def read(self):
        t = time.monotonic() - self._t0
        pose = np.eye(4)
        pose[:3, 3] = [self._amplitude * math.sin(t), 0.0, 0.0]
        if self._jog:
            # Grip released throughout: stick, then a neutral gap, then the in/out button.
            phase = t % 8.0
            jog = (1.0, 0.0, 0.0) if phase < 3.0 else (0.0, 0.0, 1.0) if 4.0 < phase < 6.0 else NO_JOG
            return pose, False, 0.0, np.eye(4), jog, False
        return pose, 1.0 < t % 8.0 < 6.0, 0.5 + 0.5 * math.sin(t), np.eye(4), NO_JOG, False


class TeleopNode(Node):
    def __init__(self, args, source):
        super().__init__("quest_teleop")
        self.args, self.source = args, source
        self.lock = threading.Lock()
        self.q_actual = None
        self.joint_time = 0.0
        self.robot_mode = self.safety_mode = None
        self.speed_scaling = None
        self.force = None
        self.force_baseline = None
        self.core = None
        self.fault = None
        self.joints_stale = False
        self.was_moving = False
        self.jog_v = None
        self.grip_closed = None   # wanted state from the trigger; None until first pulled
        self.last_log = 0.0
        self.mapping = Mapping(
            robot_forward_yaw=math.radians(args.robot_forward_yaw_deg),
            position_scale=args.position_scale,
            rotation_scale=args.rotation_scale,
            tcp_offset=_z_offset(args.tcp_z),
        )
        self.limits = Limits(
            max_linear_speed=args.max_linear_speed,
            max_angular_speed=args.max_angular_speed,
            max_joint_speed=args.max_joint_speed,
            box_half_extent=args.box_half_extent,
            min_z=args.min_z,
            gripper_length=args.gripper_length,
        )

        # Separate groups so a slow tick or gripper reply never delays the joint and force
        # readings (one shared group runs every callback in turn).
        sensors, control = MutuallyExclusiveCallbackGroup(), MutuallyExclusiveCallbackGroup()
        self.create_subscription(JointState, "/joint_states", self._on_joints, qos_profile_sensor_data, callback_group=sensors)
        # The driver publishes these only on change, so a late subscriber needs the latched value.
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(RobotMode, "/io_and_status_controller/robot_mode", self._on_robot_mode, latched)
        self.create_subscription(SafetyMode, "/io_and_status_controller/safety_mode", self._on_safety_mode, latched)
        self.create_subscription(Float64, "/speed_scaling_state_broadcaster/speed_scaling", self._on_scaling, 10)
        self.create_subscription(WrenchStamped, "/force_torque_sensor_broadcaster/wrench", self._on_wrench, qos_profile_sensor_data, callback_group=sensors)

        self.cmd_pub = self.create_publisher(Float64MultiArray, f"/{CONTROLLER}/commands", 1) if args.live else None
        self.target_pub = self.create_publisher(PoseStamped, "/teleop/target_pose", 10)
        self.engaged_pub = self.create_publisher(Bool, "/teleop/engaged", 10)
        self.status_pub = self.create_publisher(String, "/teleop/status", 10)
        self.grip_cmd_pub = self.create_publisher(Float64, "/teleop/gripper_command", 10)
        self.grip_pos_pub = self.create_publisher(Float64, "/teleop/gripper_position", 10)
        # Our own gripper connection (gripper.py), only when live: dry-run sends nothing.
        self.gripper = None
        if args.live and not args.no_gripper:
            self.gripper = GripperWorker(URCapGripper(args.gripper_ip, sid=args.gripper_sid),
                                         args.gripper_speed, args.gripper_force, self.get_logger())

        self.timer = self.create_timer(1.0 / args.rate, self._tick, callback_group=control)

    def _on_joints(self, msg: JointState) -> None:
        # The driver publishes joints out of kinematic order: always map by name.
        index = {n: i for i, n in enumerate(msg.name)}
        if not all(n in index for n in JOINT_NAMES):
            return
        with self.lock:
            self.q_actual = np.array([msg.position[index[n]] for n in JOINT_NAMES])
            self.joint_time = time.monotonic()

    def _on_robot_mode(self, msg): self.robot_mode = msg.mode
    def _on_safety_mode(self, msg): self.safety_mode = msg.mode
    def _on_scaling(self, msg): self.speed_scaling = msg.data

    def _on_wrench(self, msg):
        f = msg.wrench.force
        self.force = np.array([f.x, f.y, f.z])

    def _gate(self, q_actual) -> str | None:
        """Return why motion must stop, or None. Checked on every tick while live."""
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
            # Absolute check too: re-gripping re-baselines, which must not excuse a hard press.
            if np.linalg.norm(self.force) > self.args.max_force_abs:
                return "absolute force limit exceeded"
            if self.force_baseline is not None and np.linalg.norm(self.force - self.force_baseline) > self.args.max_force_delta:
                return "contact force limit exceeded"
        return None

    def _tick(self) -> None:
        with self.lock:
            q_actual, have = (None, False) if self.q_actual is None else (self.q_actual.copy(), True)
        if not have:
            return
        if self.core is None:
            self.core = TeleopCore(q_actual, self.mapping, self.limits)
            self.get_logger().info("Initialized from measured joints")
            if self.args.tool_down:
                self.core.set_tool_down(True)

        pose, clutch, trigger, head, jog, click = self.source.read()
        toggled = self.core.click(click)
        if toggled is not None:
            self.get_logger().info("Tool-down mode ON: turning the gripper to point down" if toggled else "Tool-down mode OFF")

        # The stick works only with the grip released and a live, tracked controller.
        jog_v = jog_velocity(jog, self.args.jog_speed, self.args.jog_yaw_speed) if pose is not None and not clutch else np.zeros(3)
        jogging = bool(np.any(jog_v))
        self.jog_v = jog_v if jogging else None

        if self.args.live:
            reason = self._gate(q_actual)
            if reason == "joint states stale" and not self.fault:
                # A short gap in readings, not a sign of trouble at the arm: hold until fresh
                # readings return, then re-engage from the current poses (like lost tracking).
                if not self.joints_stale:
                    age = (time.monotonic() - self.joint_time) * 1000
                    self.get_logger().warning(f"Holding: joint states {age:.0f} ms old. Resumes when they are fresh.")
                self.joints_stale = True
                self.core.engaged = False
                self.was_moving = False
                self._publish_state(False, reason)
                return
            if self.joints_stale and not reason:
                self.joints_stale = False
                self.core.resync(q_actual)
                self.get_logger().info("Joint states fresh again; resuming")
            if reason or self.fault:
                if reason and reason != self.fault:
                    self.get_logger().error(f"HOLD: {reason}.{self._readout()} Release the grip and stick to clear.")
                    if self.core.tool_down:
                        # Otherwise the arm would start turning by itself once the hold clears.
                        self.core.set_tool_down(False)
                        self.get_logger().warning("Tool-down mode turned OFF by the hold; click the stick to turn it on again")
                # As with re-gripping: letting go first stops a held stick pushing on after a hold.
                self.fault = reason or (self.fault if clutch or jogging else None)
                self.jog_v = None
                self.core.resync(q_actual)
                self.force_baseline = None
                self.was_moving = False
                self._publish_state(False, self.fault or "fault")
                return

        dt = 1.0 / self.args.rate
        q_cmd = self.core.jog(jog_v, dt) if jogging else self.core.update(pose, clutch, dt, head)

        moving = self.core.engaged or jogging
        if moving and not self.was_moving:
            self.force_baseline = None if self.force is None else self.force.copy()
        if not moving:
            self.force_baseline = None
        self.was_moving = moving

        if self.cmd_pub is not None:
            self.cmd_pub.publish(Float64MultiArray(data=[float(v) for v in q_cmd]))
        self._send_gripper(trigger)
        self._publish_state(moving, self.core.status)

    def _send_gripper(self, trigger: float | None) -> None:
        """Trigger pulled = close, released = open."""
        if trigger is None:
            return
        # Two thresholds so a trigger resting near one of them doesn't flicker the gripper.
        want = trigger > TRIGGER_OPEN if self.grip_closed else trigger >= TRIGGER_CLOSE
        if self.grip_closed is None and not want:
            # Leave the gripper alone until the trigger is first pulled: commanding "open"
            # at startup would drop anything it is holding.
            return
        if want == self.grip_closed:
            return
        self.grip_closed = want
        self.grip_cmd_pub.publish(Float64(data=0.0 if want else STROKE_M))
        if self.gripper is not None:
            self.gripper.command(want)

    def _readout(self) -> str:
        """Force and gripper values for the log, to compare against the limits."""
        out = " tool-down" if self.core is not None and self.core.tool_down else ""
        if self.jog_v is not None:
            f, w, z = self.jog_v
            out += f" jog: forward={f * 1000:+.1f}mm/s turn={np.degrees(w):+.1f}deg/s in/out={z * 1000:+.1f}mm/s"
        if self.force is not None:
            out += f" force={np.linalg.norm(self.force):.0f}N"
            if self.force_baseline is not None:
                out += f" change={np.linalg.norm(self.force - self.force_baseline):.0f}N"
        if self.grip_closed is not None:
            out += f" gripper={'closed' if self.grip_closed else 'open'}"
        state = self.gripper.state if self.gripper is not None else None
        if state is not None:
            gap, holding = state
            out += f" jaws at {gap * 1000:.0f}mm{' on object' if holding else ''}"
        return out

    def _publish_state(self, engaged: bool, status: str) -> None:
        self.engaged_pub.publish(Bool(data=bool(engaged)))
        state = self.gripper.state if self.gripper is not None else None
        if state is not None:
            self.grip_pos_pub.publish(Float64(data=state[0]))
        self.status_pub.publish(String(data=status))
        T = self.core.target
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = [float(v) for v in T[:3, 3]]
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = _quat(T[:3, :3])
        self.target_pub.publish(msg)
        now = time.monotonic()
        if now - self.last_log > 1.0:
            self.last_log = now
            mode = "LIVE" if self.args.live else "DRY-RUN"
            if self.core.engaged:
                f, l, u = self.core.grip_motion()
                self.get_logger().info(f"[{mode}] {status} | since grip (m): forward={f:+.3f} left={l:+.3f} up={u:+.3f} |{self._readout()}")
            else:
                self.get_logger().info(f"[{mode}] {status} |{self._readout()}")


def _z_offset(z: float) -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = z
    return T


def _quat(R: np.ndarray):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return float(x), float(y), float(z), float(w)


def preflight(node: TeleopNode, timeout: float = 5.0) -> str | None:
    """Read-only checks before going live. Returns an error string, or None if OK."""
    lc = node.create_client(ListControllers, "/controller_manager/list_controllers")
    if not lc.wait_for_service(timeout_sec=timeout):
        return "controller_manager is not reachable"
    fut = lc.call_async(ListControllers.Request())
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
    if not fut.done():
        return "list_controllers timed out"
    states = {c.name: c.state for c in fut.result().controller}
    if states.get(CONTROLLER) != "active":
        active = [n for n, s in states.items() if s == "active" and "broadcaster" not in n]
        return f"{CONTROLLER} is {states.get(CONTROLLER, 'missing')}; active: {active}. Switch it deliberately, after checking nobody else is using the arm."

    gp = node.create_client(GetParameters, f"/{CONTROLLER}/get_parameters")
    if not gp.wait_for_service(timeout_sec=timeout):
        return f"{CONTROLLER} parameters are not reachable"
    fut = gp.call_async(GetParameters.Request(names=["joints"]))
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
    if not fut.done() or not fut.result().values:
        return "could not read the controller's joint order"
    joints = tuple(fut.result().values[0].string_array_value)
    if joints != JOINT_NAMES:
        return f"unexpected controller joint order: {joints}"
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true", help="Actually command the arm and gripper")
    p.add_argument("--fake-source", action="store_true", help="Scripted motion instead of a Quest")
    p.add_argument("--fake-amplitude", type=float, default=0.03)
    p.add_argument("--fake-jog", action="store_true", help="Scripted source uses the stick and buttons instead of the grip")
    p.add_argument("--quest-ip", default=None, help="Quest IP for wireless adb; USB if omitted")
    p.add_argument("--udp-bind", default=None, help="coral's Tailscale IP: receive from quest_sender")
    p.add_argument("--udp-from", default=None, help="The only IP allowed to send (your Mac's Tailscale IP)")
    p.add_argument("--udp-port", type=int, default=DEFAULT_PORT)
    p.add_argument("--rate", type=float, default=125.0)
    p.add_argument("--robot-forward-yaw-deg", type=float, default=180.0, help="Robot's forward direction, degrees from base +x (180 = base -x, measured for this setup)")
    p.add_argument("--position-scale", type=float, default=0.5)
    p.add_argument("--rotation-scale", type=float, default=1.0)
    p.add_argument("--tcp-z", type=float, default=0.0, help="Metres from flange to fingertips; rotation pivots here")
    p.add_argument("--max-linear-speed", type=float, default=0.10)
    p.add_argument("--max-angular-speed", type=float, default=0.50)
    p.add_argument("--max-joint-speed", type=float, default=0.50)
    p.add_argument("--max-force-delta", type=float, default=30.0, help="N above the value at clutch engage")
    p.add_argument("--max-force-abs", type=float, default=60.0, help="N, measured; the reading at rest varies with pose, so compare with the log")
    p.add_argument("--tool-down", action="store_true", help="Start in tool-down mode (stick click toggles it)")
    p.add_argument("--jog-speed", type=float, default=0.01, help="Stick forward/back speed at full deflection (m/s); A/B run at half of this")
    p.add_argument("--jog-yaw-speed", type=float, default=0.3, help="Stick turning speed at full deflection (rad/s)")
    p.add_argument("--box-half-extent", type=float, default=0.40, help="Metres the TCP may move from its startup position, per axis")
    p.add_argument("--min-z", type=float, default=0.03, help="Floor in the base frame (m), 3 cm above the arm's base by default: no part of the gripper, elbow or wrist goes below it")
    p.add_argument("--gripper-length", type=float, default=0.17, help="Flange to fingertips (m), for the floor check; a little over the 2F-85 with its coupling")
    p.add_argument("--gripper-ip", default="10.168.4.208", help="UR controller running the Robotiq URCap (socket port 63352)")
    p.add_argument("--gripper-sid", type=int, default=3, help="Gripper ID on the URCap socket ('GET SID' lists them)")
    p.add_argument("--gripper-speed", type=int, default=0, help="Robotiq speed register, 0 (slowest) to 255")
    p.add_argument("--gripper-force", type=int, default=0, help="Robotiq force register, 0 (Low Force mode, Secure Grip off) to 255")
    p.add_argument("--no-gripper", action="store_true", help="Never connect to or command the gripper")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rclpy.init()
    if bool(args.udp_bind) != bool(args.udp_from):
        raise SystemExit("--udp-bind and --udp-from must be used together")
    if args.udp_bind:
        source = UdpSource(args.udp_bind, args.udp_from, args.udp_port)
    elif args.fake_source:
        source = FakeSource(args.fake_amplitude, args.fake_jog)
    else:
        source = QuestSource(args.quest_ip)
    node = TeleopNode(args, source)
    node.timer.cancel()
    if args.live:
        error = preflight(node)
        if error:
            node.get_logger().error(f"Refusing to go live: {error}")
            node.destroy_node()
            rclpy.shutdown()
            return 1
    if node.gripper is not None:
        node.gripper.start()
    node.timer.reset()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node.gripper is not None:
            node.gripper.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
