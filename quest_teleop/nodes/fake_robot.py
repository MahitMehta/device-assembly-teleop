#!/usr/bin/env python3
"""Stand-in for the UR driver, for testing on an isolated ROS domain. Never use domain 66.

Usage: fake_robot.py [--controller-state active|inactive] [--inject safety|stall|force]
Follows commands, then prints a JSON report when stopped by --duration.
"""
import argparse
import json
import time

import numpy as np
import rclpy
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers
from std_srvs.srv import Trigger
from ur_msgs.srv import SetPayload
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from ur_dashboard_msgs.msg import RobotMode, SafetyMode

ORDER = ["shoulder_pan_joint", "wrist_2_joint", "wrist_3_joint", "wrist_1_joint", "elbow_joint", "shoulder_lift_joint"]
KIN = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

a = argparse.ArgumentParser()
a.add_argument("--controller-state", default="active")
a.add_argument("--inject", default="none")
a.add_argument("--inject-at", type=float, default=4.0)
a.add_argument("--duration", type=float, default=10.0)
args = a.parse_args()

rclpy.init()
n = Node("controller_manager")
ctrl = Node("forward_position_controller", parameter_overrides=[Parameter("joints", value=KIN)], automatically_declare_parameters_from_overrides=True)
q = np.array([0.0, -1.2, 1.2, -1.57, -1.57, 0.0])
cmds, cmds_after, statuses, engaged = [], [], [], []
t0 = time.monotonic()
inject_on = lambda: args.inject != "none" and time.monotonic() - t0 > args.inject_at


def on_cmd(m):
    global q
    arr = np.array(m.data)
    cmds.append(arr)
    if inject_on():
        cmds_after.append(arr)
    if not (args.inject == "stall" and inject_on()):
        q = arr


def list_cb(req, res):
    res.controller = [ControllerState(name="forward_position_controller", state=args.controller_state),
                      ControllerState(name="scaled_joint_trajectory_controller", state="inactive")]
    return res


n.create_service(ListControllers, "/controller_manager/list_controllers", list_cb)
n.create_service(SetPayload, "/io_and_status_controller/set_payload", lambda req, res: (setattr(res, "success", True), res)[1])
n.create_service(Trigger, "/io_and_status_controller/zero_ftsensor", lambda req, res: (setattr(res, "success", True), res)[1])
n.create_subscription(Float64MultiArray, "/forward_position_controller/commands", on_cmd, 1)
n.create_subscription(String, "/teleop/status", lambda m: statuses.append(m.data), 10)
n.create_subscription(Bool, "/teleop/engaged", lambda m: engaged.append(m.data), 10)
js = n.create_publisher(JointState, "/joint_states", 1)
# Like the real driver: latched, and published only when the value changes.
latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
rm = n.create_publisher(RobotMode, "/io_and_status_controller/robot_mode", latched)
sm = n.create_publisher(SafetyMode, "/io_and_status_controller/safety_mode", latched)
rm.publish(RobotMode(mode=7))
last_safety = [None]
sp = n.create_publisher(Float64, "/speed_scaling_state_broadcaster/speed_scaling", 10)
ws = n.create_publisher(WrenchStamped, "/force_torque_sensor_broadcaster/wrench", 1)


def pub():
    m = JointState(name=ORDER)
    m.header.stamp = n.get_clock().now().to_msg()
    m.position = [float(q[KIN.index(x)]) for x in ORDER]
    js.publish(m)
    safety = 3 if args.inject == "safety" and inject_on() else 1
    if safety != last_safety[0]:
        sm.publish(SafetyMode(mode=safety))
        last_safety[0] = safety
    sp.publish(Float64(data=1.0))
    w = WrenchStamped()
    w.wrench.force.z = 100.0 if args.inject == "force" and inject_on() else 5.0
    ws.publish(w)


n.create_timer(0.002, pub)
ex = rclpy.executors.MultiThreadedExecutor(4)
ex.add_node(n)
ex.add_node(ctrl)
while time.monotonic() - t0 < args.duration:
    ex.spin_once(timeout_sec=0.05)
c = np.array(cmds) if cmds else np.zeros((0, 6))
d = np.abs(np.diff(c, axis=0)) if len(c) > 1 else np.zeros((1, 6))
print(json.dumps({
    "commands": len(c),
    "joints_moved": (np.abs(c - c[0]).max(axis=0).round(4).tolist() if len(c) else []),
    "max_step_rad": float(d.max()),
    "commands_after_inject": len(cmds_after),
    "motion_after_inject_rad": float(np.abs(np.array(cmds_after) - cmds_after[0]).max()) if cmds_after else 0.0,
    "statuses": sorted(set(statuses)),
    "engaged_seen": sorted(set(engaged)),
}))
