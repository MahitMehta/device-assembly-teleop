#!/usr/bin/env python3
"""Activate an already-loaded controller next to the active ones, e.g. force_mode_controller, which
the UR launch loads inactive. Waits (up to --timeout s) for the controller manager and for the
controller to be loaded. The image has no `ros2 control` CLI, so this calls the services directly.

    python3 scripts/activate_controller.py force_mode_controller
"""
import argparse
import sys
import time

import rclpy
from controller_manager_msgs.srv import ListControllers, SwitchController


def call(node, client, req, timeout=5.0):
    fut = client.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
    return fut.result() if fut.done() else None


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("controller")
    a.add_argument("--timeout", type=float, default=120.0)
    args = a.parse_args()
    rclpy.init()
    node = rclpy.create_node("activate_controller")
    lc = node.create_client(ListControllers, "/controller_manager/list_controllers")
    sc = node.create_client(SwitchController, "/controller_manager/switch_controller")
    end, state = time.monotonic() + args.timeout, None
    try:
        while time.monotonic() < end:
            if lc.wait_for_service(timeout_sec=1.0):
                res = call(node, lc, ListControllers.Request())
                state = {c.name: c.state for c in res.controller}.get(args.controller) if res else None
                if state == "active":
                    print(f"{args.controller} active", flush=True)
                    return 0
                if state == "inactive" and sc.wait_for_service(timeout_sec=1.0):
                    req = SwitchController.Request(activate_controllers=[args.controller],
                                                   strictness=SwitchController.Request.STRICT)
                    res = call(node, sc, req)
                    if res is not None and res.ok:
                        continue   # confirm with the next listing
            time.sleep(1.0)
        print(f"ERROR: could not activate {args.controller} (last state: {state or 'not loaded'})", file=sys.stderr, flush=True)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
