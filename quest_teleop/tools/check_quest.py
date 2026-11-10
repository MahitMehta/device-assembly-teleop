#!/usr/bin/env python3
"""Reads the USB-connected Quest for 15 s and prints a summary. Sends nothing anywhere.

Starting the reader also installs and launches its app on the headset if needed.
Run from the repo root: .venv/bin/python -m quest_teleop.tools.check_quest
While it runs, move the right controller and squeeze grip and trigger.
"""
import time

import numpy as np
from oculus_reader.reader import OculusReader

DURATION = 15.0

reader = OculusReader()
start = time.monotonic()
last = first = None
updates = 0
grip_seen = False
max_trigger = 0.0
controllers, buttons_seen = set(), set()

while time.monotonic() - start < DURATION:
    transforms, buttons = reader.get_transformations_and_buttons()
    transforms, buttons = transforms or {}, buttons or {}
    controllers |= set(transforms)
    buttons_seen |= set(buttons)
    pose = transforms.get("r")
    if pose is not None and (last is None or not np.array_equal(pose, last)):
        updates += 1
        last = pose.copy()
        first = pose.copy() if first is None else first
    grip_seen = grip_seen or bool(buttons.get("RG"))
    max_trigger = max(max_trigger, float(buttons.get("rightTrig", (0.0,))[0]))
    time.sleep(0.002)
reader.stop()

print("controllers seen:", sorted(controllers) or "NONE")
print("buttons seen:", sorted(buttons_seen) or "NONE")
print("pose updates per second:", round(updates / DURATION, 1))
print("grip pressed:", grip_seen)
print("max trigger:", round(max_trigger, 2))
if first is not None:
    print("right controller moved (m):", round(float(np.linalg.norm(last[:3, 3] - first[:3, 3])), 3))
    print("rotation determinant (should be 1):", round(float(np.linalg.det(last[:3, :3])), 3))
