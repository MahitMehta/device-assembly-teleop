#!/usr/bin/env python3
"""Runs on the Mac: reads the USB-connected Quest and sends poses to coral over Tailscale.

Checks for a new headset frame every millisecond and sends each one once, as soon as it
appears (the headset produces one pose per displayed frame). A frozen stream (no new pose)
sends nothing, so coral sees silence and the arm holds.
"""
from __future__ import annotations

import argparse
import math
import socket
import time

import numpy as np

from quest_teleop.utils.quest_link import DEFAULT_PORT, NO_JOG, pack, quest_sample


def fake_samples():
    t0 = time.monotonic()
    while True:
        t = math.floor((time.monotonic() - t0) * 120) / 120  # new pose 120 times a second, like the headset
        pose = np.eye(4)
        pose[:3, 3] = [0.03 * math.sin(t), 0.0, 0.0]
        yield pose, 1.0 < t % 8.0 < 6.0, 0.5 + 0.5 * math.sin(t), np.eye(4), NO_JOG, False


def quest_samples():
    from oculus_reader.reader import OculusReader
    reader = OculusReader()
    while True:
        transforms, buttons = reader.get_transformations_and_buttons()
        yield quest_sample(transforms or {}, buttons or {})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--to", required=True, help="coral's Tailscale IP")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--poll-hz", type=float, default=1000.0, help="How often to check for a new headset frame")
    p.add_argument("--fake", action="store_true", help="Scripted motion instead of a Quest")
    p.add_argument("--duration", type=float, default=None)
    args = p.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((args.to, args.port))
    samples = fake_samples() if args.fake else quest_samples()
    seq, last, sent, start = 0, None, 0, time.monotonic()
    while args.duration is None or time.monotonic() - start < args.duration:
        pose, clutch, trigger, head, jog, click = next(samples)
        # Each headset frame carries a new pose (tracking is never perfectly still), so a
        # changed pose means a new frame: send it once, with that frame's buttons.
        if pose is not None and (last is None or not np.array_equal(pose, last)):
            last = pose.copy()
            seq += 1
            try:
                sock.send(pack(seq, pose, head, clutch, trigger, jog, click))
                sent += 1
            except OSError:
                pass  # Link dropped: coral holds the arm, and sending resumes when it returns.
        time.sleep(1.0 / args.poll_hz)
    print(f"sent {sent} packets")


if __name__ == "__main__":
    main()
