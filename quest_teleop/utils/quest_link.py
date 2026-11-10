"""One-way Quest data link over Tailscale: poses and buttons only, no headset access."""
from __future__ import annotations

import socket
import struct
import time

import numpy as np

MAGIC = b"QST3"
PACKET = struct.Struct("<4sI16d16d?f3f?")
NO_JOG = (0.0, 0.0, 0.0)
DEFAULT_PORT = 8765
STALE_S = 0.25


def pack(seq: int, pose: np.ndarray, head: np.ndarray, clutch: bool, trigger: float, jog=NO_JOG, click: bool = False) -> bytes:
    return PACKET.pack(
        MAGIC, seq & 0xFFFFFFFF,
        *np.asarray(pose, dtype=float).ravel(), *np.asarray(head, dtype=float).ravel(),
        bool(clutch), float(trigger), *jog, bool(click),
    )


def quest_sample(transforms: dict, buttons: dict):
    """Turn reader output into (world pose or None, clutch, trigger, head or None, jog, click).

    jog is (stick x, stick y, in/out), with in/out +1 for A and -1 for B.
    click is whether the right stick is pressed in (toggles tool-down mode).

    The app reports controllers relative to the headset, so head movement would look like
    hand movement. Multiplying by the head pose gives the controller's pose in the room.
    """
    relative, head = transforms.get("r"), transforms.get("h")
    if relative is not None and head is None:
        raise RuntimeError("Quest app sends no head pose: install the rebuilt APK")
    pose = None if relative is None else head @ relative
    stick = buttons.get("rightJS", (0.0, 0.0))
    jog = (float(stick[0]), float(stick[1]), float(bool(buttons.get("A"))) - float(bool(buttons.get("B"))))
    click = bool(buttons.get("RJ", False))
    return pose, bool(buttons.get("RG", False)), float(buttons.get("rightTrig", (0.0,))[0]), head, jog, click


class UdpSource:
    """Receives from one allowed IP. Same read() contract as the other pose sources."""

    def __init__(self, bind_ip: str, allowed_ip: str, port: int = DEFAULT_PORT, stale_s: float = STALE_S):
        self.allowed_ip = allowed_ip
        self.stale_s = stale_s
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind((bind_ip, port))
        self.port = self.sock.getsockname()[1]
        self.accepted = self.rejected = 0
        self._seq = None
        self._pose = None
        self._head = None
        self._clutch = False
        self._trigger = None
        self._jog = NO_JOG
        self._click = False
        self._received = 0.0

    def accept(self, data: bytes, addr, now: float | None = None) -> bool:
        if addr[0] != self.allowed_ip or len(data) != PACKET.size:
            self.rejected += 1
            return False
        magic, seq, *rest = PACKET.unpack(data)
        pose = np.array(rest[:16]).reshape(4, 4)
        head = np.array(rest[16:32]).reshape(4, 4)
        clutch, trigger, jog, click = rest[32], rest[33], tuple(rest[34:37]), rest[37]
        valid = (
            magic == MAGIC and np.all(np.isfinite(pose)) and np.all(np.isfinite(head))
            and np.isfinite(trigger) and 0.0 <= trigger <= 1.0
            and np.all(np.isfinite(jog)) and np.all(np.abs(jog) <= 1.01)
        )
        # Late packets are dropped; a large backward gap means the sender restarted.
        late = self._seq is not None and 0 < (self._seq - seq) & 0xFFFFFFFF < 1000
        if not valid or late or seq == self._seq:
            self.rejected += 1
            return False
        self._seq, self._pose, self._head, self._clutch, self._trigger, self._jog = seq, pose, head, clutch, trigger, jog
        self._click = click
        self._received = time.monotonic() if now is None else now
        self.accepted += 1
        return True

    def read(self, now: float | None = None):
        """Return (pose or None, clutch, trigger, head, jog, click). Silence drops the clutch, jog
        and click but keeps the last trigger value, so a network dropout can never open the gripper."""
        while True:
            try:
                data, addr = self.sock.recvfrom(2048)
            except BlockingIOError:
                break
            self.accept(data, addr, now)
        current = time.monotonic() if now is None else now
        if self._pose is None or current - self._received > self.stale_s:
            return None, False, self._trigger, None, NO_JOG, False
        return self._pose, self._clutch, self._trigger, self._head, self._jog, self._click
