"""Robotiq 2F-85 through the Robotiq URCap's text socket on the UR controller (port 63352).

Bare minimum, no ROS: select the gripper, send position, speed and force, read where it is.
Registers run 0..255. POS 0 is open and 255 closed. SPE 0 is the slowest speed, and FOR 0 is
Low Force mode, which also turns Secure Grip (automatic re-grasp) off.

This never activates the gripper, because activation opens and closes the fingers. If it is
not active, activate it from the pendant.
"""
from __future__ import annotations

import socket
import threading

PORT = 63352
STROKE_M = 0.085
POS_TOUCH = 230  # the 2F-85's fingers meet a little before 255


def pos_to_gap(pos: int) -> float:
    """Jaw gap in metres for a POS register value."""
    return STROKE_M * (1.0 - min(max(int(pos), 0), POS_TOUCH) / POS_TOUCH)


class GripperError(RuntimeError):
    pass


class URCapGripper:
    def __init__(self, host: str, port: int = PORT, sid: int = 3, timeout: float = 1.0):
        self.host, self.port, self.sid, self.timeout = host, port, sid, timeout
        self._sock = None

    def connect(self) -> None:
        self.close()
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        # The URCap answers "?" to every GET until a gripper is selected ("GET SID" lists them).
        self.set(SID=self.sid)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _cmd(self, line: str) -> str:
        if self._sock is None:
            raise GripperError("not connected")
        self._sock.sendall((line + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = self._sock.recv(256)
            if not chunk:
                raise GripperError(f"socket closed during {line!r}")
            data += chunk
        return data.decode().strip()

    def set(self, **regs) -> None:
        line = "SET " + " ".join(f"{k} {int(v)}" for k, v in regs.items())
        reply = self._cmd(line)
        if reply != "ack":
            raise GripperError(f"{line!r} -> {reply!r}")

    def get(self, var: str) -> int:
        reply = self._cmd(f"GET {var}")
        name, _, value = reply.partition(" ")
        if name != var or not value.strip().isdigit():
            raise GripperError(f"GET {var} -> {reply!r}")
        return int(value)

    def move(self, pos: int, speed: int, force: int) -> None:
        """Start a move; it runs on the gripper, so this returns at once. A new move replaces it."""
        self.set(POS=_reg(pos), SPE=_reg(speed), FOR=_reg(force), GTO=1)


def _reg(v) -> int:
    return min(max(int(v), 0), 255)


class GripperWorker:
    """Talks to the gripper on its own thread, so a slow socket never delays the control loop.

    command() records the wanted state; the thread sends it, reconnecting after errors, and
    keeps `state` = (jaw gap m, holding an object) up to date for the log.
    """

    POLL_S = 0.2

    def __init__(self, gripper: URCapGripper, speed: int, force: int, log):
        self.g, self.speed, self.force, self.log = gripper, speed, force, log
        self.state = None
        self._want = None   # True = closed, False = open, None = leave it alone
        self._sent = None
        self._wake = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=2.0)
        self.g.close()

    def command(self, closed: bool) -> None:
        self._want = bool(closed)
        self._wake.set()

    def _run(self) -> None:
        connected, warned, holding = False, None, False
        while not self._stop:
            self._wake.wait(self.POLL_S)
            self._wake.clear()
            try:
                if not connected:
                    self.g.connect()
                    if self.g.get("STA") != 3:
                        raise GripperError("the gripper is not activated: activate it from the pendant")
                    connected, warned, self._sent = True, None, None
                    self.log.info(f"Gripper connected ({self.g.host}, speed {self.speed}, force {self.force})")
                want = self._want
                if want is not None and want != self._sent:
                    self.g.move(255 if want else 0, self.speed, self.force)
                    self._sent = want
                pos, obj = self.g.get("POS"), self.g.get("OBJ")
                now_holding = obj in (1, 2)  # stopped on contact while opening / closing
                if now_holding and not holding:
                    self.log.info(f"Gripper holding an object at {pos_to_gap(pos) * 1000:.0f} mm")
                holding = now_holding
                self.state = (pos_to_gap(pos), holding)
            except (OSError, GripperError) as e:
                connected = False
                self.g.close()
                self.state = None
                if str(e) != warned:
                    warned = str(e)
                    self.log.error(f"Gripper: {e}. Retrying.")
                self._wake.wait(1.0)
