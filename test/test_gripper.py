import socket
import threading
import time

import pytest

from quest_teleop.utils.gripper import GripperError, GripperWorker, URCapGripper, pos_to_gap


class FakeURCap:
    """Answers like the URCap socket: '?' to every GET until a gripper is selected."""

    def __init__(self, sta=3, stop_at=None):
        self.regs = {"STA": sta, "POS": 0, "OBJ": 3, "SPE": 0, "FOR": 0, "GTO": 0}
        self.stop_at = stop_at  # POS where an object stops the jaws
        self.lines, self.sid = [], None
        self.srv = socket.create_server(("127.0.0.1", 0))
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            conn, _ = self.srv.accept()
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn):
        buf = b""
        while True:
            data = conn.recv(256)
            if not data:
                return
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                conn.sendall((self._reply(line.decode().strip()) + "\n").encode())

    def _reply(self, line):
        self.lines.append(line)
        cmd, *rest = line.split()
        if cmd == "GET":
            return f"{rest[0]} {self.regs[rest[0]]}" if self.sid == 3 else f"{rest[0]} ?"
        regs = dict(zip(rest[::2], (int(v) for v in rest[1::2])))
        if "SID" in regs:
            self.sid = regs.pop("SID")
        self.regs.update(regs)
        if "POS" in regs:
            target = regs["POS"]
            if self.stop_at is not None and target > self.stop_at:
                self.regs["POS"], self.regs["OBJ"] = self.stop_at, 2
            else:
                self.regs["OBJ"] = 3
        return "ack"


class Log:
    def __init__(self):
        self.lines = []

    def info(self, m): self.lines.append(m)
    def error(self, m): self.lines.append(m)


def _wait(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_selects_the_gripper_before_reading():
    fake = FakeURCap()
    g = URCapGripper("127.0.0.1", fake.port, sid=3)
    g.connect()
    assert fake.lines[0] == "SET SID 3" and g.get("STA") == 3


def test_unselected_gripper_reads_fail_clearly():
    fake = FakeURCap()
    g = URCapGripper("127.0.0.1", fake.port, sid=3)
    g._sock = socket.create_connection(("127.0.0.1", fake.port))
    with pytest.raises(GripperError, match=r"\?"):
        g.get("STA")


def test_worker_closes_and_opens_with_low_force_defaults():
    fake, log = FakeURCap(stop_at=120), Log()
    w = GripperWorker(URCapGripper("127.0.0.1", fake.port), speed=0, force=0, log=log)
    w.start()
    try:
        w.command(True)
        assert _wait(lambda: "SET POS 255 SPE 0 FOR 0 GTO 1" in fake.lines)
        assert _wait(lambda: w.state is not None and w.state[1])
        assert abs(w.state[0] - pos_to_gap(120)) < 1e-9
        assert any("holding an object" in m for m in log.lines)
        w.command(False)
        assert _wait(lambda: "SET POS 0 SPE 0 FOR 0 GTO 1" in fake.lines)
        assert _wait(lambda: w.state is not None and not w.state[1])
    finally:
        w.stop()


def test_worker_never_moves_an_inactive_gripper():
    fake, log = FakeURCap(sta=0), Log()
    w = GripperWorker(URCapGripper("127.0.0.1", fake.port), speed=0, force=0, log=log)
    w.start()
    try:
        w.command(True)
        assert _wait(lambda: any("not activated" in m for m in log.lines))
        time.sleep(0.3)
        assert not any(l.startswith("SET POS") or l.startswith("SET ACT") for l in fake.lines)
    finally:
        w.stop()


def test_worker_sends_nothing_until_commanded():
    fake = FakeURCap()
    w = GripperWorker(URCapGripper("127.0.0.1", fake.port), speed=0, force=0, log=Log())
    w.start()
    try:
        assert _wait(lambda: w.state is not None)
        assert not any(l.startswith("SET POS") for l in fake.lines)
    finally:
        w.stop()


def test_gap_mapping():
    assert pos_to_gap(0) == pytest.approx(0.085)
    assert pos_to_gap(230) == 0.0 and pos_to_gap(255) == 0.0
