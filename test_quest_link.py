import socket
import time

import numpy as np

import pytest

from quest_link import MAGIC, NO_JOG, PACKET, UdpSource, pack, quest_sample

MAC = ("100.64.0.5", 40000)


def source():
    return UdpSource("127.0.0.1", MAC[0], port=0)


def test_accepts_allowed_sender():
    s = source()
    assert s.accept(pack(1, np.eye(4), np.eye(4), True, 0.5), MAC, now=0.0)
    pose, clutch, trigger, _, _, _ = s.read(now=0.1)
    assert np.array_equal(pose, np.eye(4)) and clutch and trigger == 0.5


def test_rejects_other_senders():
    s = source()
    assert not s.accept(pack(1, np.eye(4), np.eye(4), True, 0.5), ("100.64.0.6", 40000), now=0.0)
    assert s.read(now=0.0) == (None, False, None, None, NO_JOG, False)


def test_rejects_malformed_packets():
    s = source()
    bad_magic = PACKET.pack(b"XXXX", 1, *np.eye(4).ravel(), *np.eye(4).ravel(), True, 0.5, *NO_JOG, False)
    nan_pose = PACKET.pack(MAGIC, 2, *(np.eye(4) * np.nan).ravel(), *np.eye(4).ravel(), True, 0.5, *NO_JOG, False)
    for data in (b"", b"short", bad_magic, nan_pose, pack(3, np.eye(4), np.eye(4), True, 7.0)):
        assert not s.accept(data, MAC, now=0.0)
    assert s.read(now=0.0)[0] is None


def test_rejects_late_and_duplicate_packets():
    s = source()
    assert s.accept(pack(10, np.eye(4), np.eye(4), True, 0.5), MAC, now=0.0)
    assert not s.accept(pack(9, np.eye(4), np.eye(4), False, 0.1), MAC, now=0.0)
    assert not s.accept(pack(10, np.eye(4), np.eye(4), False, 0.1), MAC, now=0.0)


def test_sender_restart_is_accepted():
    s = source()
    assert s.accept(pack(50000, np.eye(4), np.eye(4), True, 0.5), MAC, now=0.0)
    assert s.accept(pack(1, np.eye(4), np.eye(4), True, 0.5), MAC, now=0.0)


def test_silence_drops_clutch_but_keeps_gripper():
    s = source()
    s.accept(pack(1, np.eye(4), np.eye(4), True, 0.9), MAC, now=0.0)
    pose, clutch, trigger, _, _, _ = s.read(now=1.0)
    assert pose is None and clutch is False and abs(trigger - 0.9) < 1e-6


def test_real_socket_filters_by_source():
    s = UdpSource("127.0.0.1", "127.0.0.1", port=0)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.sendto(pack(1, np.eye(4), np.eye(4), True, 0.25), ("127.0.0.1", s.port))
    time.sleep(0.05)
    assert s.read()[2] == 0.25


def test_head_is_delivered():
    s = source()
    head = np.eye(4)
    head[0, 3] = 0.5
    s.accept(pack(1, np.eye(4), head, True, 0.5), MAC, now=0.0)
    assert np.array_equal(s.read(now=0.0)[3], head)


def test_sample_computes_world_pose_from_head():
    head, rel = np.eye(4), np.eye(4)
    head[0, 3], rel[1, 3] = 1.0, 2.0
    pose, clutch, trigger, got, _, _ = quest_sample({"r": rel, "h": head}, {"RG": True, "rightTrig": (0.7,)})
    assert np.allclose(pose[:3, 3], [1.0, 2.0, 0.0]) and clutch and trigger == 0.7 and got is head


def test_old_app_without_head_pose_fails_clearly():
    with pytest.raises(RuntimeError, match="rebuilt APK"):
        quest_sample({"r": np.eye(4)}, {})


def test_jog_is_delivered_and_silence_stops_it():
    s = source()
    assert s.accept(pack(1, np.eye(4), np.eye(4), False, 0.0, (0.5, -1.0, 1.0)), MAC, now=0.0)
    assert np.allclose(s.read(now=0.1)[4], (0.5, -1.0, 1.0))
    assert s.read(now=1.0)[4] == NO_JOG


def test_rejects_bad_jog():
    s = source()
    for jog in ((float("nan"), 0.0, 0.0), (0.0, 5.0, 0.0)):
        assert not s.accept(pack(1, np.eye(4), np.eye(4), False, 0.0, jog), MAC, now=0.0)


def test_quest_sample_reads_stick_and_buttons():
    t = {"r": np.eye(4), "h": np.eye(4)}
    assert quest_sample(t, {"rightJS": (0.25, -0.5), "A": True})[4] == (0.25, -0.5, 1.0)
    assert quest_sample(t, {"B": True})[4] == (0.0, 0.0, -1.0)
    assert quest_sample(t, {})[4] == NO_JOG


def test_stick_click_is_delivered_and_silence_clears_it():
    t = {"r": np.eye(4), "h": np.eye(4)}
    assert quest_sample(t, {"RJ": True})[5] is True
    assert quest_sample(t, {})[5] is False
    s = source()
    assert s.accept(pack(1, np.eye(4), np.eye(4), False, 0.0, NO_JOG, True), MAC, now=0.0)
    assert s.read(now=0.1)[5] is True
    assert s.read(now=1.0)[5] is False
