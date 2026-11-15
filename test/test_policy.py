"""Policy-runner pieces that need no torch or ROS: waypoint timing, gripper hysteresis, and
TeleopCore.follow's limits."""
import numpy as np
import pytest

from quest_teleop.utils.core import Limits, Mapping, TeleopCore, fk, matrix_from_rotvec, rotvec_from_matrix
from quest_teleop.utils.policy_runner import Chunk, GripperHysteresis

DT = 1 / 125
HOME = np.array([0.0, -1.2, 1.2, -1.57, -1.57, 0.0])


def chunk_from(start: np.ndarray, steps: list[np.ndarray], grip=None, t0=0.0) -> Chunk:
    poses = np.stack([start] + steps)
    return Chunk(t0=t0, dt=0.1, poses=poses, grip=np.asarray(grip if grip is not None else [0.085] * len(poses), float))


def shifted(T, dx=0.0, dz=0.0, rot=(0, 0, 0)):
    out = T.copy()
    out[:3, 3] += (dx, 0.0, dz)
    out[:3, :3] = matrix_from_rotvec(np.array(rot, float)) @ T[:3, :3]
    return out


def test_chunk_interpolates_and_holds_at_the_ends():
    T0 = fk(HOME)
    c = chunk_from(T0, [shifted(T0, dx=0.01), shifted(T0, dx=0.02, rot=(0, 0, 0.2))], t0=5.0)
    assert np.allclose(c.at(4.0)[0], T0)                      # before t0: the first waypoint
    assert np.allclose(c.at(5.05)[0][:3, 3], T0[:3, 3] + [0.005, 0, 0])
    mid = c.at(5.15)[0]
    assert np.isclose(np.linalg.norm(rotvec_from_matrix(mid[:3, :3] @ T0[:3, :3].T)), 0.1, atol=1e-6)  # half of 0.2 rad
    assert np.allclose(c.at(99.0)[0], c.poses[-1])            # after the end: the last waypoint
    assert c.end() == pytest.approx(5.2)


def test_chunk_gripper_switches_at_the_midpoint_between_waypoints():
    T0 = fk(HOME)
    c = chunk_from(T0, [T0, T0], grip=[0.085, 0.0, 0.0])
    assert c.at(0.04)[1] == 0.085 and c.at(0.06)[1] == 0.0


def test_gripper_hysteresis():
    g = GripperHysteresis(close_below=0.03, open_above=0.055)
    assert g.update(0.085) is None          # never commands "open" before the first close
    assert g.update(0.04) is None           # in the dead band
    assert g.update(0.02) is True           # close
    assert g.update(0.05) is None           # stays closed in the dead band
    assert g.update(0.06) is False          # open
    assert g.update(0.04) is None           # stays open in the dead band


def test_follow_reaches_a_target_within_speed_limits():
    lim = Limits(max_linear_speed=0.1, max_angular_speed=0.5, max_joint_speed=1.0, box_half_extent=0.4)
    c = TeleopCore(HOME, Mapping(), lim)
    goal = shifted(c.target, dx=0.05, rot=(0, 0, 0.1))
    prev = c.target.copy()
    for _ in range(int(1.0 / DT)):
        c.follow(goal, DT)
        step = np.linalg.norm(c.target[:3, 3] - prev[:3, 3])
        assert step <= 0.1 * DT + 1e-9
        prev = c.target.copy()
    assert np.allclose(fk(c.q_cmd)[:3, 3], goal[:3, 3], atol=1e-3)
    assert c.status == "following"


def test_follow_stays_in_the_box():
    lim = Limits(max_linear_speed=1.0, max_joint_speed=3.0, box_half_extent=0.02)
    c = TeleopCore(HOME, Mapping(), lim)
    start = c.target[:3, 3].copy()
    goal = shifted(c.target, dx=0.10)
    for _ in range(200):
        c.follow(goal, DT)
    assert c.target[0, 3] - start[0] == pytest.approx(0.02, abs=1e-6)
    assert c.status == "workspace limit"
