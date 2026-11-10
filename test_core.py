import numpy as np
import pytest

from core import _chain, gripper_points, Limits, Mapping, TeleopCore, jog_velocity, pointing_down, yaw_of, operator_to_base, fk, jacobian, matrix_from_rotvec, rotvec_from_matrix

DT = 1 / 125
# Live /joint_states from coral, reordered from the driver's order into kinematic order.
LIVE_Q = np.array([-3.3441832701312464, -1.5566415621763845, -1.6043052673339844,
                   -4.440716644326681, 5.549713611602783, -1.3818686644183558])
LIVE_TCP = np.array([-0.2588041394722707, 0.26550572834747044, 0.4918510752715784])
LIVE_QUAT = (-0.8781577761521634, 0.2251265747933079, 0.11232869077011051, 0.40686510139749366)
HOME = np.array([0.0, -1.2, 1.2, -1.57, -1.57, 0.0])


def pose(x=0.0, y=0.0, z=0.0, rot=(0, 0, 0)):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    T[:3, :3] = matrix_from_rotvec(np.array(rot, dtype=float))
    return T


def run(c, path, clutch=True):
    for T in path:
        c.update(T, clutch, DT)


def ramp(start, end, secs=2.0):
    n = int(secs / DT)
    return [pose(*(np.array(start) + (np.array(end) - np.array(start)) * i / n)) for i in range(n + 1)]


def test_rotvec_round_trip_including_pi():
    for r in ([0.1, -0.2, 0.3], [np.pi, 0, 0], [0, 0, 0], [0, np.pi - 1e-7, 0]):
        got = matrix_from_rotvec(rotvec_from_matrix(matrix_from_rotvec(np.array(r))))
        assert np.allclose(got, matrix_from_rotvec(np.array(r)), atol=1e-6)


def test_jacobian_matches_finite_difference():
    J, eps, base = jacobian(HOME), 1e-6, fk(HOME)
    for i in range(6):
        q = HOME.copy()
        q[i] += eps
        assert np.allclose((fk(q)[:3, 3] - base[:3, 3]) / eps, J[:3, i], atol=1e-4)


def test_no_jump_when_clutch_engages_far_from_origin():
    c = TeleopCore(HOME)
    q = c.q_cmd.copy()
    c.update(pose(5, 5, 5), True, DT)
    assert np.array_equal(c.q_cmd, q)


def test_tracks_scaled_motion():
    c = TeleopCore(HOME, Mapping(position_scale=0.5))
    start = c.target[:3, 3].copy()
    run(c, ramp((0, 0, 0), (0.1, 0, 0)))
    run(c, [pose(0.1)] * 125)
    assert np.allclose(fk(c.q_cmd)[:3, 3] - start, [0.05, 0, 0], atol=1e-3)


def test_release_holds_and_regrip_is_relative():
    c = TeleopCore(HOME)
    run(c, ramp((0, 0, 0), (0.05, 0, 0)))
    held = c.q_cmd.copy()
    run(c, [pose(0.5)] * 20, clutch=False)
    assert np.array_equal(c.q_cmd, held)
    c.update(pose(0.5), True, DT)
    assert np.array_equal(c.q_cmd, held)


def test_speed_limits_hold_during_large_fast_motion():
    lim = Limits(max_controller_jump=99)
    c = TeleopCore(HOME, Mapping(position_scale=1.0), lim)
    c.update(pose(), True, DT)
    prev_q, prev_p = c.q_cmd.copy(), fk(c.q_cmd)[:3, 3]
    for i in range(1, 200):
        c.update(pose(i * 0.02), True, DT)
        assert np.max(np.abs(c.q_cmd - prev_q)) <= lim.max_joint_speed * DT + 1e-9
        assert np.linalg.norm(fk(c.q_cmd)[:3, 3] - prev_p) <= lim.max_linear_speed * DT * 1.05
        prev_q, prev_p = c.q_cmd.copy(), fk(c.q_cmd)[:3, 3]


def test_tracking_glitch_stops_until_regrip():
    c = TeleopCore(HOME)
    run(c, ramp((0, 0, 0), (0.02, 0, 0), 0.5))
    held = c.q_cmd.copy()
    run(c, [pose(1.0)] + ramp((1, 0, 0), (1.1, 0, 0), 0.5))
    assert np.array_equal(c.q_cmd, held) and "glitch" in c.status
    c.update(pose(1.1), False, DT)
    c.update(pose(1.1), True, DT)
    assert c.engaged


def test_lost_tracking_holds():
    c = TeleopCore(HOME)
    run(c, ramp((0, 0, 0), (0.02, 0, 0), 0.5))
    held = c.q_cmd.copy()
    c.update(None, True, DT)
    assert np.array_equal(c.q_cmd, held) and not c.engaged


def test_workspace_box_is_enforced():
    c = TeleopCore(HOME, Mapping(position_scale=1.0), Limits(box_half_extent=0.03))
    center = c.box_center.copy()
    run(c, ramp((0, 0, 0), (0.5, 0, 0), 8.0))
    assert np.max(np.abs(fk(c.q_cmd)[:3, 3] - center)) < 0.032


def test_box_default_is_40cm_and_reports_the_limit():
    assert Limits().box_half_extent == 0.40
    c = TeleopCore(HOME, Mapping(position_scale=1.0), Limits(box_half_extent=0.03))
    run(c, ramp((0, 0, 0), (0.5, 0, 0), 8.0))
    assert c.status == "workspace limit"


def test_min_z_stops_downward_motion():
    start = fk(HOME)[2, 3]
    c = TeleopCore(HOME, Mapping(position_scale=1.0), Limits(min_z=start - 0.02, max_linear_speed=0.1))
    run(c, ramp((0, 0, 0), (0, -0.3, 0), 4.0))
    run(c, ramp((0, 0, 0), (0, 0, -0.3), 4.0))
    assert fk(c.q_cmd)[2, 3] > start - 0.022


def test_starting_below_min_z_does_not_lift_the_arm():
    start = fk(HOME)[:3, 3].copy()
    c = TeleopCore(HOME, Mapping(position_scale=1.0), Limits(min_z=start[2] + 0.05))
    for _ in range(100):
        c.update(pose(0, 0, 0), True, DT)
    assert abs(fk(c.q_cmd)[2, 3] - start[2]) < 1e-3


def _jog_for(c, velocity, seconds=2.0):
    for _ in range(int(seconds / DT)):
        c.jog(velocity, DT)


def _flat(v):
    v = np.array([v[0], v[1], 0.0])
    return v / np.linalg.norm(v)


def test_jog_forward_follows_the_gripper_heading_and_stays_level():
    turned = HOME + np.array([0.0, 0.0, 0.0, 0.3, -0.4, 0.8])
    for q in (HOME, turned):
        c = TeleopCore(q)
        start = fk(c.q_cmd)
        _jog_for(c, np.array([0.01, 0.0, 0.0]))
        end = fk(c.q_cmd)
        d = end[:3, 3] - start[:3, 3]
        assert np.allclose(d, 0.02 * _flat(start[:3, 1]), atol=5e-4)  # level, along the heading
        assert np.linalg.norm(rotvec_from_matrix(end[:3, :3] @ start[:3, :3].T)) < 1e-3


def test_jog_forward_direction_turns_with_the_gripper():
    moves = []
    for q in (HOME, HOME + np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2])):
        c = TeleopCore(q)
        start = fk(c.q_cmd)[:3, 3]
        _jog_for(c, np.array([0.01, 0.0, 0.0]))
        moves.append(fk(c.q_cmd)[:3, 3] - start)
    assert np.linalg.norm(moves[0]) > 0.015 and np.linalg.norm(moves[1]) > 0.015
    assert abs(_flat(moves[0]) @ _flat(moves[1])) < 0.1


def test_jog_turn_is_about_vertical_in_place():
    c = TeleopCore(HOME)
    c.set_tool_down(True)
    run(c, [pose()] * int(6 / DT), clutch=False)  # point down first
    start = fk(c.q_cmd)
    _jog_for(c, np.array([0.0, -0.2, 0.0]), seconds=1.0)  # stick right: clockwise from above
    end = fk(c.q_cmd)
    turn = rotvec_from_matrix(end[:3, :3] @ start[:3, :3].T)
    assert np.allclose(turn, [0.0, 0.0, -0.2], atol=5e-3)
    assert np.allclose(end[:3, 3], start[:3, 3], atol=5e-4)
    assert end[:3, 2] @ DOWN > 0.9999
    # Forward then follows the new heading.
    assert np.allclose(_flat(end[:3, 1]), matrix_from_rotvec(np.array([0, 0, -0.2])) @ _flat(start[:3, 1]), atol=5e-3)


def test_jog_inout_runs_along_the_tool_z_axis():
    c = TeleopCore(HOME)
    start = fk(c.q_cmd)
    _jog_for(c, np.array([0.0, 0.0, 0.005]))
    d = fk(c.q_cmd)[:3, 3] - start[:3, 3]
    assert np.allclose(d, 0.01 * start[:3, 2], atol=5e-4)


def test_jog_velocity_shape():
    assert np.array_equal(jog_velocity((0.1, -0.14, 0.0), 0.01, 0.3), np.zeros(3))
    # Stick up = forward, stick right = clockwise (negative yaw), A = in at half speed.
    assert np.allclose(jog_velocity((0.0, 1.0, 0.0), 0.01, 0.3), [0.01, 0.0, 0.0])
    assert np.allclose(jog_velocity((1.0, 0.0, 0.0), 0.01, 0.3), [0.0, -0.3, 0.0])
    assert np.allclose(jog_velocity((0.0, 0.0, -1.0), 0.01, 0.3), [0.0, 0.0, -0.005])
    # Half stick is well under half speed, for fine control.
    assert 0 < jog_velocity((0.0, 0.5, 0.0), 0.01, 0.3)[0] < 0.002


def test_jog_without_a_heading_does_not_drive():
    c = TeleopCore(HOME)
    c.target[:3, :3] = matrix_from_rotvec(np.array([np.pi / 2, 0.0, 0.0]))  # tool +y straight up
    c.jog(np.array([0.01, 0.0, 0.0]), DT)
    assert "no heading" in c.status


def test_jog_respects_speed_box_and_floor():
    c = TeleopCore(HOME, limits=Limits(max_linear_speed=0.02))
    start = fk(c.q_cmd)[:3, 3].copy()
    c.jog(np.array([1.0, 0.0, 0.0]), DT)
    assert np.linalg.norm(c.target[:3, 3] - start) <= 0.02 * DT + 1e-9
    c = TeleopCore(HOME, limits=Limits(max_angular_speed=0.1))
    R0 = c.target[:3, :3].copy()
    c.jog(np.array([0.0, 5.0, 0.0]), DT)
    assert np.linalg.norm(rotvec_from_matrix(c.target[:3, :3] @ R0.T)) <= 0.1 * DT + 1e-9

    c = TeleopCore(HOME, limits=Limits(box_half_extent=0.01))
    center = c.box_center.copy()
    _jog_for(c, np.array([0.02, 0.0, 0.02]))
    assert c.status == "workspace limit" and np.max(np.abs(fk(c.q_cmd)[:3, 3] - center)) < 0.0105

    start_z = fk(HOME)[2, 3]
    c = TeleopCore(HOME, limits=Limits(min_z=start_z - 0.005, max_linear_speed=0.02))
    R = fk(HOME)[:3, :3]
    _jog_for(c, np.array([0.0, 0.0, -0.02 * np.sign(R[2, 2])]))  # in/out pointed downward
    assert fk(c.q_cmd)[2, 3] > start_z - 0.0055


def test_jog_stops_with_the_stick_and_grip_does_not_jump_after():
    c = TeleopCore(HOME, Mapping(position_scale=1.0))
    _jog_for(c, np.array([0.01, 0.0, 0.0]), 1.0)
    held = c.q_cmd.copy()
    for _ in range(50):
        c.update(pose(0, 0, 0), False, DT)
    assert np.array_equal(c.q_cmd, held)
    c.update(pose(0.3, 0.3, 0.3), True, DT)
    assert np.array_equal(c.q_cmd, held)


def test_singular_pose_stays_finite():
    straight = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
    c = TeleopCore(straight)
    run(c, ramp((0, 0, 0), (0, 0, 0.05)))
    assert np.all(np.isfinite(c.q_cmd))


def test_stops_at_joint_limit():
    q = HOME.copy()
    q[5] = 2 * np.pi - 0.06
    c = TeleopCore(q, limits=Limits(max_angular_speed=1.0))
    axis = fk(q)[:3, 2]
    hit = False
    for i in range(1, 300):
        c.update(pose(rot=axis * i * 0.004), True, DT)
        hit = hit or c.status == "joint limit"
    assert hit
    assert np.all(np.abs(c.q_cmd) <= 2 * np.pi - 0.05)


def test_position_mapping_uses_operator_frame():
    R = matrix_from_rotvec(np.array([0, 0, np.pi / 2]))
    c = TeleopCore(HOME, Mapping(operator_to_base=R, position_scale=1.0))
    start = c.target[:3, 3].copy()
    run(c, ramp((0, 0, 0), (0.05, 0, 0)))
    run(c, [pose(0.05)] * 125)
    assert np.allclose(fk(c.q_cmd)[:3, 3] - start, [0, 0.05, 0], atol=1e-3)


def test_fk_against_live_robot():
    """Nominal FK should match the arm's reported pose. Its TCP is set at the flange."""
    T = fk(LIVE_Q)
    assert np.linalg.norm(LIVE_TCP - T[:3, 3]) < 0.005
    x, y, z, w = LIVE_QUAT
    reported = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    assert np.linalg.norm(rotvec_from_matrix(reported @ T[:3, :3].T)) < 0.01


def _head_facing(yaw=0.0, pitch=0.0):
    """Quest world is y up; a headset at zero yaw looks along -z. Positive pitch looks up."""
    T = np.eye(4)
    T[:3, :3] = matrix_from_rotvec(np.array([0.0, yaw, 0.0])) @ matrix_from_rotvec(np.array([pitch, 0.0, 0.0]))
    return T


def _quest_forward(yaw=0.0):
    return matrix_from_rotvec(np.array([0.0, yaw, 0.0])) @ np.array([0.0, 0.0, -1.0])


@pytest.mark.parametrize("yaw", [0.0, 1.0, -2.0, 3.0])
def test_forward_follows_headset_whichever_way_you_face(yaw):
    R = operator_to_base(_head_facing(yaw), np.radians(135))
    expected = [np.cos(np.radians(135)), np.sin(np.radians(135)), 0.0]
    assert np.allclose(R @ _quest_forward(yaw), expected, atol=1e-9)


def test_axes_are_proper_and_up_stays_up():
    R = operator_to_base(_head_facing(0.7, 0.2), 0.3)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ np.array([0.0, 1.0, 0.0]), [0, 0, 1], atol=1e-9)
    # The headset's right, projected level, is the robot's right of forward.
    right = matrix_from_rotvec(np.array([0.0, 0.7, 0.0])) @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(R @ right, [np.sin(0.3), -np.cos(0.3), 0.0], atol=1e-9)


@pytest.mark.parametrize("pitch", [-1.5, -0.8, 0.0, 0.8, 1.5])
def test_tilting_your_head_does_not_change_forward(pitch):
    R = operator_to_base(_head_facing(0.5, pitch), 0.0)
    assert np.allclose(R @ _quest_forward(0.5), [1, 0, 0], atol=1e-6)


def _world_ramp(direction, metres=0.05, secs=1.0):
    n = int(secs / DT)
    d = np.asarray(direction, dtype=float)
    return [pose(*(d * metres * i / n)) for i in range(n + 1)]


def test_turning_head_with_still_hand_does_not_move_arm():
    c = TeleopCore(HOME, Mapping(position_scale=1.0))
    c.update(pose(), True, DT, _head_facing(0.0))
    q = c.q_cmd.copy()
    for i in range(1, 200):
        c.update(pose(), True, DT, _head_facing(i * 0.01, np.sin(i * 0.05)))
    assert np.allclose(c.q_cmd, q, atol=1e-6)


def test_heading_stays_fixed_while_grip_is_held():
    c = TeleopCore(HOME, Mapping(position_scale=1.0, robot_forward_yaw=0.0))
    start = c.target[:3, 3].copy()
    c.update(pose(), True, DT, _head_facing(0.0))
    forward = _quest_forward(0.0)
    for i, T in enumerate(_world_ramp(forward)):
        c.update(T, True, DT, _head_facing(i * 0.01))  # head turns during the push
    for _ in range(125):
        c.update(pose(*(forward * 0.05)), True, DT, _head_facing(1.5))
    assert np.allclose(fk(c.q_cmd)[:3, 3] - start, [0.05, 0, 0], atol=1e-3)


def test_new_grip_uses_new_heading():
    c = TeleopCore(HOME, Mapping(position_scale=1.0, robot_forward_yaw=0.0))
    c.update(pose(), True, DT, _head_facing(0.0))
    c.update(pose(), False, DT, _head_facing(1.2))
    start = c.target[:3, 3].copy()
    c.update(pose(), True, DT, _head_facing(1.2))
    forward = _quest_forward(1.2)
    for T in _world_ramp(forward):
        c.update(T, True, DT, _head_facing(1.2))
    for _ in range(125):
        c.update(pose(*(forward * 0.05)), True, DT, _head_facing(1.2))
    assert np.allclose(fk(c.q_cmd)[:3, 3] - start, [0.05, 0, 0], atol=1e-3)


def test_grip_motion_reports_robot_axes_after_turning():
    c = TeleopCore(HOME, Mapping(position_scale=1.0, robot_forward_yaw=np.radians(135)))
    head = _head_facing(1.4)
    c.update(pose(), True, DT, head)
    forward = _quest_forward(1.4)
    for T in _world_ramp(forward):
        c.update(T, True, DT, head)
    for _ in range(125):
        c.update(pose(*(forward * 0.05)), True, DT, head)
    assert np.allclose(c.grip_motion(), [0.05, 0, 0], atol=1e-3)


DOWN = np.array([0.0, 0.0, -1.0])


def _tool_z(c):
    return fk(c.q_cmd, c.map.tcp_offset)[:3, 2]


def test_pointing_down_keeps_yaw_and_handles_straight_up():
    tilted = matrix_from_rotvec(np.array([0.0, 0.0, 0.7])) @ matrix_from_rotvec(np.array([np.pi - 0.4, 0.0, 0.0]))
    R = pointing_down(tilted)
    assert np.allclose(R[:, 2], DOWN) and np.allclose(R.T @ R, np.eye(3))
    # Only a tilt: tool x keeps its heading in the horizontal plane.
    x0, x1 = tilted[:2, 0] / np.linalg.norm(tilted[:2, 0]), R[:2, 0] / np.linalg.norm(R[:2, 0])
    assert np.allclose(x0, x1, atol=1e-9)
    assert np.allclose(pointing_down(np.eye(3))[:, 2], DOWN)
    flat_down = matrix_from_rotvec(np.array([np.pi, 0.0, 0.0]))
    assert np.allclose(pointing_down(flat_down), flat_down)


def test_tool_down_turns_down_by_itself_in_place():
    c = TeleopCore(LIVE_Q)
    assert _tool_z(c) @ DOWN < 0.99
    start = fk(c.q_cmd)[:3, 3].copy()
    c.set_tool_down(True)
    run(c, [pose()] * int(6 / DT), clutch=False)
    assert _tool_z(c) @ DOWN > 0.9999
    assert np.allclose(fk(c.q_cmd)[:3, 3], start, atol=1e-3)
    # Once down it stays put.
    q = c.q_cmd.copy()
    run(c, [pose()] * 50, clutch=False)
    assert np.allclose(c.q_cmd, q, atol=1e-6)


def test_tool_down_holds_without_a_live_controller():
    c = TeleopCore(LIVE_Q)
    q = c.q_cmd.copy()
    c.set_tool_down(True)
    run(c, [None] * 50, clutch=False)
    run(c, [None] * 50, clutch=True)
    assert np.array_equal(c.q_cmd, q)


def test_tool_down_ignores_hand_tilt_but_follows_twist():
    c = TeleopCore(LIVE_Q)
    c.set_tool_down(True)
    run(c, [pose()] * int(6 / DT), clutch=False)
    R0 = fk(c.q_cmd)[:3, :3].copy()
    start = fk(c.q_cmd)[:3, 3].copy()
    # Tilt plus a 0.3 rad twist about vertical, while moving 4 cm (2 cm at scale 0.5).
    hand = [pose(0.04 * i / 250, 0, 0, rot=(0.5 * i / 250, 0.0, 0.3 * i / 250)) for i in range(251)]
    run(c, [pose()] + hand + [hand[-1]] * 250)
    R = fk(c.q_cmd)[:3, :3]
    assert R[:, 2] @ DOWN > 0.9999
    x0, x1 = R0[:, 0], R[:, 0]
    yaw = np.arctan2(np.cross(x0, x1)[2], x0 @ x1)
    assert abs(yaw - yaw_of(np.array([0.5, 0.0, 0.3]))) < 5e-3 and yaw > 0.25
    assert np.allclose(fk(c.q_cmd)[:3, 3] - start, [0.02, 0, 0], atol=2e-3)


def test_tool_down_turns_at_the_angular_limit():
    c = TeleopCore(LIVE_Q, limits=Limits(max_angular_speed=0.2))
    c.set_tool_down(True)
    prev = fk(c.q_cmd)[:3, :3]
    c.update(pose(), True, DT)  # engage
    for _ in range(20):
        c.update(pose(), True, DT)
        R = c.target[:3, :3]
        assert np.linalg.norm(rotvec_from_matrix(R @ prev.T)) <= 0.2 * DT + 1e-9
        prev = R


def test_jog_aligns_to_tool_down():
    c = TeleopCore(LIVE_Q)
    c.set_tool_down(True)
    _jog_for(c, np.array([0.0, 0.0, 0.001]), seconds=6.0)
    assert _tool_z(c) @ DOWN > 0.9999


def test_toggling_tool_down_off_mid_grip_does_not_jump():
    c = TeleopCore(LIVE_Q)
    c.set_tool_down(True)
    run(c, [pose()] * int(3 / DT))
    q = c.q_cmd.copy()
    c.set_tool_down(False)
    c.update(pose(0.0, 0, 0, rot=(0.0, 0.4, 0.0)), True, DT)  # re-engages here
    assert np.array_equal(c.q_cmd, q)


def test_each_stick_click_toggles_tool_down():
    c = TeleopCore(LIVE_Q)
    assert c.click(True) is True and c.tool_down
    assert c.click(True) is None and c.tool_down      # still held: no change
    assert c.click(False) is None and c.tool_down     # released: stays on
    assert c.click(True) is False and not c.tool_down  # second press: off
    c.click(False)
    q = c.q_cmd.copy()
    run(c, [pose()] * 50, clutch=False)
    assert np.array_equal(c.q_cmd, q)  # off means it no longer turns


def _lowest_gripper_point(c):
    flange = fk(c.q_cmd)
    pts = flange[:3, :3] @ gripper_points(0.17).T + flange[:3, 3:4]
    return pts[2].min()


def test_fingertips_stop_at_the_floor_and_slide_along_it():
    c = TeleopCore(LIVE_Q)
    c.set_tool_down(True)
    run(c, [pose()] * int(6 / DT), clutch=False)
    floor = _lowest_gripper_point(c) - 0.03
    c = TeleopCore(c.q_cmd, limits=Limits(min_z=floor, gripper_length=0.17))
    start = fk(c.q_cmd)[:3, 3].copy()
    # Push the hand 20 cm down (10 cm at scale 0.5), then 4 cm sideways while still pushing.
    path = ramp((0, 0, 0), (0, 0, -0.2), 2.0) + ramp((0, 0, -0.2), (0, 0.04, -0.2), 1.0)
    run(c, path + [path[-1]] * 250)
    assert abs(_lowest_gripper_point(c) - floor) < 2e-3
    assert np.allclose(fk(c.q_cmd)[:3, 3][:2] - start[:2], [0.0, 0.02], atol=2e-3)


def test_jog_down_stops_at_the_floor_with_the_gripper():
    c = TeleopCore(LIVE_Q)
    c.set_tool_down(True)
    run(c, [pose()] * int(6 / DT), clutch=False)
    floor = _lowest_gripper_point(c) - 0.01
    c = TeleopCore(c.q_cmd, limits=Limits(min_z=floor, gripper_length=0.17))
    _jog_for(c, np.array([0.0, 0.0, 0.02]), seconds=3.0)  # A: along tool z, straight down
    assert _lowest_gripper_point(c) > floor - 2e-3


def test_floor_counts_the_elbow_and_wrist():
    c = TeleopCore(LIVE_Q, limits=Limits(min_z=0.0, gripper_length=0.17))
    assert c._body_low(LIVE_Q) <= min(f[2, 3] for f in _chain(LIVE_Q)[2:6])
    # A floor just under the current lowest point: no step may go below it.
    low = c._body_low(c.q_cmd)
    c = TeleopCore(LIVE_Q, limits=Limits(min_z=low - 0.001, gripper_length=0.17))
    run(c, ramp((0, 0, 0), (0, 0, -0.2), 2.0))
    assert c._body_low(c.q_cmd) > low - 0.001 - 0.004


def test_turning_tool_down_lifts_to_keep_the_fingertips_clear():
    c = TeleopCore(LIVE_Q)
    floor = _lowest_gripper_point(c) - 0.005
    c = TeleopCore(LIVE_Q, limits=Limits(min_z=floor, gripper_length=0.17))
    c.set_tool_down(True)
    run(c, [pose()] * int(6 / DT), clutch=False)
    assert _tool_z(c) @ DOWN > 0.999
    assert _lowest_gripper_point(c) > floor - 3e-3
