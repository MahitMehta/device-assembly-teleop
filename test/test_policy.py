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
    assert c.status == "workspace limit" and c.limit_detail == "box +x"


def test_follow_reports_the_floor():
    lim = Limits(max_linear_speed=1.0, max_joint_speed=3.0, box_half_extent=1.0, min_z=0.01, gripper_length=0.18)
    c = TeleopCore(HOME, Mapping(), lim)
    goal = shifted(c.target, dz=-1.0)
    for _ in range(400):
        c.follow(goal, DT)
    assert c.status == "workspace limit" and c.limit_detail == "floor"


def test_auto_stop_waits_for_release_then_stillness():
    from quest_teleop.utils.policy_runner import AutoStop
    T0 = fk(HOME)
    still = chunk_from(T0, [shifted(T0, dx=0.001)] * 15)
    moving = chunk_from(T0, [shifted(T0, dx=0.01 * (i + 1)) for i in range(15)])
    a = AutoStop(seconds=2.0, mm=3.0)
    assert not a.plan(still, 8, 0.0) and not a.plan(still, 8, 5.0)   # never closed: keeps going
    a.gripper(True)
    assert not a.plan(still, 8, 6.0)                                 # closed, not released yet
    a.gripper(False)
    assert not a.plan(still, 8, 7.0)
    assert not a.plan(moving, 8, 8.0)                                # motion resets the timer
    assert not a.plan(still, 8, 8.8) and not a.plan(still, 8, 10.0)
    assert a.plan(still, 8, 10.8)                                    # 2 s of stillness after release
    a.reset()
    assert not a.plan(still, 8, 20.0)


def test_wrench_filter_zeroes_then_averages_the_last_tenth_of_a_second():
    from quest_teleop.utils.policy_runner import WrenchFilter
    f = WrenchFilter()
    assert f.value(0.0) is None and not f.zero(0.0)          # nothing yet
    for i in range(500):                                       # 1 s at 500 Hz, 2 N offset on x
        f.add(i * 0.002, [2.0, 0, 0, 0, 0, 0])
    assert f.zero(0.998) and np.allclose(f.bias, [2, 0, 0, 0, 0, 0])
    for i in range(500, 600):                                  # then a 20 N push on z
        f.add(i * 0.002, [2.0, 0, 20.0, 0, 0, 0.5])
    v = f.value(1.198)
    assert np.allclose(v, [0, 0, 20.0, 0, 0, 0.5])
    assert f.value(1.5) is None                                # stream stopped: stale


def test_tilt_of_reports_lean_in_tool_frame():
    from quest_teleop.utils.core import matrix_from_rotvec
    from quest_teleop.utils.policy_runner import tilt_of
    down = matrix_from_rotvec(np.array([np.pi, 0.0, 0.0]))
    assert tilt_of(down)[0] < 1e-6 and np.allclose(tilt_of(down)[1], 0)
    # turn about the tool's own y axis: the fingertips lean along tool x
    R = down @ matrix_from_rotvec(np.array([0.0, 0.1, 0.0]))
    deg, lean = tilt_of(R)
    assert abs(deg - np.degrees(0.1)) < 1e-6 and abs(abs(lean[0]) - 1) < 1e-6


def _wrench_at(r, F):
    r, F = np.asarray(r, float), np.asarray(F, float)
    return np.r_[F, np.cross(r, F)]


def _settle(c, w, R=np.eye(3), ticks=125, dt=0.008):
    for _ in range(ticks):
        c.filter(w, dt)
        c.step(R, dt)


def test_compliance_slides_away_from_an_off_axis_contact():
    from quest_teleop.utils.policy_runner import ContactCompliance
    c = ContactCompliance(speed=0.002, contact_z=0.19, min_r=0.0015, edge_max_side=10.0)
    c.filter(np.zeros(6), 0.008)                    # zero: free air
    _settle(c, _wrench_at([0.004, 0.0, 0.19], [0.5, 0.0, -10.0]))   # rim contact at +x, a little friction
    assert c.contact is not None and abs(c.contact[0] - 0.004) < 3e-4 and abs(c.contact[1]) < 3e-4
    assert c.offset[0] < -0.001 and abs(c.offset[1]) < 1e-4       # moved toward -x, the hole


def test_compliance_ignores_the_sensor_sign_and_centred_or_light_contact():
    from quest_teleop.utils.policy_runner import ContactCompliance
    c = ContactCompliance(speed=0.002, contact_z=0.19, min_r=0.0015, edge_max_side=10.0)
    c.filter(np.zeros(6), 0.008)
    _settle(c, -_wrench_at([0.0, -0.004, 0.19], [0.0, 0.0, -10.0]))   # flipped convention, contact at -y
    assert c.offset[1] > 0.001
    c = ContactCompliance(speed=0.002, contact_z=0.19, min_r=0.0015, edge_max_side=10.0); c.filter(np.zeros(6), 0.008)
    _settle(c, _wrench_at([0.0005, 0.0, 0.19], [0.0, 0.0, -10.0]))   # centred: inside min_r
    assert np.allclose(c.offset, 0)
    c = ContactCompliance(speed=0.002, contact_z=0.19, min_r=0.0015, edge_max_side=10.0); c.filter(np.zeros(6), 0.008)
    _settle(c, _wrench_at([0.004, 0.0, 0.19], [0.0, 0.0, -1.0]))    # too light to count
    assert np.allclose(c.offset, 0) and c.contact is None


def test_compliance_caps_offset_rebases_and_flags_pushing():
    from quest_teleop.utils.policy_runner import ContactCompliance
    c = ContactCompliance(speed=0.002, contact_z=0.19, min_r=0.0015, edge_max_side=10.0, force_gain=0.0, max_offset=0.003, push_limit=20.0)
    c.filter(np.zeros(6), 0.008)
    _settle(c, _wrench_at([0.004, 0.0, 0.19], [0.0, 0.0, -25.0]), ticks=1000)
    assert abs(np.linalg.norm(c.offset) - 0.003) < 1e-9 and c.pushing
    assert abs(np.linalg.norm(c.total) - 0.003) < 1e-9
    c.rebase(np.array([-0.002, 0.0]))
    assert np.allclose(c.offset, [-0.001, 0.0])
    _settle(c, _wrench_at([0.004, 0.0, 0.19], [0.0, 0.0, -25.0]), ticks=1000)   # still capped in total
    assert abs(np.linalg.norm(c.total) - 0.003) < 1e-9 and np.allclose(c.offset, [-0.001, 0.0])


def test_compliance_turns_tool_frame_motion_into_base():
    from quest_teleop.utils.core import matrix_from_rotvec
    from quest_teleop.utils.policy_runner import ContactCompliance
    R = matrix_from_rotvec(np.array([np.pi, 0.0, 0.0]))          # tool down: tool y = base -y
    c = ContactCompliance(speed=0.002, contact_z=0.19, min_r=0.0015, edge_max_side=10.0); c.filter(np.zeros(6), 0.008)
    _settle(c, _wrench_at([0.0, 0.004, 0.19], [0.0, 0.0, -10.0]), R=R)   # contact at tool +y
    assert c.offset[1] > 0.001                                   # away = tool -y = base +y


def _press_test_compliance(**kw):
    from quest_teleop.utils.policy_runner import ContactCompliance
    c = ContactCompliance(**kw)    # defaults, as set from the press test
    c.filter(np.zeros(6), 0.008)
    return c


def test_compliance_defaults_reproduce_the_press_test():
    # Readings from the arm (tool frame; tool -y = robot forward there). Front edge, then back edge.
    for F, tau, sign in (([0.5, -1.5, -10.0], [0.27, -0.05, 0.0], -1), ([-0.3, 2.4, -8.6], [-0.23, -0.05, 0.0], +1)):
        c = _press_test_compliance(speed=0.002)   # the edge term is off by default now
        _settle(c, np.r_[F, tau])
        assert c.contact is not None and np.sign(c.contact[1]) == sign     # front: tool -y; back: tool +y
        assert np.sign(c.offset[1]) == -sign                               # slides away from that edge


def test_compliance_defaults_follow_sideways_force():
    c = _press_test_compliance()
    _settle(c, np.r_[[0.0, -6.5, -1.6], [0.82, 0.0, 0.0]])              # pushed forward (tool -y)
    assert c.offset[1] < -0.001


def test_contact_descent_slows_backs_off_and_releases():
    from quest_teleop.utils.policy_runner import contact_descent
    dt = 0.008
    z, pz, st = contact_descent(0.090, 0.100, -1.0, 3.0, False, None, dt)          # free: as planned
    assert z == 0.090 and st == ""
    z, pz, st = contact_descent(0.090, 0.100, -5.0, 3.0, False, None, dt)          # touching: 2 mm/s
    assert abs(z - (0.100 - 0.002 * dt)) < 1e-12 and st == "contact-slow"
    target, push_z = 0.100, None
    for _ in range(200):                                                          # pushing: rise 0.5 mm, hold
        target, push_z, st = contact_descent(0.090, target, -15.0, 3.0, True, push_z, dt)
    assert st == "PUSH-BACKOFF" and abs(target - 0.1005) < 1e-9
    z, _, _ = contact_descent(0.120, 0.100, -15.0, 3.0, True, 0.100, dt)          # policy lifting: allowed
    assert z == 0.120


def test_compliance_uses_force_not_edge_with_real_sideways_contact():
    # From the first --compliance on run (tool frame): ~5 N sideways acting ~0.17 m down, 2.5 N push.
    # The edge estimate read >100 mm off and slid against the force; now only the force term acts.
    c = _press_test_compliance()
    _settle(c, np.r_[[-5.0, -0.6, -2.5], [0.08, -0.87, -0.48]])
    assert c.path == "force" and c.offset[0] < -0.001                      # along tool -x, with the force
    c = _press_test_compliance()
    _settle(c, np.r_[[-3.2, 1.5, -6.5], [-0.08, -0.61, -0.36]])
    assert c.path == "force" and c.offset[0] < 0


def test_compliance_keeps_its_zero_when_a_run_starts_holding():
    from quest_teleop.utils.policy_runner import ContactCompliance
    c = ContactCompliance()
    c.filter(np.zeros(6), 0.008)                     # zero in free air
    c.reset(keep_zero=True)
    _settle(c, np.r_[[0.0, 0.0, -10.0], [0.0, 0.0, 0.0]])   # run starts while pressing 10 N
    assert abs(c.w[2] + 10.0) < 0.1                   # still reads the press
    c.reset(keep_zero=False)
    _settle(c, np.r_[[0.0, 0.0, -10.0], [0.0, 0.0, 0.0]])
    assert abs(c.w[2]) < 1e-9                         # re-zeroed


def test_spiral_search_starts_on_a_stall_spirals_and_finds_the_hole():
    from quest_teleop.utils.policy_runner import SpiralSearch
    s, dt, t, z = SpiralSearch(pitch=0.001, speed=0.005, max_r=0.004), 0.008, 0.0, 0.050
    pos = np.zeros(2)
    for _ in range(int(0.6 / dt)):                    # pressing 10 N, 1 N sideways, not moving down
        pos += s.update(t, dt, -10.0, 1.0, z) * dt; t += dt
        if s.state != "idle":
            break
    assert s.state == "searching" and 0.45 < t < 0.55 and np.allclose(pos, 0)
    radii = []
    for _ in range(int(3.0 / dt)):
        pos += s.update(t, dt, -6.0, 1.0, z) * dt; t += dt
        radii.append(np.linalg.norm(pos))
    assert s.state == "searching" and abs(np.linalg.norm(pos) - s.r) < 1e-6
    assert 0.001 < s.r < 0.004 and radii[-1] > radii[len(radii) // 2]   # growing outward
    s.update(t, dt, -2.0, 0.5, z - 0.0015)            # the piece drops in
    assert s.state == "found" and np.allclose(s.update(t, dt, -2.0, 0.5, z - 0.002), 0)


def test_spiral_search_ignores_sideways_cues_and_progress_and_gives_up():
    from quest_teleop.utils.policy_runner import SpiralSearch
    dt = 0.008
    s = SpiralSearch()
    for i in range(200):                              # sideways force present: that is the slide's job
        s.update(i * dt, dt, -10.0, 4.0, 0.05)
    assert s.state == "idle"
    s = SpiralSearch()
    for i in range(200):                              # still going down 2 mm/s: not stalled
        s.update(i * dt, dt, -10.0, 1.0, 0.05 - 0.002 * i * dt)
    assert s.state == "idle"
    s, t = SpiralSearch(max_r=0.002), 0.0
    for _ in range(int(20 / dt)):
        s.update(t, dt, -6.0, 1.0, 0.05); t += dt
    assert s.state == "failed"


def test_spiral_search_holds_the_push_near_its_force():
    from quest_teleop.utils.policy_runner import SpiralSearch
    s = SpiralSearch(force=6.0, z_speed=0.001)
    assert s.z_target(0.05, -10.0, 0.01) > 0.05      # too hard: rise
    assert s.z_target(0.05, -1.0, 0.01) < 0.05       # too light: sink
    assert s.z_target(0.05, -4.0, 0.01) == 0.05


def test_edge_term_is_off_by_default():
    c = _press_test_compliance()
    _settle(c, np.r_[[0.5, -1.5, -10.0], [0.27, -0.05, 0.0]])   # front-edge press
    assert c.contact is not None and np.allclose(c.offset, 0)


def test_edge_toward_mode_moves_toward_the_touching_edge():
    # Front-edge press from the press test (tool -y = forward there): toward = forward.
    c = _press_test_compliance(speed=0.002, edge_dir=1.0)
    _settle(c, np.r_[[0.5, -1.5, -10.0], [0.27, -0.05, 0.0]])
    assert c.path == "edge" and c.offset[1] < -0.001
    c = _press_test_compliance(speed=0.002, edge_dir=1.0)     # gated off with real sideways force
    _settle(c, np.r_[[-5.0, -0.6, -2.5], [0.08, -0.87, -0.48]])
    assert c.path == "force"


def test_policy_anchor_must_match_the_training_frame():
    # Demos: action = inv(driver pose) @ command, with command in nominal-fk terms. With the two
    # kinematic models apart by C, holding still trains the action inv(C). Executed from the fk pose
    # it moves the arm by inv(C) every plan; from the driver pose it stays put.
    from quest_teleop.utils.core import matrix_from_rotvec
    fk_pose = np.eye(4); fk_pose[:3, :3] = matrix_from_rotvec(np.array([np.pi, 0.0, 0.3])); fk_pose[:3, 3] = [-0.38, 0.0, 0.35]
    C = np.eye(4); C[:3, :3] = matrix_from_rotvec(np.radians([0.17, -0.25, 0.0])); C[:3, 3] = [0.0004, -0.001, -0.0012]
    driver = fk_pose @ C
    still_action = np.linalg.inv(driver) @ fk_pose               # what the demos teach for "hold still"
    assert np.linalg.norm(still_action[:2, 3]) > 0.001           # ~1 mm, as measured in the dataset
    assert np.allclose(driver @ still_action, fk_pose)            # anchored on the driver pose: no motion
    moved = fk_pose @ still_action                                # anchored on fk: a step every plan
    assert np.linalg.norm(moved[:3, 3] - fk_pose[:3, 3]) > 0.001


def test_dataset_nominal_fk_matches_the_robot_nodes():
    # policy/dit_policy/geometry.ur5e_fk labels datasets; core.fk runs the robot. They must agree.
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "policy"))
    pytest = __import__("pytest")
    geometry = pytest.importorskip("dit_policy.geometry")
    from quest_teleop.utils.core import fk
    rng = np.random.default_rng(0)
    qs = rng.uniform(-np.pi, np.pi, (20, 6))
    batch = geometry.ur5e_fk(qs)
    for q, T in zip(qs, batch):
        assert np.allclose(T, fk(q), atol=1e-12)


def _descending_chunk(dz_per_step, t0=0.0, dt=0.1, n=16, z0=0.3):
    from quest_teleop.utils.policy_runner import Chunk
    poses = np.repeat(np.eye(4)[None], n, axis=0)
    poses[:, 2, 3] = z0 + dz_per_step * np.arange(n)
    return Chunk(t0=t0, dt=dt, poses=poses, grip=np.zeros(n))


def test_force_mode_settings_select_axes_and_push_down():
    from quest_teleop.utils.policy_runner import force_mode_settings
    sel, w = force_mode_settings("x,y", 5.0)
    assert sel == [True, True, False, False, False, False] and w == [0.0] * 6
    sel, w = force_mode_settings(" X, y ,z", 5.0)
    assert sel[:3] == [True, True, True] and w[2] == -5.0     # pushes down on the part
    with pytest.raises(ValueError):
        force_mode_settings("x,w", 5.0)
    with pytest.raises(ValueError):
        force_mode_settings("", 5.0)


def test_force_mode_override_commands_the_measured_pose_on_compliant_axes():
    from quest_teleop.utils.policy_runner import force_mode_override
    desired, measured = np.eye(4), np.eye(4)
    desired[:3, 3] = [0.1, 0.2, 0.3]
    measured[:3, 3] = [0.11, 0.19, 0.25]
    measured[:3, :3] = matrix_from_rotvec(np.array([0.0, 0.05, 0.0]))
    out = force_mode_override(desired, measured, [True, True, False, False, False, False])
    assert np.allclose(out[:3, 3], [0.11, 0.19, 0.3]) and np.allclose(out[:3, :3], np.eye(3))
    out = force_mode_override(desired, measured, [True, True, True, True, True, False])
    assert np.allclose(out[:3, 3], measured[:3, 3]) and np.allclose(out[:3, :3], measured[:3, :3])
    assert np.allclose(desired[:3, 3], [0.1, 0.2, 0.3])        # not changed in place


def test_force_mode_gate_enters_holding_and_descending_only():
    from quest_teleop.utils.policy_runner import ForceModeGate
    g = ForceModeGate(enter=0.001, exit=0.001, lookahead=0.3)
    down = _descending_chunk(-0.002)                            # 6 mm down over 0.3 s
    assert abs(g.planned_dz(down, 0.0) + 0.006) < 1e-9
    assert g.update(False, g.planned_dz(down, 0.0)) is None     # descending, but not holding
    assert g.update(True, 0.0) is None                          # holding, still
    assert g.update(True, -0.0005) is None                      # holding, barely moving
    assert g.update(True, g.planned_dz(down, 0.0)) is True and g.engaged
    assert g.update(True, 0.0) is None and g.engaged            # pausing on the rim keeps it
    assert g.update(True, -0.004) is None and g.engaged


def test_force_mode_gate_leaves_on_release_or_lift():
    from quest_teleop.utils.policy_runner import ForceModeGate
    g = ForceModeGate()
    g.update(True, -0.005)
    assert g.update(False, 0.0) is False and "opened" in g.reason
    g.update(True, -0.005)
    assert g.update(True, g.planned_dz(_descending_chunk(+0.002), 0.0)) is False and "lifting" in g.reason
    g.update(True, -0.005)
    g.reset()
    assert not g.engaged


def test_force_mode_gate_can_wait_for_contact():
    from quest_teleop.utils.policy_runner import ForceModeGate
    g = ForceModeGate(min_fz=3.0)
    assert g.update(True, -0.005, fz=1.0) is None               # descending in free air
    assert g.update(True, -0.005, fz=-4.0) is True              # touching (either sign)


def test_force_mode_gate_near_the_end_of_a_chunk_holds_the_last_target():
    from quest_teleop.utils.policy_runner import ForceModeGate
    g = ForceModeGate(lookahead=0.3)
    c = _descending_chunk(-0.002)
    assert abs(g.planned_dz(c, c.end())) < 1e-12               # held at the end: no motion, no switch
