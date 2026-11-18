"""Unit tests for S1a's human-entry geometry. No Isaac Sim, no torch.

A crossing rate computed by a script outside the repository cannot be
reproduced. These tests pin down the formulas measure_crossing_rate.py
depends on.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from generation.envs.s1a.mdp.entry_geometry import (
    EntrySpeedWindow,
    check_stop_angles,
    crossing_rate,
    crossing_time,
    crossing_x,
    derive_wait_x,
    derive_wait_y,
    facing_quat_xyzw,
    latch_walk_end,
    lateral_stop_fraction,
    reaches_lane,
    relative_gap_max,
    trigger_rate,
    travel_facing_yaw_deg,
    trigger_time,
    triggers_within_episode,
    yawed_axis_inertia,
)


def test_crossing_x_moves_monotonically_across_the_angle_range():
    # cot(angle) is strictly decreasing on (0, pi), so crossing_x = wait_x -
    # wait_y*cot(angle) is strictly increasing as angle sweeps the catalogue's
    # full [30, 150] deg range -- an oblique entry toward the shallow end
    # pulls the crossing point backward, toward the wide end it pushes it
    # forward, and 90 deg (pure lateral) sits exactly at wait_x.
    angles_deg = np.linspace(30.0, 150.0, 199)  # odd count lands exactly on 90.0
    angles_rad = np.radians(angles_deg)
    xs = crossing_x(wait_x=5.0, wait_y=2.0, angle_rad=angles_rad)

    assert np.all(np.diff(xs) > 0)
    # 90 deg is the crossover point where the lateral offset contributes 0.
    idx_90 = np.argmin(np.abs(angles_deg - 90.0))
    assert angles_deg[idx_90] == pytest.approx(90.0, abs=1e-9)
    assert xs[idx_90] == pytest.approx(5.0, abs=1e-9)


def test_crossing_x_matches_hand_calculation_at_30_and_150_degrees():
    wait_x, wait_y = 5.0, 2.0
    cot30 = math.cos(math.radians(30)) / math.sin(math.radians(30))  # sqrt(3)
    assert crossing_x(wait_x, wait_y, math.radians(30)) == pytest.approx(
        wait_x - wait_y * cot30
    )
    # 150 deg is 30 deg's mirror: same |cot|, opposite sign.
    assert crossing_x(wait_x, wait_y, math.radians(150)) == pytest.approx(
        wait_x + wait_y * cot30
    )


def test_derive_wait_y_clears_the_stop_band():
    # The property the waiting offset needs: a non-negative
    # reach/margin sum can only push the waiting offset further out, never
    # closer than the stop band alone.
    wait_y = derive_wait_y(
        stop_band_clearance=1.764,
        robot_lateral_reach=0.21,
        dummy_lateral_reach=0.21,
        spawn_margin_y=0.5,
    )
    assert wait_y == pytest.approx(1.764 + 0.21 + 0.21 + 0.5)
    assert wait_y > 1.764


def test_derive_wait_y_is_exactly_the_stop_band_with_zero_reach_and_margin():
    assert derive_wait_y(1.764, 0.0, 0.0, 0.0) == pytest.approx(1.764)


def test_derive_wait_y_residual_drift_margin_defaults_to_zero():
    without = derive_wait_y(1.764, 0.21, 0.21, 0.5)
    explicit_zero = derive_wait_y(1.764, 0.21, 0.21, 0.5, residual_drift_margin=0.0)
    assert without == pytest.approx(explicit_zero)


def test_derive_wait_y_residual_drift_margin_adds_on_top():
    base = derive_wait_y(1.764, 0.21, 0.21, 0.5)
    padded = derive_wait_y(1.764, 0.21, 0.21, 0.5, residual_drift_margin=0.6)
    assert padded == pytest.approx(base + 0.6)


def test_derive_wait_y_human_region_radius_defaults_to_zero_but_adds_on_top():
    # min_distance_human reports centre_distance - radius, so
    # the real centre-to-centre clearance needed is stop_band + radius, not
    # stop_band alone.
    base = derive_wait_y(1.764, 0.21, 0.21, 0.5)
    with_radius = derive_wait_y(1.764, 0.21, 0.21, 0.5, human_region_radius=0.1)
    assert with_radius == pytest.approx(base + 0.1)


def test_derive_wait_x_matches_hand_calculation():
    wait_y = 2.474
    cot30 = math.cos(math.radians(30)) / math.sin(math.radians(30))
    wait_x = derive_wait_x(spawn_margin_x=0.5, wait_y=wait_y, min_angle_rad=math.radians(30))
    assert wait_x == pytest.approx(0.5 + wait_y * cot30)


def test_derive_wait_x_trigger_distance_max_defaults_to_zero_but_adds_on_top():
    wait_y = 2.474
    cot30 = math.cos(math.radians(30)) / math.sin(math.radians(30))
    base = derive_wait_x(spawn_margin_x=0.5, wait_y=wait_y, min_angle_rad=math.radians(30))
    padded = derive_wait_x(
        spawn_margin_x=0.5, wait_y=wait_y, min_angle_rad=math.radians(30), trigger_distance_max=4.0
    )
    assert padded == pytest.approx(base + 4.0)
    assert base == pytest.approx(0.5 + wait_y * cot30)


def test_derive_wait_x_with_trigger_distance_max_prevents_a_step_zero_trigger():
    # Without trigger_distance_max, x_cross's minimum
    # (at the shallowest angle) can land inside the robot's own spawn
    # randomization range, so the along-track trigger is already satisfied
    # before the robot has moved at all. Worst case: robot spawns at the
    # closest edge of its own range (+0.5 m here), trigger_distance drawn at
    # the top of its range (4.0 m) -- with the delay term, the initial
    # along-track distance to the crossing point must still exceed
    # trigger_distance.
    spawn_margin_x = 0.5
    trigger_distance_max = 4.0
    wait_y = 3.0
    min_angle_rad = math.radians(30)
    wait_x = derive_wait_x(spawn_margin_x, wait_y, min_angle_rad, trigger_distance_max)
    x_cross_min = crossing_x(wait_x, wait_y, min_angle_rad)  # worst case: shallowest angle
    worst_case_robot_spawn_x = spawn_margin_x  # robot spawns as close as its own range allows
    dist0 = x_cross_min - worst_case_robot_spawn_x
    # Exactly equal at this exact worst case (by construction of the
    # formula) -- floating point, so compare with a tolerance rather than a
    # strict >=.
    assert dist0 == pytest.approx(trigger_distance_max)
    assert dist0 >= trigger_distance_max - 1e-9


def test_trigger_time_zero_when_already_inside_trigger_distance():
    # x_cross=5.0, robot at 4.5 -> dist0=0.5, under trigger_distance=1.0.
    assert trigger_time(x_cross=5.0, robot_spawn_x=4.5, trigger_distance=1.0,
                         robot_walk_speed=1.0) == pytest.approx(0.0)


def test_trigger_time_scales_with_remaining_distance_and_speed():
    t = trigger_time(x_cross=5.0, robot_spawn_x=0.0, trigger_distance=1.0, robot_walk_speed=1.0)
    assert t == pytest.approx(4.0)  # (5.0 - 1.0) / 1.0
    faster = trigger_time(x_cross=5.0, robot_spawn_x=0.0, trigger_distance=1.0,
                           robot_walk_speed=2.0)
    assert faster == pytest.approx(2.0)


def test_trigger_time_none_once_robot_has_already_passed_the_crossing():
    # Matches the live along-track guard in
    # S1aStraightWalkCommand._advance_dummy (dist_to_crossing >= 0.0): a
    # robot spawned past x_cross can never trigger the entry in this model.
    assert trigger_time(x_cross=5.0, robot_spawn_x=6.0, trigger_distance=1.0,
                         robot_walk_speed=1.0) is None


def test_crossing_time_hand_calculation():
    assert crossing_time(wait_y=2.0, human_entry_speed=1.0,
                          angle_rad=math.radians(90)) == pytest.approx(2.0)
    # Same wait_y, half the sideways speed component (30 deg: sin=0.5) takes
    # twice as long.
    assert crossing_time(wait_y=2.0, human_entry_speed=1.0,
                          angle_rad=math.radians(30)) == pytest.approx(4.0)


def test_reaches_lane_true_when_total_time_fits_the_episode():
    # trigger_time=4.0 + crossing_time=2.0 = 6.0s <= 20.0s episode.
    assert reaches_lane(
        wait_x=5.0, wait_y=2.0, robot_walk_speed=1.0, human_entry_speed=1.0,
        angle_rad=math.radians(90), trigger_distance=1.0, episode_length_s=20.0,
        robot_spawn_x=0.0,
    ) is True


def test_reaches_lane_false_when_total_time_exceeds_the_episode():
    # Same draw, shorter episode: 6.0s > 5.0s.
    assert reaches_lane(
        wait_x=5.0, wait_y=2.0, robot_walk_speed=1.0, human_entry_speed=1.0,
        angle_rad=math.radians(90), trigger_distance=1.0, episode_length_s=5.0,
        robot_spawn_x=0.0,
    ) is False


def test_reaches_lane_false_when_robot_already_passed_the_crossing():
    assert reaches_lane(
        wait_x=5.0, wait_y=2.0, robot_walk_speed=1.0, human_entry_speed=1.0,
        angle_rad=math.radians(90), trigger_distance=1.0, episode_length_s=100.0,
        robot_spawn_x=6.0,
    ) is False


def test_triggers_within_episode_is_weaker_than_reaches_lane():
    # A draw that reaches the lane must also have triggered -- but the human
    # entry speed can be slow enough that trigger fires while the crossing
    # itself still runs past the episode's end (trigger rate and crossing
    # rate are different numbers).
    common = dict(
        wait_x=5.0, wait_y=2.0, robot_walk_speed=1.0, angle_rad=math.radians(90),
        trigger_distance=1.0, episode_length_s=4.5, robot_spawn_x=0.0,
    )
    assert triggers_within_episode(**common) is True  # trigger_time=4.0 <= 4.5
    assert reaches_lane(human_entry_speed=1.0, **common) is False  # +2.0s crossing > 4.5


def test_triggers_within_episode_false_once_robot_has_passed_the_crossing():
    assert triggers_within_episode(
        wait_x=5.0, wait_y=2.0, robot_walk_speed=1.0, angle_rad=math.radians(90),
        trigger_distance=1.0, episode_length_s=100.0, robot_spawn_x=6.0,
    ) is False


def test_trigger_rate_is_at_least_the_crossing_rate():
    common = dict(
        robot_walk_speed_range=(0.3, 1.2),
        angle_range_deg=(30.0, 150.0),
        trigger_distance_range=(1.0, 4.0),
        episode_length_s=20.0,
        robot_spawn_x_range=(-0.5, 0.5),
    )
    wait_y = 6.5
    wait_x = derive_wait_x(0.5, wait_y, math.radians(30))
    rng = np.random.default_rng(11)
    t_rate = trigger_rate(rng, 20000, wait_x=wait_x, wait_y=wait_y, **common)
    c_rate = crossing_rate(
        rng, 20000, wait_x=wait_x, wait_y=wait_y,
        human_entry_speed_range=(0.4, 1.6), **common,
    )
    assert t_rate >= c_rate


def test_crossing_rate_is_reproducible_with_a_fixed_seed():
    kwargs = dict(
        wait_x=5.0,
        wait_y=2.0,
        robot_walk_speed_range=(0.3, 1.2),
        human_entry_speed_range=(0.4, 1.6),
        angle_range_deg=(30.0, 150.0),
        trigger_distance_range=(1.0, 4.0),
        episode_length_s=20.0,
        robot_spawn_x_range=(-0.5, 0.5),
    )
    rate_a = crossing_rate(np.random.default_rng(42), 5000, **kwargs)
    rate_b = crossing_rate(np.random.default_rng(42), 5000, **kwargs)
    assert rate_a == rate_b
    assert 0.0 <= rate_a <= 1.0


def test_crossing_rate_increases_as_the_corridor_shrinks():
    # A tighter (smaller wait_y, hence smaller wait_x) corridor should let
    # strictly more draws reach the lane within the same episode budget --
    # the qualitative relationship the crossing-rate claim depends on.
    common = dict(
        robot_walk_speed_range=(0.3, 1.2),
        human_entry_speed_range=(0.4, 1.6),
        angle_range_deg=(30.0, 150.0),
        trigger_distance_range=(1.0, 4.0),
        episode_length_s=20.0,
        robot_spawn_x_range=(-0.5, 0.5),
    )
    small_wait_y = 2.0
    big_wait_y = 6.5
    small_wait_x = derive_wait_x(0.5, small_wait_y, math.radians(30))
    big_wait_x = derive_wait_x(0.5, big_wait_y, math.radians(30))

    rng = np.random.default_rng(7)
    rate_small = crossing_rate(rng, 20000, wait_x=small_wait_x, wait_y=small_wait_y, **common)
    rate_big = crossing_rate(rng, 20000, wait_x=big_wait_x, wait_y=big_wait_y, **common)

    assert rate_small > rate_big


# ---- which way the dummy faces: a turn about the vertical (S1a keeps +x, a turn of 0) ----

def _rotate(q, v):
    """v rotated by the xyzw quaternion q (the engine's convention)."""
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, float)
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def test_facing_zero_is_the_identity_s1a_has_always_written_to_the_bit():
    assert facing_quat_xyzw(0.0) == (0.0, 0.0, 0.0, 1.0)
    assert all(type(c) is float for c in facing_quat_xyzw(0.0))


def test_facing_180_is_exact_and_turns_plus_x_to_minus_x():
    q = facing_quat_xyzw(180.0)
    assert q == (0.0, 0.0, 1.0, 0.0)                 # no 6e-17 left in w
    assert _rotate(q, [1.0, 0.0, 0.0]) == pytest.approx([-1.0, 0.0, 0.0])


@pytest.mark.parametrize("yaw", [-135.0, -90.0, 30.0, 90.0, 150.0, 270.0])
def test_facing_is_an_upright_unit_turn_about_z(yaw):
    q = facing_quat_xyzw(yaw)
    assert q[0] == 0.0 and q[1] == 0.0                # about +z only: stays upright
    assert math.fsum(c * c for c in q) == pytest.approx(1.0)
    fwd = _rotate(q, [1.0, 0.0, 0.0])
    assert math.degrees(math.atan2(fwd[1], fwd[0])) % 360 == pytest.approx(yaw % 360)


def test_facing_refuses_a_non_finite_yaw():
    with pytest.raises(ValueError, match="finite"):
        facing_quat_xyzw(float("nan"))


# -- S1d: the dummy faces the way it walks --
# It faced the robot at a fixed 180 deg while crossing sideways. The facing now follows the drawn entry
# angle: the walk goes -(cos a, sin a), a measured from the robot's heading +x, so the dummy's own
# forward axis (+x in its body frame) has to land on that direction.

@pytest.mark.parametrize("angle_deg", [30.0, 60.0, 90.0, 118.6, 150.0])
def test_the_travel_facing_turns_the_dummys_front_onto_its_walk(angle_deg):
    yaw = travel_facing_yaw_deg(angle_deg)
    fwd = _rotate(facing_quat_xyzw(yaw), [1.0, 0.0, 0.0])
    a = math.radians(angle_deg)
    assert fwd == pytest.approx([-math.cos(a), -math.sin(a), 0.0], abs=1e-12)


@pytest.mark.parametrize("angle_deg, want", [(90.0, -90.0), (30.0, -150.0), (150.0, -30.0), (0.0, 180.0),
                                             (180.0, 0.0), (-10.0, 170.0), (360.0, 180.0), (190.0, 10.0)])
def test_the_travel_facing_is_folded_into_one_turn(angle_deg, want):
    # (-180, 180]: one value per direction in the sidecar, and a quaternion with w >= 0
    yaw = travel_facing_yaw_deg(angle_deg)
    assert yaw == pytest.approx(want, abs=1e-12) and -180.0 < yaw <= 180.0
    assert facing_quat_xyzw(yaw)[3] >= 0.0


def test_the_travel_facing_is_a_python_float_and_refuses_a_non_finite_angle():
    assert type(travel_facing_yaw_deg(np.float32(90.0))) is float
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            travel_facing_yaw_deg(bad)


# The hold servo sizes its gains per world axis from the dummy's inertia about that axis. Taken once at the
# pose of the first call, the tipping axes would keep the first episode's facing for every later one.
_MASS = np.array([[4.0, 1.5, 2.5, 0.7]])
_OFFSETS = np.array([[[0.06, 0.0, 0.5], [0.0, 0.21, 0.2], [-0.03, -0.215, -0.3], [0.1, 0.09, -0.7]]])


def _inertia_tensor(mass, offsets):
    return sum(m * (np.dot(r, r) * np.eye(3) - np.outer(r, r)) for m, r in zip(mass, offsets))


def _at(yaw_deg):
    yaw = np.radians(np.atleast_1d(np.asarray(yaw_deg, float)))
    return yawed_axis_inertia(_MASS, _OFFSETS, np.cos(yaw), np.sin(yaw))


def test_the_inertia_at_no_turn_is_the_body_frames():
    xx, yy, zz = _at(0.0)
    full = _inertia_tensor(_MASS[0], _OFFSETS[0])
    assert [xx[0], yy[0], zz[0]] == pytest.approx([full[0, 0], full[1, 1], full[2, 2]], rel=1e-12)
    assert xx[0] != pytest.approx(yy[0], rel=1e-3)        # the stand-in body is not symmetric: the swap below shows


def test_a_quarter_turn_swaps_the_tipping_axes_and_a_half_turn_changes_nothing():
    xx0, yy0, zz0 = _at(0.0)
    xx90, yy90, zz90 = _at(90.0)
    assert [xx90[0], yy90[0], zz90[0]] == pytest.approx([yy0[0], xx0[0], zz0[0]], rel=1e-12)
    assert [v[0] for v in _at(180.0)] == pytest.approx([xx0[0], yy0[0], zz0[0]], rel=1e-12)


@pytest.mark.parametrize("yaw_deg", [-150.0, -90.0, -30.0, 37.0])
def test_the_inertia_at_a_turn_is_the_rotated_tensors_diagonal(yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    want = R @ _inertia_tensor(_MASS[0], _OFFSETS[0]) @ R.T
    got = [v[0] for v in _at(yaw_deg)]
    assert got == pytest.approx([want[0, 0], want[1, 1], want[2, 2]], rel=1e-12)
    assert got[2] == pytest.approx(_at(0.0)[2][0], rel=1e-12)      # yaw does not move mass about the yaw axis


def test_the_inertia_is_taken_per_env():
    mass = np.repeat(_MASS, 2, axis=0)
    offsets = np.repeat(_OFFSETS, 2, axis=0)
    yaw = np.radians([0.0, 90.0])
    xx, yy, zz = yawed_axis_inertia(mass, offsets, np.cos(yaw), np.sin(yaw))
    assert xx.shape == yy.shape == zz.shape == (2,)
    assert xx[1] == pytest.approx(yy[0], rel=1e-12) and yy[1] == pytest.approx(xx[0], rel=1e-12)


# check_s1d_scene (h) compares the servo's per-axis inertia with the live links'. Written as
# max(abs(a - b) / b ...) > tolerance, a NaN on any axis passed: max() skips or returns it, and NaN > x
# is False.

def test_the_relative_gap_is_the_worst_axis():
    assert relative_gap_max([2.0, 2.5, 0.5], [2.0, 2.5, 0.5]) == 0.0
    assert relative_gap_max([2.0, 2.6, 0.5], [2.0, 2.5, 0.5]) == pytest.approx(0.04)


@pytest.mark.parametrize("got, want", [([2.0, float("nan"), 0.5], [2.0, 2.5, 0.5]),
                                       ([float("nan"), 2.5, 0.5], [2.0, 2.5, 0.5]),
                                       ([2.0, float("inf"), 0.5], [2.0, 2.5, 0.5]),
                                       ([2.0, 2.5, 0.5], [2.0, float("nan"), 0.5]),
                                       ([2.0, 2.5, 0.5], [2.0, 0.0, 0.5]),
                                       ([2.0, -2.5, 0.5], [2.0, 2.5, 0.5]),
                                       ([2.0, 2.5], [2.0, 2.5, 0.5]), ([], [])])
def test_a_value_that_is_not_a_positive_finite_number_is_an_infinite_gap(got, want):
    assert relative_gap_max(got, want) == math.inf


# The call sites import Isaac, so they are read from the source (as test_ssm_provisional_merge does). Each
# line below, taken out, leaves every other test green and puts the sideways walk -- or the first episode's
# gains -- back into the batch; check_s1d_scene (h) is what sees the behaviour, in Isaac.
_S1A = Path(__file__).resolve().parents[2] / "generation/envs/s1a"


def test_the_reset_turns_each_dummy_to_its_own_walk():
    src = (_S1A / "mdp/events.py").read_text(encoding="utf-8")
    assert "yaw_deg = [travel_facing_yaw_deg(math.degrees(float(a))) for a in angle]" in src
    assert 'entry["facing_yaw_deg"][env_ids] = torch.tensor(yaw_deg, dtype=torch.float64, device=device)' in src
    assert 'entry["face_travel"] = bool(face_travel)' in src
    # the reach the waiting offset is derived from must not depend on the turn
    assert "_horizontal_reach(env, asset_name) if face_travel else _lateral_reach(env, asset_name)" in src
    assert "face_travel: bool = False," in src            # S1a, which does not pass it, keeps its facing


def test_the_hold_servo_takes_its_inertia_at_the_episodes_facing():
    src = (_S1A / "mdp/commands.py").read_text(encoding="utf-8")
    assert 'if entry.get("face_travel", False):' in src
    assert ("yawed_axis_inertia(\n                self._dummy_body_mass, self._dummy_region_offsets_body, "
            "qw * qw - qz * qz, 2.0 * qw * qz\n            )") in src
    assert "self._dummy_moment_of_inertia = torch.stack([i_xx + pivot, i_yy + pivot, i_zz], dim=-1)" in src
    # body-frame offsets come from the pose the dummy has when they are read -- the offsets the inertia is taken
    # over: the whole figure's from its centre of mass for the rigid dummy (axes_offset is com_centred_offset then,
    # test_gait_command_wiring), the body above the hips' for the jointed one
    assert "capture_quat = dummy.data.root_quat_w.torch.clone()" in src
    assert "capture_quat.unsqueeze(1).expand(-1, offset.shape[1], -1), axes_offset" in src


def test_the_yaw_the_servo_reads_off_the_quaternion_is_the_facings():
    # commands.py takes cos and sin of the facing from the parked quaternion (0, 0, z, w)
    for angle_deg in (30.0, 77.3, 90.0, 150.0):
        yaw = math.radians(travel_facing_yaw_deg(angle_deg))
        _, _, qz, qw = facing_quat_xyzw(travel_facing_yaw_deg(angle_deg))
        assert qw * qw - qz * qz == pytest.approx(math.cos(yaw), abs=1e-12)
        assert 2.0 * qw * qz == pytest.approx(math.sin(yaw), abs=1e-12)


def test_s1d_turns_the_dummy_to_its_walk_and_guards_its_radius():
    src = (_S1A.parent / "s1d/s1d_env_cfg.py").read_text(encoding="utf-8")
    assert 'self.events.reset_dummy_entry.params["face_travel"] = True' in src
    assert '"dummy_half_width": aisle_guard.dummy_horizontal_radius(anthropometry)' in src
    assert 'params["facing_yaw_deg"]' not in src           # a fixed facing set next to it would be ignored, silently


def test_the_recorder_writes_the_episodes_own_facing():
    src = (_S1A / "recorder.py").read_text(encoding="utf-8")
    assert '"human_facing_yaw_deg": float(entry["facing_yaw_deg"][0]),' in src


# -- S1d: the hold servo carries the dummy --
# Pulled at its centre of mass while standing on its feet, the dummy turned 12-24 deg about the vertical in
# the fast entries. The servo now holds its height too. Each line below, taken out, leaves the other tests
# green; check_s1d_scene (h) sees the behaviour in Isaac.

def test_the_hold_servo_holds_the_dummys_height_and_its_weight():
    src = (_S1A / "mdp/commands.py").read_text(encoding="utf-8")
    assert "dummy_hold_clearance_m: float | None = None" in src            # off unless a scenario turns it on
    assert "lin_force[:, 2] = self._dummy_weight + lin_stiffness * (z_target - z) - lin_damping * z_vel" in src
    assert ("z_target = (self._env.scene.env_origins[:, 2] + self.cfg.dummy_hold_stand_height_m\n"
            "                        + self.cfg.dummy_hold_clearance_m)") in src
    # a carried dummy does not tip about its feet: neither the pivot term nor gravity's
    assert "parallel_axis_term = torch.zeros_like(parallel_axis_term)" in src
    assert "gravity_destabilizing = torch.zeros_like(gravity_destabilizing)" in src
    # the vertical velocity must not be differenced across a reset's teleport
    assert 'self._dummy_prev_z[env_ids] = self._env.scene["dummy"].data.root_pos_w.torch[env_ids, 2]' in src


def test_the_applied_wrench_is_stored_after_released_envs_are_zeroed():
    # Stored before the mask, a released env's row would keep the weight the servo no longer
    # applies, and the recorder reads these buffers.
    src = (_S1A / "mdp/commands.py").read_text(encoding="utf-8")
    mask = src.index("        lin_force = lin_force * held\n        ang_torque = ang_torque * held\n")
    assert src.rindex("self.dummy_lin_force_applied = lin_force") > mask
    assert src.rindex("self.dummy_torque_applied = ang_torque") > mask


def test_s1d_carries_the_dummy_at_the_height_its_spec_gives():
    src = (_S1A.parent / "s1d/s1d_env_cfg.py").read_text(encoding="utf-8")
    assert "self.commands.base_velocity.dummy_hold_clearance_m = HOLD_CLEARANCE_M" in src
    assert ("self.commands.base_velocity.dummy_hold_stand_height_m = "
            "aisle_guard.standing_root_height(anthropometry)") in src
    assert "dummy_hold_clearance_m" not in (_S1A / "s1a_env_cfg.py").read_text(encoding="utf-8")   # S1a stands on its feet


def test_the_recorder_says_when_the_hold_force_carries_the_weight():
    src = (_S1A / "recorder.py").read_text(encoding="utf-8")
    assert 'self.randomization["human_hold_clearance_m"] = float(command.cfg.dummy_hold_clearance_m)' in src


# -- S1d: the scripted walk stops at the far side's waiting offset --
# The slice found S1d's dummy walking on into the far rack while its hold servo still pushed
# (hold_contaminated_steps 83-432 in 7 of 20, against a contract of at most 1). The walk now
# stops where the guarded waiting offset sits mirrored across the robot's lane.


def test_the_stop_fraction_is_one_until_the_line_is_within_a_step():
    assert lateral_stop_fraction(np.array([3.117]), np.array([-3.117]), np.array([0.0174]))[0] == 1.0


def test_the_step_that_reaches_the_line_moves_only_the_remainder():
    f = lateral_stop_fraction(np.array([-2.995]), np.array([-3.0]), np.array([0.01]))[0]
    assert f == pytest.approx(0.5)


@pytest.mark.parametrize("target", [-3.0, -3.0 - 1e-12])
def test_at_or_just_past_the_line_the_walk_does_not_move(target):
    assert lateral_stop_fraction(np.array([target]), np.array([-3.0]), np.array([0.01]))[0] == 0.0


def test_the_stop_leaves_every_step_before_it_bit_identical_and_lands_on_the_line():
    # The scripted target as _advance_dummy integrates it (target_y -= dy each step), with and
    # without the stop: S1d's physics up to the stop is the slice's to the bit.
    wait_y, dy, stop = 3.117, 0.87 * 0.02 * math.sin(math.radians(118.6)), -3.117
    free, held = [wait_y], [wait_y]
    for _ in range(1000):
        free.append(free[-1] - dy)
        f = lateral_stop_fraction(np.array([held[-1]]), np.array([stop]), np.array([dy]))[0]
        held.append(held[-1] - f * dy)
    arrive = next(i for i, y in enumerate(free) if y < stop)
    assert held[:arrive] == free[:arrive]
    assert all(y == pytest.approx(stop, abs=1e-12) for y in held[arrive:])
    assert min(held) >= stop - 1e-12


def test_the_fraction_works_element_wise_on_arrays():
    f = lateral_stop_fraction(np.array([3.0, -2.995, -3.0]), np.full(3, -3.0), np.full(3, 0.01))
    assert f.tolist() == pytest.approx([1.0, 0.5, 0.0])


@pytest.mark.parametrize("target, want", [(3.0, 1.0), (-2.5, 1.0), (-3.0, 0.0), (-3.1, 0.0)])
def test_a_zero_step_keeps_the_target_where_it_is_before_at_and_past_the_line(target, want):
    # The jointed dummy's scripted speed ramps to 0 at the line, so its step there is 0: a zero step is no
    # move, with no 0/0 -- a NaN fraction would make the target and the hold force NaN for the rest of the run.
    with np.errstate(all="raise"):
        f = lateral_stop_fraction(np.array([target]), np.array([-3.0]), np.array([0.0]))
    assert f.tolist() == [want]


def test_a_zero_step_keeps_the_target_in_torch_too():
    torch = pytest.importorskip("torch")
    f = lateral_stop_fraction(torch.tensor([3.0, -3.0, -2.995]), torch.full((3,), -3.0), torch.tensor([0.0, 0.0, 0.01]))
    assert f.tolist() == pytest.approx([1.0, 0.0, 0.5], rel=1e-4)   # float32


def test_a_nonzero_step_takes_the_same_arithmetic_to_the_bit():
    # The rigid dummy's step is a constant above 0: its fractions are the division they always were.
    rng = np.random.default_rng(7)
    ty, sy, dy = rng.uniform(-4, 4, 200), np.full(200, -3.117), rng.uniform(1e-4, 0.05, 200)
    assert lateral_stop_fraction(ty, sy, dy).tobytes() == ((ty - sy) / dy).clip(0.0, 1.0).tobytes()


def test_the_gait_ramp_comes_to_rest_on_the_line_without_a_nan():
    # _advance_dummy's order with the gait on: the ramp's speed (down to 0 at the line), its step, the stop
    # fraction, the latch, the target -- in float32, the target buffer's type -- and on past the stop.
    from generation.dummy.gait import gait_table as gt

    angle, dt, wait_y = math.radians(90.0), 0.02, 3.117
    entry = {"triggered": np.ones(1, bool), "target_xy": np.array([[0.0, wait_y]], np.float32),
             "stop_y": np.array([-wait_y], np.float32), "walk_ended": np.zeros(1, bool)}
    v = np.zeros(1)
    for _ in range(1500):
        remaining = gt.remaining_to_stop(entry["target_xy"][:, 1], entry["stop_y"], np.array([angle]))
        v = gt.ramp_speed(v, np.array([1.0]), dt, 1.0, 1.0, remaining)
        step = (v * dt).astype(np.float32)
        dxy = np.stack([step * np.float32(math.cos(angle)), step * np.float32(math.sin(angle))], axis=-1)
        frac = lateral_stop_fraction(entry["target_xy"][:, 1], entry["stop_y"], step * np.float32(math.sin(angle)))
        dxy = dxy * frac[:, None]
        latch_walk_end(entry, entry["triggered"], frac)
        entry["target_xy"] -= dxy
        assert np.isfinite(entry["target_xy"]).all() and np.isfinite(frac).all()
    assert bool(entry["walk_ended"][0])
    assert entry["target_xy"][0, 1] == pytest.approx(-wait_y, abs=1e-5) and v[0] == pytest.approx(0.0, abs=1e-2)


def test_the_catalogue_angle_range_reaches_the_stop_line():
    check_stop_angles((30.0, 150.0))


@pytest.mark.parametrize("rng", [(0.0, 150.0), (30.0, 180.0), (-10.0, 30.0), (150.0, 30.0)])
def test_an_angle_range_that_can_walk_parallel_to_or_away_from_the_line_is_refused(rng):
    # sin(angle) is the lateral share of the walk; at 0 or 180 deg the target never nears the line.
    with pytest.raises(ValueError, match="angle"):
        check_stop_angles(rng)


# -- the realized entry speed's window --
# The recorder measured human_entry_speed_realized from the first triggered step to the last one.
# "triggered" latches, so once S1d's walk stopped the elapsed time kept growing over a standing dummy:
# the slice's episode 2 read 0.538 m/s against 0.627 drawn (0.628 before the stop existed).


def _walk(window, n_walk, n_stand, speed=0.6, dt=0.02, ended_at=None):
    y = 0.0
    for k in range(n_walk + n_stand):
        moving = k < n_walk
        if moving:
            y -= speed * dt
        window.update(triggered=True, walk_ended=(ended_at is not None and k >= ended_at), xy=np.array([0.0, y]), t=k * dt)
    return window.speed()


def test_the_window_ends_on_the_step_the_walk_reaches_its_stop():
    # The sample of the step walk_ended turns true on is the window's last one.
    assert _walk(EntrySpeedWindow(), 300, 400, ended_at=299) == pytest.approx(0.6)


def test_without_a_stop_the_window_runs_to_the_last_triggered_step_as_it_always_did():
    # S1a: walk_ended never turns true -- the same onset-to-last-triggered span the recorder has always used.
    assert _walk(EntrySpeedWindow(), 300, 0) == pytest.approx(0.6)
    assert _walk(EntrySpeedWindow(), 300, 400) == pytest.approx(0.6 * 299 / 699)


def test_the_window_has_no_speed_before_two_triggered_steps():
    w = EntrySpeedWindow()
    assert w.speed() is None
    w.update(triggered=False, walk_ended=False, xy=np.zeros(2), t=0.0)
    w.update(triggered=True, walk_ended=False, xy=np.zeros(2), t=0.02)
    assert w.speed() is None


def test_the_first_fraction_below_one_is_the_walks_last_move():
    # The flag _advance_dummy latches (frac < 1) marks the arrival step: after it the target moves by
    # at most float noise, so ending the window there loses no walk.
    wait_y, dy, stop = 3.117, 0.87 * 0.02 * math.sin(math.radians(118.6)), -3.117
    y, arrived, moves_after = wait_y, None, []
    for k in range(1000):
        f = lateral_stop_fraction(np.array([y]), np.array([stop]), np.array([dy]))[0]
        if arrived is not None:
            moves_after.append(f * dy)
        elif f < 1.0:
            arrived = k
        y -= f * dy
    assert arrived is not None and max(moves_after) < 1e-12


# -- the window as the env steps it, and its call sites --
# The tests above hand the window its flags. With the recorder passing walk_ended=False, the command
# term's latch deleted, or the reset's clearing line deleted, every test stayed green -- and each of
# those puts the low speed back into the shipped metadata. Below, the flags come from the functions
# the command term and the recorder call, in the order the env steps them.

MDP = Path(__file__).resolve().parents[2] / "generation/envs/s1a/mdp"


def _stepped_walk(speed, angle_deg, *, stop=True, steps=1000, trigger_step=50, wait_y=3.117, dt=0.02):
    """One env through an env step's order (isaaclab ManagerBasedRLEnv.step): physics first -- the body
    takes the target the previous command update left, a perfect-tracking stand-in -- then the command
    update as _advance_dummy does it (trigger, stop fraction, latch, target), then the recorder's read."""
    angle = math.radians(angle_deg)
    move = speed * dt * np.array([math.cos(angle), math.sin(angle)])
    entry = {"triggered": np.zeros(1, bool), "target_xy": np.array([[0.0, wait_y]])}
    if stop:
        entry.update(stop_y=np.array([-wait_y]), walk_ended=np.zeros(1, bool))
    window = EntrySpeedWindow()
    for k in range(steps):
        body = entry["target_xy"][0].copy()
        entry["triggered"] |= k >= trigger_step
        moving = entry["triggered"]
        dxy = np.where(moving[:, None], move, 0.0)
        if stop:
            frac = lateral_stop_fraction(entry["target_xy"][:, 1], entry["stop_y"], np.full(1, move[1]))
            dxy = dxy * frac[:, None]
            latch_walk_end(entry, moving, frac)
        entry["target_xy"] -= dxy
        window.observe(entry, xy=body, t=k * dt)
    return window, entry


@pytest.mark.parametrize("speed, angle_deg", [(0.6, 90.0), (0.627, 118.6), (0.87, 118.6), (1.2, 60.0)])
def test_a_walk_that_stops_reads_the_speed_it_walked_at(speed, angle_deg):
    # Both ends of the window are read one command ahead of the body (the flags are set by the command
    # update, after the step's physics), so the samples span the full-length moves only: a body on its
    # target reads the drawn speed. Running on to the last triggered step reads low by the standing time
    # (the defect, 0.538 against 0.627); closing one step later mixes the last
    # partial move in and reads 0.02-0.23 % low over these four.
    window, entry = _stepped_walk(speed, angle_deg)
    assert bool(entry["walk_ended"][0]) and entry["target_xy"][0, 1] == pytest.approx(-3.117, abs=1e-12)
    assert window.speed() == pytest.approx(speed, rel=1e-9)


def test_a_walk_without_a_stop_is_read_to_its_last_triggered_step():
    # S1a: no stop_y and no walk_ended in the entry -- the window must not ask for either.
    window, entry = _stepped_walk(0.6, 90.0, stop=False)
    assert "walk_ended" not in entry
    assert window.speed() == pytest.approx(0.6, rel=1e-9)


def test_a_walk_that_never_triggers_has_no_speed():
    window, _ = _stepped_walk(0.6, 90.0, trigger_step=10**6)
    assert window.speed() is None


def test_the_command_term_latches_the_walks_end():
    # commands.py imports Isaac, so its call is read from the source (as test_ssm_provisional_merge does).
    src = (MDP / "commands.py").read_text(encoding="utf-8")
    assert "latch_walk_end(entry, moving, frac)" in src, (
        "commands.py 가 걸음의 끝을 걸지 않습니다 — 멈춘 뒤의 시간이 human_entry_speed_realized 에 다시 섞입니다")


def test_the_reset_clears_the_walks_end():
    # Left set, the next episode's window is closed before its walk starts and the speed comes out None.
    src = (MDP / "events.py").read_text(encoding="utf-8")
    assert 'entry["walk_ended"][env_ids] = False' in src


def test_the_recorder_reads_the_flags_through_the_window():
    src = (MDP.parent / "recorder.py").read_text(encoding="utf-8")
    assert "self._entry_speed.observe(entry, xy=dummy_xy, t=timestamp)" in src
    # update() takes the flags as arguments: called from the recorder, walk_ended could be passed as
    # anything (False, say) without a test seeing it.
    assert "_entry_speed.update(" not in src
    assert 'self.randomization["human_entry_speed_realized"] = self._entry_speed.speed()' in src
