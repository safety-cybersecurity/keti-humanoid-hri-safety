"""Unit tests for the S1a safety quantities. No Isaac Sim, no torch."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from generation.envs.s1a.mdp.safety import (
    aggregate_contact_by_region,
    center_of_mass,
    contact_relative_speed,
    classify_contact_type,
    command_scale,
    load_ssm_parameters,
    min_distance_human,
    separation_required,
    stability_margin,
    zero_moment_point,
)


def test_paired_regions_take_the_maximum_not_the_sum():
    # BR-08 (upper arm) is authored as a left/right pair, so one code owns two
    # prims. Summing them reports a load no single limb ever took.
    codes = ["BR-08", "BR-08", "BR-01"]
    magnitudes = np.array([30.0, 40.0, 10.0])

    out = aggregate_contact_by_region(codes, magnitudes)

    assert out["BR-08"] == 40.0
    assert out["BR-01"] == 10.0


def test_regions_with_no_contact_report_zero_not_absent():
    codes = ["BR-01", "BR-02"]
    magnitudes = np.array([0.0, 5.0])

    out = aggregate_contact_by_region(codes, magnitudes)

    assert out == {"BR-01": 0.0, "BR-02": 5.0}


def test_length_mismatch_is_an_error_not_a_silent_truncation():
    with pytest.raises(ValueError):
        aggregate_contact_by_region(["BR-01", "BR-02"], np.array([1.0]))


FULL = {
    "human_speed": 1.0,
    "robot_reaction_time": 0.1,
    "robot_stop_time": 0.4,
    "intrusion_distance": 0.2,
    "position_uncertainty_robot": 0.05,
    "position_uncertainty_human": 0.05,
}


def test_separation_grows_with_both_approach_and_robot_speed():
    slow = separation_required(FULL, human_speed=1.0, robot_speed=0.3)
    fast = separation_required(FULL, human_speed=1.0, robot_speed=1.2)
    assert fast > slow

    slower_human = separation_required(FULL, human_speed=0.4, robot_speed=0.3)
    assert slow > slower_human


def test_separation_is_the_sum_of_its_named_terms():
    # human travel + robot travel while reacting + robot travel while stopping
    # + intrusion + both uncertainties
    expected = 1.0 * (0.1 + 0.4) + 0.3 * 0.1 + 0.3 * 0.4 / 2.0 + 0.2 + 0.05 + 0.05
    assert separation_required(FULL, 1.0, 0.3) == pytest.approx(expected)


def test_a_missing_parameter_is_refused_rather_than_defaulted():
    partial = dict(FULL, intrusion_distance=None)
    with pytest.raises(ValueError, match="intrusion_distance"):
        separation_required(partial, 1.0, 0.3)


def test_loader_reports_which_parameters_the_standard_has_not_filled():
    values, missing = load_ssm_parameters("meta/rules.json")
    assert set(values) == set(FULL)
    # intrusion_distance 는 KS B ISO 13855 가 정하는 값이라 여전히 비어 있다.
    # 부속서 A 를 병합해도 이건 채워지지 않는다 — 다른 표준이다.
    assert "intrusion_distance" in missing
    # 반대로, 실측으로 채운 것과 표준이 값을 주지 않는 것은 비어 있으면 안 된다.
    for filled in ("robot_stop_time", "robot_reaction_time",
                   "position_uncertainty_robot", "position_uncertainty_human"):
        assert filled not in missing, f"{filled} 가 비었습니다 — 정본을 확인하십시오"


def test_min_distance_subtracts_the_region_radius():
    robot = np.array([[1.0, 0.0, 0.0]])
    human = np.array([[0.0, 0.0, 0.0]])
    radii = np.array([0.2])
    # centre distance is 1.0, the region's surface is 0.2 nearer
    assert min_distance_human(robot, human, radii) == pytest.approx(0.8)


def test_min_distance_is_never_negative_once_bodies_overlap():
    robot = np.array([[0.1, 0.0, 0.0]])
    human = np.array([[0.0, 0.0, 0.0]])
    radii = np.array([0.3])
    assert min_distance_human(robot, human, radii) == 0.0


# R=2.0, slowdown_factor=2.0 -> slowdown_start=4.0, stop_band_factor=1.05 -> stop_band=2.1
R = 2.0
SLOWDOWN_FACTOR = 2.0
STOP_BAND_FACTOR = 1.05


def test_command_scale_is_full_speed_outside_the_slowdown_distance():
    assert command_scale(5.0, R, SLOWDOWN_FACTOR, STOP_BAND_FACTOR) == 1.0


def test_command_scale_is_exactly_one_at_the_slowdown_boundary():
    # Continuity: approaching from outside must not jump before it ramps.
    assert command_scale(4.0, R, SLOWDOWN_FACTOR, STOP_BAND_FACTOR) == 1.0


def test_command_scale_is_exactly_zero_inside_the_stop_band():
    # Exactly 0, not merely small — a ramp that only approaches zero never
    # latches a protective stop in finite time (see command_scale's docstring).
    assert command_scale(2.1, R, SLOWDOWN_FACTOR, STOP_BAND_FACTOR) == 0.0
    assert command_scale(1.0, R, SLOWDOWN_FACTOR, STOP_BAND_FACTOR) == 0.0
    assert command_scale(0.0, R, SLOWDOWN_FACTOR, STOP_BAND_FACTOR) == 0.0


def test_command_scale_is_strictly_between_zero_and_one_mid_ramp():
    scale = command_scale(3.0, R, SLOWDOWN_FACTOR, STOP_BAND_FACTOR)
    assert 0.0 < scale < 1.0
    assert scale == pytest.approx((3.0 - 2.1) / (4.0 - 2.1))


def test_command_scale_refuses_a_non_positive_separation_required():
    with pytest.raises(ValueError):
        command_scale(3.0, 0.0, SLOWDOWN_FACTOR, STOP_BAND_FACTOR)


def test_command_scale_refuses_a_stop_band_that_does_not_leave_a_ramp():
    with pytest.raises(ValueError):
        command_scale(3.0, R, slowdown_factor=1.5, stop_band_factor=1.5)


def test_command_scale_refuses_a_stop_band_at_or_inside_the_required_separation():
    # stop_band_factor <= 1.0 puts the stop band at or inside
    # separation_required itself, so the ramp can still be nonzero exactly
    # where separation_violation is already true (distance == required):
    # command_scale(2.0, 2.0, 2.0, 0.5) returned 0.333 before this guard.
    with pytest.raises(ValueError):
        command_scale(2.0, 2.0, slowdown_factor=2.0, stop_band_factor=1.0)
    with pytest.raises(ValueError):
        command_scale(2.0, 2.0, slowdown_factor=2.0, stop_band_factor=0.5)


def test_stability_margin_is_positive_inside_and_negative_outside():
    feet = np.array([[0.0, 0.0], [0.2, 0.0], [0.2, 0.3], [0.0, 0.3]])
    assert stability_margin(feet, np.array([0.1, 0.15])) > 0.0
    assert stability_margin(feet, np.array([0.5, 0.15])) < 0.0


def test_two_contact_points_cannot_report_a_positive_margin():
    # A support polygon needs three points. Standing on a line is at best
    # marginal, never stable — reporting a positive margin would hide a fall.
    feet = np.array([[0.0, 0.0], [0.2, 0.0]])
    assert stability_margin(feet, np.array([0.1, 0.0])) <= 0.0


def test_zero_support_points_is_negative_infinity_not_nan_or_a_finite_number():
    # Both feet unloaded at once (flight phase of dynamic gait) is not a
    # degenerate polygon, it is no support at all — the documented sentinel
    # (dataset schema §6) is exactly float("-inf"), distinguishable
    # downstream from an ordinary negative (outside-the-polygon) margin.
    feet = np.zeros((0, 2))
    margin = stability_margin(feet, np.array([0.0, 0.0]))
    assert margin == float("-inf")
    assert not math.isnan(margin)


def _rules_with_ssm_values(tmp_path, **values):
    """A copy of the rules canon with some SSM parameters filled in.

    Built from the real meta/rules.json so the test exercises the shape the
    loader actually reads, not a hand-made stand-in that could drift from it.
    """
    rules = json.loads(Path("meta/rules.json").read_text(encoding="utf-8"))
    entry = next(r for r in rules["rules"] if r["rule_id"] == "R-TS15066-SSM-01")
    for name, value in values.items():
        entry["parameters"][name]["value"] = value
    path = tmp_path / "rules.json"
    # json.dumps writes NaN as the bare literal NaN and json.loads reads it
    # back as a float -- which is exactly how such a value reaches the loader.
    path.write_text(json.dumps(rules), encoding="utf-8")
    return path


def test_loader_counts_a_nan_value_as_unfilled(tmp_path):
    # nan is a float, so an isinstance check waves it through as "filled".
    # Nothing downstream ever recovers: the distance comes out nan, every
    # `distance < required` comparison is False, separation_violation never
    # fires, and the near_miss label vanishes from the dataset in silence.
    # Hand-filling the standard's numbers is exactly when this can happen.
    path = _rules_with_ssm_values(tmp_path, robot_stop_time=float("nan"), human_speed=1.6)

    values, missing = load_ssm_parameters(path)

    assert "robot_stop_time" in missing
    assert values["robot_stop_time"] is None
    assert "human_speed" not in missing


def test_loader_counts_a_negative_value_as_unfilled(tmp_path):
    # A negative term shrinks the required separation instead of growing it,
    # so the robot is allowed closer than the standard permits.
    path = _rules_with_ssm_values(tmp_path, intrusion_distance=-0.2)

    values, missing = load_ssm_parameters(path)

    assert "intrusion_distance" in missing
    assert values["intrusion_distance"] is None


def test_loader_keeps_a_legitimate_zero(tmp_path):
    # Unlike a dummy's mass or stiffness, a distance term may honestly be
    # zero (no intrusion allowance, perfect position knowledge in sim), so
    # zero must not be swept into the same bucket as nan and negatives.
    path = _rules_with_ssm_values(tmp_path, intrusion_distance=0.0)

    values, missing = load_ssm_parameters(path)

    assert "intrusion_distance" not in missing
    assert values["intrusion_distance"] == 0.0


def test_a_nan_parameter_is_refused_rather_than_producing_a_nan_distance():
    # The loader is not the only door: separation_required is called with
    # dicts built elsewhere too, and it is the function that turns one nan
    # into a permanently-False violation test.
    with pytest.raises(ValueError, match="robot_stop_time"):
        separation_required(dict(FULL, robot_stop_time=float("nan")), 1.0, 0.3)


def test_a_negative_parameter_is_refused_rather_than_shrinking_the_distance():
    with pytest.raises(ValueError, match="intrusion_distance"):
        separation_required(dict(FULL, intrusion_distance=-0.2), 1.0, 0.3)


def test_an_empty_support_list_is_the_same_sentinel_as_an_empty_array():
    # np.atleast_2d(np.asarray([])) has shape (1, 0), not (0, 2), so the
    # zero-points guard never fires for a plain empty list and the hull code
    # raises instead of returning the documented sentinel. Live callers pass
    # (0, 2), so this is the contract drifting from the implementation.
    assert stability_margin([], np.array([0.0, 0.0])) == float("-inf")


def test_a_nan_speed_argument_is_refused_rather_than_producing_a_nan_distance():
    # human_speed and robot_speed arrive as arguments, not through the params
    # dict, so guarding the dict alone leaves the same fail-open open: a nan
    # distance makes every `distance < required` test False forever.
    with pytest.raises(ValueError, match="human_speed"):
        separation_required(FULL, float("nan"), 0.3)


def test_an_infinite_robot_speed_is_refused():
    with pytest.raises(ValueError, match="robot_speed"):
        separation_required(FULL, 1.0, float("-inf"))


def test_a_negative_robot_speed_is_refused():
    # Speed, not velocity -- a negative here silently shrinks the distance.
    with pytest.raises(ValueError, match="robot_speed"):
        separation_required(FULL, 1.0, -0.3)


def test_a_parameter_missing_from_the_canon_is_still_reported_missing(tmp_path):
    # The loader iterated whatever keys the file happened to have, so deleting
    # robot_stop_time outright made it vanish from `missing` as well -- the
    # loader would report nothing wrong about a parameter that is not there.
    rules = json.loads(Path("meta/rules.json").read_text(encoding="utf-8"))
    entry = next(r for r in rules["rules"] if r["rule_id"] == "R-TS15066-SSM-01")
    del entry["parameters"]["robot_stop_time"]
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(rules), encoding="utf-8")

    values, missing = load_ssm_parameters(path)

    assert "robot_stop_time" in missing
    assert values["robot_stop_time"] is None


def test_an_xyz_support_array_is_refused_rather_than_reshaped():
    # reshape(-1, 2) turned a perfectly ordinary mistake -- passing XYZ
    # contact points -- into three unrelated XY points. (2,3) became
    # [[0,0],[100,1],[0,100]], a triangle metres wide that reads as *stable*
    # (+28.6) for a CoM inside it. Refusing the shape is the only safe answer;
    # a wrong support polygon cannot be distinguished from a real one.
    with pytest.raises(ValueError):
        stability_margin([[0.0, 0.0, 100.0], [1.0, 0.0, 100.0]], np.array([30.0, 30.0]))


def test_an_odd_length_flat_support_array_is_refused():
    with pytest.raises(ValueError):
        stability_margin([0.0, 1.0, 2.0], np.array([0.0, 0.0]))


def test_a_non_finite_support_point_is_refused_rather_than_read_as_stable():
    # Shape was checked but not finiteness, so the same fail-open the SSM
    # guard closed stayed open one function over: a NaN corner returned 0.0
    # (reads as "exactly on the boundary", not as unstable) and an Inf corner
    # returned nan, which makes every later `margin < 0` test False. A flight
    # phase would vanish from the data instead of showing up as -inf.
    feet = np.array([[0.0, 0.0], [0.2, 0.0], [0.2, 0.3], [0.0, 0.3]])
    for bad in (float("nan"), float("inf"), float("-inf")):
        corrupted = feet.copy()
        corrupted[1, 0] = bad
        with pytest.raises(ValueError):
            stability_margin(corrupted, np.array([0.1, 0.15]))


def test_a_non_finite_centre_of_mass_is_refused():
    feet = np.array([[0.0, 0.0], [0.2, 0.0], [0.2, 0.3], [0.0, 0.3]])
    with pytest.raises(ValueError):
        stability_margin(feet, np.array([float("nan"), 0.15]))


# --- 접촉 유형 분류 (스키마 §8.1) ---------------------------------------
# 정본의 구분은 지속시간이 아니라 **후퇴 가능 여부**다: transient 는 인체가
# 후퇴할 수 있는 자유 충돌, quasistatic 은 협착처럼 후퇴할 수 없는 상태.
# S1a 가 만들 수 있는 협착은 하나뿐이다 — 넘어진 더미를 로봇이 지면 쪽으로 누름.

FORCE = np.array([0.0, 50.0, 200.0])        # (n_region,) N
HIGH = np.array([1.0, 1.0, 1.0])            # 서 있는 부위 높이 (m)
LOW = np.array([0.05, 0.05, 0.05])          # 바닥에 눌린 부위 높이 (m)
KW = dict(contact_eps_n=1.0, pin_height_m=0.15)


def test_no_robot_contact_means_no_region_is_typed():
    # 이 필드는 로봇-인체 접촉을 서술한다. 로봇이 더미를 건드리지 않는 동안
    # 부위에 실린 힘은 더미 자신의 접지 하중이므로 접촉 유형이 아니다.
    out = classify_contact_type(FORCE, LOW, robot_contact=False, **KW)
    assert out == ["none", "none", "none"]


def test_a_loaded_region_off_the_floor_is_transient():
    out = classify_contact_type(FORCE, HIGH, robot_contact=True, **KW)
    assert out == ["none", "transient", "transient"]


def test_a_loaded_region_pinned_against_the_floor_is_quasistatic():
    # 넘어진 사람을 로봇이 누르는 상황 — 후퇴할 수 없다.
    out = classify_contact_type(FORCE, LOW, robot_contact=True, **KW)
    assert out == ["none", "quasistatic", "quasistatic"]


def test_regions_are_typed_independently_within_one_step():
    # 스키마 §8.1: 한 스텝에서 부위마다 다른 유형을 가질 수 있다.
    z = np.array([1.0, 0.05, 1.0])
    out = classify_contact_type(FORCE, z, robot_contact=True, **KW)
    assert out == ["none", "quasistatic", "transient"]


def test_a_region_below_the_noise_floor_is_none_even_when_pinned():
    # 눌려 있는 위치라도 하중이 없으면 접촉이 아니다.
    out = classify_contact_type(np.array([0.5]), np.array([0.0]), robot_contact=True, **KW)
    assert out == ["none"]


def test_the_pin_height_boundary_is_inclusive():
    at = classify_contact_type(np.array([50.0]), np.array([0.15]), robot_contact=True, **KW)
    just_above = classify_contact_type(np.array([50.0]), np.array([0.1501]), robot_contact=True, **KW)
    assert at == ["quasistatic"]
    assert just_above == ["transient"]


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        classify_contact_type(FORCE, np.array([1.0]), robot_contact=True, **KW)


def test_a_non_finite_height_is_refused():
    with pytest.raises(ValueError):
        classify_contact_type(FORCE, np.array([1.0, float("nan"), 1.0]),
                              robot_contact=True, **KW)


# ── 무게중심 ─────────────────────────────────────────────────────────


def test_center_of_mass_weights_by_mass_not_by_count():
    masses = np.array([1.0, 3.0])
    positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

    out = center_of_mass(masses, positions)

    # 산술평균이면 2.0. 질량 가중이면 3.0.
    assert out == pytest.approx([3.0, 0.0, 0.0])


def test_center_of_mass_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        center_of_mass(np.array([1.0, 2.0]), np.zeros((3, 3)))


def test_center_of_mass_rejects_xy_positions():
    # (N, 2) 를 넘기면 z 가 조용히 사라진 CoM 이 나온다.
    with pytest.raises(ValueError):
        center_of_mass(np.array([1.0]), np.zeros((1, 2)))


def test_center_of_mass_rejects_a_massless_body_set():
    # 질량 합 0 으로 나누면 inf/nan 이 조용히 흘러나간다.
    with pytest.raises(ValueError):
        center_of_mass(np.zeros(2), np.zeros((2, 3)))


def test_center_of_mass_rejects_an_empty_body_set():
    with pytest.raises(ValueError):
        center_of_mass(np.zeros(0), np.zeros((0, 3)))


# ── ZMP ──────────────────────────────────────────────────────────────


def test_zmp_of_a_body_at_rest_is_its_ground_projection():
    masses = np.array([10.0])
    positions = np.array([[0.3, -0.2, 1.0]])
    zero = np.zeros((1, 3))
    inertias = np.zeros((1, 3, 3))

    out = zero_moment_point(masses, positions, zero, zero, zero, inertias, gravity=9.81)

    assert out == pytest.approx([0.3, -0.2])


def test_zmp_shifts_opposite_the_horizontal_acceleration():
    # cart-table: x_zmp = x - z * a_x / (a_z + g). 앞으로 가속하면 ZMP 는 뒤로 간다.
    masses = np.array([10.0])
    positions = np.array([[0.0, 0.0, 1.0]])
    lin_acc = np.array([[2.0, 0.0, 0.0]])
    inertias = np.zeros((1, 3, 3))

    out = zero_moment_point(masses, positions, lin_acc, np.zeros((1, 3)),
                            np.zeros((1, 3)), inertias, gravity=9.81)

    assert out[0] == pytest.approx(-1.0 * 2.0 / 9.81)
    assert out[1] == pytest.approx(0.0)


def test_zmp_includes_the_angular_momentum_term():
    # I omega-dot 의 y 성분은 x_zmp 를 줄이고, x 성분은 y_zmp 를 늘린다.
    masses = np.array([10.0])
    positions = np.array([[0.0, 0.0, 1.0]])
    zero = np.zeros((1, 3))
    inertias = np.array([np.eye(3)])
    ang_acc = np.array([[3.0, 5.0, 0.0]])

    out = zero_moment_point(masses, positions, zero, zero, ang_acc, inertias, gravity=9.81)

    denom = 10.0 * 9.81
    assert out[0] == pytest.approx(-5.0 / denom)
    assert out[1] == pytest.approx(+3.0 / denom)


def test_zmp_of_a_multibody_set_is_the_weighted_combination():
    # 정지 상태의 두 물체 — CoM 의 지면 투영과 같아야 한다.
    masses = np.array([1.0, 3.0])
    positions = np.array([[0.0, 0.0, 1.0], [4.0, 2.0, 0.5]])
    zero = np.zeros((2, 3))
    inertias = np.zeros((2, 3, 3))

    out = zero_moment_point(masses, positions, zero, zero, zero, inertias, gravity=9.81)

    assert out == pytest.approx(center_of_mass(masses, positions)[:2])


def test_zmp_is_undefined_in_free_fall_rather_than_zero():
    # a_z = -g 면 분모가 사라진다. 0 을 돌려주면 지면 원점에 ZMP 가 있는 것으로 읽힌다.
    masses = np.array([10.0])
    positions = np.array([[0.5, 0.5, 1.0]])
    lin_acc = np.array([[0.0, 0.0, -9.81]])
    inertias = np.zeros((1, 3, 3))

    out = zero_moment_point(masses, positions, lin_acc, np.zeros((1, 3)),
                            np.zeros((1, 3)), inertias, gravity=9.81)

    assert np.isnan(out).all()


def test_a_non_finite_input_raises_rather_than_returning_the_free_fall_sentinel():
    """손상된 입력과 물리적 미정의를 같은 nan 으로 합치면 둘을 구분할 수 없다.

    자유낙하의 nan 은 "여기 ZMP 가 없다" 는 사실이고, NaN 질량의 nan 은 "입력이
    깨졌다" 는 사실이다. 배포본에서 둘은 다르게 다뤄져야 한다.
    """
    masses = np.array([10.0])
    positions = np.array([[0.0, 0.0, 1.0]])
    zero = np.zeros((1, 3))
    inertias = np.zeros((1, 3, 3))

    for bad in (
        dict(masses=np.array([math.nan])),
        dict(positions=np.array([[math.nan, 0.0, 1.0]])),
        dict(lin_acc=np.array([[0.0, 0.0, math.nan]])),
        dict(ang_vel=np.array([[math.inf, 0.0, 0.0]])),
        dict(ang_acc=np.array([[0.0, math.nan, 0.0]])),
        dict(inertias_w=np.full((1, 3, 3), math.nan)),
    ):
        kwargs = dict(masses=masses, positions=positions, lin_acc=zero, ang_vel=zero,
                      ang_acc=zero, inertias_w=inertias, gravity=9.81)
        kwargs.update(bad)
        with pytest.raises(ValueError):
            zero_moment_point(**kwargs)


def test_gravity_must_be_positive_and_finite():
    # gravity=0 은 예전에 ZeroDivisionError 로 터졌다. 중력이 없으면 지면 반력도
    # 없고 ZMP 라는 양 자체가 정의되지 않는다.
    args = (np.array([10.0]), np.array([[0.0, 0.0, 1.0]]), np.zeros((1, 3)),
            np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3, 3)))
    for bad in (0.0, -9.81, math.nan, math.inf):
        with pytest.raises(ValueError):
            zero_moment_point(*args, gravity=bad)


def test_zmp_includes_the_gyroscopic_term():
    """dH/dt = I·α + ω×(I·ω). 두 번째 항이 빠지면 회전 중 ZMP 가 틀린다.

    비대칭 관성 + 회전에서 링크당 0.7~17 N·m 이고 분모가 ~500 N 이므로
    ZMP 로 mm~cm 급이다.
    """
    masses = np.array([10.0])
    positions = np.array([[0.0, 0.0, 1.0]])
    zero = np.zeros((1, 3))
    inertias = np.array([np.diag([0.30, 0.05, 0.02])])
    omega = np.array([[2.0, 1.2, 0.6]])

    out = zero_moment_point(masses, positions, zero, omega, zero, inertias, gravity=9.81)

    gyro = np.cross(omega[0], inertias[0] @ omega[0])
    denom = 10.0 * 9.81
    assert out[0] == pytest.approx(-gyro[1] / denom)
    assert out[1] == pytest.approx(+gyro[0] / denom)
    assert out[0] != pytest.approx(0.0)   # 항이 빠졌다면 정확히 0 이다


def test_zmp_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        zero_moment_point(np.ones(2), np.zeros((2, 3)), np.zeros((3, 3)),
                          np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3, 3)),
                          gravity=9.81)


def test_zmp_rejects_flattened_inertias():
    # Isaac Lab 은 (9,) 로 준다. 그대로 넘기면 einsum 이 엉뚱한 축을 곱한다.
    with pytest.raises(ValueError):
        zero_moment_point(np.ones(1), np.zeros((1, 3)), np.zeros((1, 3)),
                          np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 9)),
                          gravity=9.81)


def test_zmp_rejects_an_empty_body_set():
    with pytest.raises(ValueError):
        zero_moment_point(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)),
                          np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3, 3)),
                          gravity=9.81)


def test_center_of_mass_rejects_a_non_finite_position():
    # 예전에는 NaN 좌표를 그대로 돌려줘서 평범한 좌표처럼 하류로 흘렀다.
    with pytest.raises(ValueError):
        center_of_mass(np.array([1.0]), np.array([[math.nan, 0.0, 0.0]]))


def test_center_of_mass_rejects_a_negative_mass():
    # 합이 양수면 통과하던 조합. 음수 질량은 자산이 깨졌다는 뜻이다.
    with pytest.raises(ValueError):
        center_of_mass(np.array([-1.0, 3.0]), np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]))


def test_center_of_mass_rejects_a_non_finite_mass():
    with pytest.raises(ValueError):
        center_of_mass(np.array([math.nan, 1.0]), np.zeros((2, 3)))


# ── 접촉 지점 상대속도 ────────────────────────────────────────────

def _one(pos, lin=(0, 0, 0), ang=(0, 0, 0)):
    return np.array([pos], float), np.array([list(lin) + list(ang)], float)


def test_pure_translation_is_just_the_difference_of_the_two_velocities():
    """회전이 없으면 지렛대가 얼마든 답이 같아야 한다."""
    d_pos, d_vel = _one((1.0, 0, 0), lin=(0.5, 0, 0))
    r_pos, r_vel = _one((0.0, 0, 0), lin=(-0.5, 0, 0))
    v = contact_relative_speed(d_pos, d_vel, r_pos, r_vel, lever_cap_m=0.265)
    assert v == pytest.approx([1.0])


def test_the_rotation_term_is_what_a_swinging_limb_contributes():
    """중심은 정지해 있고 회전만 하는 팔. 중심 속도만 쓰면 0 이 나온다."""
    d_pos, d_vel = _one((0.2, 0, 0))                      # 더미는 완전히 정지
    r_pos, r_vel = _one((0.0, 0, 0), ang=(0, 0, 10.0))    # 10 rad/s 로 회전
    v = contact_relative_speed(d_pos, d_vel, r_pos, r_vel, lever_cap_m=0.265)
    # 중심 거리 0.2 m → 지렛대는 0.1 m (상한 미만) → 10 rad/s x 0.1 m
    assert v == pytest.approx([1.0])


def test_the_cap_stops_the_velocity_field_being_extrapolated_off_the_body():
    """상한이 없으면 멀리 있는 물체의 회전이 말이 안 되는 속도를 만든다.

    실측으로 드러난 결함이다 — 상한 없이 돌린 첫 배치에서 1 m/s 로 걷는 로봇이
    중앙값 9 m/s · 최대 38 m/s 를 냈다. 강체의 속도장은 그 강체 안에서만 뜻이 있다.
    """
    d_pos, d_vel = _one((10.0, 0, 0))
    r_pos, r_vel = _one((0.0, 0, 0), ang=(0, 0, 10.0))
    capped = contact_relative_speed(d_pos, d_vel, r_pos, r_vel, lever_cap_m=0.265)
    assert capped == pytest.approx([2.65])                # 10 rad/s x 0.265 m
    # 상한을 크게 열면 그 허수가 그대로 돌아온다 — 상한이 하는 일이 이것뿐임을 못박는다.
    assert contact_relative_speed(
        d_pos, d_vel, r_pos, r_vel, lever_cap_m=1e3) == pytest.approx([50.0])


def test_each_dummy_body_is_paired_with_the_nearest_robot_body():
    """닿을 쌍이 아닌 바디와 맞추면 엉뚱한 속도가 나온다."""
    d_pos = np.array([[0.3, 0, 0]], float)
    d_vel = np.zeros((1, 6))
    r_pos = np.array([[0.0, 0, 0], [5.0, 0, 0]], float)
    r_vel = np.array([[0.0, 0, 0, 0, 0, 0], [7.0, 0, 0, 0, 0, 0]], float)
    v = contact_relative_speed(d_pos, d_vel, r_pos, r_vel, lever_cap_m=0.265)
    assert v == pytest.approx([0.0]), "먼 쪽(7 m/s)을 골랐습니다"


def test_coincident_centres_do_not_produce_nan():
    d_pos, d_vel = _one((0.0, 0, 0), ang=(0, 0, 3.0))
    r_pos, r_vel = _one((0.0, 0, 0), ang=(0, 0, -3.0))
    v = contact_relative_speed(d_pos, d_vel, r_pos, r_vel, lever_cap_m=0.265)
    assert np.isfinite(v).all() and v == pytest.approx([0.0])


@pytest.mark.parametrize("cap", [0.0, -0.1, float("nan"), float("inf")])
def test_a_cap_that_is_not_a_finite_positive_length_is_refused(cap):
    d_pos, d_vel = _one((1.0, 0, 0))
    r_pos, r_vel = _one((0.0, 0, 0))
    with pytest.raises(ValueError, match="지렛대"):
        contact_relative_speed(d_pos, d_vel, r_pos, r_vel, lever_cap_m=cap)


def test_the_lever_cap_comes_from_the_dummy_dimension_file():
    """수기 상수면 치수를 바꿔도 따라오지 않는다."""
    from generation.envs.s1a.mdp.safety import ANTHROPOMETRY_PATH, dummy_lever_cap

    cap = dummy_lever_cap(ANTHROPOMETRY_PATH)
    spec = json.loads(json.dumps(
        __import__("yaml").safe_load(Path(ANTHROPOMETRY_PATH).read_text(encoding="utf-8"))))
    biggest = max(max(map(float, r["size"])) / 2 for r in spec["regions"].values() if "size" in r)
    assert cap >= biggest
    assert 0.05 < cap < 1.0, f"인체 부위 반치수로 볼 수 없는 값입니다: {cap}"


# ── 부위 축약: 힘·높이·로봇접촉이 같은 서브스텝에서 와야 한다 ──────────

def test_the_region_takes_the_height_of_the_body_that_carried_its_peak():
    """쌍 부위에서 힘은 한 바디, 높이는 다른 바디에서 오면 표 A.2 열이 뒤집힙니다."""
    from generation.envs.s1a.mdp.safety import reduce_region_peaks

    codes = ["BR-11", "BR-11"]                 # 좌우 대퇴 한 쌍
    force = [900.0, 5.0]                       # 왼쪽이 최댓값
    z = [1.20, 0.03]                           # 오른쪽만 바닥에 붙어 있다
    f, h, c = reduce_region_peaks(codes, force, z, [True, True], contact_eps_n=1.0)
    assert f["BR-11"] == pytest.approx(900.0)
    assert h["BR-11"] == pytest.approx(1.20), "가장 낮은 바디의 높이를 가져왔습니다"


def test_the_region_takes_the_robot_contact_flag_of_that_same_body():
    from generation.envs.s1a.mdp.safety import reduce_region_peaks

    f, h, c = reduce_region_peaks(
        ["BR-05", "BR-05"], [700.0, 10.0], [1.3, 1.3], [False, True], contact_eps_n=1.0)
    assert c["BR-05"] is False, "다른 바디의 접촉 표식을 가져왔습니다"


def test_an_untouched_region_falls_back_to_its_lowest_body():
    """접촉이 없으면 argmax 가 임의라, 「이 부위가 바닥에 있나」 를 답하도록 되돌립니다."""
    from generation.envs.s1a.mdp.safety import reduce_region_peaks

    f, h, c = reduce_region_peaks(
        ["BR-12", "BR-12"], [0.0, 0.0], [0.5, 0.04], [False, False], contact_eps_n=1.0)
    assert h["BR-12"] == pytest.approx(0.04)


def test_a_ragged_reduction_input_is_refused():
    from generation.envs.s1a.mdp.safety import reduce_region_peaks

    with pytest.raises(ValueError, match="길이가 다릅니다"):
        reduce_region_peaks(["BR-01"], [1.0, 2.0], [0.1, 0.2], True, contact_eps_n=1.0)


def test_contact_type_can_be_gated_per_region_not_only_globally():
    """한 부위만 로봇이 닿았을 때 나머지가 함께 유형을 얻으면 안 됩니다."""
    types = classify_contact_type(
        [500.0, 500.0], [1.3, 1.3],
        robot_contact=[True, False], contact_eps_n=1.0, pin_height_m=0.2)
    assert types == ["transient", "none"]
