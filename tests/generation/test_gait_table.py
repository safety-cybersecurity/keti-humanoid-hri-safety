"""generation/dummy/gait/gait_table.py -- joint targets for a speed and a phase, the stride, the phase
and the entry's speed profile. Pure; runs in CI."""
import numpy as np
import pytest

from generation.dummy.gait import gait_table as gt


def _table():
    P, J = 8, 2
    targets = np.zeros((3, P, J), np.float32)
    targets[1, :, 0] = np.sin(2 * np.pi * np.arange(P) / P)
    targets[2, :, 0] = 2 * np.sin(2 * np.pi * np.arange(P) / P)
    targets[1:, :, 1] = 0.5
    return gt.GaitTable(("a", "b"), np.array([0.0, 0.5, 1.5]), np.array([0.0, 0.8, 1.6]), targets)


def test_round_trip_through_json():
    t = _table()
    back = gt.GaitTable.from_json(t.to_json())
    assert back.dof_names == t.dof_names
    assert (back.targets == t.targets).all() and (back.speeds == t.speeds).all()


def test_at_interpolates_speed_and_phase():
    t = _table()
    out = t.at(np.array([1.0]), np.array([0.25]))
    assert out[0, 0] == pytest.approx(1.5)                 # halfway between rows 1 and 2 at phase 0.25
    assert out[0, 1] == pytest.approx(0.5)
    mid = t.at(np.array([0.5]), np.array([0.25 + 1 / 16]))
    assert mid[0, 0] == pytest.approx((np.sin(np.pi / 2) + np.sin(2 * np.pi * 3 / 8)) / 2)


def test_below_the_slowest_row_the_gait_shrinks_toward_standing():
    t = _table()
    assert t.at(np.array([0.0]), np.array([0.25])) == pytest.approx(np.zeros((1, 2)))
    q = t.at(np.array([0.125]), np.array([0.25]))
    assert q[0, 0] == pytest.approx(np.sqrt(0.25) * 1.0)
    assert t.stride_at(np.array([0.125])) == pytest.approx([0.8 * 0.5])


def test_slowing_below_the_slowest_row_the_gait_keeps_that_rows_shape_and_stride():
    # the legs keep the slowest row's steps, ever slower, so a foot that is down stays where it landed as the body
    # comes to rest; the stance a walk ends in is the term's to set (support.rest_stance)
    t = _table()
    q = t.at(np.array([0.125]), np.array([0.25]), slowing=np.array([True]))
    assert q[0, 0] == pytest.approx(1.0)                              # the slowest row's target at that phase
    assert t.stride_at(np.array([0.02, 0.125]), slowing=np.array([True, True])) == pytest.approx([0.8, 0.8])


def test_each_env_shrinks_by_its_own_slowing():
    t = _table()
    q = t.at(np.array([0.125, 0.125]), np.array([0.25, 0.25]), slowing=np.array([True, False]))
    assert q[:, 0] == pytest.approx([1.0, 0.5])
    assert t.stride_at(np.array([0.125, 0.125]), slowing=np.array([True, False])) == pytest.approx([0.8, 0.4])


@pytest.mark.parametrize("start", [0.1, 0.3, 0.6, 0.85])
def test_a_stop_at_a_constant_deceleration_comes_to_the_stand_pose_without_a_jump(start):
    # from the slowest row's speed down to rest over a second, a step at a time: the last step's change of the targets
    # is about the size of the steps' before it -- nothing is left to the last step (shrunk by sqrt(v / v1), as a start
    # is, the last step would take 2.4 times the largest change of the ten before it)
    t = _table()
    dt, phase, prev, changes = 0.02, start, None, []
    for v in np.linspace(0.5, 0.0, 51):
        phase = float(gt.advance_phase(phase, v * dt, t.stride_at(np.array([v]), slowing=np.array([True]))[0]))
        q = t.at(np.array([v]), np.array([phase]), slowing=np.array([True]))[0]
        if prev is not None:
            changes.append(float(np.abs(q - prev).max()))
        prev = q
    assert changes[-1] <= 1.5 * max(changes[-11:-1]), changes[-11:]


def test_above_the_fastest_row_is_the_fastest_row():
    t = _table()
    assert t.at(np.array([9.0]), np.array([0.25])) == pytest.approx(t.at(np.array([1.5]), np.array([0.25])))


def test_phase_wraps_and_never_goes_back():
    p = gt.advance_phase(np.array([0.9, 0.2]), np.array([0.4, 0.0]), np.array([1.0, 1.0]))
    assert p == pytest.approx([0.3, 0.2])
    assert gt.advance_phase(np.array([0.5]), np.array([0.3]), np.array([0.0])) == pytest.approx([0.5])


def test_no_nan_at_a_standstill():
    t = _table()
    for v in (0.0, 1e-12, 1e-6):
        assert np.isfinite(t.at(np.array([v]), np.array([0.7]))).all()
        assert np.isfinite(t.stride_at(np.array([v]))).all()


def test_ramp_rises_then_comes_to_rest_at_the_line():
    v, dt, remaining, traveled = 0.0, 0.02, 3.0, 0.0
    peak = 0.0
    for _ in range(2000):
        v = float(gt.ramp_speed(np.array([v]), np.array([1.2]), dt, 1.0, 1.0, np.array([remaining - traveled]))[0])
        traveled += v * dt
        peak = max(peak, v)
    assert peak == pytest.approx(1.2)
    assert traveled == pytest.approx(3.0, abs=0.03)
    assert v < 0.05


def test_ramp_without_a_stop_line_holds_the_speed():
    v = gt.ramp_speed(np.array([1.2]), np.array([1.2]), 0.02, 1.0, 1.0)
    assert v == pytest.approx([1.2])


def test_remaining_to_stop_is_along_the_entry_line():
    assert gt.remaining_to_stop(np.array([3.0]), np.array([-1.0]), np.array([np.pi / 6])) == pytest.approx([8.0])
    assert gt.remaining_to_stop(np.array([-2.0]), np.array([-1.0]), np.array([np.pi / 2])) == pytest.approx([0.0])


@pytest.mark.parametrize("bad", ["speeds", "stride", "targets"])
def test_a_malformed_table_is_refused(bad):
    t = _table()
    kw = dict(dof_names=t.dof_names, speeds=t.speeds.copy(), stride=t.stride.copy(), targets=t.targets.copy())
    if bad == "speeds":
        kw["speeds"][2] = 0.4
    elif bad == "stride":
        kw["stride"][0] = 0.1
    else:
        kw["targets"][0, 0, 0] = 0.1
    with pytest.raises(ValueError):
        gt.GaitTable(**kw)


@pytest.mark.parametrize("field,index,value", [("speeds", 1, np.nan), ("speeds", 2, np.inf), ("stride", 1, np.nan),
                                               ("stride", 2, np.inf)])
def test_a_table_with_a_non_finite_speed_or_stride_is_refused(field, index, value):
    # A NaN fails every ordering and positivity comparison, so only a finiteness check refuses it: let through,
    # a NaN speed makes NaN targets and a NaN stride leaves the phase where it is while the target walks on.
    t = _table()
    kw = dict(dof_names=t.dof_names, speeds=t.speeds.copy(), stride=t.stride.copy(), targets=t.targets.copy())
    kw[field][index] = value
    with pytest.raises(ValueError):
        gt.GaitTable(**kw)
