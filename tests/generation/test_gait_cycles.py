"""generation/dummy/gait/cycles.py -- units, body axes, foot contacts and gait cycles. Pure; runs in CI."""
import numpy as np
import pytest

from generation.dummy.gait import bvh, cycles
from tests.generation.gait_fixture import synthetic_walk_bvh, synthetic_walk_segments_bvh


@pytest.fixture(scope="module")
def walk():
    m = bvh.parse_bvh(synthetic_walk_bvh(speed=1.2, seconds=6.0))
    s = cycles.unit_scale(m.skeleton)
    pos, rot = bvh.forward_kinematics(m, scale=s)
    return m, s, pos, rot


def test_centimetres_are_recognised(walk):
    m, s, *_ = walk
    assert s == 0.01


def test_body_axes_front_left_up(walk):
    m, *_ = walk
    C = cycles.body_axes(m.skeleton)
    assert C[2] == pytest.approx([0, 1, 0])        # BVH +Y is up
    assert C[1] == pytest.approx([1, 0, 0])        # LeftHip is at +X
    assert C[0] == pytest.approx([0, 0, 1])        # front = left x up
    assert np.linalg.det(C) == pytest.approx(1.0)


def test_cycles_last_one_period_and_alternate(walk):
    m, s, pos, rot = walk
    C = cycles.body_axes(m.skeleton)
    fps = round(1 / m.frame_time)
    feet = {}
    for side in ("left", "right"):
        h, v = cycles.foot_signals(pos, m.skeleton, C, side, fps)
        feet[side] = cycles.heel_strikes(cycles.foot_contacts(h, v, height_tol=0.04, speed_tol=0.3, min_frames=3))
    got = cycles.gait_cycles(feet["right"], feet["left"], fps, min_s=0.6, max_s=2.0)
    stride = 4 * 0.84 * np.sin(np.radians(18))
    assert len(got) >= 3
    for a, b in got:
        assert (b - a) / fps == pytest.approx(stride / 1.2, rel=0.05)


def test_resample_keeps_the_ends():
    v = np.arange(11.0)
    out = cycles.resample_cycle(v, 0, 10, 5)
    assert out == pytest.approx([0, 2, 4, 6, 8])


def test_a_skeleton_in_neither_unit_is_refused(walk):
    m, *_ = walk
    sk = m.skeleton
    bad = bvh.Skeleton(sk.names, sk.parents, sk.offsets * 10.0, sk.channels)
    with pytest.raises(ValueError, match="단위"):
        cycles.unit_scale(bad)


def test_the_pelvis_heading_follows_the_walks_turn():
    """Straight for 2 s, then turning left at 30 deg/s for 8 s: the heading is 0 through the straight part, gains
    30 deg a second after it and is unwrapped past 180 (the root's yaw is written to 1e-5 deg)."""
    m = bvh.parse_bvh(synthetic_walk_segments_bvh([(1.2, 2.0, 0.0), (1.2, 8.0, 30.0)]))
    pos, rot = bvh.forward_kinematics(m, scale=cycles.unit_scale(m.skeleton))
    heading = cycles.pelvis_heading_deg(rot, m.skeleton, cycles.body_axes(m.skeleton))
    fps = round(1 / m.frame_time)
    assert heading[:2 * fps] == pytest.approx(0.0, abs=1e-4)
    assert heading[2 * fps + 3 * fps] == pytest.approx(90.0, abs=1e-4)
    assert heading[-1] == pytest.approx(30.0 * (8.0 - 1.0 / fps), abs=1e-4)      # 239.5, not -120.5


def test_make_periodic_takes_the_drift_out_in_proportion_to_phase():
    """A cycle sampled at phases i / P whose end (phase 1) lies `drift` from its start: the drift comes off in proportion
    to phase, so the cycle ends where it began and a straight ramp becomes flat."""
    P = 50
    ramp = np.stack([np.arange(P) * 0.02, np.full(P, 3.0)], -1)          # ends at 1.0 (phase 1), constant 3
    flat = cycles.make_periodic(ramp, np.array([1.0, 0.0]))
    assert flat == pytest.approx(np.stack([np.zeros(P), np.full(P, 3.0)], -1))
