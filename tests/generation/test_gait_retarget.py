"""generation/dummy/gait/retarget.py -- 100STYLE's skeleton onto the jointed dummy. Pure; runs in CI."""
import math

import numpy as np
import pytest

from generation.dummy import joints_spec as js
from generation.dummy.gait import bvh, cycles, kinematics as kin, retarget as rt
from tests.generation.gait_fixture import synthetic_walk_bvh


@pytest.fixture(scope="module")
def spec():
    return js.load()


@pytest.fixture(scope="module")
def clip():
    m = bvh.parse_bvh(synthetic_walk_bvh(speed=1.2, seconds=6.0))
    pos, rot = bvh.forward_kinematics(m, scale=cycles.unit_scale(m.skeleton))
    return m, pos, rot, cycles.body_axes(m.skeleton)


def test_rest_pose_is_all_zero(spec, clip):
    m, pos, rot, C = clip
    rest = np.broadcast_to(np.eye(3), (1,) + rot.shape[1:]).copy()
    # arms of the fixture hang at Z rotation -80/+80 in frame 0; the T-pose itself puts the arms out, so only
    # the legs are asserted at rest
    dofs = rt.frame_dofs(rest, m.skeleton, C, spec)
    cols = js.columns(spec)
    for name in ("hip_left", "hip_right", "knee_left", "knee_right", "ankle_left", "ankle_right"):
        assert dofs[0, cols[name]] == pytest.approx(0.0, abs=1e-9)


def test_hip_flexion_comes_out_negative_y(spec, clip):
    m, pos, rot, C = clip
    dofs = rt.frame_dofs(rot, m.skeleton, C, spec)
    cols = js.columns(spec)
    hip_y = dofs[:, cols["hip_left"]][:, 1]
    assert hip_y.min() == pytest.approx(-math.radians(18), abs=math.radians(1.5))
    assert hip_y.max() == pytest.approx(math.radians(18), abs=math.radians(1.5))
    assert dofs[:, cols["knee_left"]].min() >= -1e-9


def test_the_pelvis_tilt_is_folded_into_the_hips(spec, clip):
    """The dummy's pelvis stays upright, so the legs are taken in the pelvis's heading frame -- the actor's pelvis with
    its tilt (pitch and roll) folded into the hips -- and keep their orientation to the floor: a pelvis tipped back and
    rolled over legs that stay where they are in the world leaves every leg column as it was, and so does turning the
    whole body about the vertical (the heading is not folded in)."""
    m, pos, rot, C = clip
    sk = m.skeleton
    ip = sk.index("Hips")
    assert np.allclose(rot[:, ip], np.eye(3))                  # the fixture's pelvis is level and faces the front

    def world(R):                                              # a rotation in body axes, written in the file's axes
        return C.T @ R @ C

    turn = kin.axis_rotation("z", 0.7)
    turned = np.einsum("ij,fkjl->fkil", world(turn), rot)
    tipped = turned.copy()
    tipped[:, ip] = world(turn @ kin.axis_rotation("y", -0.105) @ kin.axis_rotation("x", 0.052))
    cols = js.columns(spec)
    legs = [c for j in spec.joints if j.group in ("hip", "knee", "ankle") for c in range(cols[j.name].start, cols[j.name].stop)]
    base = rt.frame_dofs(rot, sk, C, spec)[:, legs]
    for variant in (turned, tipped):
        assert rt.frame_dofs(variant, sk, C, spec)[:, legs] == pytest.approx(base, abs=1e-9)


def test_clamp_counts_what_it_cut(spec):
    row = np.zeros((2, len(js.dof_names(spec))))
    cols = js.columns(spec)
    row[0, cols["knee_left"]] = math.radians(150)
    out, cut = rt.clamp(row, spec)
    assert out[0, cols["knee_left"]] == pytest.approx(math.radians(135))
    assert cut == {"knee_left": 1}


def test_stance_correction_puts_both_soles_on_the_floor(spec):
    cols = js.columns(spec)
    row = np.zeros((1, len(js.dof_names(spec))))
    row[0, cols["knee_left"]] = math.radians(20)            # left sole would hover in double support
    corrected, shift, change = rt.correct_stance(row, np.array([[True, True]]), spec, blend=0)
    lows = []
    for side in ("left", "right"):
        R_h, kn, R_a = rt.leg_matrices(corrected[0], spec, side)
        _, sole, _ = kin.leg_fk(js.leg_geometry(spec, side), R_h, kn, R_a)
        lows.append(sole[:, 2].min() + shift[0])
    assert lows == pytest.approx([0.0, 0.0], abs=1e-3)
    assert change > 0


def test_stance_correction_eases_in_and_out_of_double_support(spec):
    """The ankle raise the correction gives a leg in double support is carried into the `blend` phases on each side
    where that foot is still down, falling in equal steps -- so neither its knee nor the pelvis jumps where the other
    foot lands or lifts -- and phases further away, or with that foot in the air, are left as they were. Here the
    left sole hovers (knee bent, the foot kept level) so the right leg is the one flexed."""
    cols = js.columns(spec)
    ank = js.joint(spec, "ankle_left")
    P = 40
    cyc = np.zeros((P, len(js.dof_names(spec))))
    cyc[:, cols["knee_left"]] = math.radians(20)
    cyc[:, cols[ank.name].start + js.axes(ank).index("y")] = -math.radians(20)
    stance = np.zeros((P, 2), bool)
    stance[:10] = stance[20:30] = True                      # double support 0-9 and 20-29
    stance[10:20, 1] = True                                 # the right foot alone 10-19
    stance[30:, 0] = True                                   # the left foot alone 30-39 (the right swings)
    geom = js.leg_geometry(spec, "right")

    def ankle_z(row):
        return kin.leg_fk(geom, *rt.leg_matrices(row, spec, "right"))[0][2]

    sharp, sharp_shift, _ = rt.correct_stance(cyc, stance, spec, blend=0)
    soft, shift, _ = rt.correct_stance(cyc, stance, spec, blend=3)
    full = ankle_z(sharp[0]) - ankle_z(cyc[0])
    assert full > 0.005
    rise = np.array([ankle_z(soft[p]) - ankle_z(cyc[p]) for p in range(P)])
    steps = full * np.array([3, 2, 1]) / 4
    for a, b in ((0, 9), (20, 29)):
        assert rise[a:b + 1] == pytest.approx(np.full(10, full), abs=1e-6)
    assert rise[[10, 11, 12]] == pytest.approx(steps, abs=1e-6)            # easing out, the right foot alone down
    assert rise[[19, 18, 17]] == pytest.approx(steps, abs=1e-6)            # easing in
    assert rise[13:17] == pytest.approx(np.zeros(4), abs=1e-9)
    assert rise[30:40] == pytest.approx(np.zeros(10), abs=1e-9)            # the right foot in the air: not eased
    np.testing.assert_array_equal(soft[:, cols["knee_left"]], cyc[:, cols["knee_left"]])
    assert np.abs(np.diff(shift[7:15])).max() == pytest.approx(full / 4, abs=1e-6)          # the pelvis eases too
    assert np.abs(np.diff(sharp_shift[7:15])).max() == pytest.approx(full, abs=1e-6)        # where it jumped before


def test_stance_correction_can_keep_the_joint_ranges(spec):
    """With keep_ranges a leg the correction flexes stays in its joints' ranges: where keeping its foot's orientation
    would take the ankle past one, the ankle is held there (the foot turns with the leg) and the flex goes on until
    both soles reach the floor at one pelvis height; `held` counts the held raises per degree of freedom. Without it the
    ankle goes past its range."""
    cols = js.columns(spec)
    al, ar = js.joint(spec, "ankle_left"), js.joint(spec, "ankle_right")
    yl = cols[al.name].start + js.axes(al).index("y")
    yr = cols[ar.name].start + js.axes(ar).index("y")
    lower = ar.dofs[js.axes(ar).index("y")].lower
    row = np.zeros((1, len(js.dof_names(spec))))
    row[0, cols["knee_left"]] = math.radians(20)            # the left sole hovers, its foot kept level
    row[0, yl] = -math.radians(20)
    row[0, yr] = lower + math.radians(5)                     # the right ankle 5 deg inside its range
    loose, _, _ = rt.correct_stance(row, np.array([[True, True]]), spec, blend=0)
    assert loose[0, yr] < lower - math.radians(5)
    held = {}
    kept, shift, _ = rt.correct_stance(row, np.array([[True, True]]), spec, passes=8, blend=0, keep_ranges=True, held=held)
    assert kept[0, yr] == pytest.approx(lower)
    lows = [kin.leg_fk(js.leg_geometry(spec, s), *rt.leg_matrices(kept[0], spec, s))[1][:, 2].min() + shift[0]
            for s in rt.SIDES]
    assert lows == pytest.approx([0.0, 0.0], abs=1e-5)
    assert held.get("ankle_right.y", 0) >= 1 and set(held) <= {"ankle_right.y", "ankle_right.x", "ankle_right.z",
                                                                 "hip_right.x", "hip_right.y", "hip_right.z", "knee_right"}


def test_cycle_speed_matches_the_fixture_walk(spec, clip):
    m, pos, rot, C = clip
    fps = round(1 / m.frame_time)
    dofs = rt.frame_dofs(rot, m.skeleton, C, spec)
    feet = {}
    contact = {}
    for side in ("left", "right"):
        h, v = cycles.foot_signals(pos, m.skeleton, C, side, fps)
        contact[side] = cycles.foot_contacts(h, v, height_tol=0.04, speed_tol=0.3, min_frames=3)
        feet[side] = cycles.heel_strikes(contact[side])
    a, b = cycles.gait_cycles(feet["right"], feet["left"], fps, min_s=0.6, max_s=2.0)[1]
    cyc = cycles.resample_cycle(dofs, a, b, 100)
    stance = np.stack([cycles.resample_cycle(contact[s].astype(float), a, b, 100) > 0.5
                       for s in ("left", "right")], -1)
    cyc, _, _ = rt.correct_stance(cyc, stance, spec, blend=5)
    v, stride = rt.cycle_speed(cyc, stance, (b - a) / fps, spec, floor_share=0.25)
    # the dummy's legs are not the fixture's (0.79 m hip-to-ankle against 0.84 m): speed scales with them
    assert v == pytest.approx(1.2 * 0.79 / 0.84, rel=0.1)
    assert stride == pytest.approx(v * (b - a) / fps)
    adv, floored, gaps = rt.body_advance(rt.stance_advance(cyc, stance, spec), 0.25)
    assert stride == pytest.approx(float(adv.sum()), abs=1e-12)               # the body's advance over the cycle
    assert (floored, gaps) == (0, 0)


def test_stance_advance_rolls_the_stance_foot_from_heel_to_toe(spec):
    """One stance run with the hip and knee held, so the ankle stays put in the pelvis frame, while the ankle tips the
    foot from toes up to heel up. A foot that rolls without slipping -- on its heel until it is flat, then on its
    toe -- carries the pelvis forward by what those two corners travel back in the pelvis frame, one phase step at a
    time; keeping the ankle still would say the pelvis did not move. A foot that is not down at both ends of a step
    says nothing about it."""
    cols = js.columns(spec)
    ank = js.joint(spec, "ankle_left")
    y = cols[ank.name].start + js.axes(ank).index("y")
    geom = js.leg_geometry(spec, "left")

    def sole(a):
        row = np.zeros(len(js.dof_names(spec)))
        row[y] = a
        return kin.leg_fk(geom, *rt.leg_matrices(row, spec, "left"))[1]

    flat = sole(0.0)
    toe, heel = int(np.argmax(flat[:, 0])), int(np.argmin(flat[:, 0]))
    up = 0.15 if sole(0.15)[toe, 2] > sole(0.15)[heel, 2] else -0.15            # the sign that lifts the toes
    start, end = up, -up / 0.15                                                   # toes up 8.6 deg -> heel up 57 deg
    assert sole(start)[:, 2].argmin() in np.flatnonzero(np.isclose(flat[:, 0], flat[heel, 0]))
    assert sole(end)[:, 2].argmin() in np.flatnonzero(np.isclose(flat[:, 0], flat[toe, 0]))
    P, n = 200, 141
    cyc = np.zeros((P, len(js.dof_names(spec))))
    cyc[:n, y] = np.linspace(start, end, n)
    stance = np.zeros((P, 2), bool)
    stance[:n, 0] = True
    travel = (sole(start)[heel, 0] - flat[heel, 0]) + (flat[toe, 0] - sole(end)[toe, 0])
    assert travel > 0.1
    adv = rt.stance_advance(cyc, stance, spec)
    assert np.isfinite(adv[:n - 1, 0]).all() and np.isnan(adv[n - 1:, 0]).all() and np.isnan(adv[:, 1]).all()
    assert adv[:n - 1, 0].sum() == pytest.approx(travel, abs=1e-4)


def test_body_advance_follows_the_foot_down_alone_or_the_one_that_landed_last():
    """Per phase step: the foot that is down alone; in double support the foot that landed last -- its run of steps
    began later (the cycle wraps round) -- as the body goes over the foot it has just put down; feet whose runs began
    together give their mean; a step with no foot down takes the advance between its neighbours'. Counted: none
    raised, one gap."""
    nan = np.nan
    # the left foot's run of steps is 5, 6, 7, 0, 1 (it lands at 5, after the right's run 4, 5 began); the right's runs
    # are 4, 5 and 1, 2 (it lands at 1, inside the left's run)
    per_leg = np.array([[2.0, nan], [2.0, 4.0], [nan, 3.0], [nan, nan], [nan, 5.0], [6.0, 2.0], [3.0, nan], [3.0, nan]])
    adv, floored, gaps = rt.body_advance(per_leg, 0.25)
    assert adv == pytest.approx([2.0, 4.0, 3.0, 4.0, 5.0, 6.0, 3.0, 3.0])
    assert (floored, gaps) == (0, 1)
    both, _, _ = rt.body_advance(np.array([[1.0, 3.0], [1.0, 3.0]]), 0.25)           # down together all through
    assert both == pytest.approx([2.0, 2.0])


def test_body_advance_never_runs_backward():
    """Where the rolling foot would carry the body backward or barely forward, the body creeps on at a share of the
    cycle's mean step instead (and that foot slips) -- so equal shares of the advance always run forward."""
    per_leg = np.array([[-1.0, np.nan], [4.0, np.nan], [4.0, np.nan], [0.5, np.nan]])
    adv, floored, gaps = rt.body_advance(per_leg, 0.25)
    floor = 0.25 * (-1.0 + 4.0 + 4.0 + 0.5) / 4
    assert adv == pytest.approx([floor, 4.0, 4.0, 0.5])
    assert (floored, gaps) == (1, 0)
    with pytest.raises(ValueError, match="발"):
        rt.body_advance(np.full((4, 2), np.nan), 0.25)


def test_retime_puts_each_sample_at_an_equal_share_of_the_advance():
    """Sample i of the re-timed cycle sits at the share i / P of the body's advance over the cycle, so each phase sample
    carries the body the same distance: the joints are linear between the two original samples around it (the cycle
    wraps round), the stance comes from the nearer one, and the extra per-phase arrays go as the joints. Sample 0 stays
    where it is."""
    P = 10
    x = np.arange(P, dtype=float)
    cyc = np.stack([x, 2.0 * x], -1)
    stance = np.stack([x < 3, x >= 3], -1)
    adv = np.array([3.0] + [1.0] * 8 + [3.0])                 # 14 in all: 1.4 a sample; sample p at 0, 3, 4, ..., 11
    out, st, shift = rt.retime(cyc, stance, adv, -x)
    # where each new sample falls among the old ones: 1.4 and 2.8 inside the first step, 4.2 two tenths into the third,
    # ... and 11.2, 12.6 inside the last step, which runs from sample 9 back round to sample 0
    where = np.array([0.0, 1.4 / 3, 2.8 / 3, 2.2, 3.6, 5.0, 6.4, 7.8, 9.0 + 0.2 / 3, 9.0 + 1.6 / 3])
    want = np.where(where < 9.0, where, 9.0 * (1.0 - (where - 9.0)))
    assert out[:, 0] == pytest.approx(want)
    assert out[:, 1] == pytest.approx(2.0 * want)
    assert shift == pytest.approx(-want)
    near = np.where(where - np.floor(where) < 0.5, np.floor(where), np.floor(where) + 1).astype(int) % P
    np.testing.assert_array_equal(st, stance[near])


def test_retime_with_equal_advances_leaves_the_cycle_as_it_was():
    rng = np.random.default_rng(2)
    cyc = rng.normal(size=(20, 3))
    stance = rng.random((20, 2)) > 0.5
    out, st = rt.retime(cyc, stance, np.full(20, 0.7))
    assert out == pytest.approx(cyc, abs=1e-12)
    np.testing.assert_array_equal(st, stance)


def test_retime_needs_every_step_to_go_forward():
    with pytest.raises(ValueError, match="나아감"):
        rt.retime(np.zeros((3, 1)), np.ones((3, 2), bool), np.array([1.0, 0.0, 1.0]))


def test_stance_slip_is_how_far_a_foot_slides_at_a_steady_pace(spec):
    """At a steady pace with the phase advanced by distance (stride / P a phase), a stance foot slides by what the body
    goes beyond that foot's own advance: a right foot down and still for a whole 10-phase cycle (its nine steps)
    slides a tenth of the stride every step -- net and to-and-fro nine tenths, the largest step a tenth -- and two feet
    down and still together never disagree."""
    P, stride = 10, 0.5
    cyc = np.zeros((P, len(js.dof_names(spec))))
    right = np.zeros((P, 2), bool)
    right[:, 1] = True
    got = rt.stance_slip(cyc, right, stride, spec)
    assert got["net_m"] == pytest.approx(0.45)
    assert got["to_and_fro_m"] == pytest.approx(0.45)
    assert got["largest_step_m"] == pytest.approx(0.05)
    assert got["double_support_disagreement_m"] == 0.0
    both = rt.stance_slip(cyc, np.ones((P, 2), bool), stride, spec)
    assert both["double_support_disagreement_m"] == pytest.approx(0.0, abs=1e-12)


def _two_step_cycle(spec):
    """A sagittal 40-phase cycle: the left foot down 0-23 and the right 20-39 and 0-3 (double support 0-3 and
    20-23); each hip sweeps back through its stance and forward through its swing -- the left a little wider, so the
    two feet roll different distances and slip at a steady pace -- the knee bent 0.45 rad in stance (enough for the legs
    to reach every held position) and more in swing."""
    P = 40
    cols = js.columns(spec)
    cyc = np.zeros((P, len(js.dof_names(spec))))
    stance = np.zeros((P, 2), bool)
    stance[0:24, 0] = True
    stance[20:40, 1] = stance[0:4, 1] = True
    for side, start, amp in (("left", 0, 0.32), ("right", 20, 0.28)):
        hip = js.joint(spec, f"hip_{side}")
        ph = (np.arange(P) - start) % P / P                                   # 0 where that foot lands
        cyc[:, cols[hip.name].start + js.axes(hip).index("y")] = np.where(
            ph < 0.6, -amp + 2 * amp * ph / 0.6, amp - 2 * amp * (ph - 0.6) / 0.4)
        cyc[:, cols[f"knee_{side}"].start] = np.where(ph < 0.6, 0.45, 0.45 + 0.6 * np.sin(np.pi * (ph - 0.6) / 0.4))
    stride = float(rt.body_advance(rt.stance_advance(cyc, stance, spec), 0.25)[0].sum())
    return cyc, stance, stride


def _leg_columns(spec, side):
    cols = js.columns(spec)
    return [c for j in spec.joints if j.side == side and j.group in ("hip", "knee", "ankle")
            for c in range(cols[j.name].start, cols[j.name].stop)]


def _foot(row, spec, side):
    return kin.leg_fk(js.leg_geometry(spec, side), *rt.leg_matrices(row, spec, side))


def test_lock_stance_holds_each_stance_foot_where_it_landed(spec):
    """At a steady pace with the phase advanced by distance (stride / P a phase), each stance foot is held where it
    landed: from its landing sample on, the leg IK moves the foot along the pelvis frame's x until the sole corner it
    rolls on is one pace behind where it was a phase before -- every stance step then advances the body by the pace and
    nothing slides, the feet that slid before included. The landing samples and every column but the moved leg's stay
    as they were, and the foot keeps its orientation and its height in the pelvis frame (this cycle's legs reach)."""
    cyc, stance, stride = _two_step_cycle(spec)
    P = len(cyc)
    assert rt.stance_slip(cyc, stance, stride, spec)["to_and_fro_m"] > 0.01
    locked, offset, short = rt.lock_stance(cyc, stance, stride, spec)
    assert short == 0
    per = rt.stance_advance(locked, stance, spec)
    assert per[np.isfinite(per)] == pytest.approx(np.full(np.isfinite(per).sum(), stride / P), abs=1e-9)
    assert rt.stance_slip(locked, stance, stride, spec)["to_and_fro_m"] < 1e-9
    legs = {s: _leg_columns(spec, s) for s in rt.SIDES}
    other = [c for c in range(cyc.shape[1]) if c not in legs["left"] + legs["right"]]
    np.testing.assert_array_equal(locked[:, other], cyc[:, other])
    for k, side in enumerate(rt.SIDES):
        landing = rt._runs(stance[:, k])[0][0] % P
        np.testing.assert_array_equal(locked[landing, legs[side]], cyc[landing, legs[side]])
        assert offset[landing, k] == 0.0
        for p in np.flatnonzero(stance[:, k]):
            _, sole0, R0 = _foot(cyc[p], spec, side)
            _, sole1, R1 = _foot(locked[p], spec, side)
            assert R1 == pytest.approx(R0, abs=1e-9)
            assert sole1[:, 2] == pytest.approx(sole0[:, 2], abs=1e-9)
            assert sole1[:, 0] - sole0[:, 0] == pytest.approx(np.full(len(sole0), offset[p, k]), abs=1e-9)
    assert np.abs(offset).max() > 0.01


def test_lock_stance_takes_what_is_left_at_lift_off_back_over_the_swing(spec):
    """Where a held foot lifts, what is left of its move is taken back over the swing that follows, in equal steps, to
    nothing at the next landing: the swing foot moved along x by that much and nothing else."""
    cyc, stance, stride = _two_step_cycle(spec)
    locked, offset, _ = rt.lock_stance(cyc, stance, stride, spec)
    lift = offset[23, 0]                                       # the left foot's last stance sample
    assert abs(lift) > 0.005
    n = 16                                                     # its swing, 24-39
    for j in range(n):
        p = 24 + j
        assert offset[p, 0] == pytest.approx(lift * (n - j) / (n + 1), abs=1e-12)
        a0, _, R0 = _foot(cyc[p], spec, "left")
        a1, _, R1 = _foot(locked[p], spec, "left")
        assert a1 - a0 == pytest.approx([offset[p, 0], 0.0, 0.0], abs=1e-9)
        assert R1 == pytest.approx(R0, abs=1e-9)


def test_lock_stance_counts_a_stance_leg_that_cannot_reach(spec):
    """A straight leg under the hip cannot stay where it landed while the body goes on: the IK keeps the knee straight
    and aims the leg, so the foot keeps to its place along the floor but rises off it -- correct_stance then lowers the
    pelvis. Each stance sample where that happened is counted."""
    P = 10
    cyc = np.zeros((P, len(js.dof_names(spec))))
    stance = np.zeros((P, 2), bool)
    stance[0:6, 0] = True                                      # the left foot down 0-5, the right never
    locked, offset, short = rt.lock_stance(cyc, stance, 0.5, spec)
    assert short == 5
    per = rt.stance_advance(locked, stance, spec)
    assert per[0:5, 0] == pytest.approx(np.full(5, 0.05), abs=1e-6)
    lows = [_foot(locked[p], spec, "left")[1][:, 2].min() for p in range(6)]
    assert np.all(np.diff(lows) > 0)                           # further back, higher off the floor


def test_sole_pitch_is_the_soles_tilt_toes_down(spec):
    cols = js.columns(spec)
    ank = js.joint(spec, "ankle_left")
    row = np.zeros(len(js.dof_names(spec)))
    assert rt.sole_pitch_deg(row, spec, "left") == pytest.approx(0.0, abs=1e-12)
    row[cols[ank.name].start + js.axes(ank).index("y")] = 0.1
    geom = js.leg_geometry(spec, "left")
    sole = kin.leg_fk(geom, *rt.leg_matrices(row, spec, "left"))[1]
    toe, heel = int(np.argmax(sole[:, 0])), int(np.argmin(sole[:, 0]))
    sign = 1.0 if sole[toe, 2] < sole[heel, 2] else -1.0              # + when the toes are the lower end
    assert rt.sole_pitch_deg(row, spec, "left") == pytest.approx(sign * math.degrees(0.1))


def test_symmetrize_makes_right_the_mirrored_left_half_a_cycle_on(spec):
    rng = np.random.default_rng(0)
    cyc = rng.normal(scale=0.1, size=(100, len(js.dof_names(spec))))
    stance = np.zeros((100, 2), bool)
    stance[:60, 1] = True
    stance[50:, 0] = True
    stance[:10, 0] = True
    out, st = rt.symmetrize(cyc, stance, spec)
    cols = js.columns(spec)
    L, R = out[:, cols["hip_left"]], out[:, cols["hip_right"]]
    assert R == pytest.approx(np.roll(L, 50, axis=0) * np.array([-1, 1, -1]))
    assert (st[:, 1] == np.roll(st[:, 0], 50)).all()


def test_the_ankle_keeps_the_sole_and_holds_its_third_axis_at_zero(spec, clip):
    """The ankle's z is held at zero, so the foot keeps only its swing about the shank: the sole's normal is
    the actor's, and a twist of the foot about its own up axis changes nothing."""
    m, pos, rot, C = clip
    cols, sk = js.columns(spec), m.skeleton
    dofs = rt.frame_dofs(rot, sk, C, spec)
    for side in ("left", "right"):
        assert (dofs[:, cols[f"ankle_{side}"]][:, 2] == 0.0).all()
    twisted = rot.copy()
    ia = sk.index("LeftAnkle")
    twisted[:, ia] = rot[:, ia] @ (C.T @ kin.axis_rotation("z", 0.3) @ C)
    again = rt.frame_dofs(twisted, sk, C, spec)
    assert again[:, cols["ankle_left"]] == pytest.approx(dofs[:, cols["ankle_left"]], abs=1e-9)
    G_p, G_f = rot[:, sk.index("Hips")], twisted[:, ia]
    actor_up = (C @ np.swapaxes(G_p, -1, -2) @ G_f @ C.T)[:, :, 2]
    geom = js.leg_geometry(spec, "left")
    for f in range(0, len(rot), 37):
        _, _, R_foot = kin.leg_fk(geom, *rt.leg_matrices(again[f], spec, "left"))
        assert R_foot[:, 2] == pytest.approx(actor_up[f], abs=1e-9)


def test_stance_correction_keeps_the_ankles_third_axis_at_zero(spec):
    cols = js.columns(spec)
    row = np.zeros((1, len(js.dof_names(spec))))
    row[0, cols["hip_left"]] = [0.1, -0.2, 0.25]           # abducted and turned: the re-solved leg is not planar
    row[0, cols["knee_left"]] = math.radians(20)
    corrected, _, change = rt.correct_stance(row, np.array([[True, True]]), spec, blend=0)
    assert change > 0
    for side in ("left", "right"):
        assert corrected[0, cols[f"ankle_{side}"]][2] == 0.0


def test_lift_swing_raises_a_low_swing_foot_just_to_the_clearance(spec):
    """Stand pose with the left leg in swing for two phases: its sole is on the floor, 1 cm short of a 1 cm
    clearance. The lift bends that knee until the whole sole is at 1 cm (the foot keeps its orientation) and
    touches nothing else."""
    cols = js.columns(spec)
    cyc = np.zeros((4, len(js.dof_names(spec))))
    stance = np.array([[False, True], [False, True], [True, True], [True, True]])
    out, lifted, knee = rt.lift_swing(cyc, stance, np.zeros(4), spec, 0.01)
    geom = js.leg_geometry(spec, "left")
    for p in (0, 1):
        _, sole, _ = kin.leg_fk(geom, *rt.leg_matrices(out[p], spec, "left"))
        assert sole[:, 2] == pytest.approx([0.01] * 4, abs=1e-5)
        assert out[p, cols["knee_left"]][0] > 0.0
    assert lifted.tolist() == [[True, False], [True, False], [False, False], [False, False]]
    np.testing.assert_array_equal(out[2:], cyc[2:])
    np.testing.assert_array_equal(out[:, cols["knee_right"]], cyc[:, cols["knee_right"]])
    assert knee > 0.0


def test_swing_requirement_ramps_at_both_ends_of_each_swing():
    """The full clearance holds in mid-swing; over the first and last 15 % of a swing it rises from and falls back
    toward zero in equal steps, so the lift neither snaps a foot up as it leaves the floor nor drops it as it lands.
    A swing that runs over the end of the cycle is one swing. Where the foot is down nothing is required."""
    stance = np.ones((40, 2), bool)
    stance[5:26, 0] = False                                   # left: one swing of 21 phases -> 3-phase ramps
    stance[35:, 1] = stance[:5, 1] = False                    # right: 10 phases across the cycle's end -> 2-phase ramps
    req = rt.swing_requirement(stance, 0.01, 0.15)
    left = 0.01 * np.array([1, 2, 3] + [4] * 15 + [3, 2, 1]) / 4.0
    right = 0.01 * np.array([1, 2, 3, 3, 3, 3, 3, 3, 2, 1]) / 3.0
    assert req[5:26, 0] == pytest.approx(left)
    assert np.r_[req[35:, 1], req[:5, 1]] == pytest.approx(right)
    assert np.isnan(req[stance]).all()


def test_lift_swing_takes_a_requirement_per_phase(spec):
    """Given a requirement per phase, each swing phase is lifted to its own: a sole on the floor goes to 4 mm, then to
    1 cm, and a phase that asks for nothing is left alone."""
    cyc = np.zeros((3, len(js.dof_names(spec))))
    stance = np.array([[False, True], [False, True], [False, True]])
    need = np.array([[0.004, np.nan], [0.01, np.nan], [0.0, np.nan]])
    out, lifted, _ = rt.lift_swing(cyc, stance, np.zeros(3), spec, need)
    geom = js.leg_geometry(spec, "left")
    lows = [kin.leg_fk(geom, *rt.leg_matrices(out[p], spec, "left"))[1][:, 2].min() for p in range(3)]
    assert lows == pytest.approx([0.004, 0.01, 0.0], abs=1e-5)
    assert lifted.tolist() == [[True, False], [True, False], [False, False]]
    np.testing.assert_array_equal(out[2], cyc[2])


def test_lift_swing_carries_the_mid_swing_shortfall_into_its_neighbours(spec):
    """With a slope, the shortfall where the full clearance is asked (mid-swing) is carried into the neighbouring
    swing phases, falling by at most the slope per phase: a stand-pose sole (on the floor) in a 16-phase swing ends at
    the larger of its own ramped requirement and 1 cm less the slope per phase from mid-swing -- so the lifted sole,
    and the knee that lifts it, steps no further than that from one phase to the next."""
    stance = np.ones((20, 2), bool)
    stance[2:18, 0] = False
    need = rt.swing_requirement(stance, 0.01, 0.15)          # ramps of two phases: 1/3, 2/3 cm at each end
    cyc = np.zeros((20, len(js.dof_names(spec))))
    out, lifted, _ = rt.lift_swing(cyc, stance, np.zeros(20), spec, need, slope=0.002)
    geom = js.leg_geometry(spec, "left")
    lows = np.array([kin.leg_fk(geom, *rt.leg_matrices(out[p], spec, "left"))[1][:, 2].min() for p in range(2, 18)])
    from_mid = np.array([2, 1] + [0] * 12 + [1, 2])
    assert lows == pytest.approx(np.maximum(need[2:18, 0], 0.01 - 0.002 * from_mid), abs=1e-5)
    assert lows[0] > need[2, 0] + 1e-3                         # the ramp phase is lifted above its own requirement
    assert lifted[2:18, 0].all() and not lifted[:, 1].any()


def test_lift_swing_keeps_the_joint_ranges_and_still_clears(spec):
    """A swing foot tipped up almost to the ankle's limit: keeping its orientation while the knee lifts it would take
    the ankle 19 degrees past that limit. The lift holds the ankle at its range instead and lifts on until the sole
    clears -- every joint in range, the sole at the clearance -- and counts the held phase."""
    cols = js.columns(spec)
    ank = js.joint(spec, "ankle_left")
    i = js.axes(ank).index("y")
    lower = ank.dofs[i].lower
    cyc = np.zeros((1, len(js.dof_names(spec))))
    cyc[0, cols[ank.name].start + i] = lower + math.radians(1)
    held = {}
    out, lifted, _ = rt.lift_swing(cyc, np.array([[False, True]]), np.zeros(1), spec, 0.03, held=held)
    clamped, cut = rt.clamp(out, spec)
    assert cut == {}
    sole = kin.leg_fk(js.leg_geometry(spec, "left"), *rt.leg_matrices(out[0], spec, "left"))[1]
    assert sole[:, 2].min() == pytest.approx(0.03, abs=1e-5)
    assert out[0, cols[ank.name].start + i] == pytest.approx(lower)
    assert lifted[0, 0] and held == {"ankle_left.y": 1}


def test_lift_swing_reaches_the_clearance_on_a_turned_leg(spec):
    """The ankle cannot twist about the shank (its third axis is held at zero), so on a leg turned at the hip one
    ankle raise leaves the foot a little off its old orientation -- 0.6 mm short of 3 cm here. The lift raises again
    until the sole is at the clearance."""
    cols = js.columns(spec)
    ank = js.joint(spec, "ankle_left")
    cyc = np.zeros((1, len(js.dof_names(spec))))
    cyc[0, cols["hip_left"]] = [-0.1, -0.2, 0.25]
    cyc[0, cols["knee_left"]] = math.radians(15)
    cyc[0, cols[ank.name].start + js.axes(ank).index("x")] = -0.15
    geom = js.leg_geometry(spec, "left")
    low0 = kin.leg_fk(geom, *rt.leg_matrices(cyc[0], spec, "left"))[1][:, 2].min()
    out, lifted, _ = rt.lift_swing(cyc, np.array([[False, True]]), np.zeros(1), spec, low0 + 0.03)
    low1 = kin.leg_fk(geom, *rt.leg_matrices(out[0], spec, "left"))[1][:, 2].min()
    assert low1 == pytest.approx(low0 + 0.03, abs=1e-6) and lifted[0, 0]


def test_lift_swing_leaves_a_swing_foot_that_already_clears(spec):
    cyc = np.zeros((2, len(js.dof_names(spec))))
    stance = np.array([[False, True], [False, True]])
    # The pelvis 5 cm up puts the sole at 5 cm, clear of a 3 cm clearance; without the shift it would be 3 cm short.
    out, lifted, knee = rt.lift_swing(cyc, stance, np.full(2, 0.05), spec, 0.03)
    np.testing.assert_array_equal(out, cyc)
    assert not lifted.any() and knee == 0.0


def test_scale_hip_pitch_widens_only_the_hips_fore_aft_swing(spec):
    rng = np.random.default_rng(1)
    cyc = rng.normal(scale=0.2, size=(100, len(js.dof_names(spec))))
    out = rt.scale_hip_pitch(cyc, 1.5, spec)
    cols = js.columns(spec)
    ys = [cols[f"hip_{s}"].start + 1 for s in ("left", "right")]            # rotation vector x, y, z: y is 1
    others = [c for c in range(cyc.shape[1]) if c not in ys]
    np.testing.assert_array_equal(out[:, others], cyc[:, others])
    for c in ys:
        assert out[:, c].mean() == pytest.approx(cyc[:, c].mean())
        assert out[:, c] - out[:, c].mean() == pytest.approx(1.5 * (cyc[:, c] - cyc[:, c].mean()))


def test_clamp_does_not_count_float_noise_at_a_bound(spec):
    """Averaging samples pinned at a bound can land one ulp outside it: pulled back, but not counted as cut."""
    cols = js.columns(spec)
    upper = js.joint(spec, "knee_left").dofs[0].upper
    row = np.zeros((1, len(js.dof_names(spec))))
    row[0, cols["knee_left"]] = np.nextafter(upper, np.inf)
    out, cut = rt.clamp(row, spec)
    assert out[0, cols["knee_left"]][0] == upper
    assert cut == {}


# The pelvis rides where the legs put it: no force holds its height, so the path the joint angles imply -- the
# lowest sole of the two feet with the pelvis at the stand height -- is the path the body is pushed along. Where
# that path turns sharply (a foot taking over from the other) the stiff legs throw the body -- at 1.0 m/s both
# feet can leave the floor. The path is lowered (both legs flexed alike) wherever it bends more
# than a bound, never raised.


def _second(path):
    return np.roll(path, 1) - 2.0 * path + np.roll(path, -1)


def test_a_path_within_the_bend_is_left_as_it_is():
    phase = np.linspace(0.0, 1.0, 100, endpoint=False)
    path = 0.01 * np.cos(4.0 * np.pi * phase)                     # two bobs a cycle, second difference <= 1.58e-4
    assert np.abs(_second(path)).max() < 2e-4
    assert rt.smooth_below(path, 2e-4) == pytest.approx(path, abs=1e-9)


def test_a_sharp_corner_is_lowered_to_the_bend_and_nothing_is_raised():
    phase = np.linspace(0.0, 1.0, 100, endpoint=False)
    path = -0.02 * np.abs(np.sin(2.0 * np.pi * phase))             # a corner at every half cycle -- the foot change
    path[30] += 0.01                                               # and a spike
    out = rt.smooth_below(path, 5e-5)
    assert np.all(out <= path + 1e-12)
    assert np.all(np.abs(_second(out)) <= 5e-5 * 1.001)
    assert out[30] < path[30] - 0.005 and np.max(path - out) < 0.03


def test_the_bend_bound_holds_across_the_cycle_seam():
    path = np.zeros(100)
    path[0] = 0.01                                                 # a spike on the seam
    out = rt.smooth_below(path, 1e-4)
    assert np.all(np.abs(_second(out)) <= 1e-4 * 1.001) and np.all(out <= path + 1e-12)


def test_the_pelvis_path_of_the_stand_pose_is_the_stand_height(spec):
    rows = np.zeros((3, len(js.dof_names(spec))))
    assert rt.pelvis_path(rows, spec) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_lowering_the_pelvis_flexes_both_legs_by_the_same_height(spec):
    cols = js.columns(spec)
    rows = np.zeros((2, len(js.dof_names(spec))))
    rows[:, cols["knee_left"]] = math.radians(15)
    rows[:, cols["knee_right"]] = math.radians(5)
    before = rt.pelvis_path(rows, spec)
    lowered, limited = rt.lower_pelvis(rows, spec, np.array([0.0, 0.02]))
    assert rt.pelvis_path(lowered, spec) == pytest.approx(before - np.array([0.0, 0.02]), abs=1e-6)
    assert np.array_equal(lowered[0], rows[0])
    for side in ("left", "right"):                                 # each foot keeps its orientation, rises by 2 cm
        g = js.leg_geometry(spec, side)
        _, s0, R0 = kin.leg_fk(g, *rt.leg_matrices(rows[1], spec, side))
        _, s1, R1 = kin.leg_fk(g, *rt.leg_matrices(lowered[1], spec, side))
        assert np.allclose(R0, R1, atol=1e-9) and (s1[:, 2] - s0[:, 2]) == pytest.approx([0.02] * 4, abs=1e-9)
    assert limited == 0


def test_a_drop_past_the_ankles_range_goes_only_as_far_as_the_range_allows(spec):
    """Both legs stop at the same height, so the feet keep their places relative to each other."""
    rows = np.zeros((1, len(js.dof_names(spec))))
    before = rt.pelvis_path(rows, spec)
    lowered, limited = rt.lower_pelvis(rows, spec, np.array([0.5]))      # half a metre: past any ankle's range
    assert limited == 1
    assert rt.clamp(lowered, spec)[1] == {}
    drop = before - rt.pelvis_path(lowered, spec)
    assert 0.0 < drop[0] < 0.5
    rises = []
    for side in ("left", "right"):
        g = js.leg_geometry(spec, side)
        _, s0, R0 = kin.leg_fk(g, *rt.leg_matrices(rows[0], spec, side))
        _, s1, R1 = kin.leg_fk(g, *rt.leg_matrices(lowered[0], spec, side))
        assert np.allclose(R0, R1, atol=1e-9)
        rises.append(s1[:, 2] - s0[:, 2])
    assert rises[0] == pytest.approx(rises[1], abs=1e-9) and rises[0] == pytest.approx([drop[0]] * 4, abs=1e-6)
