"""generation/dummy/gait/build_gait.py -- a gait table from a walking archive: locked, deterministic,
covering the entry speeds, standing at speed 0. The archive is synthetic (gait_fixture). Pure; CI."""
import hashlib
import math
import io
import json
import zipfile

import numpy as np
import pytest
import yaml

from generation.dummy import joints_spec as js
from generation.dummy.gait import build_gait as bg
from generation.dummy.gait import bvh, cycles, kinematics as kin
from generation.dummy.gait.gait_table import GaitTable
from generation.dummy.gait import retarget as rt
from tests.generation.gait_fixture import synthetic_walk_bvh, synthetic_walk_segments_bvh

# Straight at 0.8 m/s for 12 s, then 0.6 m/s turning left at 30 deg/s: the turning half's cycles turn about 52 deg
# each, and the cycle across the change about 40 deg.
TURNING = [(0.8, 12.0, 0.0), (0.6, 12.0, 30.0)]


def _archive(tmp_path, walk=None):
    p = tmp_path / "100STYLE.zip"
    clips = ((walk or synthetic_walk_bvh(speed=0.8, seconds=12.0), "Neutral_FW.bvh"),
             (synthetic_walk_bvh(speed=0.01, seconds=12.0), "Neutral_ID.bvh"))
    with zipfile.ZipFile(p, "w") as z:
        for text, name in clips:
            info = zipfile.ZipInfo(f"100STYLE/Neutral/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            z.writestr(info, text)
    return p


def _built(tmp_path, walk=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = _archive(tmp_path, walk)
    return bg.build(p, js.load(), _spec(), _lock(p))


def _measured(report):
    return [b for b in report["bins"] if not b["extrapolated"]]


def _lock(p):
    return {"archive": {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "license": "CC-BY-4.0"}}


def _spec():
    return yaml.safe_load(bg.SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def default(tmp_path_factory):
    """The default archive (a straight 0.8 m/s walk) built once: (archive, table, report)."""
    p = _archive(tmp_path_factory.mktemp("default"))
    table, report = bg.build(p, js.load(), _spec(), _lock(p))
    return p, table, report


@pytest.fixture(scope="module")
def measured_row():
    """The default walk's one measured row as the build makes it -- the cycles of its speed bin averaged (one of the
    eight falls in the bin below), stance-corrected and re-timed -- before any extrapolation or swing lift."""
    spec, gs = js.load(), _spec()
    found, _, _ = bg._clip_cycles(bvh.parse_bvh(synthetic_walk_bvh(speed=0.8, seconds=12.0)), spec, gs)
    bins = {}
    for c in found:
        bins.setdefault(int(round(c["speed"] / gs["speeds"]["bin_mps"])), []).append(c)
    (group,) = [g for g in bins.values() if len(g) >= gs["speeds"]["min_cycles_per_bin"]]
    return bg._average(group, spec, gs)


def test_the_table_stands_at_zero_and_covers_the_range(default):
    _, table, report = default
    t = GaitTable.from_json(table)
    assert t.speeds[0] == 0.0
    # Built rows from 0.3 m/s, below the slowest entry speed: the start from rest passes those speeds, and under the
    # first row the table shrinks that row's joint angles -- with a first row at 0.4 m/s the shrunk walk puts a foot up
    # to 4 cm through the floor during the start on the 100STYLE table (measured 2026-10-14) and lifts the body off its
    # feet within the first second of the walk.
    assert t.speeds[1] <= 0.3 + 1e-9 and t.speeds[-1] >= 1.6 - 1e-9
    assert report["extrapolated_speeds"]                 # one synthetic speed cannot cover 0.4-1.6
    assert set(t.dof_names) == set(js.dof_names(js.load()))


def test_same_archive_same_bytes(default):
    p, a, _ = default
    b, _ = bg.build(p, js.load(), _spec(), _lock(p))
    assert bg.dumps(a) == bg.dumps(b)


def test_a_different_archive_is_refused(tmp_path):
    p = _archive(tmp_path)
    lock = _lock(p)
    lock["archive"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="잠금"):
        bg.build(p, js.load(), _spec(), lock)


def test_an_empty_lock_hash_is_refused(tmp_path):
    p = _archive(tmp_path)
    lock = _lock(p)
    lock["archive"]["sha256"] = ""
    with pytest.raises(ValueError, match="잠금"):
        bg.build(p, js.load(), _spec(), lock)


def test_the_report_says_what_it_measured(default):
    _, _, report = default
    for key in ("cycles", "turning", "bins", "clipped", "max_stance_correction_deg", "min_swing_clearance_m",
                "min_swing_clearance_raw_m", "walk_ratio_m_per_steps_per_min", "walk_ratio_actor_m_per_steps_per_min",
                "walk_ratio_literature", "stand_pose_deg"):
        assert key in report
    rows = report["bins"]
    assert report["min_swing_clearance_m"] == min(b["swing_clearance_m"] for b in rows)
    assert report["min_swing_clearance_raw_m"] == min(b["swing_clearance_raw_m"] for b in rows)
    # The synthetic walk's own swing foot passes under the floor late in each swing (a straight stance leg
    # cannot clear it), so before the swing lift the measured rows dip as the actor did, scaled to the dummy's
    # legs -- no more (the pipeline sank a foot) and no less (the pelvis shift was lost or turned around).
    want = _actor_lowest_foot(0.8) * _leg_ratio()
    assert _measured(report) and min(b["swing_clearance_raw_m"] for b in _measured(report)) == pytest.approx(want, abs=0.005)


def _actor_lowest_foot(speed):
    """The walk's lowest ankle or toe point over the clip -- its floor is y = 0, where the stance sole sits."""
    m = bvh.parse_bvh(synthetic_walk_bvh(speed=speed, seconds=12.0))
    pos, _ = bvh.forward_kinematics(m, scale=cycles.unit_scale(m.skeleton))
    feet = [m.skeleton.index(n) for n in ("LeftAnkle", "LeftToe", "RightAnkle", "RightToe")]
    return float(pos[:, feet, 1].min())


def _leg_ratio():
    """The dummy's hip-to-ankle length over the walk's."""
    g = js.leg_geometry(js.load(), "left")
    m = bvh.parse_bvh(synthetic_walk_bvh(seconds=0.1))
    sk, s = m.skeleton, cycles.unit_scale(m.skeleton)
    actor = s * sum(float(np.linalg.norm(sk.offsets[sk.index(n)])) for n in ("LeftKnee", "LeftAnkle"))
    return float(np.linalg.norm(g.knee - g.hip) + np.linalg.norm(g.ankle - g.knee)) / actor


def test_the_ankles_third_axis_is_zero_in_every_row(default):
    _, table, _ = default
    t = GaitTable.from_json(table)
    for side in ("left", "right"):
        z = t.dof_names.index(f"ankle_{side}:2")
        assert (t.targets[:, :, z] == 0.0).all()


def test_each_cycle_is_made_periodic_before_averaging():
    """A real cycle does not end where it began; averaged as they are, the cycles' mean jumps where it wraps round (and
    the symmetrization copies the left leg's jump to the right leg half a cycle on). Each cycle's end-minus-start is
    taken out in proportion to phase before averaging, so here -- the left hip turning sideways through the clip,
    about 4 degrees a cycle -- the row's sideways hip steps no further across the wrap than beside it."""
    spec, gs = js.load(), _spec()
    found, _, _ = bg._clip_cycles(bvh.parse_bvh(synthetic_walk_bvh(speed=0.8, seconds=12.0, drift_deg_per_s=3.0)), spec, gs)
    row = bg._average(found, spec, gs)
    x = js.columns(spec)["hip_left"].start                          # the rotation vector's x: about the front axis
    d = np.degrees(row["dofs"][:, x])
    steps = np.abs(np.diff(np.r_[d, d[:1]]))
    assert steps[-1] <= np.max(steps[[-3, -2, 0, 1]]) + 0.05
    drift = 3.0 * row["duration"]
    assert drift > 3.0


def test_cycles_whose_pelvis_turns_are_left_out(default, tmp_path):
    """The turning half's cycles leave the speed rows and the walk ratio: the table is the straight half's alone."""
    _, straight, s_rep = default
    turning, t_rep = _built(tmp_path, synthetic_walk_segments_bvh(TURNING))
    assert (t_rep["turning"]["kept"], t_rep["turning"]["dropped"]) == (8, 7)
    assert (s_rep["turning"]["kept"], s_rep["turning"]["dropped"]) == (8, 0)
    a, b = GaitTable.from_json(straight), GaitTable.from_json(turning)
    np.testing.assert_array_equal(a.speeds, b.speeds)
    np.testing.assert_array_equal(a.stride, b.stride)
    np.testing.assert_array_equal(a.targets, b.targets)
    assert t_rep["walk_ratio_m_per_steps_per_min"] == s_rep["walk_ratio_m_per_steps_per_min"]


def test_a_slower_straight_half_keeps_its_own_row(tmp_path):
    """The same two halves without the turn: the slower half is a row of its own -- the turn is what took it out."""
    _, rep = _built(tmp_path, synthetic_walk_segments_bvh([(0.8, 12.0, 0.0), (0.6, 12.0, 0.0)]))
    assert rep["turning"]["dropped"] == 0
    assert len(_measured(rep)) == 2


def test_extrapolated_rows_fill_the_grid_at_the_straight_rows_walk_ratio(default):
    _, _, rep = default
    w = rep["walk_ratio_m_per_steps_per_min"]
    assert w == pytest.approx(np.mean([b["walk_ratio"] for b in _measured(rep)]))
    for b in rep["bins"]:
        assert b["walk_ratio"] == pytest.approx(b["stride_m"] * b["cycle_s"] / 240.0)    # step / (steps per min)
    extra = [b for b in rep["bins"] if b["extrapolated"]]
    # One measured row at about 0.76 m/s on the dummy's legs (its feet stay level, so rolling them adds little):
    # the grid's 0.3-0.7 below it and 0.8-1.6 above.
    assert [b["speed_mps"] for b in extra] == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    for b in extra:
        assert b["walk_ratio"] == pytest.approx(w, rel=1e-3)


def test_extrapolated_rows_keep_their_source_rows_arms(measured_row):
    """Nothing is scaled to lengthen or shorten a stride but the hips (the arms would show a scaling of every
    joint): before an extrapolated row is re-timed to its own body advance its arms are its source row's, and its hips
    swing wider only above that row."""
    spec, gs, row = js.load(), _spec(), measured_row
    arms = [i for i, n in enumerate(js.dof_names(spec)) if n.startswith(("shoulder", "elbow"))]
    for v in (0.4, 1.6):
        ext = bg._extrapolate(row, v, row["stride"] * row["duration"] / 240.0, spec, gs)
        scaled = bg._hip_scaled(row, ext["hip_scale"], spec, gs)[0]
        # (a re-timed value at a bound can sit an ulp outside it; the scaled row's clamp pulls it back)
        np.testing.assert_allclose(scaled[:, arms], row["dofs"][:, arms], rtol=0, atol=1e-12)
        assert (ext["hip_scale"] > 1.0) == (v > row["speed"])


def test_extrapolating_scales_only_the_hips_fore_aft_swing(measured_row):
    """Where one leg alone is down, further from double support than the correction is eased, the stance correction
    has nothing to re-solve, so an extrapolated row there is exactly the measured row with its hips' fore-aft swing
    scaled (and nothing else) -- before it is re-timed to its own body advance and its stance feet held where they
    landed (_locked), which is all that happens to it then (and before the swing lift)."""
    spec, gs, row = js.load(), _spec(), measured_row
    K = gs["stance"]["blend_samples"]
    for v in (0.4, 1.6):
        ext = bg._extrapolate(row, v, row["stride"] * row["duration"] / 240.0, spec, gs)
        scaled, stride, shift, _, _ = bg._hip_scaled(row, ext["hip_scale"], spec, gs)
        both = row["stance"].all(axis=1)
        near = np.zeros_like(both)
        for j in range(-K, K + 1):
            near |= np.roll(both, j)
        single = (row["stance"].sum(axis=1) == 1) & ~near
        assert single.sum() > 20
        want, _ = rt.clamp(rt.scale_hip_pitch(row["dofs"], ext["hip_scale"], spec), spec)
        np.testing.assert_array_equal(scaled[single], want[single])
        adv, _, _ = rt.body_advance(rt.stance_advance(scaled, row["stance"], spec), gs["stance"]["advance_floor_share"])
        again, stance, shift_again = rt.retime(scaled, row["stance"], adv, shift)
        held_again, shift_again, lock, held = bg._locked(again, stance, stride, spec, gs)
        np.testing.assert_array_equal(ext["dofs"], held_again)
        np.testing.assert_array_equal(ext["stance"], stance)
        np.testing.assert_array_equal(ext["shift"], shift_again)
        assert ext["lock"] == lock and all(ext["clipped"].get(k, 0) >= n for k, n in held.items())
        assert ext["stride"] == stride == float(adv.sum())
        assert (ext["hip_scale"] > 1.0) == (v > row["speed"])


def test_extending_bends_a_swing_knee_only_where_its_foot_needs_the_clearance(measured_row):
    spec, gs, row = js.load(), _spec(), measured_row
    ext = bg._extrapolate(row, 1.6, row["stride"] * row["duration"] / 240.0, spec, gs)
    fast = bg._finish(ext, spec, gs)
    cols = js.columns(spec)
    for k, side in enumerate(rt.SIDES):
        knee = cols[f"knee_{side}"].start
        swing, lifted = ~fast["stance"][:, k], fast["lifted"][:, k]
        assert lifted.any()
        np.testing.assert_array_equal(fast["dofs"][swing & ~lifted, knee], ext["dofs"][swing & ~lifted, knee])
        assert (fast["dofs"][lifted, knee] > ext["dofs"][lifted, knee]).all()


def test_the_rows_ease_the_stance_correction_over_the_specs_phases(measured_row, monkeypatch):
    """A measured row (_average) and an extrapolated one (_hip_scaled) ask rt.correct_stance to carry each double
    support's correction into the spec's number of phases on each side (the easing itself is
    test_stance_correction_eases_in_and_out_of_double_support)."""
    spec, gs = js.load(), _spec()
    assert gs["stance"]["blend_samples"] == 5
    seen = []
    real = rt.correct_stance

    def spy(*args, **kw):
        seen.append(kw.get("blend"))
        return real(*args, **kw)

    monkeypatch.setattr(rt, "correct_stance", spy)
    found, _, _ = bg._clip_cycles(bvh.parse_bvh(synthetic_walk_bvh(speed=0.8, seconds=4.0)), spec, gs)
    assert found and seen and set(seen) == {5}                     # each cycle's correction (it feeds the cycle's speed)
    seen.clear()
    bg._average(found, spec, gs)
    assert seen == [5] * (1 + bg.LOCK_ROUNDS)                     # before the re-timing, and after each lock
    seen.clear()
    bg._hip_scaled(measured_row, 1.3, spec, gs)
    assert seen == [5]


def test_the_swing_lift_uses_the_specs_slope():
    """_finish lifts with the spec's slope: a stand-pose sole in a 16-phase swing is lifted at the swing's first phase
    to 1 cm less two slopes (from mid-swing), above that phase's own ramped requirement."""
    spec, gs = js.load(), _spec()
    stance = np.ones((20, 2), bool)
    stance[2:18, 0] = False
    row = {"dofs": np.zeros((20, len(js.dof_names(spec)))), "stance": stance, "shift": np.zeros(20)}
    out = bg._finish(row, spec, gs)
    full, slope = gs["swing"]["min_clearance_m"], gs["swing"]["lift_slope_m"]
    assert slope == pytest.approx(0.002)
    need = rt.swing_requirement(stance, full, gs["swing"]["clearance_ramp"])
    low = kin.leg_fk(js.leg_geometry(spec, "left"), *rt.leg_matrices(out["dofs"][2], spec, "left"))[1][:, 2].min()
    k = int(np.flatnonzero(np.isclose(need[2:18, 0], full))[0])
    assert low == pytest.approx(max(need[2, 0], full - slope * k), abs=1e-5)
    assert low > need[2, 0] + 1e-4


def test_the_rows_clipped_counts_include_the_phases_the_lift_held_at_a_range():
    """A swing foot tipped up almost to the ankle's limit is lifted with the ankle held at its range; the row's clipped
    counts say so, one per held phase."""
    spec, gs = js.load(), _spec()
    cols = js.columns(spec)
    ank = js.joint(spec, "ankle_left")
    i = js.axes(ank).index("y")
    stance = np.ones((20, 2), bool)
    stance[2:18, 0] = False
    dofs = np.zeros((20, len(js.dof_names(spec))))
    dofs[:, cols[ank.name].start + i] = ank.dofs[i].lower + np.radians(1)
    out = bg._finish({"dofs": dofs, "stance": stance, "shift": np.zeros(20), "clipped": {}}, spec, gs)
    held = int(np.count_nonzero(np.isclose(out["dofs"][2:18, cols[ank.name].start + i], ank.dofs[i].lower)))
    assert held > 0 and out["clipped"] == {"ankle_left.y": held}


def test_every_rows_swing_sole_clears_the_floor(default):
    """Every swing phase of every row meets its requirement: the full clearance in mid-swing, ramped at the swing's
    ends (rt.swing_requirement). The project's clearance is 2.5 cm (the walking dummy's swing toe passes about
    1 cm lower in the simulation than the table puts it, so at 1 cm it catches the floor early in the swing)."""
    _, _, rep = default
    swing = _spec()["swing"]
    assert swing["min_clearance_m"] == pytest.approx(0.025)
    assert rep["swing_requirement"] == {"min_clearance_m": swing["min_clearance_m"], "ramp": swing["clearance_ramp"],
                                        "lift_slope_m": swing["lift_slope_m"]}
    assert len(rep["bins"]) == 15
    for b in rep["bins"]:
        assert b["swing_margin_m"] >= -1e-5
        assert b["swing_clearance_mid_m"] >= swing["min_clearance_m"] - 1e-5
    assert any(b["swing_clearance_raw_m"] < swing["min_clearance_m"] - 1e-3 for b in _measured(rep))  # the lift had work


def test_each_row_reports_its_largest_leg_joint_step(default):
    """The report gives, per row, the largest change of any leg joint between neighbouring phases (the cycle wraps
    round) as the table holds it: degrees, which degree of freedom, and the phase it starts from."""
    _, table, rep = default
    t = GaitTable.from_json(table)
    names, speeds = list(t.dof_names), [float(v) for v in t.speeds]
    legs = [i for i, n in enumerate(names) if n.startswith(("hip_", "knee_", "ankle_"))]
    for b in rep["bins"]:
        d = np.degrees(t.targets[speeds.index(b["speed_mps"])][:, legs].astype(float))
        steps = np.abs(np.diff(np.vstack([d, d[:1]]), axis=0))
        p, i = np.unravel_index(int(np.argmax(steps)), steps.shape)
        got = b["max_leg_step"]
        assert got["deg"] == pytest.approx(float(steps[p, i]), abs=1e-4)
        assert (got["dof"], got["phase"]) == (names[legs[i]], int(p))


def test_rows_are_re_timed_so_each_phase_carries_the_body_the_same_distance(measured_row):
    """Each row is re-timed (rt.retime) to its own body advance, so at a steady pace with the phase advanced by distance
    its stance feet roll without slipping: the measured row's advance per phase, uneven by a fifth before, is even to
    within 6 % after (the most next to double support, where the stance mask is resampled to the nearer phase), and so
    is an extrapolated row's; the stride is the advance summed."""
    spec, gs, row = js.load(), _spec(), measured_row
    share = gs["stance"]["advance_floor_share"]
    found, _, _ = bg._clip_cycles(bvh.parse_bvh(synthetic_walk_bvh(speed=0.8, seconds=12.0)), spec, gs)
    cyc, stance = rt.symmetrize(np.mean([g["dofs"] for g in found], axis=0),
                                np.mean([g["stance"].astype(float) for g in found], axis=0) >= 0.5, spec)
    before, _, _ = rt.body_advance(rt.stance_advance(rt.correct_stance(cyc, stance, spec, blend=5)[0], stance, spec), share)
    assert np.max(np.abs(before / before.mean() - 1.0)) > 0.15
    fast = bg._extrapolate(row, 1.6, row["stride"] * row["duration"] / 240.0, spec, gs)
    for r in (row, fast):
        adv, floored, gaps = rt.body_advance(rt.stance_advance(r["dofs"], r["stance"], spec), share)
        assert (floored, gaps) == (0, 0) and r["retime"] == {"floored_steps": 0, "no_foot_steps": 0}
        assert np.max(np.abs(adv / (r["stride"] / len(adv)) - 1.0)) < 0.06
        assert float(adv.sum()) == pytest.approx(r["stride"], rel=2e-3)
    assert row["speed"] == pytest.approx(row["stride"] / row["duration"])


def _stance_sole_heights(row, spec):
    """Every stance sole's lowest corner above the floor, with the row's pelvis shift."""
    return [kin.leg_fk(js.leg_geometry(spec, s), *rt.leg_matrices(row["dofs"][p], spec, s))[1][:, 2].min() + row["shift"][p]
            for p in range(len(row["stance"])) for k, s in enumerate(rt.SIDES) if row["stance"][p, k]]


def test_rows_hold_each_stance_foot_where_it_landed(default, measured_row):
    """Every row holds its stance feet where they landed (rt.lock_stance after the re-timing, then the double support
    flexed again): at a steady pace with the phase advanced by distance nothing slides and the two feet never
    disagree, in every row the report gives; and every stance sole is on the floor with the row's pelvis shift -- the
    measured row's and an extrapolated one's."""
    _, _, rep = default
    for b in rep["bins"]:
        assert b["stance_slip"]["to_and_fro_m"] < 1e-4, b["speed_mps"]
        assert b["stance_slip"]["double_support_disagreement_m"] < 1e-4, b["speed_mps"]
    spec, gs, row = js.load(), _spec(), measured_row
    fast = bg._extrapolate(row, 1.6, row["stride"] * row["duration"] / 240.0, spec, gs)
    for r in (row, fast):
        assert max(abs(h) for h in _stance_sole_heights(r, spec)) < 2e-5


def test_each_row_reports_what_the_lock_did(default, measured_row):
    """A row's lock (_locked) is applied to the re-timed row at its own stride, the double support flexed again after
    it (the pelvis shift with it), and the joints the last round held at a range are the row's clipped counts. The
    report gives, per row, what the lock did -- the largest distance an ankle was moved along x, that distance where a
    foot lifts (taken back over its swing), the stance samples where a leg could not reach, and the largest knee
    change -- here the measured row's, rebuilt beside the build."""
    _, _, rep = default
    spec, gs = js.load(), _spec()
    found, _, _ = bg._clip_cycles(bvh.parse_bvh(synthetic_walk_bvh(speed=0.8, seconds=12.0)), spec, gs)
    bins = {}
    for c in found:
        bins.setdefault(int(round(c["speed"] / gs["speeds"]["bin_mps"])), []).append(c)
    (group,) = [g for g in bins.values() if len(g) >= gs["speeds"]["min_cycles_per_bin"]]
    cyc, stance = rt.symmetrize(np.mean([g["dofs"] for g in group], axis=0),
                                np.mean([g["stance"].astype(float) for g in group], axis=0) >= 0.5, spec)
    cyc, shift, _ = rt.correct_stance(cyc, stance, spec, blend=5)
    cyc, stance, shift, stride, _ = bg._retimed(cyc, stance, shift, spec, gs)
    locked, shift, lock, held = bg._locked(cyc, stance, stride, spec, gs)
    np.testing.assert_array_equal(measured_row["dofs"], locked)
    np.testing.assert_array_equal(measured_row["shift"], shift)
    assert measured_row["clipped"] == held and measured_row["lock"] == lock
    moved = np.array([[kin.leg_fk(js.leg_geometry(spec, s), *rt.leg_matrices(locked[p], spec, s))[0][0]
                       - kin.leg_fk(js.leg_geometry(spec, s), *rt.leg_matrices(cyc[p], spec, s))[0][0]
                       for s in rt.SIDES] for p in range(len(cyc))])
    cols = js.columns(spec)
    knees = [cols[f"knee_{s}"].start for s in rt.SIDES]
    lift = max(abs(moved[b % len(moved), k]) for k in range(2) for _, b in rt._runs(stance[:, k]))
    assert lock == pytest.approx({"largest_offset_m": float(np.abs(moved).max()), "lift_off_offset_m": float(lift),
                                  "short_samples": lock["short_samples"],
                                  "knee_change_deg": math.degrees(float(np.abs(locked[:, knees] - cyc[:, knees]).max()))})
    assert isinstance(lock["short_samples"], int) and lock["short_samples"] >= 0
    assert lock["largest_offset_m"] > 1e-3                          # the synthetic walk's feet did slide before
    assert _measured(rep)[0]["stance_lock"] == pytest.approx(lock)
    assert all(set(b["stance_lock"]) == set(lock) for b in rep["bins"])


def test_landing_knee_is_the_knee_of_the_foot_that_has_just_landed():
    """The knee of a foot on the phase it lands (its stance starts while the other foot is down) and the largest it
    bends through that double support -- not later in its stance -- the larger over both legs, in degrees."""
    spec = js.load()
    cols = js.columns(spec)
    P = 20
    stance = np.zeros((P, 2), bool)
    stance[0:12, 1] = True                                       # right lands at 0, lifts after 11
    stance[10:20, 0] = stance[0:2, 0] = True                     # left lands at 10, lifts after 1
    dofs = np.zeros((P, len(js.dof_names(spec))))
    dofs[[0, 1, 5], cols["knee_right"].start] = [0.1, 0.3, 0.5]          # 5: single support, after the landing
    dofs[[10, 11, 15], cols["knee_left"].start] = [0.2, 0.25, 0.6]
    got = bg._landing_knee({"dofs": dofs, "stance": stance}, spec)
    assert got == pytest.approx({"at_landing": math.degrees(0.2), "max_in_double_support": math.degrees(0.3)})


def test_stance_sole_pitch_is_taken_over_single_support():
    """The stance sole's tilt (rt.sole_pitch_deg) over the phases where one foot alone is down: median, min and max."""
    spec = js.load()
    cols = js.columns(spec)
    P = 6
    stance = np.array([[1, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 1]], bool)
    dofs = np.zeros((P, len(js.dof_names(spec))))
    for p, side, a in ((1, "left", 0.05), (2, "left", 0.10), (3, "right", -0.02), (4, "right", 0.2), (0, "left", 0.9)):
        ank = js.joint(spec, f"ankle_{side}")
        dofs[p, cols[ank.name].start + js.axes(ank).index("y")] = a
    want = [rt.sole_pitch_deg(dofs[p], spec, s) for p, s in ((1, "left"), (2, "left"), (3, "right"), (4, "right"))]
    got = bg._stance_sole_pitch({"dofs": dofs, "stance": stance}, spec)
    assert got == pytest.approx({"median": float(np.median(want)), "min": min(want), "max": max(want)})


def test_each_row_reports_its_slip_sole_and_landing_knee(default, measured_row):
    """The report gives, per row, the stance feet's slide at a steady pace (rt.stance_slip), the stance sole's tilt over
    single support, the landing knee, and what the re-timing raised to its floor or found with no foot down -- checked
    on the measured row rebuilt beside the build; and once, the stand clip's sole tilt and the joints that making each
    cycle periodic put back in range."""
    p, _, rep = default
    spec, gs = js.load(), _spec()
    b = _measured(rep)[0]
    row = bg._smoothed(bg._finish(measured_row, spec, gs), spec, gs)
    assert b["stance_slip"] == pytest.approx(rt.stance_slip(row["dofs"], row["stance"], row["stride"], spec))
    assert b["stance_sole_pitch_deg"] == pytest.approx(bg._stance_sole_pitch(row, spec))
    assert b["landing_knee_deg"] == pytest.approx(bg._landing_knee(row, spec))
    assert b["retime"] == measured_row["retime"]
    with zipfile.ZipFile(p) as z:
        stand = bvh.parse_bvh(z.read("100STYLE/Neutral/Neutral_ID.bvh").decode())
        walk = bvh.parse_bvh(z.read("100STYLE/Neutral/Neutral_FW.bvh").decode())
    stand_dofs = bg._clip_cycles(stand, spec, gs)[2]
    assert rep["stand_pose_sole_pitch_deg"] == pytest.approx(
        {s: float(np.median([rt.sole_pitch_deg(d, spec, s) for d in stand_dofs])) for s in rt.SIDES})
    kept = [c for c in bg._clip_cycles(walk, spec, gs)[0] if abs(c["turn_deg"]) <= gs["cycles"]["max_turn_deg"]]
    seam = {}
    for c in kept:
        for k, n in c["seam_clipped"].items():
            seam[k] = seam.get(k, 0) + n
    assert rep["seam_clipped"] == seam


def test_each_row_reports_its_hip_pitch_range_and_stance_correction(default):
    """The report shows, per row, the hips' fore-aft range as the table holds it and the row's stance correction."""
    _, table, rep = default
    t = GaitTable.from_json(table)
    speeds, y = [float(v) for v in t.speeds], list(t.dof_names).index("hip_left:1")
    for b in rep["bins"]:
        col = np.degrees(t.targets[speeds.index(b["speed_mps"])][:, y].astype(float))
        assert b["hip_pitch_deg"] == pytest.approx([col.min(), col.max()], abs=1e-4)
    assert rep["max_stance_correction_deg"] == pytest.approx(max(b["stance_correction_deg"] for b in rep["bins"]))


def test_every_rows_pelvis_path_bends_no_more_than_the_spec_allows(default):
    """The pelvis path the joint angles imply (rt.pelvis_path) turns no faster than the spec's vertical acceleration at
    the row's own pace (one sample of a cycle is cycle_s / phase_samples apart) wherever the joint ranges let the legs
    flex that far -- and never faster than before the lowering."""
    _, table, rep = default
    gs = _spec()
    accel = float(gs["pelvis"]["max_vertical_accel_mps2"])
    t = GaitTable.from_json(table)
    spec = js.load()
    for k, b in enumerate(rep["bins"], 1):
        dt = b["cycle_s"] / gs["cycles"]["phase_samples"]
        path = rt.pelvis_path(t.targets[k].astype(float), spec)
        bend = np.roll(path, 1) - 2.0 * path + np.roll(path, -1)
        got = np.max(np.abs(bend)) / dt ** 2
        assert got == pytest.approx(b["pelvis_accel_max_mps2"], rel=1e-3, abs=1e-3)      # float32 in the table
        assert b["pelvis_accel_max_mps2"] <= b["pelvis_accel_raw_mps2"] * 1.001 and b["pelvis_lowered_max_m"] >= 0.0
        if b["pelvis_limited_samples"] == 0:
            assert got <= accel * 1.05, (b["speed_mps"], got)
    # The synthetic walk's legs reach their ankle ranges before its paths are smooth (the 100STYLE table's rows up to
    # 1.3 m/s reach the bound -- 2026-10-14); here the lowering still has to bring some row's bend down.
    assert any(b["pelvis_accel_max_mps2"] < 0.9 * b["pelvis_accel_raw_mps2"] for b in rep["bins"])
