"""generation/envs/s1d/jointed_walk_metrics.py -- what the walking experiment measures and the marks it sets.
Pure; runs in CI."""
import numpy as np
import pytest

from generation.envs.s1d import jointed_walk_metrics as m

W = 700.0
STAND = 100     # standing steps before the trigger


def _rows(n=600, slip=0.01, share=0.5, hold=0.2 * W, torque=100.0):
    """Standing on both feet for STAND steps (no assist across the plane), then walking with one foot down at a time
    and the feet carrying what the share leaves; the walk ends 10 steps before the record does."""
    feet = (1.0 - share) * W
    feet_load = np.zeros((n, 2))
    feet_load[:STAND] = feet / 2
    half = STAND + (n - STAND) // 2
    feet_load[STAND:half, 0] = feet
    feet_load[half:, 1] = feet
    feet_pos = np.zeros((n, 2, 3))
    feet_pos[STAND:half, 0, 0] = np.linspace(0.0, slip, half - STAND)
    triggered = np.zeros(n, bool)
    triggered[STAND:] = True
    walk_ended = np.zeros(n, bool)
    walk_ended[-10:] = True
    hold_xy = np.where(triggered & ~walk_ended, hold, 0.0)
    torque_vec = np.zeros((n, 3))
    torque_vec[triggered & ~walk_ended, 0] = torque
    return {"hold_xy": hold_xy, "feet_force": feet_load.sum(axis=1), "feet_load": feet_load, "feet_pos": feet_pos,
            "tilt": np.full(n, 3.0), "head_z": np.full(n, 1.7), "released": np.zeros(n, bool),
            "triggered": triggered, "walk_ended": walk_ended, "robot_contact": np.zeros(n, bool),
            "foot_robot": np.zeros(n, bool), "gait_speed": np.clip((np.arange(n) - STAND) * 0.02, 0.0, 1.0),
            "hold_torque_vec": torque_vec}


def _marks(rows, share=0.5):
    return m.evaluate(m.summarize(rows, W, 0.02, share))


def test_a_good_walk_passes_every_mark():
    s = m.summarize(_rows(), W, 0.02, 0.5)
    assert s["support_share"] == 0.5
    assert s["hold_xy_walk_median_frac"] == pytest.approx(0.2)
    assert s["grf_walk_median_frac"] == pytest.approx(0.5)
    assert s["flight_frac"] == 0.0
    assert s["tilt_torque_walk_median_nm"] == pytest.approx(100.0)
    assert s["max_stance_slip_m"] == pytest.approx(0.01)
    assert all(m.evaluate(s).values()), m.evaluate(s)


def test_the_walking_marks_leave_the_standing_steps_out():
    # most of the held steps stand, with no assist across the plane and the feet at the full share: a median over
    # every held step would read the standing values
    r = _rows(n=STAND + 60)
    r["hold_xy"][:STAND] = 0.0
    s = m.summarize(r, W, 0.02, 0.5)
    assert s["hold_xy_walk_median_frac"] == pytest.approx(0.2)
    assert s["steps_walking"] == 50


def test_the_feet_are_marked_against_the_share_they_carry():
    assert _marks(_rows(share=0.5), share=0.5)["grf"] is True
    assert _marks(_rows(share=0.0), share=0.5)["grf"] is False          # the whole weight on the feet
    assert _marks(_rows(share=0.7), share=0.5)["grf"] is False          # 0.3 W where 0.5 W is theirs
    r = _rows(share=0.5)
    r["feet_force"][STAND:] = 0.59 * W
    assert _marks(r)["grf"] is True


def test_steps_with_both_feet_off_the_floor_are_counted_and_reported_not_marked():
    r = _rows()
    walking = np.flatnonzero(r["triggered"] & ~r["walk_ended"])
    off = walking[: int(0.06 * len(walking)) + 1]
    r["feet_load"][off] = 0.04 * W
    s = m.summarize(r, W, 0.02, 0.5)
    assert s["flight_frac"] == pytest.approx(len(off) / len(walking))
    e = m.evaluate(s)
    assert "flight" not in e and all(e.values())


@pytest.mark.parametrize("hold, torque, failed", [(0.31 * W, 100.0, "hold"), (0.2 * W, 151.0, "torque")])
def test_the_assist_has_an_upper_bound_across_the_plane_and_about_the_tilt_axes(hold, torque, failed):
    e = _marks(_rows(hold=hold, torque=torque))
    assert e[failed] is False and all(v for k, v in e.items() if k != failed)


def test_the_torque_mark_reads_the_tilt_axes_not_the_turn():
    r = _rows(torque=0.0)
    r["hold_torque_vec"][:, 2] = 500.0
    assert _marks(r)["torque"] is True
    r["hold_torque_vec"][:, 1] = 160.0
    assert _marks(r)["torque"] is False


def test_a_case_that_never_walks_has_no_walking_marks():
    r = _rows()
    r["triggered"][:] = False
    e = _marks(r)
    assert {k: e[k] for k in ("hold", "grf", "torque")} == dict.fromkeys(("hold", "grf", "torque"))
    assert e["tilt"] and e["upright"] and e["far_side"] and e["contamination"]


def test_a_foot_counts_as_down_from_a_fifth_of_what_the_feet_carry():
    # with 0.7 of the weight carried a foot down in double support takes about 0.15 W -- under a fifth of the whole
    # weight, over a fifth of the feet's 0.3 W; its slip has to be seen
    r = _rows(share=0.7)
    r["feet_load"][STAND:] = 0.1 * W
    r["feet_pos"][STAND:STAND + 50, 1, 0] = np.linspace(0.0, 0.05, 50)
    assert m.summarize(r, W, 0.02, 0.7)["max_stance_slip_m"] == pytest.approx(0.05, abs=1e-3)


def test_a_slip_is_measured_and_reported_not_marked():
    s = m.summarize(_rows(slip=0.05), W, 0.02, 0.5)
    assert s["max_stance_slip_m"] == pytest.approx(0.05)
    e = m.evaluate(s)
    assert "slip" not in e and all(e.values())


def test_the_marks_are_the_assist_the_feet_the_tilt_and_the_runs_own():
    assert set(m.evaluate(m.summarize(_rows(), W, 0.02, 0.5))) == {
        "hold", "grf", "torque", "tilt", "upright", "far_side", "contamination"}


def test_a_walk_the_record_ends_in_is_not_marked_on_the_far_side():
    # the record (the episode's length) ran out while the dummy still walked: whether it would have stopped at the far
    # side is not in the record -- a slow walk at a shallow angle needs longer than an episode to cross; the rest is
    # still marked
    r = _rows()
    r["walk_ended"][:] = False
    s = m.summarize(r, W, 0.02, 0.5)
    assert s["walking_at_end"] is True and s["reached_far_side"] is False
    e = m.evaluate(s)
    assert e["far_side"] is None and all(v for k, v in e.items() if k != "far_side")
    r["tilt"][300] = 12.0
    assert _marks(r)["tilt"] is False


def test_a_walk_that_ends_before_the_record_does_is_marked_on_the_far_side():
    s = m.summarize(_rows(), W, 0.02, 0.5)
    assert s["walking_at_end"] is False and m.evaluate(s)["far_side"] is True
    r = _rows()
    r["walk_ended"][:] = False
    r["triggered"][-1] = False     # the walk stopped short of the far side before the record ended
    s = m.summarize(r, W, 0.02, 0.5)
    assert s["walking_at_end"] is False and m.evaluate(s)["far_side"] is False
    r = _rows()
    r["walk_ended"][:] = False
    r["released"][-5:] = True      # released: what follows is not marked
    s = m.summarize(r, W, 0.02, 0.5)
    assert s["walking_at_end"] is False and m.evaluate(s)["far_side"] is True


def test_what_happens_after_release_is_not_marked():
    r = _rows()
    r["released"][300:] = True
    r["tilt"][300:] = 80.0
    r["head_z"][300:] = 0.3
    e = _marks(r)
    assert e["tilt"] and e["upright"] and e["far_side"]


def test_a_fall_before_release_is_marked():
    r = _rows()
    r["head_z"][200] = 0.9
    assert _marks(r)["upright"] is False


def test_two_realized_speeds_the_whole_walk_and_its_steady_part():
    s = m.summarize(_rows(), W, 0.02, 0.5)
    assert s["walk_steady_mps"] == pytest.approx(1.0)
    assert s["walk_mean_mps"] < s["walk_steady_mps"]


def test_held_contact_steps_are_counted():
    r = _rows()
    r["robot_contact"][150:153] = True
    s = m.summarize(r, W, 0.02, 0.5)
    assert s["contaminated_steps"] == 3
    assert m.evaluate(s)["contamination"] is False


# A standing step's region heights against the dimensions the dummy is built from: the check that shows a pairing of
# bodies gone wrong on the real engine (a region recorded at another body's height), whatever order the engine lists.

from pathlib import Path

import yaml

ANTHRO = yaml.safe_load((Path(__file__).resolve().parents[2] / "generation/dummy/anthropometry.yaml")
                        .read_text(encoding="utf-8"))
EXPECTED = m.standing_region_heights(ANTHRO)
CODES = sorted(EXPECTED)


def test_the_standing_heights_are_the_region_centres():
    assert CODES == sorted(ANTHRO["regions"])
    assert all(EXPECTED[c] == ANTHRO["regions"][c]["center"][2] for c in CODES)


def _standing(offset=-0.023, wobble=0.004):
    rng = np.random.default_rng(0)
    return [EXPECTED[c] + offset + rng.uniform(-wobble, wobble) for c in CODES]


def test_a_standing_dummy_has_no_problem_whatever_its_common_offset():
    assert m.standing_height_problems(CODES, _standing(), EXPECTED) == []
    assert m.standing_height_problems(CODES, _standing(offset=0.06), EXPECTED) == []


def test_two_regions_recorded_at_each_others_height_are_named():
    z = _standing()
    a, b = CODES.index("BR-01"), CODES.index("BR-04")
    z[a], z[b] = z[b], z[a]
    problems = m.standing_height_problems(CODES, z, EXPECTED)
    assert len(problems) == 2 and "BR-01" in problems[0] and "BR-04" in problems[1]


def test_the_closest_pair_of_heights_still_tells_a_swap():
    # the two region centres nearest in height; a swap of those is the smallest one the check has to see
    gaps = sorted((abs(EXPECTED[p] - EXPECTED[q]), p, q) for p in CODES for q in CODES if p < q and EXPECTED[p] != EXPECTED[q])
    gap, p, q = gaps[0]
    assert gap > m.STANDING_TOL_M
    z = _standing(wobble=0.0)
    z[CODES.index(p)], z[CODES.index(q)] = z[CODES.index(q)], z[CODES.index(p)]
    assert len(m.standing_height_problems(CODES, z, EXPECTED)) == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_height_that_is_not_a_number_is_a_problem(bad):
    z = _standing()
    z[0] = bad
    assert any(CODES[0] in p for p in m.standing_height_problems(CODES, z, EXPECTED))


def test_a_region_without_a_height_is_a_problem():
    problems = m.standing_height_problems(CODES[1:], _standing()[1:], EXPECTED)
    assert any(CODES[0] in p for p in problems)


RUNNER = (Path(__file__).resolve().parents[2] / "generation/envs/s1d/check_jointed_walk.py").read_text(encoding="utf-8")


def test_the_runner_reads_the_heights_as_the_recorder_does_and_checks_them_standing():
    # through the same sampler run_s1a installs (its heights are paired with the sensor's bodies by name), reset
    # after each env reset as run_s1a does, and checked a few steps in -- the slice measure's step
    assert "acc = substep_contact.install(base, contact_eps_n=CONTACT_EPS_N)" in RUNNER
    assert RUNNER.index("obs, _ = env.reset()") < RUNNER.index("acc.reset()")
    assert "peak = acc.drain()" in RUNNER
    assert "if step == STANDING_STEP:" in RUNNER
    assert "metrics.standing_height_problems(" in RUNNER
    assert "from generation.scene.measure_slice import STANDING_STEP" in RUNNER


def test_the_runner_does_not_say_install_refuses_a_differing_order():
    assert "refuses a dummy" not in RUNNER


def test_the_runner_records_what_tips_the_body():
    # the torso's own up and forward axes (through the library's quaternion order, so no convention is assumed here),
    # the centre of mass, the assist's force and torque as vectors, and every dummy body's floor-normal contact force
    for row in ('"up"', '"fwd"', '"com"', '"hold_force_vec"', '"hold_torque_vec"', '"body_force_z"'):
        assert row in RUNNER, row
    assert 'rows["up"].append(axis([0.0, 0.0, 1.0]))' in RUNNER and 'rows["fwd"].append(axis([1.0, 0.0, 0.0]))' in RUNNER


def test_the_runner_sweeps_the_share_over_its_cases():
    # each case run once per share, the share set on the live command term before that case's reset, and marked
    # against the share it ran with
    assert 'parser.add_argument("--support-shares", nargs="*", type=float, default=[])' in RUNNER
    loop = RUNNER[RUNNER.index("for name, seq, fixed, share, wscale, pfreq, tscale in runs:"):]
    assert loop.index("command.cfg.dummy_hold_support_share = share") < loop.index("obs, _ = env.reset()")
    assert "metrics.summarize(arrays, weight, base.step_dt, command.cfg.dummy_hold_support_share or 0.0)" in loop
    assert '"support_share": command.cfg.dummy_hold_support_share' in loop


def test_the_runner_reads_the_cases_speed_and_angle_before_its_steps():
    # the env resets itself when an episode's time is up and draws the next case's speed and angle -- read after the
    # steps, a drawn (not fixed) case reported the next draw
    loop = RUNNER[RUNNER.index("for name, seq, fixed, share, wscale, pfreq, tscale in runs:"):]
    read = loop.index('case_speed = float(entry["speed"][0])')
    assert loop.index("obs, _ = env.reset()") < read < loop.index("for step in range(args.steps):")
    assert loop.index('case_angle_deg = math.degrees(float(entry["angle"][0]))') < loop.index("for step in range(args.steps):")
    assert '"speed": case_speed,' in loop and '"angle_deg": case_angle_deg,' in loop
    assert 'float(entry["speed"][0])' not in loop[loop.index("for step in range(args.steps):"):]


def test_the_runner_sweeps_the_muscle_tone_too():
    # each case once per tone scale (of the tone gains --tone-scale left), set on the live term before the case's reset:
    # the term writes the tone gains from its config when it lets the dummy go
    assert 'parser.add_argument("--tone-scales", nargs="*", type=float, default=[])' in RUNNER
    loop = RUNNER[RUNNER.index("for name, seq, fixed, share, wscale, pfreq, tscale in runs:"):]
    set_tone = "command.cfg.dummy_tone_gains = {n: (k * tscale, d * tscale) for n, (k, d) in base_tone.items()}"
    assert loop.index(set_tone) < loop.index("obs, _ = env.reset()")
    assert '"tone_scale": args.tone_scale * (tscale if tscale is not None else 1.0)' in loop
    assert '(f"-t{tscale:g}" if args.tone_scales else "")' in RUNNER
    assert "for tscale in (args.tone_scales or [None]) for name, seq, fixed in runs]" in RUNNER
    assert "if args.tone_scales and args.rigid_dummy:" in RUNNER


def test_a_case_may_name_its_seed():
    # speed:angle runs the i-th case on the i-th seed; speed:angle:seed on that seed -- to rerun one case of an
    # earlier run on its own (the seed sets the robot's draw, and with it whether the two meet)
    assert "parts = c.split(\":\")" in RUNNER
    assert "seq = int(parts[2]) if len(parts) == 3 else i" in RUNNER
    assert 'runs.append((f"case{seq}-{parts[0]}_{parts[1]}", seq, (float(parts[0]), float(parts[1]))))' in RUNNER


def test_the_runner_prints_a_mark_not_judged_as_a_dash():
    assert "{'-' if v is None else 'O' if v else 'X'}" in RUNNER


def test_the_runner_sweeps_the_walking_gains_and_the_plane_frequency_too():
    # each case once per share x walking-gain scale x plane frequency; the scale applies to the unscaled walking gains
    # and is written to the engine after the case's reset (the term writes the walking gains on a reset only once it
    # has read its gait); the plane frequency is set on the live term like the share
    assert 'parser.add_argument("--walk-gain-scales", nargs="*", type=float, default=[])' in RUNNER
    assert 'parser.add_argument("--plane-freqs", nargs="*", type=float, default=[])' in RUNNER
    loop = RUNNER[RUNNER.index("for name, seq, fixed, share, wscale, pfreq, tscale in runs:"):]
    assert loop.index("command.cfg.dummy_hold_plane_freq_hz = pfreq") < loop.index("obs, _ = env.reset()")
    assert loop.index("obs, _ = env.reset()") < loop.index("command._write_gains(")
    assert "{n: (k * wscale, d * wscale) for n, (k, d) in base_walk.items()}" in loop
    assert '"walk_gain_scale": wscale, "plane_freq_hz": command.cfg.dummy_hold_plane_freq_hz' in loop
