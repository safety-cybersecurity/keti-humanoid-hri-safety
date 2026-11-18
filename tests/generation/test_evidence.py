"""An episode's evidence (generation/labeling/evidence.py): the smallest separation margin, the first contact, by body
region the largest force over the table A.2 limit, and when a robot camera first and last saw the person -- on
synthetic arrays and the canon (meta/rules.json, meta/scenarios.json, meta/cameras.json). Isaac-free."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from generation import cameras, scorecard, visibility
from generation.labeling import evidence
from generation.labeling.label_episodes import judgement_windows, label_episode, selected_rules

ROOT = Path(__file__).resolve().parents[2]


def _canon(name):
    return json.loads((ROOT / "meta" / name).read_text(encoding="utf-8"))


RULES = _canon("rules.json")
SCHEMA = _canon("safety_schema.json")
CAMERAS = _canon("cameras.json")
SCENARIOS = {s["scenario_id"]: s for s in _canon("scenarios.json")["scenarios"]}
S1E = SCENARIOS["S1e"]
# A scenario that does not hold touching the person to be the fail: the labeller leaves untyped force uncompared.
S1D = SCENARIOS["S1d"]
EVENT_NAMES = list(SCHEMA["safety_events"])
CONTACT_TYPES = ("quasistatic", "transient", "none")
assert sorted(CONTACT_TYPES) == sorted(SCHEMA["contact_types"])
F_QS = {code: float(body["limits"]["force_quasistatic"]["value"]) for code, body in RULES["contact_limits"].items()}
MULTIPLIER = {code: body["transient_multiplier"] for code, body in RULES["contact_limits"].items()}
FPS = 50
# The person whose height in the picture the seeing tests work out: a stature of the test's own.
STATURE_M = 1.74


def _rule_of(region):
    return f"R-TS15066-A2-{region[3:]}"


def _episode(n=100, regions=evidence.REGIONS):
    arrays = {
        "timestamp": np.arange(n) / FPS,
        "safety_event": np.zeros((n, len(EVENT_NAMES)), dtype=bool),
        "root_vel": np.zeros((n, 6)),
        "separation_violation": np.zeros(n, dtype=bool),
        "min_distance_human": np.full(n, 2.0),
        "separation_required": np.full(n, 0.5),
        "contact_force": np.zeros((n, len(regions))),
        "contact_type": np.full((n, len(regions)), "none", dtype="<U11"),
    }
    meta = {"scenario_id": "S1e", "safety_event_names": list(EVENT_NAMES),
            "contact_region_names": list(regions), "hold_release_step": -1}
    return arrays, meta


def _onset(arrays, meta, step):
    arrays["safety_event"][step, meta["safety_event_names"].index("contact_onset")] = True


def _touch(arrays, meta, step, region, force, kind):
    col = meta["contact_region_names"].index(region)
    arrays["contact_force"][step, col] = force
    arrays["contact_type"][step, col] = kind


def windows(arrays, meta, scenario=S1E):
    return judgement_windows(arrays, meta, selected_rules(RULES, scenario))[:4]


def _evidence(arrays, meta, cameras_canon=None, fps=FPS):
    return evidence.episode_evidence(arrays, meta, RULES, S1E, cameras_canon, windows=windows(arrays, meta), fps=fps)


def _limit_of(region, kind):
    """The force the region's rule compares a contact of this type against; None where any such contact fires."""
    m = MULTIPLIER[region]
    if kind == "quasistatic":
        return F_QS[region]
    if kind == "transient":
        return None if m.get("not_applicable") else F_QS[region] * float(m["value"])
    return 0.0 if m.get("not_applicable") else F_QS[region] * min(1.0, float(m["value"]))


# ---- the names ----

def test_the_regions_are_the_canons():
    assert list(evidence.REGIONS) == sorted(RULES["contact_limits"]) == sorted(SCHEMA["body_regions"])
    assert len(evidence.EVIDENCE_COLUMNS) == 17 == len(set(evidence.EVIDENCE_COLUMNS))
    assert evidence.EVIDENCE_COLUMNS == tuple(evidence.PREFIX + name for name in evidence.NAMES)


def test_the_evidence_has_its_names_in_order_as_plain_json():
    arrays, meta = _episode()
    got = _evidence(arrays, meta)
    assert list(got) == list(evidence.NAMES)
    assert all(v is None or type(v) is float for v in got.values())
    assert json.loads(json.dumps(got)) == got


# ---- force over the limit, by region ----

def random_contact_episode(seed):
    """Contacts of all three types on all twelve regions, with forces at, under and over the limit each type is held
    to, on steps the labeller judges and on steps it does not: before the first contact, held after it (the hold let
    go later than the contract's step after the first contact, or never), and the first contact step itself."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 90))
    arrays, meta = _episode(n)
    if rng.random() < 0.85:
        first = int(rng.integers(3, n - 3))
        _onset(arrays, meta, first)
        for step in rng.choice(np.arange(first + 1, n), size=int(rng.integers(0, 3)), replace=False):
            _onset(arrays, meta, int(step))
        release = (first + 1, first + int(rng.integers(2, 12)), -1, 0, int(rng.integers(0, first + 1)))[
            int(rng.integers(5))]
        touches = [first]
    else:
        release = (-1, 0, int(rng.integers(0, n)))[int(rng.integers(3))]
        touches = []
    meta["hold_release_step"] = release
    touches += [int(s) for s in rng.integers(0, n, size=int(rng.integers(4, 16)))]
    for step in touches:
        region = evidence.REGIONS[int(rng.integers(len(evidence.REGIONS)))]
        kind = CONTACT_TYPES[int(rng.integers(len(CONTACT_TYPES)))]
        limit = _limit_of(region, kind)
        scale = limit if limit else F_QS[region]
        factor = float(rng.choice([0.0, 0.5, 0.99, 1.0, 1.01, 1.5, 3.0]))
        _touch(arrays, meta, step, region, scale * factor, kind)
    return arrays, meta


@pytest.mark.parametrize("seed", range(200))
def test_a_ratio_above_one_is_the_regions_rule_firing(seed):
    arrays, meta = random_contact_episode(seed)
    label = label_episode(arrays, meta, RULES, S1E)
    got = evidence.episode_evidence(arrays, meta, RULES, S1E, None, windows=windows(arrays, meta), fps=FPS)
    for region in evidence.REGIONS:
        rule = _rule_of(region)
        assert (got[f"force_ratio.{region}"] > 1.0) == (rule in label["triggered_rules"]), (seed, region)


@pytest.mark.parametrize("seed", range(100))
def test_a_ratio_above_one_is_the_regions_rule_firing_where_a_contact_is_not_the_fail(seed):
    assert not any(r["rule_id"] == "R-HRI-CONTACT-01" for r in selected_rules(RULES, S1D))
    arrays, meta = random_contact_episode(seed)
    meta["scenario_id"] = "S1d"
    label = label_episode(arrays, meta, RULES, S1D)
    got = evidence.episode_evidence(arrays, meta, RULES, S1D, None, windows=windows(arrays, meta, S1D), fps=FPS)
    for region in evidence.REGIONS:
        rule = _rule_of(region)
        assert (got[f"force_ratio.{region}"] > 1.0) == (rule in label["triggered_rules"]), (seed, region)


def _tally(seeds=range(200)):
    """What the random episodes reach, counted by episode."""
    seen = Counter()
    for seed in seeds:
        arrays, meta = random_contact_episode(seed)
        label = label_episode(arrays, meta, RULES, S1E)
        got = _evidence(arrays, meta)
        contact, judged, after_onset, _ = windows(arrays, meta)
        ratios = [got[f"force_ratio.{r}"] for r in evidence.REGIONS]
        fired = [r for r in label["triggered_rules"] if r.startswith("R-TS15066-A2-")]
        onsets = np.flatnonzero(contact)
        release = meta["hold_release_step"]
        cases = {
            "an A2 rule fired": bool(fired),
            "no A2 rule fired": not fired,
            "BR-01 or BR-02 fired": any(r in fired for r in map(_rule_of, ("BR-01", "BR-02"))),
            "a ratio is infinite": any(v == math.inf for v in ratios),
            "a ratio is finite above 1": any(1.0 < v < math.inf for v in ratios),
            "a ratio is exactly 1": any(v == 1.0 for v in ratios),
            "a ratio is between 0 and 1": any(0.0 < v < 1.0 for v in ratios),
            "no contact": not onsets.size,
            "released the step after the first contact": bool(onsets.size) and release == onsets[0] + 1,
            "held past the first contact": bool(onsets.size) and release > onsets[0] + 1,
            "never released": release == -1,
            "free from the start": release == 0,
        }
        for col, region in enumerate(meta["contact_region_names"]):
            force, kind = arrays["contact_force"][:, col], arrays["contact_type"][:, col]
            cases[f"{region} touched with force"] = bool((force > 0).any())
            for k in CONTACT_TYPES:
                limit = _limit_of(region, k)
                over = (kind == k) & ((force >= 0) if limit is None else (force > limit))
                counted = judged & after_onset if k == "none" else judged
                for where, mask in (("a judged step", counted), ("a step not judged", ~counted)):
                    key = f"{k} over its limit on {where}"
                    cases[key] = cases.get(key, False) or bool((over & mask).any())
        seen.update(key for key, hit in cases.items() if hit)
    return seen


def test_the_random_episodes_reach_every_case():
    seen = _tally()
    expected = {"an A2 rule fired", "no A2 rule fired", "BR-01 or BR-02 fired", "a ratio is infinite",
                "a ratio is finite above 1", "a ratio is exactly 1", "a ratio is between 0 and 1", "no contact",
                "released the step after the first contact", "held past the first contact", "never released",
                "free from the start",
                *(f"{r} touched with force" for r in evidence.REGIONS),
                *(f"{k} over its limit on {w}" for k in CONTACT_TYPES for w in ("a judged step", "a step not judged"))}
    assert {key: seen[key] for key in expected if not seen[key]} == {}


def one_step_contact(region, *, kind, force, step=50):
    """One contact at ``step`` -- the hold let go the step after it, as the generator does."""
    arrays, meta = _episode()
    _onset(arrays, meta, step)
    _touch(arrays, meta, step, region, force, kind)
    meta["hold_release_step"] = step + 1
    return _evidence(arrays, meta)


def evidence_without_region(region):
    arrays, meta = _episode(regions=tuple(r for r in evidence.REGIONS if r != region))
    _onset(arrays, meta, 50)
    _touch(arrays, meta, 50, "BR-05", 10.0, "quasistatic")
    meta["hold_release_step"] = 51
    return _evidence(arrays, meta)


def test_a_contact_where_none_is_permitted_reads_infinite():
    assert MULTIPLIER["BR-01"].get("not_applicable") is True
    got = one_step_contact("BR-01", kind="transient", force=3.0)
    assert got["force_ratio.BR-01"] == math.inf and got["force_ratio_max"] == math.inf
    assert json.loads(json.dumps(got)) == got


def test_a_transient_contact_of_no_force_where_none_is_permitted_reads_infinite():
    # The rule fires on any transient contact there, whatever the force -- 0 N included -- so the ratio is inf.
    got = one_step_contact("BR-01", kind="transient", force=0.0)
    assert got["force_ratio.BR-01"] == math.inf
    arrays, meta = _episode()
    _onset(arrays, meta, 50)
    _touch(arrays, meta, 50, "BR-01", 0.0, "transient")
    meta["hold_release_step"] = 51
    assert _rule_of("BR-01") in label_episode(arrays, meta, RULES, S1E)["triggered_rules"]


def test_an_untouched_region_reads_zero_and_an_unjudged_one_null():
    got = one_step_contact("BR-05", kind="quasistatic", force=F_QS["BR-05"] / 2)     # half the canon's limit
    assert got["force_ratio.BR-05"] == pytest.approx(0.5) and got["force_ratio.BR-06"] == 0.0
    assert got["force_ratio_max"] == got["force_ratio.BR-05"]
    assert evidence_without_region("BR-12")["force_ratio.BR-12"] is None


def test_a_transient_contact_is_divided_by_the_multiplied_limit():
    m = float(MULTIPLIER["BR-05"]["value"])
    got = one_step_contact("BR-05", kind="transient", force=F_QS["BR-05"] * 1.5)
    assert got["force_ratio.BR-05"] == pytest.approx(1.5 / m)


def test_an_untyped_contact_is_divided_by_the_stricter_limit_and_reads_infinite_where_none_is_permitted():
    strict = F_QS["BR-05"] * min(1.0, float(MULTIPLIER["BR-05"]["value"]))
    assert one_step_contact("BR-05", kind="none", force=strict * 0.8)["force_ratio.BR-05"] == pytest.approx(0.8)
    assert one_step_contact("BR-02", kind="none", force=0.5)["force_ratio.BR-02"] == math.inf
    assert one_step_contact("BR-02", kind="none", force=0.0)["force_ratio.BR-02"] == 0.0


def test_an_untyped_force_before_the_first_contact_does_not_count():
    arrays, meta = _episode()
    _onset(arrays, meta, 60)
    _touch(arrays, meta, 60, "BR-07", 1.0, "transient")
    _touch(arrays, meta, 30, "BR-05", 2 * F_QS["BR-05"], "none")
    meta["hold_release_step"] = 0
    assert _evidence(arrays, meta)["force_ratio.BR-05"] == 0.0
    assert _rule_of("BR-05") not in label_episode(arrays, meta, RULES, S1E)["triggered_rules"]


def test_a_held_step_does_not_count():
    # The first contact (step 80) is light and on another region; BR-05 takes twice its limit before it (70) and while
    # the hold is still on after it (85). Neither step is judged, so BR-05 reads 0.0 -- and the labeller agrees.
    arrays, meta = _episode()
    _onset(arrays, meta, 80)
    _touch(arrays, meta, 80, "BR-07", 1.0, "transient")
    meta["hold_release_step"] = 90
    for step in (70, 85):
        _touch(arrays, meta, step, "BR-05", 2 * F_QS["BR-05"], "quasistatic")
    assert _evidence(arrays, meta)["force_ratio.BR-05"] == 0.0
    assert _rule_of("BR-05") not in label_episode(arrays, meta, RULES, S1E)["triggered_rules"]
    _touch(arrays, meta, 95, "BR-05", 2 * F_QS["BR-05"], "quasistatic")      # the same force once the hold is let go
    assert _evidence(arrays, meta)["force_ratio.BR-05"] == pytest.approx(2.0)


def test_a_scenario_that_does_not_declare_a_regions_rule_reads_null():
    arrays, meta = _episode()
    scenario = {**S1E, "rules": ["R-TS15066-A2-05", "R-TS15066-SSM-02", "R-HRI-CONTACT-01"]}
    got = evidence.episode_evidence(arrays, meta, RULES, scenario, None, windows=windows(arrays, meta), fps=FPS)
    assert got["force_ratio.BR-05"] == 0.0
    assert [r for r in evidence.REGIONS if got[f"force_ratio.{r}"] is None] == [
        r for r in evidence.REGIONS if r != "BR-05"]


def test_a_region_whose_limit_is_empty_reads_null():
    rules = json.loads(json.dumps(RULES))
    rules["contact_limits"]["BR-05"]["limits"]["force_quasistatic"]["value"] = None
    arrays, meta = _episode()
    got = evidence.episode_evidence(arrays, meta, rules, S1E, None, windows=windows(arrays, meta), fps=FPS)
    assert got["force_ratio.BR-05"] is None and got["force_ratio.BR-06"] == 0.0


def test_a_rule_pointing_at_another_regions_limit_is_refused():
    rules = json.loads(json.dumps(RULES))
    rule = next(r for r in rules["rules"] if r["rule_id"] == _rule_of("BR-05"))
    rule["limits_ref"] = "contact_limits.BR-06"
    arrays, meta = _episode()
    with pytest.raises(ValueError, match="limits_ref"):
        evidence.episode_evidence(arrays, meta, rules, S1E, None, windows=windows(arrays, meta), fps=FPS)


def test_the_largest_ratio_is_none_only_when_every_region_is():
    arrays, meta = _episode()
    scenario = {**S1E, "rules": ["R-TS15066-SSM-02", "R-HRI-CONTACT-01"]}
    got = evidence.episode_evidence(arrays, meta, RULES, scenario, None, windows=windows(arrays, meta), fps=FPS)
    assert all(got[f"force_ratio.{r}"] is None for r in evidence.REGIONS) and got["force_ratio_max"] is None


# ---- the margin and the first contact ----

def test_the_margin_is_the_smallest_distance_less_separation():
    arrays, meta = _episode(n=4)
    arrays["min_distance_human"] = np.float32([3.0, 1.0, 2.0, 4.0])
    arrays["separation_required"] = np.float32([0.2, 0.1, 0.4, 0.5])
    d64, s64 = (arrays[k].astype(np.float64) for k in ("min_distance_human", "separation_required"))
    want = float(np.min(d64 - s64))
    assert want != float(np.min(arrays["min_distance_human"] - arrays["separation_required"]))   # 32 bits differ here
    got = _evidence(arrays, meta)["min_separation_margin_m"]
    assert got == want
    facts = scorecard.episode_facts(arrays["min_distance_human"], arrays["separation_required"], np.zeros(4),
                                    np.zeros(4), [[] for _ in range(4)], 0.05)
    assert got == facts["min_margin"]


def test_the_first_contact_is_its_step_over_fps_and_null_without_one():
    arrays, meta = _episode()
    assert _evidence(arrays, meta)["first_contact_s"] is None
    for step in (37, 60):
        _onset(arrays, meta, step)
    meta["hold_release_step"] = 38
    assert _evidence(arrays, meta)["first_contact_s"] == 37 / FPS == 0.74
    assert _evidence(arrays, meta, fps=30)["first_contact_s"] == 37 / 30


@pytest.mark.parametrize("fps", [0, -50, 50.0, True, "50"])
def test_fps_is_a_positive_integer(fps):
    arrays, meta = _episode()
    with pytest.raises(ValueError, match="fps"):
        _evidence(arrays, meta, fps=fps)


# ---- when the person was seen ----

def _seen_arrays(depth_m=3.0, fraction=0.5, *, steps=slice(None), look=(1.0, 0.0, 0.0), person=None, n=100):
    """Every robot camera at the origin looking along ``look``; the first of them (head_rgb) records ``fraction`` on
    ``steps``, the others 0. The person stands ``depth_m`` ahead along world +x, or at ``person``."""
    arrays, meta = _episode(n)
    pose = cameras.pose_record(np.zeros(3), cameras.optical_frame_xyzw(look, (0.0, 0.0, 1.0)))
    for name in visibility.robot_cameras(CAMERAS):
        arrays[f"camera_pose_{name}"] = np.tile(pose, (n, 1))
        arrays[f"human_visible_fraction_{name}"] = np.zeros(n, dtype=np.float32)
    arrays["human_visible_fraction_head_rgb"][steps] = fraction
    root = (depth_m, 0.0, 0.0) if person is None else person
    arrays["human_root_pose"] = np.tile([*root, 1.0, 0.0, 0.0, 0.0], (n, 1))
    meta["human_stature_m"] = STATURE_M
    return arrays, meta


def seen_episode(*args, **kwargs):
    arrays, meta = _seen_arrays(*args, **kwargs)
    return _evidence(arrays, meta, CAMERAS)


def _height(depth_m):
    cam = CAMERAS["cameras"]["head_rgb"]
    quat = cameras.optical_frame_xyzw((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    return visibility.height_px(cam, np.zeros(3), quat, np.array([depth_m, 0.0, 0.0]), STATURE_M)


def test_seeing_needs_both_thresholds():
    rule = visibility.seeing_rule(CAMERAS)
    assert visibility.robot_cameras(CAMERAS)[0] == "head_rgb"
    assert _height(3.0) >= rule["min_height_px"] > _height(30.0)
    assert 0.5 >= rule["min_visible_fraction"] > np.float32(0.1)
    near, far = seen_episode(depth_m=3.0), seen_episode(depth_m=30.0)           # the same visible fraction, 0.5
    assert near["human_first_seen_s"] == 0.0 and far["human_first_seen_s"] is None
    assert seen_episode(depth_m=3.0, fraction=0.1)["human_last_seen_s"] is None


def test_the_camera_pose_is_read_as_wxyz():
    # A camera turned 90 deg about z from world +x looks along world +y: it sees a person ahead of it, not one beside
    # it. Read as stored (w first) its quaternion turns it elsewhere and the person ahead goes unseen.
    look = (0.0, 1.0, 0.0)
    assert seen_episode(look=look, person=(0.0, 3.0, 0.0))["human_first_seen_s"] == 0.0
    assert seen_episode(look=look, person=(3.0, 0.0, 0.0))["human_first_seen_s"] is None


def test_first_and_last_seen_are_steps_over_fps():
    got = seen_episode(steps=slice(20, 60))                                     # seen on steps 20..59 of 100
    assert (got["human_first_seen_s"], got["human_last_seen_s"]) == (0.4, 1.18) == (20 / FPS, 59 / FPS)


def test_any_robot_camera_sees():
    arrays, meta = _seen_arrays(steps=slice(0, 0))
    arrays["human_visible_fraction_rear_rgb"][10:12] = 0.5
    got = _evidence(arrays, meta, CAMERAS)
    assert (got["human_first_seen_s"], got["human_last_seen_s"]) == (10 / FPS, 11 / FPS)


@pytest.mark.parametrize("drop", ["cameras", "camera_pose_head_rgb", "human_root_pose", "human_stature_m"])
def test_what_cannot_be_told_is_null(drop):
    arrays, meta = _seen_arrays()
    full = _evidence(arrays, meta, CAMERAS)
    assert full["human_first_seen_s"] == 0.0
    canon = CAMERAS
    if drop == "cameras":
        canon = None
    elif drop == "human_stature_m":
        del meta[drop]
    else:
        del arrays[drop]
    got = _evidence(arrays, meta, canon)
    assert got["human_first_seen_s"] is None and got["human_last_seen_s"] is None
    seen_names = ("human_first_seen_s", "human_last_seen_s")
    assert {k: v for k, v in got.items() if k not in seen_names} == {
        k: v for k, v in full.items() if k not in seen_names}


@pytest.mark.parametrize("key, shape", [("human_root_pose", (99, 7)), ("human_root_pose", (100, 3)),
                                        ("camera_pose_rear_rgb", (99, 7)), ("human_visible_fraction_rear_rgb", (101,))])
def test_arrays_of_another_shape_are_refused(key, shape):
    # A camera's arrays are read step by step against the person's root: one of another length would be cut short or
    # run past the episode without a word.
    arrays, meta = _seen_arrays()
    arrays[key] = np.resize(arrays[key], shape)
    with pytest.raises(ValueError, match="스텝"):
        _evidence(arrays, meta, CAMERAS)


def test_a_stature_the_sidecar_left_null_cannot_tell_either():
    arrays, meta = _seen_arrays()
    meta["human_stature_m"] = None
    got = _evidence(arrays, meta, CAMERAS)
    assert got["human_first_seen_s"] is None and got["human_last_seen_s"] is None
