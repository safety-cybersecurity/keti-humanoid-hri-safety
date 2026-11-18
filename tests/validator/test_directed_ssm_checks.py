"""The validator's checks for the hri scenarios. Most run on a notes object, a rules document or a
copy of the canon; the tests that name a small fixture release (F.build) are the ones that read C5
or C8 off a release's own files."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from generation.directed_separation import scenario_declares_directed_ssm
from validator import make_fixture as F
from validator import validate_dataset as v

REPO_ROOT = Path(__file__).resolve().parents[2]


def _said(notes, steps=400, **kw):
    return v.notes_problems(notes, v.BODY_REGIONS, steps, **kw)


def test_a_completed_step_inside_the_episode_or_a_mark_passes():
    for value in (-1, 0, 399, None):
        assert not [p for p in _said({"task_completed_step": value}) if "task_completed_step" in p], value


def test_a_completed_step_outside_the_episode_is_named():
    assert any("task_completed_step" in p for p in _said({"task_completed_step": 400}))
    assert any("task_completed_step" in p for p in _said({"task_completed_step": -2}))


def _completion_said(notes):
    return [p for p in _said(notes, completion_required=True) if "task_completed_step" in p]


def test_an_hri_scenario_has_a_completion_step_so_null_is_named():
    """A scenario that declares the directed rule has a completion criterion: null ("no criterion")
    would leave the episode out of the scorecard's completion rate. A missing member is C5's own
    member check; here the helper names it too, so the unit does not depend on that loop."""
    assert _completion_said({"task_completed_step": None})
    assert _completion_said({})
    for value in (-1, 0, 399):
        assert not _completion_said({"task_completed_step": value}), value
    # the range check is the one the plain scenarios have
    assert _completion_said({"task_completed_step": 400})
    assert _completion_said({"task_completed_step": -2})


def _c5_completion_messages(root):
    rep = v.run(root, {"C5"}, None, 3, None)
    return [f.message for f in rep.findings
            if f.check == "C5" and f.severity == v.ERROR and "task_completed_step" in f.message]


def test_a_null_completion_step_is_an_error_for_an_hri_episode_and_not_for_a_plain_one(tmp_path):
    """The switch is wired at the C5 call site: an episode of a scenario that declares the directed
    rule is held to an integer, the others keep integer-or-null."""
    hri, plain = tmp_path / "hri", tmp_path / "plain"
    F.build(hri, n_ep=5, hri=True)
    F.build(plain, n_ep=5)
    assert not _c5_completion_messages(hri) and not _c5_completion_messages(plain)
    for root in (hri, plain):
        F._edit_episode_notes(root, 4, lambda notes: notes.update(task_completed_step=None))
    assert any("null 이거나 없음" in m for m in _c5_completion_messages(hri))
    assert not _c5_completion_messages(plain)


def test_the_episode_bounds_are_its_rows_not_its_duration(tmp_path):
    """notes.task_completed_step and notes.hold_release_step are steps of the episode, so they are
    held to its length -- the rows C7 ties to its pointers. duration_s x fps is tied to nothing: a
    release that claims 20 s for a 400-row episode would otherwise take a completion at step 1000."""
    root = tmp_path / "hri"
    F.build(root, n_ep=5, hri=True)
    eps = pd.read_parquet(F._eps_path(root))
    length = int(eps.loc[4, "length"])
    eps.loc[4, "duration_s"] = 5 * length / float(eps.loc[4, "fps"])
    eps.to_parquet(F._eps_path(root), index=False)
    F._edit_episode_notes(root, 4, lambda notes: notes.update(task_completed_step=length,
                                                              hold_release_step=length + 1))
    rep = v.run(root, {"C5"}, None, 3, None)
    said = [f.message for f in rep.findings if f.check == "C5" and f.severity == v.ERROR]
    assert any(f"task_completed_step 이 에피소드 길이({length}) 안의 스텝이 아님" in m for m in said), said
    assert any(f"hold_release_step 이 에피소드 길이({length})를 넘음" in m for m in said), said


_SIDECAR_ONLY_NOTES = ("degenerate_reason", "box_released_step", "camera_in_box_steps")


def _c5_member_messages(root, member):
    rep = v.run(root, {"C5"}, None, 3, None)
    return [f.message for f in rep.findings
            if f.check == "C5" and f.severity == v.ERROR and f"notes.{member}" in f.message]


def test_c5_reads_the_new_members_by_type(tmp_path):
    """degenerate_reason, box_released_step and camera_in_box_steps are declared in the canon as
    members of every episode's notes, each with the type the recorder writes: C5 asks for the member
    and for the type -- a string typed as a number is named, a missing member is named."""
    root = tmp_path / "hri"
    F.build(root, n_ep=5, hri=True)
    for member in _SIDECAR_ONLY_NOTES:
        assert not _c5_member_messages(root, member), member
    F._edit_episode_notes(root, 4, lambda notes: notes.update(box_released_step="312"))
    assert any("notes.box_released_step 이" in m and "integer|null" in m
               for m in _c5_member_messages(root, "box_released_step"))
    F._edit_episode_notes(root, 4, lambda notes: notes.update(box_released_step=312.5))
    assert _c5_member_messages(root, "box_released_step")
    F._edit_episode_notes(root, 4, lambda notes: notes.update(camera_in_box_steps=[0, 0, 0, 0, 0]))
    assert any("notes.camera_in_box_steps 이" in m and "object|null" in m
               for m in _c5_member_messages(root, "camera_in_box_steps"))
    F._edit_episode_notes(root, 4, lambda notes: notes.update(degenerate_reason=5))
    assert any("notes.degenerate_reason 이" in m and "string|null" in m
               for m in _c5_member_messages(root, "degenerate_reason"))
    for member in _SIDECAR_ONLY_NOTES:
        F._edit_episode_notes(root, 3, lambda notes: notes.pop(member))
        assert f"notes.{member} 없음" in _c5_member_messages(root, member), member


_TASK_RECORD = ("task_stages", "human_stature_m", "task_time_limit_s")


def _stage_messages(root):
    rep = v.run(root, {"C5"}, None, 3, None)
    return [f.message for f in rep.findings if f.check == "C5" and f.severity == v.ERROR
            and "notes.task_stages" in f.message]


def _hri_root(tmp_path):
    root = tmp_path / "hri"
    F.build(root, n_ep=5, hri=True)
    return root


def test_c5_reads_the_task_record_by_shape(tmp_path):
    """notes.task_stages is a list of stages, each with its name, a done flag, the step it ended at and its
    duration; a container that is not a list, a stage without a member, a flag that disagrees with the
    step and a key that is zero are each named by the member."""
    root = _hri_root(tmp_path)
    assert not _stage_messages(root)
    F._edit_episode_notes(root, 4, lambda notes: notes.update(task_stages={"enter": 1}))
    assert any("notes.task_stages 이" in m and "array|null" in m for m in _stage_messages(root))
    F._edit_episode_notes(root, 4, lambda notes: notes.update(task_stages=[]))
    assert any("notes.task_stages 가 비어 있음" in m for m in _stage_messages(root))
    F._edit_episode_notes(root, 4, lambda notes: notes.update(task_stages=["enter"]))
    assert any("notes.task_stages[0] 가 객체가 아님" in m for m in _stage_messages(root))
    F._edit_episode_notes(root, 4, lambda notes: notes.update(
        task_stages=[{"name": "enter", "completed": True, "duration_s": 1.0}]))
    assert any("notes.task_stages[0].end_step 없음" in m for m in _stage_messages(root))
    F._edit_episode_notes(root, 4, lambda notes: notes.update(task_stages=[{}]))
    said = _stage_messages(root)
    assert all(any(f"notes.task_stages[0].{key} 없음" in m for m in said)
               for key in v.NOTES_MEMBER_SHAPES["task_stages[]"]), said
    for member, value in (("end_step", -2), ("end_step", 3.0), ("completed", 1), ("name", " "),
                          ("duration_s", "1")):
        stage = {"name": "enter", "completed": False, "end_step": -1, "duration_s": None, member: value}
        F._edit_episode_notes(root, 4, lambda notes: notes.update(task_stages=[stage]))
        assert any(f"notes.task_stages[0].{member}" in m for m in _stage_messages(root)), (member, value)
    # the flag against the step, both ways
    for completed, end in ((True, -1), (False, 5)):
        stage = {"name": "enter", "completed": completed, "end_step": end, "duration_s": None}
        F._edit_episode_notes(root, 4, lambda notes: notes.update(task_stages=[stage]))
        assert any("notes.task_stages[0] 의 completed" in m and "end_step" in m
                   for m in _stage_messages(root)), (completed, end)


def test_c5_asks_a_scripted_scenario_for_the_task_record_and_a_plain_one_for_nothing(tmp_path):
    """Each case starts from a fresh copy of the clean release and expects its own sentence: a case left on the copy
    the one before it edited is said by the earlier edit, whatever its own check does."""
    built, built_plain = _hri_root(tmp_path), tmp_path / "plain"
    F.build(built_plain, n_ep=5)
    copies = iter(range(10 ** 6))

    def fresh(source):
        root = tmp_path / f"case-{next(copies)}"
        shutil.copytree(source, root)
        return root

    for member in _TASK_RECORD:
        assert not _c5_member_messages(built, member) and not _c5_member_messages(built_plain, member), member
        root, plain = fresh(built), fresh(built_plain)
        F._edit_episode_notes(root, 4, lambda notes: notes.update({member: None}))
        assert [m for m in _c5_member_messages(root, member) if "null 이거나 없음" in m], member
        F._edit_episode_notes(plain, 4, lambda notes: notes.update({member: None}))
        assert not _c5_member_messages(plain, member), member
        root = fresh(built)
        F._edit_episode_notes(root, 3, lambda notes: notes.pop(member))
        assert f"notes.{member} 없음" in _c5_member_messages(root, member), member
    # A number that is not a finite positive one is the finite-positive check's alone to say; a value that is no
    # number is the member type check's.
    for member in ("human_stature_m", "task_time_limit_s"):
        for bad, said in ((0, f"notes.{member} 이 유한한 양수가 아님: 0"),
                          (-1.0, f"notes.{member} 이 유한한 양수가 아님: -1.0"),
                          ("1.74", f"notes.{member} 이 number|null 이 아님: '1.74'"),
                          (True, f"notes.{member} 이 number|null 이 아님: True")):
            root = fresh(built)
            F._edit_episode_notes(root, 2, lambda notes: notes.update({member: bad}))
            assert _c5_member_messages(root, member) == [said], (member, bad)


def test_a_task_whose_stages_are_all_open_is_not_a_defect(tmp_path):
    """A task that was not done in time, or one the robot never received, leaves every stage open (end_step -1,
    duration_s null)."""
    root = _hri_root(tmp_path)
    open_stages = [{"name": name, "completed": False, "end_step": -1, "duration_s": None}
                   for name in F.HRI_STAGE_NAMES]
    F._edit_episode_notes(root, 4, lambda notes: notes.update(task_stages=open_stages, task_completed_step=-1))
    assert not _stage_messages(root)


def test_the_plain_fixture_ships_the_new_members_as_null_and_the_hri_one_with_a_value(tmp_path):
    """A scenario with no script has nothing to say for the three, so they are null there; the hri
    fixture gives them values of their own, which is what the type check above reads."""
    plain, hri = tmp_path / "plain", tmp_path / "hri"
    F.build(plain, n_ep=5)
    F.build(hri, n_ep=5, hri=True)
    plain_notes = [json.loads(n) for n in pd.read_parquet(F._eps_path(plain))["notes"]]
    hri_notes = [json.loads(n) for n in pd.read_parquet(F._eps_path(hri))["notes"]]
    assert all(n[m] is None for n in plain_notes for m in _SIDECAR_ONLY_NOTES)
    assert all(isinstance(n["box_released_step"], int) and n["box_released_step"] >= -1 for n in hri_notes)
    assert all(n["degenerate_reason"] is None for n in hri_notes)
    assert all(isinstance(n["camera_in_box_steps"], dict) and len(n["camera_in_box_steps"]) == 5
               for n in hri_notes)
    assert all(n[m] is None for n in plain_notes for m in _TASK_RECORD)
    for n in hri_notes:
        assert [s["name"] for s in n["task_stages"]] == list(F.HRI_STAGE_NAMES)
        assert all(s["completed"] == (s["end_step"] >= 0) for s in n["task_stages"])
        done = n["task_completed_step"]
        # a done task ends its last stage at the step it was done; one that was not leaves it open
        assert n["task_stages"][-1]["end_step"] == done
        assert n["human_stature_m"] > 0 and n["task_time_limit_s"] > 0


def test_an_hri_episode_carries_the_directed_parameters_and_the_table():
    names, table = v.DIRECTED_SSM_IN_NOTES, {"walking": {"empty": [[0.3, 0.5, 0.1]]}}
    good = {"ssm_params": {**{n: 0.1 for n in names}, "stop_table": table}}
    assert not [p for p in _said(good, ssm_names=names, stop_table=table) if "ssm_params" in p]
    without = {"ssm_params": {n: 0.1 for n in names}}
    assert any("stop_table 없음" in p for p in _said(without, ssm_names=names, stop_table=table))
    other = {"ssm_params": {**{n: 0.1 for n in names}, "stop_table": {"walking": {}}}}
    assert any("stop_table 이 정본" in p for p in _said(other, ssm_names=names, stop_table=table))


def _canon_stop_table() -> dict:
    rules = json.loads((REPO_ROOT / "meta" / "rules.json").read_text(encoding="utf-8"))
    return next(r for r in rules["rules"] if r["rule_id"] == v.DIRECTED_SSM_RULE_ID)["stop_table"]


def _table_said(notes_table, shipped_table):
    notes = {"ssm_params": {**{n: 0.1 for n in v.DIRECTED_SSM_IN_NOTES}, "stop_table": notes_table}}
    return [p for p in _said(notes, ssm_names=v.DIRECTED_SSM_IN_NOTES, stop_table=shipped_table)
            if "stop_table" in p]


def test_a_stop_table_reworded_after_the_batch_is_the_same_table():
    """The table's method note is written for a person: re-assembly takes a change of it for a
    rewording and ships the new sentence over notes copied before it. Every episode's S_p came from
    the same numbers, so the two tables agree."""
    shipped = copy.deepcopy(_canon_stop_table())
    shipped["method_note"] += " 문장을 다듬었다."
    assert not _table_said(_canon_stop_table(), shipped)


def _change_row(t):
    t["walking"]["empty"][0][1] = 9.9


def _change_reach(t):
    t["turning_reach_m"] = 0.5


def _change_date(t):
    t["measured_on"] = "2026-10-18"


def _change_units(t):
    t["row_units"]["walking"][0] = "km/h"


@pytest.mark.parametrize("change", [_change_row, _change_reach, _change_date, _change_units],
                         ids=["row", "turning_reach_m", "measured_on", "row_units"])
def test_a_stop_table_differing_in_a_value_is_named(change):
    shipped = copy.deepcopy(_canon_stop_table())
    change(shipped)
    assert any("stop_table 이 정본" in p for p in _table_said(_canon_stop_table(), shipped))


def _empty_mass_as_false(t):
    t["carry_mass_kg"]["empty"] = False      # == 0 in Python


def test_a_true_or_false_in_the_notes_table_is_not_the_number_it_equals():
    """In Python False == 0 and True == 1, so a plain comparison would pass an episode table with
    false where the canon has 0. The two tables are the same numbers only when the numbers are numbers --
    while 2 and 2.0 stay the same number (a recorder may write either)."""
    notes = copy.deepcopy(_canon_stop_table())
    _empty_mass_as_false(notes)
    assert _canon_stop_table()["carry_mass_kg"]["empty"] == 0
    assert any("stop_table 이 정본" in p for p in _table_said(notes, _canon_stop_table()))
    widened = copy.deepcopy(_canon_stop_table())
    widened["carry_mass_kg"] = {k: float(x) for k, x in widened["carry_mass_kg"].items()}
    assert not _table_said(widened, _canon_stop_table())


def test_the_validator_reads_prose_keys_as_the_assembly_does():
    """The validator's copy of the assembly's prose convention (it imports nothing from generation/):
    the same names, and a key ending in _note, read alike key by key."""
    from generation import assemble_batch as A

    assert v.PROSE_KEYS == A._PROSE_KEYS
    keys = sorted(A._PROSE_KEYS | {"method_note", "x_note", "note_x", "notes", "measured_on",
                                   "walking", "turning_reach_m", "robot_task", "value"})
    assert [v.is_prose_key(k) for k in keys] == [A._prose(k, k, "a b", "a c") for k in keys]


def test_the_other_scenarios_are_asked_the_ssm_01_parameters():
    assert any("human_speed" in p for p in _said({"ssm_params": {n: 0.1 for n in v.DIRECTED_SSM_IN_NOTES}}))


def test_the_directed_notes_are_the_rule_parameters():
    assert v.DIRECTED_SSM_IN_NOTES == v.REQUIRED_RULE_PARAMETERS[v.DIRECTED_SSM_RULE_ID]


def _rule(value):
    return {"rules": [{"rule_id": "R-TS15066-SSM-02",
                       "parameters": {"approach_speed_threshold": {"value": value}}}]}


@pytest.mark.parametrize("rules, threshold", [
    (_rule(0.05), 0.05), (_rule(True), None), (_rule(-0.1), None), (_rule(float("nan")), None),
    ({"rules": []}, None), ([], None),
])
def test_the_threshold_is_read_or_said_unreadable(rules, threshold):
    assert v.directed_approach_threshold(rules) == threshold


@pytest.mark.parametrize("entry, declared", [
    ({"rules": ["R-TS15066-A2-*", "R-TS15066-SSM-02"]}, True),
    ({"rules": ["R-TS15066-SSM-*"]}, True),
    ({"rules": ["R-TS15066-SSM-01"]}, False),
    ({"rules": []}, False),
    (None, False),
])
def test_which_scenarios_are_hri(entry, declared):
    assert v.declares_directed_ssm(entry) is declared


def _canon_entry(scenario_id: str) -> dict:
    canon = json.loads((REPO_ROOT / "meta" / "scenarios.json").read_text(encoding="utf-8"))
    return next(s for s in canon["scenarios"] if s["scenario_id"] == scenario_id)


def test_the_validator_and_the_generator_pick_the_same_hri_scenarios():
    """Two copies of one switch: the generator's (fnmatch on the selectors) and the validator's
    (an exact ID or a trailing *). They are held to one answer on the selectors a scenario can
    carry -- the exact ID, a trailing * that covers it and one that does not, the canon's own
    entries and an entry that selects nothing."""
    entries = [
        {"rules": ["R-TS15066-SSM-02"]},
        {"rules": ["R-TS15066-*"]},
        {"rules": ["R-TS15066-SSM-*"]},
        {"rules": ["R-TS15066-A2-*"]},
        _canon_entry("S1a"),
        _canon_entry("S1d"),
        _canon_entry("S1e"),
        {"scenario_id": "S9z"},
    ]
    answers = [(scenario_declares_directed_ssm(e), v.declares_directed_ssm(e)) for e in entries]
    assert all(g == w for g, w in answers), answers
    # Both answers occur: a pair of switches that said False to everything would agree too.
    assert {g for g, _ in answers} == {True, False}, answers


def test_an_untrusted_mapping_still_compares_the_violation_without_the_approach_column(tmp_path):
    """When the scenario mapping cannot be trusted, C8 reads every shipped rule as applying, so the
    hri condition is on for a release whose rules.json has R-TS15066-SSM-02 -- an S1a-shaped one
    included. Without safety.approach_speed_robot there is no gate to compare with, and the plain
    (min_distance < required) comparison still runs: the column's absence is its own ERROR, and
    the violation flag is not left unread."""
    root = tmp_path / "clean"
    F.build(root, n_ep=5)
    scenarios = root / "meta" / "scenarios.json"
    doc = json.loads(scenarios.read_text(encoding="utf-8"))
    doc["scenarios"][0]["rules"] = ["R-TS15066-A2-99"]          # a rule that is not there
    scenarios.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    step_file = root / "data" / "chunk-000" / "file-000.parquet"
    steps = pd.read_parquet(step_file)
    steps.loc[0, "safety.separation_violation"] = not bool(steps.loc[0, "safety.separation_violation"])
    F.write_steps(steps, step_file)

    rep = v.run(root, {"C8"}, None, 3, None)
    said = [f.message for f in rep.findings if f.check == "C8" and f.severity == v.ERROR]
    assert any(m.startswith("조건부 필수 스텝 필드 없음: safety.approach_speed_robot") for m in said), said
    assert any("separation_violation 이 (min_distance < required) 와 불일치: 1건" in m for m in said), said


def test_the_approach_gate_is_strict_on_both_sides_and_agrees_with_the_generator():
    """A step exactly at the approach threshold, or exactly at S_p, is not a near miss -- the
    generator's separation_violated and the validator's gate are two copies of that one rule."""
    from generation.directed_separation import separation_violated

    threshold, separation = 0.05, 0.8
    cases = [(d, r) for d in (0.5, 0.7999, 0.8, 0.8001) for r in (0.0, 0.0499, 0.05, 0.0501, 1.0)]
    for distance, speed in cases:
        mine = v.approach_gated_violation(
            np.array([distance]), np.array([separation]), np.array([speed]), threshold)[0]
        assert bool(mine) == separation_violated(distance, separation, speed, threshold), (distance, speed)
    # the two edges themselves, so that a copy that is non-strict in both would not agree by chance
    assert not v.approach_gated_violation(np.array([0.5]), np.array([0.8]), np.array([0.05]), 0.05)[0]
    assert not v.approach_gated_violation(np.array([0.8]), np.array([0.8]), np.array([1.0]), 0.05)[0]
    assert v.approach_gated_violation(np.array([0.5]), np.array([0.8]), np.array([0.0501]), 0.05)[0]


def _onset_messages(root):
    rep = v.run(root, {"C8"}, None, 3, None)
    return [f.message for f in rep.findings if f.check == "C8" and f.severity == v.ERROR
            and "contact_onset" in f.message]


def _move_first_onset(root, shift):
    """Move the release's first contact_onset ``shift`` steps later (None: drop it)."""
    for p in F._each_step_file(root):
        df = pd.read_parquet(p)
        events = [json.loads(s) for s in df["safety.event"]]
        hits = [i for i, e in enumerate(events) if "contact_onset" in e]
        if not hits:
            continue
        events[hits[0]].remove("contact_onset")
        if shift is not None:
            events[hits[0] + shift].append("contact_onset")
        df["safety.event"] = [F.as_published(e) for e in events]
        F.write_steps(df, p)
        return
    raise AssertionError("the fixture has no contact_onset to move")


@pytest.mark.parametrize("hri", [True, False], ids=["hri", "plain"])
def test_a_typed_contact_comes_at_or_after_its_episodes_contact_onset(tmp_path, hri):
    """observation.contact.type is typed on the steps the robot touches the person (spec §8.1) and
    contact_onset marks the step a contact begins (spec §6), so a typed step never comes before its
    episode's first contact_onset. The labeller knows contact by contact_onset alone; a touch it
    cannot see would ship as an episode without contact. Every scenario: the recorder types and
    marks contact the same way in all of them."""
    root = tmp_path / "r"
    F.build(root, n_ep=5, hri=hri)
    assert not _onset_messages(root)
    _move_first_onset(root, 1)
    said = _onset_messages(root)
    assert said == [said[0]] and "에피소드 1개" in said[0], said


def test_a_typed_contact_with_no_contact_onset_at_all_is_named(tmp_path):
    root = tmp_path / "hri"
    F.build(root, n_ep=5, hri=True)
    _move_first_onset(root, None)
    assert [m for m in _onset_messages(root) if "에피소드 1개" in m]


def test_a_contact_onset_with_no_typed_step_is_not_named(tmp_path):
    """A contact only the person's side registers is marked and not typed (on the S1d batch of
    400 episodes: 12 of the 117 with a contact_onset)."""
    root = tmp_path / "plain"
    F.build(root, n_ep=5)
    for p in F._each_step_file(root):
        df = pd.read_parquet(p)
        df["observation.contact.type"] = [F.as_published(["none"] * len(json.loads(s)))
                                          for s in df["observation.contact.type"]]
        F.write_steps(df, p)
    assert not _onset_messages(root)


def _c8_violation_messages(root):
    rep = v.run(root, {"C8"}, None, 3, None)
    return [f.message for f in rep.findings if f.check == "C8" and f.severity == v.ERROR
            and f.message.startswith("separation_violation 이")]


def test_a_step_inside_s_p_at_exactly_the_threshold_is_not_a_near_miss_in_the_columns(tmp_path):
    """On the release's own columns: a step short of S_p with the approach speed exactly at the
    threshold carries a false flag. The same step flagged true is the one mismatch."""
    root = tmp_path / "hri"
    F.build(root, n_ep=5, hri=True)
    threshold = v.directed_approach_threshold(json.loads((root / "meta" / "rules.json").read_text(encoding="utf-8")))
    assert threshold is not None
    step_file = root / "data" / "chunk-000" / "file-000.parquet"
    steps = pd.read_parquet(step_file)
    held = (steps["safety.min_distance_human"] < steps["safety.separation_required"]) \
        & ~steps["safety.separation_violation"].astype(bool)
    assert held.any(), "the fixture's standing-robot steps inside S_p are what this test edits"
    steps.loc[held, "safety.approach_speed_robot"] = threshold
    F.write_steps(steps, step_file)
    assert not _c8_violation_messages(root)

    steps.loc[held[held].index[0], "safety.separation_violation"] = True
    F.write_steps(steps, step_file)
    assert [m for m in _c8_violation_messages(root) if "불일치: 1건" in m]


def test_an_hri_field_the_fixture_does_not_know_is_refused(monkeypatch):
    """A field the schema lists that the fixture has no value for must stop it, not be filled in as
    a visible fraction (a number between 0 and 1) it is not."""
    known = F.hri_fields()
    monkeypatch.setattr(F, "hri_fields", lambda: [*known, "observation.unknown"])
    with pytest.raises(SystemExit, match="observation.unknown"):
        F.make_hri_episode(0, "nominal", np.random.default_rng(0))
