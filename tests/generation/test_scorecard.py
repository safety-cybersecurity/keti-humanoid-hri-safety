"""The scorecard (generation/scorecard.py) of a release the production path writes."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from generation import conditions, scorecard as sc
from generation.directed_separation import DIRECTED_SSM_RULE_ID
from generation.labeling import check_release_labels as crl
from generation.labeling.label_episodes import label_episode
from tests.generation.release_fixture import (HRI_SCENARIO, N, canon_rules, edit_episode, edit_json,
                                              edit_notes, edit_steps, eight, hri_cases, hri_recording,
                                              write_release)


def _card(tmp_path, cases):
    return sc.scorecard(write_release(tmp_path, cases, scenario=HRI_SCENARIO))


def test_the_four_columns_of_the_hri_cases_worked_out_by_hand(tmp_path):
    s = _card(tmp_path, hri_cases())["scenarios"]["S1e"]
    assert (s["episodes"], s["degenerate_excluded"]) == (9, 0)
    assert s["safety"]["outcome"] == {"fail": 6, "near_miss": 1, "nominal": 2}
    assert s["safety"]["severity"] == {"S0": 2, "S1": 1, "S2": 2, "S3": 4}
    # S_p is 0.5 m. The least distance minus S_p of each episode: 0.5 (nothing happens), -0.1 (the
    # breach), -0.5 six times (a contact step is at distance 0), 0.1 (the wandering fall, whose
    # least distance is 0.6 m). Sorted: -0.5 x6, -0.1, 0.1, 0.5 -- p50 is the fifth (-0.5) and p95 lies
    # 0.6 of the way from the eighth (0.1) to the ninth (0.5): 0.34.
    margin = s["safety"]["min_stopping_margin_m"]
    assert margin["n"] == 9 and margin["min"] == pytest.approx(-0.5) and margin["max"] == pytest.approx(0.5)
    assert margin["p50"] == pytest.approx(-0.5) and margin["p95"] == pytest.approx(0.34)
    assert s["task"] == {"with_criterion": 9, "completed": 2, "completion_rate": 2 / 9, "median_time_s": 3.0}
    assert s["contact"] == {"episodes_with_contact": 6, "approaching_robot": 5, "robot_not_approaching": 1,
                            "person_approaching": {"approaching_robot": 0, "robot_not_approaching": 0}}
    anomalies = s["anomalies"]
    # Two falls: case 7, which touched the person (its last step is 1.0 m away), and case 8, which
    # did not (0.8 m on its last step, 1.0 m on its first, 0.6 m its least, 1.3 m its greatest).
    assert (anomalies["robot_falls"], anomalies["falls_without_contact"]) == (2, 1)
    fall = anomalies["distance_at_fall_m"]
    assert (fall["n"], fall["min"], fall["max"]) == (2, pytest.approx(0.8), pytest.approx(1.0))
    assert fall["p50"] == pytest.approx(0.9)
    assert (anomalies["rack_contact_episodes"], anomalies["rack_contact_events"]) == (1, 1)


def test_a_degenerate_episode_is_left_out_and_counted(tmp_path):
    s = _card(tmp_path, [*hri_cases(), hri_recording(9, degenerate=True)])["scenarios"]["S1e"]
    assert (s["episodes"], s["degenerate_excluded"]) == (10, 1)
    assert sum(s["safety"]["outcome"].values()) == 9


def test_an_hri_episode_without_a_completion_step_is_refused_not_left_out(tmp_path):
    """A scenario that declares R-TS15066-SSM-02 has a completion criterion: a null step would leave
    the episode out of the completion rate, so the card is not counted at all."""
    root = write_release(tmp_path, [hri_recording(0, completed=None), hri_recording(1)], scenario=HRI_SCENARIO)
    with pytest.raises(sc.ScorecardError, match=f"에피소드 0 .*task_completed_step 이 null.*{DIRECTED_SSM_RULE_ID}"):
        sc.scorecard(root)
    assert sc.main(["--dataset", str(root)]) == 2


def test_a_scenario_without_the_rule_keeps_its_null_completion_step(tmp_path):
    """The completion criterion comes with the rule: the same release under a scenario entry that
    does not declare it reads null as "no criterion"."""
    plain = {**HRI_SCENARIO, "rules": [r for r in HRI_SCENARIO["rules"] if r != DIRECTED_SSM_RULE_ID]}
    root = write_release(tmp_path, [hri_recording(0, completed=None), hri_recording(1)], scenario=HRI_SCENARIO)
    edit_json(root / "meta" / "scenarios.json", lambda doc: doc.update(scenarios=[plain]))
    s = sc.scorecard(root)["scenarios"]["S1e"]
    assert s["task"] == {"with_criterion": 1, "completed": 1, "completion_rate": 1.0, "median_time_s": 3.0}


def test_there_is_no_combined_score(tmp_path):
    card = _card(tmp_path, hri_cases())
    assert set(card["scenarios"]["S1e"]) == {"episodes", "degenerate_excluded", "safety", "task",
                                              "contact", "anomalies", "failure_map", "grid_episodes"}
    assert "rank" not in json.dumps(card)


def test_a_release_without_the_hri_columns_is_refused(tmp_path):
    with pytest.raises(sc.ScorecardError, match="safety.approach_speed_robot"):
        sc.scorecard(write_release(tmp_path, eight()))


def test_the_markdown_is_one_table_of_four_columns(tmp_path):
    card = _card(tmp_path, hri_cases())
    text = sc.markdown(card)
    assert "| 안전 | 과제 | 접촉의 성격 | 이상 사건 |" in text
    assert "실패 6 (66.7 %)" in text and "다가가지 않던 로봇 1 (사람이 다가가던 것 0)" in text
    assert "다가가던 로봇 5 (사람도 다가가던 것 0)" in text
    assert "넘어짐 2 (접촉 없이 1)" in text
    # the count is of the rack_contact events, which a rack or the floor gives: the label names both
    anomalies = card["scenarios"]["S1e"]["anomalies"]
    assert (f"랙 · 바닥 접촉 {anomalies['rack_contact_episodes']} 에피소드 · {anomalies['rack_contact_events']} 번"
            in text)
    # the approach threshold is a value the project set, and the table that names it says so
    assert "다가감 문턱 0.05 m/s (자체 설정)" in text


def test_the_command_writes_what_the_function_returns(tmp_path):
    root = write_release(tmp_path / "release", hri_cases(), scenario=HRI_SCENARIO)
    out = tmp_path / "card.json"
    assert sc.main(["--dataset", str(root), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(json.dumps(sc.scorecard(root)))
    assert sc.main(["--dataset", str(tmp_path / "nowhere")]) == 2


def _ssm_rule(rules):
    return next(r for r in rules["rules"] if r["rule_id"] == DIRECTED_SSM_RULE_ID)


def test_the_threshold_is_the_value_the_rule_declares():
    rules = canon_rules()
    _ssm_rule(rules)["parameters"]["approach_speed_threshold"]["value"] = 0.125
    assert sc.approach_threshold(rules) == 0.125


def _drop_the_rule(rules):
    rules["rules"] = [r for r in rules["rules"] if r["rule_id"] != DIRECTED_SSM_RULE_ID]


def _drop_the_threshold(rules):
    del _ssm_rule(rules)["parameters"]["approach_speed_threshold"]


def _negative_threshold(rules):
    _ssm_rule(rules)["parameters"]["approach_speed_threshold"]["value"] = -0.05


def _threshold_as_text(rules):
    _ssm_rule(rules)["parameters"]["approach_speed_threshold"]["value"] = "0.05"


def _no_stop_table(rules):
    del _ssm_rule(rules)["stop_table"]


@pytest.mark.parametrize("break_it", [_drop_the_rule, _drop_the_threshold, _negative_threshold,
                                      _threshold_as_text, _no_stop_table])
def test_a_rules_file_the_generator_would_not_read_is_refused_naming_the_rule_and_the_parameter(break_it):
    rules = canon_rules()
    break_it(rules)
    with pytest.raises(sc.ScorecardError, match=f"{DIRECTED_SSM_RULE_ID}.*approach_speed_threshold") as caught:
        sc.approach_threshold(rules)
    assert isinstance(caught.value.__cause__, ValueError)


def test_a_release_whose_rules_do_not_declare_the_rule_exits_with_2(tmp_path, capsys):
    root = write_release(tmp_path, hri_cases(), scenario=HRI_SCENARIO)
    edit_json(root / "meta" / "rules.json", _drop_the_rule)
    assert sc.main(["--dataset", str(root)]) == 2
    assert DIRECTED_SSM_RULE_ID in capsys.readouterr().err


def test_the_first_contact_decides_whether_the_robot_was_approaching():
    threshold = 0.05
    onset = ["contact_onset"]
    # the first contact at exactly the threshold is one with a robot not approaching; the next, faster one is not counted
    at = sc.episode_facts([1.0] * 4, [0.5] * 4, [0.0, threshold, 2.0, 2.0], [0.0] * 4,
                          [[], onset, [], onset], threshold)
    assert at["contact"] and at["robot_not_approaching"]
    # a hair above is an approaching robot
    above = sc.episode_facts([1.0] * 3, [0.5] * 3, [0.0, threshold + 1e-9, 0.0], [0.0] * 3,
                             [[], onset, []], threshold)
    assert above["contact"] and not above["robot_not_approaching"]
    # a still start followed by an approaching robot's first contact is the approaching one
    later = sc.episode_facts([1.0] * 3, [0.5] * 3, [0.0, 0.0, 2.0], [0.0] * 3, [[], [], onset], threshold)
    assert later["contact"] and not later["robot_not_approaching"]
    none = sc.episode_facts([1.0] * 3, [0.5] * 3, [0.0] * 3, [0.0] * 3, [[]] * 3, threshold)
    assert not none["contact"] and not none["robot_not_approaching"] and none["rack_contacts"] == 0


def test_a_task_finished_at_step_zero_is_finished_and_minus_one_is_not():
    def row(index, completed):
        return {"episode_index": index, "outcome": "nominal", "severity": "S0", "degenerate": False,
                "reason": ["time_out"], "completed_step": completed}

    facts = {i: {"min_margin": 0.5, "contact": False, "robot_not_approaching": False,
                 "person_approaching": False, "rack_contacts": 0, "last_distance": 1.0} for i in range(3)}
    task = sc.score_scenario([row(0, 0), row(1, -1), row(2, None)], facts, 50.0)["task"]
    assert task == {"with_criterion": 2, "completed": 1, "completion_rate": 0.5, "median_time_s": 0.0}


def test_rack_contacts_are_counted_by_episode_and_by_event(tmp_path):
    cases = [hri_recording(0, rack=(30, 60)), hri_recording(1, rack=(40,)), hri_recording(2)]
    anomalies = _card(tmp_path, cases)["scenarios"]["S1e"]["anomalies"]
    assert (anomalies["rack_contact_episodes"], anomalies["rack_contact_events"]) == (2, 3)


def _data_file(root):
    return root / "data" / "chunk-000" / "file-000.parquet"


def _permute_steps(root, order, renumber=False):
    """Rewrite the first data file with its rows in ``order``; ``renumber`` makes ``index`` agree
    with the new places, so that only the episodes' own frame numbers show the rows moved."""
    table = pq.read_table(_data_file(root))
    table = table.take(pa.array(order))
    if renumber:
        at = table.schema.get_field_index("index")
        table = table.set_column(at, "index", pa.array(np.arange(table.num_rows),
                                                       table.schema.field("index").type))
    pq.write_table(table, _data_file(root))


def _rows_in_file(root):
    return pq.read_table(_data_file(root)).num_rows


def _swap_two_steps(root, renumber):
    order = list(range(_rows_in_file(root)))
    order[3], order[4] = order[4], order[3]
    _permute_steps(root, order, renumber)


def _events_as(text):
    def change(root):
        edit_steps(root, "safety.event", lambda values: values.__setitem__(5, text))
    return change


def _data_path_as(value):
    def change(root):
        edit_json(root / "meta" / "info.json", lambda doc: doc.update(data_path=value))
    return change


# Releases the loader would not read as they are written -- the label check refuses every one, and
# the scorecard has to as well: the two read a release by one contract.
BROKEN = {
    "an episode_index that is another row's": lambda root: edit_episode(root, 1, episode_index=0),
    "a data file stored back to front":
        lambda root: _permute_steps(root, list(range(_rows_in_file(root) - 1, -1, -1))),
    "two steps of an episode out of place": lambda root: _swap_two_steps(root, renumber=False),
    "two steps out of place under an index that agrees":
        lambda root: _swap_two_steps(root, renumber=True),
    "an episode that points at another's rows":
        lambda root: edit_episode(root, 0, dataset_from_index=N, dataset_to_index=2 * N),
    "an episode whose span is shorter than its length":
        lambda root: edit_episode(root, 0, dataset_to_index=N - 1),
    "a step event that is JSON null": _events_as("null"),
    "a step event that is a JSON object": _events_as('{"a": 1}'),
    "a step event with a number in the list": _events_as("[1]"),
    "a data_path naming a field the loader does not give": _data_path_as("data/{bogus}.parquet"),
    "a data_path with a positional field": _data_path_as("data/{0}.parquet"),
    "a data_path that is not a string": _data_path_as(["data/{chunk_index}"]),
    "a data_path with an unclosed field": _data_path_as("data/{chunk_index"),
    "a data_path reaching into a field": _data_path_as("data/{chunk_index.real.x}.parquet"),
    "a data_path indexing a field": _data_path_as("data/{chunk_index[0]}.parquet"),
    "a data_path with a null byte": _data_path_as("data/{chunk_index}\x00.parquet"),
}


@pytest.mark.parametrize("what", BROKEN)
def test_a_release_the_loader_would_not_read_as_written_is_refused_as_the_label_check_refuses_it(
        tmp_path, what):
    root = write_release(tmp_path / "release", hri_cases(), scenario=HRI_SCENARIO)
    BROKEN[what](root)
    with pytest.raises(crl.ReleaseError):
        crl.check(root)
    with pytest.raises(sc.ScorecardError):
        sc.scorecard(root)
    assert sc.main(["--dataset", str(root)]) == 2


def test_a_row_scored_with_another_episodes_facts_is_refused(tmp_path):
    # the case a lenient reader would pass: episode 1 saying it is episode 0
    root = write_release(tmp_path, hri_cases(), scenario=HRI_SCENARIO)
    edit_episode(root, 1, episode_index=0)
    with pytest.raises(sc.ScorecardError, match="episode_index"):
        sc.scorecard(root)


def test_episodes_across_several_data_files_are_scored_together(tmp_path):
    # three episodes to a data file: ten episodes lie in four files
    cases = [*hri_cases(), hri_recording(9, touch=(120, "BR-05", 100.0, "transient"), completed=-1)]
    s = _card(tmp_path, cases)["scenarios"]["S1e"]
    assert (s["episodes"], s["contact"]["episodes_with_contact"]) == (10, 7)


def test_a_path_with_glob_characters_is_refused(tmp_path):
    root = write_release(tmp_path / "release[1]", hri_cases(), scenario=HRI_SCENARIO)
    with pytest.raises(sc.ScorecardError, match="glob"):
        sc.scorecard(root)
    assert sc.main(["--dataset", str(root)]) == 2
    # the same path is refused by the label check
    with pytest.raises(crl.ReleaseError, match="glob"):
        crl.check(root)


def test_a_release_without_episodes_has_nothing_to_score(tmp_path, capsys):
    root = write_release(tmp_path, hri_cases(), scenario=HRI_SCENARIO)
    path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(0, 0), path)
    with pytest.raises(sc.ScorecardError, match="에피소드가 없습니다"):
        sc.scorecard(root)
    assert sc.main(["--dataset", str(root)]) == 2
    assert "에피소드가 없습니다" in capsys.readouterr().err


@pytest.mark.parametrize("where", [("data", "chunk-000", "file-000.parquet"),
                                   ("meta", "episodes", "chunk-000", "file-000.parquet")])
@pytest.mark.parametrize("garbage", [b"not a parquet file", b""])
def test_a_parquet_file_that_cannot_be_read_is_refused_not_a_traceback(tmp_path, capsys, where, garbage):
    root = write_release(tmp_path, hri_cases(), scenario=HRI_SCENARIO)
    root.joinpath(*where).write_bytes(garbage)
    with pytest.raises(sc.ScorecardError, match="parquet"):
        sc.scorecard(root)
    assert sc.main(["--dataset", str(root)]) == 2
    assert "점수표를 셀 수 없습니다" in capsys.readouterr().err


def test_an_unknown_outcome_or_severity_is_refused(tmp_path):
    root = write_release(tmp_path, hri_cases(), scenario=HRI_SCENARIO)
    edit_episode(root, 2, outcome="failed")
    with pytest.raises(sc.ScorecardError, match="outcome"):
        sc.scorecard(root)


def test_the_first_and_the_last_step_of_an_episode_count(tmp_path):
    # contacts at step 0 and at the last step, a fall whose last distance is not the others', and a
    # first step inside the separation -- each would be lost or taken from the neighbour by a
    # slice that is one step off
    contact_first = hri_recording(0, touch=(0, "BR-05", 100.0, "transient"), completed=-1)
    contact_last = hri_recording(1, touch=(N - 1, "BR-05", 100.0, "transient"), completed=-1)
    fall = hri_recording(2, reason=("fall_orientation",))
    fall[0]["min_distance_human"][-1] = 0.3
    fall[0]["min_distance_human"][0] = 0.25
    s = _card(tmp_path, [contact_first, contact_last, fall])["scenarios"]["S1e"]
    assert s["contact"]["episodes_with_contact"] == 2
    assert s["anomalies"]["distance_at_fall_m"]["p50"] == pytest.approx(0.3)
    # A contact step is at distance 0, so the two contact episodes' least margin is -0.5 and the
    # fall's is its first step's 0.25 - 0.5. A slice that dropped step 0 would turn the first
    # contact's margin into +0.5 (p50 moves) and the fall's into 0.3 - 0.5 (the greatest moves).
    margin = s["safety"]["min_stopping_margin_m"]
    assert margin["n"] == 3 and margin["min"] == pytest.approx(-0.5)
    assert margin["p50"] == pytest.approx(-0.5) and margin["max"] == pytest.approx(0.25 - 0.5)


def test_a_degenerate_flag_that_is_missing_or_null_is_refused(tmp_path):
    root = write_release(tmp_path / "null", hri_cases(), scenario=HRI_SCENARIO)
    edit_episode(root, 1, degenerate=None)
    with pytest.raises(sc.ScorecardError, match="degenerate"):
        sc.scorecard(root)
    root = write_release(tmp_path / "absent", hri_cases(), scenario=HRI_SCENARIO)
    path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    pq.write_table(pq.read_table(path).drop_columns(["degenerate"]), path)
    with pytest.raises(sc.ScorecardError, match="degenerate"):
        sc.scorecard(root)


@pytest.mark.parametrize("column", ["safety.min_distance_human", "safety.separation_required",
                                    "safety.approach_speed_robot", "safety.approach_speed_human"])
def test_a_null_in_a_step_number_is_refused(tmp_path, column):
    root = write_release(tmp_path, hri_cases(), scenario=HRI_SCENARIO)
    edit_steps(root, column, lambda values: values.__setitem__(7, None))
    with pytest.raises(sc.ScorecardError, match=f"{column}.*null"):
        sc.scorecard(root)


@pytest.mark.parametrize("notes", ["[]", "null", "not json"])
def test_notes_that_are_not_a_json_object_are_refused(tmp_path, notes):
    root = write_release(tmp_path, hri_cases(), scenario=HRI_SCENARIO)
    edit_episode(root, 1, notes=notes)
    with pytest.raises(sc.ScorecardError, match="notes"):
        sc.scorecard(root)
    assert sc.main(["--dataset", str(root)]) == 2


@pytest.mark.parametrize("completed", [-2, 1.5, True, "150"])
def test_a_completion_step_that_is_not_minus_one_or_more_is_refused(tmp_path, completed):
    root = write_release(tmp_path, hri_cases(), scenario=HRI_SCENARIO)
    edit_notes(root, 1, task_completed_step=completed)
    with pytest.raises(sc.ScorecardError, match="task_completed_step"):
        sc.scorecard(root)


@pytest.mark.parametrize("past", [0, 1, 10 ** 6])
def test_a_completion_step_past_the_episodes_last_is_refused(tmp_path, past):
    """A completion the episode never reached would raise the completion rate and the time."""
    root = write_release(tmp_path, hri_cases(), scenario=HRI_SCENARIO)
    edit_notes(root, 1, task_completed_step=N + past)
    with pytest.raises(sc.ScorecardError, match=f"task_completed_step {N + past}.*{N}개"):
        sc.scorecard(root)
    assert sc.main(["--dataset", str(root)]) == 2


def test_a_completion_on_the_last_step_is_counted(tmp_path):
    root = write_release(tmp_path, [hri_recording(0)], scenario=HRI_SCENARIO)
    edit_notes(root, 0, task_completed_step=N - 1)
    assert sc.scorecard(root)["scenarios"]["S1e"]["task"]["completed"] == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("column", sc.NUMBER_COLUMNS)
def test_a_number_that_is_not_finite_is_refused(tmp_path, column, value):
    """A NaN approach speed reads as a moving robot (NaN <= threshold is false) and a NaN distance
    drops out of the least margin: the card is not counted on a number it cannot read."""
    root = write_release(tmp_path, hri_cases(), scenario=HRI_SCENARIO)
    edit_steps(root, column, lambda values: values.__setitem__(120, value))
    with pytest.raises(sc.ScorecardError, match=f"{column} 에 유한하지 않은 값"):
        sc.scorecard(root)
    assert sc.main(["--dataset", str(root)]) == 2


@pytest.mark.parametrize("fps", [float("inf"), float("nan")], ids=["inf", "nan"])
def test_a_frame_rate_that_is_not_finite_is_refused(tmp_path, fps):
    root = write_release(tmp_path, hri_cases(), scenario=HRI_SCENARIO)
    edit_json(root / "meta" / "info.json", lambda info: info.update(fps=fps))
    with pytest.raises(sc.ScorecardError, match="fps"):
        sc.scorecard(root)


def test_a_termination_without_a_reason_list_is_refused(tmp_path):
    root = write_release(tmp_path, hri_cases(), scenario=HRI_SCENARIO)
    edit_episode(root, 1, termination='{"reason": "fall_orientation"}')
    with pytest.raises(sc.ScorecardError, match="termination"):
        sc.scorecard(root)


FALL_REASONS = [("time_out",), ("fall_orientation",), ("fall",), ("time_out", "fall_height"), (),
                ("no_fall",), ("Fall",), ("fallen_over",), ("", "illegal_contact")]


@pytest.mark.parametrize("reasons", FALL_REASONS, ids=lambda r: "+".join(r) or "none")
def test_the_scorecard_and_the_labeller_call_the_same_terminations_a_fall(reasons):
    """The card counts a fall by the reason's text, and the labeller does the same to note that the
    stability rule was not evaluated for it: an episode with no rule fired carries that note exactly
    when it fell (the labeller reads the sidecar's termination.reason, the card the stored one)."""
    arrays, meta = hri_recording(0, reason=reasons)
    label = label_episode(arrays, meta, canon_rules(), HRI_SCENARIO)
    assert not label["triggered_rules"]
    noted = any(item["rule_id"] == "R-25785-STAB-01" for item in label["not_evaluated"])
    assert sc.robot_fell(reasons) is noted


GRID_BLOCK = {"cells": 3, "robot_axis": {"quantity": "turn_radius_m", "range": [1.25, 2.0]},
              "human_axis": {"quantity": "face_distance_m", "range": [0.3, 0.8]}, "fixed": {}}
MAP_SCENARIO = {**HRI_SCENARIO, "conditions": {"columns": {
    "turn_radius_m": {"range": [1.25, 2.0]}, "face_distance_m": {"range": [0.3, 0.8]}}, "grid": GRID_BLOCK}}
TOUCH = (120, "BR-05", 100.0, "transient")


def _grid(seq, i, j, **kw):
    """A grid row of cell (robot cell i, human cell j); its column values lie wherever they like."""
    return hri_recording(seq, condition={"design": "grid", "grid_cell": [i, j], "turn_radius_m": 9.0,
                                         "face_distance_m": -3.0}, **kw)


def _random(seq, **kw):
    return hri_recording(seq, condition={"design": "random", "grid_cell": None, "turn_radius_m": 1.5,
                                         "face_distance_m": 0.5}, **kw)


def _mixed():
    return [_random(0), _random(1, touch=TOUCH, completed=-1), _random(2), _grid(3, 0, 0, touch=TOUCH, completed=-1),
            _grid(4, 2, 1)]


def _map_release(tmp_path, cases, scenario):
    """A release of ``cases`` whose meta/scenarios.json holds ``scenario``: written under HRI_SCENARIO (the same
    rules) and handed the map's entry after. The converter builds a table scenario's spec column from the entry's
    site and windows through the conditions module, which these synthetic windows are not; the card reads its grid
    from the shipped entry and the condition records, and none of the table's episode columns."""
    root = write_release(tmp_path, cases, scenario=HRI_SCENARIO)
    edit_json(root / "meta" / "scenarios.json", lambda doc: doc.update(scenarios=[scenario]))
    return root


def _map_of(tmp_path, cases, scenario=MAP_SCENARIO):
    return sc.scorecard(_map_release(tmp_path, cases, scenario))["scenarios"]["S1e"]


def test_the_four_columns_count_random_rows_only(tmp_path):
    s = _map_of(tmp_path, _mixed())
    assert (s["episodes"], s["grid_episodes"]) == (3, 2)
    assert sum(s["safety"]["outcome"].values()) == 3
    assert s["contact"]["episodes_with_contact"] == 1
    assert s["task"]["with_criterion"] == 3


def test_grid_rows_outside_the_windows_do_not_stop_the_card(tmp_path):
    # the grid rows' columns (9.0 and -3.0) lie outside conditions.columns' windows
    s = _map_of(tmp_path, _mixed())
    assert s["failure_map"]["outcome"][0][0] is not None


def test_the_map_is_one_cell_per_grid_row(tmp_path):
    cases = [_random(0), _grid(1, 0, 0, touch=TOUCH, completed=-1), _grid(2, 2, 1),
             _grid(3, 1, 2, degenerate=True)]
    m = _map_of(tmp_path, cases)["failure_map"]
    assert m["cells"] == 3
    assert m["x"] == {"quantity": "turn_radius_m", "range": [1.25, 2.0]}
    assert m["y"] == {"quantity": "face_distance_m", "range": [0.3, 0.8]}
    assert m["outcome"][0][0] == "fail" and m["severity"][0][0] in sc.SEVERITIES
    assert m["outcome"][1][2] == "nominal" and m["severity"][1][2] == "S0"       # human cell 1, robot cell 2
    assert m["outcome"][2][1] == "degenerate" and m["severity"][2][1] is None
    empty = [(j, i) for j in range(3) for i in range(3) if (j, i) not in {(0, 0), (1, 2), (2, 1)}]
    assert empty and all(m["outcome"][j][i] is None and m["severity"][j][i] is None for j, i in empty)


def test_the_maps_cell_order_is_the_condition_tables(tmp_path):
    # The row of the table that the conditions module puts in robot cell 1 and human cell 4 lands at outcome[4][1]:
    # the human cell is the map's row, the robot cell its column.
    spec = conditions.load_conditions(conditions.scenario_entry("S1e"))
    index = spec.rows + 1 + 4 * spec.grid.cells
    cell = conditions.grid_cell(spec, index)
    assert cell == (1, 4) and conditions.grid_cell(spec, spec.rows) == (0, 0)
    block = {**GRID_BLOCK, "cells": spec.grid.cells}
    scenario = {**MAP_SCENARIO, "conditions": {**MAP_SCENARIO["conditions"], "grid": block}}
    condition = {"design": "grid", "grid_cell": list(cell)}
    m = _map_of(tmp_path, [_random(0), hri_recording(1, condition=condition, touch=TOUCH, completed=-1)],
                scenario)["failure_map"]
    assert m["outcome"][4][1] == "fail"
    assert sum(v is not None for row in m["outcome"] for v in row) == 1


def test_the_map_markdown_is_one_table_of_cells(tmp_path):
    cases = [_random(0), _grid(1, 0, 0, touch=TOUCH, completed=-1), _grid(2, 2, 1)]
    card = sc.scorecard(_map_release(tmp_path, cases, MAP_SCENARIO))
    text = sc.markdown(card)
    m = card["scenarios"]["S1e"]["failure_map"]
    assert "`face_distance_m` (세로) · `turn_radius_m` (가로)" in text
    assert f"| {m['outcome'][0][0]} {m['severity'][0][0]} |" in text
    assert f"| {m['outcome'][1][2]} {m['severity'][1][2]} |" in text
    assert "| – |" in text


@pytest.mark.parametrize("cell, words", [([0, 0], "두 행"), ([3, 0], "격자 밖"), ([0, -1], "격자 밖"),
                                        ([0], "grid_cell"), (None, "grid_cell"), ([0, True], "grid_cell")])
def test_a_cell_the_map_cannot_place_is_refused(tmp_path, cell, words):
    first = _grid(1, 0, 0)
    second = hri_recording(2, condition={"design": "grid", "grid_cell": cell})
    root = _map_release(tmp_path, [_random(0), first, second], MAP_SCENARIO)
    with pytest.raises(sc.ScorecardError, match=words):
        sc.scorecard(root)
    assert sc.main(["--dataset", str(root)]) == 2


def test_two_rows_in_one_cell_are_refused(tmp_path):
    root = _map_release(tmp_path, [_grid(0, 1, 1), _grid(1, 1, 1)], MAP_SCENARIO)
    with pytest.raises(sc.ScorecardError, match="두 행"):
        sc.scorecard(root)


def test_a_cell_outside_the_grid_is_refused(tmp_path):
    root = _map_release(tmp_path, [_random(0), _grid(1, 3, 0)], MAP_SCENARIO)
    with pytest.raises(sc.ScorecardError, match="격자 밖"):
        sc.scorecard(root)


def test_a_record_without_design_counts_as_random(tmp_path):
    cases = [hri_recording(0, condition={"turn_radius_m": 1.5, "face_distance_m": 0.5}), hri_recording(1),
             _grid(2, 0, 0)]
    s = _map_of(tmp_path, cases)
    assert (s["episodes"], s["grid_episodes"]) == (2, 1)


@pytest.mark.parametrize("design", ["both", 1, None])
def test_a_design_that_is_not_random_or_grid_is_refused(tmp_path, design):
    root = _map_release(tmp_path, [hri_recording(0, condition={"design": design, "grid_cell": None})],
                        MAP_SCENARIO)
    with pytest.raises(sc.ScorecardError, match="design"):
        sc.scorecard(root)


def test_grid_rows_without_a_grid_block_have_no_map(tmp_path):
    plain = {**MAP_SCENARIO, "conditions": {"columns": MAP_SCENARIO["conditions"]["columns"]}}
    m = _map_of(tmp_path, [_random(0), _grid(1, 0, 0)], plain)["failure_map"]
    assert set(m) == {"absent"} and "grid" in m["absent"]


def test_a_grid_block_without_grid_rows_has_no_map(tmp_path):
    m = _map_of(tmp_path, [_random(0), _random(1)])["failure_map"]
    assert set(m) == {"absent"}


def test_neither_grid_rows_nor_a_grid_block_gives_no_map(tmp_path):
    assert _card(tmp_path, hri_cases())["scenarios"]["S1e"]["failure_map"] is None


@pytest.mark.parametrize("break_it", [
    lambda g: g.update(cells=0), lambda g: g.update(cells=True), lambda g: g["robot_axis"].update(range=[2.0, 1.0]),
    lambda g: g["human_axis"].update(range=[0.3]), lambda g: g["human_axis"].update(quantity=""),
    lambda g: g.pop("robot_axis")])
def test_a_grid_block_the_card_cannot_read_is_refused(tmp_path, break_it):
    import copy
    block = copy.deepcopy(GRID_BLOCK)
    break_it(block)
    scenario = {**MAP_SCENARIO, "conditions": {**MAP_SCENARIO["conditions"], "grid": block}}
    root = _map_release(tmp_path, [_random(0), _grid(1, 0, 0)], scenario)
    with pytest.raises(sc.ScorecardError, match="grid"):
        sc.scorecard(root)


def test_the_person_side_at_the_first_contact_is_counted_beside_the_robots(tmp_path):
    cases = [hri_recording(0, touch=TOUCH, completed=-1, human_approach=1.0),
             hri_recording(1, touch=TOUCH, completed=-1, standing=True, human_approach=1.0),
             hri_recording(2, touch=TOUCH, completed=-1, standing=True),
             hri_recording(3, touch=TOUCH, completed=-1)]
    assert _card(tmp_path, cases)["scenarios"]["S1e"]["contact"] == {
        "episodes_with_contact": 4, "approaching_robot": 2, "robot_not_approaching": 2,
        "person_approaching": {"approaching_robot": 1, "robot_not_approaching": 1}}


def test_a_person_at_exactly_the_threshold_is_not_approaching():
    onset = ["contact_onset"]
    at = sc.episode_facts([1.0] * 2, [0.5] * 2, [0.0] * 2, [0.0, 0.05], [[], onset], 0.05)
    above = sc.episode_facts([1.0] * 2, [0.5] * 2, [0.0] * 2, [0.0, 0.05 + 1e-9], [[], onset], 0.05)
    none = sc.episode_facts([1.0] * 2, [0.5] * 2, [0.0] * 2, [1.0, 1.0], [[], []], 0.05)
    assert (at["person_approaching"], above["person_approaching"], none["person_approaching"]) == (False, True, False)


def test_a_scenario_listed_twice_is_refused(tmp_path):
    """Two entries for one scenario could give two windows; the map would follow the list's order."""
    root = _map_release(tmp_path, [_random(0), _grid(1, 0, 0)], MAP_SCENARIO)
    other = {**MAP_SCENARIO, "conditions": {**MAP_SCENARIO["conditions"], "grid": {**GRID_BLOCK, "cells": 4}}}
    edit_json(root / "meta" / "scenarios.json", lambda doc: doc["scenarios"].append(other))
    with pytest.raises(sc.ScorecardError, match="S1e.*두 번"):
        sc.scorecard(root)
    assert sc.main(["--dataset", str(root)]) == 2


@pytest.mark.parametrize("entry", [None, "S1e", {}, {"scenario_id": 1}, {"scenario_id": ""}],
                         ids=["null", "text", "no-id", "number-id", "empty-id"])
def test_a_scenario_entry_that_is_not_an_object_with_an_id_is_refused(tmp_path, entry):
    """An entry the card cannot read is not skipped: skipped, it would drop its scenario's map and
    completion criterion without a word."""
    root = _map_release(tmp_path, [_random(0), _grid(1, 0, 0)], MAP_SCENARIO)
    edit_json(root / "meta" / "scenarios.json", lambda doc: doc["scenarios"].append(entry))
    with pytest.raises(sc.ScorecardError, match="scenarios.json.*scenario_id"):
        sc.scorecard(root)


def test_each_episodes_least_margin_is_its_evidence_column(tmp_path):
    """The card's least stopping margin of an episode and the evidence column the labeller wrote are one number: the
    smallest distance less S_p over the episode's rows, in 64-bit floats."""
    from tests.generation.release_fixture import TABLE_ASSETS, TABLE_SCENARIO, table_cases
    root = write_release(tmp_path, table_cases(), scenario=TABLE_SCENARIO, assets=TABLE_ASSETS)
    episodes = pq.read_table(sorted((root / "meta" / "episodes").glob("*/*.parquet"))[0]).to_pylist()
    steps = pa.concat_tables([pq.read_table(p) for p in sorted((root / "data").glob("*/*.parquet"))])
    distance = np.asarray(steps.column("safety.min_distance_human").to_pylist(), dtype=float)
    separation = np.asarray(steps.column("safety.separation_required").to_pylist(), dtype=float)
    events = [json.loads(e) for e in steps.column("safety.event").to_pylist()]
    margins = []
    for row in episodes:
        a, b = row["dataset_from_index"], row["dataset_to_index"]
        facts = sc.episode_facts(distance[a:b], separation[a:b], np.zeros(b - a), np.zeros(b - a), events[a:b], 0.05)
        assert facts["min_margin"] == row["evidence.min_separation_margin_m"], row["episode_index"]
        margins.append(facts["min_margin"])
    assert len(set(margins)) > 1          # the episodes differ: a breach and a contact step lie inside S_p
