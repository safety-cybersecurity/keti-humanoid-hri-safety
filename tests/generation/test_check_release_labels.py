"""배포본만으로 라벨을 다시 유도해 실린 라벨과 대조한다 (check_release_labels). lerobot 도 Isaac 도 없이 돈다.

작은 배포본은 tests/generation/release_fixture.py 가 생산 경로의 함수로 만듭니다.
"""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from generation.labeling import check_release_labels as crl
from generation.labeling.evidence import EVIDENCE_KEY
from generation.labeling.label_episodes import label_episode
from tests.generation.release_fixture import (CAMERAS, FPS, HRI_SCENARIO, N, REGIONS, SCENARIO,
                                               TABLE_ASSETS, TABLE_SCENARIO,
                                               edit_episode as _edit_episode,
                                               edit_json as _edit_json, edit_notes as _edit_notes,
                                               edit_steps as _edit_steps, eight as _eight,
                                               episodes_path as _episodes_path,
                                               hri_cases, canon_rules as _rules, recording as _recording,
                                               table_cases, table_recording, write_release as _release)

# What label_episode returns, and so what the check compares. The parametrised test below
# walks this list; test_the_members_listed_here_are_the_labellers pins it to the labeller.
MEMBERS = ("outcome", "severity", "triggered_rules", "not_evaluated", "contact_observed",
           "contact_detection_limits")


def _differing(report):
    return sorted((d["episode_index"], d["member"]) for d in report["differences"])


# ── 라벨러가 쓴 라벨은 배포본에서 다시 나온다 ────────────────────────────────

def test_labels_the_labeller_wrote_follow_from_the_release(tmp_path):
    report = crl.check(_release(tmp_path, _eight()))
    assert report["episodes"] == 8 and report["steps"] == 8 * N
    assert report["agree"] == 8
    assert report["differences"] == [] and report["refused"] == []
    assert report["narrowed"] == 0          # every number a rule reads here is a float32 as it stands


def test_the_labels_derived_are_the_ones_worked_out_by_hand(tmp_path):
    report = crl.check(_release(tmp_path, _eight()))
    assert report["derived"]["outcome"] == {"nominal": 5, "near_miss": 1, "fail": 2}
    assert report["derived"]["severity"] == {"S0": 5, "S1": 1, "S2": 1, "S3": 1}
    assert report["derived"]["triggered_rules"] == {
        "R-TS15066-SSM-01": 1, "R-10218-STOP-01": 1, "R-TS15066-A2-05": 1}
    assert report["derived"]["not_evaluated"] == {"R-TS15066-A2-*": 1, "R-25785-STAB-01": 1}
    assert report["recorded"] == report["derived"]


def test_the_report_names_the_rules_each_scenario_was_labelled_under(tmp_path):
    report = crl.check(_release(tmp_path, _eight()))
    assert report["rules"] == {"S1d": ["R-10218-STOP-01", "R-TS15066-A2-01", "R-TS15066-A2-05",
                                       "R-TS15066-SSM-01"]}


# ── 어댑터는 라벨러가 읽는 입력을 빠짐없이 만든다 ───────────────────────────────

class _Asked(dict):
    """A dict that remembers which keys it was asked for, found or not."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.asked = set()

    def __getitem__(self, key):
        self.asked.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.asked.add(key)
        return super().get(key, default)

    def __contains__(self, key):
        self.asked.add(key)
        return super().__contains__(key)


def test_the_adapter_builds_every_input_the_labeller_reads(tmp_path):
    # A key the labeller starts to read with .get() would otherwise come back None from the
    # release and label on without a word. What it reads is taken from the labeller at work on
    # the eight recordings, not from a list kept here.
    read_arrays, read_meta = set(), set()
    for arrays, meta in _eight():
        arrays, meta = _Asked(arrays), _Asked(meta)
        label_episode(arrays, meta, _rules(), SCENARIO)
        read_arrays |= arrays.asked
        read_meta |= meta.asked
    assert {"contact_force", "hold_release_step", "termination"} <= read_arrays | read_meta

    root = _release(tmp_path, _eight())
    names = json.loads((root / "meta" / "safety_schema.json").read_text(encoding="utf-8"))["safety_events"]
    episodes, steps = crl.read_episodes(root), crl.read_steps(root)
    built_arrays, built_meta = crl.episode_inputs(
        episodes[3], json.loads(episodes[3]["notes"]), steps, FPS, names, crl.event_rows(steps, names))
    assert read_arrays <= set(built_arrays), sorted(read_arrays - set(built_arrays))
    assert read_meta <= set(built_meta), sorted(read_meta - set(built_meta))


def test_the_hri_labels_follow_from_the_release(tmp_path):
    report = crl.check(_release(tmp_path, hri_cases(), scenario=HRI_SCENARIO))
    assert report["agree"] == 9 and report["differences"] == [] and report["refused"] == []
    assert report["derived"]["outcome"] == {"nominal": 2, "near_miss": 1, "fail": 6}
    assert report["derived"]["severity"] == {"S0": 2, "S1": 1, "S2": 2, "S3": 4}
    assert report["derived"]["triggered_rules"] == {
        "R-TS15066-SSM-02": 1, "R-HRI-CONTACT-01": 6, "R-TS15066-A2-05": 3}
    assert report["rules"] == {"S1e": ["R-HRI-CONTACT-01", "R-TS15066-A2-01", "R-TS15066-A2-05",
                                       "R-TS15066-SSM-02"]}


def test_the_adapter_builds_every_input_the_labeller_reads_under_the_hri_judgement(tmp_path):
    read_arrays, read_meta = set(), set()
    for arrays, meta in hri_cases():
        arrays, meta = _Asked(arrays), _Asked(meta)
        label_episode(arrays, meta, _rules(), HRI_SCENARIO)
        read_arrays |= arrays.asked
        read_meta |= meta.asked
    root = _release(tmp_path, hri_cases(), scenario=HRI_SCENARIO)
    names = json.loads((root / "meta" / "safety_schema.json").read_text(encoding="utf-8"))["safety_events"]
    episodes, steps = crl.read_episodes(root), crl.read_steps(root)
    built_arrays, built_meta = crl.episode_inputs(
        episodes[5], json.loads(episodes[5]["notes"]), steps, FPS, names, crl.event_rows(steps, names))
    assert read_arrays <= set(built_arrays), sorted(read_arrays - set(built_arrays))
    assert read_meta <= set(built_meta), sorted(read_meta - set(built_meta))


def test_every_float_the_labeller_reads_is_one_the_loader_narrows():
    # FLOAT_INPUTS is kept by hand: a float array the labeller starts to read and the list leaves
    # out would be labelled from float64 here and from float32 by a reader of the release.
    # timestamp is rebuilt from fps by the adapter, not read from the loader.
    for cases, scenario in ((_eight(), SCENARIO), (hri_cases(), HRI_SCENARIO)):
        for arrays, meta in cases:
            asked = _Asked(arrays)
            label_episode(asked, meta, _rules(), scenario)
            floats = {k for k in asked.asked if k in arrays and np.asarray(arrays[k]).dtype.kind == "f"}
            assert floats - {"timestamp"} <= set(crl.FLOAT_INPUTS), sorted(floats - set(crl.FLOAT_INPUTS))


# ── 실린 라벨이 프레임에서 나오지 않으면 그 에피소드와 항목을 짚는다 ─────────────

def test_the_members_listed_here_are_the_labellers():
    arrays, meta = _recording(0)
    assert set(label_episode(arrays, meta, _rules(), SCENARIO)) == set(MEMBERS)


@pytest.mark.parametrize("member, episode, edit", [
    ("outcome", 0, lambda root: _edit_episode(root, 0, outcome="fail")),
    ("severity", 0, lambda root: _edit_episode(root, 0, severity="S3")),
    ("triggered_rules", 0, lambda root: _edit_episode(root, 0, triggered_rules=["R-10218-STOP-01"])),
    ("not_evaluated", 5, lambda root: _edit_notes(root, 5, not_evaluated=[])),
    ("contact_observed", 4, lambda root: _edit_notes(root, 4, contact_observed=False)),
    ("contact_detection_limits", 0, lambda root: _edit_notes(root, 0, contact_detection_limits=None)),
])
def test_a_recorded_member_the_frames_do_not_give_is_reported(tmp_path, member, episode, edit):
    assert member in MEMBERS
    root = _release(tmp_path, _eight())
    edit(root)
    report = crl.check(root)
    assert _differing(report) == [(episode, member)]
    assert report["agree"] == 7


@pytest.mark.parametrize("episode, number", [(0, 0), (4, 1)])
def test_a_number_is_not_the_boolean_it_equals(tmp_path, episode, number):
    # notes.contact_observed is false for episode 0 and true for episode 4; Python calls 0 == False.
    root = _release(tmp_path, _eight())
    _edit_notes(root, episode, contact_observed=number)
    assert _differing(crl.check(root)) == [(episode, "contact_observed")]


def test_a_difference_carries_both_values_and_the_episode_id(tmp_path):
    root = _release(tmp_path, _eight())
    _edit_episode(root, 1, outcome="nominal")
    (d,) = crl.check(root)["differences"]
    assert d == {"episode_index": 1, "episode_id": "S1d_0001_seed2", "member": "outcome",
                 "recorded": "nominal", "derived": "near_miss", "read": "stored"}


def test_frames_that_give_another_label_are_reported(tmp_path):
    # The episode table says nothing happened; the rows carry a 300 N transient on BR-05.
    recordings = _eight()
    quiet = label_episode(*_recording(0), _rules(), SCENARIO)
    recordings[0] = _recording(0, touch=(120, "BR-05", 300.0, "transient"))
    report = crl.check(_release(tmp_path, recordings, labels=[quiet] + [None] * 7))
    assert _differing(report) == [(0, "contact_observed"), (0, "outcome"), (0, "severity"),
                                  (0, "triggered_rules")]


def test_the_hold_release_step_decides_which_contact_counts(tmp_path):
    # Episode 4's contact is at step 90 and the notes say the hold let go at 100. Moved to 0,
    # the same rows are a contact on a free body and the label no longer follows.
    root = _release(tmp_path, _eight())
    _edit_notes(root, 4, hold_release_step=0)
    assert _differing(crl.check(root)) == [(4, "outcome"), (4, "severity"), (4, "triggered_rules")]


def test_the_limits_are_the_ones_the_release_ships(tmp_path):
    root = _release(tmp_path, _eight())

    def empty(doc):
        doc["contact_limits"]["BR-05"]["limits"]["force_quasistatic"]["value"] = None

    _edit_json(root / "meta" / "rules.json", empty)
    assert (3, "outcome") in _differing(crl.check(root))


def test_the_step_period_is_the_releases_fps(tmp_path):
    # Episode 7 stops 17 frames after the trigger: on the 0.34 s budget at 50 fps, past it at 25.
    root = _release(tmp_path, _eight())
    _edit_json(root / "meta" / "info.json", lambda doc: doc.update(fps=25))
    assert (7, "outcome") in _differing(crl.check(root))


def test_a_label_made_under_a_period_the_release_does_not_carry_is_a_difference(tmp_path):
    # The converter accepts a recording whose period is within 1e-9 s of 1/fps, and the release
    # then carries fps alone. At 0.0200000001 s the labeller's budget was 16 frames, so the stop
    # that takes 17 was labelled a fail; from the release the budget is 17 and it is not. That
    # label cannot be had from the download, and this says so rather than agreeing.
    recordings = _eight()
    arrays, meta = recordings[7]
    arrays["timestamp"] = np.arange(N) * 0.0200000001
    assert label_episode(arrays, meta, _rules(), SCENARIO)["outcome"] == "fail"
    report = crl.check(_release(tmp_path, recordings))
    assert _differing(report) == [(7, "outcome"), (7, "severity"), (7, "triggered_rules")]


# ── 적재기가 넘기는 값으로도 같은 라벨이어야 한다 ───────────────────────────────
# LeRobot 은 항목을 torch 텐서로 넘기면서 실수를 모두 float32 로 만든다(0.6.1 실측: float64 로 실린
# observation.contact.force 가 torch.float32 로 온다). 실린 값으로만 다시 유도하면, 적재기로 읽은
# 사람에게는 나오지 않는 라벨이 「일치」로 지나간다.

def _loader_differences(report):
    return sorted((d["episode_index"], d["member"], d["recorded"] if d["member"] != "triggered_rules"
                   else tuple(d["recorded"]), d["derived"] if d["member"] != "triggered_rules"
                   else tuple(d["derived"])) for d in report["differences"] if d["read"] == "loader")


def test_a_label_that_a_loader_item_no_longer_gives_is_a_difference(tmp_path):
    # 280.000001 N transient on BR-05: over the 2 x 140 N limit as stored, exactly on it as a float32.
    report = crl.check(_release(tmp_path, [_recording(0, touch=(120, "BR-05", 280.000001, "transient"))]))
    assert report["narrowed"] == 1 and report["agree"] == 0
    assert all(d["read"] == "loader" for d in report["differences"])
    assert _loader_differences(report) == [
        (0, "outcome", "fail", "nominal"), (0, "severity", "S3", "S0"),
        (0, "triggered_rules", ("R-TS15066-A2-05",), ())]


def test_numbers_a_loader_rounds_without_moving_a_verdict_agree(tmp_path):
    report = crl.check(_release(tmp_path, [_recording(0, touch=(120, "BR-05", 300.000001, "transient"))]))
    assert report["narrowed"] == 1 and report["agree"] == 1 and report["differences"] == []


def test_a_speed_a_loader_rounds_past_the_stop_threshold_is_a_difference(tmp_path):
    # A recorder that kept the root velocity in float64. 0.1499999999 m/s is a stop on the 17th frame as
    # stored; as a float32 it is 0.15000000596, the stop comes a frame later and the budget is missed.
    arrays, meta = _recording(0, stops=[30], slow_from=48)
    arrays["root_vel"] = arrays["root_vel"].astype(np.float64)
    arrays["root_vel"][47, 0] = 0.1499999999
    report = crl.check(_release(tmp_path, [(arrays, meta)]))
    assert report["narrowed"] == 1
    assert _loader_differences(report) == [
        (0, "outcome", "nominal", "fail"), (0, "severity", "S0", "S2"),
        (0, "triggered_rules", (), ("R-10218-STOP-01",))]


def test_the_notes_numbers_are_held_to_the_stored_values_only(tmp_path):
    # not_evaluated quotes a peak force. 500.123456789 N rounds as a float32, the note was written
    # from the stored value, and no verdict turns on it: not a difference.
    report = crl.check(_release(tmp_path, [_recording(0, touch=(150, "BR-05", 500.123456789, "none"))]))
    assert report["narrowed"] == 1 and report["agree"] == 1


# ── 적재기가 경로를 패턴으로 푸는 자리에 놓인 배포본 ─────────────────────────────

@pytest.mark.parametrize("name", ["rel?ase", "rel*", "release[1]"])
def test_a_root_whose_path_a_loader_would_glob_is_refused(tmp_path, name):
    # datasets resolves each file path it is handed as a pattern (measured, datasets under lerobot
    # 0.6.1: beside x?y, a sibling x0y is read first; beside x[1], only the sibling x1 is read).
    with pytest.raises(crl.ReleaseError, match="glob"):
        crl.check(_release(tmp_path / name, _eight()))


def test_the_path_checked_is_the_one_a_loader_is_handed_not_what_it_resolves_to(tmp_path):
    # LeRobot hands datasets the path as it was given, symlink names and all. Checked after
    # resolving, a link named with a glob character passed while a loader given the same name read
    # the neighbours -- and a plainly named link to such a directory was refused for nothing.
    release = _release(tmp_path / "release", _eight())
    (tmp_path / "x[1]").symlink_to(release)
    with pytest.raises(crl.ReleaseError, match="glob"):
        crl.check(tmp_path / "x[1]")

    oddly_named = _release(tmp_path / "kept[1]", _eight())
    (tmp_path / "plain").symlink_to(oddly_named)
    assert crl.check(tmp_path / "plain")["agree"] == 8


def test_a_path_that_walks_through_such_a_name_is_refused_though_it_ends_elsewhere(tmp_path):
    # x[1]/../release is release once tidied -- and the string a loader passes on is not tidied.
    _release(tmp_path / "release", _eight())
    (tmp_path / "x[1]").mkdir()
    with pytest.raises(crl.ReleaseError, match="glob"):
        crl.check(str(tmp_path / "x[1]") + "/../release")


def test_a_relative_path_is_checked_from_where_it_is_given(tmp_path, monkeypatch):
    (tmp_path / "here[1]").mkdir()
    _release(tmp_path / "here[1]" / "release", _eight())
    monkeypatch.chdir(tmp_path / "here[1]")
    with pytest.raises(crl.ReleaseError, match="glob"):
        crl.check("release")


def test_an_episode_whose_narrowed_numbers_the_labeller_refuses_is_still_counted_as_narrowed(tmp_path):
    # 1e39 m/s is finite as stored and infinite as a float32: the first labelling goes through, the
    # second is refused -- and the episode is one whose numbers a loader changes.
    arrays, meta = _recording(0)
    arrays["root_vel"] = arrays["root_vel"].astype(np.float64)
    arrays["root_vel"][0, 0] = 1e39
    report = crl.check(_release(tmp_path, [(arrays, meta)]))
    assert len(report["refused"]) == 1 and report["agree"] == 0
    assert report["narrowed"] == 1


# ── 라벨러가 받지 않는 에피소드는 건너뛰지 않고 적는다 ──────────────────────────

def test_an_episode_the_labeller_refuses_is_reported_not_skipped(tmp_path):
    recordings = _eight()
    recordings[2] = _recording(2, stops=[20, 40])       # 계약상 보호정지 onset 은 최대 하나
    stale = label_episode(*_recording(2, stops=[20]), _rules(), SCENARIO)
    report = crl.check(_release(tmp_path, recordings, labels=[None, None, stale] + [None] * 5))
    (refused,) = report["refused"]
    assert refused["episode_index"] == 2 and "onset" in refused["reason"]
    assert report["agree"] == 7 and report["differences"] == []


def test_episodes_of_a_scenario_the_release_does_not_define_are_refused(tmp_path):
    root = _release(tmp_path, _eight())
    _edit_json(root / "meta" / "scenarios.json",
               lambda doc: doc["scenarios"][0].update(scenario_id="S9"))
    report = crl.check(root)
    assert report["agree"] == 0 and len(report["refused"]) == 8
    assert all("S1d" in r["reason"] for r in report["refused"])


def test_notes_without_a_member_of_the_label_are_refused(tmp_path):
    root = _release(tmp_path, _eight())
    notes = json.loads(pq.read_table(_episodes_path(root)).to_pylist()[0]["notes"])
    del notes["contact_observed"]
    _edit_episode(root, 0, notes=json.dumps(notes, ensure_ascii=False))
    (refused,) = crl.check(root)["refused"]
    assert refused["episode_index"] == 0
    assert "notes 에" in refused["reason"] and "contact_observed" in refused["reason"]


@pytest.mark.parametrize("column, value, said", [
    ("notes", "[]", "notes"),
    ("notes", "{not json", "Expecting"),
    ("termination", "{not json", "Expecting"),
    ("termination", None, "JSON"),
    ("termination", '["fall_orientation"]', "termination"),
])
def test_an_episode_whose_notes_or_termination_cannot_be_read_is_refused(tmp_path, column, value, said):
    root = _release(tmp_path, _eight())
    _edit_episode(root, 6, **{column: value})
    (refused,) = crl.check(root)["refused"]
    assert refused["episode_index"] == 6 and said in refused["reason"]


@pytest.mark.parametrize("value", [None, "100", 100.0, True])
def test_a_hold_release_step_that_is_not_an_integer_is_refused(tmp_path, value):
    # Coerced, "100" and 100.0 would label; True would read as step 1.
    root = _release(tmp_path, _eight())
    _edit_notes(root, 3, hold_release_step=value)
    (refused,) = crl.check(root)["refused"]
    assert refused["episode_index"] == 3 and "hold_release_step" in refused["reason"]


def test_an_episode_without_a_hold_release_step_is_refused(tmp_path):
    root = _release(tmp_path, _eight())
    notes = json.loads(pq.read_table(_episodes_path(root)).to_pylist()[3]["notes"])
    del notes["hold_release_step"]
    _edit_episode(root, 3, notes=json.dumps(notes, ensure_ascii=False))
    (refused,) = crl.check(root)["refused"]
    assert refused["episode_index"] == 3 and "hold_release_step" in refused["reason"]


# ── 에피소드 표가 가리키는 행이 그 에피소드의 것이 아니면 읽지 않는다 ─────────────

def test_a_release_with_no_episode_is_not_agreement(tmp_path):
    with pytest.raises(crl.ReleaseError, match="에피소드가 없습니다"):
        crl.check(_release(tmp_path, []))


def test_rows_of_another_episode_are_refused(tmp_path):
    root = _release(tmp_path, _eight())
    _edit_episode(root, 0, dataset_from_index=N, dataset_to_index=2 * N)
    _edit_episode(root, 1, dataset_from_index=0, dataset_to_index=N)
    with pytest.raises(crl.ReleaseError, match="episode_index"):
        crl.check(root)


def test_rows_stored_out_of_place_are_refused(tmp_path):
    # Episodes 0 and 1 trade places in the file, every column value as it was. A loader goes to
    # a row by where it lies, so the table's 0..N for episode 0 now holds episode 1's frames --
    # and sorted back by ``index`` the release would read as if nothing had moved.
    root = _release(tmp_path, _eight())
    path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    pq.write_table(pa.concat_tables([table.slice(N, N), table.slice(0, N), table.slice(2 * N)]), path)
    with pytest.raises(crl.ReleaseError, match="index 열이"):
        crl.check(root)


def _place_column_as(root, column, kind):
    for path in sorted((root / "data").glob("*/*.parquet")):
        table = pq.read_table(path)
        values = table.column(column).to_numpy()
        new = (pa.array([[int(x), int(x)] for x in values], type=pa.list_(pa.int64(), 2)) if kind == "pair"
               else pa.array(values.astype(float)))
        pq.write_table(table.set_column(table.schema.get_field_index(column), column, new), path)


@pytest.mark.parametrize("kind", ["pair", "float"])
@pytest.mark.parametrize("column", ["index", "episode_index", "frame_index"])
def test_a_place_column_that_is_not_one_integer_a_step_is_refused(tmp_path, column, kind):
    """The rows are read by these positions: a pair a step cannot be compared with a position (it
    raised from numpy), and a float that equals one is not the integer the loader takes."""
    root = _release(tmp_path, _eight())
    _place_column_as(root, column, kind)
    with pytest.raises(crl.ReleaseError, match=f"{column} 이 스텝마다 정수 하나인 열이 아닙니다"):
        crl.check(root)


def test_data_files_out_of_place_are_refused(tmp_path):
    root = _release(tmp_path, _eight())
    first, second = (root / "data" / "chunk-000" / f"file-{k:03d}.parquet" for k in (0, 1))
    held = first.read_bytes()
    first.write_bytes(second.read_bytes())
    second.write_bytes(held)
    with pytest.raises(crl.ReleaseError, match="index 열이"):
        crl.check(root)


def test_episode_rows_stored_out_of_place_are_refused(tmp_path):
    # LeRobot takes episode k's pointers from the k-th row of the table.
    root = _release(tmp_path, _eight())
    table = pq.read_table(_episodes_path(root))
    pq.write_table(pa.concat_tables([table.slice(1, 1), table.slice(0, 1), table.slice(2)]),
                   _episodes_path(root))
    with pytest.raises(crl.ReleaseError, match="episode_index 가"):
        crl.check(root)


# LeRobot reads the parquet files exactly one directory down (io_utils.load_nested_dataset:
# sorted(dir.glob("*/*.parquet"))). A file anywhere else is not in what a loader gets.

@pytest.mark.parametrize("where", ["aaa.parquet", "chunk-000/deeper/file-000.parquet"])
def test_an_episode_row_in_a_file_a_loader_does_not_read_is_not_read(tmp_path, where):
    # Episode 0's row moves to a file outside the layout. Read with the file, the table is whole
    # and in place; a loader's table starts at episode 1, and takes episode 0's pointers from it.
    root = _release(tmp_path, _eight())
    table = pq.read_table(_episodes_path(root))
    stray = root / "meta" / "episodes" / where
    stray.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.slice(0, 1), stray)
    pq.write_table(table.slice(1), _episodes_path(root))
    with pytest.raises(crl.ReleaseError, match="episode_index 가"):
        crl.check(root)


@pytest.mark.parametrize("where", ["aaa.parquet", "chunk-000/deeper/file-000.parquet"])
def test_step_rows_in_a_file_a_loader_does_not_read_are_not_read(tmp_path, where):
    root = _release(tmp_path, _eight())
    first = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(first)
    stray = root / "data" / where
    stray.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.slice(0, N), stray)            # episode 0's rows
    pq.write_table(table.slice(N), first)
    with pytest.raises(crl.ReleaseError, match="index 열이"):
        crl.check(root)


def test_a_release_whose_tables_lie_outside_the_layout_has_none(tmp_path):
    root = _release(tmp_path, _eight(), per_file=8)
    (root / "data" / "chunk-000" / "file-000.parquet").rename(root / "data" / "file-000.parquet")
    _edit_json(root / "meta" / "info.json", lambda doc: doc.update(data_path="data/file-{file_index:03d}.parquet"))
    with pytest.raises(crl.ReleaseError, match="data 에"):
        crl.check(root)


@pytest.mark.parametrize("template", ["data/{bogus}.parquet", "data/{0}.parquet", ["data/{chunk_index}"],
                                      "data/{chunk_index", "data/{chunk_index.real.x}.parquet",
                                      "data/{chunk_index[0]}.parquet", "data/{chunk_index}\x00.parquet"])
def test_a_data_path_the_loader_cannot_fill_in_is_refused_not_a_traceback(tmp_path, template):
    """The template is the release's own text: a field the loader does not give, a positional one,
    a malformed one or a value that is not text must come back as the check's refusal."""
    root = _release(tmp_path, _eight())
    _edit_json(root / "meta" / "info.json", lambda doc: doc.update(data_path=template))
    with pytest.raises(crl.ReleaseError, match="data_path"):
        crl.check(root)


def test_a_length_the_rows_do_not_have_is_refused(tmp_path):
    root = _release(tmp_path, _eight())
    _edit_episode(root, 7, length=N - 1)
    with pytest.raises(crl.ReleaseError, match="length"):
        crl.check(root)


def test_a_span_that_leaves_the_step_table_is_refused(tmp_path):
    root = _release(tmp_path, _eight())
    _edit_episode(root, 7, dataset_to_index=8 * N + 5, length=N + 5)
    with pytest.raises(crl.ReleaseError, match="구간이 아닙니다"):
        crl.check(root)


def test_a_pointer_that_is_not_an_integer_is_refused(tmp_path):
    root = _release(tmp_path, _eight())
    _edit_episode(root, 2, dataset_from_index=None)
    with pytest.raises(crl.ReleaseError, match="dataset_from_index"):
        crl.check(root)


def test_a_data_file_that_does_not_exist_is_refused(tmp_path):
    root = _release(tmp_path, _eight())
    _edit_episode(root, 0, **{"data/file_index": 9})
    with pytest.raises(crl.ReleaseError, match="file-009.parquet 이 없습니다"):
        crl.check(root)


def test_an_episode_table_without_a_column_it_is_read_by_is_refused(tmp_path):
    root = _release(tmp_path, _eight())
    pq.write_table(pq.read_table(_episodes_path(root)).drop_columns(["notes"]), _episodes_path(root))
    with pytest.raises(crl.ReleaseError, match="notes"):
        crl.check(root)


@pytest.mark.parametrize("gone, said", [("data", "data 에"), ("meta/episodes", "meta/episodes 에")])
def test_a_release_without_its_tables_is_refused(tmp_path, gone, said):
    import shutil
    root = _release(tmp_path, _eight())
    shutil.rmtree(root / gone)
    with pytest.raises(crl.ReleaseError, match=said):
        crl.check(root)


def test_a_data_file_the_rows_are_not_in_is_refused(tmp_path):
    root = _release(tmp_path, _eight())
    _edit_episode(root, 0, **{"data/file_index": 1})
    with pytest.raises(crl.ReleaseError, match="file"):
        crl.check(root)


def test_rows_no_episode_claims_are_refused(tmp_path):
    root = _release(tmp_path, _eight())
    table = pq.read_table(_episodes_path(root))
    pq.write_table(table.slice(0, 7), _episodes_path(root))
    with pytest.raises(crl.ReleaseError, match="행"):
        crl.check(root)


def test_an_episode_listed_twice_is_refused(tmp_path):
    root = _release(tmp_path, _eight())
    table = pq.read_table(_episodes_path(root))
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), _episodes_path(root))
    with pytest.raises(crl.ReleaseError, match="episode_index 가"):
        crl.check(root)


def test_an_event_the_schema_does_not_list_is_refused(tmp_path):
    root = _release(tmp_path, _eight())

    def drop(doc):
        doc["safety_events"] = [e for e in doc["safety_events"] if e != "ssm_violation"]

    _edit_json(root / "meta" / "safety_schema.json", drop)
    with pytest.raises(crl.ReleaseError, match="ssm_violation"):
        crl.check(root)


def test_a_schema_without_event_names_is_refused(tmp_path):
    root = _release(tmp_path, _eight())
    _edit_json(root / "meta" / "safety_schema.json", lambda doc: doc.pop("safety_events"))
    with pytest.raises(crl.ReleaseError, match="safety_events"):
        crl.check(root)


@pytest.mark.parametrize("value, said", [
    ("not json", "JSON 이 아닌"),
    ('{"a": 1}', "문자열 목록이 아닌"),
    ("[1, 2]", "문자열 목록이 아닌"),
])
def test_a_json_column_holding_something_else_is_refused(tmp_path, value, said):
    root = _release(tmp_path, _eight())
    _edit_steps(root, "safety.event", lambda events: events.__setitem__(5, value))
    with pytest.raises(crl.ReleaseError, match=said):
        crl.check(root)


def test_data_files_that_type_a_column_differently_are_refused(tmp_path):
    root = _release(tmp_path, _eight())                 # three data files; one of them rewritten
    path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    at = table.schema.get_field_index("safety.event")
    pq.write_table(table.set_column(at, "safety.event", pa.array([0] * table.num_rows)), path)
    with pytest.raises(crl.ReleaseError, match="형이 다릅니다"):
        crl.check(root)


def test_a_json_column_that_is_not_text_is_refused(tmp_path):
    root = _release(tmp_path, _eight(), per_file=8)
    path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    at = table.schema.get_field_index("safety.event")
    pq.write_table(table.set_column(at, "safety.event", pa.array([0] * table.num_rows)), path)
    with pytest.raises(crl.ReleaseError, match="문자열 열이 아닙니다"):
        crl.check(root)


def test_a_violation_column_that_is_not_boolean_is_refused(tmp_path):
    # label_episode casts it with astype(bool), which reads a NaN as a breach.
    root = _release(tmp_path, _eight(), per_file=8)
    path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    at = table.schema.get_field_index("safety.separation_violation")
    pq.write_table(table.set_column(at, "safety.separation_violation",
                                    pa.array([float("nan")] * table.num_rows)), path)
    with pytest.raises(crl.ReleaseError, match="bool"):
        crl.check(root)


def test_a_violation_column_with_more_than_one_value_a_step_is_refused(tmp_path):
    # Read as its first element, [False, True] would label as no breach.
    root = _release(tmp_path, _eight(), per_file=8)
    path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    at = table.schema.get_field_index("safety.separation_violation")
    pairs = [[v, True] for v in table.column("safety.separation_violation").to_pylist()]
    pq.write_table(table.set_column(at, "safety.separation_violation", pa.array(pairs)), path)
    with pytest.raises(crl.ReleaseError, match="safety.separation_violation"):
        crl.check(root)


@pytest.mark.parametrize("column", ["observation.contact.force", "observation.root_vel"])
def test_a_vector_column_stored_as_one_number_a_step_is_refused(tmp_path, column):
    root = _release(tmp_path, _eight(), per_file=8)
    path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    at = table.schema.get_field_index(column)
    pq.write_table(table.set_column(at, column, pa.array([0.0] * table.num_rows)), path)
    with pytest.raises(crl.ReleaseError, match=column):
        crl.check(root)


def test_region_names_spelled_differently_are_the_same_names(tmp_path):
    root = _release(tmp_path, _eight())

    def respell_one(names):
        names[5] = json.dumps(json.loads(names[5]), separators=(",", ":"))

    _edit_steps(root, "observation.contact.region_names", respell_one)
    assert crl.check(root)["agree"] == 8


def test_region_names_that_change_inside_an_episode_are_refused(tmp_path):
    root = _release(tmp_path, _eight())

    def reverse_one(names):
        names[5] = json.dumps(list(reversed(REGIONS)))

    _edit_steps(root, "observation.contact.region_names", reverse_one)
    with pytest.raises(crl.ReleaseError, match="region_names"):
        crl.check(root)


def test_a_contact_type_narrower_than_the_regions_is_refused(tmp_path):
    root = _release(tmp_path, _eight())

    def shorten_one(kinds):
        kinds[5] = json.dumps(["none"])

    _edit_steps(root, "observation.contact.type", shorten_one)
    with pytest.raises(crl.ReleaseError, match="contact.type"):
        crl.check(root)


def test_a_step_index_that_is_not_a_count_is_refused(tmp_path):
    root = _release(tmp_path, _eight())
    _edit_steps(root, "index", lambda index: index.__setitem__(5, 4))
    with pytest.raises(crl.ReleaseError, match="index 열이"):
        crl.check(root)


def test_frames_out_of_order_inside_an_episode_are_refused(tmp_path):
    root = _release(tmp_path, _eight())
    _edit_steps(root, "frame_index", lambda frames: frames.__setitem__(5, 99))
    with pytest.raises(crl.ReleaseError, match="frame_index 가"):
        crl.check(root)


def test_a_step_column_the_rules_read_must_be_there(tmp_path):
    root = _release(tmp_path, _eight())
    path = root / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(pq.read_table(path).drop_columns(["safety.event"]), path)
    with pytest.raises(crl.ReleaseError, match="safety.event"):
        crl.check(root)


@pytest.mark.parametrize("name", ["info.json", "rules.json", "scenarios.json", "safety_schema.json"])
def test_a_missing_canon_file_is_refused(tmp_path, name):
    root = _release(tmp_path, _eight())
    (root / "meta" / name).unlink()
    with pytest.raises(crl.ReleaseError, match=name):
        crl.check(root)


@pytest.mark.parametrize("text, said", [("{", "JSON 으로 읽을 수 없습니다"), ("[]", "JSON 객체가 아닙니다")])
def test_a_canon_file_that_is_not_a_json_object_is_refused(tmp_path, text, said):
    root = _release(tmp_path, _eight())
    (root / "meta" / "rules.json").write_text(text, encoding="utf-8")
    with pytest.raises(crl.ReleaseError, match=said):
        crl.check(root)


def test_data_files_without_a_row_are_refused(tmp_path):
    root = _release(tmp_path, _eight())
    for path in sorted((root / "data").rglob("*.parquet")):
        pq.write_table(pq.read_table(path).slice(0, 0), path)
    with pytest.raises(crl.ReleaseError, match="스텝 행이 없습니다"):
        crl.check(root)


@pytest.mark.parametrize("fps", [0, -50, "50", 50.0, True, None])
def test_an_fps_that_is_not_a_positive_integer_is_refused(tmp_path, fps):
    root = _release(tmp_path, _eight())
    _edit_json(root / "meta" / "info.json", lambda doc: doc.update(fps=fps))
    with pytest.raises(crl.ReleaseError, match="fps"):
        crl.check(root)


# ── 벡터 열을 그냥 리스트로 쓴 배포본 ─────────────────────────────────────────

def test_vector_columns_written_as_plain_lists_read_the_same(tmp_path):
    report = crl.check(_release(tmp_path, _eight(), fixed=False))
    assert report["agree"] == 8 and report["derived"]["outcome"] == {"nominal": 5, "near_miss": 1,
                                                                     "fail": 2}


def test_a_vector_of_another_length_is_refused(tmp_path):
    root = _release(tmp_path, _eight(), fixed=False)
    _edit_steps(root, "observation.contact.force", lambda force: force.__setitem__(5, [0.0]))
    with pytest.raises(crl.ReleaseError, match="observation.contact.force"):
        crl.check(root)


def test_a_null_in_a_step_column_is_refused(tmp_path):
    root = _release(tmp_path, _eight(), fixed=False)
    _edit_steps(root, "observation.root_vel", lambda vel: vel.__setitem__(5, None))
    with pytest.raises(crl.ReleaseError, match="null"):
        crl.check(root)


def test_a_null_inside_a_vector_is_refused(tmp_path):
    root = _release(tmp_path, _eight(), fixed=False)
    _edit_steps(root, "observation.contact.force", lambda force: force.__setitem__(5, [None, 0.0]))
    with pytest.raises(crl.ReleaseError, match="null"):
        crl.check(root)


# ── 명령줄 ────────────────────────────────────────────────────────────────────

def test_the_command_exits_zero_and_counts_what_it_compared(tmp_path, capsys):
    root = _release(tmp_path / "release", _eight())
    assert crl.main(["--dataset", str(root)]) == 0
    out = capsys.readouterr().out
    assert "8 / 8" in out and "1,600" in out


def test_the_command_exits_one_and_names_the_episode_on_a_difference(tmp_path, capsys):
    root = _release(tmp_path / "release", _eight())
    _edit_episode(root, 1, outcome="nominal")
    assert crl.main(["--dataset", str(root)]) == 1
    out = capsys.readouterr().out
    assert "7 / 8" in out and "S1d_0001_seed2" in out and "near_miss" in out


def test_the_command_says_which_reading_a_difference_is_of(tmp_path, capsys):
    root = _release(tmp_path / "release", [_recording(0, touch=(120, "BR-05", 280.000001, "transient"))])
    assert crl.main(["--dataset", str(root)]) == 1
    out = capsys.readouterr().out
    assert "0 / 1" in out and "달라지는 에피소드 1건" in out
    assert "outcome (적재기가 넘기는 float32 값으로)" in out       # the difference says which reading


def test_the_command_exits_one_on_a_refused_episode(tmp_path, capsys):
    root = _release(tmp_path / "release", _eight())
    notes = json.loads(pq.read_table(_episodes_path(root)).to_pylist()[3]["notes"])
    del notes["hold_release_step"]
    _edit_episode(root, 3, notes=json.dumps(notes, ensure_ascii=False))
    assert crl.main(["--dataset", str(root)]) == 1
    assert "S1d_0003_seed4" in capsys.readouterr().out


def test_the_command_exits_two_on_a_release_it_cannot_read(tmp_path, capsys):
    root = _release(tmp_path / "release", _eight())
    _edit_episode(root, 7, length=N - 1)
    assert crl.main(["--dataset", str(root)]) == 2
    assert "length" in capsys.readouterr().out


def test_the_report_file_is_the_report(tmp_path):
    root = _release(tmp_path / "release", _eight())
    _edit_episode(root, 1, outcome="nominal")
    out = tmp_path / "report.json"
    assert crl.main(["--dataset", str(root), "--out", str(out)]) == 1
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(
        json.dumps(crl.check(root), ensure_ascii=False))


# ── 조건표가 있는 시나리오의 근거 요약도 배포본에서 다시 나온다 ─────────────────────────────

def _table_release(root, cases=None, cameras=None):
    return _release(root, cases if cases is not None else table_cases(), scenario=TABLE_SCENARIO,
                    assets=TABLE_ASSETS, cameras=cameras)


def _seen_cases():
    """Two table episodes that recorded the cameras: one a robot camera sees on steps 20..59, one it never sees."""
    return [table_recording(0, 0, seen=slice(20, 60)),
            table_recording(1, 57, seen=slice(0, 0), touch=(120, "BR-05", 300.0, "transient"), completed=-1)]


def _evidence_column(root, name):
    return pq.read_table(_episodes_path(root)).column(name).to_pylist()


def test_the_members_of_a_table_scenarios_label_are_these_and_the_evidence():
    arrays, meta = table_recording(0, 0)
    assert set(label_episode(arrays, meta, _rules(), TABLE_SCENARIO)) == set(MEMBERS) | {EVIDENCE_KEY}


def test_a_table_scenarios_labels_and_evidence_follow_from_the_release(tmp_path):
    root = _table_release(tmp_path)
    report = crl.check(root)
    assert report["episodes"] == 6 and report["agree"] == 6
    assert report["differences"] == [] and report["refused"] == []
    # What the columns hold, worked out by hand (table_cases): the comparison is of values, not of empty cells.
    assert _evidence_column(root, "evidence.first_contact_s") == [None, 120 / FPS, 120 / FPS, None, 120 / FPS, None]
    assert _evidence_column(root, "evidence.force_ratio.BR-05")[1] == 300.0 / (2 * 140)
    assert _evidence_column(root, "evidence.force_ratio.BR-01")[2] == 50.0 / 130
    assert _evidence_column(root, "evidence.force_ratio.BR-01")[4] == float("inf")
    assert _evidence_column(root, "evidence.force_ratio.BR-02") == [None] * 6      # no rule of its region here
    assert _evidence_column(root, "evidence.human_first_seen_s") == [None] * 6     # no cameras canon shipped


def test_when_the_person_was_seen_follows_from_a_release_that_ships_the_cameras(tmp_path):
    root = _table_release(tmp_path, _seen_cases(), cameras=CAMERAS)
    report = crl.check(root)
    assert report["agree"] == 2 and report["differences"] == [] and report["refused"] == []
    assert _evidence_column(root, "evidence.human_first_seen_s") == [20 / FPS, None]
    assert _evidence_column(root, "evidence.human_last_seen_s") == [59 / FPS, None]


def test_the_cameras_canon_the_evidence_reads_is_the_releases(tmp_path):
    # Without its meta/cameras.json the release cannot say when the person was seen, and the label it carries says.
    root = _table_release(tmp_path, _seen_cases(), cameras=CAMERAS)
    (root / "meta" / "cameras.json").unlink()
    report = crl.check(root)
    assert _differing(report) == [(0, EVIDENCE_KEY)]
    (d,) = report["differences"]
    assert d["recorded"]["human_first_seen_s"] == 20 / FPS and d["derived"]["human_first_seen_s"] is None


def test_an_edited_evidence_cell_is_a_difference_of_the_evidence(tmp_path):
    root = _table_release(tmp_path)
    _edit_episode(root, 1, **{"evidence.force_ratio.BR-05": 0.5})
    report = crl.check(root)
    assert _differing(report) == [(1, EVIDENCE_KEY)] and report["agree"] == 5
    (d,) = report["differences"]
    assert d["read"] == "stored" and d["recorded"]["force_ratio.BR-05"] == 0.5
    assert d["derived"]["force_ratio.BR-05"] == 300.0 / (2 * 140)


def test_forces_a_loader_narrows_move_the_label_columns_only(tmp_path):
    # 300.000001 N is not a float32: the second reading labels from 300.0 N, and its ratio differs in the last digits.
    # The evidence is held to the stored numbers -- a narrowed ratio is not the one the release carries.
    root = _table_release(tmp_path, [table_recording(0, 1, touch=(120, "BR-05", 300.000001, "transient"),
                                                     completed=-1)])
    report = crl.check(root)
    assert report["narrowed"] == 1 and report["agree"] == 1 and report["differences"] == []
    assert _evidence_column(root, "evidence.force_ratio.BR-05") == [300.000001 / (2 * 140)]


def test_a_release_without_the_evidence_columns_refuses_every_table_episode(tmp_path):
    root = _table_release(tmp_path)
    table = pq.read_table(_episodes_path(root))
    pq.write_table(table.drop_columns([c for c in table.column_names if c.startswith("evidence.")]),
                   _episodes_path(root))
    report = crl.check(root)
    assert report["agree"] == 0 and [r["episode_index"] for r in report["refused"]] == list(range(6))
    assert all("evidence." in r["reason"] for r in report["refused"])


def test_a_value_stored_as_nan_is_no_value_as_a_null_is(tmp_path):
    # pandas writes NaN as a null; a table written another way may keep the NaN. Either is "no value".
    root = _table_release(tmp_path)
    table = pq.read_table(_episodes_path(root))
    at = table.schema.get_field_index("evidence.first_contact_s")
    values = [float("nan") if v is None else v for v in table.column(at).to_pylist()]
    pq.write_table(table.set_column(at, "evidence.first_contact_s", pa.array(values, pa.float64())),
                   _episodes_path(root))
    assert pq.read_table(_episodes_path(root)).column("evidence.first_contact_s").null_count == 0
    assert crl.check(root)["agree"] == 6


def test_the_adapter_builds_every_input_the_labeller_reads_for_a_table_scenario(tmp_path):
    read_arrays, read_meta = set(), set()
    for arrays, meta in _seen_cases():
        arrays, meta = _Asked(arrays), _Asked(meta)
        label_episode(arrays, meta, _rules(), TABLE_SCENARIO, CAMERAS)
        read_arrays |= arrays.asked
        read_meta |= meta.asked
    assert {"min_distance_human", "human_root_pose", "camera_pose_head_rgb", "human_stature_m"} <= read_arrays | read_meta
    root = _table_release(tmp_path, _seen_cases(), cameras=CAMERAS)
    names = json.loads((root / "meta" / "safety_schema.json").read_text(encoding="utf-8"))["safety_events"]
    episodes, steps = crl.read_episodes(root), crl.read_steps(root)
    built_arrays, built_meta = crl.episode_inputs(
        episodes[0], json.loads(episodes[0]["notes"]), steps, FPS, names, crl.event_rows(steps, names))
    assert read_arrays <= set(built_arrays), sorted(read_arrays - set(built_arrays))
    assert read_meta <= set(built_meta), sorted(read_meta - set(built_meta))
    assert built_meta["human_stature_m"] == 1.74


def test_the_evidence_columns_read_are_the_evidences():
    from generation.labeling import evidence
    assert crl.EVIDENCE_COLUMNS == evidence.EVIDENCE_COLUMNS


# ── what the judgement reads, a table release owes ───────────────────────────

def test_every_step_column_the_judgement_and_the_evidence_read_is_owed_by_a_table_release():
    """A step column the labeller or the evidence reads is one a release must carry, or the judgement could not be made
    again from the release: for each scenario with a conditions table, the inputs of the rules it declares, its
    required_fields, and the step columns the re-derivation reads (STEP_COLUMNS, and SEEN_ARRAYS where a release has
    them) are required step fields or conditional ones (hri, cameras), and none is optional -- an optional column may
    be left out. Held for the canon a release ships and the validator's constants alike."""
    from pathlib import Path

    from generation.labeling.label_episodes import selected_rules
    from validator import validate_dataset as V

    repo = Path(__file__).resolve().parents[2]
    schema = json.loads((repo / "meta" / "safety_schema.json").read_text(encoding="utf-8"))
    rules = json.loads((repo / "meta" / "rules.json").read_text(encoding="utf-8"))
    tables = [s for s in json.loads((repo / "meta" / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
              if "conditions" in s]
    assert {s["scenario_id"] for s in tables} >= {"S1e", "S1f", "S1g"}
    read = (set(crl.STEP_COLUMNS) - set(V.LEROBOT_STEP_COLUMNS)) | set(crl.SEEN_ARRAYS)
    conditions = ("hri", "cameras")
    owed_by = {
        "canon": (set(schema["required_step_fields"]).union(
            *(schema["conditional_required_step_fields"][c] for c in conditions)), set(schema["optional_step_fields"])),
        "validator": (set(V.REQUIRED_STEP_FIELDS).union(*(V.CONDITIONAL_REQUIRED_STEP_FIELDS[c] for c in conditions)),
                      set(V.OPTIONAL_STEP_FIELDS)),
    }
    for scenario in tables:
        declared = {name for rule in selected_rules(rules, scenario) for name in (rule.get("inputs") or [])}
        wanted = declared | set(scenario.get("required_fields") or []) | read
        assert declared and "observation.root_vel" in wanted
        for by, (owed, optional) in owed_by.items():
            assert not wanted - owed, (scenario["scenario_id"], by, sorted(wanted - owed))
            assert not wanted & optional, (scenario["scenario_id"], by, sorted(wanted & optional))
