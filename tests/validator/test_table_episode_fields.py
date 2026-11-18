"""The episode columns of a scenario that declares a conditions table: the shape C5 holds them to.

An episode whose scenario entry in the release's meta/scenarios.json carries a ``conditions`` object carries the
columns of the shipped schema's scenario_conditional_episode_fields (spec §5): episode_spec, the evidence values and
its place in the table. C5 says, per kind of defect, one sentence -- a column the table lacks once, with the number
of episodes; a defect of a cell at the episode it is in:

- the column is there;
- episode_spec is not null and its members are the schema's set (that it is a JSON object is the JSON check's);
- condition_design is not null (its value is the enum check's); a random row's two cells are the sentinel, a grid
  row's lie on the scenario's grid (conditions.grid.cells), and a cell is an integer;
- the flat columns say what the episode's condition record in notes says (randomization.condition -- its design, and
  its grid_cell or the sentinel twice where that is null), and the record is there to say it; the cells are held to
  the record only where the designs agree and the cells passed the check above;
- an evidence value is a number or null, and the smallest separation margin is never null.

What C5 does not check here: whether a value is the one the episode's steps give (the release re-derivation tool
does that), whether the spec's members hold what the spec says, whether a grid row's cell is the one its row number
names. Without a scenario canon it can trust (none shipped, or one with a doubled id), it says nothing -- C1 says
what is wrong with the canon.
"""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validator import make_fixture as F
from validator import validate_dataset as V

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((REPO_ROOT / "meta" / "safety_schema.json").read_text(encoding="utf-8"))
TABLE = list(SCHEMA["scenario_conditional_episode_fields"]["conditions"])
SPEC_MEMBERS = list(SCHEMA["episode_spec_members"])
EVIDENCE = [f for f in TABLE if f.startswith("evidence.")]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """clean_hri (the table scenario S1e) and clean (S1a, without a table), built once."""
    root = tmp_path_factory.mktemp("fixtures")
    F.build(root / F.HRI_CLEAN, hri=True)
    F.build(root / "clean")
    return root


def _copy(built: Path, base: str, tmp_path: Path) -> Path:
    root = tmp_path / base
    shutil.copytree(built / base, root)
    return root


def _episodes(root: Path) -> pd.DataFrame:
    return pd.read_parquet(F._eps_path(root))


def _edit_episodes(root: Path, change) -> None:
    df = _episodes(root)
    change(df)
    df.to_parquet(F._eps_path(root), index=False)


def _edit_spec(row: int, change):
    def edit(df: pd.DataFrame) -> None:
        spec = json.loads(df.at[row, "episode_spec"])
        change(spec)
        df.at[row, "episode_spec"] = json.dumps(spec, ensure_ascii=False)
    return edit


def _report(root: Path) -> V.Report:
    return V.run(root, set(V.ALL_CHECKS), None, 0, None)


def _errors(rep: V.Report) -> list[V.Finding]:
    return [f for f in rep.findings if f.severity == V.ERROR]


def _grid_cells(root: Path) -> int:
    """The shipped S1e entry's cells per axis -- read from the release, not written down."""
    scenarios = json.loads((root / "meta" / "scenarios.json").read_text(encoding="utf-8"))
    entry = next(s for s in scenarios["scenarios"] if s["scenario_id"] == F.HRI_SCENARIO)
    return int(entry["conditions"]["grid"]["cells"])


def _as_grid_row(row: int, robot: int, human: int):
    """The row made a grid row on that cell, as a release says it: the flat columns and the condition record."""
    def edit(df: pd.DataFrame) -> None:
        F.as_grid_row(df, row, robot, human)
    return edit


def _as_grid_columns(row: int, robot: int, human: int):
    """The row's flat columns alone made those of a grid row on that cell; its condition record says what it said."""
    def edit(df: pd.DataFrame) -> None:
        df.at[row, "condition_design"] = "grid"
        df.at[row, "grid_cell_robot"] = robot
        df.at[row, "grid_cell_human"] = human
    return edit


def _edit_record(row: int, change):
    """``change`` applied to the row's notes.randomization (the condition record is its ``condition``)."""
    def edit(df: pd.DataFrame) -> None:
        notes = json.loads(df.at[row, "notes"])
        change(notes["randomization"])
        df.at[row, "notes"] = json.dumps(notes, ensure_ascii=False)
    return edit


def _then(*edits):
    def edit(df: pd.DataFrame) -> None:
        for one in edits:
            one(df)
    return edit


# ── the clean releases ──────────────────────────────────────────────

def test_the_table_fixture_carries_every_column_and_passes(built):
    root = built / F.HRI_CLEAN
    eps = _episodes(root)
    assert set(TABLE) <= set(eps.columns), sorted(set(TABLE) - set(eps.columns))
    assert V.main([str(root)]) == 0
    rep = _report(root)
    assert not _errors(rep), [f.line() for f in _errors(rep)]
    assert rep.stats["table_episodes"] == len(eps)


def test_a_scenario_without_a_conditions_table_needs_none_of_the_columns(built):
    root = built / "clean"
    eps = _episodes(root)
    assert not set(TABLE) & set(eps.columns)
    assert V.main([str(root)]) == 0
    assert _report(root).stats["table_episodes"] == 0


def test_the_fixtures_values_agree_with_its_own_steps_and_labels(built):
    """A fixture is a release in the release's own representation: its evidence is what its steps give -- the
    smallest margin of the distance over the required separation, the first contact's time, a force ratio above 1
    for the region whose rule the fail label names and 0.0 for the others; no seen time without camera columns;
    every row a random one."""
    root = built / F.HRI_CLEAN
    eps = _episodes(root)
    steps = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))], ignore_index=True)
    for _, row in eps.iterrows():
        mine = steps[steps["episode_index"] == row["episode_index"]]
        margin = (mine["safety.min_distance_human"] - mine["safety.separation_required"]).min()
        assert row["evidence.min_separation_margin_m"] == pytest.approx(margin, abs=1e-12)
        onsets = [i for i, e in enumerate(mine["safety.event"]) if "contact_onset" in json.loads(e)]
        if onsets:
            assert row["evidence.first_contact_s"] == pytest.approx(onsets[0] / row["fps"])
        else:
            assert math.isnan(row["evidence.first_contact_s"])
        fired = {r.rsplit("-", 1)[-1] for r in row["triggered_rules"] if r.startswith("R-TS15066-A2-")}
        ratios = {f.rsplit(".", 1)[-1]: row[f] for f in EVIDENCE if ".force_ratio.BR-" in f}
        for code, value in ratios.items():
            assert (value > 1.0) == (code.split("-")[-1] in fired), (row["episode_id"], code, value)
            if code.split("-")[-1] not in fired:
                assert value == 0.0
        assert row["evidence.force_ratio_max"] == max(ratios.values())
        assert math.isnan(row["evidence.human_first_seen_s"]) and math.isnan(row["evidence.human_last_seen_s"])
        assert (row["condition_design"], row["grid_cell_robot"], row["grid_cell_human"]) == \
            ("random", V.GRID_NOT_A_CELL, V.GRID_NOT_A_CELL)
        assert sorted(json.loads(row["episode_spec"])) == sorted(SPEC_MEMBERS)


def test_every_evidence_column_is_a_double_even_where_no_row_has_a_value(built):
    """No value is NaN, written as a double's null -- a column of None would be written as Arrow's null type."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(F._eps_path(built / F.HRI_CLEAN))
    assert {f: str(schema.field(f).type) for f in EVIDENCE} == {f: "double" for f in EVIDENCE}


# ── one defect, one sentence ────────────────────────────────────────

@pytest.mark.parametrize("column", TABLE)
def test_a_missing_column_is_one_line_naming_it_and_counting_the_episodes(built, tmp_path, column):
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    n = len(_episodes(root))
    _edit_episodes(root, lambda df: df.pop(column))
    errors = _errors(_report(root))
    assert [f.message for f in errors] == [
        f"조건표가 있는 시나리오의 에피소드 열 없음: {column} — 조건표가 있는 시나리오의 에피소드 {n}건"], \
        [f.line() for f in errors]


def _set(column: str, value, row: int = 2):
    def edit(df: pd.DataFrame) -> None:
        df.at[row, column] = value
    return edit


def _retype(column: str, cast):
    def edit(df: pd.DataFrame) -> None:
        df[column] = cast(df[column])
    return edit


# Each case: the edit of the episode table, and the one sentence C5 says of it (every ERROR of the release).
MEMBERS = " · ".join(SPEC_MEMBERS)
SENTINEL = f"{V.GRID_NOT_A_CELL} · {V.GRID_NOT_A_CELL}"
NO_RECORD = ("조건 기록(notes.randomization.condition)의 design(문자열) · grid_cell(null 또는 정수 둘)이 없거나 꼴이 "
             "아님 — 평평한 열 condition_design · grid_cell_robot · grid_cell_human 을 대 볼 기록이 없다")
DESIGN_NOT_RECORDED = ("condition_design 이 조건 기록(notes.randomization.condition)의 design 과 다름 — 같은 에피소드를 "
                       "열과 기록이 다른 종류의 행으로 말한다")
CELLS_NOT_RECORDED = ("grid_cell_robot · grid_cell_human 이 조건 기록(notes.randomization.condition)의 grid_cell 과 "
                      f"다름 — 기록이 null 이면 {SENTINEL}")
DEFECTS = {
    "the spec is null": (_set("episode_spec", None),
                         "episode_spec 가 비어 있음 — 이 에피소드가 어떤 조건에서 기록되었는지 받는 쪽이 알 수 없다"),
    "the spec lacks a member": (_edit_spec(2, lambda s: s.pop("repro")),
                                f"episode_spec 의 멤버가 규격의 {len(SPEC_MEMBERS)}개({MEMBERS})와 다름"),
    "the spec has a member more": (_edit_spec(2, lambda s: s.update(extra={})),
                                   f"episode_spec 의 멤버가 규격의 {len(SPEC_MEMBERS)}개({MEMBERS})와 다름"),
    "the spec is not an object": (_set("episode_spec", "[]"), "episode_spec 는 object 여야 하는데 list 임"),
    "the design is null": (_set("condition_design", None),
                           "condition_design 가 비어 있음 — 무작위 행인지 격자 행인지 알 수 없다"),
    "the design is no design": (_set("condition_design", "Grid"), "condition_design 허용값 아님: 'Grid'"),
    "a random row's robot cell": (_set("grid_cell_robot", 3),
                                  f"무작위 행(condition_design=random)인데 격자 칸이 {SENTINEL} 이 아님"),
    "a random row's human cell": (_set("grid_cell_human", 0),
                                  f"무작위 행(condition_design=random)인데 격자 칸이 {SENTINEL} 이 아님"),
    "a non-integral cell": (_retype("grid_cell_robot", lambda c: c.astype(float).where(c.index != 2, 2.5)),
                            "grid_cell_robot · grid_cell_human 가 정수가 아님"),
    "a null cell on a table row": (_retype("grid_cell_human", lambda c: c.astype(float).where(c.index != 2, math.nan)),
                                   "grid_cell_robot · grid_cell_human 가 정수가 아님"),
    "a cell column of booleans": (_retype("grid_cell_human", lambda c: c == 0),
                                  "grid_cell_robot · grid_cell_human 가 정수가 아님"),
    "an evidence column of text": (_retype("evidence.force_ratio_max", lambda c: c.map(repr)),
                                   "근거 요약 열(evidence.*)의 값이 수나 null 이 아님"),
    "an evidence column of booleans": (_retype("evidence.first_contact_s", lambda c: c.isna()),
                                       "근거 요약 열(evidence.*)의 값이 수나 null 이 아님"),
    "a null smallest margin": (_set("evidence.min_separation_margin_m", float("nan")),
                               "evidence.min_separation_margin_m 가 비어 있음 — 모든 스텝에 사람과의 거리와 요구 이격이 "
                               "있으므로 가장 작은 여유는 늘 있다"),
    "the columns say grid, the record random": (_as_grid_columns(2, 0, 0), DESIGN_NOT_RECORDED),
    "a grid row's cell is not the record's": (_then(_as_grid_row(2, 1, 2), _set("grid_cell_human", 3)),
                                              CELLS_NOT_RECORDED),
    "a random row's record names a cell": (_edit_record(2, lambda r: r["condition"].update(grid_cell=[0, 0])),
                                           CELLS_NOT_RECORDED),
    "no condition record": (_edit_record(2, lambda r: r.pop("condition")), NO_RECORD),
    "a condition record that is not an object": (_edit_record(2, lambda r: r.update(condition=[])), NO_RECORD),
    "a record without its design": (_edit_record(2, lambda r: r["condition"].pop("design")), NO_RECORD),
    "a record without its cell": (_edit_record(2, lambda r: r["condition"].pop("grid_cell")), NO_RECORD),
    "a record whose cell is one number": (_edit_record(2, lambda r: r["condition"].update(grid_cell=[0])), NO_RECORD),
}


@pytest.mark.parametrize("case", list(DEFECTS))
def test_each_defect_is_its_one_sentence_on_a_release_that_was_clean(built, tmp_path, case):
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    assert V.main([str(root)]) == 0
    edit, sentence = DEFECTS[case]
    _edit_episodes(root, edit)
    assert V.main([str(root)]) == 1
    errors = _errors(_report(root))
    assert {f.message for f in errors} == {sentence}, [f.line() for f in errors]
    assert all(f.check == "C5" for f in errors)
    eids = set(_episodes(root)["episode_id"])
    assert all(f.location.split(" · ")[0] in eids for f in errors), [f.location for f in errors]


def test_a_defect_of_one_cell_is_said_at_that_episode(built, tmp_path):
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    _edit_episodes(root, _set("grid_cell_robot", 3, row=4))
    eid = _episodes(root).at[4, "episode_id"]
    [error] = _errors(_report(root))
    assert error.location == f"{eid} · grid_cell_robot 3, grid_cell_human {V.GRID_NOT_A_CELL}"


@pytest.mark.parametrize("edit, where", [
    (_as_grid_columns(2, 0, 0), "열 'grid', 기록 'random'"),
    (_then(_as_grid_row(2, 1, 2), _set("grid_cell_human", 3)), "열 1, 3, 기록 [1, 2]"),
    (_edit_record(2, lambda r: r["condition"].update(grid_cell=[0, 0])), "열 -1, -1, 기록 [0, 0]"),
    (_edit_record(2, lambda r: r.pop("condition")), "randomization.condition 없음"),
    (_edit_record(2, lambda r: r["condition"].pop("design")), "design 없음"),
    (_edit_record(2, lambda r: r["condition"].update(grid_cell=[0])), "grid_cell = [0]"),
], ids=["design", "cell", "a random row's record names a cell", "no record", "no design", "a cell of one number"])
def test_a_place_that_is_not_the_records_is_said_at_that_episode_with_both_sides(built, tmp_path, edit, where):
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    _edit_episodes(root, edit)
    eid = _episodes(root).at[2, "episode_id"]
    [error] = _errors(_report(root))
    assert error.location == f"{eid} · {where}"


def test_a_row_made_a_grid_row_in_its_columns_alone_is_refused(built, tmp_path):
    """The flat columns of every fail row rewritten to a grid row on a cell of the grid, the condition record left a
    random row's: C4 reads the column and would leave the rows out of the shares, the scorecard reads the record and
    would count them. Each such row is the design sentence, once, at that row."""
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    assert V.main([str(root)]) == 0
    eps = _episodes(root)
    fails = list(eps.index[eps["outcome"] == "fail"])
    assert fails
    _edit_episodes(root, _then(*(_as_grid_columns(i, k, k) for k, i in enumerate(fails))))
    assert V.main([str(root)]) == 1
    errors = _errors(_report(root))
    assert [f.message for f in errors] == [DESIGN_NOT_RECORDED] * len(fails), [f.line() for f in errors]
    assert sorted(f.location.split(" · ")[0] for f in errors) == sorted(eps.loc[fails, "episode_id"])


@pytest.mark.parametrize("notes, sentence", [
    ("[]", "notes 는 object 여야 하는데 list 임"),
    (None, "randomization"),
], ids=["notes not an object", "randomization not an object"])
def test_a_table_episode_whose_notes_cannot_be_read_is_left_to_the_json_and_member_checks(built, tmp_path, notes,
                                                                                         sentence):
    """notes that are not a JSON object, or whose randomization is not an object, are said by the JSON fields' check
    and the notes member check; the record is not asked for a second time."""
    root = _copy(built, F.HRI_CLEAN, tmp_path)

    def edit(df: pd.DataFrame) -> None:
        df.at[2, "notes"] = notes if notes is not None else json.dumps(
            dict(json.loads(df.at[2, "notes"]), randomization=[]), ensure_ascii=False)
    _edit_episodes(root, edit)
    errors = _errors(_report(root))
    assert errors and all(sentence in f.message for f in errors), [f.line() for f in errors]
    assert not [f for f in errors if "조건 기록" in f.message]


def test_inf_and_null_are_evidence_values(built, tmp_path):
    """inf is a ratio (a transient contact where table A.2 permits none); null is no value."""
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    _edit_episodes(root, _set("evidence.force_ratio.BR-01", math.inf))
    _edit_episodes(root, _set("evidence.force_ratio.BR-02", float("nan"), row=3))
    assert V.main([str(root)]) == 0


# ── a grid row ──────────────────────────────────────────────────────

def test_a_grid_row_on_its_last_cell_passes(built, tmp_path):
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    last = _grid_cells(root) - 1
    _edit_episodes(root, _as_grid_row(2, last, last))
    assert V.main([str(root)]) == 0


@pytest.mark.parametrize("cell", ["past the robot axis", "past the human axis", "before the grid"])
def test_a_grid_row_off_the_grid_is_refused(built, tmp_path, cell):
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    n = _grid_cells(root)
    robot, human = {"past the robot axis": (n, 0), "past the human axis": (0, n),
                    "before the grid": (V.GRID_NOT_A_CELL, 0)}[cell]
    _edit_episodes(root, _as_grid_row(2, robot, human))
    errors = _errors(_report(root))
    assert [f.message for f in errors] == [
        "격자 행(condition_design=grid)의 칸이 격자 밖 — 칸 번호는 0 부터 conditions.grid.cells − 1 까지"], \
        [f.line() for f in errors]
    assert errors[0].location.endswith(f"grid_cell_robot {robot}, grid_cell_human {human} (cells {n})")


def test_a_grid_row_of_a_scenario_without_a_grid_says_so(built, tmp_path):
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    path = root / "meta" / "scenarios.json"
    scenarios = json.loads(path.read_text(encoding="utf-8"))
    for entry in scenarios["scenarios"]:
        entry["conditions"].pop("grid", None)
    path.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
    assert V.main([str(root)]) == 0                    # random rows need no grid
    _edit_episodes(root, _as_grid_row(2, 0, 0))
    errors = _errors(_report(root))
    assert [f.message for f in errors] == [
        "격자 행(condition_design=grid)인데 시나리오 정본에 격자 블록(conditions.grid 의 cells — 1 이상의 정수)이 "
        "없음 — 칸 번호를 대 볼 격자가 없다"], [f.line() for f in errors]


# ── a release that mixes a table scenario with one without ──────────

PLAIN = "S1x"           # the table scenario's entry without its conditions, under an id of its own


def _mixed_release(built: Path, tmp_path: Path) -> tuple[Path, list[int]]:
    """clean_hri with its second half moved to a scenario without a conditions table, the episode table written as
    the converter writes it: pandas builds the table from one record per episode, and the other scenario's records
    have none of the table columns -- the grid cells widen to floats, and every table column is empty on those rows."""
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    path = root / "meta" / "scenarios.json"
    scenarios = json.loads(path.read_text(encoding="utf-8"))
    entry = next(s for s in scenarios["scenarios"] if s["scenario_id"] == F.HRI_SCENARIO)
    scenarios["scenarios"].append({**{k: v for k, v in entry.items() if k != "conditions"}, "scenario_id": PLAIN})
    path.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
    records = _episodes(root).to_dict("records")
    moved = list(range(len(records) // 2, len(records)))
    for i in moved:
        records[i] = {k: v for k, v in records[i].items() if k not in TABLE}
        records[i].update(scenario_id=PLAIN,
                          episode_id=records[i]["episode_id"].replace(f"{F.HRI_SCENARIO}_", f"{PLAIN}_", 1))
    pd.DataFrame(records).to_parquet(F._eps_path(root), index=False)
    return root, moved


def test_a_release_mixing_a_table_scenario_with_one_without_is_clean(built, tmp_path):
    """The table columns are required per scenario, so a release may hold both kinds: the table rows' cells read as
    integral floats (-1.0) and are cells, and the other rows' empties are not the check's to report."""
    import pyarrow.parquet as pq

    root, moved = _mixed_release(built, tmp_path)
    schema = pq.read_schema(F._eps_path(root))
    assert {str(schema.field(f).type) for f in ("grid_cell_robot", "grid_cell_human")} == {"double"}
    eps = _episodes(root)
    assert eps.loc[moved, TABLE].isna().all().all()
    assert V.main([str(root)]) == 0
    rep = _report(root)
    assert rep.stats["table_episodes"] == len(eps) - len(moved)
    assert not [f.line() for f in rep.findings if any(t in f.message or t in f.location for t in TABLE)]


def test_a_mixed_releases_float_cells_are_still_judged(built, tmp_path):
    """Integral floats are cells: a random row's 3.0 is the cell 3, refused as a random row's cell is."""
    root, _ = _mixed_release(built, tmp_path)
    _edit_episodes(root, _set("grid_cell_robot", 3.0, row=1))
    eid = _episodes(root).at[1, "episode_id"]
    [error] = _errors(_report(root))
    assert (error.message, error.location) == (
        f"무작위 행(condition_design=random)인데 격자 칸이 {SENTINEL} 이 아님",
        f"{eid} · grid_cell_robot 3, grid_cell_human {V.GRID_NOT_A_CELL}")


# ── a conditions entry that is not an object ────────────────────────

def test_a_conditions_entry_that_is_not_an_object_is_said_once(built, tmp_path):
    """declares_conditions takes only an object as a table, so a scenario whose conditions is something else has its
    table columns not looked at -- said once for the scenario, not per episode."""
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    path = root / "meta" / "scenarios.json"
    scenarios = json.loads(path.read_text(encoding="utf-8"))
    for entry in scenarios["scenarios"]:
        entry["conditions"] = []
    path.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
    _edit_episodes(root, _set("grid_cell_robot", 3))
    rep = _report(root)
    assert [(f.check, f.message, f.location) for f in _errors(rep)] == [
        ("C5", f"시나리오 {F.HRI_SCENARIO} 의 conditions 가 객체가 아님 — 조건표가 있는 시나리오인지 알 수 없어 그 "
               "에피소드의 조건표 열을 보지 않음", "meta/scenarios.json")]
    assert rep.stats["table_episodes"] == 0


# ── when the scenario canon cannot be read ──────────────────────────

def _scenario_canon_gone(root: Path) -> None:
    (root / "meta" / "scenarios.json").unlink()


def _scenario_id_doubled(root: Path) -> None:
    path = root / "meta" / "scenarios.json"
    scenarios = json.loads(path.read_text(encoding="utf-8"))
    scenarios["scenarios"].append(dict(scenarios["scenarios"][0]))
    path.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.mark.parametrize("spoil", [_scenario_canon_gone, _scenario_id_doubled], ids=["none shipped", "an id doubled"])
def test_without_a_scenario_canon_to_trust_the_check_is_silent_and_c1_speaks(built, tmp_path, spoil):
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    _edit_episodes(root, lambda df: df.pop("episode_spec"))
    _edit_episodes(root, _set("grid_cell_robot", 3))
    spoil(root)
    rep = _report(root)
    words = (*TABLE, "격자", "조건표")
    assert not [f.line() for f in rep.findings if any(w in f.message for w in words)]
    assert "table_episodes" not in rep.stats
    assert [f for f in rep.findings if f.check == "C1" and f.severity in (V.ERROR, V.WARN)
            and f.location == "meta/scenarios.json"], "C1 says nothing of the scenario canon"


# ── the variants that copy clean_hri ────────────────────────────────

VARIANT_SENTENCES = {
    "broken_hri_spec_missing": ("조건표가 있는 시나리오의 에피소드 열 없음: episode_spec — 조건표가 있는 시나리오의 "
                                "에피소드 {n}건"),
    "broken_hri_spec_shape": f"episode_spec 의 멤버가 규격의 {len(SPEC_MEMBERS)}개({MEMBERS})와 다름",
    "broken_hri_grid_cell": f"무작위 행(condition_design=random)인데 격자 칸이 {SENTINEL} 이 아님",
    "broken_hri_evidence_text": "근거 요약 열(evidence.*)의 값이 수나 null 이 아님",
}


@pytest.mark.parametrize("variant", list(VARIANT_SENTENCES))
def test_each_table_variant_is_only_its_defect(built, tmp_path, variant):
    """A variant proves its check only on a release with nothing else wrong: its one sentence is every ERROR."""
    assert F.variant_base(variant) == F.HRI_CLEAN
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    assert V.main([str(root)]) == 0
    sentence = VARIANT_SENTENCES[variant].format(n=len(_episodes(root)))
    F.VARIANTS[variant](root)
    errors = _errors(_report(root))
    assert {f.message for f in errors} == {sentence}, [f.line() for f in errors]
