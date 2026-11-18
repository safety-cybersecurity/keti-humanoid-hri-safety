"""The step columns the judgement does not need are optional (spec §6).

Eleven step columns are required of every release: the robot's body pose and speed, the contact with
the person, the distance and the events -- what the judgement reads, whatever the robot. The robot's own
columns (joints, action, IMU, centre of mass, ZMP, stability margin), the contact relative speed and the
contact pressure are optional: a release may leave them out, a column that is there gets every check,
and one that a rule the release's scenarios declare reads (its ``inputs``, or the scenario's
``required_fields``) is required of that release all the same.

The releases here are the clean fixtures with columns taken out by ``make_fixture.drop_step_columns``,
so an exit code is asserted on a release that is otherwise without an ERROR.
"""

import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from validator import make_fixture
from validator import validate_dataset as v

# The optional columns a release recorded in the simulator carries: all but the measured contact area.
NINE = [f for f in v.OPTIONAL_STEP_FIELDS if f != "observation.contact.pressure"]


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    """The two clean fixtures, built once: the hri distribution (S1e) and the camera-less one (S1a)."""
    root = tmp_path_factory.mktemp("optional")
    make_fixture.build(root / "hri", hri=True)
    make_fixture.build(root / "s1a")
    return root


def _copy(built: Path, name: str, tmp_path: Path) -> Path:
    root = tmp_path / name
    shutil.copytree(built / name, root)
    return root


def hri_release(built: Path, tmp_path: Path) -> Path:
    return _copy(built, "hri", tmp_path)


def s1a_release(built: Path, tmp_path: Path) -> Path:
    return _copy(built, "s1a", tmp_path)


def run_validator(root: Path) -> tuple[int, dict]:
    """The validator's exit code and its JSON report, every check run as `make validate` runs them."""
    out = root.parent / f"{root.name}-report.json"
    code = v.main([str(root), "--json", str(out)])
    return code, json.loads(out.read_text(encoding="utf-8"))


def errors(report: dict) -> list[str]:
    return [f"{f['check']} {f['message']}" for f in report["findings"] if f["severity"] == v.ERROR]


def messages(report: dict, check: str) -> list[str]:
    return [f["message"] for f in report["findings"] if f["check"] == check]


def step_columns(root: Path) -> set[str]:
    return {name for p in sorted((root / "data").rglob("*.parquet")) for name in pq.read_schema(p).names}


def test_the_two_lists_are_the_twenty_that_were_required():
    assert len(v.REQUIRED_STEP_FIELDS) == 11 and len(NINE) == 9
    assert len(set(v.REQUIRED_STEP_FIELDS)) == 11 and len(set(v.OPTIONAL_STEP_FIELDS)) == 10
    assert not set(v.REQUIRED_STEP_FIELDS) & set(v.OPTIONAL_STEP_FIELDS)
    assert len(set(v.REQUIRED_STEP_FIELDS) | set(NINE)) == 20


def test_a_release_without_the_optional_columns_validates(built, tmp_path):
    root = hri_release(built, tmp_path)
    make_fixture.drop_step_columns(root, NINE)
    assert not set(NINE) & step_columns(root)
    code, report = run_validator(root)
    assert code == 0 and not errors(report), errors(report)


@pytest.mark.parametrize("column", NINE)
def test_each_optional_column_may_be_absent_alone(built, tmp_path, column):
    root = hri_release(built, tmp_path)
    make_fixture.drop_step_columns(root, [column])
    assert column not in step_columns(root)
    code, report = run_validator(root)
    assert code == 0 and not errors(report), errors(report)


@pytest.mark.parametrize("column", v.REQUIRED_STEP_FIELDS)
def test_a_required_column_absent_is_said(built, tmp_path, column):
    root = hri_release(built, tmp_path)
    make_fixture.drop_step_columns(root, [column])
    code, report = run_validator(root)
    assert code == 1
    assert f"필수 스텝 필드 없음: {column}" in messages(report, "C8")


def test_an_optional_column_a_declared_rule_reads_is_required(built, tmp_path):
    """S1a declares the stability rules and names safety.stability_margin among its required fields: in
    that release the optional column is required -- by the scenario, not by the schema's list."""
    root = s1a_release(built, tmp_path)
    make_fixture.drop_step_columns(root, ["safety.stability_margin"])
    code, report = run_validator(root)
    said = messages(report, "C8")
    assert code == 1 and any("규칙 입력·필수 필드가 데이터에 없음" in m and "safety.stability_margin" in m
                             for m in said), said
    assert "필수 스텝 필드 없음: safety.stability_margin" not in said


def test_no_rule_and_no_scenario_names_the_optional_contact_columns():
    """Spec §6 and the schema's note say of the contact relative speed and the contact pressure that no judgement
    rule reads them. A rule that comes to list one among its ``inputs``, or a scenario among its
    ``required_fields``, makes that sentence false -- and the column required of that scenario's releases."""
    repo = Path(__file__).resolve().parents[2]
    rules = json.loads((repo / "meta" / "rules.json").read_text(encoding="utf-8"))["rules"]
    scenarios = json.loads((repo / "meta" / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    named = ({f for r in rules for f in r["inputs"]}
             | {f for s in scenarios for f in s.get("required_fields", [])})
    contact = {f for f in v.OPTIONAL_STEP_FIELDS if f.startswith("observation.contact.")}
    assert len(contact) == 2 and named & set(v.REQUIRED_STEP_FIELDS)   # the lists are there to be read
    assert not named & contact


def test_an_optional_column_that_is_present_is_still_checked(built, tmp_path):
    root = hri_release(built, tmp_path)
    path = sorted((root / "data").rglob("*.parquet"))[0]
    df = pd.read_parquet(path)
    df["observation.imu"] = [row[:9] for row in df["observation.imu"]]   # a 9-long imu
    make_fixture.write_steps(df, path)
    code, report = run_validator(root)
    assert code == 1
    assert "observation.imu 길이 9 != 규격 10" in messages(report, "C8")


@pytest.mark.parametrize("edit, said", [
    (lambda schema: schema["optional_step_fields"].remove("safety.com"),
     "규격 정의의 optional_step_fields 가 검증기와 다름"),
    (lambda schema: schema.pop("optional_step_fields"), "규격 정의에 optional_step_fields 없음"),
    (lambda schema: schema.update(optional_step_fields={f: 1 for f in v.OPTIONAL_STEP_FIELDS}),
     "규격 정의의 optional_step_fields 가 목록이 아님"),
], ids=["one-short", "absent", "an-object"])
def test_the_shipped_schema_says_the_optional_list_the_validator_holds(built, tmp_path, edit, said):
    """A reader picks what a release may leave out by meta/safety_schema.json: C1 holds its list to the
    validator's, as it does the required one."""
    root = hri_release(built, tmp_path)
    path = root / "meta" / "safety_schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    edit(schema)
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    code, report = run_validator(root)
    assert code == 1 and any(m.startswith(said) for m in messages(report, "C1")), messages(report, "C1")


def test_the_lean_fixture_is_the_hri_one_without_the_nine(tmp_path):
    lean, hri = tmp_path / make_fixture.HRI_LEAN_CLEAN, tmp_path / "hri"
    make_fixture.build_hri_lean(lean)
    make_fixture.build(hri, hri=True)
    assert step_columns(lean) == step_columns(hri) - set(NINE)
    assert set(NINE) <= step_columns(hri)
    code, report = run_validator(lean)
    assert code == 0 and not errors(report), errors(report)


def test_dropping_a_column_leaves_nothing_that_names_it(built, tmp_path):
    """Every place a release names a step column: the step files, the features of meta/info.json,
    meta/stats.json and the episode table's per-episode statistics. The fixture carries no statistics,
    so the two are put in here as LeRobot writes them -- one block per feature, and stats/<column>/<stat>."""
    root = hri_release(built, tmp_path)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = list(info["features"])
    (root / "meta" / "stats.json").write_text(
        json.dumps({name: {"min": [0.0], "max": [1.0], "count": [1]} for name in features}), encoding="utf-8")
    episodes = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    for p in episodes:
        df = pd.read_parquet(p)
        for name in features:
            df[f"stats/{name}/min"] = 0.0
            df[f"stats/{name}/count"] = 1
        df.to_parquet(p, index=False)

    make_fixture.drop_step_columns(root, NINE)

    kept = [name for name in features if name not in NINE]
    assert not set(NINE) & step_columns(root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert list(info["features"]) == kept
    stats = json.loads((root / "meta" / "stats.json").read_text(encoding="utf-8"))
    assert list(stats) == kept
    for p in episodes:
        names = pq.read_schema(p).names
        assert not [n for n in names if any(n.startswith(f"stats/{c}/") for c in NINE)], p
        assert {f"stats/{name}/min" for name in kept} <= set(names), p
