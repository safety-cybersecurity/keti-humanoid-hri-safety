"""JSON the validator reads and cannot parse is a finding, not a crash.

The validator reads JSON text in three kinds of place: the meta files, the episode table's JSON-string
fields (``json_encoded_fields.episode`` of the shipped schema) and the step table's
(``json_encoded_fields.step``). Python's json module refuses more than a syntax error: an integer
longer than the interpreter's digit limit (ValueError), nesting deeper than the decoder's recursion
limit (RecursionError) and, in a file, bytes that are not UTF-8 (UnicodeDecodeError). A reader that
catches JSONDecodeError alone ends the validator with exit 3 -- "could not check" -- on each.

Each such case is held to the report a truncated document gets, the syntax error every reader
already handled: the same findings, the parse error's own text aside. A parquet string is UTF-8, so
the table cells have no undecodable-bytes case.

A step table's JSON cell is a list (schema §6). One that does not read as one, and one whose
elements are not strings, is an ERROR naming the field -- not a list read as empty, and not a crash
on an element a set cannot hold. The episode table's list columns are native lists, which the
validator also reads when they arrive as JSON text; there too an element that is not a string is a
finding.

JSON that parses can still hold what a reader does not expect: a number where a list goes, a list
where an object goes, a list of lists, nesting a recursive walk cannot follow, a run of digits longer
than int() reads, a lone surrogate the terminal and the report file cannot encode. Each case below
is one a reader that does not expect it dies on (exit 3), or writes no report for while exiting 1 --
"ERROR found". It is held to saying something: an ERROR where the release is wrong, and its report written.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from validator import make_fixture as F
from validator import validate_dataset as V

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((REPO_ROOT / "meta" / "safety_schema.json").read_text(encoding="utf-8"))
# The episode JSON columns. The table scenario's (scenario_conditional_episode_fields) are carried by the hri
# fixture, whose scenario declares a conditions table, and not by clean: a cell of one is edited there.
TABLE_FIELDS = {f for fields in SCHEMA["scenario_conditional_episode_fields"].values() for f in fields}
EPISODE_JSON = SCHEMA["json_encoded_fields"]["episode"]
STEP_JSON = SCHEMA["json_encoded_fields"]["step"]

# Every meta JSON file a fixture ships, on the fixture that ships it -- plus meta/stats.json, which
# a release ships and the fixture does not (the case writes one).
META_FILES = [("clean", "meta/info.json"), ("clean", "meta/rules.json"), ("clean", "meta/scenarios.json"),
              ("clean", "meta/safety_schema.json"), ("clean", "meta/assets.json"), ("clean", "meta/stats.json"),
              (F.MEDIA_CLEAN, "meta/cameras.json"), (F.MEDIA_PACK_CLEAN, "meta/packs.json")]
NOT_SHIPPED_BY_THE_FIXTURE = {"meta/stats.json": "{}"}

# How a step cell that is not a list is said: region_names by C3, which reads it first; the other
# two by C8, which reads their values.
NOT_A_LIST = {"observation.contact.region_names": ("C3", "region_names 가 리스트 아님"),
              "observation.contact.type": ("C8", "observation.contact.type 가 JSON 배열로 읽히지 않음"),
              "safety.event": ("C8", "safety.event 가 JSON 배열로 읽히지 않음")}
# ... and one whose elements are not strings.
NOT_STRINGS = {"observation.contact.region_names": ("C3", "허용되지 않은 부위 코드"),
               "observation.contact.type": ("C8", "허용되지 않은 접촉 유형"),
               "safety.event": ("C8", "허용되지 않은 safety.event")}

NESTED = "[" * 100_000 + "]" * 100_000          # past the decoder's recursion limit
DATA_0 = "data/chunk-000/file-000.parquet"
EPISODES_0 = "meta/episodes/chunk-000/file-000.parquet"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The clean fixtures whose meta files differ, built once: camera-less, camera, camera with a pack, and the hri
    distribution with cameras."""
    root = tmp_path_factory.mktemp("fixtures")
    F.build(root / "clean")
    F.build(root / F.MEDIA_CLEAN, with_media=True)
    F.build(root / F.MEDIA_PACK_CLEAN, with_media=True, with_pack=True)
    F.build(root / F.HRI_MEDIA_CLEAN, with_media=True, hri=True)
    return root


def _past_the_digit_limit() -> str:
    limit = sys.get_int_max_str_digits()
    if not limit:
        pytest.skip("this interpreter converts integers of any length (PYTHONINTMAXSTRDIGITS=0)")
    return "9" * (limit + 1)


def _with_member(text: str, value: str) -> str:
    """The JSON document with ``value`` (JSON text) as one more member or element, put first."""
    text = text.strip()
    head, body = text[0], text[1:].lstrip()
    if head == "{":
        return '{"_extra": ' + value + ("}" if body.startswith("}") else ", " + body)
    if head == "[":
        return "[" + value + ("]" if body.startswith("]") else ", " + body)
    raise AssertionError(f"not an object or an array: {text[:40]!r}")


def _truncated(text: str) -> str:
    """The syntax error: the document without its last character."""
    return text.strip()[:-1]


def _truncated_bytes(text: str) -> bytes:
    return _truncated(text).encode("utf-8")


# The documents the json module will not read. Each returns the file's bytes.
UNREADABLE = {
    "an integer past the digit limit": lambda t: _with_member(t, _past_the_digit_limit()).encode("utf-8"),
    "nesting past the recursion limit": lambda t: _with_member(t, NESTED).encode("utf-8"),
    "bytes that are not UTF-8": lambda t: _with_member(t, '"@@"').encode("utf-8").replace(b"@@", b"\xff", 1),
}
IN_A_CELL = ["an integer past the digit limit", "nesting past the recursion limit"]


@pytest.fixture(scope="module")
def controls(built, tmp_path_factory):
    """The report of each truncated control, made once however many kinds are held to it."""
    cache: dict[tuple, V.Report] = {}

    def report(case, *args) -> V.Report:
        if args not in cache:
            cache[args] = case(built, tmp_path_factory.mktemp("control"), *args)
        return cache[args]
    return report


def _copy(built: Path, base: str, tmp_path: Path) -> Path:
    root = tmp_path / base
    shutil.copytree(built / base, root)
    return root


def _report(root: Path) -> V.Report:
    try:
        return V.run(root, set(V.ALL_CHECKS), None, 0, None)
    except Exception as e:  # noqa: BLE001 -- what main() turns into exit 3
        pytest.fail(f"the validator died (exit 3): {type(e).__name__}: {e}")


def _said(rep: V.Report) -> list[tuple[str, str, str, str]]:
    """The findings, each parse error's own text cut off: what the reader raised differs from
    case to case, what the report does with it should not."""
    out = []
    for f in rep.findings:
        message = f.message
        for marker in ("JSON 파싱 실패", " 가 JSON 으로 읽히지 않음"):
            if marker in message:
                message = message[:message.index(marker) + len(marker)]
        out.append((f.check, f.severity, message, f.location))
    return out


def _meta_case(built: Path, tmp_path: Path, base: str, rel: str, make) -> V.Report:
    root = _copy(built, base, tmp_path)
    path = root / rel
    text = path.read_text(encoding="utf-8") if path.exists() else NOT_SHIPPED_BY_THE_FIXTURE[rel]
    path.write_bytes(make(text))
    return _report(root)


def test_every_meta_json_the_fixtures_ship_is_a_case(built):
    shipped = {str(p.relative_to(built / base))
               for base in ("clean", F.MEDIA_CLEAN, F.MEDIA_PACK_CLEAN, F.HRI_MEDIA_CLEAN)
               for p in (built / base / "meta").glob("*.json")}
    cased = {rel for _, rel in META_FILES}
    assert shipped <= cased, sorted(shipped - cased)
    assert cased - shipped == set(NOT_SHIPPED_BY_THE_FIXTURE)


@pytest.mark.parametrize("base, rel", META_FILES, ids=[rel for _, rel in META_FILES])
@pytest.mark.parametrize("kind", list(UNREADABLE))
def test_an_unreadable_meta_file_is_reported_as_a_truncated_one_is(built, controls, tmp_path, base, rel, kind):
    control = controls(_meta_case, base, rel, _truncated_bytes)
    parse = [f for f in control.findings if "JSON 파싱 실패" in f.message]
    assert parse and all(f.check == "C1" and f.severity == V.ERROR for f in parse), \
        [f.line() for f in control.findings]
    case = _meta_case(built, tmp_path, base, rel, UNREADABLE[kind])
    assert _said(case) == _said(control)


def _edit_cell(rel: str, column: str, make):
    """Row 0 of this table's ``column``: its JSON text through ``make``."""
    def mutate(root: Path) -> None:
        df = pd.read_parquet(root / rel)
        value = df.at[0, column]
        assert isinstance(value, str), (column, type(value))
        df.at[0, column] = make(value)
        if rel.startswith("data/"):
            F.write_steps(df, root / rel)      # keeps the NaN a release carries; to_parquet writes null
        else:
            df.to_parquet(root / rel, index=False)
    return mutate


def _cell_case(built: Path, tmp_path: Path, rel: str, column: str, make) -> V.Report:
    root = _copy(built, F.HRI_MEDIA_CLEAN if column in TABLE_FIELDS else "clean", tmp_path)
    _edit_cell(rel, column, make)(root)
    return _report(root)


@pytest.mark.parametrize("column", EPISODE_JSON)
@pytest.mark.parametrize("kind", IN_A_CELL)
def test_an_unreadable_episode_json_cell_is_reported_as_a_truncated_one_is(built, controls, tmp_path, column, kind):
    control = controls(_cell_case, EPISODES_0, column, _truncated)
    assert any(f.check == "C5" and f.severity == V.ERROR and f.message.startswith(f"{column} 가 JSON 으로 읽히지 않음")
               for f in control.findings), [f.line() for f in control.findings]
    case = _cell_case(built, tmp_path, EPISODES_0, column, lambda t: UNREADABLE[kind](t).decode("utf-8"))
    assert _said(case) == _said(control)


def test_every_step_json_field_has_its_sentences():
    assert set(NOT_A_LIST) == set(STEP_JSON) == set(NOT_STRINGS)


@pytest.mark.parametrize("column", STEP_JSON)
@pytest.mark.parametrize("kind", IN_A_CELL)
def test_an_unreadable_step_json_cell_is_reported_as_a_truncated_one_is(built, controls, tmp_path, column, kind):
    control = controls(_cell_case, DATA_0, column, _truncated)
    case = _cell_case(built, tmp_path, DATA_0, column, lambda t: UNREADABLE[kind](t).decode("utf-8"))
    assert _said(case) == _said(control)


@pytest.mark.parametrize("column", STEP_JSON)
@pytest.mark.parametrize("value", ["truncated", "abc", '{"a": 1}', "5", '"BR-01"'])
def test_a_step_json_cell_that_is_not_a_list_is_an_error_naming_the_field(built, tmp_path, column, value):
    make = _truncated if value == "truncated" else (lambda _t, value=value: value)
    rep = _cell_case(built, tmp_path, DATA_0, column, make)
    check, sentence = NOT_A_LIST[column]
    assert [f.location for f in rep.findings
            if (f.check, f.severity, f.message) == (check, V.ERROR, sentence)] == [f"{DATA_0}#0"], \
        [f.line() for f in rep.findings]


@pytest.mark.parametrize("column", [c for c in STEP_JSON if NOT_A_LIST[c][0] == "C8"])
def test_a_null_step_json_cell_is_left_to_the_null_check(built, tmp_path, column):
    """A null is a missing value, said by C7 -- not a cell that does not read as a list."""
    root = _copy(built, "clean", tmp_path)
    table = pq.read_table(root / DATA_0)              # through Arrow: the null is the only change
    cells = table.column(column).to_pylist()
    table = table.set_column(table.schema.get_field_index(column), column,
                             pa.array([None] + cells[1:], table.schema.field(column).type))
    pq.write_table(table, root / DATA_0)
    rep = _report(root)
    assert [f.message for f in rep.findings if f.check == "C7" and f.message.startswith(f"{column} 에 null 1개")]
    assert not [f for f in rep.findings if f.message == NOT_A_LIST[column][1]], [f.line() for f in rep.findings]


@pytest.mark.parametrize("column", STEP_JSON)
@pytest.mark.parametrize("value", ["[[1]]", '[{"a": 1}]', "[1]", '[1, "x"]', "[null]"])
def test_a_step_json_list_of_what_is_not_a_string_is_an_error_naming_the_field(built, tmp_path, column, value):
    rep = _cell_case(built, tmp_path, DATA_0, column, lambda _t: value)
    check, sentence = NOT_STRINGS[column]
    assert [f for f in rep.findings
            if f.check == check and f.severity == V.ERROR and f.message.startswith(sentence)], \
        [f.line() for f in rep.findings]


def test_the_step_json_cells_of_the_clean_fixture_say_nothing(built):
    """The sentences above come from the defect: the fixture they are planted in has none."""
    said = {(f.check, f.message) for f in _report(built / "clean").findings if f.severity == V.ERROR}
    assert said == set()


# The episode table's native list columns, as the validator reads them when they arrive as JSON text:
# what each says of a list holding what is not a string.
EPISODE_LISTS = {"joint_names": ("C2", "joint_names 에 문자열 아닌 원소"),
                 "triggered_rules": ("C4", "triggered_rules 에 문자열 아닌 원소")}


@pytest.mark.parametrize("column", list(EPISODE_LISTS))
@pytest.mark.parametrize("value", ["[[1]]", '[{"a": 1}]', '[1, "x"]', "[null]"])
def test_an_episode_list_of_what_is_not_a_string_is_an_error_naming_the_field(built, tmp_path, column, value):
    root = _copy(built, "clean", tmp_path)
    df = pd.read_parquet(root / EPISODES_0)
    df[column] = [value if i == 0 else json.dumps([str(x) for x in v]) for i, v in enumerate(df[column])]
    df.to_parquet(root / EPISODES_0, index=False)
    check, sentence = EPISODE_LISTS[column]
    rep = _report(root)
    assert [f for f in rep.findings
            if f.check == check and f.severity == V.ERROR and f.message.startswith(sentence)], \
        [f.line() for f in rep.findings]


def test_a_joint_name_that_is_not_a_string_is_said_where_its_text_is_the_maps_name(built, tmp_path):
    """The number 1 is not the joint "1": read as text it would match the map and say nothing."""
    root = _copy(built, "clean", tmp_path)
    mapping = next((root / "meta" / "joint_map").glob("*.yaml"))
    doc = yaml.safe_load(mapping.read_text(encoding="utf-8"))
    first = doc["joint_names"][0]
    doc["joint_names"][0] = "1"
    if first in (doc.get("canonical_map") or {}):
        doc["canonical_map"]["1"] = doc["canonical_map"].pop(first)
    doc["unmapped"] = ["1" if n == first else n for n in doc.get("unmapped") or []]
    mapping.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    df = pd.read_parquet(root / EPISODES_0)
    df["joint_names"] = [json.dumps([1] + list(doc["joint_names"][1:])) for _ in range(len(df))]
    df.to_parquet(root / EPISODES_0, index=False)
    rep = _report(root)
    assert [f for f in rep.findings if f.check == "C2" and f.severity == V.ERROR
            and f.message.startswith("joint_names 에 문자열 아닌 원소")], [f.line() for f in rep.findings]


def _step_column(root: Path, column: str, array) -> None:
    """One step file's ``column`` replaced through Arrow, the rest of the file as it was."""
    table = pq.read_table(root / DATA_0)
    table = table.set_column(table.schema.get_field_index(column), column, array)
    pq.write_table(table.replace_schema_metadata(None), root / DATA_0)


@pytest.mark.parametrize("column", STEP_JSON)
def test_a_step_json_column_of_numbers_is_not_a_column_of_nulls(built, tmp_path, column):
    """A NaN in a float column is a value, not a null: the null check does not see it (Arrow
    holds no null), so the cell has to be said as what it is -- no JSON list."""
    root = _copy(built, "clean", tmp_path)
    rows = pq.read_metadata(root / DATA_0).num_rows
    _step_column(root, column, pa.array([float("nan")] * rows, pa.float64(), from_pandas=False))
    rep = _report(root)
    check, sentence = NOT_A_LIST[column]
    assert [f.location for f in rep.findings
            if (f.check, f.severity, f.message) == (check, V.ERROR, sentence)] == [f"{DATA_0}#0"], \
        [f.line() for f in rep.findings]


def _cli(root: Path, report: Path, *args: str) -> subprocess.CompletedProcess:
    """The validator as a command, its output encoded as a terminal's would be."""
    return subprocess.run([sys.executable, "-B", str(REPO_ROOT / "validator" / "validate_dataset.py"), str(root),
                           "--json", str(report), *args],
                          capture_output=True, env={**os.environ, "PYTHONPATH": "", "PYTHONIOENCODING": "utf-8",
                                                    "PYTHONDONTWRITEBYTECODE": "1"})


def _surrogate_in_a_region_code(root: Path) -> None:
    table = pq.read_table(root / DATA_0)
    cells = table.column("observation.contact.region_names").to_pylist()
    cells[0] = cells[0].replace('"BR-01"', '"\\ud800"', 1)
    _step_column(root, "observation.contact.region_names", pa.array(cells, pa.string()))


def _surrogate_in_info(root: Path) -> None:
    doc = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    doc["schema_version"] = "@@"
    (root / "meta" / "info.json").write_text(json.dumps(doc).replace('"@@"', '"\\ud800"'), encoding="utf-8")


@pytest.mark.parametrize("plant", [_surrogate_in_a_region_code, _surrogate_in_info],
                         ids=["a region code", "info.json schema_version"])
def test_a_lone_surrogate_from_the_release_reaches_the_terminal_and_the_report(built, tmp_path, plant):
    """JSON may escape a lone surrogate ("\\ud800"); UTF-8 cannot encode one. A finding quoting
    it is printed escaped and written to the report as the JSON escape it came as."""
    root = _copy(built, "clean", tmp_path)
    plant(root)
    report = tmp_path / "report.json"
    done = _cli(root, report)
    assert report.is_file(), done.stderr.decode("utf-8", "replace")[-400:]
    findings = json.loads(report.read_text(encoding="utf-8"))["findings"]
    # Quoted as the character (decoded from the report's escape) or as its repr inside a list.
    quoted = [f for f in findings if any(form in f["message"] + f["location"] for form in ("\ud800", "\\ud800"))]
    assert quoted, findings
    errors = [f for f in findings if f["severity"] == V.ERROR]
    assert done.returncode == (1 if errors else 0), done.stderr.decode("utf-8", "replace")[-400:]
    assert b"\\ud800" in done.stdout


def test_a_report_that_cannot_be_written_is_exit_3(built, tmp_path):
    """Exit 1 says the release has an ERROR. A validator that could not write its report did not
    say that -- it broke, which is exit 3."""
    in_the_way = tmp_path / "a-file"
    in_the_way.write_text("", encoding="utf-8")
    done = _cli(built / "clean", in_the_way / "report.json")
    assert done.returncode == 3, (done.returncode, done.stderr.decode("utf-8", "replace")[-400:])


def test_statistics_nested_deeper_than_a_recursive_walk_are_still_read(built, tmp_path):
    """safety.stability_margin carries -inf in the clean fixture, so its statistics have to be
    empty; a mean of 0 under 600 single-element lists is a value all the same."""
    root = _copy(built, "clean", tmp_path)
    (root / "meta" / "stats.json").write_text(
        '{"safety.stability_margin": {"mean": ' + "[" * 600 + "0" + "]" * 600 + "}}", encoding="utf-8")
    rep = _report(root)
    assert [f for f in rep.findings if f.check == "C8" and f.severity == V.ERROR
            and f.message.startswith("safety.stability_margin 은 비유한값을 담는데 정규화 통계 ['mean']")], \
        [f.line() for f in rep.findings]


def test_a_pack_key_longer_than_int_reads_is_no_episode_index(built, tmp_path):
    root = _copy(built, F.MEDIA_PACK_CLEAN, tmp_path)
    key = _past_the_digit_limit()
    packs = json.loads((root / "meta" / "packs.json").read_text(encoding="utf-8"))
    packs[key] = {}
    (root / "meta" / "packs.json").write_text(json.dumps(packs), encoding="utf-8")
    rep = _report(root)
    assert [f for f in rep.findings if f.check == "C1" and f.severity == V.ERROR
            and f.message == f"키 {key!r} 가 에피소드 번호(episode_index)가 아님"], [f.line() for f in rep.findings]


def _episode_id(root: Path, part: str, digits: str) -> str:
    df = pd.read_parquet(root / EPISODES_0)
    m = V.EPISODE_ID_RE.match(df.at[0, "episode_id"])
    eid = df.at[0, "episode_id"].replace(m.group(part), digits, 1) if part == "seq" else \
        df.at[0, "episode_id"][:m.start("seed")] + digits
    df.at[0, "episode_id"] = eid
    df.to_parquet(root / EPISODES_0, index=False)
    return eid


def test_an_episode_seed_longer_than_int_reads_is_not_the_episodes_seed(built, tmp_path):
    root = _copy(built, "clean", tmp_path)
    eid = _episode_id(root, "seed", _past_the_digit_limit())
    rep = _report(root)
    assert [f for f in rep.findings if f.check == "C5" and f.severity == V.ERROR and f.location == eid
            and f.message.startswith("episode_id 의 시드")], [f.line() for f in rep.findings][:20]


@pytest.mark.parametrize("checks", [{"C5"}, {"C9"}], ids=["C5", "C9"])
def test_an_episode_number_longer_than_int_reads_ends_no_check(built, tmp_path, checks):
    """A number that long is no run's start nor a sample's place; it is not one the checks can use,
    and not one they die on (C9 reads it before calling the hook, which never runs here)."""
    root = _copy(built, "clean", tmp_path)
    _episode_id(root, "seq", _past_the_digit_limit())
    try:
        V.run(root, checks, str(tmp_path / "no-such-hook"), 1, None)
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"the validator died (exit 3): {type(e).__name__}: {e}")


NOTES_OF_THE_WRONG_SHAPE = {
    "contact_onset_unresolved_regions": (lambda n: n.__setitem__("contact_onset_unresolved_regions", 1),
                                         "notes.contact_onset_unresolved_regions 이 array 이 아님"),
    "not_evaluated": (lambda n: n.__setitem__("not_evaluated", 1), "notes.not_evaluated 이 array 이 아님"),
    "not_evaluated[].regions": (
        lambda n: n.__setitem__("not_evaluated", [{"rule_id": "R-x", "reason": "r", "regions": 1}]),
        "notes.not_evaluated[0].regions 이 array 이 아님"),
}


@pytest.mark.parametrize("member", list(NOTES_OF_THE_WRONG_SHAPE))
def test_a_notes_member_of_the_wrong_shape_is_said_and_not_walked(built, tmp_path, member):
    change, sentence = NOTES_OF_THE_WRONG_SHAPE[member]
    root = _copy(built, "clean", tmp_path)
    df = pd.read_parquet(root / EPISODES_0)
    notes = json.loads(df.at[0, "notes"])
    change(notes)
    df.at[0, "notes"] = json.dumps(notes)
    df.to_parquet(root / EPISODES_0, index=False)
    rep = _report(root)
    assert [f for f in rep.findings if f.check == "C5" and f.severity == V.ERROR and sentence in f.message], \
        [f.line() for f in rep.findings]


def _set(rel: str, path: tuple, value):
    def change(root: Path) -> None:
        doc = json.loads((root / rel).read_text(encoding="utf-8"))
        here = doc
        for key in path[:-1]:
            here = here[key]
        here[path[-1]] = value
        (root / rel).write_text(json.dumps(doc), encoding="utf-8")
    return change


def _first_usd_path(group: str, value):
    """The first asset of the shipped canon's ``group`` naming its USD by ``value``."""
    def change(root: Path) -> None:
        doc = json.loads((root / "meta" / "assets.json").read_text(encoding="utf-8"))
        next(iter(doc[group].values()))["usd_path"] = value
        (root / "meta" / "assets.json").write_text(json.dumps(doc), encoding="utf-8")
    return change


# One per place a reader that does not expect the shape dies (measured by putting each of a few shapes in
# place of every member of the meta files, and of every member a level down).
CANON_OF_THE_WRONG_SHAPE = {
    "contact_types holds a number": _set("meta/safety_schema.json", ("contact_types",), [1, "x"]),
    "an enum holds a list": _set("meta/safety_schema.json", ("enums", "domain"), [[]]),
    "a condition holds a list": _set("meta/safety_schema.json", ("conditional_required_episode_fields",),
                                     {"a": [[]]}),
    "json_encoded_fields is a number": _set("meta/safety_schema.json", ("json_encoded_fields",), 5),
    "episode_json_shapes is a list": _set("meta/safety_schema.json", ("episode_json_shapes",), [1, "x"]),
    "notes_members is a number": _set("meta/safety_schema.json", ("notes_members",), 5),
    "the assets' scene is a number": _set("meta/assets.json", ("scene",), 5),
    "the assets' robots are a number": _set("meta/assets.json", ("robots",), 5),
    "the rules are a number": _set("meta/rules.json", ("rules",), 5),
    "a declared field shape is a list": _set("meta/safety_schema.json", ("episode_json_shapes", "termination"), [[]]),
    "a declared notes shape is a number": _set("meta/safety_schema.json", ("notes_members", "contact_observed"), 5),
    "a robot's usd_path is a list": _first_usd_path("robots", [[]]),
    "a scene asset's usd_path is a number": _first_usd_path("scene", 5),
}


@pytest.mark.parametrize("inputs", [5, {"a": [[]]}, [1, "x"]], ids=["a number", "an object", "a list holding a number"])
def test_a_rules_inputs_that_name_no_fields_are_said_not_read_as_none(built, tmp_path, inputs):
    """A rule's inputs are the step fields the data is held to (C8). Read as no inputs, a number
    or an object there would hold the data to nothing and say nothing."""
    root = _copy(built, "clean", tmp_path)
    rules = json.loads((root / "meta" / "rules.json").read_text(encoding="utf-8"))
    rule = next(r for r in rules["rules"] if r.get("inputs"))
    rule["inputs"] = inputs
    (root / "meta" / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
    rep = _report(root)
    assert [f for f in rep.findings if (f.check, f.severity, f.location) == ("C8", V.ERROR, "meta/rules.json")
            and f.message.startswith(f"규칙 {rule['rule_id']} 의 inputs 가 필드 이름의 목록이 아님")], \
        [f.line() for f in rep.findings]


@pytest.mark.parametrize("case", list(CANON_OF_THE_WRONG_SHAPE))
def test_a_canon_of_the_wrong_shape_is_an_error_not_a_crash(built, tmp_path, case):
    root = _copy(built, "clean", tmp_path)
    CANON_OF_THE_WRONG_SHAPE[case](root)
    rep = _report(root)
    assert rep.count(V.ERROR), [f.line() for f in rep.findings]


def test_a_shipped_schema_that_is_null_is_said_to_be_empty(built, tmp_path):
    """A file holding JSON null is there and declares nothing; read like an absent one it would
    pass with no ERROR (the absent file's WARN looks at the path, which exists)."""
    root = _copy(built, "clean", tmp_path)
    (root / "meta" / "safety_schema.json").write_text("null", encoding="utf-8")
    rep = _report(root)
    assert [f for f in rep.findings if (f.check, f.severity, f.message, f.location)
            == ("C1", V.ERROR, "규격 정의가 비어 있음", "meta/safety_schema.json")], [f.line() for f in rep.findings]
