"""스텝 표가 `info.json` 의 `features` 가 선언한 표인가 (C7, `c7_step_features`).

LeRobot 의 적재기는 스텝 표의 형을 파일에서 읽지 않는다. `info.json` 의 `features` 로 열과 형을 정하고
스텝 파일을 거기에 맞춰 넣는다(lerobot 0.6.1 `feature_utils.get_hf_features_from_features` · datasets 의
`cast_table_to_schema`):

  - 영상이 아닌 feature 는 모두 스텝 표의 열이다. 파일에 선언 없는 열이 있으면 여는 단계에서 멈춘다
  - 선언만 있고 파일에 없는 열은 빈 값(null)으로 채운다 — 멈추지 않는다
  - 저장된 형이 선언과 다르면 선언한 형으로 바꾸고, 바뀌지 않으면 여는 단계에서 멈춘다

검증기의 세 규칙(`_feature_column` — 선언을 적재기가 무엇으로 만드는가 · `_cast_stops` — 저장된 열이 선언한
형으로 바뀌는가 · `_null_lists_misread` — null 인 목록 뒤의 배치를 적재기가 어디서 읽는가)은 적재기의 동작을
옮긴 것이다. 옮긴 것이 맞는지는 읽어서가 아니라 **재서** 본다 —
`scripts/measure_loader_casts.py` 가 `feature_casts.py` 의 사례를 적재기에 넣어 잰 답
(`feature_casts_measured.json`)과 사례마다 대조한다. 그 아래의 사례는 손으로 지은 몇 행짜리 배포본이고,
픽스처 쪽은 맨 아래의 시험과 `test_fixture_detection_reason.py` 의 `broken_feature*` 가 본다.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import validator.validate_dataset as vd
from tests.validator.feature_casts import (BATCH_ROWS, BATCHES, DECLARED, FORMS, MEASURED, STORED, batch_column,
                                            batch_values)
from validator.make_fixture import VARIANTS, build

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "validator" / "README.md"
LOADED = "*/*.parquet"          # what LeRobot loads of data/ (io_utils.load_nested_dataset)
STEPS = "data/chunk-000/file-{:03d}.parquet"
INFO = "meta/info.json"

# One sentence for each kind of defect; the numbers are in the finding's location.
NO_FEATURES = "info.json 에 features 가 없거나 JSON 객체가 아님"
UNMADE = "info.json 의 features 에 LeRobot 이 스텝 표의 열로 만들지 못하는 선언이 있음"
UNDECLARED = "스텝 표에 info.json 의 features 가 스텝 표의 열로 선언하지 않은 열이 있음"
NO_COLUMN = "info.json 의 features 가 선언한 열이 스텝 표에 없음"
NO_CAST = "스텝 표의 열을 info.json 의 features 가 선언한 형으로 바꿀 수 없음"
MISREAD = "스텝 표의 가변 길이 목록 열에 null 인 목록이 있어 LeRobot 이 통째로 열 때 그 뒤의 스텝을 잘못 읽음"
SAID = (NO_FEATURES, UNMADE, UNDECLARED, NO_COLUMN, NO_CAST, MISREAD)
NOT_COMPARED = "대조하지 못함"
NOT_HELD = "대 보지 못함"

measured = json.loads(MEASURED.read_text(encoding="utf-8"))
FORM_STOPS = set(measured["forms"]["stops"])
CAST_STOPS = {tuple(pair) for pair in measured["casts"]["stops"]}
PLACE = vd.STEP_PLACE_COLUMNS + (vd.TASK_INDEX,)
# The validator casts with the pyarrow that is here; the measurement was taken with the loader's.
SAME_PYARROW = pa.__version__ == measured["measured_with"]["pyarrow"]


def _plain(kind: pa.DataType) -> bool:
    """A type whose casts every pyarrow does alike -- the ones a release's columns have: an
    integer, a float32 or a float64, a bool, a string, null, and a list of those. A struct too:
    nothing of it is cast, it goes into no column that is declared here."""
    t = pa.types
    if t.is_list(kind) or t.is_large_list(kind) or t.is_fixed_size_list(kind):
        return _plain(kind.value_type)
    return (t.is_integer(kind) or t.is_float32(kind) or t.is_float64(kind) or t.is_boolean(kind)
            or t.is_string(kind) or t.is_null(kind) or t.is_struct(kind))


def _leaf(declared: str) -> pa.DataType:
    kind, detail = vd._feature_column(DECLARED[declared])
    return detail if kind == "value" else detail[0]


PLAIN_STORED = [label for label, values in STORED.items() if _plain(values.type)]
PLAIN_DECLARED = [label for label in DECLARED if _plain(_leaf(label))]


# ── 규칙은 잰 것과 같은가 ────────────────────────────────────────

def test_the_measurement_is_of_the_cases_listed_here():
    """측정 파일이 다른 사례 목록의 것이면 아래 대조는 아무것도 말하지 못한다. 두 측정 방법(메모리에서의
    캐스트 · 실제 판독기)의 답이 갈린 사례가 있어도 그렇다."""
    assert measured["forms"]["all"] == list(FORMS)
    assert measured["casts"]["stored"] == list(STORED) and measured["casts"]["declared"] == list(DECLARED)
    assert FORM_STOPS <= set(FORMS) and CAST_STOPS <= {(s, d) for s in STORED for d in DECLARED}
    assert measured["batches"]["all"] == list(BATCHES)
    assert set(measured["batches"]["stops"]) | set(measured["batches"]["shifts"]) < set(BATCHES)
    assert measured["reader_and_cast_disagree"] == []
    # Both answers are in the file: a measurement in which nothing stopped, or everything, measured nothing.
    assert 0 < len(FORM_STOPS) < len(FORMS) and 0 < len(CAST_STOPS) < len(STORED) * len(DECLARED)


def _judged() -> dict[str, tuple[str, object]]:
    return {label: vd._feature_column(form) for label, form in FORMS.items()}


def test_no_declaration_the_loader_makes_a_table_of_is_said_to_stop_it():
    """거짓 거부가 없어야 한다 — 「멈춘다」고 말한 선언은 전부 적재기가 실제로 멈춘 것이다."""
    said = {label for label, (kind, _) in _judged().items() if kind == "stops"}
    assert said and said <= FORM_STOPS, sorted(said - FORM_STOPS)


def test_every_declaration_given_a_type_is_one_the_loader_makes_a_table_of():
    """형을 준 선언(값 하나 · 길이 n 의 목록)은 전부 적재기가 표를 만든 것이다 — 멈추는 선언에 형을 주면
    그 열의 값을 「바뀐다」고 통과시키게 된다."""
    typed = {label for label, (kind, _) in _judged().items() if kind in ("value", "list")}
    assert typed and not typed & FORM_STOPS, sorted(typed & FORM_STOPS)


def test_what_is_left_unjudged_is_named_and_counted():
    """판단하지 않는 선언(`unread`)은 어느 쪽으로도 말하지 않는다. 그 가운데 적재기가 멈추는 것이 이 검사가
    놓치는 것이다 — README 가 그 수를 적으므로 여기서 세어 문서와 맞춘다."""
    judged = _judged()
    unread = {label for label, (kind, _) in judged.items() if kind == "unread"}
    said = {label for label, (kind, _) in judged.items() if kind == "stops"}
    missed = unread & FORM_STOPS
    assert not {kind for kind, _ in judged.values()} - {"stops", "value", "list", "unread"}
    assert said | missed == FORM_STOPS
    text = README.read_text(encoding="utf-8")
    sentence = (f"선언의 꼴 {len(FORMS)}가지 가운데 적재기가 멈춘 것은 {len(FORM_STOPS)}가지이고, 검사는 그중 "
                f"{len(said)}가지를 말하고 {len(missed)}가지를 놓칩니다")
    assert sentence in " ".join(text.split()), sentence


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "bool", "string"])
@pytest.mark.parametrize("length", [1, 2, 19])
def test_the_declarations_a_release_makes_are_all_judged(dtype, length):
    """변환기가 선언하는 것은 numpy 의 dtype 이름과 `string`, 모양은 한 차원이다. 그 꼴이 판단 밖이면 검사는
    실제 배포본에서 아무것도 보지 않는다."""
    kind, detail = vd._feature_column({"dtype": dtype, "shape": [length], "names": None})
    assert kind == ("value" if length == 1 else "list")
    leaf = detail if length == 1 else detail[0]
    assert leaf == getattr(pa, "bool_" if dtype == "bool" else dtype)()
    assert length == 1 or detail[1] == length


def test_datasets_own_two_names_are_the_types_it_makes_of_them():
    """`double` 은 float64, `float` 은 float32 다(`datasets.Value`). 형이 주어졌다는 것만 보면 둘을 서로 바꿔
    적어도 통과한다 — 그러면 저장된 열이 「선언한 형 그대로」인지를 틀리게 답한다."""
    for name, kind in (("double", pa.float64()), ("float", pa.float32())):
        assert vd._feature_column({"dtype": name, "shape": [1], "names": None}) == ("value", kind)
        assert vd._feature_column({"dtype": name, "shape": [3], "names": None}) == ("list", (kind, 3))
    assert f"dtype 'double', shape [1]" not in FORM_STOPS and f"dtype 'float', shape [1]" not in FORM_STOPS


def test_json_is_refused_whether_or_not_pyarrow_has_loaded_its_json_reader():
    """`dtype` 이 `json` 이면 datasets 가 이름을 보기 전에 거부한다. pyarrow 의 `json` 은 형이 아니라 하위
    모듈이고, 불러온 프로세스에서만 `pyarrow.json` 으로 보인다 — 답이 그것에 달려 있으면 안 된다."""
    form = {"dtype": "json", "shape": [1], "names": None}
    assert "dtype 'json', shape [1]" in FORM_STOPS
    before = vd._feature_column(form)
    import pyarrow.json  # noqa: F401 -- from here on pyarrow has an attribute of that name

    assert hasattr(pa, "json") and vd._feature_column(form) == before and before[0] == "stops"


def _off_the_measurement(stored_labels, declared_labels) -> list[tuple]:
    """The pairs on which the rule and the loader differ: (stored, declared, what the rule said)."""
    wrong = []
    for declared in declared_labels:
        kind, detail = vd._feature_column(DECLARED[declared])
        assert kind in ("value", "list"), (declared, kind)
        for stored in stored_labels:
            why = vd._cast_stops(STORED[stored], kind, detail)
            if (why is not None) != ((stored, declared) in CAST_STOPS):
                wrong.append((stored, declared, why))
    return wrong


def test_the_cast_rule_says_what_the_loader_was_measured_to_do_on_the_types_a_release_holds():
    """배포본의 열이 갖는 형(정수 · 실수 · 참거짓 · 문자열과 그 목록)의 모든 쌍에서 — 적재기가 읽은 것을
    멈춘다고 하지 않고, 멈춘 것을 놓치지 않는다. 이 형들 사이의 변환은 pyarrow 의 판마다 같아, 검증기를
    돌리는 쪽의 pyarrow 가 잰 판이 아니어도 대조한다."""
    # Most of what was measured, not a corner of it.
    assert 2 * len(PLAIN_STORED) > len(STORED) and 2 * len(PLAIN_DECLARED) > len(DECLARED)
    assert not _off_the_measurement(PLAIN_STORED, PLAIN_DECLARED)


@pytest.mark.skipif(not SAME_PYARROW, reason=(
    f"pyarrow {pa.__version__} 는 잰 판({measured['measured_with']['pyarrow']})이 아님 — 드문 형(timestamp · decimal · "
    "binary …)의 변환은 판에 따라 다를 수 있어, 그 쌍은 잰 판과 같은 pyarrow 에서만 대조한다"))
def test_the_cast_rule_says_what_the_loader_was_measured_to_do_on_every_pair():
    """저장된 열 × 선언의 모든 쌍에서 — 값의 변환은 적재기가 부르는 Arrow 의 캐스트를 그대로 부르므로,
    같은 pyarrow 에서는 한 쌍도 다르지 않아야 한다."""
    assert not _off_the_measurement(STORED, DECLARED)


def test_a_column_stored_as_declared_is_never_one_the_loader_stops_on():
    """저장된 형이 선언과 같으면 검사는 행을 읽지 않는다. 그 지름길이 놓치는 쌍이 있으면 안 된다."""
    same = [(stored, declared) for declared, form in DECLARED.items() for stored, values in STORED.items()
            if vd._stored_as_declared(values.type, *vd._feature_column(form))]
    assert same and not set(same) & CAST_STOPS, sorted(set(same) & CAST_STOPS)


@pytest.mark.parametrize("stored", ["list<double>, of 1", "fixed<double>[1]", "struct<x: double>"])
def test_a_list_or_a_struct_stored_where_one_value_is_declared_is_refused_before_arrow_is_asked(stored, monkeypatch):
    """datasets 는 목록 · struct 를 값 하나로 바꾸려 하지 않는다 — Arrow 에 묻기 전에 거부한다(길이 1 의 목록도).
    어느 pyarrow 가 그 변환을 해 주게 되더라도 답이 달라지지 않도록, 이 규칙도 Arrow 에 묻지 않는다 —
    물었는지를 본다(이유의 글만 보면, 묻고 나서 그 답을 버려도 통과한다)."""
    asked = []
    monkeypatch.setattr(vd, "_arrow_refuses", lambda values, into: asked.append((values.type, into)))
    for declared in ("float64 [1]", "string [1]"):
        assert (stored, declared) in CAST_STOPS
        why = vd._cast_stops(STORED[stored], *vd._feature_column(DECLARED[declared]))
        assert why is not None and "값 하나" in why, (declared, why)
    assert asked == []
    # The same watch does see a cast that is Arrow's to answer.
    assert vd._cast_stops(STORED["int64, one 128"], *vd._feature_column(DECLARED["int8 [1]"])) is None
    assert asked == [(pa.int64(), pa.int8())]


def test_the_cast_is_asked_of_each_chunk_of_a_column():
    kind, detail = vd._feature_column(DECLARED["int8 [1]"])
    whole = pa.chunked_array([STORED["int64"], STORED["int64, one 128"]])
    assert vd._cast_stops(pa.chunked_array([STORED["int64"]]), kind, detail) is None
    assert vd._cast_stops(whole, kind, detail) == _arrow_says(STORED["int64, one 128"], pa.int8())
    kind, detail = vd._feature_column(DECLARED["float64 [3]"])
    rows = pa.chunked_array([STORED["list<double>, of 3"], STORED["list<double>, of 3 and one of 2"]])
    # The row is counted through the chunks: the short list is the second chunk's second row.
    assert vd._cast_stops(rows, kind, detail) == "선언한 길이(3)와 다른 목록이 있음 (4번째 행의 길이 2 · 그런 행 1개)"


def test_an_extension_array_is_cast_as_what_it_stores():
    """Arrow 의 확장 형(고정 모양 텐서)은 밑에 고정 길이 목록을 싣는다. datasets 는 그 저장 형을 바꾸므로
    적재기는 읽는다 — 확장 형을 「목록이 아닌 것」으로 보면 읽히는 배포본을 거부한다."""
    stored = STORED["fixed_shape_tensor<double>[3]"]
    assert isinstance(stored.type, pa.BaseExtensionType)
    for declared, reads in (("float64 [3]", True), ("float32 [3]", True), ("float64 [2]", False),
                            ("float64 [1]", False)):
        assert (("fixed_shape_tensor<double>[3]", declared) not in CAST_STOPS) == reads
        assert (vd._cast_stops(stored, *vd._feature_column(DECLARED[declared])) is None) == reads, declared
    assert vd._stored_as_declared(stored.type, *vd._feature_column(DECLARED["float64 [3]"]))
    assert not vd._stored_as_declared(stored.type, *vd._feature_column(DECLARED["float32 [3]"]))
    # Where one value is declared it is refused as the list it stores, before Arrow is asked.
    assert "값 하나" in vd._cast_stops(stored, *vd._feature_column(DECLARED["float64 [1]"]))


# ── null 인 목록 뒤의 배치 ───────────────────────────────────────

def _nulls(steps: set[int]) -> np.ndarray:
    return np.isin(np.arange(BATCH_ROWS), sorted(steps))


@pytest.mark.parametrize("label", sorted(BATCHES))
def test_what_the_loader_does_after_a_null_list_is_what_was_measured(label):
    """가변 길이 목록을 고정 길이로 바꿀 때 적재기는 배치마다 바꾼다(배치 = 첫 행 그룹의 행 수). null 인 목록
    뒤에, 같은 행 그룹에서 null 없는 배치가 오면 그 배치를 밀린 자리에서 읽는다 — 멈추거나, 오류 없이 다른
    스텝의 값을 내준다. 자리마다 적재기에서 잰 것과 대조한다."""
    groups, nulls = BATCHES[label]
    stops, shifts = measured["batches"]["stops"], measured["batches"]["shifts"]
    said = vd._null_lists_misread(_nulls(nulls), groups)
    if label in stops:
        # The step named is the first that was not as stored by the time the loader stopped.
        assert said == ("stops", stops[label])
    elif label in shifts:
        # Every step holds values of its own there, so the first step read at the wrong place is
        # the first the loader was seen to hand out wrong -- by the rule and by the values.
        assert said == ("shifts", shifts[label])
        assert vd._first_step_handed_another(batch_column(nulls), _nulls(nulls), groups) == shifts[label]
    else:
        assert said is None


@pytest.mark.parametrize("label", sorted(BATCHES))
def test_where_every_list_holds_the_same_a_batch_read_at_the_wrong_place_shows_nothing(label):
    """밀린 자리에서 읽은 배치가 받는 것은 다른 스텝의 목록이다 — 그 목록이 제 것과 같은 값이면 스텝은 저장된
    대로 나온다. 모든 목록이 같은 값인 열로 같은 자리를 다시 쟀다: 멈추는 자리는 값과 무관하게 멈추고, 멈추지
    않는 자리는 전부 저장된 대로 읽혔다. 「잘못 읽는다」는 값이 실제로 달라질 때만 할 수 있는 말이다."""
    groups, nulls = BATCHES[label]
    alike = measured["batches"]["alike"]
    assert set(alike["stops"]) == set(measured["batches"]["stops"])        # a stop does not turn on the values
    said = vd._null_lists_misread(_nulls(nulls), groups)
    assert (said is not None and said[0] == "stops") == (label in alike["stops"])
    if said is not None and said[0] == "shifts":
        other = vd._first_step_handed_another(batch_column(nulls, alike=True), _nulls(nulls), groups)
        assert (other is not None) == (label in alike["not_as_stored"]) and other is None


def test_the_first_step_with_other_values_is_not_always_the_first_read_at_the_wrong_place():
    """4번째와 5번째 스텝이 같은 값을 갖는 열(행 그룹 2 · 8, null 은 2번째와 9번째): 4번째 스텝은 밀린
    자리에서 5번째의 목록을 받지만 값이 같고, 처음 달라지는 것은 6번째의 값을 받는 5번째다(실측)."""
    groups, nulls = BATCHES["row groups 2 8, nulls at 2 and 9"]
    values = batch_column(nulls).to_pylist()
    values[5] = values[4]
    column = pa.chunked_array([pa.array(values, pa.list_(pa.float64()))])
    assert vd._null_lists_misread(_nulls(nulls), groups) == ("shifts", 4)
    assert vd._first_step_handed_another(column, _nulls(nulls), groups) == 5
    # NaN where a NaN was is the same value handed out.
    nan = float("nan")
    values = [None if step in nulls else [nan, 1.0, nan] for step in range(BATCH_ROWS)]
    assert vd._first_step_handed_another(pa.array(values, pa.list_(pa.float64())), _nulls(nulls), groups) is None


def test_all_three_outcomes_were_measured():
    batches = measured["batches"]
    assert batches["stops"] and batches["shifts"] and len(batches["stops"]) + len(batches["shifts"]) < len(BATCHES)
    assert not set(batches["stops"]) & set(batches["shifts"])
    # A null is among the placings that read as stored too: it is not the null that is said.
    assert any(BATCHES[label][1] for label in BATCHES
               if label not in batches["stops"] and label not in batches["shifts"])


def test_opened_for_chosen_episodes_no_placing_is_read_wrong(tmp_path):
    """잘못 읽히는 것은 배포본을 통째로 열 때다. 에피소드를 골라 열면(`episodes=[...]`) 적재기는 배치를 걸러 낸
    뒤에 바꾸고, 걸러 낸 배치는 제 행만 가진 배열이라 밀릴 자리가 없다 — 잰 자리 모두에서, 에피소드를 전부
    골라도 저장된 대로 읽힌다. 문장이 그 범위를 말해야 한다(한쪽에서 잰 것을 모든 경우의 말로 적지 않는다)."""
    batches = measured["batches"]
    assert batches["not_as_stored_when_episodes_are_chosen"] == []
    # Each episode by itself and all of them: the last is every step of the file, read right.
    assert len(batches["episodes_chosen"]) > 1 and max(batches["episodes_chosen"], key=len) == [0, 1]
    assert "통째로 열 때" in MISREAD
    root = _release(tmp_path, _with_lists({2}))
    _in_row_groups(root / STEPS.format(0), [2, 8])
    (f,) = _said(_report(root), MISREAD)
    assert "에피소드를 골라 열면 바르게 읽는다" in f.message, f.message


def test_a_file_whose_first_row_group_is_empty_has_no_batch_to_read_wrong():
    assert vd._null_lists_misread(_nulls({1}), [0, BATCH_ROWS]) is None and vd._null_lists_misread(_nulls({1}), []) is None


# ── 손으로 지은 배포본 ───────────────────────────────────────────

def _feature(dtype, shape) -> dict:
    return {"dtype": dtype, "shape": shape, "names": None}


def _columns(rows: int = 3, start: int = 0) -> dict[str, pa.Array]:
    """A step file as LeRobot's writer stores one: every column the very type its feature makes."""
    return {
        "index": pa.array(range(start, start + rows), pa.int64()),
        "episode_index": pa.array([0] * rows, pa.int64()),
        "frame_index": pa.array(range(start, start + rows), pa.int64()),
        "task_index": pa.array([0] * rows, pa.int64()),
        "timestamp": pa.array([0.02 * k for k in range(rows)], pa.float32()),
        "observation.state": pa.array([[0.5, 1.5, 2.5]] * rows, pa.list_(pa.float32(), 3)),
        "safety.margin": pa.array([0.25] * rows, pa.float64()),
        "safety.event": pa.array(["[]"] * rows, pa.string()),
    }


FEATURES = {
    "index": _feature("int64", [1]), "episode_index": _feature("int64", [1]),
    "frame_index": _feature("int64", [1]), "task_index": _feature("int64", [1]),
    "timestamp": _feature("float32", [1]), "observation.state": _feature("float32", [3]),
    "safety.margin": _feature("float64", [1]), "safety.event": _feature("string", [1]),
}
ABSENT = object()


def _release(root: Path, *files: dict, features=FEATURES, **changed) -> Path:
    """Step files (two as LeRobot writes them, unless given) and an info.json declaring ``features``,
    a feature of ``changed`` replacing or -- ABSENT -- leaving out the one of that name."""
    start = 0
    for j, columns in enumerate(files or (_columns(3), _columns(2, 3))):
        path = root / STEPS.format(j)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), path)
        start += len(next(iter(columns.values()), []))
    info = {"codebase_version": "v3.0", "fps": 50}
    if features is not ABSENT:
        info["features"] = features if not isinstance(features, dict) else {
            name: declared for name, declared in {**features, **changed}.items() if declared is not ABSENT}
    (root / INFO).parent.mkdir(parents=True, exist_ok=True)
    (root / INFO).write_text(json.dumps(info), encoding="utf-8")
    return root


def _report(root: Path) -> vd.Report:
    rep = vd.Report(root=str(root))
    vd.c7_step_features(vd.Dataset(root, rep), rep)
    return rep


def _said(rep: vd.Report, *sentences: str) -> list[vd.Finding]:
    """The report is these sentences, one C7 ERROR each, and nothing else."""
    lines = [f.line() for f in rep.findings]
    assert all((f.check, f.severity) == ("C7", vd.ERROR) for f in rep.findings), lines
    kinds = [next((s for s in SAID if f.message.startswith(s)), f.message) for f in rep.findings]
    assert sorted(kinds) == sorted(sentences), lines
    return rep.findings


def _with(columns: dict, **changed) -> dict:
    return {name: values for name, values in {**columns, **changed}.items() if values is not ABSENT}


def _arrow_says(values: pa.Array, into: pa.DataType) -> str:
    """Arrow's own words for a cast it refuses, as a finding quotes them -- asked of the pyarrow
    that is here, so that the test holds whatever that pyarrow words it as."""
    with pytest.raises(Exception) as raised:
        values.cast(into)
    return f"{type(raised.value).__name__}: {' '.join(str(raised.value).split())[:160]}"


def test_a_release_as_lerobot_writes_it_has_nothing_to_report(tmp_path):
    rep = _report(_release(tmp_path))
    assert rep.findings == [], [f.line() for f in rep.findings]
    # Nothing said is only worth something beside what it was held to: every column of both
    # files, and no value read -- each is stored as the very type declared.
    assert rep.stats["step_features"] == {"files": 2, "columns": 2 * len(FEATURES), "cast": 0}


# ── 어느 열이 표의 열인가 ────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(FEATURES))
def test_a_step_column_the_features_do_not_declare_is_reported(tmp_path, name):
    """적재기는 선언에 없는 열이 든 파일을 읽다 여는 단계에서 멈춘다 — 자리 열 넷도 같다."""
    assert measured["columns"]["a column the features do not declare"] == "stops"
    (f,) = _said(_report(_release(tmp_path, **{name: ABSENT})), UNDECLARED)
    assert f.location == f"{STEPS.format(0)} · {name} (선언에 없는 열 1개 · 파일 2개)", f.location


def test_a_column_declared_as_a_camera_stream_is_no_column_of_the_table(tmp_path):
    assert measured["columns"]["a column declared as a camera stream"] == "stops"
    video = {"dtype": "video", "shape": [4, 4, 3], "names": ["height", "width", "channels"]}
    (f,) = _said(_report(_release(tmp_path, **{"safety.margin": video})), UNDECLARED)
    assert f.location.startswith(f"{STEPS.format(0)} · safety.margin "), f.location


def test_a_camera_stream_is_no_column_to_be_missed(tmp_path):
    video = {"dtype": "video", "shape": [4, 4, 3], "names": ["height", "width", "channels"]}
    rep = _report(_release(tmp_path, **{"observation.images.head": video}))
    assert rep.findings == [], [f.line() for f in rep.findings]


def test_a_declared_column_no_file_has_is_reported(tmp_path):
    """적재기는 없는 열을 null 로 채워 연다 — 멈추지 않으므로 다른 검사에는 걸리지 않는다."""
    assert measured["columns"]["a feature no file has a column of"] == "reads"
    (f,) = _said(_report(_release(tmp_path, extra=_feature("float32", [2]))), NO_COLUMN)
    assert f.location == f"{STEPS.format(0)} · extra (파일에 없는 열 1개 · 파일 2개)", f.location


def test_a_declared_column_one_file_lacks_is_reported_for_that_file(tmp_path):
    root = _release(tmp_path, _columns(3), _with(_columns(2, 3), **{"safety.margin": ABSENT}))
    (f,) = _said(_report(root), NO_COLUMN)
    assert f.location == f"{STEPS.format(1)} · safety.margin (파일에 없는 열 1개 · 파일 1개)", f.location


@pytest.mark.parametrize("name", PLACE)
def test_a_place_column_a_file_lacks_is_left_to_the_checks_that_read_it(tmp_path, name):
    """`index` · `episode_index` · `frame_index` · `task_index` 가 없는 파일은 그 열을 읽는 검사가 말한다
    (`c7_data_spans` · `c7_task_table`) — 여기서 한 번 더 말하면 한 결함이 두 줄이 된다."""
    root = _release(tmp_path, _with(_columns(3), **{name: ABSENT}))
    assert _report(root).findings == []


def test_columns_of_a_kind_fold_into_one_line(tmp_path):
    """features 가 통째로 비면 모든 파일의 모든 열이 선언 없는 열이다 — 한 줄로 접는다."""
    (f,) = _said(_report(_release(tmp_path, features={})), UNDECLARED)
    names = list(_columns())
    assert f.location == (f"{STEPS.format(0)} · {', '.join(names[:4])} … "
                          f"(선언에 없는 열 {len(names)}개 · 파일 2개)"), f.location


# ── 저장된 형과 선언한 형 ────────────────────────────────────────

@pytest.mark.parametrize("name", PLACE)
def test_a_value_the_declared_integer_cannot_hold_is_reported_for_each_place_column_alike(tmp_path, name):
    """`int8` 로 선언한 열에 128 — 적재기는 저장된 열을 선언한 형으로 바꾸다 여는 단계에서 멈춘다. 네 열이
    한 규칙이다(실제 적재기로 넷 다 확인)."""
    assert ("int64, one 128", "int8 [1]") in CAST_STOPS and ("int64, one 127", "int8 [1]") not in CAST_STOPS
    values = pa.array([0, 128], pa.int64())
    root = _release(tmp_path, _columns(3), _with(_columns(2, 3), **{name: values}), **{name: _feature("int8", [1])})
    (f,) = _said(_report(root), NO_CAST)
    assert f.location == (f"{STEPS.format(1)} · {name}: 저장 int64, 선언 int8 [1] — "
                          f"{_arrow_says(values, pa.int8())} (바뀌지 않는 열 1개 · 파일 1개)"), f.location
    held = _columns(2, 3)[name].to_pylist()
    assert max(held) <= 127                      # what the other file holds goes in
    rep = _report(_release(tmp_path / "fits", **{name: _feature("int8", [1])}))
    assert rep.findings == [] and rep.stats["step_features"]["cast"] == 2


@pytest.mark.parametrize("declared", ["float64 [1]", "int8 [1]", "float64 [3]", "string [1]"])
def test_the_check_reports_a_column_exactly_when_the_loader_was_measured_to_stop_on_it(tmp_path, declared):
    """규칙이 맞아도 검사가 그 규칙을 그 열에 묻지 않으면 소용없다 — 잰 쌍마다 한 열짜리 배포본으로
    (배포본의 열이 갖는 형의 쌍: 판에 따라 갈리지 않는 것)."""
    assert declared in PLAIN_DECLARED
    for k, stored in enumerate(PLAIN_STORED):
        values = STORED[stored]
        root = _release(tmp_path / str(k), {"x": values}, features={"x": DECLARED[declared]})
        said = [f.message.split(" — ")[0] for f in _report(root).findings]
        assert said == ([NO_CAST] if (stored, declared) in CAST_STOPS else []), (stored, declared, said)


def test_a_list_of_another_length_is_reported_with_its_row(tmp_path):
    lists = pa.array([[0.5, 1.5, 2.5], [0.5, 1.5], [0.5]], pa.list_(pa.float32()))
    (f,) = _said(_report(_release(tmp_path, _with(_columns(3), **{"observation.state": lists}))), NO_CAST)
    assert f.location == (f"{STEPS.format(0)} · observation.state: 저장 list<element: float>, 선언 float32 [3] — "
                          "선언한 길이(3)와 다른 목록이 있음 (1번째 행의 길이 2 · 그런 행 2개) "
                          "(바뀌지 않는 열 1개 · 파일 1개)"), f.location


def test_lists_of_any_length_that_are_all_as_long_as_declared_are_read(tmp_path):
    """pandas 를 거친 파일은 고정 길이 목록이 가변 길이 목록이 된다. 길이가 모두 선언과 같으면 적재기는
    읽는다 — 형이 다르다는 것만으로 말하면 그런 배포본을 거부하게 된다."""
    assert ("list<double>, of 3", "float64 [3]") not in CAST_STOPS
    lists = pa.array([[0.5, 1.5, 2.5]] * 3, pa.list_(pa.float32()))
    rep = _report(_release(tmp_path, _with(_columns(3), **{"observation.state": lists})))
    assert rep.findings == [] and rep.stats["step_features"] == {"files": 1, "columns": len(FEATURES), "cast": 1}


def test_a_column_stored_as_an_extension_array_is_held_as_what_it_stores(tmp_path):
    tensors = pa.FixedShapeTensorArray.from_numpy_ndarray(np.asarray([[0.5, 1.5, 2.5]] * 3, dtype=np.float32))
    root = _release(tmp_path, _with(_columns(3), **{"observation.state": tensors}))
    assert isinstance(pq.read_schema(root / STEPS.format(0)).field("observation.state").type, pa.BaseExtensionType)
    rep = _report(root)
    assert rep.findings == [] and rep.stats["step_features"] == {"files": 1, "columns": len(FEATURES), "cast": 0}
    (f,) = _said(_report(_release(tmp_path / "another length", _with(_columns(3), **{"observation.state": tensors}),
                                  **{"observation.state": _feature("float32", [2])})), NO_CAST)
    assert "길이 3 로 고정된 목록은 길이 2 의 목록이 되지 못함" in f.location, f.location


def test_a_column_with_an_extension_type_under_its_top_is_not_judged(tmp_path):
    """목록의 원소가 확장 형인 열은 판단하지 않는다 — 적재기는 그런 열을 null 인 목록이 없으면 읽고 있으면
    멈추는데(실측), 그것이 전부인지 재지 않았다. 스텝 표의 열이기는 하므로 선언에 없으면 말한다."""
    tensors = pa.FixedShapeTensorArray.from_numpy_ndarray(np.asarray([[0.5], [1.5], [2.5]] * 3))
    lists = pa.ListArray.from_arrays(pa.array([0, 3, 6, 9], pa.int32()), tensors)
    assert vd._extension_inside(lists.type) and not vd._extension_inside(tensors.type.storage_type)
    assert vd._extension_inside(pa.struct([("a", tensors.type)])) and not vd._extension_inside(pa.list_(pa.float64()))
    root = _release(tmp_path, _with(_columns(3), **{"observation.state": lists}))
    rep = _report(root)
    assert rep.findings == [], [f.line() for f in rep.findings]
    assert rep.stats["step_features"] == {"files": 1, "columns": len(FEATURES) - 1, "cast": 0}
    _said(_report(_release(tmp_path / "not declared", _with(_columns(3), **{"observation.state": lists}),
                           **{"observation.state": ABSENT})), UNDECLARED)


def test_columns_that_cannot_be_cast_fold_into_one_line(tmp_path):
    words = pa.array(["a", "b"], pa.string())
    root = _release(tmp_path,
                    _with(_columns(2), timestamp=words),
                    _with(_columns(2, 2), timestamp=words, **{"safety.margin": words}))
    (f,) = _said(_report(root), NO_CAST)
    assert f.location.startswith(f"{STEPS.format(0)} · timestamp: 저장 string, 선언 float32 [1] — "), f.location
    assert f.location.endswith("(바뀌지 않는 열 2개 · 파일 2개)"), f.location


def test_a_cast_that_changes_the_values_and_goes_through_is_not_said(tmp_path):
    """`float64` 로 실리고 `float32` 로 선언된 열은 적재기가 말없이 좁혀 읽는다(1e300 은 inf 가 된다).
    멈추지 않으므로 이 검사는 말하지 않는다 — README 의 「보지 않는 것」."""
    assert ("double, one 1e300", "float32 [1]") not in CAST_STOPS
    root = _release(tmp_path, _with(_columns(3), **{"safety.margin": pa.array([0.1, 1e300, 0.3], pa.float64())}),
                    **{"safety.margin": _feature("float32", [1])})
    assert _report(root).findings == []


# ── null 인 목록 뒤의 스텝 (가변 길이 목록을 고정 길이로) ─────────

def _with_lists(nulls: set[int], alike: bool = False) -> dict:
    """Ten steps whose observation.state is stored as lists of any length, null where asked: each
    step values of its own, or -- ``alike`` -- every step the same."""
    lists = pa.array([None if step in nulls else batch_values(0 if alike else step) for step in range(BATCH_ROWS)],
                     pa.list_(pa.float32()))
    return _with(_columns(BATCH_ROWS), **{"observation.state": lists})


def _in_row_groups(path: Path, groups: list[int]) -> None:
    table = pq.read_table(path)
    assert sum(groups) == table.num_rows
    start = 0
    with pq.ParquetWriter(path, table.schema) as writer:
        for rows in groups:
            writer.write_table(table.slice(start, rows))
            start += rows


MISREAD_AT = f"{STEPS.format(0)} · observation.state: 저장 list<element: float>, 선언 float32 [3] — "


def test_a_null_list_before_a_later_batch_of_its_row_group_stops_the_loader(tmp_path):
    """행 그룹 2 · 8 에 null 이 2번째 스텝 — 적재기는 두 행씩 바꾸고, null 뒤의 배치를 한 목록씩 밀려 읽다
    끝에서 원소가 모자라 여는 단계에서 멈춘다. 파일 전체로 보면 길이도 원소도 멀쩡하다."""
    assert "row groups 2 8, a null at 2" in measured["batches"]["stops"]
    root = _release(tmp_path, _with_lists({2}))
    _in_row_groups(root / STEPS.format(0), [2, 8])
    (f,) = _said(_report(root), MISREAD)
    assert f.location == MISREAD_AT + "4번째 행부터 밀린 자리에서 읽음: 여는 단계에서 멈춤 (그런 열 1개 · 파일 1개)", f.location
    # It stops whatever the lists hold: where they are all alike, too.
    alike = _release(tmp_path / "alike", _with_lists({2}, alike=True))
    _in_row_groups(alike / STEPS.format(0), [2, 8])
    assert "row groups 2 8, a null at 2" in measured["batches"]["alike"]["stops"]
    assert _said(_report(alike), MISREAD)[0].location == f.location


def test_null_lists_at_both_ends_of_a_row_group_hand_out_other_steps_values_with_no_error(tmp_path):
    """행 그룹 2 · 8 에 null 이 2번째와 9번째 — 밀려 읽은 배치가 원소 안에서 끝나고 마지막 배치에는 null 이
    있어 바르게 읽힌다. 적재기는 열리고 4~7번째 스텝에 5~8번째의 값을 내준다(실측 — 스텝마다 값이 다른 열)."""
    assert measured["batches"]["shifts"]["row groups 2 8, nulls at 2 and 9"] == 4
    root = _release(tmp_path, _with_lists({2, 9}))
    _in_row_groups(root / STEPS.format(0), [2, 8])
    (f,) = _said(_report(root), MISREAD)
    assert f.location == (MISREAD_AT + "4번째 행부터 밀린 자리에서 읽음: 오류 없이 다른 스텝의 값을 내줌 "
                          "(값이 처음 달라지는 행 4) (그런 열 1개 · 파일 1개)"), f.location


def test_the_finding_names_the_first_step_that_comes_out_with_other_values(tmp_path):
    """4번째와 5번째 스텝의 값이 같으면, 밀린 자리에서 읽기 시작하는 것은 4번째 행이지만 값이 달라지는 것은
    5번째부터다(실측). 위치에 두 행을 따로 적는다 — 둘이 늘 같은 것이 아니다."""
    columns = _with_lists({2, 9})
    lists = columns["observation.state"].to_pylist()
    lists[5] = lists[4]
    columns = _with(columns, **{"observation.state": pa.array(lists, pa.list_(pa.float32()))})
    root = _release(tmp_path, columns)
    _in_row_groups(root / STEPS.format(0), [2, 8])
    (f,) = _said(_report(root), MISREAD)
    assert f.location == (MISREAD_AT + "4번째 행부터 밀린 자리에서 읽음: 오류 없이 다른 스텝의 값을 내줌 "
                          "(값이 처음 달라지는 행 5) (그런 열 1개 · 파일 1개)"), f.location


def test_lists_that_all_hold_the_same_are_read_as_stored_and_nothing_is_said(tmp_path):
    """같은 자리인데 모든 목록의 값이 같으면, 밀린 자리에서 읽어도 스텝은 저장된 대로 나온다(실측). 적재기가
    내주는 값이 틀리지 않으므로 말하지 않는다 — null 이 있다는 것은 `c7_step_nulls` 의 말이다."""
    assert "row groups 2 8, nulls at 2 and 9" not in measured["batches"]["alike"]["not_as_stored"]
    root = _release(tmp_path, _with_lists({2, 9}, alike=True))
    _in_row_groups(root / STEPS.format(0), [2, 8])
    rep = _report(root)
    assert rep.findings == [] and rep.stats["step_features"]["cast"] == 1


@pytest.mark.parametrize("groups, nulls", [([10], {3}), ([2, 8], {9}), ([2, 8], {2, 4, 6, 8}), ([5, 5], {2})])
def test_a_null_list_no_batch_is_read_wrong_after_is_not_said_here(tmp_path, groups, nulls):
    """null 뒤에 같은 행 그룹의 null 없는 배치가 없으면 적재기는 바르게 읽는다(실측). null 그 자체는
    `c7_step_nulls` 가 말한다 — 여기서 말하는 것은 잘못 읽히는 스텝이다."""
    label = next(label for label, case in BATCHES.items() if case == (groups, nulls))
    assert label not in measured["batches"]["stops"] and label not in measured["batches"]["shifts"]
    root = _release(tmp_path, _with_lists(nulls))
    _in_row_groups(root / STEPS.format(0), groups)
    rep = _report(root)
    assert rep.findings == [] and rep.stats["step_features"]["cast"] == 1


def test_a_list_of_another_length_is_said_as_that_and_not_as_read_wrong(tmp_path):
    lists = _with_lists({2})["observation.state"].to_pylist()
    lists[5] = [0.5]
    root = _release(tmp_path, _with(_columns(BATCH_ROWS), **{"observation.state": pa.array(lists, pa.list_(pa.float32()))}))
    _in_row_groups(root / STEPS.format(0), [2, 8])
    _said(_report(root), NO_CAST)


def test_columns_read_wrong_fold_into_one_line(tmp_path):
    def file(start):
        columns = _with_lists({2})
        return _with(columns, index=pa.array(range(start, start + BATCH_ROWS), pa.int64()),
                     extra=columns["observation.state"])
    root = _release(tmp_path, file(0), file(BATCH_ROWS), extra=_feature("float32", [3]))
    for j in (0, 1):
        _in_row_groups(root / STEPS.format(j), [2, 8])
    (f,) = _said(_report(root), MISREAD)
    assert f.location.startswith(MISREAD_AT) and f.location.endswith("(그런 열 2개 · 파일 2개)"), f.location


def test_a_fixture_step_file_read_wrong_is_the_releases_one_error_past_the_file_limit(tmp_path):
    """온전한 픽스처의 둘째 스텝 파일을 행 그룹 100 · 나머지로 다시 쓰고 `safety.com` 의 100번째 행을 null 로
    둔다. `--file-limit 1` 이면 null 을 보는 검사(`c7_step_nulls`)는 그 파일을 읽지 않는다 — 이 검사만 남고,
    exit 코드는 이 줄의 것이다. null 을 두지 않고 행 그룹만 나눈 파일은 통과한다."""
    root = tmp_path / "clean"
    build(root)
    path = root / STEPS.format(1)
    table = pq.read_table(path)
    position = table.schema.get_field_index("safety.com")
    assert pa.types.is_list(table.schema.field(position).type)
    pq.write_table(table, path)
    _in_row_groups(path, [100, table.num_rows - 100])
    assert vd.main([str(root), "--checks", "C7", "--file-limit", "1"]) == 0

    values = table.column(position).to_pylist()
    values[100] = None
    table = table.set_column(position, table.schema.field(position), pa.array(values, table.schema.field(position).type))
    pq.write_table(table, path)
    _in_row_groups(path, [100, table.num_rows - 100])
    limited = _c7(root, file_limit=1)
    (f,) = [f for f in limited.findings if f.severity == vd.ERROR]
    assert f.message.startswith(MISREAD) and f.location.startswith(f"{STEPS.format(1)} · safety.com: "), f.line()
    assert "200번째 행부터 밀린 자리에서 읽음: 여는 단계에서 멈춤" in f.location, f.location
    assert vd.main([str(root), "--checks", "C7", "--file-limit", "1"]) == 1
    # With every file read, the null is said too -- by the check that reads the rows for nulls.
    said = [f.message.split(" — ")[0] for f in _c7(root).findings if f.severity == vd.ERROR]
    assert MISREAD in said and any("null" in m and m != MISREAD for m in said), said


# ── 선언 그 자체 ─────────────────────────────────────────────────

@pytest.mark.parametrize("features, shown", [(ABSENT, "없음"), (None, "None"), ([], "[]"), ("", "''"), (3, "3")])
def test_an_info_json_without_features_is_reported(tmp_path, features, shown):
    label = "without features" if features is ABSENT else f"features {json.dumps(features)}"
    assert label in measured["info"]["stops"]
    (f,) = _said(_report(_release(tmp_path, features=features)), NO_FEATURES)
    assert f.location == f"{INFO} · features {shown}", f.location


def test_empty_features_are_features_that_declare_nothing(tmp_path):
    # {} reads as an info.json: what stops the loader is the files' columns, none of them declared.
    assert "features {}" in measured["info"]["all"] and "features {}" not in measured["info"]["stops"]
    _said(_report(_release(tmp_path, features={})), UNDECLARED)


@pytest.mark.parametrize("label", sorted(label for label, (kind, _) in _judged().items() if kind == "stops"))
def test_a_declaration_the_loader_cannot_make_a_column_of_is_reported(tmp_path, label):
    """열이 파일에 있든 없든 선언만으로 멈춘다 — 잰 것은 파일에 없는 열의 선언이다."""
    assert label in FORM_STOPS
    (f,) = _said(_report(_release(tmp_path, extra=FORMS[label])), UNMADE)
    assert f.location.startswith(f"{INFO} · extra: ") and f.location.endswith("(그런 선언 1개)"), f.location
    # Not also said as a column no file has: that sentence says the loader fills it with nulls and
    # goes on, and on this declaration it stops.


def test_declarations_that_cannot_be_made_fold_into_one_line(tmp_path):
    root = _release(tmp_path, **{"safety.margin": _feature("<f8", [1]), "safety.event": _feature("str", [1])})
    (f,) = _said(_report(root), UNMADE)
    assert f.location == (f"{INFO} · safety.margin: {_feature('<f8', [1])!r} — dtype 이 pyarrow 의 형 이름이 아님 "
                          "(그런 선언 2개)"), f.location


@pytest.mark.parametrize("name", PLACE)
@pytest.mark.parametrize("declared", [_feature("float32", [1]), _feature("int64", [2]), {"dtype": "int64"},
                                      _feature("long", [1]), "int64"])
def test_a_place_column_that_is_not_declared_one_integer_is_said_once_and_elsewhere(tmp_path, name, declared):
    """자리 열 넷이 정수 하나가 아니면 `c7_table_files` 가 말한다. 같은 선언을 여기서도 말하면 두 줄이다."""
    root = _release(tmp_path, **{name: declared})
    assert _report(root).findings == []
    files = vd.Report(root=str(root))
    vd.c7_table_files(vd.Dataset(root, files), files)
    assert [f.location for f in files.findings if f"features 가 {name} 을" in f.message] == [INFO]


@pytest.mark.parametrize("label", sorted(label for label, (kind, _) in _judged().items() if kind == "unread"))
def test_a_declaration_left_unjudged_is_still_a_column_of_the_table(tmp_path, label):
    """판단하지 않는 선언도 스텝 표의 열이다 — 그 열이 파일에 있으면 「선언 없는 열」이 아니고, 형은 대 보지
    않는다."""
    root = _release(tmp_path, _with(_columns(3), x=pa.array([0.5, 1.5, 2.5], pa.float64())), x=FORMS[label])
    rep = _report(root)
    assert rep.findings == [], [f.line() for f in rep.findings]
    assert rep.stats["step_features"]["columns"] == len(FEATURES)       # x is not among what was held


@pytest.mark.parametrize("label", ["dtype 'float32', shape [3, 2]", "dtype 'image', shape [1]",
                                   "dtype 'timestamp[s]', shape [1]", "dtype 'float32', shape [-1]"])
def test_a_declaration_left_unjudged_whose_column_no_file_has_is_still_missed(tmp_path, label):
    """판단하지 않는 것은 형이다. 그 열이 파일에 없다는 것은 말한다 — 적재기는 그 선언으로 열을 만들어 빈
    값으로 채운다(넷 다 적재기가 표를 만든 선언)."""
    assert vd._feature_column(FORMS[label])[0] == "unread" and label not in FORM_STOPS
    (f,) = _said(_report(_release(tmp_path, extra=FORMS[label])), NO_COLUMN)
    assert f.location == f"{STEPS.format(0)} · extra (파일에 없는 열 1개 · 파일 2개)", f.location


@pytest.mark.parametrize("name", vd.LANGUAGE_COLUMNS)
def test_the_two_columns_lerobot_types_itself_are_not_held_to_their_declaration(tmp_path, name):
    """LeRobot 은 이 두 열을 선언과 무관하게 제 형으로 읽는다. 선언돼 있으면 파일에 있어도 되는 열이고, 그
    밖에는 아무것도 말하지 않는다 — 선언만 있고 파일에 없을 때 적재기가 하는 일은 재지 않았다."""
    root = _release(tmp_path / "there", _with(_columns(3), **{name: pa.array(["a", "b", "c"])}), **{name: "whatever"})
    assert _report(root).findings == []
    assert _report(_release(tmp_path / "declared only", **{name: "whatever"})).findings == []
    # Not declared, it is a column like any other the features lack.
    root = _release(tmp_path / "not declared", _with(_columns(3), **{name: pa.array(["a", "b", "c"])}))
    _said(_report(root), UNDECLARED)


# ── 무엇을 읽는가 ────────────────────────────────────────────────

def _c7(root: Path, file_limit=None) -> vd.Report:
    return vd.run(root, {"C7"}, None, 0, file_limit)


def _features_said(rep: vd.Report) -> list[str]:
    return sorted(s for f in rep.findings for s in SAID if f.message.startswith(s))


def test_the_validator_runs_it_as_part_of_c7_and_only_then(tmp_path):
    root = _release(tmp_path, **{"safety.margin": ABSENT})
    assert _features_said(_c7(root)) == [UNDECLARED]
    assert _features_said(vd.run(root, {"C1"}, None, 0, None)) == []


def test_a_file_limit_does_not_change_which_files_are_held(tmp_path):
    """--file-limit 은 판독기가 읽는 스텝 파일을 줄인다. 이 검사는 footer 를 따로 읽으므로 무관하다."""
    root = _release(tmp_path, _columns(3), _with(_columns(2, 3), task_index=pa.array([0, 128], pa.int64())),
                    task_index=_feature("int8", [1]))
    rep = _c7(root, file_limit=1)
    assert _features_said(rep) == [NO_CAST]
    assert rep.stats["step_features"] == {"files": 2, "columns": 2 * len(FEATURES), "cast": 2}


def test_files_where_the_loader_does_not_look_are_not_held(tmp_path):
    root = _release(tmp_path)
    for stray in ("data/stray.parquet", "data/chunk-000/deeper/file-000.parquet"):
        (root / stray).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"nothing.declares.this": [1]}), root / stray)
    rep = _report(root)
    assert rep.findings == [] and rep.stats["step_features"]["files"] == 2


def test_a_file_with_no_row_group_is_passed_over(tmp_path):
    """행 그룹이 없는 파일은 판독기가 배치를 만들지 않아 맞추지도 않는다(실측 — 선언 없는 열이 있어도
    나머지를 모두 읽는다)."""
    root = _release(tmp_path)
    with pq.ParquetWriter(root / STEPS.format(2), pa.schema([("nothing.declares.this", pa.int64())])):
        pass
    rep = _report(root)
    assert rep.findings == [] and rep.stats["step_features"]["files"] == 2


def test_a_file_whose_row_groups_are_empty_is_left_to_the_check_that_says_it(tmp_path):
    """빈 표를 저장한 파일(0행짜리 행 그룹)에서는 판독기가 맞추기 전에 멈춘다 — `c7_table_files` 의 말이고,
    그 파일의 열을 두고 여기서 더 말하지 않는다."""
    root = _release(tmp_path)
    pq.write_table(pa.table({"nothing.declares.this": pa.array([], pa.int64())}), root / STEPS.format(2))
    rep = _report(root)
    assert rep.findings == [] and rep.stats["step_features"]["files"] == 2
    files = vd.Report(root=str(root))
    vd.c7_table_files(vd.Dataset(root, files), files)
    assert [f.location for f in files.findings if "첫 행 그룹이 0행" in f.message] == [STEPS.format(2)]


def test_a_file_with_two_columns_of_one_name_is_left_to_the_check_that_says_it(tmp_path):
    root = _release(tmp_path)
    twice = pa.Table.from_arrays([pa.array([1]), pa.array([2])], names=["index", "index"])
    pq.write_table(twice, root / STEPS.format(2))
    rep = _report(root)
    assert rep.findings == [] and rep.stats["step_features"]["files"] == 2


def test_a_step_file_that_is_not_parquet_leaves_a_line_and_the_rest_held(tmp_path):
    root = _release(tmp_path)
    (root / STEPS.format(9)).write_bytes(b"not a parquet")
    rep = _report(root)
    assert [(f.check, f.severity, f.location) for f in rep.findings] == [("C7", vd.SKIP, STEPS.format(9))]
    assert NOT_COMPARED in rep.findings[0].message and rep.stats["step_features"]["files"] == 2


def _break_the_columns_page(path: Path, name: str) -> None:
    """Leave the footer whole and overwrite the head of the column's first page: the file opens
    as parquet and its rows cannot be read."""
    group = pq.read_metadata(path).row_group(0)
    (chunk,) = [group.column(k) for k in range(group.num_columns) if group.column(k).path_in_schema == name]
    start = chunk.dictionary_page_offset or chunk.data_page_offset
    raw = bytearray(path.read_bytes())
    raw[start:start + 8] = b"\xff" * 8
    path.write_bytes(raw)


def test_rows_that_cannot_be_read_leave_a_line_and_do_not_end_the_run(tmp_path):
    """footer 는 온전한데 페이지가 깨진 파일 — 값을 읽어야 하는 열(선언과 형이 다른 열)이 있으면 이 검사가
    처음 읽는다. 잡지 않으면 검증기가 exit 3 으로 죽고 보고서가 남지 않는다."""
    root = _release(tmp_path, task_index=_feature("int32", [1]))
    _break_the_columns_page(root / STEPS.format(1), "task_index")
    with pytest.raises(Exception):
        pq.ParquetFile(root / STEPS.format(1)).read()
    rep = _report(root)
    assert [(f.check, f.severity, f.location) for f in rep.findings] == [("C7", vd.SKIP, STEPS.format(1))]
    assert NOT_HELD in rep.findings[0].message
    # The file that could not be read is not among what was held: the other file's columns only.
    assert rep.stats["step_features"] == {"files": 1, "columns": len(FEATURES), "cast": 1}


def test_rows_are_not_read_where_every_column_is_stored_as_declared(tmp_path):
    """실제 배포본은 모든 열이 선언한 형 그대로다 — 그런 파일은 footer 만 본다(행이 깨져 있어도 이 검사는
    지나간다. 그 파일의 결함은 행을 읽는 검사가 말한다)."""
    root = _release(tmp_path)
    _break_the_columns_page(root / STEPS.format(1), "task_index")
    rep = _report(root)
    assert rep.findings == [] and rep.stats["step_features"]["cast"] == 0


@pytest.mark.parametrize("text", [None, "{", "[1, 2]", "7", "null"])
def test_an_info_json_that_does_not_read_is_c1s_to_say_and_leaves_a_line_here(tmp_path, text):
    """`info.json` 이 없거나 · 파싱되지 않거나 · 객체가 아니면 그 결함은 C1 이 말한다. 이 검사는 볼 것이 없다 —
    그런데 `--checks C7` 만 돌리면 C1 은 돌지 않는다. 말없이 돌아가면 C7 이 통과로 읽힌다."""
    root = _release(tmp_path)
    if text is None:
        (root / INFO).unlink()
    else:
        (root / INFO).write_text(text, encoding="utf-8")
    said = [f for f in _report(root).findings if f.check == "C7"]
    assert [(f.severity, f.location) for f in said] == [(vd.SKIP, INFO)], [f.line() for f in said]
    assert NOT_COMPARED in said[0].message and "features" in said[0].message
    # By itself C7 is then not a pass ...
    alone = _c7(root)
    assert alone.status_of("C7") != "ok" and any(f.message == said[0].message for f in alone.findings)
    # ... and C1, where it runs, is the one that says what is wrong with the file.
    both = vd.run(root, {"C1", "C7"}, None, 0, None)
    assert [f for f in both.findings if f.check == "C1" and f.severity == vd.ERROR and "info.json" in f.line()]


def test_without_pyarrow_it_says_it_did_not_look(tmp_path, monkeypatch):
    root = _release(tmp_path, features=ABSENT)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)       # the import then raises ImportError
    rep = _report(root)
    (f,) = rep.findings
    assert (f.check, f.severity) == ("C7", vd.SKIP) and "features" in f.message, f.line()
    assert rep.stats["step_features"] == {}         # the key is there and empty: nothing was held to anything


# ── 픽스처는 배포본과 같은 표현이어야 한다 ─────────────────────────

@pytest.mark.parametrize("with_media", [False, True])
def test_the_fixture_declares_every_step_column_and_no_other(tmp_path, with_media):
    """스텝 열을 선언하지 않는 픽스처로는 이 검사의 통과 쪽을 시험하지 못한다. 선언은 적어 두지 않고
    첫 스텝에서 읽는다 — 변환기가 하는 대로다(`tests/generation/test_fixture_declares_as_the_converter.py`)."""
    root = tmp_path / "clean"
    build(root, with_media=with_media)
    features = json.loads((root / INFO).read_text(encoding="utf-8"))["features"]
    declared = {name for name, feature in features.items() if feature["dtype"] != "video"}
    files = sorted((root / "data").glob(LOADED))
    assert len(files) > 1
    for f in files:
        assert set(pq.read_schema(f).names) == declared, f
    assert all(vd._feature_column(features[name])[0] in ("value", "list") for name in declared)
    assert (len(declared) < len(features)) == with_media          # the camera streams, declared and no column

    rep = _report(root)
    assert rep.findings == [], [f.line() for f in rep.findings]
    held = rep.stats["step_features"]
    assert held["files"] == len(files) and held["columns"] == len(files) * len(declared)
    # The fixture's lists are stored at any length (pandas), a release's at a fixed one: those
    # columns are the ones whose values are read and cast.
    lists = sum(1 for name in declared if features[name]["shape"] != [1])
    assert held["cast"] >= len(files) * lists > 0


# What each variant is reported as, and where.
FEATURE_VARIANTS = {
    "broken_features_absent": (NO_FEATURES, lambda files, columns: f"{INFO} · features 없음"),
    "broken_feature_undeclared_column": (
        UNDECLARED, lambda files, columns: f"{STEPS.format(0)} · safety.com (선언에 없는 열 1개 · 파일 {files}개)"),
    "broken_feature_column_absent": (
        NO_COLUMN, lambda files, columns: f"{STEPS.format(0)} · observation.extra (파일에 없는 열 1개 · 파일 {files}개)"),
    "broken_feature_shape": (
        NO_CAST, lambda files, columns: (
            f"{STEPS.format(0)} · safety.com: 저장 list<element: double>, 선언 float64 [2] — 선언한 길이(2)와 다른 "
            f"목록이 있음 (0번째 행의 길이 3 · 그런 행 {columns['safety.com'][0]}개) (바뀌지 않는 열 1개 · 파일 {files}개)")),
    "broken_feature_dtype_name": (
        UNMADE, lambda files, columns: (
            f"{INFO} · safety.zmp: {_feature('<f8', [2])!r} — dtype 이 pyarrow 의 형 이름이 아님 (그런 선언 1개)")),
    "broken_feature_frame_index_width": (
        NO_CAST, lambda files, columns: (
            f"{STEPS.format(0)} · frame_index: 저장 int64, 선언 int8 [1] — "
            f"{_arrow_says(columns['frame_index'][1], pa.int8())} (바뀌지 않는 열 1개 · 파일 {files}개)")),
}


def test_every_feature_variant_is_held_to_its_sentence():
    assert sorted(FEATURE_VARIANTS) == sorted(v for v in VARIANTS if v.startswith("broken_feature"))


@pytest.mark.parametrize("variant", sorted(FEATURE_VARIANTS))
def test_each_feature_variant_fails_only_on_its_sentence(tmp_path, variant):
    """픽스처의 결함은 하나뿐이어야 한다 — 다른 검사가 같은 입력을 먼저 거부하면, 이 검사를 지워도 그
    픽스처는 여전히 「검출」로 세어진다."""
    sentence, place = FEATURE_VARIANTS[variant]
    root = tmp_path / "clean"
    build(root)
    files = sorted((root / "data").glob(LOADED))
    first = pq.read_table(files[0])
    columns = {name: (first.num_rows, first.column(name).combine_chunks()) for name in ("safety.com", "frame_index")}
    assert vd.main([str(root), "--checks", "C7"]) == 0
    VARIANTS[variant](root)
    (f,) = [f for f in vd.run(root, set(vd.ALL_CHECKS), None, 0, None).findings if f.severity == vd.ERROR]
    assert f.check == "C7" and f.message.startswith(sentence), f.line()
    assert f.location == place(len(files), columns), f.location
    # The one ERROR of the release, so the exit code is this finding's.
    assert vd.main([str(root), "--checks", "C7"]) == 1


def test_the_readme_states_the_counts_of_the_measurement():
    """README 가 적은 측정의 수는 여기서 센 것이다 — 측정을 다시 하면 문서가 따라오게."""
    text = " ".join(README.read_text(encoding="utf-8").split())
    with_versions = measured["measured_with"]
    batches = measured["batches"]
    for number in (f"저장된 열 {len(STORED)}가지 × 선언 {len(DECLARED)}가지 = {len(STORED) * len(DECLARED)}쌍",
                   f"적재기가 멈춘 것은 {len(CAST_STOPS)}쌍",
                   f"null 인 목록의 자리 {len(BATCHES)}가지(행 그룹의 나뉨과 null 의 자리) 가운데 적재기가 멈춘 것은 "
                   f"{len(batches['stops'])}가지, 열려서 다른 스텝의 값을 내준 것은 {len(batches['shifts'])}가지",
                   f"같은 {len(BATCHES)}가지를 에피소드를 골라 읽으면"
                   + (" 모두 저장된 대로 읽힙니다" if not batches["not_as_stored_when_episodes_are_chosen"] else " (다름)"),
                   f"모든 목록의 값이 같은 열로 같은 {len(BATCHES)}가지를 다시 재면 멈추는 {len(batches['alike']['stops'])}가지는 "
                   "그대로 멈추고 나머지는 "
                   + ("전부 저장된 대로 읽힙니다" if not batches["alike"]["not_as_stored"] else "(다름)"),
                   f"lerobot {with_versions['lerobot']} · datasets {with_versions['datasets']} · "
                   f"pyarrow {with_versions['pyarrow']}"):
        assert number in text, number
    assert re.search(r"feature_casts_measured\.json", text)
