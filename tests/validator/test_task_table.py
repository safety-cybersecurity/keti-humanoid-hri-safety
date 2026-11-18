"""스텝의 `task_index` 가 적재기가 읽는 과제 표의 행인가 (C7, `c7_task_table`).

LeRobot 은 항목마다 과제 문장(`item["task"]`)을 붙여 내준다. 문장은 `meta/tasks.parquet` 에 있고
스텝에는 번호(`task_index`)만 있다. 적재기가 읽는 방식은 이렇다(lerobot 0.6.1):

  - 표는 `info.json` 의 `total_tasks` 가 0 보다 클 때만 읽는다(`dataset_metadata._load_metadata`)
  - pandas 로 읽은 표의 **행 인덱스**가 문장이다(`io_utils.load_tasks`)
  - 스텝의 `task_index` 는 그 표의 **행 자리**다 — 표의 `task_index` 열은 읽지 않는다
    (`dataset_reader`: `tasks.iloc[task_idx].name`)

그래서 표가 없거나, 문장이 인덱스가 아니거나, 번호가 표 밖이면 항목이 나오지 않거나 다른 것이 나온다.
여기 사례는 손으로 지은 몇 행짜리 배포본이다 — 검사는 과제 표와 스텝 파일의 `task_index` 열,
`info.json` 만 읽으므로 스텝 표의 열은 하나면 된다. 픽스처 쪽은 맨 아래의 시험과
`test_fixture_detection_reason.py` 의 `broken_task_*` 가 본다.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import validator.validate_dataset as vd
from validator.make_fixture import VARIANTS, build

REPO_ROOT = Path(__file__).resolve().parents[2]
LOADED = "*/*.parquet"          # what LeRobot loads of data/ (io_utils.load_nested_dataset)
TASKS = "meta/tasks.parquet"
STEPS = "data/chunk-000/file-{:03d}.parquet"
INFO = "meta/info.json"
SENTENCES = ("과제 하나", "과제 둘")
TASK_INDEX = ((0, 0, 1), (1, 0))                # the steps' task_index, a tuple to a step file
ROWS = len(SENTENCES)
STEP_ROWS = sum(len(file) for file in TASK_INDEX)
ABSENT = object()

# One sentence for each kind of defect; the numbers are in the finding's location.
NO_TABLE = "LeRobot 이 읽는 과제 표가 없음"
UNREAD = "과제 표를 읽지 못함"
COUNT = "info.json 의 total_tasks 가 과제 표의 행 수와 다름"
NOT_SENTENCES = "과제 표의 인덱스가 과제 문장이 아님"
TWICE = "과제 표에 같은 문장이 둘 이상"
NO_NUMBER = "과제 표에 task_index 열이 없음"
NUMBER_TYPE = "과제 표의 task_index 열이 정수 열이 아님"
NUMBER_OFF = "과제 표의 task_index 열이 행의 자리와 다름"
STEP_ABSENT = "스텝 표에 task_index 열이 없음"
STEP_TYPE = "스텝 표의 task_index 열이 정수 열이 아니어서 과제 표의 행과 대조하지 못함"
OUTSIDE = "스텝의 task_index 가 과제 표의 행이 아님"
# Not this check's sentence: c7_step_features says a stored value the declared type cannot hold, for
# task_index as for every column. Here it is only seen beside what this check then does not say.
NO_CAST = "스텝 표의 열을 info.json 의 features 가 선언한 형으로 바꿀 수 없음"
SAID = (NO_TABLE, UNREAD, COUNT, NOT_SENTENCES, TWICE, NO_NUMBER, NUMBER_TYPE, NUMBER_OFF, STEP_ABSENT, STEP_TYPE,
        OUTSIDE, NO_CAST)
NOT_COMPARED = "대조하지 못함"


def _as_lerobot(root: Path, sentences=SENTENCES, numbers=None, **more) -> None:
    """The task table as LeRobot writes it (dataset_metadata.save_episode_tasks · io_utils.write_tasks):
    the sentences as the frame's index, task_index a column of 0, 1, 2, ..."""
    (root / TASKS).parent.mkdir(parents=True, exist_ok=True)
    columns = {} if numbers is ABSENT else {"task_index": list(range(len(sentences))) if numbers is None else numbers}
    pd.DataFrame({**columns, **more}, index=pd.Index(list(sentences), name="task")).to_parquet(root / TASKS)


def _steps(path: Path, values, kind=pa.int64()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"task_index": pa.array(list(values), type=kind)}), path)


def _state(root: Path, **stated) -> Path:
    """info.json with total_tasks as LeRobot counts it (len of the table), or as ``stated``."""
    info = {"codebase_version": "v3.0", "fps": 50, "total_tasks": ROWS}
    for key, value in stated.items():
        if value is ABSENT:
            info.pop(key, None)
        else:
            info[key] = value
    (root / INFO).parent.mkdir(parents=True, exist_ok=True)
    (root / INFO).write_text(json.dumps(info), encoding="utf-8")
    return root


def _release(root: Path, task_index=TASK_INDEX, sentences=SENTENCES, **stated) -> Path:
    for j, values in enumerate(task_index):
        _steps(root / STEPS.format(j), values)
    _as_lerobot(root, sentences)
    return _state(root, **{"total_tasks": len(sentences), **stated})


def _report(root: Path) -> vd.Report:
    rep = vd.Report(root=str(root))
    vd.c7_task_table(vd.Dataset(root, rep), rep)
    return rep


def _said(rep: vd.Report, *sentences: str) -> list[vd.Finding]:
    """The report is these sentences, one C7 ERROR each, and nothing else."""
    lines = [f.line() for f in rep.findings]
    assert all((f.check, f.severity) == ("C7", vd.ERROR) for f in rep.findings), lines
    kinds = [next((s for s in SAID if f.message.startswith(s)), f.message) for f in rep.findings]
    assert sorted(kinds) == sorted(sentences), lines
    return rep.findings


def test_a_release_as_lerobot_writes_it_has_nothing_to_report(tmp_path):
    rep = _report(_release(tmp_path))
    assert rep.findings == [], [f.line() for f in rep.findings]
    # Nothing said is only worth something beside what it was held to.
    assert rep.stats["task_table"] == {"tasks": ROWS, "steps": STEP_ROWS}


# ── info.json 의 total_tasks ─────────────────────────────────────

@pytest.mark.parametrize("stated", [0, -1, ROWS - 1, ROWS + 1])
def test_a_count_that_is_not_the_tables_is_reported_with_both_numbers(tmp_path, stated):
    """0 이하면 적재기는 과제 표를 읽지 않고 항목을 하나도 내주지 못한다. 다른 수는 읽는 데 차이가 없지만
    적재기가 그 수를 과제 수로 내준다(둘 다 실제 적재기로 확인)."""
    rep = _report(_release(tmp_path, total_tasks=stated))
    (f,) = _said(rep, COUNT)
    assert f.location == f"{INFO} · total_tasks {stated}, 과제 표 {ROWS}행", f.location
    # What the table has, not what info.json says of it.
    assert rep.stats["task_table"] == {"tasks": ROWS, "steps": STEP_ROWS}


def test_an_absent_count_is_what_lerobot_reads_as_zero(tmp_path):
    (f,) = _said(_report(_release(tmp_path, total_tasks=ABSENT)), COUNT)
    assert "total_tasks 없음" in f.location and f"{ROWS}행" in f.location, f.location


def test_an_absent_count_beside_a_table_of_no_rows_is_the_zero_it_is_read_as(tmp_path):
    """위 시험만으로는 「없으면 0」을 못박지 못한다 — 없는 키를 무엇으로 읽든 행이 있는 표와는 다르다.
    표가 0행이면 갈린다. 스텝이 없는 배포본이라 번호를 대 볼 것도 없다."""
    for stated in ({"total_tasks": ABSENT}, {"total_tasks": 0}):
        rep = _report(_release(tmp_path, task_index=(), sentences=(), **stated))
        assert rep.findings == [], [f.line() for f in rep.findings]
        assert rep.stats["task_table"] == {"tasks": 0, "steps": 0}


NOT_INTEGERS = {
    "a float": lambda n: float(n),      # json writes 2.0, and 2.0 == 2
    "a string": lambda n: str(n),
    "null": lambda n: None,
    "a list": lambda n: [n],
}


@pytest.mark.parametrize("kind", sorted(NOT_INTEGERS))
def test_a_count_that_is_not_an_integer_is_reported_as_it_is_written(tmp_path, kind):
    """문자열 · null · 목록은 적재기가 여는 단계에서 TypeError 로 멈춘다. 실수는 읽히지만(0 보다 큰지만
    본다) 적재기가 그 값을 그대로 과제 수로 내준다 — 적힌 그대로를 본다."""
    stated = NOT_INTEGERS[kind](ROWS)
    (f,) = _said(_report(_release(tmp_path, total_tasks=stated)), COUNT)
    assert f"total_tasks {stated!r}," in f.location, f.location


def test_a_long_number_is_shown_cut_with_a_mark_and_not_as_another_number(tmp_path):
    """백 자리 수를 앞의 여든 자만 보이면 다른 수로 읽힌다 — 잘랐다는 표시와 전체 길이를 붙인다."""
    stated = 10 ** 100 + 7
    (f,) = _said(_report(_release(tmp_path, total_tasks=stated)), COUNT)
    shown = f.location.split("total_tasks ")[1].split(",")[0]
    assert shown == f"{str(stated)[:80]}… (전체 {len(str(stated))}자)", f.location


def test_true_is_not_a_count_of_one(tmp_path):
    # One task: True == 1, and a comparison by value alone would pass.
    root = _release(tmp_path, task_index=((0, 0),), sentences=SENTENCES[:1], total_tasks=True)
    (f,) = _said(_report(root), COUNT)
    assert "True," in f.location, f.location


# ── 표가 그 자리에 있고 읽히는가 ─────────────────────────────────

def _remove(root: Path) -> None:
    (root / TASKS).unlink()


def _legacy_jsonl(root: Path) -> None:
    _remove(root)
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": SENTENCES[0]}) + "\n",
                                                encoding="utf-8")


def _dangling_link(root: Path) -> None:
    _remove(root)
    os.symlink(root / "nowhere.parquet", root / TASKS)


NOT_THERE = {"no file": _remove, "the v2 file in its place": _legacy_jsonl, "a link to nothing": _dangling_link}


@pytest.mark.parametrize("case", sorted(NOT_THERE))
@pytest.mark.parametrize("stated", [ROWS, 0])
def test_a_release_without_the_table_is_reported_once(tmp_path, case, stated):
    """적재기는 `meta/tasks.parquet` 한 자리만 읽는다. 없으면 `total_tasks` 가 1 이상일 때는 Hub 로
    메타를 다시 받으러 가고, 0 이면 표를 찾지도 않아 항목마다 실패한다(실제 적재기로 확인) — 어느 쪽이든
    이 사본으로는 항목이 나오지 않는다. 표가 없으면 셀 것도 번호를 댈 것도 없으므로 그 둘은 말하지
    않는다."""
    root = _release(tmp_path, total_tasks=stated)
    NOT_THERE[case](root)
    (f,) = _said(_report(root), NO_TABLE)
    assert f.location == TASKS, f.location
    assert _report(root).stats["task_table"] == {}


def _not_parquet(root: Path) -> None:
    (root / TASKS).write_bytes(b"not a parquet\n")


def _no_bytes(root: Path) -> None:
    (root / TASKS).write_bytes(b"")


def _two_columns_of_one_name(root: Path) -> None:
    both = pa.array([0, 1])
    pq.write_table(pa.Table.from_arrays([both, both], names=["task_index", "task_index"]), root / TASKS)


UNREADABLE = {"not a parquet": _not_parquet, "an empty file": _no_bytes,
              "two columns of one name": _two_columns_of_one_name}


@pytest.mark.parametrize("case", sorted(UNREADABLE))
def test_a_table_pandas_cannot_read_is_reported_not_crashed_on(tmp_path, case):
    root = _release(tmp_path)
    UNREADABLE[case](root)
    rep = _report(root)
    (f,) = _said(rep, UNREAD)
    assert f.location.startswith(f"{TASKS} · "), f.location
    assert rep.stats["task_table"] == {}


def test_a_directory_in_the_tables_place_is_read_as_pandas_reads_it(tmp_path):
    """pandas 는 디렉터리를 그 안의 parquet 들로 읽는다 — 빈 디렉터리는 0행의 표이고(항목이 하나도
    나오지 않는다), parquet 이 든 디렉터리는 그 표다(실제 적재기로 확인). 검증기도 pandas 로 읽으므로
    같은 표를 본다."""
    root = _release(tmp_path)
    table = pq.read_table(root / TASKS)
    _remove(root)
    (root / TASKS).mkdir()
    _said(_report(root), COUNT, OUTSIDE)                # no rows: nothing a step's number can name
    pq.write_table(table, root / TASKS / "part-0.parquet")
    assert _report(root).findings == []


# ── 문장은 표의 행 인덱스다 ──────────────────────────────────────

def _plain(root: Path, **columns) -> None:
    pq.write_table(pa.table(columns), root / TASKS)     # no pandas metadata: no index is restored


NO_INDEX = {
    "a plain table": lambda root: _plain(root, task_index=[0, 1], task=list(SENTENCES)),
    "a plain table, the sentence first": lambda root: _plain(root, task=list(SENTENCES), task_index=[0, 1]),
    "pandas without its index": lambda root: pd.DataFrame(
        {"task_index": [0, 1], "task": list(SENTENCES)}).to_parquet(root / TASKS, index=False),
    "pandas with its row numbers for index": lambda root: pd.DataFrame(
        {"task_index": [0, 1], "task": list(SENTENCES)}).to_parquet(root / TASKS),
}


@pytest.mark.parametrize("case", sorted(NO_INDEX))
def test_sentences_that_are_a_column_and_not_the_index_are_reported(tmp_path, case):
    """열 이름이 같아도(`task_index` · `task`) 문장이 pandas 의 인덱스로 저장되지 않은 표에서는 적재기가
    행 번호(0, 1, …)를 과제 문장으로 내준다 — 오류 없이(실제 적재기로 확인). 번호 열은 제자리라 그것은
    말하지 않는다."""
    root = _release(tmp_path)
    NO_INDEX[case](root)
    (f,) = _said(_report(root), NOT_SENTENCES)
    assert f.location == f"{TASKS} · 0번째 행의 인덱스 0 (문자열이 아닌 행 {ROWS}개)", f.location


def test_a_sentence_that_is_null_or_a_number_is_reported(tmp_path):
    root = _release(tmp_path)
    _as_lerobot(root, (SENTENCES[0], None))
    (f,) = _said(_report(root), NOT_SENTENCES)
    assert f.location.startswith(f"{TASKS} · 1번째 행의 인덱스 ") and f.location.endswith("(문자열이 아닌 행 1개)"), f.location

    _as_lerobot(root, (7, 8))
    (f,) = _said(_report(root), NOT_SENTENCES)
    assert f.location == f"{TASKS} · 0번째 행의 인덱스 7 (문자열이 아닌 행 2개)", f.location


@pytest.mark.parametrize("name", [None, "sentence"])
def test_the_index_need_not_be_named_task(tmp_path, name):
    # load_tasks names the index "task" itself, whatever the file calls it.
    root = _release(tmp_path)
    pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(list(SENTENCES), name=name)).to_parquet(root / TASKS)
    assert _report(root).findings == []


def test_an_empty_sentence_and_another_column_are_left_alone(tmp_path):
    # "" is a string, and the loader hands it out as one; a column it does not read does not matter.
    root = _release(tmp_path)
    _as_lerobot(root, ("", SENTENCES[1]), note=["a", "b"])
    assert _report(root).findings == []


def test_a_sentence_listed_twice_is_reported(tmp_path):
    """항목은 나온다(행의 자리로 읽으므로). 문장으로 과제 번호를 찾는 쪽(`get_task_index`)이 실패한다
    (실제 적재기로 확인: TypeError)."""
    root = _release(tmp_path)
    _as_lerobot(root, (SENTENCES[0], SENTENCES[0]))
    (f,) = _said(_report(root), TWICE)
    assert f.location == f"{TASKS} · {SENTENCES[0]!r} (겹친 문장 1개)", f.location


# ── 표의 task_index 열은 행의 자리다 ─────────────────────────────

def test_a_table_without_its_number_column_is_reported(tmp_path):
    root = _release(tmp_path)
    _as_lerobot(root, numbers=ABSENT)
    (f,) = _said(_report(root), NO_NUMBER)
    assert f.location == TASKS, f.location


@pytest.mark.parametrize("numbers, kind", [([0.0, 1.0], "float64"), (["0", "1"], None), ([False, True], "bool"),
                                           ([0, None], "float64")])
def test_a_number_column_that_is_not_integers_is_reported(tmp_path, numbers, kind):
    root = _release(tmp_path)
    _as_lerobot(root, numbers=numbers)
    (f,) = _said(_report(root), NUMBER_TYPE)
    assert f.location.startswith(f"{TASKS} · ") and (kind is None or f.location == f"{TASKS} · {kind}"), f.location


@pytest.mark.parametrize("numbers, first, off", [([1, 0], (0, 1), 2), ([0, 0], (1, 0), 1), ([0, 5], (1, 5), 1)])
def test_a_number_column_that_is_not_the_rows_place_is_reported(tmp_path, numbers, first, off):
    """적재기는 스텝의 `task_index` 를 표의 **행 자리**로 읽고 이 열은 읽지 않는다 — 행이 (둘, 1)
    (하나, 0) 순서로 놓인 표에서 `task_index` 0 인 스텝은 「둘」을 받는다(실제 적재기로 확인). 문장으로
    번호를 찾는 쪽(`get_task_index`)은 이 열을 읽으므로 둘이 다른 과제를 가리킨다."""
    root = _release(tmp_path)
    _as_lerobot(root, numbers=numbers)
    (f,) = _said(_report(root), NUMBER_OFF)
    assert f.location == f"{TASKS} · {first[0]}번째 행이 task_index {first[1]} (어긋난 행 {off}개)", f.location


def test_a_table_of_no_rows_is_not_also_called_mistyped(tmp_path):
    # pandas reads an empty column back as float64; with no row there is no number to be out of place.
    root = _release(tmp_path, sentences=(), total_tasks=0)
    (f,) = _said(_report(root), OUTSIDE)
    assert f.location.endswith(f"(과제 표 0행 · 벗어난 스텝 {STEP_ROWS}개)"), f.location


def test_what_one_finding_says_of_the_items_does_not_deny_another(tmp_path):
    """`total_tasks` 가 0 이면 항목이 하나도 나오지 않는다. 같은 보고서의 다른 줄이 「항목은 나온다」고
    하면 읽는 쪽은 어느 것도 믿지 못한다 — 표의 번호 열과 겹친 문장의 줄은 그 결함이 하는 일만 말한다."""
    root = _release(tmp_path, total_tasks=0)
    _as_lerobot(root, (SENTENCES[0], SENTENCES[0]), numbers=ABSENT)
    findings = _said(_report(root), COUNT, TWICE, NO_NUMBER)
    assert [f.message for f in findings if "항목은 나온다" in f.message] == []


# ── 스텝의 task_index 는 표의 행을 가리킨다 ──────────────────────

@pytest.mark.parametrize("value", [ROWS, ROWS + 7, -1, -ROWS - 1])
def test_a_step_whose_number_is_no_row_of_the_table_is_reported(tmp_path, value):
    """표의 행 수 이상이면 그 항목에서 IndexError 이고, `for` 로 도는 쪽은 거기서 오류 없이 끝난다.
    -1 은 오류 없이 마지막 행의 문장을 내준다(모두 실제 적재기로 확인)."""
    root = _release(tmp_path, task_index=((0, 0, 1), (1, value)))
    rep = _report(root)
    (f,) = _said(rep, OUTSIDE)
    assert f.location == f"{STEPS.format(1)} · 1번째 행의 task_index {value} (과제 표 {ROWS}행 · 벗어난 스텝 1개)", f.location
    assert rep.stats["task_table"] == {"tasks": ROWS, "steps": STEP_ROWS}


def test_the_last_row_of_the_table_is_a_row_of_it(tmp_path):
    assert _report(_release(tmp_path, task_index=((ROWS - 1,) * 3,))).findings == []


def test_steps_outside_the_table_fold_into_one_line(tmp_path):
    # Every step of a release can be off at once (a table cut short): one line, the first place, the count.
    root = _release(tmp_path, task_index=((0, 9, 9), (9, 1)))
    (f,) = _said(_report(root), OUTSIDE)
    assert f.location == f"{STEPS.format(0)} · 1번째 행의 task_index 9 (과제 표 {ROWS}행 · 벗어난 스텝 3개)", f.location


def test_a_step_with_no_number_is_no_row_of_the_table(tmp_path):
    """빈 값(null)은 적재기가 그 항목에서 실패한다(실제 적재기로 확인). 스텝의 null 은 `c7_step_nulls` 도
    말하지만 그 검사는 읽은 파일만 본다."""
    root = _release(tmp_path)
    _steps(root / STEPS.format(1), [1, None])
    (f,) = _said(_report(root), OUTSIDE)
    assert f.location == f"{STEPS.format(1)} · 1번째 행의 task_index null (과제 표 {ROWS}행 · 벗어난 스텝 1개)", f.location


def test_a_step_file_without_the_column_is_reported_and_the_others_still_compared(tmp_path):
    """열이 없는 파일의 스텝은 항목으로 나오지 않는다 — features 가 선언했으면 빈 값으로 채워져 그
    항목들에서 실패하고 다른 파일의 항목은 나온다(실제 적재기로 확인). 그래서 파일마다 말한다."""
    root = _release(tmp_path, task_index=((0, 0, 1), (1, 0), (0, ROWS)))
    pq.write_table(pa.table({"index": [3, 4]}), root / STEPS.format(1))
    rep = _report(root)
    absent, outside = sorted(_said(rep, STEP_ABSENT, OUTSIDE), key=lambda f: f.location)
    assert absent.location == STEPS.format(1), absent.location
    assert outside.location.startswith(f"{STEPS.format(2)} · 1번째 행"), outside.location
    assert rep.stats["task_table"] == {"tasks": ROWS, "steps": 5}


@pytest.mark.parametrize("kind, values", [(pa.float64(), [0.0, 1.0]), (pa.string(), ["0", "1"]), (pa.bool_(), [False, True])])
def test_a_step_column_that_is_not_integers_is_reported(tmp_path, kind, values):
    """적재기는 저장된 형을 features 가 선언한 형으로 바꿔 읽으므로 `0.0` 이나 `"0"` 으로 실린 열도
    읽어 낸다(실제 적재기로 확인). 배포본은 이 열을 정수로 싣고, 검증기는 정수가 아닌 값을 자리로 대 보지
    않는다 — 대조하지 못했다는 것을 말한다."""
    root = _release(tmp_path)
    _steps(root / STEPS.format(1), values, kind)
    rep = _report(root)
    (f,) = _said(rep, STEP_TYPE)
    assert f.location == f"{STEPS.format(1)} · {kind}", f.location
    assert rep.stats["task_table"] == {"tasks": ROWS, "steps": len(TASK_INDEX[0])}


@pytest.mark.parametrize("kind", [pa.int8(), pa.int32(), pa.uint8(), pa.uint64()])
def test_any_width_of_integer_is_a_number(tmp_path, kind):
    root = _release(tmp_path)
    _steps(root / STEPS.format(1), [1, 0], kind)
    assert _report(root).findings == []
    _steps(root / STEPS.format(1), [1, ROWS], kind)
    _said(_report(root), OUTSIDE)


def test_a_step_file_with_the_column_twice_is_another_checks_to_report(tmp_path):
    """같은 이름의 열이 둘인 파일은 `c7_table_files` 가 말한다(적재기가 열을 이름으로 찾다 실패한다).
    어느 쪽이 그 열인지 알 수 없으므로 여기서는 대 보지 않고, 죽지도 않는다."""
    root = _release(tmp_path)
    both = pa.array([0, ROWS])
    pq.write_table(pa.Table.from_arrays([both, both], names=["task_index", "task_index"]), root / STEPS.format(1))
    rep = _report(root)
    assert rep.findings == [] and rep.stats["task_table"] == {"tasks": ROWS, "steps": len(TASK_INDEX[0])}
    said = [f.location for f in _files_report(root).findings if "같은 이름의 열" in f.message]
    assert said == [f"{STEPS.format(1)} · ['task_index']"], said


@pytest.mark.parametrize("kind, value", [(pa.int16(), 256), (pa.int32(), 2 ** 16), (pa.int64(), 2 ** 40),
                                         (pa.uint64(), 2 ** 63 + 5), (pa.int64(), -2 ** 40)])
def test_a_large_number_is_compared_as_the_number_it_is(tmp_path, kind, value):
    """값을 좁은 정수로 옮겨 대면 256 이 0 이 되어 표 안의 번호로 보인다."""
    root = _release(tmp_path)
    _steps(root / STEPS.format(1), [1, value], kind)
    (f,) = _said(_report(root), OUTSIDE)
    assert f.location == f"{STEPS.format(1)} · 1번째 행의 task_index {value} (과제 표 {ROWS}행 · 벗어난 스텝 1개)", f.location


def test_a_step_file_with_no_row_group_has_no_step_to_report(tmp_path):
    """행 그룹이 없는 파일은 적재기가 배치를 만들지 않아 나머지 파일의 항목을 모두 내준다 — 그 파일에
    `task_index` 열이 없어도(실제 적재기로 확인). 없는 스텝을 두고 「내주지 못한다」고 하지 않는다."""
    root = _release(tmp_path)
    with pq.ParquetWriter(root / STEPS.format(2), pa.schema([("index", pa.int64())])):
        pass
    assert pq.read_metadata(root / STEPS.format(2)).num_row_groups == 0, "the case is gone"
    rep = _report(root)
    assert rep.findings == [] and rep.stats["task_table"] == {"tasks": ROWS, "steps": STEP_ROWS}


def test_a_step_file_whose_only_row_group_is_empty_is_another_checks_to_report(tmp_path):
    # The loader fails on such a file (c7_table_files says so); here it has no step either.
    root = _release(tmp_path)
    pq.write_table(pa.table({"index": pa.array([], type=pa.int64())}), root / STEPS.format(2))
    assert _report(root).findings == []
    said = [f.location for f in _files_report(root).findings if "첫 행 그룹이 0행" in f.message]
    assert said == [STEPS.format(2)], said


def _break_the_columns_page(path: Path, name: str = "task_index") -> None:
    """Leave the footer whole and overwrite the head of the column's first page: the file opens
    as parquet and its column cannot be read."""
    group = pq.read_metadata(path).row_group(0)
    (chunk,) = [group.column(k) for k in range(group.num_columns) if group.column(k).path_in_schema == name]
    start = chunk.dictionary_page_offset or chunk.data_page_offset
    raw = bytearray(path.read_bytes())
    raw[start:start + 8] = b"\xff" * 8
    path.write_bytes(raw)


def test_a_column_that_cannot_be_read_leaves_a_line_and_does_not_end_the_run(tmp_path):
    """footer 는 온전한데 열의 페이지가 깨진 파일 — 판독기가 --file-limit 뒤라 열지 않으면 이 검사가
    처음 읽는다. 잡지 않으면 검증기가 exit 3 으로 죽고 보고서가 남지 않는다."""
    root = _release(tmp_path, task_index=((0, 1), (1, ROWS)))
    _break_the_columns_page(root / STEPS.format(1))
    with pytest.raises(Exception):
        pq.ParquetFile(root / STEPS.format(1)).read()          # the case: readable footer, unreadable column
    rep = _report(root)
    assert [(f.check, f.severity, f.location) for f in rep.findings] == [("C7", vd.SKIP, STEPS.format(1))]
    assert rep.stats["task_table"] == {"tasks": ROWS, "steps": 2}


def test_an_unreadable_column_past_the_file_limit_is_a_skip_in_a_report_that_is_written(tmp_path):
    """온전한 픽스처의 둘째 스텝 파일에서 `task_index` 의 페이지만 깬다. --file-limit 1 이면 판독기는 그
    파일을 열지 않으므로 다른 ERROR 가 없다 — 보고서가 나오고(exit 3 이 아니다), 그 파일을 대조하지 못했다는
    줄이 있고, C7 은 통과가 아니라 skip 으로 읽힌다."""
    root = tmp_path / "clean"
    build(root)
    _break_the_columns_page(root / STEPS.format(1))
    rep = _c7(root, file_limit=1)
    said = [(f.severity, f.location) for f in rep.findings if "task_index" in f.message and NOT_COMPARED in f.message]
    assert said == [(vd.SKIP, STEPS.format(1))], [f.line() for f in rep.findings]
    assert rep.count(vd.ERROR) == 0 and rep.status_of("C7") == "skip", [f.line() for f in rep.findings]
    out = tmp_path / "report.json"
    assert vd.main([str(root), "--checks", "C7", "--file-limit", "1", "--json", str(out)]) != 3 and out.exists()


def test_steps_are_not_compared_with_a_table_that_was_not_read(tmp_path):
    root = _release(tmp_path, task_index=((0, 99),))
    _not_parquet(root)
    _said(_report(root), UNREAD)
    # What needs no table is still said.
    pq.write_table(pa.table({"index": [0, 1]}), root / STEPS.format(0))
    _said(_report(root), UNREAD, STEP_ABSENT)


# ── 무엇을 읽는가: 적재기가 읽는 스텝 파일 전부 ──────────────────

def _c7(root: Path, file_limit=None) -> vd.Report:
    return vd.run(root, {"C7"}, None, 0, file_limit)


def _task_said(rep: vd.Report) -> list[str]:
    return sorted(s for f in rep.findings for s in SAID if f.message.startswith(s))


def test_the_validator_runs_it_as_part_of_c7_and_only_then(tmp_path):
    root = _release(tmp_path, task_index=((0, ROWS),))
    assert _task_said(_c7(root)) == [OUTSIDE]
    assert _task_said(vd.run(root, {"C1"}, None, 0, None)) == []


def test_a_file_limit_does_not_change_which_steps_are_compared(tmp_path):
    """--file-limit 은 판독기가 읽는 스텝 파일을 줄인다. 이 검사는 열 하나만 따로 읽으므로 무관하다 —
    읽은 파일만 보면 둘째 파일의 번호가 표 밖이어도 통과한다."""
    root = _release(tmp_path, task_index=((0, 1), (1, ROWS)))
    rep = _c7(root, file_limit=1)
    assert _task_said(rep) == [OUTSIDE]
    assert rep.stats["task_table"] == {"tasks": ROWS, "steps": 4}


def test_steps_where_the_loader_does_not_look_are_not_compared(tmp_path):
    """적재기가 읽는 것은 한 단계 아래의 파일뿐이다(그 밖의 parquet 이 있다는 것은 `c7_table_files` 가
    말한다). 거기 든 번호를 대 보면 적재기가 받지 않는 행을 두고 말하게 된다."""
    root = _release(tmp_path)
    _steps(root / "data" / "stray.parquet", [99])
    _steps(root / "data" / "chunk-000" / "deeper" / "file-000.parquet", [99])
    rep = _report(root)
    assert rep.findings == [] and rep.stats["task_table"] == {"tasks": ROWS, "steps": STEP_ROWS}


def test_a_chunk_directory_that_is_a_symlink_is_compared(tmp_path):
    # LeRobot's glob("*/*.parquet") goes into a chunk directory that is a symlink (measured with the loader).
    root = _release(tmp_path / "release")
    outside = tmp_path / "elsewhere"
    _steps(outside / "file-000.parquet", [0, ROWS])
    (root / "data" / "chunk-001").symlink_to(outside, target_is_directory=True)
    (f,) = _said(_report(root), OUTSIDE)
    assert f.location.startswith("data/chunk-001/file-000.parquet · 1번째 행"), f.location


def test_a_step_file_that_is_not_parquet_leaves_a_line_and_the_rest_compared(tmp_path):
    """읽히지 않는 파일의 결함은 판독기가 말하지만(C1) --file-limit 뒤의 파일은 판독기가 열지 않는다.
    그 파일의 번호를 대조하지 못했다는 것은 여기서 SKIP 으로 말한다 — 말없이 꺼진 대조는 통과로 읽힌다."""
    root = _release(tmp_path)
    (root / STEPS.format(9)).write_bytes(b"not a parquet")
    rep = _report(root)
    assert [(f.check, f.severity, f.location) for f in rep.findings] == [("C7", vd.SKIP, STEPS.format(9))]
    assert NOT_COMPARED in rep.findings[0].message
    assert rep.stats["task_table"] == {"tasks": ROWS, "steps": STEP_ROWS}


@pytest.mark.parametrize("text", [None, "{", "[1, 2]", "7", "null"])
def test_an_info_json_that_is_no_object_leaves_the_count_to_c1_and_the_steps_compared(tmp_path, text):
    root = _release(tmp_path, task_index=((0, ROWS),))
    if text is None:
        (root / INFO).unlink()
    else:
        (root / INFO).write_text(text, encoding="utf-8")
    # A file that does not parse is said as it is read (C1), whoever asks for it.
    said = [f for f in _report(root).findings if f.check == "C7"]
    assert [f.message.split(" — ")[0] for f in said] == [OUTSIDE], [f.line() for f in said]


def test_without_pyarrow_it_says_it_did_not_look(tmp_path, monkeypatch):
    root = _release(tmp_path, total_tasks=0)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)       # the import then raises ImportError
    rep = _report(root)
    (f,) = rep.findings
    assert (f.check, f.severity) == ("C7", vd.SKIP) and "task_index" in f.message, f.line()
    assert rep.stats["task_table"] == {}            # the key is there and empty: nothing was held to anything


# ── features 가 task_index 를 선언한 형 (c7_table_files) ─────────

DECLARED_NOT_INTEGER = "을 정수 하나(dtype int8~uint64 · shape [1])로 선언하지 않음"


def _files_report(root: Path) -> vd.Report:
    rep = vd.Report(root=str(root))
    vd.c7_table_files(vd.Dataset(root, rep), rep)
    return rep


def _declared(root: Path, declared) -> Path:
    info = json.loads((root / INFO).read_text(encoding="utf-8"))
    info["features"] = {} if declared is ABSENT else {"task_index": declared}
    (root / INFO).write_text(json.dumps(info), encoding="utf-8")
    return root


@pytest.mark.parametrize("declared", [
    {"dtype": "int64", "shape": [1], "names": None}, {"dtype": "int32", "shape": [1]}, {"dtype": "uint8", "shape": [1]},
    ABSENT,
])
def test_task_index_declared_as_one_integer_or_not_at_all_is_left_alone(tmp_path, declared):
    said = [f for f in _files_report(_declared(_release(tmp_path), declared)).findings if "features" in f.message]
    assert said == [], [f.line() for f in said]


@pytest.mark.parametrize("declared", [
    {"dtype": "float32", "shape": [1]}, {"dtype": "float64", "shape": [1]}, {"dtype": "bool", "shape": [1]},
    {"dtype": "string", "shape": [1]}, {"dtype": "int64", "shape": [2]}, {"dtype": "int64"},
    {"dtype": "long", "shape": [1]},
])
def test_task_index_declared_as_anything_but_one_integer_is_reported(tmp_path, declared):
    """parquet 의 형은 int64 그대로다. 적재기는 스텝 표를 features 가 선언한 대로 바꿔 읽고 그 값으로
    과제 표의 행을 찾는다 — 실수 · 참거짓으로 선언하면 항목마다 TypeError, 문자열이면 AttributeError,
    모양이 `[1]` 이 아니거나 없거나 형 이름이 pyarrow 의 것이 아니면 여는 단계에서 멈춘다(일곱 다 실제
    적재기로 확인)."""
    said = [f for f in _files_report(_declared(_release(tmp_path), declared)).findings if "features" in f.message]
    assert [(f.check, f.severity, f.location) for f in said] == [("C7", vd.ERROR, INFO)], [f.line() for f in said]
    assert "features 가 task_index 을" in said[0].message and DECLARED_NOT_INTEGER in said[0].message, said[0].line()


# ── 저장된 번호가 선언한 정수 형에 들어가는가 ────────────────────
# 그 말은 `c7_step_features` 가 한다 — 자리 열 넷과 그 밖의 열에 한 규칙으로(`test_step_features.py`).
# 여기서는 두 검사를 함께 돌려, 들어가지 않는 번호가 한 줄로만 나오는지를 본다: 이 검사는 그런 값을
# 「표의 행이 아니다」로 한 번 더 말하지 않는다(적재기가 표까지 가지 못한다).

def _tasks_declared(root: Path, tasks: int, dtype: str, *files, kind=pa.int64()) -> Path:
    """A table of ``tasks`` rows, the steps' numbers as given, task_index declared ``dtype``."""
    _release(root, task_index=(), sentences=tuple(f"과제 {k}" for k in range(tasks)))
    for j, values in enumerate(files):
        _steps(root / STEPS.format(j), values, kind)
    return _declared(root, {"dtype": dtype, "shape": [1], "names": None})


def _with_the_features(root: Path) -> vd.Report:
    """c7_step_features and this check into one report, as the validator runs them."""
    rep = vd.Report(root=str(root))
    ds = vd.Dataset(root, rep)
    vd.c7_step_features(ds, rep)
    vd.c7_task_table(ds, rep)
    return rep


def _arrow_says(values, kind, into: str) -> str:
    """Arrow's own words for the cast it refuses, as the finding quotes them -- asked of the
    pyarrow that is here, whatever that pyarrow words it as."""
    with pytest.raises(Exception) as raised:
        pa.array(values, type=kind).cast(getattr(pa, into)())
    return f"{type(raised.value).__name__}: {' '.join(str(raised.value).split())[:160]}"


@pytest.mark.parametrize("tasks, dtype, value", [(129, "int8", 127), (257, "uint8", 255), (129, "int16", 128),
                                                 (2, "uint64", 1), (2, "int64", 1)])
def test_a_number_the_declared_type_holds_is_left_alone(tmp_path, tasks, dtype, value):
    rep = _with_the_features(_tasks_declared(tmp_path, tasks, dtype, [0, value]))
    assert rep.findings == [], [f.line() for f in rep.findings]


@pytest.mark.parametrize("tasks, dtype, value", [
    (129, "int8", 128), (129, "int8", -129), (257, "uint8", 256), (257, "uint8", -1),
    (2, "uint64", -1), (2, "int32", 2 ** 31),
])
def test_a_number_the_declared_type_cannot_hold_is_reported_once(tmp_path, tasks, dtype, value):
    """적재기는 저장된 열을 features 가 선언한 형으로 바꿔 읽고, 들어가지 않는 값에서 여는 단계에서
    멈춘다(여섯 다 실제 적재기로 확인). 129행짜리 표의 128 은 표 안의 번호라, 표와만 대 보면 통과한다.
    표 밖이기도 한 값(-1, 2행 표의 2**31)은 한 줄로만 말한다 — 적재기가 표까지 가지 못한다."""
    root = _tasks_declared(tmp_path, tasks, dtype, [0, 1], [1, value])
    rep = _with_the_features(root)
    (f,) = _said(rep, NO_CAST)
    assert f.location == (f"{STEPS.format(1)} · task_index: 저장 int64, 선언 {dtype} [1] — "
                          f"{_arrow_says([1, value], pa.int64(), dtype)} (바뀌지 않는 열 1개 · 파일 1개)"), f.location
    assert str(value) in f.location
    assert rep.stats["task_table"] == {"tasks": tasks, "steps": 4}
    # This check by itself says nothing of it -- not that it is no row of the table either.
    assert _report(root).findings == []


def test_a_stored_number_past_int64_is_held_to_a_declared_int64(tmp_path):
    root = _tasks_declared(tmp_path, 2, "int64", [0, 2 ** 63], kind=pa.uint64())
    (f,) = _said(_with_the_features(root), NO_CAST)
    assert f"task_index: 저장 uint64, 선언 int64 [1] — {_arrow_says([0, 2 ** 63], pa.uint64(), 'int64')} " in f.location
    assert str(2 ** 63) in f.location and _report(root).findings == []


def test_minus_one_beside_a_signed_type_is_held_and_is_no_row(tmp_path):
    # int8 holds -1, so the loader opens and reads the last row's sentence for it: the other finding.
    (f,) = _said(_with_the_features(_tasks_declared(tmp_path, 129, "int8", [0, -1])), OUTSIDE)
    assert "task_index -1 " in f.location, f.location


def test_numbers_the_type_cannot_hold_fold_into_one_line_beside_the_ones_that_are_no_row(tmp_path):
    root = _tasks_declared(tmp_path, 2, "int8", [0, 128, 5], [300, 1, None])
    (no_cast,), (outside,) = ([f for f in _said(_with_the_features(root), NO_CAST, OUTSIDE)
                               if f.message.startswith(sentence)] for sentence in (NO_CAST, OUTSIDE))
    assert outside.message.startswith(OUTSIDE) and no_cast.message.startswith(NO_CAST)
    assert no_cast.location == (f"{STEPS.format(0)} · task_index: 저장 int64, 선언 int8 [1] — "
                                f"{_arrow_says([0, 128, 5], pa.int64(), 'int8')} (바뀌지 않는 열 1개 · 파일 2개)")
    # 128 and 300 are no rows of a table of two either, and are not counted as that: 5 and the null are.
    assert outside.location == f"{STEPS.format(0)} · 2번째 행의 task_index 5 (과제 표 2행 · 벗어난 스텝 2개)"


@pytest.mark.parametrize("declared", [ABSENT, {"dtype": "float32", "shape": [1]}, {"dtype": "int8", "shape": [2]},
                                      {"dtype": "int8"}, "int8"])
def test_without_one_integer_declared_there_is_no_type_to_hold_the_numbers_to(tmp_path, declared):
    """선언이 없거나 정수 하나가 아니면 맞춰 볼 형이 없다(정수 하나가 아닌 선언은 `c7_table_files` 가
    말한다). 표 안의 128 은 그대로 표 안의 번호다."""
    root = _declared(_tasks_declared(tmp_path, 129, "int8", [0, 128]), declared)
    assert _report(root).findings == []


def test_the_declared_type_is_held_also_when_there_is_no_table(tmp_path):
    root = _tasks_declared(tmp_path, 129, "int8", [0, 128])
    _remove(root)
    _said(_with_the_features(root), NO_TABLE, NO_CAST)


def test_a_file_limit_does_not_hide_a_number_the_type_cannot_hold(tmp_path):
    root = _tasks_declared(tmp_path, 129, "int8", [0, 1], [1, 128])
    assert _task_said(_c7(root, file_limit=1)) == [NO_CAST]


# ── 픽스처는 배포본과 같은 표현이어야 한다 ─────────────────────────

def _on_disk(root: Path) -> dict[str, int]:
    """What the release holds, counted here and not by the validator."""
    return {"tasks": len(pd.read_parquet(root / TASKS)),
            "steps": sum(len(pq.read_table(f, columns=["task_index"])) for f in (root / "data").glob(LOADED))}


@pytest.mark.parametrize("with_media", [False, True])
def test_the_fixture_ships_a_task_table_as_lerobot_writes_it(tmp_path, with_media):
    """과제 표가 없는 픽스처로는 이 검사의 통과 쪽을 시험하지 못한다. 문장은 적어 두지 않고 저장소의
    시나리오 정본의 `robot_task` 에서 읽는다 — 변환기가 프레임마다 넣는 문장이 그것이다. 픽스처가 실은
    사본에서 읽으면, 생성기가 사본과 표를 같은 틀린 문장으로 써도 통과한다."""
    root = tmp_path / "clean"
    build(root, with_media=with_media)
    table = pd.read_parquet(root / TASKS)
    canon = json.loads((REPO_ROOT / "meta" / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    used = set(pd.read_parquet(next(iter(sorted((root / "meta" / "episodes").glob(LOADED)))))["scenario_id"])
    expected = [s["robot_task"] for s in canon if s["scenario_id"] in used]
    assert expected and list(table.index) == expected
    assert table["task_index"].tolist() == list(range(len(table))) and table["task_index"].dtype == "int64"
    # The sentences are the frame's index in the file itself, not only once pandas has read it.
    assert json.loads(pq.read_schema(root / TASKS).metadata[b"pandas"])["index_columns"] == ["task"]

    info = json.loads((root / INFO).read_text(encoding="utf-8"))
    counted = _on_disk(root)
    assert counted["tasks"] >= 1 and counted["steps"] >= 1, counted
    assert type(info.get("total_tasks")) is int and info["total_tasks"] == counted["tasks"]
    for f in (root / "data").glob(LOADED):
        column = pq.read_table(f, columns=["task_index"]).column(0)
        assert column.type == pa.int64() and set(column.to_pylist()) <= set(range(counted["tasks"])), f

    rep = _report(root)
    assert rep.findings == [], [f.line() for f in rep.findings]
    assert rep.stats["task_table"] == counted


# What each variant is reported as, and where: the place is the fixture's own (a table of ``tasks``
# rows, counted before the variant is made), and the step is one of the second step file.
TASK_VARIANTS = {
    "broken_task_count": (COUNT, lambda tasks: f"{INFO} · total_tasks 0, 과제 표 {tasks}행"),
    "broken_task_table_absent": (NO_TABLE, lambda tasks: TASKS),
    "broken_task_sentences_not_index": (
        NOT_SENTENCES, lambda tasks: f"{TASKS} · 0번째 행의 인덱스 0 (문자열이 아닌 행 {tasks}개)"),
    "broken_task_index_outside": (
        OUTSIDE, lambda tasks: f"{STEPS.format(1)} · 3번째 행의 task_index {tasks} (과제 표 {tasks}행 · 벗어난 스텝 1개)"),
    "broken_task_index_beyond_feature": (
        NO_CAST, lambda tasks: (f"{STEPS.format(1)} · task_index: 저장 int64, 선언 int8 [1] — "
                                f"{_arrow_says([0, 128], pa.int64(), 'int8')} (바뀌지 않는 열 1개 · 파일 1개)")),
}


def test_every_task_variant_is_held_to_its_sentence():
    assert sorted(TASK_VARIANTS) == sorted(v for v in VARIANTS if v.startswith("broken_task_"))


@pytest.mark.parametrize("variant", sorted(TASK_VARIANTS))
def test_each_task_variant_fails_only_on_its_sentence(tmp_path, variant):
    """픽스처의 결함은 하나뿐이어야 한다 — 다른 검사가 같은 입력을 먼저 거부하면, 이 검사를 지워도 그
    픽스처는 여전히 「검출」로 세어진다. 자리도 본다: 표 밖의 번호는 둘째 스텝 파일에 심으므로, 첫
    파일만 보는 검사로는 잡히지 않는다."""
    sentence, place = TASK_VARIANTS[variant]
    root = tmp_path / "clean"
    build(root)
    tasks = len(pd.read_parquet(root / TASKS))
    assert vd.main([str(root), "--checks", "C7"]) == 0
    VARIANTS[variant](root)
    (f,) = [f for f in vd.run(root, set(vd.ALL_CHECKS), None, 0, None).findings if f.severity == vd.ERROR]
    assert f.check == "C7" and f.message.startswith(sentence), f.line()
    assert f.location == place(tasks), f.location
    # The one ERROR of the release, so the exit code is this finding's.
    assert vd.main([str(root), "--checks", "C7"]) == 1
