"""에피소드 표가 가리키는 행이 그 에피소드의 행인가 (C7, `c7_data_spans`) — 그리고 그 앞의 물음,
적재기가 두 표를 열기는 하는가 (C7, `c7_table_files`).

LeRobot v3.0 은 에피소드마다 파일을 두지 않는다. 스텝은 여러 에피소드가 한 parquet 을 나눠 쓰고,
어느 행이 어느 에피소드인지는 `meta/episodes` 의 네 열이 말한다 —

    dataset_from_index · dataset_to_index   스텝 표에서의 자리 [from, to)
    data/chunk_index · data/file_index      그 행이 든 파일

적재기는 이 포인터를 믿는다(lerobot 0.6.1 `dataset_reader._get_query_indices` 가 구간을,
`dataset_metadata.get_data_file_path` 가 파일을 읽는다). 이 검사가 없던 동안 검증기는 표가 다른
에피소드의 행을 가리키는 배포본에 ERROR 0 을 냈다 — 다른 검사는 모두 행의 `episode_index` 로
묶어 읽으므로 포인터가 틀려도 보지 못한다.

여기 사례는 대부분 손으로 지은 몇 행짜리 배포본이다. 결함 하나에 문장 하나가 나오는지를 보려면
그 결함만 든 입력이 필요하고, 합성 픽스처(400행 × 10)는 그러기엔 크다. 픽스처 쪽은 맨 아래의
두 시험과 `test_fixture_detection_reason.py` 의 `broken_span_*` 가 본다.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import validator.validate_dataset as vd
from tests.validator.loader_casts import LEAVES, MEASURED, TYPES, candidate_values
from validator.make_fixture import VARIANTS, build

LOADED = "*/*.parquet"          # what LeRobot loads of data/ and meta/episodes/ (io_utils.load_nested_dataset)


def _read_release(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Path], dict]:
    """(episode table, step table in stored order, the data files, info.json)."""
    files = sorted((root / "data").glob(LOADED))
    steps = pd.concat([pd.read_parquet(f).assign(_file=k) for k, f in enumerate(files)],
                      ignore_index=True)
    eps = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))],
                    ignore_index=True)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    return eps, steps, files, info


# ── 픽스처는 배포본과 같은 표현이어야 한다 ─────────────────────────

@pytest.mark.parametrize("with_media", [False, True])
def test_the_fixture_points_each_episode_at_its_rows_as_lerobot_writes_them(tmp_path, with_media):
    """포인터 열이 없는 픽스처로는 포인터 검사의 통과 쪽을 시험하지 못한다.

    기대값은 적어 두지 않고 저장된 자리에서 읽는다 — 에피소드 k 의 행이 놓인 자리가 곧
    LeRobot 이 적는 구간이다(dataset_writer: 앞 에피소드의 dataset_to_index 에서 시작).
    """
    root = tmp_path / "clean"
    build(root, with_media=with_media)
    eps, steps, files, info = _read_release(root)

    assert steps["index"].tolist() == list(range(len(steps))), "index 는 스텝 표에서의 자리다"
    assert len(files) >= 2, "파일이 하나면 data/file_index 가 무엇을 가리켜도 같다"
    for k, row in eps.iterrows():
        at = np.flatnonzero(steps["episode_index"].to_numpy() == row["episode_index"])
        assert row["episode_index"] == k
        assert at.tolist() == list(range(at[0], at[-1] + 1)), f"에피소드 {k} 의 행이 흩어져 있다"
        assert (row["dataset_from_index"], row["dataset_to_index"]) == (at[0], at[-1] + 1)
        assert row["length"] == len(at)
        named = info["data_path"].format(chunk_index=row["data/chunk_index"],
                                         file_index=row["data/file_index"])
        held = {files[j].relative_to(root).as_posix() for j in steps["_file"].iloc[at].unique()}
        assert held == {named}, f"에피소드 {k}: {named} 을 가리키는데 행은 {sorted(held)} 에 있다"


@pytest.mark.parametrize("with_media", [False, True])
def test_the_fixture_keeps_its_tables_where_lerobot_reads_them(tmp_path, with_media):
    """에피소드 표를 meta/episodes 바로 아래에 두던 동안, 픽스처는 적재기가 열지 못하는 꼴이었고
    검증기가 「적재기가 안 읽는 자리의 표」를 말할 수도 없었다 — 말하면 픽스처부터 걸린다."""
    root = tmp_path / "clean"
    build(root, with_media=with_media)
    for table in ("data", "meta/episodes"):
        every = sorted((root / table).rglob("*.parquet"))
        assert every and every == sorted((root / table).glob(LOADED)), [str(f) for f in every]


# ── 손으로 지은 배포본 ─────────────────────────────────────────────

EPISODES = "meta/episodes/chunk-000/file-000.parquet"
FILE_0, FILE_1 = (vd.DATA_PATH.format(chunk_index=0, file_index=k) for k in (0, 1))
LENGTHS = (3, 4, 2, 5)          # rows 0..2 · 3..6 in FILE_0, 7..8 · 9..13 in FILE_1

# The sentences. Each is said for one kind of defect; the numbers of a finding are in its location.
STRAY_FILE = "LeRobot 이 읽지 않는 자리의 parquet"
STRAY_TABLE = "LeRobot 이 읽지 않는 자리의 에피소드 표"
NO_TABLE = "LeRobot 이 읽는 에피소드 표가 없음"
NO_STEP_TABLE = "LeRobot 이 읽는 스텝 표가 없음"
NO_STEP_COLUMN = "열이 없어 에피소드가 가리키는 행을 대조하지 못함"
NOT_INTEGER_STEPS = "열이 정수 열이 아니어서"
INDEX = "index 열이 행의 자리와 다름"
NO_POINTER_COLUMN = "열이 없어 에피소드의 행을 찾을 수 없음"
TABLE_ORDER = "LeRobot 은 k번 에피소드의 포인터를 k번째 행에서 읽는다"
NOT_AN_INDEX = "LeRobot 은 이 값을 그대로 자리·파일 번호로 쓴다"
NOT_A_SPAN = "dataset_from_index..dataset_to_index 가 스텝 표 안의 구간이 아님"
LENGTH = "length 가 가리키는 구간의 행 수와 다름"
NOT_ITS_ROWS = "가리키는 행이 이 에피소드의 것이 아님"
FRAME_ORDER = "가리키는 행의 frame_index 가 0 부터 차례로 이어지지 않음"
NO_SUCH_FILE = "data/chunk_index·data/file_index 가 가리키는 파일이 없음"
OTHER_FILE = "행이 data/chunk_index·data/file_index 가 가리키는 파일에 있지 않음"
TEMPLATE = "data_path 를 파일 이름으로 풀 수 없음"
NO_NAME_FOR_EPISODE = "data_path 를 이 에피소드의 data/chunk_index·data/file_index 로 풀 수 없음"
ORPHANS = "어느 에피소드도 가리키지 않는 스텝 행"
# c7_table_files: whether a loader gets a table at all.
EMPTY_FIRST_GROUP = "첫 행 그룹이 0행인 parquet"
POINTER_TYPES = "포인터 열의 형이 파일마다 다름"
NEVER_FITS = "첫 파일의 형에 맞출 수 없음"
FIRST_LACKS = "첫 파일에 없는 열이 뒤 파일에 있음"
DUPLICATE_NAMES = "같은 이름의 열이 둘 이상"
DECLARED_NOT_INTEGER = "정수 하나(dtype int8~uint64 · shape [1])로 선언하지 않음"
PATTERN_NAME = "datasets 가 패턴으로 읽는 글자"
SENTENCES = (STRAY_FILE, STRAY_TABLE, NO_TABLE, NO_STEP_TABLE, NO_STEP_COLUMN, NOT_INTEGER_STEPS, INDEX, NO_POINTER_COLUMN, TABLE_ORDER,
             NOT_AN_INDEX, NOT_A_SPAN, LENGTH, NOT_ITS_ROWS, FRAME_ORDER, NO_SUCH_FILE, OTHER_FILE,
             TEMPLATE, NO_NAME_FOR_EPISODE, ORPHANS, EMPTY_FIRST_GROUP, POINTER_TYPES, NEVER_FITS, FIRST_LACKS,
             DUPLICATE_NAMES, DECLARED_NOT_INTEGER, PATTERN_NAME)

OMIT = object()


def _release(root: Path, data_path=vd.DATA_PATH) -> Path:
    """Four episodes, two to a step file, laid out as LeRobot lays a release out."""
    eps, parts, start = [], [], 0
    for k, n in enumerate(LENGTHS):
        eps.append({"episode_index": k, "length": n, "data/chunk_index": 0, "data/file_index": k // 2,
                    "dataset_from_index": start, "dataset_to_index": start + n})
        parts.append(pd.DataFrame({"index": np.arange(start, start + n), "episode_index": k,
                                   "frame_index": np.arange(n)}))
        start += n
    for rel, held in ((FILE_0, parts[:2]), (FILE_1, parts[2:])):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(held, ignore_index=True).to_parquet(root / rel, index=False)
    (root / EPISODES).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(eps).to_parquet(root / EPISODES, index=False)
    info = {"codebase_version": "v3.0", "fps": 50}
    if data_path is not OMIT:
        info["data_path"] = data_path
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return root


def _edit(root: Path, rel: str, fn) -> None:
    df = pd.read_parquet(root / rel)
    out = fn(df)
    (df if out is None else out).to_parquet(root / rel, index=False)


def _set(k: int, **values):
    """Episode row k's columns to these values ("data/file_index" is spelled file_index)."""
    names = {"file_index": "data/file_index", "chunk_index": "data/chunk_index"}

    def fn(df):
        for column, value in values.items():
            df.loc[k, names.get(column, column)] = value
    return fn


def _report(root: Path, file_limit=None) -> vd.Report:
    """The check, fed by the validator's own readers."""
    rep = vd.Report(root=str(root))
    ds = vd.Dataset(root, rep)
    frames = vd.load_frames(ds, rep, file_limit)
    vd.c7_data_spans(ds, frames, ds.episodes(), rep)
    return rep


def _said(rep: vd.Report, *sentences: str) -> list[vd.Finding]:
    """The report's ERRORs are these sentences, one finding each, and nothing else."""
    errors = [f for f in rep.findings if f.severity == vd.ERROR]
    lines = [f.line() for f in errors]
    assert all(f.check == "C7" for f in errors), lines
    kinds = [next((s for s in SENTENCES if s in f.message), f.message) for f in errors]
    assert sorted(kinds) == sorted(sentences), lines
    return errors


def test_a_release_as_lerobot_writes_it_has_nothing_to_report(tmp_path):
    rep = _report(_release(tmp_path))
    assert rep.findings == [], [f.line() for f in rep.findings]
    # Nothing said is only worth something beside how much was looked at.
    assert rep.stats["data_spans"] == {"episodes": len(LENGTHS), "rows": sum(LENGTHS)}


def test_two_episodes_given_each_others_spans_are_both_reported(tmp_path):
    root = _release(tmp_path)
    _edit(root, EPISODES, _set(0, dataset_from_index=3, dataset_to_index=7, length=4))
    _edit(root, EPISODES, _set(1, dataset_from_index=0, dataset_to_index=3, length=3))
    first, second = _said(_report(root), NOT_ITS_ROWS, NOT_ITS_ROWS)
    assert "행 3..7 의 episode_index [1]" in first.location, first.location
    assert "행 0..3 의 episode_index [0]" in second.location, second.location
    # One sentence for the kind, the numbers in the location: the report folds equal messages
    # into a line, and a shift that puts hundreds of episodes off must not bury the rest.
    assert first.message == second.message


def test_a_span_one_row_short_is_reported_with_the_row_left_over(tmp_path):
    root = _release(tmp_path)
    _edit(root, EPISODES, _set(1, dataset_to_index=6))
    errors = _said(_report(root), LENGTH, ORPHANS)
    assert "length 4, 3..6 는 3행" in errors[0].location, errors[0].location
    assert "스텝 행 1개" in errors[1].message and "첫 행 6" in errors[1].location, errors[1].line()


def test_a_row_no_episode_points_at_is_reported(tmp_path):
    # Consistent in itself -- the last episode is one row shorter in length and span alike --
    # so only the row left over says anything.
    root = _release(tmp_path)
    _edit(root, EPISODES, _set(3, dataset_to_index=13, length=4))
    (orphan,) = _said(_report(root), ORPHANS)
    assert "스텝 행 1개" in orphan.message and FILE_1 in orphan.location and "첫 행 13" in orphan.location


def test_a_span_one_row_long_takes_a_row_of_the_next_episode(tmp_path):
    root = _release(tmp_path)
    _edit(root, EPISODES, _set(0, dataset_to_index=4, length=4))
    (f,) = _said(_report(root), NOT_ITS_ROWS)
    assert "행 0..4 의 episode_index [0, 1]" in f.location, f.location


@pytest.mark.parametrize("values, shown", [
    (dict(dataset_to_index=15, length=6), "9..15 (스텝 표 14행)"),      # past the end
    (dict(dataset_from_index=14, dataset_to_index=14, length=0), "14..14"),   # empty
    (dict(dataset_from_index=14, dataset_to_index=9), "14..9"),          # reversed
    (dict(dataset_from_index=-1, length=15), "-1..14"),                  # before the start
])
def test_a_span_that_is_not_a_stretch_of_the_table_is_reported(tmp_path, values, shown):
    root = _release(tmp_path)
    _edit(root, EPISODES, _set(3, **values))
    (f,) = _said(_report(root), NOT_A_SPAN)
    assert shown in f.location, f.location


def _with_a_null(column):
    return column.astype(float).where(column.index != 2)


NOT_INTEGERS = {
    "a null (read back as NaN among floats)": _with_a_null,
    "a fraction": lambda column: column + 0.5,
    "whole numbers kept as floats": lambda column: column.astype(float),
    "strings": lambda column: column.astype(str),
    "booleans": lambda column: column.astype(bool),
}


@pytest.mark.parametrize("column", ["dataset_from_index", "dataset_to_index", "length",
                                    "data/chunk_index", "data/file_index"])
@pytest.mark.parametrize("kind", sorted(NOT_INTEGERS))
def test_a_pointer_column_that_is_not_of_integers_is_reported_not_crashed_on(tmp_path, column, kind):
    """LeRobot 은 이 값을 그대로 자리로 쓴다 — 3.0 도 '3' 도 자리가 아니다. 값이 정수로 읽히는지가
    아니라 열이 정수 열인지를 본다."""
    root = _release(tmp_path)
    _edit(root, EPISODES, lambda df: df.assign(**{column: NOT_INTEGERS[kind](df[column])}))
    (f,) = _said(_report(root), NOT_AN_INDEX)
    assert f"{column} 열" in f.message and f.location == "meta/episodes", f.line()


def test_the_file_pointer_has_to_name_the_file_the_rows_are_in(tmp_path):
    root = _release(tmp_path)
    _edit(root, EPISODES, _set(2, file_index=0))
    (f,) = _said(_report(root), OTHER_FILE)
    assert f.location == f"episode_index=2 · 가리킨 파일 {FILE_0}, 행이 있는 파일 ['{FILE_1}']", f.location


@pytest.mark.parametrize("pointer", [dict(file_index=7), dict(file_index=-1), dict(chunk_index=1)])
def test_a_file_pointer_to_a_file_that_is_not_there_is_reported(tmp_path, pointer):
    root = _release(tmp_path)
    _edit(root, EPISODES, _set(2, **pointer))
    (f,) = _said(_report(root), NO_SUCH_FILE)
    named = vd.DATA_PATH.format(**{"chunk_index": 0, "file_index": 1, **pointer})
    assert f.location == f"episode_index=2 · {named}", f.location


def _swap_two_frames(df):
    df.loc[[4, 5], "frame_index"] = df.loc[[5, 4], "frame_index"].to_numpy()


def _start_at_five(df):
    df.loc[df["episode_index"] == 1, "frame_index"] += 5


@pytest.mark.parametrize("change, shown", [(_swap_two_frames, "1번째가 2"), (_start_at_five, "0번째가 5")])
def test_frames_that_do_not_count_up_from_zero_inside_a_span_are_reported(tmp_path, change, shown):
    """C7 의 프레임 연속성 검사는 첫 값에서부터 센다 — 0 이 아닌 데서 시작해도 통과한다."""
    root = _release(tmp_path)
    _edit(root, FILE_0, change)
    (f,) = _said(_report(root), FRAME_ORDER)
    assert f.location.startswith("episode_index=1") and shown in f.location, f.location


def test_an_index_column_that_is_not_the_rows_place_is_reported_once(tmp_path):
    root = _release(tmp_path)
    _edit(root, FILE_1, lambda df: df.assign(index=df["index"] + 1))
    (f,) = _said(_report(root), INDEX)
    assert "7번째 행이 8 (어긋난 행 7개)" in f.message and f.location == FILE_1, f.line()


def test_two_episodes_stored_in_each_others_place_are_reported(tmp_path):
    """행이 옮겨졌고 index 는 새 자리대로 다시 매겨졌다 — 행을 episode_index 로 묶어 읽는 검사는
    모두 통과한다."""
    root = _release(tmp_path)

    def swap_blocks(df):
        moved = pd.concat([df[df["episode_index"] == 1], df[df["episode_index"] == 0]], ignore_index=True)
        return moved.assign(index=np.arange(len(moved)))

    _edit(root, FILE_0, swap_blocks)
    first, second = _said(_report(root), NOT_ITS_ROWS, NOT_ITS_ROWS)
    assert "행 0..3 의 episode_index [1]" in first.location, first.location
    assert "행 3..7 의 episode_index [0, 1]" in second.location, second.location


def test_the_episode_table_is_read_by_place_as_a_loader_reads_it(tmp_path):
    """표의 두 행이 자리를 바꿨다. 행마다는 제 구간을 옳게 적고 있어, episode_index 로 찾아 읽으면
    아무 문제가 없어 보인다 — 적재기는 1번 에피소드의 구간을 둘째 행에서 읽는다."""
    root = _release(tmp_path)
    _edit(root, EPISODES, lambda df: df.iloc[[0, 2, 1, 3]].reset_index(drop=True))
    (f,) = _said(_report(root), TABLE_ORDER)
    assert "1번째 행이 episode_index 2 (어긋난 행 2개)" in f.message, f.message


@pytest.mark.parametrize("column", ["episode_index", "length", "dataset_from_index", "dataset_to_index",
                                    "data/chunk_index", "data/file_index"])
def test_a_missing_pointer_column_is_reported_by_name(tmp_path, column):
    root = _release(tmp_path)
    _edit(root, EPISODES, lambda df: df.drop(columns=[column]))
    (f,) = _said(_report(root), NO_POINTER_COLUMN)
    assert f"['{column}']" in f.message, f.message


@pytest.mark.parametrize("column", ["index", "episode_index", "frame_index"])
def test_a_missing_step_column_is_reported_not_crashed_on(tmp_path, column):
    root = _release(tmp_path)
    _edit(root, FILE_1, lambda df: df.drop(columns=[column]))
    (f,) = _said(_report(root), NO_STEP_COLUMN)
    assert f"['{column}']" in f.message and f.location == FILE_1, f.line()


STEP_VALUES_THAT_ARE_NOT_INTEGERS = {
    "whole numbers kept as floats": lambda column: column.astype(float),
    "strings": lambda column: column.astype(str),
    "lists": lambda column: [[int(v)] for v in column],
}


@pytest.mark.parametrize("column", ["index", "episode_index", "frame_index"])
@pytest.mark.parametrize("kind", sorted(STEP_VALUES_THAT_ARE_NOT_INTEGERS))
def test_a_step_column_that_is_not_of_integers_is_reported_not_crashed_on(tmp_path, column, kind):
    root = _release(tmp_path)
    _edit(root, FILE_1, lambda df: df.assign(
        **{column: STEP_VALUES_THAT_ARE_NOT_INTEGERS[kind](df[column])}))
    (f,) = _said(_report(root), NOT_INTEGER_STEPS)
    assert f"{column} 열" in f.message and f.location == FILE_1, f.line()


def _chunk_one_linked_from_outside(root: Path) -> None:
    """The second half of both tables in chunk-001, and chunk-001 a symlink to a directory outside
    the table -- as a release unpacked onto two disks would have it."""
    for table, rel, rows in (("data", FILE_1, None), ("meta/episodes", EPISODES, [2, 3])):
        outside = root.parent / "elsewhere" / table.replace("/", "-")
        outside.mkdir(parents=True)
        if rows is None:
            shutil.move(root / rel, outside / "file-001.parquet")
        else:
            eps = pd.read_parquet(root / rel)
            eps.loc[rows, "data/chunk_index"] = 1
            eps.drop(index=rows).to_parquet(root / rel, index=False)
            eps.loc[rows].to_parquet(outside / "file-000.parquet", index=False)
        (root / table / "chunk-001").symlink_to(outside, target_is_directory=True)


def test_a_chunk_directory_that_is_a_symlink_is_read_as_the_loader_reads_it(tmp_path):
    """LeRobot 이 표의 파일을 찾는 `glob("*/*.parquet")` 은 심링크인 청크 디렉터리 안으로 들어간다
    (실제 적재기로 확인 — 열 행을 모두 내준다). 재귀 탐색(`rglob`)은 들어가지 않아, 그것으로 읽으면
    표의 절반만 보고 「전부 대조 · ERROR 0」 이라고 하거나 남은 절반의 포인터를 틀렸다고 한다."""
    root = _release(tmp_path / "release")
    _chunk_one_linked_from_outside(root)
    assert sorted(f.name for f in (root / "data").rglob("*.parquet")) == ["file-000.parquet"], "the case is gone"

    rep = _report(root)
    assert rep.findings == [], [f.line() for f in rep.findings]
    assert rep.stats["data_spans"] == {"episodes": len(LENGTHS), "rows": sum(LENGTHS)}
    files = _files_report(root)
    assert files.findings == [], [f.line() for f in files.findings]
    assert files.stats["table_files"] == {"data": 2, "meta/episodes": 2}


def test_a_table_that_is_whole_only_by_a_file_the_loader_does_not_read_is_not_whole(tmp_path):
    """적재기는 data/<chunk>/<file>.parquet 만 읽는다. 그 밖의 자리로 옮겨진 파일의 행을 세어 주면
    적재기가 받지 못하는 표를 온전하다고 말하게 된다."""
    root = _release(tmp_path)
    (root / FILE_1).rename(root / "data" / "file-001.parquet")
    (stray,) = _said(_files_report(root), STRAY_FILE)
    assert stray.location == "data/file-001.parquet", stray.location
    errors = _said(_report(root), NOT_A_SPAN, NOT_A_SPAN)       # the pointers, held to the seven rows a loader has
    assert "(스텝 표 7행)" in errors[0].location, errors[0].location


def test_a_stray_parquet_does_not_shift_the_rows_the_loader_reads(tmp_path):
    # Sorted with the rest it comes first ("extra" < "file-000.parquet"), and counted in it
    # would push every row seven places on.
    root = _release(tmp_path)
    stray = root / "data" / "chunk-000" / "extra" / "file-000.parquet"
    stray.parent.mkdir()
    stray.write_bytes((root / FILE_0).read_bytes())
    (f,) = _said(_files_report(root), STRAY_FILE)
    assert f.location == "data/chunk-000/extra/file-000.parquet", f.location
    rep = _report(root)
    assert rep.findings == [] and rep.stats["data_spans"] == {"episodes": len(LENGTHS), "rows": sum(LENGTHS)}


@pytest.mark.parametrize("where", ["aaa.parquet", "chunk-000/deeper/file-000.parquet", "zzz.parquet"])
def test_an_episode_row_where_the_loader_does_not_look_is_not_part_of_the_table(tmp_path, where):
    """에피소드 0 의 행이 적재기가 읽지 않는 파일로 옮겨졌다. 그 파일까지 읽으면 앞의 둘은 표가
    온전하고 제자리다(이름 순서로 앞에 온다) — 적재기의 표는 에피소드 1 에서 시작하고, 0번 에피소드의
    구간을 그 행에서 읽는다. 셋째는 이름 순서로 뒤에 와 행이 1, 2, 3, 0 으로 읽히는데, 그 순서는
    어느 적재기의 표도 아니므로 거기에 대해서는 더 말하지 않는다."""
    root = _release(tmp_path)
    table = pd.read_parquet(root / EPISODES)
    stray = root / "meta" / "episodes" / where
    stray.parent.mkdir(parents=True, exist_ok=True)
    table.iloc[:1].to_parquet(stray, index=False)
    table.iloc[1:].to_parquet(root / EPISODES, index=False)
    (f,) = _said(_files_report(root), STRAY_TABLE)
    assert f.location == f"meta/episodes/{where}", f.location
    rep = _report(root)                                          # the pointers: nothing compared, nothing said
    assert rep.findings == [] and "data_spans" not in rep.stats, [x.line() for x in rep.findings]


def _as_json(root: Path) -> None:
    rows = pd.read_parquet(root / EPISODES).to_dict("records")
    (root / "meta" / "episodes.json").write_text(json.dumps(rows), encoding="utf-8")


def _as_jsonl(root: Path) -> None:
    rows = pd.read_parquet(root / EPISODES).to_dict("records")
    (root / "meta" / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.mark.parametrize("rewrite", [_as_json, _as_jsonl])
def test_an_episode_table_only_the_validators_fallback_reads_is_no_table_for_a_loader(tmp_path, rewrite):
    """검증기는 `meta/episodes*.json` 을 대신 읽어 준다(옛 배치의 흔적). 거기에 포인터까지 옳게 적혀 있으면
    전부 대조했다고 말하게 되는데, 적재기는 parquet 이 없다며 열지 못한다."""
    root = _release(tmp_path)
    rewrite(root)
    shutil.rmtree(root / "meta" / "episodes")
    (f,) = _said(_files_report(root), NO_TABLE)
    assert f.location == "meta/episodes", f.line()
    rep = _report(root)                                          # the pointers: nothing compared, nothing said
    assert rep.findings == [] and "data_spans" not in rep.stats, [x.line() for x in rep.findings]


def test_an_episode_table_kept_only_where_the_loader_does_not_look_is_no_table_either(tmp_path):
    root = _release(tmp_path)
    (root / EPISODES).rename(root / "meta" / "episodes" / "file-000.parquet")
    _said(_files_report(root), STRAY_TABLE, NO_TABLE)
    assert _report(root).findings == []


def test_no_table_is_said_whether_or_not_a_step_was_read(tmp_path):
    """포인터 대조는 읽은 스텝이 없으면 아무것도 하지 않고 돌아간다. 「표가 없다」를 거기서 말하던 동안,
    표도 스텝도 없는 배포본은 C7 만 돌리면 SKIP · exit 0 이었다."""
    root = _release(tmp_path)
    _as_json(root)
    shutil.rmtree(root / "meta" / "episodes")
    shutil.rmtree(root / "data")
    rep = _files_report(root)
    where = {f.location: f.message for f in _said(rep, NO_TABLE, NO_STEP_TABLE)}
    assert NO_TABLE in where["meta/episodes"] and NO_STEP_TABLE in where["data"], where
    assert rep.stats["table_files"] == {"data": 0, "meta/episodes": 0}
    assert vd.main([str(root), "--checks", "C7"]) == 1


def _split_episode_table(root: Path, names: tuple[str, ...], rows: tuple[list[int], ...]) -> list[Path]:
    """The episode table in several files of chunk-000: file ``names[i]`` holds ``rows[i]``."""
    table = pd.read_parquet(root / EPISODES)
    (root / EPISODES).unlink()
    files = [root / "meta" / "episodes" / "chunk-000" / f"{name}.parquet" for name in names]
    for file, held in zip(files, rows):
        table.iloc[held].to_parquet(file, index=False)
    return files


# The loader takes the files in name order -- sorted(dir.glob("*/*.parquet")), names as text: file-10
# comes before file-2. Episode k is the k-th row of the files read in that order.
SEVERAL_FILES = {
    "two files": (("file-000", "file-001"), ([0, 1], [2, 3])),
    "names that sort as text, not as numbers": (("file-1", "file-10", "file-2"), ([0], [1], [2, 3])),
}


@pytest.mark.parametrize("layout", sorted(SEVERAL_FILES))
def test_an_episode_table_in_several_files_is_read_in_name_order(tmp_path, layout):
    """표가 한 파일뿐인 사례만 있던 동안, 파일을 이름 역순으로 읽게 바꿔도 시험이 모두 초록이었다."""
    root = _release(tmp_path)
    _split_episode_table(root, *SEVERAL_FILES[layout])
    rep = _report(root)
    assert rep.findings == [], [f.line() for f in rep.findings]
    assert rep.stats["data_spans"] == {"episodes": len(LENGTHS), "rows": sum(LENGTHS)}


def test_episode_files_under_each_others_name_are_reported(tmp_path):
    root = _release(tmp_path)
    first, second = _split_episode_table(root, ("file-000", "file-001"), ([2, 3], [0, 1]))
    (f,) = _said(_report(root), TABLE_ORDER)
    assert "0번째 행이 episode_index 2 (어긋난 행 4개)" in f.message, f.message


def test_a_file_limit_run_compares_what_it_read_and_says_what_it_did_not(tmp_path):
    rep = _report(_release(tmp_path), file_limit=1)
    _said(rep)
    (note,) = rep.findings
    assert note.severity == vd.INFO and "에피소드 2개" in note.message and "--file-limit" in note.message
    assert rep.stats["data_spans"] == {"episodes": 2, "rows": 7}


def test_a_file_limit_run_still_reports_the_episodes_it_read(tmp_path):
    root = _release(tmp_path)
    _edit(root, EPISODES, _set(0, dataset_to_index=4, length=4))
    _said(_report(root, file_limit=1), NOT_ITS_ROWS)


def test_a_file_limit_run_holds_a_read_episode_to_its_file_pointer(tmp_path):
    # The file named is there, only not read: the rows are still not in it.
    root = _release(tmp_path)
    _edit(root, EPISODES, _set(0, file_index=1))
    (f,) = _said(_report(root, file_limit=1), OTHER_FILE)
    assert f.location == f"episode_index=0 · 가리킨 파일 {FILE_1}, 행이 있는 파일 ['{FILE_0}']", f.location


def test_a_file_limit_run_does_not_call_the_read_part_of_an_unread_stretch_left_over(tmp_path):
    # Episode 2's stretch made to start a row early: rows 6..9, of which only row 6 was read.
    # Whether it is right cannot be said from the files read, and row 6 is not "no episode's".
    root = _release(tmp_path)
    _edit(root, EPISODES, _set(1, dataset_to_index=6, length=3))
    _edit(root, EPISODES, _set(2, dataset_from_index=6, length=3))
    rep = _report(root, file_limit=1)
    _said(rep)
    assert rep.stats["data_spans"] == {"episodes": 2, "rows": 6}     # row 6 was not compared
    _said(_report(root), NOT_ITS_ROWS, OTHER_FILE)          # read whole, it is said


def test_without_a_data_path_the_v3_template_is_used(tmp_path):
    root = _release(tmp_path, data_path=OMIT)
    assert _report(root).findings == []
    _edit(root, EPISODES, _set(2, file_index=0))
    _said(_report(root), OTHER_FILE)


@pytest.mark.parametrize("data_path", ["data/{nope}.parquet", "data/{chunk_index:q}.parquet",
                                       "data/{0}.parquet", 7, ["data"], ""])
def test_a_data_path_that_cannot_be_filled_in_is_reported_once(tmp_path, data_path):
    (f,) = _said(_report(_release(tmp_path, data_path=data_path)), TEMPLATE)
    assert f.location == "meta/info.json", f.location


def test_a_data_path_that_fails_on_an_episodes_own_numbers_is_reported_for_that_episode(tmp_path):
    """틀이 (0, 0) 으로는 풀리고 실제 포인터 값으로는 안 풀린다. 풀리지 않은 에피소드의 파일 대조를 말없이
    건너뛰면, 적재기가 `get_data_file_path` 에서 죽는 배포본이 ERROR 0 이다."""
    # The zero-padding width comes from chunk_index: 3 gives file-000 · file-001, the release's own names.
    root = _release(tmp_path, data_path="data/chunk-000/file-{file_index:0{chunk_index}d}.parquet")
    _edit(root, EPISODES, lambda df: df.assign(**{"data/chunk_index": 3}))
    assert _report(root).findings == []
    _edit(root, EPISODES, _set(2, chunk_index=-1))               # "0-1d" is no format
    (f,) = _said(_report(root), NO_NAME_FOR_EPISODE)
    assert f.location == "episode_index=2 · data/chunk_index -1, data/file_index 1", f.location


def test_a_number_the_template_cannot_take_is_reported_not_crashed_on(tmp_path):
    """`{file_index:c}` 에 2**63 은 OverflowError 다 — 잡지 않으면 검증기가 exit 3 으로 끝나고, 그것은
    「막았다」가 아니라 「검사하지 못했다」다."""
    root = _release(tmp_path, data_path="data/chunk-000/file-{file_index:c}.parquet")

    def huge(df):
        out = df.assign(**{"data/file_index": df["data/file_index"].astype("uint64")})
        out.loc[2, "data/file_index"] = 2 ** 63
        return out

    _edit(root, EPISODES, huge)
    errors = _said(_report(root), NO_NAME_FOR_EPISODE, *[NO_SUCH_FILE] * 3)   # the others name "file-\x00" and "file-\x01"
    (said,) = [f for f in errors if NO_NAME_FOR_EPISODE in f.message]
    assert said.location == f"episode_index=2 · data/chunk_index 0, data/file_index {2 ** 63}", said.location


def test_a_file_name_the_file_system_refuses_names_no_file(tmp_path):
    # A path component of 300 characters: is_file() raises OSError (ENAMETOOLONG) instead of saying no.
    root = _release(tmp_path, data_path="data/" + "x" * 300 + "/file-{file_index:03d}.parquet")
    _said(_report(root), *[NO_SUCH_FILE] * len(LENGTHS))


def test_the_command_line_reports_these_and_does_not_die(tmp_path):
    """exit 1 은 결함을 보고했다는 뜻이고 exit 3 은 검증기가 죽었다는 뜻이다. 위 둘을 실행 파일로."""
    for name, data_path in (("long", "data/" + "x" * 300 + "/file-{file_index:03d}.parquet"),
                            ("char", "data/chunk-000/file-{file_index:c}.parquet")):
        root = _release(tmp_path / name, data_path=data_path)
        _edit(root, EPISODES, lambda df: df.assign(**{"data/file_index": df["data/file_index"].astype("uint64") + 2 ** 63}))
        assert vd.main([str(root), "--checks", "C7"]) == 1, name


def test_a_data_path_that_leaves_the_release_names_no_file(tmp_path):
    # The files it names are there, beside the release: followed out of the root they would read
    # as "there, only not the file the rows are in".
    root = _release(tmp_path / "release", data_path="../" + vd.DATA_PATH)
    shutil.copytree(root / "data", tmp_path / "data")
    assert (root / "../" / FILE_0).is_file()
    _said(_report(root), *[NO_SUCH_FILE] * len(LENGTHS))


def test_nothing_is_said_where_another_check_already_said_there_is_nothing_to_read(tmp_path):
    root = _release(tmp_path)
    rep = vd.Report(root=str(root))
    ds = vd.Dataset(root, rep)
    vd.c7_data_spans(ds, [], ds.episodes(), rep)                        # no step file: C1, C7
    vd.c7_data_spans(ds, vd.load_frames(ds, rep, None), None, rep)      # no episode table: C1
    assert rep.findings == []


# ── 적재기가 표를 여는가 (c7_table_files) ─────────────────────────
#
# 포인터가 옳은지보다 앞에 있는 물음이다. 아래 셋은 적재기를 멈추는데 pandas 로 읽으면 보이지 않아,
# 포인터 대조는 「전부 대조 · ERROR 0」 이라고 말한다(실제 적재기로 재현함).
# 사례마다 그 사실을 함께 못박는다.

def _files_report(root: Path) -> vd.Report:
    rep = vd.Report(root=str(root))
    vd.c7_table_files(vd.Dataset(root, rep), rep)
    return rep


def _spans_see_nothing(root: Path) -> None:
    rep = _report(root)
    assert [f.line() for f in rep.findings if f.severity == vd.ERROR] == []


def _write(table, path: Path, *more) -> None:
    """``table`` and then ``more`` as successive row groups of one parquet."""
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, table.schema) as writer:
        for part in (table, *more):
            writer.write_table(part)


def _arrow(root: Path, rel: str):
    import pyarrow.parquet as pq

    return pq.read_table(root / rel)


def test_a_release_as_lerobot_writes_it_has_files_a_loader_reads(tmp_path):
    rep = _files_report(_release(tmp_path))
    assert rep.findings == [], [f.line() for f in rep.findings]
    assert rep.stats["table_files"] == {"data": 2, "meta/episodes": 1}


@pytest.mark.parametrize("like, extra", [(FILE_0, "data/chunk-000/file-009.parquet"),
                                         (EPISODES, "meta/episodes/chunk-000/file-009.parquet")])
def test_an_empty_table_saved_among_the_files_stops_the_loader(tmp_path, like, extra):
    """빈 표를 저장하면 0행짜리 행 그룹 하나가 된다. datasets 는 파일의 첫 행 그룹의 행 수를 배치 크기로
    쓰고 0 에서 실패한다. 행을 이어 붙여 읽으면 그 파일은 그냥 사라진다."""
    root = _release(tmp_path)
    _write(_arrow(root, like).slice(0, 0), root / extra)
    (f,) = _said(_files_report(root), EMPTY_FIRST_GROUP)
    assert f.location == extra, f.location
    _spans_see_nothing(root)


def _last_episode_in_a_file_named(name: str):
    def change(root: Path) -> list[str]:
        eps = pd.read_parquet(root / EPISODES)
        eps.iloc[:-1].to_parquet(root / EPISODES, index=False)
        eps.iloc[-1:].to_parquet((root / EPISODES).with_name(name), index=False)
        return [str(Path(EPISODES).with_name(name))]
    return change


def _second_step_file_named(name: str):
    def change(root: Path) -> list[str]:
        (root / FILE_1).rename((root / FILE_1).with_name(name))
        return [str(Path(FILE_1).with_name(name))]
    return change


def _step_chunk_named(name: str):
    def change(root: Path) -> list[str]:
        (root / FILE_0).parent.rename((root / FILE_0).parent.with_name(name))
        return [f"data/{name}/{Path(rel).name}" for rel in (FILE_0, FILE_1)]
    return change


PATTERN_NAMES = {
    # Measured with the loader: this file is passed over (the class [1] matches "1", not "[1]") and
    # the loader is left with three episode rows -- while every file counts and compares as whole.
    "an episode file with a class": _last_episode_in_a_file_named("file-[1]00.parquet"),
    # Measured: a neighbour the pattern matches is read a second time.
    "a step file with a star": _second_step_file_named("file-*.parquet"),
    "a step file with a question mark": _second_step_file_named("file-00?.parquet"),
    "a step chunk with a class": _step_chunk_named("chunk-00[0]"),
}


@pytest.mark.parametrize("case", sorted(PATTERN_NAMES))
def test_a_table_file_whose_path_datasets_takes_for_a_pattern_is_reported(tmp_path, case):
    """LeRobot 은 표의 파일 경로를 `datasets` 에 넘기고, `datasets` 는 `*` · `?` · `[` 가 든 경로를 이름이
    아니라 패턴으로 푼다 — 그 파일을 건너뛰거나 이웃 파일을 한 번 더 읽는다(실제 적재기로 확인).
    파일은 제자리에 온전히 있으므로 행 수도 포인터도 「전부 대조」 로 지나간다."""
    root = _release(tmp_path)
    named = PATTERN_NAMES[case](root)
    found = _said(_files_report(root), *[PATTERN_NAME] * len(named))
    assert sorted(f.location.split(" · ")[0] for f in found) == sorted(named), [f.location for f in found]


@pytest.mark.parametrize("character", list("]{}!#$%&'()+,;=@^~ "))
def test_a_character_datasets_reads_as_itself_is_not_called_a_pattern(tmp_path, character):
    # Measured with the loader, one name for each: read by name, like any other file.
    root = _release(tmp_path)
    _second_step_file_named(f"file-0{character}1.parquet")(root)
    _said(_files_report(root))


def test_it_is_the_first_row_group_that_counts(tmp_path):
    # Measured with the loader's reader: an empty first group fails even with rows after it; an empty
    # group after rows, or no group at all, loads.
    root = _release(tmp_path)
    rows = _arrow(root, FILE_1)
    _write(rows.slice(0, 0), root / FILE_1, rows)                           # empty group, then the rows
    (f,) = _said(_files_report(root), EMPTY_FIRST_GROUP)
    assert f.location == FILE_1, f.location
    _write(rows, root / FILE_1, rows.slice(0, 0))                           # the rows, then an empty group
    _write(rows.slice(0, 0).slice(0, 0), root / "data/chunk-000/file-009.parquet")
    import pyarrow.parquet as pq
    with pq.ParquetWriter(root / "data/chunk-000/file-009.parquet", rows.schema):
        pass                                                                # a file with no row group
    assert pq.read_metadata(root / "data/chunk-000/file-009.parquet").num_row_groups == 0
    _said(_files_report(root))


def _retyped(root: Path, rel: str, column: str, kind) -> None:
    table = _arrow(root, rel)
    at = table.schema.get_field_index(column)
    _write(table.set_column(at, column, table.column(at).cast(kind)), root / rel)


def test_pointer_columns_typed_differently_between_episode_files_are_reported(tmp_path):
    """표가 두 파일이고 `data/file_index` 가 한쪽은 int8, 다른 쪽은 int16 이다. pandas 는 이어 붙이며 int16
    으로 넓혀 정수 열 하나로 보여 준다. 적재기는 첫 파일의 형을 표의 형으로 삼아, 들어가지 않는 값에서
    실패한다 — 값이 들어가면 읽히지만, 이 열은 LeRobot 이 늘 한 가지 형으로 쓴다."""
    import pyarrow as pa

    root = _release(tmp_path)
    first, second = _split_episode_table(root, ("file-000", "file-001"), ([0, 1], [2, 3]))
    _retyped(root, str(first.relative_to(root)), "data/file_index", pa.int8())
    _retyped(root, str(second.relative_to(root)), "data/file_index", pa.int16())
    (f,) = _said(_files_report(root), POINTER_TYPES)
    assert f.location == ("meta/episodes/chunk-000/file-001.parquet · data/file_index: int16 "
                          "(첫 파일 meta/episodes/chunk-000/file-000.parquet: int8)"), f.location
    _spans_see_nothing(root)


def test_step_files_typed_differently_are_not_held_to_each_other(tmp_path):
    # The loader types the step table by info.json's features, not by its first file: each file is
    # cast to what is declared, so int32 beside int64 loads. (Stored against declared is not looked at.)
    import pyarrow as pa

    root = _release(tmp_path)
    _retyped(root, FILE_1, "frame_index", pa.int32())
    table = _arrow(root, FILE_1)
    _write(table.append_column("extra", pa.array([0.5] * table.num_rows)), root / FILE_1)
    assert _files_report(root).findings == []


def _two_episode_files_with(root: Path, first, second) -> list[Path]:
    """The episode table in two files, each with a column ``x`` of the given array (None: no such column)."""
    files = _split_episode_table(root, ("file-000", "file-001"), ([0, 1], [2, 3]))
    for file, column in zip(files, (first, second)):
        if column is not None:
            _write(_arrow(root, str(file.relative_to(root))).append_column("x", column), file)
    return files


# ── 형의 차이: 적재기 자신을 잰 것과 대조한다 ──
#
# 적재기는 에피소드 표의 형을 첫 파일로 정하고 뒤 파일을 거기에 맞춘다. 어떤 차이가 「값이 무엇이든」
# 실패하는지를 쌍마다 따져서 정하면 따져 본 쌍 밖의 쌍을 놓친다.
# 그래서 재서 정한다 — 형 스무 가지의 순서쌍 400개를 적재기의 캐스트와 판독기에 넣은 결과가
# loader_casts_measured.json 이고(scripts/measure_loader_casts.py 가 쓴다 · datasets 판이 바뀌면 다시),
# 검증기의 규칙(_never_fits)은 그 파일과 쌍마다 맞아야 한다. 기대값은 여기 적지 않는다.

def _measured() -> dict:
    return json.loads(MEASURED.read_text(encoding="utf-8"))


def _said_never_to_fit() -> set[tuple[str, str]]:
    return {(first, later) for first in TYPES for later in TYPES if vd._never_fits(TYPES[later], TYPES[first])}


def test_the_measurement_covers_every_pair_of_the_types_and_its_two_ways_agreed():
    measured = _measured()
    assert measured["types"] == list(TYPES), "형 목록이 바뀌었습니다 — scripts/measure_loader_casts.py --write 로 다시 재십시오"
    assert measured["reader_and_cast_disagree"] == []
    fails = {tuple(pair) for pair in measured["fails"]}
    assert len(fails) == len(measured["fails"]) and fails <= {(a, b) for a in TYPES for b in TYPES}
    # 몇 개를 보고 있는가: 둘 다 비어 있지 않아야 대조가 뜻을 가진다.
    assert 0 < len(fails) < len(TYPES) ** 2


def test_the_rule_never_says_what_the_loader_reads():
    """적재기가 읽는 차이를 ERROR 로 말하면 멀쩡한 배포본을 막는다 — 변환기의 정상 출력(빈 목록뿐인 뒤 샤드)이
    그 예다."""
    fails = {tuple(pair) for pair in _measured()["fails"]}
    wrongly = sorted(_said_never_to_fit() - fails)
    assert not wrongly, f"적재기는 읽는데 규칙이 「맞출 수 없음」이라고 하는 (첫 파일, 뒤 파일) {len(wrongly)}쌍: {wrongly[:5]}"


def test_the_rule_misses_no_failure_that_a_shape_decides():
    """목록 · struct · 빈 형이 얽힌 쌍에서 적재기가 실패하면 규칙도 말해야 한다(목록의
    깊이 · struct 의 필드 · 고정 길이). 놓쳐도 되는 것은 잎 형끼리 Arrow 에 변환이 아예 없는 쌍뿐이고,
    그것이 몇 쌍인지는 재서 안다."""
    fails = {tuple(pair) for pair in _measured()["fails"]}
    missed = fails - _said_never_to_fit()
    shaped = sorted(pair for pair in missed if not (pair[0] in LEAVES and pair[1] in LEAVES))
    assert not shaped, f"적재기는 실패하는데 규칙이 말하지 않는 (첫 파일, 뒤 파일) {len(shaped)}쌍: {shaped[:5]}"
    assert all(TYPES[first] != TYPES[later] and "null" not in (first, later) for first, later in missed), sorted(missed)


def test_the_types_measured_are_types_a_footer_hands_back(tmp_path):
    # The names mean one thing on both sides only if parquet returns each type as written.
    import pyarrow as pa
    import pyarrow.parquet as pq

    for name in TYPES:
        values = next(iter(candidate_values(name).values()))
        pq.write_table(pa.table({"x": values}), tmp_path / "x.parquet")
        assert pq.read_schema(tmp_path / "x.parquet").field("x").type == TYPES[name], name


# The same through files: a two-file episode table, the footers, the sentence. The pairs are ones
# the producer writes and ones a rule reasoned pair by pair missed; what each should give is read
# from the measurement.
THROUGH_FILES = [
    ("list<string>", "list<null>"),                 # the converter's shards, the order the loader reads
    ("list<null>", "list<string>"),                 # ... and the order it cannot
    ("null", "string"), ("string", "null"), ("int64", "null"), ("null", "int64"),
    ("list<string>", "list<list<string>>"), ("list<list<string>>", "list<string>"),      # depth, both ways
    ("struct<a:null>", "struct<a:string>"), ("struct<a:string>", "struct<a:null>"),
    ("struct<a:string>", "struct<a:string,b:int64>"), ("struct<a:string,b:int64>", "struct<a:string>"),
    ("fixed<string>[1]", "fixed<string>[2]"), ("fixed<string>[2]", "list<string>"),
    ("list<null>", "large_list<string>"), ("large_list<string>", "list<null>"),
    ("bool", "struct<a:string>"), ("struct<a:string>", "bool"),
    ("int8", "int64"), ("int64", "double"), ("string", "large_string"),                    # by the values: not said
]


@pytest.mark.parametrize("first, later", THROUGH_FILES)
def test_a_column_that_can_never_be_cast_to_the_first_files_type_is_reported(tmp_path, first, later):
    fails = (first, later) in {tuple(pair) for pair in _measured()["fails"]}
    root = _release(tmp_path)
    files = _two_episode_files_with(root, *(next(iter(candidate_values(name).values())) for name in (first, later)))
    errors = _said(_files_report(root), *([NEVER_FITS] if fails else []))
    if fails:
        import pyarrow.parquet as pq

        stored = [pq.read_schema(f).field("x").type for f in files]      # as the footer spells them
        assert errors[0].location == (f"{files[1].relative_to(root)} · x: {stored[1]} "
                                      f"(첫 파일 {files[0].relative_to(root)}: {stored[0]})"), errors[0].location
    _spans_see_nothing(root)


def test_a_column_only_a_later_file_has_is_reported_and_one_only_the_first_has_is_not(tmp_path):
    import pyarrow as pa

    root = _release(tmp_path / "later")
    files = _two_episode_files_with(root, None, pa.array(["a", "b"]))
    (f,) = _said(_files_report(root), FIRST_LACKS)
    assert f.location == f"{files[1].relative_to(root)} · x: string (첫 파일 {files[0].relative_to(root)}: 없음)"
    root = _release(tmp_path / "first")
    _two_episode_files_with(root, pa.array(["a", "b"]), None)            # the loader fills it in with nulls
    assert _files_report(root).findings == []


def _schema_only(path: Path, like) -> None:
    """A parquet with ``like``'s schema and no row group at all."""
    import pyarrow.parquet as pq

    with pq.ParquetWriter(path, like.schema):
        pass
    assert pq.read_metadata(path).num_row_groups == 0


def test_a_later_file_without_a_row_group_is_not_typed_against_the_first(tmp_path):
    """판독기는 행 그룹이 없는 파일에서 배치를 만들지 않으므로 형을 맞추지도 않는다. 그 파일의 스키마를
    두고 「맞출 수 없음」·「첫 파일에 없는 열」을 말하면 적재기가 읽는 배포본을 막는다."""
    import pyarrow as pa

    root = _release(tmp_path)
    table = _arrow(root, EPISODES)
    _write(table.append_column("x", pa.nulls(table.num_rows)), root / EPISODES)          # x: null in the first file
    later = table.append_column("x", pa.array(["a"] * table.num_rows)).append_column("y", pa.array([1] * table.num_rows))
    at = later.schema.get_field_index("data/file_index")
    later = later.set_column(at, "data/file_index", later.column(at).cast(pa.int8()))    # and a pointer column retyped
    _schema_only(root / "meta/episodes/chunk-000/file-001.parquet", later)
    rep = _files_report(root)
    assert rep.findings == [] and rep.stats["table_files"] == {"data": 2, "meta/episodes": 2}
    assert _report(root).findings == []


def test_a_first_file_without_a_row_group_still_types_the_table(tmp_path):
    # The reader takes the first schema it can read, rows or not: the table's first two episodes
    # move to the second file, and their column has values where the schema-only first file has null.
    import pyarrow as pa

    root = _release(tmp_path)
    table = _arrow(root, EPISODES)
    (root / EPISODES).unlink()
    _schema_only(root / "meta/episodes/chunk-000/file-000.parquet", table.append_column("x", pa.nulls(table.num_rows)))
    _write(table.append_column("x", pa.array(["a"] * table.num_rows)), root / "meta/episodes/chunk-000/file-001.parquet")
    _said(_files_report(root), NEVER_FITS)


def test_pandas_types_a_column_of_empty_lists_as_a_list_of_nulls(tmp_path):
    """위 사례의 `list<null>` 은 지어낸 형이 아니다 — 변환기가 에피소드 샤드를 pandas 로 쓸 때, 목록이 전부
    빈 샤드가 이 형이 된다. pandas 가 달리 쓰게 되면 이 시험이 알려 준다."""
    import pyarrow.parquet as pq

    pd.DataFrame({"triggered_rules": [["R-1"], []]}).to_parquet(tmp_path / "some.parquet", index=False)
    pd.DataFrame({"triggered_rules": [[], []]}).to_parquet(tmp_path / "none.parquet", index=False)
    kinds = [str(pq.read_schema(tmp_path / name).field("triggered_rules").type) for name in ("some.parquet", "none.parquet")]
    assert kinds == ["list<element: string>", "list<element: null>"], kinds


@pytest.mark.parametrize("rel", [EPISODES, FILE_1])
def test_a_parquet_with_two_columns_of_one_name_is_reported_not_crashed_on(tmp_path, rel):
    """footer 는 읽히는데 이름으로 열을 찾으면 KeyError 다 — 잡지 않으면 검증기가 exit 3 으로 죽는다.
    적재기도 이름으로 찾다 실패한다."""
    import pyarrow as pa

    root = _release(tmp_path)
    table = _arrow(root, rel)
    extra = pa.array([0] * table.num_rows)
    _write(pa.Table.from_arrays([*table.columns, extra, extra], names=[*table.schema.names, "extra", "extra"]), root / rel)
    (f,) = _said(_files_report(root), DUPLICATE_NAMES)
    assert f.location == f"{rel} · ['extra']", f.location
    assert vd.main([str(root), "--checks", "C7"]) == 1


def test_a_file_with_a_doubled_name_is_said_once_and_not_typed_as_well(tmp_path):
    # The second episode file carries `extra` twice. That is the one thing said of it: which of the
    # two a type belongs to cannot be told, so it is not also "a column the first file does not have".
    import pyarrow as pa

    root = _release(tmp_path)
    _, second = _split_episode_table(root, ("file-000", "file-001"), ([0, 1], [2, 3]))
    table = _arrow(root, str(second.relative_to(root)))
    extra = pa.array(["a"] * table.num_rows)
    _write(pa.Table.from_arrays([*table.columns, extra, extra], names=[*table.schema.names, "extra", "extra"]), second)
    (f,) = _said(_files_report(root), DUPLICATE_NAMES)
    assert f.location == f"{second.relative_to(root)} · ['extra']", f.location


def _declare(root: Path, features) -> Path:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    info["features"] = features
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return root


AS_LEROBOT_DECLARES = {name: {"dtype": "int64", "shape": [1], "names": None}
                       for name in ("index", "episode_index", "frame_index")}


@pytest.mark.parametrize("features", [
    AS_LEROBOT_DECLARES,
    {**AS_LEROBOT_DECLARES, "frame_index": {"dtype": "int32", "shape": [1]}},
    {**AS_LEROBOT_DECLARES, "index": {"dtype": "uint8", "shape": (1,)}},
    {**AS_LEROBOT_DECLARES, "timestamp": {"dtype": "float32", "shape": [1]}},     # not a column of place
    {"observation.state": {"dtype": "float64", "shape": [2]}},                    # says nothing of them
    {}, None, ["index"],                                                           # nothing to read a type from
])
def test_place_columns_declared_as_integers_or_not_at_all_are_left_alone(tmp_path, features):
    assert _files_report(_declare(_release(tmp_path), features)).findings == []


@pytest.mark.parametrize("column", ["index", "episode_index", "frame_index"])
@pytest.mark.parametrize("declared", [
    {"dtype": "float32", "shape": [1]}, {"dtype": "string", "shape": [1]}, {"dtype": "video", "shape": [1]},
    {"dtype": 7, "shape": [1]}, {"dtype": ["int64"], "shape": [1]}, {"shape": [1]}, "int64", None,
    # An integer, but not one a step: the loader makes a sequence of it and cannot cast the column.
    {"dtype": "int64", "shape": [2]}, {"dtype": "int64", "shape": []}, {"dtype": "int64", "shape": 1},
    {"dtype": "int64"},                                                 # no shape: KeyError in the loader
    # numpy's names for an integer: the loader's feature type takes pyarrow's names and refuses these.
    {"dtype": "long", "shape": [1]}, {"dtype": "<i8", "shape": [1]}, {"dtype": "int", "shape": [1]},
])
def test_a_place_column_declared_as_anything_but_one_integer_is_reported(tmp_path, column, declared):
    """parquet 의 형은 int64 그대로다. 적재기는 스텝 표를 `info.json` 의 features 가 선언한 대로 바꿔
    읽는다 — `float32` 면 `episode_index` 가 0.0 으로 와서 그 값으로 에피소드 표의 행을 찾다 죽고, 모양이
    `[1]` 이 아니거나 형 이름이 pyarrow 의 것이 아니면 표를 만들다 죽는다(실제 적재기로 확인)."""
    root = _declare(_release(tmp_path), {**AS_LEROBOT_DECLARES, column: declared})
    (f,) = _said(_files_report(root), DECLARED_NOT_INTEGER)
    assert f"features 가 {column} 을" in f.message and f.location == "meta/info.json", f.line()
    _spans_see_nothing(root)


def test_a_file_that_is_not_parquet_is_another_checks_to_report(tmp_path):
    # load_frames and Dataset.episodes say so under C1, run or not; here it must only not end the run.
    root = _release(tmp_path)
    (root / FILE_1).write_bytes(b"not a parquet file")
    (root / EPISODES).write_bytes(b"nor this")
    rep = _files_report(root)
    assert rep.findings == [] and rep.stats["table_files"] == {"data": 1, "meta/episodes": 0}


def test_a_release_without_its_table_directories_is_not_crashed_on(tmp_path):
    root = _release(tmp_path)
    shutil.rmtree(root / "data")
    shutil.rmtree(root / "meta" / "episodes")
    rep = _files_report(root)
    _said(rep, NO_TABLE, NO_STEP_TABLE)
    assert rep.stats["table_files"] == {"data": 0, "meta/episodes": 0}


def test_files_where_the_loader_does_not_look_are_not_held_to_the_tables_types(tmp_path):
    # A stray file is said as one; its types and its row groups are no part of the table's.
    import pyarrow as pa

    root = _release(tmp_path)
    table = _arrow(root, EPISODES)
    at = table.schema.get_field_index("data/file_index")
    _write(table.set_column(at, "data/file_index", table.column(at).cast(pa.int8())).slice(0, 0),
           root / "meta" / "episodes" / "stray.parquet")
    _said(_files_report(root), STRAY_TABLE)


# ── 검증기에 걸려 있는가 · 픽스처가 통과하는가 ────────────────────

FIXTURE_EPISODES = EPISODES


@pytest.mark.parametrize("with_media", [False, True])
def test_the_clean_fixtures_pass_with_every_episode_compared(tmp_path, with_media):
    root = tmp_path / "clean"
    build(root, with_media=with_media)
    rep = vd.run(root, {"C7"}, None, 0, None)
    assert [f.line() for f in rep.findings if f.severity == vd.ERROR] == []
    eps, steps, files, _ = _read_release(root)
    assert rep.stats["data_spans"] == {"episodes": len(eps), "rows": len(steps)}
    assert rep.stats["table_files"] == {"data": len(files), "meta/episodes": 1}


def test_the_table_file_check_runs_as_part_of_c7_and_only_then(tmp_path):
    root = tmp_path / "clean"
    build(root)
    first = root / vd.DATA_PATH.format(chunk_index=0, file_index=0)
    _write(_arrow(root, str(first.relative_to(root))).slice(0, 0), first.with_name("file-009.parquet"))

    def said(checks, file_limit=None):
        return [f.check for f in vd.run(root, checks, None, 0, file_limit).findings if EMPTY_FIRST_GROUP in f.message]

    assert said({"C7"}) == ["C7"]
    assert said(set(vd.ALL_CHECKS) - {"C7"}) == []
    # Footers cost nothing to read, so the file --file-limit kept the readers from is looked at all
    # the same -- without this line a version that stopped there passes every test.
    assert said({"C7"}, file_limit=1) == ["C7"]


def test_the_check_runs_as_part_of_c7_and_only_then(tmp_path):
    root = tmp_path / "clean"
    build(root)
    _edit(root, FIXTURE_EPISODES, _set(1, dataset_from_index=800, dataset_to_index=1200))
    _edit(root, FIXTURE_EPISODES, _set(2, dataset_from_index=400, dataset_to_index=800))

    def spans(checks):
        return [f for f in vd.run(root, checks, None, 0, None).findings if NOT_ITS_ROWS in f.message]

    assert [f.check for f in spans({"C7"})] == ["C7", "C7"]
    assert spans(set(vd.ALL_CHECKS) - {"C7"}) == []


SPAN_VARIANTS = {
    "broken_span_swapped": (NOT_ITS_ROWS, NOT_ITS_ROWS),
    "broken_span_end_short": (LENGTH, ORPHANS),
    "broken_span_file_index": (OTHER_FILE,),
    "broken_span_rows_out_of_place": (INDEX, NOT_ITS_ROWS, NOT_ITS_ROWS),
}


def test_every_span_variant_is_held_to_its_sentences():
    assert sorted(SPAN_VARIANTS) == sorted(v for v in VARIANTS if v.startswith("broken_span_"))


@pytest.mark.parametrize("variant", sorted(SPAN_VARIANTS))
def test_each_span_variant_fails_only_on_its_defect(tmp_path, variant):
    """픽스처의 결함은 포인터 하나뿐이어야 한다 — 다른 검사가 같은 입력을 먼저 거부하면, 이 검사를
    지워도 그 픽스처는 여전히 「검출」로 세어진다."""
    root = tmp_path / "clean"
    build(root)
    VARIANTS[variant](root)
    _said(vd.run(root, set(vd.ALL_CHECKS), None, 0, None), *SPAN_VARIANTS[variant])
