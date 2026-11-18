"""`info.json` 이 말하는 수가 적재기가 읽는 표의 행 수인가 (C7, `c7_info_totals`).

LeRobot 은 표를 세지 않는다. 에피소드 수와 스텝 수를 `meta/info.json` 의 `total_episodes` ·
`total_frames` 에서 읽고(lerobot 0.6.1 `dataset_metadata.total_episodes` · `total_frames`),
`len(dataset)` 이 곧 `total_frames` 다. 그래서 표를 다시 쓴 뒤 이 수가 남으면 — 포인터는 여전히
제 행을 가리키므로 포인터 대조(`c7_data_spans`)는 아무 말도 하지 않는데 — 길이만큼 읽는 쪽은 끝의
행을 받지 못하거나 없는 행을 찾는다.

여기 사례는 손으로 지은 몇 행짜리 배포본이다. 검사는 parquet 의 footer 와 `info.json` 만 읽으므로
열은 하나면 된다. 픽스처 쪽은 맨 아래의 시험과 `test_fixture_detection_reason.py` 의
`broken_total_*` 가 본다.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

import validator.validate_dataset as vd
from validator.make_fixture import VARIANTS, build

LOADED = "*/*.parquet"          # what LeRobot loads of data/ and meta/episodes/ (io_utils.load_nested_dataset)
STEPS = "data/chunk-000/file-{:03d}.parquet"
EPISODES = "meta/episodes/chunk-000/file-{:03d}.parquet"
LENGTHS = (3, 4, 2, 5)          # two episodes to a step file
INFO = "meta/info.json"

# One sentence for each count; what was stated and what was counted are in the finding's location.
DIFFERS = {"total_episodes": "info.json 의 total_episodes 가 에피소드 표의 행 수와 다름",
           "total_frames": "info.json 의 total_frames 가 스텝 표의 행 수와 다름"}
TABLE = {"total_episodes": "에피소드 표", "total_frames": "스텝 표"}
ROWS = {"total_episodes": len(LENGTHS), "total_frames": sum(LENGTHS)}
KEYS = sorted(DIFFERS)
ABSENT = object()


def _table(path: Path, episode_index: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"episode_index": episode_index}).to_parquet(path, index=False)


def _release(root: Path, lengths=LENGTHS, **stated) -> Path:
    """Episodes of these lengths, two to a step file, counted in info.json as LeRobot counts them
    (dataset_metadata.save_episode: one more episode, ``length`` more frames). ``stated`` puts
    something else in info.json -- ABSENT takes the key out."""
    for j in range(0, len(lengths), 2):
        _table(root / STEPS.format(j // 2), [k for k in range(j, min(j + 2, len(lengths))) for _ in range(lengths[k])])
    _table(root / EPISODES.format(0), list(range(len(lengths))))
    return _state(root, **{"total_episodes": len(lengths), "total_frames": sum(lengths), **stated})


def _state(root: Path, **stated) -> Path:
    """info.json with these counts, and nothing else about the release touched."""
    info = {"codebase_version": "v3.0", "fps": 50}
    for key, value in stated.items():
        if value is not ABSENT:
            info[key] = value
    (root / INFO).write_text(json.dumps(info), encoding="utf-8")
    return root


def _rows_on_disk(root: Path) -> dict[str, int]:
    """What the two tables hold, counted here and not by the validator: the files LeRobot loads,
    each read whole."""
    return {"total_episodes": sum(len(pq.read_table(f)) for f in (root / "meta" / "episodes").glob(LOADED)),
            "total_frames": sum(len(pq.read_table(f)) for f in (root / "data").glob(LOADED))}


def _report(root: Path) -> vd.Report:
    rep = vd.Report(root=str(root))
    vd.c7_info_totals(vd.Dataset(root, rep), rep)
    return rep


def _said(rep: vd.Report, *keys: str) -> list[vd.Finding]:
    """The report is the sentence of each of these counts, once, as a C7 ERROR -- nothing else."""
    lines = [f.line() for f in rep.findings]
    assert all((f.check, f.severity) == ("C7", vd.ERROR) for f in rep.findings), lines
    assert sorted(f.message.split(" — ")[0] for f in rep.findings) == sorted(DIFFERS[k] for k in keys), lines
    return rep.findings


def test_a_release_counted_as_lerobot_counts_it_has_nothing_to_report(tmp_path):
    rep = _report(_release(tmp_path))
    assert rep.findings == [], [f.line() for f in rep.findings]
    # Nothing said is only worth something beside what it was held to.
    assert rep.stats["info_totals"] == ROWS


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("off_by", [1, -1])
def test_a_count_one_off_is_reported_with_both_numbers(tmp_path, key, off_by):
    root = _release(tmp_path, **{key: ROWS[key] + off_by})
    (f,) = _said(_report(root), key)
    assert f.location == f"{INFO} · {key} {ROWS[key] + off_by}, {TABLE[key]} {ROWS[key]}행", f.location


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("stated", [0, -1])
def test_a_count_of_nothing_is_reported(tmp_path, key, stated):
    (f,) = _said(_report(_release(tmp_path, **{key: stated})), key)
    assert f"{key} {stated}," in f.location, f.location


@pytest.mark.parametrize("key", KEYS)
def test_an_absent_count_is_what_lerobot_reads_as_zero(tmp_path, key):
    """키가 없으면 LeRobot 은 0 으로 읽는다(`DatasetInfo` 의 기본값) — 길이 0 의 데이터셋이거나
    에피소드 표를 읽지 않는 데이터셋이 된다. 「없으니 볼 것이 없다」로 지나가면 안 된다."""
    (f,) = _said(_report(_release(tmp_path, **{key: ABSENT})), key)
    assert f"{key} 없음" in f.location and f"{ROWS[key]}행" in f.location, f.location


def test_an_absent_count_beside_tables_of_no_rows_is_the_zero_it_is_read_as(tmp_path):
    """위 시험만으로는 「없으면 0」을 못박지 못한다 — 없는 키를 무엇으로 읽든 행이 있는 표와는 다르기
    때문이다. 표가 0행이면 갈린다: 0 으로 읽으면 맞는 수이고, 다른 것으로 읽으면 틀린 수다."""
    _table(tmp_path / STEPS.format(0), [])
    _table(tmp_path / EPISODES.format(0), [])
    for stated in ({}, {"total_episodes": 0, "total_frames": 0}):
        rep = _report(_state(tmp_path, **stated))
        assert rep.findings == [], [f.line() for f in rep.findings]
        assert rep.stats["info_totals"] == {"total_episodes": 0, "total_frames": 0}


NOT_INTEGERS = {
    "a float": lambda n: float(n),      # json writes 14.0, and 14.0 == 14
    "a string": lambda n: str(n),
    "null": lambda n: None,
    "a list": lambda n: [n],
}


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("kind", sorted(NOT_INTEGERS))
def test_a_count_that_is_not_an_integer_is_reported_as_it_is_written(tmp_path, key, kind):
    """값이 같아 보여도 정수가 아니면 적재기는 `len()` 이나 여는 단계에서 TypeError 로 멈춘다."""
    stated = NOT_INTEGERS[kind](ROWS[key])
    (f,) = _said(_report(_release(tmp_path, **{key: stated})), key)
    assert f"{key} {stated!r}," in f.location, f.location


@pytest.mark.parametrize("key", KEYS)
def test_a_long_number_is_shown_cut_with_a_mark_and_not_as_another_number(tmp_path, key):
    """백 자리 수를 앞의 여든 자만 보이면 다른 수로 읽힌다 — 잘랐다는 표시와 전체 길이를 붙인다."""
    stated = 10 ** 100 + 7
    (f,) = _said(_report(_release(tmp_path, **{key: stated})), key)
    shown = f.location.split(f"{key} ")[1].split(",")[0]
    assert shown == f"{str(stated)[:80]}… (전체 {len(str(stated))}자)", f.location


def test_true_is_not_a_count_of_one(tmp_path):
    # One episode of one step: True == 1, and a comparison by value alone would pass both.
    root = _release(tmp_path, lengths=(1,), total_episodes=True, total_frames=True)
    for f in _said(_report(root), *KEYS):
        assert "True," in f.location, f.location


def test_every_file_of_a_table_is_counted(tmp_path):
    root = _release(tmp_path)
    _table(root / EPISODES.format(0), [0, 1, 2])
    _table(root / EPISODES.format(1), [3])
    rep = _report(root)
    assert rep.findings == [], [f.line() for f in rep.findings]
    assert rep.stats["info_totals"] == ROWS


def test_every_row_group_of_a_file_is_counted(tmp_path):
    """LeRobot 이 쓴 스텝 파일은 에피소드마다 행 그룹이 하나다(에피소드 10개짜리 파일에 10개). 첫 그룹만
    세면 그런 파일의 수가 모두 틀린다."""
    root = _release(tmp_path)
    path = root / STEPS.format(0)
    table = pq.read_table(path)
    with pq.ParquetWriter(path, table.schema) as writer:
        writer.write_table(table.slice(0, LENGTHS[0]))
        writer.write_table(table.slice(LENGTHS[0]))
    assert pq.read_metadata(path).num_row_groups == 2
    rep = _report(root)
    assert rep.findings == [], [f.line() for f in rep.findings]
    assert rep.stats["info_totals"] == ROWS


def _skip(b: bytes, at: int, kind: int) -> int:
    """Past one value of this Thrift compact type -- as much of the protocol as a parquet footer
    uses before its row count."""
    if kind in (1, 2):                              # a bool, held in the field's own header
        return at
    if kind == 3:                                   # a byte
        return at + 1
    if kind in (4, 5, 6):                           # an integer, as a varint
        while b[at] & 0x80:
            at += 1
        return at + 1
    if kind == 8:                                   # bytes, with their length before them
        size, at = _varint(b, at)
        return at + size
    if kind == 9:                                   # a list
        head, at = b[at], at + 1
        size = head >> 4
        if size == 15:
            size, at = _varint(b, at)
        for _ in range(size):
            at = _skip(b, at, head & 15)
        return at
    if kind == 12:                                  # a struct: fields up to a stop byte
        while b[at]:
            head, at = b[at], at + 1
            if not head >> 4:                       # the field's number written out
                at = _skip(b, at, 5)
            at = _skip(b, at, head & 15)
        return at + 1
    raise ValueError(f"Thrift type {kind}")


def _varint(b: bytes, at: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        value |= (b[at] & 0x7F) << shift
        shift += 7
        at += 1
        if not b[at - 1] & 0x80:
            return value, at


def _footer_says(path: Path, rows: int) -> None:
    """Rewrite the footer's own row count (FileMetaData.num_rows, field 3) and leave the row
    groups as they are -- a file no writer makes, and one every reader still reads."""
    raw = path.read_bytes()
    size = int.from_bytes(raw[-8:-4], "little")
    footer = raw[-8 - size:-8]
    at = number = 0
    while True:
        head, at = footer[at], at + 1
        assert head >> 4, "a parquet footer numbers its first fields by short steps"
        number += head >> 4
        if number == 3:
            break
        at = _skip(footer, at, head & 15)
    end = _skip(footer, at, 6)
    zigzag, encoded = rows << 1, bytearray()
    while zigzag >= 0x80:
        encoded.append(zigzag & 0x7F | 0x80)
        zigzag >>= 7
    encoded.append(zigzag)
    footer = footer[:at] + bytes(encoded) + footer[end:]
    path.write_bytes(raw[:-8 - size] + footer + len(footer).to_bytes(4, "little") + b"PAR1")


def test_a_files_rows_are_its_row_groups_not_the_footers_own_total(tmp_path):
    """footer 는 행 수를 두 번 적는다 — 파일 전체의 수와 행 그룹마다의 수. 판독기는 행 그룹을 읽으므로
    (실제 적재기로 확인: 전체 수만 5 → 6 으로 고친 파일에서 다섯 행을 읽는다) 전체 수를 믿고 세면,
    그 수에 맞춘 `total_frames` 가 옳다고 하게 된다."""
    root = _release(tmp_path)
    path, held = root / STEPS.format(1), sum(LENGTHS[2:])
    _footer_says(path, held + 1)
    assert pq.read_metadata(path).num_rows == held + 1 and len(pq.read_table(path)) == held, "the case is gone"

    rep = _report(root)
    assert rep.findings == [], [f.line() for f in rep.findings]
    assert rep.stats["info_totals"] == ROWS
    (f,) = _said(_report(_state(root, total_episodes=ROWS["total_episodes"], total_frames=ROWS["total_frames"] + 1)),
                 "total_frames")
    assert f.location.endswith(f"스텝 표 {ROWS['total_frames']}행"), f.location


def test_a_chunk_directory_that_is_a_symlink_is_counted(tmp_path):
    """LeRobot 의 `glob("*/*.parquet")` 은 심링크인 청크 디렉터리 안의 파일도 읽는다(실제 적재기로
    확인). 재귀 탐색은 거기로 들어가지 않아, 그것으로 세면 옳은 수를 틀렸다고 한다."""
    root = _release(tmp_path / "release")
    for table, rel in (("data", STEPS.format(1)), ("meta/episodes", None)):
        outside = tmp_path / "elsewhere" / table.replace("/", "-")
        outside.mkdir(parents=True)
        if rel is None:
            _table(root / EPISODES.format(0), [0, 1])
            _table(outside / "file-000.parquet", [2, 3])
        else:
            (root / rel).rename(outside / "file-001.parquet")
        (root / table / "chunk-001").symlink_to(outside, target_is_directory=True)
    rep = _report(root)
    assert rep.findings == [], [f.line() for f in rep.findings]
    assert rep.stats["info_totals"] == ROWS


def test_rows_where_the_loader_does_not_look_are_not_counted(tmp_path):
    """적재기가 읽는 것은 한 단계 아래의 파일뿐이다. 그 밖의 자리의 행을 세면, 적재기가 받지 못하는
    행까지 센 수를 옳다고 하게 된다."""
    root = _release(tmp_path)
    _table(root / "data" / "stray.parquet", [9, 9])
    _table(root / "meta" / "episodes" / "chunk-000" / "deeper" / "file-000.parquet", [9])
    assert _report(root).findings == []

    counted_in = _release(root, total_episodes=ROWS["total_episodes"] + 1, total_frames=ROWS["total_frames"] + 2)
    for f in _said(_report(counted_in), *KEYS):
        key = next(k for k in KEYS if k in f.location)
        assert f.location.endswith(f"{TABLE[key]} {ROWS[key]}행"), f.location


def _c7(root: Path, file_limit=None) -> vd.Report:
    return vd.run(root, {"C7"}, None, 0, file_limit)


def _totals_said(rep: vd.Report) -> list[str]:
    return sorted(f.message.split(" — ")[0] for f in rep.findings if f.message.startswith("info.json 의 total_"))


def test_the_validator_runs_it_as_part_of_c7_and_only_then(tmp_path):
    root = _release(tmp_path, total_frames=ROWS["total_frames"] - 1)
    assert _totals_said(_c7(root)) == [DIFFERS["total_frames"]]
    assert _totals_said(vd.run(root, {"C1"}, None, 0, None)) == []
    assert vd.main([str(root), "--checks", "C7"]) == 1


def test_a_file_limit_does_not_change_what_is_counted(tmp_path):
    """--file-limit 은 읽는 스텝 파일을 줄인다. 읽은 행으로 세면 맞는 수가 틀린 것이 되고 틀린 수가
    맞는 것이 된다 — footer 로 세므로 무관하다."""
    root = _release(tmp_path)
    rep = _c7(root, file_limit=1)
    assert _totals_said(rep) == []
    assert rep.stats["info_totals"] == ROWS

    first_file_only = sum(LENGTHS[:2])
    assert _totals_said(_c7(_release(root, total_frames=first_file_only), file_limit=1)) == [DIFFERS["total_frames"]]


def _no_step_table(root: Path) -> None:
    for f in (root / "data").rglob("*.parquet"):
        f.unlink()


def _episodes_as_json(root: Path) -> None:
    rows = pd.read_parquet(root / EPISODES.format(0)).to_dict("records")
    (root / EPISODES.format(0)).unlink()
    (root / "meta" / "episodes.json").write_text(json.dumps(rows), encoding="utf-8")


def _a_step_file_that_is_not_parquet(root: Path) -> None:
    (root / STEPS.format(9)).write_bytes(b"not a parquet")


NOT_COUNTED = {
    # what is done to the release -> the count that is then held to nothing, and the file said to be why
    "no step table": (_no_step_table, "total_frames", None),
    "an episode table only the validator's fallback reads": (_episodes_as_json, "total_episodes", None),
    "a file of the table that is not parquet": (_a_step_file_that_is_not_parquet, "total_frames", STEPS.format(9)),
}
NOT_COMPARED = "를 대조하지 못함"


@pytest.mark.parametrize("case", sorted(NOT_COUNTED))
def test_a_table_that_cannot_be_counted_is_not_called_miscounted(tmp_path, case):
    """셀 수 없는 표를 「0행과 다르다」고 하지 않는다. 표가 없다는 것은 `c7_table_files` 가 ERROR 로
    말한다. 읽히지 않는 파일은 판독기가 말하지만(C1) --file-limit 뒤의 파일은 판독기가 열지 않으므로,
    그 수를 대조하지 못했다는 것은 여기서 SKIP 으로 말한다 — 말없이 꺼진 대조는 통과로 읽힌다."""
    change, key, unread = NOT_COUNTED[case]
    root = _release(tmp_path, **{key: 999})
    change(root)
    rep = _report(root)
    said = [(f.check, f.severity, f.location) for f in rep.findings]
    assert said == ([] if unread is None else [("C7", vd.SKIP, unread)]), [f.line() for f in rep.findings]
    assert all(key in f.message and NOT_COMPARED in f.message for f in rep.findings)
    other = next(k for k in KEYS if k != key)
    assert rep.stats["info_totals"] == {other: ROWS[other]}


def test_an_unread_file_past_the_file_limit_still_leaves_a_line(tmp_path):
    root = _release(tmp_path, total_frames=999)
    _a_step_file_that_is_not_parquet(root)
    rep = _c7(root, file_limit=1)
    assert [f.check for f in rep.findings if "parquet 읽기 실패" in f.message] == [], "the readers did not open it"
    assert [f.location for f in rep.findings if NOT_COMPARED in f.message] == [STEPS.format(9)]


def test_a_count_left_uncompared_does_not_read_as_a_check_that_passed(tmp_path):
    """INFO 는 기본 출력에 나오지 않고 검사의 상태를 바꾸지 않는다. 그렇게 말하면 요약은 「C7 OK」 이고
    보고서 JSON 의 check_status 도 ok 다 — 대조하지 못한 수가 대조해서 맞은 수와 같은 줄로 읽힌다."""
    root = tmp_path / "clean"
    build(root)
    (root / STEPS.format(9)).write_bytes(b"not a parquet")
    rep = _c7(root, file_limit=1)
    assert [(f.severity, f.location) for f in rep.findings if NOT_COMPARED in f.message] == [(vd.SKIP, STEPS.format(9))]
    assert (rep.count(vd.ERROR), rep.count(vd.WARN)) == (0, 0), [f.line() for f in rep.findings]
    assert rep.status_of("C7") == "skip"


@pytest.mark.parametrize("text", [None, "{", "[1, 2]", "7"])
def test_an_info_json_that_is_no_object_is_not_crashed_on(tmp_path, text):
    root = _release(tmp_path)
    if text is None:
        (root / INFO).unlink()
    else:
        (root / INFO).write_text(text, encoding="utf-8")
    rep = _report(root)
    assert [f.line() for f in rep.findings if f.check == "C7"] == []
    assert "info_totals" not in rep.stats


NOT_AN_OBJECT = "meta/info.json 이 JSON 객체가 아님"


@pytest.mark.parametrize("text", ["[1, 2]", "7", '"x"', "null"])
def test_an_info_json_that_is_no_object_is_c1s_to_say_and_ends_no_check(tmp_path, text):
    """위에서 이 검사가 말없이 돌아가는 것은 C1 이 말하기 때문이어야 한다. 그렇지 않았다 — 목록 · 수 ·
    문자열에서는 C1 이 `.get` 을 부르다 검증기가 죽었고(exit 3), `null` 은 파일이 없을 때와 같은 값으로
    읽혀 전 검사를 통과했다(exit 0). 적재기는 넷 다 여는 단계에서 멈춘다(실제 적재기로 확인)."""
    root = tmp_path / "clean"
    build(root)
    (root / INFO).write_text(text, encoding="utf-8")
    rep = vd.run(root, set(vd.ALL_CHECKS), None, 0, None)
    said = [(f.check, f.severity, f.location) for f in rep.findings if NOT_AN_OBJECT in f.message]
    assert said == [("C1", vd.ERROR, f"{INFO} · {json.loads(text)!r}")], [f.line() for f in rep.findings]
    assert vd.main([str(root)]) == 1


def test_an_info_json_that_does_not_parse_is_said_as_that_and_not_as_no_object(tmp_path):
    root = tmp_path / "clean"
    build(root)
    (root / INFO).write_text("{", encoding="utf-8")
    rep = vd.run(root, {"C1"}, None, 0, None)
    assert [f for f in rep.findings if NOT_AN_OBJECT in f.message] == []
    assert any("JSON 파싱 실패" in f.message for f in rep.findings)


def test_without_pyarrow_it_says_it_did_not_look(tmp_path, monkeypatch):
    root = _release(tmp_path, total_frames=0)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)       # the import then raises ImportError
    (f,) = _report(root).findings
    assert (f.check, f.severity) == ("C7", vd.SKIP) and "info.json" in f.message, f.line()


# ── 픽스처는 배포본과 같은 표현이어야 한다 ─────────────────────────

@pytest.mark.parametrize("with_media", [False, True])
def test_the_fixture_states_the_counts_of_its_own_tables(tmp_path, with_media):
    """수를 적지 않은 픽스처로는 이 검사의 통과 쪽을 시험하지 못한다 — 적지 않은 수는 0 으로 읽힌다.
    기대값은 적어 두지 않고 픽스처의 표에서 센다."""
    root = tmp_path / "clean"
    build(root, with_media=with_media)
    info = json.loads((root / INFO).read_text(encoding="utf-8"))
    counted = _rows_on_disk(root)
    assert all(counted.values()), counted
    assert {key: info.get(key) for key in KEYS} == counted

    rep = _report(root)
    assert rep.findings == [], [f.line() for f in rep.findings]
    assert rep.stats["info_totals"] == counted


TOTAL_VARIANTS = {"broken_total_episodes": ("total_episodes", +1), "broken_total_frames": ("total_frames", -1)}


def test_every_total_variant_is_held_to_its_count():
    assert sorted(TOTAL_VARIANTS) == sorted(v for v in VARIANTS if v.startswith("broken_total_"))


@pytest.mark.parametrize("variant", sorted(TOTAL_VARIANTS))
def test_each_total_variant_fails_only_on_its_count(tmp_path, variant):
    """픽스처의 결함은 수 하나뿐이어야 한다 — 다른 검사가 같은 입력을 먼저 거부하면, 이 검사를 지워도
    그 픽스처는 여전히 「검출」로 세어진다."""
    key, off_by = TOTAL_VARIANTS[variant]
    root = tmp_path / "clean"
    build(root)
    rows = _rows_on_disk(root)[key]
    VARIANTS[variant](root)
    (f,) = [f for f in vd.run(root, set(vd.ALL_CHECKS), None, 0, None).findings if f.severity == vd.ERROR]
    assert f.message.startswith(DIFFERS[key]), f.line()
    assert f.location == f"{INFO} · {key} {rows + off_by}, {TABLE[key]} {rows}행", f.location
