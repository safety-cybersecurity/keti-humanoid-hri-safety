"""에피소드 → 행 포인터를 읽는 구현이 둘이고, 같은 배포본에 같은 답을 내야 한다.

    검증기 C7                 validator/validate_dataset.py  c7_data_spans
    라벨 재유도 도구          generation/labeling/check_release_labels.py  read_episodes · read_steps · check_spans

둘 다 「LeRobot 이 k번 에피소드로 내주는 행이 그 에피소드의 행인가」를 적재기가 읽는 대로(한 단계
아래의 parquet · 이름 순서 · 저장된 순서 · 표의 k번째 행) 본다. 검증기는 `generation/` 을 import
하지 않으므로 같은 진술을 따로 구현했고, 한쪽만 고치면 한쪽만 통과하는 배포본이 생긴다 — lerobot
판을 올려 읽는 방식이 달라질 때가 그렇다. 상수의 사본이 아니라 구현의 사본이라 값으로는 못 묶고,
같은 결함을 둘에 넣어 묶는다.

여기서 보는 것은 **도구가 거부하면 검증기도 말하는가** 한 방향이다. 검증기만 말하는 것(예: 한
에피소드의 구간이 옳아도 표 전체를 보고 말하는 문장)은 도구의 일이 아니다. 문장이 무엇인지는
각자의 시험이 본다(tests/validator/test_data_spans.py · test_check_release_labels.py).
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import validator.validate_dataset as vd
from generation.labeling import check_release_labels as crl
from tests.generation.release_fixture import (N, edit_episode, edit_json, eight, episodes_path,
                                               write_release)

EPISODES = 8                    # eight(): three to a step file, so files of 3 · 3 · 2 episodes


def _data(root, k):
    return root / "data" / "chunk-000" / f"file-{k:03d}.parquet"


def _rewrite(path, change):
    pq.write_table(change(pq.read_table(path)), path)


def _spans_exchanged(root):
    edit_episode(root, 0, dataset_from_index=N, dataset_to_index=2 * N)
    edit_episode(root, 1, dataset_from_index=0, dataset_to_index=N)


def _rows_trade_places(root):
    _rewrite(_data(root, 0), lambda t: pa.concat_tables([t.slice(N, N), t.slice(0, N), t.slice(2 * N)]))


def _files_trade_places(root):
    first, second = _data(root, 0), _data(root, 1)
    held = first.read_bytes()
    first.write_bytes(second.read_bytes())
    second.write_bytes(held)


def _table_rows_trade_places(root):
    _rewrite(episodes_path(root), lambda t: pa.concat_tables([t.slice(1, 1), t.slice(0, 1), t.slice(2)]))


def _step_rows_outside_the_layout(where):
    def change(root):
        table = pq.read_table(_data(root, 0))
        stray = root / "data" / where
        stray.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.slice(0, N), stray)
        pq.write_table(table.slice(N), _data(root, 0))
    return change


def _episode_row_outside_the_layout(where):
    def change(root):
        table = pq.read_table(episodes_path(root))
        stray = root / "meta" / "episodes" / where
        stray.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.slice(0, 1), stray)
        pq.write_table(table.slice(1), episodes_path(root))
    return change


def _a_step_file_moved_out_with_its_template(root):
    _data(root, 2).rename(root / "data" / "file-002.parquet")
    edit_json(root / "meta" / "info.json",
              lambda doc: doc.update(data_path="data/chunk-000/file-{file_index:03d}.parquet"))


def _frames_from_five(root):
    def change(t):
        at = t.schema.get_field_index("frame_index")
        frames = t.column("frame_index").to_pylist()
        frames[N:2 * N] = [f + 5 for f in frames[N:2 * N]]
        return t.set_column(at, "frame_index", pa.array(frames, t.schema.field("frame_index").type))
    _rewrite(_data(root, 0), change)


def _index_one_on(root):
    def change(t):
        at = t.schema.get_field_index("index")
        return t.set_column(at, "index", pa.array([i + 1 for i in t.column("index").to_pylist()],
                                                  t.schema.field("index").type))
    _rewrite(_data(root, 1), change)


def _a_pointer_column_of_floats(root):
    def change(t):
        at = t.schema.get_field_index("length")
        return t.set_column(at, "length", t.column("length").cast(pa.float64()))
    _rewrite(episodes_path(root), change)


def _episode_files_under_each_others_name(root):
    table = pq.read_table(episodes_path(root))
    pq.write_table(table.slice(EPISODES // 2), episodes_path(root))                 # file-000: the second half
    pq.write_table(table.slice(0, EPISODES // 2), episodes_path(root).with_name("file-001.parquet"))


def _episode_table_only_as_json(root):
    import json
    import shutil
    rows = pq.read_table(episodes_path(root), columns=["episode_index", "length", "data/chunk_index",
                                                        "data/file_index", "dataset_from_index",
                                                        "dataset_to_index"]).to_pylist()
    shutil.rmtree(root / "meta" / "episodes")
    (root / "meta" / "episodes.json").write_text(json.dumps(rows), encoding="utf-8")


DEFECTS = {
    "two episodes' stretches exchanged": _spans_exchanged,
    "two episodes' rows stored in each other's place": _rows_trade_places,
    "two step files under each other's name": _files_trade_places,
    "two rows of the episode table in each other's place": _table_rows_trade_places,
    "step rows in a file at the top of data/": _step_rows_outside_the_layout("aaa.parquet"),
    "step rows in a file two directories down": _step_rows_outside_the_layout("chunk-000/deeper/file-000.parquet"),
    "an episode row in a file at the top of meta/episodes/": _episode_row_outside_the_layout("aaa.parquet"),
    "an episode row in a file two directories down":
        _episode_row_outside_the_layout("chunk-000/deeper/file-000.parquet"),
    "a step file moved out of the layout": _a_step_file_moved_out_with_its_template,
    "a length the stretch does not have": lambda root: edit_episode(root, 7, length=N - 1),
    "a stretch past the table": lambda root: edit_episode(root, 7, dataset_to_index=EPISODES * N + 5,
                                                          length=N + 5),
    "a pointer that is null": lambda root: edit_episode(root, 2, dataset_from_index=None),
    "a pointer column of floats": _a_pointer_column_of_floats,
    "a file pointer to a file that is not there": lambda root: edit_episode(root, 0, **{"data/file_index": 9}),
    "a file pointer to another step file": lambda root: edit_episode(root, 0, **{"data/file_index": 1}),
    "an episode missing from the table": lambda root: _rewrite(episodes_path(root),
                                                               lambda t: t.slice(0, EPISODES - 1)),
    "an episode listed twice": lambda root: _rewrite(episodes_path(root),
                                                     lambda t: pa.concat_tables([t, t.slice(0, 1)])),
    "frames that start at five": _frames_from_five,
    "the episode table's two files under each other's name": _episode_files_under_each_others_name,
    "an episode table only the validator's fallback reads": _episode_table_only_as_json,
    "an index column one on": _index_one_on,
}


def _validator(root):
    rep = vd.Report(root=str(root))
    ds = vd.Dataset(root, rep)
    vd.c7_table_files(ds, rep)                  # which files are the tables, as C7 runs the two
    vd.c7_data_spans(ds, vd.load_frames(ds, rep, None), ds.episodes(), rep)
    return rep


def test_both_name_the_same_characters_as_what_datasets_takes_for_a_pattern():
    """상수의 사본이다. 도구는 배포본이 놓인 경로에서, 검증기는 배포본 안의 표 파일 경로에서 본다 —
    보는 자리는 달라도 「`datasets` 가 패턴으로 읽는 글자」는 같은 사실이다."""
    assert vd.GLOB_CHARACTERS == crl.GLOB_CHARACTERS


def test_a_release_as_written_passes_both(tmp_path):
    root = write_release(tmp_path, eight())
    assert crl.check(root)["differences"] == []
    rep = _validator(root)
    assert rep.findings == [], [f.line() for f in rep.findings]
    assert rep.stats["data_spans"] == {"episodes": EPISODES, "rows": EPISODES * N}


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_what_the_label_tool_refuses_about_the_pointers_the_validator_reports(tmp_path, defect):
    root = write_release(tmp_path, eight())
    DEFECTS[defect](root)
    with pytest.raises(crl.ReleaseError):
        crl.check(root)
    errors = [f for f in _validator(root).findings if f.severity == vd.ERROR]
    assert errors, (f"라벨 재유도 도구는 「{defect}」를 거부하는데 검증기 C7 은 아무 말도 하지 않습니다 — "
                    "두 구현이 포인터를 다르게 읽고 있습니다")
    assert all(f.check == "C7" for f in errors)


# ── 생산자가 쓰는 에피소드 샤드 ──────────────────────────────────────────────
#
# 변환기는 에피소드 샤드를 pandas 로 하나씩 다시 쓴다(convert_lerobot.merge_episode_metadata). 목록이 전부
# 빈 샤드는 그때 `list<null>` 이 되고 나머지는 `list<string>` 이다. 검증기가 「파일마다 형이 다르면 거부」하면
# 이 정상 출력을 거부하게 된다. 적재기는 표의 형을 첫 파일로 정하므로
# 빈 샤드가 **첫 파일일 때만** 열지 못한다 — 실제 적재기로 확인.

POINTERS = ["episode_index", "length", "data/chunk_index", "data/file_index", "dataset_from_index",
            "dataset_to_index"]
RULE = ["R-TS15066-SSM-01"]


def _producer_shards(root, rules):
    """LeRobot's own columns of the release, a shard an episode, and then the converter's own
    post-processor adding ``triggered_rules`` to each."""
    import generation.convert_lerobot as cl

    table = pq.read_table(episodes_path(root), columns=POINTERS)
    episodes_path(root).unlink()
    for k in range(table.num_rows):
        pq.write_table(table.slice(k, 1), episodes_path(root).with_name(f"file-{k:03d}.parquet"))
    cl.merge_episode_metadata(root, [{"episode_id": f"probe-{k}", "triggered_rules": r} for k, r in enumerate(rules)])
    return [str(pq.read_schema(f).field("triggered_rules").type)
            for f in sorted(episodes_path(root).parent.glob("*.parquet"))]


def _table_file_errors(root):
    rep = vd.Report(root=str(root))
    vd.c7_table_files(vd.Dataset(root, rep), rep)
    return [f for f in rep.findings if f.severity == vd.ERROR]


def test_the_converters_shards_are_not_refused_for_an_empty_list_the_loader_fills_in(tmp_path):
    root = write_release(tmp_path, eight())
    kinds = _producer_shards(root, [RULE, [], RULE, [], [], RULE, [], []])
    assert set(kinds) == {"list<element: string>", "list<element: null>"}, kinds   # the producer does write both
    assert kinds[0] == "list<element: string>"
    assert _table_file_errors(root) == []
    assert [f for f in _validator(root).findings if f.severity == vd.ERROR] == []


def test_the_converters_shards_are_reported_when_the_first_is_the_empty_one(tmp_path):
    """생산자 쪽의 결함이다(변환기가 샤드마다 형을 따로 정한다) — 고쳐지면 변환기가 이 형을 쓰지 않게 되어 이 시험의
    첫 단언이 알려 준다. 그때까지 검증기는 적재기가 열지 못하는 이 배포본을 통과시키면 안 된다."""
    root = write_release(tmp_path, eight())
    kinds = _producer_shards(root, [[], [], RULE, [], RULE, RULE, [], []])
    assert kinds[0] == "list<element: null>" and "list<element: string>" in kinds, kinds
    errors = _table_file_errors(root)
    assert len(errors) == kinds.count("list<element: string>"), [f.line() for f in errors]
    assert all("첫 파일의 형에 맞출 수 없음" in f.message and "triggered_rules" in f.location for f in errors)
