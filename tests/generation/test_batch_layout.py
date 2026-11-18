"""공개 배치의 디렉터리 — 기록 · 청크 계획 · 봉인."""

import dataclasses
import hashlib
import json
import os
import shutil

import pytest

from generation import batch_layout as layout
from generation import conditions
from generation.batch_layout import (BATCH_FILE, DATASET_DIR, EPISODES_DIR, MEDIA_DIR, SEAL_FILE, Batch,
                                     BatchError, ChunkPlan, load_batch, open_batch, read_seal, remove_media,
                                     seal_problems, sealed_files, write_seal)
from generation.seeding import episode_seed

COMMIT = "0123456789abcdef0123456789abcdef01234567"
ASKED = dict(parts=(("S1d", tuple(range(25))),), chunk_episodes=10, generator_commit=COMMIT, repo_id="test/batch")
# How many rows each condition-table scenario's table has: read from the canon, not written here.
TABLE_ROWS = {s: conditions.load_conditions(conditions.scenario_entry(s)).total for s in conditions.table_scenarios()}


def _id(scenario, seq):
    """An episode id as the schema writes it (§3): the seed is the seq's."""
    return f"{scenario}_{seq:04d}_seed{episode_seed(scenario, seq)}"


def _open(root, parts, chunk_episodes=10):
    return open_batch(root, parts=parts, chunk_episodes=chunk_episodes, generator_commit=COMMIT, repo_id="test/batch")


def test_a_new_batch_is_recorded_and_read_back(tmp_path):
    batch = open_batch(tmp_path / "b", **ASKED)
    assert batch == Batch(root=tmp_path / "b", **ASKED)
    assert load_batch(tmp_path / "b") == batch
    assert json.loads((tmp_path / "b" / BATCH_FILE).read_text(encoding="utf-8"))["generator_commit"] == COMMIT


def test_the_last_chunk_holds_what_is_left(tmp_path):
    batch = open_batch(tmp_path / "b", **ASKED)
    plans = batch.chunks()
    assert [p.key for p in plans] == ["0000", "0010", "0020"]
    assert [len(p.rows) for p in plans] == [10, 10, 5]
    assert batch.chunk_dir(plans[2]) == tmp_path / "b" / "chunks" / "0020"
    assert batch.episode_ids(plans[2]) == [f"S1d_{q:04d}_seed{episode_seed('S1d', q)}" for q in range(20, 25)]


def test_opening_the_same_batch_again_gives_the_same_batch(tmp_path):
    assert open_batch(tmp_path / "b", **ASKED) == open_batch(tmp_path / "b", **ASKED)


@pytest.mark.parametrize("key, other", [("generator_commit", "f" * 40), ("chunk_episodes", 20), ("repo_id", "x/y")])
def test_a_batch_is_not_continued_as_another_batch(tmp_path, key, other):
    open_batch(tmp_path / "b", **ASKED)
    with pytest.raises(BatchError, match=key) as e:
        open_batch(tmp_path / "b", **dict(ASKED, **{key: other}))
    assert repr(ASKED[key]) in str(e.value) and repr(other) in str(e.value)     # both sides, to tell which is wrong


@pytest.mark.parametrize("key, bad", [("chunk_episodes", 0), ("chunk_episodes", -10), ("chunk_episodes", True),
                                      ("chunk_episodes", 2.5)])
def test_a_batch_without_a_count_is_refused(tmp_path, key, bad):
    with pytest.raises(BatchError, match=key):
        open_batch(tmp_path / "b", **dict(ASKED, **{key: bad}))
    assert not (tmp_path / "b" / BATCH_FILE).exists()


def test_a_directory_without_a_record_is_not_a_batch(tmp_path):
    with pytest.raises(BatchError, match="batch.json"):
        load_batch(tmp_path)


# ── several scenarios' rows, and the chunks they are cut into ──

def test_old_batch_records_load_as_one_part(tmp_path):
    # The record of a batch of one scenario as the S1d batch of 400 has it: a scenario and a size, no parts.
    old = {"scenario": "S1d", "episodes": 400, "chunk_episodes": 10, "generator_commit": COMMIT,
           "repo_id": "keti/humanoid-safety-s1d", "created": "2026-10-03T15:45:28+0900"}
    (tmp_path / BATCH_FILE).write_text(json.dumps(old, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    before = (tmp_path / BATCH_FILE).read_bytes()
    batch = load_batch(tmp_path)
    assert batch.parts == (("S1d", tuple(range(400))),) and batch.scenarios == ("S1d",)
    plans = batch.chunks()
    assert [p.key for p in plans] == [f"{start:04d}" for start in range(0, 400, 10)]
    assert [p.rows for p in plans] == [tuple(range(start, start + 10)) for start in range(0, 400, 10)]
    assert all(p.scenario == "S1d" and p.per_row is False for p in plans)
    assert [batch.chunk_dir(p) for p in plans] == [tmp_path / "chunks" / f"{s:04d}" for s in range(0, 400, 10)]
    assert [eid for p in plans for eid in batch.episode_ids(p)] == [_id("S1d", seq) for seq in range(400)]
    # Continued with the arguments it was made with, it is that batch -- and its record is not written again.
    again = open_batch(tmp_path, parts=(("S1d", tuple(range(400))),), chunk_episodes=10, generator_commit=COMMIT,
                       repo_id="keti/humanoid-safety-s1d")
    assert again == batch and (tmp_path / BATCH_FILE).read_bytes() == before


@pytest.mark.parametrize("episodes, said", [(True, "episodes"), (2.5, "episodes"), ("400", "episodes"), (0, "비어")],
                         ids=["a bool", "a fraction", "a string", "none"])
def test_an_old_record_without_a_whole_size_is_refused(tmp_path, episodes, said):
    # True would read as one episode and 0 as a batch of nothing: neither is a size.
    old = {"scenario": "S1d", "episodes": episodes, "chunk_episodes": 10, "generator_commit": COMMIT,
           "repo_id": "keti/humanoid-safety-s1d", "created": "2026-10-03T15:45:28+0900"}
    (tmp_path / BATCH_FILE).write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(BatchError, match=said) as e:
        load_batch(tmp_path)
    assert str(tmp_path / BATCH_FILE) in str(e.value)


def test_table_parts_interleave_their_chunks(tmp_path):
    batch = _open(tmp_path / "b", (("S1e", tuple(range(25))), ("S1f", tuple(range(10))), ("S1g", (5, 3))))
    plans = batch.chunks()
    assert [p.key for p in plans] == ["S1e-000", "S1f-000", "S1g-000", "S1e-001", "S1e-002"]
    assert [p.scenario for p in plans] == ["S1e", "S1f", "S1g", "S1e", "S1e"]
    assert [p.rows for p in plans] == [tuple(range(10)), tuple(range(10)), (5, 3), tuple(range(10, 20)),
                                       tuple(range(20, 25))]                  # S1g's rows in the order given
    assert all(p.per_row for p in plans)
    assert batch.scenarios == ("S1e", "S1f", "S1g")
    assert batch.chunk_dir(plans[2]) == tmp_path / "b" / "chunks" / "S1g-000"
    assert batch.episode_ids(plans[2]) == [_id("S1g", 5), _id("S1g", 3)]


def test_a_row_is_its_own_run_start(tmp_path):
    (table,) = _open(tmp_path / "t", (("S1g", (5, 3, 7)),)).chunks()
    assert [table.run_start(seq) for seq in table.rows] == [5, 3, 7]
    one_run = open_batch(tmp_path / "d", **ASKED).chunks()[1]
    assert [one_run.run_start(seq) for seq in one_run.rows] == [10] * 10       # a chunk made in one run
    assert ChunkPlan("S1e-004", "S1e", (44, 42), True).run_start(42) == 42
    assert ChunkPlan("0010", "S1d", (10, 11), False).run_start(11) == 10
    with pytest.raises(dataclasses.FrozenInstanceError):
        table.rows = ()


@pytest.mark.parametrize("parts, said", [
    ((), ["parts", "비어"]),
    ((("S1e", ()),), ["parts[0] (S1e)", "비어"]),
    ((("S1e", (0, 1, 0)),), ["parts[0] (S1e)", "행 0", "두 번"]),
    ((("S1e", (0, True)),), ["parts[0] (S1e)", "True", "정수"]),
    ((("S1e", (0, 1.0)),), ["parts[0] (S1e)", "1.0", "정수"]),
    ((("S1f", (0, TABLE_ROWS["S1f"])),), ["parts[0] (S1f)", f"행 {TABLE_ROWS['S1f']}", f"0..{TABLE_ROWS['S1f'] - 1}"]),
    ((("S1f", (-1, 0)),), ["parts[0] (S1f)", "행 -1", f"0..{TABLE_ROWS['S1f'] - 1}"]),
    ((("S1e", (0,)), ("S1f", (0,)), ("S1e", (1,))), ["parts[2] (S1e)", "parts[0]"]),
    ((("S1e", (0,)), ("S1d", (0,))), ["parts[1] (S1d)", "섞지"]),
    ((("S1d", (0, 1)), ("S1e", (0,))), ["parts[1] (S1e)", "섞지"]),
    ((("S1d", (1, 2, 3)),), ["parts[0] (S1d)", "0..2"]),
    ((("S1a", (0, 2, 1)),), ["parts[0] (S1a)", "0..2"]),
    ((("S1d", (0, 1)), ("S1a", (0, 1))), ["parts[1] (S1a)", "parts[0]", "S1d"]),
], ids=["no parts", "a part without rows", "a row twice in a part", "a row that is a bool",
        "a row that is a float", "a table row past the table", "a table row below it", "one scenario in two parts",
        "a table scenario then another", "another scenario then a table one", "S1d rows not from 0",
        "S1a rows out of order", "two scenarios without a table"])
def test_parts_are_refused_when(tmp_path, parts, said):
    with pytest.raises(BatchError) as e:
        _open(tmp_path / "b", parts)
    for words in said:                                       # what is wrong, and where
        assert words in str(e.value), (words, str(e.value))
    assert not (tmp_path / "b" / BATCH_FILE).exists()


def test_a_recorded_parts_list_is_checked_as_an_asked_one(tmp_path):
    # A record edited by hand (or by another version of this code) is not taken on trust.
    record = {"parts": [{"scenario": "S1e", "rows": [3, 4, 3]}], "chunk_episodes": 10, "generator_commit": COMMIT,
              "repo_id": "test/batch", "created": "2026-10-22T00:00:00+0900"}
    (tmp_path / BATCH_FILE).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(BatchError, match="두 번") as e:
        load_batch(tmp_path)
    assert str(tmp_path / BATCH_FILE) in str(e.value) and "parts[0] (S1e)" in str(e.value)


@pytest.mark.parametrize("first, then, said", [
    ((("S1e", (0, 1, 2)), ("S1f", (0, 1, 2))), (("S1e", (0, 1, 2)), ("S1f", (0, 1, 3))), ["S1f 의 행", "자리 2", "기록 2 ≠ 요청 3"]),
    ((("S1e", (0, 1, 2)), ("S1f", (0, 1, 2))), (("S1f", (0, 1, 2)), ("S1e", (0, 1, 2))), ["차례", "기록 ['S1e', 'S1f'] ≠ 요청 ['S1f', 'S1e']"]),
    ((("S1e", (0, 1, 2)), ("S1f", (0, 1, 2))), (("S1e", (0, 1, 2)),), ["차례", "기록 ['S1e', 'S1f'] ≠ 요청 ['S1e']"]),
    ((("S1e", (0, 1, 2)),), (("S1g", (0, 1, 2)),), ["차례", "기록 ['S1e'] ≠ 요청 ['S1g']"]),
    ((("S1d", tuple(range(25))),), (("S1d", tuple(range(400))),), ["S1d 의 행", "기록 25행", "요청 400행"]),
    ((("S1d", tuple(range(25))),), (("S1a", tuple(range(25))),), ["차례", "기록 ['S1d'] ≠ 요청 ['S1a']"]),
], ids=["one row", "the scenarios' order", "a part fewer", "another table scenario", "another size", "another scenario"])
def test_a_batch_resumes_only_with_the_same_parts(tmp_path, first, then, said):
    _open(tmp_path / "b", first)
    before = (tmp_path / "b" / BATCH_FILE).read_bytes()
    assert _open(tmp_path / "b", first).parts == first
    with pytest.raises(BatchError, match="parts") as e:
        _open(tmp_path / "b", then)
    for words in said:
        assert words in str(e.value), (words, str(e.value))
    with pytest.raises(BatchError, match="chunk_episodes"):          # and the chunk size, with the same parts
        _open(tmp_path / "b", first, chunk_episodes=5)
    assert (tmp_path / "b" / BATCH_FILE).read_bytes() == before


def test_the_new_record_round_trips(tmp_path):
    batch = _open(tmp_path / "b", [("S1g", [5, 3]), ("S1e", range(3))], chunk_episodes=2)
    assert batch.parts == (("S1g", (5, 3)), ("S1e", (0, 1, 2)))                # kept as tuples, whatever was given
    doc = json.loads((tmp_path / "b" / BATCH_FILE).read_text(encoding="utf-8"))
    assert sorted(doc) == ["chunk_episodes", "created", "generator_commit", "parts", "repo_id"]
    assert doc["parts"] == [{"scenario": "S1g", "rows": [5, 3]}, {"scenario": "S1e", "rows": [0, 1, 2]}]
    read = load_batch(tmp_path / "b")
    assert read == batch and read.chunks() == batch.chunks()
    assert [p.key for p in read.chunks()] == ["S1g-000", "S1e-000", "S1e-001"]


def _chunk(tmp_path):
    chunk = tmp_path / "chunks" / "0000"
    (chunk / EPISODES_DIR / MEDIA_DIR).mkdir(parents=True)
    (chunk / DATASET_DIR / "meta").mkdir(parents=True)
    (chunk / EPISODES_DIR / "S1d_0000_seed1.npz").write_bytes(b"arrays")
    (chunk / EPISODES_DIR / "S1d_0000_seed1.json").write_text("{}", encoding="utf-8")
    (chunk / EPISODES_DIR / MEDIA_DIR / "S1d_0000_seed1.head_rgb.mkv").write_bytes(b"lossless")
    (chunk / DATASET_DIR / "meta" / "info.json").write_text("{}", encoding="utf-8")
    return chunk


def test_the_seal_lists_what_the_chunk_keeps_not_the_lossless_video(tmp_path):
    chunk = _chunk(tmp_path)
    files = sealed_files(chunk)
    assert sorted(files) == ["dataset/meta/info.json", "episodes/S1d_0000_seed1.json", "episodes/S1d_0000_seed1.npz"]
    assert all(len(h) == 64 for h in files.values())


def test_a_chunk_is_sealed_only_once_its_record_is_whole(tmp_path):
    chunk = _chunk(tmp_path)
    assert read_seal(chunk) is None
    write_seal(chunk, {"start_seq": 0, "files": sealed_files(chunk)})
    assert read_seal(chunk)["start_seq"] == 0
    assert not list(chunk.glob("*.tmp")), "쓰다 만 봉인 기록이 남았습니다"


@pytest.mark.parametrize("text", ["", "{", "[]", '{"start_seq": 0}'],
                         ids=["zero length", "cut short", "not an object", "no file list"])
def test_a_seal_that_cannot_be_read_is_an_error_not_an_unsealed_chunk(tmp_path, text):
    # 못 읽는 봉인을 「봉인 안 됨」으로 보면 실행기가 그 청크를 지우고 다시 만든다.
    # 길이 0 은 하드 리셋 뒤에 fsync 없이 쓴 봉인이 돌아오는 꼴이다.
    chunk = _chunk(tmp_path)
    (chunk / SEAL_FILE).write_text(text, encoding="utf-8")
    with pytest.raises(BatchError, match=SEAL_FILE) as e:
        read_seal(chunk)
    # The error ends with what to do: the runner alone cannot tell this from a seal worth keeping.
    assert f"{SEAL_FILE} 를 지우고" in str(e.value) and "처음부터" in str(e.value)


def test_a_seal_is_on_disk_before_it_is_named_and_its_directory_after(tmp_path, monkeypatch):
    # fsync the temporary file, rename it, fsync the directory: a hard reset right after a seal must not
    # bring back a zero-length SEALED.json (the file's data not yet written) or none at all (the rename).
    chunk = _chunk(tmp_path)
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append(("fsync", os.readlink(f"/proc/self/fd/{fd}")))
        return real_fsync(fd)

    def replace(src, dst):
        events.append(("replace", str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(layout.os, "fsync", fsync)
    monkeypatch.setattr(layout.os, "replace", replace)
    write_seal(chunk, {"start_seq": 0, "files": sealed_files(chunk)})
    assert events == [("fsync", str(chunk / (SEAL_FILE + ".tmp"))), ("replace", str(chunk / SEAL_FILE)),
                      ("fsync", str(chunk))]


def test_removing_the_media_says_whether_there_was_any(tmp_path):
    chunk = _chunk(tmp_path)
    assert remove_media(chunk) is True
    assert not (chunk / EPISODES_DIR / MEDIA_DIR).exists()
    assert remove_media(chunk) is False


def test_a_sealed_chunk_that_changed_says_what_changed(tmp_path):
    chunk = _chunk(tmp_path)
    record = {"files": sealed_files(chunk)}
    remove_media(chunk)
    assert seal_problems(chunk, record) == []
    (chunk / DATASET_DIR / "meta" / "info.json").write_text('{"x": 1}', encoding="utf-8")
    (chunk / EPISODES_DIR / "S1d_0000_seed1.npz").unlink()
    (chunk / DATASET_DIR / "stray.txt").write_text("x", encoding="utf-8")
    said = " ".join(seal_problems(chunk, record))
    assert "dataset/meta/info.json" in said and "episodes/S1d_0000_seed1.npz" in said and "dataset/stray.txt" in said


def _elsewhere(tmp_path, files=("file-000.mp4",)):
    """A directory outside the chunk, as a link inside the chunk's dataset would point at."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for name in files:
        (elsewhere / name).write_bytes(b"not sealed")
    return elsewhere


@pytest.mark.parametrize("where", ["videos", "meta/episodes", "assets/textures/more"],
                         ids=["under the dataset", "beside a sealed file", "three levels down"])
def test_a_link_to_a_directory_inside_the_dataset_is_refused(tmp_path, where):
    # The walk does not go below a link to a directory, so the files under one would be in no seal --
    # and the assembly opens the dataset by path and reads them.
    chunk = _chunk(tmp_path)
    link = chunk / DATASET_DIR / where
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(_elsewhere(tmp_path), target_is_directory=True)
    with pytest.raises(BatchError, match="심링크") as e:
        sealed_files(chunk)
    assert f"{DATASET_DIR}/{where}" in str(e.value) and str(chunk) in str(e.value)      # which link, of which chunk


@pytest.mark.parametrize("target", ["an empty directory", "a directory of the dataset itself"])
def test_a_link_to_a_directory_is_refused_whatever_the_directory_holds(tmp_path, target):
    # Nothing under the link to leave out today is not a reason to pass it: what it points at can be filled later.
    chunk = _chunk(tmp_path)
    to = _elsewhere(tmp_path, files=()) if target == "an empty directory" else chunk / DATASET_DIR / "meta"
    (chunk / DATASET_DIR / "linked").symlink_to(to, target_is_directory=True)
    with pytest.raises(BatchError, match=f"{DATASET_DIR}/linked"):
        sealed_files(chunk)


def test_every_link_to_a_directory_is_named(tmp_path):
    chunk = _chunk(tmp_path)
    elsewhere = _elsewhere(tmp_path)
    for name in ("videos", "data"):
        (chunk / DATASET_DIR / name).symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(BatchError) as e:
        sealed_files(chunk)
    assert f"{DATASET_DIR}/data" in str(e.value) and f"{DATASET_DIR}/videos" in str(e.value)


def test_a_link_to_a_directory_put_in_after_the_seal_is_refused_when_the_seal_is_checked(tmp_path):
    # No sealed file is gone, added or changed -- the comparison with the record alone would say "unchanged".
    chunk = _chunk(tmp_path)
    record = {"files": sealed_files(chunk)}
    remove_media(chunk)
    (chunk / DATASET_DIR / "videos").symlink_to(_elsewhere(tmp_path), target_is_directory=True)
    with pytest.raises(BatchError, match=f"{DATASET_DIR}/videos"):
        seal_problems(chunk, record)


def test_a_link_to_a_file_is_sealed_as_the_bytes_it_points_at(tmp_path):
    # Not every link is refused: the hash follows a link to a file, and a change of what it points at shows.
    chunk = _chunk(tmp_path)
    target = _elsewhere(tmp_path) / "file-000.mp4"
    (chunk / DATASET_DIR / "meta" / "linked.mp4").symlink_to(target)
    record = {"files": sealed_files(chunk)}
    assert record["files"][f"{DATASET_DIR}/meta/linked.mp4"] == hashlib.sha256(b"not sealed").hexdigest()
    target.write_bytes(b"changed")
    assert seal_problems(chunk, record) == [f"바뀐 파일: {DATASET_DIR}/meta/linked.mp4"]


@pytest.mark.parametrize("root", [DATASET_DIR, EPISODES_DIR])
def test_a_chunk_whose_dataset_or_episodes_directory_is_itself_a_link_is_sealed_whole(tmp_path, root):
    # The refusal is of a link met inside the dataset. The directory the walk starts from may be one:
    # its files are listed all the same.
    chunk = _chunk(tmp_path)
    before = sealed_files(chunk)
    (chunk / root).rename(tmp_path / "moved")
    (chunk / root).symlink_to(tmp_path / "moved", target_is_directory=True)
    assert sealed_files(chunk) == before


def test_a_directory_under_the_episodes_is_passed_over_as_the_lossless_video_is(tmp_path):
    # episodes/ is listed flat -- the lossless media is a directory there when the seal is taken -- and the
    # assembly reads only the files directly under it. A link to a directory there is not what the refusal is of.
    chunk = _chunk(tmp_path)
    before = sealed_files(chunk)
    (chunk / EPISODES_DIR / "linked").symlink_to(_elsewhere(tmp_path), target_is_directory=True)
    assert sealed_files(chunk) == before


@pytest.mark.skipif(os.geteuid() == 0, reason="root lists a directory whatever its mode")
@pytest.mark.parametrize("where", ["meta", ""], ids=["a directory inside", "the dataset directory"])
def test_a_directory_of_the_dataset_that_cannot_be_listed_is_refused(tmp_path, where):
    # Passed over, the files under it would be missing from the seal without a word -- and a file there is
    # still opened by its path: the directory can be entered, only not listed.
    chunk = _chunk(tmp_path)
    closed = chunk / DATASET_DIR / where
    closed.chmod(0o333)
    try:
        assert (chunk / DATASET_DIR / "meta" / "info.json").read_text(encoding="utf-8") == "{}"
        with pytest.raises(BatchError, match="열지 못") as e:
            sealed_files(chunk)
    finally:
        closed.chmod(0o755)
    assert (closed.relative_to(chunk).as_posix() + " ") in str(e.value) and str(chunk) in str(e.value)


def test_a_dataset_that_is_gone_is_said_file_by_file(tmp_path):
    # Not the refusal of a directory that cannot be listed: there is nothing to list, and what the record
    # names there is what is missing.
    chunk = _chunk(tmp_path)
    (chunk / DATASET_DIR / "meta" / "stats.json").write_text("{}", encoding="utf-8")
    record = {"files": sealed_files(chunk)}
    shutil.rmtree(chunk / DATASET_DIR)
    assert seal_problems(chunk, record) == [f"없어진 파일: {DATASET_DIR}/meta/info.json",
                                            f"없어진 파일: {DATASET_DIR}/meta/stats.json"]


def test_a_link_to_a_directory_is_refused_when_the_walk_took_it_for_a_file(tmp_path, monkeypatch):
    # os.walk hands an entry back among the files when asking whether it is a directory failed (an I/O error),
    # and says nothing of it: left to that, the link would be neither gone below nor a file -- passed over.
    chunk = _chunk(tmp_path)
    (chunk / DATASET_DIR / "videos").symlink_to(_elsewhere(tmp_path), target_is_directory=True)
    real = os.walk

    def walk(top, **kw):
        for here, dirs, names in real(top, **kw):
            links = [d for d in dirs if os.path.islink(os.path.join(here, d))]
            dirs[:] = [d for d in dirs if d not in links]
            yield here, dirs, names + links

    monkeypatch.setattr(layout.os, "walk", walk)
    with pytest.raises(BatchError, match=f"{DATASET_DIR}/videos"):
        sealed_files(chunk)
