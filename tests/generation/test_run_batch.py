"""배치 실행기 — 청크 순회 · 봉인 · 재개 · 멈춤 감시.

단계는 주입한다: 진짜 단계는 Isaac 과 변환기를 부르는 서브프로세스이고, 여기서 보는 것은 그것들을
어떤 순서로 부르고 무엇을 남기고 무엇을 지우는가다. 실제 명령은 리허설(실제 Isaac 으로 2청크)이 돈다.
"""

import inspect
import io
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from generation import conditions
from generation import run_batch as rb
from generation.batch_layout import (DATASET_DIR, EPISODES_DIR, LOCK_FILE, LOGS_DIR, MEDIA_DIR, RUNNER_FILE,
                                     SEAL_FILE, STOP_FILE, BatchError, open_batch, read_seal, sealed_files)

COMMIT = "0123456789abcdef0123456789abcdef01234567"
REPO = Path(__file__).resolve().parents[2]
# The cameras whose lossless streams a recording names in its sidecar's media: the camera canon's.
CAMERAS = tuple(json.loads((REPO / "meta" / "cameras.json").read_text(encoding="utf-8"))["cameras"])


def _batch(tmp_path, episodes=25):
    return open_batch(tmp_path / "batch", parts=(("S1d", tuple(range(episodes))),), chunk_episodes=10,
                      generator_commit=COMMIT, repo_id="test/batch")


def _plan(batch, start):
    """The plan of the chunk whose first row is ``start``."""
    (plan,) = [p for p in batch.chunks() if p.rows[0] == start]
    return plan


def _chunk(batch, start):
    return batch.chunk_dir(_plan(batch, start))


def _clean():
    """What provenance.read says of the working tree the batch was started in."""
    return {"generator_commit": COMMIT, "generator_dirty": False}


def _run(batch, stages, *, say, tree=None):
    """rb.run with the tree the runner reads before a chunk and before a seal held by ``tree`` (a dict a
    stage may change): the runner never asks git in these tests."""
    state = tree if tree is not None else _clean()
    return rb.run(batch, stages, read_tree=lambda: dict(state), say=say)


def _table_batch(tmp_path, parts=(("S1e", (4, 9, 2, 7, 5)),), chunk=10):
    """A batch of condition-table rows. The rows are out of order on purpose: a chunk makes them in the order given."""
    return open_batch(tmp_path / "batch", parts=parts, chunk_episodes=chunk, generator_commit=COMMIT,
                      repo_id="test/batch")


def _eid(batch, plan, seq):
    return batch.episode_ids(plan)[plan.rows.index(seq)]


def _row_files(chunk, eid):
    """What the chunk holds of row ``eid``: its files in episodes/ and its streams, finished or not, in media/."""
    out = []
    for where in (chunk / EPISODES_DIR, chunk / EPISODES_DIR / MEDIA_DIR):
        if where.is_dir():
            out += [p for p in where.iterdir() if p.name.startswith((f"{eid}.", f".{eid}."))]
    return sorted(out)


def _write_row(batch, plan, seq, chunk, sidecar=None):
    """What a run of one row leaves when it ends well, as run_s1a.py writes it: the npz, the sidecar -- its media
    naming one lossless stream per camera of the canon under media/, with the recorder's counts -- and the streams.
    ``sidecar`` edits the sidecar before it is written."""
    eid, episodes = _eid(batch, plan, seq), chunk / EPISODES_DIR
    (episodes / MEDIA_DIR).mkdir(parents=True, exist_ok=True)
    media = {cam: {"file": f"{MEDIA_DIR}/{eid}.{cam}.mkv", "frames": 3} for cam in CAMERAS}
    media.update(rerendered_steps=[], blank_frames={cam: 0 for cam in CAMERAS}, canon_fingerprint="f" * 64)
    side = {"episode_id": eid, "run_start_seq": seq, "media": media,
            "sim_backend": {"generator_commit": batch.generator_commit, "generator_dirty": False}}
    if sidecar:
        sidecar(side)
    (episodes / f"{eid}.npz").write_bytes(b"npz " + eid.encode())
    (episodes / f"{eid}.json").write_text(json.dumps(side), encoding="utf-8")
    for cam in CAMERAS:
        (episodes / MEDIA_DIR / f"{eid}.{cam}.mkv").write_bytes(f"lossless {eid} {cam}".encode())


def _card_report(scenario):
    """What scorecard.py --out writes, as far as a seal reads it: a chunk of rows holds one scenario."""
    return {"scorecard_version": 1, "approach_speed_threshold_mps": 0.05, "fps": 50,
            "scenarios": {scenario: {"episodes": 5, "grid_episodes": 0, "degenerate_excluded": 1,
                                     "safety": {"outcome": {"fail": 2, "near_miss": 1, "nominal": 1}},
                                     "anomalies": {"robot_falls": 2, "falls_without_contact": 1,
                                                   "rack_contact_episodes": 1, "rack_contact_events": 3}}}}


# What the validator's --json and measure_slice.py's --out write, as far as the seal reads them
# (measure_slice's spread() puts math.inf in "range": the file holds Infinity, which json reads back).
VALIDATE_REPORT = {"root": "x", "schema_version": "0.1", "summary": {"error": 0, "warn": 1, "info": 4, "skip": 0},
                   "by_check": {"C1": {"WARN": 1}}, "findings": []}


def _spread(n, lo, mid, hi):
    return {"n": n, "min": lo, "p5": lo, "median": mid, "p95": hi, "max": hi, "range": [0.0, math.inf],
            "inside_fraction": 1.0}


MEASURE_REPORT = {"episodes": [], "problems": [],
                  "summary": {"episodes": 10, "entry_triggered": 9, "entry_speed_ratio": _spread(9, 0.2892, 1.0024, 1.0277),
                              "entry_speed_ratio_held": _spread(6, 1.0018, 1.0026, 1.0049),
                              "facing_off_travel_deg_max": 0.0,
                              "outcome": {"fail": 3, "near_miss": 4, "nominal": 3}}}


class FakeStages:
    """The five stages, writing what the real ones leave behind. ``codes`` maps a call to the exit codes to return
    in turn (default 0): (stage, start) for a chunk that is one run, start its first row; (stage, key) for a chunk
    of condition-table rows, and ("generate", key, row) for the run of one of its rows. ``sidecar`` edits every
    sidecar written. A row is made from nothing: its run starts with none of its files left in the chunk."""

    def __init__(self, codes=None, sidecar=None, drop=0, reports=True):
        self.calls, self.codes, self.sidecar, self.drop = [], dict(codes or {}), sidecar, drop
        self.reports = reports                  # whether validate and measure leave their --json / --out files

    def _code(self, *call):
        self.calls.append(call)
        queue = self.codes.get(call)
        return queue.pop(0) if queue else 0

    @staticmethod
    def _at(plan):
        return plan.key if plan.per_row else plan.rows[0]

    def generate(self, batch, plan, chunk):
        start = plan.rows[0]
        code = self._code("generate", start)
        if code == rb.STALLED:
            (chunk / EPISODES_DIR / "half-written.npz").write_bytes(b"x")
            attempt = len([c for c in self.calls if c == ("generate", start)])
            (chunk / LOGS_DIR / "generate.log").write_text(f"stalled, attempt {attempt}", encoding="utf-8")
            return code
        (chunk / EPISODES_DIR / MEDIA_DIR).mkdir()
        for eid in batch.episode_ids(plan)[:len(plan.rows) - self.drop]:
            side = {"episode_id": eid, "run_start_seq": start,
                    "sim_backend": {"generator_commit": batch.generator_commit, "generator_dirty": False}}
            if self.sidecar:
                self.sidecar(side)
            (chunk / EPISODES_DIR / f"{eid}.npz").write_bytes(eid.encode())
            (chunk / EPISODES_DIR / f"{eid}.json").write_text(json.dumps(side), encoding="utf-8")
            (chunk / EPISODES_DIR / MEDIA_DIR / f"{eid}.head_rgb.mkv").write_bytes(b"lossless " + eid.encode())
        return code

    def generate_row(self, batch, plan, seq, chunk):
        eid = _eid(batch, plan, seq)
        assert not _row_files(chunk, eid), f"행 {seq} 을 남은 파일 위에서 만듭니다: {_row_files(chunk, eid)}"
        code = self._code("generate", plan.key, seq)
        if code == rb.STALLED:
            (chunk / EPISODES_DIR / f"{eid}.npz").write_bytes(b"half")
            attempt = self.calls.count(("generate", plan.key, seq))
            rb.row_log(chunk, seq).write_text(f"stalled, row {seq}, attempt {attempt}", encoding="utf-8")
            return code
        _write_row(batch, plan, seq, chunk, self.sidecar)
        rb.row_log(chunk, seq).write_text(f"made row {seq}", encoding="utf-8")
        return code

    def label(self, batch, plan, chunk):
        for eid in batch.episode_ids(plan):
            (chunk / EPISODES_DIR / f"{eid}.labels.json").write_text("{}", encoding="utf-8")
        return self._code("label", self._at(plan))

    def convert(self, batch, plan, chunk):
        (chunk / DATASET_DIR / "meta").mkdir(parents=True)
        (chunk / DATASET_DIR / "meta" / "info.json").write_text(json.dumps({"start": plan.rows[0]}), encoding="utf-8")
        return self._code("convert", self._at(plan))

    def validate(self, batch, plan, chunk):
        (chunk / LOGS_DIR / "validate.log").write_text(f"validated {plan.rows[0]}", encoding="utf-8")
        if self.reports:
            (chunk / LOGS_DIR / "validate.json").write_text(json.dumps(VALIDATE_REPORT), encoding="utf-8")
        return self._code("validate", self._at(plan))

    def measure(self, batch, plan, chunk):
        if self.reports:
            report = _card_report(plan.scenario) if plan.per_row else MEASURE_REPORT
            (chunk / LOGS_DIR / "measure.json").write_text(json.dumps(report), encoding="utf-8")
        return self._code("measure", self._at(plan))


def _said():
    lines = []
    return lines, lines.append


ALL = ["generate", "label", "convert", "validate", "measure"]


def test_every_chunk_is_generated_checked_and_sealed_in_order(tmp_path):
    batch, stages = _batch(tmp_path), FakeStages()
    lines, say = _said()
    assert _run(batch, stages, say=say) == 0
    assert stages.calls == [(stage, start) for start in (0, 10, 20) for stage in ALL]
    for start in (0, 10, 20):
        record = read_seal(_chunk(batch, start))
        assert record["start_seq"] == start and record["episode_ids"] == batch.episode_ids(_plan(batch, start))
        assert record["episodes"] == len(_plan(batch, start).rows) and record["generator_commit"] == COMMIT
        assert record["files"] == sealed_files(_chunk(batch, start))
        assert set(record["stages_s"]) == set(ALL)
        assert (record["scenario"], record["rows"]) == ("S1d", list(_plan(batch, start).rows))
        assert "rows_s" not in record, "실행 하나인 청크에는 행마다의 시간이 없습니다"


def test_the_lossless_video_goes_once_the_chunk_is_sealed_and_not_before(tmp_path):
    batch = _batch(tmp_path, episodes=10)

    class Watching(FakeStages):
        def measure(self, batch, plan, chunk):
            assert (chunk / EPISODES_DIR / MEDIA_DIR).is_dir(), "측정이 읽을 무손실이 벌써 없습니다"
            assert not (chunk / SEAL_FILE).exists()
            return super().measure(batch, plan, chunk)

    assert _run(batch, Watching(), say=lambda s: None) == 0
    chunk = _chunk(batch, 0)
    assert (chunk / SEAL_FILE).is_file() and not (chunk / EPISODES_DIR / MEDIA_DIR).exists()
    assert not any(name.startswith(f"{EPISODES_DIR}/{MEDIA_DIR}/") for name in read_seal(chunk)["files"])


def test_a_sealed_chunk_is_not_made_again(tmp_path):
    batch = _batch(tmp_path)
    assert _run(batch, FakeStages(), say=lambda s: None) == 0
    again = FakeStages()
    assert _run(batch, again, say=lambda s: None) == 0
    assert again.calls == []


def test_a_chunk_that_was_not_sealed_is_made_again_from_nothing(tmp_path):
    batch = _batch(tmp_path)
    first = FakeStages(codes={("convert", 10): [1]})
    assert _run(batch, first, say=lambda s: None) == 1
    assert read_seal(_chunk(batch, 0)) is not None and read_seal(_chunk(batch, 10)) is None
    (_chunk(batch, 10) / EPISODES_DIR / "left-by-the-run-that-died.txt").write_text("x", encoding="utf-8")
    second = FakeStages()
    assert _run(batch, second, say=lambda s: None) == 0
    assert second.calls[0] == ("generate", 10), "끊긴 청크를 단계 중간에서 이어 갔습니다"
    assert not (_chunk(batch, 10) / EPISODES_DIR / "left-by-the-run-that-died.txt").exists()
    assert ("generate", 0) not in second.calls


def test_a_seal_written_just_before_an_interruption_is_finished_not_redone(tmp_path):
    # 봉인 기록이 먼저, 삭제가 나중이다. 그 사이에 끊기면 봉인됐는데 무손실이 남은 청크가 된다.
    batch = _batch(tmp_path, episodes=10)
    assert _run(batch, FakeStages(), say=lambda s: None) == 0
    media = _chunk(batch, 0) / EPISODES_DIR / MEDIA_DIR
    media.mkdir()
    (media / "S1d_0000.head_rgb.mkv").write_bytes(b"lossless")
    again = FakeStages()
    assert _run(batch, again, say=lambda s: None) == 0
    assert again.calls == [] and not media.exists()


@pytest.mark.parametrize("stage", ["label", "convert", "validate", "measure"])
def test_a_stage_that_fails_stops_the_runner_and_keeps_the_lossless_video(tmp_path, stage):
    batch = _batch(tmp_path)
    stages = FakeStages(codes={(stage, 0): [1, 1]})
    lines, say = _said()
    assert _run(batch, stages, say=say) == 1
    assert ("generate", 10) not in stages.calls, "실패한 청크를 넘어 다음 청크를 시작했습니다"
    assert read_seal(_chunk(batch, 0)) is None
    assert (_chunk(batch, 0) / EPISODES_DIR / MEDIA_DIR).is_dir(), "사람이 볼 무손실을 지웠습니다"
    assert any(stage in line and "0000" in line for line in lines), lines


def test_a_generation_that_stalls_is_tried_again_twice_from_nothing(tmp_path):
    batch = _batch(tmp_path, episodes=10)
    stages = FakeStages(codes={("generate", 0): [rb.STALLED, rb.STALLED]})
    assert _run(batch, stages, say=lambda s: None) == 0
    assert [c for c in stages.calls if c[0] == "generate"] == [("generate", 0)] * 3
    assert not (_chunk(batch, 0) / EPISODES_DIR / "half-written.npz").exists()


def test_a_generation_that_stalls_three_times_stops_the_runner(tmp_path):
    batch = _batch(tmp_path, episodes=10)
    stages = FakeStages(codes={("generate", 0): [rb.STALLED] * 3})
    lines, say = _said()
    assert _run(batch, stages, say=say) == 1
    assert len([c for c in stages.calls if c[0] == "generate"]) == 3
    assert not [c for c in stages.calls if c[0] == "label"]


def test_a_validator_that_crashed_is_run_once_more(tmp_path):
    # 종료 코드 3 은 결함 보고가 아니라 검증기가 죽은 것이다(검증기의 알려진 간헐 크래시).
    batch = _batch(tmp_path, episodes=10)
    stages = FakeStages(codes={("validate", 0): [3]})
    assert _run(batch, stages, say=lambda s: None) == 0
    assert len([c for c in stages.calls if c[0] == "validate"]) == 2
    assert (_chunk(batch, 0) / LOGS_DIR / "validate.crash.log").read_text(encoding="utf-8") == "validated 0"


@pytest.mark.parametrize("code", [-signal.SIGSEGV, -signal.SIGABRT])
def test_a_validator_killed_by_a_signal_is_run_once_more_as_well(tmp_path, code):
    # The known flake (the validator's intermittent crash) dies of SIGSEGV: a negative return code, not exit 3.
    batch = _batch(tmp_path, episodes=10)
    stages = FakeStages(codes={("validate", 0): [code]})
    lines, say = _said()
    assert _run(batch, stages, say=say) == 0
    assert len([c for c in stages.calls if c[0] == "validate"]) == 2
    assert (_chunk(batch, 0) / LOGS_DIR / "validate.crash.log").read_text(encoding="utf-8") == "validated 0"
    assert any(str(code) in line for line in lines), lines


def test_a_validator_killed_by_a_signal_twice_stops_the_runner(tmp_path):
    batch = _batch(tmp_path, episodes=10)
    stages = FakeStages(codes={("validate", 0): [-11, -11]})
    assert _run(batch, stages, say=lambda s: None) == 1
    assert len([c for c in stages.calls if c[0] == "validate"]) == 2


def test_a_validator_that_crashed_twice_stops_the_runner(tmp_path):
    batch = _batch(tmp_path, episodes=10)
    assert _run(batch, FakeStages(codes={("validate", 0): [3, 3]}), say=lambda s: None) == 1


def test_a_validator_that_found_an_error_is_not_run_again(tmp_path):
    batch = _batch(tmp_path, episodes=10)
    stages = FakeStages(codes={("validate", 0): [1]})
    assert _run(batch, stages, say=lambda s: None) == 1
    assert len([c for c in stages.calls if c[0] == "validate"]) == 1


@pytest.mark.parametrize("stages, phrase", [
    (FakeStages(drop=1), "에피소드가 요청과 다릅니다"),
    (FakeStages(sidecar=lambda s: s.update(run_start_seq=0)), "run_start_seq"),
    (FakeStages(sidecar=lambda s: s["sim_backend"].update(generator_commit="f" * 40)), "generator_commit"),
    (FakeStages(sidecar=lambda s: s["sim_backend"].update(generator_dirty=True)), "generator_dirty"),
    (FakeStages(sidecar=lambda s: s.update(episode_id="S1d_9999_seed1")), "episode_id"),
], ids=["one episode short", "another run's start", "another commit", "a dirty tree", "another id in the sidecar"])
def test_episodes_that_are_not_the_ones_asked_for_stop_the_chunk(tmp_path, stages, phrase):
    batch = _batch(tmp_path)
    lines, say = _said()
    # Chunk 0 of "another run's start" is right (its start is 0): that fault shows at chunk 10.
    code = _run(batch, stages, say=say)
    assert code == 1 and any(phrase in line for line in lines), lines
    assert not [c for c in stages.calls if c[0] == "convert" and read_seal(_chunk(batch, c[1])) is None]


def test_a_stop_file_ends_the_run_after_the_chunk_at_hand(tmp_path):
    batch = _batch(tmp_path)

    class Stopping(FakeStages):
        def measure(self, batch, plan, chunk):
            (batch.root / STOP_FILE).write_text("", encoding="utf-8")
            return super().measure(batch, plan, chunk)

    stages = Stopping()
    assert _run(batch, stages, say=lambda s: None) == 0
    assert read_seal(_chunk(batch, 0)) is not None, "지금 청크를 끝내지 않고 멈췄습니다"
    assert ("generate", 10) not in stages.calls


def test_a_batch_of_three_table_scenarios_runs_its_chunks_in_turn(tmp_path):
    batch = _table_batch(tmp_path, parts=(("S1e", (0, 1, 2)), ("S1f", (7, 3)), ("S1g", (1,))), chunk=2)
    stages = FakeStages()
    assert _run(batch, stages, say=lambda s: None) == 0
    assert [c[1] for c in stages.calls if c[0] == "label"] == [p.key for p in batch.chunks()] == [
        "S1e-000", "S1f-000", "S1g-000", "S1e-001"]
    assert [c[1:] for c in stages.calls if c[0] == "generate"] == [
        ("S1e-000", 0), ("S1e-000", 1), ("S1f-000", 7), ("S1f-000", 3), ("S1g-000", 1), ("S1e-001", 2)]
    for plan in batch.chunks():
        record = read_seal(batch.chunk_dir(plan))
        assert (record["scenario"], record["rows"], record["start_seq"]) == (plan.scenario, list(plan.rows), plan.rows[0])
        assert record["episode_ids"] == batch.episode_ids(plan)


# ── the working tree is read again: before a chunk starts and before its seal is written ──

def test_the_tree_is_read_before_each_chunk_and_just_before_its_seal(tmp_path):
    batch = _batch(tmp_path, episodes=10)
    stages = FakeStages()
    reads = lambda: stages.calls.append(("tree", 0)) or _clean()
    assert rb.run(batch, stages, read_tree=reads, say=lambda s: None) == 0
    assert stages.calls == [("tree", 0)] + [(stage, 0) for stage in ALL] + [("tree", 0)]


def test_a_tree_that_turns_dirty_while_a_chunk_is_made_is_not_sealed(tmp_path):
    # An edit of meta/rules.json or the labeller during generation: labelling and conversion ran on it,
    # and nothing downstream re-derives the labels -- the chunk would be sealed under the batch's commit.
    batch, tree = _batch(tmp_path), _clean()

    class Editing(FakeStages):
        def convert(self, batch, plan, chunk):
            tree["generator_dirty"] = True
            return super().convert(batch, plan, chunk)

    stages, (lines, say) = Editing(), _said()
    assert _run(batch, stages, say=say, tree=tree) == 1
    assert read_seal(_chunk(batch, 0)) is None
    assert (_chunk(batch, 0) / EPISODES_DIR / MEDIA_DIR).is_dir(), "봉인하지 않은 청크의 무손실을 지웠습니다"
    assert ("generate", 10) not in stages.calls
    assert any("커밋되지 않은 변경" in line and "0000" in line for line in lines), lines


def test_a_head_that_moved_while_a_chunk_is_made_is_not_sealed(tmp_path):
    batch, tree = _batch(tmp_path), _clean()
    moved = "f" * 40

    class Switching(FakeStages):
        def validate(self, batch, plan, chunk):
            tree["generator_commit"] = moved
            return super().validate(batch, plan, chunk)

    stages, (lines, say) = Switching(), _said()
    assert _run(batch, stages, say=say, tree=tree) == 1
    assert read_seal(_chunk(batch, 0)) is None and (_chunk(batch, 0) / EPISODES_DIR / MEDIA_DIR).is_dir()
    assert any(moved in line and COMMIT in line for line in lines), "어느 커밋이 보였는지 말해야 합니다"


def test_a_chunk_whose_dataset_holds_a_link_to_a_directory_is_not_sealed(tmp_path):
    # A seal lists what its walk reaches, and the walk does not go below a link to a directory: sealed, the
    # chunk would vouch for files it never hashed.
    batch = _batch(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "file-000.mp4").write_bytes(b"not sealed")

    class Linking(FakeStages):
        def convert(self, batch, plan, chunk):
            code = super().convert(batch, plan, chunk)
            (chunk / DATASET_DIR / "videos").symlink_to(elsewhere, target_is_directory=True)
            return code

    stages, (lines, say) = Linking(), _said()
    assert _run(batch, stages, say=say) == 1
    assert read_seal(_chunk(batch, 0)) is None
    assert (_chunk(batch, 0) / EPISODES_DIR / MEDIA_DIR).is_dir(), "봉인하지 않은 청크의 무손실을 지웠습니다"
    assert ("generate", 10) not in stages.calls
    # Said as a chunk that failed, like a stage that did: which chunk, at the seal, and which link.
    assert any("청크 0000 실패" in line and "seal" in line and f"{DATASET_DIR}/videos" in line for line in lines), lines


def test_a_tree_that_changed_between_chunks_stops_the_runner_before_the_next_chunk(tmp_path):
    batch, tree, lines = _batch(tmp_path), _clean(), []

    def say(line):
        lines.append(line)
        if "0000" in line and "봉인" in line:
            tree["generator_dirty"] = True

    stages = FakeStages()
    assert _run(batch, stages, say=say, tree=tree) == 1
    assert read_seal(_chunk(batch, 0)) is not None
    assert ("generate", 10) not in stages.calls and not _chunk(batch, 10).exists(), "바뀐 트리에서 다음 청크를 시작했습니다"
    assert any("0010" in line and "커밋되지 않은 변경" in line for line in lines), lines


def test_a_tree_git_cannot_read_is_not_taken_for_a_clean_one(tmp_path):
    batch = _batch(tmp_path, episodes=10)
    lines, say = _said()
    assert _run(batch, FakeStages(), say=say, tree={"generator_commit": None, "generator_dirty": None}) == 1
    assert any("git" in line for line in lines), lines


def test_the_chunks_files_are_flushed_to_disk_before_its_seal_is_written(tmp_path, monkeypatch):
    # After a hard reset the record that says a chunk is done must not outlive the files it vouches for.
    batch = _batch(tmp_path, episodes=10)
    seen = []
    monkeypatch.setattr(rb.os, "sync", lambda: seen.append((_chunk(batch, 0) / SEAL_FILE).exists()))
    assert _run(batch, FakeStages(), say=lambda s: None) == 0
    assert seen == [False]


# ── what the seal records of validation and measurement ──

def test_the_seal_records_what_validation_and_measurement_said_and_the_line_shows_it(tmp_path):
    batch = _batch(tmp_path, episodes=10)
    lines, say = _said()
    assert _run(batch, FakeStages(), say=say) == 0
    record = read_seal(_chunk(batch, 0))
    assert record["validation"] == {"error": 0, "warn": 1, "info": 4, "skip": 0}
    assert record["measurement"] == {
        "entry_triggered": 9,
        "entry_speed_ratio": {"n": 9, "min": 0.2892, "median": 1.0024, "max": 1.0277},
        "entry_speed_ratio_held": {"n": 6, "min": 1.0018, "median": 1.0026, "max": 1.0049},
        "facing_off_travel_deg_max": 0.0,
        "outcome": {"fail": 3, "near_miss": 4, "nominal": 3}}
    assert "Infinity" not in (_chunk(batch, 0) / SEAL_FILE).read_text(encoding="utf-8"), "범위(range)를 옮기지 않습니다"
    (line,) = [l for l in lines if "0000" in l and "봉인" in l]
    for shown in ("경고 1", "0.2892", "1.0277", "1.0018", "1.0049", "fail 3", "near_miss 4"):
        assert shown in line, (shown, line)


class _WritesAfter(FakeStages):
    """A stage that exits 0 and leaves ``text`` in its report instead of the real one."""

    def __init__(self, stage, report, text):
        super().__init__()
        self.stage, self.report, self.text = stage, report, text

    def _leave(self, chunk):
        (chunk / LOGS_DIR / self.report).write_text(self.text, encoding="utf-8")

    def validate(self, batch, plan, chunk):
        code = super().validate(batch, plan, chunk)
        if self.stage == "validate":
            self._leave(chunk)
        return code

    def measure(self, batch, plan, chunk):
        code = super().measure(batch, plan, chunk)
        if self.stage == "measure":
            self._leave(chunk)
        return code


@pytest.mark.parametrize("make", [
    lambda: FakeStages(reports=False),
    lambda: _WritesAfter("validate", "validate.json", "{"),
    lambda: _WritesAfter("measure", "measure.json", '{"summary": []}'),
], ids=["no file", "unreadable validate.json", "measure.json of another shape"])
def test_a_report_that_cannot_be_read_is_recorded_as_missing_and_does_not_fail_the_chunk(tmp_path, make):
    batch = _batch(tmp_path, episodes=10)
    lines, say = _said()
    assert _run(batch, make(), say=say) == 0
    record = read_seal(_chunk(batch, 0))
    assert [k for k in ("validation", "measurement") if record[k] is None]
    (line,) = [l for l in lines if "0000" in l and "봉인" in l]
    assert "요약 없음" in line, line


def test_a_report_with_a_non_finite_number_does_not_put_it_in_the_seal(tmp_path):
    class Overflowing(FakeStages):
        def measure(self, batch, plan, chunk):
            report = json.loads(json.dumps(MEASURE_REPORT))
            report["summary"]["entry_speed_ratio"]["max"] = math.inf
            report["summary"]["facing_off_travel_deg_max"] = math.nan
            (chunk / LOGS_DIR / "measure.json").write_text(json.dumps(report), encoding="utf-8")
            return 0

    batch = _batch(tmp_path, episodes=10)
    assert _run(batch, Overflowing(), say=lambda s: None) == 0
    text = (_chunk(batch, 0) / SEAL_FILE).read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    assert read_seal(_chunk(batch, 0))["measurement"]["facing_off_travel_deg_max"] is None


# ── a stalled attempt leaves its log; a chunk that cannot be wiped stops the runner ──

def test_the_log_of_a_stalled_attempt_is_kept_before_the_chunk_is_wiped(tmp_path):
    batch = _batch(tmp_path, episodes=10)
    stages = FakeStages(codes={("generate", 0): [rb.STALLED, rb.STALLED]})
    lines, say = _said()
    assert _run(batch, stages, say=say) == 0
    kept = batch.root / rb.STALLED_LOGS
    assert sorted(p.name for p in kept.iterdir()) == ["0000.attempt1.generate.log", "0000.attempt2.generate.log"]
    assert (kept / "0000.attempt1.generate.log").read_text(encoding="utf-8") == "stalled, attempt 1"
    assert any(str(kept) in line for line in lines), "진행 줄이 로그가 어디 남았는지 말해야 합니다"


@pytest.mark.parametrize("seq", [None, 6], ids=["a chunk that is one run", "a row"])
def test_a_stalled_log_kept_by_an_earlier_run_is_not_overwritten(tmp_path, seq):
    # A batch run again counts its attempts from one: the log an earlier run kept under the same name is the only
    # evidence of why that attempt stalled, and stays.
    batch = _batch(tmp_path, episodes=10) if seq is None else _table_batch(tmp_path, parts=(("S1e", (2, 6)),))
    plan = batch.chunks()[0]
    chunk = batch.chunk_dir(plan)
    log = chunk / LOGS_DIR / "generate.log" if seq is None else rb.row_log(chunk, seq)
    log.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    for run in (1, 2, 3):
        log.write_text(f"stalled in run {run}", encoding="utf-8")
        rb.keep_stalled_log(batch, plan, 1, chunk, kept.append, seq=seq)
    stem = plan.key if seq is None else f"{plan.key}.{seq:04d}"
    names = [f"{stem}.attempt1.generate.log", f"{stem}.attempt1.2.generate.log", f"{stem}.attempt1.3.generate.log"]
    there = batch.root / rb.STALLED_LOGS
    assert sorted(p.name for p in there.iterdir()) == sorted(names)
    assert [(there / n).read_text(encoding="utf-8") for n in names] == [f"stalled in run {r}" for r in (1, 2, 3)]
    assert [str(there / n) in line for n, line in zip(names, kept)] == [True] * 3, kept


def test_the_log_of_the_last_stalled_attempt_is_kept_as_well(tmp_path):
    batch = _batch(tmp_path, episodes=10)
    assert _run(batch, FakeStages(codes={("generate", 0): [rb.STALLED] * 3}), say=lambda s: None) == 1
    assert sorted(p.name for p in (batch.root / rb.STALLED_LOGS).iterdir()) == [
        f"0000.attempt{n}.generate.log" for n in (1, 2, 3)]


def test_a_chunk_that_cannot_be_wiped_stops_the_runner_and_names_the_path(tmp_path, monkeypatch):
    batch = _batch(tmp_path, episodes=10)
    chunk = _chunk(batch, 0)
    (chunk / EPISODES_DIR).mkdir(parents=True)
    (chunk / EPISODES_DIR / "left-by-the-run-that-died.txt").write_text("x", encoding="utf-8")
    real = shutil.rmtree

    def refuse(path, *args, **kwargs):
        if Path(path) != chunk:
            return real(path, *args, **kwargs)
        if kwargs.get("ignore_errors"):
            return None                                   # as shutil does: the failure is swallowed
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(rb.shutil, "rmtree", refuse)
    stages, (lines, say) = FakeStages(), _said()
    assert _run(batch, stages, say=say) == 1
    assert stages.calls == [], "지우지 못한 청크 위에서 생성을 시작했습니다"
    assert any(str(chunk) in line for line in lines), lines


# ── progress lines reach the file at once ──

def test_a_progress_line_is_flushed_when_it_is_written(monkeypatch):
    # Redirected to a file, print() holds the line in an 8 KB buffer: an 11-hour log would stay empty
    # until exit, and be lost with a hard reset.
    class Out(io.StringIO):
        flushes = 0

        def flush(self):
            Out.flushes += 1
            super().flush()

    out = Out()
    monkeypatch.setattr(sys, "stdout", out)
    rb.progress("청크 0000")
    assert out.getvalue() == "청크 0000\n" and Out.flushes >= 1


@pytest.mark.parametrize("fn", [rb.run, rb.run_chunk, rb.make_rows, rb.run_stage, rb.Lock.__init__],
                         ids=lambda f: f.__qualname__)
def test_every_say_defaults_to_the_flushing_progress(fn):
    assert inspect.signature(fn).parameters["say"].default is rb.progress


# ── run_stage: one subprocess in a session of its own, held by its PID ──

def _script(tmp_path, body):
    path = tmp_path / "stage.py"
    path.write_text(body, encoding="utf-8")
    return [sys.executable, str(path)]


FAST = dict(grace_s=0.5, stall_s=1.0, poll_s=0.1, kill_grace_s=1.0)


def test_a_stage_returns_its_exit_code_and_logs_its_output(tmp_path):
    argv = _script(tmp_path, "import sys\nprint('said this')\nsys.exit(7)\n")
    assert rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, **FAST) == 7
    assert "said this" in (tmp_path / "stage.log").read_text(encoding="utf-8")


def test_a_stage_that_keeps_writing_is_left_alone(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    argv = _script(tmp_path, "import sys, time\n"
                             "for i in range(16):\n"
                             f"    open({str(out / 'video.mkv')!r}, 'ab').write(b'x')\n"
                             "    time.sleep(0.15)\n")
    assert rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, watch_dir=out, **FAST) == 0


def test_a_stage_that_stops_writing_is_ended_with_everything_it_started(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    pid_file = tmp_path / "grandchild.pid"
    argv = _script(tmp_path, "import subprocess, sys, time\n"
                             f"open({str(out / 'video.mkv')!r}, 'wb').write(b'x')\n"
                             "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                             f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
                             "time.sleep(60)\n")
    lines, say = _said()
    started = time.monotonic()
    assert rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, watch_dir=out, say=say, **FAST) == rb.STALLED
    assert time.monotonic() - started < 20
    grandchild = int(pid_file.read_text(encoding="utf-8"))
    for _ in range(50):
        if not rb.alive(grandchild):
            break
        time.sleep(0.1)
    assert not rb.alive(grandchild), "단계가 띄운 프로세스가 남았습니다"
    assert any("쓰인 파일이 없어" in line for line in lines), lines


def test_a_stage_without_a_watch_directory_is_never_judged_stalled(tmp_path):
    argv = _script(tmp_path, "import time\ntime.sleep(2.5)\n")
    assert rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, **FAST) == 0


def test_the_stage_reports_its_process_group_while_it_runs(tmp_path):
    seen = []
    argv = _script(tmp_path, "pass\n")
    assert rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, on_start=seen.append, **FAST) == 0
    assert len(seen) == 2 and isinstance(seen[0], int) and seen[1] is None


# ── ending a stage: an interrupt, and a stage that does not answer SIGTERM ──

def _gone(pid):
    """Whether ``pid`` no longer runs. A zombie counts as gone: it is dead and waits only for its reaper."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
    except OSError:
        return True
    return state in ("Z", "X")


def _stubborn(pid_file, marker, before_pid=""):
    """A stage that answers SIGTERM by noting it in ``marker`` and carrying on -- the generator hang of
    2026-10-01 did not die of it. ``pid_file`` appears last, so it says the handler is in place."""
    return ("import os, signal, time\n"
            "def term(signum, frame):\n"
            f"    open({str(marker)!r}, 'a').write('term ')\n"
            "signal.signal(signal.SIGTERM, term)\n"
            + before_pid +
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
            "time.sleep(60)\n")


def _kill_group(pid_file):
    """Leave nothing behind when a test fails: by the PID the stage wrote, never by name."""
    if pid_file.exists() and pid_file.read_text(encoding="utf-8").strip():
        try:
            os.killpg(int(pid_file.read_text(encoding="utf-8")), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _within(seconds, fn):
    """``fn()`` in a thread; (finished, results). A stage that cannot be ended must fail its test,
    not hang the run."""
    box = []
    thread = threading.Thread(target=lambda: box.append(fn()), daemon=True)
    thread.start()
    thread.join(seconds)
    return not thread.is_alive(), box


def test_an_interrupt_while_a_stage_runs_ends_the_stage(tmp_path):
    # Ctrl-C reaches the runner and not the stage (it has a session of its own): left to itself the
    # stage would keep writing into the chunk the next run removes.
    pid_file = tmp_path / "stage.pid"
    argv = _script(tmp_path, f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n")

    def interrupt(signum, frame):
        if pid_file.exists() and pid_file.read_text(encoding="utf-8").strip():       # the stage is up
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise KeyboardInterrupt

    before = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.2, 0.1)
    try:
        with pytest.raises(KeyboardInterrupt):
            rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, **FAST)
        assert _gone(int(pid_file.read_text(encoding="utf-8"))), "중단된 실행기가 단계를 남겼습니다"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, before)
        _kill_group(pid_file)


def test_a_stalled_stage_that_does_not_answer_sigterm_is_killed(tmp_path):
    out, pid_file, marker = tmp_path / "out", tmp_path / "stage.pid", tmp_path / "terms"
    out.mkdir()
    argv = _script(tmp_path, _stubborn(pid_file, marker, f"open({str(out / 'video.mkv')!r}, 'wb').write(b'x')\n"))
    try:
        finished, results = _within(15, lambda: rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path,
                                                             watch_dir=out, say=lambda s: None, **FAST))
        assert finished, "SIGTERM 에 답하지 않는 단계가 끝나지 않았습니다"
        assert results == [rb.STALLED]
        assert "term" in marker.read_text(encoding="utf-8"), "SIGTERM 이 가지 않았다면 이 시험은 SIGKILL 을 보지 못합니다"
        assert _gone(int(pid_file.read_text(encoding="utf-8")))
    finally:
        _kill_group(pid_file)


def test_a_group_left_by_a_dead_runner_that_does_not_answer_sigterm_is_killed(tmp_path):
    # The path a later run takes for a stage a dead runner left. The group's leader is a child of this
    # process, so a thread reaps it the moment it dies: a zombie still answers killpg(pgid, 0).
    pid_file, marker = tmp_path / "stage.pid", tmp_path / "terms"
    p = subprocess.Popen(_script(tmp_path, _stubborn(pid_file, marker)), start_new_session=True)
    reaper = threading.Thread(target=p.wait, daemon=True)
    reaper.start()
    try:
        for _ in range(100):
            if pid_file.exists() and pid_file.read_text(encoding="utf-8").strip():
                break
            time.sleep(0.1)
        assert pid_file.exists(), "단계가 시작하지 않았습니다"
        rb.end_group_by_id(p.pid, 1.0)
        reaper.join(5)
        assert not reaper.is_alive(), "SIGTERM 에 답하지 않는 그룹이 남았습니다"
        assert "term" in marker.read_text(encoding="utf-8")
        assert p.returncode == -signal.SIGKILL, p.returncode
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)         # by the PID this test started; nothing if it is gone
        except ProcessLookupError:
            pass
        reaper.join(5)


def _wrapper_stage(tmp_path, out, pid_file, marker):
    """The generator's shape: a bash wrapper that runs python without ``exec`` (~/isaacsim/python.sh ends in
    `$python_exe ... || error_exit`). Bash dies of SIGTERM at once; its child answers it with a clean-up
    that takes a moment and ends in ``marker``."""
    child = tmp_path / "child.py"
    child.write_text("import os, signal, sys, time\n"
                     "def term(signum, frame):\n"
                     "    time.sleep(0.4)                 # what the simulator does before it is gone\n"
                     f"    open({str(marker)!r}, 'w').write('cleaned up')\n"
                     "    sys.exit(0)\n"
                     "signal.signal(signal.SIGTERM, term)\n"
                     f"open({str(out / 'video.mkv')!r}, 'wb').write(b'x')\n"
                     f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
                     "time.sleep(60)\n", encoding="utf-8")
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(f'#!/bin/bash\n"{sys.executable}" "{child}"\necho "back in the wrapper"\n', encoding="utf-8")
    return ["/bin/bash", str(wrapper)]


def test_the_grace_of_a_stalled_stage_goes_to_its_group_not_to_the_wrapper_that_leads_it(tmp_path):
    # Measured: behind such a wrapper the old end_group returned after 0.6 s with
    # kill_grace_s=5 -- the leader had died of SIGTERM at once, the group got SIGKILL, the child's clean-up
    # never ran and its PID still existed on return.
    out, pid_file, marker = tmp_path / "out", tmp_path / "stage.pid", tmp_path / "cleanup"
    out.mkdir()
    argv = _wrapper_stage(tmp_path, out, pid_file, marker)
    groups = []
    try:
        finished, results = _within(20, lambda: rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, watch_dir=out,
                                                             on_start=groups.append, say=lambda s: None, **FAST))
        assert finished and results == [rb.STALLED]
        assert marker.exists(), "SIGTERM 의 유예가 래퍼의 자식에게 닿지 않았습니다 — 정리 없이 SIGKILL 됐습니다"
        assert _gone(int(pid_file.read_text(encoding="utf-8"))), "단계가 끝났다고 했는데 자식이 남았습니다"
        assert not rb.group_running(groups[0]), "그룹에 아직 도는 프로세스가 있습니다"
    finally:
        _kill_group(pid_file)


def _leaving_wrapper(tmp_path, pid_file, marker, exit_code=3):
    """A wrapper that leaves its child behind: it starts python in the background, waits (at most 10 s) until
    that child has written its PID, and then ends -- with ``exit_code``, or killed by a signal (SIGKILL, as the
    out-of-memory killer would) when ``exit_code`` is None. The child answers SIGTERM with a clean-up of 0.4 s that
    ends in ``marker`` -- so a SIGKILL given at once would leave no mark."""
    child = tmp_path / "child.py"
    child.write_text("import os, signal, sys, time\n"
                     "def term(signum, frame):\n"
                     "    time.sleep(0.4)\n"
                     f"    open({str(marker)!r}, 'w').write('cleaned up')\n"
                     "    sys.exit(0)\n"
                     "signal.signal(signal.SIGTERM, term)\n"
                     f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
                     "time.sleep(60)\n", encoding="utf-8")
    leave = "kill -KILL $$" if exit_code is None else f"exit {exit_code}"
    return ["/bin/bash", "-c", f'"{sys.executable}" "{child}" & for i in $(seq 1 200); do [ -s "{pid_file}" ] && break; sleep 0.05; done; {leave}']


def _kill(pid_file):
    if pid_file.exists() and pid_file.read_text(encoding="utf-8").strip():
        try:
            os.kill(int(pid_file.read_text(encoding="utf-8")), signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_a_stage_whose_leader_was_killed_while_its_child_runs_on_is_ended_and_keeps_the_leaders_code(tmp_path):
    # Only the bash wrapper dies -- by hand, by the out-of-memory killer. wait() returns,
    # the simulator it started lives on in the group, and the next step would wipe the chunk under it. A leader
    # that ended by a signal did not wait for its child: what is left in the group is a straggler, and is ended.
    pid_file, marker = tmp_path / "child.pid", tmp_path / "cleanup"
    argv = _leaving_wrapper(tmp_path, pid_file, marker, exit_code=None)
    calls, lines = [], []
    try:
        code = rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, on_start=calls.append,
                            say=lines.append, **FAST)
        assert code == -signal.SIGKILL, "the leader must have died by a signal -- that is the condition of this test"
        assert _gone(int(pid_file.read_text(encoding="utf-8"))), "리더가 끝난 단계의 자식이 남았습니다"
        assert marker.exists(), "SIGTERM 의 유예가 남은 자식에게 주어지지 않았습니다 — 정리 없이 SIGKILL 됐습니다"
        assert len(calls) == 2 and isinstance(calls[0], int) and calls[1] is None
        assert not rb.group_running(calls[0])
        said = [line for line in lines if "시그널" in line]
        assert len(said) == 1 and str(signal.SIGKILL.value) in said[0] and "1개" in said[0], lines
    finally:
        _kill(pid_file)


@pytest.mark.parametrize("exit_code", [0, 3], ids=["exit 0", "exit 3"])
def test_a_leader_that_exits_normally_leaves_its_group_alone(tmp_path, exit_code):
    # The Omniverse hub Kit starts stays in the generator's group on purpose, and is shared with other sessions
    # (the validator's C9 hook leaves it running as well). A wrapper that exits with a code has waited for its child;
    # what is left in its group was left there by the simulator. Not ended -- and no grace waited for.
    pid_file, marker = tmp_path / "child.pid", tmp_path / "cleanup"
    argv = _leaving_wrapper(tmp_path, pid_file, marker, exit_code=exit_code)
    calls, lines = [], []
    started = time.monotonic()
    try:
        code = rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, on_start=calls.append, say=lines.append,
                            grace_s=0.5, stall_s=1.0, poll_s=0.1, kill_grace_s=5.0)
        elapsed = time.monotonic() - started
        assert code == exit_code
        member = int(pid_file.read_text(encoding="utf-8"))
        assert not _gone(member), "정상으로 끝난 단계의 그룹에 남은 프로세스(허브의 자리)를 끝냈습니다"
        assert not marker.exists()
        assert len(calls) == 2 and isinstance(calls[0], int) and calls[1] is None, calls
        assert elapsed < 4.0, f"유예를 기다렸습니다 ({elapsed:.1f} s)"
        assert not any("시그널" in line for line in lines)
    finally:
        _kill(pid_file)


def test_a_process_that_outlives_a_normally_ended_stage_is_not_ended_by_a_later_run(tmp_path):
    # The record is told None when a stage ends normally, so a later run has no group to clean up: the hub in an
    # old stage's group is never reached from the lock either.
    batch = _batch(tmp_path)
    pid_file, marker = tmp_path / "child.pid", tmp_path / "cleanup"
    argv = _leaving_wrapper(tmp_path, pid_file, marker, exit_code=0)
    try:
        with rb.Lock(batch.root) as lock:
            assert rb.stage_runner(lock)(argv, tmp_path / "stage.log", cwd=tmp_path, **FAST) == 0
        member = int(pid_file.read_text(encoding="utf-8"))
        assert not _gone(member)
        with rb.Lock(batch.root, say=lambda s: None, kill_grace_s=1.0):
            pass
        assert not _gone(member), "다음 실행이 지난 단계의 그룹에 남은 프로세스를 끝냈습니다"
    finally:
        _kill(pid_file)


def test_a_stage_that_exits_and_leaves_its_group_empty_costs_no_signal(tmp_path, monkeypatch):
    # The normal run: nothing of the group is left, so nothing is ended.
    ended = []
    monkeypatch.setattr(rb, "end_group", lambda *a, **k: ended.append(a))
    assert rb.run_stage(_script(tmp_path, "pass\n"), tmp_path / "stage.log", cwd=tmp_path, **FAST) == 0
    assert ended == []


def test_a_failing_record_write_when_the_stage_starts_does_not_leave_the_stage_running(tmp_path):
    # on_start once sat before the try, so a failure of the lock record's write (ENOSPC)
    # skipped the clean-up of the stage that had just been started.
    seen = []

    def no_space(pgid):
        seen.append(pgid)
        raise OSError(28, "No space left on device")

    argv = _script(tmp_path, "import time\ntime.sleep(60)\n")
    try:
        with pytest.raises(OSError, match="No space"):
            rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, on_start=no_space, **FAST)
        assert seen and not rb.group_running(seen[0]), "기록을 쓰지 못한 실행기가 단계를 남겼습니다"
    finally:
        if seen:
            try:
                os.killpg(seen[0], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_an_interrupt_during_the_grace_of_a_stalled_stage_leaves_no_stage(tmp_path):
    # The stage is stalled, SIGTERM went out and the runner waits its grace for
    # a stage that does not answer it; Ctrl-C arrives then. The old code had raised ``ending`` by then, and the
    # interrupted run left the stage running.
    out, pid_file, marker = tmp_path / "out", tmp_path / "stage.pid", tmp_path / "terms"
    out.mkdir()
    argv = _script(tmp_path, _stubborn(pid_file, marker, f"open({str(out / 'video.mkv')!r}, 'wb').write(b'x')\n"))

    def interrupt(signum, frame):
        if marker.exists():                      # the stage has had its SIGTERM: the runner is in the grace
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise KeyboardInterrupt

    before = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.2, 0.1)
    started, calls = time.monotonic(), []
    try:
        with pytest.raises(KeyboardInterrupt):
            rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, watch_dir=out, say=lambda s: None, on_start=calls.append,
                         grace_s=0.5, stall_s=1.0, poll_s=0.1, kill_grace_s=30.0)
        assert time.monotonic() - started < 20, "중단이 SIGTERM 의 유예가 다 지나도록 기다렸습니다"
        assert _gone(int(pid_file.read_text(encoding="utf-8"))), "멈춘 단계를 끝내는 도중 중단된 실행기가 단계를 남겼습니다"
        assert len(calls) == 2 and calls[1] is None, "그룹이 없어졌는데 기록이 그룹을 가리킵니다"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, before)
        _kill_group(pid_file)


def test_a_group_that_is_still_there_after_the_leader_exited_is_not_told_gone(tmp_path):
    # Leaving run_stage with the group running is possible by GroupAlive only -- and then the record keeps
    # naming the group, which is what the next run reads.
    calls = []
    killed = _script(tmp_path, "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n")      # a leader that ends by a signal
    with pytest.raises(rb.GroupAlive):
        rb.run_stage(killed, tmp_path / "stage.log", cwd=tmp_path, on_start=calls.append,
                     running=lambda pgid: True, bound_s=0.3, say=lambda s: None, **FAST)
    assert len(calls) == 1 and isinstance(calls[0], int), calls


def test_a_group_that_survives_sigkill_is_not_waited_for_forever_and_not_gone_on_from(tmp_path):
    # Simulated: no unkillable process is needed to see what end_group does when the group is still there.
    p = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        started = time.monotonic()
        with pytest.raises(rb.GroupAlive, match=str(p.pid)) as e:
            rb.end_group(p, 0.1, bound_s=0.3, running=lambda pgid: True)
        assert time.monotonic() - started < 10
        assert e.value.pgid == p.pid and isinstance(e.value, BatchError)
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()


def test_a_left_over_group_that_survives_sigkill_raises_instead_of_returning_silently(tmp_path):
    # SIGTERM must not end it (else the SIGKILL finds no group and the function rightly returns).
    pid_file, marker = tmp_path / "stage.pid", tmp_path / "terms"
    p = subprocess.Popen(_script(tmp_path, _stubborn(pid_file, marker)), start_new_session=True)
    reaper = threading.Thread(target=p.wait, daemon=True)
    reaper.start()
    try:
        for _ in range(100):
            if pid_file.exists() and pid_file.read_text(encoding="utf-8").strip():
                break
            time.sleep(0.1)
        assert pid_file.exists(), "단계가 시작하지 않았습니다"
        with pytest.raises(BatchError, match=str(p.pid)):
            rb.end_group_by_id(p.pid, 0.1, bound_s=0.3, running=lambda pgid: True)
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        reaper.join(5)


def test_a_group_with_a_zombie_leader_alone_counts_as_gone(tmp_path):
    # A leader that has died and waits to be reaped runs nothing and holds no file.
    p = subprocess.Popen(["true"], start_new_session=True)
    for _ in range(100):
        state = Path(f"/proc/{p.pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
        if state == "Z":
            break
        time.sleep(0.02)
    assert state == "Z", "시험이 좀비를 만들지 못했습니다"
    assert not rb.group_running(p.pid)
    p.wait()


def test_a_group_that_still_runs_is_seen_to_run(tmp_path):
    p = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        assert rb.group_running(p.pid)
    finally:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    assert not rb.group_running(p.pid)


def test_a_stage_whose_group_will_not_die_stops_the_runner_and_the_chunk_stays_as_it_was(tmp_path):
    class Stuck(FakeStages):
        def generate(self, batch, plan, chunk):
            (chunk / EPISODES_DIR / "half-written.npz").write_bytes(b"x")
            raise rb.GroupAlive(4242, 30.0)

    batch = _batch(tmp_path, episodes=10)
    with pytest.raises(rb.GroupAlive, match="4242"):
        _run(batch, Stuck(), say=lambda s: None)
    assert (_chunk(batch, 0) / EPISODES_DIR / "half-written.npz").exists(), "살아 있는 프로세스 아래서 청크를 지웠습니다"


def test_main_says_which_group_is_left_and_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rb.provenance, "read", lambda root: _clean())

    def stuck(batch, stages, *, read_tree, say=rb.progress):
        raise rb.GroupAlive(4242, 30.0)

    monkeypatch.setattr(rb, "run", stuck)
    assert rb.main(["--batch", str(tmp_path / "batch"), "--episodes", "10"]) == 2
    assert "4242" in capsys.readouterr().err


# ── the lock ──
# The runner holds flock on <batch>/LOCK for its whole run -- the kernel lets go when the process dies, so
# there is no stale lock and no PID to compare -- and records its PID and its stage's process group in
# <batch>/RUNNER (replaced atomically).

REPO = Path(__file__).resolve().parents[2]


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _record(batch, **doc):
    (batch.root / RUNNER_FILE).write_text(json.dumps(doc), encoding="utf-8")


def _holder(batch):
    """A process that takes the batch's lock the way a runner does and keeps it. (process, its PID)."""
    script = ("import sys, time\n"
              f"sys.path.insert(0, {str(REPO)!r})\n"
              "from generation import run_batch as rb\n"
              f"rb.Lock({str(batch.root)!r}, say=lambda s: None).__enter__()\n"
              "print('held', flush=True)\n"
              "time.sleep(60)\n")
    p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, cwd=REPO)
    assert p.stdout.readline().strip() == "held", "the holder did not get the lock"
    return p


def test_a_second_runner_on_the_same_batch_is_refused(tmp_path):
    batch = _batch(tmp_path)
    holder = _holder(batch)
    try:
        before = (batch.root / RUNNER_FILE).read_text(encoding="utf-8")
        with pytest.raises(BatchError, match=f"PID {holder.pid}"):
            with rb.Lock(batch.root):
                pass
        assert (batch.root / RUNNER_FILE).read_text(encoding="utf-8") == before, "남의 기록을 바꿨습니다"
        holder.kill()                       # the kernel lets go of the lock of a process that dies
        holder.wait()
        with rb.Lock(batch.root) as lock:
            assert json.loads((batch.root / RUNNER_FILE).read_text(encoding="utf-8"))["pid"] == os.getpid()
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait()


def test_two_runners_racing_for_the_lock_cannot_both_have_it(tmp_path, monkeypatch):
    # A race that is really run: both runners are held at the very call that takes the lock and let go of it
    # together, so the kernel arbitrates between two requests made at the same moment. (Taking the lock by
    # looking for a file and then writing one lets both through here; without a lock call at all the barrier is
    # never reached and both are let through as well.)
    batch = _batch(tmp_path)
    barrier = threading.Barrier(2, timeout=5)
    real_flock = rb.fcntl.flock

    def flock(fd, how):
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return real_flock(fd, how)

    monkeypatch.setattr(rb.fcntl, "flock", flock)
    results, held = [], []

    def runner():
        lock = rb.Lock(batch.root, say=lambda s: None)
        try:
            lock.__enter__()
        except BatchError:
            results.append("refused")
        else:
            results.append("got")
            held.append(lock)

    threads = [threading.Thread(target=runner) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    try:
        assert sorted(results) == ["got", "refused"], results
        assert barrier.n_waiting == 0 and not barrier.broken, "the barrier was not on the path of both runners"
    finally:
        for lock in held:
            lock.__exit__(None, None, None)


def test_runners_released_together_never_both_get_the_lock(tmp_path):
    batch = _batch(tmp_path)
    for _ in range(25):
        start, results, held = threading.Barrier(2, timeout=5), [], []

        def runner():
            lock = rb.Lock(batch.root, say=lambda s: None)
            start.wait()
            try:
                lock.__enter__()
            except BatchError:
                results.append("refused")
            else:
                results.append("got")
                held.append(lock)

        threads = [threading.Thread(target=runner) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(15)
        try:
            assert sorted(results) == ["got", "refused"], results
        finally:
            for lock in held:
                lock.__exit__(None, None, None)


def test_a_live_process_that_is_not_a_runner_does_not_hold_the_batch(tmp_path):
    # 재부팅 뒤에 남은 기록의 PID 는 다른 프로세스가 다시 쓴다. 잠금을 쥐고 있지 않으면 잠금이 아니다.
    batch = _batch(tmp_path)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _record(batch, pid=other.pid, child=None)
        with rb.Lock(batch.root):
            pass
    finally:
        other.kill()
        other.wait()


def test_a_lock_left_by_a_runner_that_died_is_taken_over(tmp_path):
    batch = _batch(tmp_path)
    (batch.root / LOCK_FILE).write_text("", encoding="utf-8")
    _record(batch, pid=_dead_pid(), child=None)
    with rb.Lock(batch.root):
        assert json.loads((batch.root / RUNNER_FILE).read_text(encoding="utf-8")) == {"pid": os.getpid(), "child": None}
    assert not (batch.root / RUNNER_FILE).exists(), "끝난 실행기의 기록이 남았습니다"
    with rb.Lock(batch.root):                  # and the lock is free again (the file itself stays: unlinking it races)
        pass


def test_the_record_is_replaced_not_rewritten_in_place(tmp_path, monkeypatch):
    # A reader must never see the record empty: it is written under another name and renamed over.
    batch = _batch(tmp_path)
    replaced = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: (replaced.append((Path(src).name, Path(dst).name)), real(src, dst))[1])
    with rb.Lock(batch.root) as lock:
        lock.child(4242)
        assert json.loads((batch.root / RUNNER_FILE).read_text(encoding="utf-8")) == {"pid": os.getpid(), "child": 4242}
    assert replaced and all(dst == RUNNER_FILE and src != RUNNER_FILE for src, dst in replaced), replaced


def test_a_stage_does_not_inherit_the_lock(tmp_path):
    # A stage outlives a runner that is killed. Were it to keep the lock's file open, the next run would find
    # the batch held by a stage.
    batch = _batch(tmp_path)
    seen = tmp_path / "fds.txt"
    argv = _script(tmp_path, "import os\n"
                             f"open({str(seen)!r}, 'w').write('\\n'.join(os.readlink('/proc/self/fd/' + n) for n in os.listdir('/proc/self/fd') if os.path.exists('/proc/self/fd/' + n)))\n")
    with rb.Lock(batch.root) as lock:
        assert rb.stage_runner(lock)(argv, tmp_path / "stage.log", cwd=tmp_path, **FAST) == 0
    held = seen.read_text(encoding="utf-8").split("\n")
    assert str(batch.root / LOCK_FILE) not in held, held


def test_a_stage_left_running_by_a_runner_that_died_is_ended_first(tmp_path):
    # 단계는 자기 세션에서 돈다: 실행기가 죽어도 살아남아, 다음 실행이 지우려는 청크에 계속 쓴다.
    batch = _batch(tmp_path)
    orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(batch.root)],
                              start_new_session=True)
    try:
        _record(batch, pid=_dead_pid(), child=orphan.pid)
        lines, say = _said()
        with rb.Lock(batch.root, say=say, kill_grace_s=1.0):
            assert orphan.wait(timeout=10) is not None
        assert any(str(orphan.pid) in line for line in lines), lines
    finally:
        if orphan.poll() is None:
            orphan.kill()


def test_a_recorded_group_whose_leader_is_dead_and_whose_child_is_alive_is_ended(tmp_path):
    # The generator's shape (a bash wrapper above the simulator): the wrapper can be gone -- killed by hand, by the
    # out-of-memory killer -- while the simulator it started runs on in the group the wrapper led.
    batch = _batch(tmp_path)
    pid_file = tmp_path / "child.pid"
    child = tmp_path / "child.py"
    child.write_text(f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n", encoding="utf-8")
    leader = subprocess.Popen(["/bin/bash", "-c", f'"{sys.executable}" "{child}" {batch.root} & exit 0'],
                              start_new_session=True)
    try:
        leader.wait(timeout=10)
        for _ in range(100):
            if pid_file.exists() and pid_file.read_text(encoding="utf-8").strip():
                break
            time.sleep(0.1)
        grandchild = int(pid_file.read_text(encoding="utf-8"))
        assert not rb.alive(leader.pid) or _gone(leader.pid), "the leader is not gone"
        assert rb.group_running(leader.pid) and not _gone(grandchild), "the group has to be alive without its leader"
        _record(batch, pid=_dead_pid(), child=leader.pid)
        lines, say = _said()
        with rb.Lock(batch.root, say=say, kill_grace_s=1.0):
            assert _gone(grandchild), "리더가 죽은 그룹의 자식이 남았습니다"
        assert any(str(leader.pid) in line for line in lines), lines
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_a_process_that_only_shares_the_recorded_pid_is_left_alone(tmp_path):
    # PID 는 다시 쓰인다. 명령줄에 이 배치가 없으면 우리 단계가 아니다.
    batch = _batch(tmp_path)
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        _record(batch, pid=_dead_pid(), child=stranger.pid)
        with rb.Lock(batch.root, kill_grace_s=1.0):
            pass
        assert stranger.poll() is None
    finally:
        stranger.kill()


def test_a_record_that_names_a_group_the_runner_could_not_end_stays_for_the_next_run(tmp_path):
    batch = _batch(tmp_path)
    with rb.Lock(batch.root) as lock:
        lock.child(4242)                      # the stage's group; run_stage raised GroupAlive before telling None
    assert json.loads((batch.root / RUNNER_FILE).read_text(encoding="utf-8")) == {"pid": os.getpid(), "child": 4242}


@pytest.mark.parametrize("child", [0, 1, -5, True, "7", None], ids=["0", "1", "negative", "bool", "string", "none"])
def test_a_record_that_names_no_real_group_ends_nothing(tmp_path, monkeypatch, child):
    # killpg(0) is the caller's own group and killpg(1) is init's: a damaged record must not reach them.
    batch = _batch(tmp_path)
    _record(batch, pid=_dead_pid(), child=child)
    ended = []
    monkeypatch.setattr(rb, "group_running", lambda pgid: True)
    monkeypatch.setattr(rb, "group_members", lambda pgid: [os.getpid()])
    monkeypatch.setattr(rb, "cmdline", lambda pid: str(batch.root))
    monkeypatch.setattr(rb, "end_group_by_id", lambda pgid, grace: ended.append(pgid))
    with rb.Lock(batch.root):
        pass
    assert ended == []
    _record(batch, pid=_dead_pid(), child=4242)
    with rb.Lock(batch.root, say=lambda s: None):
        pass
    assert ended == [4242], "the same stubs must end a real group id -- else this test proves nothing"


def test_a_left_over_group_that_will_not_die_keeps_its_record_and_gives_the_lock_back(tmp_path, monkeypatch):
    batch = _batch(tmp_path)
    orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(batch.root)], start_new_session=True)
    try:
        _record(batch, pid=_dead_pid(), child=orphan.pid)

        def stuck(pgid, kill_grace_s):
            raise rb.GroupAlive(pgid, 30.0)

        monkeypatch.setattr(rb, "end_group_by_id", stuck)
        with pytest.raises(rb.GroupAlive):
            with rb.Lock(batch.root, say=lambda s: None):
                pass
        assert json.loads((batch.root / RUNNER_FILE).read_text(encoding="utf-8"))["child"] == orphan.pid, \
            "남은 그룹을 기록에서 지웠습니다 — 다음 실행이 그 그룹을 모르게 됩니다"
        monkeypatch.undo()
        with rb.Lock(batch.root, say=lambda s: None, kill_grace_s=1.0):     # the lock was given back: the next run takes it and ends the group
            pass
        assert orphan.wait(timeout=10) is not None
    finally:
        if orphan.poll() is None:
            orphan.kill()


# ── main ──

def test_the_runner_refuses_a_dirty_tree_before_it_makes_anything(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rb.provenance, "read", lambda root: {"generator_commit": COMMIT, "generator_dirty": True})
    assert rb.main(["--batch", str(tmp_path / "batch"), "--episodes", "20"]) == 2
    assert not (tmp_path / "batch").exists()
    assert "generator_dirty" in capsys.readouterr().err


def test_the_runner_refuses_to_continue_a_batch_at_another_commit(tmp_path, monkeypatch, capsys):
    _batch(tmp_path)
    monkeypatch.setattr(rb.provenance, "read", lambda root: {"generator_commit": "f" * 40, "generator_dirty": False})
    assert rb.main(["--batch", str(tmp_path / "batch"), "--episodes", "25"]) == 2
    assert "generator_commit" in capsys.readouterr().err


def _witness(lock_path, out):
    """A stage that waits until the RUNNER record names a process group, then writes down what it read
    and its own group. It is the only way to see the record while a stage runs: the run blocks."""
    return ("import json, os, time\n"
            "held = {}\n"
            "for _ in range(250):\n"
            "    try:\n"
            f"        held = json.loads(open({str(lock_path)!r}, encoding='utf-8').read())\n"
            "    except ValueError:\n"                       # the runner is in the middle of writing it
            "        held = {}\n"
            "    if held.get('child') is not None:\n"
            "        break\n"
            "    time.sleep(0.02)\n"
            f"json.dump({{'held': held, 'pgid': os.getpgid(0)}}, open({str(out)!r}, 'w'))\n")


def test_the_runner_tells_the_lock_which_group_its_stage_is_in(tmp_path):
    # The record's "child" is the only link between a running stage and the next run's clean-up of it.
    batch = _batch(tmp_path)
    witness = tmp_path / "witness.json"
    argv = _script(tmp_path, _witness(batch.root / RUNNER_FILE, witness))
    with rb.Lock(batch.root) as lock:
        assert rb.stage_runner(lock)(argv, tmp_path / "stage.log", cwd=tmp_path, **FAST) == 0
        seen = json.loads(witness.read_text(encoding="utf-8"))
        assert seen["held"] == {"pid": os.getpid(), "child": seen["pgid"]}, "단계가 도는 동안 잠금이 그룹을 몰랐습니다"
        assert json.loads((batch.root / RUNNER_FILE).read_text(encoding="utf-8")) == {"pid": os.getpid(), "child": None}


def test_main_hands_its_stages_the_runner_that_tells_the_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(rb.provenance, "read", lambda root: {"generator_commit": COMMIT, "generator_dirty": False})
    root, witness = (tmp_path / "batch").resolve(), tmp_path / "witness.json"
    argv = _script(tmp_path, _witness(root / RUNNER_FILE, witness))
    ran = []

    def one_stage(batch, stages, *, read_tree, say=rb.progress):
        ran.append(stages.run(argv, tmp_path / "stage.log", cwd=tmp_path, **FAST))
        return 0

    monkeypatch.setattr(rb, "run", one_stage)
    assert rb.main(["--batch", str(root), "--episodes", "10"]) == 0
    assert ran == [0]
    seen = json.loads(witness.read_text(encoding="utf-8"))
    assert seen["held"] == {"pid": os.getpid(), "child": seen["pgid"]}
    assert not (root / RUNNER_FILE).exists()


def test_the_real_stages_build_the_commands_the_design_names(tmp_path):
    batch = _batch(tmp_path)
    seen = []
    stages = rb.Stages(repo=Path("/repo"), isaac_python="/isaac/python.sh", python="/venv/python",
                       lerobot_python="/lerobot/python", run=lambda argv, log, **kw: seen.append((argv, log, kw)) or 0)
    plan = _plan(batch, 20)
    chunk = batch.chunk_dir(plan)
    for name in ALL:
        getattr(stages, name)(batch, plan, chunk)
    by = {Path(log).name: ([str(a) for a in argv], kw) for argv, log, kw in seen}
    gen, kw = by["generate.log"]
    assert gen[:2] == ["/isaac/python.sh", "/repo/generation/envs/run_s1a.py"]
    assert gen[gen.index("--seq") + 1] == "20" and gen[gen.index("--episodes") + 1] == "5"       # the short last chunk
    assert gen[gen.index("--scenario") + 1] == "S1d" and gen[gen.index("--out") + 1] == str(chunk / EPISODES_DIR)
    assert kw["watch_dir"] == chunk / EPISODES_DIR                                               # only generation is watched
    assert all("watch_dir" not in k for name, (_, k) in by.items() if name != "generate.log")
    assert by["label.log"][0][:2] == ["/venv/python", "/repo/generation/labeling/label_episodes.py"]
    conv = by["convert.log"][0]
    assert conv[:2] == ["/lerobot/python", "/repo/generation/convert_lerobot.py"]
    assert conv[conv.index("--out") + 1] == str(chunk / DATASET_DIR) and conv[conv.index("--repo-id") + 1] == "test/batch"
    val = by["validate.log"][0]
    assert val[:3] == ["/venv/python", "/repo/validator/validate_dataset.py", str(chunk / DATASET_DIR)]
    assert "--repro-hook" not in val
    meas = by["measure.log"][0]
    assert meas[:2] == ["/lerobot/python", "/repo/generation/scene/measure_slice.py"]
    assert meas[meas.index("--log") + 1] == str(chunk / LOGS_DIR / "generate.log")
    assert meas[meas.index("--dataset") + 1] == str(chunk / DATASET_DIR)

    # A chunk of condition-table rows: one command a row, logged by its row, and the scorecard for its measure.
    table = _table_batch(tmp_path / "table")
    (rows,) = table.chunks()
    rchunk = table.chunk_dir(rows)
    seen.clear()
    for seq in rows.rows:
        stages.generate_row(table, rows, seq, rchunk)
    for name in ALL[1:]:
        getattr(stages, name)(table, rows, rchunk)
    by = {Path(log).name: ([str(a) for a in argv], kw) for argv, log, kw in seen}
    for seq in rows.rows:
        gen, kw = by[f"generate-{seq:04d}.log"]
        assert gen == ["/isaac/python.sh", "/repo/generation/envs/run_s1a.py", "--scenario", "S1e", "--seq", str(seq),
                       "--episodes", "1", "--max-steps", "1000", "--out", str(rchunk / EPISODES_DIR)]
        assert kw["watch_dir"] == rchunk / EPISODES_DIR and kw["first_write_s"] == rb.FIRST_WRITE_S
    # The scorecard reads the release with pyarrow alone: the interpreter of the labeller and the validator.
    assert by["measure.log"][0] == ["/venv/python", "/repo/generation/scorecard.py", "--dataset",
                                    str(rchunk / DATASET_DIR), "--out", str(rchunk / LOGS_DIR / "measure.json")]
    conv = by["convert.log"][0]
    assert conv[conv.index("--out") + 1] == str(rchunk / DATASET_DIR) and conv[conv.index("--repo-id") + 1] == "test/batch"
    with pytest.raises(ValueError, match="행"):
        stages.generate(table, rows, rchunk)                    # a chunk of rows is not one run


# ── a chunk of condition-table rows: one run of the generator a row, resumed row by row ──

class _Commands:
    """What rb.Stages hands its commands to, in place of run_stage: each command is noted with its log and keywords,
    and leaves what its stage would -- the generator's, the row its --seq names."""

    def __init__(self, batch):
        self.batch, self.seen = batch, []

    def _plan(self, chunk):
        (plan,) = [p for p in self.batch.chunks() if self.batch.chunk_dir(p) == chunk]
        return plan

    def __call__(self, argv, log, **kw):
        argv = [str(a) for a in argv]
        self.seen.append((argv, Path(log), kw))
        Path(log).write_text("ran\n", encoding="utf-8")
        script = Path(argv[1]).name

        def after(flag):
            return Path(argv[argv.index(flag) + 1])

        if script == "run_s1a.py":
            chunk = after("--out").parent
            _write_row(self.batch, self._plan(chunk), int(argv[argv.index("--seq") + 1]), chunk)
        elif script == "label_episodes.py":
            for npz in after("--episodes").glob("*.npz"):
                npz.with_suffix(".labels.json").write_text("{}", encoding="utf-8")
        elif script == "convert_lerobot.py":
            (after("--out") / "meta").mkdir(parents=True)
        elif script == "validate_dataset.py":
            after("--json").write_text(json.dumps(VALIDATE_REPORT), encoding="utf-8")
        elif script == "scorecard.py":
            report = _card_report(self._plan(after("--dataset").parent).scenario)
            after("--out").write_text(json.dumps(report), encoding="utf-8")
        return 0


def test_a_table_chunk_runs_one_process_per_row(tmp_path):
    batch = _table_batch(tmp_path)
    (plan,) = batch.chunks()
    chunk, commands = batch.chunk_dir(plan), _Commands(batch)
    stages = rb.Stages(repo=Path("/repo"), isaac_python="/isaac/python.sh", python="/venv/python",
                       lerobot_python="/lerobot/python", run=commands)
    lines, say = _said()
    assert _run(batch, stages, say=say) == 0
    runs = [(argv, log) for argv, log, _ in commands.seen if argv[1] == "/repo/generation/envs/run_s1a.py"]
    assert [argv[argv.index("--seq") + 1] for argv, _ in runs] == [str(seq) for seq in plan.rows]
    for (argv, log), seq in zip(runs, plan.rows):
        assert argv[argv.index("--episodes") + 1] == "1" and argv[argv.index("--scenario") + 1] == "S1e"
        assert log == chunk / LOGS_DIR / f"generate-{seq:04d}.log"
    assert [Path(argv[1]).name for argv, _, _ in commands.seen[len(runs):]] == [
        "label_episodes.py", "convert_lerobot.py", "validate_dataset.py", "scorecard.py"]
    record = read_seal(chunk)
    assert (record["scenario"], record["rows"], record["start_seq"]) == ("S1e", list(plan.rows), plan.rows[0])
    assert record["episode_ids"] == batch.episode_ids(plan) and record["files"] == sealed_files(chunk)
    assert list(record["rows_s"]) == [str(seq) for seq in plan.rows]
    assert all(isinstance(s, (int, float)) and s >= 0 for s in record["rows_s"].values())
    card = _card_report("S1e")["scenarios"]["S1e"]
    assert record["measurement"] == {"outcome": card["safety"]["outcome"], "degenerate": card["degenerate_excluded"],
                                     "robot_falls": card["anomalies"]["robot_falls"],
                                     "rack_contact_events": card["anomalies"]["rack_contact_events"]}
    (line,) = [l for l in lines if "S1e-000" in l and "봉인" in l]
    for shown in ("fail 2", "degenerate 1", "넘어짐 2", "접촉 3"):
        assert shown in line, (shown, line)


def test_a_resumed_chunk_skips_its_finished_rows(tmp_path):
    batch = _table_batch(tmp_path)
    (plan,) = batch.chunks()
    chunk = batch.chunk_dir(plan)
    episodes = chunk / EPISODES_DIR
    done, (fourth, fifth) = plan.rows[:3], plan.rows[3:]
    for seq in done:
        _write_row(batch, plan, seq, chunk)
    # What an earlier run left besides: the fourth row cut off after its npz, the products of the stages after
    # generation (a run that failed after its rows), and a file of no row.
    (episodes / f"{_eid(batch, plan, fourth)}.npz").write_bytes(b"npz only")
    (episodes / f"{_eid(batch, plan, done[0])}.labels.json").write_text('{"stale": true}', encoding="utf-8")
    (chunk / DATASET_DIR / "meta").mkdir(parents=True)
    (chunk / LOGS_DIR).mkdir()
    for name in (f"generate-{done[0]:04d}.log", "convert.log", "validate.json", "measure.json"):
        (chunk / LOGS_DIR / name).write_text("an earlier run", encoding="utf-8")
    (episodes / "left-by-the-run-that-died.txt").write_text("x", encoding="utf-8")
    kept = {p: p.read_bytes() for seq in done for p in _row_files(chunk, _eid(batch, plan, seq))
            if not p.name.endswith(".labels.json")}
    seen = {}

    class Looking(FakeStages):
        def generate_row(self, batch, plan, seq, chunk):
            if seq == fourth:
                seen["kept"] = {p: p.read_bytes() for p in kept if p.exists()}
                seen["left"] = sorted(p.name for p in (chunk / DATASET_DIR, *(chunk / LOGS_DIR).iterdir(),
                                                       *episodes.glob("*.labels.json"),
                                                       episodes / "left-by-the-run-that-died.txt") if p.exists())
            return super().generate_row(batch, plan, seq, chunk)

    stages, (lines, say) = Looking(), _said()
    assert _run(batch, stages, say=say) == 0
    assert [c for c in stages.calls if c[0] == "generate"] == [("generate", plan.key, fourth),
                                                                ("generate", plan.key, fifth)]
    assert seen["kept"] == kept, "끝난 행의 파일이 바뀌었습니다"
    assert seen["left"] == [f"generate-{done[0]:04d}.log"], "생성 뒤 단계의 산출물이나 어느 행의 것도 아닌 파일이 남았습니다"
    assert all(p.read_bytes() == b for p, b in kept.items() if p.parent == episodes)
    assert (chunk / LOGS_DIR / f"generate-{done[0]:04d}.log").read_text(encoding="utf-8") == "an earlier run"
    assert list(read_seal(chunk)["rows_s"]) == [str(fourth), str(fifth)], "이번 실행이 만든 행만 적습니다"
    assert any(str(fourth) in line and "다시" in line for line in lines), lines


def _streams_not_yet_in_place(episodes, eid):
    """A run cut off after its sidecar and before its streams were renamed from their part names."""
    for cam in CAMERAS:
        (episodes / MEDIA_DIR / f"{eid}.{cam}.mkv").rename(episodes / MEDIA_DIR / f".{eid}.{cam}.part.mkv")


@pytest.mark.parametrize("break_row", [
    lambda episodes, eid: (episodes / MEDIA_DIR / f"{eid}.{CAMERAS[-1]}.mkv").unlink(),
    lambda episodes, eid: (episodes / MEDIA_DIR / f"{eid}.{CAMERAS[0]}.mkv").write_bytes(b""),
    _streams_not_yet_in_place,
], ids=["one of the streams missing", "one of the streams empty", "the streams not yet in place"])
def test_a_half_written_row_is_redone(tmp_path, break_row):
    batch = _table_batch(tmp_path, parts=(("S1e", (3, 8)),))
    (plan,) = batch.chunks()
    chunk = batch.chunk_dir(plan)
    for seq in plan.rows:
        _write_row(batch, plan, seq, chunk)
    break_row(chunk / EPISODES_DIR, _eid(batch, plan, 3))
    stages, (lines, say) = FakeStages(), _said()
    assert _run(batch, stages, say=say) == 0
    assert [c for c in stages.calls if c[0] == "generate"] == [("generate", plan.key, 3)]
    assert any("행 3" in line and "다시" in line for line in lines), lines


def test_a_zero_byte_sidecar_is_redone(tmp_path):
    # What a hard reset leaves: the file is there, and its bytes never reached the disk.
    batch = _table_batch(tmp_path, parts=(("S1e", (3, 8)),))
    (plan,) = batch.chunks()
    chunk = batch.chunk_dir(plan)
    for seq in plan.rows:
        _write_row(batch, plan, seq, chunk)
    (chunk / EPISODES_DIR / f"{_eid(batch, plan, 8)}.json").write_bytes(b"")
    stages = FakeStages()
    assert _run(batch, stages, say=lambda s: None) == 0
    assert [c for c in stages.calls if c[0] == "generate"] == [("generate", plan.key, 8)]


def _edit_sidecar(change):
    def edit(episodes, eid):
        path = episodes / f"{eid}.json"
        side = json.loads(path.read_text(encoding="utf-8"))
        change(side)
        path.write_text(json.dumps(side), encoding="utf-8")
    return edit


def _name_a_pack_not_there(side):
    side["media"]["pack"] = {"file": f"{MEDIA_DIR}/{side['episode_id']}.depth.pack.npz", "frames": 3}


@pytest.mark.parametrize("break_row", [
    lambda episodes, eid: (episodes / f"{eid}.npz").unlink(),
    lambda episodes, eid: (episodes / f"{eid}.npz").write_bytes(b""),
    lambda episodes, eid: (episodes / f"{eid}.json").unlink(),
    lambda episodes, eid: (episodes / f"{eid}.json").write_text("{", encoding="utf-8"),
    lambda episodes, eid: (episodes / f"{eid}.json").write_text("[]", encoding="utf-8"),
    _edit_sidecar(lambda s: s.update(episode_id="S1e_0006_seed1")),
    _edit_sidecar(lambda s: s.update(run_start_seq=6)),
    _edit_sidecar(lambda s: s.update(run_start_seq=2.0)),
    _edit_sidecar(lambda s: s["sim_backend"].update(generator_commit="f" * 40)),
    _edit_sidecar(lambda s: s["sim_backend"].update(generator_dirty=True)),
    _edit_sidecar(lambda s: s["sim_backend"].pop("generator_dirty")),
    _edit_sidecar(lambda s: s.pop("media")),
    _edit_sidecar(lambda s: s["media"].pop(CAMERAS[3])),
    _edit_sidecar(lambda s: s["media"][CAMERAS[1]].update(file=None)),
    _edit_sidecar(_name_a_pack_not_there),
    lambda episodes, eid: (episodes / MEDIA_DIR / f"{eid}.{CAMERAS[2]}.mkv").unlink(),
    lambda episodes, eid: (episodes / MEDIA_DIR / f"{eid}.{CAMERAS[2]}.mkv").write_bytes(b""),
], ids=["no npz", "an empty npz", "no sidecar", "a sidecar that is not JSON", "a sidecar that is not an object",
        "another id", "the chunk's first row as its run's start", "its run's start as a float", "another commit",
        "a dirty tree", "no word on the tree", "no media", "a camera the media does not name",
        "a camera named without a file", "a further stream named and not there", "a stream missing",
        "an empty stream"])
def test_a_row_is_finished_only_as_its_run_leaves_a_finished_one(tmp_path, break_row):
    batch = _table_batch(tmp_path, parts=(("S1e", (6, 2)),))
    (plan,) = batch.chunks()
    chunk = batch.chunk_dir(plan)
    _write_row(batch, plan, 2, chunk)
    assert rb.row_done(batch, plan, 2, chunk) and rb.row_problem(batch, plan, 2, chunk) is None
    assert not rb.row_done(batch, plan, 6, chunk), "만든 적 없는 행이 끝난 행입니다"
    break_row(chunk / EPISODES_DIR, _eid(batch, plan, 2))
    assert not rb.row_done(batch, plan, 2, chunk)
    assert rb.row_problem(batch, plan, 2, chunk), "끝나지 않은 까닭을 말해야 합니다"


def test_a_row_that_never_writes_is_stalled_early(tmp_path):
    # A boot that hangs writes nothing at all, and the grace is for a boot that is slow. A file of an earlier row
    # in the same directory does not count: it was written before this run started.
    out = tmp_path / "episodes"
    out.mkdir()
    earlier = out / "S1e_0003_seed1.npz"
    earlier.write_bytes(b"an earlier row")
    os.utime(earlier, (time.time() - 3600,) * 2)
    argv = _script(tmp_path, "import time\ntime.sleep(60)\n")
    lines, say = _said()
    started = time.monotonic()
    code = rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, watch_dir=out, say=say,
                        **dict(FAST, grace_s=30.0, stall_s=30.0), first_write_s=0.5)
    assert code == rb.STALLED
    assert time.monotonic() - started < 15, "유예가 끝날 때까지 기다렸습니다"
    assert any("0.5 초" in line for line in lines), lines


def test_a_row_that_writes_is_not_held_to_the_first_write_deadline(tmp_path):
    # The runner looks once a poll, so a file written before the deadline can first be seen after it: it counts.
    out = tmp_path / "episodes"
    out.mkdir()
    argv = _script(tmp_path, f"import time\nopen({str(out / 'media.mkv')!r}, 'wb').write(b'x')\ntime.sleep(3.5)\n")
    code = rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, watch_dir=out,
                        **dict(FAST, grace_s=30.0, stall_s=30.0, poll_s=2.5), first_write_s=0.5)
    assert code == 0


def test_a_file_an_earlier_run_left_dated_ahead_of_the_clock_is_not_a_first_write(tmp_path):
    # A clock set back after an earlier row was written (a correction after a reboot) leaves that row's files
    # dated in the future. They say nothing of this run: one that writes nothing is ended at the deadline all the
    # same.
    out = tmp_path / "episodes"
    out.mkdir()
    earlier = out / "S1e_0003_seed1.npz"
    earlier.write_bytes(b"an earlier row")
    os.utime(earlier, (time.time() + 3600,) * 2)
    argv = _script(tmp_path, "import time\ntime.sleep(60)\n")
    started = time.monotonic()
    code = rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, watch_dir=out, say=lambda line: None,
                        **dict(FAST, grace_s=30.0, stall_s=30.0), first_write_s=0.5)
    assert code == rb.STALLED
    assert time.monotonic() - started < 15, "유예가 끝날 때까지 기다렸습니다"


@pytest.mark.parametrize("how", ["new", "changed"])
def test_a_file_the_stage_writes_is_a_first_write_whatever_time_it_carries(tmp_path, how):
    # A coarse clock can date a file the stage writes before the stage started: a file that is new, or that has
    # changed since the stage started, is this run's write -- its time is not compared with the start.
    out = tmp_path / "episodes"
    out.mkdir()
    target = out / "media.mkv"
    if how == "changed":
        target.write_bytes(b"left")
        os.utime(target, (1000, 1000))
    argv = _script(tmp_path, f"import os, time\np = {str(target)!r}\nopen(p, 'ab').write(b'written')\n"
                             "os.utime(p, (1000, 1000))\ntime.sleep(3.5)\n")
    code = rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, watch_dir=out, say=lambda line: None,
                        **dict(FAST, grace_s=30.0, stall_s=30.0), first_write_s=1.5)
    assert code == 0


def test_a_file_replaced_by_another_of_the_same_size_and_time_is_a_first_write(tmp_path):
    # A new file put in place of an earlier one, of the same size and with the same modification time: another
    # inode, and a status change the stage made -- this run's write.
    out = tmp_path / "episodes"
    out.mkdir()
    target = out / "media.mkv"
    target.write_bytes(b"A" * 32)
    os.utime(target, (1000, 1000))
    argv = _script(tmp_path, f"import os, time\np = {str(target)!r}\nopen(p + '.new', 'wb').write(b'B' * 32)\n"
                             "os.utime(p + '.new', (1000, 1000))\nos.replace(p + '.new', p)\ntime.sleep(3.5)\n")
    code = rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, watch_dir=out, say=lambda line: None,
                        **dict(FAST, grace_s=30.0, stall_s=30.0, poll_s=0.1), first_write_s=1.5)
    assert code == 0
    assert target.read_bytes() == b"B" * 32 and os.stat(target).st_mtime_ns == 1000 * 10**9   # replaced: same size and time


def test_the_first_write_deadline_runs_on_a_clock_that_moves(tmp_path, monkeypatch):
    # The wall clock can stand still or jump; the deadline is measured on the monotonic clock.
    import types

    out = tmp_path / "episodes"
    out.mkdir()
    frozen = types.SimpleNamespace(**{name: getattr(time, name) for name in dir(time) if not name.startswith("_")})
    frozen.time = lambda: 1_000_000.0
    monkeypatch.setattr(rb, "time", frozen)
    argv = _script(tmp_path, "import time\ntime.sleep(60)\n")
    started = time.monotonic()
    code = rb.run_stage(argv, tmp_path / "stage.log", cwd=tmp_path, watch_dir=out, say=lambda line: None,
                        **dict(FAST, grace_s=30.0, stall_s=30.0), first_write_s=0.5)
    assert code == rb.STALLED
    assert time.monotonic() - started < 15, "벽시계가 멈추자 기한이 오지 않았습니다"


def test_the_first_write_deadline_is_for_rows_only(tmp_path):
    assert rb.FIRST_WRITE_S == 120.0
    assert inspect.signature(rb.run_stage).parameters["first_write_s"].default is None
    seen = []
    stages = rb.Stages(repo=Path("/repo"), isaac_python="/isaac/python.sh", python="/venv/python",
                       lerobot_python="/lerobot/python", run=lambda argv, log, **kw: seen.append(kw) or 0)
    s1d = _batch(tmp_path)
    plan = _plan(s1d, 0)
    stages.generate(s1d, plan, s1d.chunk_dir(plan))
    table = _table_batch(tmp_path / "table")
    (rows,) = table.chunks()
    stages.generate_row(table, rows, rows.rows[0], table.chunk_dir(rows))
    chunk_kw, row_kw = seen
    assert "first_write_s" not in chunk_kw and chunk_kw["watch_dir"] == s1d.chunk_dir(plan) / EPISODES_DIR
    assert row_kw["first_write_s"] == rb.FIRST_WRITE_S and row_kw["watch_dir"] == table.chunk_dir(rows) / EPISODES_DIR


def test_a_stalled_row_is_retried_alone(tmp_path):
    batch = _table_batch(tmp_path, parts=(("S1e", (2, 6, 1)),))
    (plan,) = batch.chunks()
    stages, (lines, say) = FakeStages(codes={("generate", plan.key, 6): [rb.STALLED, rb.STALLED]}), _said()
    assert _run(batch, stages, say=say) == 0
    assert [c[2] for c in stages.calls if c[0] == "generate"] == [2, 6, 6, 6, 1]
    kept = batch.root / rb.STALLED_LOGS
    assert sorted(p.name for p in kept.iterdir()) == [f"{plan.key}.0006.attempt{n}.generate.log" for n in (1, 2)]
    assert (kept / f"{plan.key}.0006.attempt2.generate.log").read_text(encoding="utf-8") == "stalled, row 6, attempt 2"
    assert any(str(kept) in line for line in lines), "진행 줄이 로그가 어디 남았는지 말해야 합니다"

    # Three stalls of one row fail the chunk, naming the row; the row before it stays as it was made.
    again = _table_batch(tmp_path / "again", parts=(("S1e", (2, 6, 1)),))
    (plan,) = again.chunks()
    chunk = again.chunk_dir(plan)
    stages = FakeStages(codes={("generate", plan.key, 6): [rb.STALLED] * 3})
    with pytest.raises(rb.ChunkFailed, match="행 6") as e:
        rb.run_chunk(again, plan, stages, read_tree=_clean, say=lambda s: None)
    assert e.value.stage == "generate"
    first = _eid(again, plan, 2)
    assert (chunk / EPISODES_DIR / f"{first}.npz").read_bytes() == b"npz " + first.encode()
    assert all((chunk / EPISODES_DIR / MEDIA_DIR / f"{first}.{cam}.mkv").is_file() for cam in CAMERAS)
    assert ("generate", plan.key, 1) not in stages.calls


def test_every_row_made_is_flushed_to_disk_before_the_next_starts(tmp_path, monkeypatch):
    # A row judged finished is counted so from then on: after a hard reset its files are not left cut short with
    # bytes that pass for a finished row.
    batch = _table_batch(tmp_path, parts=(("S1e", (2, 6, 1)),))
    (plan,) = batch.chunks()
    stages = FakeStages()
    monkeypatch.setattr(rb.os, "sync", lambda: stages.calls.append(("sync",)))
    assert _run(batch, stages, say=lambda s: None) == 0
    rows = [c for c in stages.calls if c[0] in ("generate", "sync")][:6]
    assert rows == [("generate", plan.key, 2), ("sync",), ("generate", plan.key, 6), ("sync",),
                    ("generate", plan.key, 1), ("sync",)], stages.calls


def test_a_rows_run_start_is_the_row(tmp_path):
    # Every row is a run of its own: a sidecar that names the chunk's first row as its run's start was made in
    # another run -- as a chunk that is one run would have made it.
    batch = _table_batch(tmp_path, parts=(("S1e", (5, 3)),))
    (plan,) = batch.chunks()
    stages, (lines, say) = FakeStages(sidecar=lambda s: s.update(run_start_seq=plan.rows[0])), _said()
    assert _run(batch, stages, say=say) == 1
    assert any("run_start_seq" in line and "행 3" in line for line in lines), lines
    assert not [c for c in stages.calls if c[0] == "label"]


def test_a_row_whose_run_ends_well_and_leaves_no_finished_row_fails_the_chunk(tmp_path):
    # A run that exits 0 without one of its streams: a defect of the generator, not a stall -- said for the row,
    # and nothing after generation runs on it.
    batch = _table_batch(tmp_path, parts=(("S1e", (5, 3)),))
    (plan,) = batch.chunks()
    stages, (lines, say) = FakeStages(sidecar=lambda s: s["media"].pop(CAMERAS[0])), _said()
    assert _run(batch, stages, say=say) == 1
    assert any("행 5" in line and CAMERAS[0] in line for line in lines), lines
    assert [c for c in stages.calls if c[0] == "generate"] == [("generate", plan.key, 5)]
    assert not [c for c in stages.calls if c[0] == "label"]


def test_the_rows_of_a_chunk_are_checked_in_the_order_they_were_asked(tmp_path):
    batch = _table_batch(tmp_path, parts=(("S1e", (5, 3)),))
    (plan,) = batch.chunks()
    chunk = batch.chunk_dir(plan)
    for seq in plan.rows:
        _write_row(batch, plan, seq, chunk)
    assert rb.check_generated(batch, plan, chunk) == batch.episode_ids(plan)
    _write_row(batch, plan, 3, chunk, sidecar=lambda s: s.update(run_start_seq=5))
    with pytest.raises(rb.ChunkFailed, match="run_start_seq=5"):
        rb.check_generated(batch, plan, chunk)


def test_a_table_chunk_that_failed_after_its_rows_is_finished_from_its_rows(tmp_path):
    batch = _table_batch(tmp_path, parts=(("S1e", (4, 1)),))
    (plan,) = batch.chunks()
    lines, say = _said()
    assert _run(batch, FakeStages(codes={("convert", plan.key): [1]}), say=say) == 1
    assert any("실패" in line and "끝난 행" in line for line in lines), lines
    again = FakeStages()
    assert _run(batch, again, say=lambda s: None) == 0
    assert [c[0] for c in again.calls] == ALL[1:], "끝난 행을 다시 만들었습니다"
    assert read_seal(batch.chunk_dir(plan))["rows_s"] == {}


def test_the_table_measure_is_the_scorecard(tmp_path):
    from tests.generation.release_fixture import HRI_SCENARIO, hri_cases, hri_recording, write_release
    batch = _table_batch(tmp_path, parts=(("S1e", tuple(range(10))),))
    (plan,) = batch.chunks()
    chunk = batch.chunk_dir(plan)
    (chunk / LOGS_DIR).mkdir(parents=True)
    write_release(chunk / DATASET_DIR, [*hri_cases(), hri_recording(9, degenerate=True)], scenario=HRI_SCENARIO)
    # Run by the interpreter of the labeller and the validator: one without lerobot will do, and the one with it is
    # not there.
    stages = rb.Stages(repo=REPO, isaac_python="/isaac/python.sh", python=sys.executable,
                       lerobot_python=str(tmp_path / "no-such-python"))
    assert stages.measure(batch, plan, chunk) == 0, (chunk / LOGS_DIR / "measure.log").read_text(encoding="utf-8")
    card = json.loads((chunk / LOGS_DIR / "measure.json").read_text(encoding="utf-8"))["scenarios"]["S1e"]
    assert rb.measurement_summary(chunk, plan) == {
        "outcome": card["safety"]["outcome"], "degenerate": card["degenerate_excluded"],
        "robot_falls": card["anomalies"]["robot_falls"], "rack_contact_events": card["anomalies"]["rack_contact_events"]}
    # The fixture's cases as test_scorecard.py works them out, and the degenerate one left out and counted.
    assert card["degenerate_excluded"] == 1 and sum(card["safety"]["outcome"].values()) == 9
    assert (card["anomalies"]["robot_falls"], card["anomalies"]["rack_contact_events"]) == (2, 1)


def _measured_chunk(tmp_path, rows, cases):
    """A chunk of S1e rows holding ``cases`` as a distribution, measured by the scorecard: (batch, plan, chunk, card)."""
    from tests.generation.release_fixture import TABLE_ASSETS, TABLE_SCENARIO, write_release
    batch = _table_batch(tmp_path, parts=(("S1e", rows),))
    (plan,) = batch.chunks()
    chunk = batch.chunk_dir(plan)
    (chunk / LOGS_DIR).mkdir(parents=True)
    write_release(chunk / DATASET_DIR, cases, scenario=TABLE_SCENARIO, assets=TABLE_ASSETS)
    stages = rb.Stages(repo=REPO, isaac_python="/isaac/python.sh", python=sys.executable,
                       lerobot_python=str(tmp_path / "no-such-python"))
    assert stages.measure(batch, plan, chunk) == 0, (chunk / LOGS_DIR / "measure.log").read_text(encoding="utf-8")
    card = json.loads((chunk / LOGS_DIR / "measure.json").read_text(encoding="utf-8"))["scenarios"]["S1e"]
    return batch, plan, chunk, card


def _seal_record(chunk, plan):
    return {"stages_s": {"generate": 1.0}, "rows_s": {}, "validation": {"error": 0, "warn": 0},
            "measurement": rb.measurement_summary(chunk, plan)}


def _map_cells(card):
    return [c for row in card["failure_map"]["outcome"] for c in row if c is not None]


def test_a_grid_chunk_seals_its_grid_outcomes(tmp_path):
    from tests.generation.release_fixture import table_cases
    cases = table_cases()
    # Random and grid rows in one chunk: the four counts stay the random rows', the grid rows are told apart.
    _, plan, chunk, card = _measured_chunk(tmp_path / "mixed", (0, 1, 2, 50, 57, 98), cases)
    cells = _map_cells(card)
    summary = rb.measurement_summary(chunk, plan)
    assert card["grid_episodes"] == 3 and len(cells) == 3
    assert summary["grid"] == {
        "episodes": 3, "degenerate": cells.count("degenerate"),
        "outcome": {k: cells.count(k) for k in ("fail", "near_miss", "nominal")}}
    assert sum(summary["grid"]["outcome"].values()) + summary["grid"]["degenerate"] == 3
    assert summary["outcome"] == card["safety"]["outcome"] and sum(summary["outcome"].values()) == 3
    line = rb.seal_line("S1e-0000", _seal_record(chunk, plan) | {"rows_s": {"0": 1.0}})
    assert "무작위 행 3 — " in line and "격자 행 3 — " in line, line
    g = summary["grid"]
    assert (f"격자 행 3 — fail {g['outcome']['fail']} · near_miss {g['outcome']['near_miss']}"
            f" · nominal {g['outcome']['nominal']} · degenerate {g['degenerate']}") in line

    # Grid rows alone: the random counts are all zero and the line does not speak of random rows.
    _, plan, chunk, card = _measured_chunk(tmp_path / "grid", (50, 57, 98), cases[3:])
    cells = _map_cells(card)
    summary = rb.measurement_summary(chunk, plan)
    assert sum(summary["outcome"].values()) == 0 and summary["grid"]["episodes"] == 3
    assert summary["grid"]["outcome"] == {k: cells.count(k) for k in ("fail", "near_miss", "nominal")}
    line = rb.seal_line("S1e-0001", _seal_record(chunk, plan) | {"rows_s": {"50": 1.0}})
    assert "격자 행 3" in line and "무작위 행" not in line and "라벨" not in line, line


def test_a_chunk_of_random_rows_seals_as_before(tmp_path):
    from tests.generation.release_fixture import table_cases
    _, plan, chunk, card = _measured_chunk(tmp_path, (0, 1, 2), table_cases()[:3])
    summary = rb.measurement_summary(chunk, plan)
    assert "grid" not in summary and summary["outcome"] == card["safety"]["outcome"]
    line = rb.seal_line("S1e-0000", _seal_record(chunk, plan) | {"rows_s": {"0": 1.0}})
    assert "무작위 행 3 — " in line and "격자" not in line, line


def test_a_grid_map_that_cannot_be_read_leaves_no_summary(tmp_path):
    from tests.generation.release_fixture import table_cases
    _, plan, chunk, card = _measured_chunk(tmp_path, (0, 1, 2, 50, 57, 98), table_cases())
    path = chunk / LOGS_DIR / "measure.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    good = report["scenarios"]["S1e"]["failure_map"]
    first = next((i, j) for i, r in enumerate(good["outcome"]) for j, c in enumerate(r) if c is not None)

    def broken(**edit):
        r = json.loads(json.dumps(report))
        fm = r["scenarios"]["S1e"]
        fm.update(edit)
        path.write_text(json.dumps(r), encoding="utf-8")
        return rb.measurement_summary(chunk, plan)

    assert rb.measurement_summary(chunk, plan)["grid"]["episodes"] == 3
    emptied = json.loads(json.dumps(good))
    emptied["outcome"][first[0]][first[1]] = None           # fewer cells than grid rows
    renamed = json.loads(json.dumps(good))
    renamed["outcome"][first[0]][first[1]] = "broken"       # a cell that is none of the four names
    for bad in (None, {"absent": "no grid"}, {"outcome": "x"}, {"outcome": [1]}, emptied, renamed):
        assert broken(failure_map=bad) is None, bad
    assert broken(grid_episodes=0, failure_map=None) is not None      # no grid rows: no map is a map of no cells
    assert broken(grid_episodes=0, failure_map={"absent": "no grid"}) is not None
    assert broken(grid_episodes=0) is None                            # a map that holds cells does not fit 0 grid rows
    assert broken(grid_episodes=0, failure_map={"outcome": [[None, None]]}) is not None     # a map of empty cells does
    assert broken(grid_episodes=-1, failure_map=None) is None and broken(grid_episodes="0", failure_map=None) is None
    r = json.loads(json.dumps(report))
    del r["scenarios"]["S1e"]["grid_episodes"]                        # a card without its grid count cannot be read
    path.write_text(json.dumps(r), encoding="utf-8")
    assert rb.measurement_summary(chunk, plan) is None
    assert "grid" not in broken(grid_episodes=0, failure_map=None)
    assert "grid" not in broken(grid_episodes=0, failure_map={"absent": "no grid"})


def test_a_degenerate_row_is_told_apart_in_the_grid_and_in_the_random_part(tmp_path):
    from tests.generation.release_fixture import table_cases, table_recording
    cases = table_cases()
    cases[2] = table_recording(2, 2, degenerate=True)         # a random row
    cases[5] = table_recording(5, 98, degenerate=True)        # a grid row
    _, plan, chunk, card = _measured_chunk(tmp_path, (0, 1, 2, 50, 57, 98), cases)
    cells = _map_cells(card)
    summary = rb.measurement_summary(chunk, plan)
    assert cells.count("degenerate") == 1 and card["degenerate_excluded"] == 1
    assert summary["degenerate"] == 1 and sum(summary["outcome"].values()) == 2
    assert summary["grid"]["degenerate"] == 1 and summary["grid"]["episodes"] == 3
    assert sum(summary["grid"]["outcome"].values()) == 2                  # the degenerate row is no outcome
    assert summary["grid"]["outcome"] == {k: cells.count(k) for k in ("fail", "near_miss", "nominal")}
    line = rb.seal_line("S1e-0000", _seal_record(chunk, plan) | {"rows_s": {"0": 1.0}})
    assert "무작위 행 3 — " in line and "격자 행 3 — " in line, line
    assert line.count("degenerate 1") == 2, line


def test_the_seal_line_of_a_table_chunk_without_a_measurement_says_it_may_not_be_countable():
    record = {"stages_s": {"generate": 1.0}, "rows_s": {}, "validation": {"error": 0, "warn": 0}, "measurement": None}
    assert "측정 요약 없음 (logs/measure.json 을 읽지 못했거나 격자 행을 셀 수 없음)" in rb.seal_line("S1e-0000", record)
    one_run = {k: v for k, v in record.items() if k != "rows_s"}
    assert "측정 요약 없음 (logs/measure.json 을 읽지 못함)" in rb.seal_line("0000", one_run)


def _cli(root, monkeypatch, *args):
    """rb.main on a clean tree with the run itself left out: its exit code and the batch it opened (None if none)."""
    monkeypatch.setattr(rb.provenance, "read", lambda repo: _clean())
    opened = []
    monkeypatch.setattr(rb, "run", lambda batch, stages, *, read_tree, say=rb.progress: opened.append(batch) or 0)
    code = rb.main(["--batch", str(root / "batch"), *args])
    return code, (opened[0] if opened else None)


def test_the_cli_takes_rows_for_table_scenarios(tmp_path, monkeypatch, capsys):
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"S1g": [2, 5], "S1e": [3, 1], "S1f": [0]}), encoding="utf-8")
    code, batch = _cli(tmp_path / "a", monkeypatch, "--scenario", "S1e", "S1f", "S1g", "--rows", str(rows))
    assert code == 0
    assert batch.parts == (("S1e", (3, 1)), ("S1f", (0,)), ("S1g", (2, 5))), "--scenario 가 적은 차례로"
    assert (batch.repo_id, batch.chunk_episodes) == ("keti/keti-humanoid-hri-safety", 10)

    # Without --rows a table scenario makes every row of its table.
    code, batch = _cli(tmp_path / "b", monkeypatch, "--scenario", "S1f")
    assert code == 0 and batch.parts == (("S1f", tuple(range(conditions.load_conditions(
        conditions.scenario_entry("S1f")).total))),)

    # Without --scenario the batch is S1d's: its episodes and its own repository.
    code, batch = _cli(tmp_path / "c", monkeypatch, "--episodes", "20")
    assert code == 0 and batch.parts == (("S1d", tuple(range(20))),)
    assert batch.repo_id == "keti/humanoid-safety-s1d"

    not_a_list = tmp_path / "not-a-list.json"
    not_a_list.write_text(json.dumps({"S1e": 3}), encoding="utf-8")
    for args, said in [
        (("--scenario", "S1d", "S1e"), "섞"),
        (("--scenario", "S1e", "--episodes", "10"), "--episodes"),
        (("--scenario", "S1e", "S1f", "--rows", str(rows)), "S1g"),
        (("--scenario", "S1d", "--rows", str(rows)), "--rows"),
        (("--scenario", "S1e", "S1e"), "두 번"),
        (("--scenario", "S1e", "--rows", str(tmp_path / "nowhere.json")), "nowhere.json"),
        (("--scenario", "S1e", "--rows", str(not_a_list)), "S1e"),
    ]:
        capsys.readouterr()
        code, batch = _cli(tmp_path / "bad", monkeypatch, *args)
        assert (code, batch) == (2, None), args
        assert said in capsys.readouterr().err, args
        assert not (tmp_path / "bad" / "batch").exists(), args
    with pytest.raises(SystemExit) as e:                    # S1a has no batch of its own
        rb.main(["--batch", str(tmp_path / "s1a" / "batch"), "--scenario", "S1a"])
    assert e.value.code == 2
