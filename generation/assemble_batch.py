#!/usr/bin/env python3
"""봉인된 청크들을 하나의 배포본으로 조립한다.

    HF_HUB_OFFLINE=1 .venv-lerobot/bin/python generation/assemble_batch.py --batch .tmp/batches/S1d-400
    HF_HUB_OFFLINE=1 .venv-lerobot/bin/python generation/assemble_batch.py --batch <배치 A> --batch <배치 B> --out <배포본> --drop <시나리오>:<행>

영상을 다시 인코딩하지 않으므로 몇 번이든 다시 돌린다 — 카드 · NOTICE · 인벤토리 · 좁혀 싣는 정본을
고친 뒤에는 이것만 다시 돌리면 된다. 청크 배포본은 읽기만 한다.

LeRobot 0.6.1 의 aggregate_datasets 를 쓰되 그것이 깨뜨리는 셋을 고친다:
  1. 변환기가 비운 통계 블록에서 죽는다        → 통계를 뺀 뷰를 넘기고, 통계는 4단계가 다시 만든다
  2. 데이터 parquet 을 pandas 로 다시 쓴다     → 청크의 parquet 에서 Arrow 로 다시 쓴다 (NaN · 스키마 그대로)
  3. 번호 필드의 전역 통계를 옮기지 않는다     → 병합본의 에피소드별 통계에서 다시 합친다

청크마다 과제가 달라도 된다 — 시나리오 여럿의 배치는 시나리오마다 과제 문장이 하나다(meta/scenarios.json 의
robot_task). 과제 표는 청크 차례대로 처음 나온 문장의 순서이고(LeRobot 의 병합이 표를 짓는 순서), 2단계가 다시 쓰는
스텝의 task_index 는 그 청크의 과제 표의 행에서 이 표의 행으로 다시 매긴다. 병합이 지은 표가 이 순서가 아니면 거부한다.

그리고 스스로 검증한다: 통과해야 결과가 <배치>/dataset(--out 을 주면 그 자리)에 놓인다. 그 뒤에 검증기 전체를 돌린다.
  - 스텝 표 = 청크의 행 (스키마 · null · 값 — 번호 셋만 다시 매김)
  - 에피소드 메타 = 청크의 열 + 에피소드별 통계 (옮긴 만큼만 다름 · task_index 는 다시 매긴 번호의 것) ·
    에피소드마다 스텝이 과제 표에서 가리키는 문장 = 그 에피소드의 tasks · meta/info.json 의 total_tasks = 과제 표의 행 수
  - meta/stats.json = 스텝 표에서 직접 잰 값 + 에피소드별 통계를 다시 합친 값
  - 영상 = 카메라 · 에피소드마다 병합본 구간의 패킷이 그 에피소드의 청크 구간의 패킷    - 로더가 스텝과 그림을 읽음
  - 실린 라벨 = 조립한 파일만으로 라벨러를 다시 돌린 결과 (지금 싣는 정본으로 — generation/labeling/check_release_labels.py)

청크는 봉인된 것만 · 봉인 뒤에 바뀌지 않은 것만 · 이 배치의 것만 받는다 — 봉인 기록의 시나리오 · 행 · 에피소드 ID ·
생성 커밋이 배치의 계획과 같아야 한다(행을 적지 않은 기록은 첫 행부터 에피소드 수만큼의 범위 — 실행 하나인 청크).

--batch 를 여러 번 주면 같은 생성 커밋 · 같은 리포의 배치 여럿을 한 배포본으로 묶는다(release_chunks — 배치 하나에
--drop 을 줄 때도 같은 규칙). 실릴 행은 배치들이 계획한 (시나리오, 행) 전부에서 --drop 의 행을 뺀 것이고, 그 행마다
배치들을 통틀어 봉인된 청크 정확히 하나에 있어야 한다 — 두 청크에 있어도, 어느 봉인된 청크에도 없어도 거부한다(그 행을
계획한 청크와 「봉인되지 않음」 · 「디렉터리가 없음」을 청크마다 한 줄로). 봉인되지 않았거나 디렉터리가 없는 청크는 그
행이 모두 봉인된 다른 청크에 있거나 --drop 의 행일 때만 건너뛴다. --drop 의 행은 어느 배치가 계획한 행이어야 하고,
봉인된 청크에 있으면 거부한다 — 봉인된 청크는 통째로 실리고 청크에서 에피소드 하나를 빼는 길은 없다. 에피소드는 배치를
준 차례대로, 배치 안에서는 그 배치의 청크 차례대로 실린다. 배치가 여럿이면 청크를 <배치 디렉터리 이름>/<청크> 로
말한다. 배치가 여럿이거나 --drop 을 주면 건너뛴 청크와 뺀 행을 조립의 첫 줄들에 적는다. 배포본은 --out 에 놓는다 —
배치 여럿이면 꼭 주어야 한다. --out 은 링크와 .. 를 풀어 한 경로로 보고(아직 없는 디렉터리를 지나는 .. 도) 확인도
바꿔 넣기도 그 경로에서 한다. 그 경로는 어느 배치 디렉터리(batch.json 이 있는 디렉터리 — 주지 않은 배치도)의 안도
아니어야 하고(배치 하나를 줄 때 그 배치의 실제 디렉터리 바로 아래 dataset 만 예외 — --out 을 주지 않았을 때의 자리.
링크인 <배치>/dataset 은 가리키는 자리로 판단한다), 이미 있으면 앞선 조립의 결과(meta/info.json 이 있는 디렉터리)여야 한다.
조립은 그 자리의 것을 치우고 결과를 놓기 때문이다. 작업 디렉터리 · 병합에 넘기는 뷰 · 바꿔 넣는 동안의 이전 결과는
그 옆에 같은 이름으로 .assembling · .views · .previous 를 붙여 둔다(<배치>/dataset 이면 dataset.assembling …).

병합 전에 청크마다 그 청크가 싣고 온 정본(meta/rules.json · meta/scenarios.json)을 지금 작업 트리의 정본으로
좁힌 것과 대조한다. 라벨은 청크를 만들 때의 정본으로 매겨졌으므로 값이 달라졌으면 거부한다 — 배치를 다시 생성해야
한다. 설명 문장만 바뀐 것이면 --accept-canon-change 로 지금 정본을 싣고 계속한다. 그 플래그는 값의 차이도
받아들이므로, 받아들인 변경이 판정을 바꿨는지는 마지막의 라벨 검증이 본다(판정을 하나도 못 바꾸는 값은 거기서도 안 보인다).
청크의 에피소드가 말하는 로봇의 관절 매핑(meta/joint_map/<로봇>.yaml)도 같은 자리에서 대조한다 — 통째로 싣는
파일이라 바이트로, 그리고 그 차이는 플래그로도 받아들이지 않는다(라벨 검증은 관절 매핑을 읽지 않는다).
청크의 에피소드가 참조하는 재배포 자산(assets/ 의 USD · 텍스처 · 폴더째 싣는 로봇의 파일)도 같은 자리에서 대조한다 — 청크가 실은 사본의
해시(봉인 기록)를 지금 정본이 고정한 해시와. 조립은 그 파일을 작업 트리의 빌드에서 싣고, 변환기와 검증기는 실린
파일을 정본의 해시와만 대조하므로, 빌드와 그 해시를 함께 고친 것은 여기서만 보인다. 그 차이도 플래그로 받아들이지 않는다.
에피소드가 참조하지만 싣지 않는 자산(싣지 않는 로봇의 USD 등)은 실을 파일이 없으므로 그 고정 해시(meta/assets.json 의 usd_sha256)를
청크가 실은 meta/assets.json 의 것과 대조한다 — 배포본이 그 자산에 대해 말하는 것은 그 해시뿐이다. 청크와 지금 사이에 싣는지가
바뀐 자산도 해시로 대조한다(파일은 실리는 이름으로 대조하므로 이름이 같은 다른 자산이 그 자리를 채울 수 있다). 그 차이도 플래그로
받아들이지 않는다.

lerobot 과 PyAV 는 main() 이 넘긴다 (.venv-lerobot). 이 모듈의 나머지는 그 둘 없이 돈다.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation import cameras as camera_canon  # noqa: E402
from generation.assets import ASSETS_DIR, shipped_asset_files  # noqa: E402
from generation.batch_layout import (BATCH_FILE, DATASET_DIR, EPISODES_DIR, Batch, BatchError,  # noqa: E402
                                     ChunkPlan, load_batch, read_seal, seal_problems)
from generation.convert_lerobot import (NARROWED_CANON, assemble_dataset, blank_episode_stats, conversion_order,  # noqa: E402
                                        check_canon_fingerprint, normalise_stats, the_pack)
from generation import release_docs  # noqa: E402
from generation.release_canon import release_canon  # noqa: E402

# Beside the release and named after it (beside): the assembly is made in the first, the merge is handed views of
# the chunks in the second, and the result an assembly replaces is kept under the third until the new one is in
# place. For a batch's own release, <batch>/dataset, they are these three.
BESIDE = ("assembling", "views", "previous")
ASSEMBLING = "dataset.assembling"
VIEWS = "dataset.views"
PREVIOUS = "dataset.previous"
NOT_PER_STEP = ("video", "image")   # features whose statistics count frames sampled, not steps
CHANGES_SHOWN = 12                  # accepted changes named when the label check refuses; the canon check lists all
# LeRobot's own columns of meta/episodes: where an episode's rows and frames sit, and its
# statistics. Every other column is carried from the chunk as it is, and compared.
LAYOUT_PREFIXES = ("stats/", "videos/", "data/", "meta/episodes/")
SHIFTED_COLUMNS = ("episode_index", "dataset_from_index", "dataset_to_index")
LOADER_SAMPLES = 20
LABELS_SHOWN = 5                    # label differences listed in a refusal; the count beside them is of all
# What lerobot's merge does to the episode table's statistics (aggregate.py, update_meta_data): the index
# features' location statistics move by the rows / episodes before; task_index is recomputed from the episode's
# task strings through the merged task table (for an episode of one task, the constant its steps are renumbered
# to -- and its quantiles come out exact, where the chunk's carry the histogram's noise of ~1e-16); nothing else
# is touched. Measured on the two-chunk rehearsal batch (2026-10-03): the moved and the untouched
# statistics of all 20 episodes are exactly the chunk's value (plus the offset) -- difference 0 -- and
# task_index's differ by 4.0e-14 at most.
SHIFTED_STATS = ("index", "episode_index")
SHIFTED_KEYS = ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99")
RELABELED_STATS = "task_index"
RELABELED_ATOL = 1e-9
TASKS = "meta/tasks.parquet"
# The dataset statistics against the step table, each as a fraction of the dimension's largest absolute value.
# Measured on the two-chunk rehearsal batch (20 episodes, 19,099 steps, 2026-10-03), the largest difference a
# correct merge showed: min/max 4.6e-8 (a float32 column against the float64 the statistic was taken from),
# mean 9.8e-6, std 3.4e-3. The std is that far off because LeRobot takes a per-episode std as
# sqrt(E[x^2] - E[x]^2) in float32: for a column that is exactly constant (a fixed camera's pose, x = 8.55 at
# every step) the cancellation leaves a std of 2.6e-2 where the data's is 0, and a mean of 8.54992. That
# inexactness is in the chunks' own statistics -- a one-pass conversion has it too -- so the step table can only
# bound the std, loosely; what pins it is the second comparison, with the pooling of the episodes' statistics
# (POOLED_RTOL). Quantiles are not compared with the step table: LeRobot's
# are count-weighted averages of per-episode histogram estimates, approximate by construction.
EXTREME_RTOL = 1e-6
MEAN_RTOL = 1e-4
STD_RTOL = 5e-2
# The pooling of per-episode statistics is arithmetic on stored float64s: measured 0 on the rehearsal batch.
POOLED_RTOL = 1e-9
# A packet on the edge of a span belongs to the later episode (generation/scene/measure_slice.py share_by_spans).
SPAN_TOLERANCE_S = 1e-6
# An episode's first packet sits at the span's start: the real merge's offsets are exact (the largest
# |first packet - from_timestamp| over the rehearsal batch's chunks and merge, 6 cameras: 5.7e-14 s). A span
# moved earlier by less than a frame still holds the same packets, so the content comparison cannot see it;
# this bound (larger than half a tick of a 1/5000 s time base, far below a 20 ms frame) does.
FIRST_PACKET_TOL_S = 1e-4


# ── 1. the inputs ──

def sealed_rows(record: dict):
    """The rows a seal record says its chunk made, in order: its ``rows``. A record that does not name them
    names its first row (``start_seq``) and its number of ``episodes``: the chunk is one run of the generator,
    and its rows are that range. None when the record says neither."""
    if "rows" in record:
        return record["rows"]
    start, count = record.get("start_seq"), record.get("episodes")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (start, count)):
        return None
    return list(range(start, start + count))


def _no_foreign_chunks(batch: Batch, plans: list[ChunkPlan], prefix: str = "") -> None:
    """Refused: a directory under the batch's chunks/ that is none of its plans' -- named after ``prefix``."""
    wanted = {plan.key for plan in plans}
    there = {p.name for p in (batch.root / "chunks").iterdir()} if (batch.root / "chunks").is_dir() else set()
    if there - wanted:
        raise BatchError(f"배치에 속하지 않는 청크 디렉터리가 있습니다: {sorted(prefix + name for name in there - wanted)} "
                         f"(배치는 {sum(len(rows) for _, rows in batch.parts)}건 · 청크 {batch.chunk_episodes}건)")


def _not_sealed(key: str, chunk: Path) -> str:
    return f"청크 {key} 이 봉인되지 않았습니다 ({chunk}) — run_batch.py 를 이어 돌리십시오"


def _check_sealed(batch: Batch, plan: ChunkPlan, chunk: Path, record: dict, name: str) -> None:
    """What sealed_chunks asks of a sealed chunk, which is called ``name`` in what is said: unchanged since its
    seal, and this batch's chunk -- its scenario (a record that names its rows names its scenario too) and its
    rows (sealed_rows), its first row, its episode ids in the plan's order (the seed in the id is the seq's), and
    the batch's generator commit."""
    changed = seal_problems(chunk, record)
    if changed:
        raise BatchError(f"청크 {name} 이 봉인 뒤에 바뀌었습니다 — " + "; ".join(changed[:5]))
    if ("rows" in record or "scenario" in record) and record.get("scenario") != plan.scenario:
        raise BatchError(f"청크 {name} 의 봉인 기록이 다른 시나리오를 말합니다: {record.get('scenario')!r} — "
                         f"계획은 {plan.scenario}")
    rows = sealed_rows(record)
    if rows != list(plan.rows):
        said = (f"행 {rows}" if "rows" in record else
                f"시작 번호 {record.get('start_seq')!r} · 에피소드 {record.get('episodes')!r}건"
                + (f" (행 {rows[0]}..{rows[-1]})" if rows else ""))
        raise BatchError(f"청크 {name} 의 봉인 기록이 다른 행을 말합니다: {said} — 계획은 행 {list(plan.rows)}")
    if record.get("start_seq") != plan.rows[0]:
        raise BatchError(f"청크 {name} 의 봉인 기록이 다른 시작 번호를 말합니다: {record.get('start_seq')!r}")
    ids = batch.episode_ids(plan)
    if record.get("episode_ids") != ids:
        raise BatchError(f"청크 {name} 의 에피소드가 배치의 것과 다릅니다 — 기대 {len(ids)}건 "
                         f"({ids[0]} …), 봉인 기록 {len(record.get('episode_ids') or [])}건")
    if record.get("generator_commit") != batch.generator_commit:
        raise BatchError(f"청크 {name} 의 generator_commit {record.get('generator_commit')!r} 이 "
                         f"배치의 {batch.generator_commit} 과 다릅니다")


def sealed_chunks(batch: Batch) -> list[tuple[ChunkPlan, Path, dict]]:
    """(plan, chunk directory, seal record) of every chunk the batch has, in the batch's order -- after
    checking each is sealed, unchanged since, and this batch's chunk: its scenario (a record that names its
    rows names its scenario too) and its rows (sealed_rows), its first row, its episode ids in the plan's
    order (the seed in the id is the seq's), and the batch's generator commit (_check_sealed)."""
    plans = batch.chunks()
    _no_foreign_chunks(batch, plans)
    out = []
    for plan in plans:
        chunk = batch.chunk_dir(plan)
        record = read_seal(chunk)
        if record is None:
            raise BatchError(_not_sealed(plan.key, chunk))
        _check_sealed(batch, plan, chunk, record, plan.key)
        out.append((plan, chunk, record))
    return out


NOT_SEALED = "봉인되지 않음"
NO_DIRECTORY = "디렉터리가 없음"
SEVERAL_WITHOUT_OUT = ("배치 여럿을 조립할 때는 배포본을 놓을 자리를 --out 으로 줍니다 — 배치 여럿의 배포본은 어느 한 "
                       f"배치의 것이 아니므로 <배치>/{DATASET_DIR} 에 놓지 않습니다")


@dataclass(frozen=True)
class ReleaseChunks:
    """What release_chunks chose. ``chunks``: (plan, chunk directory, seal record) of every chunk the release holds,
    in its order. ``names``: how each is called in what is said -- <batch directory name>/<key> when there are
    several batches, the key when one. ``skipped``: every chunk passed over, said with why and the rows it planned.
    ``dropped``: the (scenario, row) pairs left out, in order."""
    chunks: tuple
    names: tuple[str, ...]
    skipped: tuple[str, ...]
    dropped: tuple[tuple[str, int], ...]

    def named(self) -> list:
        """``chunks`` with each plan's key the chunk's name (dataclasses.replace), for what names a chunk by its
        plan's key (check_canon) -- not to find a directory by: the directory is the second of each."""
        return [(replace(plan, key=name), chunk, record)
                for (plan, chunk, record), name in zip(self.chunks, self.names)]


def _ranges(rows) -> str:
    """Rows said shortly: runs of consecutive rows as first..last, in ascending order."""
    rows, runs = sorted(rows), []
    for row in rows:
        if runs and row == runs[-1][1] + 1:
            runs[-1][1] = row
        else:
            runs.append([row, row])
    return ", ".join(f"{a}..{b}" if b > a else str(a) for a, b in runs)


def _pairs(pairs) -> str:
    """(scenario, row) pairs as --drop is given them."""
    return ", ".join(f"{scenario}:{row}" for scenario, row in pairs)


def _one_release(batches: list[Batch]) -> None:
    """Refused: batches that do not make one release -- one directory given twice, two of one name (a chunk is
    called <batch directory name>/<key>, and the batches share keys), or a generator commit or a repo id that is
    not the others'."""
    seen: dict[Path, Path] = {}
    for batch in batches:
        real = Path(batch.root).resolve()
        if real in seen:
            raise BatchError(f"같은 배치를 두 번 주었습니다: {seen[real]} · {batch.root} — 한 배치의 청크는 한 번만 실립니다")
        seen[real] = batch.root
    names: dict[str, Path] = {}
    for batch in batches:
        if batch.root.name in names:
            raise BatchError(f"디렉터리 이름이 같은 배치가 둘입니다: {names[batch.root.name]} · {batch.root} — 청크를 "
                             "<배치 디렉터리 이름>/<청크> 로 가르므로 배치의 이름이 서로 달라야 합니다")
        names[batch.root.name] = batch.root
    for field, why in (("generator_commit", "한 배포본의 에피소드는 한 판의 생성 코드로 만든 것이어야 합니다 — 두 판의 "
                                            "코드로 만든 에피소드는 어느 한 판으로 다시 생성해도 모두 재현되지 않습니다"),
                       ("repo_id", "배포본의 리포는 하나입니다")):
        if len({getattr(batch, field) for batch in batches}) > 1:
            raise BatchError(f"배치의 {field} 가 다릅니다 — "
                             + " · ".join(f"{batch.root.name} {getattr(batch, field)}" for batch in batches)
                             + f" — {why}")


def release_chunks(batches, drop=()) -> ReleaseChunks:
    """The chunks a release of ``batches`` -- one, or several made at one generator commit for one repo -- holds:
    the batches' sealed chunks, the batches in the order given and each batch's chunks in its own order.

    The rows the release holds are every (scenario, row) any of the batches plans, less ``drop`` (pairs of a
    scenario and a row). Each of them has to be in exactly one sealed chunk, all batches counted: a row two sealed
    chunks hold is refused, and so is a row no sealed chunk holds -- with the chunks that planned it, one line a
    chunk, and whether each is not sealed or has no directory. A row of ``drop`` has to be one a batch plans (else
    it is a typo), and not in a sealed chunk: a sealed chunk ships whole, and there is no leaving one episode of it
    out. A chunk that is not sealed, or has no directory, is passed over only when every row it plans is in another,
    sealed chunk or in ``drop``.

    Every chunk that ships is held to its batch as sealed_chunks holds it (_check_sealed), and every batch's
    chunks/ to its plans. One batch and no ``drop`` is sealed_chunks itself. With one batch, a chunk that is not
    sealed and whose rows ``drop`` does not all name is refused in sealed_chunks's words: no other batch can cover it.

    Reads the batches and writes nothing; needs no lerobot."""
    batches = list(batches)
    if not batches:
        raise BatchError("조립할 배치가 없습니다")
    drop = sorted({tuple(pair) for pair in drop})
    if len(batches) == 1 and not drop:
        chunks = sealed_chunks(batches[0])
        return ReleaseChunks(tuple(chunks), tuple(plan.key for plan, _, _ in chunks), (), ())
    _one_release(batches)
    several = len(batches) > 1
    entries = []                                    # (batch, plan, chunk directory, seal record or None, name)
    for batch in batches:
        prefix = f"{batch.root.name}/" if several else ""
        plans = batch.chunks()
        _no_foreign_chunks(batch, plans, prefix)
        entries += [(batch, plan, batch.chunk_dir(plan), read_seal(batch.chunk_dir(plan)), prefix + plan.key)
                    for plan in plans]
    planned: set[tuple[str, int]] = set()
    sealed: dict[tuple[str, int], list[str]] = {}   # a row -> the sealed chunks that hold it
    for _, plan, _, record, name in entries:
        for row in plan.rows:
            planned.add((plan.scenario, row))
            if record is not None:
                sealed.setdefault((plan.scenario, row), []).append(name)
    unplanned = [pair for pair in drop if pair not in planned]
    if unplanned:
        raise BatchError(f"--drop 의 행을 어느 배치도 계획하지 않았습니다: {_pairs(unplanned)} — 시나리오와 행을 다시 "
                         f"보십시오 (배치 {' · '.join(batch.root.name for batch in batches)})")
    held = [pair for pair in drop if pair in sealed]
    if held:
        raise BatchError("--drop 의 행이 봉인된 청크에 있습니다: "
                         + ", ".join(f"{_pairs([pair])} ({' · '.join(sealed[pair])})" for pair in held)
                         + " — 봉인된 청크는 통째로 실리고, 청크에서 에피소드 하나를 빼는 길은 없습니다")
    twice: dict[tuple[str, tuple[str, ...]], list[int]] = {}
    for (scenario, row), names in sealed.items():
        if len(names) > 1:
            twice.setdefault((scenario, tuple(names)), []).append(row)
    if twice:
        raise BatchError("같은 행이 봉인된 청크 둘 이상에 있습니다 — 한 행은 한 번만 실립니다:\n  "
                         + "\n  ".join(f"{' · '.join(names)}: {scenario} 행 {_ranges(rows)}"
                                       for (scenario, names), rows in twice.items()))
    dropped = set(drop)
    skipped, uncovered = [], []
    for _, plan, chunk, record, name in entries:
        if record is not None:
            continue
        status = NOT_SEALED if chunk.is_dir() else NO_DIRECTORY
        left = [row for row in plan.rows if (plan.scenario, row) not in sealed and (plan.scenario, row) not in dropped]
        if not left:
            skipped.append(f"{name} ({status} · {plan.scenario} 행 {_ranges(plan.rows)})")
        elif not several:
            raise BatchError(_not_sealed(plan.key, chunk))
        else:
            uncovered.append(f"{name} ({status}" + (f" — {chunk}" if status == NOT_SEALED else "")
                             + f"): {plan.scenario} 행 {_ranges(left)}")
    if uncovered:
        raise BatchError("실릴 행 가운데 어느 배치의 봉인된 청크에도 없는 행이 있습니다 — 그 행을 계획한 청크:\n  "
                         + "\n  ".join(uncovered) + "\n그 청크의 배치를 run_batch.py 로 이어 돌리십시오. 그 배치로 끝낼 수 "
                         "없는 행은 같은 생성 커밋의 다른 배치로 만들어 함께 주고(--batch), 싣지 않기로 한 행은 "
                         "--drop <시나리오>:<행> 으로 이름을 댑니다")
    chunks, names = [], []
    for batch, plan, chunk, record, name in entries:
        if record is not None:
            _check_sealed(batch, plan, chunk, record, name)
            chunks.append((plan, chunk, record))
            names.append(name)
    return ReleaseChunks(tuple(chunks), tuple(names), tuple(skipped), tuple(drop))


def release_dir(batches, out=None) -> Path:
    """Where an assembly of ``batches`` puts the release: ``out`` -- or, for one batch without it, <batch>/dataset,
    a place nothing is asked of. Several batches without ``out`` are refused.

    ``out`` is resolved once (links and ``..``, through a directory that is not there yet as well) and every check
    and the swap use that path, which is what is returned: a path as written can name one place before the merge
    makes the directories beside it and another after. It is refused when it is inside a batch's directory --
    itself or any directory above it holding a batch record (batch.json), a batch given or not: a batch and its
    chunks are what an assembly reads, and a chunk's dataset/ holds meta/info.json as a release does -- but for the
    one place a batch holds a release, ``dataset`` directly under the real root of the one batch given (a
    <batch>/dataset that is a link is a place elsewhere, and judged there); and when it is there and is not an
    earlier assembly's result -- a directory with meta/info.json --, since the assembly clears that place to put
    its result there. The work directory, the merge's views and the result replaced sit beside it (beside)."""
    batches = list(batches)
    if out is None:
        if len(batches) > 1:
            raise BatchError(SEVERAL_WITHOUT_OUT)
        return batches[0].root / DATASET_DIR
    real = Path(out).resolve()
    own = Path(batches[0].root).resolve() / DATASET_DIR if len(batches) == 1 else None
    if real != own:
        for place in (real, *real.parents):
            if (place / BATCH_FILE).is_file():
                given = any(Path(batch.root).resolve() == place for batch in batches)
                raise BatchError(f"--out {out} 이 배치 {place} 의 안입니다({BATCH_FILE} 이 있는 디렉터리 · "
                                 f"{'준' if given else '주지 않은'} 배치) — 배치와 그 청크는 조립이 읽기만 하는 입력이고, "
                                 f"배치 안에 놓는 배포본은 그 배치 하나를 조립한 <배치>/{DATASET_DIR} 뿐입니다")
    if real.exists() and not (real / "meta/info.json").is_file():
        raise BatchError(f"--out {out} 이 이미 있고 앞선 조립의 결과(meta/info.json 이 있는 디렉터리)가 아닙니다 — 조립은 "
                         "그 자리의 것을 치우고 결과를 놓습니다. 아직 없는 자리나 앞선 조립의 결과를 주십시오")
    return real


def beside(out: Path) -> tuple[Path, Path, Path]:
    """The work directory, the merge's views and the place of the result replaced, of an assembly into ``out``:
    beside it, its name with BESIDE's suffixes."""
    out = Path(out)
    work, views, previous = (out.parent / f"{out.name}.{suffix}" for suffix in BESIDE)
    return work, views, previous


def drop_row(text: str) -> tuple[str, int]:
    """--drop's value, <scenario>:<row> (S1f:12): the pair."""
    match = re.fullmatch(r"([A-Za-z0-9_-]+):([0-9]+)", text)
    if not match:
        raise argparse.ArgumentTypeError(f"<시나리오>:<행> 꼴이 아닙니다 (예: S1f:12): {text!r}")
    return match.group(1), int(match.group(2))


# How a difference between two files compared whole is listed: no path inside the file, and a kind
# that is not a sentence. An asset file can also be on one side only.
WHOLE_FILE = "(파일 전체) (바이트)"
ONLY_IN_CHUNK = "(파일 전체) (청크에만 있음)"
ONLY_IN_CANON = "(파일 전체) (지금 정본에만 있음)"
ASSET_CANON = "meta/assets.json"


def referenced_assets(eps: pd.DataFrame) -> list[str]:
    """The asset paths the episodes of a table carry in scene_assets: what decides which asset files a
    distribution of those episodes holds (generation.assets.shipped_asset_files)."""
    return sorted({a["path"] for cell in eps["scene_assets"] for a in json.loads(cell)})


def referenced_pins(assets: object, referenced: list[str]) -> dict[str, tuple[str, bool, dict]]:
    """Path -> (where in meta/assets.json, whether the document lets it ship, its pin) of every asset in
    ``referenced``. The pin is ``usd_sha256`` -- empty when there is none. An asset that does not ship
    (``redistributed`` not true) has no file laid, and what a distribution says of its bytes is the pin.
    The first entry of a path is the one, as generation.assets.asset_entry reads it."""
    out: dict[str, tuple[str, bool, dict]] = {}
    for group in ("robots", "scene"):
        entries = assets.get(group) if isinstance(assets, dict) else None
        for name, entry in (entries.items() if isinstance(entries, dict) else ()):
            if not isinstance(entry, dict) or entry.get("usd_path") not in referenced \
                    or entry["usd_path"] in out:
                continue
            out[entry["usd_path"]] = (f"{group}.{name}", entry.get("redistributed") is True,
                                      {key: entry[key] for key in ("usd_sha256",) if key in entry})
    return out


def sealed_assets(record: dict) -> dict[str, str]:
    """Path in the distribution -> SHA-256 of every asset file a chunk shipped, as its seal recorded
    them: the files under the directory the converter lays the assets in, at any depth -- the textures
    beneath it and a robot's own folder (assets/<dir>/) included."""
    under = f"{DATASET_DIR}/{ASSETS_DIR}/"
    return {name[len(DATASET_DIR) + 1:]: digest for name, digest in record["files"].items()
            if name.startswith(under)}


def joint_map_of(root: Path, model: str) -> Path:
    """A robot's joint map under a repository checkout or under a distribution: where the converter
    reads it and where it lays it, the file as it is (assemble_dataset)."""
    return Path(root) / "meta" / "joint_map" / f"{model}.yaml"


def _identity(item):
    return item.get("rule_id") or item.get("scenario_id") if isinstance(item, dict) else None


# Keys whose string is written for a person to read. Any other string is read by code -- a field
# name, a rule id, a unit, a path, a hash -- and a change of it is a change of value. A key belongs
# here only if no code acts on its text: `robot_task` reads like prose and is not in the list,
# because the converter writes it into every frame as the task.
_PROSE_KEYS = frozenset({
    "note", "description", "editing_policy", "reason", "condition", "objective", "human_behavior",
    "layout", "open_issues", "seed_rule", "common_randomization",
    "unused_in_v0_1", "asset_policy", "why", "what", "how_to_update"})
# Sentences in one place only: under another key the same name holds a value (a camera's
# mount.source is the link it is read from). The whole path, not its ending.
_PROSE_PATHS = frozenset({"bbox.source", "bbox.coords"})
# Tables of meta/safety_schema.json that the validator compares, whole, with its own (C1; the layer
# table is one of them). Every string in them is read by code, whatever its key: a key there is the name
# of a member of `notes` (one member is called `reason`) and the string declares its type, or the name
# of a step field and the string is its layer, and a note is compared like the rest.
_COMPARED_TABLES = frozenset({"episode_json_shapes", "notes_members", "notes_member_shapes", "notes_bounds",
                              "step_field_layers"})


def _prose(key, path: str, shipped: str, wanted: str) -> bool:
    """Under a key written for a person, outside the tables code compares whole, and reading as
    prose: one bare word is not a rewording."""
    if path.split(".", 1)[0].split("[", 1)[0] in _COMPARED_TABLES:
        return False
    named = (isinstance(key, str) and (key.endswith("_note") or key in _PROSE_KEYS)) or path in _PROSE_PATHS
    return named and any(ch.isspace() for ch in shipped + wanted)


def _kind(shipped, wanted, key=None, path: str = "") -> str:
    """What sort of value differs: a number, a sentence, a string that code reads, or something
    else (a list, a type)."""
    def number(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    if number(shipped) or number(wanted):
        return "숫자"
    if isinstance(shipped, str) and isinstance(wanted, str):
        return "문장" if _prose(key, path, shipped, wanted) else "문자열 값"
    return "형 또는 목록"


def canon_changes(shipped, wanted, path: str = "", key=None) -> list[str]:
    """Where two parsed canon documents differ, as "path (kind)" lines. A list of rules or of scenarios is
    matched by ``rule_id`` / ``scenario_id`` and its entries are named by it: rules[R-…].parameters….
    Any other list is read place by place when the two have the same length (open_issues[4] is a
    sentence, not "a list that differs"), and is one value when they do not. ``key`` is the nearest
    dictionary key above the value: with the path it decides whether two differing strings are a
    rewording."""
    here = path or "(문서 전체)"
    if isinstance(shipped, dict) and isinstance(wanted, dict):
        out = []
        for name in sorted(set(shipped) | set(wanted)):
            sub = f"{path}.{name}" if path else str(name)
            if name not in shipped:
                out.append(f"{sub} (지금 정본에만 있음)")
            elif name not in wanted:
                out.append(f"{sub} (청크에만 있음)")
            else:
                out += canon_changes(shipped[name], wanted[name], sub, name)
        return out
    if (isinstance(shipped, list) and isinstance(wanted, list) and shipped and wanted
            and all(_identity(x) for x in shipped + wanted)):
        was, now = {_identity(x): x for x in shipped}, {_identity(x): x for x in wanted}
        out = []
        if len(was) != len(shipped) or len(now) != len(wanted):
            return [f"{here} (같은 id 의 항목이 둘 이상)"]
        for ident in list(dict.fromkeys([*was, *now])):
            sub = f"{path}[{ident}]"
            if ident not in was:
                out.append(f"{sub} (지금 정본에만 있음)")
            elif ident not in now:
                out.append(f"{sub} (청크에만 있음)")
            else:
                out += canon_changes(was[ident], now[ident], sub, key)
        return out
    if isinstance(shipped, list) and isinstance(wanted, list) and len(shipped) == len(wanted):
        out = []
        for place, (was_item, now_item) in enumerate(zip(shipped, wanted)):
            out += canon_changes(was_item, now_item, f"{path}[{place}]", key)
        return out
    if shipped == wanted and isinstance(shipped, bool) == isinstance(wanted, bool):
        return []
    return [f"{here} ({_kind(shipped, wanted, key, path)})"]


def _is_sentence(line: str) -> bool:
    """Whether a line check_canon lists is a reworded sentence: by the kind it ends in, before the
    chunks it names. Looked for anywhere in the line, a key or an id that spells the marker makes
    a changed number count as a sentence."""
    return line.rsplit(" — 청크 ", 1)[0].endswith(" (문장)")


def _value_first(line: str) -> tuple[bool, bool]:
    """The order to read canon differences in: what a label can turn on first -- anything but a
    sentence, and the rules before the scenarios."""
    return _is_sentence(line), not line.startswith("meta/rules.json")


def check_canon(chunks, repo_root: Path, accept: bool, say) -> list[str]:
    """Compare, per chunk, the canon the chunk shipped with what the converter would lay now for that same
    chunk (release_canon over the working tree's two files, narrowed to the chunk's own scenarios and
    triggered rules), as parsed documents. The labels were computed with the canon of the day the chunk was
    made; laid beside labels it did not produce, a changed threshold makes the published rules and the
    published labels disagree -- and C1-C9 pass. Refused unless ``accept``; the differences are returned
    (and said) when accepted. Every difference is listed, with a tally: the list is what decides whether
    the flag may be given, and a line left out could be the one that is not a sentence.

    The joint map of every robot the chunk's episodes name is compared as well, byte for byte: the
    converter lays that file whole from the working tree, and the episodes' joint names were read from
    the one of the day the chunk was made. The file is extracted from the asset, not written by hand, so
    there is no rewording of it to accept; and the assembly's own checks would not see a changed one
    (the label check reads no joint map; the validator, run afterwards, reads its joint names and that
    each has a canonical key, and nothing else of it). A difference is listed and counted with what is
    not a sentence, and refused with ``accept`` as without.

    So are the asset files, by their hashes. The converter lays the redistributable assets a chunk's
    episodes reference out of the builds in the working tree, each held to the hash the canon pins
    (generation.assets.shipped_asset_files), and the validator holds a laid file to the same pin: a
    build changed together with its pin passes both, and is laid beside episodes that were sealed
    with another file. A chunk's seal records the hash of every file its walk of the chunk reaches,
    and sealed_chunks (release_chunks, for every chunk that ships) holds the files to that record, so the
    records are read here and not the files again. What that walk does not reach would not be seen here
    either: it does not go below a link to a directory, nor into a directory it cannot list, and a seal
    refuses both inside the chunk's dataset (batch_layout.sealed_files), so sealed_chunks stops at such a
    chunk before this is asked. ``chunks`` is what release_chunks gave, named (ReleaseChunks.named): a
    chunk is called by its plan's key, its directory is the second of each.
    A file
    pinned to another hash than the chunk sealed, one only the chunk holds and one only the canon
    would lay now are each listed, counted with what is not a sentence, and refused with ``accept``
    as without. How many files were held is said: no file to compare reads like every file equal.
    Each chunk is asked about the assets its own episodes reference, and the lay asks once for all of
    them: that the chunks' assets can ship together is asked here too, before the merge.

    An asset the episodes reference and the canon does not let ship -- a robot kept back -- has no file
    to hold, and what a distribution says of its bytes is its pin in the asset inventory, laid whole
    from the working tree. So that pin is compared with the one in the inventory the chunk shipped (the
    canon of the day the chunk was made), each asset found by its path as the episodes reference it and
    named by the chunk's own entry (by the working tree's when the chunk's inventory has none): a pin
    that differs, one only the chunk's inventory has and one only
    the working tree's has are each listed, counted with what is not a sentence, and refused with
    ``accept`` as without. Only an asset that ships on both sides is left to the file comparison: a file
    is compared by the name it ships under, and when an asset starts or stops shipping another asset
    of that name can fill its place. How many pins of assets that ship on neither side were held is said
    too. What this holds is the inventory to the inventory, not to the USD the generator loaded.

    The working tree is read here and again when the canon is laid, after the merge: a file of it
    changed in between is laid without having been compared.

    The other files laid whole (the schema, the rest of the asset inventory as a document, the camera
    canon outside its recording fingerprint) are not compared here, nor is the joint map of a robot no
    chunk has an episode of or an asset no chunk's episodes reference: the assembly lays none. That the
    working tree's builds are the pinned bytes is the converter's check, made when the assets are laid."""
    repo_root = Path(repo_root)
    docs = [json.loads((repo_root / rel).read_text(encoding="utf-8")) for rel in NARROWED_CANON]
    try:
        assets = json.loads((repo_root / ASSET_CANON).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise BatchError(f"지금 작업 트리의 자산 정본({ASSET_CANON})을 읽지 못합니다 — 청크마다 어느 자산 파일을 "
                         f"실을지 정할 수 없습니다: {e}") from e
    found: dict[str, list[str]] = {}        # a difference of the narrowed canon -> the chunks (keys) it is true of
    maps: dict[str, list[str]] = {}         # a joint map compared byte for byte -> the chunks whose copy differs
    files: dict[str, list[str]] = {}        # an asset file and how it differs -> the chunks it is true of
    pins: dict[str, list[str]] = {}         # the pin of an asset that does not ship, and how it differs -> the chunks
    held = 0                                # asset files a chunk sealed with the hash the canon pins
    held_pins = 0                           # pins of assets that do not ship, the same in a chunk's inventory
    together: set[str] = set()              # every asset path any chunk's episodes reference
    for plan, chunk, record in chunks:
        key = plan.key
        root = chunk / DATASET_DIR
        eps = episodes_table(root)
        scenarios = sorted(set(eps["scenario_id"]))
        triggered = sorted({str(r) for rules in eps["triggered_rules"] for r in list(rules)})
        try:
            wanted = release_canon(*docs, scenarios, triggered)
        except ValueError as e:
            raise BatchError(f"청크 {key}: 지금 정본으로는 이 청크의 시나리오 · 라벨을 싣지 못합니다 — {e}") from e
        for rel, now in zip(NARROWED_CANON, wanted):
            path = root / rel
            if not path.is_file():
                raise BatchError(f"청크 {key} 이 싣고 온 정본이 없습니다: {path}")
            try:
                shipped = json.loads(path.read_text(encoding="utf-8"))
            except ValueError as e:
                raise BatchError(f"청크 {key} 의 {path} 를 읽지 못합니다: {e}") from e
            for line in canon_changes(shipped, now):
                found.setdefault(f"{rel}: {line}", []).append(key)
        for model in sorted(set(eps["robot_model"].astype(str))):
            shipped, now = joint_map_of(root, model), joint_map_of(repo_root, model)
            if not shipped.is_file():
                raise BatchError(f"청크 {key} 이 싣고 온 정본이 없습니다: {shipped}")
            if not now.is_file():
                raise BatchError(f"청크 {key} 의 로봇 {model} 의 관절 매핑이 지금 작업 트리에 없습니다: {now} — "
                                 "조립이 통째로 싣는 파일입니다")
            if shipped.read_bytes() != now.read_bytes():
                maps.setdefault(now.relative_to(repo_root).as_posix(), []).append(key)
        referenced = referenced_assets(eps)
        together.update(referenced)
        try:
            pinned = {rel: f.sha256 for rel, f in shipped_asset_files(assets, referenced, repo_root).items()}
        except (OSError, ValueError) as e:
            raise BatchError(f"청크 {key}: 지금 정본으로는 이 청크의 자산을 싣지 못합니다 — {e}") from e
        sealed = sealed_assets(record)
        for rel in sorted(set(sealed) | set(pinned)):
            if rel not in pinned:
                files.setdefault(f"{rel}: {ONLY_IN_CHUNK}", []).append(key)
            elif rel not in sealed:
                files.setdefault(f"{rel}: {ONLY_IN_CANON}", []).append(key)
            elif sealed[rel] != pinned[rel]:
                files.setdefault(f"{rel}: {WHOLE_FILE}", []).append(key)
            else:
                held += 1
        inventory = root / ASSET_CANON
        if not inventory.is_file():
            raise BatchError(f"청크 {key} 이 싣고 온 정본이 없습니다: {inventory}")
        try:
            was_doc = json.loads(inventory.read_text(encoding="utf-8"))
        except ValueError as e:
            raise BatchError(f"청크 {key} 의 {inventory} 를 읽지 못합니다: {e}") from e
        if not isinstance(was_doc, dict):
            raise BatchError(f"청크 {key} 의 {inventory} 가 JSON 객체가 아닙니다")
        was, now = referenced_pins(was_doc, referenced), referenced_pins(assets, referenced)
        for path in sorted(set(was) | set(now)):
            w_where, w_ships, w_pin = was.get(path, ("", False, {}))
            n_where, n_ships, n_pin = now.get(path, ("", False, {}))
            # Shipped on both sides, its file is compared above, and the file is what the pin pins. A file is
            # compared by the name it ships under, and another asset of that name can ship in its place.
            if w_ships and n_ships:
                continue
            # Named by the chunk's own entry, if it has one: two entries of the chunk are two names.
            changes = canon_changes(w_pin, n_pin, w_where or n_where)
            for line in changes:
                pins.setdefault(f"{ASSET_CANON}: {line}", []).append(key)
            if not changes and n_pin and not w_ships and not n_ships:
                held_pins += 1
    # The lay asks once, for the assets of every chunk. What each chunk ships alone the chunks may not
    # ship together -- two USDs of one file name, one texture name pinned to two hashes -- and the
    # converter would say so only after the merge.
    try:
        shipped_asset_files(assets, sorted(together), repo_root)
    except (OSError, ValueError) as e:
        raise BatchError(f"청크들이 참조하는 자산을 한 배포본에 함께 싣지 못합니다 — {e}") from e
    if not files:
        say(f"자산 대조 — 자산 파일 {held}건: 청크가 실은 사본의 해시가 지금 정본이 고정한 해시와 같음")
    if not pins:
        say(f"자산 대조 — 싣지 않고 참조만 하는 자산의 고정 해시 {held_pins}건: 청크가 실은 {ASSET_CANON} 의 것과 같음")
    if not found and not maps and not files and not pins:
        return []
    lines = [f"{line} — 청크 {', '.join(keys[:3])}" + (f" 외 {len(keys) - 3}" if len(keys) > 3 else "")
             for line, keys in [*found.items(), *((f"{rel}: {WHOLE_FILE}", k) for rel, k in maps.items()),
                                *files.items(), *pins.items()]]
    listed = sorted(lines, key=_value_first)
    values = sum(not _is_sentence(line) for line in lines)
    tally = f"차이 {len(lines)}건 — 문장이 아닌 것 {values}건 · 문장 {len(lines) - values}건"
    if not accept or maps or files or pins:
        advice = []
        if found:
            advice.append("값이 바뀐 것이면 라벨과 배포본의 규칙이 어긋나므로 배치를 다시 생성해야 합니다. 설명 문장만 바뀐 "
                          "것이면 --accept-canon-change 로 지금 정본을 싣고 조립합니다")
        if maps:
            advice.append(f"{' · '.join(maps)} 은 통째로 싣는 파일이라 바이트로 대조하며, 그 차이는 "
                          "--accept-canon-change 로도 받아들이지 않습니다 — 에피소드의 관절 이름은 청크를 만들 때의 그 "
                          "파일에서 읽었습니다. 파일을 청크가 실은 것으로 되돌리거나 배치를 다시 생성하십시오")
        if files:
            advice.append(f"{ASSETS_DIR}/ 의 파일은 청크가 실은 사본의 해시(봉인 기록)를 지금 정본이 고정한 해시"
                          f"({ASSET_CANON} 의 usd_sha256 · bundle.files · ship.files)와 대조하며, 그 차이 {len(files)}건은 "
                          "--accept-canon-change 로도 받아들이지 않습니다 — 조립은 그 해시의 파일을 작업 트리의 빌드에서 "
                          "싣고, 청크는 에피소드를 자기 사본과 함께 싣고 봉인했습니다. 빌드와 정본의 해시를 청크가 실은 "
                          "것으로 되돌리거나 배치를 다시 생성하십시오")
        if pins:
            advice.append(f"에피소드가 참조하는 자산의 고정 해시({ASSET_CANON} 의 usd_sha256)는 청크와 지금 양쪽에서 싣는 "
                          f"자산이 아니면 청크가 실은 {ASSET_CANON} 의 것과 대조하며(싣지 않고 참조만 하는 로봇의 USD 등), "
                          f"그 차이 {len(pins)}건은 --accept-canon-change 로도 받아들이지 않습니다 — 싣지 않는 자산은 배포본이 "
                          "파일 대신 이 해시를 싣고, 청크의 에피소드는 청크를 만들 때의 정본이 고정한 자산으로 생성되었습니다. "
                          "정본의 해시를 청크가 실은 것으로 되돌리거나 배치를 다시 생성하십시오")
        raise BatchError(
            "청크가 싣고 온 정본 · 자산이 지금 작업 트리에서 실을 것과 다릅니다 — 청크의 라벨과 에피소드 메타는 청크를 만들 "
            f"때의 정본으로 만들어졌습니다 ({tally}):\n  " + "\n  ".join(listed) + "\n" + "\n".join(advice))
    say(f"정본 변경을 받아들입니다 (--accept-canon-change) — {tally}. 이 목록의 항목을 지금 정본으로 싣습니다:")
    for line in listed:
        say("  " + line)
    return lines


def episodes_table(root: Path) -> pd.DataFrame:
    files = sorted((Path(root) / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise BatchError(f"{root} 에 에피소드 메타가 없습니다")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) \
        .sort_values("episode_index").reset_index(drop=True)


# ── 2. the merge ──

def merge(datasets: list[Path], out: Path, repo_id: str, aggregate, views: Path) -> None:
    """lerobot's aggregate_datasets over views of the chunk datasets that carry no dataset
    statistics: its aggregate_stats squares the None the converter leaves in a blanked block
    (normalise_stats). The statistics are recomputed afterwards (recompute_stats)."""
    shutil.rmtree(views, ignore_errors=True)
    roots = []
    try:
        for i, root in enumerate(datasets):
            view = views / f"{i:04d}"
            (view / "meta").mkdir(parents=True)
            for rel in ("data", "videos", "meta/episodes", "meta/tasks.parquet"):
                if (root / rel).exists():
                    (view / rel).symlink_to((root / rel).resolve())
            shutil.copy2(root / "meta/info.json", view / "meta/info.json")
            (view / "meta/stats.json").write_text("{}", encoding="utf-8")
            roots.append(view)
        aggregate(repo_ids=[repo_id] * len(roots), aggr_repo_id=repo_id, roots=roots, aggr_root=out)
    finally:
        shutil.rmtree(views, ignore_errors=True)


# ── 3. the step table ──

def declared_features(schema: pa.Schema):
    """What a step file's huggingface metadata says its columns are, or None without it. The rest
    of that metadata is a fingerprint of the writer, which differs from chunk to chunk (measured
    on two real chunks, 2026-10-02) and says nothing about the rows."""
    raw = (schema.metadata or {}).get(b"huggingface")
    if raw is None:
        return None
    try:
        return json.loads(raw).get("info")
    except (ValueError, AttributeError):
        return raw


def task_sentences(path: Path) -> list[str]:
    """The sentences of a task table, in its rows' order: the order a step's task_index counts in -- lerobot's
    loader and its merge take the row at that place (the sentences are the table's pandas index) and neither
    reads the table's task_index column. A table whose task_index column is not its rows' places 0, 1, 2, ...
    is refused: a step's number would read one way to them and another to the column."""
    try:
        table = pd.read_parquet(path)
    except (OSError, ValueError) as e:
        raise BatchError(f"{path} 를 과제 표로 읽지 못합니다: {e}") from e
    if "task_index" not in table.columns or list(table["task_index"]) != list(range(len(table))):
        raise BatchError(f"{path} 의 task_index 열이 행의 자리(0, 1, 2, …)가 아닙니다 — 스텝의 task_index 는 "
                         "표의 행 자리로 읽힙니다")
    return list(table.index)


def task_table(datasets: list[Path]) -> tuple[list[str], list[dict[int, int]]]:
    """The batch's task table and, per chunk, where its own tasks sit in it: every chunk's task sentences in the
    order they first appear, chunk after chunk and within a chunk in its table's order -- the order lerobot
    0.6.1's merge writes its table in (aggregate_datasets: the chunks' tables one after another, each sentence
    once by ``index.unique()``) -- and per chunk a map from a row of its own table (a step's task_index) to the
    row of the batch's. A batch of several scenarios has a task per scenario (meta/scenarios.json robot_task)."""
    sentences: list[str] = []
    maps: list[dict[int, int]] = []
    for root in datasets:
        mapping = {}
        for k, sentence in enumerate(task_sentences(Path(root) / TASKS)):
            if sentence not in sentences:
                sentences.append(sentence)
            mapping[k] = sentences.index(sentence)
        maps.append(mapping)
    return sentences, maps


def check_task_table(datasets: list[Path], out: Path) -> list[str]:
    """The merge's task table is task_table's, row for row; its sentences. The rows rebuilt from the chunks
    (shifted_tables) are numbered in that order, and the merge numbered the episodes' task_index statistics
    after its own table: in another order a step's number would name another sentence."""
    sentences, _ = task_table(datasets)
    merged = task_sentences(Path(out) / TASKS)
    if merged != sentences:
        k = next((k for k, (a, b) in enumerate(zip(merged, sentences)) if a != b), min(len(merged), len(sentences)))

        def at(listed):
            return repr(listed[k]) if k < len(listed) else "없음"

        raise BatchError(f"병합본의 과제 표가 청크 차례대로 처음 나온 과제 문장의 순서와 다릅니다 — 병합본 {len(merged)}행 · "
                         f"청크 {len(sentences)}행, 첫 다른 자리 {k}: 병합본 {at(merged)} ≠ 청크 {at(sentences)}. 다시 쓰는 "
                         "스텝의 task_index 는 청크 차례의 순서로 매기므로 병합본의 표에서는 다른 문장을 가리킵니다 — "
                         "lerobot 의 병합이 과제 표를 짓는 방식이 바뀌었는지 봅니다")
    return sentences


def shifted_tables(datasets: list[Path]) -> pa.Table:
    """The chunks' step tables in order, as one: episode_index and index moved past the chunks
    before, task_index renumbered from the row of its chunk's task table to the row of the batch's
    (task_table), nothing else touched -- the column types, what the huggingface metadata declares and
    every NaN as the converter wrote them. The one table carries the first chunk's schema, its
    metadata included. Read through Arrow: pandas would turn NaN into null on the way back. A step whose
    task_index is no row of its chunk's task table is refused."""
    _, maps = task_table(datasets)
    tables, episodes, rows, first = [], 0, 0, None
    for root, mapping in zip(datasets, maps):
        files = sorted((Path(root) / "data").rglob("*.parquet"))
        if not files:
            raise BatchError(f"{root} 에 스텝 parquet 이 없습니다")
        table = pa.concat_tables([pq.read_table(f) for f in files])
        if first is None:
            first = table.schema
        elif (not table.schema.equals(first, check_metadata=False)
              or declared_features(table.schema) != declared_features(first)):
            raise BatchError(f"{root} 의 스텝 스키마가 첫 청크와 다릅니다 (열의 형 또는 huggingface 메타가 말하는 피처)")
        schema = table.schema
        for name, shift in (("episode_index", episodes), ("index", rows)):
            i = schema.get_field_index(name)
            moved = pc.add(table.column(i), pa.scalar(shift, type=schema.field(i).type))
            table = table.set_column(i, schema.field(i), moved)
        i = schema.get_field_index("task_index")
        if i < 0:
            raise BatchError(f"{root} 의 스텝에 task_index 열이 없습니다")
        lookup = pa.array([mapping[k] for k in range(len(mapping))], type=schema.field(i).type)
        try:
            renumbered = pc.take(lookup, table.column(i))
        except (pa.ArrowIndexError, pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as e:
            raise BatchError(f"{root} 의 스텝 task_index 에 그 청크의 과제 표({len(mapping)}행)의 행이 아닌 값이 있습니다 — "
                             f"{e}") from e
        table = table.set_column(i, schema.field(i), renumbered)
        tables.append(table.cast(first))
        episodes += len(pc.unique(table.column("episode_index")))
        rows += table.num_rows
    return pa.concat_tables(tables)


def rebuild_data(datasets: list[Path], out: Path) -> int:
    """Replace the merge's step files with the chunks' own rows; how many files were written.

    lerobot 0.6.1 writes them through pandas: NaN becomes null, fixed_size_list becomes list and
    the huggingface schema metadata goes -- and LeRobotDataset cannot read the result. The split
    into files is the merge's (meta/episodes names each episode's file and row span); every
    file gets one row group per episode, as LeRobot's own writer leaves it. The rows' task_index is
    renumbered into the batch's task table, which the merge's has to be (check_task_table)."""
    out = Path(out)
    check_task_table(datasets, out)
    full = shifted_tables(datasets)
    index = full.column("index").to_numpy()
    if not np.array_equal(index, np.arange(len(index))):
        raise BatchError("청크를 이은 스텝의 index 가 0 부터 빠짐없이 이어지지 않습니다")
    eps = episodes_table(out)
    spans = [(int(a), int(b)) for a, b in zip(eps["dataset_from_index"], eps["dataset_to_index"])]
    if spans[0][0] != 0 or spans[-1][1] != full.num_rows or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
        raise BatchError("병합본 에피소드 메타의 행 구간이 0 부터 빠짐없이 이어지지 않습니다")
    owner = full.column("episode_index").to_numpy()
    for i, (a, b) in enumerate(spans):
        if b <= a or owner[a] != i or owner[b - 1] != i:
            raise BatchError(f"병합본 에피소드 {i} 의 행 구간 [{a}, {b}) 이 그 에피소드의 스텝이 아닙니다")
    template = json.loads((out / "meta/info.json").read_text(encoding="utf-8"))["data_path"]
    written = []
    for (chunk, file), group in eps.groupby(["data/chunk_index", "data/file_index"], sort=True):
        path = out / template.format(chunk_index=int(chunk), file_index=int(file))
        if not path.is_file():
            raise BatchError(f"병합본에 {path.relative_to(out)} 가 없습니다")
        with pq.ParquetWriter(path, full.schema, compression="snappy") as writer:
            for a, b in zip(group["dataset_from_index"], group["dataset_to_index"]):
                writer.write_table(full.slice(int(a), int(b) - int(a)))
        written.append(path)
    stray = sorted(set((out / "data").rglob("*.parquet")) - set(written))
    if stray:
        raise BatchError(f"병합본에 에피소드 메타가 가리키지 않는 스텝 파일이 있습니다: {[str(p.relative_to(out)) for p in stray]}")
    return len(written)


# ── 4. the statistics ──

def _nested(x):
    """parquet hands nested lists back as object arrays of arrays."""
    if isinstance(x, np.ndarray) and x.dtype == object:
        return np.array([_nested(v) for v in x])
    return np.asarray(x)


def recompute_stats(out: Path, aggregate_stats, write_stats) -> None:
    """meta/stats.json from the merged table's per-episode statistics: what LeRobot does episode
    by episode in a one-pass conversion. The merge's own dataset statistics leave the index
    fields unshifted (episode_index.max of the last chunk, not of the batch). The converter's
    normalise_stats then blanks the fields that hold non-finite values, as it always does --
    and so does blank_episode_stats, in every episode's own statistics. Both are needed here
    though each chunk was blanked already: a chunk was blanked for what *it* holds, and a field
    can be non-finite in one chunk and finite throughout another. The merged data is non-finite
    in it then, and the episodes of the finite chunk would still carry numbers for it, which
    the validator's C8 refuses."""
    out = Path(out)
    eps = episodes_table(out)
    columns = [c for c in eps.columns if c.startswith("stats/")]
    if not columns:
        raise BatchError("병합본 에피소드 메타에 에피소드별 통계(stats/…)가 없습니다")
    per_episode = []
    for _, row in eps.iterrows():
        stats: dict[str, dict] = {}
        for column in columns:
            _, feature, key = column.split("/", 2)
            stats.setdefault(feature, {})[key] = _nested(row[column])
        per_episode.append(stats)
    merged = aggregate_stats(per_episode)
    features = json.loads((out / "meta/info.json").read_text(encoding="utf-8"))["features"]
    order = [n for n in features if n in merged] + [n for n in merged if n not in features]
    write_stats({n: merged[n] for n in order}, out)
    normalise_stats(out)
    blank_episode_stats(out)


# ── 5. the canon, the card and the assets ──

def lay_canon(repo_root: Path, out: Path, sidecars: list[dict]) -> list[str]:
    """The converter's own assemble_dataset on the merged root, from the repository as it is now:
    what the assembly can redo without the lossless video. Refused before anything is written
    when a recording was made under another camera canon (the fingerprint the converter checks),
    or when the canon declares a pack -- packs are numbered by episode index and are not moved."""
    repo_root, out = Path(repo_root), Path(out)
    cameras = camera_canon.load(repo_root / "meta" / "cameras.json")
    if the_pack(cameras) is not None:
        raise BatchError("카메라 정본이 팩을 선언합니다 — 조립은 팩을 옮기지 않습니다 (팩 번호를 다시 매기는 경로가 없음)")
    for side in sidecars:
        try:
            check_canon_fingerprint(side, cameras)
        except (ValueError, KeyError) as e:      # KeyError: a sidecar without media
            raise BatchError(f"{side.get('episode_id')}: 녹화 지문을 지금 카메라 정본과 대조하지 못했습니다 — {e}") from e
    eps = episodes_table(out)
    scenarios = sorted(set(eps["scenario_id"]))
    triggered = sorted({str(r) for rules in eps["triggered_rules"] for r in list(rules)})
    models = sorted(set(eps["robot_model"].astype(str)))
    assets = json.loads((repo_root / ASSET_CANON).read_text(encoding="utf-8"))
    return assemble_dataset(repo_root, out, assets, scenarios, triggered, models,
                            cameras=cameras, packs=None, referenced=referenced_assets(eps))


def lay_release(repo_root: Path, out: Path, sidecars: list[dict]) -> list[str]:
    """lay_canon, and then the documents that go out beside the card (generation.release_docs): the
    card's Korean page, the reference in both languages and the picture. All of them are read from the
    repository as it is now, like the card."""
    copied = lay_canon(repo_root, out, sidecars)
    try:
        return copied + release_docs.lay(repo_root, out)
    except FileNotFoundError as e:
        raise BatchError(str(e)) from e


# ── 6. what the assembly checks about itself ──

def step_nulls(column) -> int:
    """Nulls in a step column, a list's elements included (the validator's c7_step_nulls)."""
    def inner(arr) -> int:
        t = arr.type
        if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
            flat = arr.flatten()
            return flat.null_count + inner(flat)
        return 0
    return column.null_count + sum(inner(c) for c in column.chunks)


def same_array(a: pa.Array, b: pa.Array) -> bool:
    """Equal, with NaN equal to NaN: Arrow's own equals() calls a NaN unequal to itself, so a
    table holding one is not equal to its own copy."""
    if a.type != b.type or len(a) != len(b) or a.null_count != b.null_count:
        return False
    t = a.type
    if pa.types.is_fixed_size_list(t) or pa.types.is_list(t) or pa.types.is_large_list(t):
        if not pa.types.is_fixed_size_list(t) and not pc.list_value_length(a).equals(pc.list_value_length(b)):
            return False
        return same_array(a.flatten(), b.flatten())
    if pa.types.is_floating(t):
        return np.array_equal(a.to_numpy(zero_copy_only=False), b.to_numpy(zero_copy_only=False), equal_nan=True)
    return a.equals(b)


def step_table(root: Path) -> pa.Table:
    """The step files under ``root`` as one table, in file order."""
    files = sorted((Path(root) / "data").rglob("*.parquet"))
    if not files:
        raise BatchError(f"{root} 에 스텝 parquet 이 없습니다")
    return pa.concat_tables([pq.read_table(f) for f in files])


def numeric_matrix(column) -> np.ndarray | None:
    """A numeric step column as float64, one row per step and one column per element (a fixed-size list
    spreads over its width). None for a column that holds no numbers (a string)."""
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    width = 1
    if pa.types.is_fixed_size_list(array.type):
        width, array = array.type.list_size, array.flatten()
    t = array.type
    if not (pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t)):
        return None
    return array.to_numpy(zero_copy_only=False).astype(np.float64).reshape(-1, width)


def nonfinite_features(table: pa.Table) -> set[str]:
    """The numeric columns of a step table that hold a NaN or an infinity."""
    found = set()
    for name in table.column_names:
        matrix = numeric_matrix(table.column(name))
        if matrix is not None and not np.isfinite(matrix).all():
            found.add(name)
    return found


def verify_rows(datasets: list[Path], out: Path) -> None:
    """The merged step table is the chunks' rows: the same schema (types and huggingface
    metadata), no null, and every column equal but for the two moved indices."""
    expected = shifted_tables(datasets)
    actual = step_table(out)
    if not actual.schema.equals(expected.schema, check_metadata=True):
        raise BatchError("병합본 스텝 스키마가 청크와 다릅니다 (열의 형 또는 huggingface 메타) — "
                         "LeRobotDataset 이 읽지 못하는 꼴일 수 있습니다")
    nulls = {n: step_nulls(actual.column(n)) for n in actual.column_names}
    nulls = {n: k for n, k in nulls.items() if k}
    if nulls:
        raise BatchError(f"병합본 스텝 데이터에 null 이 있습니다: {nulls}")
    differ = [n for n in expected.column_names
              if not same_array(expected.column(n).combine_chunks(), actual.column(n).combine_chunks())]
    if differ:
        raise BatchError(f"병합본의 스텝 값이 청크와 다릅니다: {differ[:6]}")


def _cell(v):
    if isinstance(v, np.ndarray):
        return [_cell(x) for x in v]
    if isinstance(v, float) and v != v:
        return "NaN"
    return v


def _array(v) -> np.ndarray:
    return np.asarray(_nested(v), dtype=np.float64)


def _relabeled_statistic(values: np.ndarray, key: str) -> float | None:
    """A statistic of an episode's renumbered task_index values -- min, max, mean, std (of the population) or a
    quantile qNN --, or None for a key the renumbering leaves as it is (the count) and for an episode of no step."""
    if not len(values):
        return None
    if key in ("min", "max", "mean", "std"):
        return float(getattr(np, key)(values))
    if key[:1] == "q" and key[1:].isdigit():
        return float(np.quantile(values, int(key[1:]) / 100))
    return None


def _statistics_problems(source: pd.DataFrame, merged: pd.DataFrame, rows_before, episodes_before,
                         nonfinite: set[str], relabeled: list[np.ndarray]) -> list[str]:
    """Where the merged episode table's ``stats/<feature>/<key>`` columns are not what they must be, given the
    chunks' own: the same columns; the index features' location statistics moved by the rows / episodes before
    the episode's chunk (counted here, from the chunks -- not read from anything the merge wrote); task_index
    the statistics of the episode's renumbered steps (``relabeled``, per merged episode -- the chunk's ``count``),
    to the histogram's noise; a feature whose merged steps hold a non-finite value
    NaN in every key but ``count`` (the converter's blanking, which a chunk whose own steps are finite did not
    apply); every other statistic exactly the chunk's, a NaN equal to a NaN."""
    ours = [c for c in source.columns if c.startswith("stats/")]
    theirs = [c for c in merged.columns if c.startswith("stats/")]
    missing, extra = [c for c in ours if c not in theirs], [c for c in theirs if c not in ours]
    problems = [f"{c} (병합본에 없음)" for c in missing] + [f"{c} (청크에 없는 열)" for c in extra]
    for column in ours:
        if column in missing:
            continue
        _, feature, key = column.split("/", 2)
        for i, (was, got) in enumerate(zip(source[column], merged[column])):
            was, got = _array(was), _array(got)
            if feature in nonfinite and key != "count":
                want, ok = np.full_like(was, np.nan), got.shape == was.shape and bool(np.isnan(got).all())
            elif feature == RELABELED_STATS:
                stat = _relabeled_statistic(relabeled[i], key)
                want = was if stat is None else np.full_like(was, stat)
                ok = got.shape == want.shape and bool(np.allclose(got, want, rtol=0, atol=RELABELED_ATOL))
            else:
                moved = feature in SHIFTED_STATS and key in SHIFTED_KEYS
                want = was + (rows_before[i] if feature == "index" else episodes_before[i]) if moved else was
                ok = got.shape == want.shape and bool(np.array_equal(got, want, equal_nan=True))
            if not ok:
                problems.append(f"{column} (에피소드 {merged['episode_id'].iloc[i]}: 병합본 {got.tolist()} ≠ 기대 {want.tolist()})")
                break
    return problems


def _per_episode(steps: pa.Table, name: str, episodes: int) -> list[np.ndarray]:
    """A step column's values cut by the steps' own episode_index (the steps of an episode are contiguous):
    one array per episode 0..episodes-1, empty for an episode with no step."""
    owner = steps.column("episode_index").to_numpy()
    values = steps.column(name).to_numpy()
    if not len(owner):
        return [values[:0]] * episodes
    cuts = np.flatnonzero(np.diff(owner)) + 1
    groups = dict(zip(owner[np.r_[0, cuts]].tolist(), np.split(values, cuts)))
    return [groups.get(i, values[:0]) for i in range(episodes)]


def _check_episode_tasks(merged: pd.DataFrame, relabeled: list[np.ndarray], sentences: list[str]) -> None:
    """Every merged episode's steps name, in the merged task table ``sentences``, the tasks its ``tasks`` lists."""
    if "tasks" not in merged.columns:
        raise BatchError("병합본 에피소드 메타에 tasks 열이 없습니다")
    for eid, numbers, tasks in zip(merged["episode_id"], relabeled, merged["tasks"]):
        numbers = numbers.tolist()          # a step column with a null comes back as floats with NaN: no row
        outside = [k for k in numbers if not (isinstance(k, int) and 0 <= k < len(sentences))]
        if outside:
            raise BatchError(f"병합본 에피소드 {eid} 의 스텝 task_index {outside[0]!r} 이 병합본 과제 표({len(sentences)}행)의 "
                             "행이 아닙니다")
        named = sorted({sentences[k] for k in numbers})
        own = sorted({str(t) for t in (tasks if tasks is not None else ())})
        if named != own:
            raise BatchError(f"병합본 에피소드 {eid} 의 스텝이 과제 표에서 가리키는 과제 {named} 가 그 에피소드의 tasks "
                             f"{own} 와 다릅니다 — 스텝의 task_index 가 제 청크의 과제가 아닌 문장을 가리킵니다")


def _check_total_tasks(out: Path, sentences: list[str]) -> None:
    """meta/info.json's ``total_tasks`` is the number of rows of the merged task table, as an integer: lerobot
    0.6.1's loader reads the table only when that number is above 0 and counts the tasks by it, and its merge writes
    it from the table it makes."""
    info = json.loads((Path(out) / "meta/info.json").read_text(encoding="utf-8"))
    total = info.get("total_tasks") if isinstance(info, dict) else None
    if isinstance(total, bool) or not isinstance(total, int) or total != len(sentences):
        raise BatchError(f"병합본 meta/info.json 의 total_tasks {total!r} 가 과제 표의 행 수 {len(sentences)} 가 아닙니다 — "
                         "적재기는 이 수가 0 보다 클 때만 과제 표를 읽고 이 수로 과제를 셉니다")


def verify_episodes(datasets: list[Path], out: Path, chunks) -> None:
    """The merged episode table lists the chunks' episodes in order, 0..N-1, and carries every
    column that is not LeRobot's layout or statistics as the chunk had it. ``chunks`` is what
    sealed_chunks or release_chunks gave: the merged ``episode_id`` column has to be the batches' ids chunk after
    chunk, and
    within a chunk in the order the converter wrote them (convert_lerobot.conversion_order -- the seal keeps
    the plan's order, in which a chunk's rows may have been asked; the seed in an id is the seq's), and every
    episode's ``run_start_seq`` the one its chunk's plan gives for its row (ChunkPlan.run_start: for a chunk
    made in one run its first row, for a chunk of condition-table rows the row itself; sealed_chunks and
    release_chunks hold the plan's rows to the seal's) -- not whatever the chunk's own table says.

    Every episode's steps name, in the merged task table, the tasks its own ``tasks`` lists -- carried from its
    chunk, as the comparison of the columns holds: so each step's renumbered task_index names its own chunk's
    task. meta/info.json's ``total_tasks`` is the number of that table's rows (_check_total_tasks).

    The per-episode statistics are compared with the chunks' (``_statistics_problems``): the merge writes
    them, ``recompute_stats`` pools them into meta/stats.json, and nothing downstream would see a wrong one.
    task_index's are those of the merged steps' renumbered numbers."""
    merged = episodes_table(out)
    tables = [episodes_table(root) for root in datasets]
    source = pd.concat(tables, ignore_index=True)
    if len(merged) != len(source) or list(merged["episode_index"]) != list(range(len(source))):
        raise BatchError(f"병합본 에피소드가 청크를 이은 것과 다릅니다: {len(merged)}건 (청크 {len(source)}건)")
    ids, starts = [], []
    for plan, _, record in chunks:
        start_of = {eid: plan.run_start(seq) for seq, eid in zip(plan.rows, record["episode_ids"])}
        for eid in conversion_order(record["episode_ids"]):
            ids.append(eid)
            starts.append(start_of[eid])
    got = [str(v) for v in merged["episode_id"]]
    if got != ids:
        first = next((i for i, (a, b) in enumerate(zip(got, ids)) if a != b), min(len(got), len(ids)))
        raise BatchError(f"병합본의 episode_id 가 배치의 순서(청크 순서대로, 청크 안에서는 변환기가 쓰는 차례로 이은 것)와 "
                         f"다릅니다 — 병합본 {len(got)}건, 배치 {len(ids)}건, 첫 불일치 {first}번째: "
                         f"{got[first:first + 1]} ≠ {ids[first:first + 1]}")
    if "run_start_seq" not in merged.columns:
        raise BatchError("병합본 에피소드 메타에 run_start_seq 열이 없습니다")
    seen = [int(v) for v in merged["run_start_seq"]]
    if seen != starts:
        first = next(i for i, (a, b) in enumerate(zip(seen, starts)) if a != b)
        raise BatchError(f"병합본의 run_start_seq 가 청크의 계획이 주는 값(실행 하나인 청크는 첫 행 · 조건표 청크는 행 "
                         f"자신)과 다릅니다 — 에피소드 {ids[first]}: {seen[first]} ≠ 계획 {starts[first]}")
    carried = [c for c in source.columns if not c.startswith(LAYOUT_PREFIXES) and c not in SHIFTED_COLUMNS]
    gone = [c for c in carried if c not in merged.columns]
    if gone:
        raise BatchError(f"병합본 에피소드 메타에 청크의 열이 없습니다: {gone[:6]}")
    differ = [c for c in carried if [_cell(v) for v in merged[c]] != [_cell(v) for v in source[c]]]
    if differ:
        raise BatchError(f"병합본 에피소드 메타의 값이 청크와 다릅니다: {differ[:6]}")
    steps = step_table(out)
    relabeled = _per_episode(steps, "task_index", len(merged))
    sentences = task_sentences(Path(out) / TASKS)
    _check_episode_tasks(merged, relabeled, sentences)
    _check_total_tasks(out, sentences)
    sizes = [len(t) for t in tables]
    rows_before = np.repeat(np.cumsum([0] + [int(t["length"].sum()) for t in tables[:-1]]), sizes)
    episodes_before = np.repeat(np.cumsum([0] + sizes[:-1]), sizes)
    problems = _statistics_problems(source, merged, rows_before, episodes_before, nonfinite_features(steps), relabeled)
    if problems:
        raise BatchError(f"병합본 에피소드 메타의 통계가 청크의 것(을 옮긴 것)과 다릅니다 — {len(problems)}건: "
                         + "; ".join(problems[:4]))


def _differs(got, want, scale, rtol) -> int | None:
    """The first dimension in which ``got`` is further from ``want`` than ``rtol`` times the dimension's
    ``scale`` (the largest absolute value it takes), or None. A NaN is never close."""
    close = np.abs(got - want) <= rtol * scale
    return None if close.all() else int(np.argmin(close))


def _filled_blank(name: str, block: dict) -> str | None:
    """The first key of a statistics block that carries a value where the converter blanked it (all but
    ``count`` and the reason), or None."""
    return next((k for k, v in block.items() if k not in ("count", "undefined_reason") and v is not None), None)


def _stats_against_steps(stats: dict, features: dict, table: pa.Table) -> None:
    """meta/stats.json against the step table itself, for every per-step numeric feature the file has: a
    feature with a non-finite value must be blanked; any other has its min, max, mean and std within the
    tolerances above of what numpy takes from the rows. The statistic's keys that the file lacks are not asked
    for (the second comparison does)."""
    for name, block in stats.items():
        if (features.get(name) or {}).get("dtype") in NOT_PER_STEP or name not in table.column_names:
            continue
        matrix = numeric_matrix(table.column(name))
        if matrix is None:
            continue
        if not np.isfinite(matrix).all():
            filled = _filled_blank(name, block)
            if filled:
                raise BatchError(f"병합본 통계 {name}.{filled} 에 값이 있습니다 — 이 필드의 스텝에는 비유한값이 있어 "
                                 "통계를 비워야 합니다(변환기의 비움: 분위수가 유한한 채 틀린 값이 됩니다)")
            continue
        scale = np.abs(matrix).max(axis=0)
        for key, rtol, want in (("min", EXTREME_RTOL, matrix.min(axis=0)), ("max", EXTREME_RTOL, matrix.max(axis=0)),
                                ("mean", MEAN_RTOL, matrix.mean(axis=0)), ("std", STD_RTOL, matrix.std(axis=0))):
            if key not in block:
                continue
            try:
                got = np.asarray(block[key], dtype=np.float64)
            except (TypeError, ValueError):
                raise BatchError(f"병합본 통계 {name}.{key} 가 숫자 목록이 아닙니다: {block[key]!r}") from None
            if got.shape != want.shape:
                raise BatchError(f"병합본 통계 {name}.{key} 의 모양 {got.shape} 이 스텝 표의 {want.shape} 와 다릅니다")
            d = _differs(got, want, scale, rtol)
            if d is not None:
                raise BatchError(f"병합본 통계 {name}.{key} 가 스텝 표에서 직접 잰 값과 다릅니다 — 차원 {d}: 파일 {got[d]!r}, "
                                 f"스텝 {want[d]!r} (허용 {rtol * scale[d]:.3g})")


def pooled_statistics(per_episode: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """What LeRobot's aggregate_feature_stats makes of one feature's per-episode statistics (each array has
    the episode on its first axis), written out here again so that the module's own aggregation is not taken
    on trust: the extremes, the counts' sum, the count-weighted mean, the pooled variance (each episode's
    variance plus its mean's squared distance to the pooled mean, count-weighted) and the count-weighted mean
    of every quantile."""
    counts = per_episode["count"]
    total = counts.sum(axis=0)

    def weights(like):
        w = counts
        while w.ndim < like.ndim:
            w = np.expand_dims(w, -1)
        return w

    pooled = {"count": total}
    if "min" in per_episode:
        pooled["min"] = per_episode["min"].min(axis=0)
    if "max" in per_episode:
        pooled["max"] = per_episode["max"].max(axis=0)
    if "mean" in per_episode:
        means = per_episode["mean"]
        pooled["mean"] = (means * weights(means)).sum(axis=0) / total
        if "std" in per_episode:
            spread = per_episode["std"] ** 2 + (means - pooled["mean"]) ** 2
            pooled["std"] = np.sqrt((spread * weights(means)).sum(axis=0) / total)
    for key, values in per_episode.items():
        if key[:1] == "q" and key[1:].isdigit():
            pooled[key] = (values * weights(values)).sum(axis=0) / total
    return pooled


def _stats_against_episodes(stats: dict, eps: pd.DataFrame, nonfinite: set[str]) -> None:
    """meta/stats.json against the pooling (``pooled_statistics``) of the merged episode table's own
    per-episode statistics -- which verify_episodes holds to the chunks' -- for every feature they cover, video
    features included, and every key they hold: nothing dropped, every value within POOLED_RTOL."""
    columns: dict[str, dict[str, str]] = {}
    for c in eps.columns:
        if c.startswith("stats/"):
            _, feature, key = c.split("/", 2)
            columns.setdefault(feature, {})[key] = c
    for feature, keys in columns.items():
        block = stats.get(feature)
        if not isinstance(block, dict):
            raise BatchError(f"병합본 meta/stats.json 에 {feature} 가 없습니다 — 에피소드별 통계에는 있습니다")
        if "count" not in keys:
            raise BatchError(f"병합본 에피소드 메타에 stats/{feature}/count 열이 없습니다 — 통계를 합칠 수 없습니다")
        per_episode = {key: np.stack([_array(v) for v in eps[column]]) for key, column in keys.items()}
        pooled = pooled_statistics(per_episode)
        if feature in nonfinite:
            filled = _filled_blank(feature, block)
            if filled:
                raise BatchError(f"병합본 통계 {feature}.{filled} 에 값이 있습니다 — 이 필드의 스텝에는 비유한값이 있어 비워야 합니다(비유한)")
            if _array(block.get("count")).tolist() != pooled["count"].tolist():
                raise BatchError(f"병합본 통계 {feature}.count = {block.get('count')} — 에피소드별 개수의 합 {pooled['count'].tolist()} 과 다릅니다")
            continue
        for key in keys:
            if key not in block or block[key] is None:
                raise BatchError(f"병합본 통계 {feature}.{key} 가 meta/stats.json 에 없습니다 — 에피소드별 통계에는 있는 키입니다")
        for key, want in pooled.items():
            try:
                got = np.asarray(block[key], dtype=np.float64)
            except (TypeError, ValueError):
                raise BatchError(f"병합본 통계 {feature}.{key} 가 숫자 목록이 아닙니다: {block[key]!r}") from None
            if got.shape != want.shape:
                raise BatchError(f"병합본 통계 {feature}.{key} 의 모양 {got.shape} 이 에피소드별 통계를 합친 {want.shape} 와 다릅니다")
            scale = np.maximum(np.abs(want), np.abs(per_episode[key]).max(axis=0))
            d = _differs(got, want, scale, POOLED_RTOL)
            if d is not None:
                at = np.unravel_index(d, got.shape)
                raise BatchError(f"병합본 통계 {feature}.{key} 가 에피소드별 통계를 합친 값과 다릅니다 — 위치 {tuple(int(i) for i in at)}: "
                                 f"파일 {got[at]!r}, 합친 값 {want[at]!r}")


def verify_stats(out: Path) -> None:
    """The one file this module computes itself, meta/stats.json, against what the tables say. LeRobot's
    own merge left the index fields unshifted (the third defect in the module docstring), and a statistic that is
    wrong in that way is a plausible number: nothing else looks at it. So -- for the features and keys the
    file has -- the largest episode_index is the episode count - 1, the largest index is the step count - 1,
    and every per-step feature (not video) counts exactly the steps. The first two have to be there.

    Then two comparisons that need neither the merge's episode statistics nor the aggregator the assembly was
    handed: with the step table itself (``_stats_against_steps`` -- min, max, mean and std of every feature
    that is finite throughout, to measured tolerances; quantiles not at all), and with this module's own
    pooling of the episode table's per-episode statistics (``_stats_against_episodes`` -- every key, tight)."""
    out = Path(out)
    eps = episodes_table(out)
    episodes, steps = len(eps), int(eps["dataset_to_index"].iloc[-1])
    stats = json.loads((out / "meta/stats.json").read_text(encoding="utf-8"))
    features = json.loads((out / "meta/info.json").read_text(encoding="utf-8"))["features"]

    def first(name: str, key: str):
        block = stats.get(name)
        value = block.get(key) if isinstance(block, dict) else None
        if not isinstance(value, list) or not value:
            raise BatchError(f"병합본 meta/stats.json 에 {name}.{key} 가 없습니다")
        return value[0]

    for name, key, want in (("episode_index", "max", episodes - 1), ("index", "max", steps - 1)):
        got = first(name, key)
        if got != want:
            raise BatchError(f"병합본 통계 {name}.{key} = {got} — 기대 {want}: 번호 필드의 통계가 청크만큼 옮겨지지 않았습니다")
    for name in stats:
        if (features.get(name) or {}).get("dtype") in NOT_PER_STEP:
            continue
        count = first(name, "count")
        if count != steps:
            raise BatchError(f"병합본 통계 {name}.count = {count} — 스텝 수 {steps} 과 다릅니다")
    table = step_table(out)
    _stats_against_steps(stats, features, table)
    _stats_against_episodes(stats, eps, nonfinite_features(table))


def read_packets(path):
    """(presentation time in seconds, payload) of every packet of a video file's first stream, in the order
    PyAV demuxes them. Needs PyAV."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if not packet.size:
                continue
            if packet.pts is None:
                raise BatchError(f"{path} 의 패킷에 표시 시각(pts)이 없습니다 — 에피소드 구간에 배정할 수 없습니다")
            yield float(packet.pts * stream.time_base), bytes(packet)


def video_keys(datasets: list[Path]) -> list[str]:
    """Every camera the chunks have: the video features of their info.json and the directories under videos/."""
    keys = set()
    for root in datasets:
        features = json.loads((Path(root) / "meta/info.json").read_text(encoding="utf-8"))["features"]
        keys |= {n for n, f in features.items() if f.get("dtype") == "video"}
        if (Path(root) / "videos").is_dir():
            keys |= {p.name for p in (Path(root) / "videos").iterdir() if p.is_dir()}
    return sorted(keys)


def _scan_video(root: Path, key: str, table: pd.DataFrame, read, stream) -> tuple[dict, int]:
    """One root's video ``key`` read through its episode table: ({episode_id: [packets, hash of their
    payloads in order, time of the earliest]}, packets that no episode's span holds). Each packet goes to the episode whose
    [from, to) span in the file it sits in holds its time. ``stream`` -- [count, hash] -- is fed every packet
    of the root's files in file order."""
    root = Path(root)
    template = json.loads((root / "meta/info.json").read_text(encoding="utf-8")).get("video_path")
    if not isinstance(template, str):
        raise BatchError(f"{root}/meta/info.json 에 video_path 가 없습니다")
    names = [f"videos/{key}/{part}" for part in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    gone = [c for c in names if c not in table.columns]
    if gone:
        raise BatchError(f"{root} 의 에피소드 메타에 {gone[0]} 열이 없습니다 — {key} 의 영상 구간을 알 수 없습니다")
    spans: dict[Path, list] = {}
    for eid, chunk, file, a, b in zip(table["episode_id"], *(table[c] for c in names)):
        path = root / template.format(video_key=key, chunk_index=int(chunk), file_index=int(file))
        spans.setdefault(path, []).append((float(a), float(b), str(eid)))
    files = sorted((root / "videos" / key).rglob("*.mp4"))
    absent = sorted(set(spans) - set(files))
    if absent:
        raise BatchError(f"{key}: 에피소드 메타가 가리키는 영상 파일이 {root} 에 없습니다 — "
                         f"{[str(p.relative_to(root)) for p in absent[:3]]}")
    per_episode: dict[str, list] = {}
    stray = 0
    for path in files:
        mine = sorted(spans.get(path, []))
        starts = [a for a, _, _ in mine]
        for t, payload in read(path):
            stream[0] += 1
            stream[1].update(payload)
            i = bisect.bisect_right(starts, t + SPAN_TOLERANCE_S) - 1
            if i >= 0 and t < mine[i][1] - SPAN_TOLERANCE_S:
                entry = per_episode.setdefault(mine[i][2], [0, hashlib.sha256(), t])
                entry[0] += 1
                entry[1].update(payload)
                entry[2] = min(entry[2], t)
            else:
                stray += 1
    return per_episode, stray


def verify_packets(datasets: list[Path], out: Path, read) -> dict[str, int]:
    """Per camera, the merged videos are the chunks' videos: the same packets in the same order across the
    files (the count and the hash of the payloads, however the merge split them into files), and -- per
    episode -- the packets inside the span the merged episode table names for that episode (its file, from its
    ``from_timestamp`` up to its ``to_timestamp``) are the packets of that same episode's span in its chunk
    dataset: as many as the episode has steps, with the same payload bytes in order, and none left over that
    no episode's span holds; and the first of them sits at the span's ``from_timestamp`` (FIRST_PACKET_TOL_S: a
    span moved by less than a frame holds the same packets). The stream alone cannot tell which episode a span
    belongs to; a span attached to another episode's file passes it. ``read`` yields (presentation time,
    payload) per packet. Returns the number of packets per camera."""
    out = Path(out)
    merged_eps = episodes_table(out)
    chunk_eps = [(root, episodes_table(root)) for root in datasets]
    counts = {}
    for key in video_keys(datasets):
        source_stream, merged_stream = [0, hashlib.sha256()], [0, hashlib.sha256()]
        source, source_stray = {}, 0
        for root, table in chunk_eps:
            part, stray = _scan_video(root, key, table, read, source_stream)
            source.update(part)
            source_stray += stray
        merged, merged_stray = _scan_video(out, key, merged_eps, read, merged_stream)
        wrong = []
        for eid, length in zip(merged_eps["episode_id"], merged_eps["length"]):
            got, want = merged.get(str(eid)), source.get(str(eid))
            got = (got[0], got[1].hexdigest()) if got else (0, "")
            want = (want[0], want[1].hexdigest()) if want else (0, "")
            if got != want or got[0] != int(length):
                why = (" — 개수는 같고 내용이 다름" if got[0] == want[0] else "") if got != want \
                    else " — 청크의 구간도 같은 패킷인데 에피소드 길이와 개수가 다름"
                wrong.append(f"{eid} (병합본 구간의 패킷 {got[0]}개 · 청크 {want[0]}개 · 길이 {int(length)}{why})")
        if wrong:
            raise BatchError(f"{key} 의 에피소드 {len(wrong)}건에서 병합본이 가리키는 영상 패킷이 그 에피소드의 청크 구간과 다릅니다: "
                             + "; ".join(wrong[:3]))
        late = []
        for eid, start in zip(merged_eps["episode_id"], merged_eps[f"videos/{key}/from_timestamp"]):
            entry = merged.get(str(eid))
            if entry and abs(entry[2] - float(start)) > FIRST_PACKET_TOL_S:
                late.append(f"{eid} (첫 패킷 {entry[2]!r} 초 · from_timestamp {float(start)!r} 초 — "
                            f"{abs(entry[2] - float(start)) * 1000:.3f} ms 어긋남)")
        if late:
            raise BatchError(f"{key} 의 에피소드 {len(late)}건에서 구간의 첫 패킷이 from_timestamp 에 있지 않습니다 "
                             f"(허용 {FIRST_PACKET_TOL_S * 1000:g} ms): " + "; ".join(late[:3]))
        if merged_stray != source_stray:
            raise BatchError(f"{key} 의 영상 패킷 중 어느 에피소드 구간에도 속하지 않는 것이 병합본에 {merged_stray}개 있습니다 "
                             f"(청크는 {source_stray}개)")
        if (merged_stream[0], merged_stream[1].hexdigest()) != (source_stream[0], source_stream[1].hexdigest()):
            raise BatchError(f"{key} 의 영상 패킷이 청크를 이은 것과 다릅니다 — 병합본 {merged_stream[0]}개, 청크 {source_stream[0]}개"
                             + (" (개수는 같고 내용이 다름)" if merged_stream[0] == source_stream[0] else ""))
        counts[key] = merged_stream[0]
    return counts


def _is_picture(x) -> bool:
    shape = tuple(getattr(x, "shape", ()))
    return len(shape) == 3 and math.prod(shape) > 0


def verify_loads(out: Path, repo_id: str, open_dataset) -> int:
    """Read the assembled distribution the way a consumer does: the first and last step of every
    episode and a few at random, each giving back its own index and episode and a picture (an array of three
    dimensions, not empty) for every video feature of meta/info.json. How many were read."""
    out = Path(out)
    eps = episodes_table(out)
    features = json.loads((out / "meta/info.json").read_text(encoding="utf-8"))["features"]
    cameras = [n for n, f in features.items() if f.get("dtype") == "video"]
    dataset = open_dataset(repo_id, out)
    total = int(eps["dataset_to_index"].iloc[-1])
    if len(dataset) != total:
        raise BatchError(f"로더가 본 스텝 수 {len(dataset)} 가 에피소드 메타의 {total} 과 다릅니다")
    owner = {}
    for episode, a, b in zip(eps["episode_index"], eps["dataset_from_index"], eps["dataset_to_index"]):
        owner[int(a)] = owner[int(b) - 1] = int(episode)
    rng = np.random.default_rng(0)
    for i in rng.integers(0, total, LOADER_SAMPLES).tolist():
        owner.setdefault(int(i), None)
    for i in sorted(owner):
        try:
            item = dataset[i]
        except Exception as e:  # noqa: BLE001 -- whatever the loader raises is the finding
            raise BatchError(f"LeRobotDataset 이 스텝 {i} 을 읽지 못했습니다: {type(e).__name__}: {e}") from e
        if int(item["index"]) != i or (owner[i] is not None and int(item["episode_index"]) != owner[i]):
            raise BatchError(f"스텝 {i} 이 다른 스텝으로 읽혔습니다 (index {int(item['index'])}, "
                             f"episode_index {int(item['episode_index'])})")
        for key in cameras:
            if key not in item:
                raise BatchError(f"스텝 {i} 의 항목에 영상 {key} 가 없습니다")
            if not _is_picture(item[key]):
                raise BatchError(f"스텝 {i} 의 영상 {key} 가 그림이 아닙니다 — 모양 {tuple(getattr(item[key], 'shape', ()))}")
    return len(owner)


def verify_labels(out: Path) -> int:
    """Every label the assembled distribution carries follows from it: each episode labelled again from
    the assembled files alone -- the step table as rebuilt, the notes as carried, the canon as laid now --
    and compared with the label it ships (check_release_labels.check: from the numbers as stored, and
    from the float32 a LeRobot item hands out where that differs). How many episodes were labelled.

    This is what stands behind ``accept_canon_change``. That flag waives check_canon's comparison of the
    two narrowed documents, for a sentence as for a value, and a threshold changed under it leaves labels
    made with the old value beside the new one: the validator compares labels with each other, the
    publisher the shipped canon with the working tree's, and both pass. A changed value that moves no
    verdict is not seen here either. Nor is a joint map: no label follows from it, which is why
    check_canon refuses a changed one with the flag as without."""
    # Imported here: the labeller's directory is the one a public split may leave out, and the rest of
    # this module -- and every test of it that brings a stand-in -- stands without it.
    from generation.labeling import check_release_labels

    try:
        report = check_release_labels.check(Path(out))
    except check_release_labels.ReleaseError as e:
        raise BatchError(f"조립한 배포본에서 라벨을 다시 유도할 수 없습니다 — {e}") from e
    if report["agree"] == report["episodes"]:
        return report["episodes"]
    lines = [f"  {r['episode_id']}: 라벨러가 받지 않음 — {r['reason']}" for r in report["refused"]]
    lines += [f"  {d['episode_id']} {d['member']}"
              + (" (적재기가 넘기는 float32 값으로)" if d["read"] == "loader" else "")
              + f": 실린 값 {json.dumps(d['recorded'], ensure_ascii=False)} · "
              f"다시 유도한 값 {json.dumps(d['derived'], ensure_ascii=False)}" for d in report["differences"]]
    more = [f"  … 외 {len(lines) - LABELS_SHOWN}건"] if len(lines) > LABELS_SHOWN else []
    # What it is, not why: whether a canon change was accepted is the assembly's to say (assemble).
    raise BatchError(
        f"실린 라벨이 조립한 배포본에서 다시 나오지 않습니다 — 일치 {report['agree']} / {report['episodes']}:\n"
        + "\n".join(lines[:LABELS_SHOWN] + more) + "\n"
        "라벨은 청크를 만들 때의 정본과 프레임으로 매겨졌습니다. 지금 싣는 정본의 값이 그때와 다르면 정본을 "
        "되돌리거나 배치를 다시 생성하십시오")


# ── the whole ──

def _said_of_the_release(batches: list[Batch], chosen: ReleaseChunks) -> list[str]:
    """The first lines of an assembly of several batches, or with rows dropped: the batches, the chunks passed
    over -- each with why -- and the rows left out."""
    lines = []
    if len(batches) > 1:
        lines.append(f"배치 {len(batches)}개 — {' · '.join(batch.root.name for batch in batches)} "
                     f"(생성 커밋 {batches[0].generator_commit})")
    lines.append(f"건너뛴 청크 {len(chosen.skipped)}개"
                 + (" — 계획한 행이 모두 봉인된 다른 청크에 있거나 --drop 의 행:" if chosen.skipped else ""))
    lines += ["  " + line for line in chosen.skipped]
    lines.append(f"뺀 행 {len(chosen.dropped)}개 (--drop)" + (f": {_pairs(chosen.dropped)}" if chosen.dropped else ""))
    return lines


def assemble(batches, repo_root: Path, *, aggregate, aggregate_stats, write_stats, read_packets,
             open_dataset, lay=lay_release, say=print, accept_canon_change: bool = False,
             check_labels=verify_labels, out=None, drop=()) -> Path:
    """Build the release of ``batches`` -- a Batch, or several made at one generator commit -- from their sealed
    chunks (release_chunks: every row the batches plan but those ``drop`` names, each in one sealed chunk) in
    ``out`` (release_dir: for one batch without it, <batch>/dataset). Made beside it and moved into place once it
    has passed its own checks: a failure leaves the previous result where it was. Refused before
    anything is made when a chunk's canon is not what would be laid now (check_canon), unless
    ``accept_canon_change`` -- which a joint map or an asset file that differs is refused with all
    the same -- and
    then, like any assembly, at the end when a label it carries does not follow from what was
    assembled (``check_labels``)."""
    batches = [batches] if isinstance(batches, Batch) else list(batches)
    final = release_dir(batches, out)
    chosen = release_chunks(batches, drop)
    chunks = list(chosen.chunks)
    datasets = [chunk / DATASET_DIR for _, chunk, _ in chunks]
    sidecars = [json.loads(p.read_text(encoding="utf-8"))
                for _, chunk, _ in chunks for p in sorted((chunk / EPISODES_DIR).glob("*.json"))
                if not p.name.endswith(".labels.json")]
    if len(batches) > 1 or chosen.dropped:
        for line in _said_of_the_release(batches, chosen):
            say(line)
    say(f"청크 {len(chunks)}개 · 에피소드 {len(sidecars)}건 확인")
    changed = check_canon(chosen.named(), repo_root, accept_canon_change, say)
    say(f"정본 대조 — 받아들인 변경 {len(changed)}건" if changed else "정본 대조 — 청크가 싣고 온 정본이 지금 정본과 같음")
    repo_id = batches[0].repo_id
    work, views, previous = beside(final)
    shutil.rmtree(work, ignore_errors=True)
    try:
        merge(datasets, work, repo_id, aggregate, views)
        files = rebuild_data(datasets, work)
        say(f"병합 — 스텝 파일 {files}개를 청크에서 다시 씀")
        recompute_stats(work, aggregate_stats, write_stats)
        copied = lay(repo_root, work, sidecars)
        say(f"정본 · 카드 · 자산 · 문서 {len(copied)}건")
        verify_rows(datasets, work)
        verify_episodes(datasets, work, chunks)
        verify_stats(work)
        counts = verify_packets(datasets, work, read_packets)
        read = verify_loads(work, repo_id, open_dataset)
        try:
            labelled = check_labels(work)
        except BatchError as e:
            if not changed:
                raise
            # In file order and cut at CHANGES_SHOWN, the number that moved the labels of two real
            # chunks sat behind twelve lines of meta/scenarios.json and was not shown (2026-10-07).
            suspects = sorted(changed, key=_value_first)
            more = [f"… 외 {len(suspects) - CHANGES_SHOWN}건"] if len(suspects) > CHANGES_SHOWN else []
            raise BatchError(
                f"{e}\n이 조립은 --accept-canon-change 로 정본 변경 {len(changed)}건을 받아들였습니다 — 그 가운데 "
                "값이 있었다면 그것이 원인입니다(문장이 아닌 것 · meta/rules.json 부터):\n  "
                + "\n  ".join(suspects[:CHANGES_SHOWN] + more)) from e
        say(f"자기 검증 통과 — 카메라 {len(counts)}대의 패킷 {sum(counts.values())}개 · 로더로 {read}개 스텝 · "
            f"조립한 파일만으로 다시 유도한 라벨 {labelled}건")
        swap_in(work, final, previous)
    finally:
        shutil.rmtree(work, ignore_errors=True)       # after a swap there is no ``work`` left to remove
    return final


def swap_in(work: Path, final: Path, previous: Path) -> None:
    """Put ``work`` in place of ``final`` without a moment at which neither result exists: the old one is
    moved aside, the new one renamed in, then the old one removed -- and moved back if the rename fails.
    (Removing the old one first meant an interruption in a tree of tens of GB, or a failed rename, lost
    both.) What an earlier, interrupted removal left under ``previous`` is older than ``final``: cleared first."""
    if final.exists():
        shutil.rmtree(previous, ignore_errors=True)
        os.replace(final, previous)
    try:
        os.replace(work, final)
    except BaseException:
        if previous.exists() and not final.exists():
            os.replace(previous, final)
        raise
    shutil.rmtree(previous, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, type=Path, action="append",
                    help="run_batch.py 가 만든 배치 디렉터리. 여러 번 주면 같은 생성 커밋 · 같은 리포의 배치 여럿을 한 "
                         "배포본으로 묶는다 — 준 차례대로, 배치 안에서는 그 배치의 청크 차례대로 싣고, --out 이 필요하다")
    ap.add_argument("--out", type=Path,
                    help="배포본을 놓을 디렉터리(그 자체가 배포본의 뿌리). 배치 하나면 생략할 수 있고 그때는 <배치>/dataset. "
                         "링크와 .. 를 풀어 한 경로로 보고 그 경로에서 확인하고 바꿔 넣는다. 어느 배치 디렉터리"
                         "(batch.json 이 있는 디렉터리 — 주지 않은 배치도)의 안도 아니어야 하고(배치 하나를 줄 때 그 "
                         "배치의 실제 디렉터리 바로 아래 dataset 만 예외), 이미 있으면 앞선 조립의 결과(meta/info.json 이 "
                         "있는 디렉터리)여야 한다")
    ap.add_argument("--drop", type=drop_row, action="append", metavar="<시나리오>:<행>",
                    help="싣지 않을 행 (여러 번). 어느 배치가 계획한 행이어야 하고 봉인된 청크에 있으면 안 된다 — 봉인된 "
                         "청크는 통째로 실린다. 실릴 행은 배치들이 계획한 행 전부에서 이것을 뺀 것이고, 그 행마다 봉인된 "
                         "청크 정확히 하나에 있어야 한다")
    ap.add_argument("--accept-canon-change", action="store_true",
                    help="청크가 싣고 온 정본이 지금 정본과 달라도 지금 것을 싣고 계속한다. 설명 문장만 바뀐 경우에만 "
                         "맞다 — 값(임계값 등)이 바뀌었으면 라벨이 옛 값으로 매겨졌으므로 배치를 다시 생성한다. "
                         "받아들인 변경이 판정을 바꾸면 조립 끝의 라벨 검증이 거부한다. 관절 매핑"
                         "(meta/joint_map/)과 자산 파일(assets/)의 차이는 이 플래그로도 받아들이지 않는다")
    return ap


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.batch) > 1 and args.out is None:
        parser.error(SEVERAL_WITHOUT_OUT)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")      # a local root: nothing is looked up on the Hub
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import write_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        batches = [load_batch(path) for path in args.batch]
        out = assemble(batches[0] if len(batches) == 1 else batches, REPO, aggregate=aggregate_datasets,
                       aggregate_stats=aggregate_stats, write_stats=write_stats,
                       read_packets=read_packets,
                       open_dataset=lambda repo_id, root: LeRobotDataset(repo_id, root=root),
                       accept_canon_change=args.accept_canon_change, out=args.out, drop=tuple(args.drop or ()))
    except BatchError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"조립 완료: {out}")
    print("다음 — 검증기 전체 (조용한 기계에서, 약 1.5시간):")
    print(f"  .venv/bin/python validator/validate_dataset.py {out} "
          "--repro-hook generation/rerun_seed.sh --repro-sample 8 --verbose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
