"""조립기 — 봉인된 청크를 하나의 배포본으로.

lerobot 0.6.1 의 aggregate_datasets 는 셋을 깨뜨린다: 비운 통계에서 죽고, 데이터 parquet 을
pandas 로 다시 써 NaN 을 null 로 만들고, 번호 통계를 옮기지 않는다. 여기의 가짜 병합은 둘째를 그대로
흉내 낸다 — 조립기가 고치지 않으면 시험이 빨강이다.
"""

import functools
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from generation import assemble_batch as ab
from generation.batch_layout import (DATASET_DIR, EPISODES_DIR, SEAL_FILE, BatchError, load_batch, open_batch,
                                     read_seal, sealed_files, write_seal)
from generation.convert_lerobot import NARROWED_CANON
from generation.release_canon import release_canon
from tests.generation import release_fixture

REPO = Path(__file__).resolve().parents[2]
COMMIT = "0123456789abcdef0123456789abcdef01234567"
DATA = "data/chunk-000/file-000.parquet"
EPISODES = "meta/episodes/chunk-000/file-000.parquet"
SCHEMA = pa.schema([
    ("index", pa.int64()), ("episode_index", pa.int64()), ("frame_index", pa.int64()),
    ("task_index", pa.int64()), ("timestamp", pa.float32()),
    ("safety.zmp", pa.list_(pa.float64(), 2)), ("safety.stability_margin", pa.float64()),
])
FPS = 50
CAMERAS = ("observation.images.head_rgb", "observation.images.cctv_rgb")
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
FEATURES = {**{name: {"dtype": "float64"} for name in SCHEMA.names},
            **{key: {"dtype": "video", "shape": [2, 2, 3]} for key in CAMERAS}}
STOP_RULE = "R-10218-STOP-01"
ROBOT = "unitree_h1"
OTHER_ROBOT = "unitree_g1_29dof"
JOINT_MAP = f"meta/joint_map/{ROBOT}.yaml"
# The builds of the stand-in working tree. A chunk's episodes reference three USDs: the canon lets two
# ship -- the second with a texture, which a scene-format lock pins -- and keeps the third back. They
# reference the robot's USD as well, which does not ship either and is pinned by its hash alone.
ASSET_FILES = {"build/a.usd": b"#usda 1.0\n# a\n", "build/b.usd": b"#usda 1.0\n# b\n",
               "build/textures/wood.jpg": b"wood pixels\n"}
KEPT_BACK = "build/kept.usd"
ROBOT_USD = "{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd"
SCENE_ASSETS = ("build/a.usd", "build/b.usd", KEPT_BACK, ROBOT_USD)
ASSET_CANON = "meta/assets.json"
TEXTURES_LOCK = "generation/scene/textures.lock"


def _repo_canon():
    """The repository's own two narrowed-canon files, parsed: the working tree the assembly reads."""
    return [json.loads((REPO / rel).read_text(encoding="utf-8")) for rel in NARROWED_CANON]


@functools.lru_cache(maxsize=None)
def _narrowed(triggered: tuple, scenario: str = "S1d"):
    """What the converter lays for a chunk of ``scenario`` whose labels point at ``triggered`` -- as it did when
    the chunk was made (the working tree has not changed since)."""
    return release_canon(*_repo_canon(), [scenario], list(triggered))


def _lay_canon_files(root: Path, docs) -> None:
    for rel, doc in zip(NARROWED_CANON, docs):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _lay_joint_map(root: Path, model: str = ROBOT) -> None:
    """The robot's joint map under ``root`` -- a checkout or a chunk's distribution -- as the repository
    holds it: the converter copies the file as it is, and it has not changed since the chunk was made."""
    rel = f"meta/joint_map/{model}.yaml"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / rel, root / rel)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lay_asset_canon(root: Path, files=ASSET_FILES, edit=None, lock=True) -> None:
    """The asset canon under a checkout whose builds hold ``files``: meta/assets.json and the texture lock
    its bundle names, every pin of a file the hash of its bytes. ``edit`` changes the document. Without
    ``lock`` it is the copy a distribution carries: the converter lays the document whole and no lock."""
    doc = {"robots": {ROBOT: {"usd_path": ROBOT_USD, "redistributed": False,
                              "usd_sha256": _sha(b"the robot the episodes were made with\n")}}, "scene": {
        "a": {"usd_path": "build/a.usd", "redistributed": True, "usd_sha256": _sha(files["build/a.usd"])},
        "b": {"usd_path": "build/b.usd", "redistributed": True, "usd_sha256": _sha(files["build/b.usd"]),
              "bundle": {"textures_dir": "build/textures", "textures_lock": TEXTURES_LOCK,
                         "files": [{"file": "wood.jpg", "sha256": _sha(files["build/textures/wood.jpg"]),
                                    "license": "CC0-1.0", "url": "https://example.invalid/wood.jpg"}]}},
        "kept": {"usd_path": KEPT_BACK, "redistributed": False},
    }}
    if edit:
        edit(doc)
    laid = [(ASSET_CANON, json.dumps(doc, ensure_ascii=False, indent=2) + "\n")]
    if lock:
        laid.append((TEXTURES_LOCK, "wood w diffuse CC0 polyhaven https://example.invalid/wood.jpg "
                                    f"{_sha(files['build/textures/wood.jpg'])}\n"))
    for rel, text in laid:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")


def _shipped(scene_assets) -> dict[str, bytes]:
    """assets/ of a chunk whose episodes reference ``scene_assets``, as the converter laid it from the
    builds: each USD the canon lets ship, and the texture of the second beneath it."""
    out = {}
    if "build/a.usd" in scene_assets:
        out["assets/a.usd"] = ASSET_FILES["build/a.usd"]
    if "build/b.usd" in scene_assets:
        out["assets/b.usd"] = ASSET_FILES["build/b.usd"]
        out["assets/textures/wood.jpg"] = ASSET_FILES["build/textures/wood.jpg"]
    return out


def _working_tree(root: Path) -> None:
    """``root`` as a repository checkout, as far as the assembly reads it before it lays anything: the two
    canon files, the joint map of the chunks' robot and the asset canon."""
    _lay_canon_files(root, _repo_canon())
    _lay_joint_map(root)
    _lay_asset_canon(root)


def _hf(fingerprint, features=None):
    """The schema metadata LeRobot's writer leaves: what the columns are, and a fingerprint of the
    writer -- which differs from one chunk to the next (two real chunks, 2026-10-02)."""
    return {b"huggingface": json.dumps({"info": {"features": features or {"timestamp": "float32"}},
                                        "fingerprint": fingerprint}).encode()}


def _episode_stats(name, parts, blank=False):
    """LeRobot's per-episode statistics columns of a scalar feature, ``parts`` being one array of values per
    episode. ``blank``: what the converter leaves for a field with non-finite values -- NaN but the count."""
    out = {}
    for key, fn in (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std), ("q50", np.median)):
        out[f"stats/{name}/{key}"] = [np.array([np.nan if blank else float(fn(np.asarray(p, dtype=np.float64)))])
                                      for p in parts]
    out[f"stats/{name}/count"] = [np.array([len(p)]) for p in parts]
    return out


def _video_packets(key, ids, lengths):
    """One packet per step of a chunk, in step order: (presentation time, payload). The payload says whose it is."""
    return [((sum(lengths[:j]) + f) / FPS, f"{key}:{ids[j]}:{f}".encode())
            for j, k in enumerate(lengths) for f in range(k)]


def _write_video(path: Path, packets) -> None:
    """The stand-in for a video file: one line per packet, ``<time> <payload>``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(repr(float(t)).encode() + b" " + payload + b"\n" for t, payload in packets))


def packets(path):
    """Stands in for the PyAV demux: (presentation time, payload) of every packet of a file, in order."""
    for line in Path(path).read_bytes().split(b"\n"):
        if line:
            t, payload = line.split(b" ", 1)
            yield float(t), payload


def _dataset(root: Path, ids: list[str], lengths: list[int], task="walk", finite_margin=False, triggered=(),
             scene_assets=SCENE_ASSETS, scenario="S1d", run_starts=None):
    """A chunk distribution: the step table as LeRobot writes it (fixed-size lists, huggingface schema
    metadata, NaN and -inf as values), the episode table as the converter leaves it (LeRobot's layout,
    statistics and video-span columns -- the episode's task strings among them -- plus ours), the task table
    with its one task, one video file per camera with a packet per step, and the asset files the converter
    laid for the assets its episodes reference.
    With finite_margin the stability margin holds no -inf, so the converter had nothing to blank: its
    statistics are finite. Every statistic is the true one of the values it describes. ``run_starts``: every
    episode's run_start_seq (None: the first id's seq, a chunk made in one run)."""
    n = sum(lengths)
    zmp = np.full((n, 2), 0.5)
    zmp[0] = np.nan                                        # a step without a support polygon -- an episode's first,
                                                           # which the loader check always reads
    margin = np.linspace(0.1, 0.2, n)
    if not finite_margin:
        margin[2] = -np.inf                                # both feet unloaded
    table = pa.Table.from_arrays([
        pa.array(np.arange(n), pa.int64()),
        pa.array(np.repeat(np.arange(len(lengths)), lengths), pa.int64()),
        pa.array(np.concatenate([np.arange(k) for k in lengths]), pa.int64()),
        pa.array(np.zeros(n, dtype=np.int64)),
        pa.array(np.concatenate([np.arange(k) * 0.02 for k in lengths]).astype(np.float32)),
        pa.FixedSizeListArray.from_arrays(pa.array(zmp.ravel()), 2),
        pa.array(margin),
    ], schema=SCHEMA.with_metadata(_hf(ids[0])))
    (root / "data/chunk-000").mkdir(parents=True)
    pq.write_table(table, root / DATA)
    ends = np.cumsum(lengths)
    starts = ends - lengths
    spans = [slice(int(a), int(b)) for a, b in zip(starts, ends)]
    timestamps = table.column("timestamp").to_numpy()
    columns = {
        "episode_index": range(len(lengths)), "episode_id": ids, "length": lengths,
        "dataset_from_index": starts, "dataset_to_index": ends,
        "data/chunk_index": 0, "data/file_index": 0,
        "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
        "scenario_id": scenario, "robot_model": ROBOT, "outcome": "nominal",
        "tasks": [np.array([task], dtype=object)] * len(lengths),
        "triggered_rules": [np.array(list(triggered), dtype=object)] + [np.array([], dtype=object)] * (len(lengths) - 1),
        "scene_assets": json.dumps([{"path": path} for path in scene_assets]),
        "randomization_seed": [int(i.split("seed")[1]) for i in ids],
        "run_start_seq": int(ids[0].split("_")[1]) if run_starts is None else list(run_starts),
    }
    # The statistics of the number fields, as LeRobot's merge leaves them in the merged table the index fields
    # shifted past the chunks before (measured on two real chunks, 2026-10-02).
    columns.update(_episode_stats("timestamp", [timestamps[s] for s in spans]))
    columns.update(_episode_stats("index", [np.arange(a, b) for a, b in zip(starts, ends)]))
    columns.update(_episode_stats("episode_index", [np.full(k, i) for i, k in enumerate(lengths)]))
    columns.update(_episode_stats("task_index", [np.zeros(k) for k in lengths]))
    columns["stats/task_index/q50"] = [np.array([4e-16]) for _ in lengths]     # the histogram's noise: a quantile of a constant is not the constant
    columns.update(_episode_stats("safety.stability_margin", [margin[s] for s in spans], blank=not finite_margin))
    for key in CAMERAS:
        columns[f"videos/{key}/chunk_index"] = 0
        columns[f"videos/{key}/file_index"] = 0
        columns[f"videos/{key}/from_timestamp"] = starts / FPS
        columns[f"videos/{key}/to_timestamp"] = ends / FPS
    eps = pd.DataFrame(columns)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    eps.to_parquet(root / EPISODES, index=False)
    pd.DataFrame({"task_index": [0]}, index=pd.Index([task], name="task")).to_parquet(root / "meta/tasks.parquet")
    (root / "meta/info.json").write_text(json.dumps({
        "features": FEATURES, "total_episodes": len(lengths), "total_frames": n, "total_tasks": 1,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet", "video_path": VIDEO_PATH}),
        encoding="utf-8")
    _lay_canon_files(root, _narrowed(tuple(sorted(triggered)), scenario))   # the canon the converter shipped with this chunk
    _lay_joint_map(root)
    _lay_asset_canon(root, lock=False)                               # the inventory of the day the chunk was made
    for rel, data in _shipped(scene_assets).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    # The converter's blanked block: what lerobot's aggregate_stats dies on.
    (root / "meta/stats.json").write_text(json.dumps({"safety.stability_margin": (
        {"mean": [float(margin.mean())], "std": [float(margin.std())], "count": [n]} if finite_margin else
        {"mean": None, "std": None, "count": [n], "undefined_reason": "non-finite"})}), encoding="utf-8")
    for key in CAMERAS:
        _write_video(root / VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=0), _video_packets(key, ids, lengths))


def _batch(tmp_path, episodes=25, lengths=(4, 3, 5), finite_margin=(), scene_assets=SCENE_ASSETS, old_seals=False):
    """A batch of three sealed chunks (10 · 10 · 5 episodes); the chunks starting at the seqs in
    finite_margin hold a stability margin without a non-finite value. Every episode references
    ``scene_assets``. The seals are as the runner writes them -- the chunk's scenario and rows among the rest --
    or, with ``old_seals``, records that name neither: the first row and the number of episodes."""
    batch = open_batch(tmp_path / "batch", parts=(("S1d", tuple(range(episodes))),), chunk_episodes=10,
                       generator_commit=COMMIT, repo_id="test/batch")
    _working_tree(tmp_path)
    for plan in batch.chunks():
        start, chunk, ids = plan.rows[0], batch.chunk_dir(plan), batch.episode_ids(plan)
        (chunk / EPISODES_DIR).mkdir(parents=True)
        for eid in ids:
            (chunk / EPISODES_DIR / f"{eid}.json").write_text(json.dumps({"episode_id": eid}), encoding="utf-8")
            (chunk / EPISODES_DIR / f"{eid}.labels.json").write_text("{}", encoding="utf-8")
        _dataset(chunk / DATASET_DIR, ids, [lengths[i % len(lengths)] for i in range(len(ids))],
                 finite_margin=start in finite_margin, triggered=(STOP_RULE,) if start else (),
                 scene_assets=scene_assets)
        record = {"start_seq": start, "episodes": len(ids), "episode_ids": ids, "generator_commit": COMMIT}
        if not old_seals:
            record.update(scenario="S1d", rows=list(plan.rows))
        write_seal(chunk, dict(record, files=sealed_files(chunk)))
    return batch


# A batch of condition-table rows: three scenarios, each with its own task, two rows to a chunk. The chunks
# take turns -- S1e-000 (37, 12), S1f-000 (3, 8), S1g-000 (44), S1e-001 (41) -- so their tasks come a, b, c, a.
# The first chunk's rows are not in ascending order: its distribution holds them as the converter writes them,
# by the recordings' file names -- 12, then 37.
ROW_PARTS = (("S1e", (37, 12, 41)), ("S1f", (3, 8)), ("S1g", (44,)))
TASK_OF = {"S1e": "a", "S1f": "b", "S1g": "c"}


def _rows_batch(tmp_path, parts=ROW_PARTS, lengths=(4, 3, 5), name="batch", commit=COMMIT, repo_id="test/batch"):
    """A sealed batch of condition-table rows: every row a run of the generator of its own, so every episode's
    run_start_seq is its row; every chunk's distribution of its scenario, with the scenario's task, its episodes in
    the order the converter writes them (the names of their recordings, <id>.npz, sorted); the seals as the runner
    writes them -- the chunk's rows and ids in the plan's order. The batch is <tmp_path>/<name>, made at ``commit``
    for ``repo_id``; <tmp_path> is the working tree."""
    batch = open_batch(tmp_path / name, parts=parts, chunk_episodes=2, generator_commit=commit, repo_id=repo_id)
    _working_tree(tmp_path)
    made = 0
    for plan in batch.chunks():
        chunk, ids = batch.chunk_dir(plan), batch.episode_ids(plan)
        (chunk / EPISODES_DIR).mkdir(parents=True)
        for eid in ids:
            (chunk / EPISODES_DIR / f"{eid}.json").write_text(json.dumps({"episode_id": eid}), encoding="utf-8")
            (chunk / EPISODES_DIR / f"{eid}.labels.json").write_text("{}", encoding="utf-8")
        row_of = dict(zip(ids, plan.rows))
        written = sorted(ids, key=lambda eid: f"{eid}.npz")
        _dataset(chunk / DATASET_DIR, written, [lengths[(made + i) % len(lengths)] for i in range(len(ids))],
                 task=TASK_OF[plan.scenario], scenario=plan.scenario, run_starts=[row_of[eid] for eid in written])
        made += len(ids)
        write_seal(chunk, {"scenario": plan.scenario, "rows": list(plan.rows), "start_seq": plan.rows[0],
                           "episodes": len(ids), "episode_ids": ids, "generator_commit": commit,
                           "rows_s": {str(row): 60.0 for row in plan.rows}, "files": sealed_files(chunk)})
    return batch


def _datasets(batch):
    return [batch.chunk_dir(plan) / DATASET_DIR for plan in batch.chunks()]


def _plan(batch, start):
    """The plan of the chunk whose first row is ``start``."""
    (plan,) = [p for p in batch.chunks() if p.rows[0] == start]
    return plan


def _chunk(batch, start):
    return batch.chunk_dir(_plan(batch, start))


def _starts(batch):
    """The first row of every chunk, in the batch's order."""
    return [plan.rows[0] for plan in batch.chunks()]


def fake_aggregate(repo_ids, aggr_repo_id, roots, aggr_root, video_per_file=None, task_order=None):
    """lerobot 0.6.1's aggregate_datasets, as far as this module leans on it: the episode table
    with its indices moved, the videos joined -- and the step table rewritten through pandas.
    ``video_per_file``: episodes to a merged video file (None: all in one), the spans in the episode
    table saying which file and where in it.

    The task table as lerobot's merge makes it (aggregate.py): the chunks' tables one after another, each
    sentence where it first appears (``pd.concat([m.tasks ...]).index.unique()``), and info.json's total_tasks the
    number of its rows (``aggr_meta.info.total_tasks = len(aggr_meta.tasks)``); a step's task_index taken
    from the row of its chunk's table to the row of that one (update_data_df); the episodes' task_index
    statistics recomputed from their task strings through it (update_meta_data). ``task_order`` reorders the
    table -- a merge that orders the tasks otherwise, its steps and statistics numbered after its own table."""
    aggr_root = Path(aggr_root)
    for root in roots:
        assert json.loads((Path(root) / "meta/stats.json").read_text(encoding="utf-8")) == {}, \
            "병합이 죽는 통계(비운 블록)를 그대로 넘겼습니다"
    own = [pd.read_parquet(Path(root) / "meta/tasks.parquet") for root in roots]
    unique = list(pd.concat(own).index.unique())
    tasks = pd.DataFrame({"task_index": range(len(unique))},
                         index=pd.Index(task_order(unique) if task_order else unique, name="task"))
    data, eps, e0, r0 = [], [], 0, 0
    pictures = {key: [] for key in CAMERAS}                 # per merged episode: the payloads of its span
    for root, src in zip(roots, own):
        d, m = pd.read_parquet(Path(root) / DATA), pd.read_parquet(Path(root) / EPISODES)
        d["episode_index"] += e0
        d["index"] += r0
        d["task_index"] = tasks.loc[src.index.take(d["task_index"].to_numpy()), "task_index"].to_numpy()
        m["episode_index"] += e0
        m["dataset_from_index"] += r0
        m["dataset_to_index"] += r0
        for name, shift in (("index", r0), ("episode_index", e0)):
            for key in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
                column = f"stats/{name}/{key}"
                if column in m.columns:
                    m[column] = [v + shift for v in m[column]]
        numbers = [np.array([tasks.loc[t, "task_index"] for t in named], dtype=np.float64) for named in m["tasks"]]
        for key, fn in (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std),
                        ("q50", lambda x: np.quantile(x, 0.5))):          # from the episode's task strings
            m[f"stats/task_index/{key}"] = [np.full_like(v, fn(k)) for v, k in zip(m[f"stats/task_index/{key}"], numbers)]
        for key in CAMERAS:
            there = list(packets(Path(root) / VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=0)))
            for a, b in zip(m[f"videos/{key}/from_timestamp"], m[f"videos/{key}/to_timestamp"]):
                pictures[key].append([p for t, p in there if a - 1e-6 <= t < b - 1e-6])
        data.append(d)
        eps.append(m)
        e0, r0 = e0 + len(m), r0 + len(d)
    eps = pd.concat(eps, ignore_index=True)
    for key, spans in pictures.items():
        per_file = video_per_file or len(spans)
        for k in range(0, len(spans), per_file):
            one, at = [], 0
            for j in range(k, min(k + per_file, len(spans))):
                eps.loc[j, [f"videos/{key}/chunk_index", f"videos/{key}/file_index"]] = [0, k // per_file]
                eps.loc[j, f"videos/{key}/from_timestamp"] = at / FPS
                one += [((at + f) / FPS, p) for f, p in enumerate(spans[j])]
                at += len(spans[j])
                eps.loc[j, f"videos/{key}/to_timestamp"] = at / FPS
            _write_video(aggr_root / VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=k // per_file), one)
    (aggr_root / "data/chunk-000").mkdir(parents=True)
    (aggr_root / "meta/episodes/chunk-000").mkdir(parents=True)
    pd.concat(data, ignore_index=True).to_parquet(aggr_root / DATA)            # NaN leaves as null here
    eps.to_parquet(aggr_root / EPISODES, index=False)
    tasks.to_parquet(aggr_root / "meta/tasks.parquet")
    info = json.loads((Path(roots[0]) / "meta/info.json").read_text(encoding="utf-8"))
    (aggr_root / "meta/info.json").write_text(json.dumps(dict(info, total_episodes=e0, total_frames=r0,
                                                              total_tasks=len(tasks))), encoding="utf-8")
    (aggr_root / "meta/stats.json").write_text("{}", encoding="utf-8")


def fake_aggregate_split(repo_ids, aggr_repo_id, roots, aggr_root, per_file=12):
    """The same merge, but with the step table and the videos in several files as a real 400-episode merge
    leaves them (LeRobot starts a new file when one is full): ``per_file`` episodes to a file, and the episode
    table's data/file_index saying which."""
    fake_aggregate(repo_ids, aggr_repo_id, roots, aggr_root, video_per_file=per_file)
    aggr_root = Path(aggr_root)
    data, eps = pd.read_parquet(aggr_root / DATA), pd.read_parquet(aggr_root / EPISODES)
    eps["data/file_index"] = eps["episode_index"] // per_file
    for k in sorted(set(eps["data/file_index"])):
        path = aggr_root / f"data/chunk-000/file-{k:03d}.parquet"
        data[data["episode_index"] // per_file == k].to_parquet(path)
    eps.to_parquet(aggr_root / EPISODES, index=False)


def fake_stats(per_episode):
    """Stands in for lerobot's aggregate_stats: its parallel algorithm -- the count-weighted mean, the pooled
    variance, the extremes, the counts' sum, the count-weighted mean of every quantile."""
    out = {}
    for name in {n for stats in per_episode for n in stats}:
        counts = np.array([float(s[name]["count"][0]) for s in per_episode])
        means = np.array([float(s[name]["mean"][0]) for s in per_episode])
        mean = float((means * counts).sum() / counts.sum())
        out[name] = {"mean": np.array([mean]), "count": np.array([counts.sum()])}
        if all("std" in s[name] for s in per_episode):
            stds = np.array([float(s[name]["std"][0]) for s in per_episode])
            out[name]["std"] = np.array([float(np.sqrt((((stds ** 2) + (means - mean) ** 2) * counts).sum() / counts.sum()))])
        for key, pick in (("min", min), ("max", max)):
            if all(key in s[name] for s in per_episode):
                out[name][key] = np.array([pick(float(s[name][key][0]) for s in per_episode)])
        if all("q50" in s[name] for s in per_episode):
            q50 = np.array([float(s[name]["q50"][0]) for s in per_episode])
            out[name]["q50"] = np.array([float((q50 * counts).sum() / counts.sum())])
    return out


def fake_write_stats(stats, root):
    (Path(root) / "meta/stats.json").write_text(
        json.dumps({k: {s: np.asarray(v).tolist() for s, v in block.items()} for k, block in stats.items()}),
        encoding="utf-8")


class FakeLoaded:
    def __init__(self, repo_id, root):
        self.rows = pa.concat_tables([pq.read_table(f) for f in sorted((Path(root) / "data").rglob("*.parquet"))])
        features = json.loads((Path(root) / "meta/info.json").read_text(encoding="utf-8"))["features"]
        self.cameras = [n for n, f in features.items() if f["dtype"] == "video"]

    def __len__(self):
        return self.rows.num_rows

    def __getitem__(self, i):
        row = self.rows.slice(i, 1).to_pylist()[0]
        if any(v is None or (isinstance(v, list) and None in v) for v in row.values()):
            raise RuntimeError("Could not infer dtype of NoneType")           # what the real loader says
        return dict(row, **{key: np.zeros((3, 2, 2)) for key in self.cameras})


def _lay(repo_root, out, sidecars):
    (Path(out) / "README.md").write_text(f"card at assembly for {len(sidecars)} episodes", encoding="utf-8")
    return ["README.md"]


# The stand-in chunks carry none of the columns the rules read, so the label check has a stand-in too;
# the real one is tested on a labelled release at the end of this file, and pinned to the assembly there.
FAKES = dict(aggregate=fake_aggregate, aggregate_stats=fake_stats, write_stats=fake_write_stats,
             read_packets=packets, open_dataset=FakeLoaded, lay=_lay, say=lambda s: None,
             check_labels=lambda out: 0)


def _nulls(root):
    table = pa.concat_tables([pq.read_table(f) for f in sorted((Path(root) / "data").rglob("*.parquet"))])
    return {n: ab.step_nulls(table.column(n)) for n in table.column_names if ab.step_nulls(table.column(n))}


# ── the inputs ──

def test_the_chunks_are_checked_against_the_batch_record(tmp_path):
    batch = _batch(tmp_path)
    chunks = ab.sealed_chunks(batch)
    assert [plan.key for plan, _, _ in chunks] == ["0000", "0010", "0020"]
    assert [plan.rows for plan, _, _ in chunks] == [tuple(range(0, 10)), tuple(range(10, 20)), tuple(range(20, 25))]
    assert [len(record["episode_ids"]) for _, _, record in chunks] == [10, 10, 5]      # the short last chunk


def test_a_chunk_that_is_not_sealed_stops_the_assembly(tmp_path):
    batch = _batch(tmp_path)
    (_chunk(batch, 10) / "SEALED.json").unlink()
    with pytest.raises(BatchError, match="청크 0010 이 봉인되지 않았습니다"):
        ab.sealed_chunks(batch)


def test_a_chunk_changed_after_its_seal_stops_the_assembly(tmp_path):
    batch = _batch(tmp_path)
    (_chunk(batch, 10) / DATASET_DIR / "meta/info.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BatchError, match="dataset/meta/info.json"):
        ab.sealed_chunks(batch)


def test_a_chunk_with_a_link_to_a_directory_in_its_dataset_stops_the_assembly(tmp_path):
    # Nothing the seal recorded is gone, added or changed: its walk does not go below the link, and the merge,
    # which opens the chunk's dataset by path, would read what is there.
    batch = _batch(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "file-000.parquet").write_bytes(b"not sealed")
    (_chunk(batch, 10) / DATASET_DIR / "data/chunk-001").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(BatchError, match=f"{DATASET_DIR}/data/chunk-001") as e:
        ab.sealed_chunks(batch)
    assert str(_chunk(batch, 10)) in str(e.value)                 # which chunk


@pytest.mark.parametrize("change, phrase", [
    (lambda r: r.update(generator_commit="f" * 40), "generator_commit"),
    (lambda r: r.update(start_seq=0), "시작 번호"),
    (lambda r: r.update(episode_ids=r["episode_ids"][:-1]), "에피소드"),
    (lambda r: r.update(episode_ids=list(reversed(r["episode_ids"]))), "에피소드"),
], ids=["another commit", "another start", "one episode short", "out of order"])
def test_a_seal_that_is_not_this_batchs_chunk_stops_the_assembly(tmp_path, change, phrase):
    batch = _batch(tmp_path)
    record = read_seal(_chunk(batch, 10))
    change(record)
    write_seal(_chunk(batch, 10), record)
    with pytest.raises(BatchError, match=phrase):
        ab.sealed_chunks(batch)


def test_a_chunk_the_batch_does_not_have_stops_the_assembly(tmp_path):
    batch = _batch(tmp_path)
    shutil.copytree(_chunk(batch, 20), batch.root / "chunks" / "0030")
    with pytest.raises(BatchError, match="0030"):
        ab.sealed_chunks(batch)


def test_old_seals_are_read_as_their_range(tmp_path):
    # A record that names no rows names its first row and its number of episodes: the chunk is one run of the
    # generator, and its rows are that range.
    batch = _batch(tmp_path, old_seals=True)
    assert not any({"rows", "scenario"} & set(read_seal(batch.chunk_dir(plan))) for plan in batch.chunks())
    chunks = ab.sealed_chunks(batch)
    assert [plan.rows for plan, _, _ in chunks] == [tuple(range(0, 10)), tuple(range(10, 20)), tuple(range(20, 25))]
    record = read_seal(_chunk(batch, 20))
    record["episodes"] = 4                                  # rows 20..23: one short of the plan's
    write_seal(_chunk(batch, 20), record)
    with pytest.raises(BatchError, match="청크 0020 의 봉인 기록이 다른 행을 말합니다") as e:
        ab.sealed_chunks(batch)
    assert "23" in str(e.value) and "24" in str(e.value)       # what the record says and what the plan says


@pytest.mark.parametrize("change, phrase", [
    (lambda r: r.update(rows=[37, 13]), "다른 행"),
    (lambda r: r.update(rows=[12, 37]), "다른 행"),
    (lambda r: r.update(rows=[37]), "다른 행"),
    (lambda r: r.pop("rows"), "다른 행"),                   # then read as the range 37, 38 of a chunk made in one run
    (lambda r: r.update(scenario="S1f"), "다른 시나리오"),
    (lambda r: r.pop("scenario"), "다른 시나리오"),
], ids=["another row", "the rows in another order", "one row short", "no rows", "another scenario", "no scenario"])
def test_a_seal_for_other_rows_is_refused(tmp_path, change, phrase):
    # The ids it names are this chunk's all the while: the rows and the scenario are held on their own.
    batch = _rows_batch(tmp_path)
    plan = batch.chunks()[0]
    assert (plan.key, plan.rows) == ("S1e-000", (37, 12))
    ab.sealed_chunks(batch)
    record = read_seal(batch.chunk_dir(plan))
    change(record)
    write_seal(batch.chunk_dir(plan), record)
    with pytest.raises(BatchError, match=f"청크 S1e-000 의 봉인 기록이 {phrase}"):
        ab.sealed_chunks(batch)


# ── the merge and what it breaks ──

def test_the_merge_is_handed_views_without_dataset_statistics_and_they_are_removed(tmp_path):
    batch = _batch(tmp_path)
    out, views = tmp_path / "merged", tmp_path / "views"
    ab.merge(_datasets(batch), out, "test/batch", fake_aggregate, views)       # the fake asserts the empty stats
    assert (out / EPISODES).is_file() and not views.exists()
    source = json.loads((_datasets(batch)[0] / "meta/stats.json").read_text(encoding="utf-8"))
    assert "safety.stability_margin" in source, "청크의 통계를 건드렸습니다"


def test_the_step_table_is_rebuilt_from_the_chunks_not_taken_from_the_merge(tmp_path):
    batch = _batch(tmp_path)
    out = tmp_path / "merged"
    ab.merge(_datasets(batch), out, "test/batch", fake_aggregate, tmp_path / "views")
    assert _nulls(out), "가짜 병합이 결함을 흉내 내지 못했습니다 — 이 시험이 아무것도 증명하지 않습니다"
    assert ab.rebuild_data(_datasets(batch), out) == 1
    assert _nulls(out) == {}
    table = pq.read_table(out / DATA)
    # The chunk's own schema as parquet hands it back (the list's child comes back named "element").
    assert table.schema.equals(pq.read_table(_datasets(batch)[0] / DATA).schema, check_metadata=True)
    assert pa.types.is_fixed_size_list(table.schema.field("safety.zmp").type)
    assert b"huggingface" in table.schema.metadata
    assert table.column("index").to_pylist() == list(range(table.num_rows))
    lengths = [4, 3, 5, 4, 3, 5, 4, 3, 5, 4] * 2 + [4, 3, 5, 4, 3]
    assert table.column("episode_index").to_pylist() == [e for e, k in enumerate(lengths) for _ in range(k)]
    assert pq.ParquetFile(out / DATA).metadata.num_row_groups == 25             # one row group per episode
    zmp = np.array(table.column("safety.zmp").to_pylist(), dtype=float)
    assert np.isnan(zmp).all(axis=1).sum() == 3 and np.isneginf(table.column("safety.stability_margin").to_numpy()).sum() == 3


def test_the_one_table_carries_the_first_chunks_schema_though_the_writers_fingerprints_differ(tmp_path):
    batch = _batch(tmp_path)
    datasets = _datasets(batch)
    prints = [json.loads(pq.read_schema(d / DATA).metadata[b"huggingface"])["fingerprint"] for d in datasets]
    assert len(set(prints)) == 3, "청크마다 지문이 다른 실물의 꼴이 아닙니다"
    full = ab.shifted_tables(datasets)
    assert json.loads(full.schema.metadata[b"huggingface"])["fingerprint"] == prints[0]


@pytest.mark.parametrize("change", [
    lambda t: t.replace_schema_metadata(_hf("x", features={"another": "feature"})),
    lambda t: t.replace_schema_metadata(None),
    lambda t: t.set_column(t.schema.get_field_index("timestamp"), "timestamp",
                           t.column("timestamp").cast(pa.float64())),
], ids=["other features declared", "no huggingface metadata", "another column type"])
def test_chunks_whose_step_files_are_not_of_one_kind_are_refused(tmp_path, change):
    batch = _batch(tmp_path)
    datasets = _datasets(batch)
    pq.write_table(change(pq.read_table(datasets[1] / DATA)), datasets[1] / DATA)
    with pytest.raises(BatchError, match="스키마"):
        ab.shifted_tables(datasets)


def _point_a_step_at(row):
    """A chunk whose sixth step names ``row`` of its task table."""
    def go(root):
        table = pq.read_table(root / DATA)
        i = table.schema.get_field_index("task_index")
        values = table.column(i).to_pylist()
        values[5] = row
        pq.write_table(table.set_column(i, table.schema.field(i), pa.array(values, pa.int64())), root / DATA)
    return go


def _number_the_task_otherwise(root):
    """A chunk whose task table's task_index column is not its row's place: lerobot reads a step's number as
    the row, the column says another."""
    pd.DataFrame({"task_index": [7]}, index=pd.Index(["walk"], name="task")).to_parquet(root / "meta/tasks.parquet")


@pytest.mark.parametrize("change, phrase", [
    (_point_a_step_at(1), "과제 표\\(1행\\)의 행이 아닌"),
    (_point_a_step_at(-1), "과제 표\\(1행\\)의 행이 아닌"),
    (_number_the_task_otherwise, "task_index 열이 행의 자리"),
], ids=["a row past the table", "a negative row", "a column that is not the rows' places"])
def test_a_chunk_whose_steps_cannot_be_read_against_its_task_table_is_refused(tmp_path, change, phrase):
    # Chunks of other tasks are renumbered into one table (below); a step's number that is no row of its own
    # chunk's table, or a table whose numbers are not its rows, cannot be.
    batch = _batch(tmp_path)
    datasets = _datasets(batch)
    change(datasets[1])
    with pytest.raises(BatchError, match=phrase) as e:
        ab.shifted_tables(datasets)
    assert str(datasets[1]) in str(e.value)


def test_the_statistics_are_recomputed_from_the_merged_episodes(tmp_path):
    batch = _batch(tmp_path)
    out = tmp_path / "merged"
    ab.merge(_datasets(batch), out, "test/batch", fake_aggregate, tmp_path / "views")
    ab.rebuild_data(_datasets(batch), out)
    ab.recompute_stats(out, fake_stats, fake_write_stats)
    stats = json.loads((out / "meta/stats.json").read_text(encoding="utf-8"))
    lengths = np.array([4, 3, 5, 4, 3, 5, 4, 3, 5, 4] * 2 + [4, 3, 5, 4, 3], dtype=float)
    assert stats["timestamp"]["count"] == [lengths.sum()]
    assert stats["timestamp"]["mean"] == pytest.approx([float((0.01 * (lengths - 1) * lengths).sum() / lengths.sum())])
    # The field with non-finite values comes out blanked, as a one-pass conversion leaves it.
    assert stats["safety.stability_margin"]["mean"] is None
    assert "undefined_reason" in stats["safety.stability_margin"]
    assert list(stats) == ["index", "episode_index", "task_index", "timestamp", "safety.stability_margin"]    # the feature order of info.json


def test_a_field_that_is_non_finite_in_only_some_chunks_is_blank_in_every_episode(tmp_path):
    """The chunks were each blanked on their own content. The merged data is non-finite in the field
    as soon as one chunk is, and then the converter's rule -- the validator's C8 -- is that no episode
    carries a statistic for it, the finite chunk's included."""
    batch = _batch(tmp_path, finite_margin=(0,))
    datasets = _datasets(batch)
    first = ab.episodes_table(datasets[0])
    assert np.isfinite(np.concatenate(first["stats/safety.stability_margin/mean"].to_list())).all(), \
        "the first chunk is not the finite one -- this test proves nothing"
    out = tmp_path / "merged"
    ab.merge(datasets, out, "test/batch", fake_aggregate, tmp_path / "views")
    ab.rebuild_data(datasets, out)
    ab.recompute_stats(out, fake_stats, fake_write_stats)
    margin = pq.read_table(out / DATA).column("safety.stability_margin").to_numpy()
    assert np.isneginf(margin).sum() == 2 and np.isfinite(margin[:int(first["length"].sum())]).all()   # non-finite only in the later chunks
    eps = ab.episodes_table(out)
    ours = [c for c in eps.columns if c.startswith("stats/safety.stability_margin/")]
    assert sorted(ours) == sorted(f"stats/safety.stability_margin/{k}" for k in ("count", "max", "mean", "min", "q50", "std"))
    mean = np.concatenate(eps["stats/safety.stability_margin/mean"].to_list())
    assert len(mean) == 25 and np.isnan(mean).all()                               # every episode, not only the later chunks'
    assert np.concatenate(eps["stats/safety.stability_margin/count"].to_list()).tolist() == eps["length"].tolist()
    assert np.isfinite(np.concatenate(eps["stats/timestamp/mean"].to_list())).all()       # a field that is finite throughout keeps its numbers


# ── what the assembly checks about itself ──

def _merged(tmp_path):
    batch = _batch(tmp_path)
    out = tmp_path / "merged"
    ab.merge(_datasets(batch), out, "test/batch", fake_aggregate, tmp_path / "views")
    ab.rebuild_data(_datasets(batch), out)
    return batch, out


def test_a_merge_that_was_repaired_passes_its_own_checks(tmp_path):
    batch, out = _merged(tmp_path)
    ab.verify_rows(_datasets(batch), out)
    ab.verify_episodes(_datasets(batch), out, ab.sealed_chunks(batch))
    assert ab.verify_packets(_datasets(batch), out, packets) == {
        "observation.images.cctv_rgb": 99, "observation.images.head_rgb": 99}       # a packet per step
    assert ab.verify_loads(out, "test/batch", FakeLoaded) > 0


def test_the_merge_as_lerobot_leaves_it_does_not_pass(tmp_path):
    batch = _batch(tmp_path)
    out = tmp_path / "merged"
    ab.merge(_datasets(batch), out, "test/batch", fake_aggregate, tmp_path / "views")      # not rebuilt
    with pytest.raises(BatchError, match="스키마|null"):
        ab.verify_rows(_datasets(batch), out)
    with pytest.raises(BatchError, match="읽지 못했습니다"):
        ab.verify_loads(out, "test/batch", FakeLoaded)


def test_one_changed_step_value_is_caught(tmp_path):
    batch, out = _merged(tmp_path)
    table = pq.read_table(out / DATA)
    i = table.schema.get_field_index("timestamp")
    values = table.column(i).to_pylist()
    values[7] += 1.0
    pq.write_table(table.set_column(i, table.schema.field(i), pa.array(values, pa.float32())), out / DATA)
    with pytest.raises(BatchError, match="timestamp"):
        ab.verify_rows(_datasets(batch), out)


def test_a_carried_episode_column_that_changed_is_caught(tmp_path):
    batch, out = _merged(tmp_path)
    eps = pd.read_parquet(out / EPISODES)
    eps.loc[12, "outcome"] = "fail"
    eps.to_parquet(out / EPISODES, index=False)
    with pytest.raises(BatchError, match="outcome"):
        ab.verify_episodes(_datasets(batch), out, ab.sealed_chunks(batch))


@pytest.mark.parametrize("change, phrase", [
    (lambda f: f.write_bytes(b"\n".join(reversed(f.read_bytes().rstrip(b"\n").split(b"\n"))) + b"\n"), "패킷"),
    (lambda f: f.write_bytes(b"\n".join(f.read_bytes().rstrip(b"\n").split(b"\n")[:-1]) + b"\n"), "패킷"),
], ids=["reordered", "one packet short"])
def test_video_packets_that_are_not_the_chunks_in_order_are_caught(tmp_path, change, phrase):
    batch, out = _merged(tmp_path)
    change(out / "videos/observation.images.cctv_rgb/chunk-000/file-000.mp4")
    with pytest.raises(BatchError, match=phrase):
        ab.verify_packets(_datasets(batch), out, packets)


def test_a_camera_the_merge_dropped_is_caught(tmp_path):
    batch, out = _merged(tmp_path)
    shutil.rmtree(out / "videos/observation.images.cctv_rgb")
    with pytest.raises(BatchError, match="observation.images.cctv_rgb"):
        ab.verify_packets(_datasets(batch), out, packets)


# ── what the merge did to the episode table's statistics and video spans ──

def _edit_episodes(edit):
    """What a faulty merge would have left in a merged root: ``edit`` changes the episode table in place."""
    def go(root):
        eps = pd.read_parquet(Path(root) / EPISODES)
        edit(eps)
        eps.to_parquet(Path(root) / EPISODES, index=False)
    return go


def _set(column, value):
    return lambda eps: eps.__setitem__(column, [np.array([value])] * len(eps))


def _from_second_chunk_on(column, change):
    """``change`` applied to the cells of the episodes after the first chunk (episode 10 on)."""
    return lambda eps: eps.__setitem__(column, [v if i < 10 else change(v) for i, v in enumerate(eps[column])])


@pytest.mark.parametrize("edit, column", [
    (_set("stats/index/mean", -12345.0), "stats/index/mean"),                      # every episode's mean overwritten
    (_set("stats/timestamp/mean", 100000.0), "stats/timestamp/mean"),               # ... and another column's
    (_from_second_chunk_on("stats/index/mean", lambda v: v - 40), "stats/index/mean"),         # not moved past the first chunk
    (_from_second_chunk_on("stats/index/q50", lambda v: v - 40), "stats/index/q50"),
    (_from_second_chunk_on("stats/episode_index/max", lambda v: v - 10), "stats/episode_index/max"),
    (_from_second_chunk_on("stats/index/std", lambda v: v + 1.0), "stats/index/std"),          # moved though nothing moves a spread
    (_from_second_chunk_on("stats/index/count", lambda v: v + 1), "stats/index/count"),
    (_from_second_chunk_on("stats/timestamp/std", lambda v: v * 2), "stats/timestamp/std"),    # an unmoved field changed
    (_from_second_chunk_on("stats/timestamp/q50", lambda v: v + 0.5), "stats/timestamp/q50"),
    (_set("stats/task_index/mean", 1.0), "stats/task_index/mean"),
    (_set("stats/safety.stability_margin/mean", 0.15), "stats/safety.stability_margin/mean"),     # a blanked field given a number back
    (lambda eps: eps.pop("stats/timestamp/std"), "stats/timestamp/std"),
    (lambda eps: eps.__setitem__("stats/another/mean", [np.array([0.0])] * len(eps)), "stats/another/mean"),
], ids=["index mean -12345", "timestamp mean 100000", "index mean unmoved", "index q50 unmoved", "episode_index max unmoved",
        "index std moved", "index count moved", "timestamp std doubled", "timestamp q50", "task_index changed",
        "a blank field filled", "a column dropped", "a column added"])
def test_episode_statistics_the_merge_got_wrong_are_caught(tmp_path, edit, column):
    batch, out = _merged(tmp_path)
    ab.verify_episodes(_datasets(batch), out, ab.sealed_chunks(batch))             # the merge as it should be
    _edit_episodes(edit)(out)
    with pytest.raises(BatchError, match=column.replace(".", r"\.")):
        ab.verify_episodes(_datasets(batch), out, ab.sealed_chunks(batch))


def test_a_task_index_statistic_the_merge_recomputed_is_not_a_difference(tmp_path):
    # lerobot recomputes it from the episode's task strings: the histogram's noise in the chunk's quantile
    # (4e-16 for a constant 0) becomes the exact 0.0 -- the same constant, not a change.
    batch, out = _merged(tmp_path)
    merged = ab.episodes_table(out)["stats/task_index/q50"].iloc[0]
    assert merged[0] == 0.0 and ab.episodes_table(_datasets(batch)[0])["stats/task_index/q50"].iloc[0][0] != 0.0
    ab.verify_episodes(_datasets(batch), out, ab.sealed_chunks(batch))


def test_a_field_that_is_non_finite_in_the_merged_steps_must_be_blank_in_every_episode(tmp_path):
    # Only the second and third chunks hold -inf: the first chunk's episodes carry numbers for the margin, and
    # the merged data is non-finite in it -- so the merged table owes NaN there (C8), not the chunk's numbers.
    batch = _batch(tmp_path, finite_margin=(0,))
    out = tmp_path / "merged"
    ab.merge(_datasets(batch), out, "test/batch", fake_aggregate, tmp_path / "views")
    ab.rebuild_data(_datasets(batch), out)
    with pytest.raises(BatchError, match=r"stats/safety\.stability_margin/mean"):
        ab.verify_episodes(_datasets(batch), out, ab.sealed_chunks(batch))              # not blanked yet
    ab.recompute_stats(out, fake_stats, fake_write_stats)
    ab.verify_episodes(_datasets(batch), out, ab.sealed_chunks(batch))


ten_to_a_file = functools.partial(fake_aggregate_split, per_file=10)


def _split_merge(tmp_path):
    """The merge with the videos in three files (ten episodes to a file): episodes 0 and 10 are the same
    length (4 steps) and the first of their files."""
    batch = _batch(tmp_path)
    out = tmp_path / "merged"
    ab.merge(_datasets(batch), out, "test/batch", ten_to_a_file, tmp_path / "views")
    ab.rebuild_data(_datasets(batch), out)
    return batch, out


def test_a_video_span_attached_to_another_episode_is_caught_though_every_file_is_as_it_was(tmp_path):
    # Two spans swapped: the bytes of the files and their order are the chunks', the table names the wrong file.
    batch, out = _split_merge(tmp_path)
    column = f"videos/{CAMERAS[0]}/file_index"

    def swap(eps):
        a, b = eps.index[eps["episode_index"] == 0][0], eps.index[eps["episode_index"] == 10][0]
        assert eps.loc[a, column] != eps.loc[b, column] and eps.loc[a, "length"] == eps.loc[b, "length"]
        eps.loc[a, column], eps.loc[b, column] = eps.loc[b, column], eps.loc[a, column]

    _edit_episodes(swap)(out)
    chunks = ab.sealed_chunks(batch)
    ab.verify_episodes(_datasets(batch), out, chunks)               # what the other checks see: nothing wrong
    ab.verify_loads(out, "test/batch", FakeLoaded)
    with pytest.raises(BatchError, match=r"패킷.*" + batch.episode_ids(_plan(batch, 0))[0]) as e:
        ab.verify_packets(_datasets(batch), out, packets)
    assert CAMERAS[0] in str(e.value)


def test_a_video_span_shifted_by_one_frame_is_caught(tmp_path):
    batch, out = _split_merge(tmp_path)

    def shift(eps):
        for edge in ("from", "to"):
            column = f"videos/{CAMERAS[1]}/{edge}_timestamp"
            eps.loc[eps["episode_index"] == 5, column] += 1 / FPS

    _edit_episodes(shift)(out)
    with pytest.raises(BatchError, match="패킷") as e:
        ab.verify_packets(_datasets(batch), out, packets)
    assert batch.episode_ids(_plan(batch, 0))[5] in str(e.value) and CAMERAS[1] in str(e.value)


@pytest.mark.parametrize("shift_s", [-0.005, -0.4 / FPS, -0.019], ids=["-5 ms", "-0.4 frame", "-19 ms"])
def test_a_video_span_moved_earlier_by_less_than_a_frame_is_caught(tmp_path, shift_s):
    # Moved earlier by less than a frame, a span holds the very same packets (the previous episode's last packet
    # is a frame before its start): only the first packet's distance to ``from_timestamp`` shows it.
    batch, out = _split_merge(tmp_path)
    ab.verify_packets(_datasets(batch), out, packets)

    def move(eps):
        for edge in ("from", "to"):
            eps.loc[eps["episode_index"] == 5, f"videos/{CAMERAS[1]}/{edge}_timestamp"] += shift_s

    _edit_episodes(move)(out)
    with pytest.raises(BatchError, match="첫 패킷") as e:
        ab.verify_packets(_datasets(batch), out, packets)
    assert batch.episode_ids(_plan(batch, 0))[5] in str(e.value) and CAMERAS[1] in str(e.value)


def test_an_episode_whose_video_has_fewer_packets_than_steps_is_caught_though_it_is_the_chunks(tmp_path):
    # The merged span holds the chunk's own packets, payload for payload -- the chunk's video just has a frame less
    # than the episode has steps. Only the comparison of the count with the episode's length refuses it.
    batch = _batch(tmp_path)
    path = _datasets(batch)[0] / VIDEO_PATH.format(video_key=CAMERAS[0], chunk_index=0, file_index=0)
    lines = path.read_bytes().rstrip(b"\n").split(b"\n")
    assert lines[15].startswith(repr(15 / FPS).encode()), "the last packet of the fourth episode (12..15)"
    path.write_bytes(b"\n".join(lines[:15] + lines[16:]) + b"\n")
    out = tmp_path / "merged"
    ab.merge(_datasets(batch), out, "test/batch", fake_aggregate, tmp_path / "views")
    ab.rebuild_data(_datasets(batch), out)
    with pytest.raises(BatchError, match=r"패킷 3개 · 청크 3개 · 길이 4") as e:
        ab.verify_packets(_datasets(batch), out, packets)
    assert batch.episode_ids(_plan(batch, 0))[3] in str(e.value)


def test_a_packet_that_belongs_to_no_episodes_span_is_caught(tmp_path):
    batch, out = _split_merge(tmp_path)
    path = out / VIDEO_PATH.format(video_key=CAMERAS[0], chunk_index=0, file_index=2)
    path.write_bytes(path.read_bytes() + b"99.0 stray\n")
    with pytest.raises(BatchError, match="패킷.*어느 에피소드"):
        ab.verify_packets(_datasets(batch), out, packets)


def test_a_merged_video_file_the_table_names_but_the_merge_did_not_write_is_caught(tmp_path):
    batch, out = _split_merge(tmp_path)
    (out / VIDEO_PATH.format(video_key=CAMERAS[0], chunk_index=0, file_index=1)).unlink()
    with pytest.raises(BatchError, match="file-001.mp4"):
        ab.verify_packets(_datasets(batch), out, packets)


def test_episodes_laid_in_a_file_in_another_order_than_the_chunks_are_refused_though_every_span_is_right(tmp_path):
    # Every episode's own packets are its chunk's and the table finds them: only the comparison of the whole
    # stream, in file order, sees that the file is not the chunks' packets one after the other.
    batch, out = _merged(tmp_path)
    key = CAMERAS[0]
    path = out / VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=0)
    payloads = [line.split(b" ", 1)[1] for line in path.read_bytes().rstrip(b"\n").split(b"\n")]
    ab.verify_packets(_datasets(batch), out, packets)
    _write_video(path, [(i / FPS, p) for i, p in enumerate(payloads[4:7] + payloads[0:4] + payloads[7:])])   # episodes 0 (4 steps) and 1 (3) swap places

    def respan(eps):
        columns = [f"videos/{key}/from_timestamp", f"videos/{key}/to_timestamp"]
        eps.loc[eps["episode_index"] == 1, columns] = [0 / FPS, 3 / FPS]
        eps.loc[eps["episode_index"] == 0, columns] = [3 / FPS, 7 / FPS]

    _edit_episodes(respan)(out)
    with pytest.raises(BatchError, match="패킷이 청크를 이은 것과 다릅니다"):
        ab.verify_packets(_datasets(batch), out, packets)


def test_the_videos_in_several_files_pass_their_check(tmp_path):
    batch, out = _split_merge(tmp_path)
    assert len(list((out / "videos" / CAMERAS[0]).rglob("*.mp4"))) == 3
    assert ab.verify_packets(_datasets(batch), out, packets) == {key: 99 for key in sorted(CAMERAS)}


class LoaderWithout(FakeLoaded):
    def __getitem__(self, i):
        item = super().__getitem__(i)
        del item[CAMERAS[1]]
        return item


class LoaderWithAPictureOf(FakeLoaded):
    shape = (3, 0, 2)

    def __getitem__(self, i):
        return dict(super().__getitem__(i), **{CAMERAS[0]: np.zeros(self.shape)})


def test_a_loader_item_without_a_camera_is_caught(tmp_path):
    batch, out = _merged(tmp_path)
    with pytest.raises(BatchError, match=CAMERAS[1].replace(".", r"\.")):
        ab.verify_loads(out, "test/batch", LoaderWithout)


@pytest.mark.parametrize("shape", [(3, 0, 2), (2, 2), (3, 2, 2, 2)], ids=["empty", "two dimensions", "four dimensions"])
def test_a_loader_item_whose_camera_is_not_a_picture_is_caught(tmp_path, shape):
    batch, out = _merged(tmp_path)

    class Loader(LoaderWithAPictureOf):
        pass

    Loader.shape = shape
    with pytest.raises(BatchError, match=CAMERAS[0].replace(".", r"\.")):
        ab.verify_loads(out, "test/batch", Loader)


# ── paths the real batch takes first: several step files, the order of the merged ids ──

def test_a_merge_that_splits_the_step_table_over_several_files_is_rebuilt_and_checked(tmp_path):
    # A 400-episode merge has many step files; the two-chunk rehearsal and the plain fake make one.
    batch = _batch(tmp_path)
    out = tmp_path / "merged"
    ab.merge(_datasets(batch), out, "test/batch", fake_aggregate_split, tmp_path / "views")
    assert len(list((out / "data").rglob("*.parquet"))) == 3, "가짜 병합이 파일을 나누지 못했습니다"
    assert _nulls(out), "가짜 병합이 결함을 흉내 내지 못했습니다 — 이 시험이 아무것도 증명하지 않습니다"
    assert ab.rebuild_data(_datasets(batch), out) == 3
    assert _nulls(out) == {}
    for k, episodes in enumerate([range(0, 12), range(12, 24), range(24, 25)]):
        table = pq.read_table(out / f"data/chunk-000/file-{k:03d}.parquet")
        assert sorted(set(table.column("episode_index").to_pylist())) == list(episodes)
        assert pq.ParquetFile(out / f"data/chunk-000/file-{k:03d}.parquet").metadata.num_row_groups == len(episodes)
    ab.verify_rows(_datasets(batch), out)


def test_the_whole_assembly_over_several_step_files_lands_with_all_of_them(tmp_path):
    batch = _batch(tmp_path)
    out = ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=fake_aggregate_split))
    assert sorted(p.name for p in (out / "data/chunk-000").iterdir()) == ["file-000.parquet", "file-001.parquet", "file-002.parquet"]
    assert _nulls(out) == {}


def test_merged_episodes_that_are_not_in_the_batchs_order_are_caught_though_they_equal_their_sources(tmp_path):
    # Chunks fed to the merge in the wrong order: the merged table is the concatenation of the (reversed)
    # chunk tables, so the comparison with them passes -- only the batch's own order says it is wrong.
    batch = _batch(tmp_path)
    chunks = ab.sealed_chunks(batch)
    out = tmp_path / "merged"
    reverse = list(reversed(_datasets(batch)))
    ab.merge(reverse, out, "test/batch", fake_aggregate, tmp_path / "views")
    ab.rebuild_data(reverse, out)
    with pytest.raises(BatchError, match="episode_id"):
        ab.verify_episodes(reverse, out, chunks)


def test_an_episodes_run_start_seq_has_to_be_its_chunks_start(tmp_path):
    # The converter wrote a wrong run_start_seq in a chunk's own table (the seal says the chunk began at 10):
    # merged and source agree, and C9 would only find it in a sampled episode.
    batch = _batch(tmp_path)
    chunks = ab.sealed_chunks(batch)                          # before the edit: the seal's hashes would refuse it
    second = _datasets(batch)[1]
    eps = pd.read_parquet(second / EPISODES)
    eps["run_start_seq"] = 0
    eps.to_parquet(second / EPISODES, index=False)
    out = tmp_path / "merged"
    ab.merge(_datasets(batch), out, "test/batch", fake_aggregate, tmp_path / "views")
    ab.rebuild_data(_datasets(batch), out)
    with pytest.raises(BatchError, match="run_start_seq") as e:
        ab.verify_episodes(_datasets(batch), out, chunks)
    assert batch.episode_ids(_plan(batch, 10))[0] in str(e.value)


def test_a_merged_table_without_run_start_seq_is_refused(tmp_path):
    batch, out = _merged(tmp_path)
    eps = pd.read_parquet(out / EPISODES).drop(columns=["run_start_seq"])
    eps.to_parquet(out / EPISODES, index=False)
    with pytest.raises(BatchError, match="run_start_seq"):
        ab.verify_episodes(_datasets(batch), out, ab.sealed_chunks(batch))


def _merge_and_rebuild(batch, tmp_path, aggregate=fake_aggregate):
    out = tmp_path / "merged"
    shutil.rmtree(out, ignore_errors=True)
    ab.merge(_datasets(batch), out, "test/batch", aggregate, tmp_path / "views")
    ab.rebuild_data(_datasets(batch), out)
    return out


@pytest.mark.parametrize("kind", ["rows", "one run"])
def test_a_rows_run_start_is_checked_per_chunk_kind(tmp_path, kind):
    # A chunk of condition-table rows is a run per row: every episode's run_start_seq is its own row. A chunk of
    # S1d is one run: every episode's is the chunk's first row. Each is held to its own -- the other kind's value
    # is refused, though merged and source agree (the converter wrote it in the chunk's own table).
    batch = _rows_batch(tmp_path) if kind == "rows" else _batch(tmp_path)
    chunks = ab.sealed_chunks(batch)                          # before the edit: the seal's hashes would refuse it
    out = _merge_and_rebuild(batch, tmp_path)
    ab.verify_episodes(_datasets(batch), out, chunks)
    assert [int(v) for v in ab.episodes_table(out)["run_start_seq"]] == (
        [12, 37, 3, 8, 44, 41] if kind == "rows" else [0] * 10 + [10] * 10 + [20] * 5)
    plan, chunk, _ = chunks[0]
    eps = pd.read_parquet(chunk / DATASET_DIR / EPISODES)
    # The other kind's: a row that is not the chunk's first given the first's, or the second's given its own row.
    at = 0 if kind == "rows" else 1
    assert int(eps.loc[at, "run_start_seq"]) != plan.rows[0 if kind == "rows" else 1]
    eps.loc[at, "run_start_seq"] = plan.rows[0] if kind == "rows" else plan.rows[1]
    eps.to_parquet(chunk / DATASET_DIR / EPISODES, index=False)
    out = _merge_and_rebuild(batch, tmp_path)
    with pytest.raises(BatchError, match="run_start_seq") as e:
        ab.verify_episodes(_datasets(batch), out, chunks)
    assert str(eps.loc[at, "episode_id"]) in str(e.value)


def test_a_chunk_whose_rows_were_asked_out_of_order_assembles_in_the_converters_order(tmp_path):
    # Rows 21 then 20, two to a chunk: the plan and the seal keep that order, the converter writes the chunk's
    # episodes by their recordings' names -- 20, then 21. The assembly holds the merged table to the converter's
    # order within a chunk, and each episode to its own row.
    batch = _rows_batch(tmp_path, parts=(("S1e", (21, 20)),))
    (plan,) = batch.chunks()
    assert plan.rows == (21, 20) and read_seal(batch.chunk_dir(plan))["episode_ids"] == batch.episode_ids(plan)
    out = ab.assemble(batch, tmp_path, **FAKES)
    eps = ab.episodes_table(out)
    assert [int(eid.split("_")[1]) for eid in eps["episode_id"]] == [20, 21]
    assert [int(v) for v in eps["run_start_seq"]] == [20, 21]


def test_a_chunk_whose_episodes_are_in_the_plans_order_and_not_the_converters_is_refused(tmp_path):
    # A chunk's distribution that holds rows 37, 12 as asked is not what the converter writes: the merged table
    # equals its sources, and only the converter's order says it is wrong.
    batch = _rows_batch(tmp_path)
    chunks = ab.sealed_chunks(batch)
    plan, chunk, _ = chunks[0]
    eps = pd.read_parquet(chunk / DATASET_DIR / EPISODES)
    assert [int(eid.split("_")[1]) for eid in eps["episode_id"]] == [12, 37]
    eps["episode_id"] = list(reversed(list(eps["episode_id"])))
    eps["run_start_seq"] = list(reversed(list(eps["run_start_seq"])))
    eps.to_parquet(chunk / DATASET_DIR / EPISODES, index=False)
    out = _merge_and_rebuild(batch, tmp_path)
    with pytest.raises(BatchError, match="episode_id"):
        ab.verify_episodes(_datasets(batch), out, chunks)


# ── chunks of several scenarios: one task table ──

def _sentences_of_the_steps(root):
    """The task sentence of every step of a distribution as the loader hands it: the row of meta/tasks.parquet
    at the step's task_index."""
    table = pd.read_parquet(Path(root) / "meta/tasks.parquet")
    steps = pa.concat_tables([pq.read_table(f) for f in sorted((Path(root) / "data").rglob("*.parquet"))])
    return [table.index[k] for k in steps.column("task_index").to_pylist()]


def test_three_scenarios_assemble_into_one_task_table(tmp_path):
    batch = _rows_batch(tmp_path)
    assert [TASK_OF[plan.scenario] for plan in batch.chunks()] == ["a", "b", "c", "a"]
    assert all(list(pd.read_parquet(d / "meta/tasks.parquet").index) == [t] and _sentences_of_the_steps(d) ==
               [t] * len(_sentences_of_the_steps(d)) for d, t in zip(_datasets(batch), "abca")), \
        "청크마다 과제가 하나(제 표의 0행)가 아닙니다 — 이 시험이 아무것도 증명하지 않습니다"
    out = ab.assemble(batch, tmp_path, **FAKES)
    assert list(pd.read_parquet(out / "meta/tasks.parquet").index) == ["a", "b", "c"]
    # Every step the task of its own chunk, as the loader reads it.
    lengths = [int(k) for d in _datasets(batch) for k in ab.episodes_table(d)["length"]]
    tasks = [t for d, t in zip(_datasets(batch), "abca") for _ in range(len(ab.episodes_table(d)))]
    assert _sentences_of_the_steps(out) == [t for t, k in zip(tasks, lengths) for _ in range(k)]
    # The episodes' task_index statistics are those of the renumbered column: a, b, c are rows 0, 1, 2.
    eps = ab.episodes_table(out)
    row = {"a": 0.0, "b": 1.0, "c": 2.0}
    for key in ("min", "max", "mean", "q50"):
        assert [float(v[0]) for v in eps[f"stats/task_index/{key}"]] == [row[t] for t in tasks], key
    assert [float(v[0]) for v in eps["stats/task_index/std"]] == [0.0] * len(eps)
    assert [int(v[0]) for v in eps["stats/task_index/count"]] == lengths
    stats = json.loads((out / "meta/stats.json").read_text(encoding="utf-8"))["task_index"]
    numbers = np.repeat([row[t] for t in tasks], lengths)
    assert stats["min"] == [0.0] and stats["max"] == [2.0] and stats["mean"] == pytest.approx([numbers.mean()])
    # The chunks in the batch's order, and within one the converter's: 37 and 12 come as 12, 37.
    assert [(eid.split("_")[0], int(eid.split("_")[1])) for eid in ab.episodes_table(out)["episode_id"]] == [
        ("S1e", 12), ("S1e", 37), ("S1f", 3), ("S1f", 8), ("S1g", 44), ("S1e", 41)]


@pytest.mark.parametrize("total", [None, 0, 2, 4, "3", 3.0], ids=["absent", "0", "fewer", "more", "text", "float"])
def test_a_merged_total_tasks_that_is_not_the_task_tables_rows_is_refused(tmp_path, total):
    # The loader reads the task table only when total_tasks is above 0 and counts the tasks by it: three chunks'
    # tasks a, b, c make three rows, and info.json has to say three -- as an integer.
    batch = _rows_batch(tmp_path)
    chunks = ab.sealed_chunks(batch)
    out = _merge_and_rebuild(batch, tmp_path)
    info = json.loads((out / "meta/info.json").read_text(encoding="utf-8"))
    assert info["total_tasks"] == 3 == len(pd.read_parquet(out / "meta/tasks.parquet"))
    ab.verify_episodes(_datasets(batch), out, chunks)
    if total is None:
        del info["total_tasks"]
    else:
        info["total_tasks"] = total
    (out / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(BatchError, match="total_tasks") as e:
        ab.verify_episodes(_datasets(batch), out, chunks)
    assert "3" in str(e.value), str(e.value)


def test_the_assembly_refuses_a_merge_that_counts_its_tasks_wrong(tmp_path):
    def merge(**kwargs):
        fake_aggregate(**kwargs)
        info_path = Path(kwargs["aggr_root"]) / "meta/info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info_path.write_text(json.dumps(dict(info, total_tasks=1)), encoding="utf-8")   # the first chunk's own count

    _refused(tmp_path, _rows_batch(tmp_path), "total_tasks", aggregate=merge)


def test_a_merge_that_orders_tasks_differently_is_refused(tmp_path):
    # A merge whose task table is in another order than the chunks' first appearances, its own steps and
    # statistics numbered after that table: consistent with itself -- but the rows rebuilt from the chunks are
    # numbered in the chunks' order, and in its table they would name other sentences.
    batch = _rows_batch(tmp_path)
    reordered = functools.partial(fake_aggregate, task_order=lambda tasks: list(reversed(tasks)))
    out = tmp_path / "merged"
    ab.merge(_datasets(batch), out, "test/batch", reordered, tmp_path / "views")
    assert list(pd.read_parquet(out / "meta/tasks.parquet").index) == ["c", "b", "a"]
    with pytest.raises(BatchError, match="병합본의 과제 표가 청크 차례대로 처음 나온") as e:
        ab.rebuild_data(_datasets(batch), out)
    assert "'c'" in str(e.value) and "'a'" in str(e.value)
    _refused(tmp_path, batch, "병합본의 과제 표가 청크 차례대로 처음 나온", aggregate=reordered)


def _reorder_the_merged_tasks(batch, out):
    """The merged task table put in another order after the rows were rebuilt."""
    table = pd.read_parquet(out / "meta/tasks.parquet")
    pd.DataFrame({"task_index": range(len(table))}, index=pd.Index(list(reversed(table.index)), name="task")) \
        .to_parquet(out / "meta/tasks.parquet")


def _another_task_in_a_chunks_episode(batch, out):
    """The second episode of the first chunk lists another task than its steps name -- in the chunk's own table,
    so merged and source agree."""
    root = _datasets(batch)[0]
    eps = pd.read_parquet(root / EPISODES)
    eps["tasks"] = [np.array(["b"] if i == 1 else list(t), dtype=object) for i, t in enumerate(eps["tasks"])]
    eps.to_parquet(root / EPISODES, index=False)
    return _merge_and_rebuild(batch, out.parent)


@pytest.mark.parametrize("change", [_reorder_the_merged_tasks, _another_task_in_a_chunks_episode],
                         ids=["the merged table reordered", "an episode listing another task"])
def test_an_episode_whose_steps_name_another_task_than_its_own_is_caught(tmp_path, change):
    batch = _rows_batch(tmp_path)
    chunks = ab.sealed_chunks(batch)
    out = _merge_and_rebuild(batch, tmp_path)
    ab.verify_episodes(_datasets(batch), out, chunks)
    out = change(batch, out) or out
    with pytest.raises(BatchError, match="tasks") as e:
        ab.verify_episodes(_datasets(batch), out, chunks)
    # The first chunk's episodes as its distribution holds them (the converter's order).
    first_chunk = [str(eid) for eid in ab.episodes_table(_datasets(batch)[0])["episode_id"]]
    assert "과제" in str(e.value) and first_chunk[0 if change is _reorder_the_merged_tasks else 1] in str(e.value)


# ── the statistics the module computes itself ──

def _stats_of(tmp_path):
    batch = _batch(tmp_path)
    out = tmp_path / "merged"
    ab.merge(_datasets(batch), out, "test/batch", fake_aggregate, tmp_path / "views")
    ab.rebuild_data(_datasets(batch), out)
    ab.recompute_stats(out, fake_stats, fake_write_stats)
    return out


def _edit_stats(out, edit):
    path = out / "meta/stats.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    edit(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_the_recomputed_statistics_pass_their_invariants(tmp_path):
    out = _stats_of(tmp_path)
    stats = json.loads((out / "meta/stats.json").read_text(encoding="utf-8"))
    steps = sum([4, 3, 5, 4, 3, 5, 4, 3, 5, 4] * 2 + [4, 3, 5, 4, 3])
    assert stats["episode_index"]["max"] == [24.0] and stats["index"]["max"] == [steps - 1.0]
    ab.verify_stats(out)


@pytest.mark.parametrize("edit, phrase", [
    (lambda d: d["index"].update(max=[9.0]), "index.max"),                  # the third defect: the last chunk's, not the batch's
    (lambda d: d["episode_index"].update(max=[4.0]), "episode_index.max"),
    (lambda d: d["timestamp"].update(count=[1]), "timestamp.count"),
    (lambda d: d["safety.stability_margin"].update(count=[1]), "safety.stability_margin.count"),   # a blanked block keeps its count
    (lambda d: d.pop("index"), "index.max"),
    (lambda d: d["episode_index"].pop("max"), "episode_index.max"),
    (lambda d: d["timestamp"].pop("count"), "timestamp.count"),
], ids=["index unshifted", "episode_index unshifted", "a count off", "a blanked block's count off",
        "no index block", "no max", "no count"])
def test_statistics_that_disagree_with_the_tables_are_caught(tmp_path, edit, phrase):
    out = _stats_of(tmp_path)
    _edit_stats(out, edit)
    with pytest.raises(BatchError, match=phrase.replace(".", r"\.")):
        ab.verify_stats(out)


def test_the_count_of_a_video_feature_is_not_the_step_count(tmp_path):
    # LeRobot samples frames for the statistics of a video: its count is the sample, not the steps.
    out = _stats_of(tmp_path)
    info = json.loads((out / "meta/info.json").read_text(encoding="utf-8"))
    info["features"]["observation.images.head_rgb"] = {"dtype": "video"}
    (out / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    _edit_stats(out, lambda d: d.update({"observation.images.head_rgb": {"count": [3]}}))
    ab.verify_stats(out)


def test_a_feature_the_statistics_do_not_have_is_not_asked_for(tmp_path):
    # The string features have no statistics at all.
    out = _stats_of(tmp_path)
    info = json.loads((out / "meta/info.json").read_text(encoding="utf-8"))
    info["features"]["safety.event"] = {"dtype": "string"}
    (out / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    ab.verify_stats(out)


def _scaled(key, factor):
    """An edit of the statistics file: every value of ``key`` of the timestamp, times ``factor``."""
    return lambda d: d["timestamp"].update({key: [v * factor for v in d["timestamp"][key]]})


@pytest.mark.parametrize("edit, phrase, said", [
    (lambda d: d["index"].update(mean=[-12345.0]), "index.mean", "스텝"),                # an overwritten mean, as the file would carry it
    (lambda d: d["timestamp"].update(mean=[100000.0]), "timestamp.mean", "스텝"),
    (_scaled("mean", 1.01), "timestamp.mean", "스텝"),
    (_scaled("std", 2.0), "timestamp.std", "스텝"),
    (lambda d: d["timestamp"].update(min=[0.5]), "timestamp.min", "스텝"),
    (lambda d: d["timestamp"].update(max=[9.0]), "timestamp.max", "스텝"),
    (lambda d: d["index"].update(std=[0.0]), "index.std", "스텝"),
    (_scaled("std", 1.001), "timestamp.std", "에피소드별"),             # inside what the step table can tell, outside the episodes'
    (_scaled("q50", 1.5), "timestamp.q50", "에피소드별"),               # the quantiles are only the episodes' weighted mean
    (lambda d: d["timestamp"].pop("std"), "timestamp.std", "에피소드별"),
    (lambda d: d["safety.stability_margin"].update(mean=[0.15]), "safety.stability_margin.mean", "비유한"),     # a blanked block given a number
], ids=["index mean", "timestamp mean", "mean 1 % off", "std doubled", "min zero", "max off", "index std zero",
        "std 0.1 % off", "q50 off", "std missing", "blank block filled"])
def test_statistics_that_are_not_what_the_tables_and_episodes_say_are_caught(tmp_path, edit, phrase, said):
    out = _stats_of(tmp_path)
    ab.verify_stats(out)
    _edit_stats(out, edit)
    with pytest.raises(BatchError, match=phrase.replace(".", r"\.") + ".*" + said):
        ab.verify_stats(out)


def test_a_statistic_wrong_in_the_episodes_as_well_is_still_compared_with_the_steps(tmp_path):
    # The file is the pooling of the episodes' columns -- which are wrong, in the way a chunk's own converter
    # could have left them. Only the comparison with the step table sees it.
    out = _stats_of(tmp_path)
    _edit_episodes(lambda eps: eps.__setitem__("stats/timestamp/mean", [v * 1.5 for v in eps["stats/timestamp/mean"]]))(out)
    ab.recompute_stats(out, fake_stats, fake_write_stats)
    with pytest.raises(BatchError, match=r"timestamp\.mean.*스텝"):
        ab.verify_stats(out)


def test_a_field_that_is_finite_throughout_is_held_to_its_numbers(tmp_path):
    batch = _batch(tmp_path, finite_margin=(0, 10, 20))
    out = ab.assemble(batch, tmp_path, **FAKES)
    stats = json.loads((out / "meta/stats.json").read_text(encoding="utf-8"))
    assert stats["safety.stability_margin"]["mean"] == pytest.approx([0.15]) and "undefined_reason" not in stats["safety.stability_margin"]
    _edit_stats(out, lambda d: d["safety.stability_margin"].update(mean=[0.16]))
    with pytest.raises(BatchError, match=r"safety\.stability_margin\.mean"):
        ab.verify_stats(out)


def test_the_unshifted_index_statistics_of_the_unrepaired_merge_fail_the_whole_assembly(tmp_path):
    # An aggregator stand-in that reproduces the third defect: the batch's largest index comes out as one
    # chunk's own. The assembly must not land.
    def unshifted(per_episode):
        out = fake_stats(per_episode)
        out["index"]["max"] = np.array([4.0])                  # the last episode's extreme within its own chunk
        return out

    batch = _batch(tmp_path)
    with pytest.raises(BatchError, match=r"index\.max"):
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate_stats=unshifted))
    assert not (batch.root / DATASET_DIR).exists()


# ── the whole assembly ──

def test_the_assembly_lands_in_the_batchs_dataset_directory(tmp_path):
    batch = _batch(tmp_path)
    out = ab.assemble(batch, tmp_path, **FAKES)
    assert out == batch.root / DATASET_DIR and (out / EPISODES).is_file()
    assert (out / "README.md").read_text(encoding="utf-8") == "card at assembly for 25 episodes"
    assert _nulls(out) == {}
    assert sorted(p.name for p in batch.root.iterdir()) == ["batch.json", "chunks", "dataset"]


def test_a_second_assembly_replaces_the_first(tmp_path):
    batch = _batch(tmp_path)
    ab.assemble(batch, tmp_path, **FAKES)
    (batch.root / DATASET_DIR / "left-from-the-first.txt").write_text("x", encoding="utf-8")
    again = dict(FAKES, lay=lambda repo_root, out, sidecars: (Path(out) / "README.md").write_text("a new card", encoding="utf-8") and ["README.md"])
    out = ab.assemble(batch, tmp_path, **again)
    assert (out / "README.md").read_text(encoding="utf-8") == "a new card"
    assert not (out / "left-from-the-first.txt").exists()


def test_a_failed_assembly_leaves_the_previous_result(tmp_path):
    batch = _batch(tmp_path)
    ab.assemble(batch, tmp_path, **FAKES)

    def broken(path):
        raise OSError("disk went away")

    def failed_runs_card(repo_root, out, sidecars):
        # A card of its own: were the failed run put in place before its checks, this is what would be left.
        (Path(out) / "README.md").write_text("the card of the failed run", encoding="utf-8")
        return ["README.md"]

    with pytest.raises(OSError, match="disk went away"):
        ab.assemble(batch, tmp_path, **dict(FAKES, read_packets=broken, lay=failed_runs_card))
    assert (batch.root / DATASET_DIR / "README.md").read_text(encoding="utf-8") == "card at assembly for 25 episodes"


def test_the_chunks_are_untouched_by_an_assembly(tmp_path):
    batch = _batch(tmp_path)
    before = {s: sealed_files(_chunk(batch, s)) for s in _starts(batch)}
    ab.assemble(batch, tmp_path, **FAKES)
    assert {s: sealed_files(_chunk(batch, s)) for s in _starts(batch)} == before


# ── each self-check is called by the assembly: an input only that call refuses ──
# A check tested only by calling it passes while the assembly stops calling it. Each of these goes through
# assemble(); the corruption is made after the step table was rebuilt and the statistics recomputed (the
# ``lay`` stand-in runs then), so nothing before the check repairs it and no other check refuses it.

def _lay_then(edit):
    """A ``lay`` stand-in that also tampers with the finished assembly: ``edit`` is given the work directory."""
    def lay(repo_root, out, sidecars):
        names = _lay(repo_root, out, sidecars)
        edit(Path(out))
        return names
    return lay


def _refused(tmp_path, batch, match, **overrides):
    with pytest.raises(BatchError, match=match):
        ab.assemble(batch, tmp_path, **dict(FAKES, **overrides))
    assert sorted(p.name for p in batch.root.iterdir()) == ["batch.json", "chunks"], "무엇인가 설치됐거나 만들다 만 것이 남았습니다"


def _tamper_step_value(out):
    table = pq.read_table(out / DATA)
    i = table.schema.get_field_index("frame_index")        # no statistic, not read by the loader check: only the rows check sees it
    values = table.column(i).to_pylist()
    values[7] += 1
    pq.write_table(table.set_column(i, table.schema.field(i), pa.array(values, pa.int64())), out / DATA)


def test_the_assembly_refuses_a_step_value_that_is_not_the_chunks(tmp_path):
    batch = _batch(tmp_path)
    _refused(tmp_path, batch, "frame_index", lay=_lay_then(_tamper_step_value))


def test_the_assembly_refuses_an_episode_column_that_is_not_the_chunks(tmp_path):
    batch = _batch(tmp_path)
    _refused(tmp_path, batch, "outcome", lay=_lay_then(_edit_episodes(lambda eps: eps.__setitem__("outcome", "fail"))))


def test_the_assembly_refuses_a_statistics_file_that_is_not_the_tables(tmp_path):
    batch = _batch(tmp_path)
    _refused(tmp_path, batch, r"timestamp\.mean", lay=_lay_then(lambda out: _edit_stats(out, _scaled("mean", 1.01))))


def test_the_assembly_refuses_episode_statistics_that_are_not_the_chunks(tmp_path):
    # The two overwritten means, made by the merge itself: the statistics are recomputed from them and the file is
    # then consistent with the wrong columns -- only the comparison with the chunks sees it.
    def corrupt(eps):
        eps["stats/index/mean"] = [np.array([-12345.0])] * len(eps)
        eps["stats/timestamp/mean"] = [np.array([100000.0])] * len(eps)

    def merge(**kwargs):
        fake_aggregate(**kwargs)
        _edit_episodes(corrupt)(kwargs["aggr_root"])

    batch = _batch(tmp_path)
    _refused(tmp_path, batch, r"stats/(index|timestamp)/mean", aggregate=merge)


def test_the_assembly_refuses_a_video_span_attached_to_another_episode(tmp_path):
    column = f"videos/{CAMERAS[0]}/file_index"

    def swap(eps):
        a, b = eps.index[eps["episode_index"] == 0][0], eps.index[eps["episode_index"] == 10][0]
        eps.loc[a, column], eps.loc[b, column] = eps.loc[b, column], eps.loc[a, column]

    batch = _batch(tmp_path)
    _refused(tmp_path, batch, "패킷", aggregate=ten_to_a_file, lay=_lay_then(_edit_episodes(swap)))


class _CannotRead(FakeLoaded):
    def __getitem__(self, i):
        if i == 4:                                          # the first step of the second episode: always sampled
            raise RuntimeError("the decoder gave up")
        return super().__getitem__(i)


def test_the_assembly_refuses_a_distribution_the_loader_cannot_read(tmp_path):
    batch = _batch(tmp_path)
    _refused(tmp_path, batch, "읽지 못했습니다.*the decoder gave up", open_dataset=_CannotRead)


def test_the_assembly_refuses_a_distribution_whose_loader_gives_no_picture(tmp_path):
    batch = _batch(tmp_path)
    _refused(tmp_path, batch, CAMERAS[1].replace(".", r"\."), open_dataset=LoaderWithout)


def test_a_refused_assembly_leaves_the_previous_result_untouched(tmp_path):
    batch = _batch(tmp_path)
    ab.assemble(batch, tmp_path, **FAKES)
    def tree():
        root = batch.root / DATASET_DIR
        return {str(f.relative_to(root)): f.read_bytes() for f in sorted(root.rglob("*")) if f.is_file()}

    before = tree()
    with pytest.raises(BatchError):
        ab.assemble(batch, tmp_path, **dict(FAKES, open_dataset=_CannotRead))
    assert tree() == before
    assert sorted(p.name for p in batch.root.iterdir()) == ["batch.json", "chunks", "dataset"]


# ── the canon: refused before anything is laid in ──

def _canon_repo(tmp_path, change=None):
    repo = tmp_path / "repo"
    (repo / "meta").mkdir(parents=True)
    canon = json.loads((Path(__file__).resolve().parents[2] / "meta/cameras.json").read_text(encoding="utf-8"))
    if change:
        change(canon)
    (repo / "meta/cameras.json").write_text(json.dumps(canon), encoding="utf-8")
    return repo


def test_a_recording_made_under_another_camera_canon_stops_the_assembly(tmp_path):
    from generation import cameras as camera_canon
    repo = _canon_repo(tmp_path)
    stamp = camera_canon.recording_fingerprint(camera_canon.load(repo / "meta/cameras.json"))
    good = {"episode_id": "S1d_0000_seed1", "media": {"canon_fingerprint": stamp}}
    stale = {"episode_id": "S1d_0001_seed2", "media": {"canon_fingerprint": "0" * 64}}
    with pytest.raises(BatchError, match="S1d_0001_seed2"):
        ab.lay_canon(repo, tmp_path / "out", [good, stale])
    assert not (tmp_path / "out").exists()


def test_a_canon_that_declares_a_pack_stops_the_assembly(tmp_path):
    from validator.make_fixture import canon_with_pack
    repo = _canon_repo(tmp_path)
    canon = json.loads((repo / "meta/cameras.json").read_text(encoding="utf-8"))
    canon.pop("packs", None)
    (repo / "meta/cameras.json").write_text(json.dumps(canon_with_pack(canon)), encoding="utf-8")
    with pytest.raises(BatchError, match="팩"):
        ab.lay_canon(repo, tmp_path / "out", [])


# ── the final swap: a failure cannot lose both results ──

def _assembled_with_card(tmp_path, card):
    batch = _batch(tmp_path)
    ab.assemble(batch, tmp_path, **dict(FAKES, lay=lambda r, o, sc: (Path(o) / "README.md").write_text(card, encoding="utf-8") and ["README.md"]))
    return batch


def test_a_replace_that_fails_puts_the_previous_result_back(tmp_path, monkeypatch):
    batch = _assembled_with_card(tmp_path, "the first result")
    real = os.replace

    def failing(src, dst):
        if Path(src).name == ab.ASSEMBLING:
            raise OSError("no space left on device")
        return real(src, dst)

    monkeypatch.setattr(ab.os, "replace", failing)
    with pytest.raises(OSError, match="no space"):
        ab.assemble(batch, tmp_path, **dict(FAKES, lay=lambda r, o, sc: (Path(o) / "README.md").write_text("the second", encoding="utf-8") and ["README.md"]))
    assert (batch.root / DATASET_DIR / "README.md").read_text(encoding="utf-8") == "the first result"
    assert sorted(p.name for p in batch.root.iterdir()) == ["batch.json", "chunks", "dataset"], "옮겨 둔 것이나 만들다 만 것이 남았습니다"


def test_an_interruption_while_the_old_result_is_removed_leaves_the_new_one_in_place(tmp_path, monkeypatch):
    # Ctrl-C during the removal of a tree of tens of GB: the new result has been renamed in by then, and the
    # ``finally`` must not take it for the work directory it used to be.
    batch = _assembled_with_card(tmp_path, "the first result")
    real, removals = shutil.rmtree, []

    def interrupted(path, *args, **kwargs):
        if Path(path).name == ab.PREVIOUS:
            removals.append(path)
            if len(removals) == 2:               # the first clears a leftover before the swap; the second removes the old result
                raise KeyboardInterrupt
        return real(path, *args, **kwargs)

    monkeypatch.setattr(ab.shutil, "rmtree", interrupted)
    with pytest.raises(KeyboardInterrupt):
        ab.assemble(batch, tmp_path, **dict(FAKES, lay=lambda r, o, sc: (Path(o) / "README.md").write_text("the second", encoding="utf-8") and ["README.md"]))
    assert len(removals) == 2, "시험이 옛 결과를 지우는 중에 끊지 못했습니다"
    assert (batch.root / DATASET_DIR / "README.md").read_text(encoding="utf-8") == "the second"
    monkeypatch.setattr(ab.shutil, "rmtree", real)
    # What the cut removal left is the old result: the next assembly goes on regardless and leaves nothing.
    assert (batch.root / ab.PREVIOUS).exists()
    out = ab.assemble(batch, tmp_path, **dict(FAKES, lay=lambda r, o, sc: (Path(o) / "README.md").write_text("the third", encoding="utf-8") and ["README.md"]))
    assert (out / "README.md").read_text(encoding="utf-8") == "the third"
    assert sorted(p.name for p in batch.root.iterdir()) == ["batch.json", "chunks", "dataset"]


# ── the canon the chunks shipped against the one laid now ──

def _edit_canon(tmp_path, rel, edit):
    path = tmp_path / rel
    doc = json.loads(path.read_text(encoding="utf-8"))
    edit(doc)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _raise_stop_time(doc):
    rule = next(r for r in doc["rules"] if r["rule_id"] == "R-TS15066-SSM-01")
    rule["parameters"]["robot_stop_time"]["value"] += 0.01


def _reword_objective(doc):
    next(s for s in doc["scenarios"] if s["scenario_id"] == "S1d")["objective"] += " (다듬은 문장)"


def _reword_the_notes(doc) -> int:
    """Reword every open issue, every top-level note and two sentences of S1d; how many that was."""
    notes = [k for k, v in doc.items() if isinstance(v, str) and (k.endswith("_note") or k == "seed_rule")]
    for key in notes:
        doc[key] += " (다듬은 문장)"
    doc["open_issues"] = [text + " (다듬은 문장)" for text in doc["open_issues"]]
    s1d = next(s for s in doc["scenarios"] if s["scenario_id"] == "S1d")
    for key in ("objective", "human_behavior"):
        s1d[key] += " (다듬은 문장)"
    return len(notes) + len(doc["open_issues"]) + 2


def _reworded_notes(tmp_path) -> int:
    """_reword_the_notes on the canon under ``tmp_path``; the count, which has to be more than the
    list used to show for the tests below to say anything."""
    counted = []
    _edit_canon(tmp_path, "meta/scenarios.json", lambda doc: counted.append(_reword_the_notes(doc)))
    assert counted[0] > ab.CHANGES_SHOWN, "시험이 잘리던 길이를 넘기지 못했습니다"
    return counted[0]


def _never_merged(**kwargs):
    pytest.fail("조립이 정본 대조 전에 병합을 시작했습니다")


def test_chunks_that_shipped_the_canon_laid_now_are_assembled(tmp_path):
    batch = _batch(tmp_path)
    lines = []
    ab.assemble(batch, tmp_path, **dict(FAKES, say=lines.append))
    assert any("정본 대조" in line and "같음" in line for line in lines), lines


def test_a_threshold_changed_since_the_chunks_were_made_is_refused_before_anything_is_merged(tmp_path):
    batch = _batch(tmp_path)
    _edit_canon(tmp_path, "meta/rules.json", _raise_stop_time)
    with pytest.raises(BatchError, match="R-TS15066-SSM-01") as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged))
    said = str(e.value)
    assert "robot_stop_time" in said and "다시 생성" in said and "--accept-canon-change" in said
    assert "meta/rules.json" in said and "(숫자)" in said
    assert not (batch.root / ab.ASSEMBLING).exists() and not (batch.root / DATASET_DIR).exists()


def test_the_same_change_with_the_flag_is_assembled_and_the_difference_is_printed(tmp_path):
    batch = _batch(tmp_path)
    _edit_canon(tmp_path, "meta/rules.json", _raise_stop_time)
    lines = []
    out = ab.assemble(batch, tmp_path, **dict(FAKES, say=lines.append, accept_canon_change=True))
    assert (out / EPISODES).is_file()
    printed = "\n".join(lines)
    assert "R-TS15066-SSM-01" in printed and "robot_stop_time" in printed and "--accept-canon-change" in printed


def test_a_change_of_wording_names_the_scenario_and_is_accepted_with_the_flag(tmp_path):
    batch = _batch(tmp_path)
    _edit_canon(tmp_path, "meta/scenarios.json", _reword_objective)
    with pytest.raises(BatchError, match=r"scenarios\[S1d\]\.objective \(문장\)"):
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged))
    ab.assemble(batch, tmp_path, **dict(FAKES, accept_canon_change=True))


def test_a_refusal_lists_every_difference_and_what_is_not_a_sentence_first(tmp_path):
    # The list is what decides whether the flag may be given: only when every line is a sentence. In file
    # order and cut at CHANGES_SHOWN, a changed number of meta/rules.json sits behind the reworded
    # sentences of meta/scenarios.json, out of sight.
    batch = _batch(tmp_path)
    reworded = _reworded_notes(tmp_path)
    _edit_canon(tmp_path, "meta/rules.json", _raise_stop_time)
    with pytest.raises(BatchError) as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged))
    said = str(e.value)
    listed = [line.strip() for line in said.splitlines() if " — 청크 " in line]
    assert "meta/rules.json" in listed[0] and "robot_stop_time" in listed[0] and "(숫자)" in listed[0]
    assert sum("(문장)" in line for line in listed) == reworded and len(listed) == reworded + 1
    assert "… 외" not in said and f"문장이 아닌 것 1건 · 문장 {reworded}건" in said


@pytest.mark.parametrize("shipped, wanted", [
    ({"x(문장)": 1}, {"x(문장)": 2}),                                   # a number
    ({"unit(문장)": "m"}, {"unit(문장)": "s"}),                          # a string code reads
    ({}, {"k(문장)": 1}),                                               # only in the canon now
    ({"k(문장)": 1}, {}),                                               # only in the chunk
    ({"k(문장)": [1]}, {"k(문장)": [1, 2]}),                             # a list
    ({"rules": [{"rule_id": "R(문장)", "v": 1}]}, {"rules": [{"rule_id": "R(문장)", "v": 2}]}),
])
def test_a_value_is_not_a_sentence_because_its_key_spells_the_marker(shipped, wanted):
    # A key or an id goes into the path as it is. The kind is what the line ends in.
    (line,) = ab.canon_changes(shipped, wanted)
    assert "(문장)" in line and not line.endswith("(문장)")
    listed = f"meta/rules.json: {line} — 청크 0000, 0010, 0020 외 37"
    assert ab._value_first(listed)[0] is False                         # read before the sentences, and counted with the values
    assert ab._value_first("meta/rules.json: note (문장) — 청크 0000")[0] is True


def test_a_sentence_stays_one_under_a_key_that_spells_the_chunk_separator():
    # The kind is read before the last " — 청크 ", the one check_canon appends. Read before the
    # first, the line ends at the key and the sentence counts as a value.
    key = "note — 청크 x_note"
    (line,) = ab.canon_changes({key: "old words"}, {key: "new words"})
    assert line.endswith("(문장)") and " — 청크 " in line
    assert ab._value_first(f"meta/rules.json: {line} — 청크 0000, 0010")[0] is True


def test_a_refusal_counts_a_value_under_a_key_that_spells_the_marker(tmp_path):
    batch = _batch(tmp_path)
    reworded = _reworded_notes(tmp_path)

    def a_new_number(doc):
        next(s for s in doc["scenarios"] if s["scenario_id"] == "S1d")["limit(문장)"] = 3

    _edit_canon(tmp_path, "meta/scenarios.json", a_new_number)
    with pytest.raises(BatchError) as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged))
    said = str(e.value)
    listed = [line.strip() for line in said.splitlines() if " — 청크 " in line]
    assert "limit(문장) (지금 정본에만 있음)" in listed[0]
    assert f"문장이 아닌 것 1건 · 문장 {reworded}건" in said


def test_the_accepted_changes_are_all_printed_with_their_tally(tmp_path):
    batch = _batch(tmp_path)
    reworded = _reworded_notes(tmp_path)
    lines = []
    ab.assemble(batch, tmp_path, **dict(FAKES, say=lines.append, accept_canon_change=True))
    listed = [line for line in lines if " — 청크 " in line]
    assert len(listed) == reworded and all("(문장)" in line for line in listed)
    assert any(f"문장이 아닌 것 0건 · 문장 {reworded}건" in line for line in lines)
    assert not any("… 외" in line for line in lines)


def test_a_chunk_without_the_canon_it_shipped_is_refused(tmp_path):
    batch = _batch(tmp_path)
    chunk = _chunk(batch, 10)
    (chunk / DATASET_DIR / "meta/rules.json").unlink()
    record = read_seal(chunk)
    write_seal(chunk, dict(record, files=sealed_files(chunk)))        # sealed that way, so the seal's own check passes
    with pytest.raises(BatchError, match="0010") as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged, accept_canon_change=True))
    assert "meta/rules.json" in str(e.value)


def test_a_canon_that_cannot_carry_the_chunks_labels_is_refused_even_with_the_flag(tmp_path):
    # The rule a chunk's labels point at is gone from the canon now: release_canon cannot narrow it, and no
    # flag makes the published rules say what the labels used.
    batch = _batch(tmp_path)

    def drop_stop_rule(doc):
        doc["rules"] = [r for r in doc["rules"] if r["rule_id"] != STOP_RULE]

    _edit_canon(tmp_path, "meta/rules.json", drop_stop_rule)
    _edit_canon(tmp_path, "meta/scenarios.json",
                lambda d: [s.update(rules=[x for x in s["rules"] if x != STOP_RULE]) for s in d["scenarios"]])
    with pytest.raises(BatchError, match="0010") as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged, accept_canon_change=True))
    assert STOP_RULE in str(e.value)


def test_a_scenario_the_canon_no_longer_has_is_refused_naming_the_chunk(tmp_path):
    batch = _batch(tmp_path)
    _edit_canon(tmp_path, "meta/scenarios.json",
                lambda d: [s.update(scenario_id="S9z") for s in d["scenarios"] if s["scenario_id"] == "S1d"])
    with pytest.raises(BatchError, match="0000"):
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged, accept_canon_change=True))


# ── the joint map: laid whole from the working tree, compared byte for byte ──
# The converter copies meta/joint_map/<robot>.yaml as it lies in the working tree, and every episode's joint
# names were read from the file of the day its chunk was made. The file is extracted from the asset, not
# written by hand: a difference is not a sentence, and the flag does not accept it.

def _remap_a_joint(text: str) -> str:
    """Two native joints given one canonical name: every key is still there, and the file parses."""
    line = "  left_hip_yaw: left_hip_yaw\n"
    assert text.count(line) == 1
    return text.replace(line, "  left_hip_yaw: right_hip_yaw\n")


def _add_a_comment(text: str) -> str:
    return text + "# a remark\n"


def _edit_joint_map(root: Path, edit, model: str = ROBOT) -> None:
    path = root / f"meta/joint_map/{model}.yaml"
    path.write_bytes(edit(path.read_bytes().decode("utf-8")).encode("utf-8"))


def _reseal(chunk: Path) -> None:
    """Sealed as it now is: the seal's own check passes, and what was changed is the chunk's."""
    write_seal(chunk, dict(read_seal(chunk), files=sealed_files(chunk)))


def _listed(said: str) -> list[str]:
    return [line.strip() for line in said.splitlines() if " — 청크 " in line]


@pytest.mark.parametrize("edit", [_remap_a_joint, _add_a_comment], ids=["a value", "a comment"])
def test_a_joint_map_changed_since_the_chunks_were_made_is_refused_before_anything_is_merged(tmp_path, edit):
    # A comment leaves the parsed document as it was: what is compared is the bytes that would ship.
    batch = _batch(tmp_path)
    _edit_joint_map(tmp_path, edit)
    with pytest.raises(BatchError, match=r"meta/joint_map/unitree_h1\.yaml") as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged))
    said = str(e.value)
    assert _listed(said) == [f"{JOINT_MAP}: (파일 전체) (바이트) — 청크 0000, 0010, 0020"]
    assert "차이 1건 — 문장이 아닌 것 1건 · 문장 0건" in said
    assert not (batch.root / ab.ASSEMBLING).exists() and not (batch.root / DATASET_DIR).exists()


@pytest.mark.parametrize("edit", [_remap_a_joint, _add_a_comment], ids=["a value", "a comment"])
def test_a_changed_joint_map_is_refused_with_the_flag_as_without(tmp_path, edit):
    # Nothing after the comparison would see either edit: the label check reads no joint map, and the
    # validator reads the joint names and that each has a canonical key -- which both edits leave alone.
    batch = _batch(tmp_path)
    _edit_joint_map(tmp_path, edit)
    lines = []
    with pytest.raises(BatchError, match=r"\(바이트\)") as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged, accept_canon_change=True,
                                            say=lines.append))
    said = str(e.value)
    assert "--accept-canon-change 로도 받아들이지 않습니다" in said and "다시 생성" in said
    assert not any("받아들입니다" in line for line in lines), lines
    assert not (batch.root / ab.ASSEMBLING).exists() and not (batch.root / DATASET_DIR).exists()


@pytest.mark.parametrize("accept", [False, True], ids=["without the flag", "with the flag"])
def test_a_changed_joint_map_among_reworded_sentences_is_counted_with_what_is_not_a_sentence(tmp_path, accept):
    # The tally is what the operator reads before giving the flag: "문장이 아닌 것 0건" must not be said
    # of a canon whose joint map differs, and the sentences beside it do not make the flag enough.
    batch = _batch(tmp_path)
    reworded = _reworded_notes(tmp_path)
    _edit_joint_map(tmp_path, _add_a_comment)
    with pytest.raises(BatchError) as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged, accept_canon_change=accept))
    said = str(e.value)
    listed = _listed(said)
    assert listed[0].startswith(f"{JOINT_MAP}: (파일 전체) (바이트)")
    assert len(listed) == reworded + 1 and f"문장이 아닌 것 1건 · 문장 {reworded}건" in said
    assert "--accept-canon-change 로도 받아들이지 않습니다" in said


def test_a_chunk_without_the_joint_map_it_shipped_is_refused(tmp_path):
    batch = _batch(tmp_path)
    chunk = _chunk(batch, 10)
    (chunk / DATASET_DIR / JOINT_MAP).unlink()
    _reseal(chunk)
    with pytest.raises(BatchError, match="0010") as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged, accept_canon_change=True))
    assert JOINT_MAP in str(e.value)


def test_a_joint_map_the_working_tree_no_longer_has_is_refused_before_anything_is_merged(tmp_path):
    # The converter would stop on it too -- after the merge, when the canon is laid.
    batch = _batch(tmp_path)
    (tmp_path / JOINT_MAP).unlink()
    with pytest.raises(BatchError, match=r"meta/joint_map/unitree_h1\.yaml") as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged, accept_canon_change=True))
    assert "작업 트리" in str(e.value)
    assert not (batch.root / ab.ASSEMBLING).exists() and not (batch.root / DATASET_DIR).exists()


def test_a_chunk_that_shipped_another_joint_map_than_the_rest_is_the_one_named(tmp_path):
    batch = _batch(tmp_path)
    chunk = _chunk(batch, 10)
    _edit_joint_map(chunk / DATASET_DIR, _add_a_comment)
    _reseal(chunk)
    with pytest.raises(BatchError) as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged))
    assert _listed(str(e.value)) == [f"{JOINT_MAP}: (파일 전체) (바이트) — 청크 0010"]


def test_each_chunk_is_held_to_the_joint_map_of_the_robot_its_own_episodes_name(tmp_path):
    batch = _batch(tmp_path)
    chunk = _chunk(batch, 20)
    _edit_episodes(lambda eps: eps.__setitem__("robot_model", OTHER_ROBOT))(chunk / DATASET_DIR)
    _lay_joint_map(chunk / DATASET_DIR, OTHER_ROBOT)
    _reseal(chunk)
    _lay_joint_map(tmp_path, OTHER_ROBOT)
    ab.assemble(batch, tmp_path, **FAKES)                    # each chunk's own robot, as shipped
    _edit_joint_map(tmp_path, _add_a_comment, OTHER_ROBOT)
    with pytest.raises(BatchError) as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged))
    assert _listed(str(e.value)) == [f"meta/joint_map/{OTHER_ROBOT}.yaml: (파일 전체) (바이트) — 청크 0020"]


def test_the_joint_map_of_a_robot_no_chunk_names_is_not_compared(tmp_path):
    # The repository holds the map of a robot this batch has no episode of: no chunk shipped it, the
    # assembly lays none, and it may change as it likes.
    batch = _batch(tmp_path)
    _lay_joint_map(tmp_path, OTHER_ROBOT)
    _edit_joint_map(tmp_path, lambda text: "not a joint map\n", OTHER_ROBOT)
    lines = []
    ab.assemble(batch, tmp_path, **dict(FAKES, say=lines.append))
    assert any("정본 대조" in line and "같음" in line for line in lines), lines


# ── the asset files: laid from the working tree's builds, held to the hashes each chunk sealed ──
# The converter copies the redistributable assets a batch's episodes reference out of the builds in the
# working tree, and holds each copy to the hash the canon pins (meta/assets.json; a texture's row, which
# its lock pins too). A chunk shipped its own copies, and its seal records their hashes. A build changed
# together with its pin passes every one of those checks, and is laid beside episodes that were sealed
# with another file.

REBUILT = [("build/a.usd", "assets/a.usd"), ("build/b.usd", "assets/b.usd"),
           ("build/textures/wood.jpg", "assets/textures/wood.jpg")]
REBUILT_IDS = ["a USD", "a USD with a texture", "a texture"]


def _rebuilt(build: str) -> dict:
    """The builds after ``build`` was made again with other bytes."""
    assert build in ASSET_FILES
    return dict(ASSET_FILES, **{build: b"built again\n"})


def _asset_refusal(tmp_path, batch, accept=False) -> str:
    """What the assembly says when it refuses, having merged nothing and made nothing."""
    with pytest.raises(BatchError) as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged, accept_canon_change=accept))
    assert not (batch.root / ab.ASSEMBLING).exists() and not (batch.root / DATASET_DIR).exists()
    return str(e.value)


@pytest.mark.parametrize("build, laid", REBUILT, ids=REBUILT_IDS)
def test_an_asset_pinned_to_other_bytes_since_the_chunks_were_made_is_refused_before_anything_is_merged(
        tmp_path, build, laid):
    batch = _batch(tmp_path)
    _lay_asset_canon(tmp_path, _rebuilt(build))             # the build and every pin of it, changed together
    said = _asset_refusal(tmp_path, batch)
    assert _listed(said) == [f"{laid}: (파일 전체) (바이트) — 청크 0000, 0010, 0020"]
    assert "차이 1건 — 문장이 아닌 것 1건 · 문장 0건" in said


@pytest.mark.parametrize("build, laid", REBUILT, ids=REBUILT_IDS)
def test_an_asset_pinned_to_other_bytes_is_refused_with_the_flag_as_without(tmp_path, build, laid):
    # Nothing after the comparison holds an asset file to the chunk's: the converter and the validator
    # hold it to the pin, and the pin changed with it.
    batch = _batch(tmp_path)
    _lay_asset_canon(tmp_path, _rebuilt(build))
    lines = []
    with pytest.raises(BatchError, match=re.escape(f"{laid}: (파일 전체) (바이트)")) as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged, accept_canon_change=True,
                                            say=lines.append))
    said = str(e.value)
    assert "--accept-canon-change 로도 받아들이지 않습니다" in said and "다시 생성" in said
    assert "usd_sha256" in said, "자산의 차이를 어디서 되돌리는지 말하지 않습니다"
    assert not any("받아들입니다" in line for line in lines), lines
    assert not (batch.root / ab.ASSEMBLING).exists() and not (batch.root / DATASET_DIR).exists()


@pytest.mark.parametrize("accept", [False, True], ids=["without the flag", "with the flag"])
def test_an_asset_pinned_to_other_bytes_among_reworded_sentences_is_counted_with_what_is_not_a_sentence(
        tmp_path, accept):
    batch = _batch(tmp_path)
    reworded = _reworded_notes(tmp_path)
    _lay_asset_canon(tmp_path, _rebuilt("build/a.usd"))
    said = _asset_refusal(tmp_path, batch, accept)
    listed = _listed(said)
    assert listed[0].startswith("assets/a.usd: (파일 전체) (바이트)")
    assert len(listed) == reworded + 1 and f"문장이 아닌 것 1건 · 문장 {reworded}건" in said
    assert "--accept-canon-change 로도 받아들이지 않습니다" in said


def test_a_chunk_that_shipped_another_asset_file_than_the_rest_is_the_one_named(tmp_path):
    batch = _batch(tmp_path)
    chunk = _chunk(batch, 10)
    (chunk / DATASET_DIR / "assets/textures/wood.jpg").write_bytes(b"another grain\n")
    _reseal(chunk)
    assert _listed(_asset_refusal(tmp_path, batch)) == [
        "assets/textures/wood.jpg: (파일 전체) (바이트) — 청크 0010"]


@pytest.mark.parametrize("accept", [False, True], ids=["without the flag", "with the flag"])
def test_an_asset_the_canon_no_longer_lets_ship_is_refused(tmp_path, accept):
    # Every chunk shipped the file, and the assembled distribution would be without it.
    batch = _batch(tmp_path)
    _lay_asset_canon(tmp_path, edit=lambda doc: doc["scene"]["a"].update(redistributed=False))
    assert _listed(_asset_refusal(tmp_path, batch, accept)) == [
        "assets/a.usd: (파일 전체) (청크에만 있음) — 청크 0000, 0010, 0020"]


@pytest.mark.parametrize("accept", [False, True], ids=["without the flag", "with the flag"])
def test_a_file_among_a_chunks_assets_that_the_canon_does_not_pin_is_refused(tmp_path, accept):
    batch = _batch(tmp_path)
    chunk = _chunk(batch, 20)
    (chunk / DATASET_DIR / "assets/textures/stray.png").write_bytes(b"pinned by nothing\n")
    _reseal(chunk)
    assert _listed(_asset_refusal(tmp_path, batch, accept)) == [
        "assets/textures/stray.png: (파일 전체) (청크에만 있음) — 청크 0020"]


@pytest.mark.parametrize("accept", [False, True], ids=["without the flag", "with the flag"])
def test_an_asset_file_the_canon_would_lay_that_a_chunk_did_not_ship_is_refused(tmp_path, accept):
    batch = _batch(tmp_path)
    chunk = _chunk(batch, 10)
    (chunk / DATASET_DIR / "assets/textures/wood.jpg").unlink()
    _reseal(chunk)
    assert _listed(_asset_refusal(tmp_path, batch, accept)) == [
        "assets/textures/wood.jpg: (파일 전체) (지금 정본에만 있음) — 청크 0010"]


@pytest.mark.parametrize("accept", [False, True], ids=["without the flag", "with the flag"])
def test_an_asset_the_canon_lets_ship_only_now_is_refused(tmp_path, accept):
    # No chunk holds a copy to hold the working tree's build to. The USD was not pinned when the chunks were
    # made, and its pin is a difference of its own: an asset that ships on one side only has its pin compared.
    batch = _batch(tmp_path)
    _lay_asset_canon(tmp_path, edit=lambda doc: doc["scene"]["kept"].update(
        redistributed=True, usd_sha256=_sha(b"the build kept back\n")))
    assert _listed(_asset_refusal(tmp_path, batch, accept)) == [
        "assets/kept.usd: (파일 전체) (지금 정본에만 있음) — 청크 0000, 0010, 0020",
        f"{ASSET_CANON}: scene.kept.usd_sha256 (지금 정본에만 있음) — 청크 0000, 0010, 0020"]


def test_each_chunk_is_held_to_the_assets_its_own_episodes_reference(tmp_path):
    batch = _batch(tmp_path)
    chunk = _chunk(batch, 20)
    _edit_episodes(lambda eps: eps.__setitem__("scene_assets", json.dumps([{"path": "build/a.usd"}])))(
        chunk / DATASET_DIR)
    (chunk / DATASET_DIR / "assets/b.usd").unlink()
    shutil.rmtree(chunk / DATASET_DIR / "assets/textures")
    _reseal(chunk)
    ab.assemble(batch, tmp_path, **FAKES)                    # the chunk without the second USD is not asked for it
    _lay_asset_canon(tmp_path, _rebuilt("build/b.usd"))
    with pytest.raises(BatchError) as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged))
    assert _listed(str(e.value)) == ["assets/b.usd: (파일 전체) (바이트) — 청크 0000, 0010"]


def test_a_chunk_is_held_to_the_assets_any_of_its_episodes_references(tmp_path):
    # The episodes of one chunk need not reference the same assets. What the converter laid for the chunk
    # is what any of them references: read from the first episode alone, the files of the others are
    # files nothing asks for.
    def only_the_first_usd_in_the_first_episode(eps):
        cells = list(eps["scene_assets"])
        cells[0] = json.dumps([{"path": "build/a.usd"}])
        eps["scene_assets"] = cells

    batch = _batch(tmp_path)
    chunk = _chunk(batch, 20)
    _edit_episodes(only_the_first_usd_in_the_first_episode)(chunk / DATASET_DIR)
    _reseal(chunk)
    lines = []
    ab.assemble(batch, tmp_path, **dict(FAKES, say=lines.append))
    assert any("자산 파일 9건" in line and "같음" in line for line in lines), lines


def test_chunks_whose_assets_cannot_ship_together_are_refused_before_anything_is_merged(tmp_path):
    # Each chunk is held to the assets its own episodes reference, and the lay asks once, for all of them.
    # Two USDs of one file name ship from a chunk each and cannot lie in one assets/: the converter refuses
    # them -- after the merge, unless the check has asked for them together.
    builds = {"first": b"the first build\n", "second": b"the second build\n"}

    def two_usds_of_one_name(doc):
        for name, data in builds.items():
            doc["scene"][name] = {"usd_path": f"{name}/same.usd", "redistributed": True, "usd_sha256": _sha(data)}

    def ships(chunk, name):
        root = chunk / DATASET_DIR
        _edit_episodes(lambda eps: eps.__setitem__("scene_assets", json.dumps([{"path": f"{name}/same.usd"}])))(root)
        shutil.rmtree(root / "assets")
        (root / "assets").mkdir()
        (root / "assets/same.usd").write_bytes(builds[name])
        _lay_asset_canon(root, edit=two_usds_of_one_name, lock=False)    # the inventory that let it ship
        _reseal(chunk)

    batch = _batch(tmp_path)
    _lay_asset_canon(tmp_path, edit=two_usds_of_one_name)
    for start in _starts(batch):
        ships(_chunk(batch, start), "first")
    lines = []
    ab.assemble(batch, tmp_path, **dict(FAKES, say=lines.append))       # every chunk the same USD: nothing to refuse
    assert any("자산 파일 3건" in line and "같음" in line for line in lines), lines
    ships(_chunk(batch, 10), "second")
    with pytest.raises(BatchError, match="assets/same.usd") as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged, accept_canon_change=True))
    said = str(e.value)
    assert "같은 이름" in said and "first" in said and "second" in said and "함께" in said


ROBOT_FOLDER = {"h1.usd": b"#usda 1.0\n# the robot\n", "instanceable_meshes.usd": b"#usda 1.0\n# its meshes\n",
                "LICENSE.txt": b"BSD 3-Clause License"}


def _robot_ships(doc):
    """The robot as the H1 ships: its own folder assets/unitree_h1/, every file a row of ship.files."""
    doc["robots"][ROBOT].update(
        redistributed=True, usd_sha256=_sha(ROBOT_FOLDER["h1.usd"]),
        ship={"dir": "unitree_h1", "source_dir": "build/unitree_h1", "license_file": "LICENSE.txt",
              "files": [{"file": n, "sha256": _sha(d), "license": "BSD-3-Clause",
                         "url": f"{{ISAACLAB_NUCLEUS_DIR}}/Robots/Unitree/H1/{n}"} for n, d in ROBOT_FOLDER.items()]})


def test_sealed_assets_include_the_robot_folder(tmp_path):
    """A chunk whose robot shipped as its folder sealed the files under assets/unitree_h1/: sealed_assets reads
    them with the rest, and the assembly holds each to the row the canon pins -- a licence file that is another
    one is the chunk named."""
    batch = _batch(tmp_path)
    _lay_asset_canon(tmp_path, edit=_robot_ships)
    for start in _starts(batch):
        root = _chunk(batch, start) / DATASET_DIR
        (root / "assets/unitree_h1").mkdir(parents=True)
        for name, data in ROBOT_FOLDER.items():
            (root / "assets/unitree_h1" / name).write_bytes(data)
        _lay_asset_canon(root, edit=_robot_ships, lock=False)        # the inventory that let it ship
        _reseal(_chunk(batch, start))
    sealed = ab.sealed_assets(read_seal(_chunk(batch, 0)))
    assert {rel for rel in sealed if rel.startswith("assets/unitree_h1/")} == {
        f"assets/unitree_h1/{name}" for name in ROBOT_FOLDER}
    assert sealed["assets/unitree_h1/LICENSE.txt"] == _sha(ROBOT_FOLDER["LICENSE.txt"])
    lines = []
    ab.assemble(batch, tmp_path, **dict(FAKES, say=lines.append))
    held = len(_starts(batch)) * (len(_shipped(SCENE_ASSETS)) + len(ROBOT_FOLDER))
    assert held == 18
    assert any(f"자산 파일 {held}건" in line and "같음" in line for line in lines), lines

    chunk = _chunk(batch, 10)
    (chunk / DATASET_DIR / "assets/unitree_h1/LICENSE.txt").write_bytes(b"another licence\n")
    _reseal(chunk)
    with pytest.raises(BatchError) as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged))
    assert _listed(str(e.value)) == ["assets/unitree_h1/LICENSE.txt: (파일 전체) (바이트) — 청크 0010"]


def test_an_asset_no_chunk_references_is_not_compared(tmp_path):
    # The canon lists every asset the project has made, and a batch uses some. One these episodes never
    # loaded may be rebuilt, repinned or left half declared: no chunk shipped it, the assembly lays none.
    batch = _batch(tmp_path)
    _lay_asset_canon(tmp_path, edit=lambda doc: doc["scene"].update(
        other={"usd_path": "build/other.usd", "redistributed": True}))     # no pin: refused if it were to ship
    lines = []
    ab.assemble(batch, tmp_path, **dict(FAKES, say=lines.append))
    assert any("정본 대조" in line and "같음" in line for line in lines), lines


def test_a_change_of_the_asset_canon_that_moves_no_shipped_byte_is_not_a_difference(tmp_path):
    # meta/assets.json is laid whole and is not compared as a document: a note, the record of a check
    # made on the same bytes, the description may change after the chunks were made. Every chunk carries
    # the document as it was, so a comparison of the two documents would have something to refuse.
    def edit(doc):
        doc["description"] = "다듬은 설명"
        doc["scene"]["a"].update(note="다듬은 문장", usd_checked={"sha256": doc["scene"]["a"]["usd_sha256"],
                                                              "checked_at": "2026-01-01"})

    batch = _batch(tmp_path)
    _lay_asset_canon(tmp_path, edit=edit)
    shipped = [(_chunk(batch, start) / DATASET_DIR / ASSET_CANON).read_bytes() for start in _starts(batch)]
    assert len(shipped) == 3 and all(doc != (tmp_path / ASSET_CANON).read_bytes() for doc in shipped)
    lines = []
    ab.assemble(batch, tmp_path, **dict(FAKES, say=lines.append))
    assert any("정본 대조" in line and "같음" in line for line in lines), lines


def _unpin_a_usd(doc):
    del doc["scene"]["a"]["usd_sha256"]


def _rows_that_are_not_the_locks(doc):
    doc["scene"]["b"]["bundle"]["files"][0]["sha256"] = _sha(b"another grain\n")


@pytest.mark.parametrize("edit, phrase", [(_unpin_a_usd, "usd_sha256"), (_rows_that_are_not_the_locks, "bundle.files")],
                         ids=["a USD without a pin", "texture rows that are not the lock's"])
def test_an_asset_canon_that_cannot_say_what_a_chunk_ships_is_refused_naming_the_chunk(tmp_path, edit, phrase):
    # The converter would stop on it too -- after the merge, when the assets are laid.
    batch = _batch(tmp_path)
    _lay_asset_canon(tmp_path, edit=edit)
    said = _asset_refusal(tmp_path, batch, accept=True)
    assert "청크 0000" in said and phrase in said


@pytest.mark.parametrize("leave", ["nothing", "not a document"])
def test_a_working_tree_whose_asset_canon_cannot_be_read_is_refused_before_anything_is_merged(tmp_path, leave):
    batch = _batch(tmp_path)
    (tmp_path / ASSET_CANON).unlink()
    if leave == "not a document":
        (tmp_path / ASSET_CANON).write_text("{", encoding="utf-8")
    assert ASSET_CANON in _asset_refusal(tmp_path, batch, accept=True)


def test_a_batch_whose_episodes_reference_no_asset_that_ships_is_assembled(tmp_path):
    batch = _batch(tmp_path, scene_assets=(KEPT_BACK,))
    assert not any((_chunk(batch, start) / DATASET_DIR / "assets").exists() for start in _starts(batch))
    lines = []
    ab.assemble(batch, tmp_path, **dict(FAKES, say=lines.append))
    assert any("정본 대조" in line and "같음" in line for line in lines), lines
    assert any("자산 파일 0건" in line for line in lines), lines


def test_the_assembly_says_how_many_asset_files_it_held_to_the_canon(tmp_path):
    # A comparison that found nothing to compare reads the same as one that found everything equal.
    batch = _batch(tmp_path)
    lines = []
    ab.assemble(batch, tmp_path, **dict(FAKES, say=lines.append))
    held = len(_starts(batch)) * len(_shipped(SCENE_ASSETS))
    assert held == 9
    assert any(f"자산 파일 {held}건" in line and "같음" in line for line in lines), lines


# ── the pin of an asset referenced and not redistributed: held to the inventory each chunk shipped ──
# No file of the robot's USD ships -- its licence lets it be referred to, by its path and its hash -- so
# no seal records its bytes. What a distribution says of it is the pin in meta/assets.json, which the
# converter lays whole from the working tree: the hash of the USD the episodes were generated with, as
# the inventory of the day the chunk was made had it. A pin changed since is laid beside episodes that
# were not made with that USD, and nothing after the comparison holds it to anything.

ROBOT_PIN = f"{ASSET_CANON}: robots.{ROBOT}.usd_sha256"


def _repin_the_robot(doc):
    doc["robots"][ROBOT]["usd_sha256"] = _sha(b"another robot\n")


def test_a_robot_usd_pinned_to_another_hash_since_the_chunks_were_made_is_refused_before_anything_is_merged(
        tmp_path):
    batch = _batch(tmp_path)
    _lay_asset_canon(tmp_path, edit=_repin_the_robot)
    said = _asset_refusal(tmp_path, batch)
    assert _listed(said) == [f"{ROBOT_PIN} (문자열 값) — 청크 0000, 0010, 0020"]
    assert "차이 1건 — 문장이 아닌 것 1건 · 문장 0건" in said


def test_a_robot_usd_pinned_to_another_hash_is_refused_with_the_flag_as_without(tmp_path):
    # The label check reads no asset inventory, and the validator reads the one the assembly laid.
    batch = _batch(tmp_path)
    _lay_asset_canon(tmp_path, edit=_repin_the_robot)
    lines = []
    with pytest.raises(BatchError, match=re.escape(f"{ROBOT_PIN} (문자열 값)")) as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged, accept_canon_change=True,
                                            say=lines.append))
    said = str(e.value)
    assert "--accept-canon-change 로도 받아들이지 않습니다" in said and "다시 생성" in said
    assert "참조만" in said, "싣지 않는 자산의 해시를 어디서 되돌리는지 말하지 않습니다"
    assert not any("받아들입니다" in line for line in lines), lines
    assert not (batch.root / ab.ASSEMBLING).exists() and not (batch.root / DATASET_DIR).exists()


@pytest.mark.parametrize("accept", [False, True], ids=["without the flag", "with the flag"])
def test_a_repinned_robot_usd_among_reworded_sentences_is_counted_with_what_is_not_a_sentence(tmp_path, accept):
    batch = _batch(tmp_path)
    reworded = _reworded_notes(tmp_path)
    _lay_asset_canon(tmp_path, edit=_repin_the_robot)
    said = _asset_refusal(tmp_path, batch, accept)
    listed = _listed(said)
    assert listed[0].startswith(f"{ROBOT_PIN} (문자열 값)")
    assert len(listed) == reworded + 1 and f"문장이 아닌 것 1건 · 문장 {reworded}건" in said


def test_a_chunk_whose_inventory_pinned_another_robot_usd_than_the_rest_is_the_one_named(tmp_path):
    batch = _batch(tmp_path)
    chunk = _chunk(batch, 10)
    _lay_asset_canon(chunk / DATASET_DIR, edit=_repin_the_robot, lock=False)
    _reseal(chunk)
    assert _listed(_asset_refusal(tmp_path, batch)) == [f"{ROBOT_PIN} (문자열 값) — 청크 0010"]


def _unpin_the_robot(doc):
    del doc["robots"][ROBOT]["usd_sha256"]


def _forget_the_robot(doc):
    del doc["robots"][ROBOT]


def _pin_the_kept_usd(doc):
    doc["scene"]["kept"]["usd_sha256"] = _sha(b"the build kept back\n")


@pytest.mark.parametrize("accept", [False, True], ids=["without the flag", "with the flag"])
@pytest.mark.parametrize("edit, line", [
    (_unpin_the_robot, f"{ROBOT_PIN} (청크에만 있음)"),
    (_forget_the_robot, f"{ROBOT_PIN} (청크에만 있음)"),
    (_pin_the_kept_usd, f"{ASSET_CANON}: scene.kept.usd_sha256 (지금 정본에만 있음)"),
], ids=["the pin dropped", "the entry dropped", "a pin where the chunk had none"])
def test_a_pin_of_an_asset_that_does_not_ship_on_one_side_only_is_refused(tmp_path, edit, line, accept):
    batch = _batch(tmp_path)
    _lay_asset_canon(tmp_path, edit=edit)
    assert _listed(_asset_refusal(tmp_path, batch, accept)) == [f"{line} — 청크 0000, 0010, 0020"]


def test_a_chunk_whose_inventory_does_not_name_an_asset_its_episodes_reference_is_refused(tmp_path):
    # The converter stops on an asset its canon does not know, so a chunk it made names every one; a
    # chunk that does not has no pin to hold the working tree's to.
    batch = _batch(tmp_path)
    chunk = _chunk(batch, 10)
    _lay_asset_canon(chunk / DATASET_DIR, edit=_forget_the_robot, lock=False)
    _reseal(chunk)
    assert _listed(_asset_refusal(tmp_path, batch)) == [f"{ROBOT_PIN} (지금 정본에만 있음) — 청크 0010"]


def _chunks_made_with(batch, edit) -> None:
    """Every chunk's inventory as ``edit`` leaves the stand-in one: the canon of the day they were made."""
    for start in _starts(batch):
        _lay_asset_canon(_chunk(batch, start) / DATASET_DIR, edit=edit, lock=False)
        _reseal(_chunk(batch, start))


SHADOW = "shadow/a.usd"


def test_an_asset_whose_file_another_asset_of_its_name_ships_now_is_held_by_its_pin(tmp_path):
    # Files are compared by the name they ship under. Two referenced USDs of one file name, one shipping in
    # the chunks' inventory and the other in the working tree's, ship the same bytes under that name -- and
    # the first is laid with a pin its episodes were not made with.
    def made(doc):
        doc["scene"]["shadow"] = {"usd_path": SHADOW, "redistributed": False, "usd_sha256": _sha(b"the shadow\n")}

    def now(doc):
        made(doc)
        doc["scene"]["a"].update(redistributed=False, usd_sha256=_sha(b"another a\n"))
        doc["scene"]["shadow"].update(redistributed=True, usd_sha256=_sha(ASSET_FILES["build/a.usd"]))

    batch = _batch(tmp_path, scene_assets=(*SCENE_ASSETS, SHADOW))
    _chunks_made_with(batch, made)
    _lay_asset_canon(tmp_path, edit=made)
    assert ab.check_canon(ab.sealed_chunks(batch), tmp_path, False, lambda line: None) == []    # as made: equal
    _lay_asset_canon(tmp_path, edit=now)
    assert _listed(_asset_refusal(tmp_path, batch, accept=True)) == [
        f"{ASSET_CANON}: scene.a.usd_sha256 (문자열 값) — 청크 0000, 0010, 0020",
        f"{ASSET_CANON}: scene.shadow.usd_sha256 (문자열 값) — 청크 0000, 0010, 0020"]


def test_two_pins_whose_entry_names_the_working_tree_reuses_are_two_differences(tmp_path):
    # Each difference is named by the chunk's own entry: named by the working tree's, two pins dropped from
    # two entries read as one line, and the chunks it is true of are each named twice.
    def made(doc):
        doc["scene"]["kept"]["usd_sha256"] = _sha(b"the build kept back\n")
        doc["scene"]["second"] = {"usd_path": SHADOW, "redistributed": False, "usd_sha256": _sha(b"the shadow\n")}

    def now(doc):
        doc["scene"]["kept"] = {"usd_path": SHADOW, "redistributed": False}

    batch = _batch(tmp_path, scene_assets=(*SCENE_ASSETS, SHADOW))
    _chunks_made_with(batch, made)
    _lay_asset_canon(tmp_path, edit=now)
    said = _asset_refusal(tmp_path, batch, accept=True)
    assert _listed(said) == [f"{ASSET_CANON}: scene.kept.usd_sha256 (청크에만 있음) — 청크 0000, 0010, 0020",
                             f"{ASSET_CANON}: scene.second.usd_sha256 (청크에만 있음) — 청크 0000, 0010, 0020"]
    assert "차이 2건 — 문장이 아닌 것 2건" in said


@pytest.mark.parametrize("leave, phrase", [(None, "싣고 온 정본이 없습니다"), ("{", "읽지 못합니다"),
                                           ("[]", "JSON 객체가 아닙니다")],
                         ids=["nothing", "not a document", "not an object"])
def test_a_chunk_whose_asset_inventory_cannot_be_read_is_refused_naming_the_chunk(tmp_path, leave, phrase):
    # Read as an inventory without its entries, a list would be refused too -- for a pin only the working
    # tree has, which is not what is wrong with the chunk.
    batch = _batch(tmp_path)
    chunk = _chunk(batch, 10)
    (chunk / DATASET_DIR / ASSET_CANON).unlink()
    if leave is not None:
        (chunk / DATASET_DIR / ASSET_CANON).write_text(leave, encoding="utf-8")
    _reseal(chunk)
    said = _asset_refusal(tmp_path, batch, accept=True)
    assert "청크 0010" in said and ASSET_CANON in said and phrase in said


def test_the_pin_of_an_asset_no_chunk_references_is_not_compared(tmp_path):
    # The inventory names a robot this batch has no episode of: its pin may change as it likes.
    batch = _batch(tmp_path)
    _lay_asset_canon(tmp_path, edit=lambda doc: doc["robots"].update({OTHER_ROBOT: {
        "usd_path": "{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd", "redistributed": False,
        "usd_sha256": _sha(b"a robot no episode ran\n")}}))
    lines = []
    ab.assemble(batch, tmp_path, **dict(FAKES, say=lines.append))
    assert any("정본 대조" in line and "같음" in line for line in lines), lines


def test_an_asset_that_ships_on_one_side_only_is_not_counted_among_the_pins_held(tmp_path):
    # It stopped shipping and kept its pin: its file is the difference, and its pin is not one of an asset
    # that ships on neither side -- which is what the count says it counts.
    batch = _batch(tmp_path)
    _lay_asset_canon(tmp_path, edit=lambda doc: doc["scene"]["a"].update(redistributed=False))
    lines = []
    with pytest.raises(BatchError, match=re.escape("assets/a.usd: (파일 전체) (청크에만 있음)")):
        ab.assemble(batch, tmp_path, **dict(FAKES, aggregate=_never_merged, say=lines.append))
    assert any("고정 해시 3건" in line and "같음" in line for line in lines), lines


def test_the_assembly_says_how_many_pins_of_assets_that_do_not_ship_it_held(tmp_path):
    # The robot is pinned in every chunk; the USD kept back has no pin, and is not counted as one held.
    batch = _batch(tmp_path)
    lines = []
    ab.assemble(batch, tmp_path, **dict(FAKES, say=lines.append))
    assert any("고정 해시 3건" in line and "같음" in line for line in lines), lines


def test_canon_changes_are_named_by_rule_and_scenario_id_and_kind():
    shipped = {"rules": [{"rule_id": "R-1", "v": 1.0, "note": "a note"}, {"rule_id": "R-2", "v": 2}],
               "limits": {"BR-01": {"force": 1}}, "description": "same"}
    wanted = {"rules": [{"rule_id": "R-2", "v": 2}, {"rule_id": "R-1", "v": 1.5, "note": "another note"}, {"rule_id": "R-3"}],
              "limits": {"BR-01": {"force": 2}}, "description": "same"}
    assert ab.canon_changes(shipped, shipped) == []
    assert ab.canon_changes(shipped, wanted) == [
        "limits.BR-01.force (숫자)", "rules[R-1].note (문장)", "rules[R-1].v (숫자)", "rules[R-3] (지금 정본에만 있음)"]


def test_a_string_that_code_reads_is_not_called_a_sentence():
    """A field name in a rule's inputs, a unit, a rule id in a scenario's list, an enum value: two
    strings differ, and it is not a rewording. The operator accepts a changed canon only when every
    line says "(문장)" -- so only a string under a key written for a person, and one that reads as
    prose, is called that."""
    shipped = {"rules": [{"rule_id": "R-1", "inputs": ["observation.contact.force"], "unit": "N", "note": "a note"}],
               "scenarios": [{"scenario_id": "S1", "rules": ["R-1"], "source": "sim", "objective": "walk straight"}]}
    wanted = {"rules": [{"rule_id": "R-1", "inputs": ["observation.root_vel"], "unit": "m", "note": "a note"}],
              "scenarios": [{"scenario_id": "S1", "rules": ["R-2"], "source": "real", "objective": "walk straight"}]}
    assert ab.canon_changes(shipped, wanted) == [
        "rules[R-1].inputs[0] (문자열 값)", "rules[R-1].unit (문자열 값)",
        "scenarios[S1].rules[0] (문자열 값)", "scenarios[S1].source (문자열 값)"]
    # `source` and `coords` are sentences in one place only, the bbox block of the camera canon; a camera's
    # mount.source is the name of the link it is read from, whatever it looks like
    assert ab.canon_changes({"bbox": {"source": "from the mask", "coords": "x then y"}},
                            {"bbox": {"source": "from the segmentation mask", "coords": "x, then y"}}) == [
        "bbox.coords (문장)", "bbox.source (문장)"]
    assert ab.canon_changes({"cameras": {"head": {"mount": {"source": "d435_rgb_module_link"}}}},
                            {"cameras": {"head": {"mount": {"source": "d435 rgb module link"}}}}) == [
        "cameras.head.mount.source (문자열 값)"]
    assert ab.canon_changes({"source": "sim"}, {"source": "a real robot"}) == ["source (문자열 값)"]
    # the place is the whole path, not its ending: a `bbox` further down, or a key that only ends like it, is not that block
    assert ab.canon_changes({"cameras": {"head": {"bbox": {"source": "from the mask"}}}},
                            {"cameras": {"head": {"bbox": {"source": "from the other mask"}}}}) == [
        "cameras.head.bbox.source (문자열 값)"]
    assert ab.canon_changes({"xbbox": {"source": "a b"}}, {"xbbox": {"source": "a c"}}) == ["xbbox.source (문자열 값)"]
    assert ab.canon_changes({"coords": "world"}, {"coords": "camera optical frame"}) == ["coords (문자열 값)"]
    assert ab.canon_changes({"note": "a"}, {"note": "b"}) == ["note (문자열 값)"]
    # robot_task reads like a sentence and is one -- but the converter writes it into every frame as the task
    assert ab.canon_changes({"robot_task": "walk in a straight line"}, {"robot_task": "walk in a line"}) == [
        "robot_task (문자열 값)"]


def _reworded(value):
    if isinstance(value, dict):
        return {k: _reworded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_reworded(v) for v in value]
    return value + " or something else" if isinstance(value, str) else value


def _strings(value) -> int:
    if isinstance(value, dict):
        return sum(_strings(v) for v in value.values())
    if isinstance(value, list):
        return sum(_strings(v) for v in value)
    return int(isinstance(value, str))


def test_a_string_in_a_table_the_validator_compares_whole_is_a_value():
    """meta/safety_schema.json holds tables that the validator compares, whole, with its own (C1).
    A key in them is the name of a member of `notes` -- one member is called `reason` -- and its
    string declares a type: `"string"` becoming `"integer | null"` is not a rewording, though the
    key is one written for a person elsewhere and the new text has a space. The notes inside those
    tables are compared like the rest. The tables are found, not listed: every table of the canon
    that equals the validator's constant of the same name."""
    import validator.validate_dataset as vd

    canon = json.loads((REPO / "meta" / "safety_schema.json").read_text(encoding="utf-8"))
    compared = {name: table for name, table in canon.items()
                if isinstance(table, dict) and getattr(vd, name.upper(), None) == table}
    assert {"notes_member_shapes", "notes_bounds"} <= set(compared), sorted(compared)
    lines = ab.canon_changes(compared, _reworded(compared))
    assert len(lines) == _strings(compared) >= 20, lines
    assert "notes_member_shapes.not_evaluated[].reason (문자열 값)" in lines
    assert [line for line in lines if not line.endswith("(문자열 값)")] == []
    # outside those tables the same canon's notes are still sentences
    note = {"machine_readable_note": canon["machine_readable_note"]}
    assert ab.canon_changes(note, _reworded(note)) == ["machine_readable_note (문장)"]


def test_a_changed_sentence_inside_a_plain_list_is_named_as_a_sentence():
    """A list without ids (open_issues, unit_changes) used to differ as one value, "형 또는 목록". The
    operator is told to accept a changed canon only when every line says "(문장)", so a reworded
    note inside such a list read as something else than it was."""
    shipped = {"open_issues": ["one thing", "another thing"], "unit_changes": [{"path": "x", "reason": "the old reason"}],
               "limits": [1, 2]}
    wanted = {"open_issues": ["one thing", "another thing, reworded"],
              "unit_changes": [{"path": "x", "reason": "the new reason"}], "limits": [1, 3]}
    assert ab.canon_changes(shipped, shipped) == []
    assert ab.canon_changes(shipped, wanted) == [
        "limits[1] (숫자)", "open_issues[1] (문장)", "unit_changes[0].reason (문장)"]


def test_a_plain_list_of_another_length_is_still_one_value():
    assert ab.canon_changes({"open_issues": ["a"]}, {"open_issues": ["a", "b"]}) == ["open_issues (형 또는 목록)"]
    assert ab.canon_changes({"open_issues": ["a"]}, {"open_issues": "a"}) == ["open_issues (형 또는 목록)"]


def test_canon_changes_are_not_fooled_by_the_order_of_the_entries_or_of_the_keys():
    a = {"rules": [{"rule_id": "R-1", "x": 1, "y": 2}, {"rule_id": "R-2"}]}
    b = {"rules": [{"rule_id": "R-2"}, {"y": 2, "x": 1, "rule_id": "R-1"}]}
    assert ab.canon_changes(a, b) == []


def test_the_flag_exists_and_main_hands_it_to_the_assembly():
    assert ab.build_parser().parse_args(["--batch", "x"]).accept_canon_change is False
    assert ab.build_parser().parse_args(["--batch", "x", "--accept-canon-change"]).accept_canon_change is True
    assert "accept_canon_change=args.accept_canon_change" in Path(ab.__file__).read_text(encoding="utf-8")


# ── the labels the assembly carries follow from it ──
# What --accept-canon-change leaves unguarded: the flag waives the comparison of the two narrowed canon
# documents (not of the joint map, refused with it all the same), the canon laid is the one of now, and the
# labels are the chunks'. The check labels every episode again from the assembled
# files alone (generation/labeling/check_release_labels.py) -- on a release the production path made
# (release_fixture), since the stand-in chunks above hold nothing a rule reads.

def _labelled(tmp_path):
    return release_fixture.write_release(tmp_path / "assembled", release_fixture.eight())


def test_labels_that_follow_from_the_assembled_files_pass_and_are_counted(tmp_path):
    assert ab.verify_labels(_labelled(tmp_path)) == 8


def test_a_threshold_that_no_longer_gives_the_label_is_caught_naming_the_episode(tmp_path):
    # Episode 3 is a fail for 300 N on BR-05, twice the 140 N limit and more. With the limit laid as 1400 the
    # same frames are no fail -- the label was made with a value the release no longer carries.
    out = _labelled(tmp_path)

    def raised(doc):
        doc["contact_limits"]["BR-05"]["limits"]["force_quasistatic"]["value"] = 1400

    release_fixture.edit_json(out / "meta" / "rules.json", raised)
    with pytest.raises(BatchError, match="S1d_0003_seed4") as e:
        ab.verify_labels(out)
    said = str(e.value)
    assert "7 / 8" in said and "outcome" in said and "nominal" in said
    assert "다시 생성" in said
    assert "--accept-canon-change" not in said      # the check does not know whether the flag was used


def test_a_label_only_a_loader_item_does_not_give_is_said_to_be_of_that_reading(tmp_path):
    # 280.000001 N on the 2 x 140 N limit: a fail as stored, no fail as the float32 a LeRobot item carries.
    out = release_fixture.write_release(tmp_path / "assembled", [
        release_fixture.recording(0, touch=(120, "BR-05", 280.000001, "transient"))])
    with pytest.raises(BatchError, match="float32") as e:
        ab.verify_labels(out)
    assert "0 / 1" in str(e.value)


def test_an_episode_that_cannot_be_labelled_from_the_assembled_files_is_caught(tmp_path):
    out = _labelled(tmp_path)
    release_fixture.edit_notes(out, 5, hold_release_step=None)
    with pytest.raises(BatchError, match="S1d_0005_seed6") as e:
        ab.verify_labels(out)
    assert "7 / 8" in str(e.value) and "hold_release_step" in str(e.value)


def test_assembled_files_the_labels_cannot_be_read_from_are_caught(tmp_path):
    out = _labelled(tmp_path)
    (out / "meta" / "safety_schema.json").unlink()
    with pytest.raises(BatchError, match="safety_schema.json"):
        ab.verify_labels(out)


def test_every_episode_whose_label_differs_is_counted_though_few_are_shown(tmp_path):
    out = _labelled(tmp_path)
    for episode in range(8):
        release_fixture.edit_episode(out, episode, severity="S9")
    with pytest.raises(BatchError) as e:
        ab.verify_labels(out)
    said = str(e.value)
    assert "0 / 8" in said and said.count("severity") == ab.LABELS_SHOWN and f"외 {8 - ab.LABELS_SHOWN}건" in said


def test_the_assembly_checks_the_labels_when_given_no_stand_in(tmp_path):
    # The stand-in chunks hold no column a rule reads: the real check can only refuse them. Were the call
    # dropped from assemble(), or its default made a no-op, this assembly would land.
    batch = _batch(tmp_path)
    without = {k: v for k, v in FAKES.items() if k != "check_labels"}
    with pytest.raises(BatchError, match="라벨"):
        ab.assemble(batch, tmp_path, **without)
    assert sorted(p.name for p in batch.root.iterdir()) == ["batch.json", "chunks"]


def test_main_assembles_and_hands_the_assembly_no_stand_in_for_the_label_check(tmp_path, monkeypatch):
    # main() run, with lerobot stood in for (it is not in this environment) and the assembly watched:
    # called once, with the flag as parsed and without a check_labels of its own -- so the default,
    # which is the real check. A reading of main's source passed with main gutted.
    import inspect
    import sys
    import types

    for name, attribute in (("lerobot", None), ("lerobot.datasets", None),
                            ("lerobot.datasets.aggregate", "aggregate_datasets"),
                            ("lerobot.datasets.compute_stats", "aggregate_stats"),
                            ("lerobot.datasets.io_utils", "write_stats"),
                            ("lerobot.datasets.lerobot_dataset", "LeRobotDataset")):
        module = types.ModuleType(name)
        if attribute:
            setattr(module, attribute, lambda *a, **k: None)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    calls = []

    def watched(batch, repo_root, **stages):
        calls.append((batch, stages))
        return tmp_path / "dataset"

    monkeypatch.setattr(ab, "load_batch", lambda path: ("the batch at", path))
    monkeypatch.setattr(ab, "assemble", watched)
    assert ab.main(["--batch", str(tmp_path), "--accept-canon-change"]) == 0
    ((batch, stages),) = calls
    assert batch == ("the batch at", tmp_path)
    assert stages["accept_canon_change"] is True and "check_labels" not in stages
    monkeypatch.undo()
    assert inspect.signature(ab.assemble).parameters["check_labels"].default is ab.verify_labels


def _labels_refused(out):
    raise BatchError("실린 라벨이 조립한 배포본에서 다시 나오지 않습니다 — 일치 24 / 25")


def test_a_refused_label_check_names_the_canon_changes_the_assembly_accepted(tmp_path):
    batch = _batch(tmp_path)
    _edit_canon(tmp_path, "meta/scenarios.json", _reword_objective)
    with pytest.raises(BatchError, match="24 / 25") as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, accept_canon_change=True, check_labels=_labels_refused))
    said = str(e.value)
    assert "--accept-canon-change" in said and "scenarios[S1d].objective" in said


def test_the_accepted_changes_a_label_can_turn_on_are_named_first(tmp_path):
    # Found assembling two real chunks: twenty changes were accepted, the list is cut at twelve and is in
    # file order, so the one that moved the labels -- a number in meta/rules.json -- was not in it.
    batch = _batch(tmp_path)

    def many_new_keys(doc):
        for scenario in doc["scenarios"]:
            if scenario["scenario_id"] == "S1d":
                scenario.update({f"a_new_key_{k:02d}": f"words {k}" for k in range(ab.CHANGES_SHOWN + 3)})

    _edit_canon(tmp_path, "meta/scenarios.json", _reword_objective)
    _edit_canon(tmp_path, "meta/scenarios.json", many_new_keys)
    _edit_canon(tmp_path, "meta/rules.json", _raise_stop_time)
    with pytest.raises(BatchError, match="24 / 25") as e:
        ab.assemble(batch, tmp_path, **dict(FAKES, accept_canon_change=True, check_labels=_labels_refused))
    listed = str(e.value).split("받아들였습니다", 1)[1].splitlines()[1:]
    assert "meta/rules.json" in listed[0] and "robot_stop_time" in listed[0] and "(숫자)" in listed[0]
    assert "(문장)" not in "".join(listed[:ab.CHANGES_SHOWN])        # the sentence is the last thing to suspect
    assert listed[-1].strip().startswith("… 외") and len(listed) == ab.CHANGES_SHOWN + 1


def test_a_refused_label_check_does_not_speak_of_the_flag_when_nothing_was_accepted(tmp_path):
    for accept in (False, True):                 # the flag given but with nothing to accept is the same
        batch = _batch(tmp_path / str(accept))
        with pytest.raises(BatchError, match="24 / 25") as e:
            ab.assemble(batch, tmp_path / str(accept),
                        **dict(FAKES, accept_canon_change=accept, check_labels=_labels_refused))
        assert "--accept-canon-change" not in str(e.value)


def test_the_labels_are_checked_after_the_canon_is_laid_and_before_the_result_is_put_in_place(tmp_path):
    batch = _batch(tmp_path)
    ab.assemble(batch, tmp_path, **FAKES)                                    # a previous result to keep

    def card_of_the_failed_run(repo_root, out, sidecars):
        (Path(out) / "README.md").write_text("the card of the failed run", encoding="utf-8")
        return ["README.md"]

    seen = []

    def refuse(out):
        seen.append((Path(out) / "README.md").read_text(encoding="utf-8"))   # what lay() put there
        raise BatchError("실린 라벨이 다시 나오지 않습니다")

    with pytest.raises(BatchError, match="다시 나오지"):
        ab.assemble(batch, tmp_path, **dict(FAKES, check_labels=refuse, lay=card_of_the_failed_run))
    assert seen == ["the card of the failed run"]
    assert (batch.root / DATASET_DIR / "README.md").read_text(encoding="utf-8") == "card at assembly for 25 episodes"


def test_the_assembly_says_how_many_labels_it_derived_again(tmp_path):
    lines = []
    ab.assemble(_batch(tmp_path), tmp_path, **dict(FAKES, say=lines.append, check_labels=lambda out: 25))
    assert any("라벨 25건" in line for line in lines), lines


# ── several batches made at one generator commit: one release ──
# A batch that stopped at a chunk it can never seal: "first" planned ROW_PARTS -- chunks S1e-000 (37, 12),
# S1f-000 (3, 8), S1g-000 (44), S1e-001 (41), taking turns -- sealed the first, could not seal the second (its row
# LEFT_OUT fails wherever it is run) and never began the rest. "rest" is made at the same commit with every row the
# first did not seal but LEFT_OUT, in chunks of its own: S1e-000 (41), S1f-000 (3), S1g-000 (44).

LEFT_OUT = ("S1f", 8)


def _stopped_batch(tmp_path):
    first = _rows_batch(tmp_path, name="first")
    plans = first.chunks()
    (first.chunk_dir(plans[1]) / SEAL_FILE).unlink()
    for plan in plans[2:]:
        shutil.rmtree(first.chunk_dir(plan))
    assert LEFT_OUT in {(plans[1].scenario, row) for row in plans[1].rows}
    return first


def _sealed_rows(batch):
    return {(plan.scenario, row) for plan in batch.chunks() if read_seal(batch.chunk_dir(plan)) is not None
            for row in plan.rows}


def _the_rest(tmp_path, first, **made):
    """The rows ``first`` did not seal, less LEFT_OUT, as a batch "rest" of their own (``made``: _rows_batch's)."""
    done = _sealed_rows(first) | {LEFT_OUT}
    parts = [(scenario, tuple(row for row in rows if (scenario, row) not in done)) for scenario, rows in first.parts]
    return _rows_batch(tmp_path, parts=tuple((s, rows) for s, rows in parts if rows), name="rest", **made)


def _shipped_chunks(*batches):
    """(batch, plan) of every sealed chunk, the batches in the order given and each batch's chunks in its order."""
    return [(batch, plan) for batch in batches for plan in batch.chunks()
            if read_seal(batch.chunk_dir(plan)) is not None]


def _two_batches(tmp_path):
    first = _stopped_batch(tmp_path)
    return first, _the_rest(tmp_path, first)


def test_a_stopped_batch_and_the_rest_of_its_rows_assemble_into_one_release_without_the_row_left_out(tmp_path):
    first, rest = _two_batches(tmp_path)
    shipped = _shipped_chunks(first, rest)
    assert {batch.root.name for batch, _ in shipped} == {"first", "rest"}, "두 배치에서 오는 청크가 없습니다"
    out, lines, labelled = tmp_path / "release", [], []

    def labels(work):
        labelled.append(work)
        return len(ab.episodes_table(work))

    got = ab.assemble([first, rest], tmp_path, **dict(FAKES, out=out, drop=[LEFT_OUT], say=lines.append,
                                                      check_labels=labels))
    assert got == out and (out / EPISODES).is_file()
    eps = ab.episodes_table(out)
    # The first batch's sealed chunks, then the second's; within a chunk the converter's order.
    want = [eid for batch, plan in shipped for eid in ab.conversion_order(batch.episode_ids(plan))]
    assert [str(eid) for eid in eps["episode_id"]] == want
    # Every row planned once, but the one left out; every run_start_seq the row itself.
    row_of = {eid: (plan.scenario, row) for batch, plan in shipped for row, eid in zip(plan.rows, batch.episode_ids(plan))}
    assert sorted(row_of[eid] for eid in want) == sorted({(s, r) for s, rows in first.parts for r in rows} - {LEFT_OUT})
    assert [int(v) for v in eps["run_start_seq"]] == [row_of[eid][1] for eid in want]
    # The label check ran on the assembly before it was put in place, over every episode.
    assert labelled == [ab.beside(out)[0]]
    assert any(f"라벨 {len(want)}건" in line for line in lines), lines
    # The first lines name the chunks passed over -- each with why -- and the row left out.
    counted = next(i for i, line in enumerate(lines) if line.startswith(f"청크 {len(shipped)}개"))
    head = lines[:counted]
    for plan in first.chunks():
        if read_seal(first.chunk_dir(plan)) is None:
            why = "봉인되지 않음" if first.chunk_dir(plan).is_dir() else "디렉터리가 없음"
            assert any(f"first/{plan.key} ({why}" in line for line in head), (plan.key, head)
    assert any(f"{LEFT_OUT[0]}:{LEFT_OUT[1]}" in line for line in head), head
    # Nothing is left beside the release, and the batches, which are read, hold no release.
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(out.name)) == [out.name]
    assert not (first.root / DATASET_DIR).exists() and not (rest.root / DATASET_DIR).exists()


def test_a_row_no_sealed_chunk_holds_is_refused_naming_the_chunk_that_planned_it(tmp_path):
    first, rest = _two_batches(tmp_path)
    out = tmp_path / "release"
    with pytest.raises(BatchError) as e:
        ab.assemble([first, rest], tmp_path, **dict(FAKES, out=out, aggregate=_never_merged))
    said = str(e.value)
    (plan,) = [p for p in first.chunks() if p.scenario == LEFT_OUT[0] and LEFT_OUT[1] in p.rows]
    # One line, for the one chunk that planned the row; its other row, sealed by the second batch, is not named.
    assert [line.strip() for line in said.splitlines() if line.startswith("  ")] == [
        f"first/{plan.key} (봉인되지 않음 — {first.chunk_dir(plan)}): {LEFT_OUT[0]} 행 {LEFT_OUT[1]}"]
    assert "run_batch.py" in said and "--drop" in said
    assert not out.exists() and not list(tmp_path.glob(f"{out.name}.*"))


def test_a_row_every_chunk_left_unmade_is_named_under_each_chunk_that_planned_it(tmp_path):
    first, rest = _two_batches(tmp_path)
    plan = rest.chunks()[-1]
    shutil.rmtree(rest.chunk_dir(plan))                          # the second batch has not got there yet
    (theirs,) = [p for p in first.chunks() if p.scenario == plan.scenario and set(plan.rows) <= set(p.rows)]
    with pytest.raises(BatchError) as e:
        ab.release_chunks([first, rest], [LEFT_OUT])
    listed = [line.strip() for line in str(e.value).splitlines() if line.startswith("  ")]
    rows = ", ".join(str(row) for row in sorted(plan.rows))
    assert listed == [f"first/{theirs.key} (디렉터리가 없음): {plan.scenario} 행 {rows}",
                      f"rest/{plan.key} (디렉터리가 없음): {plan.scenario} 행 {rows}"]


def test_one_batch_with_a_chunk_its_drop_does_not_cover_is_refused_in_the_words_of_one_batch(tmp_path):
    # No other batch to cover it: the sentence of an assembly of one batch.
    first = _stopped_batch(tmp_path)
    plan = first.chunks()[1]
    with pytest.raises(BatchError, match=re.escape(f"청크 {plan.key} 이 봉인되지 않았습니다 ({first.chunk_dir(plan)}) "
                                                   "— run_batch.py 를 이어 돌리십시오")):
        ab.release_chunks([first], [LEFT_OUT])


def test_one_batch_and_no_drop_is_the_sealed_chunks_of_the_batch(tmp_path):
    batch = _rows_batch(tmp_path)
    chosen = ab.release_chunks([batch])
    assert list(chosen.chunks) == ab.sealed_chunks(batch)
    assert list(chosen.names) == [plan.key for plan in batch.chunks()] and not chosen.skipped and not chosen.dropped


def test_a_row_two_batches_sealed_is_refused_naming_both_chunks(tmp_path):
    first = _rows_batch(tmp_path, name="first")                   # every chunk sealed
    scenario, rows = first.parts[0]
    again = _rows_batch(tmp_path, parts=((scenario, rows[-1:]),), name="again")
    (mine,) = [p for p in first.chunks() if p.scenario == scenario and rows[-1] in p.rows]
    with pytest.raises(BatchError, match="한 번만") as e:
        ab.release_chunks([first, again])
    listed = [line.strip() for line in str(e.value).splitlines() if line.startswith("  ")]
    assert listed == [f"first/{mine.key} · again/{again.chunks()[0].key}: {scenario} 행 {rows[-1]}"]


def test_a_dropped_row_a_sealed_chunk_holds_is_refused(tmp_path):
    first, rest = _two_batches(tmp_path)
    plan = rest.chunks()[0]
    pair = (plan.scenario, plan.rows[0])
    with pytest.raises(BatchError, match="통째로") as e:
        ab.release_chunks([first, rest], [LEFT_OUT, pair])
    said = str(e.value)
    assert f"{pair[0]}:{pair[1]} (rest/{plan.key})" in said and f"{LEFT_OUT[0]}:{LEFT_OUT[1]}" not in said


@pytest.mark.parametrize("typo", ["another row", "another scenario"])
def test_a_dropped_row_no_batch_planned_is_refused(tmp_path, typo):
    first, rest = _two_batches(tmp_path)
    planned = {(s, r) for batch in (first, rest) for s, rows in batch.parts for r in rows}
    pair = (LEFT_OUT[0], LEFT_OUT[1] + 1) if typo == "another row" else ("S1x", LEFT_OUT[1])
    assert pair not in planned
    with pytest.raises(BatchError, match="계획하지 않았습니다") as e:
        ab.release_chunks([first, rest], [LEFT_OUT, pair])
    assert f"{pair[0]}:{pair[1]}" in str(e.value)


@pytest.mark.parametrize("made, field", [({"commit": "f" * 40}, "generator_commit"),
                                         ({"repo_id": "test/another"}, "repo_id")],
                         ids=["another generator commit", "another repo"])
def test_batches_of_another_generator_commit_or_repo_are_refused(tmp_path, made, field):
    first = _stopped_batch(tmp_path)
    rest = _the_rest(tmp_path, first, **made)
    assert getattr(rest, field) != getattr(first, field)
    with pytest.raises(BatchError, match=field) as e:
        ab.assemble([first, rest], tmp_path, **dict(FAKES, out=tmp_path / "release", drop=[LEFT_OUT],
                                                    aggregate=_never_merged))
    said = str(e.value)
    assert f"first {getattr(first, field)}" in said and f"rest {getattr(rest, field)}" in said
    assert not (tmp_path / "release").exists()


def test_several_batches_without_an_out_are_refused(tmp_path):
    first, rest = _two_batches(tmp_path)
    with pytest.raises(BatchError, match="--out"):
        ab.assemble([first, rest], tmp_path, **dict(FAKES, drop=[LEFT_OUT], aggregate=_never_merged))
    assert not (first.root / DATASET_DIR).exists() and not (rest.root / DATASET_DIR).exists()


@pytest.mark.parametrize("inside", [lambda first, rest: first.root / "release",
                                    lambda first, rest: rest.root / "chunks" / "release",
                                    lambda first, rest: rest.root],
                         ids=["the first batch", "the second batch's chunks", "the second batch itself"])
def test_an_out_inside_a_batch_is_refused_when_there_are_several(tmp_path, inside):
    first, rest = _two_batches(tmp_path)
    with pytest.raises(BatchError, match="--out") as e:
        ab.assemble([first, rest], tmp_path, **dict(FAKES, out=inside(first, rest), drop=[LEFT_OUT],
                                                    aggregate=_never_merged))
    assert "안입니다" in str(e.value)


def test_with_one_batch_an_out_inside_it_is_refused_but_its_own_dataset(tmp_path):
    batch = _rows_batch(tmp_path)
    for inside in (batch.root / "chunks" / "release", batch.root / "release", batch.root):
        with pytest.raises(BatchError, match=re.escape(f"--out {inside} 이 배치 {batch.root} 의 안입니다")):
            ab.release_dir([batch], inside)
    assert ab.release_dir([batch], batch.root / DATASET_DIR) == batch.root / DATASET_DIR
    assert ab.release_dir([batch], None) == batch.root / DATASET_DIR


def _tree(root: Path) -> dict:
    return {f.relative_to(root).as_posix(): f.read_bytes() for f in sorted(root.rglob("*")) if f.is_file()}


@pytest.mark.parametrize("where", ["its release", "a chunk's dataset"])
@pytest.mark.parametrize("given", ["one batch", "two batches"])
def test_an_out_inside_a_batch_not_given_is_refused_and_left_as_it_was(tmp_path, given, where):
    # Both hold meta/info.json, as an earlier assembly's result does: only that they are in a batch tells them apart.
    first, rest = _two_batches(tmp_path)
    other = _rows_batch(tmp_path, name="other")
    ab.assemble(other, tmp_path, **FAKES)
    out = other.root / DATASET_DIR if where == "its release" else other.chunk_dir(other.chunks()[0]) / DATASET_DIR
    assert (out / "meta/info.json").is_file()
    before = _tree(other.root)
    batches, drop = ([rest], ()) if given == "one batch" else ([first, rest], [LEFT_OUT])
    with pytest.raises(BatchError, match=re.escape(f"--out {out} 이 배치 {other.root} 의 안입니다")) as e:
        ab.assemble(batches, tmp_path, **dict(FAKES, out=out, drop=drop, aggregate=_never_merged))
    assert "주지 않은 배치" in str(e.value)
    assert _tree(other.root) == before
    assert not list(out.parent.glob(f"{out.name}.*"))


def test_one_batch_assembles_into_its_own_dataset_given_as_the_out(tmp_path):
    batch = _rows_batch(tmp_path)
    for card in ("the first", "the second"):                     # the second replaces the first, an earlier result
        out = ab.assemble(batch, tmp_path, **dict(FAKES, out=batch.root / DATASET_DIR,
                                                  lay=lambda r, o, sc, card=card: (Path(o) / "README.md").write_text(
                                                      card, encoding="utf-8") and ["README.md"]))
        assert out == batch.root / DATASET_DIR and (out / "README.md").read_text(encoding="utf-8") == card
    assert sorted(p.name for p in batch.root.iterdir()) == ["batch.json", "chunks", "dataset"]


def test_the_batches_ship_in_the_order_given_and_not_by_their_names_or_rows(tmp_path):
    first, rest = _two_batches(tmp_path)
    shipped = _shipped_chunks(rest, first)
    want = [eid for batch, plan in shipped for eid in ab.conversion_order(batch.episode_ids(plan))]
    row_of = {eid: (plan.scenario, row) for batch, plan in shipped for row, eid in zip(plan.rows, batch.episode_ids(plan))}
    by_name = [eid for batch, plan in _shipped_chunks(*sorted((rest, first), key=lambda b: b.root.name))
               for eid in ab.conversion_order(batch.episode_ids(plan))]
    assert want != by_name and want != sorted(want, key=row_of.get), "이 시험이 준 차례를 이름 · 행의 차례와 가르지 못합니다"
    lines = []
    out = ab.assemble([rest, first], tmp_path, **dict(FAKES, out=tmp_path / "release", drop=[LEFT_OUT], say=lines.append))
    assert [str(eid) for eid in ab.episodes_table(out)["episode_id"]] == want
    assert want[0] in rest.episode_ids(rest.chunks()[0])
    assert lines[0].startswith("배치 2개 — rest · first")


def test_with_one_batch_and_a_drop_the_first_lines_speak_of_no_other_batch(tmp_path):
    # Its chunk passed over holds only the row dropped: no other batch is there to cover anything.
    batch = _rows_batch(tmp_path)
    plan = batch.chunks()[-1]
    assert len(plan.rows) == 1
    (batch.chunk_dir(plan) / SEAL_FILE).unlink()
    lines = []
    ab.assemble(batch, tmp_path, **dict(FAKES, drop=[(plan.scenario, plan.rows[0])], say=lines.append))
    head = lines[:next(i for i, line in enumerate(lines) if line.startswith("청크 "))]
    assert any(f"{plan.key} (봉인되지 않음" in line for line in head), head
    assert any(f"{plan.scenario}:{plan.rows[0]}" in line for line in head), head
    assert not any("배치" in line for line in head), head


def test_an_out_that_is_there_and_not_an_assembly_is_refused_and_left_as_it_was(tmp_path):
    first, rest = _two_batches(tmp_path)
    out = tmp_path / "release"
    out.mkdir()
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(BatchError, match="앞선 조립의 결과"):
        ab.assemble([first, rest], tmp_path, **dict(FAKES, out=out, drop=[LEFT_OUT], aggregate=_never_merged))
    assert [p.name for p in out.iterdir()] == ["notes.txt"]
    assert (out / "notes.txt").read_text(encoding="utf-8") == "mine"


def test_an_out_reached_through_a_directory_that_is_not_there_is_judged_where_it_leads(tmp_path):
    # The system cannot walk missing/.. while missing is absent, so the path does not exist as written; once the merge
    # makes missing, it names the directory beside it -- the one the swap would put aside and remove.
    first, rest = _two_batches(tmp_path)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("mine", encoding="utf-8")
    out = tmp_path / "missing" / ".." / "victim"
    assert not out.exists() and out.resolve() == victim
    before = sorted(p.name for p in tmp_path.iterdir())
    for refused in (lambda: ab.release_dir([first, rest], out),
                    lambda: ab.assemble([first, rest], tmp_path, **dict(FAKES, out=out, drop=[LEFT_OUT],
                                                                        aggregate=_never_merged))):
        with pytest.raises(BatchError, match=re.escape(f"--out {out} 이 이미 있고 앞선 조립의 결과")):
            refused()
    assert _tree(victim) == {"keep.txt": b"mine"}
    assert sorted(p.name for p in tmp_path.iterdir()) == before and not (tmp_path / "missing").exists()


def test_an_out_reached_through_a_directory_that_is_not_there_to_an_assembly_replaces_that_assembly(tmp_path):
    first, rest = _two_batches(tmp_path)
    result = tmp_path / "release"
    ab.assemble([first, rest], tmp_path, **dict(FAKES, out=result, drop=[LEFT_OUT], say=lambda _: None))
    out = tmp_path / "missing" / ".." / "release"
    again = ab.assemble([first, rest], tmp_path, **dict(
        FAKES, out=out, drop=[LEFT_OUT], say=lambda _: None,
        lay=lambda repo_root, o, sidecars: (Path(o) / "README.md").write_text("again", encoding="utf-8") and ["README.md"]))
    assert again == result and (result / "README.md").read_text(encoding="utf-8") == "again"
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("release")) == ["release"]
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("named", ["where the link points", "the link"])
def test_with_one_batch_an_out_at_a_link_of_its_dataset_into_a_chunk_is_refused_and_leaves_the_chunk(tmp_path, named):
    # <batch>/dataset is the one place a batch holds a release; as a link into a chunk it is not that place, and a
    # chunk's dataset/ holds meta/info.json as a release does.
    batch = _rows_batch(tmp_path, parts=(("S1e", (12, 37, 41)),))
    chunk_dataset = batch.chunk_dir(batch.chunks()[0]) / DATASET_DIR
    (batch.root / DATASET_DIR).symlink_to(chunk_dataset, target_is_directory=True)
    ab.sealed_chunks(batch)
    before = _tree(batch.root)
    out = chunk_dataset if named == "where the link points" else batch.root / DATASET_DIR
    for refused in (lambda: ab.release_dir([batch], out),
                    lambda: ab.assemble(batch, tmp_path, **dict(FAKES, out=out, aggregate=_never_merged))):
        with pytest.raises(BatchError, match=re.escape(f"--out {out} 이 배치 {batch.root} 의 안입니다")):
            refused()
    assert _tree(batch.root) == before and len(ab.episodes_table(chunk_dataset)) == 2
    assert not list(chunk_dataset.parent.glob(f"{chunk_dataset.name}.*"))
    ab.sealed_chunks(batch)


def test_release_dir_gives_the_resolved_out_outside_every_batch(tmp_path, monkeypatch):
    first, rest = _two_batches(tmp_path)
    here = tmp_path / "here"
    here.mkdir()
    monkeypatch.chdir(here)
    for given in (Path("..") / "release", here / ".." / "release", tmp_path / "gone" / ".." / "release",
                  Path("release"), Path("../here/../release")):
        want = here / "release" if given == Path("release") else tmp_path / "release"
        got = ab.release_dir([first, rest], given)
        assert got == want and got.is_absolute() and ".." not in got.parts, (given, got)
    assert ab.release_dir([first], here / "release") == here / "release"


def test_an_assembly_into_a_link_to_an_earlier_result_replaces_the_result_it_points_to(tmp_path):
    first, rest = _two_batches(tmp_path)
    result = tmp_path / "release"
    ab.assemble([first, rest], tmp_path, **dict(FAKES, out=result, drop=[LEFT_OUT], say=lambda _: None))
    link = tmp_path / "link"
    link.symlink_to(result, target_is_directory=True)
    out = ab.assemble([first, rest], tmp_path, **dict(
        FAKES, out=link, drop=[LEFT_OUT], say=lambda _: None,
        lay=lambda repo_root, o, sidecars: (Path(o) / "README.md").write_text("again", encoding="utf-8") and ["README.md"]))
    assert out == result and link.is_symlink() and link.resolve() == result
    assert (result / "README.md").read_text(encoding="utf-8") == "again"
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(("link", "release"))) == ["link", "release"]


def test_an_assembly_into_an_out_an_earlier_one_made_replaces_it(tmp_path):
    first, rest = _two_batches(tmp_path)
    out = tmp_path / "release"

    def card(words):
        return lambda repo_root, o, sidecars: (Path(o) / "README.md").write_text(words, encoding="utf-8") and ["README.md"]

    ab.assemble([first, rest], tmp_path, **dict(FAKES, out=out, drop=[LEFT_OUT], lay=card("the first")))
    ab.assemble([first, rest], tmp_path, **dict(FAKES, out=out, drop=[LEFT_OUT], lay=card("the second")))
    assert (out / "README.md").read_text(encoding="utf-8") == "the second"
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(out.name)) == [out.name]


def test_one_batch_assembles_into_an_out_of_its_own_and_leaves_its_directory_as_it_was(tmp_path):
    batch = _rows_batch(tmp_path)
    out = ab.assemble(batch, tmp_path, **dict(FAKES, out=tmp_path / "release"))
    assert out == tmp_path / "release" and (out / EPISODES).is_file()
    assert sorted(p.name for p in batch.root.iterdir()) == ["batch.json", "chunks"]


def test_one_batch_given_twice_is_refused(tmp_path):
    batch = _rows_batch(tmp_path)
    (tmp_path / "link").symlink_to(batch.root, target_is_directory=True)
    with pytest.raises(BatchError, match="두 번"):
        ab.release_chunks([batch, load_batch(tmp_path / "link")])


def test_two_batches_of_one_directory_name_are_refused(tmp_path):
    one, two = _rows_batch(tmp_path / "a"), _rows_batch(tmp_path / "b")
    assert one.root.name == two.root.name
    with pytest.raises(BatchError, match="이름이 같은"):
        ab.release_chunks([one, two])


def _reword_the_objective_of(scenario):
    def edit(doc):
        next(s for s in doc["scenarios"] if s["scenario_id"] == scenario)["objective"] += " (다듬은 문장)"
    return edit


def test_with_several_batches_a_canon_difference_names_every_chunk_by_its_batch(tmp_path):
    first, rest = _two_batches(tmp_path)
    scenario = first.parts[0][0]
    _edit_canon(tmp_path, "meta/scenarios.json", _reword_the_objective_of(scenario))
    with pytest.raises(BatchError) as e:
        ab.assemble([first, rest], tmp_path, **dict(FAKES, out=tmp_path / "release", drop=[LEFT_OUT],
                                                    aggregate=_never_merged))
    named = [f"{batch.root.name}/{plan.key}" for batch, plan in _shipped_chunks(first, rest) if plan.scenario == scenario]
    assert len(named) == 2 and named[0].startswith("first/") and named[1].startswith("rest/")
    assert _listed(str(e.value)) == [f"meta/scenarios.json: scenarios[{scenario}].objective (문장) — 청크 {', '.join(named)}"]


def test_with_several_batches_a_seal_check_names_the_chunk_by_its_batch(tmp_path):
    first, rest = _two_batches(tmp_path)
    plan = rest.chunks()[0]
    record = read_seal(rest.chunk_dir(plan))
    write_seal(rest.chunk_dir(plan), dict(record, generator_commit="f" * 40))
    with pytest.raises(BatchError, match=re.escape(f"청크 rest/{plan.key} 의 generator_commit")):
        ab.release_chunks([first, rest], [LEFT_OUT])


def test_with_several_batches_a_chunk_directory_no_batch_planned_is_named_by_its_batch(tmp_path):
    first, rest = _two_batches(tmp_path)
    shutil.copytree(rest.chunk_dir(rest.chunks()[0]), rest.root / "chunks" / "S1e-009")
    with pytest.raises(BatchError, match=re.escape("['rest/S1e-009']")):
        ab.release_chunks([first, rest], [LEFT_OUT])


def test_the_work_places_sit_beside_the_release_and_are_named_after_it(tmp_path):
    batch = tmp_path / "batch"
    assert ab.beside(batch / DATASET_DIR) == (batch / ab.ASSEMBLING, batch / ab.VIEWS, batch / ab.PREVIOUS)
    assert ab.beside(tmp_path / "release") == (tmp_path / "release.assembling", tmp_path / "release.views",
                                               tmp_path / "release.previous")


def _without_lerobot(monkeypatch):
    """lerobot stood in for, as main imports it (it is not in this environment)."""
    import sys
    import types

    for name, attribute in (("lerobot", None), ("lerobot.datasets", None),
                            ("lerobot.datasets.aggregate", "aggregate_datasets"),
                            ("lerobot.datasets.compute_stats", "aggregate_stats"),
                            ("lerobot.datasets.io_utils", "write_stats"),
                            ("lerobot.datasets.lerobot_dataset", "LeRobotDataset")):
        module = types.ModuleType(name)
        if attribute:
            setattr(module, attribute, lambda *a, **k: None)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def test_main_hands_the_assembly_every_batch_the_out_and_the_rows_to_drop(tmp_path, monkeypatch, capsys):
    _without_lerobot(monkeypatch)
    calls = []

    def watched(batches, repo_root, **stages):
        calls.append((batches, stages))
        return stages["out"]

    monkeypatch.setattr(ab, "load_batch", lambda path: ("the batch at", path))
    monkeypatch.setattr(ab, "assemble", watched)
    out = tmp_path / "release"
    assert ab.main(["--batch", "a", "--batch", "b", "--out", str(out), "--drop", "S1f:12", "--drop", "S1g:7"]) == 0
    ((batches, stages),) = calls
    assert batches == [("the batch at", Path("a")), ("the batch at", Path("b"))]
    assert stages["out"] == out and list(stages["drop"]) == [("S1f", 12), ("S1g", 7)]
    assert f"validator/validate_dataset.py {out} " in capsys.readouterr().out


@pytest.mark.parametrize("value", ["S1f12", "S1f:", ":12", "S1f:-1", "S1f:1x", "S1f:1:2", "S1f: 12"])
def test_main_refuses_a_drop_that_is_not_a_scenario_and_a_row(value, capsys):
    with pytest.raises(SystemExit) as e:
        ab.main(["--batch", "x", "--drop", value])
    said = capsys.readouterr().err
    assert e.value.code == 2 and "argument --drop" in said and repr(value) in said, said


def test_main_refuses_several_batches_without_an_out(capsys):
    with pytest.raises(SystemExit) as e:
        ab.main(["--batch", "x", "--batch", "y"])
    assert e.value.code == 2 and "--out" in capsys.readouterr().err
