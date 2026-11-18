#!/usr/bin/env python3
"""Make a copy of a release that LeRobot's training command takes. The release is not written.

The release is laid out for evaluation, and three of its properties stop ``lerobot-train``
(LeRobot 0.6.1) before it trains:

  1. Some step columns have no statistics: their blocks in meta/stats.json are nulls with an
     ``undefined_reason`` (the data hold non-finite values by the spec). The normaliser turns every
     statistic into a tensor when it is built, so it stops on them whether a policy reads those
     columns or not.
  2. There is no ``observation.state`` column. ACT and Diffusion Policy read the robot state by that
     one name; the release splits it into observation.state.joint_pos, .joint_vel and .joint_torque.
  3. Some observation columns hold text (observation.contact.region_names and
     observation.contact.type are JSON strings). Unless input features are given, every column whose
     name starts with ``observation`` is a policy input -- and a policy that reads past frames reads
     every ``observation.`` column at those frames and stacks them, whatever its inputs are. Text does
     not stack.

The copy leaves out the statistics in (1) -- every block that holds a null or text -- leaves every
text column whose name starts with ``observation`` out of the step files and out of info.json's
features, and adds an ``observation.state`` column: one of the release's numeric vector columns
(--state, the joint positions by default; at least two long, as LeRobot reads a column of length one
as a scalar), with its features entry and statistics. On such a copy of the 26.11 release, ACT and
Diffusion Policy each trained two steps from fresh weights, with and without --policy.input_features;
convergence and policy quality were not measured.

Everything else is the release's. The step files -- the ``data/*/*.parquet`` LeRobot reads -- keep
their other columns, their order and their row groups, and the datasets library's description of the
columns in each file follows the columns. meta/ is copied as it is apart from info.json and stats.json
-- the per-episode statistics in meta/episodes keep the columns they had. The rest of the release
(README.md, NOTICE, the licences, assets/, any other file under data/) is copied. The videos are hard
links to the release's files unless --copy-videos, and are copied where the filesystem cannot link
them. A hard link is the same file: a tool that writes into the copy's videos writes into the
release's.

Links in the release are followed, files and directories alike, as huggingface_hub's cache lays out
a download: the copy holds the files the links point at, and its hard links are to those files, not
to the links. A release in which links reach one directory twice (a link back up, two links to one
directory) is refused. The copied files are new files: the release's bytes, not its permissions, so a
read-only release gives a copy its owner can change and remove. OUT is refused if it exists (a link
included), if it is inside the release or inside any directory the release links to, or if its path
holds a character the datasets library reads as a pattern (the training command hands it the copy's
file paths). The copy is built beside OUT under a hidden name and renamed to OUT once it is whole, so
OUT either does not exist or is a complete copy.

If you share the copy, CC BY 4.0 asks you to say that it was changed -- the three changes above.

    python scripts/make_train_copy.py <release> <out> [--state COLUMN] [--copy-videos]
"""
from __future__ import annotations

import argparse
import copy
import errno
import json
import os
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

STATE = "observation.state"
DEFAULT_STATE = "observation.state.joint_pos"
# lerobot.utils.constants.OBS_STR: the prefix dataset_to_policy_features takes as a state input. The
# past-frame reading takes OBS_PREFIX, "observation.", which this covers.
POLICY_INPUT_PREFIX = "observation"
STATE_DTYPES = ("float32", "float64")
STEP_FILES = "*/*.parquet"                    # under data/: what LeRobot's loader reads
GLOB_CHARACTERS = "*?["                       # what the datasets library reads in a path as a pattern
# What a filesystem says when it cannot hold the hard link: the file is copied instead.
CANNOT_LINK = frozenset({errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EMLINK})
HF_METADATA = b"huggingface"                  # the datasets library's description of the columns
OWN_DIRECTORIES = ("data", "meta", "videos")  # what is rewritten or linked; the rest is copied


class CopyRefused(ValueError):
    """The copy was not started: nothing was written."""


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def text_columns(features: dict) -> list[str]:
    """The policy-input columns that hold text."""
    return sorted(name for name, feature in features.items()
                  if name.startswith(POLICY_INPUT_PREFIX) and feature.get("dtype") == "string")


def _numeric(value) -> bool:
    if isinstance(value, list):
        return all(_numeric(v) for v in value)
    return isinstance(value, (int, float))


def empty_statistics(stats: dict) -> list[str]:
    """The statistics blocks no tensor can be made of: a null, or the text saying why they are empty."""
    return sorted(name for name, block in stats.items()
                  if not (isinstance(block, dict) and all(_numeric(v) for v in block.values())))


def _state_problem(features: dict, stats: dict, empty: list[str], state: str) -> str | None:
    if STATE in features:
        return f"배포본에 이미 {STATE} 열이 있습니다 — 이 도구가 더할 것이 없습니다"
    feature = features.get(state)
    if feature is None:
        return f"--state {state}: 배포본의 meta/info.json 이 선언하지 않은 열입니다"
    if feature.get("dtype") not in STATE_DTYPES:
        return (f"--state {state}: 실수 열이 아닙니다(dtype {feature.get('dtype')}) — "
                f"관절 열처럼 float32 · float64 의 열을 고르십시오")
    shape = feature.get("shape") or []
    if len(shape) != 1 or not isinstance(shape[0], int) or shape[0] < 2:
        return (f"--state {state}: 길이가 2 이상인 1차원 열이 아닙니다(shape {shape}) — LeRobot 은 길이 1 인 열을 "
                f"스칼라로 읽어 정책의 입력 차원이 맞지 않습니다")
    if state not in stats or state in empty:
        return f"--state {state}: 이 열의 통계가 meta/stats.json 에 없거나 비어 있습니다"
    return None


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _raise(error: OSError) -> None:
    raise error                                # os.walk passes over a directory it cannot list unless told


def _read_directories(release: Path) -> list[Path]:
    """Every directory the copy reads from, links followed: the release and what it links to.

    Refuses a release in which links reach a directory twice -- the walk that copies would go round
    a link back up for ever, and copy a directory reached by two links twice."""
    seen: dict[Path, str] = {}
    for here, dirs, _ in os.walk(release, followlinks=True, onerror=_raise):
        real = Path(os.path.realpath(here))
        if real in seen:
            raise CopyRefused(f"{here}: 링크를 따라가면 같은 디렉터리({seen[real]})에 두 번 닿습니다 — "
                              f"되돌아오거나 겹치는 링크가 없는 배포본을 주십시오")
        seen[real] = here
    return list(seen)


def _copy_tree(src: Path, dst: Path, place, skip: frozenset[str] = frozenset()) -> None:
    """src's files placed under dst, links followed (_read_directories has refused a directory reached twice)."""
    for here, _, files in os.walk(src, followlinks=True, onerror=_raise):
        target = dst / os.path.relpath(here, src)
        target.mkdir(parents=True, exist_ok=True)
        for name in files:
            if os.path.join(here, name) not in skip:
                place(os.path.join(here, name), target / name)


def _rewritten_schema(schema: pa.Schema, dropped: list[str], state: str) -> pa.Schema:
    fields = [f for f in schema if f.name not in dropped] + [schema.field(state).with_name(STATE)]
    metadata = dict(schema.metadata or {})
    if HF_METADATA in metadata:
        doc = json.loads(metadata[HF_METADATA])
        described = (doc.get("info") or {}).get("features")
        if isinstance(described, dict):
            for name in dropped:
                described.pop(name, None)
            if state in described:
                described[STATE] = copy.deepcopy(described[state])
            metadata[HF_METADATA] = json.dumps(doc).encode()
    return pa.schema(fields, metadata=metadata or None)


def rewrite_steps(src: Path, dst: Path, dropped: list[str], state: str) -> None:
    """One step file: the text columns left out and observation.state added, row group by row group."""
    reader = pq.ParquetFile(src)
    schema = _rewritten_schema(reader.schema_arrow, dropped, state)
    with pq.ParquetWriter(dst, schema) as writer:
        for i in range(reader.num_row_groups):
            table = reader.read_row_group(i)
            table = table.drop_columns([name for name in dropped if name in table.column_names])
            table = table.append_column(STATE, table.column(state)).replace_schema_metadata(schema.metadata)
            writer.write_table(table, row_group_size=max(1, table.num_rows))


class _VideoPlacer:
    """What places each file of videos/: a hard link to the file a link points at, or a copy."""

    def __init__(self, copy_videos: bool):
        self.copy_videos, self.linked, self.copied = copy_videos, 0, 0

    def __call__(self, src: str, dst: str) -> None:
        if not self.copy_videos:
            try:
                os.link(os.path.realpath(src), dst)
                self.linked += 1
                return
            except OSError as e:
                if e.errno not in CANNOT_LINK:
                    raise
        shutil.copyfile(src, dst)
        self.copied += 1


def _build(release: Path, staging: Path, info: dict, stats: dict, steps: list[Path], dropped: list[str],
           empty: list[str], state: str, copy_videos: bool) -> dict:
    for entry in sorted(release.iterdir()):
        if entry.name in OWN_DIRECTORIES:
            continue
        if entry.is_dir():
            _copy_tree(entry, staging / entry.name, shutil.copyfile)
        else:
            shutil.copyfile(entry, staging / entry.name)

    _copy_tree(release / "meta", staging / "meta", shutil.copyfile)
    features = info["features"]
    for name in dropped:
        del features[name]
    features[STATE] = copy.deepcopy(features[state])
    _write_json(staging / "meta" / "info.json", info)
    for name in empty:
        del stats[name]
    stats[STATE] = copy.deepcopy(stats[state])
    _write_json(staging / "meta" / "stats.json", stats)

    _copy_tree(release / "data", staging / "data", shutil.copyfile, skip=frozenset(str(src) for src in steps))
    for src in steps:
        rewrite_steps(src, staging / src.relative_to(release), dropped, state)

    placer = _VideoPlacer(copy_videos)
    if (release / "videos").is_dir():
        _copy_tree(release / "videos", staging / "videos", placer)
    return {"step_files": len(steps), "dropped_columns": dropped, "dropped_stats": empty, "state": state,
            "videos_linked": placer.linked, "videos_copied": placer.copied}


def make_copy(release: Path, out: Path, state: str = DEFAULT_STATE, copy_videos: bool = False) -> dict:
    """Make the copy at OUT and say what it changed. Refuses (CopyRefused) before writing anything."""
    given = Path(out).absolute()                # not normalised: a .. after a link is the link's
    if os.path.lexists(given):
        raise CopyRefused(f"{given}: 이미 있습니다 — 없는 자리를 주십시오")
    release, out = Path(release).resolve(), given.resolve()
    if not (release / "meta" / "info.json").is_file():
        raise CopyRefused(f"{release}: meta/info.json 이 없습니다 — 배포본의 루트를 주십시오")
    if _inside(release, out) or any(_inside(out, d) for d in _read_directories(release)):
        raise CopyRefused(f"{out}: 배포본 안이거나 배포본이 링크로 가리키는 디렉터리 안입니다 — 배포본 밖의 새 자리를 주십시오")
    if os.path.lexists(out):                   # reached past a missing directory: missing/../copy
        raise CopyRefused(f"{out}: 이미 있습니다 — 없는 자리를 주십시오")
    magic = sorted(set(GLOB_CHARACTERS) & set(str(given) + str(out)))
    if magic:
        raise CopyRefused(f"{out}: 경로에 {' '.join(magic)} 가 있습니다 — 학습 명령이 사본의 파일 경로를 패턴으로 "
                          f"읽어 다른 파일을 찾습니다")
    if not out.parent.is_dir():
        raise CopyRefused(f"{out.parent}: 사본을 둘 디렉터리가 없습니다")
    staging = out.parent / f".{out.name}.partial"
    if os.path.lexists(staging):
        raise CopyRefused(f"{staging}: 지난 실행이 남긴 작업 자리가 있습니다 — 지운 뒤 다시 돌리십시오")

    steps = sorted((release / "data").glob(STEP_FILES))
    if not steps:
        raise CopyRefused(f"{release / 'data'}: 스텝 파일({STEP_FILES})이 없습니다 — 배포본의 루트를 주십시오")
    info = _read_json(release / "meta" / "info.json")
    stats_path = release / "meta" / "stats.json"
    stats = _read_json(stats_path) if stats_path.is_file() else {}
    features = info.get("features") or {}
    dropped, empty = text_columns(features), empty_statistics(stats)
    problem = _state_problem(features, stats, empty, state)
    if problem:
        raise CopyRefused(problem)

    staging.mkdir()
    try:
        report = _build(release, staging, info, stats, steps, dropped, empty, state, copy_videos)
        os.rename(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"release": str(release), "out": str(out), **report}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="배포본 옆에 lerobot-train 이 받는 사본을 만든다 — 배포본은 쓰지 않는다")
    ap.add_argument("release", type=Path, help="배포본의 루트(meta/info.json 이 있는 디렉터리)")
    ap.add_argument("out", type=Path, help="사본을 만들 새 자리 — 아직 없어야 합니다")
    ap.add_argument("--state", default=DEFAULT_STATE,
                    help=f"observation.state 로 쓸 실수 벡터 열 (기본 {DEFAULT_STATE})")
    ap.add_argument("--copy-videos", action="store_true",
                    help="영상을 하드링크 대신 복사합니다 — 영상만큼 자리가 더 듭니다")
    args = ap.parse_args(argv)
    try:
        r = make_copy(args.release, args.out, state=args.state, copy_videos=args.copy_videos)
    except CopyRefused as e:
        print(f"사본을 만들지 않았습니다 — {e}", file=sys.stderr)
        return 2
    print(f"사본 {r['out']}")
    print(f"  스텝 파일 {r['step_files']}개 — 뺀 열: {', '.join(r['dropped_columns']) or '없음'}")
    print(f"  {STATE} ← {r['state']}")
    print(f"  뺀 통계: {', '.join(r['dropped_stats']) or '없음'}")
    print(f"  영상 하드링크 {r['videos_linked']}개 · 복사 {r['videos_copied']}개")
    if r["videos_linked"]:
        print("  하드링크는 배포본의 영상과 같은 파일입니다 — 사본의 영상을 고쳐 쓰는 도구를 돌리지 마십시오")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
