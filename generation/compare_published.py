"""Compare a regenerated episode against the published parquet (C9 step 3).

npz-to-npz would only prove the generator repeats itself. What C9 has to
show is that the numbers a downloader reads came out of that seed, so the
comparison runs against the published columns -- and, for a distribution
that ships meta/cameras.json declaring a pack, against its published
depth/mask pack too. The packs are optional: the canon switches them off by
leaving ``packs`` out, and then there is no pack to compare.

The episode row is read too, where the converter copied it out of the sidecar
(compare_episode_meta): the randomization the run drew and realized, why it
ended, the step the hold let go, the SSM parameters. Columns that come back bit
for bit do not stop a recorder from writing other values there. Left out are
where and with what the run ran (RUN_ENVIRONMENT) and what the labeller wrote
(LABEL_NOTES) -- a rerun makes no labels. Of sim_backend the policy that walked
the episode is compared (POLICY -- a table scenario's checkpoint and behaviour
layer, which the hook reruns with): another policy makes another episode.

A step column the distribution's own meta/safety_schema.json lists as optional
(optional_step_fields) and the distribution does not carry is not compared,
and the success line says how many were left out; a column of the schema's
required or conditional lists that is not there fails the comparison. Whether
the distribution owes an optional column all the same -- one of its scenarios
names it in required_fields, or one of its rules in inputs -- is not asked
here: the validator's C8 holds the step table to those.

RGB video is never compared: it is not bit-reproducible across runs (the
renderer's temporal accumulation -- two runs of the same
episode never shared a frame), while the packs and the box/pose columns are.
The success line of every camera distribution says so, pack or not, and C9
carries that line into the validation report (validate_dataset.repro_hook_finding),
so a green C9 is not read as covering the video. What stands in for the pictures
is the recording's stamp: the rerun's camera-canon fingerprint has to be the
shipped meta/cameras.json's (pack_pair) -- and the recorder's count of black frames,
held as the converter holds it (no camera black in every frame). On failure the
printed reason is what C9 reports first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

import sys
from pathlib import Path

# Run as a script, sys.path[0] is this file's directory, not the repo root, so
# the `generation.*` imports below fail. run_s1a.py already carries the same
# guard; without it the documented commands need a PYTHONPATH the
# documentation never mentions (measured: all four failed).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generation import cameras as camera_canon
from generation import provenance
from generation.compare_repro import comparable_step_fields, compare_arrays, episode_seq_of
from generation.convert_lerobot import (MEDIA_PACK, PACK_ARRAYS, check_blank_frames,
                                        downsample_factors, downsample_pack, episode_column,
                                        media_files, pack_file, recorded_canon_fingerprint,
                                        require_run_start, require_termination, require_type,
                                        the_pack)
from generation.directed_separation import scenario_declares_directed_ssm
from generation.field_map import CAMERAS, EPISODE_FIELDS, HRI

# Said on every camera success line: what C9 did not compare, and why.
NOT_COMPARED = "RGB 영상은 대조하지 않음 (실행마다 렌더가 달라 비트 재현되지 않음)"

# Published episode values the converter copies out of the sidecar that C9 leaves out: where and
# with what the run ran, not what its seed made -- the commit and the Isaac build (sim_backend; its
# physics engine is compared as measurement_method) and the URL each asset loaded from. A rerun on
# another commit, build or asset root that gives every other value back has reproduced the
# episode; when one does not, commit_note names the commit.
RUN_ENVIRONMENT = ("sim_backend",)
RUN_ENVIRONMENT_NOTES = ("asset_paths_resolved",)
# The one member of sim_backend that is not the environment: the walking policy and behaviour layer a
# table scenario's episode recorded (behavior.policy_record). The hook reruns with it
# (generation/rerun_policy.py) and it is compared whole; absent on both sides (S1a, S1d) is the same.
POLICY = "sim_backend.policy"
# notes members the converter takes from the labeller, not the sidecar (build_episode_meta). A
# rerun makes no labels: with the step columns and the rest of the notes reproduced, the
# distribution's labels are re-derived from them (generation/labeling/check_release_labels.py).
LABEL_NOTES = ("not_evaluated", "contact_observed", "contact_detection_limits")


def compared_episode_fields() -> tuple[tuple[str, str], ...]:
    """(published episode field, sidecar key) for every top-level field C9 compares: those
    field_map says the converter copies straight out of the sidecar, less RUN_ENVIRONMENT.
    notes is compared member by member (compare_episode_meta)."""
    return tuple((f.schema, f.key) for f in EPISODE_FIELDS
                 if f.source == "sidecar" and f.schema not in RUN_ENVIRONMENT)


_ABSENT = object()


def converter_refusal(meta: dict) -> str | None:
    """Why the converter would refuse the rerun's sidecar in a value C9 compares, or None: the checks
    build_episode_meta makes on the sidecar values it publishes (require_termination, require_type,
    require_run_start). The comparison alone cannot say it: the published row holds termination as JSON
    text, so a rerun that wrote that text where the converter takes an object compared equal."""
    try:
        require_termination(meta)
        require_type(meta, "degenerate", bool)
        require_run_start(meta)
    except KeyError as e:
        return f"사이드카에 {e} 가 없음"
    except (ValueError, TypeError) as e:
        return str(e)
    return None


def _at(doc, dotted: str):
    """The value at a dotted sidecar key ("sim_backend.physics"), or _ABSENT."""
    for part in dotted.split("."):
        if not isinstance(doc, dict) or part not in doc:
            return _ABSENT
        doc = doc[part]
    return doc


def _published_value(cell, regenerated):
    """A published episode cell, decoded as the converter encoded ``regenerated`` (episode_column):
    JSON text where it wrote one, the value itself otherwise -- numpy's scalars and arrays back to
    Python's. Raises ValueError (or RecursionError, nested past the decoder) on JSON text that does
    not parse."""
    if isinstance(cell, np.ndarray):
        cell = cell.tolist()
    elif isinstance(cell, np.generic):
        cell = cell.item()
    if (isinstance(cell, str) and not isinstance(regenerated, str)
            and isinstance(episode_column(regenerated), str)):
        return json.loads(cell)
    return cell


def _first_difference(published, regenerated, path: str):
    """Where two decoded JSON values first part, as (path, published, regenerated) -- _ABSENT for a
    member one side lacks -- or None. Exact, as the arrays are compared: a number's type is part of
    it (true is not 1, 4 is not 4.0) and NaN equals NaN."""
    if isinstance(published, dict) and isinstance(regenerated, dict):
        for key in sorted(set(published) | set(regenerated), key=str):
            if key not in published or key not in regenerated:
                return (f"{path}.{key}", published.get(key, _ABSENT), regenerated.get(key, _ABSENT))
            found = _first_difference(published[key], regenerated[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(published, list) and isinstance(regenerated, list):
        for i, (a, b) in enumerate(zip(published, regenerated)):
            found = _first_difference(a, b, f"{path}[{i}]")
            if found:
                return found
        if len(published) != len(regenerated):
            return (f"{path}[{min(len(published), len(regenerated))}]",
                    published[len(regenerated)] if len(published) > len(regenerated) else _ABSENT,
                    regenerated[len(published)] if len(regenerated) > len(published) else _ABSENT)
        return None
    if isinstance(published, float) and isinstance(regenerated, float) \
            and published != published and regenerated != regenerated:
        return None
    if type(published) is not type(regenerated) or published != regenerated:
        return (path, published, regenerated)
    return None


def _shown(value) -> str:
    """One side of a difference, as JSON text cut to a line. Not every published cell is a JSON value
    (a column read twice, a type no JSON has): those are shown by str rather than ending the run."""
    if value is _ABSENT:
        return "(없음)"
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= 80 else text[:79] + "…"


def compare_episode_meta(meta: dict, row) -> tuple[list[str], int]:
    """Mismatches between the regenerated sidecar ``meta`` and the published episode ``row`` (a
    one-row DataFrame) in what the converter copies out of the sidecar -- first whether the converter
    would take the sidecar at all (converter_refusal), then the compared_episode_fields,
    and every member of the published notes but LABEL_NOTES and RUN_ENVIRONMENT_NOTES -- the
    converter writes those under the sidecar key of the same name (meta.get, so a key the rerun did
    not write is null). Returns (problems, notes members compared).

    The notes members come from the published row, so one missing there is not compared -- C5
    requires every member the schema names. A published value that cannot be read is one finding,
    as a value that differs is.

    POLICY is compared too, out of the published sim_backend's JSON: the record on both sides, or on
    neither."""
    problems: list[str] = []
    refusal = converter_refusal(meta)
    if refusal:
        problems.append(f"에피소드 메타: 재생성한 사이드카를 변환기가 받지 않음 — {refusal}")

    def published(column, regenerated):
        if column not in row.columns:
            problems.append(f"에피소드 메타 {column}: 배포본에 없음")
            return _ABSENT
        try:
            return _published_value(row[column].iloc[0], regenerated)
        except (ValueError, RecursionError):
            problems.append(f"에피소드 메타 {column}: 배포본의 값이 JSON 으로 읽히지 않음")
            return _ABSENT

    def compare(pub, regen, path):
        try:
            found = _first_difference(pub, regen, path)
        except RecursionError:
            problems.append(f"에피소드 메타 {path}: 너무 깊이 중첩되어 대조하지 못함")
            return
        if found:
            where, a, b = found
            problems.append(f"에피소드 메타 {where}: 배포 {_shown(a)} · 재생성 {_shown(b)}")

    for column, key in compared_episode_fields():
        regenerated = _at(meta, key)
        if regenerated is _ABSENT:
            problems.append(f"에피소드 메타 {column}: 재생성한 사이드카에 {key} 이 없음")
            continue
        pub = published(column, regenerated)
        if pub is not _ABSENT:
            compare(pub, regenerated, column)

    backend = published("sim_backend", {})
    if backend is not _ABSENT:
        if isinstance(backend, dict):
            compare(_at({"sim_backend": backend}, POLICY), _at(meta, POLICY), POLICY)
        else:
            problems.append("에피소드 메타 sim_backend: 배포본의 값이 JSON 객체가 아님")

    notes = published("notes", {})
    if notes is _ABSENT:
        return problems, 0
    if not isinstance(notes, dict):
        problems.append("에피소드 메타 notes: 배포본의 값이 JSON 객체가 아님")
        return problems, 0
    members = sorted(set(notes) - set(LABEL_NOTES) - set(RUN_ENVIRONMENT_NOTES), key=str)
    for member in members:
        compare(notes[member], meta.get(member), f"notes.{member}")
    return problems, len(members)


def published_conditions(root: Path) -> tuple[str, ...]:
    """The step-field conditions a distribution declares: CAMERAS when it ships meta/cameras.json --
    the file whose presence makes the camera fields required (schema §6); presence, not a parse: a
    cameras.json that is there but unreadable still owes the columns, and the comparison fails on
    them rather than quietly comparing fewer -- and HRI when a scenario of its meta/scenarios.json
    declares R-TS15066-SSM-02, the condition the validator reads from the same file. A
    scenarios.json that is there but unreadable, or without its scenario list, is refused:
    guessing would compare fewer arrays without a word."""
    root = Path(root)
    out = [CAMERAS] if (root / "meta" / "cameras.json").exists() else []
    path = root / "meta" / "scenarios.json"
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ValueError(f"배포본의 meta/scenarios.json 을 읽지 못해 대조할 열을 정하지 못합니다: {e}") from e
        entries = doc.get("scenarios") if isinstance(doc, dict) else None
        if not isinstance(entries, list):
            raise ValueError("배포본의 meta/scenarios.json 에 시나리오 목록(scenarios)이 없어 "
                             "대조할 열을 정하지 못합니다")
        if any(isinstance(s, dict) and scenario_declares_directed_ssm(s) for s in entries):
            out.append(HRI)
    return tuple(out)


def published_pairs(root: Path) -> tuple[tuple[str, str], ...]:
    """(schema field, npz array) pairs C9 compares against this distribution."""
    return comparable_step_fields(published_conditions(root))


def published_optional_fields(root: Path) -> frozenset[str]:
    """The step columns a distribution may leave out: the optional_step_fields of its own meta/safety_schema.json
    (schema 0.2). A distribution whose schema is not there, does not read, or names none (one before 0.2) gives none --
    every compared column is then owed, as before; a release whose schema the validator cannot trust is C1's to say."""
    path = Path(root) / "meta" / "safety_schema.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return frozenset()
    named = doc.get("optional_step_fields") if isinstance(doc, dict) else None
    if not isinstance(named, list):
        return frozenset()
    return frozenset(f for f in named if isinstance(f, str))


def column_to_array(values, reference: np.ndarray) -> np.ndarray:
    """One published column, rebuilt as an array. Decodes; never coerces.

    Three encodings arrive here: a list per row (vector fields), a scalar per
    row, and a JSON string per row (the string arrays LeRobot cannot hold as
    arrays). ``reference`` only says which of the three to expect.

    The result is not cast to the reference dtype: the cast hid four
    real differences (all reproduced) -- an int8
    reference swallowed a published 257 as 1, a float32 reference rounded
    1 + 2^-30 back to 1, a bool reference read a published 2 as True, and a
    <U4 reference truncated "noneX" to "none". compare_arrays checks dtype
    and shape itself, so decoding is all this owes it.
    """
    if len(values) == 0:
        return np.asarray(values)
    first = values[0]
    if isinstance(first, str):
        return np.array([json.loads(v) for v in values])
    if reference.ndim > 1:
        return np.stack([np.asarray(v) for v in values])
    return np.asarray(list(values))


class PackPair(NamedTuple):
    """The two packs C9 compares for one episode of a camera distribution."""
    name: str               # the pack's key in meta/cameras.json packs, and its packs/<name>/ directory
    camera: dict            # the shipped canon's entry for the pack's camera
    downsample: dict        # the shipped canon's downsample for the pack
    regenerated: Path       # the rerun's pack: full size, every recorded step
    published: Path         # packs/<name>/episode_<episode_index>.npz in the distribution


def pack_pair(root: Path, episode_index: int, meta: dict, episode_dir: Path) -> PackPair | None:
    """Find both packs before anything is compared -- or None when the distribution's own
    meta/cameras.json declares no pack (the canon switched it off). Raises
    ValueError with the sentence to print when either is missing or the distribution cannot
    say how its pack was cut.

    Name, camera and downsample come from the distribution's own meta/cameras.json, not the
    repository's: C9 checks the distribution against what it says about itself. The downsample
    is read here too (downsample_factors), so a document that cannot state one is named rather
    than left to crash the cut later. A document that is not a camera canon at all -- one the
    generator's loader refuses, bbox.camera included, pack or not -- is named as that, not as a
    pack it may not declare. The regenerated sidecar is read with the converter's
    reader (media_files), against that same canon, pack or not -- what it would refuse as a
    conversion input, a missing video, a frame count off or a pack the canon does not declare
    included, is refused here too. So is a rerun whose stamp (media.canon_fingerprint, read as the
    converter reads it) is not that canon's recording_fingerprint: the hook records with the
    repository's canon, and one that differs from the shipped copy only in what reaches RGB alone
    leaves every compared column as it was. The shipped copy's
    fx/fy/cx/cy are not in the fingerprint, so equality is exact. And so is a rerun with a camera black in
    every frame, or one that did not count them -- the converter's own check (check_blank_frames): RGB is not
    compared, and the box and pose columns of a buried camera are what they are."""
    try:
        cameras = camera_canon.load(Path(root) / "meta" / "cameras.json")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as e:
        raise ValueError(f"배포본 meta/cameras.json 을 카메라 정본으로 읽지 못합니다: "
                         f"{type(e).__name__}: {e}") from e
    try:
        name = the_pack(cameras)
        if name is not None:
            pack = cameras["packs"][name]
            camera, downsample = cameras["cameras"][pack["camera"]], pack["downsample"]
            downsample_factors(camera, downsample)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"배포본 meta/cameras.json 이 팩을 설명하지 못합니다: "
                         f"{type(e).__name__}: {e}") from e
    try:
        media = media_files(meta, episode_dir, cameras)
    except (OSError, ValueError) as e:
        raise ValueError(f"재생성한 에피소드의 매체(사이드카 media)를 읽지 못했습니다 — {e}") from e
    try:
        recorded = recorded_canon_fingerprint(meta)
    except ValueError as e:
        raise ValueError(f"재생성한 녹화의 카메라 정본 지문을 읽지 못했습니다 — {e}") from e
    if recorded != camera_canon.recording_fingerprint(cameras):
        raise ValueError(f"{meta.get('episode_id')}: 재생성한 녹화 때 카메라 정본과 배포본 meta/cameras.json 이 "
                         "다릅니다 — 훅은 저장소의 정본으로 다시 녹화하므로 저장소 정본과 이 배포본이 싣는 정본이 "
                         "갈라진 것입니다. 상자·자세 열이 같아도 RGB 에만 닿는 값(시야각 · 클리핑 · 렌더)이 다른 "
                         "녹화라 재현으로 보지 않습니다 — 배포본을 만든 정본에서 다시 돌리십시오")
    try:
        check_blank_frames(meta)
    except ValueError as e:
        raise ValueError(f"재생성한 녹화를 변환기가 받지 않습니다 — {e}") from e
    if name is None:
        return None
    regenerated = media[MEDIA_PACK]
    rel = pack_file(name, episode_index)
    # Lexical, as the validator's C1 and the converter's media_files check a path: the pack
    # name comes out of the distribution, and a download cache may hold its files as symlinks.
    if Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ValueError(f"배포본 meta/cameras.json 의 팩 이름이 배포본 밖을 가리킵니다: {rel}")
    published = Path(root) / rel
    if not published.is_file():
        raise ValueError(f"배포본에 이 에피소드의 팩이 없습니다: {rel}")
    return PackPair(name, camera, downsample, regenerated, published)


def _read_pack(path: Path) -> dict[str, np.ndarray]:
    """The PACK_ARRAYS members an npz holds; one it lacks is left out, for compare_arrays
    to report. No pickles: the published pack is the distribution's, not ours."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in PACK_ARRAYS if k in z.files}


def compare_packs(pair: PackPair) -> tuple[list[str], int]:
    """Mismatches between the regenerated pack, cut down as the converter cuts it
    (downsample_pack with the distribution's own canon), and the published pack: each of
    PACK_ARRAYS bit for bit, shape and dtype included. Returns (problems, rows compared).

    Peak memory is both packs whole -- about 1.8 GB for a 1,000-step episode at 640x480."""
    label = f"팩 {pair.name}"
    try:
        regenerated = _read_pack(pair.regenerated)
    except Exception as e:  # noqa: BLE001 -- any unreadable npz is one finding, not a crash
        return [f"{label}: 재생성한 팩을 읽지 못함 — {type(e).__name__}: {e}"], 0
    try:
        regenerated = downsample_pack(regenerated, pair.camera, pair.downsample)
    except ValueError as e:
        return [f"{label}: 재생성한 팩을 배포본 meta/cameras.json 의 축소로 줄이지 못함 — {e}"], 0
    try:
        published = _read_pack(pair.published)
    except Exception as e:  # noqa: BLE001
        return [f"{label}: 배포본 팩을 읽지 못함 — {type(e).__name__}: {e}"], 0
    pairs = [(f"{label}.{member}", member) for member in PACK_ARRAYS]
    return compare_arrays(regenerated, published, pairs), int(regenerated["step_index"].shape[0])


def commit_note(row, repo_root) -> str | None:
    """One line for a reproduction that failed: whether the rerun ran other code than the
    published episode names. The hook reruns the working tree, not the recorded commit, so
    "was it the same code" is the first thing to ask. None when it was, or cannot be told."""
    if "sim_backend" not in row.columns:
        return None
    try:
        recorded = json.loads(row["sim_backend"].iloc[0]).get(provenance.COMMIT)
    except (TypeError, ValueError, AttributeError):
        return None
    now = provenance.read(repo_root)
    head = now[provenance.COMMIT]
    if not isinstance(recorded, str) or not head:
        return None
    if recorded != head:
        what = f"기록된 생성 커밋 {recorded[:12]} ≠ 지금 HEAD {head[:12]}"
    elif now[provenance.DIRTY]:
        what = f"지금 작업 트리에 커밋되지 않은 변경이 있습니다 (HEAD {head[:12]})"
    else:
        return None
    return f"참고: {what} — 재현은 지금 작업 트리의 코드로 돌았습니다"


def main() -> int:
    ap = argparse.ArgumentParser(description="재생성 npz ↔ 배포 parquet 대조")
    ap.add_argument("--episode-id", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--regenerated", required=True)
    ap.add_argument("--published", required=True)
    ap.add_argument("--run-start-seq", type=int, default=0,
                    help="그 에피소드를 만든 실행의 첫 번호 (에피소드 메타 run_start_seq). 재생성이 "
                         "여기서 시작했으므로 재생성 에피소드 수는 번호 − 이 값 + 1 이다")
    args = ap.parse_args()

    import pandas as pd

    seq = episode_seq_of(args.episode_id)
    if not 0 <= args.run_start_seq <= seq:
        print(f"run_start_seq {args.run_start_seq} 이 에피소드 번호 {seq} 보다 큽니다(또는 음수입니다) — "
              "그 실행은 이 에피소드를 만들 수 없습니다")
        return 1
    expected = seq - args.run_start_seq + 1
    regen = sorted(Path(args.regenerated).glob("*.npz"))
    if len(regen) != expected:
        print(f"재생성 에피소드 수가 다릅니다: {len(regen)} != {expected}")
        return 1
    last = regen[-1]
    meta = json.loads(last.with_suffix(".json").read_text(encoding="utf-8"))
    if meta["episode_id"] != args.episode_id or str(meta["seed"]) != str(args.seed):
        print(f"재생성 결과가 다른 에피소드입니다: {meta['episode_id']} / {meta['seed']}")
        return 1

    root = Path(args.published)
    eps = pd.concat([pd.read_parquet(f)
                     for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))],
                    ignore_index=True)
    row = eps[eps["episode_id"] == args.episode_id]
    if len(row) != 1:
        print(f"배포본에서 에피소드를 찾지 못했습니다: {args.episode_id} ({len(row)}건)")
        return 1
    episode_index = int(row["episode_index"].iloc[0])

    def fail(*lines: str) -> int:
        """Print why the reproduction failed, then -- from here on the published row is known -- whether the
        rerun ran other code than the row names. When the code differs, the first thing to go wrong is
        usually the length of the run, so the note goes with every failure and not with one."""
        for line in lines:
            print(line)
        note = commit_note(row, _REPO_ROOT)
        if note:
            print(note)
        return 1

    data = pd.concat([pd.read_parquet(f)
                      for f in sorted((root / "data").rglob("*.parquet"))],
                     ignore_index=True)
    frames = data[data["episode_index"] == episode_index].sort_values("frame_index")
    if len(frames) != meta["steps"]:
        return fail(f"프레임 수가 다릅니다: 배포 {len(frames)} vs 재생성 {meta['steps']}")

    arrays = dict(np.load(last, allow_pickle=True))
    try:
        pairs = published_pairs(root)
    except ValueError as e:
        return fail(str(e))
    # A column the distribution's own schema lets it leave out, and which it left out, is not compared; it is counted
    # on the success line. A column of the schema's required or conditional lists that is not there fails, as does an
    # optional one that is there and differs. An optional column a scenario's required_fields or a rule's inputs name
    # is left out like any other: that the distribution owes it is the validator's C8 to say.
    optional = published_optional_fields(root)
    left_out = [p for p in pairs if p[0] in optional and p[0] not in frames.columns]
    pairs = tuple(p for p in pairs if p not in left_out)
    published: dict[str, np.ndarray] = {}
    for schema_field, npz_name in pairs:
        if schema_field not in frames.columns:
            return fail(f"배포본에 컬럼이 없습니다: {schema_field}")
        if npz_name not in arrays:
            return fail(f"재생성 에피소드에 배열이 없습니다: {npz_name} ({schema_field})")
        published[npz_name] = column_to_array(frames[schema_field].to_numpy(),
                                              np.asarray(arrays[npz_name]))

    # The same switch as the columns': a distribution without meta/cameras.json (v0.1) has
    # no camera step at all, and one with it owes its pack -- when its canon declares one -- as
    # it owes the box and pose columns.
    cameras_published = CAMERAS in published_conditions(root)
    pack = None
    if cameras_published:
        try:
            pack = pack_pair(root, episode_index, meta, last.parent)
        except ValueError as e:
            return fail(str(e))

    problems = compare_arrays(arrays, published, pairs)
    meta_problems, members = compare_episode_meta(meta, row)
    problems += meta_problems
    if pack is not None:
        pack_problems, rows = compare_packs(pack)
        problems += pack_problems
    if problems:
        return fail(f"재현 불일치 {len(problems)} 건:", *("  " + p for p in problems[:10]))
    line = (f"재현 확인: {args.episode_id} · {len(pairs)} 컬럼 · {len(frames)} 프레임 · "
            f"에피소드 메타 {len(compared_episode_fields())} 필드 · notes 멤버 {members}")
    # Said only where a column was left out, so a distribution that carries every column reads as before.
    if left_out:
        line += f" · 배포본이 싣지 않은 선택 열 {len(left_out)}개는 대조하지 않음"
    # Said when there was a policy to compare: the two sides agree here, so the rerun's record stands for both.
    if _at(meta, POLICY) is not _ABSENT:
        line += " · 정책"
    if pack is not None:
        line += f" · 팩 {pack.name} ({' · '.join(PACK_ARRAYS)}) {rows} 행"
    # Said for every camera distribution, not only one with a pack: riding on the pack phrase,
    # it would go with the pack, and a green C9 would read as covering the video.
    if cameras_published:
        line += f" · {NOT_COMPARED}"
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
