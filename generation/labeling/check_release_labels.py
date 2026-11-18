"""Re-derive every episode's label from a distribution alone, and compare it with the label
the distribution carries.

The card tells a recipient 「라벨을 다시 유도하려면 rules.json 과 프레임으로 규칙을 직접
돌리십시오」, and the validator compares labels with each other, never with the time series
(card, 「v0.1 의 한계」). This is the step that does what the card says, for every episode:
the production labeller (label_episodes.label_episode), fed from the files a download holds
and from nothing else --

    meta/info.json            fps, the step period
    meta/rules.json           the rules and their thresholds as shipped (narrowed to the batch)
    meta/scenarios.json       which rules each scenario declares, and which declares a conditions table
    meta/safety_schema.json   the names safety.event may carry
    meta/cameras.json         where shipped: when a robot camera saw the person, by its seeing rule
    meta/episodes/*.parquet   the recorded label, the notes, the evidence of a table scenario's episode,
                              and where each episode's rows are
    data/*.parquet            the step columns the rules and the evidence read

-- not the recording (npz, sidecar), and not this repository's canon. An episode of a scenario with a
conditions table carries its evidence (generation/labeling/evidence.py) in its evidence columns, and the
labeller works it out again from the same files.

What agreement says. The label follows from the release *by the labeller's own computation*:
the conversion and the assembly dropped nothing the rules read, the rows the episode table
points at are that episode's, and the shipped thresholds give the labels the release carries
(not that they are the numbers the labels were made with -- a threshold moved without moving
a verdict does not show). It is the same code on both sides, so a rule the labeller misreads
is misread here too; an independent reading of the rules is a different check.

Read as a loader reads it. The statement is about what a recipient gets, so the tables are the
files LeRobot loads, in stored order, and the labels are derived twice where it matters: from
the numbers as stored, and from the numbers as a LeRobot item carries them. Its item transform
makes a torch tensor of every value, which is float32 for any float (lerobot 0.6.1
io_utils.hf_transform_to_torch; measured: observation.contact.force, stored as float64, arrives
as torch.float32). An episode whose rule inputs survive that unchanged is labelled once; one
whose inputs round is labelled again from the rounded numbers, and its outcome, severity and
triggered rules have to come out the same. The numbers quoted in the notes (a peak force in
not_evaluated) and the evidence are held to the stored values only -- a ratio worked out again from
narrowed forces differs in its last digits, and is not the number the release carries. It is the labeller that is run on the
loaded numbers, comparing in float64 as it always does; a recipient who compares the float32
tensor itself with a threshold has the threshold narrowed too, and can come out differently on
a value within a float32 step of it (torch.tensor([0.1499999999]) <= 0.15 is true).

What this does not look at:
  - a column neither a rule nor the evidence reads (joint state, com/zmp, v_rel, images, boxes);
  - whether safety.separation_violation follows from the two distance columns -- taken as
    recorded here, checked step by step by the validator (C8);
  - whether an event was recorded at all. A release names the events that fired and its schema
    the names allowed; the recorder's own list (the sidecar's safety_event_names) is not
    shipped, so "never fired" and "not recorded" read the same from a download;
  - the timestamp column. LeRobot writes it as float32, which is not evenly spaced to the
    labeller's tolerance past the first seconds, so the period is 1 / fps. The recording's own
    period is not shipped: the converter takes one within 1e-9 s of 1 / fps, and a label that
    turned on the remainder (a frame budget one frame shorter) comes out here as a difference.

Isaac-free and lerobot-free: pyarrow reads the parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Run as a script, sys.path[0] is this file's directory (see label_episodes.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generation.field_map import STEP_FIELDS, episode_fields
from generation.labeling import evidence
from generation.labeling.label_episodes import label_episode, selected_rules

# The label members that are episode columns of every episode; the converter puts the rest of what
# label_episode returns into notes under the same names (convert_lerobot.build_episode_meta), and the
# evidence of a table scenario's episode into its evidence columns.
LABEL_COLUMNS = tuple(f.schema for f in episode_fields(False) if f.source == "label")

EPISODE_COLUMNS = ("episode_index", "episode_id", "scenario_id", "length", "data/chunk_index",
                   "data/file_index", "dataset_from_index", "dataset_to_index", "termination",
                   "notes", *LABEL_COLUMNS)

# The evidence columns of a scenario with a conditions table: read where the episode table has them.
EVIDENCE_COLUMNS = evidence.EVIDENCE_COLUMNS

# The pointers from an episode to its rows (check_spans).
INTEGER_COLUMNS = ("episode_index", "length", "data/chunk_index", "data/file_index",
                   "dataset_from_index", "dataset_to_index")

# Every step column label_episode reads, plus the three that say whose row it is.
STEP_COLUMNS = ("index", "episode_index", "frame_index", "observation.root_vel",
                "observation.contact.force", "observation.contact.type",
                "observation.contact.region_names", "safety.separation_violation", "safety.event",
                "safety.min_distance_human", "safety.separation_required")

# The step columns the evidence reads where a release has them -- the person's pelvis and each camera's pose and
# visible fraction (evidence.seen_steps reads the robot cameras' of these) -- by the recording's array name.
SEEN_ARRAYS = {f.schema: f.npz for f in STEP_FIELDS
               if f.schema == "observation.human.root_pose"
               or f.schema.startswith(("observation.camera_pose.", "observation.human_visible_fraction."))}

# LeRobot v3.0's layout, for a meta/info.json that does not spell it out.
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

# The parquet files of a table are the ones LeRobot loads, in the order it loads them:
# sorted(dir.glob("*/*.parquet")), one directory down and nowhere else (lerobot 0.6.1
# datasets/io_utils.load_nested_dataset, for data/ and meta/episodes/ alike). A file at another
# depth is not part of what a loader gets, so it is not part of what is read here -- read with
# it, a table that is whole only by that file passes for whole.
LOADED = "*/*.parquet"

SHOWN = 40      # differences printed; the report file carries all of them

# The float inputs of label_episode's verdict, which a LeRobot item hands out as float32 whatever the
# column stores (as_loaded). The evidence's other inputs (the distances, the poses, the visible fractions)
# are not among them: the evidence is held to the stored numbers only.
FLOAT_INPUTS = ("contact_force", "root_vel")

# Characters that make `datasets` take a file path for a pattern (fsspec's has_magic). LeRobot
# hands it the parquet files' paths under the root as the root was given -- not resolved, a
# symlink's own name kept; a relative one is joined to the working directory -- and a path that
# holds one of these is matched against its neighbours: measured, beside "x?y" a sibling "x0y"
# is read first, and beside "x[1]" only the sibling "x1" is read. What is checked here would
# not be what loads. So it is the path as given that is looked at (check), joined to the working
# directory and nothing more.
GLOB_CHARACTERS = "*?["


class ReleaseError(ValueError):
    """The directory cannot be read as a distribution -- nothing was compared."""


def _json_file(root: Path, rel: str) -> dict:
    path = root / rel
    if not path.is_file():
        raise ReleaseError(f"{rel} 이 없습니다")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ReleaseError(f"{rel} 을 JSON 으로 읽을 수 없습니다: {e}") from e
    if not isinstance(doc, dict):
        raise ReleaseError(f"{rel} 이 JSON 객체가 아닙니다")
    return doc


def read_episodes(root: Path) -> list[dict]:
    """The episode table's rows, in the order they are stored -- with the evidence columns where a file has them.

    Stored order is the order that counts: LeRobot takes episode k's pointers from the k-th
    row (``meta.episodes[k]``), whatever that row's episode_index says. Sorted here, a table
    whose rows lie out of place would read as the table it should have been.
    """
    files = sorted((root / "meta" / "episodes").glob(LOADED))
    if not files:
        raise ReleaseError("meta/episodes 에 적재기가 읽는 parquet(<chunk>/<file>.parquet)이 없습니다")
    rows: list[dict] = []
    missing: set[str] = set()
    for f in files:
        names = pq.read_schema(f).names
        missing |= set(EPISODE_COLUMNS) - set(names)
        wanted = [c for c in (*EPISODE_COLUMNS, *EVIDENCE_COLUMNS) if c in names]
        rows += pq.read_table(f, columns=wanted).to_pylist()
    # Asked first: an empty table that also lacks columns is an empty release, and "0 of 0
    # differ" must not be how that reads.
    if not rows:
        raise ReleaseError("에피소드가 없습니다 — 대조할 것이 없는 것은 일치가 아닙니다")
    if missing:
        raise ReleaseError(f"에피소드 표에 열이 없습니다: {sorted(missing)}")
    for r in rows:
        loose = [c for c in INTEGER_COLUMNS if not isinstance(r[c], int) or isinstance(r[c], bool)]
        if loose:
            raise ReleaseError(f"에피소드 표의 {loose} 가 정수가 아닙니다 "
                               f"(episode_index {r['episode_index']!r})")
    stray = [(k, r["episode_index"]) for k, r in enumerate(rows) if r["episode_index"] != k]
    if stray:
        k, e = stray[0]
        raise ReleaseError(f"에피소드 표의 episode_index 가 행의 자리와 다릅니다 — {k}번째 행이 "
                           f"{e} (어긋난 행 {len(stray)}개). 적재기는 k번 에피소드를 k번째 행에서 읽습니다")
    return rows


def _numeric(column: pa.ChunkedArray, name: str) -> np.ndarray:
    """A numeric step column as an (N,) or (N, k) array, through the column's flat buffer and
    copied out of Arrow's memory -- not row by row."""
    col = column.combine_chunks()
    if col.null_count:
        raise ReleaseError(f"{name} 에 null 이 {col.null_count}개 있습니다")
    kind = col.type
    if pa.types.is_fixed_size_list(kind) or pa.types.is_list(kind) or pa.types.is_large_list(kind):
        if not pa.types.is_fixed_size_list(kind):
            lengths = np.array(pc.list_value_length(col).to_numpy(zero_copy_only=False))
            if lengths.size and (lengths != lengths[0]).any():
                raise ReleaseError(f"{name} 의 길이가 행마다 다릅니다")
        flat = col.flatten()
        if flat.null_count:
            raise ReleaseError(f"{name} 의 원소에 null 이 {flat.null_count}개 있습니다")
        return np.array(flat.to_numpy(zero_copy_only=False)).reshape(len(col), -1)
    return np.array(col.to_numpy(zero_copy_only=False))


def _json_column(column: pa.ChunkedArray, name: str) -> tuple[list, np.ndarray]:
    """A JSON-string step column as (its distinct values, decoded; each row's index into them)."""
    col = column.combine_chunks()
    if col.null_count:
        raise ReleaseError(f"{name} 에 null 이 {col.null_count}개 있습니다")
    if not (pa.types.is_string(col.type) or pa.types.is_large_string(col.type)):
        raise ReleaseError(f"{name} 이 문자열 열이 아닙니다: {col.type}")
    encoded = col.dictionary_encode()
    try:
        values = [json.loads(s) for s in encoded.dictionary.to_pylist()]
    except json.JSONDecodeError as e:
        raise ReleaseError(f"{name} 에 JSON 이 아닌 값이 있습니다: {e}") from e
    bad = [v for v in values if not (isinstance(v, list) and all(isinstance(x, str) for x in v))]
    if bad:
        raise ReleaseError(f"{name} 에 문자열 목록이 아닌 값이 있습니다: {bad[0]!r}")
    return values, np.array(encoded.indices.to_numpy(zero_copy_only=False))


def _place(column: pa.ChunkedArray, name: str) -> np.ndarray:
    """A column of positions (index, episode_index, frame_index) as one integer a step: the rows
    are read by these, and a float that equals a position is not the integer a loader takes."""
    values = _numeric(column, name)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ReleaseError(f"{name} 이 스텝마다 정수 하나인 열이 아닙니다: {column.type}")
    return values


def _vector(column: pa.ChunkedArray, name: str) -> np.ndarray:
    """A step column that holds several numbers a step, as (N, k)."""
    values = _numeric(column, name)
    if values.ndim != 2:
        raise ReleaseError(f"{name} 이 벡터 열이 아닙니다: {column.type}")
    return values


def read_steps(root: Path) -> dict:
    """The step columns the rules and the evidence read, every data file together, in the order they are stored.
    The evidence's seen columns (SEEN_ARRAYS) are read as the first file has them, under the recording's names.

    As with the episode table, stored order is what a loader reads by: the files in name
    order, the rows as they lie, and dataset_from_index..dataset_to_index as positions in
    that. ``index`` has to say the same, and is checked rather than sorted by.
    """
    files = sorted((root / "data").glob(LOADED))
    if not files:
        raise ReleaseError("data 에 적재기가 읽는 parquet(<chunk>/<file>.parquet)이 없습니다")
    # The columns the evidence reads where the release has them: as the first file has them, in every file.
    seen_columns = [c for c in SEEN_ARRAYS if c in pq.read_schema(files[0]).names]
    columns = [*STEP_COLUMNS, *seen_columns]
    tables, source = [], []
    for k, f in enumerate(files):
        absent = sorted(set(columns) - set(pq.read_schema(f).names))
        if absent:
            raise ReleaseError(f"{f.relative_to(root)} 에 열이 없습니다: {absent}")
        tables.append(pq.read_table(f, columns=columns))
        source.append(np.full(tables[-1].num_rows, k))
    try:
        table = pa.concat_tables(tables)
    except pa.ArrowInvalid as e:
        raise ReleaseError("data 의 파일마다 열의 형이 다릅니다 — "
                           + " / ".join(str(e).splitlines()[:1])) from e
    n = table.num_rows
    if n == 0:
        raise ReleaseError("스텝 행이 없습니다")

    index = _place(table["index"], "index")
    stray = np.flatnonzero(index != np.arange(n))
    if stray.size:
        k = int(stray[0])
        raise ReleaseError(f"index 열이 행의 자리와 다릅니다 — {k}번째 행이 {index[k].item()!r} "
                           f"(어긋난 행 {stray.size}개). 적재기는 행을 자리로 읽습니다")
    kinds, kind_of = _json_column(table["observation.contact.type"], "observation.contact.type")
    regions, region_of = _json_column(table["observation.contact.region_names"],
                                      "observation.contact.region_names")
    events, event_of = _json_column(table["safety.event"], "safety.event")
    violation = _numeric(table["safety.separation_violation"], "safety.separation_violation")
    if violation.dtype != np.bool_:
        raise ReleaseError(f"safety.separation_violation 이 bool 이 아닙니다: {violation.dtype}")
    # One value a step, as a scalar or a list of one. Of a wider list only an element could be
    # handed on, and the labeller would never see the rest.
    if violation.reshape(n, -1).shape[1] != 1:
        raise ReleaseError("safety.separation_violation 이 스텝마다 값 하나가 아닙니다: "
                           f"{table['safety.separation_violation'].type}")
    # A visible fraction is one number a step, a pose seven.
    seen = {SEEN_ARRAYS[c]: (_one_a_step(_numeric(table[c], c), table[c].type, c)
                             if c.startswith("observation.human_visible_fraction.") else _vector(table[c], c))
            for c in seen_columns}
    return {
        "files": files, "rows": n,
        "file": np.concatenate(source),
        "episode_index": _place(table["episode_index"], "episode_index"),
        "frame_index": _place(table["frame_index"], "frame_index"),
        "root_vel": _vector(table["observation.root_vel"], "observation.root_vel"),
        "force": _vector(table["observation.contact.force"], "observation.contact.force"),
        "violation": violation.reshape(n),
        "min_distance_human": _one_a_step(_numeric(table["safety.min_distance_human"], "safety.min_distance_human"),
                                          table["safety.min_distance_human"].type, "safety.min_distance_human"),
        "separation_required": _one_a_step(
            _numeric(table["safety.separation_required"], "safety.separation_required"),
            table["safety.separation_required"].type, "safety.separation_required"),
        "seen": seen,
        "kinds": kinds, "kind_of": kind_of,
        "regions": regions, "region_of": region_of,
        "events": events, "event_of": event_of,
    }


def _one_a_step(values: np.ndarray, kind: pa.DataType, name: str) -> np.ndarray:
    """A step column of one number a step -- a scalar or a list of one -- as (N,)."""
    n = len(values)
    if values.reshape(n, -1).shape[1] != 1:
        raise ReleaseError(f"{name} 이 스텝마다 값 하나가 아닙니다: {kind}")
    return values.reshape(n)


def check_spans(root: Path, info: dict, episodes: list[dict], steps: dict) -> None:
    """Every episode's rows are its own, and every row is some episode's.

    The label and the rows meet only through these pointers. Read loosely -- by the rows'
    own episode_index, say -- a table that sends a loader to another episode's frames would
    still compare equal here.
    """
    template = info.get("data_path") or DATA_PATH
    known = {f.resolve(): k for k, f in enumerate(steps["files"])}
    claimed = np.zeros(steps["rows"], dtype=bool)
    for row in episodes:
        e, a, b = row["episode_index"], row["dataset_from_index"], row["dataset_to_index"]
        if not 0 <= a < b <= steps["rows"]:
            raise ReleaseError(f"에피소드 {e}: dataset_from_index..dataset_to_index = {a}..{b} 가 "
                               f"스텝 표({steps['rows']}행) 안의 구간이 아닙니다")
        if b - a != row["length"]:
            raise ReleaseError(f"에피소드 {e}: length {row['length']} 인데 가리키는 구간은 {b - a}행입니다")
        owners = np.unique(steps["episode_index"][a:b])
        if owners.tolist() != [e]:
            raise ReleaseError(f"에피소드 {e}: 가리키는 행 {a}..{b} 의 episode_index 가 "
                               f"{owners[:5].tolist()} 입니다")
        if not np.array_equal(steps["frame_index"][a:b], np.arange(b - a)):
            raise ReleaseError(f"에피소드 {e}: frame_index 가 0..{b - a - 1} 이 아닙니다")
        try:
            named = template.format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])
            target = (root / named).resolve()
        except (KeyError, IndexError, AttributeError, TypeError, ValueError) as err:
            # The template is the release's own text (meta/info.json data_path): a field the loader
            # does not give, a positional or malformed one, a value that is not text -- or a value
            # the file index cannot fill in.
            raise ReleaseError(f"에피소드 {e}: meta/info.json 의 data_path {template!r} 에 이 에피소드의 "
                               f"data/chunk_index={row['data/chunk_index']!r}·data/file_index="
                               f"{row['data/file_index']!r} 를 채울 수 없습니다 ({type(err).__name__}: {err})") from err
        k = known.get(target)
        if k is None:
            raise ReleaseError(f"에피소드 {e}: data/chunk_index·data/file_index 가 가리키는 "
                               f"{named} 이 없습니다")
        held = np.unique(steps["file"][a:b])
        if held.tolist() != [k]:
            where = [str(steps["files"][i].relative_to(root)) for i in held[:3]]
            raise ReleaseError(f"에피소드 {e}: 행이 data/file_index 가 가리키는 {named} 이 아니라 "
                               f"{where} 에 있습니다")
        # Two episodes cannot both pass the owner check above on one row, so no overlap is left
        # to look for; what remains is a row nobody points at.
        claimed[a:b] = True
    orphans = int((~claimed).sum())
    if orphans:
        raise ReleaseError(f"어느 에피소드에도 속하지 않는 스텝 행이 {orphans}개 있습니다")


def event_rows(steps: dict, names: list[str]) -> np.ndarray:
    """One bool row over ``names`` per distinct safety.event value."""
    unknown = sorted({x for value in steps["events"] for x in value} - set(names))
    if unknown:
        raise ReleaseError("safety.event 에 meta/safety_schema.json 의 safety_events 에 없는 "
                           f"이름이 있습니다: {unknown}")
    return np.array([[name in value for name in names] for value in steps["events"]],
                    dtype=bool).reshape(len(steps["events"]), len(names))


def episode_inputs(row: dict, notes: dict, steps: dict, fps: int, names: list[str],
                   fired: np.ndarray) -> tuple[dict, dict]:
    """(arrays, meta) for label_episode, from this episode's rows and its notes."""
    e, a, b = row["episode_index"], row["dataset_from_index"], row["dataset_to_index"]
    # By value: two spellings of one JSON list are the same names.
    spelled = [steps["regions"][int(i)] for i in np.unique(steps["region_of"][a:b])]
    regions = spelled[0]
    if any(other != regions for other in spelled[1:]):
        raise ReleaseError(f"에피소드 {e}: observation.contact.region_names 가 스텝마다 다릅니다")
    distinct, back = np.unique(steps["kind_of"][a:b], return_inverse=True)
    kinds = [steps["kinds"][int(i)] for i in distinct]
    force = steps["force"][a:b]
    if any(len(k) != len(regions) for k in kinds) or force.shape[1] != len(regions):
        raise ReleaseError(f"에피소드 {e}: observation.contact.type·force 의 폭이 region_names "
                           f"({len(regions)}개)와 다릅니다")
    arrays = {
        "timestamp": np.arange(b - a) / fps,
        "safety_event": fired[steps["event_of"][a:b]],
        "root_vel": steps["root_vel"][a:b],
        "separation_violation": steps["violation"][a:b],
        "contact_force": force,
        "contact_type": np.array(kinds)[back.reshape(-1)],
        "min_distance_human": steps["min_distance_human"][a:b],
        "separation_required": steps["separation_required"][a:b],
        # By the recording's names: the evidence reads them as it reads a recording.
        **{name: values[a:b] for name, values in steps["seen"].items()},
    }
    termination = json.loads(row["termination"])
    if not isinstance(termination, dict):
        raise ValueError(f"termination 이 JSON 객체가 아닙니다: {termination!r}")
    meta = {
        "safety_event_names": list(names),
        "contact_region_names": list(regions),
        "termination": termination,
        "contact_onset_unresolved_reason": notes.get("contact_onset_unresolved_reason"),
        # The person's stature the evidence works the height in the picture out with (null: it cannot tell).
        "human_stature_m": notes.get("human_stature_m"),
    }
    # Left out when the notes do not carry a step, so that the labeller refuses the episode in
    # its own words rather than this adapter choosing a release step for it.
    release = notes.get("hold_release_step")
    if isinstance(release, int) and not isinstance(release, bool):
        meta["hold_release_step"] = release
    return arrays, meta


def as_loaded(arrays: dict) -> dict | None:
    """``arrays`` with the numbers a LeRobot item carries -- every float narrowed to float32 --
    or None when that changes none of them, and the label derived from them is the one already
    derived."""
    narrowed, changed = dict(arrays), False
    for name in FLOAT_INPUTS:
        stored = np.asarray(arrays[name])
        if stored.dtype.kind != "f" or stored.dtype.itemsize <= 4:
            continue
        with np.errstate(over="ignore"):        # past float32's range it is inf, as the tensor is
            narrowed[name] = stored.astype(np.float32)
        changed = changed or not np.array_equal(narrowed[name].astype(stored.dtype), stored,
                                                equal_nan=True)
    return narrowed if changed else None


def _carried_evidence(row: dict) -> dict:
    """The evidence an episode's row carries, under the label's names. A value that is not there is a null, or a
    NaN read as one -- a column writes no value as NaN and pandas writes that as a null."""
    absent = [c for c in EVIDENCE_COLUMNS if c not in row]
    if absent:
        raise ValueError(f"에피소드 표에 근거 요약 열 {absent} 이 없습니다 — 실린 근거 요약과 대조할 수 없습니다")
    values = {name: row[evidence.PREFIX + name] for name in evidence.NAMES}
    return {name: None if isinstance(v, float) and v != v else v for name, v in values.items()}


def recorded_label(row: dict, notes: dict, members: list[str]) -> dict:
    """The label the release carries, member for member what label_episode returns: the label columns, the evidence
    from its columns, the rest from the notes."""
    absent = [m for m in members if m not in LABEL_COLUMNS and m != evidence.EVIDENCE_KEY and m not in notes]
    if absent:
        raise ValueError(f"notes 에 {absent} 이 없습니다 — 실린 라벨과 대조할 수 없습니다")
    return {m: (row[m] if m in LABEL_COLUMNS else _carried_evidence(row) if m == evidence.EVIDENCE_KEY
                else notes[m]) for m in members}


def _spelled(value: object) -> str:
    """A label member as canonical JSON. Members are compared in this form and not with ==,
    which calls 0 and false, or 1 and true, the same value."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _tally(labels: list[dict]) -> dict:
    def unevaluated(label):
        entries = label.get("not_evaluated") or []
        return sorted({n.get("rule_id") for n in entries if isinstance(n, dict)} - {None})

    return {
        "outcome": dict(Counter(l["outcome"] for l in labels)),
        "severity": dict(Counter(l["severity"] for l in labels)),
        "triggered_rules": dict(Counter(r for l in labels for r in l["triggered_rules"] or [])),
        "not_evaluated": dict(Counter(r for l in labels for r in unevaluated(l))),
    }


def check(root: Path) -> dict:
    """Label every episode of the distribution at ``root`` from the distribution, and report
    where the label it carries differs. Raises ReleaseError when it cannot be read as one."""
    # What a loader given this root passes on: joined to the working directory when relative, and
    # otherwise as it is. Not resolved and not tidied -- a symlink's own name and a "name/.." on the
    # way are both in the string that gets matched.
    handed = os.path.join(os.getcwd(), os.fspath(root))
    root = Path(root)
    magic = sorted(set(GLOB_CHARACTERS) & set(handed))
    if magic:
        raise ReleaseError(f"배포본의 경로에 glob 문자 {magic} 가 있습니다 — 이 경로를 받은 적재기(datasets)는 파일 "
                           f"경로를 패턴으로 풀어 이웃 디렉터리의 파일을 읽습니다: {handed}")
    info = _json_file(root, "meta/info.json")
    rules = _json_file(root, "meta/rules.json")
    scenarios = {s.get("scenario_id"): s
                 for s in _json_file(root, "meta/scenarios.json").get("scenarios") or []}
    names = _json_file(root, "meta/safety_schema.json").get("safety_events")
    if not (isinstance(names, list) and names and all(isinstance(x, str) for x in names)):
        raise ReleaseError("meta/safety_schema.json 에 safety_events 목록이 없습니다")
    fps = info.get("fps")
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        raise ReleaseError(f"meta/info.json 의 fps 가 양의 정수가 아닙니다: {fps!r}")
    # The cameras canon the evidence tells by when a robot camera saw the person; a release without one cannot.
    cameras = _json_file(root, "meta/cameras.json") if (root / "meta" / "cameras.json").exists() else None

    episodes = read_episodes(root)
    steps = read_steps(root)
    check_spans(root, info, episodes, steps)
    fired = event_rows(steps, names)

    applied: dict[str, list[str]] = {}
    recorded, derived, differences, refused = [], [], [], []
    narrowed = 0
    for row in episodes:
        who = {"episode_index": row["episode_index"], "episode_id": row["episode_id"]}
        carried = {m: row[m] for m in LABEL_COLUMNS}
        try:
            notes = json.loads(row["notes"])
            if not isinstance(notes, dict):
                raise ValueError("notes 가 JSON 객체가 아닙니다")
            carried["not_evaluated"] = notes.get("not_evaluated")
            scenario = scenarios.get(row["scenario_id"])
            if scenario is None:
                raise ValueError(f"meta/scenarios.json 에 없는 시나리오입니다: {row['scenario_id']}")
            arrays, meta = episode_inputs(row, notes, steps, fps, names, fired)
            label = json.loads(json.dumps(label_episode(arrays, meta, rules, scenario, cameras),
                                          ensure_ascii=False))
            carried = recorded_label(row, notes, list(label))
            rounded = as_loaded(arrays)
            narrowed += rounded is not None          # counted before the second labelling can refuse
            loaded = label if rounded is None else json.loads(json.dumps(
                label_episode(rounded, meta, rules, scenario, cameras), ensure_ascii=False))
        except ReleaseError:
            raise
        except (ValueError, KeyError, TypeError) as e:
            # Not skipped: an episode the labeller will not label from the release is one whose
            # label a recipient cannot re-derive, which is the finding.
            recorded.append(carried)
            refused.append({**who, "reason": str(e)})
            continue
        recorded.append(carried)
        derived.append(label)
        if row["scenario_id"] not in applied:
            applied[row["scenario_id"]] = sorted(r["rule_id"]
                                                 for r in selected_rules(rules, scenario))
        differences += [{**who, "member": m, "recorded": carried[m], "derived": label[m],
                         "read": "stored"}
                        for m in label if _spelled(carried[m]) != _spelled(label[m])]
        # The second reading, where it is a different one: the verdict a recipient reaches from
        # LeRobot items. Reported for the members rounding moved, so that a label wrong in both
        # readings is listed once.
        differences += [{**who, "member": m, "recorded": carried[m], "derived": loaded[m],
                         "read": "loader"}
                        for m in LABEL_COLUMNS
                        if _spelled(loaded[m]) != _spelled(label[m])
                        and _spelled(carried[m]) != _spelled(loaded[m])]

    differing = {d["episode_index"] for d in differences}
    return {
        "dataset": str(root),
        "episodes": len(episodes),
        "steps": steps["rows"],
        "fps": fps,
        "rules": applied,
        "recorded": _tally(recorded),
        "derived": _tally(derived),
        "agree": len(episodes) - len(differing) - len(refused),
        "narrowed": narrowed,
        "differences": differences,
        "refused": refused,
    }


def print_report(report: dict) -> None:
    print(f"배포본 {report['dataset']}")
    print(f"  에피소드 {report['episodes']:,} · 스텝 {report['steps']:,} · {report['fps']} fps")
    for sid, ids in sorted(report["rules"].items()):
        print(f"  {sid} 에 적용한 규칙 {len(ids)}개: {', '.join(ids)}")
    print(f"  {'':<34}{'실린 라벨':>10}{'다시 유도':>10}")
    for member in ("outcome", "severity", "triggered_rules", "not_evaluated"):
        had, got = report["recorded"][member], report["derived"][member]
        for key in sorted(set(had) | set(got)):
            print(f"  {member + ' ' + str(key):<34}{had.get(key, 0):>12}{got.get(key, 0):>12}")
    for r in report["refused"]:
        print(f"  라벨러가 받지 않음 — {r['episode_id']} (episode_index {r['episode_index']}): "
              f"{r['reason']}")
    print(f"  적재기가 float32 로 넘기면 규칙이 읽는 수가 달라지는 에피소드 {report['narrowed']:,}건"
          + (" — 그 수로도 라벨을 다시 유도했습니다" if report["narrowed"] else ""))
    for d in report["differences"][:SHOWN]:
        how = " (적재기가 넘기는 float32 값으로)" if d["read"] == "loader" else ""
        print(f"  다름 — {d['episode_id']} (episode_index {d['episode_index']}) {d['member']}{how}: "
              f"실린 값 {json.dumps(d['recorded'], ensure_ascii=False)} · "
              f"다시 유도한 값 {json.dumps(d['derived'], ensure_ascii=False)}")
    if len(report["differences"]) > SHOWN:
        print(f"  … 외 {len(report['differences']) - SHOWN}건 (전부는 --out 의 보고에)")
    print(f"실린 라벨이 배포본에서 다시 나온 에피소드 {report['agree']:,} / {report['episodes']:,}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="배포본만으로 에피소드마다 라벨을 다시 유도해 실린 라벨과 대조한다 "
                    "(종료 코드 0 전부 일치 · 1 다르거나 유도하지 못한 에피소드가 있음 · "
                    "2 배포본으로 읽을 수 없음)")
    ap.add_argument("--dataset", required=True, help="배포본 디렉터리 (LeRobotDataset v3.0 루트)")
    ap.add_argument("--out", default=None, help="보고를 JSON 으로 남길 경로")
    args = ap.parse_args(argv)
    try:
        report = check(Path(args.dataset))
    except ReleaseError as e:
        print(f"배포본으로 읽을 수 없습니다 — {e}")
        return 2
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    print_report(report)
    return 0 if report["agree"] == report["episodes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
