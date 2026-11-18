"""The directory a public batch is made in: its record, its chunks and their seals.

    <batch>/
      batch.json                what the batch is: its scenarios and their rows, chunk size, generator commit
      LOCK                      held with flock for the whole run of a runner (the kernel lets go when it dies)
      RUNNER                    the runner's PID, and its stage's process group while one runs
      STOP                      made by a person: finish the chunk at hand, then stop
      chunks/<key>/             <first row, 4 digits> or <scenario>-<3 digits> (ChunkPlan.key)
        episodes/               npz, sidecars, labels; media/*.mkv until the seal
        dataset/                the chunk's own distribution
        logs/
        SEALED.json             written last: with it the chunk is done
      dataset/                  the assembled distribution (assemble_batch.py)

A batch is one or more parts, each a scenario and the rows (seqs) it makes. The chunks cut each part's rows, in the
order given, chunk_episodes at a time (Batch.chunks).

* A scenario without a conditions table (S1a, S1d) is a batch's only part, and its rows are 0..n-1. A chunk of it
  is one run of the generator (run_s1a.py --seq <first row> --episodes <n>): reproduction holds per run, so a
  chunk is the unit that can be redone, and its first row is every episode's run_start_seq. The chunk is named by
  that row.
* A scenario with a conditions table (generation/conditions.py) runs one row per run of the generator, so every
  episode's run_start_seq is its own row; its rows are any of the table's, each once, in any order. Its chunks are
  named <scenario>-<their place in the part>, and the parts take turns: the first chunk of every part, then the
  second of every part, and so on -- a batch stopped part way holds every scenario up to about the same row.
  A batch holds condition-table scenarios only, or one scenario without a table.

A record that names one ``scenario`` and its number of ``episodes`` instead of ``parts`` is read as the one part of
rows 0..episodes-1.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from generation import conditions
from generation.seeding import episode_seed

BATCH_FILE = "batch.json"
SEAL_FILE = "SEALED.json"
STOP_FILE = "STOP"
LOCK_FILE = "LOCK"
RUNNER_FILE = "RUNNER"
EPISODES_DIR = "episodes"
DATASET_DIR = "dataset"
LOGS_DIR = "logs"
MEDIA_DIR = "media"          # under episodes/: the lossless streams, removed once the chunk is sealed

# batch.json: "parts" ([{"scenario", "rows"}, ...]), these, and "created".
_FIELDS = ("chunk_episodes", "generator_commit", "repo_id")


class BatchError(Exception):
    """The batch directory cannot be used as asked. The message says what and where."""


@dataclass(frozen=True)
class ChunkPlan:
    """One chunk of a batch: the name of its directory, its scenario and the rows (seqs) it makes, in order.
    ``per_row``: a condition-table scenario's chunk, whose every row is a run of the generator of its own."""
    key: str
    scenario: str
    rows: tuple[int, ...]
    per_row: bool

    def run_start(self, seq: int) -> int:
        """The run_start_seq of the chunk's episode ``seq``: the row itself when every row is a run of its own,
        the chunk's first row when the chunk is one run."""
        return seq if self.per_row else self.rows[0]


@dataclass(frozen=True)
class Batch:
    root: Path
    parts: tuple[tuple[str, tuple[int, ...]], ...]
    chunk_episodes: int
    generator_commit: str
    repo_id: str

    @property
    def scenarios(self) -> tuple[str, ...]:
        return tuple(scenario for scenario, _ in self.parts)

    def chunks(self) -> list[ChunkPlan]:
        """Every chunk, in the order they are made: each part's rows cut chunk_episodes at a time (the last cut
        may be short), the parts taking turns."""
        tables = set(conditions.table_scenarios())
        cuts = []
        for scenario, rows in self.parts:
            per_row = scenario in tables
            pieces = [rows[at:at + self.chunk_episodes] for at in range(0, len(rows), self.chunk_episodes)]
            cuts.append([ChunkPlan(key=f"{scenario}-{k:03d}" if per_row else f"{piece[0]:04d}", scenario=scenario,
                                   rows=piece, per_row=per_row) for k, piece in enumerate(pieces)])
        return [plan for turn in itertools.zip_longest(*cuts) for plan in turn if plan is not None]

    def chunk_dir(self, plan: ChunkPlan) -> Path:
        return self.root / "chunks" / plan.key

    def episode_ids(self, plan: ChunkPlan) -> list[str]:
        """The ids the chunk's episodes have to carry (schema §3), in the chunk's order: the seed is the seq's."""
        return [f"{plan.scenario}_{seq:04d}_seed{episode_seed(plan.scenario, seq)}" for seq in plan.rows]


def _table_rows(scenario: str, at: str) -> int:
    """How many rows the conditions table of ``scenario`` has (meta/scenarios.json)."""
    try:
        return conditions.load_conditions(conditions.scenario_entry(scenario)).total
    except ValueError as e:
        raise BatchError(f"{at} — 이 시나리오의 조건표를 읽지 못합니다: {e}") from e


def _checked_parts(given, where: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """``given`` -- pairs of a scenario and its rows -- as a batch keeps them, after checking they make one batch
    (the module's docstring). ``where`` starts every refusal: it names the record the parts were read from."""
    try:
        parts = tuple((scenario, tuple(rows)) for scenario, rows in given)
    except (TypeError, ValueError) as e:
        raise BatchError(f"{where}parts 가 (시나리오, 행 목록) 짝의 목록이 아닙니다 — {type(e).__name__}: {e}") from e
    if not parts:
        raise BatchError(f"{where}parts 가 비어 있습니다 — 배치에는 시나리오가 하나 이상 있어야 합니다")
    tables = set(conditions.table_scenarios())
    seen: dict[str, int] = {}
    for i, (scenario, rows) in enumerate(parts):
        if not isinstance(scenario, str) or not scenario:
            raise BatchError(f"{where}parts[{i}] 의 시나리오가 비어 있지 않은 문자열이 아닙니다: {scenario!r}")
        at = f"{where}parts[{i}] ({scenario})"
        if scenario in seen:
            raise BatchError(f"{at} — 같은 시나리오가 parts[{seen[scenario]}] 에도 있습니다. 한 시나리오의 행은 한 부분에 둡니다")
        seen[scenario] = i
        if not rows:
            raise BatchError(f"{at} 의 행 목록이 비어 있습니다")
        for row in rows:
            if not isinstance(row, int) or isinstance(row, bool):
                raise BatchError(f"{at} 의 행 {row!r} 이 정수가 아닙니다")
        counted: set[int] = set()
        for row in rows:
            if row in counted:
                raise BatchError(f"{at} 의 행 {row} 이 두 번 있습니다 — 행 하나는 에피소드 하나입니다")
            counted.add(row)
        if scenario in tables:
            n = _table_rows(scenario, at)
            outside = next((row for row in rows if not 0 <= row < n), None)
            if outside is not None:
                raise BatchError(f"{at} 의 행 {outside} 이 조건표의 행 0..{n - 1} 밖입니다")
        elif rows != tuple(range(len(rows))):
            k = next(k for k, row in enumerate(rows) if row != k)
            raise BatchError(f"{at} 의 행이 0..{len(rows) - 1} 이 아닙니다 ({k}번째가 {rows[k]}) — 조건표가 없는 시나리오의 "
                             "청크는 생성기의 실행 하나(--seq 첫 행 --episodes 행 수)라 행이 0 부터 차례로 이어져야 합니다")
    first, first_table = parts[0][0], parts[0][0] in tables
    for i, (scenario, _) in enumerate(parts[1:], 1):
        at = f"{where}parts[{i}] ({scenario})"
        if (scenario in tables) != first_table:
            raise BatchError(f"{at} — 조건표 시나리오와 조건표가 없는 시나리오를 한 배치에 섞지 않습니다 (parts[0] 의 "
                             f"{first} 는 조건표가 {'있는' if first_table else '없는'} 시나리오) — 둘은 청크를 만드는 "
                             "방식이 다릅니다")
        if not first_table:
            raise BatchError(f"{at} — 조건표가 없는 시나리오의 배치는 그 시나리오 하나입니다 (parts[0] 이 {first})")
    return parts


def _checked_chunk_episodes(value, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise BatchError(f"{where}chunk_episodes 는 1 이상의 정수여야 합니다: {value!r}")
    return value


def _parts_change(was, asked) -> str | None:
    """Where two batches' parts differ, said shortly (a part may hold hundreds of rows); None when they do not."""
    if was == asked:
        return None
    if [s for s, _ in was] != [s for s, _ in asked]:
        return f"parts: 시나리오 차례가 다릅니다 — 기록 {[s for s, _ in was]} ≠ 요청 {[s for s, _ in asked]}"
    scenario, had, wants = next((s, a, b) for (s, a), (_, b) in zip(was, asked) if a != b)
    k = next((k for k, (a, b) in enumerate(zip(had, wants)) if a != b), min(len(had), len(wants)))

    def at(rows):
        return str(rows[k]) if k < len(rows) else "없음"

    return (f"parts: {scenario} 의 행이 다릅니다 — 기록 {len(had)}행 · 요청 {len(wants)}행, 첫 다른 자리 {k}: "
            f"기록 {at(had)} ≠ 요청 {at(wants)}")


def write_atomically(path: Path, text: str) -> None:
    """Under a temporary name, on disk, then renamed, then the rename on disk: the machine this runs on
    resets under load, and a seal that comes back with no content (the data not yet written when the
    rename was) or not at all would be read as a chunk that is done or never was."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load_batch(root) -> Batch:
    root = Path(root)
    path = root / BATCH_FILE
    if not path.is_file():
        raise BatchError(f"배치 기록이 없습니다: {path} — run_batch.py 가 만든 디렉터리가 아닙니다")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "parts" in doc:
            given = [(part["scenario"], part["rows"]) for part in doc["parts"]]
        else:                               # one scenario and its number of episodes: rows 0..episodes-1
            episodes = doc["episodes"]
            if not isinstance(episodes, int) or isinstance(episodes, bool):
                raise TypeError(f"episodes 가 정수가 아닙니다: {episodes!r}")
            given = [(doc["scenario"], range(episodes))]
        values = {key: doc[key] for key in _FIELDS}
    except (ValueError, KeyError, TypeError) as e:
        raise BatchError(f"{path} 를 배치 기록으로 읽지 못합니다: {type(e).__name__}: {e}") from e
    where = f"{path} 의 "
    _checked_chunk_episodes(values["chunk_episodes"], where)
    return Batch(root=root, parts=_checked_parts(given, where), **values)


def open_batch(root, *, parts, chunk_episodes: int, generator_commit: str, repo_id: str) -> Batch:
    """Record a new batch, or check that the one recorded at ``root`` is the one asked for.

    A batch is continued only as itself: the same parts (the scenarios in the same order, each with the same rows
    in the same order), the same chunk size, and the same generator commit -- episodes of one batch made by two
    versions of the code would reproduce under neither. ``parts`` is pairs of a scenario and its rows.
    """
    root = Path(root)
    asked = Batch(root=root, parts=_checked_parts(parts, ""),
                  chunk_episodes=_checked_chunk_episodes(chunk_episodes, ""),
                  generator_commit=generator_commit, repo_id=repo_id)
    if (root / BATCH_FILE).exists():
        batch = load_batch(root)
        differ = [change for change in [_parts_change(batch.parts, asked.parts)] if change]
        differ += [f"{k}: 기록 {getattr(batch, k)!r} ≠ 요청 {getattr(asked, k)!r}" for k in _FIELDS
                   if getattr(batch, k) != getattr(asked, k)]
        if differ:
            raise BatchError(f"{root / BATCH_FILE} 의 배치와 요청이 다릅니다 — " + "; ".join(differ)
                             + ". 배치는 시작한 커밋의 깨끗한 트리에서, 같은 인자로 이어 돌립니다")
        return batch
    root.mkdir(parents=True, exist_ok=True)
    record = {"parts": [{"scenario": scenario, "rows": list(rows)} for scenario, rows in asked.parts],
              **{k: getattr(asked, k) for k in _FIELDS}, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_atomically(root / BATCH_FILE, json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    return asked


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _walked(root: Path, chunk: Path) -> tuple[list[Path], list[Path]]:
    """What is under ``root``: every entry that is not a directory, and every symbolic link to a
    directory (not gone below). A directory that cannot be listed is an error: passed over, the files
    under it would be missing from the seal without a word, and a file there can still be opened by
    its path. A ``root`` that is not there holds nothing. (What is not asked again: a real directory
    os.walk took for a file because asking failed is not gone below, and nothing says so.)"""
    def refuse(e: OSError):
        raise BatchError(f"{chunk} 의 데이터셋에서 {Path(e.filename).relative_to(chunk).as_posix()} 디렉터리를 "
                         f"열지 못했습니다 ({e.strerror}) — 봉인은 그 아래의 파일을 세지 못합니다") from e

    entries, links = [], []
    if root.is_dir():
        for here, dirs, names in os.walk(root, onerror=refuse):
            entries += [Path(here, name) for name in names]
            # Asked of every entry again, in whichever list it came, and with a failure raised: os.walk
            # takes an entry for a file when asking whether it is a directory failed, and says nothing.
            links += [p for p in (Path(here, name) for name in dirs + names) if p.is_symlink() and p.is_dir()]
    return sorted(entries), sorted(links)


def sealed_files(chunk: Path) -> dict[str, str]:
    """Relative path -> SHA-256 of everything a sealed chunk keeps: the episodes' arrays, sidecars
    and labels, and every file of its dataset. Not the lossless media, which the seal removes.

    The walk of the dataset does not go below a link. A symbolic link to a file is listed, with the
    hash of the bytes it points at. A symbolic link to a directory inside the dataset is refused: the
    files under one would be in no seal, and the assembly, which opens the dataset by path, would read
    them. So is a directory that cannot be listed, for the same reason (_walked). (The dataset
    directory itself may be a link: its files are listed.) The converter writes real directories it
    can list. episodes/ is listed flat -- the lossless media is a directory there when the
    seal is taken -- and the assembly reads only the files directly under it."""
    chunk = Path(chunk)
    files = [p for p in sorted((chunk / EPISODES_DIR).iterdir()) if p.is_file()]
    held, links = _walked(chunk / DATASET_DIR, chunk)
    if links:
        named = ", ".join(p.relative_to(chunk).as_posix() for p in links)
        raise BatchError(f"{chunk} 의 데이터셋 안에 디렉터리를 가리키는 심링크가 있습니다: {named} — "
                         "봉인은 링크 아래의 파일을 세지 않으므로 그 파일은 바뀌어도 알 수 없습니다. "
                         "데이터셋 안의 디렉터리는 링크가 아니라 실제 디렉터리여야 합니다")
    files += [p for p in held if p.is_file()]
    return {p.relative_to(chunk).as_posix(): sha256_file(p) for p in files}


def write_seal(chunk: Path, record: dict) -> None:
    """The last write of a chunk. Under a temporary name first: a seal cut short would read as
    a chunk that is done."""
    write_atomically(Path(chunk) / SEAL_FILE, json.dumps(record, ensure_ascii=False, indent=1) + "\n")


def read_seal(chunk: Path) -> dict | None:
    """The chunk's seal, or None when it has none. A seal that is there and cannot be read is an
    error: taken for "not sealed", the runner would remove the chunk and make it again."""
    path = Path(chunk) / SEAL_FILE
    if not path.is_file():
        return None
    redo = f" {SEAL_FILE} 를 지우고 다시 돌리면 이 청크를 처음부터 만듭니다 ({path})"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise BatchError(f"{path} 를 읽지 못합니다: {e}." + redo) from e
    if not isinstance(record, dict) or not isinstance(record.get("files"), dict):
        raise BatchError(f"{path} 가 봉인 기록이 아닙니다 (files 목록이 없음)." + redo)
    return record


def remove_media(chunk: Path) -> bool:
    """Remove the chunk's lossless streams; whether there were any."""
    media = Path(chunk) / EPISODES_DIR / MEDIA_DIR
    if not media.exists():
        return False
    shutil.rmtree(media)
    return True


def seal_problems(chunk: Path, record: dict) -> list[str]:
    """How the chunk differs from what its seal recorded; [] when it does not. A chunk that could not
    be sealed as it stands (sealed_files) is an error and not a difference: what is under a link to a
    directory, or in a directory that cannot be listed, cannot be listed again to compare -- whether
    or not the record names files there."""
    was, now = record["files"], sealed_files(chunk)
    out = [f"없어진 파일: {name}" for name in sorted(set(was) - set(now))]
    out += [f"봉인 뒤에 생긴 파일: {name}" for name in sorted(set(now) - set(was))]
    out += [f"바뀐 파일: {name}" for name in sorted(set(was) & set(now)) if was[name] != now[name]]
    return out
