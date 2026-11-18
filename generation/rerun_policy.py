#!/usr/bin/env python3
"""The walking policy and behaviour layer a C9 rerun gives an episode of a scenario with a conditions table.

Such an episode records the policy that walked it and the behaviour layer between its task and that policy
(``sim_backend.policy``, as behavior.policy_record writes it): ``source`` -- the checkpoint Isaac Lab publishes
for behavior.PUBLISHED_POLICY_TASK (``isaaclab-published:rsl_rl/<task>``), or a file by its path in the
repository (``file:<path>``) -- the checkpoint's ``sha256``, and the ``behavior`` spec. A rerun with other ones
makes another episode, so the reproduction hook (generation/rerun_seed.sh) reruns with these:

    .venv/bin/python generation/rerun_policy.py --published <root> --episode-id <id>

prints run_s1a.py's flags for that episode, one per line, and exits 0:

* nothing for a scenario without a conditions table -- the distribution is not read at all, and S1a and S1d
  rerun without these flags -- or for a row that records no policy;
* ``--behavior <spec>`` for the published checkpoint: the run fetches it, and its hash is compared after the
  rerun with the rest of the record (compare_published);
* ``--policy <file> --behavior <spec>`` for a file of the repository whose SHA-256 is the recorded one.

The rerun imports the behaviour's module, so it is held as the checkpoint is: every file that import runs is a
file of the repository -- for ``a.b.c``, ``<repository>/a/__init__.py`` and ``<repository>/a/b/__init__.py``
where those packages have one (or the module ``a.py`` the import takes in place of a directory without one),
and ``<repository>/a/b/c/__init__.py`` or ``<repository>/a/b/c.py`` -- every part of the name an identifier,
every file inside the repository once its links are resolved -- and no file beside them that the import could
take in their place: at every level of the name, ``<part>`` or ``__init__`` with another suffix the import system
loads (bytecode, an extension module -- module_shadows). Found by path alone; nothing is imported here.

What may run is anchored in the repository's own history, not in the distribution: the checkpoint and every
file of the behaviour are held by the path they are named by, before any link is followed -- the path the
loader is handed -- to a file git tracks under that path whose bytes are the blob HEAD holds there, compared by
git's object ids whatever flags the index sets on it; a tracked link is held so by its text, and then its target
alike (committed_problem). The repository has to be the top level of a git checkout -- a directory inside
another checkout is not (checkout_problem). A hash the distribution records comes with the same distribution and
vouches for nothing on its own: a release received into a directory under the repository would otherwise name a
file it brought, or a link of its own to a tracked file whose relative imports then run files beside the link.

Anything else cannot be rerun -- another published task or source, a path outside the repository (absolute,
through ``..`` or a link), a file the repository does not have or one with other bytes, a path git does not
track as it is named or whose bytes are not HEAD's, a repository that is not the top level of a git checkout, a
spec not of the form ``module:name`` or whose module is not a file of the repository, or one with a file the
import could take in its place, a flag that is not one line, no single row of that episode in the table, a
``sim_backend`` that does not read as a JSON object: said on stderr, exit 2, before the simulator boots.

Which scenarios have a table is meta/scenarios.json's (conditions.table_scenarios) -- the same list run_s1a.py
takes ``--policy`` and ``--behavior`` for. The row is read from meta/episodes/**/*.parquet, its ``episode_id``
and ``sim_backend`` columns alone, as compare_published finds the tables.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import posixpath
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

# Run as a script, sys.path[0] is this file's directory, not the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generation import behavior  # noqa: E402
from generation.compare_repro import episode_scenario_of  # noqa: E402

PUBLISHED_SOURCE = f"isaaclab-published:rsl_rl/{behavior.PUBLISHED_POLICY_TASK}"
FILE_SOURCE = "file:"
COLUMNS = ("episode_id", "sim_backend")


def published_policy(root: Path, episode_id: str) -> dict | None:
    """The ``policy`` of the ``sim_backend`` JSON in the one row of ``episode_id`` in the distribution at ``root``
    (meta/episodes/**/*.parquet, its episode_id and sim_backend columns alone), or None when the row records
    none. ValueError when there is no table, a table without those columns or that does not read, not exactly
    one row of the episode, a sim_backend that is not a JSON object, or a policy that is not one."""
    import pandas as pd
    import pyarrow.parquet as pq

    files = sorted((Path(root) / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise ValueError(f"배포본 {root} 에 에피소드 표(meta/episodes/**/*.parquet)가 없습니다")
    frames = []
    for path in files:
        try:
            names = pq.read_schema(path).names
        except (OSError, ValueError) as e:
            raise ValueError(f"에피소드 표 {path} 를 읽지 못합니다: {e}") from e
        missing = [c for c in COLUMNS if c not in names]
        if missing:
            raise ValueError(f"에피소드 표 {path} 에 열 {missing} 이 없습니다")
        try:
            frames.append(pd.read_parquet(path, columns=list(COLUMNS)))
        except (OSError, ValueError) as e:
            raise ValueError(f"에피소드 표 {path} 를 읽지 못합니다: {e}") from e
    table = pd.concat(frames, ignore_index=True)
    rows = table[table["episode_id"] == episode_id]
    if len(rows) != 1:
        raise ValueError(f"배포본의 에피소드 표에 {episode_id} 의 행이 {len(rows)}건입니다 — 하나여야 그 정책을 읽습니다")
    cell = rows["sim_backend"].iloc[0]
    if not isinstance(cell, str):
        raise ValueError(f"{episode_id} 의 sim_backend 가 JSON 글이 아닙니다: {type(cell).__name__}")
    try:
        backend = json.loads(cell)
    except (ValueError, RecursionError) as e:
        raise ValueError(f"{episode_id} 의 sim_backend 가 JSON 으로 읽히지 않습니다: {e}") from e
    if not isinstance(backend, dict):
        raise ValueError(f"{episode_id} 의 sim_backend 가 JSON 객체가 아닙니다")
    policy = backend.get("policy")
    if policy is not None and not isinstance(policy, dict):
        raise ValueError(f"{episode_id} 의 sim_backend.policy 가 JSON 객체가 아닙니다: {policy!r}")
    return policy


# The suffixes, but .py, that the import system loads a module from on this interpreter: bytecode and extension
# modules. A file named for a part of a dotted name and ending in one -- an extension module built for another
# interpreter's version too -- could be taken in place of the tracked source.
_LOADED_SUFFIXES = tuple(suffix for suffix in importlib.machinery.all_suffixes() if suffix != ".py")
_LINK_DEPTH = 40


def _git(root: Path, *args: str, data: bytes = b"") -> subprocess.CompletedProcess:
    """git run in ``root``, bytes in and out: its pathspecs taken as literal paths (a name with glob characters
    would otherwise match tracked files it is not), and none of the caller's GIT_* variables (a GIT_DIR or
    GIT_INDEX_FILE set around this would have git answer for another repository or index)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["git", "--literal-pathspecs", "-C", str(root), *args], env=env, input=data,
                          capture_output=True, timeout=60)


def _first_line(result: subprocess.CompletedProcess) -> str:
    return (result.stderr.decode("utf-8", "replace").strip().splitlines() or [f"종료 코드 {result.returncode}"])[0]


def checkout_problem(repo_root: Path) -> str | None:
    """Why ``repo_root`` is not the top level of a git checkout, or None when it is. A directory inside another
    checkout is not one: git asked there answers for the checkout around it, whose index is not the repository's."""
    root = Path(repo_root).resolve()
    why = f"저장소 {root} 가 git 체크아웃의 뿌리가 아니어서 무엇을 추적하는지 알 수 없습니다"
    try:
        top = _git(root, "rev-parse", "--show-toplevel")
    except (OSError, subprocess.SubprocessError) as e:
        return f"{why} (git 을 돌리지 못함 — {type(e).__name__}: {e}) — 저장소의 뿌리인 git 체크아웃에서 돌리십시오"
    said = os.fsdecode(top.stdout.rstrip(b"\n")) if top.returncode == 0 else ""
    if not said or Path(said).resolve() != root:
        answer = f" (git 이 답한 체크아웃의 뿌리: {said})" if said else ""
        return f"{why}{answer} — 저장소의 뿌리인 git 체크아웃에서 돌리십시오"
    return None


def committed_problem(relative: str, repo_root: Path, *, _depth: int = 0) -> str | None:
    """Why the file named ``relative`` in the repository at ``repo_root`` is not the file HEAD holds there, or None
    when it is. The name is taken as it is given, before any link is followed -- the path the loader is handed: a
    normalized relative path, its directories directories (not links), that git tracks under that very name, and
    whose bytes are the blob HEAD holds at that path -- compared by git's object ids, so a flag the index sets on
    the file (assume-unchanged, skip-worktree: git diff then looks away) does not hide a change. A link is held so
    by its text, then its target alike (to ``_LINK_DEPTH`` links); a link out of the repository, or one that does
    not lead where its text says, is refused. ``repo_root`` has to be the top level of a git checkout
    (checkout_problem)."""
    root = Path(repo_root)
    pure = PurePosixPath(relative) if isinstance(relative, str) else None
    if pure is None or not relative or pure.is_absolute() or any(part in ("", ".", "..") for part in relative.split("/")):
        return f"{relative!r} 이 저장소 안의 정규화된 상대 경로가 아닙니다"
    problem = checkout_problem(root)
    if problem:
        return problem
    path, real_root = root / relative, root.resolve()
    if path.parent.resolve() != real_root.joinpath(*pure.parent.parts):
        return f"{relative} 에 이르는 디렉터리 가운데 링크가 있습니다 (git 이 추적하는 그 경로가 아닌 곳의 파일을 읽게 됩니다)"
    linked = path.is_symlink()
    if not linked and not path.is_file():
        return f"{relative} 이 저장소에 없습니다"
    try:
        if _git(root, "ls-files", "--error-unmatch", "--", relative).returncode != 0:
            return (f"{relative} 은 git 이 추적하는 저장소의 파일이 아닙니다 (저장소에 두기만 한 파일 · 받은 배포본이 "
                    "가져온 파일이나 링크로는 다시 돌리지 않습니다)")
        text = os.readlink(path) if linked else None
        here = _git(root, "hash-object", "--no-filters", "--stdin",
                    data=os.fsencode(text) if linked else path.read_bytes())
        head = _git(root, "rev-parse", "--verify", "--quiet", f"HEAD:{relative}")
    except (OSError, subprocess.SubprocessError) as e:
        return (f"git 을 돌리지 못해 {relative} 이 저장소가 추적하는 파일인지 알 수 없습니다 ({type(e).__name__}: {e}) "
                "— git 체크아웃에서 돌리십시오")
    if here.returncode != 0:
        return f"git 이 {relative} 의 내용을 셈하지 못했습니다 ({_first_line(here)})"
    if head.returncode != 0 or head.stdout.strip() != here.stdout.strip():
        return (f"{relative} 이 HEAD 와 다릅니다 (HEAD 가 그 경로에 둔 바이트가 아닙니다 — 커밋되지 않은 변경은 인덱스의 "
                "표시와 상관없이 다시 돌리는 데 쓰지 않습니다)")
    if not linked:
        return None
    target = posixpath.normpath(posixpath.join(posixpath.dirname(relative), text))
    if posixpath.isabs(text) or target == ".." or target.startswith("../"):
        return f"{relative} 이 저장소 밖을 가리키는 링크입니다 ({text})"
    if os.path.realpath(path) != os.path.realpath(root / target):
        return f"{relative} 의 링크가 그 글자({text})가 말하는 곳으로 이어지지 않습니다"
    if _depth >= _LINK_DEPTH:
        return f"{relative} 에서 링크가 {_LINK_DEPTH} 번 넘게 이어집니다"
    return committed_problem(target, root, _depth=_depth + 1)


def module_files(module: str, repo_root: Path) -> list[Path] | str:
    """The files of the repository at ``repo_root`` that importing the dotted ``module`` runs, outermost first --
    at every level of the name the package's ``__init__.py`` or else the module ``<part>.py``, as the import system
    looks; a directory with neither is a namespace package and runs nothing of its own -- or why there is none:
    a part that is not an identifier (an empty part, a leading dot, a path, a space), or no file for the last
    part. Looked up by path; nothing is imported."""
    parts = module.split(".")
    if not all(part.isidentifier() for part in parts):
        return f"모듈 이름 {module!r} 이 점으로 이은 식별자가 아닙니다"
    root = Path(repo_root)
    files = []
    for depth in range(1, len(parts) + 1):
        base = root.joinpath(*parts[:depth])
        found = next((c for c in (base / "__init__.py", base.parent / f"{parts[depth - 1]}.py") if c.is_file()), None)
        if found is not None:
            files.append(found)
        elif depth == len(parts):
            shown = "/".join(parts)
            return f"모듈 {module!r} 이 저장소의 파일이 아닙니다 ({shown}.py 도 {shown}/__init__.py 도 없음)"
    return files


def _taken_instead(directory: Path, stem: str) -> list[str]:
    """The names in ``directory`` the import system could load for ``stem`` instead of ``stem``.py: ``stem``
    followed by anything that ends in a suffix it loads modules from other than .py (_LOADED_SUFFIXES)."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(name for name in names if name.startswith(f"{stem}.") and name.endswith(_LOADED_SUFFIXES))


def module_shadows(module: str, repo_root: Path) -> list[str]:
    """The files, by their paths in the repository at ``repo_root``, that importing the dotted ``module`` could take
    in place of its sources: at every level of the name, ``<part>`` beside the level's directory and ``__init__``
    inside it with a suffix of bytecode or an extension module -- the import system takes an extension module
    before a .py, and a bare .pyc where no .py is (a namespace level has none at all)."""
    parts, root = module.split("."), Path(repo_root)
    found = []
    for depth in range(1, len(parts) + 1):
        parent = root.joinpath(*parts[:depth - 1])
        base = parent / parts[depth - 1]
        found += [(parent / name).relative_to(root).as_posix() for name in _taken_instead(parent, parts[depth - 1])]
        found += [(base / name).relative_to(root).as_posix() for name in _taken_instead(base, "__init__")]
    return found


def module_problem(module: str, repo_root: Path) -> str | None:
    """Why the dotted ``module`` is not one the rerun may import, or None when it is: every file importing it runs
    (module_files) lies inside ``repo_root`` once its links are resolved, no file beside them could be taken in
    their place (module_shadows), and each is, by the path the import names it by, the file HEAD holds there
    (committed_problem)."""
    files = module_files(module, repo_root)
    if isinstance(files, str):
        return files
    root = Path(repo_root)
    for found in files:
        if not found.resolve().is_relative_to(root.resolve()):
            return f"모듈 {module!r} 이 저장소 밖의 파일을 가리킵니다 ({found.resolve()})"
    shadows = module_shadows(module, root)
    if shadows:
        return f"모듈 {module!r}: import 가 저장소의 .py 대신 집을 수 있는 파일이 있습니다 ({', '.join(shadows)})"
    for found in files:
        problem = committed_problem(found.relative_to(root).as_posix(), root)
        if problem:
            return f"모듈 {module!r} 을 불러오면 도는 파일: {problem}"
    return None


def rerun_flags(policy: Mapping | None, *, repo_root: Path) -> list[str]:
    """run_s1a.py's flags that rerun with ``policy`` (a ``sim_backend.policy`` record): [] for None;
    ``["--behavior", spec]`` for the published checkpoint; ``["--policy", str(repo_root / path), "--behavior",
    spec]`` for ``file:<path>`` when that path, its ``..`` and links resolved, is a file under ``repo_root`` whose
    SHA-256 is the recorded one and that is, by the path as named, the file HEAD holds there (committed_problem).
    ValueError naming the ``source`` and why for anything else -- the spec included, held to behavior.parse_spec's
    ``module:name`` and its module to the files HEAD holds (module_problem) whichever the checkpoint."""
    if policy is None:
        return []
    if not isinstance(policy, Mapping):
        raise ValueError(f"기록된 정책이 객체가 아닙니다: {policy!r}")
    source = policy.get("source")

    def refused(why: str) -> ValueError:
        return ValueError(f"기록된 정책 {source!r} 로 다시 돌릴 수 없습니다 — {why}")

    spec = policy.get("behavior")
    try:
        module, _ = behavior.parse_spec(spec)
    except ValueError as e:
        raise refused(f"행동 층: {e}") from e
    root = Path(repo_root)
    problem = module_problem(module, root)
    if problem:
        raise refused(f"행동 층 {spec!r}: {problem}")
    if source == PUBLISHED_SOURCE:
        return ["--behavior", spec]
    if not isinstance(source, str) or not source.startswith(FILE_SOURCE):
        raise refused(f"다시 구할 수 있는 정책은 {PUBLISHED_SOURCE!r} 와 저장소의 파일({FILE_SOURCE}<경로>)뿐입니다")
    relative = source[len(FILE_SOURCE):]
    if not relative or Path(relative).is_absolute():
        raise refused("저장소 안의 상대 경로가 아닙니다")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise refused(f"저장소 밖의 파일을 가리킵니다 ({target})")
    if not target.is_file():
        raise refused("저장소에 그 파일이 없습니다")
    recorded = policy.get("sha256")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != recorded:
        raise refused(f"저장소의 그 파일의 sha256 {digest} 이 기록된 {recorded!r} 와 다릅니다")
    problem = committed_problem(relative, root)
    if problem:
        raise refused(problem)
    return ["--policy", str(root / relative), "--behavior", spec]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="C9 재현 — 조건표 시나리오의 에피소드를 기록 때의 보행 정책 · 행동 층으로 다시 돌릴 "
                    "run_s1a.py 인자를 한 줄에 하나씩 (조건표가 없는 시나리오는 아무것도 읽지 않고 빈 출력)")
    ap.add_argument("--published", required=True, help="대조할 배포본의 루트 (meta/episodes 를 읽음)")
    ap.add_argument("--episode-id", required=True)
    args = ap.parse_args(argv)
    try:
        if episode_scenario_of(args.episode_id) not in _table_scenarios():
            return 0
        flags = rerun_flags(published_policy(Path(args.published), args.episode_id), repo_root=_REPO_ROOT)
        # One flag a line is what the hook splits on: a value holding a line break would come apart there.
        broken = [flag for flag in flags if "\n" in flag or "\r" in flag]
        if broken:
            raise ValueError(f"인자가 한 줄이 아닙니다: {broken[0]!r}")
    except ValueError as e:
        print(f"재현할 정책을 정하지 못했습니다: {e}", file=sys.stderr)
        return 2
    for flag in flags:
        print(flag)
    return 0


def _table_scenarios() -> tuple[str, ...]:
    from generation import conditions

    return conditions.table_scenarios()


if __name__ == "__main__":
    raise SystemExit(main())
