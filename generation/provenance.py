"""Which code recorded an episode: this repository's commit, and whether the tree was clean.

sim_backend (schema §5) names the simulator, the framework, the physics engine and lerobot --
not this repository, whose recorder and environment decide every number as much as they do.
The same seed has produced another trajectory after a recorder change (2026-08-25), so a
reproduction claim without the commit is half a claim.

run_s1a.py reads the two values once at run start and adds them to the sim_backend it hands
the recorder. They are not kept in environment.lock: a committed file cannot hold the commit
it is committed in.

The generator runs on a dirty tree -- experiments and control arms do. The converter does not
take what such a run recorded (problems): a distribution is made from committed code only.
"""

from __future__ import annotations

import os
import re
import subprocess

COMMIT = "generator_commit"
DIRTY = "generator_dirty"
# A full SHA-1 object name in lower case, as `git rev-parse HEAD` prints it. Use fullmatch.
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
# git reads these before -C: inside a hook they point at the hook's repository.
_GIT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX")


def read(repo_root) -> dict:
    """{generator_commit, generator_dirty} of the working tree at ``repo_root``.

    Dirty means `git status --porcelain` prints anything: a modified or staged tracked file, or
    an untracked file that is not ignored -- a new module nobody added changes the run as much
    as an edit does. Both values are None when git cannot say (no git, not a repository, no
    commit yet); the converter refuses those as it refuses a dirty tree.
    """
    env = {k: v for k, v in os.environ.items() if k not in _GIT_ENV}

    def git(*args: str) -> str:
        done = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True,
                              text=True, timeout=60, env=env)
        if done.returncode != 0:
            raise OSError(done.stderr.strip() or f"git {args[0]}: exit {done.returncode}")
        return done.stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        # --untracked-files=normal: plain `git status` follows status.showUntrackedFiles, and with `no` a new
        # module reads as clean. The value ships in a public dataset; it is not for a user's git config to set.
        dirty = bool(git("status", "--porcelain", "--untracked-files=normal").strip())
    except (OSError, subprocess.SubprocessError):
        return {COMMIT: None, DIRTY: None}
    if not COMMIT_RE.fullmatch(commit):
        return {COMMIT: None, DIRTY: None}
    return {COMMIT: commit, DIRTY: dirty}


def problems(sim_backend) -> list[str]:
    """Why ``sim_backend`` does not say which committed code made the recording; [] when it does.

    Types, not truthiness: bool("false") is True, and a string in generator_dirty would pass a
    truth test either way round.
    """
    if not isinstance(sim_backend, dict):
        return ["sim_backend 가 객체가 아닙니다"]
    out = []
    commit = sim_backend.get(COMMIT)
    if not (isinstance(commit, str) and COMMIT_RE.fullmatch(commit)):
        out.append(f"sim_backend.{COMMIT} 이 40자 16진 커밋 해시가 아닙니다: {commit!r}")
    dirty = sim_backend.get(DIRTY)
    if dirty is not False:
        out.append(f"sim_backend.{DIRTY} 가 false 가 아닙니다: {dirty!r}"
                   + (" — 커밋되지 않은 변경이 있는 트리에서 생성했습니다" if dirty is True else ""))
    return out
