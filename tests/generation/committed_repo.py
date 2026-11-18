"""A git checkout made for a test: files of this repository, as the working tree holds them, committed in a
repository of their own. A test that needs "tracked by git and the same as HEAD" uses one, so that it does not turn
on what happens to be uncommitted in the checkout that runs it."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env() -> dict:
    """The caller's environment without its GIT_* variables: a GIT_DIR or GIT_INDEX_FILE set around the test would
    have git act on another repository."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def git(repo, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "commit.gpgsign=false", *args], env=_env(), check=True, capture_output=True)


def commit_all(repo) -> None:
    """Everything under ``repo`` committed: tracked and the same as HEAD. Made a git checkout first when it is not
    one."""
    if not (Path(repo) / ".git").exists():
        git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", "files")


def committed_copy(dest, *paths) -> Path:
    """The files git tracks in this repository under ``paths`` -- copied as the working tree holds them, links as
    links -- committed in a new checkout at ``dest``; its resolved root."""
    dest = Path(dest)
    listed = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--", *paths], env=_env(),
                            check=True, capture_output=True).stdout
    for rel in filter(None, listed.split(b"\0")):
        source = REPO_ROOT / os.fsdecode(rel)
        if not (source.exists() or source.is_symlink()):
            continue                                    # removed in the working tree: not copied
        target = dest / os.fsdecode(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=False)
    commit_all(dest)
    return dest.resolve()
