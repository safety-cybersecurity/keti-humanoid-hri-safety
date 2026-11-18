#!/usr/bin/env python3
"""Fetch a robot that ships as its own folder, and verify it against meta/assets.json.

    .venv/bin/python generation/robots/fetch_robot.py --robot unitree_h1 --nucleus <Isaac Lab asset root>

meta/assets.json robots.<robot>.ship is the pin: one row per file of the folder -- where the bytes come from
(url, under {ISAACLAB_NUCLEUS_DIR}, the Isaac Lab asset root the simulation resolves) and their SHA-256.
--nucleus is that root: the address Isaac Lab resolves (an https URL), or a directory holding a copy of it
(the Isaac Sim asset pack unpacked, .../Assets/Isaac/6.0/Isaac/IsaacLab). The files land in the entry's
ship.source_dir (generation/robots/build/<dir>/, not tracked) byte for byte, and only when every one of them
hashes as its row says: a file that is not the pinned bytes lands nothing. A file already there with the
pinned bytes is left as it is. The files are fetched into a temporary folder beside that one and moved into
place once all of them hold: no name of the folder is used for anything but its own file.

The rows are what generation/assets.robot_ship_files reads; the block is held to that rule before anything
is fetched. What ships from here is the robot's own licence's -- BSD-3 for the H1 under
IsaacLab/Robots/Unitree, the licence file one of the rows.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation.assets import (NUCLEUS_TOKEN, load_assets, robot_ship_files,  # noqa: E402
                               sha256_file)
from generation.scene import fetch_textures  # noqa: E402

# Where fetched robots lie: one folder per robot, ignored by git (.gitignore).
BUILD = "generation/robots/build"


def _source(url: str, nucleus: str) -> str:
    """The row's url with the nucleus token resolved to ``nucleus``."""
    if not url.startswith(NUCLEUS_TOKEN + "/"):
        raise ValueError(f"{url}: url 이 {NUCLEUS_TOKEN} 아래가 아닙니다 — --nucleus 로 풀 수 없는 행입니다")
    return nucleus.rstrip("/") + url[len(NUCLEUS_TOKEN):]


def fetch(entry: dict, nucleus: str, repo_root: Path, name: str = "robot") -> list[Path]:
    """Make the folder of ``entry`` (a meta/assets.json robots entry with a ship block) hold the pinned files,
    fetched from ``nucleus`` -- an address or a directory -- and return their paths, in the rows' order.

    Every file is fetched into a temporary folder beside the robot's folder (on the same filesystem), under a
    name of its own there -- its row's number, so no file of the folder can share it -- and hashed before any of
    them is put in place: a source that is not the pinned bytes, or is missing, raises (ValueError,
    FileNotFoundError) and lands nothing. The temporary folder is removed either way. ``name`` is the robot's key,
    for the messages."""
    repo_root = Path(repo_root)
    files = robot_ship_files(name, entry, repo_root)
    rows = entry["ship"]["files"]
    staging: Path | None = None
    staged: list[tuple[Path, Path]] = []
    try:
        for k, (row, shipped) in enumerate(zip(rows, files.values())):
            dest = shipped.src
            if dest.is_file() and sha256_file(dest) == shipped.sha256:
                continue
            where = _source(row["url"], nucleus)
            if "://" not in where:
                src = Path(where).expanduser()
                if not src.is_file():
                    raise FileNotFoundError(f"{src}: 받아 둔 사본에 {row['file']} 이 없습니다 — --nucleus 가 "
                                            f"Isaac Lab 자산의 뿌리({NUCLEUS_TOKEN})를 가리키는지 보십시오")
            if staging is None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix=f".{dest.parent.name}.fetch-", dir=dest.parent.parent))
            tmp = staging / str(k)
            if "://" in where:
                fetch_textures._download(where, tmp)
            else:
                shutil.copyfile(src, tmp)
            got = sha256_file(tmp)
            if got != shipped.sha256:
                raise ValueError(f"{row['file']} 의 SHA-256 이 정본과 다릅니다(받은 것 {got[:12]}…, 정본 "
                                 f"{shipped.sha256[:12]}…) — {where} 는 meta/assets.json 이 고정한 파일이 "
                                 "아닙니다. 아무것도 놓지 않았습니다")
            staged.append((tmp, dest))
        for tmp, dest in staged:
            tmp.replace(dest)
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    return [shipped.src for shipped in files.values()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robot", required=True, help="the meta/assets.json robots key (unitree_h1)")
    ap.add_argument("--nucleus", required=True,
                    help="the Isaac Lab asset root: an https address, or a directory holding a copy of it")
    ap.add_argument("--repo", type=Path, default=REPO, help="the repository the canon and the folder are in")
    a = ap.parse_args(argv)
    robots = load_assets(a.repo).get("robots") or {}
    entry = robots.get(a.robot)
    if not isinstance(entry, dict):
        raise SystemExit(f"meta/assets.json 의 robots 에 {a.robot!r} 가 없습니다 — 아는 로봇: {sorted(robots)}")
    try:
        got = fetch(entry, a.nucleus, a.repo, name=a.robot)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    for path, row in zip(got, entry["ship"]["files"]):
        print(f"have  {a.robot}/{path.name}  sha256 {row['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
