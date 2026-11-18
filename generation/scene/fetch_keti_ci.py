#!/usr/bin/env python3
"""Fetch the KETI CI the box labels show, and verify it against keti_ci.json.

    .venv/bin/python generation/scene/fetch_keti_ci.py                          # download the official archive
    .venv/bin/python generation/scene/fetch_keti_ci.py --archive <KETI_CI.zip>  # an archive downloaded before

keti_ci.json is the pin: the official download, the archive's SHA-256, and the SHA-256 of the
one PNG taken from it. The archive is opened only when it hashes as pinned, and the PNG is
picked by its hash, not by its name -- the member names are Korean with a '#', and how a zip
tool decodes them is not something to depend on. The PNG lands under the scene's fetched
textures (generation/scene/textures/, not tracked) byte for byte; the carried box's build and
the warehouse's (its box labels) copy it beside their USDs from there.

The CI is the distributor's own logo, shipped under its own terms (LicenseRef-KETI-CI): the
dataset's CC BY 4.0 grants no trademark rights (its section 2(b)(2)). The lock admits no
other licence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation.scene import fetch_textures  # noqa: E402

CI_LOCK = HERE / "keti_ci.json"
OUT = fetch_textures.OUT            # beside the scene's fetched originals (gitignored)
CI_LICENSE = "LicenseRef-KETI-CI"
REQUIRED = ("url", "archive_sha256", "member", "member_sha256", "file", "license")
OPTIONAL = ("page", "note")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def load_ci_lock(path: Path = CI_LOCK) -> dict:
    """The pin, refused when it cannot say which bytes are meant or under which terms."""
    try:
        lock = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: KETI CI 고정 파일을 읽지 못했습니다 — {exc}") from exc
    if not isinstance(lock, dict):
        raise ValueError(f"{path}: JSON 객체가 아닙니다")
    unknown = sorted(set(lock) - set(REQUIRED) - set(OPTIONAL))
    if unknown:
        raise ValueError(f"{path}: 모르는 키 {unknown} — 아는 키는 {list(REQUIRED + OPTIONAL)}")
    for key in REQUIRED:
        if not isinstance(lock.get(key), str) or not lock[key]:
            raise ValueError(f"{path}: {key} 가 비어 있지 않은 문자열이 아닙니다: {lock.get(key)!r}")
    for key in ("archive_sha256", "member_sha256"):
        if not _SHA256.fullmatch(lock[key]):
            raise ValueError(f"{path}: {key} 가 64자리 소문자 16진수가 아닙니다: {lock[key]!r}")
    name = lock["file"]
    if Path(name).name != name or name in (".", "..") or "\\" in name:
        raise ValueError(f"{path}: file 이 경로가 아닌 파일 이름이어야 합니다: {name!r}")
    if not lock["url"].startswith("https://"):
        raise ValueError(f"{path}: url 이 https:// 가 아닙니다: {lock['url']!r}")
    if lock["license"] != CI_LICENSE:
        raise ValueError(f"{path}: license 가 {CI_LICENSE} 가 아닙니다: {lock['license']!r} — "
                         "KETI CI 는 배포자 자신의 CI 이고 다른 라이선스를 주장하지 않습니다")
    return lock


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive_name(url: str) -> str:
    return Path(urlparse(url).path).name or "keti_ci.zip"


def fetch(lock: dict, archive: Path | None, out_dir: Path = OUT) -> Path:
    """Make ``out_dir/<file>`` the pinned PNG and return it.

    Already the pinned bytes: left as it is (no archive needed). Otherwise the archive --
    ``archive`` if given, else downloaded once under ``out_dir/.archives/`` -- must hash as
    pinned, and the one member whose SHA-256 is ``member_sha256`` is written out. Raises
    ValueError when the archive is another one or holds no such member; nothing is written
    then."""
    dest = Path(out_dir) / lock["file"]
    if dest.is_file() and _sha256(dest) == lock["member_sha256"]:
        return dest
    if archive is None:
        archive = Path(out_dir) / fetch_textures.ARCHIVES / _archive_name(lock["url"])
        if not (archive.is_file() and _sha256(archive) == lock["archive_sha256"]):
            fetch_textures._download(lock["url"], archive)
            if _sha256(archive) != lock["archive_sha256"]:
                got = _sha256(archive)
                archive.unlink(missing_ok=True)
                raise ValueError(f"내려받은 압축 파일의 SHA-256 이 고정값과 다릅니다(받은 것 {got[:12]}…, "
                                 f"고정값 {lock['archive_sha256'][:12]}…) — 누리집의 파일이 바뀌었으면 "
                                 "고정 파일과 사용 허락의 범위를 함께 다시 보십시오. 받은 파일은 지웠습니다")
    archive = Path(archive)
    got = _sha256(archive)
    if got != lock["archive_sha256"]:
        raise ValueError(f"{archive}: 압축 파일의 SHA-256 이 고정값과 다릅니다(파일 {got[:12]}…, "
                         f"고정값 {lock['archive_sha256'][:12]}…) — 고정한 압축 파일이 아니면 열지 않습니다")
    with zipfile.ZipFile(archive) as z:
        data = next((d for d in (z.read(i) for i in z.infolist() if not i.is_dir())
                     if hashlib.sha256(d).hexdigest() == lock["member_sha256"]), None)
    if data is None:
        raise ValueError(f"{archive}: 압축 파일 안에 SHA-256 이 {lock['member_sha256'][:12]}… 인 파일이 없습니다 "
                         f"(고정 파일이 적은 이름 {lock['member']!r})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lock", type=Path, default=CI_LOCK)
    ap.add_argument("--archive", type=Path, default=None,
                    help="the official archive downloaded before (otherwise it is downloaded)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    try:
        lock = load_ci_lock(a.lock)
        got = fetch(lock, a.archive, a.out)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"have  {got.name}  sha256 {lock['member_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
