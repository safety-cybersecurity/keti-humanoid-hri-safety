#!/usr/bin/env python3
"""Download the CC0 textures textures.lock names and verify their SHA256.

    .venv/bin/python generation/scene/fetch_textures.py            # fetch what is missing
    .venv/bin/python generation/scene/fetch_textures.py --check-only

The lock is the canon: url, licence and hash per file. A file that does not
hash as recorded is deleted, not used -- a texture we cannot attribute must
not reach a CC BY 4.0 release. Sets that ship as a zip (ambientCG) have one
``archive`` line for the zip and one ``zip:<member>`` line per file taken
from it; both are hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = HERE / "textures.lock"
OUT = HERE / "textures"
ARCHIVES = ".archives"

COLUMNS = ("name", "role", "map", "license", "source", "url", "sha256")
MAPS = ("diffuse", "normal_gl", "roughness", "archive")
KNOWN_SOURCES = ("polyhaven", "ambientcg")
REQUIRED_LICENSE = "CC0"


@dataclass(frozen=True)
class TextureEntry:
    name: str
    role: str
    map: str
    license: str
    source: str
    url: str
    sha256: str

    @property
    def in_archive(self) -> bool:
        return self.url.startswith("zip:")

    @property
    def filename(self) -> str:
        """Where the file lands under the texture dir (archives under .archives/)."""
        if self.in_archive:
            return self.url[len("zip:"):]
        return self.url.rsplit("/", 1)[-1].split("?file=", 1)[-1]


def parse_lock(path: Path = LOCK) -> list[TextureEntry]:
    out: list[TextureEntry] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) != len(COLUMNS):
            raise ValueError(f"{path}:{lineno}: expected {len(COLUMNS)} columns {COLUMNS}, got {len(parts)}: {line!r}")
        e = TextureEntry(*parts)
        if e.license != REQUIRED_LICENSE:
            raise ValueError(f"{path}:{lineno}: {e.name}: licence must be {REQUIRED_LICENSE}, got {e.license!r}")
        if e.map not in MAPS:
            raise ValueError(f"{path}:{lineno}: {e.name}: map must be one of {MAPS}, got {e.map!r}")
        if e.source not in KNOWN_SOURCES:
            raise ValueError(f"{path}:{lineno}: {e.name}: source must be one of {KNOWN_SOURCES}, got {e.source!r}")
        if len(e.sha256) != 64:
            raise ValueError(f"{path}:{lineno}: {e.name}: sha256 must be 64 hex chars")
        int(e.sha256, 16)
        if not (e.url.startswith("https://") or e.in_archive):
            raise ValueError(f"{path}:{lineno}: {e.name}: url must be https:// or zip:<member>")
        out.append(e)
    archives = {e.role for e in out if e.map == "archive"}
    for e in out:
        if e.in_archive and e.role not in archives:
            raise ValueError(f"{path}: {e.name} is taken from a zip but role {e.role!r} has no archive line")
    if not out:
        # A truncated lock would otherwise verify as "everything ok" with
        # nothing checked.
        raise ValueError(f"{path}: no entries — an empty lock cannot pin anything")
    return out


def files_for_role(entries: list[TextureEntry], role: str, out_dir: Path = OUT) -> dict[str, Path]:
    """map -> path on disk for one texture set (archive lines excluded). Empty if unknown."""
    return {e.map: out_dir / e.filename for e in entries if e.role == role and e.map != "archive"}


# What a material needs from its set: every map except the archive itself.
MATERIAL_MAPS = tuple(m for m in MAPS if m != "archive")


def files_for_material(entries: list[TextureEntry], role: str, out_dir: Path = OUT) -> dict[str, Path]:
    """Like files_for_role, but a set that is unknown or incomplete is an error.

    files_for_role() returning {} for a typo let the builder's verification
    loop run zero times and author a material with no textures; the builder
    uses this instead.
    """
    files = files_for_role(entries, role, out_dir)
    if not files:
        raise ValueError(f"no texture set {role!r} in the lock — check materials.<name>.texture in the spec")
    missing = [m for m in MATERIAL_MAPS if m not in files]
    if missing:
        raise ValueError(f"texture set {role!r} lacks {missing} in the lock")
    return files


def sha256_of(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def verify(entry: TextureEntry, file: Path) -> bool:
    return file.is_file() and sha256_of(file) == entry.sha256


# ambientCG answers urllib's default User-Agent with 403 (measured 2026-09-22);
# a plain browser-like one is accepted by both hosts.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) keti-humanoid-safety/fetch_textures"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"download failed: {url} — {exc}") from exc
    tmp.replace(dest)


def _archive_of(entries: list[TextureEntry], role: str) -> TextureEntry:
    for e in entries:
        if e.role == role and e.map == "archive":
            return e
    raise KeyError(role)


def fetch(entry: TextureEntry, entries: list[TextureEntry], out_dir: Path = OUT) -> Path:
    """Make entry's file exist on disk with the recorded hash, or stop."""
    dest = (out_dir / ARCHIVES / entry.filename) if entry.map == "archive" else out_dir / entry.filename
    if verify(entry, dest):
        return dest
    if entry.in_archive:
        archive = fetch(_archive_of(entries, entry.role), entries, out_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            dest.write_bytes(z.read(entry.filename))
    else:
        _download(entry.url, dest)
    if not verify(entry, dest):
        got = sha256_of(dest) if dest.is_file() else "<missing>"
        dest.unlink(missing_ok=True)
        raise SystemExit(f"{entry.name}: SHA256 mismatch after fetch (got {got[:12]}…, lock says "
                         f"{entry.sha256[:12]}…) — the lock or the upstream changed; not keeping the file")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lock", type=Path, default=LOCK)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check-only", action="store_true", help="report, download nothing")
    a = ap.parse_args(argv)
    try:
        entries = parse_lock(a.lock)
    except ValueError as exc:
        # a lock that cannot pin anything is a failed check, not a traceback
        raise SystemExit(f"{exc}") from exc
    bad = 0
    for e in entries:
        dest = (a.out / ARCHIVES / e.filename) if e.map == "archive" else a.out / e.filename
        if a.check_only:
            ok = verify(e, dest)
            bad += not ok
            print(("ok    " if ok else "MISS  ") + e.name + "  " + e.map)
        else:
            print("have  " + fetch(e, entries, a.out).relative_to(a.out).as_posix())
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
