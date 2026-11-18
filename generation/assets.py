"""meta/assets.json -- loading, path normalisation, the licence join, and what ships.

Isaac-free on purpose: run_s1a.py uses it to check its paths against the
canon *before* spending a batch, and convert_lerobot.py uses the same join
to build the schema's scene_assets. One implementation, tested once.

shipped_asset_files() is the one answer to "which files does this batch's
assets/ hold" -- the converter copies what it says, the publish pre-flight
compares a distribution against it, and validator C6 (which cannot import
this) checks the same rule from the shipped meta/assets.json.

A robot that ships ships as its own folder, assets/<dir>/ (robot_ship_files): its USD, the layers that
USD references beside it and its licence file, byte for byte, each a row of the canon entry's ``ship``.
Its usd_path stays the nucleus token the simulation resolves -- the key every episode's asset paths join
on -- so its files are named by the rows, not by that path.

An entry that composes others (``composes`` -- the Stage, one USD that references the warehouse, the dummy,
the box and the robot by their shipped names) is used by no episode: it ships with a release exactly when
everything it composes ships there (composed_shipped), at assets/<file> beside the scene USDs, so its
references resolve inside the release and nowhere else.

required_scene_assets() is what anchors that answer: the set is drawn from what
the episodes list, so an episode that leaves an asset out would take it out of
the distribution with nothing to say so. The canon says what every episode of a
scenario used -- the scene's declared assets and its human model's USD -- and the
episode's robot_model names its robot's USD.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

NUCLEUS_TOKEN = "{ISAACLAB_NUCLEUS_DIR}"

# Where a distribution keeps its assets, and where a USD's textures go beneath that. Both
# builders write a USD's textures to a textures/ directory beside it and reference them as
# ./textures/<file> (build_logistics_scene.TEXTURE_SUBDIR, build_dummy.TEXTURE_SUBDIR), so the
# distribution keeps the same layout and the references resolve there too.
ASSETS_DIR = "assets"
TEXTURE_SUBDIR = "textures"

# One row of a bundle's meta/assets.json `files`: the provenance a licence reviewer holding
# only the distribution needs for one file under assets/textures/. `url` is where the bytes
# come from -- the file itself, or the archive it is a member of (then `member` names it).
TEXTURE_ROW_KEYS = ("file", "sha256", "license", "url")
TEXTURE_ROW_OPTIONAL = ("member",)
_SHA256 = re.compile(r"[0-9a-f]{64}")

# A meta/assets.json entry's `ship`: the folder a third-party asset ships as, whole, under its own licence.
# `dir` is the folder under assets/ (its own, so two robots' files of one name never meet), `source_dir`
# where the fetched copy lies in the repository tree (generation/robots/fetch_robot.py), `license_file`
# the row that is the licence text, `files` one row per file -- the shape of a bundle.files row, `url`
# where the bytes come from.
ROBOT_SHIP_KEYS = ("dir", "source_dir", "license_file", "files")
# A meta/assets.json entry's ``composes``: the keys (of robots or scene) of the assets it references.
COMPOSES_KEY = "composes"


def load_assets(repo_root: str | Path) -> dict:
    return json.loads(
        (Path(repo_root) / "meta" / "assets.json").read_text(encoding="utf-8")
    )


def canonical_usd_path(usd_path: str, nucleus_dir: str, repo_root: str | Path) -> str:
    """The form meta/assets.json indexes by, from whatever the sim resolved.

    Isaac Lab expands ISAACLAB_NUCLEUS_DIR into a versioned S3 URL at run
    time, and a repo asset arrives as an absolute path that carries the
    machine's home directory. Neither is stable enough to be a join key, and
    the absolute one would put a username into episode metadata.

    A path that matches neither prefix is returned unchanged rather than
    guessed at -- asset_entry() is where an unknown asset stops the run.
    """
    path = str(usd_path)
    if nucleus_dir and path.startswith(nucleus_dir):
        return NUCLEUS_TOKEN + path[len(nucleus_dir):]
    root = str(Path(repo_root))
    if path.startswith(root + "/"):
        return path[len(root) + 1:]
    return path


def asset_entry(assets: dict, usd_path: str) -> dict:
    """The schema §5 scene_assets record for one loaded asset.

    Raises on an asset the canon does not know. Publishing a dataset that
    names an asset whose licence nobody checked is worse than stopping.
    """
    for group in ("robots", "scene"):
        for key, entry in (assets.get(group) or {}).items():
            if entry.get("usd_path") != usd_path:
                continue
            licence = entry.get("license")
            if not isinstance(licence, str) or not licence.strip():
                raise ValueError(
                    f"{key}: license 가 비어 있거나 문자열이 아닙니다: {licence!r}"
                )
            redistributed = entry.get("redistributed")
            # Not bool(): the string "false" is truthy, and a licence field
            # that quietly flips to "we redistribute this" is exactly the
            # claim scene_assets exists to keep honest.
            if not isinstance(redistributed, bool):
                raise ValueError(
                    f"{key}: redistributed 가 bool 이 아닙니다: {redistributed!r}"
                )
            return {"path": entry["usd_path"], "license": licence,
                    "redistributed": redistributed, "key": key}
    raise ValueError(
        f"meta/assets.json 에 없는 자산: {usd_path} — 라이선스를 모르는 자산은 "
        "배포본에 넣지 않습니다. 정본을 먼저 채우십시오"
    )


# ---- what a distribution ships ------------------------------------------------------------

def _scene_lock_rows(repo_root: Path, entry: dict, lock: str) -> list[dict]:
    """generation/scene/textures.lock: every non-archive line is a texture the scene USD
    references (build_logistics_scene copies each used material's maps beside the USD). An
    ambientCG file is a member of the zip its set's `archive` line downloads."""
    from generation.scene import fetch_textures

    entries = fetch_textures.parse_lock(repo_root / lock)
    archive_url = {e.role: e.url for e in entries if e.map == "archive"}
    rows = []
    for e in entries:
        if e.map == "archive":
            continue                                # the zip is where members come from, not a texture
        # parse_lock admits "CC0" only (fetch_textures.REQUIRED_LICENSE); the SPDX name the
        # rest of meta/assets.json uses is CC0-1.0.
        if e.license != "CC0":
            raise ValueError(f"{e.name}: 라이선스가 CC0 가 아닙니다: {e.license!r}")
        row = {"file": e.filename, "sha256": e.sha256, "license": "CC0-1.0",
               "url": archive_url[e.role] if e.in_archive else e.url}
        if e.in_archive:
            row["member"] = e.filename
        rows.append(row)
    return rows


def _skin_lock_rows(repo_root: Path, entry: dict, lock: str) -> list[dict]:
    """generation/dummy/skin/textures.lock: its texture lines (make_skin.is_texture_row) are the
    textures the skinned dummy references; its pack-file lines made the figure and do not ship.
    Every file is a member of the MakeHuman system asset pack the spec pins."""
    import yaml

    from generation.dummy.skin import make_skin

    spec_rel = entry.get("spec")
    if not spec_rel:
        raise ValueError("스킨 텍스처 lock 의 출처(팩 URL)를 읽을 명세(spec)가 정본 항목에 없습니다")
    spec = yaml.safe_load((repo_root / spec_rel).read_text(encoding="utf-8"))
    pack = ((spec or {}).get("source") or {}).get("system_assets") or {}
    if not pack.get("url"):
        raise ValueError(f"{spec_rel}: source.system_assets.url 이 없습니다 — 텍스처의 출처를 적을 수 없습니다")
    rows = []
    for r in make_skin.parse_lock(repo_root / lock):
        if not make_skin.is_texture_row(r):
            continue
        if not r["source"].startswith("pack:"):
            raise ValueError(f"{r['name']}: 출처가 팩 안의 경로가 아닙니다: {r['source']!r}")
        rows.append({"file": r["name"], "sha256": r["sha256"], "license": r["license"],
                     "url": pack["url"], "member": r["source"][len("pack:"):]})
    return rows


def _ci_lock_rows(repo_root: Path, entry: dict, lock: str) -> list[dict]:
    """generation/scene/keti_ci.json: one row, the KETI CI the box labels show (the carried box's
    and the warehouse boxes') -- a member of the official archive, picked by its SHA-256
    (fetch_keti_ci.py), under the CI's own terms (LicenseRef-KETI-CI; load_ci_lock refuses any
    other licence)."""
    from generation.scene import fetch_keti_ci

    pin = fetch_keti_ci.load_ci_lock(repo_root / lock)
    return [{"file": pin["file"], "sha256": pin["member_sha256"], "license": pin["license"],
             "url": pin["url"], "member": pin["member"]}]


# Each lock has its own format and its own parser (the builders and fetchers read them), so
# each has its own reader here, chosen by the lock path a bundle names. A new bundle with a
# lock this table does not know stops instead of shipping textures nobody pinned.
LOCK_READERS = {
    "generation/scene/textures.lock": _scene_lock_rows,
    "generation/dummy/skin/textures.lock": _skin_lock_rows,
    "generation/scene/keti_ci.json": _ci_lock_rows,
}

# The bundle keys that say one thing per lock: the lock's path, its SHA-256 and the licence of its rows.
PER_LOCK_KEYS = ("textures_lock", "textures_lock_sha256", "textures_license")


def bundle_per_lock(bundle: dict, key: str) -> list:
    """One of PER_LOCK_KEYS of a bundle as a list, one item per lock, in the order of ``textures_lock``.

    A USD whose textures come from one pin names it as one string (the dummies' skin, the carried
    box's CI); a USD whose textures come from several names them as a list (the warehouse: its CC0
    textures and the KETI CI on its box labels), and every per-lock key is then a list of the same
    length. Raises ValueError when the lock list is empty, holds something that is not a non-empty
    string or names one lock twice, or when a per-lock key does not follow its shape."""
    locks = bundle.get("textures_lock")
    one = isinstance(locks, str)
    locks = [locks] if one else locks
    if (not isinstance(locks, list) or not locks
            or not all(isinstance(x, str) and x for x in locks) or len(set(locks)) != len(locks)):
        raise ValueError(f"텍스처 묶음의 textures_lock 이 lock 경로 하나 또는 겹치지 않는 경로의 목록이 아닙니다: "
                         f"{bundle.get('textures_lock')!r}")
    if key == "textures_lock":
        return list(locks)
    if key not in PER_LOCK_KEYS:
        raise ValueError(f"{key} 는 lock 마다 하나인 키가 아닙니다 — 그런 키는 {list(PER_LOCK_KEYS)}")
    value = bundle.get(key)
    if one:
        return [value]
    if not isinstance(value, list) or len(value) != len(locks):
        raise ValueError(f"텍스처 묶음이 lock 을 목록으로 적었으므로 {key} 도 같은 길이({len(locks)})의 "
                         f"목록이어야 합니다: {value!r}")
    return list(value)


def texture_rows_from_lock(entry: dict, repo_root: str | Path) -> list[dict]:
    """A bundle's textures as its locks pin them -- what the entry's ``bundle.files`` in
    meta/assets.json must say, row for row: each lock's rows, the locks in the order the bundle
    names them (bundle_per_lock).

    The lock is the canon the builders check against, but a distribution carries no lock:
    meta/assets.json carries these rows so the provenance travels with the files."""
    bundle = entry.get("bundle")
    if not isinstance(bundle, dict):
        raise ValueError("텍스처 묶음(bundle)이 없는 자산입니다")
    rows: list[dict] = []
    for lock in bundle_per_lock(bundle, "textures_lock"):
        reader = LOCK_READERS.get(lock)
        if reader is None:
            raise ValueError(f"텍스처 lock {lock!r} 을 읽는 방법을 모릅니다 — 아는 것: {sorted(LOCK_READERS)}")
        rows += reader(Path(repo_root), entry, lock)
    return rows


def required_scene_assets(scenario: dict, assets: dict, robot_model: object) -> set[str]:
    """The asset paths an episode of ``scenario`` run by ``robot_model`` must carry in
    scene_assets: the scene's declared ``scene.assets``, the USD of its ``scene.human_model``
    (a key of meta/assets.json ``scene``; null for a scenario without one) and the USD of its
    robot -- ``robot_model``, the episode's own field and a key of meta/assets.json ``robots``
    (no scenario names its robot; run_s1a reads the key off the joint map).

    What ships is drawn from the episodes' scene_assets (shipped_asset_files), so this is the
    completeness check on that list -- measured: a v0.1 copy with the
    dummy dropped from every scene_assets and its file deleted passed C6 and the pre-flight,
    and the S1d aisle reaches asset_paths only through getattr(scene cfg, "aisle", None). The
    robot is in it for the same reason: H1 dropped from every scene_assets passed C6
    and this pre-flight on two converted batches whose robot does not ship. For a robot that
    does not ship, what its absence costs is the licence record -- the list says which assets,
    under which licence, an episode
    loaded; a robot that ships as its folder would lose the folder too. Raises
    ValueError when the scenario or the robot cannot say: no scene block, scene.assets not a list
    of non-empty strings, a human_model or robot_model the canon does not know. Validator C6 restates this (_required_scene_assets);
    tests/validator/test_asset_checks.py holds the two together."""
    sid = scenario.get("scenario_id", "?") if isinstance(scenario, dict) else "?"
    scene = scenario.get("scene") if isinstance(scenario, dict) else None
    if not isinstance(scene, dict):
        raise ValueError(f"{sid}: scene 이 없습니다 — 이 시나리오의 에피소드가 쓴 자산을 알 수 없습니다")
    declared = scene.get("assets")
    if not isinstance(declared, list) or not all(isinstance(p, str) and p for p in declared):
        raise ValueError(f"{sid}: scene.assets 가 비어 있지 않은 문자열의 목록이 아닙니다: {declared!r}")
    out = set(declared)
    model = scene.get("human_model")
    if model is not None:
        entry = (assets.get("scene") or {}).get(model) if isinstance(model, str) else None
        if not isinstance(entry, dict) or not isinstance(entry.get("usd_path"), str) \
                or not entry["usd_path"]:
            raise ValueError(f"{sid}: scene.human_model {model!r} 이 meta/assets.json 의 scene 에 "
                             "없습니다 — 인체 모델의 자산을 알 수 없습니다")
        out.add(entry["usd_path"])
    robot = (assets.get("robots") or {}).get(robot_model) if isinstance(robot_model, str) else None
    if not isinstance(robot, dict) or not isinstance(robot.get("usd_path"), str) \
            or not robot["usd_path"]:
        raise ValueError(f"{sid}: robot_model {robot_model!r} 이 meta/assets.json 의 robots 에 "
                         "없습니다 — 로봇의 자산을 알 수 없습니다")
    out.add(robot["usd_path"])
    return out


def texture_row_problems(row: object) -> list[str]:
    """What is wrong with one ``bundle.files`` row as a statement of provenance."""
    if not isinstance(row, dict):
        return [f"항목이 객체가 아님: {type(row).__name__}"]
    out = [f"{k} 가 비어 있지 않은 문자열이 아님: {row.get(k)!r}"
           for k in TEXTURE_ROW_KEYS if not isinstance(row.get(k), str) or not row.get(k)]
    unknown = sorted(set(row) - set(TEXTURE_ROW_KEYS) - set(TEXTURE_ROW_OPTIONAL))
    if unknown:
        out.append(f"모르는 키: {unknown}")
    name = row.get("file")
    if isinstance(name, str) and name and (Path(name).name != name or name in (".", "..")
                                           or "\\" in name):
        out.append(f"file 이 경로가 아닌 파일 이름이어야 함: {name!r}")
    sha = row.get("sha256")
    if isinstance(sha, str) and not _SHA256.fullmatch(sha):
        out.append(f"sha256 이 64자리 소문자 16진수가 아님: {sha!r}")
    return out


@dataclass(frozen=True)
class ShippedFile:
    src: Path          # where the bytes are in the repository tree
    sha256: str        # what the canon pins them to
    asset: str         # the meta/assets.json key the file ships for


def _repo_relative(repo_root: Path, rel: object, what: str) -> Path:
    rel = str(rel or "")
    if not rel or "://" in rel or rel.startswith("{") or Path(rel).is_absolute():
        raise ValueError(f"{what} 가 저장소 안의 상대 경로가 아닙니다: {rel!r}")
    src = (repo_root / rel).resolve()
    if not src.is_relative_to(repo_root.resolve()):
        raise ValueError(f"{what} 가 저장소 밖을 가리킵니다: {rel}")
    return repo_root / rel


def robot_ship_files(name: str, entry: dict, repo_root: str | Path) -> dict[str, ShippedFile]:
    """What a robot that ships as its own folder puts under assets/: published path -> ShippedFile, every
    row of the entry's ``ship`` at assets/<dir>/<file>, from <source_dir>/<file>, pinned to the row's sha256.

    The robot's USD is a row like the others, and its row is the robot's pin: the row named
    ``Path(usd_path).name`` must hash as ``usd_sha256`` says -- the copy the episodes ran and the joint map
    was read from. The licence file is a row too: a folder of a licensed third-party asset ships with its
    licence text. Raises ValueError when the block cannot say what ships: not an object, a key missing or
    unknown, a folder name that is not one name under assets/ (or is the textures folder), no rows, a row
    that is not a provenance record, two rows of one file name, the USD's row missing or not its pin, no
    row for the licence file, a source outside the repository, a texture bundle beside it (the folder's
    files are all rows of ``ship``). Reads no file -- check_shipped_sources() does that."""
    ship = entry.get("ship")
    if not isinstance(ship, dict):
        raise ValueError(f"{name}: ship 이 객체가 아닙니다: {ship!r}")
    missing = [k for k in ROBOT_SHIP_KEYS if k not in ship]
    if missing:
        raise ValueError(f"{name}: ship 에 {missing} 키가 없습니다 — 폴더 이름 · 받아 둔 자리 · 라이선스 파일 · "
                         "파일 목록을 모두 적습니다")
    unknown = sorted(set(ship) - set(ROBOT_SHIP_KEYS))
    if unknown:
        raise ValueError(f"{name}: ship 에 모르는 키 {unknown} — 아는 키는 {list(ROBOT_SHIP_KEYS)}")
    if "bundle" in entry:
        raise ValueError(f"{name}: ship 과 bundle 을 함께 적었습니다 — 폴더째 싣는 자산의 파일은 모두 ship.files 의 "
                         "행입니다")
    folder = ship["dir"]
    if not isinstance(folder, str) or not folder or Path(folder).name != folder \
            or folder in (".", "..", TEXTURE_SUBDIR) or "\\" in folder:
        raise ValueError(f"{name}: ship.dir 이 assets/ 아래의 폴더 이름이 아닙니다: {folder!r} "
                         f"(경로가 아닌 이름 하나 · {TEXTURE_SUBDIR!r} 는 텍스처 자리)")
    rows = ship["files"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name}: ship.files 가 비어 있지 않은 목록이 아닙니다: {rows!r}")
    for i, row in enumerate(rows):
        bad = texture_row_problems(row)
        if bad:
            raise ValueError(f"{name}: ship.files[{i}] — {'; '.join(bad)}")
    by_file: dict[str, dict] = {}
    for row in rows:
        if row["file"] in by_file:
            raise ValueError(f"{name}: ship.files 에 같은 이름이 둘 있습니다: {row['file']} — 한 폴더에 한 이름은 "
                             "한 파일입니다")
        by_file[row["file"]] = row
    pinned = str(entry.get("usd_sha256") or "")
    if not _SHA256.fullmatch(pinned):
        raise ValueError(f"{name}: 재배포하는데 정본에 usd_sha256 이 없거나 형식이 아닙니다 — "
                         "어느 파일을 실었는지 고정되지 않습니다")
    usd_name = Path(str(entry.get("usd_path") or "")).name
    if usd_name not in by_file:
        raise ValueError(f"{name}: ship.files 에 로봇의 USD {usd_name} 의 행이 없습니다 — 폴더는 에피소드가 "
                         "쓴 USD 와 함께 실립니다")
    if by_file[usd_name]["sha256"] != pinned:
        raise ValueError(f"{name}: ship.files 의 {usd_name} 행의 sha256 이 usd_sha256 과 다릅니다 — 행 "
                         f"{by_file[usd_name]['sha256'][:12]}…, usd_sha256 {pinned[:12]}… — 에피소드가 쓰고 "
                         "관절 매핑을 뽑은 로봇은 usd_sha256 이 고정한 파일입니다")
    licence = ship["license_file"]
    if not isinstance(licence, str) or licence not in by_file:
        raise ValueError(f"{name}: license_file {licence!r} 의 행이 ship.files 에 없습니다 — 폴더째 싣는 제3자 "
                         "자산은 라이선스 전문과 함께 실립니다")
    if not isinstance(ship["source_dir"], str):
        raise ValueError(f"{name}.ship.source_dir 가 저장소 안의 상대 경로가 아닙니다: {ship['source_dir']!r}")
    source = _repo_relative(Path(repo_root), ship["source_dir"], f"{name}.ship.source_dir")
    return {f"{ASSETS_DIR}/{folder}/{row['file']}": ShippedFile(source / row["file"], row["sha256"], name)
            for row in rows}


def _entries_by_key(assets: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for group in ("robots", "scene"):
        for name, entry in (assets.get(group) or {}).items():
            out.setdefault(name, []).append(entry)
    return out


def composed_shipped(assets: dict, shipped_keys: set[str]) -> set[str]:
    """The keys of the entries that compose others (``composes``) and ship with a release whose shipped
    assets are ``shipped_keys`` -- the scene and robot keys whose files the release carries: a composed entry
    ships when it is redistributable and every key it composes is one of them. One that composes an asset the
    release does not carry would reference a file the release lacks, so it stays out.

    Raises ValueError when a ``composes`` cannot say what it composes: not a non-empty list of distinct
    non-empty strings, a key the canon does not have (or has in both groups), the entry itself or another
    composed entry (one level: a composed entry is never one of ``shipped_keys``, so it would never ship),
    or a composed entry that also carries a texture bundle or a robot folder (it ships as its one USD) -- or
    when a composed entry names no file (its usd_path not a non-empty string). Every composed entry is read,
    shipped or not. Validator C6 restates this (_composed_shipped); tests/validator/test_asset_checks.py
    holds the two together."""
    by_key = _entries_by_key(assets)
    composed = {name for name, entries in by_key.items()
                if any(isinstance(e, dict) and COMPOSES_KEY in e for e in entries)}
    out: set[str] = set()
    for name in sorted(composed):
        entries = by_key[name]
        if len(entries) != 1:
            raise ValueError(f"{name}: 엮은 자산의 키가 robots 와 scene 에 모두 있습니다 — 어느 항목인지 알 수 없습니다")
        entry = entries[0]
        path = entry.get("usd_path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{name}: usd_path 가 비어 있지 않은 문자열이 아닙니다: {path!r} — 엮은 자산이 어느 파일로 "
                             "실리는지 알 수 없습니다")
        keys = entry[COMPOSES_KEY]
        if not isinstance(keys, list) or not keys or not all(isinstance(k, str) and k for k in keys):
            raise ValueError(f"{name}: composes 가 비어 있지 않은 키 문자열의 목록이 아닙니다: {keys!r}")
        if len(set(keys)) != len(keys):
            raise ValueError(f"{name}: composes 에 같은 키가 두 번 있습니다: {keys}")
        for key in keys:
            if key == name:
                raise ValueError(f"{name}: composes 가 자기 자신을 적었습니다")
            if key not in by_key:
                raise ValueError(f"{name}: composes 의 {key} 가 meta/assets.json 의 robots · scene 에 없습니다")
            if len(by_key[key]) != 1:
                raise ValueError(f"{name}: composes 의 {key} 가 robots 와 scene 에 모두 있습니다")
            if key in composed:
                raise ValueError(f"{name}: composes 의 {key} 도 엮은 자산입니다 — 엮은 자산은 한 단계만 엮습니다")
        for block in ("bundle", "ship"):
            if block in entry:
                raise ValueError(f"{name}: 엮은 자산에 {block} 가 있습니다 — 엮은 자산은 USD 하나로 실리고, 텍스처와 "
                                 "폴더는 그것이 엮는 자산의 것입니다")
        if entry.get("redistributed") is True and set(keys) <= set(shipped_keys):
            out.add(name)
    return out


def shipped_asset_files(assets: dict, referenced: Iterable[str],
                        repo_root: str | Path) -> dict[str, ShippedFile]:
    """What a distribution's assets/ holds: published path -> ShippedFile.

    An asset ships when meta/assets.json lets it (``redistributed: true``) **and** this
    batch's episodes reference it (``referenced``: the paths of their scene_assets). The canon
    lists every asset the project has made and a batch uses some -- shipping all of them put
    a file the episodes never touch into every distribution, and would have put the S1d scene
    into a v0.1 re-conversion. A USD with a texture bundle brings its textures to
    assets/textures/, where its ./textures/<file> references resolve. An entry with a ``ship`` block
    ships as its own folder instead (robot_ship_files) -- its usd_path is the nucleus token the
    episodes name, not a file of the repository. An entry that composes others ships at
    assets/<file> when everything it composes ships (composed_shipped) -- whether the episodes
    reference it or not; that is the only way it ships.

    Raises ValueError when the canon cannot say what to ship or what the bytes are: a path
    that is not the repository's, no pinned usd_sha256, two assets with one file name, a
    bundle whose ``files`` are not what its locks pin, one texture name pinned to two hashes, a
    ``composes`` that cannot say what it composes. Reads the locks but not the files --
    check_shipped_sources() does that."""
    root = Path(repo_root)
    wanted = set(referenced)
    out: dict[str, ShippedFile] = {}
    for group in ("robots", "scene"):
        for name, entry in (assets.get(group) or {}).items():
            if not isinstance(entry, dict) or entry.get("usd_path") not in wanted:
                continue
            if entry.get("redistributed") is not True:
                continue
            if COMPOSES_KEY in entry:
                continue                            # it ships with what it composes, below
            if "ship" in entry:
                for rel, shipped in robot_ship_files(name, entry, root).items():
                    if rel in out:
                        raise ValueError(f"두 자산이 같은 이름으로 실립니다: {rel} ({out[rel].asset} · {name})")
                    out[rel] = shipped
                continue
            usd = _repo_relative(root, entry["usd_path"], f"{name}.usd_path")
            pinned = str(entry.get("usd_sha256") or "")
            if not _SHA256.fullmatch(pinned):
                raise ValueError(f"{name}: 재배포하는데 정본에 usd_sha256 이 없거나 형식이 아닙니다 — "
                                 "어느 파일을 실었는지 고정되지 않습니다")
            rel = f"{ASSETS_DIR}/{usd.name}"
            if rel in out:
                raise ValueError(f"두 자산이 같은 이름으로 실립니다: {rel} ({out[rel].asset} · {name})")
            out[rel] = ShippedFile(usd, pinned, name)
            # A bundle key that is present says the USD has textures, whatever its value: null
            # ships the USD without them unless it is refused, and C6 refuses it.
            if "bundle" not in entry:
                continue
            bundle = entry["bundle"]
            rows = bundle.get("files") if isinstance(bundle, dict) else None
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"{name}: 텍스처 묶음(bundle)에 파일 목록(files)이 없습니다 — "
                                 "싣는 텍스처의 출처를 적을 곳이 없습니다")
            for i, row in enumerate(rows):
                bad = texture_row_problems(row)
                if bad:
                    raise ValueError(f"{name}: bundle.files[{i}] — {'; '.join(bad)}")
            if rows != texture_rows_from_lock(entry, root):
                raise ValueError(f"{name}: meta/assets.json 의 bundle.files 가 "
                                 f"{' · '.join(bundle_per_lock(bundle, 'textures_lock'))} 과 다릅니다 — "
                                 "lock 이 바뀌었으면 정본의 행도 다시 쓰십시오 "
                                 "(tests/generation/test_assets_canon.py 가 기대 행을 보입니다)")
            tex_dir = _repo_relative(root, bundle.get("textures_dir"), f"{name}.bundle.textures_dir")
            for row in rows:
                rel = f"{ASSETS_DIR}/{TEXTURE_SUBDIR}/{row['file']}"
                seen = out.get(rel)
                if seen is not None:
                    if seen.sha256 != row["sha256"]:
                        raise ValueError(
                            f"두 자산의 텍스처 이름이 같은데 바이트가 다릅니다: {row['file']} — "
                            f"{seen.asset} {seen.sha256[:12]}… · {name} {row['sha256'][:12]}… "
                            "(한 이름에는 한 파일만 실립니다)")
                    continue                        # the same bytes: one copy serves both USDs
                out[rel] = ShippedFile(tex_dir / row["file"], row["sha256"], name)
    by_key = _entries_by_key(assets)
    for name in sorted(composed_shipped(assets, {f.asset for f in out.values()})):
        entry = by_key[name][0]
        usd = _repo_relative(root, entry.get("usd_path"), f"{name}.usd_path")
        pinned = str(entry.get("usd_sha256") or "")
        if not _SHA256.fullmatch(pinned):
            raise ValueError(f"{name}: 재배포하는데 정본에 usd_sha256 이 없거나 형식이 아닙니다 — "
                             "어느 파일을 실었는지 고정되지 않습니다")
        rel = f"{ASSETS_DIR}/{usd.name}"
        if rel in out:
            raise ValueError(f"두 자산이 같은 이름으로 실립니다: {rel} ({out[rel].asset} · {name})")
        out[rel] = ShippedFile(usd, pinned, name)
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_shipped_sources(files: dict[str, ShippedFile], repo_root: str | Path) -> None:
    """Every file a distribution will carry exists in the tree with the bytes the canon pins.
    Raises FileNotFoundError / ValueError naming the file -- before anything is copied."""
    root = Path(repo_root)
    for rel, f in sorted(files.items()):
        where = f.src.relative_to(root) if f.src.is_relative_to(root) else f.src
        if not f.src.is_file():
            raise FileNotFoundError(
                f"redistributed=true 인데 파일이 없습니다: {where} ({f.asset}) — 정본이 배포하지 "
                "않는 자산을 배포한다고 말하고 있거나, 빌더를 다시 돌려야 합니다(받아 오는 로봇은 "
                "generation/robots/fetch_robot.py)")
        got = sha256_file(f.src)
        if got != f.sha256:
            raise ValueError(f"{where} 의 해시가 정본과 다릅니다 ({f.asset}) — 정본 {f.sha256[:16]}…, "
                             f"파일 {got[:16]}… — 정본이 고정한 바이트가 아니면 싣지 않습니다")
