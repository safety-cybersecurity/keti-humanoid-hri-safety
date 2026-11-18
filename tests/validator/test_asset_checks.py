"""C6's asset accounting: a distribution ships the redistributable assets its own episodes use,
with the textures those USDs reference, and every file under assets/ is accounted for.

The four registered variants (broken_asset_unreferenced, broken_texture_*) run through the
CLI in test_fixture_detection_reason.py. Here each other branch of the asset accounting gets
its own in-process case on a copy of a clean fixture, asserting the one sentence it exists
to say -- one variant per check leaves the branches beside it deletable with make check green.

What C6 cannot see from a distribution alone, and who does: whether bundle.files lists every
texture a USD really references (tests/generation/test_texture_refs.py, with pxr, on the built
USDs; the converter refuses rows that are not the textures.lock's), and whether the bytes are
the ones the repository pins (the publish pre-flight, publish_dataset.shipped_assets_match_source).
"""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from validator import make_fixture as F
from validator import validate_dataset as V


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixtures")
    F.build(root / "clean")
    F.build(root / F.MEDIA_CLEAN, with_media=True)
    return root


def _copy(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    return dst


def _c6(root: Path) -> list[V.Finding]:
    rep = V.run(root, {"C6"}, None, 3, None)
    return [f for f in rep.findings if f.severity == V.ERROR]


def _edit_assets(root: Path, fn) -> None:
    p = root / "meta" / "assets.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    fn(doc)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _textured(doc: dict) -> dict:
    return next(e for e in doc["scene"].values() if e["usd_path"] == F.TEXTURED_ASSET["path"])


def _says(errors, phrase):
    return any(phrase in f.message for f in errors)


# ── the layout both sides agree on ───────────────────────────────────

def test_the_validator_uses_the_converters_layout_and_row_shape():
    # Copies, because the validator does not import generation code (CONTRIBUTING.md 「함께 고치는 곳」).
    from generation import assets as A
    assert (V.ASSETS_DIR, V.TEXTURE_SUBDIR) == (A.ASSETS_DIR, A.TEXTURE_SUBDIR)
    assert (V.TEXTURE_ROW_KEYS, V.TEXTURE_ROW_OPTIONAL) == (A.TEXTURE_ROW_KEYS, A.TEXTURE_ROW_OPTIONAL)
    assert V.ROBOT_SHIP_KEYS == A.ROBOT_SHIP_KEYS


@pytest.mark.parametrize("row", [
    {"file": "a.png", "sha256": "0" * 64, "license": "CC0-1.0", "url": "https://x/a.png"},
    {"file": "a.png", "sha256": "0" * 64, "license": "CC0-1.0", "url": "https://x/p.zip", "member": "d/a.png"},
    {"file": "../a.png", "sha256": "0" * 64, "license": "CC0-1.0", "url": "u"},
    {"file": "d/a.png", "sha256": "0" * 64, "license": "CC0-1.0", "url": "u"},
    {"file": "a\\b.png", "sha256": "0" * 64, "license": "CC0-1.0", "url": "u"},
    {"file": "a.png", "sha256": "0" * 63, "license": "CC0-1.0", "url": "u"},
    {"file": "a.png", "sha256": "A" * 64, "license": "CC0-1.0", "url": "u"},
    {"file": "a.png", "sha256": "0" * 64, "license": "", "url": "u"},
    {"file": "a.png", "sha256": "0" * 64, "license": "CC0-1.0"},
    {"file": "a.png", "sha256": "0" * 64, "license": "CC0-1.0", "url": "u", "role": "x"},
    ["not", "a", "row"],
])
def test_the_two_row_readers_agree(row):
    """The converter refuses what C6 reports and vice versa -- one reading of a row."""
    from generation.assets import texture_row_problems
    assert V._texture_row_problems(row) == texture_row_problems(row)


# ── v0.1: a redistributable asset no episode uses is not asked for ───

def test_the_v0_1_shaped_fixture_declares_an_asset_it_does_not_ship(built):
    """The invariant is only exercised while the camera-less fixture's canon marks a textured
    asset redistributable that its episodes do not use -- as a v0.1 re-conversion's shipped
    meta/assets.json marks the S1d scene."""
    root = built / "clean"
    canon = json.loads((root / "meta" / "assets.json").read_text(encoding="utf-8"))
    eps = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    used = {a["path"] for v in eps["scene_assets"] for a in json.loads(v)}
    unused = [e for e in canon["scene"].values()
              if e.get("redistributed") and e["usd_path"] not in used]
    assert unused and any(e.get("bundle") for e in unused)
    assert not (root / "assets" / "textures").exists()
    assert not _c6(root)


def test_the_s1d_shaped_fixture_ships_and_hashes_its_textures(built):
    # 「몇 개를 보고 있는가」: C6 hashed every texture the shipped bundle lists.
    root = built / F.MEDIA_CLEAN
    rep = V.run(root, {"C6"}, None, 3, None)
    assert not [f for f in rep.findings if f.severity == V.ERROR]
    rows = _textured(json.loads((root / "meta" / "assets.json").read_text(encoding="utf-8")))
    assert rep.stats["shipped_textures"] == len(rows["bundle"]["files"]) == len(F.TEXTURES)


# ── the canon's claim, narrowed to the batch ────────────────────────

def test_an_asset_the_episodes_use_but_none_calls_redistributed_is_reported(built, tmp_path):
    """Every episode's flag for the textured asset flipped to false while the shipped canon
    says true: the canon-level sentence, beside the per-episode ones."""
    root = _copy(built / F.MEDIA_CLEAN, tmp_path)
    p = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    eps = pd.read_parquet(p)
    flipped = []
    for v in eps["scene_assets"]:
        items = json.loads(v)
        for a in items:
            if a["path"] == F.TEXTURED_ASSET["path"]:
                a["redistributed"] = False
        flipped.append(json.dumps(items, ensure_ascii=False))
    eps["scene_assets"] = flipped
    eps.to_parquet(p, index=False)
    errors = _c6(root)
    assert _says(errors, f"{F.TEXTURED_ASSET['path']} 를 재배포한다고 선언하는데 그것을 쓰는 에피소드 중 "
                         "어느 것도 그렇게 말하지 않음"), [f.line() for f in errors]


# ── the shipped provenance record ───────────────────────────────────

def test_a_shipped_bundle_without_its_file_list_is_reported(built, tmp_path):
    root = _copy(built / F.MEDIA_CLEAN, tmp_path)
    _edit_assets(root, lambda d: _textured(d)["bundle"].pop("files"))
    errors = _c6(root)
    assert _says(errors, "텍스처 묶음(bundle)을 선언하는데 파일 목록(bundle.files)이 없거나 비어 있음"), \
        [f.line() for f in errors]


@pytest.mark.parametrize("breakage, phrase", [
    (lambda r: r.update(sha256="not-a-hash"), "sha256 이 64자리 소문자 16진수가 아님"),
    (lambda r: r.update(file="../elsewhere.png"), "file 이 경로가 아닌 파일 이름이어야 함"),
    (lambda r: r.pop("url"), "url 가 비어 있지 않은 문자열이 아님"),
    (lambda r: r.update(licence="CC0"), "모르는 키"),
])
def test_a_row_that_is_not_a_provenance_record_is_reported(built, tmp_path, breakage, phrase):
    root = _copy(built / F.MEDIA_CLEAN, tmp_path)
    _edit_assets(root, lambda d: breakage(_textured(d)["bundle"]["files"][0]))
    errors = _c6(root)
    assert _says(errors, "bundle.files[0] 가 출처 기록이 아님") and _says(errors, phrase), \
        [f.line() for f in errors]


def test_one_texture_name_claimed_with_two_hashes_is_reported(built, tmp_path):
    """Two shipped USDs, one file name, two pins: only one file can sit at that path."""
    root = _copy(built / F.MEDIA_CLEAN, tmp_path)
    name = sorted(F.TEXTURES)[0]

    def claim_it_again(doc):
        flat = next(e for e in doc["scene"].values() if e["usd_path"] == F.ASSETS[2]["path"])
        row = dict(next(r for r in _textured(doc)["bundle"]["files"] if r["file"] == name),
                   sha256="f" * 64)
        flat["bundle"] = {"files": [row]}
    _edit_assets(root, claim_it_again)
    errors = _c6(root)
    assert _says(errors, f"텍스처 {name} 을 두 자산이 다른 바이트로 적음"), [f.line() for f in errors]


@pytest.mark.parametrize("rel", ["assets/extra/readme.txt", "assets/textures/sub/floor_albedo.png"])
def test_a_file_outside_the_shipped_layout_is_reported(built, tmp_path, rel):
    # Not a shipped USD, not a texture at assets/textures/<file>: nothing names it.
    root = _copy(built / F.MEDIA_CLEAN, tmp_path)
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_bytes(b"x\n")
    errors = _c6(root)
    assert _says(errors, f"이 배치의 에피소드가 쓰지 않는 파일: {rel}"), [f.line() for f in errors]


# ── the anchor: each episode lists what the shipped canon says it used ──

def _edit_scenarios(root: Path, fn) -> None:
    p = root / "meta" / "scenarios.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    fn(doc)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _edit_episodes(root: Path, fn) -> None:
    p = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    eps = pd.read_parquet(p)
    fn(eps)
    eps.to_parquet(p, index=False)


def _drop_from_scene_assets(path: str):
    def fn(eps):
        eps["scene_assets"] = [json.dumps([a for a in json.loads(v) if a["path"] != path],
                                          ensure_ascii=False) for v in eps["scene_assets"]]
    return fn


@pytest.mark.parametrize("robot_model", ["bot", "nobody", "bare", None, ["bot"]])
@pytest.mark.parametrize("scenario", [
    {"scenario_id": "S9", "scene": {"assets": ["a.usd"], "human_model": "human"}},
    {"scenario_id": "S9", "scene": {"assets": [], "human_model": None}},
    {"scenario_id": "S9"},
    {"scenario_id": "S9", "scene": {"assets": "a.usd"}},
    {"scenario_id": "S9", "scene": {"assets": [""], "human_model": None}},
    {"scenario_id": "S9", "scene": {"assets": [], "human_model": "nobody"}},
    {"scenario_id": "S9", "scene": {"assets": [], "human_model": ["human"]}},
    ["not", "a", "scenario"],
])
def test_the_two_anchor_readers_agree(scenario, robot_model):
    """The converter refuses a scenario or a robot C6 cannot read, and they require the same
    paths -- the robot's included (the episode names its robot, the scenario does not)."""
    from generation.assets import required_scene_assets
    canon = {"scene": {"human": {"usd_path": "human.usd"}},
             "robots": {"bot": {"usd_path": "bot.usd"}, "bare": {"license": "x"}}}
    got, problems = V._required_scene_assets(scenario, canon, robot_model)
    try:
        want = required_scene_assets(scenario, canon, robot_model)
    except ValueError:
        assert problems and got == set()
    else:
        assert not problems and got == want and "bot.usd" in got


def test_a_used_asset_dropped_from_every_scene_assets_is_reported_by_the_anchor(built, tmp_path):
    """v0.1 shape: the dummy dropped from every scene_assets, its file deleted. human_model and
    the scenario still say it was used. A C6 that asks only for what scene_assets lists gives
    0 ERRORs here."""
    root = _copy(built / "clean", tmp_path)
    gone = F.ASSETS[F.HUMAN_MODEL_ASSET]["path"]
    _edit_episodes(root, _drop_from_scene_assets(gone))
    (root / "assets" / Path(gone).name).unlink()
    errors = _c6(root)
    assert _says(errors, f"scene_assets 에 ['{gone}'] 가 없음"), [f.line() for f in errors]
    assert _says(errors, f"재배포한다고 선언한 자산이 배포본에 없음: assets/{Path(gone).name}")


def test_the_s1d_shaped_case_asks_for_the_usd_and_every_texture(built, tmp_path):
    """S1d declares its scene and skinned dummy in scene.assets. With the textured asset in the
    fixture scenario's scene.assets, leaving it out of every episode and deleting it and its
    textures is reported -- the USD and each texture -- instead of passing."""
    root = _copy(built / F.MEDIA_CLEAN, tmp_path)
    _edit_scenarios(root, lambda d: d["scenarios"][0]["scene"]["assets"].append(F.TEXTURED_ASSET["path"]))
    assert not _c6(root), "declaring what the episodes already list must stay green"
    _edit_episodes(root, _drop_from_scene_assets(F.TEXTURED_ASSET["path"]))
    shutil.rmtree(root / "assets" / "textures")
    (root / "assets" / Path(F.TEXTURED_ASSET["path"]).name).unlink()
    errors = _c6(root)
    assert _says(errors, f"scene_assets 에 ['{F.TEXTURED_ASSET['path']}'] 가 없음"), [f.line() for f in errors]
    assert _says(errors, f"assets/{Path(F.TEXTURED_ASSET['path']).name}")
    for name in F.TEXTURES:
        assert _says(errors, f"텍스처가 배포본에 없음: assets/textures/{name}"), name


def test_the_episodes_own_human_model_is_asked_for_too(built, tmp_path):
    """Either the scenario's human model or the episode's field, changed alone, still asks:
    here the episodes name another model than their scenario's, one they do not list."""
    root = _copy(built / "clean", tmp_path)
    _edit_assets(root, lambda d: d["scene"].__setitem__(
        "other_model", dict(d["scene"]["flat"], usd_path="keti/other_model.usd")))

    def rename(eps):
        eps["human_model"] = "other_model"
    _edit_episodes(root, rename)
    errors = _c6(root)
    assert _says(errors, "scene_assets 에 ['keti/other_model.usd'] 가 없음"), [f.line() for f in errors]


@pytest.mark.parametrize("model", [None, float("nan")])
def test_an_episode_without_a_human_model_falls_back_to_its_scenarios(built, tmp_path, model):
    # human_model is optional (spec §5, null for scenarios without one); absent, the scenario
    # still names the model, so the anchor does not switch off.
    root = _copy(built / "clean", tmp_path)

    def blank(eps):
        eps["human_model"] = model
    _edit_episodes(root, blank)
    assert not _c6(root)
    gone = F.ASSETS[F.HUMAN_MODEL_ASSET]["path"]
    _edit_episodes(root, _drop_from_scene_assets(gone))
    assert _says(_c6(root), f"scene_assets 에 ['{gone}'] 가 없음")


def test_a_scenario_and_episodes_without_a_human_model_ask_for_none(built, tmp_path):
    """The producer's null case (S3a..S5b): no model in the scenario nor the episode -- the
    dummy is then not something this batch used, and is not demanded."""
    root = _copy(built / "clean", tmp_path)
    _edit_scenarios(root, lambda d: d["scenarios"][0]["scene"].__setitem__("human_model", None))
    gone = F.ASSETS[F.HUMAN_MODEL_ASSET]["path"]

    def drop(eps):
        _drop_from_scene_assets(gone)(eps)
        eps["human_model"] = None
    _edit_episodes(root, drop)
    (root / "assets" / Path(gone).name).unlink()
    assert not _c6(root), [f.line() for f in _c6(root)]


@pytest.mark.parametrize("mutate, where, phrase", [
    (lambda d: d["scenarios"][0].pop("scene"), "scenarios", "S1a: scene 이 없음"),
    (lambda d: d["scenarios"][0]["scene"].pop("assets"), "scenarios",
     "S1a: scene.assets 가 비어 있지 않은 문자열의 목록이 아님"),
    (lambda d: d["scenarios"][0]["scene"].__setitem__("human_model", "nobody"), "scenarios",
     "S1a: scene.human_model 'nobody' 이 meta/assets.json 의 scene 에 없음"),
    (lambda d: d["scenarios"][0].__setitem__("scenario_id", "S9z"), "scenarios",
     "시나리오 'S1a' 가 meta/scenarios.json 에 없어"),
    (lambda d: d["scenarios"].append(dict(d["scenarios"][0])), "scenarios",
     "meta/scenarios.json 에서 시나리오를 읽지 못해"),
    (lambda eps: eps.__setitem__("human_model", "nobody"), "episodes",
     "human_model 'nobody' 이 meta/assets.json 의 scene 에 없어"),
])
def test_a_canon_the_anchor_cannot_read_is_reported(built, tmp_path, mutate, where, phrase):
    """Said, not skipped: each way the anchor could not be read is an ERROR -- otherwise
    removing it from the shipped canon would switch the completeness check off."""
    root = _copy(built / "clean", tmp_path)
    (_edit_scenarios if where == "scenarios" else _edit_episodes)(root, mutate)
    errors = _c6(root)
    assert _says(errors, phrase), [f.line() for f in errors]


# ── the robot: asked for through the episode's robot_model ──

def test_the_robot_is_asked_for_through_the_episodes_robot_model(built, tmp_path):
    """The robot dropped from every scene_assets. No scenario
    names its robot -- the episode's robot_model does, a key of the shipped meta/assets.json
    robots. The fixture's robot ships as the H1 does, as its folder, so its folder goes too (as the
    dummy's file does in the v0.1-shaped case above); a robot that does not ship -- the next test."""
    root = _copy(built / "clean", tmp_path)
    gone = F.ASSETS[F.ROBOT_ASSET]["path"]
    _edit_episodes(root, _drop_from_scene_assets(gone))
    shutil.rmtree(root / "assets" / F.ROBOT_DIR)
    errors = _c6(root)
    assert _says(errors, f"scene_assets 에 ['{gone}'] 가 없음"), [f.line() for f in errors]
    for name in F.ROBOT_FILES:
        assert _says(errors, f"이 배포본에 없음: assets/{F.ROBOT_DIR}/{name}"), (name, [f.line() for f in errors])


def _robot_kept_back(root: Path) -> str:
    """The fixture's robot as a robot that does not ship: in the canon and every scene_assets, not
    redistributed, nothing under assets/. Returns its path."""
    path = F.ASSETS[F.ROBOT_ASSET]["path"]
    _edit_assets(root, lambda d: d["robots"]["unitree_h1"].__setitem__("redistributed", False))

    def flag(eps):
        eps["scene_assets"] = [json.dumps([dict(a, redistributed=False) if a["path"] == path else a
                                           for a in json.loads(v)], ensure_ascii=False)
                               for v in eps["scene_assets"]]
    _edit_episodes(root, flag)
    shutil.rmtree(root / "assets" / F.ROBOT_DIR)
    return path


def test_a_robot_that_does_not_ship_is_asked_for_without_a_file(built, tmp_path):
    """The shape of a distribution whose robot does not ship: listing it is required, shipping it is
    not."""
    root = _copy(built / "clean", tmp_path)
    gone = _robot_kept_back(root)
    assert not _c6(root), [f.line() for f in _c6(root)]
    _edit_episodes(root, _drop_from_scene_assets(gone))
    errors = _c6(root)
    assert _says(errors, f"scene_assets 에 ['{gone}'] 가 없음"), [f.line() for f in errors]
    assert not _says(errors, "배포본에 없음"), [f.line() for f in errors]


@pytest.mark.parametrize("model", ["nobody", None])
def test_a_robot_model_the_shipped_canon_does_not_know_is_reported(built, tmp_path, model):
    # Said, not skipped: an episode whose robot the canon cannot name would switch the robot's
    # part of the anchor off.
    root = _copy(built / "clean", tmp_path)

    def rename(eps):
        eps["robot_model"] = model
    _edit_episodes(root, rename)
    errors = _c6(root)
    said = [f for f in errors if f"robot_model {model!r} 이 meta/assets.json 의 robots 에 없음" in f.message]
    assert said, [f.line() for f in errors]
    # Located at the episode whose field it is -- not at meta/scenarios.json, where the
    # scenario's part of the anchor is located.
    eps = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert {f.location for f in said} == set(eps["episode_id"]), [f.line() for f in said]


# ── shipped USDs are hashed against usd_sha256 ─────────

def test_every_shipped_usd_is_hashed(built):
    # The USDs directly under assets/: the fixture's assets but the robot, which ships as its folder (the
    # next section), and the textured asset of the camera fixture.
    rep = V.run(built / F.MEDIA_CLEAN, {"C6"}, None, 3, None)
    shipped = [p for p in (built / F.MEDIA_CLEAN / "assets").iterdir() if p.suffix == ".usd"]
    assert rep.stats["shipped_usds"] == len(shipped) == (len(F.ASSETS) - 1) + 1


@pytest.mark.parametrize("pin", [None, "abc"])
def test_a_shipped_usd_without_a_pin_is_reported(built, tmp_path, pin):
    root = _copy(built / "clean", tmp_path)

    def unpin(d):
        if pin is None:
            d["scene"]["flat"].pop("usd_sha256")
        else:
            d["scene"]["flat"]["usd_sha256"] = pin
    _edit_assets(root, unpin)
    errors = _c6(root)
    assert _says(errors, f"재배포 자산 {F.ASSETS[2]['path']} 의 usd_sha256 이 meta/assets.json 에 없거나"), \
        [f.line() for f in errors]


def test_a_bundle_of_null_is_reported(built, tmp_path):
    # The converter refuses the same entry (generation.assets.shipped_asset_files).
    root = _copy(built / F.MEDIA_CLEAN, tmp_path)
    _edit_assets(root, lambda d: _textured(d).__setitem__("bundle", None))
    errors = _c6(root)
    assert _says(errors, "텍스처 묶음(bundle)을 선언하는데 파일 목록(bundle.files)이 없거나 비어 있음"), \
        [f.line() for f in errors]


# ── a robot that ships as its own folder ──────────────────────────────

def _robot(doc: dict) -> dict:
    return doc["robots"]["unitree_h1"]


@pytest.mark.parametrize("name", ["clean", F.MEDIA_CLEAN])
def test_c6_accepts_the_robot_folder(built, name):
    """The fixture's robot ships as the H1 does: its USD, the layer beside it and its licence file under
    assets/<dir>/, each a row of the shipped meta/assets.json ship.files. C6 hashes every one."""
    root = built / name
    rep = V.run(root, {"C6"}, None, 3, None)
    assert not [f for f in rep.findings if f.severity == V.ERROR], [f.line() for f in rep.findings]
    rows = _robot(json.loads((root / "meta" / "assets.json").read_text(encoding="utf-8")))["ship"]["files"]
    assert rep.stats["shipped_robot_files"] == len(rows) == len(F.ROBOT_FILES) == 3
    assert sorted(p.name for p in (root / "assets" / F.ROBOT_DIR).iterdir()) == sorted(F.ROBOT_FILES)


@pytest.mark.parametrize("name", sorted(F.ROBOT_FILES))
def test_c6_refuses_a_robot_file_with_another_hash(built, tmp_path, name):
    root = _copy(built / "clean", tmp_path)
    p = root / "assets" / F.ROBOT_DIR / name
    p.write_bytes(p.read_bytes() + b"# not the pinned file\n")
    errors = _c6(root)
    assert _says(errors, f"assets/{F.ROBOT_DIR}/{name} 의 SHA-256 이 meta/assets.json(unitree_h1 ship.files)과 "
                         "다름"), [f.line() for f in errors]


def test_c6_refuses_a_missing_licence_file(built, tmp_path):
    """BSD-3 asks that the licence text travels with the files: the folder without it is reported, and so is
    a ship block that names a licence file none of its rows is."""
    root = _copy(built / "clean", tmp_path)
    (root / "assets" / F.ROBOT_DIR / "LICENSE.txt").unlink()
    errors = _c6(root)
    assert _says(errors, f"재배포 로봇 unitree_h1 의 라이선스 파일이 배포본에 없음: assets/{F.ROBOT_DIR}/LICENSE.txt"), \
        [f.line() for f in errors]
    root = _copy(built / "clean", tmp_path / "unnamed")
    _edit_assets(root, lambda d: _robot(d)["ship"].update(license_file="COPYING"))
    errors = _c6(root)
    assert _says(errors, "license_file 'COPYING' 의 행이 ship.files 에 없음"), [f.line() for f in errors]


@pytest.mark.parametrize("rel", ["assets/h1.usd", f"assets/{F.ROBOT_DIR}/stray.usd", "assets/other_robot/h1.usd"])
def test_a_robot_file_outside_its_rows_is_reported(built, tmp_path, rel):
    # The robot's USD beside the scene USDs, a file no row names inside its folder, a folder of no robot:
    # nothing in the shipped canon accounts for any of them.
    root = _copy(built / "clean", tmp_path)
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_bytes(F.USD_PLACEHOLDER)
    errors = _c6(root)
    assert _says(errors, f"이 배치의 에피소드가 쓰지 않는 파일: {rel}"), [f.line() for f in errors]


def _ship_entry() -> dict:
    rows = [{"file": n, "sha256": f"{i}" * 64, "license": "BSD-3-Clause", "url": f"{{ISAACLAB_NUCLEUS_DIR}}/R/{n}"}
            for i, n in enumerate(("bot.usd", "meshes.usd", "LICENSE.txt"), start=1)]
    return {"usd_path": "{ISAACLAB_NUCLEUS_DIR}/R/bot.usd", "usd_sha256": "1" * 64, "redistributed": True,
            "ship": {"dir": "bot", "source_dir": "build/bot", "license_file": "LICENSE.txt", "files": rows}}


@pytest.mark.parametrize("edit", [
    lambda e: None,
    lambda e: e.__setitem__("ship", ["bot"]),
    lambda e: e["ship"].pop("dir"),
    lambda e: e["ship"].pop("source_dir"),
    lambda e: e["ship"].update(source_dir=""),
    lambda e: e["ship"].update(note="x"),
    lambda e: e["ship"].update(dir="../bot"),
    lambda e: e["ship"].update(dir="textures"),
    lambda e: e["ship"].update(dir=""),
    lambda e: e["ship"].update(files=[]),
    lambda e: e["ship"].update(files="bot.usd"),
    lambda e: e["ship"]["files"][1].update(file="d/meshes.usd"),
    lambda e: e["ship"]["files"][1].pop("url"),
    lambda e: e["ship"]["files"].append(dict(e["ship"]["files"][1])),
    lambda e: e["ship"].update(license_file="COPYING"),
    lambda e: e.update(usd_sha256="2" * 64),
    lambda e: e.update(usd_sha256="abc"),
    lambda e: e.pop("usd_sha256"),
    lambda e: e.update(usd_path="{ISAACLAB_NUCLEUS_DIR}/R/other.usd"),
    lambda e: e.__setitem__("bundle", {"files": []}),
])
def test_the_two_ship_readers_agree(edit):
    """The converter refuses the ship blocks C6 reports and ships the ones it accepts -- one reading of the
    block. (Where the source lies in the repository the converter checks too; a distribution has no
    repository, and C6 asks only that it is named.)"""
    from generation.assets import robot_ship_files
    entry = _ship_entry()
    edit(entry)
    problems = V._robot_ship_problems("bot", entry)
    try:
        robot_ship_files("bot", entry, Path("."))
    except ValueError:
        assert problems
    else:
        assert problems == []


# ── an asset that composes others (the Stage) ─────────────────────────

def _composed_canon() -> dict:
    """A robot that ships as its folder, three scene USDs and an entry that composes all four."""
    canon = {"robots": {"bot": {"usd_path": "{N}/R/bot.usd", "redistributed": True}},
             "scene": {k: {"usd_path": f"build/{k}.usd", "redistributed": True} for k in ("hall", "person", "box")}}
    canon["scene"]["stage"] = {"usd_path": "build/stage.usd", "redistributed": True,
                               "composes": ["hall", "person", "box", "bot"]}
    return canon


EVERY = {"hall", "person", "box", "bot"}


@pytest.mark.parametrize("edit, shipped", [
    (lambda c: None, EVERY),
    (lambda c: None, {"hall", "person", "bot"}),                  # a release of a scenario without the box
    (lambda c: None, set()),
    (lambda c: None, EVERY | {"other"}),
    (lambda c: c["scene"]["stage"].update(redistributed=False), EVERY),
    (lambda c: c["scene"]["stage"].update(redistributed="true"), EVERY),
    (lambda c: c["scene"]["stage"].update(composes="hall"), EVERY),
    (lambda c: c["scene"]["stage"].update(composes=[]), EVERY),
    (lambda c: c["scene"]["stage"].update(composes=["hall", 3]), EVERY),
    (lambda c: c["scene"]["stage"].update(composes=["hall", "hall"]), EVERY),
    (lambda c: c["scene"]["stage"].update(composes=["hall", "nobody"]), EVERY),
    (lambda c: c["scene"]["stage"].update(composes=["hall", "stage"]), EVERY),
    (lambda c: c["scene"]["stage"].update(bundle={"files": []}), EVERY),
    (lambda c: c["scene"]["stage"].update(ship={}), EVERY),
    (lambda c: c["robots"].__setitem__("hall", {"usd_path": "x.usd"}), EVERY),
    (lambda c: c["scene"].__setitem__("outer", {"usd_path": "o.usd", "redistributed": True,
                                                "composes": ["stage", "hall"]}), EVERY),
    (lambda c: c["scene"].__setitem__("second", {"usd_path": "s.usd", "redistributed": True,
                                                 "composes": ["hall", "box"]}), {"hall", "box"}),
    (lambda c: c["scene"]["stage"].pop("usd_path"), EVERY),
    (lambda c: c["scene"]["stage"].update(usd_path=3), EVERY),
    (lambda c: c["scene"]["stage"].update(usd_path=""), {"hall"}),
])
def test_the_validator_agrees_on_the_stage(edit, shipped):
    """The converter ships the composed entries C6 asks for and refuses the composes C6 reports -- one reading
    of the rule (generation.assets.composed_shipped)."""
    from generation.assets import composed_shipped
    canon = _composed_canon()
    edit(canon)
    got, problems = V._composed_shipped(canon, shipped)
    try:
        want = composed_shipped(canon, shipped)
    except ValueError:
        assert problems and got == set()
    else:
        assert not problems and got == want


@pytest.mark.parametrize("edit", [
    lambda e: e.pop("usd_path"),
    lambda e: e.update(usd_path=None),
    lambda e: e.update(usd_path=3),
    lambda e: e.update(usd_path=""),
    lambda e: e.update(usd_path=["build/stage.usd"]),
], ids=["missing", "null", "number", "empty", "list"])
def test_a_composed_entry_that_names_no_file_is_refused_on_both_sides(edit):
    """A composed entry is a file like any other asset, named by its usd_path: without a non-empty string there
    the converter refuses the canon and C6 reports it -- shipped or not, as every composes is read."""
    from generation.assets import composed_shipped
    canon = _composed_canon()
    edit(canon["scene"]["stage"])
    for shipped in (EVERY, set()):
        got, problems = V._composed_shipped(canon, shipped)
        assert got == set() and any(p.startswith("stage: usd_path ") for p in problems), problems
        with pytest.raises(ValueError, match="stage: usd_path "):
            composed_shipped(canon, shipped)


STAGE_BYTES = b"#usda 1.0\n# fixture stage\n"


def _with_stage(root: Path, composes: list[str]) -> None:
    """The fixture's canon gains an entry composing ``composes``, its file at assets/stage.usd, NOTICE its name."""
    _edit_assets(root, lambda d: d["scene"].__setitem__("stage", {
        "usd_path": "keti/stage.usd", "license": "KETI-own", "redistributed": True,
        "usd_sha256": hashlib.sha256(STAGE_BYTES).hexdigest(), "composes": composes}))
    (root / "assets" / "stage.usd").write_bytes(STAGE_BYTES)
    with open(root / "NOTICE", "a", encoding="utf-8") as fh:
        fh.write("\n5) keti/stage.usd — 엮은 장면\n")


def _clean_composes() -> list[str]:
    # What the camera-less fixture ships: its robot (a folder), its dummy and its flat scene.
    return ["unitree_h1", F.human_model_key(), "flat"]


def test_c6_hashes_a_composed_asset_whose_components_all_ship(built, tmp_path):
    root = _copy(built / "clean", tmp_path)
    _with_stage(root, _clean_composes())
    rep = V.run(root, {"C6"}, None, 3, None)
    assert not [f for f in rep.findings if f.severity == V.ERROR], [f.line() for f in rep.findings]
    assert rep.stats["shipped_usds"] == len([p for p in (root / "assets").iterdir() if p.suffix == ".usd"])


def test_c6_does_not_ask_for_a_composed_asset_a_component_of_which_does_not_ship(built, tmp_path):
    """The textured asset is redistributable but the camera-less fixture's episodes do not use it -- as a release
    of scenarios that carry no box. A stage composing it is not asked for, and its file is a file nothing ships."""
    root = _copy(built / "clean", tmp_path)
    _with_stage(root, _clean_composes() + ["textured"])
    errors = _c6(root)
    assert _says(errors, "이 배치의 에피소드가 쓰지 않는 파일: assets/stage.usd"), [f.line() for f in errors]
    (root / "assets" / "stage.usd").unlink()
    assert not _c6(root), [f.line() for f in _c6(root)]


def test_c6_refuses_a_composed_asset_missing_or_with_another_hash(built, tmp_path):
    root = _copy(built / "clean", tmp_path)
    _with_stage(root, _clean_composes())
    (root / "assets" / "stage.usd").write_bytes(STAGE_BYTES + b"# another\n")
    errors = _c6(root)
    assert _says(errors, "assets/stage.usd 의 SHA-256 이 meta/assets.json 의 usd_sha256 과 다름"), \
        [f.line() for f in errors]
    (root / "assets" / "stage.usd").unlink()
    errors = _c6(root)
    assert _says(errors, "재배포한다고 선언한 자산이 배포본에 없음: assets/stage.usd"), [f.line() for f in errors]


def test_c6_asks_notice_for_a_composed_asset(built, tmp_path):
    root = _copy(built / "clean", tmp_path)
    _with_stage(root, _clean_composes())
    notice = (root / "NOTICE").read_text(encoding="utf-8")
    (root / "NOTICE").write_text(notice.replace("stage.usd", "the stage"), encoding="utf-8")
    errors = _c6(root)
    assert _says(errors, "재배포 자산이 NOTICE 에 없음: keti/stage.usd"), [f.line() for f in errors]


def test_c6_reports_a_composes_it_cannot_read(built, tmp_path):
    root = _copy(built / "clean", tmp_path)
    _with_stage(root, _clean_composes() + ["nobody"])
    errors = _c6(root)
    assert _says(errors, "엮은 자산을 읽지 못함: stage: composes 의 nobody 가"), \
        [f.line() for f in errors]
    assert _says(errors, "이 배치의 에피소드가 쓰지 않는 파일: assets/stage.usd")


def test_the_validator_reports_a_composed_entry_without_a_path_instead_of_failing(built, tmp_path, capsys):
    """The clean fixture passes C6; given a composed entry that names no file -- composing only the robot, which
    ships -- the validator ends with an ERROR (exit 1) that names the key, not as a validator failure (exit 3)."""
    root = _copy(built / "clean", tmp_path)
    assert V.main([str(root), "--checks", "C6"]) == 0
    _edit_assets(root, lambda d: d["scene"].__setitem__("stage", {"redistributed": True, "composes": ["unitree_h1"]}))
    capsys.readouterr()
    assert V.main([str(root), "--checks", "C6"]) == 1
    out = capsys.readouterr()
    assert "검증기 실행 실패" not in out.err, out.err
    assert "엮은 자산을 읽지 못함: stage: usd_path 가 비어 있지 않은 문자열이 아님: None" in out.out, out.out


def test_c6_does_not_ask_for_a_composed_asset_an_episode_names(built, tmp_path):
    """An episode that lists the stage in its scene_assets does not make it ship (the converter ships it by
    composition alone): with a component that does not ship, C6 asks for no stage file either."""
    root = _copy(built / "clean", tmp_path)
    _with_stage(root, _clean_composes() + ["textured"])
    (root / "assets" / "stage.usd").unlink()

    def name_it(eps):
        eps["scene_assets"] = [json.dumps(json.loads(v) + [{"path": "keti/stage.usd", "license": "KETI-own",
                                                            "redistributed": True}], ensure_ascii=False)
                               for v in eps["scene_assets"]]
    _edit_episodes(root, name_it)
    assert not _c6(root), [f.line() for f in _c6(root)]
