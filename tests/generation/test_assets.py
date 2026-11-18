"""자산 경로 정규화와 정본 조인. Isaac 없이 돈다."""

import pytest

from generation.assets import asset_entry, canonical_usd_path

NUCLEUS = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab"
REPO = "/home/someone/repo"

CANON = {
    "robots": {"unitree_h1": {
        "usd_path": "{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
        "license": "BSD-3-Clause", "redistributed": False, "embodiment": "biped"}},
    "scene": {"keti_dummy_ts15066_v1": {
        "usd_path": "generation/dummy/build/keti_dummy_ts15066_v1.usd",
        "license": "CC-BY-4.0", "redistributed": True}},
}


def test_a_resolved_nucleus_url_becomes_the_template_form():
    # Isaac Lab 은 실행 시점에 ISAACLAB_NUCLEUS_DIR 을 S3 URL 로 푼다. 그대로
    # 기록하면 정본과 문자열이 달라 조인이 깨진다.
    got = canonical_usd_path(f"{NUCLEUS}/Robots/Unitree/H1/h1.usd", NUCLEUS, REPO)
    assert got == "{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd"


def test_an_absolute_repo_path_becomes_repo_relative():
    # 절대 경로는 기계마다 다르고 사용자 이름을 담는다.
    got = canonical_usd_path(f"{REPO}/generation/dummy/build/x.usd", NUCLEUS, REPO)
    assert got == "generation/dummy/build/x.usd"


def test_a_path_outside_both_is_left_alone():
    # 조용히 고치지 않는다 — 정본에 없으면 asset_entry 가 멈춘다.
    got = canonical_usd_path("/opt/other/thing.usd", NUCLEUS, REPO)
    assert got == "/opt/other/thing.usd"


def test_an_already_canonical_path_is_unchanged():
    for path in ("{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
                 "generation/dummy/build/keti_dummy_ts15066_v1.usd"):
        assert canonical_usd_path(path, NUCLEUS, REPO) == path


def test_asset_entry_returns_the_schema_shape():
    e = asset_entry(CANON, "{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd")
    assert e == {"path": "{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
                 "license": "BSD-3-Clause", "redistributed": False,
                 "key": "unitree_h1"}


def test_asset_entry_finds_scene_assets_too():
    e = asset_entry(CANON, "generation/dummy/build/keti_dummy_ts15066_v1.usd")
    assert e["key"] == "keti_dummy_ts15066_v1" and e["redistributed"] is True


def test_an_unknown_asset_stops_rather_than_defaulting():
    # 라이선스를 모르는 자산이 배포본에 들어가는 것보다 멈추는 편이 낫다.
    with pytest.raises(ValueError, match="unknown.usd"):
        asset_entry(CANON, "/somewhere/unknown.usd")


def test_a_missing_or_empty_license_stops_the_join():
    for bad in (None, "", "   ", 3):
        canon = {"robots": {}, "scene": {"x": {"usd_path": "p.usd", "license": bad,
                                               "redistributed": True}}}
        with pytest.raises(ValueError, match="license"):
            asset_entry(canon, "p.usd")


def test_a_non_boolean_redistributed_stops_the_join():
    # bool("false") 는 True 다. 배포하지 않는 자산이 배포한다고 게시될 수 있었다.
    for bad in ("false", "true", 0, 1, None):
        canon = {"robots": {}, "scene": {"x": {"usd_path": "p.usd",
                                               "license": "CC-BY-4.0",
                                               "redistributed": bad}}}
        with pytest.raises(ValueError, match="redistributed"):
            asset_entry(canon, "p.usd")


# ── what a distribution ships ─────────────────────────────

import hashlib  # noqa: E402

from generation.assets import (bundle_per_lock, check_shipped_sources, shipped_asset_files,  # noqa: E402
                               texture_rows_from_lock)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree(tmp_path):
    """A repository tree: a plain redistributable USD, a textured one whose texture a
    scene-format lock pins, and a USD the canon keeps back."""
    repo = tmp_path / "repo"
    files = {"build/a.usd": b"#usda 1.0\n# a\n", "build/b.usd": b"#usda 1.0\n# b\n",
             "build/textures/wood.jpg": b"wood pixels\n", "build/kept.usd": b"#usda 1.0\n# kept\n"}
    for rel, data in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(data)
    lock = repo / "generation/scene/textures.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("# comment\nwood w diffuse CC0 polyhaven https://example.invalid/wood.jpg "
                    f"{_sha(files['build/textures/wood.jpg'])}\n", encoding="utf-8")
    canon = {"robots": {}, "scene": {
        "a": {"usd_path": "build/a.usd", "redistributed": True, "usd_sha256": _sha(files["build/a.usd"])},
        "b": {"usd_path": "build/b.usd", "redistributed": True, "usd_sha256": _sha(files["build/b.usd"]),
              "bundle": {"textures_dir": "build/textures",
                         "textures_lock": "generation/scene/textures.lock",
                         "files": [{"file": "wood.jpg", "sha256": _sha(files["build/textures/wood.jpg"]),
                                    "license": "CC0-1.0", "url": "https://example.invalid/wood.jpg"}]}},
        "kept": {"usd_path": "build/kept.usd", "redistributed": False},
    }}
    return repo, canon


def test_only_the_redistributable_assets_the_batch_uses_ship(tmp_path):
    repo, canon = _tree(tmp_path)
    assert set(shipped_asset_files(canon, ["build/a.usd"], repo)) == {"assets/a.usd"}
    assert set(shipped_asset_files(canon, ["build/b.usd"], repo)) == {"assets/b.usd",
                                                                      "assets/textures/wood.jpg"}
    # Kept back by the canon, or unknown to it (asset_entry refuses those, earlier): nothing ships.
    assert shipped_asset_files(canon, ["build/kept.usd", "nowhere/x.usd"], repo) == {}
    got = shipped_asset_files(canon, ["build/a.usd", "build/b.usd"], repo)
    assert got["assets/textures/wood.jpg"].src == repo / "build/textures/wood.jpg"
    assert got["assets/textures/wood.jpg"].asset == "b"
    check_shipped_sources(got, repo)


def test_a_shipped_asset_without_a_pinned_hash_is_refused(tmp_path):
    repo, canon = _tree(tmp_path)
    canon["scene"]["a"]["usd_sha256"] = "abc"
    with pytest.raises(ValueError, match="a: 재배포하는데 정본에 usd_sha256 이 없거나"):
        shipped_asset_files(canon, ["build/a.usd"], repo)


def test_two_assets_under_one_file_name_are_refused(tmp_path):
    repo, canon = _tree(tmp_path)
    (repo / "other").mkdir()
    (repo / "other/a.usd").write_bytes(b"#usda 1.0\n# another a\n")
    canon["scene"]["a2"] = dict(canon["scene"]["a"], usd_path="other/a.usd")
    with pytest.raises(ValueError, match="두 자산이 같은 이름으로 실립니다: assets/a.usd"):
        shipped_asset_files(canon, ["build/a.usd", "other/a.usd"], repo)


@pytest.mark.parametrize("path", ["https://example.invalid/a.usd", "{ISAACLAB_NUCLEUS_DIR}/a.usd",
                                  "/opt/elsewhere/a.usd", "../outside/a.usd"])
def test_a_shipped_path_that_is_not_the_repositorys_is_refused(tmp_path, path):
    repo, canon = _tree(tmp_path)
    canon["scene"]["a"]["usd_path"] = path
    with pytest.raises(ValueError, match="a.usd_path 가 저장소"):
        shipped_asset_files(canon, [path], repo)


def test_a_textures_dir_outside_the_repository_is_refused(tmp_path):
    repo, canon = _tree(tmp_path)
    canon["scene"]["b"]["bundle"]["textures_dir"] = "../elsewhere"
    with pytest.raises(ValueError, match="b.bundle.textures_dir 가 저장소 밖"):
        shipped_asset_files(canon, ["build/b.usd"], repo)


@pytest.mark.parametrize("files", [None, []])
def test_a_bundle_without_its_rows_is_refused(tmp_path, files):
    repo, canon = _tree(tmp_path)
    if files is None:
        del canon["scene"]["b"]["bundle"]["files"]
    else:
        canon["scene"]["b"]["bundle"]["files"] = files
    with pytest.raises(ValueError, match="파일 목록\\(files\\)이 없습니다"):
        shipped_asset_files(canon, ["build/b.usd"], repo)


def test_a_row_that_is_not_a_provenance_record_is_refused(tmp_path):
    repo, canon = _tree(tmp_path)
    canon["scene"]["b"]["bundle"]["files"][0]["file"] = "../wood.jpg"
    with pytest.raises(ValueError, match="bundle.files\\[0\\] — file 이 경로가 아닌 파일 이름"):
        shipped_asset_files(canon, ["build/b.usd"], repo)


def test_a_lock_nobody_taught_the_reader_is_refused(tmp_path):
    repo, canon = _tree(tmp_path)
    canon["scene"]["b"]["bundle"]["textures_lock"] = "generation/other/textures.lock"
    with pytest.raises(ValueError, match="을 읽는 방법을 모릅니다"):
        shipped_asset_files(canon, ["build/b.usd"], repo)


def test_a_scene_lock_licence_other_than_cc0_does_not_become_cc0_rows(tmp_path, monkeypatch):
    # parse_lock admits one licence today; the SPDX mapping must not outlive that.
    from generation.scene import fetch_textures
    repo, canon = _tree(tmp_path)
    lock = repo / "generation/scene/textures.lock"
    lock.write_text(lock.read_text(encoding="utf-8").replace(" CC0 ", " CC-BY "), encoding="utf-8")
    monkeypatch.setattr(fetch_textures, "REQUIRED_LICENSE", "CC-BY")
    with pytest.raises(ValueError, match="라이선스가 CC0 가 아닙니다"):
        texture_rows_from_lock(canon["scene"]["b"], repo)


def _skin_tree(tmp_path, source="pack:skins/x/skin.png", url="https://example.invalid/pack.zip"):
    repo = tmp_path / "skinrepo"
    lock = repo / "generation/dummy/skin/textures.lock"
    lock.parent.mkdir(parents=True)
    sha = "1" * 64
    lock.write_text(f"skin.png {sha} 3 {source} CC0-1.0\n"
                    f"clothes/x/x.mhclo {'2' * 64} 9 pack:clothes/x/x.mhclo CC0-1.0\n", encoding="utf-8")
    spec = repo / "generation/dummy/skin/skin_spec.yaml"
    spec.write_text("source:\n  system_assets:\n" + (f"    url: {url}\n" if url else "    pack: p.zip\n"),
                    encoding="utf-8")
    entry = {"spec": "generation/dummy/skin/skin_spec.yaml",
             "bundle": {"textures_lock": "generation/dummy/skin/textures.lock"}}
    return repo, entry, sha


def test_the_skin_lock_gives_its_texture_rows_with_the_packs_url(tmp_path):
    repo, entry, sha = _skin_tree(tmp_path)
    assert texture_rows_from_lock(entry, repo) == [
        {"file": "skin.png", "sha256": sha, "license": "CC0-1.0",
         "url": "https://example.invalid/pack.zip", "member": "skins/x/skin.png"}]


def test_a_skin_texture_from_outside_the_pack_is_refused(tmp_path):
    repo, entry, _ = _skin_tree(tmp_path, source="file:somewhere/skin.png")
    with pytest.raises(ValueError, match="출처가 팩 안의 경로가 아닙니다"):
        texture_rows_from_lock(entry, repo)


def test_a_skin_spec_without_the_pack_url_is_refused(tmp_path):
    repo, entry, _ = _skin_tree(tmp_path, url=None)
    with pytest.raises(ValueError, match="source.system_assets.url 이 없습니다"):
        texture_rows_from_lock(entry, repo)
    del entry["spec"]
    with pytest.raises(ValueError, match="명세\\(spec\\)가 정본 항목에 없습니다"):
        texture_rows_from_lock(entry, repo)


def _ci_tree(tmp_path, **overrides):
    """A repository whose bundle names the KETI CI pin: one PNG, a member of the official archive."""
    import json
    repo = tmp_path / "cirepo"
    lock = repo / "generation/scene/keti_ci.json"
    lock.parent.mkdir(parents=True)
    pin = {"url": "https://example.invalid/KETI_CI.zip?a=1", "archive_sha256": "1" * 64,
           "member": "CI-가로타입#1.png", "member_sha256": "2" * 64, "file": "keti_ci_en.png",
           "license": "LicenseRef-KETI-CI"}
    pin.update(overrides)
    lock.write_text(json.dumps(pin, ensure_ascii=False), encoding="utf-8")
    return repo, {"bundle": {"textures_lock": "generation/scene/keti_ci.json"}}


def test_the_ci_lock_gives_its_one_row(tmp_path):
    repo, entry = _ci_tree(tmp_path)
    assert texture_rows_from_lock(entry, repo) == [
        {"file": "keti_ci_en.png", "sha256": "2" * 64, "license": "LicenseRef-KETI-CI",
         "url": "https://example.invalid/KETI_CI.zip?a=1", "member": "CI-가로타입#1.png"}]


def test_a_ci_pin_under_another_licence_is_refused(tmp_path):
    # the CI is the distributor's own; a row calling it CC0 or CC BY would claim a licence nobody gave
    repo, entry = _ci_tree(tmp_path, license="CC0-1.0")
    with pytest.raises(ValueError, match="LicenseRef-KETI-CI"):
        texture_rows_from_lock(entry, repo)


# ── a bundle that names several locks: CC0 textures and the KETI CI under one USD ──

SCENE_LOCK, CI_LOCK = "generation/scene/textures.lock", "generation/scene/keti_ci.json"


def _two_lock_tree(tmp_path, png=b"ci pixels\n"):
    """_tree's textured USD b whose textures come from two pins: the scene lock's wood and the KETI CI
    pinned apart, both beside the USD. Its bundle names the two locks, one item per lock for each
    per-lock key, and its files are the wood row then the CI row."""
    import json
    repo, canon = _tree(tmp_path)
    (repo / "build/textures/keti_ci_en.png").write_bytes(png)
    pin = {"url": "https://example.invalid/KETI_CI.zip?a=1", "archive_sha256": "1" * 64,
           "member": "CI-가로타입#1.png", "member_sha256": _sha(png), "file": "keti_ci_en.png",
           "license": "LicenseRef-KETI-CI"}
    (repo / CI_LOCK).write_text(json.dumps(pin, ensure_ascii=False), encoding="utf-8")
    b = canon["scene"]["b"]["bundle"]
    b["textures_lock"] = [SCENE_LOCK, CI_LOCK]
    b["textures_lock_sha256"] = [_sha((repo / rel).read_bytes()) for rel in b["textures_lock"]]
    b["textures_license"] = ["CC0-1.0", "LicenseRef-KETI-CI"]
    b["files"].append({"file": "keti_ci_en.png", "sha256": _sha(png), "license": "LicenseRef-KETI-CI",
                       "url": pin["url"], "member": pin["member"]})
    return repo, canon


def test_a_bundle_of_several_locks_gives_their_rows_in_the_order_it_names_them(tmp_path):
    repo, canon = _two_lock_tree(tmp_path)
    entry = canon["scene"]["b"]
    wood, ci = entry["bundle"]["files"]
    assert texture_rows_from_lock(entry, repo) == [wood, ci]
    entry["bundle"]["textures_lock"] = [CI_LOCK, SCENE_LOCK]
    assert texture_rows_from_lock(entry, repo) == [ci, wood]
    # the rows no longer say what the locks pin, in that order: refused, naming both locks
    with pytest.raises(ValueError, match=f"bundle.files 가 {CI_LOCK} · {SCENE_LOCK} 과 다릅니다"):
        shipped_asset_files(canon, ["build/b.usd"], repo)


def test_a_bundle_of_several_locks_ships_every_row_beside_the_usd(tmp_path):
    repo, canon = _two_lock_tree(tmp_path)
    got = shipped_asset_files(canon, ["build/b.usd"], repo)
    assert set(got) == {"assets/b.usd", "assets/textures/wood.jpg", "assets/textures/keti_ci_en.png"}
    assert got["assets/textures/keti_ci_en.png"].src == repo / "build/textures/keti_ci_en.png"
    check_shipped_sources(got, repo)


def test_a_bundle_whose_rows_skip_one_of_its_locks_is_refused(tmp_path):
    # the CI row left out of files, or the CI lock left out of the list: rows and locks no longer agree
    repo, canon = _two_lock_tree(tmp_path)
    canon["scene"]["b"]["bundle"]["files"].pop()
    with pytest.raises(ValueError, match="bundle.files 가 .* 과 다릅니다"):
        shipped_asset_files(canon, ["build/b.usd"], repo)
    repo, canon = _two_lock_tree(tmp_path / "again")
    canon["scene"]["b"]["bundle"]["textures_lock"] = [SCENE_LOCK]
    with pytest.raises(ValueError, match=f"bundle.files 가 {SCENE_LOCK} 과 다릅니다"):
        shipped_asset_files(canon, ["build/b.usd"], repo)


def test_a_lock_nobody_taught_the_reader_is_refused_inside_a_list_too(tmp_path):
    repo, canon = _two_lock_tree(tmp_path)
    canon["scene"]["b"]["bundle"]["textures_lock"] = [SCENE_LOCK, "generation/other/pin.json"]
    with pytest.raises(ValueError, match="'generation/other/pin.json' 을 읽는 방법을 모릅니다"):
        texture_rows_from_lock(canon["scene"]["b"], repo)


def test_the_per_lock_keys_follow_the_shape_of_the_lock_list():
    one = {"textures_lock": CI_LOCK, "textures_lock_sha256": "a" * 64, "textures_license": "LicenseRef-KETI-CI"}
    assert bundle_per_lock(one, "textures_lock") == [CI_LOCK]
    assert bundle_per_lock(one, "textures_lock_sha256") == ["a" * 64]
    assert bundle_per_lock(one, "textures_license") == ["LicenseRef-KETI-CI"]
    two = {"textures_lock": [SCENE_LOCK, CI_LOCK], "textures_lock_sha256": ["a" * 64, "b" * 64],
           "textures_license": ["CC0-1.0", "LicenseRef-KETI-CI"]}
    assert bundle_per_lock(two, "textures_lock") == [SCENE_LOCK, CI_LOCK]
    assert bundle_per_lock(two, "textures_lock_sha256") == ["a" * 64, "b" * 64]
    assert bundle_per_lock(two, "textures_license") == ["CC0-1.0", "LicenseRef-KETI-CI"]


@pytest.mark.parametrize("edit,phrase", [
    (lambda b: b.update(textures_lock=[]), "textures_lock 이 lock 경로 하나"),
    (lambda b: b.update(textures_lock=[SCENE_LOCK, SCENE_LOCK]), "textures_lock 이 lock 경로 하나"),
    (lambda b: b.update(textures_lock=[SCENE_LOCK, ""]), "textures_lock 이 lock 경로 하나"),
    (lambda b: b.update(textures_lock=[SCENE_LOCK, 3]), "textures_lock 이 lock 경로 하나"),
    (lambda b: b.update(textures_lock=None), "textures_lock 이 lock 경로 하나"),
    (lambda b: b.update(textures_license="CC0-1.0"), "textures_license 도 같은 길이\\(2\\)"),
    (lambda b: b.update(textures_license=["CC0-1.0"]), "textures_license 도 같은 길이\\(2\\)"),
])
def test_a_bundle_whose_lock_list_or_per_lock_key_is_malformed_is_refused(edit, phrase):
    b = {"textures_lock": [SCENE_LOCK, CI_LOCK], "textures_license": ["CC0-1.0", "LicenseRef-KETI-CI"]}
    edit(b)
    with pytest.raises(ValueError, match=phrase):
        bundle_per_lock(b, "textures_license")
    with pytest.raises(ValueError, match="lock 마다 하나인 키가 아닙니다"):
        bundle_per_lock({"textures_lock": CI_LOCK}, "files")


def test_two_usds_that_carry_the_ci_ship_it_once(tmp_path):
    """The warehouse and the carried box both show the KETI CI, each bundle with its own row read from the
    one pin: a release with both carries one assets/textures/keti_ci_en.png -- the same bytes serve both
    USDs' ./textures/ references. (Two rows of one name with different hashes are refused --
    tests/generation/test_convert_lerobot.py gives the skin bundle such a row.)"""
    repo, canon = _two_lock_tree(tmp_path)
    usd = b"#usda 1.0\n# c\n"
    (repo / "build/c.usd").write_bytes(usd)
    canon["scene"]["c"] = {"usd_path": "build/c.usd", "redistributed": True, "usd_sha256": _sha(usd),
                           "bundle": {"textures_dir": "build/textures", "textures_lock": CI_LOCK,
                                      "files": [dict(canon["scene"]["b"]["bundle"]["files"][1])]}}
    got = shipped_asset_files(canon, ["build/b.usd", "build/c.usd"], repo)
    assert set(got) == {"assets/b.usd", "assets/c.usd", "assets/textures/wood.jpg",
                        "assets/textures/keti_ci_en.png"}
    assert got["assets/textures/keti_ci_en.png"].asset == "b"          # the first that brings it
    assert set(shipped_asset_files(canon, ["build/c.usd"], repo)) == {"assets/c.usd",
                                                                      "assets/textures/keti_ci_en.png"}
    check_shipped_sources(got, repo)


def test_the_sources_are_checked_by_bytes(tmp_path):
    repo, canon = _tree(tmp_path)
    shipped = shipped_asset_files(canon, ["build/b.usd"], repo)
    (repo / "build/textures/wood.jpg").write_bytes(b"other pixels\n")
    with pytest.raises(ValueError, match="build/textures/wood.jpg 의 해시가 정본과 다릅니다 \\(b\\)"):
        check_shipped_sources(shipped, repo)
    (repo / "build/textures/wood.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="redistributed=true 인데 파일이 없습니다: build/textures/wood.jpg"):
        check_shipped_sources(shipped, repo)


# ── the anchor: what the canon says a scenario's episodes used ──

from generation.assets import required_scene_assets  # noqa: E402

_CANON_SCENE = {"scene": {"human": {"usd_path": "build/human.usd"}},
                "robots": {"bot": {"usd_path": "{NUCLEUS}/bot.usd"}}}


def test_a_scenario_requires_its_declared_assets_and_its_human_models_usd():
    sc = {"scenario_id": "S9", "scene": {"assets": ["build/scene.usd"], "human_model": "human"}}
    assert required_scene_assets(sc, _CANON_SCENE, "bot") == {
        "build/scene.usd", "build/human.usd", "{NUCLEUS}/bot.usd"}


def test_a_scenario_without_a_human_model_requires_its_declared_assets_only():
    # S3a..S5b carry human_model null in the canon; the producer copies that into the episode.
    # The robot is still the episode's own.
    sc = {"scenario_id": "S9", "scene": {"assets": [], "human_model": None}}
    assert required_scene_assets(sc, _CANON_SCENE, "bot") == {"{NUCLEUS}/bot.usd"}


@pytest.mark.parametrize("scenario, robot_model, phrase", [
    ({"scenario_id": "S9"}, "bot", "scene 이 없습니다"),
    ({"scenario_id": "S9", "scene": {"human_model": None}}, "bot",
     "scene.assets 가 비어 있지 않은 문자열의 목록"),
    ({"scenario_id": "S9", "scene": {"assets": "build/scene.usd"}}, "bot", "scene.assets 가"),
    ({"scenario_id": "S9", "scene": {"assets": [""]}}, "bot", "scene.assets 가"),
    ({"scenario_id": "S9", "scene": {"assets": [], "human_model": "nobody"}}, "bot",
     "human_model 'nobody' 이"),
    ({"scenario_id": "S9", "scene": {"assets": [], "human_model": ["human"]}}, "bot",
     "human_model \\['human'\\] 이"),
    # The robot is required through the episode's robot_model.
    ({"scenario_id": "S9", "scene": {"assets": [], "human_model": None}}, "nobody",
     "robot_model 'nobody' 이 meta/assets.json 의 robots 에 없습니다"),
    ({"scenario_id": "S9", "scene": {"assets": [], "human_model": None}}, None,
     "robot_model None 이 meta/assets.json 의 robots 에 없습니다"),
])
def test_a_scenario_that_cannot_say_what_it_used_is_refused(scenario, robot_model, phrase):
    with pytest.raises(ValueError, match=phrase):
        required_scene_assets(scenario, _CANON_SCENE, robot_model)


def test_every_canon_scenario_says_what_its_episodes_used():
    """The producer's side: the converter and C6 refuse a scenario the anchor cannot read, so
    every scenario of the canon must be readable -- including the ones not converted yet.
    The robot is the joint map's robot_model, the key run_s1a records."""
    import json
    from pathlib import Path as P

    import yaml
    repo = P(__file__).resolve().parents[2]
    canon = json.loads((repo / "meta/assets.json").read_text(encoding="utf-8"))
    scenarios = json.loads((repo / "meta/scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    robot = yaml.safe_load((repo / "meta/joint_map/unitree_h1.yaml").read_text(encoding="utf-8"))[
        "robot_model"]
    h1 = canon["robots"][robot]["usd_path"]
    got = {s["scenario_id"]: required_scene_assets(s, canon, robot) for s in scenarios}
    assert got["S1a"] == {"isaaclab://TerrainImporter/plane",
                          canon["scene"]["keti_dummy_ts15066_v1"]["usd_path"], h1}
    assert got["S1d"] >= {canon["scene"]["keti_logistics_aisle_v1"]["usd_path"],
                          canon["scene"]["keti_dummy_ts15066_v1_skin"]["usd_path"], h1}


def test_s1d_declares_the_ground_it_stands_on():
    """S1dEnvCfg is S1aEnvCfg with the aisle added: it stands on the same procedural plane
    (the aisle's floor is visual only -- meta/assets.json keti_logistics_aisle_v1 note), and
    run_s1a records that plane in every S1d sidecar's asset_paths.
    The S1d entry once replaced S1a's scene.assets instead of adding to it, so the anchor never
    asked an S1d episode for its ground."""
    import json
    from pathlib import Path as P
    repo = P(__file__).resolve().parents[2]
    by_sid = {s["scenario_id"]: s for s in json.loads(
        (repo / "meta/scenarios.json").read_text(encoding="utf-8"))["scenarios"]}
    assert set(by_sid["S1a"]["scene"]["assets"]) <= set(by_sid["S1d"]["scene"]["assets"])


# ── the three warehouse scenarios name the scene they are recorded in ──

_NEW_SCENARIOS = ("S1e", "S1f", "S1g")


def _canon_for_the_new_scenarios():
    import json
    from pathlib import Path as P

    import yaml
    repo = P(__file__).resolve().parents[2]
    canon = json.loads((repo / "meta/assets.json").read_text(encoding="utf-8"))
    by_sid = {s["scenario_id"]: s for s in json.loads(
        (repo / "meta/scenarios.json").read_text(encoding="utf-8"))["scenarios"]}
    robot = yaml.safe_load((repo / "meta/joint_map/unitree_h1.yaml").read_text(encoding="utf-8"))[
        "robot_model"]
    return repo, canon, by_sid, robot


def test_the_new_scenarios_name_their_scene():
    """S1e, S1f and S1g carry the box. Every one stands on the plane, in the warehouse, with the vested dummy and
    H1 -- the robot's token is added by the anchor from the episode's robot_model, so it is not in scene.assets."""
    _, canon, by_sid, robot = _canon_for_the_new_scenarios()
    path = {k: e["usd_path"] for k, e in canon["scene"].items()}
    floor, hall = path["isaaclab_terrain_plane"], path["keti_logistics_aisle_v1"]
    box, vest = path["keti_carried_box_v1"], path["keti_dummy_ts15066_v2_vest"]
    h1 = canon["robots"][robot]["usd_path"]
    want = {"S1e": {floor, hall, box, vest, h1}, "S1f": {floor, hall, box, vest, h1},
            "S1g": {floor, hall, box, vest, h1}}
    for sid in _NEW_SCENARIOS:
        assert required_scene_assets(by_sid[sid], canon, robot) == want[sid], sid
        scene = by_sid[sid]["scene"]
        assert scene["human_model"] == "keti_dummy_ts15066_v2_vest", sid
        assert h1 not in scene["assets"], f"{sid}: the robot is not a scene asset"
        assert scene["assets"] == [p for p in (floor, hall, box, vest) if p in want[sid]], sid
        assert isinstance(scene["asset_policy"], str) and scene["asset_policy"].strip(), sid
    # the sites stay where the site plan put them
    assert all("site" in by_sid[sid]["scene"] for sid in _NEW_SCENARIOS)


def test_a_full_new_release_ships_the_stage_the_robot_and_both_licences():
    """A release holding all three scenarios carries the warehouse, the vested dummy, the box, the Stage that
    composes them with H1, and H1 as its own folder with its licence file. The textures are counted from
    the canon's bundles -- every bundle's files, each name once -- and cross-checked against 24 (15 warehouse
    + 8 skin, both CC0, + the one KETI CI the box labels show, the warehouse's and the carried box's alike)."""
    from collections import Counter

    repo, canon, by_sid, robot = _canon_for_the_new_scenarios()
    referenced = set()
    for sid in _NEW_SCENARIOS:
        referenced |= required_scene_assets(by_sid[sid], canon, robot)
    files = shipped_asset_files(canon, referenced, repo)

    stage = canon["scene"]["keti_hri_stage_v1"]
    assert files["assets/keti_hri_stage_v1.usd"].sha256 == stage["usd_sha256"]
    for key in ("keti_logistics_aisle_v1", "keti_dummy_ts15066_v2_vest", "keti_carried_box_v1"):
        shipped = files[f"assets/{canon['scene'][key]['usd_path'].rsplit('/', 1)[-1]}"]
        assert (shipped.asset, shipped.sha256) == (key, canon["scene"][key]["usd_sha256"])
    h1 = {r["file"]: r["sha256"] for r in canon["robots"][robot]["ship"]["files"]}
    assert sorted(h1) == ["LICENSE.txt", "h1.usd", "instanceable_meshes.usd"]
    for name, sha in h1.items():
        assert files[f"assets/{canon['robots'][robot]['ship']['dir']}/{name}"].sha256 == sha, name
    # nothing else ships: no bare dummy, no plane
    assert all(f.asset in {robot, "keti_hri_stage_v1", "keti_logistics_aisle_v1",
                           "keti_dummy_ts15066_v2_vest", "keti_carried_box_v1"} for f in files.values())

    shipped = {n.removeprefix("assets/textures/"): f for n, f in files.items()
               if n.startswith("assets/textures/")}
    rows = {r["file"]: r for key in ("keti_logistics_aisle_v1", "keti_dummy_ts15066_v2_vest",
                                     "keti_carried_box_v1")
            for r in canon["scene"][key]["bundle"]["files"]}
    assert set(shipped) == set(rows)
    assert all(shipped[n].sha256 == r["sha256"] for n, r in rows.items())
    assert Counter(r["license"] for r in rows.values()) == {"CC0-1.0": 23, "LicenseRef-KETI-CI": 1}
    assert len(shipped) == 24


def test_the_ci_ships_with_the_warehouse_whether_or_not_a_box_is_carried():
    """The warehouse's box labels show the KETI CI, so a release without the carried box would carry it with the
    warehouse, and a release of S1e, S1f or S1g -- the warehouse and the carried box, both bundles naming the CI
    pin -- carries one copy of it, the bytes both bundles pin."""
    repo, canon, by_sid, robot = _canon_for_the_new_scenarios()
    hall, box = canon["scene"]["keti_logistics_aisle_v1"], canon["scene"]["keti_carried_box_v1"]
    ci_lock = "generation/scene/keti_ci.json"
    assert ci_lock in bundle_per_lock(hall["bundle"], "textures_lock")
    assert bundle_per_lock(box["bundle"], "textures_lock") == [ci_lock]
    ci_rows = [r for r in hall["bundle"]["files"] if r["license"] == "LicenseRef-KETI-CI"]
    assert ci_rows == box["bundle"]["files"] and len(ci_rows) == 1
    ci = f"assets/textures/{ci_rows[0]['file']}"
    for sid in _NEW_SCENARIOS:
        files = shipped_asset_files(canon, required_scene_assets(by_sid[sid], canon, robot), repo)
        assert [k for k in files if k.endswith("/" + ci_rows[0]["file"])] == [ci], sid
        assert files[ci].sha256 == ci_rows[0]["sha256"], sid
        assert files[ci].asset == "keti_logistics_aisle_v1", sid    # the first in the canon that brings it
        assert "assets/keti_carried_box_v1.usd" in files, sid


def test_a_bundle_of_null_is_refused_not_shipped_without_textures(tmp_path):
    """A bundle key says the USD has textures; null used to ship it without them while C6
    refused the same entry -- the producer and its checker disagreed on a malformed canon."""
    repo, canon = _tree(tmp_path)
    canon["scene"]["b"]["bundle"] = None
    with pytest.raises(ValueError, match="파일 목록\\(files\\)이 없습니다"):
        shipped_asset_files(canon, ["build/b.usd"], repo)


# ── a robot that ships as its own folder ──

_ROBOT_FILES = {"bot.usd": b"#usda 1.0\n# bot\n", "meshes.usd": b"#usda 1.0\n# meshes the bot references\n",
                "LICENSE.txt": b"BSD 3-Clause License\n\nCopyright (c) the bot makers"}
_BOT_USD = "{ISAACLAB_NUCLEUS_DIR}/Robots/Bot/bot.usd"


def _robot_tree(tmp_path):
    """A repository whose canon lets its robot ship as a folder, as the H1 does: the USD (named by
    the nucleus token, the key episodes join on), the layer that USD references beside it and the
    licence file, fetched under build/bot/. Each file is a row; the USD's row is the robot's pin."""
    repo = tmp_path / "robotrepo"
    for name, data in _ROBOT_FILES.items():
        (repo / "build/bot").mkdir(parents=True, exist_ok=True)
        (repo / "build/bot" / name).write_bytes(data)
    rows = [{"file": n, "sha256": _sha(d), "license": "BSD-3-Clause",
             "url": f"{{ISAACLAB_NUCLEUS_DIR}}/Robots/Bot/{n}"} for n, d in _ROBOT_FILES.items()]
    entry = {"usd_path": _BOT_USD, "license": "BSD-3-Clause", "redistributed": True,
             "usd_sha256": _sha(_ROBOT_FILES["bot.usd"]),
             "ship": {"dir": "bot", "source_dir": "build/bot", "license_file": "LICENSE.txt", "files": rows}}
    return repo, {"robots": {"bot": entry}, "scene": {}}


def test_a_redistributed_robot_ships_its_folder(tmp_path):
    """The robot's USD is a nucleus token, not a repository path: what ships is its ship block --
    every row at assets/<dir>/<file>, from the fetched copy, held to the row's hash."""
    from generation.assets import ROBOT_SHIP_KEYS, robot_ship_files

    assert ROBOT_SHIP_KEYS == ("dir", "source_dir", "license_file", "files")
    repo, canon = _robot_tree(tmp_path)
    got = shipped_asset_files(canon, [_BOT_USD], repo)
    assert {rel: (f.src, f.sha256, f.asset) for rel, f in got.items()} == {
        f"assets/bot/{n}": (repo / "build/bot" / n, _sha(d), "bot") for n, d in _ROBOT_FILES.items()}
    assert robot_ship_files("bot", canon["robots"]["bot"], repo) == got
    check_shipped_sources(got, repo)
    # Not used by the batch: nothing. Kept back: nothing, ship block or not.
    assert shipped_asset_files(canon, [], repo) == {}
    canon["robots"]["bot"]["redistributed"] = False
    assert shipped_asset_files(canon, [_BOT_USD], repo) == {}


def test_a_robot_folder_and_a_scene_usd_ship_side_by_side(tmp_path):
    """The folder keeps the robot's files apart from the scene's: a USD of the robot's name under
    assets/ is another file, and both ship."""
    repo, canon = _robot_tree(tmp_path)
    (repo / "build/scene").mkdir(parents=True)
    (repo / "build/scene/bot.usd").write_bytes(b"#usda 1.0\n# a scene file of that name\n")
    canon["scene"]["s"] = {"usd_path": "build/scene/bot.usd", "redistributed": True,
                           "usd_sha256": _sha(b"#usda 1.0\n# a scene file of that name\n")}
    got = shipped_asset_files(canon, [_BOT_USD, "build/scene/bot.usd"], repo)
    assert sorted(got) == ["assets/bot.usd", "assets/bot/LICENSE.txt", "assets/bot/bot.usd",
                           "assets/bot/meshes.usd"]


@pytest.mark.parametrize("edit, phrase", [
    (lambda s: s.update(license_file="COPYING"), "license_file 'COPYING' 의 행이 ship.files 에 없습니다"),
    (lambda s: s["files"].pop(), "license_file 'LICENSE.txt' 의 행이 ship.files 에 없습니다"),
    (lambda s: s.pop("license_file"), "ship 에 \\['license_file'\\] 키가 없습니다"),
], ids=["another name", "no licence row", "no license_file"])
def test_a_robot_without_its_licence_file_is_refused(tmp_path, edit, phrase):
    # BSD-3 asks that a redistribution carries the notice and the licence text: a folder without
    # its licence file does not ship.
    repo, canon = _robot_tree(tmp_path)
    edit(canon["robots"]["bot"]["ship"])
    with pytest.raises(ValueError, match=f"bot: {phrase}"):
        shipped_asset_files(canon, [_BOT_USD], repo)


def test_a_robot_whose_usd_row_is_not_its_pin_is_refused(tmp_path):
    """usd_sha256 is the robot the joint map was read from and the episodes ran; the folder's USD row
    pins the bytes that ship. Two hashes for one file name two robots."""
    repo, canon = _robot_tree(tmp_path)
    entry = canon["robots"]["bot"]
    entry["usd_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bot: ship.files 의 bot.usd 행의 sha256 이 usd_sha256 과 다릅니다"):
        shipped_asset_files(canon, [_BOT_USD], repo)
    entry["usd_sha256"] = _sha(_ROBOT_FILES["bot.usd"])
    entry["ship"]["files"] = [r for r in entry["ship"]["files"] if r["file"] != "bot.usd"]
    with pytest.raises(ValueError, match="bot: ship.files 에 로봇의 USD bot.usd 의 행이 없습니다"):
        shipped_asset_files(canon, [_BOT_USD], repo)
    entry["ship"]["files"] = _robot_tree(tmp_path / "again")[1]["robots"]["bot"]["ship"]["files"]
    entry["usd_sha256"] = "abc"
    with pytest.raises(ValueError, match="bot: 재배포하는데 정본에 usd_sha256 이 없거나"):
        shipped_asset_files(canon, [_BOT_USD], repo)


@pytest.mark.parametrize("edit, phrase", [
    (lambda e: e.__setitem__("ship", ["bot"]), "ship 이 객체가 아닙니다"),
    (lambda e: e["ship"].update(note="x"), "ship 에 모르는 키 \\['note'\\]"),
    (lambda e: e["ship"].update(dir="../bot"), "ship.dir 이 assets/ 아래의 폴더 이름이 아닙니다"),
    (lambda e: e["ship"].update(dir="a/b"), "ship.dir 이 assets/ 아래의 폴더 이름이 아닙니다"),
    (lambda e: e["ship"].update(dir="textures"), "ship.dir 이 assets/ 아래의 폴더 이름이 아닙니다"),
    (lambda e: e["ship"].update(files=[]), "ship.files 가 비어 있지 않은 목록이 아닙니다"),
    (lambda e: e["ship"]["files"][1].update(file="../meshes.usd"), "ship.files\\[1\\] — file 이 경로가 아닌"),
    (lambda e: e["ship"]["files"].append(dict(e["ship"]["files"][1])), "ship.files 에 같은 이름이 둘"),
    (lambda e: e.__setitem__("bundle", {"files": []}), "ship 과 bundle 을 함께"),
    (lambda e: e["ship"].update(source_dir="../elsewhere"), "bot.ship.source_dir 가 저장소 밖"),
    (lambda e: e["ship"].update(source_dir="{ISAACLAB_NUCLEUS_DIR}/Robots/Bot"),
     "bot.ship.source_dir 가 저장소 안의 상대 경로가 아닙니다"),
])
def test_a_ship_block_that_cannot_say_what_ships_is_refused(tmp_path, edit, phrase):
    repo, canon = _robot_tree(tmp_path)
    edit(canon["robots"]["bot"])
    with pytest.raises(ValueError, match=phrase):
        shipped_asset_files(canon, [_BOT_USD], repo)


def test_two_robots_of_one_folder_name_are_refused(tmp_path):
    """A folder is a robot's own: a second robot under the same folder name would put its files in the
    first one's folder, the same bytes or not."""
    import copy
    repo, canon = _robot_tree(tmp_path)
    twin = copy.deepcopy(canon["robots"]["bot"])
    twin["usd_path"] = "{ISAACLAB_NUCLEUS_DIR}/Robots/Twin/bot.usd"
    canon["robots"]["twin"] = twin
    with pytest.raises(ValueError, match="두 자산이 같은 이름으로 실립니다: assets/bot/bot.usd \\(bot · twin\\)"):
        shipped_asset_files(canon, [_BOT_USD, twin["usd_path"]], repo)
    twin["ship"]["dir"] = "twin"
    assert len(shipped_asset_files(canon, [_BOT_USD, twin["usd_path"]], repo)) == 6


def test_a_robot_folder_is_checked_by_bytes(tmp_path):
    repo, canon = _robot_tree(tmp_path)
    shipped = shipped_asset_files(canon, [_BOT_USD], repo)
    (repo / "build/bot/LICENSE.txt").unlink()
    with pytest.raises(FileNotFoundError, match="build/bot/LICENSE.txt \\(bot\\)"):
        check_shipped_sources(shipped, repo)
    (repo / "build/bot/LICENSE.txt").write_bytes(b"another licence\n")
    with pytest.raises(ValueError, match="build/bot/LICENSE.txt 의 해시가 정본과 다릅니다 \\(bot\\)"):
        check_shipped_sources(shipped, repo)
