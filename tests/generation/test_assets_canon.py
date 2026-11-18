"""meta/assets.json 이 스키마 §5 의 scene_assets 계약을 만족하는지."""

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "meta" / "assets.json"

REQUIRED_ASSET_KEYS = {"usd_path", "license", "redistributed"}
# Each lock of a bundle carries one licence: CC0 textures, or the KETI CI the box labels show (the
# distributor's own logo -- shipped under its own terms, not under the dataset's CC BY 4.0).
BUNDLE_LICENSES = ("CC0-1.0", "LicenseRef-KETI-CI")


@pytest.fixture(scope="module")
def canon():
    return json.loads(ASSETS.read_text(encoding="utf-8"))


def test_every_asset_declares_path_license_and_redistribution(canon):
    entries = list(canon["robots"].values()) + list(canon["scene"].values())
    assert entries, "자산이 하나도 없음"
    for e in entries:
        absent = sorted(REQUIRED_ASSET_KEYS - set(e))
        assert not absent, f"필수 키 누락: {absent}"
        assert isinstance(e["redistributed"], bool)
        assert e["license"].strip()


def test_robot_entries_carry_the_embodiment_enum(canon):
    # 스키마 §5 robot_embodiment 의 허용값. 에피소드 메타가 여기서 온다.
    for model, e in canon["robots"].items():
        assert e["embodiment"] in {"biped", "wheeled"}, model


def test_the_h1_entry_matches_the_joint_map_provenance(canon):
    # 관절 정본이 어느 사본에서 나왔는지와 자산 정본이 어긋나면, 배포 메타가
    # 실제로 돌린 것과 다른 자산을 가리키게 된다.
    jm = yaml.safe_load(
        (REPO_ROOT / "meta" / "joint_map" / "unitree_h1.yaml").read_text(encoding="utf-8")
    )
    assert canon["robots"]["unitree_h1"]["usd_sha256"] == jm["provenance"]["usd_sha256"]


def test_usd_paths_are_unique_across_the_whole_canon(canon):
    # 변환기는 경로로 조인한다. 같은 경로가 두 항목에 있으면 어느 라이선스가
    # 붙는지가 사전 순서에 달리게 된다.
    paths = [e["usd_path"] for e in list(canon["robots"].values()) + list(canon["scene"].values())]
    assert len(set(paths)) == len(paths), f"중복 경로: {paths}"


def test_no_entry_names_an_asset_nothing_produces(canon):
    """정본이 유령 자산을 가리키지 못하게 한다.

    없는 파일을 적으면 배포 메타가 존재하지 않는 자산의 라이선스를 주장하게 된다.
    빌드 산출물(*.usd 는 .gitignore 대상이라 CI 에는 없다)은 파일 대신 그것을
    만드는 스크립트가 있는지로 확인한다 — 유령과 산출물을 가르는 것이 이 검사의
    목적이지 체크아웃 상태를 재는 것이 아니다.
    """
    entries = list(canon["robots"].items()) + list(canon["scene"].items())
    for key, e in entries:
        path = e["usd_path"]
        if path.startswith(("isaaclab://", "{", "http")):
            continue                                    # 외부 자산 — 경로 규약만 본다
        builder = e.get("built_by")
        if builder:
            assert (REPO_ROOT / builder).exists(), f"{key}: 빌더 없음 {builder}"
        else:
            assert (REPO_ROOT / path).exists(), f"{key}: 파일도 빌더도 없음 {path}"


def test_every_scenario_scene_asset_joins_to_the_canon(canon):
    """카탈로그가 선언한 씬 자산이 전부 자산 정본에 있어야 한다.

    S1a 가 존재하지 않는 `keti/scene_flat_primitive.usd` 를 선언하고 있던 적이
    있고, 아무도 대조하지 않아 드러나지 않았습니다.
    """
    import json
    scenarios = json.loads(
        (REPO_ROOT / "meta" / "scenarios.json").read_text(encoding="utf-8")
    )
    known = {e["usd_path"] for e in list(canon["robots"].values())
             + list(canon["scene"].values())}
    for sc in scenarios["scenarios"]:
        for path in sc.get("scene", {}).get("assets", []):
            assert path in known, f"{sc['scenario_id']}: 자산 정본에 없음 {path}"


def test_every_scenario_that_can_be_converted_has_its_required_meta(canon):
    """스키마 §5 필수 에피소드 필드 중 시나리오가 주는 것들.

    지금은 S1a · S1d · S1e · S1f · S1g 가 완전합니다. 나머지 11개는 변환 전에 채워야 하며, 이 시험은
    '무엇이 비었는지' 를 세어 보여 줍니다 — 채워지면 목록이 줄어듭니다.
    """
    import json
    scenarios = json.loads(
        (REPO_ROOT / "meta" / "scenarios.json").read_text(encoding="utf-8")
    )
    need = {"domain", "task_archetype", "hazard_class", "safety_function"}
    incomplete = sorted(sc["scenario_id"] for sc in scenarios["scenarios"]
                        if not need <= set(sc))
    assert "S1a" not in incomplete, "변환 대상 시나리오가 불완전합니다"
    assert incomplete == ["S1b", "S1c", "S2a", "S2b", "S3a", "S3b",
                          "S4a", "S4b", "S5a", "S5b", "S6a"], incomplete


# ---- 저장소가 만드는 씬 자산 (더미 · S1d 물류 씬) ---------------------------------

def _repo_built_scene_assets(canon):
    return {k: e for k, e in canon["scene"].items()
            if str(e.get("built_by", "")).startswith("generation/")}


def test_every_repo_built_scene_asset_pins_its_hash_and_a_known_licence(canon):
    """빌더가 만드는 자산은 git 이 추적하지 않으므로 해시만이 「어느 자산인가」를 말한다."""
    built = _repo_built_scene_assets(canon)
    assert built, "저장소가 만드는 씬 자산이 하나도 없음"
    for name, e in built.items():
        sha = str(e.get("usd_sha256") or "")
        assert len(sha) == 64 and int(sha, 16), f"{name}: usd_sha256 이 없거나 형식이 아님"
        assert e["license"] in ("CC-BY-4.0", "CC0-1.0"), name


def test_notice_names_every_repo_built_scene_asset(canon):
    """NOTICE 는 배포본에 동봉되는 고지다. 정본에 있는 자체 제작 자산이 거기 없으면
    이용자는 그 자산의 출처를 알 길이 없다."""
    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    for name in _repo_built_scene_assets(canon):
        assert name in notice, f"NOTICE 가 {name} 을 언급하지 않음"


def test_notice_names_every_redistributable_usd_by_the_name_c6_reads(canon):
    """C6 요구: 재배포 자산마다 **USD 파일 이름**(`Path(usd_path).name`)이 NOTICE 에 있어야
    한다(validate_dataset.c6_asset_license). 바로 위 시험은 정본의 **키**를 찾는다 — 다른
    문자열이다. S1d 의 두 자산을 재배포로 바꾸자 NOTICE 가 키로만 적고 있어 실제 배포본의 C6
    가 ERROR 2 를 냈고, 픽스처의 NOTICE 는 합성이라 CI 는 초록이었다.
    배포본에 실릴 수 있는 것은 정본의 재배포 항목 전부이므로 전부 본다."""
    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    names = [Path(e["usd_path"]).name for g in ("robots", "scene")
             for e in (canon.get(g) or {}).values() if e.get("redistributed")]
    assert names, "재배포 자산이 하나도 없음"
    missing = [n for n in names if n not in notice]
    assert not missing, f"NOTICE 가 재배포 USD 의 파일 이름을 적지 않음(C6 가 ERROR): {missing}"


def test_a_texture_bundle_pins_the_lock_it_was_built_from(canon):
    """텍스처 묶음의 정본은 lock 이다(묶음이 lock 여럿을 적으면 그 하나하나). 정본이 고정한 lock
    해시가 실제 lock 과 다르면 「어느 텍스처를 실었나」가 두 문서에서 갈린다."""
    import hashlib

    from generation.assets import bundle_per_lock

    bundles = [(k, e["bundle"]) for k, e in canon["scene"].items() if isinstance(e.get("bundle"), dict)]
    assert bundles, "텍스처 묶음을 가진 씬 자산이 없음"
    for name, b in bundles:
        for rel, pinned, licence in zip(*(bundle_per_lock(b, k) for k in
                                          ("textures_lock", "textures_lock_sha256", "textures_license"))):
            lock = REPO_ROOT / rel
            assert lock.is_file(), f"{name}: lock 없음 {rel}"
            got = hashlib.sha256(lock.read_bytes()).hexdigest()
            assert pinned == got, (
                f"{name}: {rel} 이 바뀌었습니다 — 자산을 다시 빌드하고 정본의 해시 둘을 갱신하십시오")
            assert licence in BUNDLE_LICENSES, (name, rel)


def test_every_texture_bundle_lists_what_its_lock_pins(canon):
    """bundle.files 는 배포본으로 가는 출처 기록이다 — 배포본에는 lock 이 없다. lock 은 빌더와
    내려받기가 대조하는 정본이다. 둘은 한 진술이어야 하므로 행 하나하나가 같아야 한다.

    변환기도 같은 대조로 멈추지만(generation.assets.shipped_asset_files) 그것은 그 자산을
    싣는 배치를 변환할 때뿐이다 — 여기서는 CI 가 늘 본다. 실패하면 기대 행을 보여 주므로
    그대로 옮기면 된다. 묶음이 lock 여럿을 적으면 행은 lock 의 순서대로 이어진다. 라이선스는
    lock 마다 하나이고(CC0-1.0 또는 LicenseRef-KETI-CI), 그 lock 의 행마다 묶음이 그 lock 에
    적은 textures_license 와 같아야 한다.
    """
    from generation.assets import LOCK_READERS, bundle_per_lock, texture_rows_from_lock

    bundles = {k: e for k, e in canon["scene"].items() if isinstance(e.get("bundle"), dict)}
    assert bundles, "텍스처 묶음을 가진 씬 자산이 없음"
    for name, e in bundles.items():
        want = texture_rows_from_lock(e, REPO_ROOT)
        assert want, f"{name}: lock 에 텍스처가 한 줄도 없음"
        locks = bundle_per_lock(e["bundle"], "textures_lock")
        assert e["bundle"].get("files") == want, (
            f"{name}: meta/assets.json 의 bundle.files 가 {' · '.join(locks)} 과 다릅니다 — "
            "lock 이 의도대로 바뀌었다면 아래 행으로 바꾸십시오:\n"
            + json.dumps(want, ensure_ascii=False, indent=2))
        joined = []
        for rel, licence in zip(locks, bundle_per_lock(e["bundle"], "textures_license")):
            rows = LOCK_READERS[rel](REPO_ROOT, e, rel)
            assert rows, f"{name}: {rel} 에 텍스처가 한 줄도 없음"
            assert {r["license"] for r in rows} == {licence}, (name, rel)
            assert licence in BUNDLE_LICENSES, (name, rel)
            joined += rows
        assert joined == want, name


def test_a_dummy_that_carries_a_gait_table_credits_its_source(canon):
    """The jointed dummy's USD carries a walking table derived from 100STYLE (CC BY 4.0), so the credit,
    the licence link and what was changed go wherever the asset goes. The gait build copies the gait
    spec's source block into the table (build_gait.py) and the dummy build writes it into the USD; the
    canon entry carries the same block, and a redistributed one is credited in NOTICE as well."""
    source = yaml.safe_load(
        (REPO_ROOT / "generation/dummy/gait/gait_spec.yaml").read_text(encoding="utf-8"))["source"]
    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    gaited = {k: e for k, e in canon["scene"].items() if "--gait" in str(e.get("build_args", ""))}
    assert gaited, "걸음 표를 싣는 자산이 정본에 없습니다"
    assert any(e["redistributed"] for e in gaited.values()), "걸음 표를 싣는 재배포 자산이 없습니다"
    for name, e in gaited.items():
        assert e.get("gait_source") == source, f"{name}: gait_source 가 걸음 명세의 source 와 다릅니다"
        if e["redistributed"]:
            for key in ("credit", "license_url", "page"):
                assert source[key] in notice, f"NOTICE 가 {name} 의 걸음 표 출처({key})를 적지 않습니다"


def test_the_built_scene_matches_the_hash_the_canon_pins(canon):
    """산출물이 있는 기계에서만 — 빌드가 있는데 정본의 해시와 다르면 정본이 거짓이다.

    CI 에는 *.usd 가 없으므로 (.gitignore) 전부 skip 된다. 더미와 같은 취급."""
    import hashlib
    built = _repo_built_scene_assets(canon)
    present = {k: e for k, e in built.items() if (REPO_ROOT / e["usd_path"]).is_file()}
    if not present:
        pytest.skip("빌드된 씬 자산이 없습니다 (*.usd 는 추적하지 않음)")
    for name, e in present.items():
        got = hashlib.sha256((REPO_ROOT / e["usd_path"]).read_bytes()).hexdigest()
        assert got == e["usd_sha256"], (
            f"{name}: 빌드된 USD 의 해시가 정본과 다릅니다 — 정본(yaml·lock)이 바뀌었으면 "
            "usd_sha256 을 갱신하고, 아니면 빌드가 결정적이지 않은 것입니다")


# ---- the robot that ships as its own folder ----------------------------------------------------

def _shipping_robots(canon):
    return {k: e for k, e in canon["robots"].items() if e.get("redistributed") and "ship" in e}


def _notice_section(number: int) -> str:
    """NOTICE's section ``number`` -- the lines below its rule, up to the next section's heading -- without
    the blank lines around it."""
    lines = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{number}. "))
    assert set(lines[start + 1]) == {"-"}, lines[start + 1]
    end = next(i for i in range(start + 2, len(lines)) if lines[i].startswith(f"{number + 1}. "))
    return "\n".join(lines[start + 2:end]).strip("\n")


def test_the_robot_rows_pin_what_the_canon_already_knew(canon):
    """The H1 ships as its own folder. The row of its USD is the canon's usd_sha256 --
    the copy the joint map was extracted from and the link boxes were measured on -- so the robot that ships
    is the robot the episodes ran. The folder holds that USD, the layer it references beside it and the
    licence file, each under the robot's own licence; it has no usd_checked (its masses are real: the
    folder is held by its pins, the mass scan reads the USDs directly under assets/)."""
    from generation.assets import robot_ship_files

    assert set(_shipping_robots(canon)) == {"unitree_h1"}
    e = canon["robots"]["unitree_h1"]
    joint_map = yaml.safe_load((REPO_ROOT / "meta/joint_map/unitree_h1.yaml").read_text(encoding="utf-8"))
    boxes = json.loads((REPO_ROOT / "generation/robots/unitree_h1_link_boxes.json").read_text(encoding="utf-8"))
    rows = {r["file"]: r for r in e["ship"]["files"]}
    assert rows[Path(e["usd_path"]).name]["sha256"] == e["usd_sha256"] \
        == joint_map["provenance"]["usd_sha256"] == boxes["usd_sha256"]
    assert e["ship"]["license_file"] in rows
    assert {r["license"] for r in rows.values()} == {e["license"]}
    assert e["redistributed_note"] and "usd_checked" not in e
    got = robot_ship_files("unitree_h1", e, REPO_ROOT)
    assert sorted(got) == sorted(f"assets/{e['ship']['dir']}/{name}" for name in rows) and len(got) == 3


def test_notice_names_every_file_a_robot_folder_ships(canon):
    """The folder ships under the robot's licence, not the dataset's: NOTICE names where it lies and each
    file in it."""
    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    robots = _shipping_robots(canon)
    assert robots, "폴더째 싣는 로봇이 없습니다"
    for name, e in robots.items():
        assert f"assets/{e['ship']['dir']}/" in notice, name
        for row in e["ship"]["files"]:
            assert row["file"] in notice, (name, row["file"])


def test_notice_carries_the_licence_the_robot_ships(canon):
    """BSD-3 asks a redistribution to reproduce the notice, the conditions and the disclaimer. The folder carries
    the licence file byte for byte, and NOTICE section 4 carries the same text -- line for line, but for the line
    break NOTICE ends the text with (the file has none). Only where the robot was fetched
    (generation/robots/fetch_robot.py): the fetched folder is not tracked."""
    import hashlib

    robots = _shipping_robots(canon)
    assert robots, "폴더째 싣는 로봇이 없습니다"
    for name, e in robots.items():
        ship = e["ship"]
        licence = REPO_ROOT / ship["source_dir"] / ship["license_file"]
        if not licence.is_file():
            pytest.skip(f"{name}: 로봇을 받아 두지 않았습니다 ({ship['source_dir']}) — fetch_robot.py")
        pinned = next(r["sha256"] for r in ship["files"] if r["file"] == ship["license_file"])
        assert hashlib.sha256(licence.read_bytes()).hexdigest() == pinned, "받아 둔 라이선스 파일이 고정한 것이 아닙니다"
        assert _notice_section(4) == licence.read_text(encoding="utf-8").rstrip("\n"), (
            f"NOTICE §4 의 전문이 {name} 이 싣는 {ship['license_file']} 과 다릅니다")
