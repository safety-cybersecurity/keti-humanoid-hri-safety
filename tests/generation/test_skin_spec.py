"""generation/dummy/skin/skin_spec.yaml — 시각 메시 정본이 치수 정본을 가리키고 거부할 것을 거부하는가.

Blender 는 CI 에 없다. 여기서는 명세와 드라이버 쪽 검증만 본다 — 실제 생성은
make_skin.py 를 Blender 가 있는 기계에서 돌려 skin_manifest.json 으로 확인한다.
"""

import copy
from pathlib import Path

import pytest
import yaml

from generation.dummy.skin import make_skin

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "generation/dummy/skin/skin_spec.yaml"


@pytest.fixture(scope="module")
def spec():
    return make_skin.load_spec(SPEC)


def test_skin_spec_points_at_the_anthropometry_canon(spec):
    anthro = yaml.safe_load((REPO / spec["stature_from"]).read_text(encoding="utf-8"))
    assert anthro["stature"] > 1.0 and spec["clothes"], "spec must name clothes and a real anthropometry file"
    assert spec["source"]["license_of_output"] == "CC0-1.0"
    # the same file build_dummy.py reads, not a copy of it
    assert (REPO / spec["stature_from"]).resolve() == make_skin.ANTHRO.resolve()


def test_skin_spec_names_assets_with_subdir_label_and_type(spec):
    for a in spec["clothes"]:
        assert set(a) >= {"subdir", "label", "type"}, a
        assert a["type"] in make_skin.ASSET_TYPES, a
    assert spec["skin"]["subdir"] == "skins" and spec["skin"]["label"]
    # only this material mode carries textures through the exporter
    assert spec["material_type"] == "GAMEENGINE"


def test_skin_spec_exports_in_the_dummy_frame(spec):
    # anthropometry.yaml: origin at the sole centre, +z up, +x forward -> +y is the left side
    assert spec["frame"] == {"front": "+x", "up": "+z", "left": "+y"}
    assert spec["export"]["yaw_deg"] == 90       # MakeHuman faces -y in Blender


def test_the_arms_are_posed_down_on_both_sides(spec):
    """MakeHuman's A pose leaves the hands 0.4 m from the arm colliders (2026-09-22); the
    spec has to bring every arm bone down, and the two sides have to mirror each other."""
    aims = {a["bone"]: a["direction"] for a in spec["pose"]["aim"]}
    for bone, d in aims.items():
        assert d[2] < 0, f"{bone} does not point down: {d}"
        other = bone[:-1] + ("R" if bone.endswith("L") else "L")
        mirrored = [None if d[0] is None else -d[0], d[1], d[2]]
        assert other in aims and aims[other] == mirrored, f"{bone} has no mirrored {other}"
    assert {b.split(".")[0] for b in aims} >= {"upperarm01", "lowerarm01", "upperleg01", "lowerleg01"}
    # the arms are aimed absolutely (they must hang in the lateral plane); the legs keep
    # their fore-aft tilt and only stand up laterally
    for bone, d in aims.items():
        assert (d[1] is None) == bone.startswith(("upperleg", "lowerleg")), bone


def test_height_is_searched_not_set(spec):
    assert "height" not in spec["macro"]
    hs = spec["height_search"]
    assert 0.0 <= hs["lo"] < hs["hi"] <= 1.0 and hs["max_iter"] >= 4
    assert 0 < spec["stature_tolerance"] <= 0.05


def _mutated(spec, path, value):
    s = copy.deepcopy(spec)
    node = s
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return s


@pytest.mark.parametrize("path,value,why", [
    (("macro", "gender"), 1.5, "macro"),
    (("macro", "height"), 0.5, "height"),
    (("material_type",), "MAKESKIN", "GAMEENGINE"),
    (("stature_from",), "generation/dummy/skin/skin_spec.yaml", "anthropometry"),
    (("stature_tolerance",), 0.5, "tolerance"),
    (("clothes",), [], "clothes"),
    (("clothes", 0, "type"), "Hat", "type"),
    (("frame", "front"), "-y", "frame"),
    (("source", "license_of_output"), "CC-BY-4.0", "CC0"),
    (("height_search", "hi"), 0.2, "height_search"),
    (("pose", "rig"), "", "rig"),
    (("pose", "aim"), [], "aim"),
    (("pose", "aim", 0, "direction"), [0, 0, 0], "zero"),
    (("pose", "aim", 0, "direction"), [None, None, None], "zero"),
    (("pose", "aim", 0, "direction"), [0, "down", -1], "direction"),
    (("fit", "max_lateral_offset_m"), 1.0, "lateral"),
])
def test_validate_spec_refuses_what_the_generator_could_not_honour(spec, path, value, why):
    with pytest.raises(ValueError, match=why):
        make_skin.validate_spec(_mutated(spec, path, value))


def test_validate_spec_accepts_the_canon(spec):
    assert make_skin.validate_spec(spec) is spec


def test_the_resolved_spec_carries_the_target_stature(spec):
    resolved = make_skin.resolve(spec)
    anthro = yaml.safe_load(make_skin.ANTHRO.read_text(encoding="utf-8"))
    assert resolved["stature_target"] == anthro["stature"]
    assert resolved["anthropometry_version"] == str(anthro["anthropometry_version"])
    assert resolved["spec_sha256"] and len(resolved["spec_sha256"]) == 64


# ---- 팩 대조 — 자산 디렉터리에 있다는 것은 출처의 증명이 아니다 -------------------------------

def test_verify_against_pack_matches_bytes_not_names(tmp_path):
    import hashlib
    import zipfile
    pack = tmp_path / "pack.zip"
    with zipfile.ZipFile(pack, "w") as z:
        z.writestr("clothes/suit/suit.obj", b"v 0 0 0\n")
        z.writestr("clothes/suit/suit.mhclo", b"obj_file suit.obj\n")
    sha = hashlib.sha256(pack.read_bytes()).hexdigest()
    ok = {"name": "clothes/suit/suit.obj", "sha256": hashlib.sha256(b"v 0 0 0\n").hexdigest(),
          "bytes": 8, "source": "pack:clothes/suit/suit.obj", "license": "CC0-1.0"}
    assert make_skin.verify_against_pack(pack, sha, [ok]) == []
    swapped = dict(ok, sha256=hashlib.sha256(b"v 1 1 1\n").hexdigest())
    assert any("bytes differ" in p for p in make_skin.verify_against_pack(pack, sha, [swapped]))
    missing = dict(ok, name="clothes/suit/hat.obj", source="pack:clothes/suit/hat.obj")
    assert any("not in the pack" in p for p in make_skin.verify_against_pack(pack, sha, [missing]))
    foreign = dict(ok, source="local:clothes/suit/suit.obj")
    assert any("not a pack member" in p for p in make_skin.verify_against_pack(pack, sha, [foreign]))
    # a different zip than the pinned one is refused before any row is looked at
    assert make_skin.verify_against_pack(pack, "0" * 64, [ok])[0].startswith("pack.zip: sha256")


def test_mhclo_refs_name_the_geometry_and_material_beside_the_file(tmp_path):
    d = tmp_path / "clothes" / "suit"
    d.mkdir(parents=True)
    (d / "suit.obj").write_text("v 0 0 0\n")
    (d / "suit.mhmat").write_text("name suit\n")
    (d / "suit.mhclo").write_text("# header\nname suit\nobj_file suit.obj\nmaterial suit.mhmat\n")
    refs = make_skin.mhclo_refs(d / "suit.mhclo")
    assert [r.name for r in refs] == ["suit.obj", "suit.mhmat"]
    (d / "broken.mhclo").write_text("obj_file gone.obj\n")
    with pytest.raises(ValueError, match="does not exist"):
        make_skin.mhclo_refs(d / "broken.mhclo")
    (d / "noobj.mhclo").write_text("material suit.mhmat\n")
    with pytest.raises(ValueError, match="no obj_file"):
        make_skin.mhclo_refs(d / "noobj.mhclo")


def test_texture_rows_and_pack_rows_are_told_apart_by_name():
    assert make_skin.is_texture_row({"name": "brown_eye.png"})
    assert not make_skin.is_texture_row({"name": "eyes/low-poly/low-poly.obj"})
