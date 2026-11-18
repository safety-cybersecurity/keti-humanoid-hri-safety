

def test_the_effective_mass_is_not_used_as_rigid_body_mass():
    """표 A.3 의 유효질량은 두 물체 충돌 모형의 파라미터입니다 — 강체 질량으로 쓰면 12부위 합이 481 kg.

    v1 빌더는 질량을 정하지 않고(MassAPI 없음), v2 는 부위 질량 비율(joints.yaml)에서 **밀도**만
    씁니다(jointed.py). 어느 쪽도 표 A.3 의 값을 읽지 않고 질량 속성을 쓰지 않습니다."""
    import inspect

    import generation.dummy.build_dummy as bd
    import generation.dummy.jointed as jd

    bd_src, jd_src = inspect.getsource(bd), inspect.getsource(jd)
    assert "MassAPI" not in bd_src, "빌더가 질량 API 를 씁니다 — 질량은 jointed.py 가 밀도로만"
    assert "CreateMassAttr" not in jd_src, "v2 가 강체 질량을 씁니다 — 밀도만 씁니다(게시기가 physics:mass 를 거부)"
    assert "effective_mass" not in jd_src and "spring_constant" not in jd_src
    for banned in ("keti:effectiveMass", "keti:springConstant"):
        assert banned not in bd_src and banned not in jd_src, (
            f"{banned} 을 자산에 씁니다 — 표 A.3 의 값이고 이 자산은 CC BY 4.0 으로 재배포됩니다")


def test_the_built_asset_carries_no_standard_values():
    """소스 문자열이 아니라 **산출물**을 봅니다.

    이름을 바꾸거나 헬퍼를 거치면 소스 검사는 통과합니다.
    """
    import subprocess

    from pathlib import Path as P

    usd = P("generation/dummy/build/keti_dummy_ts15066_v1.usd")
    if not usd.is_file():
        import pytest
        pytest.skip("자산이 빌드되어 있지 않습니다")

    from generation.publish_dataset import (FORBIDDEN_USD_ATTRS,
                                            redistributed_usd_has_no_standard_values)
    import shutil
    import tempfile

    tmp = P(tempfile.mkdtemp())
    (tmp / "assets").mkdir()
    shutil.copy2(usd, tmp / "assets" / usd.name)
    try:
        # pxr 이 없는 인터프리터면 「확인 못 함」이 나온다 — 그것도 통과가 아니다. 그때는
        # 정본이 확인했다고 적은 해시와 같아야 통과한다(해시가 내용을 고정한다).
        # 「배포하면 안 되는 속성」 문구만 거르면 「확인 못 함」이 조용히 통과한다.
        verified = {usd.name: _canon()["scene"]["keti_dummy_ts15066_v1"]["usd_checked"]["sha256"]}
        problems = redistributed_usd_has_no_standard_values(tmp, verified=verified)
        assert problems == [], problems
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- 시각 메시를 입힌 더미 (S1d 미리보기) ------------------------------------------------

def _canon():
    import json
    from pathlib import Path as P
    return json.loads(P("meta/assets.json").read_text(encoding="utf-8"))


from pathlib import Path  # noqa: E402 — used by the lock test below


def test_the_skinned_dummy_is_registered_as_a_visual_variant_of_v1():
    """정본은 「물리는 v1 과 같다」고 적는다 — 그 상대가 실재하고, 근거 문서와 명세가 있다.

    재배포 자산이다. 변환기·게시기가 정본의 redistributed 항목 전부를 싣는다면 이 값이
    false 여야 하겠지만, 배포본은 그 배치의 에피소드가 쓰는 것만
    싣는다 — v0.1 재변환이 이 파일을 싣지 않는다는 것은
    test_convert_lerobot.test_a_v0_1_batch_ships_the_v1_dummy_alone 이 지킨다."""
    from pathlib import Path as P
    scene = _canon()["scene"]
    e = scene["keti_dummy_ts15066_v1_skin"]
    assert e["physics_identical_to"] in scene
    assert e["redistributed"] is True and e["redistributed_note"]
    assert P(e["spec"]).is_file() and P(e["built_by"]).is_file()
    assert "--skin" in e["build_args"]
    assert P("generation/dummy/skin/SKIN-LICENSE.md").is_file()
    assert "SKIN-LICENSE.md" in e["skin_source"] and "CC0" in e["skin_source"]
    assert set(e["semantics"]["class"]) == {f"BR-{i:02d}" for i in range(1, 13)}


def test_the_skinned_dummy_pins_the_texture_lock_make_skin_wrote():
    """textures.lock 은 make_skin.py 가 쓰고 매번 대조하는 정본이다. 정본의 lock 해시가
    실제 lock 과 다르면 「어느 텍스처를 실었나」가 두 문서에서 갈린다."""
    import hashlib
    from pathlib import Path as P
    b = _canon()["scene"]["keti_dummy_ts15066_v1_skin"]["bundle"]
    lock = P(b["textures_lock"])
    assert lock.is_file()
    assert hashlib.sha256(lock.read_bytes()).hexdigest() == b["textures_lock_sha256"]
    from generation.dummy.skin import make_skin
    rows = make_skin.parse_lock(lock)
    assert all(r["source"].startswith("pack:") for r in rows)
    textures = [r for r in rows if make_skin.is_texture_row(r)]
    pack_files = [r for r in rows if not make_skin.is_texture_row(r)]
    assert len(textures) == 8
    # the figure's own sources are pinned too: each asset's .mhclo/.mhmat, its geometry
    # and its material (directory membership is not provenance)
    assert {Path(r["name"]).suffix for r in pack_files} == {".mhclo", ".obj", ".mhmat"}
    assert sum(1 for r in pack_files if r["name"].endswith(".mhclo")) == 5   # worksuit · shoes · hair · eyes · eyebrows
    assert all(r["source"] == f"pack:{r['name']}" for r in pack_files)


def test_the_built_skinned_asset_carries_no_standard_values_either():
    """v1 과 같은 검사를 시각 변형에도 — 산출물이 있는 기계에서만."""
    from pathlib import Path as P
    usd = P("generation/dummy/build/keti_dummy_ts15066_v1_skin.usd")
    if not usd.is_file():
        import pytest
        pytest.skip("시각 메시 자산이 빌드되어 있지 않습니다")
    import shutil
    import tempfile

    from generation.publish_dataset import redistributed_usd_has_no_standard_values
    tmp = P(tempfile.mkdtemp())
    (tmp / "assets").mkdir()
    shutil.copy2(usd, tmp / "assets" / usd.name)
    try:
        verified = {usd.name: _canon()["scene"]["keti_dummy_ts15066_v1_skin"]["usd_checked"]["sha256"]}
        problems = redistributed_usd_has_no_standard_values(tmp, verified=verified)
        assert problems == [], problems
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_material_names_that_collide_after_sanitising_are_refused():
    """"Human.a-b" 와 "Human.a_b" 는 같은 프림 이름으로 접혀 뒤가 앞을 덮어쓴다 —
    서브셋도 충돌해 부위의 재질 분할이 조용히 빈다."""
    import pytest

    import generation.dummy.build_dummy as bd

    assert bd._material_tokens(["Human.body", "Human.shoes01"]) == {"Human.body": "Human_body",
                                                                    "Human.shoes01": "Human_shoes01"}
    with pytest.raises(ValueError, match="Human_a_b"):
        bd._material_tokens(["Human.a-b", "Human.a_b"])
    assert bd._material_tokens(["Human.x", "Human.x"]) == {"Human.x": "Human_x"}   # the same name twice is fine


def test_skin_joint_anchors_are_what_the_engine_reads_off_v1():
    """v1 writes a joint anchor on a box body in the box's *scaled* frame, as float32, and the
    engine multiplies it back by the box scale in float32. The skinned asset's body is an
    unscaled Xform, so writing the plain offset there is the same point to 1e-8 m but not the
    same float32: 4 of the 16 anchors came out one ulp off, and S1a episode 0 went from a
    fall at step 699 to a time-out at step 999 against the v0.1 batch (2026-09-23). The
    pinned case is BR-11 on BR-07 as measured: v1's authored anchor and scale, and the value
    the engine used (the skinned asset reproduced the batch once it carried that value)."""
    import inspect

    import numpy as np

    import generation.dummy.build_dummy as bd

    v1_local = (0.0, 0.27272725105285645, -1.75)          # BR_11_left/fixed_to_BR_07 localPos0 in v1
    v1_scale = (0.20000000298023224, 0.33000001311302185, 0.1599999964237213)   # BR_07 scale op
    engine = (0.0, 0.08999999612569809, -0.2800000011920929)
    assert bd.engine_anchor(v1_local, v1_scale) == engine
    # ... which is not the plain offset the first skin build wrote
    plain = tuple(float(x) for x in np.array([0.0, 0.09, -0.28], dtype=np.float32))
    assert plain != engine


def test_the_skinned_build_authors_that_anchor_and_checks_itself_against_v1():
    """The authoring path, pinned by its structure (CI has no pxr to build with): under
    `if split_body`, local0 = ...engine_anchor(local0, _scale_op(parent_shape)) — the anchor
    in v1's scaled frame times the collider's own scale, not any other scale — and that
    local0 is what CreateLocalPos0Attr authors. The call-site check this replaced only looked
    for two substrings and passed with engine_anchor(local0, (1.0, 1.0, 1.0)). The build-time
    comparison against a v1 built from the same inputs is the check that sees the authored
    values; this test pins that it is called."""
    import ast
    import inspect

    import generation.dummy.build_dummy as bd

    tree = ast.parse(inspect.getsource(bd.build))

    def name(node):
        return getattr(node, "id", None) or getattr(node, "attr", None)

    def is_engine_anchor_of_local0(call):
        return (isinstance(call, ast.Call) and name(call.func) == "engine_anchor" and len(call.args) == 2
                and name(call.args[0]) == "local0"
                and isinstance(call.args[1], ast.Call) and name(call.args[1].func) == "_scale_op"
                and [name(a) for a in call.args[1].args] == ["parent_shape"])

    branches = [n for n in ast.walk(tree) if isinstance(n, ast.If) and name(n.test) == "split_body"]
    assigned = [a for b in branches for a in ast.walk(b) if isinstance(a, ast.Assign)
                and [name(t) for t in a.targets] == ["local0"]
                and any(is_engine_anchor_of_local0(c) for c in ast.walk(a.value))]
    assert assigned, "the skinned branch does not author engine_anchor(local0, _scale_op(parent_shape))"
    authored = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and name(c.func) == "CreateLocalPos0Attr"]
    assert [[name(a) for a in c.args] for c in authored] == [["local0"]]
    # parent_shape is the collider child in the skinned layout, the body itself in v1
    shape_assign = [a for a in ast.walk(tree) if isinstance(a, ast.Assign)
                    and [name(t) for t in a.targets] == ["parent_shape"]]
    assert shape_assign and any(isinstance(n, ast.IfExp) and name(n.test) == "split_body"
                                for n in ast.walk(shape_assign[0].value))
    assert any(isinstance(c, ast.Call) and name(c.func) == "verify_engine_identity" for c in ast.walk(tree))


# ---- 작업복을 상의 · 바지로 가르기 (build_dummy.py --split-worksuit) ------------------------------

def test_worksuit_islands_join_the_faces_that_share_a_uv_index():
    """An island is the faces linked through shared UV indices, however long the chain; two faces whose UVs only sit
    at the same coordinates under other indices are islands of their own (the buttons are stacked that way)."""
    import generation.dummy.build_dummy as bd

    # face 2 links faces 0 and 1 through indices 2 and 4 -- 4 is not the first index of face 1
    islands = bd.worksuit_islands([[0, 1, 2], [3, 4, 5], [2, 4, 6], [7, 8, 9], [10, 11, 12]])
    assert [list(i) for i in islands] == [[0, 1, 2], [3], [4]]
    assert bd.worksuit_islands([]) == []


def _two_island_mesh(shirt_at=(0.3, 0.6, 0.7, 0.95), stacked=False):
    """A body face over SHIRT_UV, then the worksuit: a quad in the UV box ``shirt_at`` (u0, u1, v0, v1) and a quad low
    in UV -- two islands of two faces each. ``stacked`` adds a third island: the upper quad again under its own UV
    indices."""
    import numpy as np

    from generation.dummy.skin import region_split as rs

    u0, u1, v0, v1 = shirt_at
    vt = [(0.3, 0.6), (0.6, 0.6), (0.45, 0.99),                                # the body face, over SHIRT_UV too
          (u0, v0), (u1, v0), (u1, v1), (u0, v1),                             # the upper quad
          (0.1, 0.05), (0.9, 0.05), (0.9, 0.4), (0.1, 0.4)]                   # the lower quad
    ft = [[0, 1, 2], [3, 4, 5], [3, 5, 6], [7, 8, 9], [7, 9, 10]]
    if stacked:
        vt += vt[3:7]
        ft += [[11, 12, 13], [11, 13, 14]]
    n = len(ft)
    return rs.ObjMesh(V=np.zeros((3, 3)), VT=np.asarray(vt, float), VN=np.zeros((0, 3)),
                      F=np.asarray([[0, 1, 2]] * n, np.int64), FT=np.asarray(ft, np.int64),
                      FN=np.full((n, 3), -1, np.int64), FM=np.asarray([0] + [1] * (n - 1), np.int64), FO=np.zeros(n, np.int64),
                      materials=["Human.body", "Human.male_worksuit01"], objects=["Human"])


def _texture(top, bottom, rows=64, split=0.5):
    """An RGB image whose rows above ``split`` (of the height, from the top) are ``top`` and the rest ``bottom``."""
    import numpy as np

    img = np.empty((rows, rows, 3))
    img[: int(rows * split)] = top
    img[int(rows * split):] = bottom
    return img


WHITE, DENIM = (1.0, 1.0, 1.0), (0.39, 0.56, 0.81)


def test_the_shirt_is_the_white_worksuit_island_that_holds_the_shirt_uv(monkeypatch, tmp_path):
    """The island whose triangles hold SHIRT_UV, as indices into the mesh's faces -- the body face over the same point
    is not the worksuit's. v runs up the image (row 0 is v = 1)."""
    import generation.dummy.build_dummy as bd

    monkeypatch.setattr(bd, "_texture_rgb", lambda path: _texture(WHITE, DENIM))
    assert list(bd.shirt_faces(_two_island_mesh(), tmp_path / "suit.png")) == [1, 2]


def test_a_shirt_that_is_not_white_or_trousers_that_are_is_refused(monkeypatch, tmp_path):
    import pytest

    import generation.dummy.build_dummy as bd

    for top, bottom, why in ((DENIM, WHITE, "희지 않습니다"),          # the image read upside down
                             (WHITE, WHITE, "흰 섬"),
                             (DENIM, DENIM, "희지 않습니다")):
        monkeypatch.setattr(bd, "_texture_rgb", lambda path, t=top, b=bottom: _texture(t, b))
        with pytest.raises(ValueError, match=why):
            bd.shirt_faces(_two_island_mesh(), tmp_path / "suit.png")


def test_the_shirt_uv_outside_every_worksuit_island_is_refused(monkeypatch, tmp_path):
    import pytest

    import generation.dummy.build_dummy as bd

    monkeypatch.setattr(bd, "_texture_rgb", lambda path: _texture(WHITE, DENIM))
    with pytest.raises(ValueError, match="SHIRT_UV"):
        bd.shirt_faces(_two_island_mesh(shirt_at=(0.6, 0.9, 0.7, 0.95)), tmp_path / "suit.png")
    with pytest.raises(ValueError, match="SHIRT_UV .* 2개"):                 # two islands stacked over the point
        bd.shirt_faces(_two_island_mesh(stacked=True), tmp_path / "suit.png")
    mesh = _two_island_mesh()
    mesh.materials = ["Human.body", "Human.other"]
    with pytest.raises(ValueError, match=bd.WORKSUIT_MATERIAL):
        bd.shirt_faces(mesh, tmp_path / "suit.png")
