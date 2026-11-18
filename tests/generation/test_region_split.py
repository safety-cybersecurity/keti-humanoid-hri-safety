"""generation/dummy/skin/region_split.py — 시각 메시의 정점이 정본의 부위 프리미티브로 갈리는가.

순수 numpy 라 CI 에서 돈다. USD 는 보지 않는다 — 부위별 자식 메시가 실제로 붙었는지는
build_dummy.py --skin 의 자체 검사와 check_contact_resolution.py (Isaac python) 의 몫이다.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from generation.dummy.skin import region_split as rs

REPO = Path(__file__).resolve().parents[2]
ANTHRO = yaml.safe_load((REPO / "generation/dummy/anthropometry.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prims():
    return rs.primitives_from_anthropometry(ANTHRO)


# ---- 계획이 적은 네 시험 -------------------------------------------------------------

def test_points_at_primitive_centres_get_that_primitive(prims):
    V = np.array([p.center for p in prims])
    codes = rs.assign_vertices(V, prims)
    assert list(codes) == [p.code for p in prims]


def test_every_region_code_appears_in_the_primitive_list(prims):
    assert {p.code for p in prims} == set(ANTHRO["regions"])


def test_split_keeps_every_face_exactly_once(prims):
    rng = np.random.default_rng(0)
    V = rng.uniform([-0.4, -0.4, 0.0], [0.4, 0.4, 1.8], size=(300, 3))
    F = rng.integers(0, 300, size=(200, 3))
    parts = rs.split_faces(F, rs.assign_vertices(V, prims))
    assert sum(len(f) for f, _, _ in parts.values()) == len(F)
    # and the original face ids partition range(len(F))
    ids = np.concatenate([fid for _, _, fid in parts.values()])
    assert sorted(ids.tolist()) == list(range(len(F)))


def test_capsule_distance_is_zero_on_its_surface():
    p = rs.Primitive("BR-11", "capsule", center=np.array([0.0, 0.0, 0.5]),
                     radius=0.06, height=0.3, axis="z", size=None)
    assert abs(rs.distance(np.array([[0.06, 0.0, 0.5]]), p)[0]) < 1e-9
    # the cylindrical section is `height` long; the caps add a radius each side
    assert abs(rs.distance(np.array([[0.0, 0.0, 0.5 + 0.15 + 0.06]]), p)[0]) < 1e-9


# ---- 좌우 한 쌍 — 프림은 둘, 코드는 하나 -----------------------------------------------

def test_mirrored_regions_become_two_primitives_named_like_the_usd_prims(prims):
    """build_dummy.py 는 mirror_y 부위마다 BR_xx_left / BR_xx_right 프림 둘을 만든다.
    분할도 같은 이름을 써야 서브메시가 자기 강체 밑에 붙는다."""
    expected = sum(2 if s.get("mirror_y") else 1 for s in ANTHRO["regions"].values())
    assert len(prims) == expected
    by_name = {p.name: p for p in prims}
    for code, s in ANTHRO["regions"].items():
        stem = code.replace("-", "_")
        if s.get("mirror_y"):
            left, right = by_name[f"{stem}_left"], by_name[f"{stem}_right"]
            assert left.code == right.code == code
            assert left.center[1] == pytest.approx(-right.center[1]) and left.center[1] > 0
            assert (left.side, right.side) == ("left", "right")
        else:
            assert by_name[stem].side is None and by_name[stem].code == code


def test_a_vertex_on_the_right_arm_lands_on_the_right_prim(prims):
    c = np.asarray(ANTHRO["regions"]["BR-08"]["center"], float)
    V = np.array([c * [1, -1, 1], c])
    idx = rs.assign_primitives(V, prims)
    assert [prims[i].name for i in idx] == ["BR_08_right", "BR_08_left"]
    assert list(rs.assign_vertices(V, prims)) == ["BR-08", "BR-08"]


def test_prim_names_follow_the_builder_convention():
    assert rs.prim_names("BR-04", {}) == [("BR_04", 1.0)]
    assert rs.prim_names("BR-08", {"mirror_y": True}) == [("BR_08_left", 1.0), ("BR_08_right", -1.0)]


# ---- 형상별 거리 --------------------------------------------------------------------

def test_box_distance_is_negative_inside_and_zero_on_a_face():
    p = rs.Primitive("BR-05", "box", center=np.array([0.0, 0.0, 1.0]), radius=None,
                     height=None, axis=None, size=np.array([0.2, 0.4, 0.6]))
    d = rs.distance(np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.3, 0.0, 1.0]]), p)
    assert d[0] < 0 and abs(d[1]) < 1e-9 and d[2] == pytest.approx(0.2)


def test_sphere_distance_is_signed():
    p = rs.Primitive("BR-01", "sphere", center=np.array([0.0, 0.0, 1.6]), radius=0.1,
                     height=None, axis=None, size=None)
    d = rs.distance(np.array([[0.0, 0.0, 1.6], [0.0, 0.0, 1.75]]), p)
    assert d[0] == pytest.approx(-0.1) and d[1] == pytest.approx(0.05)


def test_bottom_of_each_shape(prims):
    by = {p.name: p for p in prims}
    r = ANTHRO["regions"]
    assert rs.bottom_of(by["BR_01"]) == pytest.approx(r["BR-01"]["center"][2] - r["BR-01"]["radius"])
    assert rs.bottom_of(by["BR_07"]) == pytest.approx(r["BR-07"]["center"][2] - r["BR-07"]["size"][2] / 2)
    assert rs.bottom_of(by["BR_12_left"]) == pytest.approx(
        r["BR-12"]["center"][2] - r["BR-12"]["height"] / 2 - r["BR-12"]["radius"])
    # the anthropometry has no foot: the lowest collider is the shin's cap, well above the sole
    assert min(rs.bottom_of(p) for p in prims) > 0.02
    assert rs.top_of(by["BR_01"]) == pytest.approx(r["BR-01"]["center"][2] + r["BR-01"]["radius"])
    assert rs.top_of(by["BR_12_left"]) - rs.bottom_of(by["BR_12_left"]) == pytest.approx(
        r["BR-12"]["height"] + 2 * r["BR-12"]["radius"])
    # the crown of the skull is the stature the mesh is fitted to
    assert max(rs.top_of(p) for p in prims) == pytest.approx(ANTHRO["stature"], abs=0.02)


def test_primitives_reject_a_shape_missing_its_dimensions():
    bad = {"regions": {"BR-03": {"shape": "capsule", "radius": 0.05, "center": [0, 0, 1.5]}}}
    with pytest.raises(ValueError, match="BR-03"):
        rs.primitives_from_anthropometry(bad)
    nan = {"regions": {"BR-01": {"shape": "sphere", "radius": float("nan"), "center": [0, 0, 1.6]}}}
    with pytest.raises(ValueError, match="BR-01"):
        rs.primitives_from_anthropometry(nan)


# ---- 면 분할 -----------------------------------------------------------------------

def test_face_label_is_the_majority_of_its_vertices():
    labels = np.array(["a", "a", "b", "c"])
    F = np.array([[0, 1, 2], [2, 3, 0]])
    parts = rs.split_faces(F, labels)
    # face 0: 2:1 majority -> 'a'; face 1 is a 3-way split with no primitives to
    # settle it -> its first vertex (2, 'b')
    assert set(parts) == {"a", "b"}
    faces, old, fid = parts["a"]
    assert old.tolist() == [0, 1, 2] and faces.tolist() == [[0, 1, 2]] and fid.tolist() == [0]
    faces, old, fid = parts["b"]
    assert old.tolist() == [0, 2, 3] and faces.tolist() == [[1, 2, 0]] and fid.tolist() == [1]


def test_a_three_way_split_is_settled_by_the_centroid_when_primitives_are_given(prims):
    # one vertex each on the skull, the chest and the left thigh: the centroid of that
    # triangle sits around the abdomen, so the face should go where the centroid goes
    r = ANTHRO["regions"]
    V = np.array([r["BR-01"]["center"], r["BR-05"]["center"], r["BR-11"]["center"]], float)
    F = np.array([[0, 1, 2]])
    labels = rs.assign_vertices(V, prims)
    assert len(set(labels)) == 3
    parts = rs.split_faces(F, labels, V=V, prims=prims)
    centroid = V.mean(axis=0, keepdims=True)
    assert list(parts) == [rs.assign_vertices(centroid, prims)[0]]


def test_split_reindexes_faces_into_each_part(prims):
    rng = np.random.default_rng(1)
    V = rng.uniform([-0.4, -0.4, 0.0], [0.4, 0.4, 1.8], size=(120, 3))
    F = rng.integers(0, 120, size=(80, 3))
    for label, (faces, old, fid) in rs.split_faces(F, rs.assign_vertices(V, prims)).items():
        assert faces.min() >= 0 and faces.max() < len(old)
        # mapping back through `old` returns exactly the original faces
        assert old[faces].tolist() == F[fid].tolist()


# ---- OBJ / MTL 읽기 ----------------------------------------------------------------

OBJ = """# test
mtllib skin.mtl
o Body
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
usemtl skin
f 1/1/1 2/2/1 3/3/1 4/4/1
o Suit
v 0 0 1
v 1 0 1
v 1 1 1
usemtl cloth
f 5 6 7
"""

MTL = """newmtl skin
Kd 0.8 0.7 0.6
d 1.0
Pr 0.5
map_Kd textures/skin_diffuse.png
newmtl cloth
Kd 0.2 0.3 0.4
map_Kd textures/cloth_diffuse.png
map_Bump -bm 1.0 textures/cloth_normal.png
"""


def test_read_obj_parses_objects_materials_uvs_and_fan_triangulates(tmp_path):
    (tmp_path / "skin.obj").write_text(OBJ, encoding="utf-8")
    m = rs.read_obj(tmp_path / "skin.obj")
    assert m.V.shape == (7, 3) and m.VT.shape == (4, 2) and m.VN.shape == (1, 3)
    assert m.F.tolist() == [[0, 1, 2], [0, 2, 3], [4, 5, 6]]      # quad -> two triangles
    assert m.FT.tolist() == [[0, 1, 2], [0, 2, 3], [-1, -1, -1]]
    assert m.FN.tolist() == [[0, 0, 0], [0, 0, 0], [-1, -1, -1]]
    assert m.objects == ["Body", "Suit"] and m.FO.tolist() == [0, 0, 1]
    assert m.materials == ["skin", "cloth"] and m.FM.tolist() == [0, 0, 1]
    assert m.mtllib == "skin.mtl"


def test_validate_refuses_what_the_reader_lets_through(tmp_path):
    """A NaN vertex or an index past the array parses fine and used to go all the way
    into the USD."""
    (tmp_path / "skin.obj").write_text(OBJ, encoding="utf-8")
    good = rs.read_obj(tmp_path / "skin.obj")
    good.validate()
    for mutate, why in [
        (lambda m: m.V.__setitem__((0, 0), float("nan")), "non-finite positions"),
        (lambda m: m.F.__setitem__((0, 0), 99), "face vertex index"),
        (lambda m: m.FT.__setitem__((0, 0), 99), "face uv index"),
        (lambda m: m.FN.__setitem__((0, 0), 99), "face normal index"),
        (lambda m: m.FM.__setitem__(0, 7), "face material index"),
        (lambda m: m.VT.__setitem__((0, 1), float("inf")), "non-finite uvs"),
    ]:
        m = rs.read_obj(tmp_path / "skin.obj")
        mutate(m)
        with pytest.raises(ValueError, match=why):
            m.validate()
    empty = rs.ObjMesh(np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 3)), np.zeros((0, 3), np.int64),
                       np.zeros((0, 3), np.int64), np.zeros((0, 3), np.int64), np.zeros(0, np.int64),
                       np.zeros(0, np.int64))
    with pytest.raises(ValueError, match="no vertices"):
        empty.validate()


def test_read_mtl_reads_diffuse_and_normal_maps(tmp_path):
    (tmp_path / "skin.mtl").write_text(MTL, encoding="utf-8")
    mats = rs.read_mtl(tmp_path / "skin.mtl")
    assert set(mats) == {"skin", "cloth"}
    assert mats["skin"].map_kd == "textures/skin_diffuse.png" and mats["skin"].map_bump is None
    assert mats["skin"].kd == pytest.approx((0.8, 0.7, 0.6)) and mats["skin"].pr == pytest.approx(0.5)
    assert mats["cloth"].map_bump == "textures/cloth_normal.png"
    assert mats["cloth"].bump_strength == pytest.approx(1.0)


def test_read_mtl_refuses_non_finite_scalars(tmp_path):
    for bad in ("Pr nan", "d inf", "Kd 0.1 nan 0.3", "map_Bump -bm nan textures/n.png"):
        (tmp_path / "skin.mtl").write_text(f"newmtl m\n{bad}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not finite"):
            rs.read_mtl(tmp_path / "skin.mtl")


def test_stature_and_orientation_helpers():
    # a stick figure facing +x: toes ahead of the waist, heel behind, symmetric in y,
    # and a head leaning forward the way a clothed export's remaining body does
    V = np.array([[0.0, 0.0, 0.0], [0.2, 0.05, 0.0], [-0.06, -0.05, 0.0], [0.2, -0.05, 0.0],
                  [-0.06, 0.05, 0.0], [0.02, 0.0, 0.95], [0.12, 0.0, 1.74], [0.12, 0.0, 1.6]])
    assert rs.stature_of(V) == pytest.approx(1.74)
    rep = rs.orientation_report(V, foot_height=0.05)
    assert rep["faces_plus_x"] and abs(rep["y_mean"]) < 1e-9 and rep["x_center"] == pytest.approx(0.02)
    assert rs.orientation_report(V * [-1, 1, 1], foot_height=0.05)["faces_plus_x"] is False
    with pytest.raises(ValueError, match="standing"):
        rs.orientation_report(V[:5], foot_height=0.05)         # feet only: no waist band
