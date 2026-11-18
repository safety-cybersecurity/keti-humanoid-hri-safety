"""generation/dummy/skin_seams.py -- the jointed dummy's skin seams: stray patches moved to the body round them,
blend weights near the joints, and where a piece's points go for given body poses.

Pure numpy and yaml: CI runs it. The skin USD the weights ride on (build_dummy.py --seams) is tested under the Isaac
python in tests/generation/test_jointed_build.py.

The synthetic skin is a flat grid of squares, each cut into two triangles, labelled by the x of the face's
centroid; its bodies and rigid groups are named in each test. The real skin export is used where it is built.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from generation.dummy import joints_spec as js
from generation.dummy import skin_seams as ss
from generation.dummy.skin import region_split as rs

REPO = Path(__file__).resolve().parents[2]
SKIN_OBJ = REPO / "generation/dummy/skin/build/keti_skin.obj"


def grid(nx, ny, h=0.01, x0=0.0, y0=0.0, start=0):
    """A flat grid nx by ny squares of side h; vertex (i, j) is at index start + j * (nx + 1) + i."""
    V = np.array([[x0 + i * h, y0 + j * h, 0.0] for j in range(ny + 1) for i in range(nx + 1)])
    F = []
    for j in range(ny):
        for i in range(nx):
            a = start + j * (nx + 1) + i
            b, c, d = a + 1, a + nx + 1, a + nx + 2
            F += [(a, b, d), (a, d, c)]
    return V, np.array(F, np.int64)


def by_x(V, F, edges, names):
    """A label per face: names[k] for the centroid's x in [edges[k], edges[k + 1])."""
    cx = V[F].mean(axis=1)[:, 0]
    return np.array([names[int(np.searchsorted(edges, x, side="right")) - 1] for x in cx], dtype="<U16")


# ---- the spec --------------------------------------------------------------------------------------

def test_the_spec_has_a_radius_for_every_joint_group_of_the_jointed_dummy():
    spec = ss.load_seam_spec()
    assert set(spec.radius_m) == {j.group for j in js.load().joints}
    assert all(0 < r < 0.5 for r in spec.radius_m.values())
    assert 0 < spec.stray_share < 1


@pytest.mark.parametrize("change, words", [
    (lambda d: d.update(colour="red"), "모르는 키"),
    (lambda d: d.pop("stray_share"), "키가 없습니다"),
    (lambda d: d["radius_m"].pop("knee"), "radius_m"),
    (lambda d: d["radius_m"].update(wrist=0.05), "radius_m"),
    (lambda d: d["radius_m"].update(hip=-0.1), "0 보다 커야"),
    (lambda d: d["radius_m"].update(hip=True), "수가 아닙니다"),
    (lambda d: d["radius_m"].update(hip=float("nan")), "유한한 수"),
    (lambda d: d.update(stray_share=1.0), "stray_share"),
    (lambda d: d.update(stray_share=0), "stray_share"),
    (lambda d: d.update(seams_version=""), "seams_version"),
])
def test_a_wrong_spec_is_refused_by_its_key(tmp_path, change, words):
    raw = yaml.safe_load(ss.SPEC_PATH.read_text(encoding="utf-8"))
    change(raw)
    path = tmp_path / "skin_seams.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match=words):
        ss.load_seam_spec(path)


# ---- the bodies -------------------------------------------------------------------------------------

def test_each_body_hangs_on_its_parent_by_the_joint_or_fixed():
    links = ss.body_links(js.load())
    assert links["BR_04"] == ss.Link(None, None)
    assert links["BR_07"] == ss.Link("BR_06", None)
    assert links["BR_01"] == ss.Link("BR_03", None)
    assert links["BR_08_left"] == ss.Link("BR_04", "shoulder")
    assert links["BR_09_right"] == ss.Link("BR_08_right", "elbow")
    assert links["BR_10_left"] == ss.Link("BR_09_left", None)
    assert links["BR_11_left"] == ss.Link("BR_07", "hip")
    assert links["BR_12_right"] == ss.Link("BR_11_right", "knee")
    assert links["FOOT_left"] == ss.Link("BR_12_left", "ankle")
    # every joint of the spec is one link, and nothing else turns
    spec = js.load()
    turning = {(b, l.parent, l.joint) for b, l in links.items() if l.joint}
    assert turning == {(j.child, j.parent, j.group) for j in spec.joints}


def test_the_rigid_groups_are_the_bodies_the_fixed_joints_hold():
    roots = ss.rigid_roots(ss.body_links(js.load()))
    assert {b for b, r in roots.items() if r == "BR_04"} == {f"BR_0{i}" for i in range(1, 8)}
    assert roots["BR_10_left"] == roots["BR_09_left"] == "BR_09_left"
    assert roots["FOOT_right"] == "FOOT_right"
    assert len(set(roots.values())) == 11


def test_the_radius_between_two_bodies_is_the_largest_joint_between_them():
    links = ss.body_links(js.load())
    radius = {"hip": 0.10, "knee": 0.08, "ankle": 0.05, "shoulder": 0.07, "elbow": 0.06}
    pair = lambda a, b: ss.pair_radius(a, b, links, radius)  # noqa: E731
    assert pair("BR_11_left", "BR_12_left") == pair("BR_12_left", "BR_11_left") == 0.08
    assert pair("BR_11_left", "BR_11_right") == 0.10
    assert pair("BR_07", "BR_10_left") == 0.07
    assert pair("BR_12_left", "FOOT_left") == 0.05
    assert pair("BR_05", "BR_07") == pair("BR_09_left", "BR_10_left") == 0.0


# ---- stray patches ---------------------------------------------------------------------------------

def square(i, j, nx):
    """The two faces grid(nx, ...) makes for square i of row j."""
    k = 2 * (j * nx + i)
    return [k, k + 1]


def test_a_patch_on_another_rigid_group_inside_a_body_moves_to_that_body():
    V, F = grid(12, 4)                     # A: squares 0-3 of each row, B: 4-7, C: 8-11
    want = by_x(V, F, [0.0, 0.04, 0.08], ["A", "B", "C"])
    fl = want.copy()
    fl[square(1, 1, 12)] = "C"             # strays of C and of B inside A, meeting at a corner only
    fl[square(2, 2, 12)] = "B"
    out, moves = ss.clean_labels(F, fl, {"A": "A", "B": "B", "C": "C"}, 0.05)
    assert (out == want).all()
    assert sorted((m["from"], m["to"], m["faces"]) for m in moves) == [("B", "A", 2), ("C", "A", 2)]


def test_a_patch_of_the_same_rigid_group_stays_where_it_is():
    V, F = grid(12, 4)
    fl = by_x(V, F, [0.0, 0.04, 0.08], ["A", "B", "A2"])
    fl[square(1, 1, 12)] = "A2"            # a patch of another body held to A by a fixed joint
    out, moves = ss.clean_labels(F, fl, {"A": "A", "A2": "A", "B": "B"}, 0.05)
    assert list(out[square(1, 1, 12)]) == ["A2", "A2"] and moves == []


def test_a_bodys_only_patch_stays_whatever_is_round_it():
    V, F = grid(8, 4)
    fl = by_x(V, F, [0.0, 0.04], ["A", "B"])
    fl[square(1, 1, 8)] = "C"              # all of C: its largest patch, holding all its faces
    out, moves = ss.clean_labels(F, fl, {"A": "A", "B": "B", "C": "C"}, 0.05)
    assert list(out[square(1, 1, 8)]) == ["C", "C"] and moves == []


def test_a_separate_piece_of_mesh_keeps_its_label():
    V1, F1 = grid(4, 2)
    V2, F2 = grid(1, 1, x0=1.0, start=len(V1))
    F = np.vstack([F1, F2])
    fl = np.array(["A"] * len(F1) + ["C"] * len(F2), dtype="<U16")
    out, moves = ss.clean_labels(F, fl, {"A": "A", "C": "C"}, 0.05)
    assert (out == fl).all() and moves == []


def test_a_bodys_largest_patch_in_a_piece_still_moves_below_the_share_of_that_body():
    V1, F1 = grid(10, 4)                   # D's own region: 80 faces
    V2, F2 = grid(4, 2, x0=1.0, start=len(V1))
    F = np.vstack([F1, F2])
    fl = np.array(["D"] * len(F1) + ["E"] * len(F2), dtype="<U16")
    fl[len(F1) + 3] = "D"                  # one face of D in E's piece: its largest patch there, 1 of 81
    out, moves = ss.clean_labels(F, fl, {"D": "D", "E": "E"}, 0.05)
    assert out[len(F1) + 3] == "E"
    assert [(m["from"], m["to"], m["faces"]) for m in moves] == [("D", "E", 1)]
    out_kept, _ = ss.clean_labels(F, fl, {"D": "D", "E": "E"}, 0.01)
    assert out_kept[len(F1) + 3] == "D"


# ---- blend weights ---------------------------------------------------------------------------------

def strip():
    V, F = grid(40, 2)                     # x from 0 to 0.40 in 0.01 steps
    fl = by_x(V, F, [0.0, 0.20], ["A", "B"])
    return V, F, fl


def test_a_seam_vertex_is_half_each_body_and_the_weight_fades_to_the_radius():
    V, F, fl = strip()
    W = ss.blend_weights(V, F, fl, {"A": "A", "B": "B"}, lambda a, b: 0.05)
    at = {round(float(V[v, 0]), 3): w for v, w in W.items() if V[v, 1] == 0.01}
    assert at[0.20] == pytest.approx({"A": 0.5, "B": 0.5})
    # 0.03 inside A from B's faces: B's raw weight 1 - 0.03 / 0.05 against A's 1
    assert at[0.17]["B"] == pytest.approx(0.4 / 1.4) and at[0.17]["A"] == pytest.approx(1 / 1.4)
    assert at[0.23]["A"] == pytest.approx(0.4 / 1.4)
    assert 0.15 not in at and 0.25 not in at and 0.10 not in at
    for w in W.values():
        assert sum(w.values()) == pytest.approx(1.0)


def test_bodies_of_one_rigid_group_do_not_blend():
    V, F, fl = strip()
    assert ss.blend_weights(V, F, fl, {"A": "R", "B": "R"}, lambda a, b: 0.05) == {}


def test_the_weights_name_rigid_groups_by_their_root_body():
    V, F, fl = strip()
    fl[fl == "B"] = np.where(V[F[fl == "B"]].mean(axis=1)[:, 0] < 0.3, "B", "B2")
    W = ss.blend_weights(V, F, fl, {"A": "A", "B": "B", "B2": "B"}, lambda a, b: 0.05)
    assert {g for w in W.values() for g in w} == {"A", "B"}


def test_a_piece_carries_its_blended_points_and_a_column_per_group():
    V, F, fl = strip()
    roots = {"A": "A", "B": "B"}
    W = ss.blend_weights(V, F, fl, roots, lambda a, b: 0.05)
    parts = rs.split_by_face(F, fl)
    faces, old, _ = parts["A"]
    rows, cols, M = ss.piece_blend(old, W)
    assert cols == ("A", "B") and M.dtype == np.float32
    assert [int(old[r]) for r in rows] == sorted(v for v in old if int(v) in W)
    assert np.allclose(M.sum(axis=1), 1.0, atol=1e-6)
    assert ss.piece_blend(np.array([0, 1]), W)[0].size == 0


# ---- placing the points ----------------------------------------------------------------------------

def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def two_pieces():
    V, F, fl = strip()
    roots = {"A": "A", "B": "B"}
    W = ss.blend_weights(V, F, fl, roots, lambda a, b: 0.05)
    rest = {"A": np.array([0.1, 0.01, 0.0]), "B": np.array([0.3, 0.01, 0.0])}
    pieces = {}
    for body, (faces, old, _) in rs.split_by_face(F, fl).items():
        rows, cols, M = ss.piece_blend(old, W)
        local = (V[old] - rest[body]).astype(np.float32)
        pieces[body] = (old, local, rows, cols, M)
    return V, rest, pieces


def test_at_rest_every_point_stays_where_it_was_built():
    V, rest, pieces = two_pieces()
    pose = {b: (np.eye(3), rest[b]) for b in rest}
    for body, (old, local, rows, cols, M) in pieces.items():
        out = ss.place(local, rest[body], rows, M, [pose[c] for c in cols], [rest[c] for c in cols], pose[body])
        assert np.allclose(out, local, atol=1e-6)
        unblended = np.setdiff1d(np.arange(len(local)), rows)
        assert np.array_equal(out[unblended], local[unblended])   # the built bytes, not a round trip


def test_a_seam_vertex_lands_in_the_same_place_from_both_pieces():
    V, rest, pieces = two_pieces()
    pivot = np.array([0.20, 0.01, 0.0])
    rng = np.random.default_rng(7)
    for _ in range(5):
        Ra = rot("z", rng.uniform(-30, 30)) @ rot("x", rng.uniform(-30, 30))
        xa = rng.normal(size=3)
        Rb = Ra @ rot("y", rng.uniform(-90, 90))                      # B turns about the seam's pivot
        pose = {"A": (Ra, Ra @ rest["A"] + xa)}
        pose["B"] = (Rb, Ra @ pivot + xa + Rb @ (rest["B"] - pivot))
        placed = {}
        for body, (old, local, rows, cols, M) in pieces.items():
            out = ss.place(local, rest[body], rows, M, [pose[c] for c in cols], [rest[c] for c in cols], pose[body])
            R, x = pose[body]
            placed[body] = dict(zip(old.tolist(), out.astype(np.float64) @ R.T + x))
        shared = set(placed["A"]) & set(placed["B"])
        assert len(shared) == 3
        for v in shared:
            assert np.allclose(placed["A"][v], placed["B"][v], atol=1e-5)


def test_moving_every_body_together_leaves_the_pieces_as_built():
    V, rest, pieces = two_pieces()
    R, x = rot("z", 40) @ rot("x", 10), np.array([1.0, -2.0, 0.5])
    pose = {b: (R, R @ rest[b] + x) for b in rest}
    for body, (old, local, rows, cols, M) in pieces.items():
        out = ss.place(local, rest[body], rows, M, [pose[c] for c in cols], [rest[c] for c in cols], pose[body])
        assert np.allclose(out, local, atol=1e-5)


# ---- the real skin ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    if not SKIN_OBJ.is_file():
        pytest.skip("피부 OBJ 가 빌드되어 있지 않습니다")
    spec = js.load()
    prims = rs.primitives_from_anthropometry(spec.anthropometry) + js.foot_primitives(spec)
    mesh = rs.read_obj(SKIN_OBJ)
    names = np.array([p.name for p in prims])
    fl = rs.face_labels(mesh.F, names[rs.assign_primitives(mesh.V, prims)], V=mesh.V, prims=prims,
                        label_of=lambda p: p.name)
    links = ss.body_links(spec)
    return mesh, fl, links, ss.rigid_roots(links), ss.load_seam_spec()


def test_on_the_real_skin_no_trousers_ride_a_hand_and_no_finger_a_thigh(real):
    mesh, fl, links, roots, seams = real
    mats = np.array(mesh.materials)[mesh.FM]
    before = (np.char.startswith(fl.astype(str), "BR_10") & (mats == "Human.male_worksuit01")).sum()
    assert before > 0, "the stray this rule is for is not in the export any more -- read the rule again"
    out, moves = ss.clean_labels(mesh.F, fl, roots, seams.stray_share)
    assert not (np.char.startswith(out.astype(str), "BR_10") & (mats == "Human.male_worksuit01")).any()
    assert not (np.char.startswith(out.astype(str), "BR_11") & (mats == "Human.body")).any()
    # moves only across rigid groups, and nothing on the trunk moves among the trunk's bodies
    assert all(roots[m["from"]] != roots[m["to"]] for m in moves)


def test_on_the_real_skin_the_seams_blend_and_every_weight_sums_to_one(real):
    mesh, fl, links, roots, seams = real
    out, _ = ss.clean_labels(mesh.F, fl, roots, seams.stray_share)
    W = ss.blend_weights(mesh.V, mesh.F, out, roots,
                         lambda a, b: ss.pair_radius(a, b, links, seams.radius_m))
    assert len(W) > 500
    for w in W.values():
        assert sum(w.values()) == pytest.approx(1.0)
        assert len({roots[g] for g in w}) >= 2
    # every vertex two pieces of different groups share is blended: nothing left to part at a joint
    owner = {}
    for f, lab in zip(mesh.F, out):
        for v in f:
            owner.setdefault(int(v), set()).add(roots[str(lab)])
    assert {v for v, g in owner.items() if len(g) > 1} <= set(W)
