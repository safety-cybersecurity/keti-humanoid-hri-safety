"""generation/envs/s1d/extract_camera_mounts.py -- the geometry its placement and checks rest on, on synthetic
meshes. The script needs Isaac python to read the H1 assets, but these functions are numpy only: here they run
in CI and hold when generation/cameras.py changes under them."""
import math

import numpy as np
import pytest

from generation import cameras
from generation.envs.s1d import extract_camera_mounts as ecm

LEVEL = cameras.tilted_along_x_xyzw(0.0)       # optical axis the parent's +x, image right -y, image down -z
W, H, D = ecm.MODULE_BOX_M                     # the module box: width (y), height (z), depth (x) when level
WRIST = {"width": 640, "height": 480, "fov_h_deg": 87.0, "clipping_range_m": [0.02, 40.0]}


def _mesh(points, faces):
    """(points, face vertex counts, face vertex indices), as rendered_meshes hands a mesh over."""
    return (np.asarray(points, float), np.array([len(f) for f in faces]), np.array([i for f in faces for i in f]))


def _sheet(x_bottom, x_top):
    """One triangle standing across the module box's whole footprint around the origin -- every corner far
    outside it (|y| 0.2 or |z| 0.3) -- with x linear in z: x_bottom at z = -0.2, x_top at z = 0.3."""
    return _mesh([(x_bottom, 0.2, -0.2), (x_bottom, -0.2, -0.2), (x_top, 0.0, 0.3)], [(0, 1, 2)])


def _x_at(x_bottom, x_top, z):
    return x_bottom + (x_top - x_bottom) * (z + 0.2) / 0.5


# ---- the module box against a head: a face counts although none of its vertices is inside the footprint ----

def test_a_face_reaching_into_the_box_with_every_vertex_outside_its_footprint_is_seen():
    # flat at x = -0.01: 5 mm in front of the box's back face (x = -D/2), so it reaches 5 mm into the box
    assert ecm.module_gap([_sheet(-0.01, -0.01)], (0.0, 0.0, 0.0), LEVEL) == pytest.approx(-D / 2 + 0.01)


def test_a_sloped_face_counts_at_its_frontmost_point_inside_the_footprint():
    # x grows with z, so inside |z| <= H/2 the face is frontmost on the footprint's top edge, between vertices
    front = _x_at(-0.03, 0.02, H / 2)
    assert ecm.module_gap([_sheet(-0.03, 0.02)], (0.0, 0.0, 0.0), LEVEL) == pytest.approx(-D / 2 - front)


def test_a_head_behind_the_box_is_clear_and_one_beside_it_does_not_count():
    assert ecm.module_gap([_sheet(-0.02, -0.02)], (0.0, 0.0, 0.0), LEVEL) == pytest.approx(0.02 - D / 2)
    beside = _mesh([(0.0, 0.1, -0.2), (0.0, 0.5, -0.2), (0.0, 0.3, 0.3)], [(0, 1, 2)])   # y 0.1..0.5 only
    assert ecm.module_gap([beside], (0.0, 0.0, 0.0), LEVEL) == math.inf


def test_the_gap_is_measured_in_the_module_s_own_frame():
    # the same sheet seen from a module shifted 1 cm forward: 1 cm more of it is inside the box
    here = ecm.module_gap([_sheet(-0.02, -0.02)], (0.0, 0.0, 0.0), LEVEL)
    assert ecm.module_gap([_sheet(-0.02, -0.02)], (-0.01, 0.0, 0.0), LEVEL) == pytest.approx(here - 0.01)


RULE = {"above": "below", "up_m": 0.05, "clear_front_m": D / 2, "tilt_down_deg": 0.0}
BELOW = (0.0, 0.0, -0.05)                      # the lower camera: the module goes 5 cm above it, at the origin


def test_the_module_stands_clear_front_in_front_of_a_face_crossing_its_footprint():
    pos, q, info = ecm.forehead_mount({"head": [_sheet(-0.03, 0.02)]}, BELOW, RULE)
    front = _x_at(-0.03, 0.02, H / 2)
    assert pos.tolist() == pytest.approx([round(front + D / 2, 6), 0.0, 0.0])
    assert info["governing"] == "head" and cameras._q_rot(q, (0, 0, 1)) == pytest.approx((1, 0, 0))


def test_the_frontmost_of_the_heads_governs_and_the_box_then_touches_it():
    heads = {"back": [_sheet(-0.02, -0.02)], "front": [_sheet(-0.01, -0.01)]}
    pos, q, info = ecm.forehead_mount(heads, BELOW, RULE)
    assert info["governing"] == "front" and pos[0] == pytest.approx(-0.01 + D / 2)
    assert ecm.module_gap(heads["front"], pos, q) == pytest.approx(0.0, abs=1e-9)
    assert ecm.module_gap(heads["back"], pos, q) == pytest.approx(0.01)


def test_no_head_in_the_footprint_is_refused():
    beside = _mesh([(0.0, 0.1, -0.2), (0.0, 0.5, -0.2), (0.0, 0.3, 0.3)], [(0, 1, 2)])
    with pytest.raises(ValueError, match="no head mesh"):
        ecm.forehead_mount({"head": [beside]}, BELOW, RULE)


# ---- the near plane against the camera's own link ----

def test_a_face_the_near_plane_cuts_inside_the_picture_is_a_cut():
    # a small quad straddling the near plane (0.02 m) straight ahead
    quad = _mesh([(0.01, 0.005, 0.005), (0.01, -0.005, 0.005), (0.05, -0.005, -0.005), (0.05, 0.005, -0.005)],
                 [(0, 1, 2, 3)])
    assert ecm.near_plane([quad], (0.0, 0.0, 0.0), LEVEL, WRIST)["cut"] == 1


def test_a_cut_counts_each_face_once():
    # the quad above as two triangles sharing their diagonal: two faces, not the four edges they cross with
    pts = [(0.01, 0.005, 0.005), (0.01, -0.005, 0.005), (0.05, -0.005, -0.005), (0.05, 0.005, -0.005)]
    assert ecm.near_plane([_mesh(pts, [(0, 1, 2), (0, 2, 3)])], (0.0, 0.0, 0.0), LEVEL, WRIST)["cut"] == 2


def test_a_large_face_cut_across_the_picture_is_a_cut_though_its_edges_cross_outside_it():
    # its two edges cross the near plane at y = +-0.75, far outside the picture; the cut between them runs
    # straight through the middle of it
    big = _mesh([(0.01, 1.0, 0.1), (0.01, -1.0, 0.1), (0.05, 0.0, -0.3)], [(0, 1, 2)])
    assert ecm.near_plane([big], (0.0, 0.0, 0.0), LEVEL, WRIST)["cut"] == 1


def test_a_piece_wholly_inside_the_near_plane_or_wholly_beyond_it_is_not_a_cut():
    inside = _mesh([(0.01, 0.002, 0.0), (0.015, -0.002, 0.0), (0.012, 0.0, 0.002)], [(0, 1, 2)])
    got = ecm.near_plane([inside], (0.0, 0.0, 0.0), LEVEL, WRIST)
    assert got["cut"] == 0 and got["clipped"] == 3
    beyond = _mesh([(0.1, 0.02, 0.0), (0.15, -0.02, 0.0), (0.12, 0.0, 0.02)], [(0, 1, 2)])
    assert ecm.near_plane([beyond], (0.0, 0.0, 0.0), LEVEL, WRIST)["cut"] == 0


def test_a_cut_outside_the_picture_is_not_a_cut():
    # straddles the near plane, but 1 m to the side
    aside = _mesh([(0.01, 1.0, 0.005), (0.01, 1.01, -0.005), (0.05, 1.005, 0.0)], [(0, 1, 2)])
    assert ecm.near_plane([aside], (0.0, 0.0, 0.0), LEVEL, WRIST)["cut"] == 0


# ---- the URDF cross-check belongs to the D435's link alone ----

D435 = "d435_rgb_module_link"


def test_the_d435_is_held_to_its_urdf_joint():
    _, xyz, rpy = ecm.URDF_SOURCES[D435]
    rot = ecm.rpy_rotation(*rpy)
    assert ecm.urdf_cross_check("head_down_rgb", D435, np.asarray(xyz), rot)[1] == []
    _, problems = ecm.urdf_cross_check("head_down_rgb", D435, np.asarray(xyz) + (0.002, 0.0, 0.0), rot)
    assert len(problems) == 1 and "from the official URDF" in problems[0], problems


def test_a_camera_on_another_source_link_is_not_held_to_the_d435_joint():
    say, problems = ecm.urdf_cross_check("chest_rgb", "logo_link", np.zeros(3), np.eye(3))
    assert problems == [] and any("not cross-checked" in line for line in say)


# ---- the wrist's top is read from faces, and no mounted camera may sit inside its link ----

def _box(lo, hi, split_bottom_at=None, drop=(), flip=False):
    """A closed box, faces pointing out (in with ``flip``); its bottom face fanned about ``split_bottom_at`` when
    given -- a vertex on that face -- and the faces numbered in ``drop`` left out (a mesh with a hole)."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    pts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
           (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    if split_bottom_at is None:
        bottom = [(0, 3, 2, 1)]
    else:
        pts.append(split_bottom_at)
        bottom = [(1, 0, 8), (2, 1, 8), (3, 2, 8), (0, 3, 8)]
    faces = bottom + [(4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    faces = [f[::-1] if flip else f for i, f in enumerate(faces) if i not in drop]
    return _mesh(pts, faces)


# The counterexample: the forearm's length, its top face across the search band with no vertex in it, and one
# vertex in the band -- on the bottom face, which the vertex reading took for the top.
FOREARM_LO, FOREARM_HI = (0.0, -0.035, -0.054), (0.316, 0.035, 0.040)
WRIST_RULE = {"inset_from_tip_m": 0.056, "lift_above_top_m": 0.048, "tilt_down_deg": 15.0}
BURIED = (0.260, 0.0, -0.006)                  # where the vertex reading put the camera: inside the box


def test_the_wrist_top_is_the_face_crossing_the_band_not_a_vertex_below_it():
    box = _box(FOREARM_LO, FOREARM_HI, split_bottom_at=(0.260, 0.0, -0.054))
    pos, q, info = ecm.wrist_mount([box], WRIST_RULE)
    assert info["top_z"] == pytest.approx(0.040)
    assert pos.tolist() == pytest.approx([0.26, 0.0, 0.040 + 0.048])
    assert ecm.winding_number([box], pos) == pytest.approx(0.0, abs=1e-9)


def test_a_sloped_top_counts_at_its_highest_point_inside_the_band():
    # one triangle rising along x, its vertices outside the band: the top is where it leaves the band (x0 + 0.01)
    x0 = 0.316 - WRIST_RULE["inset_from_tip_m"]
    ramp = _mesh([(0.20, -0.05, 0.0), (0.20, 0.05, 0.0), (0.316, 0.0, 0.0116)], [(0, 1, 2)])
    _, _, info = ecm.wrist_mount([ramp], WRIST_RULE)
    assert info["top_z"] == pytest.approx(0.0116 * (x0 + ecm.WRIST_BAND_X_M - 0.20) / 0.116)


def test_no_face_in_the_wrist_band_is_refused():
    far = _mesh([(0.0, 0.5, 0.0), (0.316, 0.5, 0.0), (0.316, 0.6, 0.0)], [(0, 1, 2)])
    with pytest.raises(ValueError, match="no mesh face"):
        ecm.wrist_mount([far], WRIST_RULE)


def test_the_winding_number_says_inside_and_outside_whichever_way_the_faces_point():
    for flip in (False, True):
        box = _box(FOREARM_LO, FOREARM_HI, split_bottom_at=(0.260, 0.0, -0.054), flip=flip)
        assert abs(ecm.winding_number([box], BURIED)) == pytest.approx(1.0)
        assert ecm.winding_number([box], (0.26, 0.0, 0.088)) == pytest.approx(0.0, abs=1e-9)


def test_a_mesh_with_a_hole_still_reads_a_point_deep_inside_as_inside():
    # the H1 official meshes have a few open edges; a cube missing a whole face is far worse and its centre still
    # reads 5/6 -- while a point outside, off the hole, stays near 0
    open_box = _box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), drop=(1,))       # the top face left out
    assert ecm.winding_number([open_box], (0.0, 0.0, 0.0)) == pytest.approx(5 / 6)
    assert abs(ecm.winding_number([open_box], (3.0, 0.0, 0.0))) < 0.1



def test_meshes_of_a_link_that_point_opposite_ways_do_not_cancel():
    # A link is several meshes (the simulated torso is two), and nothing makes them face the same way: one facing
    # out and one facing in, both around the point, summed to 0 -- a buried camera read as outside
    outward = _box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    inward = _box((-0.8, -0.6, -0.5), (0.6, 0.7, 0.9), flip=True)
    assert abs(ecm.winding_number([outward, inward], (0.0, 0.0, 0.0))) == pytest.approx(1.0)
    assert ecm.winding_number([outward, inward], (3.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-9)
    q = cameras.tilted_along_x_xyzw(15.0)

    class Both:
        def faces(self, copy, link):
            return [outward, inward]

    canon = {"cameras": {"left_wrist_rgb": _wrist_camera((0.0, 0.0, 0.0), q)}}
    _, problems = ecm.geometry(canon, {"left_wrist_rgb": (np.zeros(3), q)}, Both())
    assert any("is inside the simulated left_elbow_link mesh" in p for p in problems), problems


class _OneMesh:
    """A Meshes stand-in: every copy's every link is the same mesh."""

    def __init__(self, mesh):
        self._mesh = mesh

    def faces(self, copy, link):
        return [self._mesh]


def _wrist_camera(pos, q):
    return {"type": "mounted", "parent_prim": "left_elbow_link", "mount": dict(WRIST_RULE), **WRIST,
            "offset_pos": list(pos), "offset_rot_xyzw": list(q)}


def test_a_camera_buried_in_its_link_is_a_problem_the_near_plane_check_does_not_see():
    # That case end to end: at the buried offset nothing is cut at the near plane -- the box's faces are all
    # beyond it or outside the picture -- so this was green; the camera renders the inside of the box
    box = _box(FOREARM_LO, FOREARM_HI, split_bottom_at=(0.260, 0.0, -0.054))
    q = cameras.tilted_along_x_xyzw(15.0)
    canon = {"cameras": {"left_wrist_rgb": _wrist_camera(BURIED, q)}}
    assert ecm.near_plane([box], BURIED, q, canon["cameras"]["left_wrist_rgb"])["cut"] == 0
    _, problems = ecm.geometry(canon, {"left_wrist_rgb": (np.asarray(BURIED), q)}, _OneMesh(box))
    assert len(problems) == 1 and "is inside the simulated left_elbow_link mesh" in problems[0], problems
    # at the extraction's own offset it is clear
    pos, q, _ = ecm.wrist_mount([box], WRIST_RULE)
    assert ecm.geometry(canon, {"left_wrist_rgb": (pos, q)}, _OneMesh(box))[1] == []


def test_every_mounted_camera_is_held_to_its_link_and_the_add_on_module_to_both_heads():
    # the source camera (the real D435) is held to the simulated mesh alone -- its centre lies inside the official
    # head, behind the window; the add-on module to both
    head = _box((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1))
    q = LEVEL
    canon = {"cameras": {"low": {"type": "mounted", "parent_prim": "torso_link", "mount": {"source": "d435"},
                                 **WRIST},
                         "high": {"type": "mounted", "parent_prim": "torso_link",
                                  "mount": {"above": "low", "up_m": 0.05, "clear_front_m": D / 2, "tilt_down_deg": 0.0},
                                  **WRIST}}}

    class Copies:
        def faces(self, copy, link):
            return [head] if copy == ecm.OFFICIAL else [_box((-0.1, -0.1, -0.1), (-0.05, 0.1, 0.1))]

    say, problems = ecm.geometry(canon, {"low": (np.zeros(3), q), "high": (np.zeros(3), q)}, Copies())
    inside = [p for p in problems if "is inside" in p]
    assert inside == [p for p in inside if p.startswith("high:")] and len(inside) == 1, problems
    assert "official" in inside[0], inside
    assert sum("winding number about the" in line for line in say) == 3, say


# ---- the module behind the head: the forehead's mirror, looking straight back ----

BACK_RULE = {"behind": "front", "clear_back_m": D / 2, "tilt_down_deg": 0.0}
FRONT = (0.115, 0.0325, 0.03)                  # the reference camera: the rear one takes its height and its y
HEAD_CAM = {"width": 640, "height": 480, "fov_h_deg": 69.4, "clipping_range_m": [0.05, 40.0]}
# Two heads, as the two H1 copies are: closed boxes, their back faces at x -0.10 and -0.12 (their fronts at +0.10 and
# +0.08 lie behind a camera that looks back, and do not count)
HEADS = {"forward": [_box((-0.10, -0.1, -0.1), (0.10, 0.1, 0.1))],
         "hind": [_box((-0.12, -0.1, -0.1), (0.08, 0.1, 0.1))]}


def test_the_rear_camera_stands_clear_behind_the_hindmost_head():
    pos, q, info = ecm.rear_mount(HEADS, FRONT, BACK_RULE)
    assert pos.tolist() == pytest.approx([-0.12 - 0.015, 0.0325, 0.03])      # -0.135; y and z the front camera's
    assert info["governing"] == "hind" and info["backs"] == pytest.approx({"forward": -0.10, "hind": -0.12})
    # the box's front face then touches the hindmost head, and stands 2 cm clear of the other
    assert ecm.module_gap(HEADS["hind"], pos, q) == pytest.approx(0.0, abs=1e-9)
    assert ecm.module_gap(HEADS["forward"], pos, q) == pytest.approx(0.02)


def test_the_rear_camera_looks_straight_back():
    _, q, _ = ecm.rear_mount(HEADS, FRONT, BACK_RULE)
    assert cameras._q_rot(q, (0, 0, 1)) == pytest.approx((-1, 0, 0), abs=1e-12)    # optical axis: the link's -x
    assert cameras._q_rot(q, (0, 1, 0)) == pytest.approx((0, 0, -1), abs=1e-12)    # image down: the link's -z
    assert cameras._q_rot(q, (1, 0, 0)) == pytest.approx((0, 1, 0), abs=1e-12)     # image right: the robot's left
    # tilted, it turns down toward the floor behind, still in the link's x-z plane
    _, q, _ = ecm.rear_mount(HEADS, FRONT, dict(BACK_RULE, tilt_down_deg=15.0))
    axis = cameras._q_rot(q, (0, 0, 1))
    assert ecm.down_angle_deg(axis) == pytest.approx(15.0) and axis[0] < 0 and axis[1] == pytest.approx(0, abs=1e-12)
    assert cameras._q_rot(q, (1, 0, 0)) == pytest.approx((0, 1, 0), abs=1e-12)


def test_no_head_behind_the_module_is_refused():
    beside = {"head": [_box((-0.10, 0.3, -0.1), (0.10, 0.5, 0.1))]}            # y 0.3..0.5: not across the footprint
    with pytest.raises(ValueError, match="no head mesh"):
        ecm.rear_mount(beside, FRONT, BACK_RULE)


@pytest.mark.parametrize("named", ["nowhere", "third_person_rgb"], ids=["no-such-camera", "not-mounted"])
def test_a_rear_rule_naming_no_camera_is_refused(named):
    canon = {"cameras": {"rear_rgb": {"type": "mounted", "parent_prim": "torso_link",
                                      "mount": dict(BACK_RULE, behind=named), **HEAD_CAM},
                         "third_person_rgb": {"type": "follow", **HEAD_CAM}}}
    with pytest.raises(ValueError, match=f"rear_rgb is mounted behind '{named}'"):
        ecm.extract(canon, {}, None)


def test_the_rear_module_is_held_to_both_heads():
    # placed from the simulated head alone (back face -0.10: centre -0.115, box -0.13..-0.10), the module sits 2 cm
    # deep in the official one (back face -0.12), its optical centre inside it -- the forehead's case, behind; placed
    # from both, nothing is in the way: the head is behind its optical centre, the near plane cuts nothing
    class Copies:
        def faces(self, copy, link):
            return HEADS["hind"] if copy == ecm.OFFICIAL else HEADS["forward"]

    canon = {"cameras": {"rear_rgb": {"type": "mounted", "parent_prim": "torso_link", "mount": BACK_RULE, **HEAD_CAM}}}
    pos, q, _ = ecm.rear_mount({"simulated": HEADS["forward"]}, FRONT, BACK_RULE)
    say, problems = ecm.geometry(canon, {"rear_rgb": (pos, q)}, Copies())
    assert problems == ["rear_rgb: its optical centre [-0.115, 0.0325, 0.03] is inside the official torso_link mesh "
                        "(winding number 1.000) -- the camera is buried in it",
                        "rear_rgb: the official head reaches 20.000 mm into the module box"], problems
    assert sum("winding number about the" in line for line in say) == 2, say
    pos, q, _ = ecm.rear_mount({"simulated": HEADS["forward"], "official": HEADS["hind"]}, FRONT, BACK_RULE)
    say, problems = ecm.geometry(canon, {"rear_rgb": (pos, q)}, Copies())
    assert problems == [], problems
    assert any("0 faces cut inside the picture" in line for line in say), say
