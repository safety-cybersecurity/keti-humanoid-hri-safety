"""Whether a robot camera sees the person (generation/visibility.py): the projection, the view, the
occlusion by boxes, the body's sample points and the 20 % / 50 px rule, on numbers worked out by hand,
then the canon -- meta/cameras.json's seeing block, the dummy's parts and the warehouse layout.
Isaac-free."""

import copy
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from generation import cameras, visibility
from generation.dummy import joints_spec
from generation.dummy.skin import region_split
from generation.scene import layout

# A camera at the origin whose optical axis is world +x: image right is world -y, image down world -z.
LOOK_ALONG_X = np.array([-0.5, 0.5, -0.5, 0.5])
ORIGIN = np.zeros(3)
# A head-like camera written out here: 69.4 deg over 640 px, so f = 640 / (2 tan 34.7 deg) = 462.137.
CAM = {"width": 640, "height": 480, "fov_h_deg": 69.4, "clipping_range_m": [0.05, 40.0]}
NO_BOXES = (np.zeros((0, 3)), np.zeros((0, 3)))


def _box(lo, hi):
    return np.array([lo], float), np.array([hi], float)


def _turned_about_z(deg):
    t = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(t), math.cos(t)])


# ---- the projection ----

@pytest.mark.parametrize("n", [1, 3, 7])
def test_project_points_matches_project_row_by_row(n):
    # Three points make a 3 x 3 array: a project() that took its rows for x, y and z would pass it without a word.
    rng = np.random.default_rng(n)
    pos = np.array([1.0, 2.0, 1.5])
    quat = cameras.optical_frame_xyzw((1.0, 0.3, -0.2), (0.0, 0.0, 1.0))
    points = pos + rng.uniform(-4.0, 4.0, size=(n, 3)) + np.array([5.0, 0.0, 0.0])
    got = cameras.project_points(points, pos, quat, CAM)
    assert got.shape == (n, 3)
    for row, point in zip(got, points):
        assert row == pytest.approx(cameras.project(point, pos, quat, CAM), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("points", [np.zeros(3), np.zeros((4, 2)), np.zeros((2, 3, 1))])
def test_project_points_needs_rows_of_three(points):
    with pytest.raises(ValueError, match="(N, 3)"):
        cameras.project_points(points, ORIGIN, LOOK_ALONG_X, CAM)


# ---- the view ----

def test_a_point_straight_ahead_is_in_view_and_one_behind_is_not():
    assert visibility.visible_fraction(CAM, ORIGIN, LOOK_ALONG_X, np.array([[5.0, 0.0, 0.0]]), *NO_BOXES) == 1.0
    assert visibility.visible_fraction(CAM, ORIGIN, LOOK_ALONG_X, np.array([[-5.0, 0.0, 0.0]]), *NO_BOXES) == 0.0


def test_a_point_beside_the_picture_is_out_of_view():
    # 69.4 deg wide: at 5 m ahead the picture's edge is 5 tan 34.7 deg = 3.462 m to the side.
    inside = np.array([[5.0, -3.4, 0.0]])
    outside = np.array([[5.0, -3.5, 0.0]])
    assert visibility.visible_fraction(CAM, ORIGIN, LOOK_ALONG_X, inside, *NO_BOXES) == 1.0
    assert visibility.visible_fraction(CAM, ORIGIN, LOOK_ALONG_X, outside, *NO_BOXES) == 0.0


@pytest.mark.parametrize("depth,in_view", [(0.04, False), (0.06, True), (39.9, True), (40.1, False)])
def test_the_clipping_range_bounds_the_view(depth, in_view):
    # CAM clips at 0.05 and 40 m.
    point = np.array([[depth, 0.0, 0.0]])
    assert visibility.visible_fraction(CAM, ORIGIN, LOOK_ALONG_X, point, *NO_BOXES) == float(in_view)


# ---- the occlusion ----

def test_a_wall_between_hides_and_a_wall_beside_does_not():
    point = np.array([[5.0, 0.0, 0.0]])
    assert visibility.occluded(ORIGIN, point, *_box((2.0, -1.0, -1.0), (2.2, 1.0, 1.0))).tolist() == [True]
    assert visibility.occluded(ORIGIN, point, *_box((2.0, 1.5, -1.0), (2.2, 3.0, 1.0))).tolist() == [False]


def test_a_point_inside_a_box_is_hidden():
    lo, hi = _box((4.0, -1.0, -1.0), (6.0, 1.0, 1.0))
    points = np.array([[5.0, 0.0, 0.0], [7.0, 0.0, 0.0], [3.0, 0.0, 0.0]])   # inside, beyond, in front
    assert visibility.occluded(ORIGIN, points, lo, hi).tolist() == [True, True, False]


def test_a_box_round_the_camera_is_ignored():
    # The head camera sits inside the body's box; a second box further on still hides.
    point = np.array([[5.0, 0.0, 0.0]])
    round_camera = _box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    assert visibility.occluded(ORIGIN, point, *round_camera).tolist() == [False]
    turned = visibility.Obb(center=np.zeros(3), rot_xyzw=_turned_about_z(30.0), half=np.full(3, 0.5))
    assert visibility.occluded(ORIGIN, point, *NO_BOXES, obbs=[turned]).tolist() == [False]
    both = (np.vstack([round_camera[0], [[2.0, -1.0, -1.0]]]), np.vstack([round_camera[1], [[2.2, 1.0, 1.0]]]))
    assert visibility.occluded(ORIGIN, point, *both).tolist() == [True]


def test_axis_parallel_rays_are_handled():
    # Segments along x have y and z steps of 0: the slab divides by 0 (inf, and 0/0 where a face lies
    # on the segment). No warning, and the answer of the box's extent.
    point = np.array([[5.0, 0.0, 0.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        across = visibility.occluded(ORIGIN, point, *_box((2.0, -1.0, -1.0), (2.2, 1.0, 1.0)))
        beside = visibility.occluded(ORIGIN, point, *_box((2.0, 0.5, -1.0), (2.2, 1.0, 1.0)))
        beyond = visibility.occluded(ORIGIN, point, *_box((6.0, -1.0, -1.0), (6.2, 1.0, 1.0)))
        grazing = visibility.occluded(ORIGIN, point, *_box((2.0, 0.0, -1.0), (2.2, 1.0, 1.0)))
        straight_up = visibility.occluded(ORIGIN, np.array([[0.0, 0.0, 5.0]]),
                                          *_box((-1.0, -1.0, 2.0), (1.0, 1.0, 2.2)))
        fraction = visibility.visible_fraction(CAM, ORIGIN, LOOK_ALONG_X, point,
                                               *_box((2.0, 0.5, -1.0), (2.2, 1.0, 1.0)))
    assert across.tolist() == [True] and beside.tolist() == [False] and beyond.tolist() == [False]
    assert grazing.tolist() == [True]      # a segment along a face touches the box
    assert straight_up.tolist() == [True]
    assert fraction == 1.0


def test_a_turned_box_hides_what_its_axis_aligned_bounds_would_not_decide():
    # A 0.5 m half cube at (3, 0, 0) turned 45 deg about z is a diamond |x - 3| + |y| <= 0.7071 in the
    # plane; its axis-aligned bounds reach x 3.7071 and y 0.7071. The segment from (4.6, -0.5) to
    # (3.0, 1.1) runs along x + y = 4.1, through those bounds' corner and beside the diamond's edge.
    obb = visibility.Obb(center=np.array([3.0, 0.0, 0.0]), rot_xyzw=_turned_about_z(45.0), half=np.full(3, 0.5))
    origin, point = np.array([4.6, -0.5, 0.0]), np.array([[3.0, 1.1, 0.0]])
    r = 0.5 * math.sqrt(2.0)
    assert visibility.occluded(origin, point, *_box((3.0 - r, -r, -0.5), (3.0 + r, r, 0.5))).tolist() == [True]
    assert visibility.occluded(origin, point, *NO_BOXES, obbs=[obb]).tolist() == [False]
    through_centre = np.array([[1.0, 0.0, 0.0]])
    assert visibility.occluded(np.array([5.0, 0.0, 0.0]), through_centre, *NO_BOXES, obbs=[obb]).tolist() == [True]
    # Past where the cube's face would be unturned (x 3.5) the diamond still reaches: at x 3.6 it spans |y| <= 0.107.
    across_tip = visibility.occluded(np.array([3.6, -2.0, 0.0]), np.array([[3.6, 2.0, 0.0]]), *NO_BOXES, obbs=[obb])
    assert across_tip.tolist() == [True]


def test_a_turned_box_turns_the_way_its_quaternion_says():
    # A 2 m x 0.2 m plank at (3, 0, 0) turned +30 deg about z runs along (0.866, 0.5): it crosses x 3.6 at
    # y = 0.6 tan 30 deg = 0.346. Turned -30 deg it crosses at y -0.346 and the same segment passes clear.
    def plank(deg):
        return visibility.Obb(center=np.array([3.0, 0.0, 0.0]), rot_xyzw=_turned_about_z(deg),
                              half=np.array([1.0, 0.1, 0.5]))
    origin, point = np.array([3.6, 0.2, 0.0]), np.array([[3.6, 0.5, 0.0]])
    assert visibility.occluded(origin, point, *NO_BOXES, obbs=[plank(30.0)]).tolist() == [True]
    assert visibility.occluded(origin, point, *NO_BOXES, obbs=[plank(-30.0)]).tolist() == [False]


def test_the_fraction_counts_every_sample_point():
    # Four points: two in the clear, one behind a wall, one behind the camera -> 2 / 4.
    points = np.array([[5.0, 0.0, 0.0], [5.0, 0.5, 0.0], [-5.0, 0.0, 0.0], [5.0, -0.5, 0.0]])
    wall = _box((2.0, 0.15, -1.0), (2.2, 1.0, 1.0))         # the segment to (5, 0.5) crosses it at y 0.2
    assert visibility.visible_fraction(CAM, ORIGIN, LOOK_ALONG_X, points, *wall) == 0.5


def test_a_fraction_needs_points():
    with pytest.raises(ValueError, match="표본점"):
        visibility.visible_fraction(CAM, ORIGIN, LOOK_ALONG_X, np.zeros((0, 3)), *NO_BOXES)


# ---- the height and the rule ----

def test_height_is_focal_times_stature_over_depth():
    # f_y 462.137 x 1.74 m / 8.0 m = 100.51 px; the root's height above the axis does not count.
    assert visibility.height_px(CAM, ORIGIN, LOOK_ALONG_X, np.array([8.0, 0.0, 0.9]), 1.74) == pytest.approx(
        100.51, abs=1e-2)


@pytest.mark.parametrize("root", [[-8.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 3.0, 0.0]])
def test_a_person_behind_the_camera_has_no_height(root):
    # Behind, at the near clipping plane, or beside the camera in its image plane.
    assert visibility.height_px(CAM, ORIGIN, LOOK_ALONG_X, np.array(root), 1.74) == 0.0


@pytest.mark.parametrize("fraction,height,seen", [(0.2, 50.0, True), (0.19, 100.0, False), (0.5, 49.9, False)])
def test_seen_needs_both_thresholds(fraction, height, seen):
    rule = {"min_visible_fraction": 0.2, "min_height_px": 50.0, "sample_points": 120}
    assert visibility.seen(fraction, height, rule) is seen


@pytest.mark.parametrize("edit,match", [
    (lambda c: c.pop("seeing"), "seeing"),
    (lambda c: c["seeing"].pop("min_height_px"), "min_height_px"),
    (lambda c: c["seeing"].update(min_visible_fraction=0.0), "min_visible_fraction"),
    (lambda c: c["seeing"].update(min_visible_fraction=1.2), "min_visible_fraction"),
    (lambda c: c["seeing"].update(min_visible_fraction=True), "min_visible_fraction"),
    (lambda c: c["seeing"].update(min_height_px=0), "min_height_px"),
    (lambda c: c["seeing"].update(min_height_px=float("inf")), "min_height_px"),
    (lambda c: c["seeing"].update(sample_points=0), "sample_points"),
    (lambda c: c["seeing"].update(sample_points=1.5), "sample_points"),
    (lambda c: c["seeing"].update(sample_points=True), "sample_points"),
    (lambda c: c.update(seeing=[0.2, 50, 120]), "seeing"),
])
def test_a_seeing_rule_that_cannot_be_used_is_refused(edit, match):
    canon = {"seeing": {"min_visible_fraction": 0.2, "min_height_px": 50, "sample_points": 120}}
    edit(canon)
    with pytest.raises(ValueError, match=match):
        visibility.seeing_rule(canon)


def test_robot_cameras_keep_the_canon_order():
    canon = {"cameras": {"a": {"type": "follow"}, "b": {"type": "mounted"}, "c": {"type": "fixed"},
                         "d": {"type": "mounted"}}}
    assert visibility.robot_cameras(canon) == ("b", "d")


# ---- body points in the world ----

def test_points_world_turns_and_moves_each_part_in_the_points_order():
    # part b turned 90 deg about z: local +x goes to world +y. The poses are given in another order.
    part_points = {"b": np.array([[1.0, 0.0, 0.0]]), "a": np.array([[0.0, 0.0, 1.0], [0.0, 2.0, 0.0]])}
    poses = {"a": np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
             "b": np.array([0.0, 5.0, 1.0, *_turned_about_z(90.0)])}
    got = visibility.points_world(part_points, poses)
    assert got == pytest.approx(np.array([[0.0, 6.0, 1.0], [10.0, 0.0, 1.0], [10.0, 2.0, 0.0]]), abs=1e-12)


def test_points_world_needs_every_part_pose():
    with pytest.raises(ValueError, match="b"):
        visibility.points_world({"b": np.zeros((1, 3))}, {"a": np.array([0, 0, 0, 0, 0, 0, 1.0])})


# ---- the occluders of a layout ----

def test_a_turned_collider_is_refused():
    # Its slab would be its axis-aligned bounds, a box it does not fill: hand it over as an Obb instead.
    turned = layout.Box("rack/x", (0.0, 0.0, 1.0), (1.0, 1.0, 1.0), "steel_blue", True, "rack", 10.0)
    with pytest.raises(ValueError, match="rack/x"):
        visibility.layout_occluders([turned])


# ---- the canon ----

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_SPEC = REPO_ROOT / "generation/scene/logistics_aisle.yaml"


def _camera_canon() -> dict:
    return json.loads((REPO_ROOT / "meta" / "cameras.json").read_text(encoding="utf-8"))


def test_the_seeing_rule_is_the_canons():
    canon = _camera_canon()
    assert visibility.seeing_rule(canon) == {"min_visible_fraction": 0.2, "min_height_px": 50, "sample_points": 120}
    assert canon["seeing"]["determined_by"] == "project"
    assert tuple(canon["seeing"])[:3] == visibility.SEEING_KEYS


def test_robot_cameras_are_the_mounted_ones():
    # Derived from the canon: a camera added on a robot link joins without this test changing.
    canon = _camera_canon()
    want = tuple(name for name, cam in canon["cameras"].items() if cam["type"] == "mounted")
    assert want and visibility.robot_cameras(canon) == want
    from generation import field_map
    fractions = [f.schema.rsplit(".", 1)[1] for f in field_map.STEP_FIELDS
                 if f.schema.startswith("observation.human_visible_fraction.")]
    assert tuple(fractions) == want       # the recorded fraction columns name the same cameras


def _primitives(spec) -> dict:
    return {p.name: p for p in region_split.primitives_from_anthropometry(spec.anthropometry)
            + joints_spec.foot_primitives(spec)}


def _area(p) -> float:
    # Surface area written out here: sphere 4 pi r^2, capsule 2 pi r h + 4 pi r^2 (h the cylinder), box 2(ab+bc+ca).
    if p.shape == "sphere":
        return 4 * math.pi * p.radius ** 2
    if p.shape == "capsule":
        return 2 * math.pi * p.radius * p.height + 4 * math.pi * p.radius ** 2
    a, b, c = p.size
    return 2 * (a * b + b * c + c * a)


def test_the_body_has_a_fixed_number_of_points():
    spec = joints_spec.load()
    points = visibility.body_sample_points(spec, 120)
    assert tuple(points) == joints_spec.region_prims(spec) + ("FOOT_left", "FOOT_right")
    assert sum(len(p) for p in points.values()) == 120
    prims = _primitives(spec)
    for name, local in points.items():
        assert local.shape[1] == 3 and len(local) >= 2, name
        # Local to the part's centre: moved back onto the primitive, every point is in it.
        assert (region_split.distance(local + prims[name].center, prims[name]) <= 1e-9).all(), name
    again = visibility.body_sample_points(spec, 120)
    assert all(np.array_equal(points[n], again[n]) for n in points)
    for name in points:
        if name.endswith("_left"):
            right = points[name[:-len("_left")] + "_right"]
            assert np.array_equal(right, points[name] * np.array([1.0, -1.0, 1.0])), name


def test_points_follow_part_area():
    spec = joints_spec.load()
    points = visibility.body_sample_points(spec, 120)
    prims = _primitives(spec)
    total = sum(_area(p) for p in prims.values())
    for name, local in points.items():
        if len(local) == 2:
            continue
        assert abs(len(local) - _area(prims[name]) / total * 120) <= 1, name


@pytest.mark.parametrize("total", [38, 40, 60])
def test_a_small_total_still_gives_every_part_two(total):
    # Below two a part is held at two and the rest is shared again; a larger part never gets fewer.
    spec = joints_spec.load()
    points = visibility.body_sample_points(spec, total)
    prims = _primitives(spec)
    counts = {name: len(local) for name, local in points.items()}
    assert sum(counts.values()) == total and min(counts.values()) >= 2
    if total == 38:
        assert set(counts.values()) == {2}
    for a in counts:
        for b in counts:
            if _area(prims[a]) > _area(prims[b]) + 1e-12:
                assert counts[a] >= counts[b], (a, b)


@pytest.mark.parametrize("total", [0, 37, 120.0, True])
def test_a_body_total_that_cannot_be_split_is_refused(total):
    # 19 parts with at least two points each need 38.
    with pytest.raises(ValueError, match="표본점"):
        visibility.body_sample_points(joints_spec.load(), total)


def test_layout_occluders_are_colliders_walls_and_doors():
    boxes = layout.build_layout(layout.load_spec(SCENE_SPEC))
    lo, hi = visibility.layout_occluders(boxes)
    chosen = [b for b in boxes
              if b.collider or (b.name.startswith(("wall/", "door/")) and b.rot_x_deg == 0.0)]
    assert len(lo) == len(hi) == len(chosen) == (sum(b.collider for b in boxes)
                                                + sum(b.name.startswith(("wall/", "door/")) for b in boxes))
    assert lo == pytest.approx(np.array([layout.bounds(b)[0] for b in chosen]))
    assert hi == pytest.approx(np.array([layout.bounds(b)[1] for b in chosen]))
    floor = next(b for b in boxes if b.name == "floor")
    assert not np.any(np.all(np.isclose(lo, layout.bounds(floor)[0]), axis=1))


# S1e's corner: the robot walks west along the aisle north of the north front row
# (y 5..6, its west end at x -6.045) and a person stands 0.3 m from that end on the row's south face,
# facing the rack. The head camera's height while walking is 1.68 m (the median over a 20-episode S1d test batch).
CORNER_PERSON = (-5.745, 4.65)
WALK_Y, HEAD_Z = 7.25, 1.68
WEST = cameras.optical_frame_xyzw((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def test_a_standing_person_has_every_part_at_its_centre_turned_with_the_body():
    # Facing +y (a quarter turn left), a part's standing centre (x, y, z) lands at (px - y, py + x, z) and every part
    # is turned by the same quarter turn about z -- written out here, not taken from the code under test.
    spec = joints_spec.load()
    poses = visibility.standing_part_poses(spec, (2.0, -1.0), math.pi / 2)
    prims = _primitives(spec)
    assert set(poses) == set(visibility.body_sample_points(spec, 120)) == set(prims)
    for name, p in prims.items():
        x, y, z = p.center
        assert poses[name].tolist() == pytest.approx([2.0 - y, -1.0 + x, z, *_turned_about_z(90.0)], abs=1e-12), name
    upright = visibility.standing_part_poses(spec, (0.0, 0.0), 0.0)
    assert all(pose[:3].tolist() == list(prims[n].center) for n, pose in upright.items())
    with pytest.raises(ValueError, match="요"):
        visibility.standing_part_poses(spec, (0.0, 0.0), float("nan"))


def _corner_fractions(scene_spec) -> list[tuple[float, float]]:
    spec = joints_spec.load()
    body = visibility.points_world(visibility.body_sample_points(spec, 120),
                                   visibility.standing_part_poses(spec, CORNER_PERSON, math.radians(90.0)))
    lo, hi = visibility.layout_occluders(layout.build_layout(scene_spec))
    head = _camera_canon()["cameras"]["head_rgb"]
    out = []
    for x in np.arange(8.0, -6.0 - 1e-9, -0.25):
        out.append((float(x), visibility.visible_fraction(head, np.array([x, WALK_Y, HEAD_Z]), WEST, body, lo, hi)))
    return out


def test_with_floor_loads_a_person_at_the_corner_stays_hidden_before_the_turn():
    fractions = _corner_fractions(layout.load_spec(SCENE_SPEC))
    assert len(fractions) == 57
    assert max(f for _, f in fractions) < 0.2, fractions


def test_without_floor_loads_the_same_person_shows_from_far_off():
    scene = copy.deepcopy(layout.load_spec(SCENE_SPEC))
    scene["rack"]["floor_loads"]["per_bay"] = 0
    fractions = dict(_corner_fractions(layout.validate_spec(scene)))
    assert fractions[8.0] >= 0.2, fractions        # about 14 m from the corner
