"""The robot's own body for the "seeing" rule (generation/robots/extract_link_boxes.py and the loader and
placement in generation/visibility.py): one axis-aligned box per link in the link's frame, measured once from
the rendered H1 meshes. Boxes, meshes and poses worked out by hand here; the canon file is read in the last
section. Isaac-free -- the extractor's geometry is numpy only."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from generation import cameras, visibility
from generation.robots import extract_link_boxes as elb

NO_BOXES = (np.zeros((0, 3)), np.zeros((0, 3)))
REPO_ROOT = Path(__file__).resolve().parents[2]
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
YAW_90 = np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])


def _mesh(points):
    """(points, face vertex counts, face vertex indices), as rendered_meshes hands one over -- one triangle
    fan over the points; only the points matter to a box."""
    pts = np.asarray(points, float)
    faces = [(0, i, i + 1) for i in range(1, len(pts) - 1)] or [(0, 0, 0)]
    return pts, np.full(len(faces), 3), np.array([i for f in faces for i in f])


def _document(links, **extra):
    doc = {"robot": "unitree_h1", "usd_sha256": "ab" * 32, "frame": "link", "note": "x", "links": links}
    doc.update(extra)
    return doc


def _entry(center=(0.0, 0.0, 0.0), half=(0.1, 0.1, 0.1)):
    return {"center": list(center), "half_extents": list(half)}


def _load(tmp_path, doc):
    path = tmp_path / "boxes.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return visibility.load_link_boxes(path)


# ---- measuring a link ----

def test_a_link_box_bounds_every_point():
    # x -1..1, y 0..2, z 0..3: centre (0, 1, 1.5), half sizes (1, 1, 1.5)
    points = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (-1.0, 0.5, 0.25), (0.3, 1.1, 2.9)]
    centre, half = elb.link_box(points)
    assert centre.tolist() == [0.0, 1.0, 1.5]
    assert half.tolist() == [1.0, 1.0, 1.5]
    pts = np.array(points)
    assert np.all(pts >= centre - half) and np.all(pts <= centre + half)


def test_a_link_box_is_rounded_to_the_micrometre_and_stays_within_it_of_the_points():
    # x 0.1234564..0.2 -> centre 0.1617282, half 0.0382718: written as 0.161728 and 0.038272
    centre, half = elb.link_box([(0.1234564, 0.0, 0.0), (0.2, 1.0, 1.0)])
    assert centre.tolist() == [0.161728, 0.5, 0.5]
    assert half.tolist() == [0.038272, 0.5, 0.5]
    assert 0.1234564 >= centre[0] - half[0] - 1e-6 and 0.2 <= centre[0] + half[0] + 1e-6


def test_a_box_edge_at_zero_is_not_written_as_minus_zero():
    # x, y, z -8e-7..0: the centre -4e-7 rounds to -0.0 and the half size 4e-7 to 0.0
    centre, half = elb.link_box([(-8e-7, -8e-7, -8e-7), (0.0, 0.0, 0.0)])
    assert [repr(float(v)) for v in centre] == ["0.0", "0.0", "0.0"]
    assert [repr(float(v)) for v in half] == ["0.0", "0.0", "0.0"]


@pytest.mark.parametrize("points", [np.zeros((0, 3)), np.zeros((4, 2)), np.zeros(3), [(0.0, math.nan, 0.0)],
                                    [(0.0, 0.0, math.inf), (1.0, 1.0, 1.0)]])
def test_points_that_make_no_box_are_refused(points):
    with pytest.raises(ValueError, match="점"):
        elb.link_box(points)


def test_a_links_box_covers_all_of_its_meshes():
    meshes = {"arm": [_mesh([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
                      _mesh([(0.0, 0.0, -2.0), (0.0, 0.5, 0.0), (-3.0, 0.0, 0.0)])]}
    boxes = elb.boxes_from_meshes(meshes)
    centre, half = boxes["arm"]
    assert centre.tolist() == [-1.0, 0.5, -1.0]       # x -3..1, y 0..1, z -2..0
    assert half.tolist() == [2.0, 0.5, 1.0]


def test_the_opened_bytes_of_a_path_are_the_file_s_own(tmp_path):
    # A remote address goes through omni.client (Isaac python only); a path is read from disk.
    asset = tmp_path / "h1.usd"
    asset.write_bytes(b"\x00USD\xff")
    assert elb.opened_bytes(str(asset)) == b"\x00USD\xff"


# ---- the file the extractor writes, and the comparison with it ----

def test_the_document_has_the_shape_the_loader_reads(tmp_path):
    boxes = {"b_link": (np.array([0.0, 0.0, 0.5]), np.array([0.1, 0.2, 0.3])),
             "a_link": (np.array([1.0, 0.0, 0.0]), np.array([0.05, 0.05, 0.05]))}
    doc = elb.document("cd" * 32, boxes)
    assert list(doc) == ["robot", "usd_sha256", "frame", "note", "links"]
    assert (doc["robot"], doc["usd_sha256"], doc["frame"]) == ("unitree_h1", "cd" * 32, "link")
    assert list(doc["links"]) == ["a_link", "b_link"]          # sorted by link name, whatever the order handed in
    assert doc["links"]["b_link"] == {"center": [0.0, 0.0, 0.5], "half_extents": [0.1, 0.2, 0.3]}
    text = elb.serialize(doc)
    assert text.endswith("}\n") and not text.endswith("\n\n")
    assert text == json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    loaded = _load(tmp_path, json.loads(text))
    assert loaded["b_link"][1].tolist() == [0.1, 0.2, 0.3]


def test_a_comparison_passes_within_a_micrometre_and_names_what_differs_beyond_it():
    canon = _document({"a": _entry((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)), "b": _entry((1.0, 0.0, 0.0), (0.2, 0.2, 0.2))})
    same = json.loads(json.dumps(canon))
    assert elb.compare(canon, same) == []
    near = json.loads(json.dumps(canon))
    near["links"]["a"]["center"][0] = 1e-6                    # exactly the tolerance: not beyond it
    assert elb.compare(canon, near) == []
    far = json.loads(json.dumps(canon))
    far["links"]["b"]["half_extents"][2] = 0.200002           # 2 micrometres
    problems = elb.compare(canon, far)
    assert len(problems) == 1 and "b" in problems[0] and "half_extents" in problems[0]


def test_a_comparison_says_a_link_that_is_gone_or_new_and_a_changed_robot_file():
    canon = _document({"a": _entry(), "b": _entry()})
    fresh = _document({"a": _entry(), "c": _entry()}, usd_sha256="ef" * 32)
    text = "\n".join(elb.compare(canon, fresh))
    assert "b" in text and "c" in text and "usd_sha256" in text


# ---- reading the file ----

def test_link_boxes_load_as_centre_and_half_extents(tmp_path):
    boxes = _load(tmp_path, _document({"torso_link": _entry((0.0, 0.0, 0.3), (0.1, 0.15, 0.3))}))
    assert list(boxes) == ["torso_link"]
    centre, half = boxes["torso_link"]
    assert centre.tolist() == [0.0, 0.0, 0.3] and half.tolist() == [0.1, 0.15, 0.3]


@pytest.mark.parametrize("edit,match", [
    (lambda d: d["links"].__setitem__("a", _entry(half=(0.1, 0.0, 0.1))), "half_extents"),      # flat
    (lambda d: d["links"].__setitem__("a", _entry(half=(0.1, -0.1, 0.1))), "half_extents"),
    (lambda d: d["links"].__setitem__("a", _entry(half=(0.1, math.nan, 0.1))), "half_extents"),
    (lambda d: d["links"].__setitem__("a", _entry(half=(0.1, math.inf, 0.1))), "half_extents"),
    (lambda d: d["links"].__setitem__("a", _entry(center=(0.0, math.nan, 0.0))), "center"),
    (lambda d: d["links"].__setitem__("a", _entry(center=(0.0, 0.0))), "center"),
    (lambda d: d["links"].__setitem__("a", _entry(center=(True, 0.0, 0.0))), "center"),
    (lambda d: d["links"].__setitem__("a", {"center": [0.0, 0.0, 0.0]}), "half_extents"),
    (lambda d: d["links"].__setitem__("a", [0.0, 0.0, 0.0]), "객체"),
    (lambda d: d.__setitem__("links", {}), "links"),
    (lambda d: d.__setitem__("links", [1, 2]), "links"),
    (lambda d: d.__setitem__("frame", "world"), "frame"),
    (lambda d: d.pop("links"), "links"),
])
def test_link_boxes_load_refuses_non_finite_or_flat(tmp_path, edit, match):
    doc = _document({"a": _entry()})
    edit(doc)
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, doc)


def test_link_boxes_load_refuses_a_file_that_is_not_an_object(tmp_path):
    path = tmp_path / "boxes.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="객체"):
        visibility.load_link_boxes(path)


# ---- placing the boxes ----

def test_robot_obbs_place_each_box_by_its_link_pose():
    boxes = {"hand": (np.array([0.1, 0.0, 0.2]), np.array([0.05, 0.1, 0.15])),
             "foot": (np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.05, 0.05]))}
    poses = {"foot": np.array([5.0, 6.0, 0.0, *IDENTITY]),
             "hand": np.array([1.0, 2.0, 3.0, *IDENTITY]),
             "head": np.array([9.0, 9.0, 9.0, *IDENTITY])}      # a link with no box: no box
    obbs = visibility.robot_obbs(boxes, poses)
    assert len(obbs) == 2
    hand, foot = obbs                                             # in the boxes' order
    assert hand.center.tolist() == pytest.approx([1.1, 2.0, 3.2])
    assert hand.rot_xyzw.tolist() == IDENTITY.tolist()
    assert hand.half.tolist() == [0.05, 0.1, 0.15]
    assert foot.center.tolist() == [5.0, 6.0, 0.0]

    # Yawed 90 deg about z, the link's +x is the world's +y and its +y the world's -x: the box's centre
    # (0.1, 0, 0.2) in the link is (0, 0.1, 0.2) from the link's origin, and the half sizes stay with the link's axes.
    turned = visibility.robot_obbs(boxes, {**poses, "hand": np.array([1.0, 2.0, 3.0, *YAW_90])})[0]
    assert turned.center.tolist() == pytest.approx([1.0, 2.1, 3.2])
    assert turned.rot_xyzw.tolist() == pytest.approx(YAW_90.tolist())
    assert turned.half.tolist() == [0.05, 0.1, 0.15]


def test_a_placed_box_hides_what_it_stands_over_and_turns_with_its_link():
    box = {"hand": (np.array([0.1, 0.0, 0.2]), np.array([0.05, 0.1, 0.15]))}
    # A segment along x at y = 2.13: the box turned 90 deg spans y 2.05..2.15 there (its link-x half 0.05 lies
    # along y), so it is crossed; unturned it spans y 1.9..2.1 and is passed.
    origin, point = np.array([0.5, 2.13, 3.2]), np.array([[1.5, 2.13, 3.2]])
    turned = visibility.robot_obbs(box, {"hand": np.array([1.0, 2.0, 3.0, *YAW_90])})
    flat = visibility.robot_obbs(box, {"hand": np.array([1.0, 2.0, 3.0, *IDENTITY])})
    assert visibility.occluded(origin, point, *NO_BOXES, obbs=turned).tolist() == [True]
    assert visibility.occluded(origin, point, *NO_BOXES, obbs=flat).tolist() == [False]


def test_a_link_without_a_pose_is_refused():
    boxes = {"hand": (np.zeros(3), np.full(3, 0.1)), "foot": (np.zeros(3), np.full(3, 0.1))}
    with pytest.raises(ValueError, match="foot"):
        visibility.robot_obbs(boxes, {"hand": np.array([0.0, 0.0, 0.0, *IDENTITY])})


@pytest.mark.parametrize("pose", [np.zeros(6), np.zeros((2, 7)), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]),
                                  np.array([0.0, math.nan, 0.0, *IDENTITY]), np.zeros(7)])
def test_a_pose_that_is_not_a_position_and_a_unit_quaternion_is_refused(pose):
    boxes = {"hand": (np.zeros(3), np.full(3, 0.1))}
    with pytest.raises(ValueError, match="hand"):
        visibility.robot_obbs(boxes, {"hand": pose})


def test_the_float32_quaternion_the_engine_hands_back_is_taken():
    # body_link_pose_w is float32: a quaternion unit to 1e-7 would be refused by Obb's 1e-6 check once widened and
    # drifted a little; it is renormalised, not refused.
    q = np.array([0.0, 0.0, math.sin(0.3), math.cos(0.3)]) * (1.0 + 3e-6)
    obb = visibility.robot_obbs({"hand": (np.zeros(3), np.full(3, 0.1))}, {"hand": np.array([0.0, 0.0, 0.0, *q])})[0]
    assert float(np.linalg.norm(obb.rot_xyzw)) == pytest.approx(1.0, abs=1e-12)


# ---- a camera's own link ----

def _boxes_and_poses(names):
    """Boxes of different sizes for the named links, each link posed 1 m further along x, unturned."""
    boxes = {n: (np.zeros(3), np.full(3, 0.1 * (i + 1))) for i, n in enumerate(names)}
    poses = {n: np.array([float(i), 0.0, 0.0, *IDENTITY]) for i, n in enumerate(names)}
    return boxes, poses


def test_exclude_drops_exactly_the_named_link():
    boxes, poses = _boxes_and_poses(["a", "b", "c"])
    kept = visibility.robot_obbs(boxes, poses, exclude=["b"])
    assert [o.center.tolist() for o in kept] == [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    assert [o.half[0] for o in kept] == pytest.approx([0.1, 0.3])
    assert len(visibility.robot_obbs(boxes, poses)) == 3                       # nothing excluded by default
    assert len(visibility.robot_obbs(boxes, poses, exclude=())) == 3
    assert len(visibility.robot_obbs(boxes, poses, exclude=iter(["a", "c"]))) == 1        # any iterable


def test_an_excluded_link_needs_no_pose():
    boxes, poses = _boxes_and_poses(["a", "b"])
    del poses["b"]
    assert len(visibility.robot_obbs(boxes, poses, exclude=["b"])) == 1
    with pytest.raises(ValueError, match="b"):
        visibility.robot_obbs(boxes, poses)


@pytest.mark.parametrize("exclude", [["nope"], ["a", "nope"], ["pose_only"], "a"])
def test_a_name_in_exclude_that_is_not_a_link_with_a_box_is_refused(exclude):
    # "pose_only" has a pose but no box; the bare string "a" is a list of characters, not of links.
    boxes, poses = _boxes_and_poses(["a", "b"])
    poses["pose_only"] = np.array([5.0, 0.0, 0.0, *IDENTITY])
    with pytest.raises(ValueError, match="exclude"):
        visibility.robot_obbs(boxes, poses, exclude=exclude)


def _mounted_cameras():
    canon = json.loads((REPO_ROOT / "meta" / "cameras.json").read_text(encoding="utf-8"))
    return canon, {n: c["parent_prim"] for n, c in canon["cameras"].items() if c.get("type") == "mounted"}


def test_camera_robot_obbs_drops_the_link_each_mounted_camera_sits_on():
    canon, parents = _mounted_cameras()
    assert parents                                                  # the canon has cameras on the robot's links
    boxes = visibility.load_link_boxes()
    poses = {n: np.array([float(i), 0.0, 0.0, *IDENTITY]) for i, n in enumerate(boxes)}     # a place each
    everything = visibility.robot_obbs(boxes, poses)
    for camera, parent in parents.items():
        mine = visibility.camera_robot_obbs(canon, camera, boxes, poses)
        want = [o for name, o in zip(boxes, everything) if name != parent]
        assert len(mine) == len(boxes) - 1, camera
        assert [o.center.tolist() for o in mine] == [o.center.tolist() for o in want], camera
        assert [o.half.tolist() for o in mine] == [o.half.tolist() for o in want], camera


def test_camera_robot_obbs_refuses_a_camera_that_is_not_on_a_link():
    canon, parents = _mounted_cameras()
    others = [n for n, c in canon["cameras"].items() if c.get("type") != "mounted"]
    assert others                                                   # the follow and fixed cameras
    boxes = visibility.load_link_boxes()
    poses = {n: np.array([0.0, 0.0, 0.0, *IDENTITY]) for n in boxes}
    for camera in others:
        with pytest.raises(ValueError, match=camera):
            visibility.camera_robot_obbs(canon, camera, boxes, poses)
    with pytest.raises(ValueError, match="no_such_camera"):
        visibility.camera_robot_obbs(canon, "no_such_camera", boxes, poses)


def test_a_camera_that_names_a_parent_link_but_is_not_mounted_is_refused_by_its_type():
    # The type decides, not whether a parent_prim happens to be written.
    canon, parents = _mounted_cameras()
    canon = json.loads(json.dumps(canon))
    follow = next(n for n, c in canon["cameras"].items() if c.get("type") == "follow")
    canon["cameras"][follow]["parent_prim"] = next(iter(parents.values()))
    boxes = visibility.load_link_boxes()
    poses = {n: np.array([0.0, 0.0, 0.0, *IDENTITY]) for n in boxes}
    with pytest.raises(ValueError, match="type"):
        visibility.camera_robot_obbs(canon, follow, boxes, poses)


def test_a_camera_on_a_link_without_a_box_is_refused():
    canon, parents = _mounted_cameras()
    camera = next(iter(parents))
    boxes = {n: b for n, b in visibility.load_link_boxes().items() if n != parents[camera]}
    poses = {n: np.array([0.0, 0.0, 0.0, *IDENTITY]) for n in boxes}
    with pytest.raises(ValueError, match=parents[camera]):
        visibility.camera_robot_obbs(canon, camera, boxes, poses)


# ---- the canon ----

NAMED_CAMERA_PARENTS = {"torso_link", "left_elbow_link", "right_elbow_link"}


def _canon(name):
    return json.loads((REPO_ROOT / "meta" / name).read_text(encoding="utf-8"))


def test_the_canon_link_boxes_are_the_robot_the_canon_pins():
    boxes = visibility.load_link_boxes()          # as committed: finite, every half size above zero
    doc = json.loads(visibility.LINK_BOXES.read_text(encoding="utf-8"))
    assert doc["robot"] == "unitree_h1"
    assert doc["usd_sha256"] == _canon("assets.json")["robots"][doc["robot"]]["usd_sha256"]
    assert len(boxes) >= 15
    # The links the robot's cameras sit on: derived from the camera canon, and the three the canon has today.
    parents = {c["parent_prim"] for c in _canon("cameras.json")["cameras"].values() if c.get("type") == "mounted"}
    assert NAMED_CAMERA_PARENTS <= parents <= set(boxes)


def test_the_canon_file_is_what_the_extractor_writes():
    assert elb.LINK_BOXES == visibility.LINK_BOXES == REPO_ROOT / "generation" / "robots" / "unitree_h1_link_boxes.json"
    text = visibility.LINK_BOXES.read_text(encoding="utf-8")
    doc = json.loads(text)
    assert text == elb.serialize(doc)                       # key order, indentation and the trailing newline
    assert list(doc["links"]) == sorted(doc["links"])
    assert list(doc) == ["robot", "usd_sha256", "frame", "note", "links"]
    assert doc["frame"] == visibility.LINK_BOXES_FRAME


def test_the_canon_boxes_are_in_metres_and_the_size_of_a_humanoid_s_links():
    # A box in centimetres, or in another link's frame, would sit far outside the H1: 1.8 m tall, no link longer
    # than a leg. Half sizes under a metre, centres within a metre of the link's origin.
    for link, (centre, half) in visibility.load_link_boxes().items():
        assert half.max() < 1.0 and np.abs(centre).max() < 1.0, link


def _blocked_share(canon, camera, obbs):
    """The share of a 33 x 25 grid of the camera's picture whose ray, to the point 3 m out along it, is
    blocked by ``obbs``. The camera sits at its canon offset in its link's frame; the link is at the origin."""
    cam = canon["cameras"][camera]
    pos, quat = cameras.mounted_pose(np.array([0.0, 0.0, 0.0, *IDENTITY]), cam["offset_pos"], cam["offset_rot_xyzw"])
    fx, fy, cx, cy = cameras.intrinsics(cam)
    us, vs = np.meshgrid(np.linspace(0.0, cam["width"], 33), np.linspace(0.0, cam["height"], 25))
    rays = np.stack([(us.ravel() - cx) / fx, (vs.ravel() - cy) / fy, np.ones(us.size)], axis=1) * 3.0
    points = pos + np.array([cameras._q_rot(quat, r) for r in rays])
    return float(visibility.occluded(pos, points, *NO_BOXES, obbs=obbs).mean())


def test_a_wrist_camera_does_not_lose_its_picture_to_its_own_forearm_box():
    # The link at its origin is the one place its box can be set without the engine: the offsets of a mounted
    # camera are in its link's frame. Counting the camera's own link, a wrist camera 18 mm above the forearm's
    # box and tilted down through it has 45 % of its picture behind that box; without it, none.
    canon, parents = _mounted_cameras()
    boxes = visibility.load_link_boxes()
    at_origin = {n: np.array([0.0, 0.0, 0.0, *IDENTITY]) for n in boxes}
    seen = {}
    for camera, parent in parents.items():
        own = {parent: boxes[parent]}
        before = _blocked_share(canon, camera, visibility.robot_obbs(own, at_origin))
        after = _blocked_share(canon, camera, visibility.camera_robot_obbs(canon, camera, own, at_origin))
        seen[camera] = (before, after)
        assert after < before or before == 0.0, camera
        assert after == 0.0, camera
    assert max(before for before, _ in seen.values()) > 0.3         # the wrist cameras: the reason for the rule
