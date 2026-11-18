"""The H1's forward kinematics without Isaac (generation/robots/kinematics.py) and the joint frames it runs on
(generation/robots/unitree_h1_joints.json, written by generation/robots/extract_joint_frames.py).

Two kinds of test. Hand-built robots of two or three links, worked out on paper here, pin the composition rule
the physics engine uses for a joint (child = parent + joint frame 0 + the turn + joint frame 1 inverted). The
canon file is then held against things that do not come from this code: the links' stored poses in the asset,
the joint map, the pinned h1.usd, and the forearm-tip place a physics probe measured on the walking H1."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from generation import visibility
from generation.robots import extract_joint_frames as ejf
from generation.robots import kinematics as kin

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "meta" / "assets.json"
JOINT_MAP = REPO_ROOT / "meta" / "joint_map" / "unitree_h1.yaml"
JOINTS_JSON = REPO_ROOT / "generation" / "robots" / "unitree_h1_joints.json"

IDENTITY = [0.0, 0.0, 0.0, 1.0]
ZERO3 = [0.0, 0.0, 0.0]
Z_90 = [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]     # a quarter turn about +z, xyzw
HOLD = {"shoulder_pitch": 0.28, "shoulder_roll": 0.0, "shoulder_yaw": 0.0, "elbow": 0.52}


def rotation_gap(qa, qb) -> float:
    """Angle (rad) of the turn that takes one unit quaternion to the other, whatever their signs.
    atan2 form: acos of a dot product near 1 cannot see a gap below about 1e-8."""
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    qa, qb = qa / np.linalg.norm(qa), qb / np.linalg.norm(qb)
    inv_a = np.array([-qa[0], -qa[1], -qa[2], qa[3]])
    x1, y1, z1, w1 = inv_a
    x2, y2, z2, w2 = qb
    rel = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
    return 2.0 * math.atan2(float(np.linalg.norm(rel[:3])), abs(float(rel[3])))


def _joint(kind, parent, child, axis="Z", pos0=ZERO3, rot0=IDENTITY, pos1=ZERO3, rot1=IDENTITY):
    return {"type": kind, "parent": parent, "child": child, "axis": axis if kind == "revolute" else None,
            "pos0": list(pos0), "rot0_xyzw": list(rot0), "pos1": list(pos1), "rot1_xyzw": list(rot1)}


def _document(joints, root="a", **extra):
    """A joint document; ``rest`` holds every link at the origin because the kinematics never read it."""
    links = {root} | {j["child"] for j in joints.values()}
    doc = {"robot": "toy", "usd_sha256": "ab" * 32, "root_link": root, "joints": joints,
           "rest": {link: {"pos": list(ZERO3), "rot_xyzw": list(IDENTITY)} for link in sorted(links)}, "note": "x"}
    doc.update(extra)
    return doc


def _toy(**kw):
    """a --(spin about z, 1 m out along x)--> b --(fixed, 1 m out along b's x)--> c."""
    return kin.joints_from_document(_document({
        "spin": _joint("revolute", "a", "b", "Z", pos0=[1.0, 0.0, 0.0], **kw),
        "rigid": _joint("fixed", "b", "c", pos0=[1.0, 0.0, 0.0]),
    }))


# ---- the composition rule, on paper ----

def test_a_robot_at_rest_stands_where_its_frames_put_it():
    poses = kin.link_poses(_toy(), {}, frame="a")
    assert poses["a"][0].tolist() == [0.0, 0.0, 0.0]
    assert poses["b"][0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert poses["c"][0] == pytest.approx([2.0, 0.0, 0.0], abs=1e-12)


def test_a_quarter_turn_swings_what_hangs_below_the_joint():
    # b turns 90 deg about +z at (1, 0, 0); c sits 1 m along b's own x, which is now +y
    poses = kin.link_poses(_toy(), {"spin": math.pi / 2}, frame="a")
    assert poses["b"][0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert poses["c"][0] == pytest.approx([1.0, 1.0, 0.0], abs=1e-12)
    assert rotation_gap(poses["c"][1], Z_90) < 1e-12


def test_the_joints_second_frame_is_taken_off_again():
    # the child's own origin is pos1 behind the joint frame: a 0.5 m offset along the child's x. At rest the child
    # sits at 1 - 0.5; turned a quarter, the joint frame is still at (1, 0, 0) and the child's x is +y
    toy = kin.joints_from_document(_document({
        "spin": _joint("revolute", "a", "b", "Z", pos0=[1.0, 0.0, 0.0], pos1=[0.5, 0.0, 0.0])}))
    assert kin.link_poses(toy, {}, frame="a")["b"][0] == pytest.approx([0.5, 0.0, 0.0], abs=1e-12)
    turned = kin.link_poses(toy, {"spin": math.pi / 2}, frame="a")["b"][0]
    assert turned == pytest.approx([1.0, -0.5, 0.0], abs=1e-12)


def test_the_turn_is_about_the_joints_own_axis_not_the_parents():
    # the joint's first frame is rolled a quarter turn about +x, so the joint's z axis is the parent's -y. A quarter
    # turn about the parent's -y sends +x to +z (a quarter turn about +y would send +x to -z)
    rolled = [math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)]
    toy = kin.joints_from_document(_document({
        "j": _joint("revolute", "a", "b", "Z", rot0=rolled), "k": _joint("fixed", "b", "c", pos0=[1.0, 0.0, 0.0])}))
    assert kin.link_poses(toy, {"j": math.pi / 2}, frame="a")["c"][0] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)


def test_the_second_frame_may_be_turned_too():
    # the child is placed so that its own frame (pos1, rot1) lands on the joint frame: here that frame is the child's
    # origin moved 1 m along the child's x and turned a quarter about +z, so at q = 0 the child is the inverse
    toy = kin.joints_from_document(_document({
        "j": _joint("revolute", "a", "b", "Z", pos1=[1.0, 0.0, 0.0], rot1=Z_90)}))
    pos, quat = kin.link_poses(toy, {}, frame="a")["b"]
    assert rotation_gap(quat, [0.0, 0.0, -Z_90[2], Z_90[3]]) < 1e-12
    assert pos == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    # and the joint frame, found again from the child, is where it was: the origin
    assert pos + kin.quat_to_matrix(quat) @ np.array([1.0, 0.0, 0.0]) == pytest.approx(ZERO3, abs=1e-12)


def test_a_joint_turns_about_the_axis_it_names():
    for axis, point in (("X", [0.0, 1.0, 0.0]), ("Y", [0.0, 0.0, 1.0]), ("Z", [1.0, 0.0, 0.0])):
        toy = kin.joints_from_document(_document({
            "j": _joint("revolute", "a", "b", axis), "k": _joint("fixed", "b", "c", pos0=point)}))
        # a quarter turn about +x sends +y to +z, about +y sends +z to +x, about +z sends +x to +y
        want = {"X": [0.0, 0.0, 1.0], "Y": [1.0, 0.0, 0.0], "Z": [0.0, 1.0, 0.0]}[axis]
        got = kin.link_poses(toy, {"j": math.pi / 2}, frame="a")["c"][0]
        assert got == pytest.approx(want, abs=1e-12), axis


def test_poses_can_be_read_in_another_links_frame():
    poses = kin.link_poses(_toy(), {"spin": math.pi / 2}, frame="b")
    assert poses["b"][0] == pytest.approx(ZERO3, abs=1e-12)
    assert rotation_gap(poses["b"][1], IDENTITY) < 1e-12
    # c is 1 m along b's own x
    assert poses["c"][0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    # a sits at (-1, 0, 0) in b's parent's frame turned a quarter: 1 m along b's +y, and is turned the other way
    assert poses["a"][0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert rotation_gap(poses["a"][1], [0.0, 0.0, -Z_90[2], Z_90[3]]) < 1e-12


def test_a_link_keeps_its_turn_relative_to_the_frame_it_is_read_in():
    # b hangs from a rolled a quarter turn about +x (a turn that does not commute with a turn about z); c is fixed
    # to b with a quarter turn about b's own z and sits 1 m along b's x. Read in b's frame, c is exactly that
    # fixed pose whatever b's own turn is
    rolled = [math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)]
    robot = kin.joints_from_document(_document({
        "j": _joint("revolute", "a", "b", "Z", rot0=rolled),
        "k": _joint("fixed", "b", "c", pos0=[1.0, 0.0, 0.0], rot0=Z_90)}))
    for angle in (0.0, 0.3):
        poses = kin.link_poses(robot, {"j": angle}, frame="b")
        assert poses["c"][0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12), angle
        assert rotation_gap(poses["c"][1], Z_90) < 1e-12, angle
    # with the joint at 0, a is the roll the other way
    back = kin.link_poses(robot, {}, frame="b")["a"][1]
    assert rotation_gap(back, [-math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)]) < 1e-12


def test_the_default_frame_is_the_torso_and_a_robot_without_one_is_refused():
    with pytest.raises(ValueError, match="torso_link"):
        kin.link_poses(_toy(), {})


def test_an_unknown_joint_is_refused():
    with pytest.raises(ValueError, match="no_such_joint"):
        kin.link_poses(_toy(), {"no_such_joint": 0.1}, frame="a")
    with pytest.raises(ValueError, match="rigid"):             # a fixed joint has no angle
        kin.link_poses(_toy(), {"rigid": 0.1}, frame="a")
    with pytest.raises(ValueError, match="no_such_link"):
        kin.link_poses(_toy(), {}, frame="no_such_link")


@pytest.mark.parametrize("angle", [float("nan"), float("inf"), "0.1", None])
def test_an_angle_that_is_not_a_finite_number_is_refused(angle):
    with pytest.raises(ValueError, match="spin"):
        kin.link_poses(_toy(), {"spin": angle}, frame="a")


# ---- a joint document that is not a tree is refused ----

def _add_link(doc, name):
    doc["rest"][name] = {"pos": list(ZERO3), "rot_xyzw": list(IDENTITY)}


def _a_link_no_joint_reaches(doc):
    _add_link(doc, "orphan")


def _a_link_with_two_parents(doc):
    doc["joints"]["again"] = _joint("fixed", "a", "c", pos0=[0.0, 1.0, 0.0])


def _a_cycle_off_the_tree(doc):
    _add_link(doc, "x")
    _add_link(doc, "y")
    doc["joints"]["jx"] = _joint("fixed", "x", "y")
    doc["joints"]["jy"] = _joint("fixed", "y", "x")


@pytest.mark.parametrize("change, match", [
    (_a_link_no_joint_reaches, "닿지 않는"),
    (_a_link_with_two_parents, "관절 둘"),
    (_a_cycle_off_the_tree, "닿지 않는"),
    (lambda d: d["joints"].update(back=_joint("fixed", "c", "a")), "뿌리 링크"),
    (lambda d: d["joints"]["spin"].update(parent="nowhere"), "parent"),
    (lambda d: d["joints"]["spin"].update(axis="W"), "axis"),
    (lambda d: d["joints"]["rigid"].update(axis="Z"), "axis"),
    (lambda d: d["joints"]["spin"].update(type="prismatic"), "type"),
    (lambda d: d["joints"]["spin"].update(rot0_xyzw=[0, 0, 1]), "rot0"),
    (lambda d: d["joints"]["spin"].update(pos0=[0.0, float("nan"), 0.0]), "pos0"),
    (lambda d: d["joints"]["spin"].update(pos1="here"), "pos1"),
    (lambda d: d["joints"]["spin"].update(rot1_xyzw=[0, 0, 0, 0]), "rot1"),
    (lambda d: d["joints"]["spin"].update(rot0_xyzw=[0, 0, 1, 1]), "rot0"),        # two long, not a unit quaternion
    (lambda d: d.update(root_link="nowhere"), "root_link"),
    (lambda d: d.pop("usd_sha256"), "usd_sha256"),
])
def test_a_document_that_is_not_a_tree_of_frames_is_refused(change, match):
    doc = _document({"spin": _joint("revolute", "a", "b", "Z", pos0=[1.0, 0.0, 0.0]),
                     "rigid": _joint("fixed", "b", "c", pos0=[1.0, 0.0, 0.0])})
    change(doc)
    with pytest.raises(ValueError, match=match):
        kin.joints_from_document(doc)


def test_a_file_is_read_through_the_same_checks(tmp_path):
    good = _document({"spin": _joint("revolute", "a", "b", "Z")})
    path = tmp_path / "joints.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    assert kin.load_joints(path)["root_link"] == "a"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="객체"):
        kin.load_joints(path)


# ---- the canon file: held against what this code did not produce ----

@pytest.fixture(scope="module")
def h1():
    return kin.load_joints()


@pytest.fixture(scope="module")
def boxes():
    return visibility.load_link_boxes()


def test_zero_pose_reproduces_the_assets_rest_links(h1):
    raw = json.loads(JOINTS_JSON.read_text(encoding="utf-8"))["rest"]       # the asset's stored poses, as written
    poses = kin.link_poses(h1, {}, frame=h1["root_link"])
    assert sorted(poses) == sorted(raw)
    for link, stored in raw.items():
        pos, quat = poses[link]
        assert np.max(np.abs(pos - np.array(stored["pos"]))) < 1e-6, link
        assert rotation_gap(quat, stored["rot_xyzw"]) < 1e-6, link


def test_the_joint_names_are_the_joint_maps(h1):
    names = yaml.safe_load(JOINT_MAP.read_text(encoding="utf-8"))["joint_names"]
    revolute = [name for name, joint in h1["joints"].items() if joint["type"] == "revolute"]
    assert sorted(revolute) == sorted(names)
    assert len(revolute) == len(set(revolute)) == 19


def test_the_frames_name_the_pinned_robot(h1):
    pinned = json.loads(ASSETS.read_text(encoding="utf-8"))["robots"]["unitree_h1"]["usd_sha256"]
    assert h1["usd_sha256"] == pinned
    assert h1["robot"] == "unitree_h1"
    assert h1["root_link"] == "pelvis"


def test_the_torso_link_sits_at_the_pelvis_origin(h1):
    pos, quat = kin.link_poses(h1, {}, frame="pelvis")["torso_link"]
    assert pos == pytest.approx(ZERO3, abs=1e-6)
    assert rotation_gap(quat, IDENTITY) < 1e-6


def _hold_tips(h1, boxes, elbow=HOLD["elbow"], pitch=HOLD["shoulder_pitch"]):
    q = {f"{side}_{joint}": {"shoulder_pitch": pitch, "elbow": elbow}.get(joint, 0.0)
         for side in ("left", "right") for joint in HOLD}
    poses = kin.link_poses(h1, q)
    return kin.forearm_tip(boxes, poses, "left"), kin.forearm_tip(boxes, poses, "right")


def test_hold_pose_puts_the_forearm_tips_where_the_carry_test_measured(h1, boxes):
    # a physics probe on the walking H1 held at shoulder pitch 0.28 and elbow 0.52 (the policy's default arms)
    # measured the tips in torso coordinates at about left (0.161, 0.244, -0.107) and right (0.161, -0.230, -0.107)
    left, right = _hold_tips(h1, boxes)
    for tip in (left, right):
        assert tip[0] == pytest.approx(0.161, abs=0.015)
        assert abs(tip[1]) == pytest.approx(0.237, abs=0.01)
        assert tip[2] == pytest.approx(-0.107, abs=0.01)
    assert left[1] > 0 > right[1]
    # the two arms are mirror images: only the sign of y differs
    assert left[0] == pytest.approx(right[0], abs=1e-6)
    assert left[2] == pytest.approx(right[2], abs=1e-6)
    assert left[1] == pytest.approx(-right[1], abs=1e-6)


@pytest.mark.parametrize("pitch", [0.0, 0.28])
def test_elbow_flexion_lowers_the_tip(h1, boxes, pitch):
    for straight, bent in zip(_hold_tips(h1, boxes, elbow=0.0, pitch=pitch), _hold_tips(h1, boxes, elbow=0.52, pitch=pitch)):
        assert bent[2] < straight[2] - 0.05


def test_the_tip_is_the_forearm_boxs_plus_x_end(h1, boxes):
    poses = kin.link_poses(h1, {}, frame="pelvis")
    centre, half = boxes["left_elbow_link"]
    pos, quat = poses["left_elbow_link"]
    rot = kin.quat_to_matrix(quat)
    want = pos + rot @ np.array([centre[0] + half[0], centre[1], centre[2]])
    assert kin.forearm_tip(boxes, poses, "left") == pytest.approx(want, abs=1e-12)


def test_the_tip_needs_a_side_a_box_and_a_pose(h1, boxes):
    poses = kin.link_poses(h1, {})
    with pytest.raises(ValueError, match="side"):
        kin.forearm_tip(boxes, poses, "middle")
    with pytest.raises(ValueError, match="left_elbow_link"):
        kin.forearm_tip({k: v for k, v in boxes.items() if k != "left_elbow_link"}, poses, "left")
    with pytest.raises(ValueError, match="left_elbow_link"):
        kin.forearm_tip(boxes, {k: v for k, v in poses.items() if k != "left_elbow_link"}, "left")


# ---- the extractor's own checks (numpy only; the stage reading itself needs Kit) ----

def _canon():
    return json.loads(JOINTS_JSON.read_text(encoding="utf-8"))


def test_the_extractors_checks_pass_on_the_canon_file():
    doc = _canon()
    assert ejf.document_problems(doc, ejf.joint_map_names()) == []
    assert ejf.compare(doc, doc) == []


def test_a_joint_the_map_does_not_know_is_a_problem():
    names = [n for n in ejf.joint_map_names() if n != "left_elbow"] + ["left_wing"]
    problems = ejf.document_problems(_canon(), names)
    assert any("left_wing" in p for p in problems) and any("left_elbow" in p for p in problems)


def test_frames_that_do_not_reproduce_the_stored_links_are_a_problem():
    doc = _canon()
    doc["joints"]["left_elbow"]["pos0"][0] += 1e-4          # the forearm and everything below it moves 0.1 mm
    problems = ejf.document_problems(doc, ejf.joint_map_names())
    assert [p for p in problems if "left_elbow_link" in p and "저장된 자세" in p]


def test_the_comparison_names_what_differs():
    doc = _canon()
    other = copy.deepcopy(doc)
    other["usd_sha256"] = "0" * 64
    other["joints"]["torso"]["axis"] = "Y"
    other["joints"]["left_elbow"]["pos1"][2] += 1e-6
    other["joints"].pop("right_elbow")
    other["rest"]["pelvis"]["pos"][0] += 1e-6
    other["rest"]["new_link"] = other["rest"]["pelvis"]
    text = "\n".join(ejf.compare(doc, other))
    for want in ("usd_sha256", "torso: axis", "left_elbow: pos1", "right_elbow", "pelvis: pos", "new_link"):
        assert want in text, want


def test_a_difference_below_the_rounding_is_not_a_difference():
    doc = _canon()
    other = copy.deepcopy(doc)
    other["joints"]["left_elbow"]["pos0"][0] += 1e-10
    other["note"] = "another sentence"
    assert ejf.compare(doc, other) == []


def test_a_rotation_is_written_the_same_way_whatever_its_sign():
    assert ejf.canonical_xyzw([0.0, 0.0, -0.6, -0.8]).tolist() == pytest.approx([0.0, 0.0, 0.6, 0.8])
    # with no real part the first non-zero part is made positive
    assert ejf.canonical_xyzw([0.0, -0.6, 0.8, 0.0]).tolist() == pytest.approx([0.0, 0.6, -0.8, 0.0])
    assert ejf.canonical_xyzw([0.0, 0.6, -0.8, 0.0]).tolist() == pytest.approx([0.0, 0.6, -0.8, 0.0])
    assert ejf.canonical_xyzw([0.0, 0.0, 0.0, 3.0]).tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])     # and is normalised


def test_a_record_is_rounded_to_nine_decimals_with_no_negative_zero():
    record = ejf.joint_record("revolute", "a", "b", "X", [-0.0, 1.00000000049, 0.1234567894], IDENTITY, ZERO3, IDENTITY)
    assert json.dumps(record["pos0"]) == "[0.0, 1.0, 0.123456789]"
