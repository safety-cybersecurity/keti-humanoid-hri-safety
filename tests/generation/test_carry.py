"""Where the robot holds the box it carries (generation/carry.py): the canon's ``carry`` read and checked against
the forearms, the wrist cameras and the stop test's reach. Pure numpy -- runs everywhere."""

import copy
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from generation import carry
from generation import visibility
from generation.robots import kinematics
from generation.scene.carried_box import NOMINAL_SIZE_M

REPO = Path(__file__).resolve().parents[2]
WIDTH_M = 0.40
# The carried range of the box (depth x height); the width is fixed.
SIZES = [(WIDTH_M, depth, height) for depth in (0.25, 0.30) for height in (0.20, 0.30)]


def _scenarios() -> dict:
    return json.loads((REPO / "meta" / "scenarios.json").read_text(encoding="utf-8"))


def _entry(sid: str) -> dict:
    return next(s for s in _scenarios()["scenarios"] if s["scenario_id"] == sid)


def _reach_m() -> float:
    rules = json.loads((REPO / "meta" / "rules.json").read_text(encoding="utf-8"))
    rule = next(r for r in rules["rules"] if r["rule_id"] == "R-TS15066-SSM-02")
    return float(rule["stop_table"]["turning_reach_m"])


@pytest.fixture(scope="module")
def joints():
    return kinematics.load_joints()


@pytest.fixture(scope="module")
def link_boxes():
    return visibility.load_link_boxes()


@pytest.fixture(scope="module")
def cameras_canon():
    return json.loads((REPO / "meta" / "cameras.json").read_text(encoding="utf-8"))


def _joint_names() -> list[str]:
    import yaml

    doc = yaml.safe_load((REPO / "meta" / "joint_map" / "unitree_h1.yaml").read_text(encoding="utf-8"))
    return list(doc["joint_names"])


def _rot(q, v):
    """v turned by the quaternion q (xyzw) -- written out here, not borrowed from the code under test."""
    x, y, z, w = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return r @ np.asarray(v, float)


# ---- the canon's place -----------------------------------------------------------------------------------

@pytest.mark.parametrize("sid", ["S1e", "S1f", "S1g"])
def test_the_canon_carry_holds_the_box_between_the_forearms(sid, joints, link_boxes, cameras_canon):
    """At the arms' hold pose: both forearm tips beside the box and under its bottom edge, no robot camera inside the
    box, the box's corners within the stop test's turning reach and the box clear of the torso link's box -- for
    every size of the carried range."""
    spec = carry.load_carry(_entry(sid))
    assert carry.carry_problems(spec, joints, link_boxes, cameras_canon, sizes=SIZES, reach_m=_reach_m()) == []


def test_a_box_round_a_wrist_camera_is_a_problem(joints, link_boxes, cameras_canon):
    spec = carry.load_carry(_entry("S1e"))
    moved = dataclasses.replace(spec, bottom_m=(spec.bottom_m[0], 0.20, spec.bottom_m[2]))
    problems = carry.carry_problems(moved, joints, link_boxes, cameras_canon, sizes=SIZES[:1], reach_m=_reach_m())
    assert any("left_wrist_rgb" in p and "상자 안" in p for p in problems), problems
    assert not any("right_wrist_rgb" in p for p in problems), problems


@pytest.mark.parametrize("sid", ["S1e", "S1f", "S1g"])
def test_the_cameras_that_move_about_the_box_are_the_wrists(sid, cameras_canon):
    # The box is fixed to the torso: the cameras on the torso keep their place against it, the wrist cameras ride on
    # the forearms, which sway while the robot walks and turns.
    spec = carry.load_carry(_entry(sid))
    assert spec.link == "torso_link"
    assert carry.moving_cameras(spec, cameras_canon) == ("left_wrist_rgb", "right_wrist_rgb")
    on_elbows = {n for n, c in cameras_canon["cameras"].items() if str(c.get("parent_prim", "")).endswith("elbow_link")}
    assert set(carry.moving_cameras(spec, cameras_canon)) == on_elbows


def test_a_box_too_far_ahead_is_a_problem(joints, link_boxes, cameras_canon):
    spec = carry.load_carry(_entry("S1e"))
    moved = dataclasses.replace(spec, bottom_m=(0.40, *spec.bottom_m[1:]))
    problems = carry.carry_problems(moved, joints, link_boxes, cameras_canon, sizes=SIZES[:1], reach_m=_reach_m())
    assert sum("아래팔 끝" in p and "앞뒤" in p for p in problems) == 2, problems      # both tips behind the box
    assert any("수평 반지름" in p for p in problems), problems


def test_the_box_bottom_sits_at_the_forearm_tips(joints, link_boxes):
    """The canon's bottom height is the forearm tips' height at the hold pose, in the link the box is fixed to."""
    for sid in ("S1e", "S1f", "S1g"):
        spec = carry.load_carry(_entry(sid))
        poses = kinematics.link_poses(joints, carry.arm_hold_joints(spec, _joint_names()), frame=spec.link)
        for side in ("left", "right"):
            tip = kinematics.forearm_tip(link_boxes, poses, side)
            assert spec.bottom_m[2] == pytest.approx(tip[2], abs=0.001), (sid, side, tip)


def test_the_canon_bottom_is_ahead_of_the_tips_by_what_its_note_says(joints, link_boxes):
    """The note says the box's middle is 0.12 m ahead of the forearm tips -- the number the canon's x and the tips
    give."""
    spec = carry.load_carry(_entry("S1e"))
    poses = kinematics.link_poses(joints, carry.arm_hold_joints(spec, _joint_names()), frame=spec.link)
    tip_x = kinematics.forearm_tip(link_boxes, poses, "left")[0]
    assert "앞으로 0.12 m" in _entry("S1e")["carry"]["note"]
    assert round(spec.bottom_m[0] - tip_x, 2) == 0.12


def test_the_three_scenarios_carry():
    s1e, s1f, s1g = (carry.load_carry(_entry(sid)) for sid in ("S1e", "S1f", "S1g"))
    assert s1e is not None and s1e == s1f == s1g
    assert s1e.asset == "keti_carried_box_v1" and s1e.link == "torso_link" and s1e.release == "first_human_contact"
    assets = json.loads((REPO / "meta" / "assets.json").read_text(encoding="utf-8"))
    assert s1e.asset in assets["scene"]
    # The asset is one the scenario's scene names.
    usd = assets["scene"][s1e.asset]["usd_path"]
    assert all(usd in _entry(sid)["scene"]["assets"] for sid in ("S1e", "S1f", "S1g"))
    assert dict(s1e.arm_hold_rad) == {"shoulder_pitch": 0.28, "shoulder_roll": 0.0, "shoulder_yaw": 0.0, "elbow": 0.52}


# ---- reading the block ------------------------------------------------------------------------------------

def _block():
    return copy.deepcopy(_entry("S1e")["carry"])


def _with(**changes):
    block = _block()
    for key, value in changes.items():
        if value is _DROP:
            del block[key]
        else:
            block[key] = value
    return {"scenario_id": "S1e", "carry": block}


_DROP = object()


def test_load_carry_reads_the_block():
    spec = carry.load_carry(_with())
    assert spec.bottom_m == (0.27, 0.0, -0.111) and spec.yaw_deg == -90.0
    assert isinstance(spec.bottom_m, tuple) and all(isinstance(v, float) for v in spec.bottom_m)
    with pytest.raises(TypeError):
        spec.arm_hold_rad["elbow"] = 0.0                      # read-only


@pytest.mark.parametrize("changes, phrase", [
    ({"asset": _DROP}, r"S1e\.carry\.asset 가 없습니다"),
    ({"bottom_m": _DROP}, r"S1e\.carry\.bottom_m 가 없습니다"),
    ({"release": _DROP}, r"S1e\.carry\.release 가 없습니다"),
    ({"note": _DROP}, r"S1e\.carry\.note 가 없습니다"),
    ({"bottom_m": [0.27, "0", -0.111]}, r"S1e\.carry\.bottom_m 이 유한한 수 3개"),
    ({"bottom_m": [0.27, 0.0]}, r"S1e\.carry\.bottom_m 이 유한한 수 3개"),
    ({"bottom_m": [0.27, True, -0.111]}, r"S1e\.carry\.bottom_m 이 유한한 수 3개"),
    ({"bottom_m": [0.27, math.nan, -0.111]}, r"S1e\.carry\.bottom_m 이 유한한 수 3개"),
    ({"yaw_deg": "-90"}, r"S1e\.carry\.yaw_deg 이 유한한 수가 아닙니다"),
    ({"arm_hold_rad": {"elbow": "0.52"}}, r"S1e\.carry\.arm_hold_rad\.elbow 이 유한한 수가 아닙니다"),
    ({"arm_hold_rad": {}}, r"S1e\.carry\.arm_hold_rad 이 비어 있지 않은 객체가 아닙니다"),
    ({"arm_hold_rad": [0.28]}, r"S1e\.carry\.arm_hold_rad 이 비어 있지 않은 객체가 아닙니다"),
    ({"arm_hold_rad": {"left_elbow": 0.52}}, r"S1e\.carry\.arm_hold_rad\.left_elbow 는 쪽을 뺀 이름"),
    ({"release": "on_timeout"}, r"S1e\.carry\.release 가 \['first_human_contact'\] 중 하나가 아닙니다"),
    ({"link": ""}, r"S1e\.carry\.link 가 비어 있지 않은 문자열이 아닙니다"),
    ({"link": "missing_link"}, r"S1e\.carry\.link 'missing_link' 이 H1 의 링크가 아닙니다 — 링크 이름은 "
                               r"generation/robots/unitree_h1_joints\.json"),
    ({"asset": 3}, r"S1e\.carry\.asset 가 비어 있지 않은 문자열이 아닙니다"),
    ({"note": "  "}, r"S1e\.carry\.note 가 비어 있지 않은 문자열이 아닙니다"),
    ({"determined_by": "standard"}, r"S1e\.carry\.determined_by 가 'project' 가 아닙니다"),
    ({"extra": 1}, r"S1e\.carry 에 모르는 키 \['extra'\]"),
], ids=["no-asset", "no-bottom", "no-release", "no-note", "a-string", "two-numbers", "a-bool", "nan", "yaw-string",
        "hold-string", "hold-empty", "hold-list", "hold-sided", "unknown-release", "empty-link", "unknown-link", "asset-number",
        "blank-note", "not-project", "unknown-key"])
def test_load_carry_refuses_bad_blocks(changes, phrase):
    with pytest.raises(ValueError, match=phrase):
        carry.load_carry(_with(**changes))


@pytest.mark.parametrize("link", ["pelvis", "torso_link", "left_elbow_link"])
def test_any_link_of_the_h1_may_hold_the_box(link):
    assert link in kinematics.load_joints()["rest"]
    assert carry.load_carry(_with(link=link)).link == link


@pytest.mark.parametrize("block", [None, [], "torso"])
def test_a_carry_that_is_not_an_object_is_refused(block):
    with pytest.raises(ValueError, match=r"S1e\.carry 가 객체가 아닙니다"):
        carry.load_carry({"scenario_id": "S1e", "carry": block})


def test_arm_hold_finds_both_sides():
    spec = carry.load_carry(_entry("S1e"))
    names = _joint_names()
    hold = carry.arm_hold_joints(spec, names)
    assert hold == {"left_shoulder_pitch": 0.28, "right_shoulder_pitch": 0.28, "left_shoulder_roll": 0.0,
                    "right_shoulder_roll": 0.0, "left_shoulder_yaw": 0.0, "right_shoulder_yaw": 0.0,
                    "left_elbow": 0.52, "right_elbow": 0.52}
    with pytest.raises(ValueError, match=r"right_elbow"):
        carry.arm_hold_joints(spec, [n for n in names if n != "right_elbow"])
    with pytest.raises(ValueError, match=r"left_shoulder_yaw"):
        carry.arm_hold_joints(spec, [n for n in names if n != "left_shoulder_yaw"])


# ---- where the box is ---------------------------------------------------------------------------------------

def _spec(**changes) -> carry.Carry:
    return dataclasses.replace(carry.load_carry(_entry("S1e")), **changes)


def test_box_pose_lifts_the_centre_by_half_the_height():
    size = (0.40, 0.30, 0.20)
    spec = _spec(bottom_m=(0.3, -0.05, -0.1), yaw_deg=0.0)
    centre, quat = carry.box_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), spec, size)
    assert centre == pytest.approx((0.3, -0.05, 0.0), abs=1e-12)
    assert np.abs(quat) == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-12)
    # A link at (1, 2, 3) turned a quarter turn about z: the bottom's offset turns with it, the lift stays up.
    s = math.sqrt(0.5)
    centre, quat = carry.box_pose((1.0, 2.0, 3.0), (0.0, 0.0, s, s), spec, size)
    assert centre == pytest.approx((1.0 + 0.05, 2.0 + 0.3, 3.0 - 0.1 + 0.1), abs=1e-12)
    # A link pitched a quarter turn about y (its +z now along the world +x): the lift is along the link's own up.
    centre, _ = carry.box_pose((0.0, 0.0, 0.0), (0.0, s, 0.0, s), spec, size)
    assert centre == pytest.approx(_rot((0.0, s, 0.0, s), (0.3, -0.05, 0.0)), abs=1e-12)
    assert centre == pytest.approx((0.0, -0.05, -0.3), abs=1e-12)


def test_box_pose_turns_the_box_by_its_yaw_on_the_link():
    spec = _spec(yaw_deg=-90.0)
    _, quat = carry.box_pose((0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), spec, NOMINAL_SIZE_M)   # link faces -x
    # The box's +y (its depth, the top label's up) along the link's +x, which is the world -x here.
    assert _rot(quat, (0, 1, 0)) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)
    assert _rot(quat, (1, 0, 0)) == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert np.linalg.norm(quat) == pytest.approx(1.0) and quat[3] >= 0.0


def test_the_box_turns_about_the_links_own_up_on_a_tilted_link():
    """On a link pitched 30 deg about its y and turned 40 deg about the world's z, a box at yaw -90 has its +y (its depth)
    along the link's +x and its +z along the link's +z: the yaw is about the link's own up, applied after the link's
    orientation -- not about the world's up."""
    a, b = math.radians(30.0), math.radians(40.0)
    pitch = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
    turn = np.array([[math.cos(b), -math.sin(b), 0.0], [math.sin(b), math.cos(b), 0.0], [0.0, 0.0, 1.0]])
    link = turn @ pitch                                   # columns: the link's axes in the world
    _, quat = carry.box_pose((0.0, 0.0, 0.0), kinematics.quat_from_matrix(link), _spec(yaw_deg=-90.0), (0.4, 0.3, 0.2))
    assert _rot(quat, (0, 1, 0)) == pytest.approx(link[:, 0], abs=1e-12)
    assert _rot(quat, (0, 0, 1)) == pytest.approx(link[:, 2], abs=1e-12)
    assert _rot(quat, (1, 0, 0)) == pytest.approx(-link[:, 1], abs=1e-12)


def test_box_obb_points_its_depth_forward():
    """At a link with no yaw the box's depth axis (its +y) is the world +x and its half sizes are the size's."""
    size = (0.40, 0.30, 0.20)
    spec = carry.load_carry(_entry("S1e"))
    obb = carry.box_obb((0.5, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), spec, size)
    assert isinstance(obb, visibility.Obb)
    assert _rot(obb.rot_xyzw, (0, 1, 0)) == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    assert obb.half == pytest.approx((0.20, 0.15, 0.10))
    assert obb.center == pytest.approx((0.5 + 0.27, 0.0, 1.0 - 0.111 + 0.10), abs=1e-12)


@pytest.mark.parametrize("size", [(0.4, 0.3), (0.4, -0.3, 0.2), (0.4, 0.3, math.inf), "abc", (0.4, True, 0.2)])
def test_a_size_that_is_not_three_lengths_is_refused(size):
    with pytest.raises(ValueError, match="상자 크기"):
        carry.box_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), _spec(), size)


# ---- the checks one at a time ------------------------------------------------------------------------------

def _problems(spec, joints, link_boxes, cameras_canon, sizes=SIZES[:1], reach_m=None):
    return carry.carry_problems(spec, joints, link_boxes, cameras_canon, sizes=sizes,
                                reach_m=_reach_m() if reach_m is None else reach_m)


def test_a_box_too_wide_for_the_forearms_is_a_problem(joints, link_boxes, cameras_canon):
    # 0.50 m wide: the tips (|y| 0.24) are inside the box's sides.
    problems = _problems(_spec(), joints, link_boxes, cameras_canon, sizes=[(0.50, 0.25, 0.25)])
    assert sum("옆면" in p for p in problems) == 2, problems


def test_a_box_too_narrow_for_the_forearms_is_a_problem(joints, link_boxes, cameras_canon):
    # 0.30 m wide: 0.09 m of air between each side and its forearm.
    problems = _problems(_spec(), joints, link_boxes, cameras_canon, sizes=[(0.30, 0.25, 0.25)])
    assert sum("옆면" in p for p in problems) == 2, problems


@pytest.mark.parametrize("dz, ok", [(0.0, True), (0.009, True), (-0.029, True), (0.011, False), (-0.031, False)])
def test_the_bottom_must_rest_on_the_tips(joints, link_boxes, cameras_canon, dz, ok):
    """The tips at most 0.01 m under the bottom and 0.03 m above it (they hold the box's lower edge)."""
    spec = carry.load_carry(_entry("S1e"))
    poses = kinematics.link_poses(joints, carry.arm_hold_joints(spec, _joint_names()), frame=spec.link)
    tip_z = kinematics.forearm_tip(link_boxes, poses, "left")[2]
    moved = _spec(bottom_m=(spec.bottom_m[0], spec.bottom_m[1], tip_z + dz))
    problems = _problems(moved, joints, link_boxes, cameras_canon)
    assert (not any("밑면" in p for p in problems)) == ok, problems


def test_a_box_off_to_one_side_is_a_problem(joints, link_boxes, cameras_canon):
    # Moved 0.48 m left: the left forearm's tip ends up beside the box's side face, the right one far outside on the
    # same side -- the box is not between the two.
    problems = _problems(_spec(bottom_m=(0.27, 0.479, -0.111)), joints, link_boxes, cameras_canon)
    assert any("두 아래팔 사이" in p for p in problems), problems


def test_a_box_into_the_torso_is_a_problem(joints, link_boxes, cameras_canon):
    # 0.36 m deep at x 0.27: the back face at 0.09, 0.035 m inside the torso link's box (front 0.1248).
    problems = _problems(_spec(), joints, link_boxes, cameras_canon, sizes=[(0.40, 0.36, 0.25)])
    assert any("몸통" in p and "겹" in p for p in problems), problems
    # 0.30 m deep: the back face at 0.12, inside the torso box by 4.8 mm -- within the 0.01 m allowed for a box that
    # wraps the mesh.
    assert not any("몸통" in p for p in _problems(_spec(), joints, link_boxes, cameras_canon,
                                                    sizes=[(0.40, 0.30, 0.25)]))


def test_the_reach_is_the_stop_tests_with_five_millimetres_to_spare(joints, link_boxes, cameras_canon):
    # The 0.30 m deep box's front corners: sqrt(0.42^2 + 0.2^2) = 0.46519 m.
    corner = math.hypot(0.27 + 0.15, 0.20)
    size = [(0.40, 0.30, 0.25)]
    assert not any("수평 반지름" in p for p in _problems(_spec(), joints, link_boxes, cameras_canon, sizes=size,
                                                     reach_m=corner - 0.0049))
    assert any("수평 반지름" in p for p in _problems(_spec(), joints, link_boxes, cameras_canon, sizes=size,
                                                 reach_m=corner - 0.0051))


def test_every_size_is_checked(joints, link_boxes, cameras_canon):
    """A size that fails is named even when the others pass."""
    problems = _problems(_spec(), joints, link_boxes, cameras_canon, sizes=[SIZES[0], (0.40, 0.36, 0.25)])
    assert problems and all("0.36" in p for p in problems), problems


@pytest.mark.parametrize("reach", [0.0, -0.4, math.nan, "0.465"])
def test_a_reach_that_is_not_a_length_is_refused(joints, link_boxes, cameras_canon, reach):
    with pytest.raises(ValueError, match="reach_m"):
        _problems(_spec(), joints, link_boxes, cameras_canon, reach_m=reach)


def test_no_sizes_is_refused(joints, link_boxes, cameras_canon):
    with pytest.raises(ValueError, match="sizes"):
        _problems(_spec(), joints, link_boxes, cameras_canon, sizes=[])


def test_a_link_without_a_box_is_a_problem(joints, link_boxes, cameras_canon):
    boxes = {k: v for k, v in link_boxes.items() if k != "torso_link"}
    problems = _problems(_spec(), joints, boxes, cameras_canon)
    assert any("torso_link" in p and "링크 상자" in p for p in problems), problems
