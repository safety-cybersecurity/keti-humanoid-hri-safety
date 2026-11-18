"""The Stage (generation/scene/build_stage.py) and the rule that ships it (generation.assets.composed_shipped).

A release carries the Stage only with everything it composes; the Stage references each of them by the name
it ships under, relative to itself at assets/. The plan of its prims is Isaac-free and runs everywhere; the
USD tests need pxr: they skip in CI and run under the Isaac python

    PYTHONPATH= ~/isaacsim/python.sh -B -m pytest -q -p no:launch_testing -p no:launch_ros \\
        tests/generation/test_stage.py
"""

import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from generation import carry, sites, visibility
from generation.assets import composed_shipped, load_assets, shipped_asset_files
from generation.robots import kinematics
from generation.scene import build_stage as bs
from generation.scene import layout
from generation.scene.carried_box import NOMINAL_SIZE_M

REPO = Path(__file__).resolve().parents[2]
# A rest pose for the plan to stand the robot on: the lowest point 0.0558 m above the root prim, the root
# link 1.1 m above it -- roughly what the BSD-3 h1.usd measures, but any numbers do here.
REST = bs.RobotRest(0.0558, (0.0, 0.0, 1.1))
# The links the carry pose moves: the arm chain below each shoulder, both sides.
ARM_LINKS = {f"{side}_{part}_link" for side in ("left", "right")
             for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rot(q, v):
    """v turned by the quaternion q (xyzw) -- written out here, not borrowed from the code under test."""
    x, y, z, w = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return r @ np.asarray(v, float)


def _same_turn(q, r, tol=1e-6) -> bool:
    """Whether two quaternions (xyzw) are one turn: q and -q are."""
    q, r = np.asarray(q, float), np.asarray(r, float)
    return bool(min(np.max(np.abs(q - r)), np.max(np.abs(q + r))) <= tol)


def _usd_view(p: bs.PlannedPrim):
    """Where a planned USD camera looks (its -z) and its image up (its +y), in its parent's axes."""
    return _rot(p.orient_xyzw, (0, 0, -1)), _rot(p.orient_xyzw, (0, 1, 0))


# ---- the shipping rule, on a canon of its own ----------------------------------------------------------

def _ship_canon(**stage_edits) -> dict:
    """Four assets a stage composes -- a robot that ships as its folder and three scene USDs -- and the stage."""
    rows = [{"file": n, "sha256": _sha(n.encode()), "license": "BSD-3-Clause", "url": f"{{N}}/R/{n}"}
            for n in ("bot.usd", "meshes.usd", "LICENSE.txt")]
    canon = {
        "robots": {"bot": {"usd_path": "{N}/R/bot.usd", "usd_sha256": rows[0]["sha256"], "license": "BSD-3-Clause",
                           "redistributed": True,
                           "ship": {"dir": "bot", "source_dir": "build/bot", "license_file": "LICENSE.txt",
                                    "files": rows}}},
        "scene": {key: {"usd_path": f"build/{key}.usd", "usd_sha256": _sha(key.encode()), "license": "CC-BY-4.0",
                        "redistributed": True} for key in ("hall", "person", "box")},
    }
    canon["scene"]["stage"] = {"usd_path": "build/stage.usd", "usd_sha256": _sha(b"stage"), "license": "CC-BY-4.0",
                               "redistributed": True, "composes": ["hall", "person", "box", "bot"], **stage_edits}
    return canon


def _paths(canon: dict, keys) -> list[str]:
    by_key = {k: e for g in ("robots", "scene") for k, e in canon[g].items()}
    return [by_key[k]["usd_path"] for k in keys]


def test_a_composed_asset_ships_only_with_everything_it_composes(tmp_path):
    canon = _ship_canon()
    every = shipped_asset_files(canon, _paths(canon, ("hall", "person", "box", "bot")), tmp_path)
    assert "assets/stage.usd" in every
    assert every["assets/stage.usd"].sha256 == _sha(b"stage") and every["assets/stage.usd"].asset == "stage"
    assert every["assets/stage.usd"].src == tmp_path / "build/stage.usd"
    # A batch that carries no box: the stage would reference a file it lacks.
    no_box = shipped_asset_files(canon, _paths(canon, ("hall", "person", "bot")), tmp_path)
    assert "assets/stage.usd" not in no_box
    assert sorted(no_box) == sorted(rel for rel in every if rel not in ("assets/stage.usd", "assets/box.usd"))


def test_a_composed_asset_ships_by_composition_alone(tmp_path):
    """An episode that names the stage does not ship it: only everything it composes does."""
    canon = _ship_canon()
    assert "assets/stage.usd" not in shipped_asset_files(canon, _paths(canon, ("hall", "stage")), tmp_path)


@pytest.mark.parametrize("edit", [{"redistributed": False}, {"redistributed": "true"}])
def test_a_composed_asset_that_is_not_redistributable_does_not_ship(tmp_path, edit):
    canon = _ship_canon(**edit)
    assert "assets/stage.usd" not in shipped_asset_files(
        canon, _paths(canon, ("hall", "person", "box", "bot")), tmp_path)


def test_composed_shipped_names_the_keys_whose_components_all_ship():
    canon = _ship_canon()
    assert composed_shipped(canon, {"hall", "person", "box", "bot"}) == {"stage"}
    assert composed_shipped(canon, {"hall", "person", "box", "bot", "extra"}) == {"stage"}
    assert composed_shipped(canon, {"hall", "person", "bot"}) == set()
    assert composed_shipped(canon, set()) == set()


@pytest.mark.parametrize("edit, phrase", [
    ({"composes": "hall"}, "비어 있지 않은 키 문자열의 목록이 아닙니다"),
    ({"composes": []}, "비어 있지 않은 키 문자열의 목록이 아닙니다"),
    ({"composes": ["hall", 3]}, "비어 있지 않은 키 문자열의 목록이 아닙니다"),
    ({"composes": ["hall", "hall"]}, "같은 키가 두 번"),
    ({"composes": ["hall", "nobody"]}, "nobody 가 meta/assets.json 의 robots · scene 에 없습니다"),
    ({"composes": ["hall", "stage"]}, "자기 자신"),
    ({"bundle": {"files": []}}, "엮은 자산에 bundle 가 있습니다"),
    ({"ship": {}}, "엮은 자산에 ship 가 있습니다"),
    ({"usd_path": None}, "stage: usd_path 가 비어 있지 않은 문자열이 아닙니다: None"),
    ({"usd_path": 3}, "stage: usd_path 가 비어 있지 않은 문자열이 아닙니다: 3"),
    ({"usd_path": ""}, "stage: usd_path 가 비어 있지 않은 문자열이 아닙니다: ''"),
])
def test_a_composes_that_cannot_say_what_it_composes_is_refused(tmp_path, edit, phrase):
    canon = _ship_canon(**edit)
    with pytest.raises(ValueError, match=phrase):
        composed_shipped(canon, {"hall"})
    with pytest.raises(ValueError, match=phrase):
        shipped_asset_files(canon, _paths(canon, ("hall",)), tmp_path)


def test_a_composed_asset_that_composes_another_is_refused():
    canon = _ship_canon()
    canon["scene"]["outer"] = {"usd_path": "build/outer.usd", "license": "CC-BY-4.0", "redistributed": True,
                               "composes": ["stage", "hall"]}
    with pytest.raises(ValueError, match="stage 도 엮은 자산입니다"):
        composed_shipped(canon, {"hall"})


def test_a_key_in_both_groups_is_refused():
    canon = _ship_canon()
    canon["robots"]["hall"] = dict(canon["scene"]["hall"])
    with pytest.raises(ValueError, match="hall 가 robots 와 scene 에 모두 있습니다"):
        composed_shipped(canon, {"hall"})


@pytest.mark.parametrize("edit", [{"usd_sha256": None}, {"usd_sha256": "abc"}])
def test_a_shipping_composed_asset_without_a_pin_is_refused(tmp_path, edit):
    canon = _ship_canon(**edit)
    with pytest.raises(ValueError, match="stage: 재배포하는데 정본에 usd_sha256 이 없거나"):
        shipped_asset_files(canon, _paths(canon, ("hall", "person", "box", "bot")), tmp_path)
    # Not shipping, its pin is not asked for.
    assert "assets/stage.usd" not in shipped_asset_files(canon, _paths(canon, ("hall",)), tmp_path)


def test_a_composed_asset_under_another_assets_name_is_refused(tmp_path):
    canon = _ship_canon(usd_path="build/other/hall.usd")
    with pytest.raises(ValueError, match="두 자산이 같은 이름으로 실립니다: assets/hall.usd"):
        shipped_asset_files(canon, _paths(canon, ("hall", "person", "box", "bot")), tmp_path)


# ---- the plan, from the repository's canon ----------------------------------------------------------

@pytest.fixture(scope="module")
def canon():
    scenarios = json.loads((REPO / "meta" / "scenarios.json").read_text(encoding="utf-8"))
    cameras = json.loads((REPO / "meta" / "cameras.json").read_text(encoding="utf-8"))
    return scenarios, cameras, load_assets(REPO)


@pytest.fixture(scope="module")
def plan(canon):
    return bs.stage_plan(*canon, robot=REST)


def _site(scenarios, sid):
    return sites.load_site(next(s for s in scenarios["scenarios"] if s["scenario_id"] == sid))


def _hold_poses(scenarios):
    """S1e's carry block and every link at its hold pose relative to the robot's root link (pelvis)."""
    held = carry.load_carry(next(s for s in scenarios["scenarios"] if s["scenario_id"] == "S1e"))
    joints = kinematics.load_joints()
    return held, kinematics.link_poses(joints, carry.arm_hold_joints(held, joints["joints"]), frame="pelvis")


def test_stage_refs_are_the_shipped_names(plan):
    # Relative to the Stage at assets/: the scene USDs beside it, the robot in its folder.
    refs = {path: p.ref for path, p in plan.items() if p.kind == "reference"}
    assert refs == {"/World/Warehouse": "./keti_logistics_aisle_v1.usd",
                    "/World/Robot": "./unitree_h1/h1.usd",
                    "/World/Dummy": "./keti_dummy_ts15066_v2_jointed_skin_vest.usd",
                    "/World/CarriedBox": "./keti_carried_box_v1.usd"}


def test_component_refs_follow_the_shipping_rule(canon):
    # The same names shipped_asset_files gives a batch that uses all four, under assets/.
    assets = canon[2]
    by_key = {k: e for g in ("robots", "scene") for k, e in assets[g].items()}
    keys = [key for _, key in bs.COMPONENTS]
    plain = {g: {k: e for k, e in assets[g].items() if "composes" not in e} for g in ("robots", "scene")}
    shipped = shipped_asset_files(plain, [by_key[k]["usd_path"] for k in keys], REPO)
    refs = bs.component_refs(assets)
    assert {f"assets/{ref[2:]}" for ref in refs.values()} <= set(shipped)
    assert all(shipped[f"assets/{refs[k][2:]}"].asset == k for k in keys)


def test_an_asset_a_release_cannot_carry_is_refused(canon):
    assets = json.loads(json.dumps(canon[2]))
    assets["scene"]["keti_carried_box_v1"]["redistributed"] = False
    with pytest.raises(ValueError, match="keti_carried_box_v1: 배포본에서 USD keti_carried_box_v1.usd 가 실리는 자리"):
        bs.component_refs(assets)


def test_stage_plan_puts_the_robot_on_the_s1e_start(plan, canon):
    start = _site(canon[0], "S1e").robot_start
    # The plan's place for the robot: S1e's start, as the canon has it at this writing.
    assert (start.x, start.y, start.yaw_deg) == (-4.5, 7.25, 180.0)
    robot = plan["/World/Robot"]
    assert robot.translate == pytest.approx((start.x, start.y, -0.0558))      # its lowest point on the floor
    assert robot.orient_xyzw == pytest.approx((0.0, 0.0, 1.0, 0.0), abs=1e-12)
    # The dummy: the middle of S1e's person's region, facing +y.
    dummy = plan["/World/Dummy"]
    assert dummy.translate == pytest.approx((-5.145, 4.45, 0.0))
    assert _rot(dummy.orient_xyzw, (1, 0, 0)) == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)


def test_stage_holds_the_arms_at_the_carry_pose(plan, canon):
    """The eight arm links are overs that set their transform to where the kinematics puts them at S1e's carry pose,
    relative to the root link, moved to where the robot's root prim has its root link (REST) -- and nothing else is
    posed: the legs and the torso stay as the robot's USD has them."""
    _, poses = _hold_poses(canon[0])
    posed = {path: p for path, p in plan.items() if p.posed}
    assert set(posed) == {f"/World/Robot/{link}" for link in ARM_LINKS}
    for path, p in posed.items():
        assert p.kind == "over"
        pos, quat = poses[path.rsplit("/", 1)[1]]
        assert p.translate == pytest.approx(tuple(np.add(REST.root_link_m, pos)), abs=1e-6), path
        assert _same_turn(p.orient_xyzw, quat), path


def test_the_posed_forearms_end_where_the_box_bottom_is(plan, canon):
    """Through the plan's own transforms -- the robot's prim, then each elbow link's -- the forearm tips land beside
    the box at its bottom's height, ahead of the robot (it faces -x at S1e's start)."""
    boxes = visibility.load_link_boxes()
    robot = plan["/World/Robot"]
    box = plan["/World/CarriedBox"]
    for side, sign in (("left", 1.0), ("right", -1.0)):
        link = plan[f"/World/Robot/{side}_elbow_link"]
        centre, half = boxes[f"{side}_elbow_link"]
        tip_link = np.asarray(centre, float) + np.array([half[0], 0.0, 0.0])
        tip = np.add(robot.translate, _rot(robot.orient_xyzw, np.add(link.translate, _rot(link.orient_xyzw, tip_link))))
        assert tip[2] == pytest.approx(box.translate[2] - NOMINAL_SIZE_M[2] / 2, abs=0.001)      # at the bottom
        assert tip[0] > box.translate[0] - NOMINAL_SIZE_M[1] / 2                                  # behind the front
        assert tip[0] < box.translate[0] + NOMINAL_SIZE_M[1] / 2                                  # ahead of the back
        gap = sign * (box.translate[1] - tip[1]) - NOMINAL_SIZE_M[0] / 2                       # beside its side
        assert carry.TIP_SIDE_GAP_M[0] <= gap <= carry.TIP_SIDE_GAP_M[1], (side, gap)


def test_stage_puts_the_box_on_the_torso(plan, canon):
    """The box's centre is the torso's pose at the carry pose with S1e's carry added (the bottom's middle plus half
    the nominal height), placed with the robot -- and its +y (its depth, the top label's up) is the robot's forward."""
    start = _site(canon[0], "S1e").robot_start
    held, poses = _hold_poses(canon[0])
    torso_pos, torso_quat = poses[held.link]
    robot = plan["/World/Robot"]
    local = np.add(held.bottom_m, (0.0, 0.0, NOMINAL_SIZE_M[2] / 2))
    in_robot = np.add(np.add(REST.root_link_m, torso_pos), _rot(torso_quat, local))
    box = plan["/World/CarriedBox"]
    assert box.translate == pytest.approx(tuple(np.add(robot.translate, _rot(robot.orient_xyzw, in_robot))), abs=1e-6)
    # The same in numbers: the robot faces -x, the torso sits on the root link (REST) with no turn.
    assert box.translate == pytest.approx((start.x - held.bottom_m[0], start.y - held.bottom_m[1],
                                           -REST.lowest_z_m + REST.root_link_m[2] + held.bottom_m[2]
                                           + NOMINAL_SIZE_M[2] / 2), abs=1e-6)
    assert _rot(box.orient_xyzw, (0, 1, 0)) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-6)
    assert _rot(box.orient_xyzw, (0, 0, 1)) == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)


def test_the_box_sits_on_the_link_the_carry_names(canon):
    """The torso and the root link coincide at the carry pose, so the box is moved to another link to tell them
    apart: on the left forearm the box is that link's pose with the carry added, placed with the robot."""
    scenarios = copy.deepcopy(canon[0])
    next(s for s in scenarios["scenarios"] if s["scenario_id"] == "S1e")["carry"]["link"] = "left_elbow_link"
    plan = bs.stage_plan(scenarios, canon[1], canon[2], robot=REST)
    held, poses = _hold_poses(scenarios)
    link_pos, link_quat = poses["left_elbow_link"]
    local = np.add(held.bottom_m, (0.0, 0.0, NOMINAL_SIZE_M[2] / 2))
    in_robot = np.add(np.add(REST.root_link_m, link_pos), _rot(link_quat, local))
    robot, box = plan["/World/Robot"], plan["/World/CarriedBox"]
    assert box.translate == pytest.approx(tuple(np.add(robot.translate, _rot(robot.orient_xyzw, in_robot))), abs=1e-9)


def test_a_plan_without_the_placed_sites_carry_is_refused(canon):
    scenarios = copy.deepcopy(canon[0])
    del next(s for s in scenarios["scenarios"] if s["scenario_id"] == bs.PLACED_AT)["carry"]
    with pytest.raises(ValueError, match="S1e 에 carry 가 없습니다"):
        bs.stage_plan(scenarios, canon[1], canon[2], robot=REST)
    other = copy.deepcopy(canon[0])
    next(s for s in other["scenarios"] if s["scenario_id"] == bs.PLACED_AT)["carry"]["asset"] = "keti_logistics_aisle_v1"
    with pytest.raises(ValueError, match="Stage 가 엮는 상자"):
        bs.stage_plan(other, canon[1], canon[2], robot=REST)


def test_held_links_are_the_arm_chains_below_the_held_joints():
    joints = kinematics.load_joints()
    hold = {f"{side}_{name}": 0.1 for side in ("left", "right")
            for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")}
    assert set(bs.held_links(joints, hold)) == ARM_LINKS
    # The elbow alone moves its own link; a shoulder pitch moves the chain under it.
    assert bs.held_links(joints, {"left_elbow": 0.5}) == ["left_elbow_link"]
    assert set(bs.held_links(joints, {"right_shoulder_pitch": 0.5})) == {
        f"right_{part}_link" for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")}


def test_stage_plan_has_five_robot_cameras_and_a_third_person_and_a_cctv_per_placement(plan):
    cameras = {path for path, p in plan.items() if p.kind == "camera"}
    assert cameras == {
        "/World/Robot/torso_link/head_rgb", "/World/Robot/torso_link/head_down_rgb",
        "/World/Robot/left_elbow_link/left_wrist_rgb", "/World/Robot/right_elbow_link/right_wrist_rgb",
        "/World/Robot/torso_link/rear_rgb",
        *(f"/World/Sites/{sid}/{name}" for sid in ("S1e", "S1f") for name in ("third_person_rgb", "cctv_rgb")),
        # S1g's are placed per side the person comes from: one of each per side, under that side's scope
        *(f"/World/Sites/S1g/from_{side}/{name}" for side in ("west", "east")
          for name in ("third_person_rgb", "cctv_rgb")),
    }
    assert plan["/World/Sites/S1g/from_west"].kind == plan["/World/Sites/S1g/from_east"].kind == "scope"
    # The links come from the robot's USD: the Stage only adds to them -- the cameras' torso and the eight arm links
    # it holds at the carry pose (the elbow links carry the wrist cameras too).
    assert {path for path, p in plan.items() if p.kind == "over"} == {
        "/World/Robot/torso_link", *(f"/World/Robot/{link}" for link in ARM_LINKS)}
    assert not plan["/World/Robot/torso_link"].posed
    for sid in ("S1e", "S1f", "S1g"):
        assert [plan[f"/World/Sites/{sid}/{m}"].kind for m in ("robot_start", "robot_goal", "human_region")] == \
            ["disc", "disc", "plate"]
    paths = list(plan)
    assert all(path.rsplit("/", 1)[0] in paths[:i] for i, path in enumerate(paths) if path != "/World")


@pytest.mark.parametrize("name, forward", [("head_rgb", (1, 0, 0)), ("rear_rgb", (-1, 0, 0))])
def test_a_mounted_camera_looks_along_its_optical_axis(plan, canon, name, forward):
    """The head camera looks straight ahead along the link's +x, the rear one straight back -- each level, the
    picture's up the link's +z -- at the canon offset."""
    cam = canon[1]["cameras"][name]
    p = plan[f"/World/Robot/{cam['parent_prim']}/{name}"]
    assert p.translate == pytest.approx(tuple(cam["offset_pos"]))
    view, up = _usd_view(p)
    assert view == pytest.approx(forward, abs=1e-12) and up == pytest.approx((0, 0, 1), abs=1e-12)
    # The same direction the canon's ROS frame looks along (+z), and the same up (-y).
    assert view == pytest.approx(_rot(cam["offset_rot_xyzw"], (0, 0, 1)), abs=1e-12)
    assert up == pytest.approx(_rot(cam["offset_rot_xyzw"], (0, -1, 0)), abs=1e-12)


@pytest.mark.parametrize("name", ["head_down_rgb", "left_wrist_rgb", "right_wrist_rgb"])
def test_every_mounted_camera_keeps_the_canon_optical_axis(plan, canon, name):
    cam = canon[1]["cameras"][name]
    view, up = _usd_view(plan[f"/World/Robot/{cam['parent_prim']}/{name}"])
    assert view == pytest.approx(_rot(cam["offset_rot_xyzw"], (0, 0, 1)), abs=1e-12)
    assert up == pytest.approx(_rot(cam["offset_rot_xyzw"], (0, -1, 0)), abs=1e-12)


def _placement(cameras, name, sid, side):
    """The canon's placement of camera ``name`` at ``sid`` -- the side's own where the site holds one per side."""
    placement = cameras["cameras"][name]["sites"][sid]
    return placement if side is None else placement["by_approach_side"][side]


@pytest.mark.parametrize("sid, side", [("S1e", None), ("S1f", None), ("S1g", "west"), ("S1g", "east")])
def test_a_site_camera_sits_where_its_placement_puts_it(plan, canon, sid, side):
    """The CCTV at its eye looking at its aim; the third-person camera for the robot at the site's start --
    its offsets turned by the start's heading (both heading frames read the start), added to the root link."""
    scenarios, cameras, _ = canon
    start = _site(scenarios, sid).robot_start
    under = f"/World/Sites/{sid}" if side is None else f"/World/Sites/{sid}/from_{side}"
    cctv = _placement(cameras, "cctv_rgb", sid, side)
    p = plan[f"{under}/cctv_rgb"]
    assert p.translate == pytest.approx(tuple(cctv["eye_m"]))
    aim = np.subtract(cctv["aim_m"], cctv["eye_m"])
    view, up = _usd_view(p)
    assert view == pytest.approx(aim / np.linalg.norm(aim), abs=1e-9) and up[2] > 0
    placement = _placement(cameras, "third_person_rgb", sid, side)
    assert placement["frame"] in ("heading", "initial_heading")
    c, s = math.cos(math.radians(start.yaw_deg)), math.sin(math.radians(start.yaw_deg))
    root = np.array([start.x, start.y, -REST.lowest_z_m + REST.root_link_m[2]])

    def turned(v):
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])
    eye, target = root + turned(placement["eye_offset"]), root + turned(placement["lookat_offset"])
    p = plan[f"{under}/third_person_rgb"]
    assert p.translate == pytest.approx(tuple(eye), abs=1e-9)
    view, up = _usd_view(p)
    assert view == pytest.approx((target - eye) / np.linalg.norm(target - eye), abs=1e-9) and up[2] > 0


def test_a_camera_lens_is_the_simulation_cameras(plan, canon):
    """Isaac Lab's pinhole spawn: the focal length from the fov over Isaac Sim's default horizontal aperture
    (20.955 mm), the vertical aperture by the image's aspect."""
    cam = canon[1]["cameras"]["head_rgb"]
    lens = plan["/World/Robot/torso_link/head_rgb"].camera
    assert lens["focal_length_mm"] == pytest.approx(20.955 / (2 * math.tan(math.radians(cam["fov_h_deg"]) / 2)))
    assert lens["horizontal_aperture_mm"] == 20.955
    assert lens["vertical_aperture_mm"] == pytest.approx(20.955 * cam["height"] / cam["width"])
    assert lens["clipping_range_m"] == tuple(cam["clipping_range_m"])


def test_marks_lie_above_the_painted_lines_and_cover_the_region(plan, canon):
    spec = layout.load_spec(layout.SCENE_SPEC)
    paint = spec["floor_markings"]
    paint_top = spec["floor"]["z"] + layout.SLAB / 2 + paint["lift"] + paint["thickness"]
    site = _site(canon[0], "S1e")
    region = plan["/World/Sites/S1e/human_region"]
    assert region.translate[2] - bs.MARK_THICKNESS_M / 2 > paint_top
    assert region.translate[:2] == pytest.approx((sum(site.human_x_m) / 2, sum(site.human_y_m) / 2))
    assert region.scale == pytest.approx((site.human_x_m[1] - site.human_x_m[0] + 0.3,
                                          site.human_y_m[1] - site.human_y_m[0] + 0.3, bs.MARK_THICKNESS_M))
    start = plan["/World/Sites/S1e/robot_start"]
    assert start.translate[:2] == pytest.approx((site.robot_start.x, site.robot_start.y))
    assert start.scale == pytest.approx((0.5, 0.5, bs.MARK_THICKNESS_M))
    assert all(p.kind in ("disc", "plate") for path, p in plan.items() if p.color is not None)
    # one above the other, a thickness apart: the start disc, the person's region, the goal disc
    goal = plan["/World/Sites/S1e/robot_goal"]
    assert region.translate[2] - start.translate[2] == pytest.approx(bs.MARK_THICKNESS_M)
    assert goal.translate[2] - region.translate[2] == pytest.approx(bs.MARK_THICKNESS_M)


def _footprint(p):
    (x, y, _), (sx, sy, _) = p.translate, p.scale
    return x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2


@pytest.mark.parametrize("sid", ["S1e", "S1f", "S1g"])
def test_marks_that_overlap_seen_from_above_never_share_a_height(plan, sid):
    """S1g's start and goal discs both reach into its person's region: no two marks that overlap from above may lie
    in one plane (the renderer would flicker between them), so each lies at least a thickness above the one under it."""
    marks = {m: plan[f"/World/Sites/{sid}/{m}"] for m in ("robot_start", "human_region", "robot_goal")}
    for a in marks:
        for b in marks:
            if a < b:
                ax0, ax1, ay0, ay1 = _footprint(marks[a])
                bx0, bx1, by0, by1 = _footprint(marks[b])
                if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                    gap = abs(marks[a].translate[2] - marks[b].translate[2])
                    assert gap >= bs.MARK_THICKNESS_M - 1e-12, (sid, a, b, gap)


def test_the_s1g_marks_and_third_person_camera_sit_at_its_start(plan, canon):
    """S1g's robot starts in front of rack B facing it (-y): the start disc is there, and the third-person camera,
    held to the robot's first heading, stands 3.5 m along the aisle on the side the person walks toward -- east of
    the robot for a person from the west, west of it for one from the east."""
    site = _site(canon[0], "S1g")
    assert (site.robot_start.x, site.robot_start.y, site.robot_start.yaw_deg) == (2.0, 10.3, -90.0)
    assert plan["/World/Sites/S1g/robot_start"].translate[:2] == pytest.approx((2.0, 10.3))
    assert plan["/World/Sites/S1g/robot_goal"].translate[:2] == pytest.approx((2.0, 11.3))
    for side, x in (("west", 2.0 + 3.5), ("east", 2.0 - 3.5)):
        eye = plan[f"/World/Sites/S1g/from_{side}/third_person_rgb"].translate
        assert eye[:2] == pytest.approx((x, 10.3), abs=1e-6), side


def test_the_s1g_cameras_from_the_east_are_the_single_placements_and_no_other_prim_moved(plan, canon):
    # The plan of a canon whose S1g scene cameras have the east placement alone, not one per side: S1g's east
    # cameras are its cameras, and every other prim is the same prim in the same order.
    scenarios, cameras, assets = canon
    one = copy.deepcopy(cameras)
    names = ("third_person_rgb", "cctv_rgb")
    for name in names:
        one["cameras"][name]["sites"]["S1g"] = one["cameras"][name]["sites"]["S1g"]["by_approach_side"]["east"]
    before = bs.stage_plan(scenarios, one, assets, robot=REST)
    for name in names:
        assert before[f"/World/Sites/S1g/{name}"] == plan[f"/World/Sites/S1g/from_east/{name}"], name
    sided = {f"/World/Sites/S1g/{name}" for name in names}
    assert [(k, v) for k, v in before.items() if k not in sided] == \
        [(k, v) for k, v in plan.items() if not k.startswith("/World/Sites/S1g/from_")]


def test_a_plan_without_the_placed_sites_site_is_refused(canon):
    scenarios = json.loads(json.dumps(canon[0]))
    for s in scenarios["scenarios"]:
        if s["scenario_id"] == bs.PLACED_AT:
            del s["scene"]["site"]
    with pytest.raises(ValueError, match="S1e 의 scene.site 가 없습니다"):
        bs.stage_plan(scenarios, canon[1], canon[2], robot=REST)


@pytest.mark.parametrize("scene", [[0], None, "S1f", 3])
def test_a_scenario_whose_scene_is_not_an_object_is_refused(canon, scene):
    scenarios = copy.deepcopy(canon[0])
    next(s for s in scenarios["scenarios"] if s["scenario_id"] == "S1f")["scene"] = scene
    with pytest.raises(ValueError, match=r"S1f\.scene 이 객체가 아닙니다"):
        bs.stage_plan(scenarios, canon[1], canon[2], robot=REST)


def test_a_scenario_without_a_scene_is_refused(canon):
    scenarios = copy.deepcopy(canon[0])
    del next(s for s in scenarios["scenarios"] if s["scenario_id"] == "S2a")["scene"]
    with pytest.raises(ValueError, match=r"S2a\.scene 이 객체가 아닙니다"):
        bs.stage_plan(scenarios, canon[1], canon[2], robot=REST)


@pytest.mark.parametrize("site", [None, [], "here", 0])
def test_a_site_that_is_present_but_not_an_object_is_refused_not_left_out(canon, site):
    # A site key that holds no site is a canon that cannot say where that scenario's cameras and marks go: the plan
    # refuses it rather than lay out the Stage without them.
    scenarios = copy.deepcopy(canon[0])
    next(s for s in scenarios["scenarios"] if s["scenario_id"] == "S1f")["scene"]["site"] = site
    with pytest.raises(ValueError, match=r"S1f\.scene\.site 가 객체가 아닙니다"):
        bs.stage_plan(scenarios, canon[1], canon[2], robot=REST)


def test_a_scenario_without_a_site_key_has_no_site_on_the_stage(canon, plan):
    # S1d declares no site (no site key): the plan lays out the three that do, and nothing for S1d.
    s1d = next(s for s in canon[0]["scenarios"] if s["scenario_id"] == "S1d")
    assert "site" not in s1d["scene"]
    assert sorted(p.split("/")[3] for p in plan if p.count("/") == 3 and p.startswith("/World/Sites/")) \
        == ["S1e", "S1f", "S1g"]


def test_composes_problems_hold_the_canon_to_the_builder():
    assert bs.composes_problems({"scene": {}}) == ["meta/assets.json 의 scene 에 keti_hri_stage_v1 가 없습니다"]
    keys = [key for _, key in bs.COMPONENTS]
    assert bs.composes_problems({"scene": {bs.STAGE_KEY: {"composes": list(reversed(keys))}}}) == []
    for wrong in (keys[:-1], keys + ["x"], keys[:-1] + keys[:1], keys + keys[:1], "".join(keys)):
        assert bs.composes_problems({"scene": {bs.STAGE_KEY: {"composes": wrong}}}), wrong


def test_the_root_link_is_the_joint_maps_articulation_root():
    assert bs.root_link_name() == "pelvis"


# ---- what the Stage entry pins, read without pxr ----------------------------------------------------------

def _moved(plan, path, **changes):
    """The plan with one prim's fields changed."""
    out = dict(plan)
    out[path] = dataclasses.replace(plan[path], **changes)
    return out


@pytest.mark.parametrize("path", ["/World/CarriedBox", "/World/Robot/left_elbow_link"])
def test_the_plan_digest_sees_a_prim_moved_by_a_tenth_of_a_millimetre(plan, path):
    x, y, z = plan[path].translate
    assert bs.plan_digest(_moved(plan, path, translate=(x, y + 1e-4, z)), REST) != bs.plan_digest(plan, REST)


def test_the_plan_digest_sees_an_over_that_stops_setting_its_pose(plan):
    assert bs.plan_digest(_moved(plan, "/World/Robot/left_elbow_link", posed=False), REST) != \
        bs.plan_digest(plan, REST)


def test_the_plan_digest_sees_the_rest_numbers_and_the_prims_order(plan):
    assert bs.plan_digest(plan, bs.RobotRest(0.0559, REST.root_link_m)) != bs.plan_digest(plan, REST)
    assert bs.plan_digest(dict(reversed(list(plan.items()))), REST) != bs.plan_digest(plan, REST)


def test_the_plan_digest_does_not_see_what_two_numpy_builds_may_differ_by(plan):
    """A number off in its last bits, -0.0 for 0.0, a turn as -q rather than q: the same Stage."""
    base = bs.plan_digest(plan, REST)
    x, y, z = plan["/World/CarriedBox"].translate
    assert bs.plan_digest(_moved(plan, "/World/CarriedBox", translate=(x, y * (1 + 4e-16), z)), REST) == base
    assert bs.plan_digest(_moved(plan, "/World/Warehouse", translate=(-0.0, 0.0, -0.0)), REST) == base
    for path in ("/World/Robot", "/World/Sites/S1e/cctv_rgb", "/World/Robot/left_elbow_link", "/World/CarriedBox"):
        q = plan[path].orient_xyzw
        assert bs.plan_digest(_moved(plan, path, orient_xyzw=tuple(-v for v in q)), REST) == base, path


def test_the_kept_rest_numbers_are_read_back_as_recorded():
    assert bs.pinned_rest({"scene": {bs.STAGE_KEY: {bs.REST_KEY: bs.rest_record(REST)}}}) == REST
    assert bs.rest_record(REST) == {"lowest_z_m": 0.0558, "root_link_m": [0.0, 0.0, 1.1]}


@pytest.mark.parametrize("rest", [
    None, {"lowest_z_m": 0.0558}, {"lowest_z_m": "0.0558", "root_link_m": [0.0, 0.0, 1.1]},
    {"lowest_z_m": 0.0558, "root_link_m": [0.0, 1.1]}, {"lowest_z_m": math.nan, "root_link_m": [0.0, 0.0, 1.1]},
    {"lowest_z_m": 0.0558, "root_link_m": [0.0, 0.0, True]},
    {"lowest_z_m": 0.0558, "root_link_m": [0.0, 0.0, 1.1], "note": "x"},
], ids=["absent", "no-root-link", "a-string", "two-numbers", "nan", "a-bool", "an-unknown-key"])
def test_rest_numbers_the_entry_cannot_say_are_refused(rest):
    entry = {} if rest is None else {bs.REST_KEY: rest}
    with pytest.raises(ValueError, match="keti_hri_stage_v1.robot_rest 가 없거나"):
        bs.pinned_rest({"scene": {bs.STAGE_KEY: entry}})


def test_check_stage_holds_the_kept_rest_numbers_to_the_measured_ones():
    from generation.scene import check_stage

    assets = {"scene": {bs.STAGE_KEY: {bs.REST_KEY: bs.rest_record(REST)}}}
    assert check_stage.rest_problems(assets, REST) == []
    problems = check_stage.rest_problems(assets, bs.RobotRest(REST.lowest_z_m, (0.0, 0.0, 1.1 + 1e-6)))
    assert len(problems) == 1 and "받아 둔 h1.usd 에서 잰 값" in problems[0]
    assert check_stage.rest_problems({"scene": {bs.STAGE_KEY: {}}}, REST)[0].startswith(
        "meta/assets.json 의 keti_hri_stage_v1.robot_rest 가 없거나")


def test_check_stage_counts_the_posed_links_against_the_held_links():
    from generation.scene import check_stage

    scenarios = json.loads((REPO / "meta" / "scenarios.json").read_text(encoding="utf-8"))
    joints = kinematics.load_joints()
    held = bs.held_links(joints, carry.arm_hold_joints(bs.carry_of(scenarios), joints["joints"]))
    assert set(held) == ARM_LINKS
    assert check_stage.held_link_problems(held, held) == []
    assert check_stage.held_link_problems(list(reversed(held)), held) == []          # the order is not looked at
    none = check_stage.held_link_problems([], held)                                   # nothing compared says nothing
    assert len(none) == 1 and "링크 0 개가" in none[0] and all(name in none[0] for name in held), none
    short = check_stage.held_link_problems(held[1:], held)
    assert len(short) == 1 and held[0] in short[0], short
    extra = check_stage.held_link_problems([*held, "torso_link"], held)
    assert len(extra) == 1 and "torso_link" in extra[0], extra
    twice = check_stage.held_link_problems([*held, held[0]], held)
    assert len(twice) == 1 and held[0] in twice[0], twice
    # the Stage check compares the links it posed-checked with the links the hold pose moves
    source = Path(check_stage.__file__).read_text(encoding="utf-8")
    block = source[source.index("def _carry_problems"):source.index("def check(")]
    assert "out += held_link_problems(checked, build_stage.held_links(joints, hold))" in block


@pytest.mark.parametrize("pin", [None, "", "ee9c5711", "E" * 64], ids=["absent", "empty", "short", "upper-case"])
def test_an_entry_without_a_plan_hash_is_a_problem(canon, pin):
    scenarios, cameras, assets = copy.deepcopy(canon)
    entry = assets["scene"][bs.STAGE_KEY]
    entry[bs.REST_KEY] = bs.rest_record(REST)
    entry.pop(bs.PLAN_KEY, None)
    if pin is not None:
        entry[bs.PLAN_KEY] = pin
    problems = bs.pin_problems(scenarios, cameras, assets)
    assert len(problems) == 1 and "keti_hri_stage_v1.plan_sha256 가 없거나 SHA-256 이 아닙니다" in problems[0]


def test_an_entry_pinned_to_the_plan_it_lays_out_has_no_problem_and_one_moved_has_one(canon):
    """The pin a build at REST would write, against the canon it was laid out from and the same canon with S1e's
    person's region moved 0.25 m along x."""
    scenarios, cameras, assets = copy.deepcopy(canon)
    entry = assets["scene"][bs.STAGE_KEY]
    entry[bs.REST_KEY] = bs.rest_record(REST)
    entry[bs.PLAN_KEY] = bs.plan_digest(bs.stage_plan(scenarios, cameras, assets, robot=REST), REST)
    assert bs.pin_problems(scenarios, cameras, assets) == []
    s1e = next(s for s in scenarios["scenarios"] if s["scenario_id"] == "S1e")
    s1e["scene"]["site"]["human_region"]["x_m"] = [v + 0.25 for v in s1e["scene"]["site"]["human_region"]["x_m"]]
    problems = bs.pin_problems(scenarios, cameras, assets)
    assert len(problems) == 1 and "build_stage.py 로 다시 지어" in problems[0]


# ---- the USD (pxr) ----------------------------------------------------------------------------

def test_write_stage_keeps_links_as_overs_and_writes_the_same_bytes(tmp_path, plan):
    pytest.importorskip("pxr")
    from pxr import Sdf

    a, b = tmp_path / "a" / "stage.usd", tmp_path / "b" / "stage.usd"
    bs.write_stage(a, plan)
    bs.write_stage(b, plan)
    assert a.read_bytes() == b.read_bytes()
    layer = Sdf.Layer.FindOrOpen(str(a))
    assert layer.defaultPrim == "World"
    robot = layer.GetPrimAtPath("/World/Robot")
    assert [r.assetPath for r in robot.referenceList.prependedItems] == ["./unitree_h1/h1.usd"]
    assert layer.GetPrimAtPath("/World/Robot/torso_link").specifier == Sdf.SpecifierOver
    head = layer.GetPrimAtPath("/World/Robot/torso_link/head_rgb")
    assert head.specifier == Sdf.SpecifierDef and head.typeName == "Camera"
    # A posed arm link sets the robot USD's one transform op of that link; the torso's over sets nothing.
    assert "xformOp:transform" not in layer.GetPrimAtPath("/World/Robot/torso_link").attributes
    for link in sorted(ARM_LINKS):
        spec = layer.GetPrimAtPath(f"/World/Robot/{link}")
        assert spec.specifier == Sdf.SpecifierOver, link
        attr = spec.attributes["xformOp:transform"]
        assert attr.typeName == Sdf.ValueTypeNames.Matrix4d, link
        m = np.array(attr.default)
        p = plan[f"/World/Robot/{link}"]
        assert m[3] == pytest.approx((*p.translate, 1.0), abs=1e-12)
        # Row i is where the link's axis i points (USD row vectors).
        for i, axis in enumerate(np.eye(3)):
            assert m[i, :3] == pytest.approx(_rot(p.orient_xyzw, axis), abs=1e-12)
        assert m[:3, 3] == pytest.approx((0.0, 0.0, 0.0), abs=0)
    names = set()
    layer.Traverse(Sdf.Path.absoluteRootPath, lambda path: names.add(path.name) if path.IsPropertyPath() else None)
    assert not {n for n in names if n.startswith("physics:") or n.startswith("keti:")}, names
    assert not [n for n in names if "texture" in n.lower()]


# ---- the canon's Stage ---------------------------------------------------------------------------------

def test_the_canon_stage_composes_what_the_builder_places():
    assert bs.composes_problems(load_assets(REPO)) == []


def test_the_stage_pin_is_the_plan_todays_canon_lays_out():
    """The Stage entry's plan_sha256 is the plan build_stage lays out from today's canon at the rest numbers the
    entry keeps -- checked here without pxr, so a site, a camera or a placement changed and the Stage not built
    again is red wherever the tests run."""
    scenarios = json.loads((REPO / "meta" / "scenarios.json").read_text(encoding="utf-8"))
    cameras = json.loads((REPO / "meta" / "cameras.json").read_text(encoding="utf-8"))
    assert bs.pin_problems(scenarios, cameras, load_assets(REPO)) == []


def _s1e_person_moved(scenarios, cameras, monkeypatch):
    s1e = next(s for s in scenarios["scenarios"] if s["scenario_id"] == "S1e")
    s1e["scene"]["site"]["human_region"]["x_m"] = [v + 0.25 for v in s1e["scene"]["site"]["human_region"]["x_m"]]


def _s1e_cctv_moved(scenarios, cameras, monkeypatch):
    eye = cameras["cameras"]["cctv_rgb"]["sites"]["S1e"]["eye_m"]
    cameras["cameras"]["cctv_rgb"]["sites"]["S1e"]["eye_m"] = [eye[0] + 0.5, *eye[1:]]


def _box_brought_closer(scenarios, cameras, monkeypatch):
    held = next(s for s in scenarios["scenarios"] if s["scenario_id"] == "S1e")["carry"]
    held["bottom_m"] = [held["bottom_m"][0] - 0.02, *held["bottom_m"][1:]]


def _elbows_bent_more(scenarios, cameras, monkeypatch):
    held = next(s for s in scenarios["scenarios"] if s["scenario_id"] == "S1e")["carry"]
    held["arm_hold_rad"]["elbow"] = held["arm_hold_rad"]["elbow"] + 0.02


@pytest.mark.parametrize("move", [_s1e_person_moved, _s1e_cctv_moved, _box_brought_closer, _elbows_bent_more],
                         ids=["S1e-person-region", "S1e-cctv-eye", "S1e-carry-bottom", "S1e-carry-elbow"])
def test_a_canon_changed_since_the_stage_was_built_fails_the_pin(move, monkeypatch):
    """The repository's pin against a copy of the canon with one input of the Stage changed -- S1e's person's
    region 0.25 m along x, S1e's CCTV 0.5 m along x, S1e's carried box 2 cm nearer the robot, its elbows bent
    0.02 rad more: the message says to build again and re-pin."""
    scenarios = copy.deepcopy(json.loads((REPO / "meta" / "scenarios.json").read_text(encoding="utf-8")))
    cameras = copy.deepcopy(json.loads((REPO / "meta" / "cameras.json").read_text(encoding="utf-8")))
    move(scenarios, cameras, monkeypatch)
    problems = bs.pin_problems(scenarios, cameras, load_assets(REPO))
    assert len(problems) == 1, problems
    assert "keti_hri_stage_v1: 지금 정본으로 놓은 Stage 의 배치" in problems[0]
    assert "build_stage.py 로 다시 지어 usd_sha256 · usd_checked · plan_sha256 · robot_rest 를 고치고" in problems[0]


def test_the_stage_ships_only_with_everything_it_composes():
    """A release whose episodes use the warehouse, the vested dummy, the box and H1 (S1e, S1f, S1g) carries the
    Stage at assets/, pinned; one without the box does not."""
    assets = load_assets(REPO)
    entry = assets["scene"][bs.STAGE_KEY]
    paths = {k: e["usd_path"] for g in ("robots", "scene") for k, e in assets[g].items()}
    every = [paths[k] for k in ("keti_logistics_aisle_v1", "keti_dummy_ts15066_v2_vest", "keti_carried_box_v1",
                                "unitree_h1", "isaaclab_terrain_plane")]
    files = shipped_asset_files(assets, every, REPO)
    assert files["assets/keti_hri_stage_v1.usd"].sha256 == entry["usd_sha256"]
    assert files["assets/keti_hri_stage_v1.usd"].src == REPO / entry["usd_path"]
    no_box = shipped_asset_files(assets, [p for p in every if p != paths["keti_carried_box_v1"]], REPO)
    assert "assets/keti_hri_stage_v1.usd" not in no_box and "assets/keti_carried_box_v1.usd" not in no_box
    assert "assets/unitree_h1/h1.usd" in no_box


def _fetched_robot() -> Path:
    """The robot's fetched copy, for a test that reads it; the test skips only where the robot was not fetched (a
    file absent). A copy with other bytes, or a canon that cannot say what to fetch, raises: a failure."""
    try:
        return bs.robot_usd()
    except FileNotFoundError as exc:
        pytest.skip(f"로봇을 받아 두지 않았습니다: {exc}")


@pytest.mark.parametrize("error, skips", [(FileNotFoundError("redistributed=true 인데 파일이 없습니다"), True),
                                          (ValueError("h1.usd 의 해시가 정본과 다릅니다"), False),
                                          (ValueError("unitree_h1: ship 이 객체가 아닙니다"), False)])
def test_only_a_robot_not_fetched_skips_a_test_that_reads_it(monkeypatch, error, skips):
    # A copy that is there with other bytes, or a canon that cannot say what to fetch, is a failure: skipping on it
    # would turn the rebuild check off without a word.
    def robot_usd(*_args, **_kwargs):
        raise error
    monkeypatch.setattr(bs, "robot_usd", robot_usd)
    try:
        _fetched_robot()
    except pytest.skip.Exception:            # caught here: left to propagate, it would skip this test too
        outcome = "skip"
    except ValueError:
        outcome = "fail"
    assert outcome == ("skip" if skips else "fail")


def _robot_usd(path: Path, *, nested: bool = False, turned: bool = False, two_ops: bool = False) -> Path:
    """A two-link robot USD for robot_rest: /r with pelvis and arm_link, each a rigid body with one matrix op."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/r")
    stage.SetDefaultPrim(root.GetPrim())
    turn = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 10.0)) if turned else Gf.Matrix4d(1.0)
    pelvis = UsdGeom.Xform.Define(stage, "/r/pelvis")
    pelvis.AddTransformOp().Set(turn * Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, 1.0)))
    arm = UsdGeom.Xform.Define(stage, "/r/pelvis/arm_link" if nested else "/r/arm_link")
    arm.AddTransformOp().Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0.2, 1.4)))
    if two_ops:
        arm.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    for prim in (pelvis, arm):
        UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/r/pelvis/geo")
    cube.CreateSizeAttr(0.2)
    stage.GetRootLayer().Save()
    return path


@pytest.mark.parametrize("kind, phrase", [("nested", "뿌리 프림 바로 아래에"), ("two_ops", "뿌리 프림 바로 아래에"),
                                          ("turned", "돌아 있습니다")])
def test_robot_rest_refuses_a_robot_the_posed_links_cannot_be_written_on(tmp_path, kind, phrase):
    pytest.importorskip("pxr")
    assert bs.robot_rest(_robot_usd(tmp_path / "flat.usda"), "pelvis").root_link_m == pytest.approx((0, 0, 1.0))
    with pytest.raises(ValueError, match=phrase):
        bs.robot_rest(_robot_usd(tmp_path / f"{kind}.usda", **{kind: True}), "pelvis")


def test_the_stage_is_built_as_its_plan(tmp_path):
    """The built Stage is what build_stage writes from the canon today -- where the Stage and the robot's fetched
    copy are on disk (pxr)."""
    pytest.importorskip("pxr")
    built = REPO / load_assets(REPO)["scene"][bs.STAGE_KEY]["usd_path"]
    _fetched_robot()
    if not built.is_file():
        pytest.skip(f"{built.name} 가 빌드되어 있지 않습니다")
    again = bs.build(tmp_path / built.name)
    assert again.read_bytes() == built.read_bytes()
