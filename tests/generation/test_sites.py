"""Sites (generation/sites.py): where the robot starts and goes and where the person stands, read
from the canon's scene.site and checked against the warehouse layout, on numbers worked out by
hand. Isaac-free."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from generation.scene import layout
from generation.sites import (
    CLEARANCE_M,
    PATH_KINDS,
    SITE_KEYS,
    Pose2,
    Site,
    human_region_points,
    load_site,
    nominal_path,
    site_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TWO_PI = 2 * math.pi


def _scenario(start=(0.0, 0.0, 0.0), goal=(1.0, 0.0, 0.0), path=None, x_m=(0.0, 1.0), y_m=(0.0, 1.0),
              sid="T1"):
    return {"scenario_id": sid, "scene": {"site": {
        "robot_start": {"xy_m": [start[0], start[1]], "yaw_deg": start[2]},
        "robot_goal": {"xy_m": [goal[0], goal[1]], "yaw_deg": goal[2]},
        "path": {"kind": "straight"} if path is None else path,
        "human_region": {"x_m": list(x_m), "y_m": list(y_m)},
    }}}


def _same_angle(a, b):
    """Whether two yaws (rad) are the same direction: equal modulo 2 pi."""
    return abs((a - b + math.pi) % TWO_PI - math.pi) < 1e-9


def _box(name, lo, hi, z=(0.0, 3.0), collider=True, semantic="rack"):
    return layout.Box(name, ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (z[0] + z[1]) / 2),
                      (hi[0] - lo[0], hi[1] - lo[1], z[1] - z[0]), "m", collider, semantic)


# A 20 x 20 m floor, the inner faces of the walls.
FLOOR = layout.Box("floor", (0.0, 0.0, 0.0), (20.0, 20.0, 0.001), "m", False, None)


# ── the path, sampled ───────────────────────────────────────────────────────────

def test_a_straight_path_is_sampled_every_step():
    site = load_site(_scenario(start=(0.0, 0.0, 0.0), goal=(1.0, 0.0, 0.0)))
    path = nominal_path(site, step_m=0.25)
    assert path.shape == (5, 3)
    assert np.allclose(path[:, 0], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(path[:, 1], 0.0) and np.allclose(path[:, 2], 0.0)


def test_a_length_that_is_a_whole_number_of_steps_takes_exactly_that_many():
    """0.14 / 0.02 is 7.000000000000001 in floating point; it is still 7 steps and 8 rows."""
    path = nominal_path(load_site(_scenario(start=(0.0, 0.0, 0.0), goal=(0.14, 0.0, 0.0))), step_m=0.02)
    assert path.shape == (8, 3)
    assert np.allclose(path[:, 0], np.arange(8) * 0.02)


def test_a_straight_path_keeps_the_yaw_it_starts_with():
    site = load_site(_scenario(start=(0.0, 0.0, 90.0), goal=(0.0, 2.0, 90.0)))
    path = nominal_path(site, step_m=1.0)
    assert np.allclose(path, [[0.0, 0.0, math.pi / 2], [0.0, 1.0, math.pi / 2], [0.0, 2.0, math.pi / 2]])


def test_a_u_turn_goes_out_round_a_half_circle_and_back():
    """Along +x for 3 m, a half circle of radius 1 about (3, 0) turning left, then 2 m along -x.
    A step of pi/4 gives 4 rows on the first line (3 / 0.785 -> 4), 4 on the arc (pi / 0.785 = 4)
    and 3 on the last line (2 / 0.785 -> 3): 1 + 4 + 4 + 3 = 12 rows."""
    site = load_site(_scenario(
        start=(0.0, -1.0, 0.0), goal=(1.0, 1.0, 180.0),
        path={"kind": "u_turn", "turn_center_xy_m": [3.0, 0.0], "radius_m": 1.0}))
    path = nominal_path(site, step_m=math.pi / 4)
    assert path.shape == (12, 3)
    assert np.allclose(path[0], [0.0, -1.0, 0.0])
    assert np.allclose(path[4], [3.0, -1.0, 0.0])                   # where the circle begins
    assert np.allclose(path[5], [3 + math.sqrt(0.5), -math.sqrt(0.5), math.pi / 4])
    assert np.allclose(path[6], [4.0, 0.0, math.pi / 2])            # the far side of the circle
    assert np.allclose(path[8], [3.0, 1.0, math.pi])                # where it ends
    assert np.allclose(path[-1], [1.0, 1.0, math.pi])


def test_a_u_turn_along_minus_x_turns_left_too():
    site = load_site(_scenario(
        start=(6.0, 1.0, 180.0), goal=(8.0, -1.0, 0.0),
        path={"kind": "u_turn", "turn_center_xy_m": [3.0, 0.0], "radius_m": 1.0}))
    path = nominal_path(site, step_m=math.pi / 4)
    far = path[np.argmin(path[:, 0])]
    assert np.allclose(far[:2], [2.0, 0.0])                         # the west side of the circle
    assert math.isclose(far[2], 1.5 * math.pi)                      # facing -y: a left turn from facing -x
    assert math.isclose(path[-1, 2], TWO_PI)                        # the yaw keeps growing, it is not wrapped


def test_a_turn_then_walk_turns_in_five_degree_rows_and_walks_on():
    """Facing +x at (1, 1), a right turn of 90 degrees is 18 rows of -5 degrees, then 2 m facing -y."""
    site = load_site(_scenario(
        start=(1.0, 1.0, 0.0), goal=(1.0, -1.0, -90.0),
        path={"kind": "turn_then_walk", "turn_deg": -90.0, "walk_m": 2.0}))
    path = nominal_path(site, step_m=1.0)
    assert path.shape == (1 + 18 + 2, 3)
    assert np.allclose(path[:19, :2], [1.0, 1.0])                   # the turn stays where it is
    assert np.allclose(np.diff(path[:19, 2]), -math.radians(5.0))
    assert np.allclose(path[18], [1.0, 1.0, -math.pi / 2])
    assert np.allclose(path[-1], [1.0, -1.0, -math.pi / 2])


@pytest.mark.parametrize("step", [0, -0.05, float("nan"), float("inf"), True, "0.05", None], ids=repr)
def test_a_step_that_is_not_a_positive_length_is_refused(step):
    site = load_site(_scenario())
    with pytest.raises(ValueError, match="step_m"):
        nominal_path(site, step_m=step)


# ── what a site is refused for ──────────────────────────────────────────────────

@pytest.mark.parametrize("key", SITE_KEYS)
def test_a_site_with_a_missing_key_is_refused(key):
    scenario = _scenario()
    del scenario["scene"]["site"][key]
    with pytest.raises(ValueError, match=key):
        load_site(scenario)


def test_a_scenario_without_a_site_is_refused():
    s1d = next(s for s in _canon_scenarios() if s["scenario_id"] == "S1d")
    with pytest.raises(ValueError, match="S1d.*scene.site"):
        load_site(s1d)


def test_a_scenario_without_an_id_is_refused():
    scenario = _scenario()
    del scenario["scenario_id"]
    with pytest.raises(ValueError, match="scenario_id"):
        load_site(scenario)


def test_a_straight_goal_off_the_line_is_refused():
    with pytest.raises(ValueError, match="직선"):
        load_site(_scenario(start=(0.0, 0.0, 0.0), goal=(1.0, 0.5, 0.0)))


def test_a_straight_goal_behind_the_robot_is_refused():
    with pytest.raises(ValueError, match="직선"):
        load_site(_scenario(start=(0.0, 0.0, 0.0), goal=(-1.0, 0.0, 0.0)))


def test_a_straight_goal_that_turns_the_robot_is_refused():
    with pytest.raises(ValueError, match="요"):
        load_site(_scenario(start=(0.0, 0.0, 0.0), goal=(1.0, 0.0, 90.0)))


def test_an_unknown_path_kind_is_refused():
    with pytest.raises(ValueError, match="zigzag") as err:
        load_site(_scenario(path={"kind": "zigzag"}))
    assert all(kind in str(err.value) for kind in PATH_KINDS)


@pytest.mark.parametrize("kind,missing", [("u_turn", "turn_center_xy_m"), ("u_turn", "radius_m"),
                                          ("turn_then_walk", "turn_deg"), ("turn_then_walk", "walk_m")])
def test_a_path_without_what_its_kind_needs_is_refused(kind, missing):
    path = {"u_turn": {"kind": "u_turn", "turn_center_xy_m": [3.0, 0.0], "radius_m": 1.0},
            "turn_then_walk": {"kind": "turn_then_walk", "turn_deg": 90.0, "walk_m": 1.0}}[kind]
    del path[missing]
    with pytest.raises(ValueError, match=missing):
        load_site(_scenario(start=(0.0, -1.0, 0.0), goal=(1.0, 1.0, 180.0), path=path))


U_TURN = {"kind": "u_turn", "turn_center_xy_m": [3.0, 0.0], "radius_m": 1.0}


@pytest.mark.parametrize("start,goal,match", [
    pytest.param((0.0, -0.9, 0.0), (1.0, 1.0, 180.0), "출발 y", id="start off the circle's tangent"),
    pytest.param((0.0, -1.0, 0.0), (1.0, 0.9, 180.0), "목표 y", id="goal off the line the circle ends on"),
    pytest.param((0.0, -1.0, 0.0), (1.0, -1.0, 180.0), "목표 y", id="goal on the side the robot came from"),
    pytest.param((0.0, -1.0, 0.0), (1.0, 1.0, 0.0), "요", id="goal facing the way the robot started"),
    pytest.param((4.0, -1.0, 0.0), (1.0, 1.0, 180.0), "이미 지났", id="start already past the circle"),
    pytest.param((0.0, -1.0, 0.0), (4.0, 1.0, 180.0), "반원이 끝난 쪽", id="goal on the far side of the circle"),
    pytest.param((0.0, -1.0, 90.0), (1.0, 1.0, 270.0), "x 방향", id="start not along x"),
])
def test_a_u_turn_whose_ends_do_not_fit_the_circle_is_refused(start, goal, match):
    with pytest.raises(ValueError, match=match):
        load_site(_scenario(start=start, goal=goal, path=U_TURN))


@pytest.mark.parametrize("goal,match", [
    pytest.param((1.0, -1.0, -90.0), "목표 .* 어긋납니다", id="goal further than the walk"),
    pytest.param((1.0, 0.0, -90.0), "목표 .* 어긋납니다", id="goal short of the walk"),
    pytest.param((2.0, -0.5, -90.0), "목표 .* 어긋납니다", id="goal beside the walk"),
    pytest.param((1.0, -0.5, 90.0), "요", id="goal where the walk ends but facing the other way"),
])
def test_a_turn_then_walk_goal_the_walk_does_not_reach_is_refused(goal, match):
    """Facing +x at (1, 1), a right turn of 90 degrees and 1.5 m end at (1, -0.5) facing -y."""
    walk = {"kind": "turn_then_walk", "turn_deg": -90.0, "walk_m": 1.5}
    with pytest.raises(ValueError, match=match):
        load_site(_scenario(start=(1.0, 1.0, 0.0), goal=goal, path=walk))


def test_a_turn_then_walk_goal_the_walk_reaches_is_accepted():
    walk = {"kind": "turn_then_walk", "turn_deg": -90.0, "walk_m": 1.5}
    path = nominal_path(load_site(_scenario(start=(1.0, 1.0, 0.0), goal=(1.0, -0.5, -90.0), path=walk)), step_m=0.5)
    assert np.allclose(path[-1], [1.0, -0.5, -math.pi / 2])


@pytest.mark.parametrize("turn", [181.0, -181.0, float("nan"), True, "90"], ids=repr)
def test_a_turn_beyond_half_a_circle_or_not_a_number_is_refused(turn):
    with pytest.raises(ValueError, match="turn_deg"):
        load_site(_scenario(start=(1.0, 1.0, 0.0), goal=(1.0, -1.0, -90.0),
                            path={"kind": "turn_then_walk", "turn_deg": turn, "walk_m": 2.0}))


@pytest.mark.parametrize("bad", [True, False, "1", None, float("nan"), float("inf"), [1.0]], ids=repr)
def test_a_coordinate_that_is_not_a_finite_number_is_refused(bad):
    scenario = _scenario()
    scenario["scene"]["site"]["robot_start"]["xy_m"][0] = bad
    with pytest.raises(ValueError, match="robot_start"):
        load_site(scenario)


def test_a_pose_without_its_yaw_or_with_a_third_coordinate_is_refused():
    scenario = _scenario()
    del scenario["scene"]["site"]["robot_goal"]["yaw_deg"]
    with pytest.raises(ValueError, match="robot_goal"):
        load_site(scenario)
    scenario = _scenario()
    scenario["scene"]["site"]["robot_start"]["xy_m"].append(0.0)
    with pytest.raises(ValueError, match="robot_start"):
        load_site(scenario)


def test_a_region_whose_low_edge_is_above_its_high_edge_is_refused():
    with pytest.raises(ValueError, match="human_region"):
        load_site(_scenario(x_m=(1.0, 0.0)))


def test_a_region_without_one_of_its_axes_is_refused():
    scenario = _scenario()
    del scenario["scene"]["site"]["human_region"]["y_m"]
    with pytest.raises(ValueError, match="y_m"):
        load_site(scenario)


# ── the person's region ─────────────────────────────────────────────────────────

def test_a_region_with_both_axes_open_is_a_grid_with_its_corners_and_centre():
    site = load_site(_scenario(x_m=(1.0, 3.0), y_m=(9.75, 10.15)))
    points = human_region_points(site)
    assert points.shape == (9, 2)
    assert {(round(x, 6), round(y, 6)) for x, y in points} == {
        (x, y) for x in (1.0, 2.0, 3.0) for y in (9.75, 9.95, 10.15)}


def test_a_region_with_a_zero_width_axis_is_one_row():
    site = load_site(_scenario(x_m=(-5.745, -3.045), y_m=(4.65, 4.65)))
    points = human_region_points(site)
    assert points.shape == (3, 2)
    assert np.allclose(points, [[-5.745, 4.65], [-4.395, 4.65], [-3.045, 4.65]])


def test_the_grid_has_n_points_along_each_open_axis():
    site = load_site(_scenario(x_m=(0.0, 1.0), y_m=(0.0, 1.0)))
    assert human_region_points(site, n=5).shape == (25, 2)
    assert human_region_points(site, n=2).shape == (4, 2)


@pytest.mark.parametrize("n", [0, 1, -3, 2.5, True, "3", None], ids=repr)
def test_a_grid_that_is_not_two_or_more_points_is_refused(n):
    with pytest.raises(ValueError, match="n"):
        human_region_points(load_site(_scenario()), n=n)


# ── checked against the layout, on a layout of a few boxes ──────────────────────

def _problems(site, *boxes, clearance_m=CLEARANCE_M):
    return site_problems(site, [FLOOR, *boxes], clearance_m=clearance_m)


def _along_x(y=0.0, x0=-4.0, x1=4.0, **kw):
    return load_site(_scenario(start=(x0, y, 0.0), goal=(x1, y, 0.0), x_m=(0.0, 0.0), y_m=(-8.0, -8.0), **kw))


def test_a_path_beside_a_rack_at_the_clearance_is_not_a_problem():
    # the rack face is 0.5 m from the path, 0.05 m more than the clearance
    assert _problems(_along_x(), _box("rack/a", (-1.0, 0.5), (1.0, 1.5))) == []


def test_a_path_closer_to_a_rack_than_the_clearance_is_a_problem_and_names_it():
    lines = _problems(_along_x(), _box("rack/a", (-1.0, 0.4), (1.0, 1.4)))
    assert len(lines) == 1
    assert lines[0].startswith("T1 site.path: ") and "rack/a" in lines[0] and "0.400 m" in lines[0]
    assert "clearance 0.45 m" in lines[0]


def test_the_clearance_is_a_parameter():
    rack = _box("rack/a", (-1.0, 0.5), (1.0, 1.5))
    assert _problems(_along_x(), rack, clearance_m=0.6) != []
    assert _problems(_along_x(), rack, clearance_m=0.45) == []


def test_the_distance_to_a_footprint_is_euclidean_at_a_corner():
    """The path ends at (4, 0). A rack whose corner is 0.3 m over and 0.3 m along is sqrt(0.18) = 0.424 m
    away -- inside the clearance -- and one 0.4 and 0.4 away is sqrt(0.32) = 0.566 m, outside it."""
    near = _problems(_along_x(), _box("rack/n", (4.3, 0.3), (5.3, 1.3)))
    assert len(near) == 1 and "0.424 m" in near[0]
    assert _problems(_along_x(), _box("rack/f", (4.4, 0.4), (5.4, 1.4))) == []


def test_a_path_through_a_rack_is_a_problem_with_zero_distance():
    lines = _problems(_along_x(), _box("rack/a", (-1.0, -0.5), (1.0, 0.5)))
    assert len(lines) == 1 and "0.000 m" in lines[0]


def test_only_what_stands_in_the_walkers_height_counts():
    path = _along_x()
    assert _problems(path, _box("rack/high", (-1.0, 0.1), (1.0, 1.0), z=(2.5, 4.0))) == []     # above 2 m
    assert _problems(path, _box("rack/low", (-1.0, 0.1), (1.0, 1.0), z=(0.0, 0.5))) != []      # a pallet
    assert _problems(path, _box("rack/reach", (-1.0, 0.1), (1.0, 1.0), z=(1.9, 4.0))) != []    # reaches down into 2 m
    assert _problems(path, _box("light/a", (-1.0, 0.1), (1.0, 1.0), collider=False, semantic=None)) == []
    assert _problems(path, _box("rack/pit", (-1.0, 0.1), (1.0, 1.0), z=(-1.0, -0.1))) == []   # wholly under the floor


def test_each_rack_in_the_way_is_named_once_nearest_first():
    lines = _problems(_along_x(), _box("rack/b", (-1.0, 0.4), (1.0, 1.4)), _box("rack/a", (2.0, 0.3), (3.0, 1.3)))
    assert [("rack/a" in line, "rack/b" in line) for line in lines] == [(True, False), (False, True)]


def test_a_path_that_leaves_the_floor_is_a_problem():
    lines = _problems(_along_x(x0=-4.0, x1=12.0))
    assert len(lines) == 1 and lines[0].startswith("T1 site.path: ") and "건물 밖" in lines[0]


def test_a_person_inside_a_rack_footprint_is_a_problem_and_names_it():
    site = load_site(_scenario(start=(-4.0, -5.0, 0.0), goal=(4.0, -5.0, 0.0), x_m=(-0.5, 0.5), y_m=(0.8, 1.2)))
    lines = _problems(site, _box("rack/a", (-1.0, 0.5), (1.0, 1.5)))
    assert len(lines) == 1 and lines[0].startswith("T1 site.human_region: ")
    assert "rack/a" in lines[0] and "발자국 안" in lines[0]


def test_a_person_beside_a_rack_face_is_not_a_problem_however_close():
    """The canon people stand 0.35 m from a rack face by design: that is inside the walker's clearance
    and still not a problem for a person's region."""
    site = load_site(_scenario(start=(-4.0, -5.0, 0.0), goal=(4.0, -5.0, 0.0), x_m=(-0.5, 0.5), y_m=(0.15, 0.15)))
    assert _problems(site, _box("rack/a", (-1.0, 0.5), (1.0, 1.5))) == []
    touching = load_site(_scenario(start=(-4.0, -5.0, 0.0), goal=(4.0, -5.0, 0.0), x_m=(-0.5, 0.5), y_m=(0.5, 0.5)))
    assert _problems(touching, _box("rack/a", (-1.0, 0.5), (1.0, 1.5))) != []              # on the face: inside


def test_a_collider_thinner_than_the_grid_spacing_is_still_found_in_a_region():
    """A region 2 m wide has grid points every 1 m; an upright 0.1 m wide between them has none inside."""
    site = load_site(_scenario(start=(-4.0, -5.0, 0.0), goal=(4.0, -5.0, 0.0), x_m=(0.0, 2.0), y_m=(0.0, 0.0)))
    lines = _problems(site, _box("rack/upright", (0.45, -1.0), (0.55, 1.0)))
    assert len(lines) == 1 and "rack/upright" in lines[0]
    assert "(0.500, 0.000)" in lines[0]


def test_a_person_outside_the_building_is_a_problem():
    site = load_site(_scenario(start=(-4.0, -5.0, 0.0), goal=(4.0, -5.0, 0.0), x_m=(9.0, 11.0), y_m=(0.0, 0.0)))
    lines = _problems(site)
    assert len(lines) == 1 and lines[0].startswith("T1 site.human_region: ") and "건물 밖" in lines[0]


def test_every_line_starts_with_the_scenario_and_the_place():
    site = load_site(_scenario(start=(-4.0, 0.0, 0.0), goal=(12.0, 0.0, 0.0), x_m=(-0.5, 0.5), y_m=(0.8, 1.2)))
    lines = _problems(site, _box("rack/a", (-1.0, 0.3), (1.0, 1.5)))
    assert len(lines) >= 3
    assert all(line.startswith(("T1 site.path: ", "T1 site.human_region: ")) for line in lines)


def test_layout_boxes_without_a_floor_are_refused():
    with pytest.raises(ValueError, match="floor"):
        site_problems(_along_x(), [_box("rack/a", (-1.0, 0.5), (1.0, 1.5))])


@pytest.mark.parametrize("clearance", [-0.1, float("nan"), float("inf"), True, "0.45", None], ids=repr)
def test_a_clearance_that_is_not_a_non_negative_length_is_refused(clearance):
    with pytest.raises(ValueError, match="clearance_m"):
        site_problems(_along_x(), [FLOOR], clearance_m=clearance)


# ── the canon's sites on the canon's warehouse ──────────────────────────────────

def _canon_scenarios():
    return json.loads((REPO_ROOT / "meta" / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]


def _canon(sid):
    return next(s for s in _canon_scenarios() if s["scenario_id"] == sid)


@pytest.fixture(scope="module")
def boxes():
    return layout.build_layout(layout.load_spec(layout.SCENE_SPEC))


NEW_SCENARIOS = ["S1e", "S1f", "S1g"]


@pytest.mark.parametrize("sid", NEW_SCENARIOS)
def test_the_canon_sites_have_no_problems(sid, boxes):
    assert site_problems(load_site(_canon(sid)), boxes) == []


def test_only_the_new_scenarios_declare_a_site():
    assert {s["scenario_id"] for s in _canon_scenarios() if "site" in s["scene"]} == set(NEW_SCENARIOS)


@pytest.mark.parametrize("sid", NEW_SCENARIOS)
def test_a_canon_site_says_it_is_the_project_s_own(sid):
    # What the site is (the project's own setting), not how far work on it is.
    note = _canon(sid)["scene"]["site_note"]
    assert note.startswith("자체 설정. "), note
    assert "계획 단계" not in note


@pytest.mark.parametrize("sid", NEW_SCENARIOS)
def test_the_canon_path_starts_and_ends_at_the_poses_it_declares(sid):
    site = load_site(_canon(sid))
    path = nominal_path(site)
    assert np.allclose(path[0], [site.robot_start.x, site.robot_start.y, math.radians(site.robot_start.yaw_deg)])
    assert np.allclose(path[-1, :2], [site.robot_goal.x, site.robot_goal.y])
    assert _same_angle(path[-1, 2], math.radians(site.robot_goal.yaw_deg))
    assert len(path) > 2 and np.isfinite(path).all()


@pytest.mark.parametrize("sid", NEW_SCENARIOS)
def test_the_canon_path_never_jumps_further_than_a_step(sid):
    path = nominal_path(load_site(_canon(sid)), step_m=0.05)
    assert np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).max() <= 0.05 + 1e-9
    assert np.abs(np.diff(path[:, 2])).max() <= math.radians(5.0) + 1e-9      # continuous, never wrapped


def test_the_u_turn_runs_round_the_row_end():
    path = nominal_path(load_site(_canon("S1e")))
    west = path[np.argmin(path[:, 0])]
    assert west[0] == pytest.approx(-6.045 - 1.75, abs=1e-3)
    assert west[1] == pytest.approx(5.5, abs=1e-3)
    assert tuple(path[0]) == pytest.approx((-4.5, 7.25, math.pi))
    assert path[-1, 0] == pytest.approx(-2.0) and path[-1, 1] == pytest.approx(3.75)
    assert _same_angle(path[-1, 2], 0.0)
    assert path[-1, 2] == pytest.approx(TWO_PI)                               # the column is not wrapped
    assert len(path) == 1 + 31 + 110 + 81                                     # 1.545 m, a 5.498 m arc, 4.045 m


def test_the_turn_then_walk_ends_facing_the_other_rack():
    # From in front of rack B (facing -y) a half turn, then 1.0 m toward rack A (+y).
    path = nominal_path(load_site(_canon("S1g")))
    assert path[0, 2] == pytest.approx(-math.pi / 2)
    assert _same_angle(path[-1, 2], path[0, 2] + math.pi)
    assert path[-1, 2] - path[0, 2] == pytest.approx(math.pi)                 # a half turn, counted not wrapped
    assert path[-1, 1] == pytest.approx(10.3 + 1.0)
    assert path[-1, 0] == pytest.approx(2.0)
    turning = path[:37]                                                       # the start and 36 rows of 5 degrees
    assert np.allclose(turning[:, :2], [2.0, 10.3])
    assert path[36, 2] == pytest.approx(path[0, 2] + math.pi)
    assert len(path) == 1 + 36 + 20


def test_the_straight_path_runs_the_length_of_the_aisle_between_the_racks():
    path = nominal_path(load_site(_canon("S1f")))
    assert len(path) == 1 + 160
    assert np.allclose(path[:, 1], -7.25) and np.allclose(path[:, 2], 0.0)
    assert path[0, 0] == pytest.approx(-3.5) and path[-1, 0] == pytest.approx(4.5)


def test_a_path_through_a_rack_is_a_problem(boxes):
    """S1f with both ends moved to y = -6.2, 0.2 m from the rack face the aisle runs between."""
    scenario = copy.deepcopy(_canon("S1f"))
    site = scenario["scene"]["site"]
    site["robot_start"]["xy_m"][1] = -6.2
    site["robot_goal"]["xy_m"][1] = -6.2
    lines = site_problems(load_site(scenario), boxes)
    assert lines and all(line.startswith("S1f site.path: ") for line in lines)
    assert "0.200 m" in lines[0] and "clearance" in lines[0]
    assert all("clearance" in line for line in lines)


def _nearest_rack(boxes, sid):
    solid = [layout.bounds(b) for b in boxes if b.collider and layout.bounds(b)[0][2] < 2.0]
    return min(math.hypot(max(lo[0] - x, 0.0, x - hi[0]), max(lo[1] - y, 0.0, y - hi[1]))
               for x, y in human_region_points(load_site(_canon(sid))) for lo, hi in solid)


def test_the_canon_people_stand_inside_the_walkers_clearance_of_a_rack_by_design(boxes):
    """Why a region is not held to the clearance: in S1e and S1f the people stand beside a rack face, 0.3 m from it
    at the nearest in S1e and 0.35 m in S1f."""
    for sid in ("S1e", "S1f"):
        nearest = _nearest_rack(boxes, sid)
        assert 0.0 < nearest < CLEARANCE_M, (sid, nearest)
    assert _nearest_rack(boxes, "S1e") == pytest.approx(0.3) and _nearest_rack(boxes, "S1f") == pytest.approx(0.35)


def test_the_s1g_people_walk_down_the_middle_of_the_aisle(boxes):
    # S1g's person walks along the aisle within 0.3 m of its middle (y 10.75, 1.25 m from either rack row).
    assert _nearest_rack(boxes, "S1g") == pytest.approx(1.25 - 0.3)


def test_a_site_is_what_the_dataclasses_say():
    site = load_site(_canon("S1g"))
    assert isinstance(site, Site) and site.scenario_id == "S1g"
    assert site.robot_start == Pose2(2.0, 10.3, -90.0) and site.robot_goal == Pose2(2.0, 11.3, 90.0)
    assert site.path == {"kind": "turn_then_walk", "turn_deg": 180.0, "walk_m": 1.0}
    assert site.human_x_m == (1.0, 3.0) and site.human_y_m == (10.45, 11.05)
