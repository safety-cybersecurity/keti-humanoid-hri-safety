"""Conditions tables (generation/conditions.py): one canon table per new scenario, each row drawn from its own seed
and guarded against the racks, and the place each row gives the robot and the person. Isaac-free."""

import copy
import csv
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from generation import carry, conditions, sites
from generation.dummy.gait import gait_table
from generation.robots import kinematics
from generation.scene import layout
from generation.scene.carried_box import NOMINAL_SIZE_M

REPO = Path(__file__).resolve().parents[2]
NEW = ("S1e", "S1f", "S1g")
TWO_PI = 2 * math.pi


def _canon():
    return json.loads((REPO / "meta" / "scenarios.json").read_text(encoding="utf-8"))


def _entry(sid):
    return copy.deepcopy(next(s for s in _canon()["scenarios"] if s["scenario_id"] == sid))


@pytest.fixture(scope="module")
def boxes():
    return layout.build_layout(layout.load_spec(layout.SCENE_SPEC))


WIDE_ROWS = 400                  # the statistical tests look at a wider table than the canon's, drawn from the same windows


def _wide(sid):
    """The canon entry with WIDE_ROWS random rows and no grid: the same seeds, so its first rows are the canon's."""
    entry = _entry(sid)
    entry["conditions"]["rows"] = entry["episodes"] = WIDE_ROWS
    entry["conditions"].pop("grid", None)
    return entry


@pytest.fixture(scope="module")
def tables(boxes):
    return {sid: conditions.table(_wide(sid), boxes) for sid in NEW}


def _floor_only():
    return [layout.Box("floor", (0.0, 0.0, 0.0), (80.0, 80.0, 0.001), "m", False, None)]


# ── the table and its draw ─────────────────────────────────────────────────────

@pytest.mark.parametrize("sid", NEW)
def test_the_table_has_a_row_per_episode(sid, tables, boxes):
    entry = _entry(sid)
    spec = conditions.load_conditions(entry)
    assert spec.scenario_id == sid and spec.total == entry["episodes"] == spec.rows + spec.grid.cells ** 2
    table = conditions.table(entry, boxes)
    assert len(table) == spec.rows
    assert all(set(row) == set(spec.columns) | set(spec.fixed) for row in table)
    assert table == tables[sid][:spec.rows]                      # the wider table starts with the canon's rows


def test_rows_are_the_same_in_any_order(boxes):
    first = conditions.draw_row(_entry("S1e"), 7, boxes)
    for index in reversed(range(12)):
        conditions.draw_row(_entry("S1e"), index, boxes)
    assert conditions.draw_row(_entry("S1e"), 7, boxes) == first
    assert conditions.table(_entry("S1e"), boxes)[7] == first
    # another scenario's row of the same number is another draw: the scenario is in the seed
    delays = {sid: conditions.draw_row(_entry(sid), 7, boxes)["sensor_delay_ms"] for sid in NEW}
    assert len(set(delays.values())) == 3, delays


def test_the_seed_is_the_sha256_of_scenario_row_and_attempt():
    digest = hashlib.sha256(b"S1f:condition:12:3").digest()
    assert conditions.row_seed("S1f", 12, 3) == int.from_bytes(digest[:8], "big")
    assert conditions.row_seed("S1f", 12, 3) != conditions.row_seed("S1f", 12, 4) != conditions.row_seed("S1g", 12, 4)


def test_a_row_drawn_at_its_first_attempt_is_the_columns_in_their_order_from_that_seed():
    """S1f's row 0 on an empty floor (nothing to refuse it): the columns drawn in the canon's order from the
    first attempt's generator, the fixed values beside them."""
    entry = _entry("S1f")
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(b"S1f:condition:0:0").digest()[:8], "big"))
    want = {name: float(rng.uniform(*col["range"])) for name, col in entry["conditions"]["columns"].items()}
    assert conditions.draw_row(entry, 0, _floor_only()) == want


TOTAL = "the number of rows"


@pytest.mark.parametrize("index", [-1, TOTAL, 2.0, True, "3"], ids=repr)
def test_a_row_number_outside_the_table_is_refused(index, boxes):
    if index == TOTAL:
        index = conditions.load_conditions(_entry("S1f")).total
    with pytest.raises(ValueError, match="행"):
        conditions.draw_row(_entry("S1f"), index, boxes)


@pytest.mark.parametrize("sid", NEW)
def test_every_canon_row_passes_its_guard(sid, tables, boxes):
    entry = _entry(sid)
    assert all(conditions.row_problems(entry, row, boxes) == [] for row in tables[sid])


@pytest.mark.parametrize("sid", NEW)
def test_columns_stay_in_their_ranges(sid, tables):
    block = _entry(sid)["conditions"]
    for name, col in block["columns"].items():
        values = [row[name] for row in tables[sid]]
        if "range" in col:
            lo, hi = col["range"]
            assert all(isinstance(v, float) and lo <= v <= hi for v in values), name
            assert max(values) - min(values) > 0.9 * (hi - lo), name          # the draw covers the window
        else:
            assert set(values) == set(col["choice"]), name
    for name, value in block["fixed"].items():
        assert all(row[name] == value for row in tables[sid]), name


def test_a_row_that_cannot_pass_is_refused():
    """A layout whose one collider covers the whole floor: every attempt is refused, and the error names the row."""
    wall_to_wall = layout.Box("rack/everything", (0.0, 0.0, 1.0), (80.0, 80.0, 2.0), "m", True, "rack")
    with pytest.raises(ValueError, match=rf"S1e 조건표 행 7: 시도 {conditions.MAX_ATTEMPTS}번"):
        conditions.draw_row(_entry("S1e"), 7, [*_floor_only(), wall_to_wall])


def test_choices_are_drawn_both_ways(tables):
    sides = [row["approach_side"] for row in tables["S1g"]]
    assert set(sides) == {"west", "east"}
    assert sides.count("west") >= 150 and sides.count("east") >= 150, (sides.count("west"), sides.count("east"))


# ── what a row asks of the robot and the person ───────────────────────────────

def _u_turn_length(site, row):
    """Worked out from the row, not the nominal path: the straight in to the row end, the half circle, the straight out."""
    cx = site.path["turn_center_xy_m"][0]
    return abs(site.robot_start.x - cx) + math.pi * row["turn_radius_m"] + abs(site.robot_goal.x - cx)


def test_the_slowest_row_finishes_within_fifteen_seconds(tables):
    s1e = sites.load_site(_entry("S1e"))
    s1f = sites.load_site(_entry("S1f"))
    s1e_times = [_u_turn_length(s1e, row) / row["robot_speed_mps"] + 1.0 for row in tables["S1e"]]
    s1f_length = s1f.robot_goal.x - s1f.robot_start.x
    s1f_times = [s1f_length / row["robot_speed_mps"] + 1.0 for row in tables["S1f"]]
    s1g_times = [row["run_up_s"] + math.pi / row["turn_rate_rad_s"] + 0.5 + 1.0 / row["walk_speed_mps"] + 1.0
                 for row in tables["S1g"]]
    assert max(s1e_times) <= 15.0 and max(s1f_times) <= 15.0 and max(s1g_times) <= 15.0
    # and the windows' slowest corner, which bounds every row the table could ever draw
    cols = {sid: _entry(sid)["conditions"]["columns"] for sid in NEW}
    corner = dict(turn_radius_m=cols["S1e"]["turn_radius_m"]["range"][1])
    assert _u_turn_length(s1e, corner) / cols["S1e"]["robot_speed_mps"]["range"][0] + 1.0 <= 15.0
    assert s1f_length / cols["S1f"]["robot_speed_mps"]["range"][0] + 1.0 <= 15.0
    fixed = _entry("S1g")["conditions"]["fixed"]
    assert (fixed["run_up_s"] + math.pi / cols["S1g"]["turn_rate_rad_s"]["range"][0] + 0.5
            + 1.0 / fixed["walk_speed_mps"] + 1.0) <= 15.0


def test_s1e_rows_reach_into_the_robots_lane_and_clear_it(tables):
    """The gap across the robot's lane to the person's centre, g = r - f - 0.5: rows the robot walks into and rows
    it passes clear of."""
    g = np.array([row["turn_radius_m"] - row["face_distance_m"] - 0.5 for row in tables["S1e"]])
    assert (g < 0.42).mean() >= 0.20 and (g > 0.92).mean() >= 0.05, ((g < 0.42).mean(), (g > 0.92).mean())


def test_s1e_row_site_moves_the_lane_and_the_person(tables):
    site = sites.load_site(_entry("S1e"))
    row = tables["S1e"][3]
    r, c, f = row["turn_radius_m"], row["corner_distance_m"], row["face_distance_m"]
    moved = conditions.row_site(site, row)
    assert (moved.robot_start.x, moved.robot_start.y, moved.robot_start.yaw_deg) == pytest.approx((-4.5, 5.5 + r, 180.0))
    assert (moved.robot_goal.x, moved.robot_goal.y, moved.robot_goal.yaw_deg) == pytest.approx((-2.0, 5.5 - r, 0.0))
    assert moved.path == {"kind": "u_turn", "turn_center_xy_m": (-6.045, 5.5), "radius_m": r}
    assert moved.human_x_m == pytest.approx((-6.045 + c,) * 2) and moved.human_y_m == pytest.approx((5.0 - f,) * 2)
    path = sites.nominal_path(moved)
    assert np.allclose(path[-1, :2], [-2.0, 5.5 - r]) and path[:, 0].min() == pytest.approx(-6.045 - r, abs=1e-3)


def test_s1f_row_site_puts_the_person_at_the_row_s_x(tables):
    site = sites.load_site(_entry("S1f"))
    row = tables["S1f"][5]
    moved = conditions.row_site(site, row)
    assert moved.robot_start == site.robot_start and moved.robot_goal == site.robot_goal and moved.path == site.path
    assert moved.human_x_m == (row["human_start_x_m"],) * 2 and moved.human_y_m == (-6.35, -6.35)


def test_s1f_faces_the_rack_beside_it_with_its_back_to_the_lane():
    """The facing is the site's: square to the aisle, toward the rack face on the person's side of the robot's lane --
    a person on the line across the lane faces the other way."""
    site = sites.load_site(_entry("S1f"))
    assert conditions.s1f_rack_facing_yaw_deg(site) == pytest.approx(90.0)
    across = dataclasses.replace(site, human_y_m=(-8.15, -8.15))                  # 0.35 m off the far rack face
    assert conditions.s1f_rack_facing_yaw_deg(across) == pytest.approx(-90.0)
    with pytest.raises(ValueError, match="S1f"):
        conditions.s1f_rack_facing_yaw_deg(dataclasses.replace(site, human_y_m=(-7.25, -7.25)))   # on the lane


def test_s1f_trigger_distance_is_the_lead_time_at_the_robot_speed(tables):
    assert conditions.s1f_trigger_distance({"trigger_lead_s": 2.0, "robot_speed_mps": 0.8}) == pytest.approx(1.6)
    for row in tables["S1f"][:20]:
        assert conditions.s1f_trigger_distance(row) == pytest.approx(row["trigger_lead_s"] * row["robot_speed_mps"])


def test_s1f_walks_from_its_place_and_stops_before_the_far_face(tables):
    site = sites.load_site(_entry("S1f"))
    for row in tables["S1f"][:20]:
        start, stop = conditions.human_walk(site, row)
        theta = math.radians(row["entry_angle_deg"])
        assert start == pytest.approx((row["human_start_x_m"], -6.35))
        assert stop[1] == pytest.approx(-8.5 + conditions.FAR_FACE_STOP_M)
        step = np.subtract(stop, start)
        assert step / np.linalg.norm(step) == pytest.approx((math.cos(theta), -math.sin(theta)))


@pytest.mark.parametrize("side, turn, walk_x", [("west", 180.0, 1.0), ("east", -180.0, -1.0)])
def test_s1g_turns_away_from_the_person(side, turn, walk_x, tables):
    """From the west the robot turns counter-clockwise: half way round it faces east, the way the person walks."""
    site = sites.load_site(_entry("S1g"))
    row = next(r for r in tables["S1g"] if r["approach_side"] == side)
    moved = conditions.row_site(site, row)
    assert moved.path["turn_deg"] == turn and moved.robot_goal == site.robot_goal
    path = sites.nominal_path(moved)
    half = path[18]                                       # the start and 18 rows of 5 degrees: half way round
    assert np.allclose(half[:2], [2.0, 10.3])
    assert math.cos(half[2]) == pytest.approx(walk_x) and abs(math.sin(half[2])) < 1e-9
    start, stop = conditions.human_walk(site, row)
    assert math.copysign(1.0, stop[0] - start[0]) == walk_x
    assert stop[0] == pytest.approx(2.0 + walk_x * conditions.WALK_PAST_M)


def test_s1g_trigger_and_start_distances(tables):
    row = tables["S1g"][0]
    v, w, p, t = row["human_speed_mps"], row["turn_rate_rad_s"], row["pass_offset_s"], row["run_up_s"]
    assert conditions.s1g_trigger_distance(row) == pytest.approx(v * (math.pi / w + p))
    assert conditions.s1g_human_start_distance(row) == pytest.approx(v * (math.pi / w + p) + v * (t - 0.5))


@pytest.mark.parametrize("speed", [0.8, 1.2, 1.6])
def test_the_s1g_person_reaches_the_trigger_distance_at_its_run_up_at_full_speed(speed):
    """Stepped the way the dummy's gait drive ramps (gait_table.ramp_speed, 1 s from rest, 0.02 s steps): the person
    started s1g_human_start_distance away covers the gap to the trigger distance in run_up_s."""
    row = {"human_speed_mps": speed, "turn_rate_rad_s": 0.7, "pass_offset_s": 0.5, "run_up_s": 1.5}
    gap = conditions.s1g_human_start_distance(row) - conditions.s1g_trigger_distance(row)
    v, walked, t, dt = 0.0, 0.0, 0.0, 0.02
    while walked < gap:
        v = float(gait_table.ramp_speed(v, speed, dt, 1.0, 1.0))
        walked += v * dt
        t += dt
    assert t == pytest.approx(row["run_up_s"], abs=0.02) and v == pytest.approx(speed)


def _load_face(boxes, row_tag, x, side):
    """The aisle-side face (y) of a rack row's bottom-tier loads around x: the north face (side +1) or the south (-1)."""
    faces = [layout.bounds(b) for b in boxes if b.name.startswith(f"rack/load/{row_tag}/")]
    near = [(lo, hi) for lo, hi in faces if lo[0] - 1.5 <= x <= hi[0] + 1.5]
    return max(hi[1] for _, hi in near) if side > 0 else min(lo[1] for lo, _ in near)


def _box_corners(pose, size):
    """The held box's corners for the robot's root at pose (x, y, yaw deg) on a level floor, through carry.box_obb."""
    held = carry.load_carry(_entry("S1g"))
    joints = kinematics.load_joints()
    link_pos, link_quat = kinematics.link_poses(joints, carry.arm_hold_joints(held, joints["joints"]),
                                                frame=joints["root_link"])[held.link]
    turn = kinematics.axis_rotation("Z", math.radians(pose[2]))
    obb = carry.box_obb(np.array([pose[0], pose[1], 0.0]) + turn @ link_pos,
                        kinematics.quat_from_matrix(turn @ kinematics.quat_to_matrix(link_quat)), held, size)
    rot = kinematics.quat_to_matrix(obb.rot_xyzw)
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
    return obb.center + (signs * obb.half) @ rot.T


def test_s1g_start_and_goal_clear_both_racks(boxes):
    """At the nominal box: its front 0.44 m from rack B's bottom-tier loads where the robot starts, 0.34 m from rack
    A's at the goal; the circle its corners sweep while turning 0.15 m clear of rack B."""
    site = sites.load_site(_entry("S1g"))
    b_face = _load_face(boxes, "spr1", 2.0, +1)
    a_face = _load_face(boxes, "spr2", 2.0, -1)
    assert (b_face, a_face) == pytest.approx((9.45, 12.05))
    start = _box_corners((site.robot_start.x, site.robot_start.y, site.robot_start.yaw_deg), NOMINAL_SIZE_M)
    goal = _box_corners((site.robot_goal.x, site.robot_goal.y, site.robot_goal.yaw_deg), NOMINAL_SIZE_M)
    assert start[:, 1].min() - b_face == pytest.approx(0.44, abs=0.01)
    assert a_face - goal[:, 1].max() == pytest.approx(0.34, abs=0.01)
    reach = np.hypot(*(start[:, :2] - (site.robot_start.x, site.robot_start.y)).T).max()
    assert site.robot_start.y - reach - b_face >= conditions.ROBOT_CLEARANCE_M


@pytest.mark.parametrize("sid", NEW)
def test_condition_columns_sit_in_the_site_regions(sid, tables):
    site = sites.load_site(_entry(sid))
    for row in tables[sid]:
        moved = conditions.row_site(site, row)
        assert site.human_x_m[0] - 1e-9 <= moved.human_x_m[0] <= moved.human_x_m[1] <= site.human_x_m[1] + 1e-9, row
        assert site.human_y_m[0] - 1e-9 <= moved.human_y_m[0] <= moved.human_y_m[1] <= site.human_y_m[1] + 1e-9, row
    if sid == "S1e":
        xs = [row["corner_distance_m"] - 6.045 for row in tables[sid]]
        ys = [5.0 - row["face_distance_m"] for row in tables[sid]]
        assert site.human_x_m == pytest.approx((min(xs), max(xs)), abs=0.01)
        assert site.human_y_m == pytest.approx((min(ys), max(ys)), abs=0.01)


def test_the_box_windows_are_sizes_the_carry_holds():
    """The four corners of the depth and height windows: carry_problems finds nothing wrong with holding them."""
    from generation import cameras, visibility
    rules = json.loads((REPO / "meta" / "rules.json").read_text(encoding="utf-8"))
    reach = next(r for r in rules["rules"] if r["rule_id"] == "R-TS15066-SSM-02")["stop_table"]["turning_reach_m"]
    for sid in NEW:
        cols = _entry(sid)["conditions"]["columns"]
        sizes = [(NOMINAL_SIZE_M[0], d, h) for d in cols["box_depth_m"]["range"] for h in cols["box_height_m"]["range"]]
        problems = carry.carry_problems(carry.load_carry(_entry(sid)), kinematics.load_joints(),
                                        visibility.load_link_boxes(), cameras.load(REPO / "meta/cameras.json"),
                                        sizes=sizes, reach_m=reach)
        assert problems == [], (sid, problems)


# ── the guard ─────────────────────────────────────────────────────────────────

def _sampled_distance(centre, axes, half, lo, hi, n=200):
    """The distance between a turned rectangle and an axis-aligned one, from points along both outlines (0 when an
    outline point of one lies in the other) -- an oracle independent of the guard's corner-and-axis reasoning."""
    def outline(c, ax, h):
        corners = [c + a * h[0] * ax[0] + b * h[1] * ax[1] for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        t = np.linspace(0.0, 1.0, n)[:, None]
        return np.concatenate([p + (q - p) * t for p, q in zip(corners, corners[1:] + corners[:1])])
    a = outline(centre, axes, half)
    b = outline((lo + hi) / 2, np.eye(2), (hi - lo) / 2)
    a_in_b = np.all((a >= lo) & (a <= hi), axis=1).any()
    b_in_a = (np.abs((b - centre) @ axes.T) <= half + 1e-12).all(axis=1).any()
    if a_in_b or b_in_a:
        return 0.0
    return float(np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min())


def test_the_guard_measures_the_gap_between_two_rectangles_seen_from_above():
    rng = np.random.default_rng(11)
    met = 0
    for _ in range(150):
        centre, yaw = rng.uniform(-3.0, 3.0, 2), rng.uniform(-math.pi, math.pi)
        axes = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
        half = rng.uniform(0.0, 1.0, 2) * (rng.uniform() > 0.2)            # a fifth of them points
        lo = rng.uniform(-2.0, 1.0, 2)
        hi = lo + rng.uniform(0.05, 2.0, 2)
        got = conditions._rect_aabb_distance(centre[None], axes[None], half[None], lo[None], hi[None])[0, 0]
        want = _sampled_distance(centre, axes, half, lo, hi)
        met += want == 0.0
        assert got == pytest.approx(want, abs=0.01), (centre, yaw, half, lo, hi)
    assert 10 <= met <= 140                                                # both meeting and apart are tried


def _floor_x(boxes):
    lo, hi = layout.bounds(next(b for b in boxes if b.name == "floor"))
    return lo[0], hi[0]


def test_a_s1g_row_is_refused_when_its_walk_from_either_side_would_leave_the_building(boxes):
    """A start 15.7 m from the robot's x: from the east it is inside the building, from the west beyond the wall.
    The west row is refused for its own walk, the east row for the same walk from the west -- the robot stands 12 m
    from the west wall and 32 m from the east one, so refusing only the side drawn would keep long run-ups for the
    east alone."""
    row = {"approach_side": "west", "lateral_offset_m": 0.0, "human_speed_mps": 1.6, "turn_rate_rad_s": 0.5,
           "pass_offset_s": 2.5, "box_mass_kg": 2.0, "box_depth_m": 0.275, "box_height_m": 0.25,
           "sensor_delay_ms": 0.0, "walk_speed_mps": 0.5, "run_up_s": 1.5}
    west_wall, east_wall = _floor_x(boxes)
    reach = conditions.s1g_human_start_distance(row)
    assert 2.0 - reach < west_wall + conditions.WALL_INSET_M and 2.0 + reach < east_wall - conditions.WALL_INSET_M
    west = conditions.row_problems(_entry("S1g"), row, boxes)
    assert any(line.startswith("S1g 사람 길: ") and "벽" in line for line in west), west
    east = conditions.row_problems(_entry("S1g"), dict(row, approach_side="east"), boxes)
    assert east and all(line.startswith("S1g 사람 길 (반대쪽 west 에서 오면): ") for line in east), east
    assert any("벽" in line for line in east), east
    # one that fits from both sides passes from both
    near = dict(row, pass_offset_s=-1.0, human_speed_mps=0.8)
    assert all(conditions.row_problems(_entry("S1g"), dict(near, approach_side=side), boxes) == []
               for side in conditions.APPROACH_SIDES)


def test_every_s1g_row_would_pass_coming_from_the_other_side(tables, boxes):
    """The guard does not depend on the side drawn: every canon row with its approach side swapped passes too, so the
    timing columns the table keeps are the same population on both sides."""
    entry = _entry("S1g")
    for row in tables["S1g"]:
        other = next(side for side in conditions.APPROACH_SIDES if side != row["approach_side"])
        assert conditions.row_problems(entry, dict(row, approach_side=other), boxes) == [], row


def test_a_s1e_lane_into_the_rack_is_refused(boxes, tables):
    """A turn so tight the robot's way round the row end runs into it: the body and the box are both named."""
    row = dict(tables["S1e"][0], turn_radius_m=0.6)
    problems = conditions.row_problems(_entry("S1e"), row, boxes)
    assert any(line.startswith("S1e 로봇 길: ") for line in problems), problems
    assert any(line.startswith("S1e 든 상자: ") for line in problems), problems


def test_a_person_too_close_to_a_rack_is_refused(boxes, tables):
    row = dict(tables["S1e"][0], face_distance_m=0.2)
    problems = conditions.row_problems(_entry("S1e"), row, boxes)
    assert problems and all(line.startswith("S1e 사람 길: ") for line in problems), problems
    assert any(f"{conditions.HUMAN_CLEARANCE_M:g} m" in line for line in problems)


def test_a_s1g_turn_too_close_to_rack_b_is_refused(boxes, tables):
    """The robot started 0.25 m nearer rack B: the circle its box sweeps while turning comes within the clearance."""
    entry = _entry("S1g")
    entry["scene"]["site"]["robot_start"]["xy_m"][1] -= 0.25
    entry["scene"]["site"]["robot_goal"]["xy_m"][1] -= 0.25
    problems = conditions.row_problems(entry, tables["S1g"][0], boxes)
    assert any(line.startswith("S1g 돌기 원: ") for line in problems), problems


def test_a_robot_that_leaves_the_building_is_refused(tables):
    """On a floor that ends just west of the row end, the half circle round it leaves the building."""
    small = [layout.Box("floor", (0.0, 0.0, 0.0), (13.0, 40.0, 0.001), "m", False, None)]
    problems = conditions.row_problems(_entry("S1e"), tables["S1e"][0], small)
    assert any(line.startswith("S1e 로봇 길: ") and "건물" in line for line in problems), problems


def test_layout_boxes_without_a_floor_are_refused(tables):
    with pytest.raises(ValueError, match="floor"):
        conditions.row_problems(_entry("S1f"), tables["S1f"][0], [])


# ── what the canon's prose says about the table ───────────────────────────────

@pytest.mark.parametrize("sid", NEW)
def test_the_note_says_the_seed_and_the_guard_the_code_runs(sid):
    note = _entry(sid)["conditions"]["note"]
    lo, hi = sites.OBSTACLE_Z_M
    for phrase in (f'SHA-256("{sid}:condition:i:n")',
                   f"로봇 몸 반폭 {conditions.ROBOT_HALF_WIDTH_M:.2f} m",
                   f"높이 {lo:g}~{hi:g} m 충돌체에서 {conditions.ROBOT_CLEARANCE_M:g} m 밖",
                   f"충돌체에서 {conditions.HUMAN_CLEARANCE_M:g} m 밖이고 건물 벽에서 {conditions.WALL_INSET_M:g} m 안쪽",
                   f"건물 안쪽으로 {conditions.ROBOT_CLEARANCE_M:g} m 들어와 있음",
                   f"{conditions.MAX_ATTEMPTS}번이 모두 걸리면"):
        assert phrase in note, (sid, phrase)
    guard = note[note.index("가드("):note.index("에 걸리면")]
    # what the guard checks is what this scenario has: the held box, the turn on the spot, a walk, both sides
    holds = carry.load_carry(_entry(sid)) is not None
    kind = sites.load_site(_entry(sid)).path["kind"]
    assert ("든 상자" in guard) == holds, (sid, guard)
    assert ("돌기가 쓸고 가는 원이 충돌체에서" in guard) == (kind == "turn_then_walk"), (sid, guard)
    assert ("걷는 길" in guard) == (kind != "u_turn"), (sid, guard)
    assert ("어느 쪽에서 오든" in guard) == (kind == "turn_then_walk"), (sid, guard)
    if holds:
        assert f"폭은 {NOMINAL_SIZE_M[0]:.2f} m 로 고정" in note


def _span(col, unit=""):
    lo, hi = col["range"]
    return f"{lo:g}~{hi:g}{unit}".replace("-", "\u2212")


def test_the_prose_states_the_windows_the_table_draws():
    """The canon's sentences that repeat a window or a guard distance say what the table and the code do -- a window
    that moves leaves no sentence behind that states the old one."""
    e, f, g = _entry("S1e"), _entry("S1f"), _entry("S1g")
    ce, cf, cg = (x["conditions"]["columns"] for x in (e, f, g))
    corner = f"면에서 {_span(ce['face_distance_m'])} m · 줄 끝에서 {_span(ce['corner_distance_m'])} m"
    assert corner in e["human_behavior"] and f"그 {corner}" in e["scene"]["site_note"]
    assert f"반지름({e['scene']['site']['path']['radius_m']:g} m)" in e["scene"]["site_note"]
    assert f"대각선({_span(cf['entry_angle_deg'], '°')})" in f["human_behavior"]
    assert f"돌아서({_span(cf['turn_time_s'])} s" in f["human_behavior"]
    assert f"({_span(cf['human_speed_mps'])} m/s)" in f["human_behavior"]
    assert f"건너편 랙 면 {conditions.FAR_FACE_STOP_M:g} m 앞에서 멈춘다" in f["human_behavior"]
    lateral = f"가운데에서 옆으로 {_span(cg['lateral_offset_m'])} m"
    assert lateral in g["human_behavior"] and lateral in g["scene"]["site_note"]
    assert f"({_span(cg['human_speed_mps'])} m/s)" in g["human_behavior"]
    assert f"로봇을 {conditions.WALK_PAST_M:g} m 지난 곳에서 멈춘다" in g["human_behavior"]
    walk = f"{g['scene']['site']['path']['walk_m']:.1f} m"
    assert f"{walk} 걷는다" in g["scene"]["site_note"] and f"{walk} 걸어가" in g["robot_task"]
    accel = yaml.safe_load(conditions.GAIT_SPEC.read_text(encoding="utf-8"))["ramp"]["accel_time_s"]
    assert f"걸음의 {accel:g} s 가속" in g["conditions"]["note"]


# ── what a conditions block is refused for ────────────────────────────────────

_DROP = object()


def _edited(sid, path, value):
    entry = _entry(sid)
    node = entry
    for key in path[:-1]:
        node = node[key]
    if value is _DROP:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return entry


@pytest.mark.parametrize("sid, path, value, match", [
    ("S1e", ("conditions",), _DROP, r"S1e\.conditions 가 없습니다"),
    ("S1e", ("conditions",), [], r"S1e\.conditions 가 객체가 아닙니다"),
    ("S1e", ("conditions", "note"), _DROP, r"S1e\.conditions\.note 가 없습니다"),
    ("S1e", ("conditions", "extra"), 1, r"S1e\.conditions 에 모르는 키 \['extra'\]"),
    ("S1e", ("conditions", "rows"), 399, r"S1e\.conditions\.rows \+ grid\.cells² 가 episodes"),
    ("S1e", ("conditions", "rows"), True, r"S1e\.conditions\.rows 가 양의 정수가 아닙니다"),
    ("S1e", ("conditions", "rows"), 0, r"S1e\.conditions\.rows 가 양의 정수가 아닙니다"),
    ("S1e", ("conditions", "determined_by"), "standard", r"S1e\.conditions\.determined_by 가 'project' 가 아닙니다"),
    ("S1e", ("conditions", "note"), " ", r"S1e\.conditions\.note 가 비어 있지 않은 문자열이 아닙니다"),
    ("S1e", ("conditions", "columns"), {}, r"S1e\.conditions\.columns 가 비어 있지 않은 객체가 아닙니다"),
    ("S1e", ("conditions", "columns", "turn_radius_m"), _DROP, r"S1e\.conditions\.turn_radius_m 가 없습니다"),
    ("S1e", ("conditions", "columns", "turn_radius_m"), {"range": [2.0, 1.25]},
     r"S1e\.conditions\.turn_radius_m 의 range"),
    ("S1e", ("conditions", "columns", "turn_radius_m"), {"range": [1.25, float("nan")]},
     r"S1e\.conditions\.turn_radius_m 의 range"),
    ("S1e", ("conditions", "columns", "turn_radius_m"), {"range": [1.25, True]},
     r"S1e\.conditions\.turn_radius_m 의 range"),
    ("S1e", ("conditions", "columns", "turn_radius_m"), {"range": [1.25]}, r"S1e\.conditions\.turn_radius_m 의 range"),
    ("S1e", ("conditions", "columns", "turn_radius_m"), {"range": [0.0, 2.0]},
     r"S1e\.conditions\.turn_radius_m 의 아래 끝이 양수가 아닙니다"),
    ("S1e", ("conditions", "columns", "turn_radius_m"), {"range": [1.25, 2.0], "choice": [1.5]},
     r"S1e\.conditions\.turn_radius_m 은 range · choice 가운데 하나만"),
    ("S1e", ("conditions", "columns", "turn_radius_m"), [1.25, 2.0], r"S1e\.conditions\.turn_radius_m 이 객체가 아닙니다"),
    ("S1e", ("conditions", "columns", "sensor_delay_ms"), {"range": [-10.0, 100.0]},
     r"S1e\.conditions\.sensor_delay_ms 의 아래 끝이 음수입니다"),
    ("S1e", ("conditions", "columns", "wind_mps"), {"range": [0.0, 1.0]}, r"S1e\.conditions\.wind_mps 는 이 시나리오가 읽지 않는 열"),
    ("S1e", ("conditions", "fixed"), {"run_up_s": 1.5}, r"S1e\.conditions\.fixed\.run_up_s 는 이 시나리오가 읽지 않는 값"),
    ("S1g", ("conditions", "fixed", "run_up_s"), _DROP, r"S1g\.conditions\.fixed\.run_up_s 가 없습니다"),
    ("S1g", ("conditions", "fixed", "run_up_s"), "1.5", r"S1g\.conditions\.fixed\.run_up_s 이 유한한 수가 아닙니다"),
    ("S1g", ("conditions", "fixed", "walk_speed_mps"), 0.0, r"S1g\.conditions\.fixed\.walk_speed_mps 이 양수가 아닙니다"),
    ("S1g", ("conditions", "columns", "approach_side"), {"choice": []}, r"S1g\.conditions\.approach_side 의 choice"),
    ("S1g", ("conditions", "columns", "approach_side"), {"choice": ["west", "west"]},
     r"S1g\.conditions\.approach_side 의 choice"),
    ("S1g", ("conditions", "columns", "approach_side"), {"choice": ["west", "north"]},
     r"S1g\.conditions\.approach_side 의 choice 는 \['west', 'east'\]"),
    ("S1g", ("conditions", "columns", "approach_side"), {"range": [0.0, 1.0]},
     r"S1g\.conditions\.approach_side 의 choice 는 \['west', 'east'\]"),
    ("S1g", ("conditions", "columns", "box_mass_kg"), _DROP, r"S1g\.conditions\.box_mass_kg 가 없습니다"),
    ("S1f", ("carry",), _DROP, r"S1f\.conditions\.box_mass_kg 는 이 시나리오가 읽지 않는 열"),     # no box, no box columns
    ("S1f", ("conditions", "columns", "entry_angle_deg"), {"range": [45.0, 180.0]},
     r"S1f\.conditions\.entry_angle_deg 는 0 과 180 사이"),
    # a numeric column's choices are held to the column's type and domain, value by value, as its range's ends are
    ("S1g", ("conditions", "columns", "turn_rate_rad_s"), {"choice": [0]},
     r"S1g\.conditions\.turn_rate_rad_s 의 choice 값 0 이 양수가 아닙니다"),
    ("S1g", ("conditions", "columns", "turn_rate_rad_s"), {"choice": [0.5, -1.0]},
     r"S1g\.conditions\.turn_rate_rad_s 의 choice 값 -1\.0 이 양수가 아닙니다"),
    ("S1f", ("conditions", "columns", "robot_speed_mps"), {"choice": ["fast"]},
     r"S1f\.conditions\.robot_speed_mps 의 choice 값 'fast' 이 유한한 수가 아닙니다"),
    ("S1f", ("conditions", "columns", "human_speed_mps"), {"choice": [0]},
     r"S1f\.conditions\.human_speed_mps 의 choice 값 0 이 양수가 아닙니다"),
    ("S1f", ("conditions", "columns", "sensor_delay_ms"), {"choice": [-20]},
     r"S1f\.conditions\.sensor_delay_ms 의 choice 값 -20 이 음수입니다"),
    ("S1f", ("conditions", "columns", "entry_angle_deg"), {"choice": [0]},
     r"S1f\.conditions\.entry_angle_deg 의 choice 값 0 은 0 과 180 사이"),
    ("S1f", ("conditions", "columns", "entry_angle_deg"), {"choice": [60, 180]},
     r"S1f\.conditions\.entry_angle_deg 의 choice 값 180 은 0 과 180 사이"),
    ("S1g", ("conditions", "columns", "turn_rate_rad_s"), {"choice": []},
     r"S1g\.conditions\.turn_rate_rad_s 의 choice 가 서로 다른 값"),
    ("S1e", ("conditions", "columns", "box_mass_kg"), {"choice": [True]},
     r"S1e\.conditions\.box_mass_kg 의 choice 가 서로 다른 값"),
])
def test_load_conditions_refuses_bad_blocks(sid, path, value, match):
    with pytest.raises(ValueError, match=match):
        conditions.load_conditions(_edited(sid, path, value))


def test_a_bad_choice_is_refused_before_a_row_is_drawn(boxes):
    # refused where the block is read, not later where a row's value is used (a zero turn rate divided by)
    with pytest.raises(ValueError, match=r"S1g\.conditions\.turn_rate_rad_s 의 choice 값 0"):
        conditions.draw_row(_edited("S1g", ("conditions", "columns", "turn_rate_rad_s"), {"choice": [0]}), 0, boxes)
    with pytest.raises(ValueError, match=r"S1f\.conditions\.robot_speed_mps 의 choice 값 'fast'"):
        conditions.draw_row(_edited("S1f", ("conditions", "columns", "robot_speed_mps"), {"choice": ["fast"]}), 0,
                            boxes)


def test_a_numeric_choice_in_its_domain_is_drawn(boxes):
    entry = _edited("S1g", ("conditions", "columns", "turn_rate_rad_s"), {"choice": [0.5, 1.0]})
    entry["conditions"].pop("grid")                       # a grid axis is a range column
    entry["episodes"] = entry["conditions"]["rows"]
    assert conditions.load_conditions(entry).columns["turn_rate_rad_s"] == conditions.Column("choice", (0.5, 1.0))
    assert conditions.draw_row(entry, 0, boxes)["turn_rate_rad_s"] in (0.5, 1.0)
    entry = _edited("S1f", ("conditions", "columns", "sensor_delay_ms"), {"choice": [0, 100]})   # 0 is not negative
    assert conditions.load_conditions(entry).columns["sensor_delay_ms"] == conditions.Column("choice", (0, 100))


def test_an_unknown_carry_link_is_refused_before_a_row_is_drawn(boxes):
    entry = _entry("S1g")
    entry["carry"]["link"] = "missing_link"
    with pytest.raises(ValueError, match=r"S1g\.carry\.link 'missing_link' 이 H1 의 링크가 아닙니다"):
        conditions.draw_row(entry, 0, boxes)


def test_a_scenario_without_a_site_has_no_conditions_table():
    with pytest.raises(ValueError, match="S1d"):
        conditions.load_conditions(_entry("S1d"))


def test_the_spec_is_what_the_dataclasses_say():
    spec = conditions.load_conditions(_entry("S1g"))
    assert list(spec.columns) == list(_entry("S1g")["conditions"]["columns"])         # the canon's order
    assert spec.columns["approach_side"] == conditions.Column("choice", ("west", "east"))
    assert spec.columns["turn_rate_rad_s"] == conditions.Column("range", (0.85, 1.0))
    assert dict(spec.fixed) == {"walk_speed_mps": 0.5, "run_up_s": 1.5}


def test_main_writes_the_table_as_csv(tmp_path):
    out = tmp_path / "s1f.csv"
    assert conditions.main(["--scenario", "S1f", "--out", str(out)]) == 0
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    total = conditions.load_conditions(_entry("S1f")).total
    assert len(rows) == total and [int(r["row"]) for r in rows] == list(range(total))
    spec = conditions.load_conditions(_entry("S1f"))
    assert all(int(r["attempt"]) >= 0 for r in rows[:spec.rows])
    assert [r["attempt"] for r in rows[spec.rows:]] == ["grid"] * (total - spec.rows)
    assert float(rows[0]["human_start_x_m"]) == conditions.draw_row(
        _entry("S1f"), 0, layout.build_layout(layout.load_spec(layout.SCENE_SPEC)))["human_start_x_m"]
