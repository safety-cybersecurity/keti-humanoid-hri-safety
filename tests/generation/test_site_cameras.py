"""generation/scene/check_site_cameras.py -- the third-person and CCTV cameras of S1e, S1f and S1g against the
warehouse layout, by sight lines: does each see the person and the robot where its site needs them?
Pure numpy, no Isaac."""
import copy
import json
import math
import unicodedata
from pathlib import Path

import numpy as np
import pytest

from generation import cameras, conditions, sites, visibility
from generation.dummy import joints_spec
from generation.scene import check_site_cameras as csc
from generation.scene import layout

REPO = Path(__file__).resolve().parents[2]


def _site(sid: str) -> sites.Site:
    doc = json.loads((REPO / "meta/scenarios.json").read_text(encoding="utf-8"))
    return sites.load_site(next(s for s in doc["scenarios"] if s["scenario_id"] == sid))


@pytest.fixture(scope="module")
def world():
    canon = cameras.load(REPO / "meta/cameras.json")
    body = visibility.body_sample_points(joints_spec.load(), visibility.seeing_rule(canon)["sample_points"])
    return canon, layout.build_layout(layout.load_spec(layout.SCENE_SPEC)), body


def test_the_criteria_are_the_projects_own_values():
    # The checking criteria the camera placements are held to (the project's own): written out here.
    assert (csc.CCTV_MIN_PERSON_FRACTION, csc.CCTV_MIN_HEIGHT_PX, csc.CCTV_ROBOT_NEAR_M, csc.CCTV_MIN_ROBOT_FRACTION) \
        == (0.8, 60.0, 1.5, 0.8)
    assert (csc.PATH_STEP_M, csc.FOLLOW_MIN_ROBOT_FRACTION, csc.FOLLOW_PERSON_NEAR_M,
            csc.FOLLOW_MIN_PERSON_FRACTION) == (0.25, 0.8, 2.0, 0.5)
    assert csc.ROBOT_BOX_M == (0.5, 0.3, 1.75)


# Each site (S1e, S1f, S1g) with each side its cameras are placed for: S1g's scene cameras are placed per side the
# person comes from (meta/cameras.json by_approach_side).
PLACED = [("S1e", None), ("S1f", None), ("S1g", "west"), ("S1g", "east")]


def test_the_sides_checked_are_the_canon_s(world):
    canon = world[0]
    assert [(sid, side) for sid in ("S1e", "S1f", "S1g") for side in csc.camera_sides(canon, sid)] == PLACED


def test_the_site_cameras_cover_their_sites(world):
    canon, boxes, body = world
    for sid, side in PLACED:
        report = csc.coverage(csc.side_site(_site(sid), side), canon, boxes, body, approach_side=side)
        assert report["approach_side"] == side
        assert csc.coverage_problems(report) == [], (sid, side)
        # every criterion was measured on something: corners and centre, path rows near the person, path rows
        assert len(report["cctv"]["person"]) == 5, (sid, side)
        assert report["cctv"]["robot"]["rows"] > 0 and report["follow"]["robot"]["rows"] > 0, (sid, side)
        assert report["follow"]["person"]["rows"] > 0, (sid, side)
        assert not report["cctv"]["eye_inside"] and report["follow"]["eye_inside_rows"] == 0, (sid, side)


@pytest.mark.parametrize("side", ["west", "east"])
def test_the_s1g_check_turns_the_robot_the_way_the_table_turns_it_from_that_side(side):
    # The check reads the robot's path; on a turn on the spot its direction is the row's (conditions.row_site: away
    # from the person), not the canon's default -- so each side's cameras are checked along the path its episodes
    # take. The rest of the site is the canon's.
    site = _site("S1g")
    turned = csc.side_site(site, side)
    want = conditions.row_site(site, {"approach_side": side, "lateral_offset_m": 0.0}).path["turn_deg"]
    assert turned.path["turn_deg"] == want
    assert (turned.robot_start, turned.robot_goal, turned.human_x_m, turned.human_y_m) == \
        (site.robot_start, site.robot_goal, site.human_x_m, site.human_y_m)
    # the table's rows from that side turn that way
    entry = conditions.scenario_entry("S1g")
    boxes = layout.build_layout(layout.load_spec(layout.SCENE_SPEC))
    row = next(r for r in (conditions.draw_row(entry, i, boxes) for i in range(conditions.load_conditions(entry).rows))
               if r["approach_side"] == side)
    assert conditions.row_site(site, row).path["turn_deg"] == want
    assert csc.side_site(site, None) is site and csc.side_site(_site("S1e"), side) == _site("S1e")


def _people_at_centre(site, body):
    # The person standing at the region's centre at each facing the check reads, placed here through the
    # standing pose and the sample points (visibility), not through the check's own helpers.
    spec = joints_spec.load()
    centre = (sum(site.human_x_m) / 2.0, sum(site.human_y_m) / 2.0)
    return [visibility.points_world(body, visibility.standing_part_poses(spec, centre, math.radians(d)))
            for d in csc.PERSON_FACINGS_DEG]


def _in_robot_box(row, points) -> bool:
    # The robot's box written out: ROBOT_BOX_M long along the heading, wide across it, tall from the floor.
    x, y, yaw = row
    length, width, height = csc.ROBOT_BOX_M
    d = np.asarray(points) - (x, y, 0.0)
    along = d[:, 0] * math.cos(yaw) + d[:, 1] * math.sin(yaw)
    across = -d[:, 0] * math.sin(yaw) + d[:, 1] * math.cos(yaw)
    return bool(((np.abs(along) <= length / 2) & (np.abs(across) <= width / 2)
                 & (d[:, 2] >= 0.0) & (d[:, 2] <= height)).any())


@pytest.mark.parametrize("side", ["west", "east"])
def test_s1g_ends_where_the_robot_stands_in_the_person_s_place(world, side):
    # S1g's robot turns about and walks into the person's place: the rows where its box holds the person are the
    # contact itself and are left out of the third-person person criterion -- exactly those rows, found here from
    # the geometry over every path row near the person.
    canon, boxes, body = world
    site = csc.side_site(_site("S1g"), side)
    report = csc.coverage(site, canon, boxes, body, approach_side=side)
    people = _people_at_centre(site, body)
    centre = (sum(site.human_x_m) / 2.0, sum(site.human_y_m) / 2.0)
    near = [row for row in sites.nominal_path(site, step_m=csc.PATH_STEP_M)
            if math.dist(row[:2], centre) < csc.FOLLOW_PERSON_NEAR_M]
    want = [row.tolist() for row in near if any(_in_robot_box(row, p) for p in people)]
    got = [o["at"] for o in report["follow"]["person"]["overlap_rows"]]
    assert got and len(got) == len(want) and np.allclose(got, want, rtol=0.0, atol=1e-12), (got, want)
    assert report["follow"]["person"]["rows"] == len(near) - len(want)


def test_a_hidden_person_away_from_the_robot_is_still_a_problem(world):
    # The exclusion is for the robot standing in the person's place, not for any shortfall: a crate over S1f's
    # person hides it from every camera while the robot walks 0.9 m off, and the third-person criterion says so.
    canon, boxes, body = world
    site = _site("S1f")
    centre = (sum(site.human_x_m) / 2.0, sum(site.human_y_m) / 2.0)
    crate = layout.Box(name="test/crate", center=(centre[0], centre[1], 1.0), size=(0.8, 0.5, 2.0),
                       material="pallet_grey", collider=True, semantic=None)
    report = csc.coverage(site, canon, [*boxes, crate], body)
    assert report["follow"]["person"]["overlap_rows"] == []
    assert any(line.startswith("S1f third_person_rgb: 사람 몫") for line in csc.coverage_problems(report))


def test_a_third_person_eye_inside_a_rack_is_a_problem(world):
    # The sight lines leave out a box round the camera, so a camera standing in a rack would see straight through
    # it: the check holds the eye out of the layout's boxes along the path. Moved into the rack row south of S1f's
    # aisle (y -6 .. -5) from behind the robot, it is found.
    canon, boxes, body = world
    moved = copy.deepcopy(canon)
    moved["cameras"]["third_person_rgb"]["sites"]["S1f"]["eye_offset"][1] = 1.7      # the robot's left: the rack
    report = csc.coverage(_site("S1f"), moved, boxes, body)
    assert report["follow"]["eye_inside_rows"] > 0
    assert any("눈이 레이아웃 상자 안" in line for line in csc.coverage_problems(report))


def _box(name, centre, size, collider=True):
    return layout.Box(name=name, center=centre, size=size, material="pallet_grey", collider=collider, semantic=None)


def _tiny_world(canon, start_xy, path, goal, region, eye_offset=(-3.0, 0.0, 3.0)):
    """A site of its own on an empty floor, the robot starting along +x, its cameras placed like S1e's: the
    third-person camera 3 m behind the robot and 3 m up (unless ``eye_offset`` says otherwise), turning with its
    heading; the CCTV anywhere (only the eye along the path is read here)."""
    canon = copy.deepcopy(canon)
    canon["cameras"]["third_person_rgb"]["sites"]["tiny"] = {
        "frame": "heading", "eye_offset": list(eye_offset), "lookat_offset": [0.5, 0.0, -1.0]}
    canon["cameras"]["cctv_rgb"]["sites"]["tiny"] = {"eye_m": [0.0, -6.0, 5.0], "aim_m": [0.0, 0.0, 1.0]}
    site = sites.load_site({"scenario_id": "tiny", "scene": {"site": {
        "robot_start": {"xy_m": list(start_xy), "yaw_deg": 0.0},
        "robot_goal": goal, "path": path, "human_region": region}}})
    floor = _box("floor", (0.0, 0.0, -0.001), (40.0, 40.0, 0.002), collider=False)
    return canon, site, floor


def test_the_eye_between_the_two_rows_of_a_tight_u_turn_is_held_out_of_boxes(world):
    # A U-turn of 1 mm radius has only its two ends as path rows; a camera 3 m behind the robot that turns with its
    # heading swings through half a circle between them. Halfway (yaw 90 deg, the root at the turn's centre) the
    # eye is 3 m to the robot's right and 3 m above the root at 1.05 m: (0, -3, 4.05). A 10 cm box there is found.
    canon, _, body = world
    canon, site, floor = _tiny_world(
        canon, (0.0, -0.001), {"kind": "u_turn", "turn_center_xy_m": [0.0, 0.0], "radius_m": 0.001},
        {"xy_m": [0.0, 0.001], "yaw_deg": 180.0}, {"x_m": [0.0, 0.0], "y_m": [0.8, 0.8]})
    assert len(sites.nominal_path(site, step_m=csc.EYE_STEP_M)) == 2
    report = csc.coverage(site, canon, [floor, _box("test/obstacle", (0.001, -3.0, 4.05), (0.1, 0.1, 0.1))], body)
    assert report["follow"]["eye_inside_rows"] > 0
    assert any("눈이 레이아웃 상자 안" in line for line in csc.coverage_problems(report))
    clear = csc.coverage(site, canon, [floor], body)
    assert clear["follow"]["eye_inside_rows"] == 0


def test_the_eye_between_the_rows_of_an_in_place_turn_is_held_out_of_boxes(world):
    # A turn on the spot takes a path row every 5 degrees, and a camera 3 m behind the robot moves 0.26 m between
    # two of them. Halfway between the first two (yaw 2.5 deg) the eye is at (-3 cos 2.5, -3 sin 2.5, 4.05) =
    # (-2.99714, -0.13086, 4.05), 0.13 m from the eyes at either row: a 10 cm box there is found.
    canon, _, body = world
    canon, site, floor = _tiny_world(
        canon, (0.0, 0.0), {"kind": "turn_then_walk", "turn_deg": 180.0, "walk_m": 0.7},
        {"xy_m": [-0.7, 0.0], "yaw_deg": 180.0}, {"x_m": [-2.0, -2.0], "y_m": [0.0, 0.0]})
    centre = (-3.0 * math.cos(math.radians(2.5)), -3.0 * math.sin(math.radians(2.5)), 4.05)
    report = csc.coverage(site, canon, [floor, _box("test/obstacle", centre, (0.1, 0.1, 0.1))], body)
    assert report["follow"]["eye_inside_rows"] > 0
    assert any("눈이 레이아웃 상자 안" in line for line in csc.coverage_problems(report))


@pytest.mark.parametrize("sid, side", PLACED)
def test_the_eye_samples_are_no_more_than_the_eye_step_apart(world, sid, side):
    # The eye is checked at the robot's path rows cut finer where the camera moves more than EYE_STEP_M between
    # two: its places, worked out here through cameras.follow_pose, are never further apart than that.
    canon = world[0]
    site = csc.side_site(_site(sid), side)
    placement = cameras.site_placement(canon["cameras"]["third_person_rgb"], sid, name="third_person_rgb",
                                       approach_side=side)
    rows = csc.eye_rows(site, placement)
    first = sites.nominal_path(site, step_m=csc.EYE_STEP_M)
    assert np.allclose(rows[0], first[0]) and np.allclose(rows[-1], first[-1])
    eyes = np.array([cameras.follow_pose(placement, (x, y, csc.ROBOT_ROOT_Z_M), yaw, float(first[0, 2]))[0]
                     for x, y, yaw in rows])
    assert np.linalg.norm(np.diff(eyes, axis=0), axis=1).max() <= csc.EYE_STEP_M + 1e-12
    # every path row is one of the samples, in order
    kept = [int(np.argmin(np.abs(rows - row).sum(axis=1))) for row in first]
    assert kept == sorted(kept) and all(np.allclose(rows[k], row) for k, row in zip(kept, first))


def test_the_eye_samples_count_the_root_s_move_and_the_turn_together(world):
    # A camera 3 m to the right of a robot on a U-turn to the left (radius 1.75 m) runs round the turn's centre
    # 4.75 m out: between two path rows (0.05 m apart along the 1.75 m arc) it moves 0.05 x 4.75 / 1.75 = 0.136 m,
    # the root's 0.05 m and the turn's 3 m x 0.0286 rad added. Its samples are still at most EYE_STEP_M apart.
    canon, site, _ = _tiny_world(
        world[0], (0.0, -1.75), {"kind": "u_turn", "turn_center_xy_m": [0.0, 0.0], "radius_m": 1.75},
        {"xy_m": [0.0, 1.75], "yaw_deg": 180.0}, {"x_m": [3.0, 3.0], "y_m": [0.0, 0.0]}, eye_offset=(0.0, -3.0, 3.0))
    placement = canon["cameras"]["third_person_rgb"]["sites"]["tiny"]
    rows = csc.eye_rows(site, placement)
    eyes = np.array([cameras.follow_pose(placement, (x, y, csc.ROBOT_ROOT_Z_M), yaw, 0.0)[0] for x, y, yaw in rows])
    steps = np.linalg.norm(np.diff(eyes, axis=0), axis=1)
    assert 0.04 < steps.max() <= csc.EYE_STEP_M + 1e-12


def test_a_cctv_behind_a_rack_fails_coverage(world):
    canon, boxes, body = world
    moved = copy.deepcopy(canon)
    moved["cameras"]["cctv_rgb"]["sites"]["S1f"]["eye_m"][1] = -10.0      # the next aisle south, behind a rack row
    problems = csc.coverage_problems(csc.coverage(_site("S1f"), moved, boxes, body))
    assert problems and all(line.startswith("S1f cctv_rgb") for line in problems), problems
    assert any("사람" in line for line in problems) and any("로봇" in line for line in problems), problems


def test_a_third_person_camera_looking_away_fails_coverage(world):
    canon, boxes, body = world
    moved = copy.deepcopy(canon)
    moved["cameras"]["third_person_rgb"]["sites"]["S1e"]["lookat_offset"] = [-6.0, 0.0, 1.5]   # behind itself
    problems = csc.coverage_problems(csc.coverage(_site("S1e"), moved, boxes, body))
    assert problems and all(line.startswith("S1e third_person_rgb") for line in problems), problems


def test_the_robot_box_hides_the_person_behind_it(world):
    # The robot's box stands between the third-person camera and a person on its far side: from straight behind,
    # a person 1 m ahead of the robot is mostly hidden; moved out from behind the box, seen.
    canon, boxes, body = world
    cam = canon["cameras"]["third_person_rgb"]
    eye, target = (-3.0, 7.25, 2.0), (0.0, 7.25, 1.0)
    pos, quat = cameras.look_at_pose(eye, target)
    robot = csc.robot_box((0.0, 7.25, 0.0))
    person = csc.standing_person(body, (1.0, 7.25), 0.0)["points"]
    lo, hi = visibility.layout_occluders(boxes)
    hidden = visibility.visible_fraction(cam, pos, quat, person, lo, hi, (robot,))
    clear = visibility.visible_fraction(cam, pos, quat, person, lo, hi)
    assert hidden < 0.5 <= clear


def test_main_prints_a_row_per_criterion_and_exits_0(capsys):
    code = csc.main([])
    lines = capsys.readouterr().out.splitlines()
    labels = {"S1e": ["S1e"], "S1f": ["S1f"], "S1g": ["S1g · 서쪽에서", "S1g · 동쪽에서"]}
    for sid, names in labels.items():
        for name in names:
            assert sum(line.startswith(name + " ") for line in lines) == 5, (name, lines)
            # report-only: the third-person eye's distance from the boxes and its path inside rack rows
            assert sum(line.startswith(f"  {name} ") and "보고만" in line for line in lines) == 1, (name, lines)
        assert sum(line.startswith(sid + " ") for line in lines) == 5 * len(names), (sid, lines)
    for name in labels["S1g"]:
        assert any(line.startswith(f"  {name} ") and "겹침" in line for line in lines), (name, lines)
    assert not [line for line in lines if line.startswith("PROBLEM")] and code == 0
    # The camera column starts at one place on the screen in every row and in the header, and two cells further in the
    # indented report-only lines: the site column is as wide as its widest label ("S1g · 서쪽에서" -- a Hangul letter
    # takes two cells).
    def cells(text):
        return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)

    def camera_at(line):
        return cells(line[:min(line.index(c) for c in ("cctv_rgb", "third_person_rgb") if c in line)])

    every = [n for names in labels.values() for n in names]
    header = next(line for line in lines if line.startswith("자리 "))
    column = cells(header[:header.index("카메라")])
    rows = [line for line in lines if any(line.startswith(n + " ") for n in every)]
    notes = [line for line in lines if any(line.startswith(f"  {n} ") for n in every)]
    assert len(rows) == 20 and len(notes) >= 4
    assert {camera_at(line) for line in rows} == {column}, (column, rows)
    assert {camera_at(line) for line in notes} == {column + 2}, (column, notes)


def test_a_problem_names_the_side_it_was_found_from(world):
    # S1g's CCTV from the west moved behind the rack row south of its aisle: the lines say which side's placement.
    canon, boxes, body = world
    moved = copy.deepcopy(canon)
    moved["cameras"]["cctv_rgb"]["sites"]["S1g"]["by_approach_side"]["west"]["eye_m"][1] = 6.0
    west = csc.coverage_problems(csc.coverage(csc.side_site(_site("S1g"), "west"), moved, boxes, body,
                                              approach_side="west"))
    assert west and all(line.startswith("S1g · 서쪽에서 cctv_rgb") for line in west), west
    east = csc.coverage(csc.side_site(_site("S1g"), "east"), moved, boxes, body, approach_side="east")
    assert csc.coverage_problems(east) == []
    with pytest.raises(ValueError, match="사람이 오는 쪽마다"):
        csc.coverage(_site("S1g"), canon, boxes, body)
