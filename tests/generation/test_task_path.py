"""The robot's task (generation/task_path.py): an ideal unicycle driven by the follower reaches each
scenario's goal inside the 15 s window, stays on the path, keeps inside the walking policy's command
range, and S1g waits, turns away from the person and then walks; the task's stages and the step each
ended at. Expectations are the canon's rows and ranges, the site geometry and closed forms."""

import copy
import functools
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from generation import conditions, sites, task_path
from generation.scene import layout

REPO = Path(__file__).resolve().parents[2]
DT = 0.02
POSITION_TOLERANCE_M = 0.3                # CompletionTracker's position tolerance (pos_tol_m)
WINDOW_S = 15.0


def _entry(sid):
    canon = json.loads((REPO / "meta" / "scenarios.json").read_text(encoding="utf-8"))
    return copy.deepcopy(next(s for s in canon["scenarios"] if s["scenario_id"] == sid))


def _span(sid, column):
    return _entry(sid)["conditions"]["columns"][column]["range"]


def _extreme_rows():
    """The slowest and fastest row of each scenario by the canon's own ranges."""
    cases = []
    r_lo, r_hi = _span("S1e", "turn_radius_m")
    v_lo, v_hi = _span("S1e", "robot_speed_mps")
    mid = {"corner_distance_m": 0.9, "face_distance_m": 0.5}
    cases += [("S1e", "slowest", {**mid, "turn_radius_m": r_hi, "robot_speed_mps": v_lo}),
              ("S1e", "fastest", {**mid, "turn_radius_m": r_lo, "robot_speed_mps": v_hi})]
    v_lo, v_hi = _span("S1f", "robot_speed_mps")
    cases += [("S1f", "slowest", {"human_start_x_m": 1.0, "robot_speed_mps": v_lo}),
              ("S1f", "fastest", {"human_start_x_m": 1.0, "robot_speed_mps": v_hi})]
    w_lo, w_hi = _span("S1g", "turn_rate_rad_s")
    fixed = _entry("S1g")["conditions"]["fixed"]
    for side in ("west", "east"):
        for label, rate in (("slowest", w_lo), ("fastest", w_hi)):
            cases.append(("S1g", f"{label}-{side}", {"approach_side": side, "lateral_offset_m": 0.0,
                                                     "turn_rate_rad_s": rate, "walk_speed_mps": fixed["walk_speed_mps"]}))
    return cases


CASES = _extreme_rows()


def _follower(sid, row):
    site = conditions.row_site(sites.load_site(_entry(sid)), row)
    return site, task_path.PathFollower(site, row, dt=DT)


def _drive(site, follower, *, start_step=0, seconds=WINDOW_S):
    """An ideal unicycle: the command is the motion. Returns the trace, the tracker and the commands."""
    start = site.robot_start
    xy, yaw = np.array([start.x, start.y]), math.radians(start.yaw_deg)
    tracker = task_path.CompletionTracker(site.robot_goal)
    trace, commands = [], []
    for step in range(int(round(seconds / DT))):
        if step == start_step:
            follower.start(step)
        v, omega = follower.command(step * DT, xy, yaw)
        commands.append((v, omega, follower.phase))
        xy = xy + DT * v * np.array([math.cos(yaw), math.sin(yaw)])
        yaw += DT * omega
        tracker.update(step, xy, yaw, v)
        trace.append((xy.copy(), yaw))
        if tracker.step >= 0:
            break
    return trace, tracker, commands


@pytest.mark.parametrize("sid,label,row", CASES, ids=[f"{s}-{l}" for s, l, _ in CASES])
def test_the_follower_brings_a_unicycle_to_the_goal(sid, label, row):
    site, follower = _follower(sid, row)
    trace, tracker, commands = _drive(site, follower)
    assert tracker.step >= 0, f"{sid} {label}: not complete within {WINDOW_S} s"
    seconds = (tracker.step + 1) * DT
    print(f"{sid} {label}: complete at {seconds:.2f} s")
    path = sites.nominal_path(site)[:, :2]
    worst = max(np.min(np.hypot(*(path - p).T)) for p, _ in trace)
    assert worst <= 0.15
    # The run ends when the goal is held, up to the completion tolerance short of the path's end.
    length = float(np.sum(np.hypot(*np.diff(path, axis=0).T)))
    walked = length if sid != "S1g" else site.path["walk_m"]
    assert follower.progress >= 1.0 - (POSITION_TOLERANCE_M + 0.05) / walked


@pytest.mark.parametrize("sid,label,row", CASES, ids=[f"{s}-{l}" for s, l, _ in CASES])
def test_commands_stay_inside_the_policys_range(sid, label, row):
    site, follower = _follower(sid, row)
    _, _, commands = _drive(site, follower)
    assert all(0.0 <= v <= 1.0 and abs(w) <= 1.0 for v, w, _ in commands)


def test_nothing_moves_before_the_task_comes():
    site, follower = _follower("S1e", CASES[0][2])
    assert follower.started_step == -1 and follower.phase == "wait"
    assert follower.command(0.0, np.array([site.robot_start.x, site.robot_start.y]),
                            math.radians(site.robot_start.yaw_deg)) == (0.0, 0.0)
    follower.start(7)
    follower.start(9)                                 # a second call does not move the start
    assert follower.started_step == 7 and follower.phase == "walk"


@pytest.mark.parametrize("side", ["west", "east"])
def test_s1g_waits_turns_then_walks(side):
    row = next(r for s, l, r in CASES if s == "S1g" and r["approach_side"] == side and r["turn_rate_rad_s"] == _span("S1g", "turn_rate_rad_s")[0])
    site, follower = _follower("S1g", row)
    begin = 100
    trace, tracker, commands = _drive(site, follower, start_step=begin)
    assert all(c[:2] == (0.0, 0.0) and c[2] == "wait" for c in commands[:begin])
    turning = [c for c in commands if c[2] == "turn"]
    # The turn is on the spot, at the row's rate, away from the person: the walk goes along the person's way,
    # west side arrives walking east, so the half-way heading points east (+x) -- counter-clockwise.
    assert turning and all(v == 0.0 and abs(w) == pytest.approx(row["turn_rate_rad_s"]) for v, w, _ in turning)
    walk_x = 1.0 if side == "west" else -1.0
    sign = math.copysign(1.0, turning[0][1])
    assert math.cos(math.radians(site.robot_start.yaw_deg) + sign * math.pi / 2) * walk_x > 0
    # Walking starts once the yaw error is within 5 degrees, at the row's fixed speed.
    first_walk = next(i for i, c in enumerate(commands) if c[2] == "walk")
    target = math.radians(site.robot_start.yaw_deg + site.path["turn_deg"])
    yaw_at_walk = trace[first_walk - 1][1]
    assert abs((target - yaw_at_walk + math.pi) % (2 * math.pi) - math.pi) <= math.radians(5.0) + DT * row["turn_rate_rad_s"]
    assert commands[first_walk][0] == pytest.approx(row["walk_speed_mps"], rel=0.05)
    # The spin takes pi / rate (less the last 5 degrees).
    expected = (math.pi - math.radians(5.0)) / row["turn_rate_rad_s"]
    assert (first_walk - begin) * DT == pytest.approx(expected, abs=2 * DT)


def test_completion_needs_five_still_steps():
    goal = sites.Pose2(2.0, 3.0, 90.0)
    tracker = task_path.CompletionTracker(goal)
    at = (2.0, 3.0)
    for step in range(4):
        tracker.update(step, at, math.radians(90.0), 0.0)
    assert tracker.step == -1
    tracker.update(4, at, math.radians(90.0), 0.0)
    assert tracker.step == 4                                  # the fifth step in a row
    tracker.update(5, (9.0, 9.0), 0.0, 3.0)
    assert tracker.step == 4                                  # the first is kept

    def broken(**kw):
        t = task_path.CompletionTracker(goal)
        for step in range(10):
            args = dict(xy=at, yaw=math.radians(90.0), speed=0.0)
            if step == 3:
                args.update(kw)
            t.update(step, args["xy"], args["yaw"], args["speed"])
        return t.step
    assert broken(speed=0.16) == 8                            # one moving step restarts the count
    assert broken(xy=(2.31, 3.0)) == 8
    assert broken(yaw=math.radians(90.0 + 21.0)) == 8
    assert broken(speed=0.15) == 4                            # at the tolerance is still
    assert broken(yaw=math.radians(90.0 + 360.0 - 19.0)) == 4  # yaw is compared modulo a turn


# ---- the task's stages ----

@functools.cache
def _boxes():
    return layout.build_layout(layout.load_spec(layout.SCENE_SPEC))


def row_site_and_row(sid, index):
    """Row ``index`` of the scenario's conditions table and the site it moves (conditions.draw_row / row_site)."""
    entry = _entry(sid)
    row = conditions.draw_row(entry, index, _boxes())
    return conditions.row_site(sites.load_site(entry), row), row


def fresh_follower(sid):
    return task_path.PathFollower(*row_site_and_row(sid, 0), dt=DT)


def started_follower(sid, step):
    follower = fresh_follower(sid)
    follower.start(step)
    return follower


def walk_the_nominal_path(site, row):
    """The task comes at step 0 and the robot is put on each point of the follower's path in turn, one a step from
    step 1; the goal is never held."""
    follower = task_path.PathFollower(site, row, dt=DT)
    stages = task_path.StageTracker(site, row)
    follower.start(0)
    for step, (x, y, yaw) in enumerate(sites.nominal_path(site, task_path.PATH_STEP_M), start=1):
        follower.command(step * DT, np.array([x, y]), yaw)
        stages.update(step, follower, completed_step=-1)
    return follower, stages


@pytest.mark.parametrize("sid, kind", [("S1e", "u_turn"), ("S1f", "straight"), ("S1g", "turn_then_walk")])
def test_the_stages_follow_the_path_kind(sid, kind):
    site, row = row_site_and_row(sid, 0)
    assert site.path["kind"] == kind
    plan = task_path.stage_plan(site, row)
    assert tuple(s["name"] for s in plan) == task_path.STAGE_NAMES[kind]
    assert plan[-1]["goal"] == {"kind": "task_goal"} and json.loads(json.dumps(plan)) == list(plan)
    assert task_path.StageTracker(site, row).names == task_path.STAGE_NAMES[kind]
    assert task_path.StageTracker(site, row).end_steps == [-1] * len(plan)


def test_the_u_turn_marks_are_the_way_in_and_the_half_circle():
    site, row = row_site_and_row("S1e", 0)
    cx = site.path["turn_center_xy_m"][0]
    r = site.path["radius_m"]
    way_in = abs(cx - site.robot_start.x)
    plan = task_path.stage_plan(site, row)
    assert plan[0]["goal"] == {"kind": "path_length", "path_m": pytest.approx(way_in)}
    assert plan[1]["goal"] == {"kind": "path_length", "path_m": pytest.approx(way_in + math.pi * r)}
    assert all(s["target_speed_mps"] == row["robot_speed_mps"] for s in plan)
    assert all(s["target_yaw_rate_rad_s"] is None for s in plan)


def test_the_straight_path_is_one_walk_at_the_rows_speed():
    site, row = row_site_and_row("S1f", 0)
    assert task_path.stage_plan(site, row) == (
        {"name": "walk", "goal": {"kind": "task_goal"}, "target_speed_mps": row["robot_speed_mps"],
         "target_yaw_rate_rad_s": None},)


@pytest.mark.parametrize("side", ["west", "east"])
def test_the_turn_on_the_spot_is_aimed_at_the_turned_heading_at_the_rows_rate(side):
    site, row = next((s, r) for s, r in (row_site_and_row("S1g", i) for i in range(20)) if r["approach_side"] == side)
    turn, walk = task_path.stage_plan(site, row)
    yaw = turn["goal"]["yaw_deg"]
    # the start's yaw turned by the row's turn, folded into a half turn either way
    assert -180.0 <= yaw <= 180.0
    assert math.cos(math.radians(yaw - site.robot_start.yaw_deg - site.path["turn_deg"])) == pytest.approx(1.0)
    assert turn == {"name": "turn", "goal": {"kind": "heading", "yaw_deg": yaw}, "target_speed_mps": None,
                    "target_yaw_rate_rad_s": row["turn_rate_rad_s"]}
    assert walk == {"name": "walk", "goal": {"kind": "task_goal"}, "target_speed_mps": row["walk_speed_mps"],
                    "target_yaw_rate_rad_s": None}


def test_a_path_kind_without_stages_is_refused():
    site, row = row_site_and_row("S1f", 0)
    with pytest.raises(ValueError, match="circle"):
        task_path.stage_plan(sites.Site(site.scenario_id, site.robot_start, site.robot_goal, {"kind": "circle"},
                                        site.human_x_m, site.human_y_m), row)


def test_the_follower_counts_the_path_it_has_gone_along():
    site, row = row_site_and_row("S1e", 0)
    follower = task_path.PathFollower(site, row, dt=DT)
    path = sites.nominal_path(site, task_path.PATH_STEP_M)
    start = np.array([site.robot_start.x, site.robot_start.y])
    follower.command(0.0, start, math.radians(site.robot_start.yaw_deg))
    assert follower.travelled_m == 0.0                           # waiting: nothing followed
    follower.start(0)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path[:, :2], axis=0).T))])
    for k in (0, 10, 40, 70):
        follower.command(0.0, path[k, :2], path[k, 2])
        assert follower.travelled_m == pytest.approx(arc[k])     # the arc length of the nearest path point


def test_stages_end_in_order_as_the_robot_goes_and_the_last_at_completion():
    site, row = row_site_and_row("S1e", 0)
    follower, stages = walk_the_nominal_path(site, row)        # the robot put on each path point in turn
    a, b, c = stages.end_steps
    assert 0 < a < b and c == -1                                # the way out is open until the task is done
    # the way in ends at the step the robot stands on the row end's path point (path point a - 1)
    path = sites.nominal_path(site, task_path.PATH_STEP_M)
    corner = (site.path["turn_center_xy_m"][0], site.robot_start.y)
    assert math.hypot(path[a - 1, 0] - corner[0], path[a - 1, 1] - corner[1]) < 1e-9
    stages.update(900, follower, completed_step=880)
    assert stages.end_steps == [a, b, 880]
    stages.update(950, follower, completed_step=940)            # a step once written stays
    assert stages.end_steps == [a, b, 880]


def test_completion_closes_every_stage_still_open():
    stages = task_path.StageTracker(*row_site_and_row("S1g", 0))
    follower = started_follower("S1g", step=10)
    stages.update(11, follower, completed_step=300)
    assert stages.end_steps == [300, 300]


def test_nothing_ends_before_the_task_comes():
    stages = task_path.StageTracker(*row_site_and_row("S1g", 0))
    stages.update(50, fresh_follower("S1g"), completed_step=-1)
    assert stages.end_steps == [-1, -1]
    stages.update(51, fresh_follower("S1g"), completed_step=40)   # a completion before the task closes nothing
    assert stages.end_steps == [-1, -1]
    stages.update(-1, started_follower("S1g", step=0), completed_step=40)   # nor before the first recorded step
    assert stages.end_steps == [-1, -1]


def test_a_completion_before_the_task_came_closes_nothing():
    from generation.envs.hri import record_logic

    stages = task_path.StageTracker(*row_site_and_row("S1g", 0))
    stages.update(11, started_follower("S1g", step=10), completed_step=5)   # the goal held before the task came
    assert stages.end_steps == [-1, -1]
    stages.update(12, fresh_follower("S1g"), completed_step=5)              # the task never came
    assert stages.end_steps == [-1, -1]
    # what the tracker wrote, the sidecar takes
    assert not any(s["completed"] for s in record_logic.task_stages_record(stages.names, stages.end_steps, 10, 50))
    stages.update(13, started_follower("S1g", step=10), completed_step=13)  # a completion after it closes the rest
    assert stages.end_steps == [13, 13]


def test_a_completion_before_a_stage_began_leaves_it_open_and_the_marks_still_count():
    from generation.envs.hri import record_logic

    site, row = row_site_and_row("S1e", 0)
    way_in = abs(site.path["turn_center_xy_m"][0] - site.robot_start.x)
    stages = task_path.StageTracker(site, row)
    stages.update(30, NS(started_step=10, phase="walk", travelled_m=way_in), completed_step=5)
    assert stages.end_steps == [30, -1, -1]                     # the way in ends at its mark; the completion is stale
    stages.update(40, NS(started_step=10, phase="walk", travelled_m=way_in + 0.1), completed_step=20)
    assert stages.end_steps == [30, -1, -1]                     # nor one before the stage before it ended
    record = record_logic.task_stages_record(stages.names, stages.end_steps, 10, 50)
    assert [s["completed"] for s in record] == [True, False, False]
    stages.update(50, NS(started_step=10, phase="walk", travelled_m=way_in + 0.2), completed_step=50)
    assert stages.end_steps == [30, 50, 50]
    record_logic.task_stages_record(stages.names, stages.end_steps, 10, 50)


def test_the_way_in_and_the_half_circle_end_on_the_sample_where_their_segment_ends():
    # every row of S1e's table, the random rows and the grid's, so the smallest and the largest turn radius are in
    entry = _entry("S1e")
    for index in range(conditions.load_conditions(entry).total):
        site, row = row_site_and_row("S1e", index)
        path = sites.nominal_path(site, task_path.PATH_STEP_M)
        cx, cy = site.path["turn_center_xy_m"]
        heading = math.copysign(1.0, math.cos(math.radians(site.robot_start.yaw_deg)))

        def samples_at(x, y):
            return [k for k, p in enumerate(path) if math.hypot(p[0] - x, p[1] - y) < 1e-9]

        # the last sample of the straight in is the row end on the start's lane; the last sample of the half circle
        # is the point across the turn's centre from it, on the way out's lane
        (corner,) = samples_at(cx, site.robot_start.y)
        (arc_end,) = samples_at(cx, cy + heading * site.path["radius_m"])
        assert 0 < corner < arc_end < len(path) - 1
        _, stages = walk_the_nominal_path(site, row)            # sample k is stood on at step k + 1
        assert stages.end_steps[:2] == [corner + 1, arc_end + 1], f"row {index}"


def test_a_later_mark_already_passed_waits_for_the_stage_before(monkeypatch):
    site, row = row_site_and_row("S1e", 0)
    marks = ({"name": "far", "goal": {"kind": "path_length", "path_m": 5.0}},
             {"name": "near", "goal": {"kind": "path_length", "path_m": 1.0}},
             {"name": "last", "goal": {"kind": "task_goal"}})
    monkeypatch.setattr(task_path, "stage_plan", lambda s, r: marks)
    stages = task_path.StageTracker(site, row)
    stages.update(3, NS(started_step=0, phase="walk", travelled_m=2.0), completed_step=-1)
    assert stages.end_steps == [-1, -1, -1]                     # "near" is passed, "far" is not
    stages.update(4, NS(started_step=0, phase="walk", travelled_m=6.0), completed_step=-1)
    assert stages.end_steps == [4, 4, -1]                       # two end at one step


def test_two_marks_passed_at_one_step_end_at_that_step():
    site, row = row_site_and_row("S1e", 0)
    way_in = abs(site.path["turn_center_xy_m"][0] - site.robot_start.x)
    stages = task_path.StageTracker(site, row)
    past = way_in + math.pi * site.path["radius_m"] + 0.01
    stages.update(7, NS(started_step=0, phase="walk", travelled_m=past), completed_step=-1)
    assert stages.end_steps == [7, 7, -1]


def test_the_turn_on_the_spot_ends_at_the_step_the_follower_walks():
    site, row = row_site_and_row("S1g", 0)
    follower = task_path.PathFollower(site, row, dt=DT)
    stages = task_path.StageTracker(site, row)
    tracker = task_path.CompletionTracker(site.robot_goal)
    xy, yaw = np.array([site.robot_start.x, site.robot_start.y]), math.radians(site.robot_start.yaw_deg)
    begin, first_walk = 20, None
    for step in range(int(round(WINDOW_S / DT))):
        if step == begin:
            follower.start(step)
        v, omega = follower.command(step * DT, xy, yaw)
        if first_walk is None and follower.phase == "walk":
            first_walk = step
        xy = xy + DT * v * np.array([math.cos(yaw), math.sin(yaw)])
        yaw += DT * omega
        tracker.update(step, xy, yaw, v)                        # the command term's order: completion, then stages
        stages.update(step, follower, tracker.step)
        if first_walk is None:
            assert stages.end_steps == [-1, -1]                 # waiting and turning
        if tracker.step >= 0:
            break
    assert begin < first_walk < tracker.step
    assert stages.end_steps == [first_walk, tracker.step]
