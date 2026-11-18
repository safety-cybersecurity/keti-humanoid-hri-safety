"""Human scripts (generation/human_script.py): one plan per condition row -- where the person stands or starts, which
way it faces and walks, how fast, what sets it off and where it stops. Isaac-free; the expectations are the canon,
the row and closed-form kinematics."""

import copy
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from generation import conditions, human_script
from generation.scene import layout

REPO = Path(__file__).resolve().parents[2]
NEW = ("S1e", "S1f", "S1g")
DT = 0.02


def _entry(sid):
    canon = json.loads((REPO / "meta" / "scenarios.json").read_text(encoding="utf-8"))
    return copy.deepcopy(next(s for s in canon["scenarios"] if s["scenario_id"] == sid))


@pytest.fixture(scope="module")
def boxes():
    return layout.build_layout(layout.load_spec(layout.SCENE_SPEC))


WIDE_ROWS = 400                  # a wider table than the canon's, drawn from the same windows and seeds


def _wide(sid):
    entry = _entry(sid)
    entry["conditions"]["rows"] = entry["episodes"] = WIDE_ROWS
    entry["conditions"].pop("grid", None)
    return entry


@pytest.fixture(scope="module")
def tables(boxes):
    return {sid: conditions.table(_wide(sid), boxes) for sid in NEW}


def _ramp():
    return yaml.safe_load(conditions.GAIT_SPEC.read_text(encoding="utf-8"))["ramp"]


def _plan(sid, row):
    return human_script.plan_for(_entry(sid), row)


def _unit(plan):
    return np.asarray(plan.walk_dir, float)


def _row(tables, sid, **want):
    for row in tables[sid]:
        if all(row[k] == v for k, v in want.items()):
            return row
    raise AssertionError(f"no {sid} row with {want}")


def test_s1e_stands_where_the_row_says(tables):
    for row in tables["S1e"][:20]:
        plan = _plan("S1e", row)
        cx, face_edge = -6.045, 5.0                    # the canon's turn centre x and the rack face the person stands off
        assert plan.start_xy == pytest.approx((cx + row["corner_distance_m"], face_edge - row["face_distance_m"]))
        assert plan.yaw_deg == row["facing_yaw_deg"]
        assert plan.walk_dir is None and plan.stop_after_m is None and plan.speed_mps == 0.0
        assert plan.trigger == human_script.Trigger("never", 0.0)
        assert human_script.walked(plan, 5.0) == 0.0


def test_s1f_faces_the_rack_then_turns_into_the_robots_way(tables):
    row = dict(tables["S1f"][0])
    row["entry_angle_deg"] = 30.0
    plan = _plan("S1f", row)
    assert plan.yaw_deg == pytest.approx(90.0)                          # the rack, its back to the robot's lane
    assert _unit(plan) == pytest.approx((math.cos(math.radians(30.0)), -math.sin(math.radians(30.0))))
    assert human_script.walk_yaw_deg(plan) == pytest.approx(-30.0)
    assert plan.start_xy == pytest.approx((row["human_start_x_m"], -6.35))
    assert plan.turn_time_s == row["turn_time_s"] and plan.speed_mps == row["human_speed_mps"]
    assert plan.trigger == human_script.Trigger("robot_behind", pytest.approx(row["trigger_lead_s"] * row["robot_speed_mps"]))
    row["entry_angle_deg"] = 120.0                      # an obtuse angle leans back: the x part turns negative
    plan = _plan("S1f", row)
    assert _unit(plan)[0] < 0.0 and plan.yaw_deg == pytest.approx(90.0)
    assert human_script.walk_yaw_deg(plan) == pytest.approx(-120.0)


def test_the_walk_waits_for_the_turn_and_sets_off_at_the_slowest_gait():
    assert human_script.turn_gait_speed_mps() == 0.3
    plan = human_script.HumanPlan((0.0, 0.0), 90.0, (1.0, 0.0), 1.2, human_script.Trigger("time", 0.0), 10.0, 1.0)
    assert human_script.walked(plan, 1.0) == 0.0
    assert human_script.walked(plan, 1.0 + DT) == pytest.approx((0.3 + 1.2 * DT) * DT)     # one ramp step up from 0.3
    still = dataclasses.replace(plan, turn_time_s=0.0)
    assert human_script.walked(still, 1.0) == pytest.approx(0.6, abs=1.2 * DT)             # unchanged: a ramp from rest
    assert human_script.plan_from_record(human_script.plan_record(plan)) == plan


def test_a_plan_without_a_turn_walks_from_rest_and_a_record_without_one_loads_as_one():
    """A plan whose turn_time_s is 0 gives the very floats of a ramp from rest (no turn steps, no head start), and a
    record without turn_time_s reads as a plan that does not turn."""
    accel, decel = _ramp()["accel_time_s"], _ramp()["decel_time_s"]
    plan = human_script.HumanPlan((0.0, 0.0), 0.0, (1.0, 0.0), 1.2, human_script.Trigger("time", 0.0), 4.0)
    v = pos = 0.0
    want = []
    while True:
        v = float(np.minimum(1.2, v + 1.2 / accel * DT))
        v = float(min(v, math.sqrt(max(0.0, 2.0 * 1.2 / decel * max(0.0, 4.0 - pos)))))
        pos += v * DT
        want.append(pos)
        if v <= 0.0:
            break
    got = [human_script.walked(plan, k * DT) for k in range(1, len(want) + 1)]
    assert got == want
    record = human_script.plan_record(plan)
    assert record["turn_time_s"] == 0.0
    del record["turn_time_s"]
    assert human_script.plan_from_record(record) == plan
    with pytest.raises(ValueError):
        human_script.walk_yaw_deg(dataclasses.replace(plan, walk_dir=None))


def test_s1f_stops_before_the_far_face(tables):
    for row in tables["S1f"][:20]:
        plan = _plan("S1f", row)
        far_face_y = -8.5                               # the rack face across the lane
        end = np.asarray(plan.start_xy) + _unit(plan) * plan.stop_after_m
        assert end[1] == pytest.approx(far_face_y + 0.45)
        assert human_script.walked(plan, 60.0) == pytest.approx(plan.stop_after_m, abs=2e-3)


@pytest.mark.parametrize("side", ["west", "east"])
def test_s1g_reaches_the_trigger_after_its_run_up(side, tables):
    sign = 1.0 if side == "west" else -1.0
    for row in [r for r in tables["S1g"] if r["approach_side"] == side][:15]:
        plan = _plan("S1g", row)
        robot = np.array([2.0, 10.3])
        D = conditions.s1g_trigger_distance(row)
        assert _unit(plan) == pytest.approx((sign, 0.0))
        assert plan.yaw_deg == pytest.approx(0.0 if sign > 0 else 180.0)
        assert plan.start_xy[1] == pytest.approx(10.75 + row["lateral_offset_m"])
        assert plan.trigger == human_script.Trigger("time", 0.0)
        at_d = robot - sign * np.array([D, 0.0])
        at_d[1] = plan.start_xy[1]
        t = human_script.pass_time(plan, at_d)
        assert t == pytest.approx(row["run_up_s"], abs=0.02)
        speed = (human_script.walked(plan, t + 0.5 + DT) - human_script.walked(plan, t + 0.5)) / DT
        assert speed == pytest.approx(row["human_speed_mps"], abs=1e-6)
        # the same moment seen from the task: the person is just inside the distance
        here = np.asarray(plan.start_xy) + _unit(plan) * human_script.walked(plan, t + DT)
        assert human_script.within_along(plan, here, robot, D + 0.05)
        assert not human_script.within_along(plan, np.asarray(plan.start_xy), robot, D)


def test_s1g_crosses_the_robot_line_pass_offset_after_the_turn(tables):
    for row in tables["S1g"][:30]:
        plan = _plan("S1g", row)
        sign = _unit(plan)[0]
        robot = np.array([2.0, 10.3])
        D = conditions.s1g_trigger_distance(row)
        came = human_script.pass_time(plan, np.array([robot[0] - sign * D, plan.start_xy[1]]))
        crossed = human_script.pass_time(plan, np.array([robot[0], plan.start_xy[1]]))
        turn_s = math.pi / row["turn_rate_rad_s"]
        assert crossed - came == pytest.approx(turn_s + row["pass_offset_s"], abs=0.02)
        assert plan.stop_after_m == pytest.approx(conditions.s1g_human_start_distance(row) + 3.0)


def test_walked_matches_the_gait_ramp():
    ramp = _ramp()
    assert ramp["accel_time_s"] == 1.0 and ramp["decel_time_s"] == 1.0
    v = 1.2
    plan = human_script.HumanPlan((0.0, 0.0), 0.0, (1.0, 0.0), v, human_script.Trigger("time", 0.0), 10.0)
    # one second of a linear rise to v walks v / 2 (the step integration adds half a step of v)
    assert human_script.walked(plan, 1.0) == pytest.approx(v / 2, abs=v * DT)
    assert human_script.walked(plan, 3.0) == pytest.approx(v * (3.0 - 0.5), abs=v * DT)
    # it brakes to a halt at the stop distance and stays there
    end = human_script.walked(plan, 30.0)
    assert end == pytest.approx(10.0, abs=2e-3)
    assert human_script.walked(plan, 31.0) == pytest.approx(end, abs=1e-6)
    # braking over the last second walks v / 2 (a linear fall) before the stop
    assert human_script.walked(plan, 30.0) - human_script.walked(plan, 8.8) > 0.0
    assert human_script.walked(plan, 0.0) == 0.0


def test_pass_time_is_infinite_for_a_point_behind_or_off_the_walk():
    plan = human_script.HumanPlan((0.0, 0.0), 0.0, (1.0, 0.0), 1.0, human_script.Trigger("time", 0.0), 5.0)
    assert human_script.pass_time(plan, (20.0, 0.0)) == math.inf
    assert human_script.pass_time(plan, (-1.0, 3.0)) == 0.0


@pytest.mark.parametrize("sid", NEW)
def test_every_canon_grid_row_gives_a_plan(sid, boxes):
    entry = _entry(sid)
    rows = conditions.grid_table(entry, boxes)
    assert len(rows) == conditions.load_conditions(entry).grid.cells ** 2
    for row in rows:
        plan = human_script.plan_for(entry, row)
        assert math.isfinite(plan.yaw_deg) and all(math.isfinite(c) for c in plan.start_xy)
        if sid == "S1e":
            assert plan.walk_dir is None
        else:
            assert math.hypot(*plan.walk_dir) == pytest.approx(1.0) and plan.stop_after_m > 0.0


def test_every_canon_row_gives_a_plan(tables):
    for sid in NEW:
        assert len(tables[sid]) == WIDE_ROWS
        for row in tables[sid]:
            plan = _plan(sid, row)
            assert math.isfinite(plan.yaw_deg) and all(math.isfinite(c) for c in plan.start_xy)
            if sid == "S1e":
                assert plan.walk_dir is None
            else:
                assert math.hypot(*plan.walk_dir) == pytest.approx(1.0) and plan.stop_after_m > 0.0
