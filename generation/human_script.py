"""The person's script, one plan per condition row: where the person stands or starts, which way it faces and walks,
how fast, what sets it off, how long it turns on the spot before it walks and where it stops. Pure numpy.

The places are conditions.human_walk's (the guard checks the same ones). The walk's speed follows the dummy's gait
drive: each step the speed rises linearly to the target over the gait's acceleration time and, with a stop in sight,
falls no faster than the deceleration time allows, and then the position advances by that speed times the step -- the
order the command term uses. A plan that turns on the spot (S1f) spends turn_steps(turn_time_s, dt) steps after the
trigger turning without walking, then sets off at the gait table's slowest walking speed (turn_gait_speed_mps) rather
than from rest; a plan that does not turn -- turn_steps 0, a turn shorter than half a step among them -- walks from rest
at the trigger. The command term counts the turn with the same integer (command_logic.turn_steps)."""

import dataclasses
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation import conditions, sites  # noqa: E402
from generation.dummy.gait import gait_table  # noqa: E402

DEFAULT_DT = 0.02          # the control step (50 Hz)


@dataclass(frozen=True)
class Trigger:
    """What sets the walk off: never (stands), robot_behind (person x - robot root x <= value), time (value seconds)."""
    kind: Literal["never", "robot_behind", "time"]
    value: float


@dataclass(frozen=True)
class HumanPlan:
    start_xy: tuple[float, float]
    yaw_deg: float               # the way the person faces while it stands (S1f: the rack; S1g: the way it walks)
    walk_dir: tuple[float, float] | None
    speed_mps: float
    trigger: Trigger
    stop_after_m: float | None
    # Seconds the person turns on the spot, from yaw_deg to the walk's direction, after the trigger and before it walks.
    turn_time_s: float = 0.0


def plan_record(plan: HumanPlan) -> dict:
    """The plan as plain data -- the form an environment config carries it in (a config copies its parameters and
    sets their fields, which a frozen plan refuses); plan_from_record makes it a plan again."""
    return dataclasses.asdict(plan)


def plan_from_record(record: Mapping) -> HumanPlan:
    """The plan plan_record made of it. A record without ``turn_time_s`` is a plan that does not turn."""
    fields = dict(record)
    trigger = fields.pop("trigger")
    walk_dir = fields.pop("walk_dir")
    return HumanPlan(start_xy=tuple(float(v) for v in fields.pop("start_xy")),
                     walk_dir=None if walk_dir is None else tuple(float(v) for v in walk_dir),
                     trigger=Trigger(str(trigger["kind"]), float(trigger["value"])),
                     turn_time_s=float(fields.pop("turn_time_s", 0.0)), **fields)


def walk_yaw_deg(plan: HumanPlan) -> float:
    """The direction of the plan's walk (degrees, counter-clockwise from +x); ValueError for a plan that does not
    walk."""
    if plan.walk_dir is None:
        raise ValueError("이 대본은 걷지 않습니다")
    return math.degrees(math.atan2(plan.walk_dir[1], plan.walk_dir[0]))


def turn_steps(turn_time_s: float, dt: float) -> int:
    """How many control steps of ``dt`` a turn on the spot of ``turn_time_s`` seconds lasts: round(turn_time_s / dt). 0
    is no turn, a turn shorter than half a step included -- the plan's walk (_steps) and the command term
    (command_logic.turn_steps) both count the turn with this one integer, so neither has a turn the other lacks."""
    if not (math.isfinite(turn_time_s) and turn_time_s >= 0.0):
        raise ValueError(f"돌기 시간이 0 이상의 유한한 수가 아닙니다: {turn_time_s!r}")
    return int(round(float(turn_time_s) / float(dt)))


def turn_gait_speed_mps() -> float:
    """The speed the walk after a turn on the spot sets off at: the gait table's slowest walking speed
    (speeds.grid_mps[0] of the gait spec)."""
    return float(yaml.safe_load(conditions.GAIT_SPEC.read_text(encoding="utf-8"))["speeds"]["grid_mps"][0])


def plan_for(entry: Mapping, row: Mapping) -> HumanPlan:
    """The plan of one condition row of a scenario entry (meta/scenarios.json)."""
    site = sites.load_site(entry)
    kind = site.path["kind"]
    start, stop = conditions.human_walk(site, row)
    if kind == "u_turn":
        return HumanPlan(start, float(row["facing_yaw_deg"]), None, 0.0, Trigger("never", 0.0), None)
    length = math.hypot(stop[0] - start[0], stop[1] - start[1])
    direction = ((stop[0] - start[0]) / length, (stop[1] - start[1]) / length)
    yaw = math.degrees(math.atan2(direction[1], direction[0]))
    if kind == "straight":
        return HumanPlan(start, conditions.s1f_rack_facing_yaw_deg(site), direction, float(row["human_speed_mps"]),
                         Trigger("robot_behind", conditions.s1f_trigger_distance(row)), length,
                         float(row["turn_time_s"]))
    if kind == "turn_then_walk":
        return HumanPlan(start, yaw, direction, float(row["human_speed_mps"]), Trigger("time", 0.0), length)
    raise ValueError(f"{site.scenario_id}: 대본이 모르는 길의 종류입니다: {kind!r}")


def _ramp_times() -> tuple[float, float]:
    ramp = yaml.safe_load(conditions.GAIT_SPEC.read_text(encoding="utf-8"))["ramp"]
    return float(ramp["accel_time_s"]), float(ramp["decel_time_s"])


def _steps(plan: HumanPlan, dt: float):
    """(time, distance walked) after each step from the trigger, until the walk has ended: with a turn on the spot
    first turn_steps(turn_time_s, dt) steps that walk nothing (the walk cannot end in them), then the walk from
    turn_gait_speed_mps(); without one the walk from rest."""
    turn = turn_steps(plan.turn_time_s, dt)
    accel, decel = _ramp_times()
    v = pos = t = 0.0
    if turn > 0:
        for _ in range(turn):
            t += dt
            yield t, pos
        v = turn_gait_speed_mps()
    while True:
        remaining = None if plan.stop_after_m is None else max(0.0, plan.stop_after_m - pos)
        v = float(gait_table.ramp_speed(v, plan.speed_mps, dt, accel, decel, remaining))
        pos += v * dt
        t += dt
        yield t, pos
        if v <= 0.0:
            return


def walked(plan: HumanPlan, t_since_trigger: float, dt: float = DEFAULT_DT) -> float:
    """How far the person has walked t seconds after the trigger, integrated step by step like the gait drive."""
    if plan.walk_dir is None or plan.speed_mps <= 0.0 or t_since_trigger <= 0.0:
        return 0.0
    pos = 0.0
    for k, (_, pos) in enumerate(_steps(plan, dt), start=1):
        if k >= round(t_since_trigger / dt):
            return pos
    return pos


def pass_time(plan: HumanPlan, point_xy: Sequence[float], dt: float = DEFAULT_DT) -> float:
    """When, with the trigger at 0, the person reaches the foot of the perpendicular from the point onto the walk
    -- after its turn on the spot, if it turns (0 if the point is behind the start, infinity if the walk ends short
    of it)."""
    if plan.walk_dir is None:
        raise ValueError("이 대본은 걷지 않습니다")
    along = float(np.dot(np.asarray(point_xy, float) - np.asarray(plan.start_xy, float), plan.walk_dir))
    if along <= 0.0:
        return 0.0
    for t, pos in _steps(plan, dt):
        if pos >= along:
            return t
    return math.inf


def within_along(plan: HumanPlan, human_xy: Sequence[float], robot_xy: Sequence[float], distance: float) -> bool:
    """S1g's moment the robot's task comes: the robot is no more than ``distance`` ahead of the person measured along
    the walk (a person that has passed the robot stays inside)."""
    if plan.walk_dir is None:
        raise ValueError("이 대본은 걷지 않습니다")
    return float(np.dot(np.asarray(robot_xy, float) - np.asarray(human_xy, float), plan.walk_dir)) <= float(distance)
