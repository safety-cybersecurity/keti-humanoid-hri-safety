"""Directed protective separation distance S_p for the scenarios that declare R-TS15066-SSM-02.

ISO/TS 15066 5.5.4.2.3 eq. (1), read with what the simulator knows: the speeds are those of
the two closest points along the line between them (clipped at 0), the robot's stopping time
and distance come from a measured stop table read at the robot's actual speed as its upper
envelope (never smaller for a faster robot or a heavier box), and only the share of the
stopping distance that moves toward the person counts. The rule's
parameters and the table are meta/rules.json's (R-TS15066-SSM-02, schema §9.1); this module
reads them, checks them and computes. Pure numpy: the recorder computes S_p with it during
generation, and a reader of a release can recompute it from the published columns.

The validator keeps its own copy of stop_table_problems (it imports nothing from generation/);
tests/generation/test_directed_separation.py holds the two to one answer.
"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Mapping, Sequence

import numpy as np

DIRECTED_SSM_RULE_ID = "R-TS15066-SSM-02"
CONTACT_RULE_ID = "R-HRI-CONTACT-01"

# The rule's scalar parameters. The validator's REQUIRED_RULE_PARAMETERS names the same.
DIRECTED_SSM_PARAMETERS = (
    "robot_reaction_time",
    "intrusion_distance",
    "position_uncertainty_robot",
    "position_uncertainty_human",
    "approach_speed_threshold",
)

STOP_TABLE_MOTIONS = ("walking", "turning")
# Lightest first: carry_state takes the first state at least as heavy as the box, and stopping the
# largest of that state and the ones before it.
STOP_TABLE_STATES = ("empty", "carry_2kg", "carry_3kg")
# A row is [speed, stopping time, stopping distance]; turning's distance is the box corner's travel.
STOP_TABLE_ROW_UNITS = {"walking": ["m/s", "s", "m"], "turning": ["rad/s", "s", "m"]}

# Two closest points nearer than this have no direction between them.
COINCIDENT_M = 1e-9


def _number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def stop_table_problems(table: object) -> list[str]:
    """Everything wrong with a stop table, as "stop_table.<where>: <why>" lines; [] when usable.

    The threshold contract takes one number per parameter, so the table sits beside the
    parameters and is checked here instead."""
    if not isinstance(table, dict):
        return [f"stop_table: 객체가 아님 ({type(table).__name__})"]
    out: list[str] = []
    if table.get("determined_by") != "project":
        out.append(f"stop_table.determined_by: project 가 아님 ({table.get('determined_by')!r})")
    measured = table.get("measured_on")
    if not isinstance(measured, str) or not measured.strip():
        out.append("stop_table.measured_on: 잰 날짜가 없음")
    if table.get("row_units") != STOP_TABLE_ROW_UNITS:
        out.append(f"stop_table.row_units: {STOP_TABLE_ROW_UNITS} 가 아님")
    masses = table.get("carry_mass_kg")
    if not isinstance(masses, dict) or set(masses) != set(STOP_TABLE_STATES):
        out.append(f"stop_table.carry_mass_kg: 상태가 {list(STOP_TABLE_STATES)} 가 아님")
    else:
        values = [masses[s] for s in STOP_TABLE_STATES]
        if not all(_number(v) and v >= 0 for v in values):
            out.append("stop_table.carry_mass_kg: 유한한 음이 아닌 수가 아닌 값이 있음")
        elif any(b <= a for a, b in zip(values, values[1:])):
            out.append(f"stop_table.carry_mass_kg: {list(STOP_TABLE_STATES)} 순서로 커지지 않음")
    reach = table.get("turning_reach_m")
    if not _number(reach) or reach <= 0:
        out.append(f"stop_table.turning_reach_m: 유한한 양수가 아님 ({reach!r})")
    for motion in STOP_TABLE_MOTIONS:
        block = table.get(motion)
        if not isinstance(block, dict) or set(block) != set(STOP_TABLE_STATES):
            out.append(f"stop_table.{motion}: 상태가 {list(STOP_TABLE_STATES)} 가 아님")
            continue
        for state in STOP_TABLE_STATES:
            where = f"stop_table.{motion}.{state}"
            rows = block[state]
            if not isinstance(rows, list) or not rows:
                out.append(f"{where}: 행이 없음")
                continue
            if not all(isinstance(r, list) and len(r) == 3 and all(_number(x) and x >= 0 for x in r)
                       for r in rows):
                out.append(f"{where}: 행마다 유한한 음이 아닌 수 셋 [속도, 정지 시간, 정지 거리] 가 아님")
                continue
            speeds = [r[0] for r in rows]
            if speeds[0] <= 0 or any(b <= a for a, b in zip(speeds, speeds[1:])):
                out.append(f"{where}: 속도가 0 보다 크게 시작해 행마다 커지지 않음")
    return out


def scenario_declares_directed_ssm(scenario: Mapping) -> bool:
    """Whether a meta/scenarios.json entry selects R-TS15066-SSM-02 -- the switch for the hri
    step fields (schema §6) and for the judgement that reads them."""
    return any(isinstance(sel, str) and fnmatch.fnmatchcase(DIRECTED_SSM_RULE_ID, sel)
               for sel in (scenario.get("rules") or []))


def load_directed_ssm(rules: Mapping) -> tuple[dict[str, float], dict]:
    """(the five parameters by name, the stop table) of the rule in a meta/rules.json document."""
    found = [r for r in (rules.get("rules") or [])
             if isinstance(r, dict) and r.get("rule_id") == DIRECTED_SSM_RULE_ID]
    if len(found) != 1:
        raise ValueError(f"meta/rules.json 에 {DIRECTED_SSM_RULE_ID} 가 {len(found)}개 있습니다 (1개여야 함)")
    rule = found[0]
    params = rule.get("parameters") if isinstance(rule.get("parameters"), dict) else {}
    values: dict[str, float] = {}
    for name in DIRECTED_SSM_PARAMETERS:
        entry = params.get(name)
        value = entry.get("value") if isinstance(entry, dict) else None
        if not _number(value) or value < 0:
            raise ValueError(f"{DIRECTED_SSM_RULE_ID}.parameters.{name} 이 유한한 음이 아닌 수가 "
                             f"아닙니다: {value!r}")
        values[name] = float(value)
    problems = stop_table_problems(rule.get("stop_table"))
    if problems:
        raise ValueError(f"{DIRECTED_SSM_RULE_ID}.stop_table: " + "; ".join(problems))
    return values, rule["stop_table"]


def carry_state(table: Mapping, mass_kg: float) -> str:
    """The state a box of ``mass_kg`` reads: the lightest one at least as heavy (stopping then takes
    the largest of it and every lighter state)."""
    if not _number(mass_kg) or mass_kg < 0:
        raise ValueError(f"상자 무게가 유한한 음이 아닌 수가 아닙니다: {mass_kg!r}")
    masses = table["carry_mass_kg"]
    for state in STOP_TABLE_STATES:
        if mass_kg <= masses[state]:
            return state
    raise ValueError(f"상자 {mass_kg} kg 이 정지 시험 표의 가장 무거운 행"
                     f"({masses[STOP_TABLE_STATES[-1]]} kg)보다 무겁습니다 — 그 무게로 정지 시험을 해 행을 더하십시오")


def interpolate(rows: Sequence[Sequence[float]], speed: float) -> tuple[float, float]:
    """(stopping time, stopping distance) at ``speed``: linear between rows, from (0, (0, 0)),
    and the last row above the last speed."""
    speed = max(float(speed), 0.0)
    points = [(0.0, 0.0, 0.0)] + [(float(r[0]), float(r[1]), float(r[2])) for r in rows]
    if speed >= points[-1][0]:
        return points[-1][1], points[-1][2]
    for (a, ta, sa), (b, tb, sb) in zip(points, points[1:]):
        if speed <= b:
            f = (speed - a) / (b - a)
            return ta + f * (tb - ta), sa + f * (sb - sa)
    return points[-1][1], points[-1][2]


def _rising(rows: Sequence[Sequence[float]]) -> list[tuple[float, float, float]]:
    """``rows`` with each stopping time and distance raised to the largest at its speed or below."""
    out, t_top, s_top = [], 0.0, 0.0
    for speed, t, s in rows:
        t_top, s_top = max(t_top, float(t)), max(s_top, float(s))
        out.append((float(speed), t_top, s_top))
    return out


def stopping(table: Mapping, motion: str, state: str, speed: float) -> tuple[float, float]:
    """(stopping time, stopping distance) S_p reads for ``motion`` at ``speed`` with the box's
    ``state``: the measured table's upper envelope, so neither gets smaller when the robot moves
    faster or carries more. Each state's rows are raised to the largest at a lower speed and then
    interpolated, and the state takes the largest of itself and every lighter state -- the time and
    the distance each on its own. A row is the worst of five gait phases timed in 0.02 s steps, and
    between rows and states that worst case does not always rise (walking with 2 kg stops in 1.14 s
    from 1.0 m/s and in 1.10 s from 1.2 m/s)."""
    if state not in STOP_TABLE_STATES:
        raise ValueError(f"정지 시험 표에 없는 상태입니다: {state!r} — {list(STOP_TABLE_STATES)}")
    lighter = STOP_TABLE_STATES[:STOP_TABLE_STATES.index(state) + 1]
    read = [interpolate(_rising(table[motion][s]), speed) for s in lighter]
    return max(t for t, _ in read), max(s for _, s in read)


def approach_speeds(robot_point, human_point, robot_velocity, human_velocity) -> tuple[float, float]:
    """(v_r, v_h): how fast the robot's closest point moves toward the person's and the person's
    toward the robot's, along the line between them; below 0 is 0, coincident points give (0, 0)."""
    r, h, vr, vh = (np.asarray(x, dtype=float).reshape(-1)
                    for x in (robot_point, human_point, robot_velocity, human_velocity))
    if not (r.shape == h.shape == vr.shape == vh.shape) or r.size not in (2, 3):
        raise ValueError(f"점과 속도의 차원이 맞지 않습니다: {r.shape} {h.shape} {vr.shape} {vh.shape}")
    if not all(np.isfinite(x).all() for x in (r, h, vr, vh)):
        raise ValueError("점이나 속도에 유한하지 않은 값이 있습니다")
    gap = h - r
    dist = float(np.linalg.norm(gap))
    if dist < COINCIDENT_M:
        return 0.0, 0.0
    u = gap / dist
    return max(0.0, float(vr @ u)), max(0.0, float(-(vh @ u)))


def _finite(name: str, value: object, nonnegative: bool) -> float:
    if not _number(value) or (nonnegative and value < 0):
        raise ValueError(f"{name} 이 유한한{' 음이 아닌' if nonnegative else ''} 수가 아닙니다: {value!r}")
    return float(value)


def _parameter(params: Mapping[str, float], name: str) -> float:
    if name not in params:
        raise ValueError(f"S_p 의 파라미터 {name} 이 없습니다 — {list(DIRECTED_SSM_PARAMETERS)}")
    return float(params[name])


def separation_distance(params: Mapping[str, float], table: Mapping, *, state: str,
                        robot_speed: float, yaw_rate: float, v_r: float, v_h: float) -> float:
    """S_p = v_h (T_r + T_s) + v_r T_r + S_s' + C + Z_r + Z_d.

    ``robot_speed`` is the robot's actual horizontal speed and ``yaw_rate`` its actual yaw rate;
    T_s and S_s are the stop table's at those for the box's ``state``, read by ``stopping`` (the
    table's upper envelope). S_s' is the share of S_s
    that moves toward the person -- S_s * min(1, v_r / robot_speed) walking, the corner's travel *
    min(1, v_r / (|yaw_rate| * turning_reach_m)) turning. Walking and turning at once take the
    larger of the two."""
    speed = _finite("robot_speed", robot_speed, True)
    rate = abs(_finite("yaw_rate", yaw_rate, False))
    v_r = _finite("v_r", v_r, True)
    v_h = _finite("v_h", v_h, True)
    if state not in STOP_TABLE_STATES:
        raise ValueError(f"정지 시험 표에 없는 상태입니다: {state!r} — {list(STOP_TABLE_STATES)}")
    t_r = _parameter(params, "robot_reaction_time")
    fixed = (v_r * t_r + _parameter(params, "intrusion_distance")
             + _parameter(params, "position_uncertainty_robot")
             + _parameter(params, "position_uncertainty_human"))
    t_walk, s_walk = stopping(table, "walking", state, speed)
    walk_share = 1.0 if speed <= 0.0 else min(1.0, v_r / speed)
    t_turn, corner = stopping(table, "turning", state, rate)
    corner_speed = rate * float(table["turning_reach_m"])
    turn_share = 1.0 if corner_speed <= 0.0 else min(1.0, v_r / corner_speed)
    walk = v_h * (t_r + t_walk) + s_walk * walk_share
    turn = v_h * (t_r + t_turn) + corner * turn_share
    return fixed + max(walk, turn)


def separation_violated(distance: float, separation: float, v_r: float, threshold: float) -> bool:
    """A step counts toward a near miss while the robot moves toward the person faster than the
    threshold and the distance is short of S_p (schema §6 safety.separation_violation)."""
    return bool(v_r > threshold and distance < separation)


def robot_was_standing(v_r: float, threshold: float) -> bool:
    """A contact whose first step has the robot at or below the threshold is one with a robot not
    approaching the person (standing, walking past or turning) -- the exact complement of
    separation_violated's approach condition. The name is older than that reading."""
    return bool(v_r <= threshold)
