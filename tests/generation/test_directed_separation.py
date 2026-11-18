"""Directed separation S_p (generation/directed_separation.py): the stop table, the approach
speeds and the distance, on numbers worked out by hand. Isaac-free."""

import copy
import json
import math
from pathlib import Path

import pytest

from generation.directed_separation import (
    CONTACT_RULE_ID,
    DIRECTED_SSM_PARAMETERS,
    DIRECTED_SSM_RULE_ID,
    STOP_TABLE_MOTIONS,
    STOP_TABLE_ROW_UNITS,
    STOP_TABLE_STATES,
    approach_speeds,
    carry_state,
    interpolate,
    load_directed_ssm,
    robot_was_standing,
    scenario_declares_directed_ssm,
    separation_distance,
    separation_violated,
    stop_table_problems,
    stopping,
)

# The H1 stop test of 2026-10-17, the worst of five gait phases: [speed, stopping time s, distance m].
TABLE = {
    "determined_by": "project",
    "measured_on": "2026-10-17",
    "row_units": {"walking": ["m/s", "s", "m"], "turning": ["rad/s", "s", "m"]},
    "carry_mass_kg": {"empty": 0, "carry_2kg": 2, "carry_3kg": 3},
    "turning_reach_m": 0.465,
    "walking": {
        "empty": [[0.3, 0.54, 0.118], [0.5, 0.62, 0.187], [0.8, 0.72, 0.316],
                  [1.0, 0.76, 0.381], [1.2, 0.82, 0.476]],
        "carry_2kg": [[0.3, 0.74, 0.181], [0.5, 0.96, 0.316], [0.8, 1.06, 0.449],
                      [1.0, 1.22, 0.547], [1.2, 1.06, 0.603]],
        "carry_3kg": [[0.3, 1.04, 0.266], [0.5, 1.14, 0.372], [0.8, 1.28, 0.519],
                      [1.0, 1.22, 0.580], [1.2, 1.28, 0.683]],
    },
    "turning": {
        "empty": [[0.5, 0.18, 0.020], [1.0, 0.32, 0.039]],
        "carry_2kg": [[0.5, 0.36, 0.032], [1.0, 0.44, 0.053]],
        "carry_3kg": [[0.5, 0.32, 0.037], [1.0, 0.50, 0.062]],
    },
}
PARAMS = {"robot_reaction_time": 0.1, "intrusion_distance": 0.2, "position_uncertainty_robot": 0.0,
          "position_uncertainty_human": 0.0, "approach_speed_threshold": 0.05}


def _rules(table=None, **params):
    values = dict(PARAMS, **params)
    return {"rules": [{
        "rule_id": DIRECTED_SSM_RULE_ID, "outcome": "near_miss",
        "parameters": {k: {"value": v, "unit": "s", "determined_by": "project"} for k, v in values.items()},
        "stop_table": copy.deepcopy(TABLE if table is None else table)}]}


def _edited(change):
    table = copy.deepcopy(TABLE)
    change(table)
    return table


# ── interpolation ─────────────────────────────────────────────────────────────

def test_below_the_first_speed_the_table_runs_from_zero():
    assert interpolate(TABLE["walking"]["empty"], 0.15) == pytest.approx((0.27, 0.059))


def test_on_a_row_the_row_is_returned():
    assert interpolate(TABLE["walking"]["empty"], 0.5) == pytest.approx((0.62, 0.187))


def test_between_rows_it_is_linear():
    assert interpolate(TABLE["walking"]["empty"], 0.9) == pytest.approx((0.74, 0.3485))


def test_above_the_last_speed_the_last_row_holds():
    assert interpolate(TABLE["walking"]["empty"], 1.5) == pytest.approx((0.82, 0.476))


def test_a_standing_robot_needs_no_stopping():
    assert interpolate(TABLE["walking"]["carry_3kg"], 0.0) == (0.0, 0.0)


# ── which row ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mass, state", [(0, "empty"), (1.0, "carry_2kg"), (2, "carry_2kg"),
                                         (2.5, "carry_3kg"), (3, "carry_3kg")])
def test_a_box_takes_the_lightest_row_at_least_as_heavy(mass, state):
    assert carry_state(TABLE, mass) == state


@pytest.mark.parametrize("mass", [3.2, -0.1, float("nan"), True])
def test_a_box_the_table_does_not_cover_is_refused(mass):
    with pytest.raises(ValueError):
        carry_state(TABLE, mass)


# ── approach speeds ──────────────────────────────────────────────────────────

def test_two_points_closing_head_on():
    assert approach_speeds([0, 0, 1], [2, 0, 1], [1.0, 0, 0], [-1.5, 0, 0]) == pytest.approx((1.0, 1.5))


def test_moving_apart_is_zero_not_negative():
    assert approach_speeds([0, 0], [2, 0], [-1.0, 0], [1.0, 0]) == (0.0, 0.0)


def test_only_the_share_along_the_line_counts():
    v_r, v_h = approach_speeds([0, 0], [1, 1], [1.0, 0], [0, 0])
    assert v_r == pytest.approx(1 / math.sqrt(2)) and v_h == 0.0


def test_coincident_points_have_no_direction():
    assert approach_speeds([1, 1, 1], [1, 1, 1], [1, 0, 0], [-1, 0, 0]) == (0.0, 0.0)


@pytest.mark.parametrize("args", [
    ([0, 0], [1, 0, 0], [0, 0], [0, 0]),                 # the shapes differ
    ([0, 0], [1, 0], [float("nan"), 0], [0, 0]),         # not finite
])
def test_points_that_cannot_be_read_are_refused(args):
    with pytest.raises(ValueError):
        approach_speeds(*args)


# ── the envelope S_p reads ───────────────────────────────────────────────────

def test_a_row_below_a_slower_one_reads_the_slower_one():
    # walking carry_2kg: 1.22 s at 1.0 m/s but 1.06 s at 1.2 m/s -- a faster robot does not stop sooner
    assert stopping(TABLE, "walking", "carry_2kg", 1.2) == pytest.approx((1.22, 0.603))
    assert stopping(TABLE, "walking", "carry_2kg", 1.1) == pytest.approx((1.22, 0.575))


def test_a_heavier_state_below_a_lighter_one_reads_the_lighter():
    # turning at 0.5 rad/s: 0.36 s with 2 kg but 0.32 s with 3 kg -- a heavier box does not stop sooner
    assert stopping(TABLE, "turning", "carry_3kg", 0.5) == pytest.approx((0.36, 0.037))
    # the time and the distance are each the largest, whichever row or state it comes from
    assert stopping(TABLE, "walking", "carry_3kg", 1.0) == pytest.approx((1.28, 0.580))


def test_where_the_table_already_rises_the_envelope_is_the_table():
    for speed in (0.0, 0.15, 0.3, 0.65, 0.9, 1.2, 1.5):
        assert stopping(TABLE, "walking", "empty", speed) == pytest.approx(
            interpolate(TABLE["walking"]["empty"], speed))


def test_the_stopping_never_shrinks_for_a_faster_robot_or_a_heavier_box():
    speeds = [i / 100 for i in range(0, 151)]
    for motion in STOP_TABLE_MOTIONS:
        by_state = [[stopping(TABLE, motion, s, v) for v in speeds] for s in STOP_TABLE_STATES]
        for read in by_state:                                       # faster
            for (t0, s0), (t1, s1) in zip(read, read[1:]):
                assert t1 >= t0 and s1 >= s0, motion
        for lighter, heavier in zip(by_state, by_state[1:]):       # heavier
            for (t0, s0), (t1, s1) in zip(lighter, heavier):
                assert t1 >= t0 and s1 >= s0, motion


def test_s_p_does_not_drop_where_the_measured_rows_do():
    # the measured table alone would take S_p down from 0.99125 to 0.93225 m at 2 -> 2.000001 kg turning
    # at 0.5 rad/s, and from 2.959 to 2.779 m at 1.0 -> 1.2 m/s walking with 2 kg
    turn = dict(robot_speed=0.0, yaw_rate=0.5, v_r=0.5 * TABLE["turning_reach_m"], v_h=1.6)
    two, more = (separation_distance(PARAMS, TABLE, state=carry_state(TABLE, m), **turn) for m in (2.0, 2.000001))
    assert more >= two == pytest.approx(0.99125)
    walk = [separation_distance(PARAMS, TABLE, state="carry_2kg", robot_speed=v, yaw_rate=0.0, v_r=v, v_h=1.6)
            for v in (1.0, 1.2)]
    assert walk[1] >= walk[0] == pytest.approx(2.959)


def test_an_unknown_state_is_refused_by_name():
    with pytest.raises(ValueError, match="carry_9kg"):
        stopping(TABLE, "walking", "carry_9kg", 1.0)


# ── S_p ──────────────────────────────────────────────────────────────────────

def test_walking_with_three_kilos_head_on():
    # walking carry_3kg at 1.0 m/s: T_s 1.28 s (the 0.8 m/s row's, longer than 1.0's 1.22 s), S_s 0.580 m,
    # all of it toward the person. S_p = 1.6 (0.1 + 1.28) + 1.0 * 0.1 + 0.580 + 0.2 = 3.088
    got = separation_distance(PARAMS, TABLE, state="carry_3kg", robot_speed=1.0, yaw_rate=0.0,
                              v_r=1.0, v_h=1.6)
    assert got == pytest.approx(3.088)


def test_only_the_stopping_distance_toward_the_person_counts():
    # the same walk, the closest point closing at half the walking speed: S_s' = 0.580 / 2
    got = separation_distance(PARAMS, TABLE, state="carry_3kg", robot_speed=1.0, yaw_rate=0.0,
                              v_r=0.5, v_h=1.6)
    assert got == pytest.approx(1.6 * 1.38 + 0.5 * 0.1 + 0.29 + 0.2)


def test_turning_in_place_takes_the_box_corner():
    # carry_3kg turning at 1.0 rad/s: T_s 0.50 s, corner 0.062 m, the corner moving at
    # 1.0 * 0.465 m/s of which 0.3 toward the person; walking at 0.02 m/s is the smaller candidate.
    # S_p = 1.0 (0.1 + 0.5) + 0.3 * 0.1 + 0.062 * 0.3 / 0.465 + 0.2 = 0.87
    got = separation_distance(PARAMS, TABLE, state="carry_3kg", robot_speed=0.02, yaw_rate=-1.0,
                              v_r=0.3, v_h=1.0)
    assert got == pytest.approx(0.87)


def test_walking_while_turning_takes_the_larger_candidate():
    kw = dict(state="empty", v_r=1.0, v_h=0.5)
    walk = separation_distance(PARAMS, TABLE, robot_speed=1.0, yaw_rate=0.0, **kw)
    turn = separation_distance(PARAMS, TABLE, robot_speed=0.0, yaw_rate=1.0, **kw)
    both = separation_distance(PARAMS, TABLE, robot_speed=1.0, yaw_rate=1.0, **kw)
    assert walk == pytest.approx(1.111) and turn == pytest.approx(0.549)
    assert both == pytest.approx(max(walk, turn))


def test_no_approach_leaves_the_intrusion_distance():
    got = separation_distance(PARAMS, TABLE, state="empty", robot_speed=0.0, yaw_rate=0.0,
                              v_r=0.0, v_h=0.0)
    assert got == pytest.approx(0.2)


@pytest.mark.parametrize("bad", [dict(robot_speed=-0.1), dict(v_r=-0.1), dict(v_h=float("nan")),
                                 dict(yaw_rate=float("inf")), dict(state="carry_9kg")])
def test_inputs_out_of_their_domain_are_refused(bad):
    kw = dict(state="empty", robot_speed=0.5, yaw_rate=0.0, v_r=0.5, v_h=0.5)
    kw.update(bad)
    with pytest.raises(ValueError):
        separation_distance(PARAMS, TABLE, **kw)


# The four numbers the distance reads; approach_speed_threshold belongs to the gate, not to S_p.
@pytest.mark.parametrize("name", [n for n in DIRECTED_SSM_PARAMETERS if n != "approach_speed_threshold"])
def test_a_missing_parameter_is_refused_by_name(name):
    params = {k: v for k, v in PARAMS.items() if k != name}
    with pytest.raises(ValueError, match=name):
        separation_distance(params, TABLE, state="empty", robot_speed=0.5, yaw_rate=0.0,
                            v_r=0.5, v_h=0.5)


# ── the gate ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("distance, separation, v_r, expected", [
    (0.5, 1.0, 0.06, True),
    (0.5, 1.0, 0.05, False),     # on the threshold the robot is not approaching
    (1.0, 1.0, 0.5, False),      # on S_p the distance is not short of it
    (1.2, 1.0, 0.5, False),
])
def test_a_breach_needs_the_robot_to_approach(distance, separation, v_r, expected):
    assert separation_violated(distance, separation, v_r, 0.05) is expected


@pytest.mark.parametrize("v_r, standing", [(0.0, True), (0.05, True), (0.0500001, False)])
def test_a_standing_robot_is_the_complement_of_approaching(v_r, standing):
    assert robot_was_standing(v_r, 0.05) is standing
    assert separation_violated(0.0, 1.0, v_r, 0.05) is (not standing)


# ── the table's own check ────────────────────────────────────────────────────

def test_the_measured_table_is_usable():
    assert stop_table_problems(TABLE) == []


BROKEN_TABLES = {
    "not an object": ([], "객체가 아님"),
    "standard": (_edited(lambda t: t.update(determined_by="standard")), "determined_by"),
    "no date": (_edited(lambda t: t.pop("measured_on")), "measured_on"),
    "units": (_edited(lambda t: t["row_units"]["walking"].__setitem__(0, "km/h")), "row_units"),
    "a state missing": (_edited(lambda t: t["walking"].pop("carry_2kg")), "stop_table.walking"),
    "masses out of order": (_edited(lambda t: t["carry_mass_kg"].update(carry_2kg=4)), "carry_mass_kg"),
    "reach zero": (_edited(lambda t: t.update(turning_reach_m=0)), "turning_reach_m"),
    "speeds out of order": (_edited(lambda t: t["walking"]["carry_3kg"].reverse()),
                            "stop_table.walking.carry_3kg"),
    "a row at zero speed": (_edited(lambda t: t["walking"]["empty"][0].__setitem__(0, 0.0)),
                            "stop_table.walking.empty"),
    "a short row": (_edited(lambda t: t["turning"]["empty"][0].pop()), "stop_table.turning.empty"),
    "a negative time": (_edited(lambda t: t["walking"]["empty"][0].__setitem__(1, -0.5)),
                        "stop_table.walking.empty"),
    "a bool": (_edited(lambda t: t["walking"]["empty"][0].__setitem__(2, True)),
               "stop_table.walking.empty"),
    "no rows": (_edited(lambda t: t["turning"].__setitem__("carry_3kg", [])),
                "stop_table.turning.carry_3kg"),
    # determined_by, measured_on, row_units
    "no determined_by": (_edited(lambda t: t.pop("determined_by")), "determined_by"),
    "a blank date": (_edited(lambda t: t.update(measured_on="   ")), "measured_on"),
    "a date that is not text": (_edited(lambda t: t.update(measured_on=20261017)), "measured_on"),
    "row_units not an object": (_edited(lambda t: t.update(row_units=["m/s", "s", "m"])), "row_units"),
    "no row_units": (_edited(lambda t: t.pop("row_units")), "row_units"),
    "a turning unit changed": (_edited(lambda t: t["row_units"]["turning"].__setitem__(0, "deg/s")),
                               "row_units"),
    # carry_mass_kg
    "carry masses not an object": (_edited(lambda t: t.update(carry_mass_kg=[0, 2, 3])), "carry_mass_kg"),
    "a wrong carry state": (_edited(lambda t: t["carry_mass_kg"].update(
        {"carry_4kg": t["carry_mass_kg"].pop("carry_3kg")})), "carry_mass_kg: 상태가"),
    "an extra carry state": (_edited(lambda t: t["carry_mass_kg"].update(carry_5kg=5)),
                             "carry_mass_kg: 상태가"),
    "an infinite carry mass": (_edited(lambda t: t["carry_mass_kg"].update(carry_3kg=math.inf)),
                               "carry_mass_kg: 유한한"),
    "a nan carry mass": (_edited(lambda t: t["carry_mass_kg"].update(carry_2kg=math.nan)),
                         "carry_mass_kg: 유한한"),
    "a negative carry mass": (_edited(lambda t: t["carry_mass_kg"].update(empty=-1)),
                              "carry_mass_kg: 유한한"),
    "a carry mass that is a bool": (_edited(lambda t: t["carry_mass_kg"].update(carry_2kg=True)),
                                    "carry_mass_kg: 유한한"),
    "equal carry masses": (_edited(lambda t: t["carry_mass_kg"].update(carry_3kg=2)),
                           "carry_mass_kg: ['empty', 'carry_2kg', 'carry_3kg'] 순서"),
    # turning_reach_m
    "no reach": (_edited(lambda t: t.pop("turning_reach_m")), "turning_reach_m"),
    "an infinite reach": (_edited(lambda t: t.update(turning_reach_m=math.inf)), "turning_reach_m"),
    "a nan reach": (_edited(lambda t: t.update(turning_reach_m=math.nan)), "turning_reach_m"),
    "a negative reach": (_edited(lambda t: t.update(turning_reach_m=-0.465)), "turning_reach_m"),
    "a reach that is a bool": (_edited(lambda t: t.update(turning_reach_m=True)), "turning_reach_m"),
    # the motions and their states
    "a motion missing": (_edited(lambda t: t.pop("turning")), "stop_table.turning: 상태가"),
    "a motion not an object": (_edited(lambda t: t.update(walking=[])), "stop_table.walking: 상태가"),
    "an extra state": (_edited(lambda t: t["turning"].update(carry_5kg=[[0.5, 0.3, 0.03]])),
                       "stop_table.turning: 상태가"),
    "rows not a list": (_edited(lambda t: t["turning"].update(empty={"0.5": [0.5, 0.18, 0.02]})),
                        "stop_table.turning.empty: 행이 없음"),
    # one row
    "an infinite row value": (_edited(lambda t: t["walking"]["empty"][0].__setitem__(2, math.inf)),
                              "stop_table.walking.empty: 행마다"),
    "a nan row value": (_edited(lambda t: t["turning"]["carry_2kg"][1].__setitem__(1, math.nan)),
                        "stop_table.turning.carry_2kg: 행마다"),
    "a row that is a number": (_edited(lambda t: t["walking"]["empty"].__setitem__(1, 0.5)),
                               "stop_table.walking.empty: 행마다"),
    "a row that is an object": (_edited(lambda t: t["walking"]["carry_2kg"].__setitem__(
        0, {"speed": 0.3, "time": 0.74, "distance": 0.181})), "stop_table.walking.carry_2kg: 행마다"),
    "a long row": (_edited(lambda t: t["walking"]["empty"][0].append(0.0)),
                   "stop_table.walking.empty: 행마다"),
    "a text row value": (_edited(lambda t: t["walking"]["empty"][0].__setitem__(1, "0.54")),
                         "stop_table.walking.empty: 행마다"),
    "equal speeds": (_edited(lambda t: t["walking"]["empty"][1].__setitem__(0, 0.3)),
                     "stop_table.walking.empty: 속도가"),
    "a speed that is not increasing at the end": (
        _edited(lambda t: t["turning"]["carry_3kg"][1].__setitem__(0, 0.4)),
        "stop_table.turning.carry_3kg: 속도가"),
}


@pytest.mark.parametrize("name", sorted(BROKEN_TABLES))
def test_a_broken_table_is_named_where_it_breaks(name):
    table, where = BROKEN_TABLES[name]
    problems = stop_table_problems(table)
    assert problems and any(where in p for p in problems), problems


# ── reading the rule ─────────────────────────────────────────────────────────

def test_the_rule_reads_into_its_five_numbers_and_its_table():
    params, table = load_directed_ssm(_rules())
    assert params == PARAMS and table == TABLE
    assert tuple(params) == DIRECTED_SSM_PARAMETERS


def _without(name):
    rules = _rules()
    del rules["rules"][0]["parameters"][name]
    return rules


@pytest.mark.parametrize("rules, match", [
    ({"rules": []}, "0개"),
    ({"rules": _rules()["rules"] * 2}, "2개"),
    (_rules(approach_speed_threshold=-0.05), "approach_speed_threshold"),
    (_rules(robot_reaction_time=True), "robot_reaction_time"),
    (_without("intrusion_distance"), "intrusion_distance"),
    (_rules(table=[]), "stop_table"),
])
def test_a_rule_that_cannot_be_used_is_refused(rules, match):
    with pytest.raises(ValueError, match=match):
        load_directed_ssm(rules)


# ── which scenarios ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("scenario, declared", [
    ({"rules": ["R-TS15066-A2-*", DIRECTED_SSM_RULE_ID, CONTACT_RULE_ID]}, True),
    ({"rules": ["R-TS15066-SSM-*"]}, True),
    ({"rules": ["R-TS15066-A2-*", "R-TS15066-SSM-01", "R-10218-STOP-01"]}, False),
    ({"rules": []}, False),
    ({"scenario_id": "S1a"}, False),
])
def test_the_scenarios_that_declare_the_directed_rule(scenario, declared):
    assert scenario_declares_directed_ssm(scenario) is declared


# ── the canon and the validator's copy ───────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]


def _validator():
    import validator.validate_dataset as v
    return v


def _canon_rules():
    return json.loads((REPO_ROOT / "meta" / "rules.json").read_text(encoding="utf-8"))


def test_the_validator_names_the_same_parameters():
    assert _validator().REQUIRED_RULE_PARAMETERS[DIRECTED_SSM_RULE_ID] == DIRECTED_SSM_PARAMETERS


def test_the_validator_copies_the_table_layout():
    v = _validator()
    assert v.DIRECTED_SSM_RULE_ID == DIRECTED_SSM_RULE_ID
    assert (v.STOP_TABLE_MOTIONS, v.STOP_TABLE_STATES, v.STOP_TABLE_ROW_UNITS) == (
        STOP_TABLE_MOTIONS, STOP_TABLE_STATES, STOP_TABLE_ROW_UNITS)


@pytest.mark.parametrize("name", sorted(BROKEN_TABLES) + ["measured"])
def test_the_validator_and_the_generator_say_the_same_of_a_table(name):
    table = TABLE if name == "measured" else BROKEN_TABLES[name][0]
    assert _validator().stop_table_problems(copy.deepcopy(table)) == stop_table_problems(copy.deepcopy(table))


@pytest.mark.parametrize("name", DIRECTED_SSM_PARAMETERS)
@pytest.mark.parametrize("value", [0.1, 1, 0, 0.0, -0.05, float("nan"), float("inf"), -float("inf"),
                                   True, False, "0.1", None],
                         ids=repr)
def test_the_validator_and_the_generator_take_the_same_parameter_values(name, value):
    """threshold_problems (the validator's contract for every project value) and
    load_directed_ssm (what the recorder and the labeler read the rule with) accept or refuse
    a parameter's value alike -- one of them looser would let the other's refusal go unseen."""
    rules = _canon_rules()
    rule = next(r for r in rules["rules"] if r["rule_id"] == DIRECTED_SSM_RULE_ID)
    rule["parameters"][name]["value"] = value
    here = f"rules.{DIRECTED_SSM_RULE_ID}.parameters.{name}"
    validator_refuses = any(path == here for path, _ in _validator().threshold_problems(rules))
    try:
        load_directed_ssm(rules)
        generator_refuses = False
    except ValueError as err:
        assert name in str(err), err
        generator_refuses = True
    assert validator_refuses == generator_refuses


def test_the_canon_directed_rule_is_usable():
    params, table = load_directed_ssm(_canon_rules())
    assert set(params) == set(DIRECTED_SSM_PARAMETERS)
    assert stop_table_problems(table) == []
    assert carry_state(table, table["carry_mass_kg"][STOP_TABLE_STATES[-1]]) == STOP_TABLE_STATES[-1]


def test_the_canon_contact_rule_fails_on_contact_alone():
    entry = next(r for r in _canon_rules()["rules"] if r["rule_id"] == CONTACT_RULE_ID)
    assert (entry["outcome"], entry["hazard_class"]) == ("fail", "collision")
    # no limits of its own: severity is the A2 rules' to say
    assert "parameters" not in entry and "limits_ref" not in entry and "body_region" not in entry
