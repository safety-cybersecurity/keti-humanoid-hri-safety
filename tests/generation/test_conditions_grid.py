"""The optional ``grid`` block of a conditions table: fixed-cell rows after the random ones. Grids are tested on
copies of the canon's entries with a grid set on them, and the canon's own grids are read at the end."""

import copy
import csv
import math

import pytest

from generation import conditions
from generation.scene import layout

S1E_GRID = {"cells": 7, "robot_axis": {"quantity": "turn_radius_m", "range": [1.25, 2.0]},
            "human_axis": {"quantity": "face_distance_m", "range": [0.3, 0.8]},
            "fixed": {"corner_distance_m": 0.9, "facing_yaw_deg": 90.0, "robot_speed_mps": 0.925, "box_mass_kg": 2.0,
                      "box_depth_m": 0.275, "box_height_m": 0.25, "sensor_delay_ms": 50.0}, "note": "시험"}
S1F_GRID = {"cells": 7, "robot_axis": {"quantity": "robot_speed_mps", "range": [0.6, 1.0]},
            "human_axis": {"quantity": "trigger_distance_m", "range": [0.25, 2.75]},
            "fixed": {"human_start_x_m": 1.5, "turn_time_s": 1.0, "entry_angle_deg": 30.0, "human_speed_mps": 1.2,
                      "box_mass_kg": 2.0, "box_depth_m": 0.275, "box_height_m": 0.25, "sensor_delay_ms": 50.0},
            "note": "시험"}
S1G_GRID = {"cells": 7, "robot_axis": {"quantity": "turn_rate_rad_s", "range": [0.5, 1.0]},
            "human_axis": {"quantity": "arrival_s", "range": [1.0, 5.9]},
            "fixed": {"approach_side": "west", "lateral_offset_m": 0.025, "human_speed_mps": 1.2, "box_mass_kg": 2.0,
                      "box_depth_m": 0.275, "box_height_m": 0.25, "sensor_delay_ms": 50.0}, "note": "시험"}


@pytest.fixture(scope="module")
def boxes():
    return layout.build_layout(layout.load_spec(layout.SCENE_SPEC))


def _with_grid(sid, grid, rows=50):            # a copy of the canon entry: `rows` random rows and the grid after them
    entry = copy.deepcopy(conditions.scenario_entry(sid))
    entry["conditions"]["rows"] = rows
    entry["conditions"]["grid"] = copy.deepcopy(grid)
    entry["episodes"] = rows + grid["cells"] ** 2
    return entry


def test_a_grid_adds_its_cells_after_the_random_rows(boxes):
    entry = _with_grid("S1e", S1E_GRID)
    spec = conditions.load_conditions(entry)
    assert (spec.rows, spec.total) == (50, 99)
    assert [conditions.grid_cell(spec, k) for k in (49, 50, 51, 57, 98)] == [None, (0, 0), (1, 0), (0, 1), (6, 6)]
    row = conditions.draw_row(entry, 50 + 3 + 7 * 5, boxes)
    assert row["turn_radius_m"] == pytest.approx(1.25 + 0.75 * 3.5 / 7)
    assert row["face_distance_m"] == pytest.approx(0.3 + 0.5 * 5.5 / 7)
    assert row["corner_distance_m"] == 0.9 and set(row) == set(spec.columns) | set(spec.fixed)
    assert conditions.row_design(entry, 7) == ("random", None)
    assert conditions.row_design(entry, 50 + 3 + 7 * 5) == ("grid", (3, 5))
    assert len(conditions.table(entry, boxes)) == 50 and len(conditions.grid_table(entry, boxes)) == 49


def test_random_rows_do_not_depend_on_the_grid(boxes):
    assert conditions.draw_row(_with_grid("S1e", S1E_GRID), 7, boxes) == conditions.draw_row(conditions.scenario_entry("S1e"), 7, boxes)


def test_a_worked_out_axis_is_written_to_its_column(boxes):
    row = conditions.draw_row(_with_grid("S1g", S1G_GRID), 50 + 2 + 7 * 4, boxes)
    w, arrival = 0.5 + 0.5 * 2.5 / 7, 1.0 + 4.9 * 4.5 / 7
    assert row["turn_rate_rad_s"] == pytest.approx(w) and row["approach_side"] == "west"
    assert row["pass_offset_s"] == pytest.approx(arrival - math.pi / w)
    assert conditions.s1g_trigger_distance(row) == pytest.approx(1.2 * arrival)


def test_a_trigger_distance_axis_is_written_as_a_lead_time(boxes):
    row = conditions.draw_row(_with_grid("S1f", S1F_GRID), 50 + 2 + 7 * 4, boxes)
    v, d = 0.6 + 0.4 * 2.5 / 7, 0.25 + 2.5 * 4.5 / 7
    assert row["robot_speed_mps"] == pytest.approx(v) and row["turn_time_s"] == 1.0
    assert row["trigger_lead_s"] == pytest.approx(d / v)
    assert conditions.s1f_trigger_distance(row) == pytest.approx(d)


def test_a_cell_the_guard_refuses_is_named(boxes):                      # the guard runs on grid cells too
    grid = copy.deepcopy(S1E_GRID); grid["fixed"]["corner_distance_m"] = 100.0     # outside the building
    with pytest.raises(ValueError, match=r"격자 칸 \(0, 0\)"):
        conditions.draw_row(_with_grid("S1e", grid), 50, boxes)


def test_rows_past_the_table_are_refused(boxes):
    with pytest.raises(ValueError, match="행 99"):
        conditions.draw_row(_with_grid("S1e", S1E_GRID), 99, boxes)
    canon = conditions.scenario_entry("S1e")
    with pytest.raises(ValueError):
        conditions.draw_row(canon, conditions.load_conditions(canon).total, boxes)


@pytest.mark.parametrize("change, message", [
    (lambda g: g.update(cells=1), "cells"),
    (lambda g: g.update(extra=1), "모르는 키"),
    (lambda g: g["robot_axis"].update(quantity="arrival_s"), "robot_axis"),        # the robot's axis is a column
    (lambda g: g["human_axis"].update(quantity="nothing"), "human_axis"),
    (lambda g: g["human_axis"].update(quantity="turn_radius_m"), "같은 열"),
    (lambda g: g["human_axis"].update(range=[0.8, 0.3]), "range"),
    (lambda g: g["fixed"].pop("sensor_delay_ms"), "sensor_delay_ms"),
    (lambda g: g["fixed"].update(turn_radius_m=1.5), "turn_radius_m"),             # an axis column is not fixed too
    (lambda g: g["fixed"].update(box_mass_kg=-1.0), "box_mass_kg"),
])
def test_load_refuses_bad_grids(change, message):
    grid = copy.deepcopy(S1E_GRID); change(grid)
    with pytest.raises(ValueError, match=message):
        conditions.load_conditions(_with_grid("S1e", grid))


def test_a_worked_out_column_is_checked_in_every_cell():
    grid = copy.deepcopy(S1F_GRID); grid["human_axis"]["range"] = [-2.0, -1.0]      # a lead time below zero
    with pytest.raises(ValueError, match=r"human_axis.*trigger_lead_s|trigger_lead_s.*human_axis"):
        conditions.load_conditions(_with_grid("S1f", grid))
    conditions.load_conditions(_with_grid("S1f", S1F_GRID))                          # the positive range loads
    for sid in ("S1e", "S1f", "S1g"):                                                 # and the canon still loads
        assert conditions.load_conditions(conditions.scenario_entry(sid)).grid is not None


def test_a_quantity_that_reads_a_missing_value_is_refused_at_load(monkeypatch):
    column = conditions.GRID_QUANTITIES["trigger_distance_m"][0]
    monkeypatch.setitem(conditions.GRID_QUANTITIES, "trigger_distance_m",
                        (column, lambda value, row: value / float(row["no_such_column"])))
    with pytest.raises(ValueError, match="human_axis.*no_such_column"):
        conditions.load_conditions(_with_grid("S1f", S1F_GRID))


def test_episodes_is_the_random_rows_and_the_cells():
    entry = _with_grid("S1e", S1E_GRID); entry["episodes"] = 50
    with pytest.raises(ValueError, match="episodes"):
        conditions.load_conditions(entry)


def test_main_writes_the_grid_rows_after_the_random_ones(tmp_path, monkeypatch):
    entry = _with_grid("S1e", S1E_GRID, rows=3)
    monkeypatch.setattr(conditions, "scenario_entry", lambda sid, doc=None: entry)
    out = tmp_path / "s1e.csv"
    assert conditions.main(["--scenario", "S1e", "--out", str(out)]) == 0
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["row"]) for r in rows] == list(range(3 + 49))
    assert [r["attempt"] for r in rows[3:]] == ["grid"] * 49 and all(int(r["attempt"]) >= 0 for r in rows[:3])
    assert float(rows[3 + 1]["turn_radius_m"]) == pytest.approx(1.25 + 0.75 * 1.5 / 7)


@pytest.mark.parametrize("index", [99, 10_000, -1, True, 50.5, "50"])
def test_a_row_the_table_does_not_have_is_not_classified(index):
    entry = _with_grid("S1e", S1E_GRID)
    with pytest.raises(ValueError, match="행"):
        conditions.row_design(entry, index)
    with pytest.raises(ValueError, match="행"):
        conditions.grid_cell(conditions.load_conditions(entry), index)


@pytest.mark.parametrize("change, message", [
    (lambda g: g["robot_axis"].update(quantity="approach_side"), "robot_axis"),     # a choice column is no axis
    (lambda g: (g["human_axis"].update(quantity="approach_side"), g["fixed"].pop("approach_side"),
                g["fixed"].update(turn_rate_rad_s=0.75, pass_offset_s=0.0)), "human_axis"),
])
def test_an_axis_is_a_range_column(change, message):
    grid = copy.deepcopy(S1G_GRID); change(grid)
    with pytest.raises(ValueError, match=message):
        conditions.load_conditions(_with_grid("S1g", grid))


@pytest.mark.parametrize("sid", ["S1e", "S1f", "S1g"])
def test_the_canon_has_fifty_random_rows_and_a_seven_by_seven_grid(sid, boxes):
    entry = conditions.scenario_entry(sid)
    spec = conditions.load_conditions(entry)
    assert (spec.rows, spec.grid.cells, spec.total, entry["episodes"]) == (50, 7, 99, 99)
    rows = conditions.grid_table(entry, boxes)                                   # every cell passes its guard
    assert len(rows) == 49 and all(conditions.row_problems(entry, r, boxes) == [] for r in rows)


def test_s1e_keeps_the_rows_of_the_first_fifty(boxes):                           # the seed does not depend on the row count
    entry = conditions.scenario_entry("S1e")
    wide = copy.deepcopy(entry)
    wide["conditions"]["rows"] = 400
    wide["conditions"].pop("grid")
    wide["episodes"] = 400
    assert conditions.table(entry, boxes) == conditions.table(wide, boxes)[:50]
