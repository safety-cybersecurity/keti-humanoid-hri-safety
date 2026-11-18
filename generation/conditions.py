#!/usr/bin/env python3
"""The conditions table of each new scenario: one row per episode, drawn row by row from its own seed and guarded
against the racks, and the place each row gives the robot and the person.

    .venv/bin/python generation/conditions.py --scenario S1e --out <csv>      # the table as a file, to look at

The table is meta/scenarios.json's ``conditions`` (S1e, S1f, S1g) -- the project's own (self-set), none of it a
standard value:

* ``rows`` -- how many random rows; the scenario's ``episodes`` is these plus the grid's cells (``total``). Row i is the generation's ``--seq i`` (one row, one process).
* ``columns`` -- {name: {"range": [lo, hi]}} drawn uniformly, or {name: {"choice": [...]}} drawn with equal chances,
  in the order they are written. Every column is a number but ``approach_side``, whose choices are APPROACH_SIDES; a
  number column's choices are held to the column's domain one by one, as a range's ends are.
* ``fixed`` -- {name: number}, the same in every row.
* ``grid`` (optional) -- cells x cells further rows after the random ones, each a fixed cell of two axes: the robot
  axis (a column) and the human axis (a column, or a GRID_QUANTITIES name whose value is worked out into its column),
  both at the middle of their cell of ``range``; every other column is held at ``grid.fixed``. Row
  ``rows + i + cells * j`` is the cell i of the robot axis and j of the human axis. The guard checks a cell as it
  checks a drawn row but does not redraw it.

Row i's n-th attempt draws every column from ``numpy.random.default_rng`` seeded with the first 8 bytes (big-endian)
of SHA-256("<scenario>:condition:<i>:<n>"), so a row is the same whichever rows are drawn before it and in whichever
process. An attempt the guard refuses (row_problems) is followed by the next one; after MAX_ATTEMPTS refused attempts
the row does not exist (ValueError naming it).

What a row places (row_site, human_walk), by the kind of the site's path (generation/sites.py):

* ``u_turn`` (S1e) -- the robot's lanes in and out ``turn_radius_m`` from the turn's centre (the row end's middle);
  the person standing still ``corner_distance_m`` past the row end along the way out and ``face_distance_m`` off the
  rack face between the turn's centre and the way out.
* ``straight`` (S1f) -- the person at ``human_start_x_m`` on the site's line, facing the rack beside it with its back
  to the robot's lane (s1f_rack_facing_yaw_deg); when the robot coming from behind is s1f_trigger_distance() behind it
  along the aisle (``trigger_lead_s`` at the task's speed ``robot_speed_mps``) it turns on the spot for
  ``turn_time_s`` toward the robot's heading and walks toward the robot's lane at ``entry_angle_deg`` from that
  heading, stopping FAR_FACE_STOP_M before the far rack face.
* ``turn_then_walk`` (S1g) -- the robot turning away from where the person comes from (half way round it faces the
  way the person walks); the person walking along the aisle ``lateral_offset_m`` off the middle of the site's region,
  from s1g_human_start_distance() before the robot's x to WALK_PAST_M past it.

The guard (row_problems, the project's own values): the robot's body (a disc of ROBOT_HALF_WIDTH_M about its root)
and the box it holds (carry.box_obb at the row's size), along the row's nominal path, ROBOT_CLEARANCE_M clear of
every collider reaching into sites.OBSTACLE_Z_M and inside the building by as much; a turn on the spot, the circle
the body and the box sweep, as clear; the person's place and walk HUMAN_CLEARANCE_M clear of the colliders and
WALL_INSET_M inside the building's walls -- for ``turn_then_walk`` from either side the person may come from (the
row with the other approach_side must pass as well), so the timing columns the table keeps do not depend on the side
drawn. Distances are between footprints seen from above. Pure numpy, no Isaac.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation import carry as carry_canon  # noqa: E402
from generation import sites  # noqa: E402
from generation.robots import kinematics  # noqa: E402
from generation.scene import layout  # noqa: E402
from generation.scene.carried_box import NOMINAL_SIZE_M  # noqa: E402

CONDITIONS_KEY = "conditions"
CONDITIONS_KEYS = ("rows", "columns", "fixed", "determined_by", "note")
GRID_KEY = "grid"                                   # optional key of a conditions block
GRID_KEYS = ("cells", "robot_axis", "human_axis", "fixed", "note")
# A quantity a grid's human axis may name instead of a column: (the column the cell's value is written to,
# f(value, row so far) -> that column's value).
GRID_QUANTITIES: dict[str, tuple[str, Callable[[float, Mapping], float]]] = {
    "arrival_s": ("pass_offset_s", lambda value, row: value - math.pi / float(row["turn_rate_rad_s"])),
    "trigger_distance_m": ("trigger_lead_s", lambda value, row: value / float(row["robot_speed_mps"])),
}
COLUMN_KINDS = ("range", "choice")
DETERMINED_BY = "project"
MAX_ATTEMPTS = 200
APPROACH_SIDES = ("west", "east")
SCENARIOS_JSON = REPO / "meta" / "scenarios.json"
# The dummy's gait drive: its speed rises from rest to the walking speed over ramp.accel_time_s.
GAIT_SPEC = REPO / "generation" / "dummy" / "gait" / "gait_spec.yaml"

# The guard (the project's own values, m).
ROBOT_HALF_WIDTH_M = 0.30          # the robot's body, a disc about its root
ROBOT_CLEARANCE_M = 0.15           # the body and the held box from a collider, and inside the building
HUMAN_CLEARANCE_M = 0.30           # the person's centre from a collider
WALL_INSET_M = 0.40                # the person's centre inside the building's walls
FAR_FACE_STOP_M = 0.45             # S1f: the person stops this far before the far rack face
WALK_PAST_M = 3.0                  # S1g: the person walks this far past the robot's x and stops

# What each kind of path reads from a row, besides COMMON_COLUMNS and, with a carry block, CARRY_COLUMNS.
PATH_COLUMNS = {
    "u_turn": ("corner_distance_m", "face_distance_m", "facing_yaw_deg", "turn_radius_m", "robot_speed_mps"),
    "straight": ("human_start_x_m", "trigger_lead_s", "turn_time_s", "entry_angle_deg", "human_speed_mps",
                 "robot_speed_mps"),
    "turn_then_walk": ("approach_side", "lateral_offset_m", "human_speed_mps", "turn_rate_rad_s", "pass_offset_s"),
}
PATH_FIXED = {"u_turn": (), "straight": (), "turn_then_walk": ("walk_speed_mps", "run_up_s")}
COMMON_COLUMNS = ("sensor_delay_ms",)
CARRY_COLUMNS = ("box_mass_kg", "box_depth_m", "box_height_m")
# The lower end a value may take: positive, or not below zero.
POSITIVE = frozenset({"turn_radius_m", "robot_speed_mps", "human_speed_mps", "turn_rate_rad_s", "box_mass_kg",
                      "box_depth_m", "box_height_m", "trigger_lead_s", "turn_time_s", "walk_speed_mps"})
NON_NEGATIVE = frozenset({"corner_distance_m", "face_distance_m", "sensor_delay_ms", "run_up_s"})
# Angles strictly between 0 and 180 degrees.
OPEN_HALF_TURN = frozenset({"entry_angle_deg"})


@dataclass(frozen=True)
class Column:
    """One column of a conditions table: ``kind`` "range" with ``values`` (lo, hi), or "choice" with the choices."""
    kind: str
    values: tuple

    def draw(self, rng: np.random.Generator) -> float | str:
        if self.kind == "range":
            return float(rng.uniform(*self.values))
        return self.values[int(rng.integers(len(self.values)))]


@dataclass(frozen=True)
class GridAxis:
    quantity: str            # a column's name, or (human axis only) a GRID_QUANTITIES name
    lo: float
    hi: float


@dataclass(frozen=True)
class GridSpec:
    cells: int               # per axis; the grid has cells ** 2 rows
    robot_axis: GridAxis
    human_axis: GridAxis
    fixed: Mapping[str, float | str]


@dataclass(frozen=True)
class ConditionSpec:
    """The canon's ``conditions`` of one scenario, read and checked by load_conditions."""
    scenario_id: str
    rows: int                # the random rows
    columns: Mapping[str, Column]
    fixed: Mapping[str, float]
    grid: GridSpec | None = None

    @property
    def total(self) -> int:
        """Every row of the table: the random rows and the grid's cells."""
        return self.rows + (self.grid.cells ** 2 if self.grid else 0)


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _domain(where: str, name: str, lo: float, hi: float) -> None:
    if name in POSITIVE and lo <= 0.0:
        raise ValueError(f"{where} 의 아래 끝이 양수가 아닙니다: {lo!r}")
    if name in NON_NEGATIVE and lo < 0.0:
        raise ValueError(f"{where} 의 아래 끝이 음수입니다: {lo!r}")
    if name in OPEN_HALF_TURN and not (0.0 < lo and hi < 180.0):
        raise ValueError(f"{where} 는 0 과 180 사이(양 끝 제외)여야 합니다: [{lo}, {hi}]")


def _choice_value(where: str, name: str, value: object) -> None:
    """One choice of a number column: a finite number in the column's domain -- the checks a range's ends get."""
    at = f"{where} 의 choice 값 {value!r}"
    if not _number(value):
        raise ValueError(f"{at} 이 유한한 수가 아닙니다 — {name} 은 수의 열입니다")
    v = float(value)
    if name in POSITIVE and v <= 0.0:
        raise ValueError(f"{at} 이 양수가 아닙니다")
    if name in NON_NEGATIVE and v < 0.0:
        raise ValueError(f"{at} 이 음수입니다")
    if name in OPEN_HALF_TURN and not 0.0 < v < 180.0:
        raise ValueError(f"{at} 은 0 과 180 사이(양 끝 제외)여야 합니다")


def _column(where: str, name: str, node: object) -> Column:
    if not isinstance(node, Mapping):
        raise ValueError(f"{where} 이 객체가 아닙니다: {node!r}")
    kinds = [k for k in COLUMN_KINDS if k in node]
    if len(kinds) != 1 or set(node) != set(kinds):
        raise ValueError(f"{where} 은 range · choice 가운데 하나만 가져야 합니다: {dict(node)!r}")
    if name == "approach_side" and kinds[0] != "choice":
        raise ValueError(f"{where} 의 choice 는 {list(APPROACH_SIDES)} 가운데서 고른 목록이어야 합니다: {dict(node)!r}")
    if kinds[0] == "range":
        span = node["range"]
        if (not isinstance(span, (list, tuple)) or len(span) != 2 or not all(_number(v) for v in span)
                or not float(span[0]) < float(span[1])):
            raise ValueError(f"{where} 의 range 가 아래 끝 < 위 끝인 유한한 수 둘이 아닙니다: {span!r}")
        lo, hi = float(span[0]), float(span[1])
        _domain(where, name, lo, hi)
        return Column("range", (lo, hi))
    options = node["choice"]
    if (not isinstance(options, (list, tuple)) or not options
            or not all(isinstance(v, str) or _number(v) for v in options) or len(set(options)) != len(options)):
        raise ValueError(f"{where} 의 choice 가 서로 다른 값(문자열이나 유한한 수)의 비어 있지 않은 목록이 아닙니다: {options!r}")
    if name == "approach_side":
        if not set(options) <= set(APPROACH_SIDES):
            raise ValueError(f"{where} 의 choice 는 {list(APPROACH_SIDES)} 가운데서 고른 목록이어야 합니다: {options!r}")
    else:
        for value in options:
            _choice_value(where, name, value)
    return Column("choice", tuple(options))


def _grid_axis(where: str, node: object) -> GridAxis:
    if not isinstance(node, Mapping) or set(node) != {"quantity", "range"}:
        raise ValueError(f"{where} 가 quantity · range 만 가진 객체가 아닙니다: {node!r}")
    quantity, span = node["quantity"], node["range"]
    if not isinstance(quantity, str) or not quantity:
        raise ValueError(f"{where}.quantity 가 비어 있지 않은 문자열이 아닙니다: {quantity!r}")
    if (not isinstance(span, (list, tuple)) or len(span) != 2 or not all(_number(v) for v in span)
            or not float(span[0]) < float(span[1])):
        raise ValueError(f"{where}.range 가 아래 끝 < 위 끝인 유한한 수 둘이 아닙니다: {span!r}")
    return GridAxis(quantity, float(span[0]), float(span[1]))


def _grid_shape(where: str, node: object) -> tuple[int, GridAxis, GridAxis, Mapping]:
    """(cells, robot axis, human axis, the raw fixed block) of a ``grid`` block -- what can be checked without the
    scenario's columns."""
    if not isinstance(node, Mapping):
        raise ValueError(f"{where} 가 객체가 아닙니다: {node!r}")
    for key in GRID_KEYS:
        if key not in node:
            raise ValueError(f"{where}.{key} 가 없습니다")
    unknown = sorted(set(node) - set(GRID_KEYS))
    if unknown:
        raise ValueError(f"{where} 에 모르는 키 {unknown} 가 있습니다 (키: {list(GRID_KEYS)})")
    cells = node["cells"]
    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 2:
        raise ValueError(f"{where}.cells 가 2 이상의 정수가 아닙니다: {cells!r}")
    if not isinstance(node["note"], str) or not node["note"].strip():
        raise ValueError(f"{where}.note 가 비어 있지 않은 문자열이 아닙니다: {node['note']!r}")
    if not isinstance(node["fixed"], Mapping):
        raise ValueError(f"{where}.fixed 가 객체가 아닙니다: {node['fixed']!r}")
    return (cells, _grid_axis(f"{where}.robot_axis", node["robot_axis"]),
            _grid_axis(f"{where}.human_axis", node["human_axis"]), node["fixed"])


def _grid_spec(where: str, shape: tuple, columns: Mapping[str, Column],
               conditions_fixed: Mapping[str, float]) -> GridSpec:
    """The GridSpec of a grid block's shape against the scenario's columns: the axes are numeric columns (the human
    axis may be a GRID_QUANTITIES name), no two axes write one column, and ``fixed`` holds exactly the other columns,
    each at a value its column's domain allows. A human axis that names a GRID_QUANTITIES quantity is worked out into
    its column for every cell, exactly as a row would hold it, and the column's domain is checked on each result."""
    cells, robot, human, raw_fixed = shape
    ranged = {name for name, col in columns.items() if col.kind == "range"}
    if robot.quantity not in ranged:
        raise ValueError(f"{where}.robot_axis.quantity {robot.quantity!r} 가 이 시나리오의 range 열이 아닙니다 "
                         f"(range 열: {sorted(ranged)})")
    if human.quantity in ranged:
        human_col = human.quantity
    elif human.quantity in GRID_QUANTITIES and GRID_QUANTITIES[human.quantity][0] in ranged:
        human_col = GRID_QUANTITIES[human.quantity][0]
    else:
        raise ValueError(f"{where}.human_axis.quantity {human.quantity!r} 가 이 시나리오의 range 열도 아니고 쓸 수 있는 "
                         f"양도 아닙니다 (range 열: {sorted(ranged)} · 양: {list(GRID_QUANTITIES)})")
    if human_col == robot.quantity:
        raise ValueError(f"{where} 의 두 축이 같은 열 {human_col!r} 에 적힙니다")
    _domain(f"{where}.robot_axis", robot.quantity, robot.lo, robot.hi)
    if human.quantity in columns:
        _domain(f"{where}.human_axis", human.quantity, human.lo, human.hi)
    want = [name for name in columns if name not in (robot.quantity, human_col)]
    for name in want:
        if name not in raw_fixed:
            raise ValueError(f"{where}.fixed.{name} 가 없습니다 — 두 축이 적지 않는 열 {want} 의 값이 필요합니다")
    fixed = {}
    for name, value in raw_fixed.items():
        at = f"{where}.fixed.{name}"
        if name not in want:
            raise ValueError(f"{at} 는 두 축이 적지 않는 열이 아닙니다 (있어야 하는 값: {want})")
        if name == "approach_side":
            if value not in APPROACH_SIDES:
                raise ValueError(f"{at} 는 {list(APPROACH_SIDES)} 가운데 하나여야 합니다: {value!r}")
            fixed[name] = value
        else:
            _choice_value(at, name, value)
            fixed[name] = float(value)
    grid = GridSpec(cells, robot, human, MappingProxyType({n: fixed[n] for n in want}))
    if human.quantity not in columns:
        for j in range(cells):
            for i in range(cells):
                try:
                    row = _cell_row(grid, columns, conditions_fixed, (i, j))
                except (KeyError, ZeroDivisionError) as err:
                    raise ValueError(f"{where}.human_axis.quantity {human.quantity!r} 의 식이 읽는 값 {err} 을 이 "
                                     f"시나리오의 행이 갖고 있지 않거나 0 으로 나눕니다") from err
                _choice_value(f"{where}.human_axis (칸 ({i}, {j}) 에서 {human_col} 으로 풀린 값)", human_col,
                              row[human_col])
    return grid


def _required(entry: Mapping, kind: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(the columns, the fixed values) a scenario with this kind of path and its carry block reads."""
    columns = PATH_COLUMNS[kind] + (CARRY_COLUMNS if carry_canon.CARRY_KEY in entry else ()) + COMMON_COLUMNS
    return columns, PATH_FIXED[kind]


def load_conditions(entry: Mapping) -> ConditionSpec:
    """The ``conditions`` of a meta/scenarios.json scenario entry. A block that is not the canon's form -- or lacks a
    column or a fixed value the scenario reads, or carries one it does not -- is refused with a ValueError that names
    ``<scenario_id>.conditions.<column>``. The scenario must declare a site (scene.site): the kind of its path says
    which columns it reads."""
    sid = entry.get("scenario_id") if isinstance(entry, Mapping) else None
    where = f"{sid}.{CONDITIONS_KEY}"
    if not isinstance(entry, Mapping) or CONDITIONS_KEY not in entry:
        raise ValueError(f"{where} 가 없습니다")
    node = entry[CONDITIONS_KEY]
    if not isinstance(node, Mapping):
        raise ValueError(f"{where} 가 객체가 아닙니다: {node!r}")
    for key in CONDITIONS_KEYS:
        if key not in node:
            raise ValueError(f"{where}.{key} 가 없습니다")
    unknown = sorted(set(node) - set(CONDITIONS_KEYS) - {GRID_KEY})
    if unknown:
        raise ValueError(f"{where} 에 모르는 키 {unknown} 가 있습니다 (키: {list(CONDITIONS_KEYS) + [GRID_KEY]})")
    rows = node["rows"]
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError(f"{where}.rows 가 양의 정수가 아닙니다: {rows!r}")
    grid_shape = _grid_shape(f"{where}.{GRID_KEY}", node[GRID_KEY]) if GRID_KEY in node else None
    total = rows + (grid_shape[0] ** 2 if grid_shape else 0)
    if "episodes" in entry and total != entry["episodes"]:
        if grid_shape:
            raise ValueError(f"{where}.rows + grid.cells² 가 episodes({entry['episodes']!r}) 와 다릅니다: "
                             f"{rows} + {grid_shape[0] ** 2} = {total} — 행 하나가 에피소드 하나입니다")
        raise ValueError(f"{where}.rows 가 episodes({entry['episodes']!r}) 와 다릅니다: {rows!r} — 행 하나가 에피소드 하나입니다")
    if node["determined_by"] != DETERMINED_BY:
        raise ValueError(f"{where}.determined_by 가 {DETERMINED_BY!r} 가 아닙니다 — 조건표는 자체 설정입니다: "
                         f"{node['determined_by']!r}")
    if not isinstance(node["note"], str) or not node["note"].strip():
        raise ValueError(f"{where}.note 가 비어 있지 않은 문자열이 아닙니다: {node['note']!r}")
    raw_columns = node["columns"]
    if not isinstance(raw_columns, Mapping) or not raw_columns:
        raise ValueError(f"{where}.columns 가 비어 있지 않은 객체가 아닙니다: {raw_columns!r}")
    columns = {}
    for name, col in raw_columns.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{where}.columns 의 이름이 비어 있지 않은 문자열이 아닙니다: {name!r}")
        columns[name] = _column(f"{where}.{name}", name, col)
    raw_fixed = node["fixed"]
    if not isinstance(raw_fixed, Mapping):
        raise ValueError(f"{where}.fixed 가 객체가 아닙니다: {raw_fixed!r}")
    fixed = {}
    for name, value in raw_fixed.items():
        at = f"{where}.fixed.{name}"
        if not _number(value):
            raise ValueError(f"{at} 이 유한한 수가 아닙니다: {value!r}")
        if name in POSITIVE and value <= 0:
            raise ValueError(f"{at} 이 양수가 아닙니다: {value!r}")
        if name in NON_NEGATIVE and value < 0:
            raise ValueError(f"{at} 이 음수입니다: {value!r}")
        if name in columns:
            raise ValueError(f"{at} 가 columns 에도 있습니다 — 한 값은 한 자리에만")
        fixed[name] = float(value)
    kind = sites.load_site(entry).path["kind"]
    want_columns, want_fixed = _required(entry, kind)
    for name in want_columns:
        if name not in columns:
            raise ValueError(f"{where}.{name} 가 없습니다 — {sid} 는 {list(want_columns)} 를 읽습니다")
    for name in columns:
        if name not in want_columns:
            raise ValueError(f"{where}.{name} 는 이 시나리오가 읽지 않는 열입니다 (읽는 열: {list(want_columns)})")
    for name in want_fixed:
        if name not in fixed:
            raise ValueError(f"{where}.fixed.{name} 가 없습니다 — {sid} 는 {list(want_fixed)} 를 읽습니다")
    for name in fixed:
        if name not in want_fixed:
            raise ValueError(f"{where}.fixed.{name} 는 이 시나리오가 읽지 않는 값입니다 (읽는 값: {list(want_fixed)})")
    grid = _grid_spec(f"{where}.{GRID_KEY}", grid_shape, columns, fixed) if grid_shape else None
    return ConditionSpec(scenario_id=sid, rows=rows, columns=MappingProxyType(columns),
                         fixed=MappingProxyType(fixed), grid=grid)


def _scenarios_doc() -> Mapping:
    return json.loads(SCENARIOS_JSON.read_text(encoding="utf-8"))


def table_scenarios(doc: Mapping | None = None) -> tuple[str, ...]:
    """The scenarios of meta/scenarios.json (or ``doc``) that have a conditions table, in the canon's order."""
    doc = _scenarios_doc() if doc is None else doc
    return tuple(e["scenario_id"] for e in doc["scenarios"] if CONDITIONS_KEY in e)


def scenario_entry(scenario_id: str, doc: Mapping | None = None) -> Mapping:
    """The meta/scenarios.json (or ``doc``) entry of a scenario that has a conditions table; ValueError naming
    ``scenario_id`` when there is none."""
    doc = _scenarios_doc() if doc is None else doc
    entry = next((e for e in doc["scenarios"] if e.get("scenario_id") == scenario_id), None)
    if entry is None:
        raise ValueError(f"meta/scenarios.json 에 {scenario_id!r} 가 없습니다")
    if CONDITIONS_KEY not in entry:
        raise ValueError(f"{scenario_id} 는 조건표(conditions)가 없는 시나리오입니다 — 조건표가 있는 것: "
                         f"{list(table_scenarios(doc))}")
    return entry


def box_size(row: Mapping) -> tuple[float, float, float]:
    """(width, depth, height) m of the box a row's robot holds: the carried box's nominal width, the row's
    ``box_depth_m`` and ``box_height_m``."""
    return (float(NOMINAL_SIZE_M[0]), float(row["box_depth_m"]), float(row["box_height_m"]))


def row_seed(scenario_id: str, index: int, attempt: int) -> int:
    """The seed of a row's attempt: the first 8 bytes, big-endian, of SHA-256("<scenario>:condition:<row>:<attempt>")."""
    digest = hashlib.sha256(f"{scenario_id}:condition:{index}:{attempt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# ── what a row places ───────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _strips() -> tuple[tuple[float, float], ...]:
    """The strips of floor no rack row stands on (layout.aisle_strips of the scene spec): their edges are rack faces."""
    return tuple(layout.aisle_strips(layout.load_spec(layout.SCENE_SPEC)))


def _strip(sid: str, y: float, what: str) -> tuple[float, float]:
    found = [s for s in _strips() if s[0] <= y <= s[1]]
    if len(found) != 1:
        raise ValueError(f"{sid}: {what} y {y} 가 랙 줄 사이 통로 하나 위에 있지 않습니다 ({list(_strips())})")
    return found[0]


@functools.lru_cache(maxsize=1)
def _accel_time_s() -> float:
    return float(yaml.safe_load(GAIT_SPEC.read_text(encoding="utf-8"))["ramp"]["accel_time_s"])


def _ramp_distance(speed: float, seconds: float) -> float:
    """How far the dummy's gait drive walks in ``seconds`` from rest: its speed rising linearly to ``speed`` over
    the gait's acceleration time, then held."""
    ramp = _accel_time_s()
    if seconds <= ramp:
        return speed * seconds * seconds / (2.0 * ramp)
    return speed * (seconds - ramp / 2.0)


def s1g_trigger_distance(row: Mapping) -> float:
    """S1g: how far along the aisle the person is from the robot's x when the robot's task comes --
    human_speed_mps * (pi / turn_rate_rad_s + pass_offset_s), so that at full speed the person passes the robot's x
    pass_offset_s after the nominal turning time pi / turn_rate_rad_s (worked out from the turn_rate_rad_s column --
    not when the robot has in fact turned: the turn ends once the yaw error is within 5 degrees, and the robot turns
    later than commanded)."""
    return float(row["human_speed_mps"]) * (math.pi / float(row["turn_rate_rad_s"]) + float(row["pass_offset_s"]))


def s1g_human_start_distance(row: Mapping) -> float:
    """S1g: how far along the aisle from the robot's x the person starts: the trigger distance and what the gait
    drive walks from rest in run_up_s, so the person reaches the trigger distance run_up_s after setting off."""
    return s1g_trigger_distance(row) + _ramp_distance(float(row["human_speed_mps"]), float(row["run_up_s"]))


def s1f_trigger_distance(row: Mapping) -> float:
    """S1f: how far behind the person, along the aisle, the robot's root is when the person starts to turn --
    trigger_lead_s * robot_speed_mps, so that at the task's speed the robot draws level with the person trigger_lead_s
    after the trigger (worked out from the robot_speed_mps column -- not from the speed the robot in fact walks at)."""
    return float(row["trigger_lead_s"]) * float(row["robot_speed_mps"])


def s1f_rack_facing_yaw_deg(site: sites.Site) -> float:
    """S1f: the yaw (degrees, counter-clockwise from +x) of the person standing on the site's line and facing the rack
    face beside it, its back to the robot's lane across the aisle: square to the aisle the robot walks along, toward
    the edge of the lane's floor strip on the person's side. ValueError when the robot's heading does not run along
    the aisle (x) or the person's line is not between the lane and that face."""
    if abs(math.sin(math.radians(site.robot_start.yaw_deg))) > 1e-9:
        raise ValueError(f"{site.scenario_id}: straight 의 출발 요 {site.robot_start.yaw_deg}° 가 통로(x 방향)를 따르지 "
                         "않습니다 — 사람이 볼 랙 면을 알 수 없습니다")
    y0, lane = site.human_y_m[0], site.robot_start.y
    lo, hi = _strip(site.scenario_id, lane, "로봇 길")
    face = hi if y0 > lane else lo
    if not min(lane, face) < y0 < max(lane, face):
        raise ValueError(f"{site.scenario_id}: 사람의 선 y {y0} 가 로봇 길(y {lane})과 그 옆 랙 면(y {face}) 사이에 있지 "
                         "않습니다 — 사람이 볼 랙 면을 알 수 없습니다")
    return math.copysign(90.0, face - y0)


def _turn_away(site: sites.Site, side: str) -> float:
    """The sign of the turn whose half-way heading points the way the person walks (away from where it comes from)."""
    walk_x = 1.0 if side == "west" else -1.0
    yaw0 = math.radians(site.robot_start.yaw_deg)
    ahead = math.cos(yaw0 + math.pi / 2) * walk_x          # a counter-clockwise turn's half-way heading, along the walk
    if abs(ahead) < 1e-9:
        raise ValueError(f"{site.scenario_id}: 출발의 요 {site.robot_start.yaw_deg}° 에서는 반 바퀴의 가운데가 통로를 "
                         "따르지 않아 사람의 반대쪽을 고를 수 없습니다")
    return 1.0 if ahead > 0.0 else -1.0


def row_site(site: sites.Site, row: Mapping) -> sites.Site:
    """The site an episode of this row happens at -- the canon's ``site`` (sites.load_site) moved by the row: S1e's
    lanes and turn radius and the person's spot, S1f's person's spot, S1g's turn direction (away from the person) and
    the line the person walks. The person's region of the result is where the person stands or walks into the
    robot's way (a point, or for S1g a line along the site's region)."""
    kind = site.path["kind"]
    if kind == "u_turn":
        cx, cy = site.path["turn_center_xy_m"]
        radius = float(row["turn_radius_m"])
        heading = math.copysign(1.0, math.cos(math.radians(site.robot_start.yaw_deg)))
        start = sites.Pose2(site.robot_start.x, cy - heading * radius, site.robot_start.yaw_deg)
        goal = sites.Pose2(site.robot_goal.x, cy + heading * radius, site.robot_goal.yaw_deg)
        x, y = _corner_person(site, row)
        return sites.Site(site.scenario_id, start, goal, {"kind": "u_turn", "turn_center_xy_m": (cx, cy),
                                                           "radius_m": radius}, (x, x), (y, y))
    if kind == "straight":
        x = float(row["human_start_x_m"])
        y = site.human_y_m[0]
        return sites.Site(site.scenario_id, site.robot_start, site.robot_goal, dict(site.path), (x, x), (y, y))
    if kind == "turn_then_walk":
        path = dict(site.path)
        path["turn_deg"] = _turn_away(site, row["approach_side"]) * abs(float(site.path["turn_deg"]))
        line = _s1g_line(site, row)
        return sites.Site(site.scenario_id, site.robot_start, site.robot_goal, path, site.human_x_m, (line, line))
    raise ValueError(f"{site.scenario_id}: 조건표가 모르는 길의 종류입니다: {kind!r}")


def _corner_person(site: sites.Site, row: Mapping) -> tuple[float, float]:
    """S1e: corner_distance_m past the row end along the way out, face_distance_m off the rack face between the turn's
    centre and the way out (the edge of the floor strip the canon's way out runs on)."""
    cx, cy = site.path["turn_center_xy_m"]
    out = -math.copysign(1.0, math.cos(math.radians(site.robot_start.yaw_deg)))      # the way out, along x
    lo, hi = _strip(site.scenario_id, site.robot_goal.y, "나오는 길")
    face, toward_lane = (hi, -1.0) if abs(hi - cy) < abs(lo - cy) else (lo, 1.0)
    return cx + out * float(row["corner_distance_m"]), face + toward_lane * float(row["face_distance_m"])


def _s1g_line(site: sites.Site, row: Mapping) -> float:
    return (site.human_y_m[0] + site.human_y_m[1]) / 2.0 + float(row["lateral_offset_m"])


def human_walk(site: sites.Site, row: Mapping) -> tuple[tuple[float, float], tuple[float, float]]:
    """(where the person stands or starts, where it stops) for the canon's ``site`` and a row: S1e stands where it
    starts; S1f walks at entry_angle_deg from the robot's heading toward the robot's lane and stops FAR_FACE_STOP_M
    before the far rack face; S1g walks along its line from s1g_human_start_distance() before the robot's x to
    WALK_PAST_M past it."""
    kind = site.path["kind"]
    if kind == "u_turn":
        spot = _corner_person(site, row)
        return spot, spot
    if kind == "straight":
        yaw0 = math.radians(site.robot_start.yaw_deg)
        if abs(math.sin(yaw0)) > 1e-9:
            raise ValueError(f"{site.scenario_id}: straight 의 출발 요 {site.robot_start.yaw_deg}° 가 통로(x 방향)를 따르지 "
                             "않습니다 — 건너편 랙 면을 알 수 없습니다")
        x0, y0 = float(row["human_start_x_m"]), site.human_y_m[0]
        lane = site.robot_start.y
        lo, hi = _strip(site.scenario_id, lane, "로봇 길")
        toward = math.copysign(1.0, lane - y0)
        stop_y = (lo + FAR_FACE_STOP_M) if toward < 0 else (hi - FAR_FACE_STOP_M)
        theta = math.radians(float(row["entry_angle_deg"]))
        direction = (math.cos(theta) * math.cos(yaw0), toward * math.sin(theta))
        length = (stop_y - y0) / direction[1]
        return (x0, y0), (x0 + length * direction[0], stop_y)
    if kind == "turn_then_walk":
        walk_x = 1.0 if row["approach_side"] == "west" else -1.0
        x_r, line = site.robot_start.x, _s1g_line(site, row)
        return (x_r - walk_x * s1g_human_start_distance(row), line), (x_r + walk_x * WALK_PAST_M, line)
    raise ValueError(f"{site.scenario_id}: 조건표가 모르는 길의 종류입니다: {kind!r}")


# ── the guard ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Obstacles:
    names: tuple[str, ...]
    lo: np.ndarray          # (M, 2)
    hi: np.ndarray          # (M, 2)
    floor_lo: np.ndarray    # (2,) the walls' inner faces
    floor_hi: np.ndarray


def _obstacles(boxes: Sequence[layout.Box]) -> _Obstacles:
    floors = [b for b in boxes if b.name == "floor"]
    if len(floors) != 1:
        raise ValueError(f"레이아웃에 바닥 상자('floor')가 {len(floors)}개 있습니다 (1개여야 함) — 건물의 안쪽 면을 알 수 없습니다")
    f_lo, f_hi = layout.bounds(floors[0])
    z_lo, z_hi = sites.OBSTACLE_Z_M
    names, lows, highs = [], [], []
    for b in boxes:
        lo, hi = layout.bounds(b)
        if b.collider and lo[2] < z_hi and hi[2] > z_lo:
            names.append(b.name)
            lows.append(lo[:2])
            highs.append(hi[:2])
    return _Obstacles(tuple(names), np.array(lows, float).reshape(-1, 2), np.array(highs, float).reshape(-1, 2),
                      np.array(f_lo[:2], float), np.array(f_hi[:2], float))


def _rect_aabb_distance(centre: np.ndarray, axes: np.ndarray, half: np.ndarray,
                        lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """(N, M) distances between N turned rectangles -- centres (N, 2), unit axes (N, 2, 2) as rows, half sizes
    (N, 2), a zero half size making a segment or a point -- and M axis-aligned rectangles lo/hi (M, 2); 0 where
    they meet. Two convex shapes that do not meet are nearest at a corner of one, so the distance is the least of
    each one's corners to the other."""
    signs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)
    corners = centre[:, None, :] + np.einsum("ck,nk,nkd->ncd", signs, half, axes)          # (N, 4, 2)
    gap = np.maximum(np.maximum(lo[None, None] - corners[:, :, None], corners[:, :, None] - hi[None, None]), 0.0)
    to_box = np.hypot(gap[..., 0], gap[..., 1]).min(axis=1)                                   # (N, M)
    box_corners = np.stack([np.stack([lo[:, 0], lo[:, 1]], -1), np.stack([lo[:, 0], hi[:, 1]], -1),
                            np.stack([hi[:, 0], lo[:, 1]], -1), np.stack([hi[:, 0], hi[:, 1]], -1)], 1)  # (M, 4, 2)
    rel = box_corners[None] - centre[:, None, None]                                           # (N, M, 4, 2)
    along = np.abs(np.einsum("nmcd,nkd->nmck", rel, axes)) - half[:, None, None]
    out = np.maximum(along, 0.0)
    to_rect = np.hypot(out[..., 0], out[..., 1]).min(axis=2)                                  # (N, M)
    # separating axes: x, y and the rectangle's two
    sep = ((corners[:, :, 0].max(1)[:, None] < lo[None, :, 0]) | (corners[:, :, 0].min(1)[:, None] > hi[None, :, 0])
           | (corners[:, :, 1].max(1)[:, None] < lo[None, :, 1]) | (corners[:, :, 1].min(1)[:, None] > hi[None, :, 1]))
    mid, ext = (lo + hi) / 2.0, (hi - lo) / 2.0
    between = mid[None] - centre[:, None]                                                     # (N, M, 2)
    for k in range(2):
        u = axes[:, k, :]                                                                      # (N, 2)
        reach = half[:, k][:, None] + np.abs(u) @ ext.T                                        # (N, M)
        sep |= np.abs(np.einsum("nmd,nd->nm", between, u)) > reach
    return np.where(sep, np.minimum(to_box, to_rect), 0.0)


def _close(obstacles: _Obstacles, centre: np.ndarray, axes: np.ndarray, half: np.ndarray,
           limit: float) -> tuple[np.ndarray, np.ndarray]:
    """(the distances (N, K) of _rect_aabb_distance, the K colliders' indices) for the colliders the rectangles may
    come nearer than ``limit`` to. A collider is left out when even its distance from a rectangle's centre less the
    rectangle's circumradius -- never more than the true distance -- is ``limit`` or more for every rectangle, so the
    distances below ``limit`` are all there."""
    radius = np.hypot(half[:, 0], half[:, 1])
    lo, hi = centre.min(axis=0) - limit - radius.max(), centre.max(axis=0) + limit + radius.max()
    near = np.flatnonzero(((obstacles.lo <= hi) & (obstacles.hi >= lo)).all(axis=1))
    gap = np.maximum(np.maximum(obstacles.lo[near][None] - centre[:, None], centre[:, None] - obstacles.hi[near][None]),
                     0.0)
    bound = np.hypot(gap[..., 0], gap[..., 1]) - radius[:, None]
    keep = near[bound.min(axis=0) < limit]
    return _rect_aabb_distance(centre, axes, half, obstacles.lo[keep], obstacles.hi[keep]), keep


def _yaw_axes(yaw: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], 1)                          # (N, 2, 2)


@functools.lru_cache(maxsize=1)
def _joints() -> dict:
    return kinematics.load_joints()


@functools.lru_cache(maxsize=4)
def _hold_link_pose(link: str, hold: tuple) -> tuple[np.ndarray, np.ndarray]:
    joints = _joints()
    return kinematics.link_poses(joints, dict(hold), frame=joints["root_link"])[link]


def _box_footprint(entry: Mapping, row: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The held box seen from above in the robot's root frame (its root at the origin facing +x): (centre (2,),
    axes (2, 2), half sizes (2,)) -- carry.box_obb at the row's size on the link the carry block names, at the hold
    pose. None without a carry block."""
    held = carry_canon.load_carry(entry)
    if held is None:
        return None
    hold = carry_canon.arm_hold_joints(held, _joints()["joints"])
    link_pos, link_quat = _hold_link_pose(held.link, tuple(sorted(hold.items())))
    obb = carry_canon.box_obb(link_pos, link_quat, held, box_size(row))
    rot = kinematics.quat_to_matrix(obb.rot_xyzw)
    if abs(rot[2, 2]) < 1.0 - 1e-6:
        raise ValueError(f"{entry.get('scenario_id')}: 든 상자가 잡는 자세에서 수평이 아닙니다 — 위에서 본 모양이 직사각형이 아닙니다")
    return obb.center[:2].copy(), np.stack([rot[:2, 0], rot[:2, 1]]), obb.half[:2].copy()


def _places(points: np.ndarray) -> list[str]:
    return [f"({x:.3f}, {y:.3f})" for x, y in points]


def _lines(sid: str, part: str, what: str, distances: np.ndarray, near: np.ndarray, obstacles: _Obstacles,
           places: Sequence[str], clearance: float) -> list[str]:
    """One line per collider closer than ``clearance`` to the part, at the place (a row of ``distances``) nearest it,
    nearest first. ``what`` carries its particle."""
    found = []
    for j in np.flatnonzero(distances.min(axis=0) < clearance - 1e-9):
        i = int(np.argmin(distances[:, j]))
        found.append((float(distances[i, j]), obstacles.names[near[j]], places[i]))
    return [f"{sid} {part}: {place} 에서 {what} 충돌체 {name} 에서 {d:.3f} m — {clearance:g} m 안입니다"
            for d, name, place in sorted(found, key=lambda item: item[:2])]


def _inside(sid: str, part: str, what: str, points: np.ndarray, obstacles: _Obstacles, inset: float) -> list[str]:
    lo, hi = obstacles.floor_lo + inset, obstacles.floor_hi - inset
    out = ((points < lo - 1e-9) | (points > hi + 1e-9)).any(axis=1)
    if not out.any():
        return []
    x, y = points[np.argmax(out)]
    return [f"{sid} {part}: {what} 점 {int(out.sum())}개가 건물 벽에서 {inset:g} m 안쪽이 아닙니다 — 처음 ({x:.3f}, {y:.3f}) · "
            f"바닥 x {obstacles.floor_lo[0]:g}..{obstacles.floor_hi[0]:g} · y {obstacles.floor_lo[1]:g}..{obstacles.floor_hi[1]:g}"]


def _row_problems(entry: Mapping, row: Mapping, obstacles: _Obstacles) -> list[str]:
    sid = entry["scenario_id"]
    canon_site = sites.load_site(entry)
    out: list[str] = []
    if canon_site.path["kind"] == "turn_then_walk" and s1g_trigger_distance(row) <= 0.0:
        out.append(f"{sid} 사람 길: 과제가 오는 사람 거리 {s1g_trigger_distance(row):.3f} m 가 양수가 아닙니다")
    site = row_site(canon_site, row)
    path = sites.nominal_path(site)
    xy, yaw = path[:, :2], path[:, 2]
    n = len(path)
    # the body: a disc about the root along the path
    body, near = _close(obstacles, xy, _yaw_axes(yaw), np.zeros((n, 2)), ROBOT_CLEARANCE_M + ROBOT_HALF_WIDTH_M)
    out += _lines(sid, "로봇 길", f"몸(반폭 {ROBOT_HALF_WIDTH_M:g} m)이", body - ROBOT_HALF_WIDTH_M, near, obstacles,
                  _places(xy), ROBOT_CLEARANCE_M)
    rim = [xy + ROBOT_HALF_WIDTH_M * np.array(d) for d in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    out += _inside(sid, "로봇 길", "몸의", np.concatenate(rim), obstacles, ROBOT_CLEARANCE_M)
    # the held box, turned with the robot
    box = _box_footprint(entry, row)
    reach = ROBOT_HALF_WIDTH_M
    if box is not None:
        b_centre, b_axes, b_half = box
        turn = _yaw_axes(yaw)                                                                 # rows: heading, left
        centres = xy + np.einsum("d,ndk->nk", b_centre, turn)
        axes = np.einsum("ad,ndk->nak", b_axes, turn)
        corners = centres[:, None] + np.einsum("ck,k,nkd->ncd", np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float),
                                                b_half, axes)
        held, near_b = _close(obstacles, centres, axes, np.tile(b_half, (n, 1)), ROBOT_CLEARANCE_M)
        out += _lines(sid, "든 상자", "상자가", held, near_b, obstacles, _places(xy), ROBOT_CLEARANCE_M)
        out += _inside(sid, "든 상자", "상자 모서리", corners.reshape(-1, 2), obstacles, ROBOT_CLEARANCE_M)
        signs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)
        local = b_centre + (signs * b_half) @ b_axes
        reach = max(reach, float(np.hypot(local[:, 0], local[:, 1]).max()))
    # a turn on the spot sweeps a circle
    if site.path["kind"] == "turn_then_walk":
        spot = np.array([[site.robot_start.x, site.robot_start.y]])
        swept, near_t = _close(obstacles, spot, _yaw_axes(np.zeros(1)), np.zeros((1, 2)), ROBOT_CLEARANCE_M + reach)
        swept = swept - reach
        out += _lines(sid, "돌기 원", f"돌기 원(반지름 {reach:.3f} m)이", swept, near_t, obstacles, _places(spot),
                      ROBOT_CLEARANCE_M)
    # the person: where it stands and the way it walks (a segment, a point when it stands still). S1g's person is
    # checked coming from every side: a row whose walk would be refused from the other side is refused too, so what
    # the table draws for the timing does not depend on the side drawn (one wall is nearer the robot than the other).
    walks = [("사람 길", row)]
    if canon_site.path["kind"] == "turn_then_walk":
        walks += [(f"사람 길 (반대쪽 {side} 에서 오면)", {**row, "approach_side": side})
                  for side in APPROACH_SIDES if side != row["approach_side"]]
    for part, walked in walks:
        start, stop = (np.array(p, float) for p in human_walk(canon_site, walked))
        step = stop - start
        length = float(np.linalg.norm(step))
        heading = math.atan2(step[1], step[0]) if length > 0.0 else 0.0
        person, near_h = _close(obstacles, np.array([(start + stop) / 2.0]), _yaw_axes(np.array([heading])),
                                np.array([[length / 2.0, 0.0]]), HUMAN_CLEARANCE_M)
        walk = (f"({start[0]:.3f}, {start[1]:.3f}) → ({stop[0]:.3f}, {stop[1]:.3f})" if length > 0.0
                else _places([start])[0])
        out += _lines(sid, part, "사람이", person, near_h, obstacles, [walk], HUMAN_CLEARANCE_M)
        out += _inside(sid, part, "선 자리와 멈추는 자리의", np.stack([start, stop]), obstacles, WALL_INSET_M)
    return out


def row_problems(entry: Mapping, row: Mapping, boxes: Sequence[layout.Box]) -> list[str]:
    """Everything the guard refuses in a row of a scenario's conditions on a layout, as "<scenario> <part>: <why>"
    lines; [] when the row can be run. Parts: 로봇 길 (the body along the path), 든 상자 (the held box), 돌기 원 (the
    circle a turn on the spot sweeps), 사람 길 (the person's place and walk; for S1g also "사람 길 (반대쪽 <side> 에서
    오면)", the same walk from the other side). See the module docstring."""
    return _row_problems(entry, row, _obstacles(boxes))


def cell_value(axis: GridAxis, k: int, cells: int) -> float:
    """The middle of cell ``k`` of ``cells`` equal cells of the axis's range."""
    return axis.lo + (axis.hi - axis.lo) * (k + 0.5) / cells


def _check_index(spec: ConditionSpec, index: object) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < spec.total:
        raise ValueError(f"{spec.scenario_id} 조건표에 행 {index!r} 이 없습니다 — 행은 0 ~ {spec.total - 1}")


def grid_cell(spec: ConditionSpec, index: int) -> tuple[int, int] | None:
    """(robot axis cell, human axis cell) of row ``index``; None for a random row. ValueError for a row the table
    does not have."""
    _check_index(spec, index)
    if spec.grid is None or index < spec.rows:
        return None
    return (index - spec.rows) % spec.grid.cells, (index - spec.rows) // spec.grid.cells


def row_design(entry: Mapping, index: int) -> tuple[str, tuple[int, int] | None]:
    """("random", None) or ("grid", (robot cell, human cell)) for row ``index`` of the scenario's table."""
    cell = grid_cell(load_conditions(entry), index)
    return ("random", None) if cell is None else ("grid", cell)


def _grid_row(spec: ConditionSpec, cell: tuple[int, int]) -> dict:
    """The row of a grid cell: grid.fixed, the two axes at the middle of their cells, and conditions.fixed -- in the
    keys and order of a drawn row."""
    return _cell_row(spec.grid, spec.columns, spec.fixed, cell)


def _cell_row(grid: GridSpec, columns: Mapping[str, Column], conditions_fixed: Mapping[str, float],
              cell: tuple[int, int]) -> dict:
    """_grid_row on the pieces a load has before the ConditionSpec exists (the load checks the worked-out column
    with this)."""
    row: dict = dict(grid.fixed)
    row[grid.robot_axis.quantity] = cell_value(grid.robot_axis, cell[0], grid.cells)
    value = cell_value(grid.human_axis, cell[1], grid.cells)
    if grid.human_axis.quantity in columns:
        row[grid.human_axis.quantity] = value
    else:
        column, worked_out = GRID_QUANTITIES[grid.human_axis.quantity]
        row[column] = worked_out(value, row)
    row.update(conditions_fixed)
    return {name: row[name] for name in [*columns, *conditions_fixed]}


def _draw(entry: Mapping, spec: ConditionSpec, index: int, obstacles: _Obstacles) -> tuple[dict, int | str]:
    """(the row, its attempt) -- a grid cell's attempt is "grid"."""
    _check_index(spec, index)
    if index >= spec.rows:
        cell = grid_cell(spec, index)
        row = _grid_row(spec, cell)
        problems = _row_problems(entry, row, obstacles)
        if problems:
            raise ValueError(f"{spec.scenario_id} 조건표 행 {index} · 격자 칸 ({cell[0]}, {cell[1]}): "
                             f"가드에 걸렸습니다 — {problems[0]}")
        return row, "grid"
    problems: list[str] = []
    for attempt in range(MAX_ATTEMPTS):
        rng = np.random.default_rng(row_seed(spec.scenario_id, index, attempt))
        row = {name: col.draw(rng) for name, col in spec.columns.items()}
        row.update(spec.fixed)
        problems = _row_problems(entry, row, obstacles)
        if not problems:
            return row, attempt
    raise ValueError(f"{spec.scenario_id} 조건표 행 {index}: 시도 {MAX_ATTEMPTS}번이 모두 가드에 걸렸습니다 — 마지막 시도: "
                     f"{problems[0]}")


def draw_row(entry: Mapping, index: int, boxes: Sequence[layout.Box]) -> dict[str, float | str]:
    """Row ``index`` of the scenario's conditions table: the columns of its first attempt the guard lets through, and
    the fixed values; a grid cell's row for an index past the random rows (see the module docstring). ValueError when
    0 <= index < total does not hold, no attempt passes or the guard refuses a grid cell."""
    return _draw(entry, load_conditions(entry), index, _obstacles(boxes))[0]


def table(entry: Mapping, boxes: Sequence[layout.Box]) -> list[dict]:
    """The random rows of the scenario's conditions table, in order (draw_row for 0 .. rows - 1)."""
    spec, obstacles = load_conditions(entry), _obstacles(boxes)
    return [_draw(entry, spec, index, obstacles)[0] for index in range(spec.rows)]


def grid_table(entry: Mapping, boxes: Sequence[layout.Box]) -> list[dict]:
    """The grid's rows in row order (draw_row for rows .. total - 1); [] without a grid."""
    spec, obstacles = load_conditions(entry), _obstacles(boxes)
    return [_draw(entry, spec, index, obstacles)[0] for index in range(spec.rows, spec.total)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", required=True, help="S1e · S1f · S1g")
    ap.add_argument("--out", type=Path, required=True,
                    help="the table as CSV: row, attempt (\"grid\" for a grid cell), the columns, the fixed")
    a = ap.parse_args(argv)
    try:
        entry = scenario_entry(a.scenario)
    except ValueError as error:
        ap.error(str(error))
    spec = load_conditions(entry)
    obstacles = _obstacles(layout.build_layout(layout.load_spec(layout.SCENE_SPEC)))
    names = [*spec.columns, *spec.fixed]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "attempt", *names])
        for index in range(spec.total):
            row, attempt = _draw(entry, spec, index, obstacles)
            writer.writerow([index, attempt, *(repr(row[n]) if isinstance(row[n], float) else row[n] for n in names)])
    print(f"{a.scenario}: {spec.total} 행 → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
