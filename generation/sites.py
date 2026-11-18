"""Where each new scenario happens: the robot's start and goal, the path it walks and the region
the person stands in.

The place is meta/scenarios.json's scene.site (S1e, S1f, S1g); this module reads it, builds the
nominal path from it and checks both against the warehouse layout. Coordinates are the scene
layout's, which are the environment's; yaw is in degrees counter-clockwise from +x. Pure numpy:
the stage builder, the camera checks and the environment read the same place through this one
module, and a reader of a release can recompute the path from the canon.

A path is one of three kinds, each described by what it needs beyond the two end poses:

* ``straight`` -- from the start to the goal along the start's heading.
* ``u_turn`` -- along the heading (which must be along x) to ``turn_center_xy_m``'s x, half a
  circle of radius ``radius_m`` about that centre turning left, then along the opposite heading
  to the goal.
* ``turn_then_walk`` -- ``turn_deg`` (at most half a circle either way) on the spot, then
  ``walk_m`` along the new heading.

The yaw column of the nominal path is continuous (never wrapped): it starts at the start's yaw and
a left half circle adds pi to it, so a comparison with an end pose is made modulo 2 pi.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from generation.scene import layout

SITE_KEYS = ("robot_start", "robot_goal", "path", "human_region")
PATH_KINDS = ("u_turn", "straight", "turn_then_walk")
# What each kind needs besides its "kind".
PATH_PARAMETERS = {
    "u_turn": ("turn_center_xy_m", "radius_m"),
    "straight": (),
    "turn_then_walk": ("turn_deg", "walk_m"),
}

# A path point nearer than this to anything the robot could hit is a problem (project's own value).
CLEARANCE_M = 0.45
# What can be hit by a walking robot or a standing person: colliders reaching into this band above the floor.
OBSTACLE_Z_M = (0.0, 2.0)
# An in-place turn is sampled at most this many degrees per row.
TURN_ROW_DEG = 5.0
# How far a goal may be from where its path ends.
END_TOLERANCE = 1e-6
# Floating-point slack when counting how many steps fit in a length.
COUNT_SLACK = 1e-9


@dataclass(frozen=True)
class Pose2:
    x: float
    y: float
    yaw_deg: float


@dataclass(frozen=True)
class Site:
    scenario_id: str
    robot_start: Pose2
    robot_goal: Pose2
    path: Mapping[str, object]
    human_x_m: tuple[float, float]
    human_y_m: tuple[float, float]


def _number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _numbers(where: str, value: object, count: int) -> tuple[float, ...]:
    if (not isinstance(value, (list, tuple)) or len(value) != count
            or not all(_number(v) for v in value)):
        raise ValueError(f"{where} 이 유한한 수 {count}개의 목록이 아닙니다: {value!r}")
    return tuple(float(v) for v in value)


def _scalar(where: str, value: object) -> float:
    if not _number(value):
        raise ValueError(f"{where} 이 유한한 수가 아닙니다: {value!r}")
    return float(value)


def _mapping(where: str, value: object) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} 이 객체가 아닙니다: {value!r}")
    return value


def _member(where: str, node: Mapping, key: str) -> object:
    if key not in node:
        raise ValueError(f"{where} 에 {key} 가 없습니다")
    return node[key]


def _pose(sid: str, name: str, raw: object) -> Pose2:
    where = f"{sid}: scene.site.{name}"
    node = _mapping(where, raw)
    x, y = _numbers(f"{where}.xy_m", _member(where, node, "xy_m"), 2)
    return Pose2(x, y, _scalar(f"{where}.yaw_deg", _member(where, node, "yaw_deg")))


def _path(sid: str, raw: object) -> dict[str, object]:
    where = f"{sid}: scene.site.path"
    node = _mapping(where, raw)
    kind = node.get("kind")
    if kind not in PATH_KINDS:
        raise ValueError(f"{where}.kind 가 {list(PATH_KINDS)} 중 하나가 아닙니다: {kind!r}")
    out: dict[str, object] = {"kind": kind}
    for name in PATH_PARAMETERS[kind]:
        value = _member(f"{where} ({kind})", node, name)
        if name == "turn_center_xy_m":
            out[name] = _numbers(f"{where}.{name}", value, 2)
            continue
        number = _scalar(f"{where}.{name}", value)
        if name == "turn_deg" and abs(number) > 180.0:
            raise ValueError(f"{where}.turn_deg 이 반 바퀴(±180)를 넘습니다: {number!r}")
        if name in ("radius_m", "walk_m") and number <= 0.0:
            raise ValueError(f"{where}.{name} 이 양수가 아닙니다: {number!r}")
        out[name] = number
    return out


def _region(sid: str, raw: object) -> tuple[tuple[float, float], tuple[float, float]]:
    where = f"{sid}: scene.site.human_region"
    node = _mapping(where, raw)
    axes = []
    for key in ("x_m", "y_m"):
        lo, hi = _numbers(f"{where}.{key}", _member(where, node, key), 2)
        if lo > hi:
            raise ValueError(f"{where}.{key} 의 아래 끝이 위 끝보다 큽니다: [{lo}, {hi}]")
        axes.append((lo, hi))
    return axes[0], axes[1]


def load_site(scenario: Mapping) -> Site:
    """The Site of a meta/scenarios.json entry; ValueError (in Korean) names what is missing, is
    not a number, is an unknown path kind, or leaves the path short of its goal."""
    sid = scenario.get("scenario_id") if isinstance(scenario, Mapping) else None
    if not isinstance(sid, str) or not sid:
        raise ValueError("시나리오에 scenario_id 가 없습니다")
    scene = scenario.get("scene")
    site = scene.get("site") if isinstance(scene, Mapping) else None
    if not isinstance(site, Mapping):
        raise ValueError(f"{sid}: scene.site 가 없습니다 — 이 시나리오는 자리를 정본에 선언하지 않았습니다")
    missing = [key for key in SITE_KEYS if key not in site]
    if missing:
        raise ValueError(f"{sid}: scene.site 에 {missing} 가 없습니다 — {list(SITE_KEYS)}")
    x_m, y_m = _region(sid, site["human_region"])
    result = Site(sid, _pose(sid, "robot_start", site["robot_start"]), _pose(sid, "robot_goal", site["robot_goal"]),
                  _path(sid, site["path"]), x_m, y_m)
    _plan(result)                      # refuses a path that does not lead from the start to the goal
    return result


def _same_angle(a_deg: float, b_deg: float) -> bool:
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0) <= END_TOLERANCE


def _check_goal(site: Site, x: float, y: float, yaw_deg: float, what: str) -> None:
    goal = site.robot_goal
    if math.hypot(goal.x - x, goal.y - y) > END_TOLERANCE:
        raise ValueError(f"{site.scenario_id}: 목표 ({goal.x}, {goal.y}) 가 {what}의 끝 ({x:.6g}, {y:.6g}) 과 어긋납니다")
    if not _same_angle(goal.yaw_deg, yaw_deg):
        raise ValueError(f"{site.scenario_id}: 목표의 요 {goal.yaw_deg}° 가 {what}이 끝나는 요 {yaw_deg:.6g}° 와 어긋납니다")


def _plan(site: Site) -> list[tuple]:
    """The path as primitives, each starting where the one before ends:
    ("line", (x0, y0), (x1, y1), yaw) -- ("arc", centre, radius, theta0, yaw0) a left half circle --
    ("spin", (x, y), yaw0, turn), all angles in radians and yaw continuous. Refuses a site whose
    goal the path does not reach."""
    start, path = site.robot_start, site.path
    kind, yaw0 = path["kind"], math.radians(start.yaw_deg)
    p0 = (start.x, start.y)
    if kind == "straight":
        goal = (site.robot_goal.x, site.robot_goal.y)
        dx, dy = goal[0] - p0[0], goal[1] - p0[1]
        along, across = dx * math.cos(yaw0) + dy * math.sin(yaw0), -dx * math.sin(yaw0) + dy * math.cos(yaw0)
        if abs(across) > END_TOLERANCE or along <= END_TOLERANCE:
            raise ValueError(f"{site.scenario_id}: straight 의 목표가 출발에서 요 방향으로 곧게 나아간 직선 위에 있지 않습니다 "
                             f"(앞으로 {along:.6g} m · 옆으로 {across:.6g} m)")
        _check_goal(site, p0[0] + along * math.cos(yaw0), p0[1] + along * math.sin(yaw0), start.yaw_deg, "straight")
        return [("line", p0, goal, yaw0)]
    if kind == "u_turn":
        cx, cy = path["turn_center_xy_m"]
        radius = path["radius_m"]
        if abs(math.sin(yaw0)) > END_TOLERANCE:
            raise ValueError(f"{site.scenario_id}: u_turn 은 출발의 요가 x 방향(0° 나 180°)이어야 합니다: {start.yaw_deg}°")
        heading = math.copysign(1.0, math.cos(yaw0))              # +1 along +x, -1 along -x
        if abs(start.y - (cy - heading * radius)) > END_TOLERANCE:
            raise ValueError(f"{site.scenario_id}: u_turn 의 출발 y {start.y} 가 가운데 y {cy} 에서 반지름 {radius} 만큼 "
                             f"(왼쪽으로 돌 수 있는 쪽) 떨어져 있지 않습니다 — {cy - heading * radius} 여야 합니다")
        if (cx - start.x) * heading < -END_TOLERANCE:
            raise ValueError(f"{site.scenario_id}: u_turn 의 출발 x {start.x} 가 가운데 x {cx} 를 이미 지났습니다")
        theta0 = yaw0 - math.pi / 2
        out_y = cy + heading * radius                                # where the half circle ends
        goal = (site.robot_goal.x, site.robot_goal.y)
        if abs(goal[1] - out_y) > END_TOLERANCE:
            raise ValueError(f"{site.scenario_id}: u_turn 의 목표 y {goal[1]} 가 반원이 끝나는 y {out_y} 와 어긋납니다")
        if (goal[0] - cx) * (-heading) < -END_TOLERANCE:
            raise ValueError(f"{site.scenario_id}: u_turn 의 목표 x {goal[0]} 가 반원이 끝난 쪽(가운데 x {cx} 의 반대 방향)에 있지 않습니다")
        _check_goal(site, goal[0], out_y, start.yaw_deg + 180.0, "u_turn")
        return [("line", p0, (cx, start.y), yaw0), ("arc", (cx, cy), radius, theta0, yaw0),
                ("line", (cx, out_y), goal, yaw0 + math.pi)]
    turn = math.radians(path["turn_deg"])
    yaw1 = yaw0 + turn
    walk = path["walk_m"]
    end = (p0[0] + walk * math.cos(yaw1), p0[1] + walk * math.sin(yaw1))
    _check_goal(site, end[0], end[1], start.yaw_deg + path["turn_deg"], "turn_then_walk")
    return [("spin", p0, yaw0, turn), ("line", p0, end, yaw1)]


def _count(length: float, step: float) -> int:
    """Rows to take ``length`` in steps of at most ``step``."""
    return max(1, math.ceil(length / step - COUNT_SLACK))


def nominal_path(site: Site, step_m: float = 0.05) -> np.ndarray:
    """(N, 3) rows of x, y, yaw (rad): the first is the start, the last the goal. Lines and the
    half circle are sampled at most ``step_m`` apart; an in-place turn takes a row per
    ``TURN_ROW_DEG`` degrees. The yaw column is continuous, not wrapped."""
    if not _number(step_m) or step_m <= 0:
        raise ValueError(f"step_m 이 양수인 유한한 수가 아닙니다: {step_m!r}")
    rows = [[site.robot_start.x, site.robot_start.y, math.radians(site.robot_start.yaw_deg)]]
    for prim in _plan(site):
        if prim[0] == "line":
            _, (x0, y0), (x1, y1), yaw = prim
            length = math.hypot(x1 - x0, y1 - y0)
            if length < COUNT_SLACK:
                continue
            n = _count(length, step_m)
            for k in range(1, n + 1):
                t = k / n
                rows.append([x0 * (1 - t) + x1 * t, y0 * (1 - t) + y1 * t, yaw])
        elif prim[0] == "arc":
            _, (cx, cy), radius, theta0, yaw0 = prim
            n = _count(math.pi * radius, step_m)
            for k in range(1, n + 1):
                sweep = math.pi * k / n
                theta = theta0 + sweep
                rows.append([cx + radius * math.cos(theta), cy + radius * math.sin(theta), yaw0 + sweep])
        else:
            _, (x, y), yaw0, turn = prim
            n = 0 if turn == 0.0 else _count(abs(math.degrees(turn)), TURN_ROW_DEG)
            for k in range(1, n + 1):
                rows.append([x, y, yaw0 + turn * k / n])
    return np.asarray(rows, dtype=float)


def human_region_points(site: Site, n: int = 3) -> np.ndarray:
    """(k, 2) points of the person's region: an n x n grid with its corners and, for odd n, its
    centre; an axis of zero width is a single line."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError(f"격자의 점 수 n 이 2 이상의 정수가 아닙니다: {n!r}")

    def axis(lo: float, hi: float) -> np.ndarray:
        return np.linspace(lo, hi, n) if hi > lo else np.array([lo])

    xs, ys = axis(*site.human_x_m), axis(*site.human_y_m)
    return np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)


def _distances(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """(N, M) distances from points to axis-aligned rectangles (0 inside or on the edge)."""
    gap = np.maximum(np.maximum(lo[None, :, :] - points[:, None, :], points[:, None, :] - hi[None, :, :]), 0.0)
    return np.hypot(gap[..., 0], gap[..., 1])


def _floor(boxes: Sequence[layout.Box]) -> tuple[float, float, float, float]:
    floors = [b for b in boxes if b.name == "floor"]
    if len(floors) != 1:
        raise ValueError(f"레이아웃에 바닥 상자('floor')가 {len(floors)}개 있습니다 (1개여야 함) — 건물의 안쪽 면을 알 수 없습니다")
    lo, hi = layout.bounds(floors[0])
    return lo[0], hi[0], lo[1], hi[1]


def site_problems(site: Site, boxes: Sequence[layout.Box], clearance_m: float = CLEARANCE_M) -> list[str]:
    """Everything wrong with a site on a layout, as "<scenario> site.<place>: <why>" lines; [] when
    usable. The path must stay inside the building and ``clearance_m`` away from the footprint of
    every collider that reaches into the walker's height band, one line per collider with the path
    point nearest it. The person's region only has to stay inside the building and out of colliders:
    people stand against a rack face on purpose."""
    if not _number(clearance_m) or clearance_m < 0:
        raise ValueError(f"clearance_m 이 유한한 음이 아닌 수가 아닙니다: {clearance_m!r}")
    x0, x1, y0, y1 = _floor(boxes)
    z_lo, z_hi = OBSTACLE_Z_M
    solid = []
    for b in boxes:
        lo, hi = layout.bounds(b)
        if b.collider and lo[2] < z_hi and hi[2] > z_lo:
            solid.append((b.name, lo, hi))
    lows = np.array([[lo[0], lo[1]] for _, lo, _ in solid]).reshape(-1, 2)
    highs = np.array([[hi[0], hi[1]] for _, _, hi in solid]).reshape(-1, 2)
    out: list[str] = []

    def outside(points: np.ndarray) -> np.ndarray:
        eps = 1e-9
        return ((points[:, 0] < x0 - eps) | (points[:, 0] > x1 + eps)
                | (points[:, 1] < y0 - eps) | (points[:, 1] > y1 + eps))

    def leaving(place: str, points: np.ndarray) -> None:
        gone = outside(points)
        if gone.any():
            x, y = points[np.argmax(gone)]
            out.append(f"{site.scenario_id} site.{place}: 건물 밖의 점 {int(gone.sum())}개 — 처음 ({x:.3f}, {y:.3f}) 이 "
                       f"바닥 x {x0:g}..{x1:g} · y {y0:g}..{y1:g} 밖입니다")

    path = nominal_path(site)[:, :2]
    leaving("path", path)
    leaving("human_region", human_region_points(site))
    if not solid:
        return out

    near = _distances(path, lows, highs)
    found = []
    for j in np.flatnonzero(near.min(axis=0) < clearance_m):
        i = int(np.argmin(near[:, j]))
        found.append((float(near[i, j]), solid[j][0], path[i]))
    for distance, name, (x, y) in sorted(found, key=lambda item: item[:2]):
        out.append(f"{site.scenario_id} site.path: ({x:.3f}, {y:.3f}) 이 충돌체 {name} 에서 {distance:.3f} m — "
                   f"clearance {clearance_m:g} m 안입니다")
    # The region is a rectangle (a line when an axis has no width): it meets a footprint when the two
    # share a point, edges included, so a collider thinner than the grid spacing is found too.
    region_lo = np.array([site.human_x_m[0], site.human_y_m[0]])
    region_hi = np.array([site.human_x_m[1], site.human_y_m[1]])
    for j in np.flatnonzero(((lows <= region_hi) & (highs >= region_lo)).all(axis=1)):
        x, y = np.clip((lows[j] + highs[j]) / 2, region_lo, region_hi)
        out.append(f"{site.scenario_id} site.human_region: ({x:.3f}, {y:.3f}) 이 충돌체 {solid[j][0]} 의 발자국 안입니다")
    return out
