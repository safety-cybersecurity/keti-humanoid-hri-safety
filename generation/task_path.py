"""The robot's task: follow a row's path to its goal, and notice when it has arrived.

A pure-pursuit follower over ``sites.nominal_path``: it aims at the path point ``LOOKAHEAD_M`` ahead
of the nearest one, turns at omega = 2 v sin(alpha) / L (clipped to the walking policy's yaw-rate
range), and slows in proportion to the remaining length in the last ``SLOWDOWN_M``. The in-place
turn of a ``turn_then_walk`` path (S1g) is a phase of its own: wait for the task, turn on the spot
at the row's turn rate, walk once the yaw error is within ``TURN_DONE_DEG``. Pure numpy: the
environment's command term calls ``command`` once per policy step, ``CompletionTracker.update``
with the robot's pose and speed and then ``StageTracker.update``.

The task's stages (``STAGE_NAMES``, ``stage_plan``) are self-set: a U-turn path (S1e) is the way in
to the row end, the half circle about it and the way out; a straight path (S1f) is one walk; a turn
and a walk (S1g) is the turn on the spot and the walk after it. ``StageTracker`` keeps the step each
stage ended at: a stage marked by a length along the path ends at the first step the follower's
nearest path point is the sample where that length ends, the turn on the spot at the step the
follower leaves it, and every stage still open at the step the task is done
(``CompletionTracker.step``) -- a completion before the task came closes nothing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from generation import sites

LOOKAHEAD_M = 0.5            # distance ahead of the nearest path point that is aimed at
SLOWDOWN_M = 0.8             # the last stretch in which the speed falls with the remaining length
MAX_FORWARD_MPS = 1.0        # the policy's command range
MAX_YAW_RATE = 1.0
TURN_DONE_DEG = 5.0          # an in-place turn ends when the yaw error is within this
DONE_REMAINING_M = 0.05      # the follower stops commanding when this near the end of the path
PATH_STEP_M = 0.05
PHASES = ("wait", "turn", "walk", "done")
STAGE_NAMES = {"u_turn": ("enter", "turn", "exit"), "straight": ("walk",), "turn_then_walk": ("turn", "walk")}


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PathFollower:
    def __init__(self, site: sites.Site, row: Mapping, *, dt: float) -> None:
        if not dt > 0 or not math.isfinite(dt):
            raise ValueError(f"dt 가 양수인 유한한 수가 아닙니다: {dt!r}")
        self.dt = dt
        self.started_step = -1
        self.phase = "wait"
        self.progress = 0.0
        # The arc length of the nearest path point once the walk is followed (StageTracker reads it).
        self.travelled_m = 0.0
        full = sites.nominal_path(site, PATH_STEP_M)
        self._turn = 0.0
        if site.path["kind"] == "turn_then_walk":
            self._turn = math.radians(float(site.path["turn_deg"]))
            self._turn_rate = float(row["turn_rate_rad_s"])
            self._speed = float(row["walk_speed_mps"])
            self._target_yaw = math.radians(site.robot_start.yaw_deg) + self._turn
            at_start = np.hypot(full[:, 0] - full[0, 0], full[:, 1] - full[0, 1]) < 1e-9
            self._path = full[int(np.flatnonzero(at_start)[-1]):]          # the walk after the spin
        else:
            self._speed = float(row["robot_speed_mps"])
            self._path = full
        if not 0 < self._speed <= MAX_FORWARD_MPS:
            raise ValueError(f"걷는 속도가 정책의 범위(0, {MAX_FORWARD_MPS}] 밖입니다: {self._speed!r}")
        xy = self._path[:, :2]
        self._s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        self._total = float(self._s[-1])
        self._end_heading = float(self._path[-1, 2])
        self._near = 0

    def start(self, step: int) -> None:
        if self.started_step < 0:
            self.started_step = int(step)
            self.phase = "turn" if self._turn != 0.0 else "walk"

    def _remaining_turn(self, yaw: float) -> float:
        """Radians still to turn in the turn's direction, in [0, 2 pi)."""
        sign = math.copysign(1.0, self._turn)
        return ((self._target_yaw - yaw) * sign) % (2.0 * math.pi)

    def command(self, t: float, xy: np.ndarray, yaw: float) -> tuple[float, float]:
        if self.phase == "wait" or self.phase == "done":
            return 0.0, 0.0
        if self.phase == "turn":
            left = self._remaining_turn(yaw)
            if left <= math.radians(TURN_DONE_DEG) or left > 2.0 * math.pi - math.radians(TURN_DONE_DEG):
                self.phase = "walk"
            else:
                return 0.0, math.copysign(min(self._turn_rate, MAX_YAW_RATE), self._turn)
        xy = np.asarray(xy, dtype=float)
        path = self._path[:, :2]
        # Nearest path point, looking only forward from the last one so a loop back cannot jump ahead.
        window = slice(self._near, min(len(path), self._near + int(2.0 / PATH_STEP_M) + 1))
        self._near += int(np.argmin(np.hypot(*(path[window] - xy).T)))
        s_near = float(self._s[self._near])
        self.travelled_m = s_near
        remaining = self._total - s_near
        self.progress = min(1.0, s_near / self._total) if self._total > 0 else 1.0
        if remaining < DONE_REMAINING_M:
            self.phase, self.progress = "done", 1.0
            return 0.0, 0.0
        goal_s = s_near + LOOKAHEAD_M
        if goal_s <= self._total:
            aim = np.array([np.interp(goal_s, self._s, path[:, 0]), np.interp(goal_s, self._s, path[:, 1])])
        else:                                   # past the end: carry on along the final heading
            extra = goal_s - self._total
            aim = path[-1] + extra * np.array([math.cos(self._end_heading), math.sin(self._end_heading)])
        alpha = _wrap(math.atan2(aim[1] - xy[1], aim[0] - xy[0]) - yaw)
        v = self._speed * min(1.0, remaining / SLOWDOWN_M)
        omega = 2.0 * v * math.sin(alpha) / LOOKAHEAD_M
        return v, float(np.clip(omega, -MAX_YAW_RATE, MAX_YAW_RATE))


class CompletionTracker:
    """The first step at which the robot has been at its goal for ``hold_steps`` steps in a row:
    within ``pos_tol_m``, yaw within ``yaw_tol_deg`` (modulo a full turn) and horizontal root speed
    at most ``speed_tol_mps``. ``step`` is that step, or -1."""

    def __init__(self, goal: sites.Pose2, *, pos_tol_m: float = 0.3, yaw_tol_deg: float = 20.0,
                 speed_tol_mps: float = 0.15, hold_steps: int = 5) -> None:
        self.goal = goal
        self.pos_tol_m, self.yaw_tol_deg, self.speed_tol_mps = pos_tol_m, yaw_tol_deg, speed_tol_mps
        self.hold_steps = hold_steps
        self.step = -1
        self._held = 0

    def update(self, step: int, xy, yaw: float, speed: float) -> None:
        if self.step >= 0:
            return
        near = math.hypot(xy[0] - self.goal.x, xy[1] - self.goal.y) <= self.pos_tol_m
        aligned = abs(_wrap(yaw - math.radians(self.goal.yaw_deg))) <= math.radians(self.yaw_tol_deg)
        still = speed <= self.speed_tol_mps
        self._held = self._held + 1 if (near and aligned and still) else 0
        if self._held >= self.hold_steps:
            self.step = int(step)


def stage_plan(site: sites.Site, row: Mapping) -> tuple[dict, ...]:
    """The task's stages on the row's site (conditions.row_site) in order, one JSON-plain dict a stage:
    ``name``, ``goal`` -- {"kind": "path_length", "path_m"}: the follower has gone this far along its path;
    {"kind": "heading", "yaw_deg"}: the turn on the spot is done, toward this yaw (folded into -180..180);
    {"kind": "task_goal"}: the task is done, always the last stage -- and the speed (m/s) or yaw rate (rad/s) the
    row asks of it, None for the other."""
    kind = site.path["kind"]
    if kind == "u_turn":
        speed = float(row["robot_speed_mps"])
        way_in = abs(float(site.path["turn_center_xy_m"][0]) - site.robot_start.x)
        half_circle = math.pi * float(site.path["radius_m"])
        goals = ({"kind": "path_length", "path_m": way_in}, {"kind": "path_length", "path_m": way_in + half_circle},
                 {"kind": "task_goal"})
        targets = ((speed, None),) * 3
    elif kind == "straight":
        goals, targets = ({"kind": "task_goal"},), ((float(row["robot_speed_mps"]), None),)
    elif kind == "turn_then_walk":
        yaw = (site.robot_start.yaw_deg + float(site.path["turn_deg"]) + 180.0) % 360.0 - 180.0
        goals = ({"kind": "heading", "yaw_deg": yaw}, {"kind": "task_goal"})
        targets = ((None, float(row["turn_rate_rad_s"])), (float(row["walk_speed_mps"]), None))
    else:
        raise ValueError(f"{site.scenario_id}: 단계가 정해지지 않은 길의 종류입니다: {kind!r}")
    return tuple({"name": name, "goal": goal, "target_speed_mps": mps, "target_yaw_rate_rad_s": rad_s}
                 for name, goal, (mps, rad_s) in zip(STAGE_NAMES[kind], goals, targets))


class StageTracker:
    """The step each of the task's stages (stage_plan) ended at, -1 while it is open. Nothing ends before the task
    comes (the follower's ``started_step`` is -1) or before the first recorded step. A ``path_length`` stage ends at
    the first step the follower's ``travelled_m`` is within half a path sample (``PATH_STEP_M`` / 2) of its length --
    the length is the closed form, the follower's arc length runs over the samples, so the stage ends on the sample
    where its segment ends (a half circle's chords are shorter than the arc). A ``heading`` stage ends at the first
    step the follower has left its turn. Once the task is done (``completed_step``) the last stage and every stage
    still open end at that step -- unless it is before the stage began (the task came, or the stage before ended),
    a completion that closes nothing. Stages end in order -- a stage does not end while the one before it is open,
    two can end at one step -- and a step once written stays."""

    def __init__(self, site: sites.Site, row: Mapping) -> None:
        plan = stage_plan(site, row)
        self.names = tuple(stage["name"] for stage in plan)
        self.end_steps = [-1] * len(plan)
        self._goals = tuple(stage["goal"] for stage in plan)

    def _reached(self, goal: Mapping, follower: PathFollower) -> bool:
        if goal["kind"] == "path_length":
            return follower.travelled_m >= goal["path_m"] - PATH_STEP_M / 2.0
        if goal["kind"] == "heading":
            return follower.phase not in ("wait", "turn")
        return False                                     # task_goal: only the completion ends it

    def update(self, step: int, follower: PathFollower, completed_step: int) -> None:
        if step < 0 or follower.started_step < 0:
            return
        began = follower.started_step                    # the step stage k began at: the task's, or stage k - 1's end
        for k, goal in enumerate(self._goals):
            if self.end_steps[k] < 0:
                if completed_step >= began:
                    self.end_steps[k] = int(completed_step)
                elif self._reached(goal, follower):
                    self.end_steps[k] = int(step)
                else:
                    break
            began = self.end_steps[k]
