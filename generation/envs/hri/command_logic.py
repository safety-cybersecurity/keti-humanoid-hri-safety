"""The Isaac-free half of the HRI scenarios' command term (generation/envs/hri/commands.py): what the dummy's contact
sensor is filtered against and how its force matrix is read, the peak of one env step over its physics substeps, the
person's scripted trigger, turn on the spot and walk, when the robot's task comes, the first step's approach line, and
what a behaviour is told of the person the robot sees.

The dummy's contact sensor is filtered against each of the robot's links and the box the robot carries -- one prim per
expression, so one env (contact_filters). Its force matrix (B, F, 3) is then the force on each of the person's B bodies
from each of the F filters, and only from them: measured in Isaac Sim 6.0.1 / Isaac Lab 3.0.0 on 2026-10-20 with the
published H1 walking policy, the dummy's feet standing on the floor and a fallen dummy lying on it gave the matrix
nothing, the sum of a body's row was its net force whenever only the robot or the box touched it, and the robot pushed
into a rack gave the matrix nothing either. So what the robot's links and the held box give the person (body_force) is
read from the matrix and never from the robot's net force, which counts the racks and the floor.

A body's force is the larger of its row's sum and its strongest single entry (body_force): parts pushing the same way
add up, and two parts pressing one body from opposite sides -- a pinch, such as a hand between the held box and a
forearm -- keep one side's push instead of cancelling to nothing. The recorder reads the matrix with the same
function (record_logic.attribute_contacts), so the box and the hold are let go on the step it records the first touch.

An env step runs four physics substeps and a contact can live in one of them alone. TouchWindow keeps the step's peak:
it is sampled from a hook that runs before each substep, so a sample reads the substep before it -- the first sample
of a step is the previous step's last and is dropped, and the step's own last substep is read once the physics loop
is over (drain). generation/envs/s1a/substep_contact.py samples the contact sensors' net forces the same way.

Pure numpy; the command term calls these once per policy step and the tests call them on matrices written by hand.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from generation import human_script, visibility
from generation.behavior import Perceived

ENV_NS = "{ENV_REGEX_NS}"
ROBOT_PRIM = "Robot"
TRIGGER_KINDS = ("never", "robot_behind", "time")
# The site kind whose task waits for the person (S1g); every other path's task comes at the first step.
WAITING_KIND = "turn_then_walk"
# Seconds the legs take, from standing at the trigger, to mark time at the slowest walking speed while the person turns
# on the spot (turn_gait_speed): the gait's first steps grow from the stand pose instead of starting at a full step.
TURN_GAIT_RAMP_S = 0.4


def contact_filters(links: Sequence[str], box_prim: str | None) -> list[str]:
    """The dummy contact sensor's filter expressions: one per robot link (``links``, in their order), then the carried
    box's prim (``box_prim``, under the env) where there is one. The matrix's columns follow this order."""
    if isinstance(links, str):
        raise ValueError(f"links 는 링크 이름의 목록이어야 합니다 (문자열 {links!r} 이 아니라)")
    names = list(links)
    if not names or len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        raise ValueError(f"링크 이름이 비어 있지 않고 서로 다른 문자열이어야 합니다: {names!r}")
    out = [f"{ENV_NS}/{ROBOT_PRIM}/{name}" for name in names]
    if box_prim is not None:
        if not isinstance(box_prim, str) or not box_prim:
            raise ValueError(f"상자 프림 이름이 문자열이 아닙니다: {box_prim!r}")
        out.append(f"{ENV_NS}/{box_prim}")
    return out


def filter_names(filters: Sequence[str]) -> list[str]:
    """The prim name each filter expression names (its last path part): a robot link's name, or the box's prim."""
    return [str(f).rsplit("/", 1)[-1] for f in filters]


def _matrix(matrix) -> np.ndarray:
    m = np.asarray(matrix, float)
    if m.ndim != 3 or m.shape[2] != 3 or not np.isfinite(m).all():
        raise ValueError(f"힘 행렬이 유한한 수의 (사람 바디, 필터, 3) 이 아닙니다: {m.shape}")
    return m


def body_force(matrix) -> np.ndarray:
    """(B,) the force each of the person's bodies gets from the filters of ``matrix`` -- the robot's links and the
    held box: max(|sum over f of M[b, f]|, max over f of |M[b, f]|) (N). The sum where the parts push the same way;
    never less than any one part's push, so a pinch between two parts is one side's force, not nothing."""
    m = _matrix(matrix)
    together = np.linalg.norm(m.sum(axis=1), axis=-1)
    strongest = np.linalg.norm(m, axis=-1).max(axis=1, initial=0.0)
    return np.maximum(together, strongest)


def source_force(matrix) -> np.ndarray:
    """(F,) what each filter gives the person as a whole: |sum over b of M[b, f]| (N)."""
    return np.linalg.norm(_matrix(matrix).sum(axis=0), axis=-1)


@dataclass(frozen=True, eq=False)
class Touch:
    """One env step's peaks over its substeps: per person body what the robot's links and the box gave it (N) and
    per filter what it gave the person (N), and how many substeps were read."""
    body_n: np.ndarray
    source_n: np.ndarray
    samples: int

    @property
    def peak_n(self) -> float:
        """The largest force any of the person's bodies got from the robot and the box this step (0 without bodies)."""
        return float(self.body_n.max()) if self.body_n.size else 0.0

    def over(self, threshold_n: float) -> bool:
        """Whether some body got more than ``threshold_n`` from the robot and the box this step."""
        return self.peak_n > float(threshold_n)


class TouchWindow:
    """The per-substep peak of the dummy contact sensor's force matrix over one env step. ``read_matrix`` returns the
    (B, F, 3) matrix as the sensor has it now. ``sample`` is the substep hook (it reads the substep before it, so the
    first call of a step is dropped); ``drain`` reads the step's last substep, hands the step's Touch and starts the
    next step's batch; ``reset`` drops a partial batch at an episode boundary."""

    def __init__(self, read_matrix: Callable[[], object]) -> None:
        self._read = read_matrix
        self.reset()

    def reset(self) -> None:
        self._body: np.ndarray | None = None
        self._source: np.ndarray | None = None
        self._samples = 0
        self._skip_next = True

    def sample(self, *, force: bool = False) -> None:
        matrix = self._read()
        if self._skip_next and not force:
            self._skip_next = False
            return
        self._skip_next = False
        body, source = body_force(matrix), source_force(matrix)
        if self._body is None:
            self._body, self._source = body, source
        else:
            if body.shape != self._body.shape or source.shape != self._source.shape:
                raise ValueError(f"힘 행렬의 모양이 스텝 안에서 바뀌었습니다: {self._body.shape} → {body.shape}")
            self._body, self._source = np.maximum(self._body, body), np.maximum(self._source, source)
        self._samples += 1

    def drain(self) -> Touch:
        self.sample(force=True)
        touch = Touch(body_n=self._body, source_n=self._source, samples=self._samples)
        self.reset()
        return touch


def spawn_pose(plan: human_script.HumanPlan) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """(position, orientation xyzw) of the dummy's root as the environment creates it: where the row's script parks
    it (its start, on the floor) and turned to its yaw, as events.reset_dummy_scripted writes it at every reset.

    Created anywhere else, the dummy stands on the floor there while the simulator starts, and the first reset then
    moves it across the warehouse to the start; the first physics substep after that move threw it -- measured in
    Isaac Sim 6.0.1 / Isaac Lab 3.0.0 on 2026-10-20, S1g row 37 moved 26 m from the spawn place: 10 kN under one foot,
    14 steps with both feet in the air and 20 deg of tilt; created at its start, 0.48 / 0.49 kN and at most 4 deg."""
    from generation.envs.s1a.mdp.entry_geometry import facing_quat_xyzw

    x, y = (float(v) for v in plan.start_xy)
    return (x, y, 0.0), tuple(float(v) for v in facing_quat_xyzw(plan.yaw_deg))


def trigger_due(kind: str, value: float, *, t: float, person_x: float, robot_x: float) -> bool:
    """Whether the person's scripted walk sets off now (human_script.Trigger): ``never``; ``robot_behind`` -- the robot
    is behind the person along +x by less than ``value`` and has not passed them (0 <= person x - robot root x < value,
    the S1d entry's along-track trigger); ``time`` -- the episode time ``t`` has reached ``value``."""
    if kind == "never":
        return False
    if kind == "robot_behind":
        ahead = float(person_x) - float(robot_x)
        return 0.0 <= ahead < float(value)
    if kind == "time":
        return float(t) >= float(value)
    raise ValueError(f"사람 대본이 모르는 방아쇠입니다: {kind!r} (알려진 것 {list(TRIGGER_KINDS)})")


def walk_remaining(target_xy, start_xy, walk_dir, stop_after_m: float | None) -> float:
    """Metres the scripted walk has left to its stop: ``stop_after_m`` less how far the target has come from
    ``start_xy`` along ``walk_dir`` (a unit vector), 0 and up; infinite for a walk without a stop."""
    if stop_after_m is None or math.isinf(float(stop_after_m)):
        return math.inf
    along = float(np.dot(np.asarray(target_xy, float) - np.asarray(start_xy, float), np.asarray(walk_dir, float)))
    return max(0.0, float(stop_after_m) - along)


def turn_steps(turn_time_s: float, dt: float) -> int:
    """How many steps the person's turn on the spot lasts (S1f, between the trigger and the walk): the integer the
    plan's walk counts it with (human_script.turn_steps). 0 is no turn."""
    return human_script.turn_steps(turn_time_s, dt)


def turn_yaw_deg(from_deg: float, to_deg: float, k: int, n: int) -> float:
    """The person's facing (degrees about +z) k steps into an n-step turn on the spot from ``from_deg`` to ``to_deg``:
    from_deg + d * (1 - cos(pi * k / n)) / 2, d the signed shorter arc from one to the other -- the short way round,
    slow at both ends, so the yaw the hold servo aims at starts and stops turning without a jump in its rate. from_deg
    at k <= 0, to_deg at k >= n."""
    if int(n) < 1:
        raise ValueError(f"돌기의 스텝 수가 1 이상이 아닙니다: {n!r}")
    if k <= 0:
        return float(from_deg)
    if k >= n:
        return float(to_deg)
    d = (float(to_deg) - float(from_deg) + 180.0) % 360.0 - 180.0
    return float(from_deg) + d * (1.0 - math.cos(math.pi * k / n)) / 2.0


def turn_yaw_rate_deg_s(from_deg: float, to_deg: float, k: int, n: int, dt: float) -> float:
    """The rate (degrees per second) turn_yaw_deg's facing turns at k steps into an n-step turn of steps of ``dt``
    seconds: the derivative of its eased arc, d * pi / (2 n dt) * sin(pi k / n), d the same signed shorter arc -- 0 at
    both ends of the turn and outside it. The hold servo's damper reads the dummy's yaw rate against it
    (commands.HriTaskCommand._facing_turn_torque)."""
    if int(n) < 1:
        raise ValueError(f"돌기의 스텝 수가 1 이상이 아닙니다: {n!r}")
    if k <= 0 or k >= n:
        return 0.0
    d = (float(to_deg) - float(from_deg) + 180.0) % 360.0 - 180.0
    return d * math.pi / (2.0 * int(n) * float(dt)) * math.sin(math.pi * k / n)


def turn_gait_speed(k: int, slow_mps: float, dt: float) -> float:
    """The speed the legs mark time at k steps into the turn on the spot (the gait's phase advances by it, the target
    by nothing): a straight rise from standing over TURN_GAIT_RAMP_S to ``slow_mps`` -- the gait table's slowest walking
    speed (human_script.turn_gait_speed_mps), the speed the walk after the turn sets off at -- and held there:
    slow_mps * min(1, (k + 1) * dt / TURN_GAIT_RAMP_S)."""
    return float(slow_mps) * min(1.0, (int(k) + 1) * float(dt) / TURN_GAIT_RAMP_S)


def task_starts(site_kind: str, plan: human_script.HumanPlan, *, pelvis_xy, robot_xy, distance: float | None) -> bool:
    """Whether the robot's task comes at this step: at once on a path that does not wait (S1e, S1f); on a turn and a
    walk (S1g) once the person's pelvis is no more than ``distance`` from the robot measured along the person's walk
    (human_script.within_along)."""
    if site_kind != WAITING_KIND:
        return True
    if distance is None:
        raise ValueError("돌고 걷는 과제는 사람과의 거리로 옵니다 — distance 가 없습니다")
    return human_script.within_along(plan, pelvis_xy, robot_xy, float(distance))


def first_direction(robot_root, pelvis) -> np.ndarray:
    """The approach line of an episode's first step, before the closest points have given one: the unit vector from
    the robot's root to the person's pelvis."""
    v = np.asarray(pelvis, float).reshape(-1) - np.asarray(robot_root, float).reshape(-1)
    norm = float(np.linalg.norm(v)) if v.shape == (3,) and np.isfinite(v).all() else math.nan
    if not norm > 1e-9:
        raise ValueError(f"로봇 뿌리와 사람 골반이 겹치거나 유한한 세 수가 아닙니다: {robot_root!r} · {pelvis!r}")
    return v / norm


def any_seen(fractions: Mapping[str, float], heights: Mapping[str, float], rule: Mapping) -> bool:
    """Whether one of the robot's cameras sees the person (visibility.seen): enough of the body unhidden and the body
    tall enough in that camera's picture. ``fractions`` and ``heights`` name the same cameras."""
    if set(fractions) != set(heights):
        raise ValueError(f"보이는 몫과 키 높이의 카메라가 다릅니다: {sorted(fractions)} · {sorted(heights)}")
    return any(visibility.seen(float(fractions[c]), float(heights[c]), rule) for c in fractions)


def perceived(seen: bool, *, pelvis_xy, pelvis_vel_xy, robot_xy, robot_vel_xy) -> Perceived | None:
    """The person as the robot perceives them now (behavior.Perceived -- the pelvis relative to the robot's root,
    world axes), or None while no camera sees them."""
    if not seen:
        return None
    rel = np.asarray(pelvis_xy, float) - np.asarray(robot_xy, float)
    rel_v = np.asarray(pelvis_vel_xy, float) - np.asarray(robot_vel_xy, float)
    return Perceived(rel_xy=(float(rel[0]), float(rel[1])), rel_vel_xy=(float(rel_v[0]), float(rel_v[1])))


def behavior_command(out: object) -> tuple[float, float]:
    """A behaviour's answer as (forward speed m/s, yaw rate rad/s): two finite numbers, or ValueError naming what came
    back."""
    pair = list(out) if isinstance(out, (tuple, list, np.ndarray)) else None
    if pair is None or len(pair) != 2:
        raise ValueError(f"행동 층이 (앞 속도, 요 각속도) 두 수를 돌려주지 않았습니다: {out!r}")
    values = []
    for v in pair:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) \
                or not math.isfinite(float(v)):
            raise ValueError(f"행동 층의 명령이 유한한 두 수가 아닙니다: {out!r}")
        values.append(float(v))
    return values[0], values[1]
