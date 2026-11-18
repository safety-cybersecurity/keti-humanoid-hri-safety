"""The layer between the task and the walking policy: what a safety behaviour sees and returns, a
pass-through that leaves the task command alone, a delay line for what the robot perceives, and a
loader that finds a behaviour by ``module:name``. Pure Python; the environment's command term calls
a behaviour once per policy step with the task's command and returns the (forward speed, yaw rate)
the policy is given. Any callable of that shape can replace the pass-through.

The two ways in -- a walking policy of the H1 kind (an rsl_rl checkpoint) and a behaviour -- are
recorded with each episode as policy_record() writes them (``sim_backend.policy``)."""

from __future__ import annotations

import hashlib
import importlib
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# The behaviour a run uses when it names none: the reference robot, which does not react to people.
DEFAULT_BEHAVIOR = "generation.behavior:PassThrough"
# The task Isaac Lab publishes the H1 walking checkpoint for; a run that names no checkpoint walks with it.
PUBLISHED_POLICY_TASK = "Isaac-Velocity-Flat-H1-v0"


@dataclass(frozen=True)
class Perceived:
    """The person as the robot perceives them: position and velocity relative to the robot, in
    the world frame axes (metres, metres per second)."""
    rel_xy: tuple[float, float]
    rel_vel_xy: tuple[float, float]


@dataclass(frozen=True)
class BehaviorInput:
    """What a behaviour is given each policy step. ``perceived`` is None while the person is not
    seen (and for the first steps of a sensor delay)."""
    t: float
    robot_xy: tuple[float, float]
    robot_yaw: float
    robot_vel_xy: tuple[float, float]
    yaw_rate: float
    task_command: tuple[float, float]
    task_phase: str
    perceived: Perceived | None


class PassThrough:
    """The task's command, unchanged: the reference robot does not react to people."""

    def __call__(self, inp: BehaviorInput) -> tuple[float, float]:
        return inp.task_command


class DelayLine:
    """Returns each pushed value ``steps`` pushes later; None until then. Zero steps is the identity."""

    def __init__(self, steps: int) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError(f"steps 가 0 이상의 정수가 아닙니다: {steps!r}")
        self.steps = steps
        self._queue: deque = deque()

    def push(self, x):
        if self.steps == 0:
            return x
        self._queue.append(x)
        if len(self._queue) > self.steps:
            return self._queue.popleft()
        return None


def delay_steps(ms: float, dt: float) -> int:
    """A delay in milliseconds as policy steps, rounded to the nearest step (halves up)."""
    for name, value in (("ms", ms), ("dt", dt)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} 이 유한한 수가 아닙니다: {value!r}")
    if ms < 0 or dt <= 0:
        raise ValueError(f"ms 는 0 이상, dt 는 양수여야 합니다: ms={ms!r} dt={dt!r}")
    return int(math.floor(ms / 1000.0 / dt + 0.5 + 1e-9))


def parse_spec(spec: str) -> tuple[str, str]:
    """(module, name) of a ``"module:name"`` spec, without importing anything; ValueError when it is not
    of that form."""
    parts = spec.split(":") if isinstance(spec, str) else []
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"행동 층은 '모듈:이름' 꼴이어야 합니다: {spec!r}")
    return parts[0], parts[1]


def policy_record(checkpoint: Path, *, published: bool, behavior_spec: str, repo_root: Path) -> dict:
    """What an episode records of the policy that walked it (``sim_backend.policy``): ``source`` -- the
    published checkpoint by its task (``isaaclab-published:rsl_rl/<task>``), or a file by its path in the
    repository, or by its name when it lies outside (``file:<path>``; no path of the machine that ran it) --
    the file's ``sha256``, and the ``behavior`` spec."""
    path = Path(checkpoint)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if published:
        source = f"isaaclab-published:rsl_rl/{PUBLISHED_POLICY_TASK}"
    else:
        try:
            source = "file:" + path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
        except ValueError:
            source = "file:" + path.name
    return {"source": source, "sha256": digest, "behavior": behavior_spec}


def load_behavior(spec: str) -> Callable[[BehaviorInput], tuple[float, float]]:
    """``"module:name"`` as a behaviour: a class is instantiated without arguments, any other
    callable is used as it is. ValueError when the spec is malformed or names nothing loadable --
    a module that fails while importing, or a class that cannot be made without arguments, included
    (the original error chained)."""
    module_name, name = parse_spec(spec)
    try:
        module = importlib.import_module(module_name)
    except Exception as error:      # noqa: BLE001 -- any import failure is a spec that names nothing loadable
        raise ValueError(f"행동 층 {spec!r}: 모듈 {module_name} 을 불러오지 못합니다 "
                         f"({type(error).__name__}: {error})") from error
    if not hasattr(module, name):
        raise ValueError(f"행동 층 {spec!r}: 모듈 {module_name} 에 {name} 이 없습니다")
    found = getattr(module, name)
    if isinstance(found, type):
        try:
            behavior = found()
        except Exception as error:  # noqa: BLE001 -- a class that cannot be made without arguments
            raise ValueError(f"행동 층 {spec!r}: {name} 을 인자 없이 만들지 못합니다 "
                             f"({type(error).__name__}: {error})") from error
    else:
        behavior = found
    if not callable(behavior):
        raise ValueError(f"행동 층 {spec!r}: {name} 이 호출할 수 있는 것이 아닙니다")
    return behavior
