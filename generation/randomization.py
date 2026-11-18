"""What a scenario says it randomises, against what its environment config resolves to.

meta/scenarios.json declared a floor friction of 0.3-1.0 and a robot mass deviation of +-5 % for S1a and
S1d from August to 2026-10-03, and the environment drew neither: the friction event it inherits has
zero-width ranges, and the mass event scales one link by 0.8-1.25. The declaration was the catalogue's
plan; the implementation came from Isaac Lab's locomotion config; nothing compared the two.

So for a scenario that has an environment, the canon's blocks are read off the resolved config:

  randomization               every axis drawn per episode, [low, high]
  randomization_distribution  the axes that are not drawn uniformly
  randomization_fixed         what a reader may expect to vary and does not -- the value it is held at
  policy_observation_noise    half-widths of the uniform noise added, every step, to what the walking
                              policy is shown (the recorded state is the simulator's, without it)

`read` takes them off a config, `problems` compares them with a canon entry, `apply` writes them into
one. Nothing here imports Isaac: the config is read through attributes, so the tests pass a stand-in and
generation/envs/s1a/check_randomization.py passes the real one.

The scenarios with a conditions table (meta/scenarios.json ``conditions`` -- S1e, S1f, S1g) vary by the row: one
row per episode, drawn by generation/conditions.py before the environment exists. What the row sets -- the robot's
start, the person's script, the carried box, the task's speed and the sensor delay -- is not a range an episode is
drawn from, so the terms that set it are not read (FROM_THE_ROW, and a command term that carries the row), and
what is left is what the environment draws beside the row.

What this does not see:

- how a term draws. A range is read; the code that samples it is not. sensor_delay_ms is drawn in
  [0, 100) and floored to the record period, so the delays realised are 0-80 ms in five values; the
  canon says that in prose.
- anything random outside the event, command and policy-observation configs.
- whether a term reaches the body it names -- the realised values in each episode's sidecar are the
  check for that (randomization.robot_base_mass_kg_realized and the two robot_*_friction_realized).
- materials other than the robot's and the terrain's (the dummy, the racks).
- a range that is declared and never drawn because its event's mode is not handled here: only
  "startup" is refused (one draw per process, not per episode -- how the base mass was shared by every
  episode of a run until 2026-08-20).
- a draw inside a term that takes its values from the row: such an event term is known by the function it
  runs (FROM_THE_ROW) and the command term by the row it carries, and neither is read further.
"""

from __future__ import annotations

import copy
import math
from typing import Any

SAMPLED = "randomization"
DISTRIBUTION = "randomization_distribution"
FIXED = "randomization_fixed"
NOISE = "policy_observation_noise"
BLOCKS = (SAMPLED, DISTRIBUTION, FIXED, NOISE)

UNIFORM = "uniform"
STARTUP = "startup"

# What `apply` holds in the fixed block of an entry that has none yet: the quantities the catalogue
# once declared as randomised, or that a later scenario (S3b) will randomise.
DEFAULT_FIXED = (
    "robot_static_friction",
    "robot_dynamic_friction",
    "terrain_static_friction",
    "terrain_dynamic_friction",
)

# Event terms that draw nothing: a guard that checks the reset, and the carried box put back where the canon's
# carry block holds it.
NOT_A_DRAW = frozenset({"aisle_guard", "row_guard", "carried_box_place"})
# Event terms that set what a conditions-table row holds, known by the function they run: the scenarios with a
# table keep the inherited terms' names (reset_base, reset_dummy_entry) so that the reset order stays, and swap
# their functions -- the same name running the inherited draw is still read as one.
FROM_THE_ROW = frozenset({"reset_robot_to_site", "reset_dummy_scripted", "author_carried_box"})
COMMAND = "base_velocity"
# The field of a command config that carries a conditions-table row: such a term sets the walking speed and the
# sensor delay from the row at every reset, over what its inherited ranges drew.
ROW = "row"
# The key of a canon entry's conditions table; `apply` puts the blocks right after it.
TABLE = "conditions"

_POSE = {
    "x": "robot_start_offset_x", "y": "robot_start_offset_y", "z": "robot_start_offset_z",
    "roll": "robot_start_roll", "pitch": "robot_start_pitch", "yaw": "robot_start_yaw",
}
_VELOCITY = {
    "x": "robot_start_velocity_x", "y": "robot_start_velocity_y", "z": "robot_start_velocity_z",
    "roll": "robot_start_roll_rate", "pitch": "robot_start_pitch_rate", "yaw": "robot_start_yaw_rate",
}
_PUSH = {axis: f"robot_push_velocity_{axis}" for axis in _POSE}
_COM = {axis: f"robot_base_com_offset_{axis}" for axis in ("x", "y", "z")}

_TOL = 1e-12


def _real(value: Any) -> bool:
    # bool is an int; a flag is not a bound.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _pair(bounds: Any) -> tuple[float, float] | None:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or not all(_real(b) for b in bounds):
        return None
    return float(bounds[0]), float(bounds[1])


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=_TOL)


def _runs(term: Any) -> str | None:
    """The name of the function an event term runs (functools.wraps keeps the wrapped one's)."""
    return getattr(getattr(term, "func", None), "__name__", None)


def _fields(node: Any) -> dict[str, Any]:
    # Isaac Lab's managers take a dict of terms as well as a config object.
    items = node.items() if isinstance(node, dict) else vars(node).items()
    return {k: v for k, v in items if not str(k).startswith("_")}


class _Reading:
    def __init__(self) -> None:
        self.sampled: dict[str, list[float]] = {}
        self.distribution: dict[str, str] = {}
        self.fixed: dict[str, float] = {}
        self.noise: dict[str, float] = {}
        self.problems: list[str] = []

    def put(self, name: str, bounds: Any, *, where: str, distribution: str = UNIFORM,
            mode: str | None = None) -> None:
        pair = _pair(bounds)
        if pair is None:
            self.problems.append(f"{where}: {name} 의 범위를 읽지 못했습니다 — {bounds!r}")
            return
        low, high = pair[0] + 0.0, pair[1] + 0.0      # -0.0 (Isaac Lab writes one) reads as 0.0
        if low == high:
            self.fixed[name] = low
            return
        if low > high:
            self.problems.append(f"{where}: {name} 의 하한이 상한보다 큽니다 — {[low, high]}")
            return
        self.sampled[name] = [low, high]
        if distribution != UNIFORM:
            self.distribution[name] = str(distribution)
        if mode == STARTUP:
            self.problems.append(
                f"{where}: {name} 은 mode=startup 이라 실행마다 한 번만 뽑힙니다 — 정본의 "
                f"{SAMPLED} 은 에피소드마다 뽑는 축입니다")

    def table(self, term: Any, key: str, names: dict[str, str], where: str) -> None:
        """A dict of axis -> range. The table itself is required (the event does not run without it);
        an axis it leaves out is (0.0, 0.0), which is how Isaac Lab reads it (range.get(axis, (0.0, 0.0))
        in reset_root_state_uniform, push_by_setting_velocity and randomize_rigid_body_com)."""
        ranges = term.params.get(key)
        if not isinstance(ranges, dict):
            self.problems.append(f"{where}: {key} 가 없습니다 — 이 이벤트는 그 표 없이 돌지 않습니다")
            return
        for axis in ranges:
            if axis not in names:
                self.problems.append(f"{where}: {key} 의 축 {axis!r} 를 읽을 줄 모릅니다")
        for axis, name in names.items():
            self.put(name, ranges.get(axis, (0.0, 0.0)), where=where, mode=term.mode)

    def blocks(self) -> dict[str, dict]:
        return {SAMPLED: self.sampled, DISTRIBUTION: self.distribution, FIXED: self.fixed, NOISE: self.noise}


def _physics_material(r: _Reading, term: Any, where: str) -> None:
    for name, key in (("robot_static_friction", "static_friction_range"),
                      ("robot_dynamic_friction", "dynamic_friction_range"),
                      ("robot_restitution", "restitution_range")):
        r.put(name, term.params.get(key), where=where, mode=term.mode)


def _add_base_mass(r: _Reading, term: Any, where: str) -> None:
    operation = term.params.get("operation")
    if operation != "scale":
        r.problems.append(f"{where}: operation={operation!r} — 배율(scale)만 읽을 줄 압니다")
        return
    r.put("robot_base_mass_scale", term.params.get("mass_distribution_params"), where=where,
          distribution=term.params.get("distribution", UNIFORM), mode=term.mode)


def _base_com(r: _Reading, term: Any, where: str) -> None:
    r.table(term, "com_range", _COM, where)


def _external_force(r: _Reading, term: Any, where: str) -> None:
    r.put("robot_external_force", term.params.get("force_range"), where=where, mode=term.mode)
    r.put("robot_external_torque", term.params.get("torque_range"), where=where, mode=term.mode)


def _reset_base(r: _Reading, term: Any, where: str) -> None:
    r.table(term, "pose_range", _POSE, where)
    r.table(term, "velocity_range", _VELOCITY, where)


def _reset_robot_joints(r: _Reading, term: Any, where: str) -> None:
    r.put("robot_start_joint_position_scale", term.params.get("position_range"), where=where, mode=term.mode)
    r.put("robot_start_joint_velocity_scale", term.params.get("velocity_range"), where=where, mode=term.mode)


def _push_robot(r: _Reading, term: Any, where: str) -> None:
    r.table(term, "velocity_range", _PUSH, where)


def _reset_dummy_entry(r: _Reading, term: Any, where: str) -> None:
    for name, key in (("human_entry_speed", "speed_range"),
                      ("human_entry_angle_deg", "angle_range_deg"),
                      ("entry_trigger_distance", "trigger_distance_range")):
        r.put(name, term.params.get(key), where=where, mode=term.mode)


_EVENTS = {
    "physics_material": _physics_material,
    "add_base_mass": _add_base_mass,
    "base_com": _base_com,
    "base_external_force_torque": _external_force,
    "reset_base": _reset_base,
    "reset_robot_joints": _reset_robot_joints,
    "push_robot": _push_robot,
    "reset_dummy_entry": _reset_dummy_entry,
}


def read(env_cfg: Any) -> tuple[dict[str, dict], list[str]]:
    """The four blocks as the resolved config has them, and what could not be read.

    A range of zero width is a value held fixed, not an axis. An event term this module has no reader
    for is a problem, not a silence: an environment that gains a draw must gain a line here.
    """
    r = _Reading()

    for name, term in _fields(env_cfg.events).items():
        if term is None or name in NOT_A_DRAW or _runs(term) in FROM_THE_ROW:
            continue
        reader = _EVENTS.get(name)
        if reader is None:
            r.problems.append(
                f"events.{name}: 읽을 줄 모르는 이벤트 항입니다 — 무엇을 뽑는지 "
                "generation/randomization.py 에 적어야 정본과 대조할 수 있습니다")
            continue
        reader(r, term, f"events.{name}")

    for name, term in _fields(env_cfg.commands).items():
        if term is None:
            continue
        if name != COMMAND:
            r.problems.append(f"commands.{name}: 읽을 줄 모르는 커맨드 항입니다")
            continue
        if getattr(term, ROW, None) is not None:
            continue
        r.put("robot_walk_speed", getattr(term, "walk_speed_range", None), where=f"commands.{name}")
        r.put("sensor_delay_ms", getattr(term, "sensor_delay_ms_range", None), where=f"commands.{name}")

    material = getattr(getattr(env_cfg.scene, "terrain", None), "physics_material", None)
    if material is not None:
        for name, attr in (("terrain_static_friction", "static_friction"),
                           ("terrain_dynamic_friction", "dynamic_friction"),
                           ("terrain_restitution", "restitution")):
            value = getattr(material, attr, None)
            if _real(value):
                r.fixed[name] = float(value)

    policy = env_cfg.observations.policy
    if getattr(policy, "enable_corruption", False):
        for name, term in _fields(policy).items():
            noise = getattr(term, "noise", None)
            if noise is None:
                continue
            low, high = getattr(noise, "n_min", None), getattr(noise, "n_max", None)
            operation = getattr(noise, "operation", "add")      # Isaac Lab's default
            where = f"observations.policy.{name}"
            if not _real(low) or not _real(high):
                r.problems.append(f"{where}: 균등 잡음(n_min · n_max)만 읽을 줄 압니다 — {type(noise).__name__}")
            elif operation != "add":
                # "scale" multiplies the observation by the draw and "abs" replaces it: not a half-width added.
                r.problems.append(f"{where}: 더하는 잡음만 읽을 줄 압니다 — operation={operation!r}")
            elif not _same(float(low), -float(high)):
                r.problems.append(f"{where}: 잡음이 0 을 중심으로 대칭이 아닙니다 — [{low}, {high}]")
            elif high < 0:
                r.problems.append(f"{where}: 잡음의 범위가 뒤집혀 있습니다 — [{low}, {high}]")
            elif high > 0:
                r.noise[name] = float(high)

    return r.blocks(), r.problems


def declared(entry: dict) -> tuple[dict[str, dict], list[str]]:
    """The four blocks as a canon entry declares them, and what in them is not well formed."""
    found: list[str] = []
    out: dict[str, dict] = {block: {} for block in BLOCKS}
    for block in BLOCKS:
        raw = entry.get(block)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            found.append(f"{block}: 객체가 아닙니다 — {type(raw).__name__}")
            continue
        for name, value in raw.items():
            if block == SAMPLED:
                pair = _pair(value)
                if pair is None or pair[0] >= pair[1]:
                    found.append(f"{block}.{name}: [하한, 상한] 이 아닙니다 — {value!r}")
                    continue
                out[block][name] = list(pair)
            elif block == DISTRIBUTION:
                if not isinstance(value, str) or not value:
                    found.append(f"{block}.{name}: 분포 이름이 아닙니다 — {value!r}")
                    continue
                out[block][name] = value
            elif not _real(value):
                found.append(f"{block}.{name}: 수가 아닙니다 — {value!r}")
            elif block == NOISE and value <= 0:
                found.append(f"{block}.{name}: 잡음의 반폭은 0 보다 커야 합니다 — {value!r}")
            else:
                out[block][name] = float(value)
    for name in out[DISTRIBUTION]:
        if name not in out[SAMPLED]:
            found.append(f"{DISTRIBUTION}.{name}: {SAMPLED} 에 없는 축입니다")
    return out, found


def problems(entry: dict, env_cfg: Any) -> list[str]:
    """Every way the canon entry and the resolved config disagree. Empty when they say the same.

    The drawn axes, their distributions and the policy's noise are compared both ways: an axis only one
    side has is a difference. The fixed block is compared one way -- the canon lists the few values a
    reader may expect to vary, the config holds many more fixed that nobody needs told.
    """
    have, found = read(env_cfg)
    want, malformed = declared(entry)
    out = found + malformed

    for name in sorted(set(have[SAMPLED]) | set(want[SAMPLED])):
        canon, env = want[SAMPLED].get(name), have[SAMPLED].get(name)
        if canon is None:
            out.append(f"{name}: 환경은 {env} 에서 뽑는데 정본의 {SAMPLED} 에 없습니다")
        elif env is None and name in have[FIXED]:
            out.append(f"{name}: 정본은 {canon} 에서 뽑는다고 적었는데 환경은 {have[FIXED][name]} 로 고정합니다")
        elif env is None:
            out.append(f"{name}: 정본은 {canon} 에서 뽑는다고 적었는데 환경에는 그 축이 없습니다")
        elif not (_same(canon[0], env[0]) and _same(canon[1], env[1])):
            out.append(f"{name}: 범위가 다릅니다 — 정본 {canon} · 환경 {env}")

    for name in sorted(set(have[SAMPLED]) & set(want[SAMPLED])):
        canon = want[DISTRIBUTION].get(name, UNIFORM)
        env = have[DISTRIBUTION].get(name, UNIFORM)
        if canon != env:
            out.append(f"{name}: 분포가 다릅니다 — 정본 {canon} · 환경 {env}")

    for name, canon in want[FIXED].items():
        if name in have[SAMPLED]:
            out.append(f"{name}: 정본은 {canon} 로 고정이라 적었는데 환경은 {have[SAMPLED][name]} 에서 뽑습니다")
        elif name not in have[FIXED]:
            out.append(f"{name}: 정본이 고정값 {canon} 을 적었는데 환경에서 읽지 못한 이름입니다")
        elif not _same(canon, have[FIXED][name]):
            out.append(f"{name}: 고정값이 다릅니다 — 정본 {canon} · 환경 {have[FIXED][name]}")

    for name in sorted(set(have[NOISE]) | set(want[NOISE])):
        canon, env = want[NOISE].get(name), have[NOISE].get(name)
        if canon is None:
            out.append(f"{NOISE}.{name}: 환경은 ±{env} 를 더하는데 정본에 없습니다")
        elif env is None:
            out.append(f"{NOISE}.{name}: 정본은 ±{canon} 라 적었는데 환경은 그 관측에 잡음을 더하지 않습니다")
        elif not _same(canon, env):
            out.append(f"{NOISE}.{name}: 반폭이 다릅니다 — 정본 ±{canon} · 환경 ±{env}")

    return out


def apply(entry: dict, blocks: dict[str, dict]) -> dict:
    """A copy of ``entry`` whose blocks are what ``blocks`` (from `read`) holds.

    The drawn axes keep the order the entry had them in and new ones follow; a bound the entry already
    wrote as an equal number keeps its spelling (30 stays 30, not 30.0). The fixed block keeps its own
    choice of names -- DEFAULT_FIXED for an entry that has none -- and takes their values; a name the
    config no longer holds fixed leaves the block. A block the entry has stays, even empty (an empty
    fixed block is a choice of no names; dropped, the next write would read it as "none yet" and bring
    the default names back); a block the entry lacks is added only when there is something to put in it,
    beside the drawn axes.

    An entry with a conditions table gets all four, empty ones too, right after the table (unless it has
    its drawn axes already): what varies from one of its episodes to the next is the row, and an empty
    randomization says that nothing else is drawn -- left out, it would read as a block nobody wrote.
    """
    out = copy.deepcopy(entry)
    old = entry.get(SAMPLED) if isinstance(entry.get(SAMPLED), dict) else {}

    def spelled(name: str, pair: list[float]) -> list:
        before = _pair(old.get(name))
        if before is not None and _same(before[0], pair[0]) and _same(before[1], pair[1]):
            return list(old[name])
        return list(pair)

    order = [n for n in old if n in blocks[SAMPLED]] + [n for n in blocks[SAMPLED] if n not in old]
    new: dict[str, dict] = {
        SAMPLED: {n: spelled(n, blocks[SAMPLED][n]) for n in order},
        DISTRIBUTION: dict(blocks[DISTRIBUTION]),
        NOISE: dict(blocks[NOISE]),
    }
    names = list(entry[FIXED]) if isinstance(entry.get(FIXED), dict) else list(DEFAULT_FIXED)
    new[FIXED] = {n: blocks[FIXED][n] for n in names if n in blocks[FIXED]}

    table = TABLE in entry
    rebuilt: dict[str, Any] = {}
    pending = [b for b in BLOCKS if b not in entry and (new[b] or table)]
    beside = SAMPLED if SAMPLED in entry or not table else TABLE
    for key, value in out.items():
        rebuilt[key] = new[key] if key in new else value
        if key == beside:
            for block in pending:
                rebuilt[block] = new[block]
            pending = []
    for block in pending:           # an entry with no randomization key at all
        rebuilt[block] = new[block]
    return rebuilt


def entries(canon: dict) -> tuple[dict[str, dict], list[str]]:
    """The canon's scenario entries by id. An id listed twice is in neither the result nor a guess: the
    converter and the labeller take the first entry of an id, so a check that took the last one could
    pass on an entry nothing else reads."""
    found: list[str] = []
    by_id: dict[str, dict] = {}
    twice: set[str] = set()
    for entry in canon.get("scenarios") or []:
        if not isinstance(entry, dict):
            continue
        scenario = entry.get("scenario_id")
        if scenario in by_id or scenario in twice:
            twice.add(scenario)
            by_id.pop(scenario, None)
        else:
            by_id[scenario] = entry
    for scenario in sorted(twice, key=str):
        found.append(f"{scenario}: meta/scenarios.json 에 같은 scenario_id 의 항목이 둘 이상입니다")
    return by_id, found


def rewritten(entry: dict, env_cfg: Any) -> tuple[dict | None, list[str]]:
    """``entry`` with its blocks written from the config, or None and the reasons it is not written.

    Nothing is written from a config that could not be read in full, and the result is compared with
    the config once more before it is handed back: a write that leaves something to report is refused.
    """
    blocks, unread = read(env_cfg)
    if unread:
        return None, unread
    after = apply(entry, blocks)
    left = problems(after, env_cfg)
    if left:
        return None, [f"쓴 결과가 환경과 다릅니다 — {p}" for p in left]
    return after, []

