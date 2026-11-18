"""An episode's spec: the conditions it was recorded under, derived from its sidecar and the canon.

The spec has six members. ``world`` (the assets the episode loaded, the composed stage and the look drawn for it),
``humans`` (each person's model, stature, policy and script), ``task`` (what the robot is asked to do, its stages,
payload and time limit) and ``condition`` (random or grid row, the cell, the row's values) are the fixed test
conditions; ``robot`` (model, degrees of freedom, walking policy) is the part a developer swaps when another
controller is put in; ``repro`` says which code, seed and run made the episode and the command that reruns it.

Every value is plain JSON as it stands. A sidecar that lacks what the spec is built from is refused with a
ValueError naming it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from generation import assets as assets_lib
from generation import conditions, sites, task_path

SPEC_MEMBERS = ("world", "humans", "task", "robot", "condition", "repro")
HUMAN_POLICY_NAME = "scripted"
# The keys of randomization.condition that the recorder derives from the row, as opposed to the row's own columns
# and fixed values.
CONDITION_DERIVED = ("row", "robot_start", "robot_goal", "robot_path", "human_plan", "task_started_step", "design",
                     "grid_cell")


def _plain(value: object) -> object:
    """``value`` as plain JSON data: tuples as lists, numpy numbers as Python ones."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def _need(node: object, key: str, where: str) -> object:
    if not isinstance(node, Mapping) or key not in node:
        raise ValueError(f"사이드카에 {where}{key} 가 없습니다")
    return node[key]


def _number(node: object, key: str, where: str = "") -> float:
    value = _need(node, key, where)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"사이드카의 {where}{key} 가 유한한 수가 아닙니다: {value!r}")
    return float(value)


def _int(node: object, key: str) -> int:
    value = _need(node, key, "")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"사이드카의 {key} 가 0 이상의 정수가 아닙니다: {value!r}")
    return value


def rerun_command(meta: Mapping) -> str:
    """The command that reruns the episode: its id, seed and the first number of its run."""
    return (f"generation/rerun_seed.sh {_need(meta, 'episode_id', '')} {_int(meta, 'seed')} "
            f"{_int(meta, 'run_start_seq')}")


def _stage(assets: Mapping, keys: Sequence[str]) -> str | None:
    """The key of the composed asset (``composes``) whose members are all among ``keys``; None when no entry is."""
    found = [key for group in ("robots", "scene") for key, entry in (assets.get(group) or {}).items()
             if isinstance(entry.get("composes"), list) and entry["composes"] and set(entry["composes"]) <= set(keys)]
    if len(found) > 1:
        raise ValueError(f"에피소드의 자산이 엮은 자산 {found} 를 모두 채웁니다 — 어느 것이 Stage 인지 정할 수 없습니다")
    return found[0] if found else None


def _stages(meta: Mapping, scenario: Mapping, values: Mapping) -> list:
    try:
        site = conditions.row_site(sites.load_site(scenario), values)
        plan = task_path.stage_plan(site, values)
    except KeyError as missing:
        raise ValueError(f"사이드카의 조건 기록에 단계를 지을 열 {missing} 이 없습니다") from None
    recorded = _need(meta, "task_stages", "")
    names = [s.get("name") if isinstance(s, Mapping) else s for s in recorded] if isinstance(recorded, list) else None
    if names != [stage["name"] for stage in plan]:
        raise ValueError(f"사이드카의 task_stages 의 단계 {names} 이 조건 기록으로 지은 단계 "
                         f"{[stage['name'] for stage in plan]} 와 다릅니다")
    return _plain(plan)


def build_episode_spec(meta: Mapping, scenario: Mapping, assets: Mapping, *, dof: int) -> dict:
    """The spec of one episode: ``meta`` is its sidecar, ``scenario`` the scenario's meta/scenarios.json entry,
    ``assets`` meta/assets.json, ``dof`` the robot's degrees of freedom (from the joint map)."""
    randomization = _need(meta, "randomization", "")
    record = _need(randomization, "condition", "randomization.")
    if not isinstance(record, Mapping):
        raise ValueError("사이드카의 randomization.condition 이 객체가 아닙니다")
    for key in CONDITION_DERIVED:
        _need(record, key, "randomization.condition.")
    values = {k: v for k, v in record.items() if k not in CONDITION_DERIVED}
    backend = _need(meta, "sim_backend", "")
    commit = _need(backend, "generator_commit", "sim_backend.")
    policy = _need(backend, "policy", "sim_backend.")
    paths = _need(meta, "asset_paths", "")
    if not isinstance(paths, Mapping):
        raise ValueError("사이드카의 asset_paths 가 객체가 아닙니다")
    keys = [assets_lib.asset_entry(dict(assets), path)["key"] for path in paths.values()]
    stature = _number(meta, "human_stature_m")
    time_limit = _number(meta, "task_time_limit_s")
    stages = _stages(meta, scenario, values)

    payload = None
    if scenario.get("carry") is not None:
        if "box" not in paths:
            raise ValueError("사이드카의 asset_paths 에 box 가 없습니다 — 시나리오가 상자를 듭니다")
        payload = {"asset": assets_lib.asset_entry(dict(assets), paths["box"])["key"],
                   "mass_kg": _number(values, "box_mass_kg", "조건 기록의 "),
                   "size_m": list(conditions.box_size(values))}

    scene = scenario.get("scene") or {}
    spec = {
        "world": {"scenario_id": _need(meta, "scenario_id", ""), "assets": keys, "stage": _stage(assets, keys),
                  "appearance": _need(randomization, "appearance", "randomization.")},
        "humans": [{"model": scene.get("human_model"), "stature_m": stature,
                    "policy": {"name": HUMAN_POLICY_NAME, "version": commit}, "script": record["human_plan"]}],
        "task": {"archetype": scenario.get("task_archetype"), "instruction": scenario.get("robot_task"),
                 "start": record["robot_start"], "goal": record["robot_goal"], "path": record["robot_path"],
                 "payload": payload, "time_limit_s": time_limit, "stages": stages},
        "robot": {"model": _need(meta, "robot_model", ""), "dof": int(dof), "policy": policy},
        "condition": {"design": record["design"], "grid_cell": record["grid_cell"], "row": record["row"],
                      "values": values},
        "repro": {"generator_commit": commit, "seed": _int(meta, "seed"), "run_start_seq": _int(meta, "run_start_seq"),
                  "episode_seq": _int(meta, "episode_seq"), "command": rerun_command(meta)},
    }
    return _plain(spec)
