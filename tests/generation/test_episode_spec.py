"""generation/episode_spec.py: an episode's spec derived from its sidecar and the canon."""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path

import pytest

from generation import assets as assets_lib
from generation import conditions, episode_spec, human_script, sites, task_path
from generation.envs.hri import record_logic
from generation.scene import layout

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("S1e", "S1f", "S1g")
COMMIT = "0123456789abcdef0123456789abcdef01234567"
DOF = 19
POLICY = {"source": "isaaclab-published:rsl_rl/Isaac-Velocity-Flat-H1-v0", "sha256": "ab" * 32,
          "behavior": "task_follow"}
APPEARANCE = {"vest_color": "orange", "light_scale": 1.0}


@functools.cache
def _boxes():
    return layout.build_layout(layout.load_spec(layout.SCENE_SPEC))


@functools.cache
def _canon_assets():
    return assets_lib.load_assets(ROOT)


def _assets():
    return copy.deepcopy(_canon_assets())


def _entry(sid):
    return conditions.scenario_entry(sid)


def _asset_paths(entry, assets):
    scene = assets["scene"]
    paths = {"robot": assets["robots"]["unitree_h1"]["usd_path"],
             "dummy": scene[entry["scene"]["human_model"]]["usd_path"],
             "terrain": scene["isaaclab_terrain_plane"]["usd_path"],
             "aisle": scene["keti_logistics_aisle_v1"]["usd_path"]}
    if "carry" in entry:
        paths["box"] = scene[entry["carry"]["asset"]]["usd_path"]
    return paths


def sidecar(sid, index, *, assets=None):
    """The sidecar of row ``index`` of a scenario, built with the helpers the recorder uses."""
    entry, assets = _entry(sid), assets or _canon_assets()
    row = conditions.draw_row(entry, index, _boxes())
    site = conditions.row_site(sites.load_site(entry), row)
    design, cell = conditions.row_design(entry, index)
    condition = record_logic.condition_record(
        index, row, site, human_script.plan_for(entry, row), task_started_step=3, design=design, grid_cell=cell)
    names = [stage["name"] for stage in task_path.stage_plan(site, row)]
    stages = record_logic.task_stages_record(names, [10 * (k + 1) for k in range(len(names))], 3, 50)
    return {
        "episode_id": f"{sid}_{index:04d}_seed1234", "scenario_id": sid, "episode_seq": index, "run_start_seq": index,
        "seed": 1234, "robot_model": "unitree_h1", "asset_paths": _asset_paths(entry, assets),
        "sim_backend": {"generator_commit": COMMIT, "generator_dirty": False, "policy": dict(POLICY)},
        "randomization": {"condition": condition, "appearance": dict(APPEARANCE)},
        "task_stages": stages, "human_stature_m": 1.74, "task_time_limit_s": 40.0,
    }


def spec_of(meta, sid, assets=None):
    return episode_spec.build_episode_spec(meta, _entry(sid), assets or _canon_assets(), dof=DOF)


CASES = [(sid, index) for sid in SCENARIOS for index in (0, 50)]


@pytest.mark.parametrize("sid,index", CASES)
def test_the_spec_has_the_six_members_and_is_plain_json(sid, index):
    spec = spec_of(sidecar(sid, index), sid)
    assert tuple(spec) == episode_spec.SPEC_MEMBERS
    assert json.loads(json.dumps(spec)) == spec


@pytest.mark.parametrize("sid,index", CASES)
def test_the_tasks_stages_are_the_paths_stages(sid, index):
    meta = sidecar(sid, index)
    spec = spec_of(meta, sid)
    kind = sites.load_site(_entry(sid)).path["kind"]
    assert [s["name"] for s in spec["task"]["stages"]] == list(task_path.STAGE_NAMES[kind])
    row = {k: v for k, v in meta["randomization"]["condition"].items() if k not in episode_spec.CONDITION_DERIVED}
    site = conditions.row_site(sites.load_site(_entry(sid)), row)
    assert spec["task"]["stages"] == json.loads(json.dumps(task_path.stage_plan(site, row)))


@pytest.mark.parametrize("sid,index", CASES)
def test_the_condition_values_are_the_columns_and_fixed_values_of_the_canon(sid, index):
    spec = spec_of(sidecar(sid, index), sid)
    cond = _entry(sid)["conditions"]
    assert set(spec["condition"]["values"]) == set(cond["columns"]) | set(cond.get("fixed", {}))
    assert not set(spec["condition"]["values"]) & set(episode_spec.CONDITION_DERIVED)


@pytest.mark.parametrize("sid", SCENARIOS)
def test_a_grid_row_says_so_and_names_its_cell(sid):
    spec = spec_of(sidecar(sid, 50), sid)
    design, cell = conditions.row_design(_entry(sid), 50)
    assert design == "grid" and cell is not None
    assert spec["condition"]["design"] == "grid" and spec["condition"]["grid_cell"] == list(cell)
    assert spec["condition"]["row"] == 50
    random = spec_of(sidecar(sid, 0), sid)["condition"]
    assert random["design"] == "random" and random["grid_cell"] is None


@pytest.mark.parametrize("sid,index", CASES)
def test_the_world_names_the_assets_the_stage_and_the_look(sid, index):
    meta = sidecar(sid, index)
    spec = spec_of(meta, sid)
    keys = [assets_lib.asset_entry(_canon_assets(), p)["key"] for p in meta["asset_paths"].values()]
    assert spec["world"]["assets"] == keys and spec["world"]["scenario_id"] == sid
    composed = [k for k, e in _canon_assets()["scene"].items() if e.get("composes")]
    assert composed and spec["world"]["stage"] == composed[0]
    assert spec["world"]["appearance"] == APPEARANCE


@pytest.mark.parametrize("sid", SCENARIOS)
def test_the_payload_is_the_box_asset_with_the_rows_mass_and_size(sid):
    meta = sidecar(sid, 0)
    spec = spec_of(meta, sid)
    row = spec["condition"]["values"]
    box = assets_lib.asset_entry(_canon_assets(), meta["asset_paths"]["box"])["key"]
    assert spec["task"]["payload"] == {"asset": box, "mass_kg": row["box_mass_kg"],
                                       "size_m": list(conditions.box_size(row))}
    no_carry = {k: v for k, v in _entry(sid).items() if k != "carry"}
    plain = episode_spec.build_episode_spec(meta, no_carry, _canon_assets(), dof=DOF)
    assert plain["task"]["payload"] is None


def test_the_human_the_task_and_the_robot_come_from_the_canon_and_the_sidecar():
    sid, meta = "S1g", sidecar("S1g", 0)
    spec, entry = spec_of(meta, "S1g"), _entry("S1g")
    condition = meta["randomization"]["condition"]
    assert spec["humans"] == [{"model": entry["scene"]["human_model"], "stature_m": 1.74,
                               "policy": {"name": episode_spec.HUMAN_POLICY_NAME, "version": COMMIT},
                               "script": condition["human_plan"]}]
    task = spec["task"]
    assert task["archetype"] == entry["task_archetype"] and task["instruction"] == entry["robot_task"]
    assert (task["start"], task["goal"], task["path"]) == (
        condition["robot_start"], condition["robot_goal"], condition["robot_path"])
    assert task["time_limit_s"] == 40.0
    assert spec["robot"] == {"model": "unitree_h1", "dof": DOF, "policy": POLICY}


def test_the_repro_names_the_command_that_reruns_the_episode():
    meta = sidecar("S1f", 7)
    spec = spec_of(meta, "S1f")
    assert spec["repro"]["command"] == "generation/rerun_seed.sh S1f_0007_seed1234 1234 7"
    assert episode_spec.rerun_command(meta) == spec["repro"]["command"]
    assert {k: spec["repro"][k] for k in ("generator_commit", "seed", "run_start_seq", "episode_seq")} == {
        "generator_commit": COMMIT, "seed": 1234, "run_start_seq": 7, "episode_seq": 7}


def test_the_derived_keys_are_not_in_the_condition_values():
    spec = spec_of(sidecar("S1e", 0), "S1e")
    assert not set(episode_spec.CONDITION_DERIVED) & set(spec["condition"]["values"])
    assert "row" not in spec["condition"]["values"]


def _without(path, sid="S1e"):
    meta = sidecar(sid, 0)
    node = meta
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    return meta


@pytest.mark.parametrize("path", [
    ("task_stages",), ("human_stature_m",), ("task_time_limit_s",), ("seed",), ("run_start_seq",),
    ("episode_seq",), ("episode_id",), ("robot_model",), ("asset_paths",), ("sim_backend", "generator_commit"),
    ("sim_backend", "policy"), ("randomization", "condition"), ("randomization", "appearance"),
    ("randomization", "condition", "human_plan"), ("randomization", "condition", "robot_path"),
    ("randomization", "condition", "robot_speed_mps"), ("randomization", "condition", "design"),
])
def test_a_sidecar_missing_a_key_is_refused_naming_it(path):
    with pytest.raises(ValueError, match=path[-1]):
        spec_of(_without(path), "S1e")


def test_a_time_limit_that_is_not_a_number_is_refused():
    meta = sidecar("S1e", 0)
    meta["task_time_limit_s"] = True
    with pytest.raises(ValueError, match="task_time_limit_s"):
        spec_of(meta, "S1e")


def test_stage_names_that_differ_from_the_recorded_ones_are_refused():
    meta = sidecar("S1e", 0)
    meta["task_stages"][1]["name"] = "swerve"
    with pytest.raises(ValueError, match="swerve"):
        spec_of(meta, "S1e")
    meta = sidecar("S1e", 0)
    del meta["task_stages"][-1]
    with pytest.raises(ValueError, match="task_stages"):
        spec_of(meta, "S1e")


def test_an_asset_the_canon_does_not_know_is_refused():
    meta = sidecar("S1e", 0)
    meta["asset_paths"]["box"] = "generation/scene/build/unknown.usd"
    with pytest.raises(ValueError):
        spec_of(meta, "S1e")


def test_a_composed_entry_whose_members_are_not_all_there_is_not_the_stage():
    assets = _assets()
    members = next(e["composes"] for e in assets["scene"].values() if e.get("composes"))
    assets["scene"]["other_stage"] = {**copy.deepcopy(assets["scene"]["keti_hri_stage_v1"]),
                                      "usd_path": "generation/scene/build/other_stage.usd",
                                      "composes": [*members, "isaaclab_terrain_plane_missing"]}
    assert spec_of(sidecar("S1e", 0), "S1e", assets)["world"]["stage"] == "keti_hri_stage_v1"


def test_no_stage_the_episode_assets_cover_gives_null():
    meta = sidecar("S1e", 0)
    del meta["asset_paths"]["box"]
    no_carry = {k: v for k, v in _entry("S1e").items() if k != "carry"}
    spec = episode_spec.build_episode_spec(meta, no_carry, _canon_assets(), dof=DOF)
    assert spec["world"]["stage"] is None


def test_a_scenario_that_carries_a_box_needs_the_box_among_the_assets():
    meta = sidecar("S1e", 0)
    del meta["asset_paths"]["box"]
    with pytest.raises(ValueError, match="box"):
        spec_of(meta, "S1e")


def test_two_stages_the_assets_cover_are_refused():
    assets = _assets()
    assets["scene"]["twin_stage"] = {**copy.deepcopy(assets["scene"]["keti_hri_stage_v1"]),
                                     "usd_path": "generation/scene/build/twin_stage.usd"}
    with pytest.raises(ValueError, match="twin_stage"):
        spec_of(sidecar("S1e", 0), "S1e", assets)
