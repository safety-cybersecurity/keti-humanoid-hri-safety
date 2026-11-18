"""What a scenario says it randomises, against what its environment config resolves to.

A declaration written by hand can name axes the environment never draws. generation/randomization.py
reads the declaration off the resolved config, and generation/envs/s1a/check_randomization.py
runs it on the real one (Isaac).

What these tests cannot see: the real config. CI has no Isaac, so the config here is a stand-in -- the
shape S1aEnvCfg resolved to on 2026-10-03 (Isaac Lab v3.0.0-beta2.patch1), typed once below, and the
shape HriEnvCfg gives it for the scenarios with a conditions table (S1e, S1f, S1g), built at row 0. The
canon is held to them, so a canon edited by hand goes red here; a config that changed under an unchanged
canon does not, and only the Isaac script sees that. When the script's --write changes the canon,
this stand-in is updated to the config that made it change -- and the batch generated before that is
no longer described by the canon.
"""

import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from generation import carry, conditions, human_script, randomization, sites
from generation.envs.s1d import dummy_reset
from generation.randomization import DISTRIBUTION, FIXED, NOISE, SAMPLED
from generation.scene import layout

REPO = Path(__file__).resolve().parents[2]
IMPLEMENTED = ("S1a", "S1d")
TABLE = conditions.table_scenarios()


def term(mode="reset", func=None, **params):
    return NS(mode=mode, func=func, params=params)


def runs(name):
    """A stand-in for the function an event term runs, under its name -- what randomization.FROM_THE_ROW reads."""
    def function(env, env_ids, **params):
        raise AssertionError("a stand-in term is read, never run")
    function.__name__ = function.__qualname__ = name
    return function


def noisy(width):
    return NS(func=len, noise=NS(n_min=-width, n_max=width))


def stand_in():
    events = NS(
        physics_material=term("startup", static_friction_range=(0.8, 0.8), dynamic_friction_range=(0.6, 0.6),
                              restitution_range=(0.0, 0.0), num_buckets=64),
        add_base_mass=term("reset", mass_distribution_params=(1 / 1.25, 1.25), operation="scale",
                           distribution="log_uniform"),
        base_com=None,
        base_external_force_torque=term("reset", force_range=(0.0, 0.0), torque_range=(-0.0, 0.0)),
        reset_base=term("reset",
                        pose_range={"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0.0, 0.0)},
                        velocity_range={"x": (-0.5, 0.5), "y": (0.0, 0.0), "z": (-0.5, 0.5),
                                        "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (0.0, 0.0)}),
        reset_robot_joints=term("reset", position_range=(1.0, 1.0), velocity_range=(0.0, 0.0)),
        push_robot=None,
        reset_dummy_entry=term("reset", speed_range=(0.4, 1.6), angle_range_deg=(30.0, 150.0),
                               trigger_distance_range=(1.0, 4.0), stop_band_clearance=1.2894,
                               spawn_margin_x=0.5, face_travel=True),
        aisle_guard=term("reset", rack_inner_y=5.0, margin=1.0),
    )
    commands = NS(base_velocity=NS(walk_speed_range=(0.3, 1.2), sensor_delay_ms_range=(0.0, 100.0),
                                   slowdown_factor=2.0, stop_band_factor=1.05))
    policy = NS(enable_corruption=True, concatenate_terms=True,
                base_lin_vel=noisy(0.1), base_ang_vel=noisy(0.2), projected_gravity=noisy(0.05),
                velocity_commands=NS(func=len, noise=None), joint_pos=noisy(0.01), joint_vel=noisy(1.5),
                actions=NS(func=len, noise=None), height_scan=None)
    scene = NS(terrain=NS(physics_material=NS(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)))
    return NS(events=events, commands=commands, observations=NS(policy=policy), scene=scene)


def table_stand_in(scenario):
    """The shape HriEnvCfg gives the S1d config of ``scenario`` at row 0 (generation/envs/hri/hri_env_cfg.py): the
    robot reset to the row's start, no base-mass draw, the row guard in the place of the aisle guard, the person's
    script in the place of the drawn entry, the box authored and put back on the torso where the scenario carries
    one, and the task's command term carrying the row."""
    entry = conditions.scenario_entry(scenario)
    row = conditions.draw_row(entry, 0, layout.build_layout(layout.load_spec(layout.SCENE_SPEC)))
    start = conditions.row_site(sites.load_site(entry), row).robot_start
    cfg = stand_in()
    events = cfg.events
    events.reset_base = term("reset", runs("reset_robot_to_site"), x_m=start.x, y_m=start.y, yaw_deg=start.yaw_deg)
    events.add_base_mass = None
    events.aisle_guard = None
    events.row_guard = term("reset", runs("check_row_after_reset"), scenario_id=scenario, row=dict(row))
    events.reset_dummy_entry = term("reset", dummy_reset.stand_up_first(runs("reset_dummy_scripted")),
                                    plan=human_script.plan_record(human_script.plan_for(entry, row)))
    if carry.load_carry(entry) is not None:
        events.carried_box = term("prestartup", runs("author_carried_box"), scenario_id=scenario,
                                  size_m=conditions.box_size(row), mass_kg=float(row["box_mass_kg"]))
        events.carried_box_place = term("reset", runs("place_carried_box"))
    command = cfg.commands.base_velocity
    command.walk_speed_range = (0.0, 0.0)
    command.scenario_id = scenario
    command.row = dict(row)
    return cfg


def canon_entry(scenario):
    canon = json.loads((REPO / "meta/scenarios.json").read_text(encoding="utf-8"))
    return next(s for s in canon["scenarios"] if s["scenario_id"] == scenario)


def agreeing_entry():
    """An entry that says what the stand-in draws -- the starting point of every case that breaks one thing."""
    blocks, unread = randomization.read(stand_in())
    assert unread == []
    return randomization.apply({"scenario_id": "S9z", SAMPLED: {}}, blocks)


# ── the canon ──

@pytest.mark.parametrize("scenario", IMPLEMENTED)
def test_the_canon_says_what_the_recorded_config_shape_draws(scenario):
    assert randomization.problems(canon_entry(scenario), stand_in()) == []


def test_the_two_implemented_scenarios_declare_the_same_blocks():
    # S1dEnvCfg inherits S1aEnvCfg and overrides none of the draws.
    a, b = (randomization.declared(canon_entry(s))[0] for s in IMPLEMENTED)
    assert a == b


@pytest.mark.parametrize("scenario", IMPLEMENTED)
def test_the_reaction_time_is_the_upper_bound_of_the_declared_sensor_delay(scenario):
    # meta/rules.json says robot_reaction_time comes from that bound; it is baked into the corridor.
    rules = json.loads((REPO / "meta/rules.json").read_text(encoding="utf-8"))
    ssm = next(r for r in rules["rules"] if r["rule_id"] == "R-TS15066-SSM-01")
    bound_ms = canon_entry(scenario)[SAMPLED]["sensor_delay_ms"][1]
    assert ssm["parameters"]["robot_reaction_time"]["value"] == pytest.approx(bound_ms / 1000.0)


def test_the_crossing_rate_tool_still_finds_its_ranges():
    # measure_crossing_rate.py reads four ranges out of S1a's block by name.
    from generation.envs.s1a.measure_crossing_rate import load_s1a_randomization

    ranges = load_s1a_randomization()
    block = canon_entry("S1a")[SAMPLED]
    assert ranges == {
        "robot_walk_speed_range": tuple(block["robot_walk_speed"]),
        "human_entry_speed_range": tuple(block["human_entry_speed"]),
        "angle_range_deg": tuple(block["human_entry_angle_deg"]),
        "trigger_distance_range": tuple(block["entry_trigger_distance"]),
    }


# ── reading a config ──

def test_reading_the_stand_in_names_every_draw_and_holds_zero_width_ranges_fixed():
    blocks, unread = randomization.read(stand_in())
    assert unread == []
    assert blocks[SAMPLED] == {
        "robot_base_mass_scale": [0.8, 1.25],
        "robot_start_offset_x": [-0.5, 0.5],
        "robot_start_offset_y": [-0.5, 0.5],
        "robot_start_velocity_x": [-0.5, 0.5],
        "robot_start_velocity_z": [-0.5, 0.5],
        "robot_start_roll_rate": [-0.5, 0.5],
        "robot_start_pitch_rate": [-0.5, 0.5],
        "human_entry_speed": [0.4, 1.6],
        "human_entry_angle_deg": [30.0, 150.0],
        "entry_trigger_distance": [1.0, 4.0],
        "robot_walk_speed": [0.3, 1.2],
        "sensor_delay_ms": [0.0, 100.0],
    }
    assert blocks[DISTRIBUTION] == {"robot_base_mass_scale": "log_uniform"}
    assert blocks[NOISE] == {"base_lin_vel": 0.1, "base_ang_vel": 0.2, "projected_gravity": 0.05,
                             "joint_pos": 0.01, "joint_vel": 1.5}
    fixed = blocks[FIXED]
    assert (fixed["robot_static_friction"], fixed["robot_dynamic_friction"]) == (0.8, 0.6)
    assert (fixed["terrain_static_friction"], fixed["terrain_dynamic_friction"]) == (1.0, 1.0)
    # The draws S1aEnvCfg switched off are values, not axes.
    assert (fixed["robot_start_yaw"], fixed["robot_start_velocity_y"], fixed["robot_start_yaw_rate"]) == (0, 0, 0)
    assert fixed["robot_start_joint_position_scale"] == 1.0
    assert not set(fixed) & set(blocks[SAMPLED])


def test_a_term_that_checks_and_draws_nothing_is_not_read():
    cfg = stand_in()
    cfg.events.aisle_guard = term("reset", speed_range=(0.0, 9.0))
    assert randomization.read(cfg) == randomization.read(stand_in())


def test_an_event_term_nobody_wrote_a_reader_for_is_a_problem():
    cfg = stand_in()
    cfg.events.shove_the_dummy = term("reset", force_range=(0.0, 50.0))
    found = randomization.problems(agreeing_entry(), cfg)
    assert len(found) == 1 and "events.shove_the_dummy" in found[0] and "읽을 줄 모르는" in found[0]


def test_a_command_term_nobody_wrote_a_reader_for_is_a_problem():
    cfg = stand_in()
    cfg.commands.arm_swing = NS(amplitude_range=(0.0, 1.0))
    found = randomization.problems(agreeing_entry(), cfg)
    assert len(found) == 1 and "commands.arm_swing" in found[0]


def test_a_range_that_cannot_be_read_is_a_problem():
    cfg = stand_in()
    cfg.commands.base_velocity.walk_speed_range = None
    found = randomization.problems(agreeing_entry(), cfg)
    assert any("robot_walk_speed 의 범위를 읽지 못했습니다" in p for p in found)


def test_a_flag_is_not_a_bound():
    cfg = stand_in()
    cfg.commands.base_velocity.walk_speed_range = (False, True)
    assert any("robot_walk_speed 의 범위를 읽지 못했습니다" in p for p in randomization.read(cfg)[1])


def test_a_mass_event_that_does_not_scale_is_refused():
    cfg = stand_in()
    cfg.events.add_base_mass.params["operation"] = "add"
    assert any("operation='add'" in p for p in randomization.read(cfg)[1])


def test_a_draw_made_once_per_process_is_not_a_draw_per_episode():
    # mode="startup" is how every episode of a run once shared one base mass.
    cfg = stand_in()
    cfg.events.add_base_mass.mode = "startup"
    found = randomization.problems(agreeing_entry(), cfg)
    assert len(found) == 1 and "robot_base_mass_scale" in found[0] and "mode=startup" in found[0]


def test_a_value_held_fixed_at_startup_is_not_a_problem():
    # physics_material stays at startup: its ranges have no width, so there is nothing to draw again.
    assert randomization.read(stand_in())[1] == []


def test_noise_that_is_not_uniform_is_a_problem():
    cfg = stand_in()
    cfg.observations.policy.joint_pos = NS(func=len, noise=NS(mean=0.0, std=0.01))
    assert any("observations.policy.joint_pos" in p and "균등 잡음" in p for p in randomization.read(cfg)[1])


def test_noise_that_is_not_centred_is_a_problem():
    cfg = stand_in()
    cfg.observations.policy.joint_pos = NS(func=len, noise=NS(n_min=-0.01, n_max=0.02))
    assert any("observations.policy.joint_pos" in p and "대칭" in p for p in randomization.read(cfg)[1])


# ── a canon that disagrees with the config ──

def test_a_declaration_the_config_does_not_draw_is_reported():
    # What the canon once said, against the config it was said about.
    old = {"scenario_id": "S1d", SAMPLED: {
        "human_entry_speed": [0.4, 1.6], "human_entry_angle_deg": [30, 150],
        "entry_trigger_distance": [1.0, 4.0], "robot_walk_speed": [0.3, 1.2],
        "floor_friction": [0.3, 1.0], "robot_mass_deviation": [-0.05, 0.05], "sensor_delay_ms": [0, 100]}}
    found = randomization.problems(old, stand_in())
    assert "floor_friction: 정본은 [0.3, 1.0] 에서 뽑는다고 적었는데 환경에는 그 축이 없습니다" in found
    assert "robot_mass_deviation: 정본은 [-0.05, 0.05] 에서 뽑는다고 적었는데 환경에는 그 축이 없습니다" in found
    assert "robot_base_mass_scale: 환경은 [0.8, 1.25] 에서 뽑는데 정본의 randomization 에 없습니다" in found


def one_problem(entry, cfg=None):
    found = randomization.problems(entry, cfg or stand_in())
    assert len(found) == 1, found
    return found[0]


def test_a_range_that_differs_is_reported():
    entry = agreeing_entry()
    entry[SAMPLED]["robot_base_mass_scale"] = [0.95, 1.05]
    assert one_problem(entry) == "robot_base_mass_scale: 범위가 다릅니다 — 정본 [0.95, 1.05] · 환경 [0.8, 1.25]"


def test_an_axis_the_canon_leaves_out_is_reported():
    entry = agreeing_entry()
    del entry[SAMPLED]["robot_start_offset_x"]
    assert "robot_start_offset_x: 환경은 [-0.5, 0.5] 에서 뽑는데" in one_problem(entry)


def test_an_axis_the_config_started_to_draw_is_reported():
    cfg = stand_in()
    cfg.events.push_robot = NS(mode="interval", params={"velocity_range": {"x": (-0.5, 0.5)}})
    assert "robot_push_velocity_x: 환경은 [-0.5, 0.5] 에서 뽑는데" in one_problem(agreeing_entry(), cfg)


def test_an_axis_the_canon_draws_and_the_config_holds_fixed_is_reported():
    entry = agreeing_entry()
    entry[SAMPLED]["robot_static_friction"] = [0.3, 1.0]
    del entry[FIXED]["robot_static_friction"]
    assert one_problem(entry) == "robot_static_friction: 정본은 [0.3, 1.0] 에서 뽑는다고 적었는데 환경은 0.8 로 고정합니다"


def test_a_value_the_canon_holds_fixed_and_the_config_draws_is_reported():
    cfg = stand_in()
    cfg.events.physics_material = term("reset", static_friction_range=(0.3, 1.0),
                                       dynamic_friction_range=(0.6, 0.6), restitution_range=(0.0, 0.0))
    found = randomization.problems(agreeing_entry(), cfg)
    assert "robot_static_friction: 정본은 0.8 로 고정이라 적었는데 환경은 [0.3, 1.0] 에서 뽑습니다" in found


def test_a_fixed_value_that_differs_is_reported():
    entry = agreeing_entry()
    entry[FIXED]["terrain_static_friction"] = 0.7
    assert one_problem(entry) == "terrain_static_friction: 고정값이 다릅니다 — 정본 0.7 · 환경 1.0"


def test_a_fixed_name_the_config_does_not_have_is_reported():
    entry = agreeing_entry()
    entry[FIXED]["floor_friction"] = 0.8
    assert "floor_friction" in one_problem(entry) and "읽지 못한 이름" in one_problem(entry)


def test_the_canon_need_not_list_everything_the_config_holds_fixed():
    entry = agreeing_entry()
    assert "robot_start_yaw" not in entry[FIXED]
    assert randomization.problems(entry, stand_in()) == []


def test_a_distribution_that_differs_is_reported():
    entry = agreeing_entry()
    del entry[DISTRIBUTION]
    assert one_problem(entry) == "robot_base_mass_scale: 분포가 다릅니다 — 정본 uniform · 환경 log_uniform"


def test_a_distribution_named_for_an_axis_that_is_not_drawn_is_reported():
    entry = agreeing_entry()
    entry[DISTRIBUTION]["floor_friction"] = "log_uniform"
    assert "randomization_distribution.floor_friction" in one_problem(entry)


def test_noise_the_canon_leaves_out_is_reported():
    entry = agreeing_entry()
    del entry[NOISE]["joint_vel"]
    assert one_problem(entry) == "policy_observation_noise.joint_vel: 환경은 ±1.5 를 더하는데 정본에 없습니다"


def test_noise_declared_for_a_policy_that_is_shown_clean_observations_is_reported():
    cfg = stand_in()
    cfg.observations.policy.enable_corruption = False
    found = randomization.problems(agreeing_entry(), cfg)
    assert len(found) == 5 and all("잡음을 더하지 않습니다" in p for p in found)


def test_a_noise_width_that_differs_is_reported():
    entry = agreeing_entry()
    entry[NOISE]["joint_pos"] = 0.02
    assert one_problem(entry) == "policy_observation_noise.joint_pos: 반폭이 다릅니다 — 정본 ±0.02 · 환경 ±0.01"


@pytest.mark.parametrize("bad", [None, [0.3], [1.0, 0.3], [0.5, 0.5], "0.3-1.0", [True, 1.0], [0.3, float("nan")]])
def test_a_declared_range_that_is_not_a_low_and_a_high_is_reported(bad):
    # S3b carries "floor_friction": null -- a plan, not a range; an implemented scenario may not.
    entry = agreeing_entry()
    entry[SAMPLED]["robot_walk_speed"] = bad
    found = randomization.problems(entry, stand_in())
    assert any(p.startswith("randomization.robot_walk_speed: [하한, 상한] 이 아닙니다") for p in found), found


@pytest.mark.parametrize("block", [FIXED, NOISE])
def test_a_declared_value_that_is_not_a_number_is_reported(block):
    entry = agreeing_entry()
    name = next(iter(entry[block]))
    entry[block][name] = "0.8"
    assert any(p.startswith(f"{block}.{name}: 수가 아닙니다") for p in randomization.problems(entry, stand_in()))


def test_a_block_that_is_not_an_object_is_reported():
    entry = agreeing_entry()
    entry[NOISE] = [0.1, 0.2]
    assert any(p.startswith("policy_observation_noise: 객체가 아닙니다") for p in randomization.problems(entry, stand_in()))


# ── writing the blocks ──

OLD_ENTRY = {
    "scenario_id": "S1d",
    "human_behavior": "…",
    SAMPLED: {"human_entry_speed": [0.4, 1.6], "human_entry_angle_deg": [30, 150],
              "entry_trigger_distance": [1.0, 4.0], "robot_walk_speed": [0.3, 1.2],
              "floor_friction": [0.3, 1.0], "robot_mass_deviation": [-0.05, 0.05], "sensor_delay_ms": [0, 100]},
    "randomization_note": "…",
    "rules": ["R-TS15066-SSM-01"],
}


def written():
    return randomization.apply(OLD_ENTRY, randomization.read(stand_in())[0])


def test_writing_the_blocks_leaves_nothing_to_report():
    assert randomization.problems(written(), stand_in()) == []


def test_writing_drops_what_is_not_drawn_and_keeps_the_order_and_the_spelling_of_what_is():
    block = written()[SAMPLED]
    assert list(block)[:5] == ["human_entry_speed", "human_entry_angle_deg", "entry_trigger_distance",
                               "robot_walk_speed", "sensor_delay_ms"]
    assert "floor_friction" not in block and "robot_mass_deviation" not in block
    # 30 stays 30: the file's bytes do not change where its numbers did not.
    assert json.dumps(block["human_entry_angle_deg"]) == "[30, 150]"
    assert json.dumps(block["sensor_delay_ms"]) == "[0, 100]"


def test_writing_puts_the_new_blocks_beside_the_drawn_axes_and_touches_nothing_else():
    out = written()
    assert list(out) == ["scenario_id", "human_behavior", SAMPLED, DISTRIBUTION, FIXED, NOISE,
                         "randomization_note", "rules"]
    assert {k: out[k] for k in OLD_ENTRY if k != SAMPLED} == {k: v for k, v in OLD_ENTRY.items() if k != SAMPLED}
    assert OLD_ENTRY[SAMPLED]["floor_friction"] == [0.3, 1.0], "apply must not edit the entry it was given"


def test_a_new_fixed_block_holds_the_values_a_reader_may_expect_to_vary():
    assert written()[FIXED] == {"robot_static_friction": 0.8, "robot_dynamic_friction": 0.6,
                                "terrain_static_friction": 1.0, "terrain_dynamic_friction": 1.0}


def test_an_existing_fixed_block_keeps_its_own_names():
    entry = copy.deepcopy(OLD_ENTRY)
    entry[FIXED] = {"robot_restitution": 0.5, "robot_start_offset_x": 0.0}
    out = randomization.apply(entry, randomization.read(stand_in())[0])
    # The value is the config's; the axis that is drawn leaves the fixed block.
    assert out[FIXED] == {"robot_restitution": 0.0}


def test_writing_twice_changes_nothing():
    blocks = randomization.read(stand_in())[0]
    once = randomization.apply(OLD_ENTRY, blocks)
    assert json.dumps(randomization.apply(once, blocks)) == json.dumps(once)


def test_a_block_with_nothing_in_it_is_not_added_and_one_the_entry_has_is_kept():
    cfg = stand_in()
    cfg.observations.policy.enable_corruption = False
    cfg.events.add_base_mass.params["distribution"] = "uniform"
    blocks = randomization.read(cfg)[0]
    fresh = randomization.apply(OLD_ENTRY, blocks)
    assert NOISE not in fresh and DISTRIBUTION not in fresh and SAMPLED in fresh
    kept = randomization.apply(written(), blocks)
    assert kept[NOISE] == {} and kept[DISTRIBUTION] == {}
    assert randomization.problems(kept, cfg) == []


# ── the scenarios with a conditions table ──

@pytest.mark.parametrize("scenario", TABLE)
def test_a_table_scenario_s_canon_says_what_its_config_draws(scenario):
    entry = canon_entry(scenario)
    missing = [block for block in randomization.BLOCKS if block not in entry]
    assert not missing, f"{scenario}: 블록이 없습니다 {missing} — check_randomization.py --write 로 씁니다"
    assert randomization.problems(entry, table_stand_in(scenario)) == []


@pytest.mark.parametrize("scenario", TABLE)
def test_a_table_scenario_draws_nothing_per_episode_outside_its_row(scenario):
    # What varies from one of its episodes to the next is the row (generation/conditions.py), drawn before the
    # environment exists; its columns are not ranges an episode is drawn from.
    blocks, unread = randomization.read(table_stand_in(scenario))
    assert unread == []
    assert blocks[SAMPLED] == {} and blocks[DISTRIBUTION] == {}
    # The robot, its materials and what its walking policy is shown stay S1d's.
    s1d = randomization.read(stand_in())[0]
    assert blocks[NOISE] == s1d[NOISE]
    assert {n: blocks[FIXED][n] for n in randomization.DEFAULT_FIXED} == {n: s1d[FIXED][n] for n in randomization.DEFAULT_FIXED}


def test_a_term_is_the_row_s_by_the_function_it_runs_not_by_its_name():
    # The table scenarios keep the inherited names reset_base and reset_dummy_entry, so the reset order stays, and
    # swap the functions. The same name running the inherited draw is read as one.
    cfg = table_stand_in("S1f")
    cfg.events.reset_base = stand_in().events.reset_base
    blocks, unread = randomization.read(cfg)
    assert unread == []
    assert blocks[SAMPLED]["robot_start_offset_x"] == [-0.5, 0.5]

    cfg = table_stand_in("S1f")
    cfg.events.reset_dummy_entry = stand_in().events.reset_dummy_entry
    assert randomization.read(cfg)[0][SAMPLED]["human_entry_speed"] == [0.4, 1.6]


def test_a_term_that_puts_the_box_back_or_guards_the_row_draws_nothing():
    cfg = table_stand_in("S1e")
    cfg.events.row_guard.params["speed_range"] = (0.0, 9.0)
    cfg.events.carried_box_place.params["speed_range"] = (0.0, 9.0)
    blocks, unread = randomization.read(cfg)
    assert unread == []
    assert (blocks, unread) == randomization.read(table_stand_in("S1e"))


def test_a_table_command_takes_its_speed_and_delay_from_the_row():
    # HriTaskCommand sets both from the row at every reset, over the inherited term's draws; the inherited
    # sensor_delay_ms_range is still on the config and no episode is drawn from it.
    cfg = table_stand_in("S1e")
    assert cfg.commands.base_velocity.sensor_delay_ms_range == (0.0, 100.0)
    blocks, unread = randomization.read(cfg)
    assert unread == []
    assert "sensor_delay_ms" not in blocks[SAMPLED] and "robot_walk_speed" not in blocks[FIXED]
    # The same term without a row is S1a's, and both are read.
    del cfg.commands.base_velocity.row
    blocks = randomization.read(cfg)[0]
    assert blocks[SAMPLED]["sensor_delay_ms"] == [0.0, 100.0] and blocks[FIXED]["robot_walk_speed"] == 0.0


def test_the_terms_read_as_drawing_nothing_are_the_ones_the_environments_set():
    # Matched by name, a rename in an environment leaves the term to a reader that refuses it -- loud in the Isaac
    # script, which CI does not run -- so the names are pinned to the lines that set them.
    def source(path):
        return (REPO / path).read_text(encoding="utf-8")

    hri, rig = "generation/envs/hri/hri_env_cfg.py", "generation/envs/hri/carry_rig.py"
    by_function = {
        "reset_robot_to_site": (hri, r"self\.events\.reset_base = EventTerm\(func=reset_robot_to_site\b"),
        "reset_dummy_scripted": (hri, r"self\.events\.reset_dummy_entry = EventTerm\(func=dummy_reset\.stand_up_first\("
                                      r"reset_dummy_scripted\)"),
        "author_carried_box": (rig, r"cfg\.events\.carried_box = EventTerm\(func=author_carried_box\b"),
    }
    assert set(by_function) == set(randomization.FROM_THE_ROW)
    for path, line in by_function.values():
        assert len(re.findall(line, source(path))) == 1, (path, line)
    defined = {"reset_robot_to_site": hri, "reset_dummy_scripted": "generation/envs/s1a/mdp/events.py",
               "author_carried_box": rig}
    for name, path in defined.items():
        assert re.search(rf"^def {name}\(", source(path), re.M), (name, path)

    by_name = {
        "aisle_guard": ("generation/envs/s1d/s1d_env_cfg.py", r"self\.events\.aisle_guard = EventTerm\("),
        "row_guard": (hri, r"self\.events\.row_guard = EventTerm\(func=check_row_after_reset\b"),
        "carried_box_place": (rig, r"cfg\.events\.carried_box_place = EventTerm\(func=place_carried_box, mode=\"reset\", "
                                   r"params=\{\}\)"),
    }
    assert set(by_name) == set(randomization.NOT_A_DRAW)
    for path, line in by_name.values():
        assert len(re.findall(line, source(path))) == 1, (path, line)

    # The command: the row is a field of its config, and the row's speed and delay are set over the draws.
    commands = source("generation/envs/hri/commands.py")
    assert re.search(r"^    row: dict = MISSING$", commands, re.M)
    delay = r"self\._delay_steps\[env_ids\] = behavior_layer\.delay_steps\(float\(self\.row\[\"sensor_delay_ms\"\]\)"
    assert len(re.findall(delay, commands)) == 1
    assert len(re.findall(r"self\._walk_speed\[env_ids\] = float\(self\.row\.get\(", commands)) == 1


def test_a_table_entry_gets_all_four_blocks_right_after_its_table():
    # Empty ones too: an empty randomization says nothing beside the row is drawn per episode -- left out, it
    # reads as a block nobody wrote.
    entry = {"scenario_id": "S1e", "scene": {}, "conditions": {"rows": 1}, "robot_task": "…", "rules": []}
    blocks = randomization.read(table_stand_in("S1e"))[0]
    out = randomization.apply(entry, blocks)
    assert list(out) == ["scenario_id", "scene", "conditions", SAMPLED, DISTRIBUTION, FIXED, NOISE, "robot_task",
                         "rules"]
    assert out[SAMPLED] == {} and out[DISTRIBUTION] == {}
    assert out[FIXED] == {"robot_static_friction": 0.8, "robot_dynamic_friction": 0.6,
                          "terrain_static_friction": 1.0, "terrain_dynamic_friction": 1.0}
    assert randomization.problems(out, table_stand_in("S1e")) == []
    assert json.dumps(randomization.apply(out, blocks)) == json.dumps(out)
    after, found = randomization.rewritten(entry, table_stand_in("S1e"))
    assert found == [] and after == out


def test_an_entry_without_a_table_still_gets_only_the_blocks_with_something_in_them():
    entry = {"scenario_id": "S9z", "robot_task": "…", "rules": []}
    out = randomization.apply(entry, randomization.read(table_stand_in("S1e"))[0])
    assert list(out) == ["scenario_id", "robot_task", "rules", FIXED, NOISE]


# ── less common inputs ──

@pytest.mark.parametrize("event, key", [("reset_base", "pose_range"), ("reset_base", "velocity_range")])
def test_an_event_without_a_table_it_cannot_run_without_is_a_problem(event, key):
    # Read as an empty table, the axes it held vanished from both sides and nothing was said.
    cfg = stand_in()
    del getattr(cfg.events, event).params[key]
    found = randomization.read(cfg)[1]
    assert any(p.startswith(f"events.{event}: {key} ") for p in found), found
    assert randomization.problems(agreeing_entry(), cfg) != []


def test_an_axis_a_table_leaves_out_is_held_at_zero():
    # Isaac Lab reads a missing axis as (0.0, 0.0): reset_root_state_uniform, push_by_setting_velocity
    # and randomize_rigid_body_com all use range.get(axis, (0.0, 0.0)).
    cfg = stand_in()
    del cfg.events.reset_base.params["pose_range"]["yaw"]
    blocks, unread = randomization.read(cfg)
    assert unread == []
    assert blocks[FIXED]["robot_start_yaw"] == 0.0
    assert blocks[FIXED]["robot_start_offset_z"] == 0.0, "an axis the stand-in never lists"
    entry = agreeing_entry()
    entry[FIXED]["robot_start_yaw"] = 0
    assert randomization.problems(entry, cfg) == []


@pytest.mark.parametrize("operation", ["scale", "abs"])
def test_noise_that_is_not_added_is_a_problem(operation):
    # The canon says a half-width that is added; U(-0.01, 0.01) multiplied in is another thing.
    cfg = stand_in()
    cfg.observations.policy.joint_pos.noise.operation = operation
    found = randomization.problems(agreeing_entry(), cfg)
    assert any("observations.policy.joint_pos" in p and operation in p for p in found), found


def test_a_noise_range_written_backwards_is_a_problem():
    cfg = stand_in()
    cfg.observations.policy.joint_pos.noise = NS(n_min=0.01, n_max=-0.01)
    blocks, unread = randomization.read(cfg)
    assert any("observations.policy.joint_pos" in p for p in unread), unread
    assert all(width > 0 for width in blocks[NOISE].values())


def test_an_empty_fixed_block_stays_the_entry_s_choice_however_often_it_is_written():
    # Dropped on the first write, the block came back with the default names on the second.
    blocks = randomization.read(stand_in())[0]
    entry = agreeing_entry()
    entry[FIXED] = {}
    once = randomization.apply(entry, blocks)
    assert once[FIXED] == {}
    assert json.dumps(randomization.apply(once, blocks)) == json.dumps(once)


def test_a_config_whose_terms_are_held_in_dicts_is_read_the_same():
    # Isaac Lab's managers take a dict of terms as well as a config object.
    cfg = stand_in()
    cfg.events = vars(cfg.events)
    cfg.commands = vars(cfg.commands)
    assert randomization.read(cfg) == randomization.read(stand_in())


def test_a_scenario_listed_twice_is_refused():
    # The checker took the last entry of an id and the converter takes the first.
    canon = {"scenarios": [{"scenario_id": "S1a", SAMPLED: {}}, {"scenario_id": "S1d"},
                           {"scenario_id": "S1a", SAMPLED: {"robot_walk_speed": [0.3, 1.2]}}]}
    entries, found = randomization.entries(canon)
    assert list(entries) == ["S1d"]
    assert len(found) == 1 and "S1a" in found[0] and "둘 이상" in found[0]


def test_the_entries_of_the_canon_are_found_by_id():
    canon = json.loads((REPO / "meta/scenarios.json").read_text(encoding="utf-8"))
    entries, found = randomization.entries(canon)
    assert found == [] and set(IMPLEMENTED) <= set(entries)
    assert entries["S1d"] is next(s for s in canon["scenarios"] if s["scenario_id"] == "S1d")


def test_what_would_be_written_is_checked_before_it_is_written():
    # rewritten() hands back the entry only when the result says what the config does.
    after, found = randomization.rewritten(OLD_ENTRY, stand_in())
    assert found == [] and randomization.problems(after, stand_in()) == []

    cfg = stand_in()
    cfg.events.shove_the_dummy = term("reset", force_range=(0.0, 50.0))
    after, found = randomization.rewritten(OLD_ENTRY, cfg)
    assert after is None and any("shove_the_dummy" in p for p in found)
