"""generation/envs/s1d/s1d_env_cfg.py's jointed dummy (use_jointed). The module imports Isaac, so _use_jointed_dummy
is taken out of its source and run on stand-ins: the asset, one implicit actuator per joint group at the walking gains
over that group's degrees of freedom by their exact names, the command term's gait settings, and the aisle guard's width
with the feet. And the command term declares those settings with defaults that are the rigid dummy's: the gait
off, its inertia model, no support share. Isaac-free."""
import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from generation.dummy import joints_spec as js
from generation.envs.s1d import aisle_guard

REPO = Path(__file__).resolve().parents[2]
CFG = REPO / "generation/envs/s1d/s1d_env_cfg.py"
COMMANDS = REPO / "generation/envs/s1a/mdp/commands.py"


def _class(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return tree, next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def _use_jointed_dummy():
    """S1dEnvCfg._use_jointed_dummy, compiled with the module's own path constants."""
    tree, cls = _class(CFG, "S1dEnvCfg")
    consts = [n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id in ("REPO", "V2_USD", "V2_SKIN_USD", "GAIT_SPEC",
                                                                                   "HOLD_SUPPORT_SHARE", "HOLD_PLANE_FREQ_HZ")]
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_use_jointed_dummy")
    ns = {"__file__": str(CFG), "Path": Path, "yaml": yaml, "aisle_guard": aisle_guard}
    exec(compile(ast.Module(body=consts + [fn], type_ignores=[]), str(CFG), "exec"), ns)
    return ns


@pytest.fixture
def fake_isaaclab(monkeypatch):
    pkg, mod = types.ModuleType("isaaclab"), types.ModuleType("isaaclab.actuators")
    mod.ImplicitActuatorCfg = lambda **kw: NS(**kw)
    pkg.actuators = mod
    monkeypatch.setitem(sys.modules, "isaaclab", pkg)
    monkeypatch.setitem(sys.modules, "isaaclab.actuators", mod)


@pytest.mark.parametrize("skin", [True, False])
def test_use_jointed_spawns_v2_with_one_actuator_per_group_and_turns_the_gait_on(fake_isaaclab, skin):
    ns = _use_jointed_dummy()
    cfg = NS(use_skin=skin,
             scene=NS(dummy=NS(spawn=NS(usd_path="v1.usd"), init_state=NS(joint_pos={}, joint_vel={}), actuators={})),
             commands=NS(base_velocity=NS(dummy_hold_clearance_m=0.005, dummy_hold_stand_height_m=0.9)),
             events=NS(aisle_guard=NS(params={"dummy_half_width": 0.0})))
    ns["_use_jointed_dummy"](cfg)
    spec = js.load()
    assert cfg.scene.dummy.spawn.usd_path == str(ns["V2_SKIN_USD"] if skin else ns["V2_USD"])
    assert Path(cfg.scene.dummy.spawn.usd_path).name == ("keti_dummy_ts15066_v2_jointed_skin.usd" if skin
                                                         else "keti_dummy_ts15066_v2_jointed.usd")
    assert cfg.scene.dummy.init_state.joint_pos == {".*": 0.0} and cfg.scene.dummy.init_state.joint_vel == {".*": 0.0}
    names, cols = js.dof_names(spec), js.columns(spec)
    groups = {}
    for j in spec.joints:
        groups.setdefault(j.group, []).extend(names[cols[j.name].start:cols[j.name].stop])
    acts = cfg.scene.dummy.actuators
    assert set(acts) == set(groups) == {"hip", "knee", "ankle", "shoulder", "elbow"}
    for g, act in acts.items():
        assert act.joint_names_expr == [f"^{n}$" for n in groups[g]]
        assert (act.stiffness, act.damping) == spec.drives["walking"][g]
    assert sorted(n for a in acts.values() for n in a.joint_names_expr) == sorted(f"^{n}$" for n in names)   # each once
    cmd = cfg.commands.base_velocity
    ramp = yaml.safe_load((REPO / "generation/dummy/gait/gait_spec.yaml").read_text(encoding="utf-8"))["ramp"]
    assert cmd.dummy_hold_clearance_m is None and cmd.dummy_gait is True
    # the partial weight support: a share of the weight toward the stand height with the feet
    assert cmd.dummy_hold_support_share == ns["HOLD_SUPPORT_SHARE"] == 0.7
    assert cmd.dummy_hold_plane_freq_hz == ns["HOLD_PLANE_FREQ_HZ"] == 1.0
    assert cmd.dummy_hold_stand_height_m == aisle_guard.standing_root_height(js.geometry_with_feet(spec)) == 1.43
    assert (cmd.dummy_gait_accel_time_s, cmd.dummy_gait_decel_time_s) == (ramp["accel_time_s"], ramp["decel_time_s"])
    assert cmd.dummy_hold_inertia_model == "upper_body_hips"
    assert cmd.dummy_upper_body_prims == tuple(js.upper_body_prims(spec))
    assert cmd.dummy_region_prims == tuple(js.region_prims(spec))
    assert cmd.dummy_hip_above_root_m == js.hip_above_root(spec)
    for mode, gains in (("walking", cmd.dummy_walk_gains), ("tone", cmd.dummy_tone_gains)):
        assert gains == {n: spec.drives[mode][g] for g, dofs in groups.items() for n in dofs}
    width = cfg.events.aisle_guard.params["dummy_half_width"]
    assert width == aisle_guard.dummy_horizontal_radius(js.geometry_with_feet(spec))
    assert width >= aisle_guard.dummy_horizontal_radius(spec.anthropometry)


def test_s1d_turns_the_jointed_dummy_on_only_when_asked_after_the_skin_and_before_the_cameras():
    """use_jointed is off by default, and its call comes after the skin's asset path (which it replaces) and before
    the cameras."""
    _, cls = _class(CFG, "S1dEnvCfg")
    fields = {n.target.id: n.value for n in cls.body if isinstance(n, ast.AnnAssign)}
    assert isinstance(fields["use_jointed"], ast.Constant) and fields["use_jointed"].value is False
    post = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
    tests = [ast.unparse(n.test) if isinstance(n, ast.If) else "" for n in post.body]
    i_skin, i_jointed, i_cams = tests.index("self.use_skin"), tests.index("self.use_jointed"), tests.index("self.use_cameras")
    assert i_skin < i_jointed < i_cams
    assert ast.unparse(post.body[i_jointed].body[0]) == "self._use_jointed_dummy()" and not post.body[i_jointed].orelse


def test_the_command_term_declares_the_gait_settings_with_the_rigid_dummys_defaults():
    _, cls = _class(COMMANDS, "S1aStraightWalkCommandCfg")
    fields = {n.target.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign) and n.value is not None
              and n.target.id.startswith("dummy_")}
    assert {k: fields.get(k, "missing") for k in (
        "dummy_gait", "dummy_gait_accel_time_s", "dummy_gait_decel_time_s", "dummy_hold_inertia_model",
        "dummy_upper_body_prims", "dummy_region_prims", "dummy_hip_above_root_m", "dummy_walk_gains",
        "dummy_tone_gains", "dummy_hold_support_share", "dummy_hold_plane_freq_hz")} == {
        "dummy_gait": False, "dummy_gait_accel_time_s": 1.0, "dummy_gait_decel_time_s": 1.0,
        "dummy_hold_inertia_model": "whole_body_feet", "dummy_upper_body_prims": (), "dummy_region_prims": (),
        "dummy_hip_above_root_m": None, "dummy_walk_gains": None, "dummy_tone_gains": None,
        "dummy_hold_support_share": None, "dummy_hold_plane_freq_hz": None}
