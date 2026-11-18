"""The environment of the HRI scenarios (generation/envs/hri/): the canon's vested dummy, the gym ids, the box the
robot holds and the arm hold. The environment modules import Isaac, so their constants and classes are read from the
source and the registration is run against stand-ins; carry_rig's arithmetic is Isaac-free and called directly."""
import ast
import json
import math
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from generation import carry as carry_canon
from generation import conditions
from generation.envs.hri import carry_rig
from generation.scene.carried_box import NOMINAL_SIZE_M

REPO = Path(__file__).resolve().parents[2]
S1D_CFG = REPO / "generation/envs/s1d/s1d_env_cfg.py"
HRI_CFG = REPO / "generation/envs/hri/hri_env_cfg.py"
REGISTER = REPO / "generation/envs/register.py"


def _canon():
    return json.loads((REPO / "meta/scenarios.json").read_text(encoding="utf-8"))


def _assets():
    return json.loads((REPO / "meta/assets.json").read_text(encoding="utf-8"))


def _module_constants(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in names]
    ns = {"__file__": str(path), "Path": Path}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns


def test_the_vest_constant_is_the_canon_vest():
    ns = _module_constants(S1D_CFG, ("REPO", "V2_VEST_USD"))
    vest = _assets()["scene"]["keti_dummy_ts15066_v2_vest"]["usd_path"]
    assert ns["V2_VEST_USD"] == ns["REPO"] / vest
    assert ns["REPO"] == REPO


def test_every_new_scenario_wears_the_vest():
    # The canon's human model of each scenario with a conditions table is the vested dummy the constant names.
    for entry in _canon()["scenarios"]:
        if "conditions" in entry:
            assert entry["scene"]["human_model"] == "keti_dummy_ts15066_v2_vest", entry["scenario_id"]


def test_the_table_scenarios_are_the_ones_with_a_conditions_table():
    expected = tuple(e["scenario_id"] for e in _canon()["scenarios"] if "conditions" in e)
    assert conditions.table_scenarios() == expected
    assert set(expected) == {"S1e", "S1f", "S1g"}
    assert conditions.scenario_entry("S1f")["scenario_id"] == "S1f"
    with pytest.raises(ValueError, match="S1d"):
        conditions.scenario_entry("S1d")
    with pytest.raises(ValueError, match="S9z"):
        conditions.scenario_entry("S9z")


def test_the_box_size_of_a_row():
    assert conditions.box_size({"box_depth_m": 0.27, "box_height_m": 0.22}) == (NOMINAL_SIZE_M[0], 0.27, 0.22)


def _stub_registration(monkeypatch):
    calls = []
    gym = types.ModuleType("gymnasium")
    gym.register = lambda **kw: calls.append(kw)
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    chain = "isaaclab_tasks.manager_based.locomotion.velocity.config.h1"
    parts = chain.split(".")
    for k in range(1, len(parts) + 1):
        monkeypatch.setitem(sys.modules, ".".join(parts[:k]), types.ModuleType(".".join(parts[:k])))
    agents = types.ModuleType(chain + ".agents")
    monkeypatch.setitem(sys.modules, chain + ".agents", agents)
    sys.modules[chain].agents = agents
    runpy.run_path(str(REGISTER))
    return calls


def _hri_classes():
    tree = ast.parse(HRI_CFG.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            default = next((s.value.value for s in node.body if isinstance(s, ast.AnnAssign)
                            and isinstance(s.target, ast.Name) and s.target.id == "scenario_id"
                            and isinstance(s.value, ast.Constant)), None)
            out[node.name] = ([b.id for b in node.bases if isinstance(b, ast.Name)], default)
    return out


def test_new_scenarios_register(monkeypatch):
    calls = {c["id"]: c for c in _stub_registration(monkeypatch)}
    assert {"Keti-S1e-v0", "Keti-S1f-v0", "Keti-S1g-v0"} <= set(calls)
    assert "Keti-S1a-v0" in calls
    classes = _hri_classes()
    for sid in conditions.table_scenarios():
        call = calls[f"Keti-{sid}-v0"]
        module, name = call["kwargs"]["env_cfg_entry_point"].split(":")
        assert module == "generation.envs.hri.hri_env_cfg"
        bases, default = classes[name]
        assert bases == ["HriEnvCfg"] and default == sid, (name, bases, default)
        # the same agent the published checkpoint was trained with as the S1a id
        assert call["kwargs"]["rsl_rl_cfg_entry_point"] == calls["Keti-S1a-v0"]["kwargs"]["rsl_rl_cfg_entry_point"]
        assert call["entry_point"] == "isaaclab.envs:ManagerBasedRLEnv"


def test_the_carried_box_is_the_canon_asset_the_carry_names():
    entry = conditions.scenario_entry("S1e")
    held = carry_canon.load_carry(entry)
    path = carry_rig.box_usd_path(held)
    assert path == REPO / _assets()["scene"][held.asset]["usd_path"]
    assert _assets()["scene"][held.asset]["usd_path"] in entry["scene"]["assets"]


NAMES = ["left_hip_yaw", "right_elbow", "left_shoulder_pitch", "right_shoulder_pitch", "left_elbow"]


def test_the_arm_hold_finds_its_columns_by_name():
    hold = {"left_shoulder_pitch": 0.28, "right_shoulder_pitch": 0.28, "left_elbow": 0.52, "right_elbow": 0.52}
    default = {"left_hip_yaw": 0.0, "right_elbow": 0.52, "left_shoulder_pitch": 0.28, "right_shoulder_pitch": 0.28,
               "left_elbow": 0.52, "x": 0.0}
    # two action terms: the first moves one joint, the second the others in an order that is not the hold's
    terms = [(["x"], [0.0], [1.0]),
             (NAMES, [default[n] for n in NAMES], [0.5] * len(NAMES))]
    cols, values = carry_rig.arm_hold_columns(terms, hold, default)
    assert cols == [2, 3, 4, 5]                              # right_elbow, left_sh_pitch, right_sh_pitch, left_elbow
    assert values == pytest.approx([0.0, 0.0, 0.0, 0.0])
    # an offset that is not the default: the column holds (hold - offset) / scale
    shifted = [(NAMES, [0.0, 0.42, 0.28, 0.28, 0.52], [0.5, 0.25, 0.5, 0.5, 0.5])]
    cols, values = carry_rig.arm_hold_columns(shifted, hold, default)
    assert cols == [1, 2, 3, 4]
    assert values == pytest.approx([(0.52 - 0.42) / 0.25, 0.0, 0.0, 0.0])


def test_the_arm_hold_refuses_a_hold_that_is_not_the_robots_default():
    hold = {"left_elbow": 0.52, "right_elbow": 0.52}
    default = {"left_elbow": 0.52, "right_elbow": 0.30}
    with pytest.raises(ValueError, match="right_elbow"):
        carry_rig.arm_hold_columns([(["left_elbow", "right_elbow"], [0.52, 0.30], [0.5, 0.5])], hold, default)


def test_the_arm_hold_refuses_a_joint_no_action_term_moves():
    hold = {"left_elbow": 0.52, "right_elbow": 0.52}
    default = {"left_elbow": 0.52, "right_elbow": 0.52}
    with pytest.raises(ValueError, match="right_elbow"):
        carry_rig.arm_hold_columns([(["left_elbow"], [0.52], [0.5])], hold, default)


def test_the_carry_holds_both_arms_by_the_joint_maps_names():
    import yaml

    joints = yaml.safe_load((REPO / "meta/joint_map/unitree_h1.yaml").read_text(encoding="utf-8"))["joint_names"]
    for sid in ("S1e", "S1f", "S1g"):
        hold = carry_canon.arm_hold_joints(carry_canon.load_carry(conditions.scenario_entry(sid)), joints)
        assert len(hold) == 8 and set(hold) <= set(joints)
        assert {n.split("_", 1)[0] for n in hold} == {"left", "right"}


def test_pose_error():
    q = (0.0, 0.0, math.sin(0.3), math.cos(0.3))
    assert carry_rig.pose_error((1, 2, 3), q, (1, 2, 3), q) == (0.0, 0.0)
    m, deg = carry_rig.pose_error((1, 2, 3), q, (1, 2, 3.001), tuple(-v for v in q))   # q and -q: one turn
    assert m == pytest.approx(0.001) and deg == pytest.approx(0.0, abs=1e-6)
    half = math.radians(0.01) / 2.0
    turned = (0.0, 0.0, math.sin(0.3 + half), math.cos(0.3 + half))
    assert carry_rig.pose_error((0, 0, 0), q, (0, 0, 0), turned)[1] == pytest.approx(0.01, rel=1e-6)


def test_an_entity_prim_is_put_on_the_env():
    # By the time the authoring term runs, the scene has turned the cfg's {ENV_REGEX_NS} into the regex of every env.
    from types import SimpleNamespace as NS

    def env(path):
        return NS(scene=NS(env_regex_ns="/World/envs/env_.*", cfg=NS(box=NS(prim_path=path))))

    for path in ("/World/envs/env_.*/Box", "{ENV_REGEX_NS}/Box"):
        assert carry_rig._env_prim(env(path), "box", "/World/envs/env_0") == "/World/envs/env_0/Box"
    with pytest.raises(ValueError, match="/World/Box"):
        carry_rig._env_prim(env("/World/Box"), "box", "/World/envs/env_0")


# ---- the scene cameras by the side the person comes from (meta/cameras.json by_approach_side) -----------------

class _FakeCameraCfg:
    """isaaclab.sensors.CameraCfg as camera_cfg uses it: its keywords kept as attributes, OffsetCfg alike."""

    class OffsetCfg:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _camera_cfg():
    """s1d_env_cfg.camera_cfg, compiled from its source with stand-ins for the Isaac Lab classes it builds."""
    from generation import cameras

    tree = ast.parse(S1D_CFG.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "camera_cfg")
    ns = {"cam_canon": cameras, "CameraCfg": _FakeCameraCfg,
          "sim_utils": types.SimpleNamespace(PinholeCameraCfg=lambda **kw: types.SimpleNamespace(**kw))}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(S1D_CFG), "exec"), ns)
    return ns["camera_cfg"]


def _cameras_canon():
    from generation import cameras

    return cameras.load(REPO / "meta/cameras.json")


@pytest.mark.parametrize("side", ["west", "east"])
def test_the_camera_builder_puts_the_cctv_at_the_side_it_is_handed(side):
    from generation import cameras
    from generation.scene import layout

    camera_cfg, canon = _camera_cfg(), _cameras_canon()
    spec = layout.load_spec(layout.SCENE_SPEC)
    cfg = camera_cfg("cctv_rgb", canon["cameras"]["cctv_rgb"], spec, "S1g", approach_side=side)
    pos, rot = cameras.fixed_pose(canon["cameras"]["cctv_rgb"], spec, "S1g", approach_side=side)
    assert cfg.offset.pos == tuple(float(v) for v in pos) and cfg.offset.rot == tuple(float(v) for v in rot)
    follow = camera_cfg("third_person_rgb", canon["cameras"]["third_person_rgb"], spec, "S1g", approach_side=side)
    assert follow.prim_path == "{ENV_REGEX_NS}/third_person_rgb"


@pytest.mark.parametrize("name", ["cctv_rgb", "third_person_rgb"])
def test_the_camera_builder_refuses_s1g_without_a_side_naming_the_camera(name):
    from generation.scene import layout

    camera_cfg, canon = _camera_cfg(), _cameras_canon()
    with pytest.raises(ValueError, match=rf"^{name} 의 S1g 자리는 사람이 오는 쪽마다 하나입니다"):
        camera_cfg(name, canon["cameras"][name], layout.load_spec(layout.SCENE_SPEC), "S1g")


def test_the_camera_builder_places_s1d_s_cameras_whatever_side_it_is_handed():
    # S1d's sites hold one placement each: a side handed in does not move them (S1d's cfg leaves it None).
    from generation.scene import layout

    camera_cfg, canon = _camera_cfg(), _cameras_canon()
    spec = layout.load_spec(layout.SCENE_SPEC)
    base = camera_cfg("cctv_rgb", canon["cameras"]["cctv_rgb"], spec, "S1d")
    for side in ("west", "east"):
        other = camera_cfg("cctv_rgb", canon["cameras"]["cctv_rgb"], spec, "S1d", approach_side=side)
        assert (other.offset.pos, other.offset.rot) == (base.offset.pos, base.offset.rot)


def _post_init(path, cls):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return tree, node, next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")


def test_s1d_hands_its_cfg_s_side_to_the_camera_builder():
    _, cls, post = _post_init(S1D_CFG, "S1dEnvCfg")
    fields = {n.target.id: n.value for n in cls.body if isinstance(n, ast.AnnAssign)}
    assert isinstance(fields["camera_approach_side"], ast.Constant) and fields["camera_approach_side"].value is None
    calls = [ast.unparse(n) for n in ast.walk(post) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "camera_cfg"]
    assert calls == ["camera_cfg(name, cams['cameras'][name], spec, self.scenario_id, "
                     "approach_side=self.camera_approach_side)"]


def test_the_hri_cfg_sets_its_row_s_side_before_the_inherited_cameras_are_built():
    # The inherited __post_init__ builds the cameras, so the side has to be on the cfg before it runs: the row is
    # drawn (or taken) first, its approach_side set, then the inherited part.
    tree, _, post = _post_init(HRI_CFG, "HriEnvCfg")
    lines = [ast.unparse(n) for n in post.body]
    i_row = next(i for i, s in enumerate(lines) if s.startswith("if self.row is None:"))
    i_side = lines.index("self.camera_approach_side = self.row.get('approach_side')")
    i_super = lines.index("super().__post_init__()")
    assert i_row < i_side < i_super
    assert sum("camera_approach_side" in s for s in lines) == 1
    # Run on a stand-in, the part before the inherited one gives the cfg its row's side: row 0's when no row is
    # handed in, the handed row's otherwise.
    head = ast.FunctionDef(name="head", args=post.args, body=post.body[:i_super], decorator_list=[], returns=None,
                           type_params=[])
    boxes_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "layout_boxes")
    from generation.scene import layout

    ns = {"conditions": conditions, "layout": layout}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[boxes_fn, head], type_ignores=[])), str(HRI_CFG),
                 "exec"), ns)
    entry = conditions.scenario_entry("S1g")
    boxes = ns["layout_boxes"]()
    for row in (None, *(conditions.draw_row(entry, i, boxes) for i in range(4))):
        cfg = types.SimpleNamespace(scenario_id="S1g", use_jointed=True, use_skin=True, row=row,
                                    camera_approach_side=None)
        ns["head"](cfg)
        want = (row or conditions.draw_row(entry, 0, boxes))["approach_side"]
        assert cfg.camera_approach_side == want
    sides = {conditions.draw_row(entry, i, boxes)["approach_side"] for i in range(4)}
    assert sides == {"west", "east"}, sides
    cfg = types.SimpleNamespace(scenario_id="S1e", use_jointed=True, use_skin=True, row=None,
                                camera_approach_side=None)
    ns["head"](cfg)
    assert cfg.camera_approach_side is None                     # a table without the column: no side
