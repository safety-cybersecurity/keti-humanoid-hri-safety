"""generation/envs/s1a/mdp/commands.py with the jointed dummy's gait, run: the command term's gait methods taken out
of the source and called on a stand-in dummy that holds the articulation's contract -- joint indices as a list or an
int32 tensor (the PhysX articulation hands a tensor to its Warp kernels uncast, and Warp refuses int64), float32 data
of shape (len(env_ids), len(joint_ids)). The module imports Isaac, so the methods are compiled from its syntax tree.
Needs torch: runs under Isaac's python, skips where torch is missing."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from generation.dummy.gait.gait_table import GaitTable  # noqa: E402

SRC = Path(__file__).resolve().parents[2] / "generation/envs/s1a/mdp/commands.py"
HRI_SRC = Path(__file__).resolve().parents[2] / "generation/envs/hri/commands.py"
RECORDER_SRC = Path(__file__).resolve().parents[2] / "generation/envs/s1a/recorder.py"
METHODS = ("_ensure_gait", "_gait_drive", "_drive_gait", "_write_gains", "_update_hold_release", "_forget_gait_rate",
           "_gait_step_length", "_stop_distance", "_facing_turn_torque")


def _methods() -> dict:
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "S1aStraightWalkCommand")
    found = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    ns = {"torch": torch, "np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=found, type_ignores=[])), str(SRC), "exec"), ns)
    missing = [name for name in METHODS if name not in ns]
    assert not missing, f"commands.py has no {missing}"
    return {name: ns[name] for name in METHODS}


DOFS = ("hip_left:0", "knee_left", "hip_right:0", "knee_right")


def _table() -> GaitTable:
    targets = np.zeros((3, 4, len(DOFS)), np.float32)
    targets[1:] = np.linspace(0.1, 0.4, len(DOFS), dtype=np.float32)
    return GaitTable(DOFS, np.array([0.0, 0.5, 1.5]), np.array([0.0, 0.8, 1.6]), targets)


class Dummy:
    """The articulation's index contract, and a record of what was written."""

    def __init__(self, n_envs):
        # The articulation's own joint order, not the table's: the term must map by name.
        self.joint_names = ["knee_right", "hip_left:0", "knee_left", "hip_right:0"]
        self.cfg = NS(spawn=NS(usd_path="v2.usd"))
        self.data = NS(joint_pos=NS(torch=torch.arange(n_envs * 4, dtype=torch.float32).reshape(n_envs, 4) / 10))
        self.permanent_wrench_composer = NS(set_forces_and_torques_index=lambda **kw: None)
        self.writes = []

    def _check(self, kind, data, joint_ids, env_ids):
        if not isinstance(joint_ids, list) and not (isinstance(joint_ids, torch.Tensor) and joint_ids.dtype == torch.int32):
            raise TypeError(f"{kind}: joint indices {getattr(joint_ids, 'dtype', type(joint_ids))} are not int32")
        assert data.dtype == torch.float32 and tuple(data.shape) == (len(env_ids), len(joint_ids)), (kind, data.shape)
        self.writes.append((kind, [int(e) for e in env_ids], [int(j) for j in joint_ids], data.clone()))

    def set_joint_position_target_index(self, *, target, joint_ids=None, env_ids=None):
        self._check("target", target, joint_ids, env_ids)

    def set_joint_velocity_target_index(self, *, target, joint_ids=None, env_ids=None):
        self._check("velocity", target, joint_ids, env_ids)

    def write_joint_stiffness_to_sim_index(self, *, stiffness, joint_ids=None, env_ids=None):
        self._check("stiffness", stiffness, joint_ids, env_ids)

    def write_joint_damping_to_sim_index(self, *, damping, joint_ids=None, env_ids=None):
        self._check("damping", damping, joint_ids, env_ids)


def _moving_table() -> GaitTable:
    """A table whose targets move with the phase (0.1 rad per tenth of a cycle on every DOF at the walking speeds)."""
    targets = np.zeros((3, 10, len(DOFS)), np.float32)
    targets[1:] = (0.1 * np.arange(10, dtype=np.float32))[None, :, None]
    return GaitTable(DOFS, np.array([0.0, 0.5, 1.5]), np.array([0.0, 0.8, 1.6]), targets)


def _term(monkeypatch, n_envs=2, table=_table):
    from generation.dummy import jointed

    monkeypatch.setattr(jointed, "read_gait", lambda usd: table())
    m = _methods()
    dummy = Dummy(n_envs)
    tone = {n: (3.0, 0.3) for n in DOFS}
    t = NS(_gait=None, _gait_dof_ids=None, _gait_previous=None, gait_phase=torch.zeros(n_envs), gait_speed=torch.zeros(n_envs),
           _gait_slowing=torch.zeros(n_envs, dtype=torch.bool),
           hold_released=torch.zeros(n_envs, dtype=torch.bool), hold_release_step=torch.full((n_envs,), -1),
           num_envs=n_envs, device="cpu", _dummy_root_body_id=0, dummy_lin_force_applied=None,
           dummy_torque_saturated_per_axis=None, dummy_torque_applied=None,
           cfg=NS(dummy_gait=True, dummy_release_gap_m=0.1, dummy_tone_gains=tone, dummy_hold_support_share=None,
                  dummy_walk_gains={n: (30.0, 3.0) for n in DOFS}),
           _env=NS(scene={"dummy": dummy}, common_step_counter=7, step_dt=0.02))
    t._robot_contact_present = lambda: torch.zeros(n_envs, dtype=torch.bool)
    for name, fn in m.items():
        setattr(t, name, (lambda f: lambda *a, **k: f(t, *a, **k))(fn))
    return t, dummy


def test_the_gait_writes_its_targets_with_int32_joint_indices_mapped_by_name(monkeypatch):
    t, dummy = _term(monkeypatch)
    t._drive_gait(dummy, torch.zeros(2, 2))
    kind, envs, joints, target = _last(dummy, "target")
    assert envs == [0, 1]
    assert [dummy.joint_names[j] for j in joints] == list(DOFS)


def test_a_release_before_the_gait_was_first_read_still_freezes_the_pose_at_tone(monkeypatch):
    # The release gate can fire on the update before the gait's first read; the latch never retries.
    t, dummy = _term(monkeypatch)
    t._update_hold_release(torch.tensor([0.0, 10.0]))
    kinds = {k: (envs, joints, data) for k, envs, joints, data in dummy.writes}
    assert set(kinds) == {"target", "velocity", "stiffness", "damping"}
    envs, joints, target = kinds["target"]
    assert envs == [0]
    assert torch.equal(target, dummy.data.joint_pos.torch[[0]][:, joints])      # frozen where it was
    assert kinds["stiffness"][0] == [0] and torch.all(kinds["stiffness"][2] == 3.0)


# ---- the hold's release in the scenarios with a conditions table ----
# The base term's release with the HRI configuration (no distance gate, the box's touch threshold) and the HRI term's
# contact gate, both compiled from their source.

def _hri_method(name):
    tree = ast.parse(HRI_SRC.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HriTaskCommand")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(HRI_SRC), "exec"), ns)
    return ns[name]


def _contact_eps_n() -> float:
    import re

    found = re.findall(r"^_CONTACT_EPS_N = (\S+)$", RECORDER_SRC.read_text(encoding="utf-8"), re.M)
    assert len(found) == 1
    return float(found[0])


def _hri_term(monkeypatch):
    t, dummy = _term(monkeypatch, n_envs=1)
    t.cfg.dummy_release_gap_m = None
    t.cfg.release_threshold_n = _contact_eps_n()
    t.touch = None
    present = _hri_method("_robot_contact_present")
    t._robot_contact_present = lambda: present(t)
    return t, dummy


def _touch(*body_n):
    from generation.envs.hri import command_logic as logic

    return logic.Touch(body_n=np.asarray(body_n, float), source_n=np.zeros(2), samples=4)


def _release_step(t, step, distance, touch):
    t._env.common_step_counter = step
    t.touch = touch
    t._update_hold_release(torch.tensor([distance]))


def test_without_a_distance_gate_a_near_miss_keeps_the_hold_and_the_first_touch_lets_it_go(monkeypatch):
    t, dummy = _hri_term(monkeypatch)
    # a near miss: well inside S1d's 0.10 m gap, down to the surfaces meeting, and nothing touched
    for step, distance in enumerate([0.5, 0.2, 0.09, 0.03, 0.0]):
        _release_step(t, step, distance, _touch(0.0, 0.0, 0.0))
        assert not bool(t.hold_released[0]) and int(t.hold_release_step[0]) == -1
    assert dummy.writes == []
    # a light touch, below S1d's 10 N backstop, lets the hold go on its own step
    _release_step(t, 5, 0.0, _touch(0.0, 5.0, 0.0))
    assert bool(t.hold_released[0]) and int(t.hold_release_step[0]) == 5
    _release_step(t, 6, 0.0, _touch(0.0, 0.0, 0.0))
    assert int(t.hold_release_step[0]) == 5                                     # a latch


def test_a_pinch_lets_the_hold_go(monkeypatch):
    from generation.envs.hri import command_logic as logic

    t, _ = _hri_term(monkeypatch)
    pinch = np.array([[[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]])
    _release_step(t, 3, 0.0, logic.TouchWindow(lambda: pinch).drain())
    assert bool(t.hold_released[0]) and int(t.hold_release_step[0]) == 3


def test_the_hold_runs_into_at_most_one_touched_step(monkeypatch):
    # the recorder counts a step touched while the wrench in effect was the hold's (hold_contaminated_steps); the
    # wrench of a step is the one queued at the step before, so the hold is in effect up to its release's step
    t, _ = _hri_term(monkeypatch)
    eps = _contact_eps_n()
    touches = [0.0, 0.0, 3.0, 4.0, 2.0, 0.0, 6.0]                                  # light touches, under 10 N
    contaminated, held = 0, True
    for step, force in enumerate(touches):
        touch = _touch(force)
        contaminated += int(touch.over(eps) and held)
        _release_step(t, step, 0.0, touch)
        held = not bool(t.hold_released[0])
    assert contaminated == 1 and int(t.hold_release_step[0]) == 2


def test_s1ds_distance_gate_still_lets_the_hold_go_before_a_touch(monkeypatch):
    t, _ = _term(monkeypatch, n_envs=1)                                         # the base term's 0.1 m gap
    t._env.common_step_counter = 4
    t._update_hold_release(torch.tensor([0.11]))
    assert not bool(t.hold_released[0])
    t._update_hold_release(torch.tensor([0.09]))
    assert bool(t.hold_released[0]) and int(t.hold_release_step[0]) == 4


def test_a_later_release_freezes_only_the_env_let_go(monkeypatch):
    t, dummy = _term(monkeypatch)
    t._drive_gait(dummy, torch.zeros(2, 2))
    dummy.writes.clear()
    t._update_hold_release(torch.tensor([10.0, 0.0]))
    assert {(k, tuple(envs)) for k, envs, _, _ in dummy.writes} == {("target", (1,)), ("velocity", (1,)),
                                                                      ("stiffness", (1,)), ("damping", (1,))}
    t._drive_gait(dummy, torch.zeros(2, 2))
    assert _last(dummy, "target")[1] == [0] and _last(dummy, "velocity")[1] == [0]   # the held env walks on alone


# The joints follow the table through a PD whose damping pulls toward a target velocity: left at zero, a joint lags
# its moving target by damping / stiffness (measured at about 30 ms -- 12 to 17 degrees of knee at a walk), and the
# legs are not the table's lengths. The term hands the drives the rate the targets move at.


def _walk(t, dummy, steps, dxy=0.02):
    t.gait_speed = torch.ones(t.num_envs)
    for _ in range(steps):
        t._drive_gait(dummy, torch.tensor([[dxy, 0.0]] * t.num_envs))


def _last(dummy, kind):
    return [w for w in dummy.writes if w[0] == kind][-1]


def test_the_first_gait_update_after_a_reset_asks_no_joint_velocity(monkeypatch):
    t, dummy = _term(monkeypatch, table=_moving_table)
    _walk(t, dummy, 1)
    kind, envs, joints, velocity = _last(dummy, "velocity")
    assert envs == [0, 1] and [dummy.joint_names[j] for j in joints] == list(DOFS)
    assert torch.count_nonzero(velocity) == 0


def test_the_gait_asks_the_rate_its_targets_move_at(monkeypatch):
    t, dummy = _term(monkeypatch, table=_moving_table)
    _walk(t, dummy, 3)
    targets = [w[3] for w in dummy.writes if w[0] == "target"]
    velocity = _last(dummy, "velocity")[3]
    assert torch.count_nonzero(targets[-1] - targets[-2]) > 0
    assert torch.allclose(velocity, (targets[-1] - targets[-2]) / 0.02)


def test_a_release_stops_the_joint_velocity_it_had_asked(monkeypatch):
    t, dummy = _term(monkeypatch, table=_moving_table)
    _walk(t, dummy, 3)
    t._update_hold_release(torch.tensor([10.0, 0.0]))
    kind, envs, joints, velocity = _last(dummy, "velocity")
    assert envs == [1] and torch.count_nonzero(velocity) == 0


def test_a_reset_env_starts_its_rate_over(monkeypatch):
    t, dummy = _term(monkeypatch, table=_moving_table)
    _walk(t, dummy, 3)
    t._forget_gait_rate(torch.tensor([0]))
    _walk(t, dummy, 1)
    velocity = _last(dummy, "velocity")[3]
    assert torch.count_nonzero(velocity[0]) == 0 and torch.count_nonzero(velocity[1]) > 0


def test_the_episode_reset_forgets_the_rate():
    src = SRC.read_text(encoding="utf-8")
    block = src[src.index("    def _resample_command("):src.index("    def _update_command(")]
    assert "self._forget_gait_rate(env_ids)" in block


# Below the slowest row the gait shrinks by sqrt(v / v1) as a walk starts and by v / v1 as it slows to its stop
# (gait_table): the term says which, per env, from the scripted speed it just ramped.


def _ramp_term(monkeypatch, gait_speed, remaining_y):
    t, dummy = _term(monkeypatch)
    t.gait_speed = torch.tensor(gait_speed)
    t.cfg.dummy_gait_accel_time_s = t.cfg.dummy_gait_decel_time_s = 1.0
    entry = {"speed": torch.tensor([1.0, 1.0]), "target_xy": torch.tensor([[0.0, remaining_y[0]], [0.0, remaining_y[1]]]),
             "stop_y": torch.zeros(2), "angle": torch.full((2,), math.pi / 2)}
    return t, dummy, entry


def test_the_step_says_which_envs_are_slowing_to_their_stop(monkeypatch):
    # env 0 is 5 cm from its stop line at 0.5 m/s (the ramp brings it down), env 1 is 5 m away (the ramp speeds it up)
    t, _, entry = _ramp_term(monkeypatch, [0.5, 0.5], [0.05, 5.0])
    t._gait_step_length(entry, torch.tensor([True, True]))
    assert t.gait_speed[0] < 0.5 < t.gait_speed[1]
    assert t._gait_slowing.tolist() == [True, False]


def test_a_walk_at_its_speed_is_not_slowing(monkeypatch):
    t, _, entry = _ramp_term(monkeypatch, [1.0, 1.0], [5.0, 5.0])
    t._gait_step_length(entry, torch.tensor([True, True]))
    assert t._gait_slowing.tolist() == [False, False]


def test_the_gait_shrinks_each_env_by_its_own_slowing(monkeypatch):
    # at a quarter of the slowest row's speed: the slowest row's own targets slowing, shrunk by sqrt(0.25) starting
    t, dummy = _term(monkeypatch)
    t.gait_speed = torch.full((2,), 0.125)
    t._gait_slowing = torch.tensor([True, False])
    t._drive_gait(dummy, torch.zeros(2, 2))
    target = _last(dummy, "target")[3]
    torch.testing.assert_close(target[:, 0], torch.tensor([0.1, 0.05]))       # row 1 targets 0.1 on the first DOF


def test_a_reset_is_not_slowing():
    src = SRC.read_text(encoding="utf-8")
    block = src[src.index("    def _resample_command("):src.index("    def _update_command(")]
    assert "self._gait_slowing[env_ids] = False" in block


# ---- the person's turn on the spot (S1f) ----
# Between the trigger and the walk the HRI term turns the facing the hold servo aims at, step by step from the plan's
# facing to its walk, while the legs mark time through the base term's gait drive (_gait_drive) and the scripted
# target stays put. Its script methods are compiled into a subclass of the base term's gait methods -- super() needs
# a class -- and run on the stand-ins above, one update as the base term's _advance_dummy makes it with the gait on.

HRI_TURN_METHODS = ("_episode_step", "_trigger_latched", "_gait_drive", "_facing_turn_torque")
TURN_DT = 0.02
TURN_WALK = (math.cos(math.radians(-30.0)), math.sin(math.radians(-30.0)))


def _class_def(name, bases, body):
    extra = {"type_params": []} if "type_params" in ast.ClassDef._fields else {}
    return ast.ClassDef(name=name, bases=[ast.Name(id=b, ctx=ast.Load()) for b in bases], keywords=[], body=body,
                        decorator_list=[], **extra)


def _turn_class():
    from generation import human_script
    from generation.envs.hri import command_logic as logic
    from generation.envs.s1a.mdp.entry_geometry import facing_quat_xyzw

    base_cls = next(n for n in ast.parse(SRC.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.ClassDef) and n.name == "S1aStraightWalkCommand")
    base = [n for n in base_cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    hri_tree = ast.parse(HRI_SRC.read_text(encoding="utf-8"))
    helpers = [n for n in hri_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_t"]
    hri_cls = next(n for n in hri_tree.body if isinstance(n, ast.ClassDef) and n.name == "HriTaskCommand")
    hri = [n for n in hri_cls.body if isinstance(n, ast.FunctionDef) and n.name in HRI_TURN_METHODS]
    assert sorted(n.name for n in hri) == sorted(HRI_TURN_METHODS) and len(helpers) == 1
    module = ast.Module(body=helpers + [_class_def("Base", [], base), _class_def("Hri", ["Base"], hri)], type_ignores=[])
    ns = {"torch": torch, "np": np, "math": math, "logic": logic, "human_script": human_script,
          "facing_quat_xyzw": facing_quat_xyzw}
    exec(compile(ast.fix_missing_locations(module), str(HRI_SRC), "exec"), ns)
    return ns["Hri"]


def _turn_term(monkeypatch, turn_time_s=1.0, trigger_s=0.1):
    """The HRI term with one env holding a person who faces the rack (+90 deg) and walks off at -30 deg after a time
    trigger and a turn of turn_time_s, as each episode starts it (_new_episode, events.reset_dummy_scripted)."""
    from generation import human_script
    from generation.dummy import jointed
    from generation.envs.s1a.mdp.entry_geometry import facing_quat_xyzw

    monkeypatch.setattr(jointed, "read_gait", lambda usd: _moving_table())
    accel, decel = human_script._ramp_times()
    t = _turn_class()()
    t.plan = human_script.HumanPlan((0.0, 0.0), 90.0, TURN_WALK, 1.2, human_script.Trigger("time", trigger_s), 5.0,
                                    turn_time_s)
    entry = {"trigger_kind": "time", "trigger_distance": torch.tensor([trigger_s]), "x_cross": torch.zeros(1),
             "triggered": torch.zeros(1, dtype=torch.bool), "facing_quat": torch.tensor([facing_quat_xyzw(90.0)]),
             "speed": torch.tensor([1.2]), "walk_ended": torch.zeros(1, dtype=torch.bool)}
    t.__dict__.update(
        _gait=None, _gait_dof_ids=None, _gait_previous=None, gait_phase=torch.zeros(1), gait_speed=torch.zeros(1),
        _gait_slowing=torch.zeros(1, dtype=torch.bool), hold_released=torch.zeros(1, dtype=torch.bool), num_envs=1,
        device="cpu", _turn_from_step=-1, _turn_k=-1,
        cfg=NS(dummy_gait=True, dummy_hold_support_share=None, dummy_gait_accel_time_s=accel,
               dummy_gait_decel_time_s=decel),
        _env=NS(step_dt=TURN_DT, episode_length_buf=torch.zeros(1, dtype=torch.long), s1a_entry=entry))
    return t, entry, NS(data=NS(root_pos_w=torch.zeros(1, 3))), Dummy(1)


def _yaw(entry) -> float:
    _, _, z, w = (float(c) for c in entry["facing_quat"][0])
    return math.degrees(2.0 * math.atan2(z, w))


def _base_drive(t, target_dxy):
    """The three values the base term's gait is driven with, as _drive_gait has always read them."""
    return (t.gait_speed.cpu().numpy(), t._gait_slowing.cpu().numpy(),
            torch.linalg.norm(target_dxy, dim=-1).cpu().numpy())


def _turn_update(t, entry, robot, dummy, step) -> dict:
    """One update of episode step ``step``: the script's trigger, the ramp's step, the target's move along the walk and
    the gait drive, in _advance_dummy's order."""
    t._env.episode_length_buf = torch.tensor([step + 1])
    moving = t._trigger_latched(entry, robot).clone()
    latched_speed = float(t.gait_speed[0])
    length = t._gait_step_length(entry, moving)
    target_dxy = torch.where(moving.unsqueeze(-1), length.unsqueeze(-1) * torch.tensor([TURN_WALK]), torch.zeros(1, 2))
    v, slowing, distance = t._gait_drive(target_dxy)
    t._drive_gait(dummy, target_dxy)
    return {"moving": bool(moving[0]), "triggered": bool(entry["triggered"][0]), "yaw": _yaw(entry),
            "turn_k": t._turn_k, "latched_speed": latched_speed, "travel": float(torch.linalg.norm(target_dxy)),
            "v": float(v[0]), "slowing": bool(slowing[0]), "distance": float(distance[0]),
            "phase": float(t.gait_phase[0])}


def test_a_plan_without_a_turn_latches_and_walks_as_before(monkeypatch):
    from generation import human_script

    t, entry, robot, dummy = _turn_term(monkeypatch, turn_time_s=0.0)
    facing = entry["facing_quat"].clone()
    travel, trigger_step = [], None
    for step in range(60):
        t._env.episode_length_buf = torch.tensor([step + 1])
        speed = t.gait_speed.clone()
        out = t._trigger_latched(entry, robot)
        assert out is entry["triggered"]                                 # the latch itself, as the base term reads it
        assert torch.equal(entry["facing_quat"], facing) and torch.equal(t.gait_speed, speed) and t._turn_k == -1
        if trigger_step is None and bool(out[0]):
            trigger_step = step
        length = t._gait_step_length(entry, out)
        target_dxy = torch.where(out.unsqueeze(-1), length.unsqueeze(-1) * torch.tensor([TURN_WALK]), torch.zeros(1, 2))
        for got, want in zip(t._gait_drive(target_dxy), _base_drive(t, target_dxy)):
            assert got.dtype == want.dtype
            np.testing.assert_array_equal(got, want)
        t._drive_gait(dummy, target_dxy)
        travel.append(float(torch.linalg.norm(target_dxy)))
    assert trigger_step is not None and trigger_step > 0
    # walks from rest on the trigger's own step, as the plan does
    walked = np.cumsum(travel[trigger_step:])
    for k, got in enumerate(walked, start=1):
        assert got == pytest.approx(human_script.walked(t.plan, k * TURN_DT), rel=1e-5)
    assert walked[0] > 0.0


def test_the_person_turns_before_it_walks(monkeypatch):
    from generation import human_script
    from generation.envs.hri import command_logic as logic

    t, entry, robot, dummy = _turn_term(monkeypatch, turn_time_s=1.0)
    log = [_turn_update(t, entry, robot, dummy, step) for step in range(110)]
    s0 = next(i for i, r in enumerate(log) if r["triggered"])
    assert s0 > 0 and not any(r["triggered"] for r in log[:s0])           # the person stood first
    turn, walk = log[s0:s0 + 50], log[s0 + 50:]
    # fifty steps from the trigger's own: triggered (the standing ends), the target held, the legs marking time
    assert all(r["triggered"] and not r["moving"] and r["travel"] == 0.0 for r in turn)
    assert [r["turn_k"] for r in turn] == list(range(50))
    slow = human_script.turn_gait_speed_mps()
    for k, r in enumerate(turn):
        assert r["v"] == pytest.approx(logic.turn_gait_speed(k, slow, TURN_DT))
        assert r["distance"] == pytest.approx(r["v"] * TURN_DT) and r["distance"] > 0.0 and not r["slowing"]
    phases = [log[s0 - 1]["phase"]] + [r["phase"] for r in turn]
    assert all(b > a for a, b in zip(phases, phases[1:]))                  # the phase goes on without the target
    # the facing turns from +90 to -30 the short way, through the robot's heading (+30 half way)
    yaws = [r["yaw"] for r in turn]
    assert yaws[0] == pytest.approx(90.0, abs=1.0) and yaws[0] < 90.0
    assert yaws[24] == pytest.approx(30.0, abs=1e-3) and yaws[-1] == pytest.approx(-30.0, abs=1e-3)
    assert all(b < a for a, b in zip(yaws, yaws[1:]))
    # the 51st: the walk sets off at the slowest gait, once, and the facing stays the walk's
    assert walk[0]["moving"] and walk[0]["turn_k"] == -1 and walk[0]["latched_speed"] == pytest.approx(slow)
    assert walk[1]["latched_speed"] > walk[0]["latched_speed"]
    assert all(r["moving"] and r["yaw"] == pytest.approx(-30.0, abs=1e-3) for r in walk)
    # the target travels as the plan says, the turn and the walk counted with the same integer
    walked = np.cumsum([r["travel"] for r in log[s0:]])
    for k, got in enumerate(walked, start=1):
        assert got == pytest.approx(human_script.walked(t.plan, k * TURN_DT), rel=1e-5, abs=1e-12)


def test_a_release_during_the_turn_ends_it(monkeypatch):
    t, entry, robot, dummy = _turn_term(monkeypatch, turn_time_s=1.0)
    step = 0
    while not _turn_update(t, entry, robot, dummy, step)["triggered"]:
        step += 1
    for step in range(step + 1, step + 19):                                 # nineteen turning steps in all
        _turn_update(t, entry, robot, dummy, step)
    assert t._turn_k == 18
    facing = entry["facing_quat"].clone()
    t.hold_released[:] = True                                               # let go on the twentieth
    for step in range(step + 1, step + 60):                                 # past where the turn would have ended
        t._env.episode_length_buf = torch.tensor([step + 1])
        out = t._trigger_latched(entry, robot)
        assert out is entry["triggered"] and bool(out[0])
        assert torch.equal(entry["facing_quat"], facing) and t._turn_k == -1
        assert float(t.gait_speed[0]) == 0.0                                # no walk is set off at the turn's end
        target_dxy = torch.zeros(1, 2)
        for got, want in zip(t._gait_drive(target_dxy), _base_drive(t, target_dxy)):
            np.testing.assert_array_equal(got, want)


# ---- the hold's torque while the facing turns ----
# The hold's torque lines of _advance_dummy -- from its spring-and-damper line to the turning facing's addition -- taken
# out of the source and run on the HRI term above, step by step with the script, on gains and errors of one env.

def _servo_lines():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "S1aStraightWalkCommand")
    body = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_advance_dummy").body
    lines = [ast.unparse(n) for n in body]
    i = lines.index("ang_torque = ang_stiffness * ang_error - ang_damping * ang_vel")
    j = next(k for k in range(i, len(body)) if isinstance(body[k], ast.If)
             and ast.unparse(body[k].test) == "turning is not None")
    return compile(ast.fix_missing_locations(ast.Module(body=body[i:j + 1], type_ignores=[])), str(SRC), "exec")


# Per axis (roll, pitch, yaw), float32 as the env's: a tipping-axis gain far above yaw's, a turn's yaw error and rate.
SERVO = {"ang_stiffness": torch.tensor([[17965.0, 17965.0, 118.0]]), "ang_error": torch.tensor([[0.01, -0.02, 0.3]]),
         "ang_damping": torch.tensor([[2860.0, 2860.0, 18.8]]), "ang_vel": torch.tensor([[0.1, -0.05, -2.0]])}


def _hold_torque(t, code):
    ns = {"self": t, "torch": torch, **SERVO}
    exec(code, ns)
    return ns["ang_torque"]


def _spring_and_damper():
    return SERVO["ang_stiffness"] * SERVO["ang_error"] - SERVO["ang_damping"] * SERVO["ang_vel"]


def test_a_plan_without_a_turn_leaves_the_holds_torque_as_it_was(monkeypatch):
    code = _servo_lines()
    t, entry, robot, dummy = _turn_term(monkeypatch, turn_time_s=0.0)
    triggered = False
    for step in range(60):
        triggered |= _turn_update(t, entry, robot, dummy, step)["triggered"]
        assert t._facing_turn_torque(SERVO["ang_damping"]) is None
        assert torch.equal(_hold_torque(t, code), _spring_and_damper())         # the very bits
    assert triggered


def test_a_turning_facing_adds_the_dampers_pull_toward_its_rate_on_z_alone(monkeypatch):
    from generation import human_script
    from generation.envs.hri import command_logic as logic

    code = _servo_lines()
    t, entry, robot, dummy = _turn_term(monkeypatch, turn_time_s=1.0)
    n, walk = logic.turn_steps(1.0, TURN_DT), human_script.walk_yaw_deg(t.plan)
    log = []
    for step in range(110):
        r = _turn_update(t, entry, robot, dummy, step)
        log.append((r, t._facing_turn_torque(SERVO["ang_damping"]), _hold_torque(t, code)))
    s0 = next(i for i, (r, _, _) in enumerate(log) if r["triggered"])
    old = _spring_and_damper()
    # before the trigger and after the turn: no turning torque, the spring and damper as they were
    for r, turning, torque in log[:s0] + log[s0 + n:]:
        assert turning is None and torch.equal(torque, old)
    rates = []
    for k, (r, turning, torque) in enumerate(log[s0:s0 + n]):
        assert r["turn_k"] == k
        # the rate of the facing written this step -- the turn's k + 1 -- times the yaw damper, about z alone
        rate = math.radians(logic.turn_yaw_rate_deg_s(90.0, walk, k + 1, n, TURN_DT))
        torch.testing.assert_close(turning, torch.tensor([[0.0, 0.0, 18.8 * rate]]), rtol=1e-6, atol=0.0)
        assert torch.equal(torque[:, :2], old[:, :2])
        torch.testing.assert_close(torque[:, 2] - old[:, 2], turning[:, 2], rtol=1e-5, atol=1e-5)
        rates.append(rate)
    # clockwise, from +90 to -30: the pull is toward negative yaw rates, largest half way, nothing at the turn's end
    assert all(rate < 0.0 for rate in rates[:-1]) and rates[-1] == 0.0
    assert min(rates) == rates[n // 2 - 1] or min(rates) == rates[n // 2]

