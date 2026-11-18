"""generation/envs/s1a/mdp/commands.py's partial weight support for the jointed dummy: while the hold is
on, the servo pushes up with a share of the dummy's weight and pulls its root toward the stand height plus the pelvis
height the gait's joint targets imply, with a spring and a damper -- the damper acting on the speed relative to that
moving target. The module imports Isaac, so the pieces are compiled from its syntax tree and run on stand-ins.
Needs torch: runs under Isaac's python, skips where torch is missing."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from generation.dummy import joints_spec as js  # noqa: E402
from generation.dummy.gait import support  # noqa: E402
from generation.dummy.gait.gait_table import GaitTable  # noqa: E402

SRC = Path(__file__).resolve().parents[2] / "generation/envs/s1a/mdp/commands.py"
TREE = ast.parse(SRC.read_text(encoding="utf-8"))
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "S1aStraightWalkCommand")
SPEC = js.load()
NAMES = js.dof_names(SPEC)
W, M = 75.0 * 9.81, 75.0


def _compiled(*names) -> dict:
    found = [n for n in TREE.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "_GAIT_REST_BLEND_STEPS"]
    found += [n for n in TREE.body + CLASS.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"torch": torch, "math": math, "np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=found, type_ignores=[])), str(SRC), "exec"), ns)
    missing = [n for n in names if n not in ns]
    assert not missing, f"commands.py has no {missing}"
    return ns


def _method_source(name: str) -> str:
    return ast.get_source_segment(SRC.read_text(encoding="utf-8"),
                                  next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == name))


# --- the setting ---------------------------------------------------------------------------------------------------


def _cfg(**kw):
    base = {"dummy_hold_support_share": 0.5, "dummy_hold_clearance_m": None, "dummy_hold_stand_height_m": 1.43}
    return NS(**(base | kw))


def test_a_share_with_a_stand_height_is_accepted_and_no_share_needs_nothing():
    check = _compiled("check_hold_support")["check_hold_support"]
    check(_cfg())
    check(_cfg(dummy_hold_support_share=0.0))
    check(_cfg(dummy_hold_support_share=None, dummy_hold_stand_height_m=None))


# False is 0 and True is 1 to arithmetic: a flag in the share's place is refused for its type, not its value
@pytest.mark.parametrize("share", [1.0, -0.1, float("nan"), float("inf"), True, False, "0.5"])
def test_a_share_that_is_not_a_fraction_below_one_is_refused(share):
    with pytest.raises(ValueError):
        _compiled("check_hold_support")["check_hold_support"](_cfg(dummy_hold_support_share=share))


@pytest.mark.parametrize("stand", [None, 0.0, float("nan")])
def test_a_share_without_a_stand_height_is_refused(stand):
    with pytest.raises(ValueError):
        _compiled("check_hold_support")["check_hold_support"](_cfg(dummy_hold_stand_height_m=stand))


def test_a_share_and_the_carried_hold_together_are_refused():
    with pytest.raises(ValueError):
        _compiled("check_hold_support")["check_hold_support"](_cfg(dummy_hold_clearance_m=0.005))


def test_the_term_checks_the_setting_when_it_is_built():
    assert "check_hold_support(cfg)" in _method_source("__init__")


# --- the upward force ----------------------------------------------------------------------------------------------

K, C = 1000.0, 100.0


def _term(share=0.5, origins=((0.0, 0.0, 0.0), (5.0, 0.0, 0.1))):
    fn = _compiled("_support_force_z")["_support_force_z"]
    n = len(origins)
    t = NS(cfg=NS(dummy_hold_support_share=share, dummy_hold_stand_height_m=1.43),
           _dummy_weight=torch.full((n,), W), _dummy_total_mass=torch.full((n,), M), _dummy_prev_z=None,
           _support_prev_target_z=None, _gait_pelvis_rise=torch.zeros(n), num_envs=n, device="cpu",
           _env=NS(step_dt=0.02, scene=NS(env_origins=torch.tensor(origins, dtype=torch.float32))))
    t.force = lambda z: fn(t, torch.as_tensor(z, dtype=torch.float32), torch.full((n,), K), torch.full((n,), C))
    return t


def test_on_its_target_and_at_rest_the_support_is_the_share_of_the_weight():
    t = _term()
    torch.testing.assert_close(t.force([1.43, 1.53]), torch.full((2,), 0.5 * W))


def test_below_its_target_the_spring_adds_to_the_share():
    t = _term()
    torch.testing.assert_close(t.force([1.41, 1.53]), torch.tensor([0.5 * W + K * 0.02, 0.5 * W]))


def test_the_target_is_the_stand_height_plus_the_pelvis_rise_over_the_envs_floor():
    t = _term()
    t._gait_pelvis_rise = torch.tensor([-0.03, 0.01])
    torch.testing.assert_close(t.force([1.40, 1.54]), torch.full((2,), 0.5 * W))


def test_a_body_that_follows_its_moving_target_feels_only_the_share():
    t = _term()
    for rise in (0.0, -0.01, -0.025, -0.01):
        t._gait_pelvis_rise = torch.full((2,), rise)
        got = t.force([1.43 + rise, 1.53 + rise])
    torch.testing.assert_close(got, torch.full((2,), 0.5 * W))


def test_a_body_that_stays_while_its_target_moves_is_damped_toward_the_targets_speed():
    t = _term()
    t.force([1.43, 1.53])
    t._gait_pelvis_rise = torch.full((2,), -0.01)
    torch.testing.assert_close(t.force([1.43, 1.53]),
                               torch.full((2,), 0.5 * W + K * (-0.01) - C * (0.0 - (-0.01) / 0.02)))


def test_the_first_call_after_a_reset_reads_no_target_speed():
    t = _term()
    t.force([1.43, 1.53])
    t._support_prev_target_z[0] = float("nan")       # what the reset leaves for env 0
    t._gait_pelvis_rise = torch.full((2,), -0.01)
    got = t.force([1.43, 1.53])
    torch.testing.assert_close(got[0], torch.tensor(0.5 * W - K * 0.01))
    torch.testing.assert_close(got[1], torch.tensor(0.5 * W - K * 0.01 + C * (-0.01 / 0.02)))


def test_a_body_rising_off_its_target_is_damped():
    t = _term()
    t.force([1.43, 1.53])
    got = t.force([1.44, 1.53])
    # float32 heights: 1.44 - 1.43 is 0.01 to about 1e-7 m, a few mN of spring and damper
    torch.testing.assert_close(got[0], torch.tensor(0.5 * W - K * 0.01 - C * (0.01 / 0.02)), rtol=0, atol=0.01)


def test_the_hold_writes_the_support_only_with_a_share_and_leaves_the_carried_hold_alone():
    src = _method_source("_advance_dummy")
    carried = src.index("if self.cfg.dummy_hold_clearance_m is not None:\n            # The carried hold")
    share = src.index("elif self.cfg.dummy_hold_support_share is not None:")
    assert carried < share
    assert "lin_force[:, 2] = self._support_force_z(dummy.data.root_pos_w.torch[:, 2], lin_stiffness, lin_damping)" \
        in src[share:]
    # the carried hold's own line, unchanged (v1 stays bit-identical)
    assert ("lin_force[:, 2] = self._dummy_weight + lin_stiffness * (z_target - z) - lin_damping * z_vel"
            in src[carried:share])


def test_a_reset_starts_the_support_over():
    src = _method_source("_resample_command")
    assert "self._gait_pelvis_rise[env_ids] = 0.0" in src
    assert 'self._support_prev_target_z[env_ids] = float("nan")' in src


# --- the pelvis rise the gait's targets imply ----------------------------------------------------------------------


def _crouch_table() -> GaitTable:
    """Every walking row a crouch that deepens with the phase, so the pelvis rise changes as the gait goes on."""
    P = 8
    targets = np.zeros((3, P, len(NAMES)), np.float32)
    for p in range(P):
        a = 0.05 * (p + 1)
        for side in ("left", "right"):
            targets[1:, p, NAMES.index(f"hip_{side}:1")] = -a
            targets[1:, p, NAMES.index(f"knee_{side}")] = 2 * a
            targets[1:, p, NAMES.index(f"ankle_{side}:1")] = -a
    return GaitTable(tuple(NAMES), np.array([0.0, 0.5, 1.5]), np.array([0.0, 0.8, 1.6]), targets)


class Dummy:
    def __init__(self, n):
        self.joint_names = list(reversed(NAMES))        # the articulation's own order, not the table's
        self.cfg = NS(spawn=NS(usd_path="v2.usd"))
        self.writes = []

    def set_joint_position_target_index(self, *, target, joint_ids=None, env_ids=None):
        self.writes.append(("target", target.clone()))

    def set_joint_velocity_target_index(self, *, target, joint_ids=None, env_ids=None):
        self.writes.append(("velocity", target.clone()))


def _gait_term(monkeypatch, share):
    from generation.dummy import jointed

    monkeypatch.setattr(jointed, "read_gait", lambda usd: _crouch_table())
    m = _compiled("_ensure_gait", "_gait_drive", "_drive_gait", "_rest_targets")
    n = 2
    t = NS(_gait=None, _gait_dof_ids=None, _gait_previous=None, _gait_spec=None, gait_phase=torch.zeros(n),
           gait_speed=torch.ones(n), _gait_slowing=torch.zeros(n, dtype=torch.bool),
           hold_released=torch.zeros(n, dtype=torch.bool), num_envs=n, device="cpu",
           _gait_pelvis_rise=torch.zeros(n), cfg=NS(dummy_hold_support_share=share),
           _gait_rest=None, _gait_rest_from=None, _gait_rest_count=torch.zeros(n, dtype=torch.long),
           _env=NS(step_dt=0.02, s1a_entry={"walk_ended": torch.zeros(n, dtype=torch.bool)}))
    for name, fn in m.items():
        setattr(t, name, (lambda f: lambda *a, **k: f(t, *a, **k))(fn))
    return t, Dummy(n)


def test_each_gait_update_sets_the_pelvis_rise_of_the_targets_it_sends(monkeypatch):
    t, dummy = _gait_term(monkeypatch, 0.5)
    rises = []
    for _ in range(3):
        t._drive_gait(dummy, torch.tensor([[0.3, 0.0], [0.1, 0.0]]))
        sent = [w for w in dummy.writes if w[0] == "target"][-1][1]
        want = support.pelvis_rise(SPEC, NAMES, sent.numpy())
        torch.testing.assert_close(t._gait_pelvis_rise, torch.as_tensor(want, dtype=t._gait_pelvis_rise.dtype))
        rises.append(t._gait_pelvis_rise.clone())
    assert bool((rises[0] < 0.0).all()) and not torch.equal(rises[0], rises[-1])


def test_without_a_share_the_gait_leaves_the_pelvis_rise_alone(monkeypatch):
    t, dummy = _gait_term(monkeypatch, None)
    t._drive_gait(dummy, torch.tensor([[0.3, 0.0], [0.1, 0.0]]))
    assert torch.count_nonzero(t._gait_pelvis_rise) == 0


# --- gravity's tipping term with the share carried ------------------------------------------------------------------


def _gravity_term(model, share):
    fn = _compiled("_tilt_gravity_compensation")["_tilt_gravity_compensation"]
    t = NS(cfg=NS(dummy_hold_support_share=share, dummy_hold_inertia_model=model),
           _dummy_gravity_compensation=torch.tensor([[700.0, 700.0, 0.0]]))
    return fn(t)


def test_about_the_feet_the_carried_share_takes_its_part_of_gravitys_tipping_torque():
    # pushed up at the centre of mass with share x weight, the body tips about its feet under (1 - share) of it
    torch.testing.assert_close(_gravity_term("whole_body_feet", 0.6), torch.tensor([[280.0, 280.0, 0.0]]))


@pytest.mark.parametrize("model, share", [("whole_body_feet", None), ("upper_body_hips", 0.6), ("upper_body_hips", None)])
def test_without_a_share_or_about_the_hips_gravitys_term_is_the_rigid_dummys(model, share):
    # about the hips the push at the whole body's centre of mass has no lever on the upper body's own weight; and the
    # rigid dummy (no share) keeps the very tensor it had, so its arithmetic stays bit-identical
    t = NS(cfg=NS(dummy_hold_support_share=share, dummy_hold_inertia_model=model),
           _dummy_gravity_compensation=torch.tensor([[700.0, 700.0, 0.0]]))
    assert _compiled("_tilt_gravity_compensation")["_tilt_gravity_compensation"](t) is t._dummy_gravity_compensation


def test_the_tilt_stiffness_takes_the_gravity_term_from_it():
    src = _method_source("_advance_dummy")
    assert "self._dummy_moment_of_inertia * omega * omega + self._tilt_gravity_compensation()" in src


# --- the plane's own frequency --------------------------------------------------------------------------------------


def _plane(freq):
    fn = _compiled("_plane_gains")["_plane_gains"]
    t = NS(cfg=NS(dummy_hold_plane_freq_hz=freq, dummy_hold_damping_ratio=1.0), _dummy_total_mass=torch.tensor([M]))
    k, c = torch.tensor([1000.0]), torch.tensor([100.0])
    return fn(t, k, c), (k, c)


def test_without_a_plane_frequency_the_plane_keeps_the_holds_gains_as_they_are():
    (k, c), (k0, c0) = _plane(None)
    assert k is k0 and c is c0           # the very tensors: the rigid dummy's arithmetic stays bit-identical


def test_a_plane_frequency_sets_the_planes_spring_and_damper_from_the_mass():
    (k, c), _ = _plane(1.0)
    omega = 2.0 * math.pi * 1.0
    torch.testing.assert_close(k, torch.tensor([M * omega ** 2]))
    torch.testing.assert_close(c, torch.tensor([2.0 * 1.0 * M * omega]))


def test_the_plane_takes_its_gains_and_the_vertical_keeps_the_holds():
    src = _method_source("_advance_dummy")
    assert "plane_stiffness, plane_damping = self._plane_gains(lin_stiffness, lin_damping)" in src
    assert ("lin_force_xy = plane_stiffness.unsqueeze(-1) * xy_error - plane_damping.unsqueeze(-1) * (xy_vel - target_vel_xy)"
            in src)
    assert "self._support_force_z(dummy.data.root_pos_w.torch[:, 2], lin_stiffness, lin_damping)" in src



# --- the stance at rest -----------------------------------------------------------------------------------------------


def _sent(dummy):
    return [w[1] for w in dummy.writes if w[0] == "target"][-1].numpy()


def test_a_walk_at_rest_goes_to_the_stance_its_last_step_left_over_half_a_second(monkeypatch):
    t, dummy = _gait_term(monkeypatch, 0.7)
    for _ in range(3):
        t._drive_gait(dummy, torch.tensor([[0.3, 0.0], [0.3, 0.0]]))
    last = _sent(dummy)
    # env 0 reaches its stop line: the walk ends and its speed is 0; env 1 walks on
    t._env.s1a_entry["walk_ended"] = torch.tensor([True, False])
    t.gait_speed = torch.tensor([0.0, 1.0])
    rest = support.rest_stance(SPEC, NAMES, last[[0]])[0]
    seen = []
    for k in range(30):
        t._drive_gait(dummy, torch.tensor([[0.0, 0.0], [0.3, 0.0]]))
        seen.append(_sent(dummy)[0])
    steps = _compiled()["_GAIT_REST_BLEND_STEPS"]
    assert steps == 25                                               # half a second at the 50 Hz update
    np.testing.assert_allclose(seen[0], last[0] + (rest - last[0]) / 25.0, atol=1e-5)
    np.testing.assert_allclose(seen[24], rest, atol=1e-5)
    np.testing.assert_allclose(seen[29], rest, atol=1e-5)          # and stays
    assert not np.allclose(_sent(dummy)[1], last[1])                # the walking env walked on


def test_the_pelvis_rise_follows_the_stance_at_rest(monkeypatch):
    t, dummy = _gait_term(monkeypatch, 0.7)
    t._drive_gait(dummy, torch.tensor([[0.3, 0.0], [0.3, 0.0]]))
    t._env.s1a_entry["walk_ended"] = torch.tensor([True, True])
    t.gait_speed = torch.zeros(2)
    for _ in range(30):
        t._drive_gait(dummy, torch.zeros(2, 2))
    want = support.pelvis_rise(SPEC, NAMES, _sent(dummy))
    torch.testing.assert_close(t._gait_pelvis_rise, torch.as_tensor(want, dtype=t._gait_pelvis_rise.dtype))


def test_a_reset_forgets_the_stance_at_rest():
    src = _method_source("_resample_command")
    assert "self._forget_gait_rest(env_ids)" in src
    fn = _compiled("_forget_gait_rest")["_forget_gait_rest"]
    t = NS(_gait_rest=torch.ones(2, 3), _gait_rest_from=torch.ones(2, 3), _gait_rest_count=torch.tensor([7, 9]))
    fn(t, torch.tensor([1]))
    assert torch.isnan(t._gait_rest[1]).all() and not torch.isnan(t._gait_rest[0]).any()
    assert t._gait_rest_count.tolist() == [7, 0]
