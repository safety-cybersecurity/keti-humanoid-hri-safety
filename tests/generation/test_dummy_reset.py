"""generation/envs/s1d/dummy_reset.py — S1d stands the dummy upright before reset_dummy_entry
measures its reach (a fallen dummy's reach inflated the waiting offset from 3.117 m to 3.91 m at
auto-resets, measured 2026-09-23). Isaac-free parts only; the effect is checked in Isaac."""
import inspect

import numpy as np
import pytest

from generation.envs.s1d import dummy_reset


def test_upright_root_state_keeps_the_place_and_stands_still():
    state = dummy_reset.upright_root_state(np.array([[1.0, 2.0, 0.09], [3.0, -4.0, 1.4]], np.float32))
    assert state.shape == (2, 13) and state.dtype == np.float32
    assert state[:, :3].tolist() == [[1.0, 2.0, 0.09000000357627869], [3.0, -4.0, 1.399999976158142]]
    assert state[:, 3:7].tolist() == [[0.0, 0.0, 0.0, 1.0]] * 2          # identity, xyzw like the engine
    assert not state[:, 7:].any()                                         # no linear or angular velocity


def test_upright_root_state_turns_to_the_facing_it_is_given():
    # a fixed facing (the 180 deg S1d once stood at, now check_s1d_scene's control): the pose
    # reset_dummy_entry then writes. Turned to its walk, the dummy is stood up at no turn (stand_up_first)
    state = dummy_reset.upright_root_state(np.array([[1.0, 2.0, 0.09]], np.float32), yaw_deg=180.0)
    assert state[:, 3:7].tolist() == [[0.0, 0.0, 1.0, 0.0]]
    assert state[:, :3].tolist() == [[1.0, 2.0, 0.09000000357627869]] and not state[:, 7:].any()


def test_the_wrapped_term_keeps_the_entry_term_s_signature():
    # Isaac Lab checks an event term's params against its function's signature; a wrapper that
    # showed (*args, **kwargs) instead would be refused at env creation.
    def reset_dummy_entry(env, env_ids, speed_range, angle_range_deg, human_region_radius=0.0, asset_name="dummy"):
        return None

    wrapped = dummy_reset.stand_up_first(reset_dummy_entry)
    assert inspect.signature(wrapped) == inspect.signature(reset_dummy_entry)
    assert wrapped.__wrapped__ is reset_dummy_entry


class _FakeDummy:
    def __init__(self, names):
        self.joint_names, self.device, self.calls = names, "cpu", []

    def write_joint_state_to_sim_index(self, *, position, velocity, env_ids):
        self.calls.append(("state", position.clone(), velocity.clone(), list(env_ids)))

    def set_joint_position_target_index(self, *, target, env_ids):
        self.calls.append(("target", target.clone(), list(env_ids)))

    def set_joint_velocity_target_index(self, *, target, env_ids):
        self.calls.append(("velocity target", target.clone(), list(env_ids)))


def test_a_dummy_without_joints_gets_no_joint_writes():
    pytest.importorskip("torch")
    d = _FakeDummy([])
    dummy_reset.zero_joints(d, [0, 2])
    assert d.calls == []


def test_joints_go_to_the_stand_pose_for_the_reset_envs_only():
    pytest.importorskip("torch")
    d = _FakeDummy(["hip_left_x", "knee_left"])
    dummy_reset.zero_joints(d, [1])
    assert [c[0] for c in d.calls] == ["state", "target", "velocity target"]
    state, target, velocity = d.calls
    assert state[1].shape == (1, 2) and float(state[1].abs().sum()) == 0.0 and float(state[2].abs().sum()) == 0.0
    assert state[3] == [1] and target[2] == [1]
    # a walk the last episode left asking for joint velocity must not carry into the next one's stand
    assert velocity[1].shape == (1, 2) and float(velocity[1].abs().sum()) == 0.0 and velocity[2] == [1]


def test_the_wrapped_reset_stands_the_dummy_up_then_zeroes_its_joints_then_measures():
    """stand_up_first: the root upright, then the joints to the stand pose (a jointed dummy left lying down would
    measure its reach with its limbs where the fall left them), then reset_dummy_entry."""
    torch = pytest.importorskip("torch")
    order = []

    class Dummy(_FakeDummy):
        def __init__(self):
            super().__init__(["hip_left:0", "knee_left"])
            self.data = type("D", (), {"root_pos_w": type("W", (), {"torch": torch.zeros(3, 3)})()})()

        def write_root_state_to_sim(self, state, env_ids):
            order.append("root")

        def write_joint_state_to_sim_index(self, **kw):
            order.append("joint state")

        def set_joint_position_target_index(self, **kw):
            order.append("joint target")

        def set_joint_velocity_target_index(self, **kw):
            order.append("joint velocity target")

    dummy = Dummy()
    env = type("E", (), {"scene": {"dummy": dummy}})()

    def reset_dummy_entry(env, env_ids, asset_name="dummy"):
        order.append("entry")

    dummy_reset.stand_up_first(reset_dummy_entry)(env, [0, 2], asset_name="dummy")
    assert order == ["root", "joint state", "joint target", "joint velocity target", "entry"]
