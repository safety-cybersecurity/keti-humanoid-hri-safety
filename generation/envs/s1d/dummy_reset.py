"""S1d's one change to S1a's reset: stand the dummy upright before its reach is measured.

reset_dummy_entry (generation/envs/s1a/mdp/events.py) derives the dummy's waiting offset from both
bodies' lateral reach, "measured once from a standing pose at reset". In the run loop's explicit
resets the dummy does stand. The auto-reset inside env.step, though, measures it where the episode
left it: lying down after a fall its reach reads 0.83-1.01 m instead of 0.215 m, and the waiting
offset grows from 3.117 m to 3.74-3.91 m (measured 2026-09-23). run_s1a
replaces that reset with an explicit one before any physics step, so S1a's data never used it --
but S1d's aisle guard checks every reset, and it stopped on one. Standing the dummy up first makes
every reset measure what the docstring says; an explicit reset, where it already stands, writes the
state it already has.

A jointed dummy (use_jointed) also has its joints put back to the stand pose there -- positions,
velocities and targets -- so that it measures standing, not with its limbs where a fall left them; the
rigid dummy has no joints and gets no joint writes.

Torch is imported inside the term, so the module (and its tests) load without Isaac.
"""
from __future__ import annotations

import functools

import numpy as np

from generation.envs.s1a.mdp.entry_geometry import facing_quat_xyzw


def upright_root_state(pos, yaw_deg: float = 0.0) -> np.ndarray:
    """(n, 3) root positions -> (n, 13) root states [pos, quat xyzw, lin vel, ang vel]: same place,
    upright and facing yaw_deg (the pose reset_dummy_entry itself writes), at rest."""
    pos = np.asarray(pos, np.float32).reshape(-1, 3)
    quat = np.tile(np.asarray(facing_quat_xyzw(yaw_deg), np.float32), (len(pos), 1))
    return np.concatenate([pos, quat, np.zeros((len(pos), 6), np.float32)], axis=1)


def zero_joints(dummy, env_ids) -> None:
    """Joint positions, velocities and targets (position and velocity) to the stand pose for env_ids -- only when
    the dummy has joints: the rigid dummy's reset writes no joint state."""
    if not len(dummy.joint_names):
        return
    import torch

    n = len(env_ids)
    zeros = torch.zeros(n, len(dummy.joint_names), device=dummy.device)
    dummy.write_joint_state_to_sim_index(position=zeros, velocity=zeros, env_ids=env_ids)
    dummy.set_joint_position_target_index(target=zeros, env_ids=env_ids)
    dummy.set_joint_velocity_target_index(target=zeros, env_ids=env_ids)


def stand_up_first(reset_entry):
    """Wrap reset_dummy_entry so the dummy is upright when it measures: at the term's fixed facing_yaw_deg
    if it has one, and at no turn where the term turns each dummy to its own walk (face_travel) -- that
    reach is taken in the horizontal plane, the same at every turn. functools.wraps keeps the signature
    Isaac Lab checks the term's params against."""

    @functools.wraps(reset_entry)
    def reset(env, env_ids, *args, **kwargs):
        import torch

        dummy = env.scene[kwargs.get("asset_name", "dummy")]
        pos = dummy.data.root_pos_w.torch[env_ids]
        state = upright_root_state(pos.cpu().numpy(), kwargs.get("facing_yaw_deg", 0.0))
        state = torch.from_numpy(state).to(device=pos.device)
        dummy.write_root_state_to_sim(state, env_ids=env_ids)
        zero_joints(dummy, env_ids)
        return reset_entry(env, env_ids, *args, **kwargs)

    return reset
