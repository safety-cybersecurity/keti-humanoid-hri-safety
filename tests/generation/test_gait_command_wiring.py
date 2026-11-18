"""generation/envs/s1a/mdp/commands.py with the jointed dummy's gait (cfg.dummy_gait). The command term runs only in
Isaac, so the lines that keep the rigid dummy's own behaviour and the lines that drive the gait are pinned from the
source, as test_entry_geometry pins the hold servo's. Isaac-free."""
import ast
from pathlib import Path

SRC = (Path(__file__).resolve().parents[2] / "generation/envs/s1a/mdp/commands.py").read_text(encoding="utf-8")


def test_the_rigid_dummy_keeps_its_constant_scripted_step():
    assert ("        if self.cfg.dummy_gait:\n            step = self._gait_step_length(entry, moving)\n"
            "        else:\n            step = entry[\"speed\"] * self._env.step_dt\n") in SRC


def test_the_gait_follows_the_scripted_targets_travel_once_it_has_moved():
    i_move = SRC.index('        entry["target_xy"][:, 1] -= target_dxy[:, 1]\n')
    i_drive = SRC.index("        if self.cfg.dummy_gait:\n            self._drive_gait(dummy, target_dxy)\n")
    i_quat = SRC.index('        target_quat = entry["facing_quat"]\n')
    assert i_move < i_drive < i_quat


def test_a_released_dummy_holds_the_pose_it_was_let_go_in_at_muscle_tone():
    block = SRC[SRC.index("    def _update_hold_release"):SRC.index("    def _robot_contact_present")]
    # The gait is read here too if the release comes first: the latch does not come back to freeze it later.
    assert "        if self.cfg.dummy_gait:\n            self._ensure_gait(dummy)\n" in block
    assert "current = dummy.data.joint_pos.torch[ids][:, self._gait_dof_ids]" in block
    assert "self._write_gains(ids, self.cfg.dummy_tone_gains)" in block


def test_each_episode_starts_at_phase_zero_at_rest_with_the_walking_gains():
    block = SRC[SRC.index("    def _resample_command"):SRC.index("    def _update_command")]
    assert ("        if self.cfg.dummy_gait:\n            self.gait_phase[env_ids] = 0.0\n"
            "            self.gait_speed[env_ids] = 0.0\n") in block
    assert "self._write_gains(env_ids, self.cfg.dummy_walk_gains)" in block


def test_the_upper_body_inertia_only_with_its_model_the_rigid_dummy_keeping_its_values():
    assert "            axes_mass, axes_offset = body_mass, com_centred_offset\n" in SRC
    assert '            if self.cfg.dummy_hold_inertia_model == "upper_body_hips":\n' in SRC
    assert "                self._dummy_body_mass = axes_mass\n" in SRC
    assert "capture_quat.unsqueeze(1).expand(-1, offset.shape[1], -1), axes_offset" in SRC
    assert "            if upper_lever is not None:\n                gravity_destabilizing = " in SRC


def test_the_distance_is_to_the_body_regions_only_when_regions_are_named():
    block = SRC[SRC.index("    def _measure_min_distance"):]
    assert "if self.cfg.dummy_region_prims and self._dummy_region_mask is None:" in block


def test_the_gait_is_driven_with_what_the_base_term_always_used():
    block = SRC[SRC.index("    def _gait_drive"):SRC.index("    def _drive_gait")]
    for line in ("self.gait_speed.cpu().numpy()", "self._gait_slowing.cpu().numpy()",
                 "torch.linalg.norm(target_dxy, dim=-1).cpu().numpy()"):
        assert line in block
    assert "        v, slowing, distance = self._gait_drive(target_dxy)\n" in SRC


def test_the_gait_joint_indices_are_int32_and_read_once_for_drive_and_release():
    # Run under Isaac's python in test_gait_command_behaviour; pinned here for CI.
    block = SRC[SRC.index("    def _ensure_gait"):SRC.index("    def _drive_gait")]
    assert "dtype=torch.int32" in block
    drive = SRC[SRC.index("    def _drive_gait"):SRC.index("    def _write_gains")]
    assert "self._ensure_gait(dummy)" in drive and "read_gait" not in drive


def _base_methods() -> dict:
    cls = next(n for n in ast.parse(SRC).body if isinstance(n, ast.ClassDef) and n.name == "S1aStraightWalkCommand")
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def test_the_hold_takes_a_turning_facings_torque_only_through_its_seam():
    methods = _base_methods()
    # The base term's facing does not turn within an episode: its seam answers None, nothing else.
    seam = methods["_facing_turn_torque"]
    assert isinstance(seam.body[0], ast.Expr) and [ast.unparse(n) for n in seam.body[1:]] == ["return None"]
    body = methods["_advance_dummy"].body
    lines = [ast.unparse(n) for n in body]
    # The spring and damper line as it is, once; then the seam, added only when it answers -- before the cap and the
    # clamp, so the saturation flag and the applied torque are the whole torque's.
    i = lines.index("ang_torque = ang_stiffness * ang_error - ang_damping * ang_vel")
    assert sum(line.startswith("ang_torque = ang_stiffness") for line in lines) == 1
    assert lines[i + 1] == "turning = self._facing_turn_torque(ang_damping)"
    branch = body[i + 2]
    assert isinstance(branch, ast.If) and ast.unparse(branch.test) == "turning is not None" and not branch.orelse
    assert [ast.unparse(n) for n in branch.body] == ["ang_torque = ang_torque + turning"]
    assert i + 2 < lines.index("self.dummy_torque_saturated_per_axis = ang_torque.abs() >= torque_cap_axes")
    assert i + 2 < next(k for k, line in enumerate(lines) if line.startswith("ang_torque = torch.clamp("))
