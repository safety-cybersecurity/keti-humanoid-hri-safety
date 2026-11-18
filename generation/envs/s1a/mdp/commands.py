"""Straight-line walking command with an SSM supervisor on top.

The policy consumes the velocity command as an observation term, so slowing
the robot down means scaling this command — the policy itself is untouched.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

from .entry_geometry import latch_walk_end, lateral_stop_fraction, yawed_axis_inertia
from .safety import (command_scale, fill_missing_only, load_ssm_parameters,
                     min_distance_human, separation_required)

if TYPE_CHECKING:
    import numpy as np
    from isaaclab.envs import ManagerBasedRLEnv


def _clamp_vector_norm(v: torch.Tensor, max_norm: torch.Tensor) -> torch.Tensor:
    """Scale each row of v down (never up) so its norm does not exceed max_norm.

    v: (num_envs, 3). max_norm: (num_envs,). Direction is preserved — this
    limits magnitude, not any individual component. Used to saturate the
    dummy-hold linear force (S1aStraightWalkCommand._advance_dummy) so an
    edge-case position error cannot produce an unbounded — and unphysical —
    correction. The angular counterpart (torque) is capped per-axis instead,
    precisely because this function's whole-vector norm would let a large
    error on one rotational axis eat the cap headroom of a much softer one —
    see the torque_cap_axes derivation at the _advance_dummy call site.
    """
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    scale = (max_norm.unsqueeze(-1) / norm).clamp(max=1.0)
    return v * scale


# Gait updates over which a walk that has come to rest goes to the stance its last step left (half a second at the
# 50 Hz update): long enough not to snap the legs, short against the stand that follows.
_GAIT_REST_BLEND_STEPS = 25


def check_hold_support(cfg) -> None:
    """The jointed dummy's partial weight support (dummy_hold_support_share): a share of the weight in [0, 1), with
    the stand height its target starts from, and never together with the carried hold -- two different holds."""
    share = cfg.dummy_hold_support_share
    if share is None:
        return
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not (math.isfinite(share) and 0.0 <= share < 1.0):
        raise ValueError(f"dummy_hold_support_share must be a share of the weight in [0, 1), got {share!r}")
    if cfg.dummy_hold_clearance_m is not None:
        raise ValueError("dummy_hold_support_share and dummy_hold_clearance_m are two different holds -- set one")
    stand = cfg.dummy_hold_stand_height_m
    if stand is None or not (math.isfinite(stand) and stand > 0.0):
        raise ValueError(f"dummy_hold_support_share needs dummy_hold_stand_height_m > 0, got {stand!r}")


class S1aStraightWalkCommand(CommandTerm):
    """Forward velocity, scaled down as the human closes in."""

    cfg: "S1aStraightWalkCommandCfg"

    def __init__(self, cfg: S1aStraightWalkCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._walk_speed = torch.zeros(self.num_envs, device=self.device)
        self.min_distance = torch.zeros(self.num_envs, device=self.device)
        self.separation_required = torch.zeros(self.num_envs, device=self.device)
        self.protective_stop_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        # Once the robot is close enough that contact is imminent, the hold
        # servo is switched off for the rest of the episode and the dummy is
        # a free body. See _update_hold_release() for why, and
        # dummy_release_gap_m for how close "close enough" is.
        self.hold_released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Built on first use from the sensor's own body order -- never a
        # hand-written list of link names (CONTRIBUTING.md 「관절 순서」).
        self._non_foot_mask: torch.Tensor | None = None
        # (num_envs, 3) N — the linear hold force that the last physics step
        # actually applied, as opposed to the one queued for the next one.
        # See _update_command for why the two differ.
        self.hold_force_in_effect = torch.zeros(self.num_envs, 3, device=self.device)
        # (num_envs, 3) N*m — the torque half. The composer receives both, so
        # "was the hold off" is a question about both.
        self.hold_torque_in_effect = torch.zeros(self.num_envs, 3, device=self.device)
        self.hold_release_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        # A delay queue, not a filter: the supervisor sees a stale distance,
        # which is what a real sensor chain does.
        self._delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._history: list[torch.Tensor] = []

        self.ssm_params, self.ssm_missing = load_ssm_parameters(cfg.rules_path)
        if self.ssm_missing:
            # Provisional numbers keep the pipeline moving before the standard
            # copy lands -- but only for the names that are actually empty.
            self.ssm_params = fill_missing_only(
                self.ssm_params, cfg.provisional_ssm_params, self.ssm_missing)
        # Read by run_s1a.py, which hands it to EpisodeRecorder: the recorder
        # writes it to the episode metadata and the converter carries it into
        # the released episode's notes.provisional, so a provisional distance
        # is marked as one in the data.
        self.provisional = bool(self.ssm_missing)

        # Cached lazily on first _advance_dummy call (the dummy asset isn't
        # guaranteed initialized during __init__) — see _find_root_body_id
        # and _advance_dummy for why these are found/measured rather than
        # hard-coded.
        self._dummy_root_body_id: int | None = None
        self._dummy_total_mass: torch.Tensor | None = None
        self._dummy_height_span: torch.Tensor | None = None
        self._dummy_weight: torch.Tensor | None = None
        self._dummy_moment_of_inertia: torch.Tensor | None = None  # (num_envs, 3) — per-axis, see _advance_dummy
        # (num_envs, 3) — aggregate CoM minus root, in the root's *body*
        # frame (cached as a world-frame vector it would be correct only at
        # the yaw=0 spawn instant it was captured at — see _advance_dummy's
        # docstring for why that drifts wrong as soon as yaw does not stay
        # at 0).
        self._dummy_com_offset_body: torch.Tensor | None = None
        self._dummy_gravity_compensation: torch.Tensor | None = None  # (num_envs, 3) — m*g*h on x/y, 0 on z
        # Finite-difference velocity state (see _advance_dummy's docstring
        # for why PhysX's own reported root velocity is not used for the
        # damping terms). None until the first post-reset
        # _advance_dummy call has a real previous sample to difference
        # against; _resample_command seeds it fresh on every reset so the
        # first step after a reset never differences across the teleport.
        self._dummy_prev_pos_xy: torch.Tensor | None = None
        self._dummy_prev_quat: torch.Tensor | None = None
        # (num_envs, 3) bool — set every _advance_dummy call, read by
        # check_dummy_hold.py for saturation-rate reporting.
        # None until the first call; nothing reads it before then.
        self.dummy_torque_saturated_per_axis: torch.Tensor | None = None
        # (num_envs, 3) N, post-cap — set every _advance_dummy call, read by
        # check_dummy_hold.py for moving-phase hold-force reporting. None
        # until the first call.
        self.dummy_lin_force_applied: torch.Tensor | None = None
        # (num_envs, 3) N*m, post-cap — the angular half of the same wrench.
        self.dummy_torque_applied: torch.Tensor | None = None
        # The root height of the previous call, for the carried hold's vertical damping (dummy_hold_clearance_m).
        self._dummy_prev_z: torch.Tensor | None = None
        clearance = cfg.dummy_hold_clearance_m
        if clearance is not None:
            if not (math.isfinite(clearance) and clearance >= 0.0):
                raise ValueError(f"dummy_hold_clearance_m must be a finite height >= 0, got {clearance!r}")
            stand = cfg.dummy_hold_stand_height_m
            if stand is None or not (math.isfinite(stand) and stand > 0.0):
                raise ValueError(f"dummy_hold_clearance_m needs dummy_hold_stand_height_m > 0, got {stand!r}")
        check_hold_support(cfg)
        # The partial weight support's height target (dummy_hold_support_share): the pelvis rise of the joint targets
        # the gait last sent, per env (0 until then, and after a reset), and the target of the previous call, for its
        # speed -- None before the first call, NaN for an env just reset.
        self._gait_pelvis_rise = torch.zeros(self.num_envs, device=self.device)
        self._support_prev_target_z: torch.Tensor | None = None
        self._gait_spec = None
        # The jointed dummy's gait (cfg.dummy_gait). Built on first use from the spawned USD.
        self._gait = None
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        self.gait_speed = torch.zeros(self.num_envs, device=self.device)
        # Which envs' scripted speed fell at the last update: below the slowest row the gait shrinks differently as a
        # walk slows to its stop than as it starts (gait_table).
        self._gait_slowing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # The stance a walk ends in at rest (support.rest_stance of its last walking targets), the targets it is
        # blended from, and the gait updates since it came to rest -- per env; NaN rows before the rest.
        self._gait_rest = None
        self._gait_rest_from = None
        self._gait_rest_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._gait_dof_ids = None
        # The joint targets of the last gait update, per env -- the drives are handed the rate the targets move at;
        # NaN rows (and None before the first update) mean no last update since the env's reset.
        self._gait_previous = None
        self._dummy_region_mask = None

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: Sequence[int]):
        low, high = self.cfg.walk_speed_range
        self._walk_speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(low, high)
        low_ms, high_ms = self.cfg.sensor_delay_ms_range
        delay_s = torch.empty(len(env_ids), device=self.device).uniform_(low_ms, high_ms) / 1000.0
        self._delay_steps[env_ids] = (delay_s / self._env.step_dt).long()
        self.protective_stop_step[env_ids] = -1
        self.hold_released[env_ids] = False
        self.hold_release_step[env_ids] = -1

        # _reset_idx() applies the "reset" event mode (reset_root_state_uniform
        # teleports the robot) before calling command_manager.reset() -> this,
        # within the same step() call (confirmed in manager_based_rl_env.py), so
        # a measurement taken here already reflects the new episode's
        # post-teleport poses. Without this, a stale pre-reset distance sitting
        # in the delay queue (up to sensor_delay_ms_range worth of steps) could
        # falsely latch a protective stop one step into a brand-new episode —
        # protective_stop_step is a once-only latch, so that mis-record would
        # be permanent for the episode. Overwriting every queued entry for
        # exactly the envs being reset (not clearing the whole list) keeps
        # other envs' history untouched.
        fresh = self._measure_min_distance()
        for entry in self._history:
            entry[env_ids] = fresh[env_ids]

        # Seed the finite-difference velocity state with the
        # dummy's just-teleported pose for exactly the envs being reset —
        # same ordering guarantee as _measure_min_distance above
        # (reset_dummy_entry's event already ran). Without this, the first
        # _advance_dummy call of a fresh episode would difference against
        # the *previous* episode's last position, a discontinuous jump
        # across an unrelated teleport that would read as a spurious
        # multi-metre-per-second velocity for exactly one step. None-guarded
        # because this can run before _advance_dummy has ever populated the
        # cache once (e.g. the very first reset) — it seeds itself on its
        # own first call in that case.
        if self._dummy_prev_z is not None:
            self._dummy_prev_z[env_ids] = self._env.scene["dummy"].data.root_pos_w.torch[env_ids, 2]
        if self._dummy_prev_pos_xy is not None:
            dummy = self._env.scene["dummy"]
            self._dummy_prev_pos_xy[env_ids] = dummy.data.root_pos_w.torch[env_ids, :2]
            self._dummy_prev_quat[env_ids] = dummy.data.root_quat_w.torch[env_ids]
        if self.cfg.dummy_gait:
            self.gait_phase[env_ids] = 0.0
            self.gait_speed[env_ids] = 0.0
            self._gait_slowing[env_ids] = False
            self._forget_gait_rate(env_ids)
            self._forget_gait_rest(env_ids)
            self._gait_pelvis_rise[env_ids] = 0.0
        if self._support_prev_target_z is not None:
            self._support_prev_target_z[env_ids] = float("nan")
            if self._gait_dof_ids is not None:
                self._write_gains(env_ids, self.cfg.dummy_walk_gains)

    def _update_command(self):
        # Snapshot the wrench that the physics which just ran actually used,
        # before either _update_hold_release zeroes it or _advance_dummy
        # replaces it.
        #
        # CommandManager.compute() runs *after* the decimation loop
        # (manager_based_rl_env.py), so what _advance_dummy writes here is the
        # force for the *next* step's substeps. dummy_lin_force_applied is
        # therefore always one step ahead of the step being recorded, and
        # EpisodeRecorder reading it directly mislabelled every hold_force by
        # one row. It also made hold_contaminated_steps blind to the exact
        # case it was added for: on the step the contact backstop fires, the
        # release zeroes that buffer before the recorder ever sees the
        # non-zero force the dummy was actually being pushed with.
        self.hold_force_in_effect = (
            torch.zeros(self.num_envs, 3, device=self.device)
            if self.dummy_lin_force_applied is None
            else self.dummy_lin_force_applied.clone()
        )
        self.hold_torque_in_effect = (
            torch.zeros(self.num_envs, 3, device=self.device)
            if self.dummy_torque_applied is None
            else self.dummy_torque_applied.clone()
        )

        # Measured before _advance_dummy, not after, so the release decision
        # uses this step's geometry rather than last step's. Moving it is
        # numerically free: _advance_dummy only writes a wrench (which takes
        # effect during the *next* step's substeps) and advances a scripted
        # target, so no pose it would measure changes inside that call.
        measured = self._measure_min_distance()
        self._update_hold_release(measured)
        self._advance_dummy()
        self._history.append(measured)
        max_delay = int(self._delay_steps.max().item())
        if len(self._history) > max_delay + 1:
            self._history.pop(0)

        delayed = torch.stack([
            self._history[max(0, len(self._history) - 1 - int(d))][i]
            for i, d in enumerate(self._delay_steps)
        ])
        self.min_distance = measured

        # human_speed comes from ssm_params (the standard copy once meta/rules.json
        # is filled, provisional_ssm_params until then) — not a literal in this
        # file. separation_required() only validates that params["human_speed"]
        # is present; it never reads it back out, so the caller (here) is what
        # actually has to route the canonical value through.
        human_speed = float(self.ssm_params["human_speed"])
        scale = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.num_envs):
            self.separation_required[i] = separation_required(
                self.ssm_params,
                human_speed=human_speed,
                robot_speed=float(self._walk_speed[i]),
            )
            scale[i] = command_scale(
                distance=float(delayed[i]),
                separation_required=float(self.separation_required[i]),
                slowdown_factor=float(self.cfg.slowdown_factor),
                stop_band_factor=float(self.cfg.stop_band_factor),
            )

        self._command[:, 0] = self._walk_speed * scale
        self._command[:, 1] = 0.0
        self._command[:, 2] = self._heading_correction()

        stopping = (scale == 0.0) & (self.protective_stop_step < 0)
        self.protective_stop_step[stopping] = int(self._env.common_step_counter)

    def _find_root_body_id(self, dummy) -> int:
        """Which index into dummy.body_names / body_pos_w is the articulation's root link.

        root_pos_w reads from a dedicated PhysX root-transform query
        (isaaclab_physx articulation_data.py: self._root_view.get_root_transforms()),
        not from indexing the general per-body tensor — so which body-array
        index that corresponds to is not documented anywhere. Found
        empirically instead of assumed, same reasoning as CONTRIBUTING.md 「관절 순서」
        against hand-typed joint order: whichever body's own position
        exactly matches root_pos_w right now is the root link, checked once
        and cached (a fixed-joint articulation's root identity cannot change
        after spawn). Fails loudly rather than silently guessing index 0 if
        the match isn't unique.
        """
        matches = torch.isclose(
            dummy.data.body_pos_w.torch[0],
            dummy.data.root_pos_w.torch[0].unsqueeze(0),
            atol=1e-5,
        ).all(dim=-1)
        candidates = matches.nonzero(as_tuple=True)[0]
        if candidates.numel() != 1:
            raise RuntimeError(
                f"dummy 의 root LINK 를 body_pos_w 에서 유일하게 찾지 못했습니다"
                f"(후보 {candidates.numel()}개) — PD 홀드가 힘을 줄 프림을 특정할 수 없습니다."
            )
        return int(candidates.item())

    def _update_hold_release(self, measured: torch.Tensor) -> None:
        """Switch the hold servo off for good once contact is imminent.

        The dummy is held upright and dragged along its scripted crossing by
        an external wrench (see _advance_dummy). That servo is fine while
        nothing is touching the dummy and ruinous the moment something is:
        whatever force it happens to be exerting adds to, and is reported
        alongside, the robot-human contact this dataset exists to measure.
        Measured 2026-08-20 (check_dummy_hold.py, moving stage): the hold
        force averaged 1213 N and peaked at 5810 N against the dummy's own
        weight of 859 N, and the dummy was **off the ground entirely in 54 of
        150 steps (36%)** -- dragging a rigid figure horizontally against
        ground friction tips it, the attitude controller fights the tip, and
        the pair bounce it. A contact magnitude recorded under that is not a
        measurement of anything.

        Releasing makes the dummy a free body from that moment on, so the
        contact is what ISO/TS 15066 calls **transient**: robot-human
        interaction plus the human's own inertia, nothing else. That is the
        physically right model for S1a, where a walking person is struck --
        and it is the case Table A.2's transient column and the rules canon's
        `biomechanical.effective_mass` are for. A person struck this way is
        knocked over and does not get set upright again, so the latch holds
        for the rest of the episode.

        **Two triggers, and only one of them is a guarantee.**

        The distance gate fires first in practice. Contact runs at one
        recorded step (0.02 s) in this scenario, so a latch armed by contact
        alone would leave that one step -- often the only one -- still
        carrying the wrench written on the step before. Measured over six
        1000-step episodes, the gate released 3 steps (0.06 s) before the
        first contact.

        But it is best-effort, not a bound.
        `measured` is **not** a surface-to-surface
        distance: the schema is explicit that min_distance_human takes the
        robot's link *origins* as points (no robot collider size at all) and
        collapses every human region to one 0.1 m sphere, while the dummy's
        real extents reach 0.265 m along a capsule axis (schema v0.1,
        "safety.min_distance_human 산출 방법"). Measured on the assets, the
        largest body-origin-to-surface distance is 0.755 m on the robot and
        0.866 m on the dummy -- adding those back would make the gate fire
        more than 1.6 m out, long enough for a released rigid figure to
        topple before anything touched it. So the gate is kept small and
        honest about what it is.

        The **contact backstop** is the guarantee: any robot-side contact
        releases immediately, so the hold can never continue *through* a
        contact, and at most the one step that first detects it can carry a
        wrench. EpisodeRecorder counts exactly those steps
        (`hold_contaminated_steps` in the episode metadata), so the property
        is checkable in the data rather than asserted here.

        A configuration without a gap (dummy_release_gap_m None) has the
        contact backstop alone: the hold goes at the first contact and not
        before it -- the scenarios with a conditions table, whose passive
        dummy would otherwise walk on unassisted through a near miss.
        """
        gap = self.cfg.dummy_release_gap_m
        near = (measured <= gap) if gap is not None else torch.zeros_like(self.hold_released)
        newly = (near | self._robot_contact_present()) & (~self.hold_released)
        if not bool(newly.any()):
            return
        self.hold_released |= newly
        self.hold_release_step[newly] = int(self._env.common_step_counter)
        # Clear whatever the servo left in the composer. It is the *permanent*
        # composer (see _advance_dummy), which is not cleared by
        # write_data_to_sim(), so simply not writing again would leave the
        # last force applied forever.
        dummy = self._env.scene["dummy"]
        if self._dummy_root_body_id is None:
            self._dummy_root_body_id = self._find_root_body_id(dummy)
        zeros = torch.zeros(self.num_envs, 1, 3, device=self.device)
        dummy.permanent_wrench_composer.set_forces_and_torques_index(
            forces=zeros,
            torques=zeros,
            body_ids=[self._dummy_root_body_id],
            is_global=True,
        )
        # These two are what check_dummy_hold.py and EpisodeRecorder read to
        # report the hold. _advance_dummy returns early once released and so
        # stops updating them; without this they would keep reporting the
        # last force from before the release, which is exactly the wrong
        # number to have in the data next to a contact.
        released = self.hold_released.unsqueeze(-1)
        if self.dummy_lin_force_applied is not None:
            self.dummy_lin_force_applied = torch.where(
                released, torch.zeros_like(self.dummy_lin_force_applied),
                self.dummy_lin_force_applied)
        if self.dummy_torque_saturated_per_axis is not None:
            self.dummy_torque_saturated_per_axis = (
                self.dummy_torque_saturated_per_axis & ~released)
        if self.dummy_torque_applied is not None:
            self.dummy_torque_applied = torch.where(
                released, torch.zeros_like(self.dummy_torque_applied),
                self.dummy_torque_applied)
        # The jointed dummy, let go, keeps the pose it was in -- its joint targets frozen there -- at muscle tone
        # instead of the walking gains: a person struck mid-stride, not one who keeps walking.
        if self.cfg.dummy_gait:
            self._ensure_gait(dummy)
            ids = torch.nonzero(newly).flatten()
            current = dummy.data.joint_pos.torch[ids][:, self._gait_dof_ids]
            dummy.set_joint_position_target_index(target=current, joint_ids=self._gait_dof_ids, env_ids=ids)
            dummy.set_joint_velocity_target_index(target=torch.zeros_like(current), joint_ids=self._gait_dof_ids,
                                                  env_ids=ids)
            self._write_gains(ids, self.cfg.dummy_tone_gains)

    def _robot_contact_present(self) -> torch.Tensor:
        """Whether the robot is touching something other than the ground, per env.

        Read off the robot's own contact sensor with its feet masked out, the
        same signal EpisodeRecorder uses for contact_onset and for the same
        reason: neither sensor can separate robot-dummy contact from ground
        contact on its own, but the only thing the robot's non-foot links can
        reach in this scene is the dummy -- unless the robot itself has
        fallen, which ends the episode anyway.

        Deliberately not the dummy's own sensor: that one reports the dummy
        standing on the ground at ~m*g every step and cannot tell it apart
        from being hit.
        """
        sensor = self._env.scene["contact_forces"]
        if self._non_foot_mask is None:
            from generation.envs.s1a.mdp.safety import non_foot_mask

            self._non_foot_mask = torch.tensor(
                non_foot_mask(sensor.body_names), device=self.device
            )
        # Prefer the substep peak the recorder judges contact on. Reading the
        # instantaneous value here while the recorder reads the peak is an
        # asymmetry, not a difference of opinion: a contact living in an inner
        # substep got written to the data while the hold stayed applied, and the
        # contamination counter then measured the gap between two readings rather
        # than the physics.
        #
        # The fallback is for envs built by the check scripts, which install no
        # accumulator and record nothing, so there is no data for the asymmetry
        # to reach.
        acc = getattr(self._env, "_keti_substep_contact", None)
        if acc is not None:
            seen = acc.robot_contact_seen(self.cfg.robot_contact_threshold_n)
            return torch.full((self.num_envs,), seen, dtype=torch.bool, device=self.device)
        forces = torch.linalg.norm(sensor.data.net_forces_w.torch, dim=-1)  # (num_envs, B)
        return (forces * self._non_foot_mask).amax(dim=-1) > self.cfg.robot_contact_threshold_n

    def _advance_dummy(self):
        """Hold the dummy standing with a bounded, saturated spring-damper force/torque
        on its root link, and move its scripted horizontal target once the robot is
        within this episode's trigger distance of the point where the entry line
        crosses its lane.

        Runs unconditionally every step, not only while "moving": the dummy's
        17 body regions are ordinary dynamic rigid bodies (mdp/events.py
        explains why a kinematic root does not work for this asset), so
        without a continuous correction gravity topples it (observed with no
        hold: visible tilt within ~80 steps, lying down well before a
        1000-step episode ends).

        A force, not a state write: holding the dummy by rewriting its
        root position, orientation and velocity directly every step
        (write_root_state_to_sim) is a position *constraint*, not a
        physical interaction — PhysX has to resolve it as if the body had
        infinite mass and infinite stiffness. Measured consequence
        (generation/envs/s1a/check_contact_channel.py,
        check_forced_overlap.py): a dummy merely resting under its own
        weight reported exactly 0 N (no interpenetration or relative
        velocity ever accumulates between identical writes for the solver
        to turn into an impulse), while a forced robot-dummy overlap under
        the same write scheme reported ~20 kN.

        The wrench goes into the *permanent* composer, not the instantaneous
        one: this method runs once per env step (called from CommandManager via
        ManagerBasedRLEnv.step()'s `self.command_manager.compute(...)` line,
        which runs *after* the physics decimation loop, not inside it — see
        manager_based_rl_env.py). PhysX's own `handles_decimation()` returns
        `False` (isaaclab_physx never overrides the base class default), so
        `ManagerBasedRLEnv.step()` calls `Articulation.write_data_to_sim()`
        once per physics *substep* (4 times at this env's decimation=4).
        `write_data_to_sim()` unconditionally resets the instantaneous
        wrench composer at the end of every call
        (isaaclab/assets/articulation/base_articulation.py). So a force set
        into the *instantaneous* composer here survives exactly one of every
        four substeps — a 25% duty cycle, not a continuous hold. Measured
        with the force written there: mass=87.604 kg, k=13833.8 N/m,
        steady-state err_z=+0.0441 m, k*err_z=+610.2 N, but the settled
        ground reaction was ~859-868 N ≈ m*g — the signature of a body
        resting on the ground under gravity and being kicked once a step,
        not of a continuously-applied spring holding it at m*g via a
        feedforward that (if it worked) would have left ~0 N on the ground.
        The *permanent* wrench composer is not auto-reset by
        write_data_to_sim() (only cleared by an explicit set/reset call, or
        by Articulation.reset() on an env reset), so a value written once
        per env step here is what every one of that step's substeps
        actually applies.

        There is no gravity feedforward and no Z (vertical) term -- unless the
        servo carries the dummy (dummy_hold_clearance_m) -- and the output is
        saturated:

        - A feedforward is conceptually wrong for what this scene needs,
          independent of the substep question: if it worked perfectly, a
          dummy standing at rest would push ~0 N into the ground (the spring
          floats it), which directly contradicts the static-contact
          evidence the scene has to give — a standing human's weight *should*
          read ~m*g on the ground, the same way the robot's own feet do
          without any height-holding controller of their own. The ground
          plane, ordinary rigid-body contact and gravity already do this
          correctly with no PD help, exactly as a freely-settled dummy does
          (check_contact_channel.py's variant A).
        - So there is no Z spring term at all — the linear term only
          drives X/Y (tracking the scripted horizontal crossing). Height is
          whatever gravity and ground contact settle it to, starting from
          `stand_z` (mdp/events.py, captured once at first reset).
        - Angular control is still needed (an uncorrected dummy topples — a
          rigid upright stick figure is not a passively stable equilibrium)
          and is the *only* thing standing between the dummy and falling
          over, so its correctness matters. Its gains take a moment of
          inertia per axis, not one scalar (I = m*L^2/12, a rod about its
          centre, L = height_span) for all three rotational axes. The
          scalar is a reasonable order of magnitude for pitch/roll (tipping
          forward/sideways swings mass distributed along the *whole height*)
          but badly wrong for yaw (spinning about the body's own vertical
          axis swings mass distributed only across its *width* — shoulders,
          arms — a much smaller radius). Measured consequence: with the
          wrench applied on every substep (so the angular gains, sized for
          the overestimated yaw inertia, were being applied at their real
          intended strength), a moving-target test blew up in under 10 steps
          — root_ang_vel_w's z-component alone went from -0.66 to 2.20 to
          -10.48 to 47.20 rad/s step over step (dense per-step logging,
          reproduced twice), an unmistakable diverging discrete-time
          oscillation, not a slow topple. A gain sized for an inertia ~15x
          too large for the axis it is actually driving is a gain applied to
          the wrong plant — exactly what produces that signature.
          So the inertia is measured per axis instead of assumed as one
          shared scalar: each of the 17 regions treated as a point mass at
          its own live-measured body_pos_w, offset from the root, summed
          with the standard parallel-axis form (I_axis = sum(m_i *
          (offset component 1)^2 + (offset component 2)^2)). This is an
          approximation (it drops each region's own internal inertia,
          negligible next to the parallel-axis term for bodies this small
          relative to the whole assembly's span) but a measured one, not a
          single borrowed number applied to three different physical
          quantities.
        - The linear force's application point matters for the same reason,
          separately from inertia: BR-04 (the root LINK) is one rigid body
          among 17 in PhysX's own model (0-DOF fixed joints constrain them
          to move together; they are not pre-merged into a single body
          PhysX reasons about as one mass), so a force applied "at the
          root" acts at BR-04's own centre of mass, not the 17-region
          assembly's. It is applied at the assembly's actual mass-weighted
          CoM (measured, cached as an offset from the root — see the cache
          block below) instead. Applied at ground level (world z=0) it
          made the topple worse, not better — pushing near the feet
          while the mass sits well above them is its own lever arm in the
          opposite rotational sense. A force only avoids inducing rotation
          when applied at the true CoM, by construction — nowhere else is
          special.
        - Saturation: both the linear force and the torque are clamped to a
          bounded magnitude before being written. cfg.dummy_hold_force_cap_
          weight_multiple (default 10x the dummy's own measured weight) sets
          the force's scale — chosen so it never limits an in-distribution
          correction (normal operation needs a small fraction of that, see
          check_dummy_hold.py) while staying an order of magnitude below the
          ~20-50 kN measured from unbounded position-error blowups (a
          forced-teleport test, and the forced-overlap measurement of
          check_forced_overlap.py). The torque is capped per axis, at what
          each axis's own stiffness gives for a set angle — see the
          torque_cap_axes derivation in the body.

        Per-axis inertia settles the *magnitude* mismatch (yaw's inertia) but
        not a second, independent one: the *model*. The per-axis inertia
        above is measured about the root/CoM, i.e. as if the dummy were a
        free body tumbling in space. It is not — while
        its feet are on the ground it tips about them, like an inverted
        pendulum, not about its own centre of mass. Two consequences, both
        specific to the tipping axes (x = roll, y = pitch) and absent on yaw
        (z, which does not pivot on the ground contact the same way):

        - The effective inertia the tipping *dynamics* actually feel is the
          CoM inertia shifted to the ground contact point by the parallel-
          axis theorem: I_eff = I_com + m*h^2, h = how high the CoM sits
          above the ground (measured, cached as com_height in the block
          below). h dominates for a tall, narrow mass distribution like this
          one — measured I_eff ≈ 108 kg*m^2 for both x and y, an order of
          magnitude above the ~13.7 kg*m^2 a root-centred point-mass sum
          alone would give. Yaw needs no such shift: parallel-axis terms for
          a vertical offset vanish for rotation about that same (vertical)
          axis, so the CoM-centred value is already the ground-contact
          value — this is also why a root-centred estimate
          happens to be approximately right for yaw specifically (root and
          CoM differ mostly in z, so the root-centred and CoM-centred x/y
          offsets that feed into I_zz are nearly the same).
        - Gravity contributes a *negative* stiffness to the tipping axes —
          the standard inverted-pendulum result: a body tipped by a small
          angle feels a gravitational torque of +m*g*h per radian that
          grows the tilt, not one that restores it. A spring sized only for
          I_eff*omega^2 therefore nets out (k_spring - m*g*h) once gravity
          is added into the real equation of motion, short of the intended
          stiffness by exactly that amount — measured m*g*h ≈ 898 N*m/rad,
          comparable in size to the spring stiffness itself, so this is not
          a small correction. Added back explicitly (dummy_gravity_
          compensation, zero on yaw) so the *net* stiffness the plant feels
          matches what dummy_hold_natural_freq_hz/damping_ratio actually
          asked for, rather than the spring alone landing well short of it.

        Both terms were checked against an independent analytic recomputation
        from anthropometry.yaml and matched to 4 significant figures, and
        the closed-loop natural frequency/damping ratio this produces for
        pitch was verified by direct substitution to land almost exactly on
        the 2 Hz / critically-damped design target — see check_dummy_hold.py
        for the standing tilt angle this was checked against on the actual
        (nonlinear, PhysX) system, not just the linearized model above.
        """
        entry = getattr(self._env, "s1a_entry", None)
        if entry is None:
            return
        if bool(self.hold_released.all()):
            # Free body from here on -- see _update_hold_release(). Returning
            # leaves the composer at the zeros that method wrote; writing a
            # fresh servo value would be exactly the contamination this
            # exists to remove.
            return
        dummy = self._env.scene["dummy"]
        robot = self._env.scene["robot"]

        if self._dummy_root_body_id is None:
            self._dummy_root_body_id = self._find_root_body_id(dummy)
        if self._dummy_total_mass is None:
            # body_mass and body_pos_w are documented @const/read at a point
            # before any drift is possible (base_articulation_data.py; the
            # 17 regions are rigidly fixed-jointed) — fixed after
            # spawn, so reading once and caching is correct, not a snapshot
            # that could go stale.
            body_mass = dummy.data.body_mass.torch  # (num_envs, num_bodies)
            self._dummy_total_mass = body_mass.sum(dim=1)  # (num_envs,)
            body_pos = dummy.data.body_pos_w.torch  # (num_envs, num_bodies, 3)
            root_pos = dummy.data.root_pos_w.torch.unsqueeze(1)  # (num_envs, 1, 3)
            offset = body_pos - root_pos  # (num_envs, num_bodies, 3)
            # Standing-height proxy — not consumed by either saturation cap
            # (the torque cap comes from each axis's own ang_stiffness, not a
            # lever-arm-style estimate, see the torque_cap_axes derivation
            # below); kept for check_dummy_hold.py's own span-collapse
            # diagnostic, which is independent of both caps.
            span = offset[..., 2].amax(dim=-1) - offset[..., 2].amin(dim=-1)
            self._dummy_height_span = span
            # Aggregate centre of mass, offset from the root link. BR-04
            # (the root LINK) is one rigid body among 17 in PhysX's own model
            # (0-DOF fixed joints constrain them to move together; they are
            # not literally one body PhysX treats as pre-merged), so a force
            # applied at BR-04's own position acts at *its* CoM, not
            # necessarily the assembly's — the two coincide only if BR-04
            # happens to sit exactly at the whole figure's mass centre, which
            # there is no reason to expect (see _advance_dummy's docstring
            # for the toppling this causes when the linear force's
            # application point misses it).
            #
            # Captured once (the dummy is rigid, so this offset never changes
            # in the root's own body frame) at this first call, when
            # reset_dummy_entry has just teleported the root to its facing
            # (entry["facing_quat"]). At S1a's facing, the identity, a world-
            # frame reading *at this specific instant* is numerically the
            # same vector as the body-frame one; any other facing (the 180
            # deg S1d stood at from 2026-09-29, the turn to its own walk it
            # has stood at since 2026-10-01) is turned back into the body frame below —
            # read as-is it would mirror the CoM in x and y, the parasitic
            # torque described next, and at every step, not just once. It is
            # stored as body-frame and rotated out to world frame every step
            # (below, where it is applied): used as if it stayed valid in
            # world frame while the dummy's actual orientation drifts away
            # from identity, the cached vector silently misses the true CoM
            # by however far yaw has moved by that step — a parasitic-torque
            # source sized at up to ~160 N*m against a yaw spring of only
            # ~100 N*m/rad, i.e. large enough by itself to help explain why
            # yaw specifically (of all three axes) drifted furthest.
            com_offset_world = (
                (body_mass.unsqueeze(-1) * offset).sum(dim=1) / self._dummy_total_mass.unsqueeze(-1)
            )  # (num_envs, 3)
            face_travel = bool(entry.get("face_travel", False))
            if face_travel:
                # S1d since 2026-10-01: the facing follows each episode's entry angle, so this call can
                # meet any yaw. Turned back through the pose the dummy has right now -- the same
                # instant offset was read at -- rather than the pose it was parked in a step ago.
                capture_quat = dummy.data.root_quat_w.torch.clone()
                self._dummy_com_offset_body = math_utils.quat_apply_inverse(capture_quat, com_offset_world)
            elif bool((entry["facing_yaw_deg"] != 0.0).any()):
                self._dummy_com_offset_body = math_utils.quat_apply_inverse(entry["facing_quat"], com_offset_world)
            else:
                # S1a: untouched arithmetic, so its trajectories stay the v0.1 batch's to the bit
                self._dummy_com_offset_body = com_offset_world
            gravity_z = float(self._env.sim.cfg.gravity[2])
            self._dummy_weight = self._dummy_total_mass * abs(gravity_z)  # (num_envs,) N — cap derivation only, no feedforward
            # Finite-difference velocity seed (see the class-level field
            # docstrings) — zero velocity for this very first
            # call is correct, the dummy has not moved yet.
            self._dummy_prev_pos_xy = dummy.data.root_pos_w.torch[:, :2].clone()
            self._dummy_prev_quat = dummy.data.root_quat_w.torch.clone()

            # Per-axis moment of inertia and gravity's destabilizing
            # contribution — see _advance_dummy's docstring for the
            # derivation, verified empirically. Two different rotational
            # models for two different groups of axes, not one shared scalar
            # and not one shared *model* either:
            #
            # - Roll/pitch (tipping, x/y): while the dummy's feet are on the
            #   ground, it tips about its *feet*, not about its own CoM — an
            #   inverted pendulum, not a free body. Parallel-axis theorem
            #   from the CoM to a point directly below it at ground level
            #   (offset (0, 0, -com_height)) gives I_eff = I_com + m*h^2,
            #   using each region's offset from the CoM (not the root — see
            #   com_centred_offset below).
            # - Yaw (spin about the vertical axis through the CoM, z): the
            #   same vertical shift used above leaves I_zz unchanged (the
            #   parallel-axis correction term for a shift along an axis is
            #   zero for rotation about that same axis), so the CoM-centred
            #   value is already correct — no ground-pivot adjustment needed.
            #   This also means yaw is the one axis a root-centred estimate
            #   is approximately right about (root and CoM differ mostly in
            #   z, per the com_offset comment above).
            # World frame, like offset: the per-axis inertia below is taken about the world axes at
            # the capture pose. A turn about z by 180 deg only flips x and y, so every squared term
            # -- and so every gain -- is the same as at the identity. Any other turn mixes the two
            # tipping axes: with face_travel the value is recomputed per episode (after this block).
            com_centred_offset = offset - com_offset_world.unsqueeze(1)  # (num_envs, num_bodies, 3)
            cx, cy, cz = com_centred_offset[..., 0], com_centred_offset[..., 1], com_centred_offset[..., 2]
            i_com_xx = (body_mass * (cy**2 + cz**2)).sum(dim=1)
            i_com_yy = (body_mass * (cx**2 + cz**2)).sum(dim=1)
            i_com_zz = (body_mass * (cx**2 + cy**2)).sum(dim=1)
            # Height of the CoM above the ground plane (world z=0 — this
            # env's terrain is flat). root_pos here is still the
            # very first (pre-drift) reading, same standing pose the other
            # cached quantities above use.
            com_height = root_pos[..., 0, 2] + com_offset_world[..., 2]  # (num_envs,)
            parallel_axis_term = self._dummy_total_mass * com_height**2  # m*h^2, (num_envs,)
            carried = self.cfg.dummy_hold_clearance_m is not None
            if carried:
                # Carried by the servo (S1d, 2026-10-02) the dummy does not tip about its feet: the inertia
                # the tipping axes feel is the one about its own centre of mass, and gravity has no lever to
                # grow a tilt with. Left at the ground-pivot value the damping gain 2*zeta*I_eff*omega is
                # several times what a 50 Hz update can apply to the smaller inertia without diverging.
                parallel_axis_term = torch.zeros_like(parallel_axis_term)
            axes_mass, axes_offset = body_mass, com_centred_offset
            upper_lever = None
            if self.cfg.dummy_hold_inertia_model == "upper_body_hips":
                # The jointed dummy stands on its legs and bends at the hips: a wrench on the root turns the body
                # above the hips about the hip joints, not the whole figure about its feet.
                upper = torch.tensor([n in self.cfg.dummy_upper_body_prims for n in dummy.body_names],
                                     dtype=body_mass.dtype, device=self.device)
                axes_mass = body_mass * upper
                m_up = axes_mass.sum(dim=1)
                com_up = (axes_mass.unsqueeze(-1) * offset).sum(dim=1) / m_up.unsqueeze(-1)
                axes_offset = offset - com_up.unsqueeze(1)
                cx, cy, cz = axes_offset[..., 0], axes_offset[..., 1], axes_offset[..., 2]
                i_com_xx = (axes_mass * (cy ** 2 + cz ** 2)).sum(dim=1)
                i_com_yy = (axes_mass * (cx ** 2 + cz ** 2)).sum(dim=1)
                i_com_zz = (axes_mass * (cx ** 2 + cy ** 2)).sum(dim=1)
                upper_lever = com_up[..., 2] - self.cfg.dummy_hip_above_root_m    # upper CoM above the hip pivots
                parallel_axis_term = m_up * upper_lever ** 2
            i_eff_xx = i_com_xx + parallel_axis_term
            i_eff_yy = i_com_yy + parallel_axis_term
            i_eff_zz = i_com_zz  # unchanged — see docstring above
            self._dummy_moment_of_inertia = torch.stack([i_eff_xx, i_eff_yy, i_eff_zz], dim=-1)  # (num_envs, 3)
            if face_travel:
                # The three lines above are the inertia about the world axes at this call's pose: the
                # facing of whichever episode the run started with. Kept in the body frame instead and
                # taken at each episode's own facing below, every call -- no later episode's gains
                # then carry the first one's turn.
                self._dummy_body_mass = axes_mass
                self._dummy_region_offsets_body = math_utils.quat_apply_inverse(
                    capture_quat.unsqueeze(1).expand(-1, offset.shape[1], -1), axes_offset
                )  # (num_envs, num_bodies, 3), from the centre of mass (the upper body's, with its model)
                self._dummy_parallel_axis_term = parallel_axis_term
            # Gravity acts as a negative stiffness on the tipping axes only
            # (standard inverted-pendulum result: a torque of m*g*h per
            # radian of tilt that *grows* the tilt, not restores it) — zero
            # for yaw, which does not move the CoM vertically. Added to
            # ang_stiffness below so the gains solve for the *net* (spring
            # minus gravity) stiffness actually wanted, rather than the
            # spring alone landing short of it.
            gravity_destabilizing = self._dummy_weight * com_height  # m*g*h, (num_envs,)
            if upper_lever is not None:
                gravity_destabilizing = (axes_mass.sum(dim=1)) * abs(gravity_z) * upper_lever
            if carried:
                gravity_destabilizing = torch.zeros_like(gravity_destabilizing)
            self._dummy_gravity_compensation = torch.stack(
                [gravity_destabilizing, gravity_destabilizing, torch.zeros_like(gravity_destabilizing)], dim=-1
            )  # (num_envs, 3)

        if entry.get("face_travel", False):
            # Per-axis inertia at this episode's facing (entry_geometry.yawed_axis_inertia). The target
            # pose is a turn about +z alone, (0, 0, z, w): cos(yaw) = w^2 - z^2, sin(yaw) = 2wz. The
            # ground-pivot term m*h^2 on the tipping axes and its absence on yaw are as in the cache block.
            qz, qw = entry["facing_quat"][:, 2], entry["facing_quat"][:, 3]
            i_xx, i_yy, i_zz = yawed_axis_inertia(
                self._dummy_body_mass, self._dummy_region_offsets_body, qw * qw - qz * qz, 2.0 * qw * qz
            )
            pivot = self._dummy_parallel_axis_term
            self._dummy_moment_of_inertia = torch.stack([i_xx + pivot, i_yy + pivot, i_zz], dim=-1)

        moving = self._trigger_latched(entry, robot)

        if self.cfg.dummy_gait:
            step = self._gait_step_length(entry, moving)
        else:
            step = entry["speed"] * self._env.step_dt
        zero = torch.zeros_like(step)
        # angle is measured from the robot's +x heading. At 90 deg (pure
        # lateral) cos(angle)=0, so x never drifts and the crossing is a
        # straight side-to-side line. Away from 90 deg, x drifts toward -x for
        # angle<90 and toward +x for angle>90 — a mirror pair of oblique
        # crossings, not the same trajectory twice. sin(angle) is positive
        # throughout [30, 150] deg, so y always decreases: the crossing always
        # runs from the +y waiting side toward and past y=0, regardless of
        # angle. Integrated into the scripted target (mdp/events.py module
        # docstring, fourth bullet) — never read back from the live sim pose.
        # Cached (not recomputed from angle/moving a second time below) so
        # the velocity-feedforward term is the exact per-step
        # decrement target_xy itself just took, not an independently
        # re-derived approximation of it.
        target_dxy = torch.stack(
            [
                torch.where(moving, step * torch.cos(entry["angle"]), zero),
                torch.where(moving, step * torch.sin(entry["angle"]), zero),
            ],
            dim=-1,
        )  # (num_envs, 2)
        stop_y = entry.get("stop_y")
        if stop_y is not None:
            # S1d (2026-10-01): the walk stops at the far side's mirrored waiting offset instead of
            # carrying the held dummy into the far rack (events.reset_dummy_entry, stop_at_far_side).
            # A fraction of 1.0 leaves the step bit-identical; from the line on, target and target
            # velocity are 0, so the servo holds the dummy standing there.
            frac = lateral_stop_fraction(entry["target_xy"][:, 1], stop_y, step * torch.sin(entry["angle"]))
            target_dxy = target_dxy * frac.unsqueeze(-1)
            # The update whose fraction falls below 1 commands the walk's last move: the recorder ends the
            # realized entry speed's window there (entry_geometry.EntrySpeedWindow), not at the latched
            # trigger's end.
            latch_walk_end(entry, moving, frac)
        entry["target_xy"][:, 0] -= target_dxy[:, 0]
        entry["target_xy"][:, 1] -= target_dxy[:, 1]
        if self.cfg.dummy_gait:
            self._drive_gait(dummy, target_dxy)

        # Upright at the facing reset_dummy_entry parked the dummy in -- the identity (no yaw) in
        # S1a, turned to its own walk in S1d -- the same target every step of an episode.
        target_quat = entry["facing_quat"]

        omega = 2.0 * math.pi * self.cfg.dummy_hold_natural_freq_hz
        zeta = self.cfg.dummy_hold_damping_ratio
        lin_stiffness = self._dummy_total_mass * omega * omega              # N/m
        lin_damping = 2.0 * zeta * self._dummy_total_mass * omega           # N*s/m
        # Per-axis (num_envs, 3) — see the cache block above and this
        # method's docstring for why a single shared scalar
        # (assuming yaw and pitch/roll swing the same effective radius) is
        # wrong, not just imprecise, and why the tipping axes additionally
        # need a gravity-compensation term. I*omega^2 sizes the spring for
        # the *net* stiffness this system's inertia and desired natural
        # frequency call for; without adding back what gravity subtracts
        # (dummy_gravity_compensation, zero on yaw), the net stiffness the
        # plant actually feels would land short of that by exactly gravity's
        # destabilizing contribution.
        ang_stiffness = (
            self._dummy_moment_of_inertia * omega * omega + self._tilt_gravity_compensation()
        )  # N*m/rad, per axis
        ang_damping = 2.0 * zeta * self._dummy_moment_of_inertia * omega            # N*m*s/rad, per axis

        # Velocity for both damping terms comes from differencing the pose
        # this method itself already reads, not from PhysX's own root_lin_
        # vel_w / root_ang_vel_w. Those were measured (cross-checked against
        # this env's commanded/FD-reconstructed motion) running 2-4x higher
        # than the dummy's actual speed — self-
        # consistent with PhysX's own logged "TGS solver with enable_external_
        # forces_every_iteration=False may cause noisy velocities" warning for
        # this solver configuration — which fed straight into the damping
        # term above and produced multi-kN holds where the stiffness term
        # alone needed only a few hundred N of correction. current_xy/
        # current_quat are read once here and reused below both to form this
        # step's velocity estimate and to seed next step's; _dummy_prev_pos_
        # xy/_dummy_prev_quat start at zero-velocity on the very first call
        # (cache block above) and get reseeded to the fresh teleported pose
        # on every reset (_resample_command), so this never differences
        # across an unrelated episode boundary.
        current_xy = dummy.data.root_pos_w.torch[:, :2]
        current_quat = dummy.data.root_quat_w.torch
        dt = self._env.step_dt
        xy_vel = (current_xy - self._dummy_prev_pos_xy) / dt
        # quat_box_minus(q1, q2) = log(q1 * q2^-1) (isaaclab.utils.math), the
        # same world-frame axis-angle convention ang_error below uses (q1
        # minus q2 read as "the world-frame rotation that carries q2 to
        # q1") — so box-minus(current, previous)/dt is that same convention's
        # angular-velocity analogue of ang_error's angular-position one, not
        # a second, independently-chosen formula.
        ang_vel = math_utils.quat_box_minus(current_quat, self._dummy_prev_quat) / dt
        self._dummy_prev_pos_xy = current_xy.clone()
        self._dummy_prev_quat = current_quat.clone()

        # Velocity feedforward: target_xy is a ramp while moving (constant
        # target_dxy/dt), not a fixed setpoint, and a PD that damps against
        # the dummy's own raw velocity instead of its velocity *relative to
        # the target* carries a structural ramp-tracking lag of
        # 2*zeta*v/omega (at walk_speed's 1.6 m/s ceiling,
        # 2*1.0*1.6/12.566 =~ 0.255 m, matching the ~0.245-0.304 m
        # measured directly). At that
        # lag the spring and (raw-velocity) damping terms both sit around
        # +-3.6 kN and very nearly cancel — the force actually delivered
        # each step is the *difference* of two multi-kN terms, swinging
        # 131-7723 N step to step on nothing more than that near-
        # cancellation's own numerical slop, not a deliberate control
        # decision. Subtracting the target's own instantaneous velocity
        # from what the damping term reacts to removes the lag at its
        # source instead of living with it: in steady ramp-tracking (dummy
        # moving at target_vel_xy, position error constant) the damping
        # term now sees zero relative velocity, so it does not have to
        # cancel a spring term that was never supposed to be large in the
        # first place — target_vel_xy is exactly target_dxy/dt, the same
        # per-step decrement target_xy itself just took above, not a
        # second, independently-derived estimate of it.
        target_vel_xy = -target_dxy / dt
        # X/Y only — no Z term, see docstring. lin_force's z-component stays
        # exactly 0; gravity and ordinary ground contact carry the weight -- unless the servo carries
        # the dummy (dummy_hold_clearance_m, below).
        xy_error = entry["target_xy"] - current_xy
        plane_stiffness, plane_damping = self._plane_gains(lin_stiffness, lin_damping)
        lin_force_xy = plane_stiffness.unsqueeze(-1) * xy_error - plane_damping.unsqueeze(-1) * (xy_vel - target_vel_xy)
        lin_force = torch.cat([lin_force_xy, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
        if self.cfg.dummy_hold_clearance_m is not None:
            # The carried hold: the dummy's weight, fed forward, and the same spring and damper as in the
            # plane, toward the height it stands at plus the clearance. The velocity is differenced from
            # the pose, like xy_vel. Released, the whole wrench is zeroed below and the dummy comes down
            # the clearance onto its feet.
            z = dummy.data.root_pos_w.torch[:, 2]
            if self._dummy_prev_z is None:
                self._dummy_prev_z = z.clone()
            z_vel = (z - self._dummy_prev_z) / dt
            self._dummy_prev_z = z.clone()
            z_target = (self._env.scene.env_origins[:, 2] + self.cfg.dummy_hold_stand_height_m
                        + self.cfg.dummy_hold_clearance_m)
            lin_force[:, 2] = self._dummy_weight + lin_stiffness * (z_target - z) - lin_damping * z_vel
        elif self.cfg.dummy_hold_support_share is not None:
            # The jointed dummy's partial weight support: a share of the weight, and the same spring and
            # damper as in the plane toward the height the gait's joint targets put the root at (_support_force_z).
            lin_force[:, 2] = self._support_force_z(dummy.data.root_pos_w.torch[:, 2], lin_stiffness, lin_damping)

        ang_error = math_utils.quat_box_minus(target_quat, current_quat)
        # Element-wise, not unsqueeze-broadcast: ang_stiffness/ang_damping
        # are already (num_envs, 3) — one gain per rotational axis, not one
        # scalar per env applied uniformly to all three.
        ang_torque = ang_stiffness * ang_error - ang_damping * ang_vel
        turning = self._facing_turn_torque(ang_damping)
        if turning is not None:
            # A facing that turns within the episode (the conditions-table term's turn on the spot): the damper reads
            # the dummy's rate against the target's own, as the plane's damper reads target_vel_xy.
            ang_torque = ang_torque + turning

        cap_mult = self.cfg.dummy_hold_force_cap_weight_multiple
        force_cap = cap_mult * self._dummy_weight                                    # (num_envs,) N
        lin_force = _clamp_vector_norm(lin_force, force_cap)
        # Exposed for check_dummy_hold.py's moving-phase hold-force
        # reporting (the m*g gate only covers the static phase, so without
        # this nothing watches the moving phase's hold force — measured
        # averaging 2069-3800N and peaking past 15kN, invisible until asked
        # for directly).
        # Post-cap — what is actually sent to the wrench composer below, not
        # the pre-cap spring-minus-damping value.
        self.dummy_lin_force_applied = lin_force
        # Per-axis, not a shared-norm clamp. One scalar cap (derived from
        # the *linear*-force lever-arm reasoning, not from anything angular)
        # bounding the combined 3-axis torque vector's norm does not work:
        # the per-axis ang_stiffness leaves pitch/roll's gain ~4.5x larger
        # than yaw's (~17965 vs ~3998 N*m/rad when this was measured), so
        # ordinary pitch/roll tracking alone (no yaw error at all) saturates
        # the shared norm — measured at 87% of tracking steps — and once
        # saturated, the norm clamp scales the *whole* vector down by one
        # factor, crushing whatever (much smaller) yaw torque happens to be
        # riding alongside it. A per-axis bound sized to each axis's own
        # ang_stiffness cannot cross-suppress a different axis this way by
        # construction.
        #
        # Deliberately *not* scaled by cap_mult (unlike force_cap): dense
        # per-step logging during forced tracking (seed=0) showed that
        # the discrete-time closed loop is only marginally stable under a
        # sustained large tracking error — torque grows roughly 3-5x per
        # *step* once a disturbance pushes ang_error past a few tens of
        # degrees, on both this FD velocity and (measured no better, often
        # worse) PhysX's own root_ang_vel_w, and counter-intuitively raising
        # dummy_hold_damping_ratio made the runaway *faster*, not slower —
        # consistent with a discrete-derivative-gain/delay margin problem,
        # not a plain under-damping one. A cap generous enough to "never
        # trigger during normal tracking" (a cap_mult=10-scaled version
        # reaches ~282000 N*m for pitch/roll)
        # provides no protection against exactly this runaway, since it
        # never engages until the torque is already several orders of
        # magnitude past where the divergence became unrecoverable. Sized
        # instead to sit just above the measured normal operating range
        # (a few hundred to ~2000 N*m for pitch/roll during
        # ordinary — even fairly disturbed — tracking) so it engages while a
        # runaway is still small enough to matter, without the generic
        # angle-based sizing needing this axis's specific N*m number typed
        # in. See cfg field docstring for the exact angle and how narrow
        # that margin is against normal operation. The discrete-stability
        # question itself stays open: this does not, by itself, resolve
        # it.
        #
        # x/y (tipping) and z (yaw) deliberately use *different* angles, not
        # the same one broadcast across all three axes: a tight cap that
        # correctly suppresses pitch/roll's runaway measured, at the same
        # angle, caps yaw's own restoring torque below what yaw was
        # measured to need to counteract a real (not runaway-instability)
        # disturbance it is separately subject to — tightening the angle
        # uniformly fixed pitch/roll (tilt: 5.9->5.3deg) but made yaw *worse*
        # (measured 140deg, worse than the original ~40deg with an
        # effectively uncapped torque), because the cap was starving yaw's
        # own correction, not protecting it. yaw's cap stays generous in
        # angle terms (its own ang_stiffness is ~178x softer than pitch/
        # roll's, so even a large angle allowance stays a small absolute
        # N*m bound — 100.1 N*m/rad * radians(90) =~ 157 N*m, under 3.5% of
        # pitch/roll's own tightened ~4687 N*m cap, nowhere near
        # reintroducing the runaway magnitude that motivated tightening
        # pitch/roll in the first place).
        torque_cap_angle_deg = torch.tensor(
            [
                self.cfg.dummy_hold_torque_cap_angle_deg,
                self.cfg.dummy_hold_torque_cap_angle_deg,
                self.cfg.dummy_hold_torque_cap_angle_deg_yaw,
            ],
            device=self.device,
        )
        torque_cap_axes = ang_stiffness * torch.deg2rad(torque_cap_angle_deg)
        # Exposed for check_dummy_hold.py's saturation-rate reporting —
        # per-axis, per-env boolean, true wherever the clamp below actually
        # changes the value (not just "torque happens to be near the cap").
        self.dummy_torque_saturated_per_axis = ang_torque.abs() >= torque_cap_axes
        # Kept alongside dummy_lin_force_applied: the hold is a *wrench*, and
        # an audit that watches only the force can call a step "ran free"
        # while the attitude torque was still acting on the dummy.
        ang_torque = torch.clamp(ang_torque, min=-torque_cap_axes, max=torque_cap_axes)
        # Recorded *after* the clamp, because the field is named "applied" and
        # the audit exists to say what the dummy actually felt. Stored before it,
        # a saturating step reported a torque larger than the one the physics
        # composer received.
        self.dummy_torque_applied = ang_torque

        # Envs whose hold has been released get nothing, regardless of what
        # the servo computed. The early return in this method's preamble only
        # fires when *every* env is released; this is what keeps a released
        # env free while its neighbours are still being held.
        held = (~self.hold_released).unsqueeze(-1).float()
        lin_force = lin_force * held
        ang_torque = ang_torque * held
        # "Applied" is what the composer receives: a released env's row is zero from here on. Stored
        # before this mask, the buffers kept the servo's unapplied output for an env released while a
        # neighbour was still held -- with the carried hold a constant weight's worth, which the
        # recorder would have read as a hold still on. A held env's row is
        # multiplied by 1.0: the same bits.
        self.dummy_lin_force_applied = lin_force
        self.dummy_torque_applied = ang_torque

        # Applied at the assembly's aggregate centre of mass (root position
        # + the cached offset), not at the root link's own position.
        # Applied at the root (BR-04's own CoM, not the 17-region
        # assembly's) it toppled the dummy — a lin_force of a few kN,
        # entirely ordinary with the force applied at full strength on
        # every substep, is enough leverage at whatever
        # offset separates BR-04's own CoM from the assembly's to overwhelm
        # the angular spring (17-region z-spread collapsed from 1370mm
        # toward 0 within tens of steps). Moving the application point to
        # world z=0 (ground level) made it *worse*, not better — pushing at
        # the feet while the mass is concentrated well above them is its
        # own, larger lever arm in the other rotational sense. Force only
        # avoids inducing rotation at all when applied at the true centre of
        # mass, by definition — not at the root, not at the ground, at the
        # CoM specifically. Only the linear force needs this — the torque is
        # a pure couple with no position dependence, so it is unaffected
        # either way.
        #
        # _dummy_com_offset_body is cached once, in the root's own body
        # frame (see the cache block's docstring above for why that
        # capture instant is exact, not approximate). Added straight onto
        # current world-frame root_pos_w every step it is only correct
        # while the root's orientation stays at the identity it had when the
        # vector was captured — everywhere else, it points the application
        # point in a direction fixed in *world* space while the dummy itself
        # has rotated out from under it, missing the true (body-fixed) CoM
        # by however far orientation has drifted. Rotating the body-frame
        # offset by the *current* orientation every step keeps the
        # application point pinned to the true CoM regardless of how the
        # dummy is currently oriented.
        application_point = dummy.data.root_pos_w.torch + math_utils.quat_apply(
            current_quat, self._dummy_com_offset_body
        )

        # permanent, not instantaneous: instantaneous is cleared inside
        # write_data_to_sim() every call, and PhysX's handles_decimation()
        # is False, so that function runs once per physics *substep* — 4
        # times per env step at this env's decimation — while this method
        # runs once per env step. permanent_wrench_composer.set_forces_and_
        # torques_index() replaces (not accumulates onto) the previous
        # value and, unlike instantaneous, is not reset by write_data_to_sim
        # itself, so the same force applies across every substep of the
        # step it was set for, until this call replaces it again next step.
        # See the docstring above for the measurement behind this.
        dummy.permanent_wrench_composer.set_forces_and_torques_index(
            forces=lin_force.unsqueeze(1),
            torques=ang_torque.unsqueeze(1),
            positions=application_point.unsqueeze(1),
            body_ids=[self._dummy_root_body_id],
            is_global=True,
        )

    def _facing_turn_torque(self, ang_damping: torch.Tensor) -> torch.Tensor | None:
        """The torque (num_envs, 3) the hold adds to its spring and damper while the facing it aims at turns, or None
        while it does not. Here the facing is the one reset_dummy_entry parked the dummy in, the same every step of an
        episode: None, and the hold's torque is its spring and damper alone. A subclass whose script turns the facing
        within an episode (the conditions-table term's turn on the spot) answers with the damper times the facing's own
        rate, so the damper reads the dummy's rate against the target's instead of against standing still -- without
        it a facing turning at a steady rate is followed a steady 2 * zeta / omega behind."""
        return None

    def _trigger_latched(self, entry, robot) -> torch.Tensor:
        """Latch the scripted walk's trigger and return which envs walk: the robot within this episode's
        trigger distance of the point where the entry line crosses its lane, along its lane."""
        robot_x = robot.data.root_pos_w.torch[:, 0]
        dist_to_crossing = entry["x_cross"] - robot_x
        # Latch, don't just compare: once the robot is within trigger_distance
        # of the crossing point (and hasn't already passed it — dist_to_crossing
        # >= 0 excludes the "robot walked straight through, never near enough
        # to trigger anything" case from spuriously firing once it's gone
        # negative), the human commits to crossing for the rest of the episode
        # (catalogue §4.4: wait -> trigger -> enter -> pass through or stop —
        # a one-shot transition, not something that toggles back to waiting).
        # Not every episode triggers, and that's a legitimate `nominal` result
        # (catalogue §9-6 boundary allocation is what raises fail-rate, not
        # forcing every draw to interact).
        should_trigger = (dist_to_crossing >= 0.0) & (dist_to_crossing < entry["trigger_distance"])
        entry["triggered"] |= should_trigger
        return entry["triggered"]

    def _stop_distance(self, entry):
        """Metres the scripted walk has left to its stop, per env (numpy), or None for a walk without one: here the
        far-side stop line (``stop_y``) along the entry line."""
        from generation.dummy.gait import gait_table as gt

        if entry.get("stop_y") is None:
            return None
        return gt.remaining_to_stop(entry["target_xy"][:, 1].cpu().numpy(), entry["stop_y"].cpu().numpy(),
                                    entry["angle"].cpu().numpy())

    def _plane_gains(self, stiffness: torch.Tensor, damping: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The hold's spring and damper across the plane: the hold's own (dummy_hold_natural_freq_hz) unless the plane
        has a frequency of its own (dummy_hold_plane_freq_hz) -- lower, the walking legs move the body more and the hold
        less; the vertical and the tilt keep the hold's frequency."""
        freq = self.cfg.dummy_hold_plane_freq_hz
        if freq is None:
            return stiffness, damping
        omega = 2.0 * math.pi * freq
        return (self._dummy_total_mass * omega * omega,
                2.0 * self.cfg.dummy_hold_damping_ratio * self._dummy_total_mass * omega)

    def _tilt_gravity_compensation(self) -> torch.Tensor:
        """Gravity's tipping term in the tilt stiffness (per axis, 0 on yaw). With a share of the weight carried
        (dummy_hold_support_share), pushed up at the centre of mass, a body tipping about its feet feels only the rest
        of gravity's torque -- (1 - share) of it. About the hips the push has no lever on the upper body's own weight,
        and with no share the cached term is returned as it is."""
        share = self.cfg.dummy_hold_support_share
        if share is None or self.cfg.dummy_hold_inertia_model != "whole_body_feet":
            return self._dummy_gravity_compensation
        return self._dummy_gravity_compensation * (1.0 - share)

    def _support_force_z(self, z: torch.Tensor, stiffness: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
        """The partial weight support's upward force on the jointed dummy (dummy_hold_support_share): the
        share of its weight, fed forward, and a spring and a damper toward the stand height plus the pelvis rise of the
        joint targets the gait sent (_drive_gait). The damper reads the root's speed against the target's own, as the
        plane's damper reads it against the scripted walk: a body that follows the gait's pelvis path feels the share
        alone, and the spring pulls back a body that the share has made light once it leaves the path. Both speeds
        are differenced from the poses, like the plane's; the first call after a reset reads no target speed."""
        dt = self._env.step_dt
        if self._dummy_prev_z is None:
            self._dummy_prev_z = z.clone()
        z_vel = (z - self._dummy_prev_z) / dt
        self._dummy_prev_z = z.clone()
        target = self._env.scene.env_origins[:, 2] + self.cfg.dummy_hold_stand_height_m + self._gait_pelvis_rise
        if self._support_prev_target_z is None:
            self._support_prev_target_z = target.clone()
        target_vel = (target - self._support_prev_target_z) / dt
        target_vel = torch.where(torch.isnan(target_vel), torch.zeros_like(target_vel), target_vel)
        self._support_prev_target_z = target.clone()
        return (self.cfg.dummy_hold_support_share * self._dummy_weight + stiffness * (target - z)
                - damping * (z_vel - target_vel))

    def _gait_step_length(self, entry, moving) -> torch.Tensor:
        """This step's scripted travel with the gait on: the speed ramps up from rest over
        dummy_gait_accel_time_s and comes down to rest at the far-side stop line over dummy_gait_decel_time_s."""
        from generation.dummy.gait import gait_table as gt

        dt = self._env.step_dt
        remaining = self._stop_distance(entry)
        v = gt.ramp_speed(self.gait_speed.cpu().numpy(), entry["speed"].cpu().numpy(), dt,
                          self.cfg.dummy_gait_accel_time_s, self.cfg.dummy_gait_decel_time_s, remaining)
        v = torch.as_tensor(v, dtype=self.gait_speed.dtype, device=self.device)
        held = ~self.hold_released
        self._gait_slowing = moving & held & (v < self.gait_speed)
        self.gait_speed = torch.where(moving & held, v, torch.where(held, torch.zeros_like(v), self.gait_speed))
        return self.gait_speed * dt

    def _ensure_gait(self, dummy) -> None:
        """The gait table out of the spawned USD and its DOF names mapped onto the dummy's joints, once -- on
        whichever comes first, the first gait update or a release (a release that came first would otherwise
        never get its pose frozen at tone: the latch does not come back)."""
        if self._gait is not None:
            return
        from generation.dummy import jointed

        self._gait = jointed.read_gait(dummy.cfg.spawn.usd_path)
        from generation.dummy import joints_spec as js

        self._gait_spec = js.load()
        names = list(dummy.joint_names)
        if set(names) != set(self._gait.dof_names):
            raise ValueError(f"걸음 표의 DOF {sorted(self._gait.dof_names)} 가 인형의 관절 {sorted(names)} 와 다릅니다")
        # int32: the PhysX articulation hands a joint-index tensor to its Warp kernels as it is, and they take int32.
        self._gait_dof_ids = torch.tensor([names.index(n) for n in self._gait.dof_names], dtype=torch.int32,
                                          device=self.device)

    def _gait_drive(self, target_dxy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(speed, slowing, distance) per env the gait is driven with this step: the scripted speed
        (_gait_step_length's ramp), whether it is slowing to its stop, and how far the scripted target moved -- the
        phase advances by that distance over the stride at that speed. A subclass whose script moves the legs
        without moving the target (the conditions-table term's turn on the spot) answers differently here."""
        return (self.gait_speed.cpu().numpy(), self._gait_slowing.cpu().numpy(),
                torch.linalg.norm(target_dxy, dim=-1).cpu().numpy())

    def _drive_gait(self, dummy, target_dxy) -> None:
        """Joint targets from the gait table at the speed and the phase advance _gait_drive gives (the scripted speed
        and the scripted target's travel) -- for held envs only; a released dummy keeps the targets it was frozen at."""
        from generation.dummy.gait import gait_table as gt
        from generation.dummy.gait.held import held_update

        self._ensure_gait(dummy)
        held = (~self.hold_released).cpu().numpy()
        v, slowing, distance = self._gait_drive(target_dxy)
        phase = gt.advance_phase(self.gait_phase.cpu().numpy(), distance, self._gait.stride_at(v, slowing))
        self.gait_phase = torch.as_tensor(held_update(self.gait_phase.cpu().numpy(), phase, held),
                                          dtype=self.gait_phase.dtype, device=self.device)
        targets = torch.as_tensor(self._gait.at(v, self.gait_phase.cpu().numpy(), slowing), device=self.device)
        entry = getattr(self._env, "s1a_entry", None)
        ended = None if entry is None else entry.get("walk_ended")
        if ended is not None:
            # A held walk that has come to rest at its stop line stands where its last step left its feet.
            rest = ~self.hold_released & ended & (self.gait_speed == 0)
            if bool(rest.any()):
                targets = self._rest_targets(targets, rest)
        if self.cfg.dummy_hold_support_share is not None:
            from generation.dummy.gait import support

            # The height the partial weight support aims the root at: the pelvis under the targets sent this step.
            rise = support.pelvis_rise(self._gait_spec, self._gait.dof_names, targets.cpu().numpy())
            self._gait_pelvis_rise = torch.as_tensor(rise, dtype=self._gait_pelvis_rise.dtype, device=self.device)
        # The rate the targets move at, for the drives' damping: held at a zero target velocity, a joint lags its
        # moving target by damping / stiffness and the legs are not the table's lengths. None after a reset.
        if self._gait_previous is None:
            self._gait_previous = torch.full_like(targets, float("nan"))
        rate = (targets - self._gait_previous) / self._env.step_dt
        rate = torch.where(torch.isnan(rate), torch.zeros_like(rate), rate)
        ids = torch.nonzero(~self.hold_released).flatten()
        self._gait_previous[ids] = targets[ids]
        if len(ids):
            dummy.set_joint_position_target_index(target=targets[ids][:, :], joint_ids=self._gait_dof_ids, env_ids=ids)
            dummy.set_joint_velocity_target_index(target=rate[ids][:, :], joint_ids=self._gait_dof_ids, env_ids=ids)

    def _rest_targets(self, targets: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
        """`targets` with the envs in `rest` going from their last walking targets to the stance those leave
        (support.rest_stance -- a foot in the air put down where it is) over _GAIT_REST_BLEND_STEPS updates, then
        staying there. The stance is taken once, on the first update at rest."""
        from generation.dummy.gait import support

        if self._gait_rest is None:
            self._gait_rest = torch.full_like(targets, float("nan"))
            self._gait_rest_from = torch.full_like(targets, float("nan"))
        new = rest & torch.isnan(self._gait_rest[:, 0])
        if bool(new.any()):
            ids = torch.nonzero(new).flatten()
            last = targets[ids] if self._gait_previous is None else self._gait_previous[ids]
            last = torch.where(torch.isnan(last), targets[ids], last)
            self._gait_rest_from[ids] = last
            stance = support.rest_stance(self._gait_spec, self._gait.dof_names, last.cpu().numpy())
            self._gait_rest[ids] = torch.as_tensor(stance, dtype=targets.dtype, device=self.device)
            self._gait_rest_count[ids] = 0
        self._gait_rest_count = torch.where(rest, self._gait_rest_count + 1, self._gait_rest_count)
        w = (self._gait_rest_count.to(targets.dtype) / _GAIT_REST_BLEND_STEPS).clamp(max=1.0).unsqueeze(-1)
        blended = self._gait_rest_from + w * (self._gait_rest - self._gait_rest_from)
        return torch.where(rest.unsqueeze(-1), blended, targets)

    def _forget_gait_rest(self, env_ids) -> None:
        """No stance at rest for env_ids (their reset)."""
        if self._gait_rest is not None:
            self._gait_rest[env_ids] = float("nan")
            self._gait_rest_from[env_ids] = float("nan")
        self._gait_rest_count[env_ids] = 0

    def _forget_gait_rate(self, env_ids) -> None:
        """No last gait update for env_ids (their reset): the next one asks no joint velocity."""
        if self._gait_previous is not None:
            self._gait_previous[env_ids] = float("nan")

    def _write_gains(self, env_ids, gains: dict) -> None:
        dummy = self._env.scene["dummy"]
        names = list(self._gait.dof_names)
        k = torch.tensor([[gains[n][0] for n in names]] * len(env_ids), dtype=torch.float32, device=self.device)
        d = torch.tensor([[gains[n][1] for n in names]] * len(env_ids), dtype=torch.float32, device=self.device)
        dummy.write_joint_stiffness_to_sim_index(stiffness=k, joint_ids=self._gait_dof_ids, env_ids=env_ids)
        dummy.write_joint_damping_to_sim_index(damping=d, joint_ids=self._gait_dof_ids, env_ids=env_ids)

    def _heading_correction(self) -> torch.Tensor:
        """Proportional control back toward yaw=0 (the robot's spawn heading —
        S1aEnvCfg pins it there).

        Sending ang_vel_z=0 unconditionally is not a neutral choice — it
        means nothing ever steers the robot back onto a straight line, and
        the trained H1 policy does not hold a perfectly straight heading on
        its own even under a constant zero *command* (probed directly:
        yaw crept from 0 to 15+ deg over ~700 steps, curving the robot's own
        path toward wherever the dummy happened to be waiting). Correcting
        it here is not a workaround for that measurement; it is what
        catalogue §4.3's "직선 경로를 일정 속도로 보행" ("walk a straight path
        at a constant speed") actually requires — an uncorrected command
        is the thing not matching the spec.

        The exact same control law already exists
        upstream (isaaclab.envs.mdp.commands.velocity_command.
        UniformVelocityCommand._update_command, heading_command=True path) —
        the *published* H1 policy trained with rel_heading_envs=1.0 (every
        env heading-controlled) and heading_control_stiffness=0.5
        (velocity_env_cfg.py). ang_vel_z=0 was never a condition this policy
        saw in training; a constant-zero command was the out-of-distribution
        input, not this correction. Matching the exact formula (not just the
        gain) and using heading_w/wrap_to_pi rather than a hand-rolled
        quaternion-to-yaw extraction keeps this one identifiable as "the
        upstream loop, closed again" rather than a new, separately-tuned
        controller — gain and cap default to the upstream values unless a
        measurement shows they need to move (see S1aStraightWalkCommandCfg).

        Applied unconditionally (not gated on the forward-speed scale
        above): a protective stop should hold heading too, not just
        forward motion, and drift was measured continuing
        *after* the stop latched (low-level joint control does not arrest
        gait momentum instantly), which this also damps.
        """
        robot = self._env.scene["robot"]
        heading_error = math_utils.wrap_to_pi(-robot.data.heading_w.torch)
        return torch.clamp(
            self.cfg.heading_correction_gain * heading_error,
            min=self.cfg.heading_correction_ang_vel_z_range[0],
            max=self.cfg.heading_correction_ang_vel_z_range[1],
        )

    def _update_metrics(self):
        # CommandTerm.reset() (base class, not overridden here) does
        # `metric_value[env_ids]` on every entry of self.metrics — it requires a
        # (num_envs,) tensor. Assigning the already-reduced .mean() (0-d) crashes
        # the very first time an episode ends: confirmed by running run_s1a.py,
        # which reset mid-episode and raised "IndexError: too many indices for
        # tensor of dimension 0" from inside CommandTerm.reset(). .clone(), not a
        # bare reference, because that same base-class reset() zeroes
        # metric_value[env_ids] in place — aliasing self.min_distance here would
        # corrupt it as a side effect of logging.
        self.metrics["min_distance_human"] = self.min_distance.clone()
        self.metrics["separation_required"] = self.separation_required.clone()

    def _measure_min_distance(self) -> torch.Tensor:
        robot = self._env.scene["robot"]
        dummy = self._env.scene["dummy"]
        if self.cfg.dummy_region_prims and self._dummy_region_mask is None:
            # The jointed dummy's feet are bodies but not body regions: the distance is to the regions only.
            self._dummy_region_mask = torch.tensor(
                [n in self.cfg.dummy_region_prims for n in dummy.body_names], device=self.device)
        # body_pos_w is a ProxyArray (warp-backed) on this Isaac Lab version, not a
        # plain torch.Tensor — .torch is the explicit, non-deprecated accessor onto
        # its (num_envs, num_bodies, 3) view. Same pattern isaaclab's own
        # UniformVelocityCommand uses for root_lin_vel_b/root_ang_vel_b.
        radii_np = torch.full(
            (dummy.num_bodies,), self.cfg.human_region_radius, device=self.device
        ).cpu().numpy()
        out = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.num_envs):
            human = dummy.data.body_pos_w.torch[i]
            radii = radii_np
            if self._dummy_region_mask is not None:
                human = human[self._dummy_region_mask]
                radii = radii_np[self._dummy_region_mask.cpu().numpy()]
            out[i] = float(
                min_distance_human(
                    robot.data.body_pos_w.torch[i].cpu().numpy(),
                    human.cpu().numpy(),
                    radii,
                )
            )
        return out


@configclass
class S1aStraightWalkCommandCfg(CommandTermCfg):
    class_type: type = S1aStraightWalkCommand

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)   # once per episode
    rules_path: str = MISSING
    walk_speed_range: tuple[float, float] = (0.3, 1.2)            # catalogue §4.5
    sensor_delay_ms_range: tuple[float, float] = (0.0, 100.0)     # catalogue §4.5
    # Both of these shape the slowdown ramp (mdp.safety.command_scale) and are
    # safety-controller tuning, not standard-sourced numbers — see that
    # function's docstring for why stop_band_factor exists and why it must
    # stay strictly less than slowdown_factor.
    slowdown_factor: float = 2.0
    stop_band_factor: float = 1.05
    human_region_radius: float = 0.1
    provisional_ssm_params: dict = MISSING
    # Heading correction (see S1aStraightWalkCommand._heading_correction):
    # not a new controller — the published H1 policy trained under
    # heading_command=True, so these are the *same* values that command
    # already used (isaaclab_tasks velocity_env_cfg.py:
    # heading_control_stiffness=0.5, commands.base_velocity.ranges.ang_vel_z
    # = (-1.0, 1.0)). The rule against hand-copied canonical values
    # cuts the other way here: a value with an upstream source should track
    # it, not invent an unexplained departure. Change these only alongside a
    # measurement showing 0.5 is insufficient here (check_entry.py's yaw
    # tracking + tracking-reward output), and say why the trained policy's
    # own gain isn't enough for S1a specifically.
    heading_correction_gain: float = 0.5
    heading_correction_ang_vel_z_range: tuple[float, float] = (-1.0, 1.0)

    # Dummy standing/tracking PD (see S1aStraightWalkCommand._advance_dummy
    # for why it is a bounded force/torque and not a per-step teleport;
    # why it only drives X/Y + orientation — no Z term, no
    # gravity feedforward — and why it is saturated; and why the
    # angular gains use a ground-pivot (inverted-pendulum) inertia and a
    # gravity-compensation term on the tipping axes instead of one shared
    # free-body inertia estimate for all three).
    # Controller tuning, not a standard-sourced number — same category as
    # slowdown_factor/stop_band_factor above. Stiffness/damping are derived
    # per-env from the dummy's own measured mass and (for the angular gains)
    # measured per-axis moment of inertia at these two knobs — see
    # _advance_dummy's cache block — rather than typed directly as N/m and
    # N*m/rad, so the "how snappy is the hold" question has one physically-
    # interpretable answer instead of several disconnected magic constants.
    # Design target is critically damped (ratio 1.0); the plant
    # correction (ground-pivot inertia + gravity compensation) is what makes
    # the realized pitch/roll response actually land near that target
    # instead of the underdamped ~0.55 a free-body estimate
    # produces — see check_dummy_hold.py for the tilt-angle measurement this
    # is checked against. 2 Hz keeps omega*step_dt (~0.25 at this env's
    # 0.02 s step) well inside the small-argument regime a semi-implicit
    # force integrator needs to stay stable, while settling (~4/omega ≈
    # 0.3 s, ~16 steps) fast relative to the multi-second entry crossing it
    # has to track.
    dummy_hold_natural_freq_hz: float = 2.0
    # S1d (2026-10-02): the hold servo carries the dummy this far above where it stands (metres) for as
    # long as it holds it. On its feet the dummy is pulled at its centre of mass against friction at its
    # feet, and at entry speeds from about 1 m/s that turned it 12-24 deg about the vertical; carried, the
    # same episodes stay within 0.1 deg. None leaves the weight on the feet: S1a.
    dummy_hold_clearance_m: float | None = None
    # The root link's height above the floor with the dummy standing on its feet -- the spec's
    # (generation.envs.s1d.aisle_guard.standing_root_height). Read only with dummy_hold_clearance_m or
    # dummy_hold_support_share.
    dummy_hold_stand_height_m: float | None = None
    # The jointed dummy: the share of its weight, in [0, 1), the hold servo carries while it holds it --
    # pushed up at its centre of mass, with a spring and a damper toward the stand height plus the pelvis height the
    # gait's joint targets imply (_support_force_z). The feet carry the rest. None: no vertical term (unless carried).
    dummy_hold_support_share: float | None = None
    # The jointed dummy: the hold's natural frequency across the plane (Hz) when it is to differ from
    # dummy_hold_natural_freq_hz -- the walking legs, not the hold, are to move the body. None: the hold's own.
    dummy_hold_plane_freq_hz: float | None = None
    # The jointed dummy (S1d's use_jointed): its legs walk the gait table inside its asset at the scripted
    # speed, which ramps up from rest over dummy_gait_accel_time_s and back to rest at the far side's stop
    # line over dummy_gait_decel_time_s. Off, the rigid dummy is carried or slid.
    dummy_gait: bool = False
    dummy_gait_accel_time_s: float = 1.0
    dummy_gait_decel_time_s: float = 1.0
    # Which inertia the hold servo's gains are derived from: "whole_body_feet" -- the rigid dummy tipping about
    # its feet -- or "upper_body_hips" -- the jointed dummy's regions above the hips about the hip
    # pivots (dummy_hip_above_root_m above the root link's centre), the legs being driven by their own joints.
    dummy_hold_inertia_model: str = "whole_body_feet"
    dummy_upper_body_prims: tuple = ()
    dummy_region_prims: tuple = ()
    dummy_hip_above_root_m: float | None = None
    # Joint drive gains per degree of freedom name, (stiffness, damping): while walking, and after the release
    # (muscle tone holding the pose the dummy was let go in).
    dummy_walk_gains: dict | None = None
    dummy_tone_gains: dict | None = None
    dummy_hold_damping_ratio: float = 1.0
    # min_distance_human below which the hold servo is switched off for the
    # rest of the episode, so that the dummy is already a free body when the
    # first contact lands.
    #
    # **Not a surface-to-surface gap.**
    # min_distance_human uses the robot's link origins as
    # points (no robot collider size at all) and collapses every human region
    # to one 0.1 m sphere against real extents reaching 0.265 m -- schema
    # v0.1, "safety.min_distance_human 산출 방법". So this gate is best-effort
    # and the guarantee lives in the contact backstop instead; see
    # _update_hold_release. 0.10 m is what measured out to a few steps of
    # margin at this scenario's fastest closing speed -- short enough that a
    # released rigid figure has not had time to topple. None: no distance
    # gate, the contact backstop alone.
    dummy_release_gap_m: float | None = 0.10
    # Backstop threshold: force on a non-foot robot link above which the hold
    # is released immediately, whatever the distance gate thinks. Matched to
    # the noise floor EpisodeRecorder's own stance detection uses (10 N,
    # measured to sit in the empty gap between PhysX's exact-zero swing
    # readings and real contact) -- see recorder._STANCE_FORCE_THRESHOLD_N.
    robot_contact_threshold_n: float = 10.0
    # Saturation cap, as a multiple of the dummy's own measured weight (see
    # _advance_dummy's docstring for the derivation and
    # why weight — not an arbitrary Newton value — is the reference scale).
    # 10x is generous relative to in-distribution operation (a few hundred N
    # at most, check_dummy_hold.py) while staying well under the ~20-50 kN
    # measured from unbounded position-error blowups.
    dummy_hold_force_cap_weight_multiple: float = 10.0
    # Per-axis torque cap, expressed as an angular error (in degrees) rather
    # than a raw N*m value: the cap is how much torque *that axis's own*
    # ang_stiffness would produce at this many degrees of error
    # (_advance_dummy). Per-axis, not a shared vector-norm cap
    # (see the _advance_dummy call site): pitch/roll's ang_stiffness runs
    # ~4.5x yaw's, so a shared norm cap sized for pitch/roll would leave
    # enormous headroom in Newton-metre terms while still being tight enough
    # in yaw's own (much softer) terms to saturate and crush it.
    #
    # NOT scaled by dummy_hold_force_cap_weight_multiple (unlike the linear
    # cap): dense per-step logging (forced tracking, seed=0 — see
    # _advance_dummy's note at the clamp call site) showed that
    # this closed loop is only marginally stable once a disturbance pushes
    # ang_error past a few tens of degrees — torque then grows ~3-5x per
    # *step*, regardless of velocity source (FD or PhysX's own, the latter
    # if anything worse) and *worse*, not better, at a higher damping ratio.
    # A cap_mult=10-style "generous, essentially never triggers" bound
    # (~282000 N*m for pitch/roll) does not
    # protect against this — by the time torque reaches that magnitude the
    # runaway is already unrecoverable. 15 deg sits just above the normal
    # operating range measured (a few hundred to ~2000 N*m for
    # pitch/roll during ordinary, even fairly disturbed, tracking) so the
    # cap engages while a runaway is still small, at the cost of being a
    # real (not purely nominal) limit on how hard this axis's spring may
    # ever pull — a real, open trade-off, not a free safety margin: this
    # bounds the symptom, it does not fix the underlying discrete-time
    # stability margin.
    #
    # Applies to x/y (pitch/roll, the tipping axes) only — see
    # dummy_hold_torque_cap_angle_deg_yaw for why yaw needs a different
    # value, not this same one broadcast across all three axes.
    dummy_hold_torque_cap_angle_deg: float = 15.0
    # yaw's own cap angle, separate from dummy_hold_torque_cap_angle_deg
    # above (_advance_dummy's torque_cap_axes construction):
    # applying the same *tight* angle to yaw as to pitch/roll measured worse
    # (140 deg, against an effectively uncapped ~40 deg before any per-axis
    # angle split) — the tight cap suppresses pitch/roll's runaway
    # correctly, but on yaw it instead caps *below* what yaw was measured
    # to need to counteract a real disturbance it is separately subject to
    # (an open question), so the axis
    # just drifted further, unable to correct itself back. 90 deg keeps
    # yaw's restoring authority effectively unbounded in practice while
    # still being a real cap in principle: yaw's own ang_stiffness runs
    # ~178x softer than pitch/roll's (see _advance_dummy's
    # docstring), so even this much larger angle allowance stays a small
    # absolute torque (~100 N*m/rad * radians(90) =~ 157 N*m) — under 3.5%
    # of pitch/roll's own tightened cap, nowhere near reintroducing the
    # runaway magnitude that motivated tightening pitch/roll in the first
    # place.
    dummy_hold_torque_cap_angle_deg_yaw: float = 90.0
