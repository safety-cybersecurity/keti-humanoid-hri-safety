"""Human entry for S1a: wait off-path, then cross the robot's path.

Standing is enforced in software, not by a PhysX kinematic flag. A kinematic
root was the design's original intent, but it does not work for this asset —
tried twice, two different failures:

- The same prim carrying both ``UsdPhysics.ArticulationRootAPI`` and
  ``kinematicEnabled`` at once. PhysX refuses this outright — confirmed
  empirically (``~/isaacsim/python.sh generation/dummy/build_dummy.py --root
  kinematic --geometry-only`` followed by loading the result raises
  "ArticulationRootAPI definition on a kinematic rigid body is not allowed,
  articulation root will be ignored", and Isaac Lab then fails with "Failed
  to create articulation").
- Moving ``ArticulationRootAPI`` one level up onto ``/KetiDummy``
  (the Xform ancestor), leaving only the root LINK (BR-04) kinematic — the
  placement Isaac Sim's own robot assets typically use. This sidesteps the
  first constraint (the two APIs are on different prims now) but a *different*,
  more fundamental one replaces it: "Articulations with kinematic bodies are
  not supported, please exclude joint (.../BR_03/fixed_to_BR_04) from
  articulation" — one such error per fixed joint the kinematic root
  participates in. PhysX's articulation solver does not allow a kinematic
  member at all, root or not, regardless of where the root-API marker sits.
  This is the rejection that making all 17 regions kinematic would bring,
  arriving from just the root's own joints, rather than a new one.
  ``build_dummy.py`` keeps ``ArticulationRootAPI`` on the root LINK prim
  itself — the ancestor placement buys nothing once this constraint is
  known, and moving it would silently break tooling that expects the BR code
  and the articulation-root marker on the same prim (``generation/dummy/
  check_contact_resolution.py``).

Given that, the dummy's 17 body regions are ordinary dynamic rigid bodies,
held together by 16 fixed joints (0 DOF, so the assembly has no internal
pose to drift). Holding the root up against gravity by rewriting its
position, orientation and velocity outright every step
(``write_root_state_to_sim``) works for "still standing" but, per the
forced-overlap measurement (check_forced_overlap.py), also erases the
contact channel: a teleport is a position *constraint*, not a physical
interaction, so PhysX has no interpenetration or relative velocity left to
turn into a contact force (0 N at rest; ~20 kN — a solver artifact — under
forced overlap). The hold is instead a bounded spring-damper force/torque on
the root link every step (``S1aStraightWalkCommand._advance_dummy``,
mdp/commands.py, via the permanent wrench composer — the instantaneous one
gives a 25% duty cycle from being reset every physics substep; see that
method's docstring) — the dummy is still a fully dynamic articulation, but
genuinely simulated rather than repositioned, so PhysX's own contact solver
resolves interpenetration into an ordinary, finite force. The 16 fixed
joints still carry the other 16 regions along rigidly whichever way the root
moves: the 17 links move as one uniform block under an unforced root, and
under continuous correction too (z-spread of the 17 regions constant to the
hundredth of a millimetre across a full episode).
See ``_advance_dummy``'s own docstring for the gains and why they are sized
the way they are.

S1a measures the robot's reaction, not the human's gait — a scripted
crossing is the honest level of fidelity here.

The pure geometry (crossing point, waiting-offset derivation) lives in
mdp/entry_geometry.py, torch-free so CI can test it directly — this module
is the thin Isaac-dependent wrapper: two live measurements (how far the
robot's and the dummy's own bodies reach past their roots) plus the actual
scene writes.

How the waiting point, the trigger and the scripted walk are set:

- The waiting lateral offset is derived, not a hand-picked literal.
  ``min_distance_human`` (mdp/safety.py) is deliberately omnidirectional — it
  has no notion of "in the robot's lane", because the judged output S1a
  exists to produce needs exactly that. That means a *waiting* dummy sitting
  within the SSM's reach — root-to-root or, once the robot's own arms/legs
  extend past its root, closer than that — reports a protective stop before
  it has moved at all. Measured with a hand-picked offset: seed 0 latched at
  step 297 (distance 1.691 m) while the dummy was still at y=1.476, 373
  steps before the entry itself started (step 670) — the stop was never
  caused by the entry. The fix is not to change ``min_distance_human``
  (S1a's outcome table needs that omnidirectional quantity) but to keep the
  waiting point far enough out that it never falls inside the band that
  function would report as violated.

- The trigger is along-track, not straight-line. ``entry_trigger_distance``
  (catalogue §4.5) means the robot's remaining reaction margin before it
  reaches the point where this episode's entry line crosses its lane, not
  raw distance to wherever the dummy happens to be standing. Straight-line
  distance silently killed the bottom of the catalogue's [1.0, 4.0] m range
  the moment the waiting offset grew past it. ``entry["x_cross"]`` is where
  the entry line (from the waiting point, at the drawn angle) meets y=0 in
  the env-local frame; the trigger compares the robot's remaining x-distance
  to that point against the drawn trigger_distance.

- The dummy's standing z is captured once, from the very first read (before
  any drift), and reasserted every write from then on — not read back from
  the live simulator each time.

- The scripted trajectory (``entry["target_xy"]``) is integrated purely from
  the drawn speed/angle/dt in Python state, never read back from the live
  simulated position as the base for "one more step".

The corridor is not made long enough to out-run heading drift. Sized that
way (an empirical delay term plus a 4.0 m "stopped drift margin", both aimed
at out-running a heading-drift problem with distance rather than fixing the
drift itself) it gave 0 pretrigger stop-band violations over 13 tested
episodes, but by Monte Carlo only ~5-6% of
catalogue-uniform draws ever let the human actually reach the robot's lane
within one episode. The chain of causes: "정책이 yaw 를 흘림 → 로봇이 대기
더미 쪽으로 휨 → 대기 오프셋을 키워야 했음 → 인체가 차선까지 멀리 걸어야 함 →
20 s 에피소드에 안 들어감 → 교차율 5%" — fix the first link and the rest
stops being necessary. ``S1aStraightWalkCommand._heading_correction``
(mdp/commands.py) steers the robot back toward its spawn heading instead of
leaving ``ang_vel_z`` at a constant 0, so:

- There is no stopped-drift margin, and ``wait_x``'s delay term is a
  *derived* 1x trigger_distance_range[1], not an empirical multiple (see
  below for why it is not 0). What the corridor holds is the honest sum of
  derived terms: the SSM stop band, both bodies' measured reach, and the
  robot's own spawn margin — see ``mdp/entry_geometry.py``'s
  ``derive_wait_y``/``derive_wait_x``.

- The dummy's own lateral reach is in the derivation too
  (``_lateral_reach`` below, shared with the robot). ``min_distance_human``
  measures from each dummy body-region *center* minus a fixed radius, and
  the shoulder/arm regions sit ~0.2 m off the dummy's own root in y
  (anthropometry.yaml) — a waiting dummy's near-side arm is that much
  closer to the robot than its root alone, a clearance a root-to-root
  formula does not account for.

- The stop-band term stays sized to the stop band itself, not the wider
  slowdown-start distance: a waiting human causing the robot to merely slow
  down (not stop) is correct, unflagged SSM behavior, and padding the
  corridor out to the slowdown radius would only shrink the crossing rate
  again for no correctness benefit.

- "Almost" exactly the derived terms, not exactly: check_entry.py's forced
  dangerous-draw testing (walk_speed near the top of its range, wide angle
  — the combination seeds rarely land on by chance) found heading
  correction reduces drift to a few degrees, not to 0, and
  ``_lateral_reach`` is measured once from a standing pose at reset, which
  does not capture how far a swinging limb reaches during actual gait.
  ``reset_dummy_entry``'s ``residual_drift_margin`` parameter (default 0
  here; S1aEnvCfg supplies a small nonzero value, with the measurement that
  justifies it — see S1aEnvCfg for the current number rather than a second
  copy of it here) covers that — small, well under the 4.0 m above, and
  justified by this specific test rather than carried over by default.

See ``measure_crossing_rate.py`` for the crossing-rate numbers, and
``tests/generation/test_entry_geometry.py`` for the formulas pinned down as
unit tests.

Two more terms in the geometry, independent of each other:

- ``derive_wait_y`` takes ``human_region_radius``. ``min_distance_human``
  reports a *surface* distance (``centre_distance - human_region_radius``,
  clamped at 0), not the centre-to-centre distance every other term here is
  stated in — leaving this out leaves the real margin needed
  (``stop_band + human_region_radius``) short by exactly
  ``human_region_radius`` (0.1 m, ``S1aStraightWalkCommandCfg``'s value).

- ``derive_wait_x`` has a derived (not empirical) delay term —
  ``trigger_distance_max`` (the catalogue's ``entry_trigger_distance`` upper
  bound, 1x). At 0 it has an effect distinct from the heading drift above:
  x_cross's minimum (spawn_margin_x alone, ~0.5 m) sits inside the robot's
  own spawn randomization range, so ~10.5% of draws already satisfy the
  along-track trigger at step 0 — a "trigger_distance" that is recorded
  but never actually happens, and catalogue §4.4's wait -> trigger ->
  enter sequence never occurs either. 1x guarantees a real wait phase for
  every draw (with a delay term in place the step-0 rate measured 0.0%).

S1aEnvCfg also disables two unforced sources of lateral drift
(``reset_base``'s own ``velocity_range["y"]`` and ``push_robot``) — neither
the trained policy nor the heading correction above has any *lateral-
position* feedback, only a heading loop, so a lateral velocity kick from
either becomes a permanent offset rather than something that gets damped
out. Both are inherited locomotion-training robustness events, not among
catalogue §4.5's eight S1a-specific randomized axes — the same precedent
already used for yaw (S1aEnvCfg). ``residual_drift_margin`` was
measured with them disabled; see S1aEnvCfg for the current value.
"""

from __future__ import annotations

import math

import torch

from .entry_geometry import (
    check_stop_angles, crossing_x, derive_wait_x, derive_wait_y, facing_quat_xyzw, travel_facing_yaw_deg,
    walk_angle_rad,
)

# The dummy stands as built (generation/dummy/anthropometry.yaml encodes the
# standing posture), turned only about +z by facing_yaw_deg: 0 -- the identity,
# S1a -- faces the robot's own heading (+x), so the robot sees its back. S1d
# turns it to its own walk instead (face_travel, 2026-10-01): a facing per
# episode, derived from the drawn entry angle. Between 2026-09-29 and then S1d
# passed 180, facing the robot -- and the dummy crossed the aisle sideways.


def _lateral_reach(env, asset_name: str) -> float:
    """How far asset_name's own links reach sideways from its root, right now.

    Measured from the live asset, not a spec number — the same reason joint
    order is read from the asset rather than typed by hand (CONTRIBUTING.md
    「관절 순서」). Used for both the robot (``separation_required`` treats it as a
    single point at its root) and the dummy (``min_distance_human`` treats
    each body region as a point at its own center, not the dummy's root —
    see module docstring). Max over every body and every env —
    a single scalar applied uniformly, matching how the waiting point itself
    is not per-env.
    """
    asset = env.scene[asset_name]
    lateral = (asset.data.body_pos_w.torch[:, :, 1] - asset.data.root_pos_w.torch[:, 1:2]).abs()
    return float(lateral.max())


def _horizontal_reach(env, asset_name: str) -> float:
    """How far asset_name's links reach from its root in the horizontal plane, right now.

    _lateral_reach without an axis chosen: the same number however the asset is turned about its
    root's vertical axis, and never less than its |y| reach at any such turn. The waiting offset
    of a dummy that stands at a different yaw every episode (face_travel) is derived from this,
    so the offset does not move with the turn -- or with whichever turn the dummy still has when
    it is measured.
    """
    asset = env.scene[asset_name]
    offset = asset.data.body_pos_w.torch[:, :, :2] - asset.data.root_pos_w.torch[:, None, :2]
    return float(offset.norm(dim=-1).max())


def reset_dummy_entry(
    env,
    env_ids: torch.Tensor,
    speed_range: tuple[float, float],
    angle_range_deg: tuple[float, float],
    trigger_distance_range: tuple[float, float],
    stop_band_clearance: float,
    spawn_margin_x: float,
    spawn_margin_y: float,
    human_region_radius: float = 0.0,
    residual_drift_margin: float = 0.0,
    facing_yaw_deg: float = 0.0,
    stop_at_far_side: bool = False,
    face_travel: bool = False,
    asset_name: str = "dummy",
):
    """Draw this episode's entry speed, angle and trigger distance; park the dummy off-path.

    ``facing_yaw_deg`` turns the parked dummy about +z (entry_geometry.facing_quat_xyzw). It is
    not drawn -- no random number is consumed, so S1a's draws are unchanged -- and it is stashed
    for the hold servo (commands._advance_dummy), which holds the dummy at the same facing.

    ``face_travel`` (S1d, 2026-10-01) turns each env's dummy to its own walk instead: the facing is
    entry_geometry.travel_facing_yaw_deg of the angle drawn here, so ``facing_yaw_deg`` is not read.
    Nothing more is drawn for it. The dummy's reach is then taken in the horizontal plane
    (_horizontal_reach), not along y.

    ``stop_at_far_side`` (S1d, 2026-10-01) stashes ``stop_y``: the waiting offset mirrored across the
    robot's lane, where _advance_dummy stops the scripted walk (entry_geometry.lateral_stop_fraction).
    Without it S1d's dummy walked on into the far rack with its hold servo still on. The aisle guard's
    check on the waiting offset covers the mirrored one too. Like the facing, nothing is drawn for it,
    and S1a, which leaves it off, never gets the key.

    ``stop_band_clearance``/``spawn_margin_x``/``spawn_margin_y``/
    ``human_region_radius`` are computed in S1aEnvCfg from meta/rules.json,
    the command cfg, and the inherited reset_base event — not literals here
    (see module docstring). ``residual_drift_margin`` defaults to 0 (see
    ``entry_geometry.derive_wait_y`` for what it covers and why it is not 0
    in S1aEnvCfg).
    """
    dummy = env.scene[asset_name]
    n = len(env_ids)
    device = env.device

    def uniform(lo, hi):
        return torch.empty(n, device=device).uniform_(lo, hi)

    speed = uniform(*speed_range)
    angle = uniform(*angle_range_deg) * math.pi / 180.0
    trigger = uniform(*trigger_distance_range)

    wait_y = derive_wait_y(
        stop_band_clearance=stop_band_clearance,
        robot_lateral_reach=_lateral_reach(env, "robot"),
        dummy_lateral_reach=_horizontal_reach(env, asset_name) if face_travel else _lateral_reach(env, asset_name),
        spawn_margin_y=spawn_margin_y,
        human_region_radius=human_region_radius,
        residual_drift_margin=residual_drift_margin,
    )
    wait_x = derive_wait_x(
        spawn_margin_x=spawn_margin_x,
        wait_y=wait_y,
        trigger_distance_max=trigger_distance_range[1],
        min_angle_rad=math.radians(angle_range_deg[0]),
    )

    # Stash on the env: the command term reads them every step
    # (_advance_dummy), the recorder writes them into the episode meta as
    # realized randomization values.
    entry = getattr(env, "s1a_entry", None)
    if entry is None:
        entry = {
            "speed": torch.zeros(env.num_envs, device=device),
            "angle": torch.zeros(env.num_envs, device=device),
            "trigger_distance": torch.zeros(env.num_envs, device=device),
            # Latched, not re-evaluated: see _advance_dummy for why a bare
            # "distance < trigger_distance" re-checked every step lets the
            # crossing stall partway (observed: an oblique angle can make the
            # dummy's own x-drift widen separation faster than its y-drift
            # closes it, flipping the raw comparison back off before the
            # dummy ever reaches the robot's lane).
            "triggered": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            # Captured once, from the very first (pre-drift) read, and never
            # touched again — see the module docstring's bullet on the standing z.
            "stand_z": dummy.data.root_pos_w.torch[:, 2].clone(),
            # The scripted target, integrated purely from speed/angle/dt —
            # never read back from the simulator. World frame (env_origins
            # already folded in).
            "target_xy": torch.zeros(env.num_envs, 2, device=device),
            # Where this episode's entry line crosses the robot's lane
            # (y=0 in the env-local frame — S1aEnvCfg pins robot spawn yaw to
            # 0, so that lane is a stable invariant). World frame. Drives the
            # along-track trigger in _advance_dummy.
            "x_cross": torch.zeros(env.num_envs, device=device),
            # The pose the dummy is parked in and held at (_advance_dummy's target).
            "facing_quat": torch.tensor(facing_quat_xyzw(0.0), device=device).repeat(env.num_envs, 1),
            # Its yaw in degrees, per env, as the sidecar records it. float64: with face_travel the
            # recorded value is exactly travel_facing_yaw_deg of the recorded angle.
            "facing_yaw_deg": torch.zeros(env.num_envs, dtype=torch.float64, device=device),
        }
        env.s1a_entry = entry
    entry["speed"][env_ids] = speed
    entry["angle"][env_ids] = angle
    entry["trigger_distance"][env_ids] = trigger
    entry["triggered"][env_ids] = False

    origin_xy = env.scene.env_origins[env_ids, :2]
    entry["target_xy"][env_ids, 0] = wait_x + origin_xy[:, 0]
    entry["target_xy"][env_ids, 1] = wait_y + origin_xy[:, 1]
    # crossing_x is numpy-only (mdp/entry_geometry.py, torch-free so CI can
    # test it) — cross the torch/numpy boundary explicitly here rather than
    # calling a numpy ufunc on a CUDA tensor directly, which either raises
    # or silently misbehaves. Same boundary pattern
    # S1aStraightWalkCommand._measure_min_distance already uses for
    # min_distance_human.
    x_cross_np = crossing_x(wait_x, wait_y, angle.detach().cpu().numpy())
    entry["x_cross"][env_ids] = (
        torch.from_numpy(x_cross_np).to(device=device, dtype=entry["x_cross"].dtype)
        + origin_xy[:, 0]
    )

    pos = torch.cat(
        [entry["target_xy"][env_ids], entry["stand_z"][env_ids].unsqueeze(-1)], dim=-1
    )
    if face_travel:
        yaw_deg = [travel_facing_yaw_deg(math.degrees(float(a))) for a in angle]
    else:
        yaw_deg = [float(facing_yaw_deg)] * n
    quat = torch.tensor([facing_quat_xyzw(y) for y in yaw_deg], device=device)
    entry["facing_quat"][env_ids] = quat
    entry["facing_yaw_deg"][env_ids] = torch.tensor(yaw_deg, dtype=torch.float64, device=device)
    # Whether the facing differs per episode: the hold servo then takes the dummy's inertia at each
    # episode's facing (commands._advance_dummy).
    entry["face_travel"] = bool(face_travel)
    if stop_at_far_side:
        check_stop_angles(angle_range_deg)
        if "stop_y" not in entry:
            entry["stop_y"] = torch.zeros(env.num_envs, device=device)
            entry["walk_ended"] = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        entry["stop_y"][env_ids] = origin_xy[:, 1] - wait_y
        entry["walk_ended"][env_ids] = False
    zero_vel = torch.zeros(n, 6, device=device)
    root_state = torch.cat([pos, quat, zero_vel], dim=-1)
    # write_root_state_to_sim is deprecated in this Isaac Lab version in favour
    # of write_root_link_pose_to_sim_index + write_root_com_velocity_to_sim_index,
    # but it still does exactly that under the hood (read from source) — one
    # call, and the explicit zero velocity is what keeps gravity from
    # accumulating in the dynamic root between writes (see module docstring).
    dummy.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_dummy_scripted(env, env_ids: torch.Tensor, plan, asset_name: str = "dummy"):
    """Park the dummy where a condition row's script starts it (generation/human_script.HumanPlan): at the plan's start,
    standing turned to its yaw, with ``s1a_entry`` filled from the plan for the command term -- nothing is drawn.

    The entry has the keys reset_dummy_entry gives it, read the same way by the hold servo and the recorder: the walk's
    speed, its angle in the command term's convention (entry_geometry.walk_angle_rad -- the target moves along the
    plan's direction), the trigger's value as the trigger distance and the person's start x as the crossing point (a
    ``robot_behind`` trigger compares the robot's root x with it), the target at the start, the facing, and
    ``face_travel`` on (the facing differs per row). Four more say how the walk goes and ends --
    ``trigger_kind`` (human_script.Trigger), ``start_xy``, ``walk_dir`` and ``stop_after_m`` (infinite without a stop)
    -- and are read by the conditions-table command term's _trigger_latched and _stop_distance; there is no
    ``stop_y``, so the far-side stop line of S1d's entry is not applied. A plan that does not walk has speed 0.

    ``plan`` may come as human_script.plan_record's plain data, the form an environment config carries it in."""
    if not hasattr(plan, "trigger"):
        from generation.human_script import plan_from_record

        plan = plan_from_record(plan)
    dummy = env.scene[asset_name]
    n = len(env_ids)
    device = env.device
    entry = getattr(env, "s1a_entry", None)
    if entry is None:
        entry = {
            "speed": torch.zeros(env.num_envs, device=device),
            "angle": torch.zeros(env.num_envs, device=device),
            "trigger_distance": torch.zeros(env.num_envs, device=device),
            "triggered": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            "stand_z": dummy.data.root_pos_w.torch[:, 2].clone(),
            "target_xy": torch.zeros(env.num_envs, 2, device=device),
            "x_cross": torch.zeros(env.num_envs, device=device),
            "facing_quat": torch.tensor(facing_quat_xyzw(0.0), device=device).repeat(env.num_envs, 1),
            "facing_yaw_deg": torch.zeros(env.num_envs, dtype=torch.float64, device=device),
            "walk_ended": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            "start_xy": torch.zeros(env.num_envs, 2, device=device),
            "walk_dir": torch.zeros(env.num_envs, 2, device=device),
            "stop_after_m": torch.zeros(env.num_envs, device=device),
        }
        env.s1a_entry = entry
    walking = plan.walk_dir is not None and plan.speed_mps > 0.0
    origin_xy = env.scene.env_origins[env_ids, :2]
    start = torch.tensor(plan.start_xy, dtype=entry["target_xy"].dtype, device=device).unsqueeze(0) + origin_xy
    entry["speed"][env_ids] = float(plan.speed_mps) if walking else 0.0
    entry["angle"][env_ids] = walk_angle_rad(plan.walk_dir) if walking else 0.0
    entry["trigger_distance"][env_ids] = float(plan.trigger.value)
    entry["trigger_kind"] = str(plan.trigger.kind)
    entry["triggered"][env_ids] = False
    entry["target_xy"][env_ids] = start
    entry["start_xy"][env_ids] = start
    entry["x_cross"][env_ids] = start[:, 0]
    entry["walk_dir"][env_ids] = torch.tensor(plan.walk_dir if walking else (0.0, 0.0), dtype=entry["walk_dir"].dtype,
                                              device=device)
    entry["stop_after_m"][env_ids] = math.inf if plan.stop_after_m is None else float(plan.stop_after_m)
    quat = torch.tensor(facing_quat_xyzw(plan.yaw_deg), device=device).repeat(n, 1)
    entry["facing_quat"][env_ids] = quat
    entry["facing_yaw_deg"][env_ids] = float(plan.yaw_deg)
    entry["face_travel"] = True
    entry["walk_ended"][env_ids] = False
    pos = torch.cat([start, entry["stand_z"][env_ids].unsqueeze(-1)], dim=-1)
    dummy.write_root_state_to_sim(torch.cat([pos, quat, torch.zeros(n, 6, device=device)], dim=-1), env_ids=env_ids)
