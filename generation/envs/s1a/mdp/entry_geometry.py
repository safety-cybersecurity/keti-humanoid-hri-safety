"""Pure-numpy geometry for S1a's human entry: the crossing point, the derived
waiting offset, and whether a given draw lets the human reach the robot's
lane at all within one episode.

Everything here takes plain floats or numpy arrays and returns the same —
no torch, no Isaac Sim — for the same reason mdp/safety.py is numpy-only:
CI runs this with requirements.txt alone. It also keeps the crossing rate
reproducible: this module plus measure_crossing_rate.py are what the rate is
computed with, not a one-off script.

mdp/events.py is the Isaac-dependent half: a thin torch wrapper around these
functions plus two live measurements (how far the robot's and dummy's own
bodies reach past their roots) that only exist once the simulator is
running.
"""

from __future__ import annotations

import math

import numpy as np


def facing_quat_xyzw(yaw_deg: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) of an upright pose turned yaw_deg about +z: which way the dummy faces.

    0 is the identity S1a has always written (the dummy faces the robot's own heading, +x,
    so the robot sees its back); 180 faces the robot (S1d, 2026-09-29). Components within
    1e-12 of 0 are snapped to exactly 0, so 180 is (0, 0, 1, 0) rather than carrying cos's
    6e-17 -- and 0 stays the identity to the bit.
    """
    yaw = float(yaw_deg)
    if not math.isfinite(yaw):
        raise ValueError(f"facing yaw must be finite, got {yaw_deg!r}")
    half = math.radians(yaw) / 2.0
    z, w = math.sin(half), math.cos(half)
    return (0.0, 0.0, 0.0 if abs(z) < 1e-12 else z, 0.0 if abs(w) < 1e-12 else w)


def travel_facing_yaw_deg(angle_deg) -> float:
    """The facing (deg about +z, for facing_quat_xyzw) that turns the dummy's front onto its own walk.

    The entry angle is measured from the robot's heading +x and the walk goes -(cos, sin)(angle)
    (crossing_x), so the facing is the angle plus a half turn. Folded into (-180, 180]: one value per
    direction in the sidecar, and a quaternion with w >= 0. S1d (2026-10-01): until then its dummy
    faced the robot at a fixed 180 deg and crossed the aisle sideways.
    """
    angle = float(angle_deg)
    if not math.isfinite(angle):
        raise ValueError(f"entry angle must be finite, got {angle_deg!r}")
    yaw = math.fmod(angle + 180.0, 360.0)
    if yaw > 180.0:
        yaw -= 360.0
    elif yaw <= -180.0:
        yaw += 360.0
    return yaw


def walk_angle_rad(walk_dir) -> float:
    """The entry angle (rad, from the robot's heading +x) whose scripted walk runs along ``walk_dir``.

    The command term moves the scripted target by -(cos, sin)(angle) per metre (crossing_x), so a person
    who walks along (dx, dy) -- a condition row's script, events.reset_dummy_scripted -- has the angle
    atan2(-dy, -dx). ``walk_dir`` need not be of unit length but must have one.
    """
    dx, dy = (float(v) for v in walk_dir)
    norm = math.hypot(dx, dy)
    if not (math.isfinite(norm) and norm > 0.0):
        raise ValueError(f"walk direction must be a finite, non-zero vector, got {walk_dir!r}")
    return math.atan2(-dy / norm, -dx / norm)


def yawed_axis_inertia(mass, offsets_body, cos_yaw, sin_yaw):
    """Point-mass inertia about the world x, y and z axes through the centre of mass, for a body turned
    about +z: ``mass`` (..., B), ``offsets_body`` (..., B, 3) from the centre of mass in the body frame,
    ``cos_yaw`` / ``sin_yaw`` (...,). Returns three (...,) arrays.

    The hold servo sizes one gain per world axis from these (commands._advance_dummy). Taken once at the
    pose of its first call they were right while every episode stood the dummy at the same facing; with
    the facing following each episode's entry angle (travel_facing_yaw_deg) the two tipping axes are
    taken at that episode's facing instead, so the first episode of a run leaves nothing in the gains of
    the ones after it. The yaw axis does not depend on the turn. numpy arrays and torch tensors alike --
    arrays throughout: a Python float for the cosine or sine has no axis to add and is refused.
    """
    c, s = cos_yaw[..., None], sin_yaw[..., None]
    bx, by, z = offsets_body[..., 0], offsets_body[..., 1], offsets_body[..., 2]
    x = bx * c - by * s
    y = bx * s + by * c
    return ((mass * (y * y + z * z)).sum(-1), (mass * (x * x + z * z)).sum(-1), (mass * (x * x + y * y)).sum(-1))


def relative_gap_max(got, want) -> float:
    """Largest |got - want| / want over the pairs; ``math.inf`` unless both hold the same number of
    positive finite values (at least one). max() over the gaps alone returns 0 for [x, nan, x], and a NaN
    result compares False against any tolerance -- either way a missing number would pass."""
    got, want = [float(v) for v in got], [float(v) for v in want]
    if not got or len(got) != len(want) or not all(math.isfinite(v) and v > 0.0 for v in (*got, *want)):
        return math.inf
    return max(abs(a - b) / b for a, b in zip(got, want))


def crossing_x(wait_x, wait_y, angle_rad):
    """x where the entry line (from (wait_x, wait_y), direction -(cos, sin)(angle))
    meets the robot's lane, y=0.

    Accepts scalars or numpy arrays (broadcasts). Uses cos/sin rather than a
    literal tan() so it stays well-defined at angle=90 deg, where tan()
    blows up but cos/sin does not.
    """
    return wait_x - wait_y * np.cos(angle_rad) / np.sin(angle_rad)


def derive_wait_y(
    stop_band_clearance,
    robot_lateral_reach,
    dummy_lateral_reach,
    spawn_margin_y,
    human_region_radius: float = 0.0,
    residual_drift_margin: float = 0.0,
):
    """Lateral offset the dummy waits at.

    Clears the SSM stop band (at the robot's fastest walk speed) plus both
    bodies' own reach past their roots plus the robot's own spawn-position
    margin plus human_region_radius plus an optional residual_drift_margin.
    All terms are non-negative by construction (each is a distance, a
    measured/derived clearance, or 0), so the result is always at least
    stop_band_clearance — the property that keeps a dummy that is only
    waiting from tripping the protective stop.

    human_region_radius is not optional in spirit, only in signature
    (default 0 keeps old callers/tests honest about what they
    are and are not modeling) — ``min_distance_human`` (mdp/safety.py)
    reports ``max(centre_distance - human_region_radius, 0.0)``, a *surface*
    distance, not the centre-to-centre distance this module's other terms
    are all stated in. Leaving this out means the violation condition is
    actually ``centre_distance - human_region_radius < stop_band``, i.e. the
    real centre-to-centre margin needed is ``stop_band + human_region_radius``
    — a clearance the formula without it silently assumes away.

    residual_drift_margin defaults to 0 — the honest sum of derived terms,
    with nothing empirical layered on.
    check_entry.py's forced dangerous-draw testing (walk_speed near max,
    wide angle) found that heading correction (mdp/commands.py) does not
    drive drift to exactly 0, only to a few degrees, and robot_lateral_reach
    is measured once from a *standing* pose at reset — it does not capture
    how far a swinging arm/leg reaches during actual gait. A small nonzero
    value here is what covers that residual, sized well below the empirical
    4.0 m an uncorrected heading needed because that root cause is fixed at
    its source, not because the residual itself is exactly known. It is
    measured with two more sources of unforced lateral drift removed
    (S1aEnvCfg's reset_base velocity_range["y"] and push_robot, neither of
    which either the policy or the heading correction has any feedback
    against) — see S1aEnvCfg for the current value and what it is based on.
    """
    return (
        stop_band_clearance
        + robot_lateral_reach
        + dummy_lateral_reach
        + spawn_margin_y
        + human_region_radius
        + residual_drift_margin
    )


def derive_wait_x(spawn_margin_x, wait_y, min_angle_rad, trigger_distance_max: float = 0.0):
    """Longitudinal offset.

    Clears the worst-case (shallowest-angle) crossing-point swing
    (wait_y * cot(min_angle_rad) — the entry line's crossing point moves
    this far from wait_x at the most oblique allowed angle), plus the
    robot's own spawn margin, plus trigger_distance_max.

    trigger_distance_max is not there to buy time against heading drift —
    heading correction (mdp/commands.py) fixes that at its source. Leaving
    it at 0 has a different effect: without it, x_cross's minimum
    (spawn_margin_x alone) sits inside the robot's own spawn
    randomization (+-0.5 m by default), so up to ~10% of draws already
    satisfy the along-track trigger condition at step 0, before the robot
    has moved and before any "wait" phase happened at all — a scripted
    trigger_distance that was never actually reached. trigger_distance_max
    (the catalogue's entry_trigger_distance upper bound, 4.0 m — pass
    exactly 1x it, not an empirical multiple) is what guarantees a real wait
    phase exists for every draw regardless of which trigger_distance gets
    drawn, independent of whatever residual_drift_margin derive_wait_y
    carries. Default 0 keeps old callers/tests that don't care about this
    property honest about not modeling it.
    """
    cot_min_angle = abs(np.cos(min_angle_rad) / np.sin(min_angle_rad))
    return spawn_margin_x + trigger_distance_max + wait_y * cot_min_angle


def lateral_stop_fraction(target_y, stop_y, dy):
    """Share of one step's lateral move (the scripted target goes ``target_y -= dy``) that keeps the
    target from passing ``stop_y``: 1 while the line is more than a step away, the remainder on the
    step that reaches it, 0 from then on.

    S1d only (2026-10-01): its dummy stops at the waiting offset mirrored across the robot's lane
    instead of walking on into the far rack with its hold servo still on (the slice measured
    hold_contaminated_steps 83-432 in 7 of 20 episodes against a contract of at most 1). A fraction of
    exactly 1.0 leaves the step's arithmetic bit-identical, so the walk up to the line is unchanged.
    ``dy`` must point toward ``stop_y`` -- check_stop_angles guarantees it for every draw. A ``dy`` of 0 is
    no move: the jointed dummy's scripted speed ramps down to 0 at the line, and there the fraction is 1
    before the line and 0 at or past it, never the NaN of 0/0. A step above 0 divides by ``dy`` exactly (``still`` adds 0 there).
    Works on numpy arrays and torch tensors alike.
    """
    still = dy == 0
    frac = ((target_y - stop_y) / (dy + still)).clip(0.0, 1.0)
    if still.any():
        ahead = (target_y - stop_y) > 0
        frac[still & ahead] = 1.0
        frac[still & ~ahead] = 0.0
    return frac


def latch_walk_end(entry, moving, frac) -> None:
    """Latch ``entry["walk_ended"]`` on the command update whose stop fraction (lateral_stop_fraction)
    first falls below 1: the update that commands the walk's last, partial move. From then on the
    fraction is 0 and the target stands.

    The recorder ends the realized entry speed's window on this flag (EntrySpeedWindow) -- "triggered"
    latches for the rest of the episode and so cannot say when the walk stopped. Here rather than in
    commands.py so the tests step the same latch the command term does. numpy arrays and torch tensors alike.
    """
    entry["walk_ended"] |= moving & (frac < 1.0)


def check_stop_angles(angle_range_deg) -> None:
    """The stop needs every drawn entry angle to carry the walk toward the far side: sin > 0 over the
    whole range, so the lateral step lateral_stop_fraction divides by is never reversed, and 0 only where
    the scripted speed itself is 0."""
    lo, hi = (float(v) for v in angle_range_deg)
    if not 0.0 < lo <= hi < 180.0:
        raise ValueError(f"entry angle range {angle_range_deg} deg must lie strictly inside (0, 180) for the "
                         "walk to reach the far-side stop line")


def trigger_time(x_cross, robot_spawn_x, trigger_distance, robot_walk_speed):
    """Seconds until the robot (walking at a constant robot_walk_speed from
    robot_spawn_x toward +x) comes within trigger_distance of x_cross.

    Ignores the SSM's own speed reduction as the robot nears a still-waiting
    human — a simplification, not a bug: the real robot is briefly slower
    near the corridor (wait_y is derived to allow *some* slowdown,
    just not a full stop, while waiting), so this slightly *overestimates*
    the true reachable fraction of draws, not the reverse.

    Returns None (not NaN — see reaches_lane, which needs a clean boolean)
    if the robot's spawn is already past x_cross: the entry can never
    trigger from a point the robot has already walked beyond in this
    model, matching the live along-track check in
    S1aStraightWalkCommand._advance_dummy (dist_to_crossing >= 0.0).

    Scalar-only (the None-return case does not vectorize over an array
    cleanly) — measure_crossing_rate.py calls this per-draw.
    """
    dist0 = x_cross - robot_spawn_x
    if dist0 < 0.0:
        return None
    if dist0 < trigger_distance:
        return 0.0
    return (dist0 - trigger_distance) / robot_walk_speed


def crossing_time(wait_y, human_entry_speed, angle_rad):
    """Seconds for the human to walk from the waiting offset to y=0 (the
    robot's lane) once moving, at a constant human_entry_speed."""
    return wait_y / (human_entry_speed * np.sin(angle_rad))


def triggers_within_episode(
    wait_x: float,
    wait_y: float,
    robot_walk_speed: float,
    angle_rad: float,
    trigger_distance: float,
    episode_length_s: float,
    robot_spawn_x: float = 0.0,
) -> bool:
    """Whether the entry trigger fires at all before the episode times out —
    weaker than reaches_lane (trigger_time not None and within budget, but
    not requiring the human to have also finished crossing). This is the
    "트리거가 걸릴 수 있는 draw" rate, distinct from the "인체가 실제로 로봇
    차선에 도달하는" rate reaches_lane measures.
    """
    x_cross = crossing_x(wait_x, wait_y, angle_rad)
    t = trigger_time(x_cross, robot_spawn_x, trigger_distance, robot_walk_speed)
    # crossing_x is numpy-vectorized (accepts arrays), so even scalar inputs
    # come back as numpy.float64 -- bool(...) here keeps the return type a
    # genuine Python bool rather than numpy.bool_ (which `is True`/`is False`
    # comparisons, e.g. in tests, would otherwise fail against).
    return t is not None and bool(t <= episode_length_s)


def reaches_lane(
    wait_x: float,
    wait_y: float,
    robot_walk_speed: float,
    human_entry_speed: float,
    angle_rad: float,
    trigger_distance: float,
    episode_length_s: float,
    robot_spawn_x: float = 0.0,
) -> bool:
    """Whether, for this one draw, the human's crossing actually reaches the
    robot's lane (y=0) before the episode ends.

    This is the "does the scenario actually happen" question: a trigger
    firing is necessary but not sufficient — the human
    also has to have enough remaining episode time to walk the lateral
    distance. See trigger_time and crossing_time for what this ignores
    (both make the estimate slightly optimistic, not pessimistic).
    """
    x_cross = crossing_x(wait_x, wait_y, angle_rad)
    t_trigger = trigger_time(x_cross, robot_spawn_x, trigger_distance, robot_walk_speed)
    if t_trigger is None:
        return False
    total = t_trigger + crossing_time(wait_y, human_entry_speed, angle_rad)
    return bool(total <= episode_length_s)


def crossing_rate(
    rng: np.random.Generator,
    n_samples: int,
    wait_x: float,
    wait_y: float,
    robot_walk_speed_range: tuple[float, float],
    human_entry_speed_range: tuple[float, float],
    angle_range_deg: tuple[float, float],
    trigger_distance_range: tuple[float, float],
    episode_length_s: float,
    robot_spawn_x_range: tuple[float, float] = (0.0, 0.0),
) -> float:
    """Monte Carlo fraction of catalogue-uniform draws that reach the robot's
    lane within one episode, for a fixed (wait_x, wait_y) corridor.

    Draws every axis independently and uniformly, matching
    meta/scenarios.json's S1a randomization (robot spawn x is drawn the same
    way via the inherited reset_base event, +-0.5 m by default). Loops in
    plain Python over trigger_time's scalar branch rather than vectorizing —
    n_samples in the hundreds of thousands still runs in well under a second
    for this cheap a per-draw computation, and the scalar form is what stays
    directly comparable to reaches_lane's own (tested) logic.
    """
    robot_speed = rng.uniform(*robot_walk_speed_range, size=n_samples)
    human_speed = rng.uniform(*human_entry_speed_range, size=n_samples)
    angle_rad = np.radians(rng.uniform(*angle_range_deg, size=n_samples))
    trigger_distance = rng.uniform(*trigger_distance_range, size=n_samples)
    robot_spawn_x = rng.uniform(*robot_spawn_x_range, size=n_samples)

    reached = 0
    for i in range(n_samples):
        if reaches_lane(
            wait_x,
            wait_y,
            float(robot_speed[i]),
            float(human_speed[i]),
            float(angle_rad[i]),
            float(trigger_distance[i]),
            episode_length_s,
            robot_spawn_x=float(robot_spawn_x[i]),
        ):
            reached += 1
    return reached / n_samples


def trigger_rate(
    rng: np.random.Generator,
    n_samples: int,
    wait_x: float,
    wait_y: float,
    robot_walk_speed_range: tuple[float, float],
    angle_range_deg: tuple[float, float],
    trigger_distance_range: tuple[float, float],
    episode_length_s: float,
    robot_spawn_x_range: tuple[float, float] = (0.0, 0.0),
) -> float:
    """Monte Carlo fraction of catalogue-uniform draws where the entry
    trigger fires at all within one episode (weaker than crossing_rate — see
    triggers_within_episode). Independent of human_entry_speed, so that axis
    is not drawn here."""
    robot_speed = rng.uniform(*robot_walk_speed_range, size=n_samples)
    angle_rad = np.radians(rng.uniform(*angle_range_deg, size=n_samples))
    trigger_distance = rng.uniform(*trigger_distance_range, size=n_samples)
    robot_spawn_x = rng.uniform(*robot_spawn_x_range, size=n_samples)

    triggered = 0
    for i in range(n_samples):
        if triggers_within_episode(
            wait_x,
            wait_y,
            float(robot_speed[i]),
            float(angle_rad[i]),
            float(trigger_distance[i]),
            episode_length_s,
            robot_spawn_x=float(robot_spawn_x[i]),
        ):
            triggered += 1
    return triggered / n_samples


class EntrySpeedWindow:
    """The span the recorder measures ``human_entry_speed_realized`` over: from the first triggered step to
    the step the scripted walk's end is flagged.

    Without a stop (S1a) there is no such flag and the span runs to the last triggered step, as the recorder
    has always measured. With S1d's far-side stop (lateral_stop_fraction) the dummy stands from the line
    on, still "triggered" (the flag latches): measuring on would divide a frozen displacement by a growing
    time. The slice's episode 2 read 0.538 m/s that way against 0.627 drawn (2026-10-01).

    Both ends are read one command ahead of the body. An env step runs its physics first and the command
    update last (isaaclab ManagerBasedRLEnv.step), and the recorder reads after both: on the step it first
    sees "triggered" the body has not moved yet, and on the step it first sees ``walk_ended``
    (latch_walk_end) the body has not yet made the last, partial move that update commanded. The window
    therefore spans the full-length moves and nothing else -- a body that sat on its target would read the
    drawn speed exactly. Closing one step later would add the partial move over a whole step and read low
    by up to one step's share of the walk, 0.02-0.23 % over the draws the tests step, so the window does
    not close later. What the value still carries is the servo's lag behind the target.
    """

    def __init__(self) -> None:
        self.onset_xy = None
        self.onset_t: float | None = None
        self.last_xy = None
        self.last_t: float | None = None
        self._ended = False

    def update(self, *, triggered: bool, walk_ended: bool, xy, t: float) -> None:
        # The positions keep the dtype they come in (the recorder's float32 reads), so S1a's speed is the
        # same arithmetic as before this window existed.
        if triggered and self.onset_xy is None:
            self.onset_xy, self.onset_t = np.array(xy, copy=True), float(t)
        if triggered and not self._ended:
            self.last_xy, self.last_t = np.array(xy, copy=True), float(t)
        self._ended = bool(walk_ended)

    def observe(self, entry, *, xy, t: float) -> None:
        """One recorder step: env 0's flags as the command update left them (recording runs one env).
        ``walk_ended`` exists only where the walk can stop (S1d, events.reset_dummy_entry)."""
        walk_ended = "walk_ended" in entry and bool(entry["walk_ended"][0])
        self.update(triggered=bool(entry["triggered"][0]), walk_ended=walk_ended, xy=xy, t=t)

    def speed(self) -> float | None:
        """Net displacement over elapsed time across the window; None until it spans two steps."""
        if self.onset_xy is None or self.last_t is None or self.last_t <= self.onset_t:
            return None
        return float(np.linalg.norm(self.last_xy - self.onset_xy) / (self.last_t - self.onset_t))
