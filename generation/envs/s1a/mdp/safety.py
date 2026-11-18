"""Safety quantities for S1a, computed without Isaac Sim.

Everything here takes numpy arrays and returns numpy or plain Python. The
env layer converts; keeping this module free of torch is what lets CI run
it. Nothing in here hard-codes a body-region code or a standard's value.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np


# The dummy's dimensions are a project decision, not a standard's value, and
# they live in one file (its own header says so). contact_relative_speed()'s
# lever bound is a property of those dimensions.
ANTHROPOMETRY_PATH = Path(__file__).resolve().parents[3] / "dummy" / "anthropometry.yaml"


def dummy_lever_cap(anthropometry_path: str | Path) -> float:
    """Largest half-extent among the dummy's own regions, in metres.

    contact_relative_speed() evaluates each body's velocity away from its centre
    and must not push that evaluation point off the body. The bound is a property
    of the geometry, so it is read from the file that defines the geometry rather
    than typed here -- change a segment's size and this follows.
    """
    import yaml

    spec = yaml.safe_load(Path(anthropometry_path).read_text(encoding="utf-8"))
    extents: list[float] = []
    for region in (spec.get("regions") or {}).values():
        if "size" in region:
            extents.append(max(float(v) for v in region["size"]) / 2.0)
        if "radius" in region:
            extents.append(float(region["radius"]))
        if "height" in region:
            # A capsule's half-length is height/2 plus the cap it carries.
            extents.append(float(region["height"]) / 2.0 + float(region.get("radius", 0.0)))
    if not extents:
        raise ValueError(f"부위 치수를 하나도 읽지 못했습니다: {anthropometry_path}")
    return max(extents)


def contact_relative_speed(
    dummy_pos: np.ndarray,
    dummy_vel: np.ndarray,
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    lever_cap_m: float,
) -> np.ndarray:
    """Relative speed at the contact-point proxy, one value per dummy body.

    Table A.4's transferred-energy limit and equation (A.6)'s velocity limit both
    need the relative velocity **at the contact point**. PhysX contact points are
    not available on this pipeline -- ``track_contact_points`` needs a filter
    expression and all three variants of that failed (``check_contact_filter.py``:
    two attach and report nothing over 700 steps, the third PhysX rejects) -- so
    this approximates.

    Body-centre velocities alone are not enough: a swinging arm's centre can be
    nearly still while its hand moves fast, and that is exactly the contact whose
    speed matters. So each body's velocity is evaluated away from its centre,
    toward the other body::

        v = v_com + omega x (p_eval - p_com)

    **The lever arm is capped, and that is not a tuning knob.** A rigid body's
    velocity field only describes the body; extrapolating it to a point metres
    away is not an approximation of anything. Uncapped, this returned a median of
    9 m/s and a maximum of 38 m/s on a robot walking at 1 m/s, purely from
    swinging limbs projected out to a midpoint several metres away. Capping at
    the largest body half-extent keeps the evaluation point on or near the body
    that owns the velocity field.

    Where the bodies actually touch, half the centre distance is smaller than the
    cap, so the evaluation point is the midpoint and the cap does nothing. It only
    bites when the two are far apart, which is where a "contact point" does not
    exist and the value is not used.

    ``*_vel`` are (N, 6), linear then angular, world frame -- the layout
    ``body_com_vel_w`` already has. Each dummy body is paired with the nearest
    robot body by centre distance: the pair that would touch.
    """
    dummy_pos = np.atleast_2d(np.asarray(dummy_pos, dtype=float))
    robot_pos = np.atleast_2d(np.asarray(robot_pos, dtype=float))
    dummy_vel = np.atleast_2d(np.asarray(dummy_vel, dtype=float))
    robot_vel = np.atleast_2d(np.asarray(robot_vel, dtype=float))
    if lever_cap_m <= 0 or not np.isfinite(lever_cap_m):
        raise ValueError(f"지렛대 상한이 유한한 양수가 아닙니다: {lever_cap_m!r}")

    separation = dummy_pos[:, None, :] - robot_pos[None, :, :]
    pair = np.argmin(np.linalg.norm(separation, axis=-1), axis=1)
    r_pos = robot_pos[pair]
    r_vel = robot_vel[pair]

    offset = dummy_pos - r_pos
    distance = np.linalg.norm(offset, axis=-1, keepdims=True)
    # Coincident centres leave the direction undefined; the lever is then zero
    # anyway, so any unit vector gives the same answer.
    direction = np.divide(offset, distance, out=np.zeros_like(offset), where=distance > 0)
    lever = np.minimum(distance / 2.0, lever_cap_m)

    v_robot = r_vel[:, :3] + np.cross(r_vel[:, 3:], direction * lever)
    v_dummy = dummy_vel[:, :3] + np.cross(dummy_vel[:, 3:], -direction * lever)
    return np.linalg.norm(v_robot - v_dummy, axis=-1)


def aggregate_contact_by_region(
    region_codes: Sequence[str], magnitudes: np.ndarray
) -> dict[str, float]:
    """Collapse per-prim contact magnitudes onto body-region codes.

    Paired regions contribute two prims under one code. Take the maximum,
    never the sum: summing reports a load no single limb took and trips a
    Table A.2 limit that was never reached.
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    if magnitudes.ndim != 1 or len(region_codes) != magnitudes.shape[0]:
        raise ValueError(
            f"region_codes({len(region_codes)}) 와 magnitudes({magnitudes.shape}) 의 길이가 다릅니다"
        )
    out: dict[str, float] = {}
    for code, value in zip(region_codes, magnitudes):
        value = float(value)
        out[code] = value if code not in out else max(out[code], value)
    return out


# The asset names both feet ``*_ankle_link`` and nothing else ends that way, so
# the suffix is the definition of "this body is a foot" for this robot. It lives
# here, in the one module with no Isaac dependency, because three places need it
# and they must agree: the recorder's contact gate, the dummy's hold-release
# backstop, and the substep accumulator's per-region robot-contact flag.
#
# They did not agree. The accumulator was added without the mask, so its flag was
# true on essentially every step -- the feet carry ground load continuously --
# and every region with any force got a contact type. Measured on the batch that
# ran with it: 41,990 typed steps across 52 episodes against 229 before, a factor
# of 180 that is not a physical change.
FOOT_LINK_SUFFIX = "ankle_link"


def non_foot_mask(body_names) -> np.ndarray:
    """True for bodies that are not feet, in the given order."""
    return np.array([not str(n).endswith(FOOT_LINK_SUFFIX) for n in body_names], dtype=bool)


GROUND_REGION_CODE = "BR-12"


def dummy_body_masks(codes, ground_region: str = GROUND_REGION_CODE):
    """(region_mask, ground_mask) over the dummy's contact-sensor bodies. A body with no region code (None)
    is a ground-contact body -- the jointed dummy's feet -- and those then carry the standing load; with no
    such body the ground region does (the rigid dummy stands on its shins)."""
    codes = list(codes)
    region = np.array([c is not None for c in codes])
    ground = ~region if (~region).any() else np.array([c == ground_region for c in codes])
    return region, ground


# What the per-step contact_onset cannot see, written into every episode's metadata. Which bodies form the
# dummy's ground channel decides it: the rigid dummy stands on its shins (BR-12), which the dummy-side channel
# therefore leaves out; the jointed dummy stands on its feet, outside the twelve regions.
_ONSET_GAP_SHINS = (
    "로봇 발이 더미 정강이(BR-12)를 치는 접촉은 잡지 못함. 더미 쪽 채널은 "
    "BR-12를 지면 접촉과 구분할 수 없어 구조적으로 제외했음. 로봇 쪽 채널의 "
    "발 외 콜라이더 부족은 h1.usd 교체(2026-08-20)로 해소됐으나, 발 자체가 "
    "무엇에 닿았는지는 여전히 부위로 분해되지 않음. BR-12의 지면 반력 기저선 "
    "초과분으로 탐지를 시도했으나, 로봇 접촉이 전혀 없는 에피소드에서도 "
    "그 기저선 자체가 0~12000N 대로 튀어(홀드 서보 진동, T7b) 신뢰할 수 있는 "
    "임계값이 없었음."
)
_ONSET_GAP_FEET = (
    "로봇 발이 더미의 발을 밟는 접촉은 잡지 못함. 더미의 발은 12부위 밖의 몸이고 서 있는 무게를 "
    "받아 더미 쪽 채널에서 지면 접촉으로 제외했으며, 로봇 쪽 채널은 로봇의 발을 제외함. 발에 닿은 "
    "힘은 부위별 접촉력에도 기록하지 않음. 정강이(BR-12)는 지면 하중을 받지 않으므로 더미 쪽 채널이 봄."
)


def contact_onset_gap(codes) -> tuple[list, str]:
    """(unresolved regions, reason) for the episode metadata, from the dummy's contact-sensor body codes."""
    region, _ = dummy_body_masks(codes)
    if region.all():
        return [GROUND_REGION_CODE], _ONSET_GAP_SHINS
    return [], _ONSET_GAP_FEET


def reduce_region_peaks(region_codes, force, z, robot_gate, *, contact_eps_n):
    """Collapse per-body substep peaks onto regions, keeping each peak's own state.

    ``aggregate_contact_by_region`` takes the maximum force and stops there. That
    is not enough to choose a Table A.2 column, because the column also depends
    on the region's height and on whether the robot was touching -- and if those
    are reduced separately they can come from different bodies and different
    substeps. A region could then be typed ``quasistatic`` because some other
    body of the pair was near the floor while the force that was actually
    measured came from a body standing upright.

    So the reduction is by argmax: the body carrying the region's peak force
    hands over its own height and the robot-contact gate as it stood at the
    substep that peak occurred in (see substep_contact.py).

    ``robot_gate`` is global, not per region: it says the robot was touching
    something that is not the ground, not that it was touching *this* region.
    Per-pair attribution is not available here (schema §8.1), so two regions
    peaking in the same substep get the same answer. Narrowing the time axis does
    not narrow the space axis, and the parameter is named gate so that reducing
    it per region is not mistaken for attributing it per region.

    When a region is not in contact at all the argmax is arbitrary, so the height
    falls back to the *lowest* body of the region -- the old reduction, and the
    one that answers "is any of this region on the floor" for a reader auditing a
    step where nothing was touching.

    Returns (force, z, robot_contact) dicts keyed by region code.
    """
    force = np.asarray(force, dtype=float)
    z = np.asarray(z, dtype=float)
    touching = np.broadcast_to(np.asarray(robot_gate, dtype=bool), force.shape)
    if not (len(region_codes) == force.shape[0] == z.shape[0]):
        raise ValueError(
            f"region_codes({len(region_codes)}) · force{force.shape} · z{z.shape} 의 길이가 다릅니다")

    picked: dict[str, int] = {}
    lowest: dict[str, float] = {}
    for i, code in enumerate(region_codes):
        if code not in picked or force[i] > force[picked[code]]:
            picked[code] = i
        lowest[code] = z[i] if code not in lowest else min(lowest[code], z[i])

    out_f, out_z, out_c = {}, {}, {}
    for code, i in picked.items():
        out_f[code] = float(force[i])
        out_z[code] = float(z[i]) if force[i] > contact_eps_n else float(lowest[code])
        out_c[code] = bool(touching[i])
    return out_f, out_z, out_c


CONTACT_TYPE_NONE = "none"
CONTACT_TYPE_TRANSIENT = "transient"
CONTACT_TYPE_QUASISTATIC = "quasistatic"


def classify_contact_type(
    contact_force,
    region_z,
    *,
    robot_contact,
    contact_eps_n: float,
    pin_height_m: float,
) -> list[str]:
    """Per-region contact type for one step: none / transient / quasistatic.

    The canon's distinction is not how long a contact lasts but whether the
    body can get away from it (dataset schema v0.1 §8.1, meta/rules.json
    contact_types): transient is a free collision the human can recede from,
    quasistatic is a clamped one they cannot. Table A.2 sets different limits
    for the two, so this field is what decides which limit a contact is
    measured against.

    Two things follow, and both are why this is not a duration rule.

    A dummy that has been knocked over and is lying loose is still free to be
    pushed, so its contacts stay transient however many steps they run for.
    The one clamped case S1a can produce is the robot pressing a region
    against the floor -- a fallen person being stood on -- and that is what
    `region_z <= pin_height_m` detects. Height, not speed: a free collision
    also passes through zero velocity at the moment it reverses, so a
    velocity test would need a duration threshold of its own and would still
    be answering "did it move" rather than the canon's "could it move".

    And when the robot is not touching the dummy at all, no region has a
    contact type. Force on a region then is the dummy's own weight on the
    ground, which is not a robot-human contact and must not be measured
    against a Table A.2 limit. `robot_contact` is the robot-side channel
    (a non-foot robot link carrying load), the same signal EpisodeRecorder
    uses for contact_onset.

    Args:
        contact_force: (n_region,) N, per-region contact magnitude.
        region_z: (n_region,) m, height of the region above the floor.
        robot_contact: whether the robot is touching something that is not
            the ground. A scalar applies to every region; an array lets the
            caller vary it, which the substep accumulator does by sampling the
            gate at the substep each region's force peaked -- so one substep's
            reading cannot type another substep's peak. It stays a *gate*, not
            an attribution: it never says which region the robot touched, and
            schema §8.1 records that limit.
        contact_eps_n: below this, a region counts as untouched.
        pin_height_m: at or below this height a loaded region counts as
            pressed against the floor rather than free.
    """
    force = np.asarray(contact_force, dtype=float)
    z = np.asarray(region_z, dtype=float)
    if force.shape != z.shape:
        raise ValueError(f"contact_force{force.shape} 와 region_z{z.shape} 의 길이가 다릅니다")
    if not np.isfinite(z).all():
        raise ValueError("region_z 에 유한하지 않은 값이 있습니다")
    if not np.isfinite(force).all():
        raise ValueError("contact_force 에 유한하지 않은 값이 있습니다")

    touching = np.broadcast_to(np.asarray(robot_contact, dtype=bool), force.shape)
    out = []
    for f, height, robot in zip(force, z, touching):
        if not robot or f <= contact_eps_n:
            out.append(CONTACT_TYPE_NONE)
        elif height <= pin_height_m:
            out.append(CONTACT_TYPE_QUASISTATIC)
        else:
            out.append(CONTACT_TYPE_TRANSIENT)
    return out


SSM_RULE_ID = "R-TS15066-SSM-01"

# The parameters the SSM distance is a function of. Named here rather than
# read off whatever keys the canon happens to carry: iterating the file's own
# keys meant a parameter deleted from meta/rules.json was not reported missing
# either, so the loader stayed silent about a value that was simply not there.
SSM_PARAMETER_NAMES = (
    "human_speed",
    "robot_reaction_time",
    "robot_stop_time",
    "intrusion_distance",
    "position_uncertainty_robot",
    "position_uncertainty_human",
)


def load_ssm_parameters(rules_path: str | Path) -> tuple[dict[str, float | None], list[str]]:
    """Read the SSM parameters from the rules canon.

    Returns the values and the names still unfilled. The standard's numbers
    live in meta/rules.json and nowhere else, so an empty copy of the
    standard shows up here as a list of names rather than as a default that
    silently produces a plausible-looking distance.
    """
    rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    entry = next((r for r in rules["rules"] if r["rule_id"] == SSM_RULE_ID), None)
    if entry is None:
        raise ValueError(f"{rules_path} 에 {SSM_RULE_ID} 가 없습니다")
    params = entry.get("parameters") or {}
    if not params:
        raise ValueError(f"{SSM_RULE_ID}.parameters 가 비어 있습니다 — 항목부터 만드십시오")

    values: dict[str, float | None] = {}
    missing: list[str] = []
    for name in SSM_PARAMETER_NAMES:
        item = params.get(name)
        value = item.get("value") if isinstance(item, dict) else None
        if not _is_usable_ssm_value(value):
            missing.append(name)
            values[name] = None
        else:
            values[name] = float(value)
    return values, missing


def fill_missing_only(loaded: dict, provisional: dict, missing: list[str]) -> dict:
    """Provisional values fill the gaps; a value the canon has always wins.

    `dict(loaded, **provisional)` reads like "fill in the blanks" and is not:
    it overwrites every key provisional carries. provisional_ssm_params carries
    all six, and something has always been missing, so measured values were
    being replaced by placeholders on every run -- robot_stop_time went back to
    0.5 s after we measured 0.31 s, and the canon and the generator disagreed
    with CI green either way (CONTRIBUTING.md 「함께 고치는 곳」).
    """
    out = dict(loaded)
    for name in missing:
        if name in provisional:
            out[name] = provisional[name]
    return out


def _is_usable_ssm_value(value: object) -> bool:
    """Whether an SSM parameter is filled in with a number the maths can use.

    None and non-numbers are the obvious cases. nan is the dangerous one: it
    is a float, so a bare isinstance check calls it filled, the separation
    distance comes out nan, and every `distance < required` comparison against
    a nan is False -- separation_violation never fires and the near_miss label
    leaves the dataset without a word anywhere. Hand-filling the standard's
    numbers is precisely when a placeholder like that can arrive. The same
    guard already protects the dummy's masses in dummy/build_dummy.py.

    A negative is refused for the opposite reason: it shrinks the required
    separation instead of growing it, letting the robot closer than the
    standard permits.

    Zero stays allowed, unlike in build_dummy.py: a mass or a spring constant
    of zero is not physical, but an intrusion distance or a position
    uncertainty of zero is an honest value for a simulated sensor.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) >= 0.0


def separation_required(
    params: dict[str, float | None], human_speed: float, robot_speed: float
) -> float:
    """SSM separation distance, in metres.

    The sum of how far the human travels while the robot reacts and stops,
    how far the robot travels over those same two intervals, the intrusion
    distance, and both position uncertainties. Every number comes from
    params — this function holds no value of its own.
    """
    # The two speeds arrive as arguments rather than through params, so
    # guarding params alone left the same fail-open open: one nan here and the
    # distance is nan, `distance < required` is False forever, and
    # separation_violation never fires.
    for name, value in (("human_speed", human_speed), ("robot_speed", robot_speed)):
        if not _is_usable_ssm_value(value):
            raise ValueError(f"{name} 가 쓸 수 없는 값입니다: {value!r}")

    for name in SSM_PARAMETER_NAMES:
        if not _is_usable_ssm_value(params.get(name)):
            # Not only the loader's door: this function is called with dicts
            # assembled elsewhere too, and it is the one that would turn a
            # single nan into a violation test that is False forever.
            raise ValueError(
                f"SSM 파라미터 {name} 가 비어 있거나 쓸 수 없습니다: {params.get(name)!r}"
            )

    reaction = float(params["robot_reaction_time"])
    stopping = float(params["robot_stop_time"])
    return (
        human_speed * (reaction + stopping)
        + robot_speed * reaction
        + robot_speed * stopping / 2.0      # linear decel over the stopping interval
        + float(params["intrusion_distance"])
        + float(params["position_uncertainty_robot"])
        + float(params["position_uncertainty_human"])
    )


def command_scale(
    distance: float,
    separation_required: float,
    slowdown_factor: float,
    stop_band_factor: float,
) -> float:
    """Command-velocity multiplier for the SSM slowdown ramp. Always in [0, 1].

    1.0 at distance >= slowdown_factor*separation_required (deceleration has not
    started yet), ramping linearly down to exactly 0.0 at
    distance <= stop_band_factor*separation_required (the stop band), linear
    in between.

    stop_band_factor is not separation_required itself (a factor of 1.0) — it
    is slightly larger (>1, e.g. 1.05), so the stop band sits just outside the
    bare separation distance rather than exactly on it. That margin is load-
    bearing, not decoration. A ramp whose command speed is proportional to the
    remaining gap above separation_required makes the closing rate proportional
    to that same gap, and a quantity that decays proportionally to itself only
    reaches zero in the limit, never in finitely many steps: iterating this
    ramp's discrete recurrence against a stationary target, the gap above
    separation_required was still ~1e-14 m after 5000 steps, decaying by
    roughly 1% per step — closing the last float32 ULP would take on the
    order of 1400 steps, longer than this env's 1000-step episode. A stop band
    ends the ramp with a margin still open, so the command reaches exactly 0.0
    in finitely many steps instead of approaching it forever.

    This is not evidence that the un-banded ramp is wrong for the scenario
    this command term exists to run: S1a has the *human* closing the gap, not
    a stationary dummy, so distance keeps falling on its own once the robot's
    commanded speed has already bottomed out near zero, and
    distance < separation_required is reached that way regardless of the
    band. A stationary target simply can never demonstrate a stop from the
    un-banded ramp alone — a proof limitation, not a correctness one. The
    stop band is worth keeping anyway: a real safety controller commits to a
    full stop with a margin before the boundary it protects, not
    asymptotically at it.

    stop_band_factor and slowdown_factor are both safety-controller tuning —
    they shape this ramp, not what ISO/TS 15066 requires the separation
    distance itself to be — so they belong wherever the command term is
    configured, not in meta/rules.json.

    Raises:
        ValueError: if separation_required is not positive, or stop_band_factor
            is not strictly less than slowdown_factor (the ramp needs a
            positive span to interpolate over), or stop_band_factor is not
            strictly greater than 1.0 (the stop band would then sit at or
            inside the bare separation distance, so the command could still
            be nonzero exactly where separation_violation is already true).
    """
    if separation_required <= 0.0:
        raise ValueError("separation_required 는 0보다 커야 합니다")
    if stop_band_factor <= 1.0:
        raise ValueError("stop_band_factor 는 1보다 커야 정지 구간이 요구 이격거리 밖에 놓입니다")
    if stop_band_factor >= slowdown_factor:
        raise ValueError("stop_band_factor 는 slowdown_factor 보다 작아야 램프 구간이 생깁니다")

    slowdown_start = separation_required * slowdown_factor
    stop_band = separation_required * stop_band_factor
    if distance >= slowdown_start:
        return 1.0
    if distance <= stop_band:
        return 0.0
    return (distance - stop_band) / (slowdown_start - stop_band)


def min_distance_human(
    robot_points: np.ndarray, human_points: np.ndarray, human_radii: np.ndarray
) -> float:
    """Closest approach between any robot link and any human region surface.

    Link origins against region centres, less the region radius. This is an
    approximation of surface-to-surface distance and it is the robot side
    that is unmodelled, so it reads slightly long. Clamped at zero: once the
    bodies overlap the quantity that matters is the contact force, not a
    negative distance.
    """
    robot_points = np.atleast_2d(np.asarray(robot_points, dtype=float))
    human_points = np.atleast_2d(np.asarray(human_points, dtype=float))
    human_radii = np.asarray(human_radii, dtype=float).reshape(-1)
    if human_points.shape[0] != human_radii.shape[0]:
        raise ValueError("human_points 와 human_radii 의 개수가 다릅니다")

    deltas = robot_points[:, None, :] - human_points[None, :, :]
    centre_distance = np.linalg.norm(deltas, axis=-1)          # (R, H)
    surface_distance = centre_distance - human_radii[None, :]
    return float(max(surface_distance.min(), 0.0))


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2-D cross product as a scalar.

    numpy removed the 2-D form of np.cross in 2.0 — it raises rather than
    warns — so the sign test the hull and the inside/outside check depend on
    has to be written out.
    """
    return float(a[0] * b[1] - a[1] * b[0])


def stability_margin(support_points_xy: np.ndarray, com_xy: np.ndarray) -> float:
    """Signed distance from the CoM ground projection to the support polygon.

    Positive inside, negative outside. Two branches never read positive, for
    two different reasons:

    - Zero support points (both feet unloaded at once -- the flight phase of
      dynamic gait) is not a degenerate polygon, it is no support at all.
      Returns exactly float("-inf"), not NaN and not an ordinary negative
      number -- the schema's documented sentinel for this case (dataset
      schema v0.1 §6, safety.stability_margin 산출 방법), so a flight-phase
      step is distinguishable downstream from a merely-outside-the-polygon
      one instead of collapsing into the same negative range.
    - Fewer than three (but at least one) contact points is not a polygon
      either: the result is the negated distance to the segment (or point),
      so a degenerate support can never read as stable.
    """
    # Empty is the one shape that gets normalised: np.atleast_2d would turn a
    # plain empty list into (1, 0), which reads as one point with no
    # coordinates and slips past the zero-points branch below.
    #
    # Everything else must already be (N, 2). An earlier reshape(-1, 2) here
    # accepted any size, which turned an ordinary mistake -- handing it XYZ
    # points -- into a silently different polygon: (2, 3) became three XY
    # points metres apart and reported a CoM inside that phantom triangle as
    # *stable*. A support polygon that is wrong cannot be told from one that
    # is right, so the shape is refused rather than guessed at.
    pts = np.asarray(support_points_xy, dtype=float)
    if pts.size == 0:
        pts = pts.reshape(0, 2)
    elif pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"support_points_xy 는 (N, 2) 여야 합니다 (받은 shape: {pts.shape}). "
            "XYZ 를 넘겼다면 xy 만 잘라서 넘기십시오"
        )
    # Shape alone was not enough. A NaN corner came back as 0.0 -- which reads
    # as "exactly on the boundary", not as unstable -- and an Inf corner as
    # nan, after which every `margin < 0` test is False and a flight phase
    # leaves the data instead of showing up as the -inf sentinel. Same
    # fail-open the SSM parameters were guarded against, one function over.
    if not np.isfinite(pts).all():
        raise ValueError("support_points_xy 에 유한하지 않은 값이 있습니다")
    com = np.asarray(com_xy, dtype=float).reshape(2)
    if not np.isfinite(com).all():
        raise ValueError(f"com_xy 에 유한하지 않은 값이 있습니다: {com_xy!r}")
    if pts.shape[0] == 0:
        return float("-inf")

    hull = _convex_hull(pts)
    if hull.shape[0] < 3:
        return -float(_distance_to_polyline(hull, com))

    edge_distances = []
    inside = True
    for i in range(hull.shape[0]):
        a, b = hull[i], hull[(i + 1) % hull.shape[0]]
        edge = b - a
        # hull is counter-clockwise, so a negative cross product means outside
        if _cross2(edge, com - a) < 0.0:
            inside = False
        edge_distances.append(_distance_to_segment(a, b, com))
    nearest = float(min(edge_distances))
    return nearest if inside else -nearest


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise hull, monotone chain. Few points, so O(n log n) is plenty."""
    pts = np.unique(points, axis=0)
    if pts.shape[0] < 3:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def build(seq):
        out: list[np.ndarray] = []
        for p in seq:
            while len(out) >= 2 and _cross2(out[-1] - out[-2], p - out[-2]) <= 0:
                out.pop()
            out.append(p)
        return out

    lower = build(pts)
    upper = build(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _distance_to_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    edge = b - a
    length_sq = float(edge @ edge)
    if length_sq == 0.0:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ edge / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * edge)))


def _distance_to_polyline(pts: np.ndarray, p: np.ndarray) -> float:
    if pts.shape[0] == 1:
        return float(np.linalg.norm(p - pts[0]))
    return min(_distance_to_segment(pts[i], pts[i + 1], p) for i in range(pts.shape[0] - 1))


# Below this fraction of the body set's own weight, the ZMP denominator
# (sum of m*(a_z + g)) carries no information: the set is in free fall and
# there is no ground reaction whose line of action a ZMP could name. Written
# as a fraction rather than an absolute newton figure so it scales with the
# robot instead of encoding H1's mass as a constant here.
_ZMP_DENOM_REL_EPS = 1e-2


def _finite(name: str, arr: np.ndarray) -> np.ndarray:
    """Reject a non-finite input outright.

    A NaN that came from a diverging simulation and a NaN that means "this
    quantity is not defined here" are different facts, and folding them into
    one value makes the second unreadable. The physical sentinel stays; the
    corrupt input raises.
    """
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} 에 비유한 값이 있습니다 — 입력이 손상됐습니다")
    return arr


def center_of_mass(masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Mass-weighted centre of mass of a rigid-body set, in the frame of ``positions``.

    Not the root link position. The schema calls safety.com the centre of
    mass, so it has to be one, and the old approximation also moved
    stability_margin by however far the two happened to be apart on a step.
    """
    m = np.asarray(masses, dtype=float).reshape(-1)
    p = np.asarray(positions, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"positions 는 (N, 3) 이어야 합니다: {p.shape}")
    if len(m) != len(p):
        raise ValueError(f"길이 불일치: masses {len(m)} vs positions {len(p)}")
    if len(m) == 0:
        raise ValueError("빈 바디 집합의 무게중심은 정의되지 않습니다")
    _finite("masses", m)
    _finite("positions", p)
    # Per body, not just the total: a negative mass cancelled by a larger
    # positive one used to pass the sum check and shift the CoM silently.
    if (m < 0).any():
        raise ValueError(f"음수 질량이 있습니다: {float(m.min())!r}")
    total = float(m.sum())
    if total <= 0.0:
        raise ValueError(f"질량 합이 사용할 수 없는 값입니다: {total!r}")
    return (m[:, None] * p).sum(axis=0) / total


def zero_moment_point(
    masses: np.ndarray,
    positions: np.ndarray,
    lin_acc: np.ndarray,
    ang_vel: np.ndarray,
    ang_acc: np.ndarray,
    inertias_w: np.ndarray,
    gravity: float,
) -> np.ndarray:
    """ZMP on the z=0 ground plane, from full multibody dynamics.

    The angular term is the full rate of change of each body's angular
    momentum, ``I·alpha + omega x (I·omega)``. The gyroscopic half is not
    optional: with an asymmetric link inertia it reaches 0.7-17 N*m over the
    rotation rates this scene produces, against a denominator near 500 N --
    millimetres to centimetres of ZMP, on a quantity whose whole job is to
    say where the ground reaction acts.

    ``inertias_w`` is already world-frame -- Isaac Lab's ``body_inertia`` is
    documented as such and arrives flattened (9,), so the caller reshapes to
    (N, 3, 3) and no body-to-world rotation happens here. The (N, 9) shape is
    rejected rather than reshaped: accepting it would let a caller who forgot
    the reshape get a plausible-looking wrong answer.

    ``gravity`` has no default and must be positive and finite. It belongs to
    the running simulation, and without gravity there is no ground reaction
    whose line of action a ZMP could name.

    Returns ``[nan, nan]`` only for the one physical case -- free fall, where
    the vertical term vanishes. Every corrupt input raises instead, so the
    sentinel keeps meaning one thing.
    """
    m = np.asarray(masses, dtype=float).reshape(-1)
    p = np.asarray(positions, dtype=float)
    a = np.asarray(lin_acc, dtype=float)
    omega = np.asarray(ang_vel, dtype=float)
    alpha = np.asarray(ang_acc, dtype=float)
    inert = np.asarray(inertias_w, dtype=float)
    for name, arr in (("positions", p), ("lin_acc", a), ("ang_vel", omega),
                      ("ang_acc", alpha)):
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{name} 는 (N, 3) 이어야 합니다: {arr.shape}")
    if inert.ndim != 3 or inert.shape[1:] != (3, 3):
        raise ValueError(f"inertias_w 는 (N, 3, 3) 이어야 합니다: {inert.shape}")
    if not (len(m) == len(p) == len(a) == len(omega) == len(alpha) == len(inert)):
        raise ValueError(
            "길이 불일치: "
            f"masses {len(m)} · positions {len(p)} · lin_acc {len(a)} · "
            f"ang_vel {len(omega)} · ang_acc {len(alpha)} · inertias_w {len(inert)}"
        )
    if len(m) == 0:
        raise ValueError("빈 바디 집합의 ZMP 는 정의되지 않습니다")
    if not math.isfinite(gravity) or gravity <= 0.0:
        raise ValueError(f"gravity 는 유한한 양수여야 합니다: {gravity!r}")
    for name, arr in (("masses", m), ("positions", p), ("lin_acc", a),
                      ("ang_vel", omega), ("ang_acc", alpha), ("inertias_w", inert)):
        _finite(name, arr)

    mz = m * (a[:, 2] + gravity)
    denom = float(mz.sum())
    weight = float(m.sum()) * gravity
    if abs(denom) < _ZMP_DENOM_REL_EPS * weight:
        return np.array([float("nan"), float("nan")])

    # dH/dt per body, world frame.
    torque = (np.einsum("nij,nj->ni", inert, alpha)
              + np.cross(omega, np.einsum("nij,nj->ni", inert, omega)))
    x = (float((mz * p[:, 0]).sum())
         - float((m * p[:, 2] * a[:, 0]).sum())
         - float(torque[:, 1].sum())) / denom
    y = (float((mz * p[:, 1]).sum())
         - float((m * p[:, 2] * a[:, 1]).sum())
         + float(torque[:, 0].sum())) / denom
    return np.array([x, y])
