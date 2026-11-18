"""The walking experiment's measures of the jointed dummy and the marks they have to make. Pure numpy.

The hold servo carries a share of the dummy's weight (commands.S1aStraightWalkCommand._support_force_z) and the feet
carry the rest. The marks on how the dummy walks -- what its feet carry, the assist across the plane and about the
tilt axes -- are taken over the walking steps: held, triggered, before the far-side stop. Standing steps would pull
those medians toward the standing values. Tilt and falls are taken over every held step; nothing after the release
is marked but the release's own counts. Two more measures of the walk are reported with the marks and mark nothing:
the walking steps with both feet off the floor and the largest slide of a foot that is down (a foot is down while it
carries more than LOADED_FRACTION of what the feet carry)."""
from __future__ import annotations

import numpy as np

GRF_BAND = 0.10              # |median foot ground reaction over body weight - (1 - share)|
FLIGHT_LOAD = 0.05           # reported: a foot under this fraction of the body weight is off the floor
HOLD_FRACTION_MAX = 0.30     # median horizontal assist force over body weight
TILT_TORQUE_MAX_NM = 150.0   # median assist torque about the two tilt axes (about body weight x 0.2 m)
TILT_MAX_DEG = 10.0          # torso tilt
LOADED_FRACTION = 0.2
HEAD_FALL_M = 1.0
TILT_FALL_DEG = 45.0
WALKING_MARKS = ("hold", "grf", "torque")
# A standing step's region heights against the dimensions, after the offset the whole body shares: wider than a
# standing dummy's sway, narrower than the closest two region centres (the test holds the second against the spec).
STANDING_TOL_M = 0.015
# The same check for a dummy that has walked its script and stopped. A dummy that stops can keep the leg of the stride
# its walk ended in: in 13 rows measured 2026-10-26 the largest residual after the stop (median offset removed) was
# 2.4-4.8 cm in eleven rows, 6.0 cm in one and 8.1 cm in one (a shank 6.3 cm high for all 479 steps after the stop,
# the person upright and untouched); before the script set off it was 0.0 cm in every one. So the tolerance is wider
# than the largest of those; self-set. It still tells a limb held further off and regions swapped more than its width
# apart. A region recorded at another body's height is told by the narrow tolerance on a row that has held steps
# before the script sets off; a row whose script sets off at the first step has none.
STOPPED_STANDING_TOL_M = 0.10


def _before_release(released) -> np.ndarray:
    r = np.asarray(released, bool)
    first = int(np.argmax(r)) if r.any() else len(r)
    out = np.zeros(len(r), bool)
    out[:first] = True
    return out


def _runs(mask) -> list[tuple[int, int]]:
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[::2], edges[1::2]))


def _median(values, mask) -> float:
    v = np.asarray(values, float)[mask]
    return float(np.median(v)) if v.size else float("nan")


def summarize(rows: dict, weight_n: float, dt: float, share: float = 0.0) -> dict:
    pre = _before_release(rows["released"])
    triggered = np.asarray(rows["triggered"], bool)
    ended = np.asarray(rows["walk_ended"], bool)
    walk = pre & triggered & ~ended
    hold = np.asarray(rows["hold_xy"], float)
    feet_load = np.asarray(rows["feet_load"], float)
    slips = []
    for k in range(feet_load.shape[1]):
        loaded = (feet_load[:, k] > LOADED_FRACTION * (1.0 - share) * weight_n) & pre
        for a, b in _runs(loaded):
            xy = np.asarray(rows["feet_pos"])[a:b, k, :2]
            slips.append(float(np.max(np.linalg.norm(xy - xy[0], axis=1))))
    tilt = np.asarray(rows["tilt"], float)[pre]
    head = np.asarray(rows["head_z"], float)[pre]
    speed = np.asarray(rows["gait_speed"], float)
    tilt_torque = np.linalg.norm(np.asarray(rows["hold_torque_vec"], float)[:, :2], axis=1)
    off = (feet_load < FLIGHT_LOAD * weight_n).all(axis=1)
    fell = bool((head < HEAD_FALL_M).any() or (tilt > TILT_FALL_DEG).any())
    return {"support_share": float(share), "steps_held": int(pre.sum()), "steps_walking": int(walk.sum()),
            "seconds_held": float(pre.sum() * dt),
            "hold_xy_walk_median_frac": _median(hold, walk) / weight_n,
            "grf_walk_median_frac": _median(rows["feet_force"], walk) / weight_n,
            "flight_frac": float(off[walk].mean()) if walk.any() else float("nan"),
            "tilt_torque_walk_median_nm": _median(tilt_torque, walk),
            "max_stance_slip_m": max(slips) if slips else 0.0,
            "max_tilt_deg": float(tilt.max()) if tilt.size else float("nan"),
            "fell": fell, "triggered": bool(triggered.any()), "released": bool(np.asarray(rows["released"], bool).any()),
            # the two realized speeds of the walk: over the whole walk (start to stop, as the recorder's window
            # reads it) and over its steady part (within 2 % of its top speed)
            "walk_mean_mps": float(np.mean(speed[walk])) if walk.any() else float("nan"),
            "walk_steady_mps": float(np.median(speed[walk][speed[walk] >= 0.98 * speed[walk].max()])) if walk.any() else float("nan"),
            "reached_far_side": bool((ended & pre).any()),
            # the record ran out mid-walk: a slow walk at a shallow angle takes longer than an episode to cross
            "walking_at_end": bool(walk[-1]) if walk.size else False,
            "contaminated_steps": int((np.asarray(rows["robot_contact"], bool) & (hold > 0.0)).sum()),
            "foot_robot_steps": int(np.asarray(rows["foot_robot"], bool).sum())}


def evaluate(s: dict) -> dict:
    """Each mark True or False; the walking marks None for a case that never walked, and the far-side stop None for a
    walk the record ended in (nothing to mark)."""
    walked = s["steps_walking"] > 0
    return {"hold": s["hold_xy_walk_median_frac"] < HOLD_FRACTION_MAX if walked else None,
            "grf": abs(s["grf_walk_median_frac"] - (1.0 - s["support_share"])) <= GRF_BAND if walked else None,
            "torque": s["tilt_torque_walk_median_nm"] < TILT_TORQUE_MAX_NM if walked else None,
            "tilt": s["max_tilt_deg"] < TILT_MAX_DEG,
            "upright": not s["fell"],
            "far_side": None if s["walking_at_end"] else (not s["triggered"]) or s["released"] or s["reached_far_side"],
            "contamination": s["contaminated_steps"] <= 1}


def standing_region_heights(anthropometry: dict) -> dict[str, float]:
    """Each body region's centre height in the standing pose the dimensions are written in (above the sole)."""
    return {code: float(region["center"][2]) for code, region in anthropometry["regions"].items()}


def standing_height_problems(codes, heights, expected: dict, tol: float = STANDING_TOL_M) -> list[str]:
    """What is wrong with one standing step's region heights, region by region; empty when nothing is.

    The whole body may stand higher or lower than the dimensions say (the feet, the settling), so the offset every
    region shares -- the median of recorded minus expected -- comes off first; then each region has to be within
    ``tol`` of its own height. A region recorded at another body's height is off by the gap between the two."""
    z = {str(c): float(h) for c, h in zip(codes, heights)}
    problems = [f"{c}: 높이가 기록되지 않음" for c in sorted(expected) if c not in z]
    problems += [f"{c}: 높이가 수가 아님 ({h})" for c, h in sorted(z.items()) if not np.isfinite(h)]
    usable = [c for c in sorted(z) if c in expected and np.isfinite(z[c])]
    if not usable:
        return problems
    offset = float(np.median([z[c] - expected[c] for c in usable]))
    for c in usable:
        residual = z[c] - offset - expected[c]
        if abs(residual) > tol:
            problems.append(f"{c}: 서 있는 높이 {z[c]:.3f} m — 명세 {expected[c]:.3f} m 에 공통 오프셋 {offset:+.3f} m 를 더한 것과 "
                            f"{residual:+.3f} m 다름")
    return problems
