"""100STYLE's skeleton onto the jointed dummy's degrees of freedom. Pure numpy.

Legs: the actor and the dummy both rest standing with straight legs and flat feet, and a BVH's rest
rotations are the identity, so a hip is the thigh's rotation in the pelvis's heading frame -- the pelvis with
its tilt (pitch and roll) folded into the hips, because the dummy's pelvis stays upright and its legs are to
keep their orientation to the floor (the pelvis's turn about the vertical stays out, as its twist does) -- and
an ankle the foot's in the dummy's own shank frame (the foot keeps its orientation to the floor) -- its swing
only: the ankle's third axis, along the shank, is held at zero, so the sole's normal is kept and the foot's
twist about the shank is dropped (the hip's turn about the leg carries the foot's heading). The knee is the
flexion angle of the shank in the thigh's sagittal plane. Arms: the actor rests in a T-pose and the dummy
with its arms down, so a shoulder is rebuilt from directions -- the upper arm's direction and the plane
the elbow bends in -- and the elbow is the angle between upper arm and forearm. Everything is expressed in
the dummy's body axes through C (cycles.body_axes): the actor's chest stands in for the dummy's rigid torso."""
from __future__ import annotations

import math

import numpy as np

from generation.dummy import joints_spec as js
from generation.dummy.gait import kinematics as kin

SIDES = ("left", "right")
BVH = {"pelvis": "Hips", "chest": "Chest4"}
CHILD = {}
for _side, _S in (("left", "Left"), ("right", "Right")):
    BVH[("thigh", _side)] = f"{_S}Hip"
    BVH[("shank", _side)] = f"{_S}Knee"
    BVH[("foot", _side)] = f"{_S}Ankle"
    BVH[("upper_arm", _side)] = f"{_S}Shoulder"
    BVH[("forearm", _side)] = f"{_S}Elbow"
    CHILD[("upper_arm", _side)] = f"{_S}Elbow"
    CHILD[("forearm", _side)] = f"{_S}Wrist"
STRAIGHT = math.sin(math.radians(10.0))     # below this, the elbow's bending plane is not defined


def _global(rot, sk, key):
    return rot[:, sk.index(BVH[key])]


def _relative(rot, sk, C, parent, child):
    R = np.swapaxes(_global(rot, sk, parent), -1, -2) @ _global(rot, sk, child)
    return C @ R @ C.T


def _arm(rot, sk, C, side):
    Gc_T = np.swapaxes(_global(rot, sk, "chest"), -1, -2)

    def direction(seg):
        child = sk.offsets[sk.index(CHILD[(seg, side)])]
        d = np.einsum("fij,fjk,k->fi", Gc_T, _global(rot, sk, (seg, side)), child / np.linalg.norm(child))
        return d @ C.T

    u, w = direction("upper_arm"), direction("forearm")
    S = np.zeros((len(u), 3, 3))
    elbow = np.zeros(len(u))
    prev = None
    for f in range(len(u)):
        z = -u[f] / np.linalg.norm(u[f])
        n = np.cross(w[f], u[f])
        if np.linalg.norm(n) > STRAIGHT * np.linalg.norm(w[f]) * np.linalg.norm(u[f]):
            y = n / np.linalg.norm(n)
        elif prev is not None:
            y = prev
        else:
            y = kin.rotation_between(np.array([0.0, 0.0, -1.0]), u[f]) @ np.array([0.0, 1.0, 0.0])
        y = y - np.dot(y, z) * z
        y = y / np.linalg.norm(y)
        S[f] = np.stack([np.cross(y, z), y, z], axis=1)
        prev = y
        elbow[f] = math.atan2(float(np.dot(np.cross(u[f], w[f]), y)), float(np.dot(u[f], w[f])))
    return S, elbow


def _ankle_dofs(R, spec: js.JointedSpec, ankle: js.JointDef) -> np.ndarray:
    """An ankle's positions for the foot's rotation R in the shank frame: the swing about the shank axis, its
    z exactly 0."""
    if spec.d6_convention != "rotvec":
        raise NotImplementedError("the ankle's swing is written as a rotation vector")
    return kin.swing_rotvec(R)[..., ["xyz".index(a) for a in js.axes(ankle)]]


def pelvis_tilt(rot, sk, C) -> np.ndarray:
    """The pelvis's tilt per frame: its rotation in body axes with its heading (its front axis on the floor) taken
    out, so T = Ry(pitch) @ Rx(roll) -- what an upright pelvis must fold into the hips."""
    P = C @ _global(rot, sk, "pelvis") @ C.T
    heading = np.arctan2(P[:, 1, 0], P[:, 0, 0])
    return np.swapaxes(kin.axis_rotation("z", heading), -1, -2) @ P


def frame_dofs(rot, sk, C, spec: js.JointedSpec) -> np.ndarray:
    per = {}
    T = pelvis_tilt(rot, sk, C)
    for side in SIDES:
        H = _relative(rot, sk, C, "pelvis", ("thigh", side))
        K = _relative(rot, sk, C, ("thigh", side), ("shank", side))
        knee = np.arctan2(K[:, 0, 2], K[:, 2, 2])
        foot = _relative(rot, sk, C, "pelvis", ("foot", side))
        shank = H @ kin.axis_rotation("y", knee)
        per[f"hip_{side}"] = T @ H                  # the ankle (shank -> foot) is the same in either frame
        per[f"knee_{side}"] = knee
        per[f"ankle_{side}"] = np.swapaxes(shank, -1, -2) @ foot
        per[f"shoulder_{side}"], per[f"elbow_{side}"] = _arm(rot, sk, C, side)
    cols = []
    for j in spec.joints:
        v = per[j.name]
        if j.kind == "revolute":
            cols.append(v[:, None])
        elif j.group == "ankle":
            cols.append(_ankle_dofs(v, spec, j))
        else:
            cols.append(kin.matrix_to_dofs(v, js.axes(j), spec.d6_convention))
    return np.concatenate(cols, axis=1)


CUT_TOLERANCE = 1e-9      # rad; a mean of samples pinned at a bound can sit an ulp outside it -- not a cut


def clamp(dofs, spec: js.JointedSpec):
    """Every position into its joint's range; counts the samples that were outside it by more than float noise."""
    out = np.array(dofs, float, copy=True)
    cut = {}
    cols = js.columns(spec)
    for j in spec.joints:
        for k, d in enumerate(j.dofs):
            c = cols[j.name].start + k
            n = int(np.count_nonzero((out[:, c] < d.lower - CUT_TOLERANCE) | (out[:, c] > d.upper + CUT_TOLERANCE)))
            if n:
                label = j.name if j.kind == "revolute" else f"{j.name}.{d.axis}"
                cut[label] = n
            out[:, c] = np.clip(out[:, c], d.lower, d.upper)
    return out, cut


def leg_matrices(row, spec: js.JointedSpec, side: str):
    cols = js.columns(spec)
    hip, ank = js.joint(spec, f"hip_{side}"), js.joint(spec, f"ankle_{side}")
    return (kin.dofs_to_matrix(row[cols[hip.name]], js.axes(hip), spec.d6_convention),
            float(row[cols[f"knee_{side}"]][0]),
            kin.dofs_to_matrix(row[cols[ank.name]], js.axes(ank), spec.d6_convention))


def _sole_low(row, spec, side, geom) -> float:
    _, sole, _ = kin.leg_fk(geom, *leg_matrices(row, spec, side))
    return float(sole[:, 2].min())


def _move_ankle(row, spec, side, geom, step):
    """`row` with that side's ankle moved by `step` (pelvis frame) by the leg IK: the knee from the hip-to-ankle
    distance, the hip turned onto the new ankle, the foot keeping its orientation (the ankle cannot twist about the
    shank). Where the leg is too short the knee stays straight and the leg is aimed at the target. Returns (row, the
    knee's change in rad, whether the leg reached)."""
    R_h, kn, R_a = leg_matrices(row, spec, side)
    a0, _, R_foot = kin.leg_fk(geom, R_h, kn, R_a)
    target = a0 + step
    L1 = np.linalg.norm(geom.knee - geom.hip)
    L2 = np.linalg.norm(geom.ankle - geom.knee)
    d = np.linalg.norm(target - geom.hip)
    cos_knee = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    kn1 = float(np.arccos(np.clip(cos_knee, -1.0, 1.0)))
    a1, _, _ = kin.leg_fk(geom, R_h, kn1, R_a)
    R_h1 = kin.rotation_between(a1 - geom.hip, target - geom.hip) @ R_h
    R_a1 = (R_h1 @ kin.axis_rotation("y", kn1)).T @ R_foot
    cols = js.columns(spec)
    hip, ank = js.joint(spec, f"hip_{side}"), js.joint(spec, f"ankle_{side}")
    new = np.array(row, float, copy=True)
    new[cols[hip.name]] = kin.matrix_to_dofs(R_h1, js.axes(hip), spec.d6_convention)
    new[cols[f"knee_{side}"]] = kn1
    new[cols[ank.name]] = _ankle_dofs(R_a1, spec, ank)
    return new, abs(kn1 - kn), bool(cos_knee <= 1.0)


def _raise_ankle(row, spec, side, geom, dz):
    new, change, _ = _move_ankle(row, spec, side, geom, np.array([0.0, 0.0, dz]))
    return new, change


def pelvis_path(cycle, spec: js.JointedSpec) -> np.ndarray:
    """Per sample, how far above its stand height the pelvis rides when the lower of the two soles is on the floor:
    minus the lowest sole corner of the two feet with the pelvis at the stand height. Nothing holds the pelvis's height
    but the legs, so this is the path the joint angles push the body along."""
    geoms = {s: js.leg_geometry(spec, s) for s in SIDES}
    return np.array([-min(_sole_low(r, spec, s, geoms[s]) for s in SIDES) for r in np.asarray(cycle, float)])


def smooth_below(path, bend: float, *, rho: float = 1e3, iters: int = 20000, tol: float = 1e-11) -> np.ndarray:
    """The periodic `path` lowered, never raised, as little as it can be (least squares) for no sample's second
    difference to exceed `bend` in size. Solved by ADMM on the two constraints -- the bend through the second-difference
    operator (circulant, so the linear step is one FFT), the floor through `path` itself -- then held under `path`."""
    target = np.asarray(path, float)
    n = len(target)

    def second(x):
        return np.roll(x, 1) - 2.0 * x + np.roll(x, -1)

    eig = -4.0 * np.sin(np.pi * np.arange(n) / n) ** 2              # eigenvalues of the periodic second difference
    den = 1.0 + rho * eig ** 2 + rho
    x = target.copy()
    z, w = np.clip(second(x), -bend, bend), x.copy()
    u, v = np.zeros(n), np.zeros(n)
    for _ in range(iters):
        x = np.real(np.fft.ifft(np.fft.fft(target + rho * second(z - u) + rho * (w - v)) / den))
        dx = second(x)
        z = np.clip(dx + u, -bend, bend)
        w = np.minimum(x + v, target)
        u += dx - z
        v += x - w
        if max(np.max(np.abs(dx - z)), np.max(np.abs(x - w))) < tol:
            break
    out = np.minimum(x, target)
    if np.max(np.abs(second(out))) > bend * (1.0 + 1e-3) + tol:
        raise ValueError(f"the path could not be brought under the bend {bend:g}")
    return out


def _in_ranges(row, spec: js.JointedSpec) -> bool:
    return not clamp(np.asarray(row, float)[None], spec)[1]


def lower_pelvis(cycle, spec: js.JointedSpec, drop):
    """Each sample's pelvis `drop` metres lower over the floor: both ankles raised by the same height (_raise_ankle --
    the knee flexes, the foot keeps its orientation), so the feet keep their places relative to each other. Where the
    joint ranges stop a leg short of the full drop, both legs go only as far as the ranges allow. Returns (cycle,
    the number of samples the ranges limited)."""
    geoms = {s: js.leg_geometry(spec, s) for s in SIDES}
    out = np.array(cycle, float, copy=True)
    limited = 0

    def raised(row, d):
        for side in SIDES:
            row, _ = _raise_ankle(row, spec, side, geoms[side], d)
        return row

    for i, d in enumerate(np.asarray(drop, float)):
        if d <= 0.0:
            continue
        new = raised(out[i], d)
        if not _in_ranges(new, spec):
            limited += 1
            lo, hi = 0.0, d
            for _ in range(20):
                mid = (lo + hi) / 2.0
                lo, hi = (mid, hi) if _in_ranges(raised(out[i], mid), spec) else (lo, mid)
            new = raised(out[i], lo)
        out[i] = new
    return out, limited


def correct_stance(cycle, stance, spec: js.JointedSpec, passes: int = 3, *, blend: int, keep_ranges: bool = False,
                   held: dict | None = None):
    """In double support, flex the leg that would hold the pelvis higher until both soles reach the floor at one
    pelvis height. The ankle raise this gives a leg is carried into the `blend` phases on each side of every
    double-support run where that foot is still down, falling in equal steps to zero, so neither its knee nor the
    pelvis jumps where the other foot lands or lifts. (A foot in the air is the swing lift's to place.) With
    `keep_ranges` a flexed leg stays in its joints' ranges: where keeping its foot's orientation would take a joint
    past one, the joint is held there (the foot turns with the leg) and the flex goes on until the sole reaches the
    floor (_lift_to); `held`, if given, counts those raises per degree of freedom. Returns (corrected, pelvis shift
    per phase, largest knee change in rad)."""
    geoms = {s: js.leg_geometry(spec, s) for s in SIDES}
    out = np.array(cycle, float, copy=True)
    P = len(out)
    raised = np.zeros((P, len(SIDES)))
    change = 0.0

    def count(names, s):
        if held is not None:
            for name in names:
                if name.split(".")[0] in (f"hip_{s}", f"knee_{s}", f"ankle_{s}"):
                    held[name] = held.get(name, 0) + 1

    def raise_ankle(row, s, dz):                        # by dz, in range if asked (an eased raise has no target)
        new, dk = _raise_ankle(row, spec, s, geoms[s], dz)
        if not keep_ranges:
            return new, dk
        kept, cut = clamp(new[None], spec)
        count(cut, s)
        return kept[0], dk

    def raise_to(row, s, gap, top):                     # until the sole is at the floor `top`; the ankle's rise
        if not keep_ranges:
            new, dk = _raise_ankle(row, spec, s, geoms[s], gap)
            return new, dk, gap
        z0 = kin.leg_fk(geoms[s], *leg_matrices(row, spec, s))[0][2]
        new, at_range = _lift_to(row, spec, s, geoms[s], top, 0.0)
        count(at_range, s)
        k = js.columns(spec)[f"knee_{s}"].start
        return new, abs(float(new[k]) - float(row[k])), float(kin.leg_fk(geoms[s], *leg_matrices(new, spec, s))[0][2] - z0)

    for p in range(P):
        on = [s for k, s in enumerate(SIDES) if stance[p, k]]
        for _ in range(passes):
            lows = {s: _sole_low(out[p], spec, s, geoms[s]) for s in SIDES}
            top = max(lows[s] for s in on) if on else min(lows.values())
            gaps = {s: top - lows[s] for s in on if top - lows[s] > 1e-5}
            if not gaps:
                break
            for s, gap in gaps.items():
                out[p], dk, rise = raise_to(out[p], s, gap, top)
                raised[p, SIDES.index(s)] += rise
                change = max(change, dk)
    both = np.asarray(stance, bool).all(axis=1)
    if blend > 0 and both.any() and not both.all():
        ease = np.zeros_like(raised)
        for a, b in _runs(both):
            for j in range(1, blend + 1):
                w = 1.0 - j / (blend + 1)
                ease[(b + j) % P] = np.maximum(ease[(b + j) % P], w * raised[b % P])
                ease[(a - j) % P] = np.maximum(ease[(a - j) % P], w * raised[a % P])
        for p in np.flatnonzero(~both):
            for k, s in enumerate(SIDES):
                if stance[p, k] and ease[p, k] > 1e-9:
                    out[p], dk = raise_ankle(out[p], s, float(ease[p, k]))
                    change = max(change, dk)
    shift = np.zeros(P)
    for p in range(P):
        on = [s for k, s in enumerate(SIDES) if stance[p, k]]
        lows = {s: _sole_low(out[p], spec, s, geoms[s]) for s in SIDES}
        shift[p] = -(max(lows[s] for s in on) if on else min(lows.values()))
    return out, shift, change


def swing_requirement(stance, clearance: float, ramp: float) -> np.ndarray:
    """Per phase and leg, how high a swing sole's lowest corner must be: `clearance` in mid-swing, reached in equal
    steps over the first and last `ramp` of each swing run (at least one phase each) so that a foot is neither
    snapped up as it leaves the floor nor dropped as it lands. NaN where the foot is down."""
    st = np.asarray(stance, bool)
    P = len(st)
    out = np.full((P, len(SIDES)), np.nan)
    for k in range(len(SIDES)):
        for a, b in _runs(~st[:, k]):
            n = b - a + 1
            m = max(1, int(round(ramp * n)))
            i = np.arange(n)
            out[np.arange(a, b + 1) % P, k] = clearance * np.minimum(1.0, np.minimum(i + 1, n - i) / (m + 1))
    return out


def _lift_to(row, spec, side, geom, target: float, shift: float):
    """`row` with that side's sole lifted until its lowest corner (with the pelvis shift) is at `target`, every joint
    in its range: ankle raises by the remaining gap while keeping the foot's orientation stays in range (the ankle
    cannot twist about the shank, so on a turned leg one raise can fall a little short); otherwise the smallest
    raise that reaches the target once the joints are held at their ranges (found by bisection). Returns (row,
    labels of the degrees of freedom held at a range)."""
    def at(dz):
        r, _ = _raise_ankle(row, spec, side, geom, dz)
        kept, cut = clamp(r[None], spec)
        return kept[0], set(cut), _sole_low(kept[0], spec, side, geom) + shift

    cur = row
    for _ in range(8):
        gap = target - (_sole_low(cur, spec, side, geom) + shift)
        if gap <= 1e-9:
            return cur, set()
        r, _ = _raise_ankle(cur, spec, side, geom, gap)
        kept, cut = clamp(r[None], spec)
        if cut:
            break
        cur = kept[0]
    else:
        return cur, set()
    hi = target - (_sole_low(row, spec, side, geom) + shift)
    r_hi, c_hi, l_hi = at(hi)
    lo = 0.0
    for _ in range(40):
        if l_hi >= target:
            break
        lo, hi = hi, 1.5 * hi
        r_hi, c_hi, l_hi = at(hi)
    for _ in range(60):
        if hi - lo < 1e-9:
            break
        mid = 0.5 * (lo + hi)
        r_m, c_m, l_m = at(mid)
        if l_m >= target:
            hi, r_hi, c_hi, l_hi = mid, r_m, c_m, l_m
        else:
            lo = mid
    return r_hi, c_hi


def lift_swing(cycle, stance, shift, spec: js.JointedSpec, clearance, *, slope: float = 0.0, held: dict | None = None):
    """Where a swing sole's lowest corner (with the pelvis shift) is below `clearance` -- one height, or one per
    phase and leg as swing_requirement gives (NaN: none) -- raise that ankle by the shortfall: the leg IK flexes the
    knee just enough and the foot keeps its orientation. With a `slope` (metres per phase), the shortfall where the
    full clearance is asked (mid-swing) is also carried into the neighbouring swing phases, falling by at most the
    slope per phase, so the lifted sole -- and the knee that lifts it -- does not jump from one phase to the next.
    Every joint stays in its range: where keeping the foot's orientation would take a joint past it, the joint is
    held at its range and the lift goes on (the knee flexes further) until the sole clears; `held`, if given, counts
    those phases per degree of freedom. Stance legs and swing feet that already clear are left as they are.
    Returns (lifted cycle, (P, 2) mask of the lifted phases, largest knee change in rad)."""
    geoms = {s: js.leg_geometry(spec, s) for s in SIDES}
    cols = js.columns(spec)
    out = np.array(cycle, float, copy=True)
    P = len(out)
    need = np.array(np.broadcast_to(np.asarray(clearance, float), (P, len(SIDES))), copy=True)
    if slope > 0.0:
        lows = swing_lows(out, stance, shift, spec)
        for k in range(len(SIDES)):
            for a, b in _runs(~np.asarray(stance[:, k], bool)):
                idx = np.arange(a, b + 1) % P
                n = need[idx, k]
                if not np.isfinite(n).any():
                    continue
                full = np.isfinite(n) & (n >= np.nanmax(n) - 1e-12)
                src = np.where(full, n - lows[idx, k], -np.inf)
                pos = np.arange(len(idx))
                carried = np.max(src[None, :] - slope * np.abs(pos[:, None] - pos[None, :]), axis=1)
                need[idx, k] = np.where(np.isfinite(n), np.maximum(n, lows[idx, k] + carried), n)
    lifted = np.zeros((P, len(SIDES)), bool)
    change = 0.0
    for p in range(len(out)):
        for k, s in enumerate(SIDES):
            if stance[p, k] or not np.isfinite(need[p, k]):
                continue
            knee = float(out[p, cols[f"knee_{s}"]][0])
            if float(need[p, k]) - (_sole_low(out[p], spec, s, geoms[s]) + shift[p]) <= 1e-6:
                continue
            out[p], at_range = _lift_to(out[p], spec, s, geoms[s], float(need[p, k]), float(shift[p]))
            lifted[p, k] = True
            if held is not None:
                for name in at_range:
                    held[name] = held.get(name, 0) + 1
            if lifted[p, k]:
                change = max(change, abs(float(out[p, cols[f"knee_{s}"]][0]) - knee))
    return out, lifted, change


def scale_hip_pitch(cycle, s: float, spec: js.JointedSpec):
    """Each hip's fore-aft swing (its y, about the lateral axis) widened or narrowed by s about its cycle mean;
    every other column as it was."""
    out = np.array(cycle, float, copy=True)
    cols = js.columns(spec)
    for side in SIDES:
        hip = js.joint(spec, f"hip_{side}")
        c = cols[hip.name].start + js.axes(hip).index("y")
        mean = out[:, c].mean()
        out[:, c] = mean + s * (out[:, c] - mean)
    return out


def _runs(mask) -> list[tuple[int, int]]:
    """Inclusive (start, end) runs of True in a cyclic mask; a run through the end continues past it."""
    m = np.asarray(mask, bool)
    P = len(m)
    if m.all():
        return [(0, P - 1)]
    k = int(np.argmin(m))                      # start after a False so no run is cut in two
    out, i = [], 0
    while i < P:
        j = (k + i) % P
        if m[j]:
            s = i
            while i < P and m[(k + i) % P]:
                i += 1
            out.append((k + s, k + i - 1))
        else:
            i += 1
    return out


def stance_advance(cycle, stance, spec: js.JointedSpec) -> np.ndarray:
    """Per phase step (p -> p+1) and leg, how far the pelvis goes forward if that leg's foot, down at both phases,
    rolls without slipping on its lowest sole corner at p -- the heel, the flat sole, then the toe, as the actor's
    foot does: the backward travel of that corner in the pelvis frame. (Keeping the ankle still instead loses the
    distance the ankle itself travels while the foot rolls -- about a foot's length per step.) (P, 2); NaN where the
    foot is not down at both phases."""
    P = len(cycle)
    st = np.asarray(stance, bool)
    out = np.full((P, len(SIDES)), np.nan)
    for k, s in enumerate(SIDES):
        geom = js.leg_geometry(spec, s)
        soles = {}

        def sole(p):
            if p not in soles:
                soles[p] = kin.leg_fk(geom, *leg_matrices(cycle[p], spec, s))[1]
            return soles[p]

        for p in range(P):
            q = (p + 1) % P
            if st[p, k] and st[q, k]:
                c = int(np.argmin(sole(p)[:, 2]))
                out[p, k] = -float(sole(q)[c, 0] - sole(p)[c, 0])
    return out


def body_advance(per_leg, floor_share: float):
    """The body's advance per phase step from its stance feet's (stance_advance): the foot that is down alone; in double
    support the foot that landed last -- its run of steps began later (the cycle wraps round), as the body goes over
    the foot it has just put down, and the other foot, about to lift, is fitted to it (lock_stance); feet whose runs
    began together give their mean. A step with no foot down takes the advance between its neighbours'. No step falls
    below `floor_share` of the cycle's mean step -- the body never runs backward; where the rolling foot would carry it
    so, that foot would slip (lock_stance fits its leg instead). Returns (advance, steps raised to the floor, steps
    with no foot down)."""
    a = np.asarray(per_leg, float)
    P = len(a)
    down = np.isfinite(a)
    since = np.full(a.shape, np.inf)                 # steps since that foot's run began
    for k in range(a.shape[1]):
        for s, e in _runs(down[:, k]):
            since[np.arange(s, e + 1) % P, k] = np.arange(e - s + 1)
    later = np.where(since[:, 0] < since[:, 1], a[:, 0], np.where(since[:, 1] < since[:, 0], a[:, 1], a.mean(axis=1)))
    adv = np.where(down.all(axis=1), later, np.where(down[:, 0], a[:, 0], a[:, 1]))
    gap = ~down.any(axis=1)
    if gap.all():
        raise ValueError("발이 닿은 위상이 없는 주기입니다")
    if gap.any():
        i = np.arange(P)
        ok = i[~gap]
        adv[gap] = np.interp(i[gap], np.r_[ok - P, ok, ok + P], np.r_[adv[ok], adv[ok], adv[ok]])
    mean = float(adv.mean())
    if mean <= 0.0:
        raise ValueError("몸이 앞으로 가지 않는 주기입니다")
    low = adv < floor_share * mean
    return np.where(low, floor_share * mean, adv), int(np.count_nonzero(low)), int(np.count_nonzero(gap))


def cycle_speed(cycle, stance, duration_s: float, spec: js.JointedSpec, *, floor_share: float):
    """Pelvis speed over the cycle and its stride (metres per cycle): the body's advance (body_advance of the stance
    feet rolling without slipping) summed over the cycle."""
    adv, _, _ = body_advance(stance_advance(cycle, stance, spec), floor_share)
    stride = float(adv.sum())
    return stride / duration_s, stride


def retime(cycle, stance, advance, *rest):
    """The cycle resampled so that sample i sits at the share i / P of the body's advance over it (body_advance), so each
    phase sample carries the body the same distance -- at a steady pace with the phase advanced by distance the stance
    foot then rolls without slipping. The joints go linearly between the two original samples around a new one (the
    cycle wraps round), the stance comes from the nearer one, and each of `rest` (per-phase arrays such as the pelvis
    shift) goes as the joints. Sample 0 stays where it is. Returns (cycle, stance, *rest)."""
    adv = np.asarray(advance, float)
    if np.any(adv <= 0.0):
        raise ValueError("나아감이 0 이하인 위상 걸음이 있습니다")
    P = len(adv)
    D = np.concatenate([[0.0], np.cumsum(adv)])
    want = np.arange(P) * D[-1] / P
    p = np.clip(np.searchsorted(D, want, side="right") - 1, 0, P - 1)
    f = (want - D[p]) / adv[p]
    q = (p + 1) % P

    def at(values):
        v = np.asarray(values, float)
        w = f.reshape((-1,) + (1,) * (v.ndim - 1))
        return (1.0 - w) * v[p] + w * v[q]

    return (at(cycle), np.asarray(stance, bool)[np.where(f < 0.5, p, q)], *(at(r) for r in rest))


def lock_stance(cycle, stance, stride: float, spec: js.JointedSpec, passes: int = 8, *, held: dict | None = None):
    """Each stance foot held where it landed, at a steady pace with the phase advanced by distance (stride / P a
    phase): through each stance run, from its landing sample (left as it is), the foot is moved along the pelvis
    frame's x by the leg IK (_move_ankle -- it keeps its orientation and its height in the pelvis frame) until the sole
    corner it rolls on, its lowest a phase before, is one pace behind where it was then -- so every stance step
    advances the body by the pace (stance_advance) and nothing slides. What is left of the move where the foot lifts is
    taken back over the swing that follows, in equal steps, to nothing at the next landing. Only the moved leg's
    columns change; a swing foot is not lifted (lift_swing's work) nor the pelvis height set (correct_stance's). Where a
    stance leg is too short the IK keeps it straight and aimed, so the foot keeps its place along the floor but rises.
    Every moved leg stays in its joints' ranges: where keeping the foot's orientation would take a joint past one, it
    is held there (the foot turns with the leg) and the move goes on until the corner is where it should be; `held`,
    if given, counts those samples per degree of freedom. Returns (cycle, (P, 2) how far each foot was moved along x
    in metres, stance samples where a leg could not reach)."""
    out = np.array(cycle, float, copy=True)
    P = len(out)
    st = np.asarray(stance, bool)
    pace = stride / P
    cols = js.columns(spec)
    moved = np.zeros((P, len(SIDES)))
    short = 0
    for k, s in enumerate(SIDES):
        geom = js.leg_geometry(spec, s)
        joints = [j for j in spec.joints if j.side == s and j.group in ("hip", "knee", "ankle")]
        legs = [c for j in joints for c in range(cols[j.name].start, cols[j.name].stop)]

        def sole(row):
            return kin.leg_fk(geom, *leg_matrices(row, spec, s))[1]

        def move(p, dx):                               # the input's sample p with this foot moved by dx, in range
            if dx == 0.0:
                return cycle[p], True, {}
            row, _, reached = _move_ankle(cycle[p], spec, s, geom, np.array([dx, 0.0, 0.0]))
            kept, cut = clamp(row[None], spec)
            return kept[0], reached, {n: c for n, c in cut.items() if n.split(".")[0] in {j.name for j in joints}}

        def count(cut):
            if held is not None:
                for name in cut:
                    held[name] = held.get(name, 0) + 1

        for a, b in _runs(st[:, k]):
            before = sole(out[a % P])
            for p in range(a, b):
                q = (p + 1) % P
                c = int(np.argmin(before[:, 2]))
                want = before[c, 0] - pace
                base = sole(cycle[q])[c, 0]
                dx, row, reached, cut = 0.0, cycle[q], True, {}
                for _ in range(passes):
                    gap = want - sole(row)[c, 0]
                    if abs(gap) < 1e-12:
                        break
                    dx += gap
                    row, reached, cut = move(q, dx)
                out[q, legs] = row[legs]
                before = sole(out[q])
                moved[q, k] = before[c, 0] - base
                short += not reached
                count(cut)
        lifted = {b % P: moved[b % P, k] for _, b in _runs(st[:, k])}
        for a, b in _runs(~st[:, k]):
            n = b - a + 1
            left = lifted.get((a - 1) % P, 0.0)
            for j in range(n):
                q = (a + j) % P
                dx = left * (n - j) / (n + 1)
                row, _, cut = move(q, dx)
                out[q, legs] = row[legs]
                moved[q, k] = dx
                count(cut)
    return out, moved, short


def stance_slip(cycle, stance, stride: float, spec: js.JointedSpec) -> dict:
    """At a steady pace with the phase advanced by distance (stride / P a phase), how far the stance feet slide on the
    floor: per stance run the running sum of (stride / P - that foot's advance, stance_advance) -- its net over the
    run, its to-and-fro (the running sum's max - min) and its largest one-phase step, each the largest over both legs'
    runs -- and the two feet's disagreement over a double support (their advances' difference summed), the largest."""
    per = stance_advance(cycle, stance, spec)
    P = len(per)
    pace = stride / P
    st = np.asarray(stance, bool)
    net = to_and_fro = largest = 0.0
    for k in range(len(SIDES)):
        for a, b in _runs(st[:, k]):
            run = pace - per[np.arange(a, b) % P, k]
            cum = np.concatenate([[0.0], np.cumsum(run)])
            net, to_and_fro = max(net, abs(float(cum[-1]))), max(to_and_fro, float(cum.max() - cum.min()))
            largest = max(largest, float(np.abs(run).max()) if len(run) else 0.0)
    both = np.isfinite(per).all(axis=1)
    disagree = max((abs(float(np.sum(per[np.arange(a, b + 1) % P, 0] - per[np.arange(a, b + 1) % P, 1])))
                    for a, b in _runs(both)), default=0.0) if both.any() else 0.0
    return {"net_m": net, "to_and_fro_m": to_and_fro, "largest_step_m": largest, "double_support_disagreement_m": disagree}


def sole_pitch_deg(row, spec: js.JointedSpec, side: str) -> float:
    """The sole's tilt about the body's lateral axis in degrees, + when the toes are lower than the heel."""
    _, _, R = kin.leg_fk(js.leg_geometry(spec, side), *leg_matrices(row, spec, side))
    return math.degrees(math.atan2(R[0, 2], R[2, 2]))


def symmetrize(cycle, stance, spec: js.JointedSpec):
    """Each side becomes the mean of itself and the other side's mirror image half a cycle on. Mirroring
    across the body's midplane negates a rotation's x and z (y stays); a hinge angle stays."""
    P = len(cycle)
    half = P // 2
    cols = js.columns(spec)
    out = np.array(cycle, float, copy=True)
    for j in spec.joints:
        if j.side != "left":
            continue
        r = js.joint(spec, j.name.replace("_left", "_right"))
        sign = np.array([-1.0 if d.axis in "xz" and j.kind == "d6" else 1.0 for d in j.dofs])
        L, R = cycle[:, cols[j.name]], cycle[:, cols[r.name]]
        Ls = (L + np.roll(R, -half, axis=0) * sign) / 2.0
        out[:, cols[j.name]] = Ls
        out[:, cols[r.name]] = np.roll(Ls, half, axis=0) * sign
    left = (stance[:, 0].astype(float) + np.roll(stance[:, 1], -half).astype(float)) / 2.0 >= 0.5
    st = np.stack([left, np.roll(left, half)], -1)
    return out, st


def swing_lows(cycle, stance, shift, spec: js.JointedSpec) -> np.ndarray:
    """Per phase and leg, the height of the swing sole's lowest corner above the floor (with the pelvis shift);
    NaN where the foot is down."""
    geoms = {s: js.leg_geometry(spec, s) for s in SIDES}
    out = np.full((len(cycle), len(SIDES)), np.nan)
    for p in range(len(cycle)):
        for k, s in enumerate(SIDES):
            if not stance[p, k]:
                out[p, k] = _sole_low(cycle[p], spec, s, geoms[s]) + shift[p]
    return out


def swing_clearance(cycle, stance, shift, spec: js.JointedSpec) -> float:
    lows = swing_lows(cycle, stance, shift, spec)
    return float(np.nanmin(lows)) if np.isfinite(lows).any() else math.inf
