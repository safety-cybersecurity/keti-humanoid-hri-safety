#!/usr/bin/env python3
"""100STYLE 의 걷기에서 인형의 속도별 걸음 표를 만든다.

    .venv/bin/python generation/dummy/gait/build_gait.py --archive ~/Downloads/100STYLE.zip

원본(BVH)은 사람이 내려받아 주고(gait.lock 의 주소 · 해시), 산출물은 표 JSON 과 빌드 보고다.
표는 build_dummy.py --jointed --gait 가 v2 USD 안에 싣는다. 같은 입력이면 바이트가 같다.

순서: 클립을 읽어 cm → m · 몸 축 → 발 접지와 뒤꿈치 닿음 → 주기(오른 닿음 → 오른 닿음) → 주기마다 인형 DOF(배우
골반의 기울기는 엉덩이에) · 끝과 처음이 맞게(주기성) · 가동 범위 · 디딤발 보정 · 속도(디딘 발이 구르는 만큼 몸이
나아감 — 두 발이 다 닿은 동안은 마지막에 닿은 발) · 골반이 돈 각 → 도는 주기를 빼고 곧게 걷는 주기만 속도 폭으로 묶어
평균 → 좌우 대칭 · 다시 보정 → 한 위상에 몸이 같은 거리를 가도록 다시 나눔 → 디딘 발을 닿은 자리에 붙잡고(다리를
맞춤 · 들릴 때 남은 만큼은 흔들기 동안 되돌림) 두 발 지지를 다시 보정 → 잰 칸 밖은 곧게 걷는 칸의 걷기 비(걸음 길이
÷ 분당 걸음 수)가 주는 보폭이 되도록 가장 가까운 칸의 엉덩이 앞뒤 흔들림만 넓히거나 좁힌 칸을 격자마다(같이 다시
나누고 붙잡음) → 칸마다 흔드는 발바닥이 바닥 위에 오도록 모자란 위상의 발목을 올림 → 0 행(선 자세)을 앞에 →
GaitTable."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation.dummy import joints_spec as js  # noqa: E402
from generation.dummy.gait import bvh, cycles, kinematics as kin, retarget as rt  # noqa: E402
from generation.dummy.gait.gait_table import GaitTable  # noqa: E402

HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "gait_spec.yaml"
LOCK_PATH = HERE / "gait.lock"
DEFAULT_OUT = REPO / "generation/dummy/build/keti_gait_100style.json"
# The stance feet held, then the double support flexed again, this many times: a joint held at its range turns the
# foot, which moves the corner it rolls on and the soles' heights a little, and each round takes up what the last left.
LOCK_ROUNDS = 3


def dumps(table: dict) -> str:
    return json.dumps(table, sort_keys=True, separators=(",", ":")) + "\n"


def _member(z: zipfile.ZipFile, clip: str) -> str:
    hits = sorted(n for n in z.namelist() if n.endswith(f"/{clip}.bvh") or n == f"{clip}.bvh")
    if len(hits) != 1:
        raise ValueError(f"압축 파일에서 {clip}.bvh 를 하나로 찾지 못했습니다: {hits}")
    return hits[0]


def _clip_cycles(motion, spec, gs):
    sk = motion.skeleton
    scale = cycles.unit_scale(sk)
    pos, rot = bvh.forward_kinematics(motion, scale=scale)
    C = cycles.body_axes(sk)
    fps = 1.0 / motion.frame_time
    heading = cycles.pelvis_heading_deg(rot, sk, C)
    floor = pos[:, sk.index("Hips")] @ C[:2].T              # the pelvis on the floor plane, body axes
    contact, strikes = {}, {}
    for side in rt.SIDES:
        h, v = cycles.foot_signals(pos, sk, C, side, fps)
        contact[side] = cycles.foot_contacts(h, v, height_tol=gs["contacts"]["height_tol_m"],
                                             speed_tol=gs["contacts"]["speed_tol_mps"],
                                             min_frames=gs["contacts"]["min_frames"])
        strikes[side] = cycles.heel_strikes(contact[side])
    found = cycles.gait_cycles(strikes["right"], strikes["left"], fps,
                               min_s=gs["cycles"]["min_s"], max_s=gs["cycles"]["max_s"])
    dofs, clipped = rt.clamp(rt.frame_dofs(rot, sk, C, spec), spec)
    P = gs["cycles"]["phase_samples"]
    out = []
    for a, b in found:
        cyc, seam_cut = rt.clamp(cycles.make_periodic(cycles.resample_cycle(dofs, a, b, P), dofs[b] - dofs[a]), spec)
        stance = np.stack([cycles.resample_cycle(contact[s].astype(float), a, b, P) > 0.5 for s in rt.SIDES], -1)
        cyc, _, _ = rt.correct_stance(cyc, stance, spec, blend=int(gs["stance"]["blend_samples"]))
        duration = (b - a) / fps
        v, _ = rt.cycle_speed(cyc, stance, duration, spec, floor_share=_advance_floor(gs))
        out.append({"dofs": cyc, "stance": stance, "duration": duration, "speed": v,
                    "turn_deg": float(heading[b] - heading[a]), "seam_clipped": seam_cut, "frames": (int(a), int(b)),
                    "actor_stride": float(np.linalg.norm(floor[b] - floor[a]))})
    return out, clipped, dofs


def _advance_floor(gs) -> float:
    return float(gs["stance"]["advance_floor_share"])


def _retimed(cyc, stance, shift, spec, gs):
    """The row re-timed so that each phase sample carries the body the same distance (rt.retime), and its stride -- the
    body's advance summed over the cycle -- with the steps raised to the floor and those with no foot down."""
    adv, floored, gaps = rt.body_advance(rt.stance_advance(cyc, stance, spec), _advance_floor(gs))
    cyc, stance, shift = rt.retime(cyc, stance, adv, shift)
    return cyc, stance, shift, float(adv.sum()), {"floored_steps": floored, "no_foot_steps": gaps}


def _locked(cyc, stance, stride, spec, gs):
    """The re-timed row with each stance foot held where it landed at the row's own pace (rt.lock_stance) and the
    double support flexed again (rt.correct_stance -- which also gives the pelvis height, lowered where a straight leg
    could not reach), both keeping the joints in their ranges, LOCK_ROUNDS times; and what the lock did: the largest
    distance an ankle was moved along x, that distance where a foot lifts (taken back over its swing), the stance
    samples where a leg could not reach and the largest knee change (degrees) -- and, per degree of freedom, the
    samples the last round held at a range."""
    blend = int(gs["stance"]["blend_samples"])
    out, shift, short, held = cyc, None, 0, {}
    for _ in range(LOCK_ROUNDS):
        held = {}
        out, _, short = rt.lock_stance(out, stance, stride, spec, held=held)
        out, shift, _ = rt.correct_stance(out, stance, spec, passes=8, blend=blend, keep_ranges=True, held=held)
    P = len(cyc)
    moved = np.zeros((P, len(rt.SIDES)))
    for k, s in enumerate(rt.SIDES):
        geom = js.leg_geometry(spec, s)
        for p in range(P):
            moved[p, k] = (kin.leg_fk(geom, *rt.leg_matrices(out[p], spec, s))[0][0]
                           - kin.leg_fk(geom, *rt.leg_matrices(cyc[p], spec, s))[0][0])
    cols = js.columns(spec)
    knees = [cols[f"knee_{s}"].start for s in rt.SIDES]
    lift = max((abs(moved[b % P, k]) for k in range(len(rt.SIDES)) for _, b in rt._runs(stance[:, k])), default=0.0)
    return out, shift, {"largest_offset_m": float(np.abs(moved).max()), "lift_off_offset_m": float(lift),
                        "short_samples": int(short),
                        "knee_change_deg": math.degrees(float(np.abs(out[:, knees] - cyc[:, knees]).max()))}, held


def _average(group, spec, gs):
    cyc = np.mean([g["dofs"] for g in group], axis=0)
    stance = np.mean([g["stance"].astype(float) for g in group], axis=0) >= 0.5
    duration = float(np.mean([g["duration"] for g in group]))
    cyc, stance = rt.symmetrize(cyc, stance, spec)
    cyc, shift, change = rt.correct_stance(cyc, stance, spec, blend=int(gs["stance"]["blend_samples"]))
    cyc, stance, shift, stride, steps = _retimed(cyc, stance, shift, spec, gs)
    cyc, shift, lock, held = _locked(cyc, stance, stride, spec, gs)
    return {"dofs": cyc, "stance": stance, "duration": duration, "speed": stride / duration, "stride": stride,
            "shift": shift, "change": change, "extrapolated": False, "retime": steps, "lock": lock, "clipped": held}


def _walk_ratio(stride: float, duration: float) -> float:
    """Step length over steps per minute: (stride / 2) / (120 / duration)."""
    return stride * duration / 240.0


def _hip_scaled(row, s, spec, gs):
    cyc, cut = rt.clamp(rt.scale_hip_pitch(row["dofs"], s, spec), spec)
    cyc, shift, change = rt.correct_stance(cyc, row["stance"], spec, blend=int(gs["stance"]["blend_samples"]))
    _, stride = rt.cycle_speed(cyc, row["stance"], row["duration"], spec, floor_share=_advance_floor(gs))
    return cyc, stride, shift, change, cut


def _extrapolate(row, target_v, walk_ratio, spec, gs):
    """A row at target_v from a measured one: only the hips' fore-aft swing is widened or narrowed, until the
    stride is the one the walk ratio gives at target_v. The stance correction then re-solves the legs in double
    support; every other joint keeps the measured row's values; and the row is re-timed to its own body advance and its
    stance feet held where they landed (_locked -- the swing lift comes after, in _finish)."""
    want = 2.0 * math.sqrt(walk_ratio * 60.0 * target_v)
    lo, hi = 0.2, 3.0
    for _ in range(30):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if _hip_scaled(row, mid, spec, gs)[1] < want else (lo, mid)
    s = (lo + hi) / 2.0
    cyc, stride, shift, change, cut = _hip_scaled(row, s, spec, gs)
    if abs(stride - want) > 1e-3 * want:
        raise ValueError(f"엉덩이 앞뒤 흔들림만으로는 {target_v:.2f} m/s 의 보폭 {want:.3f} m 를 내지 못합니다"
                         f"(배율 {s:.3f} 에서 {stride:.3f} m)")
    cyc, stance, shift, stride, steps = _retimed(cyc, row["stance"], shift, spec, gs)
    cyc, shift, lock, held = _locked(cyc, stance, stride, spec, gs)
    clipped = {k: cut.get(k, 0) + held.get(k, 0) for k in sorted(set(cut) | set(held))}
    return {"dofs": cyc, "stance": stance, "duration": stride / target_v, "speed": target_v,
            "stride": stride, "shift": shift, "change": change, "extrapolated": True,
            "hip_scale": s, "source_speed": row["speed"], "clipped": clipped, "retime": steps, "lock": lock}


def _finish(row, spec, gs):
    """Lift every swing sole that is short of its clearance -- the full one in mid-swing, ramped at the swing's
    ends (rt.swing_requirement) -- with rt.lift_swing, the mid-swing shortfall carried into its neighbours at the
    spec's slope; keeping the joint ranges."""
    full = float(gs["swing"]["min_clearance_m"])
    need = rt.swing_requirement(row["stance"], full, float(gs["swing"]["clearance_ramp"]))
    mid = np.isclose(need, full)
    before = rt.swing_lows(row["dofs"], row["stance"], row["shift"], spec)
    held = {}
    cyc, lifted, knee = rt.lift_swing(row["dofs"], row["stance"], row["shift"], spec, need,
                                      slope=float(gs["swing"]["lift_slope_m"]), held=held)
    cyc, cut = rt.clamp(cyc, spec)
    clipped = dict(row.get("clipped", {}))
    for k, n in list(cut.items()) + list(held.items()):
        clipped[k] = clipped.get(k, 0) + n
    after = rt.swing_lows(cyc, row["stance"], row["shift"], spec)
    return row | {"dofs": cyc, "lifted": lifted, "knee_added": knee, "clearance_raw": float(np.nanmin(before)),
                  "clearance": float(np.nanmin(after)), "clearance_mid": float(np.nanmin(np.where(mid, after, np.nan))),
                  "margin": float(np.nanmin(after - need)), "clipped": clipped}


def _pelvis_accel(dofs, duration, spec, P) -> float:
    """The largest vertical acceleration of the pelvis path the joint angles imply (rt.pelvis_path), at the row's pace."""
    path = rt.pelvis_path(dofs, spec)
    return float(np.max(np.abs(np.roll(path, 1) - 2.0 * path + np.roll(path, -1)))) / (duration / P) ** 2


def _smoothed(row, spec, gs, passes: int = 3):
    """Lower the pelvis path where it bends faster than the spec's vertical acceleration: both legs flexed alike
    (rt.lower_pelvis), so the feet keep their places relative to each other and the floor and only the body rides
    lower -- as far as the joint ranges allow. Where they stop a sample short, the path that came out is smoothed and
    lowered again (up to `passes` times), and the row keeps the pass that bent least -- never more than it came in.
    Nothing holds the pelvis's height but the legs; a sharp turn in that path throws the body off its feet."""
    P = len(row["dofs"])
    bend = float(gs["pelvis"]["max_vertical_accel_mps2"]) * (row["duration"] / P) ** 2
    raw = _pelvis_accel(row["dofs"], row["duration"], spec, P)
    cyc, start = row["dofs"], rt.pelvis_path(row["dofs"], spec)
    best, stopped = (raw, row["dofs"], 0.0), 0
    for _ in range(passes):
        path = rt.pelvis_path(cyc, spec)
        cyc, limited = rt.lower_pelvis(cyc, spec, path - rt.smooth_below(path, bend))
        stopped = max(stopped, limited)
        got = _pelvis_accel(cyc, row["duration"], spec, P)
        if got < best[0]:
            best = (got, cyc, float(np.max(start - rt.pelvis_path(cyc, spec))))
        if not limited:
            break
    accel, cyc, lowered = best
    # pelvis_limited: the most samples the ranges stopped short in any pass -- 0 only where the lowering went through
    return row | {"dofs": cyc, "pelvis_limited": stopped, "pelvis_lowered": lowered, "pelvis_accel_raw": raw,
                  "pelvis_accel": accel}


def _landing_knee(row, spec) -> dict:
    """The knee of a foot on the phase it lands (its stance starts while the other foot is down) and the largest it
    bends through that double support, the larger over both legs (degrees)."""
    st, P = row["stance"], len(row["stance"])
    cols = js.columns(spec)
    at, most = -math.inf, -math.inf
    for k, side in enumerate(rt.SIDES):
        knee = row["dofs"][:, cols[f"knee_{side}"].start]
        for a, _ in rt._runs(st[:, k]):
            if not st[a % P, 1 - k]:
                continue
            both = [a % P]
            while st[(both[-1] + 1) % P].all() and len(both) < P:
                both.append((both[-1] + 1) % P)
            at, most = max(at, float(knee[a % P])), max(most, float(knee[both].max()))
    return {"at_landing": math.degrees(at), "max_in_double_support": math.degrees(most)}


def _stance_sole_pitch(row, spec) -> dict:
    """The stance sole's tilt (rt.sole_pitch_deg, + toes down) over the phases where one foot alone is down."""
    st = row["stance"]
    pitch = [rt.sole_pitch_deg(row["dofs"][p], spec, s) for p in np.flatnonzero(st.sum(axis=1) == 1)
             for k, s in enumerate(rt.SIDES) if st[p, k]]
    return {"median": float(np.median(pitch)), "min": float(min(pitch)), "max": float(max(pitch))}


def _bin_report(r, spec) -> dict:
    cols = js.columns(spec)
    knees = [cols[f"knee_{s}"].start for s in rt.SIDES]
    hip = js.joint(spec, "hip_left")
    pitch = np.degrees(r["dofs"][:, cols[hip.name].start + js.axes(hip).index("y")].astype(np.float32).astype(float))
    names = js.dof_names(spec)
    legs = [c for j in spec.joints if j.group in ("hip", "knee", "ankle") for c in range(cols[j.name].start, cols[j.name].stop)]
    held = np.degrees(r["dofs"][:, legs].astype(np.float32).astype(float))
    steps = np.abs(np.diff(np.vstack([held, held[:1]]), axis=0))           # the cycle wraps round
    p, i = np.unravel_index(int(np.argmax(steps)), steps.shape)
    out = {"speed_mps": r["speed"], "stride_m": r["stride"], "cycle_s": r["duration"],
           "walk_ratio": _walk_ratio(r["stride"], r["duration"]), "swing_clearance_m": r["clearance"],
           "swing_clearance_raw_m": r["clearance_raw"], "swing_clearance_mid_m": r["clearance_mid"],
           "swing_margin_m": r["margin"], "knee_added_deg": math.degrees(r["knee_added"]),
           "max_knee_deg": math.degrees(float(r["dofs"][:, knees].max())),
           "hip_pitch_deg": [float(pitch.min()), float(pitch.max())],
           "stance_correction_deg": math.degrees(r["change"]), "clipped": r["clipped"],
           "max_leg_step": {"deg": float(steps[p, i]), "dof": names[legs[i]], "phase": int(p)},
           "stance_slip": rt.stance_slip(r["dofs"], r["stance"], r["stride"], spec),
           "stance_sole_pitch_deg": _stance_sole_pitch(r, spec), "landing_knee_deg": _landing_knee(r, spec),
           "retime": r["retime"], "stance_lock": r["lock"], "extrapolated": r["extrapolated"],
           "pelvis_accel_max_mps2": r["pelvis_accel"], "pelvis_accel_raw_mps2": r["pelvis_accel_raw"],
           "pelvis_lowered_max_m": r["pelvis_lowered"], "pelvis_limited_samples": r["pelvis_limited"]}
    if r["extrapolated"]:
        return out | {"source_speed_mps": r["source_speed"], "hip_scale": r["hip_scale"]}
    return out | {"cycles": r["cycles"]}


def build(archive: Path, spec: js.JointedSpec, gs: dict, lock: dict):
    want = (lock.get("archive") or {}).get("sha256") or ""
    if len(want) != 64:
        raise ValueError("잠금(gait.lock)에 원본의 SHA-256 이 없습니다")
    got = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    if got != want:
        raise ValueError(f"원본이 잠금과 다릅니다: {got} ≠ {want}")
    with zipfile.ZipFile(archive) as z:
        walk = bvh.parse_bvh(z.read(_member(z, gs["clips"]["walk"])).decode("utf-8"))
        stand = bvh.parse_bvh(z.read(_member(z, gs["clips"]["stand"])).decode("utf-8"))
    found, clipped, _ = _clip_cycles(walk, spec, gs)
    if not found:
        raise ValueError("걷기 클립에서 주기를 하나도 찾지 못했습니다")
    limit = float(gs["cycles"]["max_turn_deg"])
    kept = [c for c in found if abs(c["turn_deg"]) <= limit]
    width = float(gs["speeds"]["bin_mps"])
    bins = {}
    for c in kept:
        bins.setdefault(int(round(c["speed"] / width)), []).append(c)
    rows = [_average(g, spec, gs) | {"cycles": len(g)} for k, g in sorted(bins.items())
            if len(g) >= gs["speeds"]["min_cycles_per_bin"]]
    if not rows:
        raise ValueError("곧게 걷는 주기가 충분한 속도 칸이 없습니다")
    rows.sort(key=lambda r: r["speed"])
    walk_ratio = float(np.mean([_walk_ratio(r["stride"], r["duration"]) for r in rows]))
    lo, hi = (float(v) for v in gs["speeds"]["grid_mps"])
    step = float(gs["speeds"]["grid_step_mps"])
    grid = [round(lo + k * step, 6) for k in range(int(round((hi - lo) / step)) + 1)]
    below = [_extrapolate(rows[0], g, walk_ratio, spec, gs) for g in grid if g < rows[0]["speed"] - 1e-6]
    above = [_extrapolate(rows[-1], g, walk_ratio, spec, gs) for g in grid if g > rows[-1]["speed"] + 1e-6]
    rows = [_smoothed(_finish(r, spec, gs), spec, gs) for r in below + rows + above]
    extrapolated = [r["speed"] for r in rows if r["extrapolated"]]
    P = gs["cycles"]["phase_samples"]
    names = js.dof_names(spec)
    targets = np.zeros((len(rows) + 1, P, len(names)), np.float32)
    for k, r in enumerate(rows, 1):
        targets[k] = r["dofs"].astype(np.float32)
    table = GaitTable(tuple(names), np.array([0.0] + [r["speed"] for r in rows]),
                      np.array([0.0] + [r["stride"] for r in rows]), targets)
    stand_found, _, stand_dofs = _clip_cycles(stand, spec, gs)
    seam = {}
    for c in kept:
        for k, n in c["seam_clipped"].items():
            seam[k] = seam.get(k, 0) + n
    turns = np.abs([c["turn_deg"] for c in found])
    report = {
        "archive_sha256": got, "walk_clip": gs["clips"]["walk"], "cycles": len(found),
        "turning": {"max_turn_deg": limit, "kept": len(kept), "dropped": len(found) - len(kept),
                    "abs_turn_deg_percentiles": {str(q): float(np.percentile(turns, q)) for q in (10, 50, 90)}},
        "bins": [_bin_report(r, spec) for r in rows],
        "extrapolated_speeds": extrapolated, "walk_ratio_m_per_steps_per_min": walk_ratio,
        "walk_ratio_actor_m_per_steps_per_min": float(np.mean([_walk_ratio(c["actor_stride"], c["duration"])
                                                               for c in kept])) if kept else None,
        "walk_ratio_literature": gs["literature"]["walk_ratio"],
        "clipped": clipped, "max_stance_correction_deg": math.degrees(max(r["change"] for r in rows)),
        "swing_requirement": {"min_clearance_m": float(gs["swing"]["min_clearance_m"]),
                              "ramp": float(gs["swing"]["clearance_ramp"]),
                              "lift_slope_m": float(gs["swing"]["lift_slope_m"])},
        "min_swing_clearance_m": min(r["clearance"] for r in rows),
        "min_swing_clearance_raw_m": min(r["clearance_raw"] for r in rows),
        "stand_pose_deg": {n: math.degrees(float(np.median(stand_dofs[:, i]))) for i, n in enumerate(names)},
        "stand_pose_sole_pitch_deg": {s: float(np.median([rt.sole_pitch_deg(d, spec, s) for d in stand_dofs]))
                                      for s in rt.SIDES},
        "seam_clipped": seam,
    }
    return table.to_json() | {"source": gs["source"], "archive_sha256": got}, report


def main() -> int:
    p = argparse.ArgumentParser(description="100STYLE → 인형의 속도별 걸음 표")
    p.add_argument("--archive", required=True)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--report", default=None)
    a = p.parse_args()
    gs = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    lock = yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8"))
    table, report = build(Path(a.archive).expanduser(), js.load(), gs, lock)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(table), encoding="utf-8")
    rep = Path(a.report) if a.report else out.with_suffix(".report.json")
    rep.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"걸음 표: {out} · 행 {len(table['speeds'])} · 보고 {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
