"""An episode's evidence: the numbers behind its label, from the same recorded arrays and the same canon.

- min_separation_margin_m -- the smallest min_distance_human - separation_required over every step, in 64-bit
  floats.
- first_contact_s -- the first contact_onset step / fps; None without a contact.
- force_ratio.<region> -- for each body region whose R-TS15066-A2 rule the scenario declares, whose limit is filled
  and which the episode records (contact_region_names): the largest force / limit over the steps the labeller judges
  (judgement_windows' ``judged``). The limit is the one the rule compares that step's contact against -- the
  quasi-static force for a quasi-static contact, that times the transient multiplier for a transient one, and for an
  untyped contact after the first contact, in a scenario that holds touching the person to be the fail, the smaller of
  the two. Where table A.2 permits no transient contact the transient ratio is inf, and the untyped one is inf for any
  force above 0. Other steps do not count; with nothing to count the ratio is 0.0, and a region that misses any of the
  three conditions is None. So a ratio is above 1 exactly when the region's rule fires.
- force_ratio_max -- the largest of the regions' ratios that are not None; None when every one is.
- human_first_seen_s, human_last_seen_s -- the first and the last step on which a robot camera sees the person by the
  ``seeing`` rule of the cameras canon (generation/visibility.py: the recorded visible fraction and the person's height
  in the picture, counted again from the camera's pose, the person's root and stature), / fps. None when it cannot be
  told (no cameras canon, a robot camera's pose or visible fraction not recorded, no human_root_pose, no stature) or
  when the person was never seen.

The names and definitions are the project's own; no standard defines them. Values are Python floats and None only,
so the result is plain JSON (json writes inf as Infinity and reads it back).

Isaac-free: the npz arrays, the episode sidecar and the canon documents are the whole input.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from generation.labeling import label_episodes

EVIDENCE_KEY = "evidence"                                   # the label member
PREFIX = "evidence."                                        # the episode columns
REGIONS = tuple(f"BR-{n:02d}" for n in range(1, 13))
NAMES = ("min_separation_margin_m", "first_contact_s", *(f"force_ratio.{r}" for r in REGIONS),
         "force_ratio_max", "human_first_seen_s", "human_last_seen_s")
EVIDENCE_COLUMNS = tuple(PREFIX + name for name in NAMES)   # 17

# The rules a region's force ratio stands beside: one per body region, each comparing that region's contact force.
A2_RULE_PREFIX = "R-TS15066-A2-"


def _ratio(force: np.ndarray, limit: float) -> np.ndarray:
    """force / limit; with a limit of 0, inf for a force above it and 0.0 for none. For a limit above 0 the quotient
    is above 1 exactly where force > limit: division rounds monotonically, and the next float above a limit divided
    by it rounds above 1."""
    if limit > 0.0:
        return force / limit
    return np.where(force > 0.0, math.inf, 0.0)


def force_ratios(arrays: Mapping, meta: Mapping, rules: Mapping, scenario: Mapping, *, judged, after_onset,
                 contact_judged: bool) -> dict[str, float | None]:
    """{region code: the largest force / limit over the judged steps, or None} for every region of REGIONS -- the
    limits label_episode's A2 branch compares against (``judged``, ``after_onset`` and ``contact_judged`` as
    judgement_windows gives them)."""
    names = list(meta["contact_region_names"])
    force_all = np.asarray(arrays["contact_force"], dtype=float)
    kind_all = np.asarray(arrays["contact_type"])
    judged = np.asarray(judged, dtype=bool)
    after_onset = np.asarray(after_onset, dtype=bool)
    out: dict[str, float | None] = dict.fromkeys(REGIONS)
    for rule in label_episodes.selected_rules(rules, scenario):
        code = rule.get("body_region")
        if not str(rule.get("rule_id")).startswith(A2_RULE_PREFIX) or code not in out:
            continue
        if not label_episodes.limits_are_filled(rules, rule) or code not in names:
            continue
        ref = str(rule.get("limits_ref") or "")
        if ref != f"contact_limits.{code}":
            raise ValueError(f"{rule.get('rule_id')} 의 limits_ref={ref!r} 가 'contact_limits.{code}' 가 아닙니다 — "
                             "다른 부위의 한계로 나눈 비율이 됩니다")
        region = label_episodes._region_of(rules, rule) or {}
        f_qs = float(region["limits"]["force_quasistatic"]["value"])
        tm = region.get("transient_multiplier") or {}
        col = names.index(code)
        force, kind = force_all[:, col], kind_all[:, col]

        step = np.zeros(len(force))
        quasistatic = kind == "quasistatic"
        step[quasistatic] = _ratio(force[quasistatic], f_qs)
        transient = kind == "transient"
        if tm.get("not_applicable"):
            step[transient] = math.inf
        else:
            step[transient] = _ratio(force[transient], f_qs * float(tm["value"]))
        if contact_judged:
            strict = 0.0 if tm.get("not_applicable") else f_qs * min(1.0, float(tm["value"]))
            untyped = (kind == "none") & after_onset
            step[untyped] = _ratio(force[untyped], strict)
        counted = step[judged]
        out[code] = float(counted.max()) if counted.size else 0.0
    return out


def seen_steps(arrays: Mapping, meta: Mapping, cameras: Mapping | None) -> np.ndarray | None:
    """A bool a step: whether a robot camera sees the person on it by the cameras canon's ``seeing`` rule. None when it
    cannot be told -- no cameras canon, a robot camera's pose or visible-fraction array missing, no human_root_pose, or
    no stature in the sidecar.

    The recorded camera pose is pos3 + quat4 (wxyz) in the camera's ROS optical frame; visibility.height_px takes the
    quaternion xyzw. The height is worked out only on the steps whose fraction already meets its threshold -- the other
    steps fail seen() whatever the height."""
    if cameras is None:
        return None
    from generation import visibility

    stature = meta.get("human_stature_m")
    names = visibility.robot_cameras(cameras)
    keys = ["human_root_pose", *(f"{prefix}{name}" for name in names
                                 for prefix in ("camera_pose_", "human_visible_fraction_"))]
    if stature is None or any(key not in arrays for key in keys):
        return None
    rule = visibility.seeing_rule(cameras)
    root = np.asarray(arrays["human_root_pose"], dtype=float)
    n = len(root)
    if root.shape != (n, 7):
        raise ValueError(f"human_root_pose 의 모양이 (스텝, 7) 이 아닙니다: {root.shape}")
    seen = np.zeros(n, dtype=bool)
    for name in names:
        cam = cameras["cameras"][name]
        pose = np.asarray(arrays[f"camera_pose_{name}"], dtype=float)
        fraction = np.asarray(arrays[f"human_visible_fraction_{name}"], dtype=float)
        if pose.shape != (n, 7) or fraction.shape != (n,):
            raise ValueError(f"{name} 의 자세 {pose.shape} · 보이는 몫 {fraction.shape} 이 스텝 {n} 과 맞지 않습니다")
        quat_xyzw = pose[:, [4, 5, 6, 3]]
        for k in np.flatnonzero((fraction >= rule["min_visible_fraction"]) & ~seen):
            height = visibility.height_px(cam, pose[k, :3], quat_xyzw[k], root[k, :3], stature)
            seen[k] = visibility.seen(fraction[k], height, rule)
    return seen


def episode_evidence(arrays: Mapping, meta: Mapping, rules: Mapping, scenario: Mapping, cameras: Mapping | None, *,
                     windows: tuple, fps: int) -> dict[str, float | None]:
    """The evidence of one episode, keys NAMES in order. ``windows`` is (contact, judged, after_onset, contact_judged)
    from label_episodes.judgement_windows; ``fps`` turns steps into seconds."""
    contact, judged, after_onset, contact_judged = windows
    if isinstance(fps, bool) or not isinstance(fps, (int, np.integer)) or fps <= 0:
        raise ValueError(f"fps 가 양의 정수가 아닙니다: {fps!r}")
    fps = int(fps)

    margin = (np.asarray(arrays["min_distance_human"], dtype=np.float64)
              - np.asarray(arrays["separation_required"], dtype=np.float64))
    onsets = np.flatnonzero(np.asarray(contact, dtype=bool))
    ratios = force_ratios(arrays, meta, rules, scenario, judged=judged, after_onset=after_onset,
                          contact_judged=contact_judged)
    known = [v for v in ratios.values() if v is not None]
    seen = seen_steps(arrays, meta, cameras)
    seen_at = np.flatnonzero(seen) if seen is not None else np.zeros(0, dtype=int)

    out = {
        "min_separation_margin_m": float(margin.min()),
        "first_contact_s": int(onsets[0]) / fps if onsets.size else None,
        **{f"force_ratio.{r}": ratios[r] for r in REGIONS},
        "force_ratio_max": max(known) if known else None,
        "human_first_seen_s": int(seen_at[0]) / fps if seen_at.size else None,
        "human_last_seen_s": int(seen_at[-1]) / fps if seen_at.size else None,
    }
    return {name: out[name] for name in NAMES}
