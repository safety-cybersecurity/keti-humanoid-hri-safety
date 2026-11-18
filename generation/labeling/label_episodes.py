"""Turn one recorded episode into outcome / triggered_rules / severity.

Kept out of the converter on purpose. The schema (§9) promises that changing
a threshold means relabelling only -- no regeneration -- and that promise is
only structurally true if labelling is a step of its own. When a threshold
changes, this is the one thing that reruns.

The promise is scoped to thresholds the rules only compare against. A parameter
that also enters the robot's behaviour during generation -- the
separation-monitoring inputs, which set the corridor the trajectory is shaped by
-- is baked into the data, and relabelling cannot move it (2026-08-25).

Isaac-free: npz arrays and the episode sidecar are the whole input.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

# Run as a script, sys.path[0] is this file's directory, not the repo root, so
# the `generation.*` imports below fail. run_s1a.py already carries the same
# guard; without it the documented commands need a PYTHONPATH the
# documentation never mentions (measured: all four failed).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generation.directed_separation import CONTACT_RULE_ID, DIRECTED_SSM_RULE_ID
# TABLE: a scenario that declares a conditions table, whose episodes carry their evidence (schema §5).
from generation.field_map import TABLE, step_dt_from_timestamps

# In the scenarios that declare R-HRI-CONTACT-01 (schema §9.1) a fail is a contact, and S2 or S3 is
# read off the table A.2 limits instead: S3 when an A2 rule fires or the head or face takes force.
# Elsewhere severity is defined by rule class, not by force magnitude.
# A magnitude definition needs the S0-S3 bands, which the schema leaves open
# (§12), and a required field that cannot be computed is worse than a coarser
# one.
SEVERITY_NOMINAL = "S0"
SEVERITY_NEAR_MISS = "S1"
SEVERITY_FAIL_NO_CONTACT = "S2"
SEVERITY_FAIL_WITH_CONTACT = "S3"
SEVERITY_CONTACT_WITHIN_LIMITS = "S2"
SEVERITY_CONTACT_BEYOND_LIMITS = "S3"
# Table A.2 permits no contact with these (its transient column is "not applicable").
HEAD_FACE_REGIONS = ("BR-01", "BR-02")


# Which safety.event names each rule's computation reads. A rule whose event
# is absent from the episode cannot be evaluated, and must not be quietly
# treated as "did not fire" -- that is the difference between a near miss the
# data missed and a near miss that did not happen.
RULE_REQUIRED_EVENTS = {
    "R-TS15066-SSM-01": ("contact_onset",),
    "R-10218-STOP-01": ("protective_stop_triggered",),
    "R-TS15066-SSM-02": ("contact_onset",),
    "R-HRI-CONTACT-01": ("contact_onset",),
}


def selected_rules(rules: dict, scenario: dict) -> list[dict]:
    """The rules this scenario declared, expanding wildcards the way C8 does.

    Raises on an empty result. An empty selector or an empty catalogue used to
    fall through to nominal/S0, which reads as "nothing happened" when the
    truth is "the canon was not loaded".
    """
    declared = scenario.get("rules") or []
    if not declared:
        raise ValueError(
            f"{scenario.get('scenario_id')}: 시나리오가 rules 를 선언하지 않았습니다 — "
            "규칙 없이 라벨을 붙이면 전 에피소드가 nominal 이 됩니다"
        )
    catalogue = rules.get("rules") or []
    if not catalogue:
        raise ValueError("meta/rules.json 의 rules 가 비었습니다")
    out = [rule for rule in catalogue
           if any(fnmatch.fnmatch(rule.get("rule_id", ""), sel) for sel in declared)]
    unmatched = [sel for sel in declared
                 if not any(fnmatch.fnmatch(r.get("rule_id", ""), sel) for r in catalogue)]
    if unmatched:
        raise ValueError(f"rules.json 에 없는 규칙을 가리킵니다: {unmatched}")
    return out


def _validate_arrays(arrays: dict, meta: dict, needed_events: set[str]) -> np.ndarray:
    """Check the step arrays before any rule reads them. Returns the event matrix.

    Every failure here used to be silent: a short event array labelled from a
    truncated timeline, a float array turned every NaN into True through
    astype(bool), and a missing event name produced an all-False column.
    """
    n = len(np.asarray(arrays["timestamp"]).reshape(-1))
    names = list(meta["safety_event_names"])
    absent = sorted(needed_events - set(names))
    if absent:
        raise ValueError(
            f"이 시나리오의 규칙이 읽는 이벤트가 기록에 없습니다: {absent} — "
            "규칙이 성립하지 않은 것과 평가할 수 없는 것은 다릅니다"
        )
    events = np.asarray(arrays["safety_event"])
    if events.shape != (n, len(names)):
        raise ValueError(
            f"safety_event shape {events.shape} != (스텝 {n}, 이벤트 {len(names)})"
        )
    if events.dtype != np.bool_:
        raise ValueError(
            f"safety_event 의 dtype 이 bool 이 아닙니다: {events.dtype} — "
            "astype(bool) 은 NaN 을 True 로 바꿉니다"
        )
    root_vel = np.asarray(arrays["root_vel"])
    if root_vel.shape != (n, 6):
        raise ValueError(f"root_vel shape {root_vel.shape} != ({n}, 6)")
    if not np.isfinite(root_vel).all():
        raise ValueError("root_vel 에 비유한 값이 있습니다")
    return events


def _parameter(rule: dict, name: str) -> float:
    """A project threshold that must be present and usable.

    Raises rather than defaulting. A missing threshold that silently labels
    an episode nominal turns a whole batch into what looks like a control
    group -- the same fail-open _is_usable_ssm_value exists to stop on the
    generation side.
    """
    entry = (rule.get("parameters") or {}).get(name)
    value = entry.get("value") if isinstance(entry, dict) else None
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{rule.get('rule_id')}.{name} 값이 없거나 수치가 아닙니다: {value!r} — "
            "meta/rules.json 을 먼저 채우십시오"
        )
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{rule.get('rule_id')}.{name} 값이 사용할 수 없습니다: {value!r}")
    return float(value)


def _event(arrays: dict, meta: dict, name: str) -> np.ndarray:
    """One event column. Absent names give all-False *only* for events no
    selected rule needs -- _validate_arrays has already raised for the rest."""
    names = list(meta["safety_event_names"])
    n = len(np.asarray(arrays["timestamp"]).reshape(-1))
    if name not in names:
        return np.zeros(n, dtype=bool)
    return np.asarray(arrays["safety_event"])[:, names.index(name)]


# meta/rules.json carries the standard's thresholds as values with their clause
# references, published with the copyright holder's permission (2026-08-27). The
# transcription they were copied from is not in the public repository; where it is
# present the labeller merges it at run time, filling only what the canon leaves empty.
STANDARD_VALUES_PATH = _REPO_ROOT / "internal" / "standards" / "ts15066-annex-a.json"

# Which entries of a contact_limits region R-TS15066-A2-* actually reads. The
# region carries more than this (pressure, energy), and those cannot be computed
# in v0.1. Pressure needs contact area, which PhysX does not report. Energy needs
# v_rel, which the recorder does write now -- but as an approximation sampled at
# the record instant while the force is a substep peak, so the two cannot be
# combined (schema §6). Requiring *every* limit to be filled would mean the rule
# stays unevaluated forever over inputs it never uses.
A2_REQUIRED_LIMITS = ("force_quasistatic",)

# 항목마다 쓸 수 있는 단위. 라벨러는 단위를 읽지 않고 기록된 값(N)과 직접 비교하므로,
# 정본과 사본이 **같은 틀린 단위**에 합의해도 판정이 어긋난다 — 문자열 일치만으로는
# 잡히지 않는 경우다. 검증기 쪽 같은 계약은 validate_dataset.py CANONICAL_UNITS.
CANONICAL_UNITS = {
    "force_quasistatic": "N",
    "pressure_quasistatic": "N/cm^2",
    "energy_transient": "J",
    "effective_mass": "kg",
    "spring_constant": "N/mm",
    "transient_multiplier": "1",
}


def load_standard_values(path: Path | None = None) -> dict | None:
    """Annex A values, or None when the internal copy is not present.

    Absent is a normal state, not an error: a clone without that copy still
    labels with the values the canon carries. What the missing copy costs is the
    rules whose limit is empty in the canon as well: each lands in not_evaluated,
    saying so. (A rule lands there for other reasons too -- a region the episode
    does not record, a contact whose type stayed unresolved -- and the copy does
    not change those.) Silently treating absent as zero would turn "we could not
    check" into "it was fine".
    """
    p = path or STANDARD_VALUES_PATH
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("regions") or None


def _is_usable_limit(value: object) -> bool:
    """Whether a merged threshold is a number the comparison can use.

    nan is the dangerous one: it is a float, so a bare None check calls it
    filled, and then `force > nan` is False for every force there is -- the rule
    never fires and the episode reads nominal however hard the robot hit. This
    is the same contract as _is_usable_ssm_value in
    generation/envs/s1a/mdp/safety.py, and the two must stay the same
    (CONTRIBUTING.md 「함께 고치는 곳」).
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) >= 0.0


def merge_standard_values(rules: dict, values: dict | None) -> dict:
    """Fill the citation-only thresholds from the transcription.

    Only fills what is empty. A value already in rules.json wins -- the canon
    carries the numbers now, and this must not quietly shadow them with a
    second copy.
    """
    if not values:
        return rules

    def _check_unit(where: str, dst: dict, src_entry: dict, field: str) -> None:
        """사본의 단위가 공개 정본과 같은지.

        사본이 kN 인데 정본이 N 이면 숫자만 옮겨져 1000배 어긋난 판정이 나온다.
        사본에 단위가 아예 없어도 같은 결과이므로, 없는 것도 거부한다.
        """
        canonical = CANONICAL_UNITS.get(field)
        want = dst.get("unit")
        got = src_entry.get("unit")
        if got is None:
            raise ValueError(f"표준값 사본의 {where} 에 unit 이 없습니다 — "
                             f"정본은 {want!r} 을 기다립니다")
        if got != want:
            raise ValueError(f"표준값 사본의 {where} 단위가 {got!r} 인데 정본은 "
                             f"{want!r} 입니다 — 숫자만 옮기면 판정이 어긋납니다")
        # 둘이 **같은 오답**에 합의한 경우는 위 비교로 잡히지 않는다. 라벨러는 단위를
        # 읽지 않고 기록된 N 과 직접 비교하므로, 양쪽이 kN 이면 141 N 이 140 kN 을
        # 넘었다고 판정한다. 필드마다 쓸 수 있는 단위는 하나뿐이므로 못박는다.
        if canonical is not None and got != canonical:
            raise ValueError(
                f"{where} 의 단위가 {got!r} 입니다 — 이 항목은 {canonical!r} 만 쓸 수 "
                "있습니다. 정본과 사본이 같은 단위를 쓰고 있어도, 그 단위가 산출식이 "
                "가정하는 것과 다르면 판정이 통째로 어긋납니다")

    def _checked(where: str, value: object):
        if not _is_usable_limit(value):
            raise ValueError(
                f"표준값 사본의 {where} 가 쓸 수 없는 값입니다: {value!r} — "
                "비유한·음수·비수치는 비교를 영영 거짓으로 만들어 라벨을 지웁니다")
        return value

    for region, body in (rules.get("contact_limits") or {}).items():
        src = values.get(region)
        if not isinstance(body, dict):
            continue
        if not isinstance(src, dict):
            raise ValueError(
                f"표준값 사본에 {region} 이 없습니다 — 부위가 빠지면 그 규칙만 조용히 "
                "평가되지 않은 채 배치가 끝나고, 왜인지는 아무 데도 남지 않습니다")
        for group, names in (("limits", ("force_quasistatic", "pressure_quasistatic",
                                         "energy_transient")),
                             ("biomechanical", ("effective_mass", "spring_constant"))):
            g = body.get(group)
            if not isinstance(g, dict):
                continue
            for name in names:
                dst, sv = g.get(name), src.get(name)
                if not isinstance(dst, dict) or dst.get("value") is not None:
                    continue
                if not isinstance(sv, dict):
                    raise ValueError(
                        f"표준값 사본의 {region} 에 {name} 이 없습니다 — 공개 정본이 "
                        "그 자리를 비워 두고 기다리는데 사본에도 없으면 아무도 채우지 못합니다")
                _check_unit(f"{region}.{group}.{name}", dst, sv, name)
                dst["value"] = _checked(f"{region}.{group}.{name}", sv.get("value"))
        tm, s = body.get("transient_multiplier"), src.get("transient_multiplier")
        if isinstance(tm, dict) and tm.get("value") is None and not tm.get("not_applicable"):
            # 정본이 값을 기다리는데 사본에 항목이 없거나 값이 비어 있으면, 그 부위의
            # 동적 접촉은 영영 평가되지 않는다. 다른 항목과 같은 기준으로 거부한다.
            if not isinstance(s, dict) or s.get("value") is None:
                raise ValueError(
                    f"표준값 사본의 {region}.transient_multiplier 가 비어 있습니다 — "
                    "정본이 그 자리를 기다리는데 사본에도 없으면 그 부위의 동적 접촉은 "
                    "판정되지 않습니다")
        if isinstance(tm, dict) and isinstance(s, dict) and tm.get("value") is None:
            # 표준이 「해당 없음」이라 한 자리에 사본이 값을 들고 있으면 둘 중 하나가
            # 거짓이다. 조용히 하나를 고르지 않는다.
            if tm.get("not_applicable") is True and s.get("value") is not None:
                raise ValueError(
                    f"{region}.transient_multiplier 는 not_applicable 인데 사본에 값 "
                    f"{s.get('value')!r} 이 있습니다 — 정본과 사본 중 하나가 틀렸습니다")
            if s.get("value") is not None:
                _check_unit(f"{region}.transient_multiplier", tm, s, "transient_multiplier")
                tm["value"] = _checked(f"{region}.transient_multiplier", s.get("value"))
    return rules


def _region_of(rules: dict, rule: dict) -> dict | None:
    ref = rule.get("limits_ref")
    if not ref:
        return None
    node: object = rules
    for part in ref.split("."):
        node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            return None
    return node if isinstance(node, dict) else None


def limits_are_filled(rules: dict, rule: dict) -> bool:
    """Whether the limits this rule's computation reads actually have values."""
    region = _region_of(rules, rule)
    if region is None:
        return not rule.get("limits_ref")
    limits = region.get("limits")
    if not isinstance(limits, dict) or not limits:
        return False
    for name in A2_REQUIRED_LIMITS:
        e = limits.get(name)
        if not isinstance(e, dict) or not _is_usable_limit(e.get("value")):
            return False
    # A transient contact is compared against quasi-static x multiplier, so the
    # multiplier has to be there too -- unless the standard says this region
    # takes no transient contact at all, which is itself a usable answer.
    tm = region.get("transient_multiplier")
    if not isinstance(tm, dict):
        return False
    return _is_usable_limit(tm.get("value")) or tm.get("not_applicable") is True


def judgement_windows(arrays: dict, meta: dict,
                      rules_here: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """(contact, judged, after_onset, contact_judged): the steps the contact rules read, as label_episode reads them.

    contact -- the contact_onset event column. judged -- the steps whose force is compared against a limit.
    after_onset -- the steps from the first contact_onset on. contact_judged -- whether ``rules_here`` (the scenario's
    selected rules) holds touching the person to be the fail (R-HRI-CONTACT-01). Raises when the sidecar has no
    hold_release_step.
    """
    contact = _event(arrays, meta, "contact_onset")

    # 더미가 자유 물체인 구간. 홀드 해제 이전 스텝은 접촉 판정에 쓰지 않는다.
    # -1 은 「끝까지 해제되지 않음」이므로 자유 구간이 없다.
    n_steps = len(np.asarray(arrays["timestamp"]))
    # 없으면 전 스텝이 오염 취급이 되어 **접촉 규칙이 하나도 안 걸린다** — 조용히
    # 라벨이 달라지므로 거부한다 (이 필드를 담지 않은 메타는 시험의 가짜 메타라도
    # 여기서 멈춘다).
    if "hold_release_step" not in meta:
        raise ValueError(
            "에피소드 메타에 hold_release_step 이 없습니다 — 홀드 해제 이전 스텝을 "
            "접촉 판정에서 뺄 수 없고, 그대로 두면 서보 힘이 섞인 접촉력으로 라벨이 "
            "정해집니다")
    release = int(meta["hold_release_step"])
    free_body = (np.arange(n_steps) >= release) if release >= 0 else np.zeros(n_steps, bool)
    contact_steps = np.flatnonzero(contact)
    after_onset = (np.arange(n_steps) >= contact_steps[0]) if contact_steps.size \
        else np.zeros(n_steps, bool)
    # The hri judgement (schema §9.1): touching the person is the fail and severity is read off the
    # table A.2 limits. A scenario that does not declare it (S1a, S1d) is labelled by rule class, below.
    contact_judged = any(r["rule_id"] == CONTACT_RULE_ID for r in rules_here)
    # The steps the force is judged on. In the hri judgement the dummy's hold is let go at the first
    # touch, so the first contact_onset step is the one step whose physics still ran with the hold
    # (hold_contaminated_steps 1, hold_release_step the step after it); it is judged all the same, so
    # a contact that lasts a single step is compared. Held steps after it are not.
    judged = free_body.copy()
    if contact_judged and contact_steps.size:
        judged[contact_steps[0]] = True
    return contact, judged, after_onset, contact_judged


# The arrays the evidence reads that hold a number a step: one with a value that is not finite would come out of
# the arithmetic as a NaN, which is how an evidence column says "no value". The recorder writes them finite.
EVIDENCE_FINITE = ("min_distance_human", "separation_required", "contact_force")


def _evidence_inputs_problem(arrays: dict, n: int, cameras: dict | None) -> str | None:
    """Why ``arrays`` cannot give an episode of ``n`` steps its evidence, or None: an array the evidence reads with
    another step count, or a number of EVIDENCE_FINITE that is not finite. With a cameras canon the evidence also reads
    the person's pelvis and each robot camera's pose and visible fraction, where they are recorded."""
    names = [*EVIDENCE_FINITE, "contact_type"]
    if cameras is not None:
        from generation import visibility

        names += [k for k in ("human_root_pose",
                              *(f"{prefix}{cam}" for cam in visibility.robot_cameras(cameras)
                                for prefix in ("camera_pose_", "human_visible_fraction_")))
                  if k in arrays]
    for name in names:
        values = np.asarray(arrays[name])
        if values.ndim == 0 or len(values) != n:
            return f"근거 요약이 읽는 {name} 의 스텝 수가 {values.shape} 로 에피소드의 {n} 과 다릅니다"
    for name in EVIDENCE_FINITE:
        bad = int((~np.isfinite(np.asarray(arrays[name], dtype=float))).sum())
        if bad:
            return (f"근거 요약이 읽는 {name} 에 유한하지 않은 값이 {bad}개 있습니다 — 근거 요약의 열은 NaN 을 「값 "
                    "없음」으로 싣습니다. 계산에서 나온 NaN 은 배포본에서 값 없음으로 읽혀 라벨을 다시 낼 수 없습니다")
    return None


def label_episode(arrays: dict, meta: dict, rules: dict, scenario: dict, cameras: dict | None = None) -> dict:
    """outcome / triggered_rules / severity / not_evaluated for one episode -- and, when the scenario declares a
    conditions table, its evidence (generation/labeling/evidence.py) under ``evidence``. ``cameras`` is the cameras
    canon the evidence tells by when a robot camera saw the person (None: it cannot tell)."""
    dt = step_dt_from_timestamps(arrays["timestamp"])
    rules_here = selected_rules(rules, scenario)
    needed = {name for rule in rules_here
              for name in RULE_REQUIRED_EVENTS.get(rule["rule_id"], ())}
    _validate_arrays(arrays, meta, needed)
    table = TABLE in scenario
    if table:
        problem = _evidence_inputs_problem(arrays, len(np.asarray(arrays["timestamp"]).reshape(-1)), cameras)
        if problem:
            raise ValueError(problem)

    triggered: list[str] = []
    not_evaluated: list[dict] = []

    contact, judged, after_onset, contact_judged = judgement_windows(arrays, meta, rules_here)
    had_contact = bool(contact.any())

    for rule in rules_here:
        rid = rule["rule_id"]

        if not limits_are_filled(rules, rule):
            not_evaluated.append({
                "rule_id": rid,
                "reason": f"{rule.get('limits_ref')} 의 한계값이 비어 있어 비교할 수 없음 "
                          "— 표준 수치 사본(--standard-values)이 있어야 병합된다",
            })
            continue

        if rid == "R-TS15066-SSM-01":
            # 정본의 조건은 "이격거리 미만이나 접촉 없음" 이다. 접촉이 있으면
            # 근접사고가 아니다 (rule_precedence 2번과 같은 이야기).
            if "separation_violation" in arrays:
                violated = bool(np.asarray(arrays["separation_violation"]).astype(bool).any())
            else:
                violated = bool(_event(arrays, meta, "ssm_violation").any())
            if violated and not had_contact:
                triggered.append(rid)
            continue

        if rid == DIRECTED_SSM_RULE_ID:
            # The flag the recorder computed with the directed S_p and the approach gate
            # (generation/directed_separation.py); the validator holds it to its columns (C8).
            if "separation_violation" not in arrays:
                raise ValueError(f"{rid} 는 기록된 separation_violation 을 읽습니다 — 이 에피소드에 그 배열이 없습니다")
            violated = bool(np.asarray(arrays["separation_violation"]).astype(bool).any())
            if violated and not had_contact:
                triggered.append(rid)
            continue

        if rid == CONTACT_RULE_ID:
            if had_contact:
                triggered.append(rid)
            continue

        if rid == "R-10218-STOP-01":
            budget = _parameter(rule, "max_stop_time")
            speed_limit = _parameter(rule, "stop_speed_threshold")
            # |v_x|: a robot sliding backwards at 1 m/s has not stopped.
            vx = np.abs(np.asarray(arrays["root_vel"])[:, 0].astype(float))
            onsets = np.flatnonzero(_event(arrays, meta, "protective_stop_triggered"))
            # recorder latches this edge once per episode (recorder.py's
            # _prev_stop_step). More than one means the recording is damaged or
            # the contract changed; either way it is not something to label
            # through.
            if len(onsets) > 1:
                raise ValueError(
                    f"protective_stop onset 이 {len(onsets)}개입니다 — 계약상 최대 1개"
                )
            # Frames, not seconds. 3 * 0.1 is 0.30000000000000004, which failed
            # an episode that stopped exactly on budget.
            budget_frames = math.floor(budget / dt + 1e-9)
            for onset in onsets:
                # "이하" -- the canon, this code and measure_stop_time.py all
                # use at-or-below; see rules.json's condition text.
                below = np.flatnonzero(vx[onset:] <= speed_limit)
                if below.size == 0 or int(below[0]) > budget_frames:
                    triggered.append(rid)
            continue

        if rid.startswith("R-TS15066-A2-"):
            region_code = rule.get("body_region")
            names = list(meta["contact_region_names"])
            if region_code not in names:
                not_evaluated.append({
                    "rule_id": rid,
                    "reason": f"{region_code} 이 이 에피소드의 contact_region_names 에 없음",
                })
                continue
            col = names.index(region_code)
            ref = str(rule.get("limits_ref") or "")
            # 규칙이 자기 부위가 아닌 한계를 가리키면 다른 부위의 기준으로 판정한다.
            # 조용히 계산되고 라벨은 그럴듯해 보인다.
            if ref != f"contact_limits.{region_code}":
                raise ValueError(
                    f"{rid} 의 limits_ref={ref!r} 가 'contact_limits.{region_code}' 가 "
                    "아닙니다 — 다른 부위나 다른 트리의 기준으로 판정됩니다")
            region = _region_of(rules, rule) or {}
            f_qs = float(region["limits"]["force_quasistatic"]["value"])
            tm = region.get("transient_multiplier") or {}

            force = np.asarray(arrays["contact_force"], dtype=float)[:, col]
            ctype = np.asarray(arrays["contact_type"])[:, col]

            # 준정적 접촉은 표 A.2 의 힘을 그대로, 동적 접촉은 그 배수를 쓴다.
            # 배수는 「최소 2배 이상」의 하한이므로(표 A.2 footnote c) 참값은 2 이상이고,
            # 2 를 쓰면 임계가 가장 낮아 초과 판정이 가장 많이 나온다 — 허용 범위 안에서
            # 가장 보수적인 선택이다. 배수를 키우는 쪽이 놓치는 방향이다.
            exceeded = (ctype == "quasistatic") & (force > f_qs)
            if tm.get("not_applicable"):
                # 표준이 이 부위의 동적 칸을 「해당 없음」으로 두는 이유는 접촉
                # 자체가 허용되지 않기 때문이다(5.5.5.3). 그러면 어떤 크기든
                # 성립이고, 비교할 한계가 없는 것이 아니라 한계가 0 인 것이다.
                exceeded |= (ctype == "transient")
            else:
                exceeded |= (ctype == "transient") & (force > f_qs * float(tm["value"]))

            if contact_judged:
                # A contact the recorder could not type is held to the stricter limit -- the
                # smaller of the quasistatic force and the transient multiple, any force where
                # transient contact is not permitted -- from the first contact on.
                strict = 0.0 if tm.get("not_applicable") else f_qs * min(1.0, float(tm["value"]))
                exceeded |= (ctype == "none") & (force > strict) & after_onset

            # 홀드가 걸려 있던 스텝은 쓰지 않는다. 그 스텝의 접촉력에는 더미를 세워
            # 두는 서보 힘이 섞여 있어(recorder 의 hold_contaminated_steps 계약),
            # 「로봇이 그 크기로 때렸다」가 아니다. **해제 직전 스텝 하나**만으로도
            # A2 근거가 생겨 fail/S3 가 실릴 수 있다.
            # The hri judgement's first contact step is the one exception (``judged``, above).
            exceeded &= judged
            if bool(exceeded.any()):
                triggered.append(rid)
            continue

        not_evaluated.append({
            "rule_id": rid,
            "reason": "이 시나리오의 라벨링에 산출식이 구현되어 있지 않음",
        })

    # **넘어졌는데 아무 규칙도 성립하지 않는 경우가 있다.** 이 배포본의 규칙 범위에
    # 안정성 규칙(R-25785-STAB-01)이 없기 때문이다 — S1 계열의 어느 시나리오도 선언하지
    # 않고, 그 규칙의 min_duration 도 정본에서 비어 있다. 그대로 두면 **로봇이 넘어진
    # 궤적이 outcome=nominal 로 실린다.**
    # 판정을 지어내지 않되, 「평가되지 않았다」는 사실은 남긴다.
    reasons = list((meta.get("termination") or {}).get("reason") or [])
    if any(str(r).startswith("fall") for r in reasons) and not triggered:
        not_evaluated.append({
            "rule_id": "R-25785-STAB-01",
            "reason": f"이 에피소드는 낙상으로 끝났으나({', '.join(reasons)}) 이 배포본의 "
                      "규칙 범위에 안정성 규칙이 없어 평가되지 않았음 — outcome 이 "
                      "fail 이 아닌 것은 「넘어지지 않았다」는 뜻이 아니다. 종료 사유는 "
                      "termination 필드에 있다",
        })

    # 접촉이 있었는데 접촉 규칙이 하나도 성립하지 않았다면, 그것이 "한계 안이었다"
    # 인지 "볼 수 없었다" 인지 구분되어야 한다. 유형이 하나도 매겨지지 않았다면
    # 후자다 — 힘은 기록됐는데 어느 열(준정적/동적)과 비교할지 정해지지 않아 비교
    # 자체가 일어나지 않았다.
    # The hri judgement compares untyped force (above), so nothing is left uncompared to list.
    if had_contact and not contact_judged:
        typed = np.asarray(arrays["contact_type"]) != "none"
        onsets = np.flatnonzero(contact)
        # 「에피소드에 분류된 접촉이 하나라도 있는가」로 물으면, 한계 아래의 접촉 하나가
        # 뒤따르는 미분류 접촉을 가립니다(재현: 900 N 이 조용히 사라짐).
        # 물어야 할 것은 **각 접촉 시점에 무언가 분류됐는가** 입니다.
        # 전역 contact_onset 이 선 스텝만 보면, 다른 부위의 접촉이 이어지는 동안 새
        # 부위가 닿아도 새 onset 이 없어 검사되지 않는다(재현: 900 N 이 사라짐).
        #
        # 그렇다고 onset 이후 전 구간을 (스텝, 부위) 쌍으로 세면 실측 1,846건이 되어
        # 읽을 수 없다 — 넘어진 더미의 지면 하중이 대부분이다. 부위별 상승 에지도
        # 242건까지 갔다(힘이 0 을 오르내림).
        #
        # 그래서 **부위별로 한 줄씩** 요약한다. 최대 12줄이고, 어느 부위의 힘이 얼마나
        # 오래 어느 크기까지 분류되지 않았는지가 그대로 드러난다.
        force_all = np.asarray(arrays["contact_force"], dtype=float)
        start = int(onsets[0]) if onsets.size else 0
        per_region: list[tuple[str, int, float]] = []
        names_all = list(meta["contact_region_names"])
        for j, name in enumerate(names_all):
            m = (force_all[start:, j] > 0.0) & ~typed[start:, j]
            if m.any():
                per_region.append((name, int(m.sum()), float(force_all[start:, j][m].max())))
        if per_region:
            # 에피소드 전체 최댓값을 쓰면 안 된다 — 넘어진 더미가 받는 지면 하중이
            # 섞여 로봇이 그 크기로 때린 것처럼 읽힌다(실측: 전체 11,838 N 대
            # 접촉 시점 743 N). 분류되지 않은 접촉 시점의 값만 본다.
            detail = "; ".join(f"{n} {c}스텝 최대 {pk:.0f} N" for n, c, pk in per_region)
            not_evaluated.append({
                "rule_id": "R-TS15066-A2-*",
                "reason": f"첫 접촉(스텝 {start}) 이후 contact_type 이 none 인 채로 힘이 "
                          f"기록된 부위가 있음 — {detail}. 이 힘들은 어떤 한계와도 "
                          "비교되지 않았으므로, outcome 이 fail 이 아니라고 해서 한계 "
                          "안이라는 뜻은 아니다. 넘어진 뒤에는 더미 자신의 지면 하중이 "
                          "여기 섞이므로 크기는 상한으로만 읽을 것",
                # 반올림하지 않는다. 선택 기준은 `force > 0` 인데 소수 첫째 자리로
                # 반올림하면 0.04 N 이 0.0 으로 저장되어, 「양수 힘이 걸렸는데
                # 비교하지 못했다」는 이 항목이 스스로 모순된다 — 검증기가 정당한
                # 데이터를 거부한다. 표시용 반올림은
                # 위 `detail` 문자열이 이미 하고 있다.
                "regions": [{"region": n, "steps": c, "peak_force_n": float(pk)}
                            for n, c, pk in per_region],
            })

    head_face_contact = False
    if contact_judged:
        region_names = list(meta["contact_region_names"])
        cols = [region_names.index(code) for code in HEAD_FACE_REGIONS if code in region_names]
        if cols:
            head = np.asarray(arrays["contact_force"], dtype=float)[:, cols]
            head_face_contact = bool(((head > 0).any(axis=1) & judged & after_onset).any())

    by_id = {r["rule_id"]: r for r in rules.get("rules", [])}
    outcomes = {by_id[r]["outcome"] for r in triggered if r in by_id}
    if "fail" in outcomes:
        outcome = "fail"
        if contact_judged:
            beyond = head_face_contact or any(r.startswith("R-TS15066-A2-") for r in triggered)
            severity = SEVERITY_CONTACT_BEYOND_LIMITS if beyond else SEVERITY_CONTACT_WITHIN_LIMITS
        else:
            severity = SEVERITY_FAIL_WITH_CONTACT if had_contact else SEVERITY_FAIL_NO_CONTACT
    elif "near_miss" in outcomes:
        outcome, severity = "near_miss", SEVERITY_NEAR_MISS
    else:
        outcome, severity = "nominal", SEVERITY_NOMINAL

    label = {
        "outcome": outcome,
        "triggered_rules": sorted(triggered),
        "severity": severity,
        "not_evaluated": not_evaluated,
        # Recorded separately from the outcome, because the two can still
        # disagree. Contact force is compared only once the contact has a type,
        # and the classifier has a blind spot (a foot on a fallen dummy is
        # indistinguishable from ground load). An episode whose contacts never
        # got a type labels nominal without anything having been checked -- the
        # not_evaluated entry above says so, and this flag is what a consumer
        # filters on.
        "contact_observed": had_contact,
        # severity 가 물려받는 사각지대. 로봇 발이 넘어진 더미를 밟는 접촉은
        # contact_onset 에 잡히지 않으므로 그런 에피소드는 S3 가 아니라 S2 로 찍힌다.
        "contact_detection_limits": meta.get("contact_onset_unresolved_reason"),
    }
    if table:
        # Imported here: the evidence module calls this one's selected_rules and limit readers.
        from generation.labeling import evidence

        label[evidence.EVIDENCE_KEY] = evidence.episode_evidence(
            arrays, meta, rules, scenario, cameras,
            windows=(contact, judged, after_onset, contact_judged), fps=round(1.0 / dt))
    return label


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="에피소드 라벨링")
    ap.add_argument("--episodes", default=".tmp/episodes/S1a")
    ap.add_argument("--rules", default="meta/rules.json")
    ap.add_argument("--scenarios", default="meta/scenarios.json")
    ap.add_argument("--cameras", default="meta/cameras.json",
                    help="카메라 정본. 조건표가 있는 시나리오의 근거 요약이 사람이 로봇 카메라에 보인 때를 이것으로 센다 — "
                         "파일이 없으면 그 두 값은 null")
    ap.add_argument("--standard-values", default=None,
                    help="부속서 A 수치 사본. 기본 경로의 사본은 공개 저장소에 없으며, 없으면 meta/rules.json 에 "
                         "이미 실린 값만 쓴다(한계가 비어 있는 규칙은 not_evaluated 로 남긴다)")
    args = ap.parse_args(argv)

    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    values = load_standard_values(Path(args.standard_values) if args.standard_values else None)
    if values:
        merge_standard_values(rules, values)
        print(f"표준 임계값 병합: {len(values)}개 부위 "
              f"({(Path(args.standard_values) if args.standard_values else STANDARD_VALUES_PATH).name})")
    else:
        print("표준 임계값 사본 없음 — meta/rules.json 에 실린 값으로 판정합니다 (정본에도 한계가 비어 있는 규칙은 not_evaluated 로 남습니다)")
    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    by_sid = {s["scenario_id"]: s for s in scenarios["scenarios"]}
    cameras_path = Path(args.cameras)
    cameras = json.loads(cameras_path.read_text(encoding="utf-8")) if cameras_path.is_file() else None

    counts: dict[str, int] = {}
    severities: dict[str, int] = {}
    unevaluated: set[str] = set()
    for npz_path in sorted(Path(args.episodes).glob("*.npz")):
        meta = json.loads(npz_path.with_suffix(".json").read_text(encoding="utf-8"))
        arrays = dict(np.load(npz_path, allow_pickle=True))
        label = label_episode(arrays, meta, rules, by_sid[meta["scenario_id"]], cameras)
        npz_path.with_suffix(".labels.json").write_text(
            json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        counts[label["outcome"]] = counts.get(label["outcome"], 0) + 1
        severities[label["severity"]] = severities.get(label["severity"], 0) + 1
        unevaluated |= {n["rule_id"] for n in label["not_evaluated"]}
    print("outcome  :", counts)
    print("severity :", severities)
    if unevaluated:
        print(f"평가하지 못한 규칙 {len(unevaluated)}종: {sorted(unevaluated)[:4]} …")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
