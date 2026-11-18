#!/usr/bin/env python3
"""
AI 휴머노이드 안전성 데이터셋 검증기
==================================

스키마 정의서 v0.2 §10 의 배포 전 필수 검사 항목을 구현한다.

설계 원칙
---------
1. Isaac Sim / lerobot 설치 없이 동작한다. CI 와 배포 직전 점검 양쪽에서 쓸 수 있어야 한다.
2. 파일이 없으면 죽지 않고 SKIP 으로 기록한다. 부분 완성 데이터셋에도 돌려볼 수 있어야 한다.
3. 모든 지적은 (검사ID, 심각도, 위치, 메시지) 4 요소를 갖는다. 재현 가능해야 고칠 수 있다.
4. 결과를 JSON 으로 떨어뜨린다. 품질검증 결과로 그대로 첨부할 수 있어야 한다.

사용
----
    python validate_dataset.py <데이터셋_루트>
    python validate_dataset.py <루트> --json report.json --strict
    python validate_dataset.py <루트> --checks C2,C4       # 일부만
    python validate_dataset.py <루트> --repro-hook ./rerun_seed.sh   # C9 활성화

종료 코드: 0 통과 / 1 ERROR 존재 / 2 (--strict 시) WARN 존재 / 3 실행 실패
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # 매핑 파일 검사만 건너뛴다
    yaml = None


# ────────────────────────────────────────────────────────────────
# 규격 상수 — 스키마 정의서 v0.2 와 일치해야 한다
# ────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "0.2"

BODY_REGIONS = {f"BR-{i:02d}" for i in range(1, 13)}  # §8

ENUMS: dict[str, set[str]] = {
    "domain": {"baseline", "logistics", "manufacturing", "service", "integrated"},
    "source": {"sim", "real"},
    "robot_embodiment": {"biped", "wheeled"},
    "task_archetype": {
        "transport", "pick_place", "machine_tending", "handover", "locomotion", "hri",
    },
    "hazard_class": {
        "collision", "entrapment", "fall", "drop", "slip", "malfunction", "comm_error",
    },
    "outcome": {"nominal", "near_miss", "fail"},
    # `action` 이 무엇인지 — 위치 목표(rad)인가 토크(N·m)인가. 이 표기가 없으면
    # 소비자가 열의 단위를 알 수 없다.
    "action_space": {"position_rad", "torque_nm"},
    "severity": {"S0", "S1", "S2", "S3"},
    "safety_function": {
        "none", "protective_stop", "ssm", "pfl", "position_hold", "mode_switch",
    },
    "measurement_method": {
        "physx", "newton", "ft_sensor", "dummy_active", "motion_capture",
    },
    # Which kind of row of its scenario's conditions table an episode is (a column of the table scenarios).
    "condition_design": {"random", "grid"},
}

# §6 observation.contact.type — Table A.2 는 과도적/준정적 한계를 따로 규정하므로
# 어느 쪽을 적용할지 데이터가 말해 주어야 한다.
CONTACT_TYPES = {"none", "transient", "quasistatic"}

SAFETY_EVENTS = {
    "protective_stop_triggered", "ssm_violation", "pfl_limit_exceeded", "mode_switch",
    "contact_onset", "fall_detected", "object_dropped", "comm_timeout",
    "sensor_fault_injected", "rack_contact",
}

# source 값에 따라서만 필수인 필드 (§5). REQUIRED_EPISODE_FIELDS 는 무조건 필수만
# 담으므로 이쪽을 따로 둔다 -- 정본(meta/safety_schema.json)이 이 구분을 그대로
# 실어야 스키마만 읽는 소비자가 검증기와 같은 판단을 한다.
CONDITIONAL_REQUIRED_EPISODE_FIELDS = {
    "sim": ("randomization_seed", "sim_backend", "run_start_seq"),
}

# The episode columns required of an episode whose scenario declares a conditions table (the key: the scenario's
# entry in the release's meta/scenarios.json carries it) -- the spec of the episode, the evidence of its label and
# its place in the table (§5, the project's own). The canon is meta/safety_schema.json's
# scenario_conditional_episode_fields, which C1 compares with this whole; generation/field_map.py holds the same
# list (its fields whose condition is "conditions"), and tests/generation/test_field_map.py holds it to the canon.
TABLE_CONDITION = "conditions"
EVIDENCE_PREFIX = "evidence."
_EVIDENCE_NAMES = ("min_separation_margin_m", "first_contact_s",
                   *(f"force_ratio.{code}" for code in sorted(BODY_REGIONS)), "force_ratio_max",
                   "human_first_seen_s", "human_last_seen_s")
SCENARIO_CONDITIONAL_EPISODE_FIELDS = {
    TABLE_CONDITION: ("episode_spec", *(f"{EVIDENCE_PREFIX}{name}" for name in _EVIDENCE_NAMES),
                      "condition_design", "grid_cell_robot", "grid_cell_human"),
}
# The members of episode_spec (§5): the fixed test conditions and the part a developer swaps.
EPISODE_SPEC_MEMBERS = ("world", "humans", "task", "robot", "condition", "repro")
# The evidence value every episode has: each step records the distance and the required separation, so their
# smallest difference is never missing. The others are null where there is nothing to give (spec §5).
EVIDENCE_NEVER_NULL = (f"{EVIDENCE_PREFIX}min_separation_margin_m",)
# The cell number both grid columns of a random row carry -- the same value as generation/convert_lerobot.py's
# GRID_NOT_A_CELL (the validator does not import generation/; tests/generation/test_convert_lerobot.py holds the two
# together).
GRID_NOT_A_CELL = -1

# The generator's commit in sim_backend: a full SHA-1 name in lower case (fullmatch). The same
# pattern as generation/provenance.COMMIT_RE -- the validator does not import generation/, and
# tests/generation/test_provenance.py holds the two together.
GENERATOR_COMMIT_RE = re.compile(r"[0-9a-f]{40}")

# source 와 모순되는 계측 방법. sim 데이터가 실측 장비를 썼다고 말하거나 그 반대면
# 둘 중 하나가 틀린 것이다.
FORBIDDEN_MEASUREMENT_METHODS = {
    "sim": ("ft_sensor", "dummy_active", "motion_capture"),
    "real": ("physx", "newton"),
}

# 규격서 §3 의 식별 체계. 형식만 보는 것이 아니라 안에 담긴 두 값을 같은 행의
# 컬럼과 대조한다 — 형식만 맞고 시드가 다른 id 는 "파일명으로 어느 시드인지 알 수
# 있다" 는 §3 의 주장을 깨면서도 정규식은 통과한다.
#
# 이 검사가 없던 동안 규격서 §3 은 실제로 만들어지는 것과 **다른 형식**을 싣고
# 있었다(`{domain}-{scenario_id}-{robot_model}-{seed}-{seq:05d}`). 아무것도 대조하지
# 않으니 아무도 몰랐다. 이 검사가 못 보는 것: 일련번호가 맞는지 — 에피소드 메타에
# 그 값을 담는 컬럼이 없다.
# C9 가 표본을 고를 접두 범위(실행 안에서의 위치). 재현은 그 에피소드를 만든 실행을 첫 번호부터
# 그 에피소드까지 다시 생성하므로 비용이 그 위치에 비례한다 — 한 실행으로 만든 400건 배포본에서
# 380번을 고르면 3시간이 넘고 훅 타임아웃에 걸린다.
REPRO_MAX_SEQ = 20

# The time one C9 hook call gets, in seconds; a hook still running then is reported as failed.
REPRO_HOOK_TIMEOUT_S = 1800
# After SIGTERM, how long the hook's process group gets to clean up (the hook's EXIT trap, the
# simulator's own shutdown) before SIGKILL. A group that ignores SIGTERM costs this much on top.
REPRO_KILL_GRACE_S = 30.0
# How long to keep collecting the hook's output once its group has been ended. The pipes are also
# held by anything that left the group (setsid), so waiting for them to close is bounded.
REPRO_COLLECT_TIMEOUT_S = 5.0
# A hook that stalls: nothing written under its TMPDIR for REPRO_STALL_S, not judged during the first
# REPRO_STALL_GRACE_S (boot and scene build). The repository hook writes its rerun under the TMPDIR this
# validator hands it: an S1d rerun writes a lossless stream per camera as it steps (the longest gap on a real
# sample, 2026-10-06 with six cameras: 9.8 s), and an S1a rerun, which has no cameras, writes each episode's files when the
# episode ends (every 30.4 s, at most 35.4 s, over the 400 saves of the S1a batch on 2026-08-25).
# So a directory nothing is written to is a run that hangs -- Isaac stalled on 3 attempts of 43 in the S1d-400
# batch, and one C9 sample on 2026-10-06 waited the whole REPRO_HOOK_TIMEOUT_S. The values are the batch
# runner's (generation/run_batch.py STALL_AFTER_S · STALL_GRACE_S), which held on those 43 attempts
# without a false stall; this validator does not import generation/, and a test compares the copies. A
# stalled sample is run again, REPRO_STALL_ATTEMPTS times in all -- a stall says nothing about the data.
REPRO_STALL_S = 300.0
REPRO_STALL_GRACE_S = 480.0
REPRO_STALL_POLL_S = 30.0
REPRO_STALL_ATTEMPTS = 3
# The prefix bound for a camera distribution (one that ships meta/cameras.json), derived so the
# slowest prefix it allows still fits the hook's time. The default above is not: a camera rerun
# renders and encodes every camera of the canon each step, and 21 camera episodes do not fit 1800 s.
# Per 1,000-step camera episode, measured 2026-10-22 on the development machine with the seven RGB cameras of the
# canon (42 rows of S1e, S1f and S1g that ran all 1,000 steps, one process a row): stepping 143.5-180.5 s, save
# 8.4-14.0 s -- closing the seven lossless videos and writing the npz; there is no depth/mask pack to compress. The
# median row took 158.7 s and the cost is the slowest, 190.4 s (176.4 + 14.0), rounded up to the second. The share of
# the hook's time below is what leaves room for a run slower than the rows measured: the prefix this bound allows,
# every episode as slow as the slowest row, still ends inside the hook's whole time (test_c9_repro_cost). With other
# work on the PC a rerun is slower, so C9 is a run for a quiet machine.
REPRO_CAMERA_EPISODE_S = 191.0
# Boot, shutdown and the comparison, per hook call. Measured: 17-20 s per row outside its stepping and its
# save (a row process's wall time minus the two, shader cache warm: 17.2-20.3 s over the 60 rows of 2026-10-22).
# The rest is room for a cold shader cache, which was not measured.
REPRO_OVERHEAD_S = 120.0
# The share of the hook's time the slowest prefix may take.
REPRO_TIMEOUT_SHARE = 0.75
# The hook reruns the episode's run from its first seq up to the episode: at most max_seq + 1 episodes
# for an episode at place max_seq of its run.
REPRO_MAX_SEQ_CAMERAS = int((REPRO_HOOK_TIMEOUT_S * REPRO_TIMEOUT_SHARE - REPRO_OVERHEAD_S)
                            // REPRO_CAMERA_EPISODE_S) - 1

EPISODE_ID_RE = re.compile(r"^(?P<scenario>[A-Za-z0-9]+)_(?P<seq>\d{4,})_seed(?P<seed>\d+)$")

# 어느 JSON 필드가 어떤 그릇인가. 「dict 또는 list」로 뭉뚱그리면 termination="[]"
# 이 통과한다 — 종료 사유가 통째로 비어 있는데 오류가 안 났다.
# 정본은 meta/safety_schema.json 의 episode_json_shapes 이고 C1 이
# 이 상수와 대조한다.
EPISODE_JSON_SHAPES = {
    "termination": "object",
    "notes": "object",
    "scene_assets": "array",
    "sim_backend": "object",
    "episode_spec": "object",
}

# notes 안의 계약. 새 provenance 필드를 실어 놓고 아무도 대조하지 않아, 변환기에서
# 지워도 통과했다 — 남은 여덟 필드 때문에 「비어 있지 않다」가 성립했다.
# "|null" 은 그 값이 없을 수 있다는 선언이다.
NOTES_MEMBERS = {
    "provisional": "boolean",
    "not_evaluated": "array",
    "contact_observed": "boolean",
    "contact_detection_limits": "string|null",
    "hold_contaminated_steps": "integer",
    "hold_release_step": "integer",
    "ssm_params": "object",
    "human_region_radius": "number",
    "contact_onset_unresolved_regions": "array",
    "contact_onset_unresolved_reason": "string|null",
    "contact_type_unresolved_reason": "string|null",
    "imu_link": "string",
    "asset_paths_resolved": "object",
    "randomization": "object",
    "task_completed_step": "integer|null",
    # The three the hri recorder keeps beside the step table; null where a scenario has no script.
    "degenerate_reason": "string|null",
    "box_released_step": "integer|null",
    "camera_in_box_steps": "object|null",
    # What the hri recorder keeps of the task itself: the stages it ends in, the person's stature and the
    # task's time limit; null where a scenario has no script.
    "task_stages": "array|null",
    "human_stature_m": "number|null",
    "task_time_limit_s": "number|null",
}

_JSON_TYPES = {
    "object": dict, "array": list, "string": str,
    "boolean": bool, "integer": int, "number": (int, float),
}


def json_type_ok(value, spec: str) -> bool:
    """`"integer|null"` 같은 선언에 값이 맞는지.

    The declaration comes from the shipped schema; one that is not a string declares nothing a value
    can match (C1 says it differs from the validator's)."""
    if not isinstance(spec, str):
        return False
    allowed = spec.split("|")
    if value is None:
        return "null" in allowed
    for name in allowed:
        want = _JSON_TYPES.get(name)
        if want is None:
            continue
        # bool 은 int 의 하위형이라 그냥 두면 True 가 integer 로도 통과한다.
        if isinstance(value, bool) != (name == "boolean"):
            continue
        if isinstance(value, want):
            # NaN·inf 는 float 이지만 「수치」가 아니다. 회랑을 재현하는 입력이
            # NaN 이어도 통과했다.
            if name in ("number", "integer") and not math.isfinite(value):
                return False
            return True
    return False


def task_record_problems(notes: dict, required: bool) -> list[str]:
    """notes.task_stages, notes.human_stature_m and notes.task_time_limit_s.

    A scenario with a completion criterion (`required`) names the stages the task ends in (a non-empty
    list), the person's stature and the time limit (finite positive numbers); another scenario may leave
    all three null. A stage is done exactly when its end_step is a step (not -1). Only shapes are held
    here: the member type itself is said by C5's member check, so a wrong container is not said twice."""
    out: list[str] = []
    stages = notes.get("task_stages")
    if stages is None and required:
        out.append(f"notes.task_stages 가 null 이거나 없음 — {DIRECTED_SSM_RULE_ID} 를 선언한 시나리오는 "
                   "과제의 단계가 있으므로 단계마다 끝난 스텝을 적어야 한다")
    elif isinstance(stages, list):
        if not stages and required:
            out.append(f"notes.task_stages 가 비어 있음 — {DIRECTED_SSM_RULE_ID} 를 선언한 시나리오의 과제는 단계가 하나 이상이다")
        for i, stage in enumerate(stages):
            if not isinstance(stage, dict):
                out.append(f"notes.task_stages[{i}] 가 객체가 아님")
                continue
            for key, spec in NOTES_MEMBER_SHAPES["task_stages[]"].items():
                where = f"notes.task_stages[{i}].{key}"
                floor = None
                if ">=" in spec:
                    spec, floor = spec.split(">=")
                if key not in stage:
                    out.append(f"{where} 없음")
                elif not json_type_ok(stage[key], spec):
                    out.append(f"{where} 이 {spec} 이 아님: {stage[key]!r:.40}")
                elif floor is not None and stage[key] < int(floor):
                    out.append(f"{where} 이 {floor} 보다 작음: {stage[key]}")
                elif spec == "string" and not stage[key].strip():
                    out.append(f"{where} 이 비어 있음")
            done, end = stage.get("completed"), stage.get("end_step")
            if isinstance(done, bool) and isinstance(end, int) and not isinstance(end, bool) \
                    and done != (end >= 0):
                out.append(f"notes.task_stages[{i}] 의 completed({done}) 가 end_step({end}) 과 어긋남 — "
                           "단계는 끝난 스텝이 있을 때만(end_step 0 이상) 끝난 것이다")
    for name in ("human_stature_m", "task_time_limit_s"):
        value = notes.get(name)
        if value is None:
            if required:
                out.append(f"notes.{name} 이 null 이거나 없음 — {DIRECTED_SSM_RULE_ID} 를 선언한 시나리오는 "
                           "유한한 양수를 적어야 한다")
        elif isinstance(value, (int, float)) and not isinstance(value, bool) \
                and not (math.isfinite(value) and value > 0):
            out.append(f"notes.{name} 이 유한한 양수가 아님: {value!r}")
    return out


def notes_problems(notes: dict, regions: Iterable[str], steps=None,
                   carried: dict[str, float] | None = None,
                   ssm_names: tuple[str, ...] | None = None,
                   stop_table: dict | None = None,
                   completion_required: bool = False) -> list[str]:
    """얕은 형만 보면 provenance 가 **무의미해도** 통과한다.

    `ssm_params={}` · 잘못된 부위 코드 · 음수 스텝 수가 전부 오류 0이었다.
    이 값들은 받는 쪽이 회랑과 미평가 사유를
    재현하는 근거이므로 비었으면 없느니만 못하다.

    `completion_required`: the episode's scenario declares R-TS15066-SSM-02, so it has a completion
    criterion and notes.task_completed_step is an integer (null would read as "no criterion").
    """
    out: list[str] = []
    known = set(regions)

    ssm = notes.get("ssm_params")
    if isinstance(ssm, dict):
        for name in (SSM_REQUIRED_IN_NOTES if ssm_names is None else ssm_names):
            if name not in ssm:
                out.append(f"notes.ssm_params.{name} 없음 — 회랑을 재현할 수 없다")
            elif not isinstance(ssm[name], (int, float)) or isinstance(ssm[name], bool) \
                    or not math.isfinite(ssm[name]) or ssm[name] < 0:
                out.append(f"notes.ssm_params.{name} 이 유한한 음이 아닌 수가 아님: {ssm[name]!r}")
            elif name in (carried or {}) and not math.isclose(
                    float(ssm[name]), carried[name],
                    rel_tol=SSM_VALUE_TOLERANCE, abs_tol=SSM_VALUE_TOLERANCE):
                out.append(
                    f"notes.ssm_params.{name}({ssm[name]!r}) 이 정본의 값"
                    f"({carried[name]!r})과 다름 — 회랑은 이 값으로 구워졌으므로 "
                    "받는 쪽이 rules.json 만 보고 재현하면 어긋난다")
        if stop_table is not None:
            if "stop_table" not in ssm:
                out.append("notes.ssm_params.stop_table 없음 — 정지에 필요한 거리를 재현할 수 없다")
            # The values only: the table's method note is a sentence (is_prose_key), and rewording
            # it after a batch is generated changes no S_p.
            elif bools_apart(without_prose(ssm["stop_table"])) != bools_apart(without_prose(stop_table)):
                out.append("notes.ssm_params.stop_table 이 정본의 R-TS15066-SSM-02.stop_table 과 다름 — "
                           "설명 문장 밖의 값이 다르다. S_p 는 이 표로 셈했으므로 받는 쪽이 rules.json 만 보고 "
                           "다시 세면 어긋난다")

    radius = notes.get("human_region_radius")
    if isinstance(radius, (int, float)) and not isinstance(radius, bool) and radius <= 0:
        out.append(f"notes.human_region_radius 가 양수가 아님: {radius!r}")

    # hold_release_step 은 -1 이 「끝까지 해제되지 않음」이다 (recorder.py 의 초기값).
    # 400건 중 312건이 그렇다 — 로봇이 더미 0.10 m 안에 들어오지 않은 에피소드다.
    # 그래서 이 표식을 결함으로 세지 않는다 — 뜻은 정본의 notes_bounds 가 적는다.
    release = notes.get("hold_release_step")
    if isinstance(release, int) and not isinstance(release, bool):
        if release < HOLD_NEVER_RELEASED:
            out.append(f"notes.hold_release_step 이 {HOLD_NEVER_RELEASED} 보다 작음: {release}")
        elif steps is not None and release > steps:
            out.append(f"notes.hold_release_step 이 에피소드 길이({steps})를 넘음: {release}")
    # 생성기 계약은 「0 또는 1」이다 — 해제 게이트가 최선노력이고 접촉 백스톱이
    # 손해를 접촉을 처음 보는 스텝 하나로 묶기 때문이다(recorder.py 주석). 그보다
    # 크면 홀드가 접촉을 관통했다는 뜻이고, 그 스텝들의 접촉력은 쓸 수 없다.
    contaminated = notes.get("hold_contaminated_steps")
    if isinstance(contaminated, int) and not isinstance(contaminated, bool):
        if contaminated < 0:
            out.append(f"notes.hold_contaminated_steps 이 음수: {contaminated}")
        elif contaminated > HOLD_CONTAMINATION_MAX:
            out.append(
                f"notes.hold_contaminated_steps 이 {contaminated} — 생성기 계약은 "
                f"{HOLD_CONTAMINATION_MAX} 이하다. 홀드가 접촉을 관통했다는 뜻이므로 "
                "그 스텝의 접촉력은 서보 힘과 섞여 있다")

    # The step the task was done at, the not-done mark, or null for no criterion (notes_bounds).
    done = notes.get("task_completed_step")
    if done is None and completion_required:
        out.append(f"notes.task_completed_step 이 null 이거나 없음 — {DIRECTED_SSM_RULE_ID} 를 선언한 시나리오는 "
                   f"완료 기준이 있으므로 완료한 스텝이나 {TASK_NOT_COMPLETED}(시간 안에 끝내지 못함)이어야 한다")
    if isinstance(done, int) and not isinstance(done, bool):
        if done < TASK_NOT_COMPLETED:
            out.append(f"notes.task_completed_step 이 {TASK_NOT_COMPLETED} 보다 작음: {done}")
        elif steps is not None and done >= steps:
            out.append(f"notes.task_completed_step 이 에피소드 길이({steps}) 안의 스텝이 아님: {done}")

    out.extend(task_record_problems(notes, completion_required))

    # A member of the wrong shape is said by C5's shape check (notes_member_shapes); here only
    # the lists are walked.
    unresolved = notes.get("contact_onset_unresolved_regions")
    bad = [x for x in (unresolved if isinstance(unresolved, list) else []) if str(x) not in known]
    if bad:
        out.append(f"notes.contact_onset_unresolved_regions 에 없는 부위 코드: {bad[:3]}")

    not_evaluated = notes.get("not_evaluated")
    for i, item in enumerate(not_evaluated if isinstance(not_evaluated, list) else []):
        if not isinstance(item, dict):
            out.append(f"notes.not_evaluated[{i}] 가 객체가 아님")
            continue
        # 정본이 string/string/array 라고 선언하고 문서가 「C5 가 강제한다」고 적는데
        # 정작 str() 로 강제 변환하면 rule_id=7 이 통과한다.
        # 정본 형태를 그대로 순회한다.
        for key, spec in NOTES_MEMBER_SHAPES["not_evaluated[]"].items():
            optional = spec.endswith("?")
            spec = spec[:-1] if optional else spec
            if key not in item:
                if not optional:
                    out.append(f"notes.not_evaluated[{i}].{key} 없음")
            elif not json_type_ok(item[key], spec):
                out.append(
                    f"notes.not_evaluated[{i}].{key} 이 {spec} 이 아님: {item[key]!r:.40}")
            elif spec == "string" and not str(item[key]).strip():
                out.append(f"notes.not_evaluated[{i}].{key} 이 비어 있음")
        # 한 층 아래도 본다. 이 목록이 「어느 부위에 얼마가 걸렸는데 비교하지
        # 못했다」를 말하는 자리이므로, 값이 거짓이면 미평가 사유 자체가 거짓이 된다.
        # 값이 개별로 유효해도 **서로 모순**이면 provenance 가 거짓이다. 생성기는
        # 접촉이 관측된 경우에만 이 목록을 만들고, 힘이 0 을 넘은 스텝만 센다
        # (label_episodes). 그러니 목록이 있으면 contact_observed 가 참이고
        # steps>=1 · peak_force_n>0 이어야 한다.
        listed = item.get("regions")
        regions = listed if isinstance(listed, list) else []
        # 「선택」을 일괄로 만들었더니 미분류 힘 항목이 출처·크기 없이 통과했다.
        # 생산자의 집계 경로(`R-…-A2-*`)는 미분류 힘이 실제로 있을 때만 돌고
        # 반드시 부위를 하나 이상 적는다.
        if str(item.get("rule_id", "")).endswith("*") and not listed:
            out.append(
                f"notes.not_evaluated[{i}] 은 집계 항목인데 regions 가 비어 있음 — "
                "어느 부위에 얼마가 걸렸는지 없이 「비교하지 못했다」만 남는다")
        if regions and notes.get("contact_observed") is False:
            out.append(
                f"notes.not_evaluated[{i}] 에 부위가 있는데 contact_observed 가 거짓 — "
                "접촉이 관측되지 않았다면 비교하지 못한 부위도 없다")
        seen_codes = set()
        for j, reg in enumerate(regions):
            where = f"notes.not_evaluated[{i}].regions[{j}]"
            if not isinstance(reg, dict):
                out.append(f"{where} 가 객체가 아님")
                continue
            code = str(reg.get("region", ""))
            if code not in known:
                out.append(f"{where}.region 이 없는 부위 코드: {code!r}")
            elif code in seen_codes:
                out.append(f"{where}.region 이 중복: {code}")
            seen_codes.add(code)
            n = reg.get("steps")
            if not isinstance(n, int) or isinstance(n, bool) or n < 1:
                out.append(f"{where}.steps 가 1 이상의 정수가 아님: {n!r}")
            elif steps is not None and n > steps:
                out.append(f"{where}.steps 가 에피소드 길이({steps})를 넘음: {n}")
            f = reg.get("peak_force_n")
            if not isinstance(f, (int, float)) or isinstance(f, bool) \
                    or not math.isfinite(f) or f <= 0:
                out.append(f"{where}.peak_force_n 이 유한한 양수가 아님: {f!r}")
    return out


# C9 훅을 부를 때 쓰는 PATH. 호출자 것을 그대로 넘기면 가짜 bash 로 훅을 무력화할
# 수 있다.
_SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# notes.hold_release_step 의 「끝까지 해제되지 않음」 표식. recorder 의 초기값이다.
HOLD_NEVER_RELEASED = -1

# 홀드가 접촉과 겹칠 수 있는 최대 스텝 수. 접촉 백스톱이 「접촉을 처음 보는 스텝」
# 하나로 묶으므로 0 또는 1 이다 (generation/envs/s1a/recorder.py).
HOLD_CONTAMINATION_MAX = 1

# notes.task_completed_step's "not done in time" mark; null is a scenario without a completion criterion.
TASK_NOT_COMPLETED = -1

# notes.ssm_params 가 담아야 할 이름. SSM_PARAMETER_NAMES 와 같아야 하는데,
# 여기 것은 「배포본이 실은 값」이고 저기 것은 「생성기가 읽는 값」이다.
SSM_REQUIRED_IN_NOTES = (
    "human_speed", "robot_reaction_time", "robot_stop_time",
    "intrusion_distance", "position_uncertainty_robot", "position_uncertainty_human",
)

# notes.ssm_params of a scenario that declares R-TS15066-SSM-02: that rule's five parameters, and
# its stop table under "stop_table" -- the episode's S_p column was computed with them.
DIRECTED_SSM_IN_NOTES = (
    "robot_reaction_time", "intrusion_distance", "position_uncertainty_robot",
    "position_uncertainty_human", "approach_speed_threshold",
)

# Keys whose text is written for a person to read: a member ending in "_note", or one of these
# names. A copy of the batch assembly's convention (generation/assemble_batch.py _PROSE_KEYS), which
# takes a change under such a key for a rewording and ships the new sentence over episodes recorded
# before it -- so a comparison of a shipped canon block with an episode's copy leaves them out. A
# test holds the two to one reading.
PROSE_KEYS = frozenset({
    "note", "description", "editing_policy", "reason", "condition", "objective", "human_behavior",
    "layout", "open_issues", "seed_rule", "common_randomization",
    "unused_in_v0_1", "asset_policy", "why", "what", "how_to_update"})


def is_prose_key(key: object) -> bool:
    return isinstance(key, str) and (key.endswith("_note") or key in PROSE_KEYS)


def without_prose(value: Any) -> Any:
    """A parsed JSON value with every object member under a prose key (is_prose_key) left out."""
    if isinstance(value, dict):
        return {k: without_prose(v) for k, v in value.items() if not is_prose_key(k)}
    if isinstance(value, list):
        return [without_prose(v) for v in value]
    return value


def bools_apart(value: Any) -> Any:
    """A parsed JSON value whose true and false no longer equal 1 and 0 (as they do in Python), so
    that == compares JSON values -- while 2 and 2.0 stay one number."""
    if isinstance(value, dict):
        return {k: bools_apart(v) for k, v in value.items()}
    if isinstance(value, list):
        return [bools_apart(v) for v in value]
    if isinstance(value, bool):
        return ("bool", value)
    return value

# 회랑은 생성 시점에 궤적으로 구워지고, 그때 쓴 입력이 notes.ssm_params 에 남는다.
# 정본이 같은 파라미터에 **값을 실으면 사본이 둘**이 되므로 둘이 같은지 봐야 한다.
# 대조하지 않으면 정본의 human_speed 1.6 에 에피소드 9.9 를 넣어도 검증기가 조용하다.
#
# **값이 없는 항목은 비교하지 않는다.** 정본이 조항 참조만 싣는 항목이 바로
# ssm_params 가 존재하는 이유이므로, 거기서 다르다고 우는 것은 계약을 뒤집는 것이다.
SSM_VALUE_TOLERANCE = 1e-9


def carried_ssm_values(rules: dict | None, rule_id: str = "R-TS15066-SSM-01") -> dict[str, float]:
    """정본의 SSM 규칙(기본 SSM-01)에서 **값이 실린** 파라미터만 이름→값으로 돌려준다."""
    if not isinstance(rules, dict):
        return {}
    for rule in json_member(rules, "rules", list):
        if isinstance(rule, dict) and rule.get("rule_id") == rule_id:
            params = rule.get("parameters")
            if not isinstance(params, dict):
                return {}
            return {name: float(entry["value"])
                    for name, entry in params.items()
                    if isinstance(entry, dict)
                    and isinstance(entry.get("value"), (int, float))
                    and not isinstance(entry.get("value"), bool)}
    return {}

# notes 안쪽 구조와 경계값. 검증기 상수에만 있으면 **배포본을 받은 쪽이 새 계약을
# 알 수 없다** — 정본은 「받은 쪽이 읽는 규격」이다.
NOTES_MEMBER_SHAPES = {
    "randomization": {
        "note": "축 이름은 시나리오·구현에 따라 다르므로 여기서 고정하지 않는다. "
                "`_drawn` 은 뽑힌 값, `_realized` 는 시뮬레이션이 실제로 낸 값이다.",
    },
    # regions 는 **선택**이다. 라벨러는 한계값 미확보·없는 부위·미구현 규칙에서
    # {rule_id, reason} 만 낸다. 필수로 두었더니 정상 산출물을 거부했다.
    "not_evaluated[]": {"rule_id": "string", "reason": "string", "regions": "array?"},
    "not_evaluated[].regions[]": {
        "region": "body_region_code", "steps": "integer>=1", "peak_force_n": "number>0",
    },
    # A stage of the task: -1 and null mean the stage did not end.
    "task_stages[]": {
        "name": "string", "completed": "boolean", "end_step": "integer>=-1", "duration_s": "number|null",
    },
}
NOTES_BOUNDS = {
    "hold_contaminated_steps": {"min": 0, "max": HOLD_CONTAMINATION_MAX,
                                "note": "생성기 계약 — 접촉 백스톱이 접촉을 처음 보는 스텝 하나로 묶는다"},
    "hold_release_step": {"sentinel": HOLD_NEVER_RELEASED,
                          "sentinel_meaning": "끝까지 해제되지 않음"},
    "task_completed_step": {"sentinel": TASK_NOT_COMPLETED,
                            "sentinel_meaning": "시간 안에 과제를 끝내지 못함 — null 은 완료 기준이 없는 시나리오"},
}

REQUIRED_EPISODE_FIELDS = [
    "episode_id", "schema_version", "domain", "source", "robot_model",
    "robot_embodiment", "dof", "joint_names", "scenario_id", "task_archetype",
    "hazard_class", "outcome", "severity", "triggered_rules", "safety_function",
    "standard_refs", "measurement_method", "scene_assets", "fps", "duration_s",
    "action_space", "termination", "degenerate",
    # 규격서 §5 가 필수라고 적는다. 카드가 소비자에게 이걸
    # 읽으라고 지시하므로, 없애는 회귀가 조용히 통과해서는 안 된다.
    "notes",
]

# 스텝 시계열에서 관절 차원을 가져야 하는 필드
JOINT_DIM_FIELDS = [
    "observation.state.joint_pos",
    "observation.state.joint_vel",
    "observation.state.joint_torque",
    "action",
]

# Step fields every release carries (spec §6): what the judgement reads, whatever the robot -- the
# body's pose and speed, the contact with the person, the distance and the events.
REQUIRED_STEP_FIELDS = [
    "timestamp", "frame_index", "observation.root_pose", "observation.root_vel",
    "observation.contact.force", "observation.contact.region_names", "observation.contact.type",
    "safety.min_distance_human", "safety.separation_required", "safety.separation_violation", "safety.event",
]
# Step fields the schema allows without requiring them (spec §6): the robot's own columns, whose shape
# differs from robot to robot (joints, action, IMU, centre of mass, ZMP, stability margin), and the
# columns no judgement rule reads (the contact relative speed; the contact pressure, measured
# contact area -- real hardware only). A present one gets every check a required one gets; one that a
# rule the release's scenarios declare reads is required by C8's declared inputs all the same. C1
# compares this with the shipped schema's optional_step_fields.
OPTIONAL_STEP_FIELDS = (
    "observation.state.joint_pos", "observation.state.joint_vel", "observation.state.joint_torque", "action",
    "observation.imu", "observation.contact.v_rel", "safety.com", "safety.zmp", "safety.stability_margin",
    "observation.contact.pressure",
)

# Step fields required only under a condition (§6 「조건부 필드 — 카메라」 · 「조건부 필드 — 사람과 로봇」) -- the step-side
# twin of CONDITIONAL_REQUIRED_EPISODE_FIELDS. "cameras" means the distribution ships
# meta/cameras.json; the same string keys meta/safety_schema.json and
# generation/field_map.py StepField.condition. When the condition holds, C1 requires each
# observation.images.* field as a LeRobot video feature (info.json + videos/ -- not a
# parquet column) and C8 requires the others as parquet columns. C1 also compares this
# constant with the shipped schema.
CONDITIONAL_REQUIRED_STEP_FIELDS = {
    "cameras": (
        "observation.images.head_rgb",
        "observation.images.head_down_rgb",
        "observation.images.left_wrist_rgb",
        "observation.images.right_wrist_rgb",
        "observation.images.rear_rgb",
        "observation.images.third_person_rgb",
        "observation.images.cctv_rgb",
        "observation.bbox2d.human",
        "observation.camera_pose.head_rgb",
        "observation.camera_pose.head_down_rgb",
        "observation.camera_pose.left_wrist_rgb",
        "observation.camera_pose.right_wrist_rgb",
        "observation.camera_pose.rear_rgb",
        "observation.camera_pose.third_person_rgb",
        "observation.camera_pose.cctv_rgb",
    ),
    # "hri": a scenario of the distribution declares R-TS15066-SSM-02. Every field is a parquet
    # column; the visible fractions follow the robot's own cameras in meta/cameras.json (type mounted).
    "hri": (
        "observation.command_velocity",
        "observation.human.root_pose",
        "observation.human.root_vel",
        "observation.human_visible_fraction.head_rgb",
        "observation.human_visible_fraction.head_down_rgb",
        "observation.human_visible_fraction.left_wrist_rgb",
        "observation.human_visible_fraction.right_wrist_rgb",
        "observation.human_visible_fraction.rear_rgb",
        "safety.approach_speed_robot",
        "safety.approach_speed_human",
    ),
}

FIXED_SHAPES = {  # 필드 → 기대 길이
    "observation.root_pose": 7,
    "observation.root_vel": 6,
    "observation.imu": 10,
    "safety.com": 3,
    "safety.zmp": 2,
    # Camera fields (conditional above). C8 checks a length only where the column
    # exists, so a camera-less distribution (v0.1) is untouched by these.
    "observation.bbox2d.human": 4,
    "observation.camera_pose.head_rgb": 7,
    "observation.camera_pose.head_down_rgb": 7,
    "observation.camera_pose.left_wrist_rgb": 7,
    "observation.camera_pose.right_wrist_rgb": 7,
    "observation.camera_pose.rear_rgb": 7,
    "observation.camera_pose.third_person_rgb": 7,
    "observation.camera_pose.cctv_rgb": 7,
    # hri fields (conditional above): the walking command (vx, vy, yaw rate) and the person's pelvis.
    "observation.command_velocity": 3,
    "observation.human.root_pose": 7,
    "observation.human.root_vel": 6,
}

# ── Camera media: checked only when the distribution ships meta/cameras.json ──
CAMERAS_CONDITION = "cameras"
HRI_CONDITION = "hri"
VISIBLE_FRACTION_PREFIX = "observation.human_visible_fraction."
# LeRobot names a camera stream observation.images.<camera> and keeps it under
# videos/<field>/ (spec §4); <camera> is a key of meta/cameras.json "cameras".
VIDEO_FIELD_PREFIX = "observation.images."
VIDEO_DIR = "videos"
# The MP4 (ISO-BMFF) signature: a file's first box is ftyp, its type at bytes 4..8 after the
# 4-byte box size -- LeRobot 0.6.1's streams open "\0\0\0 ftypisom". C1
# reads these eight bytes of a stream and nothing else: a signature, not a decode.
MP4_SIGNATURE_AT = 4
MP4_SIGNATURE = b"ftyp"
# The dummy's box in its camera's picture (meta/cameras.json bbox.camera): pixel indices
# inclusive at both ends, or the not-visible marker (spec §6, bbox.coords · bbox.invalid).
BBOX_FIELD = "observation.bbox2d.human"
BBOX_INVALID = (-1, -1, -1, -1)
# The layer of every step field (spec §6 「층 — 로우와 가공」): raw is what the simulator and its
# sensors gave, processed what the judgement computed from it. One step table, names unchanged;
# C1 compares this with the shipped schema's step_field_layers whole, and C8 asks every column
# of the step files for a layer. LeRobot's own index columns have none.
LAYERS = ("raw", "processed")
LEROBOT_STEP_COLUMNS = frozenset({"index", "episode_index", "task_index"})
_HRI_RAW = ("observation.command_velocity", "observation.human.root_pose", "observation.human.root_vel")
STEP_FIELD_LAYERS = {
    **dict.fromkeys(("timestamp", "frame_index", "observation.state.joint_pos",
                     "observation.state.joint_vel", "observation.state.joint_torque",
                     "observation.root_pose", "observation.root_vel", "observation.imu",
                     "observation.contact.force", "observation.contact.pressure",
                     "observation.contact.region_names", "action"), "raw"),
    **dict.fromkeys(("observation.contact.v_rel", "observation.contact.type", "safety.com",
                     "safety.zmp", "safety.stability_margin", "safety.min_distance_human",
                     "safety.separation_required", "safety.separation_violation", "safety.event"),
                    "processed"),
    **{f: ("processed" if f == BBOX_FIELD else "raw")
       for f in CONDITIONAL_REQUIRED_STEP_FIELDS[CAMERAS_CONDITION]},
    **{f: ("raw" if f in _HRI_RAW else "processed")
       for f in CONDITIONAL_REQUIRED_STEP_FIELDS[HRI_CONDITION]},
}
# One npz per episode (spec §6): the pictures, and the step each row belongs to.
PACK_STEP_INDEX = "step_index"
PACK_PICTURES = ("depth", "mask")
PACK_MEMBERS = PACK_PICTURES + (PACK_STEP_INDEX,)
# (to - from) x fps from meta/episodes is float noise away from the step count in a real
# release -- 998.9999999999999 and 999.0000000000002 for 999 steps -- while
# a stream one frame off is a whole 1.0 away.
VIDEO_FRAME_TOLERANCE = 0.01

# ── Where an episode's rows are (C7): LeRobot's own pointers, read as its loader reads them ──
# A table -- the steps under data/, the episodes under meta/episodes/ -- is the files LeRobot
# loads of its directory: <chunk>/<file>.parquet, one directory down and nowhere else (lerobot
# 0.6.1 io_utils.load_nested_dataset: sorted(dir.glob("*/*.parquet"))), in name order, their rows
# as stored. Under the table's directory that is a path of two parts.
TABLE_FILE_PARTS = 2
# The characters that make `datasets` take a file's path for a pattern (fsspec's has_magic).
# LeRobot hands it the paths of a table's files, and one that holds any of these is matched
# against what is there instead of opened by name.
GLOB_CHARACTERS = "*?["
# The integer types a feature of info.json can declare: pyarrow's names (datasets.Value).
INTEGER_FEATURE_DTYPES = frozenset(f"{sign}int{bits}" for sign in ("", "u") for bits in (8, 16, 32, 64))
DATA_DIR = "data"
EPISODES_DIR = "episodes"
# LeRobot v3.0's template for a step file, for a meta/info.json that does not spell it out.
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
# meta/episodes' pointers from an episode to its rows: how many, which stretch of the step table
# (dataset_reader._get_query_indices takes episode k's from the k-th row) and which file holds it
# (dataset_metadata.get_data_file_path -- the one step file fetched when episodes are chosen).
DATA_SPAN_COLUMNS = ("length", "dataset_from_index", "dataset_to_index",
                     "data/chunk_index", "data/file_index")
# The step columns that say where a row is and whose it is.
STEP_PLACE_COLUMNS = ("index", "episode_index", "frame_index")
# What LeRobot itself writes into the episode table to find an episode by: always one type.
EPISODE_POINTER_COLUMNS = ("episode_index",) + DATA_SPAN_COLUMNS
# The two counts of rows LeRobot keeps in meta/info.json and reads back instead of counting its
# tables (dataset_metadata.total_episodes · total_frames; an absent one is 0, utils.DatasetInfo):
# the table each one counts, that table's name in a finding, and what the loader does with the
# number -- only what it does with it in every release (c7_info_totals has the rest).
INFO_TOTALS = {
    "total_episodes": (
        f"meta/{EPISODES_DIR}", "에피소드 표",
        "LeRobot 은 표를 세지 않고 이 수를 에피소드 수로 내준다. 0 이면 에피소드 표를 읽지 않고, 0 부터 "
        "이 수 앞까지의 에피소드가 스텝 표에 다 있지 않으면 받은 사본을 두고 Hub 에서 다시 받으려 한다"),
    "total_frames": (
        DATA_DIR, "스텝 표",
        "LeRobot 은 표를 세지 않고 이 수를 len(dataset) 으로 내준다. 길이만큼 읽는 쪽은 크면 없는 행에서 "
        "실패하고, 작으면 끝의 행을 받지 못한다"),
}
# The task table (C7): the one file LeRobot reads the tasks from (utils.DEFAULT_TASKS_PATH) --
# with pandas, the sentences the frame's index (io_utils.load_tasks) -- and the step column that
# names a row of it (dataset_reader: tasks.iloc[task_index].name).
TASKS_PATH = "meta/tasks.parquet"
TASK_INDEX = "task_index"
# The step table beside info.json's features (C7). LeRobot's loader types the step table by the
# features, not by a file: every feature that is not a camera stream is a column of it, a file's
# column they lack stops the loader, and what a file stores is cast to what they declare (lerobot
# 0.6.1 feature_utils.get_hf_features_from_features · datasets' cast_table_to_schema).
# A camera stream's dtype -- no column of the step table -- and a picture's, which is a column but
# not one typed by its dtype and shape.
VIDEO_FEATURE_DTYPE = "video"
IMAGE_FEATURE_DTYPE = "image"
# The two columns LeRobot types by a schema of its own, whatever the features say of them
# (lerobot's datasets/language.py, LANGUAGE_COLUMNS).
LANGUAGE_COLUMNS = ("language_persistent", "language_events")
# The dtype names read here. datasets.Value makes a type of a name by calling pyarrow's function
# of that name, or of that name and "_" (bool is bool_): these are the ones that take no argument
# and give a type a release's column can have. "double" and "float" are Value's own names.
FEATURE_DTYPES = ("null", "bool_", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                  "float16", "float32", "float64", "string", "utf8", "large_string", "large_utf8",
                  "binary", "large_binary", "date32", "date64")
FEATURE_DTYPE_ALIASES = {"double": "float64", "float": "float32"}
# The most dimensions a feature's shape can have (feature_utils: Array5D is the last).
FEATURE_SHAPE_DIMENSIONS = 5

# ── Shipped assets (C6): the layout the converter writes (generation/assets.py) ──
# A redistributable USD the batch's episodes use sits at assets/<file>; the textures it
# references as ./textures/<file> sit at assets/textures/<file>, each one a row of the
# shipped meta/assets.json entry's bundle.files -- its provenance and SHA-256. A robot that
# ships ships as its own folder, assets/<dir>/<file>, each file a row of the entry's `ship`
# (ROBOT_SHIP_KEYS). An entry that composes others (COMPOSES_KEY -- the Stage) sits at assets/<file>
# when everything it composes ships. Copies of generation/assets.py's constants (the validator does not
# import generation code); tests/validator/test_asset_checks.py holds the two to one value.
ASSETS_DIR = "assets"
TEXTURE_SUBDIR = "textures"
TEXTURE_ROW_KEYS = ("file", "sha256", "license", "url")
TEXTURE_ROW_OPTIONAL = ("member",)
ROBOT_SHIP_KEYS = ("dir", "source_dir", "license_file", "files")
COMPOSES_KEY = "composes"


def camera_video_fields() -> list[str]:
    """The camera fields that are LeRobot video features, not parquet columns."""
    return [f for f in CONDITIONAL_REQUIRED_STEP_FIELDS[CAMERAS_CONDITION]
            if f.startswith(VIDEO_FIELD_PREFIX)]


def camera_column_fields() -> list[str]:
    """The camera fields that are parquet columns: the box and the camera poses."""
    return [f for f in CONDITIONAL_REQUIRED_STEP_FIELDS[CAMERAS_CONDITION]
            if not f.startswith(VIDEO_FIELD_PREFIX)]


# 물리적으로 불가능하거나 명백히 오류인 값의 경계
SANITY = {
    "joint_pos_abs_max": 2 * math.pi,   # rad. 초과 시 단위 혼동(deg) 의심
    "joint_vel_abs_max": 100.0,         # rad/s
    "joint_torque_abs_max": 5000.0,     # N·m
    # N. 단위 혼동을 잡는 경계이지 「인체 접촉으로 그럴듯한 값」이 아니다. 넘어진
    # 더미가 지면에 부딪히는 임펄스는 시뮬레이터에서 실제로 크다 — 140 에피소드
    # 실측에서 에피소드별 최대의 중앙값이 12,217 N, p95 49,045 N, 최대 119,647 N
    # (전부 BR-12 등 접지 부위, 2026-08-25). 그 값들은 로봇-인체 접촉이 아니며
    # contact_type 이 매겨지지 않아 표 A.2 와 비교되지도 않는다(스키마 §8.1).
    # 단위를 dyn 으로 잘못 적으면 1e5 배가 되므로 이 상한도 그것은 잡는다.
    "contact_force_max": 1_000_000.0,
    # m/s. 접촉 지점 상대속도. 이족 로봇의 사지 끝단이 낼 수 있는 속도의 상한을
    # 넉넉히 잡은 값으로, 이보다 크면 단위 혼동이거나 산식이 물체 밖으로 뻗은
    # 것이다 — 지렛대 상한이 없던 구현이 1 m/s 보행에서 38 m/s 를 냈다.
    "contact_v_rel_max": 30.0,
    "duration_min_s": 0.5,
    "fps_min": 1,
    "fps_max": 10000,
}

ERROR, WARN, INFO, SKIP = "ERROR", "WARN", "INFO", "SKIP"


# ────────────────────────────────────────────────────────────────
# 리포트 자료구조
# ────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    check: str
    severity: str
    message: str
    location: str = ""

    def line(self) -> str:
        loc = f"  [{self.location}]" if self.location else ""
        return f"{self.severity:<5} {self.check}  {self.message}{loc}"


@dataclass
class Report:
    root: str
    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    # 어떤 검사를 돌리기로 했는가. 비면 전부다. 미실행을 OK 와 구별하는 데 쓴다.
    requested: set[str] | None = None

    def add(self, check: str, severity: str, message: str, location: str = "") -> None:
        self.findings.append(Finding(check, severity, message, location))

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    def by_check(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for f in self.findings:
            out[f.check][f.severity] += 1
        return {k: dict(v) for k, v in out.items()}

    def status_of(self, check: str) -> str:
        """C1~C9 각각이 어떤 상태로 끝났는가. 미실행을 OK 와 구별한다."""
        if self.requested is None:
            return "unknown"
        if check not in self.requested:
            return "not_run"
        counts = self.by_check().get(check, {})
        for level, name in ((ERROR, "error"), (WARN, "warn"), (SKIP, "skip")):
            if counts.get(level):
                return name
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "schema_version": SCHEMA_VERSION,
            "summary": {
                "error": self.count(ERROR),
                "warn": self.count(WARN),
                "info": self.count(INFO),
                "skip": self.count(SKIP),
            },
            "by_check": self.by_check(),
            # 텍스트만 「미실행」을 말하고 JSON 은 말하지 않으면, 증빙으로 JSON 을
            # 읽는 쪽은 부분 실행과 전체 통과를 여전히 구별하지 못한다.
            # 빈 집합을 「전부 요청」으로 읽으면, 검사를 하나도 돌리지 않은 실행이
            # 전 검사 통과로 기록된다 — 없는 root 로 C0 에서 돌아온 경우가 그랬다.
            # 요청을 못 세웠으면 null 이다.
            "requested_checks": sorted(self.requested) if self.requested is not None else None,
            "check_status": {c: self.status_of(c) for c in ALL_CHECKS},
            "stats": self.stats,
            "findings": [asdict(f) for f in self.findings],
        }


# ────────────────────────────────────────────────────────────────
# 데이터셋 로딩 — 레이아웃 변화에 관대하게
# ────────────────────────────────────────────────────────────────

def _parquet_under(directory: Path) -> list[Path]:
    """Every parquet file under a table's directory, in name order.

    A recursive search, and with it LeRobot's own search (io_utils.load_nested_dataset:
    ``glob("*/*.parquet")``). The two differ where a chunk directory is a symlink: ``*/`` goes
    into it and ``**`` does not, so the recursive search alone finds half of a release that the
    loader reads all of (measured with the loader, lerobot 0.6.1).
    """
    return sorted(set(directory.rglob("*.parquet")) | set(directory.glob("*/*.parquet")))


class Dataset:
    """LeRobotDataset v3.0 레이아웃 + 본 과제 확장 파일을 읽는다."""

    def __init__(self, root: Path, rep: Report):
        self.root = root
        self.rep = rep
        self.meta = root / "meta"
        # load_frames() 가 --file-limit 로 일부만 읽었는지. 전역 대조 검사가 본다.
        self.frames_truncated = False

    # -- 메타 파일 --------------------------------------------------
    def _json(self, rel: str) -> Any | None:
        p = self.meta / rel
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        # ValueError: a syntax error (JSONDecodeError), bytes that are not UTF-8 (UnicodeDecodeError)
        # and an integer past the interpreter's digit limit. RecursionError: nesting past the
        # decoder's depth. All are files the json module will not read -- findings, not exit 3.
        except (ValueError, RecursionError) as e:
            self.rep.add("C1", ERROR, f"JSON 파싱 실패: {e}", f"meta/{rel}")
            return None

    def info(self) -> dict | None:
        """meta/info.json, when it holds the JSON object LeRobot reads it as. None for a file
        that is absent, does not parse (said as it is read) or holds anything else -- a list, a
        number, null -- which c1_layout says: every caller asks this for keys."""
        doc = self._json("info.json")
        return doc if isinstance(doc, dict) else None

    def rules(self) -> dict | None:
        return self._json("rules.json")

    def scenarios(self) -> dict | list | None:
        return self._json("scenarios.json")

    def safety_schema(self) -> dict | None:
        return self._json("safety_schema.json")

    def assets(self) -> dict | None:
        return self._json("assets.json")

    def ships_cameras(self) -> bool:
        """The camera condition: the file's presence, not whether it parses -- an unreadable
        meta/cameras.json still declares a camera distribution."""
        return (self.meta / "cameras.json").exists()

    def cameras(self) -> Any | None:
        return self._json("cameras.json")

    def packs_manifest(self) -> tuple[Any | None, list[str]]:
        """meta/packs.json and what is wrong with it as JSON.

        Parsed with a duplicate-key hook: json.loads keeps the last of two equal keys, so an
        episode listed twice would read as listed once, and "every episode exactly once"
        could not be said. Absent file: (None, [])."""
        p = self.meta / "packs.json"
        if not p.exists():
            return None, []
        dupes: list[str] = []

        def pairs(items):
            out = {}
            for k, v in items:
                if k in out:
                    dupes.append(k)
                out[k] = v
            return out

        try:
            doc = json.loads(p.read_text(encoding="utf-8"), object_pairs_hook=pairs)
        # What _json catches, and why: an integer past the digit limit is a ValueError too.
        except (ValueError, RecursionError) as e:
            return None, [f"JSON 파싱 실패: {type(e).__name__}: {e}"]
        problems = []
        if dupes:
            problems.append(f"같은 키가 두 번 이상 나옴: {sorted(set(dupes))[:3]} — JSON 을 읽으면 "
                            "마지막 것만 남으므로 앞의 항목은 아무도 보지 못한다")
        return doc, problems

    def joint_maps(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        d = self.meta / "joint_map"
        if not d.is_dir() or yaml is None:
            return out
        for p in sorted(d.glob("*.yaml")):
            try:
                out[p.stem] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception as e:  # noqa: BLE001
                self.rep.add("C2", ERROR, f"매핑 파일 파싱 실패: {e}", str(p.relative_to(self.root)))
        return out

    # -- 에피소드 메타 ----------------------------------------------
    def episodes(self) -> pd.DataFrame | None:
        """meta/episodes/*.parquet, meta/episodes.jsonl, meta/episodes.json 순으로 탐색."""
        d = self.meta / "episodes"
        if d.is_dir():
            files = _parquet_under(d)
            if files:
                try:
                    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
                except Exception as e:  # noqa: BLE001
                    self.rep.add("C1", ERROR, f"에피소드 메타 읽기 실패: {e}", "meta/episodes/")
                    return None
        for name in ("episodes.jsonl", "episodes.json"):
            p = self.meta / name
            if p.exists():
                try:
                    if name.endswith(".jsonl"):
                        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
                    else:
                        rows = json.loads(p.read_text(encoding="utf-8"))
                        if isinstance(rows, dict):
                            rows = list(rows.values())
                    return pd.DataFrame(rows)
                except Exception as e:  # noqa: BLE001
                    self.rep.add("C1", ERROR, f"에피소드 메타 읽기 실패: {e}", f"meta/{name}")
                    return None
        return None

    # -- 스텝 데이터 ------------------------------------------------
    def data_files(self) -> list[Path]:
        d = self.root / "data"
        return _parquet_under(d) if d.is_dir() else []

    def notice_text(self) -> str | None:
        for name in ("NOTICE", "NOTICE.md", "NOTICE.txt"):
            p = self.root / name
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace")
        return None


# ────────────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────────────

def as_list(v: Any) -> list | None:
    """parquet 의 list 컬럼은 ndarray 로 올 수 있다. 리스트로 정규화."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                return json.loads(s)
            except (ValueError, RecursionError):     # what Dataset._json catches, for the same reasons
                return None
    return None


def json_member(doc: Any, key: str, kind: type) -> Any:
    """doc[key] when doc is a JSON object and that member is of ``kind`` (dict or list), else an
    empty one. A member of another shape is said by the check that reads its shape; a reader that
    walks it as given ends the validator instead (a number has no items, a list no .get)."""
    value = doc.get(key) if isinstance(doc, dict) else None
    return value if isinstance(value, kind) else kind()


def asset_entries(canon_assets: Any) -> list[tuple[str, dict]]:
    """(key, entry) of the shipped asset canon's robots and scene that name a USD by a non-empty
    string -- the path is a key below, and a list or a number there cannot be one. An entry
    without such a path is said by C6 where an episode needs it (_required_scene_assets)."""
    return [(name, entry)
            for group in ("robots", "scene")
            for name, entry in json_member(canon_assets, group, dict).items()
            if isinstance(entry, dict) and isinstance(entry.get("usd_path"), str) and entry["usd_path"]]


def as_obj(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if s.startswith(("[", "{")):
            try:
                return json.loads(s)
            except (ValueError, RecursionError):     # what Dataset._json catches, for the same reasons
                return v
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def elem_len(v: Any) -> int | None:
    """스텝 셀 하나의 길이. 스칼라면 0.

    JSON 배열 문자열도 센다. LeRobotDataset v3.0 은 문자열 배열 컬럼을 담지
    못해(HF Value("string") 은 스칼라 전용) 리스트 필드가 JSON 문자열 하나로
    온다. as_list() 는 이미 그 형태를 읽으므로 길이 검사도 같은 것을 읽어야
    한다 — 안 그러면 같은 파일에서 region_names 는 12, contact.type 은 0 으로
    읽혀 존재하지 않는 불일치가 보고된다.
    """
    if isinstance(v, np.ndarray):
        return int(v.shape[-1]) if v.ndim else 0
    if isinstance(v, (list, tuple)):
        return len(v)
    if isinstance(v, str):
        parsed = as_list(v)
        if parsed is not None:
            return len(parsed)
    return 0


def flat(v: Any) -> np.ndarray:
    a = np.asarray(v, dtype=object)
    try:
        return np.asarray(v, dtype=float).ravel()
    except (ValueError, TypeError):
        return np.array([], dtype=float)


# ────────────────────────────────────────────────────────────────
# 검사 구현
# ────────────────────────────────────────────────────────────────

def c1_layout(ds: Dataset, rep: Report, eps: pd.DataFrame | None = None,
              frames: list[tuple[Path, pd.DataFrame]] = ()) -> None:
    """C1 — 필수 파일 존재 및 스키마 버전. 카메라 배포본이면 영상·팩 파일까지."""
    required = {
        "meta/info.json": ERROR,
        "meta/rules.json": ERROR,
        "meta/scenarios.json": WARN,
        "meta/safety_schema.json": WARN,
        "LICENSE": ERROR,
        "README.md": WARN,
    }
    for rel, sev in required.items():
        if not (ds.root / rel).exists():
            rep.add("C1", sev, "필수 파일 없음", rel)

    if not ds.data_files():
        rep.add("C1", ERROR, "data/ 에 parquet 파일이 없음", "data/")

    if not (ds.meta / "joint_map").is_dir():
        rep.add("C1", ERROR, "meta/joint_map/ 없음 — 관절 해석 불가", "meta/joint_map/")
    elif yaml is None:
        rep.add("C2", SKIP, "pyyaml 미설치로 매핑 파일 검사 생략")

    info = ds.info()
    if info is None and (ds.meta / "info.json").exists():
        # Not absent (said above). What does not parse was said as it was read; what is left is
        # JSON that is no object. Every check reads that like no file at all, so nothing else
        # says it -- and the loader stops on it as it opens (measured: a list, a number, a
        # string, null).
        try:
            doc = json.loads((ds.meta / "info.json").read_text(encoding="utf-8"))
        except (ValueError, RecursionError):        # what Dataset._json said as it read the file
            pass
        else:
            rep.add("C1", ERROR, "meta/info.json 이 JSON 객체가 아님 — LeRobot 은 이 파일을 키와 값으로 읽고, "
                                 "다른 것이면 여는 단계에서 멈춘다", f"meta/info.json · {repr(doc)[:60]}")
    if info is not None:
        sv = str(info.get("schema_version", ""))
        if not sv:
            rep.add("C1", WARN, "info.json 에 schema_version 없음", "meta/info.json")
        elif sv != SCHEMA_VERSION:
            rep.add("C1", WARN, f"schema_version 불일치: {sv} (검증기 기준 {SCHEMA_VERSION})",
                    "meta/info.json")

    _c1_safety_schema_sound(ds, rep)
    _c1_rules_complete(ds, rep)
    _c1_scenarios_sound(ds, rep)
    _c1_camera_media(ds, info, eps, frames, rep)


def _set_or_none(values) -> set | None:
    """set(values), or None when one of them cannot go into a set (a list, an object)."""
    try:
        return set(values)
    except TypeError:
        return None


def _c1_safety_schema_sound(ds: Dataset, rep: Report) -> None:
    """동봉된 기계가독 규격이 실제로 이 데이터의 규격인지.

    이 파일의 **존재만** 보면 배포본의 파일을 `{}` 로
    바꿔도 검증 결과가 변하지 않는다. 「받은 쪽이 읽는
    규격」이 배포본 안에서 검증되지 않으면, 그 파일이 하는 약속은 근거가 없다.

    저장소 정본이 아니라 **검증기 자신의 상수**와 대조한다. 배포본이 실은 사본을
    자기 자신과 맞춰 보는 것은 검사가 아니다.
    """
    loc = "meta/safety_schema.json"
    canon = ds.safety_schema()
    if canon is None:
        # Absent: said above as a WARN. Not JSON: said as it was read. A file holding null is
        # neither -- it is there and declares nothing.
        path = ds.meta / "safety_schema.json"
        try:
            holds_null = path.is_file() and json.loads(path.read_text(encoding="utf-8")) is None
        except (ValueError, RecursionError):
            holds_null = False
        if not holds_null:
            return
    if not isinstance(canon, dict) or not canon:
        rep.add("C1", ERROR, "규격 정의가 비어 있음", loc)
        return

    sv = str(canon.get("schema_version", ""))
    if sv != SCHEMA_VERSION:
        # 경고로 끝내면 버전만 바꿔 놓고 내용 전체를 대조 없이 통과시킬 수 있다.
        # 기본 실행은 WARN 에서 exit 0 이므로 배포 게이트가 통째로 fail-open 이 된다.
        rep.add("C1", ERROR,
                f"규격 정의의 schema_version {sv!r} 이 검증기 기준 {SCHEMA_VERSION} 과 "
                "다름 — 내용을 대조할 수 없으므로 이 배포본을 검증했다고 말할 수 없다",
                loc)
        return

    expected: dict[str, object] = {
        "body_regions": BODY_REGIONS,
        "contact_types": CONTACT_TYPES,
        "safety_events": SAFETY_EVENTS,
        "required_episode_fields": set(REQUIRED_EPISODE_FIELDS),
        "required_step_fields": set(REQUIRED_STEP_FIELDS),
        # What a release may leave out: a reader decides by this list which columns to expect.
        "optional_step_fields": set(OPTIONAL_STEP_FIELDS),
        # 아래 셋은 배포본에서 변조돼도 아무도 보지 않고 있었다. 「받은 쪽이 읽는
        # 규격」이 검증기와 다른 말을 하면 소비자가 다른 판단을 한다.
        "fixed_shapes": set(FIXED_SHAPES),
    }
    # 컨테이너 형식까지 본다. `set(got)` 만 쓰면 목록이어야 할 자리에 같은 키를 담은
    # 객체가 와도, 객체여야 할 자리에 목록이 와도 통과한다 — fixed_shapes 만이 아니라
    # 형제들도 형식을 본다.
    # body_regions 는 설계상 매핑이다(부위 코드 → 표시명). 나머지는 목록이어야 한다.
    LIST_LIKE = {"contact_types", "safety_events",
                 "required_episode_fields", "required_step_fields", "optional_step_fields"}
    # 반대쪽도 못박는다. body_regions 를 목록으로 바꾸면 표시명이 사라지는데
    # `set(got)` 대조는 통과한다.
    MAPPING_LIKE = {"body_regions"}
    for key, want in expected.items():
        got = canon.get(key)
        if got is None:
            rep.add("C1", ERROR, f"규격 정의에 {key} 없음", loc)
            continue
        if key in LIST_LIKE and not isinstance(got, (list, tuple)):
            rep.add("C1", ERROR,
                    f"규격 정의의 {key} 가 목록이 아님: {type(got).__name__}", loc)
            continue
        if key in MAPPING_LIKE and not isinstance(got, dict):
            rep.add("C1", ERROR,
                    f"규격 정의의 {key} 가 객체가 아님: {type(got).__name__} — "
                    "부위 표시명을 담지 못한다", loc)
            continue
        try:
            got_set = set(got)
        except TypeError:
            rep.add("C1", ERROR, f"규격 정의의 {key} 형식이 목록이 아님", loc)
            continue
        if got_set != want:
            extra = sorted(got_set - want, key=str)[:3]       # a number beside a string: no order
            missing = sorted(want - got_set)[:3]
            rep.add("C1", ERROR,
                    f"규격 정의의 {key} 가 검증기와 다름 — 없음 {missing}, 잉여 {extra}",
                    loc)

    enums = canon.get("enums")
    if not isinstance(enums, dict):
        rep.add("C1", ERROR, "규격 정의에 enums 없음", loc)
    else:
        for name, allowed in ENUMS.items():
            got = enums.get(name)
            if got is not None and not isinstance(got, (list, tuple)):
                rep.add("C1", ERROR,
                        f"규격 정의의 enum {name} 이 목록이 아님: {type(got).__name__}", loc)
                continue
            if got is None or _set_or_none(got) != allowed:
                rep.add("C1", ERROR, f"규격 정의의 enum {name} 이 검증기와 다름", loc)

    # 값까지 대조한다 — 키만 맞고 길이가 다른 fixed_shapes 는 배열 길이 검사를
    # 통과시키면서 규격서만 읽는 소비자에게 다른 형상을 알려 준다.
    shapes = canon.get("fixed_shapes")
    # 키만 맞는 목록이면 위의 `set(got)` 대조는 통과하지만 길이는 어디에도 없다 —
    # 형태를 말하는 자리가 형태를 담고 있는지부터 본다.
    if not isinstance(shapes, dict):
        rep.add("C1", ERROR,
                "규격 정의의 fixed_shapes 가 객체가 아님 — 길이를 담지 못한다", loc)
    else:
        wrong = sorted(k for k, v in FIXED_SHAPES.items() if shapes.get(k) != v)
        if wrong:
            rep.add("C1", ERROR,
                    f"규격 정의의 fixed_shapes 길이가 검증기와 다름: {wrong[:3]}", loc)
    # 이 둘은 정본이 `source=sim` 처럼 조건을 키에 적고 검증기 상수는 `sim` 만
    # 쓴다. 표기가 다를 뿐 같은 것을 가리켜야 하므로 접두를 떼고 대조한다.
    def _cond_key(k: str) -> str:
        return str(k).split("=", 1)[-1]

    for key, want in (("forbidden_measurement_methods", FORBIDDEN_MEASUREMENT_METHODS),
                      ("conditional_required_episode_fields",
                       CONDITIONAL_REQUIRED_EPISODE_FIELDS)):
        got = canon.get(key)
        if not isinstance(got, dict):
            rep.add("C1", ERROR, f"규격 정의의 {key} 가 객체가 아님", loc)
            continue
        norm = {_cond_key(k): _set_or_none(v) for k, v in got.items() if isinstance(v, (list, tuple))}
        expect = {k: set(v) for k, v in want.items()}
        if norm != expect:
            rep.add("C1", ERROR,
                    f"규격 정의의 {key} 가 검증기와 다름 — 정본 {sorted(norm)} vs "
                    f"검증기 {sorted(expect)}", loc)

    # The table scenario's episode columns and the members of its spec, compared whole -- the order too, which the
    # spec's §5 rows read in. The key is a scenario canon key ("conditions"), compared as written.
    key = "scenario_conditional_episode_fields"
    got = canon.get(key)
    want_table = {k: list(fields) for k, fields in SCENARIO_CONDITIONAL_EPISODE_FIELDS.items()}
    if not isinstance(got, dict) or not got:
        rep.add("C1", ERROR,
                f"규격 정의에 {key} 가 없음 — 조건표가 있는 시나리오의 에피소드가 무엇을 반드시 싣는지 받는 쪽이 "
                "알 수 없다", loc)
    elif got != want_table:
        def listed(k):
            value = got.get(k)
            return value if isinstance(value, list) else []

        diff = {k: {"없음": [f for f in want_table.get(k, []) if f not in listed(k)],
                    "잉여": [f for f in listed(k) if f not in want_table.get(k, [])][:3]}
                for k in sorted(set(got) | set(want_table)) if got.get(k) != want_table.get(k)}
        rep.add("C1", ERROR, f"규격 정의의 {key} 가 검증기와 다름(차례까지) — {diff}", loc)
    key = "episode_spec_members"
    got = canon.get(key)
    if not isinstance(got, list) or not got:
        rep.add("C1", ERROR, f"규격 정의에 {key} 가 없음 — episode_spec 의 멤버를 받는 쪽이 알 수 없다", loc)
    elif got != list(EPISODE_SPEC_MEMBERS):
        rep.add("C1", ERROR,
                f"규격 정의의 {key} 가 검증기와 다름 — 정본 {got[:8]} vs 검증기 {list(EPISODE_SPEC_MEMBERS)}", loc)

    # The step-side condition is a bare name ("cameras": the distribution ships
    # meta/cameras.json), not "source=<value>", so the keys are compared as written --
    # _cond_key would also accept "anything=cameras". A value must be a list of strings
    # (a bare string would set() into characters). Order and duplicates are not
    # compared: the set is the contract, as for required_step_fields above.
    key = "conditional_required_step_fields"
    got = canon.get(key)
    if not isinstance(got, dict):
        rep.add("C1", ERROR,
                f"규격 정의에 {key} 가 없거나 객체가 아님 — 카메라가 있는 배포본이 무엇을 "
                "반드시 싣는지 받는 쪽이 알 수 없다", loc)
    else:
        not_lists = sorted(k for k, fields in got.items()
                           if not isinstance(fields, list)
                           or not all(isinstance(f, str) for f in fields))
        declared = {k: set(fields) for k, fields in got.items() if k not in not_lists}
        expect = {k: set(fields) for k, fields in CONDITIONAL_REQUIRED_STEP_FIELDS.items()}
        if not_lists:
            rep.add("C1", ERROR, f"규격 정의의 {key} 의 {not_lists} 가 문자열 목록이 아님", loc)
        elif declared != expect:
            diff = {k: {"없음": sorted(expect.get(k, set()) - declared.get(k, set())),
                        "잉여": sorted(declared.get(k, set()) - expect.get(k, set()))}
                    for k in sorted(set(declared) | set(expect))
                    if declared.get(k) != expect.get(k)}
            rep.add("C1", ERROR, f"규격 정의의 {key} 가 검증기와 다름 — {diff}", loc)

    # 이 둘이 없으면 C5 의 그릇·멤버 검사가 **조용히 건너뛴다** — 옛 정본을 실은
    # 배포본이 새 검사를 통과한 것처럼 보인다. 없으면 빨간불을 내야 한다.
    # 그릇을 선언한 필드는 **반드시** json_encoded_fields 에도 있어야 한다. C5 는
    # 그 목록을 순회하므로, 이름 하나만 빼면 그 필드의 검사가 통째로 꺼진다.
    listed_ep = {f for f in json_member(json_member(canon, "json_encoded_fields", dict), "episode", list)
                 if isinstance(f, str)}
    unrouted = sorted(set(EPISODE_JSON_SHAPES) - listed_ep)
    if unrouted:
        rep.add("C1", ERROR,
                f"json_encoded_fields.episode 에 {unrouted} 가 없음 — 그릇이 선언된 "
                "필드가 그 목록에 없으면 C5 가 그 필드를 아예 보지 않는다", loc)

    for key, want in (("episode_json_shapes", EPISODE_JSON_SHAPES),
                      ("notes_members", NOTES_MEMBERS),
                      ("notes_member_shapes", NOTES_MEMBER_SHAPES),
                      ("notes_bounds", NOTES_BOUNDS)):
        got = canon.get(key)
        if not isinstance(got, dict) or not got:
            rep.add("C1", ERROR,
                    f"규격 정의에 {key} 가 없음 — 이것이 없으면 C5 의 해당 검사가 "
                    "돌지 않고, 검사하지 않은 것이 통과한 것처럼 보인다", loc)
        elif got != want:
            missing = sorted(set(want) - set(got))
            extra = sorted(set(got) - set(want))
            wrong = sorted(k for k in set(got) & set(want) if got[k] != want[k])
            rep.add("C1", ERROR,
                    f"규격 정의의 {key} 가 검증기와 다름 — 없음 {missing}, "
                    f"잉여 {extra}, 값 다름 {wrong}", loc)

    # The layer table is compared whole, as the notes tables are: a reader picks the raw columns
    # by it, and a table the validator does not know would sort a column into the wrong layer.
    got = canon.get("step_field_layers")
    if not isinstance(got, dict) or not got:
        rep.add("C1", ERROR,
                "규격 정의에 step_field_layers 가 없음 — 열마다 로우인지 가공인지 받는 쪽이 알 수 없다", loc)
    elif got != STEP_FIELD_LAYERS:
        missing = sorted(set(STEP_FIELD_LAYERS) - set(got))
        extra = sorted(set(got) - set(STEP_FIELD_LAYERS))
        wrong = sorted(k for k in set(got) & set(STEP_FIELD_LAYERS) if got[k] != STEP_FIELD_LAYERS[k])
        rep.add("C1", ERROR,
                f"규격 정의의 step_field_layers 가 검증기와 다름 — 없음 {missing[:3]}, 잉여 {extra[:3]}, "
                f"값 다름 {wrong[:3]}", loc)

    encoded = canon.get("json_encoded_fields")
    if not isinstance(encoded, dict) or not encoded.get("step") or not encoded.get("episode"):
        rep.add("C1", ERROR,
                "규격 정의의 json_encoded_fields 가 비었거나 형식이 아님 — 리스트 필드가 "
                "JSON 문자열로 실린다는 사실을 받는 쪽이 알 수 없다", loc)
    elif True:
        # 비었는지만 보면 엉뚱한 이름이 들어 있어도 통과한다. 이 목록은 받는 쪽이
        # json.loads 를 쓸 필드를 정하는 근거이므로, 실재하지 않는 필드를 가리키면
        # 파싱 코드가 엉뚱한 열을 건드린다. 목록을 여기 또 적지 않고 **필수 필드에
        # 실재하는 이름인지**로 본다.
        for scope, known in (("step", set(REQUIRED_STEP_FIELDS)),
                             ("episode", set(REQUIRED_EPISODE_FIELDS)
                              | {f for v in CONDITIONAL_REQUIRED_EPISODE_FIELDS.values() for f in v}
                              | {f for v in SCENARIO_CONDITIONAL_EPISODE_FIELDS.values() for f in v})):
            listed = encoded.get(scope)
            if not isinstance(listed, list) or not all(isinstance(x, str) for x in listed):
                rep.add("C1", ERROR,
                        f"규격 정의의 json_encoded_fields.{scope} 가 문자열 목록이 아님", loc)
                continue
            unknown = sorted(set(listed) - known)
            if unknown:
                rep.add("C1", ERROR,
                        f"규격 정의의 json_encoded_fields.{scope} 가 실재하지 않는 필드를 "
                        f"가리킴: {unknown[:3]}", loc)


# rules.json 의 임계값 항목이 지켜야 할 계약. 픽스처 생성기도 같은 계약을 쓴다.
THRESHOLD_FIELDS = ("value", "unit", "determined_by")
# 규칙이 계산되려면 반드시 있어야 하는 파라미터 이름. 없는 파라미터는 값이 빈 것보다
# 나쁘다 — 빈 값은 위의 임계값 검사가 잡지만, 아예 없는 이름은 순회 대상 자체가 되지
# 않아 "문제 없음" 으로 통과한다(생성기 쪽은 missing 으로 거른다).
#
# 지금은 저장소가 산출식을 실제로 구현한 규칙 하나뿐이다. 그 산출식이
# generation/envs/s1a/mdp/safety.py 의 SSM_PARAMETER_NAMES 이고, 두 목록은 같아야
# 한다 (CONTRIBUTING.md 「함께 고치는 곳」). 검증기는 생성 코드를 import 하지 않는다 — Isaac 없이
# 도는 것이 이 파일의 조건이고, S1a 전용 모듈에 묶이지 않아야 한다.
# The outcome shares C4 warns below: the line where the dataset's purpose starts to suffer, not a target. A target is
# each scenario's own (meta/scenarios.json outcome_target, spec §9 「시나리오별 구성 비율」) and a release that mixes
# scenarios has none; a batch a little off its target is a normal batch, and warning on it would redden one every
# time. So the validator holds no target and C4 compares these floors -- the project's own, one set for the whole
# release -- with the shares of its random rows (a grid row of a conditions table is a cell of the failure map, not a
# draw).
OUTCOME_FLOOR = {"fail": 0.25, "near_miss": 0.15}

REQUIRED_RULE_PARAMETERS = {
    # 라벨러가 산출식을 구현한 규칙은 그 파라미터 **이름**이 있어야 한다. 이름이 없으면
    # iter_threshold_entries 의 순회 대상이 되지 않아 값 검사도 비껴간다 — 지우면
    # 검증기는 조용하고 라벨러만 실행 시 죽는다.
    "R-10218-STOP-01": ("max_stop_time", "stop_speed_threshold"),
    "R-TS15066-SSM-01": (
        "human_speed",
        "robot_reaction_time",
        "robot_stop_time",
        "intrusion_distance",
        "position_uncertainty_robot",
        "position_uncertainty_human",
    ),
    # The directed S_p of the hri scenarios. The same names as
    # generation/directed_separation.py DIRECTED_SSM_PARAMETERS (a test compares them).
    "R-TS15066-SSM-02": (
        "robot_reaction_time",
        "intrusion_distance",
        "position_uncertainty_robot",
        "position_uncertainty_human",
        "approach_speed_threshold",
    ),
}
DETERMINED_BY = {"standard", "project"}

# 부위마다 반드시 있어야 하는 항목. 순회만 하면 **없는 것은 아무도 보지 않는다** —
# 임계값을 지우거나 그룹을 통째로 없애도 검사가 조용히 통과한다.
# 그래서 「무엇이 있어야 하는가」를 여기서 못박고 순회 전에 확인한다.
REQUIRED_LIMIT_NAMES = ("force_quasistatic", "pressure_quasistatic", "energy_transient")
# 항목마다 쓸 수 있는 단위. 「비어 있지 않은 문자열」만 요구하면 배포본이 kN 을 싣고도
# 통과하고, 라벨러는 단위를 읽지 않고 기록된 N 과 비교하므로 판정이 통째로 어긋난다.
# generation/labeling/label_episodes.py CANONICAL_UNITS 와 같아야 한다 (CONTRIBUTING.md 「함께 고치는 곳」).
CANONICAL_UNITS = {
    "force_quasistatic": "N", "pressure_quasistatic": "N/cm^2",
    "energy_transient": "J", "effective_mass": "kg",
    "spring_constant": "N/mm", "transient_multiplier": "1",
}
REQUIRED_BIOMECHANICAL_NAMES = ("effective_mass", "spring_constant")

# R-TS15066-SSM-02's stop table (spec §9.1) sits beside the rule's parameters, not among them --
# threshold_problems takes one number per parameter -- so nothing else looks at it. A copy of
# generation/directed_separation.py stop_table_problems and its constants (this file imports
# nothing from generation/); tests/generation/test_directed_separation.py holds the two to one answer.
DIRECTED_SSM_RULE_ID = "R-TS15066-SSM-02"
RULES_WITH_STOP_TABLE = (DIRECTED_SSM_RULE_ID,)
STOP_TABLE_MOTIONS = ("walking", "turning")
STOP_TABLE_STATES = ("empty", "carry_2kg", "carry_3kg")
STOP_TABLE_ROW_UNITS = {"walking": ["m/s", "s", "m"], "turning": ["rad/s", "s", "m"]}


def _table_number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def stop_table_problems(table: object) -> list[str]:
    """Everything wrong with a stop table, as "stop_table.<where>: <why>" lines; [] when usable."""
    if not isinstance(table, dict):
        return [f"stop_table: 객체가 아님 ({type(table).__name__})"]
    out: list[str] = []
    if table.get("determined_by") != "project":
        out.append(f"stop_table.determined_by: project 가 아님 ({table.get('determined_by')!r})")
    measured = table.get("measured_on")
    if not isinstance(measured, str) or not measured.strip():
        out.append("stop_table.measured_on: 잰 날짜가 없음")
    if table.get("row_units") != STOP_TABLE_ROW_UNITS:
        out.append(f"stop_table.row_units: {STOP_TABLE_ROW_UNITS} 가 아님")
    masses = table.get("carry_mass_kg")
    if not isinstance(masses, dict) or set(masses) != set(STOP_TABLE_STATES):
        out.append(f"stop_table.carry_mass_kg: 상태가 {list(STOP_TABLE_STATES)} 가 아님")
    else:
        values = [masses[s] for s in STOP_TABLE_STATES]
        if not all(_table_number(v) and v >= 0 for v in values):
            out.append("stop_table.carry_mass_kg: 유한한 음이 아닌 수가 아닌 값이 있음")
        elif any(b <= a for a, b in zip(values, values[1:])):
            out.append(f"stop_table.carry_mass_kg: {list(STOP_TABLE_STATES)} 순서로 커지지 않음")
    reach = table.get("turning_reach_m")
    if not _table_number(reach) or reach <= 0:
        out.append(f"stop_table.turning_reach_m: 유한한 양수가 아님 ({reach!r})")
    for motion in STOP_TABLE_MOTIONS:
        block = table.get(motion)
        if not isinstance(block, dict) or set(block) != set(STOP_TABLE_STATES):
            out.append(f"stop_table.{motion}: 상태가 {list(STOP_TABLE_STATES)} 가 아님")
            continue
        for state in STOP_TABLE_STATES:
            where = f"stop_table.{motion}.{state}"
            rows = block[state]
            if not isinstance(rows, list) or not rows:
                out.append(f"{where}: 행이 없음")
                continue
            if not all(isinstance(r, list) and len(r) == 3
                       and all(_table_number(x) and x >= 0 for x in r) for r in rows):
                out.append(f"{where}: 행마다 유한한 음이 아닌 수 셋 [속도, 정지 시간, 정지 거리] 가 아님")
                continue
            speeds = [r[0] for r in rows]
            if speeds[0] <= 0 or any(b <= a for a, b in zip(speeds, speeds[1:])):
                out.append(f"{where}: 속도가 0 보다 크게 시작해 행마다 커지지 않음")
    return out


def _is_filled_text(value: object) -> bool:
    """비어 있지 않은 **문자열**인지. 숫자나 리스트는 참조가 될 수 없다.

    str(123) 도 str(["a"]) 도 비어 있지 않으므로, 형변환 뒤에 검사하면 사람이
    따라갈 수 없는 참조가 계약을 만족한다.
    """
    return isinstance(value, str) and bool(value.strip())


def rule_region_problems(rules: dict) -> list[tuple[str, str]]:
    """접촉 규칙이 자기 부위의 한계를 가리키는지.

    body_region 과 limits_ref 가 어긋나면 다른 부위의 기준으로 판정이 계산되고,
    결과는 그럴듯해 보인다 — 두 값 모두 유효한 부위 코드이기 때문이다. 어느 검사도
    이 대응을 보지 않았다.
    """
    out: list[tuple[str, str]] = []
    for r in json_member(rules, "rules", list):
        if not isinstance(r, dict):
            continue
        region, ref = r.get("body_region"), r.get("limits_ref")
        if region is None and ref is None:
            continue
        rid = r.get("rule_id", "?")
        if ref is None:
            out.append((f"rules.{rid}", f"body_region={region!r} 인데 limits_ref 가 없음"))
            continue
        if region is None:
            out.append((f"rules.{rid}", f"limits_ref={ref!r} 인데 body_region 이 없음"))
            continue
        # 마지막 토큰만 보면 `alternate.BR-05` 처럼 **다른 트리**를 가리켜도 통과한다.
        # 그 트리에 다른 값을 넣어 두면 판정이 조용히 그 값으로 계산된다.
        # 경로 전체를 못박는다.
        if str(ref) != f"contact_limits.{region}":
            out.append((f"rules.{rid}",
                        f"limits_ref={ref!r} 가 'contact_limits.{region}' 이 아님 — "
                        "다른 부위나 다른 트리의 기준으로 판정된다"))
    return out


def rule_parameter_exemption_problems(rules: dict) -> list[tuple[str, str]]:
    """규칙 파라미터에 not_applicable 면제가 붙었는지.

    면제의 뜻은 「표준이 그 자리에 수치를 두지 않는다」이고, 그런 자리는 접촉 금지
    부위의 동적 칸뿐이다(표 A.2 의 「해당 없음」). 규칙 파라미터에는 그 사정이
    없으므로 면제 자체가 성립하지 않는다 — 막지 않으면 SSM 파라미터를 면제로 비워
    두고도 배포 게이트를 통과한다.

    접촉 금지 검사(contact_permission_problems)는 contact_limits 만 순회하므로
    이 경로를 보지 못한다.
    """
    out: list[tuple[str, str]] = []
    for r in json_member(rules, "rules", list):
        if not isinstance(r, dict):
            continue
        rid = r.get("rule_id", "?")
        params = r.get("parameters")
        if params is not None and not isinstance(params, dict):
            # truthy 리스트·문자열을 dict 로 가정하면 .items() 에서 죽는다. 검증기가
            # 죽는 것은 결함 보고가 아니다 — exit 3 은 「검사하지 못했다」이지 「막았다」가 아니다.
            out.append((f"rules.{rid}.parameters",
                        f"객체가 아님: {type(params).__name__}"))
            continue
        for name, e in (params or {}).items():
            if isinstance(e, dict) and e.get("not_applicable") is not None:
                out.append((f"rules.{rid}.parameters.{name}",
                            "규칙 파라미터에는 not_applicable 면제가 없다 — 표준이 "
                            "수치를 두지 않는 자리는 접촉 금지 부위뿐이다"))
    return out


def contact_limits_shape_problems(rules: dict) -> list[tuple[str, str]]:
    """contact_limits 가 있어야 할 것을 갖췄는지. 값이 아니라 **구조**를 본다."""
    out: list[tuple[str, str]] = []
    cl = rules.get("contact_limits")
    if not isinstance(cl, dict):
        return out                       # 부재는 _c1_rules_complete 가 이미 본다
    for region in sorted(BODY_REGIONS & set(cl)):
        body = cl[region]
        loc = f"contact_limits.{region}"
        if not isinstance(body, dict):
            out.append((loc, f"부위 항목이 객체가 아님: {type(body).__name__}"))
            continue
        for group, names in (("limits", REQUIRED_LIMIT_NAMES),
                             ("biomechanical", REQUIRED_BIOMECHANICAL_NAMES)):
            g = body.get(group)
            if not isinstance(g, dict):
                out.append((f"{loc}.{group}",
                            f"그룹이 객체가 아니거나 없음: {type(g).__name__}"))
                continue
            for name in names:
                if name not in g:
                    out.append((f"{loc}.{group}.{name}", "항목이 없음 — 순회 대상이 "
                                                        "되지 않아 임계값 검사가 통째로 비껴간다"))
        if not isinstance(body.get("transient_multiplier"), dict):
            out.append((f"{loc}.transient_multiplier", "항목이 없거나 객체가 아님"))
    return out



def iter_threshold_entries(rules: dict):
    """임계값이 있어야 할 위치를 명시적으로 순회한다.

    "value 와 determined_by 를 가진 객체" 같은 모양 추측으로 찾으면, 필드가 빠진
    항목이 임계값으로 인식되지 않아 검사에서 통째로 빠진다. 위치를 스키마로 못박는다.
    """
    cl = rules.get("contact_limits")
    if isinstance(cl, dict):
        for region, body in cl.items():
            if not isinstance(body, dict):
                continue
            for group in ("limits", "biomechanical"):
                g = body.get(group)
                if isinstance(g, dict):
                    for k, v in g.items():
                        yield f"contact_limits.{region}.{group}.{k}", v
            # transient_multiplier sits beside the two groups, not inside them:
            # it scales both force and pressure, so it belongs to neither. It is
            # still a threshold and must be walked -- an entry nobody iterates is
            # an entry nobody checks.
            if "transient_multiplier" in body:
                yield (f"contact_limits.{region}.transient_multiplier",
                       body["transient_multiplier"])

    rl = rules.get("rules")
    if isinstance(rl, list):
        for i, r in enumerate(rl):
            if not isinstance(r, dict):
                continue
            rid = r.get("rule_id") or f"[{i}]"
            params = r.get("parameters")
            if isinstance(params, dict):
                for k, v in params.items():
                    yield f"rules.{rid}.parameters.{k}", v


def citation_only_thresholds(rules: dict) -> list[str]:
    """값 대신 조항 참조만 실린 임계값의 경로. 계약 위반은 아니지만 조용하면 안 된다."""
    return [path for path, e in iter_threshold_entries(rules)
            if isinstance(e, dict) and e.get("value") is None
            and e.get("not_applicable") is not True
            and e.get("determined_by") == "standard"
            and _is_filled_text(e.get("standard_ref"))]


def threshold_problems(rules: dict) -> list[tuple[str, str]]:
    """임계값 계약 위반을 (경로, 사유) 로 돌려준다.

    계약은 "값이 있어야 한다" 가 아니라 **"라벨의 근거를 재현할 수 있어야 한다"** 이다.
    그 목적을 지키는 길이 셋이고, 셋 다 각자의 증거를 요구한다.

    1. 값이 있다 — 가장 강한 형태. project 임계값은 여기밖에 없다. 우리가 재서
       채우는 값이므로 "표준에 있다" 같은 면제가 성립하지 않는다
    2. 값 대신 standard_ref 가 있다 — 표준이 그 자리에 상수를 주지 않거나(감지 구성별
       식으로 주는 intrusion_distance) 값을 싣지 않은 경우다. 어느 조항의 어느 표인지
       가리키면 받는 쪽이 자기 사본으로 같은 판정을 재현할 수 있다. 표준이 상수로 주는
       값은 이용 허락을 받아 조항 참조와 함께 실려 있다(meta/rules.json)
    3. not_applicable 이다 — 표준이 그 자리에 수치를 두지 않는다(표 A.2 의 「해당 없음」).
       값을 요구하면 정본이 표준과 어긋나므로 받아들이되, reason 을 적게 한다

    2와 3이 없으면 이 검사는 **표준을 다 확보해도 통과할 수 없다** — BR-01·BR-02 의
    동적 접촉 칸이 표준에서 「해당 없음」인데 여기서는 유한한 수치를 요구하게 되기 때문이다.
    """
    out: list[tuple[str, str]] = []
    for path, e in iter_threshold_entries(rules):
        if not isinstance(e, dict):
            out.append((path, "임계값 항목이 객체가 아님"))
            continue
        missing = [f for f in THRESHOLD_FIELDS if f not in e]
        if missing:
            out.append((path, f"필드 누락: {missing}"))
            continue
        v = e["value"]
        na = e.get("not_applicable")
        # 진릿값으로 보면 문자열 "false" 도 참이다. 면제는 명시적으로만 성립한다.
        if na is not None and na is not True and na is not False:
            out.append((path, f"not_applicable 이 불리언이 아님: {na!r}"))

        # 조항 참조는 값이 **없을 때만** 필요한 것이 아니다. 이용 허락을 받아 수치를
        # 함께 싣게 되면 그것은 실린 값의 출처가 되고, not_applicable 일 때는 면제
        # 주장의 근거가 된다. 허락이 놓인 형태 자체가 「수치만 조항 인용과 함께」이므로,
        # 참조가 썩으면 출처 없는 표준 수치가 CC BY 4.0 으로 나간다.
        # 값 미기재 경로는 아래에서 따로 보고하므로 여기서 뺀다 — 한 결함을 두 줄로
        # 보고하면 어느 픽스처가 무엇을 잡는지 흐려진다.
        if (e["determined_by"] == "standard" and (v is not None or na is True)
                and not _is_filled_text(e.get("standard_ref"))):
            out.append((path, "standard 인데 쓸 수 있는 standard_ref 가 없음 — 실린 값과 "
                              "면제 주장은 어느 조항에서 왔는지 가리켜야 한다"))

        if na is True:
            # 면제를 주장하면서 값을 함께 두면 둘 중 하나가 거짓이고, 어느 쪽인지
            # 읽는 사람이 알 수 없다.
            if v is not None:
                out.append((path, f"not_applicable 인데 값이 있음: {v!r}"))
            if not _is_filled_text(e.get("reason")):
                out.append((path, "not_applicable 에 reason 이 없음 — 근거 없는 면제는 은폐다"))
            # 면제는 표준이 그 자리에 수치를 두지 않을 때만 성립한다. project 값에는
            # 그런 사정이 없다 — 우리가 재서 채우는 값이다.
            if e["determined_by"] != "standard":
                out.append((path, f"determined_by={e['determined_by']!r} 인데 not_applicable "
                                  "— 이 면제는 standard 에만 성립한다"))
        elif v is None:
            if e.get("determined_by") == "standard" and _is_filled_text(e.get("standard_ref")):
                pass  # 위 2번
            elif e.get("determined_by") == "standard":
                out.append((path, "값도 쓸 수 있는 standard_ref 도 없음 — 어느 조항의 "
                                  "값인지 문자열로 가리켜야 한다"))
            else:
                out.append((path, "값 미기재"))
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            out.append((path, f"값이 수치가 아님: {v!r}"))
        # NaN 과 ±inf 는 float 이라 위의 수치 검사를 통과한다. 그대로 두면 배포
        # 검증기가 "채워짐" 으로 읽고, 그 임계값을 쓰는 비교는 영영 거짓이 되어
        # 라벨이 조용히 사라진다 — 생성기 쪽 같은 함정을 막는 검사가
        # generation/envs/s1a/mdp/safety.py 의 _is_usable_ssm_value 이며,
        # 둘은 같은 계약이어야 한다 (CONTRIBUTING.md 「함께 고치는 곳」).
        elif not math.isfinite(float(v)):
            out.append((path, f"값이 유한하지 않음: {v!r}"))
        # 이 파일의 임계값은 전부 시간·거리·속도·힘·압력·질량·강성·배수라
        # 음수가 될 수 있는 것이 하나도 없다. 음수는 요구 이격거리를 늘리는 대신
        # 줄이는 쪽으로 작동한다.
        elif float(v) < 0:
            out.append((path, f"임계값이 음수: {v!r}"))

        if not _is_filled_text(e["unit"]):
            out.append((path, f"단위가 비었거나 문자열이 아님: {e['unit']!r} "
                              "— 무단위 임계값은 재현 불가"))
        else:
            field = path.rsplit(".", 1)[-1]
            want = CANONICAL_UNITS.get(field)
            if want is not None and e["unit"] != want:
                out.append((path, f"단위가 {e['unit']!r} 인데 이 항목은 {want!r} 만 쓸 수 "
                                  "있음 — 산출식이 그 단위를 가정한다"))
        if e["determined_by"] not in DETERMINED_BY:
            out.append((path, f"determined_by 가 {sorted(DETERMINED_BY)} 밖의 값: {e['determined_by']!r}"))
    return out


def contact_permission_problems(rules: dict) -> list[tuple[str, str]]:
    """not_applicable 면제가 실제로 접촉 금지 부위에서만 쓰였는지 확인한다.

    이 면제가 없으면 표준을 다 확보해도 C1 을 통과할 수 없다. 그러나 아무 데서나
    쓸 수 있으면 빈 칸을 덮는 가장 싼 방법이 되어 버린다. 면제의 근거는 부위가
    접촉 금지라는 사실 하나뿐이므로, 그 사실과 묶어 둔다.
    """
    out: list[tuple[str, str]] = []
    cl = rules.get("contact_limits")
    if not isinstance(cl, dict):
        return out
    for region, body in cl.items():
        if not isinstance(body, dict):
            continue
        permitted = body.get("contact_permitted")
        if permitted is None:
            out.append((f"contact_limits.{region}", "contact_permitted 없음"))
            continue
        if not isinstance(permitted, bool):
            out.append((f"contact_limits.{region}.contact_permitted",
                        f"불리언이 아님: {permitted!r}"))
            continue
        if not permitted and not _is_filled_text(body.get("contact_prohibited_ref")):
            out.append((f"contact_limits.{region}",
                        "contact_permitted=false 인데 contact_prohibited_ref 가 없음 "
                        "— 어느 조항이 금지하는지 가리켜야 한다"))
        for group in ("limits", "biomechanical"):
            g = body.get(group)
            if not isinstance(g, dict):
                continue
            for k, e in g.items():
                # 면제가 성립하는 자리는 표 A.2 가 「해당 없음」을 둔 **동적 칸** 하나뿐이다.
                # 접촉 금지 부위라고 해서 힘·압력·질량 칸까지 비울 수 있는 것은 아니다 —
                # 표준은 그 값들을 준다.
                if isinstance(e, dict) and e.get("not_applicable") is True:
                    out.append((f"contact_limits.{region}.{group}.{k}",
                                "이 항목에는 not_applicable 면제가 없다 — 표준이 수치를 "
                                "두지 않는 자리는 transient_multiplier 뿐이다"))
        tm = body.get("transient_multiplier")
        if isinstance(tm, dict) and tm.get("not_applicable") is True and permitted:
            out.append((f"contact_limits.{region}.transient_multiplier",
                        "접촉이 허용된 부위인데 not_applicable — 면제 사유가 성립하지 않는다"))
    return out


def _c1_rules_complete(ds: Dataset, rep: Report) -> None:
    """rules.json 의 임계값이 계약을 지키는지 확인한다.

    임계값이 비었거나 단위가 없으면 라벨의 근거를 재현할 수 없다. 라벨이 규칙에서
    자동 산출된다는 것이 이 데이터셋의 핵심 주장이므로, 미기재는 배포 차단 사유다.
    """
    rules = ds.rules()
    if rules is None:
        return  # 파일 없음은 위에서 이미 잡았다
    if not isinstance(rules, dict):
        rep.add("C1", ERROR, f"rules.json 최상위가 객체가 아님: {type(rules).__name__}",
                "meta/rules.json")
        return

    if not isinstance(rules.get("contact_limits"), dict):
        rep.add("C1", ERROR, "rules.json 에 contact_limits 객체 없음", "meta/rules.json")
    else:
        absent = sorted(BODY_REGIONS - set(rules["contact_limits"]))
        if absent:
            rep.add("C1", ERROR, f"contact_limits 에 없는 부위 {len(absent)}개: {absent[:5]}",
                    "meta/rules.json")
    if not isinstance(rules.get("rules"), list):
        rep.add("C1", ERROR, "rules.json 의 rules 가 배열이 아님", "meta/rules.json")

    # 계산에 필요한 입력이 스키마에 없다고 스스로 밝힌 규칙은 적용될 수 없다. 그런 규칙을
    # 담은 채 배포하면 rules.json 이 "무엇을 적용했는가" 를 잘못 말하게 된다.
    if isinstance(rules.get("rules"), list):
        uncomputable = [str(r.get("rule_id")) for r in rules["rules"]
                        if isinstance(r, dict) and r.get("inputs_missing")]
        if uncomputable:
            rep.add("C1", ERROR,
                    f"입력이 없어 계산할 수 없는 규칙 {uncomputable[:3]} — "
                    "스키마에 필드를 추가하거나 배포본에서 제외하십시오",
                    "meta/rules.json")

    # 임계값 "값" 검사와 별개로, 파라미터 이름 자체가 사라진 경우를 잡는다.
    if isinstance(rules.get("rules"), list):
        ids = [str(r.get("rule_id")) for r in rules["rules"] if isinstance(r, dict)]
        # 중복 id 를 그대로 두면 이 dict 는 마지막 항목을 남기는데 생성기 로더는
        # next() 로 첫 항목을 읽는다. 결함본을 앞에, 정상본을 뒤에 두면 검증기는
        # 통과하고 생성기는 미기재로 읽는 상태가 만들어진다 — 실행 재현 확인.
        # scenario_index() 가 시나리오 id 에 대해 이미 하는 것과 같은 거부다.
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            rep.add("C1", ERROR,
                    f"rules.json 에 중복된 rule_id {dupes[:3]} — 어느 항목이 유효한지 "
                    "정해지지 않으므로 검증기와 생성기가 서로 다른 것을 읽는다",
                    "meta/rules.json")
        by_id = {str(r.get("rule_id")): r for r in rules["rules"] if isinstance(r, dict)}
        for rule_id, names in REQUIRED_RULE_PARAMETERS.items():
            entry = by_id.get(rule_id)
            if entry is None:
                continue  # 규칙 자체의 부재는 C4·C8 이 본다
            params = entry.get("parameters")
            if not isinstance(params, dict):
                rep.add("C1", ERROR, f"{rule_id}.parameters 가 객체가 아님", "meta/rules.json")
                continue
            absent = [n for n in names if n not in params]
            if absent:
                rep.add("C1", ERROR,
                        f"{rule_id} 에 필수 파라미터 {absent} 가 없음 — "
                        "산출식이 그 값을 쓰므로 라벨 근거를 재현할 수 없음",
                        "meta/rules.json")

        # The stop table is checked here or nowhere: it is not a parameter (threshold_problems
        # does not walk it), and the near misses cannot be recomputed without it.
        for rule_id in RULES_WITH_STOP_TABLE:
            entry = by_id.get(rule_id)
            if entry is None:
                continue
            for problem in stop_table_problems(entry.get("stop_table")):
                rep.add("C1", ERROR, f"{rule_id}.{problem} — 정지에 필요한 거리를 재현할 수 없음",
                        "meta/rules.json")

    for path, why in rule_region_problems(rules):
        rep.add("C1", ERROR, f"{path}: {why}", "meta/rules.json")

    # 구조를 먼저 본다 — 순회는 있는 것만 보므로, 없는 것은 여기서만 잡힌다.
    for path, why in contact_limits_shape_problems(rules):
        rep.add("C1", ERROR, f"{path}: {why}", "meta/rules.json")

    # 면제는 접촉 금지 부위에서만 성립한다. 아니면 빈 칸을 덮는 가장 싼 방법이 된다.
    for path, why in contact_permission_problems(rules):
        rep.add("C1", ERROR, f"{path}: {why}", "meta/rules.json")

    for path, why in rule_parameter_exemption_problems(rules):
        rep.add("C1", ERROR, f"{path}: {why}", "meta/rules.json")

    problems = threshold_problems(rules)
    status = str(rules.get("status", "")).strip().lower()

    # 값 대신 조항 참조만 실린 임계값은 계약 위반이 아니다(threshold_problems 참고).
    # 그러나 조용히 넘어가면 "채워진 것" 과 구별되지 않고, 배포본만 읽는 사람은 자기
    # 사본 없이 판정을 재현할 수 없다는 사실을 모른 채 쓰게 된다. 세어서 말한다.
    citation_only = citation_only_thresholds(rules)
    if citation_only:
        rep.add("C1", WARN,
                f"임계값 {len(citation_only)}건이 값 없이 조항 참조만 실림 — "
                "받는 쪽이 판정을 재현하려면 해당 표준 사본이 필요하다",
                "meta/rules.json")

    if problems:
        # 사유를 오류 줄에 담는다. INFO 로만 내면 --verbose 없이는 안 보이고, 서로
        # 다른 네 가지 계약 위반이 화면에서 같은 한 줄이 된다 — 하나를 못 잡게 되어도
        # 다른 하나가 같은 문구를 내므로 회귀가 초록이다.
        first = f"{problems[0][0]}: {problems[0][1]}"
        more = f" 외 {len(problems) - 1}건" if len(problems) > 1 else ""
        rep.add("C1", ERROR,
                f"rules.json 임계값 계약 위반 {len(problems)}건 — {first}{more} "
                "— 라벨 근거를 재현할 수 없음",
                "meta/rules.json")
        for path, why in problems[1:6]:
            rep.add("C1", INFO, f"{path}: {why}", "meta/rules.json")
    elif not status:
        rep.add("C1", WARN, "rules.json 에 status 없음", "meta/rules.json")
    elif status != "complete":
        rep.add("C1", WARN,
                f"임계값은 모두 채워졌으나 status={status} — complete 로 갱신하십시오",
                "meta/rules.json")


def _c1_scenarios_sound(ds: Dataset, rep: Report) -> None:
    """scenarios.json 의 적용 규칙 선언이 쓸 수 있는 상태인지 확인한다.

    파일이 있는데 내용이 어긋나면 applicable_rule_ids() 가 보수적 폴백으로 돌아서고,
    왜 갑자기 전 규칙의 입력을 요구하는지 알 수 없게 된다. 폴백은 안전장치이지 정상
    동작이 아니므로, 원인을 여기서 이름 붙여 보고한다.
    """
    loc = "meta/scenarios.json"
    if not (ds.meta / "scenarios.json").exists():
        return                              # 존재 여부는 위에서 이미 보고했다

    raw = ds.scenarios()
    entries = raw.get("scenarios") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        rep.add("C1", ERROR, "scenarios.json 에 시나리오 배열이 없음", loc)
        return

    by_id = scenario_index(ds)
    if by_id is None:
        rep.add("C1", ERROR,
                "scenario_id 가 비었거나 중복임 — 어느 항목이 정본인지 알 수 없음", loc)
        return

    known = known_rule_ids(ds)
    if not known:
        rep.add("C1", ERROR,
                "rules.json 에서 규칙 id 를 하나도 읽지 못함 — 시나리오의 적용 규칙 선언을 "
                "검증할 수 없고 입력 필드 검사가 무력화됨", "meta/rules.json")
        return

    for sid, entry in by_id.items():
        declared = declared_selectors(entry)
        if declared is None:
            rep.add("C1", ERROR,
                    f"{sid}: rules 선언이 비었거나 문자열이 아님 — "
                    "적용 규칙이 공집합이 되어 입력 필드 검사가 무력화됨", loc)
            continue
        dangling = [sel for sel in declared if not selector_resolves(sel, known)]
        if dangling:
            rep.add("C1", ERROR,
                    f"{sid}: rules.json 에 없는 규칙을 가리킴: {dangling[:3]}", loc)

        req = entry.get("required_fields")
        if req is not None and (not isinstance(req, list)
                                or not all(isinstance(r, str) and r.strip() for r in req)):
            rep.add("C1", ERROR, f"{sid}: required_fields 가 문자열 배열이 아님", loc)


# ────────────────────────────────────────────────────────────────
# 카메라 매체 — 배포본이 meta/cameras.json 을 실을 때만 (규격서 §4 · §6 「조건부 필드 — 카메라」)
#
# 검증기는 영상을 디코드하지 않는다. 영상의 크기는 info.json 의 영상 피처에서, 프레임 수는
# meta/episodes 가 적은 영상 구간에서 읽고, 영상 파일은 첫 8바이트(MP4 서명)만 읽는다 — 그래서
# 아래 검사는 **영상 내용에 대해 아무것도 말하지 않는다** (코덱 · 그림 · 멈춘 프레임은 대상 밖).
# 팩은 npz 머리와 step_index 한 멤버만 읽는다. 팩은 동봉한 meta/cameras.json 이 packs 로 선언할
# 때만 싣는다(팩 코드는 두고 정본에서 끈다). 선언하지 않았는데 packs/ 나 meta/packs.json
# 이 있으면 C1 오류다. camera 가 없는 배포본(v0.1)에는 이 절이 한 줄도 내지 않는다.
# ────────────────────────────────────────────────────────────────

@dataclass
class CameraSetup:
    """What the camera checks read from the shipped meta/cameras.json. The pack fields are
    None when the canon declares no pack: then there is no pack to check (C1 refuses one that
    is there anyway)."""
    fps: float
    sizes: dict[str, tuple[int, int]]      # camera -> (width, height), native
    bbox_camera: str                       # bbox.camera: the camera whose picture the box is in
    pack: str | None = None                # packs/<pack>/
    pack_camera: str | None = None
    every_n_steps: int | None = None
    pack_size: tuple[int, int] | None = None   # (width, height) of the shipped pack pictures
    pack_dtypes: dict[str, np.dtype] | None = None   # PACK_PICTURES member -> dtype

    def pack_file(self, episode_index: int) -> str:
        return f"packs/{self.pack}/episode_{episode_index:06d}.npz"   # spec §4


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def read_camera_setup(cams: object) -> tuple[CameraSetup | None, list[str]]:
    """The shipped meta/cameras.json as the camera checks need it, and what is wrong with it.

    Only what C1/C7/C8 read is required: fps, each camera's size, the box's camera
    (bbox.camera, one of the cameras -- C8 holds the box to its size) and, when the canon
    declares one, the pack (its camera -- the box's, as both come from one segmentation --
    downsample and picture dtypes). Packs are optional: none (``packs`` absent, null or {}),
    or one; a canon with more is refused, as the converter refuses it
    (generation/convert_lerobot.the_pack). Mounting, clipping, channels and the pose
    conventions are the consumer's and are not checked. The camera set has to be exactly the
    one the conditional video fields name -- a camera this validator has no field for is a
    camera nothing here checks. None when anything needed is unusable: the checks that need it
    then say they could not compare rather than guess.
    """
    if not isinstance(cams, dict):
        return None, [f"최상위가 객체가 아님: {type(cams).__name__}"]
    problems: list[str] = []
    fps = numeric_or_none(cams.get("fps"))
    if fps is None or fps <= 0:
        problems.append(f"fps 가 양수가 아님: {cams.get('fps')!r}")
    cameras = cams.get("cameras")
    sizes: dict[str, tuple[int, int]] = {}
    if not isinstance(cameras, dict) or not cameras:
        problems.append("cameras 가 비었거나 객체가 아님")
    else:
        for name, spec in cameras.items():
            if (not isinstance(spec, dict) or not _positive_int(spec.get("width"))
                    or not _positive_int(spec.get("height"))):
                problems.append(f"카메라 {name} 의 width·height 가 양의 정수가 아님")
                continue
            sizes[name] = (spec["width"], spec["height"])
        want = {f[len(VIDEO_FIELD_PREFIX):] for f in camera_video_fields()}
        if set(cameras) != want:
            problems.append(
                f"카메라 {sorted(cameras)} 가 규격의 영상 필드가 말하는 카메라 {sorted(want)} 와 "
                "다름 — 규격에 필드가 없는 카메라는 검사받지 않고, 빠진 카메라는 영상이 없다")
    # A consumer finds the box column and its not-visible marker here; C8 checks the data
    # against the validator's own two, so the shipped block has to name the same ones.
    box = cams.get("bbox")
    if (not isinstance(box, dict) or box.get("field") != BBOX_FIELD
            or box.get("invalid") != list(BBOX_INVALID)):
        problems.append(f"bbox 의 field·invalid 가 규격의 {BBOX_FIELD} · {list(BBOX_INVALID)} 가 아님")
    box_camera = box.get("camera") if isinstance(box, dict) else None
    # Unhashable (a list, an object) is refused before `in`, which would raise TypeError.
    if not isinstance(box_camera, str) or not isinstance(cameras, dict) or box_camera not in cameras:
        problems.append(f"bbox.camera {box_camera!r} 가 cameras 의 카메라가 아님 — 상자가 어느 카메라의 "
                        "그림에 있는지 모르면 C8 이 상자의 범위를 대조하지 못한다")
        box_camera = None
    packs = cams.get("packs")
    if packs is None:
        packs = {}
    if not isinstance(packs, dict) or len(packs) > 1:
        problems.append("packs 가 팩 하나를 담은 객체가 아님 — 팩은 정본이 선언할 때만 싣고, 선언하면 "
                        "에피소드마다 하나다 (사이드카의 media.pack 이 하나)")
        return None, problems
    if not packs:
        if problems:
            return None, problems
        return CameraSetup(fps=float(fps), sizes=sizes, bbox_camera=box_camera), []
    ((name, pack),) = packs.items()
    if not isinstance(pack, dict):
        return None, problems + [f"팩 {name} 이 객체가 아님"]
    camera = pack.get("camera")
    # A list or an object here is unhashable: `in sizes` alone raised TypeError (exit 3).
    if not isinstance(camera, str) or camera not in sizes:
        problems.append(f"팩 {name} 의 camera {camera!r} 가 크기를 가진 카메라가 아님")
    elif box_camera is not None and camera != box_camera:
        problems.append(f"팩 {name} 의 camera {camera!r} 가 bbox.camera {box_camera!r} 와 다름 — 상자와 "
                        "팩의 마스크는 한 카메라의 분할에서 나온다")
    down = pack.get("downsample")
    if not isinstance(down, dict) or not all(
            _positive_int(down.get(k)) for k in ("width", "height", "every_n_steps")):
        problems.append(f"팩 {name} 의 downsample 이 width·height·every_n_steps 양의 정수가 아님")
    elif isinstance(camera, str) and camera in sizes:
        # The converter keeps every k-th pixel (generation/convert_lerobot.downsample_factors),
        # so a size that does not divide the camera's is one no conversion produces -- and C7
        # would hold a shipped pack to it. Restated here, as this file
        # does not import generation; tests/validator/test_media_checks.py holds the two.
        for axis, native in zip(("width", "height"), sizes[camera]):
            if native % down[axis]:
                problems.append(
                    f"팩 {name} 의 downsample.{axis} {down[axis]} 가 카메라 {camera} 의 원 해상도 "
                    f"{native} 의 정수 분의 일이 아님 — 축소는 픽셀을 골라 남기므로(평균 내지 않음) "
                    "원 해상도를 나눠떨어뜨려야 한다")
    dtypes: dict[str, np.dtype] = {}
    for member in PACK_PICTURES:
        block = pack.get(member)
        declared = block.get("dtype") if isinstance(block, dict) else None
        try:
            # np.dtype(None) is float64 -- only a written-down name counts. A string numpy
            # cannot parse raises TypeError, ValueError or SyntaxError (200,000 random
            # strings, 2026-09-28: "," raised SyntaxError past a TypeError-only handler).
            dtypes[member] = np.dtype(declared) if isinstance(declared, str) else None
        except (TypeError, ValueError, SyntaxError):
            dtypes[member] = None
        if dtypes[member] is None:
            problems.append(f"팩 {name} 의 {member}.dtype 이 numpy 형 이름이 아님: {declared!r}")
    if problems:
        return None, problems
    return CameraSetup(fps=float(fps), sizes=sizes, bbox_camera=box_camera, pack=str(name),
                       pack_camera=camera, every_n_steps=down["every_n_steps"],
                       pack_size=(down["width"], down["height"]), pack_dtypes=dtypes), []


def declared_packs(cams: object) -> bool | None:
    """Whether the shipped meta/cameras.json declares a pack: packs ship only when it does.
    None when the document cannot say (it is not an object -- C1 says why). A
    ``packs`` block that is there but malformed still declares one: read_camera_setup says what
    is wrong with it, and the pack files are still looked at."""
    if not isinstance(cams, dict):
        return None
    return cams.get("packs") not in (None, {})


def camera_setup_of(ds: Dataset) -> CameraSetup | None:
    """The setup for a check other than C1: C1 reports why it is None."""
    cams = ds.cameras()
    return read_camera_setup(cams)[0] if cams is not None else None


def camera_packs_of(ds: Dataset) -> tuple[CameraSetup | None, bool | None]:
    """The setup and whether the canon declares a pack (declared_packs), from one read of
    meta/cameras.json -- each read of a file that does not parse is reported again."""
    cams = ds.cameras()
    return (read_camera_setup(cams)[0] if cams is not None else None), declared_packs(cams)


def camera_media_without_canon(ds: Dataset, info: dict | None,
                               frames: Iterable[tuple[Path, pd.DataFrame]]) -> list[str]:
    """Camera media in a distribution that does not ship meta/cameras.json."""
    found: list[str] = []
    features = info.get("features") if isinstance(info, dict) else None
    if isinstance(features, dict):
        found += sorted(k for k, v in features.items()
                        if isinstance(v, dict) and v.get("dtype") == "video")
    found += [f"{rel}/" for rel in ("videos", "packs") if (ds.root / rel).exists()]
    if (ds.meta / "packs.json").exists():
        found.append("meta/packs.json")
    columns = set(camera_column_fields())
    found += sorted({c for _, df in frames for c in df.columns if c in columns})
    return found


def _inside(root: Path, rel: object) -> Path | None:
    """root/rel when rel is a relative path that stays inside the distribution."""
    if not isinstance(rel, str) or not rel:
        return None
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        return None
    return root / p


def _index_or_none(value: object) -> int | None:
    n = numeric_or_none(value)
    if n is None or n < 0 or not float(n).is_integer():
        return None
    return int(n)


def _episode_labels(eps: pd.DataFrame) -> list[str]:
    """What a per-episode finding is located by: episode_id, as C2~C6 do."""
    if "episode_id" in eps.columns:
        return [str(x) for x in eps["episode_id"]]
    if "episode_index" in eps.columns:
        return [f"episode_index={x}" for x in eps["episode_index"]]
    return [f"meta/episodes#{i}" for i in range(len(eps))]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def npz_layout(path: Path) -> dict[str, tuple[tuple, np.dtype]]:
    """name -> (shape, dtype) of each array in an npz, from the .npy headers alone.

    The converter has the same reader (generation/convert_lerobot.pack_layout); this is a
    copy because the validator does not import generation code. Nothing is inflated."""
    out: dict[str, tuple[tuple, np.dtype]] = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            with zf.open(info) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    raise ValueError(f"읽을 수 없는 npy 형식 {version}: {info.filename}")
            out[info.filename.removesuffix(".npy")] = (tuple(shape), np.dtype(dtype))
    return out


def bbox_problems(values: Iterable, size: tuple[int, int],
                  camera: str) -> tuple[list[str], int]:
    """[x_min, y_min, x_max, y_max] per step: integer pixel indices, inclusive at both ends, in
    the native picture of the box's camera (meta/cameras.json bbox.camera, bbox.coords) -- or
    the not-visible marker. The box is computed from that camera's segmentation at record time,
    before any pack downsample, so the range is the camera's size, not a pack's. Returns
    (problems, rows range-checked).

    Rows of another length are C8's fixed-shape finding. A row that passes that check without
    being an array of four -- a JSON string "[0, 0, 99, 99]" (elem_len parses it), a nested
    list -- is said here: skipping it printed OK over boxes nobody compared."""
    width, height = size
    values = list(values)
    rows = [np.asarray(v) for v in values]
    boxes = [r for r in rows if r.shape == (len(BBOX_INVALID),)]
    out: list[str] = []
    unread = [v for v, r in zip(values, rows)
              if r.shape != (len(BBOX_INVALID),) and elem_len(v) == len(BBOX_INVALID)]
    if unread:
        out.append(f"{BBOX_FIELD} 가 수 4개의 배열이 아닌 행 {len(unread)}개 (예: {unread[0]!r:.40}) — "
                   "범위를 대조하지 못함. 상자는 JSON 문자열이 아니라 정수 배열 열이다")
    if not boxes:
        return out, 0
    arr = np.stack(boxes)
    if arr.dtype.kind not in "iu":
        return out + [f"{BBOX_FIELD} 가 정수 픽셀 번호가 아님 ({arr.dtype}) — 양끝을 포함하는 픽셀 "
                      "번호다 (meta/cameras.json bbox.coords)"], 0
    hidden = (arr == np.asarray(BBOX_INVALID)).all(axis=1)
    x0, y0, x1, y1 = arr.T
    inside = (0 <= x0) & (x0 <= x1) & (x1 < width) & (0 <= y0) & (y0 <= y1) & (y1 < height)
    bad = np.flatnonzero(~hidden & ~inside)
    if len(bad):
        out.append(f"{BBOX_FIELD} 가 영상 범위 밖이거나 뒤집힌 행 {len(bad)}개 (예: {arr[bad[0]].tolist()}) "
                   f"— 양끝을 포함하는 픽셀 번호라 0 ≤ x_min ≤ x_max < {width} · 0 ≤ y_min ≤ y_max < "
                   f"{height} ({camera} 원 해상도) 이거나, 보이지 않으면 {list(BBOX_INVALID)} 이어야 한다 "
                   "(meta/cameras.json bbox.coords)")
    return out, len(arr)


def _c1_camera_media(ds: Dataset, info: dict | None, eps: pd.DataFrame | None,
                     frames: Iterable[tuple[Path, pd.DataFrame]], rep: Report) -> None:
    """C1 — a camera distribution's files: meta/cameras.json, the streams, the packs.

    Only when the distribution ships meta/cameras.json: that file is the condition that
    makes the camera fields required (spec §4). Without it the one thing looked at is
    whether camera media are there anyway -- then the condition file was lost, and every
    camera check below would be off without a word.
    """
    if not ds.ships_cameras():
        found = camera_media_without_canon(ds, info, frames)
        if found:
            rep.add("C1", ERROR,
                    f"meta/cameras.json 이 없는데 카메라 매체가 있음: {found[:4]} — 이 파일이 "
                    "카메라 필드를 필수로 만드는 조건이라, 없으면 영상·팩·상자 검사가 전부 꺼진다 "
                    "(규격서 §4)", "meta/cameras.json")
        return
    cams = ds.cameras()                  # a parse failure is reported by _json
    setup, problems = read_camera_setup(cams) if cams is not None else (None, [])
    for problem in problems:
        rep.add("C1", ERROR, f"meta/cameras.json 의 {problem}", "meta/cameras.json")
    if isinstance(info, dict):
        _c1_video_features(info, setup, rep)
    stats = rep.stats.setdefault("camera_media", {})
    stats["video_files"] = _c1_video_files(ds, info, eps, rep)
    stats["pack_files"] = _c1_pack_files(ds, setup, declared_packs(cams), rep)


def _c1_video_features(info: dict, setup: CameraSetup | None, rep: Report) -> None:
    """Every camera stream is a LeRobot video feature at its camera's size and the dataset's
    rate: the shipped meta/cameras.json fps, info.json fps and each feature's video.fps are
    one number -- otherwise a video frame is not a step. Read from info.json only."""
    loc = "meta/info.json"
    features = info.get("features")
    if not isinstance(features, dict):
        rep.add("C1", ERROR, "info.json 에 features 가 없어 영상 피처를 확인할 수 없음", loc)
        return
    fps = numeric_or_none(info.get("fps"))
    if setup is not None and (fps is None or not math.isclose(fps, setup.fps, rel_tol=1e-9)):
        rep.add("C1", ERROR,
                f"meta/cameras.json 의 fps {setup.fps:g} 가 info.json 의 fps {info.get('fps')!r} 와 "
                "다름 — 영상 한 프레임이 스텝 하나라는 전제가 깨진다", loc)
    wanted = camera_video_fields()
    extra = sorted(k for k, v in features.items()
                   if isinstance(v, dict) and v.get("dtype") == "video" and k not in wanted)
    if extra:
        rep.add("C1", ERROR,
                f"info.json 에 규격에 없는 영상 피처 {extra} — 검증기가 모르는 영상은 검사받지 않는다", loc)
    for key in wanted:
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            rep.add("C1", ERROR,
                    f"info.json 에 영상 피처 {key} (dtype video) 가 없음 — meta/cameras.json 을 싣는 "
                    "배포본은 이 영상도 싣는다", loc)
            continue
        size = setup.sizes.get(key[len(VIDEO_FIELD_PREFIX):]) if setup is not None else None
        if size is not None and feature.get("shape") != [size[1], size[0], 3]:
            rep.add("C1", ERROR,
                    f"영상 피처 {key} 의 shape {feature.get('shape')!r} 가 meta/cameras.json 의 "
                    f"[{size[1]}, {size[0]}, 3] 과 다름", loc)
        encoded = feature.get("info")
        if not isinstance(encoded, dict):
            rep.add("C1", ERROR,
                    f"영상 피처 {key} 에 info 가 없어 인코딩된 영상의 크기·fps 를 대조할 수 없음", loc)
            continue
        video_fps = numeric_or_none(encoded.get("video.fps"))
        if fps is not None and (video_fps is None
                                or not math.isclose(video_fps, fps, rel_tol=1e-9)):
            rep.add("C1", ERROR,
                    f"영상 피처 {key} 의 video.fps {encoded.get('video.fps')!r} 가 info.json 의 fps "
                    f"{fps:g} 와 다름 — 영상 한 프레임이 스텝 하나가 아니다", loc)
        if size is not None and (encoded.get("video.width"), encoded.get("video.height")) != size:
            rep.add("C1", ERROR,
                    f"영상 피처 {key} 의 video.width·height {encoded.get('video.width')!r}×"
                    f"{encoded.get('video.height')!r} 가 meta/cameras.json 의 {size[0]}×{size[1]} 과 다름",
                    loc)


def _starts_as_mp4(path: Path) -> bool:
    """The file opens with an ISO-BMFF ftyp box (MP4_SIGNATURE at MP4_SIGNATURE_AT). Eight
    bytes are read; nothing is decoded."""
    with open(path, "rb") as f:
        head = f.read(MP4_SIGNATURE_AT + len(MP4_SIGNATURE))
    return head[MP4_SIGNATURE_AT:] == MP4_SIGNATURE


def _c1_video_files(ds: Dataset, info: dict | None, eps: pd.DataFrame | None,
                    rep: Report) -> int:
    """The stream directories, and the file each episode's span sits in. Returns how many
    distinct files were looked at.

    The file a span names is info.json video_path filled with the camera's field and the
    span's (chunk_index, file_index) -- a template from the distribution, so what it resolves
    to is checked, not trusted: a path inside the distribution, under this camera's own
    videos/<field>/ (else another camera's stream, or any file at all, reads as this one --
    "meta/info.json" passed as both cameras), one file per (chunk_index,
    file_index) (a template that drops a number sends one file's episodes to the other's
    pictures), present, non-empty and opening with the MP4 signature. Nothing is decoded:
    codec, frame count and pictures stay out of reach (validator/README.md)."""
    checked = 0
    template = info.get("video_path") if isinstance(info, dict) else None
    if eps is not None and (not isinstance(template, str) or not template.strip()):
        rep.add("C1", ERROR, "info.json 에 video_path 가 없어 에피소드의 영상 파일을 찾을 수 없음",
                "meta/info.json")
        template = None
    labels = _episode_labels(eps) if eps is not None else []
    for key in camera_video_fields():
        if not (ds.root / VIDEO_DIR / key).is_dir():
            rep.add("C1", ERROR,
                    f"영상 디렉터리 없음 — meta/cameras.json 을 싣는 배포본은 {key} 영상을 싣는다",
                    f"{VIDEO_DIR}/{key}/")
            continue
        if eps is None or template is None:
            continue
        cols = [f"videos/{key}/chunk_index", f"videos/{key}/file_index"]
        absent = [c for c in cols if c not in eps.columns]
        if absent:
            rep.add("C1", ERROR,
                    f"meta/episodes 에 {absent} 열이 없어 에피소드의 {key} 영상 파일을 찾을 수 없음",
                    "meta/episodes")
            continue
        refs: dict[str, list[str]] = {}
        numbers: dict[str, set[tuple[int, int]]] = {}
        for label, chunk, file in zip(labels, eps[cols[0]].tolist(), eps[cols[1]].tolist()):
            c, f = _index_or_none(chunk), _index_or_none(file)
            if c is None or f is None:
                rep.add("C1", ERROR,
                        f"{key} 의 chunk_index·file_index 가 0 이상의 정수가 아님: {chunk!r}, {file!r}",
                        label)
                continue
            try:
                rel = template.format(video_key=key, chunk_index=c, file_index=f)
            # "{video_key.x}" raises AttributeError and "{chunk_index[0]}" TypeError -- a
            # template from info.json is input, and the validator must not die on it.
            except (KeyError, IndexError, ValueError, AttributeError, TypeError) as e:
                rep.add("C1", ERROR, f"info.json video_path {template!r} 를 채울 수 없음: "
                                     f"{type(e).__name__}: {e}", "meta/info.json")
                return checked
            refs.setdefault(rel, []).append(label)
            numbers.setdefault(rel, set()).add((c, f))
        for rel, owners in sorted(refs.items()):
            path = _inside(ds.root, rel)
            if path is None:
                rep.add("C1", ERROR, f"영상 경로가 배포본 밖을 가리킴: {rel}", "meta/info.json")
                continue
            if Path(rel).parts[:2] != (VIDEO_DIR, key):
                rep.add("C1", ERROR,
                        f"{key} 의 영상 경로 {rel} 가 이 카메라의 디렉터리 {VIDEO_DIR}/{key}/ 밖을 "
                        "가리킴 — info.json video_path 가 다른 카메라의 영상이나 영상이 아닌 파일을 이 "
                        "카메라의 영상으로 읽게 한다", "meta/info.json")
                continue
            if len(numbers[rel]) > 1:
                rep.add("C1", ERROR,
                        f"{key} 의 파일 번호 (chunk_index, file_index) {sorted(numbers[rel])[:3]} 가 "
                        f"모두 한 파일 {rel} 로 풀림 — info.json video_path 가 영상 파일을 구분하지 "
                        "않아, 다른 파일에 있다고 적힌 에피소드가 이 파일의 그림을 가리킨다",
                        "meta/info.json")
            checked += 1
            if not path.is_file() or path.stat().st_size == 0:
                rep.add("C1", ERROR,
                        f"영상 파일이 없거나 비어 있음 — 에피소드 {len(owners)}건의 영상 구간이 이 "
                        f"파일에 있다고 적혀 있다 (예: {owners[:3]})", rel)
            elif not _starts_as_mp4(path):
                rep.add("C1", ERROR,
                        f"영상 파일이 MP4 가 아님 — 첫 상자가 ftyp 가 아니다(ISO-BMFF 서명. 첫 8바이트만 "
                        f"읽고 디코드하지 않는다). 에피소드 {len(owners)}건의 영상 구간이 이 파일에 있다고 "
                        f"적혀 있다 (예: {owners[:3]})", rel)
    return checked


def _c1_pack_files(ds: Dataset, setup: CameraSetup | None, declared: bool | None,
                   rep: Report) -> int:
    """packs/ and meta/packs.json: every listed pack at its spec path, present, and with the
    SHA-256 the list states; no pack the list does not name. Returns how many were hashed.

    **Only when the shipped canon declares a pack** (``declared``, declared_packs): the packs
    are switched off in the canon, not deleted. A canon that declares none has
    no pack to check -- and then packs/ or meta/packs.json being there is the finding (work
    done on top of an old conversion's copy: packs no canon describes, and no check below
    would look at them). The same shape as camera_media_without_canon, one level down. A
    canon that cannot say (None: not an object) is said, not guessed either way.

    **The SHA-256 is verified here, on every run.** The list states it so a consumer can
    check what they downloaded. The converter computes it; the publish pre-flight checks it
    once more on its own before an upload (publish_dataset.shipped_packs_match_index); the
    Hub round trip compares the upload with the local files, not with the list. Cost,
    measured 2026-09-28: 1.141 GB of packs in 0.30 s (3.8 GB/s, page
    cache warm) -- about 25 s for a 400-episode camera batch (~90 GB) when warm, disk-bound
    when cold, against the ~6.5 s the rest of the validator took on the 400-episode camera-less release
    it was measured with (383,462 steps).
    """
    loc = "meta/packs.json"
    if declared is None:
        rep.add("C1", ERROR, "meta/cameras.json 을 읽지 못해 팩을 선언했는지 알 수 없음 — packs/ · "
                             "meta/packs.json 을 대조하지 못함 (사유는 meta/cameras.json 의 다른 C1 항목)",
                "meta/cameras.json")
        return 0
    if not declared:
        found = [rel for rel in ("packs/", "meta/packs.json") if (ds.root / rel).exists()]
        if found:
            rep.add("C1", ERROR,
                    f"정본이 팩을 선언하지 않았는데 팩이 있음: {found} — 팩은 meta/cameras.json 이 packs "
                    "로 선언할 때만 싣는다. 선언 없는 팩은 어느 카메라·축소의 것인지 알 수 없고 아무 검사도 "
                    "받지 않는다 (옛 변환 사본 위에서 작업했다면 다시 변환하라)", found[0])
        return 0
    if not (ds.root / "packs").is_dir():
        rep.add("C1", ERROR, "packs/ 없음 — 팩을 선언한 meta/cameras.json 을 싣는 배포본은 에피소드마다 "
                             "깊이·마스크 팩을 싣는다", "packs/")
    if not (ds.meta / "packs.json").exists():
        rep.add("C1", ERROR, "meta/packs.json 없음 — 팩 목록(파일·프레임 수·SHA-256)이 없으면 "
                             "받는 쪽이 팩을 에피소드에 잇지도, 받은 것을 확인하지도 못한다", loc)
        return 0
    manifest, problems = ds.packs_manifest()
    for problem in problems:
        rep.add("C1", ERROR, f"meta/packs.json 의 {problem}", loc)
    if manifest is None:
        return 0
    if not isinstance(manifest, dict):
        rep.add("C1", ERROR, f"meta/packs.json 최상위가 객체가 아님: {type(manifest).__name__}", loc)
        return 0
    hashed = 0
    listed: set[str] = set()
    for key, entry in manifest.items():
        where = f"meta/packs.json[{key}]"
        # ASCII digits only, no leading zero. str.isdigit accepts "²", which int() refuses.
        index = int_of_digits(key) if re.fullmatch(r"0|[1-9][0-9]*", key) else None
        if index is None:
            rep.add("C1", ERROR, f"키 {key!r} 가 에피소드 번호(episode_index)가 아님", where)
            continue
        if not isinstance(entry, dict):
            rep.add("C1", ERROR, f"항목이 객체가 아님: {type(entry).__name__}", where)
            continue
        rel = entry.get("file")
        if setup is not None and rel != setup.pack_file(index):
            rep.add("C1", ERROR,
                    f"규격 경로 {setup.pack_file(index)} 가 아닌 {rel!r} 를 가리킴 — 다른 "
                    "에피소드의 팩이 이 에피소드로 읽힌다", where)
            continue
        path = _inside(ds.root, rel)
        if path is None:
            rep.add("C1", ERROR, f"file 이 배포본 안의 상대 경로가 아님: {rel!r}", where)
            continue
        listed.add(str(Path(rel)))
        if not path.is_file():
            rep.add("C1", ERROR, f"meta/packs.json 이 가리키는 팩이 배포본에 없음: {rel}", where)
            continue
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            rep.add("C1", ERROR, f"sha256 이 64자리 소문자 16진수가 아님: {digest!r}", where)
            continue
        hashed += 1
        if _sha256(path) != digest:
            rep.add("C1", ERROR,
                    f"팩 {rel} 의 SHA-256 이 meta/packs.json 과 다름 — 받는 쪽의 무결성 확인이 "
                    "바르게 받은 파일을 거부하거나, 바뀐 팩을 통과시킨다", where)
    if (ds.root / "packs").is_dir():
        unlisted = sorted(str(q.relative_to(ds.root)) for q in (ds.root / "packs").rglob("*")
                          if q.is_file() and str(q.relative_to(ds.root)) not in listed)
        if unlisted:
            rep.add("C1", ERROR,
                    f"packs/ 에 meta/packs.json 이 적지 않은 파일 {len(unlisted)}개: {unlisted[:3]} — "
                    "목록에 없는 팩은 어느 에피소드의 것인지 알 수 없다", "packs/")
    return hashed


def c2_joint_dims(ds: Dataset, eps: pd.DataFrame | None, rep: Report) -> None:
    """C2 — len(joint_names) == dof == 관절 벡터 차원, 매핑 파일 정합."""
    if eps is None:
        rep.add("C2", SKIP, "에피소드 메타 없음")
        return

    maps = ds.joint_maps()
    dof_by_model: dict[str, int] = {}

    for _, row in eps.iterrows():
        eid = str(row.get("episode_id", "?"))
        names = as_list(row.get("joint_names"))
        dof = row.get("dof")

        if names is None:
            rep.add("C2", ERROR, "joint_names 없음 또는 리스트 아님", eid)
            continue
        if pd.isna(dof):
            rep.add("C2", ERROR, "dof 없음", eid)
            continue

        # A name that is not a string is said as such, as C4 says it of triggered_rules: read as
        # text, the number 1 would match a map's joint "1". The rest is read as text all the same --
        # a list or an object among the names cannot go into the set and the lookups below.
        nonstr = [n for n in names if not isinstance(n, str)]
        if nonstr:
            rep.add("C2", ERROR,
                    f"joint_names 에 문자열 아닌 원소: {[type(n).__name__ for n in nonstr[:3]]}", eid)
        names = [str(n) for n in names]
        dof = int(dof)
        if len(names) != dof:
            rep.add("C2", ERROR, f"len(joint_names)={len(names)} != dof={dof}", eid)
        if len(set(names)) != len(names):
            dups = [n for n, c in Counter(names).items() if c > 1]
            rep.add("C2", ERROR, f"joint_names 중복: {dups[:5]}", eid)

        model = str(row.get("robot_model", ""))
        if model:
            prev = dof_by_model.get(model)
            if prev is not None and prev != dof:
                rep.add("C2", ERROR, f"동일 robot_model 의 dof 불일치: {prev} vs {dof}", eid)
            dof_by_model[model] = dof

            jm = maps.get(model)
            if jm is None:
                rep.add("C2", ERROR, f"매핑 파일 없음: meta/joint_map/{model}.yaml", eid)
            else:
                mnames = as_list(jm.get("joint_names")) or []
                if list(mnames) != list(names):
                    rep.add("C2", ERROR,
                            "joint_names 가 매핑 파일과 불일치 — 순서까지 같아야 함", eid)
                cmap = jm.get("canonical_map") or {}
                unmapped = [n for n in names if n not in cmap]
                declared = set(as_list(jm.get("unmapped")) or [])
                undeclared = [n for n in unmapped if n not in declared]
                if undeclared:
                    rep.add("C2", WARN,
                            f"의미 키 미매핑 관절 {len(undeclared)}개 (unmapped 미선언): {undeclared[:5]}",
                            eid)

    rep.stats["dof_by_model"] = dof_by_model


def c3_contact_regions(ds: Dataset, frames: list[tuple[Path, pd.DataFrame]], rep: Report) -> None:
    """C3 — 접촉 부위 코드 및 배열 길이 정합.

    **전 프레임을 봅니다.** 부위 구성은 에피소드 안에서 고정이지만, 그건 생성기가 옳게
    동작했을 때의 이야기다. 첫 프레임만 보면 중간부터 어긋난 파일이 그대로 통과한다 —
    실제로 2번째 프레임의 부위 코드와 배열 길이가 검사를 빠져나갔다.
    파일당 사유별로 한 번씩만 보고해 출력이 폭주하지 않게 한다.
    """
    if not frames:
        rep.add("C3", SKIP, "스텝 데이터 없음")
        return

    # 있는 것은 모두 region_names 와 길이가 맞아야 한다. 없다고 ERROR 를 내는 것은
    # 필수인 둘뿐이다 -- v_rel 과 pressure 는 선택 필드다 (스키마 §6). v_rel 은 어느
    # 판정 규칙도 읽지 않고, pressure 는 접촉 면적을 실측할 수 있을 때만 실린다
    # (「선택 필드 — observation.contact.pressure」 — PhysX 는 힘을 보고하고 면적을
    # 보고하지 않으므로 sim 데이터에는 없다).
    ARRAYS = ("observation.contact.force", "observation.contact.v_rel",
              "observation.contact.pressure", "observation.contact.type")
    REQUIRED_ARRAYS = tuple(c for c in ARRAYS if c in REQUIRED_STEP_FIELDS)
    seen_regions: set[str] = set()

    for path, df in frames:
        loc = str(path)
        if "observation.contact.region_names" not in df.columns:
            rep.add("C3", ERROR, "region_names 컬럼 없음", loc)
            continue

        absent = [c for c in REQUIRED_ARRAYS if c not in df.columns]
        for c in absent:
            rep.add("C3", ERROR, f"{c} 컬럼 없음", loc)
        present = [c for c in ARRAYS if c in df.columns]

        # 컬럼을 한 번에 파이썬 리스트로 꺼내 행마다 .iloc 하는 비용을 피한다.
        region_col = df["observation.contact.region_names"].tolist()
        other_cols = {c: df[c].tolist() for c in present}

        bad_codes: set[str] = set()
        non_list = False
        len_bad: dict[str, tuple[int, int, int]] = {}   # 컬럼 → (프레임, 실제, 기대)
        region_len: set[int] = set()

        for i, regions in enumerate(region_col):
            rl = as_list(regions)
            if rl is None:
                if not non_list:
                    rep.add("C3", ERROR, "region_names 가 리스트 아님", f"{loc}#{i}")
                    non_list = True
                continue
            # A JSON list may hold what is not a string -- a number, null, a list, an object -- and
            # a list or an object cannot go into a set, nor a number be sorted beside a string.
            # Read each element as text: what is not a code is a code that is not allowed.
            rl = [str(r) for r in rl]
            bad_codes |= set(rl) - BODY_REGIONS
            seen_regions.update(rl)
            region_len.add(len(rl))

            for c in present:
                n = elem_len(other_cols[c][i])
                if n != len(rl) and c not in len_bad:
                    len_bad[c] = (i, n, len(rl))

        if bad_codes:
            rep.add("C3", ERROR, f"허용되지 않은 부위 코드: {sorted(bad_codes)[:5]}", loc)
        if len(region_len) > 1:
            rep.add("C3", ERROR,
                    f"region_names 길이가 프레임마다 다름: {sorted(region_len)[:5]}", loc)
        for c, (i, n, want) in len_bad.items():
            rep.add("C3", ERROR,
                    f"{c} 길이 {n} != region_names 길이 {want}", f"{loc}#{i}")

    missing = sorted(BODY_REGIONS - seen_regions)
    if missing:
        rep.add("C3", INFO, f"데이터에 등장하지 않은 부위 코드 {len(missing)}개: {missing}")
    rep.stats["regions_seen"] = sorted(seen_regions)


def c4_label_evidence(ds: Dataset, eps: pd.DataFrame | None, rep: Report) -> None:
    """C4 — outcome 과 triggered_rules 의 쌍방 정합. 규칙 ID 존재 확인.

    The outcome shares are counted over the random rows: a row whose condition_design is "grid" -- a cell of a
    conditions table's grid, set and not drawn -- is left out of them (the count is grid_episodes_not_counted); a
    release without the column, and a row whose design is null, is counted whole. The label checks run on every row."""
    if eps is None:
        rep.add("C4", SKIP, "에피소드 메타 없음")
        return

    rules = ds.rules()
    known: set[str] = set()
    if isinstance(rules, dict):
        if isinstance(rules.get("rules"), list):
            known = {str(r.get("rule_id")) for r in rules["rules"] if isinstance(r, dict)}
        else:
            known = {str(k) for k in rules.keys()}
    elif isinstance(rules, list):
        known = {str(r.get("rule_id")) for r in rules if isinstance(r, dict)}
    if not known:
        rep.add("C4", WARN, "rules.json 에서 rule_id 집합을 찾지 못함 — 존재 검사 생략")

    # 와일드카드 선언은 오타를 통과시킨다. R-TS15066-A2-* 처럼 뭉뚱그리면
    # R-TS15066-A2-99 같은 없는 부위 코드도 검사를 빠져나간다.
    wild = sorted(k for k in known if k.endswith("*"))
    if wild:
        rep.add("C4", WARN,
                f"rules.json 에 와일드카드 rule_id {wild[:3]} — 부위별로 전개하면 오타를 잡을 수 있음",
                "meta/rules.json")

    dist: Counter = Counter()
    grid_rows = 0
    for _, row in eps.iterrows():
        eid = str(row.get("episode_id", "?"))
        outcome = row.get("outcome")
        raw_trig = row.get("triggered_rules")
        trig = as_list(raw_trig)
        if trig is None:
            # 목록으로 읽히지 않는 값을 빈 목록으로 삼키면, 스칼라 "R-INVENTED" 가
            # 「근거 없음」이 되어 없는 규칙 ID 검사 자체를 비껴간다.
            if not scalar_missing(raw_trig):
                rep.add("C4", ERROR,
                        f"triggered_rules 가 목록이 아님: {raw_trig!r} — "
                        "근거 목록을 읽지 못했으므로 라벨을 검사할 수 없다", eid)
            trig = []

        design = row.get("condition_design")
        if isinstance(design, str) and design == "grid":
            grid_rows += 1
        else:
            dist[outcome] += 1

        if outcome == "nominal" and trig:
            rep.add("C4", ERROR, f"outcome=nominal 인데 triggered_rules 존재: {trig[:3]}", eid)
        if outcome in ("near_miss", "fail") and not trig:
            rep.add("C4", ERROR, f"outcome={outcome} 인데 판정 근거 없음", eid)

        if known:
            # 문자열이 아닌 원소가 섞이면 그 자체가 결함이다. 예외로 검증기를 죽이지 않고
            # 문자열로 정규화해 검사하면 finding 으로 보고된다.
            unknown = [str(r) for r in trig
                       if str(r) not in known and not _wildcard_hit(str(r), known)]
            if unknown:
                rep.add("C4", ERROR, f"rules.json 에 없는 rule_id: {unknown[:3]}", eid)
            nonstr = [r for r in trig if not isinstance(r, str)]
            if nonstr:
                rep.add("C4", ERROR,
                        f"triggered_rules 에 문자열 아닌 원소: {[type(r).__name__ for r in nonstr[:3]]}",
                        eid)

    rep.stats["outcome_distribution"] = {str(k): int(v) for k, v in dist.items()}
    rep.stats["grid_episodes_not_counted"] = grid_rows

    total = sum(dist.values())
    if total:
        counted = f"센 에피소드 {total}건 · 세지 않은 격자 행 {grid_rows}건" if grid_rows else ""
        fail = dist.get("fail", 0) / total
        near = dist.get("near_miss", 0) / total
        if fail < OUTCOME_FLOOR["fail"]:
            rep.add("C4", WARN,
                    f"fail 비율 {fail:.1%} — 경고 하한 {OUTCOME_FLOOR['fail']:.0%}. "
                    "위험 데이터 부족은 데이터셋 목적을 훼손함", counted)
        if near < OUTCOME_FLOOR["near_miss"]:
            rep.add("C4", WARN,
                    f"near_miss 비율 {near:.1%} — 경고 하한 {OUTCOME_FLOOR['near_miss']:.0%}. "
                    "경계 판별 학습 근거 부족", counted)


def _wildcard_hit(rule_id: str, known: set[str]) -> bool:
    """R-TS15066-A2-* 형태의 와일드카드 선언을 허용."""
    for k in known:
        if k.endswith("*") and rule_id.startswith(k[:-1]):
            return True
    return False


def termination_problems(obj: dict) -> list[str]:
    """`{terminated, truncated, reason[]}` 계약 위반을 사람 말로 돌려준다.

    규격서 §5 가 세 멤버를 요구한다. 그릇만 보면 빈 객체가 통과하고, 그러면
    「왜 끝났는가」가 비어 있는 채로 배포된다.
    """
    out = []
    for name in ("terminated", "truncated"):
        if name not in obj:
            out.append(f"{name} 없음")
        elif not isinstance(obj[name], bool):
            out.append(f"{name} 가 참·거짓이 아님: {obj[name]!r}")
    reason = obj.get("reason")
    if reason is None:
        out.append("reason 없음")
    elif not isinstance(reason, list) or not all(isinstance(x, str) for x in reason):
        out.append(f"reason 이 문자열 목록이 아님: {reason!r}")
    elif not reason:
        out.append("reason 이 비어 있음 — 끝난 이유를 적어야 한다")
    return out


def generator_provenance_problems(backend: dict) -> list[str]:
    """Why a sim_backend does not say which committed code made the episode; [] when it does.

    A distribution is made from committed code only: the converter refuses a recording from a
    dirty tree, and this is the same rule on what was shipped. Types, not truthiness --
    "false" and 0 are not False."""
    out = []
    commit = backend.get("generator_commit")
    if not (isinstance(commit, str) and GENERATOR_COMMIT_RE.fullmatch(commit)):
        out.append(f"sim_backend.generator_commit 이 40자 16진 커밋 해시가 아님: {commit!r:.50} — "
                   "어느 코드가 만든 데이터인지 알 수 없다")
    dirty = backend.get("generator_dirty")
    if dirty is not False:
        out.append(f"sim_backend.generator_dirty 가 false 가 아님: {dirty!r:.30} — "
                   "배포본은 커밋된 코드로 만든 것만 싣는다")
    return out


def c5_repro_meta(ds: Dataset, eps: pd.DataFrame | None, rep: Report) -> None:
    """C5 — sim 데이터의 재현 메타(seed, backend) 및 enum 검사."""
    if eps is None:
        rep.add("C5", SKIP, "에피소드 메타 없음")
        return

    carried_ssm = carried_ssm_values(ds.rules())
    carried_directed = carried_ssm_values(ds.rules(), DIRECTED_SSM_RULE_ID)
    shipped_table = directed_rule(ds.rules()).get("stop_table")
    directed_table = shipped_table if isinstance(shipped_table, dict) else None
    scenario_by_id = scenario_index(ds) or {}

    for f in REQUIRED_EPISODE_FIELDS:
        if f not in eps.columns:
            rep.add("C5", ERROR, f"필수 에피소드 필드 없음: {f}", "meta/episodes")
            continue
        # 컬럼이 있는 것과 값이 있는 것은 다르다. null 은 enum 검사도(`pd.notna`
        # 로 감싸여 있다) 조건부 필수 검사도 그냥 지나가므로, 필수 필드가 비어
        # 있는 채로 통과한다.
        blank = int(eps[f].isna().sum())
        if blank:
            rep.add("C5", ERROR,
                    f"필수 에피소드 필드 {f} 가 {blank}건에서 비어 있음", "meta/episodes")

    # 「비어 있지 않다」와 「그 형태다」는 다르다. termination="not-json" 도
    # degenerate=17 도 위 검사를 그냥 지났다. 정본
    # 목록을 순회하므로 JSON 필드를 늘려도 함께 검사된다.
    # 정본은 배포본에 동봉된 규격 정의다. 검증기가 사본을 들면 또 하나 어긋날
    # 자리가 생긴다 — C1 이 이미 그 목록의 이름이 실재하는지 대조한다.
    canon = ds.safety_schema() or {}
    declared = json_member(json_member(canon, "json_encoded_fields", dict), "episode", list)
    json_fields = [f for f in declared if isinstance(f, str) and f in eps.columns]
    shapes = json_member(canon, "episode_json_shapes", dict)
    members = json_member(canon, "notes_members", dict)
    for _, row in eps.iterrows():
        eid = str(row.get("episode_id", "?"))
        for f in json_fields:
            v = row.get(f)
            if scalar_missing(v):
                continue
            try:
                parsed = json.loads(v) if isinstance(v, str) else v
            except (TypeError, ValueError, RecursionError):
                rep.add("C5", ERROR, f"{f} 가 JSON 으로 읽히지 않음: {v!r:.60}", eid)
                continue
            # 필드마다 그릇이 정해져 있다. 「객체 또는 배열」로 뭉뚱그리면
            # termination="[]" 이 통과하고, 그러면 종료 사유가 통째로 비어 있는
            # 채로 배포된다.
            want = shapes.get(f)
            if want and not json_type_ok(parsed, want):
                rep.add("C5", ERROR,
                        f"{f} 는 {want} 여야 하는데 {type(parsed).__name__} 임", eid)
                continue
            if not want and not isinstance(parsed, (dict, list)):
                rep.add("C5", ERROR,
                        f"{f} 는 객체나 배열이어야 하는데 {type(parsed).__name__} 임", eid)
                continue
            # 바깥 그릇만 보면 `{}` 도 `{"terminated": "yes"}` 도 통과한다. 종료
            # 사유는 「이 에피소드가 왜 끝났는가」를 말하는 유일한 필드다.
            if f == "termination":
                for problem in termination_problems(parsed):
                    rep.add("C5", ERROR, f"termination: {problem}", eid)
            # notes 는 결과 값을 과대해석하지 않게 막는 단서 묶음이다. 하나를
            # 빼도 나머지 때문에 「비어 있지 않다」가 성립해 통과했다.
            if f == "notes":
                for name, spec in sorted(members.items()):
                    if name not in parsed:
                        rep.add("C5", ERROR, f"notes.{name} 없음", eid)
                    elif not json_type_ok(parsed[name], spec):
                        rep.add("C5", ERROR,
                                f"notes.{name} 이 {spec} 이 아님: {parsed[name]!r:.40}", eid)
                # The episode's steps are its rows: length, which C7 holds to the rows its pointers
                # name. duration_s x fps is held to nothing, so it bounds only an episode without one.
                frames = numeric_or_none(row.get("length"))
                if frames is not None and float(frames).is_integer():
                    steps = int(frames)
                else:
                    duration, fps = row.get("duration_s"), row.get("fps")
                    steps = (int(round(float(duration) * float(fps)))
                             if numeric_or_none(duration) and numeric_or_none(fps) else None)
                if declares_directed_ssm(scenario_by_id.get(str(row.get("scenario_id")))):
                    problems = notes_problems(parsed, BODY_REGIONS, steps, carried_directed,
                                              ssm_names=DIRECTED_SSM_IN_NOTES, stop_table=directed_table,
                                              completion_required=True)
                else:
                    problems = notes_problems(parsed, BODY_REGIONS, steps, carried_ssm)
                for problem in problems:
                    rep.add("C5", ERROR, problem, eid)
        if "degenerate" in eps.columns:
            v = row.get("degenerate")
            if not scalar_missing(v) and not isinstance(v, (bool, np.bool_)):
                rep.add("C5", ERROR, f"degenerate 가 참·거짓이 아님: {v!r}", eid)

    for _, row in eps.iterrows():
        eid = str(row.get("episode_id", "?"))

        for fname, allowed in ENUMS.items():
            if fname in eps.columns:
                v = row.get(fname)
                if pd.notna(v) and str(v) not in allowed:
                    rep.add("C5", ERROR, f"{fname} 허용값 아님: {v!r}", eid)

        if str(row.get("schema_version", "")) != SCHEMA_VERSION:
            rep.add("C5", WARN,
                    f"schema_version={row.get('schema_version')!r} (기준 {SCHEMA_VERSION})", eid)

        source = str(row.get("source"))
        for field in CONDITIONAL_REQUIRED_EPISODE_FIELDS.get(source, ()):
            value = as_obj(row.get(field)) if field == "sim_backend" else row.get(field)
            missing = (not isinstance(value, dict) or not value) if field == "sim_backend" \
                else pd.isna(value)
            if missing:
                rep.add("C5", ERROR, f"source={source} 인데 {field} 없음", eid)
        if source == "sim":
            backend = as_obj(row.get("sim_backend"))
            if isinstance(backend, dict) and backend:
                for k in ("isaacsim", "isaaclab", "physics"):
                    if k not in backend:
                        rep.add("C5", WARN, f"sim_backend 에 {k} 버전 없음 — 재현성 저하", eid)
                for problem in generator_provenance_problems(backend):
                    rep.add("C5", ERROR, problem, eid)

        mm = str(row.get("measurement_method", ""))
        if mm in FORBIDDEN_MEASUREMENT_METHODS.get(source, ()):
            rep.add("C5", ERROR,
                    f"source={source} 와 모순되는 measurement_method={mm}", eid)

        fps = row.get("fps")
        if not scalar_missing(fps):
            n = numeric_or_none(fps)
            if n is None:
                # 문자열이나 불리언은 float() 에서 죽거나(exit 3) 1.0 으로 통과한다.
                rep.add("C5", ERROR, f"fps 가 수치가 아님: {fps!r}", eid)
            elif not (SANITY["fps_min"] <= n <= SANITY["fps_max"]):
                rep.add("C5", ERROR, f"fps 이상: {fps}", eid)
        dur = row.get("duration_s")
        if not scalar_missing(dur):
            n = numeric_or_none(dur)
            if n is None:
                rep.add("C5", ERROR, f"duration_s 가 수치가 아님: {dur!r}", eid)
            elif n < SANITY["duration_min_s"]:
                rep.add("C5", WARN, f"duration_s 과소: {dur}", eid)

    if "episode_id" in eps.columns:
        dup = [k for k, v in Counter(eps["episode_id"].astype(str)).items() if v > 1]
        if dup:
            rep.add("C5", ERROR, f"episode_id 중복: {dup[:5]}", "meta/episodes")

        for _, row in eps.iterrows():
            eid = str(row.get("episode_id", ""))
            m = EPISODE_ID_RE.match(eid)
            if m is None:
                rep.add("C5", ERROR,
                        "episode_id 형식이 규격서 §3 과 다름 "
                        "({scenario_id}_{episode_seq:04d}_seed{seed})", eid)
                continue
            sid = row.get("scenario_id")
            if pd.notna(sid) and str(sid) != m.group("scenario"):
                rep.add("C5", ERROR,
                        f"episode_id 의 시나리오 {m.group('scenario')} 가 "
                        f"scenario_id={sid} 와 다름", eid)
            seed = row.get("randomization_seed")
            if pd.notna(seed) and int(seed) != int_of_digits(m.group("seed")):
                rep.add("C5", ERROR,
                        f"episode_id 의 시드 {m.group('seed')} 가 "
                        f"randomization_seed={int(seed)} 와 다름 "
                        "— 파일명으로 시드를 알 수 있다는 규격서 §3 의 주장이 깨진다", eid)
            # The run that made the episode began at run_start_seq (§3): the rerun that
            # reproduces it starts there, so a start past the episode's own number names a run
            # that could not have made it. A missing value is the conditional check's to report.
            start = row.get("run_start_seq")
            if not scalar_missing(start):
                n = numeric_or_none(start)
                seq = int_of_digits(m.group("seq"))       # None: a number no start can pass
                if n is None or n != int(n):
                    rep.add("C5", ERROR, f"run_start_seq 가 정수가 아님: {start!r}", eid)
                elif not 0 <= n or (seq is not None and n > seq):
                    rep.add("C5", ERROR,
                            f"run_start_seq={int(n)} 가 0 과 자기 번호 {seq} 사이가 아님 — 실행의 첫 "
                            "번호는 그 실행이 만든 에피소드의 번호보다 클 수 없다 (규격서 §3)", eid)


def _grid_cells(entry: dict) -> int | None:
    """The cells along each axis of a table scenario's grid (its conditions.grid.cells), or None when the entry has
    no grid block or the count is not a whole number above 0."""
    grid = entry[TABLE_CONDITION].get("grid")
    cells = grid.get("cells") if isinstance(grid, dict) else None
    return cells if isinstance(cells, int) and not isinstance(cells, bool) and cells > 0 else None


def _is_cell(value) -> bool:
    """A grid cell is a whole number, judged by value: an integer, or a float that is integral. A release that mixes a
    table scenario with one without writes the cells as floats -- the converter builds the episode table from one
    record per episode, the other scenario's records lack the cell keys, and pandas widens the column to hold their
    NaN -- so -1.0 is the cell -1. A bool, text, a non-integral float, NaN and None are not cells."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    return isinstance(value, (float, np.floating)) and math.isfinite(value) and float(value).is_integer()


def _c5_grid_cells(rep: Report, eid: str, sid: str, entry: dict, design, robot, human) -> bool:
    """A table episode's two cells: whole numbers (_is_cell), -1 and -1 on a random row, on the scenario's grid on a
    grid row. A design that is neither is the ENUMS check's to say, and its cells are not judged. True when a defect
    of the cells was said."""
    if not (_is_cell(robot) and _is_cell(human)):
        rep.add("C5", ERROR, "grid_cell_robot · grid_cell_human 가 정수가 아님", f"{eid} · {robot!r}, {human!r}")
        return True
    robot, human = int(robot), int(human)
    if isinstance(design, str) and design == "random":
        if (robot, human) != (GRID_NOT_A_CELL, GRID_NOT_A_CELL):
            rep.add("C5", ERROR,
                    f"무작위 행(condition_design=random)인데 격자 칸이 {GRID_NOT_A_CELL} · {GRID_NOT_A_CELL} 이 아님",
                    f"{eid} · grid_cell_robot {robot}, grid_cell_human {human}")
            return True
    elif isinstance(design, str) and design == "grid":
        cells = _grid_cells(entry)
        if cells is None:
            rep.add("C5", ERROR,
                    "격자 행(condition_design=grid)인데 시나리오 정본에 격자 블록(conditions.grid 의 cells — "
                    "1 이상의 정수)이 없음 — 칸 번호를 대 볼 격자가 없다", f"{eid} · {sid}")
            return True
        if not (0 <= robot < cells and 0 <= human < cells):
            rep.add("C5", ERROR,
                    "격자 행(condition_design=grid)의 칸이 격자 밖 — 칸 번호는 0 부터 conditions.grid.cells − 1 까지",
                    f"{eid} · grid_cell_robot {robot}, grid_cell_human {human} (cells {cells})")
            return True
    return False


# The episode's notes are not a JSON object with an object under randomization: the JSON fields' check and the notes
# member check say so, and the record is not looked for.
_RECORD_UNREAD = object()


def _recorded_place(notes) -> object:
    """A table episode's place in its table as its condition record says it -- notes.randomization.condition, from
    which the converter derives the flat columns: (design, (robot cell, human cell) or None for a null grid_cell).
    What the record lacks, as a few words for the finding's location, when there is no condition object, no design
    string, or no grid_cell of null or two whole numbers (_is_cell); _RECORD_UNREAD when notes or its randomization
    is not an object."""
    if isinstance(notes, str):
        try:
            notes = json.loads(notes)
        except (ValueError, RecursionError):
            return _RECORD_UNREAD
    if not isinstance(notes, dict) or not isinstance(notes.get("randomization"), dict):
        return _RECORD_UNREAD
    randomization = notes["randomization"]
    if "condition" not in randomization:
        return "randomization.condition 없음"
    record = randomization["condition"]
    if not isinstance(record, dict):
        return f"randomization.condition = {record!r:.40}"
    if not isinstance(record.get("design"), str):
        return "design 없음" if "design" not in record else f"design = {record['design']!r:.40}"
    if "grid_cell" not in record:
        return "grid_cell 없음"
    cell = record["grid_cell"]
    if cell is None:
        return record["design"], None
    if isinstance(cell, list) and len(cell) == 2 and all(_is_cell(c) for c in cell):
        return record["design"], (int(cell[0]), int(cell[1]))
    return f"grid_cell = {cell!r:.40}"


def _c5_recorded_place(rep: Report, eid: str, notes, design, robot, human, cells_said: bool) -> None:
    """A table episode's flat columns against its condition record (_recorded_place): condition_design is the
    record's design, and the two cells are its grid_cell (-1 and -1 where it is null). The design is compared only
    when it is one of ENUMS (null and other values are said above), the cells only when the designs agree and the
    cells passed _c5_grid_cells (``cells_said`` false; ``robot`` / ``human`` None when a column is missing) -- a row
    whose design is not the record's has the record's cell for another kind of row, and one defect is one sentence."""
    place = _recorded_place(notes)
    if place is _RECORD_UNREAD:
        return
    if isinstance(place, str):
        rep.add("C5", ERROR,
                "조건 기록(notes.randomization.condition)의 design(문자열) · grid_cell(null 또는 정수 둘)이 없거나 "
                "꼴이 아님 — 평평한 열 condition_design · grid_cell_robot · grid_cell_human 을 대 볼 기록이 없다",
                f"{eid} · {place}")
        return
    recorded_design, recorded_cell = place
    if not (isinstance(design, str) and design in ENUMS["condition_design"]):
        return
    if design != recorded_design:
        rep.add("C5", ERROR,
                "condition_design 이 조건 기록(notes.randomization.condition)의 design 과 다름 — 같은 에피소드를 "
                "열과 기록이 다른 종류의 행으로 말한다",
                f"{eid} · 열 {design!r}, 기록 {recorded_design!r:.40}")
        return
    if cells_said or robot is None or human is None:
        return
    want = recorded_cell if recorded_cell is not None else (GRID_NOT_A_CELL, GRID_NOT_A_CELL)
    if (int(robot), int(human)) != want:
        rep.add("C5", ERROR,
                "grid_cell_robot · grid_cell_human 이 조건 기록(notes.randomization.condition)의 grid_cell 과 다름 — "
                f"기록이 null 이면 {GRID_NOT_A_CELL} · {GRID_NOT_A_CELL}",
                f"{eid} · 열 {int(robot)}, {int(human)}, 기록 "
                + ("null" if recorded_cell is None else f"[{recorded_cell[0]}, {recorded_cell[1]}]"))


def c5_table_fields(ds: Dataset, eps: pd.DataFrame | None, rep: Report) -> None:
    """C5 -- the episode columns of a scenario that declares a conditions table (spec §5, the shipped schema's
    scenario_conditional_episode_fields), held to their shape:

    - every column is in the table: one ERROR per column missing, the number of episodes in its sentence;
    - episode_spec is not null and its members are EPISODE_SPEC_MEMBERS (that it reads as a JSON object is the
      JSON fields' check in c5_repro_meta, which says so of a cell this one passes over);
    - condition_design is not null (its value is the ENUMS check's); a cell is a whole number, judged by value (an
      integral float is one -- a release that mixes table scenarios with others holds the cells as floats); a random
      row's two cells are GRID_NOT_A_CELL and a grid row's lie in 0 .. conditions.grid.cells - 1 of its scenario (a
      grid row of a scenario without a grid says that);
    - the flat columns say what the episode's condition record says (notes.randomization.condition, from which the
      converter derives them -- _c5_recorded_place): condition_design is its design and the two cells its grid_cell
      (GRID_NOT_A_CELL twice where it is null); a record without a design or a grid_cell of null or two whole numbers
      is said, a notes or randomization that is not an object is the JSON and member checks' to say. C4 tells the
      grid rows by the column and the scorecard by the record, so the two must not part;
    - an evidence value is a number (inf included) or null, and EVIDENCE_NEVER_NULL is not null.

    One sentence per kind of defect, the episode and what was found in its location. An episode is of a table
    scenario when its scenario's entry in the shipped scenarios.json declares one (declares_conditions) -- an entry
    whose conditions is there but not an object is said once, its episodes then not looked at; the rows of other
    scenarios, whose table columns are empty in a mixed release, are not looked at. With no scenario canon it can trust
    (scenario_index is None) the check says nothing -- C1 says what is wrong with the canon. Values are read by column
    (Series.tolist), so each keeps the type its column was written with.

    Not looked at: whether a value is the one the episode's steps give (generation/labeling/check_release_labels.py
    derives the evidence again), what the spec's members hold, whether a grid row's cell is the one its row names."""
    if eps is None or "scenario_id" not in eps.columns:
        return
    by_id = scenario_index(ds)
    if by_id is None:
        return
    sids = [str(s) for s in eps["scenario_id"].tolist()]
    # declares_conditions takes only an object as a table: an entry whose conditions is something else is said once
    # here, since its episodes' table columns go unlooked at.
    for sid in sorted(set(sids)):
        entry = by_id.get(sid)
        if isinstance(entry, dict) and TABLE_CONDITION in entry and not declares_conditions(entry):
            rep.add("C5", ERROR,
                    f"시나리오 {sid} 의 conditions 가 객체가 아님 — 조건표가 있는 시나리오인지 알 수 없어 그 에피소드의 "
                    "조건표 열을 보지 않음", "meta/scenarios.json")
    rows = [i for i, sid in enumerate(sids) if declares_conditions(by_id.get(sid))]
    rep.stats["table_episodes"] = len(rows)
    if not rows:
        return

    names = SCENARIO_CONDITIONAL_EPISODE_FIELDS[TABLE_CONDITION]
    present = {f: eps[f].tolist() for f in names if f in eps.columns}
    for f in names:
        if f not in present:
            rep.add("C5", ERROR,
                    f"조건표가 있는 시나리오의 에피소드 열 없음: {f} — 조건표가 있는 시나리오의 에피소드 {len(rows)}건",
                    "meta/episodes")
    ids = eps["episode_id"].tolist() if "episode_id" in eps.columns else ["?"] * len(eps)
    notes = eps["notes"].tolist() if "notes" in eps.columns else None
    members = set(EPISODE_SPEC_MEMBERS)

    for i in rows:
        eid = str(ids[i])
        if "episode_spec" in present:
            spec = present["episode_spec"][i]
            if scalar_missing(spec):
                rep.add("C5", ERROR,
                        "episode_spec 가 비어 있음 — 이 에피소드가 어떤 조건에서 기록되었는지 받는 쪽이 알 수 없다", eid)
            else:
                try:
                    parsed = json.loads(spec) if isinstance(spec, str) else spec
                except (TypeError, ValueError, RecursionError):
                    parsed = None
                if isinstance(parsed, dict) and set(parsed) != members:
                    missing = [m for m in EPISODE_SPEC_MEMBERS if m not in parsed]
                    extra = sorted(str(k) for k in parsed if k not in members)[:3]
                    rep.add("C5", ERROR,
                            f"episode_spec 의 멤버가 규격의 {len(EPISODE_SPEC_MEMBERS)}개"
                            f"({' · '.join(EPISODE_SPEC_MEMBERS)})와 다름",
                            f"{eid} · 없음 {missing} · 잉여 {extra}")

        design = present["condition_design"][i] if "condition_design" in present else None
        if "condition_design" in present and scalar_missing(design):
            rep.add("C5", ERROR, "condition_design 가 비어 있음 — 무작위 행인지 격자 행인지 알 수 없다", eid)
        robot = human = None
        cells_said = False
        if "grid_cell_robot" in present and "grid_cell_human" in present:
            robot, human = present["grid_cell_robot"][i], present["grid_cell_human"][i]
            cells_said = _c5_grid_cells(rep, eid, sids[i], by_id[sids[i]], design, robot, human)
        if notes is not None and not scalar_missing(notes[i]):
            _c5_recorded_place(rep, eid, notes[i], design, robot, human, cells_said)

        for f in names:
            if not f.startswith(EVIDENCE_PREFIX) or f not in present:
                continue
            value = present[f][i]
            if scalar_missing(value):
                if f in EVIDENCE_NEVER_NULL:
                    rep.add("C5", ERROR,
                            f"{f} 가 비어 있음 — 모든 스텝에 사람과의 거리와 요구 이격이 있으므로 가장 작은 여유는 늘 있다",
                            eid)
            elif isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                rep.add("C5", ERROR, "근거 요약 열(evidence.*)의 값이 수나 null 이 아님", f"{eid} · {f} = {value!r:.40}")


def c6_asset_license(ds: Dataset, eps: pd.DataFrame | None, rep: Report) -> None:
    """C6 — scene_assets 라이선스 기재 및 NOTICE 등재."""
    if eps is None:
        rep.add("C6", SKIP, "에피소드 메타 없음")
        return

    notice = ds.notice_text()
    if notice is None:
        rep.add("C6", ERROR, "NOTICE 파일 없음 — BSD-3 등 고지 의무 미충족", "NOTICE")
        notice = ""

    redistributed: set[str] = set()
    # Every path some episode's scene_assets names, whatever else is wrong with the item: the
    # batch's use of an asset is what makes the canon's "redistributed" a demand on assets/.
    referenced: set[str] = set()
    licenses: Counter = Counter()
    per_episode_assets: list[tuple[str, str, bool, str]] = []
    # (episode label, its row, the paths its scene_assets names) -- the anchor's input.
    listed: list[tuple[str, pd.Series, set[str]]] = []

    for _, row in eps.iterrows():
        eid = str(row.get("episode_id", "?"))
        assets = as_obj(row.get("scene_assets"))
        if not isinstance(assets, list) or not assets:
            rep.add("C6", ERROR, "scene_assets 없음 또는 비어 있음", eid)
            continue
        paths_here: set[str] = set()
        listed.append((eid, row, paths_here))
        for a in assets:
            if not isinstance(a, dict):
                rep.add("C6", ERROR, f"scene_assets 항목 형식 오류: {a!r}", eid)
                continue
            path = str(a.get("path", "?"))
            referenced.add(path)
            paths_here.add(path)
            lic = a.get("license")
            if not lic:
                rep.add("C6", ERROR, f"자산 라이선스 미기재: {path}", eid)
                continue
            licenses[str(lic)] += 1
            # `bool("false")` 는 True 이고 `bool(0)` 은 False 다. 강제 변환하면
            # 재배포 여부가 조용히 뒤집히는데, 이 값이 NOTICE 고지 의무를 가른다.
            flag = a.get("redistributed")
            if not isinstance(flag, bool):
                rep.add("C6", ERROR,
                        f"redistributed 가 참·거짓이 아님: {flag!r} ({path}) — "
                        "고지 의무를 가르는 값이므로 추측하지 않는다", eid)
                continue
            if flag:
                redistributed.add(path)
            per_episode_assets.append((path, str(lic), flag, eid))

    for path in sorted(redistributed):
        key = Path(path).name
        if key and key not in notice:
            rep.add("C6", ERROR, f"재배포 자산이 NOTICE 에 없음: {path}", "NOTICE")

    # 「재배포한다」고 선언했으면 파일이 실제로 있어야 한다. 없어도 C6 도 preflight 도
    # 통과했다 — 자산을 싣는다고 말하면서 싣지 않은 배포본이 올라간다.
    # 변환기는 재배포 자산을 배포본의 `assets/<basename>` 한 곳에 놓는다
    # (convert_lerobot.assemble_dataset). 아무 데나 같은 이름이 있으면 통과하게
    # 두면 다른 파일을 그 이름으로 복사해 놓기만 해도 넘어간다 — 실제로 README 를
    # `meta/dummy_ts15066_v1.usd` 로 복사해 우회했다.
    # 에피소드가 아니라 **동봉된 자산 정본**도 함께 본다. 에피소드의 플래그만 보면
    # 전부 false 로 바꾸고 파일을 지우는 것으로 이 검사를 통째로 끌 수 있다.
    # 두 선언이 어긋나는 것 자체도 오류다.
    canon_assets = ds.assets() or {}
    required = _c6_required_assets(ds, canon_assets, listed, rep)
    canon_redistributed = {entry["usd_path"]: name for name, entry in asset_entries(canon_assets)
                           if entry.get("redistributed")}
    # 양방향이어야 한다. 한쪽만 보면 반대쪽을 false 로 바꿔 검사를 끌 수 있다.
    # **이 배치의 에피소드가 쓰는 자산만** 요구한다. 정본은 과제가 만든 자산을
    # 전부 적고 배치는 그중 일부를 쓴다 — 전부를 요구하면 S1d 의 씬·입힌 더미를 재배포로
    # 적을 수 없다(v0.1 배포본이 그 파일을 요구받는다). 쓰는 에피소드가 있는데 그중 아무도
    # 재배포라고 말하지 않으면 여전히 어긋남이다 — 아래 에피소드별 대조가 행마다도 잡는다.
    for usd_path in sorted(canon_redistributed):
        if usd_path in referenced and usd_path not in redistributed:
            rep.add("C6", ERROR,
                    f"meta/assets.json 은 {usd_path} 를 재배포한다고 선언하는데 "
                    "그것을 쓰는 에피소드 중 어느 것도 그렇게 말하지 않음 — 두 선언이 어긋난다",
                    "meta/assets.json")
    # 집합으로만 대조하면 한 행만 뒤집혀도 다른 행의 true 가 집합을 채워 통과한다.
    # 라이선스는 아예 대조하지 않고 있었다.
    canon_all = {entry["usd_path"]: bool(entry.get("redistributed")) for _, entry in asset_entries(canon_assets)}
    canon_license = {entry["usd_path"]: str(entry.get("license") or "") for _, entry in asset_entries(canon_assets)}
    for path, lic, flag, eid in per_episode_assets:
        want_lic = canon_license.get(path)
        if want_lic is not None and lic != want_lic:
            rep.add("C6", ERROR,
                    f"{path} 의 라이선스가 정본과 다름 — 에피소드 {lic!r} vs 정본 "
                    f"{want_lic!r}", eid)
        want_flag = canon_all.get(path)
        if want_flag is not None and flag != want_flag:
            rep.add("C6", ERROR,
                    f"{path} 의 재배포 여부가 정본과 다름 — 에피소드 {flag} vs 정본 "
                    f"{want_flag}", eid)

    # 재배포하지 않는 자산도 정본에 있어야 한다. `redistributed=false` 만 보고 있어서
    # 정본에 없는 제3자 자산을 「참조만 한다」며 실을 수 있었다.
    for path, _lic, _flag, eid in per_episode_assets:
        if path not in canon_all:
            rep.add("C6", ERROR,
                    f"에피소드가 쓰는 자산 {path} 가 meta/assets.json 에 없음 — "
                    "라이선스 판정의 근거가 없다", eid)
    for usd_path in sorted(redistributed):
        if usd_path not in canon_all:
            pass  # 위에서 이미 에피소드별로 보고했다
        elif not canon_all[usd_path]:
            rep.add("C6", ERROR,
                    f"에피소드는 {usd_path} 를 재배포한다고 말하는데 meta/assets.json 은 "
                    "아니라고 말함 — 두 선언이 어긋난다", "meta/assets.json")
    # The batch uses what its episodes list and what the canon says they used (the anchor
    # above): an asset left out of scene_assets is still asked for, so its absence from
    # assets/ is reported too rather than excused. An entry that composes others is asked for
    # only with everything it composes, whoever references it (_composed_shipped).
    composed_paths = {entry["usd_path"] for _, entry in asset_entries(canon_assets) if COMPOSES_KEY in entry}
    shipped = (set(redistributed) | (set(canon_redistributed) & (referenced | required))) - composed_paths
    shipped_keys = {name for name, entry in asset_entries(canon_assets)
                    if entry["usd_path"] in shipped and entry.get("redistributed") is True}
    stages, problems = _composed_shipped(canon_assets, shipped_keys)
    for problem in problems:
        rep.add("C6", ERROR, f"엮은 자산을 읽지 못함: {problem}", "meta/assets.json")
    by_name = dict(asset_entries(canon_assets))
    for name in sorted(stages):
        path = by_name[name]["usd_path"]
        shipped.add(path)
        if Path(path).name not in notice:
            rep.add("C6", ERROR, f"재배포 자산이 NOTICE 에 없음: {path}", "NOTICE")
    pins = {entry["usd_path"]: entry.get("usd_sha256") for _, entry in asset_entries(canon_assets)}
    # A robot with a ship block ships as its own folder: its rows, not its USD's name under assets/.
    ships = {entry["usd_path"]: (name, entry) for name, entry in asset_entries(canon_assets) if "ship" in entry}
    usds = 0
    robot_files: set[str] = set()
    robot_hashed = 0
    for path in sorted(shipped):
        if path in ships:
            accounted, hashed = _c6_robot_folder(ds, *ships[path], rep)
            robot_files |= accounted
            robot_hashed += hashed
            continue
        name = Path(path).name
        if not name:
            continue
        here = ds.root / ASSETS_DIR / name
        if not here.is_file():
            rep.add("C6", ERROR,
                    f"재배포한다고 선언한 자산이 배포본에 없음: assets/{name} — "
                    "메타데이터가 싣는다고 말하는 것을 싣지 않았다", "assets/")
        elif here.stat().st_size == 0:
            rep.add("C6", ERROR, f"재배포 자산이 비어 있음: assets/{name}", "assets/")
        elif path in pins:                      # a path the canon lacks is reported above
            # The bytes, not only the name: the shipped canon pins every redistributable USD
            # (usd_sha256), and the converter copies only that. A distribution whose USD is
            # another file under the right name passed C6 on existence and size alone.
            pin = pins[path]
            if not isinstance(pin, str) or not re.fullmatch(r"[0-9a-f]{64}", pin):
                rep.add("C6", ERROR,
                        f"재배포 자산 {path} 의 usd_sha256 이 meta/assets.json 에 없거나 64자리 소문자 "
                        f"16진수가 아님: {pin!r} — 실린 assets/{name} 이 어느 파일인지 대조할 수 없다",
                        "meta/assets.json")
            else:
                usds += 1
                if _sha256(here) != pin:
                    rep.add("C6", ERROR,
                            f"assets/{name} 의 SHA-256 이 meta/assets.json 의 usd_sha256 과 다름 — "
                            "정본이 고정한 자산이 아니다", "assets/")
    textures = _c6_shipped_files(ds, canon_assets, shipped - set(ships), rep, robot_files)

    rep.stats["asset_licenses"] = dict(licenses)
    rep.stats["redistributed_assets"] = sorted(redistributed)
    rep.stats["shipped_textures"] = textures
    rep.stats["shipped_usds"] = usds
    rep.stats["shipped_robot_files"] = robot_hashed


def _composed_shipped(canon_assets: dict, shipped_keys: set[str]) -> tuple[set[str], list[str]]:
    """The keys of the shipped canon's entries that compose others (``composes``) and ship with this release
    -- redistributable, every key they compose one of ``shipped_keys`` (the scene and robot keys whose files
    the release carries) -- and what keeps a ``composes`` from saying what it composes. The converter's
    generation.assets.composed_shipped, restated (this file does not import generation);
    tests/validator/test_asset_checks.py holds the two together. A composed entry is a file like any other, so
    one whose usd_path is not a non-empty string is a problem too (asset_entries leaves it out, and nothing
    else could say where it ships). Any problem leaves the set empty: the converter ships nothing it cannot
    read."""
    by_key: dict[str, list[Any]] = {}
    for group in ("robots", "scene"):
        for name, entry in json_member(canon_assets, group, dict).items():
            by_key.setdefault(name, []).append(entry)
    composed = {name for name, entries in by_key.items()
                if any(isinstance(e, dict) and COMPOSES_KEY in e for e in entries)}
    out: set[str] = set()
    problems: list[str] = []
    for name in sorted(composed):
        entries = by_key[name]
        if len(entries) != 1:
            problems.append(f"{name}: 키가 robots 와 scene 에 모두 있음")
            continue
        entry = entries[0]
        path = entry.get("usd_path")
        if not isinstance(path, str) or not path:
            problems.append(f"{name}: usd_path 가 비어 있지 않은 문자열이 아님: {path!r}")
        keys = entry[COMPOSES_KEY]
        if not isinstance(keys, list) or not keys or not all(isinstance(k, str) and k for k in keys):
            problems.append(f"{name}: composes 가 비어 있지 않은 키 문자열의 목록이 아님: {keys!r}")
            continue
        bad = []
        if len(set(keys)) != len(keys):
            bad.append(f"composes 에 같은 키가 두 번: {keys}")
        for key in keys:
            if key == name:
                bad.append("composes 가 자기 자신을 적음")
            elif key not in by_key:
                bad.append(f"composes 의 {key} 가 meta/assets.json 의 robots · scene 에 없음")
            elif len(by_key[key]) != 1:
                bad.append(f"composes 의 {key} 가 robots 와 scene 에 모두 있음")
            elif key in composed:
                bad.append(f"composes 의 {key} 도 엮은 자산 — 엮은 자산은 한 단계만 엮는다")
        bad += [f"엮은 자산에 {block} 가 있음 — 엮은 자산은 USD 하나로 실린다"
                for block in ("bundle", "ship") if block in entry]
        if bad:
            problems += [f"{name}: {b}" for b in bad]
        elif entry.get("redistributed") is True and set(keys) <= set(shipped_keys):
            out.add(name)
    return (set() if problems else out), problems


def _robot_ship_problems(key: str, entry: dict) -> list[str]:
    """What keeps an entry's ship block from saying which files its folder holds -- the converter's
    generation.assets.robot_ship_files, restated (this file does not import generation);
    tests/validator/test_asset_checks.py holds the two together. Where the fetched copy lies in the
    repository is the converter's to check: a distribution has no repository, and here the source only has
    to be named."""
    ship = entry.get("ship")
    if not isinstance(ship, dict):
        return [f"ship 이 객체가 아님: {ship!r}"]
    missing = [k for k in ROBOT_SHIP_KEYS if k not in ship]
    if missing:
        return [f"ship 에 {missing} 키가 없음"]
    out = []
    unknown = sorted(set(ship) - set(ROBOT_SHIP_KEYS))
    if unknown:
        out.append(f"ship 에 모르는 키 {unknown}")
    if "bundle" in entry:
        out.append("ship 과 bundle 을 함께 적음 — 폴더째 싣는 자산의 파일은 모두 ship.files 의 행이다")
    folder = ship["dir"]
    if not isinstance(folder, str) or not folder or Path(folder).name != folder \
            or folder in (".", "..", TEXTURE_SUBDIR) or "\\" in folder:
        out.append(f"ship.dir 이 assets/ 아래의 폴더 이름이 아님: {folder!r}")
    if not isinstance(ship["source_dir"], str) or not ship["source_dir"]:
        out.append(f"ship.source_dir 가 비어 있지 않은 문자열이 아님: {ship['source_dir']!r}")
    rows = ship["files"]
    if not isinstance(rows, list) or not rows:
        return out + [f"ship.files 가 비어 있지 않은 목록이 아님: {rows!r}"]
    by_file: dict[str, dict] = {}
    for i, row in enumerate(rows):
        bad = _texture_row_problems(row)
        if bad:
            out.append(f"ship.files[{i}] 가 출처 기록이 아님: {'; '.join(bad)}")
        elif row["file"] in by_file:
            out.append(f"ship.files 에 같은 이름이 둘: {row['file']}")
        else:
            by_file[row["file"]] = row
    pinned = entry.get("usd_sha256")
    if not isinstance(pinned, str) or not re.fullmatch(r"[0-9a-f]{64}", pinned):
        out.append(f"usd_sha256 이 없거나 64자리 소문자 16진수가 아님: {pinned!r}")
        pinned = None
    usd_name = Path(str(entry.get("usd_path") or "")).name
    if usd_name not in by_file:
        out.append(f"ship.files 에 로봇의 USD {usd_name} 의 행이 없음")
    elif pinned is not None and by_file[usd_name]["sha256"] != pinned:
        out.append(f"ship.files 의 {usd_name} 행의 sha256 이 usd_sha256 과 다름 — 에피소드가 쓴 로봇은 "
                   "usd_sha256 이 고정한 파일이다")
    licence = ship["license_file"]
    if not isinstance(licence, str) or licence not in by_file:
        out.append(f"license_file {licence!r} 의 행이 ship.files 에 없음")
    return out


def _c6_robot_folder(ds: Dataset, key: str, entry: dict, rep: Report) -> tuple[set[str], int]:
    """A robot that ships ships as its own folder: every row of its ship block is a file at
    assets/<dir>/<file> that hashes to the row's sha256 -- the USD its episodes ran (that row is its
    usd_sha256), the layers that USD references beside it, its licence file. Returns the files it accounts
    for, as paths in the distribution, and how many it hashed."""
    problems = _robot_ship_problems(key, entry)
    for problem in problems:
        rep.add("C6", ERROR, f"재배포 로봇 {key} 의 ship 이 실을 폴더를 말하지 못함: {problem}", "meta/assets.json")
    if problems:
        return set(), 0
    ship = entry["ship"]
    accounted: set[str] = set()
    hashed = 0
    for row in ship["files"]:
        rel = f"{ASSETS_DIR}/{ship['dir']}/{row['file']}"
        accounted.add(rel)
        here = ds.root / rel
        if not here.is_file():
            what = "라이선스 파일" if row["file"] == ship["license_file"] else "파일"
            rep.add("C6", ERROR,
                    f"재배포 로봇 {key} 의 {what}이 배포본에 없음: {rel} — 로봇은 폴더째(USD · 그것이 참조하는 "
                    "레이어 · 라이선스 전문) 실린다", f"{ASSETS_DIR}/")
            continue
        hashed += 1
        if _sha256(here) != row["sha256"]:
            rep.add("C6", ERROR,
                    f"{rel} 의 SHA-256 이 meta/assets.json({key} ship.files)과 다름 — 정본이 고정한 파일이 아니다",
                    f"{ASSETS_DIR}/")
    return accounted, hashed


def _required_scene_assets(scenario: object, canon_assets: dict,
                           robot_model: object) -> tuple[set[str], list[tuple[str | None, str]]]:
    """The asset paths an episode of ``scenario`` run by ``robot_model`` must list in
    scene_assets -- the scenario's declared ``scene.assets``, the USD of its
    ``scene.human_model`` and the USD of the robot, a key of meta/assets.json ``robots`` named
    by the episode (no scenario names its robot) -- and what keeps the canon from saying so,
    each as (where, why): where is meta/scenarios.json for the scenario's part and None for
    the episode's robot_model. The converter's generation.assets.required_scene_assets,
    restated (this file does not import generation); tests/validator/test_asset_checks.py
    holds the two together."""
    sid = scenario.get("scenario_id", "?") if isinstance(scenario, dict) else "?"
    loc = "meta/scenarios.json"
    scene = scenario.get("scene") if isinstance(scenario, dict) else None
    if not isinstance(scene, dict):
        return set(), [(loc, f"{sid}: scene 이 없음")]
    declared = scene.get("assets")
    if not isinstance(declared, list) or not all(isinstance(p, str) and p for p in declared):
        return set(), [(loc, f"{sid}: scene.assets 가 비어 있지 않은 문자열의 목록이 아님: {declared!r}")]
    out = set(declared)
    model = scene.get("human_model")
    if model is not None:
        entry = json_member(canon_assets, "scene", dict).get(model) if isinstance(model, str) else None
        if not isinstance(entry, dict) or not isinstance(entry.get("usd_path"), str) \
                or not entry["usd_path"]:
            return set(), [(loc, f"{sid}: scene.human_model {model!r} 이 meta/assets.json 의 scene 에 없음")]
        out.add(entry["usd_path"])
    robot = json_member(canon_assets, "robots", dict).get(robot_model) if isinstance(robot_model, str) else None
    if not isinstance(robot, dict) or not isinstance(robot.get("usd_path"), str) \
            or not robot["usd_path"]:
        return set(), [(None, f"robot_model {robot_model!r} 이 meta/assets.json 의 robots 에 없음")]
    out.add(robot["usd_path"])
    return out, []


def _c6_required_assets(ds: Dataset, canon_assets: dict,
                        listed: list[tuple[str, pd.Series, set[str]]], rep: Report) -> set[str]:
    """Each episode's scene_assets names every asset the shipped canon says it used. Returns
    the union of those required paths.

    What ships is drawn from the episodes' scene_assets, and the checks below it ask only for
    what the batch uses. Drawn from the list alone, an episode that leaves an asset out takes
    it out of assets/ and of every demand -- measured: a v0.1 copy with
    the dummy dropped from every scene_assets and its file deleted got 0 C6 ERRORs, where a
    C6 that demands every redistributable canon asset got 2. The
    anchor is the shipped canon: the scenario's scene.assets and its human_model's USD
    (meta/scenarios.json · meta/assets.json), and the episode's own human_model too -- a
    field the producer copies from the scenario, so either one changed alone still asks --
    and the USD of the episode's robot_model (meta/assets.json robots). Without the robot
    in the anchor, H1 dropped from every scene_assets of two converted batches whose robot does not
    ship got 0 C6 ERRORs. Its absence cost no file there -- it cost the
    licence record; a robot that ships as its folder would lose the folder as well.
    """
    by_id = scenario_index(ds)
    if by_id is None:
        if listed:
            rep.add("C6", ERROR,
                    "meta/scenarios.json 에서 시나리오를 읽지 못해 에피소드가 써야 하는 씬 자산을 대조하지 "
                    "못함 (사유는 C1)", "meta/scenarios.json")
        return set()
    scene_canon = json_member(canon_assets, "scene", dict)
    required_all: set[str] = set()
    for eid, row, paths in listed:
        sid = row.get("scenario_id")
        entry = by_id.get(sid) if isinstance(sid, str) else None
        if entry is None:
            rep.add("C6", ERROR,
                    f"시나리오 {sid!r} 가 meta/scenarios.json 에 없어 이 에피소드가 써야 하는 씬 자산을 "
                    "대조하지 못함", eid)
            continue
        need, problems = _required_scene_assets(entry, canon_assets, row.get("robot_model"))
        for where, problem in problems:
            rep.add("C6", ERROR, f"{problem} — 에피소드가 써야 하는 자산을 알 수 없음", where or eid)
        if problems:
            continue
        model = row.get("human_model")
        if not scalar_missing(model):
            spec = scene_canon.get(model) if isinstance(model, str) else None
            if not isinstance(spec, dict) or not isinstance(spec.get("usd_path"), str) \
                    or not spec["usd_path"]:
                rep.add("C6", ERROR,
                        f"human_model {model!r} 이 meta/assets.json 의 scene 에 없어 인체 모델의 자산을 "
                        "대조하지 못함", eid)
            else:
                need.add(spec["usd_path"])
        required_all |= need
        missing = sorted(need - paths)
        if missing:
            rep.add("C6", ERROR,
                    f"scene_assets 에 {missing} 가 없음 — 시나리오 {sid} 의 scene.assets · 인체 모델과 이 "
                    "에피소드의 robot_model 이 이 에피소드가 쓴 자산이라고 말한다. 배포본에 싣는 자산과 자산 "
                    "라이선스 기록은 scene_assets 에서 정해지므로 빠진 자산은 둘 다에서 빠진다", eid)
    return required_all


def _c6_shipped_files(ds: Dataset, canon_assets: dict, shipped: set[str], rep: Report,
                      robot_files: set[str] = frozenset()) -> int:
    """Everything under assets/ is accounted for, and nothing a shipped USD needs is missing.

    A file directly under assets/ is a USD this batch ships (``shipped``: what its episodes
    reference and the canon lets travel, an entry that composes others when everything it composes
    ships, the robots that ship as their folder left out). A file
    under assets/textures/ is a row of the shipped meta/assets.json bundle.files of such a USD --
    the record that says where it came from, under which licence -- and hashes to that row's
    sha256. Every row of a shipped USD's bundle is there. A file in a robot's folder is one of
    ``robot_files`` (the rows _c6_robot_folder accounted for and hashed). Returns how many
    textures were hashed.

    What it does not see: whether bundle.files lists every texture the USD really references
    (that needs pxr; tests/generation/test_texture_refs.py asserts it on the built USDs, and
    the converter refuses rows that are not the textures.lock's); a USD's other external
    references (OmniPBR.mdl is the renderer's own library, not a file of the asset)."""
    names = {Path(p).name for p in shipped if Path(p).name}
    claims: dict[str, tuple[str, str]] = {}             # texture file -> (sha256, asset key)
    loc = "meta/assets.json"
    for key, entry in asset_entries(canon_assets):
        if entry["usd_path"] not in shipped:
            continue
        if "bundle" not in entry:
            continue
        bundle = entry["bundle"]
        rows = bundle.get("files") if isinstance(bundle, dict) else None
        if not isinstance(rows, list) or not rows:
            rep.add("C6", ERROR,
                    f"재배포 자산 {key} 은 텍스처 묶음(bundle)을 선언하는데 파일 목록(bundle.files)이 "
                    "없거나 비어 있음 — 싣는 텍스처의 출처·라이선스·해시를 대조할 수 없다", loc)
            continue
        for i, row in enumerate(rows):
            bad = _texture_row_problems(row)
            if bad:
                rep.add("C6", ERROR,
                        f"{key} 의 bundle.files[{i}] 가 출처 기록이 아님: {'; '.join(bad)}", loc)
                continue
            name, sha = row["file"], row["sha256"]
            seen = claims.get(name)
            if seen is not None and seen[0] != sha:
                rep.add("C6", ERROR,
                        f"텍스처 {name} 을 두 자산이 다른 바이트로 적음 — {seen[1]} {seen[0][:12]}… · "
                        f"{key} {sha[:12]}… — assets/textures/ 에는 한 이름에 한 파일만 실린다", loc)
                continue
            claims.setdefault(name, (sha, key))
    base = ds.root / ASSETS_DIR
    textures = base / TEXTURE_SUBDIR
    hashed = 0
    for name, (sha, key) in sorted(claims.items()):
        here = textures / name
        if not here.is_file():
            rep.add("C6", ERROR,
                    f"재배포 자산 {key} 이 쓰는 텍스처가 배포본에 없음: assets/textures/{name} — "
                    "USD 가 ./textures/ 로 참조하는 파일이다", "assets/textures/")
            continue
        hashed += 1
        if _sha256(here) != sha:
            rep.add("C6", ERROR,
                    f"assets/textures/{name} 의 SHA-256 이 meta/assets.json({key} bundle.files)과 다름 — "
                    "출처 기록이 가리키는 파일이 아니다", "assets/textures/")
    if base.is_dir():
        for q in sorted(base.rglob("*")):
            if q.is_dir():
                continue
            rel = q.relative_to(base)
            if len(rel.parts) == 1 and rel.name in names:
                continue
            if f"{ASSETS_DIR}/{rel.as_posix()}" in robot_files:
                continue
            if rel.parts[0] == TEXTURE_SUBDIR and len(rel.parts) == 2:
                if rel.name not in claims:
                    rep.add("C6", ERROR,
                            f"assets/{rel.as_posix()} 의 출처 기록이 없음 — 싣는 USD 의 텍스처 목록"
                            "(meta/assets.json bundle.files)에 없는 파일이다", "assets/textures/")
                continue
            rep.add("C6", ERROR,
                    f"배포본 assets/ 에 이 배치의 에피소드가 쓰지 않는 파일: assets/{rel.as_posix()} — "
                    "에피소드의 scene_assets 가 재배포 자산으로 가리키는 USD 와 그 텍스처, 폴더째 싣는 로봇의 "
                    "ship.files 행, 엮는 자산이 모두 실릴 때의 엮은 자산(composes)만 싣는다",
                    "assets/")
    return hashed


def _texture_row_problems(row: object) -> list[str]:
    """What is wrong with one bundle.files row as a statement of provenance (the converter's
    generation.assets.texture_row_problems, restated: this file does not import generation)."""
    if not isinstance(row, dict):
        return [f"항목이 객체가 아님: {type(row).__name__}"]
    out = [f"{k} 가 비어 있지 않은 문자열이 아님: {row.get(k)!r}"
           for k in TEXTURE_ROW_KEYS if not isinstance(row.get(k), str) or not row.get(k)]
    unknown = sorted(set(row) - set(TEXTURE_ROW_KEYS) - set(TEXTURE_ROW_OPTIONAL))
    if unknown:
        out.append(f"모르는 키: {unknown}")
    name = row.get("file")
    if isinstance(name, str) and name and (Path(name).name != name or name in (".", "..")
                                           or "\\" in name):
        out.append(f"file 이 경로가 아닌 파일 이름이어야 함: {name!r}")
    sha = row.get("sha256")
    if isinstance(sha, str) and not re.fullmatch(r"[0-9a-f]{64}", sha):
        out.append(f"sha256 이 64자리 소문자 16진수가 아님: {sha!r}")
    return out


def c7_timeseries(frames: list[tuple[Path, pd.DataFrame]],
                  eps: pd.DataFrame | None, rep: Report,
                  partial: bool = False) -> None:
    """C7 — 타임스탬프 단조 증가, 프레임 결측, fps 정합."""
    if not frames:
        rep.add("C7", SKIP, "스텝 데이터 없음")
        return

    fps_key, fps_by_ep = episode_lookup(eps, "fps")
    report_unjoined_episodes(frames, any_frame_group_key(frames, fps_key),
                             fps_by_ep, "C7", rep, partial=partial)

    for path, df in frames:
        loc = str(path)
        for col in ("timestamp", "frame_index"):
            if col not in df.columns:
                rep.add("C7", ERROR, f"{col} 컬럼 없음", loc)
                return

        # 한 parquet 에 여러 에피소드가 들어간다 (LeRobotDataset v3.0 의 기본
        # 배치). 단조 증가와 프레임 연속성은 에피소드 단위 성질이므로 파일
        # 전체로 재면 에피소드 경계마다 거짓 위반이 나온다. episode_id 가
        # 없으면 LeRobot 자신의 episode_index 로 묶는다.
        key = frame_group_key(df, fps_key)
        groups: Iterable[tuple[Any, pd.DataFrame]] = (
            df.groupby(key, sort=False) if key else [("<file>", df)]
        )

        for eid, g in groups:
            tag = f"{loc}::{eid}"
            ts = pd.to_numeric(g["timestamp"], errors="coerce").to_numpy()
            fi = pd.to_numeric(g["frame_index"], errors="coerce").to_numpy()

            if np.isnan(ts).any():
                rep.add("C7", ERROR, "timestamp 에 결측/비수치", tag)
                continue
            d = np.diff(ts)
            if (d <= 0).any():
                bad = int(np.argmax(d <= 0))
                rep.add("C7", ERROR, f"timestamp 단조 증가 위반 (index {bad})", tag)

            if np.isnan(fi).any():
                rep.add("C7", ERROR, "frame_index 에 결측", tag)
            else:
                expect = np.arange(fi[0], fi[0] + len(fi))
                if not np.array_equal(fi, expect):
                    gaps = int(np.sum(np.diff(fi) != 1))
                    rep.add("C7", ERROR, f"frame_index 불연속 {gaps}곳 — 프레임 결측", tag)

            declared = numeric_or_none(fps_by_ep.get(str(eid)))
            # 0 이나 음수 fps 는 C5 가 오류로 보고한다. 여기서 나누면 검증기가 통째로
            # 죽어 exit 3 이 되는데, 그건 「막았다」가 아니라 「검사하지 못했다」이다.
            if len(d) and declared is not None and declared > 0:
                nominal = 1.0 / declared
                med = float(np.median(d))
                if nominal > 0 and abs(med - nominal) / nominal > 0.05:
                    rep.add("C7", WARN,
                            f"실측 주기 {1/med:.1f}Hz 가 선언 fps {declared:.1f}Hz 와 5% 이상 차이",
                            tag)
                jitter = float(np.std(d))
                if nominal > 0 and jitter / nominal > 0.1:
                    rep.add("C7", WARN, f"샘플링 지터 과다 (σ/Δt = {jitter/nominal:.1%})", tag)


def c7_step_nulls(ds: Dataset, frames: list[tuple[Path, pd.DataFrame]], rep: Report) -> None:
    """C7 — no null in the step table, a list's elements included.

    The spec's non-finite values are NaN and ±inf (safety.zmp, safety.stability_margin): values,
    and LeRobotDataset reads them. A null is a missing value: measured with lerobot 0.6.1, a step
    whose value or whole list is null comes out with None in that column, and a step whose list
    holds a null fails as its item is read (validator/README.md, 「features 가 선언한 스텝 표」).
    Read through pandas the two look the same -- which is why this reads the files through Arrow
    (lerobot 0.6.1's aggregate_datasets rewrites the step table through pandas). The converter
    writes no null.

    ``frames`` is load_frames' list: its paths are relative to the distribution's root, and the
    same files it read are the ones read here (so --file-limit bounds this check too).
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        rep.add("C7", SKIP, "pyarrow 가 없어 스텝 데이터의 null 을 확인하지 못함")
        return

    def inner(arr) -> int:
        t = arr.type
        if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
            flat = arr.flatten()
            return flat.null_count + inner(flat)
        return 0

    for path, _ in frames:
        try:
            table = pq.read_table(ds.root / path)
        except Exception as e:  # noqa: BLE001 -- an unreadable file is a finding, not a crash
            rep.add("C7", ERROR, f"스텝 parquet 을 Arrow 로 읽지 못함: {type(e).__name__}: {e}", str(path))
            continue
        for name in table.column_names:
            col = table.column(name)
            n = col.null_count + sum(inner(c) for c in col.chunks)
            if n:
                rep.add("C7", ERROR,
                        f"{name} 에 null {n}개 — 스텝 데이터에 결측은 없다. 규격의 비유한값은 NaN·±inf 로 "
                        "실린다", str(path))


def _integer_column(column: pd.Series) -> np.ndarray | None:
    """The column's values when it is a column of integers -- not floats that happen to be whole
    (what an integer column with a null reads back as), booleans or strings."""
    values = column.to_numpy()
    return values if values.dtype.kind in "iu" else None


def _as_written(value: object, limit: int = 80) -> str:
    """A value of info.json as a finding shows it: its repr, and of a long one the start with a
    mark. Cut without the mark, a number of a hundred digits reads as another number."""
    text = repr(value)
    return text if len(text) <= limit else f"{text[:limit]}… (전체 {len(text)}자)"


def _beyond(values: np.ndarray, low: int, high: int) -> np.ndarray:
    """Which of an integer column's values lie outside ``low``..``high``, the bounds of an
    integer type. Each bound is compared in the column's own type, and only where that type
    reaches past it: -1 has no uint64, and 2**63 no int64, to be compared as."""
    kind = np.iinfo(values.dtype)
    none = np.zeros(len(values), dtype=bool)
    below = values < values.dtype.type(low) if low > kind.min else none
    above = values > values.dtype.type(high) if high < kind.max else none
    return below | above


def _table_files(directory: Path) -> tuple[list[Path], list[Path]]:
    """A table directory's parquet files, in name order: (the ones LeRobot loads -- one directory
    down --, the ones anywhere else under it). Both empty for a directory that is not there."""
    every = _parquet_under(directory)
    loaded = [f for f in every if len(f.relative_to(directory).parts) == TABLE_FILE_PARTS]
    return loaded, [f for f in every if len(f.relative_to(directory).parts) != TABLE_FILE_PARTS]


def _never_fits(later, first) -> bool:
    """A column typed ``later`` in a later file cannot be put into a table whose first file
    types it ``first``, whatever its values are (pyarrow types, as the footers give them).

    The loader types the episode table by its first file and casts every later file to that,
    with datasets' cast_array_to_feature. This is that routine, kept to what turns on the types
    alone:
      - a struct goes into a struct that has all of its fields, field by field (a field the
        later file lacks is filled with nulls) -- into nothing else;
      - a list, of any of the three kinds, goes into a list by its elements; a fixed-length
        list into a fixed-length list only of the same length; a list into nothing else;
      - null goes into anything;
      - a leaf with values goes into no list or struct, and not into null. Into another leaf
        it is Arrow's cast, which succeeds or not by the values or by the pair of kinds.

    Held to the loader by measurement, not by reading (tests/validator/loader_casts_measured.json,
    taken with scripts/measure_loader_casts.py: every ordered pair of twenty types through the
    loader's cast and its reader). Of those 400 pairs this says none that the loader reads, and
    misses only pairs of leaves that Arrow has no cast between at all (binary into int64, a
    timestamp into bool): those it leaves alone, like every difference that turns on the values.
    """
    import pyarrow as pa

    t = pa.types

    def a_list(kind) -> bool:
        return t.is_list(kind) or t.is_large_list(kind) or t.is_fixed_size_list(kind)

    if t.is_struct(later):
        if not t.is_struct(first):
            return True
        into = {field.name: field.type for field in first}
        return any(field.name not in into or _never_fits(field.type, into[field.name]) for field in later)
    if a_list(later):
        if not a_list(first):
            return True
        if t.is_fixed_size_list(later) and t.is_fixed_size_list(first) and later.list_size != first.list_size:
            return True
        return _never_fits(later.value_type, first.value_type)
    if t.is_null(later):
        return False
    return t.is_struct(first) or a_list(first) or t.is_null(first)


def _declares_one_integer(declared: object) -> bool:
    """info.json's feature for a column is one integer a step, as the loader makes it.

    LeRobot turns a feature of shape (1,) into a datasets.Value of its dtype (feature_utils.
    get_hf_features_from_features), and a Value takes pyarrow's type names. Measured with the
    loader: another shape is a sequence the int64
    column cannot be cast to, no shape is a KeyError, and numpy's names for an integer ("long",
    "<i8", "int") are refused.
    """
    if not isinstance(declared, dict):
        return False
    dtype, shape = declared.get("dtype"), declared.get("shape")
    return (isinstance(dtype, str) and dtype in INTEGER_FEATURE_DTYPES
            and isinstance(shape, (list, tuple)) and list(shape) == [1])


def c7_table_files(ds: Dataset, rep: Report) -> None:
    """C7 — the files of the two tables are files LeRobot's loader reads as a table.

    What comes before the pointers (c7_data_spans): whether a loader gets a table at all, and
    from which files. None of this shows when the files are read through pandas -- the pointer
    check compared every episode of such releases and said nothing (each case rerun with the
    real loader, 2026-10-08):

      - which files: a table is the parquet one directory down (TABLE_FILE_PARTS). A parquet
        anywhere else under data/ or meta/episodes/ is no part of what a loader gets, and a
        table with no file there is no table -- the validator's own readers take any depth,
        and for the episodes a JSON file in place of the parquet;
      - a file whose path under the release holds a character `datasets` takes for a pattern
        (GLOB_CHARACTERS). Such a path is matched, not opened: measured, "file-[1]00.parquet"
        is passed over (the class matches "1"), "file-00?.parquet" and "file-*.parquet" bring
        in "file-000.parquet" a second time, and under "chunk-00[0]" nothing is found. The file
        is there and whole, so its rows count and its pointers hold. Said for the three
        characters wherever they stand -- also where the pattern happens to match the file
        alone, since what is read then turns on what else is in the directory;
      - a file whose first row group has no rows, which is what an empty table saved to parquet
        is. datasets takes a file's first row group's row count for its batch size and fails on 0
        (packaged_modules/parquet; measured: rows after it do not help, an empty group after
        rows or no group at all loads). Concatenated, such a file just vanishes;
      - two columns of one name in a file: the reader looks columns up by name and fails;
      - the episode table's files typed so that the loader cannot put them together. It types
        that table by its first file and casts the others to it. Only what fails is said: a
        pointer column (the ones LeRobot itself writes, always as one type) typed differently --
        the cast fails on a value that does not fit, and pandas widens instead, int8 beside
        int16 reading as one int16 column --; a column whose type can never be cast to the
        first file's (_never_fits: the first file left it null and a later one has values, a
        list of another depth, a struct with another field, ...); a column the first file does
        not have. A later file with no row group is not typed against the first at all: the
        reader makes no batch of it, so nothing is cast. The step table is typed by info.json's
        features, not by a file, so its files are not held to each other (c7_step_features
        holds each to the features);
      - info.json's features declaring index, episode_index, frame_index or task_index as
        anything but one integer a step (_declares_one_integer). The loader reads the steps as
        declared, and looks an episode up in the episode table by the episode_index it read and
        a task in the task table by the task_index: 0.0 is no row number, and a shape or a type
        name it cannot make a column of stops it earlier.

    Footers and info.json only -- no row is read, so every file is looked at whatever
    --file-limit says and whether or not a step was read. A file that is not parquet is said
    by the readers (C1) and passed over.

    Not looked at:
      - the characters of the path the release itself is put at: that is not the release's;
      - differences between the episode table's files that the loader reads through or that
        turn on the values: outside the pointer columns, integer widths, an integer beside a
        float, kinds of string, a later file's null or missing column -- and pairs of leaf
        types Arrow has no cast between at all (_never_fits says which were measured);
      - whether the step files are the columns the features declare and can be cast to them --
        a column of the four that is not declared at all, an episode_index of 128 beside a
        declared int8: c7_step_features;
      - the types and row groups of a file where the loader does not look.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        rep.add("C7", SKIP, "pyarrow 가 없어 표의 파일(빈 행 그룹 · 파일마다의 형)을 확인하지 못함")
        return

    tables = (
        (DATA_DIR, "스텝 표",
         "LeRobot 이 읽지 않는 자리의 parquet — 스텝 표는 data/<chunk>/<file>.parquet 한 단계 아래의 "
         "파일뿐이라 이 파일의 행은 내주지 않는다"),
        (f"meta/{EPISODES_DIR}", "에피소드 표",
         "LeRobot 이 읽지 않는 자리의 에피소드 표 — meta/episodes/<chunk>/<file>.parquet 한 단계 아래의 "
         "파일뿐이라 이 파일의 행은 적재기의 표에 없다"),
    )
    looked: dict[str, int] = {}
    for rel_dir, what, stray_said in tables:
        loaded, stray = _table_files(ds.root / rel_dir)
        for f in stray:
            rep.add("C7", ERROR, stray_said, str(f.relative_to(ds.root)))
        if not loaded:
            # Every file a stray, or none at all -- for the episodes, also when what the
            # validator read is its fallback (meta/episodes.jsonl · .json).
            rep.add("C7", ERROR,
                    f"LeRobot 이 읽는 {what}가 없음 — {rel_dir}/<chunk>/<file>.parquet 이 하나도 없어 "
                    "적재기는 이 배포본을 열지 못한다", rel_dir)
        # The first file with a schema is what the loader types the episode table by.
        first: tuple[str, dict] | None = None
        looked[rel_dir] = 0
        for f in loaded:
            rel = str(f.relative_to(ds.root))
            magic = sorted(set(GLOB_CHARACTERS) & set(rel))
            if magic:
                rep.add("C7", ERROR,
                        f"{what}의 파일 경로에 datasets 가 패턴으로 읽는 글자가 있음 — LeRobot 은 표의 파일 "
                        "경로를 datasets 에 넘기고, datasets 는 이런 경로를 이름으로 열지 않고 패턴으로 풀어 "
                        "맞는 파일을 읽는다(이 파일을 건너뛰거나 이웃 파일을 한 번 더 읽는다)",
                        f"{rel} · {' '.join(magic)}")
            try:
                footer = pq.read_metadata(f)
                fields = list(footer.schema.to_arrow_schema())
            except Exception:  # noqa: BLE001 -- not a parquet: the readers' finding (C1), not this check's
                continue
            looked[rel_dir] += 1
            if footer.num_row_groups and footer.row_group(0).num_rows == 0:
                rep.add("C7", ERROR,
                        f"{what}에 첫 행 그룹이 0행인 parquet — LeRobot 이 쓰는 datasets 는 그 행 수를 배치 "
                        "크기로 써서 읽다 실패한다(빈 표를 저장하면 이런 파일이 된다)", rel)
            # By position, not by name: schema.field(name) raises on a name that is there twice.
            names = [field.name for field in fields]
            twice = sorted({name for name in names if names.count(name) > 1})
            if twice:
                rep.add("C7", ERROR,
                        f"{what}의 parquet 에 같은 이름의 열이 둘 이상 — LeRobot 이 쓰는 datasets 는 열을 "
                        "이름으로 찾다 실패한다", f"{rel} · {twice}")
                continue                    # which of the two a type belongs to cannot be said
            if rel_dir == DATA_DIR:
                continue                    # typed by the features, not by a file
            types = {field.name: field.type for field in fields}
            if first is None:
                first = (rel, types)        # with or without rows: the reader takes the first schema it can read
                continue
            if not footer.num_row_groups:
                continue                    # no batch is made of it, so nothing of it is ever cast
            for name, kind in types.items():
                shown = f"{rel} · {name}: {kind}"
                if name not in first[1]:
                    rep.add("C7", ERROR,
                            "에피소드 표의 첫 파일에 없는 열이 뒤 파일에 있음 — LeRobot 은 첫 파일의 열로 표를 "
                            "정해 이 파일을 넣지 못한다", f"{shown} (첫 파일 {first[0]}: 없음)")
                elif name in EPISODE_POINTER_COLUMNS and kind != first[1][name]:
                    rep.add("C7", ERROR,
                            "에피소드 표의 포인터 열의 형이 파일마다 다름 — LeRobot 은 첫 파일의 형에 맞춰 읽고 "
                            "들어가지 않는 값에서 실패한다. pandas 로 읽으면 넓은 쪽으로 합쳐져 보이지 않는다",
                            f"{shown} (첫 파일 {first[0]}: {first[1][name]})")
                elif _never_fits(kind, first[1][name]):
                    rep.add("C7", ERROR,
                            "에피소드 표의 이 열을 첫 파일의 형에 맞출 수 없음 — LeRobot 은 첫 파일의 형으로 표를 "
                            "정하고, 이 차이는 값이 무엇이든 실패한다(첫 파일의 값이 전부 비어 빈 형으로 실린 "
                            "경우 · 목록의 깊이 · struct 의 필드 · 고정 길이)",
                            f"{shown} (첫 파일 {first[0]}: {first[1][name]})")

    info = ds.info()
    features = info.get("features") if isinstance(info, dict) else None
    if isinstance(features, dict):
        for name in STEP_PLACE_COLUMNS + (TASK_INDEX,):
            if name not in features:
                continue
            if not _declares_one_integer(features[name]):
                rep.add("C7", ERROR,
                        f"info.json 의 features 가 {name} 을 정수 하나(dtype int8~uint64 · shape [1])로 선언하지 "
                        f"않음({features[name]!r}) — LeRobot 은 스텝 표를 선언한 대로 바꿔 읽고 이 열의 값을 "
                        "자리로 쓴다", "meta/info.json")
    # How many files of each table were looked at.
    rep.stats["table_files"] = looked


def _feature_column(declared: object) -> tuple[str, object]:
    """What LeRobot's loader makes of one of info.json's features, for the step table.

    (kind, detail):
      - "video": a camera stream -- no column of the step table;
      - "value": shape [1] -- one value a step, ``detail`` its pyarrow type;
      - "list": shape [n], n above 1 -- a list of n a step, ``detail`` (the elements' type, n);
      - "stops": the loader cannot make a table with it and stops at opening, ``detail`` why;
      - "unread": a column, of a form not judged here.

    The loader's way (lerobot 0.6.1): info.json into a DatasetInfo, which turns a shape that is a
    list into a tuple and nothing else; then feature_utils.get_hf_features_from_features -- a
    "video" is passed over, an "image" is a picture, a shape equal to (1,) a datasets.Value of
    the dtype, a shape of length 1 a list of that length of such Values, of length 2 to 5 an
    array of that many dimensions, any other length a ValueError. A Value makes its type of the
    dtype by calling pyarrow's function of that name (FEATURE_DTYPES).

    What stops and what gives a type is held to the loader by measurement, not by reading
    (tests/validator/feature_casts_measured.json, taken with scripts/measure_loader_casts.py:
    each form as the feature of a column no file has). "unread" is what this does not say
    either way: more dimensions than one, a length that is no integer above 0 ([-1] is a list
    of any length to the loader, [true] one value), a dtype pyarrow has a function for that is
    not in FEATURE_DTYPES (the loader calls whatever carries the name) or one written with an
    argument ("timestamp[s]"), a picture.
    """
    import pyarrow as pa

    if not isinstance(declared, dict):
        return "stops", "객체가 아님"
    if "dtype" not in declared:
        return "stops", "dtype 이 없음"
    dtype = declared["dtype"]
    if isinstance(dtype, str) and dtype == VIDEO_FEATURE_DTYPE:
        return "video", None
    if isinstance(dtype, str) and dtype == IMAGE_FEATURE_DTYPE:
        return "unread", None
    if "shape" not in declared:
        return "stops", "shape 이 없음"
    shape = declared["shape"]
    try:
        dimensions = len(shape)
    except TypeError:                   # null, a number: nothing a length can be taken of
        return "stops", "shape 이 목록이 아님"
    if dimensions == 0 or dimensions > FEATURE_SHAPE_DIMENSIONS:
        return "stops", f"shape 의 길이가 1~{FEATURE_SHAPE_DIMENSIONS} 이 아님"
    if dimensions > 1:
        return "unread", None
    if isinstance(shape, dict):         # a length is looked up by position: no key 0 in a JSON object
        return "stops", "shape 이 목록이 아님"
    length = shape[0]
    if length is None or isinstance(length, str):
        return "stops", "shape 의 길이가 정수가 아님"
    if type(length) is not int or length < 0:      # true, 3.0, -1: read by the loader in ways of their own
        return "unread", None
    if length == 0:
        return "stops", "shape 의 길이가 0"
    if not isinstance(dtype, str):
        return "stops", "dtype 이 문자열이 아님"
    name = FEATURE_DTYPE_ALIASES.get(dtype, dtype)
    if name == "json":
        return "stops", "dtype 'json' 은 datasets 가 받지 않음"
    known = next((n for n in (name, name + "_") if n in FEATURE_DTYPES), None)
    if known is None:
        # datasets looks the name up among pyarrow's own names (its module's dict) and calls what
        # it finds; only a name that is not there, and is not one it reads by pattern, is refused.
        if name in vars(pa) or name + "_" in vars(pa) or name.endswith(("]", ")")):
            return "unread", None
        return "stops", "dtype 이 pyarrow 의 형 이름이 아님"
    kind = getattr(pa, known)()
    return ("value", kind) if length == 1 else ("list", (kind, length))


def _storage_type(stored):
    """The type an Arrow extension type stores its values as -- a fixed-shape tensor is a
    fixed-length list underneath -- and any other type itself. datasets casts the storage of a
    column that is an extension array."""
    import pyarrow as pa

    return stored.storage_type if isinstance(stored, pa.BaseExtensionType) else stored


def _extension_inside(stored) -> bool:
    """An extension type somewhere under a stored type's top: the elements of a list, a field of
    a struct, or the storage of an extension type itself. What the loader does with such a column
    is not judged: measured, a list of extension values reads where no list is null and stops
    where one is, and that is not all there is to it."""
    import pyarrow as pa

    t = pa.types
    if isinstance(stored, pa.BaseExtensionType):
        return True
    if t.is_list(stored) or t.is_large_list(stored) or t.is_fixed_size_list(stored):
        return _extension_inside(stored.value_type)
    if t.is_struct(stored):
        return any(_extension_inside(field.type) for field in stored)
    return t.is_dictionary(stored) and _extension_inside(stored.value_type)


def _stored_as_declared(stored, kind: str, detail) -> bool:
    """A stored column's type is the very type the feature makes (_feature_column's "value" or
    "list"): nothing is cast then. LeRobot's writer stores every column so."""
    import pyarrow as pa

    stored = _storage_type(stored)
    if kind == "value":
        return stored == detail
    leaf, length = detail
    return pa.types.is_fixed_size_list(stored) and stored.list_size == length and stored.value_type == leaf


def _arrow_refuses(values, into) -> str | None:
    """Arrow's own words for refusing its checked cast of ``values`` to ``into``, or None where it
    casts. The very call datasets makes for a value that is no list and no struct."""
    try:
        values.cast(into)
    except Exception as e:  # noqa: BLE001 -- whatever Arrow raises on this cast, the loader's cast raises too
        return f"{type(e).__name__}: {' '.join(str(e).split())[:160]}"
    return None


def _cast_stops(column, kind: str, detail) -> str | None:
    """Why LeRobot's loader cannot turn a stored step column into what its feature declares, or
    None where it can (``column``: a pyarrow Array or ChunkedArray; ``kind``, ``detail``:
    _feature_column's "value" or "list").

    The loader casts every batch of a step file to the features' schema with datasets'
    cast_array_to_feature. This is that routine for the two kinds of feature a column is held
    to here:
      - one value a step: a list or a struct stored is no value; a null column goes into
        anything; another leaf is Arrow's own cast, the checked one -- it is called here as the
        loader calls it, so what it refuses there it refuses here (128 into int8, 2.5 or NaN
        into an integer, "abc" into a number, an integer a float cannot hold exactly);
      - a list of n a step: a fixed-length list goes in only at that length; a list of any
        length when every list that is not null is n long; then the elements as above. A value
        or a struct stored is no list; a null column goes in.
    A column that is an Arrow extension array is cast as what it stores, as datasets casts it
    (one with an extension type further down is not brought here: _extension_inside).
    A cast that goes through may still change what is stored -- 1e300 into float32 is inf, 0.4
    into bool true, a float64 into float16 -- and that is not said here: only what stops.

    This asks of the column as a whole what the loader asks batch by batch, and the answers are
    the same but for one thing, which is not said here: a null list among lists of any length,
    which the loader counts wrong in a later batch (_null_lists_misread).

    Held to the loader by measurement (tests/validator/feature_casts_measured.json: every
    stored column beside every declaration of tests/validator/feature_casts.py).
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    t = pa.types

    def a_list(stored) -> bool:
        return t.is_list(stored) or t.is_large_list(stored) or t.is_fixed_size_list(stored)

    def stored_as(values):
        return values.storage if isinstance(values, pa.ExtensionArray) else values

    def value(values, into) -> str | None:
        stored = values.type
        if stored == into:
            return None
        if a_list(stored) or t.is_struct(stored):
            return f"{stored} 은 값 하나({into})가 되지 못함"
        if t.is_null(into) and not t.is_null(stored):
            return f"{stored} 은 빈 형(null)이 되지 못함"
        return _arrow_refuses(values, into)

    chunks = [stored_as(chunk) for chunk in (column.chunks if isinstance(column, pa.ChunkedArray) else [column])]
    if kind == "value":
        return next((why for why in (value(chunk, detail) for chunk in chunks) if why), None)
    leaf, length = detail
    row = 0
    for chunk in chunks:
        stored = chunk.type
        if t.is_fixed_size_list(stored):
            if stored.list_size != length:
                return f"길이 {stored.list_size} 로 고정된 목록은 길이 {length} 의 목록이 되지 못함"
        elif t.is_list(stored) or t.is_large_list(stored):
            lengths = chunk.value_lengths()
            off = pc.fill_null(pc.not_equal(lengths, length), False)
            if pc.any(off).as_py():
                k = pc.index(off, True).as_py()
                return (f"선언한 길이({length})와 다른 목록이 있음 ({row + k}번째 행의 길이 {lengths[k].as_py()} · "
                        f"그런 행 {pc.sum(off).as_py()}개)")
        elif t.is_null(stored):
            row += len(chunk)
            continue
        else:
            return f"{stored} 은 목록({leaf} {length}개)이 되지 못함"
        why = value(chunk.flatten(), leaf)
        if why:
            return why
        row += len(chunk)
    return None


def _batches_read_off(null: np.ndarray, groups: list[int]) -> Iterator[tuple[int, int, int, int, bool]]:
    """The batches LeRobot's loader reads at the wrong place after a null list (see
    _null_lists_misread), in file order: (the row group's first row, the row group's rows, the
    batch's first row within it, the batch's rows, whether it runs past the row group's
    elements -- which stops the opening)."""
    size = groups[0] if groups else 0
    if size <= 0:
        return                          # no batch can be made: the reader fails before (c7_table_files)
    start = 0
    for rows in groups:
        here = null[start:start + rows]
        lists = rows - int(here.sum())  # the lists of the row group that have elements
        for at in range(0, rows, size):
            batch = here[at:at + size]
            if not batch.any() and here[:at].any():
                yield start, rows, at, len(batch), at + len(batch) > lists
        start += rows


def _null_lists_misread(null: np.ndarray, groups: list[int]) -> tuple[str, int] | None:
    """What LeRobot's loader does after a null list in a column of lists of any length that it
    turns into lists of a fixed length: ("stops" | "shifts", the first row of the first batch it
    reads at the wrong place), or None where it reads every batch at its own place. ``null``:
    which rows of the file hold a null list; ``groups``: the row count of each of the file's row
    groups, in order.

    The loader casts a file batch by batch, a batch being as many rows as the file's first row
    group holds, cut out of one row group at a time. datasets turns a batch of lists into
    fixed-length ones in two ways: a batch with a null in it by slicing every list, which is
    right; a batch with none by taking the elements at its row offset times the length -- and
    a null list before it in its row group has no elements, so the place is off by one list for
    each. That batch then gets the elements of later rows ("shifts": nothing is raised, and the
    steps come out with the lists of other steps), or runs past the row group's elements and
    comes out short, which stops the opening ("stops").

    So: a batch of no null that comes after a null list of its row group is read at the wrong
    place, and the opening stops if any such batch reaches past its row group's elements.
    Whether a step read at the wrong place comes out with other values than its own turns on
    what the lists hold -- where they all hold the same, nothing shows -- and is not this
    function's to say (_first_step_handed_another). Measured with the loader
    (tests/validator/feature_casts_measured.json "batches": the nulls and row groups of
    tests/validator/feature_casts.py, each step's value compared with what was stored), on a
    release with camera streams and on one without.

    That is the release opened whole. Opened for chosen episodes (episodes=[...]) the loader
    filters each batch before it casts it, a filtered batch is rows of its own, and every one of
    the measured placings is read as stored -- also with every episode chosen.
    """
    first: int | None = None
    for start, _, at, _, short in _batches_read_off(null, groups):
        if first is None:
            first = start + at
        if short:
            return "stops", first
    return None if first is None else ("shifts", first)


def _same_values(own: object, other: object) -> bool:
    """Two rows' values, as Python hands them out, are the same -- a NaN being the same as a NaN."""
    if isinstance(own, list) and isinstance(other, list):
        return len(own) == len(other) and all(_same_values(a, b) for a, b in zip(own, other))
    return own == other or (isinstance(own, float) and isinstance(other, float) and own != own and other != other)


def _first_step_handed_another(column, null: np.ndarray, groups: list[int]) -> int | None:
    """Of a column the loader reads through at the wrong place (_null_lists_misread's "shifts"):
    the first row that comes out with other values than it stores, or None where every list
    read in another's place holds what that one holds -- the steps then come out as stored
    (measured: lists that are all alike, and two neighbours that are).

    A batch read at the wrong place gets, for its k-th row of the row group, the k-th of the
    row group's lists that have elements."""
    import pyarrow as pa

    whole = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if isinstance(whole, pa.ExtensionArray):
        whole = whole.storage
    for start, rows, at, length, _ in _batches_read_off(null, groups):
        group = whole.slice(start, rows)
        own, got = group.slice(at, length), group.drop_null().slice(at, length)
        if own.equals(got):
            continue
        for k, (a, b) in enumerate(zip(own.to_pylist(), got.to_pylist())):
            if not _same_values(a, b):
                return start + at + k
    return None


def c7_step_features(ds: Dataset, rep: Report) -> None:
    """C7 — the step files are the table info.json's features declare, as LeRobot's loader has it.

    LeRobot's loader does not type the step table by a file. It makes the table's columns and
    their types of info.json's features and puts every step file into that: a release whose
    files and features do not say the same is one a loader does not open, or opens with a
    column of nothing -- and read through pandas, as every other check reads the steps, the
    files are whole. What the loader does (lerobot 0.6.1 · datasets 4.8.5, measured 2026-10-10
    on a release with camera streams and on one without. Whether it opens differed on one case
    of 216, and by the rows the two hold: the box declared uint16 stops only where a -1 is):

      - without features, or with something that is no JSON object in their place, it stops
        at opening; so it does on one feature it cannot make a column of (_feature_column:
        no dtype, no shape, a dtype that is no type name of pyarrow -- numpy's "<f8", "long",
        "str" --, a shape of no length or of more than five, a length that is no number);
      - every feature but a camera stream's is a column. A step file's column that is not
        among them stops the opening (also a column named as a camera stream is);
      - a column among them that a step file does not have is filled with nulls. Nothing
        stops: the file's steps come out with None under that key, or, where more dimensions
        than one are declared, with an array of NaN (the four columns of place are said by the
        checks that read them -- c7_data_spans, c7_task_table: without index no item is read,
        without episode_index the loader goes to the Hub). Not said of a declaration the
        loader cannot make a column of: that one stops it;
      - a column stored as another type than declared is cast to it, and where the cast fails
        the opening stops (_cast_stops): a list where one value is declared or the other way
        round, another fixed length, a list of another length, a value the declared type
        cannot hold. This is the one rule for the widths of index, episode_index, frame_index
        and task_index too: 128 beside a declared int8 stops the opening for each alike;
      - lists of any length that are all as long as declared are read -- unless one of them is
        null and a batch without a null follows it in its row group: the loader then reads
        that batch at the wrong place, and stops or hands out other steps' lists with no
        error (_null_lists_misread). Said where it stops, and where a step comes out with
        other values than it stores (_first_step_handed_another); where the lists read in
        each other's place hold the same, the steps come out as stored and nothing is said. A
        null is a finding by itself (c7_step_nulls), but only in the files --file-limit lets
        that check read.

    Read here: info.json and the schema of every step file the loader reads (_table_files) --
    every file, whatever --file-limit says. A file's rows are read only where a column is not
    stored as the very type declared, and then cast as the loader casts them. A file that holds
    no row is passed over -- with no row group the reader makes no batch of it and nothing of it
    is cast, and on an empty first one it stops, which c7_table_files says -- and one whose rows
    cannot be read is left with a SKIP and out of the count of what was held. Without an
    info.json that reads as a JSON object nothing is held, and a SKIP says so: what is wrong
    with that file is C1's to say, and C1 may not have been asked to run.

    All of this is the release opened whole, as LeRobotDataset opens it by default. Opened for
    chosen episodes (episodes=[...]) the loader filters the rows before it casts them (measured):
    what stops it by a declaration or by a column's type stops it all the same, a column no file
    has is as empty, but a value or a list that does not go into the declared type stops it only
    if a chosen episode holds it, and no step is read wrong after a null list.

    Each kind of defect has one sentence, and a finding's numbers are in its location: features
    that are gone make every column of every file an undeclared one, and those fold into a line.

    Not looked at:
      - a cast that goes through and changes the values: a float64 stored beside a declared
        float32 or float16, a number beside a declared bool or string. The loader hands out
        what the cast made, saying nothing;
      - the declarations _feature_column leaves "unread" -- more dimensions than one, a
        picture, a dtype with an argument -- beyond their being a column of the table: a file
        without one is said, its type is not;
      - the two language columns, which the loader types itself (LANGUAGE_COLUMNS): declared,
        they are columns a file may have, and nothing else is said of them;
      - a column stored with an Arrow extension type under its top -- in a list's elements, in
        a struct -- (_extension_inside): a column of the table, its type not held to anything;
      - whether a place column is declared an integer (c7_table_files), and that a null is in
        a column at all (c7_step_nulls): of nulls, only the steps read wrong after one are said;
      - the step files against each other: two files typed differently are each cast to the
        features, and are read if each can be;
      - the validator's pyarrow is not the loader's: Arrow's cast is called as the loader
        calls it, by whichever pyarrow is here.
    """
    # What was held to the features. A number that is not here was not compared.
    held: dict[str, int] = {}
    rep.stats["step_features"] = held
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        rep.add("C7", SKIP, "pyarrow 가 없어 스텝 표를 info.json 의 features 와 대조하지 못함")
        return

    info = ds.info()
    if info is None:
        # No info.json, one that does not parse, or one that is no JSON object. C1 says which --
        # where C1 runs; a check that looked at nothing and said nothing would read as passed.
        rep.add("C7", SKIP, "meta/info.json 을 JSON 객체로 읽지 못해 스텝 표를 features 와 대조하지 못함 "
                            "(그 파일의 결함은 C1 이 말한다)", "meta/info.json")
        return
    features = info.get("features")
    if not isinstance(features, dict):
        rep.add("C7", ERROR,
                "info.json 에 features 가 없거나 JSON 객체가 아님 — LeRobot 은 스텝 표의 열과 형을 features 로 "
                "정하며, 없으면 여는 단계에서 멈춘다",
                "meta/info.json · features " + (_as_written(features) if "features" in info else "없음"))
        return

    made = {name: ("unread", None) if name in LANGUAGE_COLUMNS else _feature_column(declared)
            for name, declared in features.items()}
    place = STEP_PLACE_COLUMNS + (TASK_INDEX,)
    # Of the four columns of place c7_table_files says what is not one integer, this among it.
    unmade = [name for name, (kind, _) in made.items() if kind == "stops" and name not in place]
    if unmade:
        name = unmade[0]
        rep.add("C7", ERROR,
                "info.json 의 features 에 LeRobot 이 스텝 표의 열로 만들지 못하는 선언이 있음 — 적재기는 스텝 표의 "
                "형을 features 로 정하며, 이런 선언이 하나라도 있으면 여는 단계에서 멈춘다",
                f"meta/info.json · {name}: {_as_written(features[name])} — {made[name][1]} "
                f"(그런 선언 {len(unmade)}개)")
    # The columns a file is missed for lacking: every feature the loader makes a column of (a
    # camera stream is none, and a declaration it stops on is said as that).
    columns = [name for name, (kind, _) in made.items() if kind not in ("video", "stops")]

    loaded, _ = _table_files(ds.root / DATA_DIR)
    undeclared: dict[str, list[str]] = {}       # a column -> the files that have it
    absent: dict[str, list[str]] = {}           # a feature -> the files that lack its column
    uncast: dict[str, list[str]] = {}           # a column -> the files it cannot be cast in
    misread: dict[str, list[str]] = {}          # a column -> the files it is read wrong in
    first_uncast: str | None = None
    first_misread: str | None = None
    files = compared = cast = 0
    for f in loaded:
        rel = str(f.relative_to(ds.root))
        try:
            file = pq.ParquetFile(f)
            fields = list(file.schema_arrow)
            footer = file.metadata
            has_rows = any(footer.row_group(k).num_rows for k in range(footer.num_row_groups))
        except Exception:  # noqa: BLE001 -- no parquet: the readers' to say (C1)
            rep.add("C7", SKIP, "스텝 표에 parquet 으로 읽히지 않는 파일이 있어 그 파일을 features 와 대조하지 못함",
                    rel)
            continue
        names = [field.name for field in fields]
        if not has_rows or len(set(names)) != len(names):
            continue                    # no step in it · two columns of one name: c7_table_files' finding
        for name in names:
            if name not in made or made[name][0] == "video":
                undeclared.setdefault(name, []).append(rel)
        for name in columns:
            if name not in names and name not in place and name not in LANGUAGE_COLUMNS:
                absent.setdefault(name, []).append(rel)
        table = None
        held_here: int | None = 0
        cast_here = 0
        for position, field in enumerate(fields):
            kind, detail = made.get(field.name, ("video", None))
            if kind not in ("value", "list"):
                continue
            if field.name in place and not _declares_one_integer(features[field.name]):
                continue                # not an integer a step: c7_table_files' finding
            stored = _storage_type(field.type)
            if _extension_inside(stored):
                continue                # an extension type under the top: not judged
            held_here += 1
            if _stored_as_declared(field.type, kind, detail):
                continue
            if table is None:
                try:
                    table = file.read()
                except Exception:  # noqa: BLE001 -- a footer that reads and rows that do not: the readers' to say (C1)
                    rep.add("C7", SKIP,
                            "스텝 표에 행이 읽히지 않는 파일이 있어 그 파일의 값을 features 가 선언한 형에 대 보지 못함",
                            rel)
                    held_here = None    # not held: none of its columns is counted
                    break
            cast_here += 1
            column = table.column(position)
            declared = features[field.name]
            shown = f"{rel} · {field.name}: 저장 {field.type}, 선언 {declared['dtype']} {list(declared['shape'])}"
            why = _cast_stops(column, kind, detail)
            if why is not None:
                uncast.setdefault(field.name, []).append(rel)
                first_uncast = first_uncast or f"{shown} — {why}"
                continue
            if kind == "list" and not pa.types.is_fixed_size_list(stored) and column.null_count:
                # Lists of any length, every one as long as declared, and a null among them. (A
                # fixed-length list keeps the place of a null one, so nothing is read off after it
                # -- and parquet does not hand a file with a null one back at all.)
                null = column.is_null().to_numpy()
                groups = [footer.row_group(k).num_rows for k in range(footer.num_row_groups)]
                wrong = _null_lists_misread(null, groups)
                outcome = None
                if wrong is not None and wrong[0] == "stops":
                    outcome = "여는 단계에서 멈춤"
                elif wrong is not None:
                    other = _first_step_handed_another(column, null, groups)
                    if other is not None:   # else every list read in another's place holds the same
                        outcome = f"오류 없이 다른 스텝의 값을 내줌 (값이 처음 달라지는 행 {other})"
                if outcome is not None:
                    misread.setdefault(field.name, []).append(rel)
                    first_misread = first_misread or f"{shown} — {wrong[1]}번째 행부터 밀린 자리에서 읽음: {outcome}"
        if held_here is None:
            continue
        files += 1
        compared += held_here
        cast += cast_here
    held.update(files=files, columns=compared, cast=cast)

    def folded(found: dict[str, list[str]], what: str) -> str:
        """The first file and its columns of the kind, and how many there are in all."""
        first = min(rels[0] for rels in found.values())
        names = [name for name, rels in found.items() if first in rels]
        shown = ", ".join(names[:4]) + (" …" if len(names) > 4 else "")
        return f"{first} · {shown} ({what} {len(found)}개 · 파일 {len({r for rels in found.values() for r in rels})}개)"

    if undeclared:
        rep.add("C7", ERROR,
                "스텝 표에 info.json 의 features 가 스텝 표의 열로 선언하지 않은 열이 있음 — LeRobot 은 스텝 표의 "
                "열을 features 로 정하며(영상으로 선언한 것은 스텝 표의 열이 아니다), 선언에 없는 열이 있는 파일을 "
                "읽다 여는 단계에서 멈춘다", folded(undeclared, "선언에 없는 열"))
    if absent:
        rep.add("C7", ERROR,
                "info.json 의 features 가 선언한 열이 스텝 표에 없음 — LeRobot 은 파일에 없는 열을 빈 값(null)으로 "
                "채워 읽으며 이것으로 멈추지 않는다. 그 파일의 스텝에는 그 열의 값이 없다",
                folded(absent, "파일에 없는 열"))
    if first_uncast is not None:
        rep.add("C7", ERROR,
                "스텝 표의 열을 info.json 의 features 가 선언한 형으로 바꿀 수 없음 — LeRobot 은 스텝 표를 선언한 "
                "형으로 바꿔 읽으며, 바뀌지 않는 행을 읽으면 여는 단계에서 멈춘다(배포본을 통째로 열면 언제나)",
                f"{first_uncast} (바뀌지 않는 열 {len(uncast)}개 · "
                f"파일 {len({r for rels in uncast.values() for r in rels})}개)")
    if first_misread is not None:
        rep.add("C7", ERROR,
                "스텝 표의 가변 길이 목록 열에 null 인 목록이 있어 LeRobot 이 통째로 열 때 그 뒤의 스텝을 잘못 읽음 — "
                "적재기가 쓰는 datasets 는 이런 열을 선언한 고정 길이로 바꿀 때 null 인 목록을 자리로 세지 않아, 같은 행 "
                "그룹에서 그 뒤에 오는 null 없는 배치를 밀린 자리에서 읽는다: 다른 스텝의 값을 내주거나 여는 단계에서 "
                "멈춘다(에피소드를 골라 열면 바르게 읽는다)",
                f"{first_misread} (그런 열 {len(misread)}개 · "
                f"파일 {len({r for rels in misread.values() for r in rels})}개)")


def c7_info_totals(ds: Dataset, rep: Report) -> None:
    """C7 — info.json's total_episodes and total_frames are the row counts of the two tables.

    LeRobot does not count the tables it reads. It takes the two numbers from info.json, where
    it wrote them as it saved each episode, and a release whose tables were rewritten afterwards
    can state numbers its tables no longer have. The pointer check (c7_data_spans) does not see
    that: every episode still points at its own rows.

    What the loader does with each number (lerobot 0.6.1; measured 2026-10-09 with one number
    changed, on a release with camera streams and on one without):

      - total_frames is len(dataset) and nothing else. A reader that goes by the length fails
        on the last items when it is larger than the step table (IndexError) and never gets the
        last rows when it is smaller; a plain ``for`` over the dataset stops where the rows
        stop and gets them all either way. 0 is an empty dataset to whoever asks its length;
      - total_episodes is the number of episodes the loader reports, and two things turn on it.
        At 0 the episode table is not read: items still come out of a release without camera
        streams, and none does out of one with them (the stream of an item is found through
        that table). And the loader takes the copy it has for whole only if the step table has
        steps of every episode from 0 up to the number: when it is larger than that, it goes to
        the Hub for the rest (offline, the release does not open). A number that is smaller
        only makes the loader report fewer episodes;
      - an absent number is read as 0;
      - a number that is not an integer (9636.0, "9636", null) stops len() or the opening with a
        TypeError, and true is taken for 1. So it is the number as written that is compared.

    The tables are counted as the loader reads them: the files one directory down
    (_table_files), every one whatever --file-limit says, each by the rows of its row groups.
    Not by the footer's own total, which is written beside them and which no reader goes by:
    measured, a file whose total alone was raised from 5 to 6 still gives five rows.

    A count whose table cannot be counted is not compared, and is not in the report's stats:
    no file where the loader reads (c7_table_files says that), or a file that is not parquet.
    What is wrong with such a file is the readers' to say (C1), but they do not open a file
    past --file-limit, so that the count went uncompared is said here -- as a SKIP, which shows
    in the summary where an INFO would leave the check reading as passed.

    Not looked at:
      - total_tasks, the third count info.json keeps: c7_task_table holds it to the task table;
      - splits and chunks_size, which the loader does not use to read a release (measured:
        splits naming fewer or more episodes, chunks_size 1, either absent -- the same items);
      - the counts a release states elsewhere. Its README.md's batch marker is held to the
        release when it is published -- the number of episodes and of each label, not the
        number of steps and not what the prose says; meta/stats.json is held to nothing here.
    """
    info = ds.info()
    if info is None:
        return                  # no info.json, or one that is no JSON object: C1's finding
    try:
        import pyarrow.parquet as pq
    except ImportError:
        rep.add("C7", SKIP, "pyarrow 가 없어 info.json 의 total_episodes · total_frames 를 표의 행 수와 "
                            "대조하지 못함")
        return

    counted: dict[str, int] = {}
    for key, (rel_dir, what, why) in INFO_TOTALS.items():
        loaded, _ = _table_files(ds.root / rel_dir)
        if not loaded:
            continue                # no table where a loader reads: c7_table_files' finding
        rows, unread = 0, None
        for f in loaded:
            try:
                footer = pq.read_metadata(f)
            except Exception:  # noqa: BLE001 -- not a parquet: what is wrong with it is the readers' to say (C1)
                unread = f
                break
            rows += sum(footer.row_group(k).num_rows for k in range(footer.num_row_groups))
        if unread is not None:
            rep.add("C7", SKIP, f"{what}에 parquet 으로 읽히지 않는 파일이 있어 info.json 의 {key} 를 대조하지 못함",
                    str(unread.relative_to(ds.root)))
            continue
        counted[key] = rows
        stated = info.get(key, 0)
        if type(stated) is int and stated == rows:      # not bool: True == 1
            continue
        shown = _as_written(stated) if key in info else "없음(LeRobot 은 0 으로 읽음)"
        rep.add("C7", ERROR, f"info.json 의 {key} 가 {what}의 행 수와 다름 — {why}",
                f"meta/info.json · {key} {shown}, {what} {rows}행")
    # What each count was held to. A count that is not here was not compared.
    rep.stats["info_totals"] = counted


def c7_task_table(ds: Dataset, rep: Report) -> None:
    """C7 — a step's task_index names a row of the task table, read as LeRobot reads it.

    LeRobot hands every item a "task": a sentence. The steps carry only a number, task_index,
    and the sentences are in meta/tasks.parquet. No other check opens that file, so a release
    whose table is gone, or is not the table its steps number into, passed them all. How the
    loader gets from a step to its sentence (lerobot 0.6.1, measured 2026-10-10: every case on
    a release with camera streams and on one without, and the two never differed -- but the
    cases of two step files, which were run without camera streams only):

      - it reads the table only if info.json's total_tasks is above 0
        (dataset_metadata._load_metadata). At 0, below it or absent, no table is read and no
        item can be read (AttributeError on every one); "1", null and [1] stop the opening
        with a TypeError. Any other number above 0 -- 5 beside a table of one row, 1.0, true --
        reads the table and every item, and is handed out as the number of tasks. So it is the
        number as written that is compared, as c7_info_totals compares the other two;
      - the table is the one file, read with pandas (io_utils.load_tasks). If it is not there
        while total_tasks is above 0 the loader takes its copy for incomplete and goes to the
        Hub for the metadata (offline, the release does not open); meta/tasks.jsonl, where v2
        kept the tasks, is not read. A file pandas cannot read stops the opening. A directory
        in its place is read as pandas reads one -- the parquet files in it, an empty one a
        table of no rows -- and so it is here: the same call reads it;
      - the sentence of a row is the label of the frame's index. A table written without
        pandas' index -- a plain Arrow table of the same two columns, or to_parquet(index=
        False) -- reads back with row numbers for labels, and the loader hands out 0 where the
        sentence belongs, raising nothing. A null or a number in the index is handed out as it
        is. What the index is called does not matter (load_tasks names it);
      - a step's task_index is the row's place in the table (dataset_reader: tasks.iloc
        [task_index].name; the streaming reader the same). The table's own task_index column is
        not read on that way: with the rows stored (B, 1) (A, 0), a step numbered 0 gets B. It
        is read where LeRobot looks a sentence's number up (dataset_metadata.get_task_index:
        dataset_writer goes through it for every frame it saves, dataset_tools where it copies
        episodes into another dataset) -- absent, that raises AttributeError, and with a
        sentence listed twice TypeError. So with a column that is not 0, 1, 2, ... the two ways
        do not arrive at one task: the number the column gives a sentence is another
        sentence's place, or no row's. The column is held to integers: it is a place, and
        LeRobot writes it as int64 also when the table grows by a task (measured);
      - a number that is the table's row count or above raises IndexError on that item, and a
        plain ``for`` over the dataset ends there without an error -- the rows before it come
        out and nothing says the rest did not. A negative number counts from the table's end:
        -1 is the last row's sentence, raising nothing. A null fails on that item;
      - a number the integer type declared in the features cannot hold stops the opening: the
        loader casts what is stored to what is declared (128 beside a declared int8, 256 or -1
        beside uint8; 127 and 255 read). -1 beside a signed type is held and read as above.
        That is c7_step_features' to say, for this column as for every other;
      - a step file without the column gives none of its items (declared in the features, the
        loader fills it with nulls and those items fail, the other files' come out; declared
        nowhere, it looks the column up and fails; not declared but in another file, the
        opening stops);
      - a step file with no row group has no step: the loader makes no batch of it and reads
        the rest, whether or not it has the column.

    Read here: the table with pandas, as the loader does; the task_index column of every step
    file the loader reads (_table_files), by itself -- every file whatever --file-limit says.
    The step column is held to an integer type as stored: the loader casts what is stored to
    what the features declare, so 0.0 or "0" stored beside a declared int64 reads (measured),
    but a release carries the column as integers and nothing else is taken for a place here.
    Whether the feature declares one integer a step is c7_table_files' to say, and whether the
    values fit that integer c7_step_features'. A value that does not fit is not also said here
    as no row of the table: the loader never gets as far as the table with it. A file whose
    column cannot be read is left with a SKIP, as one that is no parquet is.

    Each kind of defect has one sentence, and a finding's numbers are in its location: a table
    cut short puts every step of a task off, and those fold into one line.

    Not looked at:
      - the episode table's ``tasks`` column, the sentences of an episode. The loader does not
        go by it to read an item (measured: naming a sentence the table lacks, or absent, the
        same items) -- it is what LeRobot's own tools split and merge by;
      - whether the sentence is the one the episode's scenario declares, and a task no step
        has;
      - task_index in meta/stats.json and in the episodes' statistics;
      - the validator's pandas is not the loader's (a release is read by whichever the
        recipient has); the two it was measured with, 3.0.5 here and the loader's 2.3.3, read
        every case alike.
    """
    # What was held to what. A number that is not here was not compared.
    held: dict[str, int] = {}
    rep.stats["task_table"] = held
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        rep.add("C7", SKIP, f"pyarrow 가 없어 과제 표({TASKS_PATH})와 스텝의 {TASK_INDEX} 를 대조하지 못함")
        return

    info = ds.info()                    # None: no info.json, or one that is no JSON object -- C1's finding
    path = ds.root / TASKS_PATH
    tasks: pd.DataFrame | None = None
    if not path.exists():               # also a link to nothing
        rep.add("C7", ERROR,
                "LeRobot 이 읽는 과제 표가 없음 — 적재기는 스텝의 과제 문장을 이 파일에서 찾는다. 없으면 이 "
                "사본만으로는 항목을 내주지 못한다(total_tasks 가 0 보다 크면 여는 단계에서 Hub 로 메타를 다시 "
                "받으러 간다)", TASKS_PATH)
    else:
        try:
            tasks = pd.read_parquet(path)
        except Exception as e:  # noqa: BLE001 -- whatever pandas raises on it, the loader meets too
            rep.add("C7", ERROR,
                    "과제 표를 읽지 못함 — LeRobot 은 이 파일을 pandas 로 읽어 스텝의 과제 문장을 찾는다. "
                    "읽히지 않으면 항목을 내주지 못한다",
                    f"{TASKS_PATH} · {type(e).__name__}: {' '.join(str(e).split())[:120]}")

    rows = 0
    if tasks is not None:
        rows = held["tasks"] = len(tasks)
        if info is not None:
            stated = info.get("total_tasks", 0)
            if not (type(stated) is int and stated == rows):        # not bool: True == 1
                shown = _as_written(stated) if "total_tasks" in info else "없음(LeRobot 은 0 으로 읽음)"
                rep.add("C7", ERROR,
                        "info.json 의 total_tasks 가 과제 표의 행 수와 다름 — LeRobot 은 표를 세지 않고 이 수를 "
                        "과제 수로 내주며, 0 보다 크지 않으면 과제 표를 읽지 않아 항목을 하나도 내주지 못한다",
                        f"meta/info.json · total_tasks {shown}, 과제 표 {rows}행")

        labels = list(tasks.index)
        not_sentences = [(k, label) for k, label in enumerate(labels) if not isinstance(label, str)]
        if not_sentences:
            k, label = not_sentences[0]
            rep.add("C7", ERROR,
                    "과제 표의 인덱스가 과제 문장이 아님 — LeRobot 은 pandas 로 읽은 표의 행 인덱스를 과제 문장으로 "
                    "내준다. 문장을 인덱스로 저장하지 않은 표(pandas 의 인덱스 없이 쓴 parquet)에서는 문장 자리에 "
                    "행 번호가 나온다",
                    f"{TASKS_PATH} · {k}번째 행의 인덱스 {label!r} (문자열이 아닌 행 {len(not_sentences)}개)")
        twice = [label for label, n in Counter(x for x in labels if isinstance(x, str)).items() if n > 1]
        if twice:
            rep.add("C7", ERROR,
                    "과제 표에 같은 문장이 둘 이상 — LeRobot 은 문장으로 과제 번호를 찾다 실패한다(get_task_index). "
                    "스텝의 과제는 행의 자리로 읽으므로 이것만으로 항목이 막히지는 않는다",
                    f"{TASKS_PATH} · {twice[0][:80]!r} (겹친 문장 {len(twice)}개)")

        if rows and TASK_INDEX not in tasks.columns:
            rep.add("C7", ERROR,
                    f"과제 표에 {TASK_INDEX} 열이 없음 — LeRobot 은 문장으로 과제 번호를 찾을 때 이 열을 "
                    "읽는다(get_task_index). 스텝의 과제는 행의 자리로 읽으므로 이것만으로 항목이 막히지는 않는다",
                    TASKS_PATH)
        elif rows:                      # with no row there is no number, and pandas types the empty column float64
            numbers = _integer_column(tasks[TASK_INDEX])
            if numbers is None:
                rep.add("C7", ERROR,
                        f"과제 표의 {TASK_INDEX} 열이 정수 열이 아님 — 이 열은 행의 자리(0, 1, 2, …)이고 LeRobot 은 "
                        "정수로 적는다", f"{TASKS_PATH} · {tasks[TASK_INDEX].dtype}")
            else:
                off = np.flatnonzero(numbers != np.arange(rows))
                if off.size:
                    k = int(off[0])
                    rep.add("C7", ERROR,
                            f"과제 표의 {TASK_INDEX} 열이 행의 자리와 다름 — LeRobot 은 스텝의 {TASK_INDEX} 를 표의 "
                            "행 자리로 읽고 이 열은 읽지 않는다. 문장으로 과제 번호를 찾을 때는 이 열을 읽으므로"
                            "(get_task_index) 두 길이 같은 과제에 닿지 않는다 — 열이 적은 번호는 다른 문장의 "
                            "자리이거나 표에 없는 자리다",
                            f"{TASKS_PATH} · {k}번째 행이 {TASK_INDEX} {int(numbers[k])} (어긋난 행 {off.size}개)")

    # The integer type the features declare the column as, where they declare one integer a step
    # (c7_table_files says when they do not): the loader casts what is stored to it, and a value
    # that does not go in stops it before any table is read (c7_step_features says those).
    features = info.get("features") if info is not None else None
    feature = features.get(TASK_INDEX) if isinstance(features, dict) else None
    declared: tuple[str, int, int] | None = None
    if _declares_one_integer(feature):
        bounds = np.iinfo(feature["dtype"])
        declared = (feature["dtype"], int(bounds.min), int(bounds.max))

    # The steps: every file the loader reads, one column of each.
    loaded, _ = _table_files(ds.root / DATA_DIR)
    compared = outside = 0
    first: tuple[str, int, str] | None = None
    for f in loaded:
        rel = str(f.relative_to(ds.root))
        try:
            file = pq.ParquetFile(f)
            names = list(file.schema_arrow.names)
            footer = file.metadata
            has_rows = any(footer.row_group(k).num_rows for k in range(footer.num_row_groups))
            column = (file.read(columns=[TASK_INDEX]).column(0)
                      if has_rows and names.count(TASK_INDEX) == 1 else None)
        except Exception:  # noqa: BLE001 -- no parquet, or a column that cannot be read: the readers' to say (C1)
            rep.add("C7", SKIP,
                    f"스텝 표에 parquet 으로 읽히지 않는 파일이 있어 그 파일의 {TASK_INDEX} 를 과제 표와 대조하지 못함",
                    rel)
            continue
        if not has_rows:
            continue                    # no step in it; a first row group of no rows is c7_table_files' finding
        if names.count(TASK_INDEX) > 1:
            continue                    # two columns of one name: c7_table_files' finding
        if column is None:
            rep.add("C7", ERROR,
                    f"스텝 표에 {TASK_INDEX} 열이 없음 — LeRobot 은 스텝마다 이 열의 값으로 과제 문장을 찾아, "
                    "이 파일의 스텝은 항목으로 내주지 못한다", rel)
            continue
        if not pa.types.is_integer(column.type):
            rep.add("C7", ERROR,
                    f"스텝 표의 {TASK_INDEX} 열이 정수 열이 아니어서 과제 표의 행과 대조하지 못함",
                    f"{rel} · {column.type}")
            continue
        absent = column.is_null().to_numpy()
        values = column.fill_null(0).to_numpy()
        # A value the declared type cannot hold stops the loader before it reads any table. (A
        # null stands here as 0, which every integer type holds.)
        no_fit = _beyond(values, *declared[1:]) if declared else np.zeros(len(values), dtype=bool)
        if tasks is None:
            continue                    # no table to hold them to: said above
        compared += len(column)
        off = np.flatnonzero((absent | (values < 0) | (values >= rows)) & ~no_fit)
        if off.size:
            outside += int(off.size)
            if first is None:
                k = int(off[0])
                first = (rel, k, "null" if absent[k] else str(int(values[k])))
    if tasks is not None:
        held["steps"] = compared
    if first is not None:
        rel, k, shown = first
        rep.add("C7", ERROR,
                f"스텝의 {TASK_INDEX} 가 과제 표의 행이 아님 — LeRobot 은 이 값을 과제 표의 행 자리로 읽는다. 표의 "
                "행 수 이상이면 그 항목은 나오지 않고(표를 읽은 적재기는 IndexError 를 내며, for 로 도는 쪽은 "
                "거기서 오류 없이 끝난다), 음수는 표의 끝에서부터 센다(-1 은 마지막 행의 문장)",
                f"{rel} · {k}번째 행의 {TASK_INDEX} {shown} (과제 표 {rows}행 · 벗어난 스텝 {outside}개)")


def c7_data_spans(ds: Dataset, frames: list[tuple[Path, pd.DataFrame]],
                  eps: pd.DataFrame | None, rep: Report) -> None:
    """C7 — the rows meta/episodes points an episode at are that episode's rows.

    LeRobot v3.0 keeps several episodes to a parquet and finds an episode's steps by pointers in
    meta/episodes (DATA_SPAN_COLUMNS), not by the rows' own episode_index. Every other check here
    joins rows to episodes by that episode_index, so a table that sends a loader to another
    episode's rows passed them all -- the data side of what c7_camera_alignment holds the video
    spans to. Read as the loader reads it:

      - a table is the files it loads (TABLE_FILE_PARTS), in name order, rows as stored. A
        parquet elsewhere under data/ or meta/episodes/ is not part of it -- counted in, a table
        that is whole only by that file would pass for whole. c7_table_files says such files;
        here the step table is read without them, and the episode table (Dataset.episodes, any
        depth) is not compared at all, as its rows no longer are the rows a loader has. Nor is
        it when there is no episode parquet where a loader reads and the table came from the
        validator's own fallback reader;
      - episode k's pointers are the k-th row of the episode table, whatever that row's
        episode_index says;
      - dataset_from_index..dataset_to_index are places in the step table. ``index`` has to say
        the same -- a recipient who picks rows by it gets the rows a loader gets -- and the rows
        of a stretch are frame_index 0, 1, 2, ... (c7_timeseries counts from a group's first
        value, so a stretch starting at 5 passes it);
      - data/chunk_index · data/file_index, through info.json's data_path, name the file that
        holds the stretch.

    The pointers and the three step columns are integer columns in a release and are held to
    that: LeRobot uses the values as they are, and 3.0 or "3" is not a place.

    Each kind of defect has one sentence, and a finding's numbers are in its location: a row
    dropped early puts every later episode off, and those fold into one line of the report
    instead of burying the rest.

    Not looked at:
      - whether info.json counts the tables right (c7_info_totals);
      - whether a row holds that step's values. The pointers say whose row it is; what is in it
        is C8's and C9's;
      - under --file-limit, an episode whose stretch lies in a file that was not read (said);
      - the video spans (c7_camera_alignment).
    """
    if eps is None or not frames:
        return                  # no episode table is C1's finding; no step file C1's and C7's

    def loaded_by_lerobot(path: Path) -> bool:
        return len(path.parts[1:]) == TABLE_FILE_PARTS         # path: data/..., under the root

    # Which files are the tables is c7_table_files' to say (it runs first, and whether or not a
    # step was read); here that only decides what is compared.
    table = [(path, df) for path, df in frames if loaded_by_lerobot(path)]
    if not table:
        return

    # The step table's three columns of place, every loaded file together.
    readable = True
    for path, df in table:
        absent = [c for c in STEP_PLACE_COLUMNS if c not in df.columns]
        if absent:
            rep.add("C7", ERROR, f"스텝 표에 {absent} 열이 없어 에피소드가 가리키는 행을 대조하지 못함",
                    str(path))
            readable = False
        for name in STEP_PLACE_COLUMNS:
            if name in df.columns and _integer_column(df[name]) is None:
                rep.add("C7", ERROR,
                        f"스텝 표의 {name} 열이 정수 열이 아니어서({df[name].dtype}) 에피소드가 가리키는 "
                        "행을 대조하지 못함", str(path))
                readable = False
    file_of = np.repeat(np.arange(len(table)), [len(df) for _, df in table])
    n = len(file_of)
    steps: dict[str, np.ndarray] = {}
    if readable:
        steps = {name: np.concatenate([_integer_column(df[name]) for _, df in table])
                 for name in STEP_PLACE_COLUMNS}
        out_of_place = np.flatnonzero(steps["index"] != np.arange(n))
        if out_of_place.size:
            i = int(out_of_place[0])
            rep.add("C7", ERROR,
                    f"index 열이 행의 자리와 다름 — {i}번째 행이 {int(steps['index'][i])} (어긋난 행 "
                    f"{out_of_place.size}개). LeRobot 은 행을 자리로 읽으므로 index 로 고른 행은 다른 "
                    "행이다", str(table[file_of[i]][0]))

    # The episode table's pointers.
    # Not the table a loader has -- a file of it where the loader does not look, or none where
    # it does and ``eps`` the fallback's: said by c7_table_files, and nothing to compare here.
    loaded_tables, stray_tables = _table_files(ds.meta / EPISODES_DIR)
    if stray_tables or not loaded_tables:
        return
    absent = [c for c in EPISODE_POINTER_COLUMNS if c not in eps.columns]
    if absent:
        rep.add("C7", ERROR,
                f"meta/episodes 에 {absent} 열이 없어 에피소드의 행을 찾을 수 없음 — LeRobot 은 에피소드의 "
                "스텝을 이 열로 찾는다", "meta/episodes")
        return
    not_integers = [name for name in DATA_SPAN_COLUMNS if _integer_column(eps[name]) is None]
    for name in not_integers:
        rep.add("C7", ERROR,
                f"meta/episodes 의 {name} 열이 정수 열이 아님 ({eps[name].dtype}) — LeRobot 은 이 값을 "
                "그대로 자리·파일 번호로 쓴다", "meta/episodes")
    if not readable or not_integers:
        return
    pointers = {name: _integer_column(eps[name]) for name in DATA_SPAN_COLUMNS}
    misplaced = [(k, raw) for k, raw in enumerate(eps["episode_index"].tolist())
                 if _index_or_none(raw) != k]
    if misplaced:
        k, raw = misplaced[0]
        rep.add("C7", ERROR,
                f"meta/episodes 의 {k}번째 행이 episode_index {raw!r} (어긋난 행 {len(misplaced)}개) — "
                "LeRobot 은 k번 에피소드의 포인터를 k번째 행에서 읽는다", "meta/episodes")
        return

    info = ds.info()
    template = info["data_path"] if isinstance(info, dict) and "data_path" in info else DATA_PATH

    def named_file(chunk: int, file: int) -> str | None:
        # The template is the release's own text and the numbers are its own too: whatever
        # filling it in raises is a finding. Caught by kind, "{file_index:c}" given 2**63
        # (OverflowError) would end the validator with exit 3.
        try:
            return template.format(chunk_index=chunk, file_index=file)
        except Exception:  # noqa: BLE001
            return None

    def is_there(named: str) -> bool:
        there = _inside(ds.root, named)           # None: a data_path that leaves the release
        try:
            return there is not None and there.is_file()
        except OSError:                           # a name the file system refuses (too long) names no file
            return False

    names_files = isinstance(template, str) and bool(template) and named_file(0, 0) is not None
    if not names_files:
        rep.add("C7", ERROR, f"info.json 의 data_path 를 파일 이름으로 풀 수 없음: {template!r}",
                "meta/info.json")
    loaded = {path: j for j, (path, _) in enumerate(table)}

    claimed = np.zeros(n, dtype=bool)
    compared = rows = unread = 0
    unplaced = False            # an episode whose stretch could not be placed: its rows are not orphans
    for k, label in enumerate(_episode_labels(eps)):
        a, b = int(pointers["dataset_from_index"][k]), int(pointers["dataset_to_index"][k])
        if 0 <= a < b and b > n and ds.frames_truncated:
            unread += 1
            claimed[a:n] = True
            continue
        if not 0 <= a < b <= n:
            rep.add("C7", ERROR, "dataset_from_index..dataset_to_index 가 스텝 표 안의 구간이 아님",
                    f"{label} · {a}..{b} (스텝 표 {n}행)")
            unplaced = True
            continue
        compared += 1
        rows += b - a
        claimed[a:b] = True
        length = int(pointers["length"][k])
        if b - a != length:
            rep.add("C7", ERROR, "length 가 가리키는 구간의 행 수와 다름",
                    f"{label} · length {length}, {a}..{b} 는 {b - a}행")
        owners = steps["episode_index"][a:b]
        if (owners != k).any():
            rep.add("C7", ERROR,
                    "가리키는 행이 이 에피소드의 것이 아님 — LeRobot 은 이 구간을 이 에피소드로 내준다",
                    f"{label} · 행 {a}..{b} 의 episode_index {np.unique(owners)[:5].tolist()}")
        else:
            frame = steps["frame_index"][a:b]
            off = np.flatnonzero(frame != np.arange(b - a))
            if off.size:
                rep.add("C7", ERROR, "가리키는 행의 frame_index 가 0 부터 차례로 이어지지 않음",
                        f"{label} · 행 {a}..{b} 의 {int(off[0])}번째가 {int(frame[off[0]])}")
        if not names_files:
            continue
        chunk, file = int(pointers["data/chunk_index"][k]), int(pointers["data/file_index"][k])
        named = named_file(chunk, file)
        if named is None:
            # Said per episode: a template that fills in for (0, 0) can still fail on these
            # numbers, and the loader fails on them too (get_data_file_path).
            rep.add("C7", ERROR,
                    "info.json 의 data_path 를 이 에피소드의 data/chunk_index·data/file_index 로 풀 수 없음",
                    f"{label} · data/chunk_index {chunk}, data/file_index {file}")
            continue
        held = np.unique(file_of[a:b])
        j = loaded.get(Path(named))
        if j is None and not is_there(named):
            rep.add("C7", ERROR, "data/chunk_index·data/file_index 가 가리키는 파일이 없음",
                    f"{label} · {named}")
        elif held.tolist() != [j]:
            # In a file that is there but was not read (--file-limit) or is not one the loader
            # reads, too: the rows are not in it either way.
            rep.add("C7", ERROR,
                    "행이 data/chunk_index·data/file_index 가 가리키는 파일에 있지 않음 — 에피소드를 골라 "
                    "받으면 LeRobot 은 그 파일만 받는다",
                    f"{label} · 가리킨 파일 {named}, 행이 있는 파일 {[str(table[i][0]) for i in held[:3]]}")

    orphans = np.flatnonzero(~claimed)
    if orphans.size and not unplaced:
        first = int(orphans[0])
        rep.add("C7", ERROR,
                f"어느 에피소드도 가리키지 않는 스텝 행 {orphans.size}개 — LeRobot 은 이 행도 내주지만 "
                "어느 에피소드의 구간에도 들지 않는다", f"{table[file_of[first]][0]} · 첫 행 {first}")
    if unread:
        rep.add("C7", INFO,
                f"읽지 않은 파일에 구간이 있는 에피소드 {unread}개는 가리키는 행을 대조하지 않음 (--file-limit)")
    # What was looked at, beside what was said about it.
    rep.stats["data_spans"] = {"episodes": compared, "rows": rows}


def c7_camera_alignment(ds: Dataset, frames: list[tuple[Path, pd.DataFrame]],
                        eps: pd.DataFrame | None, rep: Report) -> None:
    """C7 — a camera distribution's streams and packs line up with the steps, per episode.

    Stream: the span meta/episodes records for each camera, times the dataset fps, is the
    episode's step count, to well under a frame (VIDEO_FRAME_TOLERANCE). That is the span
    LeRobot wrote down -- the pictures themselves are not decoded or counted.
    Pack, when the canon declares one (C1 reports packs it does not declare):
    meta/packs.json lists the episode, its ``frames`` is the number of steps kept at
    the canon's every_n_steps, and the pack's own step_index is arange(0, steps, k) with
    depth and mask one row per kept step at the pack size and dtype. Read from the npz
    headers and the one step_index member: a ~230 MB pack costs milliseconds.

    The step count is the episode's rows in data/. An episode none of whose rows were read
    (--file-limit; or rows missing, which C7 reports on its own) falls back to meta/episodes
    ``length`` when there is one.
    """
    if not ds.ships_cameras() or eps is None:
        return
    if "episode_index" not in eps.columns:
        rep.add("C7", ERROR, "meta/episodes 에 episode_index 가 없어 영상·팩을 에피소드에 잇지 못함",
                "meta/episodes")
        return
    counts: Counter = Counter()
    for _, df in frames:
        if "episode_index" in df.columns:
            index = pd.to_numeric(df["episode_index"], errors="coerce").dropna()
            # Not int(k): a float column holding inf survives dropna and int(inf) raised
            # OverflowError (exit 3). Such rows join no episode; C7's timeline checks see them.
            for k, v in index.value_counts().items():
                if (i := _index_or_none(k)) is not None:
                    counts[i] += int(v)
    lengths = eps["length"].tolist() if "length" in eps.columns else [None] * len(eps)
    episodes: list[tuple[int, str, int, int | None]] = []   # (row, label, episode_index, steps)
    for row, (label, raw, length) in enumerate(
            zip(_episode_labels(eps), eps["episode_index"].tolist(), lengths)):
        idx = _index_or_none(raw)
        if idx is None:
            rep.add("C7", ERROR, f"episode_index 가 0 이상의 정수가 아님: {raw!r}", label)
            continue
        steps = counts.get(idx) or _index_or_none(length)
        if steps is None:
            # Said, not skipped: without it C7 printed OK over streams and a pack it never
            # compared (a --file-limit run on a release without length).
            rep.add("C7", ERROR,
                    "이 에피소드의 스텝 수를 알 수 없어(읽은 데이터 행이 없고 meta/episodes 에 length 도 "
                    "없음) 영상·팩 정렬을 대조하지 못함", label)
        episodes.append((row, label, idx, steps))

    stats = rep.stats.setdefault("camera_media", {})
    stats["video_spans"], stats["video_file_layouts"] = _c7_video_spans(ds, eps, episodes, rep)
    setup, declared = camera_packs_of(ds)
    if declared is False:
        # No pack declared, none to align; one that is there anyway is C1's finding.
        stats["packs_aligned"] = 0
        return
    manifest, _ = ds.packs_manifest()                # its problems are C1's to report
    if setup is None:
        rep.add("C7", ERROR, "meta/cameras.json 을 읽지 못해 팩의 스텝 간격·크기를 알 수 없음 — "
                             "팩 정렬을 대조하지 못함 (사유는 C1)", "meta/cameras.json")
        return
    if not isinstance(manifest, dict):
        rep.add("C7", ERROR, "meta/packs.json 을 읽지 못해 팩 정렬을 대조하지 못함 (사유는 C1)",
                "meta/packs.json")
        return
    aligned = 0
    for _row, label, idx, steps in episodes:
        entry = manifest.get(str(idx))
        if entry is None:
            rep.add("C7", ERROR,
                    f"meta/packs.json 에 이 에피소드(episode_index={idx})의 팩이 없음 — 깊이·마스크는 "
                    "에피소드마다 하나씩 실린다", label)
            continue
        if not isinstance(entry, dict):
            rep.add("C7", ERROR,
                    f"meta/packs.json 의 이 에피소드 항목이 객체가 아니어서({type(entry).__name__}) 팩 "
                    "정렬을 대조하지 못함 (사유는 C1)", label)
            continue
        if steps is not None:
            aligned += _c7_pack(ds, setup, entry, steps, label, rep)
    stats["packs_aligned"] = aligned
    known = {str(idx) for _, _, idx, _ in episodes}
    extra = sorted(set(manifest) - known)
    if extra:
        rep.add("C7", ERROR,
                f"meta/packs.json 에 에피소드 메타에 없는 팩 {len(extra)}개: {extra[:3]} — 어느 "
                "에피소드의 스텝인지 알 수 없다", "meta/packs.json")


def _c7_video_spans(ds: Dataset, eps: pd.DataFrame,
                    episodes: list[tuple[int, str, int, int | None]],
                    rep: Report) -> tuple[int, int]:
    """Each camera's recorded span, in frames, against the episode's steps, and the spans that
    share each of its files against the layout LeRobot writes (_c7_video_layout). Returns how
    many spans were compared and how many camera files were laid out."""
    info = ds.info()
    fps = numeric_or_none(info.get("fps")) if isinstance(info, dict) else None
    if fps is None or fps <= 0:
        rep.add("C7", ERROR, "info.json 에 fps 가 없어 영상 구간을 프레임 수로 바꿀 수 없음",
                "meta/info.json")
        return 0, 0
    compared = laid_out = 0
    for key in camera_video_fields():
        cols = (f"videos/{key}/from_timestamp", f"videos/{key}/to_timestamp")
        absent = [c for c in cols if c not in eps.columns]
        if absent:
            rep.add("C7", ERROR,
                    f"meta/episodes 에 영상 구간 열 {absent} 이 없어 {key} 의 프레임 수를 알 수 없음",
                    "meta/episodes")
            continue
        starts, ends = eps[cols[0]].tolist(), eps[cols[1]].tolist()
        for i, label, _idx, steps in episodes:
            t0, t1 = numeric_or_none(starts[i]), numeric_or_none(ends[i])
            if t0 is None or t1 is None:
                rep.add("C7", ERROR,
                        f"{key} 의 영상 구간이 수치가 아님: {starts[i]!r} ~ {ends[i]!r}", label)
                continue
            if steps is None:
                continue
            compared += 1
            span = (t1 - t0) * fps
            if abs(span - steps) > VIDEO_FRAME_TOLERANCE:
                rep.add("C7", ERROR,
                        f"{key} 의 영상 구간 {t0:.6g}~{t1:.6g} s 는 {fps:g} fps 로 {span:.3f} 프레임인데 "
                        f"이 에피소드의 스텝은 {steps}개 — 영상 프레임과 스텝이 1:1 이 아니다 "
                        "(meta/episodes 가 적은 구간을 센 것이고 영상은 디코드하지 않는다)", label)
        laid_out += _c7_video_layout(eps, key, fps, rep)
    return compared, laid_out


def _c7_video_layout(eps: pd.DataFrame, key: str, fps: float, rep: Report) -> int:
    """The spans that share one of ``key``'s files, against the layout LeRobot writes. Returns
    how many of the camera's files were looked at.

    LeRobot 0.6.1 appends each episode's stream to the camera's current file and records its
    span from where the file's previous episode ended -- 0 s in a new file
    (dataset_writer._save_episode_video: from_timestamp is the previous to_timestamp); its own
    editing tools keep that layout (dataset_tools re-encodes the kept episodes from 0 s,
    aggregate offsets whole files), and a converted S1d dataset holds it to the bit. So within one
    (chunk_index, file_index), in episode order, each span starts where the one before it
    ended, and the first at 0 s, to well under a frame (VIDEO_FRAME_TOLERANCE). Two spans that
    overlap point two episodes at the same pictures -- episode 1 given
    episode 0's spans passes a C7 that holds each span only to its own length -- and a gap, a
    shifted file or two swapped episodes of one length point an episode at pictures that are
    not its own. Each overlap is said; the order of a file is said once, at its first break.

    Read from meta/episodes alone: whether the file holds pictures up to its last span's end
    needs the file and is not looked at. A row whose span or episode_index cannot be read (said
    by the duration check and C7) leaves its file's order unchecked -- overlaps among the other
    rows are still reported; one whose file numbers cannot be read is C1's."""
    cols = [f"videos/{key}/{c}" for c in ("chunk_index", "file_index", "from_timestamp",
                                         "to_timestamp")]
    if "episode_index" not in eps.columns or any(c not in eps.columns for c in cols):
        return 0
    files: dict[tuple[int, int], list[tuple[int | None, str, float, float]]] = {}
    unordered: set[tuple[int, int]] = set()
    for label, raw, chunk, file, start, end in zip(
            _episode_labels(eps), eps["episode_index"].tolist(), *(eps[c].tolist() for c in cols)):
        c, f = _index_or_none(chunk), _index_or_none(file)
        if c is None or f is None:
            continue
        t0, t1, idx = numeric_or_none(start), numeric_or_none(end), _index_or_none(raw)
        if t0 is None or t1 is None or idx is None:
            unordered.add((c, f))
        if t0 is None or t1 is None:
            continue
        files.setdefault((c, f), []).append((idx, label, t0, t1))
    for (c, f), spans in sorted(files.items()):
        where = f"chunk_index {c} · file_index {f}"
        overlapped = False
        reach = None                       # the span reaching furthest so far, in time order
        for span in sorted(spans, key=lambda s: (s[2], s[3], -1 if s[0] is None else s[0])):
            if reach is not None and (reach[3] - span[2]) * fps > VIDEO_FRAME_TOLERANCE:
                overlapped = True
                rep.add("C7", ERROR,
                        f"{key} 의 영상 구간 {span[2]:.6g}~{span[3]:.6g} s 가 {reach[1]} 의 구간 "
                        f"{reach[2]:.6g}~{reach[3]:.6g} s 와 {(min(reach[3], span[3]) - span[2]) * fps:.3f} "
                        f"프레임 겹침 — 같은 파일({where})이라 두 에피소드가 같은 그림을 가리킨다", span[1])
            if reach is None or span[3] > reach[3]:
                reach = span
        if overlapped or (c, f) in unordered:
            continue
        end, before = 0.0, None
        for idx, label, t0, t1 in sorted(spans, key=lambda s: s[0]):
            gap = (t0 - end) * fps
            if abs(gap) > VIDEO_FRAME_TOLERANCE:
                said = (f"0 s 에서 시작하지 않음 ({gap:+.3f} 프레임) — 파일({where})의 첫 에피소드다"
                        if before is None else
                        f"앞 에피소드 {before} 의 구간이 끝난 {end:.6g} s 에서 시작하지 않음 "
                        f"({gap:+.3f} 프레임) — 같은 파일({where})")
                rep.add("C7", ERROR,
                        f"{key} 의 영상 구간 {t0:.6g}~{t1:.6g} s 가 {said}. LeRobot 은 한 파일에 에피소드를 "
                        "순서대로 0 s 부터 틈 없이 이어 붙인다 — 어긋나면 에피소드가 제 것이 아닌 그림을 "
                        "가리킨다", label)
                break
            end, before = t1, label
    return len(files)


def _c7_pack(ds: Dataset, setup: CameraSetup, entry: dict, steps: int, label: str,
             rep: Report) -> int:
    """One episode's pack against its steps; 1 if the pack could be read."""
    k = setup.every_n_steps
    rows = len(range(0, steps, k))
    frames = entry.get("frames")
    if not isinstance(frames, int) or isinstance(frames, bool):
        rep.add("C7", ERROR, f"meta/packs.json 의 frames 가 정수가 아님: {frames!r}", label)
    elif frames != rows:
        rep.add("C7", ERROR,
                f"meta/packs.json 의 frames {frames} 가 스텝 {steps}개를 {k} 스텝마다 남긴 {rows} 와 "
                "다름", label)
    rel = entry.get("file")
    path = _inside(ds.root, rel)
    if path is None:
        # Said here too: a --checks C7 run printed OK over a pack it never opened.
        rep.add("C7", ERROR, f"팩 경로 {rel!r} 가 배포본 안의 상대 경로가 아니어서 스텝 정렬을 대조하지 "
                             "못함 (사유는 C1)", label)
        return 0
    if not path.is_file():
        rep.add("C7", ERROR, f"팩 {rel} 이 배포본에 없어 스텝 정렬을 대조하지 못함", label)
        return 0
    # Any exception while reading the pack is a finding about the pack, never a validator
    # crash -- a crash loses the whole report, the C1 SHA-256 line that names the damage
    # included. A single flipped byte in a pack raised seven types on this path (13,972
    # corruptions of a fixture pack, 2026-09-28): BadZipFile, zlib.error, ValueError,
    # tokenize.TokenError, NotImplementedError, OSError, RuntimeError -- four of them past the
    # tuple this block first had. The list is zipfile's and numpy's, not this file's; only
    # their reads of the pack sit inside the try.
    try:
        layout = npz_layout(path)
        header = layout.get(PACK_STEP_INDEX)
        step_index = None
        # Inflate step_index only when its header already has the one shape and kind it can
        # have: a damaged header claiming 10^12 rows would be allocated before failing.
        if header is not None and header[0] == (rows,) and header[1].kind in "iu":
            with np.load(path) as z:
                step_index = z[PACK_STEP_INDEX]
    except Exception as e:  # noqa: BLE001
        kind = type(e).__name__                     # zlib.error's own name is just "error"
        if type(e).__module__ != "builtins":
            kind = f"{type(e).__module__}.{kind}"
        rep.add("C7", ERROR, f"팩 {rel} 을 열지 못해 스텝 정렬을 대조하지 못함 — {kind}: {e}", label)
        return 0
    if set(layout) != set(PACK_MEMBERS):
        rep.add("C7", ERROR, f"팩 {rel} 의 배열이 {sorted(layout)} — {list(PACK_MEMBERS)} 여야 한다",
                label)
    if header is not None and (step_index is None
                               or not np.array_equal(step_index, np.arange(0, steps, k))):
        rep.add("C7", ERROR,
                f"팩 {rel} 의 {PACK_STEP_INDEX} 가 arange(0, {steps}, {k}) 가 아님 — 팩의 행이 "
                "parquet 의 frame_index 와 맞지 않는다", label)
    width, height = setup.pack_size
    for member, dtype in setup.pack_dtypes.items():
        if member not in layout:
            continue
        shape, got = layout[member]
        if shape != (rows, height, width) or got != dtype:
            rep.add("C7", ERROR,
                    f"팩 {rel} 의 {member} 가 {shape} {got} — ({rows}, {height}, {width}) {dtype} "
                    "여야 한다 (남긴 스텝 수 · meta/cameras.json 의 팩 크기와 형)", label)
    return 1


def known_rule_ids(ds: Dataset) -> set[str]:
    """rules.json 이 실제로 정의한 규칙 id."""
    rules = ds.rules()
    if isinstance(rules, dict) and isinstance(rules.get("rules"), list):
        return {str(r.get("rule_id")) for r in rules["rules"] if isinstance(r, dict)}
    if isinstance(rules, list):
        return {str(r.get("rule_id")) for r in rules if isinstance(r, dict)}
    return set()


def scenario_index(ds: Dataset) -> dict[str, dict] | None:
    """scenario_id → 항목. 중복이나 빈 id 가 있으면 None (매핑을 믿을 수 없음)."""
    sc = ds.scenarios()
    if isinstance(sc, dict):
        sc = sc.get("scenarios")
    if not isinstance(sc, list) or not sc:
        return None

    by_id: dict[str, dict] = {}
    for s in sc:
        if not isinstance(s, dict):
            return None
        sid = s.get("scenario_id")
        # 중복을 허용하면 나중 항목이 앞 항목을 조용히 덮는다. 어느 쪽이 정본인지
        # 파일만 보고는 알 수 없으므로 매핑 전체를 믿지 않는다.
        if not isinstance(sid, str) or not sid.strip() or sid in by_id:
            return None
        by_id[sid] = s
    return by_id


def declared_selectors(entry: dict) -> list[str] | None:
    """시나리오가 선언한 규칙 selector. 비었거나 형식이 어긋나면 None."""
    declared = entry.get("rules")
    if not isinstance(declared, list) or not declared:
        return None
    if not all(isinstance(r, str) and r.strip() for r in declared):
        return None
    return declared


def selector_resolves(sel: str, known: set[str]) -> bool:
    """selector 가 실재하는 규칙을 하나라도 가리키는지."""
    if sel.endswith("*"):
        return any(k.startswith(sel[:-1]) for k in known)
    return sel in known


def applicable_rule_ids(ds: Dataset, eps: pd.DataFrame | None) -> set[str] | None:
    """이 데이터셋에 실제로 적용되는 규칙 id 집합.

    규칙마다 적용 범위가 다르다. 파지물이 없는 보행 시나리오 데이터에 낙하 규칙의 입력을
    요구하면 통과할 수 없다. scenarios.json 이 시나리오별 적용 규칙을 선언하므로 그것을
    쓰되, 매핑을 믿을 수 없으면 None 을 돌려 보수적으로 전 규칙을 검사하게 한다.

    "믿을 수 없다" 에는 선언이 비어 있거나 실재하지 않는 규칙을 가리키는 경우가 포함된다.
    그 경우 적용 집합이 공집합이 되어 C8 이 아무 입력도 요구하지 않게 되는데, 이는
    검사를 건너뛰는 것과 같아 보수적 폴백의 취지와 정반대다.
    """
    if eps is None or "scenario_id" not in eps.columns:
        return None

    by_id = scenario_index(ds)
    if by_id is None:
        return None

    col = eps["scenario_id"]
    # 결측 id 는 그 에피소드에 어떤 규칙이 적용되는지 알 수 없다는 뜻이다. 빼고 계산하면
    # 그 에피소드만 검사에서 조용히 빠진다.
    if col.isna().any():
        return None
    used = {str(s) for s in col.unique()}
    if not used or any(not u.strip() for u in used) or not used <= set(by_id):
        return None

    # 규칙 정본을 읽을 수 없으면 selector 가 실재하는지 판정할 방법이 없다. 예전에는
    # `if known` 으로 검사를 건너뛰었는데, 그러면 rules 가 빈 배열일 때 오타 selector 가
    # 그대로 적용 집합이 되어 C8 이 아무 입력도 요구하지 않았다 (검사를 건너뛴 것과 같다).
    known = known_rule_ids(ds)
    if not known:
        return None

    out: set[str] = set()
    for sid in used:
        declared = declared_selectors(by_id[sid])
        if declared is None:
            return None
        if not all(selector_resolves(sel, known) for sel in declared):
            return None
        out |= set(declared)
    return out


def _required_fields_of(entry: object) -> set[str]:
    req = entry.get("required_fields") if isinstance(entry, dict) else None
    if not isinstance(req, list):
        return set()
    return {r for r in req if isinstance(r, str) and r.strip()}


def scenario_required_fields(ds: Dataset, eps: pd.DataFrame | None) -> set[str]:
    """시나리오가 선언한 필수 스텝 필드.

    매핑을 믿을 수 없을 때의 방침은 applicable_rule_ids() 와 같아야 한다. 예전에는 결측
    scenario_id 를 dropna() 로 버려, 그 에피소드의 시나리오에만 선언된 필드가 검사에서
    조용히 사라졌다. 규칙 쪽은 보수적으로 전수 검사하는데 필드 쪽만 느슨하면, 두 검사가
    같은 데이터에 대해 다른 이야기를 하게 된다.
    """
    by_id = scenario_index(ds)
    if by_id is None:
        return set()

    if eps is None or "scenario_id" not in eps.columns:
        used = None
    else:
        col = eps["scenario_id"]
        names = {str(s) for s in col.dropna().unique()}
        # 결측·공백·미등록 id 가 있으면 어느 시나리오가 쓰였는지 알 수 없다.
        used = None if (col.isna().any() or any(not n.strip() for n in names)
                        or not names <= set(by_id)) else names

    entries = by_id.values() if used is None else [by_id[s] for s in used]
    out: set[str] = set()
    for entry in entries:
        out |= _required_fields_of(entry)
    return out


def declares_directed_ssm(entry: object) -> bool:
    """Whether a scenarios.json entry selects R-TS15066-SSM-02 (a selector ending in * included):
    the hri condition (spec §6 「조건부 필드 — 사람과 로봇」)."""
    declared = declared_selectors(entry) if isinstance(entry, dict) else None
    return any(sel == DIRECTED_SSM_RULE_ID
               or (sel.endswith("*") and DIRECTED_SSM_RULE_ID.startswith(sel[:-1]))
               for sel in declared or [])


def declares_conditions(entry: object) -> bool:
    """Whether a scenarios.json entry declares a conditions table (an object under "conditions"): the condition of
    the table scenario's episode columns (spec §5 「조건표가 있는 시나리오의 에피소드 열」)."""
    return isinstance(entry, dict) and isinstance(entry.get(TABLE_CONDITION), dict)


def uses_directed_ssm(ds: Dataset, eps: pd.DataFrame | None) -> bool:
    """The hri condition of a distribution: its scenarios apply R-TS15066-SSM-02. When the scenario
    mapping cannot be trusted (applicable_rule_ids is None), every rule the shipped rules.json has
    applies -- the conservative reading C8 takes for the rule inputs."""
    applicable = applicable_rule_ids(ds, eps)
    if applicable is None:
        return DIRECTED_SSM_RULE_ID in known_rule_ids(ds)
    return DIRECTED_SSM_RULE_ID in applicable or _wildcard_hit(DIRECTED_SSM_RULE_ID, applicable)


def directed_rule(rules: object) -> dict:
    """The shipped R-TS15066-SSM-02 entry, or {} -- what is wrong with it is C1's to say."""
    for rule in json_member(rules, "rules", list):
        if isinstance(rule, dict) and rule.get("rule_id") == DIRECTED_SSM_RULE_ID:
            return rule
    return {}


def approach_gated_violation(distance, separation, approach, threshold):
    """What separation_violation must be under the hri condition: the distance is short of S_p
    (strictly) while the robot moves toward the person faster than the threshold (strictly). The
    generator's separation_violated is the other copy; a test holds the two to one answer."""
    return (distance < separation) & (approach > threshold)


def directed_approach_threshold(rules: object) -> float | None:
    """R-TS15066-SSM-02's approach_speed_threshold when it is a finite number >= 0, else None."""
    entry = json_member(json_member(directed_rule(rules), "parameters", dict),
                        "approach_speed_threshold", dict)
    value = entry.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def c8_values(ds: Dataset, frames: list[tuple[Path, pd.DataFrame]],
              eps: pd.DataFrame | None, rep: Report) -> None:
    """C8 — NaN, 음수 접촉력, 범위 이탈, 차원 불일치, event 허용값."""
    if not frames:
        rep.add("C8", SKIP, "스텝 데이터 없음")
        return

    dof_key, dof_by_ep = episode_lookup(eps, "dof")
    report_unjoined_episodes(frames, any_frame_group_key(frames, dof_key),
                             dof_by_ep, "C8", rep, partial=ds.frames_truncated)

    nan_hits: Counter = Counter()
    zmp_sentinel_hits: Counter = Counter()
    zmp_total_hits: Counter = Counter()
    # safety.stability_margin only — -inf (flight phase) is a documented,
    # expected value there (see the C8 block below), not a NaN-style defect,
    # so it gets its own counters rather than being folded into nan_hits.
    neg_inf_hits: Counter = Counter()
    finite_or_neg_inf_hits: Counter = Counter()

    # 규칙 조회와 적용 범위 판별은 파일마다 달라지지 않는다. 루프 밖에서 한 번만 한다.
    rules = ds.rules()
    applicable = applicable_rule_ids(ds, eps)
    # 규칙이 읽는 입력에 더해, 시나리오가 스스로 "이건 반드시 기록한다" 고 선언한 필드도
    # 함께 요구한다. 선언만 하고 검사하지 않으면 그 선언은 주석과 다르지 않다.
    declared_inputs: set[str] = scenario_required_fields(ds, eps)
    if isinstance(rules, dict) and isinstance(rules.get("rules"), list):
        for r in rules["rules"]:
            if not isinstance(r, dict):
                continue
            rid = str(r.get("rule_id"))
            if applicable is not None and rid not in applicable \
                    and not _wildcard_hit(rid, applicable):
                continue
            # The fields the data is held to. Another shape names none: said, not read as no inputs.
            inputs = r.get("inputs")
            if inputs is not None and (not isinstance(inputs, list)
                                       or not all(isinstance(x, str) for x in inputs)):
                rep.add("C8", ERROR, f"규칙 {rid} 의 inputs 가 필드 이름의 목록이 아님: {inputs!r:.60}",
                        "meta/rules.json")
            declared_inputs |= {x for x in json_member(r, "inputs", list) if isinstance(x, str)}

    # Camera columns: required when meta/cameras.json ships (the video fields are C1's --
    # they are not parquet columns). The box is read in its camera's native pixels (bbox.camera).
    ships_cameras = ds.ships_cameras()
    box_setup = camera_setup_of(ds) if ships_cameras else None
    if ships_cameras and box_setup is None:
        rep.add("C8", ERROR, f"meta/cameras.json 을 읽지 못해 {BBOX_FIELD} 의 범위를 대조하지 못함 "
                             "(사유는 C1)", "meta/cameras.json")
    box_rows = 0
    # A camera pose is the 7-element form of observation.root_pose and, like it, has no
    # sentinel: it joins the finiteness sweep below wherever the column exists.
    pose_columns = tuple(c for c in camera_column_fields() if c != BBOX_FIELD)
    # The hri columns (spec §6 「조건부 필드 — 사람과 로봇」): required when the distribution's
    # scenarios apply R-TS15066-SSM-02. Their violation flag is gated by the robot's approach
    # speed, so the plain (min_distance < required) comparison below takes the gate for them.
    hri = uses_directed_ssm(ds, eps)
    approach_threshold = directed_approach_threshold(rules) if hri else None
    if hri and approach_threshold is None:
        rep.add("C8", ERROR, f"{DIRECTED_SSM_RULE_ID}.parameters.approach_speed_threshold 를 읽지 못해 "
                             "safety.separation_violation 을 대조하지 못함 (사유는 C1)", "meta/rules.json")
    hri_columns = CONDITIONAL_REQUIRED_STEP_FIELDS[HRI_CONDITION]
    visible_columns = tuple(c for c in hri_columns if c.startswith(VISIBLE_FRACTION_PREFIX))

    for path, df in frames:
        loc = str(path)

        for col in REQUIRED_STEP_FIELDS:
            if col not in df.columns:
                rep.add("C8", ERROR, f"필수 스텝 필드 없음: {col}", loc)
        if ships_cameras:
            for col in camera_column_fields():
                if col not in df.columns:
                    rep.add("C8", ERROR,
                            f"조건부 필수 스텝 필드 없음: {col} — meta/cameras.json 을 싣는 배포본은 "
                            "이 열도 싣는다 (규격서 §6 「조건부 필드 — 카메라」)", loc)
            if box_setup is not None and BBOX_FIELD in df.columns and len(df):
                problems, checked = bbox_problems(df[BBOX_FIELD],
                                                  box_setup.sizes[box_setup.bbox_camera],
                                                  box_setup.bbox_camera)
                box_rows += checked                 # rows actually compared, not rows present
                for problem in problems:
                    rep.add("C8", ERROR, problem, loc)
        if hri:
            for col in hri_columns:
                if col not in df.columns:
                    rep.add("C8", ERROR,
                            f"조건부 필수 스텝 필드 없음: {col} — {DIRECTED_SSM_RULE_ID} 를 선언한 시나리오의 "
                            "배포본은 이 열도 싣는다 (규격서 §6 「조건부 필드 — 사람과 로봇」)", loc)
        # Every column of a step file has a layer (spec §6 「층 — 로우와 가공」) -- LeRobot's own aside.
        unlayered = sorted(set(df.columns) - set(STEP_FIELD_LAYERS) - LEROBOT_STEP_COLUMNS)
        if unlayered:
            rep.add("C8", ERROR,
                    f"층 표식이 없는 스텝 열: {unlayered[:5]} — 규격 정의의 step_field_layers 에 없는 열은 "
                    "로우인지 가공인지 받는 쪽이 알 수 없다", loc)

        # 고정 길이 필드
        for col, n in FIXED_SHAPES.items():
            if col in df.columns and len(df):
                got = elem_len(df[col].iloc[0])
                if got != n:
                    rep.add("C8", ERROR, f"{col} 길이 {got} != 규격 {n}", loc)
                # 첫 행만 보면 뒤에서 길이가 무너진 열이 통과한다. 길이는 값마다
                # 성질이므로 전 행을 센다.
                bad_rows = sum(1 for v in df[col] if elem_len(v) != n)
                if bad_rows and bad_rows != len(df):
                    rep.add("C8", ERROR,
                            f"{col} 길이가 규격 {n} 과 다른 행 {bad_rows}개", loc)

        # 관절 차원. 한 parquet 에 여러 에피소드가 들어가므로 첫 행 하나로
        # 파일 전체를 대표시키지 않고 에피소드마다 본다.
        gkey = frame_group_key(df, dof_key)
        if dof_by_ep and gkey:
            for eid_g, g in df.groupby(gkey, sort=False):
                dof = dof_by_ep.get(str(eid_g))
                if not dof or not len(g):
                    continue
                for col in JOINT_DIM_FIELDS:
                    if col in g.columns:
                        # 첫 행만 보면 뒤에서 폭이 무너진 열이 통과한다. 바로 위
                        # fixed_shapes 검사처럼 전 행을 본다.
                        widths = {elem_len(v) for v in g[col]}
                        if len(widths) > 1:
                            rep.add("C8", ERROR,
                                    f"{col} 폭이 에피소드 안에서 일정하지 않음: "
                                    f"{sorted(widths)[:4]}", f"{loc}::{eid_g}")
                        got = elem_len(g[col].iloc[0])
                        if got != int(dof):
                            rep.add("C8", ERROR,
                                    f"{col} 차원 {got} != dof {dof}", f"{loc}::{eid_g}")

        # 값 검사
        checks = [
            ("observation.state.joint_pos", SANITY["joint_pos_abs_max"], "rad(도 단위 혼동 의심)"),
            ("observation.state.joint_vel", SANITY["joint_vel_abs_max"], "rad/s"),
            ("observation.state.joint_torque", SANITY["joint_torque_abs_max"], "N·m"),
        ]
        for col, limit, unit in checks:
            if col not in df.columns:
                continue
            arr = np.concatenate([flat(v) for v in df[col]]) if len(df) else np.array([])
            if arr.size == 0:
                continue
            if np.isnan(arr).any():
                nan_hits[col] += int(np.isnan(arr).sum())
            mx = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
            if mx > limit:
                rep.add("C8", ERROR, f"{col} 최대 |{mx:.3g}| 이 한계 {limit:g} 초과 [{unit}]", loc)

        # 크기·상대속도는 모두 음수가 될 수 없고 상한이 있다. v_rel 이 여기 없던
        # 동안 400행을 전부 NaN 으로 바꿔도 C3·C8 이 아무 말도 하지 않았다
        # — 필수 필드를 늘리면서 값 계약은 안 늘린 것이다.
        for col, limit_key in (("observation.contact.force", "contact_force_max"),
                               ("observation.contact.pressure", "contact_force_max"),
                               ("observation.contact.v_rel", "contact_v_rel_max")):
            if col not in df.columns:
                continue
            arr = np.concatenate([flat(v) for v in df[col]]) if len(df) else np.array([])
            if arr.size == 0:
                continue
            if np.isnan(arr).any():
                nan_hits[col] += int(np.isnan(arr).sum())
            if np.isinf(arr).any():
                rep.add("C8", ERROR, f"{col} 에 무한대 존재 — 정의된 범위 밖", loc)
            if (arr < 0).any():
                rep.add("C8", ERROR, f"{col} 에 음수 존재 (최소 {float(np.nanmin(arr)):.3g})", loc)
            if arr.size and float(np.nanmax(arr)) > SANITY[limit_key]:
                rep.add("C8", ERROR, f"{col} 최대값 비현실적: {float(np.nanmax(arr)):.3g}", loc)

        # 필수 벡터 열의 유한성. 아무도 보지 않던 자리라, 첫 행이 멀쩡하면 뒤가
        # NaN·무한대여도 통과했다. safety.zmp 와 stability_margin 은 각각 NaN·-inf
        # 가 문서화된 의미를 갖는 열이라 여기서 제외하고 아래에서 따로 본다.
        for col in ("observation.root_pose", "observation.root_vel",
                    "observation.imu", "action", "safety.com") + pose_columns + (
                    "observation.command_velocity", "observation.human.root_pose",
                    "observation.human.root_vel"):
            if col not in df.columns or not len(df):
                continue
            arr = np.concatenate([flat(v) for v in df[col]])
            if arr.size and not np.isfinite(arr).all():
                n_bad = int((~np.isfinite(arr)).sum())
                rep.add("C8", ERROR, f"{col} 에 유한하지 않은 값 {n_bad}개", loc)

        for col in ("safety.min_distance_human", "safety.separation_required"):
            if col not in df.columns:
                continue
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy()
            if np.isnan(arr).any():
                nan_hits[col] += int(np.isnan(arr).sum())
            if (arr[~np.isnan(arr)] < 0).any():
                rep.add("C8", ERROR, f"{col} 에 음수 거리", loc)

        # The two approach speeds are speeds toward: below 0 is written as 0 (spec §6).
        for col in ("safety.approach_speed_robot", "safety.approach_speed_human"):
            if col not in df.columns:
                continue
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if np.isnan(arr).any():
                nan_hits[col] += int(np.isnan(arr).sum())
            if np.isinf(arr).any():
                rep.add("C8", ERROR, f"{col} 에 무한대 존재 — 정의된 범위 밖", loc)
            if (arr[np.isfinite(arr)] < 0).any():
                rep.add("C8", ERROR, f"{col} 에 음수 — 다가가는 속도는 0 아래를 0 으로 적는다", loc)
        for col in visible_columns:
            if col not in df.columns:
                continue
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if np.isnan(arr).any():
                nan_hits[col] += int(np.isnan(arr).sum())
            outside = int((~np.isnan(arr) & ((arr < 0) | (arr > 1))).sum())
            if outside:
                rep.add("C8", ERROR, f"{col} 이 0~1 밖인 스텝 {outside}개 — 보이는 몫은 0 과 1 사이", loc)

        # safety.stability_margin: -inf is the documented sentinel for the
        # flight phase (both feet unloaded at once — schema §6 "산출 방법"),
        # not a defect, so it must not be folded into the generic NaN/range
        # checks above. np.isnan(-inf) is False, so the plain NaN sweep this
        # loop already runs for other float columns silently lets -inf pass
        # through without ever having looked at it — this block is the
        # explicit check that closes that gap. Actual NaN (a real defect —
        # the support-polygon computation itself failing) and +inf (not a
        # value this function's contract ever produces) are still errors.
        if "safety.stability_margin" in df.columns:
            arr = pd.to_numeric(df["safety.stability_margin"], errors="coerce").to_numpy()
            if np.isnan(arr).any():
                nan_hits["safety.stability_margin"] += int(np.isnan(arr).sum())
            if np.isposinf(arr).any():
                rep.add("C8", ERROR,
                        "safety.stability_margin 에 +inf 존재 — 정의된 값 범위 밖", loc)
            neg_inf_hits["safety.stability_margin"] += int(np.isneginf(arr).sum())
            finite_or_neg_inf_hits["safety.stability_margin"] += int((~np.isnan(arr)).sum())

        # safety.zmp / safety.com: 둘 다 필수 벡터 필드인데 어느 일반 검사도
        # 이들을 보지 않는다. ZMP 는 두 성분이
        # 모두 NaN 인 것이 스키마가 정의한 표식이고(§6 산출 방법), 한쪽만 NaN 인
        # 값은 산출 오류다. CoM 에는 표식이 없으므로 NaN 자체가 결함이다.
        if "safety.zmp" in df.columns and len(df):
            zmp = np.stack([np.asarray(v, dtype=float).reshape(-1) for v in df["safety.zmp"]])
            if zmp.shape[1] == 2:
                nan_count = np.isnan(zmp).sum(axis=1)
                half = int((nan_count == 1).sum())
                if half:
                    rep.add("C8", ERROR,
                            f"safety.zmp 의 한쪽 성분만 NaN 인 스텝 {half}개 — "
                            "표식은 두 성분 모두 NaN 이어야 한다 (스키마 §6)", loc)
                if np.isposinf(zmp).any() or np.isneginf(zmp).any():
                    rep.add("C8", ERROR, "safety.zmp 에 무한대 존재 — 정의된 범위 밖", loc)
                zmp_sentinel_hits[loc] += int((nan_count == 2).sum())
                zmp_total_hits[loc] += int(len(zmp))

        if "safety.com" in df.columns and len(df):
            com = np.stack([np.asarray(v, dtype=float).reshape(-1) for v in df["safety.com"]])
            bad = int((~np.isfinite(com)).any(axis=1).sum())
            if bad:
                rep.add("C8", ERROR,
                        f"safety.com 에 비유한 값이 있는 스텝 {bad}개 — 무게중심에는 "
                        "표식이 없으며 NaN 은 입력 손상을 뜻한다", loc)

        # separation_violation 파생 정합 — under the hri condition the robot must also be approaching.
        # Without the approach-speed column there is no gate to hold the flag to, and the plain
        # comparison stands: the hri condition is also on when the scenario mapping cannot be
        # trusted (uses_directed_ssm), for a release that may carry no hri column at all, and the
        # flag must not go unread there. The missing column is said above.
        need = {"safety.min_distance_human", "safety.separation_required",
                "safety.separation_violation"}
        gated = hri and "safety.approach_speed_robot" in df.columns
        if need <= set(df.columns) and not (gated and approach_threshold is None):
            a = pd.to_numeric(df["safety.min_distance_human"], errors="coerce").to_numpy()
            b = pd.to_numeric(df["safety.separation_required"], errors="coerce").to_numpy()
            v = df["safety.separation_violation"].astype("boolean").to_numpy(dtype=object)
            ok = ~np.isnan(a) & ~np.isnan(b)
            if not gated:
                expect = np.where(ok, a < b, None)
                said = "(min_distance < required)"
            else:
                r = pd.to_numeric(df["safety.approach_speed_robot"], errors="coerce").to_numpy(dtype=float)
                ok &= ~np.isnan(r)
                expect = np.where(ok, approach_gated_violation(a, b, r, approach_threshold), None)
                said = f"(min_distance < required 이면서 approach_speed_robot > {approach_threshold:g})"
            mism = sum(1 for i in range(len(df))
                       if ok[i] and v[i] is not None and bool(v[i]) != bool(expect[i]))
            if mism:
                rep.add("C8", ERROR, f"separation_violation 이 {said} 와 불일치: {mism}건", loc)

        # 적용되는 규칙이 선언한 입력 필드가 실제 데이터에 있는지 본다. 선언과 데이터가
        # 어긋나면 라벨을 산출한 근거를 재계산할 수 없다.
        absent = sorted(declared_inputs - set(df.columns))
        if absent:
            rep.add("C8", ERROR,
                    f"시나리오가 선언한 규칙 입력·필수 필드가 데이터에 없음: {absent[:5]}", loc)

        # 접촉 유형 · safety.event 허용값. 스텝마다 달라지므로 첫 프레임만 보면 사실상 검사하지 않는 것과 같다.
        # A cell that does not read as a list is not an empty list: the schema ships these fields
        # as JSON text (json_encoded_fields.step), and a reader that parses it gets an error, not
        # "no events". A null is c7_step_nulls' to say. Each element is read as text, as the
        # region codes are (c3_contact_regions).
        for col, allowed, label in (("observation.contact.type", CONTACT_TYPES, "접촉 유형"),
                                    ("safety.event", SAFETY_EVENTS, "safety.event")):
            if col not in df.columns:
                continue
            outside: set[str] = set()
            unread: int | None = None
            # A null reaches pandas as None or NaN in a column of text. In a column of numbers a
            # NaN is a value (Arrow holds no null there, so C7 says nothing) and no JSON list.
            numbers = pd.api.types.is_numeric_dtype(df[col])
            for i, v in enumerate(df[col]):
                items = as_list(v)
                if items is None:
                    if unread is None and (numbers or not scalar_missing(v)):
                        unread = i
                    continue
                outside |= {str(x) for x in items} - allowed
            if unread is not None:
                rep.add("C8", ERROR, f"{col} 가 JSON 배열로 읽히지 않음", f"{loc}#{unread}")
            if outside:
                rep.add("C8", ERROR, f"허용되지 않은 {label}: {sorted(outside)[:5]}", loc)

        # A contact type is given on the steps the robot touches the person (spec §8.1) and
        # contact_onset marks the step a contact begins (spec §6): a typed step before its episode's
        # first contact_onset is a touch no event marks. The labeller knows contact by contact_onset
        # alone (R-HRI-CONTACT-01 reads nothing else), so that touch would go unlabelled. A cell that
        # does not read as a list is said above. An episode lies in one data file (C7).
        if {"observation.contact.type", "safety.event", "episode_index"} <= set(df.columns):
            begun: set = set()
            unmarked: dict = {}
            for i, (ep, kinds, events) in enumerate(zip(df["episode_index"], df["observation.contact.type"],
                                                        df["safety.event"])):
                if ep in begun or ep in unmarked:
                    continue
                if "contact_onset" in [str(x) for x in as_list(events) or []]:
                    begun.add(ep)
                elif any(str(x) != "none" for x in as_list(kinds) or []):
                    unmarked[ep] = i
            if unmarked:
                rep.add("C8", ERROR,
                        f"접촉 유형이 매겨진 스텝보다 앞선 contact_onset 이 없는 에피소드 {len(unmarked)}개 — 유형은 "
                        "로봇이 사람에 닿은 스텝에 매겨지고 contact_onset 은 접촉이 시작한 스텝이다. 라벨의 접촉 "
                        "판정은 contact_onset 을 읽으므로 그 접촉이 라벨에 들지 않는다",
                        f"{loc}#{min(unmarked.values())}")

    for col, n in nan_hits.items():
        rep.add("C8", ERROR, f"{col} 에 NaN {n}개")
    rep.stats["nan_counts"] = dict(nan_hits)
    if ships_cameras:
        rep.stats.setdefault("camera_media", {})["bbox_rows"] = box_rows

    # Visibility, not a finding: -inf is expected (flight phase), so this is
    # reported as a stat for a report to cite, the same way
    # outcome_distribution/regions_seen are, not raised as WARN/ERROR.
    if "safety.stability_margin" in finite_or_neg_inf_hits:
        total = finite_or_neg_inf_hits["safety.stability_margin"]
        neg_inf = neg_inf_hits.get("safety.stability_margin", 0)
        rep.stats["stability_margin_neg_inf_fraction"] = (neg_inf / total) if total else None

    # Same reasoning for the ZMP sentinel: expected, so counted rather than
    # flagged -- but counted out loud, because a reader who sees NaN in a
    # required column without knowing it is a sentinel will read it as
    # missing data. Note this is NOT the flight phase: measured on S1a, 27 of
    # 74 sentinel steps had a finite stability_margin (schema §6).
    sentinel = sum(zmp_sentinel_hits.values())
    zmp_total = sum(zmp_total_hits.values())
    if zmp_total:
        rep.stats["zmp_undefined_fraction"] = sentinel / zmp_total
        rep.add("C8", INFO,
                f"safety.zmp 표식(두 성분 NaN) {sentinel}/{zmp_total} 스텝 — "
                "결측이 아니라 ZMP 미정의 (스키마 §6)")

    check_stats_of_non_finite_columns(ds, frames, eps, rep)


def check_stats_of_non_finite_columns(ds: Dataset, frames, eps, rep: Report) -> None:
    """비유한값이 있는 열의 정규화 통계가 **유한한 채로 틀린** 값을 싣고 있지 않은지.

    `NaN` 은 시끄럽게 실패한다 — JSON 으로도 안 나가고 쓰는 쪽에서 바로 터진다.
    분위수는 그렇지 않다. LeRobot 의 통계 경로는 `-inf` 가 하나만 있어도 그 필드의
    q01~q99 를 **전부 최댓값 부근으로** 무너뜨리고, 그 값들은 유한하므로 앞선
    `NaN → null` 청소를 그대로 통과한다.

    실측(2026-08-29, 배포본 400건): `safety.stability_margin` 의 다섯 분위수가 전부
    +0.1235 로 실려 있었고 유한값의 실제 중앙값은 -0.1302 였다. **「음수면
    지지다각형 밖」인 필드의 중앙값이 부호가 뒤집힌 채** 나가고 있었다.

    변환기는 그런 필드의 통계를 비우고 사유를 적는다(`normalise_stats`). 이 검사는
    **나간 물건 쪽에서** 같은 것을 본다 — 어느 필드가 대상인지는 스텝 데이터에서
    유도하므로 목록이 낡지 않는다.
    """
    stats = ds._json("stats.json")
    if not isinstance(stats, dict) or not frames:
        return

    non_finite: set[str] = set()
    for _, frame in frames:
        for column in frame.columns:
            if column in non_finite or column not in stats:
                continue
            try:
                values = np.asarray(frame[column].tolist(), dtype=float)
            except (TypeError, ValueError):
                continue
            if values.size and not np.isfinite(values).all():
                non_finite.add(column)

    for column in sorted(non_finite):
        block = stats.get(column)
        if not isinstance(block, dict):
            continue
        # 성분이 전부 null 인 리스트는 「비어 있다」다. `v is not None` 로만 보면
        # 이미 비운 `[null, null]` 이 「값 있음」으로 잡힌다 (2026-08-29 자가 확인).
        # Walked with a stack: JSON nests deeper than a recursive walk can follow.
        def has_number(value) -> bool:
            stack = [value]
            while stack:
                v = stack.pop()
                if isinstance(v, list):
                    stack.extend(v)
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    return True
            return False

        present = sorted(k for k, v in block.items()
                         if k not in ("count", "undefined_reason") and has_number(v))
        if present:
            rep.add("C8", ERROR,
                    f"{column} 은 비유한값을 담는데 정규화 통계 {present} 가 값으로 "
                    "실려 있습니다 — LeRobot 의 분위수는 그런 열에서 최댓값 부근으로 "
                    "무너지므로 유한하지만 틀립니다. 변환기가 비우도록 되어 있습니다 "
                    "(convert_lerobot.normalise_stats)",
                    "meta/stats.json")
        elif "undefined_reason" not in block:
            rep.add("C8", WARN,
                    f"{column} 의 통계가 비어 있는데 사유가 없습니다 — 받는 쪽이 "
                    "결측으로 읽습니다", "meta/stats.json")

    # 같은 값이 옆 파일에도 있다. 전역 통계는 에피소드별 통계를 합친 것이고
    # **무너지는 것은 에피소드 단계**이므로, 전역만 비우면 틀린 숫자가 그대로 남는다.
    if eps is None or not len(eps):
        return
    for column in sorted(non_finite):
        prefix = f"stats/{column}/"
        stale = []
        for name in eps.columns:
            if not name.startswith(prefix) or name.endswith("/count"):
                continue
            values = np.asarray(pd.Series(eps[name]).map(
                lambda v: np.asarray(v, dtype=float).ravel()).tolist(), dtype=float)
            if values.size and np.isfinite(values).any():
                stale.append(name[len(prefix):])
        if stale:
            rep.add("C8", ERROR,
                    f"{column} 은 비유한값을 담는데 에피소드별 통계 {sorted(stale)} 가 "
                    "값으로 실려 있습니다 — 전역 통계만 비우면 같은 틀린 값이 옆 "
                    "파일에 남습니다 (convert_lerobot.blank_episode_stats)",
                    "meta/episodes/**.parquet")


def scalar_missing(value) -> bool:
    """`pd.isna` 를 스칼라가 아닌 값에도 안전하게. 배열이면 「없음」이 아니다.

    `pd.notna(배열)` 은 불리언 배열을 돌려주고 `if` 에서 ValueError 가 난다 —
    검증기가 죽으면 exit 3 이고, 그건 「검사하지 못했다」이다.
    """
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def int_of_digits(digits: str) -> int | None:
    """int() of a string of ASCII digits, or None past the interpreter's digit limit, where int()
    raises ValueError (sys.get_int_max_str_digits). No index, number or seed a release holds is
    that long."""
    try:
        return int(digits)
    except ValueError:
        return None


def numeric_or_none(value) -> float | None:
    """수치로 읽히면 float, 아니면 None. bool 은 수치가 아니다.

    `float("not-a-number")` 는 ValueError 로 검증기를 통째로 죽여 exit 3 을 만들고,
    `True` 는 1.0 으로 조용히 통과한다 — 둘 다 「검사했다」가 아니다.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def episode_lookup(eps, value_col: str) -> tuple[str | None, dict]:
    """(step-frame join key, {key -> value}) for a per-episode quantity.

    Step frames and episode metadata are keyed differently. LeRobot writes
    ``episode_index`` on every frame; our ``episode_id`` lives only in the
    episode table. A fixture that puts ``episode_id`` on frames too lets the
    per-episode checks join there and never on a real distribution --
    green on the fixture, silent on the thing being shipped. Both maps are
    built and the caller uses whichever key its frames actually carry.
    """
    if eps is None or value_col not in eps.columns:
        return None, {}
    for key in ("episode_index", "episode_id"):
        if key in eps.columns:
            rows = [(str(getattr(r, key)), getattr(r, value_col))
                    for r in eps.itertuples()
                    if not scalar_missing(getattr(r, value_col))
                    and not scalar_missing(getattr(r, key))]
            table = dict(rows)
            # 같은 키가 둘이면 dict 가 조용히 뒤엣것만 남긴다 — 그러면 앞 에피소드의
            # 값으로 검사받아야 할 프레임이 다른 값으로 검사받는다.
            if len(table) != len(rows):
                dupes = sorted({k for k, _ in rows if [x for x, _ in rows].count(k) > 1})
                table["__duplicate_keys__"] = dupes
            if table:
                return key, table
    return None, {}


def any_frame_group_key(frames, preferred: str | None) -> str | None:
    """어느 파일에서든 찾은 조인 키. 첫 파일에 없으면 전역 검사가 통째로 꺼졌다."""
    for _, df in frames:
        key = frame_group_key(df, preferred)
        if key:
            return key
    return None


def frame_group_key(df, preferred: str | None) -> str | None:
    """Which column groups these step frames into episodes."""
    order = [preferred] if preferred else []
    order += ["episode_index", "episode_id"]
    return next((c for c in order if c and c in df.columns), None)


def report_unjoined_episodes(frames, key: str | None, table: dict, check: str,
                             rep: "Report", partial: bool = False) -> None:
    """프레임에는 있는데 에피소드 메타에는 없는 에피소드를 보고한다.

    에피소드 단위 검사는 메타에서 값을 못 찾으면 그 그룹을 건너뛴다. 조용히
    건너뛰면 메타에 없는 에피소드의 프레임은 **아무 검사도 받지 않고** 통과한다 —
    주기가 달라도, 관절 폭이 달라도.
    """
    if not key or not table:
        return
    # 파일을 일부만 읽었으면 「메타에는 있는데 프레임이 없다」를 말할 수 없다 —
    # 안 읽은 파일에 있을 뿐이다. --file-limit 로 빠른 점검을 하면 정상 배포본이
    # 통째로 빨간불이 됐다.
    if partial:
        rep.add(check, INFO,
                "파일을 일부만 읽어 에피소드 대조를 건너뜀 (--file-limit)")
        return
    # **배포본 전체**를 한 번에 본다. LeRobot 은 프레임을 여러 parquet 으로 쪼개므로
    # 파일마다 대조하면 다른 파일에 있는 에피소드가 전부 「프레임 없음」이 된다 —
    # 140 에피소드 드라이런에서 그 거짓 오류가 138건 나왔다 (2026-08-25).
    loc = "data/"
    dupes = table.get("__duplicate_keys__")
    if dupes:
        rep.add(check, ERROR,
                f"에피소드 메타에 중복된 조인 키 {len(dupes)}개: {dupes[:3]} — "
                "뒤엣것이 앞엣것을 덮어 프레임이 남의 값으로 검사된다", loc)
    known = {k for k in table if k != "__duplicate_keys__"}
    frames_have: set[str] = set()
    for path, df in frames:
        if key in df.columns:
            frames_have |= {str(v) for v in df[key].unique()}
    orphan = sorted(frames_have - known)
    if orphan:
        rep.add(check, ERROR,
                f"에피소드 메타에 없는 프레임 에피소드 {len(orphan)}개: {orphan[:3]} — "
                "이 프레임들은 에피소드 단위 검사를 받지 못한다", loc)
    # 반대쪽도 본다 — 메타에는 있는데 프레임이 없는 에피소드는 데이터가 통째로
    # 빠진 것이고, 집합 차이 한쪽만 보면 안 보인다.
    absent = sorted(known - frames_have)
    if absent:
        rep.add(check, ERROR,
                f"프레임이 없는 에피소드 메타 {len(absent)}개: {absent[:3]} — "
                "선언된 에피소드의 시계열이 배포본에 없다", loc)


def _c9_picks(eligible: pd.DataFrame, n: int) -> pd.DataFrame:
    """The episodes C9 reruns, in the order it reruns them: with a ``scenario_id`` column holding two
    scenarios or more, one of each first -- the scenarios in name order, each one's
    ``sample(1, random_state=0)`` -- up to ``n``, then what ``n`` leaves from the rest,
    ``sample(k, random_state=0)``; otherwise ``eligible.sample(n, random_state=0)``, the picks C9 always
    made. ``n`` is held to the number of eligible episodes.

    Each scenario is generated by its own code path, and a few drawn from the whole miss one: of three
    scenarios of 400 episodes each, two picks always do and three do 78 times in 100."""
    n = min(n, len(eligible))
    names = eligible["scenario_id"].map(str) if "scenario_id" in eligible.columns else None
    if names is None or names.nunique() < 2:
        return eligible.sample(n, random_state=0)
    first = [eligible[names == name].sample(1, random_state=0) for name in sorted(names.unique())][:n]
    picked = pd.concat(first)
    rest = eligible.drop(picked.index)
    return pd.concat([picked, rest.sample(n - len(picked), random_state=0)])


def c9_seed_repro(ds: Dataset, eps: pd.DataFrame | None, hook: str | None,
                  sample: int, rep: Report, max_seq: int | None = None) -> None:
    """C9 — 동일 시드 재실행 재현성. 외부 훅이 있을 때만 수행.

    **재현 비용은 그 에피소드가 자기 실행에서 몇 번째인가에 비례합니다.** 에피소드 단독
    재현이 성립하지 않으므로(규격서 §3) 훅은 그 에피소드를 만든 실행을 첫 번호
    (`run_start_seq`)부터 그 에피소드까지 다시 생성합니다 — 한 실행으로 만든 배포본에서
    380번을 고르면 381건을 다시 돌립니다. 무작위로 고르면 400건짜리 배포본에서 표본 하나가
    몇 시간이 되고, 훅 타임아웃에 걸려 「재현 실패」로 보고됩니다. 데이터의 문제가
    아닌데 빨간불이 서는 것입니다.

    그래서 표본은 **실행 안에서 `max_seq` 번 이내인 에피소드만** 고르고, 그 바깥은 확인하지
    않았다고 보고에 남깁니다. 약한 보장이지만 참인 보장입니다 — 확인하지 않은 것을 확인한
    것처럼 말하지 않는 쪽을 택했습니다. 범위 안인지는 위치(번호 − `run_start_seq`)가 정합니다:
    행 하나가 실행 하나인 배포본(조건표 시나리오)은 모든 에피소드가 위치 0 이라 카메라가
    있어도 전부 범위 안이고, 10건 청크 하나가 실행 하나인 배포본은 카메라가 없으면 전부,
    카메라 배포본은 실행의 앞쪽만 범위 안입니다(아래). 전수에 가깝게 보려면
    `--repro-max-seq` 를 올리십시오.

    Of the episodes within reach, _c9_picks chooses which and in what order: one of every scenario
    before a second of any.

    Without ``max_seq`` the distribution decides: REPRO_MAX_SEQ_CAMERAS when it ships
    meta/cameras.json (Dataset.ships_cameras, the camera condition), REPRO_MAX_SEQ otherwise --
    a camera-less distribution keeps the wider REPRO_MAX_SEQ. A given ``max_seq`` is kept
    as given, on a camera distribution too: going deeper is the caller's call, at its cost.
    """
    if hook is None:
        rep.add("C9", SKIP, "재현성 훅 미지정 (--repro-hook). 배포 전 반드시 1회 수행할 것")
        return
    # 훅을 주고 표본을 0 으로 두면 아무것도 실행하지 않은 채 OK 가 된다 —
    # 「확인했다」와 구별되지 않는다.
    if sample < 1:
        # SKIP 으로 두면 exit 0 이라 검증기만으로는 못 막는다 — 훅을 주면서 0 을
        # 달라는 것은 모순이므로 오류다.
        rep.add("C9", ERROR,
                f"--repro-hook 을 주고 --repro-sample 을 {sample} 로 두었음 — 훅을 한 번도 "
                "부르지 않으면서 재현을 확인했다고 말하게 된다. 1 이상이어야 한다")
        return
    if eps is None:
        rep.add("C9", SKIP, "에피소드 메타 없음")
        return

    sims = eps[eps.get("source") == "sim"] if "source" in eps.columns else eps
    if sims.empty:
        rep.add("C9", SKIP, "sim 에피소드 없음")
        return

    # The bound on the camera cost: a bounded prefix, not a longer timeout. The timeout is what
    # reports a hook that hangs, and the cost grows with the prefix -- 21 camera episodes are
    # ~26 min typical and ~40 min at the slowest measured, per sample.
    camera_default = max_seq is None and ds.ships_cameras()
    if max_seq is None:
        max_seq = REPRO_MAX_SEQ_CAMERAS if camera_default else REPRO_MAX_SEQ

    # 접두 비용 상한. episode_id 에서 일련번호를 읽어 거른다 — C5 가 형식을 이미
    # 강제하므로 여기서 못 읽는 것은 그 자체로 보고할 일이다.
    def seq_of(eid: str) -> int | None:
        m = EPISODE_ID_RE.match(str(eid))
        return int_of_digits(m.group("seq")) if m else None

    # Where each episode sits in the run that made it. The hook reruns that run from its first
    # seq (the episode's run_start_seq, schema §3) up to the episode, so the cost is the place in
    # the run, not the number. A distribution without the column was made before the field:
    # every run began at 0, as the hook assumed then.
    def start_of(v) -> float:
        n = numeric_or_none(v)
        return n if n is not None and n == int(n) and n >= 0 else float("nan")

    seqs = (pd.to_numeric(sims["episode_id"].map(seq_of), errors="coerce")
            if "episode_id" in sims.columns else None)
    starts = (sims["run_start_seq"].map(start_of) if "run_start_seq" in sims.columns
              else pd.Series(0.0, index=sims.index))
    if seqs is None or seqs.isna().all():
        rep.add("C9", WARN,
                "episode_id 에서 일련번호를 읽지 못해 재현 비용을 가늠할 수 없음 — "
                "재현은 실행의 첫 번호부터 다시 돌리므로 전수 표본은 몇 시간이 될 수 있음")
        eligible = sims
        skipped = 0
    else:
        place = seqs - starts               # NaN where either cannot be read
        keep = place.notna() & (place >= 0) & (place <= max_seq)
        eligible = sims[keep]
        skipped = int((~keep).sum())
    if eligible.empty:
        # 훅을 지정했다는 것은 재현을 확인하겠다는 뜻이다. 한 번도 부르지 못했는데
        # 경고로 끝나면 기본 실행이 exit 0 이라 「C9 를 돌렸다」가 아무것도 확인하지
        # 않고 성공한다 — 비용 상한이 여는 구멍이다.
        rep.add("C9", ERROR,
                f"재현 훅을 지정했으나 실행 안에서 접두 {max_seq} 번 이내의 에피소드가 없어 한 번도 "
                "실행하지 못함 — --repro-max-seq 를 올리십시오(재생성 비용이 그 에피소드가 자기 "
                "실행에서 몇 번째인가에 비례합니다)")
        return

    picks = _c9_picks(eligible, sample)
    if skipped:
        why = (f" (카메라 배포본 — 에피소드 하나를 다시 만드는 데 최대 {REPRO_CAMERA_EPISODE_S:g} 초가 "
               f"들어 훅 제한 시간 {REPRO_HOOK_TIMEOUT_S} 초에 맞춘 범위)" if camera_default else "")
        rep.add("C9", INFO,
                f"재현 확인 범위: 실행 안에서 접두 {max_seq} 번 이내{why}. 나머지 {skipped}건은 확인하지 "
                "않았음 — 재현은 그 에피소드를 만든 실행을 첫 번호부터 다시 돌리므로 비용이 실행 "
                "안에서의 위치에 비례하기 때문 (--repro-max-seq)")
    for position, (idx, row) in enumerate(picks.iterrows()):
        eid = str(row.get("episode_id"))
        seed = row.get("randomization_seed")
        start = starts.loc[idx]
        start = int(start) if start == start else 0     # unreadable only when the id was too
        # 훅에게 **어느 배포본과 대조할지** 알려 준다. 없으면 훅이 자기 기본값으로
        # 떨어지는데 그건 검증 중인 데이터셋이 아니다 — 그러고도 우연히 통과하면
        # C9 는 **다른 데이터를 대조하고 초록**이 된다. 실제로 400건 배치를 검증하며
        # 10건짜리 옛 배포본과 대조했다 (2026-08-25).
        # 훅이 시험용으로 여는 문을 **생산 경로에서는 닫는다.** 열어 두면
        # `C9_ISAAC_PYTHON=/bin/true` 하나로 시뮬레이터도 비교기도 돌지 않은 채
        # 「재현 확인」이 나온다 — 실제로 400건에 대해 그렇게 만들어 봤다.
        # 시험은 훅을 직접 부르므로 영향이 없다.
        env = {k: v for k, v in os.environ.items() if not k.startswith("C9_")}
        env["C9_PUBLISHED_ROOT"] = str(ds.root.resolve())
        # PATH 도 고정한다. 훅의 `#!/usr/bin/env bash` 가 호출자 PATH 에서 bash 를
        # 찾으므로, 거기 가짜 bash 를 두면 **정확한 저장소 훅이 아무것도 하지 않고
        # exit 0** 이 된다. 이것으로 악의를 막지는
        # 못한다 — 훅을 부를 수 있는 사람은 어차피 무엇이든 할 수 있다. 막는 것은
        # 이상한 환경에서 조용히 초록이 되는 사고 쪽이다.
        env["PATH"] = _SAFE_PATH
        # A stalled hook is run again: a stall says nothing about the data, and the publisher does not retry a
        # verdict, so one stalled sample stopped a publish and the whole C9 had to be run again. Nothing else is
        # retried -- a mismatch is a verdict, a timeout is a hook that kept writing and still did not finish.
        group_left = False
        for attempt in range(1, REPRO_STALL_ATTEMPTS + 1):
            try:
                run = run_hook([hook, str(eid), str(seed), str(start)], env, REPRO_HOOK_TIMEOUT_S)
            except Exception as e:  # noqa: BLE001
                rep.add("C9", ERROR, f"재현 훅 실행 실패: {e}", eid)
                break
            returncode, stdout, stderr = run
            stalled = returncode is None and getattr(run, "stalled", False)
            group_left = getattr(run, "group_left", False)
            # A member left running would share the GPU with the next run: that one is not run again.
            again = stalled and not group_left and attempt < REPRO_STALL_ATTEMPTS
            if returncode is None:
                # The fixed sentence first, then where the hook was stuck: the tails of what it had written. The
                # group is said to be ended only when it was seen to end.
                ended = ("훅의 프로세스 그룹에 SIGKILL 을 보냈으나 끝났는지 확인하지 못함" if group_left
                         else "훅의 프로세스 그룹을 끝냄")
                stuck = _stuck_hook_output(stdout, stderr)
                quiet = f"훅의 TMPDIR 에 {REPRO_STALL_S:g} 초 동안 쓰인 파일이 없음"
                if again:
                    rep.add("C9", WARN,
                            f"재현 훅이 멈춤 ({attempt}/{REPRO_STALL_ATTEMPTS}) — {quiet} · {ended} · 다시 실행함{stuck}",
                            eid)
                elif stalled and group_left:
                    rep.add("C9", ERROR,
                            f"재현 훅이 멈춤 — {quiet} · {ended} · 다시 실행하지 않음(남은 구성원이 다음 실행과 GPU 를 "
                            f"나눌 수 있음) — 재현 여부를 판정하지 못함{stuck}", eid)
                elif stalled:
                    rep.add("C9", ERROR,
                            f"재현 훅이 {REPRO_STALL_ATTEMPTS}번 모두 멈춤 — {quiet} · {ended} · 재현 여부를 판정하지 "
                            f"못함{stuck}", eid)
                else:
                    rep.add("C9", ERROR, f"재현 훅 시간 초과({REPRO_HOOK_TIMEOUT_S:g} s) — {ended}{stuck}", eid)
            else:
                rep.add("C9", *repro_hook_finding(returncode, stdout, stderr), eid)
            # What the cleanup after the hook could not do (a stand-in's plain tuple has nothing to say).
            for note in getattr(run, "notes", ()):
                rep.add("C9", WARN, note, eid)
            if not again:
                break
        if group_left:
            # Whatever ended it, a group not seen to end may still hold the GPU: the next sample's rerun would share
            # it, as a retry of this one would.
            rest = [str(r.get("episode_id")) for _, r in picks.iloc[position + 1:].iterrows()]
            if rest:
                rep.add("C9", ERROR,
                        f"남은 표본 {len(rest)}개를 돌리지 않음 — 앞 훅의 프로세스 그룹이 끝났는지 확인하지 못해 다음 "
                        f"재생성이 남은 구성원과 GPU 를 나눌 수 있음: {', '.join(rest)}")
            return


def _stuck_hook_output(stdout: str | None, stderr: str | None, lines: int = 3, chars: int = 300) -> str:
    """" (훅 stderr 끝: …; 훅 stdout 끝: …)" -- the last few lines the hook wrote before it was ended, or
    "" when it wrote nothing. stderr first: the hook's own log and the simulator's are what show where it
    was stuck."""
    parts = []
    for label, text in (("stderr", stderr), ("stdout", stdout)):
        tail = " | ".join((text or "").strip().splitlines()[-lines:])[-chars:]
        if tail:
            parts.append(f"훅 {label} 끝: …{tail}")
    return f" ({'; '.join(parts)})" if parts else ""


def _signal_group(p: "subprocess.Popen[str]", sig: int) -> None:
    try:
        os.killpg(p.pid, sig)
    except ProcessLookupError:
        pass


def _group_running(p: "subprocess.Popen[str]") -> bool:
    """Whether any member of the process group ``p`` leads still runs. ``os.killpg(pgid, 0)`` also counts a
    member that has died and not yet been reaped -- a zombie, which runs nothing and holds no file -- so where
    /proc can be read a group of zombies alone counts as gone: the leader after SIGKILL until it is reaped here
    (the poll below), a member whose parent died with it until init reaps it. Without /proc such a group is
    waited for until the caller's bound."""
    p.poll()                            # reap the leader once it has died
    try:
        os.killpg(p.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return True
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
                state, _ppid, pgrp = f.read().rsplit(")", 1)[1].split()[:3]
        except (OSError, ValueError):
            continue                    # gone meanwhile, or not a line to read
        if int(pgrp) == p.pid and state not in ("Z", "X"):
            return True
    return False


def _never() -> bool:
    return False


def _wait_for_group(p: "subprocess.Popen[str]", until: float, cut: "Callable[[], bool]" = _never) -> bool:
    """Wait until no member of ``p``'s group runs, or until the monotonic time ``until`` or ``cut()``; whether it is
    gone."""
    while _group_running(p):
        if time.monotonic() >= until or cut():
            return False
        time.sleep(0.05)
    return True


def _end_what_is_left(p: "subprocess.Popen[str]", grace_over: float, cut: "Callable[[], bool]" = _never) -> bool:
    """SIGKILL whatever of ``p``'s group still runs at ``grace_over`` -- or once ``cut()`` says so: a signal held
    meanwhile -- then wait for it (bounded); whether the group was seen to end. The leader's exit is not the group's:
    a member that ignores SIGTERM and has closed its pipes outlives it, and its TMPDIR would be removed under it.
    An exception during the wait (a caller's own signal handler) sends the SIGKILL at once. Not
    seen to end after the SIGKILL is a member in uninterruptible sleep (D) -- the caller says so rather than that
    the group was ended."""
    gone = False
    try:
        gone = _wait_for_group(p, grace_over, cut)
    finally:
        if not gone:
            _signal_group(p, signal.SIGKILL)
            gone = _wait_for_group(p, time.monotonic() + REPRO_COLLECT_TIMEOUT_S)
    return gone


def _text(data: "bytes | str | None") -> str:
    """The output a TimeoutExpired carries: raw bytes even for a text-mode Popen."""
    return data.decode("utf-8", "replace") if isinstance(data, bytes) else (data or "")


def _collect_hook_output(p: "subprocess.Popen[str]") -> tuple[str, str]:
    """What the hook wrote, once its group is ended. Bounded: the pipes are also held by anything that left
    the group (a setsid'd child of the simulator), and a plain communicate() waits for every holder to close
    them -- as long as the escapee lives. On expiry the pipes are closed and what was read so far returned."""
    try:
        return p.communicate(timeout=REPRO_COLLECT_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        for pipe in (p.stdout, p.stderr):
            try:
                pipe.close()
            except OSError:
                pass
        p.poll()                        # reap the leader; it is dead or dying (SIGKILL)
        return _text(e.stdout), _text(e.stderr)


def _remove_hook_tmpdir(work: str) -> "str | None":
    """Remove the hook's TMPDIR; None once it is gone, else the sentence saying what is left. Not raised -- the hook's
    own result still has to come back -- and not ignored either: a subdirectory the hook has made read-only keeps its
    files, and the directory would stay behind without a word."""
    shutil.rmtree(work, ignore_errors=True)
    if not os.path.lexists(work):
        return None
    left = [os.path.join(top, name) for top, dirs, files in os.walk(work) for name in dirs + files]
    return (f"훅의 임시 디렉터리를 다 지우지 못함 — {work} 가 남음"
            + (f" (항목 {len(left)}개, 예: {left[-1]})" if left else ""))


class HookRun(tuple):
    """What run_hook returns: ``(returncode, stdout, stderr)`` -- a tuple, so that a caller or a stand-in that unpacks
    three values is unchanged -- and ``notes``, the sentences saying what the cleanup after the hook could not do,
    for C9 to report; ``group_left`` among them: the group it ended was not seen to end after SIGKILL. ``stalled``:
    the code is None because the hook stopped writing (REPRO_STALL_S), not because its time ran out."""

    notes: "tuple[str, ...]" = ()
    group_left = False
    stalled = False

    def __new__(cls, returncode: "int | None", stdout: str, stderr: str, notes: "Iterable[str]" = (),
                group_left: bool = False, stalled: bool = False) -> "HookRun":
        run = super().__new__(cls, (returncode, stdout, stderr))
        run.notes = tuple(notes)
        run.group_left = group_left
        run.stalled = stalled
        return run


def _newest_write(directory: str) -> "float | None":
    """The latest modification time of any file under ``directory``; None when it holds none. The batch runner's
    newest_write (generation/run_batch.py), copied: this validator does not import generation/."""
    newest = None
    for dirpath, _, names in os.walk(directory):
        for name in names:
            try:
                mtime = os.stat(os.path.join(dirpath, name)).st_mtime
            except OSError:
                continue
            if newest is None or mtime > newest:
                newest = mtime
    return newest


_TIMED_OUT = "timed out"
_STALLED = "stalled"


def _await_hook(p: "subprocess.Popen[str]", work: str, timeout_s: float) -> "tuple[str, str] | str":
    """Wait for the hook: its output, or why it was given up -- _TIMED_OUT once ``timeout_s`` is over, _STALLED once
    nothing under ``work`` (its TMPDIR) was written for REPRO_STALL_S, not judged during the first
    REPRO_STALL_GRACE_S. In slices of REPRO_STALL_POLL_S; retrying communicate() after TimeoutExpired loses no
    output, and a signal is raised where it lands, as in one long wait."""
    started = time.time()
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return p.communicate(timeout=max(0.0, min(REPRO_STALL_POLL_S, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            if time.monotonic() >= deadline:
                return _TIMED_OUT
            now = time.time()
            if now - started < REPRO_STALL_GRACE_S:
                continue
            last = _newest_write(work)
            if now - (last if last is not None else started) > REPRO_STALL_S:
                return _STALLED


def _communicate_within(p: "subprocess.Popen[str]", until: float,
                        cut: "Callable[[], bool]") -> "tuple[str, str] | None":
    """``p.communicate()`` until the monotonic time ``until`` or ``cut()``: the output, or None if it was not complete
    by then. In slices, so that a signal held meanwhile is seen at once; retrying after TimeoutExpired loses no
    output."""
    while True:
        try:
            return p.communicate(timeout=max(0.0, min(0.1, until - time.monotonic())))
        except subprocess.TimeoutExpired:
            if cut() or time.monotonic() >= until:
                return None


def _end_hook_group(p: "subprocess.Popen[str]", cut: "Callable[[], bool]" = _never) -> tuple[str, str, bool]:
    """End the process group ``p`` leads -- SIGTERM, then SIGKILL for whatever of it still runs once
    REPRO_KILL_GRACE_S is over, or once ``cut()`` says so (a second Ctrl+C, held) -- and return what the hook
    wrote, and whether the group was seen to end. The SIGKILL is in a ``finally``: an exception during the grace must not skip it, and ends the grace at
    once. Collecting the output is not the end: the leader exits and the pipes close while a member that ignores
    SIGTERM and has closed its pipes runs on, so the group itself is waited for. A group that is already gone still
    has to be collected: that reaps the child and binds the output the caller returns."""
    grace_over = time.monotonic() + REPRO_KILL_GRACE_S
    out = None
    interrupted = True
    try:
        try:
            os.killpg(p.pid, signal.SIGTERM)
            out = _communicate_within(p, grace_over, cut)
        except ProcessLookupError:
            pass                        # nothing left to signal
        interrupted = False
    finally:
        gone = _end_what_is_left(p, time.monotonic() if interrupted else grace_over, cut)
    out, err = out if out is not None else _collect_hook_output(p)
    return out, err, gone


class _HookSignals:
    """While run_hook runs the hook: SIGTERM and SIGHUP become an exit (128 + the signal number) so that the hook's
    group is ended on the way out, and SIGINT stays a KeyboardInterrupt. One is raised only while run_hook waits for
    the hook. While the hook is being launched, and while run_hook cleans up after it -- ends its group, removes its
    TMPDIR, puts the handlers back -- a signal is held, and acted on once that is over: handled inside Popen it would
    raise before ``p`` is bound and the group's end would be skipped; raised inside the cleanup it would
    skip the SIGKILL, the group's end, the removal or the rest of the handlers. The
    first signal wins: once one has been raised, or while one is held, later ones are dropped -- they ask for the same
    exit. A signal held during the SIGTERM grace still ends the grace at once (``held``), as a second Ctrl+C raised
    there would.

    The hook is in a session of its own, so a `kill`, `timeout` or SSH hangup aimed at the validator does not
    reach it, and Python dies of the signal without an exception -- the simulator would run on. Only a
    handler that is still the default is replaced: a validator started under `nohup` has SIGHUP ignored on
    purpose, and a caller's own handler is the caller's. Handlers can only be installed from the main thread;
    elsewhere this does nothing.

    Held in the handler, not by blocking the signal: the signal mask is per thread, and a signal sent to the
    process goes to any thread that has it unblocked -- pyarrow's pool threads, there once C1-C8 have read the
    parquet files (29 threads on the development machine after one read, 2026-09-30) -- while CPython runs the handler in the
    main thread all the same. The hook inherits no mask from this."""

    def __init__(self) -> None:
        self.holding = True             # the launch
        self._pending: int | None = None
        self._previous: dict[int, Any] = {}

    def install(self) -> None:
        """Put the handlers in -- called inside run_hook's ``try``, and each one recorded as it goes in, so that
        restore() puts back whatever was replaced even if installing stops halfway."""
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGTERM, signal.SIGHUP):
            if signal.getsignal(sig) == signal.SIG_DFL:
                self._previous[sig] = signal.SIG_DFL
                signal.signal(sig, self._handle)
        if signal.getsignal(signal.SIGINT) is signal.default_int_handler:
            self._previous[signal.SIGINT] = signal.default_int_handler
            signal.signal(signal.SIGINT, self._handle)

    @staticmethod
    def _act(signum: int) -> BaseException:
        return KeyboardInterrupt() if signum == signal.SIGINT else SystemExit(128 + signum)

    def _handle(self, signum: int, frame: Any) -> None:
        if self.holding:
            if self._pending is None:
                self._pending = signum
            return
        self.holding = True             # raised once: what follows is the cleanup
        raise self._act(signum)

    def launched(self) -> None:
        """The launch is over -- ``p`` bound, or Popen raised: a signal held meanwhile acts now, and until the
        cleanup begins a signal is raised where it lands."""
        self.holding = False
        self.release()

    def hold(self) -> None:
        """The cleanup begins: a signal from here on is held."""
        self.holding = True

    @property
    def held(self) -> bool:
        return self._pending is not None

    def release(self) -> None:
        """Act on the signal held meanwhile, if one was."""
        if self._pending is not None:
            signum, self._pending = self._pending, None
            self.holding = True
            raise self._act(signum)

    def restore(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)


def run_hook(argv: list[str], env: dict[str, str], timeout_s: float) -> HookRun:
    """Run the C9 hook in its own process group; return (returncode, stdout, stderr), with the notes of what the
    cleanup after it could not do (HookRun). On the way out by an exception -- a signal to the validator -- there is
    no result to carry them, and they go to stderr.

    A hook still running after ``timeout_s`` has its whole group ended and returns a code of None.
    subprocess.run(timeout=) kills only the direct child -- bash -- so the simulator the hook started
    would go on running, sharing the GPU with the next sample, and the hook's EXIT trap would never run, leaving
    its work directory (~1.4-1.5 GB per seven-camera episode) behind. The group gets SIGTERM so the trap runs
    (rerun_seed.sh turns TERM into an exit), and SIGKILL for whatever of it is still there after the grace --
    the leader's exit is not the group's end. A hook that returns an exit status has
    ended as it meant to, and what it leaves running is left alone: on the development machine Kit starts the Omniverse Hub, a
    shared cache daemon another Isaac session may be using, in the hook's group whenever none runs.
    The exit status decides, not when a signal comes: one to the validator after the hook has returned its status
    leaves the group alone too (it is acted on after the cleanup). A hook killed by a signal -- a negative
    returncode -- has not ended as it meant to, and its group is ended as on a timeout.

    A session of its own also means that no signal aimed at the validator reaches the hook -- neither the
    terminal's Ctrl+C nor a SIGTERM or SIGHUP -- so an interrupt, and a termination signal turned into an
    exit (_HookSignals), end the group here too, then propagate -- one that arrives while the hook is being
    launched included. The cleanup -- ending the group, removing the TMPDIR, putting the handlers back -- is one
    ``finally`` that decides from what happened whether to end the group, and runs with signals held: a signal
    that lands in it is acted on once it is over.

    The hook's TMPDIR is a directory made here, where its own `mktemp -d` would have gone (``env``'s TMPDIR,
    else the system default), and removed once the hook has ended, however it ended: a group still ignoring
    SIGTERM when the grace is over dies of SIGKILL, and no trap of the hook's runs to remove its work directory.
    The rest of ``env`` is handed over as it is."""
    notes: list[str] = []
    signals = _HookSignals()
    try:
        return _run_hook(argv, env, timeout_s, notes, signals)
    except BaseException as e:
        for note in notes:
            print(f"경고: {note}", file=sys.stderr)
        if isinstance(e, Exception):
            # An error started the cleanup (an OSError out of the wait), and the cleanup held signals all the same:
            # one held meanwhile outranks the error, which it carries as its context. Dropped with the error, C9
            # would take it for a failed run and go on to the next sample.
            # A cleanup that a signal started keeps that first signal.
            signals.release()
        raise


def _run_hook(argv: list[str], env: dict[str, str], timeout_s: float, notes: list[str],
              signals: "_HookSignals") -> HookRun:
    """run_hook's body; ``notes`` gathers what the cleanup could not do as it goes, and a signal held at the end of a
    cleanup that an exception started is left in ``signals`` for run_hook to act on."""
    work = None
    p = None
    timed_out = stalled = False
    group_left = False
    out = err = None
    try:
        try:
            signals.install()
            work = os.path.abspath(tempfile.mkdtemp(prefix="c9-hook-", dir=env.get("TMPDIR") or None))
            # Decoded with replacement: one byte that is not UTF-8 (a simulator log line) would raise out of
            # communicate() in place of the hook's exit, or of the signal being acted on.
            # UTF-8 as _text decodes what a TimeoutExpired carries.
            p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
                                 errors="replace", env={**env, "TMPDIR": work}, start_new_session=True)
        finally:
            signals.launched()
        # A stall ends the hook as its time running out does.
        waited = _await_hook(p, work, timeout_s)
        if isinstance(waited, str):
            timed_out, stalled = True, waited == _STALLED
        else:
            out, err = waited
    finally:
        # The hold is alone in a try whose finally is the rest: a signal that lands as it is entered, before it takes
        # effect, is raised there (it is the first; the handler holds from then on) and the whole cleanup still runs.
        # A signal raised anywhere in the wait -- on a JUMP_BACKWARD of _await_hook's polling loop, where CPython 3.12
        # checks for signals, or inside its directory walk -- leaves _await_hook as an exception and enters this same
        # cleanup by its exception path, with no status yet, so the group is ended.
        try:
            signals.hold()
        finally:
            try:
                # A hook that returned an exit status has ended as it meant to: what it left in its group is left
                # alone. Anything else ends the group -- a timeout, a signal or an error while waiting (no status
                # yet), a leader killed by a signal (a negative status: the OOM killer, a crash).
                if p is not None and (timed_out or p.returncode is None or p.returncode < 0):
                    ended_out, ended_err, gone = _end_hook_group(p, lambda: signals.held)
                    if timed_out:
                        out, err = ended_out, ended_err
                    if not gone:
                        group_left = True
                        notes.append(f"훅의 프로세스 그룹({p.pid})이 SIGKILL 뒤 {REPRO_COLLECT_TIMEOUT_S:g} 초 안에 "
                                     "끝나지 않음 — 끝났는지 확인하지 못함 (끊을 수 없는 대기(D)에 든 구성원일 수 있음)")
            finally:
                try:
                    left = _remove_hook_tmpdir(work) if work is not None else None
                    if left:
                        notes.append(left)
                finally:
                    signals.restore()
    signals.release()                   # a signal held during the cleanup, now that it is over
    return HookRun(None if timed_out else p.returncode, out or "", err or "", notes, group_left, stalled)


def repro_hook_finding(returncode: int, stdout: str | None, stderr: str | None) -> tuple[str, str]:
    """(severity, message) for one C9 hook run.

    The hook's stdout is the comparison's report -- the repository hook sends the simulator's
    stdout to /dev/null, so what reaches stdout is compare_published's output. On success its
    last line says what was compared (columns, frames, the pack of a camera distribution whose
    canon declares one, and that RGB was not), and the finding carries it so the report says so
    too. On failure its
    tail is the reason ("팩 head.depth: 값 불일치"), and it comes first: stderr always holds
    something -- the hook names the distribution it compares against, then the simulator logs
    -- so preferring stderr, as this did until 2026-09-29, hid every comparison failure behind
    a log tail. stderr's tail follows the reason, and stands alone when the hook died before
    comparing (stdout empty). Tails, never heads: a head is the simulator's boot log, and the
    first 200 characters once held nothing but two gRPC warnings."""
    out, err = (stdout or "").strip(), (stderr or "").strip()
    if returncode != 0:
        if out:
            detail = f"…{out[-400:]}" + (f" (훅 stderr 끝: …{err[-200:]})" if err else "")
        else:
            detail = f"…{err[-400:]}"
        return ERROR, f"재현 실패 (exit {returncode}): {detail}"
    last = out.splitlines()[-1].strip() if out else ""
    return INFO, f"시드 재현 확인 — {last}" if last else "시드 재현 확인"


# ────────────────────────────────────────────────────────────────
# 실행
# ────────────────────────────────────────────────────────────────

ALL_CHECKS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]


def load_frames(ds: Dataset, rep: Report, limit: int | None) -> list[tuple[Path, pd.DataFrame]]:
    """모든 프레임을 한 번에 들고 있습니다. v0.1 규모에서는 그래도 됩니다.

    C7·C8 이 에피소드를 가로질러 대조하므로 스트리밍으로 바꾸려면 검사부터
    다시 써야 합니다. 실측(2026-08-25): 10 에피소드 9,990 프레임에 294 MB,
    40 에피소드 38,408 프레임에 435 MB — 프레임당 약 5 KB 입니다. 400 에피소드
    약 40만 프레임이면 2 GB 남짓이고, 그보다 커지면 `--file-limit` 로 나눠
    읽습니다. 그때는 전역 대조가 아래 `frames_truncated` 를 보고 「안 읽은
    파일의 에피소드」를 「없는 에피소드」로 보고하지 않습니다.
    """
    out: list[tuple[Path, pd.DataFrame]] = []
    files = ds.data_files()
    # 일부만 읽었다는 사실을 데이터셋에 남긴다. 전역 대조 검사는 이걸 봐야 「안 읽은
    # 파일에 있는 에피소드」를 「없는 에피소드」로 보고하지 않는다.
    ds.frames_truncated = bool(limit) and len(files) > limit
    if limit:
        files = files[:limit]
    for f in files:
        try:
            out.append((f.relative_to(ds.root), pd.read_parquet(f)))
        except Exception as e:  # noqa: BLE001
            rep.add("C1", ERROR, f"parquet 읽기 실패: {e}", str(f.relative_to(ds.root)))
    return out


def run(root: Path, checks: set[str], repro_hook: str | None,
        repro_sample: int, file_limit: int | None,
        repro_max_seq: int | None = None) -> Report:
    rep = Report(root=str(root))
    ds = Dataset(root, rep)

    if not root.is_dir():
        rep.add("C0", ERROR, "데이터셋 루트가 디렉터리가 아님", str(root))
        return rep

    eps = ds.episodes()
    if eps is None:
        rep.add("C1", ERROR, "에피소드 메타를 찾을 수 없음 (meta/episodes*)", "meta/")
    frames = load_frames(ds, rep, file_limit)

    rep.stats["n_episodes_meta"] = 0 if eps is None else int(len(eps))
    rep.stats["n_data_files"] = len(frames)
    rep.stats["n_steps"] = int(sum(len(df) for _, df in frames))

    if "C1" in checks: c1_layout(ds, rep, eps, frames)
    if "C2" in checks: c2_joint_dims(ds, eps, rep)
    if "C3" in checks: c3_contact_regions(ds, frames, rep)
    if "C4" in checks: c4_label_evidence(ds, eps, rep)
    if "C5" in checks:
        c5_repro_meta(ds, eps, rep)
        c5_table_fields(ds, eps, rep)
    if "C6" in checks: c6_asset_license(ds, eps, rep)
    if "C7" in checks:
        c7_timeseries(frames, eps, rep, ds.frames_truncated)
        c7_step_nulls(ds, frames, rep)
        c7_table_files(ds, rep)
        c7_step_features(ds, rep)
        c7_info_totals(ds, rep)
        c7_task_table(ds, rep)
        c7_data_spans(ds, frames, eps, rep)
        c7_camera_alignment(ds, frames, eps, rep)
    if "C8" in checks: c8_values(ds, frames, eps, rep)
    if "C9" in checks: c9_seed_repro(ds, eps, repro_hook, repro_sample, rep, repro_max_seq)
    rep.requested = set(checks)

    return rep


CHECK_TITLES = {
    "C1": "레이아웃·필수 파일",
    "C2": "관절 차원·매핑 정합",
    "C3": "접촉 부위 코드",
    "C4": "라벨-근거 정합",
    "C5": "재현 메타·enum",
    "C6": "자산 라이선스·NOTICE",
    "C7": "시계열 무결성",
    "C8": "값 범위·이상치",
    "C9": "시드 재현성",
}


def print_report(rep: Report, verbose: bool, ran: set[str] | None = None) -> None:
    print("=" * 78)
    print(f"AI 휴머노이드 안전성 데이터셋 검증  (스키마 v{SCHEMA_VERSION})")
    print(f"대상: {rep.root}")
    print("=" * 78)

    s = rep.stats
    print(f"에피소드 {s.get('n_episodes_meta', 0)}개 / "
          f"데이터 파일 {s.get('n_data_files', 0)}개 / 스텝 {s.get('n_steps', 0)}행")
    if "outcome_distribution" in s:
        d = s["outcome_distribution"]
        tot = sum(d.values()) or 1
        parts = "  ".join(f"{k} {v}({v/tot:.0%})" for k, v in sorted(d.items()))
        grid = s.get("grid_episodes_not_counted")
        print(f"outcome 분포: {parts}" + (f"  (격자 행 {grid}건은 세지 않음)" if grid else ""))
    print("-" * 78)

    by = rep.by_check()
    for c in ALL_CHECKS:
        counts = by.get(c, {})
        if not counts:
            continue
        if counts.get(ERROR):
            mark, tail = "✗", f"ERROR {counts[ERROR]}"
        elif counts.get(WARN):
            mark, tail = "!", f"WARN {counts[WARN]}"
        elif counts.get(SKIP):
            mark, tail = "-", "SKIP"
        else:
            mark, tail = "✓", "OK"
        print(f" {mark}  {c} {CHECK_TITLES.get(c, ''):<18} {tail}")
    # 돌리지 않은 검사를 OK 로 찍으면 안 된다. `--checks C1` 하나만 줘도 C2~C9 가
    # 전부 초록으로 보였고 exit 0 이었다 — 확인하지 않은 것을 확인한 것처럼 말하는
    # 가장 직접적인 형태다.
    for c in ALL_CHECKS:
        if c not in by:
            if ran is not None and c not in ran:
                print(f" ·  {c} {CHECK_TITLES.get(c, ''):<18} 미실행 (--checks)")
            else:
                print(f" ✓  {c} {CHECK_TITLES.get(c, ''):<18} OK")

    shown = [f for f in rep.findings if verbose or f.severity in (ERROR, WARN)]
    if shown:
        print("-" * 78)
        # 400 에피소드 배포본에서는 같은 문제가 에피소드마다 하나씩 붙어 수백 줄이
        # 된다. 그대로 앞 60줄만 찍으면 **경고가 오류를 가려** 정작 무엇이 배포를
        # 막는지 보이지 않는다. 그래서 (1) 오류를 먼저,
        # (2) 같은 검사·같은 메시지는 한 줄로 접고 건수와 예시 위치를 붙인다.
        order = {ERROR: 0, WARN: 1, INFO: 2, SKIP: 3}
        grouped: dict[tuple, list] = {}
        for f in shown:
            grouped.setdefault((order.get(f.severity, 9), f.severity, f.check,
                                f.message), []).append(f.location)
        rows = sorted(grouped.items(), key=lambda kv: kv[0][:3])
        for (_, severity, check, message), locations in rows[: (None if verbose else 60)]:
            # 위치를 하나만 찍으면 접힌 줄이 「어디들인가」를 통째로 삼킨다 —
            # 필수 파일 3개가 없을 때 파일 이름 둘이 사라졌다.
            # 기본은 셋까지, --verbose 는 전부 보여 준다.
            spots = [x for x in locations if x]
            uniq = list(dict.fromkeys(spots))
            if len(locations) > 1:
                if not uniq:
                    where = f"  [{len(locations)}건]"
                else:
                    head = uniq if verbose else uniq[:3]
                    more = "" if len(head) == len(uniq) else f" 외 {len(uniq) - len(head)}"
                    where = f"  [{len(locations)}건, 예: {', '.join(head)}{more}]"
            else:
                where = f"  [{spots[0]}]" if spots else ""
            print(f"{severity:<5} {check}  {message}{where}")
        if not verbose and len(rows) > 60:
            print(f"... 외 {len(rows) - 60}종 (--verbose 로 전체 출력)")

    print("=" * 78)
    print(f"ERROR {rep.count(ERROR)}  WARN {rep.count(WARN)}  "
          f"INFO {rep.count(INFO)}  SKIP {rep.count(SKIP)}")


def main(argv: list[str] | None = None) -> int:
    # The fixture regression has died with SIGSEGV twice (2026-08-28, 09-01) and
    # neither run left a stack. faulthandler writes the Python stack to stderr
    # on a fatal signal, which is the only evidence such a crash can leave.
    try:
        faulthandler.enable()
    except (ValueError, OSError, AttributeError):
        # stderr without a file descriptor (in-process test capture): nothing a
        # signal handler could write to, so there is nothing to enable.
        pass
    ap = argparse.ArgumentParser(description="AI 휴머노이드 안전성 데이터셋 검증기")
    ap.add_argument("root", type=Path, help="데이터셋 루트 디렉터리")
    ap.add_argument("--json", type=Path, help="검증 리포트 JSON 저장 경로")
    ap.add_argument("--checks", type=str, help="수행할 검사 (예: C2,C4). 기본 전체")
    ap.add_argument("--strict", action="store_true", help="WARN 도 실패로 처리")
    ap.add_argument("--verbose", action="store_true", help="INFO/SKIP 포함 전체 출력")
    ap.add_argument("--repro-hook", type=str,
                    help="C9 재현 검사 실행 파일. 인자로 (episode_id, seed, run_start_seq) 전달")
    ap.add_argument("--repro-sample", type=int, default=3,
                    help="C9 표본 수 (기본 3) — 시나리오가 둘 이상이면 시나리오마다 하나씩 먼저 고름")
    ap.add_argument("--repro-max-seq", type=int, default=None,
                    help=f"C9 표본을 고를 접두 범위 (기본: meta/cameras.json 을 싣는 카메라 배포본 "
                         f"{REPRO_MAX_SEQ_CAMERAS}, 그 밖 {REPRO_MAX_SEQ}). 재현 비용이 그 에피소드가 "
                         "자기 실행(run_start_seq 부터)에서 몇 번째인가에 비례하므로 상한이 필요하고, "
                         "카메라 배포본은 에피소드당 비용이 커서 "
                         f"훅 제한 시간 {REPRO_HOOK_TIMEOUT_S} 초에 맞춰 더 낮다")
    ap.add_argument("--file-limit", type=int, help="읽을 parquet 파일 수 제한 (빠른 점검용)")
    a = ap.parse_args(argv)

    checks = set(ALL_CHECKS)
    if a.checks:
        req = {c.strip().upper() for c in a.checks.split(",") if c.strip()}
        bad = req - set(ALL_CHECKS)
        if bad:
            print(f"알 수 없는 검사 ID: {sorted(bad)}", file=sys.stderr)
            return 3
        checks = req

    try:
        rep = run(a.root, checks, a.repro_hook, a.repro_sample, a.file_limit,
                  a.repro_max_seq)
    except Exception as e:  # noqa: BLE001
        # Exit 3 means the validator itself broke, not that the dataset is
        # defective. Print the full traceback: without it an intermittent crash
        # (two SIGSEGVs and one garbage IndexError so far, all unreproducible
        # from the message alone) leaves nothing to investigate.
        print(f"검증기 실행 실패: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 3

    # Findings quote the release, and JSON can carry a lone surrogate ("\ud800") that UTF-8 cannot
    # encode: it is printed escaped and written as the JSON escape it came as (backslashreplace
    # writes exactly that). A report that still cannot be made is the validator failing (exit 3),
    # not the release holding an ERROR (exit 1).
    try:
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(errors="backslashreplace")
        print_report(rep, a.verbose, checks)
        if a.json:
            a.json.parent.mkdir(parents=True, exist_ok=True)
            a.json.write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8", errors="backslashreplace")
            print(f"\n리포트 저장: {a.json}")
    except Exception as e:  # noqa: BLE001
        print(f"검증기 보고 실패: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 3

    if rep.count(ERROR):
        return 1
    if a.strict and rep.count(WARN):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
