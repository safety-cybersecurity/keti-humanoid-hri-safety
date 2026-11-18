"""npz + sidecar + labels -> LeRobotDataset v3.0.

The two pure functions here own the whole schema mapping and import without
lerobot, so the field contract is tested in .venv rather than only in
.venv-lerobot. generation/field_map.py is the map they follow; the C9 hook
follows the same one, and generation/assets.py does the licence join that
run_s1a.py already pre-checked before the batch ran.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

# Run as a script, sys.path[0] is this file's directory, not the repo root, so
# the `generation.*` imports below fail. run_s1a.py already carries the same
# guard; without it the documented commands need a PYTHONPATH the
# documentation never mentions (measured: all four failed).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generation import provenance
from generation.assets import (asset_entry, check_shipped_sources, required_scene_assets,
                               sha256_file, shipped_asset_files)
from generation.directed_separation import DIRECTED_SSM_RULE_ID, scenario_declares_directed_ssm
from generation.field_map import (CAMERAS, HRI, STEP_FIELDS, active_step_fields, hri_recorded,
                                  step_dt_from_timestamps)
from generation.release_canon import release_canon

SCHEMA_VERSION = "0.2"


def build_episode_meta(meta: dict, label: dict, scenarios: dict, joint_map: dict,
                       assets: dict, rules: dict, step_dt: float) -> dict:
    """The schema §5 episode record for one episode.

    ``step_dt`` is passed in rather than read off the sidecar: the sidecar's
    global_time_offset_s is the episode's start time on the global clock, not
    a period, and only looks like one for episode 0 (see
    field_map.step_dt_from_timestamps).
    """
    sid = meta["scenario_id"]
    scenario = next((s for s in scenarios["scenarios"] if s["scenario_id"] == sid), None)
    if scenario is None:
        raise ValueError(f"meta/scenarios.json 에 없는 시나리오입니다: {sid}")
    # Named explicitly rather than left to a KeyError: safety_function is set
    # for S1a alone, so every other scenario hits this the first time it is
    # converted, and the message has to say what to do about it.
    absent = [k for k in ("safety_function", "domain", "task_archetype", "hazard_class")
              if k not in scenario]
    if absent:
        raise ValueError(
            f"{sid}: meta/scenarios.json 에 필수 필드가 없습니다: {absent} — "
            "스키마 §5 의 필수 에피소드 필드이며, 시나리오마다 사람이 정해야 합니다 "
            "(scenarios.json 의 safety_function_note 참고)"
        )
    model = meta["robot_model"]
    robot = (assets.get("robots") or {}).get(model)
    if robot is None:
        raise ValueError(f"meta/assets.json 에 로봇 {model} 이 없습니다")
    if joint_map["robot_model"] != model:
        raise ValueError(
            f"관절 정본이 다른 로봇입니다: {joint_map['robot_model']} vs {model}"
        )

    by_rule = {r["rule_id"]: r for r in rules.get("rules", [])}
    refs = sorted({
        by_rule[rid]["standard"]
        for sel in (scenario.get("rules") or [])
        for rid in by_rule
        if fnmatch.fnmatch(rid, sel) and by_rule[rid].get("standard")
    })

    return {
        "episode_id": meta["episode_id"],
        # Which units `action` carries. The recorder declares it rather than the
        # converter inferring it, because the recorder is what chose the value.
        "action_space": meta["action_space"],
        # 이 에피소드가 왜 끝났는가. 사이드카에만 있고 배포본에는 없어서, 받는 쪽이
        # 「로봇이 넘어져서 끝난 에피소드」를 고를 수 없었다. degenerate 는 그
        # 에피소드를 쓰지 말라는 표시다.
        "termination": require_termination(meta),
        # bool() 이 아니라 형 검사다. bool("false") 는 True 이므로, 사이드카가
        # 문자열을 담았다면 조용히 뒤집힌 값이 배포된다.
        "degenerate": require_type(meta, "degenerate", bool),
        "schema_version": SCHEMA_VERSION,
        "domain": scenario["domain"],
        "source": "sim",
        "robot_model": model,
        "robot_embodiment": robot["embodiment"],
        "dof": joint_map["dof"],
        "joint_names": list(joint_map["joint_names"]),
        "scenario_id": sid,
        "task_archetype": scenario["task_archetype"],
        "hazard_class": scenario["hazard_class"],
        "outcome": label["outcome"],
        "severity": label["severity"],
        "triggered_rules": list(label["triggered_rules"]),
        "safety_function": scenario["safety_function"],
        "standard_refs": refs,
        "measurement_method": meta["sim_backend"]["physics"],
        # The canonical paths, not asset_paths_resolved -- the resolved URL is
        # provenance and would not join against the licence canon.
        "scene_assets": [asset_entry(assets, p) for p in meta["asset_paths"].values()],
        "sim_backend": meta["sim_backend"],
        "randomization_seed": meta["seed"],
        "run_start_seq": require_run_start(meta),
        "fps": round(1.0 / step_dt),
        "duration_s": meta["steps"] * step_dt,
        "human_model": scenario["scene"]["human_model"],
        # Everything a reader needs to not over-read the outcome: which SSM
        # parameters were provisional, which rules could not be evaluated at
        # all, and whether the robot actually touched the dummy despite a
        # nominal label (see label_episodes on why those can disagree).
        "notes": json.dumps({
            "provisional": meta.get("provisional"),
            "not_evaluated": label.get("not_evaluated", []),
            "contact_observed": label.get("contact_observed"),
            "contact_detection_limits": label.get("contact_detection_limits"),
            # 홀드 서보가 접촉과 겹친 스텝 수. 해제 게이트는 표면 대 표면 거리를
            # 못 쓰므로 최선노력이고, 접촉 백스톱이 손해를 「접촉을 처음 본 스텝」
            # 하나로 묶는다. 그 스텝의 contact_force 에는 서보 힘이 섞여 있으므로
            # 받는 쪽이 알아야 한다 — 사이드카에만 있어서 볼 수 없었다.
            "hold_contaminated_steps": meta.get("hold_contaminated_steps"),
            # 이 스텝부터 더미가 자유 물체다. 그 이전 구간의 접촉력은 인체를
            # 세워 두는 외력과 함께 읽어야 한다.
            "hold_release_step": meta.get("hold_release_step"),
            # 회랑을 만든 실제 입력. 공개 rules.json 은 intrusion_distance 를
            # 조항 참조로만 싣고(그 표준이 상수로 주지 않는다), 이것이 없으면
            # 받는 쪽은 separation_required 가 어떤 값에서 나왔는지 복원할 수
            # 없다 — 사이드카에만 있었다.
            # human_speed 는 2026-09-01 에 값을 실었고 이 배치가 쓴 값과 같다.
            "ssm_params": meta.get("ssm_params"),
            # 인체 부위 반경. separation_required 와 min_distance_human 을
            # 대조하려면 필요하다.
            "human_region_radius": meta.get("human_region_radius"),
            # 어느 접촉 경로가 잡히지 않는가. 카드가 이 두 필드를 읽으라고
            # 지시하는데 배포본에 없었다.
            "contact_onset_unresolved_regions": meta.get("contact_onset_unresolved_regions"),
            "contact_onset_unresolved_reason": meta.get("contact_onset_unresolved_reason"),
            "contact_type_unresolved_reason": meta.get("contact_type_unresolved_reason"),
            # 이 에피소드에서 **실제로 뽑힌** 랜덤화 값. 시나리오 정본은 범위를 적을
            # 뿐이라 배포본에 seed 밖에 없으면 실제 분포를 복원할 수 없었다.
            # 2026-10-03 전의 정본은 구현과 다른 범위 둘을 선언하고 있었고, 그 뒤로는
            # 환경 설정에서 읽은 범위를 적는다 (generation/randomization.py).
            "randomization": meta.get("randomization"),
            "imu_link": meta.get("imu_link"),
            "asset_paths_resolved": meta.get("asset_paths_resolved"),
            # The first step the robot stood at the script's goal place (-1: not done in time;
            # null: the scenario has no completion criterion). The scorecard's task column reads it.
            "task_completed_step": meta.get("task_completed_step"),
            # What the hri recorder keeps beside the step table (null for a scenario with no script): why the episode
            # is degenerate (null when it is not), the step the held box was let go (-1: it never was, or none was
            # carried) and the steps each robot camera's lens sat inside that box.
            "degenerate_reason": meta.get("degenerate_reason"),
            "box_released_step": meta.get("box_released_step"),
            "camera_in_box_steps": meta.get("camera_in_box_steps"),
            # The task itself (null for a scenario with no script): the stages it ends in with the step each
            # ended at, the person's stature and the task's time limit.
            "task_stages": meta.get("task_stages"),
            "human_stature_m": meta.get("human_stature_m"),
            "task_time_limit_s": meta.get("task_time_limit_s"),
        }, ensure_ascii=False),
        # A scenario with a conditions table: the episode's spec, its evidence and its place in the table.
        **table_columns(meta, label, scenario, assets, joint_map),
    }


# A random row's grid cells: the column is an integer, and a random row has no cell.
GRID_NOT_A_CELL = -1


def _table_problems(meta: dict, label: dict) -> list[str]:
    """What an episode of a scenario with a conditions table lacks of what its table columns come from: the
    condition record's design and grid cell (randomization.condition -- a grid row's cell is two counts, a random
    row's null), and the label's evidence, one number or null for each name of evidence.NAMES (inf counts; NaN is not
    a value -- the column writes no value as NaN)."""
    from generation.labeling import evidence

    problems: list[str] = []
    record = (meta.get("randomization") or {}).get("condition") if isinstance(meta.get("randomization"), dict) \
        else None
    if not isinstance(record, dict):
        problems.append("사이드카에 조건 기록(randomization.condition)이 없습니다")
    else:
        design, cell = record.get("design"), record.get("grid_cell", ...)
        if design not in ("random", "grid"):
            problems.append(f"조건 기록의 design 이 random · grid 가 아닙니다: {design!r}")
        elif cell is ...:
            problems.append("조건 기록에 grid_cell 이 없습니다")
        elif design == "random" and cell is not None:
            problems.append(f"무작위 행의 조건 기록에 grid_cell 이 있습니다: {cell!r}")
        elif design == "grid" and not (isinstance(cell, list) and len(cell) == 2
                                        and all(type(c) is int and c >= 0 for c in cell)):
            problems.append(f"격자 행의 조건 기록의 grid_cell 이 0 이상의 정수 둘이 아닙니다: {cell!r}")
    found = label.get(evidence.EVIDENCE_KEY)
    if not isinstance(found, dict):
        problems.append("라벨에 근거 요약(evidence)이 없습니다")
    else:
        if list(found) != list(evidence.NAMES):
            problems.append(f"라벨의 근거 요약의 이름이 {list(evidence.NAMES)} 가 아닙니다 — 없는 것 "
                            f"{[n for n in evidence.NAMES if n not in found]}, 남는 것 "
                            f"{[n for n in found if n not in evidence.NAMES]}")
        wrong = {n: v for n, v in found.items() if v is not None and (
            isinstance(v, bool) or not isinstance(v, (int, float)) or v != v)}
        if wrong:
            problems.append(f"라벨의 근거 요약에 수나 null 이 아닌 값이 있습니다: {wrong}")
    return problems


def table_columns(meta: dict, label: dict, scenario: dict, assets: dict, joint_map: dict) -> dict:
    """The episode columns of a scenario that declares a conditions table (field_map's TABLE fields, in their order):
    episode_spec (episode_spec.build_episode_spec), the evidence of the label (evidence.<name>; a value that is not
    there is NaN, which pandas writes as a double's null), condition_design and the two grid cells (GRID_NOT_A_CELL
    for a random row). {} for a scenario without a table."""
    from generation.episode_spec import build_episode_spec
    from generation.field_map import TABLE
    from generation.labeling import evidence

    if TABLE not in scenario:
        return {}
    problems = _table_problems(meta, label)
    if problems:
        raise ValueError(f"{meta.get('episode_id')}: " + "; ".join(problems))
    record = meta["randomization"]["condition"]
    robot, human = record["grid_cell"] or (GRID_NOT_A_CELL, GRID_NOT_A_CELL)
    found = label[evidence.EVIDENCE_KEY]
    return {
        "episode_spec": build_episode_spec(meta, scenario, assets, dof=joint_map["dof"]),
        **{evidence.PREFIX + name: (float("nan") if found[name] is None else float(found[name]))
           for name in evidence.NAMES},
        "condition_design": record["design"],
        "grid_cell_robot": robot,
        "grid_cell_human": human,
    }


def require_type(meta: dict, key: str, want: type):
    """사이드카 값이 기대한 형인지 확인하고 그대로 돌려준다.

    강제 변환하지 않는다 -- `bool("false")` 는 True 라, 형이 어긋난 사이드카는
    거부되는 대신 **뒤집힌 값이 배포**된다.
    """
    value = meta[key]
    if not isinstance(value, want) or isinstance(value, bool) is not (want is bool):
        raise ValueError(
            f"에피소드 메타 {key} 의 형이 {want.__name__} 이 아닙니다: {value!r}"
        )
    return value


def require_run_start(meta: dict) -> int:
    """The first seq of the run that recorded this episode (schema §3), checked against the
    episode's own seq: a run cannot have produced an episode numbered before its start."""
    if "run_start_seq" not in meta:
        raise ValueError(
            f"{meta.get('episode_id')}: 사이드카에 run_start_seq 가 없습니다 — 실행의 첫 번호를 "
            "적기 전의 생성기로 만든 녹화입니다. 다시 생성하십시오 (규격서 §3)")
    start = require_type(meta, "run_start_seq", int)
    seq = require_type(meta, "episode_seq", int)
    if not 0 <= start <= seq:
        raise ValueError(
            f"{meta.get('episode_id')}: run_start_seq={start} 가 0 과 자기 번호 {seq} 사이가 아닙니다")
    return start


def check_generator_provenance(meta: dict) -> None:
    """Refuse a recording whose sim_backend does not name committed code (provenance.problems).

    The generator records on a dirty tree; this is where that stops. A distribution whose
    episodes say "made from uncommitted changes" cannot be regenerated by anyone."""
    said = provenance.problems(meta.get("sim_backend"))
    if said:
        raise ValueError(
            f"{meta.get('episode_id')}: " + "; ".join(said)
            + " — 배포본은 커밋된 코드로 만든 녹화만 받습니다. 커밋한 뒤 다시 생성하십시오")


def require_termination(meta: dict) -> dict:
    """`{terminated, truncated, reason[]}` 인지 확인한다. dict 여부만으로는 부족하다.

    빈 객체도 dict 이고, 그러면 배포본이 「왜 끝났는가」를 비운 채 나간다.
    """
    obj = require_type(meta, "termination", dict)
    for name in ("terminated", "truncated"):
        if not isinstance(obj.get(name), bool):
            raise ValueError(f"termination.{name} 이 참·거짓이 아닙니다: {obj.get(name)!r}")
    reason = obj.get("reason")
    if not isinstance(reason, list) or not reason or not all(isinstance(x, str) for x in reason):
        raise ValueError(f"termination.reason 이 비어 있지 않은 문자열 목록이 아닙니다: {reason!r}")
    return obj


# Step field kinds step_frames() never builds: a camera stream goes to LeRobot as
# a video feature and the pack is copied beside the data, neither per frame here.
NOT_IN_FRAMES = ("video", "pack")


def active_conditions(arrays: dict, meta: dict) -> tuple[str, ...]:
    """The step-field conditions this episode meets: CAMERAS when its sidecar names ``media``, HRI
    when its npz carries the hri arrays (field_map.hri_recorded -- all of them or none); neither
    for a v0.1 episode.

    The recorder writes the camera arrays only together with media
    (recorder.save(media=...)), so camera arrays without media mean the npz and
    the sidecar did not come out of one save; the conversion stops rather than
    guess which half to believe. Half the hri arrays stop it the same way.
    """
    conditions: list[str] = []
    if "media" in meta:
        if not isinstance(meta["media"], dict) or not meta["media"]:
            raise ValueError(f"사이드카의 media 가 비었거나 객체가 아닙니다: {meta['media']!r}")
        conditions.append(CAMERAS)
    else:
        stray = sorted(f.npz for f in STEP_FIELDS
                       if f.condition == CAMERAS and f.npz is not None and f.npz in arrays)
        if stray:
            raise ValueError(
                f"사이드카에 media 가 없는데 npz 에 카메라 배열이 있습니다: {stray} — "
                "기록기는 매체와 함께만 이 배열을 씁니다")
    if hri_recorded(list(arrays)):
        conditions.append(HRI)
    return tuple(conditions)


def step_frames(arrays: dict, meta: dict, joint_map: dict) -> list[dict]:
    """One dict per step, keyed by schema §6 field name -- the fields this episode
    carries (active_conditions): without media, exactly the unconditional ones."""
    fields = [f for f in active_step_fields(active_conditions(arrays, meta))
              if f.kind not in NOT_IN_FRAMES]
    n = len(arrays["timestamp"])
    dof = int(joint_map["dof"])
    # 관절 폭을 쓰는 열이 넷인데 하나만 대조하고 있었다. 넷이 **일관되게** 잘못된
    # 배치는 그대로 feature 가 되고, 배포본은 dof 와 다른 폭을 실은 채 나간다.
    for name in ("joint_pos", "joint_vel", "joint_torque", "action"):
        got = arrays[name].shape[1] if arrays[name].ndim == 2 else None
        if got != dof:
            raise ValueError(
                f"dof 불일치: joint_map {dof} vs 데이터 {name} {got} — "
                "관절 정본과 다른 자산으로 생성된 배치입니다"
            )
    regions = list(meta["contact_region_names"])
    event_names = list(meta["safety_event_names"])
    events = np.asarray(arrays["safety_event"])

    # Resolved once: a KeyError here names the missing array before any frame
    # is built, rather than after n-1 of them.
    columns = {}
    for f in fields:
        if f.kind == "array":
            if f.npz not in arrays:
                raise KeyError(f"npz 에 {f.npz} 배열이 없습니다 ({f.schema})")
            columns[f.schema] = np.asarray(arrays[f.npz])

    # Every array has to agree on the step count before any of them is read.
    # The loop below runs over the timestamp length alone, which used to drop
    # the extra rows of a longer array without a word and raise IndexError
    # late on a shorter one.
    lengths = {name: int(arr.shape[0]) for name, arr in columns.items()}
    lengths["safety_event"] = int(events.shape[0])
    if "steps" in meta:
        lengths["meta.steps"] = int(meta["steps"])
    disagree = {name: length for name, length in lengths.items() if length != n}
    if disagree:
        raise ValueError(
            f"스텝 수가 어긋납니다 (timestamp {n}): {disagree}"
        )
    if events.ndim != 2 or events.shape[1] != len(event_names):
        raise ValueError(
            f"safety_event shape {events.shape} != ({n}, {len(event_names)})"
        )
    if events.dtype != np.bool_:
        raise ValueError(f"safety_event 의 dtype 이 bool 이 아닙니다: {events.dtype}")

    frames = []
    for i in range(n):
        frame: dict[str, object] = {}
        for f in fields:
            if f.kind == "array":
                value = columns[f.schema][i]
                frame[f.schema] = value.tolist() if value.ndim else value.item()
            elif f.kind == "sidecar":
                frame[f.schema] = list(regions)
            elif f.schema == "frame_index":
                frame[f.schema] = i
            elif f.schema == "safety.event":
                frame[f.schema] = [name for name, on in zip(event_names, events[i]) if on]
            else:
                raise ValueError(f"대응표에 처리 규칙이 없는 파생 필드: {f.schema}")
        frames.append(frame)
    return frames


# §6 fields that are lists of strings. LeRobot's "string" dtype is scalar
# only (feature_utils.validate_feature_string), and HF's Value("string")
# cannot sit inside a fixed-length Sequence either, so these travel as one
# JSON string per frame. validate_dataset.as_list() already parses a
# JSON-array string, so C3 reads them without a validator change.
JSON_STRING_FIELDS = (
    "observation.contact.region_names",
    "observation.contact.type",
    "safety.event",
)

# LeRobot generates these itself from the frame index and fps. Ours would be
# identical (both are index x dt), so writing them twice only risks drift.
LEROBOT_OWNED = ("timestamp", "frame_index")


def to_lerobot_frame(frame: dict, dtypes: dict) -> dict:
    """One schema frame, in the types LeRobot's validator demands.

    dtypes carries each numeric field's *recorded* numpy dtype. Casting
    everything to float32 would be simpler and would break C9: the hook
    compares a regenerated float64 npz against the published column, and a
    silent narrowing shows up there as a reproduction failure.
    """
    out: dict[str, object] = {}
    for key, value in frame.items():
        if key in LEROBOT_OWNED:
            continue
        if key in JSON_STRING_FIELDS:
            out[key] = json.dumps(list(value), ensure_ascii=False)
        elif isinstance(value, bool):
            out[key] = np.array([value], dtype=np.bool_)
        elif isinstance(value, (int, float)):
            out[key] = np.array([value], dtype=dtypes.get(key, np.float64))
        elif isinstance(value, list):
            out[key] = np.asarray(value, dtype=dtypes.get(key, np.float64))
        else:
            raise ValueError(f"LeRobot 로 옮길 수 없는 값: {key} = {type(value)}")
    return out


def lerobot_features(frame: dict) -> dict:
    """Feature spec from an already-converted frame -- never hand-written."""
    features: dict[str, dict] = {}
    for key, value in frame.items():
        if isinstance(value, str):
            features[key] = {"dtype": "string", "shape": (1,), "names": None}
        else:
            features[key] = {"dtype": str(value.dtype), "shape": tuple(value.shape),
                             "names": None}
    return features


def recorded_dtypes(arrays: dict, meta: dict) -> dict:
    """schema field -> the dtype the recorder actually wrote, for the fields this
    episode carries (active_conditions -- the same switch as step_frames)."""
    return {f.schema: np.asarray(arrays[f.npz]).dtype
            for f in active_step_fields(active_conditions(arrays, meta))
            if f.kind == "array" and f.npz in arrays}


# ── Camera media (schema §6 「조건부 필드 — 카메라」) ─────────────────────────
#
# An episode recorded with cameras carries its media beside its npz, named by the sidecar's
# ``media`` (generation/envs/s1a/media_writer.py): one lossless RGB stream per camera and --
# only when meta/cameras.json declares one -- the head depth/mask pack. The packs are switched
# off in the canon, not deleted from the code: without ``packs`` in the canon a
# recording carries none and the distribution ships none. The conversion is the one lossy step
# -- each stream is decoded once and handed to LeRobot as a video feature, which encodes it --
# and a pack is copied beside the data with the canon's downsample. meta/cameras.json ships
# only with such a batch: its presence is what makes the camera fields required (schema §6).

# LeRobot's video feature names, channel-last: what validate_frame accepts for (H, W, 3).
VIDEO_NAMES = ["height", "width", "channels"]
# Sidecar ``media`` members that are not a camera stream.
MEDIA_PACK = "pack"
# What the recorder adds to ``media`` besides the streams, read here and not shipped: the steps rendered
# again (not a notes member yet), each camera's all-black frames (check_blank_frames)
# and the fingerprint of the camera canon the recording was made with (check_canon_fingerprint).
MEDIA_NOT_CARRIED = ("rerendered_steps", "blank_frames", "canon_fingerprint")
# What cameras.recording_fingerprint returns: a sha256 as lower-case hex.
_FINGERPRINT = re.compile(r"[0-9a-f]{64}")


def video_cameras() -> dict[str, str]:
    """schema field -> the camera whose stream it is, for every video field in the map.

    The camera is the field name's last member (observation.images.<camera>); the sidecar
    names its file under media.<camera>, and meta/cameras.json describes it under
    cameras.<camera>."""
    out = {}
    for f in STEP_FIELDS:
        if f.kind != "video":
            continue
        prefix, _, cam = f.schema.rpartition(".")
        if prefix != "observation.images" or not cam:
            raise ValueError(f"영상 필드 이름에서 카메라를 읽을 수 없습니다: {f.schema}")
        out[f.schema] = cam
    return out


def camera_video_features(cameras: dict) -> dict:
    """The LeRobot video feature for every video field, at the canon's resolution.

    Declared, not derived from a frame: lerobot_features() reads a frame's dtype, and an
    (H, W, 3) uint8 array handed to it would become a parquet column."""
    features = {}
    for schema, cam in video_cameras().items():
        spec = (cameras.get("cameras") or {}).get(cam)
        if spec is None:
            raise ValueError(f"meta/cameras.json 에 카메라 {cam} 이 없습니다 ({schema})")
        features[schema] = {"dtype": "video", "shape": (int(spec["height"]), int(spec["width"]), 3),
                            "names": list(VIDEO_NAMES)}
    return features


def dataset_features(first_frame: dict, cameras: dict | None) -> dict:
    """The dataset's feature spec: every column of the first converted frame, plus -- for a
    batch recorded with cameras -- one video feature per camera. With ``cameras`` None this
    is lerobot_features() alone, exactly the v0.1 feature list."""
    features = lerobot_features(first_frame)
    if cameras is not None:
        videos = camera_video_features(cameras)
        clash = sorted(set(videos) & set(features))
        if clash:
            raise ValueError(f"영상 피처가 프레임 열과 이름이 겹칩니다: {clash}")
        features.update(videos)
    return features


def the_pack(cameras: dict) -> str | None:
    """The name of the pack meta/cameras.json declares, or None when it declares none.

    Packs are optional: the canon switches them off by leaving ``packs`` out;
    null and {} say the same. The sidecar has a single ``media.pack`` entry, so a canon with
    more than one is not one this reads, and a ``packs`` that is not an object is refused
    rather than read as none. The validator restates this (validate_dataset.read_camera_setup);
    tests/validator/test_media_checks.py holds the two to one answer."""
    packs = cameras.get("packs")
    if packs is None:
        return None
    if not isinstance(packs, dict):
        raise ValueError(f"meta/cameras.json 의 packs 가 객체가 아닙니다: {type(packs).__name__}")
    if len(packs) > 1:
        raise ValueError(f"meta/cameras.json 의 packs 는 많아야 하나여야 합니다: {sorted(packs)} — "
                         "사이드카의 media.pack 은 하나입니다")
    return next(iter(packs), None)


def media_files(meta: dict, episode_dir: Path, cameras: dict) -> dict[str, Path]:
    """The sidecar's ``media``, checked against the camera canon: camera -> its stream, plus
    MEDIA_PACK -> the pack when the canon declares one (the_pack).

    Refuses rather than skips. Every stream the map declares has to be named, with a
    relative path that stays under the episode directory, a file that exists and a frame
    count equal to the episode's steps (a stream the recorder cut short would otherwise be
    padded or truncated by someone downstream). An unknown member is refused too -- it
    would be dropped without a word -- and so is a pack the canon does not declare: the
    recording was made under another canon (the pack switched off since, or an old recording),
    and converting would drop the pack as silently."""
    media = meta.get("media")
    if not isinstance(media, dict) or not media:
        raise ValueError(f"사이드카의 media 가 비었거나 객체가 아닙니다: {media!r}")
    steps = meta.get("steps")
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise ValueError(f"에피소드 메타 steps 가 정수가 아닙니다: {steps!r}")
    pack = the_pack(cameras)
    if pack is None and MEDIA_PACK in media:
        raise ValueError(f"{meta.get('episode_id')}: 정본(meta/cameras.json)이 팩을 선언하지 않았는데 녹화가 "
                         f"팩을 가리킴(media.{MEDIA_PACK}) — 녹화 때와 정본이 다름. 팩을 끈 정본으로 다시 "
                         "녹화하거나, 녹화 때의 정본으로 변환하십시오")
    streams = sorted(set(video_cameras().values())) + ([MEDIA_PACK] if pack is not None else [])
    missing = [k for k in streams if k not in media]
    unknown = sorted(set(media) - set(streams) - set(MEDIA_NOT_CARRIED))
    if missing or unknown:
        raise ValueError(f"{meta.get('episode_id')}: 사이드카 media 가 대응표와 다릅니다 — "
                         f"빠짐 {missing} · 모르는 항목 {unknown}")
    if not isinstance(media.get("rerendered_steps", []), list):
        raise ValueError(f"media.rerendered_steps 가 목록이 아닙니다: {media['rerendered_steps']!r}")
    out: dict[str, Path] = {}
    for key in streams:
        entry = media[key]
        if not isinstance(entry, dict):
            raise ValueError(f"media.{key} 가 객체가 아닙니다: {entry!r}")
        rel, frames = entry.get("file"), entry.get("frames")
        if not isinstance(rel, str) or not rel:
            raise ValueError(f"{meta.get('episode_id')}: media.{key} 에 파일이 없습니다: {entry!r}")
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"media.{key}.file 이 에피소드 디렉터리 밖을 가리킵니다: {rel}")
        if not isinstance(frames, int) or isinstance(frames, bool) or frames != steps:
            raise ValueError(f"{meta.get('episode_id')}: media.{key} 의 프레임 수 {frames!r} 가 "
                             f"스텝 수 {steps} 와 다릅니다 — 기록기의 결함이며 채우거나 자르지 않습니다")
        path = Path(episode_dir) / rel_path
        if not path.is_file():
            raise FileNotFoundError(f"{meta.get('episode_id')}: media.{key} 파일이 없습니다: {path}")
        out[key] = path
    return out


def check_blank_frames(meta: dict) -> None:
    """Refuse a camera whose every recorded frame is black. A camera that sees nothing -- buried in some
    geometry, or mounted in the wrong place; the 2026-09-30 wrist preview, a separate script, put one inside
    the forearm and rendered it black -- is recorded all the same: camera_io.Rig never takes a black frame
    for a discarded render (all zeros repeat whether drawn again or not), and the recorder counts per camera
    the frames whose every pixel is 0 (media.blank_frames, media_writer.MediaWriter). Without this check
    such a stream would convert and ship. A camera black for some of its
    frames goes through.

    The count is required, one per camera the map declares: a recording without it was made before the
    recorder counted, and a camera left out of it would never be looked at. Called after
    media_files, which has held every stream's frame count to the episode's steps. Shared with C9's comparison
    (compare_published.pack_pair): a rerun the converter would refuse is no reproduction either."""
    eid, media, steps = meta.get("episode_id"), meta["media"], meta["steps"]
    blank = media.get("blank_frames")
    if blank is None:
        raise ValueError(f"{eid}: 사이드카에 media.blank_frames 가 없습니다 — 카메라마다 검은 프레임을 세기 전"
                         "의 녹화입니다. 재생성하십시오")
    cams = sorted(set(video_cameras().values()))
    if (not isinstance(blank, dict) or sorted(blank) != cams
            or not all(isinstance(n, int) and not isinstance(n, bool) and 0 <= n <= steps for n in blank.values())):
        raise ValueError(f"{eid}: media.blank_frames 가 카메라 {cams} 마다 0~{steps} 의 정수가 아닙니다: {blank!r}")
    dark = [cam for cam in cams if blank[cam] == steps]
    if dark:
        raise ValueError(f"{eid}: 카메라 {', '.join(dark)} 의 프레임이 모두 검습니다({steps}/{steps}) — 카메라가 "
                         "형상 속에 묻혔거나 장착 자리가 잘못됐습니다. meta/cameras.json 의 장착 규칙(링크에 붙는 "
                         "카메라는 generation/envs/s1d/extract_camera_mounts.py 의 추출)을 확인하고 재생성하십시오")


def recorded_canon_fingerprint(meta: dict) -> str:
    """The sidecar's media.canon_fingerprint, held to a sha256; raises ValueError with the sentence to print.

    Required, like check_blank_frames' counts: a recording without one predates the stamp and a value
    nobody could have written is a broken sidecar -- neither is compared as if it were "another canon".
    Shared with C9's comparison (compare_published.pack_pair), which holds a rerun to the shipped canon."""
    eid, media = meta.get("episode_id"), meta["media"]
    if "canon_fingerprint" not in media:
        raise ValueError(f"{eid}: 녹화가 지문을 남기지 않았습니다 — 지문을 남기기 전의 녹화이므로 재생성하십시오 "
                         "(사이드카 media.canon_fingerprint 가 없습니다)")
    recorded = media["canon_fingerprint"]
    if not isinstance(recorded, str) or not _FINGERPRINT.fullmatch(recorded):
        raise ValueError(f"{eid}: media.canon_fingerprint 가 sha256(소문자 16진 64자)이 아닙니다 — 녹화 지문을 "
                         f"읽을 수 없습니다: {recorded!r}")
    return recorded


def check_canon_fingerprint(meta: dict, cameras: dict) -> None:
    """Refuse a recording made under another camera canon than the one being converted with.

    The pictures were baked from meta/cameras.json as it was -- field of view, resolution, mount,
    channels, the box's camera -- and the conversion ships the canon it has now as their description.
    The recorder stamps media.canon_fingerprint (cameras.recording_fingerprint of the canon it recorded
    with: every key a recording path reads, without prose, the converter's derived intrinsics or a pack's
    downsample) and this holds it to the same function of the canon in hand. Nothing else notices most
    of these edits -- a mount is read by the recorder alone -- and the shipped canon would describe
    pictures it did not produce.

    The stamp is required and has to be a sha256 (recorded_canon_fingerprint). Called after media_files,
    which has held ``media`` to an object."""
    eid, recorded = meta.get("episode_id"), recorded_canon_fingerprint(meta)
    from generation import cameras as camera_canon
    if recorded != camera_canon.recording_fingerprint(cameras):
        raise ValueError(f"{eid}: 녹화 때 카메라 정본과 지금 meta/cameras.json 이 다릅니다 — 픽셀에 구워지는 값"
                         "(시야각 · 해상도 · 장착 · 채널 · bbox 카메라)이 바뀌었으면 재생성하고, 아니면 녹화 "
                         "때 정본으로 되돌리십시오(SOP §10.1)")


def check_video(cam: str, cameras: dict, probed: tuple[int, int, int], steps: int) -> None:
    """A stream's own (width, height, frame count) against the canon and the episode."""
    spec = cameras["cameras"][cam]
    width, height, frames = probed
    if (width, height) != (int(spec["width"]), int(spec["height"])):
        raise ValueError(f"{cam}: 영상 크기 {width}×{height} 가 meta/cameras.json 의 "
                         f"{spec['width']}×{spec['height']} 와 다릅니다")
    if frames != steps:
        raise ValueError(f"{cam}: 영상 프레임 {frames} 개가 스텝 수 {steps} 와 다릅니다 — "
                         "기록기의 결함이며 채우거나 자르지 않습니다")


def check_canon_rate(cameras: dict, step_dt: float, episode_id: str = "?") -> None:
    """The camera canon's ``fps`` against an episode's recording period. Raises ValueError.

    The videos hold one picture per recorded step, so their frame period is the step period.
    The camera rig holds the canon to that when it records (camera_io.Rig, which applies
    check_record_rate to this pair); the canon can be edited afterwards, and then nothing else
    in the conversion reads the number: the dataset's rate comes from the timestamps and the
    pictures are decoded one by one. meta/cameras.json would ship with a rate that is not the
    dataset's, which the validator's C1 refuses (fps differs from info.json's) once the whole
    conversion has been paid for. The tolerance is check_record_rate's."""
    fps = cameras.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not fps > 0:
        raise ValueError(f"카메라 정본(meta/cameras.json)의 fps 가 양수가 아닙니다: {fps!r}")
    if abs(step_dt - 1.0 / fps) > 1e-9:
        raise ValueError(
            f"{episode_id}: 카메라 정본(meta/cameras.json)의 fps {fps:g} 가 이 에피소드의 기록 주기 "
            f"{step_dt:.6f}s({round(1.0 / step_dt)} fps)와 다릅니다 — 영상은 기록 스텝마다 한 프레임이라 "
            "정본의 fps 는 기록 주기와 같아야 합니다(녹화할 때도 같은 이유로 멈춥니다). 녹화 뒤에 정본을 "
            "고쳤다면 되돌리십시오 — 그대로 변환하면 검증기 C1 이 거부하는 meta/cameras.json 이 실립니다")


def with_video_frames(frames: Sequence[dict], streams: dict[str, Iterable],
                      shapes: dict[str, tuple]) -> Iterable[dict]:
    """Each converted frame with one decoded picture per video field, in step order.

    ``streams`` maps a video field to an iterator of (H, W, 3) uint8 pictures -- decoded
    lazily, so an episode's video is never held whole. A stream that ends early or runs
    past the last step raises: the recorder writes one picture per recorded step, and a
    count that disagrees is its defect, not something to pad or cut. The overrun is seen
    after the last frame is yielded and before the caller saves the episode."""
    its = {key: iter(s) for key, s in streams.items()}
    n = len(frames)
    for i, frame in enumerate(frames):
        out = dict(frame)
        for key, it in its.items():
            picture = next(it, None)
            if picture is None:
                raise ValueError(f"{key}: 영상이 {i} 프레임에서 끝났습니다 — 스텝 {n}")
            picture = np.asarray(picture)
            if picture.shape != tuple(shapes[key]) or picture.dtype != np.uint8:
                raise ValueError(f"{key}: 프레임 {i} 가 {picture.shape} {picture.dtype} 입니다 — "
                                 f"{tuple(shapes[key])} uint8 이어야 합니다")
            out[key] = picture
        yield out
    for key, it in its.items():
        if next(it, None) is not None:
            raise ValueError(f"{key}: 영상 프레임이 스텝 {n} 개보다 많습니다")


def pyav_probe(path: Path) -> tuple[int, int, int]:
    """(width, height, frames) of a recorded stream, by demuxing -- no decode. mkv does not
    store a frame count, so the packets are counted (one per picture for this stream)."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for packet in container.demux(stream) if packet.size > 0)
        return int(stream.codec_context.width), int(stream.codec_context.height), frames


def pyav_decode(path: Path) -> Iterable[np.ndarray]:
    """The recorded stream's pictures, one at a time, as (H, W, 3) uint8 RGB.

    PyAV and imageio_ffmpeg decode the lossless stream to identical bytes (measured
    2026-09-28 over all 999 pictures of both cameras of one S1d episode)."""
    import av

    with av.open(str(path)) as container:
        for picture in container.decode(container.streams.video[0]):
            yield picture.to_ndarray(format="rgb24")


# The arrays a pack holds (media_writer.close); depth and mask dtypes are the canon's.
PACK_ARRAYS = ("depth", "mask", "step_index")


def downsample_factors(camera: dict, downsample: dict) -> tuple[int, int, int]:
    """(every_n_steps, row stride, column stride) for the canon's ``downsample``.

    Width and height have to be whole fractions of the camera's: pixels are picked, never
    averaged -- averaging depth mixes in the invalid 0, and a class mask cannot be averaged.
    Validator C1 restates this rule (validate_dataset.read_camera_setup, which does not import
    generation); tests/validator/test_media_checks.py holds the two to one answer."""
    k = downsample.get("every_n_steps")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"downsample.every_n_steps 는 1 이상의 정수여야 합니다: {k!r}")
    strides = []
    for axis in ("height", "width"):
        full, want = int(camera[axis]), downsample.get(axis)
        if (not isinstance(want, int) or isinstance(want, bool) or want < 1
                or full % want):
            raise ValueError(f"downsample.{axis} {want!r} 가 원 해상도 {full} 의 정수 분의 일이 아닙니다")
        strides.append(full // want)
    return k, strides[0], strides[1]


def downsample_pack(pack, camera: dict, downsample: dict) -> dict[str, np.ndarray]:
    """A recorded pack at the canon's downsample. Pure; imports nothing heavy.

    Steps are kept at every_n_steps and pixels at the row/column strides (downsample_factors).
    step_index keeps the recorded step numbers -- the kept rows' own, not renumbered -- so a
    reader joins a pack row to its parquet row by value. C9's comparison runs a regenerated pack
    through this same function before comparing it with the published one."""
    k, sy, sx = downsample_factors(camera, downsample)
    missing = [name for name in PACK_ARRAYS if name not in pack]
    if missing:
        raise ValueError(f"팩에 배열이 없습니다: {missing}")
    depth, mask = np.asarray(pack["depth"]), np.asarray(pack["mask"])
    step_index = np.asarray(pack["step_index"])
    size = (int(camera["height"]), int(camera["width"]))
    for name, arr in (("depth", depth), ("mask", mask)):
        if arr.ndim != 3 or arr.shape[1:] != size or arr.shape[0] != step_index.shape[0]:
            raise ValueError(f"팩 {name} 의 모양 {arr.shape} 가 (N={step_index.shape[0]}, {size[0]}, "
                             f"{size[1]}) 가 아닙니다")
    if step_index.ndim != 1:
        raise ValueError(f"팩 step_index 가 1차원이 아닙니다: {step_index.shape}")
    out = {"depth": depth[::k, ::sy, ::sx], "mask": mask[::k, ::sy, ::sx],
           "step_index": step_index[::k]}
    # Postcondition, not a re-statement: with equal strides a row/column mix-up is invisible,
    # and C9 compares regenerated packs through this same function, so it would agree
    # with a converter that swapped them. The canon names the result's shape; hold to it.
    rows = len(range(0, step_index.shape[0], k))
    want = (rows, int(downsample["height"]), int(downsample["width"]))
    for name in ("depth", "mask"):
        if out[name].shape != want:
            raise ValueError(f"축소한 팩 {name} 의 모양 {out[name].shape} 가 정본 downsample 의 "
                             f"(스텝, height, width) = {want} 와 다릅니다")
    if out["step_index"].shape != (rows,):
        raise ValueError(f"축소한 팩 step_index 의 모양 {out['step_index'].shape} 가 ({rows},) 가 아닙니다")
    return out


def pack_layout(path: Path) -> dict[str, tuple[tuple, np.dtype]]:
    """name -> (shape, dtype) of each array in an npz, read from the .npy headers alone --
    the pre-flight looks at every pack before a frame is written, without inflating ~900 MB each."""
    import zipfile

    out = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            with zf.open(info) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    raise ValueError(f"읽을 수 없는 npy 형식 {version}: {path}:{info.filename}")
            out[info.filename.removesuffix(".npy")] = (tuple(shape), np.dtype(dtype))
    return out


def check_pack_layout(layout: dict, cameras: dict, steps: int) -> None:
    """The pack's arrays, shapes and dtypes against the canon: exactly PACK_ARRAYS, depth and
    mask (steps, H, W) in the canon's dtypes at the pack camera's resolution, step_index (steps,)
    integer. Also checks the canon's downsample is one this converter can apply. Only for a
    canon that declares a pack."""
    pack = _declared_pack(cameras)
    camera = cameras["cameras"][pack["camera"]]
    if set(layout) != set(PACK_ARRAYS):
        raise ValueError(f"팩의 배열이 {sorted(layout)} 입니다 — {list(PACK_ARRAYS)} 여야 합니다")
    size = (steps, int(camera["height"]), int(camera["width"]))
    for name in ("depth", "mask"):
        shape, dtype = layout[name]
        if shape != size or dtype != np.dtype(pack[name]["dtype"]):
            raise ValueError(f"팩 {name} 가 {shape} {dtype} 입니다 — {size} {pack[name]['dtype']} 여야 합니다")
    shape, dtype = layout["step_index"]
    if shape != (steps,) or dtype.kind not in "iu":
        raise ValueError(f"팩 step_index 가 {shape} {dtype} 입니다 — ({steps},) 정수여야 합니다")
    downsample_factors(camera, pack["downsample"])


def check_pack_steps(path: Path, steps: int) -> None:
    """The pack's step_index has to be the recorded 0..steps-1: pack row i is parquet row i
    (frame_index), and nothing else in the distribution ties the two together.

    Reads that one member: np.load inflates an npz member only when it is asked for, so the
    depth and mask stay compressed (2026-09-28, a real 234 MB pack: 23 ms and +2 MB RSS for
    step_index against 1.6 s and ~0.9 GB for every member). Cheap enough for the pre-flight,
    where a bad pack in the last episode costs nothing rather than the whole run."""
    with np.load(path) as z:
        step_index = z["step_index"]
    if not np.array_equal(step_index, np.arange(steps)):
        raise ValueError(f"팩 step_index 가 기록된 스텝 0..{steps - 1} 이 아닙니다 — 팩의 행이 parquet 의 "
                         f"frame_index 와 맞지 않습니다: {path}")


def pack_file(name: str, episode_index: int) -> str:
    return f"packs/{name}/episode_{episode_index:06d}.npz"


def _declared_pack(cameras: dict) -> dict:
    """The canon entry of the pack a pack step works on. Called only where a pack is declared;
    a canon without one reaching here is the caller's mistake, said rather than a KeyError."""
    name = the_pack(cameras)
    if name is None:
        raise ValueError("meta/cameras.json 이 팩을 선언하지 않았습니다 — 팩 단계는 정본이 팩을 선언할 때만 돕니다")
    return cameras["packs"][name]


def write_pack(src: Path, out: Path, episode_index: int, cameras: dict, steps: int) -> dict:
    """Copy one episode's pack into the distribution at the canon's downsample; returns its
    meta/packs.json entry {file, frames, sha256}.

    Relies on the pre-flight (preflight_media: the layout from the .npy headers, step_index ==
    0..steps-1), which has already seen every pack before a frame is written. At the identity
    downsample the recorded file is therefore copied byte for byte without being inflated;
    otherwise it is loaded, downsampled (downsample_pack) and re-compressed."""
    import hashlib
    import shutil

    pack = _declared_pack(cameras)
    camera = cameras["cameras"][pack["camera"]]
    rel = pack_file(the_pack(cameras), episode_index)
    dst = Path(out) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if downsample_factors(camera, pack["downsample"]) == (1, 1, 1):
        shutil.copy2(src, dst)
        frames = steps
    else:
        with np.load(src) as z:
            shipped = downsample_pack({key: z[key] for key in PACK_ARRAYS}, camera, pack["downsample"])
        np.savez_compressed(dst, **shipped)
        frames = int(shipped["step_index"].shape[0])
    digest = hashlib.sha256()
    with open(dst, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return {"file": rel, "frames": frames, "sha256": digest.hexdigest()}


def shipped_cameras(cameras: dict) -> dict:
    """The distribution's meta/cameras.json: the canon, with each camera's pinhole intrinsics
    filled in (generation/cameras.intrinsics). The repository canon keeps the field of view
    only, so the focal length is written down once (meta/cameras.json intrinsics_note)."""
    import copy

    from generation.cameras import intrinsics

    doc = copy.deepcopy(cameras)
    for spec in doc["cameras"].values():
        fx, fy, cx, cy = intrinsics(spec)
        spec.update({"fx": fx, "fy": fy, "cx": cx, "cy": cy})
    return doc


# LeRobot writes each video picture to images/<video key>/episode-XXXXXX/ as a PNG before encoding
# (lerobot/datasets/utils.py DEFAULT_IMAGE_PATH), removes each episode's directory once it is
# encoded, and leaves the per-key directories behind, empty.
IMAGE_STAGING = "images"


def remove_image_staging(root: Path) -> bool:
    """After finalize(): remove LeRobot's frame staging tree if it holds nothing. Returns whether
    there was one to remove. A file still in it is a picture LeRobot never encoded -- the conversion
    stops and names it rather than shipping, or deleting, what did not become a video."""
    import shutil

    staging = Path(root) / IMAGE_STAGING
    if not staging.is_dir():
        return False
    left = sorted(str(p.relative_to(root)) for p in staging.rglob("*")
                  if p.is_symlink() or not p.is_dir())
    if left:
        raise ValueError(f"LeRobot 이 인코딩하지 않은 프레임 {len(left)}개가 {IMAGE_STAGING}/ 에 남았습니다 "
                         f"(예: {left[:3]}) — 영상에 들어가지 않은 프레임입니다")
    shutil.rmtree(staging)
    return True


def batch_conditions(per_episode: dict[Path, tuple[str, ...]]) -> tuple[str, ...]:
    """The one condition set of a batch. LeRobot keeps one feature list per dataset, so a
    batch whose episodes disagree -- some recorded with cameras, some without -- cannot be
    one dataset, and is refused before a frame is written.

    Keyed by the episode's npz path, not its episode_id: two files carrying one id are two
    episodes to the conversion loop, and keyed by id one of them would vanish from this check."""
    sets = {tuple(sorted(c)) for c in per_episode.values()}
    if len(sets) > 1:
        by = {c: sorted(Path(e).name for e, got in per_episode.items() if tuple(sorted(got)) == c)
              for c in sets}
        raise ValueError("배치 안에서 에피소드의 조건이 다릅니다 — 피처 목록이 에피소드마다 달라질 수 "
                         "없으므로 한 데이터셋으로 변환할 수 없습니다: "
                         + "; ".join(f"{list(c) or '조건 없음'}: {len(e)}건 (예: {e[0]})"
                                     for c, e in sorted(by.items())))
    return next(iter(sets), ())


def check_hri_declared(paths: Sequence[Path], by_sid: dict[str, dict]) -> None:
    """Every episode records the hri arrays exactly when its scenario declares R-TS15066-SSM-02. A
    recording without them cannot carry the columns that scenario's judgement reads, and one with
    them under another scenario would ship columns its rules do not explain. A recording with some of the hri
    arrays and not the others is named with its scenario. Before a frame."""
    wrong: list[str] = []
    for p in paths:
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(p, allow_pickle=True) as arrays:
            try:
                recorded = HRI in active_conditions(arrays, meta)
            except ValueError as error:
                wrong.append(f"{p.name}: 시나리오 {meta.get('scenario_id')} — {error}")
                continue
        declared = scenario_declares_directed_ssm(by_sid[meta["scenario_id"]])
        if recorded != declared:
            wrong.append(f"{p.name}: 시나리오 {meta['scenario_id']} 는 {DIRECTED_SSM_RULE_ID} 를 "
                         + ("선언하는데 npz 에 hri 배열이 없습니다" if declared
                            else "선언하지 않는데 npz 에 hri 배열이 있습니다"))
    if wrong:
        raise ValueError(f"시나리오의 판정과 녹화가 맞지 않습니다({len(wrong)}건) — " + "; ".join(wrong[:3]))


# The sidecar keys of the task's record that a scenario with a completion criterion must name.
TASK_RECORD_KEYS = ("task_stages", "human_stature_m", "task_time_limit_s")


def check_hri_completion(paths: Sequence[Path], by_sid: dict[str, dict]) -> None:
    """A scenario that declares R-TS15066-SSM-02 has a completion criterion, so its sidecar names the
    step the task was done at (an integer from -1: not done in time, and short of the recording's
    step count). A missing or null member would ship as "no criterion" and the scorecard would leave
    the episode out of the completion rate; a step past the last would count a completion the episode
    never reached. Its sidecar also names the stages the task ended in, the person's stature and the
    task's time limit (task_stages, human_stature_m, task_time_limit_s): a missing or null one would ship
    as "no script". The other scenarios may leave all of it null. Before a frame."""
    wrong: list[str] = []
    for p in paths:
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        if not scenario_declares_directed_ssm(by_sid[meta["scenario_id"]]):
            continue
        done = meta.get("task_completed_step")
        if isinstance(done, bool) or not isinstance(done, int) or done < -1:
            wrong.append(f"{p.name}: 시나리오 {meta['scenario_id']} 는 {DIRECTED_SSM_RULE_ID} 를 선언하므로 사이드카의 "
                         "task_completed_step 이 -1 이상의 정수여야 합니다(-1: 시간 안에 끝내지 못함) — "
                         + ("없음" if "task_completed_step" not in meta else f"받은 값 {done!r}"))
            continue
        absent = [key for key in TASK_RECORD_KEYS if meta.get(key) is None]
        if absent:
            wrong.append(f"{p.name}: 시나리오 {meta['scenario_id']} 는 {DIRECTED_SSM_RULE_ID} 를 선언하므로 사이드카에 "
                         + ", ".join(absent) + " 이(가) 있어야 합니다(없거나 null)")
            continue
        with np.load(p, allow_pickle=True) as arrays:
            steps = len(arrays["timestamp"])
        if done >= steps:
            wrong.append(f"{p.name}: 사이드카의 task_completed_step {done} 이 녹화의 스텝({steps}개 — 0..{steps - 1}) "
                         "안이 아닙니다")
    if wrong:
        raise ValueError(f"과제 완료 스텝이 없거나 맞지 않는 녹화입니다({len(wrong)}건) — " + "; ".join(wrong[:3]))


def check_table_recordings(paths: Sequence[Path], by_sid: dict[str, dict], assets: dict, repo_root: Path) -> None:
    """Every recording of a scenario that declares a conditions table carries what its table columns come from: its
    sidecar's condition record names the row's design and grid cell, and its label (<id>.labels.json) the evidence,
    under evidence.NAMES as they are. A recording or label made under a schema before 0.2 has neither, and this
    converter does not convert it. Where both are there, the columns are built once (table_columns, with the canon
    ``assets`` and the joint map of the recording's robot under ``repo_root``, read as the conversion reads them) and
    let go: what the episode spec refuses -- task_stages that are not the stages the row's path has, a draw without
    its appearance, an asset path the canon does not know -- is said here, by the file, and not after the episodes
    before it are written. Before a frame."""
    import yaml

    from generation.field_map import TABLE

    wrong: list[str] = []
    refused: list[str] = []
    joint_maps: dict[str, dict] = {}
    for p in paths:
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        scenario = by_sid[meta["scenario_id"]]
        if TABLE not in scenario:
            continue
        label = json.loads(p.with_suffix(".labels.json").read_text(encoding="utf-8"))
        label = label if isinstance(label, dict) else {}
        problems = _table_problems(meta, label)
        if problems:
            wrong.append(f"{p.name}: 시나리오 {meta['scenario_id']} — " + "; ".join(problems))
            continue
        model = meta["robot_model"]
        if model not in joint_maps:
            joint_maps[model] = yaml.safe_load(
                (Path(repo_root) / f"meta/joint_map/{model}.yaml").read_text(encoding="utf-8"))
        try:
            table_columns(meta, label, scenario, assets, joint_maps[model])
        except ValueError as e:
            refused.append(f"{p.name}: 시나리오 {meta['scenario_id']} — {e}")
    if wrong:
        raise ValueError(f"조건표가 있는 시나리오의 녹화 · 라벨에 에피소드 열을 낼 것이 없습니다({len(wrong)}건) — "
                         "데이터 구조 0.2 전의 녹화 · 라벨은 이 변환기로 변환하지 않습니다: " + "; ".join(wrong[:3]))
    if refused:
        raise ValueError(f"조건표가 있는 시나리오의 녹화로 에피소드 열을 짓지 못합니다({len(refused)}건) — "
                         + "; ".join(refused[:3]))


def episode_column(value: object) -> object:
    """One §5 field, in a form parquet can hold and the validator can read.

    Nested structures (scene_assets, sim_backend) become JSON strings --
    validate_dataset's as_obj/as_list already parse those, and a parquet
    struct column would pin the schema's shape into the file format. Plain
    lists of strings stay native; the validator's as_list handles the
    ndarray they come back as.
    """
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    return json.dumps(value, ensure_ascii=False)


STATS_UNDEFINED_NOTE = (
    "이 필드의 데이터에는 비유한값이 있습니다 — safety.zmp 는 지지다각형이 정의되지 "
    "않는 구간에서 NaN 이고, safety.stability_margin 은 양발이 모두 미하중인 구간에서 "
    "-inf 입니다(스키마 §6). 둘 다 규격이며 결측이 아닙니다. LeRobot 의 통계 경로는 "
    "그런 값을 다루지 못해 min·mean·std 가 비유한이 되고 **분위수 다섯이 전부 최댓값 "
    "부근으로 무너집니다** — 그대로 두면 「음수면 지지다각형 밖」인 필드의 중앙값이 "
    "양수로 실립니다(2026-08-29 실측: 실린 q50 +0.1235 대 유한값 실제 q50 -0.1302). "
    "그래서 이 필드의 통계는 싣지 않습니다. 유한값만의 분포는 데이터셋 카드에 있습니다."
)


def features_with_non_finite(root: Path) -> set[str]:
    """스텝 parquet 에 비유한값이 실제로 들어 있는 열. **나가는 바이트를 봅니다.**

    어느 필드가 그런지 코드에 적어 두지 않는 이유는 목록이 조용히 낡기 때문입니다.
    새 필드가 ±inf 나 NaN 을 담게 되면 여기서 저절로 잡힙니다.
    """
    import numpy as np
    import pandas as pd

    found: set[str] = set()
    for path in sorted((root / "data").rglob("*.parquet")):
        frame = pd.read_parquet(path)
        for column in frame.columns:
            if column in found:
                continue
            try:
                values = np.asarray(frame[column].tolist(), dtype=float)
            except (TypeError, ValueError):
                continue          # 문자열·JSON 열은 통계를 갖지 않는다
            if values.size and not np.isfinite(values).all():
                found.add(column)
    return found


def normalise_stats(root: Path) -> None:
    """`meta/stats.json` 을 표준 JSON 으로 만들고, **믿을 수 없는 통계는 비운다.**

    첫째, `NaN` 은 JSON 이 아니다 — 파이썬의 `json` 은 확장으로 받아 주지만
    `JSON.parse` 는 거부한다. 배포본의 기계가독 파일이 표준 파서로 안 열리면 「받는
    쪽이 읽는 규격」이 아니다.

    둘째 — **그 처리만으로는 부족하다.** 비유한값이 있는 필드는 `min`·`mean`·`std`
    가 비유한이 되어 `null` 로 정리되지만, **분위수는 유한한 채로 틀린 값**이 되어
    이 청소를 그대로 통과한다. 실측(2026-08-29): `safety.stability_margin` 의
    q01~q99 가 전부 0.1235 로 실려 있었고 유한값의 실제 중앙값은 -0.1302 다.
    「음수면 지지다각형 밖」인 필드의 중앙값이 **부호가 뒤집힌 채** 나가고 있었다.
    NaN 은 시끄럽게 실패하고 틀린 유한값은 조용히 통과한다.

    그래서 그런 필드는 통계 블록을 통째로 비우고 사유를 적는다. 어느 필드인지는
    **배포본의 스텝 parquet 에서 유도**한다 — 손으로 적은 목록은 낡는다.
    """
    import math

    p = root / "meta" / "stats.json"
    if not p.is_file():
        return

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x

    doc = clean(json.loads(p.read_text(encoding="utf-8")))
    for column in sorted(features_with_non_finite(root)):
        block = doc.get(column)
        if not isinstance(block, dict):
            continue
        doc[column] = {key: (value if key == "count" else None)
                       for key, value in block.items()}
        doc[column]["undefined_reason"] = STATS_UNDEFINED_NOTE
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                 encoding="utf-8")


def blank_episode_stats(root: Path) -> int:
    """에피소드별 통계에서도 믿을 수 없는 값을 비운다. 비운 열 수를 돌려준다.

    `meta/stats.json` 만 고치면 **같은 틀린 숫자가 옆 파일에 그대로 남는다** —
    `meta/episodes/**.parquet` 의 `stats/<필드>/<키>` 열이 LeRobot 이 에피소드마다
    계산한 값이고, 무너지는 것도 거기서부터다(400건 전부). 전역 통계는 그것들을
    합친 것이므로 원인은 합치는 단계가 아니라 에피소드 단계에 있다.

    `count` 는 남긴다 — 몇 표본에서 나왔을 값인지는 여전히 사실이다.
    비운 상태로도 `LeRobotDataset` 이 로드되는 것을 실측으로 확인했다(2026-08-29).
    """
    import numpy as np
    import pandas as pd

    features = features_with_non_finite(root)
    if not features:
        return 0
    blanked = 0
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        frame = pd.read_parquet(path)
        columns = [c for c in frame.columns
                   if c.startswith("stats/") and not c.endswith("/count")
                   and c.split("/", 2)[1] in features]
        if not columns:
            continue
        for column in columns:
            width = len(np.asarray(frame[column].iloc[0]).ravel())
            frame[column] = [np.full(width, np.nan)] * len(frame)
        frame.to_parquet(path, index=False)
        blanked += len(columns)
    return blanked


def merge_episode_metadata(root: Path, records: list[dict]) -> Path:
    """Add the schema §5 columns to LeRobot's own meta/episodes parquet.

    LeRobot v3.0 owns that table (episode_index, tasks, length, per-feature
    stats) and has no hook for extra columns, so they are joined in after
    finalize(). The validator reads meta/episodes/*.parquet and takes every
    column it finds, so one table carries both.
    """
    import pandas as pd

    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise ValueError("에피소드 메타 parquet 이 없습니다")
    # LeRobot splits this table by size, so at 400 episodes it is no longer one
    # file. Joining on episode_index rather than concatenating positionally --
    # position is only meaningful within a file, and the failure mode of getting
    # it wrong is every episode carrying another episode's label.
    extra = pd.DataFrame([{k: episode_column(v) for k, v in r.items()} for r in records])
    extra.insert(0, "episode_index", range(len(records)))

    ids = [r["episode_id"] for r in records]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"변환 기록에 중복된 episode_id: {dup[:5]}")

    # 먼저 전부 읽고 전부 검사한 뒤에 쓴다. 파일마다 검사하고 바로 쓰면 (a) 샤드
    # 하나만 보는 one_to_one 은 **샤드를 가로지르는 중복**을 못 본다 -- 두 샤드가
    # 각각 episode_index 0 을 담고 1 을 빠뜨리면 합계가 맞아 통과했고 0 번이 두 번
    # 실렸다, (b) 뒤 샤드에서 실패해도 앞 샤드는 이미 열이 붙어 있어 재실행이
    # 겹침으로 거부된다.
    tables = []
    seen: dict[int, Path] = {}
    for path in files:
        table = pd.read_parquet(path)
        if "episode_index" not in table.columns:
            raise ValueError(f"LeRobot 메타에 episode_index 가 없습니다: {path}")
        overlap = sorted((set(extra.columns) - {"episode_index"}) & set(table.columns))
        if overlap:
            raise ValueError(f"LeRobot 열과 이름이 겹칩니다: {overlap}")
        for idx in table["episode_index"]:
            idx = int(idx)
            if idx in seen:
                raise ValueError(
                    f"episode_index {idx} 가 두 파일에 있습니다: "
                    f"{seen[idx].name}, {path.name}"
                )
            seen[idx] = path
        tables.append((path, table))

    want = set(range(len(records)))
    if seen.keys() != want:
        missing = sorted(want - seen.keys())[:5]
        extra_idx = sorted(seen.keys() - want)[:5]
        raise ValueError(
            f"에피소드 인덱스 불일치: LeRobot {len(seen)} vs 변환 {len(records)}"
            + (f" · 빠짐 {missing}" if missing else "")
            + (f" · 남음 {extra_idx}" if extra_idx else "")
        )

    for path, table in tables:
        merged = table.merge(extra, on="episode_index", how="left", validate="one_to_one")
        merged.to_parquet(path, index=False)
    return files[0]


# Copied into the published dataset so it validates and reads on its own.
# A dataset whose rules.json lives only in the source repo cannot have its
# labels re-derived by anyone who downloads it, which is the claim this
# dataset makes.
CANON_FILES = (
    "NOTICE",
    "meta/safety_schema.json", "meta/assets.json",
    # NOTICE 와 assets.json 이 둘 다 이 경로를 라이선스 판정의 근거로 가리키는데,
    # 싣지 않으면 **배포본 안에서 그 포인터가 아무 데도 닿지 않는다** — 라이선스
    # 검토자가 산출물만으로 자산 조사를 따라갈 수 없다. 11.5 KB 이고 이미 공개
    # 경로의 문서다.
    "docs/assets/asset-license-inventory-v0.1.md",
)

# rules.json and scenarios.json are not in that tuple because they are not
# copied. A distribution carries the scenarios it has episodes for and the
# rules those declare, written by release_canon() -- see that module for why.
NARROWED_CANON = ("meta/scenarios.json", "meta/rules.json")

# 배포본에서 이름이 달라지는 정본. 데이터셋 루트의 README.md 는 Hugging Face 에서
# 데이터셋 카드 그 자체이므로(YAML front matter 가 라이선스·태그·뷰어를 결정한다)
# 저장소 소개를 그대로 올려서는 안 된다 — 스키마 §4 의 「Dataset Card (국/영)」.
# 배포본의 LICENSE 는 **데이터셋의** 라이선스여야 한다. 저장소 루트의 LICENSE 는
# 코드의 Apache-2.0 이라, 그것을 그대로 복사하면 카드는 CC BY 4.0 이라
# 선언하는데 같이 받은 파일은 Apache-2.0 이어서 받는 쪽이 어느 조건이 적용되는지
# 확정할 수 없다. 코드 라이선스도 함께 싣는다 — 배포본에 검증기 코드가
# 들어가지는 않지만 스크립트 인용이 있고, 무엇보다 두 라이선스가 각각 무엇에
# 적용되는지는 NOTICE 가 말한다.
CANON_RENAMES = {
    "docs/dataset-card.md": "README.md",
    "LICENSE-DATASET": "LICENSE",
    "LICENSE": "LICENSE-CODE",
}
# 관절 정본은 **이 배치가 쓴 로봇의 것만** 싣는다. 디렉터리째 복사하면 이 배치에
# 없는 G1 의 매핑이 따라 들어가고, 그 파일이 가리키는 제3자 자산은 NOTICE 에도
# assets.json 에도 없어서 받는 쪽이 필요한 자산인지 판단할 수 없다.
CANON_DIRS: tuple[str, ...] = ()


def check_record_rate(step_dt: float, fps: int, first_step_dt: float | None = None,
                      episode_id: str = "?") -> None:
    """Refuse a rate the published timestamps cannot express. Raises ValueError.

    LeRobot stores one rate per dataset and rebuilds every timestamp as
    ``frame_index / fps``, so two things have to hold and only one was checked.
    The rate has to match the rest of the batch -- and it has to be expressible
    as an integer fps at all, or a batch that is internally consistent at
    0.0201 s still ships timestamps 10 ms adrift by frame 1000.

    Extracted so a test can drive it. The check used to live inside main(), and
    the test named for it only compared two metadata dicts -- deleting the check
    left that test green.
    """
    # 배치 불일치를 먼저 본다 — 더 구체적인 진단이고, 뒤 에피소드에서는 표현 가능성
    # 검사가 같은 사실을 덜 유용한 말로 먼저 보고해 버린다.
    if first_step_dt is not None and abs(step_dt - first_step_dt) > 1e-9:
        raise ValueError(
            f"{episode_id}: 기록 주기 {step_dt:.6f}s 가 이 배치의 "
            f"{first_step_dt:.6f}s 와 다릅니다 — LeRobot 은 데이터셋 하나에 주기 "
            "하나를 쓰고 timestamp 를 그 주기로 다시 만들므로, 섞인 배치는 한 "
            "데이터셋으로 변환할 수 없습니다"
        )
    if abs(step_dt - 1.0 / fps) > 1e-9:
        raise ValueError(
            f"기록 주기 {step_dt:.6f}s 가 정수 fps 로 표현되지 않습니다 "
            f"(반올림 {fps} → {1.0 / fps:.6f}s). LeRobot 은 timestamp 를 "
            "frame_index/fps 로 다시 쓰므로 배포본의 시각이 원본과 어긋납니다"
        )


def assemble_dataset(repo_root: Path, out: Path, assets: dict,
                     scenario_ids: Sequence[str], triggered: Iterable[str] = (),
                     robot_models: Iterable[str] = (), cameras: dict | None = None,
                     packs: dict | None = None, *, referenced: Iterable[str]) -> list[str]:
    """Copy the canon and the redistributable assets in beside the parquet.

    ``scenario_ids`` is what this batch has episodes for and ``triggered`` every
    rule its labels point at; both narrow the two canon files that describe
    scope rather than format.

    ``referenced`` is every asset path the batch's episodes carry in scene_assets.
    It has no default: the assets that ship are the redistributable ones this batch
    uses (generation.assets.shipped_asset_files), and a caller that forgot to say
    which those are must not fall back to shipping every one the canon knows.

    A batch recorded with cameras ships meta/cameras.json (``cameras``, with intrinsics:
    shipped_cameras); a batch without ships none -- a cameras.json in a v0.1 re-conversion
    would make C1 demand videos. ``packs`` is the meta/packs.json index of the packs already
    copied, given exactly when ``cameras`` declares a pack (the_pack): an index the canon does
    not declare, or a declared pack without its index, is what C1 refuses.
    """
    import shutil

    if cameras is None and packs is not None:
        raise ValueError("packs 는 cameras 가 있을 때만 줍니다 — 카메라 없는 배치에는 팩이 없습니다")
    if cameras is not None and (the_pack(cameras) is None) != (packs is None):
        raise ValueError("packs 는 정본(meta/cameras.json)이 팩을 선언할 때만, 그리고 그때는 반드시 줍니다 — "
                         "선언 없는 meta/packs.json 이나 목록 없는 팩 선언은 검증기 C1 이 거부합니다")
    # Settled before anything is written: which files, and that each is the one the canon pins.
    shipped = shipped_asset_files(assets, referenced, repo_root)
    check_shipped_sources(shipped, repo_root)
    copied = []
    if cameras is not None:
        docs = [("meta/cameras.json", shipped_cameras(cameras))]
        if packs is not None:
            docs.append(("meta/packs.json", {str(i): e for i, e in sorted(packs.items())}))
        for rel, doc in docs:
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            copied.append(rel)
    scenarios = json.loads((repo_root / "meta/scenarios.json").read_text(encoding="utf-8"))
    rules = json.loads((repo_root / "meta/rules.json").read_text(encoding="utf-8"))
    out_scenarios, out_rules = release_canon(scenarios, rules, scenario_ids, triggered)
    for rel, doc in zip(NARROWED_CANON, (out_scenarios, out_rules)):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        copied.append(rel)
    for rel, dst_rel in [(r, r) for r in CANON_FILES] + list(CANON_RENAMES.items()):
        src = repo_root / rel
        if not src.exists():
            raise FileNotFoundError(f"배포에 필요한 파일이 없습니다: {rel}")
        dst = out / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst_rel)
    for rel in CANON_DIRS:
        src = repo_root / rel
        if not src.is_dir():
            raise FileNotFoundError(f"배포에 필요한 디렉터리가 없습니다: {rel}")
        shutil.copytree(src, out / rel, dirs_exist_ok=True)
        copied.append(rel + "/")

    # 이 배치가 실제로 쓴 로봇의 관절 정본만 싣는다.
    for model in sorted(robot_models):
        src = repo_root / "meta" / "joint_map" / f"{model}.yaml"
        if not src.is_file():
            raise FileNotFoundError(f"관절 정본이 없습니다: {src}")
        dst = out / "meta" / "joint_map" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(f"meta/joint_map/{src.name}")

    # The redistributable assets this batch's episodes use travel with the data, a USD's
    # textures beneath it where its ./textures/ references resolve; everything else is
    # referenced by path and SHA256 only (NOTICE §1). Each copy is hashed again: the bytes
    # that ship are the bytes the canon pins, not merely the ones that were read a moment ago.
    for rel, f in sorted(shipped.items()):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f.src, dst)
        if sha256_file(dst) != f.sha256:
            raise ValueError(f"{rel}: 복사한 파일의 해시가 정본과 다릅니다 ({f.asset})")
        copied.append(rel)

    # LeRobot writes info.json without our schema_version; C1 wants it.
    info_path = out / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["schema_version"] = SCHEMA_VERSION
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return copied


def sidecar_asset_paths(paths: Sequence[Path], assets: dict,
                        scenarios: dict[str, dict]) -> list[str]:
    """Every asset path the batch's sidecars say was loaded, joined to the canon: what the
    episodes' scene_assets will carry (build_episode_meta), known before a frame is written.

    Each sidecar must name what the canon says its scenario used and its robot is
    (required_scene_assets: the scene's declared assets, its human model's USD and the USD of
    the sidecar's robot_model) -- what ships is drawn from these, and the licence record is
    this list, so one left out would leave the distribution unnoticed. ``scenarios`` maps
    scenario_id to its meta/scenarios.json entry. Raises on an asset the canon does not know,
    as build_episode_meta would, and on a sidecar that omits a required asset."""
    used: set[str] = set()
    short: list[str] = []
    for p in paths:
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        loaded = {asset_entry(assets, path)["path"] for path in meta["asset_paths"].values()}
        missing = required_scene_assets(scenarios[meta["scenario_id"]], assets,
                                        meta.get("robot_model")) - loaded
        if missing:
            short.append(f"{p.stem}: {sorted(missing)}")
        used |= loaded
    if short:
        raise ValueError(
            f"사이드카 {len(short)}건의 asset_paths 가 시나리오가 쓴다고 말하는 자산을 빠뜨렸습니다 "
            "(scene.assets · human_model 의 USD · robot_model 의 USD) — 배포본에 싣는 자산과 자산 "
            "라이선스 기록은 에피소드의 scene_assets 에서 정해지므로 빠진 자산은 둘 다에서 빠집니다. "
            f"예: {'; '.join(short[:3])}")
    return sorted(used)


def preflight_media(paths: Sequence[Path], cameras_path: Path,
                    probe_video) -> tuple[dict | None, dict[Path, dict[str, Path]]]:
    """Everything about the batch's camera media that can be refused before a frame is
    written: returns (the camera canon, or None for a camera-less batch; npz path -> its
    media files).

    One condition set per batch (batch_conditions); for a camera batch, the canon's fps
    against every episode's recording period (check_canon_rate), every sidecar's ``media``
    against the canon (media_files: a pack exactly when the canon declares one), the recording's
    canon fingerprint against the canon's (check_canon_fingerprint), every stream's
    size and frame count (probe_video, a demux rather than a decode), no camera black in every
    frame (check_blank_frames) and, when the canon
    declares a pack, every pack's array headers (pack_layout) and its step_index
    (check_pack_steps, that one member inflated). A fault found
    halfway through the loop would leave a half-written dataset directory, as the
    scenario pre-check next to this already says."""
    from generation import cameras as camera_canon

    metas, per_episode = {}, {}
    for p in paths:
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(p, allow_pickle=True) as arrays:
            per_episode[p] = active_conditions(arrays, meta)
        metas[p] = meta
    if CAMERAS not in batch_conditions(per_episode):
        return None, {}
    if not cameras_path.is_file():
        raise FileNotFoundError(f"카메라가 기록된 배치인데 카메라 정본이 없습니다: {cameras_path}")
    cameras = camera_canon.load(cameras_path)
    pack = the_pack(cameras)
    files: dict[Path, dict[str, Path]] = {}
    for p, meta in metas.items():
        with np.load(p) as arrays:
            check_canon_rate(cameras, step_dt_from_timestamps(arrays["timestamp"]), meta["episode_id"])
        media = media_files(meta, p.parent, cameras)
        check_canon_fingerprint(meta, cameras)
        for cam in sorted(set(video_cameras().values())):
            check_video(cam, cameras, probe_video(media[cam]), meta["steps"])
        check_blank_frames(meta)
        if pack is not None:
            check_pack_layout(pack_layout(media[MEDIA_PACK]), cameras, meta["steps"])
            check_pack_steps(media[MEDIA_PACK], meta["steps"])
        files[p] = media
    return cameras, files


def conversion_order(episode_ids) -> list[str]:
    """``episode_ids`` in the order convert writes their episodes: their recordings' file names (<id>.npz) sorted,
    as convert takes a directory's recordings (recordings). A batch asks a chunk's rows in any order; whoever holds
    a converted distribution to the episodes it was asked for compares in this order."""
    return sorted(episode_ids, key=lambda eid: f"{eid}.npz")


def recordings(episodes: Path) -> list[Path]:
    """The recordings under ``episodes`` (its *.npz), in the order convert writes their episodes
    (conversion_order)."""
    found = {p.stem: p for p in Path(episodes).glob("*.npz")}
    return [found[eid] for eid in conversion_order(found)]


def convert(episodes: Path, out: Path, repo_id: str, *, repo_root: Path, create_dataset,
            decode_video=pyav_decode, probe_video=pyav_probe) -> int:
    """The conversion. ``create_dataset`` is LeRobotDataset.create in main(); the video
    decoder and prober are PyAV's. Injected so the whole path -- pre-checks, frames, packs
    (when the canon declares one), shipped canon -- runs in tests without lerobot or av."""
    import yaml

    root = Path(repo_root)
    out = Path(out)
    scenarios = json.loads((root / "meta/scenarios.json").read_text(encoding="utf-8"))
    assets = json.loads((root / "meta/assets.json").read_text(encoding="utf-8"))
    rules = json.loads((root / "meta/rules.json").read_text(encoding="utf-8"))

    scenario_task = {s["scenario_id"]: s["robot_task"] for s in scenarios["scenarios"]}

    paths = recordings(episodes)
    if not paths:
        print(f"에피소드가 없습니다: {episodes}")
        return 1

    # Before a single frame, like the checks below: which code recorded each episode and where
    # its run began. Found at the second episode, the first is already written.
    for p in paths:
        sidecar = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        check_generator_provenance(sidecar)
        require_run_start(sidecar)

    # Every scenario this batch references, checked before a single frame is
    # written. build_episode_meta raises on a missing field anyway, but it
    # would do so partway through -- leaving a half-written dataset directory
    # and costing the episodes already converted.
    sids = sorted({json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))["scenario_id"]
                   for p in paths})
    by_sid = {sc["scenario_id"]: sc for sc in scenarios["scenarios"]}
    need = ("safety_function", "domain", "task_archetype", "hazard_class")
    incomplete = {sid: [k for k in need if k not in by_sid.get(sid, {})]
                  for sid in sids}
    incomplete = {sid: absent for sid, absent in incomplete.items() if absent or sid not in by_sid}
    if incomplete:
        print("meta/scenarios.json 이 이 배치의 시나리오를 다 설명하지 못합니다:")
        for sid, absent in incomplete.items():
            print(f"  {sid}: {absent or '시나리오 자체가 없음'}")
        print("스키마 §5 의 필수 에피소드 필드입니다 — scenarios.json 을 먼저 채우십시오 "
              "(safety_function_note 참고).")
        return 1
    print(f"시나리오 정본 확인: {sids}")
    check_hri_declared(paths, by_sid)
    check_hri_completion(paths, by_sid)
    check_table_recordings(paths, by_sid, assets, root)
    # The same place, for the camera media: a mixed batch or a bad stream stops here.
    cameras, media = preflight_media(paths, root / "meta" / "cameras.json", probe_video)
    # The pack's name when the canon declares one (switched off by leaving it out).
    pack = the_pack(cameras) if cameras is not None else None
    if cameras is not None:
        print(f"카메라 매체 확인: 에피소드 {len(media)}건 · 영상 {sorted(set(video_cameras().values()))} · "
              + (f"팩 {pack}" if pack is not None else "팩 없음(정본이 선언하지 않음)"))
    # And for the assets: which files this batch ships and that each is the one the canon
    # pins. assemble_dataset settles the same after the last episode -- a missing texture
    # found there has cost the whole conversion.
    shipped = shipped_asset_files(assets, sidecar_asset_paths(paths, assets, by_sid), root)
    check_shipped_sources(shipped, root)
    print(f"자산 동봉 확인: {len(shipped)}개 파일 — "
          f"{sorted({f.asset for f in shipped.values()}) or '재배포 자산 없음'}")

    dataset = None
    episode_records: list[dict] = []
    packs: dict[int, dict] = {}
    for npz_path in paths:
        meta = json.loads(npz_path.with_suffix(".json").read_text(encoding="utf-8"))
        label = json.loads(npz_path.with_suffix(".labels.json").read_text(encoding="utf-8"))
        joint_map = yaml.safe_load(
            (root / f"meta/joint_map/{meta['robot_model']}.yaml").read_text(encoding="utf-8")
        )
        arrays = dict(np.load(npz_path, allow_pickle=True))
        step_dt = step_dt_from_timestamps(arrays["timestamp"])
        episode_meta = build_episode_meta(meta, label, scenarios, joint_map, assets,
                                          rules, step_dt=step_dt)
        frames = step_frames(arrays, meta, joint_map)

        # LeRobot requires a natural-language "task" on every frame. It is a
        # LeRobot concept, not a schema §6 field, so step_frames() does not
        # produce it and field_map does not list it -- it is attached here,
        # from meta/scenarios.json's own robot_task, so the string is not
        # written down a second time.
        task = scenario_task[meta["scenario_id"]]
        dtypes = recorded_dtypes(arrays, meta)
        lr_frames = [to_lerobot_frame(f, dtypes) for f in frames]

        if dataset is None:
            dataset_fps = episode_meta["fps"]
            # 반올림한 fps 만 대조하면 0.0200 과 0.0201 이 둘 다 50 이 되어 통과하고,
            # LeRobot 은 timestamp 를 frame_index/fps 로 다시 만든다 — 100번 프레임이
            # 원본 2.01 s 인데 배포본에는 2.00 s 로 실린다. 실제 주기로도 대조한다.
            dataset_step_dt = step_dt
            check_record_rate(step_dt, dataset_fps)
            features = dataset_features(lr_frames[0], cameras)
            dataset = create_dataset(
                repo_id=repo_id, fps=dataset_fps, root=str(out), features=features,
                # Pictures go to disk as they arrive, so an episode's video is never held whole.
                **({"image_writer_threads": 4} if cameras is not None else {}),
            )
        else:
            check_record_rate(step_dt, dataset_fps, dataset_step_dt,
                              episode_meta["episode_id"])
        if cameras is not None:
            streams = {schema: decode_video(media[npz_path][cam])
                       for schema, cam in video_cameras().items()}
            shapes = {schema: features[schema]["shape"] for schema in streams}
            lr_frames = with_video_frames(lr_frames, streams, shapes)
        for frame in lr_frames:
            dataset.add_frame(dict(frame, task=task))
        # No episode_data argument: LeRobot's save_episode(episode_data=...)
        # expects a whole frame buffer, not extra metadata. The schema §5
        # record is merged into meta/episodes/*.parquet after finalize().
        dataset.save_episode()
        if pack is not None:
            index = len(episode_records)
            packs[index] = write_pack(media[npz_path][MEDIA_PACK], out, index, cameras,
                                      meta["steps"])
        episode_records.append(episode_meta)
        print(f"  {episode_meta['episode_id']}: {len(frames)} 프레임 "
              f"({episode_meta['outcome']}/{episode_meta['severity']})")
    dataset.finalize()
    if remove_image_staging(out):
        print(f"LeRobot 이 남긴 빈 프레임 스테이징 {IMAGE_STAGING}/ 를 지웠습니다")
    normalise_stats(out)
    blanked = blank_episode_stats(out)
    if blanked:
        print(f"에피소드별 통계 {blanked}개 열을 비웠습니다 — 비유한값이 있는 필드")
    meta_path = merge_episode_metadata(out, episode_records)
    print(f"에피소드 메타 병합: {meta_path}")
    triggered = sorted({r for rec in episode_records
                        for r in (rec.get("triggered_rules") or [])})
    models = sorted({str(rec["robot_model"]) for rec in episode_records})
    # What the episodes say they use, as shipped in meta/episodes -- the set C6 reads back.
    used = sorted({a["path"] for rec in episode_records for a in rec["scene_assets"]})
    copied = assemble_dataset(root, out, assets, sids, triggered, models,
                              cameras=cameras, packs=packs if pack is not None else None,
                              referenced=used)
    print(f"정본·자산 동봉 {len(copied)}건: {', '.join(copied)}")
    # The recorded streams stay where they are: they are the ground for a re-conversion,
    # and removing them is a person's call.
    print(f"변환 완료: {out} · 에피소드 {len(paths)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="LeRobotDataset v3.0 변환")
    ap.add_argument("--episodes", default=".tmp/episodes/S1a")
    ap.add_argument("--out", default=".tmp/lerobot/S1a")
    ap.add_argument("--repo-id", default="keti/humanoid-safety-s1a")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return convert(Path(args.episodes), Path(args.out), args.repo_id,
                   repo_root=Path(__file__).resolve().parents[1],
                   create_dataset=LeRobotDataset.create)


if __name__ == "__main__":
    raise SystemExit(main())
