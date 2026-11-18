#!/usr/bin/env python3
"""
검증기 테스트용 합성 데이터셋 생성기
==================================

실데이터가 나오기 전에 validate_dataset.py 가 제대로 잡아내는지 확인하기 위한 픽스처.
정상 데이터셋 하나와, 결함을 하나씩 심은 변형들을 만든다.

    python make_fixture.py [--with-media] <출력_디렉터리>

생성물
    clean/              결함 없음 → ERROR 0 이어야 함. 카메라 없는 배포본(v0.1 과 같은 꼴). 자산 정본은
                        텍스처가 딸린 재배포 자산 하나를 더 적지만 에피소드가 쓰지 않아 싣지 않는다
    clean_media/        --with-media 일 때만. clean 에 카메라 매체를 더한 배포본 — meta/cameras.json ·
                        videos/ · 상자·카메라 자세 열 — 이고, S1d 처럼 에피소드가 그 텍스처 자산을
                        써서 assets/ 와 assets/textures/ 에 싣는다. 정본이 팩을 선언하지 않아
                        packs/ · meta/packs.json 이 없다(팩은 정본에서 끈다).
                        ERROR 0 이어야 함. broken_media_* · broken_texture_* 변종은 이것을 복사해
                        만들므로 --with-media 없이는 만들지 않는다
    clean_media_pack/   --with-media 일 때만. clean_media 에 팩 하나를 선언한 정본과 packs/ ·
                        meta/packs.json 을 더한 배포본. ERROR 0 이어야 함. 팩 코드는 지우지 않고
                        정본에서 끄므로, 팩을 통과시키는 쪽도 계속 시험받게 둔다. 팩의 결함을 심는
                        변종(PACK_VARIANTS)은 이것을 복사한다
    clean_hri/          hri 배포본 — 정본의 S1e(규칙 · 필수 열 · 목표 비율 · 조건표)와 그 열, 조건표가 있는
                        시나리오의 에피소드 열 스물하나(모두 무작위 행). 카메라 없음.
                        ERROR 0 이어야 함. broken_hri_* 변종은 이것을 복사한다
    clean_hri_media/    --with-media 일 때만. clean_hri 에 clean_media 의 카메라 매체를 더한 배포본 —
                        정본의 카메라 전부와 로봇 몸의 카메라마다 보이는 몫 열. ERROR 0 이어야 함
    clean_hri_lean/     clean_hri 에서 선택 스텝 열 가운데 시뮬레이터가 기록하는 아홉(관절 · 행동 · IMU ·
                        접촉 상대속도 · 무게중심 · ZMP · 안정 여유)을 뺀 배포본 — 스텝 파일 · features ·
                        통계 어디에도 그 열이 없다. ERROR 0 이어야 함
    broken_dof/         C2: joint_names 길이 != dof
    broken_region/      C3: 허용되지 않은 부위 코드
    broken_label/       C4: outcome=fail 인데 triggered_rules 비어 있음
    broken_seed/        C5: source=sim 인데 seed 없음
    broken_notice/      C6: 재배포 자산이 NOTICE 에 없음
    broken_asset_unreferenced/ C6: 에피소드가 쓰지 않는 재배포 자산을 assets/ 에 실음
    broken_asset_left_out/     C6: 인체 모델의 USD 를 모든 scene_assets 에서 빼고 파일도 지움
    broken_asset_usd_bytes/    C6: 재배포 USD 가 이름은 맞는데 usd_sha256 이 고정한 바이트가 아님
    broken_time/        C7: frame_index 결측
    broken_values/      C8: 접촉력 음수 + NaN, stability_margin +inf
    broken_rules/       C1: 값도 조항 참조도 없는 임계값
    broken_not_applicable_on_permitted/ C1: 접촉 허용 부위에 not_applicable 면제
    broken_not_applicable_with_value/   C1: not_applicable 인데 값도 있음
    broken_contact_permitted_dropped/   C1: contact_permitted 플래그 삭제
    broken_threshold_nonfinite/ C1: 임계값이 NaN (float 이라 수치 검사를 통과한다)
    broken_missing_parameter/  C1: 필수 파라미터 이름 자체가 없음
    broken_duplicate_rule_id/  C1: 같은 rule_id 가 둘 (검증기와 생성기가 다른 것을 읽는다)
    broken_threshold_negative/  C1: 임계값이 음수
    broken_contact_type/ C3: 허용되지 않은 접촉 유형
    broken_zmp_half_nan/ C8: ZMP 한쪽 성분만 NaN (표식은 두 성분 모두)
    broken_com_nan/     C8: CoM 에 NaN (무게중심에는 표식이 없음)
    broken_inputs_missing/ C1: 입력이 없어 계산할 수 없는 규칙
    broken_rule_inputs/ C8: 적용 규칙이 데이터에 없는 입력을 요구
    broken_scenario_rules/ C1: 적용 규칙 선언이 실재하지 않는 규칙을 가리킴
    broken_required_fields/ C8: 시나리오가 선언한 필수 필드가 데이터에 없음
    broken_empty_rules/ C1: 규칙 정본이 비어 적용 규칙을 검증할 수 없음
    broken_safety_schema/ C1: 동봉된 기계가독 규격이 검증기와 다름
    broken_span_swapped/            C7: 두 에피소드의 데이터 구간(dataset_from_index·dataset_to_index)이 서로 바뀜
    broken_span_end_short/          C7: 한 에피소드의 dataset_to_index 가 한 행 짧음
    broken_span_file_index/         C7: 한 에피소드의 data/file_index 가 다른 스텝 파일을 가리킴
    broken_span_rows_out_of_place/  C7: 두 에피소드의 행이 파일 안에서 서로의 자리에 놓임 (포인터는 그대로)
    broken_total_frames/            C7: info.json 의 total_frames 가 스텝 표의 행 수보다 하나 적음
    broken_total_episodes/          C7: info.json 의 total_episodes 가 에피소드 표의 행 수보다 하나 많음
    broken_task_count/              C7: info.json 의 total_tasks 가 0 (과제 표와 스텝은 그대로)
    broken_task_table_absent/       C7: meta/tasks.parquet 이 없음
    broken_task_sentences_not_index/ C7: 과제 표의 문장이 pandas 인덱스가 아니라 열로만 실림
    broken_task_index_outside/      C7: 한 스텝의 task_index 가 과제 표의 행 수와 같음 (표 밖)
    broken_task_index_beyond_feature/ C7: features 가 task_index 를 int8 로 선언했는데 한 스텝이 128
    broken_features_absent/         C7: info.json 에 features 가 없음
    broken_feature_undeclared_column/ C7: features 가 스텝 표의 열 하나를 선언하지 않음 (파일은 그대로)
    broken_feature_column_absent/   C7: features 가 어느 스텝 파일에도 없는 열을 선언
    broken_feature_shape/           C7: features 가 길이 3 인 열을 길이 2 로 선언
    broken_feature_dtype_name/      C7: 한 feature 의 dtype 이 pyarrow 의 이름이 아니라 numpy 의 이름("<f8")
    broken_feature_frame_index_width/ C7: features 가 frame_index 를 int8 로 선언 (에피소드는 127 스텝을 넘음)
    broken_media_frame_count/       C7: 카메라 영상 구간이 에피소드 스텝보다 한 프레임 짧음
    broken_media_pack_missing/      C1·C7: meta/packs.json 이 적은 팩 하나가 없음 (clean_media_pack 복사)
    broken_media_bbox_out_of_range/ C8: 상자 하나가 영상 오른쪽 끝을 한 픽셀 넘음
    broken_media_stray_pack/        C1: 정본이 팩을 선언하지 않았는데 packs/ · meta/packs.json 이 있음
    broken_texture_missing/         C6: 싣는 USD 가 쓰는 텍스처 하나가 없음
    broken_texture_bytes/           C6: 텍스처 하나의 바이트가 출처 기록(bundle.files)의 해시와 다름
    broken_texture_unlisted/        C6: assets/textures/ 에 출처 기록이 없는 파일
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_dataset import (  # noqa: E402
    GRID_NOT_A_CELL, VISIBLE_FRACTION_PREFIX, iter_threshold_entries, threshold_problems)

SCHEMA_VERSION = "0.2"
REGIONS = [f"BR-{i:02d}" for i in range(1, 13)]

REPO_ROOT = Path(__file__).resolve().parent.parent
JOINT_MAP = REPO_ROOT / "meta" / "joint_map" / "unitree_h1.yaml"
SAFETY_SCHEMA = REPO_ROOT / "meta" / "safety_schema.json"
CAMERAS = REPO_ROOT / "meta" / "cameras.json"


def json_encoded_fields() -> dict[str, list[str]]:
    """어느 필드가 parquet 에 JSON 문자열로 실리는지 -- 정본이 정한다.

    픽스처가 실제 배포본과 다른 표현을 쓰면, 검증기의 JSON 문자열 처리가 깨져도
    픽스처 회귀는 초록으로 남는다. 그 경로가 배포본이 실제로 쓰는 경로다.
    """
    canon = json.loads(SAFETY_SCHEMA.read_text(encoding="utf-8"))
    return canon["json_encoded_fields"]


def as_published(value: object) -> str:
    """변환기가 배포본에 싣는 형태 -- 한 칸짜리 JSON 문자열."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return json.dumps(value, ensure_ascii=False)


def write_steps(df: pd.DataFrame, path: Path) -> None:
    """Write a step table the way a distribution holds it: NaN stays NaN.

    ``df.to_parquet`` turns NaN into null on the way out -- in a float column and inside a
    list alike. A distribution carries NaN (LeRobot writes the arrays as they are), the loader
    dies on a null, and C7 reports one (c7_step_nulls). The schema is the one pandas would have
    written, so nothing else about the file changes.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    def plain(v):
        return [plain(x) for x in v] if isinstance(v, np.ndarray) else v

    schema = pa.Table.from_pandas(df, preserve_index=False).schema
    arrays = []
    for field in schema:
        col = df[field.name]
        values = [plain(v) for v in col] if col.dtype == object else col.to_numpy()
        arrays.append(pa.array(values, type=field.type, from_pandas=False))
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path)


def load_joint_map() -> dict:
    """관절 이름·순서를 매핑 파일에서 읽는다.

    스키마 정의서 §7.1 — 관절 순서의 정본은 자산에서 추출한 값이다. 픽스처가 값을
    수기로 복제하면 그 복제본이 자산과 어긋나도 아무도 알아채지 못하므로, 여기서는
    `generation/dump_joint_names.py` 의 출력을 그대로 읽어 쓰고 파일도 그대로 복사한다.
    """
    if not JOINT_MAP.exists():
        raise SystemExit(
            f"매핑 파일이 없습니다: {JOINT_MAP.relative_to(REPO_ROOT)}\n"
            "  ~/isaacsim/python.sh generation/dump_joint_names.py \\\n"
            "      --usd <H1 USD 경로> --robot-model unitree_h1"
        )
    return yaml.safe_load(JOINT_MAP.read_text(encoding="utf-8"))


_JM = load_joint_map()
JOINT_NAMES = list(_JM["joint_names"])
DOF = int(_JM["dof"])
CANONICAL = dict(_JM["canonical_map"])

LOCK_PATH = REPO_ROOT / "environment.lock"


def load_sim_backend() -> dict:
    """environment.lock 의 sim_backend 를 그대로 쓴다 (스키마 §5).

    버전 문자열을 픽스처에 수기로 적으면 실제 생성 환경과 어긋난다. 파이프라인이
    lock 에서 스탬프하도록 설계했으므로 픽스처도 같은 경로를 탄다.
    """
    if not LOCK_PATH.exists():
        raise SystemExit(
            f"버전 고정본이 없습니다: {LOCK_PATH.relative_to(REPO_ROOT)}\n"
            "  make lock"
        )
    d = yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8")) or {}
    backend = d.get("sim_backend") or {}
    # "unknown" 을 채워 넘어가면 C5 는 키 존재만 보므로 정본이 사라진 채로 clean 픽스처가
    # 통과한다. 다른 정본과 같이 없으면 실패시킨다.
    missing = [k for k in ("isaacsim", "isaaclab", "physics", "lerobot")
               if not backend.get(k)]
    if missing:
        raise SystemExit(f"environment.lock 의 sim_backend 가 비어 있습니다: {missing}\n"
                         "  make lock")
    return {k: str(v) for k, v in backend.items()}


RULES_PATH = REPO_ROOT / "meta" / "rules.json"
SCENARIOS_PATH = REPO_ROOT / "meta" / "scenarios.json"

# What a distribution's sim_backend says about this repository's code. A constant, not read from
# git: the fixture has to come out the same from a shallow clone and from an unpacked archive.
FIXTURE_GENERATOR_COMMIT = "0123456789abcdef0123456789abcdef01234567"

# 단위별 합성 임계값. 실제 표준 수치가 아니며 검증기 시험 용도로만 쓴다.
SYNTHETIC = {"N": 100.0, "N/cm^2": 10.0, "kg": 1.0, "N/mm": 10.0,
             "s": 0.5, "m": 0.05, "ratio": 0.2}


def load_rules_filled() -> dict:
    """정본 rules.json 을 읽어 비어 있는 임계값만 합성값으로 채운다.

    rule_id 는 정본과 같아야 C4 의 "rules.json 에 없는 rule_id" 검사가 의미를 가지므로
    픽스처가 목록을 복제하지 않고 읽어 쓴다. 반면 임계값은 유료 표준에서 전재하는
    수치이므로 픽스처에 실제 값을 넣지 않고 합성값으로 대체한다.
    """
    if not RULES_PATH.exists():
        raise SystemExit(f"판정 규칙 파일이 없습니다: {RULES_PATH.relative_to(REPO_ROOT)}")
    d = json.loads(RULES_PATH.read_text(encoding="utf-8"))

    # 검증기와 같은 계약으로 임계값 위치를 찾는다. 모양을 추측하면 정본 구조가 바뀌었을 때
    # 픽스처가 그 결함을 조용히 덮는다.
    for path, e in iter_threshold_entries(d):
        if not isinstance(e, dict) or "unit" not in e:
            raise SystemExit(f"rules.json 의 임계값 항목 형식이 예상과 다릅니다: {path}")
        if e["unit"] is None:
            # 단위를 픽스처가 지어내면 의미가 어긋난 픽스처가 만들어진다 (초 단위 낙하높이 등).
            raise SystemExit(f"rules.json 에 단위가 없습니다: {path} — 정본을 먼저 채우십시오")
        if e.get("not_applicable"):
            # 표준이 그 자리에 수치를 두지 않는다고 선언한 항목이다. 채우면 픽스처가
            # 정본과 다른 주장을 하게 된다.
            continue
        if (e.get("determined_by") == "standard"
                and str(e.get("standard_ref") or "").strip()):
            # standard 항목은 손대지 않고 정본을 그대로 물려받는다 — 값이 실려
            # 있으면 값을, 조항 참조만 실려 있으면 그 형태를. 픽스처가 합성값으로
            # 덮으면 배포본에 없는 표현을 시험하게 되고, 정본이 실제로 쓰는 경로의
            # 결함이 초록으로 남는다 (CONTRIBUTING.md 「함께 고치는 곳」: 임계값 계약의 세 형태).
            continue
        if e.get("value") is None:
            e["value"] = SYNTHETIC.get(e["unit"], 0.5)

    # 배포 가능한 데이터셋은 계산할 수 없는 규칙을 담지 않는다 (C1). 스키마에 입력 필드가
    # 생기면 정본에서 inputs_missing 이 사라지고 이 규칙도 픽스처에 다시 들어온다.
    dropped = [r["rule_id"] for r in d["rules"] if r.get("inputs_missing")]
    d["rules"] = [r for r in d["rules"] if not r.get("inputs_missing")]
    if dropped:
        d["excluded_uncomputable_rules"] = dropped

    d["status"] = "complete"

    # 채운 뒤 계약 위반이 남아 있으면 픽스처가 결함을 숨기게 된다.
    remaining = threshold_problems(d)
    if remaining:
        raise SystemExit("합성 임계값을 채운 뒤에도 계약 위반이 남았습니다: "
                         + "; ".join(f"{a}: {b}" for a, b in remaining[:3]))
    d["description"] = "검증기 시험용 합성 픽스처. 임계값은 실제 표준 수치가 아니다."
    return d


FIXTURE_SCENARIO = "S1a"


def load_scenarios_slice() -> dict:
    """정본 scenarios.json 에서 픽스처가 쓰는 시나리오만 남긴다.

    적용 규칙 선언을 픽스처가 복제하면, 정본이 데이터에 없는 입력을 요구하도록 바뀌어도
    픽스처는 계속 통과한다 — C8 이 막아야 할 바로 그 상황을 픽스처가 덮는 것이다.
    joint_map 과 같은 이유로 여기서도 정본을 읽어 쓴다.
    """
    if not SCENARIOS_PATH.exists():
        raise SystemExit(f"시나리오 정의 파일이 없습니다: {SCENARIOS_PATH.relative_to(REPO_ROOT)}")
    d = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    picked = [s for s in d.get("scenarios", []) if s.get("scenario_id") == FIXTURE_SCENARIO]
    if len(picked) != 1:
        # 0 이면 정본이 바뀐 것이고, 2 이상이면 어느 쪽이 정본인지 알 수 없다. 둘 다
        # 조용히 넘어가면 픽스처가 엉뚱한 선언을 시험하게 된다.
        raise SystemExit(
            f"정본에서 {FIXTURE_SCENARIO} 를 {len(picked)}건 찾았습니다 (1건이어야 함): "
            f"{SCENARIOS_PATH.relative_to(REPO_ROOT)}")
    return {"scenarios_version": d.get("scenarios_version"),
            "schema_version": d.get("schema_version"),
            "status": "complete",
            "description": f"검증기 시험용 합성 픽스처. 정본에서 {FIXTURE_SCENARIO} 만 발췌했다.",
            "scenarios": picked}


HRI_CLEAN = "clean_hri"
# The hri distribution with the camera media: every camera of the canon, and a visible fraction per
# camera on the robot.
HRI_MEDIA_CLEAN = "clean_hri_media"
# The hri distribution without the optional step fields a simulator records (build_hri_lean).
HRI_LEAN_CLEAN = "clean_hri_lean"
HRI_SCENARIO = "S1e"
HRI_VARIANT_PREFIX = "broken_hri_"


def load_hri_scenarios_slice() -> dict:
    """The hri distribution's meta/scenarios.json: the canon's S1e -- the rules, required fields and
    outcome target the validator holds the data to -- with the scene of the camera-less fixture
    scenario (fixture_scene). The fixture makes no file for a redistributed scene asset, so the
    scene is borrowed, for the reason the camera-less fixture keeps to its own scenario."""
    if not SCENARIOS_PATH.exists():
        raise SystemExit(f"시나리오 정의 파일이 없습니다: {SCENARIOS_PATH.relative_to(REPO_ROOT)}")
    d = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    picked = [s for s in d.get("scenarios", []) if s.get("scenario_id") == HRI_SCENARIO]
    if len(picked) != 1:
        raise SystemExit(f"정본에서 {HRI_SCENARIO} 를 {len(picked)}건 찾았습니다 (1건이어야 함): "
                         f"{SCENARIOS_PATH.relative_to(REPO_ROOT)}")
    return {"scenarios_version": d.get("scenarios_version"),
            "schema_version": d.get("schema_version"),
            "status": "complete",
            "description": (f"검증기 시험용 합성 픽스처. 정본에서 {HRI_SCENARIO} 만 발췌하고 장면은 "
                            f"{FIXTURE_SCENARIO} 의 것을 빌렸다."),
            "scenarios": [dict(picked[0], scene=fixture_scene())]}

ASSETS = [
    {"path": "robots/unitree_h1/h1.usd", "license": "BSD-3", "redistributed": True},
    {"path": "keti/dummy_ts15066_v1.usd", "license": "KETI-own", "redistributed": True},
    {"path": "keti/scene_flat_primitive.usd", "license": "KETI-own", "redistributed": True},
]
# ASSETS[1] is the fixture's human model: its meta/assets.json key is the fixture scenario's
# scene.human_model, read from the canon (human_model_key), so C6's anchor -- each episode lists
# its scenario's scene.assets and its human model's USD -- holds for the same reason it holds
# in a release, and fails when the canon moves. ASSETS[0] is the robot:
# meta/assets.json robots.unitree_h1, the episodes' robot_model, which the anchor asks for too.
# It ships as the H1 does, as its own folder (ROBOT_DIR, below).
HUMAN_MODEL_ASSET = 1
ROBOT_ASSET = 0


def fixture_scene() -> dict:
    """The fixture scenario's scene block, from the canon (load_scenarios_slice)."""
    return load_scenarios_slice()["scenarios"][0]["scene"]


def human_model_key() -> str:
    model = fixture_scene().get("human_model")
    if not isinstance(model, str) or not model:
        raise SystemExit(f"정본의 {FIXTURE_SCENARIO} scene.human_model 이 문자열이 아닙니다: {model!r} — "
                         "픽스처의 인체 모델 자산을 정할 수 없습니다")
    return model


def scenario_scene_assets() -> list[tuple[str, dict]]:
    """The fixture scenario's scene.assets as the canon registers them: (key, entry) each.

    Copied from the canon, not restated: an episode must list them (C6's anchor), and C6
    compares each item with the shipped meta/assets.json. A redistributable one would need a
    file in assets/ the fixture does not make -- refused instead of shipped wrong."""
    canon = json.loads((REPO_ROOT / "meta" / "assets.json").read_text(encoding="utf-8"))
    by_path = {e["usd_path"]: (k, e) for g in ("robots", "scene")
               for k, e in (canon.get(g) or {}).items() if isinstance(e, dict) and e.get("usd_path")}
    out = []
    for path in fixture_scene().get("assets") or []:
        if path not in by_path:
            raise SystemExit(f"정본의 {FIXTURE_SCENARIO} scene.assets 가 meta/assets.json 에 없는 자산을 "
                             f"가리킵니다: {path}")
        key, entry = by_path[path]
        if entry.get("redistributed"):
            raise SystemExit(f"정본의 {FIXTURE_SCENARIO} scene.assets 의 {key} 가 재배포 자산입니다 — "
                             "픽스처는 그 파일을 만들지 않으므로 자리표시 바이트와 해시를 더해야 합니다")
        out.append((key, {"usd_path": path, "license": entry["license"], "redistributed": False}))
    return out


def episode_assets() -> list[dict]:
    """What a fixture episode's scene_assets lists: ASSETS and the scenario's scene assets."""
    return ASSETS + [{"path": e["usd_path"], "license": e["license"],
                      "redistributed": e["redistributed"]} for _, e in scenario_scene_assets()]


# A redistributable asset with textures, like the S1d scene and skinned dummy. Both
# fixtures' meta/assets.json declare it, as a release's does whatever the batch; only the camera
# fixture's episodes use it, so only clean_media ships it -- the USD at assets/<file>, its
# textures at assets/textures/<file> with their provenance in bundle.files. clean is v0.1's
# shape: the canon marks it redistributable and nothing may ask for it there.
TEXTURED_ASSET = {"path": "keti/scene_textured.usd", "license": "KETI-own", "redistributed": True}
USD_PLACEHOLDER = b"#usda 1.0\n"
TEXTURES = {"floor_albedo.png": b"fixture texture: floor albedo\n",
            "floor_normal.png": b"fixture texture: floor normal\n"}
TEXTURE_VARIANT_PREFIX = "broken_texture_"


def textured_asset_entry() -> dict:
    """The meta/assets.json entry of TEXTURED_ASSET: a bundle.files row per texture, as the
    converter's canon carries them (generation/assets.py TEXTURE_ROW_KEYS + member)."""
    return {
        "usd_path": TEXTURED_ASSET["path"], "license": TEXTURED_ASSET["license"],
        "redistributed": True, "usd_sha256": hashlib.sha256(USD_PLACEHOLDER).hexdigest(),
        "bundle": {
            "textures_dir": "keti/textures", "textures_license": "CC0-1.0",
            "files": [{"file": name, "sha256": hashlib.sha256(data).hexdigest(),
                       "license": "CC0-1.0", "url": "https://example.invalid/fixture-textures.zip",
                       "member": f"floor/{name}"}
                      for name, data in sorted(TEXTURES.items())],
        },
    }


# The robot ships as the H1 does: its own folder assets/<ROBOT_DIR>/, the USD the episodes name, the layer
# it references beside it and its licence file, each a row of meta/assets.json robots.unitree_h1.ship.files.
# The USD's row is the robot's usd_sha256.
ROBOT_DIR = "unitree_h1"
ROBOT_FILES = {"h1.usd": USD_PLACEHOLDER, "instanceable_meshes.usd": b"#usda 1.0\n# fixture meshes\n",
               "LICENSE.txt": b"fixture licence text\n"}


def robot_ship_entry() -> dict:
    """The ship block of the fixture's robot (generation/assets.py ROBOT_SHIP_KEYS)."""
    return {"dir": ROBOT_DIR, "source_dir": "robots/unitree_h1", "license_file": "LICENSE.txt",
            "files": [{"file": name, "sha256": hashlib.sha256(data).hexdigest(), "license": "BSD-3",
                       "url": f"{{ISAACLAB_NUCLEUS_DIR}}/Robots/Unitree/{name}"}
                      for name, data in ROBOT_FILES.items()]}


def ship_robot(root: Path) -> None:
    (root / "assets" / ROBOT_DIR).mkdir(parents=True, exist_ok=True)
    for name, data in ROBOT_FILES.items():
        (root / "assets" / ROBOT_DIR / name).write_bytes(data)


def ship_textured_asset(root: Path) -> None:
    (root / "assets" / Path(TEXTURED_ASSET["path"]).name).write_bytes(USD_PLACEHOLDER)
    (root / "assets" / "textures").mkdir(parents=True, exist_ok=True)
    for name, data in TEXTURES.items():
        (root / "assets" / "textures" / name).write_bytes(data)

NOTICE = """NOTICE — 포함 자산의 출처 및 라이선스

1) Unitree H1 (h1.usd)
   출처: NVIDIA Isaac Sim 로봇 자산 카탈로그
   라이선스: BSD-3-Clause
   [BSD-3-Clause 전문을 여기에 그대로 수록할 것]

2) keti/dummy_ts15066_v1.usd
   ISO/TS 15066 Table A.2 신체 부위 구분 및 생체역학 파라미터에 기반해
   한국전자기술연구원이 자체 제작. 표준 문서 자체는 포함하지 않음.

3) keti/scene_flat_primitive.usd
   한국전자기술연구원 자체 제작 기하 씬.

4) keti/scene_textured.usd (카메라 배포본에만 실림)
   한국전자기술연구원 자체 제작 씬. 텍스처는 CC0 — 파일마다의 출처는
   meta/assets.json 의 bundle.files.
"""

FPS = 100.0
DUR = 4.0


# 정본이 조항 참조만 싣는 SSM 파라미터에 넣을 합성값. 회랑을 만든 입력 자리이므로
# 비워 둘 수 없고, 정본과 대조되지도 않는다.
_SSM_SYNTHETIC = {
    "human_speed": 1.6, "robot_reaction_time": 0.1, "robot_stop_time": 0.34,
    "intrusion_distance": 0.2,
    "position_uncertainty_robot": 0.0, "position_uncertainty_human": 0.0,
}


def _synthetic_ssm_params() -> dict:
    """정본이 값을 실은 항목은 정본에서, 나머지는 합성값에서 가져온다."""
    carried = {}
    for rule in json.loads(RULES_PATH.read_text(encoding="utf-8")).get("rules") or []:
        if rule.get("rule_id") == "R-TS15066-SSM-01":
            carried = {name: entry["value"]
                       for name, entry in (rule.get("parameters") or {}).items()
                       if isinstance(entry, dict) and entry.get("value") is not None}
            break
    return {name: carried.get(name, default)
            for name, default in _SSM_SYNTHETIC.items()}


def hri_fields() -> list[str]:
    """The hri step fields, read from the shipped schema's conditional list."""
    schema = json.loads(SAFETY_SCHEMA.read_text(encoding="utf-8"))
    return list(schema["conditional_required_step_fields"]["hri"])


def _directed_ssm_params() -> dict:
    """notes.ssm_params of an hri episode: the canon's R-TS15066-SSM-02 values and its stop table."""
    rule = next(r for r in json.loads(RULES_PATH.read_text(encoding="utf-8"))["rules"]
                if r.get("rule_id") == "R-TS15066-SSM-02")
    return {**{name: entry["value"] for name, entry in rule["parameters"].items()},
            "stop_table": rule["stop_table"]}


def make_episode(seq: int, outcome: str, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    n = int(FPS * DUR)
    seed = 20260901_00000 + seq
    # 배포본과 같은 형식이어야 한다 (규격서 §3). 픽스처가 다른 형식을 쓰면 C5 의
    # 식별자 검사가 시험되지 않은 채로 남는다 — 실제로 그렇게 남아 있었고, 그
    # 사이에 규격서 §3 이 배포본과 다른 형식을 싣고 있었다.
    eid = f"S1a_{seq:04d}_seed{seed}"

    if outcome == "fail":
        trig = ["R-TS15066-A2-05"]
        # 접촉이 있는 fail 이므로 S3 이다. S2 는 접촉 없는 fail 이고,
        # 이 에피소드는 아래에서 contact_onset 을 함께 낸다 — 픽스처가 규칙과 다른
        # 조합을 보여 주면 "정상 데이터셋" 의 예시로서 거짓말을 한다.
        sev = "S3"
    elif outcome == "near_miss":
        trig = ["R-TS15066-SSM-01"]
        sev = "S1"
    else:
        trig = []
        sev = "S0"

    t = np.arange(n) / FPS
    ph = rng.uniform(0, 2 * np.pi, DOF)
    jp = 0.3 * np.sin(2 * np.pi * 0.8 * t[:, None] + ph)
    jv = np.gradient(jp, 1 / FPS, axis=0)
    jt = 12.0 * jv + rng.normal(0, 0.3, jp.shape)

    # nominal 은 요구 이격거리 아래로 내려가면 안 된다. 3.0 - 0.55t 는 t=3.8s 에
    # 0.9 를 밑돌아, 「nominal 인데 separation_violation 18프레임」 이라는 스스로
    # 모순된 에피소드를 만들었다 — C4 는 메타끼리만 대조하므로 잡지 못한다.
    req = np.full(n, 0.9 if outcome == "nominal" else 1.3)
    approach = 0.15 if outcome == "nominal" else 0.55
    dist = np.clip(3.0 - approach * t + rng.normal(0, 0.01, n), 0.02, None)
    if outcome == "nominal":
        dist = np.maximum(dist, req + 0.2)       # 여유를 두고 접근만 한다
    viol = dist < req

    # A handful of flight-phase steps (both feet unloaded at once) read
    # -inf, the documented sentinel (schema §6 "stability_margin 산출
    # 방법") — real S1a data has this on most steps of ordinary dynamic
    # gait. A sprinkling here is enough to prove C8 does not flag the
    # sentinel itself as a defect on an otherwise-clean dataset.
    stability_margin = np.full(n, 0.08)
    stability_margin[::37] = float("-inf")

    force = np.zeros((n, len(REGIONS)))
    press = np.zeros((n, len(REGIONS)))
    # 접촉 지점 상대속도(§6). 접촉이 없는 스텝에도 값이 있습니다 — 로봇과 인체는
    # 닿지 않아도 서로에 대해 움직이므로 0 이 아니고, 배포본도 그렇게 기록합니다.
    # 픽스처가 접촉 스텝만 0 이 아니게 만들면 배포본과 다른 표현을 시험하게 됩니다.
    v_rel = np.abs(rng.normal(0.6, 0.15, (n, len(REGIONS))))
    # 부위별 접촉 유형. Table A.2 는 과도적/준정적 한계가 다르므로 어느 쪽을
    # 적용했는지 데이터가 말해 주어야 한다 (§6, §8).
    ctype = [["none"] * len(REGIONS) for _ in range(n)]
    events: list[list[str]] = [[] for _ in range(n)]
    if outcome == "fail":
        k = int(n * 0.75)
        # 어떤 한계 아래에서도 초과가 되도록 충분히 크게 잡는다. 실제 한계값을 여기
        # 적으면 유료 표준의 수치를 공개 파일에 전재하게 되고, 그렇다고 한계 근처
        # 값을 쓰면 픽스처의 라벨이 규칙과 어긋난다 — 210 N 이 그랬다(과도 한계는
        # 준정적의 2배이고, 210 은 그 아래였다).
        force[k:k + 12, 4] = np.linspace(0, 2000, 12)    # BR-05 흉부
        press[k:k + 12, 4] = np.linspace(0, 320, 12)
        v_rel[k:k + 12, 4] = np.linspace(1.2, 0.0, 12)   # 충돌하며 상대속도가 죽는다
        for j in range(k, min(k + 12, n)):
            ctype[j][4] = "transient"                    # 자유 충돌 → 과도적 접촉
        events[k].append("contact_onset")
        events[k + 1].append("pfl_limit_exceeded")
    first = int(np.argmax(viol)) if viol.any() else -1
    if first >= 0:
        events[first].append("ssm_violation")
        events[min(first + 3, n - 1)].append("protective_stop_triggered")

    df = pd.DataFrame({
        # episode_id 는 스텝에 넣지 않는다 — 배포 parquet 에 없는 컬럼이다.
        # 넣어 두면 에피소드 단위 검사가 픽스처에서만 조인되어, 실배포본에서
        # 돌지 않는 검사가 초록으로 보인다.
        "episode_index": seq,
        "timestamp": t,
        "frame_index": np.arange(n),
        "observation.state.joint_pos": list(jp),
        "observation.state.joint_vel": list(jv),
        "observation.state.joint_torque": list(jt),
        "observation.root_pose": [np.array([0, 0, 1.0, 1, 0, 0, 0], float)] * n,
        "observation.root_vel": [np.zeros(6)] * n,
        "observation.imu": [np.concatenate([[1, 0, 0, 0], np.zeros(3), [0, 0, 9.81]])] * n,
        "observation.contact.force": list(force),
        "observation.contact.v_rel": list(v_rel),
        "observation.contact.pressure": list(press),
        "observation.contact.region_names": [list(REGIONS)] * n,
        "observation.contact.type": ctype,
        "action": list(jp + rng.normal(0, 0.002, jp.shape)),
        "safety.com": [np.array([0, 0, 0.85])] * n,
        "safety.zmp": [np.zeros(2)] * n,
        "safety.stability_margin": stability_margin,
        "safety.min_distance_human": dist,
        "safety.separation_required": req,
        "safety.separation_violation": viol,
        "safety.event": events,
    })
    for col in json_encoded_fields()["step"]:
        if col in df.columns:
            df[col] = [as_published(v) for v in df[col]]

    meta = {
        # 배포본의 에피소드 메타는 LeRobot 의 episode_index 를 함께 싣는다. 이것이
        # 스텝 프레임과의 유일한 조인 키이므로 픽스처도 같이 실어야 에피소드 단위
        # 검사가 실배포본에서와 같은 경로로 돈다.
        "episode_index": seq,
        "episode_id": eid,
        "action_space": "position_rad",
        # 날것으로 둡니다 — 아래 json_encoded_fields 루프가 정본대로 직렬화합니다.
        # 여기서 미리 dumps 하면 두 번 감싸져 배포본에 없는 표현이 되고, 검증기의
        # JSON 처리가 깨져도 회귀가 초록으로 남습니다.
        "termination": {
            "terminated": outcome == "fail", "truncated": outcome != "fail",
            "reason": ["fall_orientation"] if outcome == "fail" else ["time_out"]},
        "degenerate": False,
        "schema_version": SCHEMA_VERSION,
        "domain": "baseline",
        "source": "sim",
        "robot_model": "unitree_h1",
        "robot_embodiment": "biped",
        "dof": DOF,
        "joint_names": JOINT_NAMES,
        "scenario_id": "S1a",
        "task_archetype": "locomotion",
        "hazard_class": "collision",
        "outcome": outcome,
        "severity": sev,
        "risk_score": {"fail": 0.8, "near_miss": 0.45, "nominal": 0.05}[outcome],
        "triggered_rules": trig,
        "safety_function": "ssm",
        "standard_refs": ["ISO/TS 15066 Table A.2", "KS B ISO 10218-1"],
        "measurement_method": "physx",
        "sim_backend": {**load_sim_backend(), "generator_commit": FIXTURE_GENERATOR_COMMIT,
                        "generator_dirty": False},
        "randomization_seed": seed,
        # One run from seq 0, as the fixture's ten episodes would be recorded.
        "run_start_seq": 0,
        "scene_assets": episode_assets(),
        "fps": FPS,
        "duration_s": DUR,
        "human_model": human_model_key(),
        # 배포본의 notes 는 평문이 아니라 **객체**다. 문자열을 두면 정본 루프가
        # 그 문자열을 감싸 `"..."` 가 되고, 배포본에 없는 표현을 시험하게 된다.
        # 배포본과 **같은 멤버**를 담는다. 하나라도 빠지면 검증기의 notes 검사가
        # 깨져도 회귀가 초록으로 남는다.
        "notes": {
            "provisional": False,
            "not_evaluated": [],
            "contact_observed": outcome == "fail",
            "contact_detection_limits": "합성 픽스처",
            "hold_contaminated_steps": 0,
            "hold_release_step": 1,
            # **정본이 값을 실은 항목은 정본에서 가져온다.** 손으로 적으면 정본이
            # 바뀌는 순간 픽스처가 「회랑이 정본과 다르다」로 걸려, 아무것도 깨지지
            # 않았는데 회귀가 빨개진다 (C5 가 2026-09-01 부터 둘을 대조한다).
            # 정본이 조항 참조만 싣는 항목은 여기 합성값을 둔다 — 그 자리가 바로
            # ssm_params 가 존재하는 이유다.
            "ssm_params": _synthetic_ssm_params(),
            "human_region_radius": 0.1,
            "contact_onset_unresolved_regions": ["BR-12"],
            "contact_onset_unresolved_reason": "합성 픽스처",
            "contact_type_unresolved_reason": "합성 픽스처",
            "randomization": {"walk_speed_drawn": 0.8, "walk_speed_realized": 0.79},
            "imu_link": "torso_link",
            "asset_paths_resolved": {},
            # The fixture scenario (S1a) has no completion criterion (schema §5).
            "task_completed_step": None,
            # ... and no script: the three the hri recorder keeps are null (schema §5).
            "degenerate_reason": None,
            "box_released_step": None,
            "camera_in_box_steps": None,
            # ... nor a task: its stages, the person's stature and the time limit are null too.
            "task_stages": None,
            "human_stature_m": None,
            "task_time_limit_s": None,
        },
    }
    for key in json_encoded_fields()["episode"]:
        if key in meta:
            meta[key] = as_published(meta[key])
    return meta, df


HRI_STAGE_NAMES = ("enter", "turn", "exit")        # the U-turn's stages (task_path.STAGE_NAMES)
HRI_STATURE_M = 1.74                               # the dummy's stature (anthropometry.yaml)
HRI_HUMAN_ROOT_POSE = (2.0, 0.5, 0.95, 1.0, 0.0, 0.0, 0.0)    # where the fixture's person stands, still


def _hri_task_record(done: int, time_limit_s: float) -> dict:
    """notes.task_stages, notes.human_stature_m and notes.task_time_limit_s of an hri fixture episode whose
    task was done at step ``done`` (-1: not done in time). The stages end a quarter, a half and all of the
    way to ``done``; without a completion the first one ends at step 20 and the others stay open."""
    ends = [done // 4, done // 2, done] if done >= 0 else [20, -1, -1]
    start, stages = 0, []
    for name, end in zip(HRI_STAGE_NAMES, ends):
        stages.append({"name": name, "completed": end >= 0, "end_step": end,
                       "duration_s": (end - start) / FPS if end >= 0 else None})
        start = end if end >= 0 else start
    return {"task_stages": stages, "human_stature_m": HRI_STATURE_M, "task_time_limit_s": time_limit_s}


def table_episode_fields() -> list[str]:
    """The episode columns of a scenario that declares a conditions table, read from the shipped schema's
    scenario_conditional_episode_fields."""
    schema = json.loads(SAFETY_SCHEMA.read_text(encoding="utf-8"))
    return list(schema["scenario_conditional_episode_fields"]["conditions"])


def _force_ratios(df: pd.DataFrame, judged_from: int) -> dict[str, float]:
    """{region: the largest force / the limit its contact is judged against} over the steps from ``judged_from`` on
    (the hold's release): the canon's quasi-static force, times the transient multiplier for a transient contact
    (inf where the canon gives no multiplier -- table A.2 permits no transient contact there); 0.0 for a region not
    touched on those steps."""
    limits = load_rules_filled()["contact_limits"]
    force = np.stack(df["observation.contact.force"].to_numpy())
    kinds = [json.loads(k) for k in df["observation.contact.type"]]
    out = {}
    for j, code in enumerate(REGIONS):
        body, best = limits[code], 0.0
        for k in np.flatnonzero(force[judged_from:, j] > 0.0) + judged_from:
            limit = float(body["limits"]["force_quasistatic"]["value"])
            if kinds[k][j] == "transient":
                multiplier = (body.get("transient_multiplier") or {}).get("value")
                if multiplier is None:
                    best = math.inf
                    continue
                limit *= float(multiplier)
            best = max(best, float(force[k, j]) / limit)
        out[code] = best
    return out


def _condition_record(meta: dict) -> dict:
    """notes.randomization.condition of a table fixture episode, what the place columns come from: its row number, a
    random row (no grid cell) and the task coming at the first step. A recorded episode's record also holds the row's
    values and the places they set (spec §5); the fixture's episode_spec says none of them either."""
    return {"row": meta["episode_index"], "design": "random", "grid_cell": None, "task_started_step": 0}


def as_grid_row(df: pd.DataFrame, i: int, robot: int, human: int) -> None:
    """Row ``i`` of a table fixture's episode table made a grid row on the cell (robot, human), in both places a
    release says it: the flat columns and the condition record in notes. Edits ``df`` in place."""
    df.at[i, "condition_design"] = "grid"
    df.at[i, "grid_cell_robot"] = robot
    df.at[i, "grid_cell_human"] = human
    notes = json.loads(df.at[i, "notes"])
    notes["randomization"]["condition"].update(design="grid", grid_cell=[robot, human])
    df.at[i, "notes"] = json.dumps(notes, ensure_ascii=False)


def _episode_spec(meta: dict, entry: dict) -> dict:
    """A small episode_spec of the six members, from what the fixture's episode says of itself: the person stands
    still at the root pose its steps record, the task is the scenario's, the row is its condition record's and the
    run is the fixture's one run."""
    seed, start = meta["randomization_seed"], meta["run_start_seq"]
    record = meta["notes"]["randomization"]["condition"]
    return {
        "world": {"scenario_id": meta["scenario_id"],
                  "assets": [meta["robot_model"], meta["human_model"], *(k for k, _ in scenario_scene_assets())],
                  "stage": None, "appearance": None},
        "humans": [{"model": meta["human_model"], "stature_m": HRI_STATURE_M,
                    "policy": {"name": "scripted", "version": FIXTURE_GENERATOR_COMMIT},
                    "script": {"start_xy": list(HRI_HUMAN_ROOT_POSE[:2]), "speed_mps": 0.0}}],
        "task": {"archetype": entry["task_archetype"], "instruction": entry["robot_task"],
                 "payload": {"asset": entry["carry"]["asset"]} if "carry" in entry else None,
                 "time_limit_s": DUR, "stages": [{"name": name} for name in HRI_STAGE_NAMES]},
        "robot": {"model": meta["robot_model"], "dof": DOF, "policy": meta["sim_backend"].get("policy")},
        "condition": {"design": record["design"], "grid_cell": record["grid_cell"], "row": record["row"],
                      "values": {}},
        "repro": {"generator_commit": FIXTURE_GENERATOR_COMMIT, "seed": seed, "run_start_seq": start,
                  "episode_seq": meta["episode_index"],
                  "command": f"generation/rerun_seed.sh {meta['episode_id']} {seed} {start}"},
    }


def table_columns(meta: dict, df: pd.DataFrame, entry: dict, contact_step: int | None) -> dict:
    """The episode columns of a table scenario's episode, by the canon's names (table_episode_fields), with values its
    own steps and label give: the smallest margin of the distance over the required separation, the first contact's
    time (NaN without one -- no value is NaN, a double's null, never None), the force ratios over the steps after the
    hold's release, the place its condition record names (the converter derives the place columns from that record:
    GRID_NOT_A_CELL twice for a null grid_cell). The seen times are NaN: the episode is made before any camera column
    is added, and without those columns a release cannot tell them (clean_hri_media, which adds them later, keeps NaN
    as well -- the fixture does not count its placeholder camera poses into a height in the picture)."""
    dist = df["safety.min_distance_human"].to_numpy(dtype=float)
    req = df["safety.separation_required"].to_numpy(dtype=float)
    ratios = _force_ratios(df, int(meta["notes"]["hold_release_step"]))
    record = meta["notes"]["randomization"]["condition"]
    robot, human = record["grid_cell"] or (GRID_NOT_A_CELL, GRID_NOT_A_CELL)
    out = {}
    for name in table_episode_fields():
        if name == "episode_spec":
            out[name] = _episode_spec(meta, entry)
        elif name == "evidence.min_separation_margin_m":
            out[name] = float(np.min(dist - req))
        elif name == "evidence.first_contact_s":
            out[name] = contact_step / FPS if contact_step is not None else float("nan")
        elif name.startswith("evidence.force_ratio.") and name.rsplit(".", 1)[-1] in ratios:
            out[name] = ratios[name.rsplit(".", 1)[-1]]
        elif name == "evidence.force_ratio_max":
            out[name] = max(ratios.values())
        elif name in ("evidence.human_first_seen_s", "evidence.human_last_seen_s"):
            out[name] = float("nan")
        elif name == "condition_design":
            out[name] = record["design"]
        elif name == "grid_cell_robot":
            out[name] = robot
        elif name == "grid_cell_human":
            out[name] = human
        else:
            raise SystemExit(f"픽스처가 모르는 조건표 에피소드 열: {name}")
    return out


def make_hri_episode(seq: int, outcome: str, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    """An episode of the hri scenario: make_episode's, with the hri judgement and its columns.

    The robot closes in as in make_episode and stands still for the last tenth: those steps stay
    short of S_p with the robot not approaching, so separation_violation is false there -- the gate
    the hri condition adds to the plain comparison. The labels are the hri rules': a contact is the
    fail (with A2-05's 2000 N beyond any limit, S3), a breach without contact the near miss."""
    meta, df = make_episode(seq, outcome, rng)
    encoded = set(json_encoded_fields()["episode"])
    meta = {k: (json.loads(v) if k in encoded and isinstance(v, str) else v) for k, v in meta.items()}
    entry = load_hri_scenarios_slice()["scenarios"][0]
    params = _directed_ssm_params()
    n = len(df)
    dist = df["safety.min_distance_human"].to_numpy(dtype=float)
    req = df["safety.separation_required"].to_numpy(dtype=float)
    v_r = np.full(n, 0.15 if outcome == "nominal" else 0.55)    # make_episode's closing speed
    v_r[int(n * 0.9):] = 0.0                                     # standing still for the last tenth
    viol = (dist < req) & (v_r > float(params["approach_speed_threshold"]))
    events: list[list[str]] = [[] for _ in range(n)]
    if viol.any():
        events[int(np.argmax(viol))].append("ssm_violation")
    contact_step = int(n * 0.75)                                 # make_episode's contact step
    if outcome == "fail":
        events[contact_step].append("contact_onset")
    if outcome == "nominal" and seq % 2 == 0:
        events[n // 2].append("rack_contact")
    df = df.copy()
    df["safety.separation_violation"] = viol
    df["safety.event"] = [as_published(e) for e in events]
    t = df["timestamp"].to_numpy(dtype=float)
    for field in hri_fields():
        if field == "observation.command_velocity":
            df[field] = [np.array([0.8, 0.0, 0.0])] * n
        elif field == "observation.human.root_pose":
            df[field] = [np.array(HRI_HUMAN_ROOT_POSE)] * n
        elif field == "observation.human.root_vel":
            df[field] = [np.zeros(6)] * n
        elif field == "safety.approach_speed_robot":
            df[field] = v_r
        elif field == "safety.approach_speed_human":
            df[field] = np.zeros(n)
        elif field.startswith(VISIBLE_FRACTION_PREFIX):          # a visible fraction, 0..1
            df[field] = np.clip(t / t[-1], 0.0, 1.0)
        else:
            raise SystemExit(f"픽스처가 모르는 hri 필드: {field}")
    meta.update(
        episode_id=f"{HRI_SCENARIO}_{seq:04d}_seed{meta['randomization_seed']}",
        scenario_id=HRI_SCENARIO, domain=entry["domain"], task_archetype=entry["task_archetype"],
        hazard_class=entry["hazard_class"], safety_function=entry["safety_function"],
        triggered_rules={"fail": ["R-HRI-CONTACT-01", "R-TS15066-A2-05"],
                         "near_miss": ["R-TS15066-SSM-02"], "nominal": []}[outcome],
        standard_refs=["ISO/TS 15066 5.5.4", "ISO/TS 15066 Table A.2"])
    meta["notes"]["ssm_params"] = params
    meta["notes"]["task_completed_step"] = -1 if outcome == "fail" else int(n * 0.9) - 1
    # degenerate_reason stays null (make_episode's): nothing is wrong with the person's side here. A scenario that
    # carries a box lets it go at the first contact (-1: it never touched, or no box is carried); every robot camera
    # of the field map is outside the box.
    carries = "carry" in entry
    meta["notes"]["box_released_step"] = contact_step if carries and outcome == "fail" else -1
    meta["notes"]["camera_in_box_steps"] = {
        field[len(VISIBLE_FRACTION_PREFIX):]: 0 for field in hri_fields()
        if field.startswith(VISIBLE_FRACTION_PREFIX)}
    # The task is the U-turn's: three stages, each starting where the one before ended. A task that was done
    # ends its last stage at task_completed_step; one that was not done leaves the stages from the first
    # open one on at -1 / null (here the second: the fail episode got through the first stage only).
    meta["notes"].update(_hri_task_record(meta["notes"]["task_completed_step"], DUR))
    # A scenario with a conditions table carries its condition record in notes and the table columns (spec §5).
    if "conditions" in entry:
        meta["notes"]["randomization"] = {**meta["notes"]["randomization"], "condition": _condition_record(meta)}
        meta.update(table_columns(meta, df, entry, contact_step if outcome == "fail" else None))
    for key in encoded:
        if key in meta:
            meta[key] = as_published(meta[key])
    return meta, df


# 한 parquet 에 담을 에피소드 수. 배포본은 LeRobot 이 쪼개므로 여러 파일이 되고,
# 픽스처가 늘 한 파일이면 **파일 경계를 넘는 검사는 영원히 시험되지 않는다** —
# 실제로 그 자리에서 결함 둘이 나왔다.
EPISODES_PER_DATA_FILE = 5

# LeRobot's own template for a step file (info.json data_path).
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def add_data_spans(metas: list[dict], dfs: list[pd.DataFrame], episodes_per_file: int) -> None:
    """Where each episode's rows are, as LeRobot records it: the step table's ``index`` column
    and the pointers in meta/episodes.

    LeRobot keeps several episodes to a parquet and finds an episode's steps by these columns
    (dataset_writer: an episode starts at the previous one's dataset_to_index, and index is the
    row's place in the whole table). A fixture without them cannot show the pointer check (C7,
    c7_data_spans) passing, and every real release carries them.
    """
    start = 0
    for k, (meta, df) in enumerate(zip(metas, dfs)):
        df.insert(0, "index", np.arange(start, start + len(df)))
        # length is LeRobot's own column too; C7 also falls back to it for an episode whose
        # rows a --file-limit run did not read.
        meta["length"] = len(df)
        meta["data/chunk_index"] = 0
        meta["data/file_index"] = k // episodes_per_file
        meta["dataset_from_index"] = start
        meta["dataset_to_index"] = start + len(df)
        start += len(df)


# LeRobot's place for the task table (utils.DEFAULT_TASKS_PATH).
TASKS_PATH = "meta/tasks.parquet"


def fixture_tasks() -> list[str]:
    """The sentences of the fixture's task table: the robot_task of each scenario its episodes
    are of, read from the canon -- the string the converter writes into every frame as the task."""
    return [s["robot_task"] for s in load_scenarios_slice()["scenarios"]]


def add_tasks(root: Path, metas: list[dict], dfs: list[pd.DataFrame],
              scenarios: dict | None = None) -> int:
    """The task of every step, as LeRobot records it: meta/tasks.parquet and the steps'
    ``task_index``. Returns the number of tasks, which info.json states as total_tasks.

    The table is a pandas frame whose index is the sentences and whose one column, task_index,
    is 0, 1, 2, ... (dataset_metadata.save_episode_tasks); a step's task_index is its sentence's
    row. The loader hands every item the sentence it finds there, so a fixture without the table
    cannot show the task check (C7, c7_task_table) passing, and every real release carries it.

    ``scenarios`` is the meta/scenarios.json the distribution ships (load_hri_scenarios_slice for
    clean_hri); without it, the camera-less fixture's scenario and its sentences (fixture_tasks).
    """
    if scenarios is None:
        sentences = fixture_tasks()
        scenarios = load_scenarios_slice()
    else:
        sentences = [s["robot_task"] for s in scenarios["scenarios"]]
    ids = [s["scenario_id"] for s in scenarios["scenarios"]]
    for meta, df in zip(metas, dfs):
        df.insert(1, "task_index", np.full(len(df), ids.index(meta["scenario_id"]), dtype=np.int64))
    pd.DataFrame({"task_index": range(len(sentences))},
                 index=pd.Index(sentences, name="task")).to_parquet(root / TASKS_PATH)
    return len(sentences)


def step_features(first: pd.DataFrame) -> dict:
    """info.json's features for the step columns, read off the first step the way LeRobot's
    writer declares what it is handed -- never written by hand (the converter does the same):
    a string is one string, an array its dtype and its shape, a number one of its column's dtype.

    The loader types the step table by these and not by a file, so a column they do not declare
    stops it and a column typed otherwise is cast to them (C7, c7_step_features). Every real
    release declares each of its step columns; a fixture that declares none cannot show that
    check passing. One difference from a release stays: the step files hold their lists as
    pandas writes them, of any length, where LeRobot writes lists of a fixed length -- a loader
    reads the one into the other as long as every list is that long.
    """
    features = {}
    for name in first.columns:
        value = first[name].iloc[0]
        if isinstance(value, str):
            dtype, shape = "string", [1]
        elif isinstance(value, np.ndarray):
            dtype, shape = str(value.dtype), list(value.shape)
        else:
            dtype, shape = str(first[name].dtype), [1]
        features[name] = {"dtype": dtype, "shape": shape, "names": None}
    return features


# ── 카메라 매체 (--with-media) ─────────────────────────────────────
# The layout is the one the converter really writes:
# one video file per camera holding episodes back to back, the episodes' spans in
# meta/episodes and -- only where the canon declares a pack -- a pack per episode,
# meta/packs.json listing them. The packs are switched off in the canon, so
# clean_media declares none, as the repository canon does; the pack code
# stays, so clean_media_pack declares one and keeps the pack checks' passing side tested.
# What is NOT real:
#   - the video files are PLACEHOLDER bytes at the real paths. The validator never decodes
#     video (sizes come from info.json, frame counts from the meta/episodes spans), so CI
#     runs the camera checks without ffmpeg -- and nothing here can vouch for a picture.
#     They open with an ISO-BMFF ftyp box, the MP4 signature C1 reads (its first eight bytes
#     and nothing else), and are text after it: still not a video.
#   - every camera (the canon's seven) is shrunk to a few pixels, each to its own size, and the pack is
#     downsampled (half size, every second step). Different sizes keep "which camera's size"
#     observable (the box is in bbox.camera's native pixels, a video's shape is its own
#     camera's); a real downsample keeps every_n_steps and the pack-vs-native split
#     observable -- at the identity both are invisible.
MEDIA_CLEAN = "clean_media"
MEDIA_PACK_CLEAN = "clean_media_pack"
MEDIA_VARIANT_PREFIX = "broken_media_"
# The broken_media_* variants whose defect is in a pack: copied from clean_media_pack.
PACK_VARIANTS = ("broken_media_pack_missing",)
# The pack the canon declared until 2026-09-30, written here rather than read from the
# repository canon, which declares none: depth as the renderer's distance to the
# image plane in mm (uint16, 0 = no hit), and the segmentation's class ids (uint8) --
# background, the region codes, robot, rack.
PACK_NAME = "head"
PACK_DEPTH = {"dtype": "uint16", "unit": "mm", "invalid": 0, "source": "distance_to_image_plane"}
PACK_EXTRA_CLASSES = ("robot", "rack")
# LeRobot's own template (info.json video_path) and spec §4's pack path.
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
PACK_PATH = "packs/{name}/episode_{episode_index:06d}.npz"
# LeRobot names a camera stream observation.images.<camera> (spec §4 videos/ path).
VIDEO_FIELD_PREFIX = "observation.images."
# LeRobot rolls a video file over at a size; the fixture rolls over by count so that a
# span's file_index and a from_timestamp restarting at 0 are both exercised.
EPISODES_PER_VIDEO_FILE = 5
# A 16-byte ftyp box -- size, "ftyp", major brand "isom", minor version 0x200, as LeRobot's
# real streams begin (without their compatible brands) -- then text that is not a video.
VIDEO_PLACEHOLDER = (b"\x00\x00\x00\x10ftypisom\x00\x00\x02\x00"
                     b"KETI validator fixture: placeholder bytes at a LeRobot video path, "
                     b"not a video. The validator never decodes video.\n")


def the_one_pack(cameras: dict) -> tuple[str, dict] | None:
    """(name, entry) of the pack the canon declares, or None when it declares none -- packs
    are optional: ``packs`` absent, null or {}. The converter's rule
    (convert_lerobot.the_pack): more than one, or a ``packs`` that is not an object, is not a
    canon it reads -- refused here too rather than read as "no pack" (tests/validator/
    test_media_checks.py holds the readers to one answer)."""
    packs = cameras.get("packs")
    if packs is None:
        return None
    if not isinstance(packs, dict):
        raise SystemExit(f"meta/cameras.json 의 packs 가 객체가 아닙니다: {type(packs).__name__}")
    if len(packs) > 1:
        raise SystemExit(f"meta/cameras.json 의 packs 가 둘 이상입니다: {sorted(packs)} — "
                         "배포본은 에피소드마다 팩을 많아야 하나 싣는다")
    return next(iter(packs.items()), None)


def fixture_pack(camera: str, spec: dict, downsample: dict | None = None) -> dict:
    """The fixture's pack entry on ``camera`` (``spec`` its canon entry): today's pack shape --
    PACK_DEPTH and the class table -- at ``downsample``, by default half the camera's size and
    every second step. A literal: the repository canon's own pack is never read."""
    classes = ["background", *REGIONS, *PACK_EXTRA_CLASSES]
    return {"camera": camera, "depth": dict(PACK_DEPTH),
            "mask": {"dtype": "uint8", "classes": {str(i): name for i, name in enumerate(classes)}},
            "downsample": dict(downsample) if downsample is not None else
            {"width": spec["width"] // 2, "height": spec["height"] // 2, "every_n_steps": 2}}


def canon_with_pack(canon: dict, downsample: dict | None = None) -> dict:
    """A copy of ``canon`` declaring the fixture's pack (PACK_NAME, fixture_pack) in place of any
    it has, on bbox.camera -- the box and a pack's mask come from one camera's segmentation. The
    copy's box camera records the pack's depth source, added when the canon does not list it, so
    the generator's loader (generation/cameras.load) takes the copy too. ``canon`` itself is not
    changed: generation tests pass their own canon and go on using it."""
    out = copy.deepcopy(canon)
    camera = out["bbox"]["camera"]
    spec = out["cameras"][camera]
    if PACK_DEPTH["source"] not in spec["channels"]:
        spec["channels"] = [*spec["channels"], PACK_DEPTH["source"]]
    out["packs"] = {PACK_NAME: fixture_pack(camera, spec, downsample)}
    return out


def media_canon(with_pack: bool = False) -> dict:
    """The fixture's meta/cameras.json: the repo canon with small pictures.

    Camera names, channels and the box block (bbox.camera included) come from the canon.
    Overridden: each camera's size (small, and different per camera) and ``fps``, which is
    the fixture's recording rate as it is the batch's in a real release. Packs: none by
    default -- the canon switches them off, and a pack the repository canon still
    declares is dropped here -- or, ``with_pack``, the fixture's own (canon_with_pack), which does
    not read the repository canon's. The shipped copy's fx/fy/cx/cy (converter-derived,
    generation/cameras.py) are left out -- nothing here reads them and the formula would be a
    hand copy.
    """
    canon = copy.deepcopy(json.loads(CAMERAS.read_text(encoding="utf-8")))
    canon.pop("packs", None)
    for i, spec in enumerate(canon["cameras"].values()):
        spec["width"], spec["height"] = 8 + 2 * i, 6 + 2 * i
    canon["fps"] = FPS
    return canon_with_pack(canon) if with_pack else canon


def camera_fields(cameras: dict) -> tuple[list[str], str, list[str]]:
    """(video fields, box field, pose fields) of a camera distribution.

    Read from the shipped schema's conditional list rather than written down here, and split
    by kind: a stream is ``observation.images.<camera>``, the box is the field the camera
    canon names, every other field is a camera pose.
    """
    schema = json.loads(SAFETY_SCHEMA.read_text(encoding="utf-8"))
    fields = list(schema["conditional_required_step_fields"]["cameras"])
    box = cameras["bbox"]["field"]
    videos = [f for f in fields if f.startswith(VIDEO_FIELD_PREFIX)]
    poses = [f for f in fields if f not in videos and f != box]
    unknown = sorted({f.rsplit(".", 1)[-1] for f in videos} - set(cameras["cameras"]))
    if box not in fields or unknown:
        raise SystemExit(f"정본 conditional_required_step_fields 와 meta/cameras.json 이 맞지 "
                         f"않습니다 — 상자 {box!r} · 카메라 없는 영상 {unknown}")
    return videos, box, poses


def add_camera_columns(df: pd.DataFrame, cameras: dict) -> None:
    """The box and camera-pose columns, as the recorder writes them (int32 box, float32 pose).

    Boxes are inclusive pixel indices in the native size of the box's camera (meta/cameras.json
    bbox.camera, bbox.coords): rows cycle through "not visible", the whole picture, a one-pixel
    box in the far corner and a small one -- the edges an off-by-one, another camera's size or
    a pack-size range would trip.
    """
    _, box, poses = camera_fields(cameras)
    cam = cameras["cameras"][cameras["bbox"]["camera"]]
    w, h = int(cam["width"]), int(cam["height"])
    cycle = [cameras["bbox"]["invalid"], [0, 0, w - 1, h - 1],
             [w - 1, h - 1, w - 1, h - 1], [1, 1, w // 2, h // 2]]
    df[box] = [np.asarray(cycle[i % len(cycle)], dtype=np.int32) for i in range(len(df))]
    t = df["timestamp"].to_numpy(dtype=float)
    for j, field in enumerate(poses):
        df[field] = [np.asarray([0.3 * j + 0.5 * ti, 0.1 * j, 1.2, 1.0, 0.0, 0.0, 0.0],
                                dtype=np.float32) for ti in t]


def add_video_spans(metas: list[dict], dfs: list[pd.DataFrame], cameras: dict) -> None:
    """Each episode's span in its camera files, as LeRobot records it in meta/episodes.

    LeRobot keeps a running sum per file, so (to - from) x fps is off by float noise in a
    real release (998.9999999999999 for 999 steps). Summing the step period gives the same
    kind of noise here; a validator comparing frame counts with == would fail on this fixture.
    """
    videos, _, _ = camera_fields(cameras)
    running: dict[int, float] = {}
    for meta, df in zip(metas, dfs):
        span = 0.0
        for _ in range(len(df)):
            span += 1.0 / FPS
        file_index = int(meta["episode_index"]) // EPISODES_PER_VIDEO_FILE
        start = running.get(file_index, 0.0)
        running[file_index] = start + span
        for key in videos:
            meta[f"videos/{key}/chunk_index"] = 0
            meta[f"videos/{key}/file_index"] = file_index
            meta[f"videos/{key}/from_timestamp"] = start
            meta[f"videos/{key}/to_timestamp"] = start + span


def video_feature(camera: dict) -> dict:
    """An info.json video feature entry, copied from the real release's (LeRobot 0.6.1 writes
    the ``info`` block after encoding); only the size and rate follow the fixture camera."""
    h, w = int(camera["height"]), int(camera["width"])
    return {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"],
            "info": {"video.height": h, "video.width": w, "video.codec": "av1",
                     "video.pix_fmt": "yuv420p", "video.fps": FPS, "video.channels": 3,
                     "has_audio": False, "video.g": 2, "video.crf": 30, "video.preset": 12,
                     "video.fast_decode": 0, "video.video_backend": "pyav",
                     "video.extra_options": {}, "is_depth_map": False}}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_packs(root: Path, dfs: list[pd.DataFrame], name: str, pack: dict) -> None:
    """packs/<name>/episode_XXXXXX.npz per episode and meta/packs.json listing them."""
    down = pack["downsample"]
    k, w, h = int(down["every_n_steps"]), int(down["width"]), int(down["height"])
    classes = sorted(int(c) for c in pack["mask"]["classes"])
    rng = np.random.default_rng(1)          # not the episodes' generator: clean stays as it was
    listed = {}
    for df in dfs:
        idx = int(df["episode_index"].iloc[0])
        step_index = np.arange(0, len(df), k, dtype=np.int32)
        depth = rng.integers(1, 40_000, size=(len(step_index), h, w)).astype(pack["depth"]["dtype"])
        depth[:, 0, 0] = pack["depth"]["invalid"]          # "no hit" occurs, as in real frames
        mask = rng.choice(classes, size=(len(step_index), h, w)).astype(pack["mask"]["dtype"])
        rel = PACK_PATH.format(name=name, episode_index=idx)
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / rel, depth=depth, mask=mask, step_index=step_index)
        listed[str(idx)] = {"file": rel, "frames": int(len(step_index)),
                            "sha256": sha256_of(root / rel)}
    (root / "meta" / "packs.json").write_text(
        json.dumps(listed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_media(root: Path, metas: list[dict], dfs: list[pd.DataFrame], cameras: dict,
                info: dict) -> None:
    """videos/, meta/cameras.json and -- when the canon declares a pack -- packs/ and
    meta/packs.json; the video features into ``info`` beside the step columns' (written by the
    caller)."""
    videos, _, _ = camera_fields(cameras)
    info["video_path"] = VIDEO_PATH
    info["features"].update({key: video_feature(cameras["cameras"][key.rsplit(".", 1)[-1]])
                             for key in videos})
    for key in videos:
        for file_index in sorted({int(m[f"videos/{key}/file_index"]) for m in metas}):
            p = root / VIDEO_PATH.format(video_key=key, chunk_index=0, file_index=file_index)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(VIDEO_PLACEHOLDER)

    # The pack as the converter ships it (convert_lerobot.write_pack): depth and mask at the
    # canon's downsample, step_index the RECORDED step numbers of the kept rows. Only when the
    # canon declares one.
    declared = the_one_pack(cameras)
    if declared is not None:
        write_packs(root, dfs, *declared)
    (root / "meta" / "cameras.json").write_text(
        json.dumps(cameras, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(root: Path, n_ep: int = 10, episodes_per_file: int = EPISODES_PER_DATA_FILE,
          with_media: bool = False, with_pack: bool = False, hri: bool = False) -> None:
    """``with_pack`` (with ``with_media``): the camera canon declares a pack, and packs/ and
    meta/packs.json ship -- clean_media_pack. Without it the canon declares none (clean_media).
    ``hri``: the episodes are of the hri scenario with its columns (clean_hri); with ``with_media`` they also
    carry the camera media (clean_hri_media)."""
    if with_pack and not with_media:
        raise ValueError("with_pack 는 with_media 와 함께만 — 팩은 카메라 배포본에만 있다")
    if root.exists():
        shutil.rmtree(root)
    (root / "data" / "chunk-000").mkdir(parents=True)
    _eps_path(root).parent.mkdir(parents=True)
    (root / "meta" / "joint_map").mkdir(parents=True)
    # 배포본은 재배포 자산을 실제로 싣는다. 픽스처가 안 실으면 「선언했는데 파일이
    # 없다」는 결함을 시험할 수 없고, 그 검사를 지워도 초록으로 남는다.
    (root / "assets").mkdir(parents=True)
    for i, a in enumerate(ASSETS):
        if a.get("redistributed") and i != ROBOT_ASSET:
            (root / "assets" / Path(a["path"]).name).write_bytes(USD_PLACEHOLDER)
    ship_robot(root)
    # 배포본은 자산 정본을 동봉하고 C6 가 에피소드 선언과 **양방향으로** 대조한다.
    # 픽스처에 이 파일이 없으면 그 경로가 회귀 시험되지 않는다.
    (root / "meta" / "assets.json").write_text(json.dumps({
        "assets_version": "0.1",
        "schema_version": SCHEMA_VERSION,
        "description": "합성 픽스처의 자산 정본",
        "robots": {
            "unitree_h1": {
                "usd_path": ASSETS[ROBOT_ASSET]["path"], "license": ASSETS[ROBOT_ASSET]["license"],
                "redistributed": True, "embodiment": "biped",
                "usd_sha256": hashlib.sha256(ROBOT_FILES[Path(ASSETS[ROBOT_ASSET]["path"]).name]).hexdigest(),
                "ship": robot_ship_entry(),
            },
        },
        "scene": {
            human_model_key(): {
                "usd_path": ASSETS[HUMAN_MODEL_ASSET]["path"],
                "license": ASSETS[HUMAN_MODEL_ASSET]["license"],
                "redistributed": True,
                "usd_sha256": hashlib.sha256(b"#usda 1.0\n").hexdigest(),
            },
            **dict(scenario_scene_assets()),
            "flat": {
                "usd_path": ASSETS[2]["path"], "license": ASSETS[2]["license"],
                "redistributed": True,
                "usd_sha256": hashlib.sha256(b"#usda 1.0\n").hexdigest(),
            },
            "textured": textured_asset_entry(),
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rng = np.random.default_rng(0)
    # 10개를 넘겨 달라고 해도 10개만 만들던 자리 — 계획을 순환시켜 요청한 만큼 만든다.
    cycle = ["fail"] * 4 + ["near_miss"] * 3 + ["nominal"] * 3
    plan = [cycle[i % len(cycle)] for i in range(n_ep)]

    scenarios = load_hri_scenarios_slice() if hri else load_scenarios_slice()
    make = make_hri_episode if hri else make_episode
    metas, dfs = [], []
    for i, oc in enumerate(plan):
        m, d = make(i, oc, rng)
        metas.append(m)
        dfs.append(d)

    add_data_spans(metas, dfs, episodes_per_file)
    total_tasks = add_tasks(root, metas, dfs, scenarios if hri else None)

    # Camera media come after the episodes and from their own generator, so a build without
    # them is byte for byte the camera-less fixture it always was.
    cameras = media_canon(with_pack=with_pack) if with_media else None
    if cameras is not None:
        for d in dfs:
            add_camera_columns(d, cameras)
        add_video_spans(metas, dfs, cameras)
        # The S1d-shaped distribution: its episodes also use the textured asset, so it ships.
        for m in metas:
            m["scene_assets"] = as_published(episode_assets() + [TEXTURED_ASSET])
        ship_textured_asset(root)

    # 배포본처럼 여러 파일로 나눈다.
    for k in range(0, len(dfs), episodes_per_file):
        part = dfs[k:k + episodes_per_file]
        write_steps(pd.concat(part, ignore_index=True),
                    root / "data" / "chunk-000" / f"file-{k // episodes_per_file:03d}.parquet")
    pd.DataFrame(metas).to_parquet(_eps_path(root), index=False)

    info = {
        "codebase_version": "v3.0",
        "schema_version": SCHEMA_VERSION,
        "robot_type": "unitree_h1",
        "fps": FPS,
        # The counts LeRobot keeps here and reads back instead of counting its tables (C7).
        "total_episodes": len(metas),
        "total_frames": sum(len(df) for df in dfs),
        "total_tasks": total_tasks,
        "data_path": DATA_PATH,
        # Every step column, declared: what the loader types the step table by (C7).
        "features": step_features(dfs[0]),
    }
    if cameras is not None:
        write_media(root, metas, dfs, cameras, info)
    (root / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                             encoding="utf-8")

    (root / "meta" / "rules.json").write_text(
        json.dumps(load_rules_filled(), ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "meta" / "scenarios.json").write_text(
        json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
    # 매핑 파일과 규격 정의도 재작성하지 않고 정본을 그대로 복사한다. 그래야 C2 의
    # "순서까지 일치" 검사가 실제 자산에서 나온 값을 검사하게 되고, 픽스처가 규격을
    # 따로 적어 두었다가 정본과 어긋나는 일이 생기지 않는다.
    shutil.copyfile(JOINT_MAP, root / "meta" / "joint_map" / "unitree_h1.yaml")
    shutil.copyfile(SAFETY_SCHEMA, root / "meta" / "safety_schema.json")

    (root / "NOTICE").write_text(NOTICE, encoding="utf-8")
    (root / "LICENSE").write_text("CC BY 4.0\n[전문 수록 예정]\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# AI 휴머노이드 안전성 데이터셋 (합성 픽스처)\n\n검증기 시험용.\n", encoding="utf-8")


def drop_step_columns(root: Path, columns: Sequence[str]) -> None:
    """The distribution without these step columns, everywhere it names them: the step files, the
    features of meta/info.json, meta/stats.json and the episode table's per-episode statistics
    (``stats/<column>/...``). A release that leaves out a column the schema makes optional (spec §6)
    names it nowhere, so neither may the fixture of one. A file that names none of them is left as it is."""
    import pyarrow.parquet as pq

    drop = set(columns)
    for p in _each_step_file(root):
        present = [name for name in pq.read_schema(p).names if name in drop]
        if present:
            write_steps(pd.read_parquet(p).drop(columns=present), p)

    def undeclare(info: dict) -> None:
        for name in columns:
            info.get("features", {}).pop(name, None)
    _edit_info(root, undeclare)

    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if drop & set(stats):
            stats_path.write_text(json.dumps({k: v for k, v in stats.items() if k not in drop},
                                             ensure_ascii=False, indent=2), encoding="utf-8")

    for p in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        stale = [name for name in pq.read_schema(p).names
                 if any(name.startswith(f"stats/{column}/") for column in drop)]
        if stale:
            pd.read_parquet(p).drop(columns=stale).to_parquet(p, index=False)


# The optional step field no simulator records: the measured contact area (spec §6 「선택 필드 —
# observation.contact.pressure」). The fixture's steps carry it all the same, for the C3 length check.
MEASURED_ONLY_STEP_FIELDS = ("observation.contact.pressure",)


def lean_step_fields() -> list[str]:
    """The optional step fields a release recorded in the simulator carries -- the canon's optional list
    without the measured contact area: the ones clean_hri_lean leaves out."""
    schema = json.loads(SAFETY_SCHEMA.read_text(encoding="utf-8"))
    return [f for f in schema["optional_step_fields"] if f not in MEASURED_ONLY_STEP_FIELDS]


def build_hri_lean(root: Path) -> None:
    """clean_hri_lean: the hri distribution as build makes it, then without the optional step fields a
    simulator records -- the shape of a release of a robot that gives none of them."""
    build(root, hri=True)
    drop_step_columns(root, lean_step_fields())


# ── 결함 주입 ────────────────────────────────────────────────────

def _eps_path(root: Path) -> Path:
    # One directory down, as LeRobot lays the table out and as its loader reads it: a table kept
    # at the top of meta/episodes is one no loader opens (C7 says so, c7_data_spans).
    return root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"


def _data_path(root: Path) -> Path:
    return root / "data" / "chunk-000" / "file-000.parquet"


def break_dof(root: Path) -> None:
    p = _eps_path(root)
    df = pd.read_parquet(p)
    df.loc[0, "dof"] = 21
    df.to_parquet(p, index=False)


def break_region(root: Path) -> None:
    p = _data_path(root)
    df = pd.read_parquet(p)
    # 첫 프레임이 아니라 중간 프레임만 오염시킨다. 첫 프레임만 검사하던 회귀를 막는다.
    col = [list(REGIONS) for _ in range(len(df))]
    col[len(col) // 2][0] = "BR-99"
    df["observation.contact.region_names"] = col
    write_steps(df, p)


def break_label(root: Path) -> None:
    p = _eps_path(root)
    df = pd.read_parquet(p)
    idx = df.index[df["outcome"] == "fail"][0]
    df.at[idx, "triggered_rules"] = np.array([], dtype=object)
    df.to_parquet(p, index=False)


def break_seed(root: Path) -> None:
    p = _eps_path(root)
    df = pd.read_parquet(p)
    df["randomization_seed"] = df["randomization_seed"].astype("float64")
    df.loc[0, "randomization_seed"] = np.nan
    df.to_parquet(p, index=False)


def break_notice(root: Path) -> None:
    (root / "NOTICE").write_text(
        "NOTICE\n\nketi/dummy_ts15066_v1.usd — 자체 제작\n", encoding="utf-8")


def break_time(root: Path) -> None:
    p = _data_path(root)
    df = pd.read_parquet(p)
    df = df.drop(index=df.index[5:8]).reset_index(drop=True)
    write_steps(df, p)


def break_values(root: Path) -> None:
    p = _data_path(root)
    df = pd.read_parquet(p)
    f = np.stack(df["observation.contact.force"].to_numpy())
    f[3, 2] = -15.0
    f[9, 1] = np.nan
    df["observation.contact.force"] = list(f)
    # +inf is never a legitimate safety.stability_margin value (only -inf,
    # the flight-phase sentinel, is) — proves C8's dedicated stability_margin
    # check actually fires, not just the generic NaN/range sweeps.
    sm = df["safety.stability_margin"].to_numpy(dtype=float, copy=True)
    sm[5] = float("inf")
    df["safety.stability_margin"] = sm
    write_steps(df, p)


def break_rule_inputs(root: Path) -> None:
    """데이터에 없는 입력을 요구하는 규칙을 시나리오 적용 범위에 넣어 C8 을 시험한다."""
    p = root / "meta" / "scenarios.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["scenarios"][0]["rules"].append("R-OBJ-DROP-01")   # 이 픽스처에는 파지물 필드가 없다
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_inputs_missing(root: Path) -> None:
    """계산할 수 없다고 선언된 규칙을 되살려 C1 을 시험한다."""
    p = root / "meta" / "rules.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["rules"].append({
        "rule_id": "R-OBJ-DROP-01", "condition": "파지물 낙하", "outcome": "fail",
        "hazard_class": "drop", "standard": "-", "inputs": [],
        "inputs_missing": "파지물 상태 필드가 스키마에 없음",
    })
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_rules(root: Path) -> None:
    """추적 불가능한 임계값을 만들어 C1 의 완결성 검사를 시험한다.

    값을 비우는 것만으로는 더 이상 결함이 아니다 — 계약이 「값」과 「값 대신 조항
    참조」를 둘 다 허용하기 때문이다(규격서 §9.3.1). 결함은 **값도 참조도 없는**
    상태이고, 그때 받는 쪽은 자기 사본이 있어도 어느 표를 봐야 하는지 알 수 없다.
    """
    p = root / "meta" / "rules.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    e = d["contact_limits"]["BR-05"]["limits"]["force_quasistatic"]
    e["value"] = None
    e.pop("standard_ref", None)
    d["status"] = "thresholds_pending"
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_not_applicable_on_permitted(root: Path) -> None:
    """접촉이 허용된 부위에 not_applicable 면제를 달아 C1 을 시험한다.

    이 면제가 없으면 표준을 다 확보해도 C1 을 통과할 수 없다(BR-01·BR-02 의 동적
    칸이 「해당 없음」이다). 그러나 아무 데서나 쓸 수 있으면 빈 칸을 덮는 가장 싼
    방법이 된다. 면제의 근거는 부위가 접촉 금지라는 사실 하나뿐이다.
    """
    p = root / "meta" / "rules.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    tm = d["contact_limits"]["BR-05"]["transient_multiplier"]   # BR-05 는 접촉 허용
    tm["value"] = None
    tm["not_applicable"] = True
    tm["reason"] = "면제 사유가 없는데 면제를 주장한다"
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_not_applicable_with_value(root: Path) -> None:
    """not_applicable 이면서 값도 있는 항목을 만들어 C1 을 시험한다.

    둘 중 하나는 거짓인데 읽는 쪽은 어느 쪽인지 알 수 없다. 「해당 없음」이라 해
    놓고 비교에 쓸 수 있는 수치를 남기면, 그 수치가 조용히 판정에 쓰인다.
    """
    p = root / "meta" / "rules.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["contact_limits"]["BR-01"]["transient_multiplier"]["value"] = 2
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_contact_permitted_dropped(root: Path) -> None:
    """contact_permitted 를 지워 C1 을 시험한다.

    이 플래그가 없으면 not_applicable 면제가 성립하는지 판단할 근거가 사라지고,
    「접촉이 금지된 부위」라는 사실 자체가 배포본에서 없어진다.
    """
    p = root / "meta" / "rules.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    del d["contact_limits"]["BR-01"]["contact_permitted"]
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_threshold_nonfinite(root: Path) -> None:
    """임계값에 NaN 을 넣어 C1 이 "채워짐" 으로 통과시키지 않는지 시험한다.

    NaN 은 float 이라 수치 검사를 통과한다. 통과시키면 그 임계값을 쓰는 비교가
    영영 거짓이 되어 라벨이 조용히 사라진다 — 미기재(broken_rules)보다 나쁘다.
    """
    p = root / "meta" / "rules.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    e = next(r for r in d["rules"] if r["rule_id"] == "R-TS15066-SSM-01")
    e["parameters"]["robot_stop_time"]["value"] = float("nan")
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_missing_parameter(root: Path) -> None:
    """필수 파라미터 이름 자체를 지워 C1 을 시험한다.

    값이 빈 것(broken_rules)은 임계값 검사가 잡지만, 이름이 없으면 순회 대상이 되지
    않아 조용히 통과한다 — 생성기 쪽은 같은 상황을 missing 으로 거른다.
    """
    p = root / "meta" / "rules.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    e = next(r for r in d["rules"] if r["rule_id"] == "R-TS15066-SSM-01")
    del e["parameters"]["robot_stop_time"]
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_duplicate_rule_id(root: Path) -> None:
    """같은 rule_id 를 둘 두고 앞엣것만 망가뜨려 C1 을 시험한다.

    검증기는 dict 로 뒤엣것을 남기고 생성기 로더는 next() 로 앞엣것을 읽으므로,
    이 배치에서는 검증기만 통과한다 — 중복 자체를 거부하는 것이 유일한 방어다.
    """
    p = root / "meta" / "rules.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    i = next(k for k, r in enumerate(d["rules"]) if r["rule_id"] == "R-TS15066-SSM-01")
    broken = json.loads(json.dumps(d["rules"][i]))
    del broken["parameters"]["robot_stop_time"]
    d["rules"].insert(i, broken)          # 결함본이 앞, 정상본이 뒤
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_threshold_negative(root: Path) -> None:
    """임계값에 음수를 넣어 C1 이 통과시키지 않는지 시험한다."""
    p = root / "meta" / "rules.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    e = next(r for r in d["rules"] if r["rule_id"] == "R-TS15066-SSM-01")
    e["parameters"]["intrusion_distance"]["value"] = -0.2
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_zmp_half_nan(root: Path) -> None:
    """ZMP 의 한쪽 성분만 NaN 으로 만들어 C8 을 시험한다.

    두 성분 모두 NaN 은 스키마가 정의한 표식이라 통과해야 하고, 한쪽만 NaN 인
    것은 산출 오류다. 둘을 가르지 못하면 표식이 결함을 숨긴다.
    """
    p = _data_path(root)
    df = pd.read_parquet(p)
    vals = [list(x) for x in df["safety.zmp"]]
    vals[len(vals) // 2][1] = float("nan")
    df["safety.zmp"] = vals
    write_steps(df, p)


def break_step_null(root: Path) -> None:
    """A null where the step table holds values: what a rewrite through pandas leaves of NaN.

    Read through pandas the row is [nan, nan] -- "no support polygon", which the spec allows --
    so only the check that reads the file through Arrow sees it (C7)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    p = _data_path(root)
    table = pq.read_table(p)
    i = table.schema.get_field_index("safety.zmp")
    rows = table.column(i).to_pylist()
    rows[len(rows) // 2] = [None, None]
    table = table.set_column(i, table.schema.field(i), pa.array(rows, type=table.schema.field(i).type))
    pq.write_table(table, p)


# ── Where an episode's rows are (C7, c7_data_spans) ──
# LeRobot finds an episode's steps by meta/episodes' pointers, not by the rows' own episode_index.
# Each of these leaves every row with the right episode_index, frame_index and timestamp, so the
# checks that join rows to episodes by episode_index pass all four.

def break_span_swapped(root: Path) -> None:
    """Two episodes' stretches exchanged in meta/episodes (C7): a loader asked for episode 1
    hands out episode 2's steps. Same length, same file -- only whose rows they are is wrong."""
    p = _eps_path(root)
    df = pd.read_parquet(p)
    span = ["dataset_from_index", "dataset_to_index"]
    df.loc[[1, 2], span] = df.loc[[2, 1], span].to_numpy()
    df.to_parquet(p, index=False)


def break_span_end_short(root: Path) -> None:
    """One episode's dataset_to_index a row short (C7): the stretch no longer has ``length``
    rows, and the episode's last step is in no episode's stretch. An episode of the second step
    file, so a check that looks only at the first does not see it."""
    p = _eps_path(root)
    df = pd.read_parquet(p)
    df.loc[EPISODES_PER_DATA_FILE + 2, "dataset_to_index"] -= 1
    df.to_parquet(p, index=False)


def break_span_file_index(root: Path) -> None:
    """One episode's data/file_index names the other step file (C7). Loading everything still
    finds the rows; loading that episode alone fetches a file they are not in."""
    p = _eps_path(root)
    df = pd.read_parquet(p)
    df.loc[EPISODES_PER_DATA_FILE + 2, "data/file_index"] = 0
    df.to_parquet(p, index=False)


def break_span_rows_out_of_place(root: Path) -> None:
    """Two episodes' rows stored in each other's place while the pointers stay (C7): the rows
    moved whole, index included, as a step file re-sorted on another key would leave them."""
    p = _data_path(root)
    df = pd.read_parquet(p)
    first, second, rest = (df[df["episode_index"] == 0], df[df["episode_index"] == 1],
                           df[df["episode_index"] > 1])
    write_steps(pd.concat([second, first, rest], ignore_index=True), p)


# ── What info.json counts (C7, c7_info_totals) ──
# LeRobot takes the number of episodes and of steps from meta/info.json and does not count its
# tables. Each of these leaves both tables and every pointer as they were: only the number is off.

def _recount(root: Path, key: str, by: int) -> None:
    p = root / "meta" / "info.json"
    info = json.loads(p.read_text(encoding="utf-8"))
    info[key] += by
    p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def break_total_frames(root: Path) -> None:
    """info.json's total_frames one less than the step table has rows (C7): len(dataset) is one
    short, and whoever reads that many items never gets the last step."""
    _recount(root, "total_frames", -1)


def break_total_episodes(root: Path) -> None:
    """info.json's total_episodes one more than the episode table has rows (C7): the loader looks
    for an episode the release does not have."""
    _recount(root, "total_episodes", +1)


# ── The task of a step (C7, c7_task_table) ──
# LeRobot hands every item a sentence: the row of meta/tasks.parquet at the step's task_index,
# read only when info.json's total_tasks is above 0. Each of these breaks one of those and leaves
# the rest as it was.

def break_task_count(root: Path) -> None:
    """info.json's total_tasks 0 beside a table that has its row (C7): the loader does not read
    the table then, and no item can be read."""
    p = root / "meta" / "info.json"
    info = json.loads(p.read_text(encoding="utf-8"))
    info["total_tasks"] = 0
    p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def break_task_table_absent(root: Path) -> None:
    """No meta/tasks.parquet (C7): the loader takes the copy for incomplete and goes to the Hub
    for the metadata."""
    (root / TASKS_PATH).unlink()


def break_task_sentences_not_index(root: Path) -> None:
    """The same two columns written as a plain table, the sentences a column and not the frame's
    index (C7): the loader hands out the row number, 0, where the sentence belongs -- and raises
    nothing."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pd.read_parquet(root / TASKS_PATH)
    pq.write_table(pa.table({"task_index": table["task_index"].tolist(), "task": list(table.index)}),
                   root / TASKS_PATH)


def break_task_index_outside(root: Path) -> None:
    """One step's task_index the number of rows the table has, one past its last row (C7): that
    item raises IndexError, and a plain ``for`` over the release ends there without one. A step of
    the second step file, so a check that looks only at the first does not see it."""
    p = root / "data" / "chunk-000" / "file-001.parquet"
    df = pd.read_parquet(p)
    df.loc[3, "task_index"] = len(pd.read_parquet(root / TASKS_PATH))
    write_steps(df, p)


def break_task_index_beyond_feature(root: Path) -> None:
    """The features declare task_index an int8 and one step holds 128 (C7): the loader casts the
    stored column to the declared type and stops at opening on a value it cannot hold. 128 is no
    row of the table either, but the loader never gets as far as the table with it."""
    p = root / "meta" / "info.json"
    info = json.loads(p.read_text(encoding="utf-8"))
    info.setdefault("features", {})["task_index"] = {"dtype": "int8", "shape": [1], "names": None}
    p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    q = root / "data" / "chunk-000" / "file-001.parquet"
    df = pd.read_parquet(q)
    df.loc[3, "task_index"] = 128
    write_steps(df, q)


def _edit_info(root: Path, change) -> None:
    p = root / "meta" / "info.json"
    info = json.loads(p.read_text(encoding="utf-8"))
    change(info)
    p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def break_features_absent(root: Path) -> None:
    """info.json without its features (C7): the loader makes the step table's columns and types
    of them and stops at opening without them. Every file is as it was."""
    _edit_info(root, lambda info: info.pop("features"))


def break_feature_undeclared_column(root: Path) -> None:
    """The features do not declare one of the step columns (C7): the loader makes the table of
    what is declared and stops at opening on a file's column that is not. The files are as
    they were -- read through pandas, nothing is missing."""
    _edit_info(root, lambda info: info["features"].pop("safety.com"))


def break_feature_column_absent(root: Path) -> None:
    """The features declare a column no step file has (C7): the loader fills it with nulls and
    opens, and every item comes out with nothing under that key."""
    def declare(info: dict) -> None:
        info["features"]["observation.extra"] = {"dtype": "float64", "shape": [3], "names": None}
    _edit_info(root, declare)


def break_feature_shape(root: Path) -> None:
    """The features declare safety.com a list of 2 where every step holds 3 (C7): the loader
    casts what is stored to what is declared, and a list of another length does not go in."""
    _edit_info(root, lambda info: info["features"]["safety.com"].update(shape=[2]))


def break_feature_dtype_name(root: Path) -> None:
    """A feature's dtype written as numpy names the type, "<f8" (C7): the loader makes a type of
    the name by asking pyarrow for it, and stops at opening on a name pyarrow does not have."""
    _edit_info(root, lambda info: info["features"]["safety.zmp"].update(dtype="<f8"))


def break_feature_frame_index_width(root: Path) -> None:
    """The features declare frame_index an int8 and the episodes run past 127 steps (C7): the
    loader casts the stored int64 to it and stops at opening on the first step that does not
    fit. The same rule that holds task_index's width, on another column."""
    _edit_info(root, lambda info: info["features"]["frame_index"].update(dtype="int8"))


def break_duplicate_episode_key(root: Path) -> None:
    """조인 키를 중복시켜 C7·C8 을 시험한다.

    dict 로 만들면 뒤엣것이 앞엣것을 조용히 덮어, 앞 에피소드의 프레임이 남의 값으로
    검사된다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    df.loc[1, "episode_index"] = df.loc[0, "episode_index"]
    df.to_parquet(p, index=False)


def break_bad_triggered_rules(root: Path) -> None:
    """triggered_rules 를 목록이 아닌 값으로 만들어 C4 를 시험한다.

    목록으로 안 읽히는 값을 빈 목록으로 삼키면 없는 규칙 ID 검사를 통째로 비껴간다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    df["triggered_rules"] = [json.dumps(list(v)) for v in df["triggered_rules"]]
    df.loc[0, "triggered_rules"] = "R-INVENTED"
    df.to_parquet(p, index=False)


def break_canon_version(root: Path) -> None:
    """배포 규격 정의의 schema_version 만 바꿔 C1 을 시험한다.

    경고로 끝내던 동안, 버전만 바꾸면 내용 전체가 대조 없이 통과했다 — 기본 실행은
    WARN 에서 exit 0 이므로 배포 게이트가 통째로 열려 있었다.
    """
    p = root / "meta" / "safety_schema.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["schema_version"] = "0.1"
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_canon_list_as_object(root: Path) -> None:
    """목록이어야 할 자리에 같은 키를 담은 객체를 넣어 C1 을 시험한다."""
    p = root / "meta" / "safety_schema.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["contact_types"] = {k: True for k in d["contact_types"]}
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_orphan_frame_episode(root: Path) -> None:
    """에피소드 메타에 없는 프레임 에피소드를 넣어 C7·C8 을 시험한다.

    조인이 안 되면 그 그룹은 조용히 건너뛰어져 아무 검사도 받지 않았다.
    """
    p = _data_path(root)
    df = pd.read_parquet(p)
    idx = df.index[-3:]
    df.loc[idx, "episode_index"] = 999
    write_steps(df, p)


def break_fps_text(root: Path) -> None:
    """fps 를 문자열로 만들어 C5 를 시험한다. float() 가 죽으면 exit 3 이 된다."""
    p = _eps_path(root)
    df = pd.read_parquet(p)
    df["fps"] = ["not-a-number"] + [str(v) for v in df["fps"].iloc[1:]]
    df.to_parquet(p, index=False)


def break_fps_zero(root: Path) -> None:
    """fps 를 0 으로 만들어 C5 가 보고하는지, 그리고 C7 이 죽지 않는지 시험한다.

    C7 이 양수 검사 전에 1/fps 를 하던 동안 검증기가 통째로 죽어 exit 3 이 됐다 —
    그건 「막았다」가 아니라 「검사하지 못했다」이다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    df["fps"] = 0.0
    df.to_parquet(p, index=False)


def break_late_joint_width(root: Path) -> None:
    """에피소드 뒤쪽 행의 관절 폭만 줄여 C8 을 시험한다.

    폭 검사가 에피소드별 첫 행만 보던 동안 이런 데이터는 통과했다.
    """
    p = _data_path(root)
    df = pd.read_parquet(p)
    vals = [list(x) for x in df["action"]]
    vals[-1] = vals[-1][:-1]
    df["action"] = vals
    write_steps(df, p)


def break_canon_shape_list(root: Path) -> None:
    """fixed_shapes 를 키 목록으로 바꿔 C1 을 시험한다. 길이를 담지 못하는 형태다."""
    p = root / "meta" / "safety_schema.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["fixed_shapes"] = list(d["fixed_shapes"])
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_canon_json_fields(root: Path) -> None:
    """json_encoded_fields 가 실재하지 않는 필드를 가리키게 해 C1 을 시험한다."""
    p = root / "meta" / "safety_schema.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["json_encoded_fields"]["step"] = ["observation.not.a.field"]
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_required_null(root: Path) -> None:
    """필수 에피소드 필드를 null 로 만들어 C5 를 시험한다.

    컬럼이 있는 것과 값이 있는 것은 다르다. enum 검사가 `pd.notna` 로 감싸여 있어
    null 은 그냥 지나갔다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    df["hazard_class"] = df["hazard_class"].astype("object")
    df.loc[1, "hazard_class"] = None
    df.to_parquet(p, index=False)


def break_late_shape(root: Path) -> None:
    """뒤쪽 프레임의 고정 길이 열을 무너뜨려 C8 을 시험한다.

    첫 행만 보던 동안 이런 데이터는 통과했다.
    """
    p = _data_path(root)
    df = pd.read_parquet(p)
    vals = [list(x) for x in df["observation.root_pose"]]
    vals[len(vals) // 2] = vals[len(vals) // 2][:-1]
    df["observation.root_pose"] = vals
    write_steps(df, p)


def break_action_nan(root: Path) -> None:
    """필수 벡터 열에 NaN 을 넣어 C8 을 시험한다. 유한성을 아무도 안 보고 있었다."""
    p = _data_path(root)
    df = pd.read_parquet(p)
    vals = [list(x) for x in df["action"]]
    vals[len(vals) // 4][0] = float("nan")
    df["action"] = vals
    write_steps(df, p)


def break_canon_shapes(root: Path) -> None:
    """배포된 규격 정의의 fixed_shapes 길이를 바꿔 C1 을 시험한다."""
    p = root / "meta" / "safety_schema.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["fixed_shapes"]["observation.imu"] = 3
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_dof_mismatch(root: Path) -> None:
    """에피소드 메타의 dof 를 관절 배열 폭과 다르게 만들어 C8 을 시험한다.

    이 검사는 스텝 프레임과 에피소드 메타를 조인해야 도는데, 픽스처가 배포본에 없는
    `episode_id` 를 스텝에 넣고 있던 동안 **픽스처에서만 조인되고 실배포본에서는
    돌지 않았다**.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    df["dof"] = int(df["dof"].iloc[0]) + 7
    df.to_parquet(p, index=False)


def break_v_rel_nan(root: Path) -> None:
    """v_rel 을 NaN 으로 만들어 C8 을 시험한다.

    이 픽스처가 없던 동안 400행 전체를 NaN 으로 바꿔도 검증기가 아무 말도 하지
    않았다 — 필수 필드 목록에만 이름이 들어가 있었기 때문이다.
    """
    p = _data_path(root)
    df = pd.read_parquet(p)
    vals = [list(x) for x in df["observation.contact.v_rel"]]
    vals[len(vals) // 3][0] = float("nan")
    df["observation.contact.v_rel"] = vals
    write_steps(df, p)


def break_v_rel_negative(root: Path) -> None:
    """음수 상대속도로 C8 을 시험한다. 크기이므로 음수가 될 수 없다."""
    p = _data_path(root)
    df = pd.read_parquet(p)
    vals = [list(x) for x in df["observation.contact.v_rel"]]
    vals[len(vals) // 2][3] = -1.5
    df["observation.contact.v_rel"] = vals
    write_steps(df, p)


def break_v_rel_length(root: Path) -> None:
    """부위 수와 다른 길이로 C3 을 시험한다."""
    p = _data_path(root)
    df = pd.read_parquet(p)
    vals = [list(x) for x in df["observation.contact.v_rel"]]
    vals[1] = vals[1][:-1]
    df["observation.contact.v_rel"] = vals
    write_steps(df, p)


def break_episode_id_format(root: Path) -> None:
    """규격서 §3 과 다른 형식의 episode_id 로 C5 를 시험한다."""
    p = _eps_path(root)
    df = pd.read_parquet(p)
    ids = list(df["episode_id"].astype(str))
    ids[1] = "baseline-S1a-h1-2026090100001-00001"   # 규격서가 싣고 있던 옛 형식
    df["episode_id"] = ids
    df.to_parquet(p, index=False)


def break_episode_id_seed(root: Path) -> None:
    """형식은 맞으나 담긴 시드가 randomization_seed 와 다른 경우를 시험한다.

    형식만 보는 검사는 이것을 통과시킨다. 그러면 "파일명으로 어느 시드인지 알 수
    있다" 는 규격서 §3 의 주장이 깨진 채로 배포된다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    ids = list(df["episode_id"].astype(str))
    seq = int(ids[2].split("_")[1])
    ids[2] = f"S1a_{seq:04d}_seed999999999"
    df["episode_id"] = ids
    df.to_parquet(p, index=False)


def break_run_start_missing(root: Path) -> None:
    """sim 에피소드 하나에서 run_start_seq 를 비운다 (C5 조건부 필수)."""
    p = _eps_path(root)
    df = pd.read_parquet(p)
    df["run_start_seq"] = df["run_start_seq"].astype("float64")
    df.loc[1, "run_start_seq"] = np.nan
    df.to_parquet(p, index=False)


def break_run_start_past_seq(root: Path) -> None:
    """2번 에피소드가 7번에서 시작한 실행의 것이라고 말한다 — 그 실행은 2번을 만들 수 없다."""
    p = _eps_path(root)
    df = pd.read_parquet(p)
    df.loc[2, "run_start_seq"] = 7
    df.to_parquet(p, index=False)


def _edit_sim_backend(root: Path, row: int, change) -> None:
    p = _eps_path(root)
    df = pd.read_parquet(p)
    backend = json.loads(df.loc[row, "sim_backend"])
    change(backend)
    df.loc[row, "sim_backend"] = as_published(backend)
    df.to_parquet(p, index=False)


def break_generator_dirty(root: Path) -> None:
    """커밋되지 않은 변경이 있는 트리에서 만든 에피소드 — 변환기가 거부하는 것이 실렸다."""
    _edit_sim_backend(root, 3, lambda b: b.update(generator_dirty=True))


def break_generator_commit(root: Path) -> None:
    """생성 커밋 자리에 줄인 해시 — 어느 커밋인지 하나로 정해지지 않는다."""
    _edit_sim_backend(root, 4, lambda b: b.update(generator_commit="9ab48ae"))


def break_com_nan(root: Path) -> None:
    """CoM 에 NaN 을 넣어 C8 을 시험한다. 무게중심에는 표식이 없다."""
    p = _data_path(root)
    df = pd.read_parquet(p)
    vals = [list(x) for x in df["safety.com"]]
    vals[len(vals) // 3][2] = float("nan")
    df["safety.com"] = vals
    write_steps(df, p)


def break_contact_type(root: Path) -> None:
    """접촉 유형에 허용되지 않은 값을 넣어 **enum 검사(C8)** 를 시험한다.

    이 열은 JSON 문자열로 실린다(`json_encoded_fields`). `list(x)` 를 그대로 쓰면
    문자열이 **글자 단위로** 쪼개져 길이 96 짜리 배열이 되고, 그러면 C3 의 길이
    검사가 먼저 걸린다 — 의도한 enum 위반은 시험되지 않은 채 「검출」로 보인다.
    디코드해서 한 원소만 바꾸고 다시 인코딩한다.
    """
    p = _data_path(root)
    df = pd.read_parquet(p)
    decoded = [json.loads(x) for x in df["observation.contact.type"]]
    # 첫 프레임이 아닌 곳을 오염시킨다. 첫 프레임만 검사하면 통과하던 회귀를 막는다.
    decoded[len(decoded) // 2][0] = "sustained"          # 허용값 아님
    df["observation.contact.type"] = [json.dumps(v, ensure_ascii=False) for v in decoded]
    write_steps(df, p)


def break_scenario_rules(root: Path) -> None:
    """적용 규칙 선언이 실재하지 않는 규칙을 가리키게 해 C1 을 시험한다.

    이 상태를 방치하면 적용 집합이 공집합이 되어 C8 이 아무 입력도 요구하지 않는다.
    """
    p = root / "meta" / "scenarios.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["scenarios"][0]["rules"] = ["R-TS15066-A2-99"]      # 없는 부위 코드
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_required_fields(root: Path) -> None:
    """시나리오가 선언한 필수 필드가 데이터에 없는 상태를 만들어 C8 을 시험한다."""
    p = root / "meta" / "scenarios.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["scenarios"][0].setdefault("required_fields", []).append("safety.absent_field")
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_safety_schema(root: Path) -> None:
    """동봉된 기계가독 규격이 데이터의 실제 규격과 다르다.

    C1 이 존재만 보던 시절에는 이 파일을 통째로 비워도 검증이 통과했다. 배포본을
    받은 쪽은 이 파일을 규격으로 읽으므로, 조용히 어긋나면 안 된다.
    """
    p = root / "meta" / "safety_schema.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["body_regions"].pop("BR-12", None)
    d["enums"]["outcome"] = ["nominal", "near_miss"]      # fail 이 빠졌다
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_empty_rules(root: Path) -> None:
    """규칙 목록을 비워 selector 를 검증할 수 없게 만든다.

    예전에는 이 상태에서 오타 selector 가 그대로 적용 집합이 되어 C8 이 아무 입력도
    요구하지 않았다. 검사를 건너뛰는 것과 같으므로 배포를 막아야 한다.
    """
    p = root / "meta" / "rules.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["rules"] = []
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_termination_list(root: Path) -> None:
    """`termination` 을 배열로 만들어 C5 를 시험한다.

    「객체 또는 배열」로 뭉뚱그린 검사는 이것을 통과시켰고, 그러면 종료 사유가
    통째로 비어 있는데 오류가 안 난다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["termination"].astype(str))
    vals[1] = "[]"
    df["termination"] = vals
    df.to_parquet(p, index=False)


def break_termination_member(root: Path) -> None:
    """`termination` 의 멤버를 망가뜨려 C5 를 시험한다. 빈 객체도 dict 다."""
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["termination"].astype(str))
    vals[1] = json.dumps({"terminated": "yes", "truncated": 0, "reason": "fall"},
                         ensure_ascii=False)
    df["termination"] = vals
    df.to_parquet(p, index=False)


def break_notes_member_dropped(root: Path) -> None:
    """`notes` 에서 provenance 필드를 빼 C5 를 시험한다.

    나머지가 남아 있으면 「비어 있지 않다」가 성립해 통과했다. 회랑을 만든
    입력이 사라져도 아무도 울지 않았다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["notes"].astype(str))
    obj = json.loads(vals[1])
    obj.pop("ssm_params", None)
    vals[1] = json.dumps(obj, ensure_ascii=False)
    df["notes"] = vals
    df.to_parquet(p, index=False)


def break_notes_member_type(root: Path) -> None:
    """`notes` 멤버의 형을 바꿔 C5 를 시험한다."""
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["notes"].astype(str))
    obj = json.loads(vals[1])
    obj["human_region_radius"] = "0.1"      # 문자열은 수치가 아니다
    vals[1] = json.dumps(obj, ensure_ascii=False)
    df["notes"] = vals
    df.to_parquet(p, index=False)


def break_canon_notes_members(root: Path) -> None:
    """동봉 규격에서 `notes_members` 를 빼 C1 을 시험한다.

    없으면 C5 의 notes 검사가 **조용히 건너뛴다** — 검사하지 않은 것이 통과한
    것처럼 보인다.
    """
    p = root / "meta" / "safety_schema.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d.pop("notes_members", None)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def break_redistributed_type(root: Path) -> None:
    """`scene_assets[].redistributed` 를 문자열로 만들어 C6 를 시험한다.

    `bool("false")` 는 True 다. 강제 변환하면 재배포하지 않는 자산이 재배포된
    것으로 뒤집힌다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["scene_assets"].astype(str))
    items = json.loads(vals[1])
    items[0]["redistributed"] = "false"
    vals[1] = json.dumps(items, ensure_ascii=False)
    df["scene_assets"] = vals
    df.to_parquet(p, index=False)


def break_canon_unrouted_shape(root: Path) -> None:
    """`json_encoded_fields.episode` 에서 이름을 빼 C1 을 시험한다.

    그릇·멤버 계약이 다 있어도 이 목록에 없으면 C5 가 그 필드를 보지 않는다 —
    한 줄로 검사를 통째로 끌 수 있었다.
    """
    p = root / "meta" / "safety_schema.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["json_encoded_fields"]["episode"] = [
        x for x in d["json_encoded_fields"]["episode"] if x != "notes"]
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def break_notes_empty_ssm(root: Path) -> None:
    """`notes.ssm_params` 를 빈 객체로 만들어 C5 를 시험한다.

    얕은 형만 보면 `{}` 도 object 라 통과한다. 회랑을 재현할 근거가 통째로 없는데
    정상 메타데이터로 게시된다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["notes"].astype(str))
    obj = json.loads(vals[1])
    obj["ssm_params"] = {}
    vals[1] = json.dumps(obj, ensure_ascii=False)
    df["notes"] = vals
    df.to_parquet(p, index=False)


def break_notes_ssm_diverges_from_canon(root: Path) -> None:
    """회랑을 만든 입력이 정본의 값과 다르게 만들어 C5 를 시험한다.

    정본이 값을 실은 파라미터는 배포본 안에 **사본이 둘**이 된다 — rules.json 과
    각 에피소드의 `notes.ssm_params`. 둘이 갈라지면 받는 쪽이 rules.json 만 보고
    회랑을 재현할 때 어긋나는데, 형·유한성만 보던 동안에는 아무도 몰랐다.
    어느 이름을 흔들지는 **정본에서 고른다**.
    """
    carried = _synthetic_ssm_params()
    rules = json.loads((root / "meta" / "rules.json").read_text(encoding="utf-8"))
    names = [name for rule in rules.get("rules") or []
             if rule.get("rule_id") == "R-TS15066-SSM-01"
             for name, entry in (rule.get("parameters") or {}).items()
             if isinstance(entry, dict) and entry.get("value") is not None]
    if not names:
        raise SystemExit("정본에 값이 실린 SSM 파라미터가 없어 이 픽스처를 만들 수 없습니다")
    name = sorted(names)[0]

    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["notes"].astype(str))
    obj = json.loads(vals[1])
    obj.setdefault("ssm_params", dict(carried))[name] = float(carried[name]) + 1.0
    vals[1] = json.dumps(obj, ensure_ascii=False)
    df["notes"] = vals
    df.to_parquet(p, index=False)


def break_notes_bad_region(root: Path) -> None:
    """`notes` 의 미해결 부위에 없는 코드를 넣어 C5 를 시험한다."""
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["notes"].astype(str))
    obj = json.loads(vals[1])
    obj["contact_onset_unresolved_regions"] = ["BR-99"]
    vals[1] = json.dumps(obj, ensure_ascii=False)
    df["notes"] = vals
    df.to_parquet(p, index=False)


def break_asset_missing(root: Path) -> None:
    """자산을 지우고 **다른 자리에 같은 이름**을 남겨 C6 를 시험한다.

    지우기만 하면 이름으로 재귀 탐색하는 구현도 같은 오류를 내서, 정확한 경로를 요구하는
    검사가 보호되지 않는다. 미끼를 함께 둬야 두 구현이
    갈린다.
    """
    for q in sorted((root / "assets").rglob("*")):
        if q.is_file():
            shutil.copy2(q, root / "meta" / q.name)   # 미끼 — 재귀 탐색이면 이걸 찾는다
            q.unlink()
            return
    raise RuntimeError("재배포 자산이 픽스처에 없습니다 — 이 결함을 심을 수 없습니다")


def break_notes_region_detail(root: Path) -> None:
    """`notes.not_evaluated[].regions[]` 를 망가뜨려 C5 를 시험한다.

    한 층 위(rule_id·reason)만 보고 있었다. 이 목록이 「어느 부위에 얼마가
    걸렸는데 비교하지 못했다」를 말하는 자리이므로, 값이 거짓이면 미평가 사유
    자체가 거짓이 된다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["notes"].astype(str))
    obj = json.loads(vals[1])
    obj["not_evaluated"] = [{
        "rule_id": "R-TS15066-A2-01", "reason": "합성 픽스처",
        "regions": [{"region": "BR-99", "steps": -7, "peak_force_n": float("nan")}],
    }]
    vals[1] = json.dumps(obj, ensure_ascii=False)
    df["notes"] = vals
    df.to_parquet(p, index=False)


def break_notes_hold_contaminated(root: Path) -> None:
    """`hold_contaminated_steps` 를 생성기 계약 밖으로 올려 C5 를 시험한다.

    계약은 0 또는 1 이다 — 접촉 백스톱이 손해를 접촉을 처음 보는 스텝 하나로
    묶는다. 그보다 크면 홀드가 접촉을 관통했다는 뜻이다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["notes"].astype(str))
    obj = json.loads(vals[1])
    obj["hold_contaminated_steps"] = 2
    vals[1] = json.dumps(obj, ensure_ascii=False)
    df["notes"] = vals
    df.to_parquet(p, index=False)


def break_notes_region_contradiction(root: Path) -> None:
    """개별로는 유효한데 **서로 모순**인 notes 를 만들어 C5 를 시험한다.

    `contact_observed=false` 인데 비교하지 못한 부위가 있다는 것은 성립하지 않는다.
    형식만 보면 통과했다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["notes"].astype(str))
    obj = json.loads(vals[1])
    obj["contact_observed"] = False
    obj["not_evaluated"] = [{
        "rule_id": "R-TS15066-A2-01", "reason": "합성 픽스처",
        "regions": [{"region": "BR-01", "steps": 3, "peak_force_n": 999.0}],
    }]
    vals[1] = json.dumps(obj, ensure_ascii=False)
    df["notes"] = vals
    df.to_parquet(p, index=False)


def break_canon_notes_bounds(root: Path) -> None:
    """동봉 규격에서 `notes_bounds` 를 빼 C1 을 시험한다."""
    p = root / "meta" / "safety_schema.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d.pop("notes_bounds", None)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def break_canon_conditional_steps(root: Path) -> None:
    """Drop one field from the shipped schema's camera-conditional step fields (C1).

    A consumer reads that list to know what a distribution with meta/cameras.json must
    carry; one field short, it would accept a release the validator refuses. The clean
    fixture has no cameras, so this is caught by the schema comparison, not the data.
    """
    p = root / "meta" / "safety_schema.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["conditional_required_step_fields"]["cameras"].remove("observation.bbox2d.human")
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def break_stop_table(root: Path) -> None:
    """R-TS15066-SSM-02's stop table with two rows of one state swapped (C1): the speeds no longer
    rise, and the table sits beside the parameters, where the threshold check does not go."""
    p = root / "meta" / "rules.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = next(r for r in d["rules"]
                if r["rule_id"] == "R-TS15066-SSM-02")["stop_table"]["walking"]["carry_3kg"]
    rows[0], rows[1] = rows[1], rows[0]
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_directed_parameter_missing(root: Path) -> None:
    """R-TS15066-SSM-02 without approach_speed_threshold (C1): a name that is not there is not
    walked by the threshold check, and the near misses cannot be recomputed without it."""
    p = root / "meta" / "rules.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    del next(r for r in d["rules"]
             if r["rule_id"] == "R-TS15066-SSM-02")["parameters"]["approach_speed_threshold"]
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def break_canon_step_layers(root: Path) -> None:
    """The shipped schema says safety.event is raw (C1): a reader choosing inputs by layer would
    take the judgement's output for a sensor's -- the table is compared whole for that reason."""
    p = root / "meta" / "safety_schema.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["step_field_layers"]["safety.event"] = "raw"
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _each_step_file(root: Path) -> list[Path]:
    return sorted((root / "data").rglob("*.parquet"))


def _edit_episode_notes(root: Path, episode: int, change) -> None:
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["notes"].astype(str))
    obj = json.loads(vals[episode])
    change(obj)
    vals[episode] = json.dumps(obj, ensure_ascii=False)
    df["notes"] = vals
    df.to_parquet(p, index=False)


def break_hri_column_missing(root: Path) -> None:
    """The hri distribution without safety.approach_speed_robot, in every step file and in the
    features alike (C8): the column a reader recomputes S_p and the near misses from."""
    for p in _each_step_file(root):
        write_steps(pd.read_parquet(p).drop(columns=["safety.approach_speed_robot"]), p)
    _edit_info(root, lambda info: info["features"].pop("safety.approach_speed_robot"))


def break_hri_violation_gate(root: Path) -> None:
    """separation_violation set on the steps the robot stood inside S_p without approaching (C8):
    a near miss is counted only while the robot moves toward the person."""
    p = _data_path(root)
    df = pd.read_parquet(p)
    held = ((df["safety.min_distance_human"] < df["safety.separation_required"])
            & ~df["safety.separation_violation"].astype(bool))
    if not held.any():
        raise SystemExit("hri 픽스처에 로봇이 선 채 S_p 안에 든 스텝이 없어 이 결함을 심을 수 없습니다")
    df.loc[held, "safety.separation_violation"] = True
    write_steps(df, p)


def break_hri_negative_approach(root: Path) -> None:
    """A negative approach speed (C8): the column is a speed toward the person, below 0 written as 0."""
    p = _data_path(root)
    df = pd.read_parquet(p)
    df.loc[len(df) // 2, "safety.approach_speed_human"] = -0.3
    write_steps(df, p)


def break_hri_visible_fraction(root: Path) -> None:
    """A visible fraction of 1.5 (C8): a share of the body's sample points lies between 0 and 1."""
    p = _data_path(root)
    df = pd.read_parquet(p)
    df.loc[len(df) // 2, "observation.human_visible_fraction.head_rgb"] = 1.5
    write_steps(df, p)


def break_hri_ssm_params(root: Path) -> None:
    """An episode whose notes carry a stop table other than the shipped rule's (C5): its S_p column
    came from that table, and a reader recomputing it from rules.json would get another."""
    def change(notes: dict) -> None:
        notes["ssm_params"]["stop_table"]["turning"]["empty"][0][1] = 9.9
    _edit_episode_notes(root, 1, change)


def break_hri_task_step(root: Path) -> None:
    """notes.task_completed_step past the episode's last step (C5)."""
    _edit_episode_notes(root, 4, lambda notes: notes.update(task_completed_step=10 ** 6))


def break_hri_task_null(root: Path) -> None:
    """notes.task_completed_step null in an hri episode (C5): every scenario that declares the
    directed rule has a completion criterion, and null would read as "no criterion"."""
    _edit_episode_notes(root, 4, lambda notes: notes.update(task_completed_step=None))


def break_hri_task_stages(root: Path) -> None:
    """A stage of notes.task_stages that says it was completed with no step it ended at (C5): a stage is
    done exactly when it has an end_step, and a reader taking the stage's flag would count a stage the
    episode never finished."""
    def change(notes: dict) -> None:
        notes["task_stages"][-1].update(completed=True, end_step=-1)
    _edit_episode_notes(root, 4, change)


def break_hri_contact_without_onset(root: Path) -> None:
    """A touch whose contact_onset was dropped (C8): the steps stay typed and the force stays, and
    the labeller, which knows contact by contact_onset alone, would not see the contact."""
    for p in _each_step_file(root):
        df = pd.read_parquet(p)
        events = [json.loads(s) for s in df["safety.event"]]
        hits = [i for i, e in enumerate(events) if "contact_onset" in e]
        if hits:
            events[hits[0]].remove("contact_onset")
            df["safety.event"] = [as_published(e) for e in events]
            write_steps(df, p)
            return
    raise SystemExit("hri 픽스처에 contact_onset 이 없어 이 결함을 심을 수 없습니다")


def _edit_episode_table(root: Path, change) -> None:
    p = _eps_path(root)
    df = pd.read_parquet(p)
    change(df)
    df.to_parquet(p, index=False)


def break_hri_spec_missing(root: Path) -> None:
    """The episode table without episode_spec (C5): a table scenario's episodes say nothing of the conditions they
    were recorded in."""
    _edit_episode_table(root, lambda df: df.pop("episode_spec"))


def break_hri_spec_shape(root: Path) -> None:
    """One episode's episode_spec without its repro member (C5): the spec does not say how to run the episode again."""
    def change(df: pd.DataFrame) -> None:
        spec = json.loads(df.at[1, "episode_spec"])
        del spec["repro"]
        df.at[1, "episode_spec"] = json.dumps(spec, ensure_ascii=False)
    _edit_episode_table(root, change)


def break_hri_grid_cell(root: Path) -> None:
    """A random row whose robot cell is 3 (C5): a random row lies on no cell of the grid, and a reader counting the
    failure map by its cells would put it there."""
    def change(df: pd.DataFrame) -> None:
        df.at[2, "grid_cell_robot"] = 3
    _edit_episode_table(root, change)


def break_hri_evidence_text(root: Path) -> None:
    """evidence.force_ratio_max written as text (C5): the evidence is a number column a reader filters the episode
    table by."""
    def change(df: pd.DataFrame) -> None:
        df["evidence.force_ratio_max"] = [repr(v) for v in df["evidence.force_ratio_max"]]
    _edit_episode_table(root, change)


def break_layer_missing_column(root: Path) -> None:
    """A step column the schema gives no layer, declared in the features so the loader takes it
    (C8): a reader picking raw inputs by layer cannot place it."""
    for p in _each_step_file(root):
        df = pd.read_parquet(p)
        df["observation.extra_signal"] = 0.0
        write_steps(df, p)
    _edit_info(root, lambda info: info["features"].update(
        {"observation.extra_signal": {"dtype": "float64", "shape": [1], "names": None}}))


def break_asset_canon_flag(root: Path) -> None:
    """동봉 자산 정본에서 재배포 플래그를 뒤집어 C6 를 시험한다.

    한 방향만 보던 동안 이쪽에서 검사를 끌 수 있었다.
    """
    p = root / "meta" / "assets.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["scene"][human_model_key()]["redistributed"] = False
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def break_notes_rule_id_type(root: Path) -> None:
    """`not_evaluated[].rule_id` 를 수치로 바꿔 C5 를 시험한다.

    정본이 string 이라 선언하고 문서가 「C5 가 강제한다」고 적는데, 정작 `str()` 로
    강제 변환하고 있어 통과했다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["notes"].astype(str))
    obj = json.loads(vals[1])
    obj["contact_observed"] = True
    obj["not_evaluated"] = [{"rule_id": 7, "reason": 9, "regions": []}]
    vals[1] = json.dumps(obj, ensure_ascii=False)
    df["notes"] = vals
    df.to_parquet(p, index=False)


def break_asset_license_one_episode(root: Path) -> None:
    """**한 에피소드만** 자산 라이선스를 바꿔 C6 를 시험한다.

    집합으로만 대조하던 동안 통과했고, 에피소드별 대조를 넣은 뒤에도 그것을
    지켜 주는 픽스처가 없었다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["scene_assets"].astype(str))
    items = json.loads(vals[1])
    items[1]["license"] = "MIT"
    vals[1] = json.dumps(items, ensure_ascii=False)
    df["scene_assets"] = vals
    df.to_parquet(p, index=False)


def break_asset_flag_one_episode(root: Path) -> None:
    """**한 행만** 재배포 플래그를 뒤집어 C6 를 시험한다."""
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["scene_assets"].astype(str))
    items = json.loads(vals[2])
    items[1]["redistributed"] = False
    vals[2] = json.dumps(items, ensure_ascii=False)
    df["scene_assets"] = vals
    df.to_parquet(p, index=False)


def break_asset_unknown_path(root: Path) -> None:
    """정본에 없는 자산을 「참조만 한다」며 넣어 C6 를 시험한다.

    재배포 플래그가 거짓이면 아무 검사도 걸리지 않았다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["scene_assets"].astype(str))
    items = json.loads(vals[1])
    items.append({"path": "nucleus://third-party/unknown.usd",
                  "license": "Proprietary", "redistributed": False})
    vals[1] = json.dumps(items, ensure_ascii=False)
    df["scene_assets"] = vals
    df.to_parquet(p, index=False)


def break_notes_aggregate_without_regions(root: Path) -> None:
    """집계 항목에서 부위를 지워 C5 를 시험한다.

    `regions` 를 일괄 선택으로 만들었더니 출처·크기 없는 미분류 힘 항목이
    통과했다.
    """
    p = _eps_path(root)
    df = pd.read_parquet(p)
    vals = list(df["notes"].astype(str))
    obj = json.loads(vals[1])
    obj["contact_observed"] = True
    obj["not_evaluated"] = [{"rule_id": "R-TS15066-A2-*", "reason": "미분류 힘"}]
    vals[1] = json.dumps(obj, ensure_ascii=False)
    df["notes"] = vals
    df.to_parquet(p, index=False)


def break_asset_unreferenced(root: Path) -> None:
    """Ship the canon's textured asset in a batch whose episodes do not use it (C6).

    A converter that ships every redistributable asset the canon knows puts the v1 dummy into
    an S1d distribution whose episodes never load it. The file travels under this
    batch's licence and names nothing in it.
    """
    (root / "assets" / Path(TEXTURED_ASSET["path"]).name).write_bytes(USD_PLACEHOLDER)


def break_asset_left_out(root: Path) -> None:
    """Leave the human model's USD out of every episode's scene_assets and delete its file (C6).

    What ships is drawn from scene_assets, so with the dummy gone from every list and from
    assets/, nothing in the lists asks for it -- a C6 that reads them alone gives 0 ERRORs, where
    one that demands every redistributable canon asset gives 2. human_model still names it, and
    the scenario (meta/scenarios.json) still says which model it uses.
    """
    gone = ASSETS[HUMAN_MODEL_ASSET]["path"]
    p = _eps_path(root)
    df = pd.read_parquet(p)
    df["scene_assets"] = [json.dumps([a for a in json.loads(v) if a.get("path") != gone],
                                     ensure_ascii=False) for v in df["scene_assets"].astype(str)]
    df.to_parquet(p, index=False)
    (root / "assets" / Path(gone).name).unlink()


def break_asset_usd_bytes(root: Path) -> None:
    """Ship another file under a redistributed USD's name (C6): right name, non-empty, not the
    bytes meta/assets.json pins (usd_sha256). C6 looked at existence and size only."""
    p = root / "assets" / Path(ASSETS[2]["path"]).name
    p.write_bytes(p.read_bytes() + b"# not the pinned asset\n")


def break_texture_missing(root: Path) -> None:
    """Remove one texture the shipped USD references (C6): its ./textures/ path resolves to nothing."""
    (root / "assets" / "textures" / sorted(TEXTURES)[-1]).unlink()


def break_texture_bytes(root: Path) -> None:
    """Change one shipped texture's bytes (C6): the provenance record names another file."""
    p = root / "assets" / "textures" / sorted(TEXTURES)[0]
    p.write_bytes(p.read_bytes() + b"retouched\n")


def break_texture_unlisted(root: Path) -> None:
    """Add a file under assets/textures/ that no bundle.files row accounts for (C6): a file
    with no stated source or licence, shipping under the dataset's."""
    (root / "assets" / "textures" / "stray_decal.png").write_bytes(b"from nowhere\n")


def _shipped_cameras(root: Path) -> dict:
    return json.loads((root / "meta" / "cameras.json").read_text(encoding="utf-8"))


def break_media_frame_count(root: Path) -> None:
    """One camera span one frame short of its episode's steps (C7).

    meta/episodes says which stretch of a camera file is which episode; a span one frame off
    the step count means the stream's pictures and the parquet rows are not one to one. The
    last camera, in an episode past the first video file, so a check that looks only at the
    first of either does not see it.
    """
    cameras = _shipped_cameras(root)
    videos, _, _ = camera_fields(cameras)
    p = _eps_path(root)
    df = pd.read_parquet(p)
    row = df.index[df["episode_index"] == EPISODES_PER_VIDEO_FILE + 2][0]
    df.loc[row, f"videos/{videos[-1]}/to_timestamp"] -= 1.0 / float(cameras["fps"])
    df.to_parquet(p, index=False)


def break_media_pack_missing(root: Path) -> None:
    """Remove one episode's pack while meta/packs.json still lists it (C1, C7). Copied from
    clean_media_pack (PACK_VARIANTS): the one clean fixture whose canon declares a pack.

    The list is a consumer's map from an episode to its depth and mask; a listed pack that
    was not shipped is a hole, and the pack alignment of that episode cannot be checked.
    """
    listed = json.loads((root / "meta" / "packs.json").read_text(encoding="utf-8"))
    (root / listed["3"]["file"]).unlink()


def break_media_stray_pack(root: Path) -> None:
    """packs/ and meta/packs.json in a distribution whose canon declares no pack (C1).

    Work done on top of an old conversion's copy: the canon
    switched the packs off, the files stayed. Laid out as a real pack distribution lists one
    -- the fixture's pack at its spec path and an index that hashes it true -- so what is
    wrong is only that no canon declares it.
    """
    cameras = media_canon(with_pack=True)
    write_packs(root, [pd.read_parquet(_data_path(root)).query("episode_index == 0")],
                *the_one_pack(cameras))


def break_media_bbox_out_of_range(root: Path) -> None:
    """One box one pixel past the right edge, mid-way through the second data file (C8).

    x_max == width is the off-by-one the inclusive convention rules out (x_max < width);
    meta/cameras.json bbox.coords, in bbox.camera's picture. The second file, so a check that
    reads the first only does not see it.
    """
    cameras = _shipped_cameras(root)
    _, box, _ = camera_fields(cameras)
    width = int(cameras["cameras"][cameras["bbox"]["camera"]]["width"])
    p = root / "data" / "chunk-000" / "file-001.parquet"
    df = pd.read_parquet(p)
    vals = [np.asarray(v, dtype=np.int32) for v in df[box]]
    vals[len(vals) // 2] = np.asarray([1, 1, width, 2], dtype=np.int32)
    df[box] = vals
    write_steps(df, p)


VARIANTS = {
    "broken_canon_unrouted_shape": break_canon_unrouted_shape,
    "broken_notes_empty_ssm": break_notes_empty_ssm,
    "broken_notes_ssm_diverges_from_canon": break_notes_ssm_diverges_from_canon,
    "broken_notes_bad_region": break_notes_bad_region,
    "broken_notes_region_detail": break_notes_region_detail,
    "broken_notes_region_contradiction": break_notes_region_contradiction,
    "broken_canon_notes_bounds": break_canon_notes_bounds,
    "broken_asset_canon_flag": break_asset_canon_flag,
    "broken_asset_license_one_episode": break_asset_license_one_episode,
    "broken_asset_flag_one_episode": break_asset_flag_one_episode,
    "broken_asset_unknown_path": break_asset_unknown_path,
    "broken_notes_aggregate_without_regions": break_notes_aggregate_without_regions,
    "broken_notes_rule_id_type": break_notes_rule_id_type,
    "broken_notes_hold_contaminated": break_notes_hold_contaminated,
    "broken_asset_missing": break_asset_missing,
    "broken_termination_list": break_termination_list,
    "broken_termination_member": break_termination_member,
    "broken_notes_member_dropped": break_notes_member_dropped,
    "broken_notes_member_type": break_notes_member_type,
    "broken_canon_notes_members": break_canon_notes_members,
    "broken_redistributed_type": break_redistributed_type,
    "broken_dof": break_dof,
    "broken_region": break_region,
    "broken_label": break_label,
    "broken_seed": break_seed,
    "broken_notice": break_notice,
    "broken_time": break_time,
    "broken_values": break_values,
    "broken_rules": break_rules,
    "broken_threshold_nonfinite": break_threshold_nonfinite,
    "broken_missing_parameter": break_missing_parameter,
    "broken_duplicate_rule_id": break_duplicate_rule_id,
    "broken_threshold_negative": break_threshold_negative,
    "broken_contact_type": break_contact_type,
    "broken_zmp_half_nan": break_zmp_half_nan,
    "broken_step_null": break_step_null,
    "broken_com_nan": break_com_nan,
    "broken_inputs_missing": break_inputs_missing,
    "broken_rule_inputs": break_rule_inputs,
    "broken_scenario_rules": break_scenario_rules,
    "broken_required_fields": break_required_fields,
    "broken_empty_rules": break_empty_rules,
    "broken_safety_schema": break_safety_schema,
    "broken_not_applicable_on_permitted": break_not_applicable_on_permitted,
    "broken_not_applicable_with_value": break_not_applicable_with_value,
    "broken_contact_permitted_dropped": break_contact_permitted_dropped,
    "broken_dof_mismatch": break_dof_mismatch,
    "broken_required_null": break_required_null,
    "broken_fps_zero": break_fps_zero,
    "broken_canon_version": break_canon_version,
    "broken_duplicate_episode_key": break_duplicate_episode_key,
    "broken_bad_triggered_rules": break_bad_triggered_rules,
    "broken_canon_list_as_object": break_canon_list_as_object,
    "broken_orphan_frame_episode": break_orphan_frame_episode,
    "broken_fps_text": break_fps_text,
    "broken_late_joint_width": break_late_joint_width,
    "broken_canon_shape_list": break_canon_shape_list,
    "broken_canon_json_fields": break_canon_json_fields,
    "broken_late_shape": break_late_shape,
    "broken_action_nan": break_action_nan,
    "broken_canon_shapes": break_canon_shapes,
    "broken_v_rel_nan": break_v_rel_nan,
    "broken_v_rel_negative": break_v_rel_negative,
    "broken_v_rel_length": break_v_rel_length,
    "broken_episode_id_format": break_episode_id_format,
    "broken_episode_id_seed": break_episode_id_seed,
    "broken_run_start_missing": break_run_start_missing,
    "broken_run_start_past_seq": break_run_start_past_seq,
    "broken_generator_dirty": break_generator_dirty,
    "broken_generator_commit": break_generator_commit,
    "broken_canon_conditional_steps": break_canon_conditional_steps,
    "broken_stop_table": break_stop_table,
    "broken_directed_parameter_missing": break_directed_parameter_missing,
    "broken_canon_step_layers": break_canon_step_layers,
    # Copied from clean_hri (variant_base): the hri columns, the gated violation and the notes.
    "broken_hri_column_missing": break_hri_column_missing,
    "broken_hri_violation_gate": break_hri_violation_gate,
    "broken_hri_negative_approach": break_hri_negative_approach,
    "broken_hri_visible_fraction": break_hri_visible_fraction,
    "broken_hri_ssm_params": break_hri_ssm_params,
    "broken_hri_task_step": break_hri_task_step,
    "broken_hri_task_null": break_hri_task_null,
    "broken_hri_task_stages": break_hri_task_stages,
    "broken_hri_contact_without_onset": break_hri_contact_without_onset,
    "broken_hri_spec_missing": break_hri_spec_missing,
    "broken_hri_spec_shape": break_hri_spec_shape,
    "broken_hri_grid_cell": break_hri_grid_cell,
    "broken_hri_evidence_text": break_hri_evidence_text,
    "broken_layer_missing_column": break_layer_missing_column,
    "broken_asset_unreferenced": break_asset_unreferenced,
    "broken_asset_left_out": break_asset_left_out,
    "broken_asset_usd_bytes": break_asset_usd_bytes,
    "broken_span_swapped": break_span_swapped,
    "broken_span_end_short": break_span_end_short,
    "broken_span_file_index": break_span_file_index,
    "broken_span_rows_out_of_place": break_span_rows_out_of_place,
    "broken_total_frames": break_total_frames,
    "broken_total_episodes": break_total_episodes,
    "broken_task_count": break_task_count,
    "broken_task_table_absent": break_task_table_absent,
    "broken_task_sentences_not_index": break_task_sentences_not_index,
    "broken_task_index_outside": break_task_index_outside,
    "broken_task_index_beyond_feature": break_task_index_beyond_feature,
    "broken_features_absent": break_features_absent,
    "broken_feature_undeclared_column": break_feature_undeclared_column,
    "broken_feature_column_absent": break_feature_column_absent,
    "broken_feature_shape": break_feature_shape,
    "broken_feature_dtype_name": break_feature_dtype_name,
    "broken_feature_frame_index_width": break_feature_frame_index_width,
    # Copied from clean_media -- or clean_media_pack for a pack's defect -- not clean
    # (variant_base): made only with --with-media.
    "broken_media_frame_count": break_media_frame_count,
    "broken_media_pack_missing": break_media_pack_missing,
    "broken_media_bbox_out_of_range": break_media_bbox_out_of_range,
    "broken_media_stray_pack": break_media_stray_pack,
    "broken_texture_missing": break_texture_missing,
    "broken_texture_bytes": break_texture_bytes,
    "broken_texture_unlisted": break_texture_unlisted,
}


def variant_base(name: str) -> str:
    """The clean fixture a variant is copied from: the camera distribution for broken_media_*,
    and for broken_texture_* too -- it is the one whose episodes use the textured asset. A
    pack's defect (PACK_VARIANTS) needs a canon that declares a pack: clean_media_pack. An hri
    defect (broken_hri_*) needs the hri distribution: clean_hri."""
    if name.startswith(HRI_VARIANT_PREFIX):
        return HRI_CLEAN
    if name in PACK_VARIANTS:
        return MEDIA_PACK_CLEAN
    if name.startswith((MEDIA_VARIANT_PREFIX, TEXTURE_VARIANT_PREFIX)):
        return MEDIA_CLEAN
    return "clean"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="검증기 시험용 합성 픽스처")
    ap.add_argument("out", nargs="?", default="fixtures", type=Path)
    ap.add_argument("--with-media", action="store_true",
                    help=f"카메라 배포본 {MEDIA_CLEAN} · {MEDIA_PACK_CLEAN}(팩 선언) · {HRI_MEDIA_CLEAN}(hri) 과 "
                         f"그것을 복사하는 {MEDIA_VARIANT_PREFIX}* · {TEXTURE_VARIANT_PREFIX}* 변종도 만든다")
    a = ap.parse_args(argv)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)

    bases = {"clean": out / "clean", HRI_CLEAN: out / HRI_CLEAN}
    if a.with_media:
        bases[MEDIA_CLEAN] = out / MEDIA_CLEAN
        bases[MEDIA_PACK_CLEAN] = out / MEDIA_PACK_CLEAN
        bases[HRI_MEDIA_CLEAN] = out / HRI_MEDIA_CLEAN
    for base, path in bases.items():
        build(path, with_media=base in (MEDIA_CLEAN, MEDIA_PACK_CLEAN, HRI_MEDIA_CLEAN),
              with_pack=base == MEDIA_PACK_CLEAN, hri=base in (HRI_CLEAN, HRI_MEDIA_CLEAN))
        print(f"생성: {path}")
    # Clean and copied by no variant: the hri distribution without the optional columns.
    build_hri_lean(out / HRI_LEAN_CLEAN)
    print(f"생성: {out / HRI_LEAN_CLEAN}")

    for name, fn in VARIANTS.items():
        base = variant_base(name)
        if base not in bases:
            # Said out loud: a variant that is silently not built is a check nobody runs.
            print(f"건너뜀: {name} — {base} 가 없음 (--with-media)")
            continue
        d = out / name
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(bases[base], d)
        fn(d)
        print(f"생성: {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
