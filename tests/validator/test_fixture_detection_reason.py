"""결함 픽스처가 **의도한 검사로** 잡히는지.

`make validate` 는 검증기의 exit code 가 0 이 아니면 「검출」로 셉니다. 그러면
검사 코드가 사라지고 대신 관련 없는 크래시가 생겨도 회귀 시험이 초록입니다 —
`determined_by=[]` 로 `TypeError`/exit 3 을 만들어 실제로 확인한 형태입니다.

여기서는 두 가지를 함께 요구합니다.
  1. exit code 가 정확히 1 이어야 한다 (검증기가 결함을 **보고**했다는 뜻).
     3 은 크래시이고, 그건 검출이 아니라 다른 사고입니다
  2. 기대한 검사 코드의 ERROR 가 실제로 나와야 한다
"""

import atexit
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from validator.make_fixture import VARIANTS, camera_fields, media_canon

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / ".tmp" / "fixtures"
VALIDATOR = REPO_ROOT / "validator" / "validate_dataset.py"
# What the two media fixtures break, read from the camera fixture's own canon (media_canon) rather than
# written down: break_media_frame_count shortens the last camera stream's span, and
# break_media_bbox_out_of_range writes x_max = bbox.camera's width.
_MEDIA = media_canon()
_LAST_VIDEO = camera_fields(_MEDIA)[0][-1]
_BOX_WIDTH = _MEDIA["cameras"][_MEDIA["bbox"]["camera"]]["width"]

# CI 는 `.venv` 를 만들지 않고 시스템 파이썬에 pip install 한다. 이 경로를 박아 두면
# 갓 클론한 트리와 CI 에서 FileNotFoundError 가 난다 — 픽스처를 늘 새로 만들므로
# 이 인터프리터가 매번 불린다.
_VENV = REPO_ROOT / ".venv" / "bin" / "python"
PY = _VENV if _VENV.exists() else Path(sys.executable)

# 각 결함이 어느 검사에서 잡혀야 하는가. validator/README 의 표와 같은 내용이며,
# 여기 것이 실행되는 쪽입니다.
EXPECTED_CHECK = {
    "broken_termination_list": "C5", "broken_termination_member": "C5",
    "broken_notes_member_dropped": "C5", "broken_notes_member_type": "C5",
    "broken_canon_notes_members": "C1", "broken_redistributed_type": "C6",
    "broken_canon_unrouted_shape": "C1", "broken_notes_empty_ssm": "C5",
    "broken_notes_ssm_diverges_from_canon": "C5",
    "broken_notes_bad_region": "C5", "broken_asset_missing": "C6",
    "broken_notes_region_detail": "C5", "broken_notes_hold_contaminated": "C5",
    "broken_notes_region_contradiction": "C5", "broken_canon_notes_bounds": "C1",
    "broken_asset_canon_flag": "C6", "broken_notes_rule_id_type": "C5",
    "broken_asset_license_one_episode": "C6", "broken_asset_flag_one_episode": "C6",
    "broken_asset_unknown_path": "C6", "broken_notes_aggregate_without_regions": "C5",
    "broken_asset_unreferenced": "C6", "broken_texture_missing": "C6",
    "broken_texture_bytes": "C6", "broken_texture_unlisted": "C6",
    "broken_asset_left_out": "C6", "broken_asset_usd_bytes": "C6",
    "broken_dof": "C2", "broken_region": "C3", "broken_label": "C4",
    "broken_seed": "C5", "broken_notice": "C6", "broken_time": "C7",
    "broken_values": "C8", "broken_rules": "C1", "broken_contact_type": "C8",
    "broken_inputs_missing": "C1", "broken_rule_inputs": "C8",
    "broken_scenario_rules": "C1", "broken_required_fields": "C8",
    "broken_threshold_nonfinite": "C1", "broken_threshold_negative": "C1",
    "broken_missing_parameter": "C1", "broken_duplicate_rule_id": "C1",
    "broken_empty_rules": "C1", "broken_zmp_half_nan": "C8",
    "broken_com_nan": "C8", "broken_safety_schema": "C1",
    "broken_not_applicable_on_permitted": "C1",
    "broken_not_applicable_with_value": "C1",
    "broken_contact_permitted_dropped": "C1",
    "broken_episode_id_format": "C5", "broken_episode_id_seed": "C5",
    "broken_dof_mismatch": "C8", "broken_required_null": "C5",
    "broken_fps_zero": "C5", "broken_late_joint_width": "C8",
    "broken_canon_version": "C1", "broken_canon_list_as_object": "C1",
    "broken_orphan_frame_episode": ("C7", "C8"), "broken_fps_text": "C5",
    "broken_duplicate_episode_key": "C7", "broken_bad_triggered_rules": "C4",
    "broken_canon_shape_list": "C1", "broken_canon_json_fields": "C1",
    "broken_late_shape": "C8", "broken_action_nan": "C8",
    "broken_canon_shapes": "C1",
    "broken_v_rel_nan": "C8", "broken_v_rel_negative": "C8",
    "broken_v_rel_length": "C3",
    "broken_canon_conditional_steps": "C1",
    "broken_stop_table": "C1", "broken_directed_parameter_missing": "C1",
    "broken_canon_step_layers": "C1",
    # The hri distribution (clean_hri) and a step column without a layer.
    "broken_hri_column_missing": "C8", "broken_hri_violation_gate": "C8",
    "broken_hri_negative_approach": "C8", "broken_hri_visible_fraction": "C8",
    "broken_hri_ssm_params": "C5", "broken_hri_task_step": "C5", "broken_hri_task_null": "C5",
    "broken_hri_task_stages": "C5",
    "broken_hri_contact_without_onset": "C8",
    "broken_hri_spec_missing": "C5", "broken_hri_spec_shape": "C5", "broken_hri_grid_cell": "C5",
    "broken_hri_evidence_text": "C5",
    "broken_layer_missing_column": "C8",
    "broken_media_frame_count": "C7",
    # The listed file is missing (C1) and so the episode's pack alignment cannot be checked
    # (C7) -- a C7 that went quiet on an unreadable pack would read as aligned.
    "broken_media_pack_missing": ("C1", "C7"),
    "broken_media_bbox_out_of_range": "C8",
    # packs/ and meta/packs.json left in a distribution whose canon declares no pack.
    "broken_media_stray_pack": "C1",
    # The run's first seq and the generator's commit.
    "broken_run_start_missing": "C5", "broken_run_start_past_seq": "C5",
    "broken_generator_dirty": "C5", "broken_generator_commit": "C5",
    # A null where the step table holds values: only C7 reads the file through Arrow.
    "broken_step_null": "C7",
    # meta/episodes' pointers to an episode's rows: every row keeps its own episode_index, so
    # only the check that reads the pointers as a loader does (c7_data_spans) sees these.
    "broken_span_swapped": "C7", "broken_span_end_short": "C7",
    "broken_span_file_index": "C7", "broken_span_rows_out_of_place": "C7",
    # info.json's counts of the two tables: the tables and every pointer are as they should be, so
    # only the check that holds the counts to the tables (c7_info_totals) sees these.
    "broken_total_episodes": "C7", "broken_total_frames": "C7",
    # The task of a step: no other check opens meta/tasks.parquet or reads the steps' task_index,
    # so only the check that reads them as a loader does (c7_task_table) sees these.
    "broken_task_count": "C7", "broken_task_table_absent": "C7",
    "broken_task_sentences_not_index": "C7", "broken_task_index_outside": "C7",
    "broken_task_index_beyond_feature": "C7",
    # info.json's features beside the step files: read through pandas the files are whole, so only
    # the check that holds them to the features as a loader does (c7_step_features) sees these.
    "broken_features_absent": "C7", "broken_feature_undeclared_column": "C7",
    "broken_feature_column_absent": "C7", "broken_feature_shape": "C7",
    "broken_feature_dtype_name": "C7", "broken_feature_frame_index_width": "C7",
}


def test_the_expected_check_map_covers_every_variant():
    missing = sorted(set(VARIANTS) - set(EXPECTED_CHECK))
    assert not missing, (
        f"결함 픽스처 {missing} 에 기대 검사가 없습니다 — 어느 검사가 잡아야 하는지 "
        "적지 않으면 아무 이유로 실패해도 통과로 셉니다")
    phantom = sorted(set(EXPECTED_CHECK) - set(VARIANTS))
    assert not phantom, f"만들어지지 않는 픽스처를 기대하고 있습니다: {phantom}"


# 검사 ID 만으로는 부족하다. 한 픽스처가 그 검사에서 **여러 오류**를 내면, 의도한
# 진단을 지워도 부수 오류가 남아 통과한다 — `broken_inputs_missing` 이 실제로
# 그랬다: 「입력이 없어 계산할 수 없는 규칙」을 억제해도 픽스처가 만든 중복
# rule_id 오류가 대신 났다. 그래서 **결함 픽스처마다** 그
# 픽스처가 존재하는 이유인 문장을 못박는다. 값이 튜플이면 전부 나와야 한다.
EXPECTED_PHRASE = {
    "broken_action_nan": "action 에 유한하지 않은 값",
    "broken_canon_notes_members": "규격 정의에 notes_members 가 없음",
    "broken_canon_unrouted_shape": "json_encoded_fields.episode 에",
    "broken_notes_empty_ssm": "notes.ssm_params.human_speed 없음",
    "broken_notes_ssm_diverges_from_canon": "이 정본의 값",
    "broken_notes_bad_region": "없는 부위 코드",
    "broken_notes_region_detail": (".region 이 없는 부위 코드",
                                   ".steps 가 1 이상의 정수가 아님",
                                   ".peak_force_n 이 유한한 양수가 아님"),
    "broken_notes_hold_contaminated": "생성기 계약은",
    "broken_notes_region_contradiction": "contact_observed 가 거짓",
    "broken_canon_notes_bounds": "규격 정의에 notes_bounds 가 없음",
    "broken_asset_canon_flag": "meta/assets.json 은 아니라고 말함",
    "broken_asset_license_one_episode": "라이선스가 정본과 다름",
    "broken_asset_flag_one_episode": "재배포 여부가 정본과 다름",
    "broken_asset_unknown_path": "meta/assets.json 에 없음",
    "broken_notes_aggregate_without_regions": "집계 항목인데 regions 가 비어 있음",
    "broken_notes_rule_id_type": (".rule_id 이 string 이 아님", ".reason 이 string 이 아님"),
    "broken_asset_missing": "재배포한다고 선언한 자산이 배포본에 없음",
    "broken_asset_unreferenced": ("이 배치의 에피소드가 쓰지 않는 파일: "
                                  "assets/scene_textured.usd"),
    "broken_texture_missing": "이 쓰는 텍스처가 배포본에 없음: assets/textures/floor_normal.png",
    "broken_texture_bytes": ("assets/textures/floor_albedo.png 의 SHA-256 이 "
                             "meta/assets.json(textured bundle.files)과 다름"),
    "broken_texture_unlisted": "assets/textures/stray_decal.png 의 출처 기록이 없음",
    "broken_asset_left_out": ("scene_assets 에 ['keti/dummy_ts15066_v1.usd'] 가 없음",
                              "재배포한다고 선언한 자산이 배포본에 없음: assets/dummy_ts15066_v1.usd"),
    "broken_asset_usd_bytes": ("assets/scene_flat_primitive.usd 의 SHA-256 이 meta/assets.json 의 "
                               "usd_sha256 과 다름"),
    "broken_notes_member_dropped": "notes.ssm_params 없음",
    "broken_notes_member_type": "notes.human_region_radius 이 number 이 아님",
    "broken_redistributed_type": "redistributed 가 참·거짓이 아님",
    "broken_termination_list": "termination 는 object 여야 하는데 list 임",
    "broken_termination_member": ("terminated 가 참·거짓이 아님",
                                  "truncated 가 참·거짓이 아님",
                                  "reason 이 문자열 목록이 아님"),
    "broken_bad_triggered_rules": "triggered_rules 가 목록이 아님",
    "broken_canon_json_fields": "json_encoded_fields.step 가 실재하지 않는 필드를 가리킴",
    "broken_canon_conditional_steps": "conditional_required_step_fields 가 검증기와 다름",
    "broken_stop_table": "R-TS15066-SSM-02.stop_table.walking.carry_3kg",
    "broken_directed_parameter_missing": "R-TS15066-SSM-02 에 필수 파라미터 ['approach_speed_threshold'] 가 없음",
    "broken_canon_step_layers": "step_field_layers 가 검증기와 다름",
    "broken_hri_column_missing": "조건부 필수 스텝 필드 없음: safety.approach_speed_robot",
    "broken_hri_violation_gate": "approach_speed_robot > 0.05) 와 불일치",
    "broken_hri_negative_approach": "safety.approach_speed_human 에 음수",
    "broken_hri_visible_fraction": "observation.human_visible_fraction.head_rgb 이 0~1 밖",
    "broken_hri_ssm_params": "notes.ssm_params.stop_table 이 정본의 R-TS15066-SSM-02.stop_table 과 다름",
    "broken_hri_task_step": "notes.task_completed_step 이 에피소드 길이(400) 안의 스텝이 아님",
    "broken_hri_task_null": "notes.task_completed_step 이 null 이거나 없음",
    "broken_hri_task_stages": "notes.task_stages[2] 의 completed(True) 가 end_step(-1) 과 어긋남",
    "broken_hri_contact_without_onset": "접촉 유형이 매겨진 스텝보다 앞선 contact_onset 이 없는 에피소드 1개",
    "broken_hri_spec_missing": "조건표가 있는 시나리오의 에피소드 열 없음: episode_spec",
    "broken_hri_spec_shape": "episode_spec 의 멤버가 규격의",
    "broken_hri_grid_cell": "무작위 행(condition_design=random)인데 격자 칸이",
    "broken_hri_evidence_text": "근거 요약 열(evidence.*)의 값이 수나 null 이 아님",
    "broken_layer_missing_column": "층 표식이 없는 스텝 열: ['observation.extra_signal']",
    "broken_media_frame_count": (f"{_LAST_VIDEO} 의 영상 구간",
                                 "399.000 프레임인데 이 에피소드의 스텝은 400개"),
    "broken_media_pack_missing": (
        "meta/packs.json 이 가리키는 팩이 배포본에 없음: packs/head/episode_000003.npz",
        "팩 packs/head/episode_000003.npz 이 배포본에 없어 스텝 정렬을 대조하지 못함"),
    "broken_media_bbox_out_of_range": ("observation.bbox2d.human 가 영상 범위 밖이거나 뒤집힌 행 1개 "
                                       f"(예: [1, 1, {_BOX_WIDTH}, 2])"),
    "broken_media_stray_pack": "정본이 팩을 선언하지 않았는데 팩이 있음: ['packs/', 'meta/packs.json']",
    "broken_canon_list_as_object": "contact_types 가 목록이 아님",
    "broken_canon_shape_list": "fixed_shapes 가 객체가 아님",
    "broken_canon_shapes": "fixed_shapes 길이가 검증기와 다름",
    "broken_canon_version": "규격 정의의 schema_version",
    "broken_com_nan": "무게중심에는 표식이 없으며 NaN 은 입력 손상을 뜻한다",
    "broken_contact_permitted_dropped": "contact_permitted 없음",
    "broken_contact_type": "허용되지 않은 접촉 유형",
    "broken_dof": "len(joint_names)=19 != dof=21",
    "broken_dof_mismatch": ("observation.state.joint_pos 차원",
                            "observation.state.joint_vel 차원",
                            "observation.state.joint_torque 차원"),
    "broken_duplicate_episode_key": "중복된 조인 키",
    "broken_duplicate_rule_id": "중복된 rule_id",
    "broken_empty_rules": "규칙 id 를 하나도 읽지 못함",
    "broken_episode_id_format": "episode_id 형식이 규격서 §3 과 다름",
    "broken_episode_id_seed": "randomization_seed=2026090100002 와 다름",
    "broken_fps_text": "fps 가 수치가 아님",
    "broken_fps_zero": "fps 이상",
    "broken_inputs_missing": "입력이 없어 계산할 수 없는 규칙",
    "broken_label": "outcome=fail 인데 판정 근거 없음",
    "broken_late_joint_width": "action 폭이 에피소드 안에서 일정하지 않음",
    "broken_late_shape": "observation.root_pose 길이가 규격 7 과 다른 행",
    "broken_missing_parameter": "필수 파라미터 ['robot_stop_time'] 가 없음",
    "broken_not_applicable_on_permitted": "접촉이 허용된 부위인데 not_applicable",
    "broken_not_applicable_with_value": "not_applicable 인데 값이 있음",
    "broken_notice": "재배포 자산이 NOTICE 에 없음",
    "broken_orphan_frame_episode": "메타에 없는 프레임 에피소드",
    "broken_region": "허용되지 않은 부위 코드",
    "broken_required_fields": "safety.absent_field",
    "broken_required_null": "필수 에피소드 필드 hazard_class 가 1건에서 비어 있음",
    "broken_rule_inputs": "observation.object.grasped",
    "broken_rules": "값도 쓸 수 있는 standard_ref 도 없음",
    "broken_safety_schema": "body_regions 가 검증기와 다름",
    "broken_scenario_rules": "rules.json 에 없는 규칙을 가리킴",
    "broken_seed": "randomization_seed 없음",
    "broken_threshold_negative": "임계값이 음수",
    "broken_threshold_nonfinite": "값이 유한하지 않음: nan",
    "broken_time": "frame_index 불연속",
    "broken_v_rel_length": "v_rel 길이 11 != region_names 길이 12",
    "broken_v_rel_nan": "observation.contact.v_rel 에 NaN",
    "broken_v_rel_negative": "observation.contact.v_rel 에 음수 존재",
    "broken_values": ("observation.contact.force 에 음수 존재",
                      "safety.stability_margin 에 +inf 존재",
                      "observation.contact.force 에 NaN"),
    "broken_zmp_half_nan": "한쪽 성분만 NaN 인 스텝",
    "broken_run_start_missing": "source=sim 인데 run_start_seq 없음",
    "broken_run_start_past_seq": "run_start_seq=7 가 0 과 자기 번호 2 사이가 아님",
    "broken_generator_dirty": "sim_backend.generator_dirty 가 false 가 아님: True",
    "broken_generator_commit": "sim_backend.generator_commit 이 40자 16진 커밋 해시가 아님: '9ab48ae'",
    "broken_step_null": "safety.zmp 에 null 2개",
    "broken_span_swapped": "가리키는 행이 이 에피소드의 것이 아님",
    "broken_span_end_short": ("length 가 가리키는 구간의 행 수와 다름",
                              "어느 에피소드도 가리키지 않는 스텝 행 1개"),
    "broken_span_file_index": "행이 data/chunk_index·data/file_index 가 가리키는 파일에 있지 않음",
    "broken_span_rows_out_of_place": ("index 열이 행의 자리와 다름 — 0번째 행이 400",
                                      "가리키는 행이 이 에피소드의 것이 아님"),
    "broken_total_episodes": "info.json 의 total_episodes 가 에피소드 표의 행 수와 다름",
    "broken_total_frames": "info.json 의 total_frames 가 스텝 표의 행 수와 다름",
    "broken_task_count": "info.json 의 total_tasks 가 과제 표의 행 수와 다름",
    "broken_task_table_absent": "LeRobot 이 읽는 과제 표가 없음",
    "broken_task_sentences_not_index": "과제 표의 인덱스가 과제 문장이 아님",
    "broken_task_index_outside": "스텝의 task_index 가 과제 표의 행이 아님",
    "broken_task_index_beyond_feature": ("스텝 표의 열을 info.json 의 features 가 선언한 형으로 바꿀 수 없음",
                                         "data/chunk-000/file-001.parquet · task_index: 저장 int64, 선언 int8 [1]"),
    "broken_features_absent": "info.json 에 features 가 없거나 JSON 객체가 아님",
    "broken_feature_undeclared_column": "스텝 표에 info.json 의 features 가 스텝 표의 열로 선언하지 않은 열이 있음",
    "broken_feature_column_absent": "info.json 의 features 가 선언한 열이 스텝 표에 없음",
    "broken_feature_shape": ("스텝 표의 열을 info.json 의 features 가 선언한 형으로 바꿀 수 없음",
                             "safety.com: 저장 list<element: double>, 선언 float64 [2]"),
    "broken_feature_dtype_name": "info.json 의 features 에 LeRobot 이 스텝 표의 열로 만들지 못하는 선언이 있음",
    "broken_feature_frame_index_width": ("스텝 표의 열을 info.json 의 features 가 선언한 형으로 바꿀 수 없음",
                                         "frame_index: 저장 int64, 선언 int8 [1]"),
}


def test_every_variant_has_an_expected_phrase():
    """검사 ID 만 적힌 픽스처는 부수 오류로 통과할 수 있습니다.

    문장을 못박기 전에는 픽스처 다섯이 실제로 주 진단을
    억제해도 초록이었습니다.
    """
    missing = sorted(set(VARIANTS) - set(EXPECTED_PHRASE))
    assert not missing, (
        f"결함 픽스처 {missing} 에 기대 문장이 없습니다 — 검사 ID 만으로는 그 픽스처가 "
        "존재하는 이유가 실제로 잡혔는지 알 수 없습니다")
    phantom = sorted(set(EXPECTED_PHRASE) - set(VARIANTS))
    assert not phantom, f"만들어지지 않는 픽스처를 기대하고 있습니다: {phantom}"


_BUILT: list[Path] = []


def ensure_fixture(variant: str) -> Path:
    """이 pytest 세션에서 **한 번은 반드시** 생성기를 돌린 픽스처를 돌려준다.

    `.tmp` 는 추적되지 않으므로, 이미 있는 픽스처만 쓰면 갓 클론한 트리에서는 검출
    시험이 전부 skip 이고 `make test` 가 검출 로직을 **한 줄도 돌리지 않고** 초록이다.

    「없으면 만든다」도 모자라다 — 디렉터리가 있으면
    생성기를 부르지 않으므로, `break_*` 가 사라지거나 픽스처 표현이 배포본과
    달라져도 **지난 실행이 남긴 픽스처**로 초록이 된다. 그래서
    세션마다 저장소 밖 임시 위치에 새로 만든다. 공용 `.tmp/fixtures` 를 시험
    입력 캐시로 재사용하지 않는다.
    """
    if not _BUILT:
        root = Path(tempfile.mkdtemp(prefix="keti-fixtures-"))
        # --with-media: the broken_media_* variants copy a camera distribution (clean_media, or
        # clean_media_pack for a pack's defect), which the generator builds only when asked (make
        # fixture asks too).
        r = subprocess.run([str(PY), "validator/make_fixture.py", "--with-media", str(root)],
                           capture_output=True, text=True, cwd=REPO_ROOT)
        assert r.returncode == 0, f"픽스처 생성 실패:\n{r.stdout[-800:]}{r.stderr[-800:]}"
        _BUILT.append(root)
        atexit.register(shutil.rmtree, root, True)
    path = _BUILT[0] / variant
    assert path.exists(), f"{variant} 픽스처가 만들어지지 않았습니다"
    return path


@pytest.mark.parametrize("variant", sorted(EXPECTED_CHECK))
def test_variant_is_reported_by_the_intended_check(variant):
    root = ensure_fixture(variant)
    r = subprocess.run([str(PY), str(VALIDATOR), str(root)],
                       capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 1, (
        f"{variant}: exit {r.returncode} — 1 이어야 합니다. 3 은 검증기가 죽은 것이고 "
        f"그건 결함 검출이 아닙니다.\n{r.stdout[-600:]}")
    # 하나의 픽스처가 여러 검사를 물어야 하는 경우가 있다. 하나만 기대하면 나머지
    # 검사를 지워도 초록이다 — orphan 픽스처가 C7 만 기대해 C8 쪽 조인 검사가
    # 보호되지 않고 있었다.
    want = EXPECTED_CHECK[variant]
    wanted = (want,) if isinstance(want, str) else tuple(want)
    # 어떤 픽스처는 의도한 진단 말고 다른 오류도 함께 낸다. 검사 ID 만 요구하면
    # 정작 이름 붙인 진단을 지워도 초록이다 — 두 픽스처가 그랬다.
    # 그런 것은 메시지 조각까지 요구한다.
    phrase = EXPECTED_PHRASE.get(variant)
    errors = [ln for ln in r.stdout.splitlines() if re.match(r"^\s*ERROR\s+C\d", ln)]
    for w in wanted:
        assert any(re.match(rf"^\s*ERROR\s+{w}\b", ln) for ln in errors), (
            f"{variant}: {w} 의 ERROR 가 없습니다 — 다른 이유로 실패하고 있습니다.\n"
            + "\n".join(errors[:4]))
    # 여러 결함을 한 픽스처에 심었으면 전부 요구한다. 하나만 요구하면 나머지를
    # 못 잡게 되어도 초록이다.
    phrases = (phrase,) if isinstance(phrase, str) else tuple(phrase or ())
    for want_phrase in phrases:
        assert any(want_phrase in ln for ln in errors), (
            f"{variant}: 의도한 진단(\"{want_phrase}\")이 없습니다 — 다른 오류가 이 "
            f"픽스처를 대신 실패시키고 있어, 그 진단을 지워도 초록입니다.\n"
            + "\n".join(errors[:4]))


def test_the_clean_fixture_mirrors_the_canon_on_standard_thresholds():
    """픽스처는 배포본과 **같은 표현**이어야 하고, 단언은 정본에서 유도합니다.

    「값이 하나도 없어야 한다」고 적어 두면 배포본이 조항 참조만 싣는 동안에만
    맞습니다. 이용 허락을 받아 수치를 싣는 지금은(2026-08-27 부터) 그런 단언이
    **배포본과 정반대**입니다. 정본에서 유도하면
    방침이 다시 바뀌어도 이 시험이 따라옵니다.

    막으려는 실패 양식은 이것입니다: 합성값으로 표준 임계값을 채우면 픽스처가
    배포본에 없는 형태를 시험하게 되고, 그 경로의 결함이 초록으로 남습니다.
    이 시험이 없으면, 픽스처 생성기에서 표준 항목을 건너뛰는 블록을 지워 72건이
    합성값으로 채워져도 `make check` 가 초록입니다.
    """
    import json

    from validator.validate_dataset import iter_threshold_entries

    canon = json.loads((REPO_ROOT / "meta" / "rules.json").read_text(encoding="utf-8"))
    fixture = json.loads(
        (ensure_fixture("clean") / "meta" / "rules.json").read_text(encoding="utf-8"))

    # `transient_multiplier` 는 limits·biomechanical 밖에 홀로 있어서, 두 그룹만
    # 훑으면 12칸을 통째로 보지 않습니다 — BR-05 배수를 999 로 바꾼
    # 반례가 통과합니다. 정본이 값을 두는 자리를
    # 나열하는 것이므로 새 자리가 생기면 여기에 더해야 합니다.
    def entries(body):
        for group in ("limits", "biomechanical"):
            for name, e in (body.get(group) or {}).items():
                yield (group, name), e, (fixture["contact_limits"][br][group] or {}).get(name)
        yield (("", "transient_multiplier"), body.get("transient_multiplier"),
               fixture["contact_limits"][br].get("transient_multiplier"))

    wrong = []
    for br, body in canon["contact_limits"].items():
        for (group, name), e, f in entries(body):
            if not isinstance(e, dict) or e.get("determined_by") != "standard":
                continue
            where = f"{br}.{group}.{name}" if group else f"{br}.{name}"
            if not isinstance(f, dict):
                wrong.append(f"{where}: 픽스처에 항목이 없음")
                continue
            if f.get("value") != e.get("value"):
                wrong.append(f"{where}: 정본 {e.get('value')!r} 픽스처 {f.get('value')!r}")
            # 값만 보면 참조가 썩은 픽스처로 참조 경로를 시험하게 됩니다.
            if f.get("standard_ref") != e.get("standard_ref"):
                wrong.append(f"{where} 참조가 다름")
            if bool(f.get("not_applicable")) != bool(e.get("not_applicable")):
                wrong.append(f"{where} not_applicable 이 다름")
    assert not wrong, (
        f"픽스처의 표준 임계값이 정본과 다릅니다 ({len(wrong)}건): {wrong[:4]} — "
        "픽스처가 배포본과 다른 표현을 쓰면 그 경로의 결함이 초록으로 남습니다")

    # 조항 참조만 실린 형태가 픽스처에서 사라지면 그 표현을 도는 검사가 아무 시험의
    # 보호도 받지 않습니다. 지금 그 자리를 지키는 것은 `intrusion_distance`(KS B ISO 13855)
    # 한 항목이고, 표준이 그 값을 상수로 주지 않아 조항 참조만 실려 있습니다.
    cite_only = [path for path, e in iter_threshold_entries(fixture)
                 if isinstance(e, dict) and e.get("determined_by") == "standard"
                 and e.get("value") is None and not e.get("not_applicable")]
    assert cite_only, (
        "픽스처에 조항 참조만 실린 임계값이 하나도 없습니다 — 그 표현을 도는 검사가 "
        "시험 없이 남습니다. 배포본이 정말 전부 값을 싣게 되었다면 이 단언을 지우되, "
        "그 경로를 시험하는 자리를 먼저 만드십시오")


def test_the_fixture_is_split_across_files_like_a_real_distribution():
    """한 파일짜리 픽스처로는 파일 경계를 넘는 검사를 영원히 시험하지 못합니다.

    실제로 그 자리에서 결함 둘이 나왔습니다 — 파일별로 전체 메타와 대조해 거짓
    오류 138건, 그리고 첫 파일에 키가 없으면 뒤 파일 검사가 통째로 꺼지던 것.
    """
    root = ensure_fixture("clean")
    files = sorted((root / "data" / "chunk-000").glob("*.parquet"))
    assert len(files) >= 2, (
        f"픽스처가 파일 {len(files)}개뿐입니다 — 배포본은 여러 파일로 쪼개지므로 "
        "파일 경계 검사가 시험되지 않습니다")


def test_json_fields_are_encoded_exactly_once(tmp_path):
    """픽스처가 배포본보다 한 겹 더 감싸면 검증기의 JSON 처리를 시험하지 못합니다.

    `termination` 이 정확히 그렇게 들어갔습니다 — 손으로 `json.dumps` 한 값을
    정본 루프가 다시 직렬화했습니다. 한 겹 벗겨서
    문자열이 또 나오면 배포본에 없는 표현입니다. 필드마다 따로 못박지 않고
    정본 목록을 순회해, 다음에 필드를 늘려도 같은 실수를 잡습니다.
    """
    import json

    import pandas as pd

    from validator.make_fixture import build, json_encoded_fields

    root = tmp_path / "clean"
    build(root)
    table = pd.concat([pd.read_parquet(p)
                       for p in sorted((root / "meta" / "episodes").rglob("*.parquet"))])

    for field in json_encoded_fields()["episode"]:
        if field not in table.columns:
            continue
        cell = table[field].iloc[0]
        assert isinstance(cell, str), f"{field} 가 JSON 문자열이 아닙니다: {type(cell)}"
        assert not isinstance(json.loads(cell), str), (
            f"{field} 가 두 번 직렬화됐습니다 — 배포본은 한 번만 감쌉니다")


def _step_nulls(root: Path) -> dict[str, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    def inner(arr):
        t = arr.type
        if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
            flat = arr.flatten()
            return flat.null_count + inner(flat)
        return 0

    out: dict[str, int] = {}
    for f in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(f)
        for name in table.column_names:
            col = table.column(name)
            n = col.null_count + sum(inner(c) for c in col.chunks)
            if n:
                out[name] = out.get(name, 0) + n
    return out


def test_only_the_null_fixture_holds_a_null():
    """픽스처는 배포본과 같은 표현이어야 한다. 「NaN 을 심는」 변종 다섯이 pandas 로 다시 쓰면서
    null 을 심고 있었다(2026-10-02 실측) — 그러면 NaN 검사를 시험하는 픽스처가 null 검사로도
    잡혀, NaN 검사를 지워도 초록이다."""
    holding = {name: nulls for name in sorted(VARIANTS) + ["clean", "clean_media", "clean_media_pack", "clean_hri",
                                                           "clean_hri_media", "clean_hri_lean"]
               if (nulls := _step_nulls(ensure_fixture(name)))}
    assert holding == {"broken_step_null": {"safety.zmp": 2}}, holding
