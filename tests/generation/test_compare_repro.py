"""재현 대조 로직. Isaac 도 lerobot 도 없이 배열만 비교한다."""

import numpy as np
import pytest

from generation.compare_repro import (comparable_step_fields, compare_arrays,
                                      episode_scenario_of, episode_seq_of)

MALFORMED_IDS = ("S1a-broken", "S1a_7_seed1", "", "S1a_0007")


def test_episode_seq_is_read_from_the_id_not_guessed():
    assert episode_seq_of("S1a_0007_seed1727147689") == 7


def test_a_malformed_episode_id_raises_rather_than_defaulting_to_zero():
    # 0 으로 떨어지면 훅이 늘 첫 에피소드만 대조하고 늘 통과한다.
    for bad in MALFORMED_IDS:
        with pytest.raises(ValueError):
            episode_seq_of(bad)


def test_episode_scenario_is_read_from_the_id_not_defaulted():
    # 훅이 이 값으로 run_s1a.py --scenario 를 준다. 기본값(S1a)으로 떨어지면 S1d 배포본의
    # C9 는 S1a 를 다시 돌려 「다른 에피소드」만 보고한다.
    assert episode_scenario_of("S1d_0003_seed1504267427") == "S1d"
    assert episode_scenario_of("S1a_0007_seed1727147689") == "S1a"


def test_a_malformed_episode_id_has_no_scenario_either():
    for bad in MALFORMED_IDS:
        with pytest.raises(ValueError):
            episode_scenario_of(bad)


def test_identical_arrays_compare_equal():
    a = {"joint_pos": np.zeros((3, 2)), "com": np.ones((3, 3))}
    assert compare_arrays(a, dict(a), [("x", "joint_pos"), ("y", "com")]) == []


def test_a_single_differing_element_is_reported_with_its_field():
    a = {"joint_pos": np.zeros((3, 2))}
    b = {"joint_pos": np.zeros((3, 2))}
    b["joint_pos"][2, 1] = 1e-9
    out = compare_arrays(a, b, [("observation.state.joint_pos", "joint_pos")])
    assert len(out) == 1 and "observation.state.joint_pos" in out[0]


def test_a_missing_array_is_a_mismatch_not_a_skip():
    out = compare_arrays({"com": np.zeros(3)}, {}, [("safety.com", "com")])
    assert len(out) == 1


def test_a_shape_difference_is_reported_before_values():
    out = compare_arrays({"com": np.zeros((3, 3))}, {"com": np.zeros((2, 3))},
                         [("safety.com", "com")])
    assert len(out) == 1 and "shape" in out[0]


def test_nan_matches_nan_in_the_same_place():
    # zmp 는 비행 구간에서 nan 이다. nan != nan 으로 두면 훅이 늘 실패한다.
    a = {"zmp": np.array([[np.nan, 1.0], [2.0, 3.0]])}
    assert compare_arrays(a, {"zmp": a["zmp"].copy()}, [("safety.zmp", "zmp")]) == []


def test_nan_in_only_one_side_is_still_a_mismatch():
    a = {"zmp": np.array([[np.nan, 1.0]])}
    b = {"zmp": np.array([[0.0, 1.0]])}
    assert len(compare_arrays(a, b, [("safety.zmp", "zmp")])) == 1


def test_string_arrays_compare_too():
    a = {"contact_type": np.array([["none", "transient"]])}
    assert compare_arrays(a, {"contact_type": a["contact_type"].copy()},
                          [("observation.contact.type", "contact_type")]) == []


def test_timestamp_is_excluded_from_the_comparison():
    # LeRobot 이 frame_index/fps 로 직접 만들고 float32 로 저장한다. 대조하면
    # 물리 재현이 아니라 LeRobot 의 산술을 재는 것이 되고, 정밀도 때문에 늘 깨진다.
    fields = dict(comparable_step_fields())
    assert "timestamp" not in fields
    assert "observation.state.joint_pos" in fields


@pytest.mark.parametrize("active", [(), ("cameras",)], ids=["none", "cameras"])
def test_the_comparison_covers_every_recorded_array_except_the_excluded(active):
    from generation.field_map import npz_backed_step_fields
    assert len(comparable_step_fields(active)) == len(npz_backed_step_fields(active)) - 1


def test_the_exclusion_list_names_only_fields_the_map_has():
    from generation.compare_repro import EXCLUDED_FROM_COMPARISON
    from generation.field_map import npz_backed_step_fields
    mapped = {s for s, _ in npz_backed_step_fields()}
    assert set(EXCLUDED_FROM_COMPARISON) <= mapped, "존재하지 않는 필드를 빼고 있음"


def test_the_compared_set_is_every_schema_step_field_the_converter_copies_across():
    # comparable_step_fields() 와 그것을 쓰는 픽스처는 같은 목록을 읽으므로, 목록에서
    # 필드 하나가 빠져도 둘이 함께 줄어 초록이다 (action 을
    # 빼도 대조 시험 4개 통과). 그래서 여기서는 정본(검증기의 필수 · 선택 필드)에서 유도한다.
    # 빼는 넷은 구조적이다 — timestamp 는 LeRobot 이 다시 만들고, frame_index 와
    # safety.event 는 변환기가 유도하며, region_names 는 사이드카에서 온다. 선택 필드 가운데
    # 접촉 압력은 시뮬레이터가 기록하지 않는다(면적을 재지 않음).
    import validator.validate_dataset as v
    not_copied = {"timestamp", "frame_index", "safety.event", "observation.contact.region_names"}
    recorded = set(v.REQUIRED_STEP_FIELDS) | (set(v.OPTIONAL_STEP_FIELDS) - {"observation.contact.pressure"})
    expected = (recorded | {"observation.contact.type"}) - not_copied
    assert {s for s, _ in comparable_step_fields()} == expected


def test_with_cameras_the_comparison_adds_the_box_and_the_poses_but_no_video():
    # Derived from the validator's canon constant, not from the map under test. The
    # observation.images.* fields are LeRobot video features, not columns, and RGB is not
    # bit-reproducible across runs (measured) -- the camera arrays C9 can compare are
    # the parquet columns of the camera condition.
    import validator.validate_dataset as v
    columns = {f for f in v.CONDITIONAL_REQUIRED_STEP_FIELDS["cameras"]
               if not f.startswith("observation.images.")}
    added = ({s for s, _ in comparable_step_fields(("cameras",))}
             - {s for s, _ in comparable_step_fields()})
    assert added == columns


def test_the_hri_condition_compares_its_arrays_and_only_under_it():
    from generation.field_map import HRI, STEP_FIELDS
    hri = {(f.schema, f.npz) for f in STEP_FIELDS if f.condition == HRI and f.kind == "array"}
    assert hri <= set(comparable_step_fields((HRI,)))
    assert not hri & set(comparable_step_fields())
