"""라벨링 규칙. Isaac 없이 합성 시계열로만 돈다."""

import numpy as np
import pytest

from generation.labeling.label_episodes import label_episode

EVENT_NAMES = ["ssm_violation", "protective_stop_triggered", "contact_onset"]


REGION_NAMES = ["BR-01", "BR-05"]      # 접촉 금지 부위 하나 + 허용 부위 하나


def _limit(value, unit="N", **extra):
    e = {"value": value, "unit": unit, "determined_by": "standard",
         "standard_ref": "KS B ISO TS 15066:2016 표 A.2"}
    e.update(extra)
    return e


def _rules(max_stop_time=1.5, stop_speed=0.15, force_br05=140, filled=True):
    """정본과 같은 모양의 contact_limits. 값만 합성이다.

    모양이 다르면 A2 분기가 시험에서만 다른 경로를 타고, 배포본에서 깨져도 초록으로
    남는다 — 배포본은 값 없이 조항 참조만 싣고 라벨러가 internal 사본을 병합한다.
    """
    def region(force, permitted, mult):
        r = {
            "name_ko": "합성", "name_en": "synthetic",
            "contact_permitted": permitted,
            "limits": {
                "force_quasistatic": _limit(force if filled else None),
                "pressure_quasistatic": _limit(None, "N/cm^2"),
                "energy_transient": _limit(None, "J"),
            },
            "biomechanical": {
                "effective_mass": _limit(None, "kg"),
                "spring_constant": _limit(None, "N/mm"),
            },
        }
        if permitted:
            r["transient_multiplier"] = _limit(mult if filled else None, "1")
        else:
            r["contact_prohibited_ref"] = "5.5.5.3"
            r["transient_multiplier"] = _limit(None, "1", not_applicable=True,
                                               reason="접촉이 허용되지 않는다")
        return r

    return {
        "rules": [
            {"rule_id": "R-TS15066-SSM-01", "outcome": "near_miss",
             "standard": "KS B ISO/TS 15066 (SSM)", "parameters": {}},
            {"rule_id": "R-10218-STOP-01", "outcome": "fail",
             "standard": "KS B ISO 10218-1",
             "parameters": {
                 "max_stop_time": {"value": max_stop_time, "unit": "s",
                                   "determined_by": "project"},
                 "stop_speed_threshold": {"value": stop_speed, "unit": "m/s",
                                          "determined_by": "project"},
             }},
            {"rule_id": "R-TS15066-A2-01", "outcome": "fail", "body_region": "BR-01",
             "standard": "KS B ISO TS 15066 표 A.2", "parameters": {},
             "limits_ref": "contact_limits.BR-01"},
            {"rule_id": "R-TS15066-A2-05", "outcome": "fail", "body_region": "BR-05",
             "standard": "KS B ISO TS 15066 표 A.2", "parameters": {},
             "limits_ref": "contact_limits.BR-05"},
        ],
        "contact_limits": {
            "BR-01": region(130, permitted=False, mult=None),
            "BR-05": region(force_br05, permitted=True, mult=2),
        },
        "rule_precedence": ["1. fail 이 near_miss 보다 우선한다."],
    }


def _scenario(rules=("R-TS15066-A2-*", "R-TS15066-SSM-01", "R-10218-STOP-01")):
    return {"scenario_id": "S1a", "rules": list(rules)}


def _episode(n=100, violation=None, stop_at=None, contact_at=None, vx=None,
             touch=None):
    """touch: (step, region_code, force, contact_type) — 유형이 매겨진 접촉 하나."""
    events = np.zeros((n, 3), dtype=bool)
    if violation is not None:
        events[violation, 0] = True
    if stop_at is not None:
        events[stop_at, 1] = True
    if contact_at is not None:
        events[contact_at, 2] = True
    ncol = len(REGION_NAMES)
    arrays = {
        "safety_event": events,
        "root_vel": np.zeros((n, 6)),
        "timestamp": np.arange(n) * 0.02,
        "separation_violation": events[:, 0].copy(),
        "contact_force": np.zeros((n, ncol)),
        "contact_type": np.full((n, ncol), "none", dtype="<U11"),
    }
    if touch is not None:
        step, region, force, ctype = touch
        col = REGION_NAMES.index(region)
        arrays["contact_force"][step, col] = force
        arrays["contact_type"][step, col] = ctype
        events[step, 2] = True
    arrays["root_vel"][:, 0] = 1.0 if vx is None else vx
    # 홀드 해제 스텝. 0 은 「처음부터 자유 물체」이며, 접촉 판정이 전 스텝을 본다.
    # 이 필드가 없으면 라벨러가 거부한다 — 없는 채로 두면 접촉 규칙이 하나도 걸리지
    # 않고 조용히 라벨이 달라지기 때문이다.
    meta = {"safety_event_names": EVENT_NAMES, "steps": n, "scenario_id": "S1a",
            "contact_region_names": list(REGION_NAMES), "hold_release_step": 0}
    return arrays, meta


def test_no_rule_fires_is_nominal_and_s0():
    arrays, meta = _episode(vx=np.zeros(100))
    out = label_episode(arrays, meta, _rules(), _scenario())
    assert out["outcome"] == "nominal"
    assert out["triggered_rules"] == []
    assert out["severity"] == "S0"


def test_separation_violation_without_contact_is_near_miss_s1():
    arrays, meta = _episode(violation=slice(10, 20), vx=np.zeros(100))
    out = label_episode(arrays, meta, _rules(), _scenario())
    assert out["outcome"] == "near_miss"
    assert out["triggered_rules"] == ["R-TS15066-SSM-01"]
    assert out["severity"] == "S1"


def test_separation_violation_with_contact_is_not_a_near_miss():
    # 정본의 near_miss 조건은 "접촉이 발생하지 않음" 을 명시한다.
    arrays, meta = _episode(violation=slice(10, 20), contact_at=15, vx=np.zeros(100))
    out = label_episode(arrays, meta, _rules(), _scenario())
    assert "R-TS15066-SSM-01" not in out["triggered_rules"]
    assert out["outcome"] == "nominal"


def test_a_stop_that_never_slows_is_a_fail_s2():
    arrays, meta = _episode(stop_at=10, vx=np.ones(100))
    out = label_episode(arrays, meta, _rules(), _scenario())
    assert out["outcome"] == "fail"
    assert out["triggered_rules"] == ["R-10218-STOP-01"]
    assert out["severity"] == "S2"


def test_a_stop_within_the_budget_does_not_fire():
    vx = np.ones(100)
    vx[20:] = 0.0            # 트리거(10) 후 0.2 s
    arrays, meta = _episode(stop_at=10, vx=vx)
    out = label_episode(arrays, meta, _rules(max_stop_time=1.5), _scenario())
    assert out["triggered_rules"] == []


def test_a_stop_just_past_the_budget_fires():
    vx = np.ones(100)
    vx[10 + 16:] = 0.0       # 트리거(10) 후 0.32 s > 0.31 s
    arrays, meta = _episode(stop_at=10, vx=vx)
    out = label_episode(arrays, meta, _rules(max_stop_time=0.31), _scenario())
    assert out["triggered_rules"] == ["R-10218-STOP-01"]


def test_the_threshold_is_a_speed_not_a_signed_velocity():
    # 뒤로 1 m/s 로 밀려나는 것은 정지가 아니다.
    arrays, meta = _episode(stop_at=10, vx=-np.ones(100))
    out = label_episode(arrays, meta, _rules(), _scenario())
    assert out["triggered_rules"] == ["R-10218-STOP-01"]


def test_a_fail_with_contact_is_s3_not_s2():
    arrays, meta = _episode(stop_at=10, contact_at=40, vx=np.ones(100))
    out = label_episode(arrays, meta, _rules(), _scenario())
    assert out["outcome"] == "fail"
    assert out["severity"] == "S3"


def test_fail_wins_over_near_miss():
    arrays, meta = _episode(violation=slice(10, 20), stop_at=10, vx=np.ones(100))
    out = label_episode(arrays, meta, _rules(), _scenario())
    assert out["outcome"] == "fail"
    assert set(out["triggered_rules"]) == {"R-10218-STOP-01", "R-TS15066-SSM-01"}


def test_a_rule_with_a_null_threshold_is_reported_not_skipped():
    # "성립하지 않았다" 와 "평가하지 못했다" 는 다른 사실이다. 배포본만 가진 소비자가
    # 정확히 이 상태다 — 조항 참조는 있고 값은 internal 사본에만 있다.
    arrays, meta = _episode(contact_at=40, vx=np.zeros(100))
    out = label_episode(arrays, meta, _rules(filled=False), _scenario())
    ids = [n["rule_id"] for n in out["not_evaluated"]]
    assert "R-TS15066-A2-01" in ids
    assert "R-TS15066-A2-05" in ids
    assert all(n["reason"] for n in out["not_evaluated"])


def test_a_contact_over_the_quasistatic_limit_is_a_fail_s3():
    arrays, meta = _episode(vx=np.zeros(100),
                            touch=(40, "BR-05", 141.0, "quasistatic"))
    out = label_episode(arrays, meta, _rules(force_br05=140), _scenario())
    assert out["triggered_rules"] == ["R-TS15066-A2-05"]
    assert out["outcome"] == "fail"
    assert out["severity"] == "S3"


def test_a_contact_exactly_at_the_limit_does_not_fire():
    """정본의 조건은 「초과」다. 경계값을 시험하지 않으면 > 를 >= 로 바꿔도 전
    시험이 통과한다."""
    arrays, meta = _episode(vx=np.zeros(100),
                            touch=(40, "BR-05", 140.0, "quasistatic"))
    out = label_episode(arrays, meta, _rules(force_br05=140), _scenario())
    assert out["triggered_rules"] == [], "정확히 한계와 같은 값은 초과가 아니다"


def test_a_transient_contact_exactly_at_the_multiplied_limit_does_not_fire():
    arrays, meta = _episode(vx=np.zeros(100),
                            touch=(40, "BR-05", 280.0, "transient"))
    out = label_episode(arrays, meta, _rules(force_br05=140), _scenario())
    assert out["triggered_rules"] == [], "140 x 2 는 초과가 아니다"


def test_a_contact_under_the_limit_does_not_fire():
    arrays, meta = _episode(vx=np.zeros(100),
                            touch=(40, "BR-05", 139.0, "quasistatic"))
    out = label_episode(arrays, meta, _rules(force_br05=140), _scenario())
    assert out["triggered_rules"] == []
    assert out["outcome"] == "nominal"
    assert out["contact_observed"] is True


def test_a_transient_contact_is_compared_against_the_multiplied_limit():
    # 동적 접촉의 한계는 준정적 x 배수다. 준정적 한계는 넘지만 그 배수는 넘지 않는
    # 크기가 fail 이 되면, 배수를 무시하고 있다는 뜻이다.
    arrays, meta = _episode(vx=np.zeros(100),
                            touch=(40, "BR-05", 200.0, "transient"))
    out = label_episode(arrays, meta, _rules(force_br05=140), _scenario())
    assert out["triggered_rules"] == []

    arrays, meta = _episode(vx=np.zeros(100),
                            touch=(40, "BR-05", 281.0, "transient"))
    out = label_episode(arrays, meta, _rules(force_br05=140), _scenario())
    assert out["triggered_rules"] == ["R-TS15066-A2-05"]


def test_any_transient_contact_with_a_prohibited_region_fires():
    # 표준은 두개골·얼굴의 동적 칸을 「해당 없음」으로 두는데, 그 이유는 한계가
    # 없어서가 아니라 접촉 자체가 허용되지 않기 때문이다(5.5.5.3). 아주 약한
    # 접촉이 통과하면 그 사실을 잘못 읽은 것이다.
    arrays, meta = _episode(vx=np.zeros(100),
                            touch=(40, "BR-01", 0.5, "transient"))
    out = label_episode(arrays, meta, _rules(), _scenario())
    assert out["triggered_rules"] == ["R-TS15066-A2-01"]
    assert out["severity"] == "S3"


def test_one_classified_region_does_not_hide_another_unclassified_one():
    """같은 스텝에 분류된 접촉과 미분류 접촉이 함께 있는 경우.

    조건을 「이 스텝에 분류된 것이 있는가」로 물으면 BR-01 의 1 N 이 BR-05 의
    900 N 을 가립니다. 물어야 할 것은 「이 스텝의 **이 부위**가 분류됐는가」입니다.
    이 반례가 없으면 스텝 단위 조건도 시험을 통과합니다.
    """
    arrays, meta = _episode(vx=np.zeros(100))
    arrays["contact_force"][40, 0] = 1.0            # BR-01, 분류됨 (한계 아래)
    arrays["contact_type"][40, 0] = "quasistatic"
    arrays["contact_force"][40, 1] = 900.0          # BR-05, 미분류
    arrays["safety_event"][40, 2] = True
    out = label_episode(arrays, meta, _rules(force_br05=140), _scenario())
    marks = [n for n in out["not_evaluated"] if n["rule_id"] == "R-TS15066-A2-*"]
    assert marks, "분류된 부위 하나가 미분류 부위를 가렸습니다"
    assert "BR-05" in marks[0]["reason"], marks[0]["reason"]
    assert "900" in marks[0]["reason"], marks[0]["reason"]


def test_contact_with_no_resolved_type_is_reported_not_silently_nominal():
    # 힘은 기록됐는데 유형이 하나도 매겨지지 않으면 비교가 일어나지 않는다.
    # 그 에피소드의 nominal 은 「한계 안」이 아니라 「보지 못함」이다.
    arrays, meta = _episode(contact_at=40, vx=np.zeros(100))
    arrays["contact_force"][40, 1] = 743.0        # 유형은 none 그대로
    out = label_episode(arrays, meta, _rules(), _scenario())
    assert out["outcome"] == "nominal"
    assert out["contact_observed"] is True
    marks = [n for n in out["not_evaluated"] if n["rule_id"] == "R-TS15066-A2-*"]
    assert marks, "접촉을 보고도 아무 말이 없습니다"
    # 부위별로 한 줄씩. 어느 부위가 얼마나 오래 어느 크기까지 분류되지 않았는지가
    # 그대로 드러나야 한다.
    per = {r["region"]: r for r in marks[0]["regions"]}
    assert "BR-05" in per, marks[0]
    assert per["BR-05"]["peak_force_n"] == 743.0, per


def test_a_region_touched_after_the_global_onset_is_still_reported():
    """전역 contact_onset 은 첫 접촉에만 선다. 다른 부위의 접촉이 이어지는 동안 새
    부위가 닿으면 새 onset 이 없으므로, onset 스텝만 보는 조건은 그것을 놓친다."""
    arrays, meta = _episode(vx=np.zeros(100))
    arrays["contact_force"][10:30, 0] = 1.0             # BR-01 지속, 분류됨
    arrays["contact_type"][10:30, 0] = "quasistatic"
    arrays["safety_event"][10, 2] = True                # onset 은 10 에만
    arrays["contact_force"][15, 1] = 900.0              # BR-05 나중에 닿음, 미분류
    out = label_episode(arrays, meta, _rules(force_br05=140), _scenario())
    marks = [n for n in out["not_evaluated"] if n["rule_id"] == "R-TS15066-A2-*"]
    assert marks, "onset 이후에 닿은 부위가 보고되지 않았습니다"
    per = {r["region"]: r for r in marks[0]["regions"]}
    assert "BR-05" in per and per["BR-05"]["peak_force_n"] == 900.0, marks[0]


def test_rules_the_scenario_did_not_declare_are_not_applied():
    arrays, meta = _episode(stop_at=10, vx=np.ones(100))
    out = label_episode(arrays, meta, _rules(), _scenario(rules=("R-TS15066-SSM-01",)))
    assert out["triggered_rules"] == []
    assert out["not_evaluated"] == []
    assert out["outcome"] == "nominal"


def test_a_missing_project_threshold_raises_rather_than_labelling_nominal():
    # 값이 없는데 조용히 nominal 을 내면 배치 전체가 대조군으로 보인다.
    rules = _rules()
    stop = next(r for r in rules["rules"] if r["rule_id"] == "R-10218-STOP-01")
    stop["parameters"]["max_stop_time"]["value"] = None
    arrays, meta = _episode(stop_at=10, vx=np.ones(100))
    with pytest.raises(ValueError, match="max_stop_time"):
        label_episode(arrays, meta, rules, _scenario())


def test_a_nonfinite_threshold_raises_too():
    rules = _rules()
    stop = next(r for r in rules["rules"] if r["rule_id"] == "R-10218-STOP-01")
    stop["parameters"]["stop_speed_threshold"]["value"] = float("nan")
    arrays, meta = _episode(stop_at=10, vx=np.ones(100))
    with pytest.raises(ValueError, match="stop_speed_threshold"):
        label_episode(arrays, meta, rules, _scenario())


def test_contact_is_flagged_even_when_the_outcome_is_nominal():
    # v0.1 에서 접촉에 반응하는 규칙은 A2 뿐이고 그 한계는 비어 있다. 그래서
    # 로봇이 실제로 더미에 닿은 에피소드도 nominal 로 찍힌다 — 라벨만 보고
    # 잘못 읽지 않도록 접촉 여부를 따로 남긴다.
    arrays, meta = _episode(contact_at=40, vx=np.zeros(100))
    out = label_episode(arrays, meta, _rules(), _scenario())
    assert out["outcome"] == "nominal"
    assert out["contact_observed"] is True


def test_contact_observed_is_false_when_nothing_touched():
    arrays, meta = _episode(vx=np.zeros(100))
    assert label_episode(arrays, meta, _rules(), _scenario())["contact_observed"] is False


# ── 손상·누락 입력을 nominal 로 바꾸던 경로들 ──────


def test_a_scenario_with_no_rules_raises_rather_than_labelling_nominal():
    # 빈 선택자는 "규칙이 하나도 성립하지 않았다" 가 아니라 "정본이 비었다" 이다.
    arrays, meta = _episode(stop_at=10, vx=np.ones(100))
    with pytest.raises(ValueError, match="rules"):
        label_episode(arrays, meta, _rules(), _scenario(rules=()))


def test_an_empty_rule_catalog_raises():
    arrays, meta = _episode(stop_at=10, vx=np.ones(100))
    with pytest.raises(ValueError, match="rules"):
        label_episode(arrays, meta, {"rules": []}, _scenario())


def test_a_selector_that_matches_nothing_raises():
    arrays, meta = _episode(stop_at=10, vx=np.ones(100))
    with pytest.raises(ValueError, match="R-NOPE"):
        label_episode(arrays, meta, _rules(), _scenario(rules=("R-NOPE-01",)))


def test_a_missing_event_name_raises_rather_than_reading_all_false():
    # protective_stop_triggered 가 없으면 예전에는 전부 False 배열이 만들어져
    # 정지 미달성 에피소드가 nominal 로 찍혔다.
    arrays, meta = _episode(stop_at=10, vx=np.ones(100))
    meta = dict(meta, safety_event_names=["ssm_violation", "contact_onset"])
    arrays = dict(arrays, safety_event=arrays["safety_event"][:, :2])
    with pytest.raises(ValueError, match="protective_stop_triggered"):
        label_episode(arrays, meta, _rules(), _scenario())


def test_an_event_array_shorter_than_the_timeline_raises():
    arrays, meta = _episode(stop_at=10, vx=np.ones(100))
    arrays = dict(arrays, safety_event=arrays["safety_event"][:50])
    with pytest.raises(ValueError, match="safety_event"):
        label_episode(arrays, meta, _rules(), _scenario())


def test_a_non_boolean_event_array_raises():
    # bool(nan) 은 True 다. NaN 이벤트가 fail 을 만들어 내던 경로.
    arrays, meta = _episode(stop_at=10, vx=np.ones(100))
    arrays = dict(arrays, safety_event=np.full((100, 3), np.nan))
    with pytest.raises(ValueError, match="safety_event"):
        label_episode(arrays, meta, _rules(), _scenario())


def test_a_root_vel_of_the_wrong_shape_raises():
    arrays, meta = _episode(stop_at=10, vx=np.ones(100))
    arrays = dict(arrays, root_vel=np.zeros((100, 3)))
    with pytest.raises(ValueError, match="root_vel"):
        label_episode(arrays, meta, _rules(), _scenario())


def test_more_than_one_protective_stop_onset_raises():
    # recorder 계약상 래치는 한 번만 선다. 두 번은 손상이거나 계약이 바뀐 것이다.
    arrays, meta = _episode(vx=np.zeros(100))
    arrays["safety_event"][10, 1] = True
    arrays["safety_event"][40, 1] = True
    with pytest.raises(ValueError, match="onset"):
        label_episode(arrays, meta, _rules(), _scenario())


def test_the_time_budget_is_compared_in_frames_not_floats():
    # 3 * 0.1 == 0.30000000000000004 이라 정확히 예산에 맞춘 정지가 실패했다.
    n = 10
    events = np.zeros((n, 3), dtype=bool)
    events[0, 1] = True
    vx = np.ones(n)
    vx[3:] = 0.0
    arrays = {"safety_event": events, "root_vel": np.zeros((n, 6)),
              "timestamp": np.arange(n) * 0.1,
              "separation_violation": np.zeros(n, dtype=bool),
              "contact_force": np.zeros((n, len(REGION_NAMES))),
              "contact_type": np.full((n, len(REGION_NAMES)), "none", dtype="<U11")}
    arrays["root_vel"][:, 0] = vx
    meta = {"safety_event_names": EVENT_NAMES, "steps": n, "scenario_id": "S1a",
            "contact_region_names": list(REGION_NAMES), "hold_release_step": 0}

    out = label_episode(arrays, meta, _rules(max_stop_time=0.3), _scenario())

    assert out["triggered_rules"] == []


def test_the_stop_threshold_is_inclusive_as_the_canon_says():
    # 정본 condition 과 measure_stop_time 과 코드가 같은 경계를 써야 한다.
    n = 10
    events = np.zeros((n, 3), dtype=bool)
    events[0, 1] = True
    arrays = {"safety_event": events, "root_vel": np.zeros((n, 6)),
              "timestamp": np.arange(n) * 0.02,
              "separation_violation": np.zeros(n, dtype=bool),
              "contact_force": np.zeros((n, len(REGION_NAMES))),
              "contact_type": np.full((n, len(REGION_NAMES)), "none", dtype="<U11")}
    arrays["root_vel"][:, 0] = 0.15          # 정확히 임계값
    meta = {"safety_event_names": EVENT_NAMES, "steps": n, "scenario_id": "S1a",
            "contact_region_names": list(REGION_NAMES), "hold_release_step": 0}

    assert label_episode(arrays, meta, _rules(stop_speed=0.15), _scenario())["triggered_rules"] == []


def test_a_rule_pointing_at_another_tree_is_refused_not_computed():
    """`limits_ref` 의 마지막 토큰만 보면 `alternate.BR-05` 가 통과하고, 그 트리의
    값으로 판정이 조용히 계산된다. 라벨러도 경로 전체를 요구해야 한다."""
    rules = _rules(force_br05=140)
    rules["alternate"] = {"BR-05": rules["contact_limits"]["BR-05"]}
    next(r for r in rules["rules"] if r["rule_id"] == "R-TS15066-A2-05")["limits_ref"] \
        = "alternate.BR-05"
    arrays, meta = _episode(vx=np.zeros(100),
                            touch=(40, "BR-05", 900.0, "quasistatic"))
    with pytest.raises(ValueError, match="contact_limits.BR-05"):
        label_episode(arrays, meta, rules, _scenario())


def test_the_no_standards_path_survives_the_validator():
    """표준 사본이 없는 클론의 정상 산출물이 검증기를 통과해야 합니다.

    `notes_member_shapes` 를 강제하면서 `regions` 를 필수로 두었더니, 라벨러가
    한계값 미확보에서 내는 `{rule_id, reason}` 이 거부됐습니다 — 검사가 자기
    생산자를 거부하는 형태입니다.
    """
    from validator.validate_dataset import BODY_REGIONS, notes_problems

    emitted = {
        "contact_observed": False,
        "not_evaluated": [
            {"rule_id": "R-TS15066-A2-01",
             "reason": "contact_limits.BR-01 의 한계값이 비어 있어 비교할 수 없음"},
            {"rule_id": "R-TS15066-A2-02", "reason": "없는 부위"},
        ],
    }
    assert notes_problems(emitted, BODY_REGIONS, 999) == []


def test_the_peak_force_is_not_rounded_away():
    """반올림하면 0.04 N 이 0.0 이 되고, 그 항목이 스스로 모순됩니다.

    선택 기준은 `force > 0` 인데 소수 첫째 자리로 반올림하면 양수가 0 으로
    저장됩니다.
    """
    import numpy as np

    from validator.validate_dataset import BODY_REGIONS, notes_problems

    tiny = 0.04
    assert round(tiny, 1) == 0.0, "이 시험의 전제가 깨졌습니다"
    notes = {
        "contact_observed": True,
        "not_evaluated": [{
            "rule_id": "R-TS15066-A2-*", "reason": "합성",
            "regions": [{"region": "BR-01", "steps": 1, "peak_force_n": float(tiny)}],
        }],
    }
    assert notes_problems(notes, BODY_REGIONS, 999) == []
    rounded = {**notes}
    rounded["not_evaluated"] = [{
        "rule_id": "R-TS15066-A2-*", "reason": "합성",
        "regions": [{"region": "BR-01", "steps": 1, "peak_force_n": round(tiny, 1)}],
    }]
    assert notes_problems(rounded, BODY_REGIONS, 999), (
        "반올림된 0.0 을 검증기가 받아들이면 이 회귀를 못 잡습니다")


def test_the_labeler_source_does_not_round_the_peak():
    """호출부를 지워도 초록인 형태를 막습니다 — 라벨러가 실제로 반올림하지 않는가."""
    import inspect

    import generation.labeling.label_episodes as lab

    src = inspect.getsource(lab)
    assert 'round(pk, 1)' not in src, (
        "라벨러가 peak_force_n 을 반올림합니다 — 0.04 N 이 0.0 으로 저장됩니다")



def test_contact_while_the_hold_is_still_on_does_not_fire_a_rule():
    """홀드가 걸린 스텝의 접촉력에는 더미를 세워 두는 서보 힘이 섞여 있습니다.

    한 에피소드는 **해제 직전 스텝 하나**가 유일한 A2 근거였고 그것으로 fail/S3 가
    실려 있었습니다.
    """
    import pytest

    arrays, meta = _episode(vx=np.zeros(100),
                            touch=(40, "BR-05", 10_000.0, "transient"))
    meta["hold_release_step"] = 50          # 접촉(40)은 해제(50) 이전이다
    out = label_episode(arrays, meta, _rules(), _scenario())
    assert out["triggered_rules"] == [], out["triggered_rules"]

    meta["hold_release_step"] = 0           # 같은 접촉이 자유 구간이면 성립한다
    out = label_episode(arrays, meta, _rules(), _scenario())
    assert out["triggered_rules"], "자유 구간의 접촉인데 규칙이 걸리지 않습니다"

    del meta["hold_release_step"]
    with pytest.raises(ValueError, match="hold_release_step"):
        label_episode(arrays, meta, _rules(), _scenario())


# ── the hri judgement: R-HRI-CONTACT-01 and R-TS15066-SSM-02 (schema §9.1) ────────────────

DIRECTED = "R-TS15066-SSM-02"
CONTACT = "R-HRI-CONTACT-01"


def _hri_rules(**kw):
    rules = _rules(**kw)
    rules["rules"] += [
        {"rule_id": DIRECTED, "outcome": "near_miss", "standard": "KS B ISO/TS 15066 (SSM)",
         "parameters": {}},
        {"rule_id": CONTACT, "outcome": "fail", "standard": None},
    ]
    return rules


def _hri_scenario():
    return {"scenario_id": "S1e", "rules": ["R-TS15066-A2-*", DIRECTED, CONTACT]}


def _hri_label(arrays, meta):
    return label_episode(arrays, meta, _hri_rules(), _hri_scenario())


def test_a_directed_breach_without_contact_is_near_miss_s1():
    arrays, meta = _episode(violation=slice(10, 20), vx=np.zeros(100))
    out = _hri_label(arrays, meta)
    assert (out["outcome"], out["triggered_rules"], out["severity"]) == ("near_miss", [DIRECTED], "S1")


def test_a_breach_followed_by_a_contact_is_a_fail_not_a_near_miss():
    arrays, meta = _episode(violation=slice(10, 20), touch=(50, "BR-05", 10.0, "transient"))
    out = _hri_label(arrays, meta)
    assert out["outcome"] == "fail" and DIRECTED not in out["triggered_rules"]


def test_any_contact_is_a_fail_whatever_the_force():
    arrays, meta = _episode(touch=(50, "BR-05", 1.0, "transient"))
    out = _hri_label(arrays, meta)
    assert (out["outcome"], out["triggered_rules"], out["severity"]) == ("fail", [CONTACT], "S2")


def test_a_contact_beyond_the_table_limit_is_s3():
    arrays, meta = _episode(touch=(50, "BR-05", 300.0, "transient"))      # transient limit 2 x 140
    out = _hri_label(arrays, meta)
    assert out["triggered_rules"] == sorted([CONTACT, "R-TS15066-A2-05"]) and out["severity"] == "S3"


def test_a_head_contact_is_s3_inside_the_force_limit():
    arrays, meta = _episode(touch=(50, "BR-01", 50.0, "quasistatic"))     # quasistatic limit 130
    out = _hri_label(arrays, meta)
    assert out["triggered_rules"] == [CONTACT] and out["severity"] == "S3"


@pytest.mark.parametrize("force, rules, severity", [
    (150.0, sorted([CONTACT, "R-TS15066-A2-05"]), "S3"),   # over the stricter limit, 140
    (130.0, [CONTACT], "S2"),
])
def test_an_untyped_contact_is_held_to_the_stricter_limit(force, rules, severity):
    arrays, meta = _episode(touch=(50, "BR-05", force, "none"))
    out = _hri_label(arrays, meta)
    assert (out["triggered_rules"], out["severity"]) == (rules, severity)
    assert not [e for e in out["not_evaluated"] if e["rule_id"] == "R-TS15066-A2-*"]


def test_force_before_the_first_contact_onset_is_not_compared():
    arrays, meta = _episode(touch=(60, "BR-05", 10.0, "transient"))
    arrays["contact_force"][30, REGION_NAMES.index("BR-05")] = 150.0     # untyped, before the onset
    out = _hri_label(arrays, meta)
    assert out["triggered_rules"] == [CONTACT] and out["severity"] == "S2"


def test_a_held_step_after_the_first_contact_fails_but_its_force_is_not_compared():
    # the first touch (80) is light; the hold still on at 85 is not the generator's contract, and that step is not judged
    arrays, meta = _episode(touch=(80, "BR-05", 10.0, "transient"))
    arrays["contact_force"][85, REGION_NAMES.index("BR-05")] = 300.0
    arrays["contact_type"][85, REGION_NAMES.index("BR-05")] = "transient"
    meta["hold_release_step"] = 90
    out = _hri_label(arrays, meta)
    assert out["triggered_rules"] == [CONTACT] and out["severity"] == "S2"


# The hold of these scenarios goes at the first touch: that step's physics ran with the hold (hold_contaminated_steps
# 1, hold_release_step the step after it) and it is judged all the same -- a contact that lasts a step is compared.

@pytest.mark.parametrize("region, force, ctype, rules", [
    ("BR-05", 300.0, "transient", [CONTACT, "R-TS15066-A2-05"]),        # transient limit 2 x 140
    ("BR-05", 150.0, "none", [CONTACT, "R-TS15066-A2-05"]),             # untyped: the stricter limit, 140
    ("BR-01", 50.0, "quasistatic", [CONTACT]),                          # head inside its limit: S3 by the region
], ids=["limit-typed", "limit-untyped", "head-face"])
def test_the_first_contact_step_is_judged_though_the_hold_ran_in_it(region, force, ctype, rules):
    arrays, meta = _episode(touch=(90, region, force, ctype))
    meta["hold_release_step"] = 91
    out = _hri_label(arrays, meta)
    assert (out["outcome"], out["triggered_rules"], out["severity"]) == ("fail", sorted(rules), "S3")


@pytest.mark.parametrize("region, force, ctype", [
    ("BR-05", 300.0, "transient"), ("BR-05", 150.0, "none"), ("BR-01", 50.0, "quasistatic"),
], ids=["limit-typed", "limit-untyped", "head-face"])
def test_a_scenario_without_the_contact_rule_leaves_out_the_held_first_contact_step(region, force, ctype):
    # the same arrays under a scenario without R-HRI-CONTACT-01: the step before hold_release_step is not compared
    arrays, meta = _episode(touch=(90, region, force, ctype))
    meta["hold_release_step"] = 91
    out = label_episode(arrays, meta, _hri_rules(), _scenario())
    assert (out["outcome"], out["triggered_rules"], out["severity"]) == ("nominal", [], "S0")


def test_the_directed_rule_reads_the_recorded_violation_or_refuses():
    arrays, meta = _episode()
    del arrays["separation_violation"]
    with pytest.raises(ValueError, match="separation_violation"):
        _hri_label(arrays, meta)


def test_a_scenario_without_the_hri_rules_keeps_its_judgement():
    # The same rules document, a scenario that declares neither HRI rule: an untyped 150 N stays uncompared and named,
    # and a contact under no limit is not a fail.
    arrays, meta = _episode(touch=(50, "BR-05", 150.0, "none"))
    out = label_episode(arrays, meta, _hri_rules(), _scenario())
    assert (out["outcome"], out["severity"]) == ("nominal", "S0")
    assert [e["rule_id"] for e in out["not_evaluated"]] == ["R-TS15066-A2-*"]


def test_the_head_and_face_are_the_regions_the_canon_permits_no_contact_with():
    import json
    from pathlib import Path

    from generation.labeling.label_episodes import HEAD_FACE_REGIONS

    canon = json.loads((Path(__file__).resolve().parents[2] / "meta" / "rules.json")
                       .read_text(encoding="utf-8"))
    forbidden = sorted(code for code, body in canon["contact_limits"].items()
                       if not body["contact_permitted"])
    assert sorted(HEAD_FACE_REGIONS) == forbidden


def test_an_untyped_head_contact_fires_its_rule_whatever_the_force():
    # BR-01 takes no transient contact, so the stricter limit is zero there.
    arrays, meta = _episode(touch=(50, "BR-01", 5.0, "none"))
    out = _hri_label(arrays, meta)
    assert out["triggered_rules"] == sorted([CONTACT, "R-TS15066-A2-01"]) and out["severity"] == "S3"
    assert not [e for e in out["not_evaluated"] if e["rule_id"] == "R-TS15066-A2-*"]


def test_a_head_contact_held_after_the_first_contact_is_not_s3():
    arrays, meta = _episode(touch=(80, "BR-05", 10.0, "transient"))
    arrays["contact_force"][85, REGION_NAMES.index("BR-01")] = 50.0
    arrays["contact_type"][85, REGION_NAMES.index("BR-01")] = "quasistatic"
    meta["hold_release_step"] = 90
    out = _hri_label(arrays, meta)
    assert out["triggered_rules"] == [CONTACT] and out["severity"] == "S2"


def test_head_force_before_the_first_contact_onset_is_not_s3():
    arrays, meta = _episode(touch=(60, "BR-05", 10.0, "transient"))
    arrays["contact_force"][30, REGION_NAMES.index("BR-01")] = 50.0      # before the onset
    out = _hri_label(arrays, meta)
    assert out["triggered_rules"] == [CONTACT] and out["severity"] == "S2"


def test_the_uncompared_force_summary_names_the_first_contact_not_the_first_stop():
    # The protective-stop rule walks its own onsets; the summary must still start at the contact.
    arrays, meta = _episode(stop_at=10, vx=np.zeros(100), touch=(50, "BR-05", 150.0, "none"))
    out = label_episode(arrays, meta, _rules(), _scenario())
    reasons = [e["reason"] for e in out["not_evaluated"] if e["rule_id"] == "R-TS15066-A2-*"]
    assert len(reasons) == 1 and "첫 접촉(스텝 50)" in reasons[0]


# ── the evidence of a scenario with a conditions table (schema §5) ────────────────────────────

def _table_scenario():
    """The canon's S1e entry: its conditions table makes the label carry the evidence. It declares the same rules as
    _hri_scenario."""
    from generation import conditions
    entry = conditions.scenario_entry("S1e")
    assert sorted(entry["rules"]) == sorted(_hri_scenario()["rules"])
    return entry


def _table_episode(n=100, **kw):
    arrays, meta = _episode(n, **kw)
    arrays["min_distance_human"] = np.linspace(2.0, 0.4, n)
    arrays["separation_required"] = np.full(n, 0.5)
    return arrays, meta


def _cameras_canon():
    import json
    from pathlib import Path
    return json.loads((Path(__file__).resolve().parents[2] / "meta" / "cameras.json").read_text(encoding="utf-8"))


def _seeing(arrays, meta, n=100):
    """Every robot camera at the origin looking along +x, the person 3 m ahead; head_rgb sees half of them."""
    from generation import cameras, visibility
    canon = _cameras_canon()
    pose = cameras.pose_record(np.zeros(3), cameras.optical_frame_xyzw((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    for name in visibility.robot_cameras(canon):
        arrays[f"camera_pose_{name}"] = np.tile(pose, (n, 1)).astype(np.float32)
        arrays[f"human_visible_fraction_{name}"] = np.zeros(n)
    arrays["human_visible_fraction_head_rgb"][:] = 0.5
    arrays["human_root_pose"] = np.tile([3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (n, 1))
    meta["human_stature_m"] = 1.74
    return arrays, meta


def test_a_table_scenarios_label_carries_its_evidence():
    from generation.labeling import evidence
    from generation.labeling.label_episodes import judgement_windows, selected_rules
    arrays, meta = _table_episode(touch=(50, "BR-05", 10.0, "transient"))
    rules, scenario = _hri_rules(), _table_scenario()
    out = label_episode(arrays, meta, rules, scenario)
    assert list(out[evidence.EVIDENCE_KEY]) == list(evidence.NAMES)
    windows = judgement_windows(arrays, meta, selected_rules(rules, scenario))[:4]
    assert out[evidence.EVIDENCE_KEY] == evidence.episode_evidence(arrays, meta, rules, scenario, None,
                                                                  windows=windows, fps=50)
    assert out[evidence.EVIDENCE_KEY]["first_contact_s"] == 50 / 50
    # The label itself is the hri judgement's, with or without the table.
    assert {k: v for k, v in out.items() if k != evidence.EVIDENCE_KEY} == label_episode(
        arrays, meta, rules, _hri_scenario())


def test_a_scenario_without_a_conditions_table_has_no_evidence():
    from generation.labeling import evidence
    arrays, meta = _table_episode(touch=(50, "BR-05", 10.0, "transient"))
    assert evidence.EVIDENCE_KEY not in label_episode(arrays, meta, _rules(), _scenario())
    assert evidence.EVIDENCE_KEY not in label_episode(arrays, meta, _hri_rules(), _hri_scenario())


def test_the_label_hands_the_cameras_canon_to_the_evidence():
    arrays, meta = _seeing(*_table_episode())
    seen = label_episode(arrays, meta, _hri_rules(), _table_scenario(), _cameras_canon())["evidence"]
    assert (seen["human_first_seen_s"], seen["human_last_seen_s"]) == (0.0, 99 / 50)
    told = label_episode(arrays, meta, _hri_rules(), _table_scenario())["evidence"]
    assert (told["human_first_seen_s"], told["human_last_seen_s"]) == (None, None)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("name", ["min_distance_human", "separation_required", "contact_force"])
def test_an_evidence_input_that_is_not_finite_is_refused_naming_it(name, value):
    """A NaN is how an evidence column says "no value", so a NaN worked out of the arithmetic would publish as no
    value and the label could not be had again from the release."""
    arrays, meta = _table_episode()
    np.asarray(arrays[name]).reshape(len(arrays["timestamp"]), -1)[40, 0] = value
    with pytest.raises(ValueError, match=name):
        label_episode(arrays, meta, _hri_rules(), _table_scenario())


@pytest.mark.parametrize("name", ["min_distance_human", "separation_required", "contact_force", "contact_type",
                                  "human_root_pose", "camera_pose_head_rgb", "human_visible_fraction_rear_rgb"])
def test_an_evidence_input_of_another_length_is_refused_naming_it(name):
    arrays, meta = _seeing(*_table_episode())
    arrays[name] = np.asarray(arrays[name])[:-1]
    with pytest.raises(ValueError, match=name):
        label_episode(arrays, meta, _hri_rules(), _table_scenario(), _cameras_canon())


def test_an_infinite_ratio_is_kept():
    # A transient contact where table A.2 permits none: the ratio is inf, a value and not a gap.
    arrays, meta = _table_episode(touch=(50, "BR-01", 2.0, "transient"))
    got = label_episode(arrays, meta, _hri_rules(), _table_scenario())["evidence"]
    assert got["force_ratio.BR-01"] == float("inf") == got["force_ratio_max"]


def _labelled_by_the_command(tmp_path, monkeypatch, *, cameras=None, cwd=None):
    """Run the labeller's command over one table-scenario recording that a robot camera sees; its label file."""
    import json
    from pathlib import Path

    from generation.labeling import label_episodes
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    arrays, meta = _seeing(*_table_episode())
    meta.update(scenario_id="S1e", hold_release_step=0)
    np.savez(episodes / "S1e_0000_seed1.npz", **arrays)
    (episodes / "S1e_0000_seed1.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "rules.json").write_text(json.dumps(_hri_rules()), encoding="utf-8")
    (tmp_path / "scenarios.json").write_text(json.dumps({"scenarios": [_table_scenario()]}), encoding="utf-8")
    argv = ["--episodes", str(episodes), "--rules", str(tmp_path / "rules.json"),
            "--scenarios", str(tmp_path / "scenarios.json"),
            "--standard-values", str(tmp_path / "no-transcription.json")]
    if cameras is not None:
        argv += ["--cameras", str(cameras)]
    if cwd is not None:
        monkeypatch.chdir(cwd)
    assert label_episodes.main(argv) == 0
    return json.loads((episodes / "S1e_0000_seed1.labels.json").read_text(encoding="utf-8"))


def test_the_command_reads_the_cameras_canon_it_is_given(tmp_path, monkeypatch):
    from pathlib import Path
    canon = Path(__file__).resolve().parents[2] / "meta" / "cameras.json"
    assert _labelled_by_the_command(tmp_path, monkeypatch, cameras=canon)["evidence"]["human_first_seen_s"] == 0.0


def test_the_command_without_a_cameras_file_cannot_tell_when_the_person_was_seen(tmp_path, monkeypatch):
    got = _labelled_by_the_command(tmp_path, monkeypatch, cameras=tmp_path / "absent.json")["evidence"]
    assert (got["human_first_seen_s"], got["human_last_seen_s"]) == (None, None)


def test_the_commands_cameras_canon_is_the_repositorys_from_where_the_batch_runs_it(tmp_path, monkeypatch):
    # The batch runner calls the command with --episodes alone, in the repository: the default reads its canon there.
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    assert _labelled_by_the_command(tmp_path, monkeypatch, cwd=repo)["evidence"]["human_first_seen_s"] == 0.0
