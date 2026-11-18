"""C1 임계값 계약을 함수 단위로. 픽스처만으로는 각 검사가 red-green 이 되지 않습니다.

결함 픽스처는 「이 조합이 잡히는가」를 보고, 이 파일은 「이 검사가 무엇을 요구하는가」를
봅니다. 둘이 필요한 이유는 이렇습니다 — 코드만 고치고 시험을 안 붙이면
되돌려도 초록이고, 그 사실을 아무도 모릅니다.
"""

import copy
import json
from pathlib import Path

import pytest

from validator import validate_dataset as V

REPO_ROOT = Path(__file__).resolve().parents[2]
CANON = json.loads((REPO_ROOT / "meta" / "rules.json").read_text(encoding="utf-8"))


def _all(rules):
    return (V.contact_limits_shape_problems(rules) + V.rule_region_problems(rules)
            + V.rule_parameter_exemption_problems(rules)
            + V.contact_permission_problems(rules) + V.threshold_problems(rules))


BASELINE = len(_all(CANON))


def _mutated(fn):
    r = copy.deepcopy(CANON)
    fn(r)
    return _all(r)


def test_the_canon_has_only_the_declared_open_items():
    """기준선이 흔들리면 아래 시험들의 「늘었다」가 의미를 잃습니다."""
    paths = sorted(p for _, p, *_ in ((None, p, w) for p, w in _all(CANON)))
    assert BASELINE == 4, f"기준 문제가 4건이 아닙니다: {paths}"
    assert all("parameters" in p for p in paths), paths


# --- limits_ref 는 contact_limits 경로여야 한다 ------------------------------

def test_a_limits_ref_naming_another_region_is_rejected():
    def m(r):
        next(x for x in r["rules"] if x["rule_id"] == "R-TS15066-A2-05")["limits_ref"] \
            = "contact_limits.BR-06"
    assert len(_mutated(m)) > BASELINE


def test_a_limits_ref_naming_another_tree_is_rejected():
    """마지막 토큰만 비교하면 `alternate.BR-05` 가 통과합니다. 그 트리에 다른 값을
    넣어 두면 판정이 조용히 그 값으로 계산됩니다."""
    def m(r):
        next(x for x in r["rules"] if x["rule_id"] == "R-TS15066-A2-05")["limits_ref"] \
            = "alternate.BR-05"
        r["alternate"] = {"BR-05": copy.deepcopy(r["contact_limits"]["BR-05"])}
    probs = _mutated(m)
    assert len(probs) > BASELINE, "다른 트리를 가리켜도 통과했습니다"
    assert any("alternate.BR-05" in w for _, w in probs)


# --- not_applicable 면제의 자리 ---------------------------------------------

@pytest.mark.parametrize("rule_id,param", [
    ("R-TS15066-SSM-01", "human_speed"),
    ("R-10218-STOP-01", "max_stop_time"),
])
def test_not_applicable_on_a_rule_parameter_is_rejected(rule_id, param):
    """면제의 뜻은 「표준이 그 자리에 수치를 두지 않는다」이고, 그런 자리는 접촉 금지
    부위의 동적 칸뿐입니다. 규칙 파라미터에 허용하면 SSM 파라미터를 면제로 비워 두고도
    배포 게이트를 통과합니다."""
    def m(r):
        p = next(x for x in r["rules"] if x["rule_id"] == rule_id)["parameters"][param]
        p.update(value=None, not_applicable=True, reason="면제 주장")
    probs = _mutated(m)
    assert len(probs) > BASELINE
    assert any(param in p for p, _ in probs)


def test_not_applicable_on_a_permitted_region_is_rejected():
    def m(r):
        tm = r["contact_limits"]["BR-05"]["transient_multiplier"]
        tm.update(value=None, not_applicable=True, reason="사유")
    assert len(_mutated(m)) > BASELINE


def test_not_applicable_must_be_a_real_boolean():
    def m(r):
        r["contact_limits"]["BR-01"]["transient_multiplier"]["not_applicable"] = "false"
    assert len(_mutated(m)) > BASELINE


# --- citation 형식 -----------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("standard_ref", 123), ("standard_ref", ["표 A.2"]), ("standard_ref", "  "),
    ("unit", " "), ("unit", None),
])
def test_a_non_textual_reference_or_unit_is_rejected(field, value):
    def m(r):
        r["contact_limits"]["BR-05"]["limits"]["force_quasistatic"][field] = value
    assert len(_mutated(m)) > BASELINE, f"{field}={value!r} 가 통과했습니다"


def test_a_prohibited_region_needs_a_textual_clause_reference():
    def m(r):
        r["contact_limits"]["BR-01"]["contact_prohibited_ref"] = 123
    assert len(_mutated(m)) > BASELINE


# --- 구조 --------------------------------------------------------------------

@pytest.mark.parametrize("drop", [
    ("BR-05", "limits", "force_quasistatic"),
    ("BR-05", "biomechanical", "effective_mass"),
])
def test_a_deleted_threshold_is_rejected(drop):
    region, group, name = drop
    def m(r):
        del r["contact_limits"][region][group][name]
    assert len(_mutated(m)) > BASELINE


def test_a_deleted_group_is_rejected():
    def m(r):
        del r["contact_limits"]["BR-05"]["limits"]
    assert len(_mutated(m)) > BASELINE


def test_a_region_that_is_not_an_object_is_rejected():
    def m(r):
        r["contact_limits"]["BR-01"] = "not a dict"
    assert len(_mutated(m)) > BASELINE


# --- 단위는 항목마다 하나뿐이다 ----------------------------------------------

@pytest.mark.parametrize("field,bad_unit", [
    ("force_quasistatic", "kN"), ("pressure_quasistatic", "N/m^2"),
    ("energy_transient", "kJ"),
])
def test_a_wrong_but_wellformed_unit_is_rejected(field, bad_unit):
    """「비어 있지 않은 문자열」만 요구하면 배포본이 kN 을 싣고도 통과하고, 라벨러는
    단위를 읽지 않고 기록된 N 과 비교하므로 판정이 통째로 어긋난다."""
    def m(r):
        r["contact_limits"]["BR-05"]["limits"][field]["unit"] = bad_unit
    assert len(_mutated(m)) > BASELINE, f"{field}={bad_unit!r} 가 통과했습니다"


def test_the_two_canonical_unit_tables_agree():
    """검증기와 라벨러가 다른 단위를 정식으로 여기면, 한쪽만 통과하는 배포본이 생긴다."""
    from generation.labeling.label_episodes import CANONICAL_UNITS as LABELLER
    assert V.CANONICAL_UNITS == LABELLER


# --- 면제가 성립하는 자리는 하나뿐 --------------------------------------------

@pytest.mark.parametrize("group,field", [
    ("limits", "force_quasistatic"), ("limits", "energy_transient"),
    ("biomechanical", "effective_mass"),
])
def test_not_applicable_on_a_field_the_standard_does_fill_is_rejected(group, field):
    """접촉 금지 부위라고 해서 힘·에너지·질량 칸까지 비울 수 있는 것은 아니다 —
    표준은 그 값들을 준다. 「해당 없음」은 동적 칸 하나뿐이다."""
    def m(r):
        e = r["contact_limits"]["BR-01"][group][field]
        e.update(value=None, not_applicable=True, reason="면제 주장")
    assert len(_mutated(m)) > BASELINE


# --- 이름이 사라진 파라미터 ---------------------------------------------------

def test_a_deleted_rule_parameter_name_is_reported():
    """이름이 없으면 iter_threshold_entries 의 순회 대상이 되지 않아 값 검사도
    비껴간다 — 검증기는 조용하고 라벨러만 실행 시 죽는다."""
    import copy as _copy
    r = _copy.deepcopy(CANON)
    del next(x for x in r["rules"] if x["rule_id"] == "R-10218-STOP-01") \
        ["parameters"]["max_stop_time"]
    assert "R-10218-STOP-01" in V.REQUIRED_RULE_PARAMETERS
    assert "max_stop_time" in V.REQUIRED_RULE_PARAMETERS["R-10218-STOP-01"]


def test_a_non_object_parameters_container_is_reported_not_crashed():
    """truthy 리스트를 dict 로 가정하면 .items() 에서 죽는다. 검증기가 죽는 것은
    결함 보고가 아니다 — exit 3 은 「검사하지 못했다」이지 「막았다」가 아니다."""
    import copy as _copy
    r = _copy.deepcopy(CANON)
    next(x for x in r["rules"] if x["rule_id"] == "R-10218-STOP-01")["parameters"] = ["x"]
    probs = V.rule_parameter_exemption_problems(r)      # 예외를 내면 시험이 실패한다
    assert any("객체가 아님" in w for _, w in probs), probs
