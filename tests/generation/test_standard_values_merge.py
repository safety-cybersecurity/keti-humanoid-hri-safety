"""실행 시 병합 경로. 이 파일이 없으면 그 경로를 도는 시험이 하나도 없습니다.

`test_labeling.py` 는 처음부터 값이 채워진 합성 정본을 `label_episode` 에 직접
넘깁니다. 그래서 `load_standard_values` 를 예외를 던지도록 바꿔도 전 시험이
통과했습니다 — 「실행 시 병합이 실제로 작동한다」와 「공개 값이 사본 값보다
우선한다」가 아무 시험의 보호도 받지 않았습니다.
"""

import pytest

from generation.labeling.label_episodes import (load_standard_values,
                                                merge_standard_values)

def _rules(value=None):
    return {"contact_limits": {"BR-05": {
        "name_ko": "합성", "name_en": "synthetic", "contact_permitted": True,
        "limits": {
            "force_quasistatic": {"value": value, "unit": "N",
                                  "determined_by": "standard", "standard_ref": "표 A.2"},
            "pressure_quasistatic": {"value": None, "unit": "N/cm^2",
                                     "determined_by": "standard", "standard_ref": "표 A.2"},
            "energy_transient": {"value": None, "unit": "J",
                                 "determined_by": "standard", "standard_ref": "표 A.4"},
        },
        "transient_multiplier": {"value": None, "unit": "1",
                                 "determined_by": "standard", "standard_ref": "표 A.2"},
        "biomechanical": {
            "effective_mass": {"value": None, "unit": "kg",
                               "determined_by": "standard", "standard_ref": "표 A.3"},
            "spring_constant": {"value": None, "unit": "N/mm",
                                "determined_by": "standard", "standard_ref": "표 A.3"},
        },
    }}}


def _values(force=140, mult=2):
    """실제 사본과 같은 모양. 단위가 빠져 있으면 사본 단위 검증을 시험하지 못한다 —
    시험 픽스처가 배포 경로와 다른 모양이 된다."""
    return {"BR-05": {
        "contact_permitted": True,
        "force_quasistatic": {"value": force, "unit": "N"},
        "pressure_quasistatic": {"value": 120, "unit": "N/cm^2"},
        "energy_transient": {"value": 1.6, "unit": "J"},
        "transient_multiplier": {"value": mult, "unit": "1"},
        "effective_mass": {"value": 40, "unit": "kg"},
        "spring_constant": {"value": 25, "unit": "N/mm"},
    }}


def test_absent_copy_is_absent_not_zero(tmp_path):
    assert load_standard_values(tmp_path / "nope.json") is None


def test_merge_fills_the_empty_entries():
    r = _rules()
    merge_standard_values(r, _values())
    lim = r["contact_limits"]["BR-05"]
    assert lim["limits"]["force_quasistatic"]["value"] == 140
    assert lim["transient_multiplier"]["value"] == 2
    assert lim["biomechanical"]["effective_mass"]["value"] == 40


def test_a_value_already_in_the_public_canon_wins():
    """이용 허락이 나오면 값이 meta/ 로 옮겨 온다. 그때 사본이 그것을 덮으면
    두 벌이 조용히 갈라진다."""
    r = _rules(value=999)
    merge_standard_values(r, _values(force=140))
    assert r["contact_limits"]["BR-05"]["limits"]["force_quasistatic"]["value"] == 999


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, "140", True, None])
def test_an_unusable_copy_value_is_refused_not_merged(bad):
    """nan 은 float 이라 「채워짐」으로 읽히고, 그러면 force > nan 이 영영 거짓이라
    아무리 큰 접촉도 규칙을 성립시키지 못한다.

    None 도 거부합니다 — 사본은 우리가 만드는 파일이고, 값이 빠진 채로 있는 것은
    지원할 상태가 아니라 실수입니다. 조용히 넘기면 그 규칙만 평가되지 않은 채
    배치가 끝나고, 왜인지는 아무 데도 남지 않습니다.
    """
    v = _values()
    v["BR-05"]["force_quasistatic"]["value"] = bad
    with pytest.raises(ValueError):
        merge_standard_values(_rules(), v)


def test_a_copy_value_where_the_canon_says_not_applicable_is_refused():
    r = _rules()
    r["contact_limits"]["BR-05"]["transient_multiplier"].update(
        not_applicable=True, reason="합성")
    with pytest.raises(ValueError):
        merge_standard_values(r, _values(mult=2))


def test_a_unit_mismatch_between_copy_and_canon_is_refused():
    """사본이 kN 인데 정본이 N 이면 숫자만 옮겨져 1000배 어긋난 판정이 나온다."""
    v = _values()
    v["BR-05"]["force_quasistatic"]["unit"] = "kN"
    with pytest.raises(ValueError, match="단위"):
        merge_standard_values(_rules(), v)


def test_a_copy_entry_with_no_unit_is_refused():
    v = _values()
    del v["BR-05"]["force_quasistatic"]["unit"]
    with pytest.raises(ValueError, match="unit"):
        merge_standard_values(_rules(), v)


def test_a_missing_entry_in_the_copy_is_refused_not_skipped():
    """정본이 그 자리를 비워 두고 기다리는데 사본에도 없으면 아무도 채우지 못한다.
    조용히 넘기면 그 규칙만 평가되지 않은 채 배치가 끝난다."""
    v = _values()
    del v["BR-05"]["force_quasistatic"]
    with pytest.raises(ValueError, match="force_quasistatic"):
        merge_standard_values(_rules(), v)


def test_a_missing_region_in_the_copy_is_refused():
    with pytest.raises(ValueError, match="BR-05"):
        merge_standard_values(_rules(), {"BR-99": {}})


def test_both_files_agreeing_on_a_wrong_unit_is_still_refused():
    """정본과 사본의 문자열이 같은지만 보면, 둘이 **같은 오답**에 합의한 경우가 통과한다.
    라벨러는 단위를 읽지 않고 기록된 N 과 직접 비교하므로, 양쪽이 kN 이면 141 N 이
    140 kN 을 넘었다고 판정한다."""
    r = _rules()
    r["contact_limits"]["BR-05"]["limits"]["force_quasistatic"]["unit"] = "kN"
    v = _values()
    v["BR-05"]["force_quasistatic"]["unit"] = "kN"
    with pytest.raises(ValueError, match="'N'"):
        merge_standard_values(r, v)


@pytest.mark.parametrize("how", ["missing", "null"])
def test_a_missing_transient_multiplier_in_the_copy_is_refused(how):
    """일반 항목은 누락 시 거부하면서 배수만 조용히 넘기면, 그 부위의 동적 접촉은
    영영 평가되지 않고 이유는 어디에도 남지 않는다."""
    v = _values()
    if how == "missing":
        del v["BR-05"]["transient_multiplier"]
    else:
        v["BR-05"]["transient_multiplier"]["value"] = None
    with pytest.raises(ValueError, match="transient_multiplier"):
        merge_standard_values(_rules(), v)


def test_a_prohibited_region_does_not_need_a_multiplier_in_the_copy():
    """정본이 not_applicable 이라고 선언한 자리는 사본에 값이 없어도 정상이다."""
    r = _rules()
    tm = r["contact_limits"]["BR-05"]["transient_multiplier"]
    tm.update(value=None, not_applicable=True, reason="합성")
    v = _values()
    v["BR-05"]["transient_multiplier"]["value"] = None
    merge_standard_values(r, v)          # raises 하면 시험 실패
    assert tm["value"] is None


def test_the_canon_values_stay_frozen_without_the_transcription():
    """표준 수치 70개가 전사본과 같은지·올바른 표에서 왔는지를 보는 대조는 전사본이
    있어야 돕니다. 이 저장소에는 전사본이 없습니다(표준 전사본 — 공개 저장소에는 없음).
    검증기는 유한값·단위·비어 있지 않은 참조만 보므로, 그 대조 없이는 유한한 오답
    `999` 가 그대로 지나갈 수 있습니다.

    전사본 없이 지킬 수 있는 것은 「굳힌 뒤로 아무도 값을 손대지 않았다」이고, 이 지문이
    그것을 지킵니다. **skip 이 없습니다** — 어디서든 돕니다.
    """
    from tests.generation.standard_digest import (digest_of, load_canon,
                                                  load_frozen, standard_rows)
    rules = load_canon()
    frozen = load_frozen()
    rows = standard_rows(rules)
    assert len(rows) == frozen["entries"], (
        f"표준 항목 수가 {len(rows)} 인데 굳힌 것은 {frozen['entries']} 입니다 — "
        "자리가 늘거나 줄었습니다")
    assert digest_of(rules) == frozen["sha256"], (
        "meta/rules.json 의 표준 임계값이 굳힌 지문과 다릅니다. 값·단위·standard_ref·"
        "not_applicable 중 무엇이 바뀌었는지 먼저 확인하십시오. **의식적인 변경이라면** "
        "`python tests/generation/refreeze_standard_digest.py` 로 다시 굳히고 이유를 "
        "커밋 메시지에 적으십시오 — 빨간불을 끄려고 먼저 굳히면 이 지문이 지키려던 것을 "
        "스스로 지웁니다")
