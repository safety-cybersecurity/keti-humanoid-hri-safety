"""배포본이 담는 규칙은 **선언된 시나리오가 쓰는 것만** 이어야 한다.

v0.1 은 S1 계열만 담습니다. 그러면 배포본의 rules.json 이
`R-25785-STAB-01`·`-02`·`R-22440-CTRL-01`·`R-OBJ-DROP-01` 을 실을 이유가 없습니다 —
넷 다 S1 계열이 선언하지 않는 규칙이고, C1 을 빨갛게 만들던 임계값 4개가 전부 그
안에 있습니다. 담지 않은 규칙의 빈 임계값 때문에 배포가 막히는 것은 「라벨의 근거를
재현할 수 있어야 한다」 는 계약이 요구하는 바가 아닙니다.

거르는 쪽이 조용히 잘못되면 배포본이 「무엇을 적용했는가」 를 틀리게 말하므로,
아래 가드가 전부 예외를 던집니다. 하나라도 무음이면 이 파일이 빨간불을 냅니다.

S1a · S1d 의 배포본은 S1e · S1f · S1g 의 판정(`R-TS15066-SSM-02` · `R-HRI-CONTACT-01`)도 싣지
않습니다 — 둘은 서로 다른 판정이고, 어느 쪽이 실리는지도 배치의 시나리오가 정합니다.
"""

import json
from pathlib import Path

import pytest

from generation.release_canon import release_canon

REPO_ROOT = Path(__file__).resolve().parents[2]


def canon():
    return (json.loads((REPO_ROOT / "meta/scenarios.json").read_text(encoding="utf-8")),
            json.loads((REPO_ROOT / "meta/rules.json").read_text(encoding="utf-8")))


def test_an_s1_only_release_carries_only_the_rules_s1_declares():
    scenarios, rules = canon()
    sc, ru = release_canon(scenarios, rules, ["S1a"])
    assert [s["scenario_id"] for s in sc["scenarios"]] == ["S1a"]
    kept = {r["rule_id"] for r in ru["rules"]}
    assert {"R-TS15066-SSM-01", "R-10218-STOP-01"} <= kept
    assert len([k for k in kept if k.startswith("R-TS15066-A2-")]) == 12
    assert kept.isdisjoint({"R-25785-STAB-01", "R-25785-STAB-02",
                            "R-22440-CTRL-01", "R-OBJ-DROP-01"})
    assert kept.isdisjoint({"R-TS15066-SSM-02", "R-HRI-CONTACT-01"})   # the judgement of S1e-S1g


def test_that_release_has_no_unfilled_threshold_left():
    """이것이 C1 을 닫는다는 주장 자체다. 정본에서 세면 4건이 남는다."""
    from validator.validate_dataset import threshold_problems

    scenarios, rules = canon()
    assert len(threshold_problems(rules)) == 4, "정본의 미기재 임계값이 4건이 아닙니다"
    _, ru = release_canon(scenarios, rules, ["S1a"])
    assert threshold_problems(ru) == []


def test_an_s1a_only_release_does_not_ship_s1d():
    """v0.1 은 `S1a` 400건만 배포한다 — 정본에 `S1d`(물류 통로 변형)가 있어도 새면 안 된다.

    `S1d`는 `S1a`와 `scenario_id` 접두(`S1`)와 `hazard_class`(`collision`)가 같아, 거르기가
    접두나 hazard_class 로 되어 있었다면 조용히 새는 자리다. 실제로는 `scenario_id` 정확히
    일치로 거르므로 새지 않아야 한다."""
    scenarios, rules = canon()
    sc, _ = release_canon(scenarios, rules, ["S1a"])
    ids = {s["scenario_id"] for s in sc["scenarios"]}
    assert ids == {"S1a"}
    assert "S1d" not in ids


def test_contact_limits_survive_the_filter():
    """A2 규칙은 남으므로 부위별 한계도 남아야 한다 — C1 이 BR-01~12 를 따로 요구한다."""
    scenarios, rules = canon()
    _, ru = release_canon(scenarios, rules, ["S1a"])
    assert set(ru["contact_limits"]) == set(rules["contact_limits"])


def test_the_excluded_rules_are_named_not_silently_dropped():
    scenarios, rules = canon()
    _, ru = release_canon(scenarios, rules, ["S1a"])
    assert set(ru["excluded_rules"]["rule_ids"]) == {
        "R-25785-STAB-01", "R-25785-STAB-02", "R-22440-CTRL-01", "R-OBJ-DROP-01",
        "R-TS15066-SSM-02", "R-HRI-CONTACT-01"}
    assert ru["excluded_rules"]["reason"].strip()


def test_more_scenarios_keep_more_rules():
    """거르기가 시나리오에서 유도되는지. 하드코딩이면 이 시험이 빨강이다."""
    scenarios, rules = canon()
    _, ru = release_canon(scenarios, rules, ["S1a", "S3a"])
    kept = {r["rule_id"] for r in ru["rules"]}
    assert {"R-25785-STAB-01", "R-25785-STAB-02"} <= kept
    assert "R-OBJ-DROP-01" not in kept


def test_an_empty_release_is_refused():
    """시나리오가 없으면 규칙도 없고 C1 은 아무것도 검사하지 않는다 — 초록이 무의미해진다."""
    scenarios, rules = canon()
    with pytest.raises(ValueError, match="시나리오"):
        release_canon(scenarios, rules, [])


def test_an_unknown_scenario_is_refused():
    scenarios, rules = canon()
    with pytest.raises(ValueError, match="S9z"):
        release_canon(scenarios, rules, ["S1a", "S9z"])


def test_a_selector_that_matches_nothing_is_refused():
    """선언이 오타면 조용히 규칙이 줄어든다. 그러면 배포본이 적용 범위를 틀리게 말한다."""
    scenarios, rules = canon()
    scenarios = json.loads(json.dumps(scenarios))
    next(s for s in scenarios["scenarios"] if s["scenario_id"] == "S1a")["rules"] = ["R-NOPE-*"]
    with pytest.raises(ValueError, match="R-NOPE"):
        release_canon(scenarios, rules, ["S1a"])


def test_a_triggered_rule_may_not_be_dropped():
    """라벨이 가리키는 규칙이 배포본에서 빠지면 C4 가 「없는 rule_id」로 걸린다.
    거기서 알기 전에 여기서 멈춘다."""
    scenarios, rules = canon()
    with pytest.raises(ValueError, match="R-25785-STAB-01"):
        release_canon(scenarios, rules, ["S1a"], triggered=["R-25785-STAB-01"])


def test_the_canon_itself_is_not_mutated():
    scenarios, rules = canon()
    before = json.dumps(rules, sort_keys=True)
    release_canon(scenarios, rules, ["S1a"])
    assert json.dumps(rules, sort_keys=True) == before


def test_the_shipped_status_says_complete_when_it_is():
    """정본은 미기재가 남아 thresholds_pending 이 맞지만, 배포본은 다 채워져 있다.

    그대로 두면 C1 이 「complete 로 갱신하십시오」 를 매번 경고하는데, 받는 쪽은
    그럴 수 없습니다 — 정본은 계속 pending 이어야 하기 때문입니다.
    """
    scenarios, rules = canon()
    assert rules["status"] == "thresholds_pending"
    _, ru = release_canon(scenarios, rules, ["S1a"])
    assert ru["status"] == "complete"


def test_the_status_is_not_promoted_when_a_threshold_is_still_missing():
    """상태를 무조건 complete 로 적으면 미기재를 덮는다."""
    scenarios, rules = canon()
    _, ru = release_canon(scenarios, rules, ["S1a", "S3a"])   # STAB 의 임계값이 비어 있다
    assert ru["status"] == "thresholds_pending"


def test_excluded_scenarios_is_not_tied_to_excluded_rules():
    """빠진 시나리오와 빠진 규칙은 별개입니다.

    시나리오 셋이 전 규칙을 포괄하면 excluded 가 비고, 그러면 아홉 시나리오가
    빠졌는데도 공개 정본이 그 사실을 말하지 않았습니다.
    """
    import json
    from pathlib import Path

    from generation.release_canon import release_canon

    repo = Path(__file__).resolve().parents[2]
    scenarios = json.loads((repo / "meta/scenarios.json").read_text(encoding="utf-8"))
    rules = json.loads((repo / "meta/rules.json").read_text(encoding="utf-8"))
    all_ids = [s["scenario_id"] for s in scenarios["scenarios"]]

    # 규칙을 전부 포괄하되 시나리오는 일부만 담는 조합을 찾습니다.
    for k in range(1, len(all_ids)):
        picked = all_ids[:k]
        out_s, out_r = release_canon(scenarios, rules, picked, [])
        if "excluded_rules" not in out_r:
            missing = [x for x in all_ids if x not in picked]
            assert missing, "시나리오를 전부 담았다면 이 시험이 성립하지 않습니다"
            assert out_s.get("excluded_scenarios", {}).get("scenario_ids") == missing, (
                f"규칙이 전부 남았다는 이유로 빠진 시나리오 {missing} 을 숨겼습니다")
            return
    # 그런 조합이 없으면 이 시험이 아무것도 확인하지 못한 것입니다.
    raise AssertionError("전 규칙을 포괄하는 부분 시나리오 조합을 찾지 못했습니다")


HRI_RULES = {"R-TS15066-SSM-02", "R-HRI-CONTACT-01"}


@pytest.mark.parametrize("sid", ["S1e", "S1f", "S1g"])
def test_an_hri_release_carries_the_hri_judgement_and_not_the_ssm_layer(sid):
    scenarios, rules = canon()
    sc, ru = release_canon(scenarios, rules, [sid])
    kept = {r["rule_id"] for r in ru["rules"]}
    assert HRI_RULES <= kept
    assert len([k for k in kept if k.startswith("R-TS15066-A2-")]) == 12
    assert kept.isdisjoint({"R-TS15066-SSM-01", "R-10218-STOP-01"})
    assert ru["status"] == "complete"
    directed = next(r for r in ru["rules"] if r["rule_id"] == "R-TS15066-SSM-02")
    assert directed["stop_table"] == next(r for r in rules["rules"]
                                          if r["rule_id"] == "R-TS15066-SSM-02")["stop_table"]


def test_the_hri_scenarios_say_what_the_converter_needs():
    scenarios, _ = canon()
    by_id = {s["scenario_id"]: s for s in scenarios["scenarios"]}
    for sid in ("S1e", "S1f", "S1g"):
        entry = by_id[sid]
        assert {"domain", "task_archetype", "hazard_class", "safety_function", "robot_task", "scene"} <= set(entry)
        assert entry["hazard_class"] == "collision" and entry["safety_function"] == "none"
        assert "safety.approach_speed_robot" in entry["required_fields"]
    ids = [s["scenario_id"] for s in scenarios["scenarios"]]
    assert ids[ids.index("S1d") + 1: ids.index("S1d") + 4] == ["S1e", "S1f", "S1g"]


def test_the_hri_scenarios_require_every_field_the_directed_rules_read():
    """A scenario's required_fields is what the validator holds a release of it to (C8: a field a
    rule it applies reads must be in the data). Both sides come from the canon: the scenarios that
    select the directed rule, and the inputs that rule and the contact rule list."""
    from generation.directed_separation import scenario_declares_directed_ssm

    scenarios, rules = canon()
    reads = {r["rule_id"]: set(r["inputs"]) for r in rules["rules"] if r["rule_id"] in HRI_RULES}
    assert set(reads) == HRI_RULES and all(reads.values()), "the rules to compare against must list inputs"
    declaring = {s["scenario_id"]: s for s in scenarios["scenarios"] if scenario_declares_directed_ssm(s)}
    assert set(declaring) == {"S1e", "S1f", "S1g"}
    for sid, entry in declaring.items():
        required = set(entry["required_fields"])
        for rule_id, inputs in reads.items():
            assert inputs <= required, (sid, rule_id, sorted(inputs - required))
