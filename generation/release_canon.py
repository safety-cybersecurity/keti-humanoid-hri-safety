"""Narrow the canon to what one release actually contains.

The repository's ``meta/scenarios.json`` and ``meta/rules.json`` describe the
whole project: every scenario and every rule, most of which no release has
episodes for yet. (Not counted here: the count read "twelve" for a canon that
had thirteen once S1d was declared.) Copying them verbatim into a distribution
makes it say it applied rules it never ran, and drags their thresholds along --
four of which are values this project has not measured yet, which is what kept
C1 red.

So a distribution carries the scenarios it has episodes for and the rules those
scenarios declare. That is not a weaker claim than before; it is the accurate
one. C1's contract is "the basis of a label can be reproduced", and a rule no
episode was judged by has no label to reproduce.

Filtering silently would be worse than not filtering, because the distribution
would then misstate its own scope. Every way this can go wrong raises:

* no scenarios at all -- C1 would pass by having nothing to check;
* a scenario that is not in the canon;
* a rule selector that matches nothing, which is how a typo turns into a
  quietly smaller rule set;
* a rule some label already triggered. C4 would catch that downstream as
  "rule_id not in rules.json", but by then the dataset is written.
"""

from __future__ import annotations

import copy
import fnmatch
from collections.abc import Iterable, Sequence

# Keys that describe a rule set as a whole rather than one rule. They stay as
# they are: standard_refs is a reference table, not a claim about what ran, and
# contact_limits is required in full by C1 whatever the rule list holds.
_RULES_PASSTHROUGH_NOTE = (
    "이 배포본은 담고 있는 시나리오가 선언한 규칙만 싣습니다. 과제 전체의 규칙 목록은 "
    "코드 저장소의 meta/rules.json 과 규격서 §9 에 있습니다."
)


def _scenario_entries(scenarios: dict) -> list[dict]:
    entries = scenarios.get("scenarios")
    if not isinstance(entries, list):
        raise ValueError("scenarios.json 에 시나리오 배열이 없습니다")
    return entries


def release_canon(
    scenarios: dict,
    rules: dict,
    scenario_ids: Sequence[str],
    triggered: Iterable[str] = (),
) -> tuple[dict, dict]:
    """The scenarios and rules one distribution should carry.

    ``scenario_ids`` is what the batch actually has episodes for; the caller
    reads it off the episodes, not off a list someone maintains by hand.
    ``triggered`` is every rule id the labels of that batch point at.
    """
    wanted = list(dict.fromkeys(scenario_ids))
    if not wanted:
        raise ValueError(
            "시나리오가 하나도 없습니다 — 규칙도 전부 빠지고 C1 은 검사할 것이 없어져 "
            "무의미하게 초록이 됩니다")

    entries = _scenario_entries(scenarios)
    by_id = {s.get("scenario_id"): s for s in entries if isinstance(s, dict)}
    absent = [sid for sid in wanted if sid not in by_id]
    if absent:
        raise ValueError(f"meta/scenarios.json 에 없는 시나리오입니다: {absent}")

    picked = [s for s in entries if s.get("scenario_id") in wanted]

    catalogue = rules.get("rules")
    if not isinstance(catalogue, list):
        raise ValueError("rules.json 에 규칙 배열이 없습니다")
    known = [str(r.get("rule_id", "")) for r in catalogue]

    selectors: list[str] = []
    for s in picked:
        declared = s.get("rules")
        if not isinstance(declared, list) or not declared:
            raise ValueError(
                f"{s.get('scenario_id')}: 적용 규칙 선언이 비었습니다 — "
                "무엇을 실을지 정할 수 없습니다")
        selectors.extend(str(sel) for sel in declared)
    selectors = list(dict.fromkeys(selectors))

    dangling = [sel for sel in selectors
                if not any(fnmatch.fnmatch(rid, sel) for rid in known)]
    if dangling:
        raise ValueError(
            f"실재하는 규칙을 가리키지 않는 선언입니다: {dangling} — "
            "그대로 거르면 배포본이 적용 범위를 틀리게 말합니다")

    kept = [r for r in catalogue
            if any(fnmatch.fnmatch(str(r.get("rule_id", "")), sel) for sel in selectors)]
    kept_ids = {str(r.get("rule_id", "")) for r in kept}

    lost = sorted({str(t) for t in triggered} - kept_ids)
    if lost:
        raise ValueError(
            f"라벨이 가리키는 규칙이 배포본에서 빠집니다: {lost} — "
            "시나리오의 적용 규칙 선언과 라벨이 어긋나 있습니다")

    excluded = [rid for rid in known if rid not in kept_ids]
    # 빠진 시나리오는 빠진 **규칙**과 별개다. 시나리오 셋이 전 규칙을 포괄하면
    # excluded 가 비고, 그러면 아홉 시나리오가 빠졌는데도 공개 정본이 그 사실을
    # 말하지 않았다. 각자의 조건으로 적는다.
    excluded_scenarios = [sid for sid in by_id if sid not in wanted]

    out_scenarios = copy.deepcopy({k: v for k, v in scenarios.items() if k != "scenarios"})
    out_scenarios["scenarios"] = copy.deepcopy(picked)
    if excluded_scenarios:
        out_scenarios["excluded_scenarios"] = {
            "scenario_ids": excluded_scenarios,
            "reason": "이 배포본에 에피소드가 없는 시나리오입니다. 과제 전체의 "
                      "시나리오는 코드 저장소의 meta/scenarios.json 과 시나리오 카탈로그에 있습니다.",
        }

    out_rules = copy.deepcopy({k: v for k, v in rules.items() if k != "rules"})
    out_rules["rules"] = copy.deepcopy(kept)

    # The canon stays thresholds_pending while any rule in the project is
    # unfilled, and that is right for the canon. It is wrong for a distribution
    # whose own rules are all filled -- C1 warns there and tells the reader to
    # update a status they cannot update. So it is recomputed, and recomputed by
    # the validator's own function rather than a second copy of the contract
    # (CONTRIBUTING.md 「함께 고치는 곳」). Imported here, not at module scope, so
    # importing this module does not drag the validator in.
    from validator.validate_dataset import threshold_problems

    if not threshold_problems(out_rules):
        out_rules["status"] = "complete"

    if excluded:
        out_rules["excluded_rules"] = {
            "rule_ids": excluded,
            "reason": _RULES_PASSTHROUGH_NOTE,
        }
    return out_scenarios, out_rules
