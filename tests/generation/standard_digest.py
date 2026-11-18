"""표준 임계값 지문 — 시험과 재동결 스크립트가 **같은 함수**를 씁니다.

둘이 따로 직렬화하면 지문이 맞는데도 다른 것을 재고 있을 수 있고, 그러면 지문이
지키는 것이 무엇인지 아무도 모르게 됩니다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CANON = REPO_ROOT / "meta" / "rules.json"
DIGEST_FILE = Path(__file__).with_name("standard_values_digest.json")


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """같은 키가 두 번 나오면 거부한다.

    `json.loads` 는 마지막 값을 취하므로 `"value": 999, "value": 130` 이 있으면
    파이썬은 130 만 보고 지문도 검증기도 초록입니다. **배포되는 바이트에는 둘 다
    있고**, 첫 키를 취하는 파서를 쓰는 소비자는 999 로 판정합니다.
    지문은 바이트에 대한 진술이어야 하므로 여기서 막습니다.
    """
    seen = [k for k, _ in pairs]
    dup = sorted({k for k in seen if seen.count(k) > 1})
    if dup:
        raise ValueError(
            f"JSON 에 중복된 키가 있습니다: {dup} — 파서마다 다른 값을 읽으므로 "
            "같은 파일이 소비자에 따라 다른 임계값을 냅니다")
    return dict(pairs)


def standard_rows(rules: dict) -> list[list]:
    """`determined_by=standard` 인 접촉 한계 항목을 정렬된 행으로.

    값만이 아니라 **단위·조항 참조·면제 표시까지** 담습니다. 값이 맞아도 참조가
    썩으면 출처 없는 표준 수치가 나가고, 그것이 이 저장소가 받은 이용 허락의
    형태(「수치를 조항 인용과 함께」)를 깨뜨립니다.

    **부위 수준의 `contact_permitted` 도 담습니다.** 그것은 「표준이 이 부위와의
    접촉을 허용하는가」라는 실질 주장인데, 중첩된 수치 항목만 훑던 처음 판은 그
    불리언을 통째로 놓쳤습니다 — 목을 금지로 뒤집어도 지문·전사 대조·검증기가
    전부 초록이었습니다.
    """
    rows: list[list] = []
    for region, body in sorted((rules.get("contact_limits") or {}).items()):
        rows.append([region, "contact_permitted", body.get("contact_permitted"),
                     None, body.get("contact_prohibited_ref"), False])
        for key, node in sorted(body.items()):
            if key == "transient_multiplier":
                entries = {key: node}
            elif key in ("limits", "biomechanical") and isinstance(node, dict):
                entries = node
            else:
                continue
            for name, e in sorted(entries.items()):
                if not isinstance(e, dict) or e.get("determined_by") != "standard":
                    continue
                rows.append([region, name, e.get("value"), e.get("unit"),
                             e.get("standard_ref"), bool(e.get("not_applicable"))])
    return rows


def digest_of(rules: dict) -> str:
    blob = json.dumps(standard_rows(rules), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_canon() -> dict:
    return json.loads(CANON.read_text(encoding="utf-8"),
                      object_pairs_hook=_no_duplicate_keys)


def load_frozen() -> dict:
    return json.loads(DIGEST_FILE.read_text(encoding="utf-8"))
