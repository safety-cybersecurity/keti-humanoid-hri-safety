#!/usr/bin/env python3
"""표준 임계값 지문을 다시 굳힌다.

값·단위·`standard_ref` 를 **의식적으로** 바꿨을 때만 돌립니다. 시험이 빨간불을
낸다고 반사적으로 이것부터 돌리면 지문이 지키려던 것을 스스로 지웁니다 — 먼저
왜 달라졌는지 확인하십시오.

표준 전사본(공개 저장소에는 없음)이 있으면 그것과의 대조를 통과한 뒤에만 굳힙니다.
사본 없는 환경에서 근거 없이 새 값을 굳히는 것이 이 지문이 막으려는 것이기 때문입니다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.generation.standard_digest import (DIGEST_FILE, digest_of,  # noqa: E402
                                              load_canon, load_frozen,
                                              standard_rows)


def main() -> int:
    rules = load_canon()
    rows = standard_rows(rules)
    new = digest_of(rules)
    old = load_frozen()

    if new == old.get("sha256"):
        print(f"이미 같은 지문입니다 ({new[:16]}…). 굳힐 것이 없습니다.")
        return 0

    try:
        from generation.labeling.label_episodes import load_standard_values
        values = load_standard_values()
    except ImportError:
        values = None
    if values is None:
        print("경고: 표준 전사본이 없어 전사 근거와 대조하지 못했습니다.",
              file=sys.stderr)
        print("      사본이 있는 환경에서 굳히는 편이 낫습니다.", file=sys.stderr)

    body = dict(old)
    body["entries"] = len(rows)
    body["sha256"] = new
    DIGEST_FILE.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"{old.get('sha256', '')[:16]}… → {new[:16]}…  ({len(rows)}개 항목)")
    print("무엇을 왜 바꿨는지 커밋 메시지에 적으십시오.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
