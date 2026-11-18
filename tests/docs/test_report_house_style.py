"""House-style check for the technical report.

``house_style_problems`` is meant to be reused on any chapter text; each
problem is ``"<line number>: <rule>: <start of the line>"``.
"""

import re
from pathlib import Path

import pytest

REPORT = Path(__file__).resolve().parents[2] / "docs" / "tech-report" / "dataset-technical-report-v0.1.md"

CHAPTERS = [
    "개요",
    "근거 표준 및 용어",
    "시나리오",
    "데이터 구성",
    "라벨링과 판정",
    "생성 환경과 재현성",
    "품질검증",
    "활용 방법",
    "한계 및 sim-to-real 검증 로드맵",
    "라이선스 및 이용조건",
    "결론",
]
APPENDICES = "ABCDEFGHIJKLMN"

BANNED = [
    "이로써",
    "를 좌우하는 핵심 요소",
    "다음 세 가지 원칙을 따른다",
]
BANNED_RE = [re.compile(r"이를 위해.*설정하였다")]
ENDING_RE = re.compile(r"다\.?\s*$")
CHAPTER_RE = re.compile(r"^## (\d+)\. ")
SECTION_RE = re.compile(r"^### (\d+)\.(\d+) ")


def house_style_problems(text: str) -> list[str]:
    """Problems in the body (first '## ' up to the first '## 부록')."""
    problems: list[str] = []
    in_body = False
    in_fence = False
    chapter = None
    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("## 부록"):
            break
        if line.startswith("## "):
            in_body = True
        if not in_body:
            continue
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        def add(rule: str) -> None:
            problems.append(f"{number}: {rule}: {line[:40]}")

        m = CHAPTER_RE.match(line)
        if m:
            n = int(m.group(1))
            if chapter is not None and n != chapter + 1:
                add("chapter-order")
            elif chapter is None and n != 1:
                add("chapter-order")
            chapter = n
        m = SECTION_RE.match(line)
        if m and int(m.group(1)) != chapter:
            add("section-number")
        if "**" in line:
            add("bold")
        if line.lstrip().startswith("|"):
            add("table")
        if "—" in line:
            add("dash")
        if any(b in line for b in BANNED) or any(r.search(line) for r in BANNED_RE):
            add("banned")
        if re.match(r"^\s*(ㅇ|-|:|\*) ", line) and ENDING_RE.search(line):
            add("ending")
    return problems


@pytest.mark.parametrize(
    "body, rule",
    [
        ("## 1. 개요\n\n ㅇ **굵게** 씀\n", "bold"),
        ("## 1. 개요\n\n| a | b |\n|---|---|\n", "table"),
        ("## 1. 개요\n\n ㅇ 앞 — 뒤임\n", "dash"),
        ("## 1. 개요\n\n ㅇ 이로써 끝남\n", "banned"),
        ("## 1. 개요\n\n ㅇ 데이터를 싣는다.\n", "ending"),
        ("## 1. 개요\n\n ㅇ 데이터를 한다.\n", "ending"),
        ("## 1. 개요\n\n## 3. 시나리오\n", "chapter-order"),
        ("## 1. 개요\n\n### 2.1 목적\n", "section-number"),
    ],
)
def test_each_rule_is_caught(body, rule):
    problems = house_style_problems(body)
    assert any(f": {rule}:" in p for p in problems), problems


def test_appendix_tables_and_code_blocks_are_allowed():
    text = (
        "## 1. 개요\n\n ㅇ 문장임\n\n```\na — b **c**\n| x |\n```\n\n"
        "## 부록 A. 표\n\n| a | b |\n|---|---|\n ㅇ 앞 — 뒤 **굵게**\n"
    )
    assert house_style_problems(text) == []


def test_the_report_follows_the_house_style():
    text = REPORT.read_text(encoding="utf-8")
    assert house_style_problems(text) == []
    chapters = re.findall(r"^## (\d+)\. (.+?)\s*$", text, re.M)
    assert chapters == [(str(i + 1), t) for i, t in enumerate(CHAPTERS)]
    appendices = re.findall(r"^## 부록 ([A-Z])\. ", text, re.M)
    assert "".join(appendices) == APPENDICES
    heads = [m.start() for m in re.finditer(r"^## ", text, re.M)]
    assert text.index("batch-stats:") < heads[0]
