"""코드 저장소의 첫 화면(README.md · README.ko.md)이 되풀이하는 것을 그 출처와 대조한다.

두 첫 화면은 데이터셋 카드에서 태그 · 에피소드 수 · 시나리오 · 카메라 수 · 관절 수 · BibTeX 를 옮겨 적는다. 이 둘은
배포본에 실리지 않아 게시할 때의 대조가 닿지 않으므로, 여기서 카드 · 표식 · 정본과 댄다. 상대 링크는 GitHub 가 저장소
안의 경로로 풀므로, 가리키는 것이 저장소에 있는지와 두 언어가 같은 링크를 같은 차례로 거는지를 본다.

**못 보는 것**: 표 밖의 산문(개요 · 한계 · 라이선스 문단), 배지의 글자, 그림의 내용, GitHub 가 실제로 그리는 모습, 명령이
실제로 도는지(도구의 파일과 플래그가 있는지만 본다).
"""
import re

import pytest

from tests.docs.test_card_front_page import code_of, marker
from tests.docs.test_spec_prose_matches_canon import CARD, REPO, canon, card_scenarios, only, read

PAGES = {"en": REPO / "README.md", "ko": REPO / "README.ko.md"}
# What each page says, as patterns over its 「at a glance」 table.
GLANCE = {
    "en": {"tag": r"\| Dataset \| .* — release `[\d.]+`, tag `(v[\d.]+)` \|",
           "episodes": r"\| Episodes \| \*\*(\d+)\*\* in (\w+) scenarios \(([^)]*)\), (\w+) cameras \|",
           "dof": r"\| Robot \| Unitree H1 \((\d+) DOF\) \|"},
    "ko": {"tag": r"\| 데이터셋 \| .* — 버전 `[\d.]+`, 태그 `(v[\d.]+)` \|",
           "episodes": r"\| 에피소드 \| \*\*(\d+)건\*\* — 시나리오 (\d+)종\(([^)]*)\), 카메라 (\d+)대 \|",
           "dof": r"\| 로봇 \| Unitree H1 \((\d+) DOF\) \|"},
}
WORDS = {"three": 3, "seven": 7}
LINK = re.compile(r"\]\(([^)\s]+)\)")


def links(text: str) -> list[str]:
    return LINK.findall(text)


def cards_tag() -> str:
    return only(r"\| Release \| `[\d.]+` — tag `(v[\d.]+)` \|", read(CARD), "카드 · 한눈에 표의 태그").group(1)


def count(word: str) -> int:
    return int(word) if word.isdigit() else WORDS[word]


def test_the_language_buttons_point_at_each_other():
    for text in map(read, PAGES.values()):
        head = text[:text.index("\n## ")]
        assert "(README.md)" in head and "(README.ko.md)" in head, "언어 버튼이 두 첫 화면을 모두 가리켜야 합니다"


def test_the_two_pages_link_the_same_things_in_the_same_order():
    """링크는 언어와 상관이 없다 — 한쪽에만 더하거나 빼면 다른 쪽 독자가 그 문서를 못 찾는다. 배지의 그림 주소는
    글자와 색이 언어마다 달라 빼고, 같은 문서의 언어판(`….ko.md`)은 같은 링크로 본다."""
    def targets(text):
        return [t.replace(".ko.md", ".md") for t in links(text) if not t.startswith("https://img.shields.io/")]
    assert targets(read(PAGES["en"])) == targets(read(PAGES["ko"]))


@pytest.mark.parametrize("language", sorted(PAGES))
def test_relative_links_point_at_files_of_this_repository(language):
    """GitHub 는 상대 링크를 저장소 안의 경로로 푼다 — 공개 저장소에 없는 파일을 가리키면 그 트리의 이 시험이 운다."""
    relative = [t for t in links(read(PAGES[language])) if not re.match(r"[a-z]+:|#", t)]
    assert relative, "상대 링크를 하나도 읽지 못했습니다 — 링크의 꼴이 바뀌었습니까?"
    missing = [t for t in relative if not (REPO / t).exists()]
    assert not missing, f"{PAGES[language].name} 의 상대 링크가 저장소에 없는 자리를 가리킵니다: {missing}"


@pytest.mark.parametrize("language", sorted(PAGES))
def test_the_page_names_the_cards_dataset_tag(language):
    """첫 화면이 말하는 데이터셋 태그는 카드의 것 하나 — 받는 명령의 `--revision` 도. 카드의 태그를 고치면 이 화면도
    함께 고쳐야 한다."""
    text = read(PAGES[language])
    tag = cards_tag()
    assert only(GLANCE[language]["tag"], text, f"{PAGES[language].name} · 한눈에 표").group(1) == tag
    assert re.findall(r"--revision (\S+)", text) == [tag]
    assert set(re.findall(r"(?<![\w/.-])v\d+\.\d+\.\d+\b", text)) == {tag}


@pytest.mark.parametrize("language", sorted(PAGES))
def test_the_glance_counts_are_the_cards_and_the_canons(language):
    import yaml

    text = read(PAGES[language])
    episodes, scenarios_n, scenarios, cameras = only(GLANCE[language]["episodes"], text,
                                                     f"{PAGES[language].name} · 에피소드 행").groups()
    assert int(episodes) == marker()["episodes"]
    assert re.findall(r"`(\w+)`", scenarios) == card_scenarios()
    assert count(scenarios_n) == len(card_scenarios())
    assert count(cameras) == len(canon("cameras.json")["cameras"])
    joints = yaml.safe_load((REPO / "meta/joint_map/unitree_h1.yaml").read_text(encoding="utf-8"))["joint_names"]
    assert int(only(GLANCE[language]["dof"], text, f"{PAGES[language].name} · 로봇 행").group(1)) == len(joints)


@pytest.mark.parametrize("language", sorted(PAGES))
def test_the_citation_is_the_cards(language):
    assert code_of(read(PAGES[language]), "bibtex") == code_of(read(CARD), "bibtex")


@pytest.mark.parametrize("language", sorted(PAGES))
def test_the_quick_start_runs_tools_this_repository_has(language):
    """`.venv/bin/python <도구> --플래그` — 도구가 있고 그 플래그를 도구가 아는지(글자로)."""
    calls = re.findall(r"^\.venv/bin/python (\S+)([^#\n]*)", read(PAGES[language]), re.M)
    assert len(calls) >= 4, "바로 해 보기의 명령을 읽지 못했습니다"
    for tool, rest in calls:
        assert (REPO / tool).is_file(), f"없는 도구: {tool}"
        source = (REPO / tool).read_text(encoding="utf-8")
        for flag in re.findall(r"--[\w-]+", rest):
            assert f'"{flag}"' in source or f"'{flag}'" in source, f"{tool} 이 모르는 플래그: {flag}"
