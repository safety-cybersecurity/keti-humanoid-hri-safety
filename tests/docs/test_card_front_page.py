"""데이터셋 카드의 첫 화면(배포본의 README.md · README.ko.md)이 되풀이하는 것을 그 출처와 대조한다.

첫 화면은 짧고, 표와 수의 상세는 상세 문서(`docs/dataset-reference*.md`)에 있다. 상세 문서의 표는
`test_spec_prose_matches_canon.py` 가 정본과 대조한다. 여기서는 첫 화면이 다시 적는 것 — 에피소드 수와 결과 수, 시나리오,
카메라 수, 열 이름, 구성 트리, 링크, 예제 — 을 표식 · 정본 · 상세 문서에서 유도해 댄다.

**못 보는 것**: 표 밖의 산문(개요 두 문단 · 한계 다섯 항 · 라이선스 문단), 그림의 내용, 배지의 글자, Hub 가 실제로 그리는
모습. 스텝 수는 상세 문서의 같은 문장과만 댄다 — 배포본과의 대조는 조립할 때의 일이다.
"""
import json
import re

import pytest

from tests.docs.test_spec_prose_matches_canon import (
    CARD,
    CARD_KO,
    REFERENCE,
    REFERENCE_KO,
    REPO,
    bundled_paths,
    canon,
    card_scenarios,
    only,
    read,
    reference_tree,
    release_doc_paths,
    top_sections,
    tree_names,
)

PAGES = {"en": CARD, "ko": CARD_KO}
REFERENCES = {"en": REFERENCE, "ko": REFERENCE_KO}
HUB = "https://huggingface.co/datasets/KETI-IRRC/keti-humanoid-hri-safety"

# 두 화면의 절 — 같은 차례
SECTIONS = {
    "en": ["Overview", "Scenarios", "Get started", "Dataset structure", "Labels", "Release 26.11", "Limitations",
           "Documentation", "License", "Citation"],
    "ko": ["개요", "시나리오", "시작하기", "데이터 구조", "라벨", "버전 26.11 의 결과", "한계", "문서", "라이선스", "인용"],
}
WORDS = {"three": 3, "five": 5, "twelve": 12}
# A word that, standing before a count word, makes it the tail of a longer number.
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
                "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"}


def page(language: str) -> str:
    return read(PAGES[language])


def section(language: str, number: int) -> str:
    """첫 화면의 ``number`` 번째 절(0부터)의 글 — 다음 절의 머리 앞까지."""
    text, heads = page(language), SECTIONS[language]
    start = text.index(f"\n## {heads[number]}\n")
    end = text.index(f"\n## {heads[number + 1]}\n") if number + 1 < len(heads) else len(text)
    return text[start:end]


def marker() -> dict:
    from generation.publish_dataset import parse_batch_marker
    values, problems = parse_batch_marker(read(CARD), "카드")
    assert not problems, problems
    return values


def fenced(text: str) -> list[str]:
    """글의 코드 블록들 — 울타리 줄을 뺀 안쪽."""
    return re.findall(r"^```[^\n]*\n(.*?)^```", text, re.M | re.S)


@pytest.mark.parametrize("language", sorted(PAGES))
def test_the_front_page_has_its_sections_in_order(language):
    assert top_sections(PAGES[language]) == SECTIONS[language]


def test_the_markers_and_the_front_matter_are_on_the_english_page_alone():
    """게시기와 내보내기 게이트는 배포본의 README.md — 영문 화면 — 에서 표식을 읽는다. 한국어 화면이나 상세 문서에 표식이
    하나 더 있으면 어느 것이 그 배포본의 것인지 갈린다. 머리말도 Hub 가 README.md 에서만 읽는다."""
    english = read(CARD)
    assert english.startswith("---\n"), "영문 첫 화면에 머리말이 없습니다"
    assert len(re.findall(r"<!--\s*batch-stats:", english)) == 1
    assert len(re.findall(r"<!--\s*generator-commit:\s*[0-9a-f]{40}\s*-->", english)) == 1
    for other in (CARD_KO, REFERENCE, REFERENCE_KO):
        text = read(other)
        assert "batch-stats" not in text and "generator-commit" not in text, f"{other.name} 에 표식이 있습니다"
        assert not text.startswith("---\n"), f"{other.name} 에 머리말이 있습니다"


GLANCE_EPISODES = {
    "en": r"\| Episodes \| \*\*(\d+)\*\* — (\w+) scenarios × (\d+) conditions · ([\d,]+) steps at (\d+) fps \|",
    "ko": r"\| 에피소드 \| \*\*(\d+)건\*\* — 시나리오 (\d+)종 × 조건 (\d+)개 · ([\d,]+) 스텝\((\d+) fps\) \|",
}
REFERENCE_STEPS = {"en": r"with ([\d,]+) steps at (\d+) fps", "ko": r"스텝 ([\d,]+)개 · (\d+) fps"}


def grid_rows() -> int:
    """정본의 격자 행 수 — 카드가 말하는 시나리오마다 같아야 한다."""
    by_id = {s["scenario_id"]: s for s in canon("scenarios.json")["scenarios"]}
    cells = {by_id[sid]["conditions"]["grid"]["cells"] for sid in card_scenarios()}
    assert len(cells) == 1, f"시나리오마다 격자의 칸 수가 다릅니다: {cells}"
    return cells.pop() ** 2


@pytest.mark.parametrize("language", sorted(PAGES))
def test_the_glance_counts_are_the_markers(language):
    """한눈에 표의 에피소드 수 · 시나리오 수 · 시나리오마다의 조건 수 ↔ 표식과 정본. 스텝 수와 기록 주기 ↔ 상세 문서의 문장."""
    said = only(GLANCE_EPISODES[language], section(language, 0), f"첫 화면({language}) · 에피소드 행")
    episodes, scenarios, per_scenario = int(said.group(1)), said.group(2), int(said.group(3))
    scenarios = WORDS[scenarios] if language == "en" else int(scenarios)
    stats = marker()
    assert episodes == stats["episodes"]
    assert scenarios == len(card_scenarios())
    assert per_scenario * scenarios == stats["episodes"]
    assert per_scenario == (stats["episodes"] - stats["grid_episodes"]) // scenarios + grid_rows()
    steps = only(REFERENCE_STEPS[language], read(REFERENCES[language]), f"상세 문서({language}) · 스텝 수")
    assert (said.group(4), said.group(5)) == (steps.group(1), steps.group(2))


RESULT_HEAD = {"en": r"\| Scenario \| Random rows \((\d+)\) \| Grid rows \((\d+)\) \|",
               "ko": r"\| 시나리오 \| 무작위 행 \((\d+)\) \| 격자 행 \((\d+)\) \|"}
RESULT_ROW = r"^\| `(\w+)` \| `fail` (\d+) · `near_miss` (\d+) \| `fail` (\d+) · `near_miss` (\d+) \|$"
RESULT_SENTENCE = {
    "en": r"Of the (\d+) random rows, (\d+) ended in contact, and in (\d+) of those a contact force exceeded its "
          r"standard limit\. No episode is `nominal`",
    "ko": r"무작위 (\d+)행 가운데 (\d+)행이 접촉으로 끝났고, 그 가운데 (\d+)행에서 접촉력이 표준 한계를 넘었습니다\. "
          r"`nominal` 은 없습니다",
}
REFERENCE_RELEASE = {"en": "## This release\n", "ko": "## 이 배포본\n"}
REFERENCE_ROW = (r"^\| `(\w+)` \| (\d+) \| (\d+)(?: \([\d.]+%\))? \| (\d+)(?: \([\d.]+%\))? \| (\d+)(?: \([\d.]+%\))? \|")


def reference_results(language: str) -> dict:
    """상세 문서 「This release」 · 「이 배포본」의 두 표 — {시나리오: (무작위 행 수, fail, near_miss, nominal, 격자 행 수,
    fail, near_miss, nominal)}. 앞의 표가 무작위 행, 뒤의 표가 격자 행이다."""
    text = read(REFERENCES[language])
    body = text[text.index(REFERENCE_RELEASE[language]):]
    body = body[:body.index("\n## ", 1)]
    rows = [(m.group(1), *map(int, m.groups()[1:])) for m in re.finditer(REFERENCE_ROW, body, re.M)]
    out = {}
    for sid in card_scenarios():
        mine = [row[1:] for row in rows if row[0] == sid]
        assert len(mine) == 2, f"상세 문서({language})에서 `{sid}` 의 결과 행을 둘 찾지 못했습니다: {mine}"
        out[sid] = (*mine[0], *mine[1])
    return out


@pytest.mark.parametrize("language", sorted(PAGES))
def test_the_result_table_is_the_references_and_the_markers(language):
    """결과 표(시나리오마다 무작위 행 · 격자 행의 fail · near_miss) ↔ 상세 문서의 두 표 ↔ 표식의 합.

    「nominal 은 없다」는 문장은 표식의 nominal 과 상세 문서의 격자 행 nominal 이 모두 0 일 때만 참이다."""
    body = section(language, 5)
    head = only(RESULT_HEAD[language], body, f"첫 화면({language}) · 결과 표의 머리")
    random_rows, grid = int(head.group(1)), int(head.group(2))
    drawn = {m.group(1): tuple(map(int, m.groups()[1:])) for m in re.finditer(RESULT_ROW, body, re.M)}
    assert list(drawn) == card_scenarios(), f"결과 표의 시나리오가 {list(drawn)} 입니다 — 카드의 시나리오는 {card_scenarios()}"
    want = reference_results(language)
    for sid, (fail, near, grid_fail, grid_near) in drawn.items():
        rows, ref_fail, ref_near, ref_nominal, ref_grid, ref_grid_fail, ref_grid_near, ref_grid_nominal = want[sid]
        assert (random_rows, fail, near) == (rows, ref_fail, ref_near), f"`{sid}` 의 무작위 행이 상세 문서와 다릅니다"
        assert (grid, grid_fail, grid_near) == (ref_grid, ref_grid_fail, ref_grid_near), f"`{sid}` 의 격자 행이 상세 문서와 다릅니다"
        assert fail + near + ref_nominal == rows and grid_fail + grid_near + ref_grid_nominal == ref_grid
        assert ref_nominal == 0 and ref_grid_nominal == 0, f"`{sid}` 에 nominal 이 있습니다 — 첫 화면의 문장이 거짓이 됩니다"
    stats = marker()
    assert grid == grid_rows() and grid * len(drawn) == stats["grid_episodes"]
    assert random_rows * len(drawn) == stats["episodes"] - stats["grid_episodes"]
    assert sum(v[0] for v in drawn.values()) == stats["fail"]
    assert sum(v[1] for v in drawn.values()) == stats["near_miss"]
    assert stats["nominal"] == 0
    said = only(RESULT_SENTENCE[language], body, f"첫 화면({language}) · 결과 문장")
    assert tuple(map(int, said.groups())) == (stats["episodes"] - stats["grid_episodes"], stats["fail"],
                                               stats["contact_exceeded"])


GLANCE_FACTS = {
    "en": (r"\| Cameras \| (\d+) RGB, .*? — (\w+) on the robot, one following it, one fixed in the warehouse \|",
           r"body-region breakdown \((\d+) regions\)", r"(\w+)[ -](\w+) body regions",
           r"Unitree H1 \((\d+) DOF\)", r"extension \(schema `([\d.]+)`\)"),
    "ko": (r"\| 카메라 \| RGB (\d+)대, .*? — 로봇에 (\d+)대 · 로봇을 따라가는 1대 · 창고에 고정한 1대 \|",
           r"인체 부위 구분\((\d+)부위\)", r"(?<![\d,.])()(\d+)개 부위",
           r"Unitree H1\((\d+) DOF\)", r"확장 \(스키마 `([\d.]+)`\)"),
}


@pytest.mark.parametrize("language", sorted(PAGES))
def test_the_glance_facts_are_the_canons(language):
    """카메라 수와 로봇에 단 수 · 인체 부위 수 · 관절 수 · 스키마 버전 ↔ 정본."""
    import yaml

    cameras_said, regions_said, regions_again, dof_said, schema_said = GLANCE_FACTS[language]
    body = section(language, 0)
    cameras = canon("cameras.json")["cameras"]
    said = only(cameras_said, body, f"첫 화면({language}) · 카메라 행")
    on_robot = WORDS[said.group(2)] if language == "en" else int(said.group(2))
    assert int(said.group(1)) == len(cameras)
    assert on_robot == sum(1 for c in cameras.values() if c["type"] == "mounted")
    assert len(cameras) - on_robot == 2, "따라가는 카메라 하나 · 고정 카메라 하나라는 문장이 정본과 다릅니다"
    regions = len(canon("rules.json")["contact_limits"])
    assert int(only(regions_said, body, f"첫 화면({language}) · 사람 행").group(1)) == regions
    only(regions_again, body, f"첫 화면({language}) · 개요의 부위 수")
    # The page counts the regions in prose more than once (the overview, the note under the structure): every one,
    # and the whole number — what stands before it is not the tail of a longer one ("hundred twelve", "1,012").
    mentions = re.findall(regions_again, page(language))
    assert len(mentions) == page(language).count({"en": " body regions", "ko": "개 부위"}[language]), (
        f"첫 화면({language})이 부위 수를 말하는 자리 가운데 수로 읽지 못한 곳이 있습니다")
    for before, again in mentions:
        assert before not in NUMBER_WORDS and not before.isdigit(), f"부위 수 앞에 수가 더 있습니다: {before} {again}"
        assert again in WORDS if language == "en" else again.isdigit(), f"부위 수로 읽지 못하는 낱말입니다: {again!r}"
        assert (WORDS[again] if language == "en" else int(again)) == regions, f"첫 화면({language})이 부위를 {again} 로 셉니다"
    joints = yaml.safe_load((REPO / "meta/joint_map/unitree_h1.yaml").read_text(encoding="utf-8"))["joint_names"]
    assert int(only(dof_said, body, f"첫 화면({language}) · 로봇 행").group(1)) == len(joints)
    assert only(schema_said, body, f"첫 화면({language}) · 형식 행").group(1) == canon("safety_schema.json")["schema_version"]


NEAR_MISS_ROW = {
    "en": r"\| `near_miss` \| No contact, but the robot moved toward the person faster than ([\d.]+) (\S+) "
          r"\(a self-set threshold\) while closer than its protective separation distance ",
    "ko": r"\| `near_miss` \| 접촉은 없지만, 로봇이 ([\d.]+) (\S+)\(자체 설정 값\)보다 빠르게 사람 쪽으로 움직이는 동안 보호 이격거리",
}


@pytest.mark.parametrize("language", sorted(PAGES))
def test_the_near_miss_row_keeps_the_canons_approach_threshold(language):
    """라벨 표의 근접사고 행 ↔ 정본. 보호 이격거리 안에 있는 것만으로는 근접사고가 아니다 — 로봇이 접근 속도 문턱보다
    빠르게 다가와야 하고, 그 문턱은 자체 설정 값이다."""
    rule = next(r for r in canon("rules.json")["rules"] if r["rule_id"] == "R-TS15066-SSM-02")
    threshold = rule["parameters"]["approach_speed_threshold"]
    said = only(NEAR_MISS_ROW[language], section(language, SECTIONS[language].index({"en": "Labels", "ko": "라벨"}[language])),
                f"첫 화면({language}) · 근접사고 행")
    assert float(said.group(1)) == threshold["value"] and said.group(2) == threshold["unit"]
    assert threshold["determined_by"] == "project", "화면이 자체 설정이라고 부르는 값의 출처가 정본에서 바뀌었습니다"


@pytest.mark.parametrize("language", sorted(PAGES))
def test_the_scenario_table_names_the_cards_scenarios(language):
    rows = re.findall(r"^\| \*\*`(\w+)`\*\* — ", section(language, 1), re.M)
    assert rows == card_scenarios(), f"첫 화면({language})의 시나리오 표는 {rows}, 상세 문서의 「한눈에」는 {card_scenarios()}"


STEP_LAYERS = {"en": {"Raw": "raw", "Derived": "processed"}, "ko": {"Raw": "raw", "가공": "processed"}}
PER_CAMERA = re.compile(r"`([\w.]+)\.<[^>]+>` \((\d+)\)")


@pytest.mark.parametrize("language", sorted(PAGES))
def test_the_step_columns_are_the_canons_layers(language):
    """접어 둔 스텝 표의 열 ↔ `step_field_layers` 를 변환기가 싣는 열로 좁힌 것 — 이름과 층, 양쪽으로.

    카메라마다 하나씩인 열은 `<camera>` 한 번으로 적혀 있다. 괄호 안의 수만큼 정본에 그 머리로 시작하는 열이 있어야 한다."""
    from generation.field_map import STEP_FIELDS

    layers = canon("safety_schema.json")["step_field_layers"]
    carried = {field.schema for field in STEP_FIELDS}
    want = {name: layer for name, layer in layers.items() if name in carried}
    words = STEP_LAYERS[language]
    rows = re.findall(r"^\| (%s) — [^|]*\|(.*)\|$" % "|".join(map(re.escape, words)), section(language, 3), re.M)
    assert sorted(word for word, _ in rows) == sorted(words), f"첫 화면({language})의 스텝 표를 파싱하지 못했습니다"
    drawn = {}
    for word, cells in rows:
        for prefix, count in PER_CAMERA.findall(cells):
            expanded = sorted(name for name in want if name.startswith(prefix + "."))
            assert len(expanded) == int(count), f"`{prefix}.<…>` 는 {count}개라는데 정본에는 {len(expanded)}개입니다"
            for name in expanded:
                assert name not in drawn, f"`{name}` 이 두 번 나옵니다"
                drawn[name] = words[word]
        for name in re.findall(r"`([^`<]+)`", PER_CAMERA.sub("", cells)):
            assert name not in drawn, f"`{name}` 이 두 번 나옵니다"
            drawn[name] = words[word]
    assert drawn == want, (
        f"첫 화면({language})의 스텝 표가 정본과 다릅니다 — 문서만: {sorted(set(drawn) - set(want))}, "
        f"정본만: {sorted(set(want) - set(drawn))}, 층이 다름: {sorted(k for k in set(drawn) & set(want) if drawn[k] != want[k])}")


EPISODE_TABLE = {"en": "<summary><b>Episode table</b>", "ko": "<summary><b>에피소드 표</b>"}


@pytest.mark.parametrize("language", sorted(PAGES))
def test_the_episode_columns_are_the_schemas(language):
    """접어 둔 에피소드 표가 부르는 열 ↔ 규격의 에피소드 열. 조건표 시나리오의 열(21개)은 하나도 빠지지 않아야 한다."""
    schema = canon("safety_schema.json")
    table = schema["scenario_conditional_episode_fields"]["conditions"]
    columns = (set(schema["required_episode_fields"]) | set(table)
               | {name for names in schema["conditional_required_episode_fields"].values() for name in names})
    body = section(language, 3)
    body = body[body.index(EPISODE_TABLE[language]):]
    body = body[:body.index("</details>")]
    named = set()
    for prefix, count in PER_CAMERA.findall(body):
        expanded = {name for name in columns if name.startswith(prefix + ".")}
        assert len(expanded) == int(count), f"`{prefix}.<…>` 는 {count}개라는데 규격에는 {len(expanded)}개입니다"
        named |= expanded
    named |= set(re.findall(r"`([^`<]+)`", PER_CAMERA.sub("", body)))
    assert named <= columns, f"첫 화면({language})의 에피소드 표가 규격에 없는 열을 부릅니다: {sorted(named - columns)}"
    assert set(table) <= named, f"조건표 시나리오의 열 가운데 첫 화면({language})에 없는 것: {sorted(set(table) - named)}"


def front_tree(language: str) -> str:
    return fenced(section(language, 3))[0]


@pytest.mark.parametrize("language", sorted(PAGES))
def test_the_front_tree_draws_what_a_release_carries(language):
    """첫 화면의 구성 트리 ↔ 배포본에 동봉되는 파일. 화면이 바로 위에 보여 주는 그림만 빼고 모두 이름으로 나와야 하고,
    `ext` 표시의 수는 상세 문서의 트리와 같다."""
    drawn = front_tree(language)
    carried = (bundled_paths() | release_doc_paths()) - {"docs/media/teaser.png"}
    missing = sorted(carried - tree_names(drawn))
    assert not missing, f"배포본에 동봉하는데 첫 화면({language})의 트리가 그리지 않는 파일: {missing}"
    marks = len(re.findall(r"\bext\b", drawn))
    assert marks == len(re.findall(r"\bext\b", reference_tree(language))), "ext 표시의 수가 상세 문서의 트리와 다릅니다"


def test_the_two_front_trees_draw_the_same_names():
    """두 화면의 구성 트리는 같은 이름을 그린다 — 설명만 언어가 다르다."""
    assert tree_names(front_tree("en")) == tree_names(front_tree("ko"))


def shipped_paths() -> set[str]:
    return bundled_paths() | release_doc_paths()


LINK = re.compile(r"\]\(([^)\s]+)\)")


@pytest.mark.parametrize("path", [CARD, CARD_KO], ids=["en", "ko"])
def test_relative_links_point_at_files_of_the_release(path):
    """첫 화면의 상대 링크와 그림 ↔ 배포본에 실리는 경로. 실리지 않는 경로를 가리키면 Hub 에서 죽은 링크다."""
    relative = sorted({t for t in LINK.findall(read(path)) if not t.startswith(("http://", "https://", "#"))})
    assert relative, "상대 링크를 하나도 찾지 못했습니다 — 화면의 꼴이 바뀌었습니까?"
    # The Hub writes a relative link whose last part has no extension as a folder (`/tree/main/LICENSE`), and a
    # file there answers 404 (measured on this dataset, 2026-11-20). Such a file is linked by its full blob address.
    folders = [t for t in relative if "." not in t.rsplit("/", 1)[-1]]
    assert not folders, f"{path.name} 의 상대 링크 {folders} 는 Hub 에서 폴더 주소가 되어 열리지 않습니다 — 전체 주소(/blob/main/…)로"
    dead = [t for t in relative if t not in shipped_paths()]
    assert not dead, f"{path.name} 의 상대 링크가 배포본에 없는 경로를 가리킵니다: {dead}"


@pytest.mark.parametrize("path", [CARD, CARD_KO, REFERENCE, REFERENCE_KO], ids=lambda p: p.name)
def test_hub_links_point_at_this_dataset_and_its_files(path):
    """Hub 의 전체 주소는 이 데이터셋의 것이고, 파일 보기 주소(`/blob/main/…`)는 배포본에 실리는 경로를 가리킨다."""
    for target in LINK.findall(read(path)):
        if not target.startswith("https://huggingface.co/"):
            continue
        assert target == HUB or target.startswith(HUB + "/"), f"{path.name}: 다른 리포를 가리킵니다 — {target}"
        blob = re.fullmatch(re.escape(HUB) + r"/blob/main/(.+)", target)
        if blob:
            assert blob.group(1) in shipped_paths(), f"{path.name}: 배포본에 없는 파일을 가리킵니다 — {target}"


ADDRESS = re.compile(r"https?://[^\s)\]\"'<>`|]+")
# The destination a renderer takes: of a Markdown link or picture, of an autolink, of an HTML attribute.
DESTINATION = re.compile(r"\]\(([^)\s]+)\)|<(https?://[^>\s]+)>|(?:href|src)=\"([^\"]+)\"")


@pytest.mark.parametrize("path", [CARD, CARD_KO, REFERENCE, REFERENCE_KO], ids=lambda p: p.name)
def test_every_address_is_written_in_ascii(path):
    """문서의 주소는 ASCII 로 적는다 — 배지의 한글은 퍼센트 인코딩으로. 한글을 그대로 적은 주소는 그리는 쪽이
    고쳐 줄 때만 열리고, 주소를 그대로 여는 도구는 열지 못한다."""
    text = read(path)
    addresses = ADDRESS.findall(text)
    assert addresses, f"{path.name} 에서 주소를 하나도 찾지 못했습니다"
    addresses += [d for found in DESTINATION.findall(text) for d in found if d.startswith(("http://", "https://"))]
    odd = sorted({a for a in addresses if not a.isascii()})
    assert not odd, f"{path.name} 의 주소에 ASCII 가 아닌 글자가 있습니다: {odd}"


def test_the_language_buttons_point_at_each_other():
    """두 화면과 두 상세 문서의 맨 위 버튼 — 영문과 한국어가 하나씩, 서로의 자리를 가리킨다."""
    button = r"\[!\[%s\]\(https://img\.shields\.io/badge/[^)]*\?style=for-the-badge\)\]\(([^)]+)\)"
    want = {
        CARD: (HUB, "README.ko.md"),
        CARD_KO: (HUB, "README.ko.md"),
        REFERENCE: (HUB + "/blob/main/docs/dataset-reference.md", HUB + "/blob/main/docs/dataset-reference.ko.md"),
        REFERENCE_KO: (HUB + "/blob/main/docs/dataset-reference.md", HUB + "/blob/main/docs/dataset-reference.ko.md"),
    }
    for path, (english, korean) in want.items():
        text = read(path)
        assert re.findall(button % "English", text) == [english], f"{path.name} 의 English 버튼"
        assert re.findall(button % "한국어", text) == [korean], f"{path.name} 의 한국어 버튼"


def slug(heading: str) -> str:
    """Hub 가 제목에 다는 앵커 — 소문자로, 글자 · 숫자 · 빈칸 · `-` · `_` 만 남기고, 빈칸은 `-` 로."""
    kept = "".join(c for c in heading.strip().lower() if c.isalnum() or c in " -_")
    return kept.replace(" ", "-")


@pytest.mark.parametrize("path", [CARD, CARD_KO, REFERENCE, REFERENCE_KO], ids=lambda p: p.name)
def test_anchor_links_point_at_headings_of_the_same_document(path):
    text = read(path)
    heads = {slug(h) for h in re.findall(r"^#{1,6} (.+)$", re.sub(r"^```.*?^```", "", text, flags=re.M | re.S), re.M)}
    dead = sorted({t for t in LINK.findall(text) if t.startswith("#") and t[1:] not in heads})
    assert not dead, f"{path.name} 의 앵커가 가리키는 제목이 없습니다: {dead}"


def test_the_reference_contents_list_every_section():
    """상세 문서 맨 위의 차례 ↔ 그 문서의 최상위 절, 같은 차례로."""
    for path, label in ((REFERENCE, "Contents"), (REFERENCE_KO, "차례")):
        lines = re.findall(r"^\*\*%s\*\* — (.+)$" % label, read(path), re.M)
        assert len(lines) == 1, f"{path.name} 에서 차례를 {len(lines)}개 찾았습니다"
        listed = re.findall(r"\[([^\]]+)\]\(#[^)]+\)", lines[0])
        assert listed == top_sections(path), f"{path.name} 의 차례가 절과 다릅니다: {listed}"


def code_of(text: str, opener: str) -> list[str]:
    """``opener`` 로 여는 코드 블록들 — 주석과 빈 줄을 뺀 줄."""
    blocks = re.findall(r"^```%s\n(.*?)^```" % opener, text, re.M | re.S)
    return ["\n".join(line.split("  #")[0].rstrip() for line in block.splitlines()
                      if line.strip() and not line.lstrip().startswith("#")) for block in blocks]


def test_the_examples_are_the_same_in_all_four_documents():
    """설치 · 로드 예제와 BibTeX ↔ 영문 상세 문서의 것. 네 문서가 같은 글자를 싣는다 — 한쪽만 고치면 갈린다."""
    reference = read(REFERENCE)
    loading = reference[reference.index("## Loading\n"):]
    install, load = code_of(loading, "bash")[0], code_of(loading, "python")[0]
    cite = code_of(reference, "bibtex")
    assert len(cite) == 1
    for path in (CARD, CARD_KO, REFERENCE_KO):
        text = read(path)
        assert install in code_of(text, "bash"), f"{path.name} 의 설치 명령이 영문 상세 문서와 다릅니다"
        assert load in code_of(text, "python"), f"{path.name} 의 로드 예제가 영문 상세 문서와 다릅니다"
        assert code_of(text, "bibtex") == cite, f"{path.name} 의 BibTeX 가 영문 상세 문서와 다릅니다"


def test_all_four_documents_name_one_dataset_tag():
    """첫 화면과 상세 문서가 말하는 데이터셋 태그 — 로드 예제의 `revision`, 한눈에 표, 배지, BibTeX 의 버전 — 는 하나다."""
    tags = set()
    for path in (CARD, CARD_KO, REFERENCE, REFERENCE_KO):
        text = read(path)
        places = {
            "로드 예제의 revision": re.findall(r'revision="(v[\d.]+)"', text),
            "BibTeX 의 버전": ["v" + v for v in re.findall(r"note   = \{Version ([\d.]+)\. ", text)],
            "한눈에 표의 태그": re.findall(r"(?:tag|태그) `(v[\d.]+)` \|", text),
        }
        if path in (CARD, CARD_KO):
            places["배지의 링크"] = re.findall(re.escape(HUB) + r"/tree/(v[\d.]+)", text)
        for place, said in places.items():
            assert said, f"{path.name} 의 {place} 에서 태그를 읽지 못했습니다 — 그 자리의 꼴이 바뀌었습니까?"
            tags |= set(said)
    assert len(tags) == 1, f"문서들이 데이터셋 태그를 여럿 말합니다: {sorted(tags)}"
    # A Hub address of a file or a tree names a revision too: `main`, or the one tag.
    for path in (CARD, CARD_KO, REFERENCE, REFERENCE_KO):
        revisions = set(re.findall(re.escape(HUB) + r"/(?:blob|resolve|tree)/([^/\s)]+)", read(path)))
        assert revisions <= {"main"} | tags, f"{path.name} 의 Hub 주소가 다른 revision 을 가리킵니다: {sorted(revisions)}"
    # The front pages name no other dataset tag anywhere. (The references also name the code repository's tag and
    # the release's first tag, so this net is cast over the front pages only; a GitHub address carries the code's tag.)
    for path in (CARD, CARD_KO):
        elsewhere = set(re.findall(r"(?<![\w/.-])v\d+\.\d+\.\d+", re.sub(r"https://github\.com/[^\s)]+", "", read(path))))
        assert elsewhere == tags, f"{path.name} 이 다른 태그를 말합니다: {sorted(elsewhere - tags)}"


def test_the_two_reference_trees_draw_the_same_paths():
    """상세 문서 두 벌의 구성 트리는 같은 경로를 같은 차례로 그린다 — 설명만 언어가 다르다."""
    def paths(tree: str) -> list[str]:
        return [line.split("  ")[0].strip() for line in tree.splitlines() if line.strip() and not line.startswith("    ")]
    assert paths(reference_tree("en")) == paths(reference_tree("ko"))


def test_the_korean_reference_parses_json_the_way_the_english_one_does():
    english = code_of(read(REFERENCE)[read(REFERENCE).index("### Three step columns are JSON strings\n"):], "python")[0]
    korean = code_of(read(REFERENCE_KO)[read(REFERENCE_KO).index("### 스텝 컬럼 셋은 JSON 문자열입니다\n"):], "python")[0]
    assert english == korean


def test_the_picture_is_a_png_the_release_carries():
    """두 화면이 보여 주는 그림 — 같은 경로, 배포본에 실리는 경로, 실제 PNG."""
    for path in (CARD, CARD_KO):
        shown = re.findall(r"!\[[^\]]*\]\(((?!https?://)[^)]+)\)", read(path))
        assert shown == ["docs/media/teaser.png"], f"{path.name} 의 그림: {shown}"
    data = (REPO / "docs/media/teaser.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) < 500_000, "그림이 PNG 가 아니거나 너무 큽니다"


def test_the_front_matter_reads_as_yaml_with_the_viewer_config():
    import yaml

    lines = read(CARD).splitlines()
    end = next(i for i, line in enumerate(lines[1:], 1) if line.rstrip(" \t") == "---")
    front = yaml.safe_load("\n".join(lines[1:end]))
    assert front["license"] == "cc-by-4.0" and front["language"] == ["en", "ko"]
    assert json.dumps(front["configs"]) == json.dumps([{"config_name": "default", "data_files": "data/*/*.parquet"}])


def test_each_reference_names_the_generator_commit_in_a_sentence():
    """영문 첫 화면의 `generator-commit` 표식은 주석이라 보이지 않는다 — 읽는 사람이 보는 문장은 상세 문서에 있고,
    두 언어에 한 번씩 있어야 한다(게시기는 네 문서를 통틀어 두 번을 요구한다)."""
    said = re.search(r"<!--\s*generator-commit:\s*([0-9a-f]{40})\s*-->", read(CARD)).group(1)
    for path in (REFERENCE, REFERENCE_KO):
        visible = re.sub(r"<!--.*?-->", "", read(path), flags=re.S)
        assert said in visible, f"{path.name} 의 보이는 본문에 생성 커밋 {said[:12]} 이 없습니다"
