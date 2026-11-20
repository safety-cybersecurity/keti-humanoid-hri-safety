"""매니페스트에 등록된 문서가 부르는 저장소 경로가 실재하는지 본다.

**왜 필요한가.** 전주기 SOP 는 정본의 값을 복제하지 않는다 — 임계값도 통계도
적지 않고 「그 표는 규격서 §9 에」라고 가리킨다. 그래서 이 문서가 정본과 어긋날
경로는 하나뿐이다: **파일 경로**. 경로가 옮겨지면 문서는 조용히 썩고, 그 썩음을
보는 것이 아무것도 없다.

**대상은 아래 `PUBLIC_DOCUMENTS` 다.** 이 저장소가 내놓는 문서 다섯 편이고, 문서를
더하면 여기에도 적는다.

**이 시험이 보지 못하는 것 (먼저 적는다).**

  1. **`meta/` 경로.** 저장소의 `meta/` 와 배포본의 `meta/` 가 이름을 공유하고
     문서 안에 둘을 가를 표시가 없다. 실측으로 확인했다 — 데이터셋 카드의
     `meta/info.json` · `meta/stats.json` 은 **배포본 파일**이고 저장소에는
     없다. 빼도 손실이 작다: `meta/*.json` 은 코드가 읽으므로 사라지면 시험
     수십 건이 즉시 빨간불을 낸다. 조용히 썩는 것은 문서만 부르는
     `generation/*.py` · `scripts/*.py` 쪽이다.
  2. **이 저장소 밖의 문서.** 대상 문서가 아니다 — 이 시험은 거기에 닿지 않는다.
  3. **매니페스트 밖의 문서.** `README.md` 를 비롯해 대상 목록에 없는 문서. 예외
     목록을 손으로 적어 막은 것이 아니라 유도의 결과로 빠진다.
  4. **백틱 밖의 경로**와 자리표시자(`{robot_model}` · 글롭)를 포함한 경로.
  5. **경로가 존재하기만 하면 통과한다.** 그 파일이 문서가 말하는 일을 실제로
     하는지는 보지 않는다.
  6. **저장소 루트의 파일.** `README.md` · `NOTICE` · `Makefile` · `environment.lock`
     처럼 최상위 디렉터리 아래에 있지 않은 것은 추출되지 않는다.
  7. **백틱 밖.** 산문에 그냥 적은 경로는 보지 않는다. 백틱 구간 안이면 공백이
     끼어 있어도(명령줄·pytest 노드 식별자) 토큰 단위로 본다.

**그러므로 「문서가 부르는 모든 경로를 지킨다」고 말하지 말 것.** 이 시험이 지키는
것은 위 사각지대를 뺀 나머지다.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# The documents a reader of this repository gets.
PUBLIC_DOCUMENTS = (
    "docs/schema/dataset-schema-v0.2.md", "docs/scenarios/scenario-catalog-v0.1.md",
    "docs/assets/asset-license-inventory-v0.1.md", "docs/sop/data-lifecycle-sop-v0.1.md",
    "docs/dataset-card.md", "docs/dataset-card.ko.md", "docs/dataset-reference.md", "docs/dataset-reference.ko.md",
    "docs/tech-report/dataset-technical-report-v0.1.md",
)

# 저장소의 최상위 디렉터리. meta/ 는 배포본과 이름이 겹쳐 뺀다 (독스트링 1번).
REPO_TOP_LEVEL = ("docs", "generation", "validator", "scripts", "tests")

# 공개 트리에 실리지 않는 최상위 디렉터리. 문서가 그 아래를 부르면 받은 사람에게는 없는 경로다.
# 그 디렉터리가 있는 트리에서도 실패로 다룬다 — 같은 문서가 트리에 따라 다른 판정을 받지 않게.
UNPUBLISHED_TOP_LEVEL = ("internal",)

# 인라인 코드만 본다. 여는/닫는 펜스(```)는 백틱 짝을 어긋나게 해서, 그냥
# `[^`]+` 로 훑으면 펜스 뒤의 구간 경계가 통째로 밀린다 — 규격서에서 실측 11 → 0.
FENCE = re.compile(r"^\s*```")
BACKTICK_SPAN = re.compile(r"`([^`\n]+)`")
# 구간 **전체**가 경로여야 한다고 요구하면 `` `python generation/x.py --flag` `` 나
# `` `tests/x.py::test_y[param id]` `` 처럼 공백이 낀 형태가 통째로 빠진다. 그래서
# 구간을 공백으로 쪼갠 뒤 토큰마다 본다.


def path_token(top_level: tuple[str, ...]) -> re.Pattern:
    return re.compile(r"^(?:" + "|".join(top_level) + r")/\S+$")

# `meta/joint_map/{robot_model}.yaml` 같은 자리표시자와 글롭.
PLACEHOLDER_CHARS = ("{", "<", "*", "…", "?")

# 경로 뒤에 붙는 것들. 떼지 않으면 실재하는 파일을 「없다」고 집는다 — 검사가
# 자기 생산자를 거부하는 형태다.
ANCHOR = re.compile(r"#.*$")                  # `docs/x.md#section`
PYTEST_NODE = re.compile(r"::.*$")            # `tests/x.py::test_y`
LINE_SUFFIX = re.compile(r":\d+(?:-\d+)?$")   # `docs/x.md:162` · `:162-170`
TRAILING_PUNCT = re.compile(r"[,.;:)\]]+$")    # 산문에 딸려 들어온 문장부호

# 추출 규칙이 깨져도 조용히 초록이 되지 않게 하는 것은 아래 두 가지다.
#   - test_extraction_rules: 합성 입력으로 규칙 자체를 못박는다 (강한 보장)
#   - MINIMUM_TOTAL_PATHS: 말뭉치가 통째로 안 읽히는 것만 잡는 smoke 값
# 후자는 문서별·접두별 보장을 하지 못한다. 그것이 필요하면 전자를 늘린다.
MINIMUM_TOTAL_PATHS = 40          # measured 2026-10-09: the five public documents yield 60


def documents() -> tuple[str, ...]:
    return PUBLIC_DOCUMENTS


def _without_fenced_blocks(text: str) -> str:
    """펜스로 감싼 블록을 뺀다. 그 안은 인라인 코드가 아니다."""
    kept, inside = [], False
    for line in text.splitlines():
        if FENCE.match(line):
            inside = not inside
            continue
        if not inside:
            kept.append(line)
    return "\n".join(kept)


def referenced_paths(text: str, top_level: tuple[str, ...] = REPO_TOP_LEVEL) -> list[str]:
    """백틱 안에서 저장소 경로처럼 보이는 것을 뽑는다.

    떼는 순서가 있다. 앵커와 pytest 노드 식별자를 먼저 자르고, 그다음 문장부호를
    털고, 마지막에 줄번호를 뗀다 — `…md:12,` 는 쉼표를 먼저 떼야 줄번호가 끝에 온다.
    """
    found = set()
    token_of_the_repository = path_token(top_level)
    for span in BACKTICK_SPAN.findall(_without_fenced_blocks(text)):
        for token in span.split():
            if not token_of_the_repository.match(token):
                continue
            cleaned = PYTEST_NODE.sub("", ANCHOR.sub("", token))
            cleaned = LINE_SUFFIX.sub("", TRAILING_PUNCT.sub("", cleaned))
            if not cleaned or any(ch in cleaned for ch in PLACEHOLDER_CHARS):
                continue
            found.add(cleaned)
    return sorted(found)


def unpublished_paths(text: str) -> list[str]:
    """문서가 부르는 경로 가운데 공개 트리에 실리지 않는 디렉터리 아래의 것. 실재하는지는 묻지 않는다."""
    return referenced_paths(text, UNPUBLISHED_TOP_LEVEL)


def escapes_repo(rel: str) -> bool:
    """해석했을 때 저장소 밖을 가리키는가.

    `exists()` 만 보면 밖을 가리키는 경로도 참이 되어 통과한다. 문서가 저장소 밖을
    부를 이유가 없으므로 통과가 아니라 실패로 다룬다.

    **`..` 가 있다는 것만으로 판정하지 않는다.** `docs/../NOTICE` 는 `..` 를 담고도
    저장소 안에 남는다 — 그렇게 보면 멀쩡한 경로를 거부하고, 그것이 이 저장소가
    「검사가 생산자를 거부하면 안 된다」로 정리한 형태다. 해석해서 나무 안인지를 본다.
    """
    return not (REPO / rel).resolve().is_relative_to(REPO.resolve())


@pytest.mark.parametrize("rel", documents())
def test_referenced_repo_paths_exist(rel: str) -> None:
    text = (REPO / rel).read_text(encoding="utf-8")
    unpublished = unpublished_paths(text)
    assert not unpublished, (
        f"{rel} 가 공개 트리에 실리지 않는 디렉터리의 경로를 부릅니다: {unpublished}\n"
        "문서 이름으로 부르거나 공개 자리를 가리키십시오."
    )
    paths = referenced_paths(text)
    escaping = [p for p in paths if escapes_repo(p)]
    assert not escaping, (
        f"{rel} 가 저장소 밖으로 나가는 경로를 부릅니다: {escaping}"
    )
    missing = [p for p in paths if not (REPO / p).exists()]
    assert not missing, (
        f"{rel} 가 존재하지 않는 저장소 경로를 부릅니다: {missing}\n"
        "경로를 옮겼으면 문서도 함께 고치십시오."
    )


@pytest.mark.parametrize("text,expected", [
    # 뽑는다
    ("`generation/seeding.py`", ["generation/seeding.py"]),
    ("`docs/schema/dataset-schema-v0.2.md:162`", ["docs/schema/dataset-schema-v0.2.md"]),
    ("`docs/schema/dataset-schema-v0.2.md:12-20`", ["docs/schema/dataset-schema-v0.2.md"]),
    ("`docs/x.md#section`", ["docs/x.md"]),
    ("`docs/x.md:12,`", ["docs/x.md"]),
    ("`tests/x.py::test_y`", ["tests/x.py"]),
    ("`docs/../../etc/passwd`", ["docs/../../etc/passwd"]),  # 뽑되 escapes_repo 가 잡는다
    ("`docs/../NOTICE`", ["docs/../NOTICE"]),           # `..` 지만 나무 안이다
    ("`python generation/x.py --flag`", ["generation/x.py"]),   # 공백 낀 명령줄
    ("`tests/x.py::test_y[param id]`", ["tests/x.py"]),  # 파라미터 붙은 노드 식별자
    # 안 뽑는다
    ("`meta/rules.json`", []),                          # 배포본과 이름이 겹친다
    ("`README.md`", []),                                # 루트 파일
    ("`meta/joint_map/{robot_model}.yaml`", []),        # 자리표시자
    ("`meta/episodes/**.parquet`", []),                 # 글롭
    ("generation/seeding.py", []),                      # 백틱 밖
    # 펜스 뒤의 인라인 코드. 펜스를 걷어내지 않으면 백틱 짝이 어긋나 이 경로가
    # 통째로 사라진다 — 실측으로 규격서 11 → 0, 전체 53 → 25 를 낸 형태다.
    ("```\nsome code\n```\n\n`generation/seeding.py`", ["generation/seeding.py"]),
    # 펜스 **안**의 경로는 인라인 코드가 아니므로 보지 않는다.
    ("```\n`generation/seeding.py`\n```", []),
])
def test_extraction_rules(text: str, expected: list[str]) -> None:
    """말뭉치가 어떻게 바뀌어도 규칙 자체가 지켜지는지 본다."""
    assert referenced_paths(text) == expected


def test_escapes_repo_judges_by_resolution_not_by_dotdot() -> None:
    assert escapes_repo("docs/../../etc/passwd")
    assert not escapes_repo("docs/../NOTICE")           # `..` 지만 나무 안
    assert not escapes_repo("docs/schema/dataset-schema-v0.2.md")


def test_the_extraction_actually_finds_paths() -> None:
    """추출이 조용히 0 을 반환하면 위 시험은 아무것도 안 보면서 초록이 됩니다."""
    total = sum(len(referenced_paths((REPO / rel).read_text(encoding="utf-8"))) for rel in documents())
    assert total >= MINIMUM_TOTAL_PATHS, (
        f"추출된 경로가 {total} 개뿐입니다 — 추출 규칙이 깨졌을 수 있습니다."
    )


def test_a_path_under_an_unpublished_directory_is_found() -> None:
    """그 디렉터리가 이 트리에 있든 없든 같은 답을 낸다 — 경로가 실재하는지를 묻지 않는다."""
    cited = f"{UNPUBLISHED_TOP_LEVEL[0]}/plan/note.md"
    assert unpublished_paths(f"근거는 `{cited}` 에 있습니다.") == [cited]
    assert unpublished_paths(f"`python {cited} --flag`") == [cited]
    assert unpublished_paths("근거는 `docs/schema/dataset-schema-v0.2.md` 에 있습니다.") == []


def test_the_document_test_asks_about_unpublished_paths() -> None:
    """유도해 놓고 부르지 않으면 없는 것과 같다 — 호출부를 못박는다."""
    import inspect

    assert "unpublished_paths(" in inspect.getsource(test_referenced_repo_paths_exist)
