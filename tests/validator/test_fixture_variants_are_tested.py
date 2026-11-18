"""make_fixture.py 가 만드는 결함 픽스처가 전부 실제로 시험되는지.

`make validate` 의 검출 목록은 Makefile 의 `BROKEN` 변수이고, 픽스처 생성 목록은
`make_fixture.VARIANTS` 이다. 둘은 별개라서, 변형을 추가하고 Makefile 을 잊으면
**픽스처는 만들어지는데 아무도 돌려 보지 않는 상태**가 된다. 그 상태에서 CI 는
초록이고, 새 검사가 실제로 결함을 잡는지는 아무도 모른다.

시험이 무엇을 재고 무엇을 안 재는지가 갈리는 자리다. 두 목록을 여기서 묶는다.
"""

import re
from pathlib import Path

from validator.make_fixture import VARIANTS

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"


def _broken_from_makefile() -> set[str]:
    text = MAKEFILE.read_text(encoding="utf-8")
    m = re.search(r"^BROKEN\s*:=\s*((?:.*\\\n)*.*)$", text, re.MULTILINE)
    assert m, "Makefile 에서 BROKEN 정의를 찾지 못했습니다"
    return set(m.group(1).replace("\\\n", " ").split())


def test_every_generated_variant_is_in_the_makefile_list():
    generated = set(VARIANTS)
    tested = _broken_from_makefile()
    untested = sorted(generated - tested)
    assert not untested, (
        f"픽스처는 만들어지는데 `make validate` 가 돌려 보지 않는 변형: {untested} — "
        "Makefile 의 BROKEN 에 추가하십시오. 목록에 없으면 그 검사가 실제로 결함을 "
        "잡는지 아무도 확인하지 않습니다"
    )


def test_the_makefile_list_has_no_phantom_variants():
    generated = set(VARIANTS)
    tested = _broken_from_makefile()
    phantom = sorted(tested - generated)
    assert not phantom, (
        f"Makefile 이 만들어지지 않는 픽스처를 시험하려 합니다: {phantom} — "
        "make validate 가 없는 디렉터리에서 실패하고, 그 실패가 검증기 결함처럼 보입니다"
    )
