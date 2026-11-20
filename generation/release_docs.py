"""The documents a release carries beside its card.

The card, ``README.md``, is laid by the converter (``convert_lerobot.CANON_RENAMES``). The assembly lays
these next to it from the repository as it is, and the publisher holds each to the repository's bytes:
the card's Korean page, the reference the two pages point at in both languages, and the picture the
pages show. A release made by the converter alone does not have them, and the publisher refuses it.

The card's two pages link to these paths, so a path changed here is changed there too
(``tests/docs/test_card_front_page.py`` holds the links to this list).
"""
from __future__ import annotations

import shutil
from pathlib import Path

# (path in the repository, path in the release)
RELEASE_DOCS = (
    ("docs/dataset-card.ko.md", "README.ko.md"),
    ("docs/dataset-reference.md", "docs/dataset-reference.md"),
    ("docs/dataset-reference.ko.md", "docs/dataset-reference.ko.md"),
    ("docs/media/teaser.png", "docs/media/teaser.png"),
)


def lay(repo_root: Path, out: Path) -> list[str]:
    """Copy the documents from ``repo_root`` into the release at ``out``; the release paths written, in
    the list's order. A document the repository does not have stops it before anything is written."""
    repo_root, out = Path(repo_root), Path(out)
    missing = [source for source, _ in RELEASE_DOCS if not (repo_root / source).is_file()]
    if missing:
        raise FileNotFoundError("배포본에 실을 문서가 저장소에 없습니다: " + ", ".join(missing))
    for source, shipped in RELEASE_DOCS:
        target = out / shipped
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / source, target)
    return [shipped for _, shipped in RELEASE_DOCS]


def problems(repo_root: Path, dataset: Path) -> list[str]:
    """What keeps ``dataset`` from being the release of this repository's documents: one line for each
    document that is missing from it or whose bytes are not the repository's."""
    repo_root, dataset = Path(repo_root), Path(dataset)
    out = []
    for source, shipped in RELEASE_DOCS:
        if not (dataset / shipped).is_file():
            out.append(f"동봉되어야 할 파일이 없습니다: {shipped} — 조립이 싣는 문서입니다 (변환기만으로 만든 배포본에는 없습니다)")
        elif (repo_root / source).read_bytes() != (dataset / shipped).read_bytes():
            out.append(f"{shipped} 이 저장소의 {source} 과 다릅니다 — 문서를 고친 뒤 다시 조립하지 않은 배포본입니다")
    return out
