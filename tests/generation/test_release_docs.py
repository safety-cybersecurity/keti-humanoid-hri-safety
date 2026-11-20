"""The documents the assembly lays beside the card (generation.release_docs), and the publisher's hold on them."""
import inspect
from pathlib import Path

import pytest

from generation import assemble_batch as ab
from generation import publish_dataset as pub
from generation import release_docs
from generation.batch_layout import BatchError

REPO = Path(__file__).resolve().parents[2]


def test_every_document_is_in_the_repository():
    missing = [source for source, _ in release_docs.RELEASE_DOCS if not (REPO / source).is_file()]
    assert not missing, f"배포본에 실을 문서가 저장소에 없습니다: {missing}"


def test_no_document_takes_a_path_the_converter_writes():
    """The converter lays the card and the canon; a document laid over one of them would replace it in silence."""
    from generation.convert_lerobot import CANON_FILES, CANON_RENAMES, NARROWED_CANON

    shipped = [target for _, target in release_docs.RELEASE_DOCS]
    assert len(set(shipped)) == len(shipped), f"같은 자리에 두 문서를 싣습니다: {shipped}"
    taken = set(CANON_FILES) | set(NARROWED_CANON) | set(CANON_RENAMES.values())
    assert not set(shipped) & taken, f"변환기가 쓰는 자리에 문서를 싣습니다: {sorted(set(shipped) & taken)}"
    assert all(not Path(p).is_absolute() and ".." not in Path(p).parts for pair in release_docs.RELEASE_DOCS for p in pair)


def test_lay_copies_every_document_to_its_release_path(tmp_path):
    written = release_docs.lay(REPO, tmp_path)
    assert written == [target for _, target in release_docs.RELEASE_DOCS]
    for source, target in release_docs.RELEASE_DOCS:
        assert (tmp_path / target).read_bytes() == (REPO / source).read_bytes(), target


def _repo_with_documents(root: Path, leave_out: str | None = None) -> Path:
    for source, _ in release_docs.RELEASE_DOCS:
        if source != leave_out:
            (root / source).parent.mkdir(parents=True, exist_ok=True)
            (root / source).write_bytes((REPO / source).read_bytes())
    return root


@pytest.mark.parametrize("gone", [source for source, _ in release_docs.RELEASE_DOCS])
def test_lay_writes_nothing_when_a_document_is_missing(tmp_path, gone):
    repo, out = _repo_with_documents(tmp_path / "repo", leave_out=gone), tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileNotFoundError, match=gone):
        release_docs.lay(repo, out)
    assert list(out.iterdir()) == [], "문서 하나가 없는데 다른 문서를 먼저 실었습니다"


def test_the_assembly_lays_the_documents_after_the_canon(tmp_path, monkeypatch):
    """lay_release is what assemble() lays with: the converter's canon and card, then the documents."""
    assert inspect.signature(ab.assemble).parameters["lay"].default is ab.lay_release
    monkeypatch.setattr(ab, "lay_canon", lambda repo_root, out, sidecars: ["README.md"])
    out = tmp_path / "out"
    assert ab.lay_release(REPO, out, []) == ["README.md"] + [target for _, target in release_docs.RELEASE_DOCS]
    assert (out / "README.ko.md").is_file()
    with pytest.raises(BatchError, match="저장소에 없습니다"):
        ab.lay_release(_repo_with_documents(tmp_path / "repo", leave_out="docs/dataset-card.ko.md"), tmp_path / "other", [])


def test_a_release_with_every_document_has_no_problem(tmp_path):
    release_docs.lay(REPO, tmp_path)
    assert release_docs.problems(REPO, tmp_path) == []


@pytest.mark.parametrize("target", [target for _, target in release_docs.RELEASE_DOCS])
def test_a_missing_or_stale_document_is_named(tmp_path, target):
    release_docs.lay(REPO, tmp_path)
    (tmp_path / target).write_bytes((tmp_path / target).read_bytes() + b"\n")
    stale = release_docs.problems(REPO, tmp_path)
    assert len(stale) == 1 and target in stale[0] and "다릅니다" in stale[0], stale
    (tmp_path / target).unlink()
    missing = release_docs.problems(REPO, tmp_path)
    assert len(missing) == 1 and target in missing[0] and "없습니다" in missing[0], missing


def test_the_publisher_asks_for_the_documents():
    """preflight holds the documents to the repository's bytes, as it holds the card."""
    assert "problems += release_docs.problems(repo_root, dataset)" in inspect.getsource(pub.preflight)


def test_the_documents_a_reader_is_sent_to_are_the_card_and_the_laid_text():
    """The publisher counts the generator commit's sentences over these; a text document added to the list of
    laid documents is one more place a reader is sent to."""
    laid_text = {target for _, target in release_docs.RELEASE_DOCS if target.endswith(".md")}
    assert set(pub.READER_DOCUMENTS) == {"README.md"} | laid_text


def test_the_commit_sentence_may_stand_in_the_reference(tmp_path):
    """The card's pages carry the marker in a comment; the sentences a reader sees are in the references."""
    import pandas as pd

    commit = "a" * 40
    ds = tmp_path / "dist"
    (ds / "meta" / "episodes").mkdir(parents=True)
    (ds / "docs").mkdir()
    pd.DataFrame({"outcome": ["nominal"], "sim_backend": [{"generator_commit": commit, "generator_dirty": False}]}
                 ).to_parquet(ds / "meta" / "episodes" / "file-000.parquet", index=False)
    (ds / "README.md").write_text(f"<!-- generator-commit: {commit} -->\n\nfront page\n", encoding="utf-8")
    assert any("0번 나옵니다" in x for x in pub.card_generator_problems(ds, required=True))
    (ds / "docs" / "dataset-reference.md").write_text(f"made by `{commit}`\n", encoding="utf-8")
    assert any("1번 나옵니다" in x for x in pub.card_generator_problems(ds, required=True))
    (ds / "docs" / "dataset-reference.ko.md").write_text(f"<!-- {commit} -->\n", encoding="utf-8")
    assert any("1번 나옵니다" in x for x in pub.card_generator_problems(ds, required=True)), "주석 안의 해시를 셌습니다"
    (ds / "docs" / "dataset-reference.ko.md").write_text(f"생성 커밋 `{commit}`\n", encoding="utf-8")
    assert pub.card_generator_problems(ds, required=True) == []


def _docs_linked(tmp_path):
    """A release whose documents are all there, with `docs/` then turned into a link to a directory elsewhere."""
    from tests.generation.test_publish_dataset import _REPO, _min_dist

    ds = _min_dist(tmp_path, _REPO)
    release_docs.lay(_REPO, ds)
    (ds / "docs").rename(tmp_path / "docs-elsewhere")
    (ds / "docs").symlink_to(tmp_path / "docs-elsewhere", target_is_directory=True)
    return ds, _REPO


def test_a_linked_directory_hides_its_files_from_the_upload_but_not_from_the_comparison(tmp_path):
    """The gap the publisher's rule closes: the comparison reads through the link, the manifest does not descend."""
    ds, repo = _docs_linked(tmp_path)
    assert release_docs.problems(repo, ds) == []
    assert not any(path.startswith("docs/") for path in pub.file_manifest(ds))
    assert pub.linked_directories(ds) == ["docs"]


def test_the_publisher_names_a_linked_directory(tmp_path):
    """Through preflight(): a release with a linked `docs/` is not one to upload, and the line names the link."""
    from tests.generation.test_publish_dataset import FakeValidator

    ds, repo = _docs_linked(tmp_path)
    problems = pub.preflight(ds, repo, repro_hook="", log_dir=tmp_path / "logs", run=FakeValidator([0]))
    assert [p for p in problems if "배포본 안의 docs 이 디렉터리를 가리키는 심링크입니다" in p], problems


def test_the_publisher_does_not_name_a_real_directory_or_a_linked_file(tmp_path):
    """The rule must not refuse what the assembly makes (real directories) or how a downloaded release is laid
    (files that are links): both listings count a linked file as a file."""
    from tests.generation.test_publish_dataset import _REPO, FakeValidator, _min_dist

    ds = _min_dist(tmp_path, _REPO)
    release_docs.lay(_REPO, ds)
    real = tmp_path / "README.ko.md"
    (ds / "README.ko.md").rename(real)
    (ds / "README.ko.md").symlink_to(real)
    assert pub.linked_directories(ds) == []
    assert "README.ko.md" in pub.file_manifest(ds)
    problems = pub.preflight(ds, _REPO, repro_hook="", log_dir=tmp_path / "logs", run=FakeValidator([0]))
    assert not [p for p in problems if "심링크" in p], problems


def test_the_publisher_names_a_directory_it_cannot_list(tmp_path):
    """The same gap in another shape: `docs/` that may be entered but not listed. A file under it opens by its path,
    so the comparison passes; neither listing sees it."""
    import os

    from tests.generation.test_publish_dataset import _REPO, FakeValidator, _min_dist

    if os.geteuid() == 0:
        pytest.skip("root lists every directory")
    ds = _min_dist(tmp_path, _REPO)
    release_docs.lay(_REPO, ds)
    (ds / "docs").chmod(0o311)
    try:
        assert release_docs.problems(_REPO, ds) == []
        assert not any(path.startswith("docs/") for path in pub.file_manifest(ds))
        assert pub.unlisted_directories(ds) == ["docs"]
        problems = pub.preflight(ds, _REPO, repro_hook="", log_dir=tmp_path / "logs", run=FakeValidator([0]))
        assert [p for p in problems if "배포본 안의 docs 디렉터리의 목록을 읽지 못했습니다" in p], problems
    finally:
        (ds / "docs").chmod(0o755)


def test_a_release_the_assembly_would_make_has_no_directory_the_listings_miss(tmp_path):
    from tests.generation.test_publish_dataset import _REPO, _min_dist

    ds = _min_dist(tmp_path, _REPO)
    release_docs.lay(_REPO, ds)
    assert pub.linked_directories(ds) == [] and pub.unlisted_directories(ds) == []

