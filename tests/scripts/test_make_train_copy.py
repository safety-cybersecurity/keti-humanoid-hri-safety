"""scripts/make_train_copy.py -- the copy of a release that LeRobot's training command takes.

The release here is the fixture generator's (camera media, the hri columns) with a
meta/stats.json laid the way the converter leaves one: the statistics of every numeric step
column, and those of a column holding non-finite values blanked by the converter's own
normalise_stats. LeRobot is not installed for these tests, so what the copy does in
lerobot-train is not tested here. What is: the copy is the release with exactly three changes,
and the release is never written.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from generation.convert_lerobot import normalise_stats
from scripts import make_train_copy as tool
from validator.make_fixture import build

REPO = Path(__file__).resolve().parents[2]
JSON_STEP_FIELDS = json.loads((REPO / "meta" / "safety_schema.json").read_text(encoding="utf-8"))[
    "json_encoded_fields"]["step"]
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
DEFAULT_STATE = "observation.state.joint_pos"


def step_files(root: Path) -> list[Path]:
    return sorted((root / "data").glob("*/*.parquet"))


def info_of(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))


def stats_of(root: Path) -> dict:
    return json.loads((root / "meta" / "stats.json").read_text(encoding="utf-8"))


def write_stats(root: Path) -> None:
    """The statistics LeRobot writes for every numeric step column, then the converter's blanking."""
    features = info_of(root)["features"]
    tables = [pq.read_table(f) for f in step_files(root)]
    stats = {}
    for name, feature in features.items():
        if feature["dtype"] in ("string", "video"):
            continue
        values = np.asarray([v for t in tables for v in t.column(name).to_pylist()], dtype=float)
        with np.errstate(invalid="ignore"):
            block = {"min": values.min(axis=0), "max": values.max(axis=0),
                     "mean": values.mean(axis=0), "std": values.std(axis=0)}
            block.update({k: np.quantile(values, q, axis=0) for k, q in QUANTILES.items()})
        stats[name] = {k: np.atleast_1d(v).tolist() for k, v in block.items()}
        stats[name]["count"] = [len(values)]
    (root / "meta" / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    normalise_stats(root)


def digest_tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def videos_of(root: Path) -> list[Path]:
    found = sorted((root / "videos").rglob("*.mp4"))
    assert found, "the fixture ships camera media -- without them the video tests say nothing"
    return found


def no_partial_left(out: Path) -> bool:
    return not out.exists() and not any(p.name.startswith(f".{out.name}.") for p in out.parent.iterdir())


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("built") / "dataset"
    build(root, n_ep=4, episodes_per_file=2, with_media=True, hri=True)
    write_stats(root)
    return root


@pytest.fixture
def release(built: Path, tmp_path: Path) -> Path:
    root = tmp_path / "release"
    shutil.copytree(built, root)
    return root


def make(release: Path, tmp_path: Path, **options) -> tuple[Path, dict]:
    out = tmp_path / "copy"
    return out, tool.make_copy(release, out, **options)


# ── the three changes ────────────────────────────────────────────


def test_the_observation_columns_holding_json_text_are_left_out(release, tmp_path):
    gone = sorted(c for c in JSON_STEP_FIELDS if c.startswith("observation."))
    kept = [c for c in JSON_STEP_FIELDS if not c.startswith("observation.")]
    assert gone and kept, "the canon names text columns of both kinds -- else this says nothing"
    out, report = make(release, tmp_path)
    for f in step_files(out):
        names = pq.ParquetFile(f).schema_arrow.names
        assert not set(gone) & set(names)
        assert set(kept) <= set(names)
    features = info_of(out)["features"]
    assert not set(gone) & set(features)
    assert set(kept) <= set(features)
    assert report["dropped_columns"] == gone


def test_every_other_step_column_comes_through_as_it_was(release, tmp_path):
    out, _ = make(release, tmp_path)
    assert [f.relative_to(out) for f in step_files(out)] == [f.relative_to(release) for f in step_files(release)]
    for src, dst in zip(step_files(release), step_files(out)):
        a, b = pq.read_table(src), pq.read_table(dst)
        assert b.num_rows == a.num_rows
        kept = [n for n in a.column_names if not (n in JSON_STEP_FIELDS and n.startswith("observation."))]
        assert b.column_names == kept + ["observation.state"]
        for name in kept:
            assert b.column(name).equals(a.column(name)), name


def test_observation_state_is_the_joint_positions_by_default(release, tmp_path):
    out, report = make(release, tmp_path)
    for f in step_files(out):
        t = pq.read_table(f)
        assert t.column("observation.state").equals(t.column(DEFAULT_STATE))
    assert info_of(out)["features"]["observation.state"] == info_of(release)["features"][DEFAULT_STATE]
    assert stats_of(out)["observation.state"] == stats_of(release)[DEFAULT_STATE]
    assert report["state"] == DEFAULT_STATE


def test_observation_state_can_be_another_step_column(release, tmp_path):
    column = "observation.state.joint_vel"
    out, report = make(release, tmp_path, state=column)
    for f in step_files(out):
        t = pq.read_table(f)
        assert t.column("observation.state").equals(t.column(column))
    assert info_of(out)["features"]["observation.state"] == info_of(release)["features"][column]
    assert stats_of(out)["observation.state"] == stats_of(release)[column]
    assert report["state"] == column


def test_the_statistics_the_release_leaves_empty_are_left_out(release, tmp_path):
    before = stats_of(release)
    empty = sorted(k for k, v in before.items() if "undefined_reason" in v)
    assert empty, "the fixture has a column the converter blanks -- else this says nothing"
    out, report = make(release, tmp_path)
    after = stats_of(out)
    assert sorted(set(before) - set(after)) == empty
    assert set(after) - set(before) == {"observation.state"}
    assert all(after[k] == before[k] for k in before if k not in empty)
    assert report["dropped_stats"] == empty


# ── what stays as it was ─────────────────────────────────────────


def test_the_release_is_left_as_it_was(release, tmp_path):
    before = digest_tree(release)
    make(release, tmp_path)
    assert digest_tree(release) == before


def test_the_rest_of_the_release_is_copied_byte_for_byte_not_linked(release, tmp_path):
    out, _ = make(release, tmp_path)
    rewritten = {Path("meta/info.json"), Path("meta/stats.json")}
    checked = 0
    for p in sorted(release.rglob("*")):
        rel = p.relative_to(release)
        if not p.is_file() or rel.parts[0] in ("data", "videos") or rel in rewritten:
            continue
        q = out / rel
        assert q.read_bytes() == p.read_bytes(), rel
        assert not os.path.samefile(p, q), rel
        checked += 1
    assert {"README.md", "NOTICE", "LICENSE"} <= {p.name for p in out.iterdir()}
    assert (out / "assets").is_dir() and (out / "meta" / "episodes").is_dir()
    assert checked > 10


def test_the_row_groups_of_each_step_file_are_kept(release, tmp_path):
    f = step_files(release)[0]
    pq.write_table(pq.read_table(f), f, row_group_size=37)
    sizes = lambda p: [pq.ParquetFile(p).metadata.row_group(i).num_rows
                       for i in range(pq.ParquetFile(p).num_row_groups)]
    assert len(sizes(f)) > 2
    out, _ = make(release, tmp_path)
    for src, dst in zip(step_files(release), step_files(out)):
        assert sizes(dst) == sizes(src)


def test_the_column_description_datasets_reads_follows_the_columns(release, tmp_path):
    # LeRobot's writer leaves the datasets library's description of the columns in each step file.
    for f in step_files(release):
        t = pq.read_table(f)
        described = {n: {"dtype": str(t.schema.field(n).type), "_type": "Value"} for n in t.column_names}
        doc = {"info": {"features": described}, "fingerprint": "f00d"}
        pq.write_table(t.replace_schema_metadata({b"huggingface": json.dumps(doc).encode()}), f)
    out, _ = make(release, tmp_path)
    for src, dst in zip(step_files(release), step_files(out)):
        before = json.loads(pq.ParquetFile(src).schema_arrow.metadata[b"huggingface"])
        after = json.loads(pq.ParquetFile(dst).schema_arrow.metadata[b"huggingface"])
        assert list(after["info"]["features"]) == pq.ParquetFile(dst).schema_arrow.names
        assert after["info"]["features"]["observation.state"] == before["info"]["features"][DEFAULT_STATE]
        assert after["fingerprint"] == before["fingerprint"]


# ── the videos ───────────────────────────────────────────────────


def test_the_videos_are_hard_links_by_default(release, tmp_path):
    out, report = make(release, tmp_path)
    videos = videos_of(release)
    for v in videos:
        assert os.path.samefile(v, out / v.relative_to(release))
    assert (report["videos_linked"], report["videos_copied"]) == (len(videos), 0)


def test_copy_videos_writes_files_of_their_own(release, tmp_path):
    out, report = make(release, tmp_path, copy_videos=True)
    videos = videos_of(release)
    for v in videos:
        w = out / v.relative_to(release)
        assert w.read_bytes() == v.read_bytes()
        assert not os.path.samefile(v, w)
    assert (report["videos_linked"], report["videos_copied"]) == (0, len(videos))


@pytest.mark.parametrize("code", [errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.EMLINK])
def test_videos_are_copied_where_the_filesystem_cannot_link_them(release, tmp_path, monkeypatch, code):
    def cannot(*args, **kwargs):
        raise OSError(code, os.strerror(code))
    monkeypatch.setattr(tool.os, "link", cannot)
    out, report = make(release, tmp_path)
    videos = videos_of(release)
    for v in videos:
        w = out / v.relative_to(release)
        assert w.read_bytes() == v.read_bytes()
        assert not os.path.samefile(v, w)
    assert (report["videos_linked"], report["videos_copied"]) == (0, len(videos))


def test_a_link_error_of_another_kind_stops_the_copy_and_leaves_nothing(release, tmp_path, monkeypatch):
    def refused(*args, **kwargs):
        raise OSError(errno.EACCES, os.strerror(errno.EACCES))
    monkeypatch.setattr(tool.os, "link", refused)
    out = tmp_path / "copy"
    with pytest.raises(OSError):
        tool.make_copy(release, out)
    assert no_partial_left(out)


# ── what is refused ──────────────────────────────────────────────


def test_an_existing_out_is_refused_and_left_alone(release, tmp_path):
    out = tmp_path / "copy"
    out.mkdir()
    (out / "keep").write_text("x")
    with pytest.raises(tool.CopyRefused, match="이미 있습니다"):
        tool.make_copy(release, out)
    assert [p.name for p in out.iterdir()] == ["keep"]


@pytest.mark.parametrize("through_a_link", [False, True])
def test_an_out_inside_the_release_is_refused(release, tmp_path, through_a_link):
    if through_a_link:
        (tmp_path / "link").symlink_to(release)
        out = tmp_path / "link" / "copy"
    else:
        out = release / "copy"
    before = digest_tree(release)
    with pytest.raises(tool.CopyRefused, match="배포본 안"):
        tool.make_copy(release, out)
    assert not (release / "copy").exists()
    assert digest_tree(release) == before


def test_a_directory_that_is_not_a_release_is_refused(tmp_path):
    (tmp_path / "plain").mkdir()
    with pytest.raises(tool.CopyRefused, match="meta/info.json"):
        tool.make_copy(tmp_path / "plain", tmp_path / "copy")
    assert no_partial_left(tmp_path / "copy")


@pytest.mark.parametrize("column, says", [
    ("observation.state.joint_angle", "선언하지 않은"),
    ("observation.contact.type", "실수"),
    ("episode_index", "실수"),
    ("observation.images.head_rgb", "실수"),
    ("safety.min_distance_human", "길이"),     # shape [1]: LeRobot reads it as a scalar
    ("timestamp", "길이"),
], ids=lambda v: v if v.isascii() else "")
def test_a_state_column_lerobot_cannot_take_is_refused(release, tmp_path, column, says):
    out = tmp_path / "copy"
    with pytest.raises(tool.CopyRefused, match=says):
        tool.make_copy(release, out, state=column)
    assert no_partial_left(out)


@pytest.mark.parametrize("blank", ["emptied", "missing"])
def test_a_state_column_without_statistics_is_refused(release, tmp_path, blank):
    column = "observation.state.joint_vel"
    stats = stats_of(release)
    if blank == "emptied":
        stats[column] = {k: (v if k == "count" else None) for k, v in stats[column].items()}
    else:
        del stats[column]
    (release / "meta" / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    out = tmp_path / "copy"
    with pytest.raises(tool.CopyRefused, match="통계"):
        tool.make_copy(release, out, state=column)
    assert no_partial_left(out)


def test_a_release_that_already_has_observation_state_is_refused(release, tmp_path):
    info = info_of(release)
    info["features"]["observation.state"] = info["features"][DEFAULT_STATE]
    (release / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    out = tmp_path / "copy"
    with pytest.raises(tool.CopyRefused, match="이미 observation.state"):
        tool.make_copy(release, out)
    assert no_partial_left(out)


def test_out_does_not_exist_until_the_copy_is_whole(release, tmp_path, monkeypatch):
    out = tmp_path / "copy"
    real, seen = tool.rewrite_steps, []

    def fails_second(*args, **kwargs):
        seen.append(os.path.lexists(out))
        if len(seen) == 2:
            raise RuntimeError("disk full")
        return real(*args, **kwargs)
    monkeypatch.setattr(tool, "rewrite_steps", fails_second)
    before = digest_tree(release)
    with pytest.raises(RuntimeError, match="disk full"):
        tool.make_copy(release, out)
    assert seen == [False, False]
    assert no_partial_left(out)
    assert digest_tree(release) == before


def test_out_does_not_exist_while_the_videos_are_placed(release, tmp_path, monkeypatch):
    out = tmp_path / "copy"
    real, seen = os.link, []

    def watched(*args, **kwargs):
        seen.append(os.path.lexists(out))
        return real(*args, **kwargs)
    monkeypatch.setattr(tool.os, "link", watched)
    tool.make_copy(release, out)
    assert seen and not any(seen)
    assert out.is_dir()


def test_an_out_that_exists_once_its_links_are_followed_is_refused(release, tmp_path):
    (tmp_path / "sub" / "dir").mkdir(parents=True)
    (tmp_path / "sub" / "copy").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "sub" / "dir")
    out = Path(f"{tmp_path / 'link'}/../copy")      # tmp/copy as written; sub/copy on disk
    assert not os.path.lexists(os.path.abspath(out)) and out.is_dir()
    with pytest.raises(tool.CopyRefused, match="이미 있습니다"):
        tool.make_copy(release, out)
    assert not any((tmp_path / "sub" / "copy").iterdir())


def test_an_out_that_exists_past_a_missing_directory_is_refused(release, tmp_path):
    (tmp_path / "copy").mkdir()
    out = Path(f"{tmp_path / 'missing'}/../copy")    # the OS cannot reach it; Path.resolve() can
    assert not os.path.lexists(out)
    with pytest.raises(tool.CopyRefused, match="이미 있습니다"):
        tool.make_copy(release, out)
    assert not any((tmp_path / "copy").iterdir())


def test_a_dangling_link_at_out_is_refused(release, tmp_path):
    out, target = tmp_path / "copy", tmp_path / "elsewhere"
    out.symlink_to(target)
    with pytest.raises(tool.CopyRefused, match="이미 있습니다"):
        tool.make_copy(release, out)
    assert out.is_symlink() and not target.exists()


@pytest.mark.parametrize("name", ["train[act]", "train*", "train?"])
def test_an_out_whose_path_the_training_command_reads_as_a_pattern_is_refused(release, tmp_path, name):
    out = tmp_path / name
    with pytest.raises(tool.CopyRefused, match="패턴"):
        tool.make_copy(release, out)
    assert not out.exists()


def test_an_out_whose_resolved_path_holds_a_pattern_is_refused(release, tmp_path):
    (tmp_path / "train[act]").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "train[act]")
    out = tmp_path / "alias" / "copy"
    with pytest.raises(tool.CopyRefused, match="패턴"):
        tool.make_copy(release, out)
    assert not any((tmp_path / "train[act]").iterdir())


def test_an_out_inside_a_directory_the_release_links_to_is_refused(release, tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    shutil.move(release / "meta", store / "meta")
    (release / "meta").symlink_to(store / "meta")
    before = digest_tree(store)
    for out in (store / "meta" / "copy", release / "meta" / "copy"):
        with pytest.raises(tool.CopyRefused, match="배포본 안"):
            tool.make_copy(release, out)
    assert digest_tree(store) == before
    assert not any(p.name.startswith(".copy.") for p in (store / "meta").iterdir())


def test_a_release_without_step_files_is_refused(release, tmp_path):
    for f in step_files(release):
        f.unlink()
    out = tmp_path / "copy"
    with pytest.raises(tool.CopyRefused, match="스텝 파일"):
        tool.make_copy(release, out)
    assert no_partial_left(out)


# ── releases laid out with links ─────────────────────────────────


def as_hub_cache(release: Path, blobs: Path) -> None:
    """Lay the release as huggingface_hub's cache does: each file a relative link to a blob."""
    blobs.mkdir()
    for p in sorted(release.rglob("*")):
        if p.is_file() and not p.is_symlink():
            blob = blobs / hashlib.sha256(p.read_bytes()).hexdigest()
            if not blob.exists():
                shutil.move(p, blob)
            else:
                p.unlink()
            p.symlink_to(os.path.relpath(blob, p.parent))


def test_a_release_laid_as_the_hub_cache_lays_it(release, tmp_path):
    expected_steps = [f.relative_to(release) for f in step_files(release)]
    as_hub_cache(release, tmp_path / "blobs")
    before = digest_tree(release)
    out = tmp_path / "elsewhere" / "copy"
    out.parent.mkdir()
    report = tool.make_copy(release, out)
    assert [f.relative_to(out) for f in step_files(out)] == expected_steps
    for p in sorted(out.rglob("*")):
        assert not p.is_symlink(), p
    for v in videos_of(release):
        w = out / v.relative_to(release)
        assert os.path.samefile(w, v.resolve())
    assert report["videos_linked"] == len(videos_of(release))
    assert (out / "README.md").read_bytes() == (release / "README.md").read_bytes()
    assert digest_tree(release) == before
    assert all(p.is_symlink() for p in release.rglob("*") if not p.is_dir())


def test_chunk_directories_that_are_links_are_followed(release, built, tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    linked = [release / "data" / "chunk-000"] + sorted((release / "videos").glob("*/chunk-000"))
    for i, d in enumerate(linked):
        shutil.move(d, store / str(i))
        d.symlink_to(store / str(i))
    out, report = make(release, tmp_path)
    assert [f.relative_to(out) for f in step_files(out)] == [f.relative_to(built) for f in step_files(built)]
    assert sorted(v.relative_to(out) for v in (out / "videos").rglob("*.mp4")) == sorted(
        v.relative_to(built) for v in (built / "videos").rglob("*.mp4"))
    assert report["step_files"] == len(step_files(built))
    for f in step_files(out):
        names = pq.ParquetFile(f).schema_arrow.names
        assert "observation.state" in names and "observation.contact.type" not in names


@pytest.mark.parametrize("shape", ["a link back up", "two links to one directory"])
def test_a_directory_reached_twice_is_refused(release, tmp_path, shape):
    if shape == "a link back up":
        (release / "data" / "back").symlink_to(release / "data")
    else:
        (release / "data" / "again").symlink_to(release / "data" / "chunk-000")
    out = tmp_path / "copy"
    with pytest.raises(tool.CopyRefused, match="두 번"):
        tool.make_copy(release, out)
    assert no_partial_left(out)


def test_a_release_of_read_only_files_and_directories_is_copied(release, tmp_path):
    modes = {}
    for p in sorted(release.rglob("*"), reverse=True):     # files before the directories holding them
        modes[p] = p.stat().st_mode
        p.chmod(0o555 if p.is_dir() else 0o444)
    modes[release] = release.stat().st_mode
    release.chmod(0o555)
    try:
        before = digest_tree(release)
        out, report = make(release, tmp_path)
        assert digest_tree(release) == before
        assert report["step_files"] == len(step_files(release))
        assert "observation.state" in info_of(out)["features"]
        linked = {v.relative_to(release) for v in videos_of(release)}
        for p in out.rglob("*"):
            if p.is_file() and p.relative_to(out) not in linked:
                assert os.access(p, os.W_OK), p            # the bytes are copied, not the mode
        shutil.rmtree(out)                                 # a copy its owner can remove
    finally:
        for p, mode in modes.items():
            p.chmod(mode)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a directory whatever its mode")
def test_a_directory_the_copy_cannot_list_stops_it_before_anything_is_written(release, tmp_path, monkeypatch):
    hidden = release / "assets" / "unitree_h1"
    hidden.chmod(0o000)
    built = []
    monkeypatch.setattr(tool, "_build", lambda *args: built.append(1))
    try:
        out = tmp_path / "copy"
        with pytest.raises(PermissionError):
            tool.make_copy(release, out)
        assert not built and no_partial_left(out)
    finally:
        hidden.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a directory whatever its mode")
def test_the_copying_walk_does_not_pass_over_a_directory_it_cannot_list(tmp_path):
    src = tmp_path / "src"
    (src / "hidden").mkdir(parents=True)
    (src / "hidden" / "f").write_text("x")
    (src / "hidden").chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            tool._copy_tree(src, tmp_path / "dst", shutil.copyfile)
    finally:
        (src / "hidden").chmod(0o755)


# ── what the copy derives from the release ───────────────────────


def test_a_text_column_the_release_declares_is_left_out_whatever_its_name(release, tmp_path):
    info = info_of(release)
    for name in ("observation_note", "safety.note"):
        info["features"][name] = {"dtype": "string", "shape": [1], "names": None}
    (release / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    for f in step_files(release):
        t = pq.read_table(f)
        for name in ("observation_note", "safety.note"):
            t = t.append_column(name, pa.array(["x"] * t.num_rows))
        pq.write_table(t, f)
    out, report = make(release, tmp_path)
    assert "observation_note" in report["dropped_columns"]
    assert "safety.note" not in report["dropped_columns"]
    for f in step_files(out):
        names = pq.ParquetFile(f).schema_arrow.names
        assert "observation_note" not in names and "safety.note" in names
    features = info_of(out)["features"]
    assert "observation_note" not in features and "safety.note" in features


def test_a_statistics_block_holding_a_null_is_left_out_without_a_reason(release, tmp_path):
    stats = stats_of(release)
    stats["safety.min_distance_human"]["q50"] = None
    (release / "meta" / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    out, report = make(release, tmp_path)
    assert "safety.min_distance_human" in report["dropped_stats"]
    assert "safety.min_distance_human" not in stats_of(out)


def test_info_json_keeps_every_other_key(release, tmp_path):
    out, _ = make(release, tmp_path)
    before, after = info_of(release), info_of(out)
    assert {k: v for k, v in after.items() if k != "features"} == {k: v for k, v in before.items() if k != "features"}
    assert list(after) == list(before)


def test_other_files_under_data_come_along(release, tmp_path):
    (release / "data" / "README.txt").write_text("top")
    (release / "data" / "chunk-000" / "notes.txt").write_text("inside")
    out, _ = make(release, tmp_path)
    assert (out / "data" / "README.txt").read_text() == "top"
    assert (out / "data" / "chunk-000" / "notes.txt").read_text() == "inside"


# ── the command ──────────────────────────────────────────────────


def test_the_command_makes_the_copy_and_says_what_it_changed(release, tmp_path, capsys):
    out = tmp_path / "copy"
    assert tool.main([str(release), str(out)]) == 0
    said = capsys.readouterr().out
    assert info_of(out)["features"]["observation.state"]
    for name in ["observation.contact.type", "observation.contact.region_names", DEFAULT_STATE,
                 "safety.stability_margin"]:
        assert name in said
    assert "같은 파일" in said            # the hard links are named as the release's own files


def test_the_command_does_not_warn_about_links_it_did_not_make(release, tmp_path, capsys):
    assert tool.main([str(release), str(tmp_path / "copy"), "--copy-videos"]) == 0
    assert "같은 파일" not in capsys.readouterr().out


def test_the_command_says_why_it_refused(release, tmp_path, capsys):
    assert tool.main([str(release), str(release / "copy")]) == 2
    assert "배포본 안" in capsys.readouterr().err
