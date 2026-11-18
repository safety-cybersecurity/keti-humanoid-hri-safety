"""The walking policy and behaviour layer a C9 rerun of a table scenario's episode is given: read off
the published episode row (``sim_backend.policy``, as behavior.policy_record writes it) and checked
before the simulator boots -- a checkpoint the rerun cannot get back is refused there, not after it, and so is
any file the rerun would run or load that git does not hold in the repository, by the path it is named by, with the
bytes of HEAD.

Every test but one runs against a checkout made for it (tests/generation/committed_repo.py): the one on this
repository's own checkout is red while generation/behavior.py has changes not committed, as a rerun from such a
tree would be refused."""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from generation import behavior, conditions
from generation.convert_lerobot import episode_column
from generation.rerun_policy import module_files, published_policy, rerun_flags
from tests.generation.committed_repo import commit_all as _commit
from tests.generation.committed_repo import committed_copy
from tests.generation.committed_repo import git as _git

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "generation" / "rerun_policy.py"
TABLE = conditions.table_scenarios()[0]
EPISODE = f"{TABLE}_0003_seed7"
BACKEND = {"isaacsim": "6.0.1", "isaaclab": "v3.0.0", "physics": "physx", "lerobot": "0.6.1",
           "generator_commit": "a" * 40, "generator_dirty": False}


def _published(tmp_path, checkpoint=None, spec=behavior.DEFAULT_BEHAVIOR):
    """What policy_record writes for the published checkpoint -- the record of the run that named none."""
    checkpoint = checkpoint or tmp_path / "published.pt"
    if not checkpoint.exists():
        checkpoint.write_bytes(b"published weights")
    return behavior.policy_record(checkpoint, published=True, behavior_spec=spec, repo_root=tmp_path)


@pytest.fixture(scope="module")
def copy_root(tmp_path_factory):
    """This repository's generation/, meta/ and LICENSE as the working tree holds them, committed in a checkout of
    their own: what the rerun reads of a repository, whatever is uncommitted here."""
    return committed_copy(tmp_path_factory.mktemp("copy"), "generation", "meta", "LICENSE")


def _repository_file(tmp_path, data=b"weights"):
    """A repository -- a git checkout with everything committed -- holding a checkpoint ckpt/a.pt and a behaviour
    module mod.py, and the record of a run on them. It ignores .tmp/, as this repository does."""
    repo = tmp_path / "repo"
    (repo / "ckpt").mkdir(parents=True)
    (repo / "ckpt" / "a.pt").write_bytes(data)
    (repo / "mod.py").write_text("def Name(inp):\n    return inp.task_command\n", encoding="utf-8")
    (repo / ".gitignore").write_text(".tmp/\n", encoding="utf-8")
    _commit(repo)
    record = behavior.policy_record(repo / "ckpt" / "a.pt", published=False, behavior_spec="mod:Name",
                                    repo_root=repo)
    assert record["source"] == "file:ckpt/a.pt"
    return repo, record


def _table(root, rows):
    """meta/episodes as the converter writes it: episode_id, and sim_backend as JSON text."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pd.DataFrame([{"episode_index": i, "episode_id": eid,
                   "sim_backend": episode_column(dict(BACKEND, **({"policy": p} if p is not None else {})))}
                  for i, (eid, p) in enumerate(rows)]).to_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return root


def _cli(root, episode_id, script=SCRIPT):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, str(script), "--published", str(root), "--episode-id", episode_id],
                          cwd="/", env=env, capture_output=True, text=True, timeout=120)


def test_no_policy_means_no_flags(tmp_path):
    assert rerun_flags(None, repo_root=tmp_path) == []


def test_the_published_policy_reruns_with_its_behaviour_only(tmp_path, copy_root):
    # The published checkpoint is fetched by the run itself (no --policy); its hash is the rerun's to record,
    # and C9 compares the recorded policy, hash included.
    record = _published(tmp_path)
    assert record["source"] == f"isaaclab-published:rsl_rl/{behavior.PUBLISHED_POLICY_TASK}"
    assert rerun_flags(record, repo_root=copy_root) == ["--behavior", behavior.DEFAULT_BEHAVIOR]
    repo, _ = _repository_file(tmp_path)
    assert rerun_flags(dict(record, behavior="mod:Name"), repo_root=repo) == ["--behavior", "mod:Name"]


def test_a_repository_file_reruns_from_the_repository(tmp_path):
    repo, record = _repository_file(tmp_path)
    assert rerun_flags(record, repo_root=repo) == ["--policy", str(repo / "ckpt" / "a.pt"), "--behavior", "mod:Name"]


def test_a_file_that_changed_is_refused(tmp_path):
    repo, record = _repository_file(tmp_path)
    (repo / "ckpt" / "a.pt").write_bytes(b"other weights")
    with pytest.raises(ValueError, match="sha256") as err:
        rerun_flags(record, repo_root=repo)
    assert "file:ckpt/a.pt" in str(err.value)


def _outside(tmp_path, repo, how):
    """A record whose file is not a file of the repository, though one with its bytes may exist elsewhere."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(exist_ok=True)
    (elsewhere / "a.pt").write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    if how == "recorded outside":         # what policy_record writes for a checkpoint outside the repository
        return behavior.policy_record(elsewhere / "a.pt", published=False, behavior_spec="mod:Name", repo_root=repo)
    if how == "dot-dot":
        return {"source": "file:../elsewhere/a.pt", "sha256": digest, "behavior": "mod:Name"}
    if how == "absolute":
        return {"source": f"file:{elsewhere / 'a.pt'}", "sha256": digest, "behavior": "mod:Name"}
    if how == "symlink out":
        (repo / "link.pt").symlink_to(elsewhere / "a.pt")
        return {"source": "file:link.pt", "sha256": digest, "behavior": "mod:Name"}
    if how == "empty":
        return {"source": "file:", "sha256": digest, "behavior": "mod:Name"}
    if how == "a directory":
        return {"source": "file:ckpt", "sha256": digest, "behavior": "mod:Name"}
    raise AssertionError(how)


@pytest.mark.parametrize("how, why", [
    ("recorded outside", "저장소에 그 파일이 없습니다"),
    ("dot-dot", "저장소 밖의 파일을 가리킵니다"),
    ("absolute", "저장소 안의 상대 경로가 아닙니다"),
    ("symlink out", "저장소 밖의 파일을 가리킵니다"),
    ("empty", "저장소 안의 상대 경로가 아닙니다"),
    ("a directory", "저장소에 그 파일이 없습니다"),
])
def test_a_file_outside_the_repository_is_refused(tmp_path, how, why):
    # Every one but "recorded outside" names a file whose bytes hash as recorded: the place is refused, not the
    # bytes. Each by its own check, so that dropping one is seen though another would refuse the path too.
    repo, _ = _repository_file(tmp_path)
    record = _outside(tmp_path, repo, how)
    with pytest.raises(ValueError) as err:
        rerun_flags(record, repo_root=repo)
    assert record["source"] in str(err.value) and why in str(err.value), str(err.value)


@pytest.mark.parametrize("source", [
    "isaaclab-published:rsl_rl/Isaac-Velocity-Rough-H1-v0",
    f"isaaclab-published:skrl/{behavior.PUBLISHED_POLICY_TASK}",
    f"hf:{behavior.PUBLISHED_POLICY_TASK}",
    f"isaaclab-published:rsl_rl/{behavior.PUBLISHED_POLICY_TASK} ",
])
def test_an_other_published_task_is_refused(tmp_path, source, copy_root):
    with pytest.raises(ValueError) as err:
        rerun_flags(dict(_published(tmp_path), source=source), repo_root=copy_root)
    assert repr(source) in str(err.value)


@pytest.mark.parametrize("policy", [
    "isaaclab-published:rsl_rl/Isaac-Velocity-Flat-H1-v0", ["a"], {"sha256": "0" * 64, "behavior": "mod:Name"},
    {"source": None, "sha256": "0" * 64, "behavior": "mod:Name"},
])
def test_a_policy_that_is_not_a_record_is_refused(tmp_path, policy):
    with pytest.raises(ValueError):
        rerun_flags(policy, repo_root=tmp_path)


@pytest.mark.parametrize("spec", ["nocolon", "a:b:c", ":Name", None, 5])
def test_a_behaviour_that_is_not_a_spec_is_refused_before_isaac(tmp_path, spec, copy_root):
    # parse_spec's own test of the shape, without importing the module: the rerun imports it, in Isaac.
    record = dict(_published(tmp_path), behavior=spec)
    with pytest.raises(ValueError) as err:
        rerun_flags(record, repo_root=copy_root)
    assert record["source"] in str(err.value)


def _record(kind, spec, root):
    """A record of either kind of checkpoint the rerun takes, against the repository at ``root``: the published one,
    or file:LICENSE (a file the repository holds, with its hash) -- the behaviour module is held alike for both."""
    if kind == "published":
        return {"source": f"isaaclab-published:rsl_rl/{behavior.PUBLISHED_POLICY_TASK}", "sha256": "0" * 64,
                "behavior": spec}
    return {"source": "file:LICENSE", "sha256": hashlib.sha256((root / "LICENSE").read_bytes()).hexdigest(),
            "behavior": spec}


@pytest.mark.parametrize("kind", ["published", "file"])
@pytest.mark.parametrize("spec", ["os:system", "json:loads", "numpy:array", "generation.nowhere:Name",
                                  "generation:behavior", "validator:validate_dataset"])
def test_a_behaviour_module_that_is_not_a_repository_file_is_refused(kind, spec, copy_root):
    # The rerun imports the module the distribution names: like the checkpoint, it has to be a file of the
    # repository -- an installed module, one the repository lacks or a directory without __init__.py is not.
    record = _record(kind, spec, copy_root)
    with pytest.raises(ValueError) as err:
        rerun_flags(record, repo_root=copy_root)
    assert "저장소의 파일이 아닙니다" in str(err.value) and record["source"] in str(err.value), str(err.value)


@pytest.mark.parametrize("spec", [".generation.behavior:PassThrough", "generation..behavior:PassThrough",
                                  "generation.behavior.:PassThrough", "/etc/passwd:x", "../generation/behavior:x",
                                  "generation/behavior:PassThrough", "generation. behavior:PassThrough",
                                  "generation.behavior\n:PassThrough"])
def test_an_odd_module_name_is_refused(spec, copy_root):
    with pytest.raises(ValueError, match="식별자"):
        rerun_flags(_record("published", spec, copy_root), repo_root=copy_root)


@pytest.mark.parametrize("how", ["module linked out", "directory linked out", "package linked out"])
def test_a_behaviour_module_linked_out_of_the_repository_is_refused(tmp_path, how):
    repo, _ = _repository_file(tmp_path)
    outside = tmp_path / "outside"
    (outside / "pkg").mkdir(parents=True)
    (outside / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (outside / "pkg" / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    if how == "module linked out":
        (repo / "ext.py").symlink_to(outside / "mod.py")
        spec = "ext:x"
    elif how == "directory linked out":
        (repo / "ext").symlink_to(outside)
        spec = "ext.mod:x"
    else:
        (repo / "ext").mkdir()
        (repo / "ext" / "__init__.py").symlink_to(outside / "pkg" / "__init__.py")
        spec = "ext:x"
    with pytest.raises(ValueError, match="저장소 밖"):
        rerun_flags(_record("published", spec, repo), repo_root=repo)


@pytest.mark.parametrize("kind", ["published", "file"])
def test_the_repositorys_own_behaviour_module_passes(tmp_path, kind, copy_root):
    # This repository's own checkout: the one test that reads it -- generation/behavior.py has to be committed.
    if kind == "published":
        assert rerun_flags(_record(kind, behavior.DEFAULT_BEHAVIOR, REPO_ROOT), repo_root=REPO_ROOT) == [
            "--behavior", behavior.DEFAULT_BEHAVIOR]
    # Its files as the working tree holds them, committed apart.
    record = _record(kind, behavior.DEFAULT_BEHAVIOR, copy_root)
    assert rerun_flags(record, repo_root=copy_root)[-2:] == ["--behavior", behavior.DEFAULT_BEHAVIOR]
    # A package by its __init__.py, and a module under a directory without one (as generation/ is).
    repo, _ = _repository_file(tmp_path)
    (repo / "pkg" / "sub").mkdir(parents=True)
    (repo / "pkg" / "sub" / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "pkg" / "leaf.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "nested" / "inner").mkdir(parents=True)
    for rel in ("nested/__init__.py", "nested/inner/__init__.py", "nested/inner/leaf.py"):
        (repo / rel).write_text("x = 1\n", encoding="utf-8")
    _commit(repo)
    for spec in ("pkg.sub:x", "pkg.leaf:x", "mod:Name", "nested.inner.leaf:x"):
        assert rerun_flags(_record("published", spec, repo), repo_root=repo) == ["--behavior", spec]


def test_the_files_an_import_runs_are_every_level_of_the_name(tmp_path):
    # import nested.inner.leaf runs nested/__init__.py, nested/inner/__init__.py, then the leaf -- each is held.
    # A directory without __init__.py runs nothing of its own; a module beside it is what the import takes.
    repo, _ = _repository_file(tmp_path)
    (repo / "nested" / "inner").mkdir(parents=True)
    for rel in ("nested/__init__.py", "nested/inner/__init__.py", "nested/inner/leaf.py"):
        (repo / rel).write_text("x = 1\n", encoding="utf-8")
    (repo / "plain").mkdir()
    (repo / "plain" / "leaf.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "shadow").mkdir()
    (repo / "shadow.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "shadow" / "leaf.py").write_text("x = 1\n", encoding="utf-8")
    assert module_files("nested.inner.leaf", repo) == [repo / "nested/__init__.py", repo / "nested/inner/__init__.py",
                                                       repo / "nested/inner/leaf.py"]
    assert module_files("nested.inner", repo) == [repo / "nested/__init__.py", repo / "nested/inner/__init__.py"]
    assert module_files("plain.leaf", repo) == [repo / "plain/leaf.py"]
    assert module_files("shadow.leaf", repo) == [repo / "shadow.py", repo / "shadow/leaf.py"]
    assert "저장소의 파일이 아닙니다" in module_files("plain.nowhere", repo)


def _received(repo, ignored):
    """Where a release received under the repository lies: a directory git ignores (.tmp/), or one it merely does
    not track."""
    return repo / (".tmp/received" if ignored else "received_release")


@pytest.mark.parametrize("ignored", [True, False], ids=["ignored", "untracked"])
def test_a_checkpoint_a_received_release_brought_is_refused(tmp_path, ignored):
    # The record's hash is the one of the file the same release brought: it vouches for nothing. Only a file the
    # repository's history holds is one the rerun may load.
    repo, _ = _repository_file(tmp_path)
    payload = _received(repo, ignored) / "payload.pt"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"weights that came with a release")
    relative = payload.relative_to(repo).as_posix()
    record = {"source": f"file:{relative}", "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
              "behavior": "mod:Name"}
    with pytest.raises(ValueError, match="git 이 추적하는") as err:
        rerun_flags(record, repo_root=repo)
    assert record["source"] in str(err.value) and relative in str(err.value), str(err.value)


@pytest.mark.parametrize("ignored", [True, False], ids=["ignored", "untracked"])
@pytest.mark.parametrize("kind", ["published", "file"])
def test_a_behaviour_module_a_received_release_brought_is_refused(tmp_path, ignored, kind):
    repo, record = _repository_file(tmp_path)
    where = repo / "received_release"
    if ignored:
        (repo / ".gitignore").write_text(".tmp/\nreceived_release/\n", encoding="utf-8")
        _commit(repo)
    where.mkdir()
    (where / "behavior.py").write_text("class Agent:\n    pass\n", encoding="utf-8")
    policy = dict(record if kind == "file" else _published(tmp_path), behavior="received_release.behavior:Agent")
    with pytest.raises(ValueError, match="git 이 추적하는") as err:
        rerun_flags(policy, repo_root=repo)
    assert "received_release/behavior.py" in str(err.value) and policy["source"] in str(err.value), str(err.value)


@pytest.mark.parametrize("how, why", [
    ("linked out", "저장소 밖"),
    ("not tracked", "git 이 추적하는"),
    ("a module beside the directory", "git 이 추적하는"),
])
def test_a_parent_package_file_the_import_runs_is_held_like_the_module(tmp_path, how, why):
    # import pkg.leaf runs what pkg is before pkg/leaf.py: its __init__.py -- or, beside a directory without one,
    # pkg.py. A leaf inside the repository and tracked is not enough.
    repo, record = _repository_file(tmp_path)
    (repo / "pkg").mkdir()
    (repo / "pkg" / "leaf.py").write_text("class Agent:\n    pass\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    if how == "linked out":
        (repo / "pkg" / "__init__.py").symlink_to(outside)
        _commit(repo)               # git holds the link, not the file it points at
    elif how == "not tracked":
        _commit(repo)
        (repo / "pkg" / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    else:
        _commit(repo)
        (repo / "pkg.py").write_text("x = 1\n", encoding="utf-8")
    for policy in (dict(record, behavior="pkg.leaf:Agent"), dict(_published(tmp_path), behavior="pkg.leaf:Agent")):
        with pytest.raises(ValueError, match=why) as err:
            rerun_flags(policy, repo_root=repo)
        assert policy["source"] in str(err.value), str(err.value)


@pytest.mark.parametrize("what", ["checkpoint", "module", "parent", "staged"])
def test_a_tracked_file_that_differs_from_head_is_refused(tmp_path, what):
    # Uncommitted bytes are not the repository's: a checkpoint edited and its record's hash with it, a module or
    # a package initializer edited, a file staged but never committed.
    repo, record = _repository_file(tmp_path)
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "pkg" / "leaf.py").write_text("class Agent:\n    pass\n", encoding="utf-8")
    _commit(repo)
    policy = dict(record)
    if what == "checkpoint":
        (repo / "ckpt" / "a.pt").write_bytes(b"other weights")
        policy["sha256"] = hashlib.sha256(b"other weights").hexdigest()
        changed = "ckpt/a.pt"
    elif what == "module":
        (repo / "mod.py").write_text("def Name(inp):\n    return None\n", encoding="utf-8")
        changed = "mod.py"
    elif what == "parent":
        (repo / "pkg" / "__init__.py").write_text("x = 2\n", encoding="utf-8")
        policy["behavior"] = "pkg.leaf:Agent"
        changed = "pkg/__init__.py"
    else:
        (repo / "ckpt" / "b.pt").write_bytes(b"staged weights")
        _git(repo, "add", "ckpt/b.pt")
        policy = behavior.policy_record(repo / "ckpt" / "b.pt", published=False, behavior_spec="mod:Name",
                                        repo_root=repo)
        changed = "ckpt/b.pt"
    with pytest.raises(ValueError, match="HEAD 와 다릅니다") as err:
        rerun_flags(policy, repo_root=repo)
    assert changed in str(err.value) and policy["source"] in str(err.value), str(err.value)


def test_a_repository_that_is_not_a_git_checkout_is_refused(tmp_path):
    # Without git's index nothing says which files are the repository's: refused, with what to do -- and so
    # whether or not the temporary directory lies inside another checkout.
    repo, record = _repository_file(tmp_path)
    shutil.rmtree(repo / ".git")
    for policy in (record, _published(tmp_path)):
        with pytest.raises(ValueError, match="git 체크아웃의 뿌리가 아니어서") as err:
            rerun_flags(dict(policy, behavior="mod:Name"), repo_root=repo)
        assert "git 체크아웃에서 돌리십시오" in str(err.value), str(err.value)


@pytest.mark.parametrize("outer_tracks", [False, True], ids=["outer ignores it", "outer tracks its files"])
def test_a_directory_inside_another_checkout_is_not_that_checkouts_root(tmp_path, outer_tracks):
    # git asked inside a plain directory answers for the checkout around it, whose index is not the repository's.
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "README").write_text("outer\n", encoding="utf-8")
    inner = outer / "inner"
    (inner / "ckpt").mkdir(parents=True)
    (inner / "ckpt" / "a.pt").write_bytes(b"weights")
    (inner / "mod.py").write_text("def Name(inp):\n    return inp.task_command\n", encoding="utf-8")
    if not outer_tracks:
        (outer / ".gitignore").write_text("inner/\n", encoding="utf-8")
    _commit(outer)
    record = behavior.policy_record(inner / "ckpt" / "a.pt", published=False, behavior_spec="mod:Name",
                                    repo_root=inner)
    for policy in (record, _published(tmp_path, spec="mod:Name")):
        with pytest.raises(ValueError, match="git 체크아웃의 뿌리가 아니어서") as err:
            rerun_flags(policy, repo_root=inner)
        assert str(outer.resolve()) in str(err.value), str(err.value)


def test_a_name_with_glob_characters_is_one_path_not_a_pattern(tmp_path):
    # ckpt/a?.pt would match the tracked ckpt/ab.pt as a pattern; it is the untracked file of that very name.
    repo, _ = _repository_file(tmp_path)
    (repo / "ckpt" / "ab.pt").write_bytes(b"tracked")
    _commit(repo)
    odd = repo / "ckpt" / "a?.pt"
    odd.write_bytes(b"not tracked")
    record = {"source": "file:ckpt/a?.pt", "sha256": hashlib.sha256(b"not tracked").hexdigest(), "behavior": "mod:Name"}
    with pytest.raises(ValueError, match="git 이 추적하는"):
        rerun_flags(record, repo_root=repo)


def test_the_callers_git_variables_do_not_decide(tmp_path, monkeypatch):
    # A GIT_DIR or GIT_INDEX_FILE set around the hook would have git answer for another repository.
    repo, record = _repository_file(tmp_path)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "no-such-repository"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "no-such-index"))
    assert rerun_flags(record, repo_root=repo) == ["--policy", str(repo / "ckpt" / "a.pt"), "--behavior", "mod:Name"]


def test_a_file_with_a_hash_that_is_not_text_is_refused(tmp_path):
    repo, record = _repository_file(tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        rerun_flags(dict(record, sha256=None), repo_root=repo)


def test_published_policy_reads_the_episode_row(tmp_path):
    record = _published(tmp_path)
    root = _table(tmp_path / "published", [(EPISODE, record), (f"{TABLE}_0004_seed8", None)])
    assert published_policy(root, EPISODE) == record
    assert published_policy(root, f"{TABLE}_0004_seed8") is None


@pytest.mark.parametrize("rows, why", [
    ([], "0"),
    ([(EPISODE, None), (EPISODE, None)], "2"),
])
def test_published_policy_wants_exactly_one_row(tmp_path, rows, why):
    root = _table(tmp_path / "published", rows or [(f"{TABLE}_0009_seed9", None)])
    with pytest.raises(ValueError, match=f"{why} ?건"):
        published_policy(root, EPISODE)


def test_published_policy_refuses_a_distribution_without_an_episode_table(tmp_path):
    with pytest.raises(ValueError):
        published_policy(tmp_path / "absent", EPISODE)


@pytest.mark.parametrize("cell, why", [
    ("{not json", "JSON"),
    ("[1, 2]", "객체"),
    (json.dumps(dict(BACKEND, policy="isaaclab-published")), "policy"),
])
def test_published_policy_refuses_a_sim_backend_it_cannot_read(tmp_path, cell, why):
    root = tmp_path / "published"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pd.DataFrame([{"episode_id": EPISODE, "sim_backend": cell}]).to_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    with pytest.raises(ValueError, match=why):
        published_policy(root, EPISODE)


def test_published_policy_names_a_table_without_the_column(tmp_path):
    root = tmp_path / "published"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pd.DataFrame([{"episode_id": EPISODE}]).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    with pytest.raises(ValueError, match="sim_backend"):
        published_policy(root, EPISODE)


@pytest.mark.parametrize("episode_id", ["S1d_0003_seed7", "S1a_0000_seed1"])
def test_the_cli_says_nothing_for_s1d_without_reading(tmp_path, episode_id):
    # Neither a root that is not there nor an episode table that does not parse is read for a scenario without a
    # conditions table: for S1a and S1d the hook prints no flag.
    assert episode_id.split("_")[0] not in conditions.table_scenarios()
    r = _cli(tmp_path / "absent", episode_id)
    assert (r.returncode, r.stdout, r.stderr) == (0, "", "")
    broken = tmp_path / "broken" / "meta" / "episodes"
    broken.mkdir(parents=True)
    (broken / "file-000.parquet").write_bytes(b"not a parquet file")
    r = _cli(tmp_path / "broken", episode_id)
    assert (r.returncode, r.stdout, r.stderr) == (0, "", "")


def test_the_cli_prints_one_flag_per_line(tmp_path, copy_root):
    root = _table(tmp_path / "published", [(EPISODE, _published(tmp_path))])
    r = _cli(root, EPISODE, copy_root / "generation" / "rerun_policy.py")
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["--behavior", behavior.DEFAULT_BEHAVIOR]


@pytest.mark.parametrize("policy, said", [
    pytest.param({"source": "isaaclab-published:rsl_rl/Isaac-Velocity-Rough-H1-v0", "sha256": "0" * 64,
                  "behavior": behavior.DEFAULT_BEHAVIOR}, "Isaac-Velocity-Rough-H1-v0", id="another task"),
    pytest.param({"source": "file:no/such/keti-policy.pt", "sha256": "0" * 64, "behavior": behavior.DEFAULT_BEHAVIOR},
                 "file:no/such/keti-policy.pt", id="a file the repository does not have"),
    pytest.param({"source": f"isaaclab-published:rsl_rl/{behavior.PUBLISHED_POLICY_TASK}", "sha256": "0" * 64,
                  "behavior": "os:system"}, "저장소의 파일이 아닙니다", id="a module not of the repository"),
    pytest.param({"source": f"isaaclab-published:rsl_rl/{behavior.PUBLISHED_POLICY_TASK}", "sha256": "0" * 64,
                  "behavior": "generation.behavior:Pass\nThrough"}, "Pass", id="a behaviour that is two lines"),
])
def test_the_cli_refuses_with_exit_2(tmp_path, policy, said, copy_root):
    root = _table(tmp_path / "published", [(EPISODE, policy)])
    r = _cli(root, EPISODE, copy_root / "generation" / "rerun_policy.py")
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    assert r.stdout == ""
    assert said in r.stderr, r.stderr


def test_the_cli_refuses_a_table_scenario_without_its_row(tmp_path):
    r = _cli(tmp_path / "absent", EPISODE)
    assert r.returncode == 2 and r.stdout == "", (r.returncode, r.stdout, r.stderr)


@pytest.mark.parametrize("how", ["module", "package initializer", "checkpoint"])
def test_a_link_the_repository_does_not_track_is_refused_though_its_target_is_tracked(tmp_path, how):
    # The path the loader is given is the one held to git, as it is named: an untracked link to a tracked file is
    # not a file of the repository.
    repo, record = _repository_file(tmp_path)
    (repo / "rr").mkdir()
    if how == "module":
        (repo / "rr" / "behavior.py").symlink_to("../mod.py")
        policy, named = dict(record, behavior="rr.behavior:Name"), "rr/behavior.py"
    elif how == "package initializer":
        (repo / "rr" / "leaf.py").write_text("def Name(inp):\n    return None\n", encoding="utf-8")
        _commit(repo)
        (repo / "rr" / "__init__.py").symlink_to("../mod.py")
        policy, named = dict(record, behavior="rr.leaf:Name"), "rr/__init__.py"
    else:
        (repo / "rr" / "w.pt").symlink_to("../ckpt/a.pt")
        policy, named = dict(record, source="file:rr/w.pt"), "rr/w.pt"
    with pytest.raises(ValueError, match="git 이 추적하는") as err:
        rerun_flags(policy, repo_root=repo)
    assert named in str(err.value), str(err.value)


@pytest.mark.parametrize("target, why", [
    ("../mod.py", None),
    ("../untracked.py", "git 이 추적하는"),
    ("../../outside.py", "저장소 밖"),
], ids=["to a tracked file", "to a file git does not track", "out of the repository"])
def test_a_tracked_link_is_held_by_its_target_as_well(tmp_path, target, why):
    repo, record = _repository_file(tmp_path)
    (tmp_path / "outside.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "rr").mkdir()
    (repo / "rr" / "behavior.py").symlink_to(target)
    _commit(repo)
    (repo / "untracked.py").write_text("def Name(inp):\n    return None\n", encoding="utf-8")
    policy = dict(record, behavior="rr.behavior:Name")
    if why is None:
        assert rerun_flags(policy, repo_root=repo)[-2:] == ["--behavior", "rr.behavior:Name"]
        return
    with pytest.raises(ValueError, match=why):
        rerun_flags(policy, repo_root=repo)


def test_a_tracked_link_that_does_not_lead_where_its_text_says_is_refused(tmp_path):
    # rr/behavior.py -> sub/../../mod.py reads as mod.py, a tracked file; but rr/sub is a link the repository does not
    # track, and through it the path leads to .tmp/mod.py -- a file of the receiver's.
    repo, record = _repository_file(tmp_path)
    (repo / "rr").mkdir()
    (repo / "rr" / "behavior.py").symlink_to("sub/../../mod.py")
    _commit(repo)
    (repo / ".tmp" / "a" / "b").mkdir(parents=True)
    (repo / ".tmp" / "mod.py").write_text("def Name(inp):\n    return None\n", encoding="utf-8")
    (repo / "rr" / "sub").symlink_to("../.tmp/a/b")
    assert (repo / "rr" / "behavior.py").resolve() == (repo / ".tmp" / "mod.py").resolve()
    with pytest.raises(ValueError, match="이어지지 않습니다") as err:
        rerun_flags(dict(record, behavior="rr.behavior:Name"), repo_root=repo)
    assert "rr/behavior.py" in str(err.value), str(err.value)


def test_a_tracked_file_reached_through_a_directory_that_became_a_link_is_refused(tmp_path):
    # The index still lists ckpt/a.pt, but ckpt/ is now a link to a directory git ignores, inside the repository,
    # holding a file of the same bytes: what the loader reads is not the repository's file.
    repo, record = _repository_file(tmp_path)
    elsewhere = repo / ".tmp" / "elsewhere"
    elsewhere.mkdir(parents=True)
    (elsewhere / "a.pt").write_bytes(b"weights")
    shutil.rmtree(repo / "ckpt")
    (repo / "ckpt").symlink_to(".tmp/elsewhere")
    with pytest.raises(ValueError, match="디렉터리 가운데 링크") as err:
        rerun_flags(record, repo_root=repo)
    assert "ckpt/a.pt" in str(err.value), str(err.value)


@pytest.mark.parametrize("kind", ["published", "file"])
def test_a_linked_module_whose_relative_imports_would_run_files_beside_the_link_is_refused(tmp_path, kind):
    # A link to a tracked module in a directory of its own: the import resolves the module's relative imports
    # against the link's directory, so an untracked sibling there would run. The link is not a tracked file.
    repo, record = _repository_file(tmp_path)
    (repo / "real").mkdir()
    (repo / "real" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "real" / "events.py").write_text("from .entry_geometry import x\nclass Agent:\n    pass\n",
                                             encoding="utf-8")
    (repo / "real" / "entry_geometry.py").write_text("x = 1\n", encoding="utf-8")
    _commit(repo)
    received = repo / "received_release"
    received.mkdir()
    (received / "events.py").symlink_to("../real/events.py")
    (received / "entry_geometry.py").write_text("x = 'not the repository'\n", encoding="utf-8")
    policy = dict(record if kind == "file" else _published(tmp_path), behavior="received_release.events:Agent")
    with pytest.raises(ValueError, match="git 이 추적하는") as err:
        rerun_flags(policy, repo_root=repo)
    assert "received_release/events.py" in str(err.value), str(err.value)


def _shadow_repository(tmp_path):
    """A repository with a package pkg (pkg/__init__.py, pkg/leaf.py) and a namespace level rr (rr/behavior.py,
    no __init__.py), all committed."""
    repo, record = _repository_file(tmp_path)
    (repo / "pkg").mkdir()
    (repo / "rr").mkdir()
    for rel in ("pkg/__init__.py", "pkg/leaf.py", "rr/behavior.py"):
        (repo / rel).write_text("class Agent:\n    pass\n", encoding="utf-8")
    _commit(repo)
    return repo, record


@pytest.mark.parametrize("shadow, spec", [
    ("rr/__init__.pyc", "rr.behavior:Agent"),
    ("rr/__init__.cpython-311-x86_64-linux-gnu.so", "rr.behavior:Agent"),
    ("rr.pyc", "rr.behavior:Agent"),
    ("rr/behavior.pyc", "rr.behavior:Agent"),
    ("pkg/leaf.abi3.so", "pkg.leaf:Agent"),
    ("pkg/leaf.cpython-311-x86_64-linux-gnu.so", "pkg.leaf:Agent"),
    ("pkg/__init__.cpython-312-x86_64-linux-gnu.so", "pkg.leaf:Agent"),
    ("pkg.so", "pkg.leaf:Agent"),
])
def test_a_file_the_import_could_take_in_place_of_the_tracked_source_is_refused(tmp_path, shadow, spec):
    # The import system takes an extension module before a .py and a bare .pyc where no .py is (a namespace level
    # has none at all): beside the tracked sources, a file of the same name with another suffix it loads is refused
    # -- an extension module of another interpreter's version as well.
    repo, record = _shadow_repository(tmp_path)
    (repo / shadow).write_bytes(b"\0")
    for policy in (dict(record, behavior=spec), dict(_published(tmp_path), behavior=spec)):
        with pytest.raises(ValueError, match="대신") as err:
            rerun_flags(policy, repo_root=repo)
        assert Path(shadow).name in str(err.value), str(err.value)


def test_a_bare_bytecode_initializer_is_what_the_import_runs_at_a_namespace_level(tmp_path):
    # What the refusal above rests on, measured on this interpreter: rr/ has no __init__.py, and an
    # rr/__init__.pyc beside the tracked rr/behavior.py is run when rr.behavior is imported.
    repo, _ = _shadow_repository(tmp_path)
    source = tmp_path / "init_src.py"
    marker = tmp_path / "marker"
    source.write_text(f"open({str(marker)!r}, 'w').write('ran')\n", encoding="utf-8")
    py_compile.compile(str(source), cfile=str(repo / "rr" / "__init__.pyc"), doraise=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-B", "-c", "import rr.behavior"], cwd=repo, env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert marker.read_text(encoding="utf-8") == "ran"
    with pytest.raises(ValueError, match="대신"):
        rerun_flags({"source": f"isaaclab-published:rsl_rl/{behavior.PUBLISHED_POLICY_TASK}", "sha256": "0" * 64,
                     "behavior": "rr.behavior:Agent"}, repo_root=repo)


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
@pytest.mark.parametrize("what", ["module", "checkpoint"])
def test_a_change_an_index_flag_hides_from_git_diff_is_still_refused(tmp_path, flag, what):
    # The receiver's index can be told to look away from a file; its bytes are compared with HEAD's all the same.
    repo, record = _repository_file(tmp_path)
    policy = dict(record)
    if what == "module":
        (repo / "mod.py").write_text("def Name(inp):\n    return None\n", encoding="utf-8")
        _git(repo, "update-index", flag, "mod.py")
        changed = "mod.py"
    else:
        (repo / "ckpt" / "a.pt").write_bytes(b"other weights")
        _git(repo, "update-index", flag, "ckpt/a.pt")
        policy["sha256"] = hashlib.sha256(b"other weights").hexdigest()
        changed = "ckpt/a.pt"
    hidden = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", changed],
                            env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")})
    assert hidden.returncode == 0, "git diff 가 이 변경을 보고 있습니다 — 이 시험이 아무것도 증명하지 않습니다"
    with pytest.raises(ValueError, match="HEAD 와 다릅니다") as err:
        rerun_flags(policy, repo_root=repo)
    assert changed in str(err.value), str(err.value)
