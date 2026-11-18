"""생성 코드의 커밋을 데이터에 남기는 경로.

sim_backend 는 시뮬레이터 · 프레임워크 · 물리엔진 · lerobot 의 버전만 남겼고 이 저장소의 커밋은
없었습니다. 기록 방식을 고쳤더니 같은 시드가 다른 궤적을 낸 적이 있어(2026-08-25) 커밋 없는 재현
주장은 반쪽입니다.
"""

import subprocess
from pathlib import Path

import pytest

from generation import provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
GOOD = "0123456789abcdef0123456789abcdef01234567"


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / ".gitignore").write_text(".tmp/\n", encoding="utf-8")
    (root / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "first")
    return root


def test_a_clean_tree_gives_its_commit_and_says_clean(repo):
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert provenance.read(repo) == {"generator_commit": head, "generator_dirty": False}
    assert provenance.problems(provenance.read(repo)) == []


def test_an_edited_tracked_file_is_dirty(repo):
    (repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    assert provenance.read(repo)["generator_dirty"] is True


def test_a_staged_change_is_dirty(repo):
    (repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    assert provenance.read(repo)["generator_dirty"] is True


def test_an_untracked_file_is_dirty(repo):
    # 추가하지 않은 새 모듈은 편집만큼 실행을 바꿉니다.
    (repo / "new_module.py").write_text("y = 1\n", encoding="utf-8")
    assert provenance.read(repo)["generator_dirty"] is True


def test_an_untracked_file_is_dirty_whatever_the_users_git_configuration_says(repo):
    # `git status` follows status.showUntrackedFiles, and with `no` a new module reads as clean. The value
    # ships in a public dataset: it must not depend on the configuration of whoever runs the batch.
    _git(repo, "config", "status.showUntrackedFiles", "no")
    (repo / "new_module.py").write_text("y = 1\n", encoding="utf-8")
    plain = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True,
                           check=True).stdout
    assert plain.strip() == "", "시험의 전제: 이 설정에서는 git 이 추적 안 된 파일을 말하지 않습니다"
    assert provenance.read(repo)["generator_dirty"] is True


def test_an_ignored_file_is_not_dirty(repo):
    # .tmp/ 의 녹화 · 로그는 무시되는 파일입니다. 이것이 더티면 배치를 돌리는 동안 늘 더티입니다.
    (repo / ".tmp").mkdir()
    (repo / ".tmp" / "episode.npz").write_bytes(b"x")
    assert provenance.read(repo)["generator_dirty"] is False


def test_outside_a_repository_both_values_are_none(tmp_path):
    assert provenance.read(tmp_path) == {"generator_commit": None, "generator_dirty": None}


def test_git_variables_of_a_hook_do_not_redirect_the_read(repo, tmp_path, monkeypatch):
    # 훅 안에서는 GIT_DIR 이 훅의 저장소를 가리킵니다. 그대로 두면 -C 를 줘도 그쪽을 읽습니다.
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "nowhere"))
    assert provenance.read(repo)["generator_dirty"] is False


@pytest.mark.parametrize("backend, phrase", [
    ({"generator_commit": GOOD, "generator_dirty": True}, "generator_dirty 가 false 가 아닙니다: True"),
    ({"generator_commit": GOOD, "generator_dirty": "false"}, "generator_dirty 가 false 가 아닙니다: 'false'"),
    ({"generator_commit": GOOD, "generator_dirty": 0}, "generator_dirty 가 false 가 아닙니다: 0"),
    ({"generator_commit": GOOD}, "generator_dirty 가 false 가 아닙니다: None"),
    ({"generator_commit": GOOD[:7], "generator_dirty": False}, "40자 16진 커밋 해시가 아닙니다"),
    ({"generator_commit": GOOD.upper(), "generator_dirty": False}, "40자 16진 커밋 해시가 아닙니다"),
    ({"generator_commit": GOOD + "0", "generator_dirty": False}, "40자 16진 커밋 해시가 아닙니다"),
    ({"generator_commit": None, "generator_dirty": None}, "40자 16진 커밋 해시가 아닙니다"),
    ({"generator_dirty": False}, "40자 16진 커밋 해시가 아닙니다"),
    ("physx", "sim_backend 가 객체가 아닙니다"),
    (None, "sim_backend 가 객체가 아닙니다"),
])
def test_a_backend_that_does_not_name_committed_code_is_refused(backend, phrase):
    said = provenance.problems(backend)
    assert said and any(phrase in s for s in said), said


def test_a_backend_that_names_committed_code_has_no_problem():
    assert provenance.problems({"isaacsim": "6.0.1", "generator_commit": GOOD, "generator_dirty": False}) == []


# ── 배선: 기록기는 Isaac 을 import 하므로 호출부를 원문에서 못박는다 ──
# 각 줄을 빼도 다른 시험은 초록이다. 값 자체는 실제 Isaac 으로 돌린 리허설의 사이드카가 보여 준다.

import ast

RUN_S1A = REPO_ROOT / "generation/envs/run_s1a.py"
RECORDER = REPO_ROOT / "generation/envs/s1a/recorder.py"


def _sidecar_required_keys():
    tree = ast.parse(RECORDER.read_text(encoding="utf-8"))
    (node,) = [n for n in tree.body if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "SIDECAR_REQUIRED_KEYS" for t in n.targets)]
    return ast.literal_eval(node.value)


def test_the_generator_adds_the_commit_to_the_backend_it_hands_the_recorder():
    src = RUN_S1A.read_text(encoding="utf-8")
    assert "from generation import provenance" in src
    assert 'sim_backend = dict(lock["sim_backend"], **provenance.read(REPO_ROOT))' in src
    assert 'sim_backend = lock["sim_backend"]\n' not in src      # the bare lock would carry no commit


def test_the_generator_tells_the_recorder_where_the_run_started():
    src = RUN_S1A.read_text(encoding="utf-8")
    assert "run_start_seq=args.seq," in src


def test_the_recorder_writes_the_runs_first_seq_into_the_sidecar():
    src = RECORDER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    (init,) = [n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body
               if isinstance(n, ast.FunctionDef) and n.name == "__init__"
               and "run_start_seq" in [a.arg for a in n.args.kwonlyargs]]
    kwonly = {a.arg: d for a, d in zip(init.args.kwonlyargs, init.args.kw_defaults)}
    assert kwonly["run_start_seq"] is None                          # keyword-only, no default: a caller has to say it
    assert "self.run_start_seq = run_start_seq" in src
    assert '"run_start_seq": self.run_start_seq,' in src
    assert "run_start_seq" in _sidecar_required_keys()


# ── 검증기의 사본: validator/ 는 generation/ 을 import 하지 않으므로 같은 규칙을 따로 든다 ──

import validator.validate_dataset as vd

BACKENDS = [
    {"generator_commit": GOOD, "generator_dirty": False},
    {"generator_commit": GOOD, "generator_dirty": True},
    {"generator_commit": GOOD, "generator_dirty": "false"},
    {"generator_commit": GOOD, "generator_dirty": 0},
    {"generator_commit": GOOD},
    {"generator_commit": GOOD[:7], "generator_dirty": False},
    {"generator_commit": GOOD.upper(), "generator_dirty": False},
    {"generator_commit": GOOD + "\n", "generator_dirty": False},
    {"generator_commit": None, "generator_dirty": None},
    {},
]


def test_the_validator_holds_the_generators_commit_pattern():
    assert vd.GENERATOR_COMMIT_RE.pattern == provenance.COMMIT_RE.pattern


@pytest.mark.parametrize("backend", BACKENDS, ids=repr)
def test_the_validator_and_the_converter_refuse_the_same_backends(backend):
    # 한쪽만 고치면 변환기는 거부하는 녹화를 배포 검증기가 통과시킨다 — 또는 그 반대.
    assert bool(vd.generator_provenance_problems(backend)) == bool(provenance.problems(backend))


def test_the_fixture_carries_what_a_distribution_carries():
    from validator import make_fixture
    assert provenance.problems({"generator_commit": make_fixture.FIXTURE_GENERATOR_COMMIT,
                                "generator_dirty": False}) == []
