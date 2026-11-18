"""배포 스크립트의 순수함수. HF 를 부르지 않으므로 .venv 에서 돈다.

8단계에서 확인한 실패 양식 둘을 시험으로 고정한다 — 태그 이름의 출처가 비어
있는 것, 그리고 왕복에서 파일이 바뀌었는데 못 알아채는 것.
"""

import json
from pathlib import Path

import pytest

from generation.publish_dataset import (
    codebase_version,
    file_manifest,
    manifest_problems,
)


def _dataset(tmp_path, version="v3.0"):
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    info = {"codebase_version": version, "fps": 50} if version is not None else {"fps": 50}
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "README.md").write_text("card", encoding="utf-8")
    return root


def test_the_tag_name_comes_from_info_json(tmp_path):
    assert codebase_version(_dataset(tmp_path)) == "v3.0"


def test_a_missing_codebase_version_says_why_it_matters(tmp_path):
    # 이 값이 없으면 태그를 못 만들고, 태그가 없으면 LeRobot 이 Hub 에서
    # 데이터셋을 열지 못한다. 조용히 넘어가면 공개일에 발견된다.
    with pytest.raises(ValueError, match="codebase_version"):
        codebase_version(_dataset(tmp_path, version=None))


def test_a_missing_info_json_is_not_a_silent_empty_tag(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        codebase_version(empty)


def test_the_manifest_covers_nested_files(tmp_path):
    root = _dataset(tmp_path)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"x")
    m = file_manifest(root)
    assert set(m) == {"meta/info.json", "README.md", "data/chunk-000/file-000.parquet"}


def test_the_manifest_skips_the_downloaders_work_directory(tmp_path):
    root = _dataset(tmp_path)
    (root / ".cache" / "huggingface").mkdir(parents=True)
    (root / ".cache" / "huggingface" / "x.lock").write_bytes(b"")
    assert ".cache/huggingface/x.lock" not in file_manifest(root, skip=(".cache",))


def test_an_identical_round_trip_reports_nothing():
    m = {"a": "1", "b": "2"}
    assert manifest_problems(m, dict(m)) == []


def test_the_hub_adding_gitattributes_is_not_a_problem():
    assert manifest_problems({"a": "1"}, {"a": "1", ".gitattributes": "z"}) == []


def test_a_changed_file_is_reported():
    problems = manifest_problems({"a": "1"}, {"a": "changed"})
    assert problems == ["내용 다름: a"]


def test_a_missing_file_is_reported():
    assert manifest_problems({"a": "1", "b": "2"}, {"a": "1"}) == ["누락: b"]


def test_a_file_we_never_uploaded_is_reported():
    problems = manifest_problems({"a": "1"}, {"a": "1", "stray.txt": "z"})
    assert problems == ["올린 적 없는 파일: stray.txt"]


# ── publish() 의 실패 경로 ────────────────────────────────────────
#
# 실제 Hub 에 올려 보지 않고는 밟을 수 없는 경로라 가짜 API 로 시험한다.
#
# 가짜를 손으로 쓰면 **코드가 가진 가정을 시험이 그대로 물려받아 둘이 함께 틀린
# 채로 초록**이 될 수 있다. 실제로 그랬다 —
# 다운로드를 호출 기록에 남기지 않아, 확인을 태그 뒤로 되돌려도 순서 단언이
# 그대로 통과했다. 그래서 아래 가짜는 세 가지를 실물에 맞춘다.
#
#   - 다운로드도 같은 호출 기록에 남긴다 (순서를 진짜로 재게)
#   - 태그는 이름만이 아니라 `target_commit` 을 가진다 (GitRefInfo 와 같게)
#   - 없는 태그 삭제·있는 태그 중복 생성은 예외를 던진다 (Hub 처럼)

import types


class FakeHubError(RuntimeError):
    """Hub 이 4xx 를 돌려줄 자리. 가짜가 조용히 성공하지 않게 한다."""


class FakeApi:
    def __init__(self, private=True, tags=None, flip_private_on_upload=False,
                 create_tag_raises=False, create_tag_applies_then_raises=False,
                 refs_raises_from=None, oid="a" * 40, calls=None,
                 resolves_to=None, remote_files=None, on_create_tag=None,
                 exists=True, create_takes_the_name=False):
        self._private = private
        # 이름 → target_commit. 실물 GitRefInfo 가 그 둘을 준다.
        self.tags = dict(tags or {})
        self._flip = flip_private_on_upload
        self._create_tag_raises = create_tag_raises
        # 실물 POST 는 서버가 적용한 **뒤** 응답이 유실될 수 있다. 예외가
        # "안 만들어졌다" 는 뜻이 아니라는 것이 이 옵션의 요지다.
        self._create_applies_then_raises = create_tag_applies_then_raises
        self._refs_raises_from = refs_raises_from
        self._refs_calls = 0
        self.calls = calls if calls is not None else []
        self.uploaded_oid = oid
        # A renamed repository's old name answers for the repository at its new
        # name. `resolves_to` is that answer; None means the name is its own.
        self._resolves_to = resolves_to
        # Whether the name answers at all. A renamed repository's old name does.
        self._exists = exists
        # What the Hub does with create_repo on a name that only redirects: it
        # makes a new repository there and the redirect is gone (measured on the
        # Hub 2026-10-06 and 2026-11-08). False keeps the redirect, as a refused
        # creation that the client swallows does.
        self._create_takes_the_name = create_takes_the_name
        self.asked_exists = []
        self.remote_files = list(remote_files or [])
        # The tags asked for, in order — names alone cannot tell which came first.
        self.tag_log = []
        # Called with (api, tag) right after a tag was created: what another
        # writer does to the repository between two of our requests.
        self._on_create_tag = on_create_tag

    def repo_exists(self, **kw):
        self.calls.append("repo_exists")
        # Which name of which kind was asked: a dataset name answers only when it is
        # asked as a dataset (a model of that name is another repository).
        self.asked_exists.append((kw.get("repo_type"), kw.get("repo_id")))
        return self._exists and kw.get("repo_type") == "dataset"

    def create_repo(self, **kw):
        self.calls.append("create_repo")
        if self._create_takes_the_name:
            self._resolves_to = None
        self._exists = True

    def repo_info(self, **kw):
        self.calls.append("repo_info")
        if not self._exists:
            raise FakeHubError("404 repository not found")
        return types.SimpleNamespace(private=self._private,
                                     id=self._resolves_to or kw.get("repo_id"))

    def list_repo_files(self, **kw):
        self.calls.append("list_repo_files")
        return list(self.remote_files)

    def upload_folder(self, **kw):
        self.calls.append("upload_folder")
        if self._flip:
            self._private = not self._private
        return types.SimpleNamespace(oid=self.uploaded_oid)

    def list_repo_refs(self, *a, **kw):
        self.calls.append("list_repo_refs")
        self._refs_calls += 1
        if self._refs_raises_from is not None and self._refs_calls >= self._refs_raises_from:
            raise FakeHubError("500 refs unavailable")
        return types.SimpleNamespace(tags=[
            types.SimpleNamespace(name=n, target_commit=c)
            for n, c in self.tags.items()])

    def delete_tag(self, repo_id, tag, **kw):
        self.calls.append("delete_tag")
        if tag not in self.tags:
            raise FakeHubError(f"404 tag not found: {tag}")
        del self.tags[tag]

    def create_tag(self, repo_id, tag, revision=None, **kw):
        self.calls.append("create_tag")
        self.tag_log.append(tag)
        if self._create_tag_raises:
            raise FakeHubError("503 upstream unavailable")
        if tag in self.tags:
            raise FakeHubError(f"409 tag already exists: {tag}")
        self.tags[tag] = revision
        if self._on_create_tag is not None:
            self._on_create_tag(self, tag)
        if self._create_applies_then_raises:
            raise FakeHubError("timeout waiting for response")


def _download_of(files, calls, seen=None):
    """스냅샷을 흉내낸다.

    호출을 기록해야 순서 단언이 의미를 갖고, `revision` 을 기록해야 "무엇을
    확인했는가" 를 잴 수 있다. 기록하지 않으면 생산 코드가 업로드 커밋 대신
    `main` 을 내려받아도 시험이 통과한다.
    """
    def download(repo_id, repo_type, local_dir, revision):
        calls.append("download")
        if seen is not None:
            seen.append(revision)
        root = Path(local_dir)
        for rel, body in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(body)
    return download


def _mirror(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


OID = "a" * 40


def test_an_existing_public_repo_is_refused_before_anything_is_uploaded(tmp_path):
    # create_repo 의 private 는 이미 있는 리포에 무시된다. 요청값만 찍으면
    # 승인 안 받은 데이터를 공개로 올리면서 "비공개" 라고 보고하게 된다.
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    api = FakeApi(private=False, calls=calls)
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=api, download=_download_of({}, calls))
    assert rc == 1
    assert "upload_folder" not in calls


def test_visibility_flipping_during_the_upload_stops_the_tag(tmp_path):
    # 올리기 전 확인은 그 시점의 값일 뿐이다. 업로드가 도는 동안 누가 Hub 에서
    # 공개로 바꾸면, 확인이 통과했다는 이유로 태그까지 걸어서는 안 된다.
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    api = FakeApi(private=True, flip_private_on_upload=True, calls=calls)
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=api, download=_download_of(_mirror(root), calls))
    assert rc == 1
    assert "create_tag" not in calls
    assert api.tags == {}


def test_a_failed_round_trip_leaves_the_existing_tag_alone(tmp_path):
    # 태그를 먼저 옮기면 exit 1 이어도 소비자에게는 잘못된 커밋이 v3.0 으로 보인다.
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    api = FakeApi(private=True, tags={"v3.0": "b" * 40}, calls=calls)
    corrupt = dict(_mirror(root))
    corrupt["stale/old.parquet"] = b"left over from a previous release"

    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=api, download=_download_of(corrupt, calls))
    assert rc == 1
    assert "create_tag" not in calls and "delete_tag" not in calls
    assert api.tags == {"v3.0": "b" * 40}


def test_the_round_trip_happens_between_the_upload_and_the_tag(tmp_path):
    """순서가 계약이므로 세 단계를 모두 기록해 대조한다.

    다운로드를 기록하지 않으면 확인을 태그 뒤로 옮겨도 이 시험이 통과한다.
    """
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    api = FakeApi(private=True, tags={"v3.0": "b" * 40}, calls=calls)
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=api, download=_download_of(_mirror(root), calls))
    assert rc == 0
    assert calls.index("repo_info") < calls.index("upload_folder")
    assert calls.index("upload_folder") < calls.index("download")
    assert calls.index("download") < calls.index("create_tag")
    assert api.tags == {"v3.0": OID}


def test_a_tag_already_on_the_uploaded_commit_is_left_untouched(tmp_path):
    # 지웠다 다시 만드는 사이는 태그가 없는 구간이다. 옮길 필요가 없으면
    # 그 구간을 아예 만들지 않는다.
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    api = FakeApi(private=True, tags={"v3.0": OID}, calls=calls)
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=api, download=_download_of(_mirror(root), calls))
    assert rc == 0
    assert "delete_tag" not in calls and "create_tag" not in calls
    assert api.tags == {"v3.0": OID}


def test_a_tag_lost_between_delete_and_create_is_reported_loudly(tmp_path, capsys):
    # Hub API 에 태그를 원자적으로 옮기는 방법이 없다. 그 사이에서 죽으면
    # 리포에 태그가 없어 LeRobot 이 열지 못하므로, 조용히 끝나면 안 된다.
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    api = FakeApi(private=True, tags={"v3.0": "b" * 40},
                  create_tag_raises=True, calls=calls)
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=api, download=_download_of(_mirror(root), calls))
    assert rc == 1
    assert api.tags == {}, "삭제는 됐고 생성이 실패한 상태"
    err = capsys.readouterr().err
    assert "태그가 없어" in err
    assert OID in err, "복구 명령에 태그할 커밋이 있어야 한다"


def test_the_round_trip_downloads_the_commit_that_was_uploaded(tmp_path):
    """무엇을 확인했는지도 재야 한다.

    revision 을 기록하지 않으면 생산 코드가 업로드 커밋 대신 `main` 이나 태그
    이름을 내려받아도 순서·무결성 시험이 통과한다.
    """
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls, seen = [], []
    api = FakeApi(private=True, calls=calls)
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=api, download=_download_of(_mirror(root), calls, seen))
    assert rc == 0
    assert seen == [OID], "업로드 커밋이 아니라 다른 revision 을 확인했다"


def test_a_tag_that_was_created_despite_the_error_is_reported_as_created(tmp_path):
    """예외가 '서버가 적용하지 않았다' 는 뜻은 아니다.

    POST 가 반영된 뒤 응답이 유실되면 태그는 실제로 존재한다. 그때 "태그가 없다"
    고 말하면 사람을 잘못된 복구로 보낸다 — 다시 읽어 사실만 말해야 한다.
    """
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    api = FakeApi(private=True, tags={"v3.0": "b" * 40},
                  create_tag_applies_then_raises=True, calls=calls)
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=api, download=_download_of(_mirror(root), calls))
    assert rc == 0, "실제로는 태그가 올바르게 만들어졌다"
    assert api.tags == {"v3.0": OID}


def test_failing_to_read_the_refs_back_is_not_reported_as_success(tmp_path):
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    # 첫 조회는 되고 태그를 건 뒤의 재조회에서 죽는다.
    api = FakeApi(private=True, tags={"v3.0": "b" * 40}, refs_raises_from=2,
                  calls=calls)
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=api, download=_download_of(_mirror(root), calls))
    assert rc == 1


def test_an_upload_without_a_commit_oid_stops_before_tagging(tmp_path):
    # 무엇을 태그할지 모르는 채로 태그를 걸면 안 된다.
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    api = FakeApi(private=True, oid="", calls=calls)
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=api, download=_download_of(_mirror(root), calls))
    assert rc == 1
    assert "create_tag" not in calls and "download" not in calls


def test_the_recovery_command_names_a_real_cli_subcommand(tmp_path, capsys):
    """복구 명령이 실제로 도는 형태여야 한다.

    안내가 `hf tag create` 였던 적이 있는데 설치된 CLI 의 명령은
    `hf repos tag create` 다. 그대로 실행하면 exit 2 였다. 복구 안내가 틀리면
    안내가 없는 것보다 나쁘다.
    """
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    api = FakeApi(private=True, tags={"v3.0": "b" * 40},
                  create_tag_raises=True, calls=calls)
    publish(root, "o/n", private=True, message="m", verify=True,
            api=api, download=_download_of(_mirror(root), calls))
    err = capsys.readouterr().err
    assert "hf repos tag create o/n v3.0" in err
    assert f"--revision {OID}" in err
    assert "--repo-type dataset" in err


def test_preflight_refuses_a_distribution_built_before_the_canon_changed(tmp_path):
    """올릴 디렉터리가 지금 코드의 산출물인지 묻습니다.

    변환기는 정본과 카드를 배포본 안에 **복사**하므로, 정본을 고친 뒤 재변환하지
    않으면 배포본이 옛 카드를 실은 채 남습니다. 디렉터리 이름만 보고 올리면 그게
    나가고, 실제로 10건 배치를 「현재 배치」라고 설명하는 카드가 업로드됐습니다.
    """
    import shutil

    from generation.publish_dataset import preflight

    repo = Path(__file__).resolve().parents[2]
    ds = tmp_path / "dist"
    ds.mkdir()
    for rel, dst in (("LICENSE", "LICENSE"), ("NOTICE", "NOTICE"),
                     ("meta/safety_schema.json", "meta/safety_schema.json"),
                     ("meta/assets.json", "meta/assets.json"),
                     ("docs/dataset-card.md", "README.md")):
        (ds / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, ds / dst)
    shutil.copytree(repo / "meta/joint_map", ds / "meta/joint_map")

    # 카드 한 글자만 달라도 「재변환하지 않았다」는 사실입니다.
    (ds / "README.md").write_text(
        (ds / "README.md").read_text(encoding="utf-8") + "\nstale\n", encoding="utf-8")
    problems = preflight(ds, repo)
    assert any("README.md" in x and "재변환" in x for x in problems), problems


def test_preflight_runs_the_validator_and_not_only_the_byte_compare(tmp_path):
    """두 시험 모두 바이트 대조로 실패하면 검증기 호출을 지워도 초록입니다.

    실제로 그 블록을 통째로 지우고 두 시험을 돌려 **2 passed** 였습니다.
    정본은 전부 최신이고 데이터만 망가진 배포본이 필요합니다.
    """
    import shutil

    from generation.publish_dataset import preflight

    repo = Path(__file__).resolve().parents[2]
    src = repo / ".tmp" / "fixtures" / "clean"
    if not src.is_dir():
        pytest.skip("clean 픽스처가 없습니다 — 먼저 `make fixture` 를 돌리십시오")

    ds = tmp_path / "dist"
    shutil.copytree(src, ds)
    # 정본은 그대로 두고 **데이터만** 망가뜨립니다. 바이트 대조는 통과하고
    # 검증기만 걸려야 합니다.
    for rel, dst in (("LICENSE", "LICENSE"), ("NOTICE", "NOTICE"),
                     ("meta/safety_schema.json", "meta/safety_schema.json"),
                     ("meta/assets.json", "meta/assets.json"),
                     ("docs/dataset-card.md", "README.md")):
        (ds / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, ds / dst)
    shutil.rmtree(ds / "meta" / "joint_map", ignore_errors=True)
    shutil.copytree(repo / "meta/joint_map", ds / "meta/joint_map")
    for q in sorted((ds / "data").rglob("*.parquet")):
        q.write_bytes(b"not a parquet")

    problems = preflight(ds, repo)
    assert any("검증기" in x for x in problems), (
        f"정본은 최신인데 데이터가 망가진 배포본을 검증기 단계가 잡지 못했습니다: {problems}")


def test_main_actually_runs_the_preflight(tmp_path):
    """헬퍼만 시험하면 호출부를 지워도 초록입니다 — 이 저장소에서 네 번 나온 형태.

    소스에 이름이 있는지로 보면 `if False:` 로 감싸도 통과하므로, 진입점을 실제로
    실행합니다. 낡은 배포본이면 Hub 를 건드리기 전에 exit 1 이어야 합니다.
    """
    import os
    import shutil
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    ds = tmp_path / "dist"
    ds.mkdir()
    shutil.copy2(repo / "LICENSE", ds / "LICENSE")     # 나머지는 빠져 있어 점검이 걸린다

    r = subprocess.run(
        [venv_or_current(repo), str(repo / "generation/publish_dataset.py"),
         "--dataset", str(ds), "--repo-id", "nobody/should-never-be-created"],
        capture_output=True, text=True, cwd=repo, timeout=300,
        # 토큰을 지워, 만에 하나 점검을 지나쳐도 Hub 에 닿지 못하게 한다.
        env={**os.environ, "HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": ""})
    assert r.returncode == 1, f"낡은 배포본인데 exit {r.returncode}:\n{r.stdout[-500:]}"
    assert "올리기 전 점검" in r.stderr, r.stderr[-500:]



def venv_or_current(repo: Path) -> str:
    """`.venv` 가 있으면 그것, 없으면 지금 이 인터프리터.

    CI 는 `.venv` 를 만들지 않는다.
    """
    import sys

    venv = repo / ".venv" / "bin" / "python"
    return str(venv if venv.exists() else Path(sys.executable))


def _min_dist(tmp_path, repo) -> Path:
    """정본만 최신인 최소 배포본. 나머지 결함은 각 시험이 심습니다."""
    import shutil

    ds = tmp_path / "dist"
    ds.mkdir()
    for rel, dst in (("LICENSE", "LICENSE"), ("NOTICE", "NOTICE"),
                     ("meta/safety_schema.json", "meta/safety_schema.json"),
                     ("meta/assets.json", "meta/assets.json"),
                     ("docs/dataset-card.md", "README.md")):
        (ds / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, ds / dst)
    shutil.copytree(repo / "meta/joint_map", ds / "meta/joint_map")
    return ds


def test_preflight_compares_the_narrowed_canon(tmp_path):
    """좁혀 쓰는 둘을 대조하지 않으면 임계값을 바꿔도 통과합니다.

    실제로 0.34 를 9.99 로 바꾼 배포본이 검증기와 preflight 를 모두 통과했습니다.
    그 뒤 이 동작을 지켜 주는 단언이 없었습니다.
    """
    import json
    import shutil

    from generation.publish_dataset import preflight

    repo = Path(__file__).resolve().parents[2]
    src = repo / ".tmp" / "fixtures" / "clean"
    if not src.is_dir():
        pytest.skip("clean 픽스처가 없습니다")
    ds = tmp_path / "dist"
    shutil.copytree(src, ds)

    rules = json.loads((ds / "meta" / "rules.json").read_text(encoding="utf-8"))
    for rule in rules.get("rules", []):
        for param in (rule.get("parameters") or {}).values():
            if isinstance(param, dict) and isinstance(param.get("value"), (int, float)):
                param["value"] = 9.99
    (ds / "meta" / "rules.json").write_text(
        json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    problems = preflight(ds, repo, repro_hook="")
    assert any("rules.json" in x and "좁힌 것" in x for x in problems), (
        f"임계값을 바꾼 배포본을 좁힌 정본 대조가 잡지 못했습니다: {problems}")


def test_preflight_refuses_a_zero_sample_hook(tmp_path):
    """표본 0 은 훅을 한 번도 부르지 않으면서 C9 를 OK 로 만듭니다."""
    from generation.publish_dataset import preflight

    repo = Path(__file__).resolve().parents[2]
    problems = preflight(_min_dist(tmp_path, repo), repo,
                         repro_hook="/bin/true", repro_sample=0)
    assert any("repro-sample" in x for x in problems), problems


def test_c9_skip_is_recognised_in_the_validator_report():
    """SKIP 은 exit 0 입니다. 훅 없이 올리면 재현성을 한 번도 확인하지 않습니다.

    작은 함수로 떼어 낸 이유는 배포본을 통째로 만들어 확인하려 하면 다른 검사가 먼저
    실패하고 그 실패 메시지 안에 SKIP 이라는 글자가 딸려 들어와 **엉뚱한 이유로
    초록**이 되기 때문입니다 — 실제로 그렇게 썼다가 잡았습니다.
    """
    from generation.publish_dataset import c9_skipped

    assert c9_skipped(" -  C9 시드 재현성             SKIP")
    assert not c9_skipped(" ✓  C9 시드 재현성             OK")
    assert not c9_skipped(" ✗  C9 시드 재현성             ERROR 2")
    assert not c9_skipped(
        " ✓  C9 시드 재현성             OK\n -  C1 레이아웃·필수 파일         SKIP")


def test_preflight_calls_the_c9_skip_check():
    """헬퍼만 시험하면 호출부를 지워도 초록입니다 — 이 저장소에서 다섯 번 나온 형태."""
    import inspect

    import generation.publish_dataset as pub

    assert "c9_skipped(" in inspect.getsource(pub.preflight), (
        "올리기 전 점검이 C9 SKIP 을 보지 않습니다")


def test_the_card_must_describe_the_batch_it_ships_with(tmp_path):
    """바이트 대조는 「카드가 저장소 것과 같은가」만 봅니다.

    카드 자체가 다른 배치를 설명하고 있으면 그것은 안 잡힙니다.
    산문을 훑던 첫 판본은 **국·영 두 벌 중 한쪽만 틀려도 통과**했고 줄바꿈만 바꿔도
    정상 카드를 거부했습니다. 그래서 표식 한 줄만 봅니다.
    """
    import pandas as pd

    from generation.publish_dataset import card_matches_batch

    ds = tmp_path / "dist"
    (ds / "meta" / "episodes").mkdir(parents=True)
    # 10 에피소드 중 2건이 접촉 금지 부위(BR-01) 규칙을 답니다.
    pd.DataFrame({
        "outcome": ["fail"] * 3 + ["near_miss"] * 5 + ["nominal"] * 2,
        "triggered_rules": [["R-TS15066-A2-01"], ["R-TS15066-A2-01", "R-TS15066-A2-03"],
                            ["R-TS15066-A2-03"]] + [[]] * 7,
    }).to_parquet(ds / "meta" / "episodes" / "file-000.parquet", index=False)
    (ds / "meta").mkdir(exist_ok=True)
    rules = {
        "contact_limits": {"BR-01": {"contact_permitted": False},
                           "BR-03": {"contact_permitted": True}},
        "rules": [{"rule_id": "R-TS15066-A2-01", "body_region": "BR-01"},
                  {"rule_id": "R-TS15066-A2-03", "body_region": "BR-03"}],
    }
    rules_path = ds / "meta" / "rules.json"
    rules_path.write_text(json.dumps(rules, ensure_ascii=False), encoding="utf-8")

    card = ds / "README.md"
    card.write_text("표식이 없는 카드\n", encoding="utf-8")
    assert any("표식이 없습니다" in x for x in card_matches_batch(ds))

    card.write_text("<!-- batch-stats: episodes=400 fail=100 near_miss=161 nominal=139 "
                    "contact_exceeded=3 contact_prohibited=2 -->\n", encoding="utf-8")
    problems = card_matches_batch(ds)
    assert any("episodes=400" in x for x in problems), problems
    assert any("fail=100" in x for x in problems), problems

    # 접촉 금지 부위 건수만 틀려도 걸립니다. 이 수를 손으로 세던 동안 13건이 1건으로
    # 실린 적이 있습니다.
    card.write_text("<!-- batch-stats: episodes=10 fail=3 near_miss=5 nominal=2 "
                    "contact_exceeded=3 contact_prohibited=1 -->\n", encoding="utf-8")
    problems = card_matches_batch(ds)
    assert any("contact_prohibited=1" in x and "배치(2)" in x for x in problems), problems

    # 한계 초과 건수만 틀려도 걸립니다 — 「접촉이 표준을 넘은 것이 몇 건인가」에
    # 답하는 수입니다.
    card.write_text("<!-- batch-stats: episodes=10 fail=3 near_miss=5 nominal=2 "
                    "contact_exceeded=9 contact_prohibited=2 -->\n", encoding="utf-8")
    problems = card_matches_batch(ds)
    assert any("contact_exceeded=9" in x and "배치(3)" in x for x in problems), problems

    # 금지 부위를 보는 규칙이 배포본에 없으면 **0 으로 통과시키지 않고** 말합니다.
    rules_path.write_text(json.dumps(
        {"contact_limits": {"BR-01": {"contact_permitted": False}}, "rules": []},
        ensure_ascii=False), encoding="utf-8")
    problems = card_matches_batch(ds)
    assert any("무엇을 세는지 확인할 수 없습니다" in x for x in problems), problems
    assert not any("모르는 항목" in x for x in problems), (
        f"유도하지 못한 항목을 「모르는 항목」으로도 세었습니다: {problems}")
    rules_path.write_text(json.dumps(rules, ensure_ascii=False), encoding="utf-8")

    # 유도할 재료가 없으면 **말하고** 넘어갑니다. 조용히 건너뛰면 그 표식이 무엇을
    # 세는지 아무도 모르는 채로 초록이 됩니다.
    rules_path.unlink()
    problems = card_matches_batch(ds)
    assert any("유도하지 못했습니다" in x for x in problems), problems
    assert not any("모르는 항목" in x for x in problems), problems
    rules_path.write_text(json.dumps(rules, ensure_ascii=False), encoding="utf-8")

    # 줄바꿈이 달라도, 다른 언어 문단이 옆에 있어도 표식만 맞으면 통과합니다.
    card.write_text(
        "The batch is 400 episodes: 100 fail.\n\n"
        "<!--\n batch-stats: episodes=10\n fail=3 near_miss=5 nominal=2\n"
        " contact_exceeded=3 contact_prohibited=2\n-->\n\n"
        "배치는 400 에피소드이고 fail 100 입니다.\n", encoding="utf-8")
    assert card_matches_batch(ds) == []


def test_preflight_calls_the_card_check():
    import inspect

    import generation.publish_dataset as pub

    assert "card_matches_batch(" in inspect.getsource(pub.preflight)


def test_a_foreign_reproduction_hook_is_refused(tmp_path):
    """훅은 호출자가 고르는 프로그램이고 검증기는 그 exit 0 을 믿습니다.

    `--repro-hook /bin/true` 로 게이트를 통째로 지나갈 수 있었습니다.
    """
    import os
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    r = subprocess.run(
        [venv_or_current(repo), str(repo / "generation/publish_dataset.py"),
         "--dataset", str(tmp_path), "--repo-id", "nobody/never",
         "--repro-hook", "/bin/true"],
        capture_output=True, text=True, cwd=repo, timeout=300,
        env={**os.environ, "HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": ""})
    assert r.returncode == 1, r.stdout[-400:]
    assert "저장소 것이 아닙니다" in r.stderr, r.stderr[-400:]


def test_a_duplicate_batch_marker_is_refused(tmp_path):
    """첫 표식만 읽히므로 뒤엣것이 틀려도 통과했습니다."""
    import pandas as pd

    from generation.publish_dataset import card_matches_batch

    ds = tmp_path / "dist"
    (ds / "meta" / "episodes").mkdir(parents=True)
    pd.DataFrame({"outcome": ["fail"] * 2}).to_parquet(
        ds / "meta" / "episodes" / "f.parquet", index=False)
    (ds / "README.md").write_text(
        "<!-- batch-stats: episodes=2 fail=2 near_miss=0 nominal=0 -->\n"
        "<!-- batch-stats: episodes=400 fail=100 near_miss=161 nominal=139 -->\n",
        encoding="utf-8")
    assert any("2개 있습니다" in x for x in card_matches_batch(ds))


def test_the_shipped_asset_must_match_the_pinned_hash(tmp_path):
    """원본이 git 에 없으므로 「원본과 배포본이 같다」는 어느 자산인지 말하지 않습니다.

    둘을 함께 바꾸면 성립합니다.
    """
    import json
    import shutil

    from generation.publish_dataset import shipped_assets_match_source

    repo = Path(__file__).resolve().parents[2]
    canon = json.loads((repo / "meta/assets.json").read_text(encoding="utf-8"))
    entry = canon["scene"]["keti_dummy_ts15066_v1"]
    src = repo / entry["usd_path"]
    if not src.is_file():
        pytest.skip("더미 자산이 빌드되어 있지 않습니다")

    # What an S1a-shaped batch ships besides the dummy: the robot its episodes ran, as its folder.
    from generation.assets import shipped_asset_files
    robot = canon["robots"]["unitree_h1"]["usd_path"]
    plan = shipped_asset_files(canon, [entry["usd_path"], robot], repo)
    missing = sorted(str(f.src.relative_to(repo)) for f in plan.values() if not f.src.is_file())
    if missing:
        pytest.skip(f"실을 원본이 빌드 · 받기되어 있지 않습니다: {missing}")

    ds = tmp_path / "dist"
    for rel, f in plan.items():
        (ds / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f.src, ds / rel)
    _episodes_using(ds, [entry["usd_path"], robot])
    assert shipped_assets_match_source(ds, repo) == []

    (ds / "assets" / src.name).write_bytes(b"#usda 1.0\n# different\n")
    problems = shipped_assets_match_source(ds, repo)
    assert any("다릅니다" in x for x in problems), problems


def _episodes_using(ds, paths, rows=2, scenario="S1a", human_model=None, robot_model="unitree_h1"):
    """meta/episodes as the converter writes it, as far as this check reads: scene_assets per
    episode (a JSON string of {path, license, redistributed} items), scenario_id,
    human_model (None as for a scenario without one) and robot_model (a meta/assets.json
    robots key, as every episode's is)."""
    import pandas as pd

    d = ds / "meta" / "episodes" / "chunk-000"
    d.mkdir(parents=True, exist_ok=True)
    items = json.dumps([{"path": p, "license": "x", "redistributed": True} for p in paths])
    pd.DataFrame({"episode_index": range(rows), "scene_assets": [items] * rows,
                  "scenario_id": [scenario] * rows, "human_model": [human_model] * rows,
                  "robot_model": [robot_model] * rows}).to_parquet(
        d / "file-000.parquet", index=False)


def _fake_repo(tmp_path, *, texture_bytes=b"pixels\n"):
    """A repository tree with two redistributable assets -- a plain USD and one that brings a
    texture pinned by a scene-format lock -- and one the canon keeps back."""
    import hashlib

    repo = tmp_path / "repo"
    sha = lambda b: hashlib.sha256(b).hexdigest()           # noqa: E731
    usd_a, usd_b, tex = b"#usda 1.0\n# a\n", b"#usda 1.0\n# b\n", texture_bytes
    for rel, data in (("build/a.usd", usd_a), ("build/b.usd", usd_b),
                      ("build/textures/wood.jpg", tex)):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(data)
    lock = repo / "generation/scene/textures.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(f"wood w diffuse CC0 polyhaven https://example.invalid/wood.jpg {sha(tex)}\n",
                    encoding="utf-8")
    (repo / "meta").mkdir()
    # The robot as the real H1 is: known to the canon, never shipped.
    (repo / "meta" / "assets.json").write_text(json.dumps({"robots": {
        "unitree_h1": {"usd_path": "{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
                       "redistributed": False},
    }, "scene": {
        "a": {"usd_path": "build/a.usd", "redistributed": True, "usd_sha256": sha(usd_a)},
        "b": {"usd_path": "build/b.usd", "redistributed": True, "usd_sha256": sha(usd_b),
              "bundle": {"textures_dir": "build/textures",
                         "textures_lock": "generation/scene/textures.lock",
                         "files": [{"file": "wood.jpg", "sha256": sha(tex), "license": "CC0-1.0",
                                    "url": "https://example.invalid/wood.jpg"}]}},
        "kept": {"usd_path": "isaaclab://kept", "redistributed": False},
    }}), encoding="utf-8")
    # S1a declares nothing, so what a batch uses is what its episodes list; S1x is a scenario
    # that says its episodes used b (the anchor: generation.assets.required_scene_assets).
    (repo / "meta" / "scenarios.json").write_text(json.dumps({"scenarios": [
        {"scenario_id": "S1a", "scene": {"assets": [], "human_model": None}},
        {"scenario_id": "S1x", "scene": {"assets": ["build/b.usd"], "human_model": None}},
    ]}), encoding="utf-8")
    return repo


def _ship(repo, ds, rels):
    import shutil

    for rel in rels:
        src = {"assets/a.usd": "build/a.usd", "assets/b.usd": "build/b.usd",
               "assets/textures/wood.jpg": "build/textures/wood.jpg"}[rel]
        (ds / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / src, ds / rel)


def test_a_batch_is_held_to_the_assets_its_episodes_use(tmp_path):
    """The canon's other redistributable assets are not demanded -- a v0.1 batch
    must not need the S1d scene -- and what the episodes use must be there, textures too."""
    from generation.publish_dataset import shipped_assets_match_source

    repo = _fake_repo(tmp_path)
    ds = tmp_path / "dist"
    _episodes_using(ds, ["build/a.usd", "isaaclab://kept"])
    _ship(repo, ds, ["assets/a.usd"])
    assert shipped_assets_match_source(ds, repo) == []          # b is the canon's, not this batch's

    _episodes_using(ds, ["build/a.usd", "build/b.usd"])
    problems = shipped_assets_match_source(ds, repo)
    assert any("배포본에 자산이 없습니다: assets/b.usd" in x for x in problems), problems
    assert any("배포본에 자산이 없습니다: assets/textures/wood.jpg" in x for x in problems), problems
    _ship(repo, ds, ["assets/b.usd", "assets/textures/wood.jpg"])
    assert shipped_assets_match_source(ds, repo) == []


def test_a_shipped_texture_is_held_to_its_lock(tmp_path):
    from generation.publish_dataset import shipped_assets_match_source

    repo = _fake_repo(tmp_path)
    ds = tmp_path / "dist"
    _episodes_using(ds, ["build/b.usd"])
    _ship(repo, ds, ["assets/b.usd", "assets/textures/wood.jpg"])
    assert shipped_assets_match_source(ds, repo) == []
    (ds / "assets/textures/wood.jpg").write_bytes(b"other pixels\n")
    problems = shipped_assets_match_source(ds, repo)
    assert any("assets/textures/wood.jpg 이 저장소의 build/textures/wood.jpg 와 다릅니다" in x
               for x in problems), problems

    # Source and shipped copy changed together still differ from the lock's pin.
    (repo / "build/textures/wood.jpg").write_bytes(b"other pixels\n")
    problems = shipped_assets_match_source(ds, repo)
    assert any("실린 assets/textures/wood.jpg 의 해시가 정본과 다릅니다" in x for x in problems), problems


def test_a_canon_row_the_lock_does_not_pin_stops_the_comparison(tmp_path):
    from generation.publish_dataset import shipped_assets_match_source

    repo = _fake_repo(tmp_path)
    canon = json.loads((repo / "meta/assets.json").read_text(encoding="utf-8"))
    canon["scene"]["b"]["bundle"]["files"][0]["url"] = "https://example.invalid/elsewhere.jpg"
    (repo / "meta/assets.json").write_text(json.dumps(canon), encoding="utf-8")
    ds = tmp_path / "dist"
    _episodes_using(ds, ["build/b.usd"])
    _ship(repo, ds, ["assets/b.usd", "assets/textures/wood.jpg"])
    problems = shipped_assets_match_source(ds, repo)
    assert any("정하지 못했습니다" in x and "textures.lock 과 다릅니다" in x for x in problems), problems


def test_a_file_the_batch_does_not_use_is_refused(tmp_path):
    """A stale S1d conversion carried the v1 dummy its episodes never load."""
    from generation.publish_dataset import shipped_assets_match_source

    repo = _fake_repo(tmp_path)
    ds = tmp_path / "dist"
    _episodes_using(ds, ["build/a.usd"])
    _ship(repo, ds, ["assets/a.usd", "assets/b.usd", "assets/textures/wood.jpg"])
    problems = shipped_assets_match_source(ds, repo)
    assert any("이 배치가 싣지 않는 파일 2개" in x and "assets/b.usd" in x for x in problems), problems


def test_a_source_the_canon_points_at_but_the_tree_lacks_is_reported(tmp_path):
    from generation.publish_dataset import shipped_assets_match_source

    repo = _fake_repo(tmp_path)
    ds = tmp_path / "dist"
    _episodes_using(ds, ["build/b.usd"])
    _ship(repo, ds, ["assets/b.usd", "assets/textures/wood.jpg"])
    (repo / "build/textures/wood.jpg").unlink()
    problems = shipped_assets_match_source(ds, repo)
    assert any("b: 정본이 가리키는 원본이 없습니다: build/textures/wood.jpg" in x for x in problems), problems


def test_an_asset_the_episodes_leave_out_is_still_demanded(tmp_path):
    """The episodes' scene_assets drop an asset and its files are
    deleted. Drawn from the lists alone, nothing asked for it; the repository canon says the
    scenario used it (scene.assets), and the episode's own human_model names one too."""
    from generation.publish_dataset import shipped_assets_match_source

    repo = _fake_repo(tmp_path)
    ds = tmp_path / "dist"
    _episodes_using(ds, ["build/a.usd"], scenario="S1x")          # S1x says it used b
    _ship(repo, ds, ["assets/a.usd"])
    problems = shipped_assets_match_source(ds, repo)
    assert any("배포본에 자산이 없습니다: assets/b.usd" in x for x in problems), problems
    assert any("배포본에 자산이 없습니다: assets/textures/wood.jpg" in x for x in problems), problems

    canon = json.loads((repo / "meta/assets.json").read_text(encoding="utf-8"))
    canon["scene"]["human"] = canon["scene"].pop("a")              # a.usd is a human model's USD
    (repo / "meta/assets.json").write_text(json.dumps(canon), encoding="utf-8")
    _episodes_using(ds, [], scenario="S1a", human_model="human")
    problems = shipped_assets_match_source(ds, repo)
    assert problems == [], problems                                # a.usd demanded, and there
    (ds / "assets/a.usd").unlink()
    assert any("배포본에 자산이 없습니다: assets/a.usd" in x
               for x in shipped_assets_match_source(ds, repo))


@pytest.mark.parametrize("scenario, model, phrase", [
    ("S9z", None, "시나리오 S9z 가 저장소 정본 meta/scenarios.json 에 없어"),
    ("S1a", "nobody", "human_model 'nobody' 이 저장소 정본 meta/assets.json 의 scene 에 없어"),
])
def test_a_batch_whose_use_the_canon_cannot_say_is_not_passed(tmp_path, scenario, model, phrase):
    from generation.publish_dataset import shipped_assets_match_source

    repo = _fake_repo(tmp_path)
    ds = tmp_path / "dist"
    _episodes_using(ds, ["build/a.usd"], scenario=scenario, human_model=model)
    _ship(repo, ds, ["assets/a.usd"])
    problems = shipped_assets_match_source(ds, repo)
    assert any(phrase in x for x in problems), problems


def test_the_robot_an_episode_ran_is_part_of_what_the_batch_uses(tmp_path):
    """No scenario names its robot -- the episode's robot_model does. A
    robot the canon lets travel is then demanded like any asset the batch used, even when the
    episodes' scene_assets leave it out."""
    import hashlib

    from generation.publish_dataset import shipped_assets_match_source

    repo = _fake_repo(tmp_path)
    data = b"#usda 1.0\n# bot\n"
    (repo / "build/bot.usd").write_bytes(data)
    canon = json.loads((repo / "meta/assets.json").read_text(encoding="utf-8"))
    canon["robots"]["bot"] = {"usd_path": "build/bot.usd", "redistributed": True,
                              "usd_sha256": hashlib.sha256(data).hexdigest()}
    (repo / "meta/assets.json").write_text(json.dumps(canon), encoding="utf-8")
    ds = tmp_path / "dist"
    _episodes_using(ds, ["build/a.usd"], robot_model="bot")
    _ship(repo, ds, ["assets/a.usd"])
    problems = shipped_assets_match_source(ds, repo)
    assert any("배포본에 자산이 없습니다: assets/bot.usd" in x for x in problems), problems


@pytest.mark.parametrize("model, phrase", [
    ("nobody", "robot_model 'nobody' 이 meta/assets.json 의 robots 에 없습니다"),
    (None, "robot_model None 이 meta/assets.json 의 robots 에 없습니다"),
])
def test_a_robot_the_canon_cannot_name_is_not_passed(tmp_path, model, phrase):
    from generation.publish_dataset import shipped_assets_match_source

    repo = _fake_repo(tmp_path)
    ds = tmp_path / "dist"
    _episodes_using(ds, ["build/a.usd"], robot_model=model)
    _ship(repo, ds, ["assets/a.usd"])
    problems = shipped_assets_match_source(ds, repo)
    assert any(phrase in x for x in problems), problems


def test_a_scenario_the_anchor_cannot_read_is_not_passed(tmp_path):
    from generation.publish_dataset import shipped_assets_match_source

    repo = _fake_repo(tmp_path)
    doc = json.loads((repo / "meta/scenarios.json").read_text(encoding="utf-8"))
    doc["scenarios"][0]["scene"].pop("assets")
    (repo / "meta/scenarios.json").write_text(json.dumps(doc), encoding="utf-8")
    ds = tmp_path / "dist"
    _episodes_using(ds, ["build/a.usd"])
    _ship(repo, ds, ["assets/a.usd"])
    assert any("정본에서 알 수 없습니다: S1a: scene.assets" in x
               for x in shipped_assets_match_source(ds, repo))


def test_a_distribution_without_episode_assets_is_not_passed(tmp_path):
    from generation.publish_dataset import shipped_assets_match_source

    repo = _fake_repo(tmp_path)
    ds = tmp_path / "dist"
    _ship(repo, ds, ["assets/a.usd"])
    assert any("대조하지 못했습니다" in x for x in shipped_assets_match_source(ds, repo))
    import pandas as pd
    (ds / "meta/episodes").mkdir(parents=True)
    pd.DataFrame({"episode_index": [0]}).to_parquet(ds / "meta/episodes/f.parquet", index=False)
    assert any("['scene_assets', 'scenario_id'] 열이 없어" in x
               for x in shipped_assets_match_source(ds, repo))
    # Without scenario_id the anchor cannot be read: that is not a pass either.
    pd.DataFrame({"episode_index": [0], "scene_assets": ["[]"]}).to_parquet(
        ds / "meta/episodes/f.parquet", index=False)
    assert any("['scenario_id'] 열이 없어" in x for x in shipped_assets_match_source(ds, repo))
    # Nor without robot_model: the robot's part of the anchor.
    pd.DataFrame({"episode_index": [0], "scene_assets": ["[]"], "scenario_id": ["S1a"]}).to_parquet(
        ds / "meta/episodes/f.parquet", index=False)
    assert any("['robot_model'] 열이 없어" in x for x in shipped_assets_match_source(ds, repo))


ROBOT_TOKEN = "{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd"
ROBOT_FILES = {"h1.usd": b"#usda 1.0\n# robot\n", "instanceable_meshes.usd": b"#usda 1.0\n# its meshes\n",
               "LICENSE.txt": b"BSD 3-Clause License"}


def _ship_robot(repo):
    """The fake repository's robot as the H1 ships: its own folder, fetched under build/unitree_h1/ -- the USD
    (named by the nucleus token, as every episode references it), the layer beside it and the licence file,
    each a row of ship.files."""
    import hashlib

    for name, data in ROBOT_FILES.items():
        (repo / "build/unitree_h1").mkdir(parents=True, exist_ok=True)
        (repo / "build/unitree_h1" / name).write_bytes(data)
    canon = json.loads((repo / "meta/assets.json").read_text(encoding="utf-8"))
    canon["robots"]["unitree_h1"] = {
        "usd_path": ROBOT_TOKEN, "redistributed": True,
        "usd_sha256": hashlib.sha256(ROBOT_FILES["h1.usd"]).hexdigest(),
        "ship": {"dir": "unitree_h1", "source_dir": "build/unitree_h1", "license_file": "LICENSE.txt",
                 "files": [{"file": n, "sha256": hashlib.sha256(d).hexdigest(), "license": "BSD-3-Clause",
                            "url": f"{{ISAACLAB_NUCLEUS_DIR}}/Robots/Unitree/H1/{n}"} for n, d in ROBOT_FILES.items()]}}
    (repo / "meta/assets.json").write_text(json.dumps(canon), encoding="utf-8")


def _ship_robot_folder(repo, ds):
    import shutil

    (ds / "assets/unitree_h1").mkdir(parents=True, exist_ok=True)
    for name in ROBOT_FILES:
        shutil.copy2(repo / "build/unitree_h1" / name, ds / "assets/unitree_h1" / name)


def test_a_robot_that_ships_as_its_folder_is_held_to_its_rows(tmp_path):
    """The robot's USD is a nucleus token, not a path of the repository: what the canon says of it is its ship
    block, and the folder is demanded, compared byte for byte and taken by the check of files the batch does not
    use -- every row of it, and nothing else under it."""
    from generation.publish_dataset import shipped_assets_match_source

    repo = _fake_repo(tmp_path)
    _ship_robot(repo)
    ds = tmp_path / "dist"
    _episodes_using(ds, ["build/a.usd", ROBOT_TOKEN])
    _ship(repo, ds, ["assets/a.usd"])
    problems = shipped_assets_match_source(ds, repo)
    assert not any("저장소 경로가 아닙니다" in x for x in problems), problems
    for name in ROBOT_FILES:
        assert any(f"unitree_h1: 배포본에 자산이 없습니다: assets/unitree_h1/{name}" in x for x in problems), problems
    _ship_robot_folder(repo, ds)
    assert shipped_assets_match_source(ds, repo) == []

    (ds / "assets/unitree_h1/LICENSE.txt").write_bytes(b"another licence")
    problems = shipped_assets_match_source(ds, repo)
    assert any("assets/unitree_h1/LICENSE.txt 이 저장소의 build/unitree_h1/LICENSE.txt 와 다릅니다" in x
               for x in problems), problems
    _ship_robot_folder(repo, ds)
    for stray in ("assets/unitree_h1/stray.usd", "assets/h1.usd"):
        (ds / stray).write_bytes(b"#usda 1.0\n")
        problems = shipped_assets_match_source(ds, repo)
        assert any("이 배치가 싣지 않는 파일 1개" in x and stray in x for x in problems), problems
        (ds / stray).unlink()


def test_a_robot_whose_ship_block_cannot_say_what_ships_is_reported_whatever_the_batch(tmp_path):
    """The canon's own statement: a robot that ships must say which files, whatever this batch uses -- and a
    robot that would ship without a ship block is a token the repository has no file for."""
    from generation.publish_dataset import shipped_assets_match_source

    repo = _fake_repo(tmp_path)
    _ship_robot(repo)
    canon = json.loads((repo / "meta/assets.json").read_text(encoding="utf-8"))
    canon["robots"]["unitree_h1"]["ship"]["license_file"] = "COPYING"
    (repo / "meta/assets.json").write_text(json.dumps(canon), encoding="utf-8")
    ds = tmp_path / "dist"
    _episodes_using(ds, ["build/a.usd"], robot_model="other")
    _ship(repo, ds, ["assets/a.usd"])
    problems = shipped_assets_match_source(ds, repo)
    assert any("unitree_h1: license_file 'COPYING' 의 행이 ship.files 에 없습니다" in x for x in problems), problems
    del canon["robots"]["unitree_h1"]["ship"]
    (repo / "meta/assets.json").write_text(json.dumps(canon), encoding="utf-8")
    problems = shipped_assets_match_source(ds, repo)
    assert any("unitree_h1: 재배포한다는데 저장소 경로가 아닙니다" in x for x in problems), problems


def test_duplicate_keys_inside_one_marker_are_refused(tmp_path):
    """표식 개수는 하나여도 안의 키가 중복이면 마지막 값만 남습니다.

    `episodes=999 episodes=2` 가 실제 2건 배치에서 통과했습니다.
    """
    import pandas as pd

    from generation.publish_dataset import card_matches_batch

    ds = tmp_path / "dist"
    (ds / "meta" / "episodes").mkdir(parents=True)
    pd.DataFrame({"outcome": ["fail"] * 2}).to_parquet(
        ds / "meta" / "episodes" / "f.parquet", index=False)
    (ds / "README.md").write_text(
        "<!-- batch-stats: episodes=999 episodes=2 fail=999 fail=2 "
        "near_miss=0 nominal=0 -->\n", encoding="utf-8")
    assert any("중복된 항목" in x for x in card_matches_batch(ds))


@pytest.mark.parametrize("token", [
    "episodes=7.9", "fail=2garbage", "fail=-2", "fail=", "fail=2,", "fail=٢", "=2", "S1d",
])
def test_a_marker_token_is_read_whole_or_refused(token):
    """앞부분만 읽던 파서는 `episodes=7.9` 를 7 로, `fail=2garbage` 를 2 로 읽고
    문제를 내지 않았습니다 — 깨진 표식이 맞는 배치와 대조를 통과합니다.
    게시 전 점검과 인계 색인이 같은 파서를 씁니다."""
    from generation.publish_dataset import parse_batch_marker

    good = ["episodes=7", "fail=2", "near_miss=3", "nominal=2"]
    text = "<!-- batch-stats: " + " ".join(good + [token]) + " -->"
    values, problems = parse_batch_marker(text, "카드")
    assert values == {}, values
    assert any("읽을 수 없는 항목" in p and token in p for p in problems), problems


def test_a_sound_marker_still_reads_across_lines():
    """토큰 단위로 바꿔도 줄바꿈 · 여러 칸 공백은 그대로 읽힙니다."""
    from generation.publish_dataset import parse_batch_marker

    text = "<!--\n batch-stats: episodes=10\n fail=3   near_miss=5 nominal=2\n-->"
    assert parse_batch_marker(text, "카드") == (
        {"episodes": 10, "fail": 3, "near_miss": 5, "nominal": 2}, [])


def test_the_usd_path_must_stay_inside_the_repository(tmp_path):
    """`../outside.usd` 로 비교 기준 자체를 옮길 수 있었습니다."""
    import json

    from generation.publish_dataset import shipped_assets_match_source

    fake_repo = tmp_path / "repo"
    (fake_repo / "meta").mkdir(parents=True)
    outside = tmp_path / "outside.usd"
    outside.write_bytes(b"#usda 1.0\n")
    (fake_repo / "meta" / "assets.json").write_text(json.dumps({
        "scene": {"x": {"usd_path": "../outside.usd", "redistributed": True,
                        "usd_sha256": "0" * 64}},
    }), encoding="utf-8")
    ds = tmp_path / "dist"
    (ds / "assets").mkdir(parents=True)
    (ds / "assets" / "outside.usd").write_bytes(b"#usda 1.0\n")

    problems = shipped_assets_match_source(ds, fake_repo)
    assert any("저장소 밖" in x for x in problems), problems


def test_a_redistributed_usd_must_not_carry_standard_values(tmp_path):
    """이 자산은 CC BY 4.0 으로 재배포됩니다.

    유효질량·스프링상수를 주석 속성으로 실었더니 유료 표준 수치 34개가 상업적
    이용 허용 라이선스로 나갈 뻔했습니다. 민감정보 검사는 `.usd` 를 제외하므로
    여기서 봅니다.
    """
    from generation.publish_dataset import redistributed_usd_has_no_standard_values

    ds = tmp_path / "dist"
    (ds / "assets").mkdir(parents=True)
    (ds / "assets" / "x.usd").write_bytes(b"#usda 1.0\n")
    # pxr 이 없으면 「확인 못 함」이어야 합니다 — 통과가 아닙니다.
    problems = redistributed_usd_has_no_standard_values(ds)
    try:
        from pxr import Usd  # noqa: F401
        has_pxr = True
    except ImportError:
        has_pxr = False
    if not has_pxr:
        assert any("확인하지 못했습니다" in x for x in problems), problems
        # 정본이 확인한 해시와 같으면 그 확인을 물려받습니다.
        import hashlib
        got = hashlib.sha256((ds / "assets" / "x.usd").read_bytes()).hexdigest()
        assert redistributed_usd_has_no_standard_values(ds, {"x.usd": got}) == []
        assert redistributed_usd_has_no_standard_values(ds, {"x.usd": "0" * 64})


def test_preflight_calls_the_usd_check():
    import inspect

    import generation.publish_dataset as pub

    assert "redistributed_usd_has_no_standard_values(" in inspect.getsource(pub.preflight)


def test_the_mass_check_reads_only_each_files_own_layer(tmp_path):
    """The pre-flight opens each USD directly under assets/ as its own layer, not composed. A USD that only
    references another file -- a stage referencing the robot folder -- does not bring that file's masses
    with it; a mass written in a file's own layer is caught, a value or time samples, and an attribute declared
    without a value is not a value. The robot folder is not scanned: it is held by its pinned hashes."""
    pytest.importorskip("pxr")
    from pxr import Usd

    from generation.publish_dataset import layer_forbidden_attrs, redistributed_usd_has_no_standard_values

    ds = tmp_path / "dist"
    (ds / "assets/unitree_h1").mkdir(parents=True)
    robot = ds / "assets/unitree_h1/h1.usd"
    robot.write_text('#usda 1.0\n(\n  defaultPrim = "H1"\n)\ndef Xform "H1" {\n  def Xform "pelvis" {\n'
                     '    float physics:mass = 5\n  }\n}\n', encoding="utf-8")
    stage = ds / "assets/stage.usd"
    stage.write_text('#usda 1.0\ndef Xform "World" {\n  def "Robot" (\n    prepend references = @./unitree_h1/h1.usd@\n'
                     '  ) {\n  }\n}\n', encoding="utf-8")
    # Composed, the stage does carry the robot's mass: what a composing scan reads.
    composed = Usd.Stage.Open(str(stage))
    assert any(a.GetName() == "physics:mass" and a.HasAuthoredValue()
               for prim in composed.Traverse() for a in prim.GetAttributes())
    assert layer_forbidden_attrs(stage) == set()
    assert layer_forbidden_attrs(robot) == {"physics:mass"}
    assert redistributed_usd_has_no_standard_values(ds) == []

    (ds / "assets/declared.usd").write_text('#usda 1.0\ndef Xform "Box" {\n  float physics:mass\n}\n', encoding="utf-8")
    assert redistributed_usd_has_no_standard_values(ds) == []
    (ds / "assets/own.usd").write_text('#usda 1.0\ndef Xform "Box" {\n  def Xform "lid" {\n'
                                       '    custom float keti:springConstant = 1\n  }\n}\n', encoding="utf-8")
    (ds / "assets/sampled.usd").write_text('#usda 1.0\ndef Xform "Box" {\n  float physics:mass.timeSamples = {\n'
                                           '    0: 2,\n  }\n}\n', encoding="utf-8")
    problems = redistributed_usd_has_no_standard_values(ds)
    assert len(problems) == 2, problems
    assert "assets/own.usd" in problems[0] and "keti:springConstant" in problems[0]
    assert "assets/sampled.usd" in problems[1] and "physics:mass" in problems[1]


def test_every_redistributable_usd_passes_the_standard_value_scan(tmp_path):
    """The pre-flight scans every assets/*.usd (redistributed_usd_has_no_standard_values):
    USD is outside the repository's text scans, and a distribution is CC BY 4.0. Which USDs can
    ship is the canon's redistributed list -- read from it, so the S1d scene and skinned dummy
    are in it and so is whatever becomes redistributable next. Each must pass with
    the record the canon keeps (usd_checked), which is what a pxr-less interpreter inherits.
    Only on a machine with the builds (*.usd is not tracked)."""
    import shutil

    from generation.publish_dataset import redistributed_usd_has_no_standard_values

    repo = Path(__file__).resolve().parents[2]
    canon = json.loads((repo / "meta/assets.json").read_text(encoding="utf-8"))
    # A robot that ships as its own folder is not scanned (assets/*.usd only): its pins hold it.
    entries = [e for g in ("robots", "scene") for e in (canon.get(g) or {}).values()
               if e.get("redistributed") and "ship" not in e]
    assert len(entries) >= 3, "재배포 자산이 v1 더미 · 입힌 더미 · 물류 씬보다 적습니다"
    built = [e for e in entries if (repo / e["usd_path"]).is_file()]
    if not built:
        pytest.skip("재배포 자산이 빌드되어 있지 않습니다")
    ds = tmp_path / "dist"
    (ds / "assets").mkdir(parents=True)
    for e in built:
        shutil.copy2(repo / e["usd_path"], ds / "assets" / Path(e["usd_path"]).name)
    verified = {Path(e["usd_path"]).name: e["usd_checked"]["sha256"] for e in built}
    assert redistributed_usd_has_no_standard_values(ds, verified) == []
    assert sorted(p.name for p in (ds / "assets").glob("*.usd")) == sorted(verified)


# ── the camera canon and the pack index a camera distribution ships ──

def _camera_dist(tmp_path, repo, packs=True):
    """A camera distribution's two meta files, as the converter writes them: the derived
    cameras.json, and packs.json over two small packs. The camera canon is the repository's with
    its packs switched off; ``packs`` declares the fixture's pack in it
    (make_fixture.canon_with_pack, a literal -- never the repository canon's own) and ships the
    pack files; False ships neither."""
    import hashlib

    import numpy as np

    from generation import cameras as camera_canon
    from generation.convert_lerobot import shipped_cameras
    from validator.make_fixture import canon_with_pack

    ds = tmp_path / "dist"
    (ds / "meta").mkdir(parents=True)
    canon = camera_canon.load(repo / "meta" / "cameras.json")
    canon.pop("packs", None)
    doc = shipped_cameras(canon_with_pack(canon) if packs else canon)
    (ds / "meta" / "cameras.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    if not packs:
        return ds
    listed = {}
    (name,) = doc["packs"]
    for i in range(2):
        rel = f"packs/{name}/episode_{i:06d}.npz"
        (ds / rel).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ds / rel, step_index=np.arange(3, dtype=np.int32) + i)
        listed[str(i)] = {"file": rel, "frames": 3,
                          "sha256": hashlib.sha256((ds / rel).read_bytes()).hexdigest()}
    (ds / "meta" / "packs.json").write_text(json.dumps(listed, indent=2) + "\n", encoding="utf-8")
    return ds


def test_a_stale_shipped_camera_canon_is_refused(tmp_path):
    """Re-derived with the converter's own function: the canon plus the intrinsics. The shipped
    copy starts as exactly that derivation of the repository canon, whatever it declares."""
    from generation import cameras as camera_canon
    from generation.convert_lerobot import shipped_cameras
    from generation.publish_dataset import shipped_camera_canon_problems

    repo = Path(__file__).resolve().parents[2]
    ds = tmp_path / "dist"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "cameras.json").write_text(
        json.dumps(shipped_cameras(camera_canon.load(repo / "meta" / "cameras.json")),
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert shipped_camera_canon_problems(ds, repo) == []

    doc = json.loads((ds / "meta" / "cameras.json").read_text(encoding="utf-8"))
    cam = next(iter(doc["cameras"]))
    doc["cameras"][cam]["fov_h_deg"] += 1.0            # a canon edit the copy did not follow
    (ds / "meta" / "cameras.json").write_text(json.dumps(doc), encoding="utf-8")
    problems = shipped_camera_canon_problems(ds, repo)
    assert any("지금 카메라 정본에서 유도한 것과 다릅니다(다른 항목 ['cameras'])" in x
               for x in problems), problems

    # The repository canon verbatim is not what ships: it lacks the derived intrinsics.
    (ds / "meta" / "cameras.json").write_bytes((repo / "meta" / "cameras.json").read_bytes())
    assert shipped_camera_canon_problems(ds, repo)


def test_a_camera_less_distribution_is_not_asked_for_camera_files(tmp_path):
    from generation.publish_dataset import (shipped_camera_canon_problems,
                                            shipped_packs_match_index)

    repo = Path(__file__).resolve().parents[2]
    ds = _min_dist(tmp_path, repo)
    assert shipped_camera_canon_problems(ds, repo) == []
    assert shipped_packs_match_index(ds) == []


def test_every_pack_the_index_names_is_there_with_its_hash(tmp_path):
    from generation.publish_dataset import shipped_packs_match_index

    repo = Path(__file__).resolve().parents[2]
    ds = _camera_dist(tmp_path, repo)
    assert shipped_packs_match_index(ds) == []

    pack = ds / "packs/head/episode_000001.npz"
    pack.write_bytes(pack.read_bytes() + b"\0")
    problems = shipped_packs_match_index(ds)
    assert problems == [p for p in problems if "meta/packs.json[1]" in p] and \
        any("의 SHA-256 이 목록과 다릅니다" in p for p in problems), problems

    pack.unlink()
    assert any("목록이 가리키는 팩이 배포본에 없습니다: packs/head/episode_000001.npz" in p
               for p in shipped_packs_match_index(ds))

    listed = json.loads((ds / "meta/packs.json").read_text(encoding="utf-8"))
    listed["1"]["file"] = "../elsewhere.npz"
    (ds / "meta/packs.json").write_text(json.dumps(listed), encoding="utf-8")
    assert any("배포본 안의 상대 경로가 아닙니다" in p for p in shipped_packs_match_index(ds))

    (ds / "meta/packs.json").unlink()
    assert any("meta/packs.json 이 없습니다" in p for p in shipped_packs_match_index(ds))


@pytest.mark.parametrize("stray", [["packs/"], ["meta/packs.json"], ["packs/", "meta/packs.json"]],
                         ids=["dir", "index", "both"])
@pytest.mark.parametrize("canon", ["packless", "camera_less"])
def test_packs_the_canon_does_not_declare_are_refused(tmp_path, canon, stray):
    """Packs ship only where the distribution's meta/cameras.json declares one. Work
    done on top of an old conversion's copy leaves packs/ or meta/packs.json behind; published,
    a consumer finds packs no canon describes. A camera-less distribution declares none either."""
    from generation.publish_dataset import shipped_packs_match_index

    repo = Path(__file__).resolve().parents[2]
    ds = _camera_dist(tmp_path, repo, packs=False) if canon == "packless" else _min_dist(tmp_path, repo)
    assert shipped_packs_match_index(ds) == []
    if "packs/" in stray:
        (ds / "packs" / "head").mkdir(parents=True)
    if "meta/packs.json" in stray:
        (ds / "meta").mkdir(exist_ok=True)
        (ds / "meta/packs.json").write_text("{}", encoding="utf-8")
    problems = shipped_packs_match_index(ds)
    assert len(problems) == 1 and f"정본이 팩을 선언하지 않았는데 {stray} 가 있습니다" in problems[0], problems


@pytest.mark.parametrize("text", ["{", "[]"], ids=["unreadable", "not_an_object"])
def test_a_camera_canon_that_cannot_say_whether_it_declares_a_pack_is_not_passed(tmp_path, text):
    # Read to learn whether packs are owed; a document that cannot say is not taken as "none".
    from generation.publish_dataset import shipped_packs_match_index

    repo = Path(__file__).resolve().parents[2]
    ds = _camera_dist(tmp_path, repo, packs=False)
    (ds / "meta" / "cameras.json").write_text(text, encoding="utf-8")
    problems = shipped_packs_match_index(ds)
    assert len(problems) == 1 and "팩을 선언했는지 알 수 없습니다" in problems[0], problems


def test_preflight_runs_the_camera_and_pack_checks(tmp_path):
    """Through preflight(), not only the helpers -- a helper can be tested while its call is
    gone. The camera fixture's cameras.json is its own shrunk canon, not the repository's
    derivation, so a pre-flight that looks must say so; its packs match their list. The fixture
    is clean_media_pack, the clean camera fixture whose canon declares a pack."""
    import shutil

    from generation.publish_dataset import preflight

    repo = Path(__file__).resolve().parents[2]
    src = repo / ".tmp" / "fixtures" / "clean_media_pack"
    if not src.is_dir():
        pytest.skip("clean_media_pack 픽스처가 없습니다 — 먼저 `make fixture` 를 돌리십시오")
    ds = tmp_path / "dist"
    shutil.copytree(src, ds)
    listed = json.loads((ds / "meta/packs.json").read_text(encoding="utf-8"))
    first = ds / next(iter(listed.values()))["file"]
    first.write_bytes(first.read_bytes() + b"\0")
    problems = preflight(ds, repo, repro_hook="")
    assert any("meta/cameras.json 이 지금 카메라 정본에서 유도한 것과 다릅니다" in x for x in problems), problems
    assert any("의 SHA-256 이 목록과 다릅니다" in x and x.startswith("meta/packs.json[")
               for x in problems), problems


# ── 점검: 검증기의 죽음과 판정 ─────────────────────────────────────
#
# 검증기는 실데이터에서 간헐적으로 죽는다 (2026-10-04: 다섯 번 중 둘, SIGSEGV).
# 죽음은 판정이 아니므로 「검증 실패」로 읽혀도 안 되고 조용히 묻혀도 안 된다.

_REPO = Path(__file__).resolve().parents[2]
_HOOK = str(_REPO / "generation" / "rerun_seed.sh")


class FakeValidator:
    """`subprocess.run` 자리에 서는 가짜 검증기.

    종료 코드를 차례로 돌려주고, 실물처럼 **넘겨받은 stdout 파일에** 보고를 쓴다.
    반환값에만 실으면 생산 코드가 로그 파일을 읽지 않아도 초록이 된다.
    """

    OK = " ✓  C9 시드 재현성             OK\n"
    SKIP = " -  C9 시드 재현성             SKIP\n"

    def __init__(self, codes, report=OK, reports=None):
        self.codes = list(codes)
        self.report = report
        # exit code -> report, for a verdict other than a pass
        self.reports = dict(reports or {})
        self.commands = []

    def __call__(self, cmd, stdout=None, stderr=None, text=None, cwd=None):
        self.commands.append([str(c) for c in cmd])
        code = self.codes[len(self.commands) - 1]
        stdout.write(f"attempt {len(self.commands)} exit {code}\n")
        if code in self.reports:
            stdout.write(self.reports[code])
        else:
            stdout.write(self.report if code == 0 else "Fatal Python error: Segmentation fault\n")
        return types.SimpleNamespace(returncode=code)


@pytest.mark.parametrize("code,died", [
    (-11, True),     # SIGSEGV — what happened twice on 2026-10-04
    (-6, True),      # SIGABRT — the C++ abort
    (-7, True), (-8, True), (-4, True),     # SIGBUS · SIGFPE · SIGILL — crashes as well
    (3, True),       # the validator's own "I broke" exit
    (0, False), (1, False), (2, False),
    (143, False),    # 128 + SIGTERM: somebody stopped it on purpose, while it waited for the C9 hook
    # Outside the hook's wait the same stops arrive as a negative code.
    (-15, False), (-2, False), (-9, False), (-1, False),
    (-13, False),    # SIGPIPE: neither a crash nor a stop we know; not retried
])
def test_a_validator_that_died_is_told_from_one_that_gave_a_verdict(code, died):
    from generation.publish_dataset import validator_died

    assert validator_died(code) is died


@pytest.mark.parametrize("code,name", [
    (-15, "SIGTERM"), (-2, "SIGINT"), (-9, "SIGKILL"), (-1, "SIGHUP"),
    (143, "SIGTERM"), (129, "SIGHUP"), (130, "SIGINT"),
])
def test_a_validator_somebody_stopped_is_not_run_again_nor_called_a_failed_validation(
        tmp_path, code, name):
    """Ctrl+C 나 kill 로 세운 검증기를 두 번 더 띄우면 C9 하나에 수십 분이다. 그리고 C9 훅을
    기다리던 검증기는 SIGTERM 을 143 으로 끝내는데, 그것은 「통과하지 못했습니다」로 읽혔다."""
    fake = FakeValidator([code, 0, 0])
    lines = _validator_lines(_preflight_with(tmp_path, fake))
    assert len(fake.commands) == 1
    assert len(lines) == 1, lines
    assert "멈췄습니다" in lines[0] and name in lines[0] and "판정이 아닙니다" in lines[0]
    assert "통과하지 못했습니다" not in lines[0]
    assert "validate-attempt-1.log" in lines[0]


def test_a_killed_validator_is_said_to_maybe_have_run_out_of_memory(tmp_path):
    (line,) = _validator_lines(_preflight_with(tmp_path, FakeValidator([-9])))
    assert "메모리" in line


def test_a_code_that_is_no_verdict_is_not_called_a_failed_validation(tmp_path):
    fake = FakeValidator([127], reports={127: "python: No such file or directory\n"})
    lines = _validator_lines(_preflight_with(tmp_path, fake))
    assert len(fake.commands) == 1 and len(lines) == 1, lines
    assert "판정이 아닙니다" in lines[0] and "exit 127" in lines[0]
    assert "통과하지 못했습니다" not in lines[0]
    assert "No such file" in lines[0]


def test_deaths_before_a_stop_are_named_and_no_verdict_is_claimed(tmp_path):
    notes = []
    fake = FakeValidator([-11, -15])
    lines = _validator_lines(_preflight_with(tmp_path, fake, notes=notes))
    assert len(fake.commands) == 2
    assert len(lines) == 1 and "SIGTERM" in lines[0] and "validate-attempt-1.log" in lines[0], lines
    assert notes == [], "a death followed by a stop gave no verdict"


def test_the_broke_code_is_the_one_the_validator_exits_with():
    """두 파일이 같은 숫자를 따로 든다 — 검증기는 generation 을 import 하지 않는다."""
    import inspect
    import re

    from generation.publish_dataset import VALIDATOR_BROKE
    from validator import validate_dataset

    source = inspect.getsource(validate_dataset.main)
    found = re.findall(r"검증기 실행 실패.*?return (\d+)", source, flags=re.S)
    assert found == [str(VALIDATOR_BROKE)], found


def test_a_death_is_followed_by_another_attempt_and_the_verdict_is_kept(tmp_path):
    from generation.publish_dataset import run_validator

    fake = FakeValidator([-11, -11, 0])
    code, text, died, last = run_validator(["v"], tmp_path, tmp_path / "logs", run=fake)
    assert code == 0 and len(fake.commands) == 3
    assert last.name == "validate-attempt-3.log"
    assert "attempt 3 exit 0" in text and "attempt 1" not in text
    assert [p.name for p in died] == ["validate-attempt-1.log", "validate-attempt-2.log"]
    assert "attempt 1 exit -11" in died[0].read_text(encoding="utf-8")
    assert "attempt 2 exit -11" in died[1].read_text(encoding="utf-8")


@pytest.mark.parametrize("code", [1, 2, 143])
def test_a_verdict_is_not_asked_for_twice(tmp_path, code):
    from generation.publish_dataset import run_validator

    fake = FakeValidator([code])
    got, _, died, _ = run_validator(["v"], tmp_path, tmp_path / "logs", run=fake)
    assert (got, len(fake.commands), died) == (code, 1, [])


def test_the_attempts_end(tmp_path):
    from generation.publish_dataset import VALIDATOR_ATTEMPTS, run_validator, validator_died

    fake = FakeValidator([-11] * (VALIDATOR_ATTEMPTS + 2))
    code, _, died, _ = run_validator(["v"], tmp_path, tmp_path / "logs", run=fake)
    assert validator_died(code)
    assert len(fake.commands) == VALIDATOR_ATTEMPTS == len(died)


def test_a_log_directory_used_before_keeps_its_logs(tmp_path):
    """재부팅 뒤 같은 --log-dir 로 다시 돌리면, 복구하려던 그 크래시의 로그가 남아야 한다."""
    from generation.publish_dataset import run_validator

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validate-attempt-2.log").write_text("the crash before the reboot", encoding="utf-8")
    run_validator(["v"], tmp_path, logs, run=FakeValidator([0]))
    assert (logs / "validate-attempt-2.log").read_text(encoding="utf-8") == "the crash before the reboot"
    assert (logs / "validate-attempt-3.log").exists()


def _preflight_with(tmp_path, fake, **kw):
    from generation.publish_dataset import preflight

    return preflight(_min_dist(tmp_path, _REPO), _REPO, repro_hook="",
                     log_dir=tmp_path / "logs", run=fake, **kw)


def _validator_lines(problems):
    return [x for x in problems if x.startswith("검증기가")]


def test_preflight_does_not_call_a_dead_validator_a_failed_validation(tmp_path):
    lines = _validator_lines(_preflight_with(tmp_path, FakeValidator([-11, 3, -11])))
    assert len(lines) == 1, lines
    assert "죽었습니다" in lines[0] and "판정이 아닙니다" in lines[0]
    assert "통과하지 못했습니다" not in lines[0]
    assert "validate-attempt-1.log" in lines[0] and "validate-attempt-3.log" in lines[0]


def test_preflight_takes_the_verdict_that_followed_a_death_and_says_there_was_one(tmp_path):
    notes = []
    problems = _preflight_with(tmp_path, FakeValidator([-11, 0]), notes=notes)
    assert _validator_lines(problems) == []
    assert len(notes) == 1 and "1번 죽은 뒤" in notes[0] and "validate-attempt-1.log" in notes[0]


def test_a_run_without_a_death_leaves_no_note(tmp_path):
    notes = []
    _preflight_with(tmp_path, FakeValidator([0]), notes=notes)
    assert notes == []


def test_preflight_still_refuses_a_skipped_c9_after_a_death(tmp_path):
    problems = _preflight_with(tmp_path, FakeValidator([-11, 0], report=FakeValidator.SKIP))
    assert any("C9 재현성이 SKIP" in x for x in problems), problems


def test_a_real_failure_keeps_its_own_sentence(tmp_path):
    lines = _validator_lines(_preflight_with(tmp_path, FakeValidator([1])))
    assert len(lines) == 1 and "통과하지 못했습니다 (exit 1)" in lines[0]


def test_preflight_asks_for_the_report_of_the_run_that_gated(tmp_path):
    fake = FakeValidator([0])
    _preflight_with(tmp_path, fake)
    cmd = fake.commands[0]
    assert cmd[cmd.index("--json") + 1] == str(tmp_path / "logs" / "validate-attempt-1.json")
    assert "--verbose" in cmd


def _report_of(cmd):
    assert cmd.count("--json") == 1, cmd
    return cmd[cmd.index("--json") + 1]


def test_each_attempt_writes_its_report_beside_its_log(tmp_path):
    """한 파일을 모든 시도가 쓰면, 죽은 시도는 보고서를 안 쓰므로 그 자리에 앞 실행의 것이 남는다."""
    from generation.publish_dataset import run_validator

    fake = FakeValidator([-11, 0])
    run_validator(["v"], tmp_path, tmp_path / "logs", run=fake)
    assert [_report_of(c) for c in fake.commands] == [
        str(tmp_path / "logs" / "validate-attempt-1.json"),
        str(tmp_path / "logs" / "validate-attempt-2.json")]


def test_a_log_directory_used_before_keeps_its_reports(tmp_path):
    """같은 --log-dir 로 다시 돌리면 validate.json 이 덮어써졌다."""
    from generation.publish_dataset import run_validator

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validate-attempt-1.log").write_text("the run before", encoding="utf-8")
    (logs / "validate-attempt-1.json").write_text('{"before": true}', encoding="utf-8")
    fake = FakeValidator([0])
    run_validator(["v"], tmp_path, logs, run=fake)
    assert _report_of(fake.commands[0]) == str(logs / "validate-attempt-2.json")
    assert (logs / "validate-attempt-1.json").read_text(encoding="utf-8") == '{"before": true}'


def test_a_relative_log_directory_keeps_the_logs_and_the_report_together(tmp_path, monkeypatch):
    """검증기는 저장소 루트에서 돈다. 상대 경로를 그대로 넘기면 로그는 호출한 자리에,
    보고서는 저장소 루트에 갈라져 놓였다."""
    from generation.publish_dataset import preflight

    dist = _min_dist(tmp_path, _REPO)
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    fake = FakeValidator([0])
    problems = preflight(Path("..") / dist.name, _REPO, repro_hook="", log_dir=Path("rel/logs"), run=fake)
    # The byte comparisons found the dataset (the minimal one lacks other files on purpose).
    assert "동봉되어야 할 파일이 없습니다: meta/safety_schema.json" not in problems, problems
    cmd = fake.commands[0]
    assert (caller / "rel" / "logs" / "validate-attempt-1.log").exists()
    assert _report_of(cmd) == str(caller / "rel" / "logs" / "validate-attempt-1.json")
    # The dataset the validator is pointed at is the one the byte comparisons read.
    assert cmd[2] == str(dist), cmd


def test_a_verbose_report_does_not_confuse_the_c9_skip_reader():
    """점검이 --verbose 를 넘기므로 INFO · SKIP 줄이 함께 온다 (2026-10-04 · 10-06 의 실제 줄)."""
    from generation.publish_dataset import c9_skipped

    passed = (" ✓  C8 값 범위·이상치           OK\n ✓  C9 시드 재현성             OK\n"
              "INFO  C9  시드 재현 확인 — 재현 확인: S1d_0132_seed30756832 · 23 컬럼 · 999 프레임 · "
              "RGB 영상은 대조하지 않음 (실행마다 렌더가 달라 비트 재현되지 않음)\n")
    skipped = (" -  C9 시드 재현성             SKIP\n"
               "SKIP  C9  재현성 훅 미지정 (--repro-hook). 배포 전 반드시 1회 수행할 것\n")
    assert not c9_skipped(passed)
    assert c9_skipped(skipped)


def test_the_default_log_directory_sits_beside_the_dataset(tmp_path):
    from generation.publish_dataset import default_log_dir

    assert default_log_dir(tmp_path / "batch" / "dataset").parent == tmp_path / "batch" / "publish"


def _run_main(monkeypatch, argv, problems=(), note=None, publish_rc=0):
    """진입점을 그 자리에서 돌린다 — 무거운 두 절반만 갈아 끼우고."""
    import sys

    import generation.publish_dataset as pub

    seen = {"preflight": [], "publish": []}

    def fake_preflight(*a, **k):
        seen["preflight"].append((a, k))
        if note is not None and k.get("notes") is not None:
            k["notes"].append(note)
        return list(problems)

    def fake_publish(*a, **k):
        seen["publish"].append((a, k))
        return publish_rc

    monkeypatch.setattr(pub, "preflight", fake_preflight)
    monkeypatch.setattr(pub, "publish", fake_publish)
    monkeypatch.setattr(sys, "argv", ["publish_dataset.py", *argv])
    return pub.main(), seen


def _ds_dir(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    return ds


def test_preflight_only_stops_before_the_hub(tmp_path, monkeypatch):
    rc, seen = _run_main(monkeypatch, ["--dataset", str(_ds_dir(tmp_path)),
                                       "--preflight-only", "--repro-hook", _HOOK])
    assert rc == 0 and len(seen["preflight"]) == 1 and seen["publish"] == []


def test_without_preflight_only_the_same_command_goes_on_to_publish(tmp_path, monkeypatch):
    rc, seen = _run_main(monkeypatch, ["--dataset", str(_ds_dir(tmp_path)), "--repo-id", "o/n",
                                       "--repro-hook", _HOOK])
    assert rc == 0 and len(seen["preflight"]) == 1 and len(seen["publish"]) == 1


def test_preflight_only_still_fails_when_the_preflight_does(tmp_path, monkeypatch):
    rc, seen = _run_main(monkeypatch, ["--dataset", str(_ds_dir(tmp_path)),
                                       "--preflight-only", "--repro-hook", _HOOK], problems=["x"])
    assert rc == 1 and seen["publish"] == []


def test_preflight_only_and_skip_preflight_contradict_each_other(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as stop:
        _run_main(monkeypatch, ["--dataset", str(_ds_dir(tmp_path)),
                                "--preflight-only", "--skip-preflight"])
    assert stop.value.code == 2


def test_publishing_still_needs_a_repository(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as stop:
        _run_main(monkeypatch, ["--dataset", str(_ds_dir(tmp_path)), "--repro-hook", _HOOK])
    assert stop.value.code == 2


def test_main_hands_the_log_directory_to_preflight(tmp_path, monkeypatch):
    _, seen = _run_main(monkeypatch, ["--dataset", str(_ds_dir(tmp_path)), "--preflight-only",
                                      "--repro-hook", _HOOK, "--log-dir", str(tmp_path / "L")])
    assert seen["preflight"][0][1]["log_dir"] == tmp_path / "L"


def test_main_prints_what_the_preflight_noted(tmp_path, monkeypatch, capsys):
    _run_main(monkeypatch, ["--dataset", str(_ds_dir(tmp_path)), "--preflight-only",
                            "--repro-hook", _HOOK], note="검증기가 1번 죽은 뒤 판정을 냈습니다")
    assert "검증기가 1번 죽은 뒤 판정을 냈습니다" in capsys.readouterr().out


# ── 올리기 전에: 리포의 정체와 남는 파일 ───────────────────────────


def test_the_extras_are_what_the_repository_holds_and_the_distribution_does_not():
    from generation.publish_dataset import remote_extras

    local = {"README.md": "1", "meta/info.json": "2"}
    assert remote_extras(["README.md", ".gitattributes", "b/old.usd", "a.txt"], local) == [
        "a.txt", "b/old.usd"]
    assert remote_extras([".gitattributes"], local) == []
    assert remote_extras([], local) == []


def test_a_redirected_name_is_refused_before_anything_is_uploaded(tmp_path, capsys):
    """이름을 바꾼 리포의 옛 이름은 새 위치로 답한다 — 그대로 올리면 그 리포에 올라간다."""
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    api = FakeApi(resolves_to="o/n-s1a", calls=calls)
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=api, download=_download_of(_mirror(root), calls))
    assert rc == 1
    assert "upload_folder" not in calls and "list_repo_files" not in calls
    err = capsys.readouterr().err
    assert "o/n-s1a" in err and "리다이렉트" in err


def test_a_redirecting_name_is_not_taken_over_by_a_new_repository(tmp_path, capsys):
    """옛 이름에 create_repo 를 부르면 Hub 은 그 이름에 새 리포를 만들고 리다이렉트를 없앤다 —
    그러면 「답한 이름」 검사를 지나 새 리포에 올라간다. 이름은 만들기 전에 묻는다."""
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    api = FakeApi(resolves_to="o/n-public", create_takes_the_name=True, calls=calls)
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=api, download=_download_of(_mirror(root), calls))
    assert rc == 1
    assert api.asked_exists == [("dataset", "o/n")]
    assert "create_repo" not in calls and "upload_folder" not in calls
    err = capsys.readouterr().err
    assert "o/n-public" in err and "리다이렉트" in err
    assert "아무것도 만들거나 올리지 않았습니다" in err


def test_a_name_not_on_the_hub_is_created_then_published(tmp_path):
    """없는 이름은 묻기만 하고(읽지 않고) 만든다 — 묻는 자리가 새 리포를 막으면 안 된다."""
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    api = FakeApi(exists=False, calls=calls)
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=api, download=_download_of(_mirror(root), calls))
    assert rc == 0
    assert api.asked_exists == [("dataset", "o/n")]
    assert calls.index("repo_exists") < calls.index("create_repo") < calls.index("upload_folder")
    assert calls.index("create_repo") < calls.index("repo_info")


def test_a_name_that_answers_as_another_once_created_is_refused(tmp_path, capsys):
    """두 물음 사이에 이름이 다른 리포로 답하게 되면(만들기가 거부되고 삼켜진 경우도) 올리지 않는다."""
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    api = FakeApi(exists=False, resolves_to="o/other", calls=calls)
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=api, download=_download_of(_mirror(root), calls))
    assert rc == 1
    assert "create_repo" in calls and "upload_folder" not in calls
    err = capsys.readouterr().err
    assert "o/other" in err
    # create_repo was called: an empty repository may be there now, and the refusal says so.
    assert "만들거나" not in err and "빈 리포가 생겼을 수 있습니다" in err


def test_a_repository_that_does_not_say_its_name_is_not_passed(tmp_path):
    from generation.publish_dataset import publish

    class Nameless(FakeApi):
        def repo_info(self, **kw):
            self.calls.append("repo_info")
            return types.SimpleNamespace(private=self._private)

    root = _dataset(tmp_path)
    calls = []
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=Nameless(calls=calls), download=_download_of(_mirror(root), calls))
    assert rc == 1 and "upload_folder" not in calls


def test_stale_remote_files_stop_the_upload_before_it_starts(tmp_path, capsys):
    """왕복 대조도 이것을 잡지만 22 GB 를 올리고 내려받은 뒤다. 단언은 호출 기록으로 —
    종료 코드만 보면 왕복 대조가 대신 거부해도 초록이다."""
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    api = FakeApi(remote_files=["README.md", ".gitattributes", "assets/old.usd"], calls=calls)
    stale = dict(_mirror(root))
    stale["assets/old.usd"] = b"left over"
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=api, download=_download_of(stale, calls))
    assert rc == 1
    assert "upload_folder" not in calls and "download" not in calls
    assert "올린 적 없는 파일: assets/old.usd" in capsys.readouterr().err


def test_a_new_repository_holds_only_what_the_hub_put_there(tmp_path):
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=FakeApi(remote_files=[".gitattributes"], calls=calls),
                 download=_download_of(_mirror(root), calls))
    assert rc == 0


def test_a_partly_uploaded_repository_is_resumed_not_refused(tmp_path):
    """끊긴 업로드는 배포본의 일부만 남긴다. 그것은 남는 파일이 아니다."""
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    rc = publish(root, "o/n", private=True, message="m", verify=True,
                 api=FakeApi(remote_files=[".gitattributes", "README.md"], calls=calls),
                 download=_download_of(_mirror(root), calls))
    assert rc == 0 and "upload_folder" in calls


def test_the_name_is_read_then_the_listing_then_the_upload(tmp_path):
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    calls = []
    publish(root, "o/n", private=True, message="m", verify=True,
            api=FakeApi(calls=calls), download=_download_of(_mirror(root), calls))
    assert calls.index("repo_info") < calls.index("list_repo_files") < calls.index("upload_folder")


# ── 릴리스 태그 ────────────────────────────────────────────────────
#
# v3.0 은 LeRobot 이 요구하는 태그라 게시마다 옮긴다. 릴리스 태그는 공개본이
# 가리키는 고정점이라 한 번 걸고 옮기지 않는다.


@pytest.mark.parametrize("tag", ["0.1.0", "v0.1", "v0.1.0-rc1", "latest", "", "v0.1.0 "])
def test_an_ill_formed_release_tag_is_named(tag):
    from generation.publish_dataset import release_tag_problem

    assert release_tag_problem(tag, "v3.0") is not None


def test_a_release_tag_is_not_the_load_tag():
    from generation.publish_dataset import release_tag_problem

    assert release_tag_problem("v0.1.0", "v3.0") is None
    assert "codebase_version" in release_tag_problem("v3.0.0", "v3.0.0")


def _publish_release(tmp_path, api, calls, tag="v0.1.0", verify=True, files=None):
    from generation.publish_dataset import publish

    root = _dataset(tmp_path)
    return publish(root, "o/n", private=True, message="m", verify=verify, api=api,
                   download=_download_of(_mirror(root) if files is None else files, calls),
                   release_tag=tag)


def test_the_release_tag_is_pinned_after_the_round_trip_and_the_load_tag(tmp_path):
    calls = []
    api = FakeApi(calls=calls)
    assert _publish_release(tmp_path, api, calls) == 0
    assert api.tags == {"v3.0": OID, "v0.1.0": OID}
    assert api.tag_log == ["v3.0", "v0.1.0"]
    assert calls.index("download") < calls.index("create_tag")


def test_a_release_tag_on_another_commit_stops_before_the_load_tag_moves(tmp_path, capsys):
    """적재용 태그만 새 내용으로 넘어가고 릴리스가 옛 내용에 남는 상태를 만들지 않는다."""
    calls = []
    held = {"v3.0": "b" * 40, "v0.1.0": "b" * 40}
    api = FakeApi(tags=held, calls=calls)
    assert _publish_release(tmp_path, api, calls) == 1
    assert api.tags == held
    assert "delete_tag" not in calls and "create_tag" not in calls
    err = capsys.readouterr().err
    assert "릴리스 태그 v0.1.0" in err and "옮기지 않습니다" in err


def test_a_release_tag_already_on_this_commit_is_left_alone(tmp_path, capsys):
    calls = []
    api = FakeApi(tags={"v3.0": OID, "v0.1.0": OID}, calls=calls)
    assert _publish_release(tmp_path, api, calls) == 0
    assert "delete_tag" not in calls and "create_tag" not in calls
    assert capsys.readouterr().err == "", "a re-run with everything in place has nothing to warn about"


def test_an_ill_formed_release_tag_is_refused_before_the_hub_is_touched(tmp_path):
    calls = []
    assert _publish_release(tmp_path, FakeApi(calls=calls), calls, tag="latest") == 1
    assert calls == []


def test_a_release_tag_is_not_pinned_without_the_round_trip(tmp_path):
    calls = []
    assert _publish_release(tmp_path, FakeApi(calls=calls), calls, verify=False) == 1
    assert calls == []


def test_a_release_tag_is_not_pinned_on_a_failed_round_trip(tmp_path):
    calls = []
    api = FakeApi(calls=calls)
    assert _publish_release(tmp_path, api, calls, files={"README.md": b"changed"}) == 1
    assert api.tags == {}


def test_a_release_tag_created_despite_the_error_is_reported_as_created(tmp_path):
    calls = []
    api = FakeApi(tags={"v3.0": OID}, create_tag_applies_then_raises=True, calls=calls)
    assert _publish_release(tmp_path, api, calls) == 0
    assert api.tags == {"v3.0": OID, "v0.1.0": OID}


def test_a_release_tag_that_was_not_created_comes_with_a_command_that_exists(tmp_path, capsys):
    calls = []
    api = FakeApi(tags={"v3.0": OID}, create_tag_raises=True, calls=calls)
    assert _publish_release(tmp_path, api, calls) == 1
    err = capsys.readouterr().err
    assert f"hf repos tag create o/n v0.1.0 --revision {OID} --repo-type dataset" in err


def test_refs_that_cannot_be_read_stop_before_any_tag(tmp_path):
    calls = []
    api = FakeApi(tags={"v3.0": "b" * 40}, refs_raises_from=1, calls=calls)
    assert _publish_release(tmp_path, api, calls) == 1
    assert "delete_tag" not in calls and "create_tag" not in calls


def test_without_a_release_tag_only_the_load_tag_is_touched(tmp_path):
    calls = []
    api = FakeApi(calls=calls)
    assert _publish_release(tmp_path, api, calls, tag=None) == 0
    assert api.tags == {"v3.0": OID}


def test_main_hands_the_release_tag_to_publish(tmp_path, monkeypatch):
    # A dataset with meta/info.json: the entry point reads codebase_version to check the tag.
    _, seen = _run_main(monkeypatch, ["--dataset", str(_dataset(tmp_path)), "--repo-id", "o/n",
                                      "--repro-hook", _HOOK, "--release-tag", "v0.1.0"])
    assert seen["publish"][0][1]["release_tag"] == "v0.1.0"


# ── 릴리스 태그는 점검 전에 ─────────────────
#
# 점검은 20~90분이다. 그 뒤에야 태그 이름을 거부하면 그 시간을 다시 써야 하고,
# --preflight-only 로 공개 날의 명령을 미리 돌려 봐도 틀린 태그가 초록으로 지나갔다.


@pytest.mark.parametrize("version,extra,phrase", [
    ("v3.0", ["--release-tag", "latest"], "v숫자.숫자.숫자"),
    ("v3.0.0", ["--release-tag", "v3.0.0"], "codebase_version"),
    ("v3.0", ["--release-tag", "v0.1.0", "--no-verify"], "--no-verify"),
], ids=["ill-formed", "the load tag", "without the round trip"])
@pytest.mark.parametrize("mode", [["--repo-id", "o/n"], ["--preflight-only"]],
                         ids=["publish", "preflight-only"])
def test_a_release_tag_that_cannot_be_pinned_is_refused_before_the_preflight(
        tmp_path, monkeypatch, capsys, version, extra, phrase, mode):
    rc, seen = _run_main(monkeypatch, ["--dataset", str(_dataset(tmp_path, version=version)), *mode,
                                       "--repro-hook", _HOOK, *extra])
    assert rc == 1
    assert seen["preflight"] == [] and seen["publish"] == []
    assert phrase in capsys.readouterr().err


def test_a_release_tag_without_a_version_to_compare_is_refused_without_a_traceback(
        tmp_path, monkeypatch, capsys):
    rc, seen = _run_main(monkeypatch, ["--dataset", str(_ds_dir(tmp_path)), "--repo-id", "o/n",
                                       "--repro-hook", _HOOK, "--release-tag", "v0.1.0"])
    assert rc == 1 and seen["preflight"] == []
    assert "meta/info.json" in capsys.readouterr().err


def test_a_release_tag_that_can_be_pinned_goes_on_to_the_preflight(tmp_path, monkeypatch):
    rc, seen = _run_main(monkeypatch, ["--dataset", str(_dataset(tmp_path)), "--preflight-only",
                                       "--repro-hook", _HOOK, "--release-tag", "v0.1.0"])
    assert rc == 0 and len(seen["preflight"]) == 1 and seen["publish"] == []


def test_a_release_tag_that_appeared_meanwhile_is_not_asked_to_move(capsys):
    """충돌 확인과 태그 걸기 사이에 다른 작성자가 그 태그를 건 경우.

    Hub 는 있는 태그의 생성을 거부하므로 결과는 같지만, 옮기려는 요청을 보내지
    않고 무엇이 그 자리에 있는지 말해야 한다.
    """
    from generation.publish_dataset import _pin_release_tag

    calls = []
    api = FakeApi(tags={"v0.1.0": "b" * 40}, calls=calls)
    assert _pin_release_tag(api, "o/n", "v0.1.0", OID) == 1
    assert "create_tag" not in calls and "delete_tag" not in calls
    assert api.tags == {"v0.1.0": "b" * 40}
    assert "옮기지 않습니다" in capsys.readouterr().err


# ── 이력 감사 ──────────────────────────────────────────────────────
#
# 공개로 바뀌는 것은 head 가 아니라 리포다. 어떤 ref 에서든 닿는 커밋이 전부 함께
# 나가므로, 「무엇이 공개되는가」는 추측하지 않고 읽는다.


class FakeHistoryApi:
    """ref → 닿는 커밋들 → 커밋의 트리로 이루어진 리포.

    실물에 맞춘 것 다섯: PR ref 는 `include_pull_requests=True` 로 청하지 않으면
    `None` 이고, 트리에는 `blob_id` 가 없는 폴더 항목이 섞여 오고, 커밋 목록은
    revision 에서 닿는 것 전부이고, **revision 을 주지 않으면 main** 이며, 트리는
    **`recursive=True` 를 주지 않으면 맨 위 한 층**만 온다. 뒤의 둘을 가짜가 무시하던
    동안에는 생산 코드에서 그 인자를 지워도 초록이었다.
    """

    def __init__(self, branches, tags=None, pull_requests=None, reach=None, trees=None,
                 private=True, answers_as=None, commits_raise=False):
        self.branches, self.tags = dict(branches), dict(tags or {})
        self.pull_requests = dict(pull_requests or {})
        self.reach, self.trees = dict(reach or {}), dict(trees or {})
        self.private, self.answers_as, self.commits_raise = private, answers_as, commits_raise

    @staticmethod
    def _refs(named):
        return [types.SimpleNamespace(name=n, ref=n, target_commit=c) for n, c in named.items()]

    def repo_info(self, **kw):
        return types.SimpleNamespace(id=self.answers_as or kw.get("repo_id"), private=self.private)

    def list_repo_refs(self, repo_id, repo_type=None, include_pull_requests=False):
        return types.SimpleNamespace(
            branches=self._refs(self.branches), tags=self._refs(self.tags), converts=[],
            pull_requests=self._refs(self.pull_requests) if include_pull_requests else None)

    def list_repo_commits(self, repo_id, repo_type=None, revision=None):
        if self.commits_raise:
            raise FakeHubError("500 commits unavailable")
        revision = revision if revision is not None else self.branches["main"]
        return [types.SimpleNamespace(commit_id=c, title=f"commit {c}") for c in self.reach[revision]]

    def list_repo_tree(self, repo_id, repo_type=None, revision=None, recursive=False):
        files = self.trees[revision]
        folders = sorted({str(parent) for p in files for parent in Path(p).parents if str(parent) != "."})
        if not recursive:
            files = {p: b for p, b in files.items() if "/" not in p}
            folders = [f for f in folders if "/" not in f]
        yield from (types.SimpleNamespace(path=f, tree_id="t") for f in folders)
        yield from (types.SimpleNamespace(path=p, blob_id=b, size=1) for p, b in files.items())

    def list_repo_files(self, repo_id=None, repo_type=None, revision=None):
        return list(self.trees[self.branches["main"]])


def _history(**kw):
    """c1 → c2 → c3 on main. Tests override what they need."""
    base = dict(
        branches={"main": "c3"},
        reach={"c3": ["c3", "c2", "c1"]},
        trees={"c1": {"README.md": "r1", "meta/info.json": "i1"},
               "c2": {"README.md": "r1", "meta/info.json": "i1"},
               "c3": {"README.md": "r1", "meta/info.json": "i1"}})
    base.update(kw)
    return FakeHistoryApi(**base)


def test_a_history_that_only_ever_held_the_head_is_clean():
    from generation.publish_dataset import history_exposure

    report = history_exposure(_history(), "o/n")
    assert report["commits"] == 3
    assert report["history_only"] == {} and report["multi_version"] == {}


def test_a_path_that_left_the_head_is_reported_with_the_commits_that_held_it():
    from generation.publish_dataset import history_exposure

    api = _history(trees={
        "c1": {"README.md": "r1", "meta/joint_map/g1.yaml": "g"},
        "c2": {"README.md": "r1", "meta/joint_map/g1.yaml": "g"},
        "c3": {"README.md": "r1"}})
    assert history_exposure(api, "o/n")["history_only"] == {"meta/joint_map/g1.yaml": 2}


def test_a_path_whose_content_changed_is_counted_not_judged():
    from generation.publish_dataset import history_exposure

    api = _history(trees={"c1": {"README.md": "r1"}, "c2": {"README.md": "r2"},
                          "c3": {"README.md": "r2"}})
    report = history_exposure(api, "o/n")
    assert report["multi_version"] == {"README.md": 2} and report["history_only"] == {}


def test_commits_only_a_tag_reaches_are_walked():
    from generation.publish_dataset import history_exposure

    api = _history(branches={"main": "c3"}, tags={"old": "x1"},
                   reach={"c3": ["c3"], "x1": ["x1"]},
                   trees={"c3": {"README.md": "r1"}, "x1": {"README.md": "r1", "secret.bin": "s"}})
    report = history_exposure(api, "o/n")
    assert report["history_only"] == {"secret.bin": 1} and report["commits"] == 2


def test_commits_only_a_pull_request_reaches_are_walked():
    """PR ref 는 청해야 온다 — 청하지 않으면 그 커밋들은 목록에 없다."""
    from generation.publish_dataset import history_exposure

    api = _history(pull_requests={"refs/pr/1": "p1"},
                   reach={"c3": ["c3"], "p1": ["p1", "c3"]},
                   trees={"c3": {"README.md": "r1"}, "p1": {"README.md": "r1", "draft.md": "d"}})
    assert history_exposure(api, "o/n")["history_only"] == {"draft.md": 1}


def test_a_repository_without_pull_request_refs_is_read():
    from generation.publish_dataset import history_exposure

    api = _history()
    api.list_repo_refs = lambda repo_id, repo_type=None, include_pull_requests=False: (
        types.SimpleNamespace(branches=FakeHistoryApi._refs({"main": "c3"}), tags=[],
                              converts=[], pull_requests=[]))
    assert history_exposure(api, "o/n")["history_only"] == {}


def test_folders_are_not_paths():
    from generation.publish_dataset import history_exposure

    report = history_exposure(_history(), "o/n")
    assert "meta" not in report["history_only"] and "meta" not in report["multi_version"]


def test_the_audit_exits_one_on_a_withdrawn_path_and_zero_on_a_clean_history(capsys):
    from generation.publish_dataset import audit_history

    assert audit_history("o/n", api=_history()) == 0
    dirty = _history(trees={"c1": {"README.md": "r1", "gone.usd": "u"},
                            "c2": {"README.md": "r1"}, "c3": {"README.md": "r1"}})
    assert audit_history("o/n", api=dirty) == 1
    assert "gone.usd" in capsys.readouterr().out


def test_a_clean_audit_claims_only_what_it_walked(capsys):
    """exit 0 의 문장이 「head 밖의 파일은 따라 나가지 않는다」고 했다. 감사는 ref 에서 닿는
    커밋만 걷는다 — 닿지 않는 커밋은 목록에 없다."""
    from generation.publish_dataset import audit_history

    assert audit_history("o/n", api=_history()) == 0
    out = capsys.readouterr().out
    assert "따라 나가지 않습니다" not in out
    assert "닿지 않는 커밋" in out


def _hub_file_changed_once():
    trees = {c: {"README.md": "r1", ".gitattributes": "g2"} for c in ("c2", "c3")}
    trees["c1"] = {".gitattributes": "g1"}            # the Hub's own file, before the first upload
    return trees


def test_the_hub_s_own_file_does_not_ask_for_a_reading(capsys):
    """첫 업로드가 Hub 이 만든 .gitattributes 를 고치므로 모든 감사에 ×2 가 나와 「철회한 내용이
    있었는지 읽으십시오」를 불렀다."""
    from generation.publish_dataset import audit_history

    assert audit_history("o/n", api=_history(trees=_hub_file_changed_once())) == 0
    out = capsys.readouterr().out
    assert "읽으십시오" not in out
    hub_line = next(line for line in out.splitlines() if ".gitattributes" in line)
    assert "×2" in hub_line and "Hub" in hub_line


def test_a_file_of_ours_with_two_versions_still_asks_for_a_reading(capsys):
    from generation.publish_dataset import audit_history

    trees = _hub_file_changed_once()
    trees["c3"] = {"README.md": "r2", ".gitattributes": "g2"}
    assert audit_history("o/n", api=_history(trees=trees)) == 0
    out = capsys.readouterr().out
    assert "README.md  ×2" in out and "읽으십시오" in out


def test_an_unreadable_history_is_not_reported_as_clean(capsys):
    from generation.publish_dataset import audit_history

    assert audit_history("o/n", api=_history(commits_raise=True)) == 1
    assert "읽지 못했습니다" in capsys.readouterr().err


def test_the_audit_refuses_a_name_that_answers_for_another_repository(capsys):
    from generation.publish_dataset import audit_history

    assert audit_history("o/n", api=_history(answers_as="o/n-s1a")) == 1
    assert "o/n-s1a" in capsys.readouterr().err


def test_main_audits_without_a_dataset(monkeypatch):
    import sys

    import generation.publish_dataset as pub

    asked = []
    monkeypatch.setattr(pub, "audit_history", lambda repo_id: asked.append(repo_id) or 0)
    monkeypatch.setattr(pub, "publish", lambda *a, **k: pytest.fail("the audit must not publish"))
    monkeypatch.setattr(pub, "preflight", lambda *a, **k: pytest.fail("nor run the preflight"))
    monkeypatch.setattr(sys, "argv", ["publish_dataset.py", "--audit-history", "--repo-id", "o/n"])
    assert pub.main() == 0 and asked == ["o/n"]


def test_commits_only_a_conversion_ref_reaches_are_walked():
    """공개 데이터셋에는 Hub 이 만든 변환 ref(refs/convert/…)가 붙는다. 그것도 ref 다."""
    from generation.publish_dataset import history_exposure

    api = _history(reach={"c3": ["c3"], "v1": ["v1"]},
                   trees={"c3": {"README.md": "r1"}, "v1": {"default/train/0000.parquet": "q"}})
    api.list_repo_refs = lambda repo_id, repo_type=None, include_pull_requests=False: (
        types.SimpleNamespace(branches=FakeHistoryApi._refs({"main": "c3"}), tags=[],
                              converts=FakeHistoryApi._refs({"refs/convert/parquet": "v1"}),
                              pull_requests=[]))
    report = history_exposure(api, "o/n")
    assert report["history_only"] == {"default/train/0000.parquet": 1}
    assert ("convert", "refs/convert/parquet", "v1") in report["refs"]


# ── 읽지 못한 것을 빈 것으로 읽지 않는다 ─────────────────────────────
#
# 감사의 exit 0 은 「공개해도 head 밖의 것은 안 나간다」는 진술이다. 읽지 못한 것이
# 빈 것으로 읽히면 그 진술이 거짓이 된다 — 아래는 전부 그 형태다.


def _refs_reply(**lists):
    reply = dict(branches=FakeHistoryApi._refs({"main": "c3"}), tags=[], converts=[],
                 pull_requests=[])
    reply.update(lists)
    return lambda repo_id, repo_type=None, include_pull_requests=False: (
        types.SimpleNamespace(**reply))


def test_a_ref_without_a_target_is_not_read_as_main():
    """클라이언트는 revision=None 을 main 으로 읽는다 — 그 ref 의 이력은 안 읽힌다."""
    from generation.publish_dataset import history_exposure

    api = _history(tags={"old": None})
    with pytest.raises(ValueError, match="가리키는 커밋"):
        history_exposure(api, "o/n")


def test_a_tip_missing_from_its_own_history_is_not_a_clean_history():
    from generation.publish_dataset import history_exposure

    with pytest.raises(ValueError, match="커밋 목록"):
        history_exposure(_history(reach={"c3": []}), "o/n")


@pytest.mark.parametrize("missing", ["branches", "tags", "converts", "pull_requests"])
def test_a_ref_list_that_did_not_come_is_not_an_empty_one(missing):
    from generation.publish_dataset import history_exposure

    api = _history()
    api.list_repo_refs = _refs_reply(**{missing: None})
    with pytest.raises(ValueError, match=missing):
        history_exposure(api, "o/n")


def test_a_repository_that_names_no_branch_is_not_a_clean_history():
    from generation.publish_dataset import history_exposure

    api = _history()
    api.list_repo_refs = _refs_reply(branches=[])
    with pytest.raises(ValueError, match="main"):
        history_exposure(api, "o/n")


def test_a_file_whose_content_cannot_be_named_is_not_skipped_as_a_folder():
    from generation.publish_dataset import history_exposure

    api = _history()
    real = api.list_repo_tree

    def tree(repo_id, repo_type=None, revision=None, recursive=False):
        yield from real(repo_id, repo_type=repo_type, revision=revision, recursive=recursive)
        if revision == "c1":
            yield types.SimpleNamespace(path="withdrawn.usd", blob_id=None, size=1)

    api.list_repo_tree = tree
    with pytest.raises(ValueError, match="withdrawn.usd"):
        history_exposure(api, "o/n")


def test_a_head_tree_that_disagrees_with_the_file_listing_is_not_trusted():
    """빈 트리가 「파일이 없다」인지 「못 읽었다」인지는 head 에서만 대조할 수 있다."""
    from generation.publish_dataset import history_exposure

    api = _history()
    api.list_repo_files = lambda repo_id=None, repo_type=None, revision=None: [
        "README.md", "meta/info.json", "data/file-000.parquet"]
    with pytest.raises(ValueError, match="head"):
        history_exposure(api, "o/n")


def test_the_audit_reports_what_it_could_not_read_and_exits_one(capsys):
    from generation.publish_dataset import audit_history

    assert audit_history("o/n", api=_history(tags={"old": None})) == 1
    assert "읽지 못했습니다" in capsys.readouterr().err


def test_the_load_tag_moved_by_someone_else_meanwhile_is_not_reported_as_success(tmp_path, capsys):
    """우리가 릴리스 태그를 건 직후 다른 작성자가 v3.0 을 옮긴 경우 — 성공이 아니다."""
    def meddle(api, tag):
        if tag == "v0.1.0":
            api.tags["v3.0"] = "c" * 40

    calls = []
    api = FakeApi(calls=calls, on_create_tag=meddle)
    assert _publish_release(tmp_path, api, calls) == 1
    err = capsys.readouterr().err
    assert "v3.0 → " + "c" * 40 in err and "v0.1.0 → " + OID in err


def test_a_release_tag_that_appeared_after_the_load_tag_moved_says_where_both_are(tmp_path, capsys):
    """충돌 확인 뒤에 릴리스 태그가 다른 커밋에 생긴 경우 — v3.0 은 이미 옮겨졌다.

    실패로 끝나는 것은 맞지만, 「옮기지 않았다」만 말하면 두 태그가 갈라진 것을 숨긴다.
    """
    def meddle(api, tag):
        if tag == "v3.0":
            api.tags["v0.1.0"] = "b" * 40

    calls = []
    api = FakeApi(calls=calls, on_create_tag=meddle)
    assert _publish_release(tmp_path, api, calls) == 1
    assert api.tags == {"v3.0": OID, "v0.1.0": "b" * 40}
    err = capsys.readouterr().err
    assert "v3.0 → " + OID in err and "v0.1.0 → " + "b" * 40 in err


def test_a_pin_that_could_not_be_confirmed_is_not_called_a_failed_creation(capsys):
    """만들기는 됐고 확인하는 조회가 죽었다 — 「안 걸렸다」가 아니라 「모른다」."""
    from generation.publish_dataset import _pin_release_tag

    calls = []
    api = FakeApi(refs_raises_from=2, calls=calls)
    assert _pin_release_tag(api, "o/n", "v0.1.0", OID) == 1
    assert api.tags == {"v0.1.0": OID}
    err = capsys.readouterr().err
    assert "확인하지 못했습니다" in err and "걸리지 않았습니다" not in err


def test_a_pin_confirmed_by_the_last_reading_counts_as_done(tmp_path):
    """핀의 확인 조회만 죽고 마지막 조회는 된 경우 — 두 태그가 그 커밋에 있으면 그것이 사실이다."""
    calls = []
    api = FakeApi(tags={"v3.0": OID}, calls=calls)
    real, seen = api.list_repo_refs, []

    def flaky(*a, **kw):
        seen.append(1)
        if len(seen) == 4:                  # conflict check, move, pin-before, [pin-after], final
            raise FakeHubError("500 refs unavailable")
        return real(*a, **kw)

    api.list_repo_refs = flaky
    assert _publish_release(tmp_path, api, calls) == 0
    assert api.tags == {"v3.0": OID, "v0.1.0": OID}


# ── 릴리스 태그의 말 ─────────────────


def test_a_release_tag_conflict_says_main_already_holds_the_upload(tmp_path, capsys):
    """충돌은 업로드 뒤에 보인다. 「옮기지 않았습니다」만 말하면 아무것도 안 바뀐 것처럼 읽힌다."""
    calls = []
    api = FakeApi(tags={"v3.0": "b" * 40, "v0.1.0": "b" * 40}, calls=calls)
    assert _publish_release(tmp_path, api, calls) == 1
    err = capsys.readouterr().err
    assert "main" in err and OID[:12] in err, err


def _refs_failing_on(api, *numbers, hide=None):
    """list_repo_refs 의 n 번째 호출을 실패시키거나(hide 없음) 그 태그를 뺀 채 답하게 한다."""
    real, seen = api.list_repo_refs, []

    def refs(*a, **kw):
        seen.append(1)
        if len(seen) in numbers:
            if hide is None:
                raise FakeHubError("500 refs unavailable")
            reply = real(*a, **kw)
            return types.SimpleNamespace(tags=[t for t in reply.tags if t.name != hide])
        return real(*a, **kw)

    api.list_repo_refs = refs


def test_a_pin_confirmed_by_the_last_reading_leaves_no_recovery_to_run(tmp_path, capsys):
    """마지막 읽기가 두 태그를 그 커밋에서 보면 게시는 성공이다. 그 stderr 에 돌리면 안 되는
    복구 명령이 남아 있었다."""
    calls = []
    api = FakeApi(tags={"v3.0": OID}, calls=calls)
    _refs_failing_on(api, 4)                # conflict check, move, pin-before, [pin-after], final
    assert _publish_release(tmp_path, api, calls) == 0
    out = capsys.readouterr()
    assert "복구" not in out.err and "확인하지 못했습니다" not in out.err, out.err
    assert "마지막 읽기에서 확인했습니다" in out.out


def test_a_pin_that_read_absent_and_then_was_seen_leaves_no_recovery_to_run(tmp_path, capsys):
    """핀의 확인 조회가 태그를 못 보고 마지막 읽기가 본 경우도 같다 — 마지막에 본 것이 사실이다."""
    calls = []
    api = FakeApi(tags={"v3.0": OID}, calls=calls)
    _refs_failing_on(api, 4, hide="v0.1.0")
    assert _publish_release(tmp_path, api, calls) == 0
    out = capsys.readouterr()
    assert "복구" not in out.err and "걸리지 않았습니다" not in out.err, out.err


def test_a_pin_nobody_could_confirm_still_says_so_with_the_recovery(tmp_path, capsys):
    calls = []
    api = FakeApi(tags={"v3.0": OID}, calls=calls)
    _refs_failing_on(api, 4, 5)
    assert _publish_release(tmp_path, api, calls) == 1
    err = capsys.readouterr().err
    assert "v0.1.0 가 " + OID[:12] + " 에 걸렸는지 확인하지 못했습니다" in err, err
    assert f"hf repos tag create o/n v0.1.0 --revision {OID} --repo-type dataset" in err



# ── 실패한 판정의 요약 · 핀이 태그를 읽는 자리 ─────────────────────────


_VERBOSE_FAILURE = (
    " ✗  C9 시드 재현성             ERROR 1\n"
    "------------------------------------------------------------------------------\n"
    "ERROR C9  재현 훅 시간 초과(1800 s) — 훅의 프로세스 그룹을 끝냄  [S1d_0155_seed853888546]\n"
    "WARN  C1  임계값 1건이 값 없이 조항 참조만 실림  [meta/rules.json]\n"
    + "".join(f"INFO  C9  시드 재현 확인 — 재현 확인: S1d_{i:04d} · 23 컬럼 · 999 프레임 · "
              "RGB 영상은 대조하지 않음 (실행마다 렌더가 달라 비트 재현되지 않음)\n" for i in range(7))
    + "==============================================================================\n"
    "ERROR 1  WARN 1  INFO 8  SKIP 0\n")


def test_a_failed_verdict_names_its_cause_even_behind_a_verbose_report(tmp_path):
    """--verbose 를 넘기면 INFO 줄이 ERROR 줄 뒤에 온다. 끝 500 자만 보이면 원인이 잘린다.

    2026-10-06 의 실제 거부가 그랬다 — 운영자에게 재현 성공 줄 셋과 「ERROR 1」만 보였다.
    """
    assert len(_VERBOSE_FAILURE.split("ERROR C9", 1)[1]) > 500
    lines = _validator_lines(_preflight_with(tmp_path, FakeValidator([1], reports={1: _VERBOSE_FAILURE})))
    assert len(lines) == 1, lines
    assert "ERROR C9  재현 훅 시간 초과(1800 s)" in lines[0]
    assert "ERROR 1  WARN 1  INFO 8  SKIP 0" in lines[0]
    assert "INFO  C9" not in lines[0]
    assert str(tmp_path / "logs" / "validate-attempt-1.log") in lines[0]


def test_a_failed_verdict_without_finding_lines_still_shows_its_end(tmp_path):
    """보고 형식을 못 알아보면(검증기가 다른 것을 찍었으면) 지금처럼 끝을 보인다."""
    lines = _validator_lines(_preflight_with(tmp_path, FakeValidator([1], reports={1: "usage: oops\n"})))
    assert len(lines) == 1 and "usage: oops" in lines[0]


def test_the_pin_leaves_a_tag_already_on_this_commit_quietly(capsys):
    """끊긴 게시를 다시 돌린 경우 — 핀만 따로 본다. 게시 전체로 보면 마지막 읽기가 덮는다."""
    from generation.publish_dataset import _pin_release_tag

    calls = []
    api = FakeApi(tags={"v0.1.0": OID}, calls=calls)
    assert _pin_release_tag(api, "o/n", "v0.1.0", OID) == 0
    assert "create_tag" not in calls and "delete_tag" not in calls
    out = capsys.readouterr()
    assert "그대로 둡니다" in out.out and out.err == ""


def test_the_pin_does_not_go_on_when_it_cannot_read_the_tags(capsys):
    from generation.publish_dataset import _pin_release_tag

    calls = []
    api = FakeApi(refs_raises_from=1, calls=calls)
    assert _pin_release_tag(api, "o/n", "v0.1.0", OID) == 1
    assert "create_tag" not in calls
    assert f"hf repos tag create o/n v0.1.0 --revision {OID}" in capsys.readouterr().err


def test_a_release_tag_the_last_reading_sees_is_not_asked_to_be_created_again(tmp_path, capsys):
    """핀의 확인 조회가 릴리스 태그를 못 보고, 마지막 읽기는 그것을 보지만 v3.0 이 다른 데로 옮겨진 경우.
    게시는 실패지만, 이미 있는 릴리스 태그를 다시 만들라는 복구 명령은 거짓이다."""
    calls = []
    api = FakeApi(tags={"v3.0": OID}, calls=calls)
    real, seen = api.list_repo_refs, []

    def refs(*a, **kw):
        seen.append(1)
        reply = real(*a, **kw)
        if len(seen) == 4:                  # the pin's own reading: the release tag not yet visible
            return types.SimpleNamespace(tags=[t for t in reply.tags if t.name != "v0.1.0"])
        if len(seen) == 5:                  # the final reading: v3.0 moved by another writer meanwhile
            return types.SimpleNamespace(tags=[
                types.SimpleNamespace(name=t.name, target_commit="c" * 40 if t.name == "v3.0" else t.target_commit)
                for t in reply.tags])
        return reply

    api.list_repo_refs = refs
    assert _publish_release(tmp_path, api, calls) == 1
    err = capsys.readouterr().err
    assert "복구" not in err and "걸리지 않았습니다" not in err, err
    assert "v3.0 → " + "c" * 40 in err


# ── the generator commit the card names ──

import generation.publish_dataset as pub  # noqa: E402

GEN_A = "6091b90b9bab964d92d42c9e06d09930fe79b010"
GEN_B = "0123456789abcdef0123456789abcdef01234567"
PROSE = f"\n한국어 문장 — `{GEN_A}`.\n\nEnglish sentence — `{GEN_A}`.\n"


def _marker_dataset(tmp_path, backends, card):
    import pandas as pd

    ds = tmp_path / "dist"
    (ds / "meta" / "episodes").mkdir(parents=True)
    pd.DataFrame({"outcome": ["nominal"] * len(backends), "sim_backend": backends}).to_parquet(
        ds / "meta" / "episodes" / "file-000.parquet", index=False)
    (ds / "README.md").write_text(card, encoding="utf-8")
    return ds


def _backend_of(commit):
    return {"generator_commit": commit, "generator_dirty": False}


def test_a_card_without_the_marker_is_fine_until_a_release_tag_asks_for_it(tmp_path):
    ds = _marker_dataset(tmp_path, [_backend_of(GEN_A)] * 3, "카드\n")
    assert pub.card_generator_problems(ds, required=False) == []
    problems = pub.card_generator_problems(ds, required=True)
    assert len(problems) == 1 and "generator-commit 표식이 없습니다" in problems[0]


def test_the_marker_must_be_the_commit_the_episodes_carry(tmp_path):
    ds = _marker_dataset(tmp_path, [_backend_of(GEN_A)] * 3, f"<!-- generator-commit: {GEN_A} -->\n" + PROSE)
    assert pub.card_generator_problems(ds, required=True) == []
    ds = _marker_dataset(tmp_path / "other", [_backend_of(GEN_B)] * 3, f"<!-- generator-commit: {GEN_A} -->\n" + PROSE)
    problems = pub.card_generator_problems(ds, required=True)
    assert any(GEN_A[:12] in x and GEN_B[:12] in x and "다릅니다" in x for x in problems), problems


def test_the_backend_is_read_whether_it_ships_as_a_struct_or_as_json_text(tmp_path):
    ds = _marker_dataset(tmp_path, [json.dumps(_backend_of(GEN_A))] * 2, f"<!-- generator-commit: {GEN_A} -->\n" + PROSE)
    assert pub.card_generator_problems(ds, required=True) == []


def test_two_generator_commits_in_one_release_cannot_be_one_sentence(tmp_path):
    ds = _marker_dataset(tmp_path, [_backend_of(GEN_A), _backend_of(GEN_B)], f"<!-- generator-commit: {GEN_A} -->\n" + PROSE)
    assert any("하나가 아닙니다" in x for x in pub.card_generator_problems(ds, required=False))


def test_an_episode_whose_commit_cannot_be_read_is_said(tmp_path):
    ds = _marker_dataset(tmp_path, [_backend_of(GEN_A), {"generator_commit": None, "generator_dirty": False}],
                  f"<!-- generator-commit: {GEN_A} -->\n" + PROSE)
    assert any("1건" in x and "읽지 못했습니다" in x for x in pub.card_generator_problems(ds, required=False))


@pytest.mark.parametrize("card,needle", [
    (f"<!-- generator-commit: {GEN_A} -->\n<!-- generator-commit: {GEN_A} -->\n" + PROSE, "2개"),
    ("<!-- generator-commit: 6091b90 -->\n" + PROSE, "40자 16진"),
    (f"<!-- generator-commit: {GEN_A.upper()} -->\n" + PROSE, "40자 16진"),
    (f"<!-- generator-commit: {GEN_A} extra -->\n" + PROSE, "40자 16진"),
])
def test_a_marker_that_does_not_read_whole_is_refused(tmp_path, card, needle):
    problems = pub.card_generator_problems(_marker_dataset(tmp_path, [_backend_of(GEN_A)], card), required=False)
    assert len(problems) == 1 and needle in problems[0], problems


def test_the_hash_must_also_stand_in_both_languages(tmp_path):
    card = f"<!-- generator-commit: {GEN_A} -->\n\nOnly once — `{GEN_A}`.\n"
    problems = pub.card_generator_problems(_marker_dataset(tmp_path, [_backend_of(GEN_A)], card), required=True)
    assert any("1번 나옵니다" in x for x in problems), problems


def test_a_marker_reads_across_lines():
    assert pub.parse_generator_marker(f"<!--\n generator-commit:\n {GEN_A}\n-->", "카드") == (GEN_A, [])
    assert pub.parse_generator_marker("no marker here", "카드") == (None, [])


def test_preflight_calls_the_generator_check():
    import inspect

    assert "card_generator_problems(dataset, required=release)" in inspect.getsource(pub.preflight)


def test_main_requires_the_marker_exactly_when_a_release_tag_is_pinned():
    import inspect

    assert "release=a.release_tag is not None" in inspect.getsource(pub.main)


def test_a_release_cannot_pass_the_generator_check_for_want_of_something_to_read(tmp_path):
    """With a release tag the check is required. A dataset with no card, or with no episode table,
    gave it nothing to compare -- and nothing compared is not the same as nothing wrong."""
    ds = _marker_dataset(tmp_path, [_backend_of(GEN_A)], f"<!-- generator-commit: {GEN_A} -->\n" + PROSE)
    (ds / "README.md").unlink()
    assert pub.card_generator_problems(ds, required=False) == []
    assert any("대조하지 못했습니다" in x for x in pub.card_generator_problems(ds, required=True))
    ds = _marker_dataset(tmp_path / "second", [_backend_of(GEN_A)], f"<!-- generator-commit: {GEN_A} -->\n" + PROSE)
    for q in (ds / "meta" / "episodes").glob("*.parquet"):
        q.unlink()
    assert any("대조하지 못했습니다" in x for x in pub.card_generator_problems(ds, required=True))


def test_the_hash_in_a_hidden_comment_does_not_stand_for_a_sentence(tmp_path):
    card = (f"<!-- generator-commit: {GEN_A} -->\n<!-- {GEN_A} {GEN_A} -->\n"
            f"\n한국어 문장 — `{GEN_B}`.\n\nEnglish sentence — `{GEN_B}`.\n")
    problems = pub.card_generator_problems(_marker_dataset(tmp_path, [_backend_of(GEN_A)], card), required=True)
    assert any("0번 나옵니다" in x for x in problems), problems


_GRID_RULES = {
    "contact_limits": {"BR-01": {"contact_permitted": False},
                       "BR-03": {"contact_permitted": True}},
    "rules": [{"rule_id": "R-TS15066-A2-01", "body_region": "BR-01"},
              {"rule_id": "R-TS15066-A2-03", "body_region": "BR-03"}],
}


def _batch_with_design(tmp_path, design):
    """Ten episodes: six random (2 fail, 3 near_miss, 1 nominal) and four grid (3 fail, 1 near_miss).

    `design` is the `condition_design` column, or None for a table without the column."""
    import pandas as pd

    ds = tmp_path / "dist"
    (ds / "meta" / "episodes").mkdir(parents=True)
    cols = {
        "outcome": ["fail", "fail", "near_miss", "near_miss", "near_miss", "nominal",
                    "fail", "fail", "fail", "near_miss"],
        "triggered_rules": [["R-TS15066-A2-01"], [], [], [], [], [],
                            ["R-TS15066-A2-01", "R-TS15066-A2-03"], [], [], []],
    }
    if design is not None:
        cols["condition_design"] = design
    pd.DataFrame(cols).to_parquet(ds / "meta" / "episodes" / "file-000.parquet", index=False)
    (ds / "meta" / "rules.json").write_text(json.dumps(_GRID_RULES, ensure_ascii=False),
                                            encoding="utf-8")
    return ds


def _mixed_batch(tmp_path):
    return _batch_with_design(tmp_path, ["random"] * 6 + ["grid"] * 4)


def _write_marker(ds, body):
    (ds / "README.md").write_text(f"<!-- batch-stats: {body} -->\n", encoding="utf-8")


def test_the_marker_counts_outcomes_over_the_random_rows(tmp_path):
    from generation.publish_dataset import card_matches_batch

    ds = _mixed_batch(tmp_path)
    _write_marker(ds, "episodes=10 grid_episodes=4 fail=2 near_miss=3 nominal=1 "
                      "contact_exceeded=1 contact_prohibited=1")
    assert card_matches_batch(ds) == []

    # The share of every row, grid rows included, is not what the marker says.
    _write_marker(ds, "episodes=10 grid_episodes=4 fail=5 near_miss=4 nominal=1 "
                      "contact_exceeded=2 contact_prohibited=2")
    problems = card_matches_batch(ds)
    assert any("fail=5" in x and "배치(2)" in x for x in problems), problems
    assert any("contact_exceeded=2" in x and "배치(1)" in x for x in problems), problems
    assert any("contact_prohibited=2" in x and "배치(1)" in x for x in problems), problems


def test_a_release_with_grid_rows_needs_the_grid_count(tmp_path):
    from generation.publish_dataset import card_matches_batch

    ds = _mixed_batch(tmp_path)
    _write_marker(ds, "episodes=10 fail=2 near_miss=3 nominal=1 "
                      "contact_exceeded=1 contact_prohibited=1")
    assert any("카드 표식에 grid_episodes 가 없습니다" in x for x in card_matches_batch(ds))

    _write_marker(ds, "episodes=10 grid_episodes=3 fail=2 near_miss=3 nominal=1 "
                      "contact_exceeded=1 contact_prohibited=1")
    problems = card_matches_batch(ds)
    assert any("grid_episodes=3" in x and "배치(4)" in x for x in problems), problems


@pytest.mark.parametrize("design", [None, ["random"] * 10], ids=["no-column", "all-random"])
def test_a_release_without_grid_rows_may_omit_the_grid_count(tmp_path, design):
    from generation.publish_dataset import card_matches_batch

    ds = _batch_with_design(tmp_path, design)
    counted = "fail=5 near_miss=4 nominal=1 contact_exceeded=2 contact_prohibited=2"
    _write_marker(ds, f"episodes=10 {counted}")
    assert card_matches_batch(ds) == []
    _write_marker(ds, f"episodes=10 grid_episodes=0 {counted}")
    assert card_matches_batch(ds) == []
    _write_marker(ds, f"episodes=10 grid_episodes=1 {counted}")
    problems = card_matches_batch(ds)
    assert any("grid_episodes=1" in x and "배치(0)" in x for x in problems), problems


def _write_design_table(ds, design, dtype=None):
    import pandas as pd

    path = ds / "meta" / "episodes" / "file-000.parquet"
    table = pd.read_parquet(path).drop(columns=["condition_design"], errors="ignore")
    table["condition_design"] = pd.Series(design, dtype=dtype) if dtype else pd.Series(design, dtype=object)
    table.to_parquet(path, index=False)


@pytest.mark.parametrize("dtype", [None, "string"], ids=["object", "string-dtype"])
def test_a_row_without_a_design_is_a_counted_row(tmp_path, dtype):
    from generation.publish_dataset import card_matches_batch

    # The seventh row (a fail) has no design: like the validator, it is not a grid row.
    ds = _batch_with_design(tmp_path, ["random"] * 6 + ["grid"] * 4)
    _write_design_table(ds, ["random"] * 6 + [None, "grid", "grid", "grid"], dtype)
    _write_marker(ds, "episodes=10 grid_episodes=3 fail=3 near_miss=3 nominal=1 "
                      "contact_exceeded=2 contact_prohibited=2")
    assert card_matches_batch(ds) == []
    # Counting that row as a grid row, or leaving it out of both, is what the marker must not say.
    _write_marker(ds, "episodes=10 grid_episodes=4 fail=2 near_miss=3 nominal=1 "
                      "contact_exceeded=1 contact_prohibited=1")
    assert card_matches_batch(ds) != []


@pytest.mark.parametrize("cell", [["grid"], ["grid", "random"]], ids=["one-element", "two-elements"])
def test_a_list_valued_design_is_a_counted_row(tmp_path, cell):
    from generation.publish_dataset import card_matches_batch

    # A list is not the string "grid" (parquet keeps a list column to lists only, so every row holds one): every row
    # is a counted row, none is a grid row, and no cell makes the reading raise.
    ds = _batch_with_design(tmp_path, ["random"] * 10)
    _write_design_table(ds, [["random"]] * 6 + [cell] + [["grid"]] * 3)
    _write_marker(ds, "episodes=10 grid_episodes=0 fail=5 near_miss=4 nominal=1 "
                      "contact_exceeded=2 contact_prohibited=2")
    assert card_matches_batch(ds) == []
