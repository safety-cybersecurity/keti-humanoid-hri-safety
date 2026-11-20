"""LeRobotDataset v3.0 배포본을 Hugging Face Hub 에 올린다.

실측으로 확인한 것(2026-08-24): 파일을 전부 올리고 검증기가 통과해도,
리포에 `meta/info.json` 의 `codebase_version` 과 **같은 이름의 git 태그**가 없으면
`LeRobotDataset(repo_id)` 이 로드를 거부한다(RuntimeError). 즉 이 데이터셋을 쓰려는
사람의 표준 경로가 통째로 막힌다.

태그를 손으로 기억하는 절차에 맡기면 공개일에 빠진다. 그래서 업로드·태그·왕복
무결성 확인을 한 명령으로 묶고, 셋 중 하나라도 어긋나면 exit 1 로 끝낸다.

    .venv-lerobot/bin/python generation/publish_dataset.py \\
        --dataset .tmp/lerobot/S1a --repo-id <owner>/<name>

기본은 비공개다. 공개는 `--public` 을 명시해야 하며, 대외공개 승인 없이 쓰면
안 된다.

**확인은 태그를 지키지 `main` 을 지키지 않습니다.** `upload_folder()` 는 `main` 에
직접 커밋하고, xet 백엔드에서는 한 호출이 여러 커밋으로 나뉠 수 있다. 즉 왕복
확인이 끝나기 전에도 기본 브랜치에는 새 내용이 올라가 있다. 버전을 고정해 받는
소비자(=태그를 보는 LeRobot)는 영향을 받지 않지만, 공개 리포를 브라우저로 보는
사람에게는 미확인 내용이 잠깐 보인다.

그래서 공개 순서는 **비공개로 올려 확인하고, 승인을 받은 뒤 Hub 에서 공개로
바꾸는 것**이다. 이미 공개인 리포에 바로 올리는 경로는 이 창을 피할 수 없다.

**공개로 바뀌는 것은 head 가 아니라 리포다.** 어떤 ref 에서든 닿는 커밋이 전부
함께 공개되므로, 공개될 리포에는 처음부터 공개할 것만 올린다. 뺀 것이 한 번이라도
커밋됐으면 이력을 누르지 않고 리포를 새로 만든다 — 무엇이 이력에 있는지는
`--audit-history` 가 읽는다.

업로드는 설치된 클라이언트(huggingface_hub 1.27.0 · hf_xet 1.6.0)에서 한 호출이
여러 커밋으로 나뉠 수 있고, 돌려받는 것은 **마지막** 커밋이다. 끊기면 같은 명령을
다시 돌린다 — 이미 커밋된 파일은 건너뛴다. 이 스크립트가 태그하는 것은 언제나
돌려받은 그 커밋이고, 그 커밋을 통째로 내려받아 대조한 뒤에만 태그한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
import tempfile
from pathlib import Path

# Run as a script, sys.path[0] is this file's directory, so the `generation.*`
# import in preflight() fails. compare_published.py and run_s1a.py carry the
# same guard; without it the documented publish command needs a
# PYTHONPATH the documentation never mentions.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Hub 이 리포에 스스로 넣는 파일. 우리가 올린 적이 없으므로 왕복 대조에서
# 「추가됨」으로 잡히는 것이 정상이다.
HUB_ADDED = (".gitattributes",)

# snapshot_download 가 local_dir 안에 만드는 작업 디렉터리.
DOWNLOAD_NOISE = (".cache",)

# How a validator run ended. 0, 1 and 2 are verdicts. A death is 3, the exit code
# the validator uses for "I broke" (the except clause in validate_dataset.main — a
# test compares the two), or a crash signal: on the real dataset it died of SIGSEGV
# twice in five runs (2026-10-04), and reading a death as "validation failed" would
# stop a publish at that rate, so a death is run again. A stop is a signal somebody
# sent -- a person, `timeout`, the out-of-memory killer: negative, or 128 + N when
# it came while the validator waited for the C9 hook (validate_dataset._HookSignals
# turns SIGTERM and SIGHUP into that exit). Running a stopped validator again would
# undo the stop, so it is not.
VERDICTS = (0, 1, 2)
VALIDATOR_BROKE = 3
VALIDATOR_ATTEMPTS = 3
CRASH_SIGNALS = (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE, signal.SIGILL)
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGKILL)


def validator_died(returncode: int) -> bool:
    """판정 없이 죽었는가 — 다시 돌릴 끝. 누가 세운 것은 죽음이 아니다(validator_stopped)."""
    return returncode == VALIDATOR_BROKE or -returncode in CRASH_SIGNALS


def validator_stopped(returncode: int) -> signal.Signals | None:
    """누가 세운 것이면 그 시그널. 음수 코드로, 또는 C9 훅을 기다리던 중이면 128+N 으로 온다."""
    for number in (-returncode, returncode - 128):
        if number in STOP_SIGNALS:
            return signal.Signals(number)
    return None


def default_log_dir(dataset: Path) -> Path:
    """검증기의 출력을 둘 자리. 임시 디렉터리가 아니다 — 재부팅을 넘겨야 한다."""
    import time

    return dataset.resolve().parent / "publish" / time.strftime("%Y%m%d-%H%M%S")


def run_validator(cmd: list[str], cwd: Path, log_dir: Path,
                  run=None) -> tuple[int, str, list[Path], Path]:
    """판정이 나올 때까지, 많아야 VALIDATOR_ATTEMPTS 번 검증기를 돌린다.

    돌려주는 것: (마지막 시도의 종료 코드, 그 시도의 출력, 죽은 시도들의 로그,
    마지막 시도의 로그).
    시도마다 로그(`validate-attempt-N.log`)와 JSON 보고서(`…-N.json`, 검증기의
    `--json` 으로)를 같은 번호로 받고 덮어쓰지 않는다 — 같은 `log_dir` 로 다시 돌려도
    앞 실행의 것이 남는다. 보고서를 한 파일로 받던 동안에는, 죽은 시도는 보고서를
    쓰지 않으므로 세 번 다 죽은 실행 옆에 앞 실행의 보고서가 새것처럼 남았다.

    안 보는 것: 죽지 않고 **멈춘** 검증기(제한 시간이 없다)와 죽은 **원인**. 세 번의
    시도는 원인을 가리지 않는다 — 스택을 읽는 것은 사람의 일로 남는다.
    """
    import subprocess

    run = run if run is not None else subprocess.run
    log_dir.mkdir(parents=True, exist_ok=True)
    numbers = [p.stem.rsplit("-", 1)[-1] for p in log_dir.glob("validate-attempt-*.*")]
    taken = max((int(n) for n in numbers if n.isdigit()), default=0)
    died: list[Path] = []
    code, text, log = VALIDATOR_BROKE, "", log_dir
    for attempt in range(1, VALIDATOR_ATTEMPTS + 1):
        log = log_dir / f"validate-attempt-{taken + attempt}.log"
        report = ["--json", str(log.with_suffix(".json"))]
        with log.open("w", encoding="utf-8") as out:
            code = run(cmd + report, stdout=out, stderr=subprocess.STDOUT, text=True,
                       cwd=cwd).returncode
        text = log.read_text(encoding="utf-8", errors="replace")
        if not validator_died(code):
            break
        died.append(log)
        print(f"검증기가 판정 없이 끝났습니다 (exit {code}, {attempt}/{VALIDATOR_ATTEMPTS}) "
              f"— {log}", file=sys.stderr)
    return code, text, died, log


def codebase_version(root: Path) -> str:
    """meta/info.json 의 codebase_version. 태그 이름이 여기서 나온다."""
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json 이 없습니다: {info_path}")
    version = json.loads(info_path.read_text(encoding="utf-8")).get("codebase_version")
    if not version:
        raise ValueError(
            f"{info_path} 에 codebase_version 이 없습니다 — "
            "이 값이 없으면 LeRobot 이 Hub 에서 데이터셋을 열지 못합니다"
        )
    return str(version)


def file_manifest(root: Path, skip: tuple[str, ...] = ()) -> dict[str, str]:
    """상대경로 → SHA-256. 왕복 대조의 기준."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in skip for part in rel.parts):
            continue
        out[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def linked_directories(root: Path) -> list[str]:
    """배포본 안에서 디렉터리를 가리키는 심링크 — 뿌리 기준의 경로.

    올릴 파일을 고르는 Hub 클라이언트의 탐색(`glob("**/*")`)과 왕복 대조의 `file_manifest` 는 그런 링크 아래로
    내려가지 않는다(huggingface_hub 1.27.0 · Python 3.12 에서 잼). 「이 파일이 있는가 · 저장소 것과 같은가」를 묻는
    대조는 링크를 따라가 읽으므로, `docs/` 가 링크인 배포본은 그 대조를 통과하고 그 아래의 문서 없이 올라가며 왕복
    대조도 같은 목록끼리 견주어 말하지 않는다. 파일을 가리키는 링크는 두 탐색이 모두 파일로 세므로 여기서 세지 않는다.
    """
    root = Path(root)
    # Asked of every entry, in whichever list it came: os.walk takes an entry for a file when asking whether it is
    # a directory failed.
    return sorted(p.relative_to(root).as_posix() for here, dirs, names in os.walk(root)
                  for p in (Path(here, name) for name in dirs + names) if p.is_symlink() and p.is_dir())


def unlisted_directories(root: Path) -> list[str]:
    """배포본 안에서 목록을 읽지 못하는 디렉터리 — 뿌리 기준의 경로(뿌리 자신은 `.`).

    같은 틈의 다른 꼴이다. 들어갈 권한만 있고 읽을 권한이 없는 디렉터리는 경로를 알고 여는 대조에는 파일을 내주는데,
    올릴 파일을 고르는 탐색과 `file_manifest` 는 그 디렉터리를 오류 없이 지나친다.
    """
    root, failed = Path(root), []
    for _ in os.walk(root, onerror=lambda error: failed.append(Path(error.filename).relative_to(root).as_posix())):
        pass
    return sorted(failed)


def manifest_problems(local: dict[str, str], remote: dict[str, str]) -> list[str]:
    """올린 것과 내려받은 것의 차이. 빈 리스트면 비트 단위로 같다."""
    problems = []
    for rel in sorted(set(local) - set(remote)):
        problems.append(f"누락: {rel}")
    for rel in sorted(set(remote) - set(local)):
        if rel in HUB_ADDED:
            continue
        problems.append(f"올린 적 없는 파일: {rel}")
    for rel in sorted(set(local) & set(remote)):
        if local[rel] != remote[rel]:
            problems.append(f"내용 다름: {rel}")
    return problems


def remote_extras(remote_paths, local: dict[str, str]) -> list[str]:
    """리포에 있고 이 배포본에는 없는 경로. Hub 이 스스로 넣는 것은 뺀다.

    업로드는 더하기만 하므로 이것들은 올린 뒤에도 남아 함께 나간다. 왕복 대조가
    같은 것을 잡지만 전부 올리고 전부 내려받은 뒤다 — 여기서는 빨리 실패할 뿐이고,
    이 검사와 업로드 사이에 생긴 파일은 여전히 왕복 대조의 몫이다.
    """
    return sorted(p for p in remote_paths if p not in local and p not in HUB_ADDED)


def verdict_findings(validator_stdout: str, limit: int = 20) -> str:
    """검증기 보고에서 ERROR · WARN 줄과 요약 줄만 고른다.

    점검은 --verbose 를 넘기므로 INFO 줄이 ERROR 줄 **뒤에** 온다. 끝의 몇백 자만
    보이면 원인이 잘리고 성공 줄만 남는다 — 2026-10-06 의 실제 거부에서 운영자에게
    재현 성공 셋과 「ERROR 1」만 보였다. 보고 형식을 못 알아보면 끝을 그대로 보인다.
    """
    picked = [line for line in validator_stdout.splitlines()
              if line.startswith(("ERROR", "WARN"))]
    if not picked:
        return "…" + validator_stdout.strip()[-500:]
    if len(picked) > limit:
        picked = picked[:limit - 1] + [f"… (줄 {len(picked) - limit + 1}개 더)", picked[-1]]
    return "\n".join(picked)


def c9_skipped(validator_stdout: str) -> bool:
    """검증기 보고에서 C9 가 SKIP 인가.

    작은 함수로 떼어 낸 것은 시험 때문이다 — 배포본을 통째로 만들어 확인하려 하면
    다른 검사가 먼저 실패하고 그 실패 메시지 안에 SKIP 이라는 글자가 딸려 들어와
    **엉뚱한 이유로 초록**이 된다. 실제로 그렇게 썼다가 잡았다 (2026-08-26).
    """
    for line in validator_stdout.splitlines():
        if "C9" in line and "재현성" in line:
            return line.lstrip().startswith("-")
    return False


def preflight(dataset: Path, repo_root: Path, repro_hook: str | None = None,
              repro_sample: int = 2, log_dir: Path | None = None, run=None,
              notes: list[str] | None = None, release: bool = False) -> list[str]:
    """올리기 직전에 「이 디렉터리가 지금 코드의 산출물인가」를 묻는다.

    변환기는 정본과 데이터셋 카드를 배포본 안에 **복사**한다. 그러니 정본을 고친
    뒤 재변환하지 않으면 배포본이 옛 카드·옛 규격을 실은 채 남아 있고, 디렉터리
    이름만 보고 올리면 그게 나간다. 실제로 카드가 10건 배치를 「현재 배치」라고
    설명하는 판본이 업로드됐다.

    검증기를 부르는 것만으로는 부족하다 — 배포본 안의 옛 규격은 그 자신과
    일관되므로 통과한다. 그래서 **소스와 바이트로 대조**한다.

    검증기가 **죽은 것**(시그널 · 자기 예외)은 판정이 아니다. 다시 돌리고, 끝내
    판정을 못 얻으면 「검증 실패」가 아니라 그 사실을 말한다. 죽은 뒤 통과한 것은
    문제가 아니지만 `notes` 에 남겨 호출부가 찍게 한다.
    """
    from generation import release_docs
    from generation.convert_lerobot import CANON_DIRS, CANON_FILES, CANON_RENAMES

    problems: list[str] = []

    # Absolute, both: the validator runs in the repository root, so a relative
    # path meant one place to this process and another to it -- the attempt logs
    # landed beside the caller and the report under the repository.
    log_dir = Path(log_dir).resolve() if log_dir is not None else default_log_dir(dataset)
    py = repo_root / ".venv" / "bin" / "python"
    # run_validator adds --json per attempt, which keeps the report of the very
    # run that gated this publish; --verbose puts the C9 sample lines into the log.
    cmd = [str(py) if py.exists() else sys.executable,
           str(repo_root / "validator" / "validate_dataset.py"), str(Path(dataset).resolve()),
           "--verbose"]
    # 훅을 주지 않으면 C9 는 SKIP 이고 SKIP 은 exit 0 이다. 즉 재현성을 **한 번도
    # 확인하지 않은** 데이터가 같은 명령으로 게시된다.
    # 표본 0 은 훅을 한 번도 부르지 않지만 SKIP 도 아니라 C9 가 OK 로 찍힌다 —
    # 새 게이트를 숫자 하나로 우회한다.
    if repro_hook and repro_sample < 1:
        problems.append(
            f"--repro-sample 이 {repro_sample} 입니다 — 훅을 한 번도 부르지 않으면서 "
            "C9 가 OK 로 찍힙니다. 1 이상이어야 합니다")
    elif repro_hook:
        cmd += ["--repro-hook", str(repro_hook), "--repro-sample", str(repro_sample)]
    code, text, died, last_log = run_validator(cmd, repo_root, log_dir, run=run)
    logs = ", ".join(str(p) for p in died)
    earlier = f" (그 전에 {len(died)}번 죽음: {logs})" if died else ""
    stopped = validator_stopped(code)
    if validator_died(code):
        problems.append(
            f"검증기가 {len(died)}번 모두 판정 없이 죽었습니다 (마지막 exit {code}) — "
            "판정이 아닙니다. 배포본이 틀렸다는 뜻도 맞다는 뜻도 아닙니다. 스택을 읽고 "
            f"다시 돌리십시오: {logs}")
    elif stopped is not None:
        oom = (" SIGKILL 은 메모리가 모자라 커널이 끝낸 것일 수도 있습니다 — "
               "`journalctl -k` 에서 그 시각의 oom-kill 을 보십시오."
               if stopped == signal.SIGKILL else "")
        problems.append(
            f"검증기가 {stopped.name} 로 멈췄습니다 (exit {code}) — 누가 세운 것으로 보고 다시 "
            f"돌리지 않았습니다. 판정이 아닙니다.{oom} 로그: {last_log}{earlier}")
    elif code not in VERDICTS:
        problems.append(
            f"검증기가 판정이 아닌 코드로 끝났습니다 (exit {code}) — 판정은 0 · 1 · 2 뿐이라 "
            f"판정이 아닙니다. 로그: {last_log}{earlier}\n" + verdict_findings(text))
    elif code != 0:
        problems.append(f"검증기가 통과하지 못했습니다 (exit {code}) — 전체 보고는 {last_log}:\n"
                        + verdict_findings(text))
    elif c9_skipped(text):
        problems.append(
            "C9 재현성이 SKIP 입니다 — SKIP 은 exit 0 이라 검증기만으로는 막지 못합니다. "
            "`--repro-hook generation/rerun_seed.sh` 를 주고 다시 실행하십시오")
    if died and code in VERDICTS and notes is not None:
        notes.append(
            f"검증기가 {len(died)}번 죽은 뒤 판정을 냈습니다 — 스택을 읽으십시오 "
            f"(알려진 간헐 크래시와 같은 스택이면 그것이고, 다르면 결함입니다): {logs}")

    # 좁혀서 쓰는 둘(scenarios/rules)은 배포본에서 내용이 달라지는 것이 정상이라
    # 대조 대상이 아니다. 그대로 복사되는 것만 본다.
    pairs = [(r, r) for r in CANON_FILES] + list(CANON_RENAMES.items())
    for src_rel, dst_rel in pairs:
        src, dst = repo_root / src_rel, dataset / dst_rel
        if not dst.exists():
            problems.append(f"동봉되어야 할 파일이 없습니다: {dst_rel}")
        elif src.read_bytes() != dst.read_bytes():
            problems.append(
                f"{dst_rel} 이 저장소의 {src_rel} 과 다릅니다 — 정본을 고친 뒤 "
                "재변환하지 않은 배포본입니다")
    # 카드 옆에 조립이 싣는 문서(카드의 한국어 화면 · 상세 문서 · 그림)도 같은 대조를 받는다 — 카드의 두 화면이
    # 그 경로를 가리키므로, 없거나 옛것이면 올라간 카드의 링크가 죽거나 옛 문서로 간다.
    problems += release_docs.problems(repo_root, dataset)
    # 위의 대조들은 링크를 따라가 읽는다. 올리기는 디렉터리 링크 아래로 내려가지 않으므로, 그런 링크가 있으면
    # 「있다 · 같다」고 본 파일이 올라가지 않는다.
    for linked in linked_directories(Path(dataset)):
        problems.append(
            f"배포본 안의 {linked} 이 디렉터리를 가리키는 심링크입니다 — 올리기와 왕복 대조는 그 아래로 내려가지 "
            "않으므로 그 아래의 파일이 빠진 채 올라가고, 빠진 것을 아무도 말하지 않습니다")
    for unlisted in unlisted_directories(Path(dataset)):
        problems.append(
            f"배포본 안의 {unlisted} 디렉터리의 목록을 읽지 못했습니다 — 올리기와 왕복 대조는 그 아래의 파일을 "
            "보지 못하므로 그 파일이 빠진 채 올라가고, 빠진 것을 아무도 말하지 않습니다")
    for rel in CANON_DIRS:
        for src in sorted((repo_root / rel).rglob("*")):
            if not src.is_file():
                continue
            dst = dataset / src.relative_to(repo_root)
            if not dst.exists() or src.read_bytes() != dst.read_bytes():
                problems.append(f"{src.relative_to(repo_root)} 이 저장소 정본과 다릅니다")

    # 좁혀 쓰는 둘은 내용이 달라지는 것이 정상이라 바이트로 못 댄다. 그래서 **다시
    # 좁혀서** 댄다 — 그러지 않으면 배포본의 규칙 정본이 저장소 정본이나 실제 궤적
    # 입력과 어긋난 채로 나가고, 그것은 자기 자신과 일관되므로 검증기도 통과한다
    # (임계값을 9.99 로 바꿔도 통과했다).
    problems.extend(narrowed_canon_problems(dataset, repo_root))
    problems.extend(card_matches_batch(dataset))
    problems.extend(card_generator_problems(dataset, required=release))
    problems.extend(shipped_assets_match_source(dataset, repo_root))
    problems.extend(shipped_camera_canon_problems(dataset, repo_root))
    problems.extend(shipped_packs_match_index(dataset))
    canon_assets = json.loads((repo_root / "meta/assets.json").read_text(encoding="utf-8"))
    verified = {
        Path(e["usd_path"]).name: str((e.get("usd_checked") or {}).get("sha256") or "")
        for group in ("robots", "scene")
        for e in (canon_assets.get(group) or {}).values()
        if isinstance(e, dict) and e.get("usd_path")
    }
    problems.extend(redistributed_usd_has_no_standard_values(dataset, verified))
    return problems


# 재배포 USD 에 있어서는 안 되는 속성. 유료 표준의 수치이고 배포본은 CC BY 4.0 이라,
# 실리는 순간 재배포 권한 없는 값을 상업적 이용 허용 라이선스로 내보내게 된다
# (빌더가 실제로 34개를 쓰고 있었다). `preflight` 가 배포본의 `.usd` 를 열어
# 이 속성이 실렸는지 본다(`redistributed_usd_has_no_standard_values`).
FORBIDDEN_USD_ATTRS = ("keti:effectiveMass", "keti:springConstant", "physics:mass")


def layer_forbidden_attrs(path: Path) -> set[str]:
    """The FORBIDDEN_USD_ATTRS that ``path``'s own layer writes a value for -- a default or time samples --
    on any prim spec, as the layer holds them: not composed, so what another file it references or sublayers
    writes is not this file's. Raises ImportError without pxr and ValueError when the layer cannot be opened."""
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        raise ValueError(f"{path}: USD 레이어를 열 수 없습니다")
    found: set[str] = set()

    def visit(spec_path):
        if not spec_path.IsPropertyPath() or spec_path.name not in FORBIDDEN_USD_ATTRS:
            return
        attr = layer.GetAttributeAtPath(spec_path)
        if attr is not None and (attr.HasDefaultValue() or layer.GetNumTimeSamplesForPath(spec_path)):
            found.add(spec_path.name)

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return found


def redistributed_usd_has_no_standard_values(dataset: Path,
                                            verified: dict[str, str] | None = None) -> list[str]:
    """배포본의 USD 를 열어 금지 속성을 찾는다.

    보는 것은 `assets/` 바로 아래의 USD — 우리가 만들어 싣는 자산 — 이고, 각 파일을 **자기
    레이어로만** 연다(`layer_forbidden_attrs` — 합성하지 않음). 합성해서 열면 다른 파일을
    참조만 하는 USD(로봇 폴더를 참조하는 Stage)가 그 파일의 질량을 제 것처럼 보이고, 로봇의
    질량은 그 로봇의 실제 값이다. 폴더째 싣는 로봇(`assets/<dir>/`)은 여기서 열지 않고
    행마다 고정한 해시로 대조된다(`shipped_assets_match_source` · 검증기 C6).

    pxr 이 없는 인터프리터(배포 파이프라인의 `.venv-lerobot`)에서도 돌아야 하므로,
    **정본이 「이 해시는 확인했다」고 기록한 것과 같으면** 통과시킨다. 해시가 그
    내용을 고정하므로 참인 진술이다. 해시가 다르거나 기록이 없으면 확인하지 못한
    것이고, 확인하지 못한 것은 통과시키지 않는다.
    """
    import hashlib

    verified = verified or {}
    out = []
    for usd in sorted((dataset / "assets").glob("*.usd")):
        try:
            found = layer_forbidden_attrs(usd)
        except ValueError:
            out.append(f"assets/{usd.name} 을 USD 로 열 수 없습니다 — 금지 속성을 확인하지 못했습니다")
            continue
        except ImportError:
            # usdc 는 토큰 표를 압축하므로 바이트에서 속성 이름이 안 보인다 —
            # 바이트로 훑는 대체 경로를 넣었다가 **아무것도 못 찾으면서 통과**하는
            # 것을 확인하고 지웠다.
            got = hashlib.sha256(usd.read_bytes()).hexdigest()
            if verified.get(usd.name) == got:
                continue
            out.append(
                f"assets/{usd.name} 을 열 수 없고 정본의 확인 기록과도 다릅니다 — "
                "금지 속성을 확인하지 못했습니다. pxr 이 있는 인터프리터로 한 번 "
                "확인하고 meta/assets.json 의 usd_checked 를 갱신하십시오")
            continue
        if found:
            out.append(
                f"assets/{usd.name} 에 배포하면 안 되는 속성이 있습니다: "
                f"{sorted(found)} — 유료 표준의 수치이고 배포본은 CC BY 4.0 입니다")
    return out


def batch_asset_paths(dataset: Path, canon_assets: dict,
                      canon_scenarios: dict) -> tuple[set[str] | None, str | None]:
    """The asset paths this batch uses -- the set its assets/ is drawn from
    (generation.assets.shipped_asset_files): what the distribution's episodes carry in
    scene_assets, **and** what the repository canon says each episode's scenario used and
    its robot_model is (generation.assets.required_scene_assets) plus the USD of the
    episode's own human_model.

    The second half is the anchor, the robot included: drawn
    from scene_assets alone, an episode that leaves an asset out takes it out of the
    comparison too. (None, why) when meta/episodes or the canon cannot say: then nothing can
    be compared, and that is said, not passed."""
    import pandas as pd

    from generation.assets import required_scene_assets

    eps = sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    if not eps:
        return None, ("에피소드 메타가 없어 이 배치가 쓰는 자산을 알 수 없습니다 — "
                      "assets/ 를 대조하지 못했습니다")
    by_sid = {s.get("scenario_id"): s for s in (canon_scenarios.get("scenarios") or [])
              if isinstance(s, dict)}
    scene_canon = canon_assets.get("scene") or {}
    paths: set[str] = set()
    for q in eps:
        table = pd.read_parquet(q)
        absent = [c for c in ("scene_assets", "scenario_id") if c not in table.columns]
        if absent:
            return None, (f"{q.relative_to(dataset)} 에 {absent} 열이 없어 이 배치가 쓰는 자산을 "
                          "알 수 없습니다 — assets/ 를 대조하지 못했습니다")
        if "robot_model" not in table.columns:
            return None, (f"{q.relative_to(dataset)} 에 ['robot_model'] 열이 없어 이 배치의 로봇 자산을 "
                          "알 수 없습니다 — assets/ 를 대조하지 못했습니다")
        for value in table["scene_assets"]:
            for item in as_list(value):
                if isinstance(item, dict) and isinstance(item.get("path"), str):
                    paths.add(item["path"])
        # A robot_model that is not a string is no key of the canon: None says so without
        # letting an unhashable value into the set.
        uses = {(str(s), r if isinstance(r, str) else None)
                for s, r in zip(table["scenario_id"], table["robot_model"])}
        for sid, robot in sorted(uses, key=lambda u: (u[0], u[1] or "")):
            if sid not in by_sid:
                return None, (f"에피소드의 시나리오 {sid} 가 저장소 정본 meta/scenarios.json 에 없어 "
                              "이 배치가 쓰는 자산을 알 수 없습니다 — assets/ 를 대조하지 못했습니다")
            try:
                paths |= required_scene_assets(by_sid[sid], canon_assets, robot)
            except ValueError as exc:
                return None, f"이 배치가 쓰는 자산을 정본에서 알 수 없습니다: {exc}"
        models = table["human_model"].tolist() if "human_model" in table.columns else []
        for model in {m for m in models if isinstance(m, str)}:
            entry = scene_canon.get(model)
            if not isinstance(entry, dict) or not entry.get("usd_path"):
                return None, (f"에피소드의 human_model {model!r} 이 저장소 정본 meta/assets.json 의 "
                              "scene 에 없어 인체 모델의 자산을 알 수 없습니다 — assets/ 를 대조하지 못했습니다")
            paths.add(entry["usd_path"])
    return paths, None


def shipped_assets_match_source(dataset: Path, repo_root: Path) -> list[str]:
    """`assets/` 가 이 배치가 실어야 할 파일과 하나하나 같은가 — 바이트까지.

    존재만 보면 아무 비어 있지 않은 파일이나 통과한다 — 엉뚱한 로봇 자산이 그대로
    데이터셋에 실린다. 정본이 저장소 안 경로를 가리키는
    것만 댈 수 있다. Nucleus·URL 경로는 애초에 재배포 대상이 아니다.

    **실어야 할 것은 이 배포본의 에피소드가 정한다.** 정본의 재배포 자산 전부가
    아니라 에피소드의 `scene_assets` 가 가리키는 것 — 그리고 정본이 그 에피소드의 시나리오가
    썼다고 말하는 것(batch_asset_paths: `scene.assets` · 인체 모델의 USD · `robot_model` 의
    USD. 에피소드 목록에서 빠져도 요구가 빠지지 않는다)이고, 텍스처 묶음이 있는 USD 는 그
    텍스처를 `assets/textures/` 에 함께 싣는다 — 변환기와 같은 함수
    (generation.assets.shipped_asset_files)로 다시 정해 대조한다. 텍스처의 기준은 그
    자산의 textures.lock 이 고정한 SHA-256 이다(그 함수가 정본의 행을 lock 과 대조한다).
    폴더째 싣는 로봇은 `assets/<dir>/` 의 행마다 같은 대조를 받는다(같은 함수가 그 행을 낸다).
    그 밖의 파일이 `assets/` 에 있으면 그것도 문제다 — 이 배치가 쓰지 않는 파일이 이 배치의
    라이선스로 나간다.
    """
    from generation.assets import robot_ship_files, shipped_asset_files

    canon = json.loads((repo_root / "meta/assets.json").read_text(encoding="utf-8"))
    out = []
    # 정본 자체의 진술 — 배치와 무관하게, 재배포한다는 항목은 전부 본다.
    for group in ("robots", "scene"):
        for name, entry in (canon.get(group) or {}).items():
            if not isinstance(entry, dict) or not entry.get("redistributed"):
                continue
            # A robot that ships as its folder has the Nucleus token as its usd_path (the episodes join on it):
            # the rows of its ship block say what ships, and the shipping rule judges whether they say it.
            if "ship" in entry:
                try:
                    robot_ship_files(name, entry, repo_root)
                except ValueError as exc:
                    out.append(str(exc))
                continue
            usd = str(entry.get("usd_path") or "")
            if not usd or "://" in usd or usd.startswith("{"):
                out.append(f"{name}: 재배포한다는데 저장소 경로가 아닙니다: {usd!r}")
                continue
            # 저장소 밖을 가리키면 「원본과 같다」가 아무것도 뜻하지 않는다 —
            # `../outside.usd` 로 기준 자체를 옮길 수 있었다.
            if not (repo_root / usd).resolve().is_relative_to(repo_root.resolve()):
                out.append(f"{name}: usd_path 가 저장소 밖입니다: {usd}")
                continue
            # 그리고 **양쪽 모두 정본이 고정한 해시와** 대조한다. 원본이 git 에
            # 추적되지 않으므로, 「원본과 배포본이 서로 같다」는 둘을 함께 바꾸면
            # 성립한다 — 어느 자산인지는 말해 주지 않는다.
            if not entry.get("usd_sha256"):
                out.append(
                    f"{name}: 재배포하는데 정본에 usd_sha256 이 없습니다 — "
                    "어느 자산을 실었는지 고정되지 않습니다")
    if out:
        return out                                  # a canon that says this cannot say what ships

    scenarios = json.loads((repo_root / "meta/scenarios.json").read_text(encoding="utf-8"))
    referenced, why = batch_asset_paths(dataset, canon, scenarios)
    if referenced is None:
        return [why]
    try:
        plan = shipped_asset_files(canon, referenced, repo_root)
    except ValueError as exc:
        return [f"자산 정본으로 이 배치가 실을 파일을 정하지 못했습니다: {exc}"]
    root = repo_root.resolve()
    for rel, f in sorted(plan.items()):
        src_rel = f.src.resolve().relative_to(root)
        dst = dataset / rel
        if not f.src.is_file():
            out.append(f"{f.asset}: 정본이 가리키는 원본이 없습니다: {src_rel}")
            continue
        if not dst.is_file():
            out.append(f"{f.asset}: 배포본에 자산이 없습니다: {rel}")
            continue
        src_bytes, dst_bytes = f.src.read_bytes(), dst.read_bytes()
        if src_bytes != dst_bytes:
            out.append(
                f"{f.asset}: {rel} 이 저장소의 {src_rel} 와 다릅니다 — "
                "다른 파일이 그 이름으로 실려 있습니다")
            continue
        got = hashlib.sha256(dst_bytes).hexdigest()
        if got != f.sha256:
            out.append(
                f"{f.asset}: 실린 {rel} 의 해시가 정본과 다릅니다 — "
                f"정본 {f.sha256[:16]}…, 실린 것 {got[:16]}…")
    assets_dir = dataset / "assets"
    extra = sorted(str(q.relative_to(dataset)) for q in assets_dir.rglob("*")
                   if not q.is_dir() and str(q.relative_to(dataset)) not in plan) \
        if assets_dir.is_dir() else []
    if extra:
        out.append(
            f"배포본 assets/ 에 이 배치가 싣지 않는 파일 {len(extra)}개: {extra[:3]} — 에피소드가 쓰는 "
            "재배포 자산과 그 텍스처, 폴더째 싣는 로봇의 ship.files 행, 엮는 자산이 모두 실릴 때의 엮은 자산만 "
            "싣습니다. 정본이 바뀐 뒤 재변환하지 않은 배포본이거나 손대어진 것입니다")
    return out


def shipped_camera_canon_problems(dataset: Path, repo_root: Path) -> list[str]:
    """카메라 배포본의 `meta/cameras.json` 이 지금 카메라 정본에서 유도한 것과 같은가.

    변환기는 정본에 내부 파라미터(fx · fy · cx · cy)를 채워 **복사**한다
    (convert_lerobot.shipped_cameras). 정본을 고친 뒤 재변환하지 않으면 옛 사본이 남는데,
    그 사본은 자기 영상·팩과 일관되므로 검증기가 통과시킨다 — 좁혀 쓰는 두 정본과 같은
    처지라 같은 방법으로, **같은 함수로 다시 유도해** 댄다. 카메라 없는
    배포본에는 이 파일이 없고 요구하지도 않는다.
    """
    shipped = dataset / "meta" / "cameras.json"
    if not shipped.exists():
        return []
    from generation import cameras as camera_canon
    from generation.convert_lerobot import shipped_cameras

    try:
        got = json.loads(shipped.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        return [f"meta/cameras.json 을 읽지 못했습니다: {exc}"]
    try:
        want = shipped_cameras(camera_canon.load(repo_root / "meta" / "cameras.json"))
    except (OSError, ValueError, KeyError) as exc:
        return [f"저장소의 카메라 정본에서 meta/cameras.json 을 유도하지 못했습니다: {exc}"]
    if got == want:
        return []
    keys = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k)) \
        if isinstance(got, dict) else ["(최상위가 객체가 아님)"]
    return [f"meta/cameras.json 이 지금 카메라 정본에서 유도한 것과 다릅니다(다른 항목 {keys}) — "
            "정본을 고친 뒤 재변환하지 않았거나, 배포본이 손대어졌습니다"]


def shipped_packs_match_index(dataset: Path) -> list[str]:
    """`meta/packs.json` 이 적은 팩마다 — 배포본 안의 그 파일이 있고, SHA-256 이 적힌 값인가.

    받는 쪽은 이 목록으로 내려받은 팩을 확인한다. 검증기 C1 도 매번 같은 것을 보지만
    (_c1_pack_files), 올리기 직전 이 점검이 스스로 한 번 더 본다 — 목록이 틀린 채 올라가면
    그 확인이 바른 파일을 거부하거나 바뀐 팩을 통과시킨다.

    팩은 배포본의 `meta/cameras.json` 이 `packs` 로 선언할 때만 싣는다(팩 코드는
    두고 정본에서 끈다). 선언했는데 목록이 없으면 문제다. 선언하지 않았는데(카메라 없는
    배포본 포함) `packs/` 나 `meta/packs.json` 이 있어도 문제다 — 옛 변환 사본 위에서 작업하면
    선언 없는 팩이 실려 나가고, 받는 쪽은 그것이 어느 카메라·축소의 팩인지 알 수 없다.
    정본을 읽지 못하면 팩이 있어야 하는지 모르므로 통과시키지 않는다. 선언을 읽는 규칙은
    변환기의 것(convert_lerobot.the_pack)을 그대로 쓴다 — 사본을 두면 `[]` 같은 값을 한쪽만
    「팩 없음」으로 읽는다. 그 규칙이 거부하는 packs(객체가 아님 · 둘 이상)도 선언을 알 수 없는 것이다.
    """
    from generation.assets import sha256_file
    from generation.convert_lerobot import the_pack

    index_path = dataset / "meta" / "packs.json"
    cameras_path = dataset / "meta" / "cameras.json"
    declared = False
    if cameras_path.exists():
        try:
            canon = json.loads(cameras_path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            return [f"meta/cameras.json 을 읽지 못해 팩을 선언했는지 알 수 없습니다 — packs/ · "
                    f"meta/packs.json 을 대조하지 못했습니다: {exc}"]
        if not isinstance(canon, dict):
            return [f"meta/cameras.json 최상위가 객체가 아니어서({type(canon).__name__}) 팩을 선언했는지 "
                    "알 수 없습니다 — packs/ · meta/packs.json 을 대조하지 못했습니다"]
        try:
            declared = the_pack(canon) is not None
        except ValueError as exc:
            return [f"meta/cameras.json 의 packs 로는 팩을 선언했는지 알 수 없습니다 — packs/ · "
                    f"meta/packs.json 을 대조하지 못했습니다: {exc}"]
    if not declared:
        stray = [rel for rel in ("packs/", "meta/packs.json") if (dataset / rel).exists()]
        if stray:
            return [f"배포본의 정본이 팩을 선언하지 않았는데 {stray} 가 있습니다 — 팩은 meta/cameras.json 이 "
                    "packs 로 선언할 때만 싣습니다. 옛 변환 사본 위에서 작업했거나 손대어진 배포본이고, "
                    "받는 쪽은 그것이 어느 카메라·축소의 팩인지 알 수 없습니다"]
        return []
    if not index_path.exists():
        return ["카메라 정본(meta/cameras.json)이 팩을 선언했는데 meta/packs.json 이 없습니다 — 팩을 "
                "에피소드에 잇지도, 받은 것을 확인하지도 못합니다"]
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        return [f"meta/packs.json 을 읽지 못했습니다: {exc}"]
    if not isinstance(index, dict):
        return [f"meta/packs.json 최상위가 객체가 아닙니다: {type(index).__name__}"]
    out = []
    for key, entry in sorted(index.items()):
        rel = entry.get("file") if isinstance(entry, dict) else None
        if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
            out.append(f"meta/packs.json[{key}]: file 이 배포본 안의 상대 경로가 아닙니다: {rel!r}")
            continue
        path = dataset / rel
        if not path.is_file():
            out.append(f"meta/packs.json[{key}]: 목록이 가리키는 팩이 배포본에 없습니다: {rel}")
            continue
        got = sha256_file(path)
        if got != entry.get("sha256"):
            out.append(f"meta/packs.json[{key}]: {rel} 의 SHA-256 이 목록과 다릅니다 — 목록 "
                       f"{str(entry.get('sha256'))[:16]}…, 파일 {got[:16]}…")
    return out


def parse_batch_marker(text: str, where: str) -> tuple[dict[str, int], list[str]]:
    """`<!-- batch-stats: k=v ... -->` 한 줄을 읽어 (값, 문제 목록) 을 돌려준다.

    표식을 가진 문서들이 같은 배치를 말하는지 대조하려면 **모두 같은 코드로** 읽어야
    한다. 파서가 둘이면 표식이 글자로 같아도 다르게 읽히는 길이 생기고, 그때 어긋난
    쪽을 가리키는 것은 아무것도 없다.

    산문을 훑지 않는 이유는 card_matches_batch 의 주석에 있다.
    """
    markers = re.findall(r"<!--\s*batch-stats:(.*?)-->", text, re.S)
    if len(markers) > 1:
        return {}, [f"{where}에 batch-stats 표식이 {len(markers)}개 있습니다 — 첫 것만 "
                    "읽히므로 뒤엣것이 틀려도 통과합니다. 하나만 두십시오"]
    if not markers:
        return {}, [f"{where}에 batch-stats 표식이 없습니다 — {where}가 어느 배치를 "
                    "설명하는지 기계가 확인할 수 없습니다 "
                    "(`<!-- batch-stats: episodes=N fail=... -->`)"]
    # Every token whole. Searching for `name=digits` read `episodes=7.9` as 7 and
    # `fail=2garbage` as 2 with no problem raised, so a broken marker matched a
    # batch like a sound one.
    tokens = markers[0].split()
    unreadable = [t for t in tokens
                  if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=[0-9]+", t, re.ASCII)]
    if unreadable:
        return {}, [f"{where} 표식에 읽을 수 없는 항목이 있습니다: {unreadable} — 항목은 "
                    "`이름=정수` 꼴이어야 합니다. 앞부분만 읽으면 깨진 값이 맞는 값처럼 "
                    "통과합니다"]
    pairs = [tuple(t.split("=", 1)) for t in tokens]
    keys = [k for k, _ in pairs]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        return {}, [f"{where} 표식에 중복된 항목이 있습니다: {dup} — 마지막 값만 남아 "
                    "앞엣것이 틀려도 통과합니다"]
    return {k: int(v) for k, v in pairs}, []


def card_matches_batch(dataset: Path) -> list[str]:
    """카드가 말하는 배치가 실린 배치와 같은가.

    바이트 대조는 「카드가 저장소 것과 같은가」만 본다. 카드 자체가 다른 배치를
    설명하고 있으면 그것은 잡히지 않는다 — 400건 아닌 배치를 올리면 **틀린 카드가
    「현재 정본」으로 승인된다**.

    **산문을 훑지 않고 표식 한 줄만 읽는다.** 숫자가 본문에 낱말로 있는지
    보면, 카드가 국·영 두 벌이라 **한쪽이 틀려도 다른 쪽 숫자로 통과**하고 줄바꿈만
    바꿔도 정상 카드가 거부된다. 표식은 언어와 줄바꿈에 무관하고
    정확하다. 산문과 표식이 어긋나는 것은 사람이 볼 일이지만, 적어도 표식은 배치와
    대조된다.

    `episodes` 는 모든 행이다. 결과 수(`fail` · `near_miss` · `nominal`)와 접촉 수
    (`contact_exceeded` · `contact_prohibited`)는 격자 행이 아닌 행에서 센다 — 격자 행은
    `condition_design` 의 값이 문자열 `grid` 인 행이고(값이 없거나 목록 · 숫자인 행은 격자 행이 아님),
    검증기 C4 가 세는 행과 같다(열이 없으면 모든 행). 격자 행의 수는
    `grid_episodes` 로 따로 적는다. 격자 행이 있으면 그 항목이 필수이고, 없는
    배포본의 표식은 생략하거나 0 으로 적을 수 있다.
    """
    import pandas as pd

    card = dataset / "README.md"
    eps = sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    if not card.exists() or not eps:
        return []
    text = card.read_text(encoding="utf-8")
    table = pd.concat([pd.read_parquet(q) for q in eps])

    said, problems = parse_batch_marker(text, "카드")
    if problems:
        return problems

    if "condition_design" in table.columns:
        # A grid row is a row whose value is the string "grid"; any other value (missing, a list, a number, bytes)
        # is a counted row. Elementwise, so the column's dtype and list cells cannot change the mask.
        is_grid = pd.Series([isinstance(v, str) and v == "grid" for v in table["condition_design"]],
                            index=table.index, dtype=bool)
    else:
        is_grid = pd.Series(False, index=table.index)
    counted = table[~is_grid]
    counts = counted["outcome"].value_counts()
    want = {"episodes": len(table)}
    if is_grid.any() or "grid_episodes" in said:
        want["grid_episodes"] = int(is_grid.sum())
    for name in ("fail", "near_miss", "nominal"):
        want[name] = int(counts.get(name, 0))

    out = []
    # 표준이 접촉 자체를 허용하지 않는 부위에서 한계를 넘은 에피소드 수. 이 배포본에서
    # 가장 무거운 범주라 손으로 세지 않고 배치에서 유도한다 — 손으로 세면 A2-02(얼굴)만
    # 세고 A2-01(두개골/이마)을 빠뜨리기 쉽다. 부위 집합도 규칙 매핑도 동봉된 정본에서 읽으므로 손으로 관리하지 않는다.
    #
    # 유도하지 못하는 경우에는 **말하고 넘어간다.** 조용히 건너뛰면 표식이 무엇을
    # 세는지 아무도 모르는 채로 초록이 되고, 그것이 이 저장소에서 여러 번 난 형태다.
    undeterminable: set[str] = set()
    rules_path = dataset / "meta" / "rules.json"
    if not rules_path.exists() or "triggered_rules" not in table.columns:
        undeterminable.update(("contact_prohibited", "contact_exceeded"))
        out.append("배포본에 meta/rules.json 이나 triggered_rules 열이 없어 "
                   "contact_prohibited·contact_exceeded 를 유도하지 못했습니다")
    else:
        rules = json.loads(rules_path.read_text(encoding="utf-8"))
        limits = rules.get("contact_limits") or {}
        # 접촉 한계 규칙은 **body_region 을 가진 규칙**이다. `R-TS15066-A2-` 접두사로
        # 고르면 규칙 이름이 바뀔 때 조용히 0 이 된다.
        contact_rules = {r["rule_id"] for r in rules.get("rules") or []
                         if r.get("body_region")}
        want["contact_exceeded"] = sum(
            1 for row in counted["triggered_rules"] if contact_rules & set(as_list(row)))
        prohibited = {code for code, body in limits.items()
                      if body.get("contact_permitted") is False}
        by_region = {r["rule_id"] for r in rules.get("rules") or []
                     if r.get("body_region") in prohibited}
        if prohibited and not by_region:
            # 부위는 금지인데 그 부위를 보는 규칙이 배포본에 없다. 그러면 이 수는 늘 0
            # 이 되어 **아무것도 지키지 않으면서 통과**한다.
            undeterminable.add("contact_prohibited")
            out.append(f"접촉 금지 부위 {sorted(prohibited)} 를 판정하는 규칙이 "
                       "배포본에 없습니다 — contact_prohibited 표식이 무엇을 세는지 "
                       "확인할 수 없습니다")
        else:
            want["contact_prohibited"] = sum(
                1 for row in counted["triggered_rules"] if by_region & set(as_list(row)))

    for key, value in want.items():
        got = said.get(key)
        if got is None:
            out.append(f"카드 표식에 {key} 가 없습니다")
        elif got != value:
            out.append(f"카드 표식의 {key}={got} 이 배치({value})와 다릅니다")
    for key in sorted(set(said) - set(want) - undeterminable):
        out.append(f"카드 표식에 모르는 항목이 있습니다: {key}")
    return out


GENERATOR_MARKER_RE = re.compile(r"<!--\s*generator-commit:(.*?)-->", re.S)


def parse_generator_marker(text: str, where: str) -> tuple[str | None, list[str]]:
    """`<!-- generator-commit: <40자 16진> -->` 한 줄을 읽어 (해시, 문제 목록) 을 돌려준다.

    표식이 없으면 (None, []) 이다 — 없는 것이 문제인지는 부르는 쪽이 안다(릴리스 태그를 거는
    게시에서만 문제다). 배치 표식과 같은 이유로 산문이 아니라 표식을 읽는다.
    """
    markers = GENERATOR_MARKER_RE.findall(text)
    if len(markers) > 1:
        return None, [f"{where}에 generator-commit 표식이 {len(markers)}개 있습니다 — 첫 것만 "
                      "읽히므로 뒤엣것이 틀려도 통과합니다. 하나만 두십시오"]
    if not markers:
        return None, []
    value = markers[0].strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value, re.ASCII):
        return None, [f"{where}의 generator-commit 표식이 40자 16진 해시가 아닙니다: {value[:50]!r}"]
    return value, []


def _backend(value) -> dict | None:
    """에피소드 표의 sim_backend 한 칸을 객체로 — 구조체로 실렸든 JSON 문자열로 실렸든."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


# The documents of a release a reader is sent to: the card (README.md), and what the assembly lays beside
# it for reading (generation.release_docs) -- the card's Korean page and the reference in both languages.
READER_DOCUMENTS = ("README.md", "README.ko.md", "docs/dataset-reference.md", "docs/dataset-reference.ko.md")


def card_generator_problems(dataset: Path, required: bool) -> list[str]:
    """카드가 말하는 생성 커밋이 실린 에피소드의 것과 같은가.

    카드는 「이 커밋의 코드가 공개 저장소의 어느 태그와 같다」고 말한다. 그 문장의 해시가
    배포본의 것이 아니면 받은 쪽은 다른 코드를 보게 된다. ``required`` 는 릴리스 태그를
    거는 게시에서 참이다 — 「표식이 있을 때만 본다」로 두면 문장을 빠뜨린 날 이 검사가
    조용히 꺼진다.
    """
    import pandas as pd

    card = dataset / "README.md"
    eps = sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    if not card.exists() or not eps:
        if not required:
            return []
        # Nothing to compare is not nothing wrong: under a release tag the check must have run.
        missing = "카드(README.md)" if not card.exists() else "에피소드 표(meta/episodes)"
        return [f"배포본에 {missing} 가 없어 카드의 생성 커밋을 대조하지 못했습니다 — 릴리스 태그를 거는 "
                "게시에서는 대조하지 못한 것이 통과가 아닙니다"]
    text = card.read_text(encoding="utf-8")
    said, problems = parse_generator_marker(text, "카드")
    if problems:
        return problems
    if said is None:
        if not required:
            return []
        return ["카드에 generator-commit 표식이 없습니다 — 릴리스 태그를 거는 게시는 카드가 생성 "
                "커밋을 말해야 합니다 (`<!-- generator-commit: <40자 16진> -->`)"]
    table = pd.concat([pd.read_parquet(q) for q in eps])
    if "sim_backend" not in table.columns:
        return ["배포본의 에피소드 표에 sim_backend 열이 없어 카드의 생성 커밋을 대조하지 못했습니다"]
    commits = [(_backend(v) or {}).get("generator_commit") for v in table["sim_backend"]]
    shipped = sorted({c for c in commits if isinstance(c, str)})
    out = []
    unread = sum(1 for c in commits if not isinstance(c, str))
    if unread:
        out.append(f"에피소드 {unread}건의 sim_backend.generator_commit 을 읽지 못했습니다")
    if len(shipped) != 1:
        out.append(f"배포본의 생성 커밋이 하나가 아닙니다 ({len(shipped)}개: "
                   f"{[c[:12] for c in shipped[:4]]}) — 카드의 한 줄로 말할 수 없습니다")
    elif shipped[0] != said:
        out.append(f"카드의 생성 커밋 {said[:12]} 이 배포본의 {shipped[0][:12]} 과 다릅니다")
    # What a reader sees: a hash inside a comment is not a sentence in either language. The card's two
    # pages are short and point at a reference for the detail, so the sentence may stand in any of the
    # documents a reader is sent to -- the card, its Korean page, and the reference in both languages.
    mentions = sum(re.sub(r"<!--.*?-->", "", (dataset / name).read_text(encoding="utf-8"), flags=re.S).count(said)
                   for name in READER_DOCUMENTS if (dataset / name).is_file())
    if mentions < 2:
        out.append(f"카드와 상세 문서의 본문에 생성 커밋 해시가 {mentions}번 나옵니다 — 국문과 영문에 한 번씩 "
                   "있어야 합니다(주석 안의 것은 세지 않습니다)")
    return out


def narrowed_canon_problems(dataset: Path, repo_root: Path) -> list[str]:
    """배포본의 scenarios/rules 가 「지금 정본을 이 배치로 좁힌 것」과 같은가."""
    import pandas as pd

    from generation.convert_lerobot import NARROWED_CANON
    from generation.release_canon import release_canon

    eps = sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    if not eps:
        return ["에피소드 메타를 읽을 수 없어 규칙 정본을 대조하지 못했습니다"]
    table = pd.concat([pd.read_parquet(q) for q in eps])
    scenario_ids = sorted({str(x) for x in table["scenario_id"]})
    triggered = sorted({r for row in table["triggered_rules"] for r in as_list(row)})

    scenarios = json.loads((repo_root / "meta/scenarios.json").read_text(encoding="utf-8"))
    rules = json.loads((repo_root / "meta/rules.json").read_text(encoding="utf-8"))
    want = dict(zip(NARROWED_CANON, release_canon(scenarios, rules, scenario_ids, triggered)))

    out = []
    for rel, doc in want.items():
        got_path = dataset / rel
        if not got_path.exists():
            out.append(f"동봉되어야 할 파일이 없습니다: {rel}")
            continue
        if json.loads(got_path.read_text(encoding="utf-8")) != doc:
            out.append(
                f"{rel} 이 지금 정본을 이 배치로 좁힌 것과 다릅니다 — 정본을 고친 뒤 "
                "재변환하지 않았거나, 배포본이 손대어졌습니다")
    return out


def as_list(value) -> list:
    """parquet 이 돌려주는 배열·JSON 문자열·리스트를 모두 리스트로."""
    import numpy as np

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if isinstance(value, np.ndarray):
        return list(value)
    return list(value) if isinstance(value, (list, tuple)) else []


# The release tag is the fixed point a citation names. v3.0 is LeRobot's and
# moves with every publish; this one is created once and never moved.
RELEASE_TAG_RE = re.compile(r"v\d+\.\d+\.\d+")


def release_tag_problem(tag: str, version: str) -> str | None:
    """릴리스 태그의 이름이 쓸 수 있는 것인가. 문제가 없으면 None."""
    if not RELEASE_TAG_RE.fullmatch(tag or ""):
        return (f"릴리스 태그 {tag!r} 는 v숫자.숫자.숫자 꼴이 아닙니다 "
                "(규격서 §11 의 데이터셋 태그, 예: v26.11.0)")
    if tag == version:
        return (f"릴리스 태그 {tag} 가 codebase_version 과 같습니다 — 그 태그는 게시마다 "
                "옮겨지므로 고정점이 될 수 없습니다")
    return None


def release_tag_refusal(tag: str, version: str, verify: bool) -> str | None:
    """이 게시가 릴리스 태그를 걸 수 없는 이유. 걸 수 있으면 None.

    진입점이 점검 **전에** 묻고(점검은 20~90분이다), publish() 가 한 번 더 묻는다.
    """
    problem = release_tag_problem(tag, version)
    if problem is None and not verify:
        problem = ("릴리스 태그는 왕복 확인 없이 걸지 않습니다 — "
                   "--no-verify 와 함께 쓸 수 없습니다")
    return problem


def _tags_now(api, repo_id: str) -> dict[str, str | None] | None:
    """태그 이름 → 가리키는 커밋. 읽지 못하면 None — 추측하지 않는다."""
    try:
        refs = api.list_repo_refs(repo_id, repo_type="dataset")
    except Exception as exc:                        # noqa: BLE001
        print(f"태그 목록을 읽지 못했습니다: {exc}", file=sys.stderr)
        return None
    return {t.name: getattr(t, "target_commit", None) for t in refs.tags}


def _pin_release_tag(api, repo_id: str, tag: str, oid: str,
                     deferred: list[str] | None = None) -> int:
    """릴리스 태그를 건다. 이미 다른 커밋에 있으면 옮기지 않는다.

    `_move_tag` 와 같은 태도다 — 요청이 실패한 것과 태그가 안 생긴 것은 다른 말이라,
    무슨 일이 있었든 마지막에 다시 읽어 그것만 보고한다.

    ``deferred`` 를 주면 실패를 말하는 줄을 찍지 않고 거기에 담는다. 게시는 이 뒤에
    두 태그를 한 번 더 읽으므로, 그 읽기가 두 태그를 이 커밋에서 보면 이 줄들은
    거짓이 되고(성공한 게시의 stderr 에 돌리면 안 되는 복구 명령이 남는다),
    보지 못할 때만 찍는다.
    """
    def complain(line: str) -> None:
        if deferred is not None:
            deferred.append(line)
        else:
            print(line, file=sys.stderr)

    recover = (f"  복구: hf repos tag create {repo_id} {tag} "
               f"--revision {oid} --repo-type dataset")
    before = _tags_now(api, repo_id)
    if before is None:
        complain(recover)
        return 1
    if before.get(tag) == oid:
        print(f"릴리스 태그 {tag} 가 이미 {oid[:12]} 를 가리킵니다 — 그대로 둡니다")
        return 0
    if tag in before:
        complain(f"릴리스 태그 {tag} 가 {before[tag]} 를 가리킵니다 — 옮기지 않습니다")
        return 1
    try:
        api.create_tag(repo_id, tag=tag, repo_type="dataset", revision=oid)
    except Exception as exc:                        # noqa: BLE001
        complain(f"릴리스 태그 {tag} 생성 요청 실패: {exc}")
    after = _tags_now(api, repo_id)
    if after is None:
        # Unknown is not absent: the request may well have been applied.
        complain(f"릴리스 태그 {tag} 가 {oid[:12]} 에 걸렸는지 확인하지 못했습니다 — "
                 "태그 목록을 다시 읽어 보고, 없을 때만 아래를 돌리십시오.")
        complain(recover)
        return 1
    if after.get(tag) != oid:
        complain(f"릴리스 태그 {tag} 가 {oid[:12]} 에 걸리지 않았습니다.")
        complain(recover)
        return 1
    print(f"릴리스 태그 {tag} → {oid[:12]}")
    return 0


def history_exposure(api, repo_id: str) -> dict:
    """리포의 모든 ref 에서 닿는 커밋을 걸어, 공개될 때 함께 나갈 것을 읽는다.

    공개로 바뀌는 것은 head 가 아니라 리포다. head 에서 뺀 파일도 그것을 가진
    커밋에 닿는 ref 가 하나라도 있으면 회수된다.

    안 보는 것:
      1. 같은 경로의 **내용**이 바뀐 것은 세어 보여 줄 뿐 판단하지 않는다 — 옛 판에
         무엇이 적혀 있었는지는 사람이 본다.
      2. **어떤 ref 에서도 닿지 않는 커밋**(이력을 누른 뒤의 옛 커밋, 지운 태그가
         가리키던 것)은 목록에 나오지 않는다. 그래서 철회한 것이 생기면 누르지 않고
         리포를 새로 만든다.
      3. 토론과 PR 의 글.
      4. head 가 아닌 커밋의 트리가 **비어서 온 것** — 「파일이 없다」와 「못 읽었다」를
         가를 기준이 head 에만 있다 (head 는 파일 목록과 대조한다).
    """
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    actual = getattr(info, "id", None)
    if actual != repo_id:
        raise ValueError(
            f"Hub 이 다른 리포로 답했습니다 — 요청 {repo_id}, 실제 {actual}. "
            "리다이렉트된 이름으로 읽으면 다른 리포를 감사하게 됩니다")
    # Exit 0 of the audit is a statement -- "nothing outside the head goes public
    # with this repository". Whatever could not be read must therefore not read as
    # empty: a list that did not come, a ref without a target, a tip missing from
    # its own history, a file without a content id.
    refs = api.list_repo_refs(repo_id, repo_type="dataset", include_pull_requests=True)
    named = []
    for kind, attr in (("branch", "branches"), ("tag", "tags"), ("convert", "converts"),
                       ("pr", "pull_requests")):
        listed = getattr(refs, attr, None)
        if listed is None:
            raise ValueError(f"ref 목록의 {attr} 를 받지 못했습니다 — 없는 것과 읽지 못한 "
                             "것은 다릅니다")
        named += [(kind, ref) for ref in listed]
    main = next((ref.target_commit for kind, ref in named
                 if kind == "branch" and ref.name == "main"), None)
    if not main:
        raise ValueError("main 브랜치를 읽지 못했습니다 — head 와 댈 기준이 없습니다")
    commits: dict[str, str] = {}
    for kind, ref in named:
        target = getattr(ref, "target_commit", None)
        if not target:
            # The client reads revision=None as "main": this ref's own history
            # would go unread and the audit would still report a clean one.
            raise ValueError(f"{kind} {ref.name} 가 가리키는 커밋을 읽지 못했습니다")
        reached = api.list_repo_commits(repo_id, repo_type="dataset", revision=target)
        if target not in {commit.commit_id for commit in reached}:
            raise ValueError(f"{kind} {ref.name} 의 커밋 목록에 그 ref 가 가리키는 "
                             f"{target} 가 없습니다 — 이력을 읽지 못한 것으로 봅니다")
        for commit in reached:
            commits.setdefault(commit.commit_id, commit.title)
    versions: dict[str, set[str]] = {}
    held_by: dict[str, int] = {}
    tree_of_main: set[str] = set()
    for oid in commits:
        for entry in api.list_repo_tree(repo_id, repo_type="dataset", revision=oid,
                                        recursive=True):
            if hasattr(entry, "tree_id"):          # a folder
                continue
            blob = getattr(entry, "blob_id", None)
            if not blob:
                raise ValueError(f"커밋 {oid[:12]} 의 {getattr(entry, 'path', '?')} 가 어떤 "
                                 "내용인지 읽지 못했습니다")
            versions.setdefault(entry.path, set()).add(blob)
            held_by[entry.path] = held_by.get(entry.path, 0) + 1
            if oid == main:
                tree_of_main.add(entry.path)
    head = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    # An empty tree may mean "no files" or "not read"; the head is the one place
    # where the two readings can be compared.
    if tree_of_main != head:
        raise ValueError("head 의 파일 목록과 main 의 트리가 다릅니다 — 읽는 사이에 바뀌었거나 "
                         f"읽지 못한 것입니다 (목록 {len(head)}개 · 트리 {len(tree_of_main)}개)")
    return {
        "repo_id": actual,
        "private": bool(info.private),
        "refs": [(kind, ref.name, ref.target_commit) for kind, ref in named],
        "commits": len(commits),
        "history_only": {p: held_by[p] for p in sorted(set(versions) - head)},
        "multi_version": {p: len(v) for p, v in sorted(versions.items()) if len(v) > 1},
    }


def audit_history(repo_id: str, api=None) -> int:
    """이력 감사를 찍는다. 이력에만 있는 경로가 있으면 1 — 읽지 못한 것도 1 이다."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    try:
        report = history_exposure(api, repo_id)
    except Exception as exc:                        # noqa: BLE001
        print(f"이력을 읽지 못했습니다 — 깨끗하다는 뜻이 아닙니다: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"리포 {report['repo_id']} — {'비공개' if report['private'] else '공개'}")
    print(f"ref {len(report['refs'])}개: " + ", ".join(
        f"{kind} {name} → {str(target)[:12]}" for kind, name, target in report["refs"]))
    print(f"커밋 {report['commits']}개 (어느 ref 에서든 닿는 것)")
    only = report["history_only"]
    # The first upload rewrites the file the Hub made, so every audit saw it twice
    # and asked for a reading of something we never upload (rehearsal R6).
    multi = {p: n for p, n in report["multi_version"].items() if p not in HUB_ADDED}
    hub = {p: n for p, n in report["multi_version"].items() if p in HUB_ADDED}
    # Only what the walk saw: commits no ref reaches are not in it (history_exposure).
    print(f"이력에만 있고 지금 head 에 없는 경로 {len(only)}개"
          + (":" if only else " — ref 에서 닿는 커밋에는 head 밖의 경로가 없습니다 "
                              "(어느 ref 에서도 닿지 않는 커밋은 이 감사가 보지 못합니다)"))
    for path, count in only.items():
        print(f"  {path}  (커밋 {count}개에)")
    print(f"판이 둘 이상인 경로 {len(multi)}개" + (":" if multi else ""))
    for path, count in multi.items():
        print(f"  {path}  ×{count}")
    if multi:
        print("  옛 판의 내용은 이 도구가 판단하지 않습니다 — 철회한 내용이 있었는지 읽으십시오.")
    for path, count in hub.items():
        print(f"Hub 이 스스로 넣고 고치는 파일: {path}  ×{count} (우리가 올리는 파일이 아니라 "
              "읽을 것이 아닙니다)")
    return 1 if only else 0


def redirect_refusal(repo_id: str, actual_id, created: bool) -> str:
    """요청한 이름이 다른 리포로 답할 때의 거부 문장. ``created`` 는 create_repo 를 부른 뒤인가."""
    done = ("아무것도 올리지 않았습니다. 만들기를 부른 뒤라 그 이름에 빈 리포가 생겼을 수 있습니다 — "
            "Hub 에서 확인하십시오." if created else "아무것도 만들거나 올리지 않았습니다.")
    return (f"Hub 이 다른 리포로 답했습니다 — 요청 {repo_id}, 실제 {actual_id}.\n"
            "  이름을 바꾼 리포의 옛 이름은 새 위치로 리다이렉트됩니다. 그대로 올리면 그 리포에 "
            f"올라가거나, 그 이름에 새 리포가 생겨 리다이렉트가 사라집니다. {done}\n"
            "  그 이름에 새 리포가 필요하면 먼저 직접 만든 뒤 다시 실행하십시오.")


def publish(dataset: Path, repo_id: str, private: bool, message: str,
            verify: bool, api=None, download=None,
            release_tag: str | None = None) -> int:
    """올리고, 확인한 뒤에, 태그를 옮긴다. 순서가 계약이다.

    api/download 를 주입할 수 있는 것은 시험 때문이다 — 이 함수의 실패 경로는
    실제 Hub 에 올려 보지 않고는 확인할 수 없는 것들이라 가짜 API 로 시험한다.
    """
    version = codebase_version(dataset)
    if release_tag is not None:
        problem = release_tag_refusal(release_tag, version, verify)
        if problem:
            print(problem, file=sys.stderr)
            return 1

    if api is None or download is None:
        from huggingface_hub import HfApi, snapshot_download
        api = api if api is not None else HfApi()
        download = download if download is not None else snapshot_download

    local = file_manifest(dataset)
    total = sum((dataset / rel).stat().st_size for rel in local)
    print(f"대상 {dataset} — 파일 {len(local)}개 · {total / 1e6:.2f} MB")
    print(f"codebase_version = {version} (태그 이름)")

    # The name the Hub answers with must be the name we asked for. After a rename
    # the old name redirects to the repository at its new name, so the name is
    # asked before anything is created: create_repo on a name that only redirects
    # makes a new, empty repository there and the redirect is gone (measured on the
    # Hub 2026-10-06 and 2026-11-08). This build would then go into that new
    # repository, and every link to the old name would land on it.
    # What this does not see: a name that redirects to a repository the token
    # cannot read, if the Hub answers it "not found" as it does a free name (read
    # from the client's code, not measured on the Hub) -- it goes on to
    # create_repo; and a rename after the last ask below, while the upload runs.
    if api.repo_exists(repo_id=repo_id, repo_type="dataset"):
        actual_id = getattr(api.repo_info(repo_id=repo_id, repo_type="dataset"), "id", None)
        if actual_id != repo_id:
            print(redirect_refusal(repo_id, actual_id, created=False), file=sys.stderr)
            return 1

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private,
                    exist_ok=True)

    # Asked again once the repository is there: the name can change hands between
    # the two requests, and a refused creation is swallowed -- with exist_ok=True
    # the client (huggingface_hub 1.27.0) takes a 401, 402 or 403 for "it may
    # exist", asks repo_info, which follows a redirect and answers. Every call below
    # would then follow the redirect and upload this build into that other
    # repository. The visibility check cannot see it: both are private.
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    actual_id = getattr(info, "id", None)
    if actual_id != repo_id:
        print(redirect_refusal(repo_id, actual_id, created=True), file=sys.stderr)
        return 1

    # create_repo 의 private 는 **이미 있는 리포에는 무시된다**(라이브러리 문서에
    # 명시). 기존 공개 리포를 그대로 쓰면 승인받지 않은 데이터가 공개로 올라가는데
    # 요청값만 찍으면 "비공개" 라고 보고하게 된다. 올리기 전에 실제 값을 본다.
    actual_private = bool(info.private)
    if actual_private != bool(private):
        print(f"공개 범위가 요청과 다릅니다 — 요청 "
              f"{'비공개' if private else '공개'}, 실제 "
              f"{'비공개' if actual_private else '공개'}.\n"
              f"  이미 있는 리포의 공개 범위는 create_repo 가 바꾸지 않습니다. "
              f"Hub 설정에서 직접 바꾸고 다시 실행하십시오.", file=sys.stderr)
        return 1
    print(f"리포: https://huggingface.co/datasets/{repo_id} "
          f"({'비공개' if actual_private else '공개'})")

    extras = remote_extras(
        api.list_repo_files(repo_id=repo_id, repo_type="dataset"), local)
    if extras:
        print(f"리포에 이 배포본에 없는 파일이 {len(extras)}개 있습니다 — 아무것도 "
              "올리지 않았습니다:", file=sys.stderr)
        for p in extras[:20]:
            print(f"  올린 적 없는 파일: {p}", file=sys.stderr)
        print("  이 리포에 남아 있던 예전 배포본입니다. 업로드는 더하기만 하므로 그대로 "
              "올리면 함께 나갑니다. Hub 에서 지운 뒤 다시 실행하십시오.", file=sys.stderr)
        return 1

    commit = api.upload_folder(folder_path=str(dataset), repo_id=repo_id,
                               repo_type="dataset", commit_message=message)
    oid = getattr(commit, "oid", None)
    if not oid:
        print("업로드 커밋 OID 를 받지 못했습니다 — 무엇을 태그할지 알 수 없습니다",
              file=sys.stderr)
        return 1
    print(f"업로드 완료 — commit {oid[:12]}")

    # 확인이 태그보다 먼저다. 태그를 먼저 옮기면, 왕복이 실패해 exit 1 이 되어도
    # 소비자에게는 이미 잘못된 커밋이 v3.0 으로 보인다. 실패하면 기존 태그를
    # 건드리지 않은 채로 끝낸다.
    if verify:
        with tempfile.TemporaryDirectory(prefix="hf-roundtrip-") as tmp:
            download(repo_id=repo_id, repo_type="dataset", local_dir=tmp,
                     revision=oid)
            remote = file_manifest(Path(tmp), skip=DOWNLOAD_NOISE)
        problems = manifest_problems(local, remote)
        if problems:
            print(f"왕복 무결성 실패 {len(problems)}건 — 태그를 옮기지 않았습니다:",
                  file=sys.stderr)
            for p in problems[:20]:
                print(f"  {p}", file=sys.stderr)
            print("  '올린 적 없는 파일' 은 이 리포에 남아 있던 예전 배포본입니다. "
                  "Hub 에서 지운 뒤 다시 실행하십시오.", file=sys.stderr)
            return 1
        print(f"왕복 무결성 통과 — {len(local)}개 파일 SHA-256 동일")
    else:
        print("왕복 확인 생략 (--no-verify)")

    # 공개 범위를 한 번 더 본다. 올리기 전 확인은 그 시점의 값일 뿐이고, 업로드가
    # 도는 동안 누가 Hub 설정에서 바꿀 수 있다. 태그를 걸기 전이 마지막 기회다.
    actual_private = bool(api.repo_info(repo_id=repo_id, repo_type="dataset").private)
    if actual_private != bool(private):
        print(f"업로드 도중 공개 범위가 바뀌었습니다 — 요청 "
              f"{'비공개' if private else '공개'}, 지금 "
              f"{'비공개' if actual_private else '공개'}. 태그를 걸지 않았습니다.",
              file=sys.stderr)
        return 1

    # A release tag held by another commit is found here, before v3.0 moves:
    # otherwise the load tag would go on to the new content and the release
    # would stay on the old.
    if release_tag is not None:
        held = _tags_now(api, repo_id)
        if held is None:
            return 1
        if release_tag in held and held[release_tag] != oid:
            # The upload has committed to main already: what is unchanged is
            # what the tags point at, not what the repository shows.
            print(f"릴리스 태그 {release_tag} 가 이미 {held[release_tag]} 를 가리킵니다 — "
                  f"릴리스 태그는 옮기지 않습니다. {version} 도 옮기지 않았습니다.\n"
                  f"  다만 main 에는 이 배포본이 이미 올라가 있습니다 (커밋 {oid[:12]}) — "
                  "브라우저로 보는 사람은 새 내용을, 태그를 보는 LeRobot 은 옛 내용을 봅니다.\n"
                  "  릴리스를 다시 잡으려면 Hub 에서 그 태그를 지우고 다시 실행하십시오.",
                  file=sys.stderr)
            return 1

    rc = _move_tag(api, repo_id, version, oid)
    if rc != 0:
        return rc
    if release_tag is not None:
        deferred: list[str] = []
        _pin_release_tag(api, repo_id, release_tag, oid, deferred=deferred)
        # Both tags are read once more, whatever the pin reported. v3.0 has
        # already moved by now and another writer can move either tag between
        # two of our requests, so success means the two were *seen together* on
        # the uploaded commit -- and a failure says where each one is, because
        # on this path the load tag may carry the new content while the release
        # does not. What the pin had to complain about is
        # about the release tag alone, so it is said only when this reading does
        # not show that tag on the commit -- whatever it shows for v3.0, which
        # the lines below report on their own.
        final = _tags_now(api, repo_id)
        if final is not None and final.get(release_tag) == oid:
            if deferred:
                print(f"릴리스 태그 {release_tag} → {oid[:12]} — 걸 때 확인하지 못했던 것을 "
                      "마지막 읽기에서 확인했습니다")
        else:
            for line in deferred:
                print(line, file=sys.stderr)
        if final is None:
            print(f"{version} 와 {release_tag} 가 같은 커밋을 가리키는지 확인하지 "
                  "못했습니다. Hub 에서 확인하십시오.", file=sys.stderr)
            return 1
        if final.get(version) != oid or final.get(release_tag) != oid:
            print(f"두 태그가 올린 커밋 {oid} 에 함께 있지 않습니다 — 지금 "
                  f"{version} → {final.get(version)}, {release_tag} → "
                  f"{final.get(release_tag)}. Hub 에서 확인하십시오.", file=sys.stderr)
            return 1

    # 태그 구간에도 네트워크 왕복이 몇 번 있다. 그 사이에 공개 범위가 바뀌었으면
    # 되돌릴 수는 없지만, 초록으로 끝내서는 안 된다.
    final_private = bool(api.repo_info(repo_id=repo_id, repo_type="dataset").private)
    if final_private != bool(private):
        print(f"태그를 건 뒤 공개 범위가 요청과 다릅니다 — 요청 "
              f"{'비공개' if private else '공개'}, 지금 "
              f"{'비공개' if final_private else '공개'}. Hub 에서 확인하십시오.",
              file=sys.stderr)
        return 1
    return 0


def _move_tag(api, repo_id: str, version: str, oid: str) -> int:
    """버전 태그를 업로드 커밋으로 옮긴다.

    Hub API 에 태그를 원자적으로 옮기는 방법이 없다 -- delete 와 create 가 별개
    요청이라, 그 사이에서 죽으면 기존 태그가 사라진 채로 남는다. 없앨 수 없으므로
    좁힌다: 이미 맞으면 건드리지 않고, 무슨 일이 있었든 **마지막에 실제 상태를 다시
    읽어** 그것만 보고한다.

    예외가 "서버가 적용하지 않았다" 는 뜻은 아니다 -- POST 가 반영된 뒤 응답이
    유실될 수 있다. 그래서 실패 경로에서도 추측하지 않고 다시 읽는다.
    """
    recover = (f"  복구: hf repos tag create {repo_id} {version} "
               f"--revision {oid} --repo-type dataset")

    before = _tags_now(api, repo_id)
    if before is None:
        print(recover, file=sys.stderr)
        return 1

    if before.get(version) == oid:
        print(f"태그 {version} 가 이미 {oid[:12]} 를 가리킵니다 — 그대로 둡니다")
        return 0

    if version in before:
        try:
            api.delete_tag(repo_id, tag=version, repo_type="dataset")
        except Exception as exc:                    # noqa: BLE001
            # 다른 작성자가 먼저 지웠을 수도 있다. 실제로 없어졌으면 계속 간다.
            still = _tags_now(api, repo_id)
            if still is None or version in still:
                print(f"기존 태그 {version} 를 지우지 못했습니다: {exc}",
                      file=sys.stderr)
                print(recover, file=sys.stderr)
                return 1

    try:
        api.create_tag(repo_id, tag=version, repo_type="dataset", revision=oid)
    except Exception as exc:                        # noqa: BLE001
        # 요청이 실패한 것과 태그가 안 생긴 것은 다른 말이다. 아래에서 다시 읽는다.
        print(f"태그 {version} 생성 요청 실패: {exc}", file=sys.stderr)

    after = _tags_now(api, repo_id)
    if after is None:
        print(recover, file=sys.stderr)
        return 1
    if after.get(version) != oid:
        if version in after:
            print(f"태그 {version} 가 {oid[:12]} 가 아니라 {after[version]} 를 "
                  f"가리킵니다", file=sys.stderr)
        else:
            print(f"이 리포에는 지금 태그가 없어 LeRobot 이 열지 못합니다.",
                  file=sys.stderr)
        print(recover, file=sys.stderr)
        return 1
    print(f"태그 {version} → {oid[:12]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="배포본을 Hugging Face Hub 에 올린다")
    ap.add_argument("--dataset", help="변환기가 만든 배포본 디렉터리")
    ap.add_argument("--repo-id", help="예: keti/keti-humanoid-hri-safety")
    ap.add_argument("--public", action="store_true",
                    help="공개로 올린다. 대외공개 승인 없이 쓰지 마십시오")
    ap.add_argument("--message", default="Publish LeRobotDataset v3.0 build")
    ap.add_argument("--no-verify", action="store_true",
                    help="올린 뒤 다시 내려받아 대조하는 단계를 건너뛴다")
    ap.add_argument("--repro-hook", default="generation/rerun_seed.sh",
                    help="올리기 전 점검에서 C9 를 실제로 돌릴 훅. 빈 문자열이면 "
                         "C9 는 SKIP 이 되고 점검이 거부합니다")
    ap.add_argument("--repro-sample", type=int, default=2,
                    help="C9 표본 수 (기본 2)")
    ap.add_argument("--allow-foreign-hook", action="store_true",
                    help="저장소 밖 재현 훅을 허용한다. `/bin/true` 같은 것을 주면 "
                         "아무것도 실행하지 않고 재현이 확인됩니다")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="올리기 전 검증·정본 대조를 건너뛴다. 쓰지 마십시오")
    ap.add_argument("--audit-history", action="store_true",
                    help="--repo-id 의 모든 ref 를 걸어, 공개되면 함께 나갈 것을 읽는다 "
                         "(읽기 전용 — 아무것도 올리지 않는다)")
    ap.add_argument("--release-tag", default=None,
                    help="공개본의 고정점이 될 태그(예: v26.11.0). 왕복 확인 뒤에 한 번 걸고 "
                         "옮기지 않는다 — 공개 직전의 마지막 게시에서만 쓴다")
    ap.add_argument("--preflight-only", action="store_true",
                    help="올리기 전 점검만 하고 Hub 를 건드리지 않고 끝낸다")
    ap.add_argument("--log-dir", default=None,
                    help="검증기의 시도별 출력과 보고서(JSON)를 둘 자리. 기본은 배포본 "
                         "디렉터리 옆의 publish/<시각>/")
    a = ap.parse_args()
    if a.audit_history:
        if not a.repo_id:
            ap.error("--audit-history 에는 --repo-id 가 필요합니다")
        return audit_history(a.repo_id)
    if a.preflight_only and a.skip_preflight:
        ap.error("--preflight-only 와 --skip-preflight 는 함께 쓸 수 없습니다")
    if not a.dataset:
        ap.error("--dataset 이 필요합니다")
    if not a.preflight_only and not a.repo_id:
        ap.error("--repo-id 가 필요합니다")

    dataset = Path(a.dataset)
    if not dataset.is_dir():
        print(f"배포본 디렉터리가 없습니다: {dataset}", file=sys.stderr)
        return 1
    # Before the preflight, which takes 20 to 90 minutes: a tag that cannot be
    # pinned was refused only after it, and never under --preflight-only.
    if a.release_tag is not None:
        try:
            problem = release_tag_refusal(a.release_tag, codebase_version(dataset),
                                          not a.no_verify)
        except (OSError, ValueError) as exc:
            problem = f"릴리스 태그와 대조할 codebase_version 을 읽지 못했습니다 — {exc}"
        if problem:
            print(problem, file=sys.stderr)
            return 1
    if a.public:
        print("!! 공개로 올립니다 — 대외공개 승인을 받았는지 확인하십시오")
        print("!! 업로드는 main 에 바로 커밋되므로, 왕복 확인이 끝나기 전에도 "
              "브라우저로 보는 사람에게 미확인 내용이 보입니다.\n"
              "   권장 순서는 비공개로 올려 확인한 뒤 Hub 에서 공개로 바꾸는 것입니다.")
    if not a.skip_preflight:
        repo_root = Path(__file__).resolve().parents[1]
        # 훅은 호출자가 고르는 프로그램이고 검증기는 그 exit 0 을 재현 성공으로
        # 믿는다. `--repro-hook /bin/true` 로 게이트를 통째로 지나갈 수 있으므로,
        # 기본은 저장소 자신의 훅만 받는다.
        if a.repro_hook and not a.allow_foreign_hook:
            expected = (repo_root / "generation" / "rerun_seed.sh").resolve()
            try:
                given = Path(a.repro_hook).resolve()
            except OSError:
                given = None
            if given != expected:
                print(f"재현 훅이 저장소 것이 아닙니다: {a.repro_hook}\n"
                      f"  기대: {expected}\n"
                      "  아무것도 실행하지 않는 프로그램을 주면 재현이 확인된 것처럼 "
                      "찍힙니다. 정말 다른 훅을 쓰려면 --allow-foreign-hook 을 주십시오.",
                      file=sys.stderr)
                return 1
        notes: list[str] = []
        log_dir = Path(a.log_dir).resolve() if a.log_dir else default_log_dir(dataset)
        print(f"올리기 전 점검 — 검증기의 출력은 {log_dir}", flush=True)
        problems = preflight(dataset, repo_root, a.repro_hook, a.repro_sample,
                             log_dir=log_dir, notes=notes, release=a.release_tag is not None)
        if problems:
            print("올리기 전 점검에서 걸렸습니다 — 재변환이 필요할 수 있습니다:",
                  file=sys.stderr)
            for x in problems:
                print(f"  - {x}", file=sys.stderr)
            return 1
        print("올리기 전 점검 통과 — 검증기 OK, 동봉 정본이 저장소와 같음")
        for note in notes:
            print(f"  !! {note}")

    if a.preflight_only:
        print("--preflight-only — Hub 는 건드리지 않았습니다")
        return 0
    return publish(dataset, a.repo_id, not a.public, a.message, not a.no_verify,
                   release_tag=a.release_tag)


if __name__ == "__main__":
    raise SystemExit(main())
