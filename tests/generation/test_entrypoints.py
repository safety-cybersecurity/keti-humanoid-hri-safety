"""문서에 적힌 명령이 실제로 도는지.

generation/README.md 가 이 스크립트들을 `PYTHONPATH` 없이
부르라고 적어 두는데, 2026-08-23 확인 시점에 넷 다 ModuleNotFoundError 로 죽었습니다.
스크립트로 실행하면 sys.path[0] 이 저장소 루트가 아니라 파일이 있는 디렉터리이기
때문입니다. 다음에 새 진입점이 생겨도 같은 함정에 빠지지 않도록 시험으로 고정합니다.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Isaac 이 필요한 것은 뺀다 -- 부팅 없이는 --help 조차 돌지 않는다.
ENTRYPOINTS = [
    "generation/labeling/label_episodes.py",
    "generation/labeling/check_release_labels.py",
    "generation/convert_lerobot.py",
    "generation/publish_dataset.py",
    "generation/compare_published.py",
    "generation/rerun_policy.py",
    "generation/envs/s1a/measure_stop_time.py",
    "generation/scene/measure_slice.py",
    "generation/scene/detect_humans.py",
    "generation/run_batch.py",
    "generation/assemble_batch.py",
    "generation/scorecard.py",
]


@pytest.mark.parametrize("script", ENTRYPOINTS)
def test_the_script_runs_without_pythonpath(script):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"{script}\n{proc.stderr[-800:]}"


@pytest.mark.parametrize("script", ENTRYPOINTS)
def test_the_script_also_runs_from_another_directory(script):
    # 저장소 루트에서만 도는 것은 우연히 sys.path 에 '' 가 들어간 것일 수 있다.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / script), "--help"],
        cwd="/", env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"{script}\n{proc.stderr[-800:]}"


# run_s1a.py boots Isaac, but only after it has parsed and checked its arguments: those refusals run without it.
RUN_S1A = "generation/envs/run_s1a.py"


def _run_s1a(*args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, RUN_S1A, *args], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                          timeout=120)


def test_run_s1a_help_names_the_new_scenarios():
    proc = _run_s1a("--help")
    assert proc.returncode == 0, proc.stderr[-800:]
    for word in ("S1e", "S1f", "S1g", "--policy", "--behavior"):
        assert word in proc.stdout


@pytest.mark.parametrize("args, said", [
    (["--scenario", "S1e", "--episodes", "2"], "--episodes 1"),
    (["--scenario", "S1g", "--behavior", "nocolon"], "모듈:이름"),
    (["--scenario", "S1f", "--behavior", "a:b:c"], "모듈:이름"),
    (["--scenario", "S1d", "--policy", "x.pt"], "조건표 시나리오"),
    (["--scenario", "S1a", "--behavior", "generation.behavior:PassThrough"], "조건표 시나리오"),
    (["--scenario", "S1f", "--no-skin", "--out", "x"], "조끼"),
    (["--scenario", "S1e", "--seq", "400"], "행 400"),
    (["--scenario", "S1g", "--policy", "/nonexistent/keti-policy.pt"], "/nonexistent/keti-policy.pt"),
])
def test_run_s1a_refuses_before_booting(args, said):
    # ``said`` is a phrase of the refusal itself, never an option name: argparse's usage line names every option.
    proc = _run_s1a(*args)
    assert proc.returncode == 2, (proc.returncode, proc.stdout[-400:], proc.stderr[-800:])
    assert said in proc.stderr, proc.stderr[-800:]
