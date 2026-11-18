"""C9 표본은 접두 재현 비용을 감당할 수 있는 범위에서 골라야 한다.

에피소드 단독 재현이 성립하지 않으므로(규격서 §3) 재현 훅은 그 에피소드를 만든 실행을
첫 번호(`run_start_seq`)부터 그 에피소드까지 다시 생성합니다. 비용이 그 에피소드가 자기 실행에서
몇 번째인가에 비례한다는 뜻이고, 한 실행으로 만든 400건 배포본에서
380번을 고르면 3시간이 넘어 훅 타임아웃(1800초)에 걸립니다 — 그러면 데이터에 아무
문제가 없는데 「재현 실패」로 빨간불이 섭니다.

이 파일이 없던 동안 표본은 전 구간 무작위였습니다. 10건짜리 픽스처에서는 드러나지
않고 실제 배포본 규모에서만 드러나는 형태입니다.
"""

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

import validator.validate_dataset as vd
from generation.compare_published import NOT_COMPARED
from generation.compare_repro import comparable_step_fields
from generation.field_map import CAMERAS
from validator.validate_dataset import (REPRO_CAMERA_EPISODE_S, REPRO_HOOK_TIMEOUT_S, REPRO_MAX_SEQ,
                                        REPRO_MAX_SEQ_CAMERAS, REPRO_OVERHEAD_S,
                                        REPRO_TIMEOUT_SHARE, Dataset, Report, c9_seed_repro)


def episodes(n, first=0, scenario="S1a"):
    return pd.DataFrame({
        "episode_id": [f"{scenario}_{i:04d}_seed{1000 + i}" for i in range(first, first + n)],
        "randomization_seed": [1000 + i for i in range(first, first + n)],
        "source": ["sim"] * n,
    })


def run_c9(eps, monkeypatch, sample=3, max_seq=REPRO_MAX_SEQ):
    calls = []

    def fake_run(argv, env, timeout_s):
        calls.append(argv)
        return 0, "", ""

    monkeypatch.setattr(vd, "run_hook", fake_run)
    rep = Report(Path("."))
    # 실물처럼 root 를 가진 대역. 맨 object() 였을 때는 C9 가 훅에게 넘기는 것이
    # 무엇인지 이 시험이 볼 수 없었다.
    ds = type("DS", (), {"root": Path(".")})()
    c9_seed_repro(ds, eps, "/bin/true", sample, rep, max_seq)
    seqs = [int(c[1].split("_")[1]) for c in calls]
    return seqs, [f for f in rep.findings if f.check == "C9"]


def test_a_large_batch_is_sampled_from_the_cheap_end(monkeypatch):
    seqs, findings = run_c9(episodes(400), monkeypatch)
    assert seqs, "표본을 하나도 고르지 않았습니다"
    assert max(seqs) <= REPRO_MAX_SEQ, (
        f"접두 {REPRO_MAX_SEQ} 번을 넘는 에피소드를 골랐습니다: {seqs} — "
        "재현 훅이 그만큼을 다시 생성하므로 타임아웃에 걸립니다")


def test_the_unchecked_remainder_is_reported_not_hidden(monkeypatch):
    """확인하지 않은 것을 확인한 것처럼 말하지 않는다."""
    _, findings = run_c9(episodes(400), monkeypatch)
    said = " ".join(f.message for f in findings)
    assert "확인하지" in said and "379" in said, said


def test_a_small_batch_is_sampled_as_before(monkeypatch):
    """10건짜리 배포본에서는 동작이 달라지지 않아야 한다."""
    seqs, findings = run_c9(episodes(10), monkeypatch)
    assert len(seqs) == 3
    assert not any("확인하지" in f.message for f in findings)


def test_c9_samples_every_scenario_before_a_second_of_any(monkeypatch):
    names = ["S1e", "S1f", "S1g"]
    eligible = pd.concat([episodes(5, scenario=s).assign(scenario_id=s) for s in ("S1g", "S1e", "S1f")],
                         ignore_index=True)
    firsts = [eligible[eligible["scenario_id"] == s].sample(1, random_state=0).index[0] for s in names]

    three = vd._c9_picks(eligible, 3)
    assert list(three["scenario_id"]) == names and list(three.index) == firsts
    two = vd._c9_picks(eligible, 2)
    assert list(two["scenario_id"]) == names[:2] and list(two.index) == firsts[:2]
    five = vd._c9_picks(eligible, 5)
    rest = eligible.drop(firsts)
    assert list(five.index) == firsts + list(rest.sample(2, random_state=0).index)
    assert len(vd._c9_picks(eligible, 40)) == len(eligible) and vd._c9_picks(eligible, 40).index.is_unique

    # One scenario, or no scenario_id at all: the picks C9 always made.
    one = eligible[eligible["scenario_id"] == "S1f"]
    assert list(vd._c9_picks(one, 3).index) == list(one.sample(3, random_state=0).index)
    bare = eligible.drop(columns="scenario_id")
    assert list(vd._c9_picks(bare, 3).index) == list(bare.sample(3, random_state=0).index)

    # The hook is called in that order: every scenario once before any twice.
    calls = []
    monkeypatch.setattr(vd, "run_hook", lambda argv, env, timeout_s: calls.append(argv) or (0, "", ""))
    ds = type("DS", (), {"root": Path("."), "ships_cameras": lambda self: False})()
    c9_seed_repro(ds, eligible, "/bin/true", 4, Report(Path(".")), REPRO_MAX_SEQ)
    assert [c[1].split("_")[0] for c in calls][:3] == names, calls
    assert len(calls) == 4


def test_a_batch_starting_past_the_bound_fails_instead_of_passing_silently(monkeypatch):
    """훅을 지정했는데 한 번도 못 불렀으면 실패다.

    경고로 끝내면 기본 실행이 exit 0 이라 「C9 를 돌렸다」가 아무것도 확인하지 않고
    성공합니다 — 비용 상한이 여는 구멍입니다.
    """
    seqs, findings = run_c9(episodes(5, first=900), monkeypatch)
    assert seqs == []
    assert any(f.severity == "ERROR" for f in findings), findings
    assert not any(f.severity == "WARN" for f in findings), "경고로 끝나면 exit 0 입니다"


# ── Runs that do not start at 0: a batch generated in chunks ──
#
# The hook reruns the run that made the episode, from its first seq up to the episode. The cost
# is the episode's place in its run, not its number: with runs of ten, every episode of a
# 400-episode batch is within reach of the bound for a distribution without cameras. A run longer
# than the camera bound loses its back episodes (test_a_camera_distribution_whose_run_is_longer_...);
# a batch made one row to a run has every episode at place 0 (test_a_camera_distribution_made_one_row_to_a_run_...).

def chunked(n, chunk=10, scenario="S1d"):
    eps = episodes(n, scenario=scenario)
    eps["run_start_seq"] = [i - i % chunk for i in range(n)]
    return eps


def run_c9_calls(eps, monkeypatch, sample=3, max_seq=REPRO_MAX_SEQ):
    calls = []

    def fake_run(argv, env, timeout_s):
        calls.append(argv)
        return 0, "", ""

    monkeypatch.setattr(vd, "run_hook", fake_run)
    rep = Report(Path("."))
    ds = type("DS", (), {"root": Path(".")})()
    c9_seed_repro(ds, eps, "/bin/true", sample, rep, max_seq)
    return calls, [f for f in rep.findings if f.check == "C9"]


def _seq(argv):
    return int(argv[1].split("_")[1])


def test_every_episode_of_a_batch_in_runs_of_ten_is_within_reach(monkeypatch):
    calls, findings = run_c9_calls(chunked(400), monkeypatch, sample=400)
    assert sorted(_seq(c) for c in calls) == list(range(400))
    assert not any("확인하지" in f.message for f in findings), [f.message for f in findings]


def test_the_hook_is_told_where_the_episodes_run_began(monkeypatch):
    calls, _ = run_c9_calls(chunked(400), monkeypatch, sample=40)
    assert calls and all(len(c) == 4 for c in calls), calls
    assert [c[3] for c in calls] == [str(_seq(c) - _seq(c) % 10) for c in calls]


def test_the_bound_is_on_the_place_in_the_run_not_on_the_number(monkeypatch):
    # With runs of 20, only the front REPRO_MAX_SEQ_CAMERAS + 1 of each run (places 0..REPRO_MAX_SEQ_CAMERAS) are
    # within the hook's time limit.
    calls, findings = run_c9_calls(chunked(400, chunk=20), monkeypatch, sample=400, max_seq=REPRO_MAX_SEQ_CAMERAS)
    assert sorted(_seq(c) for c in calls) == [i for i in range(400) if i % 20 <= REPRO_MAX_SEQ_CAMERAS]
    said = " ".join(f.message for f in findings)
    assert "확인하지" in said and str(400 - 20 * (REPRO_MAX_SEQ_CAMERAS + 1)) in said, said


def test_a_camera_distribution_whose_run_is_longer_than_the_bound_is_sampled_only_in_the_front_of_each_run(monkeypatch):
    # A chunk that is one run (run_start_seq is the chunk's first seq, as a batch without a conditions table is
    # made): the runs are longer than the camera bound, so the back of every run is outside the reach and the report
    # says how many were not looked at.
    runs = REPRO_MAX_SEQ_CAMERAS + 4
    calls, findings = run_c9_calls(chunked(400, chunk=runs), monkeypatch, sample=400, max_seq=REPRO_MAX_SEQ_CAMERAS)
    reachable = [i for i in range(400) if i % runs <= REPRO_MAX_SEQ_CAMERAS]
    assert sorted(_seq(c) for c in calls) == reachable
    said = " ".join(f.message for f in findings)
    assert "확인하지" in said and f"나머지 {400 - len(reachable)}건" in said, said


def test_a_camera_distribution_made_one_row_to_a_run_is_sampled_whole(monkeypatch):
    # The published layout of the directed scenarios: a row is a run, so run_start_seq is the episode's own seq and
    # every episode sits at place 0 whatever its number or its neighbours -- rows with gaps, several scenarios,
    # numbers far past the bound and a run of numbers long enough to pass any chunk length.
    runs = REPRO_MAX_SEQ_CAMERAS + 4
    rows = ([("S1e", i) for i in range(3 * runs)] + [("S1e", 399)]
            + [("S1f", n) for n in (0, 3, 17, 41, 142, 398)] + [("S1g", n) for n in (5, 6, 58, 59, 200, 397)])
    eps = pd.DataFrame({
        "episode_id": [f"{s}_{n:04d}_seed{1000 + n}" for s, n in rows],
        "randomization_seed": [1000 + n for _, n in rows],
        "source": ["sim"] * len(rows),
        "scenario_id": [s for s, _ in rows],
        "run_start_seq": [n for _, n in rows],
    })
    calls, findings = run_c9_calls(eps, monkeypatch, sample=len(rows), max_seq=REPRO_MAX_SEQ_CAMERAS)
    assert sorted((c[1].split("_")[0], _seq(c)) for c in calls) == sorted(rows)
    assert {c[3] for c in calls} == {str(n) for _, n in rows}              # the hook is told each row's own start
    assert not any("확인하지" in f.message for f in findings), [f.message for f in findings]


def test_a_distribution_without_the_column_is_sampled_as_runs_from_zero(monkeypatch):
    calls, _ = run_c9_calls(episodes(30, scenario="S1d"), monkeypatch, sample=30, max_seq=REPRO_MAX_SEQ_CAMERAS)
    assert sorted(_seq(c) for c in calls) == list(range(REPRO_MAX_SEQ_CAMERAS + 1))
    assert {c[3] for c in calls} == {"0"}


def test_a_start_widened_to_float_is_read_as_the_integer_it_is(monkeypatch):
    # 한 건이 비면 열 전체가 float 이 된다: 10.0 은 10 이고, 빈 칸의 에피소드는 표본이 아니다.
    eps = chunked(20)
    eps["run_start_seq"] = eps["run_start_seq"].astype("float64")
    eps.loc[5, "run_start_seq"] = float("nan")
    calls, _ = run_c9_calls(eps, monkeypatch, sample=20)
    assert sorted(_seq(c) for c in calls) == [i for i in range(20) if i != 5]
    assert {c[3] for c in calls} == {"0", "10"}


@pytest.mark.parametrize("bad", [7, -1, 2.5, "three"], ids=["past its number", "negative", "fraction", "text"])
def test_an_episode_whose_start_could_not_be_its_runs_is_not_sampled(monkeypatch, bad):
    # C5 가 그 값을 오류로 낸다. C9 는 그 에피소드를 다시 만들 명령을 모른다.
    eps = chunked(10)
    eps["run_start_seq"] = eps["run_start_seq"].astype(object)
    eps.loc[3, "run_start_seq"] = bad
    calls, _ = run_c9_calls(eps, monkeypatch, sample=10)
    assert sorted(_seq(c) for c in calls) == [i for i in range(10) if i != 3]


# ── Camera distributions: a prefix bound that fits the hook's time at the camera cost ──
#
# A camera rerun renders and encodes every camera of the canon each step. At REPRO_MAX_SEQ its
# slowest prefix would not fit the hook's timeout, so a distribution that
# ships meta/cameras.json gets its own default bound, derived from the measured cost.

def _distribution(tmp_path, *, cameras):
    """A distribution root with 400 episodes in meta/episodes, and meta/cameras.json when
    ``cameras`` -- all C9's default needs to tell the two kinds apart."""
    root = tmp_path / "published"
    (root / "meta" / "episodes").mkdir(parents=True)
    episodes(400, scenario="S1d" if cameras else "S1a").to_parquet(
        root / "meta" / "episodes" / "file-000.parquet")
    if cameras:
        (root / "meta" / "cameras.json").write_text("{}", encoding="utf-8")
    return root


def run_c9_by_default(root, monkeypatch, sample=3, **kw):
    """C9 the way the command line runs it -- no --repro-max-seq -- on a real Dataset."""
    calls, timeouts = [], []

    def fake_run(argv, env, timeout_s):
        calls.append(argv)
        timeouts.append(timeout_s)
        return 0, "", ""

    monkeypatch.setattr(vd, "run_hook", fake_run)
    rep = Report(root)
    ds = Dataset(root, rep)
    c9_seed_repro(ds, ds.episodes(), "/bin/true", sample, rep, **kw)
    seqs = [int(c[1].split("_")[1]) for c in calls]
    return seqs, [f for f in rep.findings if f.check == "C9"], timeouts


def test_a_camera_distribution_is_sampled_within_the_camera_bound(tmp_path, monkeypatch):
    seqs, findings, _ = run_c9_by_default(_distribution(tmp_path, cameras=True), monkeypatch)
    assert seqs and max(seqs) <= REPRO_MAX_SEQ_CAMERAS, seqs
    said = " ".join(f.message for f in findings)
    assert f"접두 {REPRO_MAX_SEQ_CAMERAS} 번 이내 (카메라 배포본" in said, said
    assert f"나머지 {400 - (REPRO_MAX_SEQ_CAMERAS + 1)}건은 확인하지" in said, said


def test_a_distribution_without_cameras_is_sampled_exactly_as_before(tmp_path, monkeypatch):
    # The v0.1 default: the same picks and the same sentence as with REPRO_MAX_SEQ given.
    root = _distribution(tmp_path, cameras=False)
    seqs, findings, _ = run_c9_by_default(root, monkeypatch)
    given, given_findings, _ = run_c9_by_default(root, monkeypatch, max_seq=REPRO_MAX_SEQ)
    assert seqs == given and max(seqs) <= REPRO_MAX_SEQ
    assert [f.message for f in findings] == [f.message for f in given_findings]
    assert "카메라" not in " ".join(f.message for f in findings)


def test_a_given_bound_is_kept_on_a_camera_distribution(tmp_path, monkeypatch):
    # --repro-max-seq is the way to look deeper, at the caller's cost -- not overridden.
    _, findings, _ = run_c9_by_default(_distribution(tmp_path, cameras=True), monkeypatch,
                                       max_seq=REPRO_MAX_SEQ)
    said = " ".join(f.message for f in findings)
    assert f"나머지 {400 - (REPRO_MAX_SEQ + 1)}건은 확인하지" in said and "카메라" not in said, said


def test_the_command_line_leaves_the_bound_to_the_distribution(tmp_path, monkeypatch, capsys):
    # main() -> run() -> C9 without --repro-max-seq: a default of REPRO_MAX_SEQ anywhere on
    # that path would hide the camera bound from every real run.
    calls = []

    def fake_run(argv, env, timeout_s):
        calls.append(argv)
        return 0, "", ""

    monkeypatch.setattr(vd, "run_hook", fake_run)
    root = _distribution(tmp_path, cameras=True)
    vd.main([str(root), "--checks", "C9", "--repro-hook", "/bin/true", "--verbose"])
    out = capsys.readouterr().out
    assert calls and max(int(c[1].split("_")[1]) for c in calls) <= REPRO_MAX_SEQ_CAMERAS, calls
    assert f"나머지 {400 - (REPRO_MAX_SEQ_CAMERAS + 1)}건" in out, out


def test_the_camera_bound_fits_the_hook_timeout_at_the_measured_cost():
    """The bound is derived, not picked: the slowest prefix it allows fits the share of the
    hook's time, and one more episode would not.

    The cost has to cover the slowest 1,000-step camera episode measured with the seven RGB cameras
    (2026-10-22, 42 rows of S1e, S1f and S1g): 176.4 s of stepping and 14.0 s of save
    -- written here because no file a test reads holds it. Lowering the cost below it to widen the range
    makes this red."""
    episodes_rerun = REPRO_MAX_SEQ_CAMERAS + 1          # the run is rerun from its start to the episode
    budget = REPRO_TIMEOUT_SHARE * REPRO_HOOK_TIMEOUT_S
    assert episodes_rerun * REPRO_CAMERA_EPISODE_S + REPRO_OVERHEAD_S <= budget
    assert (episodes_rerun + 1) * REPRO_CAMERA_EPISODE_S + REPRO_OVERHEAD_S > budget
    assert REPRO_CAMERA_EPISODE_S >= 176.4 + 14.0
    assert 0 < REPRO_TIMEOUT_SHARE < 1 and REPRO_OVERHEAD_S > 0
    assert REPRO_MAX_SEQ_CAMERAS >= 0, "카메라 배포본에서 표본을 하나도 못 고릅니다"


def test_the_slowest_measured_row_still_fits_the_hook_timeout():
    """The share of the hook's time left over is what absorbs a run slower than the rows measured
    (2026-10-22, 42 full-length rows of the three directed scenarios): median 158.7 s, the slowest
    190.4 s (176.4 s of stepping and 14.0 s of save). A prefix as long as the bound allows, every
    episode as slow as that row, still has to end before the hook is cut off.
    Widening the bound (a lower cost or a larger share) past what this allows makes this red."""
    slowest_measured_row_s = 176.4 + 14.0
    assert (REPRO_MAX_SEQ_CAMERAS + 1) * slowest_measured_row_s + REPRO_OVERHEAD_S <= REPRO_HOOK_TIMEOUT_S


def test_the_hook_runs_with_the_timeout_the_bound_was_derived_for(tmp_path, monkeypatch):
    _, _, timeouts = run_c9_by_default(_distribution(tmp_path, cameras=True), monkeypatch)
    assert timeouts and set(timeouts) == {REPRO_HOOK_TIMEOUT_S}, timeouts


# ── What C9 says: the comparison's own words, not the simulator's log ──
#
# The hook's stdout is compare_published's output; its stderr always starts with the published
# root and then holds the simulator's log. Preferring stderr hid every comparison failure.

# What compare_published prints for a camera distribution -- the column count derived from the field
# list (the unconditional arrays, the box and a pose per camera), not written down; no pack phrase,
# since the camera canon declares none.
COMPARED = (f"재현 확인: S1d_0002_seed820211214 · {len(comparable_step_fields((CAMERAS,)))} 컬럼 · 999 프레임 · "
            f"{NOT_COMPARED}")
HOOK_STDERR = "대조 대상 배포본: /data/S1d-p4\n[Kit] shutting down\n[Kit] exit 0"


def _c9_finding(monkeypatch, returncode, stdout, stderr):
    def fake_run(argv, env, timeout_s):
        return returncode, stdout, stderr

    monkeypatch.setattr(vd, "run_hook", fake_run)
    rep = Report(Path("."))
    ds = type("DS", (), {"root": Path(".")})()
    eps = pd.DataFrame([{"episode_id": "S1d_0002_seed820211214", "randomization_seed": 820211214}])
    c9_seed_repro(ds, eps, "/bin/true", 1, rep, 20)
    (finding,) = [f for f in rep.findings if f.check == "C9"]
    return finding


def test_a_reproduced_episode_is_reported_with_what_was_compared(monkeypatch):
    f = _c9_finding(monkeypatch, 0, f"some earlier line\n{COMPARED}\n", HOOK_STDERR)
    assert (f.severity, f.message) == ("INFO", f"시드 재현 확인 — {COMPARED}")


def test_a_hook_that_says_nothing_is_still_reported_as_reproduced(monkeypatch):
    # A hook of one's own (README 「C9 훅 규약」) owes only its exit code.
    f = _c9_finding(monkeypatch, 0, "", HOOK_STDERR)
    assert (f.severity, f.message) == ("INFO", "시드 재현 확인")


def test_a_failed_comparison_is_reported_by_its_reason_then_the_hook_s_stderr(monkeypatch):
    reason = "재현 불일치 1 건:\n  팩 head.depth: 값 불일치 (depth)"
    f = _c9_finding(monkeypatch, 1, reason + "\n", HOOK_STDERR)
    assert f.severity == "ERROR"
    assert f.message == (f"재현 실패 (exit 1): …{reason} (훅 stderr 끝: …{HOOK_STDERR})"), f.message


def test_a_hook_that_died_before_comparing_is_reported_by_its_stderr(monkeypatch):
    # The simulator failed, set -e stopped the hook, the comparison never ran: nothing on stdout.
    err = HOOK_STDERR + "\nTraceback (most recent call last):\nRuntimeError: boom"
    f = _c9_finding(monkeypatch, 1, "", err)
    assert (f.severity, f.message) == ("ERROR", f"재현 실패 (exit 1): …{err}")


def test_long_output_is_cut_to_its_tail_not_its_head(monkeypatch):
    # The head of a real run is the simulator's boot log; the reason is at the end.
    reason = "팩 head.mask: 값 불일치 (mask)"
    f = _c9_finding(monkeypatch, 1, "boot\n" * 500 + reason, "log\n" * 500 + "END")
    assert reason in f.message and f.message.endswith("END)"), f.message[-80:]
    assert f.message.count("boot") < 100 and f.message.count("log") < 60, len(f.message)


def test_the_hook_is_told_which_dataset_to_compare_against(tmp_path, monkeypatch):
    """훅이 자기 기본값으로 떨어지면 C9 는 다른 데이터를 대조하고 초록이 됩니다.

    실제로 400건 배치를 검증하면서 훅이 10건짜리 옛 배포본과 대조했습니다
    (2026-08-25). 그때는 빨간불이었지만, 우연히 일치했다면 **검증 중이 아닌
    데이터셋에 대해 재현성을 통과**했다고 보고했을 것입니다.
    """
    import pandas as pd

    seen = {}

    def fake_run(argv, env, timeout_s):
        seen["env"] = env or {}
        return 0, "", ""

    monkeypatch.setattr(vd, "run_hook", fake_run)

    ds = type("DS", (), {"root": tmp_path})()
    eps = pd.DataFrame([{"episode_id": "S1a_0000_seed7", "randomization_seed": 7}])
    rep = vd.Report(tmp_path)
    vd.c9_seed_repro(ds, eps, "/bin/true", 1, rep, 20)

    assert seen["env"].get("C9_PUBLISHED_ROOT") == str(tmp_path.resolve()), (
        "C9 가 훅에게 검증 중인 데이터셋 경로를 넘기지 않습니다")


def test_the_hook_passes_the_published_root_to_the_comparison(tmp_path):
    """훅이 변수를 **찍는 것**과 **비교에 넘기는 것**은 다릅니다.

    앞 시험은 잘못된 episode_id 를 줘서 시드 파싱에서 죽습니다. 그 지점은 sentinel 을
    출력한 **뒤**이자 비교를 부르기 **전**이라, `--published` 를 옛 기본 경로로
    하드코딩해도 초록이었습니다. 여기서는 두 인터프리터를
    argv 를 받아 적는 가짜로 바꿔, 훅이 실제로 만드는 명령줄을 봅니다.
    """
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    hook = repo / "generation" / "rerun_seed.sh"
    log = tmp_path / "argv.jsonl"
    fake = tmp_path / "fake-python"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(log)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8")
    fake.chmod(0o755)

    sentinel = str(tmp_path / "published-root")
    r = subprocess.run([str(hook), "S1a_0000_seed7", "7"],
                       capture_output=True, text=True, timeout=180,
                       env={**os.environ, "C9_PUBLISHED_ROOT": sentinel,
                            # 시드 파싱은 가짜로 바꾸지 않는다 -- 그 값이 재생성
                            # 구간의 길이를 정하므로 진짜로 풀려야 한다. 다만
                            # `.venv` 가 있는 기계를 가정하면 CI 에서 이 시험 자체가
                            # exit 127 로 죽는다. 돌고 있는 인터프리터를 준다.
                            "C9_SEQ_PYTHON": sys.executable,
                            "C9_ISAAC_PYTHON": str(fake),
                            "C9_COMPARE_PYTHON": str(fake)})
    assert r.returncode == 0, f"가짜 실행기로도 훅이 실패했습니다:\n{r.stderr[-500:]}"
    calls = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]
    compare = [c for c in calls if any("compare_published.py" in x for x in c)]
    assert compare, f"훅이 비교를 부르지 않았습니다: {calls}"
    argv = compare[-1]
    assert "--published" in argv, argv
    assert argv[argv.index("--published") + 1] == sentinel, (
        f"훅이 검증 중인 배포본이 아니라 {argv[argv.index('--published') + 1]} 와 "
        "대조합니다 — C9 가 다른 데이터에 대해 초록이 됩니다")


@pytest.mark.parametrize("episode_id, scenario, episodes", [
    ("S1a_0003_seed7", "S1a", 4), ("S1d_0003_seed7", "S1d", 4), ("S1d_0000_seed9", "S1d", 1)])
def test_the_hook_regenerates_the_episode_s_own_scenario(tmp_path, episode_id, scenario,
                                                          episodes):
    """run_s1a.py 의 기본 시나리오는 S1a 입니다. 훅이 `--scenario` 를 넘기지 않던 동안 S1d
    배포본의 C9 는 S1a 를 다시 돌려 「재생성 결과가 다른 에피소드입니다」로 끝났습니다.
    시나리오와 접두 길이 둘 다 id 에서 옵니다."""
    import json
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    log = tmp_path / "argv.jsonl"
    fake = tmp_path / "fake-python"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(log)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8")
    fake.chmod(0o755)
    r = subprocess.run([str(repo / "generation" / "rerun_seed.sh"), episode_id, "7"],
                       capture_output=True, text=True, timeout=180,
                       env={**os.environ, "C9_PUBLISHED_ROOT": str(tmp_path / "published"),
                            "C9_SEQ_PYTHON": sys.executable,
                            "C9_ISAAC_PYTHON": str(fake), "C9_COMPARE_PYTHON": str(fake)})
    assert r.returncode == 0, r.stderr[-500:]
    calls = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]
    (argv,) = [c for c in calls if any(x.endswith("run_s1a.py") for x in c)]
    assert argv[argv.index("--scenario") + 1] == scenario, argv
    assert argv[argv.index("--episodes") + 1] == str(episodes), argv
    assert argv[argv.index("--seq") + 1] == "0", argv


def _hook_argv(tmp_path, *hook_args, repo=None):
    """rerun_seed.sh 를 가짜 인터프리터로 돌려, 그것이 만든 명령줄들을 돌려준다. ``repo``: 훅을 꺼낼 저장소(기본은 이 저장소)."""
    import json
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2] if repo is None else Path(repo)
    log = tmp_path / "argv.jsonl"
    fake = tmp_path / "fake-python"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(log)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8")
    fake.chmod(0o755)
    r = subprocess.run([str(repo / "generation" / "rerun_seed.sh"), *hook_args],
                       capture_output=True, text=True, timeout=180,
                       env={**os.environ, "C9_PUBLISHED_ROOT": str(tmp_path / "published"),
                            "C9_SEQ_PYTHON": sys.executable,
                            "C9_ISAAC_PYTHON": str(fake), "C9_COMPARE_PYTHON": str(fake)})
    calls = ([json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]
             if log.exists() else [])
    return r, calls


@pytest.mark.parametrize("episode_id, start, seq, episodes", [
    ("S1d_0013_seed7", "10", "10", "4"),     # the fourth episode of the run that began at 10
    ("S1d_0010_seed7", "10", "10", "1"),     # the first of its run
    ("S1d_0399_seed7", "390", "390", "10"),
    ("S1d_0003_seed7", "0", "0", "4"),
])
def test_the_hook_reruns_the_run_from_its_first_seq(tmp_path, episode_id, start, seq, episodes):
    r, calls = _hook_argv(tmp_path, episode_id, "7", start)
    assert r.returncode == 0, r.stderr[-500:]
    (isaac,) = [c for c in calls if any(x.endswith("run_s1a.py") for x in c)]
    assert isaac[isaac.index("--seq") + 1] == seq, isaac
    assert isaac[isaac.index("--episodes") + 1] == episodes, isaac
    (compare,) = [c for c in calls if any(x.endswith("compare_published.py") for x in c)]
    assert compare[compare.index("--run-start-seq") + 1] == start, compare


@pytest.mark.parametrize("start", ["5", "-1", "abc", "1.0", ""], ids=["past the episode", "negative", "text", "fraction", "empty"])
def test_the_hook_refuses_a_start_the_episode_could_not_have(tmp_path, start):
    r, calls = _hook_argv(tmp_path, "S1d_0003_seed7", "7", start)
    assert r.returncode != 0
    assert not [c for c in calls if any(x.endswith("run_s1a.py") for x in c)], "거부하기 전에 시뮬레이터를 띄웠습니다"
    assert "run_start_seq" in r.stderr, r.stderr[-300:]


def _publish_policies(tmp_path, rows):
    """meta/episodes of the distribution the hook compares against (``tmp_path / "published"``, as _hook_argv
    hands it over): each row's sim_backend as JSON text, with ``policy`` when given."""
    from generation.convert_lerobot import episode_column

    root = tmp_path / "published"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    backend = {"physics": "physx", "generator_commit": "a" * 40, "generator_dirty": False}
    pd.DataFrame([{"episode_id": eid, "sim_backend": episode_column(dict(backend, **({"policy": p} if p else {})))}
                  for eid, p in rows]).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return root


def _table_scenario():
    from generation import conditions

    return conditions.table_scenarios()[0]


def _isaac_argv(calls):
    (isaac,) = [c for c in calls if any(x.endswith("run_s1a.py") for x in c)]
    return isaac


@pytest.mark.parametrize("source", ["published", "repository file"])
def test_the_hook_passes_the_recorded_policy_for_a_table_scenario(tmp_path, source):
    import hashlib

    from generation import behavior
    from tests.generation.committed_repo import committed_copy

    # The hook and the files the rerun reads, as the working tree holds them, committed in a checkout of their own:
    # the rerun takes only committed files, and this test is not about what is uncommitted here.
    repo = committed_copy(tmp_path / "repo", "generation", "meta", "LICENSE")
    if source == "published":
        policy = {"source": f"isaaclab-published:rsl_rl/{behavior.PUBLISHED_POLICY_TASK}", "sha256": "0" * 64,
                  "behavior": behavior.DEFAULT_BEHAVIOR}
        tail = ["--behavior", behavior.DEFAULT_BEHAVIOR]
    else:                                   # a checkpoint the repository holds: any of its files will do
        policy = {"source": "file:LICENSE", "sha256": hashlib.sha256((repo / "LICENSE").read_bytes()).hexdigest(),
                  "behavior": behavior.DEFAULT_BEHAVIOR}
        tail = ["--policy", str(repo / "LICENSE"), "--behavior", behavior.DEFAULT_BEHAVIOR]
    episode = f"{_table_scenario()}_0003_seed7"
    _publish_policies(tmp_path, [(episode, policy)])
    r, calls = _hook_argv(tmp_path, episode, "7", "3", repo=repo)
    assert r.returncode == 0, r.stderr[-500:]
    isaac = _isaac_argv(calls)
    assert isaac[0] == str(repo / "generation" / "envs" / "run_s1a.py"), isaac      # the copy's, not this one's
    assert isaac[-len(tail):] == tail, isaac
    assert isaac[isaac.index("--out") + 2:] == tail, isaac            # after everything the hook passed before
    assert isaac[isaac.index("--seq") + 1] == "3" and isaac[isaac.index("--episodes") + 1] == "1", isaac


def test_the_hook_passes_no_policy_for_a_scenario_without_a_table(tmp_path):
    # S1d's rerun takes no --policy or --behavior, whatever the distribution's rows say.
    _publish_policies(tmp_path, [("S1d_0003_seed7", {"source": "file:/nowhere.pt", "sha256": "0" * 64,
                                                     "behavior": "mod:Name"})])
    r, calls = _hook_argv(tmp_path, "S1d_0003_seed7", "7", "0")
    assert r.returncode == 0, r.stderr[-500:]
    isaac = _isaac_argv(calls)
    assert isaac[-2] == "--out", isaac
    assert "--policy" not in isaac and "--behavior" not in isaac, isaac


@pytest.mark.parametrize("rows", [
    pytest.param("another task", id="another published task"),
    pytest.param("absent file", id="a file the repository does not have"),
    pytest.param("installed module", id="a behaviour module not of the repository"),
    pytest.param("no table", id="no episode table"),
])
def test_the_hook_stops_before_isaac_on_an_unrecoverable_policy(tmp_path, rows):
    from generation import behavior

    episode = f"{_table_scenario()}_0003_seed7"
    spec, said = behavior.DEFAULT_BEHAVIOR, None
    if rows == "another task":
        source = said = "isaaclab-published:rsl_rl/Isaac-Velocity-Rough-H1-v0"
    elif rows == "absent file":
        source = said = "file:no/such/keti-policy.pt"
    elif rows == "installed module":
        source = f"isaaclab-published:rsl_rl/{behavior.PUBLISHED_POLICY_TASK}"
        spec, said = "os:system", "저장소의 파일이 아닙니다"
    if rows != "no table":
        _publish_policies(tmp_path, [(episode, {"source": source, "sha256": "0" * 64, "behavior": spec})])
    r, calls = _hook_argv(tmp_path, episode, "7", "3")
    assert r.returncode == 2, (r.returncode, r.stderr[-500:])
    assert calls == [], f"정책을 되찾지 못했는데 시뮬레이터나 비교기를 불렀습니다: {calls}"
    if said:
        assert said in r.stderr, r.stderr[-500:]


def test_the_hook_actually_consumes_the_variable():
    """검증기가 넘기는 것과 훅이 읽는 것은 다릅니다.

    앞 시험은 `run_hook` 을 대역으로 바꿔 넘겨받은 `env` 만 봅니다. 그래서
    `rerun_seed.sh` 에서 `C9_PUBLISHED_ROOT` 사용을 지워도 초록이었고, 그러면 C9 는
    다시 기본 경로의 옛 배포본과 대조하면서 초록이 됩니다.

    훅을 실제로 실행합니다. 잘못된 episode_id 를 주면 시드 계산에서 죽으므로
    시뮬레이터를 띄우지 않고, 그 전에 훅이 어느 배포본을 대조하는지 말합니다.
    """
    import os
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    hook = repo / "generation" / "rerun_seed.sh"
    sentinel = "/tmp/keti-c9-sentinel-root"
    r = subprocess.run([str(hook), "not-an-episode-id", "0"],
                       capture_output=True, text=True, timeout=120,
                       env={**os.environ, "C9_PUBLISHED_ROOT": sentinel})
    assert r.returncode != 0, "잘못된 episode_id 인데 훅이 성공했습니다"
    assert sentinel in (r.stdout + r.stderr), (
        "훅이 C9_PUBLISHED_ROOT 를 쓰지 않습니다 — 검증기가 넘겨도 소용이 없고, "
        "C9 는 검증 중이 아닌 데이터셋과 대조하게 됩니다")


def test_the_episode_id_reaches_python_as_data_not_as_code(tmp_path):
    """episode_id 는 검증 중인 배포본(meta/episodes)에서 옵니다. 하나도 형식에 맞지 않으면
    C9 는 그 id 를 전부 그대로 훅에 넘깁니다(c9_seed_repro 의 `eligible = sims`).

    훅이 그 값을 `python -c` 소스 안에 '$EPISODE_ID' 로 끼워 넣던 동안, 따옴표 하나로 id 가
    코드가 되었습니다 — 아래 id 는 파일을 만들고 나서야 형식 오류로 죽었습니다 (2026-09-28).
    """
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    hook = repo / "generation" / "rerun_seed.sh"
    marker = tmp_path / "ran"
    episode_id = f"x') if open({str(marker)!r}, 'w') else print('"
    r = subprocess.run([str(hook), episode_id, "0"],
                       capture_output=True, text=True, timeout=120,
                       env={**os.environ, "C9_PUBLISHED_ROOT": str(tmp_path / "published"),
                            "C9_SEQ_PYTHON": sys.executable,
                            "C9_ISAAC_PYTHON": "/bin/false", "C9_COMPARE_PYTHON": "/bin/false"})
    assert r.returncode != 0, "형식에 맞지 않는 episode_id 인데 훅이 성공했습니다"
    assert "episode_id 형식이 아닙니다" in r.stderr, r.stderr[-500:]
    assert not marker.exists(), (
        "episode_id 가 파이썬 코드로 실행되었습니다 — 배포본의 문자열이 검증하는 사람의 "
        "기계에서 돕니다")


def test_an_episode_id_that_fits_the_pattern_reaches_both_lookups_as_data(tmp_path):
    """앞 시험의 id 는 형식에 맞지 않아 훅이 첫 조회(SEQ)에서 죽고, 둘째
    조회(SCENARIO)에는 닿지 않습니다 — 그 줄만 id 를 소스에 끼워 넣는 방식('$EPISODE_ID')으로 되돌려도 초록이었습니다.
    이 id 는 형식에 맞습니다: 시나리오 자리 `[^_]+` 에 밑줄이 없어야 하므로
    표식은 상대 경로 'ran' 이고 훅을 tmp_path 에서 돌립니다(tmp_path 에는 밑줄이 있습니다).
    두 조회를 다 지나 Isaac 자리에 닿고, 거기 넘어간 --scenario 는 문자열 그대로여야 합니다."""
    import json
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    scenario = "x') if open('ran','w') else print('x"
    episode_id = f"{scenario}_0000_seed1"
    log = tmp_path / "isaac-argv.jsonl"
    stub = tmp_path / "fake-isaac"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(log)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8")
    stub.chmod(0o755)
    r = subprocess.run([str(repo / "generation" / "rerun_seed.sh"), episode_id, "1"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=120,
                       env={**os.environ, "C9_PUBLISHED_ROOT": str(tmp_path / "published"),
                            "C9_SEQ_PYTHON": sys.executable,
                            "C9_ISAAC_PYTHON": str(stub), "C9_COMPARE_PYTHON": "/bin/false"})
    assert not (tmp_path / "ran").exists(), (
        "형식에 맞는 episode_id 가 파이썬 코드로 실행되었습니다 — 훅의 두 조회 중 하나가 id 를 "
        "소스에 끼워 넣습니다")
    assert r.returncode != 0, "비교기 자리가 /bin/false 인데 훅이 성공했습니다"
    (argv,) = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]
    assert argv[argv.index("--scenario") + 1] == scenario, argv
    assert argv[argv.index("--episodes") + 1] == "1", argv


def test_the_validator_does_not_forward_the_test_only_overrides(monkeypatch):
    """훅의 인터프리터 주입은 시험용 문입니다. 생산 경로에서 열려 있으면 안 됩니다.

    `C9_ISAAC_PYTHON=/bin/true` 하나로 시뮬레이터도 비교기도 돌지 않은 채 C9 가
    통과했습니다.
    """
    import pandas as pd

    seen = {}

    def fake_run(argv, env, timeout_s):
        seen["env"] = env or {}
        return 0, "", ""

    # 훅이 여는 문 **전부** 를 넣는다. 하나만 넣으면 나중에 문이 늘어도 이 시험은
    # 초록이고, 실제로 `C9_SEQ_PYTHON` 이 그렇게 늘었다.
    injected = ["C9_ISAAC_PYTHON", "C9_COMPARE_PYTHON", "C9_SEQ_PYTHON"]
    for name in injected:
        os.environ[name] = "/bin/true"
    try:
        monkeypatch.setattr(vd, "run_hook", fake_run)
        ds = type("DS", (), {"root": Path(".")})()
        eps = pd.DataFrame([{"episode_id": "S1a_0000_seed7", "randomization_seed": 7}])
        vd.c9_seed_repro(ds, eps, "/bin/true", 1, vd.Report(Path(".")), 20)
    finally:
        for name in injected:
            os.environ.pop(name, None)

    leaked = [name for name in injected if name in seen["env"]]
    assert not leaked, (
        f"검증기가 시험용 인터프리터 주입을 훅에 그대로 넘깁니다: {leaked} — "
        "아무것도 실행하지 않고 재현 확인이 나옵니다")
    assert seen["env"].get("C9_PUBLISHED_ROOT"), "대조 대상은 계속 넘겨야 합니다"
    assert [k for k in seen["env"] if k.startswith("C9_")] == ["C9_PUBLISHED_ROOT"], (
        "훅에 넘어가는 C9_ 변수는 대조 대상 하나뿐이어야 합니다 — "
        f"실제로 넘어간 것: {sorted(k for k in seen['env'] if k.startswith('C9_'))}")


def test_zero_samples_is_an_error_not_a_skip():
    """SKIP 은 exit 0 입니다 — 검증기 혼자서도 막아야 합니다.

    SKIP 으로 두면 publish 게이트만 막고, 검증기를 직접 부르면
    `--repro-sample 0` 이 그대로 통과합니다.
    """
    import pandas as pd

    import validator.validate_dataset as vd

    rep = vd.Report(Path("."))
    ds = type("DS", (), {"root": Path(".")})()
    eps = pd.DataFrame([{"episode_id": "S1a_0000_seed7", "randomization_seed": 7,
                         "source": "sim"}])
    vd.c9_seed_repro(ds, eps, "/bin/true", 0, rep, 20)
    assert rep.count(vd.ERROR) == 1, [f.message for f in rep.findings]


def test_checks_not_run_are_not_reported_as_ok(capsys):
    """돌리지 않은 검사를 OK 로 찍으면 안 됩니다.

    `--checks C1` 하나만 줘도 C2~C9 가 전부 초록이었고 exit 0 이었습니다.
    """
    import validator.validate_dataset as vd

    rep = vd.Report(Path("."))
    rep.add("C1", vd.INFO, "무언가")
    vd.print_report(rep, verbose=False, ran={"C1"})
    out = capsys.readouterr().out
    unrun = [ln for ln in out.splitlines() if any(c in ln for c in
             ("C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"))]
    assert unrun and all("미실행" in ln for ln in unrun), out
    assert not any("OK" in ln for ln in unrun), out


def test_the_hook_gets_a_fixed_path(monkeypatch):
    """훅의 `#!/usr/bin/env bash` 는 호출자 PATH 에서 bash 를 찾습니다.

    거기 가짜 bash 를 두면 정확한 저장소 훅이 아무것도 하지 않고 exit 0 이 됩니다.
    악의를 막는 장치는 아닙니다 — 이상한 환경에서
    조용히 초록이 되는 사고를 막습니다.
    """
    import pandas as pd

    seen = {}

    def fake_run(argv, env, timeout_s):
        seen["env"] = env or {}
        return 0, "", ""

    monkeypatch.setenv("PATH", "/tmp/evil:" + os.environ["PATH"])
    monkeypatch.setattr(vd, "run_hook", fake_run)
    ds = type("DS", (), {"root": Path(".")})()
    eps = pd.DataFrame([{"episode_id": "S1a_0000_seed7", "randomization_seed": 7}])
    vd.c9_seed_repro(ds, eps, "/bin/true", 1, vd.Report(Path(".")), 20)

    assert "/tmp/evil" not in seen["env"].get("PATH", ""), (
        "검증기가 호출자 PATH 를 훅에 그대로 넘깁니다")
    assert seen["env"].get("PATH") == vd._SAFE_PATH, seen["env"].get("PATH")


def test_the_json_report_says_which_checks_did_not_run():
    """텍스트만 「미실행」을 말하면 증빙으로 JSON 을 읽는 쪽은 못 구별합니다.

    검증 결과로 이 JSON 을 인용하는 쪽이 있으므로, 부분 실행과 전체 통과가 같아 보이면
    안 됩니다.
    """
    import validator.validate_dataset as vd

    rep = vd.Report(Path("."))
    rep.requested = {"C1", "C5"}
    rep.add("C1", vd.WARN, "무언가")
    d = rep.to_dict()
    assert d["requested_checks"] == ["C1", "C5"]
    assert d["check_status"]["C1"] == "warn"
    assert d["check_status"]["C5"] == "ok"
    assert d["check_status"]["C9"] == "not_run"


def test_an_aborted_run_does_not_claim_every_check_passed():
    """검사를 하나도 돌리지 않은 실행이 전 검사 통과로 기록되면 안 됩니다.

    없는 root 는 `requested` 를 세우기 전에 C0 로 돌아왔고, 빈 집합이 「전부 요청」
    으로 읽혀 C1~C9 가 모두 `ok` 였습니다.
    """
    import validator.validate_dataset as vd

    rep = vd.Report(Path("."))
    rep.add("C0", vd.ERROR, "데이터셋 루트가 디렉터리가 아님")
    d = rep.to_dict()
    assert d["requested_checks"] is None
    assert set(d["check_status"].values()) == {"unknown"}, d["check_status"]


def test_the_real_run_records_what_it_requested(tmp_path):
    """`Report.requested` 를 시험이 직접 세우면 production 경로를 안 봅니다."""
    import json
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    fixture = repo / ".tmp" / "fixtures" / "clean"
    if not fixture.is_dir():
        import pytest
        pytest.skip("clean 픽스처가 없습니다")
    out = tmp_path / "r.json"
    subprocess.run([sys.executable, str(repo / "validator/validate_dataset.py"),
                    str(fixture), "--checks", "C1,C5", "--json", str(out)],
                   capture_output=True, text=True, cwd=repo)
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["requested_checks"] == ["C1", "C5"]
    assert d["check_status"]["C9"] == "not_run"


# ── A hook that runs out of time is ended with everything it started ──
#
# subprocess.run(timeout=) kills only the direct child. The hook is bash and the simulator is its
# child, so a timed-out hook would leave the simulator running -- sharing the GPU with the next sample --
# and its EXIT trap would never run, which leaves the work directory behind (about 1.4-1.5 GB per seven-camera
# episode, up to six of them). So the hook runs in its own process group and the group is ended.

needs_proc = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="reads /proc/<pid>/stat")


def _running(pid):
    """Whether the process is alive: it exists and is not a zombie. A killed child whose parent died
    with it lingers as a zombie until init collects it -- gone, for this purpose."""
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (ProcessLookupError, FileNotFoundError):
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _wait_until_gone(pid, seconds=3.0):
    """Death is asynchronous to the signal, so poll briefly instead of asking once."""
    deadline = time.monotonic() + seconds
    while _running(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    return not _running(pid)


def _pid_or_none(pidfile):
    """The pid a hook recorded, or None while it has not (a failure before the hook got that far)."""
    try:
        return int(pidfile.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _kill_leftover(pid):
    """A test that fails must not leave a `sleep 600` behind. Read the pid in the `finally`, so a
    failure before the test's own read still finds the child."""
    if pid is not None and _running(pid):
        os.kill(pid, signal.SIGKILL)


def _long_running_hook(tmp_path, *, ignore_term=False):
    """A stand-in hook shaped like the real one: a bash script that holds a work directory (removed by
    its EXIT trap), starts a long-running child and waits for it. It records the child's pid and the
    directory, and writes a line to stdout and one to stderr before it waits. With ``ignore_term`` it
    and its child ignore SIGTERM -- only SIGKILL ends them."""
    pidfile, dirfile = tmp_path / "child.pid", tmp_path / "work.path"
    hook = tmp_path / "hook"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        f'WORK="$(mktemp -d {shlex.quote(str(tmp_path))}/work.XXXXXX)"\n'
        "trap 'rm -rf \"$WORK\"' EXIT\n"
        + ("trap '' TERM\n" if ignore_term else "trap 'exit 143' TERM\n")
        + "echo partial output\n"
        "echo stuck in step three >&2\n"
        "sleep 600 &\n"
        f'echo "$!" > {shlex.quote(str(pidfile))}\n'
        f'echo "$WORK" > {shlex.quote(str(dirfile))}\n'
        "wait\n", encoding="utf-8")
    hook.chmod(0o755)
    return hook, pidfile, dirfile


class _HangGuard:
    """Frees a test that a broken group kill would leave hanging. run_hook waits for the pipes its
    hook's child holds, so a group that is not ended keeps it blocked for that child's 600 s. After
    ``seconds`` this kills the recorded child instead, and ``fired`` says it had to -- the test then
    fails on that, instead of waiting ten minutes and passing on a child that ended by itself."""

    def __init__(self, pidfile, seconds=10.0):
        self.fired = False
        self._pidfile = pidfile
        self._timer = threading.Timer(seconds, self._fire)
        self._timer.daemon = True

    def _fire(self):
        self.fired = True
        pid = _pid_or_none(self._pidfile)
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    def __enter__(self):
        self._timer.start()
        return self

    def __exit__(self, *exc):
        self._timer.cancel()


def _read_pid(pidfile):
    assert pidfile.exists(), "훅이 자식을 띄우기 전에 시간이 다 되었습니다 — 시험이 무엇도 보지 못했습니다"
    return _pid_or_none(pidfile)


def _signal_self_once_started(marker, sig, delay=0.2):
    """Send this process ``sig`` from a helper thread once the hook has written ``marker`` (so it is
    really running), ``delay`` seconds later; Python runs the handler in the main thread, where
    run_hook waits. Returns the function that cancels it -- call it before the handler is restored,
    so a signal cannot arrive later into whatever runs next."""
    stop = threading.Event()

    def send():
        deadline = time.monotonic() + 60
        while not marker.exists() and time.monotonic() < deadline:
            if stop.wait(0.01):
                return
        if not stop.wait(delay):
            os.kill(os.getpid(), sig)

    thread = threading.Thread(target=send, daemon=True)
    thread.start()

    def cancel():
        stop.set()
        thread.join(5)

    return cancel


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


@needs_proc
def test_a_hook_that_runs_out_of_time_is_ended_with_everything_it_started(tmp_path, monkeypatch):
    hook, pidfile, dirfile = _long_running_hook(tmp_path)
    monkeypatch.setattr(vd, "REPRO_HOOK_TIMEOUT_S", 2)      # room for the hook to record its child
    monkeypatch.setattr(vd, "REPRO_KILL_GRACE_S", 1)
    try:
        rep = Report(tmp_path)
        ds = type("DS", (), {"root": tmp_path})()
        eps = pd.DataFrame([{"episode_id": "S1d_0000_seed7", "randomization_seed": 7}])
        started = time.monotonic()
        with _HangGuard(pidfile) as guard:
            c9_seed_repro(ds, eps, str(hook), 1, rep, 20)
        elapsed = time.monotonic() - started
        pid = _read_pid(pidfile)
        (finding,) = [f for f in rep.findings if f.check == "C9"]
        work = Path(dirfile.read_text(encoding="utf-8").strip())

        problems = []
        if finding.severity != "ERROR" or "시간 초과(2 s) — 훅의 프로세스 그룹을 끝냄" not in finding.message:
            problems.append(f"C9 가 시간 초과를 오류로 말하지 않음: {finding.severity} {finding.message!r}")
        elif "partial output" not in finding.message or "stuck in step three" not in finding.message:
            # Where the hook was stuck: the tails of what it wrote, after the fixed sentence.
            problems.append(f"시간 초과 보고가 훅이 어디서 멈췄는지 싣지 않음: {finding.message!r}")
        if not _wait_until_gone(pid):
            problems.append(f"훅이 띄운 자식 {pid} 이 계속 돎")
        if work.exists():
            problems.append(f"훅의 작업 디렉터리 {work.name} 이 남음 (EXIT 트랩이 돌지 않음)")
        # The normal path is the timeout (2 s) and a prompt exit on SIGTERM, within the grace (1 s). 8 s leaves a
        # slow runner room and stays under _HangGuard's 10 s, where the guard takes over.
        if guard.fired or elapsed >= 8:
            problems.append(f"{elapsed:.1f} 초 걸림 — 훅의 그룹이 끝나지 않아 시험이 자식을 죽여야 했음"
                            if guard.fired else f"{elapsed:.1f} 초 걸림")
        assert not problems, "; ".join(problems)
    finally:
        _kill_leftover(_pid_or_none(pidfile))


@needs_proc
def test_a_hook_that_ignores_sigterm_is_killed_once_the_grace_is_over(tmp_path, monkeypatch):
    hook, pidfile, _ = _long_running_hook(tmp_path, ignore_term=True)
    timeout, grace = 1.5, 0.5
    monkeypatch.setattr(vd, "REPRO_KILL_GRACE_S", grace)
    try:
        started = time.monotonic()
        with _HangGuard(pidfile) as guard:
            code, out, err = vd.run_hook([str(hook)], dict(os.environ), timeout)
        elapsed = time.monotonic() - started
        pid = _read_pid(pidfile)
        assert not guard.fired, "SIGKILL 이 훅의 그룹에 닿지 않아 run_hook 이 돌아오지 못했음"
        assert code is None, code
        # It could only have ended by SIGKILL: SIGTERM is ignored, so the grace ran out first.
        assert elapsed >= timeout + grace - 0.05, elapsed
        assert _wait_until_gone(pid), f"SIGTERM 을 무시하는 자식 {pid} 이 SIGKILL 뒤에도 돎"
        assert "partial output" in out and "stuck in step three" in err, (out, err)
    finally:
        _kill_leftover(_pid_or_none(pidfile))


@needs_proc
def test_a_killed_hook_leaves_nothing_under_the_temp_directory(tmp_path, monkeypatch):
    """A hook still ignoring SIGTERM when the grace is over dies of SIGKILL, and no trap of its runs: the
    repository hook's work directory (`mktemp -d`, about 1.4-1.5 GB per regenerated camera episode) would stay behind
    under TMPDIR. So the validator makes the hook's TMPDIR itself, under the caller's, and removes it once the
    hook has ended, however it ended."""
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    pidfile, dirfile = tmp_path / "child.pid", tmp_path / "work.path"
    hook = tmp_path / "hook"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        'WORK="$(mktemp -d)"\n'                              # as rerun_seed.sh makes its own: under TMPDIR
        "trap 'rm -rf \"$WORK\"' EXIT\n"
        "trap '' TERM\n"
        'echo rendered > "$WORK/episode.mkv"\n'
        f'echo "$WORK" > {shlex.quote(str(dirfile))}\n'
        "sleep 600 &\n"
        f'echo "$!" > {shlex.quote(str(pidfile))}\n'
        "wait\n", encoding="utf-8")
    hook.chmod(0o755)
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setattr(vd, "REPRO_KILL_GRACE_S", 0.5)
    try:
        with _HangGuard(pidfile) as guard:
            code, _, _ = vd.run_hook([str(hook)], dict(os.environ), 1.5)
        pid = _read_pid(pidfile)
        assert not guard.fired and code is None, code
        assert _wait_until_gone(pid), f"SIGTERM 을 무시하는 자식 {pid} 이 SIGKILL 뒤에도 돎"
        work = Path(dirfile.read_text(encoding="utf-8").strip())
        assert scratch in work.parents, f"훅의 작업 디렉터리 {work} 가 TMPDIR 아래에 있어야 시험이 뜻을 가집니다"
        assert list(scratch.iterdir()) == [], f"SIGKILL 로 끝난 훅이 TMPDIR 에 남김: {list(scratch.iterdir())}"
    finally:
        _kill_leftover(_pid_or_none(pidfile))


def _hook_leaving_a_member(tmp_path, *, exits_by_itself):
    """A hook whose group outlives it: it starts a member that lets go of stdout and stderr -- so the hook's output
    is complete once the hook itself has exited. The hook then waits for SIGTERM (its trap exits 143), and its
    member ignores SIGTERM and keeps writing under TMPDIR, making the directory again if it is gone; or, with
    ``exits_by_itself``, the hook exits 0 at once and its member is a daemon with SIGTERM at its default that
    only sleeps -- shaped like the Omniverse Hub Kit starts in the hook's group when none runs. Returns (hook,
    the member's pid file)."""
    pidfile = tmp_path / "member.pid"
    hook = tmp_path / "hook"
    member = ("( exec >/dev/null 2>&1; exec sleep 600 ) &\n" if exits_by_itself else
              "( trap '' TERM; exec >/dev/null 2>&1\n"
              '  while :; do mkdir -p "$TMPDIR/out" && touch "$TMPDIR/out/frame"; sleep 0.05; done ) &\n')
    hook.write_text(
        "#!/usr/bin/env bash\n"
        "trap 'exit 143' TERM\n"
        + member
        + f'echo "$!" > {shlex.quote(str(pidfile))}\n'
        "echo partial output\n"
        + ("exit 0\n" if exits_by_itself else "wait\n"), encoding="utf-8")
    hook.chmod(0o755)
    return hook, pidfile


@needs_proc
def test_a_member_left_in_the_hook_s_group_is_ended_before_its_temp_directory_goes(tmp_path, monkeypatch):
    """The hook ran out of time; on SIGTERM its leader exits and its pipes reach
    EOF, so collecting the output returns -- while a member of its group that ignores SIGTERM and has closed
    both pipes still runs. It was left running, and its TMPDIR removed under it: it made the directory again.
    The group's end is its members', not its leader's: SIGKILL for what is still there when the grace is over,
    before the temp directory is removed."""
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    hook, pidfile = _hook_leaving_a_member(tmp_path, exits_by_itself=False)
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setattr(vd, "REPRO_KILL_GRACE_S", 1.0)
    try:
        with _HangGuard(pidfile, 20) as guard:
            code, out, _ = vd.run_hook([str(hook)], dict(os.environ), 1.0)
        pid = _read_pid(pidfile)
        assert not guard.fired
        assert (code, out) == (None, "partial output\n"), (code, out)
        assert not _running(pid), f"훅의 그룹에 남은 {pid} 이 run_hook 이 돌아온 뒤에도 돎"
        time.sleep(0.3)                  # a member still running would have made its directory again by now
        assert list(scratch.iterdir()) == [], f"TMPDIR 에 남음: {list(scratch.iterdir())}"
    finally:
        _kill_leftover(_pid_or_none(pidfile))


@needs_proc
def test_what_a_hook_that_exits_by_itself_leaves_in_its_group_is_left_alone(tmp_path, monkeypatch):
    """A hook that returns an exit status has ended as it meant to:
    what its simulator started and left running is not the validator's to end. On the development machine Kit starts the Omniverse
    Hub (`hub --mode=shared`, SIGTERM at its default, pipes on /dev/null) in the hook's group whenever none runs,
    and a SIGTERM to the group would take down a shared cache another Isaac session may be using. Ending the group
    is for the timeout, signal and interrupt paths. The temp directory is removed all the same."""
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    hook, pidfile = _hook_leaving_a_member(tmp_path, exits_by_itself=True)
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setattr(vd, "REPRO_KILL_GRACE_S", 1.0)
    try:
        with _HangGuard(pidfile, 20) as guard:
            code, out, _ = vd.run_hook([str(hook)], dict(os.environ), 30)
        pid = _read_pid(pidfile)
        assert not guard.fired
        assert (code, out) == (0, "partial output\n"), (code, out)
        time.sleep(0.3)                  # a SIGTERM sent to the group would have ended the daemon by now
        assert _running(pid), f"정상 종료한 훅의 그룹에 남은 데몬 {pid} 을 검증기가 끝냄"
        assert list(scratch.iterdir()) == [], f"TMPDIR 에 남음: {list(scratch.iterdir())}"
    finally:
        _kill_leftover(_pid_or_none(pidfile))


@needs_proc
def test_a_hook_killed_by_a_signal_has_what_it_left_in_its_group_ended(tmp_path, monkeypatch):
    """A negative returncode -- bash killed by a signal: the OOM killer's
    SIGKILL, a segfault -- was taken for an exit status, and what the hook left in its group ran on: here a member that
    ignores SIGTERM and has let go of the pipes, whose TMPDIR was then removed under it. A hook that died of a signal
    did not end as it meant to, so its group is ended as on a timeout; the code it died of is what comes back."""
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    pidfile = tmp_path / "member.pid"
    hook = tmp_path / "hook"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        "( trap '' TERM; exec >/dev/null 2>&1; exec sleep 600 ) &\n"
        f'echo "$!" > {shlex.quote(str(pidfile))}\n'
        "echo partial output\n"
        "kill -KILL $$\n", encoding="utf-8")
    hook.chmod(0o755)
    monkeypatch.setattr(vd, "REPRO_KILL_GRACE_S", 1.0)
    try:
        with _HangGuard(pidfile, 20) as guard:
            code, out, _ = vd.run_hook([str(hook)], {**os.environ, "TMPDIR": str(scratch)}, 30)
        pid = _read_pid(pidfile)
        assert not guard.fired
        assert (code, out) == (-signal.SIGKILL, "partial output\n"), (code, out)
        assert not _running(pid), f"신호로 죽은 훅의 그룹에 남은 {pid} 이 run_hook 이 돌아온 뒤에도 돎"
        assert list(scratch.iterdir()) == [], list(scratch.iterdir())
    finally:
        _kill_leftover(_pid_or_none(pidfile))


@needs_proc
def test_output_that_is_not_utf8_does_not_replace_the_hook_s_exit(tmp_path):
    """The hook's output was decoded strictly, so one byte that is not UTF-8 --
    a simulator log line, a progress bar -- raised UnicodeDecodeError out of communicate(). A hook that exited 0 came
    back as 「재현 훅 실행 실패」, and the error took the ending path besides: the daemon the hook had left in its group
    (the Omniverse Hub's shape) was sent SIGTERM. The output is decoded with replacement."""
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    pidfile = tmp_path / "member.pid"
    hook = tmp_path / "hook"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        "( exec >/dev/null 2>&1; exec sleep 600 ) &\n"
        f'echo "$!" > {shlex.quote(str(pidfile))}\n'
        "printf 'ok \\xff\\xfe bytes\\n'\n"
        "printf 'err \\xff\\n' >&2\n"
        "exit 0\n", encoding="utf-8")
    hook.chmod(0o755)
    try:
        code, out, err = vd.run_hook([str(hook)], {**os.environ, "TMPDIR": str(scratch)}, 30)
        assert (code, out, err) == (0, "ok �� bytes\n", "err �\n"), (code, out, err)
        time.sleep(0.3)                  # a SIGTERM sent to the group would have ended the daemon by now
        pid = _read_pid(pidfile)
        assert _running(pid), f"정상 종료한 훅의 그룹에 남은 데몬 {pid} 을 검증기가 끝냄"
        assert list(scratch.iterdir()) == [], list(scratch.iterdir())
    finally:
        _kill_leftover(_pid_or_none(pidfile))


@needs_proc
def test_output_that_is_not_utf8_does_not_swallow_a_terminating_signal(tmp_path):
    """The same byte on the way out: a SIGTERM to the validator ended the wait, and collecting the output while the
    group was being ended raised UnicodeDecodeError in place of the exit (143). C9 reported it as 「재현 훅 실행 실패」
    and went on to the next sample -- the validator did not stop."""
    pidfile = tmp_path / "child.pid"
    hook = tmp_path / "hook"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        "trap 'exit 143' TERM\n"
        "printf 'progress \\xff\\n'\n"
        "sleep 600 &\n"
        f'echo "$!" > {shlex.quote(str(pidfile))}\n'
        "wait\n", encoding="utf-8")
    hook.chmod(0o755)
    repo = Path(__file__).resolve().parents[2]
    runner = subprocess.Popen(
        [sys.executable, "-c",
         "import os, sys; import validator.validate_dataset as vd; "
         "vd.run_hook([sys.argv[1]], dict(os.environ), 600)", str(hook)],
        cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    try:
        deadline = time.monotonic() + 60
        while _pid_or_none(pidfile) is None and runner.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        pid = _pid_or_none(pidfile)
        assert pid is not None, f"훅이 시작되지 않았습니다: {runner.poll()}"
        time.sleep(0.2)                  # the byte is in the pipe, and read, before the signal
        runner.send_signal(signal.SIGTERM)
        _, err = runner.communicate(timeout=30)
        assert runner.returncode == 143, (runner.returncode, err[-2000:])
        assert _wait_until_gone(pid), f"검증기가 SIGTERM 으로 끝난 뒤에도 훅의 자식 {pid} 이 돎"
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.communicate()
        _kill_leftover(_pid_or_none(pidfile))


@needs_proc
def test_an_interrupt_while_waiting_for_what_the_leader_left_still_kills_it(tmp_path, monkeypatch):
    """The wait for the rest of the group (the leader has exited on SIGTERM) is where a second Ctrl+C now lands:
    it must end the wait with the SIGKILL, not skip it."""
    hook, pidfile = _hook_leaving_a_member(tmp_path, exits_by_itself=False)
    monkeypatch.setattr(vd, "REPRO_KILL_GRACE_S", 30)       # long: only the interrupt can end the wait
    before = signal.signal(signal.SIGALRM, _raise_keyboard_interrupt)
    # The timeout fires 1 s after the launch and the leader exits on SIGTERM at once; the interrupt lands 2 s
    # after the hook recorded its member, a second into the wait for it.
    cancel = _signal_self_once_started(pidfile, signal.SIGALRM, delay=2.0)
    try:
        with _HangGuard(pidfile, 20) as guard, pytest.raises(KeyboardInterrupt):
            vd.run_hook([str(hook)], dict(os.environ), 1.0)
        pid = _read_pid(pidfile)
        assert not guard.fired
        assert not _running(pid), f"SIGTERM 을 무시하는 구성원 {pid} 이 기다리는 중의 중단 뒤에도 돎"
    finally:
        cancel()
        signal.signal(signal.SIGALRM, before)
        _kill_leftover(_pid_or_none(pidfile))


def test_a_group_that_is_already_gone_is_still_collected(tmp_path, monkeypatch):
    """The group can end between the timeout and the signal (`os.killpg` then raises
    ProcessLookupError). What the hook wrote must still come back, and the child be reaped:
    leaving on the exception without collecting left ``out`` unbound."""
    hook = tmp_path / "hook"
    hook.write_text("#!/usr/bin/env bash\necho partial output\necho partial error >&2\nsleep 1\n",
                    encoding="utf-8")
    hook.chmod(0o755)

    def gone(pgid, sig):
        raise ProcessLookupError(pgid)

    monkeypatch.setattr(os, "killpg", gone)
    code, out, err = vd.run_hook([str(hook)], dict(os.environ), 0.3)
    assert (code, out, err) == (None, "partial output\n", "partial error\n")


@needs_proc
@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
def test_collecting_the_output_does_not_wait_for_what_left_the_group(tmp_path, monkeypatch):
    """Something that left the hook's group (setsid) still holds the pipes, and the hook's output is
    only complete once every holder has closed them -- an unbounded final collect kept run_hook, and
    with it C9, waiting as long as the escapee lived. The wait is bounded and what was read comes back."""
    escapee = tmp_path / "escapee.pid"
    hook = tmp_path / "hook"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        "echo partial output\n"
        f"setsid bash -c 'echo $$ > {shlex.quote(str(escapee))}; exec sleep 30' &\n"
        "sleep 600\n", encoding="utf-8")
    hook.chmod(0o755)
    monkeypatch.setattr(vd, "REPRO_KILL_GRACE_S", 0.5)
    monkeypatch.setattr(vd, "REPRO_COLLECT_TIMEOUT_S", 0.5)
    try:
        started = time.monotonic()
        with _HangGuard(escapee) as guard:
            code, out, err = vd.run_hook([str(hook)], dict(os.environ), 1.0)
        elapsed = time.monotonic() - started
        assert not guard.fired, "run_hook 이 그룹을 떠난 프로세스가 pipe 를 쥔 동안 돌아오지 못했음"
        assert code is None and "partial output" in out, (code, out)
        assert elapsed < 5, elapsed
    finally:
        _kill_leftover(_pid_or_none(escapee))


@needs_proc
def test_an_interrupted_validator_takes_the_hook_s_group_down_with_it(tmp_path):
    """The hook's own session no longer receives the terminal's Ctrl+C, so an interrupt reaches the
    validator alone -- and the simulator would run on. The interrupt still propagates."""
    hook, pidfile, dirfile = _long_running_hook(tmp_path)
    before = signal.signal(signal.SIGALRM, _raise_keyboard_interrupt)
    cancel = _signal_self_once_started(pidfile, signal.SIGALRM)
    try:
        with _HangGuard(pidfile) as guard, pytest.raises(KeyboardInterrupt):
            vd.run_hook([str(hook)], dict(os.environ), 600)
        pid = _read_pid(pidfile)
        assert not guard.fired, "중단된 검증기가 훅의 그룹을 끝내지 못해 시험이 자식을 죽여야 했음"
        assert _wait_until_gone(pid), f"검증기가 중단된 뒤에도 훅의 자식 {pid} 이 돎"
        assert not Path(dirfile.read_text(encoding="utf-8").strip()).exists(), "작업 디렉터리가 남음"
    finally:
        cancel()
        signal.signal(signal.SIGALRM, before)
        _kill_leftover(_pid_or_none(pidfile))


@needs_proc
def test_an_interrupt_during_the_sigterm_grace_still_ends_the_group(tmp_path, monkeypatch):
    """The SIGKILL is the part that must not be skipped: a second Ctrl+C while the group is still
    winding down after SIGTERM raised out of the wait and the escalation with it, leaving a group that
    ignores SIGTERM running."""
    hook, pidfile, _ = _long_running_hook(tmp_path, ignore_term=True)
    monkeypatch.setattr(vd, "REPRO_KILL_GRACE_S", 30)       # long: only the interrupt can end the wait
    before = signal.signal(signal.SIGALRM, _raise_keyboard_interrupt)
    # The timeout fires 1 s after the launch and SIGTERM is ignored; the interrupt lands 2 s after the
    # hook recorded its child, a second into the grace.
    cancel = _signal_self_once_started(pidfile, signal.SIGALRM, delay=2.0)
    try:
        with _HangGuard(pidfile, 20) as guard, pytest.raises(KeyboardInterrupt):
            vd.run_hook([str(hook)], dict(os.environ), 1.0)
        pid = _read_pid(pidfile)
        assert not guard.fired
        assert _wait_until_gone(pid), f"SIGTERM 을 무시하는 자식 {pid} 이 유예 중의 중단 뒤에도 돎"
    finally:
        cancel()
        signal.signal(signal.SIGALRM, before)
        _kill_leftover(_pid_or_none(pidfile))


@needs_proc
@pytest.mark.parametrize("sig, status", [(signal.SIGTERM, 143), (signal.SIGHUP, 129)])
def test_a_terminating_signal_to_the_validator_takes_the_hook_s_group_down_with_it(tmp_path, sig, status):
    """SIGTERM or SIGHUP (an SSH hangup, `timeout`, `kill`) does not reach the hook either -- it is in
    its own session -- and Python dies of it without an exception, so nothing ended the group: the
    validator exited 143 with the simulator running on. The validator now turns the signal into an
    exit (128 + the signal number) while it waits, and that runs the group's ending."""
    hook, pidfile, dirfile = _long_running_hook(tmp_path)
    repo = Path(__file__).resolve().parents[2]
    runner = subprocess.Popen(
        [sys.executable, "-c",
         "import os, sys; import validator.validate_dataset as vd; "
         "vd.run_hook([sys.argv[1]], dict(os.environ), 600)", str(hook)],
        cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 60
        while _pid_or_none(pidfile) is None and runner.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        pid = _pid_or_none(pidfile)
        assert pid is not None, f"훅이 시작되지 않았습니다: {runner.poll()}"
        runner.send_signal(sig)
        runner.communicate(timeout=30)
        assert runner.returncode == status, (
            f"{sig.name} 뒤 검증기의 종료 코드 {runner.returncode} (기대 {status}) — 신호로 그냥 죽으면 "
            "그룹을 끝내는 코드가 돌지 않습니다")
        assert _wait_until_gone(pid), f"검증기가 {sig.name} 로 끝난 뒤에도 훅의 자식 {pid} 이 돎"
        assert not Path(dirfile.read_text(encoding="utf-8").strip()).exists(), "작업 디렉터리가 남음"
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.communicate()
        _kill_leftover(_pid_or_none(pidfile))


# The window between the hook's creation and run_hook holding its Popen object.
# A SIGTERM handled in there raised out of Popen before `p` was bound, so the group's end was skipped.
_LAUNCH_RUNNER = """
import os, signal, subprocess, sys, time
import validator.validate_dataset as vd

hook, started, mode = sys.argv[1], sys.argv[2], sys.argv[3]
sig = {"TERM": signal.SIGTERM, "HUP": signal.SIGHUP, "INT": signal.SIGINT}[sys.argv[4]]
# As a validator started from a terminal has them, whatever this test was started under (a runner started in
# the background has SIGINT ignored)
signal.signal(signal.SIGINT, signal.default_int_handler)
signal.signal(signal.SIGTERM, signal.SIG_DFL)
signal.signal(signal.SIGHUP, signal.SIG_DFL)
real = subprocess.Popen


def signal_self():
    os.kill(os.getpid(), sig)
    for _ in range(200000):     # Python runs the handler in here, still inside Popen
        pass


class SignalledInsidePopen(real):
    # "created": the real Popen, then -- once the hook has started its child -- the signal to this process
    # before the object reaches run_hook. "failed": the signal, then a launch that fails (no such hook).
    def __init__(self, *a, **k):
        if mode == "failed":
            signal_self()
            super().__init__(*a, **k)
            return
        super().__init__(*a, **k)
        deadline = time.monotonic() + 30
        while not os.path.exists(started) and time.monotonic() < deadline:
            time.sleep(0.01)
        signal_self()


subprocess.Popen = SignalledInsidePopen
try:
    vd.run_hook([hook], dict(os.environ), 600)
except OSError as e:
    print(f"OSError: {e}")
    raise SystemExit(3)
except KeyboardInterrupt:
    raise SystemExit(130)
"""


def _launch_signalled(tmp_path, hook, started, mode, sig):
    repo = Path(__file__).resolve().parents[2]
    runner = subprocess.Popen([sys.executable, "-c", _LAUNCH_RUNNER, str(hook), str(started), mode, sig],
                              cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = runner.communicate(timeout=60)
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.communicate()
    return runner.returncode, out, err


@needs_proc
@pytest.mark.parametrize("sig, status", [("TERM", 143), ("HUP", 129), ("INT", 130)])
def test_a_signal_while_the_hook_is_being_launched_still_ends_its_group(tmp_path, sig, status):
    """The signal is held until the Popen object is bound and then acted on: the validator exits as the signal
    says and the hook's child is ended with its group. Ctrl+C (SIGINT) is the same window: the hook's own session
    does not get it either."""
    hook, pidfile, _ = _long_running_hook(tmp_path)
    try:
        code, out, err = _launch_signalled(tmp_path, hook, pidfile, "created", sig)
        pid = _read_pid(pidfile)
        assert code == status, (code, out, err[-2000:])
        assert _wait_until_gone(pid), f"훅을 띄우는 사이 받은 SIG{sig} 뒤에도 훅의 자식 {pid} 이 돎"
    finally:
        _kill_leftover(_pid_or_none(pidfile))


@pytest.mark.parametrize("sig, status", [("TERM", 143), ("INT", 130)])
def test_a_signal_held_through_a_launch_that_fails_is_not_swallowed(tmp_path, sig, status):
    """Held and then lost: a launch that fails after the signal arrived raises OSError, which C9 reports as
    「재현 훅 실행 실패」 before going on to the next sample -- the validator must end as the signal says."""
    code, out, err = _launch_signalled(tmp_path, tmp_path / "no-such-hook", tmp_path / "unused", "failed", sig)
    assert code == status, (code, out, err[-2000:])


# A signal that lands inside run_hook's cleanup. Raised there it left the
# cleanup halfway: just before the SIGKILL (a member ignoring SIGTERM ran on), on entering the group's ending (nothing
# was signalled at all), on entering the temp directory's removal (the directory stayed), between putting back one
# handler and the next (SIGHUP's and SIGINT's stayed run_hook's). The runner pins the signal to that point by wrapping
# the one call made there; the signal and the processes are real.
_CLEANUP_RUNNER = """
import json, os, shutil, signal, subprocess, sys
import validator.validate_dataset as vd

hook, point, timeout, grace = sys.argv[1], sys.argv[2], float(sys.argv[4]), float(sys.argv[5])
sig = {"TERM": signal.SIGTERM, "HUP": signal.SIGHUP, "INT": signal.SIGINT}[sys.argv[3]]
# As a validator started from a terminal has them, whatever this test was started under
signal.signal(signal.SIGINT, signal.default_int_handler)
signal.signal(signal.SIGTERM, signal.SIG_DFL)
signal.signal(signal.SIGHUP, signal.SIG_DFL)
vd.REPRO_KILL_GRACE_S = grace
real_signal = signal.signal


def signal_self():
    os.kill(os.getpid(), sig)
    for _ in range(200000):     # Python runs the handler in here
        pass


def once(call, when, after=False):
    fired = []

    def wrapped(*a, **k):
        if fired or not when(*a):
            return call(*a, **k)
        fired.append(1)
        if after:
            result = call(*a, **k)
            signal_self()
            return result
        signal_self()
        return call(*a, **k)
    return wrapped


if point == "sigkill":                 # just before the SIGKILL syscall
    os.killpg = once(os.killpg, lambda pgid, s: s == signal.SIGKILL)
elif point == "ending":                # on entering the group's ending
    vd._end_hook_group = once(vd._end_hook_group, lambda *a: True)
elif point == "hold":                  # on entering the hold, before it takes effect
    vd._HookSignals.hold = once(vd._HookSignals.hold, lambda *a: True)
elif point == "rmtree":                # on entering the temp directory's removal
    shutil.rmtree = once(shutil.rmtree, lambda *a: True)
elif point == "restore":               # SIGTERM's handler put back, SIGHUP's not yet
    signal.signal = once(signal.signal, lambda s, h: s == signal.SIGHUP and h is signal.SIG_DFL)
elif point == "grace":                 # right after the group's SIGTERM: the grace has begun
    os.killpg = once(os.killpg, lambda pgid, s: s == signal.SIGTERM, after=True)
elif point == "error-grace":           # as "grace", in a cleanup that an error out of the main wait started
    real_communicate = subprocess.Popen.communicate
    main = []

    def communicate(self, input=None, timeout=None):
        if main:
            return real_communicate(self, input, timeout=timeout)
        main.append(1)
        try:
            return real_communicate(self, input, timeout=1.0)
        except subprocess.TimeoutExpired:
            raise OSError("stand-in: an error out of the main wait")

    subprocess.Popen.communicate = communicate
    os.killpg = once(os.killpg, lambda pgid, s: s == signal.SIGTERM, after=True)


def handlers():
    names = {signal.SIG_DFL: "SIG_DFL", signal.SIG_IGN: "SIG_IGN", signal.default_int_handler: "default_int_handler"}
    return {s.name: names.get(signal.getsignal(s), repr(signal.getsignal(s)))
            for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}


try:
    vd.run_hook([hook], dict(os.environ), timeout)
except SystemExit:
    signal.signal = real_signal
    print("HANDLERS " + json.dumps(handlers()))
    raise
except KeyboardInterrupt:
    signal.signal = real_signal
    print("HANDLERS " + json.dumps(handlers()))
    raise SystemExit(130)
print("RETURNED")
"""


def _signal_in_cleanup(tmp_path, hook, point, sig, *, timeout, grace=1.0):
    """Run the validator with ``sig`` pinned to ``point`` of run_hook's cleanup; (its exit status, the handlers it
    had after run_hook, its stderr, seconds)."""
    repo = Path(__file__).resolve().parents[2]
    started = time.monotonic()
    runner = subprocess.Popen([sys.executable, "-c", _CLEANUP_RUNNER, str(hook), point, sig, str(timeout), str(grace)],
                              cwd=repo, env={**os.environ, "TMPDIR": str(tmp_path / "tmp")},
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    try:
        out, err = runner.communicate(timeout=60)
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.communicate()
    lines = [line for line in out.splitlines() if line.startswith("HANDLERS ")]
    return runner.returncode, (json.loads(lines[-1].split(" ", 1)[1]) if lines else None), err, \
        time.monotonic() - started


_STATUS = {"TERM": 143, "HUP": 129, "INT": 130}


@needs_proc
@pytest.mark.parametrize("sig", ["TERM", "INT"])
@pytest.mark.parametrize("point", ["sigkill", "ending", "hold"])
def test_a_signal_during_the_group_s_ending_does_not_stop_it(tmp_path, point, sig):
    """The hook ran out of time and its member ignores SIGTERM. A signal to the validator just before the
    SIGKILL -- or as the group's ending begins -- is held until the group is ended and the temp directory removed,
    then acted on: the validator exits as the signal says, and nothing of the group runs on. One that lands as the
    hold is entered, before it takes effect, is raised there -- and the group is still ended."""
    (tmp_path / "tmp").mkdir()
    hook, pidfile = _hook_leaving_a_member(tmp_path, exits_by_itself=False)
    try:
        code, _, err, _ = _signal_in_cleanup(tmp_path, hook, point, sig, timeout=1.0)
        pid = _read_pid(pidfile)
        assert code == _STATUS[sig], (code, err[-2000:])
        assert _wait_until_gone(pid), f"정리 중({point}) 받은 SIG{sig} 뒤에도 SIGTERM 을 무시하는 구성원 {pid} 이 돎"
        time.sleep(0.3)                  # a member still running would have made its directory again by now
        assert list((tmp_path / "tmp").iterdir()) == [], list((tmp_path / "tmp").iterdir())
    finally:
        _kill_leftover(_pid_or_none(pidfile))


@needs_proc
@pytest.mark.parametrize("sig", ["TERM", "INT"])
def test_a_signal_during_the_temp_directory_s_removal_does_not_stop_it(tmp_path, sig):
    """The hook exited by itself and left a daemon in its group. A signal to the validator as the temp
    directory is being removed is held until it is gone, then acted on. The daemon is still the hook's to leave: the
    hook ended as it meant to before the signal came."""
    (tmp_path / "tmp").mkdir()
    hook, pidfile = _hook_leaving_a_member(tmp_path, exits_by_itself=True)
    try:
        code, _, err, _ = _signal_in_cleanup(tmp_path, hook, "rmtree", sig, timeout=30)
        pid = _read_pid(pidfile)
        assert code == _STATUS[sig], (code, err[-2000:])
        assert list((tmp_path / "tmp").iterdir()) == [], f"TMPDIR 에 남음: {list((tmp_path / 'tmp').iterdir())}"
        assert _running(pid), f"스스로 끝난 훅이 남긴 데몬 {pid} 을 검증기가 끝냄"
    finally:
        _kill_leftover(_pid_or_none(pidfile))


@pytest.mark.parametrize("sig", ["HUP", "INT"])
def test_a_signal_while_the_handlers_are_put_back_leaves_none_of_run_hook_s(tmp_path, sig):
    """A signal between putting back SIGTERM's handler and SIGHUP's raised out of the loop, and SIGHUP and
    SIGINT kept run_hook's handler -- a caller that catches the exit and goes on had later signals held forever or
    raised into whatever ran next. It is held until every handler is back, then acted on."""
    (tmp_path / "tmp").mkdir()
    hook = tmp_path / "hook"
    hook.write_text("#!/usr/bin/env bash\necho done\n", encoding="utf-8")
    hook.chmod(0o755)
    code, after, err, _ = _signal_in_cleanup(tmp_path, hook, "restore", sig, timeout=30)
    assert code == _STATUS[sig], (code, err[-2000:])
    assert after == {"SIGTERM": "SIG_DFL", "SIGHUP": "SIG_DFL", "SIGINT": "default_int_handler"}, after


@needs_proc
def test_a_signal_held_during_the_grace_still_ends_it_at_once(tmp_path):
    """Held is not ignored: a second Ctrl+C during the SIGTERM grace (30 s here) sends the SIGKILL at once, as it did
    when it was raised there -- and the validator then exits 130."""
    (tmp_path / "tmp").mkdir()
    hook, pidfile, _ = _long_running_hook(tmp_path, ignore_term=True)
    try:
        code, _, err, seconds = _signal_in_cleanup(tmp_path, hook, "grace", "INT", timeout=1.0, grace=30.0)
        pid = _read_pid(pidfile)
        assert code == 130, (code, err[-2000:])
        assert seconds < 15, f"{seconds:.1f} 초 — 유예(30 초) 중의 Ctrl+C 가 유예를 끝내지 않았음"
        assert _wait_until_gone(pid), f"SIGTERM 을 무시하는 자식 {pid} 이 돎"
    finally:
        _kill_leftover(_pid_or_none(pidfile))


@needs_proc
def test_a_group_not_seen_to_end_after_sigkill_is_not_reported_as_ended(tmp_path, monkeypatch):
    """After SIGKILL the group is waited for, and the wait is bounded -- a
    member in uninterruptible sleep (D: stuck I/O, a driver call) can outlast it -- yet C9 said 「훅의 프로세스 그룹을
    끝냄」 all the same. It says what it knows: SIGKILL was sent and the end was not seen. Stand-in for the D state: a
    group that never reads as gone (the real processes die of the SIGKILL)."""
    hook, pidfile, _ = _long_running_hook(tmp_path)
    monkeypatch.setattr(vd, "REPRO_HOOK_TIMEOUT_S", 2)      # room for the hook to record its child
    monkeypatch.setattr(vd, "REPRO_KILL_GRACE_S", 0.5)
    monkeypatch.setattr(vd, "REPRO_COLLECT_TIMEOUT_S", 0.5)
    real = vd._group_running
    monkeypatch.setattr(vd, "_group_running", lambda p: real(p) or True)     # reaps as it does, never gone
    try:
        rep = Report(tmp_path)
        ds = type("DS", (), {"root": tmp_path})()
        eps = pd.DataFrame([{"episode_id": "S1d_0000_seed7", "randomization_seed": 7}])
        with _HangGuard(pidfile, 20) as guard:
            c9_seed_repro(ds, eps, str(hook), 1, rep, 20)
        findings = [(f.severity, f.message) for f in rep.findings if f.check == "C9"]
        assert not guard.fired
        (severity, message), *rest = findings
        assert severity == "ERROR" and "시간 초과(2 s)" in message, findings
        assert "훅의 프로세스 그룹을 끝냄" not in message and "끝났는지 확인하지 못함" in message, message
        assert [s for s, m in rest if "확인하지 못함" in m] == ["WARN"], findings
    finally:
        _kill_leftover(_pid_or_none(pidfile))


def _hook_locking_its_tmpdir(tmp_path):
    """A hook that leaves a read-only subdirectory with a file in its TMPDIR, then exits 0 as a passed comparison
    would."""
    hook = tmp_path / "hook"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        'mkdir -p "$TMPDIR/locked" && echo frame > "$TMPDIR/locked/frame" && chmod 500 "$TMPDIR/locked"\n'
        "echo compared 16 columns\n"
        "exit 0\n", encoding="utf-8")
    hook.chmod(0o755)
    return hook


def _unlock_and_remove(scratch):
    """Give back what the hook took away and remove what it left: the test's own leftover, not the validator's."""
    subprocess.run(["chmod", "-R", "u+rwx", str(scratch)], check=False)
    for left in scratch.iterdir():
        shutil.rmtree(left, ignore_errors=True)


needs_non_root = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                                    reason="root removes files from a read-only directory")


@needs_non_root
def test_a_temp_directory_that_could_not_be_removed_is_reported(tmp_path, monkeypatch):
    """Removing the hook's TMPDIR with every error ignored would leave a subdirectory the hook
    made read-only, with its file, behind without a word -- up to 1.4-1.5 GB a camera episode. The hook's own result stands; C9 also says what was left, as a WARN."""
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    hook = _hook_locking_its_tmpdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(scratch))
    rep = Report(tmp_path)
    ds = type("DS", (), {"root": tmp_path})()
    eps = pd.DataFrame([{"episode_id": "S1d_0000_seed7", "randomization_seed": 7}])
    try:
        c9_seed_repro(ds, eps, str(hook), 1, rep, 20)
        findings = [(f.severity, f.message) for f in rep.findings if f.check == "C9"]
        (left,) = scratch.iterdir()          # still there: the test means something
        assert [severity for severity, _ in findings] == ["INFO", "WARN"], findings
        assert "compared 16 columns" in findings[0][1], findings
        assert "지우지 못함" in findings[1][1] and str(left) in findings[1][1], findings
    finally:
        _unlock_and_remove(scratch)


@needs_non_root
def test_a_temp_directory_left_on_the_way_out_is_said_on_stderr(tmp_path):
    """On the way out -- a signal to the validator -- there is no result to carry it, so the same sentence goes to
    stderr before the exit."""
    (tmp_path / "tmp").mkdir()
    hook = _hook_locking_its_tmpdir(tmp_path)
    try:
        code, _, err, _ = _signal_in_cleanup(tmp_path, hook, "rmtree", "TERM", timeout=30)
        (left,) = (tmp_path / "tmp").iterdir()
        assert code == 143, (code, err[-2000:])
        assert "지우지 못함" in err and str(left) in err, err[-2000:]
    finally:
        _unlock_and_remove(tmp_path / "tmp")


@needs_proc
@pytest.mark.parametrize("sig", ["TERM", "INT"])
def test_a_signal_held_through_a_cleanup_an_error_started_is_not_dropped(tmp_path, sig):
    """A cleanup can also start from an error -- an OSError out of the
    main wait (the stand-in here). A signal held during that cleanup, dropped with the error, would let C9 catch the
    error as 「재현 훅 실행 실패」 and go on to the next sample, where without the hold the signal would have ended the
    validator. The held signal outranks the error (which it carries as its context)."""
    (tmp_path / "tmp").mkdir()
    hook, pidfile, _ = _long_running_hook(tmp_path, ignore_term=True)
    try:
        code, _, err, seconds = _signal_in_cleanup(tmp_path, hook, "error-grace", sig, timeout=600, grace=5.0)
        pid = _read_pid(pidfile)
        assert code == _STATUS[sig], (code, err[-2000:])
        assert _wait_until_gone(pid), f"SIGTERM 을 무시하는 자식 {pid} 이 돎"
        assert list((tmp_path / "tmp").iterdir()) == [], list((tmp_path / "tmp").iterdir())
    finally:
        _kill_leftover(_pid_or_none(pidfile))


@needs_proc
def test_the_hook_runs_with_termination_signals_unblocked(tmp_path):
    """Whatever run_hook does to hold a signal back while it launches the hook must not reach the hook: a
    blocked SIGTERM is inherited through fork and exec, and the hook's traps -- and the simulator's shutdown
    -- would never see the group's SIGTERM."""
    seen = tmp_path / "status"
    hook = tmp_path / "hook"
    hook.write_text(f"#!/usr/bin/env bash\ncat /proc/$$/status > {shlex.quote(str(seen))}\n", encoding="utf-8")
    hook.chmod(0o755)
    assert vd.run_hook([str(hook)], dict(os.environ), 30)[0] == 0
    fields = dict(line.split(":", 1) for line in seen.read_text(encoding="utf-8").splitlines() if ":" in line)
    blocked = int(fields["SigBlk"].strip(), 16)
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        assert not blocked & (1 << (sig - 1)), f"훅이 {sig.name} 을 막은 채 돎 (SigBlk {fields['SigBlk'].strip()})"


def test_run_hook_leaves_the_callers_signal_handlers_as_it_found_them(tmp_path):
    """The handlers exist only while run_hook waits, and only where they can be installed."""
    quick = tmp_path / "quick"
    quick.write_text("#!/usr/bin/env bash\necho done\n", encoding="utf-8")
    quick.chmod(0o755)
    slow = tmp_path / "slow"
    slow.write_text("#!/usr/bin/env bash\nsleep 30\n", encoding="utf-8")
    slow.chmod(0o755)
    watched = (signal.SIGTERM, signal.SIGHUP)
    original = {sig: signal.getsignal(sig) for sig in watched}
    # Start from the default whatever an earlier test left behind: a handler that leaked out of one
    # run_hook would otherwise be "found" here, and the comparison below would pass on it.
    for sig in watched:
        signal.signal(sig, signal.SIG_DFL)
    before = {sig: signal.SIG_DFL for sig in watched}
    try:
        assert vd.run_hook([str(quick)], dict(os.environ), 30) == (0, "done\n", "")
        assert {sig: signal.getsignal(sig) for sig in watched} == before
        code, _, _ = vd.run_hook([str(slow)], dict(os.environ), 0.3)
        assert code is None and {sig: signal.getsignal(sig) for sig in watched} == before

        # Off the main thread signal.signal raises ValueError: the hook must still run.
        result = []
        worker = threading.Thread(
            target=lambda: result.append(vd.run_hook([str(quick)], dict(os.environ), 30)))
        worker.start()
        worker.join(30)
        assert result == [(0, "done\n", "")], result
    finally:
        for sig, handler in original.items():
            signal.signal(sig, handler)


def test_handlers_already_installed_are_put_back_when_installing_stops_halfway(tmp_path, monkeypatch):
    """The handlers went in before run_hook's ``try``: an exception
    between installing one and entering the ``try`` left SIGTERM and SIGHUP held by a handler of a run_hook that
    no longer runs -- a later SIGTERM would be held forever, or raised into whatever runs next."""
    quick = tmp_path / "quick"
    quick.write_text("#!/usr/bin/env bash\necho done\n", encoding="utf-8")
    quick.chmod(0o755)
    watched = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    original = {sig: signal.getsignal(sig) for sig in watched}
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGHUP, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    real = signal.signal

    def refuses_sigint(sig, handler):
        if sig == signal.SIGINT and handler is not signal.default_int_handler:
            raise ValueError("SIGINT refused")
        return real(sig, handler)

    try:
        monkeypatch.setattr(signal, "signal", refuses_sigint)
        scratch = tmp_path / "tmp"
        scratch.mkdir()
        with pytest.raises(ValueError, match="SIGINT refused"):
            vd.run_hook([str(quick)], {**os.environ, "TMPDIR": str(scratch)}, 30)
        monkeypatch.undo()
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL, signal.getsignal(signal.SIGTERM)
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_DFL, signal.getsignal(signal.SIGHUP)
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        assert list(scratch.iterdir()) == []
    finally:
        monkeypatch.undo()
        for sig, handler in original.items():
            signal.signal(sig, handler)


def test_a_validator_that_ignores_sighup_keeps_ignoring_it(tmp_path):
    """`nohup` starts the validator with SIGHUP ignored so that it outlives the terminal. A handler
    installed over that would turn the hangup back into an exit."""
    started = tmp_path / "started"
    hook = tmp_path / "hook"
    hook.write_text(f"#!/usr/bin/env bash\ntouch {shlex.quote(str(started))}\nsleep 1\n", encoding="utf-8")
    hook.chmod(0o755)
    before = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    cancel = _signal_self_once_started(started, signal.SIGHUP)
    try:
        code, _, _ = vd.run_hook([str(hook)], dict(os.environ), 30)
        assert code == 0, code
    finally:
        cancel()
        signal.signal(signal.SIGHUP, before)


def test_the_hook_is_run_in_the_environment_the_validator_built(tmp_path, monkeypatch):
    """The other tests patch run_hook and see what C9 hands over; this one runs the real thing and
    reads what the hook received. It guards two doors -- `C9_ISAAC_PYTHON=/bin/true`
    reaching the hook and passing C9 with nothing run, and a fake bash first on the caller's
    PATH running in place of the repository hook. Dropping `env=` from the Popen call reopens
    both and no run_hook-patching test notices."""
    seen = tmp_path / "seen.env"
    hook = tmp_path / "hook"
    hook.write_text(f"#!/usr/bin/env bash\nenv > {shlex.quote(str(seen))}\n", encoding="utf-8")
    hook.chmod(0o755)
    for name in ("C9_ISAAC_PYTHON", "C9_COMPARE_PYTHON", "C9_SEQ_PYTHON", "C9_PUBLISHED_ROOT", "C9_ANYTHING"):
        monkeypatch.setenv(name, "/bin/true")
    monkeypatch.setenv("KETI_PASSED_THROUGH", "yes")
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setenv("PATH", "/tmp/evil:" + os.environ["PATH"])

    rep = Report(tmp_path)
    ds = type("DS", (), {"root": tmp_path})()
    eps = pd.DataFrame([{"episode_id": "S1a_0000_seed7", "randomization_seed": 7}])
    c9_seed_repro(ds, eps, str(hook), 1, rep, 20)
    (finding,) = [f for f in rep.findings if f.check == "C9"]
    assert finding.severity == "INFO", finding.message

    got = dict(line.split("=", 1) for line in seen.read_text(encoding="utf-8").splitlines() if "=" in line)
    assert got.get("PATH") == vd._SAFE_PATH, got.get("PATH")
    assert sorted(k for k in got if k.startswith("C9_")) == ["C9_PUBLISHED_ROOT"], sorted(got)
    assert got["C9_PUBLISHED_ROOT"] == str(tmp_path.resolve())
    assert got.get("KETI_PASSED_THROUGH") == "yes", "the rest of the caller's environment is kept"
    # The one value the validator sets besides those: TMPDIR, a directory of its own under the caller's,
    # removed once the hook has ended (F6).
    given = Path(got["TMPDIR"])
    assert given.parent == scratch and not given.exists() and list(scratch.iterdir()) == [], (
        given, list(scratch.iterdir()))


@needs_proc
@pytest.mark.parametrize("sig, code", [(signal.SIGTERM, 143), (signal.SIGINT, 130)])
def test_the_repository_hook_exits_and_cleans_up_when_its_group_is_signalled(tmp_path, sig, code):
    """The EXIT trap of generation/rerun_seed.sh removes the work directory; the TERM and INT traps make
    a signal end the script through it (bash may skip an EXIT trap when a signal kills it). The exit
    status tells the trap from the default death: 143 / 130 by the trap, -15 / -2 without it."""
    repo = Path(__file__).resolve().parents[2]
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    started = tmp_path / "isaac.started"
    fake_isaac = tmp_path / "fake-isaac"
    fake_isaac.write_text(f"#!/usr/bin/env bash\ntouch {shlex.quote(str(started))}\nsleep 600\n",
                          encoding="utf-8")
    fake_isaac.chmod(0o755)
    env = {**os.environ, "TMPDIR": str(scratch), "C9_PUBLISHED_ROOT": str(tmp_path / "published"),
           "C9_SEQ_PYTHON": sys.executable, "C9_ISAAC_PYTHON": str(fake_isaac),
           "C9_COMPARE_PYTHON": "/bin/false"}
    p = subprocess.Popen([str(repo / "generation" / "rerun_seed.sh"), "S1a_0000_seed7", "7"],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    try:
        deadline = time.monotonic() + 60
        while not started.exists() and p.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert started.exists(), f"훅이 시뮬레이터 자리까지 가지 못했습니다: {p.poll()}"
        assert len(list(scratch.iterdir())) == 1, "훅이 작업 디렉터리를 하나 만들었어야 시험이 뜻을 가집니다"
        os.killpg(p.pid, sig)
        p.communicate(timeout=30)
        assert p.returncode == code, (
            f"신호 {sig.name} 뒤 훅의 종료 코드 {p.returncode} (트랩이면 {code}) — 트랩이 없으면 신호로 죽습니다")
        assert list(scratch.iterdir()) == [], "신호로 끝난 훅이 작업 디렉터리를 남김"
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.communicate()


# ── A hook that stalls is ended, and the sample run again ──
#
# Isaac sometimes stalls -- 3 attempts of 43 in the S1d-400 batch, and one C9 sample on 2026-10-06, which
# then waited the whole 1800 s. The generator writes its lossless streams as it steps, so the batch runner
# reads a directory nothing is written to for 300 s as a stall; the hook writes its rerun under the TMPDIR
# the validator hands it, and C9 watches that the same way.


def test_the_stall_watch_uses_the_batch_runner_s_measured_thresholds():
    """The same generator and the same writer: the runner's values held on 43 attempts without a false
    stall. The validator does not import generation/, so it carries copies."""
    from generation import run_batch

    assert vd.REPRO_STALL_S == run_batch.STALL_AFTER_S
    assert vd.REPRO_STALL_GRACE_S == run_batch.STALL_GRACE_S
    assert vd.REPRO_STALL_ATTEMPTS == 1 + run_batch.GENERATE_RETRIES


def _fast_stall_watch(monkeypatch, *, grace=0.5, stall=1.0, poll=0.1, kill_grace=0.5):
    monkeypatch.setattr(vd, "REPRO_STALL_GRACE_S", grace)
    monkeypatch.setattr(vd, "REPRO_STALL_S", stall)
    monkeypatch.setattr(vd, "REPRO_STALL_POLL_S", poll)
    monkeypatch.setattr(vd, "REPRO_KILL_GRACE_S", kill_grace)


def _hook(tmp_path, body):
    """A bash hook that keeps its work under TMPDIR, as the repository hook does."""
    hook = tmp_path / "hook"
    hook.write_text("#!/usr/bin/env bash\n"
                    'WORK="$(mktemp -d)"\n'
                    "trap 'rm -rf \"$WORK\"' EXIT\n"
                    "trap 'exit 143' TERM\n" + body, encoding="utf-8")
    hook.chmod(0o755)
    return hook


@needs_proc
@pytest.mark.parametrize("writes_first", [True, False], ids=["stalls after writing", "stalls at boot"])
def test_a_hook_that_stops_writing_is_ended_as_stalled_long_before_its_timeout(tmp_path, monkeypatch,
                                                                              writes_first):
    _fast_stall_watch(monkeypatch)
    pidfile = tmp_path / "child.pid"
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    hook = _hook(tmp_path, ('echo frame > "$WORK/episode.mkv"\n' if writes_first else "")
                 + "echo stuck at the policy >&2\n"
                 "sleep 600 &\n"
                 f'echo "$!" > {shlex.quote(str(pidfile))}\n'
                 "wait\n")
    try:
        started = time.monotonic()
        with _HangGuard(pidfile, seconds=20) as guard:
            run = vd.run_hook([str(hook)], dict(os.environ), 60)
        elapsed = time.monotonic() - started
        assert not guard.fired
        assert run[0] is None and run.stalled, (run, run.stalled)
        assert elapsed < 10, f"{elapsed:.1f} 초 — 제한 시간(60 초)이 아니라 멈춤 감시가 끝냈어야 합니다"
        assert "stuck at the policy" in run[2]
        assert _wait_until_gone(_read_pid(pidfile)), "멈춘 훅의 자식이 남음"
        assert list(scratch.iterdir()) == [], "멈춘 훅의 TMPDIR 이 남음"
    finally:
        _kill_leftover(_pid_or_none(pidfile))


@needs_proc
def test_a_hook_that_keeps_writing_is_not_a_stall(tmp_path, monkeypatch):
    _fast_stall_watch(monkeypatch)
    hook = _hook(tmp_path, "for i in $(seq 1 15); do echo $i >> \"$WORK/episode.mkv\"; sleep 0.2; done\n"
                           "echo compared\n")
    run = vd.run_hook([str(hook)], dict(os.environ), 60)
    assert tuple(run) == (0, "compared\n", "") and not run.stalled, (tuple(run), run.stalled)


@needs_proc
def test_nothing_is_judged_during_the_grace(tmp_path, monkeypatch):
    """Boot and scene build write nothing; the batch runner gives them 480 s."""
    _fast_stall_watch(monkeypatch, grace=4.0, stall=0.5)
    hook = _hook(tmp_path, "sleep 2\necho compared\n")
    run = vd.run_hook([str(hook)], dict(os.environ), 60)
    assert run[0] == 0 and not run.stalled, (tuple(run), run.stalled)


@needs_proc
def test_a_hook_that_is_merely_out_of_time_is_not_called_stalled(tmp_path, monkeypatch):
    _fast_stall_watch(monkeypatch, grace=60.0)
    hook = _hook(tmp_path, "sleep 600 &\nwait\n")
    run = vd.run_hook([str(hook)], dict(os.environ), 1.0)
    assert run[0] is None and not run.stalled


def _c9_with_runs(monkeypatch, runs):
    """C9 on one sample, with run_hook answering from ``runs`` in turn."""
    calls = []

    def fake_run(argv, env, timeout_s):
        calls.append(argv)
        return runs[len(calls) - 1]

    monkeypatch.setattr(vd, "run_hook", fake_run)
    rep = Report(Path("."))
    ds = type("DS", (), {"root": Path(".")})()
    eps = pd.DataFrame([{"episode_id": "S1d_0002_seed820211214", "randomization_seed": 820211214}])
    c9_seed_repro(ds, eps, "/bin/true", 1, rep, 20)
    return calls, [f for f in rep.findings if f.check == "C9"]


def _stalled():
    return vd.HookRun(None, "", "[Kit] loading the policy", stalled=True)


def test_a_stalled_sample_is_run_again_and_its_verdict_kept(monkeypatch):
    calls, findings = _c9_with_runs(monkeypatch, [_stalled(), (0, f"{COMPARED}\n", HOOK_STDERR)])
    assert len(calls) == 2 and calls[0] == calls[1]
    assert [f.severity for f in findings] == ["WARN", "INFO"], findings
    assert "멈춤" in findings[0].message and f"1/{vd.REPRO_STALL_ATTEMPTS}" in findings[0].message
    assert "loading the policy" in findings[0].message
    assert findings[1].message == f"시드 재현 확인 — {COMPARED}"


def test_a_sample_that_stalls_every_time_is_no_verdict(monkeypatch):
    n = vd.REPRO_STALL_ATTEMPTS
    calls, findings = _c9_with_runs(monkeypatch, [_stalled()] * (n + 2))
    assert len(calls) == n
    assert [f.severity for f in findings] == ["WARN"] * (n - 1) + ["ERROR"], findings
    assert f"{n}번 모두 멈춤" in findings[-1].message and "판정하지 못함" in findings[-1].message
    assert "loading the policy" in findings[-1].message


def test_a_stall_whose_group_was_not_seen_to_end_is_not_run_again(monkeypatch):
    """남은 구성원이 다음 실행과 GPU 를 나누면 그 재현은 다른 실행이다."""
    left = vd.HookRun(None, "", "", stalled=True, group_left=True)
    calls, findings = _c9_with_runs(monkeypatch, [left, (0, f"{COMPARED}\n", "")])
    assert len(calls) == 1
    assert [f.severity for f in findings] == ["ERROR"]
    assert "확인하지 못함" in findings[0].message and "다시 실행하지 않음" in findings[0].message


@pytest.mark.parametrize("run", [
    (None, "", "slow"),                    # out of time, not stalled: the timeout is not retried
    (1, "팩 head.depth: 값 불일치\n", ""),     # a mismatch is a verdict
    (2, "", "run_start_seq 가 0 이상의 정수가 아닙니다"),
], ids=["timeout", "mismatch", "hook error"])
def test_only_a_stall_is_run_again(monkeypatch, run):
    calls, findings = _c9_with_runs(monkeypatch, [run, (0, f"{COMPARED}\n", "")])
    assert len(calls) == 1
    assert [f.severity for f in findings] == ["ERROR"]


# ── A group that may still run: no further sample is started beside it ──


@pytest.mark.parametrize("left", [
    vd.HookRun(None, "", "", stalled=True, group_left=True),
    vd.HookRun(None, "", "", group_left=True),          # out of time, the group not seen to end
    vd.HookRun(-9, "", "", group_left=True),            # the leader killed, the group not seen to end
], ids=["stall", "timeout", "killed leader"])
def test_no_further_sample_is_started_beside_a_group_that_may_still_run(monkeypatch, left):
    """A member left running would share the GPU with the next sample's rerun -- the next sample, not only a retry
    of the same one."""
    calls = []

    def fake_run(argv, env, timeout_s):
        calls.append(argv)
        return left if len(calls) == 1 else (0, f"{COMPARED}\n", "")

    monkeypatch.setattr(vd, "run_hook", fake_run)
    rep = Report(Path("."))
    ds = type("DS", (), {"root": Path(".")})()
    c9_seed_repro(ds, episodes(2), "/bin/true", 2, rep, 20)
    findings = [f for f in rep.findings if f.check == "C9"]
    assert len(calls) == 1, calls
    (unrun,) = [f for f in findings if "돌리지 않음" in f.message and "표본" in f.message]
    assert unrun.severity == "ERROR"
    other = next(eid for eid in episodes(2)["episode_id"] if eid != calls[0][1])
    assert other in unrun.message, unrun.message
