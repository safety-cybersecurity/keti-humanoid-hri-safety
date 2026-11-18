#!/usr/bin/env python3
"""공개 배치를 청크로 생성한다.

    .venv/bin/python generation/run_batch.py --batch .tmp/batches/S1d-400
    .venv/bin/python generation/run_batch.py --batch <dir> --scenario S1e S1f S1g --rows <rows.json> --chunk 10

청크마다 생성 → 라벨링 → 변환 → 검증(ERROR 0) → 측정을 **하나씩** 돌리고, 통과하면 봉인 기록을 쓰고
무손실 영상을 지운다. 끊기면 같은 명령으로 다시 돌린다 — 봉인된 청크는 건너뛴다.

* 조건표가 없는 시나리오(S1d)의 청크는 생성기의 실행 하나다 (run_s1a.py --seq <시작> --episodes 10).
  봉인 안 된 청크는 지우고 처음부터 한다.
* 조건표 시나리오(S1e · S1f · S1g)는 행 하나가 생성기의 실행 하나라, 청크의 행마다 프로세스를 하나씩
  띄운다 (run_s1a.py --seq <행> --episodes 1). 봉인 안 된 청크는 **행 단위로 이어 간다** — 끝난 행은 두고,
  끝나지 않은 행의 파일을 지우고 그 행부터 다시 만든다. 생성 뒤 단계의 산출물은 처음부터 다시 만든다.
  측정은 점수표(scorecard.py)다.

멈추려면 배치 디렉터리에 STOP 파일을 만든다: 지금 청크를 끝내고 종료한다. 끝나면 assemble_batch.py.

Isaac 을 import 하지 않는다. 깨끗한 작업 트리에서, 배치를 시작한 커밋에서만 돈다 — 그리고 그 트리를 청크를
시작하기 전과 봉인하기 직전에 다시 읽는다: 생성기는 시작할 때 한 번만 읽으므로, 청크를 만드는 동안 정본이나
라벨러를 고치면 그 청크는 봉인되지 않는다. 배치가 열려 있는 동안 다른 작업은 별도 워크트리에서 한다.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation import conditions, provenance  # noqa: E402
from generation.batch_layout import (DATASET_DIR, EPISODES_DIR, LOCK_FILE, LOGS_DIR, MEDIA_DIR,  # noqa: E402
                                     RUNNER_FILE, STOP_FILE, Batch, BatchError, ChunkPlan, open_batch,
                                     read_seal, remove_media, sealed_files, write_atomically, write_seal)
from generation.envs.s1a.media_writer import clear_media  # noqa: E402

STALLED = "stalled"
STALLED_LOGS = "stalled"     # <batch>/stalled/<chunk key>[.<row>].attempt<n>[.<k>].generate.log: why an attempt stalled
# The generator writes its lossless streams as it steps, so a directory nothing is written to is
# a run that hangs (once: 2026-10-01, the main thread spinning in system time, cause unknown).
STALL_GRACE_S = 480.0        # boot and scene build before the first file
STALL_AFTER_S = 300.0
# The run of a condition-table row (seven cameras) writes its first file 16-18 s after it starts. One that has
# written nothing by this is taken for a boot that hangs -- a hang with no output at all was 2 of the 43 generation
# attempts of a 400-episode batch -- and is not left to the grace. Set by this project, not by a standard.
FIRST_WRITE_S = 120.0
STALL_POLL_S = 30.0
STALL_KILL_GRACE_S = 30.0    # SIGTERM, then this long, then SIGKILL
KILL_BOUND_S = 30.0          # after SIGKILL: how long the group may take to be gone before the runner stops
GENERATE_RETRIES = 2
# The validator's own crash, not a finding (exit 1): seen about once in thousands of runs.
VALIDATOR_CRASHED = 3
AFTER_GENERATE = ("label", "convert", "validate", "measure")


class ChunkFailed(Exception):
    def __init__(self, stage: str, detail: str):
        super().__init__(f"{stage}: {detail}")
        self.stage, self.detail = stage, detail


class GroupAlive(BatchError):
    """A stage's process group is still there after SIGKILL and ``bound_s``. Not gone on from: the
    runner would wipe the chunk under a live process. It stops and says which group is left."""

    def __init__(self, pgid: int, bound_s: float):
        super().__init__(
            f"프로세스 그룹 {pgid} 가 SIGKILL 뒤 {bound_s:g} 초가 지나도 남아 있습니다 — 청크를 지우지 않고 멈춥니다. "
            f"`ps -eo pid,pgid,stat,args` 에서 pgid 가 {pgid} 인 줄이 남은 프로세스입니다 (D 상태이면 재부팅 말고는 "
            "끝낼 수 없습니다). 모두 끝난 것을 확인한 뒤 다시 돌리십시오")
        self.pgid, self.bound_s = pgid, bound_s


def progress(line: str) -> None:
    """A progress line, flushed as it is written: redirected to a file, print() would hold it in an 8 KB
    buffer -- an 11-hour log empty until exit, and lost with a hard reset."""
    print(line, flush=True)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def newest_write(directory: Path) -> float | None:
    """The latest modification time of any file under ``directory``; None when it holds none."""
    newest = None
    for dirpath, _, names in os.walk(directory):
        for name in names:
            try:
                mtime = os.stat(os.path.join(dirpath, name)).st_mtime
            except OSError:
                continue
            if newest is None or mtime > newest:
                newest = mtime
    return newest


def file_states(directory: Path) -> dict[str, tuple[int, int, int, int]]:
    """Every file under ``directory`` by its path, with its inode, size, and modification and status-change times in
    nanoseconds; {} when it holds none or is not there. A file put in place of another under the same path, of
    the same size and modification time, is another inode -- and its status changed when it was put there."""
    states = {}
    for dirpath, _, names in os.walk(directory):
        for name in names:
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            states[path] = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    return states


def wrote_since(directory: Path, before: dict[str, tuple[int, int, int, int]]) -> bool:
    """Whether a file under ``directory`` is new, or has another inode, size, modification or status-change time,
    than in ``before`` (file_states taken earlier). No time is compared with a clock: a file an earlier run left
    dated ahead of the clock (set back since) is not a write, and one written now that a coarse clock dates before
    the start is."""
    return any(before.get(path) != state for path, state in file_states(directory).items())


def group_members(pgid: int) -> list[int] | None:
    """The PIDs of the members of process group ``pgid`` that still run, read from /proc: a member that has
    died and is not yet reaped -- a zombie, which runs nothing and holds no file -- is not one. None where
    /proc cannot be listed."""
    try:
        pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return None
    members = []
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
                state, _ppid, pgrp = f.read().rsplit(")", 1)[1].split()[:3]
        except (OSError, ValueError):
            continue                    # gone meanwhile, or not a line to read
        if int(pgrp) == pgid and state not in ("Z", "X"):
            members.append(pid)
    return members


def group_running(pgid: int) -> bool:
    """Whether any member of process group ``pgid`` still runs. ``os.killpg(pgid, 0)`` also counts a member
    that has died and is not yet reaped -- a zombie, which runs nothing and holds no file -- so where /proc
    can be read a group of zombies alone counts as gone. Without /proc such a group is waited for until the
    caller's bound."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    members = group_members(pgid)
    return True if members is None else bool(members)


def _wait_gone(pgid, within_s, running, reap=None, poll_s=0.05) -> bool:
    """Wait until no member of the group runs, or ``within_s`` has passed; whether it is gone. ``reap`` is
    called while waiting: the leader, if it is our child, is a zombie until it is."""
    until = time.monotonic() + within_s
    while True:
        if reap is not None:
            reap()
        if not running(pgid):
            return True
        if time.monotonic() >= until:
            return False
        time.sleep(poll_s)


def end_group(p: subprocess.Popen, kill_grace_s: float, *, bound_s: float = KILL_BOUND_S,
              running=group_running) -> None:
    """SIGTERM to the whole process group ``p`` leads, then ``kill_grace_s`` for the *group* to end -- not
    for the leader: the generator's leader is a bash wrapper that dies of SIGTERM at once while its child
    still cleans up -- then SIGKILL to what is left, then up to ``bound_s`` to see it gone. GroupAlive when it
    is not. By PID: nothing is looked up by name.

    An interrupt while the group is being ended -- Ctrl-C during the grace of a stage that ignores SIGTERM --
    does not leave it running: the interrupted runner gives up the grace, SIGKILLs the group and waits for it
    (further interrupts held until it is gone), and only then lets the interrupt through."""
    pgid = p.pid
    try:
        for sig, wait in ((signal.SIGTERM, kill_grace_s), (signal.SIGKILL, bound_s)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass
            if _wait_gone(pgid, wait, running, reap=p.poll):
                p.poll()
                return
    except KeyboardInterrupt:
        _kill_for_good(p, bound_s, running)
        raise
    raise GroupAlive(pgid, bound_s)


def _kill_for_good(p: subprocess.Popen, bound_s: float, running) -> None:
    """SIGKILL to ``p``'s group and wait up to ``bound_s`` for it to be gone; an interrupt meanwhile is held
    (the wait goes on with what is left of the bound). GroupAlive when it is not gone."""
    pgid, until = p.pid, time.monotonic() + bound_s
    while True:
        try:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if _wait_gone(pgid, max(until - time.monotonic(), 0.0), running, reap=p.poll):
                p.poll()
                return
            raise GroupAlive(pgid, bound_s)
        except KeyboardInterrupt:
            continue


def end_group_by_id(pgid: int, kill_grace_s: float, *, bound_s: float = KILL_BOUND_S,
                    running=group_running) -> None:
    """The same for a group that is not this process's child (a stage a dead runner left)."""
    for sig, wait in ((signal.SIGTERM, kill_grace_s), (signal.SIGKILL, bound_s)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        if _wait_gone(pgid, wait, running):
            return
    raise GroupAlive(pgid, bound_s)


def run_stage(argv, log_path, *, cwd, on_start=None, watch_dir=None, grace_s=STALL_GRACE_S,
              first_write_s: float | None = None, stall_s=STALL_AFTER_S, poll_s=STALL_POLL_S,
              kill_grace_s=STALL_KILL_GRACE_S, bound_s=KILL_BOUND_S, running=group_running, say=progress):
    """Run ``argv`` in a session of its own, its output to ``log_path``; its exit code.

    With ``watch_dir``: STALLED, after ending the stage's process group, when nothing under that
    directory was written for ``stall_s`` -- not judged during the first ``grace_s``. With
    ``first_write_s`` as well: STALLED when no file under it is new or changed since just before the stage
    started (wrote_since: what the directory held then, by path, inode, size, and modification and status-change
    times -- no file's time is compared with a clock) once ``first_write_s`` has passed on the monotonic clock -- a run that writes
    nothing at all is not left to the grace, and what earlier runs left in the directory does not count,
    whatever time it carries. Once a file is written the grace and the stall are as without it. The stage
    is held by its PID and never looked up by name: a pattern matches the shell that waits too.
    ``on_start`` is told the group's id when the stage starts and None when the stage is over.

    Everything after the start is inside one clean-up: however this function is left, the stage's group is
    not left running -- except after a leader that exited normally. The group is ended (``end_group``: SIGTERM,
    the grace, SIGKILL; ``GroupAlive`` if it survives) when the stage stalls, when this function is interrupted
    or ``on_start`` fails (the leader still runs then), and when the leader has been ended by a signal -- its
    return code is negative: killed by hand or by the out-of-memory killer, it did not wait for its child, and
    the simulator it started is a straggler. The leader of the generator is a bash wrapper that can die while the
    simulator does not.

    After a leader that exited with a code (zero or not) the group is left alone. A wrapper that exits normally
    has waited for its child, so what remains in its group was left there by the simulator on purpose: the
    Omniverse hub that Kit starts stays in the group, shared with other sessions on the machine (the validator's
    C9 hook leaves it running as well). Nothing is spared by its name -- no name patterns. What this knowingly
    does not catch: a leader that exits with a normal code while a child of its own still runs. ``on_start`` is
    told None whenever the group is over for this runner; after ``GroupAlive`` it is not, and the record keeps
    naming the group. ``bound_s`` and ``running`` are those of ``end_group``.
    """
    # What the directory holds just before the stage starts: its first write is a file that is new or has changed
    # since, and the deadline for it is on the monotonic clock.
    before = file_states(Path(watch_dir)) if watch_dir is not None and first_write_s is not None else {}
    began = time.monotonic()
    # The start is read before the stage is: a file it writes is not dated before it on a coarse clock.
    started, code, normal, wrote = time.time(), None, False, False
    with open(log_path, "wb") as log:
        p = subprocess.Popen([str(a) for a in argv], cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        if on_start:
            on_start(p.pid)
        while code is None:
            try:
                code = p.wait(timeout=poll_s)
                continue
            except subprocess.TimeoutExpired:
                pass
            if watch_dir is None:
                continue
            now = time.time()
            if first_write_s is not None and not wrote:
                wrote = wrote_since(Path(watch_dir), before)
                if not wrote and time.monotonic() - began > first_write_s:
                    say(f"  {watch_dir} 에 띄운 지 {first_write_s:g} 초가 지나도 이 단계가 쓴 파일이 없어 단계를 끝냅니다 "
                        f"(PID {p.pid}) — 부팅에서 멈춘 것으로 봅니다")
                    code = STALLED
                    continue
            if now - started < grace_s:
                continue
            last = newest_write(Path(watch_dir))
            if now - (last if last is not None else started) > stall_s:
                say(f"  {watch_dir} 에 {stall_s:g} 초 동안 쓰인 파일이 없어 단계를 끝냅니다 (PID {p.pid})")
                code = STALLED
        normal = isinstance(code, int) and code >= 0
    finally:
        try:
            if not normal and (p.poll() is None or running(p.pid)):
                _sweep(p, kill_grace_s, bound_s, running, say)
        except KeyboardInterrupt:       # end_group lets an interrupt through only once the group is gone
            if on_start:
                on_start(None)
            raise
        if on_start:
            on_start(None)
    return code


def _sweep(p: subprocess.Popen, kill_grace_s: float, bound_s: float, running, say) -> None:
    """End the group of ``p``; when its leader was ended by a signal and members are left, say so."""
    signalled = p.returncode if p.returncode is not None and p.returncode < 0 else None
    members = group_members(p.pid) if signalled is not None else None
    end_group(p, kill_grace_s, bound_s=bound_s, running=running)
    if signalled is not None:
        say(f"  단계의 리더(PID {p.pid})가 시그널 {-signalled} 로 끝났고 그룹에 남은 프로세스 "
            + (f"{len(members)}개" if members is not None else "(수를 읽지 못함)") + "를 끝냈습니다")


def row_log(chunk: Path, seq: int) -> Path:
    """The log of the run of row ``seq`` in a chunk of condition-table rows."""
    return chunk / LOGS_DIR / f"generate-{seq:04d}.log"


class Stages:
    """The five commands of a chunk, each a subprocess of its own -- never two at once. A chunk of condition-table
    rows is generated by ``generate_row``, once a row; a chunk that is one run by ``generate``."""

    def __init__(self, *, repo, isaac_python, python, lerobot_python, run=run_stage):
        self.repo, self.run = Path(repo), run
        self.isaac_python, self.python, self.lerobot_python = isaac_python, python, lerobot_python

    def generate(self, batch: Batch, plan: ChunkPlan, chunk: Path):
        if plan.per_row:
            raise ValueError(f"청크 {plan.key} 은 조건표 시나리오의 행들입니다 — 행마다 generate_row 로 만듭니다")
        return self.run(
            [self.isaac_python, self.repo / "generation/envs/run_s1a.py", "--scenario", plan.scenario,
             "--seq", plan.rows[0], "--episodes", len(plan.rows), "--max-steps", 1000,
             "--out", chunk / EPISODES_DIR],
            chunk / LOGS_DIR / "generate.log", cwd=self.repo, watch_dir=chunk / EPISODES_DIR)

    def generate_row(self, batch: Batch, plan: ChunkPlan, seq: int, chunk: Path):
        # A run of its own: the row is drawn before the simulator boots. Watched like a chunk's run, and held to a
        # first write as well -- the directory holds the earlier rows' files, which say nothing of this run.
        return self.run(
            [self.isaac_python, self.repo / "generation/envs/run_s1a.py", "--scenario", plan.scenario,
             "--seq", seq, "--episodes", 1, "--max-steps", 1000, "--out", chunk / EPISODES_DIR],
            row_log(chunk, seq), cwd=self.repo, watch_dir=chunk / EPISODES_DIR, first_write_s=FIRST_WRITE_S)

    def label(self, batch: Batch, plan: ChunkPlan, chunk: Path):
        return self.run([self.python, self.repo / "generation/labeling/label_episodes.py",
                         "--episodes", chunk / EPISODES_DIR],
                        chunk / LOGS_DIR / "label.log", cwd=self.repo)

    def convert(self, batch: Batch, plan: ChunkPlan, chunk: Path):
        return self.run([self.lerobot_python, self.repo / "generation/convert_lerobot.py",
                         "--episodes", chunk / EPISODES_DIR, "--out", chunk / DATASET_DIR,
                         "--repo-id", batch.repo_id],
                        chunk / LOGS_DIR / "convert.log", cwd=self.repo)

    def validate(self, batch: Batch, plan: ChunkPlan, chunk: Path):
        # No hook: C9 runs once, on the assembled distribution. ERROR 0 is the bar -- with ten
        # episodes the outcome shares swing, and C4's warning about them is for the whole batch.
        return self.run([self.python, self.repo / "validator/validate_dataset.py", chunk / DATASET_DIR,
                         "--json", chunk / LOGS_DIR / "validate.json"],
                        chunk / LOGS_DIR / "validate.log", cwd=self.repo)

    def measure(self, batch: Batch, plan: ChunkPlan, chunk: Path):
        if plan.per_row:
            # The scorecard of the chunk's distribution: what its rows came to, read from what ships. It reads the
            # release with pyarrow alone, so it runs where the labeller and the validator do.
            return self.run([self.python, self.repo / "generation/scorecard.py",
                             "--dataset", chunk / DATASET_DIR, "--out", chunk / LOGS_DIR / "measure.json"],
                            chunk / LOGS_DIR / "measure.log", cwd=self.repo)
        # What C9 does not compare: the sidecar's facing against the drawn angle, the realised
        # entry speed against the drawn one. Reads the lossless streams, so before the seal.
        return self.run([self.lerobot_python, self.repo / "generation/scene/measure_slice.py",
                         "--episodes", chunk / EPISODES_DIR, "--log", chunk / LOGS_DIR / "generate.log",
                         "--dataset", chunk / DATASET_DIR, "--out", chunk / LOGS_DIR / "measure.json"],
                        chunk / LOGS_DIR / "measure.log", cwd=self.repo)


def check_generated(batch: Batch, plan: ChunkPlan, chunk: Path) -> list[str]:
    """The chunk's episode ids, after checking they are the ones asked for: every id of the
    chunk's rows and no other, each sidecar naming the file it sits beside, each recorded in the
    run the plan says (ChunkPlan.run_start) by the batch's commit."""
    want = batch.episode_ids(plan)
    got = sorted(p.stem for p in (chunk / EPISODES_DIR).glob("*.npz"))
    if got != sorted(want):                 # the rows of a chunk come in the order they were asked, not sorted
        raise ChunkFailed("generate", f"에피소드가 요청과 다릅니다 — 기대 {len(want)}건 "
                                      f"({want[0]} … {want[-1]}), 실제 {len(got)}건 {got[:3]}")
    for seq, eid in zip(plan.rows, want):
        side = json.loads((chunk / EPISODES_DIR / f"{eid}.json").read_text(encoding="utf-8"))
        backend = side.get("sim_backend") if isinstance(side.get("sim_backend"), dict) else {}
        if side.get("episode_id") != eid:
            raise ChunkFailed("generate", f"{eid}: 사이드카의 episode_id={side.get('episode_id')!r} 가 "
                                          "파일 이름과 다릅니다")
        start = plan.run_start(seq)
        if side.get("run_start_seq") != start:
            began = f"자기 행 {start}" if plan.per_row else f"청크의 시작 {start}"
            raise ChunkFailed("generate", f"{eid}: run_start_seq={side.get('run_start_seq')!r} 가 "
                                          f"{began} 가 아닙니다")
        if backend.get(provenance.COMMIT) != batch.generator_commit:
            raise ChunkFailed("generate", f"{eid}: generator_commit={backend.get(provenance.COMMIT)!r} 가 "
                                          f"배치의 {batch.generator_commit} 가 아닙니다")
        if backend.get(provenance.DIRTY) is not False:
            raise ChunkFailed("generate", f"{eid}: generator_dirty={backend.get(provenance.DIRTY)!r} — "
                                          "생성 도중 작업 트리가 바뀌었습니다")
    return want


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _has_bytes(path: Path) -> bool:
    """A file with something in it. A file of none is what a hard reset leaves of one whose bytes had not reached
    the disk."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _recorded_cameras() -> list[str]:
    """The cameras whose lossless streams a recording names in its sidecar's media: those of the camera canon,
    as media_writer.MediaWriter writes them."""
    return list(json.loads((REPO / "meta" / "cameras.json").read_text(encoding="utf-8"))["cameras"])


def row_problem(batch: Batch, plan: ChunkPlan, seq: int, chunk: Path) -> str | None:
    """Why row ``seq`` of a chunk of condition-table rows is not a finished row; None when it is.

    A finished row is what the run of the row leaves when it ends well: its npz, with bytes; its sidecar, read as a
    JSON object that names the row's episode id, the row itself as its run_start_seq, the batch's commit and a clean
    tree; and the files its media names -- one lossless stream for every camera of the canon, and any further stream
    it names -- each there with bytes. The generator writes the npz, then the sidecar, then puts the streams in
    place (run_s1a.py), so a run cut off between them leaves a row this says is not finished."""
    eid = batch.episode_ids(plan)[plan.rows.index(seq)]
    episodes = chunk / EPISODES_DIR
    if not _has_bytes(episodes / f"{eid}.npz"):
        return "npz 가 없거나 비었습니다"
    try:
        side = json.loads((episodes / f"{eid}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "사이드카가 없거나 JSON 으로 읽히지 않습니다"
    if not isinstance(side, dict):
        return "사이드카가 JSON 객체가 아닙니다"
    backend = side.get("sim_backend") if isinstance(side.get("sim_backend"), dict) else {}
    start = plan.run_start(seq)
    if side.get("episode_id") != eid:
        return f"사이드카의 episode_id={side.get('episode_id')!r} 가 {eid} 가 아닙니다"
    if not _int(side.get("run_start_seq")) or side.get("run_start_seq") != start:
        return f"사이드카의 run_start_seq={side.get('run_start_seq')!r} 가 자기 행 {start} 가 아닙니다"
    if backend.get(provenance.COMMIT) != batch.generator_commit:
        return f"사이드카의 generator_commit={backend.get(provenance.COMMIT)!r} 가 배치의 커밋이 아닙니다"
    if backend.get(provenance.DIRTY) is not False:
        return f"사이드카의 generator_dirty={backend.get(provenance.DIRTY)!r} 입니다"
    media = side.get("media")
    if not isinstance(media, dict):
        return "사이드카에 media 가 없습니다"
    named = {key: entry.get("file") for key, entry in media.items() if isinstance(entry, dict) and "file" in entry}
    unnamed = [cam for cam in _recorded_cameras() if cam not in named]
    if unnamed:
        return f"사이드카의 media 가 카메라 {unnamed} 의 영상을 적지 않았습니다"
    for key, rel in named.items():
        if not isinstance(rel, str) or not rel or not _has_bytes(episodes / rel):
            return f"media.{key} 의 파일 {rel!r} 이 없거나 비었습니다"
    return None


def row_done(batch: Batch, plan: ChunkPlan, seq: int, chunk: Path) -> bool:
    """Whether row ``seq`` of a chunk of condition-table rows is finished (row_problem)."""
    return row_problem(batch, plan, seq, chunk) is None


def _of(eid: str, name: str) -> bool:
    """Whether a file named ``name`` is episode ``eid``'s: <eid>.* and the hidden .<eid>.* a run writes before it is
    done (an id holds no dot, so no other episode's name starts so)."""
    return name.startswith((f"{eid}.", f".{eid}."))


def clear_row(chunk: Path, eid: str) -> None:
    """Remove what the chunk holds of episode ``eid`` -- its npz, sidecar and labels (every file of episodes/ that is
    its) and its lossless streams, put in place or not (media_writer.clear_media) -- so the row is made from nothing.
    The other rows' files stay. ChunkFailed when a file cannot be removed."""
    episodes = chunk / EPISODES_DIR
    try:
        for p in episodes.iterdir():
            if _of(eid, p.name) and not p.is_dir():
                p.unlink()
        clear_media(episodes, eid)
    except OSError as e:
        raise ChunkFailed("clean", f"{eid} 의 파일을 지우지 못했습니다 — {type(e).__name__}: {e}") from e


def _clear_other_files(chunk: Path, eids) -> None:
    """Remove the files of episodes/ and of its media/ that are no episode's of the chunk."""
    try:
        for where in (chunk / EPISODES_DIR, chunk / EPISODES_DIR / MEDIA_DIR):
            if where.is_dir():
                for p in where.iterdir():
                    if not p.is_dir() and not any(_of(eid, p.name) for eid in eids):
                        p.unlink()
    except OSError as e:
        raise ChunkFailed("clean", f"{chunk / EPISODES_DIR} 의 파일을 지우지 못했습니다 — {type(e).__name__}: {e}") from e


def wipe_downstream(chunk: Path) -> None:
    """Remove what the stages after generation left in a chunk of rows: its dataset/, the labels in episodes/ and
    every log but the rows' own (generate-*.log). Those stages run again on all the rows."""
    logs = chunk / LOGS_DIR
    try:
        if (chunk / DATASET_DIR).exists():
            shutil.rmtree(chunk / DATASET_DIR)
        for p in (chunk / EPISODES_DIR).glob("*.labels.json"):
            p.unlink()
        for p in logs.iterdir() if logs.is_dir() else ():
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            elif not (p.name.startswith("generate-") and p.name.endswith(".log")):
                p.unlink()
    except OSError as e:
        raise ChunkFailed("clean", f"{chunk} 에서 생성 뒤 단계의 산출물을 지우지 못했습니다 — {type(e).__name__}: {e}") from e


def make_rows(batch: Batch, plan: ChunkPlan, stages, chunk: Path, *, say=progress) -> dict[str, float]:
    """Generate the rows of a chunk of condition-table rows that are not finished, one run each; the seconds the
    run that made each row took, by row, for the rows made here (a row an earlier run finished is not among them).

    Before any row: the products of the stages after generation go (wipe_downstream), and so does every file
    that is no row's of the chunk. A finished row's files (row_problem) are left as they are. Every other row gets
    up to GENERATE_RETRIES + 1 attempts, each from nothing (clear_row); a stalled attempt's log is kept aside. ChunkFailed, naming the row, when every attempt stalled, when the run exits with
    another code than 0, or when it exits 0 and leaves no finished row."""
    episodes = chunk / EPISODES_DIR
    try:
        episodes.mkdir(parents=True, exist_ok=True)
        (chunk / LOGS_DIR).mkdir(exist_ok=True)
    except OSError as e:
        raise ChunkFailed("clean", f"{chunk} 를 만들지 못했습니다 — {type(e).__name__}: {e}") from e
    wipe_downstream(chunk)
    ids = dict(zip(plan.rows, batch.episode_ids(plan)))
    finished = []
    for seq, eid in ids.items():
        problem = row_problem(batch, plan, seq, chunk)
        if problem is None:
            finished.append(seq)
            continue
        left = [p for where in (episodes, episodes / MEDIA_DIR) if where.is_dir()
                for p in where.iterdir() if _of(eid, p.name)]
        if left:
            say(f"  행 {seq}: 끝난 행이 아닙니다 ({problem}) — 남은 파일 {len(left)}개를 지우고 다시 만듭니다")
    _clear_other_files(chunk, list(ids.values()))
    if finished:
        say(f"  끝난 행 {len(finished)}개는 그대로 둡니다: {finished}")
    todo = [seq for seq in plan.rows if seq not in finished]
    made: dict[str, float] = {}
    for n, seq in enumerate(todo, 1):
        for attempt in range(GENERATE_RETRIES + 1):
            clear_row(chunk, ids[seq])
            began = time.monotonic()
            code = stages.generate_row(batch, plan, seq, chunk)
            took = round(time.monotonic() - began, 1)
            if code != STALLED:
                break
            say(f"  행 {seq} 의 생성이 멈췄습니다 ({attempt + 1}/{GENERATE_RETRIES + 1})")
            keep_stalled_log(batch, plan, attempt + 1, chunk, say, seq=seq)
        else:
            raise ChunkFailed("generate", f"행 {seq}: {GENERATE_RETRIES + 1}번 모두 멈췄습니다")
        if code != 0:
            raise ChunkFailed("generate", f"행 {seq}: 종료 코드 {code}")
        problem = row_problem(batch, plan, seq, chunk)
        if problem is not None:
            raise ChunkFailed("generate", f"행 {seq}: 생성은 0 으로 끝났는데 끝난 행이 아닙니다 — {problem}")
        # The row is counted as finished from here on: its files on disk now, so a hard reset leaves it whole
        # rather than cut short with bytes enough to pass row_problem on the next run.
        os.sync()
        made[str(seq)] = took
        say(f"  행 {seq} ({n}/{len(todo)}) 생성 {took:g}s")
    return made


def check_tree(batch: Batch, read_tree, when: str) -> None:
    """The working tree is the batch's commit, clean -- read again, not remembered from the start. The
    generator read the tree once, at its own start; labelling, conversion, validation and measurement
    run on whatever the tree is when they run, and nothing downstream re-derives the labels, so a
    canon or labeller edited meanwhile would be sealed under the batch's commit."""
    state = read_tree()
    commit, dirty = state.get(provenance.COMMIT), state.get(provenance.DIRTY)
    if commit is None or dirty is None:
        raise ChunkFailed("tree", f"{when}: git 이 작업 트리의 상태를 말하지 못했습니다")
    if commit != batch.generator_commit:
        raise ChunkFailed("tree", f"{when}: HEAD 가 {commit} 입니다 — 배치의 커밋 {batch.generator_commit} 이 아닙니다")
    if dirty is not False:
        raise ChunkFailed("tree", f"{when}: 작업 트리에 커밋되지 않은 변경이 있습니다 (HEAD {commit})")


def wipe(chunk: Path) -> None:
    """Remove the chunk's directory, and stop when that fails: an attempt cannot be made on top of what a
    dead run left, and a swallowed failure shows up as a bare FileExistsError at the next mkdir."""
    try:
        if chunk.exists():
            shutil.rmtree(chunk)
    except OSError as e:
        raise ChunkFailed("clean", f"{chunk} 를 지우지 못했습니다 — {type(e).__name__}: {e}") from e


def keep_stalled_log(batch: Batch, plan: ChunkPlan, attempt: int, chunk: Path, say, seq: int | None = None) -> None:
    """Copy a stalled attempt's log aside before the retry wipes or overwrites it: why it stalled is the
    only evidence there is. ``seq``: the row whose run stalled, in a chunk of condition-table rows. A name an
    earlier run of the batch kept -- its attempts are counted from one as well -- is not overwritten: the copy
    takes the next number, <name>.attempt<n>.<k>.generate.log for k = 2, 3, ..."""
    if seq is None:
        stem = f"{plan.key}.attempt{attempt}"
        log = chunk / LOGS_DIR / "generate.log"
    else:
        stem = f"{plan.key}.{seq:04d}.attempt{attempt}"
        log = row_log(chunk, seq)
    dest = batch.root / STALLED_LOGS / f"{stem}.generate.log"
    try:
        dest.parent.mkdir(exist_ok=True)
        k = 1
        while dest.exists() or dest.is_symlink():
            k += 1
            dest = dest.with_name(f"{stem}.{k}.generate.log")
        shutil.copy2(log, dest)
    except OSError as e:
        say(f"  멈춘 시도의 로그를 옮기지 못했습니다 — {type(e).__name__}: {e}")
        return
    say(f"  멈춘 시도의 로그를 {dest} 에 남겼습니다")


def _read_report(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _finite(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def validation_summary(chunk: Path) -> dict | None:
    """The validator's counts of findings by severity, from its --json report; None when it cannot be read."""
    summary = (_read_report(chunk / LOGS_DIR / "validate.json") or {}).get("summary")
    if not isinstance(summary, dict):
        return None
    counts = {k: summary.get(k) for k in ("error", "warn", "info", "skip")}
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in counts.values()):
        return None
    return counts


GRID_OUTCOMES = ("fail", "near_miss", "nominal")


def _card_summary(chunk: Path, scenario: str) -> dict | None:
    """The parts of the scorecard (scorecard.py --out) of a chunk of rows that a seal keeps: the counts of the three
    outcomes (degenerate episodes left out, as the card leaves them), of degenerate episodes, of robot falls and of
    rack and floor contact events -- all over the random rows, as the card counts them. When the chunk holds grid
    rows, ``grid`` adds what they came to, counted off the card's failure map: the grid rows, the three outcomes and
    the degenerate ones. None when the card cannot be read or does not count them for ``scenario``, or when its
    failure map's cells do not add up to its number of grid rows."""
    scenarios = (_read_report(chunk / LOGS_DIR / "measure.json") or {}).get("scenarios")
    card = scenarios.get(scenario) if isinstance(scenarios, dict) else None
    if not isinstance(card, dict):
        return None
    safety, anomalies = card.get("safety"), card.get("anomalies")
    outcome = safety.get("outcome") if isinstance(safety, dict) else None
    if not isinstance(outcome, dict) or not isinstance(anomalies, dict):
        return None
    counts = {"degenerate": card.get("degenerate_excluded"), "robot_falls": anomalies.get("robot_falls"),
              "rack_contact_events": anomalies.get("rack_contact_events")}
    if not all(_int(v) for v in (*counts.values(), *outcome.values())):
        return None
    summary = {"outcome": dict(outcome), **counts}
    grid_rows = card.get("grid_episodes")
    if not _int(grid_rows) or grid_rows < 0:       # the scorecard always writes the count: without it, it is unreadable
        return None
    grid = _grid_summary(card.get("failure_map"), grid_rows)
    if grid is None:
        return None
    return {**summary, "grid": grid} if grid_rows else summary


def _grid_summary(failure_map, grid_rows: int) -> dict | None:
    """What the grid rows came to, counted off the cells of the scorecard's failure map. One rule: the map's cells
    number ``grid_rows``. A map of no cells is None or the scorecard's ``{"absent": ...}``; otherwise it is a dict whose
    ``outcome`` is a list of lists and every cell that is not None names one of the four kinds. None when the map
    is anything else, a cell is of no known kind, or the cells are not ``grid_rows`` many."""
    if failure_map is None or (isinstance(failure_map, dict) and "absent" in failure_map):
        cells = []
    else:
        rows = failure_map.get("outcome") if isinstance(failure_map, dict) else None
        if not isinstance(rows, list) or not all(isinstance(r, list) for r in rows):
            return None
        cells = [c for r in rows for c in r if c is not None]
    kinds = (*GRID_OUTCOMES, "degenerate")
    if len(cells) != grid_rows or not all(isinstance(c, str) and c in kinds for c in cells):
        return None
    return {"episodes": grid_rows, "outcome": {k: cells.count(k) for k in GRID_OUTCOMES},
            "degenerate": cells.count("degenerate")}


def measurement_summary(chunk: Path, plan: ChunkPlan) -> dict | None:
    """What a seal keeps of the chunk's measurement. A chunk of rows: the scorecard's counts (_card_summary).
    A chunk that is one run: the parts of measure_slice.py's summary that say how the entry went -- the realised ÷
    drawn speed over all entries and over the held ones, the largest facing deviation, the outcomes. Numbers only
    if finite (the report's "range" holds Infinity, which a seal must not). None when it cannot be read."""
    if plan.per_row:
        return _card_summary(chunk, plan.scenario)
    summary = (_read_report(chunk / LOGS_DIR / "measure.json") or {}).get("summary")
    if not isinstance(summary, dict):
        return None

    def spread(x):
        if not isinstance(x, dict):
            return None
        return {"n": x.get("n"), "min": _finite(x.get("min")), "median": _finite(x.get("median")),
                "max": _finite(x.get("max"))}

    outcome = summary.get("outcome")
    return {"entry_triggered": summary.get("entry_triggered"),
            "entry_speed_ratio": spread(summary.get("entry_speed_ratio")),
            "entry_speed_ratio_held": spread(summary.get("entry_speed_ratio_held")),
            "facing_off_travel_deg_max": _finite(summary.get("facing_off_travel_deg_max")),
            "outcome": dict(outcome) if isinstance(outcome, dict) else None}


def seal_line(key: str, record: dict) -> str:
    """The progress line of a sealed chunk: its stage times and the key numbers of its seal."""
    line = f"청크 {key} 봉인 — " + " · ".join(f"{k} {v:g}s" for k, v in record["stages_s"].items())
    v, m = record.get("validation"), record.get("measurement")
    line += (f" · 검증 오류 {v['error']} · 경고 {v['warn']}" if v
             else " · 검증 요약 없음 (logs/validate.json 을 읽지 못함)")
    if m is None:
        if "rows_s" in record:              # a chunk of rows: the file may have been read and its grid rows not counted
            return line + " · 측정 요약 없음 (logs/measure.json 을 읽지 못했거나 격자 행을 셀 수 없음)"
        return line + " · 측정 요약 없음 (logs/measure.json 을 읽지 못함)"
    if "rows_s" in record:                  # a chunk of rows: the scorecard's counts, random rows and grid rows apart
        random_rows = sum(m["outcome"].values()) + m["degenerate"]
        grid = m.get("grid")
        if random_rows or grid is None:
            line += (f" · 무작위 행 {random_rows} — " + " · ".join(f"{k} {n}" for k, n in m["outcome"].items())
                     + f" · degenerate {m['degenerate']} · 로봇 넘어짐 {m['robot_falls']}"
                     + f" · 랙 · 바닥 접촉 {m['rack_contact_events']}번")
        if grid is not None:
            line += (f" · 격자 행 {grid['episodes']} — " + " · ".join(f"{k} {n}" for k, n in grid["outcome"].items())
                     + f" · degenerate {grid['degenerate']}")
        return line

    def span(x):
        return f"{x['min']:.4f}~{x['max']:.4f}" if x and x["min"] is not None and x["max"] is not None else "없음"

    line += f" · 진입 {m['entry_triggered']}건 속도 비 {span(m['entry_speed_ratio'])}"
    if m["entry_speed_ratio_held"]:
        line += f" (붙잡힌 것 {span(m['entry_speed_ratio_held'])})"
    if m["facing_off_travel_deg_max"] is not None:
        line += f" · 방향 편차 최대 {m['facing_off_travel_deg_max']:.2e}°"
    if m["outcome"]:
        line += " · 라벨 " + " · ".join(f"{k} {n}" for k, n in m["outcome"].items())
    return line


def run_chunk(batch: Batch, plan: ChunkPlan, stages, *, read_tree, say=progress) -> dict:
    """Make one chunk and seal it; its seal record. ChunkFailed says which stage.

    A chunk that is one run is made from nothing. A chunk of condition-table rows goes on from its finished rows
    (make_rows); the stages after generation run on all of them. The seal record names the chunk's scenario and
    rows, and for a chunk of rows how long the run of each row made here took (``rows_s``, by row).

    The tree is read before the chunk starts and again just before its seal is written (check_tree).
    A chunk that cannot be sealed as it is (batch_layout.sealed_files) fails at the stage "seal".
    The lossless media stays when a stage fails -- a person looks at it -- and goes only after
    the seal is written: an interruption between the two leaves a sealed chunk with media, which
    the next run finishes (run)."""
    chunk = batch.chunk_dir(plan)
    took: dict[str, float] = {}
    rows_s: dict[str, float] | None = None
    check_tree(batch, read_tree, "청크를 시작하기 전")
    if plan.per_row:
        began = time.monotonic()
        rows_s = make_rows(batch, plan, stages, chunk, say=say)
        took["generate"] = round(time.monotonic() - began, 1)
    else:
        code = _generate_chunk(batch, plan, stages, chunk, took, say)
        if code != 0:
            raise ChunkFailed("generate", f"종료 코드 {code}")
    ids = check_generated(batch, plan, chunk)
    for name in AFTER_GENERATE:
        began = time.monotonic()
        code = getattr(stages, name)(batch, plan, chunk)
        # Exit 3 is the validator's own crash; a negative code is the validator killed by a signal (the
        # segfault of the known flake). Neither is a finding.
        if name == "validate" and isinstance(code, int) and (code == VALIDATOR_CRASHED or code < 0):
            log = chunk / LOGS_DIR / "validate.log"
            if log.exists():
                shutil.copy2(log, chunk / LOGS_DIR / "validate.crash.log")
            say(f"  검증기가 죽었습니다(종료 코드 {code}) — 로그를 validate.crash.log 로 두고 한 번 다시 돌립니다")
            code = stages.validate(batch, plan, chunk)
        took[name] = round(time.monotonic() - began, 1)
        if code != 0:
            raise ChunkFailed(name, f"종료 코드 {code}")
    # The chunk's files on disk before the record that vouches for them; then the tree once more, as
    # late as it can be read.
    os.sync()
    try:
        files = sealed_files(chunk)
    except BatchError as e:                 # a chunk that cannot be sealed as it is: said as the chunk's failure
        raise ChunkFailed("seal", str(e)) from e
    check_tree(batch, read_tree, "봉인하기 직전")
    record = {"scenario": plan.scenario, "rows": list(plan.rows), "start_seq": plan.rows[0], "episodes": len(ids),
              "episode_ids": ids, "generator_commit": batch.generator_commit, "stages_s": took}
    if rows_s is not None:
        record["rows_s"] = rows_s
    record.update(validation=validation_summary(chunk), measurement=measurement_summary(chunk, plan),
                  sealed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), files=files)
    write_seal(chunk, record)
    remove_media(chunk)
    return record


def _generate_chunk(batch: Batch, plan: ChunkPlan, stages, chunk: Path, took: dict, say) -> int:
    """Generate a chunk that is one run, from nothing, up to GENERATE_RETRIES + 1 times while it stalls; the code of
    the last attempt. ``took["generate"]`` is that attempt's seconds."""
    for attempt in range(GENERATE_RETRIES + 1):
        wipe(chunk)
        (chunk / EPISODES_DIR).mkdir(parents=True)
        (chunk / LOGS_DIR).mkdir()
        began = time.monotonic()
        code = stages.generate(batch, plan, chunk)
        took["generate"] = round(time.monotonic() - began, 1)
        if code != STALLED:
            return code
        say(f"  생성이 멈췄습니다 ({attempt + 1}/{GENERATE_RETRIES + 1})")
        keep_stalled_log(batch, plan, attempt + 1, chunk, say)
    raise ChunkFailed("generate", f"{GENERATE_RETRIES + 1}번 모두 멈췄습니다")


def run(batch: Batch, stages, *, read_tree, say=progress) -> int:
    """Every chunk not yet sealed, in the batch's order (Batch.chunks). 0 when all are sealed or a STOP file
    ended the run, 1 when a chunk failed. ``read_tree`` is what check_tree reads the working tree with. STOP is
    looked for between chunks: a chunk of rows is finished, not left between two of its rows."""
    plans = batch.chunks()
    for n, plan in enumerate(plans, 1):
        chunk = batch.chunk_dir(plan)
        if read_seal(chunk) is not None:
            if remove_media(chunk):
                say(f"청크 {plan.key}: 봉인은 됐는데 무손실 영상이 남아 있어 지웠습니다")
            continue
        if (batch.root / STOP_FILE).exists():
            say(f"{STOP_FILE} 파일이 있어 청크 {plan.key} 앞에서 멈춥니다")
            return 0
        say(f"청크 {plan.key} ({n}/{len(plans)}) — "
            + (f"{plan.scenario} 행 {len(plan.rows)}개 (행마다 실행 하나)" if plan.per_row
               else f"에피소드 {len(plan.rows)}건"))
        try:
            record = run_chunk(batch, plan, stages, read_tree=read_tree, say=say)
        except ChunkFailed as e:
            where = f" 로그는 {chunk / LOGS_DIR}," if chunk.exists() else ""
            media = chunk / EPISODES_DIR / MEDIA_DIR
            kept = f" 무손실 영상은 {media} 에 남겼습니다." if media.exists() else ""
            hint = (" 작업 트리를 배치의 커밋의 깨끗한 상태로 돌려 놓고 다시 돌립니다. 코드를 고쳐야 한다면 "
                    "이 배치는 그 커밋으로 끝낼 수 없습니다 — 새 배치로 시작하십시오" if e.stage == "tree" else "")
            again = ("다시 돌리면 끝난 행은 그대로 두고 끝나지 않은 행부터 이어 갑니다 (생성 뒤 단계는 처음부터)."
                     if plan.per_row else "다시 돌리면 이 청크를 지우고 처음부터 합니다.")
            say(f"청크 {plan.key} 실패 — {e.stage}: {e.detail}.{where}{kept} {again}{hint}")
            return 1
        say(seal_line(plan.key, record))
    say(f"모든 청크가 봉인됐습니다 ({len(plans)}개). 다음: generation/assemble_batch.py --batch {batch.root}")
    return 0


class Lock:
    """One runner per batch: ``flock`` on <batch>/LOCK for the whole run, and its record in <batch>/RUNNER.

    The lock is the kernel's: exclusive and non-blocking, taken in one call, and let go of when the process
    holding it dies however it dies -- there is no stale lock to reason about and no PID to compare. (Looking for
    a lock file and then writing one let two runners both find none, both write it and both go on to wipe and
    generate the same chunk.) The file itself is never removed: unlinking it would let a runner that opened the
    old file lock a file nobody else sees. Stages do not inherit the descriptor (it is close-on-exec).

    The record -- the runner's PID and, while a stage runs, the stage's process group -- is a separate file,
    replaced atomically, so a reader never sees it empty. A stage runs in a session of its own, so it outlives a
    runner that is killed -- and keeps writing into the chunk the next run is about to remove. The next run
    ends it first, by the group id recorded here: if some member of that group still runs (not only the leader:
    the generator's leader is a bash wrapper that can be gone while the simulator is not) and a member's
    command line names this batch -- a PID is reused, and one that merely matches is not ours. The record
    survives a run that left a group behind, so that the run after finds it.

    The record names a group only while a stage runs: after a stage that ended normally it is told None, so
    what a normal stage leaves in its group on purpose -- the shared Omniverse hub -- is never ended by a later
    run. If the runner itself is killed *during* a stage whose group holds the hub, the next run ends that
    group, hub included; that is the abnormal path and accepted."""

    def __init__(self, batch_root: Path, say=progress, kill_grace_s: float = STALL_KILL_GRACE_S):
        self.root, self.say, self.kill_grace_s = Path(batch_root), say, kill_grace_s
        self.path, self.record_path = self.root / LOCK_FILE, self.root / RUNNER_FILE
        self._fd, self._child = None, None

    def _read(self) -> dict:
        try:
            doc = json.loads(self.record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return doc if isinstance(doc, dict) else {}

    def _write(self, child) -> None:
        self._child = child
        write_atomically(self.record_path, json.dumps({"pid": os.getpid(), "child": child}))

    def child(self, pgid) -> None:
        self._write(pgid)

    def _end_left_stage(self) -> None:
        child = self._read().get("child")
        if not isinstance(child, int) or isinstance(child, bool) or child <= 1 or child == os.getpgrp():
            return                     # never a group of ours, nor "every process" (0, 1)
        if not group_running(child):
            return
        if not any(str(self.root) in cmdline(pid) for pid in group_members(child) or []):
            return
        self.say(f"앞 실행기가 남긴 단계 프로세스 그룹 {child} 를 끝냅니다")
        end_group_by_id(child, self.kill_grace_s)

    def __enter__(self):
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    raise
                pid = self._read().get("pid")
                who = f"PID {pid}" if isinstance(pid, int) and alive(pid) else "PID 를 기록에서 읽지 못했습니다"
                raise BatchError(f"다른 실행기가 이 배치를 돌리고 있습니다 ({who}, {self.path}). "
                                 "그 실행기가 끝나면 잠금은 저절로 풀립니다") from None
            self._end_left_stage()
            self._write(None)
        except BaseException:
            os.close(fd)               # gives the lock back; the record keeps whatever it named
            raise
        self._fd = fd
        return self

    def __exit__(self, *exc) -> None:
        try:
            if self._child is None:    # a stage that could not be ended stays on record for the next run
                self.record_path.unlink(missing_ok=True)
        finally:
            os.close(self._fd)
            self._fd = None


def stage_runner(lock: Lock):
    """What ``Stages`` runs its commands with: ``run_stage``, telling ``lock`` the stage's process
    group while the stage runs -- the only way a later run learns of a stage this one left behind."""
    def run(argv, log, **kw):
        return run_stage(argv, log, on_start=lock.child, **kw)
    return run


S1D_EPISODES = 400
S1D_REPO_ID = "keti/humanoid-safety-s1d"
TABLE_REPO_ID = "keti/keti-humanoid-hri-safety"


class UsageError(ValueError):
    """Arguments that do not make one batch. The message says which."""


def batch_parts(scenarios, rows_path, episodes) -> tuple[tuple[tuple[str, tuple], ...], str]:
    """The batch's parts and its default repository from the command line: S1d alone with its number of episodes
    (rows 0..n-1), or condition-table scenarios with the rows ``rows_path`` names for each -- a JSON object
    {scenario: [row, ...]} whose keys are the scenarios asked for -- or, without it, every row of each one's table.
    The parts follow the order the scenarios are asked in. UsageError for a combination that is not one batch."""
    tables = set(conditions.table_scenarios())
    twice = sorted({s for s in scenarios if scenarios.count(s) > 1})
    if twice:
        raise UsageError(f"--scenario 에 같은 시나리오가 두 번 있습니다: {twice}")
    if {s in tables for s in scenarios} == {True, False}:
        raise UsageError(f"--scenario {list(scenarios)}: 조건표 시나리오와 S1d 를 한 배치에 섞지 않습니다 — 둘은 청크를 "
                         "만드는 방식이 다릅니다")
    if scenarios[0] not in tables:
        if rows_path is not None:
            raise UsageError("--rows 는 조건표 시나리오의 행입니다 — S1d 의 배치는 --episodes 로 정합니다")
        return ((scenarios[0], tuple(range(S1D_EPISODES if episodes is None else episodes))),), S1D_REPO_ID
    if episodes is not None:
        raise UsageError("--episodes 는 S1d 의 것입니다 — 조건표 시나리오의 행은 --rows 로 고릅니다 (없으면 조건표의 행 전부)")
    if rows_path is None:
        try:
            rows = {s: list(range(conditions.load_conditions(conditions.scenario_entry(s)).total)) for s in scenarios}
        except ValueError as e:
            raise UsageError(f"조건표를 읽지 못했습니다: {e}") from e
    else:
        try:
            rows = json.loads(Path(rows_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise UsageError(f"--rows {rows_path} 를 JSON 으로 읽지 못했습니다: {type(e).__name__}: {e}") from e
        if not isinstance(rows, dict):
            raise UsageError(f"--rows {rows_path} 가 {{시나리오: [행, …]}} 꼴의 JSON 객체가 아닙니다")
        if set(rows) != set(scenarios):
            raise UsageError(f"--rows 의 시나리오 {sorted(rows)} 가 --scenario {sorted(scenarios)} 와 다릅니다")
        bad = [s for s in scenarios if not isinstance(rows[s], list)]
        if bad:
            raise UsageError(f"--rows 의 {bad} 의 값이 행의 목록이 아닙니다")
    return tuple((s, tuple(rows[s])) for s in scenarios), TABLE_REPO_ID


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, type=Path, help="배치 디렉터리 (예: .tmp/batches/S1d-400)")
    ap.add_argument("--scenario", nargs="+", default=["S1d"], choices=("S1d", *conditions.table_scenarios()),
                    metavar="S", help="S1d 하나, 또는 조건표 시나리오 여럿 (S1e S1f S1g — 청크가 시나리오를 번갈아 돎)")
    ap.add_argument("--rows", type=Path,
                    help="조건표 시나리오의 행: JSON {시나리오: [행, …]} — 키가 --scenario 와 같아야 함. "
                         "없으면 조건표의 행 전부")
    ap.add_argument("--episodes", type=int, help=f"S1d 배치의 에피소드 수 (기본 {S1D_EPISODES})")
    ap.add_argument("--chunk", type=int, default=10,
                    help="청크의 에피소드 수 (기본 10 — C9 훅이 제한 시간 안에 다시 만드는 범위)")
    ap.add_argument("--repo-id", help=f"기본: S1d 는 {S1D_REPO_ID}, 조건표 시나리오는 {TABLE_REPO_ID}")
    ap.add_argument("--isaac-python", default=str(Path.home() / "isaacsim" / "python.sh"))
    ap.add_argument("--python", default=str(REPO / ".venv" / "bin" / "python"))
    ap.add_argument("--lerobot-python", default=str(REPO / ".venv-lerobot" / "bin" / "python"))
    args = ap.parse_args(argv)
    try:
        parts, repo_id = batch_parts(args.scenario, args.rows, args.episodes)
    except UsageError as e:
        print(e, file=sys.stderr)
        return 2

    state = provenance.read(REPO)
    said = provenance.problems(state)
    if said:
        print("배치는 커밋된 깨끗한 트리에서만 돌립니다 — " + "; ".join(said)
              + " (`git status --short` 가 비어 있어야 합니다)", file=sys.stderr)
        return 2
    try:
        # Resolved: the lock looks for this path in a left-over stage's command line, and a
        # relative spelling in one run would not match an absolute one in the next.
        batch = open_batch(args.batch.resolve(), parts=parts, chunk_episodes=args.chunk,
                           generator_commit=state[provenance.COMMIT], repo_id=args.repo_id or repo_id)
        with Lock(batch.root) as lock:
            stages = Stages(repo=REPO, isaac_python=args.isaac_python, python=args.python,
                            lerobot_python=args.lerobot_python,
                            run=stage_runner(lock))
            return run(batch, stages, read_tree=lambda: provenance.read(REPO))
    except BatchError as e:
        print(e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
