"""Measure the stop-time distribution the R-10218-STOP-01 thresholds come from.

Isaac-free: reads a recorded npz batch and nothing else, so it runs in the
validator venv.

What it measures is the time from the
protective-stop onset step to the *first* step whose |v_x| falls at or below
a candidate threshold. Deliberately not the strict "and never rises again"
criterion: that held in 0 of 11 observed stops, because once the human has
crossed, separation recovers and the scenario legitimately commands speed
again. A re-rise is SSM returning to service, not a stop that failed.

Prints two things and picks nothing. stop_speed_threshold is chosen from an
empty gap in the trough histogram -- the same method that settled
_STANCE_FORCE_THRESHOLD_N and _PIN_HEIGHT_M -- and the time budget follows
from whichever threshold that is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

# Run as a script, sys.path[0] is this file's directory, not the repo root, so
# the `generation.*` imports below fail. run_s1a.py already carries the same
# guard; without it the documented commands need a PYTHONPATH the
# documentation never mentions (measured: all four failed).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generation.field_map import step_dt_from_timestamps


def stop_onsets(events: np.ndarray, names: list[str]) -> list[int]:
    """Steps where protective_stop_triggered fired (an edge, so already onsets)."""
    col = names.index("protective_stop_triggered")
    return np.flatnonzero(np.asarray(events)[:, col]).tolist()


def time_to_below(speed: np.ndarray, onset: int, threshold: float, dt: float) -> float | None:
    """Seconds from onset to the first step at or below ``threshold``. None if never."""
    after = np.flatnonzero(np.abs(speed[onset:]) <= threshold)
    return None if after.size == 0 else float(after[0]) * dt


def _histogram(values: np.ndarray, bins: int = 20) -> str:
    hist, edges = np.histogram(values, bins=bins)
    scale = max(1, hist.max()) / 40.0
    lines = []
    for n, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * int(round(n / scale)) if n else ""
        mark = "" if n else "   <- 빈 구간"
        lines.append(f"  [{lo:7.4f}, {hi:7.4f})  {n:4d} {bar}{mark}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="정지 시간 분포 측정")
    ap.add_argument("--episodes", default=".tmp/episodes/S1a")
    ap.add_argument("--candidates", type=float, nargs="*",
                    default=[0.02, 0.05, 0.10, 0.15, 0.20, 0.30])
    args = ap.parse_args()

    per_onset: list[tuple[str, int, np.ndarray, float]] = []
    joint_tails: list[np.ndarray] = []
    root_tails: list[np.ndarray] = []
    for npz_path in sorted(Path(args.episodes).glob("*.npz")):
        meta = json.loads(npz_path.with_suffix(".json").read_text(encoding="utf-8"))
        d = np.load(npz_path, allow_pickle=True)
        # global_time_offset_s 가 아니다 — 그건 이 에피소드가 전역 시계에서 시작한
        # 시각이고, 0번 에피소드에서만 우연히 한 스텝과 같은 값이 된다.
        dt = step_dt_from_timestamps(d["timestamp"])
        names = list(meta["safety_event_names"])
        vx = np.abs(d["root_vel"][:, 0].astype(float))
        # 관절 속도도 함께 잰다. 2026-08-22 이전에는 정본이
        # stop_speed_threshold 를 rad/s 로 선언했고, 이 축이 정지를 가르지
        # 못한다는 것이 단위를 m/s 로 바꾼 근거였다(rules.json unit_changes).
        # 그 근거가 재현 가능하도록 두 축을 계속 나란히 출력한다.
        jv = d["joint_vel"].astype(float)
        jmax = np.abs(jv).max(axis=1)
        for onset in stop_onsets(d["safety_event"], names):
            per_onset.append((npz_path.stem, onset, vx, dt))
            joint_tails.append(jmax[onset:])
            root_tails.append(vx[onset:])

    if not per_onset:
        print("보호정지 트리거가 한 번도 없었습니다 — 임계값을 정할 근거가 없습니다.")
        return 1

    troughs = np.array([np.abs(vx[onset:]).min() for _, onset, vx, _ in per_onset])
    print(f"보호정지 {len(per_onset)} 건 (에피소드 "
          f"{len({s for s, _, _, _ in per_onset})} 개)")
    print("\n트로프 |v_x| (m/s) — 트리거 이후 최솟값")
    print(f"  min {troughs.min():.4f} · p05 {np.percentile(troughs, 5):.4f} · "
          f"median {np.median(troughs):.4f} · p95 {np.percentile(troughs, 95):.4f} · "
          f"max {troughs.max():.4f}")
    print(_histogram(troughs))

    print("\n후보 임계값별 정지 시간 (s)")
    for cand in args.candidates:
        times = [time_to_below(vx, onset, cand, dt) for _, onset, vx, dt in per_onset]
        reached = [t for t in times if t is not None]
        if not reached:
            print(f"  threshold {cand:.3f}: 도달 0/{len(times)}  <- 쓸 수 없음")
            continue
        arr = np.array(reached)
        flag = "" if len(reached) == len(times) else "   <- 미도달 있음, p95 는 낙관 편향"
        print(f"  threshold {cand:.3f}: 도달 {len(reached)}/{len(times)} · "
              f"median {np.median(arr):.3f} · p95 {np.percentile(arr, 95):.3f} · "
              f"max {arr.max():.3f}{flag}")

    # 정본의 단위는 m/s(루트 전진속도)다. 관절 축을 나란히 두는 것은 그 선택의
    # 근거를 재현 가능하게 남기기 위해서다 -- 관절 속도로는 정지가 갈리지 않는다.
    pooled_root = np.concatenate(root_tails)
    pooled_joint = np.concatenate(joint_tails)
    print("\n트리거 이후 분포 — 어느 축이 정지를 가르는가")
    for label, arr in (("루트 |v_x| (m/s)", pooled_root),
                       ("관절 max|v| (rad/s)", pooled_joint)):
        print(f"\n  {label}  n={arr.size}  범위 {arr.min():.3f}~{arr.max():.3f}")
        print(_histogram(arr, bins=16).replace("  [", "    ["))

    print("\n고르는 방법: 체류 군집과 통과 구간이 갈리는 축에서 임계값을 잡고,")
    print("그 임계값 행의 p95 를 robot_stop_time = max_stop_time 으로 씁니다.")
    print("한 축도 안 갈리면 임계값을 채우지 말고 그 사실을 보고하십시오.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
