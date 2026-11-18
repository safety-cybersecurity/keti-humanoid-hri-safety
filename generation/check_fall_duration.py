#!/usr/bin/env python3
"""`R-25785-STAB-01` 의 `min_duration` 이 낙상을 가를 수 있는지 배포본에서 잰다.

    .venv-lerobot/bin/python generation/check_fall_duration.py .tmp/lerobot/S1a-400

Isaac 을 부팅하지 않는다 — **이미 만들어진 배포본만** 읽으므로 배치를 다시 돌리지
않고 확인할 수 있다. parquet 을 읽으므로 `.venv-lerobot` 이 필요하다.

**왜 있는가.** 규격서 §6 이 「동적 보행 중에는 음수가 정상이므로 전도는 부호가 아니라
**지속시간**으로 가른다」고 적고, `min_duration` 을 그 경계로 둔다. 그 근거는
2026-08-20 의 18 에피소드(낙상 2건) 측정이었다. 현재 배치는 낙상이 43건이므로
같은 질문을 다시 물을 수 있다.

재는 것은 규칙이 실제로 보는 양이다 — 에피소드마다 `stability_margin < 0` 인 **가장
긴 연속 구간**. 규칙은 「그 길이가 `min_duration` 이상인가」이므로, 낙상과 정상이 이
양에서 갈리지 않으면 `min_duration` 을 어떤 값으로 채워도 규칙은 서지 않는다.

비교군으로 `-inf` 비율(양발 모두 미하중 = 발로 서 있지 않음)도 함께 낸다. 2026-08-20
기록이 그쪽에 신호가 있다고 적었고 표본이 2건이라 결론을 미뤘던 항목이다.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def as_list(value):
    if isinstance(value, str):
        value = json.loads(value)
    return list(value) if value is not None else []


def main() -> int:
    import numpy as np
    import pandas as pd

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="배포본 루트 (예: .tmp/lerobot/S1a-400)")
    args = p.parse_args()

    root = args.dataset
    meta = sorted(glob.glob(os.path.join(root, "meta/episodes/**/*.parquet"), recursive=True))
    data = sorted(glob.glob(os.path.join(root, "data/**/*.parquet"), recursive=True))
    if not meta or not data:
        print(f"배포본이 아닙니다: {root}", file=sys.stderr)
        return 2

    episodes = pd.concat([pd.read_parquet(f) for f in meta])
    fell = {}
    for _, row in episodes.iterrows():
        reasons = as_list(json.loads(row["termination"])["reason"])
        fell[int(row["episode_index"])] = any(r.startswith("fall") for r in reasons)

    rows = []
    for f in data:
        d = pd.read_parquet(f, columns=["episode_index", "frame_index",
                                        "safety.stability_margin"])
        for idx, group in d.groupby("episode_index"):
            m = group.sort_values("frame_index")["safety.stability_margin"].to_numpy(float)
            negative = m < 0
            # 연속 구간의 길이. 앞뒤에 0 을 붙여 시작/끝을 경계로 만든다.
            edges = np.diff(np.concatenate(([0], negative.view(np.int8), [0])))
            runs = np.where(edges == -1)[0] - np.where(edges == 1)[0]
            rows.append({
                "episode_index": int(idx),
                "fell": fell[int(idx)],
                "max_run": int(runs.max()) if len(runs) else 0,
                "inf_fraction": float(np.isneginf(m).mean()),
            })
    table = pd.DataFrame(rows)

    n_fell = int(table["fell"].sum())
    print(f"에피소드 {len(table)} · 낙상 {n_fell} · 정상 {len(table) - n_fell}")
    if n_fell == 0 or n_fell == len(table):
        print("한쪽 집단이 비어 있어 분리를 잴 수 없습니다.")
        return 0

    for column, label in (("max_run", "에피소드 최장 음수 런 (스텝) — 규칙이 보는 양"),
                          ("inf_fraction", "-inf 비율 (양발 미하중)")):
        normal = table.loc[~table["fell"], column]
        fallen = table.loc[table["fell"], column]
        # 「임계 이상이면 낙상」으로 불렀을 때 가장 많이 맞히는 경계
        best_t, best_hit = None, -1
        for t in np.unique(table[column]):
            hit = int((fallen >= t).sum() + (normal < t).sum())
            if hit > best_hit:
                best_t, best_hit = t, hit
        tp, fp = int((fallen >= best_t).sum()), int((normal >= best_t).sum())
        overlap = normal.max() >= fallen.min()
        print(f"\n### {label}")
        print(f"  정상 중앙 {normal.median():.6g} · p95 {normal.quantile(.95):.6g} "
              f"· 최대 {normal.max():.6g}")
        print(f"  낙상 중앙 {fallen.median():.6g} · p95 {fallen.quantile(.95):.6g} "
              f"· 최대 {fallen.max():.6g}")
        print(f"  최적 경계 {best_t:.6g} → 낙상 {tp}/{len(fallen)} 검출, "
              f"정상 {fp}/{len(normal)} 오검")
        print(f"  분포 겹침: {'있음 — 단일 경계로 분리 불가' if overlap else '없음'}")

    print("\n낙상 에피소드는 자세 기준 종료(root height + tilt)로 끝나므로 넘어진 뒤의"
          "\n여유 이력이 기록되지 않습니다. 지속시간이 쌓일 구간 자체가 없다는 뜻이며,"
          "\n`min_duration` 을 어떤 값으로 채워도 이 배치에서는 규칙이 서지 않습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
