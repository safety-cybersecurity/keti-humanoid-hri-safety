#!/usr/bin/env python3
"""Monte Carlo trigger/crossing rate for S1a's human entry. Pure numpy, no
torch, no Isaac Sim — runs with plain `python3`.

    python3 generation/envs/s1a/measure_crossing_rate.py --wait-x 5.2 --wait-y 1.9

A corridor widened by an empirical margin and a delay term lets only ~5-6% of
catalogue-uniform draws actually reach the robot's lane within one episode
(29-39% ever trigger, fewer than that finish crossing) — "13 에피소드 중
진입 원인 정지가 1건" matches that rate. A figure like that has to come
from a script anyone can re-run, and this is that script. The corridor
geometry (wait_x/wait_y) is still produced by mdp/events.py (it needs a
live Isaac Sim asset to measure both bodies' reach — see
mdp/events.py's `_lateral_reach`), but the rate calculation itself is pure
trigonometry (mdp/entry_geometry.py) and belongs where it can be re-run
without booting the simulator.

Randomization ranges are read from meta/scenarios.json (the canonical
source, CONTRIBUTING.md 「함께 고치는 곳」) rather than re-typed here. episode_length_s
defaults to 20.0, traced to
isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py's
LocomotionVelocityRoughEnvCfg.__post_init__ (S1aEnvCfg inherits this
unchanged) — pass --episode-length-s to override if that ever changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

import sys  # noqa: E402 — needs REPO_ROOT above first

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.envs.s1a.mdp.entry_geometry import crossing_rate, trigger_rate  # noqa: E402


def load_s1a_randomization() -> dict:
    scenarios = json.loads((REPO_ROOT / "meta/scenarios.json").read_text(encoding="utf-8"))
    s1a = next(s for s in scenarios["scenarios"] if s["scenario_id"] == "S1a")
    r = s1a["randomization"]
    return {
        "robot_walk_speed_range": tuple(r["robot_walk_speed"]),
        "human_entry_speed_range": tuple(r["human_entry_speed"]),
        "angle_range_deg": tuple(r["human_entry_angle_deg"]),
        "trigger_distance_range": tuple(r["entry_trigger_distance"]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wait-x", type=float, required=True, help="Corridor longitudinal offset (m)")
    p.add_argument("--wait-y", type=float, required=True, help="Corridor lateral offset (m)")
    p.add_argument("--episode-length-s", type=float, default=20.0)
    p.add_argument("--spawn-margin-x", type=float, default=0.5,
                    help="Robot's own x spawn randomization half-range (m) — matches "
                         "S1aEnvCfg's reset_base pose_range unless overridden there")
    p.add_argument("--n-samples", type=int, default=200_000, help="Monte Carlo sample count")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ranges = load_s1a_randomization()
    rng = np.random.default_rng(args.seed)
    spawn_x_range = (-args.spawn_margin_x, args.spawn_margin_x)

    t_rate = trigger_rate(
        rng, args.n_samples,
        wait_x=args.wait_x, wait_y=args.wait_y,
        robot_walk_speed_range=ranges["robot_walk_speed_range"],
        angle_range_deg=ranges["angle_range_deg"],
        trigger_distance_range=ranges["trigger_distance_range"],
        episode_length_s=args.episode_length_s,
        robot_spawn_x_range=spawn_x_range,
    )
    c_rate = crossing_rate(
        rng, args.n_samples,
        wait_x=args.wait_x, wait_y=args.wait_y,
        robot_walk_speed_range=ranges["robot_walk_speed_range"],
        human_entry_speed_range=ranges["human_entry_speed_range"],
        angle_range_deg=ranges["angle_range_deg"],
        trigger_distance_range=ranges["trigger_distance_range"],
        episode_length_s=args.episode_length_s,
        robot_spawn_x_range=spawn_x_range,
    )

    print(f"wait_x={args.wait_x:.3f} m  wait_y={args.wait_y:.3f} m  "
          f"episode_length_s={args.episode_length_s:.1f}  n_samples={args.n_samples}")
    print(f"randomization ranges (meta/scenarios.json S1a): {ranges}")
    print(f"트리거가 걸릴 수 있는 draw: {t_rate * 100:.1f}%")
    print(f"인체가 실제로 로봇 차선에 도달하는 draw: {c_rate * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
