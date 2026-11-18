#!/usr/bin/env python3
"""S1a 인체 진입이 실제로 보호정지의 원인인지, 방향 보정이 요 드리프트를
줄이는지 스텝별 수치로 확인한다.

    ~/isaacsim/python.sh generation/envs/s1a/check_entry.py \
        [--seeds 0,1,2] [--episodes-per-seed 3] \
        [--force-walk-speed 1.15] [--force-angle-deg 120] [--force-trigger-distance 1.5] \
        [--force-wait-y 1000]

배경 1: `min_distance_human` 은 무방향이다 — 로봇의 진행 차선이라는
개념이 없어, 대기 중인 더미가 진입을
시작하기도 전에 이미 정지 대역 안에 있으면 정지가 진입과 무관하게 걸린다.

배경 2: 그것을 경험적 여유 + 지연항으로 완화하면 교차율이
5~6% 로 떨어진다. 그래서 근본 원인(정책이 ang_vel_z=0
에서도 완벽한 직진을 못 함)을 커맨드 항의 방향 보정으로 고치고, 그 위에서
경험적 여유 없이 유도항만으로도 사전 위반이 없는지, 요 드리프트가 실제로
줄었는지, 교차율이 올랐는지를 이 스크립트로 확인한다(교차율 자체는 Isaac 이
필요 없는 순수 삼각법이라 measure_crossing_rate.py 가 따로 잰다 — 이
스크립트는 그 삼각법이 가정하는 실제 물리 거동이 맞는지를 확인하는 쪽이다).

매 스텝, 강제 설정 없이:

  1. 트리거되기 전 구간 내내 `min_distance_human` 이 정지 대역
     (`separation_required * stop_band_factor`) 밖에 있는가
  2. along-track 트리거(로봇 -> 교차점 x 잔여거리)가 실제로 그 시점에 걸리는가
  3. 보호정지가 걸린다면, 그 시점에 진입이 이미 트리거되어 있었는가
  4. 로봇의 실제 요(yaw)가 얼마나 흐르는가 (방향 보정 효과 확인)

`--force-*` 를 주면 매 에피소드 리셋 직후 해당 값을 덮어써 draw 를 강제한다
(교차점은 강제된 각도로 다시 계산한다) — 위험 구간(walk_speed>=1.1,
angle>=110, trigger<=2.0)처럼 무작위 시드로는 잘 안 뽑히는 조합을 확인할 때
쓴다. 하나도 안 주면 평소처럼 무작위로 뽑는다.

`--seeds` 는 쉼표로 구분한 시드 목록이다 — 한 번의 스크립트 실행(Kit 부팅
1회) 안에서 시드마다 새 환경을 만들어 순서대로 돈다.

run_s1a.py 는 생성 실행기라 확인을 넣지 않는다 — 이 스크립트가 그 대신
인체 진입의 확인용이다.

`--force-wait-y` 는 진단이 아니라 **종료 코드 게이트 자체를 시연**하기 위한
것이다 — 크게 주면(예: 1000) 더미가 경로에서 아주 멀어져 트리거가 한 번도
안 걸리는 무기력 구성이 되고, 그때 이 스크립트가 실제로 실패하는지 확인할 수
있다.

종료 코드: 0 = 트리거 전 정지대역 위반 0건 **이면서** 모든 시드가 최소 1회
트리거 **이면서** 진입발 보호정지가 최소 1건 (아래 게이트 주석에 왜 앞은
시드별이고 뒤는 실행 전체인지 적어 뒀다) / 1 = 실행 실패 또는 위 셋 중 하나라도
불충족
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# ~/isaacsim/python.sh <script> only puts the script's own directory on
# sys.path, not the repository root — same fix as check_scene.py/check_ssm.py.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="S1a 인체 진입발 보호정지 · 요 드리프트 확인")
parser.add_argument("--seeds", type=str, default="0,1,2", help="쉼표로 구분된 환경 시드 목록")
parser.add_argument("--episodes-per-seed", type=int, default=3, help="시드당 돌릴 에피소드 수")
parser.add_argument("--max-steps", type=int, default=1000, help="에피소드당 최대 스텝")
parser.add_argument("--force-walk-speed", type=float, default=None, help="로봇 보행속도 강제")
parser.add_argument("--force-angle-deg", type=float, default=None, help="진입각(도) 강제")
parser.add_argument("--force-trigger-distance", type=float, default=None, help="트리거 거리 강제")
parser.add_argument(
    "--force-wait-y",
    type=float,
    default=None,
    help="더미 대기 횡오프셋 강제 (무기력 구성 시연용 — 크게 주면 아무 일도 안 일어난다)",
)
args = parser.parse_args()
seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

from generation.isaac_boot import launch_app

# Booting through the helper is what makes a boot failure exit nonzero: the
# try/finally below only arms once simulation_app exists, and until it does,
# Kit's own atexit handler would close with its default exit_code=0 and hand
# back a green process. See generation/isaac_boot.py for the measurement.
simulation_app = launch_app()[1]

try:
    import importlib.metadata as metadata
    import math

    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    import isaaclab_tasks  # noqa: F401  — registers the H1 tasks the agent cfg lives in
    import generation.envs.register  # noqa: F401  — registers Keti-S1a-v0

    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_tasks.utils.hydra import resolve_presets

    from generation.envs.s1a.mdp.entry_geometry import crossing_x
    from generation.envs.s1a.s1a_env_cfg import S1aEnvCfg

    PRINT_EVERY = 50
    REWARD_LOG_KEY = "Episode_Reward/track_lin_vel_xy_exp"
    forcing = (
        args.force_walk_speed is not None
        or args.force_angle_deg is not None
        or args.force_trigger_distance is not None
        or args.force_wait_y is not None
    )

    def robot_yaw_deg(robot) -> float:
        qx, qy, qz, qw = robot.data.root_quat_w.torch[0].tolist()
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return math.degrees(yaw)

    pretrigger_violations = 0
    total_episodes = 0
    total_triggered = 0
    total_entry_caused_stops = 0
    # Per-seed tallies as well as run totals: a run total cannot show a seed
    # that produced nothing at all, and that is exactly what the exit-code
    # gate at the bottom has to be able to see.
    triggered_by_seed = {seed: 0 for seed in seeds}
    entry_caused_stops_by_seed = {seed: 0 for seed in seeds}
    all_summaries = []

    for seed in seeds:
        print(f"\n########## 시드 {seed} ##########")
        env_cfg = resolve_presets(S1aEnvCfg(), selected=("physx",))
        env_cfg.seed = seed
        agent_cfg = load_cfg_from_registry("Keti-S1a-v0", "rsl_rl_cfg_entry_point")
        installed_version = metadata.version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Flat-H1-v0"))
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        robot = env.unwrapped.scene["robot"]
        dummy_contact = env.unwrapped.scene["dummy_contact"]
        entry = env.unwrapped.s1a_entry

        if seed == seeds[0]:
            print(f"stop_band_factor={cmd_term.cfg.stop_band_factor} "
                  f"heading_correction_gain={cmd_term.cfg.heading_correction_gain} "
                  f"heading_correction_ang_vel_z_range={cmd_term.cfg.heading_correction_ang_vel_z_range}")

        obs = env.get_observations()

        for episode in range(args.episodes_per_seed):
            if forcing:
                # Overwrite this episode's draw right after reset
                # (touch the command term's/entry's own state directly, not a
                # new code path). x_cross has to be recomputed with the forced
                # angle or the along-track trigger would still use the
                # original (random) crossing point.
                #
                # target_xy and x_cross are both WORLD frame (env_origins
                # folded in — mdp/events.py adds them when writing target_xy
                # and x_cross). crossing_x itself works in the env-LOCAL
                # frame (see mdp/events.py's own call site), so the origin
                # has to come back out before calling it and go back in
                # after — skipping this only looks correct at num_envs=1
                # (env_origins==0), which is silently wrong for num_envs>1.
                origin_xy = env.unwrapped.scene.env_origins[0, :2]
                wait_x = float(entry["target_xy"][0, 0].item()) - float(origin_xy[0].item())
                wait_y = float(entry["target_xy"][0, 1].item()) - float(origin_xy[1].item())
                if args.force_walk_speed is not None:
                    cmd_term._walk_speed[0] = args.force_walk_speed
                if args.force_angle_deg is not None:
                    entry["angle"][0] = math.radians(args.force_angle_deg)
                if args.force_trigger_distance is not None:
                    entry["trigger_distance"][0] = args.force_trigger_distance
                if args.force_wait_y is not None:
                    # Only the lateral offset is forced; wait_x stays whatever
                    # reset_dummy_entry derived. Large values push the dummy so
                    # far off the path that nothing can trigger — the inert
                    # configuration the exit-code gate has to reject.
                    wait_y = args.force_wait_y
                    entry["target_xy"][0, 1] = wait_y + float(origin_xy[1].item())
                entry["x_cross"][0] = float(
                    crossing_x(wait_x, wait_y, float(entry["angle"][0].item()))
                ) + float(origin_xy[0].item())

            angle_deg = float(entry["angle"][0]) * 180.0 / math.pi
            print(f"\n=== 에피소드 {episode} (시드 {seed}) ===")
            print(
                f"진입 draw: speed={float(entry['speed'][0]):.3f} m/s angle={angle_deg:.1f} deg "
                f"trigger_distance={float(entry['trigger_distance'][0]):.3f} m "
                f"walk_speed={float(cmd_term._walk_speed[0]):.3f} m/s"
                f"{'  [강제 draw]' if forcing else ''}"
            )
            print(
                f"대기 위치(target_xy)={[round(v, 3) for v in entry['target_xy'][0].tolist()]} "
                f"stand_z={float(entry['stand_z'][0]):.4f} x_cross={float(entry['x_cross'][0]):.3f} "
                f"spawn_yaw={robot_yaw_deg(robot):+.2f}deg"
            )

            triggered_before = False
            trigger_step = None
            stop_step = None
            prev_stop = -1
            min_pretrigger_margin = float("inf")
            min_distance_seen = float("inf")
            max_abs_yaw_deg = 0.0
            step_in_episode = 0
            done = False
            tracking_reward = None

            for step in range(args.max_steps):
                with torch.inference_mode():
                    obs, _, dones, extras = env.step(policy(obs))
                step_in_episode = step + 1

                dist = float(cmd_term.min_distance[0])
                required = float(cmd_term.separation_required[0])
                stop_band = required * float(cmd_term.cfg.stop_band_factor)
                is_triggered = bool(entry["triggered"][0])
                robot_x = float(robot.data.root_pos_w.torch[0, 0])
                dist_to_crossing = float(entry["x_cross"][0]) - robot_x
                was_min = dist < min_distance_seen
                min_distance_seen = min(min_distance_seen, dist)
                yaw_deg = robot_yaw_deg(robot)
                max_abs_yaw_deg = max(max_abs_yaw_deg, abs(yaw_deg))

                # If the entry produces actual contact, log
                # dummy_contact's per-region forces (not just min_distance).
                # 0.1 m is comfortably inside "could plausibly be touching"
                # given the dummy's own region radii (BR-08 arm ~0.05 m) and
                # the flat-geometry approximation min_distance_human uses.
                if dist < 0.1 and was_min:
                    forces = dummy_contact.data.net_forces_w.norm(dim=-1).squeeze(0)
                    names = dummy_contact.body_names
                    nonzero = {n: round(float(f), 4) for n, f in zip(names, forces) if float(f) > 1e-6}
                    print(
                        f"  [접촉 근접] step={step_in_episode} min_distance={dist:.4f}m "
                        f"dummy_contact 힘(N, 0 초과만)={nonzero if nonzero else '전부 0'}"
                    )

                if not is_triggered:
                    margin = dist - stop_band
                    min_pretrigger_margin = min(min_pretrigger_margin, margin)
                    if margin < 0.0:
                        pretrigger_violations += 1
                        print(
                            f"  [결함] step={step_in_episode:4d} 트리거 전인데 "
                            f"min_distance={dist:.3f} < stop_band={stop_band:.3f} "
                            f"(margin={margin:+.3f})"
                        )
                elif not triggered_before:
                    trigger_step = step_in_episode
                    total_triggered += 1
                    triggered_by_seed[seed] += 1
                    print(
                        f"  진입 시작: step={step_in_episode} "
                        f"dist_to_crossing={dist_to_crossing:+.3f}m "
                        f"(trigger_distance={float(entry['trigger_distance'][0]):.3f}m) "
                        f"min_distance={dist:.3f}m"
                    )
                triggered_before = is_triggered

                stop_at = int(cmd_term.protective_stop_step[0])
                newly_stopped = stop_at >= 0 and prev_stop < 0
                prev_stop = stop_at
                if newly_stopped:
                    stop_step = step_in_episode
                    caused_by_entry = is_triggered
                    if caused_by_entry:
                        total_entry_caused_stops += 1
                        entry_caused_stops_by_seed[seed] += 1
                    tag = "" if caused_by_entry else "  <- [결함] 진입 전 정지"
                    print(
                        f"  보호정지: step={step_in_episode} min_distance={dist:.3f}m "
                        f"required={required:.3f}m stop_band={stop_band:.3f}m "
                        f"진입 트리거됨={caused_by_entry}{tag}"
                    )
                    if not caused_by_entry:
                        pretrigger_violations += 1

                if step_in_episode % PRINT_EVERY == 0:
                    print(
                        f"  step={step_in_episode:4d} dist={dist:6.3f}m stop_band={stop_band:.3f}m "
                        f"triggered={is_triggered!s:5s} dist_to_crossing={dist_to_crossing:7.3f}m "
                        f"robot_yaw={yaw_deg:+6.2f}deg"
                    )

                if bool(dones[0]):
                    done = True
                    tracking_reward = extras["log"].get(REWARD_LOG_KEY)
                    break

            if not done:
                print(f"  [주의] {args.max_steps} 스텝 안에 dones 가 서지 않음 (비정상)")

            total_episodes += 1
            summary = {
                "seed": seed,
                "episode": episode,
                "steps": step_in_episode,
                "trigger_step": trigger_step,
                "stop_step": stop_step,
                "min_pretrigger_margin": (
                    None if min_pretrigger_margin == float("inf") else min_pretrigger_margin
                ),
                "min_distance_seen": min_distance_seen,
                "max_abs_yaw_deg": max_abs_yaw_deg,
                "tracking_reward": (
                    float(tracking_reward.item()) if tracking_reward is not None else None
                ),
            }
            all_summaries.append(summary)
            reward_str = (
                f"{summary['tracking_reward']:.4f}" if summary["tracking_reward"] is not None
                else "N/A"
            )
            print(
                f"에피소드 {episode} 종료: {step_in_episode} 스텝, trigger_step={trigger_step}, "
                f"stop_step={stop_step}, 최소거리={min_distance_seen:.3f}m, "
                f"최대|yaw|={max_abs_yaw_deg:.2f}deg, 추종보상={reward_str}"
            )

        env.close()

    print("\n=== 전체 요약 ===")
    for s in all_summaries:
        print(
            f"seed={s['seed']} episode={s['episode']} steps={s['steps']} "
            f"trigger_step={s['trigger_step']} stop_step={s['stop_step']} "
            f"min_distance={s['min_distance_seen']:.3f}m 최대|yaw|={s['max_abs_yaw_deg']:.2f}deg "
            f"추종보상={s['tracking_reward']}"
        )
    print(
        f"\n에피소드 {total_episodes}건 (시드 {seeds}), "
        f"트리거된 에피소드 {total_triggered}건, 진입발 보호정지 {total_entry_caused_stops}건, "
        f"트리거 전 정지대역 위반 {pretrigger_violations}건 (0 이어야 정상)"
    )

    env_closed_reward = [s["tracking_reward"] for s in all_summaries if s["tracking_reward"] is not None]
    if env_closed_reward:
        avg_reward = sum(env_closed_reward) / len(env_closed_reward)
        print(f"평균 추종보상(완주 에피소드만): {avg_reward:.4f} (기준 0.95 근방)")

    print(
        "시드별 (트리거 / 진입발 정지): "
        + ", ".join(
            f"seed={seed} {triggered_by_seed[seed]}/{entry_caused_stops_by_seed[seed]}"
            for seed in seeds
        )
    )

    # Exit-code gate. pretrigger_violations==0 on its own is satisfied
    # vacuously by any configuration in which nothing ever happens — a wait_y
    # wide enough that the robot never comes near shows 0 violations forever
    # without exercising a single thing this script exists to check
    # (reproduce it with --force-wait-y). A run-wide
    # "at least one trigger" clause alone is half-wired;
    # the gate needs both halves:
    #
    #  - Triggers are required PER SEED. A trigger is a property of the
    #    *geometry* (does the robot ever come within trigger_distance of the
    #    crossing point?), which the derived corridor is supposed to hold for
    #    essentially every draw — so a seed that never triggers at all is that
    #    seed's evidence being empty, and a run-wide sum hides it: one lively
    #    seed would carry a whole batch of inert ones.
    #
    #  - Entry-caused stops are required PER RUN, not per seed. Whether a
    #    triggered entry actually closes into the stop band before the episode
    #    ends is a drawn *outcome* (entry speed/angle/trigger distance against
    #    the robot's walk speed), and the catalogue deliberately wants a mix of
    #    stop and no-stop episodes (meta/scenarios.json outcome_target) — a
    #    seed whose draws all end `nominal` is correct behavior, not a defect,
    #    so requiring one per seed would be requiring an outcome the
    #    randomization is designed to sometimes withhold. Requiring one
    #    somewhere in the run is the weakest clause that still keeps this
    #    script's actual question ("did the stop follow the entry, or precede
    #    it?") from being answered vacuously: with zero entry-caused stops the
    #    "stop follows entry" property was never once observed, and without
    #    this clause that run would still have exited 0.
    # Practical note on the per-seed clause: in a measured run
    # (5 seeds x 2 episodes) 6 of 10 episodes triggered and all 5 seeds
    # triggered at least once — the shortfall is early-terminated episodes
    # (a fall ends an episode before the robot ever reaches the crossing
    # point) plus draws whose crossing point sits beyond where the robot
    # walks in 20 s. One episode per seed therefore makes this clause a coin
    # flip on a perfectly good configuration; --episodes-per-seed defaults to
    # 3 for that reason, and 2 is the practical floor.
    inert_seeds = [seed for seed in seeds if triggered_by_seed[seed] == 0]
    if inert_seeds:
        print(
            f"[결함] 트리거가 한 번도 안 걸린 시드 {inert_seeds} — 무기력 구성"
            "(해당 시드에서 진입이 전혀 검증되지 않음)"
        )
    if total_entry_caused_stops == 0:
        print(
            "[결함] 진입발 보호정지가 0건 — 정지가 진입 뒤에 걸리는지를"
            " 이 실행이 한 번도 관측하지 못함(공허한 통과)"
        )
    code = 0 if (
        pretrigger_violations == 0
        and not inert_seeds
        and total_entry_caused_stops > 0
    ) else 1
except Exception:
    import traceback

    print("check_entry 실행 실패:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    code = 1
finally:
    # See check_scene.py/check_ssm.py: close(exit_code=code) is what preserves
    # a nonzero code through this build's fast-shutdown path.
    simulation_app.close(exit_code=code)

raise SystemExit(code)
