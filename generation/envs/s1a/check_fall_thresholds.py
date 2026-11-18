#!/usr/bin/env python3
"""s1a_env_cfg.py 의 fall_height/fall_orientation 임계값이 정상 보행과 실제 낙상을
가르는지 실측으로 확인한다.

    ~/isaacsim/python.sh generation/envs/s1a/check_fall_thresholds.py

s1a_env_cfg.py 의 종료항 주석에 적힌 "정상 보행 최악 root 높이 0.865m·최대 상향각
14.3deg, 낙상 시 ~0.07-0.08m·~85-86deg" 수치가 나온 스크립트다 — 그 주석이 가리키는
근거는 다시 돌릴 수 있는 스크립트로 저장소에 있어야 한다(커밋되지 않은 스크립트에
기댄 결론은 뒤집혀도 확인할 길이 없다).

에피소드마다 base_contact 를 끈 채(운영 종료항이 조기에 자르지 않도록) 실제 정책으로
최대 --max-steps 스텝을 돌리며 로봇 root 높이의 최솟값과 상향각(bad_orientation 이
쓰는 것과 같은 계산, projected_gravity_b 기반)의 최댓값을 잰다. "낙상"과 "정상"의
구분은 s1a_env_cfg.py 가 실제로 쓰는 --minimum-height/--limit-angle-deg 가 아니라
훨씬 명백한 별도 기준(높이 0.3m 미만 또는 각도 45deg 초과)으로 한다 — 판정 임계값
자체로 낙상 여부를 가르면 "임계값이 낙상을 가른다"는 순환논증이 된다.

종료 코드: 0 정상 에피소드 전부가 두 임계값에 여유(--margin-m/--margin-deg)를 두고
못 미쳤고, 낙상 에피소드가 하나 이상 있으며 전부 임계값을 넘었음 / 1 위 조건 위반
또는 실행 실패
"""

import argparse
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# ~/isaacsim/python.sh <script> only puts the script's own directory on
# sys.path, not the repository root — same fix as check_scene.py/check_ssm.py.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="S1a 낙상 종료 임계값 실측 근거")
parser.add_argument("--seeds", type=str, default="0,7", help="쉼표로 구분된 환경 시드 목록")
parser.add_argument("--episodes-per-seed", type=int, default=4, help="시드당 돌릴 에피소드 수")
parser.add_argument("--max-steps", type=int, default=1000, help="에피소드당 최대 스텝")
# s1a_env_cfg.py 의 실제 운영값과 같은 기본값 — 재입력이 아니라 "이 값이 지금도
# 맞는지" 재확인하는 것이 이 스크립트의 목적이므로, 여기서 바뀌면 그쪽도 같이 봐야 한다.
parser.add_argument("--minimum-height", type=float, default=0.6, help="fall_height 임계값(m)")
parser.add_argument("--limit-angle-deg", type=float, default=math.degrees(0.5), help="fall_orientation 임계값(deg)")
parser.add_argument("--margin-m", type=float, default=0.15, help="정상 보행 최악값과 임계값 사이 요구 여유(m)")
parser.add_argument("--margin-deg", type=float, default=5.0, help="정상 보행 최악값과 임계값 사이 요구 여유(deg)")
# 명백한 낙상 판정 기준 — 운영 임계값과 분리 (순환논증 방지, docstring 참고).
parser.add_argument("--obvious-fall-height", type=float, default=0.3, help="이 아래면 낙상으로 분류(m)")
parser.add_argument("--obvious-fall-angle-deg", type=float, default=45.0, help="이 위면 낙상으로 분류(deg)")
args = parser.parse_args()
seeds = [int(s) for s in args.seeds.split(",")]

from generation.isaac_boot import launch_app

# Booting through the helper is what makes a boot failure exit nonzero: the
# try/finally below only arms once simulation_app exists, and until it does,
# Kit's own atexit handler would close with its default exit_code=0 and hand
# back a green process. See generation/isaac_boot.py for the measurement.
simulation_app = launch_app()[1]

try:
    import importlib.metadata as metadata

    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    import isaaclab_tasks  # noqa: F401  — registers the H1 tasks the agent cfg lives in
    import generation.envs.register  # noqa: F401  — registers Keti-S1a-v0

    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_tasks.utils.hydra import resolve_presets

    from generation.envs.s1a.s1a_env_cfg import S1aEnvCfg

    normal_worst_z = float("inf")       # min over all normal episodes' min-z
    normal_worst_angle = 0.0            # max over all normal episodes' max-angle
    fall_best_z = float("inf")          # min over all fall episodes' min-z (for reporting)
    fall_best_angle = 0.0
    n_normal = 0
    n_fall = 0
    fall_missed = []  # falls that never crossed either operational threshold

    for seed in seeds:
        env_cfg = resolve_presets(S1aEnvCfg(), selected=("physx",))
        env_cfg.seed = seed
        # Disabled so a fall plays out to its actual collapsed state instead
        # of being cut short — this script needs the true trajectory shape,
        # not the operational episode length.
        env_cfg.terminations.fall_height = None
        env_cfg.terminations.fall_orientation = None
        agent_cfg = load_cfg_from_registry("Keti-S1a-v0", "rsl_rl_cfg_entry_point")
        installed_version = metadata.version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Flat-H1-v0"))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        robot = env.unwrapped.scene["robot"]
        obs = env.get_observations()

        for episode in range(args.episodes_per_seed):
            min_z = float("inf")
            max_angle = 0.0
            for step in range(args.max_steps):
                with torch.inference_mode():
                    obs, _, dones, extras = env.step(policy(obs))
                z = float(robot.data.root_pos_w.torch[0, 2].item())
                grav = robot.data.projected_gravity_b.torch[0]
                angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(-grav[2])))))
                min_z = min(min_z, z)
                max_angle = max(max_angle, angle_deg)
                if bool(dones[0]):
                    break

            fell = min_z < args.obvious_fall_height or max_angle > args.obvious_fall_angle_deg
            tag = "낙상" if fell else "정상"
            print(f"seed={seed} ep={episode} [{tag}] steps={step + 1} min_z={min_z:.3f}m "
                  f"max_angle={max_angle:.2f}deg")
            if fell:
                n_fall += 1
                fall_best_z = min(fall_best_z, min_z)
                fall_best_angle = max(fall_best_angle, max_angle)
                crossed = min_z < args.minimum_height or max_angle > args.limit_angle_deg
                if not crossed:
                    fall_missed.append((seed, episode, min_z, max_angle))
            else:
                n_normal += 1
                normal_worst_z = min(normal_worst_z, min_z)
                normal_worst_angle = max(normal_worst_angle, max_angle)
        env.close()

    print(f"\n=== 요약 ({n_normal}건 정상, {n_fall}건 낙상) ===")
    if n_normal:
        print(f"정상 보행 최악: min_z={normal_worst_z:.3f}m (임계값 {args.minimum_height}m 대비 여유 "
              f"{normal_worst_z - args.minimum_height:+.3f}m), "
              f"max_angle={normal_worst_angle:.2f}deg (임계값 {args.limit_angle_deg:.2f}deg 대비 여유 "
              f"{args.limit_angle_deg - normal_worst_angle:+.2f}deg)")
    if n_fall:
        print(f"낙상 최댓값(덜 극단적인 쪽): min_z={fall_best_z:.3f}m, max_angle={fall_best_angle:.2f}deg")
        if fall_missed:
            print(f"[결함] 임계값을 못 넘은 낙상 {len(fall_missed)}건: {fall_missed}")

    failures = []
    if n_normal == 0:
        failures.append("정상(비낙상) 에피소드가 하나도 없음 — 여유를 판단할 근거 없음")
    elif normal_worst_z - args.minimum_height < args.margin_m:
        failures.append(
            f"정상 보행 최악 높이({normal_worst_z:.3f}m)가 임계값+여유"
            f"({args.minimum_height + args.margin_m:.3f}m)에 못 미침"
        )
    elif args.limit_angle_deg - normal_worst_angle < args.margin_deg:
        failures.append(
            f"정상 보행 최악 각도({normal_worst_angle:.2f}deg)가 임계값-여유"
            f"({args.limit_angle_deg - args.margin_deg:.2f}deg)를 넘음"
        )
    if n_fall == 0:
        failures.append("낙상 에피소드가 하나도 관측되지 않음 — 임계값이 실제로 낙상을 잡는지 못 봄")
    if fall_missed:
        failures.append(f"임계값을 못 넘은 낙상 {len(fall_missed)}건 (위 목록)")

    if failures:
        print("\n[결함]")
        for f in failures:
            print(f"  - {f}")
        code = 1
    else:
        print(f"\nminimum_height={args.minimum_height}m, limit_angle={args.limit_angle_deg:.2f}deg 가 "
              f"정상 보행과 낙상을 여유를 두고 가릅니다.")
        code = 0
except Exception:
    import traceback

    print("check_fall_thresholds 실행 실패:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    code = 1
finally:
    # See check_scene.py: close(exit_code=code) is what preserves a nonzero
    # code through this build's fast-shutdown path.
    simulation_app.close(exit_code=code)

raise SystemExit(code)
