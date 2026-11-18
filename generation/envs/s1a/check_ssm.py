#!/usr/bin/env python3
"""SSM 감속·보호정지가 실제로 일어나는지 스텝별 수치로 확인한다.

    ~/isaacsim/python.sh generation/envs/s1a/check_ssm.py

run_s1a.py 의 종료 코드와 완주만으로는 감속이 실제로 걸렸는지 보이지 않는다.
이 스크립트는 run_s1a.py 와 같은 방식으로 정책을 얹어 같은 한 에피소드를 돌리되,
S1aStraightWalkCommand 가 노출하는 버퍼(min_distance, separation_required, command,
protective_stop_step)를 몇 스텝마다 그대로 찍는다. run_s1a.py 는 다음 작업이 쓰므로
건드리지 않는다 — 이 스크립트가 그 대신 확인용이다.

명령 텐서의 모양이 기존 UniformVelocityCommand 와 같은지도 함께 찍는다: (num_envs, 3)
이 아니면 정책이 학습한 관찰 배열 길이가 어긋난다.

종료 코드: 0 완주 / 1 실행 실패
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# Same sys.path fix as check_scene.py / run_s1a.py — see those files for why.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

    MAX_STEPS = 1000
    PRINT_EVERY = 25

    env_cfg = resolve_presets(S1aEnvCfg(), selected=("physx",))
    env_cfg.seed = 0
    agent_cfg = load_cfg_from_registry("Keti-S1a-v0", "rsl_rl_cfg_entry_point")
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Flat-H1-v0"))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    print(f"command shape={tuple(cmd_term.command.shape)} (UniformVelocityCommand 도 (num_envs, 3))")
    print(f"provisional={cmd_term.provisional} ssm_missing_params={cmd_term.ssm_missing}")
    print(f"ssm_params(사용값)={cmd_term.ssm_params}")
    print(f"slowdown_factor={cmd_term.cfg.slowdown_factor} stop_band_factor={cmd_term.cfg.stop_band_factor}")

    obs = env.get_observations()
    print(f"policy 관찰 배열 shape={tuple(obs['policy'].shape)}")

    min_seen = float("inf")
    stop_step = None
    prev_stop = -1
    step = -1
    for step in range(MAX_STEPS):
        with torch.inference_mode():
            obs, _, dones, extras = env.step(policy(obs))

        dist = float(cmd_term.min_distance[0])
        required = float(cmd_term.separation_required[0])
        walk_speed = float(cmd_term._walk_speed[0])
        cmd_x = float(cmd_term.command[0, 0])
        scale = cmd_x / walk_speed if walk_speed > 1e-9 else 0.0
        min_seen = min(min_seen, dist)

        stop_at = int(cmd_term.protective_stop_step[0])
        newly_stopped = stop_at >= 0 and prev_stop < 0
        prev_stop = stop_at
        if newly_stopped and stop_step is None:
            stop_step = stop_at

        done = bool(dones[0])
        if step % PRINT_EVERY == 0 or newly_stopped or done:
            tag = " <- 보호정지 개시" if newly_stopped else ""
            print(
                f"step={step:4d} min_distance={dist:6.3f} m  separation_required={required:6.3f} m  "
                f"walk_speed={walk_speed:5.3f} m/s  command_x={cmd_x:6.3f} m/s  scale={scale:4.2f}"
                f"{tag}"
            )
        if done:
            print(f"에피소드 종료 (env 내부 스텝 {step})")
            break

    print(
        f"요약: 최소거리 최솟값={min_seen:.3f} m, "
        f"명령이 0 이 된 스텝(protective_stop_step)={stop_step}, "
        f"provisional={cmd_term.provisional}"
    )
    env.close()
    code = 0
except Exception:
    import traceback

    print("check_ssm 실행 실패:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    code = 1
finally:
    # See check_scene.py / run_s1a.py: close(exit_code=code) is what preserves a
    # nonzero code through this build's fast-shutdown path.
    simulation_app.close(exit_code=code)

raise SystemExit(code)
