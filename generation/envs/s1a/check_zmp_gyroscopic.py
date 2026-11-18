"""Measure what the gyroscopic term is worth, on the real scene.

zero_moment_point uses the full dH/dt = I*alpha + omega x (I*omega). This
runs the S1a scene and reports how far the ZMP moves when the second
half is dropped, so "it matters" is a measurement rather than an estimate.

    ~/isaacsim/python.sh generation/envs/s1a/check_zmp_gyroscopic.py --steps 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCENARIO_ID = "S1a"

parser = argparse.ArgumentParser(description="ZMP 자이로 항 기여도 측정")
parser.add_argument("--steps", type=int, default=400)
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.isaac_boot import launch_app  # noqa: E402

app_launcher, simulation_app = launch_app()

code = 1
try:
    import importlib.metadata as metadata

    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    import isaaclab_tasks  # noqa: F401
    import generation.envs.register  # noqa: F401

    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_tasks.utils.hydra import resolve_presets

    from generation.envs.s1a.mdp.safety import zero_moment_point
    from generation.envs.s1a.s1a_env_cfg import S1aEnvCfg
    from generation.seeding import episode_seed, scenario_startup_seed

    env_cfg = resolve_presets(S1aEnvCfg(), selected=("physx",))
    env_cfg.seed = scenario_startup_seed(SCENARIO_ID)
    agent_cfg = load_cfg_from_registry("Keti-S1a-v0", "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Flat-H1-v0"))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    robot = base.scene["robot"]
    gravity = abs(float(base.sim.cfg.gravity[2]))
    env.seed(episode_seed(SCENARIO_ID, 0))
    with torch.inference_mode():
        obs, _ = env.reset()

    deltas, torques = [], []
    for _ in range(args.steps):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        if bool(dones[0]):
            continue
        m = robot.data.body_mass.torch[0].cpu().numpy()
        p = robot.data.body_com_pose_w.torch[0, :, :3].cpu().numpy()
        acc = robot.data.body_com_acc_w.torch[0].cpu().numpy()
        vel = robot.data.body_com_vel_w.torch[0].cpu().numpy()
        inert = robot.data.body_inertia.torch[0].cpu().numpy().reshape(-1, 3, 3)
        omega = vel[:, 3:]

        full = zero_moment_point(m, p, acc[:, :3], omega, acc[:, 3:], inert,
                                 gravity=gravity)
        # The formula before the fix: alpha only, i.e. omega treated as zero.
        without = zero_moment_point(m, p, acc[:, :3], np.zeros_like(omega), acc[:, 3:],
                                    inert, gravity=gravity)
        if np.isfinite(full).all() and np.isfinite(without).all():
            deltas.append(float(np.linalg.norm(full - without)))
            gyro = np.cross(omega, np.einsum("nij,nj->ni", inert, omega))
            torques.append(float(np.linalg.norm(gyro.sum(axis=0))))

    d, t = np.array(deltas), np.array(torques)
    print(f"\n비교 가능한 스텝 {d.size} / {args.steps}")
    print("ZMP 이동량 |full - alpha_only| (m)")
    print(f"  중앙값 {np.median(d):.5f} · 평균 {d.mean():.5f} · "
          f"p95 {np.percentile(d, 95):.5f} · 최대 {d.max():.5f}")
    print("자이로 토크 합 |Σ ω×(Iω)| (N·m)")
    print(f"  중앙값 {np.median(t):.3f} · p95 {np.percentile(t, 95):.3f} · 최대 {t.max():.3f}")
    env.close()
    code = 0
finally:
    simulation_app.close()

raise SystemExit(code)
