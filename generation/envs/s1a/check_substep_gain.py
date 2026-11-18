#!/usr/bin/env python3
"""서브스텝 표집이 스냅샷보다 무엇을 더 잡는지 센다.

    ~/isaacsim/python.sh generation/envs/s1a/check_substep_gain.py [--steps 400] [--seq 0]

`check_contact_history.py` 는 「서브스텝 해상도를 얻는 경로가 실재하는가」 를 물었고
답은 예였습니다. 이 스크립트는 그 경로를 실제로 채택한 뒤(`substep_contact.py`)
**운영 구성에서** 이득의 크기를 잽니다 — 두 값이 같은 실행, 같은 스텝에서 나옵니다.

  스냅샷   : 기록 시점의 `net_forces_w`. 고치기 전 recorder 가 쓰던 값
  서브스텝 : 그 env 스텝의 물리 서브스텝 최댓값. 지금 recorder 가 쓰는 값

세는 것은 둘입니다.

  놓친 접촉 : 스냅샷이 0 인데 서브스텝 최댓값은 0 이 아닌 (스텝, 바디) 쌍.
              **이것이 데이터에서 통째로 빠지던 접촉입니다** — 짧게 기록된 것이
              아니라 없던 것이 됩니다
  과소평가  : 둘 다 0 이 아니지만 서브스텝 쪽이 큰 경우. 접촉은 남았지만 크기가
              작게 적히던 스텝입니다

종료 코드: 0 판정이 나옴 / 1 실행 실패
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="서브스텝 표집의 이득을 잰다")
parser.add_argument("--steps", type=int, default=400, help="측정할 env 스텝 수")
parser.add_argument("--seq", type=int, default=0, help="에피소드 시퀀스 번호")
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.isaac_boot import launch_app

simulation_app = launch_app()[1]

try:
    import importlib.metadata as metadata

    import numpy as np
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    import generation.envs.register  # noqa: F401

    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_tasks.utils.hydra import resolve_presets

    from generation.envs.s1a import substep_contact
    from generation.envs.s1a.s1a_env_cfg import S1aEnvCfg
    from generation.seeding import episode_seed, scenario_startup_seed

    SCENARIO_ID = "S1a"
    SENSORS = ("dummy_contact", "contact_forces")

    cfg = resolve_presets(S1aEnvCfg(), selected=("physx",))
    cfg.seed = scenario_startup_seed(SCENARIO_ID)
    agent_cfg = load_cfg_from_registry("Keti-S1a-v0", "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Flat-H1-v0"))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    from generation.envs.s1a.recorder import CONTACT_EPS_N

    acc = substep_contact.install(base, contact_eps_n=CONTACT_EPS_N)
    env.seed(episode_seed(SCENARIO_ID, args.seq))
    with torch.inference_mode():
        obs, _ = env.reset()
    acc.reset()

    snapshots = {n: [] for n in SENSORS}
    peaks = {n: [] for n in SENSORS}
    samples = []
    for _ in range(args.steps):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        # The snapshot the pre-fix recorder read: the value at this instant.
        # Taken before drain() so nothing about it depends on the accumulator.
        for name in SENSORS:
            snapshots[name].append(torch.linalg.norm(
                base.scene[name].data.net_forces_w.torch[0], dim=-1).cpu().numpy())
        batch = acc.drain()
        samples.append(batch["samples"])
        peaks["dummy_contact"].append(batch["dummy_force"])
        peaks["contact_forces"].append(batch["robot_force"])
        if bool(dones[0]):
            break

    n = len(samples)
    print(f"\ndecimation={base.cfg.decimation}  physics_dt={base.physics_dt}s  "
          f"step_dt={base.step_dt}s  측정 스텝={n}")
    print(f"env 스텝당 접촉 표본: 최소 {min(samples)} · 최대 {max(samples)} "
          f"(decimation={base.cfg.decimation} 과 같아야 함)")
    mechanism_ok = set(samples) == {base.cfg.decimation}

    EPS = 1e-6
    verdict_rows = []
    for name in SENSORS:
        snap = np.stack(snapshots[name])
        peak = np.stack(peaks[name])
        missed = (snap <= EPS) & (peak > EPS)
        under = (snap > EPS) & (peak > snap * (1 + 1e-9))
        ratio = np.divide(peak, snap, out=np.ones_like(peak), where=snap > EPS)
        print(f"\n=== {name} ===")
        print(f"  놓친 접촉 (스냅샷 0, 서브스텝 >0): {int(missed.sum())} 쌍 / "
              f"스텝 {len(np.unique(np.nonzero(missed)[0]))} 개"
              + (f"  최대 {float(peak[missed].max()):.1f} N" if missed.any() else ""))
        print(f"  과소평가 (둘 다 >0, 서브스텝이 큼): {int(under.sum())} 쌍"
              + (f"  최대 배율 {float(ratio[under].max()):.2f}x" if under.any() else ""))
        verdict_rows.append((name, int(missed.sum()), int(under.sum())))

    print("\n=== 판정 ===")
    if not mechanism_ok:
        print("**실패 — 서브스텝 훅이 걸리지 않았습니다.**")
        print(f"  스텝당 표본이 {sorted(set(samples))} 이고 decimation 은 "
              f"{base.cfg.decimation} 입니다. 표본이 1 이면 스냅샷 한 번과 같고, 그러면")
        print("  두 기록 스텝 사이의 접촉이 데이터에서 통째로 빠집니다.")
        print("  이 스크립트가 초록이면 그 기전이 살아 있다는 뜻이어야 합니다.")
    elif all(m == 0 and u == 0 for _, m, u in verdict_rows):
        print("기전은 정상입니다(표본 수 일치). 다만 이 실행에서는 스냅샷과 서브스텝")
        print("  최댓값이 같았습니다 — 이득이 0 이라는 뜻이 아니라 이 구간에 짧은 접촉이")
        print("  없었다는 뜻입니다. --seq 를 바꿔 접촉이 있는 에피소드에서 다시 재십시오.")
    else:
        print("스냅샷은 위 「놓친 접촉」 을 데이터에 남기지 못합니다. 짧게 기록되는 것이")
        print("아니라 0 이 되어 사라집니다. 지금 recorder 는 서브스텝 최댓값을 씁니다.")

    env.close()
    # 붉은 게이트: 기전이 죽었는데 초록으로 끝나면 이 스크립트는 아무것도 지키지
    # 않습니다. 「이번 구간에 이득이 없었다」 는 실패가 아니지만 「훅이 안 걸렸다」 는
    # 실패입니다.
    code = 0 if mechanism_ok else 1
except Exception:
    import traceback

    print("실행 실패:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    code = 1
finally:
    simulation_app.close(exit_code=code)

raise SystemExit(code)
