#!/usr/bin/env python3
"""`h1_minimal.usd` → `h1.usd` 교체가 사전학습 정책과 호환되는지 잰다.

    ~/isaacsim/python.sh generation/envs/s1a/check_asset_swap.py [--episodes 3] [--steps 400]

**왜.** S1a 가 실제로 싣는 `h1_minimal.usd` 는 20개 바디 중 **3개**(양 발목·토르소)
에만 콜라이더가 있습니다. 팔·다리로 인체를 건드려도 접촉이 원리적으로 생성되지
않는다는 뜻이고, 부위별 접촉이 이 데이터셋의 존재 이유이므로 직접적인 영향입니다.
자산 인벤토리가 BSD-3 정본으로 못박은 `h1.usd` 는 **20/20** 입니다
(`docs/assets/asset-license-inventory-v0.1.md` §2.3).

교체를 막고 있던 것은 하나 — 사전학습 정책(`Isaac-Velocity-Flat-H1-v0`)이
`h1_minimal.usd` 로 학습됐을 수 있다는 것입니다. 이 스크립트가 그것을 잽니다.

**자산 외에는 아무것도 다르지 않습니다.** Isaac Lab 의 `H1_MINIMAL_CFG` 는
`H1_CFG.copy()` 에 `spawn.usd_path` 한 줄만 바꾼 것입니다(`isaaclab_assets/robots/
unitree.py`) — 액추에이터·관절 설정·초기 자세가 전부 같습니다. USD 두 파일도
움직이는 관절 19개의 **이름과 순서가 동일**하고 바디 20개의 이름과 질량이
동일합니다(별도 실측). 따라서 정책의 관찰·행동 배열은 영향을 받지 않으며, 남는
위험은 콜라이더가 늘어 **자기 충돌과 접촉 동역학이 달라지는 것** 하나입니다.
그것은 걸어 보게 해서 재는 수밖에 없습니다.

두 변형을 같은 시드로 돌리고 비교합니다.

  A. `h1_minimal.usd` — 지금 운영 방식
  B. `h1.usd`         — 인벤토리가 가리키는 BSD-3 정본

종료 코드: 0 측정이 나옴 / 1 실행 실패. **초록이 곧 "교체하라" 는 아닙니다** —
판단 근거를 내놓을 뿐이고, 결론은 사람이 봅니다.
"""

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="H1 자산 교체의 정책 호환성을 잰다")
parser.add_argument("--episodes", type=int, default=3, help="변형당 에피소드 수")
parser.add_argument("--steps", type=int, default=400, help="에피소드당 최대 env 스텝")
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
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    import isaaclab_tasks  # noqa: F401
    import generation.envs.register  # noqa: F401

    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_tasks.utils.hydra import resolve_presets

    from generation.envs.s1a.s1a_env_cfg import S1aEnvCfg
    from generation.seeding import episode_seed, scenario_startup_seed

    SCENARIO_ID = "S1a"
    CONTACT_N = 1.0  # 접촉 있음/없음 판정용 잡음 문턱. Table A.2 비교가 아니다

    def build(usd_override):
        cfg = resolve_presets(S1aEnvCfg(), selected=("physx",))
        cfg.seed = scenario_startup_seed(SCENARIO_ID)
        if usd_override is not None:
            # The only difference between H1_CFG and H1_MINIMAL_CFG upstream.
            cfg.scene.robot.spawn.usd_path = usd_override
        agent_cfg = load_cfg_from_registry("Keti-S1a-v0", "rsl_rl_cfg_entry_point")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Flat-H1-v0"))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        return env, policy

    def run_variant(label, description, usd_override):
        env, policy = build(usd_override)
        base = env.unwrapped
        robot = base.scene["robot"]
        robot_contact = base.scene["contact_forces"]
        resolved = str(robot.cfg.spawn.usd_path)

        print(f"\n=== 변형 {label}: {description} ===")
        print(f"  로드한 자산: {resolved}")
        print(f"  접촉 센서 바디 수: {len(robot_contact.body_names)}")

        rows = []
        touched = {}
        for ep in range(args.episodes):
            env.seed(episode_seed(SCENARIO_ID, ep))
            with torch.inference_mode():
                obs, _ = env.reset()
            start = time.perf_counter()
            x0 = float(robot.data.root_pos_w.torch[0, 0])
            z_min, z_max = 1e9, -1e9
            vx = []
            steps = 0
            done = False
            for _ in range(args.steps):
                with torch.inference_mode():
                    obs, _, dones, _ = env.step(policy(obs))
                steps += 1
                z = float(robot.data.root_pos_w.torch[0, 2])
                z_min, z_max = min(z_min, z), max(z_max, z)
                vx.append(float(robot.data.root_lin_vel_b.torch[0, 0]))
                f = torch.linalg.norm(
                    robot_contact.data.net_forces_w.torch[0], dim=-1).cpu().numpy()
                for name, mag in zip(robot_contact.body_names, f):
                    if mag > CONTACT_N:
                        touched[name] = max(touched.get(name, 0.0), float(mag))
                if bool(dones[0]):
                    done = True
                    break
            x1 = float(robot.data.root_pos_w.torch[0, 0])
            rows.append(dict(ep=ep, steps=steps, done=done, travel=x1 - x0,
                             vx=float(np.mean(vx)), z_min=z_min, z_max=z_max,
                             wall=time.perf_counter() - start))

        for r in rows:
            end = "조기 종료(낙상/시간초과)" if r["done"] else "완주"
            print(f"  ep{r['ep']}: {r['steps']:4d} 스텝 {end}  전진 {r['travel']:+.2f} m  "
                  f"평균 v_x {r['vx']:+.3f} m/s  base z {r['z_min']:.2f}~{r['z_max']:.2f} m  "
                  f"{r['wall']:.1f}s")
        completed = sum(1 for r in rows if not r["done"])
        print(f"  요약: 완주 {completed}/{len(rows)}, "
              f"평균 전진 {np.mean([r['travel'] for r in rows]):+.2f} m, "
              f"평균 v_x {np.mean([r['vx'] for r in rows]):+.3f} m/s, "
              f"평균 {np.mean([r['wall'] for r in rows]):.1f}s/에피소드")
        print(f"  접촉을 한 번이라도 보고한 바디 {len(touched)}/"
              f"{len(robot_contact.body_names)}: "
              f"{', '.join(f'{k}({v:.0f}N)' for k, v in sorted(touched.items()))}")
        # --- forced overlap: what the asset *can* report at all ------------
        # The walking episodes above answer compatibility, not coverage: in a
        # normal episode the only thing the robot touches is the ground, so
        # both assets report the same two ankles and the collider difference
        # never shows. Driving the dummy through the robot is what makes the
        # difference visible -- with 3/20 collision scopes the arms and legs
        # cannot report a contact no matter what touches them.
        cmd_term = base.command_manager.get_term("base_velocity")
        cmd_term._advance_dummy = lambda: None  # stop the hold fighting the teleport
        dummy = base.scene["dummy"]
        env.seed(episode_seed(SCENARIO_ID, 0))
        with torch.inference_mode():
            obs, _ = env.reset()
        for _ in range(20):  # let the robot settle into a normal stance first
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))

        overlap = {}
        with torch.inference_mode():
            for _ in range(20):
                # Put the dummy where the robot is, upright and still, every
                # step -- the robot keeps walking, so the two sweep through
                # each other rather than resting in one pose.
                root = robot.data.root_pos_w.torch[0]
                state = torch.zeros(1, 13, device=base.device)
                state[0, :3] = root
                state[0, 6] = 1.0  # (x, y, z, w) identity quaternion
                dummy.write_root_state_to_sim(state)
                obs, _, _, _ = env.step(policy(obs))
                f = torch.linalg.norm(
                    robot_contact.data.net_forces_w.torch[0], dim=-1).cpu().numpy()
                for name, mag in zip(robot_contact.body_names, f):
                    if mag > CONTACT_N:
                        overlap[name] = max(overlap.get(name, 0.0), float(mag))

        non_foot = sorted(n for n in overlap if not n.endswith("ankle_link"))
        print(f"  강제 겹침 20 스텝: 접촉을 낸 바디 {len(overlap)}/"
              f"{len(robot_contact.body_names)}, 그중 발이 아닌 것 {len(non_foot)}: "
              f"{', '.join(f'{n}({overlap[n]:.0f}N)' for n in non_foot) or '없음'}")

        # Read the body count before close(): the sensor's PhysX view is torn
        # down with the env, and body_names then raises on a None view.
        n_bodies = len(robot_contact.body_names)
        env.close()
        return dict(rows=rows, touched=touched, resolved=resolved, n_bodies=n_bodies,
                    overlap=overlap, non_foot=non_foot)

    # Both variants name their asset explicitly. Letting variant A fall
    # through to the cfg default made this script compare h1.usd with itself
    # the moment the default was switched to h1.usd -- and it still exited 0,
    # so the comparison it prints would have looked like a result. The default
    # is where the paths come from, but which file each variant loads is
    # stated here, and a mismatch is fatal rather than a warning.
    default_usd = str(resolve_presets(S1aEnvCfg(), selected=("physx",)).scene.robot.spawn.usd_path)
    if default_usd.endswith("h1_minimal.usd"):
        minimal_usd, full_usd = default_usd, default_usd.replace("h1_minimal.usd", "h1.usd")
    elif default_usd.endswith("h1.usd"):
        full_usd, minimal_usd = default_usd, default_usd.replace("h1.usd", "h1_minimal.usd")
    else:
        # RuntimeError, not SystemExit: this is inside the try whose handler
        # is `except Exception`, which does not catch SystemExit (it derives
        # from BaseException). Raising SystemExit here skipped the handler,
        # left `code` unassigned, and the finally block died with NameError --
        # so the diagnosis was replaced by an unrelated error and
        # close(exit_code=1) never ran. Reproduced on the same control flow.
        raise RuntimeError(f"로봇 자산 경로가 예상 밖입니다: {default_usd}")

    a = run_variant("A", "h1_minimal.usd (콜라이더 3/20)", minimal_usd)
    b = run_variant("B", "h1.usd (BSD-3 정본, 콜라이더 20/20)", full_usd)
    for label, got, want in (("A", a["resolved"], minimal_usd), ("B", b["resolved"], full_usd)):
        if got != want:
            raise RuntimeError(f"변형 {label} 가 요청한 자산을 싣지 않았습니다: {got} != {want}")
    if a["resolved"] == b["resolved"]:
        raise RuntimeError(f"두 변형이 같은 자산을 실었습니다: {a['resolved']}")

    print("\n=== 비교 ===")
    ca = sum(1 for r in a["rows"] if not r["done"])
    cb = sum(1 for r in b["rows"] if not r["done"])
    va = float(np.mean([r["vx"] for r in a["rows"]]))
    vb = float(np.mean([r["vx"] for r in b["rows"]]))
    wa = float(np.mean([r["wall"] for r in a["rows"]]))
    wb = float(np.mean([r["wall"] for r in b["rows"]]))
    print(f"  완주      A {ca}/{len(a['rows'])}   B {cb}/{len(b['rows'])}")
    print(f"  평균 v_x  A {va:+.3f}   B {vb:+.3f} m/s")
    print(f"  접촉 바디 A {len(a['touched'])}/{a['n_bodies']}   B {len(b['touched'])}/{b['n_bodies']}")
    print(f"  속도      A {wa:.1f}   B {wb:.1f} s/에피소드")
    gained = sorted(set(b["touched"]) - set(a["touched"]))
    print(f"  정상 보행에서 B 가 새로 접촉을 낸 바디 {len(gained)}: {', '.join(gained) or '없음'}")
    print(f"  강제 겹침: A {len(a['overlap'])}/{a['n_bodies']} (발 외 {len(a['non_foot'])})"
          f"   B {len(b['overlap'])}/{b['n_bodies']} (발 외 {len(b['non_foot'])})")
    new_regions = sorted(set(b["overlap"]) - set(a["overlap"]))
    print(f"  강제 겹침에서 B 만 낸 바디 {len(new_regions)}: {', '.join(new_regions) or '없음'}")
    print("\n  ※ 이 스크립트는 근거만 냅니다. 교체 여부는 사람이 판단합니다 —"
          " 걸음이 무너지면(완주 수·평균 v_x 가 A 보다 뚜렷이 나쁨) 정책 재학습이"
          " 필요하다는 뜻입니다.")
    code = 0
except Exception:
    import traceback

    print("실행 실패:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    code = 1
finally:
    # See check_scene.py: close(exit_code=code) is what preserves a nonzero code.
    simulation_app.close(exit_code=code)

raise SystemExit(code)
