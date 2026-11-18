#!/usr/bin/env python3
"""`net_forces_w_history` 가 기록 스텝 사이의 접촉을 잡아 주는지 잰다.

    ~/isaacsim/python.sh generation/envs/s1a/check_contact_history.py [--steps 400] [--seq 2]

**왜 재는가.** `EpisodeRecorder.step()` 이 읽는 `contact.data.net_forces_w` 는 기록
스텝 시점의 순간 스냅샷입니다. 두 기록 스텝 사이(env 스텝 하나 = 물리 서브스텝 4회)
에서 생겼다 사라진 접촉은 지속시간이 짧게 잡히는 것이 아니라 **정확히 0 으로 기록되어
데이터에서 통째로 빠집니다.** 근접사고·실패를 담는 것이 이 데이터셋의 존재 이유이므로
결정적입니다 — 기록 주기(fps)를 정하는 근거가 됩니다.

센서는 이미 `history_length=3` 으로 설정돼 있어(`s1a_env_cfg.py`), 그 버퍼를 창 안
최댓값으로 읽으면 fps 를 올리지 않고도 잡을 수 있다는 **미확인 대안**이 있었습니다.
이 스크립트가 그 대안을 확인합니다.

**무엇이 답을 가르는가.** Isaac Lab 의 센서는 지연 평가입니다 —
`SensorBase.update(dt)` 는 outdated 플래그만 세우고, 실제 계산
(`_update_buffers_impl`, 히스토리를 한 칸 미는 곳)은 `.data` 에 접근할 때 일어납니다
(`sensor_base.py`). PhysX 백엔드는 `handles_decimation()` 이 False 라 `scene.update()`
가 서브스텝마다 불리지만, 그것은 플래그만 세웁니다. 그래서 히스토리 한 칸이
**서브스텝 하나**인지 **env 스텝 하나**인지는 코드만 읽어서는 갈리지 않고, 접근 패턴에
달려 있습니다. 여기서는 재서 답합니다.

**판정 방법.** 매 env 스텝에서 스냅샷과 히스토리를 함께 저장한 뒤,
`hist[t][k]` 가 `cur[t-k]` 와 같은지 셉니다.

  - 전부 일치 → 히스토리 한 칸 = **env 스텝 하나**. 그냥 읽는 것만으로는 대안이
    성립하지 않습니다 (이미 기록한 과거 스텝을 다시 볼 뿐입니다)
  - 불일치 → 한 칸이 서브스텝. 그때는 창 안 최댓값이 스냅샷보다 큰 스텝 수가
    이득의 크기입니다

**변형 둘을 각각 잽니다.**

  A. 그대로            — 운영 방식. `.data` 를 env 스텝당 한 번만 읽습니다
  B. 서브스텝 강제 읽기 — 액션 항의 `apply_actions()` 를 감싸 서브스텝마다 `.data` 를
                        건드립니다. `ManagerBasedRLEnv.step()` 의 데시메이션 루프가
                        `apply_action()` 을 서브스텝마다 부르므로(PhysX 는
                        `handles_decimation()` 이 False), 이것이 외부에서 센서를
                        서브스텝마다 계산시키는 **유일하게 찾은 훅**입니다.
                        `check_contact_channel.py` 가 쓰는 것과 같은 "살아 있는 항을
                        갈아 끼우는" 방식이라 나머지는 운영과 동일합니다

변형마다 환경을 새로 만듭니다(Kit 부팅은 1회). 시드가 같으므로 두 변형은 같은
에피소드를 돌립니다 — 차이는 읽기 시점 하나뿐입니다.

종료 코드: 0 판정이 나옴 / 1 실행 실패
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="접촉 히스토리 버퍼의 시간 해상도를 잰다")
parser.add_argument("--steps", type=int, default=400, help="측정할 env 스텝 수")
parser.add_argument("--seq", type=int, default=2, help="에피소드 시퀀스 번호 (시드가 파생된다)")
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.isaac_boot import launch_app

# Booting through the helper is what makes a boot failure exit nonzero.
simulation_app = launch_app()[1]

try:
    import importlib.metadata as metadata

    import numpy as np
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
    from generation.seeding import episode_seed, scenario_startup_seed

    SCENARIO_ID = "S1a"

    def build(force_substep_read):
        """Fresh env + policy. Optionally force a per-substep sensor compute."""
        cfg = resolve_presets(S1aEnvCfg(), selected=("physx",))
        cfg.seed = scenario_startup_seed(SCENARIO_ID)
        agent_cfg = load_cfg_from_registry("Keti-S1a-v0", "rsl_rl_cfg_entry_point")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Flat-H1-v0"))
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        base = env.unwrapped
        if force_substep_read:
            # ManagerBasedRLEnv.step() calls action_manager.apply_action() once
            # per physics substep on the PhysX path, and nothing else it calls
            # there touches sensor data. Wrapping a live action term is the one
            # hook available from outside the env for making the sensor compute
            # its buffers -- and therefore shift its history -- every substep.
            # Reading .data here sees the *previous* substep's physics result
            # (scene.update() marks outdated after sim.step()), so the history
            # ends up one substep behind; that offset does not change whether a
            # transient is captured at all, which is what is being measured.
            term = base.action_manager._terms[base.action_manager.active_terms[0]]
            original = term.apply_actions

            def apply_and_read():
                original()
                base.scene["contact_forces"].data.net_forces_w
                base.scene["dummy_contact"].data.net_forces_w

            term.apply_actions = apply_and_read

        env.seed(episode_seed(SCENARIO_ID, args.seq))
        with torch.inference_mode():
            obs, _ = env.reset()
        return env, policy, obs

    def read(sensor):
        """(snapshot (B,), history (H, B)) as force magnitudes in N."""
        cur = torch.linalg.norm(sensor.data.net_forces_w.torch[0], dim=-1).cpu().numpy()
        hist = sensor.data.net_forces_w_history
        if hist is None:
            return cur, None
        return cur, torch.linalg.norm(hist.torch[0], dim=-1).cpu().numpy()

    def run_variant(label, description, force_substep_read):
        env, policy, obs = build(force_substep_read)
        base = env.unwrapped
        # The robot's own foot sensor is the honest instrument for the timing
        # question: its forces change fast and are rarely all-zero, so "does
        # entry k equal the snapshot k steps ago" is answerable rather than
        # trivially true because everything is zero. The dummy sensor carries
        # the count that actually matters -- contacts a window max would catch
        # and a snapshot would not.
        robot_contact = base.scene["contact_forces"]
        dummy_contact = base.scene["dummy_contact"]

        robot_cur, robot_hist, dummy_cur, dummy_hist = [], [], [], []
        for _ in range(args.steps):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            c, h = read(robot_contact)
            robot_cur.append(c)
            robot_hist.append(h)
            c, h = read(dummy_contact)
            dummy_cur.append(c)
            dummy_hist.append(h)
            if bool(dones[0]):
                break

        n = len(robot_cur)
        out = dict(
            robot_cur=np.stack(robot_cur), robot_hist=np.stack(robot_hist),
            dummy_cur=np.stack(dummy_cur), dummy_hist=np.stack(dummy_hist), n=n,
            decimation=base.cfg.decimation, physics_dt=base.physics_dt,
            step_dt=base.step_dt, history_length=np.stack(robot_hist).shape[1],
        )
        env.close()

        print(f"\n=== 변형 {label}: {description} ===")
        print(f"decimation={out['decimation']}  physics_dt={out['physics_dt']}s  "
              f"step_dt={out['step_dt']}s  history_length={out['history_length']}  "
              f"측정 스텝={n}")

        H = out["history_length"]
        print("  히스토리 칸 ↔ 과거 스냅샷 (로봇 발 센서):")
        matches = []
        for k in range(H):
            if k >= n:
                break
            fwd = float(np.mean(np.all(
                out["robot_hist"][k:, k, :] == out["robot_cur"][: n - k, :], axis=-1)))
            bwd = float(np.mean(np.all(
                out["robot_hist"][k:, H - 1 - k, :] == out["robot_cur"][: n - k, :], axis=-1)))
            print(f"    k={k}: 앞에서 {fwd:6.1%} / 뒤에서 {bwd:6.1%}")
            matches.append(max(fwd, bwd))
        out["per_env_step"] = all(m > 0.99 for m in matches)

        print("  창 안 최댓값 대 스냅샷:")
        for name, cur, hist in (("로봇 발", out["robot_cur"], out["robot_hist"]),
                                ("더미", out["dummy_cur"], out["dummy_hist"])):
            win_max = hist.max(axis=1)
            missed = (cur <= 1e-6) & (win_max > 1e-6)
            rows = np.unique(np.nonzero(missed)[0])
            extra = f"  최대 {float(win_max[missed].max()):.2f} N" if missed.any() else ""
            print(f"    {name}: 스냅샷 0 인데 창 안에는 접촉 — (스텝,바디) "
                  f"{int(missed.sum())} 건 / 스텝 {len(rows)} 개{extra}")
            out[f"missed_{name}"] = int(missed.sum())
        return out

    a = run_variant("A", "그대로 (운영 방식, env 스텝당 1회 읽기)", False)
    b = run_variant("B", "액션 항을 감싸 서브스텝마다 읽기", True)

    print("\n=== 판정 ===")
    if a["per_env_step"]:
        print("변형 A: 히스토리 한 칸 = env 스텝 하나.")
        print("  → 그냥 읽는 것만으로는 얻는 것이 없습니다. 창 안 최댓값은 이미 기록한")
        print("    과거 스텝의 값일 뿐이고, 두 기록 스텝 사이에서 생겼다 사라진 접촉은")
        print("    여전히 데이터에 없습니다. 원인은 지연 평가입니다 — scene.update() 가")
        print("    서브스텝마다 불려도 outdated 플래그만 세우고, 히스토리를 미는")
        print("    _update_buffers_impl 은 .data 접근 시점(env 스텝당 1회)에만 돕니다.")
    else:
        print("변형 A: 히스토리가 env 스텝과 대응하지 않습니다 — 위 수치를 보십시오.")

    if not b["per_env_step"]:
        print("\n변형 B: 서브스텝마다 읽게 하면 히스토리가 env 스텝과 대응하지 않습니다")
        print("  → 서브스텝 해상도를 얻는 경로가 실재합니다. 위 '스냅샷 0 인데 창 안에는")
        print("    접촉' 수치가 fps 를 올리지 않고 얻는 이득의 크기입니다.")
        # A 와 B 의 같은 이름 수치를 나란히 놓으면 안 된다 — 뜻이 다르다.
        # A 의 건수는 이득이 아니라 허수다(이미 자기 시각에 기록된 과거 env 스텝의
        # 값을 다시 본 것). B 의 건수만이 "스냅샷이 통째로 놓치는 접촉" 이다.
        print(f"  더미 기준 실제 이득: {b['missed_더미']} 건 "
              f"(변형 B 에서만 유효 — A 의 {a['missed_더미']} 건은 과거 env 스텝을")
        print("    다시 본 것이라 새로 얻는 정보가 아닙니다)")
        print(f"  history_length({a['history_length']}) 가 decimation({a['decimation']}) 보다")
        print("  작으면 창의 일부만 덮습니다 — 채택 시 함께 올려야 합니다.")
    else:
        print("\n변형 B: 서브스텝마다 읽어도 히스토리는 여전히 env 스텝 단위입니다")
        print("  → 이 훅으로는 해상도를 얻을 수 없습니다. fps 를 올리는 것 외의 경로를")
        print("    이 스크립트는 찾지 못했습니다.")

    code = 0
except Exception:
    import traceback

    print("실행 실패:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    code = 1
finally:
    # See check_scene.py: close(exit_code=code) is what preserves a nonzero
    # exit code through Kit's fast-shutdown path.
    simulation_app.close(exit_code=code)

raise SystemExit(code)
