#!/usr/bin/env python3
"""여러 시드·진입 기하에 걸쳐 로봇-더미 접촉력의 분포(자릿수·폭)를 잰다.

    ~/isaacsim/python.sh generation/envs/s1a/check_contact_force_sweep.py

`check_fall_termination.py` 가 접촉력이 물리적 범위(수백 N)임을 시드 하나로
확인했다. `observation.contact.force` 기록기를 짜기 전에, 그 값이
시드마다·진입 기하(속도·각도·트리거 거리)마다 어느 자릿수·폭으로 나오는지
미리 재 두는 것이 기록기를 다 짠 뒤에 재는 것보다 싸다.

앞서 잰 349~2803N 폭은 재현되지 않았는데, GPU 비결정성이
아니라 **로봇 발 착지가 섞인 오염 리스트**였을 가능성이 높다(730.04N
을 비트 동일 재현하면서 BR_04/torso_link 페어로 귀속을 확인함). 이 스크립트는
`check_fall_termination.py` 와 같은 귀속 방식(더미 자신의 센서, 자기 무게
부위 제외 — 1차 근거)을 그대로 써서, 로봇 발이 섞이지 않은 깨끗한 분포를 낸다.

시드마다 실제 정책으로 여러 에피소드를 돌리며 자연 발생 접촉(강제 없음)을
찾는다. `--seeds`/`--episodes-per-seed`/`--max-steps` 로 범위를 조절한다.

집계에서 지키는 두 가지:

  1. "에피소드" 루프가 `--max-steps` 안에 `dones` 가 서는지 확인한다
     — 안 서면(리셋이 실제로 일어나지 않으면) 다음 `episode` 반복은 새로운
     독립 시행이 아니라 같은 에피소드의 연장이다.
     `check_entry.py` 가 그런 에피소드에 내는 것과 같은 `[주의]` 경고를 내고, 리셋이 확인된
     횟수만 "독립 시행" 분모로 보고한다(시도 횟수는 참고로 남긴다).
  2. `all_forces` 의 개별 스텝 표본을 그대로 통계 모집단으로 쓰지 않는다 — 한 번의
     접촉은 물리적으로 여러 스텝에 걸쳐 힘이 오르내리며 찍히고, 문턱값
     (1e-6N) 아래로 잠깐 빠지는 중간 스텝도 있다(실측: 430.59N/
     1551.44N/2050.67N, seed=2 ep=0 BR_04, step 408/410/416 — 간격 2·6
     스텝이지만 독립된 세 접촉이 아니라 하나의 상승 국면). (seed, episode,
     body) 안에서 `--event-gap-steps` 이내로 떨어진 스텝을 하나의 "접촉
     사건"으로 묶고, 사건별 피크 힘을 1차 분포 통계로 쓴다. 원시 스텝 표본
     통계는 참고로만 남긴다.

종료 코드: 0 접촉이 하나 이상 관측되고 전부 물리적 범위(5000N 미만, 0 초과)
/ 1 접촉이 하나도 안 관측됐거나 비물리적 값이 있음 또는 실행 실패
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

parser = argparse.ArgumentParser(description="시드 스윕 — 접촉력 분포 실측")
parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7", help="쉼표로 구분된 환경 시드 목록")
parser.add_argument("--episodes-per-seed", type=int, default=2, help="시드당 돌릴 에피소드 수")
parser.add_argument("--max-steps", type=int, default=1000, help="에피소드당 최대 스텝")
parser.add_argument("--force-cap-n", type=float, default=5000.0, help="이 값을 넘으면 비물리적으로 판정")
# 한 번의 물리적 접촉이라도 매 스텝 문턱값(1e-6N) 이상으로 잡히리라는 보장이
# 없다 — 이전 실측(430.59N/1551.44N/2050.67N, seed=2 ep=0 BR_04, step 408/
# 410/416)이 정확히 그 예다: 같은 몸-바디의 같은 사건인데도 중간 스텝은
# 문턱값 아래로 빠졌다(간격 2, 6 스텝). 이 값에서 유도한 물리 상수는 아니다
# — 알려진 예의 최대 간격(6)을 여유 있게 덮으면서, 시드 스윕에서 서로 다른
# 에피소드/사건 사이의 간격(보통 수백 스텝)과는 자릿수가 다른, 명시적이고
# 조정 가능한 값이다.
parser.add_argument("--event-gap-steps", type=int, default=10, help="같은 접촉 사건으로 묶을 최대 스텝 간격")
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

    from generation.envs.s1a.recorder import ground_contact_body_names
    from generation.envs.s1a.s1a_env_cfg import DUMMY_USD, S1aEnvCfg

    def nonzero_map(sensor, forces) -> dict:
        return {n: f for n, f in zip(sensor.body_names, forces.tolist()) if f > 1e-6}

    def _summarize_event(seed: int, episode: int, body: str, run: list[tuple[int, float]]) -> dict:
        """One contiguous run of consecutive-step force samples -> one event.

        See the module docstring's second point: a single physical
        contact is not one instantaneous sample, it is force rising and
        falling over several consecutive steps as the two bodies press
        together and separate. peak_force (not mean — a contact's most
        safety-relevant number is how hard it got, not its time-average) is
        what feeds the event-level distribution reported below.
        """
        steps = [p[0] for p in run]
        forces = [p[1] for p in run]
        return {
            "seed": seed,
            "episode": episode,
            "body": body,
            "step_start": steps[0],
            "step_end": steps[-1],
            "n_steps": len(steps),
            "peak_force": max(forces),
        }

    all_forces: list[tuple[int, int, int, float, str]] = []  # (seed, ep, step, force, body)
    episodes_with_contact = 0
    episodes_total = 0
    # Attempted episodes where `dones` never fired within --max-steps are not
    # independent draws — the *next* loop iteration would just be continuing
    # the same unfinished episode further, not starting a fresh one (the
    # same guard as check_entry.py's `dones`-never-fired warning). Counted
    # separately so the summary's denominator is the confirmed-independent
    # count, not the raw attempt count.
    episodes_confirmed_reset = 0

    for seed in seeds:
        cfg = resolve_presets(S1aEnvCfg(), selected=("physx",))
        cfg.seed = seed
        agent_cfg = load_cfg_from_registry("Keti-S1a-v0", "rsl_rl_cfg_entry_point")
        installed_version = metadata.version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Flat-H1-v0"))
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        dummy_sensor = env.unwrapped.scene["dummy_contact"]
        robot_sensor = env.unwrapped.scene["contact_forces"]
        ground_bodies = {n for n in robot_sensor.body_names if n.endswith("ankle_link")}
        # Structural ground-contact set, read through the asset's own
        # keti:bodyRegion tag (BR-12) rather than hand-typed as
        # BR_12_left/right (CONTRIBUTING.md 「관절 순서」) — the same helper
        # check_fall_termination.py/check_dummy_hold.py use.
        dummy_own_weight_bodies = ground_contact_body_names(str(DUMMY_USD), dummy_sensor.body_names)

        obs = env.get_observations()
        for episode in range(args.episodes_per_seed):
            episodes_total += 1
            episode_had_contact = False
            done = False
            for step in range(args.max_steps):
                with torch.inference_mode():
                    obs, _, dones, extras = env.step(policy(obs))
                dummy_forces = nonzero_map(dummy_sensor, dummy_sensor.data.net_forces_w.norm(dim=-1).squeeze(0))
                dummy_new = {n: f for n, f in dummy_forces.items() if n not in dummy_own_weight_bodies}
                if dummy_new:
                    episode_had_contact = True
                    for body, force in dummy_new.items():
                        all_forces.append((seed, episode, step, force, body))
                if bool(dones[0]):
                    done = True
                    break
            if episode_had_contact:
                episodes_with_contact += 1
            if done:
                episodes_confirmed_reset += 1
            else:
                # The same guard as check_entry.py's warning: without a confirmed reset,
                # the *next* `episode` iteration continues this one further
                # rather than drawing a fresh, independent scenario — any
                # force samples collected during it are real, but this
                # "episode" is not the independent statistical unit its
                # index implies.
                print(f"  [주의] {args.max_steps} 스텝 안에 dones 가 서지 않음 (비정상) — "
                      f"seed={seed} ep={episode} 는 독립 시행으로 세지 않음")
            print(f"seed={seed} ep={episode} 접촉관측={'예' if episode_had_contact else '아니오'} "
                  f"리셋확인={'예' if done else '아니오'}")
        env.close()

    print(f"\n=== 요약: 시도 {episodes_total}개 중 리셋으로 확인된 독립 시행 {episodes_confirmed_reset}개 ===")
    if episodes_confirmed_reset < episodes_total:
        print(
            f"  [참고] {episodes_total - episodes_confirmed_reset}개 시행은 --max-steps 안에 리셋이 "
            f"확인되지 않아 독립 시행 수에서 제외함 (접촉 표본 자체는 all_forces 에 남아 있음)"
        )
    print(f"  접촉 관측된 시행(리셋 확인 여부 무관) {episodes_with_contact}개 / 시도 {episodes_total}개")

    if all_forces:
        values = [f for (_, _, _, f, _) in all_forces]
        values.sort()
        n = len(values)
        print(f"\n원시 스텝 표본 {n}개 (더미 센서, 자기무게 부위 제외 — 참고용, 독립 표본이 아님):")
        print(f"  최솟값={values[0]:.2f}N 최댓값={values[-1]:.2f}N "
              f"중앙값={values[n // 2]:.2f}N 평균={sum(values) / n:.2f}N")
        # A handful of extremes, with attribution, for spot-checking.
        by_force = sorted(all_forces, key=lambda t: -t[3])
        print("  최댓값 상위 5개 (seed, ep, step, force, body):")
        for row in by_force[:5]:
            print(f"    {row}")

        # Primary distribution: one peak force per contact *event*, not per
        # step — a single physical contact grows and
        # decays over several consecutive steps (previously measured:
        # 430.59N/1551.44N/2050.67N at steps 408/410/416 was one rising
        # event, not three independent draws), so treating every raw step
        # as its own sample overstates how many independent observations
        # there actually are and biases the distribution toward however
        # long a typical event happens to last. Grouped per (seed, episode,
        # body): a gap in that body's own recorded steps starts a new event.
        events: list[dict] = []
        by_series: dict[tuple[int, int, str], list[tuple[int, float]]] = {}
        for s, ep, step, force, body in all_forces:
            by_series.setdefault((s, ep, body), []).append((step, force))
        for (s, ep, body), pts in by_series.items():
            pts.sort()
            run = [pts[0]]
            for step, force in pts[1:]:
                # <= gap tolerance, not strict step+1 adjacency (see
                # --event-gap-steps above) — a real recorded example (seed=2
                # ep=0 BR_04, steps 408/410/416) has gaps of 2 and 6 within
                # what is otherwise clearly one rising-then-recorded contact,
                # so requiring immediate adjacency would still split it into
                # three "events" and miss the exact case this fix targets.
                if step - run[-1][0] <= args.event_gap_steps:
                    run.append((step, force))
                else:
                    events.append(_summarize_event(s, ep, body, run))
                    run = [(step, force)]
            events.append(_summarize_event(s, ep, body, run))

        peaks = sorted(e["peak_force"] for e in events)
        m = len(peaks)
        print(f"\n접촉 사건(연속 스텝 묶음) {m}개 — 사건별 피크 힘 분포 (1차 통계):")
        print(f"  최솟값={peaks[0]:.2f}N 최댓값={peaks[-1]:.2f}N "
              f"중앙값={peaks[m // 2]:.2f}N 평균={sum(peaks) / m:.2f}N")
        by_peak = sorted(events, key=lambda e: -e["peak_force"])
        print("  피크 상위 5개 사건 (seed, ep, body, step범위, n스텝, peak):")
        for e in by_peak[:5]:
            print(
                f"    seed={e['seed']} ep={e['episode']} body={e['body']} "
                f"step={e['step_start']}~{e['step_end']} n={e['n_steps']} peak={e['peak_force']:.2f}N"
            )

        over_cap = [v for v in values if v >= args.force_cap_n]
        under_zero = [v for v in values if v <= 0.0]
    else:
        over_cap = []
        under_zero = []

    failures = []
    if not all_forces:
        failures.append("접촉이 시드 스윕 전체에서 한 번도 관측되지 않음")
    if over_cap:
        failures.append(f"{len(over_cap)}개 표본이 --force-cap-n={args.force_cap_n}N 이상 — 비물리적 의심")
    if under_zero:
        failures.append(f"{len(under_zero)}개 표본이 0 이하 — 센서 읽기 결함 의심")

    if failures:
        print("\n[결함]")
        for f in failures:
            print(f"  - {f}")
        code = 1
    else:
        print("\n접촉력이 물리적 범위 안에서 관측됨.")
        code = 0
except Exception:
    import traceback

    print("check_contact_force_sweep 실행 실패:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    code = 1
finally:
    # See check_scene.py: close(exit_code=code) is what preserves a nonzero
    # code through this build's fast-shutdown path.
    simulation_app.close(exit_code=code)

raise SystemExit(code)
