#!/usr/bin/env python3
"""더미를 매 스텝 강제로 세워 두는 쓰기 방식이 PhysX 접촉 보고를 죽이는지 확인한다.

참고: "변형 C(현재 운영 방식)"라는 라벨은 이 스크립트가 작성된 시점
기준이다. 지금의 _advance_dummy 는 텔레포트(write_root_state_to_sim 매 스텝)가 아니라
유한한 스프링-댐퍼 힘(permanent_wrench_composer)을 쓰므로, `patch=None`
분기는 더 이상 실제 운영 방식이 아니라 "폐기된 이전 방식"을 재현한다. 세 변형의
상대적 비교(A 정상, B 발산, C 정적 접촉 0)는 그대로 유효한 역사적 근거이지만,
현재 운영 방식의 접촉 채널 검증은 check_dummy_hold.py·check_fall_termination.py 를
보십시오 — 두 스크립트 모두 정적 접촉이 0 이 아님(수백 N)을 새 방식으로 실측했습니다.

    ~/isaacsim/python.sh generation/envs/s1a/check_contact_channel.py [--steps 60]

근접 조우에서 `dummy_contact` 힘이 전부 0.0000 N 으로 관측됐다. 한 가지
설명은 "근사 지표(min_distance_human)가 실제보다 보수적이라 근사만 0 에
먼저 도달한 것"이다. 그 설명이 맞는지 가르는
진단 두 개 중 하나가 이 스크립트다(나머지 하나는 check_forced_overlap.py).

**여기서 나오는 0 을 "접촉 채널이 죽었다"로 읽지 말 것.**
check_forced_overlap.py 로 보면, 같은 쓰기 방식에서도 실제 관통이
생기면 양쪽 센서가 2만 N 대를 정상적으로 보고한다. 변형 C 의 0 은 그보다
좁은 사실이다 — 매 스텝 위치·자세·속도를 다시 쓰면 지면으로의 침투도
상대속도도 남지 않아 솔버가 낼 접촉 임펄스가 없다. 즉 **이 쓰기 방식 아래서는
"눌리고 있으나 움직이지 않는" 정적 접촉(자기 무게로 지면을 딛는 것, 사람이
로봇에 기대는 것)이 힘으로 보고되지 않는다**는 것이 이 스크립트가 재는 것이다.

여기서 묻는 것: **접촉이 보고되지 않는 원인이 쓰기 방식 자체인가.**
`check_scene.py` 는 같은 씬·같은 자산에서 428.7 N/435.0 N 을 냈는데 인체 진입을 넣은
실행은 0.0000 N 을 냈다. 둘 사이의 차이는 `S1aStraightWalkCommand._advance_dummy`
(mdp/commands.py) 가 매 스텝 루트의 위치·자세·속도를 다시 쓴다는 것 하나뿐이라,
그 쓰기만 갈아 끼우고 나머지를 전부 고정한 채 접지 반력을 잰다.

세 변형(로봇 행동은 전부 0 으로 고정 — 정책을 태우지 않는다):

  A. 쓰기 없음      — 더미를 자유롭게 두고 중력에 맡긴다(정착/기울어짐)
  B. 위치만 텔레포트 — `write_root_pose_to_sim_index` (속도는 안 건드림)
  C. 위치+속도 강제  — 운영 방식 그대로(`_advance_dummy` 를 갈아 끼우지 않음)

변형마다 새 환경을 만든다(Kit 부팅은 1회) — 한 변형이 남긴 물리 상태가 다음
변형에 새지 않게 하려는 것이다. 로봇 행동이 0 이면 로봇은 기본 자세를 유지한
채 거의 제자리에 있어 along-track 트리거가 걸리지 않으므로, 변형 C 의 운영
`_advance_dummy` 는 여기서 "대기 자세를 매 스텝 다시 쓰는 것"과 같다 — 즉 세
변형의 차이는 쓰기 방식 하나뿐이다.

**더미의 쓰기 방식은 이 스크립트가 고치는 대상이 아니다** — 무엇을 채택할지는
이 스크립트 밖에서 정한다. 이 스크립트는 채택된 방식에서 채널이 실제로 살아 있는지를
재는 계측기다.

종료 코드: 0 = 운영 방식(C)에서 더미를 떠받치는 접지 반력이 보고됨 /
1 = C 에서 보고되지 않음(이 글을 쓰는 시점의 상태) 또는 실행 실패

`generation/dummy/check_contact_resolution.py` 와 같은 선례다: 아직 해결되지
않은 문제를 재는 스크립트라 **현재 실패(1) 상태로 커밋한다.** 더미를 떠받치는
방식이 바뀌어(예: 힘 제어) 정적 접촉이 다시 잡히면 이 스크립트가 그대로 0 으로
뒤집힌다.
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

parser = argparse.ArgumentParser(description="더미 루트 쓰기 방식별 접촉 보고 확인")
parser.add_argument("--steps", type=int, default=60, help="변형마다 돌릴 스텝 수")
parser.add_argument("--seed", type=int, default=0, help="환경 시드")
args = parser.parse_args()

from generation.isaac_boot import launch_app

# Booting through the helper is what makes a boot failure exit nonzero: the
# try/finally below only arms once simulation_app exists, and until it does,
# Kit's own atexit handler would close with its default exit_code=0 and hand
# back a green process. See generation/isaac_boot.py for the measurement.
simulation_app = launch_app()[1]

try:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.utils.hydra import resolve_presets

    from generation.envs.s1a.s1a_env_cfg import S1aEnvCfg

    def hold_pose(dummy, entry, num_envs, device, *, with_velocity: bool):
        """Rewrite the dummy's waiting pose — with or without the velocity write.

        Same three quantities _advance_dummy writes (position, upright
        orientation, zero velocity), minus the velocity term when
        with_velocity is False. Nothing else differs between the variants.
        """
        pos = torch.cat([entry["target_xy"], entry["stand_z"].unsqueeze(-1)], dim=-1)
        quat = torch.zeros(num_envs, 4, device=device)
        quat[:, 3] = 1.0  # (x, y, z, w) identity — upright, no yaw
        if with_velocity:
            zero_vel = torch.zeros(num_envs, 6, device=device)
            dummy.write_root_state_to_sim(torch.cat([pos, quat, zero_vel], dim=-1))
        else:
            dummy.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1))

    def run_variant(label: str, description: str, patch):
        cfg = resolve_presets(S1aEnvCfg(), selected=("physx",))
        cfg.seed = args.seed
        env = ManagerBasedRLEnv(cfg=cfg)
        env.reset()

        cmd_term = env.command_manager.get_term("base_velocity")
        dummy = env.scene["dummy"]
        sensor = env.scene["dummy_contact"]
        entry = env.s1a_entry
        if patch is not None:
            # Swap the write scheme on the live term — no new code path, and
            # everything else in the episode stays exactly what production runs.
            cmd_term._advance_dummy = patch(dummy, entry, env.num_envs, env.device)

        zero_action = torch.zeros_like(env.action_manager.action)
        for _ in range(args.steps):
            env.step(zero_action)

        forces_n = sensor.data.net_forces_w.norm(dim=-1).squeeze(0)  # (num_bodies,)
        names = sensor.body_names
        nonzero = [(n, f) for n, f in zip(names, forces_n.tolist()) if f > 1e-6]
        total = float(forces_n.sum().item())
        print(f"\n=== 변형 {label}: {description} ===")
        print(f"  {args.steps} 스텝 후 접촉력 합계 {total:.4f} N, 최대 {forces_n.max().item():.4f} N")
        print("  0 초과 부위: " + (
            ", ".join(f"{n}={f:.2f}N" for n, f in nonzero) if nonzero else "없음 (전부 0.0000N)"
        ))
        env.close()
        return total

    def no_write(dummy, entry, num_envs, device):
        return lambda: None

    def position_only(dummy, entry, num_envs, device):
        return lambda: hold_pose(dummy, entry, num_envs, device, with_velocity=False)

    totals = {}
    totals["A"] = run_variant("A", "쓰기 없음(자유 낙하/정착)", no_write)
    totals["B"] = run_variant("B", "위치만 텔레포트(속도 미변경)", position_only)
    # patch=None: leave production's own _advance_dummy in place.
    totals["C"] = run_variant("C", "위치+속도 강제 — 현재 운영 방식", None)

    print("\n=== 요약 ===")
    for label in ("A", "B", "C"):
        print(f"  변형 {label}: 접촉력 합계 {totals[label]:.4f} N")

    # Same relative threshold and reasoning as check_scene.py: a geometry-only
    # build has no meta/rules.json masses, so there is no physically meaningful
    # absolute newton value to compare against — variant A's own settled load
    # is what "the dummy's weight" means here, and 1% of it clears solver noise
    # by two orders of magnitude.
    threshold_n = max(0.01 * totals["A"], 1e-6)
    alive = totals["C"] > threshold_n
    if alive:
        print(f"운영 방식(C)에서 접지 반력이 보고됩니다 ({totals['C']:.4f} N > {threshold_n:.4f} N)"
              " — 접촉 채널이 살아 있습니다")
        code = 0
    else:
        print(f"[결함] 운영 방식(C)에서 접지 반력이 보고되지 않습니다 ({totals['C']:.4f} N <= {threshold_n:.4f} N)"
              " — 매 스텝 위치·자세·속도를 다시 쓰면 침투도 상대속도도 남지 않아,")
        print("       솔버가 낼 접촉 임펄스가 없습니다. 정적 접촉(자기 무게로 딛기, 기대기)이"
              " observation.contact.force 에 0 으로 남습니다.")
        print("       (관통이 실제로 생기는 접촉은 이 방식에서도 보고됩니다 —"
              " check_forced_overlap.py 2단계 참고. 이 스크립트의 0 은 그보다 좁은 사실입니다.)")
        code = 1
except Exception:
    import traceback

    print("check_contact_channel 실행 실패:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    code = 1
finally:
    # See check_scene.py: close(exit_code=code) is what preserves a nonzero
    # code through this build's fast-shutdown path.
    simulation_app.close(exit_code=code)

raise SystemExit(code)
