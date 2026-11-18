#!/usr/bin/env python3
"""더미를 로봇 안에 강제로 관통시켜도 접촉이 보고되는지 확인한다.

참고: 여기서 스냅에 쓰는 write_root_state_to_sim 텔레포트와, 1단계가
끄려는 base_contact 종료항 둘 다 이 스크립트가 작성된 시점의 운영 방식을 반영한다.
지금의 _advance_dummy 는 PD 스프링이고, base_contact 자리에는 fall_height/
fall_orientation 이 있다 — s1a_env_cfg.py 에 base_contact 항목 자체가 더 이상
없으므로 1단계의 disable_base_contact=False/True 구분은 이제 둘 다 이미 None 이라
사실상 같은 구성이다. 아래 2단계 수치(재설정 0회, 더미 센서 최대 22249N)는 "텔레포트로
갑자기 겹치면 채널이 살아 있다"는 역사적 근거로는 유효하지만, 그 힘의 크기 자체는
텔레포트가 만드는 인위적 관통의 산물이지 지금의 운영 방식이 내는 값이 아니다(같은
현상이 check_fall_termination.py 를 만드는 과정에서도 재현됐다 — 순간이동으로 시작한
겹침은 PD 홀드에서도 수만 N 짜리 솔버 충격을 낸다). 현재 운영 방식에서의 접촉
크기·정지 여부 검증은 check_dummy_hold.py·check_fall_termination.py 를 보십시오
(자연스러운 접근에서 실측 수백 N, 접촉이 에피소드를 끝내지 않음).

    ~/isaacsim/python.sh generation/envs/s1a/check_forced_overlap.py \
        [--steps 40] [--baseline-steps 10]

check_contact_channel.py 가 "운영 쓰기 방식(위치+속도 강제)에서 더미의 접지
반력이 0 이다"까지 보인다. 거기에는 무해한 설명이 하나 있다 — 더미를 매 스텝
정확히 서 있는 자세·속도 0 으로 다시 쓰면 지면으로의 침투도 상대속도도 0 이라,
솔버가 낼 접촉 임펄스가 애초에 없다. 이 스크립트는 그 설명을 배제하기 위해
**침투를 인위적으로 만든다**: 쓰기 방식은 운영과 똑같이 두고
(`write_root_state_to_sim`), 목표 위치만 로봇의 현재 루트 위치로 매 스텝
스냅해 루트 간 거리를 0.0000 m 로 고정한다.

**두 단계로 잰다 — 1단계만 재면 결과를 잘못 읽는다.**

  1단계: 환경을 그대로 둔 채 스냅. 로봇 몸통이 더미와 겹치므로 로코모션 환경의
         `base_contact` 종료항(`.*torso_link` 접촉 1.0 N 초과)이 걸려 **매 스텝
         에피소드가 재설정된다.** 재설정은 센서 버퍼를 0 으로 비우고, 그 뒤에
         읽으면 당연히 0 이 나온다 — 쓰기 방식과 무관한 0 이다. 이 단계는
         재설정 횟수와 어떤 종료항이 걸렸는지를 함께 찍어 그 사실을 드러낸다.
  2단계: `base_contact` 종료항만 끈 새 환경에서 같은 스냅을 반복한다. 재설정이
         없으므로 관통이 유지되고, 그 상태에서 센서가 실제로 무엇을 보고하는지
         읽을 수 있다. **판정은 이 단계로 한다.**

로봇 행동은 0 으로 고정한다(정책을 태우지 않는다). Isaac Lab 로코모션 환경의
행동은 기본 관절자세 기준 오프셋이라 행동 0 은 "기본 자세를 유지하고 서 있으라"
이지 "관절을 풀어라"가 아니다 — 로봇이 제자리에 서 있어야 루트 간 거리가
0 으로 유지된다.

더미 자신의 센서(`dummy_contact`)와 로봇 자신의 센서(`contact_forces`) 를 둘 다
읽는다. 로봇 센서는 발이 지면을 딛고 있어 접지 반력이 잡히는 것이 정상이므로,
스냅 전 대조 구간에서 힘이 잡히는 부위 집합을 먼저 재고(= 접지 부위), 스냅
구간에서는 **그 집합 밖의 부위**에 잡히는 힘을 로봇-더미 접촉의 증거로 본다.
대조 구간은 "이 센서가 애초에 힘을 잡기는 하는가"도 함께 확인한다 — 없으면
0 이라는 결과를 해석할 수 없다.

**더미의 쓰기 방식은 이 스크립트가 고치는 대상이 아니다** — 무엇을 채택할지는
이 스크립트 밖에서 정한다. 이 스크립트는 계측기다.

종료 코드: 0 = 재설정 없는 2단계에서 접촉이 보고됨(채널 살아 있음) /
1 = 2단계에서도 전부 0 (채널이 죽어 있음) 또는 실행 실패
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

parser = argparse.ArgumentParser(description="강제 관통 상태의 접촉 보고 확인")
parser.add_argument("--steps", type=int, default=40, help="관통을 강제할 스텝 수")
parser.add_argument(
    "--baseline-steps",
    type=int,
    default=10,
    help="관통을 걸기 전 대조로 돌릴 스텝 수 (로봇 센서가 애초에 힘을 잡기는 하는지 확인)",
)
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

    def nonzero_map(sensor, forces) -> dict:
        return {n: f for n, f in zip(sensor.body_names, forces.tolist()) if f > 1e-6}

    def fmt(force_map) -> str:
        return ", ".join(f"{n}={f:.2f}N" for n, f in force_map.items()) or "없음 (전부 0.0000N)"

    def run_phase(label: str, disable_base_contact: bool) -> dict:
        print(f"\n########## {label} ##########")
        cfg = resolve_presets(S1aEnvCfg(), selected=("physx",))
        cfg.seed = args.seed
        if disable_base_contact:
            # The only reason to touch a termination term in a diagnostic: with
            # it on, the overlap this script exists to create is erased by a
            # reset before anything can be read from it (phase 1 measures
            # exactly that). time_out stays on — it cannot fire in 40 steps.
            cfg.terminations.base_contact = None
        env = ManagerBasedRLEnv(cfg=cfg)
        env.reset()

        cmd_term = env.command_manager.get_term("base_velocity")
        dummy = env.scene["dummy"]
        robot = env.scene["robot"]
        dummy_sensor = env.scene["dummy_contact"]
        robot_sensor = env.scene["contact_forces"]
        zero_action = torch.zeros_like(env.action_manager.action)

        # Baseline: production write scheme untouched, dummy still parked at its
        # waiting point. Whatever the robot's sensor reports here is this
        # scene's "sensor working, feet on the ground" reading, and the set of
        # bodies it names is the ground-contact set the snap phase discounts.
        for _ in range(args.baseline_steps):
            env.step(zero_action)
        baseline_robot = nonzero_map(robot_sensor, robot_sensor.data.net_forces_w.norm(dim=-1).squeeze(0))
        baseline_dummy = nonzero_map(dummy_sensor, dummy_sensor.data.net_forces_w.norm(dim=-1).squeeze(0))
        print(f"대조({args.baseline_steps} 스텝, 스냅 전) 로봇 센서: {fmt(baseline_robot)}")
        print(f"대조({args.baseline_steps} 스텝, 스냅 전) 더미 센서: {fmt(baseline_dummy)}")
        ground_bodies = set(baseline_robot)

        def snap_onto_robot():
            """Production's write scheme and write call — only the target moves.

            _advance_dummy writes (position, upright orientation, zero
            velocity) every step; this writes exactly the same three things,
            with the position taken from the robot's live root instead of the
            scripted entry trajectory.
            """
            pos = robot.data.root_pos_w.torch.clone()
            quat = torch.zeros(env.num_envs, 4, device=env.device)
            quat[:, 3] = 1.0  # (x, y, z, w) identity — upright, no yaw
            zero_vel = torch.zeros(env.num_envs, 6, device=env.device)
            dummy.write_root_state_to_sim(torch.cat([pos, quat, zero_vel], dim=-1))

        cmd_term._advance_dummy = snap_onto_robot

        resets = 0
        fired = set()
        worst_dummy = 0.0
        worst_robot_offground = 0.0
        offground_bodies = set()
        for step in range(args.steps):
            # A step that ends in a reset reports its sensors *after* the reset
            # zeroed them, so an unnoticed reset loop manufactures exactly the
            # 0 N reading this script looks for. Counting resets (and naming
            # the term that caused them) is what keeps the result attributable
            # to the write scheme rather than to a reset.
            _, _, terminated, truncated, _ = env.step(zero_action)
            if bool(terminated[0]) or bool(truncated[0]):
                resets += 1
                fired.update(
                    name for name in env.termination_manager.active_terms
                    if bool(env.termination_manager.get_term(name)[0])
                )

            root_sep = float(
                (dummy.data.root_pos_w.torch[0] - robot.data.root_pos_w.torch[0]).norm().item()
            )
            robot_forces = nonzero_map(
                robot_sensor, robot_sensor.data.net_forces_w.norm(dim=-1).squeeze(0)
            )
            dummy_max = float(dummy_sensor.data.net_forces_w.norm(dim=-1).max().item())
            offground = {n: f for n, f in robot_forces.items() if n not in ground_bodies}
            worst_dummy = max(worst_dummy, dummy_max)
            if offground:
                worst_robot_offground = max(worst_robot_offground, max(offground.values()))
                offground_bodies.update(offground)

            if step < 3 or step == args.steps - 1:
                # robot_z distinguishes "held in place" from "flung away by the
                # overlap" and, in phase 1, shows the reset putting it back at
                # its spawn height every single step.
                robot_z = float(robot.data.root_pos_w.torch[0, 2].item())
                print(
                    f"step={step:3d} root_sep={root_sep:.4f}m robot_z={robot_z:.4f}m "
                    f"dummy_max_force={dummy_max:.4f}N 로봇 센서={fmt(robot_forces)}"
                )

        print(
            f"{args.steps} 스텝 요약: 재설정 {resets}회{' (' + ', '.join(sorted(fired)) + ')' if fired else ''}, "
            f"더미 센서 최대 {worst_dummy:.4f} N, "
            f"접지 외 로봇 부위 최대 {worst_robot_offground:.4f} N {sorted(offground_bodies) or ''}"
        )
        env.close()
        return {
            "resets": resets,
            "fired": sorted(fired),
            "dummy_max": worst_dummy,
            "robot_offground_max": worst_robot_offground,
            "offground_bodies": sorted(offground_bodies),
        }

    phase1 = run_phase("1단계 — 환경 그대로 (재설정이 일어나는지 확인)", disable_base_contact=False)
    phase2 = run_phase("2단계 — base_contact 종료항만 끈 환경 (판정용)", disable_base_contact=True)

    print("\n=== 요약 ===")
    print(f"  1단계: 재설정 {phase1['resets']}/{args.steps}회 {phase1['fired']}, "
          f"더미 센서 최대 {phase1['dummy_max']:.4f} N, "
          f"접지 외 로봇 부위 최대 {phase1['robot_offground_max']:.4f} N")
    print(f"  2단계: 재설정 {phase2['resets']}/{args.steps}회 {phase2['fired']}, "
          f"더미 센서 최대 {phase2['dummy_max']:.4f} N, "
          f"접지 외 로봇 부위 최대 {phase2['robot_offground_max']:.4f} N {phase2['offground_bodies']}")

    reported = phase2["dummy_max"] > 1e-6 or phase2["robot_offground_max"] > 1e-6
    if reported:
        print("재설정 없는 관통 상태에서 접촉이 보고됩니다 — 접촉 채널은 죽어 있지 않습니다.")
        print("  (1단계의 '전부 0' 은 매 스텝 재설정이 센서 버퍼를 비운 결과이지,"
              " 쓰기 방식이 접촉 보고를 무효화한 결과가 아닙니다.)")
        code = 0
    else:
        print("[결함] 재설정이 없는 관통 상태인데도 양쪽 센서가 전부 0 입니다 —"
              " 매 스텝 루트 상태 강제 쓰기가 접촉 생성/보고를 무효화합니다.")
        code = 1
except Exception:
    import traceback

    print("check_forced_overlap 실행 실패:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    code = 1
finally:
    # See check_scene.py: close(exit_code=code) is what preserves a nonzero
    # code through this build's fast-shutdown path.
    simulation_app.close(exit_code=code)

raise SystemExit(code)
