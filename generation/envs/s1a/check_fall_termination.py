#!/usr/bin/env python3
"""로봇-인체 접촉이 에피소드를 끝내지 않고, 낙상은 여전히 끝내는지 확인한다.

    ~/isaacsim/python.sh generation/envs/s1a/check_fall_termination.py

상속받은 base_contact 종료항(mdp.illegal_contact, 로봇 토르소가 무엇에든
1 N 넘게 닿으면 즉시 에피소드를 끝낸다 — 로코모션 *학습*용으로는 "토르소가 닿으면
넘어진 것"이라는 타당한 근사지만, S1a 에서는 로봇-인체 접촉 그 자체가 기록해야
할 사건이다)가 걸려 있는 한 접촉 직후 1 스텝 만에 에피소드가 끝난다(실측).
그래서 S1a 는 로봇 자신의 자세(루트 높이·상향 각도)만 보는 종료항
둘을 쓴다(s1a_env_cfg.py 의 fall_height/fall_orientation, 둘 다 isaaclab 상류
함수 — root_height_below_minimum/bad_orientation — 그대로 사용, 임계값은
check_fall_thresholds.py 실측 근거를 코드 주석에 남겼다). 이 스크립트가 그 구성을
두 국면으로 검증한다.

  A - 실제 정책으로 자연스럽게 접촉까지 돌린다, 두 구성을 모두: --contact-seed 는
      강제 없이도 로봇-더미가 실제로 접촉하는 것으로 이미 확인된 시드다
      (check_entry.py --seeds 0,1,2,3,4 --episodes-per-seed 2 실행에서
      seed=2 episode=0 이 min_distance=0.017m 까지 좁혀지며 dummy_contact 힘
      40~260N 대를 여러 스텝 연속으로 냈다). 같은 시드·같은 정책을 두 종료항
      구성으로 각각 돌린다 — 운영 구성(fall_height/fall_orientation)과,
      상속받던 원래 base_contact 를 되살린 음성 대조. 두 실행이
      반대로 갈려야("접촉이 곧바로 안 끝남" vs "접촉 직후 리셋됨") 이 검사가
      공허하지 않다는 증거다(catalogue 의 "성공만 보이는 게이트는 게이트가
      아니다" 지시에 대한 답 — 음성 대조 쪽이 실제로 "실패하는" 시연이다).

      텔레포트로 겹치는 방식은 시도했다가 버렸다 — 더미가 대기 위치(로봇에서
      수 m)에서 로봇 루트로 한 프레임에 순간이동하면 PD 스프링이 아니라 PhysX
      접촉 솔버 자체가 그 인위적인 관통을 되돌리려 하며 수만 N 짜리 충격을
      낸다(실측, 아래 "버린 접근" 참고) — 그건 이 방식의 결함이 아니라 "갑자기
      겹친 상태로 스폰"이라는 시나리오 자체가 비현실적이기 때문이다. 실제
      접근을 그대로 돌리는 것이 유일하게 물리적으로 정직한 강제 접촉이다.

  B - 실제 정책으로 낙상까지 돌린다: 넘어지는 것으로 이미 확인된 시드
      (check_fall_thresholds.py, --fall-seed)를 돌려, 에피소드가 도중에 끝나는지,
      끝났다면 어떤 종료항이 발화했는지, 그 직전 루트 높이·상향각이 어땠는지
      확인한다. --fall-seed 에 낙상이 없는 시드(예: 0)를 주면 이 국면은 정직하게
      실패한다 — 게이트가 공허하지 않다는 두 번째 시연.

두 국면 모두 리셋 횟수와 발화한 종료항 이름을 센다 — 그것을 안 세면
접촉력 0 에서 틀린 결론이 나온다(리셋이 센서 버퍼를 0 으로 비우는데
그걸 모르면 0 을 다른 것으로 오독한다, isaaclab_physx 의 update_net_forces_kernel
의 `if timestamp[env] == 0.0: return` 조기 반환). 음성 대조에서는 그 조기 반환 자체가
관측을 막는다 — base_contact 가 걸린 스텝부터는 매 스텝 리셋되어 그 스텝의 진짜
접촉력을 이 스크립트가 읽을 기회 자체가 없다(종료항 발화라는 사실 하나가 유일한 증거).
이 국면의 판정이 힘 수치가 아니라 발화한 종료항 이름만 요구하는 것은 그래서다.

버린 접근 (기록용): entry["target_xy"] 를 로봇의 현재 루트로 매 스텝 스냅하고 더미를
그 자리로 한 번 순간이동시켜 시작하는 버전을 먼저 시도했다. root_sep 이 몇십 cm인
상태에서도 더미/로봇 센서가 4만~5만 N 대를 냈고, 로봇이 fall_orientation 으로 넘어졌다
— 인위적인 순간 관통이 만든 솔버 충격이지 PD 홀드가 유지하는 접촉의 성질이 아니다.

접촉 귀속과 "서 있는지": 로봇 접지 부위는 10 스텝
스냅샷이 아니라 구조적 기준(.*ankle_link — H1RoughEnvCfg 의 feet_air_time/
feet_slide 보상항이 이미 같은 센서에 쓰는 바로 그 패턴)으로 잡는다 — 이족 로봇은
발을 번갈아 딛으므로 스냅샷은 한쪽 발만 실린 순간을 잡을 수 있다. 판정은
더미 자신의 센서(자기 무게 부위 제외, 자가 충돌이 꺼져 있고 씬에 로봇 말고 다른
게 없어 모호하지 않은 1차 근거)로 하고, 로봇 접지외 읽기는 보조로만 쓴다. 그리고
접촉 중 더미의 17-부위 z 스프레드("서 있음" 불변식)를 재서 로봇이 실제로
미는 상황에서도 안 무너지는지 확인한다 — check_dummy_hold.py 는 로봇에 zero_action
을 줘서 이 경우를 덮지 못한다.

기울기 게이트: `_advance_dummy` 의 각속도 이득은 지면 접촉점 기준(도립진자)
관성·중력 음강성을 반영한다(mdp/commands.py). 그 근거가
이 스크립트의 실측이다 — A 국면 접촉 중 span 이 1358mm(초기 1370mm 대비 거의
안 줄어든 것처럼)로 읽혔지만, 실제로는 더미가 요를 뺀 축으로 약 16° 기운
상태였다(고정조인트라 span 부족은 항상 강체 기울기이지만, 기운 방향에 따라
span 자체는 거의 안 줄 수도 있다). span 만으로는 이 전도를 놓친다 — 요를 뺀
기울기 각도(`tilt_deg`, 로봇 자신의 bad_orientation 이 쓰는 것과 같은 개념,
`projected_gravity_b` 경유)를 별도로 재고 게이트한다.

요 게이트: `tilt_deg` 는 정의상 요 불변(몸-위 vs 세계-위 각도)이라 요만
틀어지는 오작동은 못 잡는다 — 실측에서 tilt 가 16->6deg 로
좋아지는 동안 커맨드-실현 요 오차는 오히려 16.4->48.9deg 로 나빠졌고, 접촉력이
기록된 바로 그 3 스텝에서 요가 +24.0/+26.8/+33.0° 였다(커맨드는 항상 0°). 더미
인체측정치가 좌우·전후 비대칭이라(BR-02 얼굴 오프셋, BR_11/BR_12 좌우 분리
채널) 요는 tilt 와 달리 "어느 부위가 맞았는가" 자체를 바꾼다. 그래서 `yaw_deg()` 를
tilt 와 별도로 재고, 접촉 중 요의 범위뿐 아니라 **접촉력이 기록된 시점별
요**(dummy_contact_yaw, force 와 index-정렬)도 함께 출력·게이트한다. 그 요 오차는
`mdp/commands.py` 가 다룬다(잡음 섞인 PhysX 속도 대신 유한차분,
CoM 오프셋은 바디 프레임+매 스텝 회전, 축별 토크 상한) —
이 스크립트는 그 결과가 실제로 요를 줄였는지 재는 쪽이다.

종료 코드: 0 A 는 운영 구성에서 접촉 발생 후 곧바로 리셋되지 않고 더미 센서(1차
근거) 힘이 물리적 범위(수백 N, 5000N 미만)이며 접촉 중 span 이 무너지지 않고
(1000mm 초과) **기울기·요 모두 20° 미만**이며, 음성 대조에서는 접촉 직후 리셋됨
(base_contact 발화), B 는 낙상 draw 가 fall_height/fall_orientation 로 조기
종료됨. 아무 조건이나 위반하면 1.
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

parser = argparse.ArgumentParser(description="S1a 낙상/접촉 종료 조건 확인")
parser.add_argument("--contact-seed", type=int, default=2, help="A 국면: 자연 접촉이 관측된 환경 시드")
parser.add_argument("--contact-max-steps", type=int, default=450, help="A 국면 최대 스텝")
parser.add_argument("--fall-seed", type=int, default=7, help="B 국면: 낙상이 관측된 환경 시드")
parser.add_argument("--fall-max-steps", type=int, default=300, help="B 국면 최대 스텝")
args = parser.parse_args()

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
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    import isaaclab.envs.mdp as base_mdp
    import isaaclab.utils.math as math_utils
    import isaaclab_tasks  # noqa: F401  — registers the H1 tasks the agent cfg lives in
    import generation.envs.register  # noqa: F401  — registers Keti-S1a-v0

    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_tasks.utils.hydra import resolve_presets

    from generation.envs.s1a.recorder import ground_contact_body_names
    from generation.envs.s1a.s1a_env_cfg import DUMMY_USD, S1aEnvCfg

    def nonzero_map(sensor, forces) -> dict:
        return {n: f for n, f in zip(sensor.body_names, forces.tolist()) if f > 1e-6}

    def fmt(force_map) -> str:
        return ", ".join(f"{n}={f:.2f}N" for n, f in force_map.items()) or "없음 (전부 0.0000N)"

    def load_policy_env(seed: int, negative_control: bool):
        cfg = resolve_presets(S1aEnvCfg(), selected=("physx",))
        cfg.seed = seed
        if negative_control:
            # Reconstruct exactly the term H1RoughEnvCfg sets
            # (rough_env_cfg.py: self.terminations.base_contact.params
            # ["sensor_cfg"].body_names = ".*torso_link", threshold=1.0).
            cfg.terminations.fall_height = None
            cfg.terminations.fall_orientation = None
            cfg.terminations.base_contact = DoneTerm(
                func=base_mdp.illegal_contact,
                params={
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
                    "threshold": 1.0,
                },
            )
        agent_cfg = load_cfg_from_registry("Keti-S1a-v0", "rsl_rl_cfg_entry_point")
        installed_version = metadata.version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Flat-H1-v0"))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        return env, policy

    # Both assets' ground-contact bodies identified structurally, not from a
    # short empirical snapshot: a biped alternates load between feet, so a
    # 10-step window can catch only one foot loaded and misread the other
    # foot's later, perfectly normal stance load as "new" contact. Both are
    # structural facts about the assets instead: the robot's own reward
    # terms (H1RoughEnvCfg's feet_air_time/feet_slide) already use
    # ".*ankle_link" as *the* definition of "this is a foot" for this exact
    # sensor; the dummy's ground-contact prims are read through the asset's
    # own keti:bodyRegion tag (BR-12) via ground_contact_body_names()
    # (recorder.py) rather than hand-typed as BR_12_left/right — CONTRIBUTING.md
    # 「관절 순서」: the prim name is not the canon, the tag on the prim is.

    def robot_ground_bodies(sensor) -> set[str]:
        return {n for n in sensor.body_names if n.endswith("ankle_link")}

    def run_contact_phase(label: str, negative_control: bool) -> dict:
        print(f"\n########## A{'(음성 대조)' if negative_control else ''} - {label} ##########")
        env, policy = load_policy_env(args.contact_seed, negative_control)
        unwrapped = env.unwrapped
        dummy = unwrapped.scene["dummy"]
        dummy_sensor = unwrapped.scene["dummy_contact"]
        robot_sensor = unwrapped.scene["contact_forces"]
        obs = env.get_observations()

        ground_bodies = robot_ground_bodies(robot_sensor)
        dummy_own_weight_bodies = ground_contact_body_names(str(DUMMY_USD), dummy_sensor.body_names)
        print(f"로봇 접지 부위(구조상 고정, .*ankle_link): {sorted(ground_bodies)}")
        print(f"더미 자체 접지 부위(구조상 고정): {sorted(dummy_own_weight_bodies)}")

        def span_mm() -> float:
            z = dummy.data.body_pos_w.torch[0, :, 2]
            return float((z.max() - z.min()).item()) * 1000.0

        def tilt_deg() -> float:
            # Angle between the dummy's own "up" axis and world-up, yaw
            # excluded — span alone can miss a real tilt (a run here
            # once read span=1358mm during contact, "펴짐",
            # while the dummy was actually tipped ~16 deg — a fixed-joint
            # assembly's span shortfall is always a rigid tilt, but not
            # every tilt direction shortens span much). Same construction as
            # check_dummy_hold.py's tilt_deg.
            up_local = torch.tensor([[0.0, 0.0, 1.0]], device=unwrapped.device)
            up_world = math_utils.quat_apply(dummy.data.root_quat_w.torch[0:1], up_local)
            cos_tilt = torch.clamp(up_world[0, 2], -1.0, 1.0)
            return float(torch.acos(cos_tilt).item()) * 180.0 / 3.141592653589793

        def yaw_deg() -> float:
            # Commanded-vs-realized yaw about world-up — separate from
            # tilt_deg() on purpose: tilt is yaw-
            # invariant by construction (an up-vector angle cannot see
            # rotation about up), so it read near 6 deg
            # while the commanded-vs-realized yaw this function measures
            # got *worse* (16.4->48.9 deg) over the same interval — and it
            # matters for exactly this script's judgment: the dummy's
            # anthropometry is not yaw-symmetric (BR-02 face offset, BR_11/
            # BR_12 are separate left/right sensor channels), so which
            # region/side actually reports the contact force below depends
            # on yaw in a way tilt cannot capture. Signed, XYZ-extrinsic
            # convention (isaaclab.utils.math.euler_xyz_from_quat), range
            # (-180, 180] — directly comparable to the 0 deg commanded value.
            _, _, yaw = math_utils.euler_xyz_from_quat(dummy.data.root_quat_w.torch[0:1])
            return float(yaw.item()) * 180.0 / 3.141592653589793

        span0 = span_mm()

        first_contact_step = None
        first_reset_step = None
        fired_at_first_reset: set[str] = set()
        resets = 0
        # Primary evidence: the dummy's own sensor with its own
        # resting-weight bodies excluded is unambiguous (self-collision is
        # off and nothing else is in this scene, so any *other* dummy body
        # loading can only be the robot). The robot's off-foot reading is
        # kept as supporting context, not the figure this script's pass
        # condition is computed from.
        dummy_forces_seen: list[float] = []
        robot_offground_seen: list[float] = []
        span_during_contact: list[float] = []
        tilt_during_contact: list[float] = []
        yaw_during_contact: list[float] = []
        # Yaw specifically at the steps a dummy contact *force* was recorded
        # (index-aligned with dummy_forces_seen, not with the broader
        # contact_now-driven lists above, which also fire on offground-only
        # steps) — this is the correlation that matters:
        # +24.0/+26.8/+33.0 deg was measured at the 3 steps force was
        # recorded, against a 0 deg command, and that correlation (not just
        # "yaw got worse somewhere in the window") is what makes it the
        # attribution-relevant quantity.
        dummy_contact_yaw: list[tuple[int, float, float]] = []
        z_before, angle_before = None, None
        robot = unwrapped.scene["robot"]
        z_before_reset = float(robot.data.root_pos_w.torch[0, 2].item())
        grav0 = robot.data.projected_gravity_b.torch[0]
        angle_before_reset = math.degrees(math.acos(max(-1.0, min(1.0, float(-grav0[2])))))
        for step in range(args.contact_max_steps):
            with torch.inference_mode():
                obs, _, dones, extras = env.step(policy(obs))
            done = bool(dones[0])

            dummy_forces = nonzero_map(dummy_sensor, dummy_sensor.data.net_forces_w.norm(dim=-1).squeeze(0))
            # A body the dummy's own standing weight never loads (e.g. BR_04,
            # back/shoulders) lighting up is unambiguous new contact.
            dummy_new = {n: f for n, f in dummy_forces.items() if n not in dummy_own_weight_bodies}
            dummy_max = max(dummy_new.values(), default=0.0)
            robot_forces = nonzero_map(robot_sensor, robot_sensor.data.net_forces_w.norm(dim=-1).squeeze(0))
            offground = {n: f for n, f in robot_forces.items() if n not in ground_bodies}
            contact_now = bool(dummy_new) or bool(offground)
            sp, te, ye = span_mm(), tilt_deg(), yaw_deg()
            if contact_now:
                if first_contact_step is None:
                    first_contact_step = step
                if dummy_new:
                    dummy_forces_seen.append(dummy_max)
                    dummy_contact_yaw.append((step, dummy_max, ye))
                robot_offground_seen.extend(offground.values())
                # The "still standing" invariant, specifically during
                # the one scenario check_dummy_hold.py does not cover: the
                # robot actually walking into and pushing on the dummy (not
                # zero_action). Span, tilt, and yaw — span
                # alone missed a real ~16 deg tilt once, and tilt
                # alone missed a real yaw-only drift another time
                # (tilt is yaw-invariant by construction).
                span_during_contact.append(sp)
                tilt_during_contact.append(te)
                yaw_during_contact.append(abs(ye))

            if done:
                resets += 1
                fired_now = {
                    name for name in unwrapped.termination_manager.active_terms
                    if bool(unwrapped.termination_manager.get_term(name)[0])
                }
                if first_reset_step is None:
                    first_reset_step = step
                    fired_at_first_reset = fired_now
                    z_before, angle_before = z_before_reset, angle_before_reset

            if contact_now or done or step % 50 == 0:
                print(f"  step={step:3d} 더미(자기무게 제외)={fmt(dummy_new)} 로봇(접지외)={fmt(offground)} "
                      f"span={sp:.2f}mm tilt={te:.2f}deg yaw={ye:.2f}deg{' <- 종료 ' + str(sorted(fired_now)) if done else ''}")

            if not done:
                z_before_reset = float(robot.data.root_pos_w.torch[0, 2].item())
                grav = robot.data.projected_gravity_b.torch[0]
                angle_before_reset = math.degrees(math.acos(max(-1.0, min(1.0, float(-grav[2])))))

            # Once we have both a contact reading and (for the positive case)
            # enough steps past it to judge "did not end immediately", or (for
            # the negative control) the first reset itself, there is nothing
            # more this phase needs — stop early rather than burn steps on a
            # config that repeatedly resets (check_forced_overlap.py saw this:
            # "재설정 39회" once base_contact starts firing every step).
            if first_contact_step is not None and first_reset_step is not None:
                break

        env.close()
        gap = (first_reset_step - first_contact_step) if (first_contact_step is not None and first_reset_step is not None) else None
        print(f"요약: first_contact_step={first_contact_step} first_reset_step={first_reset_step} "
              f"(간격={gap}) 발화={sorted(fired_at_first_reset) or '없음'} "
              f"(종료 한 스텝 전 실측: z={z_before}, angle={angle_before})")
        if dummy_forces_seen:
            print(f"  더미 센서(자기무게 제외, 1차 근거) 힘 범위: {min(dummy_forces_seen):.2f}N ~ {max(dummy_forces_seen):.2f}N")
        if robot_offground_seen:
            print(f"  로봇 센서(접지외, 보조 근거) 힘 범위: {min(robot_offground_seen):.2f}N ~ {max(robot_offground_seen):.2f}N")
        if span_during_contact:
            print(f"  접촉 중 span: {min(span_during_contact):.2f}mm ~ {max(span_during_contact):.2f}mm "
                  f"(초기 {span0:.2f}mm), 기울기: {min(tilt_during_contact):.2f}deg ~ "
                  f"{max(tilt_during_contact):.2f}deg, 요(절대값): {min(yaw_during_contact):.2f}deg ~ "
                  f"{max(yaw_during_contact):.2f}deg — 로봇이 실제로 미는 상황에서도 더미가 서 있는지")
        if dummy_contact_yaw:
            # Yaw at
            # exactly the steps a contact force was recorded, not just its
            # range somewhere in the broader contact window.
            per_step = ", ".join(f"step{s}:force={f:.2f}N,yaw={y:+.2f}deg" for s, f, y in dummy_contact_yaw)
            print(f"  접촉력 기록 시점별 요(커맨드 0deg 대비): {per_step}")
        return {
            "first_contact_step": first_contact_step,
            "first_reset_step": first_reset_step,
            "gap": gap,
            "fired": fired_at_first_reset,
            "dummy_forces": dummy_forces_seen,
            "robot_offground_forces": robot_offground_seen,
            "span_during_contact": span_during_contact,
            "tilt_during_contact": tilt_during_contact,
            "yaw_during_contact": yaw_during_contact,
            "dummy_contact_yaw": dummy_contact_yaw,
            "span0": span0,
        }

    positive = run_contact_phase("실제 정책, 운영 구성(fall_height/fall_orientation)", negative_control=False)
    negative = run_contact_phase("실제 정책, 음성 대조(base_contact 되살림)", negative_control=True)

    # Positive: contact must be observed, must not immediately end the
    # episode (a >=5-step grace window — a *later* termination from an
    # actual balance loss the contact caused is legitimate and expected,
    # see module docstring; an *immediate* one is what this is checking
    # against), the dummy's own sensor reading (the unambiguous evidence)
    # must be physical (hundreds of N), not the ~20 kN artifact
    # measured under the teleport scheme, and the dummy must still
    # be standing (span not collapsed) while the robot is actually in
    # contact with it — the "still standing" invariant, checked here
    # under a real pushing robot rather than a stationary zero_action one.
    # span, tilt, and yaw are all checked, not span alone
    # and not span+tilt alone either: a run here
    # once read span_during_contact as high as 1358mm ("펴짐") while the
    # dummy was actually tipped ~16 deg — a fixed-joint assembly's span
    # shortfall is always a rigid tilt, but which *direction* it tips does
    # not have to shorten span much. tilt is itself yaw-invariant by
    # construction (an up-vector angle cannot see rotation about up), so a
    # separate run measured commanded-vs-realized yaw getting *worse*
    # (16.4->48.9 deg) across an interval where tilt looked fine (16->6 deg)
    # — and because BR-02/BR_11/BR_12 make the dummy's anthropometry yaw-
    # asymmetric, that is not a cosmetic gap: it changes which body region
    # the contact is attributed to. 20 deg matches check_dummy_hold.py's
    # moving-phase tilt/yaw tolerance — real contact is at least as
    # disruptive as scripted tracking alone, not a tighter bar.
    a_pos_ok = (
        positive["first_contact_step"] is not None
        and (positive["first_reset_step"] is None or positive["gap"] >= 5)
        and bool(positive["dummy_forces"])
        and max(positive["dummy_forces"]) < 5000.0
        and bool(positive["span_during_contact"])
        and min(positive["span_during_contact"]) > 1000.0
        and bool(positive["tilt_during_contact"])
        and max(positive["tilt_during_contact"]) < 20.0
        and bool(positive["yaw_during_contact"])
        and max(positive["yaw_during_contact"]) < 20.0
    )
    # Negative control: with the old termination back, contact must end the
    # episode, named base_contact as the cause. Note what this does *not*
    # require: a directly-observed nonzero force reading at the moment of
    # contact. That is not a relaxation — it is structurally impossible to
    # observe here, and that impossibility is itself part of what this
    # control demonstrates. base_contact reset every single step once it
    # started firing (this run: steps 393-449 straight), and a reset zeroes
    # the sensor buffer before this script's next read (isaaclab_physx's
    # update_net_forces_kernel, `if timestamp[env] == 0.0: return`) — so the
    # contact that caused reset 393 is only visible through the termination
    # flag itself, never through this script's own sensor read. Requiring a
    # force reading here would make the control unpassable by construction,
    # not a meaningful check.
    a_neg_ok = negative["first_reset_step"] is not None and "base_contact" in negative["fired"]
    print(f"\nA(운영) 판정: {'접촉이 에피소드를 곧바로 끝내지 않고 힘이 물리적이며 더미가 서 있음(정상)' if a_pos_ok else '[결함]'} "
          f"(contact={positive['first_contact_step']}, reset={positive['first_reset_step']}, "
          f"gap={positive['gap']}, "
          f"max_dummy_force={max(positive['dummy_forces']) if positive['dummy_forces'] else None}, "
          f"min_span={min(positive['span_during_contact']) if positive['span_during_contact'] else None}, "
          f"max_tilt={max(positive['tilt_during_contact']) if positive['tilt_during_contact'] else None}, "
          f"max_yaw={max(positive['yaw_during_contact']) if positive['yaw_during_contact'] else None})")
    print(f"A(음성 대조) 판정: {'예상대로 접촉 직후 리셋됨(정상)' if a_neg_ok else '[결함] 음성 대조가 실패로 재현되지 않음'} "
          f"(contact={negative['first_contact_step']}, reset={negative['first_reset_step']}, "
          f"gap={negative['gap']}, fired={sorted(negative['fired'])})")

    # ------------------------------------------------------------------
    # B - a real fall must still end the episode
    # ------------------------------------------------------------------
    print("\n########## B - 실제 정책으로 낙상까지 돌리기 ##########")
    env_b, policy_b = load_policy_env(args.fall_seed, negative_control=False)
    robot_b = env_b.unwrapped.scene["robot"]
    obs = env_b.get_observations()

    def read_pose(robot_asset):
        z = float(robot_asset.data.root_pos_w.torch[0, 2].item())
        grav = robot_asset.data.projected_gravity_b.torch[0]
        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(-grav[2])))))
        return z, angle_deg

    fired_b: set[str] = set()
    ended_step = None
    z_at_end = None
    angle_at_end = None
    # last_z/last_angle: the reading from *before* the step call that ends
    # the episode, not after — ManagerBasedRLEnv._reset_idx() runs inside
    # the same step() call that reports done=True, before step() returns
    # (confirmed in run_s1a.py's own comment on this exact hazard), so a
    # read taken right after a terminal step() already belongs to the next
    # episode's fresh spawn. This is a diagnostic readout, not the pass
    # condition (which trusts termination_manager's own exact evaluation,
    # not this one-step-stale reconstruction of it).
    last_z, last_angle = read_pose(robot_b)
    for step in range(args.fall_max_steps):
        with torch.inference_mode():
            obs, _, dones, extras = env_b.step(policy_b(obs))
        if bool(dones[0]):
            fired_b.update(
                name for name in env_b.unwrapped.termination_manager.active_terms
                if bool(env_b.unwrapped.termination_manager.get_term(name)[0])
            )
            ended_step = step
            z_at_end = last_z
            angle_at_end = last_angle
            break
        last_z, last_angle = read_pose(robot_b)
        if step % 25 == 0:
            print(f"  step={step:3d} z={last_z:.3f}m angle={last_angle:.2f}deg")

    print(f"B 요약: seed={args.fall_seed} 종료 스텝={ended_step} 발화 종료항={sorted(fired_b) or '없음'} "
          f"(종료 한 스텝 전 실측: z={z_at_end}, angle={angle_at_end})")
    env_b.close()

    b_ok = (
        ended_step is not None
        and ended_step < args.fall_max_steps - 1
        and bool(fired_b)
        and fired_b.issubset({"fall_height", "fall_orientation"})
    )
    print(f"B 판정: {'낙상이 fall_height/fall_orientation 로 조기 종료됨(정상)' if b_ok else '[결함] 낙상이 감지되지 않았거나 다른 종료항이 발화함'}")

    code = 0 if (a_pos_ok and a_neg_ok and b_ok) else 1
    print(f"\n=== 최종 판정: A(운영)={a_pos_ok} A(음성대조)={a_neg_ok} B={b_ok} -> {'통과' if code == 0 else '실패'} ===")
except Exception:
    import traceback

    print("check_fall_termination 실행 실패:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    code = 1
finally:
    # See check_scene.py: close(exit_code=code) is what preserves a nonzero
    # code through this build's fast-shutdown path.
    simulation_app.close(exit_code=code)

raise SystemExit(code)
