#!/usr/bin/env python3
"""PD 홀드가 더미를 세워 두고 대본대로 따라가게 하는지 확인한다.

    ~/isaacsim/python.sh generation/envs/s1a/check_dummy_hold.py

더미의 루트는 매 스텝 `write_root_state_to_sim` 으로 강제로 다시 쓰지 않고,
루트에 거는 유한한 스프링-댐퍼 힘/토크(`S1aStraightWalkCommand.
_advance_dummy`)로 붙잡는다. 그 대가로 더미가 "순간이동"하지 않고 실제
동역학을 따르므로, 대본 위치를 놓치지 않고 계속 따라가는지, 자기 무게만으로
넘어지지 않는지를 이 스크립트가 직접 잰다.

힘을 거는 방식: `instantaneous_wrench_composer` 는 쓰지 않는다. 이
컴포저는 `write_data_to_sim()` 호출마다 무조건 비워지고 그 함수는(PhysX 의
`handles_decimation()` 이 False 라서) env 스텝당 물리 서브스텝 수만큼(이 env는
4회) 불린다 — `_advance_dummy` 자체는 env 스텝당 1회만 도니, 힘이 서브스텝
4개 중 1개에만 걸린다(25% 듀티). 그리고 중력 전방보상이 있으면 "정적 접촉이
보고됨" 이라는 이 스크립트의 결론 자체가 성립할 수 없다 — 전방보상이 제대로
작동하면 목표에 선 더미는 지면에 ~0 N 을 가하기 때문이다. 그래서
`permanent_wrench_composer`(서브스텝 무관하게 유지됨) + 수직 전방보상 없음(중력·
지면 접촉이 정상적인 물체처럼 체중을 그대로 받게 둠) + 출력 포화를 쓴다.

각속도 이득: 축별 관성을 "자유물체가 자기 무게중심을 중심으로 회전한다"고
가정하면 틀린다 — 더미는 발이 지면에 붙어 있는 동안 무게중심이
아니라 **지면 접촉점을 중심으로 기운다**(도립진자). 기울어지는 축(x=roll,
y=pitch)은 평행축 정리로 옮긴 유효 관성(I_eff=I_com+m·h², h=무게중심 높이)을
쓰고, 중력이 그 축에 더하는 음강성(m·g·h)도 보상한다 — yaw(z)는 그대로
자유물체 근사가 맞다(수직축 이동은 그 축 자체의 관성을 안 바꾼다). 근거 전문은
`mdp/commands.py` 의 `_advance_dummy` docstring 에 있다.

요 게이트: `tilt_deg()`(요를 뺀, 몸-위 방향과 세계-위 방향 사이 각도)는
정의상 요에 불변이다 — 요만 틀어지는 오작동은 tilt 로는 절대 안 잡힌다.
실측에서 tilt 가 16->6deg 로 좋아지는 동안 커맨드-실현 요 오차는 오히려
16.4->48.9deg 로 나빠졌다
(더미 인체측정치가 좌우·전후 비대칭이라 BR_11/BR_12 좌우 채널 중 어디에 접촉이
잡히는지가 요에 따라 달라지므로, tilt 와 달리 요는 판정 결과 자체를 바꾼다).
그래서 `yaw_deg()` 를 tilt 와 별도로 재고 게이트한다. 그 요 오차의 원인은
(a) PhysX 가 보고하는
루트 속도가 실측보다 2-4배 부풀려져(TGS 솔버 자체가 로그로 경고하는 잡음) 감쇠
항이 과도해진 것, (b) `_dummy_com_offset` 을 세계좌표로 캐싱해 요가 0 이 아닌
동안 힘 작용점이 실제 무게중심을 벗어나 기생 토크를 만든 것이었다 — `mdp/
commands.py` 는 그래서 유한차분 속도와 바디 프레임 오프셋(매 스텝 회전)을 쓴다.
토크 상한도 축별로 유도한다(공유 노름
클램프가 더 무른 yaw 성분을 짓누르지 않도록).

두 국면을 본다:
  1단계 — 대기(트리거 전): 로봇 행동을 0 으로 고정해 진입이 걸리지 않게 하고,
          목표가 고정된 채로 위치·자세 오차가 수렴해 유지되는지, 지면 반력이
          더미 자신의 실측 체중(m·g) 근처에 정착하는지, **기울기 각도**(요를
          뺀, 몸의 위 방향과 세계 위 방향 사이 각도 — span 이 좋아 보여도 가릴
          수 있는 진짜 전도 신호)가 작은지, 그리고
          **요 오차**(기울기와 별도로 재는 커맨드-실현 요 — 기울기는 정의상
          요 불변이라 요만 틀어지는 오작동은 못 잡는다)가 작은지 잰다.
  2단계 — 진입(강제 트리거): `entry["triggered"]` 를 강제로 켜서 목표가
          매 스텝 이동하기 시작하게 하고, 실제 더미가 그 이동을 얼마나
          뒤처지지 않고 따라가는지(추종오차 자체를 게이트로), 그리고 "서 있음"
          불변식(17개 부위의 z 스프레드가 일정 = 강체로 붙어서 서
          있음)과 기울기·요 오차가 깨지지 않는지 잰다.

두 국면 모두 리셋이 일어났는지 세고 어떤 종료항이 발화했는지 남긴다 — 로봇
행동이 0(제자리 서기)이라 리셋이 없어야 정상이며, 리셋이 있었다면 그 이후의
위치·자세·접지력 비교는 다른 에피소드의 값과 섞인 것이라 무의미해진다. 정상상태
누산기는 리셋마다 비운다 — 안 비우면 "마지막 리셋 이후 N 스텝" 이라는 라벨이
실제로는 여러 에피소드에 걸친 값을 담게 된다(방향은 보수적이지만 라벨이 틀림).

종료 코드: 0 두 국면 모두 리셋 없이 오차·기울기가 허용 범위 안에서 수렴하고,
지면 반력이 실측 체중 근처이며, z 스프레드가 유지되고, 2단계 추종오차도 허용
범위 안임 / 1 위 중 하나라도 위반 또는 실행 실패
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

parser = argparse.ArgumentParser(description="더미 PD 홀드 수렴·추종 확인")
parser.add_argument("--wait-steps", type=int, default=150, help="1단계(대기) 스텝 수")
parser.add_argument("--move-steps", type=int, default=150, help="2단계(강제 진입) 스텝 수")
parser.add_argument("--seed", type=int, default=0, help="환경 시드")
# What this script measures is the hold servo itself, so by default it keeps
# the servo on for the whole run. In production the servo is switched off for
# good once the robot is within dummy_release_gap_m of the dummy's surface
# (S1aStraightWalkCommand._update_hold_release) -- from that step on the
# dummy is a free body and there is no hold left to measure. Pass the
# production gap here to see what production actually does instead; the
# thresholds below are unchanged either way, and stage 2 prints the step the
# release fired on so a short held phase cannot be mistaken for a good one.
parser.add_argument("--release-gap", type=float, default=-1.0,
                    help="홀드 해제 간격(m). 음수면 해제하지 않는다(기본 — 홀드 자체를 잰다). "
                         "운영 기본값은 S1aStraightWalkCommandCfg.dummy_release_gap_m")
# Settling tolerance: loose enough that the fixed-joint children's own small
# inertial reaction (a similar few-percent gap was measured
# between commanded and realized speed) doesn't false-positive this, tight
# enough that a genuinely diverging or sagging hold still trips it. No Z
# term in the controller — this is a 3-D norm, so it
# also catches height drift even though nothing is actively driving height.
parser.add_argument("--pos-tol-m", type=float, default=0.05, help="정상 상태 위치 오차 허용치")
parser.add_argument("--ang-tol-deg", type=float, default=5.0, help="정상 상태 자세 오차 허용치")
parser.add_argument("--span-tol-mm", type=float, default=50.0, help="1단계(정지) 17개 부위 z 스프레드 허용 변화폭")
# span 만으로는 전도를 놓칠 수 있다(실측: 접촉 중 span 이 1358mm 로
# "펴져" 보였지만 실제로는 더미가 약 16° 기운 상태였다 — 고정조인트라 휨은 없고
# span 부족은 항상 강체 기울기이지만, 기운 방향에 따라 span 자체는 거의 안 줄 수도
# 있다). 요를 뺀 몸-축 대 세계-위 각도(전도의 직접 지표)를 span 과 별도로 게이트한다.
parser.add_argument("--tilt-tol-deg", type=float, default=5.0, help="1단계(정지) 기울기 각도 허용치")
parser.add_argument("--tilt-tol-moving-deg", type=float, default=20.0, help="2단계(추종) 기울기 각도 허용치")
# tilt_deg() 는 정의상 요 불변(몸-위 vs 세계-위 각도) — 요만 틀어지는 오작동은
# 별도로 재지 않으면 안 잡힌다(실측: tilt 가
# 16->6deg 로 좋아지는 동안 커맨드-실현 요 오차는 16.4->48.9deg 로 더 나빠졌다).
# 같은 축별 PD 설계가 대상이므로 tilt 와 같은 허용치를 재사용한다 — 별도로
# 고안한 숫자가 아니라 이미 근거를 댄 "정상상태/추종 중 각오차 어디까지
# 허용하는가" 를 요 축에도 그대로 적용하는 것.
parser.add_argument("--yaw-tol-deg", type=float, default=5.0, help="1단계(정지) 커맨드-실현 요 오차 허용치")
parser.add_argument("--yaw-tol-moving-deg", type=float, default=20.0, help="2단계(추종) 커맨드-실현 요 오차 허용치")
# 2단계는 목표가 계속 움직이는 동안 재는 것이라 1단계보다 느슨하다 —
# 실측(움직이는 목표를 쫓는 동안 최대 ~100mm, 항상 1370mm 부근으로 회복)이 근거다.
# 실제 전도(정상 보행 낙상 시 span 이 수백 mm대로 붕괴)와는 자릿수가 다르므로, 이
# 여유를 둬도 진짜 전도를 놓치지 않는다.
parser.add_argument("--span-tol-moving-mm", type=float, default=150.0, help="2단계(추종) span 허용 변화폭")
# 이론적 정상상태 추종오차(임계감쇠 2차계, 등속 목표): 2*zeta*v/omega, 최고
# 진입속도(카탈로그 §4.5 상한 1.6 m/s)·이 커맨드의 기본 이득(2Hz, zeta=1.0)에서
# 약 0.255m. 여유를 둔 상한 — 출력만 하고 게이트하지 않으면
# 추종오차가 커져도 종료 코드에 드러나지 않는다.
parser.add_argument("--pos-tol-moving-m", type=float, default=0.4, help="2단계(추종) 위치 추종오차 허용치")
# Not "> 0": a nonzero-but-wrong reading, e.g. a
# 25%-duty-cycle kick averaging out to something that happened to look like
# a plausible number, is exactly the failure mode this gate is for.
# Judged against the dummy's own measured weight, with a tolerance wide
# enough for ordinary contact-solver noise but tight enough to catch the
# hold doing anything but leaving the ground to carry it.
parser.add_argument("--weight-tol-frac", type=float, default=0.3, help="접지 반력이 m*g 대비 벗어나도 되는 비율")
# The moving phase's zero-ground-reaction fraction and mean hold force have no
# threshold arguments here. Both numbers are computed and printed every run,
# but they are not exit-code thresholds — search "No gate on the
# zero-ground-reaction" below for why.
parser.add_argument("--print-every", type=int, default=25, help="스텝 로그 출력 간격")
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

    from generation.envs.s1a.recorder import ground_contact_body_names
    from generation.envs.s1a.s1a_env_cfg import DUMMY_USD, S1aEnvCfg

    cfg = resolve_presets(S1aEnvCfg(), selected=("physx",))
    cfg.seed = args.seed
    # A negative gap can never be reached (the gap is a distance), so this is
    # how the servo is kept on for the whole run — see --release-gap.
    cfg.commands.base_velocity.dummy_release_gap_m = args.release_gap
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()

    cmd_term = env.command_manager.get_term("base_velocity")
    dummy = env.scene["dummy"]

    # _advance_dummy runs from _update_command, which the command manager
    # calls during env.step() — not during env.reset() (reset only resamples
    # the command). One zero-action step is what actually populates the
    # lazy caches (root body id, mass, height span) for the first time.
    zero_action = torch.zeros_like(env.action_manager.action)
    env.step(zero_action)
    if cmd_term._dummy_root_body_id is None:
        raise RuntimeError("root body id 가 첫 스텝 이후에도 캐시되지 않았습니다 — _advance_dummy 호출 경로가 바뀌었을 수 있습니다")
    print(
        f"root_body_id={cmd_term._dummy_root_body_id} "
        f"({dummy.body_names[cmd_term._dummy_root_body_id]}) "
        f"total_mass={float(cmd_term._dummy_total_mass[0]):.3f}kg "
        f"height_span={float(cmd_term._dummy_height_span[0]) * 1000.0:.2f}mm "
        f"com_offset_body={[round(float(v), 5) for v in cmd_term._dummy_com_offset_body[0]]}"
    )
    import isaaclab.utils.math as math_utils

    # Live per-axis gains actually in effect — not a re-derivation. Printing
    # a free-body scalar formula here would silently drift from
    # whatever mdp/commands.py actually computes (a re-derived line
    # was once off by ~19x on yaw). Reading the cached
    # tensors is what keeps this honest whichever formula is currently live.
    omega = 2.0 * 3.141592653589793 * cmd_term.cfg.dummy_hold_natural_freq_hz
    zeta = cmd_term.cfg.dummy_hold_damping_ratio
    lin_k = float(cmd_term._dummy_total_mass[0]) * omega * omega
    ang_k = cmd_term._dummy_moment_of_inertia[0] * omega * omega + cmd_term._dummy_gravity_compensation[0]
    print(f"natural_freq_hz={cmd_term.cfg.dummy_hold_natural_freq_hz} damping_ratio={zeta} "
          f"-> lin_stiffness={lin_k:.1f}N/m "
          f"ang_stiffness(x,y,z)={[round(float(v), 1) for v in ang_k]}N*m/rad "
          f"(moment_of_inertia={[round(float(v), 3) for v in cmd_term._dummy_moment_of_inertia[0]]}kg*m^2, "
          f"gravity_term={[round(float(v), 1) for v in cmd_term._dummy_gravity_compensation[0]]}N*m/rad)")

    def span_mm() -> float:
        z = dummy.data.body_pos_w.torch[0, :, 2]
        return float((z.max() - z.min()).item()) * 1000.0

    def pos_err_m() -> float:
        entry = env.s1a_entry
        target = torch.cat([entry["target_xy"][0], entry["stand_z"][0:1]])
        return float((target - dummy.data.root_pos_w.torch[0]).norm().item())

    def ang_err_deg() -> float:
        target_quat = torch.zeros(1, 4, device=env.device)
        target_quat[:, 3] = 1.0
        err = math_utils.quat_box_minus(target_quat, dummy.data.root_quat_w.torch[0:1])
        return float(err.norm().item()) * 180.0 / 3.141592653589793

    def tilt_deg() -> float:
        # Angle between the dummy's own "up" axis and world-up — yaw
        # (spinning about vertical) does not tip the body over, so this
        # isolates exactly the roll/pitch the hold's ground-pivot model
        # targets, the same concept the robot's own bad_orientation
        # termination uses via projected_gravity_b. Direct evidence
        # of tipping: span can look fine while this is not (measured —
        # span read 1358mm, up ~16 deg, because the dummy tipped in a
        # direction that does not much shorten the tallest-to-lowest gap).
        up_local = torch.tensor([[0.0, 0.0, 1.0]], device=env.device)
        up_world = math_utils.quat_apply(dummy.data.root_quat_w.torch[0:1], up_local)
        cos_tilt = torch.clamp(up_world[0, 2], -1.0, 1.0)
        return float(torch.acos(cos_tilt).item()) * 180.0 / 3.141592653589793

    def yaw_deg() -> float:
        # Commanded-vs-realized yaw about world-up — commanded is always 0
        # (_advance_dummy's target_quat is identity, no yaw ever scripted
        # in). Deliberately separate from tilt_deg(): tilt is yaw-invariant
        # by construction (an up-vector angle cannot see rotation about
        # up), so a hold that is drifting in yaw alone would read as *zero*
        # tilt throughout — exactly what tilt_deg
        # alone misses. euler_xyz_from_quat's yaw component (XYZ
        # extrinsic convention, range (-pi, pi]) is a signed error directly
        # comparable to the 0 deg target, not a [0, 360) wrap that would
        # make a small negative drift read as ~360deg.
        _, _, yaw = math_utils.euler_xyz_from_quat(dummy.data.root_quat_w.torch[0:1])
        return float(yaw.item()) * 180.0 / 3.141592653589793

    span0 = span_mm()
    print(f"초기 z 스프레드={span0:.2f}mm 초기 기울기={tilt_deg():.3f}deg 초기 요={yaw_deg():.3f}deg")

    dummy_sensor = env.scene["dummy_contact"]
    expected_weight_n = float(cmd_term._dummy_total_mass[0]) * 9.81

    # The same structural ground-contact set check_scene.py/
    # check_fall_termination.py use, read through the asset's own
    # keti:bodyRegion tag (BR-12) rather than hand-typed as
    # BR_12_left/right — CONTRIBUTING.md 「관절 순서」: the prim name is not the canon.
    _DUMMY_GROUND_NAMES = ground_contact_body_names(str(DUMMY_USD), dummy_sensor.body_names)
    _dummy_ground_idx = [dummy_sensor.body_names.index(n) for n in _DUMMY_GROUND_NAMES]

    def dummy_ground_force_n() -> float:
        # Static ground contact is exactly the case that is blind under
        # a teleport write scheme, and a PD hold applied through the
        # instantaneous composer is *also* blind to it in a subtler way (25%-duty-cycle
        # force, read as if continuous) — mdp/commands.py's
        # _advance_dummy docstring has the full account. If the hold's own
        # weight settles the dummy's feet onto the ground with a reading
        # near m*g here, that is direct evidence the same mechanism will
        # report static robot-dummy contact too, with a physically sane
        # magnitude rather than a lucky-looking one.
        #
        # Sum of the two feet, not max over all 17 bodies:
        # weight legitimately shifts between the two feet over
        # time (measured per-region range 431-1025N while the *sum* stayed
        # 853.94-865.16N, <1% off m*g) — taking the max of a single region
        # reads as low as ~half of m*g whenever the load happens to be
        # sitting close to evenly split, which is a real state this
        # controller produces, not a fault, and would fail the m*g gate
        # below on nothing more than that.
        forces = dummy_sensor.data.net_forces_w.norm(dim=-1)[0]
        return float(forces[_dummy_ground_idx].sum().item())

    # foot_xy() is there for the differential-foot-friction hypothesis
    # for the residual yaw (a low-friction material on BR_12_left/right
    # was tried and not kept — see s1a_env_cfg.py).
    #
    # The raw foot displacement this function feeds is *not* a slip
    # measurement — it is dominated by the commanded target itself moving
    # (28 steps * 0.02 s * 0.7 m/s =~ 0.40 m accounts for the bulk of what
    # reads as "foot slide"). This dummy has no ankle DOF or anything
    # else that could let a foot move relative to the rest of the rigid,
    # fixed-jointed body — the 17 regions move as one, so "foot position"
    # and "whole-body position" are the same signal, and a body that is
    # *supposed* to be tracking a moving target is supposed to have this
    # number be large. What can actually distinguish "successful tracking
    # of a moving target" from "something dragging the feet beyond that"
    # is comparing this against how far the *target* moved over the same
    # window — see foot_vs_target_residual_mm below. Kept per-foot (not summed/averaged)
    # since an asymmetric drag would show up as a difference between them.
    _dummy_body_ground_idx = [dummy.body_names.index(n) for n in _DUMMY_GROUND_NAMES]

    def foot_xy() -> torch.Tensor:
        return dummy.data.body_pos_w.torch[0, _dummy_body_ground_idx, :2].clone()

    def target_xy() -> torch.Tensor:
        return env.s1a_entry["target_xy"][0].clone()

    def step_and_check_reset(action) -> tuple[bool, set[str]]:
        _, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated[0]) or bool(truncated[0])
        fired: set[str] = set()
        if done:
            fired = {
                name for name in env.termination_manager.active_terms
                if bool(env.termination_manager.get_term(name)[0])
            }
        return done, fired

    # The robot falling under a raw zero_action (unrelated to the dummy or
    # its PD hold — the trained policy actively balances, a held default
    # pose alone eventually doesn't over long enough windows) resets
    # reset_dummy_entry too, handing the dummy a brand new draw. A fixed
    # "last third of the configured window" steady-state region can straddle
    # that boundary and average a stale episode's tail with a fresh one's
    # transient. Anchor the window to the *last* reset instead — steady
    # state is judged only over settled steps since whichever episode is
    # still running at the end, never across a reset.
    # 10 steps is generous, not tight: printed traces consistently show
    # pos_err at its steady 0.0443 m value within 1-2 steps of a reset (the
    # design's settling time is ~0.3 s = 15 steps for a *first* transient
    # from a cold start, but this is a repeat of the same equilibrium the
    # controller was already at before the interrupting reset, not a fresh
    # transient). The
    # robot's own zero_action instability resets often enough (every ~40-70
    # steps, several times inside even a few hundred steps) that a large
    # margin starves this of a valid window more often than it protects
    # against anything real.
    SETTLE_MARGIN_STEPS = 10
    MIN_STEADY_STEPS = 10
    print(f"\n--- 1단계: 대기(트리거 전) --- (예상 접지 반력 m*g={expected_weight_n:.2f}N)")
    worst_pos_settled = 0.0
    worst_ang_settled = 0.0
    worst_tilt_settled = 0.0
    worst_yaw_settled = 0.0
    worst_span_dev1 = 0.0
    ground_force_settled = 0.0
    resets1 = 0
    fired1: set[str] = set()
    last_reset_step1 = -1
    steady_span1 = 0
    for step in range(args.wait_steps):
        done, fired = step_and_check_reset(zero_action)
        if done:
            resets1 += 1
            fired1 |= fired
            last_reset_step1 = step
            # Clear, don't just shift the window forward: without this the
            # running max keeps whatever peaked in an *earlier* reset-
            # delimited segment even after "마지막 리셋 이후 N스텝" no longer
            # describes what is actually in the accumulator — direction is
            # conservative (can only make a pass stricter, never a fail
            # softer) but the label would be wrong.
            worst_pos_settled = 0.0
            worst_ang_settled = 0.0
            worst_tilt_settled = 0.0
            worst_yaw_settled = 0.0
            ground_force_settled = 0.0
        pe, ae, te, ye, sp, gf = pos_err_m(), ang_err_deg(), tilt_deg(), abs(yaw_deg()), span_mm(), dummy_ground_force_n()
        worst_span_dev1 = max(worst_span_dev1, abs(sp - span0))
        steady_start = last_reset_step1 + 1 + SETTLE_MARGIN_STEPS
        steady_span1 = max(0, step - steady_start + 1)
        if step >= steady_start:
            worst_pos_settled = max(worst_pos_settled, pe)
            worst_ang_settled = max(worst_ang_settled, ae)
            worst_tilt_settled = max(worst_tilt_settled, te)
            worst_yaw_settled = max(worst_yaw_settled, ye)
            ground_force_settled = max(ground_force_settled, gf)
        if step % args.print_every == 0 or step == args.wait_steps - 1 or done:
            print(f"  step={step:4d} pos_err={pe:.4f}m ang_err={ae:.3f}deg tilt={te:.3f}deg yaw={ye:.3f}deg span={sp:.2f}mm "
                  f"dummy_ground={gf:.4f}N{' <- 리셋 ' + str(sorted(fired)) if done else ''}")
    print(f"1단계 정상상태(마지막 리셋 이후 {steady_span1}스텝) 최악 pos_err={worst_pos_settled:.4f}m "
          f"ang_err={worst_ang_settled:.3f}deg tilt={worst_tilt_settled:.3f}deg yaw={worst_yaw_settled:.3f}deg, "
          f"접지 반력 최대={ground_force_settled:.4f}N (m*g={expected_weight_n:.2f}N), "
          f"전체 구간 span 변화폭={worst_span_dev1:.2f}mm, 리셋 {resets1}회 {sorted(fired1) or ''}")

    print("\n--- 2단계: 강제 진입(목표가 매 스텝 이동) ---")
    env.s1a_entry["triggered"][:] = True
    worst_pos_moving = 0.0
    worst_tilt_moving = 0.0
    worst_yaw_moving = 0.0
    worst_span_dev2 = 0.0
    span_before_move = span_mm()
    resets2 = 0
    fired2: set[str] = set()
    # Saturation rate of the per-axis torque cap: counted per axis, over every step of
    # this phase — not just "moving" steps, since a reset's fresh
    # (untriggered) step or two still calls _advance_dummy and should still
    # count toward the denominator.
    torque_cap_hits = torch.zeros(3, dtype=torch.long)
    torque_cap_steps = 0
    # Reset-aware, like span_before_move above — a reset's teleport is not
    # travel, so both reference points move with every reset.
    foot_xy_ref = foot_xy()
    target_xy_ref = target_xy()
    worst_foot_travel_mm = 0.0
    worst_foot_vs_target_residual_mm = 0.0
    # Moving-phase ground-reaction / hold-force visibility: the m*g gate
    # covers only the *static* phase (ground_force_settled above) — it does
    # not watch whether the dummy stays grounded once its own target starts
    # moving, which is exactly where it was found bouncing (0 N ground
    # reaction on a large fraction of steps, hold force averaging kN and
    # peaking >15kN). dummy_ground_force_n() is reused here, not redefined,
    # so both phases
    # read the same two-feet-sum quantity. 1e-6 N matches this codebase's
    # existing "is this reading actually zero" threshold (nonzero_map's
    # f > 1e-6 in check_fall_termination.py / check_contact_force_sweep.py).
    ground_force_zero_steps = 0
    ground_force_moving_max = 0.0
    lin_force_sum = 0.0
    lin_force_moving_max = 0.0
    for step in range(args.move_steps):
        done, fired = step_and_check_reset(zero_action)
        if done:
            resets2 += 1
            fired2 |= fired
            # A robot-caused reset (unrelated to the dummy — see 1단계's
            # comment) hands reset_dummy_entry's fresh draw triggered=False,
            # which would silently turn this phase back into "waiting" for
            # its remaining steps. Re-force it so a mid-phase reset shortens
            # nothing about what this phase is actually testing.
            env.s1a_entry["triggered"][:] = True
            foot_xy_ref = foot_xy()
            target_xy_ref = target_xy()
        pe, te, ye, sp = pos_err_m(), tilt_deg(), abs(yaw_deg()), span_mm()
        worst_pos_moving = max(worst_pos_moving, pe)
        worst_tilt_moving = max(worst_tilt_moving, te)
        worst_yaw_moving = max(worst_yaw_moving, ye)
        worst_span_dev2 = max(worst_span_dev2, abs(sp - span_before_move))
        # Per-foot displacement from the reset-relative reference, in mm —
        # this is *travel*, not slip (see foot_xy()'s docstring above): a
        # body tracking a moving target is supposed to have this be large.
        # foot_vs_target_residual_mm is the more informative number — how
        # far the foot moved *beyond* how far the target itself moved over
        # the same window, which is ~0 for lag-free tracking regardless of
        # how far the target travelled.
        cur_foot_xy = foot_xy()
        cur_target_xy = target_xy()
        foot_travel_mm = float((cur_foot_xy - foot_xy_ref).norm(dim=-1).max().item()) * 1000.0
        target_travel_mm = float((cur_target_xy - target_xy_ref).norm().item()) * 1000.0
        foot_vs_target_residual_mm = abs(foot_travel_mm - target_travel_mm)
        worst_foot_travel_mm = max(worst_foot_travel_mm, foot_travel_mm)
        worst_foot_vs_target_residual_mm = max(worst_foot_vs_target_residual_mm, foot_vs_target_residual_mm)
        torque_cap_steps += 1
        torque_cap_hits += cmd_term.dummy_torque_saturated_per_axis[0].long().cpu()
        gf = dummy_ground_force_n()
        if gf < 1e-6:
            ground_force_zero_steps += 1
        ground_force_moving_max = max(ground_force_moving_max, gf)
        lf = float(cmd_term.dummy_lin_force_applied[0].norm().item())
        lin_force_sum += lf
        lin_force_moving_max = max(lin_force_moving_max, lf)
        if step % args.print_every == 0 or step == args.move_steps - 1 or done:
            print(f"  step={step:4d} pos_err(추종 지연)={pe:.4f}m tilt={te:.3f}deg yaw={ye:.3f}deg span={sp:.2f}mm "
                  f"foot_travel={foot_travel_mm:.2f}mm 잔차={foot_vs_target_residual_mm:.2f}mm "
                  f"지면반력={gf:.2f}N 홀드력={lf:.2f}N"
                  f"{' <- 리셋 ' + str(sorted(fired)) if done else ''}")
    x_hits, y_hits, z_hits = (int(v) for v in torque_cap_hits)
    print(
        f"2단계 토크 상한 포화율(축별, {torque_cap_steps}스텝 중): "
        f"x={x_hits}/{torque_cap_steps} y={y_hits}/{torque_cap_steps} z(yaw)={z_hits}/{torque_cap_steps}"
    )
    lin_force_moving_mean = lin_force_sum / torque_cap_steps if torque_cap_steps else 0.0
    print(
        f"2단계 지면 반력=0 인 스텝 {ground_force_zero_steps}/{torque_cap_steps} "
        f"({100.0 * ground_force_zero_steps / torque_cap_steps if torque_cap_steps else 0.0:.1f}%), "
        f"최대 지면 반력={ground_force_moving_max:.2f}N, "
        f"홀드력 평균={lin_force_moving_mean:.2f}N 최대={lin_force_moving_max:.2f}N"
    )
    print(f"2단계 최악 추종오차={worst_pos_moving:.4f}m, 최악 기울기={worst_tilt_moving:.3f}deg, "
          f"최악 요={worst_yaw_moving:.3f}deg, "
          f"span 변화폭={worst_span_dev2:.2f}mm, "
          f"발 이동거리 최대={worst_foot_travel_mm:.2f}mm(참고용, 대부분 목표 이동량), "
          f"목표 대비 잔차 최대={worst_foot_vs_target_residual_mm:.2f}mm, "
          f"리셋 {resets2}회 {sorted(fired2) or ''}")

    span_final = span_mm()
    failures = []
    # Resets themselves are not failures here — the robot falling under a
    # raw held zero_action over many steps is an unrelated, pre-existing
    # property of this environment (the trained policy actively balances;
    # a fixed default pose alone eventually doesn't), not something this
    # script is testing. What would be a failure is not having a clean
    # settled window to judge afterward (1단계, steady_span1 below) or the
    # forced-movement premise silently lapsing (2단계, re-forced above).
    if steady_span1 < MIN_STEADY_STEPS:
        failures.append(
            f"1단계 마지막 리셋 이후 정상상태 구간이 {steady_span1}스텝뿐 — 판정 근거로 너무 짧음 "
            f"(--wait-steps 를 늘리거나 재실행 하십시오)"
        )
    if worst_pos_settled > args.pos_tol_m:
        failures.append(f"1단계 정상상태 위치오차 {worst_pos_settled:.4f}m > 허용 {args.pos_tol_m}m")
    if worst_ang_settled > args.ang_tol_deg:
        failures.append(f"1단계 정상상태 자세오차 {worst_ang_settled:.3f}deg > 허용 {args.ang_tol_deg}deg")
    if worst_tilt_settled > args.tilt_tol_deg:
        failures.append(f"1단계 정상상태 기울기 {worst_tilt_settled:.3f}deg > 허용 {args.tilt_tol_deg}deg (전도 의심)")
    # tilt_deg() 는 요 불변이라 요만 틀어지는 오작동을 못 잡는다 —
    # span/tilt 가 모두 정상으로 보여도 이 게이트가 따로 걸린다.
    if worst_yaw_settled > args.yaw_tol_deg:
        failures.append(f"1단계 정상상태 요 오차 {worst_yaw_settled:.3f}deg > 허용 {args.yaw_tol_deg}deg")
    if worst_span_dev1 > args.span_tol_mm:
        failures.append(f"1단계 span 변화폭 {worst_span_dev1:.2f}mm > 허용 {args.span_tol_mm}mm (전도 의심)")
    # pos_err/span alone can both look fine while the dummy is measurably
    # tipped (measured: span read 1358mm — "펴짐" — while tilt was
    # ~16deg, because a fixed-joint assembly's span shortfall is always a
    # rigid tilt but the *direction* of tilt does not have to shrink span
    # much). A hold that stops tracking entirely would also pass a bare
    # span/pos_err check trivially (nothing left to disturb it) — the
    # tilt and tracking-error gates below are what a frozen hold cannot.
    if worst_pos_moving > args.pos_tol_moving_m:
        failures.append(f"2단계 추종오차 {worst_pos_moving:.4f}m > 허용 {args.pos_tol_moving_m}m")
    if worst_tilt_moving > args.tilt_tol_moving_deg:
        failures.append(f"2단계 최악 기울기 {worst_tilt_moving:.3f}deg > 허용 {args.tilt_tol_moving_deg}deg (전도 의심)")
    if worst_yaw_moving > args.yaw_tol_moving_deg:
        failures.append(f"2단계 최악 요 오차 {worst_yaw_moving:.3f}deg > 허용 {args.yaw_tol_moving_deg}deg")
    if worst_span_dev2 > args.span_tol_moving_mm:
        failures.append(f"2단계 span 변화폭 {worst_span_dev2:.2f}mm > 허용 {args.span_tol_moving_mm}mm (전도 의심)")
    # No gate on the zero-ground-reaction fraction or the mean hold force
    # of phase 2. Thresholds for the two were sized against a bouncing
    # configuration (mu=0.4, no feedforward: 65% zero-ground steps, mean hold
    # force 3800N) and the same with the velocity feedforward added
    # (43%, 2039.73N) — both numbers measured in *this* script's phase 2,
    # where the robot is held at zero_action and the dummy only chases its
    # own scripted target.
    #
    # Such gates pass the very bouncing configuration they are written to
    # catch, fail the feedforward configuration at seed 3, and make the script
    # exit 1 at its own documented default arguments. The problem is not
    # the threshold values but where they were measured: this zero_action
    # phase-2 window is not the regime the "bounce" is a real concern in.
    # The actual policy-driven entry phase (the robot walking, e.g.
    # check_fall_termination.py's "A" phase) reports zero-ground-force
    # fractions of 36-53% under the *same* underlying physics — a
    # threshold sized in one window does not transfer to the other. A gate
    # that passes the configuration it was written to catch and fails the
    # configuration that replaced it is worse than no gate: it
    # manufactures false confidence. Deriving a gate here means
    # measuring in the window where the effect is real (the entry phase),
    # not this one.
    #
    # ground_force_zero_steps/torque_cap_steps and lin_force_moving_mean
    # are computed and printed above, unconditionally — they remain
    # informative about the servo-vs-contact confound generation/README.md
    # documents; they just do not decide this script's exit code.
    # The static-phase weight gate: without it ground_force_settled
    # is computed and printed but never checked against anything, so a
    # regression back to 0 N (or any other wrong value) still exits 0.
    weight_lo = expected_weight_n * (1.0 - args.weight_tol_frac)
    weight_hi = expected_weight_n * (1.0 + args.weight_tol_frac)
    if not (weight_lo <= ground_force_settled <= weight_hi):
        failures.append(
            f"1단계 정상상태 접지 반력(두 발 합) {ground_force_settled:.4f}N 이 m*g={expected_weight_n:.2f}N "
            f"의 허용범위 [{weight_lo:.2f}, {weight_hi:.2f}]N 밖 — 0 이면 결함 재발, 그 밖이면 PD 가 "
            f"수직 방향에 관여하고 있다는 뜻"
        )

    print(f"\n최종 z 스프레드={span_final:.2f}mm (초기 {span0:.2f}mm) 최종 기울기={tilt_deg():.3f}deg 최종 요={yaw_deg():.3f}deg")
    if failures:
        print("[결함]")
        for f in failures:
            print(f"  - {f}")
        code = 1
    else:
        print("PD 홀드가 대기 중 수렴하고, 진입 중에도 서 있는 채로 목표를 따라갑니다.")
        code = 0

    env.close()
except Exception:
    import traceback

    print("check_dummy_hold 실행 실패:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    code = 1
finally:
    # See check_scene.py: close(exit_code=code) is what preserves a nonzero
    # code through this build's fast-shutdown path.
    simulation_app.close(exit_code=code)

raise SystemExit(code)
