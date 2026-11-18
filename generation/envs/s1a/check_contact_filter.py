#!/usr/bin/env python3
"""더미 접촉 센서가 로봇에 대한 접촉만 따로 보고할 수 있는지 확인한다.

    ~/isaacsim/python.sh generation/envs/s1a/check_contact_filter.py [--steps 700]

**왜.** 지금 `observation.contact.type` 은 부위별 힘에 **전역 불리언 하나**(로봇의
발이 아닌 링크가 하중을 받는지)를 곱해서 정합니다. 두 가지가 틀어집니다.

  - 로봇 **발**이 넘어진 더미를 밟는 접촉은 게이트가 발을 빼므로 아예 안 잡힙니다 —
    `quasistatic` 이라는 값이 존재하는 이유가 정확히 그 상황인데도 그렇습니다
  - 게이트가 켜지면 **지면 하중만 받는 부위까지** 유형이 매겨집니다. 넘어진 더미는
    BR-12 외 부위도 지면에 닿으므로, 로봇이 건드린 적 없는 부위가 Table A.2 의
    한계와 비교될 수 있습니다

둘 다 원인이 하나입니다 — **부위별 힘이 어느 상대와의 접촉인지 모릅니다.**

**이 스크립트가 묻는 것.** `filter_prim_paths_expr` 로 로봇을 지정하면 PhysX 가
`force_matrix_w`(부위 × 필터 대상)를 채워 주는가. 이전 시도는 로봇의 바디 21개를
글로브로 지정했다가 "Filter pattern ... did not match the correct number of entries
(expected 17, found 21)" 로 거부당했습니다(s1a_env_cfg.py 주석). 그 제약이 **필터
대상 개수**에 대한 것이라면, 로봇 전체를 **프림 하나**로 지정하면 통과할 수 있습니다.
확인해 본 적이 없어 확인합니다.

성공하면 부위별로 "로봇이 준 힘" 이 직접 나오고, 위 두 결함이 함께 닫힙니다.

**실측 결과 (2026-08-21) — 세 변형 모두 귀속을 얻지 못했습니다.**

| 변형 | 결과 |
|---|---|
| `root` (`/Robot`) | 센서는 붙고 모양도 `(1, 17, 1, 3)` 으로 나오지만 700 스텝에서 로봇 접촉 **0건** |
| `torso` (`/Robot/torso_link`) | 동일 — 붙고, 0건 |
| `glob` (`/Robot/.*`) | 생성 실패. PhysX: `"did not match any rigid contact for filters"` → `AttributeError: 'NoneType' object has no attribute 'filter_count'` |

`glob` 의 오류 메시지가 이전 기록(`"expected 17, found 21"`)과 다릅니다 — 개수 문제가
아니라 **그 센서/필터 조합에 해당하는 rigid contact 항목 자체가 없다**는 것입니다.
즉 필터가 붙는 것과 값이 나오는 것은 별개이고, 이 경로로는 부위별 로봇 귀속을 얻지
못합니다. 다른 경로(예: PhysX 접촉 보고 쌍을 미리 등록하는 방식)를 찾아야 합니다.

종료 코드: 0 판정이 나옴(성립/불성립 무관) / 1 실행 실패
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="더미 접촉 센서의 로봇 필터 가능 여부")
parser.add_argument("--steps", type=int, default=700, help="측정할 env 스텝 수")
parser.add_argument("--seq", type=int, default=0, help="에피소드 시퀀스 번호")
parser.add_argument("--filter-variant", default="root", choices=("root", "torso", "glob"),
                    help="필터 대상 프림 표기 방식")
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
    from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    import isaaclab_tasks  # noqa: F401
    import generation.envs.register  # noqa: F401

    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_tasks.utils.hydra import resolve_presets

    from generation.envs.s1a.recorder import read_region_codes
    from generation.envs.s1a.s1a_env_cfg import DUMMY_USD, S1aEnvCfg
    from generation.seeding import episode_seed, scenario_startup_seed

    SCENARIO_ID = "S1a"

    cfg = resolve_presets(S1aEnvCfg(), selected=("physx",))
    cfg.seed = scenario_startup_seed(SCENARIO_ID)

    # The one thing under test: a filtered sensor alongside the unfiltered one
    # the env already has, so both are read from the same physics.
    # Several spellings of "the robot", because the shape being accepted and
    # the values being non-zero are different questions. An earlier attempt
    # (s1a_env_cfg.py's comment) passed the robot's 21 body paths as a glob
    # and was rejected outright; the root prim alone is accepted but may have
    # no collider of its own to report against.
    FILTER_VARIANTS = {
        "root": ["{ENV_REGEX_NS}/Robot"],
        "torso": ["{ENV_REGEX_NS}/Robot/torso_link"],
        "glob": ["{ENV_REGEX_NS}/Robot/.*"],
    }
    variant = args.filter_variant
    if variant not in FILTER_VARIANTS:
        raise RuntimeError(f"--filter-variant 는 {sorted(FILTER_VARIANTS)} 중 하나여야 합니다")
    print(f"필터 변형: {variant} = {FILTER_VARIANTS[variant]}")
    cfg.scene.dummy_contact_vs_robot = PhysXContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Dummy/.*",
        history_length=0,
        filter_prim_paths_expr=FILTER_VARIANTS[variant],
    )

    agent_cfg = load_cfg_from_registry("Keti-S1a-v0", "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Flat-H1-v0"))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    filtered = base.scene["dummy_contact_vs_robot"]
    plain = base.scene["dummy_contact"]
    codes = read_region_codes(str(DUMMY_USD), list(plain.body_names))

    fm = filtered.data.force_matrix_w
    print(f"필터 센서 생성 성공. force_matrix_w = {'None' if fm is None else tuple(fm.torch.shape)}")
    if fm is None:
        print("→ 필터가 붙지 않았습니다. 이 경로로는 쌍 단위 귀속을 얻을 수 없습니다.")
        code = 0
        raise SystemExit  # falls to finally

    env.seed(episode_seed(SCENARIO_ID, args.seq))
    with torch.inference_mode():
        obs, _ = env.reset()

    robot_only_steps = 0
    ground_only_steps = 0
    both = 0
    worst = {}
    for _ in range(args.steps):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        f_all = torch.linalg.norm(plain.data.net_forces_w.torch[0], dim=-1).cpu().numpy()
        f_rob = torch.linalg.norm(
            filtered.data.force_matrix_w.torch[0].sum(dim=1), dim=-1).cpu().numpy()
        any_all = bool((f_all > 1.0).any())
        any_rob = bool((f_rob > 1.0).any())
        if any_rob and any_all:
            both += 1
        elif any_rob:
            robot_only_steps += 1
        elif any_all:
            ground_only_steps += 1
        for c, v in zip(codes, f_rob):
            if v > 1.0:
                worst[c] = max(worst.get(c, 0.0), float(v))
        if bool(dones[0]):
            break

    print(f"\n측정 {args.steps} 스텝")
    print(f"  로봇 접촉만 있는 스텝      : {robot_only_steps}")
    print(f"  지면 등 다른 접촉만 있는 스텝: {ground_only_steps}")
    print(f"  둘 다                       : {both}")
    print(f"  로봇 접촉을 낸 부위 {len(worst)}: "
          f"{', '.join(f'{k}({v:.0f}N)' for k, v in sorted(worst.items())) or '없음'}")
    print("\n=== 판정 ===")
    if worst:
        print("필터가 실제로 로봇 접촉을 보고합니다 — 부위별 '로봇이 준 힘' 을 얻을 수 있습니다.")
        print("→ contact.type 의 전역 불리언 게이트를 이 값으로 대체하면 발 접촉도 잡히고,")
        print("  지면 하중만 받는 부위가 유형을 받는 문제도 함께 사라집니다.")
    else:
        print("필터는 붙었지만(모양이 나옴) 접촉을 하나도 보고하지 않았습니다.")
        print("→ 필터 대상 프림에 콜라이더가 없으면 이렇게 됩니다. 모양이 맞는 것과")
        print("  값이 나오는 것은 다른 문제이고, 이 변형으로는 귀속을 얻지 못합니다.")
    code = 0
except SystemExit:
    pass
except Exception:
    import traceback

    print("실행 실패:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    code = 1
finally:
    simulation_app.close(exit_code=locals().get("code", 1))

raise SystemExit(locals().get("code", 1))
