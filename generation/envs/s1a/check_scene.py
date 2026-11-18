#!/usr/bin/env python3
"""더미가 Isaac Lab 씬에 아티큘레이션으로 올라가는지 확인한다.

    ~/isaacsim/python.sh generation/envs/s1a/check_scene.py

더미는 고정조인트로만 묶인 0-DOF 아티큘레이션이다. Isaac Lab 의 Articulation 이
이것을 받아들이는지가 S1a 씬 구성의 갈림길이라, 환경을 더 쌓기 전에 확인한다.

**물리 백엔드를 physx 로 고정한다.** H1FlatEnvCfg 의 `sim.physics` 와
`scene.contact_forces` 는 Hydra 프리셋(`PresetCfg`) 으로 감싸여 있다. Hydra CLI 나
`parse_env_cfg` 를 거치지 않고 `S1aEnvCfg()` 를 바로 인스턴스화하면 이 프리셋이
풀리지 않은 채로 남는데, `parse_env_cfg` 의 주석이 정확히 이 문제를 말한다 —
"so the config is usable without a Hydra CLI override (e.g. in tests)".
풀리지 않은 `PresetCfg` 는 `PhysxCfg`/`ContactSensorCfg` 가 아니므로 시뮬레이션
컨텍스트 구성이 깨지거나 접촉 센서가 조용히 등록되지 않는다. `resolve_presets` 를
`parse_env_cfg` 와 같은 방식으로 직접 호출해 physx 프리셋으로 명시적으로 고정한다.

10 스텝 뒤 접촉력도 함께 확인한다: 위치 변위만으로는 자가 충돌을 판정할 수 없다 —
고정조인트가 상대 운동을 막고, 링크 간 접촉력은 강체 내부력이라 어느 쪽이든 위치에는
똑같이 균일한 변위로 보인다. 실제 위험(비인접 부위끼리의 겹침)은 힘에서만 드러난다.

종료 코드: 0 아티큘레이션으로 올라가고 접지 외 부위 접촉력이 임계값 이하 /
1 아티큘레이션 거부됨(대안은 kinematic 강체 묶음) 또는 자가 충돌 의심
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# `~/isaacsim/python.sh <script>` puts only the script's own directory on
# sys.path, not the repository root, so `generation` (an implicit namespace
# package — no generation/__init__.py) isn't importable without this. pytest
# gets the same effect from pytest.ini's `pythonpath = .`; a bare script
# invocation has no equivalent, so it's done here explicitly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

    # selected=("physx",) matches the "physx" alias field H1FlatEnvCfg's local
    # PhysicsCfg preset declares (physx = default); other PresetCfg nodes in
    # the tree (e.g. scene.contact_forces) don't have a "physx"-named field,
    # so _pick_alternative() falls back to their "default", which is also the
    # PhysX class in this env. Either way, physx is what ends up wired in.
    cfg = resolve_presets(S1aEnvCfg(), selected=("physx",))

    env = ManagerBasedRLEnv(cfg=cfg)
    dummy = env.scene["dummy"]
    sensor = env.scene["dummy_contact"]
    print(f"더미 링크 {dummy.num_bodies}개 · DOF {dummy.num_joints}개")
    print(f"접촉 센서 바디 {len(sensor.body_names)}개: {sensor.body_names[:4]} …")
    # The two counts above are compared, not only printed:
    # otherwise a contact sensor that only wires up part of the
    # articulation (e.g. 12 of 17 bodies) would print that half-wired count
    # and still exit 0. The articulation's own num_bodies is the
    # asset-derived expectation (not a hand-typed 17); the sensor is
    # supposed to cover every one of them.
    if len(sensor.body_names) != dummy.num_bodies:
        raise RuntimeError(
            f"접촉 센서가 더미 바디 일부만 찾음 — 센서 {len(sensor.body_names)}개 vs "
            f"아티큘레이션 {dummy.num_bodies}개 (절반쯤 와이어링된 센서가 조용히 통과해서는 안 됨)"
        )

    env.reset()
    # check_scene.py 가 눈으로 흔들림을 볼 수 없으므로, 스텝 전후 더미 링크의 월드
    # 위치를 찍어 비교한다. 이동량 자체는 참고용 진단이다 — 아래에서 이것만으로는
    # 자가 충돌을 판정할 수 없는 이유를 설명한다.
    # body_pos_w/net_forces_w are ProxyArray (warp-backed) on this Isaac Lab
    # version -- .torch is the explicit, non-deprecated accessor, the same
    # convention recorder.py/commands.py already use (implicit indexing
    # works too but emits a DeprecationWarning).
    before = dummy.data.body_pos_w.torch.clone()
    for _ in range(10):
        env.step(torch.zeros_like(env.action_manager.action))
    after = dummy.data.body_pos_w.torch.clone()

    disp_mm = (after - before).norm(dim=-1).squeeze(0) * 1000.0  # (num_bodies,)
    worst_idx = int(disp_mm.argmax().item())
    print(f"10 스텝 최대 링크 이동: {disp_mm[worst_idx].item():.3f} mm ({dummy.body_names[worst_idx]})")
    print("  부위별 이동(mm): " + ", ".join(
        f"{name}={d:.3f}" for name, d in zip(dummy.body_names, disp_mm.tolist())
    ))

    # Position can't tell self-collision apart from a healthy fixed-joint
    # assembly: contact forces between two linked bodies are internal to the
    # rigid structure, and the fixed joints forbid relative motion regardless
    # of whether self-collision is on — uniform displacement is what BOTH
    # states look like. PhysX also excludes joint-connected parent/child pairs
    # from self-collision by construction, so the real risk (non-adjacent
    # overlaps this asset has by construction — e.g. an upper arm against the
    # chest, or the two thighs against each other) never shows up as a
    # position artifact either; the rigid assembly just absorbs it silently.
    # The only place it's observable is force: with self-collision truly off,
    # nothing should be pushing on any body except the ground, and nothing in
    # this scene touches the ground except the lowest regions (BR-12, the
    # shins — confirmed against anthropometry.yaml). Every other body should
    # read ~0.
    forces_n = sensor.data.net_forces_w.torch.norm(dim=-1).squeeze(0)  # (num_bodies,)
    force_names = sensor.body_names
    print("  부위별 접촉력(N): " + ", ".join(
        f"{name}={f:.4f}" for name, f in zip(force_names, forces_n.tolist())
    ))
    # Ground-contact prim names read through the asset's own keti:bodyRegion
    # tag (BR-12), not hand-typed -- CONTRIBUTING.md 「관절 순서」: a prim name like
    # "BR_12_left" is not the canon, the BR code on the prim is.
    GROUND_CONTACT_REGIONS = ground_contact_body_names(str(DUMMY_USD), force_names)
    max_force = forces_n.max().item()
    # Relative to the load-bearing force rather than an absolute newton value:
    # geometry-only builds have no meta/rules.json mass values (all null), so
    # PhysX assigns default collision-derived masses and there is no
    # standard's-worth of a physically meaningful absolute threshold to check
    # against yet. 1% of whatever is holding the dummy up leaves two orders of
    # magnitude of margin against solver noise while still catching a real
    # leak (a self-collision push is a fraction of body weight, not 1% of it).
    leak_threshold_n = max(0.01 * max_force, 1e-6)
    leaks = [
        (name, f) for name, f in zip(force_names, forces_n.tolist())
        if name not in GROUND_CONTACT_REGIONS and f > leak_threshold_n
    ]
    if leaks:
        print(f"자가 충돌 의심 — 접지와 무관한 부위에 힘이 잡혔습니다 (임계값 {leak_threshold_n:.4f} N 초과):")
        for name, f in leaks:
            print(f"  {name}: {f:.4f} N")
        raise RuntimeError(f"Self-collision leak on non-ground bodies: {leaks}")
    print(f"접지 부위({sorted(GROUND_CONTACT_REGIONS)}) 외 전부 임계값 이하 — 자가 충돌이 꺼져 있습니다")

    print("10 스텝 완주 — 더미가 아티큘레이션으로 올라갑니다")
    env.close()
    code = 0
except Exception:
    import traceback

    print("씬 구성 실패:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    code = 1
finally:
    # SimulationApp.close() defaults to exit_code=0 and, on this build's fast-
    # shutdown path, hard-exits the process via os._exit() before any code
    # after close() runs — confirmed empirically: a bare close() here would
    # silently turn a caught, printed failure (code = 1) into process exit 0.
    # Passing exit_code explicitly is what the method exists for: nonzero
    # values are flushed and preserved instead of being replaced by Kit's
    # own 0. Without this, the exit-code contract this script documents
    # (0 = up as articulation, 1 = rejected) would not hold.
    simulation_app.close(exit_code=code)

raise SystemExit(code)
