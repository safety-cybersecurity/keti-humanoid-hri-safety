#!/usr/bin/env python3
"""관절 인형(v2)의 부위별 겉보기 유효질량 — 부위에 짧은 수평 충격량을 주고 그 부위의 속도 변화로 나눈다.

    PYTHONPATH= ~/isaacsim/python.sh generation/dummy/measure_apparent_mass.py --out <json> [--tone-scale 1.0]

확인용 측정이다: 관절로 이어진 몸에서 부딪힌 부위가 얼마만큼의 질량처럼 움직이는가를 재어, 충돌 모형의
파라미터(부위별 유효질량)와 나란히 놓는다. 인형만 바닥에 세우고 붙잡지 않으며 관절은 놓인 몸의 강성이다.
부위마다 다시 세워 한 번씩 민다. 속도 변화와 함께 그 스텝의 위치 변화도 남겨, 둘이 두 배 넘게 어긋나는 부위는
표시한다(엔진이 한 스텝의 속도를 부풀리는 경우). 결과 파일은 측정 기록이고 자산에는 아무것도 쓰지 않는다."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--tone-scale", type=float, default=1.0)
parser.add_argument("--impulse", type=float, default=10.0, help="N*s, given over one physics step")
args = parser.parse_args()

from generation.isaac_boot import launch_app  # noqa: E402

simulation_app = launch_app({"headless": True})[1]
code = 1
try:
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg

    from generation.dummy import joints_spec as js
    from generation.envs.s1d.s1d_env_cfg import V2_USD

    spec = js.load()
    names = js.dof_names(spec)
    by_group = {}
    for j in spec.joints:
        start = js.columns(spec)[j.name].start
        by_group.setdefault(j.group, []).extend(names[start:start + len(j.dofs)])
    dt = 0.005
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    dummy = Articulation(ArticulationCfg(
        prim_path="/World/Dummy",
        spawn=sim_utils.UsdFileCfg(usd_path=str(V2_USD), activate_contact_sensors=False,
                                   articulation_props=PhysxArticulationRootPropertiesCfg(enabled_self_collisions=False)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        actuators={g: ImplicitActuatorCfg(joint_names_expr=[f"^{n}$" for n in dofs],
                                          stiffness=spec.drives["tone"][g][0] * args.tone_scale,
                                          damping=spec.drives["tone"][g][1] * args.tone_scale)
                   for g, dofs in by_group.items()}))
    sim.reset()
    state0 = dummy.data.root_state_w.torch.clone()
    bodies = list(dummy.body_names)
    rules = json.loads((REPO / "meta/rules.json").read_text(encoding="utf-8"))["contact_limits"]
    result = {}
    for code_, region in spec.anthropometry["regions"].items():
        name = js.prim_of(code_, "left", spec.anthropometry["regions"])
        b = bodies.index(name)
        dummy.write_root_state_to_sim(state0.clone())
        zeros = torch.zeros(1, len(dummy.joint_names), device=sim.device)
        dummy.write_joint_state_to_sim_index(position=zeros, velocity=zeros)
        dummy.set_joint_position_target_index(target=zeros)
        for _ in range(4):
            dummy.write_data_to_sim()
            sim.step()
            dummy.update(dt)
        v0 = dummy.data.body_lin_vel_w.torch[0, b, 0].item()
        p0 = dummy.data.body_pos_w.torch[0, b, 0].item()
        force = torch.zeros(1, 1, 3, device=sim.device)
        force[0, 0, 0] = args.impulse / dt
        dummy.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force, torques=torch.zeros_like(force), body_ids=[b], is_global=True)
        dummy.write_data_to_sim()
        sim.step()
        dummy.update(dt)
        dummy.permanent_wrench_composer.set_forces_and_torques_index(
            forces=torch.zeros_like(force), torques=torch.zeros_like(force), body_ids=[b], is_global=True)
        v1 = dummy.data.body_lin_vel_w.torch[0, b, 0].item()
        p1 = dummy.data.body_pos_w.torch[0, b, 0].item()
        dv = v1 - v0
        # the step's own displacement over dt, less the speed the body already had: the speed change the position saw
        dv_from_dx = (p1 - p0) / dt - v0
        agree = bool(dv and dv_from_dx and 0.5 <= dv_from_dx / dv <= 2.0)
        table = ((rules.get(code_) or {}).get("biomechanical") or {}).get("effective_mass") or {}
        result[code_] = {"body": name, "apparent_mass_kg": args.impulse / dv if dv else float("nan"),
                         "dv_mps": dv, "dx_m": p1 - p0, "dv_from_dx_mps": dv_from_dx, "dv_dx_agree": agree,
                         "rigid_mass_kg": spec.mass[name], "table_a3_effective_mass_kg": table.get("value")}
        print(f"{code_} {name:12s} 겉보기 {result[code_]['apparent_mass_kg']:.2f} kg · 강체 {spec.mass[name]:.2f} kg"
              f" · 표 A.3 {table.get('value')} · dv {dv:.3f} · dx/dt {dv_from_dx:.3f}{'' if agree else ' (어긋남)'}")
    Path(args.out).write_text(json.dumps({"tone_scale": args.tone_scale, "impulse_ns": args.impulse, "dt_s": dt,
                                          "regions": result}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    code = 0
except Exception:
    import traceback
    traceback.print_exc()
    code = 1
finally:
    simulation_app.close(exit_code=code)
raise SystemExit(code)
