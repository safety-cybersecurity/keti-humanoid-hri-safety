#!/usr/bin/env python3
"""How Isaac Lab reads a multi-axis (D6) joint: what it names the axes and what a joint position means.

    PYTHONPATH= ~/isaacsim/python.sh generation/dummy/probe_d6.py --out tests/generation/d6_convention_measured.json

Builds a scratch articulation -- a fixed base with four links: on a three-axis D6 ("tri"), a two-axis D6
with rotZ locked ("duo"), a revolute joint ("hinge") and a three-axis D6 whose third axis has a narrow
range ("ankle", as the dummy's ankle: x and y move while z is driven to zero) -- drives each one degree of
freedom at a time and then several together, with stiff drives and gravity off, then drives the narrow
axis past its range, and writes the joint names, their limits, the positions the engine reports and each
link's rotation relative to the base. tests/generation/test_d6_convention.py decides from those which
axis each name is, the quaternion order and the convention. Nothing here is a choice: the file records
what the engine did."""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args = parser.parse_args()

from generation.isaac_boot import launch_app  # noqa: E402

simulation_app = launch_app({"headless": True})[1]
code = 1
try:
    import importlib.metadata as metadata

    import torch
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as mu
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg

    JOINTS = {"tri": "xyz", "duo": "xy", "hinge": "y", "ankle": "xyz"}   # no name is a prefix of another
    NARROW_DEG = {"ankle": {"z": 10.0}}       # half-range of an axis held near zero; the others get 170 deg
    COMBINED = {"tri": [[0.3, -0.2, 0.5], [-0.4, 0.1, -0.3], [0.6, 0.6, 0.0]], "duo": [[0.3, -0.5], [-0.2, 0.4]],
                "ankle": [[0.3, -0.5, 0.0], [-0.4, 0.3, 0.0], [0.5, 0.5, 0.0]]}
    PAST_LIMIT = {"ankle": [[0.3, -0.3, 0.5]]}   # the narrow axis driven beyond its range while the others move

    def make_usd(path: Path) -> None:
        st = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(st, 1.0)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/Probe").GetPrim())

        def body(name, z):
            cube = UsdGeom.Cube.Define(st, f"/Probe/{name}")
            cube.CreateSizeAttr(0.1)
            cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
            return cube.GetPrim()

        base = body("base", 1.0)
        UsdPhysics.ArticulationRootAPI.Apply(base)
        for i, (name, axes) in enumerate(JOINTS.items()):
            z = 1.0 - 0.3 * (i + 1)
            link = body(f"link_{name}", z)
            if len(axes) == 1:
                j = UsdPhysics.RevoluteJoint.Define(st, f"/Probe/link_{name}/{name}")
                j.CreateAxisAttr(axes.upper())
                UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
            else:
                j = UsdPhysics.Joint.Define(st, f"/Probe/link_{name}/{name}")
                for dof in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
                    lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), dof)
                    if dof.startswith("rot") and dof[-1].lower() in axes:
                        half = NARROW_DEG.get(name, {}).get(dof[-1].lower(), 170.0)
                        lim.CreateLowAttr(-half)
                        lim.CreateHighAttr(half)
                        UsdPhysics.DriveAPI.Apply(j.GetPrim(), dof)
                    else:
                        lim.CreateLowAttr(1.0)
                        lim.CreateHighAttr(-1.0)            # low > high locks the axis
            j.CreateBody0Rel().SetTargets([base.GetPath()])
            j.CreateBody1Rel().SetTargets([link.GetPath()])
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, z + 0.15 - 1.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.15))
        st.GetRootLayer().Save()

    usd = Path(tempfile.mkdtemp()) / "probe_d6.usd"
    make_usd(usd)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, gravity=(0.0, 0.0, 0.0)))
    probe = Articulation(ArticulationCfg(
        prim_path="/World/Probe",
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd),
                                   articulation_props=PhysxArticulationRootPropertiesCfg(fix_root_link=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=1.0e5, damping=1.0e3)}))
    sim.reset()
    names, bodies = list(probe.joint_names), list(probe.body_names)
    print("joint names:", names, "bodies:", bodies)

    def run(joint: str, targets: dict) -> dict:
        target = torch.zeros(1, len(names), device=sim.device)
        for n, v in targets.items():
            target[0, names.index(n)] = float(v)
        probe.set_joint_position_target_index(target=target)
        for _ in range(400):
            probe.write_data_to_sim()
            sim.step()
            probe.update(sim.get_physics_dt())
        q = probe.data.body_quat_w.torch[0]
        rel = mu.quat_mul(mu.quat_inv(q[bodies.index("base")].unsqueeze(0)), q[bodies.index(f"link_{joint}")].unsqueeze(0))[0]
        jp = probe.data.joint_pos.torch[0]
        mine = [n for n in names if n.startswith(joint)]
        return {"joint": joint, "targets": {n: float(v) for n, v in targets.items()},
                "joint_pos": {n: float(jp[names.index(n)]) for n in mine}, "relative_quat": [float(v) for v in rel]}

    def single_target(joint: str, index: int) -> float:
        # A single case stays inside the range: half the narrow axis's range, else 0.5 rad. The index is
        # taken as the joint's axes in order (x, y, z) -- the test re-derives each name's axis anyway.
        narrow = NARROW_DEG.get(joint, {}).get(JOINTS[joint][index])
        return 0.5 if narrow is None else 0.5 * math.radians(narrow)

    cases = []
    for joint in JOINTS:
        mine = [n for n in names if n.startswith(joint)]
        for i, n in enumerate(mine):
            cases.append(dict(run(joint, {n: single_target(joint, i)}), kind="single"))
        for vec in COMBINED.get(joint, []):
            cases.append(dict(run(joint, dict(zip(mine, vec))), kind="combined"))
        for vec in PAST_LIMIT.get(joint, []):
            cases.append(dict(run(joint, dict(zip(mine, vec))), kind="past_limit"))
    limits = probe.data.joint_pos_limits.torch[0]
    out = {"isaac_lab": metadata.version("isaaclab"), "joint_names": names, "body_names": bodies,
           "joint_pos_limits": {n: [float(limits[i, 0]), float(limits[i, 1])] for i, n in enumerate(names)},
           "cases": cases,
           "api": {m: hasattr(probe, m) for m in ("write_joint_stiffness_to_sim_index", "write_joint_damping_to_sim_index",
                                                  "set_joint_position_target_index", "write_joint_state_to_sim_index")}}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote", args.out)
    code = 0
except Exception:
    import traceback
    traceback.print_exc()
    code = 1
finally:
    simulation_app.close(exit_code=code)
raise SystemExit(code)
