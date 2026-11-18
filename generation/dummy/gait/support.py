"""The height the hold servo's partial weight support aims the jointed dummy at, per step: how far above its stand
height the pelvis rides with the lower sole on the floor under the joint targets the gait sends -- the path those
angles push the body along (retarget.pelvis_path, the formula the gait build smooths). Following the sent targets,
not a stored path, it follows the start and stop ramps and the blend between speed rows. Pure numpy."""
from __future__ import annotations

import numpy as np

from generation.dummy import joints_spec as js
from generation.dummy.gait import kinematics as kin
from generation.dummy.gait import retarget as rt


def pelvis_rise(spec: js.JointedSpec, dof_names, targets) -> np.ndarray:
    """Per row of `targets` (columns named by `dof_names`), the pelvis height over its stand height in metres."""
    return rt.pelvis_path(_spec_rows(spec, dof_names, targets)[2], spec)


def _spec_rows(spec: js.JointedSpec, dof_names, targets):
    names = list(dof_names)
    want = js.dof_names(spec)
    if len(names) != len(set(names)) or set(names) != set(want):
        raise ValueError(f"관절 목표의 DOF {sorted(names)} 가 명세의 DOF {sorted(want)} 와 다릅니다")
    rows = np.asarray(targets, float)
    if rows.ndim != 2 or rows.shape[1] != len(names):
        raise ValueError(f"관절 목표의 모양 {rows.shape} 가 (행, {len(names)}) 이 아닙니다")
    return names, want, rows[:, [names.index(n) for n in want]]


def rest_stance(spec: js.JointedSpec, dof_names, targets) -> np.ndarray:
    """Each row with a foot in the air put down: its ankle lowered by the leg IK (retarget._raise_ankle -- the foot
    keeps its orientation) to the other foot's lowest sole corner, or, where that leg cannot reach so far, the other
    foot raised by a bent knee to meet it (the pelvis comes down by as much -- pelvis_rise says how far). It is the
    stance a walk ends in at rest wherever its last step was: the feet where the walk put them, so the legs do not drag
    them toward the stand pose. The other degrees of freedom are as they were. Rows and columns as given (columns
    named by `dof_names`)."""
    names, want, rows = _spec_rows(spec, dof_names, targets)
    geoms = {s: js.leg_geometry(spec, s) for s in rt.SIDES}

    def lows(row):
        return {s: rt._sole_low(row, spec, s, geoms[s]) for s in rt.SIDES}

    out = rows.copy()
    for i in range(len(rows)):
        low = lows(out[i])
        down, up = sorted(rt.SIDES, key=low.get)
        if low[up] - low[down] > 1e-4:
            out[i], _ = rt._raise_ankle(out[i], spec, up, geoms[up], low[down] - low[up])
            low = lows(out[i])
            if low[up] - low[down] > 1e-4:
                out[i], _ = rt._raise_ankle(out[i], spec, down, geoms[down], low[up] - low[down])
    out, _ = rt.clamp(out, spec)
    back = np.empty_like(out)
    back[:, [names.index(n) for n in want]] = out
    return back
