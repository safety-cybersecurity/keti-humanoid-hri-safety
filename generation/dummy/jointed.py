"""Turn the rigid dummy's USD into the jointed one (v2), and read v2 back. Needs pxr (imported inside).

apply() runs inside build_dummy.build() after v1's bodies and fixed joints exist and before the skin is
attached: v1's code path is untouched, so v1 rebuilds to the byte. It adds a foot under each ankle (a
rigid body with no body region -- tagged keti:groundContact instead), replaces the eight fixed joints the
spec names with revolute or D6 joints and joins each foot by its D6 ankle, all at the derived pivots, sets
one density per collider from the spec's segment masses (no physics:mass anywhere: the release pre-flight
refuses it) and authors the walking drives so the asset describes itself; the environment sets the same
gains from the spec."""
from __future__ import annotations

import hashlib
import math

from generation.dummy import joints_spec as js

COLLIDER_CHILD = "collider"


def _shape_prim(stage, name: str, split_body: bool):
    return stage.GetPrimAtPath(f"/KetiDummy/{name}/{COLLIDER_CHILD}" if split_body else f"/KetiDummy/{name}")


def _add_foot(stage, foot: js.FootBox, split_body: bool):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    path = f"/KetiDummy/{foot.name}"
    translate = Gf.Vec3d(*foot.center)
    if split_body:
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(translate)
        geom = UsdGeom.Cube.Define(stage, f"{path}/{COLLIDER_CHILD}")
        UsdGeom.Imageable(geom.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)
    else:
        geom = UsdGeom.Cube.Define(stage, path)
        geom.AddTranslateOp().Set(translate)
    geom.CreateSizeAttr(1.0)
    geom.AddScaleOp().Set(Gf.Vec3f(*foot.size))
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    prim.CreateAttribute("keti:groundContact", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
    return prim


def _local(stage, body_name: str, split_body: bool, point):
    """A world point as a joint's localPos on that body is read -- the way build_dummy writes v1's anchors:
    in the shape's (scaled) frame, and with the skin, where the body is an unscaled Xform over its collider,
    as the float32 product with the collider's scale the engine reads off the bare asset (engine_anchor)."""
    from pxr import Gf, Usd, UsdGeom

    from generation.dummy.build_dummy import _scale_op, engine_anchor

    shape = _shape_prim(stage, body_name, split_body)
    xf = UsdGeom.Xformable(shape).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    local = Gf.Vec3f(*[float(v) for v in xf.GetInverse().Transform(Gf.Vec3d(*point))])
    return Gf.Vec3f(*engine_anchor(local, _scale_op(shape))) if split_body else local


def _joint(stage, j: js.JointDef, split_body: bool, gains):
    from pxr import Gf, Sdf, UsdPhysics

    path = f"/KetiDummy/{j.child}/{j.name}"
    if j.kind == "revolute":
        jt = UsdPhysics.RevoluteJoint.Define(stage, path)
        jt.CreateAxisAttr(j.dofs[0].axis.upper())
        jt.CreateLowerLimitAttr(math.degrees(j.dofs[0].lower))
        jt.CreateUpperLimitAttr(math.degrees(j.dofs[0].upper))
        drives = [("angular", j.dofs[0])]
    else:
        jt = UsdPhysics.Joint.Define(stage, path)
        prim = jt.GetPrim()
        for t in ("transX", "transY", "transZ"):
            lim = UsdPhysics.LimitAPI.Apply(prim, t)
            lim.CreateLowAttr(1.0)
            lim.CreateHighAttr(-1.0)                  # low > high: locked
        free = {d.axis: d for d in j.dofs}
        drives = []
        for axis in "xyz":
            lim = UsdPhysics.LimitAPI.Apply(prim, f"rot{axis.upper()}")
            if axis in free:
                lim.CreateLowAttr(math.degrees(free[axis].lower))
                lim.CreateHighAttr(math.degrees(free[axis].upper))
                drives.append((f"rot{axis.upper()}", free[axis]))
            else:
                lim.CreateLowAttr(1.0)
                lim.CreateHighAttr(-1.0)
    jt.CreateBody0Rel().SetTargets([Sdf.Path(f"/KetiDummy/{j.parent}")])
    jt.CreateBody1Rel().SetTargets([Sdf.Path(f"/KetiDummy/{j.child}")])
    jt.CreateLocalPos0Attr(_local(stage, j.parent, split_body, j.pivot))
    jt.CreateLocalPos1Attr(_local(stage, j.child, split_body, j.pivot))
    jt.CreateLocalRot0Attr(Gf.Quatf(1.0))
    jt.CreateLocalRot1Attr(Gf.Quatf(1.0))
    stiffness, damping = gains
    for instance, _ in drives:
        drive = UsdPhysics.DriveAPI.Apply(jt.GetPrim(), instance)
        drive.CreateTypeAttr("force")
        # USD angular drives are per degree; the spec's gains are per radian.
        drive.CreateStiffnessAttr(stiffness * math.pi / 180.0)
        drive.CreateDampingAttr(damping * math.pi / 180.0)


def apply(stage, spec: js.JointedSpec, *, split_body: bool) -> dict:
    from pxr import UsdPhysics

    for foot in spec.feet:
        _add_foot(stage, foot, split_body)
    for j in spec.joints:
        old = stage.GetPrimAtPath(f"/KetiDummy/{j.child}/fixed_to_{j.parent}")
        if j.group != "ankle":
            if not old or old.GetTypeName() != "PhysicsFixedJoint":
                raise SystemExit(f"{j.name}: 바꿀 고정 관절이 없습니다 — /KetiDummy/{j.child}/fixed_to_{j.parent}")
            stage.RemovePrim(old.GetPath())
        _joint(stage, j, split_body, spec.drives["walking"][j.group])
    for name, rho in spec.density.items():
        shape = _shape_prim(stage, name, split_body)
        UsdPhysics.MassAPI.Apply(shape).CreateDensityAttr(float(rho))
    root = stage.GetPrimAtPath("/KetiDummy")
    root.SetCustomDataByKey("keti:dummyVersion", "v2-jointed")
    root.SetCustomDataByKey("keti:jointsSpecSha256", hashlib.sha256(spec.path.read_bytes()).hexdigest())
    return {"feet": [f.name for f in spec.feet], "joints": [j.name for j in spec.joints],
            "replaced_fixed_joints": sum(j.group != "ankle" for j in spec.joints),
            "dofs": len(js.dof_names(spec)), "total_mass_kg": spec.total_mass}


def embed_gait(stage, table: dict, sha256: str) -> None:
    from pxr import Sdf, UsdGeom, Vt

    from generation.dummy.gait.gait_table import GaitTable

    GaitTable.from_json(table)                       # refuse a malformed table before writing it
    prim = UsdGeom.Scope.Define(stage, "/KetiDummy/Gait").GetPrim()
    prim.CreateAttribute("keti:gait:dofNames", Sdf.ValueTypeNames.StringArray, custom=True).Set(
        Vt.StringArray(table["dof_names"]))
    prim.CreateAttribute("keti:gait:speeds", Sdf.ValueTypeNames.DoubleArray, custom=True).Set(Vt.DoubleArray(table["speeds"]))
    prim.CreateAttribute("keti:gait:stride", Sdf.ValueTypeNames.DoubleArray, custom=True).Set(Vt.DoubleArray(table["stride"]))
    prim.CreateAttribute("keti:gait:shape", Sdf.ValueTypeNames.IntArray, custom=True).Set(Vt.IntArray(table["shape"]))
    prim.CreateAttribute("keti:gait:targets", Sdf.ValueTypeNames.FloatArray, custom=True).Set(Vt.FloatArray(table["targets"]))
    src = table.get("source") or {}
    prim.CreateAttribute("keti:gait:source", Sdf.ValueTypeNames.String, custom=True).Set(
        f"{src.get('credit', '')} ({src.get('license', '')}, {src.get('page', '')}) — {src.get('changes', '')}")
    prim.SetCustomDataByKey("keti:gaitSha256", sha256)


def read_body_codes(usd_path: str, body_names) -> list:
    """Per contact-sensor body, its region code (keti:bodyRegion), or None for a ground-contact body
    (keti:groundContact -- the jointed dummy's feet). A body with neither tag is refused."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    code, ground = {}, set()
    for prim in stage.Traverse():
        attr = prim.GetAttribute("keti:bodyRegion")
        if attr and attr.Get():
            code[prim.GetName()] = attr.Get()
        flag = prim.GetAttribute("keti:groundContact")
        if flag and flag.Get() is True:
            ground.add(prim.GetName())
    missing = [n for n in body_names if n not in code and n not in ground]
    if missing:
        raise ValueError(f"keti:bodyRegion 도 keti:groundContact 도 없는 바디: {missing}")
    return [code.get(n) for n in body_names]


def read_gait(usd_path: str):
    from pxr import Usd

    from generation.dummy.gait.gait_table import GaitTable

    stage = Usd.Stage.Open(str(usd_path))          # held: a prim expires with its stage
    prim = stage.GetPrimAtPath("/KetiDummy/Gait")
    if not prim:
        raise ValueError(f"{usd_path}: 걸음 표(/KetiDummy/Gait)가 없는 인형입니다 — build_dummy.py --jointed --gait")
    get = lambda n: prim.GetAttribute(f"keti:gait:{n}").Get()  # noqa: E731
    return GaitTable.from_json({"dof_names": list(get("dofNames")), "speeds": list(get("speeds")),
                                "stride": list(get("stride")), "shape": list(get("shape")),
                                "targets": list(get("targets"))})
