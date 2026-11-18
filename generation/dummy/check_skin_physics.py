#!/usr/bin/env python3
"""입힌 더미가 v1 과 물리적으로 같은지 — 물리엔진이 읽는 값으로 — 확인한다.

    ~/isaacsim/python.sh generation/dummy/check_skin_physics.py

`meta/assets.json` 은 입힌 더미를 「물리는 v1 과 같다」(`physics_identical_to`)고 적는다.
그런데 두 자산은 프림 배치가 다르다 — v1 은 형상 프림 자체가 강체이고 상자는 scale 로
키웠으며, 입힌 더미는 강체가 Xform 이고 형상은 자식 `collider` 다. 그래서 같은 값을 다른
표현으로 적는다. 이 검사는 USD 를 그대로 비교하지 않고, 물리엔진이 받는 값으로 바꿔
비교한다:

- 강체 — 월드 위치(float32), 강체 역할의 물리 API·속성·관계, `keti:*` 속성
- 형상 — 종류와 실제 치수(상자 반폭 · 구 반지름 · 캡슐 반지름과 반높이, 월드 배율 반영),
  강체 기준 상대 위치(배정밀도로 뺀 뒤 float32), 형상 역할의 물리 API·속성·관계,
  실제로 묶이는 물리 재질의 값
- 고정조인트 앵커 — `localPos × 그 강체의 월드 배율` (float32 곱). v1 은 상자의 **배율이
  들어간** 로컬 프레임에 앵커를 적고 물리엔진이 배율을 곱해 되돌린다. 입힌 더미는 배율
  없는 Xform 에 적으므로, 오프셋을 그대로 적으면 이 왕복과 마지막 비트가 다를 수 있다
- 아티큘레이션 루트가 어느 프림에 있는가, 충돌 제외 쌍(강체 이름으로), 스테이지 단위

역할은 적용된 API 스키마의 정의에서 유도한다 — v1 은 한 프림이 강체이면서 형상이므로,
속성을 프림째 모으면 루트 API 나 kinematic 이 강체에서 콜라이더로 옮겨가도 모른다.

**모르는 것은 통과시키지 않는다.** 회전·전단이 들어간 변환, 균일하지 않은 배율의 구와
캡슐, 밀도가 아닌 명시 질량 속성(MassAPI 의 질량 · 무게중심 · 관성 — 좌표가 저작 프림의
배율 공간이라 정규화가 필요), 역할을 모르는 물리 속성·API, 강체당 형상이 하나가 아닌 경우는
「지원하지 않는 입력」으로 실패한다. 이 자산의 빌더는 그중 어느 것도 만들지 않는다. 관절
인형(`--jointed`)이 형상마다 쓰는 밀도(`physics:density`)는 배율과 무관한 kg/m³ 라 형상의
값으로 그대로 대조하고, 관절의 구동(`drive:`)과 범위(`limit:`)도 관절의 값으로 대조한다 —
그때 빌더는 입힌 판을 같은 판의 맨몸과 대조한다.

**왜 비트까지 보는가.** 2026-09-23 실측: 앵커 16개 중 4개가 float32 한 칸(ulp) 달랐고,
질량·관성·무게중심은 비트까지 같았는데도 S1a 에피소드 0 이 v0.1 배치와 갈라졌다 —
699 스텝 낙상(fail)이 999 스텝 시간 초과로 바뀌었다. 폐루프가 한 칸을 결과로 키운다.
이 검사는 그 조건을 정적으로 막고(빌더도 입힌 더미를 쓸 때마다 부른다), 끝단 확인은
`run_s1a.py --compare-with` 가 한다.

종료 코드: 0 같음 / 1 다름 또는 지원하지 않는 입력 / 2 자산 없음.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CANON = REPO_ROOT / "meta/assets.json"
SKIN_ID = "keti_dummy_ts15066_v1_skin"
# drive: and limit: are the jointed dummy's joint gains and ranges (UsdPhysics DriveAPI / LimitAPI)
PHYSICS_PREFIXES = ("physics:", "physx", "drive:", "limit:")
DENSITY = "physics:density"
MASS_ATTRS = ("physics:mass", DENSITY, "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes")


class Unsupported(ValueError):
    """An input this check does not normalise; comparing it anyway could pass a real difference."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="입힌 더미와 v1 의 물리 기술을 물리엔진이 읽는 값으로 대조한다")
    p.add_argument("--v1", default=None, help="v1 USD (기본: 정본 physics_identical_to 의 usd_path)")
    p.add_argument("--skin", default=None, help="입힌 더미 USD (기본: 정본의 usd_path)")
    return p.parse_args()


def _api_role(api: str) -> str:
    base = api.split(":")[0]
    if "Mass" in base:
        return "mass"
    if "Collision" in base:
        return "shape"
    if any(w in base for w in ("RigidBody", "Articulation", "ContactReport")):
        return "body"
    if base == "PhysicsFilteredPairsAPI":
        return "filter"
    return "other"


def _property_roles(prim) -> dict[str, str]:
    """Property name -> role, from the definitions of the API schemas applied to the prim."""
    from pxr import Usd
    registry = Usd.SchemaRegistry()
    roles = {}
    for api in prim.GetAppliedSchemas():
        base, _, instance = api.partition(":")
        definition = registry.FindAppliedAPIPrimDefinition(base)
        if definition is None:
            continue
        for name in definition.GetPropertyNames():
            roles[name.replace("__INSTANCE_NAME__", instance) if instance else name] = _api_role(api)
    return roles


def _physics_properties(prim, role: str, unsupported: list[str], where: str) -> list[str]:
    """This prim's physics schemas, attributes and relationships that belong to `role`.

    Of the mass attributes only the density passes (role "mass", collected with the shape); describe()
    also reports every MassAPI prim that carries anything else."""
    from pxr import Usd
    roles = _property_roles(prim)
    out = []
    for api in prim.GetAppliedSchemas():
        if not api.startswith(("Physics", "Physx")):
            continue
        r = _api_role(api)
        if r == "other":
            unsupported.append(f"{where}: {api} — 이 검사가 정규화하지 않는 API")
        elif r == role:
            out.append(f"api {api}")
    for prop in prim.GetProperties():
        name = prop.GetName()
        if not name.startswith(PHYSICS_PREFIXES):
            continue
        if isinstance(prop, Usd.Relationship):
            value = sorted(str(t) for t in prop.GetTargets())
            if not value:
                continue
        else:
            if not prop.HasAuthoredValue():
                continue
            value = prop.Get()
        r = roles.get(name)
        if r is None or r == "other" or (r == "mass" and name != DENSITY):
            unsupported.append(f"{where}: {name} — 역할을 모르거나 정규화하지 않는 물리 속성")
        elif r == role:
            out.append(f"{name}={value}")
    return sorted(out)


def _decompose(m, where: str):
    """World matrix -> (translation f64, scale f64) for an unrotated, unsheared transform."""
    lin = np.array([[m[i][j] for j in range(3)] for i in range(3)], dtype=np.float64)
    if np.any(lin[~np.eye(3, dtype=bool)] != 0.0):
        raise Unsupported(f"{where}: 회전 또는 전단이 들어간 변환 — 이 검사는 축 정렬 배율만 정규화한다")
    scale = np.diag(lin).copy()
    if np.any(scale <= 0.0):
        raise Unsupported(f"{where}: 0 이나 음수 배율")
    return np.array(m.ExtractTranslation(), dtype=np.float64), scale


def _physics_material(shape) -> dict | None:
    from pxr import UsdShade
    material, _ = UsdShade.MaterialBindingAPI(shape).ComputeBoundMaterial("physics")   # the purpose UsdPhysics binds materials with
    if not material:
        return None
    prim = material.GetPrim()
    return {a.GetName(): a.Get() for a in prim.GetAttributes()
            if a.GetName().startswith(PHYSICS_PREFIXES) and a.HasAuthoredValue()}


def _shape(shape, body_t, where: str, unsupported: list[str]) -> dict:
    """What the physics engine builds from one collision prim, relative to its body."""
    from pxr import UsdGeom
    t, scale = _decompose(UsdGeom.XformCache().GetLocalToWorldTransform(shape), where)
    kind = shape.GetTypeName()
    f32 = lambda v: float(np.float32(v))                         # noqa: E731
    if kind == "Cube":
        dims = {"half_extents": [f32(0.5 * UsdGeom.Cube(shape).GetSizeAttr().Get() * s) for s in scale]}
    elif kind == "Sphere":
        if not np.all(scale == scale[0]):
            raise Unsupported(f"{where}: 균일하지 않은 배율의 구(타원체)")
        dims = {"radius": f32(UsdGeom.Sphere(shape).GetRadiusAttr().Get() * scale[0])}
    elif kind == "Capsule":
        c = UsdGeom.Capsule(shape)
        axis = "XYZ".index(c.GetAxisAttr().Get())
        transverse = [scale[i] for i in range(3) if i != axis]
        if transverse[0] != transverse[1]:
            raise Unsupported(f"{where}: 가로 배율이 다른 캡슐")
        dims = {"axis": "XYZ"[axis], "radius": f32(c.GetRadiusAttr().Get() * transverse[0]),
                "half_height": f32(0.5 * c.GetHeightAttr().Get() * scale[axis])}
    else:
        raise Unsupported(f"{where}: 형상 종류 {kind} 은 대조하지 않는다")
    return {
        "kind": kind, **dims,
        # difference taken in double, then rounded — rounding the two world positions first
        # would hide a collider offset smaller than one float32 step at the body's position
        "local_pos": [f32(v) for v in (t - body_t)],
        "physics": _physics_properties(shape, "shape", unsupported, where),
        # a density is per volume, so it needs no scale: compared where it is authored, on the shape
        "mass": _physics_properties(shape, "mass", unsupported, where),
        "physics_material": _physics_material(shape),
    }


def describe(path: Path) -> tuple[dict, list[str]]:
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    unsupported: list[str] = []
    bodies, joints, scales, shape_owner = {}, {}, {}, {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        name, where = prim.GetName(), str(prim.GetPath())
        if name in bodies or name in {Path(p).name for p in scales}:
            # Bodies are compared by name; a second one of the same name would overwrite the first
            # and an extra rigid body would pass unseen.
            unsupported.append(f"{where}: 강체 이름 {name} 이 둘 이상 — 이름으로 대조하므로 중복을 받지 않는다")
            continue
        try:
            t, scale = _decompose(cache.GetLocalToWorldTransform(prim), where)
        except Unsupported as e:
            unsupported.append(str(e))
            continue
        scales[where] = scale
        shapes = [c for c in Usd.PrimRange(prim) if c.HasAPI(UsdPhysics.CollisionAPI)]
        if len(shapes) != 1:
            unsupported.append(f"{where}: 형상이 {len(shapes)}개 — 강체당 하나만 대조한다")
            continue
        shape_owner[str(shapes[0].GetPath())] = name
        try:
            shape = _shape(shapes[0], t, str(shapes[0].GetPath()), unsupported)
        except Unsupported as e:
            unsupported.append(str(e))
            continue
        bodies[name] = {
            "pos": [float(np.float32(v)) for v in t],
            "physics": _physics_properties(prim, "body", unsupported, where),
            "keti": {a.GetName(): a.Get() for a in prim.GetAttributes()
                     if a.GetName().startswith("keti:") and a.HasAuthoredValue()},
            "shape": shape,
        }
    body_paths = {p: str(Path(p).name) for p in scales}

    def owner(path: str) -> str | None:
        return body_paths.get(path) or shape_owner.get(path)

    roots, filtered = [], []
    for prim in stage.Traverse():
        where = str(prim.GetPath())
        if "PhysicsArticulationRootAPI" in prim.GetAppliedSchemas():
            roots.append(f"body {body_paths[where]}" if where in body_paths
                         else f"shape of {shape_owner[where]}" if where in shape_owner
                         else f"other prim {where}")
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                a, b = owner(where), owner(str(target))
                if a is None or b is None:
                    unsupported.append(f"{where}: 충돌 제외 쌍 {target} 을 강체 이름으로 옮기지 못함")
                else:
                    filtered.append(tuple(sorted((a, b))))
        if prim.HasAPI(UsdPhysics.MassAPI):
            authored = {a.GetName() for a in prim.GetAttributes()
                        if a.GetName() in MASS_ATTRS and a.HasAuthoredValue()}
            if authored != {DENSITY}:
                unsupported.append(f"{where}: MassAPI 의 {sorted(authored)} — 밀도 하나만 정규화 없이 대조한다 "
                                   "(그 밖의 질량 속성은 저작 프림의 배율 공간이라 정규화가 필요하다)")
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j, where = UsdPhysics.Joint(prim), str(prim.GetPath())
        targets = [r.GetTargets() for r in (j.GetBody0Rel(), j.GetBody1Rel())]
        if any(len(t) != 1 or str(t[0]) not in scales for t in targets):
            unsupported.append(f"{where}: 조인트 양쪽이 대조한 강체가 아님")
            continue
        b0, b1 = (str(t[0]) for t in targets)
        anchors = [(np.array(lp, np.float32) * scales[b].astype(np.float32)).tolist()   # the engine's float32 multiply
                   for lp, b in ((j.GetLocalPos0Attr().Get(), b0), (j.GetLocalPos1Attr().Get(), b1))]
        other = sorted(f"{a.GetName()}={a.Get()}" for a in prim.GetAttributes()
                       if a.GetName().startswith(PHYSICS_PREFIXES) and a.HasAuthoredValue()
                       and not a.GetName().startswith(("physics:localPos", "physics:localRot")))
        joints[where] = {"type": prim.GetTypeName(), "body0": body_paths[b0], "body1": body_paths[b1],
                         "anchor0": anchors[0], "anchor1": anchors[1],
                         "rot0": str(j.GetLocalRot0Attr().Get()), "rot1": str(j.GetLocalRot1Attr().Get()),
                         "other": other}
    units = {k: stage.GetMetadata(k) for k in ("metersPerUnit", "kilogramsPerUnit", "upAxis")}
    return ({"bodies": bodies, "joints": joints, "units": units,
             "articulation_roots": sorted(roots), "filtered_pairs": sorted(set(filtered))},
            [f"{path.name}: {u}" for u in unsupported])


def differences(a: dict, b: dict) -> list[str]:
    out = []
    for key in ("units", "articulation_roots", "filtered_pairs"):
        if a[key] != b[key]:
            out.append(f"{key}: v1 {a[key]} ≠ skin {b[key]}")
    for kind in ("bodies", "joints"):
        for name in sorted(set(a[kind]) | set(b[kind])):
            x, y = a[kind].get(name), b[kind].get(name)
            if x is None or y is None:
                out.append(f"{kind} {name}: 한쪽에만 있음")
                continue
            for field in sorted(set(x) | set(y)):
                if x.get(field) != y.get(field):
                    out.append(f"{kind} {name}.{field}: v1 {x.get(field)} ≠ skin {y.get(field)}")
    return out


def compare_assets(v1: Path, skin: Path) -> list[str]:
    """Every difference and every unsupported input, in the engine's terms. Empty means identical."""
    (a, ua), (b, ub) = describe(v1), describe(skin)
    return [f"지원하지 않는 입력 — {u}" for u in ua + ub] + [f"다름 — {d}" for d in differences(a, b)]


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    canon = json.loads(CANON.read_text(encoding="utf-8"))["scene"]
    skin_entry = canon[SKIN_ID]
    v1 = Path(args.v1) if args.v1 else REPO_ROOT / canon[skin_entry["physics_identical_to"]]["usd_path"]
    skin = Path(args.skin) if args.skin else REPO_ROOT / skin_entry["usd_path"]
    missing = [str(p) for p in (v1, skin) if not p.is_file()]
    if missing:
        print(f"자산이 없습니다: {missing} — build_dummy.py 로 먼저 만드십시오", file=sys.stderr)
        return 2
    problems = compare_assets(v1, skin)
    a, _ = describe(v1)
    print(f"대조: 강체 {len(a['bodies'])} · 고정조인트 {len(a['joints'])} · 아티큘레이션 루트 {a['articulation_roots']}")
    if problems:
        for p in problems:
            print(f"  {p}")
        print(f"물리 기술이 같다고 할 수 없습니다: {len(problems)}건 (정본은 physics_identical_to 로 같다고 적습니다)")
        return 1
    print("물리엔진이 읽는 값으로 v1 과 같습니다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
