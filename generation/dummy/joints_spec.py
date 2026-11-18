"""The jointed dummy's spec (generation/dummy/joints.yaml): read, checked and derived.

Pure -- numpy and yaml, no pxr. The v2 build, the gait build and the environment read the dummy's
joints, feet, masses and gains through here, so a pivot or a range is derived in one place. Pivots come
from anthropometry.yaml's capsules, feet from the stature, densities from segment fractions."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from generation.dummy.gait.kinematics import LegGeometry
from generation.dummy.skin import region_split as rs

HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "joints.yaml"
ANTHRO_PATH = HERE / "anthropometry.yaml"
FOOT = "FOOT"
SIDES = ("left", "right")
GROUPS = ("hip", "knee", "ankle", "shoulder", "elbow")
DRIVE_MODES = ("walking", "tone")
LEG_CODES = ("BR-11", "BR-12")


@dataclass(frozen=True)
class Dof:
    axis: str
    lower: float          # rad
    upper: float          # rad


@dataclass(frozen=True)
class JointDef:
    name: str             # "hip_left"
    group: str            # "hip"
    side: str
    parent: str           # parent body prim, "BR_07"
    child: str            # child body prim, "BR_11_left" / "FOOT_left"
    pivot: tuple[float, float, float]
    dofs: tuple[Dof, ...]

    @property
    def kind(self) -> str:
        return "revolute" if len(self.dofs) == 1 else "d6"


@dataclass(frozen=True)
class FootBox:
    name: str
    side: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class JointedSpec:
    path: Path
    anthropometry: dict
    joints: tuple[JointDef, ...]
    feet: tuple[FootBox, ...]
    density: dict
    mass: dict
    total_mass: float
    drives: dict
    d6_convention: str
    dof_name: str
    semantic_class_feet: str


def _sign(side: str) -> float:
    return 1.0 if side == "left" else -1.0


def capsule_ends(region: dict, sign: float) -> tuple[np.ndarray, np.ndarray]:
    """(upper, lower) ends of a z capsule's axis segment -- the centres of its end caps."""
    if region.get("shape") != "capsule" or str(region.get("axis", "z")).lower() != "z":
        raise ValueError("관절 피벗은 z 축 캡슐에서만 유도합니다")
    c = np.array(region["center"], float) * np.array([1.0, sign, 1.0])
    half = np.array([0.0, 0.0, float(region["height"]) / 2.0])
    return c + half, c - half


def region_volume(region: dict) -> float:
    shape = region["shape"]
    if shape == "sphere":
        return 4.0 / 3.0 * math.pi * float(region["radius"]) ** 3
    if shape == "capsule":
        r, h = float(region["radius"]), float(region["height"])
        return math.pi * r * r * h + 4.0 / 3.0 * math.pi * r ** 3
    if shape == "box":
        sx, sy, sz = (float(v) for v in region["size"])
        return sx * sy * sz
    raise ValueError(f"알 수 없는 형상: {shape!r}")


def mirror_range(axis: str, lo: float, hi: float) -> tuple[float, float]:
    return (-hi, -lo) if axis in "xz" else (lo, hi)


def prim_of(code: str, side: str, regions: dict) -> str:
    """The body prim a code names on one side: that side's prim if the region is mirrored, else its one prim."""
    if code == FOOT:
        return f"{FOOT}_{side}"
    if code not in regions:
        raise ValueError(f"anthropometry.yaml 에 없는 부위: {code}")
    names = [n for n, _ in rs.prim_names(code, regions[code])]
    return names[0] if len(names) == 1 else next(n for n in names if n.endswith("_" + side))


def _pivot(rule, side: str, regions: dict) -> np.ndarray:
    def ends(code):
        reg = regions[code]
        return capsule_ends(reg, _sign(side) if reg.get("mirror_y") else 1.0)

    if not isinstance(rule, dict) or len(rule) != 1:
        raise ValueError(f"피벗 규칙은 top · bottom · between 중 하나입니다: {rule!r}")
    (kind, arg), = rule.items()
    if kind == "top":
        return ends(arg)[0]
    if kind == "bottom":
        return ends(arg)[1]
    if kind == "between" and isinstance(arg, list) and len(arg) == 2:
        return (ends(arg[0])[1] + ends(arg[1])[0]) / 2.0
    raise ValueError(f"피벗 규칙을 모릅니다: {rule!r}")


def _joints(raw: dict, regions: dict) -> tuple[JointDef, ...]:
    out = []
    for group, j in raw.items():
        if group not in GROUPS:
            raise ValueError(f"모르는 관절 묶음: {group}")
        dofs_raw = j.get("dofs") or {}
        if not dofs_raw or any(a not in "xyz" or len(a) != 1 for a in dofs_raw):
            raise ValueError(f"{group}: dofs 는 x · y · z 의 부분집합입니다")
        if len(dofs_raw) == 2:
            # A D6 with one rotation locked turns about the locked axis too: its two positions follow no
            # convention (tests/generation/test_d6_convention.py). Hold an unwanted axis with a drive instead.
            raise ValueError(f"{group}: 여러 축 관절은 세 축을 모두 엽니다 — 두 축만 연 관절은 엔진이 "
                             "관절 위치를 어느 규약으로도 읽지 않습니다. 쓰지 않는 축은 좁은 범위로 두고 구동으로 0 에 붙잡으십시오")
        for side in SIDES:
            dofs = []
            for axis in "xyz":
                if axis not in dofs_raw:
                    continue
                lo, hi = (math.radians(float(v)) for v in dofs_raw[axis])
                if side == "right":
                    lo, hi = mirror_range(axis, lo, hi)
                if not lo < hi:
                    raise ValueError(f"{group}.{axis}: 범위의 아래가 위보다 작아야 합니다")
                dofs.append(Dof(axis, lo, hi))
            out.append(JointDef(
                name=f"{group}_{side}", group=group, side=side,
                parent=prim_of(j["parent"], side, regions), child=prim_of(j["child"], side, regions),
                pivot=tuple(float(v) for v in _pivot(j["pivot"], side, regions)), dofs=tuple(dofs)))
    order = {g: i for i, g in enumerate(GROUPS)}
    return tuple(sorted(out, key=lambda jd: (order[jd.group], SIDES.index(jd.side))))


def _feet(raw: dict, stature: float, joints) -> tuple[FootBox, ...]:
    L = float(raw["length_per_stature"]) * stature
    W = float(raw["width_per_stature"]) * stature
    H = float(raw["height_per_stature"]) * stature
    heel = float(raw["heel_fraction"])
    out = []
    for j in joints:
        if j.group != "ankle":
            continue
        ax, ay, az = j.pivot
        if not H < az:
            raise ValueError("발 상자의 윗면이 발목 피벗보다 높습니다")
        out.append(FootBox(j.child, j.side, (ax + (0.5 - heel) * L, ay, H / 2.0), (L, W, H)))
    return tuple(out)


def _masses(raw: dict, regions: dict, feet) -> tuple[dict, dict, float]:
    total = float(raw["total_kg"])
    mass, density, percent = {}, {}, 0.0
    for gname, g in raw["groups"].items():
        pct = float(g["percent"])
        per_side = bool(g.get("per_side", False))
        percent += pct * (2 if per_side else 1)
        for side in (SIDES if per_side else (None,)):
            bodies = {}
            for code in g["regions"]:
                if code == FOOT:
                    f = next(f for f in feet if f.side == side)
                    bodies[f.name] = f.size[0] * f.size[1] * f.size[2]
                    continue
                for name, _ in rs.prim_names(code, regions[code]):
                    if side is None or name.endswith("_" + side):
                        bodies[name] = region_volume(regions[code])
            rho = total * pct / 100.0 / sum(bodies.values())
            for name, vol in bodies.items():
                if name in density:
                    raise ValueError(f"{name} 이 두 질량 묶음에 들어 있습니다")
                density[name], mass[name] = rho, rho * vol
    if abs(percent - 100.0) > 0.01:
        raise ValueError(f"질량 백분율의 합이 100 이 아닙니다: {percent:.2f}")
    expected = {n for code, reg in regions.items() for n, _ in rs.prim_names(code, reg)} | {f.name for f in feet}
    if set(density) != expected:
        raise ValueError(f"질량이 없는 몸 또는 모르는 몸: {sorted(set(density) ^ expected)}")
    return mass, density, total


def _drives(raw: dict) -> dict:
    out = {}
    for mode in DRIVE_MODES:
        groups = (raw or {}).get(mode) or {}
        if set(groups) != set(GROUPS):
            raise ValueError(f"drives.{mode} 는 관절 묶음 다섯을 모두 적어야 합니다")
        out[mode] = {}
        for g, v in groups.items():
            k, d = float(v["stiffness"]), float(v["damping"])
            if not (k > 0 and d > 0 and math.isfinite(k) and math.isfinite(d)):
                raise ValueError(f"drives.{mode}.{g}: 강성과 감쇠는 양의 유한값입니다")
            out[mode][g] = (k, d)
    return out


def load(spec_path: Path = SPEC_PATH, anthro_path: Path = ANTHRO_PATH) -> JointedSpec:
    raw = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    anthro = yaml.safe_load(Path(anthro_path).read_text(encoding="utf-8"))
    regions = anthro["regions"]
    joints = _joints(raw["joints"], regions)
    feet = _feet(raw["feet"], float(anthro["stature"]), joints)
    mass, density, total = _masses(raw["mass"], regions, feet)
    isaac = raw.get("isaac") or {}
    return JointedSpec(path=Path(spec_path), anthropometry=anthro, joints=joints, feet=feet,
                       density=density, mass=mass, total_mass=total, drives=_drives(raw.get("drives")),
                       d6_convention=str(isaac["d6_convention"]), dof_name=str(isaac["dof_name"]),
                       semantic_class_feet=str(raw["feet"]["semantic_class"]))


def joint(spec: JointedSpec, name: str) -> JointDef:
    for j in spec.joints:
        if j.name == name:
            return j
    raise KeyError(name)


def axes(j: JointDef) -> str:
    return "".join(d.axis for d in j.dofs)


def columns(spec: JointedSpec) -> dict:
    """{joint name: slice} into a row of every joint's positions in spec order."""
    out, k = {}, 0
    for j in spec.joints:
        out[j.name] = slice(k, k + len(j.dofs))
        k += len(j.dofs)
    return out


def dof_names(spec: JointedSpec) -> list[str]:
    """The names the engine gives the degrees of freedom, in spec order: a revolute joint's own name, a
    D6 joint's axes by the measured pattern."""
    out = []
    for j in spec.joints:
        if j.kind == "revolute":
            out.append(j.name)
        else:
            out += [spec.dof_name.format(joint=j.name, axis=d.axis, index=i) for i, d in enumerate(j.dofs)]
    return out


def geometry_with_feet(spec: JointedSpec) -> dict:
    """anthropometry.yaml's dict with the feet as one more mirrored box (key FOOT), for helpers that walk
    every region's geometry -- the aisle guard's reach and standing height."""
    d = copy.deepcopy(spec.anthropometry)
    left = next(f for f in spec.feet if f.side == "left")
    d["regions"][FOOT] = {"shape": "box", "size": list(left.size), "center": list(left.center),
                          "mirror_y": True, "parent": "BR-12"}
    return d


def foot_primitives(spec: JointedSpec) -> list:
    return [rs.Primitive(spec.semantic_class_feet, "box", np.array(f.center, float), None, None, None,
                         np.array(f.size, float), name=f.name, side=f.side) for f in spec.feet]


def leg_geometry(spec: JointedSpec, side: str) -> LegGeometry:
    foot = next(f for f in spec.feet if f.side == side)
    cx, cy, _ = foot.center
    L, W, _ = foot.size
    sole = np.array([[cx + sx * L / 2, cy + sy * W / 2, 0.0] for sx in (1, -1) for sy in (1, -1)])
    piv = {g: np.array(joint(spec, f"{g}_{side}").pivot) for g in ("hip", "knee", "ankle")}
    return LegGeometry(piv["hip"], piv["knee"], piv["ankle"], sole)


def region_prims(spec: JointedSpec) -> tuple[str, ...]:
    regions = spec.anthropometry["regions"]
    return tuple(n for code, reg in regions.items() for n, _ in rs.prim_names(code, reg))


def upper_body_prims(spec: JointedSpec) -> tuple[str, ...]:
    regions = spec.anthropometry["regions"]
    return tuple(n for code, reg in regions.items() if code not in LEG_CODES for n, _ in rs.prim_names(code, reg))


def hip_above_root(spec: JointedSpec) -> float:
    """The hip pivots' height relative to the root region's centre (negative: the hips are below it) --
    the root link is the region without a parent, at its centre (build_dummy.py)."""
    roots = [r for r in spec.anthropometry["regions"].values() if r.get("parent") is None]
    if len(roots) != 1:
        raise ValueError("parent 가 null 인 부위는 정확히 하나여야 합니다")
    return float(joint(spec, "hip_left").pivot[2]) - float(roots[0]["center"][2])
