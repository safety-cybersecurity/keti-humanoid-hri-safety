#!/usr/bin/env python3
"""Applies a drawn look (``generation.appearance.appearance_edits``) to an opened stage, in memory only.

The edits go to the stage's current edit target; no layer is saved and the referenced USD files (the warehouse, the
dummy, the carried box) are not written, so the built files stay byte for byte as they are. Needs pxr.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pxr import Gf, Sdf, Usd, UsdLux

from generation.appearance import LightEdit, ShaderEdit

# MDL input -> the Sdf type it is created with when the built shader has none.
_INPUT_TYPES = {
    "diffuse_color_constant": Sdf.ValueTypeNames.Color3f,
    "diffuse_tint": Sdf.ValueTypeNames.Color3f,
    "albedo_desaturation": Sdf.ValueTypeNames.Float,
    "emissive_intensity": Sdf.ValueTypeNames.Float,
}
# MDL default of an input that is not authored: what a "scale" multiplies (OmniPBR.mdl: diffuse_tint color(1.0),
# albedo_desaturation 0.0, emissive_intensity 40.0).
_MDL_DEFAULTS = {"diffuse_tint": (1.0, 1.0, 1.0), "albedo_desaturation": 0.0, "emissive_intensity": 40.0}


def _vec(value):
    return Gf.Vec3f(*(float(v) for v in value))


def _scaled(base, factor):
    if isinstance(base, (int, float)):
        return float(base) * float(factor)
    return Gf.Vec3f(*(float(c) * float(factor) for c in base))


def _apply_shader_edit(stage: Usd.Stage, roots: Mapping[str, str], edit: ShaderEdit) -> None:
    if edit.target not in roots:
        raise ValueError(f"고칠 거리의 대상 {edit.target!r} 의 뿌리 경로가 없습니다")
    if edit.name not in _INPUT_TYPES:
        raise ValueError(f"{edit.shader} 의 입력 {edit.name!r} 은 이 모듈이 아는 입력이 아닙니다")
    path = f"{roots[edit.target]}/{edit.shader}"
    prim = stage.GetPrimAtPath(path)
    if not prim or prim.GetTypeName() != "Shader":
        raise ValueError(f"{path}: 셰이더 프림이 열린 장면에 없습니다")
    type_name = _INPUT_TYPES[edit.name]
    attr = prim.GetAttribute(f"inputs:{edit.name}")
    if edit.op == "set":
        new = edit.value
    elif edit.op == "scale":
        base = attr.Get() if attr and attr.HasAuthoredValue() else None
        new = _scaled(_MDL_DEFAULTS[edit.name] if base is None else base, edit.value)
    else:
        raise ValueError(f"{path}: 알 수 없는 고침 {edit.op!r}")
    if not attr:
        attr = prim.CreateAttribute(f"inputs:{edit.name}", type_name)
    attr.Set(_vec(new) if type_name == Sdf.ValueTypeNames.Color3f else float(new))


def _apply_light_edit(stage: Usd.Stage, aisle_root: str, light: LightEdit) -> None:
    root = stage.GetPrimAtPath(aisle_root)
    lights = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdLux.LightAPI)] if root else []
    if not lights:
        raise ValueError(f"{aisle_root}: 창고 아래에 빛이 하나도 없습니다")
    for prim in lights:
        api = UsdLux.LightAPI(prim)
        base = api.GetIntensityAttr().Get()
        api.GetIntensityAttr().Set(float(base) * float(light.scale))
        api.CreateEnableColorTemperatureAttr().Set(True)
        api.CreateColorTemperatureAttr().Set(float(light.temperature_k))


def apply(stage: Usd.Stage, roots: Mapping[str, str], edits: Iterable[ShaderEdit], light: LightEdit) -> None:
    """Writes ``edits`` and ``light`` into ``stage``'s edit target. ``roots`` maps "aisle" | "dummy" | "box" to the
    root prim of that asset in the opened stage. Raises ValueError, before anything is written, when a shader the
    edits name is missing or the warehouse holds no light."""
    edits = list(edits)
    for edit in edits:
        path = f"{roots.get(edit.target, '<' + edit.target + '>')}/{edit.shader}"
        prim = stage.GetPrimAtPath(path)
        if not prim or prim.GetTypeName() != "Shader":
            raise ValueError(f"{path}: 셰이더 프림이 열린 장면에 없습니다")
    if "aisle" not in roots:
        raise ValueError("창고의 뿌리 경로가 없습니다")
    aisle = stage.GetPrimAtPath(roots["aisle"])
    if not aisle or not any(p.HasAPI(UsdLux.LightAPI) for p in Usd.PrimRange(aisle)):
        raise ValueError(f"{roots['aisle']}: 창고 아래에 빛이 하나도 없습니다")
    for edit in edits:
        _apply_shader_edit(stage, roots, edit)
    _apply_light_edit(stage, roots["aisle"], light)
