#!/usr/bin/env python3
"""The look axes of each new scenario and the draw of one row's look.

meta/scenarios.json's ``appearance`` block (S1e, S1f, S1g) is the project's own (self-set) choice of what varies in
the picture: the vest, the shirt, the trousers, the colour of the carried box, the warehouse light (brightness and
colour temperature), the rack uprights and beams, and the brightness of floor and walls. Nothing in it is a standard
value and nothing in it touches the physics.

* ``axes`` -- {name: {"choice": {choice name: [r, g, b] or null}}} for a colour axis, drawn with equal chances, or
  {name: {"range": [lo, hi]}} for a continuous axis, drawn uniformly, in the order they are written. A colour is
  written to the shader input as it stands (the same raw [r, g, b] the scene and dummy builders use); ``null`` is the
  colour as built.

Row i draws every axis from ``numpy.random.default_rng`` seeded with the first 8 bytes (big-endian) of
SHA-256("<scenario>:appearance:<i>") -- a stream of its own, apart from the conditions table's, so a row's physical
conditions do not depend on its look and its look does not depend on the table. A run without rendering draws and
records the same values. Pure numpy, no Isaac.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

APPEARANCE_KEY = "appearance"
APPEARANCE_KEYS = ("determined_by", "axes", "note")
DETERMINED_BY = "project"
# The axes the scene knows how to apply, and what each is.
AXES = ("vest_color", "shirt_color", "trousers_color", "box_color",
        "light_scale", "light_temperature_k", "upright_color", "beam_color", "floor_wall_scale")
COLOR_AXES = ("vest_color", "shirt_color", "trousers_color", "box_color", "upright_color", "beam_color")
RANGE_AXES = ("light_scale", "light_temperature_k", "floor_wall_scale")
# The domain of a range axis: (lowest end, highest end, whether the lowest end is itself allowed).
RANGE_DOMAINS = {
    "light_scale": (0.0, math.inf, False),
    "floor_wall_scale": (0.0, math.inf, False),
    # UsdLux takes a colour temperature from 1000 to 10000 K; the renderer draws a higher value as 10000 K.
    "light_temperature_k": (1000.0, 10000.0, True),
}


@dataclass(frozen=True)
class AppearanceAxis:
    name: str
    choices: Mapping[str, tuple[float, float, float] | None] | None   # colour axes: name -> rgb, None = as built
    lo: float | None                                                    # range axes
    hi: float | None


@dataclass(frozen=True)
class AppearanceSpec:
    scenario_id: str
    axes: tuple[AppearanceAxis, ...]          # in the canon's written order


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _color_axis(sid: str, name: str, node: object) -> AppearanceAxis:
    where = f"{sid} appearance.axes.{name}"
    if not isinstance(node, Mapping) or set(node) != {"choice"}:
        raise ValueError(f"{where}: 색 축은 choice 하나만 가진 객체여야 합니다")
    choice = node["choice"]
    if not isinstance(choice, Mapping) or not choice:
        raise ValueError(f"{where}: choice 는 비지 않은 객체여야 합니다")
    out: dict[str, tuple[float, float, float] | None] = {}
    for label, rgb in choice.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{where}: 색 이름은 비지 않은 글자여야 합니다 ({label!r})")
        if rgb is None:
            out[label] = None
            continue
        if (not isinstance(rgb, (list, tuple)) or len(rgb) != 3
                or not all(_is_number(v) and v >= 0 for v in rgb)):
            raise ValueError(f"{where}: {label!r} 는 null 이거나 유한하고 음수가 아닌 수 셋이어야 합니다 ({rgb!r})")
        out[label] = tuple(float(v) for v in rgb)
    return AppearanceAxis(name, out, None, None)


def _range_axis(sid: str, name: str, node: object) -> AppearanceAxis:
    where = f"{sid} appearance.axes.{name}"
    if not isinstance(node, Mapping) or set(node) != {"range"}:
        raise ValueError(f"{where}: 범위 축은 range 하나만 가진 객체여야 합니다")
    rng = node["range"]
    if not isinstance(rng, (list, tuple)) or len(rng) != 2 or not all(_is_number(v) for v in rng):
        raise ValueError(f"{where}: range 는 유한한 수 둘이어야 합니다 ({rng!r})")
    lo, hi = float(rng[0]), float(rng[1])
    if not lo < hi:
        raise ValueError(f"{where}: range 의 아래({lo})는 위({hi})보다 작아야 합니다")
    floor, ceiling, floor_allowed = RANGE_DOMAINS[name]
    if lo < floor or (lo == floor and not floor_allowed) or hi > ceiling:
        raise ValueError(f"{where}: range [{lo}, {hi}] 가 이 축의 정의역 밖입니다")
    return AppearanceAxis(name, None, lo, hi)


def load_appearance(entry: Mapping) -> AppearanceSpec | None:
    """The checked ``appearance`` block of a scenarios.json entry; None when the entry has none."""
    block = entry.get(APPEARANCE_KEY)
    sid = str(entry.get("scenario_id"))
    if block is None:
        return None
    if "conditions" not in entry:
        # The look is drawn from a conditions-table row; a block on a scenario without a table would be ignored.
        raise ValueError(f"{sid} appearance: 조건표가 없는 시나리오는 겉모습 블록을 두지 못합니다 (행 번호로 뽑습니다)")
    if not isinstance(block, Mapping):
        raise ValueError(f"{sid} appearance: 객체여야 합니다")
    if set(block) != set(APPEARANCE_KEYS):
        raise ValueError(f"{sid} appearance: 키는 {list(APPEARANCE_KEYS)} 여야 합니다 "
                         f"(없는 것 {sorted(set(APPEARANCE_KEYS) - set(block))}, 더한 것 {sorted(set(block) - set(APPEARANCE_KEYS))})")
    if block["determined_by"] != DETERMINED_BY:
        raise ValueError(f"{sid} appearance.determined_by: {DETERMINED_BY!r} 여야 합니다 ({block['determined_by']!r})")
    if not isinstance(block["note"], str) or not block["note"].strip():
        raise ValueError(f"{sid} appearance.note: 비지 않은 글자여야 합니다")
    axes = block["axes"]
    if not isinstance(axes, Mapping):
        raise ValueError(f"{sid} appearance.axes: 객체여야 합니다")
    if set(axes) != set(AXES):
        raise ValueError(f"{sid} appearance.axes: 축은 {list(AXES)} 와 같은 집합이어야 합니다 "
                         f"(없는 것 {sorted(set(AXES) - set(axes))}, 더한 것 {sorted(set(axes) - set(AXES))})")
    out = tuple(_color_axis(sid, n, node) if n in COLOR_AXES else _range_axis(sid, n, node)
                for n, node in axes.items())
    return AppearanceSpec(sid, out)


def appearance_seed(scenario_id: str, row: int) -> int:
    """The seed of a row's look: the first 8 bytes, big-endian, of SHA-256("<scenario>:appearance:<row>")."""
    digest = hashlib.sha256(f"{scenario_id}:appearance:{row}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def draw_appearance(entry: Mapping, row: int) -> dict[str, str | float] | None:
    """Row ``row``'s look: axis -> the colour's name (colour axes) or the value (range axes), in the written order;
    None when the entry has no block."""
    spec = load_appearance(entry)
    if spec is None:
        return None
    rng = np.random.default_rng(appearance_seed(spec.scenario_id, row))
    drawn: dict[str, str | float] = {}
    for axis in spec.axes:
        if axis.choices is not None:
            names = list(axis.choices)
            drawn[axis.name] = names[int(rng.integers(len(names)))]
        else:
            drawn[axis.name] = float(rng.uniform(axis.lo, axis.hi))
    return drawn


@dataclass(frozen=True)
class ShaderEdit:
    """One edit of a material's shader input.

    ``target`` names whose prim tree ``shader`` is under ("aisle", "dummy" or "box"). ``op`` "set" writes ``value``;
    "scale" multiplies the input's authored value (the MDL default, 1, when none is authored) by ``value``."""
    target: str
    shader: str
    name: str
    op: str
    value: float | tuple[float, float, float]


@dataclass(frozen=True)
class LightEdit:
    """Every light under the warehouse: ``inputs:intensity`` times ``scale``, colour temperature on at ``temperature_k``."""
    scale: float
    temperature_k: float


# Names of the dummy's work-clothes materials: Looks/<name> in the built dummy USD (defined by the dummy builder).
SHIRT_MATERIAL = "Human_male_worksuit01_shirt"
TROUSERS_MATERIAL = "Human_male_worksuit01_trousers"

# colour axis -> (target, shader path, ((input, fixed value or None for the drawn colour), ...)) in the order applied.
# A tint only multiplies the texture, so the tinted textured materials are first made grey (albedo_desaturation 1).
_COLOR_EDITS = {
    "vest_color": ("dummy", "Looks/vest_yellow/Shader", (("diffuse_color_constant", None),)),
    "shirt_color": ("dummy", f"Looks/{SHIRT_MATERIAL}/Shader", (("diffuse_tint", None),)),
    "trousers_color": ("dummy", f"Looks/{TROUSERS_MATERIAL}/Shader",
                       (("albedo_desaturation", 1.0), ("diffuse_tint", None))),
    "box_color": ("box", "Looks/cardboard/Shader", (("diffuse_color_constant", None),)),
    "upright_color": ("aisle", "Looks/steel_blue/Shader", (("albedo_desaturation", 1.0), ("diffuse_tint", None))),
    "beam_color": ("aisle", "Looks/steel_orange/Shader", (("albedo_desaturation", 1.0), ("diffuse_tint", None))),
}
_FLOOR_WALL_SHADERS = ("Looks/concrete_floor/Shader", "Looks/concrete_wall/Shader", "Looks/wall_cladding/Shader")
_LENS_SHADER = "Looks/fixture_lens/Shader"


def appearance_edits(spec: AppearanceSpec, drawn: Mapping[str, str | float]) -> tuple[list[ShaderEdit], LightEdit]:
    """The shader edits and the light edit of a draw, in the spec's written axis order, except that the floor/wall and
    lens scale edits (from floor_wall_scale and light_scale) come last. A colour drawn as built
    (``null`` in the canon) edits nothing; the draw must name every axis of the spec with one of its values."""
    if set(drawn) != {a.name for a in spec.axes}:
        missing = sorted({a.name for a in spec.axes} - set(drawn))
        extra = sorted(set(drawn) - {a.name for a in spec.axes})
        raise ValueError(f"{spec.scenario_id} appearance: 뽑은 값의 축이 정본과 다릅니다 (없는 것 {missing}, 더한 것 {extra})")
    edits: list[ShaderEdit] = []
    scales: dict[str, float] = {}
    for axis in spec.axes:
        value = drawn[axis.name]
        if axis.choices is not None:
            if value not in axis.choices:
                raise ValueError(f"{spec.scenario_id} appearance.{axis.name}: {value!r} 는 정본의 색 이름이 아닙니다")
            rgb = axis.choices[value]
            if rgb is None:
                continue
            target, shader, inputs = _COLOR_EDITS[axis.name]
            edits += [ShaderEdit(target, shader, name, "set", rgb if fixed is None else fixed)
                      for name, fixed in inputs]
        else:
            if isinstance(value, bool) or not _is_number(value) or not axis.lo <= value <= axis.hi:
                raise ValueError(f"{spec.scenario_id} appearance.{axis.name}: {value!r} 는 범위 "
                                 f"[{axis.lo}, {axis.hi}] 안의 수가 아닙니다")
            scales[axis.name] = float(value)
    edits += [ShaderEdit("aisle", s, "diffuse_tint", "scale", scales["floor_wall_scale"]) for s in _FLOOR_WALL_SHADERS]
    edits.append(ShaderEdit("aisle", _LENS_SHADER, "emissive_intensity", "scale", scales["light_scale"]))
    return edits, LightEdit(scales["light_scale"], scales["light_temperature_k"])


_APPLY_CLAIMED = False


def claim_single_apply() -> None:
    """Called right before a look is written into the opened stage. The write compounds (scales multiply the
    values already there), so a process applies it once; a second claim raises."""
    global _APPLY_CLAIMED
    if _APPLY_CLAIMED:
        raise RuntimeError("겉모습은 한 프로세스에서 한 번만 입힙니다 — 두 번 입히면 배율이 겹칩니다")
    _APPLY_CLAIMED = True
