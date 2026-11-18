"""Logistics-aisle scene layout: spec (yaml) -> list of boxes.

Pure numpy so CI can test it without pxr. build_logistics_scene.py turns the
boxes into USD prims; nothing here knows about USD.

Frame: the aisle runs along +x, y=0 is the aisle centre line, z up, the floor
at z=0 -- the same frame S1a's robot walks in, so the scene drops in without
an offset. Box names are paths ("rack/upright/sp0/b3k1"); the builder maps
them to prim names.

Only the racks, the boxes on them and the loads standing on the bottom tier
collide. Everything else -- floor, ceiling, the walls closing the floor's
edges, dock doors, roof trusses, light fixtures and floor lines -- is
dressing the cameras see and the physics does not, and it comes after
the racks and boxes so the collider prims keep their order.

The box labels (box_labels: a white card and the KETI CI on the two y faces of
every shelf box and every bottom-tier carton) are not boxes of the list: the
list is what the colliders, the occlusion geometry, the aisle guard and the
measuring tools read, and a label is none of those -- a plate a millimetre or
two off its box's face, with no collider. The builder writes them as two meshes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REQUIRED_SECTIONS = ("aisle", "floor", "ceiling", "walls", "roof", "lights", "floor_markings",
                     "rack", "boxes", "labels", "materials")

# The spec the S1d hall is built from, absolute. The S1d environment (s1d_env_cfg) and its camera rig
# (camera_io, torch-free: the fixed camera's recorded pose) both read it from here, so the fixed camera's
# prim and its recorded pose come from one spec. build_logistics_scene.py spells the same path itself.
SCENE_SPEC = Path(__file__).resolve().with_name("logistics_aisle.yaml")

# Every semantic class a scene prim can carry. Racks and the
# boxes on them are one class: a detector's "not a person" is what matters.
SEMANTIC_RACK = "rack"

# The bottom tier's loads (rack.floor_loads) are named rack/load/<row tag>/b<bay>/<slot>/{pallet,carton}: the
# rack/ prefix puts them in every rack check, and they carry the rack's class and collide like it.
FLOOR_LOAD_PREFIX = "rack/load/"

# Modelling constant: the least air a load leaves under the first beam, so a load never touches it.
FLOOR_LOAD_HEADROOM_M = 0.15

# Modelling constants, not scene dimensions: the floor and ceiling are thin
# slabs, and a fixture's light sits this far under it so the fixture cannot
# shade it (a RectLight emits along its -Z).
SLAB = 0.001
LIGHT_GAP = 0.005

TEXTURED_KEYS = {"texture", "tile_m", "tint"}
PLAIN_KEYS = {"color", "roughness", "metallic", "emissive_intensity"}
# A picture material: a picture its own pin fixes, on a plate's UVs (labels only -- a cube has no UVs).
PICTURE_KEYS = {"picture", "roughness"}
MATERIAL_KINDS = ("texture", "color", "picture")

# The box labels (spec labels): on which boxes, which faces, and the two plates of a label -- the white card,
# then the CI on it (the carried box's PLATE_KINDS). A face is named by its outward normal; a plate's height
# runs along z and its width along LABEL_UP x normal -- the viewer's right when facing that face.
LABEL_BOXES = ("shelf_box", "floor_carton")
LABEL_FACES = {"+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0)}
LABEL_UP = (0.0, 0.0, 1.0)
LABEL_PLATES = ("base", "ci")
PLATE_NAMES = {"base": "white card", "ci": "CI"}       # what a refusal calls each plate
LABEL_KEYS = {"material", "ci_material", "lift_m", "ci_lift_m", *LABEL_BOXES}
LABEL_BOX_KEYS = {"size", "ci_width"}


@dataclass(frozen=True)
class Box:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    material: str
    collider: bool
    semantic: str | None
    # Turned about x after scaling (a truss's web members, in the y-z plane). 0 = axis-aligned.
    rot_x_deg: float = 0.0


def _walk_numbers(node, path=""):
    """Yield (path, value) for every number in a nested mapping/list."""
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        yield path, float(node)
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _walk_numbers(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            yield from _walk_numbers(v, f"{path}[{i}]")


def _pair(spec_value, name: str) -> tuple[int, int]:
    lo, hi = spec_value
    if not (isinstance(lo, int) and isinstance(hi, int)) or lo < 1 or hi < lo:
        raise ValueError(f"{name} must be [lo, hi] integers with 1 <= lo <= hi, got {spec_value!r}")
    return lo, hi


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _color(value, what: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(_number(v) and v >= 0 for v in value):
        raise ValueError(f"{what} must be three non-negative numbers")


def _lengths(value, n: int, what: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != n or not all(_number(v) and v > 0 for v in value):
        raise ValueError(f"{what} must be {n} positive lengths")


def material_kind(m: dict) -> str:
    """"texture", "color" or "picture": the one of MATERIAL_KINDS the (validated) material has."""
    return next(k for k in MATERIAL_KINDS if k in m)


def _validate_material(name: str, m) -> None:
    """A material is textured (a textures.lock set), plain (a colour) or a picture (one its own pin
    fixes) -- exactly one of them: a plain one carries no file, so it needs no licence row."""
    if not isinstance(m, dict) or sum(k in m for k in MATERIAL_KINDS) != 1:
        raise ValueError(f"materials.{name} must have exactly one of texture (a textures.lock set), "
                         "color (a plain colour) or picture (a pinned picture)")
    if "picture" in m:
        if not isinstance(m["picture"], str) or not m["picture"].strip():
            raise ValueError(f"materials.{name}.picture must name a pinned picture")
        if not (_number(m.get("roughness")) and 0 <= m["roughness"] <= 1):
            raise ValueError(f"materials.{name}.roughness must lie in [0, 1]")
        unknown = set(m) - PICTURE_KEYS
    elif "texture" in m:
        if not isinstance(m["texture"], str) or not m["texture"].strip():
            raise ValueError(f"materials.{name}.texture must name a textures.lock set")
        if not _number(m.get("tile_m")) or m["tile_m"] <= 0:
            raise ValueError(f"materials.{name}.tile_m must be positive")
        if m.get("tint") is not None:
            _color(m["tint"], f"materials.{name}.tint")
        unknown = set(m) - TEXTURED_KEYS
    else:
        _color(m["color"], f"materials.{name}.color")
        for key in ("roughness", "metallic"):
            if key in m and not (_number(m[key]) and 0 <= m[key] <= 1):
                raise ValueError(f"materials.{name}.{key} must lie in [0, 1]")
        if "emissive_intensity" in m and not (_number(m["emissive_intensity"]) and m["emissive_intensity"] > 0):
            raise ValueError(f"materials.{name}.emissive_intensity must be positive")
        unknown = set(m) - PLAIN_KEYS
    if unknown:
        raise ValueError(f"materials.{name}: unknown keys {sorted(unknown)}")


def validate_spec(spec: dict) -> dict:
    """Refuse a spec whose geometry cannot be built.

    Every rule here is one a wrong value would otherwise turn into an invalid
    collider: a rack thinner than its own uprights flips the faces into the
    aisle, beams closer than beam + shelf overlap, a NaN anywhere propagates
    into nonfinite centres for every collider. Rejecting is cheaper than
    debugging a scene.

    The built layout has the last word: a spec every
    rule lets through is still refused when its boxes break validate_layout,
    so the rules and the geometry cannot disagree. A missing key or a value of
    the wrong type is a ValueError naming it, like every other refusal.
    """
    try:
        _validate_spec(spec)
        label_fit(spec, build_layout(spec))
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"scene spec: a key is missing or a value has the wrong shape — {exc!r}") from exc
    return spec


def _validate_spec(spec: dict) -> None:
    if not isinstance(spec, dict):
        raise ValueError("scene spec is not a mapping")
    missing = [k for k in REQUIRED_SECTIONS if k not in spec]
    if missing:
        raise ValueError(f"scene spec is missing {missing} (aisle, floor, ... see layout.REQUIRED_SECTIONS)")
    for path, v in _walk_numbers({k: spec[k] for k in REQUIRED_SECTIONS}):
        if not math.isfinite(v):
            raise ValueError(f"{path}: every number must be finite, got {v!r}")
    a, r, b, f, c, lg = (spec[k] for k in ("aisle", "rack", "boxes", "floor", "ceiling", "lights"))
    if a["half_width"] <= 0:
        raise ValueError("aisle.half_width must be positive")
    if "guard_margin_m" not in a or a["guard_margin_m"] <= 0:
        raise ValueError("aisle.guard_margin_m must be a positive margin (the aisle guard's, design 3.3)")
    if a["x_max"] <= a["x_min"]:
        raise ValueError("aisle.x_max must be greater than aisle.x_min")
    for key in ("bay_width", "depth", "upright_size", "beam_size", "shelf_thickness"):
        if r[key] <= 0:
            raise ValueError(f"rack.{key} must be positive")
    if r["row_gap"] < 0:
        raise ValueError("rack.row_gap must not be negative")
    if not isinstance(r["rows_per_side"], int) or r["rows_per_side"] < 1:
        raise ValueError("rack.rows_per_side must be an integer >= 1")
    member = max(r["upright_size"], r["beam_size"])
    if r["depth"] <= 2 * member:
        raise ValueError(f"rack.depth must exceed twice the widest member ({member}); "
                         "thinner racks put the faces inside each other")
    if r["bay_width"] <= r["upright_size"]:
        raise ValueError("rack.bay_width must exceed rack.upright_size")
    hs = list(r["beam_heights"])
    if not hs or hs[0] <= 0:
        raise ValueError("rack.beam_heights must start above the floor")
    pitch = r["beam_size"] + r["shelf_thickness"]
    for lo, hi in zip(hs, hs[1:]):
        if hi - lo <= pitch:
            raise ValueError(f"rack.beam_heights must be strictly increasing by more than beam + shelf ({pitch}); "
                             f"{lo} -> {hi} overlaps")
    if c["z"] <= hs[-1] + r["beam_size"]:
        raise ValueError("ceiling.z must be above the top beam")
    if f["margin"] < 0:
        raise ValueError("floor.margin must not be negative")
    if f["z"] - SLAB / 2 < 0:
        raise ValueError("floor.z puts the floor slab under the ground plane (z = 0) the walls stand on")
    lo, hi = b["size_min"], b["size_max"]
    if len(lo) != 3 or len(hi) != 3 or any(v <= 0 for v in lo + hi):
        raise ValueError("boxes.size_min / size_max must be three positive lengths")
    if any(l_ > h_ for l_, h_ in zip(lo, hi)):
        raise ValueError("boxes.size_min must not exceed boxes.size_max")
    _pair(b["stacks_per_bay"], "boxes.stacks_per_bay")
    _pair(b["layers"], "boxes.layers")
    if b["edge_margin"] < 0 or r["bay_width"] - r["upright_size"] - 2 * b["edge_margin"] <= 0:
        raise ValueError("boxes.edge_margin leaves no usable width in a bay")
    if b["top_shelf_clearance"] <= 0:
        raise ValueError("boxes.top_shelf_clearance must be positive")
    _validate_floor_loads(spec)
    _validate_dressing(spec)
    w, roof, m = spec["walls"], spec["roof"], spec["floor_markings"]
    used = [f["material"], c["material"], b["material"], *r["materials"].values(), w["material"],
            roof["material"], lg["fixture_material"], m["material"]]
    if r["floor_loads"]["per_bay"] > 0:
        used += [r["floor_loads"]["pallet"]["material"], r["floor_loads"]["carton"]["material"]]
    # a part that is not built does not need its material (base_height 0, no doors)
    if w["base_height"] > 0:
        used.append(w["base_material"])
    if w["dock_doors"]["per_end_wall"] > 0:
        used.append(w["dock_doors"]["material"])
    for mat in used:
        if mat not in spec["materials"]:
            raise ValueError(f"material {mat!r} is used but not declared under 'materials'")
        _validate_material(mat, spec["materials"][mat])
        if "picture" in spec["materials"][mat]:
            raise ValueError(f"material {mat!r} is a picture, which needs a plate's UVs — a box has none")
    _validate_labels(spec)


def _validate_labels(spec: dict) -> None:
    """labels: a plain card and a picture on it, lifted off the face in that order, each label no larger than
    the smallest face it goes on -- a shelf box's y face is boxes.size_min's x by z at the least, a carton's
    is its own x by z -- and its CI no wider than the card. The built boxes are checked too (label_fit), since
    a bay too narrow for size_min shrinks a box below it."""
    lb, b = spec["labels"], spec["boxes"]
    if not isinstance(lb, dict):
        raise ValueError("labels must be a mapping (material, ci_material, lift_m, ci_lift_m, shelf_box, floor_carton)")
    unknown = set(lb) - LABEL_KEYS
    if unknown:
        raise ValueError(f"labels: unknown keys {sorted(unknown)}")
    for key, kind in (("material", "color"), ("ci_material", "picture")):
        mat = lb[key]
        if not isinstance(mat, str) or mat not in spec["materials"]:
            raise ValueError(f"labels.{key} {mat!r} is not declared under 'materials'")
        _validate_material(mat, spec["materials"][mat])
        if material_kind(spec["materials"][mat]) != kind:
            raise ValueError(f"labels.{key} must be a {kind} material, {mat!r} is not")
    if not (_number(lb["lift_m"]) and lb["lift_m"] > 0):
        raise ValueError(f"labels.lift_m must be positive — a card on its face would share it, got {lb['lift_m']!r}")
    if not (_number(lb["ci_lift_m"]) and lb["ci_lift_m"] > lb["lift_m"]):
        raise ValueError(f"labels.ci_lift_m must exceed labels.lift_m — the CI lies on the card, got {lb['ci_lift_m']!r}")
    faces = {"shelf_box": (b["size_min"][0], b["size_min"][2]),
             "floor_carton": (spec["rack"]["floor_loads"]["carton"]["size"][0],
                              spec["rack"]["floor_loads"]["carton"]["size"][2])}
    for which in LABEL_BOXES:
        one = lb[which]
        if not isinstance(one, dict) or set(one) != LABEL_BOX_KEYS:
            raise ValueError(f"labels.{which} must be a mapping of exactly {sorted(LABEL_BOX_KEYS)}")
        _lengths(one["size"], 2, f"labels.{which}.size (width, height)")
        if not (_number(one["ci_width"]) and 0 < one["ci_width"] <= one["size"][0]):
            raise ValueError(f"labels.{which}.ci_width must be positive and no wider than the card "
                             f"({one['size'][0]} m), got {one['ci_width']!r}")
        if which == "floor_carton" and spec["rack"]["floor_loads"]["per_bay"] == 0:
            continue
        room = faces[which]
        if one["size"][0] > room[0] + 1e-9 or one["size"][1] > room[1] + 1e-9:
            raise ValueError(f"labels.{which}.size {list(one['size'])} does not fit the smallest face it goes on "
                             f"({room[0]} x {room[1]} m)")


def _validate_floor_loads(spec: dict) -> None:
    """rack.floor_loads: whole loads that fit between a bay's uprights, inside the row's depth and
    under the first beam with FLOOR_LOAD_HEADROOM_M to spare. 0 per bay switches them off."""
    r = spec["rack"]
    fl = r["floor_loads"]
    if not isinstance(fl, dict):
        raise ValueError("rack.floor_loads must be a mapping (per_bay, gap_m, pallet, carton)")
    per_bay, gap = fl["per_bay"], fl["gap_m"]
    if isinstance(per_bay, bool) or not isinstance(per_bay, int) or per_bay < 0:
        raise ValueError(f"rack.floor_loads.per_bay must be an integer >= 0, got {per_bay!r}")
    if not _number(gap) or gap < 0:
        raise ValueError(f"rack.floor_loads.gap_m must not be negative, got {gap!r}")
    for part in ("pallet", "carton"):
        spec_part = fl[part]
        if not isinstance(spec_part, dict):
            raise ValueError(f"rack.floor_loads.{part} must be a mapping (size, material)")
        _lengths(spec_part["size"], 3, f"rack.floor_loads.{part}.size")
        if not isinstance(spec_part["material"], str) or not spec_part["material"].strip():
            raise ValueError(f"rack.floor_loads.{part}.material must name a material under 'materials'")
        if per_bay > 0 and spec_part["material"] not in spec["materials"]:
            raise ValueError(f"rack.floor_loads.{part}.material {spec_part['material']!r} is not declared "
                             "under 'materials'")
    if per_bay == 0:
        return
    pallet, carton = fl["pallet"]["size"], fl["carton"]["size"]
    width = max(pallet[0], carton[0])
    room = r["bay_width"] - r["upright_size"]
    if per_bay * width + (per_bay - 1) * gap > room + 1e-9:
        raise ValueError(f"rack.floor_loads: {per_bay} loads of {width} m with {gap} m between them do not fit "
                         f"between a bay's uprights ({room:.3f} m)")
    if max(pallet[1], carton[1]) > r["depth"] + 1e-9:
        raise ValueError(f"rack.floor_loads: a load {max(pallet[1], carton[1])} m deep does not fit the row's "
                         f"{r['depth']} m depth")
    stacked = pallet[2] + carton[2] + FLOOR_LOAD_HEADROOM_M
    if stacked > min(r["beam_heights"]) + 1e-9:
        raise ValueError(f"rack.floor_loads: pallet + carton + {FLOOR_LOAD_HEADROOM_M} m of air is {stacked:.3f} m, "
                         f"more than the first beam's {min(r['beam_heights'])} m")


def _validate_dressing(spec: dict) -> None:
    """The enclosure's rules (2026-09-29): the racks stand inside the walls, the walls reach
    from the floor to the ceiling, the doors face the main aisle and stay short of the racks'
    ends, the roof has at least one truss with members that fit their truss, the trusses hang
    between the fixtures and the ceiling, the fixtures between the racks' top and the trusses,
    between two trusses and within their aisle, the floor lines inside the main aisle."""
    a, c, lg = spec["aisle"], spec["ceiling"], spec["lights"]
    w, roof, m = spec["walls"], spec["roof"], spec["floor_markings"]
    d = w["dock_doors"]
    x0, x1, _ = building_extent(spec)
    rx0, rx1 = rack_x_extent(spec)
    if rx0 < x0 or rx1 > x1:
        raise ValueError(f"floor.margin leaves the racks' ends (x {rx0:.3f}..{rx1:.3f}) outside the walls "
                         f"(x {x0}..{x1})")
    if w["thickness"] <= 0:
        raise ValueError("walls.thickness must be positive")
    if not 0 <= w["base_height"] < c["z"]:
        raise ValueError("walls.base_height must lie between the floor and the ceiling")
    if not isinstance(d["per_end_wall"], int) or isinstance(d["per_end_wall"], bool) or d["per_end_wall"] < 0:
        raise ValueError("walls.dock_doors.per_end_wall must be an integer >= 0")
    for key in ("width", "height", "thickness"):
        if d[key] <= 0:
            raise ValueError(f"walls.dock_doors.{key} must be positive")
    if d["gap"] < 0:
        raise ValueError("walls.dock_doors.gap must not be negative")
    if d["height"] >= c["z"]:
        raise ValueError("walls.dock_doors.height must stay below the ceiling")
    n = d["per_end_wall"]
    if n * d["width"] + max(n - 1, 0) * d["gap"] > 2 * a["half_width"]:
        raise ValueError(f"walls.dock_doors: {n} doors side by side are wider than the "
                         f"{2 * a['half_width']} m main aisle they face")
    if d["thickness"] >= min(rx0 - x0, x1 - rx1):
        raise ValueError(f"walls.dock_doors.thickness reaches past the gap between an end wall and the racks' ends "
                         f"({min(rx0 - x0, x1 - rx1):.3f} m)")
    for key in ("truss_spacing_m", "chord_size", "web_size", "panel_m", "purlin_spacing_m"):
        if roof[key] <= 0:
            raise ValueError(f"roof.{key} must be positive")
    _lengths(roof["purlin_size"], 2, "roof.purlin_size (width, height)")
    if roof["truss_depth"] <= roof["chord_size"]:
        raise ValueError("roof.truss_depth must exceed roof.chord_size")
    if _truss_bays(spec) < 2:
        raise ValueError("roof.truss_spacing_m leaves the hall one bay long — no truss to carry the roof")
    if roof["web_size"] > roof["chord_size"]:
        raise ValueError("roof.web_size must not exceed roof.chord_size — a thicker web member pokes out of its truss")
    if roof["panel_m"] <= roof["web_size"]:
        raise ValueError("roof.panel_m must exceed roof.web_size — a panel narrower than its member holds none")
    if roof["purlin_spacing_m"] <= roof["purlin_size"][0]:
        raise ValueError("roof.purlin_spacing_m must exceed the purlin width — the purlins would overlap")
    _lengths(lg["fixture_size"], 3, "lights.fixture_size")
    fx, fy, fz = lg["fixture_size"]
    narrowest = min(hi - lo for lo, hi in aisle_strips(spec))
    if fy > narrowest:
        raise ValueError(f"lights.fixture_size: a {fy} m wide fixture is wider than the narrowest aisle ({narrowest} m)")
    top = rack_top(spec)
    if lg["z"] - fz / 2 <= top:
        raise ValueError(f"lights.z puts the fixtures at or below the racks' top ({top} m)")
    if roof_levels(spec)["truss_bottom"] <= lg["z"] + fz / 2:
        raise ValueError("roof: the trusses come down to the fixtures — raise the ceiling, "
                         "lower lights.z or make roof.truss_depth shallower")
    if fx >= truss_pitch(spec) - roof["chord_size"]:
        raise ValueError(f"lights.fixture_size: a {fx} m fixture does not fit between two trusses "
                         f"({truss_pitch(spec):.3f} m apart)")
    if lg["intensity"] <= 0:
        raise ValueError("lights.intensity must be positive")
    _color(lg["color"], "lights.color")
    if lg["fill"]["intensity"] < 0:
        raise ValueError("lights.fill.intensity must not be negative")
    _color(lg["fill"]["color"], "lights.fill.color")
    for key in ("width", "thickness"):
        if m[key] <= 0:
            raise ValueError(f"floor_markings.{key} must be positive")
    for key in ("inset", "lift"):
        if m[key] < 0:
            raise ValueError(f"floor_markings.{key} must not be negative")
    if m["inset"] + m["width"] >= a["half_width"]:
        raise ValueError("floor_markings: inset + width leaves the main aisle")


def load_spec(path: str | Path) -> dict:
    return validate_spec(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def bounds(b: Box) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(lo, hi) corners of the space a box takes, its turn about x included."""
    th = math.radians(b.rot_x_deg)
    c, s = abs(math.cos(th)), abs(math.sin(th))
    h = (b.size[0] / 2, c * b.size[1] / 2 + s * b.size[2] / 2, s * b.size[1] / 2 + c * b.size[2] / 2)
    return (tuple(b.center[i] - h[i] for i in range(3)), tuple(b.center[i] + h[i] for i in range(3)))


def validate_layout(spec: dict, boxes: list[Box]) -> list[Box]:
    """The built layout, not just the spec: finite, positive, nothing of the rack inside the
    aisle, and -- since the hall was closed (2026-09-29) -- every collider inside the walls'
    inner faces, everything inside their outer faces between the ground and the ceiling, the
    roof between the racks' top and the ceiling, and the fixtures and their lights between
    the racks' top and the trusses. build_layout runs this before returning."""
    eps = 1e-9
    half = spec["aisle"]["half_width"]
    for b in boxes:
        if not all(math.isfinite(v) for v in (*b.center, *b.size, b.rot_x_deg)):
            raise ValueError(f"{b.name}: centre/size/rotation must be finite")
        if min(b.size) <= 0:
            raise ValueError(f"{b.name}: size must be positive, got {b.size}")
        if b.semantic == SEMANTIC_RACK and abs(b.center[1]) - b.size[1] / 2 < half - 1e-9:
            raise ValueError(f"{b.name}: reaches into the aisle (inner face "
                             f"{abs(b.center[1]) - b.size[1] / 2:.3f} < half_width {half})")
    x0, x1, y_extent = building_extent(spec)
    t = float(spec["walls"]["thickness"])
    top_z = float(spec["ceiling"]["z"]) + SLAB / 2
    for b in boxes:
        lo, hi = bounds(b)
        if b.collider and (lo[0] < x0 - eps or hi[0] > x1 + eps or lo[1] < -y_extent - eps or hi[1] > y_extent + eps):
            raise ValueError(f"{b.name}: a collider outside the walls' inner faces — floor.margin must leave room "
                             "for the racks")
        if (lo[0] < x0 - t - eps or hi[0] > x1 + t + eps or lo[1] < -y_extent - t - eps
                or hi[1] > y_extent + t + eps or lo[2] < -eps or hi[2] > top_z + eps):
            raise ValueError(f"{b.name}: outside the hall (the walls' outer faces, the ground, the ceiling)")
    racks_top = max((bounds(b)[1][2] for b in boxes if b.semantic == SEMANTIC_RACK), default=0.0)
    underside = float(spec["ceiling"]["z"]) - SLAB / 2
    roof = [b for b in boxes if b.name.startswith("roof/")]
    for b in roof:
        lo, hi = bounds(b)
        if lo[2] <= racks_top or hi[2] > underside + eps:
            raise ValueError(f"{b.name}: a roof member outside the space between the racks' top ({racks_top:.3f}) "
                             f"and the ceiling ({underside})")
    truss_bottom = min((bounds(b)[0][2] for b in roof if b.name.startswith("roof/truss")), default=underside)
    for b in boxes:
        if b.name.startswith(("fixture/", "light/")):
            lo, hi = bounds(b)
            if lo[2] <= racks_top or hi[2] >= truss_bottom:
                raise ValueError(f"{b.name}: not between the racks' top ({racks_top:.3f}) and the trusses "
                                 f"({truss_bottom:.3f})")
    return boxes


def rack_inner_face_y(spec: dict) -> float:
    """Distance from the aisle centre line to the front row's inner face."""
    return float(spec["aisle"]["half_width"])


def building_extent(spec: dict) -> tuple[float, float, float]:
    """(x0, x1, y_extent): the floor's edges, which are the walls' inner faces. The floor
    reaches floor.margin past the robot's run and the outermost rack rows."""
    a, f, r = spec["aisle"], spec["floor"], spec["rack"]
    y_extent = a["half_width"] + r["rows_per_side"] * (r["depth"] + r["row_gap"]) + f["margin"]
    return float(a["x_min"] - f["margin"]), float(a["x_max"] + f["margin"]), float(y_extent)


def rack_top(spec: dict) -> float:
    """Highest point of a rack or of the boxes on it: the top shelf's board lies on the top
    beam, and _boxes stops the top shelf's stacks top_shelf_clearance above the beam height
    (with a smaller clearance no box fits and the board is the top)."""
    r = spec["rack"]
    h = max(r["beam_heights"])
    return float(h + max(r["beam_size"] + r["shelf_thickness"], spec["boxes"]["top_shelf_clearance"]))


def rack_bays(spec: dict) -> int:
    """Bays in a rack row: enough to cover the robot's run."""
    a, r = spec["aisle"], spec["rack"]
    return int(np.ceil((a["x_max"] - a["x_min"]) / r["bay_width"]))


def rack_x_extent(spec: dict) -> tuple[float, float]:
    """x from the first upright's outer face to the last one's (_rack_row)."""
    a, r = spec["aisle"], spec["rack"]
    u = float(r["upright_size"])
    return float(a["x_min"] - u / 2), float(a["x_min"] + rack_bays(spec) * r["bay_width"] + u / 2)


def roof_levels(spec: dict) -> dict[str, float]:
    """Heights of the roof structure, hung from the ceiling slab's underside: the purlins
    right under it, the truss's top chord under them, its bottom chord truss_depth lower."""
    roof = spec["roof"]
    underside = float(spec["ceiling"]["z"]) - SLAB / 2
    purlin_h = float(roof["purlin_size"][1])
    chord = float(roof["chord_size"])
    top = underside - purlin_h - chord / 2
    bottom = top - float(roof["truss_depth"])
    return {"underside": underside, "purlin": underside - purlin_h / 2, "top_chord": top,
            "bottom_chord": bottom, "truss_bottom": bottom - chord / 2}


def _truss_bays(spec: dict) -> int:
    x0, x1, _ = building_extent(spec)
    return max(1, round((x1 - x0) / spec["roof"]["truss_spacing_m"]))


def truss_pitch(spec: dict) -> float:
    """roof.truss_spacing_m is a target: the floor's length is cut into the nearest whole
    number of bays, so the trusses end one bay from each end wall."""
    x0, x1, _ = building_extent(spec)
    return (x1 - x0) / _truss_bays(spec)


def truss_positions(spec: dict) -> list[float]:
    x0, _, _ = building_extent(spec)
    return [x0 + i * truss_pitch(spec) for i in range(1, _truss_bays(spec))]


def bay_centres(spec: dict) -> list[float]:
    """x of the middle of each bay between two trusses (or a truss and an end wall)."""
    x0, _, _ = building_extent(spec)
    return [x0 + (i + 0.5) * truss_pitch(spec) for i in range(_truss_bays(spec))]


def rack_row_bands(spec: dict) -> list[tuple[float, float]]:
    """(lo, hi) in y of every rack row, from one depth face to the other -- the rows _rack_row builds, both
    sides, in increasing y. aisle_strips is the floor between them."""
    a, r = spec["aisle"], spec["rack"]
    bands = []
    for row in range(int(r["rows_per_side"])):
        inner = float(a["half_width"] + row * (r["depth"] + r["row_gap"]))
        bands += [(inner, inner + float(r["depth"])), (-inner - float(r["depth"]), -inner)]
    return sorted(bands)


def aisle_strips(spec: dict) -> list[tuple[float, float]]:
    """(lo, hi) in y of every strip of floor no rack row stands on: the main aisle, each gap
    between two rows, and the strip between the outermost row and the side wall, both sides."""
    a, r = spec["aisle"], spec["rack"]
    _, _, y_extent = building_extent(spec)
    side: list[tuple[float, float]] = []
    face = a["half_width"] + r["depth"]           # outer face of the front row
    for _ in range(1, int(r["rows_per_side"])):
        if r["row_gap"] > 0:
            side.append((float(face), float(face + r["row_gap"])))
        face += r["row_gap"] + r["depth"]
    if y_extent > face:
        side.append((float(face), float(y_extent)))
    half = float(a["half_width"])
    return [(-hi, -lo) for lo, hi in reversed(side)] + [(-half, half)] + side


def aisle_centre_lines(spec: dict) -> list[float]:
    """y of the middle of each aisle strip -- where a row of fixtures hangs."""
    return [(lo + hi) / 2 for lo, hi in aisle_strips(spec)]


def _shelf_key(side: int, row: int, bay: int, level: int) -> str:
    return f"s{'p' if side > 0 else 'n'}r{row}/b{bay}l{level}"


def _rack_row(spec: dict, side: int, row: int) -> list[Box]:
    """One row of bays on one side. Row 0 touches the aisle at half_width."""
    r = spec["rack"]
    a = spec["aisle"]
    u = float(r["upright_size"])
    depth = float(r["depth"])
    beam = float(r["beam_size"])
    shelf_t = float(r["shelf_thickness"])
    y_inner = a["half_width"] + row * (depth + r["row_gap"])
    y_center = side * (y_inner + depth / 2)
    n_bays = rack_bays(spec)
    tag = f"s{'p' if side > 0 else 'n'}r{row}"
    top = max(r["beam_heights"]) + beam
    # uprights and beams sit flush with the two depth faces of the row, so the
    # inner face of the front row is exactly y_inner whichever member is wider
    # (a beam centred on the upright line would poke beam/2 - u/2 into the aisle)
    y_faces = (-depth / 2 + u / 2, depth / 2 - u / 2)
    y_faces_beam = (-depth / 2 + beam / 2, depth / 2 - beam / 2)
    out: list[Box] = []
    for i in range(n_bays + 1):
        x = a["x_min"] + i * r["bay_width"]
        for k, yoff in enumerate(y_faces):
            out.append(Box(f"rack/upright/{tag}/b{i}k{k}", (float(x), float(y_center + yoff), top / 2),
                           (u, u, float(top)), r["materials"]["upright"], True, SEMANTIC_RACK))
    for i in range(n_bays):
        x = a["x_min"] + (i + 0.5) * r["bay_width"]
        for level, h in enumerate(r["beam_heights"]):
            for k, yoff in enumerate(y_faces_beam):
                out.append(Box(f"rack/beam/{tag}/b{i}l{level}k{k}", (float(x), float(y_center + yoff), h + beam / 2),
                               (r["bay_width"] - u, beam, beam), r["materials"]["beam"], True, SEMANTIC_RACK))
            out.append(Box(f"rack/shelf/{_shelf_key(side, row, i, level)}",
                           (float(x), float(y_center), h + beam + shelf_t / 2),
                           (r["bay_width"] - u, depth - u, shelf_t), r["materials"]["shelf"], True, SEMANTIC_RACK))
    return out


def _boxes(spec: dict, racks: list[Box], rng: np.random.Generator) -> list[Box]:
    """Cardboard boxes on every shelf: n stacks per bay, each 1..layers high.

    A stack stops at the beam above the shelf (or top_shelf_clearance on the
    top shelf), and stacks share the bay width so they never overlap.
    """
    b = spec["boxes"]
    r = spec["rack"]
    heights = sorted(float(h) for h in r["beam_heights"])
    shelves = [s for s in racks if s.name.startswith("rack/shelf/")]
    lo, hi = np.asarray(b["size_min"], float), np.asarray(b["size_max"], float)
    out: list[Box] = []
    for s in shelves:
        level = int(s.name.rsplit("l", 1)[1])
        shelf_h = heights[level]
        ceiling = heights[level + 1] if level + 1 < len(heights) else shelf_h + float(b["top_shelf_clearance"])
        shelf_top = s.center[2] + s.size[2] / 2
        n_stacks = int(rng.integers(b["stacks_per_bay"][0], b["stacks_per_bay"][1] + 1))
        # stack centres share the usable width; the slot pitch bounds the box footprint
        usable = s.size[0] - 2 * float(b["edge_margin"])
        pitch = usable / n_stacks
        max_xy = min(pitch, s.size[1]) - 1e-3
        for j in range(n_stacks):
            x = s.center[0] - usable / 2 + (j + 0.5) * pitch
            layers = int(rng.integers(b["layers"][0], b["layers"][1] + 1))
            z = shelf_top
            for L in range(layers):
                size = rng.uniform(lo, hi)
                size[0] = min(size[0], max_xy)
                size[1] = min(size[1], max_xy)
                if z + size[2] > ceiling:
                    break
                out.append(Box(f"box/{s.name.split('/', 2)[2]}/s{j}l{L}",
                               (float(x), float(s.center[1]), float(z + size[2] / 2)),
                               tuple(float(v) for v in size), b["material"], True, SEMANTIC_RACK))
                z += float(size[2])
    return out


def _floor_loads(spec: dict) -> list[Box]:
    """The bottom tier's loads: per_bay slots across every bay, each a pallet on the floor with a
    large carton on it, side by side (gap_m between slots) round the bay's middle and centred on
    the row's depth. Same order as the racks -- side (+y, -y), row, bay, slot -- and the pallet
    comes before its carton. Uses no random numbers, so the boxes on the shelves are untouched."""
    r = spec["rack"]
    fl = r["floor_loads"]
    per_bay = int(fl["per_bay"])
    if per_bay == 0:
        return []
    a = spec["aisle"]
    (pw, pd, ph), (cw, cd, ch) = (tuple(float(v) for v in fl[k]["size"]) for k in ("pallet", "carton"))
    pitch = max(pw, cw) + float(fl["gap_m"])
    out: list[Box] = []
    for side in (+1, -1):
        for row in range(int(r["rows_per_side"])):
            tag = f"s{'p' if side > 0 else 'n'}r{row}"
            y = side * (a["half_width"] + row * (r["depth"] + r["row_gap"]) + r["depth"] / 2)
            for bay in range(rack_bays(spec)):
                x_bay = a["x_min"] + (bay + 0.5) * r["bay_width"]
                for slot in range(per_bay):
                    x = x_bay + (slot - (per_bay - 1) / 2) * pitch
                    name = f"{FLOOR_LOAD_PREFIX}{tag}/b{bay}/{slot}"
                    out.append(Box(f"{name}/pallet", (float(x), float(y), ph / 2), (pw, pd, ph),
                                   fl["pallet"]["material"], True, SEMANTIC_RACK))
                    out.append(Box(f"{name}/carton", (float(x), float(y), ph + ch / 2), (cw, cd, ch),
                                   fl["carton"]["material"], True, SEMANTIC_RACK))
    return out


def _walls(spec: dict) -> list[Box]:
    """A wall on each floor edge, from the floor to the ceiling slab's top, in two bands: the
    concrete base and the cladding above it. The end walls run across the side walls' ends
    so the corners are closed."""
    w = spec["walls"]
    x0, x1, y_extent = building_extent(spec)
    t = float(w["thickness"])
    top = float(spec["ceiling"]["z"]) + SLAB / 2
    base = float(w["base_height"])
    bands = ([("base", 0.0, base, w["base_material"])] if base > 0 else []) + [("upper", base, top, w["material"])]
    faces = (("x0", (x0 - t / 2, 0.0), (t, 2 * y_extent + 2 * t)),
             ("x1", (x1 + t / 2, 0.0), (t, 2 * y_extent + 2 * t)),
             ("y0", ((x0 + x1) / 2, -y_extent - t / 2), (x1 - x0, t)),
             ("y1", ((x0 + x1) / 2, y_extent + t / 2), (x1 - x0, t)))
    return [Box(f"wall/{tag}/{band}", (cx, cy, (z0 + z1) / 2), (sx, sy, z1 - z0), mat, False, None)
            for tag, (cx, cy), (sx, sy) in faces for band, z0, z1, mat in bands]


def _dock_doors(spec: dict) -> list[Box]:
    """Roller doors against the inner face of both end walls, side by side and centred on the
    main aisle's centre line."""
    d = spec["walls"]["dock_doors"]
    x0, x1, _ = building_extent(spec)
    n, width, depth, height = int(d["per_end_wall"]), float(d["width"]), float(d["thickness"]), float(d["height"])
    pitch = width + float(d["gap"])
    return [Box(f"door/{tag}/{k}", (face + inward * depth / 2, (k - (n - 1) / 2) * pitch, height / 2),
                (depth, width, height), d["material"], False, None)
            for tag, face, inward in (("x0", x0, 1.0), ("x1", x1, -1.0)) for k in range(n)]


def _roof(spec: dict) -> list[Box]:
    """Warren trusses across the building (y), one between each two bays, and purlins along it
    (x) between the trusses' top chords and the ceiling. A truss is two chords and a zig-zag
    of web members from wall to wall, each member turned about x from one chord to the other."""
    roof = spec["roof"]
    x0, x1, y_extent = building_extent(spec)
    lv = roof_levels(spec)
    chord, web, depth = float(roof["chord_size"]), float(roof["web_size"]), float(roof["truss_depth"])
    mat = roof["material"]
    n_panels = max(2, round(2 * y_extent / roof["panel_m"]))
    panel = 2 * y_extent / n_panels
    length = math.hypot(panel, depth)
    rise = math.degrees(math.atan2(depth, panel))
    mid = (lv["top_chord"] + lv["bottom_chord"]) / 2
    out: list[Box] = []
    for i, x in enumerate(truss_positions(spec)):
        out.append(Box(f"roof/truss{i}/top", (x, 0.0, lv["top_chord"]), (chord, 2 * y_extent, chord), mat, False, None))
        out.append(Box(f"roof/truss{i}/bottom", (x, 0.0, lv["bottom_chord"]), (chord, 2 * y_extent, chord),
                       mat, False, None))
        for k in range(n_panels):
            # up from the bottom chord on even panels, back down on odd ones
            out.append(Box(f"roof/truss{i}/web{k}", (x, -y_extent + (k + 0.5) * panel, mid), (web, length, web),
                           mat, False, None, rise if k % 2 == 0 else -rise))
    pw, ph = (float(v) for v in roof["purlin_size"])
    n_purlins = max(1, round(2 * y_extent / roof["purlin_spacing_m"]))
    for j in range(n_purlins):
        out.append(Box(f"roof/purlin{j}", ((x0 + x1) / 2, -y_extent + (j + 0.5) * 2 * y_extent / n_purlins, lv["purlin"]),
                       (x1 - x0, pw, ph), mat, False, None))
    return out


def _fixtures(spec: dict) -> list[Box]:
    """A row of fixtures along every aisle, one in the middle of each bay between trusses, and
    under each its light -- the fixture is what the cameras see, the light what lights the scene."""
    lg = spec["lights"]
    fx, fy, fz = (float(v) for v in lg["fixture_size"])
    z = float(lg["z"])
    out: list[Box] = []
    for r, y in enumerate(aisle_centre_lines(spec)):
        for i, x in enumerate(bay_centres(spec)):
            out.append(Box(f"fixture/r{r}/{i}", (x, y, z), (fx, fy, fz), lg["fixture_material"], False, None))
            out.append(Box(f"light/r{r}/{i}", (x, y, z - fz / 2 - LIGHT_GAP), (fx, fy, LIGHT_GAP), "light", False, None))
    return out


def _floor_markings(spec: dict) -> list[Box]:
    """A painted line along each edge of the main aisle, inset from the rack face, wall to wall,
    lifted off the floor slab so the two faces never coincide."""
    m, a = spec["floor_markings"], spec["aisle"]
    x0, x1, _ = building_extent(spec)
    width, thick = float(m["width"]), float(m["thickness"])
    z = float(spec["floor"]["z"]) + SLAB / 2 + float(m["lift"]) + thick / 2
    y = float(a["half_width"]) - float(m["inset"]) - width / 2
    return [Box(f"marking/aisle_{tag}", ((x0 + x1) / 2, side * y, z), (x1 - x0, width, thick), m["material"], False, None)
            for tag, side in (("n", -1.0), ("p", 1.0))]


def build_layout(spec: dict) -> list[Box]:
    f = spec["floor"]
    c = spec["ceiling"]
    r = spec["rack"]
    x0, x1, y_extent = building_extent(spec)
    x_len = x1 - x0
    xc = (x0 + x1) / 2
    out = [Box("floor", (xc, 0.0, float(f["z"])), (x_len, 2 * y_extent, SLAB), f["material"], False, None),
           Box("ceiling", (xc, 0.0, float(c["z"])), (x_len, 2 * y_extent, SLAB), c["material"], False, None)]
    racks: list[Box] = []
    for side in (+1, -1):
        for row in range(int(r["rows_per_side"])):
            racks += _rack_row(spec, side, row)
    rng = np.random.default_rng(int(spec["boxes"]["seed"]))
    out += racks + _boxes(spec, racks, rng) + _floor_loads(spec)
    # Dressing after every collider: the rack and box prims keep the order and paths the
    # control arms were measured with (2026-09-23).
    out += _walls(spec) + _dock_doors(spec) + _roof(spec) + _fixtures(spec) + _floor_markings(spec)
    return validate_layout(spec, out)


# ---- the box labels ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelPlate:
    """One plate of a box label, in the scene frame: the white card (kind "base") or the CI on it ("ci")."""
    box: str                                  # the layout box it lies on (Box.name)
    face: str                                 # "+y" or "-y" (LABEL_FACES)
    kind: str                                 # LABEL_PLATES
    center: tuple[float, float, float]
    normal: tuple[float, float, float]        # outward, the face's
    width: float                              # along LABEL_UP x normal: the viewer's right facing the face
    height: float                             # along LABEL_UP

    def corners(self) -> np.ndarray:
        """The four corners, counter-clockwise seen from outside -- bottom-left, bottom-right, top-right,
        top-left: the order (and so the UVs (0,0) (1,0) (1,1) (0,1)) the carried box's plates use."""
        up = np.asarray(LABEL_UP, dtype=float)
        right = np.cross(up, np.asarray(self.normal, dtype=float)) * (self.width / 2.0)
        up = up * (self.height / 2.0)
        # + 0.0 turns a negated zero into 0.0, so the USD does not carry -0 coordinates
        return np.asarray(self.center, dtype=float) + np.array([-right - up, right - up, right + up, -right + up]) + 0.0


def label_kind(b: Box) -> str | None:
    """Which labels entry a layout box carries (LABEL_BOXES): a box on a shelf, a bottom-tier carton, or none."""
    if b.name.startswith("box/"):
        return "shelf_box"
    if b.name.startswith(FLOOR_LOAD_PREFIX) and b.name.endswith("/carton"):
        return "floor_carton"
    return None


def label_fit(spec: dict, boxes: list[Box]) -> list[Box]:
    """The labelled boxes, in layout order, once each label's card is known to fit both y faces of its box and
    every plate of it -- the white card at labels.lift_m and the CI at labels.ci_lift_m off the face, each on
    its own -- to lie within the box's rack row in y (rack_row_bands: between the row's two depth faces), so no
    plate stands in an aisle or outside the hall. Raises ValueError naming the box, the face and the plate
    otherwise. Needs no picture: the CI's height is box_labels' to check."""
    lb = spec["labels"]
    bands = rack_row_bands(spec)
    out = []
    for b in boxes:
        which = label_kind(b)
        if which is None:
            continue
        w, h = (float(v) for v in lb[which]["size"])
        if w > b.size[0] + 1e-9 or h > b.size[2] + 1e-9:
            raise ValueError(f"{b.name}: its y face ({b.size[0]:.3f} x {b.size[2]:.3f} m) does not hold the "
                             f"labels.{which} card ({w} x {h} m)")
        band = next(((lo, hi) for lo, hi in bands if lo < b.center[1] < hi), None)
        if band is None:
            raise ValueError(f"{b.name}: its centre (y {b.center[1]:.4f}) is in no rack row — a label goes on a "
                             "box that stands in one")
        for face, normal in LABEL_FACES.items():
            for kind, lift_key in zip(LABEL_PLATES, ("lift_m", "ci_lift_m")):
                y = b.center[1] + normal[1] * (b.size[1] / 2 + float(lb[lift_key]))
                if not band[0] - 1e-9 <= y <= band[1] + 1e-9:
                    raise ValueError(f"{b.name}: the {PLATE_NAMES[kind]} on its {face} face would lie at y {y:.4f}, "
                                     f"outside its rack row (y {band[0]:.3f} .. {band[1]:.3f}) — labels.{lift_key} "
                                     "takes it past the row's face")
        out.append(b)
    return out


def box_labels(spec: dict, boxes: list[Box], ci_aspect: float) -> list[LabelPlate]:
    """The label plates of the layout ``boxes``: for every labelled box (label_kind) in layout order, on its +y
    then its -y face, the white card (labels.<kind>.size, labels.lift_m off the face) then the CI (labels.<kind>.
    ci_width wide, ci_width / ``ci_aspect`` high -- the picture's width / height in pixels -- labels.ci_lift_m
    off it), both centred on the face. Raises ValueError when a card does not fit its face, a plate leaves its
    box's rack row (label_fit) or the CI is taller than its card."""
    aspect = float(ci_aspect)
    if not math.isfinite(aspect) or aspect <= 0:
        raise ValueError(f"the CI picture's width / height is not a positive finite number: {ci_aspect!r}")
    lb = spec["labels"]
    out: list[LabelPlate] = []
    for b in label_fit(spec, boxes):
        which = label_kind(b)
        w, h = (float(v) for v in lb[which]["size"])
        ci_w = float(lb[which]["ci_width"])
        ci_h = ci_w / aspect
        if ci_h > h + 1e-9:
            raise ValueError(f"labels.{which}: the CI {ci_w} x {ci_h:.4f} m is taller than its card ({h} m)")
        for face, normal in LABEL_FACES.items():
            for kind, lift, size in ((LABEL_PLATES[0], lb["lift_m"], (w, h)), (LABEL_PLATES[1], lb["ci_lift_m"], (ci_w, ci_h))):
                centre = (b.center[0], b.center[1] + normal[1] * (b.size[1] / 2 + float(lift)), b.center[2])
                out.append(LabelPlate(b.name, face, kind, tuple(float(v) for v in centre), normal,
                                      float(size[0]), float(size[1])))
    return out
