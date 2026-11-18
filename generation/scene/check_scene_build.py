#!/usr/bin/env python3
"""Is the built logistics-aisle USD what the layout says? (~/isaacsim/python.sh)

    ~/isaacsim/python.sh generation/scene/check_scene_build.py

Reads the *product*, not the builder: prim counts, colliders, semantic labels,
every cube's transform against its layout box (the truss members' turn
included), the front row's inner face measured from the USD transforms,
texture references that resolve beside the USD (and none on a plain-colour
material), the lights' values and the one shadowless fill dome, the spec hash
it carries, and that a rebuild is byte-identical (otherwise the usd_sha256
meta/assets.json pins would mean nothing). The box labels too: two meshes with
no collider and the boxes' class, two plates per kind on every shelf box and
every bottom-tier carton -- each on its y face, inside it, lifted off it by the
spec's distance, at the spec's size, wound and mapped so the CI reads the right
way round -- the KETI CI beside the USD at its pinned hash, and the card and CI
materials input for input those of a carried box built from the same picture.
Exit 0 only if everything holds.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdSemantics, UsdShade

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation.scene import build_logistics_scene as bls  # noqa: E402
from generation.scene import carried_box, fetch_keti_ci, fetch_textures, layout  # noqa: E402

# How far a label plate's stored corner may sit from where the layout puts it: the points are float32, and
# 34 m from the origin a float32 step is 4e-6 m.
LABEL_TOL_M = 2e-5


def _labels(prim) -> list[str]:
    api = UsdSemantics.LabelsAPI(prim, "class")
    attr = api.GetLabelsAttr() if api else None
    return list(attr.Get() or []) if attr and attr.HasAuthoredValue() else []


def _f32(v) -> float:
    """What a float attribute holds once authored: USD stores it as float32."""
    return float(Gf.Vec3f(float(v), 0.0, 0.0)[0])


def _f32s(values) -> tuple[float, ...]:
    return tuple(_f32(v) for v in values)


def _as_plain(value):
    """Gf vectors as tuples, so an attribute compares with a tuple of floats."""
    return tuple(value) if isinstance(value, (Gf.Vec2f, Gf.Vec3f, Gf.Vec3d)) else value


def _world(prim) -> Gf.Matrix4d:
    # world, not local: a transform on the root moves every box with it
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _close(got: Gf.Matrix4d, want: Gf.Matrix4d) -> bool:
    """Equal up to double rounding of the same authored values -- relative, because a 44.1 m
    floor is not 44.1 once stored as float32."""
    return all(abs(got[i][j] - want[i][j]) <= 1e-9 * max(1.0, abs(want[i][j])) for i in range(4) for j in range(4))


def check(usd: Path, spec_path: Path) -> list[str]:
    problems: list[str] = []
    spec = layout.load_spec(spec_path)
    boxes = layout.build_layout(spec)
    by_name = {bls.prim_name(b.name): b for b in boxes}
    stage = Usd.Stage.Open(str(usd))
    root = stage.GetPrimAtPath(bls.ROOT)
    if not root or stage.GetDefaultPrim() != root:
        problems.append(f"default prim is not {bls.ROOT}")
    prims = [p for p in stage.Traverse() if p.GetTypeName() in ("Cube", "RectLight")]
    if len(prims) != len(boxes):
        problems.append(f"prim count {len(prims)} != layout boxes {len(boxes)}")
    names = {p.GetName() for p in prims}
    if names != set(by_name):
        problems.append(f"prim names differ from layout: {sorted(names ^ set(by_name))[:5]}")
    n_coll = sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in prims)
    want_coll = sum(b.collider for b in boxes)
    if n_coll != want_coll:
        problems.append(f"colliders {n_coll} != {want_coll}")
    # ... and nothing else on the stage collides: the labels and the rest of what is not a layout box add none
    colliding = {bls.prim_name(b.name) for b in boxes if b.collider}
    stray = [str(p.GetPath()) for p in stage.Traverse()
             if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetName() not in colliding]
    if stray:
        problems.append(f"colliders outside the layout's boxes: {stray[:5]}")
    n_lab = sum(1 for p in prims if _labels(p))
    want_lab = sum(1 for b in boxes if b.semantic)
    if n_lab != want_lab:
        problems.append(f"labelled prims {n_lab} != {want_lab}")
    for p in prims:
        b = by_name.get(p.GetName())
        if b is None:
            continue
        if b.collider != p.HasAPI(UsdPhysics.CollisionAPI):
            problems.append(f"{p.GetName()}: collider {p.HasAPI(UsdPhysics.CollisionAPI)} != layout {b.collider}")
        if (_labels(p) or None) != ([b.semantic] if b.semantic else None):
            problems.append(f"{p.GetName()}: labels {_labels(p)} != layout {b.semantic}")
        if UsdGeom.Imageable(p).ComputeVisibility() == UsdGeom.Tokens.invisible:
            problems.append(f"{p.GetName()}: invisible — a hidden box shows nothing, a hidden light lights nothing")
        if p.GetTypeName() == "Cube":
            # translate, turn about x, scale a unit cube (row vectors: S * R * T), from what the
            # builder authors: a double3 centre, a float angle, a float3 scale
            want = (Gf.Matrix4d().SetScale(Gf.Vec3d(Gf.Vec3f(*b.size)))
                    * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), _f32(b.rot_x_deg)))
                    * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*b.center)))
            if not _close(_world(p), want) or UsdGeom.Cube(p).GetSizeAttr().Get() != 1.0:
                problems.append(f"{p.GetName()}: transform differs from its layout box")
            bound = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
            if not bound or str(bound.GetPath()) != f"{bls.ROOT}/Looks/{b.material}":
                problems.append(f"{p.GetName()}: bound to {bound.GetPath() if bound else None}, not {b.material}")
        else:
            light = UsdLux.RectLight(p)
            lg = spec["lights"]
            # a light only moves: no turn (it shines along its -Z, down) and no scale
            if not _close(_world(p), Gf.Matrix4d().SetTranslate(Gf.Vec3d(*b.center))):
                problems.append(f"{p.GetName()}: not at its layout box, or turned away from straight down")
            if (light.GetIntensityAttr().Get() != _f32(lg["intensity"])
                    or tuple(light.GetColorAttr().Get()) != _f32s(lg["color"])
                    or (light.GetWidthAttr().Get(), light.GetHeightAttr().Get()) != _f32s(b.size[:2])):
                problems.append(f"{p.GetName()}: intensity/colour/size differ from lights and its layout box")
    # The fill: one dome, shadowless, with the spec's values and no texture (nothing from outside).
    domes = [p for p in stage.Traverse() if p.GetTypeName() == "DomeLight"]
    if [str(p.GetPath()) for p in domes] != [f"{bls.ROOT}/{bls.FILL_LIGHT}"]:
        problems.append(f"dome lights {[str(p.GetPath()) for p in domes]} != the one fill light")
    else:
        dome, fill = UsdLux.DomeLight(domes[0]), spec["lights"]["fill"]
        shadow = UsdLux.ShadowAPI(domes[0]).GetShadowEnableAttr()
        if dome.GetIntensityAttr().Get() != _f32(fill["intensity"]) or tuple(dome.GetColorAttr().Get()) != _f32s(fill["color"]):
            problems.append("fill light: intensity/colour differ from lights.fill")
        if UsdGeom.Imageable(domes[0]).ComputeVisibility() == UsdGeom.Tokens.invisible:
            problems.append("fill light: invisible — it lights nothing")
        if not shadow or shadow.Get() is not False:
            problems.append("fill light: shadows are not off — the hall's ceiling would stay black")
        if dome.GetTextureFileAttr().HasAuthoredValue():
            problems.append("fill light: carries a texture — nothing from outside belongs in the hall")
    # Front-row inner face, measured from the USD transforms (translate + scale on a unit cube),
    # in world space: the root's own transform counts.
    inner = None
    for p in prims:
        if p.GetTypeName() != "Cube" or not (p.GetName().startswith("rack_") or p.GetName().startswith("box_")):
            continue
        m = _world(p)
        ty = m.ExtractTranslation()[1]
        sy = m.GetRow3(1).GetLength() * UsdGeom.Cube(p).GetSizeAttr().Get()
        face = abs(ty) - sy / 2
        inner = face if inner is None else min(inner, face)
    half = spec["aisle"]["half_width"]
    if inner is None or inner < half - 1e-6:
        problems.append(f"rack inner face {inner} < aisle.half_width {half}")
    # Texture references resolve beside the USD *and* hash as the lock says --
    # is_file() alone let a swapped bundled texture pass.
    # And every used material carries every map: one shader
    # with none was invisible to a global "some textures" count.
    entries = fetch_textures.parse_lock()
    by_filename = {e.filename: e for e in entries if e.map != "archive"}
    pin = fetch_keti_ci.load_ci_lock()
    lb = spec["labels"]
    used_materials = {b.material for b in boxes if b.material != "light"} | {lb["material"], lb["ci_material"]}
    n_tex = 0
    per_material: dict[str, int] = {}
    for p in stage.Traverse():
        if p.GetTypeName() != "Shader":
            continue
        mat_name = p.GetParent().GetName()
        per_material.setdefault(mat_name, 0)
        for inp in UsdShade.Shader(p).GetInputs():
            if inp.GetTypeName().type.typeName != "SdfAssetPath":
                continue
            ref = inp.Get().path
            n_tex += 1
            per_material[mat_name] += 1
            file = usd.parent / ref
            entry = by_filename.get(Path(ref).name)
            if Path(ref).name == pin["file"]:
                if not file.is_file() or hls(file) != pin["member_sha256"]:
                    problems.append(f"{p.GetPath()}.{inp.GetBaseName()}: {ref} missing beside the USD or its "
                                    f"SHA256 differs from the KETI CI pin")
            elif entry is None:
                problems.append(f"{p.GetPath()}.{inp.GetBaseName()}: {ref} is not a file the lock knows")
            elif not fetch_textures.verify(entry, file):
                problems.append(f"{p.GetPath()}.{inp.GetBaseName()}: {ref} missing beside the USD or its "
                                f"SHA256 differs from the lock ({entry.name})")
    want_maps = len(fetch_textures.MATERIAL_MAPS)
    textured = {m for m in used_materials if "texture" in spec["materials"][m]}
    pictured = {m for m in used_materials if "picture" in spec["materials"][m]}
    ci_png = usd.parent / bls.TEXTURE_SUBDIR / pin["file"]
    # a picture names its file twice when it is cut out by its own alpha (diffuse and opacity), once otherwise
    picture_refs = 2 if ci_png.is_file() and carried_box.png_has_alpha(ci_png) else 1
    for m in sorted(used_materials):
        got = per_material.get(m)
        want = want_maps if m in textured else picture_refs if m in pictured else 0   # a plain colour names no file
        if got != want:
            problems.append(f"material {m!r}: {got} texture inputs, expected {want}")
    if n_tex != len(textured) * want_maps + len(pictured) * picture_refs:
        problems.append(f"texture references {n_tex} != textured materials {len(textured)} x maps {want_maps} "
                        f"+ pictures {len(pictured)} x {picture_refs}")
    # Every constant the builder authors, against the spec -- a flag alone let a fixture glow
    # at intensity 0. None: the spec does not set it, so it must not be authored.
    for m in sorted(used_materials):
        sh = UsdShade.Shader(stage.GetPrimAtPath(f"{bls.ROOT}/Looks/{m}/Shader"))
        ms = spec["materials"][m]
        if m in textured:
            want = {"project_uvw": True, "world_or_object": True,
                    "texture_scale": (_f32(1.0 / ms["tile_m"]),) * 2,
                    "diffuse_tint": _f32s(ms["tint"]) if "tint" in ms else None}
        elif m in pictured:
            # on the plates' UVs, not projected; the cut-out is the carried box's (_check_labels compares)
            want = {"reflection_roughness_constant": _f32(ms["roughness"]), "project_uvw": None,
                    "diffuse_color_constant": None, "diffuse_tint": None}
        else:
            glows = "emissive_intensity" in ms
            want = {"diffuse_color_constant": _f32s(ms["color"]),
                    "reflection_roughness_constant": _f32(ms["roughness"]) if "roughness" in ms else None,
                    "metallic_constant": _f32(ms["metallic"]) if "metallic" in ms else None,
                    "enable_emission": True if glows else None,
                    "emissive_color": _f32s(ms["color"]) if glows else None,
                    "emissive_intensity": _f32(ms["emissive_intensity"]) if glows else None}
        for name, value in want.items():
            inp = sh.GetInput(name) if sh else None
            got = _as_plain(inp.Get()) if inp else None
            if got != value:
                problems.append(f"material {m!r}: {name} is {got!r}, the spec says {value!r}")
    got = root.GetCustomDataByKey("keti:specSha256") if root else None
    if got != hls(spec_path):
        problems.append("keti:specSha256 does not match logistics_aisle.yaml")
    if root and (root.GetCustomDataByKey("keti:ciFile"), root.GetCustomDataByKey("keti:ciSha256")) != (
            pin["file"], pin["member_sha256"]):
        problems.append("keti:ciFile / keti:ciSha256 do not name the KETI CI keti_ci.json pins")
    label_problems, n_plates = _check_labels(stage, usd, spec, boxes, ci_png)
    problems += label_problems
    # Determinism: rebuild elsewhere and compare bytes.
    with tempfile.TemporaryDirectory() as tmp:
        again = bls.build(spec_path, Path(tmp) / usd.name)
        if again.read_bytes() != usd.read_bytes():
            problems.append("a rebuild is not byte-identical — usd_sha256 cannot be pinned")
    print(f"prims {len(prims)}/{len(boxes)}  colliders {n_coll}/{want_coll}  labelled {n_lab}/{want_lab}  "
          f"textures {n_tex}  label plates {n_plates}  rack_inner_y {inner:.3f} (>= {half})  sha256 {hls(usd)}")
    return problems


def _shader_inputs(stage, path: str) -> dict:
    """A shader's authored inputs, an asset by its path."""
    sh = UsdShade.Shader(stage.GetPrimAtPath(path))
    if not sh:
        return {}
    return {i.GetBaseName(): (i.Get().path if i.GetTypeName().type.typeName == "SdfAssetPath" else _as_plain(i.Get()))
            for i in sh.GetInputs()}


def _check_labels(stage, usd: Path, spec: dict, boxes: list, ci_png: Path) -> tuple[list[str], dict]:
    """The two label meshes against the layout's boxes and the spec's labels (see the module docstring)."""
    problems: list[str] = []
    lb = spec["labels"]
    if not ci_png.is_file():
        return [f"{ci_png.name} is not beside the USD — the labels cannot be checked"], {}
    width, height = carried_box.png_size(ci_png)
    aspect = width / height
    # how many boxes carry a label, counted from the layout's names: every box on a shelf, every carton
    shelf = [b for b in boxes if b.name.startswith("box/")]
    cartons = [b for b in boxes if b.name.startswith(layout.FLOOR_LOAD_PREFIX) and b.name.endswith("/carton")]
    want_n = 2 * (len(shelf) + len(cartons))
    by_name = {b.name: b for b in boxes}
    plates = layout.box_labels(spec, boxes, aspect)
    counted: dict[str, int] = {}
    up = np.array(layout.LABEL_UP)
    for kind, mat_key, lift_key in zip(layout.LABEL_PLATES, ("material", "ci_material"), ("lift_m", "ci_lift_m")):
        path = bls.LABEL_MESHES[kind]
        prim = stage.GetPrimAtPath(path)
        if not prim or prim.GetTypeName() != "Mesh":
            problems.append(f"{path}: no label mesh")
            continue
        mine = [p for p in plates if p.kind == kind]
        if len(mine) != want_n:
            problems.append(f"{kind}: the layout gives {len(mine)} plates, 2 x ({len(shelf)} shelf boxes + "
                            f"{len(cartons)} cartons) = {want_n}")
        physics = [s for s in prim.GetAppliedSchemas() if s.startswith("Physics") or s.startswith("Physx")]
        if physics:
            problems.append(f"{path}: carries physics {physics} — a label has no collider")
        if _labels(prim) != [layout.SEMANTIC_RACK]:
            problems.append(f"{path}: labels {_labels(prim)} != [{layout.SEMANTIC_RACK!r}], the boxes' class")
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            problems.append(f"{path}: invisible")
        if not _close(_world(prim), Gf.Matrix4d(1.0)):
            problems.append(f"{path}: moved — its points are in the scene frame")
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if not bound or str(bound.GetPath()) != f"{bls.ROOT}/Looks/{lb[mat_key]}":
            problems.append(f"{path}: bound to {bound.GetPath() if bound else None}, not {lb[mat_key]}")
        mesh = UsdGeom.Mesh(prim)
        pts = np.array(mesh.GetPointsAttr().Get() or [], dtype=float)
        n = len(pts) // 4
        counted[kind] = n
        if n != want_n or len(pts) != 4 * n:
            problems.append(f"{path}: {len(pts)} points, {want_n} plates of 4 expected")
            continue
        if list(mesh.GetFaceVertexCountsAttr().Get()) != [4] * n or \
                list(mesh.GetFaceVertexIndicesAttr().Get()) != list(range(4 * n)):
            problems.append(f"{path}: not one quad of its own four points per plate")
        q = pts.reshape(n, 4, 3)
        normals = np.array(mesh.GetNormalsAttr().Get(), dtype=float).reshape(n, 4, 3)
        if kind == layout.LABEL_PLATES[1]:
            st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
            uv = np.array(st.Get() if st else [], dtype=float)
            if not st or st.GetInterpolation() != UsdGeom.Tokens.vertex or \
                    uv.shape != (4 * n, 2) or not np.array_equal(uv.reshape(n, 4, 2), np.tile([[0, 0], [1, 0], [1, 1], [0, 1]], (n, 1, 1))):
                problems.append(f"{path}: its UVs are not (0,0) (1,0) (1,1) (0,1) on every plate's corners")
        wrong: list[str] = []
        for i, plate in enumerate(mine):
            b = by_name[plate.box]
            normal = np.array(plate.normal)
            sign = normal[1]
            face_y = b.center[1] + sign * b.size[1] / 2
            corners = q[i]
            size = (tuple(lb[layout.label_kind(b)]["size"]) if kind == layout.LABEL_PLATES[0]
                    else (lb[layout.label_kind(b)]["ci_width"], lb[layout.label_kind(b)]["ci_width"] / aspect))
            right = np.cross(up, normal)
            checks = {
                "on its face, lifted by the spec": np.abs(corners[:, 1] - (face_y + sign * lb[lift_key])).max() <= LABEL_TOL_M,
                "inside the face": (corners[:, 0].min() >= b.center[0] - b.size[0] / 2 - LABEL_TOL_M
                                    and corners[:, 0].max() <= b.center[0] + b.size[0] / 2 + LABEL_TOL_M
                                    and corners[:, 2].min() >= b.center[2] - b.size[2] / 2 - LABEL_TOL_M
                                    and corners[:, 2].max() <= b.center[2] + b.size[2] / 2 + LABEL_TOL_M),
                "centred on the face": (abs(corners[:, 0].mean() - b.center[0]) <= LABEL_TOL_M
                                        and abs(corners[:, 2].mean() - b.center[2]) <= LABEL_TOL_M),
                "the spec's size": (abs(np.ptp(corners[:, 0]) - size[0]) <= LABEL_TOL_M
                                    and abs(np.ptp(corners[:, 2]) - size[1]) <= LABEL_TOL_M),
                # first corner to second along the viewer's right, first to fourth up: the picture reads
                "wound to read the right way round": (
                    np.allclose((corners[1] - corners[0]) / size[0], right, atol=1e-3)
                    and np.allclose((corners[3] - corners[0]) / size[1], up, atol=1e-3)),
                "facing out of its face": np.allclose(normals[i], normal),
            }
            failed = [k for k, ok in checks.items() if not ok]
            if failed:
                wrong.append(f"{plate.box} {plate.face}: not {', '.join(failed)}")
        if wrong:
            problems.append(f"{path}: {len(wrong)} plate(s) wrong — {wrong[:3]}")
    # The look: the card and the CI materials are, input for input, those of a carried box built from this picture.
    with tempfile.TemporaryDirectory() as tmp:
        box_usd = Path(tmp) / "box.usd"
        carried_box.write_box_usd(box_usd, carried_box.NOMINAL_SIZE_M, ci_png)
        box = Usd.Stage.Open(str(box_usd))
        for scene_mat, box_mat in ((lb["material"], "label_white"), (lb["ci_material"], "keti_ci")):
            mine = _shader_inputs(stage, f"{bls.ROOT}/Looks/{scene_mat}/Shader")
            theirs = _shader_inputs(box, f"{carried_box.ROOT}/Looks/{box_mat}/Shader")
            if not theirs or mine != theirs:
                problems.append(f"material {scene_mat!r}: inputs {mine} differ from the carried box's {box_mat} {theirs}")
    return problems, counted


def hls(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--usd", type=Path, default=bls.OUT)
    ap.add_argument("--spec", type=Path, default=bls.SPEC)
    a = ap.parse_args()
    if not a.usd.is_file():
        print(f"no USD at {a.usd} — run build_logistics_scene.py first", file=sys.stderr)
        return 1
    problems = check(a.usd, a.spec)
    for p in problems:
        print("PROBLEM " + p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
