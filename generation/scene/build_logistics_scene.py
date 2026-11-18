#!/usr/bin/env python3
"""layout.py's boxes -> USD. Run with Isaac's python (pxr):

    ~/isaacsim/python.sh generation/scene/build_logistics_scene.py

Materials are OmniPBR (MDL): textured ones with the CC0 textures textures.lock
pins, bound through world-space box projection because UsdGeom.Cube has no
UVs; plain ones a colour (the fixtures' also emissive), with no file at all;
the one picture, the KETI CI keti_ci.json pins, on the label plates' UVs.
Racks and boxes get static box colliders and the `rack` semantic class,
baked into the asset because labels applied after spawn invalidate the
PhysX tensor view (measured). Every box on a shelf and every bottom-tier
carton carries a label on its two y faces -- a white card and the KETI CI
on it, as the carried box's (layout.box_labels) -- written as two meshes,
every card in one and every CI in the other, with no collider and the class
of the boxes they lie on. Everything else is visual only: the
ground's physics stays with Isaac Lab's procedural plane, and the
walls, doors, trusses, fixtures and floor lines are dressing.

The scene carries its own light: a RectLight under every fixture and a
shadowless fill dome, so an enclosed hall is not black wherever the downward
lights do not reach (the ceiling, the walls' upper half). S1dEnvCfg drops the
sky dome it would otherwise inherit from Isaac Lab.

Output is self-contained: the USD plus a textures/ copy beside it, referenced
relatively, so generation/scene/build/ is what a release bundles. Nothing here
carries a standard value; every dimension comes from logistics_aisle.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdSemantics, UsdShade, Vt

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation.scene import carried_box, fetch_keti_ci, fetch_textures, layout  # noqa: E402

ROOT = "/KetiLogisticsAisle"
SPEC = REPO / "generation/scene/logistics_aisle.yaml"
OUT = REPO / "generation/scene/build/keti_logistics_aisle_v1.usd"
TEXTURE_SUBDIR = "textures"
FILL_LIGHT = "FillLight"
# The label meshes: every white card, then every CI (layout.LABEL_PLATES).
LABELS = "BoxLabels"
LABEL_MESHES = {kind: f"{ROOT}/{LABELS}/{kind}" for kind in layout.LABEL_PLATES}
# The pictures a picture material may name: the KETI CI, which keti_ci.json pins (fetch_keti_ci), is the only one.
PICTURES = ("keti_ci",)
# OmniPBR input names, one per lock map. Missing maps are simply not set.
MAP_INPUTS = {"diffuse": "diffuse_texture", "normal_gl": "normalmap_texture",
              "roughness": "reflectionroughness_texture"}


def prim_name(box_name: str) -> str:
    return box_name.replace("/", "_")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_beside(src: Path, out_dir: Path) -> str:
    """``src`` copied byte for byte to ``out_dir/textures/`` (unless already there); its relative reference."""
    dst = out_dir / TEXTURE_SUBDIR / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.read_bytes() != src.read_bytes():
        shutil.copy2(src, dst)
    return f"./{TEXTURE_SUBDIR}/{src.name}"


def _material(stage, name: str, mat_spec: dict, files: dict[str, Path], out_dir: Path):
    """One OmniPBR material: textured (textures copied beside the USD and referenced
    relatively), plain (constants only -- no file, so nothing to attribute) or a picture (its
    one file copied beside the USD, on the plates' UVs, cut out by its alpha when it has one --
    input for input the carried box's CI material, carried_box._omnipbr)."""
    path = f"{ROOT}/Looks/{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    if "picture" in mat_spec:
        (src,) = files.values()
        ref = _copy_beside(src, out_dir)
        sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(mat_spec["roughness"]))
        sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(ref)
        if carried_box.png_has_alpha(src):
            sh.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
            sh.CreateInput("enable_opacity_texture", Sdf.ValueTypeNames.Bool).Set(True)
            sh.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(ref)
            sh.CreateInput("opacity_mode", Sdf.ValueTypeNames.Int).Set(0)
            sh.CreateInput("opacity_threshold", Sdf.ValueTypeNames.Float).Set(0.5)
        return _connect(mat, sh)
    if "color" in mat_spec:
        color = Gf.Vec3f(*[float(v) for v in mat_spec["color"]])
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(color)
        if "roughness" in mat_spec:
            sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(mat_spec["roughness"]))
        if "metallic" in mat_spec:
            sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(mat_spec["metallic"]))
        if "emissive_intensity" in mat_spec:
            sh.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
            sh.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(color)
            sh.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(mat_spec["emissive_intensity"]))
        return _connect(mat, sh)
    for m, inp in MAP_INPUTS.items():
        if m not in files:
            continue
        sh.CreateInput(inp, Sdf.ValueTypeNames.Asset).Set(_copy_beside(files[m], out_dir))
    tile = float(mat_spec["tile_m"])
    sh.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
    sh.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
    sh.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1.0 / tile, 1.0 / tile))
    if "tint" in mat_spec:
        sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(v) for v in mat_spec["tint"]]))
    return _connect(mat, sh)


def _connect(mat, sh):
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(out)
    return mat


def pinned_picture(picture: str, ci_lock: Path = fetch_keti_ci.CI_LOCK,
                   pictures_dir: Path = fetch_keti_ci.OUT) -> tuple[Path, dict]:
    """The file a picture material names and its pin: at the pinned bytes, or the build stops -- a picture we
    cannot attribute must not be written into the asset."""
    if picture not in PICTURES:
        raise SystemExit(f"picture {picture!r}: no pin fixes it (known: {list(PICTURES)})")
    pin = fetch_keti_ci.load_ci_lock(ci_lock)
    src = Path(pictures_dir) / pin["file"]
    if not src.is_file() or sha256_of(src) != pin["member_sha256"]:
        raise SystemExit(f"KETI CI missing or not the pinned bytes: {src} — run fetch_keti_ci.py first")
    return src, pin


def _plates_mesh(stage, path: str, plates: list, uv: bool):
    """Every plate of one kind as the faces of one mesh, four points each in the scene frame (no transform),
    with the plate's outward normal at its points and, for a picture, the carried box's UVs."""
    corners = np.concatenate([p.corners() for p in plates])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, c)) for c in corners]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * len(plates)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(4 * len(plates)))))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p.normal)) for p in plates for _ in range(4)]))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
    if uv:
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                      UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)] * len(plates)))
    return mesh


def build(spec_path: Path = SPEC, out: Path = OUT, lock_path: Path = fetch_textures.LOCK,
          textures_dir: Path = fetch_textures.OUT, ci_lock: Path = fetch_keti_ci.CI_LOCK,
          pictures_dir: Path = fetch_keti_ci.OUT) -> Path:
    spec = layout.load_spec(spec_path)
    boxes = layout.build_layout(spec)
    lb = spec["labels"]
    ci_png, ci_pin = pinned_picture(spec["materials"][lb["ci_material"]]["picture"], ci_lock, pictures_dir)
    width, height = carried_box.png_size(ci_png)
    plates = layout.box_labels(spec, boxes, width / height)
    entries = fetch_textures.parse_lock(lock_path)
    # Every texture the used materials need must be on disk with the pinned
    # hash -- a file we cannot attribute must not be written into the asset.
    used_materials = {b.material for b in boxes if b.material != "light"}
    files_by_material: dict[str, dict[str, Path]] = {}
    for m in sorted(used_materials):
        if "texture" not in spec["materials"][m]:
            files_by_material[m] = {}          # plain colour: no file
            continue
        role = spec["materials"][m]["texture"]
        try:
            # strict: an unknown or incomplete set stops the build instead of
            # authoring a material with no textures
            files = fetch_textures.files_for_material(entries, role, textures_dir)
        except ValueError as exc:
            raise SystemExit(f"material {m!r}: {exc}") from exc
        by_map = {e.map: e for e in entries if e.role == role}
        for map_name, path in files.items():
            if not fetch_textures.verify(by_map[map_name], path):
                raise SystemExit(f"texture missing or wrong hash: {path} — run fetch_textures.py first")
        files_by_material[m] = files

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    rp = root.GetPrim()
    rp.SetCustomDataByKey("keti:sceneVersion", str(spec["scene_version"]))
    rp.SetCustomDataByKey("keti:sceneName", str(spec["name"]))
    rp.SetCustomDataByKey("keti:specSha256", sha256_of(spec_path))
    rp.SetCustomDataByKey("keti:texturesLockSha256", sha256_of(lock_path))
    rp.SetCustomDataByKey("keti:ciFile", ci_png.name)
    rp.SetCustomDataByKey("keti:ciSha256", ci_pin["member_sha256"])
    rp.SetCustomDataByKey("keti:builtBy", "generation/scene/build_logistics_scene.py")
    UsdGeom.Scope.Define(stage, f"{ROOT}/Looks")

    lights = spec["lights"]
    mats = {}
    for b in boxes:
        path = f"{ROOT}/{prim_name(b.name)}"
        if b.material == "light":
            light = UsdLux.RectLight.Define(stage, path)
            light.CreateIntensityAttr(float(lights["intensity"]))
            light.CreateColorAttr(Gf.Vec3f(*[float(v) for v in lights["color"]]))
            light.CreateWidthAttr(float(b.size[0]))
            light.CreateHeightAttr(float(b.size[1]))
            # A RectLight emits along its -Z; with z up that is straight down.
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*b.center))
            continue
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*b.center))
        if b.rot_x_deg:
            xf.AddRotateXOp().Set(float(b.rot_x_deg))     # after scaling: the op order is T, R, S
        xf.AddScaleOp().Set(Gf.Vec3f(*b.size))
        if b.material not in mats:
            mats[b.material] = _material(stage, b.material, spec["materials"][b.material],
                                         files_by_material[b.material], out.parent)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mats[b.material])
        if b.collider:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if b.semantic:
            UsdSemantics.LabelsAPI.Apply(cube.GetPrim(), "class").CreateLabelsAttr([b.semantic])
    # The box labels, after every box: the cards in one mesh, the CIs in another. No collider -- the
    # layout's boxes are the colliders -- and the class of the boxes they lie on, so a box's pixels
    # read as the box's in a segmentation picture whether they are its cardboard or its label.
    classes = {b.semantic for b in boxes if layout.label_kind(b)}
    if len(classes) != 1 or None in classes:
        raise SystemExit(f"the labelled boxes carry {sorted(map(str, classes))} — the label meshes take one class")
    UsdGeom.Scope.Define(stage, f"{ROOT}/{LABELS}")
    for kind, mat_key in zip(layout.LABEL_PLATES, ("material", "ci_material")):
        name = lb[mat_key]
        if name not in mats:
            files = {"picture": ci_png} if kind == layout.LABEL_PLATES[1] else {}
            mats[name] = _material(stage, name, spec["materials"][name], files, out.parent)
        mesh = _plates_mesh(stage, LABEL_MESHES[kind], [p for p in plates if p.kind == kind],
                            uv=kind == layout.LABEL_PLATES[1])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mats[name])
        UsdSemantics.LabelsAPI.Apply(mesh.GetPrim(), "class").CreateLabelsAttr(sorted(classes))
    # The fill: a uniform dome with its shadows off reaches every surface, the ceiling's
    # underside included. No texture -- nothing of the outside is on the stage.
    fill = UsdLux.DomeLight.Define(stage, f"{ROOT}/{FILL_LIGHT}")
    fill.CreateIntensityAttr(float(lights["fill"]["intensity"]))
    fill.CreateColorAttr(Gf.Vec3f(*[float(v) for v in lights["fill"]["color"]]))
    UsdLux.ShadowAPI.Apply(fill.GetPrim()).CreateShadowEnableAttr(False)
    stage.GetRootLayer().Save()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", type=Path, default=SPEC)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    out = build(a.spec, a.out)
    print(f"{out}  sha256 {sha256_of(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
