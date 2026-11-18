#!/usr/bin/env python3
"""The box the robot carries: plain cardboard, a white label on four sides and the top, the KETI CI on each.

    .venv/bin/python generation/scene/fetch_keti_ci.py --archive <KETI_CI.zip>   # the CI, pinned by keti_ci.json
    PYTHONPATH= ~/isaacsim/python.sh generation/scene/carried_box.py              # -> build/keti_carried_box_v1.usd

Everything above the pxr section runs without Isaac: the label layout (label_rects), the per-prim
overrides that resize the box (box_overrides) and the PNG header readers (png_size, png_has_alpha -- the
warehouse's box labels read the same picture with them). write_box_usd,
apply_box_size and main() need pxr.

The body is a unit cube mesh scaled to the box size. Each label is a plate authored at its real size
and only moved when the box is resized, so a label never stretches -- apply_box_size scales the body
and moves the plates to the middle of their faces. The KETI CI is the official PNG copied byte for
byte beside the USD (./textures/<file>) and cut out by its alpha, the way the dummy's hair cards are.
The body is the one collider, approximated by its box; mass and the rigid body are left to whoever
carries the box. Every number here is the project's choice (self-set) -- none is a standard value.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import numbers
import shutil
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Width fixed at 0.40 m; the carried range is 0.25-0.30 m deep and 0.20-0.30 m tall, and the
# nominal box is its middle. A label is a white card with the CI on it, on the middle of a face.
NOMINAL_SIZE_M = (0.40, 0.275, 0.25)
LABEL_M = (0.20, 0.07)                 # width x height
CI_WIDTH_M = 0.18                      # its height follows the picture's aspect
LABEL_LIFT_M = 0.001                   # the label 1 mm off the face
CI_LIFT_M = 0.002                      # the CI 1 mm above the label
BODY_SRGB = (179 / 255, 141 / 255, 81 / 255)
BODY_ROUGHNESS = 0.85
LABEL_WHITE = 0.95
LABEL_ROUGHNESS = 0.85                 # matte paper, like the body

ROOT = "/CarriedBox"
BODY = "body"
LABELS = "labels"
PLATE_KINDS = ("base", "ci")           # the white label, then the CI on it
OUT = REPO / "generation/scene/build/keti_carried_box_v1.usd"
TEXTURE_SUBDIR = "textures"
BUILT_BY = "generation/scene/carried_box.py"
BOX_VERSION = "1"

# Face -> (outward normal, the label's up). The label's width runs along up x normal: the
# viewer's right when facing that side. On the top, up is +y, so the width runs along x.
FACES = ("+x", "-x", "+y", "-y", "+z")
_FRAMES = {
    "+x": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "-x": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "+y": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "-y": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "+z": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
}
_BOTTOM = ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0))   # the body's sixth face; it carries no label
FACE_PRIMS = {"+x": "pos_x", "-x": "neg_x", "+y": "pos_y", "-y": "neg_y", "+z": "pos_z"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class LabelRect:
    face: str               # "+x" "-x" "+y" "-y" "+z"
    center: np.ndarray      # box frame, the box's centre at the origin (m)
    normal: np.ndarray      # outward
    up: np.ndarray          # the plate's height runs along this; its width along up x normal
    width: float
    height: float


def _size(size_m: Sequence[float]) -> np.ndarray:
    values = list(size_m) if not isinstance(size_m, (str, bytes)) else []
    if len(values) != 3 or not all(isinstance(v, numbers.Real) and not isinstance(v, bool)
                                   and math.isfinite(v) and v > 0 for v in values):
        raise ValueError(f"상자 크기는 양의 유한한 길이 셋(폭 · 깊이 · 높이 m)이어야 합니다: {size_m!r}")
    return np.array(values, dtype=float)


def _frame(face: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal, up = (np.array(v) for v in _FRAMES[face])
    return normal, up, np.cross(up, normal)


def _plates(size: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    """(face, kind, centre) for every plate, face by face, the label before its CI. Refuses a box
    a label does not fit on."""
    out = []
    for face in FACES:
        normal, up, right = _frame(face)
        room = (float(np.abs(right) @ size), float(np.abs(up) @ size))
        if LABEL_M[0] > room[0] or LABEL_M[1] > room[1]:
            raise ValueError(f"{face} 면({room[0]:.3f} × {room[1]:.3f} m)에 라벨 "
                             f"{LABEL_M[0]:.2f} × {LABEL_M[1]:.2f} m 이 들어가지 않습니다 — 상자 크기 {tuple(size)}")
        half = float(np.abs(normal) @ size) / 2.0
        out.append((face, PLATE_KINDS[0], normal * (half + LABEL_LIFT_M)))
        out.append((face, PLATE_KINDS[1], normal * (half + CI_LIFT_M)))
    return out


def label_rects(size_m: Sequence[float], ci_aspect: float) -> list[LabelRect]:
    """The ten plates of a box of ``size_m`` (width, depth, height): on each of +x -x +y -y +z, the
    white label then the CI, both centred on the face. ``ci_aspect`` is the CI picture's width /
    height in pixels (png_size). Raises ValueError when a label does not fit its face or the CI
    its label."""
    size = _size(size_m)
    aspect = float(ci_aspect)
    if not math.isfinite(aspect) or aspect <= 0:
        raise ValueError(f"CI 그림의 가로 / 세로 비가 양의 유한한 수가 아닙니다: {ci_aspect!r}")
    ci_height = CI_WIDTH_M / aspect
    if CI_WIDTH_M > LABEL_M[0] or ci_height > LABEL_M[1]:
        raise ValueError(f"CI {CI_WIDTH_M:.3f} × {ci_height:.3f} m 가 라벨 {LABEL_M[0]:.2f} × {LABEL_M[1]:.2f} m "
                         "안에 들어가지 않습니다")
    out = []
    for face, kind, centre in _plates(size):
        normal, up, _ = _frame(face)
        w, h = (LABEL_M if kind == PLATE_KINDS[0] else (CI_WIDTH_M, ci_height))
        out.append(LabelRect(face, centre, normal, up, float(w), float(h)))
    return out


def plate_corners(rect: LabelRect) -> np.ndarray:
    """The plate's four corners relative to its centre, counter-clockwise seen from outside
    (bottom-left, bottom-right, top-right, top-left) -- the order the mesh and its UVs use."""
    right = np.cross(rect.up, rect.normal) * (rect.width / 2.0)
    up = np.asarray(rect.up, dtype=float) * (rect.height / 2.0)
    # + 0.0 turns a negated zero into 0.0, so the USD does not carry -0 coordinates
    return np.array([-right - up, right - up, right + up, -right + up]) + 0.0


def plate_path(face: str, kind: str) -> str:
    return f"{LABELS}/{FACE_PRIMS[face]}/{kind}"


def box_overrides(size_m: Sequence[float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Prim path under the box root -> (translate, scale) for a box of ``size_m``. Only the body
    is scaled; every plate keeps scale 1 and moves to the middle of its face."""
    size = _size(size_m)
    out = {BODY: (np.zeros(3), size.copy())}
    for face, kind, centre in _plates(size):
        out[plate_path(face, kind)] = (centre, np.ones(3))
    return out


def png_size(path: Path) -> tuple[int, int]:
    """(width, height) from a PNG's header -- the first 24 bytes, nothing decoded."""
    with open(path, "rb") as fh:
        head = fh.read(26)
    if len(head) < 24 or head[:8] != PNG_SIGNATURE or head[12:16] != b"IHDR":
        raise ValueError(f"{path}: PNG 머리(서명과 IHDR)가 아닙니다")
    width, height = struct.unpack(">II", head[16:24])
    if not width or not height:
        raise ValueError(f"{path}: PNG 의 크기가 0 입니다 ({width} × {height})")
    return width, height


def png_has_alpha(path: Path) -> bool:
    """IHDR colour type 4 (grey + alpha) or 6 (RGBA)."""
    with open(path, "rb") as fh:
        head = fh.read(26)
    return len(head) >= 26 and head[25] in (4, 6)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- the USD (pxr) ----------------------------------------------------------------------------

def _quad_mesh(stage, path: str, corners: np.ndarray, normal: np.ndarray, uv: bool):
    from pxr import Gf, Sdf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = [Gf.Vec3f(*map(float, p)) for p in corners]
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * (len(pts) // 4)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(len(pts)))))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, n)) for n in normal]))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
    if uv:
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                      UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    xf = UsdGeom.Xformable(mesh)
    xf.AddTranslateOp()
    xf.AddScaleOp()
    return mesh


def _omnipbr(stage, name: str, *, color=None, texture: str | None = None, cutout: bool = False,
             roughness: float):
    from pxr import Gf, Sdf, UsdShade

    path = f"{ROOT}/Looks/{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
    if color is not None:
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
    if texture is not None:
        sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(texture)
        if cutout:
            # the same alpha cut-out the dummy's hair cards use (OmniPBR mono mode 0 = alpha)
            sh.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
            sh.CreateInput("enable_opacity_texture", Sdf.ValueTypeNames.Bool).Set(True)
            sh.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(texture)
            sh.CreateInput("opacity_mode", Sdf.ValueTypeNames.Int).Set(0)
            sh.CreateInput("opacity_threshold", Sdf.ValueTypeNames.Float).Set(0.5)
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(out)
    return mat


def apply_box_size(stage, root_path: str, size_m: Sequence[float]) -> None:
    """Resize the box under ``root_path`` on ``stage`` to ``size_m``: the body scaled, every label
    moved to the middle of its face and left unscaled. Authors on the stage's edit target, so a
    stage that references the box resizes its own copy and the box file is untouched. Raises
    ValueError when a prim or its translate / scale op is missing (not a carried-box USD)."""
    from pxr import Gf, UsdGeom

    for rel, (translate, scale) in box_overrides(size_m).items():
        path = f"{root_path}/{rel}"
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError(f"{path} 가 없습니다 — 들고 다니는 상자 USD 의 프림이 아닙니다")
        ops = {op.GetOpType(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
        if UsdGeom.XformOp.TypeTranslate not in ops or UsdGeom.XformOp.TypeScale not in ops:
            raise ValueError(f"{path} 에 옮김 · 늘임 변환이 없습니다 — 들고 다니는 상자 USD 의 프림이 아닙니다")
        ops[UsdGeom.XformOp.TypeTranslate].Set(Gf.Vec3d(*map(float, translate)))
        ops[UsdGeom.XformOp.TypeScale].Set(Gf.Vec3f(*map(float, scale)))


def write_box_usd(path: Path, size_m: Sequence[float], ci_png: Path) -> None:
    """Build the box USD at ``path`` for a box of ``size_m``, copying ``ci_png`` byte for byte to
    ``path.parent/textures/`` under its own name. The caller has checked the PNG is the pinned
    one (main() does); this reads only its size and alpha."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    path, ci_png = Path(path), Path(ci_png)
    width, height = png_size(ci_png)
    rects = label_rects(size_m, width / height)
    texture = f"./{TEXTURE_SUBDIR}/{ci_png.name}"
    dst = path.parent / TEXTURE_SUBDIR / ci_png.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.is_file() or dst.read_bytes() != ci_png.read_bytes():
        shutil.copy2(ci_png, dst)

    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    rp = root.GetPrim()
    rp.SetCustomDataByKey("keti:builtBy", BUILT_BY)
    rp.SetCustomDataByKey("keti:boxVersion", BOX_VERSION)
    rp.SetCustomDataByKey("keti:ciFile", ci_png.name)
    rp.SetCustomDataByKey("keti:ciSha256", _sha256(ci_png))
    UsdGeom.Scope.Define(stage, f"{ROOT}/Looks")
    cardboard = _omnipbr(stage, "cardboard", color=BODY_SRGB, roughness=BODY_ROUGHNESS)
    white = _omnipbr(stage, "label_white", color=(LABEL_WHITE,) * 3, roughness=LABEL_ROUGHNESS)
    ci = _omnipbr(stage, "keti_ci", texture=texture, cutout=png_has_alpha(ci_png),
                  roughness=LABEL_ROUGHNESS)

    # The body: a unit cube, four corners per face so every face is flat-shaded, scaled to the box.
    corners, normals = [], []
    for normal, up in [_FRAMES[f] for f in FACES] + [_BOTTOM]:
        n, u = np.array(normal), np.array(up)
        unit = LabelRect("", n * 0.5, n, u, 1.0, 1.0)
        corners.extend(unit.center + plate_corners(unit))
        normals.extend([n] * 4)
    body = _quad_mesh(stage, f"{ROOT}/{BODY}", np.array(corners), np.array(normals), uv=False)
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(cardboard)
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(body.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.boundingCube)

    UsdGeom.Scope.Define(stage, f"{ROOT}/{LABELS}")
    for i, rect in enumerate(rects):
        kind = PLATE_KINDS[i % 2]
        face_path = f"{ROOT}/{LABELS}/{FACE_PRIMS[rect.face]}"
        UsdGeom.Xform.Define(stage, face_path)
        plate = _quad_mesh(stage, f"{face_path}/{kind}", plate_corners(rect), np.array([rect.normal] * 4),
                           uv=kind == "ci")
        UsdShade.MaterialBindingAPI.Apply(plate.GetPrim()).Bind(ci if kind == "ci" else white)
    apply_box_size(stage, ROOT, size_m)
    stage.GetRootLayer().Save()


def main(argv: list[str] | None = None) -> int:
    from generation.scene import fetch_keti_ci

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    lock = fetch_keti_ci.load_ci_lock()
    src = fetch_keti_ci.OUT / lock["file"]
    if not src.is_file() or _sha256(src) != lock["member_sha256"]:
        raise SystemExit(f"KETI CI 가 없거나 고정값과 다릅니다: {src} — fetch_keti_ci.py 를 먼저 돌리십시오")
    write_box_usd(a.out, NOMINAL_SIZE_M, src)
    print(f"{a.out}  sha256 {_sha256(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
