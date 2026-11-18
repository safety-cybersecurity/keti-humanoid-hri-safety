"""Assign a visual mesh's vertices to the dummy's body-region primitives.

Nearest-primitive assignment against the same shapes anthropometry.yaml
declares (sphere / capsule / box), so the split follows the canon and not a
hand-drawn seam. A mirrored region (``mirror_y``) becomes two primitives that
share the BR code and differ in side, named exactly as build_dummy.py names
the USD prims (``BR_08_left``), so a face lands on the prim whose rigid body
it has to follow.

Pure numpy: CI tests it, make_skin.py measures the Blender export with
``read_obj`` (Blender's Python has numpy but no yaml, so nothing else here may
import more than that), and build_dummy.py --skin splits with it under Isaac's
python.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Suffixes a mirrored region's two prims carry; the +y prim is the left one
# (the dummy faces +x with +z up, so +y is its left-hand side).
SIDES = ("left", "right")
SHAPES = ("sphere", "capsule", "box")


def prim_names(code: str, spec: dict) -> list[tuple[str, float]]:
    """Prim names one region makes and the sign its centre's y takes: two if mirrored, else one."""
    stem = code.replace("-", "_")
    if not spec.get("mirror_y"):
        return [(stem, 1.0)]
    return [(f"{stem}_{side}", 1.0 if side == "left" else -1.0) for side in SIDES]


@dataclass(frozen=True)
class Primitive:
    code: str
    shape: str
    center: np.ndarray
    radius: float | None
    height: float | None
    axis: str | None
    size: np.ndarray | None
    name: str = ""                 # USD prim name; defaults to the code's stem
    side: str | None = None        # "left" / "right" for a mirrored region

    def __post_init__(self):
        if not self.name:
            object.__setattr__(self, "name", self.code.replace("-", "_"))


def _finite_positive(code: str, key: str, value) -> float:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{code}: {key} must be a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{code}: {key} must be finite and positive, got {value!r}")
    return v


def primitives_from_anthropometry(anthro: dict) -> list[Primitive]:
    """One Primitive per USD prim, in the canon's order (mirrored: left then right)."""
    regions = anthro.get("regions")
    if not isinstance(regions, dict) or not regions:
        raise ValueError("anthropometry has no regions")
    out: list[Primitive] = []
    for code, s in regions.items():
        shape = s.get("shape")
        if shape not in SHAPES:
            raise ValueError(f"{code}: unknown shape {shape!r}")
        center = np.asarray(s.get("center"), float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError(f"{code}: center must be three finite numbers")
        radius = height = axis = None
        size = None
        if shape in ("sphere", "capsule"):
            radius = _finite_positive(code, "radius", s.get("radius"))
        if shape == "capsule":
            height = _finite_positive(code, "height", s.get("height"))
            axis = str(s.get("axis", "z")).lower()
            if axis not in "xyz" or len(axis) != 1:
                raise ValueError(f"{code}: axis must be x, y or z")
        if shape == "box":
            raw = s.get("size")
            if raw is None or len(raw) != 3:
                raise ValueError(f"{code}: box needs size [sx, sy, sz]")
            size = np.array([_finite_positive(code, f"size[{i}]", v) for i, v in enumerate(raw)])
        for name, sign in prim_names(code, s):
            side = name.rsplit("_", 1)[1] if s.get("mirror_y") else None
            out.append(Primitive(code, shape, center * np.array([1.0, sign, 1.0]),
                                 radius, height, axis, size, name=name, side=side))
    return out


def distance(V: np.ndarray, p: Primitive) -> np.ndarray:
    """Signed distance from each point to the primitive's surface (negative inside)."""
    V = np.asarray(V, float)
    d = V - p.center
    if p.shape == "sphere":
        return np.linalg.norm(d, axis=1) - p.radius
    if p.shape == "box":
        q = np.abs(d) - p.size / 2
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    if p.shape == "capsule":
        # UsdGeom.Capsule: `height` is the cylindrical section, the caps add a radius each end.
        k = "xyz".index(p.axis)
        half = p.height / 2
        a = d[:, k]
        d2 = d.copy()
        d2[:, k] = a - np.clip(a, -half, half)
        return np.linalg.norm(d2, axis=1) - p.radius
    raise ValueError(p.shape)


def bottom_of(p: Primitive) -> float:
    """The lowest z the primitive reaches."""
    if p.shape == "sphere":
        return float(p.center[2] - p.radius)
    if p.shape == "box":
        return float(p.center[2] - p.size[2] / 2)
    half = p.height / 2 if p.axis == "z" else 0.0
    return float(p.center[2] - half - p.radius)


def top_of(p: Primitive) -> float:
    """The highest z the primitive reaches."""
    if p.shape == "sphere":
        return float(p.center[2] + p.radius)
    if p.shape == "box":
        return float(p.center[2] + p.size[2] / 2)
    half = p.height / 2 if p.axis == "z" else 0.0
    return float(p.center[2] + half + p.radius)


def assign_primitives(V: np.ndarray, prims: list[Primitive]) -> np.ndarray:
    """Index (into prims) of the nearest primitive for every point."""
    D = np.stack([distance(V, p) for p in prims], axis=1)
    return D.argmin(axis=1)


def assign_vertices(V: np.ndarray, prims: list[Primitive]) -> np.ndarray:
    """Region code of the nearest primitive for every point."""
    codes = np.array([p.code for p in prims])
    return codes[assign_primitives(V, prims)]


def face_labels(F: np.ndarray, labels: np.ndarray, V: np.ndarray | None = None,
                prims: list[Primitive] | None = None, label_of=None) -> np.ndarray:
    """A label per face: the majority of its three vertices. A three-way split is
    settled by the centroid's nearest primitive when V and prims are given, else
    by the first vertex."""
    labels = np.asarray(labels)
    fl = labels[F]
    two_of_three = np.where(fl[:, 1] == fl[:, 2], fl[:, 1], fl[:, 0])
    out = np.where((fl[:, 0] == fl[:, 1]) | (fl[:, 0] == fl[:, 2]), fl[:, 0], two_of_three)
    if V is not None and prims is not None:
        split = (fl[:, 0] != fl[:, 1]) & (fl[:, 1] != fl[:, 2]) & (fl[:, 0] != fl[:, 2])
        if split.any():
            label_of = label_of or (lambda p: p.code)
            names = np.array([label_of(p) for p in prims], dtype=out.dtype)
            centroids = np.asarray(V, float)[F[split]].mean(axis=1)
            out = out.copy()
            out[split] = names[assign_primitives(centroids, prims)]
    return out


def split_faces(F: np.ndarray, labels: np.ndarray, V: np.ndarray | None = None,
                prims: list[Primitive] | None = None, label_of=None
                ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """faces -> {label: (faces re-indexed into that label's vertex subset, old vertex ids,
    original face ids)}.

    Every face goes to exactly one label. `old[new_index]` is the original vertex id,
    so callers slice V with `old`; `face_ids` index the original faces, so per-face
    data (uv / normal corners, material) is sliced with them."""
    return split_by_face(F, face_labels(np.asarray(F, np.int64), labels, V, prims, label_of))


def split_by_face(F: np.ndarray, fl: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """faces grouped by a label per face (split_faces once face_labels has run) -> the same
    {label: (faces re-indexed, old vertex ids, original face ids)}."""
    F = np.asarray(F, np.int64)
    fl = np.asarray(fl)
    if fl.shape != (len(F),):
        raise ValueError(f"face labels must be one per face: {len(F)} faces, got {fl.shape}")
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label in np.unique(fl):
        face_ids = np.nonzero(fl == label)[0]
        faces = F[face_ids]
        old = np.unique(faces)
        remap = -np.ones(int(F.max()) + 1, dtype=np.int64)
        remap[old] = np.arange(len(old))
        out[str(label)] = (remap[faces], old, face_ids)
    return out


# ---- OBJ / MTL ---------------------------------------------------------------------

@dataclass
class ObjMesh:
    V: np.ndarray                      # (n, 3) positions
    VT: np.ndarray                     # (t, 2) uvs, may be empty
    VN: np.ndarray                     # (k, 3) normals, may be empty
    F: np.ndarray                      # (m, 3) vertex ids, triangles
    FT: np.ndarray                     # (m, 3) uv ids or -1
    FN: np.ndarray                     # (m, 3) normal ids or -1
    FM: np.ndarray                     # (m,) material index or -1
    FO: np.ndarray                     # (m,) object index or -1
    materials: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    mtllib: str | None = None

    def validate(self) -> None:
        """Refuse what the reader accepts syntactically but no mesh may carry: no
        vertices or faces, non-finite coordinates, corner indices past their arrays,
        material or object indices past their lists (a NaN vertex went through
        attachment and verification unnoticed)."""
        if len(self.V) == 0 or len(self.F) == 0:
            raise ValueError("mesh has no vertices or no faces")
        for name, arr in (("positions", self.V), ("uvs", self.VT), ("normals", self.VN)):
            if len(arr) and not np.all(np.isfinite(arr)):
                raise ValueError(f"non-finite {name}")
        if self.F.min() < 0 or self.F.max() >= len(self.V):
            raise ValueError("face vertex index out of range")
        for name, idx, arr in (("uv", self.FT, self.VT), ("normal", self.FN, self.VN)):
            used = idx[idx >= 0]
            if len(used) and used.max() >= len(arr):
                raise ValueError(f"face {name} index out of range")
        for name, idx, items in (("material", self.FM, self.materials), ("object", self.FO, self.objects)):
            used = idx[idx >= 0]
            if len(used) and used.max() >= len(items):
                raise ValueError(f"face {name} index out of range")


def read_obj(path: str | Path) -> ObjMesh:
    """Read a Wavefront OBJ: positions, uvs, normals, triangulated faces, and which
    object / material each face belongs to. Polygons are fan-triangulated."""
    V, VT, VN, F, FT, FN, FM, FO = [], [], [], [], [], [], [], []
    materials: list[str] = []
    objects: list[str] = []
    mtllib = None
    cur_m = cur_o = -1
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        t = line.split()
        if not t or t[0].startswith("#"):
            continue
        k = t[0]
        if k == "v":
            V.append([float(x) for x in t[1:4]])
        elif k == "vt":
            VT.append([float(x) for x in t[1:3]])
        elif k == "vn":
            VN.append([float(x) for x in t[1:4]])
        elif k == "mtllib":
            mtllib = " ".join(t[1:])
        elif k == "o":
            objects.append(" ".join(t[1:]))
            cur_o = len(objects) - 1
        elif k == "usemtl":
            name = " ".join(t[1:])
            if name not in materials:
                materials.append(name)
            cur_m = materials.index(name)
        elif k == "f":
            idx = [w.split("/") for w in t[1:]]

            def vid(w):
                return int(w[0]) - 1

            def tid(w):
                return int(w[1]) - 1 if len(w) > 1 and w[1] else -1

            def nid(w):
                return int(w[2]) - 1 if len(w) > 2 and w[2] else -1

            for i in range(1, len(idx) - 1):
                tri = (idx[0], idx[i], idx[i + 1])
                F.append([vid(w) for w in tri])
                FT.append([tid(w) for w in tri])
                FN.append([nid(w) for w in tri])
                FM.append(cur_m)
                FO.append(cur_o)
    return ObjMesh(np.asarray(V, float).reshape(-1, 3), np.asarray(VT, float).reshape(-1, 2),
                   np.asarray(VN, float).reshape(-1, 3), np.asarray(F, np.int64).reshape(-1, 3),
                   np.asarray(FT, np.int64).reshape(-1, 3), np.asarray(FN, np.int64).reshape(-1, 3),
                   np.asarray(FM, np.int64), np.asarray(FO, np.int64), materials, objects, mtllib)


@dataclass
class MtlMaterial:
    name: str
    kd: tuple[float, float, float] = (0.8, 0.8, 0.8)
    d: float = 1.0
    pr: float | None = None            # roughness (Blender's PBR extension)
    map_kd: str | None = None
    map_bump: str | None = None
    bump_strength: float = 1.0


def read_mtl(path: str | Path) -> dict[str, MtlMaterial]:
    """Read the maps and scalars build_dummy needs from a Wavefront MTL."""
    mats: dict[str, MtlMaterial] = {}
    cur: MtlMaterial | None = None
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        t = line.split()
        if not t or t[0].startswith("#"):
            continue
        k = t[0]
        if k == "newmtl":
            cur = MtlMaterial(" ".join(t[1:]))
            mats[cur.name] = cur
        elif cur is None:
            continue
        elif k == "Kd":
            cur.kd = tuple(_finite_scalar(cur.name, "Kd", x) for x in t[1:4])
        elif k == "d":
            cur.d = _finite_scalar(cur.name, "d", t[1])
        elif k == "Pr":
            cur.pr = _finite_scalar(cur.name, "Pr", t[1])
        elif k == "map_Kd":
            cur.map_kd = " ".join(t[1:])
        elif k in ("map_Bump", "bump", "norm"):
            args = t[1:]
            if "-bm" in args:
                i = args.index("-bm")
                cur.bump_strength = _finite_scalar(cur.name, "-bm", args[i + 1])
                args = args[:i] + args[i + 2:]
            cur.map_bump = " ".join(args)
    return mats


def _finite_scalar(material: str, key: str, text: str) -> float:
    """MTL scalars are authored straight into shader inputs; `Pr nan` reached the USD
    unnoticed."""
    v = float(text)
    if not math.isfinite(v):
        raise ValueError(f"material {material!r}: {key} is not finite ({text})")
    return v


# ---- measurements make_skin.py takes on the export ------------------------------------

def stature_of(V: np.ndarray) -> float:
    V = np.asarray(V, float)
    return float(V[:, 2].max() - V[:, 2].min())


def orientation_report(V: np.ndarray, foot_height: float = 0.06) -> dict:
    """Does the mesh stand in the dummy frame (faces +x, +z up, symmetric in y)?

    The feet decide the facing: from the waist's mid-plane (the hips stand over the
    ankles), the toes reach further forward than the heel reaches back, so on a
    +x-facing figure the foot band's forward extent is the longer one. Neither the
    whole mesh's median nor the waist's is a usable reference — the clothes' delete
    groups strip the body under them, and the belly's front carries more vertices
    than the back — so the reference is the middle of the torso's x-extent."""
    V = np.asarray(V, float)
    zmin = V[:, 2].min()
    stature = float(V[:, 2].max() - zmin)
    feet = V[V[:, 2] <= zmin + foot_height]
    # the torso only: hands hanging by the sides reach into this band too
    waist = V[(V[:, 2] > zmin + 0.5 * stature) & (V[:, 2] < zmin + 0.6 * stature) & (np.abs(V[:, 1]) < 0.15)]
    if len(waist) == 0 or len(feet) == 0:
        raise ValueError("mesh has no vertices in the waist or foot band — not a standing figure")
    x_center = float((waist[:, 0].min() + waist[:, 0].max()) / 2)
    front = float(feet[:, 0].max() - x_center)
    back = float(x_center - feet[:, 0].min())
    y_mean = float(V[:, 1].mean())
    y_span = (float(V[:, 1].min()), float(V[:, 1].max()))
    return {"faces_plus_x": front > back, "foot_front_m": front, "foot_back_m": back,
            "y_mean": y_mean, "y_span": y_span, "z_min": float(zmin), "x_center": x_center}
