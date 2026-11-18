"""The jointed dummy's hi-vis vest (generation/dummy/vest.yaml): read, checked, and made over the skin.

Pure numpy and yaml, no pxr. build_dummy.py --vest reads the spec here, makes the vest over the skin OBJ it
has just split into body parts, and attaches one piece under each trunk body it covers (attach_vest).

The vest is a smooth shell laid over the clothes on the trunk, then cut to a vest:

1. Rays leave a point inside the chest (shell.centre_m) on a regular grid of azimuth and elevation and meet
   the source object's faces that the skinned build puts on the spec's regions. A ray takes the outermost
   face within layer_m of the first one it meets -- the T-shirt, the overall over it and the overall's
   buckles are one layer; a sleeve farther out is not the trunk's.
2. A ray that met nothing takes the horizontal distances (from the vertical axis through the centre) of the
   rays round it, diffused in, with the axis beyond the top row: over the neck opening the shell closes in
   instead of rising along the neck. Then the distances along the rays are draped drape_passes times --
   each time two 3 x 3 means, clamped between the clothes and drape_max_m beyond them -- so the shell lies
   over the buckles and pocket flaps instead of following them; where the rays meet the clothes slantwise
   (down at the hem) the means lift it a little off them, as a vest hangs loose. Then it is smoothed with
   smooth_passes 3 x 3 means, still never below the clothes.
3. Every grid point goes offset_m farther out along its ray than the draped distance. The grid is the
   shell's mesh (it wraps round in azimuth); its normals are area-weighted over the whole shell.
4. Grid vertices go to the nearest primitive of the spec's regions and faces to the majority of their three
   vertices -- the rule the skinned build splits the skin by.
5. The shell is cut by the hem and the band edges (horizontal planes: a point made on one gets its height
   exactly), the two armholes (an ellipsoid and its mirror in y), the neck (a vertical cylinder) and the
   V-neck (the front, x > front_x_m, above two planes rising from bottom_z_m). A point made on an edge
   solves the cut exactly along the edge, takes the edge's normals interpolated and normalised, and is made
   from the edge's two ends in one fixed order, so the faces sharing the edge make the same point.

A cut face lies on one side of every cut. It stays unless it is below the hem, inside an armhole or the
neck, or in the V; it takes the band material between a band's two planes. Coordinates are the dummy's
frame, the skin's: the build moves them into each body's frame.

At every grid point the shell lies at least offset_m beyond the clothes its ray met (the drape may add to
that); between the points it is the flat faces joining them. The checks refuse a vest that would not be one:
a centre outside the clothes (a ray of the row nearest the horizontal meets nothing), a grid whose lowest row
is not below the hem everywhere (the vest would end in the grid's ragged row, not the hem), any of the grid's
highest row left after the cuts, a hem, armhole or neck that cuts nothing (off the shell, or too small for
the grid to show), and a spec region left without a face.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from generation.dummy.skin import region_split as rs

VEST_SPEC = Path(__file__).with_name("vest.yaml")
YELLOW = "vest_yellow"
BAND = "vest_band"
MATERIALS = (YELLOW, BAND)
SPEC_KEYS = {"vest_version", "license", "determined_by", "source_object", "regions", "shell", "offset_m",
             "hem_z_m", "bands_z_m", "armhole", "neck", "v_neck", "materials"}
SHELL_KEYS = {"centre_m", "azimuth_steps", "elevation_deg", "elevation_steps", "layer_m", "drape_passes",
              "drape_max_m", "smooth_passes"}
ARMHOLE_KEYS = {"centre_m", "radii_m"}
NECK_KEYS = {"centre_m", "radius_m"}
V_NECK_KEYS = {"front_x_m", "bottom_z_m", "slope"}
MATERIAL_KEYS = {"color", "roughness", "metallic"}
# Jacobi passes that carry the measured distances into the rays that met nothing
FILL_PASSES = 500
# rays per batch of the ray-triangle test (rays x faces x 3 doubles at a time)
RAY_BATCH = 128
# a ray down the edge two faces share meets both: barycentric coordinates this far outside a face still count --
# a ray slipping between the two would take a farther garment for the first it meets
EDGE_SLACK = 1e-12
# a point this near a cut (its value, metres for a plane, a fraction for an ellipsoid or a cylinder) is on it:
# an edge from it to the other side is not cut a hair's breadth from its end into a face of no size
ON_CUT = 1e-12


@dataclass(frozen=True)
class VestSpec:
    source_object: str
    regions: tuple[str, ...]
    centre: tuple[float, float, float]
    azimuth_steps: int
    elevation_deg: tuple[float, float]
    elevation_steps: int
    layer_m: float
    drape_passes: int
    drape_max_m: float
    smooth_passes: int
    offset_m: float
    hem_z_m: float
    bands_z_m: tuple[tuple[float, float], ...]
    armhole_centre: tuple[float, float, float]
    armhole_radii: tuple[float, float, float]
    neck_centre: tuple[float, float]
    neck_radius: float
    v_front_x: float
    v_bottom_z: float
    v_slope: float
    materials: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class VestPiece:
    prim: str             # the body prim the piece hangs under, "BR_05"
    V: np.ndarray         # (n, 3) vertices, dummy frame
    F: np.ndarray         # (m, 3) triangles into V, wound outward
    N: np.ndarray         # (n, 3) unit normals of the shell
    material: np.ndarray  # (m,) YELLOW or BAND per face


# ---- vest.yaml ------------------------------------------------------------------------------------

def _number(where: str, value, *, positive: bool = False, unit: bool = False, nonnegative: bool = False) -> float:
    # bool is an int in Python; a vest dimension of True is a typo, not 1
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"vest.yaml {where}: 수가 아닙니다 ({value!r})")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"vest.yaml {where}: 유한한 수가 아닙니다 ({value!r})")
    if positive and v <= 0:
        raise ValueError(f"vest.yaml {where}: 0 보다 커야 합니다 ({value!r})")
    if nonnegative and v < 0:
        raise ValueError(f"vest.yaml {where}: 0 이상이어야 합니다 ({value!r})")
    if unit and not 0.0 <= v <= 1.0:
        raise ValueError(f"vest.yaml {where}: 0 과 1 사이여야 합니다 ({value!r})")
    return v


def _count(where: str, value, *, least: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"vest.yaml {where}: 정수가 아닙니다 ({value!r})")
    if value < least:
        raise ValueError(f"vest.yaml {where}: {least} 이상이어야 합니다 ({value!r})")
    return value


def _numbers(where: str, value, n: int, **kw) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"vest.yaml {where}: 수 {n} 개의 목록이어야 합니다 ({value!r})")
    return tuple(_number(where, v, **kw) for v in value)


def _text(raw: dict, key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"vest.yaml {key}: 비어 있지 않은 글이어야 합니다 ({value!r})")
    return value


def _keys(where: str, raw, keys: set[str]) -> dict:
    """The mapping at ``where`` with exactly ``keys``: a misspelt key would otherwise leave a value unread."""
    if not isinstance(raw, dict):
        raise ValueError(f"vest.yaml {where}: 매핑이 아닙니다 ({raw!r})")
    unknown = sorted(set(raw) - keys)
    if unknown:
        raise ValueError(f"vest.yaml {where}: 모르는 키 {unknown} — 아는 키: {sorted(keys)}")
    missing = sorted(keys - set(raw))
    if missing:
        raise ValueError(f"vest.yaml {where}: 키가 없습니다 {missing}")
    return raw


def load_vest_spec(path: Path = VEST_SPEC) -> VestSpec:
    """The vest spec, checked. ValueError names the key that is wrong; an unknown key is refused too."""
    raw = _keys("(최상위)", yaml.safe_load(Path(path).read_text(encoding="utf-8")), SPEC_KEYS)
    for key in ("vest_version", "license", "determined_by"):
        _text(raw, key)
    source = _text(raw, "source_object")

    regions = raw["regions"]
    if not isinstance(regions, list) or not regions or not all(isinstance(r, str) and r for r in regions):
        raise ValueError(f"vest.yaml regions: 부위 코드의 비어 있지 않은 목록이어야 합니다 ({regions!r})")
    if len(set(regions)) != len(regions):
        raise ValueError(f"vest.yaml regions: 같은 부위가 두 번 있습니다 ({regions!r})")

    shell = _keys("shell", raw["shell"], SHELL_KEYS)
    centre = _numbers("shell.centre_m", shell["centre_m"], 3)
    azimuth = _count("shell.azimuth_steps", shell["azimuth_steps"], least=3)
    elevation = _numbers("shell.elevation_deg", shell["elevation_deg"], 2)
    if not -90.0 < elevation[0] < 0.0 < elevation[1] < 90.0:
        raise ValueError(f"vest.yaml shell.elevation_deg: -90 < 아래 < 0 < 위 < 90 이어야 합니다 ({shell['elevation_deg']!r})")
    rows = _count("shell.elevation_steps", shell["elevation_steps"], least=2)
    layer = _number("shell.layer_m", shell["layer_m"], nonnegative=True)
    drape = _count("shell.drape_passes", shell["drape_passes"], least=0)
    drape_max = _number("shell.drape_max_m", shell["drape_max_m"], nonnegative=True)
    smooth = _count("shell.smooth_passes", shell["smooth_passes"], least=0)

    offset = _number("offset_m", raw["offset_m"], positive=True)
    hem = _number("hem_z_m", raw["hem_z_m"])
    bands_raw = raw["bands_z_m"]
    if not isinstance(bands_raw, list):
        raise ValueError(f"vest.yaml bands_z_m: [아래, 위] 의 목록이어야 합니다 ({bands_raw!r})")
    bands = []
    for i, band in enumerate(bands_raw):
        if not isinstance(band, list) or len(band) != 2:
            raise ValueError(f"vest.yaml bands_z_m[{i}]: [아래, 위] 여야 합니다 ({band!r})")
        lo, hi = (_number(f"bands_z_m[{i}]", v) for v in band)
        if not lo < hi:
            raise ValueError(f"vest.yaml bands_z_m[{i}]: 아래가 위보다 작아야 합니다 ({band!r})")
        bands.append((lo, hi))

    armhole = _keys("armhole", raw["armhole"], ARMHOLE_KEYS)
    arm_centre = _numbers("armhole.centre_m", armhole["centre_m"], 3)
    if not arm_centre[1] > 0:
        raise ValueError(f"vest.yaml armhole.centre_m: 왼쪽(+y) 진동둘레여야 합니다 — 오른쪽은 y 를 뒤집어 만듭니다 "
                         f"({armhole['centre_m']!r})")
    arm_radii = _numbers("armhole.radii_m", armhole["radii_m"], 3, positive=True)
    neck = _keys("neck", raw["neck"], NECK_KEYS)
    neck_centre = _numbers("neck.centre_m", neck["centre_m"], 2)
    neck_radius = _number("neck.radius_m", neck["radius_m"], positive=True)
    v_neck = _keys("v_neck", raw["v_neck"], V_NECK_KEYS)
    v_front = _number("v_neck.front_x_m", v_neck["front_x_m"])
    v_bottom = _number("v_neck.bottom_z_m", v_neck["bottom_z_m"])
    v_slope = _number("v_neck.slope", v_neck["slope"], positive=True)

    mats_raw = raw["materials"]
    if not isinstance(mats_raw, dict) or set(mats_raw) != set(MATERIALS):
        got = sorted(mats_raw) if isinstance(mats_raw, dict) else mats_raw
        raise ValueError(f"vest.yaml materials: 재질은 {list(MATERIALS)} 둘이어야 합니다 ({got!r})")
    materials = {}
    for name in MATERIALS:
        m = mats_raw[name]
        if not isinstance(m, dict) or set(m) != MATERIAL_KEYS:
            got = sorted(m) if isinstance(m, dict) else m
            raise ValueError(f"vest.yaml materials.{name}: 키는 {sorted(MATERIAL_KEYS)} 셋이어야 합니다 ({got!r})")
        color = m["color"]
        if not isinstance(color, list) or len(color) != 3:
            raise ValueError(f"vest.yaml materials.{name}.color: sRGB 세 값이어야 합니다 ({color!r})")
        materials[name] = {
            "color": tuple(_number(f"materials.{name}.color", c, unit=True) for c in color),
            "roughness": _number(f"materials.{name}.roughness", m["roughness"], unit=True),
            "metallic": _number(f"materials.{name}.metallic", m["metallic"], unit=True)}
    return VestSpec(source_object=source, regions=tuple(regions), centre=centre, azimuth_steps=azimuth,
                    elevation_deg=elevation, elevation_steps=rows, layer_m=layer, drape_passes=drape,
                    drape_max_m=drape_max, smooth_passes=smooth, offset_m=offset, hem_z_m=hem, bands_z_m=tuple(bands),
                    armhole_centre=arm_centre, armhole_radii=arm_radii, neck_centre=neck_centre,
                    neck_radius=neck_radius, v_front_x=v_front, v_bottom_z=v_bottom, v_slope=v_slope,
                    materials=materials)


# ---- the cuts -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Cut:
    """f(P) = 0. A plane: f = n·P - k. A quadric: f = sum(w (P - c)^2) - 1, w = 1 / radius^2 per axis (0 for an
    axis the cut does not limit). ``level`` is set for a horizontal plane: a point made on it gets that height.
    ``on`` is the side (-1 / +1) a polygon lying wholly on the cut takes."""
    normal: np.ndarray | None = None
    k: float = 0.0
    centre: np.ndarray | None = None
    weight: np.ndarray | None = None
    level: float | None = None
    on: int = -1

    def value(self, P: np.ndarray) -> np.ndarray:
        if self.normal is not None:
            f = P @ self.normal - self.k
        else:
            f = ((P - self.centre) ** 2 * self.weight).sum(axis=-1) - 1.0
        return np.where(np.abs(f) < ON_CUT, 0.0, f)

    def crossing(self, A: np.ndarray, B: np.ndarray, fa: float, fb: float) -> float:
        """The t in (0, 1) where A + t (B - A) is on the cut; fa and fb have opposite signs."""
        if self.normal is not None:
            return fa / (fa - fb)
        d = B - A
        e = A - self.centre
        a = float((self.weight * d * d).sum())
        b = 2.0 * float((self.weight * e * d).sum())
        c = fa
        disc = max(b * b - 4.0 * a * c, 0.0)
        q = -0.5 * (b + math.copysign(math.sqrt(disc), b))
        roots = [r for r in ((q / a) if a else math.inf, (c / q) if q else math.inf) if 0.0 <= r <= 1.0]
        if not roots:                                 # rounding at an end: the end nearer the cut
            return 0.0 if abs(fa) < abs(fb) else 1.0
        return min(roots, key=lambda r: abs(r - fa / (fa - fb)))


def _plane(normal, k, level=None, on=-1) -> _Cut:
    return _Cut(normal=np.asarray(normal, float), k=float(k), level=level, on=on)


def _quadric(centre, radii) -> _Cut:
    radii = np.asarray(radii, float)
    weight = np.where(np.isfinite(radii), 1.0 / np.where(np.isfinite(radii), radii, 1.0) ** 2, 0.0)
    return _Cut(centre=np.asarray(centre, float), weight=weight, on=1)


@dataclass(frozen=True)
class _Cuts:
    cuts: tuple[_Cut, ...]
    hem: int
    bands: tuple[tuple[int, int], ...]
    armholes: tuple[int, int]
    neck: int
    v: tuple[int, int, int]

    def keep(self, s) -> bool:
        """Whether a polygon on sides ``s`` (one -1 / +1 per cut) stays in the vest."""
        if s[self.hem] < 0 or s[self.armholes[0]] < 0 or s[self.armholes[1]] < 0 or s[self.neck] < 0:
            return False
        return not all(s[i] > 0 for i in self.v)

    def band(self, s) -> bool:
        return any(s[lo] > 0 and s[hi] < 0 for lo, hi in self.bands)

    def kept_points(self, P: np.ndarray) -> np.ndarray:
        """``keep`` for single points, each on the side its value says (a point on a cut counts as outside it)."""
        f = [c.value(P) for c in self.cuts]
        keep = (f[self.hem] >= 0) & (f[self.armholes[0]] >= 0) & (f[self.armholes[1]] >= 0) & (f[self.neck] >= 0)
        return keep & ~((f[self.v[0]] > 0) & (f[self.v[1]] > 0) & (f[self.v[2]] > 0))


def _cuts(spec: VestSpec) -> _Cuts:
    """The cuts in a fixed order. A polygon lying wholly on one keeps the side a point on it keeps
    (``kept_points``): on the hem or an armhole or the neck it stays; on a band edge it goes with the slab below;
    on a plane of the V it is not in the V."""
    up = (0.0, 0.0, 1.0)
    cuts: list[_Cut] = [_plane(up, spec.hem_z_m, spec.hem_z_m, on=1)]
    bands = []
    for lo, hi in spec.bands_z_m:
        bands.append((len(cuts), len(cuts) + 1))
        cuts += [_plane(up, lo, lo), _plane(up, hi, hi)]
    left = np.asarray(spec.armhole_centre, float)
    armholes = (len(cuts), len(cuts) + 1)
    cuts += [_quadric(left, spec.armhole_radii), _quadric(left * np.array([1.0, -1.0, 1.0]), spec.armhole_radii)]
    neck = len(cuts)
    cuts.append(_quadric([*spec.neck_centre, 0.0], [spec.neck_radius, spec.neck_radius, math.inf]))
    v = (len(cuts), len(cuts) + 1, len(cuts) + 2)
    k = spec.v_slope
    cuts += [_plane((1.0, 0.0, 0.0), spec.v_front_x),
             _plane((0.0, -k, 1.0), spec.v_bottom_z), _plane((0.0, k, 1.0), spec.v_bottom_z)]
    return _Cuts(tuple(cuts), 0, tuple(bands), armholes, neck, v)


# ---- the shell ------------------------------------------------------------------------------------

def _ray_distances(origin: np.ndarray, dirs: np.ndarray, V0: np.ndarray, E1: np.ndarray, E2: np.ndarray,
                   layer: float) -> np.ndarray:
    """Per ray, the outermost face within ``layer`` of the first one it meets (Möller-Trumbore, both sides of
    a face); NaN where it meets none."""
    out = np.full(len(dirs), np.nan)
    T = origin - V0
    Q = np.cross(T, E1)
    tq = np.einsum("fk,fk->f", E2, Q)
    for i in range(0, len(dirs), RAY_BATCH):
        D = dirs[i:i + RAY_BATCH]
        P = np.cross(D[:, None, :], E2[None])
        det = np.einsum("rfk,fk->rf", P, E1)
        ok = np.abs(det) > 1e-15
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        u = np.einsum("rfk,fk->rf", P, T) * inv
        v = (D @ Q.T) * inv
        t = tq[None, :] * inv
        inside = (u >= -EDGE_SLACK) & (v >= -EDGE_SLACK) & (u + v <= 1 + EDGE_SLACK)
        t = np.where(ok & inside & (t > 0), t, np.inf)
        first = t.min(axis=1)
        outer = np.where(t <= first[:, None] + layer, t, -np.inf).max(axis=1)
        out[i:i + RAY_BATCH] = np.where(np.isfinite(first), outer, np.nan)
    return out


def _blur(R: np.ndarray, passes: int) -> np.ndarray:
    """``passes`` times the mean of each cell's 3 x 3 block; azimuth (axis 1) wraps, elevation repeats its ends."""
    R = R.copy()
    for _ in range(passes):
        a = (np.roll(R, 1, axis=1) + R + np.roll(R, -1, axis=1)) / 3.0
        R = (np.vstack([a[:1], a[:-1]]) + a + np.vstack([a[1:], a[-1:]])) / 3.0
    return R


@dataclass(frozen=True)
class _Shell:
    P: np.ndarray          # (rows * cols, 3) the shell's points, row by row (elevation) and round (azimuth)
    G: np.ndarray          # (m, 3) the grid's triangles, wound outward
    N: np.ndarray          # (rows * cols, 3) area-weighted unit normals
    rows: int
    cols: int


def _shell(spec: VestSpec, V0, E1, E2, cuts: _Cuts) -> _Shell:
    centre = np.asarray(spec.centre, float)
    rows, cols = spec.elevation_steps, spec.azimuth_steps
    azimuth = 2.0 * np.pi * np.arange(cols) / cols
    elevation = np.radians(np.linspace(spec.elevation_deg[0], spec.elevation_deg[1], rows))
    A, B = np.meshgrid(azimuth, elevation)                              # (rows, cols)
    D = np.stack([np.cos(B) * np.cos(A), np.cos(B) * np.sin(A), np.sin(B)], axis=-1).reshape(-1, 3)
    raw = _ray_distances(centre, D, V0, E1, E2, spec.layer_m).reshape(rows, cols)
    missing = np.isnan(raw)
    if missing.all():
        raise ValueError(f"어느 광선도 옷에 닿지 않습니다 — shell.centre_m {list(spec.centre)} 이 옷 안에 있고 "
                         f"shell.elevation_deg {list(spec.elevation_deg)} 가 옷을 향하는지 보십시오")
    # inside the clothes, every ray of the row nearest the horizontal meets them: the trunk is closed round the
    # chest. From outside, the rays facing away meet nothing and the near side alone would make the vest.
    ring = int(np.argmin(np.abs(elevation)))
    if missing[ring].any():
        raise ValueError(f"shell.centre_m {list(spec.centre)} 에서 수평으로 쏜 광선 {int(missing[ring].sum())} 개가 옷에 "
                         "닿지 않습니다 — 그 점이 옷 안에 있지 않습니다")
    cos_b = np.cos(B)
    # the rays that met nothing take their neighbours' horizontal distances (from the vertical axis through the
    # centre) by diffusion
    known = np.where(missing, 0.0, raw * cos_b)
    floor = np.where(missing, known[~missing].mean(), known)
    for _ in range(FILL_PASSES if missing.any() else 0):
        # beyond the top row is the axis (horizontal distance 0): over an opening at the top -- the neck --
        # the shell closes in instead of rising along it
        a = (np.roll(floor, 1, axis=1) + floor + np.roll(floor, -1, axis=1)) / 3.0
        mean = (np.vstack([a[:1], a[:-1]]) + a + np.vstack([a[1:], np.zeros((1, cols))])) / 3.0
        floor = np.where(missing, mean, known)
    # draped along the ray: where the rays meet the clothes slantwise -- down at the hem -- the averaging lifts
    # the shell off them, so the vest hangs a little loose there rather than clinging
    floor = floor / cos_b
    draped = floor.copy()
    ceiling = floor + spec.drape_max_m
    for _ in range(spec.drape_passes):
        draped = np.clip(_blur(draped, 2), floor, ceiling)
    # the last smoothing too stays on the clothes: blurred over a sharp ridge -- a shoulder's edge -- it would
    # dip below them
    draped = np.maximum(_blur(draped, spec.smooth_passes), floor)
    P = centre + (draped + spec.offset_m).reshape(-1, 1) * D

    low = P.reshape(rows, cols, 3)[0]
    if not np.all(low[:, 2] < spec.hem_z_m):
        raise ValueError(f"광선 격자의 맨 아래 줄이 아랫단 {spec.hem_z_m} m 보다 높은 곳이 있습니다(최고 "
                         f"{low[:, 2].max():.3f} m) — shell.elevation_deg 의 아래 끝을 낮추십시오")

    idx = np.arange(rows * cols).reshape(rows, cols)
    a0, a1 = idx[:-1], np.roll(idx, -1, axis=1)[:-1]
    b0, b1 = idx[1:], np.roll(idx, -1, axis=1)[1:]
    G = np.concatenate([np.stack([a0, a1, b1], -1).reshape(-1, 3), np.stack([a0, b1, b0], -1).reshape(-1, 3)])
    cross = np.cross(P[G[:, 1]] - P[G[:, 0]], P[G[:, 2]] - P[G[:, 0]])
    acc = np.zeros_like(P)
    for k in range(3):
        np.add.at(acc, G[:, k], cross)
    length = np.linalg.norm(acc, axis=1)
    if not np.all(length > 0):
        raise ValueError("법선이 없는 정점이 있습니다 — 그 정점의 면들이 서로 상쇄됩니다")
    return _Shell(P=P, G=G, N=acc / length[:, None], rows=rows, cols=cols)


# ---- the vest -------------------------------------------------------------------------------------

def vest_pieces(mesh: rs.ObjMesh, face_prims: np.ndarray, prims: Sequence[rs.Primitive],
                spec: VestSpec) -> dict[str, VestPiece]:
    """{body prim: VestPiece}, prims in name order. ``face_prims`` is the body prim of every face of the mesh as
    the skinned build splits it; ``prims`` the primitives it split by. In a piece the grid's vertices come first
    (in grid order), then the points made on edges.

    Raises ValueError when ``face_prims`` is not one prim per face, the source object is not in the mesh, a
    spec region has no primitive or the source object no face on the spec's regions, and for the vest the
    shell's checks refuse (see the module docstring)."""
    face_prims = np.asarray(face_prims)
    if face_prims.shape != (len(mesh.F),):
        raise ValueError(f"face_prims 가 면마다 하나가 아닙니다 — 면 {len(mesh.F)} · 받은 것 {face_prims.shape}")
    if spec.source_object not in mesh.objects:
        raise ValueError(f"조끼의 원본 객체 {spec.source_object!r} 가 메시에 없습니다 — 있는 것: {mesh.objects}")
    region_prims = [p for p in prims if p.code in spec.regions]
    absent = sorted(set(spec.regions) - {p.code for p in region_prims})
    if absent:
        raise ValueError(f"조끼의 부위 {absent} 의 프리미티브가 없습니다")
    code_of = {p.name: p.code for p in prims}
    region = np.array([code_of.get(str(p), "") for p in face_prims])
    clothes = (mesh.FO == mesh.objects.index(spec.source_object)) & np.isin(region, spec.regions)
    if not clothes.any():
        raise ValueError(f"조끼가 덮을 옷의 면이 없습니다 — {spec.source_object} 의 면 가운데 {list(spec.regions)} 에 든 면")
    F = mesh.F[clothes]
    V0 = mesh.V[F[:, 0]]
    cuts = _cuts(spec)
    shell = _shell(spec, V0, mesh.V[F[:, 1]] - V0, mesh.V[F[:, 2]] - V0, cuts)

    pieces, kept = _cut_pieces(shell, _grid_prims(shell.P, shell.G, region_prims), cuts)
    _check_cut(pieces, kept, shell, cuts, spec, code_of)
    return pieces


def _grid_prims(P: np.ndarray, G: np.ndarray, region_prims: Sequence[rs.Primitive]) -> np.ndarray:
    """The body prim of every grid face: each vertex to its nearest primitive, each face by the majority of its
    three vertices (a three-way split by its centroid's) -- the rule the skinned build splits the skin by."""
    names = np.array([p.name for p in region_prims])
    return rs.face_labels(G, names[rs.assign_primitives(P, region_prims)], V=P, prims=region_prims,
                          label_of=lambda p: p.name)


def _cut_pieces(shell: _Shell, grid_prims: np.ndarray, cuts: _Cuts) -> tuple[dict[str, VestPiece], set[tuple]]:
    """{body prim: VestPiece} in name order, and the keys of every point the kept faces use: (0, grid index) for a
    grid vertex, (1, end, end, cut) for a point made on an edge."""
    P, N, G = shell.P, shell.N, shell.G
    values = [c.value(P) for c in cuts.cuts]                          # every grid point's value on every cut
    made: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def at(key: tuple) -> tuple[np.ndarray, np.ndarray]:
        return (P[key[1]], N[key[1]]) if key[0] == 0 else made[key]

    def value(key: tuple, c: int) -> float:
        if key[0] == 0:
            return float(values[c][key[1]])
        if key[3] == c:
            return 0.0
        return float(cuts.cuts[c].value(made[key][0][None])[0])

    def on_edge(a: tuple, b: tuple, c: int, fa: float, fb: float) -> tuple:
        """The point where edge a-b crosses cut c, made from the edge in one order whichever face asks."""
        if b < a:
            a, b, fa, fb = b, a, fb, fa
        key = (1, a, b, c)
        if key not in made:
            (pa, na), (pb, nb) = at(a), at(b)
            t = cuts.cuts[c].crossing(pa, pb, fa, fb)
            p = pa + t * (pb - pa)
            if cuts.cuts[c].level is not None:
                p = p.copy()
                p[2] = cuts.cuts[c].level                            # exactly on the plane
            n = na + t * (nb - na)
            size = np.linalg.norm(n)
            if not size > 0:
                raise ValueError(f"{a} · {b} 사이의 법선이 서로 상쇄됩니다 — 자른 점에 법선이 없습니다")
            made[key] = (p, n / size)
        return key

    def side(lowest: float, highest: float, c: int) -> int:
        """The side of cut c a polygon with these extreme values lies on (none of them on opposite sides)."""
        return 1 if highest > 0 else -1 if lowest < 0 else cuts.cuts[c].on

    def split(poly: list[tuple], signs: tuple, c: int) -> list[tuple[list[tuple], tuple]]:
        f = [value(k, c) for k in poly]
        if min(f) >= 0 or max(f) <= 0:
            return [(poly, signs + (side(min(f), max(f), c),))]
        below, above = [], []
        for i, ka in enumerate(poly):
            kb, fa, fb = poly[(i + 1) % len(poly)], f[i], f[(i + 1) % len(poly)]
            if fa <= 0:
                below.append(ka)
            if fa >= 0:
                above.append(ka)
            if (fa < 0 < fb) or (fb < 0 < fa):
                m = on_edge(ka, kb, c, fa, fb)
                below.append(m)
                above.append(m)
        return [(part, signs + (side,)) for part, side in ((below, -1), (above, 1)) if len(part) >= 3]

    # a face no cut crosses keeps its three vertices' side of every cut
    vf = np.stack([v[G] for v in values])                              # (cuts, faces, 3)
    lowest, highest = vf.min(axis=2), vf.max(axis=2)
    crossed = ((lowest < 0) & (highest > 0)).any(axis=0)
    on = np.array([c.on for c in cuts.cuts])[:, None]
    whole_sides = np.where(highest > 0, 1, np.where(lowest < 0, -1, on)).T   # (faces, cuts)

    triangles: dict[str, list[tuple[tuple, tuple, tuple, str]]] = {}
    for fid in range(len(G)):
        if crossed[fid]:
            polys = [([(0, int(i)) for i in G[fid]], ())]
            for c in range(len(cuts.cuts)):
                polys = [piece for poly, signs in polys for piece in split(poly, signs, c)]
        else:
            polys = [([(0, int(i)) for i in G[fid]], tuple(int(s) for s in whole_sides[fid]))]
        for poly, signs in polys:
            if not cuts.keep(signs):
                continue
            material = BAND if cuts.band(signs) else YELLOW
            for i in range(1, len(poly) - 1):
                corner = (poly[0], poly[i], poly[i + 1])
                p0, p1, p2 = (at(k)[0] for k in corner)
                if not np.linalg.norm(np.cross(p1 - p0, p2 - p0)) > 0:
                    continue                                          # three points on one line: no face
                triangles.setdefault(str(grid_prims[fid]), []).append((*corner, material))

    pieces: dict[str, VestPiece] = {}
    for prim in sorted(triangles):
        keys = sorted({k for t in triangles[prim] for k in t[:3]})   # grid vertices first, then made points
        index = {k: i for i, k in enumerate(keys)}
        pieces[prim] = VestPiece(prim=prim, V=np.array([at(k)[0] for k in keys]),
                                 F=np.array([[index[k] for k in t[:3]] for t in triangles[prim]], dtype=np.int64),
                                 N=np.array([at(k)[1] for k in keys]),
                                 material=np.array([t[3] for t in triangles[prim]]))
    return pieces, {k for t in triangles.values() for tri in t for k in tri[:3]}


def _check_cut(pieces: Mapping[str, VestPiece], kept: set[tuple], shell: _Shell, cuts: _Cuts, spec: VestSpec,
               code_of: Mapping[str, str]) -> None:
    """Refuse a cut vest that is not one: a spec region without a face, any of the grid's highest row left (the
    vest would end in the grid's ragged row there), a hem, armhole or neck that cuts nothing."""
    bare = sorted(set(spec.regions) - {code_of[prim] for prim in pieces})
    if bare:
        raise ValueError(f"조끼의 부위 {bare} 에 든 면이 없습니다 — 그 부위가 빠진 조끼가 됩니다")
    top = (shell.rows - 1) * shell.cols

    def on_top(key: tuple) -> bool:
        return key[1] >= top if key[0] == 0 else on_top(key[1]) and on_top(key[2])

    left = sum(1 for k in kept if on_top(k))
    if left:
        raise ValueError(f"광선 격자의 맨 위 줄이 잘리지 않고 조끼에 남습니다({left} 점) — "
                         "목둘레(neck)를 넓히거나 shell.elevation_deg 의 위 끝을 높이십시오")
    for c, name in ((cuts.hem, "아랫단이"), (cuts.armholes[0], "왼쪽 진동둘레가"), (cuts.armholes[1], "오른쪽 진동둘레가"),
                    (cuts.neck, "목둘레가")):
        if not any(k[0] == 1 and k[3] == c for k in kept):
            raise ValueError(f"{name} 판을 자르지 않습니다 — 판에 닿지 않거나 광선 격자의 칸보다 작아 어느 격자점도 "
                             "그 안에 들지 않습니다")
