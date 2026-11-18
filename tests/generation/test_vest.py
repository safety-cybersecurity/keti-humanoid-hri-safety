"""generation/dummy/vest.py -- the dummy's hi-vis vest: a smooth shell over the clothes on the trunk, cut to a vest.

Pure numpy and yaml: CI runs it. The USD the vest is attached to (build_dummy.py --vest) is
tested under the Isaac python in tests/generation/test_jointed_build.py.

The synthetic clothes are a 96-sided tube of radius 0.15 m about the z axis from 0.90 m to 1.45 m, closed on
top by a flat disc, all one object; faces below 1.20 m lie on BR_06, the rest on BR_05. The primitives are two
boxes as wide as the tube -- BR_06 from 0.90 to 1.20 m, BR_05 from 1.20 to 1.60 m -- and an upper arm
(BR_08_left), a capsule beside the tube nearer the tube's side than the chest box is. Rays leave (0, 0, 1.25)
on a 72 x 36 grid from 60 degrees below the horizontal to 80 above: the lowest row meets the tube below the
hem, the highest meets the disc inside the neck.

The rays are cast here with a Möller-Trumbore of the test's own, so the shell is measured by something other
than the code that made it.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from generation.dummy import joints_spec as js
from generation.dummy import vest
from generation.dummy.skin import region_split as rs

REPO = Path(__file__).resolve().parents[2]
ANTHRO = yaml.safe_load((REPO / "generation/dummy/anthropometry.yaml").read_text(encoding="utf-8"))

RADIUS = 0.15
SIDES = 96
Z0, Z1, SEAM = 0.90, 1.45, 1.20
CENTRE = np.array([0.0, 0.0, 1.25])
OFFSET = 0.008
DRAPE_MAX = 0.06
HEM = 1.04
BANDS = ((1.085, 1.135), (1.235, 1.285))
# the band material vest.yaml carries: light grey, nearly matt -- a strongly metallic band mirrors the
# cardboard round it and reads brown
BAND_COLOR, BAND_METALLIC, BAND_ROUGHNESS = (0.88, 0.88, 0.88), 0.1, 0.35


def _spec(**changes) -> vest.VestSpec:
    """The synthetic clothes' spec: the canon's hem, bands and materials; a grid, cuts and offset sized to the tube."""
    values = dict(source_object="Human.male_worksuit01", regions=("BR-05", "BR-06"),
                  centre=tuple(CENTRE), azimuth_steps=72, elevation_deg=(-60.0, 80.0), elevation_steps=36,
                  layer_m=0.02, drape_passes=6, drape_max_m=DRAPE_MAX, smooth_passes=1, offset_m=OFFSET,
                  hem_z_m=HEM, bands_z_m=BANDS,
                  armhole_centre=(0.0, 0.15, 1.38), armhole_radii=(0.08, 0.05, 0.10),
                  neck_centre=(0.0, 0.0), neck_radius=0.08, v_front_x=0.0, v_bottom_z=1.36, v_slope=1.6,
                  materials=vest.load_vest_spec().materials)
    values.update(changes)
    return vest.VestSpec(**values)


PRIMS = [
    rs.Primitive("BR-05", "box", np.array([0.0, 0.0, 1.40]), None, None, None, np.array([0.30, 0.30, 0.40])),
    rs.Primitive("BR-06", "box", np.array([0.0, 0.0, 1.05]), None, None, None, np.array([0.30, 0.30, 0.30])),
    rs.Primitive("BR-08", "capsule", np.array([0.0, 0.21, 1.29]), 0.048, 0.26, "z", None,
                 name="BR_08_left", side="left"),
]


class Clothes:
    """Faces with a body prim each; ``mesh()`` makes the ObjMesh (object 1 is the workwear)."""

    def __init__(self):
        self.V, self.F, self.prims, self.FO = [], [], [], []

    def _face(self, a, b, c, prim, obj=1):
        self.F.append([a, b, c]); self.prims.append(prim); self.FO.append(obj)

    def tube(self, radius=RADIUS, z0=Z0, z1=Z1, rows=22, arc=(0.0, 360.0), prim=None, obj=1, keep=None):
        """Outward-wound quads (two triangles) round the z axis; ``prim(z)`` by the quad's middle height,
        ``keep(azimuth_deg, z)`` leaves a quad out when false."""
        full = arc == (0.0, 360.0)
        cols = SIDES if full else max(2, int(round(SIDES * (arc[1] - arc[0]) / 360.0)))
        angles = np.radians(np.linspace(arc[0], arc[1], cols, endpoint=not full))
        zs = np.linspace(z0, z1, rows + 1)
        base = len(self.V)
        self.V += [[radius * math.cos(a), radius * math.sin(a), z] for z in zs for a in angles]
        n = len(angles)
        for r in range(rows):
            for c in range(n if full else n - 1):
                c2 = (c + 1) % n
                p0, p1 = base + r * n + c, base + r * n + c2
                p2, p3 = base + (r + 1) * n + c2, base + (r + 1) * n + c
                mid_a = math.degrees(math.atan2(*reversed(np.mean([self.V[p0][:2], self.V[p2][:2]], axis=0))))
                mid_z = 0.5 * (zs[r] + zs[r + 1])
                if keep is not None and not keep(mid_a % 360.0, mid_z):
                    continue
                name = prim(mid_z) if prim else ("BR_06" if mid_z < SEAM else "BR_05")
                self._face(p0, p1, p2, name, obj)
                self._face(p0, p2, p3, name, obj)
        return self

    def disc(self, z=Z1, radius=RADIUS):
        """The tube's lid, a fan facing up."""
        angles = np.radians(np.linspace(0.0, 360.0, SIDES, endpoint=False))
        c = len(self.V)
        self.V.append([0.0, 0.0, z])
        ring = len(self.V)
        self.V += [[radius * math.cos(a), radius * math.sin(a), z] for a in angles]
        for k in range(SIDES):
            self._face(c, ring + k, ring + (k + 1) % SIDES, "BR_05")
        return self

    def box(self, lo, hi, prim="BR_05"):
        """A closed box, outward-wound."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        b = len(self.V)
        self.V += [[(lo, hi)[i & 1][0], (lo, hi)[(i >> 1) & 1][1], (lo, hi)[(i >> 2) & 1][2]] for i in range(8)]
        quads = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5))
        for q in quads:
            self._face(b + q[0], b + q[1], b + q[2], prim)
            self._face(b + q[0], b + q[2], b + q[3], prim)
        return self

    def mesh(self):
        n = len(self.F)
        m = rs.ObjMesh(V=np.asarray(self.V, float), VT=np.zeros((0, 2)), VN=np.zeros((0, 3)),
                       F=np.asarray(self.F, np.int64), FT=-np.ones((n, 3), np.int64), FN=-np.ones((n, 3), np.int64),
                       FM=np.zeros(n, np.int64), FO=np.asarray(self.FO, np.int64),
                       materials=["m"], objects=["Human.body", "Human.male_worksuit01"])
        m.validate()
        return m, np.asarray(self.prims)


def _plain():
    return Clothes().tube().disc()


def _pieces(clothes=None, **changes):
    mesh, face_prims = (clothes or _plain()).mesh()
    return vest.vest_pieces(mesh, face_prims, PRIMS, _spec(**changes))


def _hits(origin, dirs, V, F):
    """(rays, faces) distances along each ray to each triangle it meets, inf where it meets none."""
    V0 = V[F[:, 0]]; E1 = V[F[:, 1]] - V0; E2 = V[F[:, 2]] - V0
    out = np.full((len(dirs), len(F)), np.inf)
    for i in range(0, len(dirs), 64):
        D = dirs[i:i + 64]
        P = np.cross(D[:, None, :], E2[None])
        det = np.einsum("rfk,fk->rf", P, E1)
        ok = np.abs(det) > 1e-15
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        T = origin - V0
        u = np.einsum("rfk,fk->rf", P, T) * inv
        Q = np.cross(T, E1)
        v = np.einsum("rk,fk->rf", D, Q) * inv
        t = np.einsum("fk,fk->f", E2, Q)[None, :] * inv
        hit = ok & (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12) & (t > 0)
        out[i:i + 64] = np.where(hit, t, np.inf)
    return out


def _merged(pieces):
    V = np.concatenate([p.V for p in pieces.values()])
    F = np.concatenate([p.F + o for p, o in zip(pieces.values(),
                                                  np.cumsum([0] + [len(p.V) for p in pieces.values()])[:-1])])
    return V, F


def _direction(azimuth_deg, elevation_deg):
    a, b = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.cos(b) * math.cos(a), math.cos(b) * math.sin(a), math.sin(b)])


def _vest_distance(pieces, azimuth_deg, elevation_deg):
    V, F = _merged(pieces)
    t = _hits(CENTRE, _direction(azimuth_deg, elevation_deg)[None], V, F)[0]
    assert np.isfinite(t).any(), f"광선 {azimuth_deg}°, {elevation_deg}° 가 조끼에 닿지 않습니다"
    return float(t.min())


def _tube_distance(elevation_deg, radius=RADIUS):
    """Along a ray through a tube vertex's azimuth (every 3.75°): the tube itself, radius / cos(elevation)."""
    return radius / math.cos(math.radians(elevation_deg))


# ---- the shell ------------------------------------------------------------------------------------

def test_the_drape_is_along_the_ray_so_the_vest_hangs_looser_where_the_rays_are_slant():
    """On a straight tube the distance along a ray is radius / cos(elevation): flat at the chest's height, steep
    towards the hem. Averaged along the ray, the shell lifts off the tube where the rays are slant and stays close
    where they are level -- the vest hangs a little loose low down instead of clinging. Averaged as horizontal
    distance (constant on the tube), it would stand off by the offset alone everywhere."""
    pieces = _pieces()
    level = _vest_distance(pieces, 7.5, 0.0) - _tube_distance(0.0)
    slant = _vest_distance(pieces, 7.5, -50.0) - _tube_distance(-50.0)
    assert OFFSET <= level < OFFSET + 0.005, level
    assert slant > level + 0.005, (level, slant)


def test_the_last_smoothing_does_not_pull_the_vest_into_the_clothes():
    """The lid's rim is a sharp ridge: averaged with the rows below it, a ray at the rim would come out short of
    the rim. Smoothed hard (ten passes) with a tiny offset, the vest still lies outside the clothes along every
    ray through a point of the grid."""
    spec = _spec(offset_m=0.0005, smooth_passes=10)
    V, _ = _merged(_pieces(offset_m=0.0005, smooth_passes=10))
    # the grid's own points: a point made on a cut lies on a chord between two of them, and a chord across a
    # sharp ridge passes inside it
    V = V[~np.any([np.abs(v) < _ON[name] for name, v in _cut_values(spec, V).items()], axis=0)]
    mesh, _ = _plain().mesh()
    d = V - CENTRE
    t = np.linalg.norm(d, axis=1)
    cloth = _hits(CENTRE, d / t[:, None], mesh.V, mesh.F).min(axis=1)
    assert (t - cloth).min() > 0, (t - cloth).min()


def test_the_drape_never_stands_the_vest_off_by_more_than_drape_max():
    """Draped thirty times, the tube's slant rays and the lid would lift the shell far off the clothes; capped at
    1 cm, along every ray through a vest point the vest is no farther out than the clothes by more than the cap
    and the offset (and 2 mm: the last smoothing averages capped neighbours over a varying floor). Uncapped, the
    same drape stands it off by more -- the cap is what holds it."""
    mesh, _ = _plain().mesh()

    def lift(**changes):
        V, _ = _merged(_pieces(drape_passes=30, **changes))
        d = V - CENTRE
        t = np.linalg.norm(d, axis=1)
        cloth = _hits(CENTRE, d / t[:, None], mesh.V, mesh.F).min(axis=1)
        assert np.isfinite(cloth).all()
        return t - cloth

    capped = lift(drape_max_m=0.01)
    assert capped.min() > 0
    assert capped.max() <= 0.01 + OFFSET + 0.002, capped.max()
    assert lift(drape_max_m=1.0).max() > 0.01 + OFFSET + 0.005, "상한이 없어도 들뜨지 않아 시험이 상한을 보지 못합니다"


def test_over_an_opening_at_the_top_the_shell_closes_in():
    """The tube without its lid: the top rows meet nothing. Filled from the rows round them with the axis beyond
    the top row, the shell closes over the opening inside the neck instead of rising up the tube's radius --
    so the grid's top row is cut away. Filled with the top row repeated, it would rise as a chimney of the
    tube's radius, outside the neck, and the check would refuse it."""
    open_top = Clothes().tube()
    pieces = _pieces(open_top)
    V, _ = _merged(pieces)
    assert V[:, 2].max() < 1.62                                  # nothing climbs far above the tube


def test_a_buckle_is_draped_over_not_followed():
    """A box 12 mm proud of the tube's front (|y| < 2 cm, 1.30-1.34 m) is inside the vest along every ray through
    its corners, and the vest already rises two grid cells beside it -- it is draped, not stepped. Without the
    drape the vest would lift only where the box is."""
    clothes = _plain().box([0.146, -0.02, 1.30], [0.162, 0.02, 1.34])
    pieces = _pieces(clothes)
    V, F = _merged(pieces)
    corners = np.array([[x, y, z] for x in (0.162,) for y in (-0.02, 0.02) for z in (1.30, 1.34)])
    dirs = (corners - CENTRE) / np.linalg.norm(corners - CENTRE, axis=1)[:, None]
    t = _hits(CENTRE, dirs, V, F).min(axis=1)
    assert np.isfinite(t).all(), "버클 모서리를 지나는 광선이 조끼에 닿지 않습니다"
    assert np.all(t > np.linalg.norm(corners - CENTRE, axis=1))
    beside = _vest_distance(pieces, 15.0, 22.0) - _vest_distance(_pieces(), 15.0, 22.0)
    assert beside > 0.001, beside
    flat = _pieces(clothes, drape_passes=0)
    assert _vest_distance(flat, 15.0, 22.0) - _vest_distance(_pieces(drape_passes=0), 15.0, 22.0) < 0.0005


def test_a_garment_beyond_the_layer_is_not_followed():
    """A strip 5 cm out from the tube's side (80-100°, 1.10-1.20 m, on BR_06) is farther than layer_m beyond the
    first cloth a ray meets there: the vest is the plain tube's. A strip within the layer is followed. The ray at
    90° runs down an edge of the tube (its columns are 3.75° apart): it must meet the tube there, or the strip
    would be the first cloth it meets."""
    far = _plain().tube(radius=RADIUS + 0.05, z0=1.10, z1=1.20, rows=4, arc=(80.0, 100.0))
    assert _vest_distance(_pieces(far), 90.0, -23.0) == _vest_distance(_pieces(), 90.0, -23.0)
    near = _plain().tube(radius=RADIUS + 0.01, z0=1.10, z1=1.20, rows=4, arc=(80.0, 100.0))
    assert _vest_distance(_pieces(near), 90.0, -23.0) > _tube_distance(-23.0, RADIUS + 0.01)


def test_clothes_off_the_spec_regions_are_not_draped():
    """The same near strip on the upper arm (BR_08_left): not the trunk's, so the vest is the plain tube's."""
    arm = _plain().tube(radius=RADIUS + 0.01, z0=1.10, z1=1.20, rows=4, arc=(80.0, 100.0),
                        prim=lambda z: "BR_08_left")
    assert _vest_distance(_pieces(arm), 90.0, -23.0) == _vest_distance(_pieces(), 90.0, -23.0)


def test_rays_meeting_nothing_take_their_neighbours():
    """A hole in the back of the tube (170-190°, 1.12-1.22 m): the rays through it meet nothing and take the
    distances round them, so the vest closes over the hole as if the cloth were there."""
    holed = Clothes().tube(keep=lambda a, z: not (170.0 < a < 190.0 and 1.12 < z < 1.22)).disc()
    t = _vest_distance(_pieces(holed), 180.0, -23.0)
    assert abs(t - _vest_distance(_pieces(), 180.0, -23.0)) < 0.002, t


def test_the_vest_faces_outward():
    pieces = _pieces()
    for p in pieces.values():
        assert np.allclose(np.linalg.norm(p.N, axis=1), 1.0, atol=1e-12)
        assert np.all(np.einsum("ij,ij->i", p.N, p.V - CENTRE) > 0)
        P = p.V[p.F]
        cross = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        assert np.all(np.einsum("ij,ij->i", cross, P.mean(axis=1) - CENTRE) > 0), p.prim
        assert np.all(np.linalg.norm(cross, axis=1) > 0)


# ---- the cuts -------------------------------------------------------------------------------------

def _cut_values(spec, V):
    """Every cut's value at the points, by the spec's geometry: zero on the cut."""
    c = np.asarray(spec.armhole_centre); r = np.asarray(spec.armhole_radii)
    mirror = c * np.array([1.0, -1.0, 1.0])
    nx, ny = spec.neck_centre
    out = {"hem": V[:, 2] - spec.hem_z_m}
    for i, (lo, hi) in enumerate(spec.bands_z_m):
        out[f"band{i}lo"], out[f"band{i}hi"] = V[:, 2] - lo, V[:, 2] - hi
    out["armhole left"] = (((V - c) / r) ** 2).sum(axis=1) - 1.0
    out["armhole right"] = (((V - mirror) / r) ** 2).sum(axis=1) - 1.0
    out["neck"] = ((V[:, 0] - nx) ** 2 + (V[:, 1] - ny) ** 2) / spec.neck_radius ** 2 - 1.0
    out["v front"] = V[:, 0] - spec.v_front_x
    out["v left"] = V[:, 2] - spec.v_bottom_z - spec.v_slope * V[:, 1]
    out["v right"] = V[:, 2] - spec.v_bottom_z + spec.v_slope * V[:, 1]
    return out


# How near a cut a point counts as on it. A plane cuts exactly; an ellipsoid or a cylinder is followed by chords,
# and where a second cut crosses such a chord its point lies a little inside the curve: 1e-3 of the value is
# tens of micrometres at these radii.
_ON = {"hem": 1e-9, "band0lo": 1e-9, "band0hi": 1e-9, "band1lo": 1e-9, "band1hi": 1e-9,
       "armhole left": 1e-3, "armhole right": 1e-3, "neck": 1e-3, "v front": 1e-9, "v left": 1e-9, "v right": 1e-9}


def test_nothing_is_left_below_the_hem_or_in_an_armhole_the_neck_or_the_v():
    spec = _spec()
    V, _ = _merged(_pieces())
    f = _cut_values(spec, V)
    assert V[:, 2].min() == HEM                                  # the hem, exactly
    for name in ("armhole left", "armhole right", "neck"):
        assert f[name].min() > -_ON[name], name
    in_v = (f["v front"] > 1e-9) & (f["v left"] > 1e-9) & (f["v right"] > 1e-9)
    assert not in_v.any()


def test_the_vests_only_open_edges_are_its_cuts():
    """Merged by position, the pieces make one surface whose open edges all lie on a cut: the faces either side
    of a cut edge share the point made on it, the pieces share their seams, and the cut faces are cut, not
    dropped whole (a dropped face leaves an open edge off every cut). Every outer cut makes an edge."""
    spec = _spec()
    V, F = _merged(_pieces())
    key = np.round(V, 12)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    Fu = inv.reshape(-1)[F]
    edges = np.sort(np.concatenate([Fu[:, [0, 1]], Fu[:, [1, 2]], Fu[:, [2, 0]]]), axis=1)
    e, count = np.unique(edges, axis=0, return_counts=True)
    open_edges = e[count == 1]
    assert len(open_edges) and count.max() == 2
    f = _cut_values(spec, uniq)
    on_cut = np.zeros(len(open_edges), bool)
    for name, v in f.items():
        on = np.abs(v) < _ON[name]
        both = on[open_edges[:, 0]] & on[open_edges[:, 1]]
        if name in ("hem", "armhole left", "armhole right", "neck", "v left", "v right"):
            assert both.any(), f"{name} 에 잘린 가장자리가 없습니다"
        on_cut |= both
    assert on_cut.all(), uniq[open_edges[~on_cut][:3]]


def test_the_hem_and_band_edges_are_straight_and_the_bands_take_the_band_material():
    pieces = _pieces()
    for p in pieces.values():
        z = p.V[p.F][:, :, 2]
        for plane in (HEM,) + tuple(z for band in BANDS for z in band):
            assert not np.any((z.min(axis=1) < plane) & (z.max(axis=1) > plane)), (p.prim, plane)
        cz = z.mean(axis=1)
        inside = np.zeros(len(cz), bool)
        for lo, hi in BANDS:
            inside |= (cz > lo) & (cz < hi)
        assert list(p.material) == [vest.BAND if b else vest.YELLOW for b in inside], p.prim
    V, _ = _merged(pieces)
    for plane in (HEM,) + tuple(z for band in BANDS for z in band):
        near = np.abs(V[:, 2] - plane) < 1e-9
        assert near.any() and np.all(V[near, 2] == plane), plane   # points made exactly on every plane
    # the cut only divides: the band faces lie wholly between their planes
    band = np.concatenate([p.material == vest.BAND for p in pieces.values()])
    assert band.any() and not band.all()


def test_a_cut_point_takes_its_edge_normals_interpolated():
    """Points made on a plane carry unit normals that still face out; on the tube's side, where every shell normal
    is horizontal to within the drape, so are they."""
    for p in _pieces().values():
        made = np.isin(p.V[:, 2], (HEM,) + tuple(z for band in BANDS for z in band))
        if not made.any():
            continue
        assert np.allclose(np.linalg.norm(p.N[made], axis=1), 1.0, atol=1e-12)
        assert np.all(np.einsum("ij,ij->i", p.N[made], p.V[made] - CENTRE) > 0)
        assert np.abs(p.N[made, 2]).max() < 0.2


def _hand_shell(points, normals, faces, cols):
    """A shell of the given faces in its first row; a second row of unused points so the top row is not theirs."""
    P = np.vstack([np.asarray(points, float), np.full((cols - len(points), 3), 9.0), np.full((cols, 3), 9.0)])
    N = np.vstack([np.asarray(normals, float), np.tile([1.0, 0.0, 0.0], (2 * cols - len(points), 1))])
    return vest._Shell(P=P, G=np.asarray(faces, np.int64), N=N / np.linalg.norm(N, axis=1)[:, None], rows=2, cols=cols)


# cuts far from the hand-made faces: only the hem and the bands reach them
FAR = dict(armhole_centre=(0.0, 5.0, 5.0), armhole_radii=(0.1, 0.1, 0.1), neck_centre=(5.0, 5.0), neck_radius=0.1,
           v_front_x=5.0)


def test_a_face_lying_in_a_plane_keeps_the_side_a_point_on_it_keeps():
    """A flat face exactly on the hem is kept, as a point on the hem is; a face exactly on a band's lower edge is
    not band and one on its upper edge is -- a face on a plane goes with the slab below it, except at the hem,
    which keeps its floor."""
    spec = _spec(regions=("BR-05",), bands_z_m=((1.10, 1.20),), **FAR)

    def flat(z, x0):
        return [[x0, 0.0, z], [x0 + 0.1, 0.0, z], [x0, 0.1, z]]

    points = flat(HEM, 0.0) + flat(1.10, 0.2) + flat(1.20, 0.4)
    shell = _hand_shell(points, [[0.0, 0.0, 1.0]] * 9, [[0, 1, 2], [3, 4, 5], [6, 7, 8]], cols=12)
    pieces, _ = vest._cut_pieces(shell, np.array(["BR_05"] * 3), vest._cuts(spec))
    (piece,) = pieces.values()
    z = piece.V[piece.F][:, :, 2].mean(axis=1)
    assert sorted(zip(z.round(3), piece.material)) == [(1.04, vest.YELLOW), (1.10, vest.YELLOW), (1.20, vest.BAND)]


def test_a_face_on_the_hem_that_another_cut_crosses_keeps_its_part_outside_that_cut():
    """A flat face on the hem with one corner inside an armhole (a sphere of 5 cm round that corner): the armhole
    splits it, and the part outside, still lying wholly on the hem, stays -- as a point on the hem does."""
    far = dict(FAR, armhole_centre=(0.6, 0.0, HEM), armhole_radii=(0.05, 0.05, 0.05))
    spec = _spec(regions=("BR-05",), bands_z_m=(), **far)
    shell = _hand_shell([[0.6, 0.0, HEM], [0.8, 0.0, HEM], [0.6, 0.2, HEM]], [[0.0, 0.0, 1.0]] * 3, [[0, 1, 2]], cols=6)
    pieces, _ = vest._cut_pieces(shell, np.array(["BR_05"]), vest._cuts(spec))
    (piece,) = pieces.values()
    assert np.all(piece.V[:, 2] == HEM)
    whole = 0.5 * 0.2 * 0.2
    assert whole - math.pi * 0.05 ** 2 / 4 - 1e-3 < _area(piece) < whole


def _area(piece) -> float:
    P = piece.V[piece.F]
    return float(0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1).sum())


def test_a_cut_point_takes_its_edge_normals_interpolated_exactly():
    """One triangle across the hem: from (0, 0, 1.00) with normal +x to (0, 0, 1.10) with normal +y, the hem at
    1.04 is t = 0.4 along the edge, so the point made there has the normal 0.6 (1, 0, 0) + 0.4 (0, 1, 0),
    normalised."""
    spec = _spec(regions=("BR-05",), bands_z_m=(), **FAR)
    shell = _hand_shell([[0.0, 0.0, 1.00], [0.0, 0.0, 1.10], [0.0, 0.1, 1.10]],
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], [[0, 1, 2]], cols=6)
    pieces, _ = vest._cut_pieces(shell, np.array(["BR_05"]), vest._cuts(spec))
    (piece,) = pieces.values()
    at = np.nonzero((piece.V[:, 2] == HEM) & (piece.V[:, 1] == 0.0))[0]
    assert len(at) == 1
    want = np.array([0.6, 0.4, 0.0]) / np.hypot(0.6, 0.4)
    assert np.allclose(piece.N[at[0]], want, atol=1e-12)


def test_grid_faces_go_to_the_majority_of_their_vertices_primitives():
    """Two of the face's vertices are nearest the abdomen box and its centroid is nearest the chest box: the face
    goes with its vertices, as the skin's faces do."""
    P = np.array([[0.15, -0.01, 1.19], [0.15, 0.01, 1.19], [0.15, 0.0, 1.23]])
    region_prims = [p for p in PRIMS if p.code in ("BR-05", "BR-06")]
    centroid_nearest = region_prims[rs.assign_primitives(P.mean(axis=0)[None], region_prims)[0]].name
    assert centroid_nearest == "BR_05"
    assert list(vest._grid_prims(P, np.array([[0, 1, 2]]), region_prims)) == ["BR_06"]


# ---- the pieces -----------------------------------------------------------------------------------

def test_each_face_hangs_on_the_nearest_primitive_of_the_spec_regions():
    """Above the boxes' seam (1.20 m) the vest is on BR_05, below it on BR_06 -- and nothing goes to the upper
    arm, though it is nearer the tube's side than the chest box is: only the spec's regions take pieces."""
    pieces = _pieces()
    assert sorted(pieces) == ["BR_05", "BR_06"]
    for prim, p in pieces.items():
        assert p.prim == prim
        # a face goes with its grid face, and a grid face across the seam (rows are about a centimetre apart
        # there) with the majority of its corners
        z = p.V[p.F][:, :, 2]
        if prim == "BR_05":
            assert not np.any(z.max(axis=1) < SEAM - 0.02)
        else:
            assert not np.any(z.min(axis=1) > SEAM + 0.02)
        assert p.F.min() == 0 and p.F.max() == len(p.V) - 1 and len(np.unique(p.F)) == len(p.V)


def test_vest_pieces_are_deterministic():
    a, b = _pieces(), _pieces()
    assert list(a) == list(b)
    for k in a:
        for field in ("V", "F", "N", "material"):
            assert np.array_equal(getattr(a[k], field), getattr(b[k], field)), (k, field)


# ---- refusals -------------------------------------------------------------------------------------

@pytest.mark.parametrize("changes, phrase", [
    ({"source_object": "Human.coat"}, "Human.coat"),
    ({"regions": ("BR-09",)}, "프리미티브"),
    ({"centre": (0.0, 0.0, 10.0), "elevation_deg": (-10.0, 10.0)}, "광선"),
    ({"elevation_deg": (-10.0, 80.0)}, "아랫단"),
    ({"neck_radius": 0.01}, "맨 위 줄"),
    # the top row's vertices each lie in the neck or the V, but the edges between two of them do not
    ({"elevation_deg": (-60.0, 60.0), "neck_centre": (-0.06, 0.0), "neck_radius": 0.17, "v_front_x": 0.08,
      "v_slope": 0.1}, "맨 위 줄"),
    # the centre 2 cm in front of the tube: the rays to the front meet nothing
    ({"centre": (0.17, 0.0, 1.25), "elevation_deg": (-85.0, 89.0)}, "수평"),
    # an armhole 4 cm across on a grid of 30° x 20° cells: no grid vertex is inside it, so it cuts nothing
    ({"azimuth_steps": 12, "elevation_steps": 8, "armhole_centre": (0.02625292, 0.15002003, 1.33189021),
      "armhole_radii": (0.02, 0.02, 0.02)}, "진동둘레"),
])
def test_a_spec_the_clothes_cannot_make_a_vest_from_is_refused(changes, phrase):
    with pytest.raises(ValueError, match=phrase):
        _pieces(**changes)


def test_a_region_without_clothes_is_refused():
    prims = PRIMS + [rs.Primitive("BR-07", "box", np.array([0.0, 0.0, 0.50]), None, None, None,
                                  np.array([0.30, 0.30, 0.20]))]
    mesh, face_prims = _plain().mesh()
    with pytest.raises(ValueError, match="면이 없습니다"):
        vest.vest_pieces(mesh, face_prims, prims, _spec(regions=("BR-07",)))


def test_a_region_the_vest_never_reaches_is_refused():
    prims = PRIMS + [rs.Primitive("BR-04", "box", np.array([0.0, 0.0, 3.0]), None, None, None,
                                  np.array([0.30, 0.30, 0.20]))]
    mesh, face_prims = _plain().mesh()
    with pytest.raises(ValueError, match="BR-04"):
        vest.vest_pieces(mesh, face_prims, prims, _spec(regions=("BR-04", "BR-05", "BR-06")))


def test_a_face_list_that_is_not_one_per_face_is_refused():
    mesh, face_prims = _plain().mesh()
    with pytest.raises(ValueError, match="면마다"):
        vest.vest_pieces(mesh, face_prims[:-1], PRIMS, _spec())


# ---- vest.yaml ------------------------------------------------------------------------------------

def _canon_raw() -> dict:
    return yaml.safe_load(vest.VEST_SPEC.read_text(encoding="utf-8"))


def _write(tmp_path, raw) -> Path:
    path = tmp_path / "vest.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def _drop(*keys):
    def f(raw):
        d = raw
        for k in keys[:-1]:
            d = d[k]
        del d[keys[-1]]
    return f


def _set(*keys_and_value):
    *keys, value = keys_and_value

    def f(raw):
        d = raw
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
    return f


@pytest.mark.parametrize("mutate, phrase", [
    (_drop("offset_m"), "offset_m"),
    (_set("offset_m", 0.0), "offset_m"),
    (_set("hem_z_m", float("nan")), "hem_z_m"),
    (_set("bands_z_m", [[1.135, 1.085]]), "bands_z_m"),
    (_set("regions", []), "regions"),
    (_set("regions", ["BR-04", "BR-04"]), "regions"),
    (_set("source_object", ""), "source_object"),
    (_set("license", ""), "license"),
    (_set("colour", [1, 1, 1]), "colour"),
    (_drop("shell"), "shell"),
    (_set("shell", "centre_m", [0.0, 1.0]), "shell.centre_m"),
    (_set("shell", "azimuth_steps", 2), "shell.azimuth_steps"),
    (_set("shell", "azimuth_steps", 180.5), "shell.azimuth_steps"),
    (_set("shell", "elevation_steps", True), "shell.elevation_steps"),
    (_set("shell", "elevation_deg", [10.0, -10.0]), "shell.elevation_deg"),
    (_set("shell", "elevation_deg", [-90.0, 82.0]), "shell.elevation_deg"),
    (_set("shell", "elevation_deg", [5.0, 82.0]), "shell.elevation_deg"),           # the grid must reach below the horizontal
    (_set("shell", "layer_m", -0.01), "shell.layer_m"),
    (_set("shell", "drape_passes", -1), "shell.drape_passes"),
    (_set("shell", "drape_max_m", -0.01), "shell.drape_max_m"),
    (_set("shell", "grid", 3), "grid"),
    (_set("armhole", "radii_m", [0.16, 0.0, 0.17]), "armhole.radii_m"),
    (_set("armhole", "centre_m", [0.0, -0.23, 1.42]), "armhole.centre_m"),
    (_set("neck", "radius_m", -0.075), "neck.radius_m"),
    (_set("neck", "centre_m", [0.0, 0.0, 1.5]), "neck.centre_m"),
    (_set("v_neck", "slope", 0.0), "v_neck.slope"),
    (_drop("v_neck", "bottom_z_m"), "v_neck"),
    (_drop("materials"), "materials"),
    (_set("materials", "vest_band", "metalic", 0.9), "metalic"),
    (_set("materials", "vest_yellow", "color", [0.8, 1.2, 0.0]), "color"),
    (_set("materials", "vest_band", "roughness", True), "roughness"),
])
def test_a_malformed_vest_spec_is_refused(tmp_path, mutate, phrase):
    raw = _canon_raw()
    mutate(raw)
    with pytest.raises(ValueError, match=phrase):
        vest.load_vest_spec(_write(tmp_path, raw))


def test_the_canon_vest_spec_loads():
    spec = vest.load_vest_spec()
    assert spec.source_object == "Human.male_worksuit01"
    assert spec.regions == ("BR-04", "BR-05", "BR-06", "BR-07")
    assert spec.bands_z_m == ((1.085, 1.135), (1.235, 1.285))
    assert spec.armhole_centre[1] > 0                              # the left one; the right is its mirror
    assert isinstance(spec.azimuth_steps, int) and isinstance(spec.elevation_steps, int)
    assert set(spec.materials) == {"vest_yellow", "vest_band"}
    assert tuple(spec.materials["vest_yellow"]["color"]) == (0.80, 1.00, 0.00)
    assert tuple(spec.materials["vest_band"]["color"]) == BAND_COLOR
    assert spec.materials["vest_band"]["metallic"] == BAND_METALLIC
    assert spec.materials["vest_band"]["roughness"] == BAND_ROUGHNESS
    assert spec.materials["vest_yellow"]["roughness"] == 0.6


def test_the_hem_is_the_bottom_of_the_abdomen_box():
    abdomen = next(p for p in rs.primitives_from_anthropometry(ANTHRO) if p.code == "BR-06")
    assert vest.load_vest_spec().hem_z_m == pytest.approx(rs.bottom_of(abdomen), abs=1e-12)


def test_the_canon_vest_covers_only_the_rigid_torso():
    """A vest piece follows its body rigidly. On a body a joint moves -- or one hanging below such
    a body -- the piece would part from its neighbours on the trunk and cut into the body as the
    joint turned, so the vest's regions must lie wholly on bodies no joint of the jointed dummy
    moves."""
    spec = js.load()
    regions = spec.anthropometry["regions"]
    parent = {}                                       # body prim -> the body prim it hangs from
    for code, reg in regions.items():
        if reg.get("parent") is None:
            continue
        for name, _ in rs.prim_names(code, reg):
            side = name.rsplit("_", 1)[1] if reg.get("mirror_y") else "left"
            parent[name] = js.prim_of(reg["parent"], side, regions)
    moving = {j.child for j in spec.joints}
    assert moving, "관절 인형의 관절이 하나도 없습니다"
    grew = True
    while grew:
        below = {name for name, up in parent.items() if up in moving} - moving
        grew = bool(below)
        moving |= below
    vest_spec = vest.load_vest_spec()
    assert set(vest_spec.regions) <= set(regions)
    covered = {name for code in vest_spec.regions for name, _ in rs.prim_names(code, regions[code])}
    assert covered and not covered & moving, sorted(covered & moving)


# ---- the real skin --------------------------------------------------------------------------------

def test_the_real_vest_covers_the_trunk_clothes_and_hangs_on_four_trunk_parts():
    """On the exported skin, through the face labels the skinned build splits the mesh by: four pieces, the hem
    straight all round, bands on the chest and the abdomen -- and the clothes the vest lies over stay under
    it. Those are the workwear vertices of the trunk within layer_m of the first workwear a ray from the centre
    meets on its way to them (the sleeve beside the chest is farther out and hangs outside the vest, as an arm
    does) and where the vest is kept (above the hem, outside the armholes, the neck and the V) there and as far
    out as the vest can stand. Along the ray through each the vest is farther out than the vertex, or less than a millimetre in
    -- the grid's rays sample the clothes, and under the arm the sleeve's inner side lies within layer_m of the
    chest on one ray and beyond it on the next (0.3 mm there). A ray through an inner fold may still leave the
    vest where it is cut away; those are few."""
    obj = REPO / "generation/dummy/skin/build/keti_skin.obj"
    if not obj.is_file():
        pytest.skip("피부 OBJ 가 빌드되어 있지 않습니다")
    from generation.dummy import build_dummy as bd

    mesh = rs.read_obj(obj)
    prims = rs.primitives_from_anthropometry(ANTHRO) + list(js.foot_primitives(js.load()))
    spec = vest.load_vest_spec()
    face_prims = bd.skin_face_prims(mesh, prims)
    pieces = vest.vest_pieces(mesh, face_prims, prims, spec)
    want = {name for code in spec.regions for name, _ in rs.prim_names(code, ANTHRO["regions"][code])}
    assert set(pieces) == want == {"BR_04", "BR_05", "BR_06", "BR_07"}
    # the bands run over the chest and the abdomen (the lower one reaches the pelvis's piece where the loose hem
    # lies nearer the pelvis box)
    assert {"BR_05", "BR_06"} <= {k for k, p in pieces.items() if vest.BAND in p.material}
    V, F = _merged(pieces)
    assert V[:, 2].min() == spec.hem_z_m
    # straight all round: in every 15° sector about the abdomen box's vertical axis the lowest point is on the hem
    axis = np.asarray(ANTHRO["regions"]["BR-06"]["center"][:2], float)
    angle = np.degrees(np.arctan2(V[:, 1] - axis[1], V[:, 0] - axis[0]))
    for start in range(-180, 180, 15):
        sector = (angle >= start) & (angle < start + 15)
        assert sector.any() and V[sector, 2].min() == spec.hem_z_m, f"{start}°–{start + 15}°: {V[sector, 2].min()}"

    code = {p.name: p.code for p in prims}
    region = np.array([code.get(str(x), "") for x in face_prims])
    trunk = (mesh.FO == mesh.objects.index(spec.source_object)) & np.isin(region, spec.regions)
    W = mesh.V[np.unique(mesh.F[trunk])]
    centre = np.asarray(spec.centre)
    dist = np.linalg.norm(W - centre, axis=1)
    dirs = (W - centre) / dist[:, None]

    def kept(P):
        f = _cut_values(spec, P)
        return ((f["hem"] > 0) & (f["armhole left"] > 0) & (f["armhole right"] > 0) & (f["neck"] > 0)
                & ~((f["v front"] > 0) & (f["v left"] > 0) & (f["v right"] > 0)))

    first = _hits(centre, dirs, mesh.V, mesh.F[trunk]).min(axis=1)
    # where the vest is kept at the vertex and as far out along its ray as the vest can be (the cap, the offset and
    # a centimetre) -- near an edge of the vest the ray may leave through the cut
    reach = spec.drape_max_m + spec.offset_m + 0.01
    draped = kept(W) & kept(centre + dirs * (dist + reach)[:, None]) & (dist <= first + spec.layer_m + 1e-9)
    t = _hits(centre, dirs[draped], V, F).min(axis=1)
    met = np.isfinite(t)
    assert draped.sum() > 1000 and (~met).sum() < 0.01 * draped.sum(), (int(draped.sum()), int((~met).sum()))
    beyond = dist[draped][met] - t[met]
    assert beyond.max() < 0.001, (f"조끼 밖으로 1 mm 넘게 나온 작업복 정점 {int((beyond >= 0.001).sum())}개 · "
                                  f"최대 {beyond.max() * 1000:.1f} mm")
