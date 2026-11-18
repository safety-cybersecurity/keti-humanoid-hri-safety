"""Whether a robot camera sees the person, by geometry -- the ``seeing`` rule of meta/cameras.json.

A camera on the robot (type "mounted") sees the person when at least ``min_visible_fraction`` of the
person's ``sample_points`` are in its view and unoccluded, and the person's full-body height in its
picture -- f_y * stature / the depth of the person's root -- is at least ``min_height_px``. Both
thresholds are the project's own (``determined_by: project``); the canon block is read here
(seeing_rule) and nowhere else.

- The sample points are fixed in the dummy's parts (generation/dummy/anthropometry.yaml's regions and
  the feet of generation/dummy/joints.yaml): each part gets its share of the body's surface area, at
  least MIN_POINTS_PER_PART, picked from a fixed lattice inside its primitive, in the part's own
  coordinates about its centre -- the origin of the dummy's body for that part. points_world moves
  them with the parts' poses; standing_part_poses gives the poses of a person standing at a point.
- In view: projected inside the picture (0 <= u <= width, 0 <= v <= height) at a depth inside the
  camera's clipping range. The fraction's denominator is every sample point, those out of view too.
- Occluded: the segment from the camera to the point crosses a box -- the layout's colliders, walls
  and doors (layout_occluders, axis-aligned) and the turned boxes (Obb) a caller adds: the robot's own
  links (robot_obbs, from the boxes measured once into robots/unitree_h1_link_boxes.json) and what it
  carries. A box round the camera is ignored for that camera, a point inside a box is occluded, the floor
  occludes nothing and the body does not occlude itself. The arms sway while the robot walks and turns, so a
  wrist camera can come inside the box the robot carries: that camera then counts the box out like any box round
  it (its picture shows cardboard there), and cameras_in_box names such cameras for the run to count.
- A camera on a robot link never counts that link's own box (camera_robot_obbs): the mount rules keep the
  link out of the camera's picture, and its box is an over-estimate of the link -- a wrist camera just above
  its forearm's box, looking down through it, would otherwise lose a large share of its picture to it.
  Containment stays how a box round a camera is ignored; the camera's own link is left out by name.

Pure numpy, no Isaac. The height can be recounted from a release's columns (a camera's pose,
observation.human.root_pose) and the dummy's stature; the fraction needs the parts' poses, which a
release does not carry, so the fraction is what the visible-fraction columns record.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generation import cameras
from generation.dummy import joints_spec
from generation.dummy.joints_spec import JointedSpec
from generation.dummy.skin import region_split
from generation.scene import layout

# meta/cameras.json seeing: the two thresholds and the number of sample points on the body.
SEEING_KEYS = ("min_visible_fraction", "min_height_px", "sample_points")
# The cameras the rule is for: those on the robot's links.
ROBOT_CAMERA_TYPE = "mounted"
# Every part keeps at least this many points, so a small part (a hand, the face) is never left out.
MIN_POINTS_PER_PART = 2
# Cells per axis of a part's bounding box; the centres of those inside the primitive are the candidates.
# Odd, so the part's centre is one of them.
LATTICE_CELLS = 9
# Dressing that blocks the view besides the colliders. The roof's turned web members do not take part.
WALL_PREFIXES = ("wall/", "door/")
# A segment hits a box when the box begins more than this share of the segment's length before the point
# (enter < 1 - eps) and ends at least this share past the origin (leave >= eps), and does not end before it
# begins (leave >= enter). Touching counts: a zero-length pass through an edge or a face is a hit.
SLAB_EPS = 1e-6
_UNIT_TOL = 1e-6
# The robot's links as boxes in each link's own frame, measured once from the rendered H1 meshes by
# generation/robots/extract_link_boxes.py.
LINK_BOXES = Path(__file__).resolve().parent / "robots" / "unitree_h1_link_boxes.json"
LINK_BOXES_FRAME = "link"
# A pose quaternion this far from unit length is a mistake; nearer is the engine's float32 and is renormalised.
_POSE_UNIT_TOL = 1e-4


def _number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


# ---- the rule ----

def seeing_rule(cameras_canon: Mapping) -> dict[str, float]:
    """The rule of a meta/cameras.json document: {min_visible_fraction, min_height_px, sample_points}."""
    block = cameras_canon.get("seeing")
    if not isinstance(block, Mapping):
        raise ValueError(f"meta/cameras.json 의 seeing 이 객체가 아닙니다: {type(block).__name__}")
    missing = [k for k in SEEING_KEYS if k not in block]
    if missing:
        raise ValueError(f"meta/cameras.json seeing 에 {missing} 이 없습니다")
    fraction, height, points = (block[k] for k in SEEING_KEYS)
    if not _number(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError(f"seeing.min_visible_fraction 이 0 보다 크고 1 이하인 수가 아닙니다: {fraction!r}")
    if not _number(height) or height <= 0:
        raise ValueError(f"seeing.min_height_px 이 유한한 양수가 아닙니다: {height!r}")
    if isinstance(points, bool) or not isinstance(points, int) or points <= 0:
        raise ValueError(f"seeing.sample_points 이 양의 정수가 아닙니다: {points!r}")
    return {"min_visible_fraction": float(fraction), "min_height_px": float(height), "sample_points": points}


def robot_cameras(cameras_canon: Mapping) -> tuple[str, ...]:
    """The cameras on the robot's links, in the canon's order -- the ones the rule is read for."""
    return tuple(name for name, cam in cameras_canon["cameras"].items() if cam.get("type") == ROBOT_CAMERA_TYPE)


def seen(fraction: float, height: float, rule: Mapping) -> bool:
    """Both thresholds: enough of the body in view and unoccluded, and the body tall enough in the picture."""
    return bool(fraction >= rule["min_visible_fraction"] and height >= rule["min_height_px"])


# ---- the body's sample points ----

def _area(p: region_split.Primitive) -> float:
    if p.shape == "sphere":
        return 4.0 * math.pi * p.radius ** 2
    if p.shape == "capsule":                      # height is the cylinder; the caps make a sphere
        return 2.0 * math.pi * p.radius * p.height + 4.0 * math.pi * p.radius ** 2
    a, b, c = (float(v) for v in p.size)
    return 2.0 * (a * b + b * c + c * a)


def _half_extent(p: region_split.Primitive) -> np.ndarray:
    if p.shape == "sphere":
        return np.full(3, p.radius)
    if p.shape == "capsule":
        half = np.full(3, p.radius)
        half["xyz".index(p.axis)] += p.height / 2.0
        return half
    return np.asarray(p.size, float) / 2.0


def _allocate(areas: np.ndarray, total: object) -> np.ndarray:
    """Points per part: largest remainders of total * area share, a part below the minimum held at it and
    the rest shared again among the others."""
    n = len(areas)
    if isinstance(total, bool) or not isinstance(total, int) or total < MIN_POINTS_PER_PART * n:
        raise ValueError(f"표본점 수 {total!r} 이 부위 {n} 개에 {MIN_POINTS_PER_PART} 개씩 줄 수 있는 정수가 "
                         f"아닙니다 (최소 {MIN_POINTS_PER_PART * n})")
    held = np.zeros(n, bool)
    while True:
        free = ~held
        if not free.any():
            break
        rest = total - MIN_POINTS_PER_PART * int(held.sum())
        quota = np.where(free, areas / areas[free].sum() * rest, MIN_POINTS_PER_PART)
        below = free & (quota < MIN_POINTS_PER_PART)
        if not below.any():
            break
        held |= below
    counts = np.where(free, np.floor(quota), MIN_POINTS_PER_PART).astype(int)
    order = sorted(np.flatnonzero(free), key=lambda i: (-(quota[i] - counts[i]), i))
    for i in order[:total - int(counts.sum())]:
        counts[i] += 1
    return counts


def _part_points(p: region_split.Primitive, k: int) -> np.ndarray:
    """k points of the part's lattice, farthest-first from the one at its centre, about its centre. A
    right-side part takes the left one's points mirrored, so the body's sample is mirror-symmetric."""
    steps = (np.arange(LATTICE_CELLS) + 0.5) / LATTICE_CELLS * 2.0 - 1.0
    grid = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 3) * _half_extent(p)
    candidates = grid[region_split.distance(grid + p.center, p) <= 0.0]
    if len(candidates) < k:
        raise ValueError(f"부위 {p.name} 의 격자 점 {len(candidates)} 개가 표본점 {k} 개보다 적습니다")
    first = int(np.argmin(np.einsum("ij,ij->i", candidates, candidates)))
    chosen = [first]
    gap = np.linalg.norm(candidates - candidates[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(gap))
        chosen.append(nxt)
        gap = np.minimum(gap, np.linalg.norm(candidates - candidates[nxt], axis=1))
    points = candidates[chosen]
    return points * np.array([1.0, -1.0, 1.0]) if p.side == "right" else points


def body_primitives(spec: JointedSpec) -> dict[str, region_split.Primitive]:
    """{body prim name: its primitive} for every region prim and the two feet, in the builder's prim order -- the
    parts the sample points are fixed in, and the shapes generation/hri_geometry.py measures distances to."""
    prims = region_split.primitives_from_anthropometry(spec.anthropometry) + joints_spec.foot_primitives(spec)
    by_name = {p.name: p for p in prims}
    names = joints_spec.region_prims(spec) + tuple(f.name for f in spec.feet)
    if set(names) != set(by_name) or len(names) != len(prims):
        raise ValueError(f"부위 프림의 이름이 맞지 않습니다: {sorted(set(names) ^ set(by_name))}")
    return {n: by_name[n] for n in names}


def body_sample_points(spec: JointedSpec, total: int) -> dict[str, np.ndarray]:
    """{body prim name: (k, 3) points about the part's centre} for every region prim and the two feet, in
    the builder's prim order; the k add up to ``total``. The same spec gives the same points."""
    parts = body_primitives(spec)
    counts = _allocate(np.array([_area(p) for p in parts.values()]), total)
    return {n: _part_points(p, int(k)) for (n, p), k in zip(parts.items(), counts)}


def standing_part_poses(spec: JointedSpec, xy, yaw_rad: float) -> dict[str, np.ndarray]:
    """{body prim name: pose (pos3 + quat xyzw)} of a person standing upright with the soles on the floor at ``xy``
    and facing ``yaw_rad`` (counter-clockwise from +x): every part at its standing centre -- the dimensions'
    centres, in the dummy's body frame whose origin is on the floor under the body and whose +x is the front --
    turned with the body about the vertical. Every part body_sample_points gives points for; points_world places
    them."""
    x0, y0 = (float(v) for v in np.asarray(xy, float).reshape(2))
    if not _number(yaw_rad):
        raise ValueError(f"요가 유한한 수가 아닙니다: {yaw_rad!r}")
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    quat = np.array([0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)])
    prims = region_split.primitives_from_anthropometry(spec.anthropometry) + joints_spec.foot_primitives(spec)
    out = {}
    for p in prims:
        x, y, z = (float(v) for v in p.center)
        out[p.name] = np.array([x0 + c * x - s * y, y0 + s * x + c * y, z, *quat])
    return out


def points_world(part_points: Mapping[str, np.ndarray], part_poses: Mapping[str, np.ndarray]) -> np.ndarray:
    """Every part's points moved by that part's pose (pos3 + quat xyzw, as body_link_pose_w hands it back),
    stacked in ``part_points``' order -- not the poses'."""
    out = []
    for name, local in part_points.items():
        if name not in part_poses:
            raise ValueError(f"부위 {name} 의 자세가 없습니다")
        pose = np.asarray(part_poses[name], float).reshape(-1)
        if pose.shape != (7,):
            raise ValueError(f"부위 {name} 의 자세가 pos3 + quat xyzw 일곱 수가 아닙니다: {pose.shape}")
        out.append(pose[:3] + cameras._q_rot(pose[3:7], np.asarray(local, float).reshape(-1, 3)))
    return np.vstack(out) if out else np.zeros((0, 3))


# ---- what blocks the view ----

@dataclass(frozen=True, eq=False)
class Obb:
    """A turned box: its centre, the rotation from its own axes to the world (quat xyzw) and its half
    sizes along its own axes. Compared by identity -- the fields are arrays."""
    center: np.ndarray
    rot_xyzw: np.ndarray
    half: np.ndarray

    def __post_init__(self):
        values = {k: np.asarray(getattr(self, k), float) for k in ("center", "rot_xyzw", "half")}
        for key, n in (("center", 3), ("rot_xyzw", 4), ("half", 3)):
            if values[key].shape != (n,) or not np.isfinite(values[key]).all():
                raise ValueError(f"Obb.{key} 가 유한한 수 {n} 개가 아닙니다: {getattr(self, key)!r}")
            object.__setattr__(self, key, values[key])
        if not (self.half > 0).all():
            raise ValueError(f"Obb.half 가 양수가 아닙니다: {self.half!r}")
        if abs(float(np.linalg.norm(self.rot_xyzw)) - 1.0) > _UNIT_TOL:
            raise ValueError(f"Obb.rot_xyzw 가 단위 사원수가 아닙니다: {self.rot_xyzw!r}")


def layout_occluders(boxes: Sequence[layout.Box]) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi), each (M, 3): the corners of every collider of a layout (generation/scene/layout.py
    build_layout) and of its walls and doors that are not turned. The floor slab is neither, so it is
    not one. A turned collider is refused -- its axis-aligned bounds are a box it does not fill; hand it
    over as an Obb."""
    lo, hi = [], []
    for b in boxes:
        if b.collider:
            if b.rot_x_deg != 0.0:
                raise ValueError(f"충돌체 {b.name} 이 돌아 있습니다 (rot_x_deg {b.rot_x_deg}) — 축 정렬 경계는 "
                                 "그 상자를 채우지 않으므로 Obb 로 넘기십시오")
        elif not (b.name.startswith(WALL_PREFIXES) and b.rot_x_deg == 0.0):
            continue
        corner_lo, corner_hi = layout.bounds(b)
        lo.append(corner_lo)
        hi.append(corner_hi)
    return np.array(lo, float).reshape(-1, 3), np.array(hi, float).reshape(-1, 3)


# ---- the robot's links ----

def _three_numbers(value: object) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 3 or not all(_number(v) for v in value):
        return None
    return np.array(value, float)


def load_link_boxes(path: Path = LINK_BOXES) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{link: (centre, half sizes)} of the file's boxes, each in its link's frame and axis-aligned there. Every
    number is finite and every half size above zero -- a flat box hides nothing and is refused."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, Mapping):
        raise ValueError(f"링크 상자 파일이 객체가 아닙니다: {path}")
    if doc.get("frame") != LINK_BOXES_FRAME:
        raise ValueError(f"링크 상자 파일의 frame 이 {LINK_BOXES_FRAME!r} 이 아닙니다: {doc.get('frame')!r}")
    links = doc.get("links")
    if not isinstance(links, Mapping) or not links:
        raise ValueError(f"링크 상자 파일의 links 가 비어 있지 않은 객체가 아닙니다: {type(links).__name__}")
    out = {}
    for name, entry in links.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"링크 {name} 의 상자가 객체가 아닙니다: {entry!r}")
        centre = _three_numbers(entry.get("center"))
        half = _three_numbers(entry.get("half_extents"))
        if centre is None:
            raise ValueError(f"링크 {name} 의 center 가 유한한 수 세 개가 아닙니다: {entry.get('center')!r}")
        if half is None or not (half > 0).all():
            raise ValueError(f"링크 {name} 의 half_extents 가 0 보다 큰 유한한 수 세 개가 아닙니다: "
                             f"{entry.get('half_extents')!r}")
        out[name] = (centre, half)
    return out


def robot_obbs(link_boxes: Mapping[str, tuple[np.ndarray, np.ndarray]],
               link_poses: Mapping[str, np.ndarray], exclude: Iterable[str] = ()) -> list[Obb]:
    """The link boxes placed in the world: one Obb per box, in the boxes' order, by its link's pose (pos3 +
    quat xyzw, as body_link_pose_w hands it back). A link with a box and no pose is refused; poses of links
    without a box are not looked at. The links named in ``exclude`` are left out (and need no pose); a name
    that is not a link with a box is refused, so a misspelt link cannot keep its box in.

    A camera's own link is what ``exclude`` is for -- see camera_robot_obbs."""
    if isinstance(exclude, str):
        raise ValueError(f"exclude 는 링크 이름의 목록이어야 합니다 (문자열 {exclude!r} 이 아니라)")
    drop = set(exclude)
    unknown = sorted(drop - set(link_boxes))
    if unknown:
        raise ValueError(f"exclude 의 {unknown} 은 상자가 있는 링크가 아닙니다 (링크: {sorted(link_boxes)})")
    out = []
    for name, (centre, half) in link_boxes.items():
        if name in drop:
            continue
        if name not in link_poses:
            raise ValueError(f"링크 {name} 의 자세가 없습니다")
        pose = np.asarray(link_poses[name], float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"링크 {name} 의 자세가 유한한 pos3 + quat xyzw 일곱 수가 아닙니다: {pose.shape}")
        norm = float(np.linalg.norm(pose[3:7]))
        if abs(norm - 1.0) > _POSE_UNIT_TOL:
            raise ValueError(f"링크 {name} 의 사원수가 단위 길이가 아닙니다 (길이 {norm}): {pose[3:7]!r}")
        quat = pose[3:7] / norm
        out.append(Obb(center=pose[:3] + cameras._q_rot(quat, np.asarray(centre, float)), rot_xyzw=quat,
                       half=np.asarray(half, float)))
    return out


def camera_robot_obbs(cameras_canon: Mapping, camera: str, link_boxes: Mapping[str, tuple[np.ndarray, np.ndarray]],
                      link_poses: Mapping[str, np.ndarray]) -> list[Obb]:
    """The robot's link boxes as one of its cameras sees them: every link's box except the link the camera is
    mounted on. The mount rules keep that link out of the camera's picture, and its box is an over-estimate of
    the link (a wrist camera sits just above its forearm's box and looks down through it), so it is not an
    occluder from that camera; a box that contains a camera is ignored by ``occluded`` whatever its link.
    The camera must be of type "mounted" in the canon."""
    cams = cameras_canon.get("cameras")
    cam = cams.get(camera) if isinstance(cams, Mapping) else None
    if not isinstance(cam, Mapping):
        raise ValueError(f"카메라 {camera!r} 이 meta/cameras.json 에 없습니다")
    if cam.get("type") != ROBOT_CAMERA_TYPE:
        raise ValueError(f"카메라 {camera} 의 type 이 {ROBOT_CAMERA_TYPE!r} 이 아니라 {cam.get('type')!r} 입니다 — "
                         "로봇 링크에 붙은 카메라만 자기 링크를 뺍니다")
    parent = cam.get("parent_prim")
    if not isinstance(parent, str) or not parent:
        raise ValueError(f"카메라 {camera} 의 parent_prim 이 링크 이름이 아닙니다: {parent!r}")
    return robot_obbs(link_boxes, link_poses, exclude=(parent,))


def _rows_of_three(points: object, what: str) -> np.ndarray:
    array = np.asarray(points, float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{what} 는 (N, 3) 배열이어야 합니다: {array.shape}")
    return array


def _crosses(origin: np.ndarray, points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """For each point, whether the segment origin -> point crosses one of the boxes (slab test). A step of
    0 along an axis divides by 0: inf where the origin is off that face, 0/0 where it lies on it -- the
    NaN-ignoring reductions leave that axis to the others."""
    step = points - origin
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo[None, :, :] - origin) / step[:, None, :]
        t2 = (hi[None, :, :] - origin) / step[:, None, :]
        enter = np.fmax.reduce(np.minimum(t1, t2), axis=2)       # nanmax without its all-NaN warning
        leave = np.fmin.reduce(np.maximum(t1, t2), axis=2)
        hit = (leave >= np.maximum(enter, SLAB_EPS)) & (enter < 1.0 - SLAB_EPS)
    return hit.any(axis=1)


def occluded(origin: np.ndarray, points: np.ndarray, lo: np.ndarray, hi: np.ndarray,
             obbs: Sequence[Obb] = ()) -> np.ndarray:
    """(N,) bool: the segment from ``origin`` to each point crosses an axis-aligned box (lo, hi) or a turned
    one. A box with ``origin`` inside it is left out; a point inside a box is behind it. Only the boxes that
    meet the bounds of the segments are tested."""
    o = np.asarray(origin, float).reshape(3)
    P = _rows_of_three(points, "점")
    lo, hi = _rows_of_three(lo, "lo"), _rows_of_three(hi, "hi")
    if lo.shape != hi.shape:
        raise ValueError(f"lo 와 hi 의 모양이 다릅니다: {lo.shape} {hi.shape}")
    hidden = np.zeros(len(P), bool)
    if not len(P):
        return hidden
    if len(lo):
        round_origin = np.all((lo < o) & (o < hi), axis=1)
        reach_lo, reach_hi = np.minimum(o, P.min(axis=0)), np.maximum(o, P.max(axis=0))
        near = np.all((lo <= reach_hi) & (hi >= reach_lo), axis=1) & ~round_origin
        if near.any():
            hidden |= _crosses(o, P, lo[near], hi[near])
    for box in obbs:
        if _holds(box, o):
            continue
        to_box = np.array([-box.rot_xyzw[0], -box.rot_xyzw[1], -box.rot_xyzw[2], box.rot_xyzw[3]])
        hidden |= _crosses(cameras._q_rot(to_box, o - box.center), cameras._q_rot(to_box, P - box.center),
                           -box.half[None], box.half[None])
    return hidden


def _box_excess(box: Obb, point) -> np.ndarray:
    """|the point in the box's frame| less the box's half sizes, per axis: all below 0 strictly inside."""
    to_box = np.array([-box.rot_xyzw[0], -box.rot_xyzw[1], -box.rot_xyzw[2], box.rot_xyzw[3]])
    return np.abs(cameras._q_rot(to_box, np.asarray(point, float).reshape(3) - box.center)) - box.half


def _holds(box: Obb, point: np.ndarray) -> bool:
    """Whether ``point`` is strictly inside the turned box -- the box occluded leaves out for a camera there."""
    return bool(np.all(_box_excess(box, point) < 0.0))


def box_clearance(box: Obb, point) -> float:
    """How far ``point`` is from the turned box's surface (m): its distance to the box outside it, 0.0 on a face, and
    minus its depth below the nearest face inside it -- below 0 exactly where _holds (and so occluded) counts the
    point inside."""
    excess = _box_excess(box, point)
    if np.all(excess < 0.0):
        return float(excess.max())
    return float(np.linalg.norm(np.maximum(excess, 0.0)))


# ---- the view, the fraction and the height ----

def _clipping(cam: Mapping) -> tuple[float, float]:
    rng = cam.get("clipping_range_m")
    if (not isinstance(rng, (list, tuple)) or len(rng) != 2 or not all(_number(v) for v in rng)
            or not 0.0 < rng[0] < rng[1]):
        raise ValueError(f"카메라의 clipping_range_m 이 0 < 근 < 원 인 두 수가 아닙니다: {rng!r}")
    return float(rng[0]), float(rng[1])


def _depth(points: np.ndarray, pos, quat_xyzw) -> np.ndarray:
    q = np.asarray(quat_xyzw, float)
    return cameras._q_rot([-q[0], -q[1], -q[2], q[3]], points - np.asarray(pos, float))[:, 2]


def visible_fraction(cam: Mapping, pos, quat_xyzw, points_w, lo, hi, obbs: Sequence[Obb] = ()) -> float:
    """The share of ``points_w`` the camera at pos/quat_xyzw (a ROS optical frame) has in view and
    unoccluded, out of all of them."""
    P = _rows_of_three(points_w, "표본점")
    if not len(P):
        raise ValueError("표본점이 없습니다 — 몫의 분모가 0 입니다")
    near, far = _clipping(cam)
    depth = _depth(P, pos, quat_xyzw)
    in_depth = (depth >= near) & (depth <= far)
    view = np.zeros(len(P), bool)
    if in_depth.any():
        u, v, _ = cameras.project_points(P[in_depth], pos, quat_xyzw, cam).T
        view[in_depth] = (u >= 0.0) & (u <= cam["width"]) & (v >= 0.0) & (v <= cam["height"])
    clear = view.copy()
    if view.any():
        clear[view] = ~occluded(pos, P[view], lo, hi, obbs)
    return float(clear.sum()) / len(P)


def height_px(cam: Mapping, pos, quat_xyzw, human_root_w, stature_m: float) -> float:
    """The person's full-body height in the picture, hidden parts included: f_y * stature / the depth of
    the person's root. 0.0 at or nearer than the near clipping plane, behind the camera included."""
    if not _number(stature_m) or stature_m <= 0:
        raise ValueError(f"키가 유한한 양수가 아닙니다: {stature_m!r}")
    near, _ = _clipping(cam)
    depth = float(_depth(np.asarray(human_root_w, float).reshape(1, 3), pos, quat_xyzw)[0])
    if depth <= near:
        return 0.0
    _, fy, _, _ = cameras.intrinsics(cam)
    return float(fy * stature_m / depth)


def _robot_camera_poses(cameras_canon: Mapping, link_poses: Mapping[str, np.ndarray]) -> dict[str, tuple]:
    """{camera: (position, quaternion xyzw)} of every camera on the robot (robot_cameras -- the canon's order), each
    placed on its link's pose (cameras.mounted_pose)."""
    out = {}
    for name in robot_cameras(cameras_canon):
        cam = cameras_canon["cameras"][name]
        parent = cam.get("parent_prim")
        if parent not in link_poses:
            raise ValueError(f"카메라 {name} 가 붙은 링크 {parent!r} 의 자세가 없습니다")
        out[name] = cameras.mounted_pose(np.asarray(link_poses[parent], float), cam["offset_pos"],
                                         cam["offset_rot_xyzw"])
    return out


def robot_camera_fractions(cameras_canon: Mapping, link_poses: Mapping[str, np.ndarray],
                           link_boxes: Mapping[str, tuple[np.ndarray, np.ndarray]], box_obb: Obb | None,
                           points_w, lo, hi) -> dict[str, float]:
    """{camera: visible fraction} for every camera on the robot (robot_cameras -- the canon's order): each placed on
    its link's pose (cameras.mounted_pose), the person's sample points ``points_w`` hidden by the layout's boxes
    (``lo``, ``hi`` -- layout_occluders), the robot's link boxes but the camera's own link (camera_robot_obbs) and
    ``box_obb``, the box the robot carries (None when it carries none or has let it go).

    A camera inside the carried box leaves the box out, as occluded leaves out any box round the camera: its fraction
    is what it would see with no box there, while its picture shows the box's inside. carry.carry_problems keeps every
    camera out of the box at the hold pose; a wrist camera comes in only when the arm sways, and
    cameras_in_box names the cameras that do (from robot_camera_box_clearances), for the run to count."""
    out = {}
    for name, (pos, quat) in _robot_camera_poses(cameras_canon, link_poses).items():
        cam = cameras_canon["cameras"][name]
        obbs = camera_robot_obbs(cameras_canon, name, link_boxes, link_poses)
        if box_obb is not None and not _holds(box_obb, pos):
            obbs.append(box_obb)
        out[name] = visible_fraction(cam, pos, quat, points_w, lo, hi, obbs)
    return out


def robot_camera_box_clearances(cameras_canon: Mapping, link_poses: Mapping[str, np.ndarray],
                                box_obb: Obb) -> dict[str, float]:
    """{camera: box_clearance of its optical centre from ``box_obb``} for every camera on the robot, in the canon's
    order: below 0 for a camera inside the box -- one robot_camera_fractions leaves the box out for."""
    poses = _robot_camera_poses(cameras_canon, link_poses)
    return {name: box_clearance(box_obb, pos) for name, (pos, _) in poses.items()}


def cameras_in_box(clearances: Mapping[str, float]) -> tuple[str, ...]:
    """The cameras inside the box, in the order of ``clearances`` (robot_camera_box_clearances): those whose clearance
    is below 0 -- the ones robot_camera_fractions sees past the box with. A camera on a face (0.0) is not inside."""
    return tuple(name for name, c in clearances.items() if c < 0.0)
