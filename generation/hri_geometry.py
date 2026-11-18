"""The HRI geometry of a step, for the scenarios that declare R-TS15066-SSM-02 (schema §6): where the robot and the
person are nearest, how fast those two points close on each other, the directed S_p and its breach.

The robot is its link boxes (visibility.robot_obbs -- generation/robots/unitree_h1_link_boxes.json, boxes measured
round the rendered meshes, so each is larger than its link and the distance comes out short of the real one, on the
strict side) and, until it lets go, the box it carries (carry.box_obb). The person is the dummy's parts as their
primitives -- the spheres, capsules and boxes of generation/dummy/anthropometry.yaml and the two feet of
generation/dummy/joints.yaml -- each about the origin of its body, the frame the engine reports (human_shapes).

- closest_pair: the nearest robot box and part and their two points, solved exactly rather than iterated: a sphere
  is a point and a capsule a segment grown by the radius, a box is near another box at one of the two boxes' edges,
  and the squared distance from a point moving along a segment to a box is a convex piecewise quadratic whose least
  value is one of a few closed-form candidates. Shapes that overlap are 0 apart, their two points one point inside
  both.
- point_velocity: the velocity of a point fixed in a body -- its linear velocity at its origin plus the spin, w x r.
- hri_step: the step's distance, the two approach speeds along the line between the closest points
  (directed_separation.approach_speeds -- when the points touch, along the previous step's line), S_p for the
  carried box's mass (empty once it is let go) and whether the step breaches it.

Pure numpy, no Isaac: the recorder calls it every step and a test can check it on distances worked out by hand.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from generation import directed_separation, visibility
from generation.dummy.joints_spec import JointedSpec
from generation.dummy.skin import region_split

# The region whose body is the person's root -- observation.human.root_pose, the depth of the 50 px rule and the
# position handed to a behaviour layer. The dummy's articulation root is the back (BR-04); the pelvis is where a
# person's position is read from.
PELVIS_REGION = "BR-07"
SHAPE_KINDS = ("sphere", "capsule", "box")
AXES = ("x", "y", "z")
# Numbers in a Shape's half: the radius for a sphere; the radius and half the cylinder's length for a capsule; the
# three half sizes for a box.
HALF_COUNTS = {"sphere": 1, "capsule": 2, "box": 3}
# Two closest points nearer than this touch: there is no line between them to measure an approach along.
TOUCHING_M = 1e-6
# Rounding room when a pair's lower bound is compared with the least upper bound.
_BOUND_SLACK_M = 1e-9
# A part's quaternion this far from unit length is a mistake; nearer is the engine's float32, renormalised.
_POSE_UNIT_TOL = 1e-4
# A box's twelve edges in its own frame: each starts at the corner with these signs times the half sizes and runs
# along the axis whose sign is -1, the whole length of the box.
_EDGE_SIGNS = np.array([[-1.0, sy, sz] for sy in (-1, 1) for sz in (-1, 1)]
                       + [[sx, -1.0, sz] for sx in (-1, 1) for sz in (-1, 1)]
                       + [[sx, sy, -1.0] for sx in (-1, 1) for sy in (-1, 1)])
_EDGE_AXES = np.repeat(np.arange(3), 4)


def _number(value: object) -> bool:
    return (isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value)))


# ---- the person's parts ----

@dataclass(frozen=True)
class Shape:
    """A part's primitive about the origin of its body: ``kind`` "sphere", "capsule" or "box"; ``half`` the radius
    (sphere), the radius and half the cylinder's length (capsule) or the three half sizes (box); ``axis`` the
    capsule's axis in the body's frame ("x", "y" or "z"), None for the others."""
    kind: str
    half: tuple
    axis: str | None

    def __post_init__(self):
        if self.kind not in SHAPE_KINDS:
            raise ValueError(f"Shape.kind 가 {list(SHAPE_KINDS)} 중 하나가 아닙니다: {self.kind!r}")
        n = HALF_COUNTS[self.kind]
        half = list(self.half) if isinstance(self.half, (tuple, list, np.ndarray)) else []
        if len(half) != n or not all(_number(v) and float(v) > 0.0 for v in half):
            raise ValueError(f"Shape.half 가 {self.kind} 의 양의 유한한 수 {n} 개가 아닙니다: {self.half!r}")
        object.__setattr__(self, "half", tuple(float(v) for v in half))
        if self.kind == "capsule":
            if self.axis not in AXES:
                raise ValueError(f"Shape.axis 가 캡슐 축 {list(AXES)} 중 하나가 아닙니다: {self.axis!r}")
        elif self.axis is not None:
            raise ValueError(f"Shape.axis 는 캡슐만 가집니다 ({self.kind}): {self.axis!r}")


def _shape_of(p: region_split.Primitive) -> Shape:
    if p.shape == "sphere":
        return Shape("sphere", (p.radius,), None)
    if p.shape == "capsule":
        return Shape("capsule", (p.radius, p.height / 2.0), p.axis)
    return Shape("box", tuple(np.asarray(p.size, float) / 2.0), None)


def human_shapes(spec: JointedSpec) -> dict[str, Shape]:
    """{body prim name: its part's Shape} for every region prim and the two feet, in the builder's prim order --
    the same names and order as visibility.body_sample_points."""
    return {name: _shape_of(p) for name, p in visibility.body_primitives(spec).items()}


def pelvis_body(spec: JointedSpec) -> str:
    """The body prim of the pelvis (PELVIS_REGION) -- the person's root."""
    region = spec.anthropometry["regions"].get(PELVIS_REGION)
    if not isinstance(region, Mapping):
        raise ValueError(f"인형 치수에 골반 부위 {PELVIS_REGION} 가 없습니다")
    names = [name for name, _ in region_split.prim_names(PELVIS_REGION, region)]
    if len(names) != 1:
        raise ValueError(f"골반 부위 {PELVIS_REGION} 의 프림이 하나가 아닙니다: {names}")
    return names[0]


# ---- the closest pair ----

@dataclass(frozen=True, eq=False)
class Closest:
    """The nearest robot box and part: their distance (0 when they overlap), a point on each (the same point inside
    both when they overlap) and their places in the two sequences closest_pair was given."""
    distance: float
    robot_point: np.ndarray
    human_point: np.ndarray
    robot_index: int
    human_index: int


def _matrices(quats: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotations of (N, 4) unit quaternions (xyzw)."""
    x, y, z, w = quats.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1)], axis=1)


def _parts(human: Sequence) -> tuple[list[Shape], np.ndarray, np.ndarray]:
    shapes, centres, quats = [], [], []
    for i, entry in enumerate(human):
        if not isinstance(entry, (tuple, list)) or len(entry) != 3 or not isinstance(entry[0], Shape):
            raise ValueError(f"사람 부위 {i} 가 (Shape, 자리, 사원수) 가 아닙니다: {entry!r}")
        centre = np.asarray(entry[1], float).reshape(-1)
        quat = np.asarray(entry[2], float).reshape(-1)
        if centre.shape != (3,) or not np.isfinite(centre).all():
            raise ValueError(f"사람 부위 {i} 의 자리가 유한한 수 셋이 아닙니다: {entry[1]!r}")
        norm = float(np.linalg.norm(quat)) if quat.shape == (4,) and np.isfinite(quat).all() else math.nan
        if not abs(norm - 1.0) <= _POSE_UNIT_TOL:
            raise ValueError(f"사람 부위 {i} 의 사원수가 단위 길이의 유한한 수 넷이 아닙니다: {entry[2]!r}")
        shapes.append(entry[0])
        centres.append(centre)
        quats.append(quat / norm)
    return shapes, np.array(centres).reshape(-1, 3), np.array(quats).reshape(-1, 4)


def _segment_box(a: np.ndarray, d: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For N segments a + t d (t in [0, 1]) and N boxes |x| <= h, all (N, 3) in each box's own frame: the t of the
    segment point nearest its box and that point's squared distance to the box, both (N,).

    The squared distance, sum_k max(|a_k + t d_k| - h_k, 0)^2, is convex and quadratic between the t where a
    coordinate crosses a face plane (at most six): its least value is at one of those t, an end, or the clipped
    stationary point of the quadratic that holds between two of them."""
    n = len(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = np.concatenate([(h - a) / d, (-h - a) / d], axis=1)
    cross = np.where(np.isfinite(cross), np.clip(cross, 0.0, 1.0), 0.0)
    bounds = np.sort(np.concatenate([np.zeros((n, 1)), np.ones((n, 1)), cross], axis=1), axis=1)
    lo, hi = bounds[:, :-1], bounds[:, 1:]
    mid = (lo + hi) / 2.0
    x = a[:, None, :] + mid[..., None] * d[:, None, :]
    outside = np.abs(x) > h[:, None, :]
    sign = np.sign(x)
    alpha = np.where(outside, sign * a[:, None, :] - h[:, None, :], 0.0)
    beta = np.where(outside, sign * d[:, None, :], 0.0)
    den = (beta * beta).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        stationary = np.where(den > 0.0, -(alpha * beta).sum(axis=-1) / den, mid)
    candidates = np.concatenate([bounds, np.clip(stationary, lo, hi)], axis=1)
    points = a[:, None, :] + candidates[..., None] * d[:, None, :]
    squared = (np.maximum(np.abs(points) - h[:, None, :], 0.0) ** 2).sum(axis=-1)
    best = np.argmin(squared, axis=1)
    rows = np.arange(n)
    return candidates[rows, best], squared[rows, best]


def _to_local(points: np.ndarray, centres: np.ndarray, mats: np.ndarray) -> np.ndarray:
    """Points (..., 3) into the frames (centre, rotation) they are paired with, broadcast over the leading axes."""
    return np.einsum("...ji,...j->...i", mats, points - centres)


def _rounded(rc, rm, rh, ia, a, b, radius):
    """Robot boxes ``ia`` against segments a -> b grown by ``radius`` (spheres and capsules), all (P, ...) world:
    (distance, robot point, human point) each (P, ...)."""
    centre, mat, half = rc[ia], rm[ia], rh[ia]
    a_loc = _to_local(a, centre, mat)
    d_loc = np.einsum("pji,pj->pi", mat, b - a)
    t, squared = _segment_box(a_loc, d_loc, half)
    core_loc = a_loc + t[:, None] * d_loc
    box_loc = np.clip(core_loc, -half, half)
    core = centre + np.einsum("pij,pj->pi", mat, core_loc)
    robot = centre + np.einsum("pij,pj->pi", mat, box_loc)
    gap = np.sqrt(squared)
    apart = gap > radius
    with np.errstate(divide="ignore", invalid="ignore"):
        toward = np.where(apart[:, None], (robot - core) / gap[:, None], 0.0)
    human = np.where(apart[:, None], core + radius[:, None] * toward, robot)
    return np.where(apart, gap - radius, 0.0), robot, human


def _edges(centre: np.ndarray, mat: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every box's twelve edges in the world: (P, 12, 3) starts and (P, 12, 3) runs."""
    start = centre[:, None, :] + np.einsum("pij,pkj->pki", mat, _EDGE_SIGNS[None, :, :] * half[:, None, :])
    run = 2.0 * half[:, _EDGE_AXES][..., None] * np.swapaxes(mat[:, :, _EDGE_AXES], 1, 2)
    return start, run


def _boxes(rc, rm, rh, ia, hc, hm, hh):
    """Robot boxes against the person's boxes, pair by pair (P,): the least over the twelve edges of each box against
    the other box -- a vertex is on an edge, so this is the boxes' distance, and 0 when they overlap."""
    p = len(ia)
    r_centre, r_mat, r_half = rc[ia], rm[ia], rh[ia]
    starts_h, runs_h = _edges(hc, hm, hh)
    starts_r, runs_r = _edges(r_centre, r_mat, r_half)
    a = np.concatenate([_to_local(starts_h, r_centre[:, None, :], r_mat[:, None, :, :]),
                        _to_local(starts_r, hc[:, None, :], hm[:, None, :, :])], axis=1)
    d = np.concatenate([np.einsum("pji,pkj->pki", r_mat, runs_h), np.einsum("pji,pkj->pki", hm, runs_r)], axis=1)
    half = np.concatenate([np.repeat(r_half[:, None, :], 12, axis=1), np.repeat(hh[:, None, :], 12, axis=1)], axis=1)
    t, squared = _segment_box(a.reshape(-1, 3), d.reshape(-1, 3), half.reshape(-1, 3))
    t, squared = t.reshape(p, 24), squared.reshape(p, 24)
    best = np.argmin(squared, axis=1)
    rows = np.arange(p)
    edge_loc = a[rows, best] + t[rows, best][:, None] * d[rows, best]
    on_box_loc = np.clip(edge_loc, -half[rows, best], half[rows, best])
    human_edge = best < 12                                     # the person's edge against the robot's box
    frame_c = np.where(human_edge[:, None], r_centre, hc)
    frame_m = np.where(human_edge[:, None, None], r_mat, hm)
    edge = frame_c + np.einsum("pij,pj->pi", frame_m, edge_loc)
    on_box = frame_c + np.einsum("pij,pj->pi", frame_m, on_box_loc)
    robot = np.where(human_edge[:, None], on_box, edge)
    human = np.where(human_edge[:, None], edge, on_box)
    distance = np.sqrt(squared[rows, best])
    human = np.where((distance > 0.0)[:, None], human, robot)
    return distance, robot, human


def closest_pair(robot: Sequence[visibility.Obb], human: Sequence[tuple]) -> Closest:
    """The nearest of the robot's boxes and the person's parts. ``robot`` is turned boxes (visibility.Obb); ``human``
    is (Shape, the body's position, its quaternion xyzw) per part, the Shape about the body's origin. Pairs that a
    bound shows cannot be nearest are left out; of the rest, the least distance, then the lowest robot index, then
    the lowest human index."""
    if not len(robot):
        raise ValueError("로봇 상자가 없습니다")
    if not len(human):
        raise ValueError("사람 부위가 없습니다")
    if not all(isinstance(b, visibility.Obb) for b in robot):
        raise ValueError("로봇 상자는 visibility.Obb 여야 합니다")
    rc = np.array([b.center for b in robot])
    rm = _matrices(np.array([b.rot_xyzw for b in robot]) / np.linalg.norm([b.rot_xyzw for b in robot], axis=1)[:, None])
    rh = np.array([b.half for b in robot])
    shapes, hc, hq = _parts(human)
    hm = _matrices(hq)
    radius = np.array([s.half[0] if s.kind != "box" else 0.0 for s in shapes])
    reach = np.array([s.half[0] + (s.half[1] if s.kind == "capsule" else 0.0) if s.kind != "box"
                      else float(np.linalg.norm(s.half)) for s in shapes])

    # Bounds from each part's centre (inside the part, on the core of a rounded one): the distance from the centre
    # to a box less the part's reach is a lower bound, less its radius an upper one.
    excess = np.maximum(np.abs(_to_local(hc[None, :, :], rc[:, None, :], rm[:, None, :, :])) - rh[:, None, :], 0.0)
    centre_gap = np.linalg.norm(excess, axis=-1)                                # (R, S)
    upper = np.maximum(centre_gap - radius[None, :], 0.0)
    lower = np.maximum(centre_gap - reach[None, :], 0.0)
    ia, ib = np.nonzero(lower <= upper.min() + _BOUND_SLACK_M)

    rounded = np.array([shapes[j].kind != "box" for j in ib], bool)
    distance = np.empty(len(ia))
    robot_point = np.empty((len(ia), 3))
    human_point = np.empty((len(ia), 3))
    if rounded.any():
        js = ib[rounded]
        along = np.zeros((len(js), 3))
        for k, j in enumerate(js):
            if shapes[j].kind == "capsule":
                along[k] = hm[j][:, AXES.index(shapes[j].axis)] * shapes[j].half[1]
        out = _rounded(rc, rm, rh, ia[rounded], hc[js] - along, hc[js] + along, radius[js])
        distance[rounded], robot_point[rounded], human_point[rounded] = out
    if (~rounded).any():
        js = ib[~rounded]
        half = np.array([shapes[j].half for j in js]).reshape(-1, 3)
        out = _boxes(rc, rm, rh, ia[~rounded], hc[js], hm[js], half)
        distance[~rounded], robot_point[~rounded], human_point[~rounded] = out
    k = int(np.lexsort((ib, ia, distance))[0])
    return Closest(distance=float(distance[k]), robot_point=robot_point[k].copy(), human_point=human_point[k].copy(),
                   robot_index=int(ia[k]), human_index=int(ib[k]))


# ---- speeds and the step ----

def _vector(where: str, value: object) -> np.ndarray:
    v = np.asarray(value, float).reshape(-1) if value is not None else np.zeros(0)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError(f"{where} 가 유한한 수 셋이 아닙니다: {value!r}")
    return v


def point_velocity(origin, lin_vel, ang_vel, point) -> np.ndarray:
    """The velocity of ``point``, fixed in a body whose ``origin`` moves at ``lin_vel`` and which spins at ``ang_vel``
    (world, rad/s): lin_vel + ang_vel x (point - origin)."""
    o, v, w, p = (_vector(name, x) for name, x in
                  (("origin", origin), ("lin_vel", lin_vel), ("ang_vel", ang_vel), ("point", point)))
    return v + np.cross(w, p - o)


def _state(where: str, value: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{where} 가 (원점, 선속도, 각속도) 가 아닙니다: {value!r}")
    return tuple(_vector(f"{where}[{k}]", x) for k, x in enumerate(value))


def _real(where: str, value: object) -> float:
    if not _number(value):
        raise ValueError(f"{where} 가 유한한 실수가 아닙니다: {value!r}")
    return float(value)


def _unit(where: str, value: object) -> np.ndarray:
    v = _vector(where, value)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError(f"{where} 가 방향이 아닙니다 (길이 {norm}): {value!r}")
    return v / norm


@dataclass(frozen=True, eq=False)
class HriStep:
    """One step's HRI geometry: the surface distance (m), the robot's and the person's approach speeds (m/s, 0 and
    up), S_p (m), whether the step breaches it, the two closest points and the unit direction the speeds were read
    along (robot point -> person point; the previous step's when the points touch) -- the next step's
    ``last_direction``."""
    distance: float
    v_r: float
    v_h: float
    s_p: float
    violated: bool
    robot_point: np.ndarray
    human_point: np.ndarray
    direction: np.ndarray


def hri_step(*, robot_obbs: Sequence[visibility.Obb], robot_states: Sequence, box_obb: visibility.Obb | None,
             box_state, released: bool, human: Sequence[tuple], human_states: Sequence, robot_speed: float,
             yaw_rate: float, box_mass_kg, params: Mapping[str, float], table: Mapping,
             last_direction) -> HriStep:
    """The step's HRI geometry (HriStep).

    ``robot_obbs`` are the robot's link boxes in the world and ``robot_states`` each one's link state, (origin,
    linear velocity, angular velocity) in the world; ``box_obb`` / ``box_state`` the carried box and its state (both
    None when there is no box). ``released``: the box has left the robot -- it is left out of the robot and S_p reads
    the empty rows. ``human`` is (Shape, position, quaternion xyzw) per part (human_shapes placed by the bodies'
    poses) and ``human_states`` each part's body state. ``robot_speed`` is the root's horizontal speed and
    ``yaw_rate`` its yaw rate; ``box_mass_kg`` the carried box's mass, which picks the stop table's state
    (directed_separation.carry_state). ``params`` and ``table`` are directed_separation.load_directed_ssm's.
    ``last_direction`` is the previous step's ``direction`` -- on the first step, from the robot's root to the
    person's pelvis: the closest points of touching shapes coincide and have no line of their own, and without it
    every contact would read as one with a robot not approaching.

    Numbers may be numpy scalars; they are handed on as Python floats."""
    links = list(robot_obbs)
    states = [_state(f"robot_states[{i}]", s) for i, s in enumerate(robot_states)]
    if len(states) != len(links):
        raise ValueError(f"robot_states 가 {len(states)} 개 — robot_obbs {len(links)} 개와 같아야 합니다")
    if (box_obb is None) != (box_state is None):
        missing = "box_state" if box_state is None else "box_obb"
        raise ValueError(f"든 상자는 box_obb 와 box_state 를 함께 줍니다 — {missing} 가 없습니다")
    if not isinstance(released, bool):
        raise ValueError(f"released 가 bool 이 아닙니다: {released!r}")
    holding = box_obb is not None and not released
    if holding:
        links.append(box_obb)
        states.append(_state("box_state", box_state))
    parts = list(human)
    part_states = [_state(f"human_states[{i}]", s) for i, s in enumerate(human_states)]
    if len(part_states) != len(parts):
        raise ValueError(f"human_states 가 {len(part_states)} 개 — human {len(parts)} 개와 같아야 합니다")
    speed = _real("robot_speed", robot_speed)
    rate = _real("yaw_rate", yaw_rate)
    state = directed_separation.carry_state(table, _real("box_mass_kg", box_mass_kg)) if holding else "empty"
    previous = _unit("last_direction", last_direction)

    near = closest_pair(links, parts)
    v_robot = point_velocity(*states[near.robot_index], near.robot_point)
    v_human = point_velocity(*part_states[near.human_index], near.human_point)
    if near.distance < TOUCHING_M:
        direction = previous
        v_r, v_h = directed_separation.approach_speeds(near.robot_point, near.robot_point + direction, v_robot,
                                                       v_human)
    else:
        gap = near.human_point - near.robot_point
        direction = gap / float(np.linalg.norm(gap))
        v_r, v_h = directed_separation.approach_speeds(near.robot_point, near.human_point, v_robot, v_human)
    s_p = directed_separation.separation_distance(params, table, state=state, robot_speed=speed, yaw_rate=rate,
                                                  v_r=v_r, v_h=v_h)
    threshold = _real("params.approach_speed_threshold", params.get("approach_speed_threshold"))
    return HriStep(distance=near.distance, v_r=float(v_r), v_h=float(v_h), s_p=float(s_p),
                   violated=directed_separation.separation_violated(near.distance, s_p, v_r, threshold),
                   robot_point=near.robot_point, human_point=near.human_point, direction=direction.copy())
