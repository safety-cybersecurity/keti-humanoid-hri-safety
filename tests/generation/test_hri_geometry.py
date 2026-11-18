"""The HRI geometry of a step (generation/hri_geometry.py): the closest points of the robot's boxes and the dummy's
parts, the speeds of those points, the directed S_p they feed and what the robot's cameras see past the box it
carries. The closest pairs are checked on distances worked out by hand here; the speeds and S_p against
generation/directed_separation.py; the canon (meta/rules.json, meta/cameras.json, the dummy's parts) where the
geometry reads it. Isaac-free."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from generation import cameras, carry, directed_separation, hri_geometry as hri, visibility
from generation.dummy import joints_spec
from generation.robots import kinematics
from generation.scene.carried_box import NOMINAL_SIZE_M

REPO = Path(__file__).resolve().parents[2]
COMMANDS = (REPO / "generation" / "envs" / "hri" / "commands.py").read_text(encoding="utf-8")
TOL = 1e-4
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
S2 = math.sqrt(2.0)


def _about(axis: str, deg: float) -> np.ndarray:
    """The quaternion (xyzw) of a turn of ``deg`` about the world axis ``axis`` -- written out here."""
    half = math.radians(deg) / 2.0
    q = np.zeros(4)
    q["xyz".index(axis)] = math.sin(half)
    q[3] = math.cos(half)
    return q


def _obb(centre, half, rot=IDENTITY) -> visibility.Obb:
    return visibility.Obb(center=np.asarray(centre, float), rot_xyzw=np.asarray(rot, float),
                          half=np.asarray(half, float))


def _sphere(r):
    return hri.Shape("sphere", (r,), None)


def _capsule(r, half_length, axis="z"):
    return hri.Shape("capsule", (r, half_length), axis)


def _box(*half):
    return hri.Shape("box", tuple(half), None)


def _part(shape, centre, rot=IDENTITY):
    return (shape, np.asarray(centre, float), np.asarray(rot, float))


@pytest.fixture(scope="module")
def ssm():
    rules = json.loads((REPO / "meta" / "rules.json").read_text(encoding="utf-8"))
    return directed_separation.load_directed_ssm(rules)


@pytest.fixture(scope="module")
def spec():
    return joints_spec.load()


# ---- the closest pair, on distances worked out by hand --------------------------------------------------------

def _edge_to_edge():
    """Box A: a 0.2 m cube at the origin. Box B: a 0.2 m cube turned so that one of its edges runs along
    e = (1, 0, -1)/sqrt2 past A's top edge on the +x side (x = z = 0.1, along y) at g = 0.07 m, the two edges
    crossing at right angles -- B's faces at that edge lean away from A (its other two axes are
    (n + y)/sqrt2 and (n - y)/sqrt2 with n = (1, 0, 1)/sqrt2), so the closest points are the two edges' and
    nothing else: A's (0.1, 0, 0.1) and B's (0.1, 0, 0.1) + g n. B's centre is that point + 0.1 sqrt2 n."""
    g = 0.07
    n = np.array([1.0, 0.0, 1.0]) / S2
    e = np.array([1.0, 0.0, -1.0]) / S2
    y = np.array([0.0, 1.0, 0.0])
    rot = np.column_stack([e, (n + y) / S2, (n - y) / S2])
    a_point = np.array([0.1, 0.0, 0.1])
    b_point = a_point + g * n
    return g, a_point, b_point, b_point + 0.1 * S2 * n, kinematics.quat_from_matrix(rot)


def _analytic_cases():
    cases = []
    # A sphere beside a box turned 90 degrees about z: the box's world half sizes are (0.1, 0.2, 0.3), so the
    # nearest box point to the centre (0.5, 0.35, 0.1) is its corner edge (0.1, 0.2, 0.1).
    centre = np.array([0.5, 0.35, 0.1])
    corner = np.array([0.1, 0.2, 0.1])
    gap = centre - corner
    length = math.sqrt(0.4 ** 2 + 0.15 ** 2)
    cases.append(pytest.param(
        [_obb([0, 0, 0], [0.2, 0.1, 0.3], _about("z", 90.0))], [_part(_sphere(0.05), centre)],
        length - 0.05, corner, centre - 0.05 * gap / length, id="sphere-turned box"))
    # A capsule of half length 0.2 turned 45 degrees about y beside a 0.4 m cube: its axis runs along
    # (sin 45, 0, cos 45); its lower end (0.6 - 0.2 sin 45, 0, -0.2 cos 45) faces the cube's +x face.
    end = np.array([0.6 - 0.2 * math.sin(math.radians(45.0)), 0.0, -0.2 * math.cos(math.radians(45.0))])
    cases.append(pytest.param(
        [_obb([0, 0, 0], [0.2, 0.2, 0.2])], [_part(_capsule(0.05, 0.2), [0.6, 0.0, 0.0], _about("y", 45.0))],
        end[0] - 0.2 - 0.05, np.array([0.2, 0.0, end[2]]), end - np.array([0.05, 0.0, 0.0]), id="capsule-box"))
    # Two unturned boxes face to face along x: the robot's +x face at 0.1, the person's -x face at 0.4 - 0.05.
    cases.append(pytest.param(
        [_obb([0, 0, 0], [0.1, 0.1, 0.1])], [_part(_box(0.05, 0.2, 0.1), [0.4, 0.05, 0.02])],
        0.25, None, None, id="parallel boxes"))
    # A 0.2 m cube turned 45 degrees about z in front of a 0.4 m cube: its edge at x = 0.5 - 0.1 sqrt2.
    cases.append(pytest.param(
        [_obb([0, 0, 0], [0.2, 0.2, 0.2])], [_part(_box(0.1, 0.1, 0.1), [0.5, 0.0, 0.05], _about("z", 45.0))],
        0.5 - 0.1 * S2 - 0.2, None, None, id="oblique box"))
    # The same with the roles swapped: the robot's cube turned 45 degrees about z, its edge at x = 0.1 sqrt2 facing
    # the person's unturned 0.4 m cube whose -x face is at 0.5 - 0.2.
    cases.append(pytest.param(
        [_obb([0, 0, 0], [0.1, 0.1, 0.1], _about("z", 45.0))], [_part(_box(0.2, 0.2, 0.2), [0.5, 0.0, 0.05])],
        0.3 - 0.1 * S2, None, None, id="oblique robot box"))
    g, a_point, b_point, b_centre, b_rot = _edge_to_edge()
    cases.append(pytest.param(
        [_obb([0, 0, 0], [0.1, 0.1, 0.1])], [_part(_box(0.1, 0.1, 0.1), b_centre, b_rot)],
        g, a_point, b_point, id="edge to edge, the person's edge"))
    cases.append(pytest.param(
        [_obb(b_centre, [0.1, 0.1, 0.1], b_rot)], [_part(_box(0.1, 0.1, 0.1), [0, 0, 0])],
        g, b_point, a_point, id="edge to edge, the robot's edge"))
    return cases


@pytest.mark.parametrize("robot,human,distance,robot_point,human_point", _analytic_cases())
def test_closest_pair_matches_analytic_cases(robot, human, distance, robot_point, human_point):
    near = hri.closest_pair(robot, human)
    assert near.distance == pytest.approx(distance, abs=TOL)
    assert float(np.linalg.norm(near.human_point - near.robot_point)) == pytest.approx(distance, abs=TOL)
    if robot_point is not None:
        assert near.robot_point == pytest.approx(robot_point, abs=TOL)
        assert near.human_point == pytest.approx(human_point, abs=TOL)
    assert (near.robot_index, near.human_index) == (0, 0)


def test_the_parallel_boxes_meet_where_their_faces_overlap():
    near = hri.closest_pair([_obb([0, 0, 0], [0.1, 0.1, 0.1])], [_part(_box(0.05, 0.2, 0.1), [0.4, 0.05, 0.02])])
    assert near.robot_point[0] == pytest.approx(0.1, abs=TOL)
    assert near.human_point[0] == pytest.approx(0.35, abs=TOL)
    assert near.human_point[1:] == pytest.approx(near.robot_point[1:], abs=TOL)
    assert -0.1 - TOL <= near.robot_point[1] <= 0.1 + TOL        # y where both faces are: [-0.1, 0.1]
    assert -0.08 - TOL <= near.robot_point[2] <= 0.1 + TOL       # z where both faces are: [-0.08, 0.1]


@pytest.mark.parametrize("robot,human,robot_xy,human_xy,z_range", [
    pytest.param(_obb([0, 0, 0], [0.2, 0.2, 0.2]), _part(_box(0.1, 0.1, 0.1), [0.5, 0.0, 0.05], _about("z", 45.0)),
                 [0.2, 0.0], [0.5 - 0.1 * S2, 0.0], (-0.05, 0.15), id="the person's edge"),
    pytest.param(_obb([0, 0, 0], [0.1, 0.1, 0.1], _about("z", 45.0)), _part(_box(0.2, 0.2, 0.2), [0.5, 0.0, 0.05]),
                 [0.1 * S2, 0.0], [0.3, 0.0], (-0.1, 0.1), id="the robot's edge"),
])
def test_the_oblique_box_meets_at_its_edge(robot, human, robot_xy, human_xy, z_range):
    near = hri.closest_pair([robot], [human])
    assert near.robot_point[:2] == pytest.approx(robot_xy, abs=TOL)
    assert near.human_point[:2] == pytest.approx(human_xy, abs=TOL)
    # Anywhere along the upright edge where the other cube's face is too: both points at the same height.
    assert near.human_point[2] == pytest.approx(near.robot_point[2], abs=TOL)
    assert z_range[0] - TOL <= near.robot_point[2] <= z_range[1] + TOL


def _tilted_capsule(deg, gap, r=0.05, half_length=0.2):
    """A capsule along x tilted up by ``deg`` (a turn of -deg about y) whose lower end is ``gap`` above the
    top face (z = 0.2) of a 0.4 m cube, over that face: its lowest point is the lower end's cap."""
    t = math.radians(deg)
    low = np.array([-0.15, 0.0, 0.2 + r + gap])
    return _part(_capsule(r, half_length, "x"), low + half_length * np.array([math.cos(t), 0.0, math.sin(t)]),
                 _about("y", -deg))


def _tilted_box(deg, gap, half=(0.15, 0.1, 0.05)):
    """A box tilted up by ``deg`` (a turn of -deg about y) whose lowest edge -- local x = -hx, z = -hz -- is
    ``gap`` above the top face (z = 0.2) of a 0.4 m cube, over that face."""
    t = math.radians(deg)
    hx, _, hz = half
    lowest = -hx * math.sin(t) - hz * math.cos(t)              # local (-hx, ., -hz) turned: z' = x sin t + z cos t
    return _part(_box(*half), [0.0, 0.0, 0.2 + gap - lowest], _about("y", -deg))


@pytest.mark.parametrize("part,gap", [
    pytest.param(_tilted_capsule(3.0, 0.0), 0.0, id="capsule at 3 deg touching"),
    pytest.param(_tilted_capsule(5.0, 0.0), 0.0, id="capsule at 5 deg touching"),
    pytest.param(_tilted_capsule(3.0, 0.01), 0.01, id="capsule at 3 deg, 1 cm"),
    pytest.param(_tilted_capsule(0.5, 0.05), 0.05, id="capsule at half a degree, 5 cm"),
    pytest.param(_tilted_box(3.0, 0.0), 0.0, id="box at 3 deg touching"),
    pytest.param(_tilted_box(2.0, 0.02), 0.02, id="box at 2 deg, 2 cm"),
])
def test_a_nearly_parallel_part_is_measured_exactly(part, gap):
    # A part lying almost flat on a face is where sliding toward the touching end is slowest; the distance must
    # still be the hand-worked gap, not a few millimetres short of the end.
    near = hri.closest_pair([_obb([0, 0, 0], [0.2, 0.2, 0.2])], [part])
    assert near.distance == pytest.approx(gap, abs=1e-6)


def _inside_box(point, obb, tol=1e-9) -> bool:
    x, y, z, w = obb.rot_xyzw
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return bool(np.all(np.abs(rot.T @ (np.asarray(point) - obb.center)) <= obb.half + tol))


CUBE = [0.2, 0.2, 0.2]


@pytest.mark.parametrize("robot_half,human,inside", [
    pytest.param(CUBE, _part(_sphere(0.05), [0.1, 0.0, 0.0]),
                 lambda p: np.linalg.norm(p - [0.1, 0, 0]) <= 0.05 + 1e-9, id="a sphere's centre in the box"),
    pytest.param(CUBE, _part(_capsule(0.02, 0.5, "x"), [0.0, 0.0, 0.0]),
                 lambda p: math.hypot(p[1], p[2]) <= 0.02 + 1e-9 and abs(p[0]) <= 0.52 + 1e-9,
                 id="a capsule through the box"),
    pytest.param([0.5, 0.05, 0.05], _part(_box(0.05, 0.5, 0.05), [0.0, 0.0, 0.0]),
                 lambda p: abs(p[0]) <= 0.05 + 1e-9 and abs(p[1]) <= 0.5 + 1e-9 and abs(p[2]) <= 0.05 + 1e-9,
                 id="two boxes crossing, no corner inside the other"),
    pytest.param(CUBE, _tilted_capsule(3.0, -0.001), None, id="a capsule at 3 deg, 1 mm in"),
    pytest.param(CUBE, _tilted_box(3.0, -0.001), None, id="a box at 3 deg, 1 mm in"),
])
def test_overlap_is_zero_distance(robot_half, human, inside):
    robot = [_obb([0, 0, 0], robot_half)]
    near = hri.closest_pair(robot, [human])
    assert near.distance == 0.0
    assert near.robot_point == pytest.approx(near.human_point, abs=1e-9)
    assert _inside_box(near.robot_point, robot[0])
    if inside is not None:
        assert inside(near.human_point)


# ---- the closest pair against a reference written here, on seeded random pairs ------------------------------------
#
# The reference does not use the closed form's candidates. A point's distance to a box is the clipped excess. A
# segment's distance to a box is convex along the segment, so a golden-section search over it finds the least. Two
# boxes that no separating axis parts (the fifteen axes of the separating axis theorem) overlap; two that one parts are
# nearest at a vertex of one against the other or at a pair of their edges (vertex-face, edge-edge -- a face-to-face or
# edge-to-face pair is nearest where one of those is), and two segments' nearest points are worked out in closed form.

DENSE_PAIRS = 600
DENSE_TOL_M = 1e-9          # closed form against the reference, and the closest points against their shapes
_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def _matrix(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _ref_point_box(p, centre, rot, half):
    local = rot.T @ (np.asarray(p, float) - centre)
    return float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0)))


def _ref_segment_box(a, b, centre, rot, half):
    f = lambda t: _ref_point_box(a + t * (b - a), centre, rot, half)   # noqa: E731 -- convex in t
    lo, hi = 0.0, 1.0
    g = (math.sqrt(5.0) - 1.0) / 2.0
    m1, m2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = f(m1), f(m2)
    for _ in range(200):
        if f1 <= f2:
            hi, m2, f2 = m2, m1, f1
            m1 = hi - g * (hi - lo)
            f1 = f(m1)
        else:
            lo, m1, f1 = m1, m2, f2
            m2 = lo + g * (hi - lo)
            f2 = f(m2)
    return min(f(0.0), f(1.0), f1, f2)


def _ref_segment_segment(p1, q1, p2, q2):
    """The least distance between two segments (the closest points of their lines, clamped in turn)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    denom = a * e - b * b
    s = min(max((b * f - c * e) / denom, 0.0), 1.0) if denom > 1e-18 else 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t, s = 0.0, min(max(-c / a, 0.0), 1.0)
    elif t > 1.0:
        t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def _ref_box_box(ca, ra, ha, cb, rb, hb):
    axes = [ra[:, i] for i in range(3)] + [rb[:, j] for j in range(3)]
    axes += [np.cross(ra[:, i], rb[:, j]) for i in range(3) for j in range(3)]
    parted = False
    for axis in axes:
        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            continue
        u = axis / n
        if abs(float((cb - ca) @ u)) > float(np.abs(ra.T @ u) @ ha + np.abs(rb.T @ u) @ hb):
            parted = True
            break
    if not parted:
        return 0.0
    va = ca + (_CORNERS * ha) @ ra.T
    vb = cb + (_CORNERS * hb) @ rb.T
    best = min(min(_ref_point_box(v, cb, rb, hb) for v in va), min(_ref_point_box(v, ca, ra, ha) for v in vb))

    def edges(verts):
        return [(verts[i], verts[j]) for i in range(8) for j in range(i + 1, 8)
                if np.count_nonzero(_CORNERS[i] != _CORNERS[j]) == 1]

    for p1, q1 in edges(va):
        for p2, q2 in edges(vb):
            best = min(best, _ref_segment_segment(p1, q1, p2, q2))
    return best


def _random_pairs(seed=20261021, n=DENSE_PAIRS):
    """(robot box, person part) pairs: boxes and parts of every kind at random sizes, places and turns -- some apart,
    some overlapping, some nearly parallel (a turn of under 2 degrees off the robot box's axes)."""
    rng = np.random.default_rng(seed)

    def turn():
        q = rng.normal(size=4)
        return q / np.linalg.norm(q)

    pairs = []
    for k in range(n):
        robot = _obb(rng.uniform(-0.2, 0.2, 3), rng.uniform(0.02, 0.3, 3), turn())
        kind = ("sphere", "capsule", "box")[k % 3]
        if kind == "sphere":
            shape = _sphere(rng.uniform(0.02, 0.15))
        elif kind == "capsule":
            shape = _capsule(rng.uniform(0.02, 0.1), rng.uniform(0.05, 0.3), "xyz"[k % 9 // 3])
        else:
            shape = _box(*rng.uniform(0.02, 0.25, 3))
        if k % 5 == 4:                              # nearly parallel to the robot box
            small = _about("xyz"[k % 3], rng.uniform(-2.0, 2.0))
            rot = kinematics.quat_from_matrix(_matrix(robot.rot_xyzw) @ _matrix(small))
        else:
            rot = turn()
        pairs.append((robot, _part(shape, rng.uniform(-0.35, 0.35, 3), rot)))
    return pairs


def _reference(robot, part):
    shape, centre, quat = part
    rc, rr, rh = robot.center, _matrix(robot.rot_xyzw), robot.half
    rot = _matrix(quat)
    if shape.kind == "sphere":
        return max(_ref_point_box(centre, rc, rr, rh) - shape.half[0], 0.0)
    if shape.kind == "capsule":
        along = rot[:, AXES_INDEX[shape.axis]] * shape.half[1]
        return max(_ref_segment_box(centre - along, centre + along, rc, rr, rh) - shape.half[0], 0.0)
    return _ref_box_box(rc, rr, rh, centre, rot, np.asarray(shape.half))


AXES_INDEX = {"x": 0, "y": 1, "z": 2}


def _in_part(point, part, tol):
    shape, centre, quat = part
    local = _matrix(quat).T @ (np.asarray(point, float) - centre)
    if shape.kind == "sphere":
        return float(np.linalg.norm(local)) <= shape.half[0] + tol
    if shape.kind == "capsule":
        i = AXES_INDEX[shape.axis]
        core = np.zeros(3)
        core[i] = np.clip(local[i], -shape.half[1], shape.half[1])
        return float(np.linalg.norm(local - core)) <= shape.half[0] + tol
    return bool(np.all(np.abs(local) <= np.asarray(shape.half) + tol))


def test_the_closest_pair_agrees_with_a_reference_on_random_pairs():
    pairs = _random_pairs()
    kinds = {"sphere": [0, 0], "capsule": [0, 0], "box": [0, 0]}      # [apart, overlapping]
    worst = 0.0
    for robot, part in pairs:
        near = hri.closest_pair([robot], [part])
        want = _reference(robot, part)
        worst = max(worst, abs(near.distance - want))
        assert near.distance == pytest.approx(want, abs=DENSE_TOL_M), (part[0], want, near.distance)
        # the two points are on their shapes and as far apart as the distance (one point inside both when 0)
        assert _inside_box(near.robot_point, robot, DENSE_TOL_M)
        assert _in_part(near.human_point, part, DENSE_TOL_M)
        gap = float(np.linalg.norm(near.human_point - near.robot_point))
        assert gap == pytest.approx(near.distance, abs=DENSE_TOL_M)
        kinds[part[0].kind][want == 0.0] += 1
    # the draw covers what it says: each kind both apart and overlapping
    assert all(apart >= 50 and over >= 50 for apart, over in kinds.values()), kinds
    assert worst <= DENSE_TOL_M


def test_the_nearest_of_many_pairs_is_named_by_its_indices():
    robot = [_obb([0, 0, 0], [0.1, 0.1, 0.1]), _obb([1.0, 0, 0], [0.1, 0.1, 0.1]), _obb([5.0, 0, 0], [0.1, 0.1, 0.1])]
    human = [_part(_sphere(0.1), [3.0, 0.0, 0.0]), _part(_sphere(0.1), [1.5, 0.0, 0.0]),
             _part(_box(0.1, 0.1, 0.1), [1.0, 1.0, 0.0])]
    near = hri.closest_pair(robot, human)
    # robot 1's +x face (1.1) to sphere 1 (1.5 - 0.1): 0.3; robot 1 to the box above it: 1.0 - 0.1 - 0.1 = 0.8.
    assert (near.robot_index, near.human_index) == (1, 1)
    assert near.distance == pytest.approx(0.3, abs=TOL)


@pytest.mark.parametrize("robot,human,match", [
    ([], [_part(_sphere(0.1), [0, 0, 0])], "로봇"),
    ([_obb([0, 0, 0], [0.1, 0.1, 0.1])], [], "사람"),
    ([_obb([0, 0, 0], [0.1, 0.1, 0.1])], [_part(_sphere(0.1), [0, 0, np.nan])], "자리"),
    ([_obb([0, 0, 0], [0.1, 0.1, 0.1])], [_part(_sphere(0.1), [0, 0, 0], [0, 0, 0, 2.0])], "사원수"),
])
def test_a_pair_that_cannot_be_measured_is_refused(robot, human, match):
    with pytest.raises(ValueError, match=match):
        hri.closest_pair(robot, human)


@pytest.mark.parametrize("args,match", [
    (("cone", (0.1,), None), "kind"),
    (("sphere", (0.1, 0.2), None), "half"),
    (("capsule", (0.1, 0.2), None), "axis"),
    (("capsule", (0.1, 0.2), "w"), "axis"),
    (("box", (0.1, 0.0, 0.1), None), "half"),
    (("box", (0.1, 0.1, 0.1), "z"), "axis"),
])
def test_a_shape_that_is_not_a_primitive_is_refused(args, match):
    with pytest.raises(ValueError, match=match):
        hri.Shape(*args)


# ---- the dummy's parts --------------------------------------------------------------------------------------

def test_the_shapes_are_the_parts_the_sample_points_are_on(spec):
    shapes = hri.human_shapes(spec)
    assert list(shapes) == list(visibility.body_sample_points(spec, 120))


def test_the_shapes_are_the_anthropometry_primitives_about_their_centres(spec):
    shapes = hri.human_shapes(spec)
    regions = spec.anthropometry["regions"]
    assert shapes["BR_01"] == hri.Shape("sphere", (regions["BR-01"]["radius"],), None)
    arm = regions["BR-08"]
    assert shapes["BR_08_left"] == hri.Shape("capsule", (arm["radius"], arm["height"] / 2.0), arm.get("axis", "z"))
    assert shapes["BR_07"] == hri.Shape("box", tuple(v / 2.0 for v in regions["BR-07"]["size"]), None)
    foot = next(f for f in spec.feet if f.side == "left")
    assert shapes[foot.name] == hri.Shape("box", tuple(v / 2.0 for v in foot.size), None)


def test_pelvis_is_br07(spec):
    rules = json.loads((REPO / "meta" / "rules.json").read_text(encoding="utf-8"))
    assert hri.PELVIS_REGION == "BR-07"
    assert rules["contact_limits"][hri.PELVIS_REGION]["name_en"] == "pelvis"
    assert hri.pelvis_body(spec) == "BR_07"
    # The articulation's root is the back (the region without a parent) -- not the body the person's root names.
    roots = [code for code, r in spec.anthropometry["regions"].items() if r.get("parent") is None]
    assert roots == ["BR-04"]


def test_the_site_cameras_measure_the_height_to_the_pelvis(spec):
    from generation.scene import check_site_cameras as csc

    body = visibility.body_sample_points(spec, 120)
    person = csc.standing_person(body, (1.0, 2.0), 0.3, spec)
    centre = spec.anthropometry["regions"]["BR-07"]["center"]
    assert person["root"] == pytest.approx([1.0 + math.cos(0.3) * centre[0] - math.sin(0.3) * centre[1],
                                            2.0 + math.sin(0.3) * centre[0] + math.cos(0.3) * centre[1], centre[2]])


# ---- speeds and the step --------------------------------------------------------------------------------------

def test_point_velocity_adds_the_spin():
    # (0, 0, 2) x (0.5, 0, 0) = (0, 1, 0); (1, 0, 0) x (0, 0, 0.3) = (0, -0.3, 0).
    assert hri.point_velocity([1, 2, 3], [0.1, 0, 0], [0, 0, 2.0], [1.5, 2, 3]) == pytest.approx([0.1, 1.0, 0.0])
    assert hri.point_velocity([0, 0, 0], [0, 0, 0], [1.0, 0, 0], [0, 0, 0.3]) == pytest.approx([0.0, -0.3, 0.0])
    with pytest.raises(ValueError):
        hri.point_velocity([0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])


def _still():
    return (np.zeros(3), np.zeros(3), np.zeros(3))


def _step(ssm, **kw):
    params, table = ssm
    args = dict(robot_obbs=[_obb([0, 0, 1.0], [0.1, 0.1, 0.1])], robot_states=[_still()], box_obb=None,
                box_state=None, released=False, human=[_part(_sphere(0.1), [1.0, 0.0, 1.0])],
                human_states=[_still()], robot_speed=0.0, yaw_rate=0.0, box_mass_kg=0.0, params=params,
                table=table, last_direction=np.array([1.0, 0.0, 0.0]))
    args.update(kw)
    return hri.hri_step(**args)


def test_hri_step_agrees_with_directed_separation(ssm):
    params, table = ssm
    # The robot box's +x face (x 0.1) faces the sphere's near side (x 0.9): distance 0.8 along +x. The link turns
    # about z at 0.4 rad/s round (0, -0.2, 1): at its point (0.1, 0, 1) the spin adds (0, 0, 0.4) x (0.1, 0.2, 0)
    # = (-0.08, 0.04, 0) to the 0.5 m/s, so v_r = 0.42. The person's point (0.9, 0, 1) spins round (1, 0, 1) at
    # 0.5 rad/s: (0, 0, 0.5) x (-0.1, 0, 0) = (0, -0.05, 0) on top of (-0.3, 0.1, 0), so v_h = 0.3.
    robot_state = (np.array([0.0, -0.2, 1.0]), np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.4]))
    human_state = (np.array([1.0, 0.0, 1.0]), np.array([-0.3, 0.1, 0.0]), np.array([0.0, 0.0, 0.5]))
    step = _step(ssm, robot_states=[robot_state], human_states=[human_state], robot_speed=np.float32(0.5),
                 yaw_rate=np.float64(0.4))
    assert step.distance == pytest.approx(0.8, abs=TOL)
    assert step.robot_point == pytest.approx([0.1, 0.0, 1.0], abs=TOL)
    assert step.human_point == pytest.approx([0.9, 0.0, 1.0], abs=TOL)
    assert step.direction == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert (step.v_r, step.v_h) == pytest.approx((0.42, 0.3), abs=1e-9)
    v_r, v_h = directed_separation.approach_speeds([0.1, 0.0, 1.0], [0.9, 0.0, 1.0], [0.42, 0.04, 0.0],
                                                   [-0.3, 0.05, 0.0])
    assert (step.v_r, step.v_h) == pytest.approx((v_r, v_h), abs=1e-9)
    s_p = directed_separation.separation_distance(params, table, state="empty", robot_speed=0.5, yaw_rate=0.4,
                                                  v_r=0.42, v_h=0.3)
    assert step.s_p == pytest.approx(s_p, rel=1e-6)
    assert step.violated == directed_separation.separation_violated(0.8, s_p, 0.42,
                                                                    params["approach_speed_threshold"])
    assert type(step.violated) is bool and type(step.s_p) is float


def _with_box(ssm, *, released, mass=3.0):
    # The link box (x to 0.1) and, ahead of it, a held box (x 0.2 .. 0.6); the person's sphere at x 1.2.
    moving = (np.array([0.0, 0.0, 1.0]), np.array([0.5, 0.0, 0.0]), np.zeros(3))
    return _step(ssm, robot_states=[moving], box_obb=_obb([0.4, 0.0, 1.0], [0.2, 0.15, 0.1]), box_state=moving,
                 released=released, human=[_part(_sphere(0.1), [1.2, 0.0, 1.0])], robot_speed=0.5,
                 box_mass_kg=mass)


def test_a_released_box_leaves_the_robot(ssm):
    params, table = ssm
    held, dropped = _with_box(ssm, released=False), _with_box(ssm, released=True)
    assert held.distance == pytest.approx(1.2 - 0.1 - 0.6, abs=TOL)
    assert held.robot_point[0] == pytest.approx(0.6, abs=TOL)
    assert dropped.distance == pytest.approx(1.2 - 0.1 - 0.1, abs=TOL)
    assert dropped.robot_point[0] == pytest.approx(0.1, abs=TOL)

    def s_p(state):
        return directed_separation.separation_distance(params, table, state=state, robot_speed=0.5, yaw_rate=0.0,
                                                       v_r=0.5, v_h=0.0)

    assert s_p("carry_3kg") != pytest.approx(s_p("empty"))
    assert held.s_p == pytest.approx(s_p("carry_3kg"), rel=1e-9)
    assert dropped.s_p == pytest.approx(s_p("empty"), rel=1e-9)
    # Moving toward the person at 0.5 m/s: held, the box's front is inside the carrying S_p -- a breach; let go, the
    # link 1.0 m off is outside the empty one.
    assert held.distance < held.s_p and held.v_r > params["approach_speed_threshold"]
    assert held.violated is True
    assert dropped.distance > dropped.s_p and dropped.violated is False


@pytest.mark.parametrize("mass,state", [(0.0, "empty"), (1.0, "carry_2kg"), (2.0, "carry_2kg"), (2.5, "carry_3kg")])
def test_the_box_mass_picks_the_carry_row(ssm, mass, state):
    params, table = ssm
    step = _with_box(ssm, released=False, mass=mass)
    read = {s: directed_separation.separation_distance(params, table, state=s, robot_speed=0.5, yaw_rate=0.0,
                                                       v_r=0.5, v_h=0.0) for s in directed_separation.STOP_TABLE_STATES}
    assert len({round(v, 9) for v in read.values()}) == 3          # the three rows tell apart here
    assert step.s_p == pytest.approx(read[state], rel=1e-9)


def test_a_standing_robot_is_never_approaching(ssm):
    params, _ = ssm
    walking = (np.array([0.4, 0.0, 1.0]), np.array([-1.2, 0.0, 0.0]), np.zeros(3))
    step = _step(ssm, human=[_part(_sphere(0.1), [0.4, 0.0, 1.0])], human_states=[walking])
    assert step.distance == pytest.approx(0.2, abs=TOL)
    assert step.v_r == 0.0
    assert step.v_h == pytest.approx(1.2)
    assert step.s_p > step.distance                                 # short of S_p, yet not a breach
    assert step.violated is False
    assert directed_separation.robot_was_standing(step.v_r, params["approach_speed_threshold"])


def test_touching_points_keep_the_last_direction(ssm):
    # The sphere (centre x 0.15, radius 0.1) is inside the link box's face (x 0.1): the closest points coincide and
    # have no direction of their own. The robot moving +x at 0.6 m/s along the last step's direction approaches.
    moving = (np.array([0.0, 0.0, 1.0]), np.array([0.6, 0.0, 0.0]), np.zeros(3))
    walking = (np.array([0.15, 0.0, 1.0]), np.array([-0.2, 0.0, 0.0]), np.zeros(3))
    step = _step(ssm, robot_states=[moving], human=[_part(_sphere(0.1), [0.15, 0.0, 1.0])], human_states=[walking],
                 robot_speed=0.6, last_direction=np.array([2.0, 0.0, 0.0]))
    assert step.distance == 0.0
    assert step.direction == pytest.approx([1.0, 0.0, 0.0])
    assert (step.v_r, step.v_h) == pytest.approx((0.6, 0.2))
    # touching while moving toward the person: 0 is short of any S_p, so the step breaches it
    assert step.violated is True and step.s_p > 0.0
    # Without the last direction the module would read the touching step as a standing robot.
    assert directed_separation.approach_speeds(step.robot_point, step.human_point, [0.6, 0, 0], [-0.2, 0, 0]) \
        == (0.0, 0.0)


@pytest.mark.parametrize("bad", [np.zeros(3), np.array([np.nan, 0.0, 0.0]), np.array([1.0, 0.0])])
def test_a_last_direction_that_is_not_a_direction_is_refused(ssm, bad):
    with pytest.raises(ValueError, match="last_direction"):
        _step(ssm, last_direction=bad)


@pytest.mark.parametrize("kw,match", [
    (dict(robot_speed=True), "robot_speed"),
    (dict(yaw_rate=float("inf")), "yaw_rate"),
    (dict(robot_states=[]), "robot_states"),
    (dict(human_states=[]), "human_states"),
    (dict(box_obb=_obb([0.4, 0, 1.0], [0.1, 0.1, 0.1])), "box_state"),
    (dict(box_state=_still()), "box_obb"),
])
def test_a_step_that_cannot_be_read_is_refused(ssm, kw, match):
    with pytest.raises(ValueError, match=match):
        _step(ssm, **kw)


# ---- what the robot's cameras see -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def carried_robot():
    """The H1 at the S1e hold pose standing on the floor (its ankle boxes' lowest corner at z 0), facing +x, the
    canon box of the nominal size in its arms: (link poses, link boxes, the box)."""
    entry = next(s for s in json.loads((REPO / "meta" / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
                 if s["scenario_id"] == "S1e")
    held = carry.load_carry(entry)
    joints, boxes = kinematics.load_joints(), visibility.load_link_boxes()
    names = yaml.safe_load((REPO / "meta" / "joint_map" / "unitree_h1.yaml").read_text(encoding="utf-8"))["joint_names"]
    poses = kinematics.link_poses(joints, carry.arm_hold_joints(held, names), frame=held.link)
    low = min(float((p + kinematics.quat_to_matrix(q) @ (boxes[k][0] + np.array(s) * boxes[k][1]))[2])
              for k, (p, q) in poses.items() if k.endswith("ankle_link")
              for s in [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    lift = np.array([0.0, 0.0, -low])
    world = {k: np.r_[p + lift, q] for k, (p, q) in poses.items()}
    link_pos, link_quat = poses[held.link]
    return world, boxes, carry.box_obb(link_pos + lift, link_quat, held, NOMINAL_SIZE_M)


def _cameras_canon():
    return json.loads((REPO / "meta" / "cameras.json").read_text(encoding="utf-8"))


def test_camera_fractions_count_the_carried_box(carried_robot, spec):
    canon = _cameras_canon()
    world, boxes, box = carried_robot
    body = visibility.body_sample_points(spec, visibility.seeing_rule(canon)["sample_points"])
    # A person 0.8 m ahead, facing the robot: below and behind the box from the wrists and the downward camera.
    person = visibility.points_world(body, visibility.standing_part_poses(spec, (0.8, 0.0), math.pi))
    lo = hi = np.zeros((0, 3))
    bare = visibility.robot_camera_fractions(canon, world, boxes, None, person, lo, hi)
    held = visibility.robot_camera_fractions(canon, world, boxes, box, person, lo, hi)
    assert list(bare) == list(held) == list(visibility.robot_cameras(canon))
    for name in bare:
        cam = canon["cameras"][name]
        pos, quat = cameras.mounted_pose(world[cam["parent_prim"]], cam["offset_pos"], cam["offset_rot_xyzw"])
        assert bare[name] == visibility.visible_fraction(
            cam, pos, quat, person, lo, hi, visibility.camera_robot_obbs(canon, name, boxes, world))
    for name in ("head_down_rgb", "left_wrist_rgb", "right_wrist_rgb"):
        assert held[name] < bare[name], name
    for name in ("head_rgb", "rear_rgb"):                           # the box is below the one, behind the other
        assert held[name] == bare[name], name


def _camera_at(canon, world, name):
    cam = canon["cameras"][name]
    return cameras.mounted_pose(world[cam["parent_prim"]], cam["offset_pos"], cam["offset_rot_xyzw"])[0]


def _shifted(box, offset):
    return _obb(box.center + np.asarray(offset, float), box.half, box.rot_xyzw)


# The canon box moved 5 cm toward the robot's left (+y here): the left wrist camera, 38 mm off the box's side at the
# hold pose, is then inside it -- as when the arm sways in while the robot walks or turns.
TOWARD_LEFT = (0.0, 0.05, 0.0)


def test_a_camera_inside_the_held_box_sees_past_it_and_the_others_do_not(carried_robot, spec):
    canon = _cameras_canon()
    world, boxes, box = carried_robot
    moved = _shifted(box, TOWARD_LEFT)
    body = visibility.body_sample_points(spec, visibility.seeing_rule(canon)["sample_points"])
    person = visibility.points_world(body, visibility.standing_part_poses(spec, (0.8, 0.0), math.pi))
    lo = hi = np.zeros((0, 3))
    bare = visibility.robot_camera_fractions(canon, world, boxes, None, person, lo, hi)
    got = visibility.robot_camera_fractions(canon, world, boxes, moved, person, lo, hi)   # not refused
    assert list(got) == list(visibility.robot_cameras(canon))
    # the camera buried in the box is left with the robot's links only, as if there were no box
    assert got["left_wrist_rgb"] == bare["left_wrist_rgb"]
    # the box still hides the person from the cameras outside it
    for name in ("head_down_rgb", "right_wrist_rgb"):
        assert got[name] < bare[name], name
    assert visibility.cameras_in_box(visibility.robot_camera_box_clearances(canon, world, moved)) == ("left_wrist_rgb",)
    assert visibility.cameras_in_box(visibility.robot_camera_box_clearances(canon, world, box)) == ()


def test_the_cameras_inside_the_box_are_those_with_a_clearance_below_zero():
    # in the order given; one on a face (0.0) is not inside
    assert visibility.cameras_in_box({"a": 0.1, "b": -0.002, "c": -1.0}) == ("b", "c")
    assert visibility.cameras_in_box({"c": -1.0, "a": 0.1, "b": -0.002}) == ("c", "b")
    assert visibility.cameras_in_box({"a": 0.0, "b": 0.1}) == ()
    assert visibility.cameras_in_box({}) == ()


def test_the_command_term_counts_cameras_with_the_shared_helper():
    # the term keeps what the recorder counts from the one helper, not from a copy of its rule
    assert "self.cameras_in_box = visibility.cameras_in_box(self.camera_box_clearance)" in COMMANDS
    assert "if c < 0.0" not in COMMANDS


def test_the_cameras_clearance_from_the_held_box(carried_robot):
    canon = _cameras_canon()
    world, _, box = carried_robot
    clear = visibility.robot_camera_box_clearances(canon, world, box)
    assert list(clear) == list(visibility.robot_cameras(canon))
    # At the hold pose each wrist camera is beside the box's side face (0.40 m wide, its middle on the torso's
    # x-z plane), within its depth and height: the clearance is the camera's |y| less the half width.
    for side in ("left", "right"):
        name = f"{side}_wrist_rgb"
        y = abs(float(_camera_at(canon, world, name)[1]))
        assert clear[name] == pytest.approx(y - 0.20, abs=1e-9) and clear[name] > 0.03, name
    assert all(clear[n] > 0.0 for n in clear)
    # Moved 5 cm left the box holds the left camera: minus the depth to the nearest face, the left side face.
    moved = visibility.robot_camera_box_clearances(canon, world, _shifted(box, TOWARD_LEFT))
    left_y = abs(float(_camera_at(canon, world, "left_wrist_rgb")[1]))
    assert moved["left_wrist_rgb"] == pytest.approx(-(0.20 - (left_y - 0.05)), abs=1e-9)
    assert moved["left_wrist_rgb"] < 0.0
    assert moved["right_wrist_rgb"] == pytest.approx(clear["right_wrist_rgb"] + 0.05, abs=1e-9)


@pytest.mark.parametrize("point,expected", [
    pytest.param([0.5, 0.0, 0.0], 0.3, id="off a face"),
    pytest.param([0.5, 0.4, 0.0], math.hypot(0.3, 0.3), id="off an edge"),
    pytest.param([0.2, 0.1, 0.05], 0.0, id="at a corner"),
    pytest.param([0.18, 0.0, 0.0], -0.02, id="inside, nearest the +x face"),
    pytest.param([0.0, 0.0, 0.0], -0.05, id="the centre: nearest the z faces"),
])
def test_box_clearance_is_signed(point, expected):
    box = _obb([0.0, 0.0, 0.0], [0.2, 0.1, 0.05])
    assert visibility.box_clearance(box, point) == pytest.approx(expected, abs=1e-12)
    turned = _obb([1.0, 2.0, 0.5], [0.2, 0.1, 0.05], _about("z", 90.0))
    # the same point about a box turned a quarter about z and moved: (x, y) -> (-y, x)
    p = np.asarray(point, float)
    assert visibility.box_clearance(turned, [1.0 - p[1], 2.0 + p[0], 0.5 + p[2]]) == pytest.approx(expected, abs=1e-12)
