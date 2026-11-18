"""The jointed dummy's skin seams (generation/dummy/skin_seams.yaml): read, checked, and worked out on the skin.

Pure numpy and yaml, no pxr. The skinned jointed build (build_dummy.py --jointed --skin --seams) splits the skin
OBJ into one piece per body by the nearest-primitive rule (region_split) and hangs each piece rigidly under its
body. Two things then go wrong where the joints turn, and this module works out the remedy for each:

1. clean_labels -- the rule gives some faces to a body of another rigid group than everything round them: the
   trousers beside a hanging hand land on the hand, finger skin beside a thigh on the thigh. Such a patch leaves
   its clothes when the joint turns, so it goes to the body most of its edge neighbours are on.
2. blend_weights -- a piece's edge parts from its neighbour's when the joint between them bends. Every skin vertex
   within the joint's radius of another rigid group's faces (measured along the skin) gets a weight per rigid
   group: 1 on its own faces, falling to 0 at the radius, then normalised. A vertex two pieces share has the same
   weights in both, so place() puts both copies in the same spot and the seam stays closed.

place is what the run does before every render (generation/envs/s1d/skin_blend.py): each blended point goes to the
weighted mean of where its groups would carry it, written in its own body's frame; a point that does not blend keeps
the coordinates it was built with, to the byte. The face-corner normals stay as built.

A rigid group is the bodies the fixed joints hold together (the trunk with the head and neck, a forearm with its
hand). It is named by its root -- the body nearest the articulation root -- so a weight's column is a body whose
pose the run reads. A pose is (R, x): the body's rotation and the position of its frame's origin, which at rest is
the body's translation in the dummy's frame (the bodies carry no rotation at rest).
"""
from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import yaml

from generation.dummy import joints_spec as js
from generation.dummy.skin import region_split as rs

HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "skin_seams.yaml"
SPEC_KEYS = {"seams_version", "determined_by", "stray_share", "radius_m"}

# The blend rides on each skin mesh as three custom attributes: build_dummy writes them, skin_blend reads them.
ATTR_POINTS = "keti:seamBlend:points"     # int[]: the piece's point indices that blend
ATTR_BODIES = "keti:seamBlend:bodies"     # token[]: the rigid groups' root bodies -- the weights' columns
ATTR_WEIGHTS = "keti:seamBlend:weights"   # float[]: len(points) x len(bodies), row major; each row sums to 1
CUSTOM_DATA_KEY = "keti:seams"            # on /KetiDummy: the spec, its hash and what the build did

# A raw weight this close to 0 is the radius itself, reached through a sum of edge lengths that rounds either way.
_EDGE_OF_RADIUS = 1e-9


@dataclass(frozen=True)
class SeamSpec:
    version: str
    stray_share: float
    radius_m: Mapping[str, float]     # joint group -> radius


@dataclass(frozen=True)
class Link:
    parent: str | None                # the body this one hangs on, None for the articulation root
    joint: str | None                 # the joint group it turns on, None where a fixed joint holds it


# ---- skin_seams.yaml -------------------------------------------------------------------------------

def _number(where: str, value, *, positive: bool = False) -> float:
    # bool is an int in Python; a radius of True is a typo, not 1
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"skin_seams.yaml {where}: 수가 아닙니다 ({value!r})")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"skin_seams.yaml {where}: 유한한 수가 아닙니다 ({value!r})")
    if positive and v <= 0:
        raise ValueError(f"skin_seams.yaml {where}: 0 보다 커야 합니다 ({value!r})")
    return v


def load_seam_spec(path: Path = SPEC_PATH, joint_groups: Sequence[str] = js.GROUPS) -> SeamSpec:
    """The seam spec, checked: a radius for exactly the joint groups the jointed dummy has. ValueError names the key."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"skin_seams.yaml: 매핑이 아닙니다 ({raw!r})")
    unknown, missing = sorted(set(raw) - SPEC_KEYS), sorted(SPEC_KEYS - set(raw))
    if unknown:
        raise ValueError(f"skin_seams.yaml: 모르는 키 {unknown} — 아는 키: {sorted(SPEC_KEYS)}")
    if missing:
        raise ValueError(f"skin_seams.yaml: 키가 없습니다 {missing}")
    for key in ("seams_version", "determined_by"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"skin_seams.yaml {key}: 비어 있지 않은 글이어야 합니다 ({raw[key]!r})")
    share = _number("stray_share", raw["stray_share"])
    if not 0.0 < share < 1.0:
        raise ValueError(f"skin_seams.yaml stray_share: 0 과 1 사이여야 합니다 ({raw['stray_share']!r})")
    radii = raw["radius_m"]
    if not isinstance(radii, dict) or set(radii) != set(joint_groups):
        got = sorted(radii) if isinstance(radii, dict) else radii
        raise ValueError(f"skin_seams.yaml radius_m: 관절 묶음마다 하나씩이어야 합니다 — 관절 {sorted(joint_groups)} · 적힌 것 {got}")
    radius = {g: _number(f"radius_m.{g}", radii[g], positive=True) for g in joint_groups}
    return SeamSpec(version=raw["seams_version"], stray_share=share, radius_m=radius)


# ---- the bodies ------------------------------------------------------------------------------------

def body_links(spec: js.JointedSpec) -> dict[str, Link]:
    """{body prim: Link} for every body of the jointed dummy: the anthropometry's parent of each region (a mirrored
    region hangs on its own side's parent), turned by the spec's joint where one joins the two, and the feet on their
    ankles. Raises ValueError when a spec joint is not one of these links."""
    regions = spec.anthropometry["regions"]
    joints = {(j.parent, j.child): j.group for j in spec.joints}
    links: dict[str, Link] = {}
    for code, reg in regions.items():
        for name, _ in rs.prim_names(code, reg):
            side = name.rsplit("_", 1)[1] if reg.get("mirror_y") else "left"
            parent = reg.get("parent")
            parent = js.prim_of(parent, side, regions) if parent else None
            links[name] = Link(parent, joints.get((parent, name)))
    for foot in spec.feet:
        j = next((j for j in spec.joints if j.child == foot.name), None)
        if j is None:
            raise ValueError(f"{foot.name}: 발을 잇는 관절이 명세에 없습니다")
        links[foot.name] = Link(j.parent, j.group)
    turning = {(b, l.parent, l.joint) for b, l in links.items() if l.joint}
    if turning != {(j.child, j.parent, j.group) for j in spec.joints}:
        raise ValueError("관절 명세의 관절이 부위의 부모 사슬과 맞지 않습니다")
    return links


def rigid_roots(links: Mapping[str, Link]) -> dict[str, str]:
    """{body: the root of its rigid group} -- up the fixed links until a joint or the articulation root."""
    def root(body):
        seen = set()
        while links[body].joint is None and links[body].parent is not None:
            if body in seen:
                raise ValueError(f"부모 사슬이 제자리를 돕니다: {body}")
            seen.add(body)
            body = links[body].parent
        return body

    return {b: root(b) for b in links}


def _up(body: str, links: Mapping[str, Link]) -> list[tuple[str, str | None]]:
    out = []
    while body is not None:
        out.append((body, links[body].joint))
        body = links[body].parent
    return out


def pair_radius(a: str, b: str, links: Mapping[str, Link], radius: Mapping[str, float]) -> float:
    """The blend radius between two bodies: the largest of the joints on the path between them, 0 if none turns."""
    up_a, up_b = _up(a, links), _up(b, links)
    common = {x for x, _ in up_a} & {x for x, _ in up_b}
    joints = [j for chain in (up_a, up_b) for x, j in chain if x not in common and j]
    return max((radius[j] for j in joints), default=0.0)


# ---- stray patches ---------------------------------------------------------------------------------

def _edge_neighbours(F: np.ndarray) -> list[list[int]]:
    by_edge = defaultdict(list)
    for i, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            by_edge[(min(u, v), max(u, v))].append(i)
    nb: list[list[int]] = [[] for _ in range(len(F))]
    for faces in by_edge.values():
        for i in faces:
            nb[i].extend(j for j in faces if j != i)
    return nb


def _components(n: int, nb, same) -> np.ndarray:
    comp = -np.ones(n, np.int64)
    k = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack, comp[s] = [s], k
        while stack:
            i = stack.pop()
            for j in nb[i]:
                if comp[j] < 0 and same(i, j):
                    comp[j] = k
                    stack.append(j)
        k += 1
    return comp


def clean_labels(F: np.ndarray, labels: np.ndarray, roots: Mapping[str, str], share: float,
                 max_rounds: int = 10) -> tuple[np.ndarray, list[dict]]:
    """The face labels with every stray patch moved, and what moved: [{faces, from, to, face_ids}].

    A patch is a body's faces joined by edges. Within one piece of mesh (faces joined by edges whatever their
    labels) a body keeps its largest patch if that patch holds at least ``share`` of the body's faces in the whole
    mesh; any other patch of it is stray when no edge neighbour is on the same rigid group, and goes to the body
    most of its edge neighbours are on (the alphabetically first of a tie). A patch with no edge neighbour -- a
    piece of mesh of its own -- stays. Rounds repeat until nothing moves."""
    F = np.asarray(F, np.int64)
    fl = np.asarray(labels).astype(str).astype(object)
    nb = _edge_neighbours(F)
    piece = _components(len(F), nb, lambda i, j: True)
    moves: list[dict] = []
    for _ in range(max_rounds):
        patch = _components(len(F), nb, lambda i, j: fl[i] == fl[j])
        members = defaultdict(list)
        for i, p in enumerate(patch):
            members[int(p)].append(i)
        total = Counter(fl.tolist())
        largest: dict[tuple[int, str], int] = {}
        for p, faces in members.items():
            key = (int(piece[faces[0]]), fl[faces[0]])
            if key not in largest or len(faces) > len(members[largest[key]]):
                largest[key] = p
        keep = {p for p in largest.values() if len(members[p]) >= share * total[fl[members[p][0]]]}
        moved = 0
        for p in sorted(members):
            if p in keep:
                continue
            faces = members[p]
            label = fl[faces[0]]
            around = Counter(fl[j] for i in faces for j in nb[i] if patch[j] != p)
            if not around or roots[label] in {roots[x] for x in around}:
                continue
            to = min(around, key=lambda x: (-around[x], x))
            fl[faces] = to
            moves.append({"faces": len(faces), "from": label, "to": to, "face_ids": sorted(faces)})
            moved += 1
        if not moved:
            return fl.astype(str), moves
    raise ValueError(f"떨어진 조각을 {max_rounds} 번 옮겨도 멈추지 않습니다")


# ---- blend weights ---------------------------------------------------------------------------------

def _graph(V: np.ndarray, F: np.ndarray) -> dict[int, dict[int, float]]:
    adj: dict[int, dict[int, float]] = defaultdict(dict)
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            d = float(np.linalg.norm(V[u] - V[v]))
            adj[int(u)][int(v)] = d
            adj[int(v)][int(u)] = d
    return adj


def _distance_from(adj, sources, limit: float) -> dict[int, float]:
    dist = {int(s): 0.0 for s in sources}
    heap = [(0.0, int(s)) for s in sorted(sources)]
    heapq.heapify(heap)
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        for v, w in adj[u].items():
            nd = d + w
            if nd < limit and nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def blend_weights(V: np.ndarray, F: np.ndarray, labels: np.ndarray, roots: Mapping[str, str],
                  radius: Callable[[str, str], float]) -> dict[int, dict[str, float]]:
    """{vertex: {rigid group root: weight}} for the vertices that blend -- those with more than one rigid group in
    reach. A vertex's own groups (of the faces it is on) weigh 1; another group g weighs 1 - d/R, with d the distance
    along the edges to g's faces and R the largest radius(own, g) over its own groups, and nothing at R or beyond.
    The weights are normalised to sum to 1. ``radius`` takes two group roots and returns 0 for none."""
    F = np.asarray(F, np.int64)
    fl = np.asarray(labels).astype(str)
    own: dict[int, set[str]] = defaultdict(set)
    region: dict[str, set[int]] = defaultdict(set)
    for f, lab in zip(F, fl):
        g = roots[lab]
        for v in f:
            own[int(v)].add(g)
            region[g].add(int(v))
    groups = sorted(region)
    reach = {(a, b): radius(a, b) for a in groups for b in groups if a != b}
    adj = _graph(np.asarray(V, float), F)
    raw: dict[int, dict[str, float]] = defaultdict(dict)
    for g in groups:
        limit = max((reach[(o, g)] for o in groups if o != g), default=0.0)
        if limit <= 0:
            continue
        for v, d in _distance_from(adj, region[g], limit).items():
            if g in own[v]:
                continue
            R = max((reach[(o, g)] for o in own[v]), default=0.0)
            w = 1.0 - d / R if R > 0 else 0.0
            if w > _EDGE_OF_RADIUS:
                raw[v][g] = w
    out: dict[int, dict[str, float]] = {}
    for v in sorted(set(raw) | {v for v, g in own.items() if len(g) > 1}):
        ws = {g: 1.0 for g in own[v]}
        ws.update(raw[v])
        total = sum(ws[g] for g in sorted(ws))
        out[v] = {g: ws[g] / total for g in sorted(ws)}
    return out


def piece_blend(old: np.ndarray, weights: Mapping[int, Mapping[str, float]]
                ) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """For one piece (``old``: its points' vertex ids in the skin mesh, in point order): the indices of its points
    that blend, the group roots in name order, and their weights (float32, one row per blended point)."""
    rows = np.array([i for i, v in enumerate(old) if int(v) in weights], np.int64)
    cols = tuple(sorted({g for i in rows for g in weights[int(old[i])]}))
    M = np.zeros((len(rows), len(cols)), np.float32)
    at = {g: k for k, g in enumerate(cols)}
    for r, i in enumerate(rows):
        for g, w in weights[int(old[i])].items():
            M[r, at[g]] = w
    return rows, cols, M


# ---- placing the points ----------------------------------------------------------------------------

Pose = tuple[np.ndarray, np.ndarray]      # (R 3x3, x 3)


def place(local: np.ndarray, own_rest: np.ndarray, rows: np.ndarray, weights: np.ndarray,
          col_poses: Sequence[Pose], col_rests: Sequence[np.ndarray], own_pose: Pose) -> np.ndarray:
    """A piece's points in its body's frame for the given poses (float32). ``local`` are the points as built (the
    body's frame, rest pose), ``own_rest`` the body's rest translation; a blended row goes to the weighted mean of
    where each column's body carries its rest point, every other point is ``local`` unchanged."""
    out = np.array(local, np.float32, copy=True)
    if len(rows) == 0:
        return out
    P = np.asarray(local, np.float64)[rows] + np.asarray(own_rest, np.float64)
    carried = np.stack([(P - np.asarray(t, np.float64)) @ np.asarray(R, np.float64).T + np.asarray(x, np.float64)
                        for (R, x), t in zip(col_poses, col_rests)])            # (columns, rows, 3)
    Q = np.einsum("rc,cri->ri", np.asarray(weights, np.float64), carried)
    R0, x0 = (np.asarray(a, np.float64) for a in own_pose)
    out[rows] = ((Q - x0) @ R0).astype(np.float32)
    return out
