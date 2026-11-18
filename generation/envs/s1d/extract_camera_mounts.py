#!/usr/bin/env python3
"""Where S1d's link-mounted cameras sit on H1, read from the H1 assets (~/isaacsim/python.sh).

    ~/isaacsim/python.sh generation/envs/s1d/extract_camera_mounts.py           # check the canon
    ~/isaacsim/python.sh generation/envs/s1d/extract_camera_mounts.py --write   # write the offsets

meta/cameras.json keeps each mounted camera's mount rule; its offset_pos and offset_rot_xyzw are this
script's output, never typed by hand. Only USD stages are opened -- no
environment, no physics. Two copies of H1: the BSD-3 one the simulation spawns ("simulated": what
renders) and the MPL-2.0 one ("official": the official URDF's meshes and sensor frames).

- mount.source (head_down_rgb): the real D435 RGB module -- the pose of that link in the camera's parent
  link, read from the official copy (the simulated copy has no sensor frames), cross-checked against the
  official URDF's joint on record for that link (URDF_SOURCES; a link with none is said to be unchecked).
  The links both copies share must sit at the same poses relative to the parent link, or an offset read
  on one copy would not carry to the other.
- mount.above (head_rgb): a module added on the forehead, not on the real H1. On the lower camera's
  left-right line, up_m above it, level (tilt_down_deg). The module is a box (MODULE_BOX_M) centred on
  the optical centre; it must stand in front of BOTH heads -- the simulated one (renders: the camera
  must not sit in the shell) and the official one (an add-on module the real robot could carry)
  -- so the optical centre is clear_front_m in front of the frontmost surface of either mesh inside
  the box's width and height. Face-aware (surface_front): a face counts where it crosses that footprint,
  though every vertex of it lies outside. clear_front_m has to be half the box's depth (checked): the
  box's back face then touches the head. The simulated head recedes sooner above the D435 than the
  official STL, which is why one mesh alone does not do. The box must also stay out of the lower
  camera's view.
- mount.behind (rear_rgb): a second module, on the back of the head and also not on the real H1, looking
  straight back -- level, or turned down by tilt_down_deg. At the height and on the left-right line of the
  camera it names (the forehead module's), and the same box held to both heads the same way, mirrored: the
  optical centre is clear_back_m behind the hindmost surface of either mesh inside the box's width and height
  there (surface_front in a frame that looks back), and clear_back_m has to be half the box's depth (checked):
  the box's front face then touches the back of the head.
- mount.inset_from_tip_m (the wrist cameras): on the forearm, inset_from_tip_m back from its tip (the
  simulated mesh's max x -- 4.6 cm longer than the URDF mesh), lift_above_top_m above the forearm's top
  surface there, turned down toward the fist by tilt_down_deg. The top is the highest point of the surface
  inside the search band (WRIST_BAND_X_M, WRIST_BAND_Y_M) -- face-aware, like the forehead's front: a face
  crossing the band counts where it crosses it, though no vertex of it lies inside (surface_front). Read
  from the vertices in the band, a box whose top face spans the band without a vertex there gave its bottom
  face as the top, and a camera inside the box. The mirror links must agree within
  1 mm; the canon gives both the value of the first in canon order, the left link's (the mirror test
  compares them with ==).

Every mounted camera: no face of its own link's simulated mesh -- the one body that never moves in its
picture -- may be cut by its near clipping plane inside the picture: the render would show that body cut
open (a wrist camera 1.7 cm from the fist did). A piece of the link wholly inside the near plane is not a
cut: the real D435 looks through a window in its guard, and the simulated guard in front of the lens,
which has none, is clipped away whole. And its optical centre must not lie inside its link's simulated mesh
(winding_number; an add-on head module inside neither head, the official one too) -- a camera wholly
inside a body cuts no face at its near plane and renders black, which the near-plane check alone passes.
The real D435's centre does lie inside the official head mesh, behind its
window, so the source camera is held to the simulated mesh alone. What the winding number does not see is a
hollow: the simulated H1 forearm and head are shells, and a point inside the shell reads 0. There the
near-plane check is what refuses the camera -- the walls are near: 36-41 faces cut for points in the forearm's
hollow, 35-96 in the head's (2026-09-30) -- and a hollow wider than the near plane in view would pass both.

The canon is read with generation/cameras.load, so what it refuses (a camera its type cannot place) stops
the script before anything is extracted.

Check mode (the default) exits 1 when a canon value is more than CHECK_TOL from the extraction, or a
check fails -- the geometry checks at the canon's own offsets. --write writes offset_pos and
offset_rot_xyzw only (key order kept, the file as json.dumps(indent=2, ensure_ascii=False) writes it),
after every check passes at the offsets it is about to write, and prints what changed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from generation import cameras  # noqa: E402

SIMULATED, OFFICIAL = "simulated", "official"

CHECK_TOL = 1e-4             # canon vs extraction: metres, and quaternion components
POS_DECIMALS = 6             # positions written to the micrometre; rotations as computed
BOX_TOL_M = 10.0 ** -POS_DECIMALS   # module box vs a head: less than the written precision is rounding
MIRROR_TOL_M = 1e-3          # the wrist links' mounts may differ by this much, each component
COPIES_TOL_M = 1e-5          # the links both copies share: same pose relative to the parent link
COPIES_TOL_DEG = 1e-3
# Search windows of the wrist extraction, not canon.
WRIST_BAND_X_M = 0.01        # the top surface around the mount's x, +/- this
WRIST_BAND_Y_M = 0.02        # ... and around the forearm's centre line, +/- this
# An optical centre whose winding number about a link's mesh is above this (in magnitude) is inside it: 1 inside a
# closed mesh, 0 outside; a mesh with holes reads in between, away from 0.5 unless the point is near a hole.
INSIDE_WINDING = 0.5
IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)

# The official H1 URDF (unitreerobotics/unitree_ros, robots/h1_description/urdf/h1.urdf): the fixed joint, xyz and
# rpy in the parent link (torso_link) of each sensor link a camera's mount.source may name -- the cross-check for
# the pose read from the official copy. Keyed by that link: a camera on a link not listed here is not held to
# another link's joint.
URDF_SOURCES = {"d435_rgb_module_link": ("d435_rgb_module_joint", (0.10848, 0.0325, 0.69317),
                                          (-2.45735, 0.0, -1.5708))}   # its optical axis 50.8 deg down
URDF_POS_TOL_M = 1e-3
URDF_ROT_TOL_DEG = 0.1
# The module added above the D435: 9 cm wide, 2.5 cm high, 3 cm deep, in its
# own camera's axes (image right, image down, optical axis) and centred on its optical centre. The module added
# on the back of the head is the same box.
MODULE_BOX_M = (0.09, 0.025, 0.03)
# The rules that place such an add-on module: held to both heads, not to the simulated one alone.
ADD_ON_RULES = ("above", "behind")


# ---- geometry (numpy only) ----------------------------------------------------------------------

def rotation_of(q) -> np.ndarray:
    """3x3 rotation (columns: the frame's axes in the parent) of an xyzw quaternion."""
    return np.column_stack([cameras._q_rot(q, e) for e in np.eye(3)])


def rpy_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis rpy: Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def down_angle_deg(forward) -> float:
    """How far below its parent's horizontal an optical axis points."""
    f = np.asarray(forward, float)
    return math.degrees(math.atan2(-f[2], math.hypot(f[0], f[1])))


def frame_xyzw(rot: np.ndarray) -> np.ndarray:
    """xyzw of a ROS optical frame given as a rotation: its forward (z) and image-up (-y) rebuild it."""
    q = cameras.optical_frame_xyzw(rot[:, 2], -rot[:, 1])
    if not np.allclose(rotation_of(q), rot, rtol=0.0, atol=1e-6):
        raise ValueError(f"not a proper rotation (a reflection or a skew?):\n{rot}")
    return q


def in_camera(pts, pos, q) -> np.ndarray:
    """Points of the parent's frame in a camera's frame (x image right, y image down, z the optical axis)."""
    return (np.asarray(pts, float) - np.asarray(pos, float)) @ rotation_of(q)


def _triangles(counts, idx) -> tuple[np.ndarray, np.ndarray]:
    """Fan triangulation of polygon faces: vertex indices (T, 3) and each triangle's face number (T,)."""
    counts, idx = np.asarray(counts, int), np.asarray(idx, int)
    starts, faces = np.cumsum(counts) - counts, np.arange(len(counts))
    tris, owner = [np.zeros((0, 3), int)], [np.zeros(0, int)]
    for k in range(1, int(counts.max(initial=0)) - 1):
        m = counts >= k + 2
        tris.append(np.stack([idx[starts[m]], idx[starts[m] + k], idx[starts[m] + k + 1]], axis=1))
        owner.append(faces[m])
    return np.concatenate(tris), np.concatenate(owner)


def _clip_to_rect(poly: np.ndarray, half) -> np.ndarray:
    """Sutherland-Hodgman: the part of a convex polygon (rows: points in a camera frame) inside the prism
    |x| <= half[0], |y| <= half[1] along the frame's z. New vertices are interpolated on the polygon's own edges,
    so they stay on its plane."""
    for axis, sign in ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)):
        if not len(poly):
            break
        s = sign * poly[:, axis] - half[axis]            # <= 0: inside this side
        keep = []
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            if s[i] <= 0.0:
                keep.append(poly[i])
            if (s[i] <= 0.0) != (s[j] <= 0.0):
                keep.append(poly[i] + (s[i] / (s[i] - s[j])) * (poly[j] - poly[i]))
        poly = np.array(keep).reshape(-1, 3)
    return poly


def surface_front(meshes, pos, q, half) -> float:
    """How far forward along the optical axis of a frame at pos/q (xyzw) the surfaces of ``meshes`` ((points,
    face counts, face indices) in the parent's frame) reach inside the prism |x| <= half[0], |y| <= half[1] around
    that axis (image right, image down). Face-aware: each face, fan-triangulated, is clipped to the prism, and depth
    is linear over a triangle, so its frontmost point in the prism is a vertex of the clipped polygon -- a face
    whose vertices all lie outside the prism still counts where it crosses it. -inf when no face reaches it."""
    best = -math.inf
    for pts, counts, idx in meshes:
        tris, _ = _triangles(counts, idx)
        t = in_camera(pts, pos, q)[tris]                 # (T, vertex, xyz)
        lo, hi = t.min(axis=1), t.max(axis=1)
        near = (lo[:, 0] <= half[0]) & (hi[:, 0] >= -half[0]) & (lo[:, 1] <= half[1]) & (hi[:, 1] >= -half[1])
        order = np.argsort(-hi[near, 2])                 # furthest-reaching first; stop when none can do better
        for tri, reach in zip(t[near][order], hi[near, 2][order]):
            if reach <= best:
                break
            clipped = _clip_to_rect(tri, half)
            if len(clipped):
                best = max(best, float(clipped[:, 2].max()))
    return best


def forehead_mount(heads: dict[str, list], below_pos, rule: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Above the lower camera by up_m on its left-right line; clear_front_m in front of the frontmost surface any
    head mesh has inside the module box's width and height there (surface_front: face-aware). Placed level --
    x is the depth along a level optical axis; a tilted module is still held to the heads, in its own frame, by
    module_gap."""
    w, h, _ = MODULE_BOX_M
    y, z = float(below_pos[1]), round(float(below_pos[2]) + float(rule["up_m"]), POS_DECIMALS)
    q = cameras.tilted_along_x_xyzw(rule["tilt_down_deg"])
    fronts = {mesh: surface_front(faces, (0.0, y, z), q, (w / 2, h / 2)) for mesh, faces in heads.items()}
    fronts = {mesh: x for mesh, x in fronts.items() if x > -math.inf}
    if not fronts:
        raise ValueError(f"no head mesh reaches into the module's {w} x {h} m around y {y}, z {z}")
    governing = max(fronts, key=fronts.get)
    pos = np.array([round(fronts[governing] + float(rule["clear_front_m"]), POS_DECIMALS), y, z])
    return pos, q, {"fronts": {mesh: round(x, 6) for mesh, x in fronts.items()}, "governing": governing}


def behind_xyzw(tilt_down_deg: float) -> np.ndarray:
    """Optical frame along the parent's -x (straight back), turned toward -z by the tilt, image-up toward +z:
    cameras.tilted_along_x_xyzw turned half a turn about the parent's z. Tilt 0 is (-0.5, -0.5, 0.5, 0.5) up to
    sign -- image right the parent's +y."""
    t = math.radians(float(tilt_down_deg))
    return cameras.optical_frame_xyzw((-math.cos(t), 0.0, -math.sin(t)), (-math.sin(t), 0.0, math.cos(t)))


def rear_mount(heads: dict[str, list], ref_pos, rule: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """The forehead module's mirror on the back of the head: at the height and on the left-right line of the camera
    rule["behind"] names (``ref_pos``, that camera's offset_pos); clear_back_m behind the hindmost surface any head
    mesh has inside the module box's width and height there. surface_front measures it in a frame at x = 0 that
    looks back, so how far that frame's axis reaches is how far behind x = 0 the surface lies (face-aware, as on the
    forehead). Placed level -- x is the depth along a level optical axis; a tilted module is still held to the
    heads, in its own frame, by module_gap."""
    w, h, _ = MODULE_BOX_M
    y, z = float(ref_pos[1]), float(ref_pos[2])
    q = behind_xyzw(rule["tilt_down_deg"])
    reach = {mesh: surface_front(faces, (0.0, y, z), q, (w / 2, h / 2)) for mesh, faces in heads.items()}
    backs = {mesh: -d for mesh, d in reach.items() if d > -math.inf}
    if not backs:
        raise ValueError(f"no head mesh reaches into the module's {w} x {h} m around y {y}, z {z}")
    governing = min(backs, key=backs.get)
    pos = np.array([round(backs[governing] - float(rule["clear_back_m"]), POS_DECIMALS), y, z])
    return pos, q, {"backs": {mesh: round(x, 6) for mesh, x in backs.items()}, "governing": governing}


def wrist_mount(meshes, rule: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """On the forearm's upper side (+z), inset_from_tip_m back from its tip (+x), lift_above_top_m above it. ``meshes``:
    (points, face counts, face indices) in the link's frame. The tip is the furthest vertex (a face's furthest point
    is one of its vertices); the top is the highest point of any face inside the search band around (x, 0) --
    surface_front in a frame whose axis is the link's +z -- so a face crossing the band counts though none of its
    vertices lies inside."""
    tip = max(float(pts[:, 0].max()) for pts, _, _ in meshes)
    x = tip - float(rule["inset_from_tip_m"])
    top = surface_front(meshes, (x, 0.0, 0.0), IDENTITY_XYZW, (WRIST_BAND_X_M, WRIST_BAND_Y_M))
    if top == -math.inf:
        raise ValueError(f"no mesh face within {WRIST_BAND_X_M} m of x {x:.4f} and {WRIST_BAND_Y_M} m of y 0")
    pos = np.array([round(x, POS_DECIMALS), 0.0, round(top + float(rule["lift_above_top_m"]), POS_DECIMALS)])
    return pos, cameras.tilted_along_x_xyzw(rule["tilt_down_deg"]), {"tip_x": round(tip, 6), "top_z": round(top, 6)}


def winding_number(meshes, point) -> float:
    """The generalized winding number about ``point`` of each of ``meshes`` (points, face counts, face indices;
    fan-triangulated) -- the solid angle its faces subtend there over 4 pi, each triangle's by van Oosterom and
    Strackee's formula -- and of those the one largest in magnitude. A closed mesh gives 1 inside and 0 outside (-1
    inside if its faces point inward; hence the magnitude is compared); a mesh with holes -- the official H1 meshes
    have a few open edges -- reads in between, near 1 deep inside and near 0 well outside, and falls off only close
    to a hole (Jacobson et al. 2013). Each mesh on its own, not summed: a link is several meshes (the simulated
    torso is two) and nothing makes them face the same way -- one facing out and one facing in around the point
    summed to 0, a buried camera read as outside."""
    best = 0.0
    p = np.asarray(point, float)
    for pts, counts, idx in meshes:
        tris, _ = _triangles(counts, idx)
        t = np.asarray(pts, float)[tris] - p
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
        num = np.einsum("ij,ij->i", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", b, c) * la
               + np.einsum("ij,ij->i", c, a) * lb)
        w = float(np.sum(2.0 * np.arctan2(num, den))) / (4.0 * math.pi)
        if abs(w) > abs(best):
            best = w
    return best


def module_corners(pos, q) -> np.ndarray:
    half = np.asarray(MODULE_BOX_M, float) / 2.0
    signs = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    return np.asarray(pos, float) + (signs * half) @ rotation_of(q).T


def module_gap(meshes, pos, q) -> float:
    """How far a head sits behind the module box's back face (m), in the module's own frame: the frontmost of its
    surface inside the box's width and height (surface_front: face-aware). Negative: the head reaches into the
    box, or past it. inf when no face of it is behind or in the box's footprint at all."""
    w, h, d = MODULE_BOX_M
    front = surface_front(meshes, pos, q, (w / 2, h / 2))
    return math.inf if front == -math.inf else -d / 2 - front


def box_frustum_separation(corners, cam_pos, cam_q, cam: dict) -> tuple[str | None, float]:
    """Is a convex box outside a camera's view frustum? The frustum is the intersection of four half-spaces
    through the optical centre (the image's left, right, top and bottom edges) and the one in front of it;
    all eight corners strictly outside one of them keeps the whole box out. Returns that plane's name and
    the smallest angle (deg) by which a corner clears it, or (None, 0.0) when no plane separates."""
    fx, fy, cx, cy = cameras.intrinsics(cam)
    w, h = float(cam["width"]), float(cam["height"])
    c = in_camera(corners, cam_pos, cam_q)
    normals = {"left": (fx, 0.0, cx), "right": (-fx, 0.0, w - cx), "top": (0.0, fy, cy),
               "bottom": (0.0, -fy, h - cy), "behind": (0.0, 0.0, 1.0)}     # inside: n . p >= 0
    best: tuple[str | None, float] = (None, 0.0)
    for name, n in normals.items():
        n = np.asarray(n) / np.linalg.norm(n)
        s = c @ n / np.linalg.norm(c, axis=1)
        if np.all(s < 0.0):
            clear = float(np.degrees(np.arcsin(-s.max())))
            if clear > best[1]:
                best = (name, clear)
    return best


def _segment_meets_rect(a, b, w: float, h: float) -> bool:
    """Liang-Barsky: does the segment a-b (picture coordinates) meet [0, w] x [0, h]?"""
    t0, t1 = 0.0, 1.0
    du, dv = b[0] - a[0], b[1] - a[1]
    for p, r in ((-du, a[0]), (du, w - a[0]), (-dv, a[1]), (dv, h - a[1])):
        if p == 0.0:
            if r < 0.0:
                return False
            continue
        t = r / p
        if p < 0.0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def near_plane(meshes, pos, q, cam: dict) -> dict:
    """What a camera's near clipping plane does to its own link. ``meshes``: (points, face counts, face indices)
    in the link's frame. ``cut``: faces the near plane cuts inside the picture -- the render shows the body cut
    open there. A (fan-triangulated) face straddling the plane meets it in a segment between the two edges that
    cross it; the face is cut when that segment crosses the picture, whether or not its ends lie in it, and each
    face counts once however many of its triangles or edges do. ``clipped``: vertices in view nearer than the
    plane (a piece wholly inside it is removed whole: no cut). ``nearest``: the nearest vertex, any direction."""
    fx, fy, cx, cy = cameras.intrinsics(cam)
    w, h, near = float(cam["width"]), float(cam["height"]), float(cam["clipping_range_m"][0])
    cut = clipped = 0
    nearest = math.inf
    for pts, counts, idx in meshes:
        c = in_camera(pts, pos, q)
        nearest = min(nearest, float(np.linalg.norm(c, axis=1).min()))
        z = c[:, 2]
        ahead = z > 0.0
        u, v = fx * c[ahead, 0] / z[ahead] + cx, fy * c[ahead, 1] / z[ahead] + cy
        clipped += int(((u >= 0) & (u < w) & (v >= 0) & (v < h) & (z[ahead] < near)).sum())
        tris, owner = _triangles(counts, idx)
        t = c[tris]
        d = t[:, :, 2] - near
        straddling = (d.min(axis=1) < 0.0) & (d.max(axis=1) > 0.0)
        cut_faces: set[int] = set()
        for tri, dd, face in zip(t[straddling], d[straddling], owner[straddling]):
            if int(face) in cut_faces:
                continue
            ends = [tri[i] + (dd[i] / (dd[i] - dd[j])) * (tri[j] - tri[i])
                    for i, j in ((0, 1), (1, 2), (2, 0)) if (dd[i] < 0.0) != (dd[j] < 0.0)]
            a, b = ((fx * e[0] / near + cx, fy * e[1] / near + cy) for e in ends)
            if _segment_meets_rect(a, b, w, h):
                cut_faces.add(int(face))
        cut += len(cut_faces)
    return {"cut": cut, "clipped": clipped, "nearest": nearest, "near": near}


def urdf_cross_check(name: str, source: str, pos, rot) -> tuple[list[str], list[str]]:
    """(lines, problems): the pose read for ``source`` against the official URDF's joint on record for that
    link (URDF_SOURCES). A link with no record is said to be unchecked, not held to another link's joint."""
    record = URDF_SOURCES.get(source)
    if record is None:
        return [f"{name}: no official URDF joint on record for {source} -- not cross-checked"], []
    joint, xyz, rpy = record
    urdf = rpy_rotation(*rpy)
    dp = float(np.max(np.abs(np.asarray(pos, float) - np.asarray(xyz))))
    da = angle_between_deg(rot, urdf)
    say = [f"{name}: official URDF {joint} xyz {list(xyz)} rpy {list(rpy)} -- optical axis "
           f"{down_angle_deg(urdf[:, 2]):.4f} deg down; USD vs URDF {dp * 1e3:.4f} mm, {da:.6f} deg"]
    problems = []
    if dp > URDF_POS_TOL_M or da > URDF_ROT_TOL_DEG:
        problems.append(f"{name}: the {OFFICIAL} copy's {source} is {dp * 1e3:.3f} mm / {da:.4f} deg "
                        f"from the official URDF (tolerance {URDF_POS_TOL_M * 1e3} mm / {URDF_ROT_TOL_DEG} deg)")
    return say, problems


# ---- the canon ----------------------------------------------------------------------------------

def _rule(cam: dict) -> str:
    m = cam.get("mount") or {}
    for key in ("source", "above", "behind", "inset_from_tip_m"):
        if key in m:
            return key
    raise ValueError(f"a mounted camera's rule is none this script knows: {sorted(m)}")


def _offsets(cam: dict) -> tuple[np.ndarray, np.ndarray] | None:
    p, q = cam.get("offset_pos"), cam.get("offset_rot_xyzw")
    if not (isinstance(p, list) and len(p) == 3 and isinstance(q, list) and len(q) == 4):
        return None
    try:
        return np.asarray(p, float), np.asarray(q, float)
    except (TypeError, ValueError):
        return None


def compare(canon: dict, extracted: dict) -> list[str]:
    """Canon value vs extraction per mounted camera; a quaternion and its negation are one rotation."""
    problems = []
    for name, (pos, q) in extracted.items():
        cam = canon["cameras"][name]
        have = _offsets(cam)
        if have is None:
            problems.append(f"{name}: canon offset_pos {cam.get('offset_pos')!r} / offset_rot_xyzw "
                            f"{cam.get('offset_rot_xyzw')!r} are not 3 and 4 numbers")
            continue
        dp = float(np.max(np.abs(have[0] - pos)))
        dq = min(float(np.max(np.abs(have[1] - s * q))) for s in (1.0, -1.0))
        if not (dp <= CHECK_TOL and dq <= CHECK_TOL):
            problems.append(f"{name}: canon offset_pos {cam['offset_pos']} / offset_rot_xyzw {cam['offset_rot_xyzw']} "
                            f"differ from the extraction {pos.tolist()} / {q.tolist()} (by {dp:.2e} m / {dq:.2e})")
    return problems


def write(path: Path, canon: dict, extracted: dict) -> list[str]:
    changed = []
    for name, (pos, q) in extracted.items():
        cam = canon["cameras"][name]
        for key, value in (("offset_pos", pos.tolist()), ("offset_rot_xyzw", q.tolist())):
            if cam.get(key) != value:
                changed.append(f"{name}.{key}: {cam.get(key)} -> {value}")
                cam[key] = value
    path.write_text(json.dumps(canon, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


# ---- the assets (pxr, inside Kit) ---------------------------------------------------------------

def _link(stage, name: str):
    found = [p for p in stage.Traverse() if p.GetName() == name]
    if len(found) != 1:
        raise ValueError(f"{len(found)} prims named {name} in {stage.GetRootLayer().identifier}")
    return found[0]


def _world(prim) -> np.ndarray:
    from pxr import Usd, UsdGeom

    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), float)


def pose_in(stage, child: str, parent: str) -> tuple[np.ndarray, np.ndarray]:
    """(position, rotation) of a link in another link's frame. Gf matrices act on row vectors."""
    rel = _world(_link(stage, child)) @ np.linalg.inv(_world(_link(stage, parent)))
    rot = rel[:3, :3].T
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6) or np.linalg.det(rot) < 0:
        raise ValueError(f"{child} in {parent}: not a rotation (scaled or mirrored):\n{rot}")
    return rel[3, :3].copy(), rot


def rendered_meshes(stage, link: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The meshes a camera would see under the link -- render purpose, not invisible (the collision meshes are
    guides) -- as (points in the link's frame, face vertex counts, face vertex indices). They are instance
    proxies."""
    from pxr import Usd, UsdGeom

    prim = _link(stage, link)
    to_link = np.linalg.inv(_world(prim))
    out = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(p)
        if img.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        if img.ComputeVisibility(Usd.TimeCode.Default()) == UsdGeom.Tokens.invisible:
            continue
        mesh = UsdGeom.Mesh(p)
        pts = np.asarray(mesh.GetPointsAttr().Get(), float)
        m = _world(p) @ to_link
        out.append((pts @ m[:3, :3] + m[3, :3], np.asarray(mesh.GetFaceVertexCountsAttr().Get(), int),
                    np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int)))
    if not out:
        raise ValueError(f"no rendered mesh under {link} in {stage.GetRootLayer().identifier}")
    return out


class Meshes:
    """Each copy's rendered meshes per link, read once."""

    def __init__(self, stages: dict):
        self._stages, self._cache = stages, {}

    def faces(self, copy: str, link: str):
        if (copy, link) not in self._cache:
            self._cache[(copy, link)] = rendered_meshes(self._stages[copy], link)
        return self._cache[(copy, link)]


def shared_link_deviation(a, b, parent: str) -> tuple[int, float, float]:
    """How far apart the links both stages have sit, relative to ``parent``: (links, metres, degrees)."""
    from pxr import UsdPhysics

    def links(stage):
        return {p.GetName() for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}

    shared = sorted(links(a) & links(b) - {parent})
    dm = dd = 0.0
    for name in shared:
        (pa, ra), (pb, rb) = pose_in(a, name, parent), pose_in(b, name, parent)
        dm, dd = max(dm, float(np.max(np.abs(pa - pb)))), max(dd, angle_between_deg(ra, rb))
    return len(shared), dm, dd


def extract(canon: dict, stages: dict, meshes: Meshes) -> tuple[dict, list[str], list[str]]:
    """{camera: (offset_pos, offset_rot_xyzw)} for every mounted camera, the lines to print, the problems."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    say: list[str] = []
    problems: list[str] = []
    mounted = {n: c for n, c in canon["cameras"].items() if c.get("type") == "mounted"}
    by_rule: dict[str, list[str]] = {}
    for name, cam in mounted.items():
        by_rule.setdefault(_rule(cam), []).append(name)

    for name in by_rule.get("source", []):
        cam = mounted[name]
        pos, rot = pose_in(stages[OFFICIAL], cam["mount"]["source"], cam["parent_prim"])
        q = frame_xyzw(rot)
        out[name] = (pos.round(POS_DECIMALS), q)
        say += [f"{name}: {cam['mount']['source']} in {cam['parent_prim']} ({OFFICIAL} copy) pos {pos.tolist()}",
                f"{name}: rotation xyzw {q.tolist()} -- optical axis {down_angle_deg(rot[:, 2]):.4f} deg down"]
        u_say, u_problems = urdf_cross_check(name, cam["mount"]["source"], pos, rot)
        say += u_say
        problems += u_problems
        n, dm, dd = shared_link_deviation(stages[OFFICIAL], stages[SIMULATED], cam["parent_prim"])
        say.append(f"{name}: {n} links both copies have, relative to {cam['parent_prim']}: at most {dm:.2e} m, {dd:.2e} deg apart")
        if n == 0 or dm > COPIES_TOL_M or dd > COPIES_TOL_DEG:
            problems.append(f"{name}: the two H1 copies' links differ relative to {cam['parent_prim']} by {dm:.2e} m / "
                            f"{dd:.2e} deg over {n} links -- an offset read on one does not carry to the other")

    for name in by_rule.get("above", []):
        cam = mounted[name]
        below = cam["mount"]["above"]
        if below not in out or mounted[below]["parent_prim"] != cam["parent_prim"]:
            raise ValueError(f"{name} is mounted above {below!r}, which is not a camera this script placed on "
                             f"{cam['parent_prim']}")
        depth = MODULE_BOX_M[2]
        if abs(float(cam["mount"]["clear_front_m"]) - depth / 2) > 1e-9:
            problems.append(f"{name}: mount.clear_front_m {cam['mount']['clear_front_m']} is not half the module box's "
                            f"depth ({depth / 2}) -- the box's back face would not sit on the head")
        heads = {copy: meshes.faces(copy, cam["parent_prim"]) for copy in (SIMULATED, OFFICIAL)}
        pos, q, info = forehead_mount(heads, out[below][0], cam["mount"])
        out[name] = (pos, q)
        say.append(f"{name}: frontmost head surface (x) inside the module's footprint -- {info['fronts']}; the "
                   f"{info['governing']} head governs -> pos {pos.tolist()}, xyzw {q.tolist()}")

    for name in by_rule.get("behind", []):
        cam = mounted[name]
        ref = cam["mount"]["behind"]
        if ref not in out or mounted[ref]["parent_prim"] != cam["parent_prim"]:
            raise ValueError(f"{name} is mounted behind {ref!r}, which is not a camera this script placed on "
                             f"{cam['parent_prim']}")
        depth = MODULE_BOX_M[2]
        if abs(float(cam["mount"]["clear_back_m"]) - depth / 2) > 1e-9:
            problems.append(f"{name}: mount.clear_back_m {cam['mount']['clear_back_m']} is not half the module box's "
                            f"depth ({depth / 2}) -- the box's front face would not sit on the back of the head")
        heads = {copy: meshes.faces(copy, cam["parent_prim"]) for copy in (SIMULATED, OFFICIAL)}
        pos, q, info = rear_mount(heads, out[ref][0], cam["mount"])
        out[name] = (pos, q)
        say.append(f"{name}: hindmost head surface (x) inside the module's footprint -- {info['backs']}; the "
                   f"{info['governing']} head governs -> pos {pos.tolist()}, xyzw {q.tolist()}")

    wrists = by_rule.get("inset_from_tip_m", [])
    for name in wrists:
        cam = mounted[name]
        pos, q, info = wrist_mount(meshes.faces(SIMULATED, cam["parent_prim"]), cam["mount"])
        out[name] = (pos, q)
        say.append(f"{name}: {cam['parent_prim']} tip x {info['tip_x']}, top z {info['top_z']} (the surface's highest "
                   f"point within {WRIST_BAND_X_M} m / {WRIST_BAND_Y_M} m) -> pos {pos.tolist()}, xyzw {q.tolist()}")
    if wrists:
        first = out[wrists[0]]
        for name in wrists[1:]:
            d = float(np.max(np.abs(out[name][0] - first[0])))
            say.append(f"{name}: {d * 1e3:.4f} mm from {wrists[0]}")
            if d > MIRROR_TOL_M or mounted[name]["mount"] != mounted[wrists[0]]["mount"]:
                problems.append(f"{name}: {d * 1e3:.3f} mm from {wrists[0]} (or another rule) -- not a mirror mount")
            out[name] = (first[0].copy(), first[1].copy())    # one value for both: the first in canon order's
    return out, say, problems


def geometry(canon: dict, offsets: dict, meshes: Meshes) -> tuple[list[str], list[str]]:
    """The checks on where the cameras stand, at ``offsets`` ({camera: (pos, xyzw)}): the canon's own in check
    mode, the extraction's before a write."""
    say: list[str] = []
    problems: list[str] = []
    for name, (pos, q) in offsets.items():
        cam = canon["cameras"][name]
        rule = _rule(cam)
        n = near_plane(meshes.faces(SIMULATED, cam["parent_prim"]), pos, q, cam)
        say.append(f"{name}: own link ({cam['parent_prim']}) nearest {n['nearest'] * 100:.2f} cm; {n['clipped']} vertices in "
                   f"view nearer than the near plane {n['near']} m; {n['cut']} faces cut inside the picture")
        if n["cut"]:
            problems.append(f"{name}: its near plane ({n['near']} m) cuts {n['cut']} faces of {cam['parent_prim']} inside "
                            f"the picture -- the render would show that body cut open")
        for copy in (SIMULATED, OFFICIAL) if rule in ADD_ON_RULES else (SIMULATED,):
            w = winding_number(meshes.faces(copy, cam["parent_prim"]), pos)
            say.append(f"{name}: optical centre's winding number about the {copy} {cam['parent_prim']} mesh {w:.4f}")
            if abs(w) > INSIDE_WINDING:
                problems.append(f"{name}: its optical centre {np.asarray(pos).tolist()} is inside the {copy} "
                                f"{cam['parent_prim']} mesh (winding number {w:.3f}) -- the camera is buried in it")
        if rule not in ADD_ON_RULES:
            continue
        for copy in (SIMULATED, OFFICIAL):
            gap = module_gap(meshes.faces(copy, cam["parent_prim"]), pos, q)
            say.append(f"{name}: the {copy} head is {gap * 1000:.3f} mm behind the module box's back face")
            if gap < -BOX_TOL_M:
                problems.append(f"{name}: the {copy} head reaches {-gap * 1000:.3f} mm into the module box")
        if rule != "above":
            continue
        below = cam["mount"]["above"]
        if below in offsets:
            plane, clear = box_frustum_separation(module_corners(pos, q), *offsets[below], canon["cameras"][below])
            box = " x ".join(f"{v * 100:g}" for v in MODULE_BOX_M)
            if plane is None:
                problems.append(f"{name}: the {box} cm module box reaches into {below}'s view frustum")
            else:
                say.append(f"{name}: the {box} cm module box is outside {below}'s view -- all 8 corners beyond its "
                           f"{plane} plane, the nearest by {clear:.2f} deg")
    return say, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="write offset_pos / offset_rot_xyzw into the canon")
    args = ap.parse_args()

    from generation.isaac_boot import launch_app

    simulation_app = launch_app({"headless": True})[1]
    code = 1
    try:
        from pxr import Usd

        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        from generation.envs.s1d.s1d_env_cfg import CAMERAS, S1dEnvCfg

        try:
            # The per-type rules first. load() hands back the file's own dict, so --write keeps its key order.
            canon = cameras.load(CAMERAS)
        except ValueError as exc:
            print(f"PROBLEM {exc}\n1 problem(s)")
            raise SystemExit(1) from exc
        urls = {OFFICIAL: ISAAC_NUCLEUS_DIR + "/Robots/Unitree/H1/h1.usd",
                SIMULATED: S1dEnvCfg(use_cameras=False, use_skin=False).scene.robot.spawn.usd_path}
        stages = {}
        for copy, url in urls.items():
            print(f"{copy} copy: {url}")
            stages[copy] = Usd.Stage.Open(url)
            if not stages[copy]:
                raise RuntimeError(f"could not open the {copy} copy: {url}")
        meshes = Meshes(stages)
        extracted, say, problems = extract(canon, stages, meshes)
        if args.write:
            at = extracted
        else:
            at = {n: o for n in extracted if (o := _offsets(canon["cameras"][n])) is not None}
        g_say, g_problems = geometry(canon, at, meshes)
        for line in say + [f"at the {'extracted' if args.write else 'canon'} offsets -- {x}" for x in g_say]:
            print(line)
        problems += g_problems
        if not args.write:
            problems += compare(canon, extracted)
        elif not problems:                  # a failed check writes nothing
            changed = write(Path(CAMERAS), canon, extracted)
            for line in changed:
                print(f"wrote {line}")
            print(f"{CAMERAS}: {len(changed)} value(s) changed")
        for p in problems:
            print(f"PROBLEM {p}")
        print("OK" if not problems else f"{len(problems)} problem(s)")
        code = 0 if not problems else 1
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        simulation_app.close(exit_code=code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
