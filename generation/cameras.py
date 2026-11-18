"""meta/cameras.json loader, intrinsics, follow camera derivatives, and S1d helpers.
- load(): parse and validate meta/cameras.json (required camera keys; what each camera's type needs to place it;
  bbox.camera; packs, when declared)
- intrinsics(), focal_length_mm(): pinhole camera math
- site_placement(), site_sides(): a follow or fixed camera's placement at one scenario's site (its ``sites``
  block) -- the side's own where the site holds one per side the person comes from (``by_approach_side``)
- follow_eye_target(), follow_pose(), heading_yaw(): third-person camera eye/target from the robot root, the
  offsets turned by the placement's frame
- pose_record(): encode pose to parquet layout (pos3 + quat4 wxyz)
- mounted_pose(): world pose of camera fixed to an articulation link
- project(), project_points(): world point(s) to pixel in ROS optical frame
- optical_frame_xyzw(), look_at_pose(), tilted_along_x_xyzw(), fixed_pose(): S1d camera helpers
  (optical frames, fixed camera pose from a site's placement, recording fingerprint, ordered names)
Pure numpy (Isaac-free).

The cameras on the robot ride their links wherever the robot is. The third-person and CCTV cameras are
placed per scenario: each holds a ``sites`` block, one placement per scenario id, and every consumer
names the scenario it places them for -- a scenario without a placement is refused before a camera is
spawned. A site may hold its placement per side the person comes from instead: ``{"by_approach_side": {"west":
<placement>, "east": <placement>}}``, keyed by the conditions table's ``approach_side``, and every consumer then
names the side as well -- the episode takes the placement of its row's side."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from generation.scene import layout

HORIZONTAL_APERTURE_MM = 20.955   # Isaac Sim's default; the fov is the canon, this is the unit it is expressed in

_REQUIRED_CAMERA_KEYS = ("width", "height", "fov_h_deg", "channels", "type")
# How a camera is placed: on a robot link at a fixed offset, following the robot root, or fixed in
# the hall at a pose derived from its site's placement (fixed_pose). Anything else nothing here can place.
_CAMERA_TYPES = ("mounted", "follow", "fixed")
# The types placed per scenario, through a ``sites`` block.
_SITED_TYPES = ("follow", "fixed")
# What a follow placement's offsets are turned by: nothing (added to the root in the world's axes), the robot's
# heading at each step, or the heading it had right after the reset -- kept for the episode.
FOLLOW_FRAMES = ("world", "heading", "initial_heading")
# The mount rule fixed_pose reads: the rack face it stands in front of, four numbers and the aim's x range.
_FIXED_MOUNT_KEYS = ("side", "inset_from_rack_face_m", "height_m", "aim_x_range_m", "aim_y_m", "aim_z_m")
# A fixed placement's other form: the eye and the point it looks at, in the scene layout's coordinates.
_POINT_KEYS = ("eye_m", "aim_m")
_FIXED_MOUNT_NUMBERS = ("inset_from_rack_face_m", "height_m", "aim_y_m", "aim_z_m")
# A site's placement per side the person comes from: the key, and the sides -- the conditions table's approach_side
# (generation/conditions.py APPROACH_SIDES, copied so the camera canon does not import the table; the tests compare
# the two), in that order.
BY_APPROACH_SIDE = "by_approach_side"
APPROACH_SIDES = ("west", "east")
_SIDES = {"-y": -1.0, "+y": 1.0}
_UNIT_TOL = 1e-6


def load(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, cam in d["cameras"].items():
        for key in _REQUIRED_CAMERA_KEYS:
            if key not in cam:
                raise ValueError(f"cameras.json: {name} lacks {key}")
        _check_placement(name, cam)
    _check_bbox_camera(d)
    return d


def _finite_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _finite_numbers(value, n: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == n and all(_finite_number(v) for v in value)


def _vector(name: str, cam: dict, key: str, n: int) -> list[float]:
    value = cam.get(key)
    if value is None:
        raise ValueError(f"cameras.json: {name} ({cam['type']}) lacks {key}")
    if not _finite_numbers(value, n):
        raise ValueError(f"cameras.json: {name}.{key} is not {n} finite numbers: {value!r}")
    return [float(v) for v in value]


def _check_placement(name: str, cam: dict) -> None:
    """What the camera's type needs to be placed. mounted: the link and the offset from it, a ROS optical frame
    (mounted_pose composes it with the link's pose; the environment hands it to OffsetCfg as convention "ros"),
    and no ``sites`` -- it rides its link at every site. follow and fixed: a ``sites`` block with a placement
    per scenario -- follow: the frame and the offsets from the robot root (follow_pose); fixed: the mount rule
    or the eye and aim point fixed_pose reads."""
    kind = cam["type"]
    if kind == "mounted":
        if "sites" in cam:
            raise ValueError(f"cameras.json: {name} (mounted) has sites -- a camera on a robot link rides it "
                             "wherever the robot is")
        parent = cam.get("parent_prim")
        if not isinstance(parent, str) or not parent:
            raise ValueError(f"cameras.json: {name} (mounted) lacks parent_prim")
        _vector(name, cam, "offset_pos", 3)
        q = _vector(name, cam, "offset_rot_xyzw", 4)
        norm = math.sqrt(sum(v * v for v in q))
        if abs(norm - 1.0) > _UNIT_TOL:
            raise ValueError(f"cameras.json: {name}.offset_rot_xyzw {q} is not a unit quaternion (|q| = {norm!r})")
        if cam.get("convention") != "ros":
            raise ValueError(f"cameras.json: {name}.convention is {cam.get('convention')!r}, not 'ros' -- the offset "
                             "is a ROS optical frame (+z the optical axis, +y image-down)")
    elif kind in _SITED_TYPES:
        sites = cam.get("sites")
        if not isinstance(sites, dict) or not sites:
            raise ValueError(f"cameras.json: {name} ({kind}) lacks sites -- a placement per scenario, by its id")
        check = _check_follow_site if kind == "follow" else _check_fixed_site
        for sid, placement in sites.items():
            where = f"{name}.sites.{sid}"
            if not isinstance(placement, dict) or not placement:
                raise ValueError(f"cameras.json: {where} is not a placement: {placement!r}")
            if BY_APPROACH_SIDE in placement:
                _check_by_side(where, placement, check)
            else:
                check(where, placement)
    else:
        raise ValueError(f"cameras.json: {name}'s type {kind!r} is not one of {list(_CAMERA_TYPES)}")


def _check_by_side(where: str, placement: dict, check) -> None:
    """A site's placement per side the person comes from: ``by_approach_side`` its only key, an object holding a
    placement for each of APPROACH_SIDES and for no other side, each of them one the camera's type reads (``check``)
    and none holding sides of its own."""
    beside = sorted(k for k in placement if k != BY_APPROACH_SIDE)
    if beside:
        raise ValueError(f"cameras.json: {where} has {beside} beside {BY_APPROACH_SIDE} -- a placement per side is "
                         "the site's only key")
    by_side = placement[BY_APPROACH_SIDE]
    if not isinstance(by_side, dict):
        raise ValueError(f"cameras.json: {where}.{BY_APPROACH_SIDE} is not an object: {by_side!r}")
    unknown = [side for side in by_side if side not in APPROACH_SIDES]
    if unknown:
        raise ValueError(f"cameras.json: {where}.{BY_APPROACH_SIDE} has {unknown}, not a side the person comes from "
                         f"{list(APPROACH_SIDES)}")
    missing = [side for side in APPROACH_SIDES if side not in by_side]
    if missing:
        raise ValueError(f"cameras.json: {where}.{BY_APPROACH_SIDE} lacks a placement for {missing}")
    for side in APPROACH_SIDES:
        at, inner = f"{where}.{BY_APPROACH_SIDE}.{side}", by_side[side]
        if not isinstance(inner, dict) or not inner:
            raise ValueError(f"cameras.json: {at} is not a placement: {inner!r}")
        if BY_APPROACH_SIDE in inner:
            raise ValueError(f"cameras.json: {at} has a {BY_APPROACH_SIDE} of its own -- the side is chosen once")
        check(at, inner)


def _site_vector(where: str, placement: dict, key: str, n: int) -> list[float]:
    if key not in placement:
        raise ValueError(f"cameras.json: {where} lacks {key}")
    value = placement[key]
    if not _finite_numbers(value, n):
        raise ValueError(f"cameras.json: {where}.{key} is not {n} finite numbers: {value!r}")
    return [float(v) for v in value]


def _check_look(where: str, eye: list[float], target: list[float]) -> None:
    """An eye that can look at its target with a level picture: not the target itself, and not straight above or
    below it -- look_at_pose's image-up would be undefined."""
    d = np.subtract(target, eye)
    if not np.any(d):
        raise ValueError(f"cameras.json: {where}: the eye {eye} is the point it looks at")
    if math.hypot(d[0], d[1]) < 1e-9:
        raise ValueError(f"cameras.json: {where}: the eye {eye} looks straight up or down at {target} -- "
                         "the picture's up is undefined")


def _check_follow_site(where: str, placement: dict) -> None:
    """A follow placement follow_pose can read: one of FOLLOW_FRAMES and the eye and look-at offsets, three
    finite numbers each, apart and not one straight above the other."""
    frame = placement.get("frame")
    if frame not in FOLLOW_FRAMES:
        raise ValueError(f"cameras.json: {where}.frame {frame!r} is not one of {list(FOLLOW_FRAMES)}")
    eye = _site_vector(where, placement, "eye_offset", 3)
    _check_look(where, eye, _site_vector(where, placement, "lookat_offset", 3))


def _check_fixed_site(where: str, placement: dict) -> None:
    """A fixed placement fixed_pose can read, in exactly one of two forms: the rack-face mount rule (six keys) or
    the eye and the point it aims at (eye_m, aim_m -- three finite numbers each, apart and not one straight above
    the other)."""
    has_mount, has_point = "mount" in placement, any(k in placement for k in _POINT_KEYS)
    if has_mount and has_point:
        raise ValueError(f"cameras.json: {where} has both a mount rule and eye_m/aim_m -- one form or the other")
    if not has_mount and not has_point:
        raise ValueError(f"cameras.json: {where} has neither a mount rule nor eye_m and aim_m: {sorted(placement)}")
    if has_point:
        eye = _site_vector(where, placement, "eye_m", 3)
        _check_look(where, eye, _site_vector(where, placement, "aim_m", 3))
        return
    mount = placement["mount"]
    if not isinstance(mount, dict):
        raise ValueError(f"cameras.json: {where}.mount is not an object: {mount!r}")
    missing = [k for k in _FIXED_MOUNT_KEYS if k not in mount]
    if missing:
        raise ValueError(f"cameras.json: {where}.mount lacks {missing}")
    _check_fixed_mount(where, mount)


def _check_fixed_mount(where: str, mount: dict) -> None:
    """The values fixed_pose reads, held to what it can read: the side one of _SIDES, four finite numbers (not
    bools), and the aim's x range two finite numbers, low before high. Refused here, naming the canon, rather than
    met by fixed_pose later as a bare TypeError or ValueError."""
    if mount["side"] not in _SIDES:
        raise ValueError(f"cameras.json: {where}.mount.side {mount['side']!r} is not one of {sorted(_SIDES)}")
    for key in _FIXED_MOUNT_NUMBERS:
        if not _finite_number(mount[key]):
            raise ValueError(f"cameras.json: {where}.mount.{key} is not a finite number: {mount[key]!r}")
    aim = mount["aim_x_range_m"]
    if not _finite_numbers(aim, 2):
        raise ValueError(f"cameras.json: {where}.mount.aim_x_range_m is not 2 finite numbers: {aim!r}")
    if not aim[0] < aim[1]:
        raise ValueError(f"cameras.json: {where}.mount.aim_x_range_m {aim!r} is not a range (lo < hi)")


# The channel the box is computed from (the dummy's pixels in the segmentation, bbox.source).
_BBOX_CHANNEL = "semantic_segmentation"


def _check_bbox_camera(d: dict) -> None:
    """bbox.camera is the camera whose segmentation the box is computed from: one of the canon's
    cameras, recording that channel. Packs are optional -- the canon switches them off by leaving
    ``packs`` out -- and a pack's mask comes from the same segmentation, so every
    pack the canon declares is on that camera."""
    box = d.get("bbox")
    camera = box.get("camera") if isinstance(box, dict) else None
    if not isinstance(camera, str) or camera not in d["cameras"]:
        raise ValueError(f"cameras.json: bbox.camera {camera!r} is not one of the cameras "
                         f"{sorted(d['cameras'])}")
    if _BBOX_CHANNEL not in d["cameras"][camera]["channels"]:
        raise ValueError(f"cameras.json: bbox.camera {camera} does not record {_BBOX_CHANNEL} -- "
                         "the box is computed from it")
    packs = d.get("packs")
    if packs is None:
        return
    if not isinstance(packs, dict):
        raise ValueError(f"cameras.json: packs is not an object: {type(packs).__name__}")
    for name, pack in packs.items():
        pack_camera = pack.get("camera") if isinstance(pack, dict) else None
        if pack_camera != camera:
            raise ValueError(f"cameras.json: pack {name}'s camera {pack_camera!r} is not bbox.camera "
                             f"{camera!r} -- the box and the pack's mask come from one camera's "
                             f"{_BBOX_CHANNEL}")


def focal_length_mm(fov_h_deg: float, horizontal_aperture_mm: float = HORIZONTAL_APERTURE_MM) -> float:
    return horizontal_aperture_mm / (2.0 * math.tan(math.radians(fov_h_deg) / 2.0))


def intrinsics(cam: dict) -> tuple[float, float, float, float]:
    w, h = cam["width"], cam["height"]
    fx = w / (2.0 * math.tan(math.radians(cam["fov_h_deg"]) / 2.0))
    return fx, fx, w / 2.0, h / 2.0


def follow_eye_target(root_pos: np.ndarray, eye_offset, lookat_offset) -> tuple[np.ndarray, np.ndarray]:
    """Where the third-person camera sits and what it looks at: world offsets added to the
    robot root, as generation/envs/s1a/watch.py's recordings do."""
    root = np.asarray(root_pos, float)
    return root + np.asarray(eye_offset, float), root + np.asarray(lookat_offset, float)


def _site_entry(cam: Mapping, scenario_id: str, label: str) -> Mapping:
    """The camera's ``sites`` entry for the scenario as the canon holds it. A camera on a robot link has no sites,
    and a scenario the camera has no placement for is refused."""
    sites = cam.get("sites")
    if not isinstance(sites, Mapping):
        raise ValueError(f"{label} 에 자리(sites)가 없습니다 — 로봇 링크에 붙은 카메라(mounted)는 자리마다 놓지 않습니다")
    if scenario_id not in sites:
        raise ValueError(f"{label} 에 {scenario_id} 의 자리가 없음 — 자리가 있는 시나리오: {sorted(sites)}")
    return sites[scenario_id]


def site_placement(cam: Mapping, scenario_id: str, *, name: str | None = None,
                   approach_side: str | None = None) -> dict:
    """The placement of a follow or fixed camera at one scenario's site: its ``sites`` entry for that scenario
    id; where that entry holds a placement per side the person comes from (``by_approach_side``), the one of
    ``approach_side`` -- the conditions table's column of that name, the side the person comes from. A camera on a
    robot link has no sites, a scenario the camera has no placement for is refused, and so is a side the entry
    holds no placement for (none given among them), naming the sides it holds -- each naming the camera when
    ``name`` is given. A site with one placement does not read ``approach_side``."""
    label = name or f"{cam.get('type')} 카메라"
    placement = _site_entry(cam, scenario_id, label)
    by_side = placement.get(BY_APPROACH_SIDE) if isinstance(placement, Mapping) else None
    if by_side is None:
        return dict(placement)
    if not isinstance(approach_side, str) or approach_side not in by_side:
        raise ValueError(f"{label} 의 {scenario_id} 자리는 사람이 오는 쪽마다 하나입니다 — {approach_side!r} 의 자리가 없음 "
                         f"(고를 수 있는 쪽: {list(by_side)} — 조건표의 approach_side)")
    return dict(by_side[approach_side])


def site_sides(cam: Mapping, scenario_id: str, *, name: str | None = None) -> tuple[str, ...]:
    """The sides the person comes from that the camera's site at ``scenario_id`` holds a placement for, in the
    canon's order; () where one placement serves every episode there. Refused like site_placement."""
    placement = _site_entry(cam, scenario_id, name or f"{cam.get('type')} 카메라")
    by_side = placement.get(BY_APPROACH_SIDE) if isinstance(placement, Mapping) else None
    return tuple(by_side) if isinstance(by_side, Mapping) else ()


def heading_yaw(quat_xyzw) -> float:
    """The yaw of a body's +x seen from above (rad, counter-clockwise from +x) for its orientation (xyzw) -- the
    robot's heading as Isaac Lab's heading_w reads it, so a robot leaning forward has the heading it walks along.
    What follow_pose's ``root_yaw`` and ``initial_yaw`` are."""
    forward = _q_rot(np.asarray(quat_xyzw, float), (1.0, 0.0, 0.0))
    return float(math.atan2(forward[1], forward[0]))


def _turned_about_z(v, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    x, y, z = (float(t) for t in v)
    return np.array([c * x - s * y, s * x + c * y, z])


def follow_pose(placement: Mapping, root_pos, root_yaw: float,
                initial_yaw: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Where a follow camera sits and what it looks at, (eye, target), from the robot root and its placement:
    ``world`` adds the offsets to the root as they are (follow_eye_target); ``heading`` turns them about the
    vertical by the robot's yaw ``root_yaw`` (rad), so the camera keeps its place behind or beside the robot;
    ``initial_heading`` by the yaw the robot had right after the reset, ``initial_yaw`` -- kept for the episode
    while the robot turns. Yaw is counter-clockwise from +x."""
    frame = placement.get("frame")
    if frame not in FOLLOW_FRAMES:
        raise ValueError(f"3인칭 자리의 frame {frame!r} 이 {list(FOLLOW_FRAMES)} 중 하나가 아닙니다")
    eye, look = placement["eye_offset"], placement["lookat_offset"]
    if frame == "world":
        return follow_eye_target(root_pos, eye, look)
    yaw = root_yaw if frame == "heading" else initial_yaw
    if yaw is None:
        raise ValueError("initial_heading 자리에 처음 요가 없습니다 — 리셋 뒤의 요를 먼저 잡으십시오(camera_io.Rig.reset)")
    if not _finite_number(yaw):
        raise ValueError(f"요가 유한한 수가 아닙니다: {yaw!r}")
    root = np.asarray(root_pos, float)
    return root + _turned_about_z(eye, yaw), root + _turned_about_z(look, yaw)


# ---- poses (xyzw, the engine's order; the recorded fields are wxyz like observation.root_pose) ----

def _q_mul(a, b) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def _q_rot(q, v) -> np.ndarray:
    u, w = np.asarray(q[:3], float), float(q[3])
    v = np.asarray(v, float)
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


def mounted_pose(parent_pose, offset_pos, offset_rot_xyzw) -> tuple[np.ndarray, np.ndarray]:
    """World pose of a camera fixed to a link: the link's pose (pos3 + quat xyzw, as
    body_link_pose_w hands it back) composed with the canon offset. With the canon's ros offset
    this is the camera's ROS optical frame.

    Derived rather than read: for a camera under an articulation link the engine keeps
    reporting the spawn pose while the render follows the link (measured 2026-09-23)."""
    p = np.asarray(parent_pose, float)
    return p[:3] + _q_rot(p[3:7], offset_pos), _q_mul(p[3:7], np.asarray(offset_rot_xyzw, float))


def project(point_w, pos, quat_xyzw, cam: dict) -> tuple[float, float, float]:
    """Pixel (u, v) and distance to the image plane of a world point, seen from a ROS optical
    frame (x right, y down, z forward) at pos/quat_xyzw — the frame the pose fields describe."""
    fx, fy, cx, cy = intrinsics(cam)
    q = np.asarray(quat_xyzw, float)
    x, y, z = _q_rot([-q[0], -q[1], -q[2], q[3]], np.asarray(point_w, float) - np.asarray(pos, float))
    return fx * x / z + cx, fy * y / z + cy, z


def project_points(points_w, pos, quat_xyzw, cam: dict) -> np.ndarray:
    """project() for many points at once: an (N, 3) array of world points to an (N, 3) array of u, v and
    the distance to the image plane, row for row the numbers project() gives. project() itself takes one
    point -- handed three it reads the rows as x, y and z and answers without a word."""
    points = np.asarray(points_w, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"점은 (N, 3) 배열이어야 합니다: {points.shape}")
    fx, fy, cx, cy = intrinsics(cam)
    q = np.asarray(quat_xyzw, float)
    local = _q_rot([-q[0], -q[1], -q[2], q[3]], points - np.asarray(pos, float))
    x, y, z = local[:, 0], local[:, 1], local[:, 2]
    return np.column_stack([fx * x / z + cx, fy * y / z + cy, z])


def pose_record(pos, quat_xyzw) -> np.ndarray:
    """(pos3, quat xyzw) -> the seven float32 numbers a pose field stores: pos3 + quat4 (wxyz)."""
    q = np.asarray(quat_xyzw, float)
    return np.array([*np.asarray(pos, float), q[3], q[0], q[1], q[2]], dtype=np.float32)


# ---- pure helpers: optical frames, fixed poses, recording fingerprint, camera names ----

# The keys the converter adds to the shipped copy (shipped_cameras): derived, not recorded.
_DERIVED_KEYS = ("fx", "fy", "cx", "cy")


def _mat_to_xyzw(R) -> np.ndarray:
    R = np.asarray(R, float)
    t = float(np.trace(R))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4.0])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        q = np.zeros(4)
        q[i], q[3] = s / 4.0, (R[k, j] - R[j, k]) / s
        q[j], q[k] = (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    q /= np.linalg.norm(q)
    return q if q[3] >= 0.0 else -q


def optical_frame_xyzw(forward, up) -> np.ndarray:
    """ROS optical frame (x right, y down, z forward) with z along `forward` and image-up as close to `up`
    as `forward` allows, in the parent's axes."""
    z = np.asarray(forward, float)
    z_norm = float(np.linalg.norm(z))
    if z_norm < 1e-9:
        raise ValueError(f"forward {tuple(forward)} is zero-length: optical axis is undefined")
    z = z / z_norm
    x = np.cross(z, np.asarray(up, float))
    n = float(np.linalg.norm(x))
    if n < 1e-9:
        raise ValueError(f"forward {tuple(forward)} is parallel to up {tuple(up)}: the image roll is undefined")
    x /= n
    return _mat_to_xyzw(np.column_stack([x, np.cross(z, x), z]))


def look_at_pose(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    eye = np.asarray(eye, float)
    return eye, optical_frame_xyzw(np.asarray(target, float) - eye, up)


def tilted_along_x_xyzw(tilt_down_deg: float) -> np.ndarray:
    """Optical frame along the parent's +x, turned toward -z by the tilt, image-up toward +z. Tilt 0 is the
    horizontal head offset (-0.5, 0.5, -0.5, 0.5) up to sign."""
    t = math.radians(float(tilt_down_deg))
    return optical_frame_xyzw((math.cos(t), 0.0, -math.sin(t)), (math.sin(t), 0.0, math.cos(t)))


def fixed_pose(cam: dict, scene_spec: dict, scenario_id: str,
               approach_side: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """World pose of a fixed camera at one scenario's site (site_placement -- of ``approach_side`` where the site
    holds a placement per side the person comes from), in the scene layout's frame. The
    mount rule: in front of the rack face on its side, at its height, aimed at the middle of its aim range --
    derived like the intrinsics, so the canon holds the rule, not the pose. The point form: at ``eye_m``,
    looking at ``aim_m`` with a level picture (look_at_pose); the scene spec is not read."""
    placement = site_placement(cam, scenario_id, approach_side=approach_side)
    has_mount, has_point = "mount" in placement, any(k in placement for k in _POINT_KEYS)
    if has_mount == has_point:
        raise ValueError(f"{scenario_id} 의 고정 카메라 자리는 mount 와 eye_m/aim_m 중 하나만 씁니다 (both / neither): "
                         f"{sorted(placement)}")
    if has_point:
        return look_at_pose(placement["eye_m"], placement["aim_m"])
    m = placement["mount"]
    if m.get("side") not in _SIDES:
        raise ValueError(f"fixed camera side must be one of {sorted(_SIDES)}, not {m.get('side')!r}")
    x0, x1 = (float(v) for v in m["aim_x_range_m"])
    target = np.array([(x0 + x1) / 2.0, float(m["aim_y_m"]), float(m["aim_z_m"])])
    y = _SIDES[m["side"]] * (layout.rack_inner_face_y(scene_spec) - float(m["inset_from_rack_face_m"]))
    return look_at_pose((target[0], y, float(m["height_m"])), target)


def camera_names(canon: dict) -> list[str]:
    return list(canon["cameras"])


def _baked(node):
    """``node`` without commentary or derived keys at any depth: ``*_note`` and ``description``, and the fx/fy/cx/cy
    the converter adds to the shipped copy (shipped_cameras). Every other key under ``node`` is kept --
    recording_fingerprint chooses which subtrees go in."""
    if isinstance(node, dict):
        return {k: _baked(v) for k, v in node.items()
                if not k.endswith("_note") and k not in ("description", *_DERIVED_KEYS)}
    if isinstance(node, list):
        return [_baked(v) for v in node]
    return node


def recording_fingerprint(canon: dict) -> str:
    """sha256 of the camera canon as a recording depends on it. media_writer stamps it on the sidecar; the converter
    (check_canon_fingerprint) and C9 (compare_published.pack_pair) hold a recording to it.

    The rule is one way: every key a recording path reads is fingerprinted -- not only those. ``fps``, ``render``
    and the whole ``cameras`` subtree go in (``type``, the mounted cameras' ``mount`` rule, which
    extract_camera_mounts.py reads and the recorder does not, and every site's placement of the third-person and
    CCTV cameras, the other sites' too), ``bbox`` with ``field``, ``classes_union`` and ``invalid``, the
    ``seeing`` rule (the person's sample points the visible-fraction columns count, and the thresholds they are
    read with -- generation/visibility.py), and each declared pack. Left out: commentary (``*_note`` and
    ``description`` at any depth, ``bbox.source``, ``bbox.coords``), the ``pose`` block (the pose fields' layout
    and frames, stated as strings), the fx/fy/cx/cy the converter derives for the shipped copy, a pack's
    ``downsample`` (applied at conversion) and the other top-level keys (``cameras_version``,
    ``schema_version``). A site's placements per side the person comes from are under ``cameras`` too."""
    baked = {k: canon[k] for k in ("fps", "cameras", "render", "seeing") if k in canon}
    if "bbox" in canon:
        # bbox.source and bbox.coords describe the box in words; nothing computes it from them
        baked["bbox"] = {k: v for k, v in canon["bbox"].items()
                         if k not in ("source", "coords")}
    # A pack's downsample is applied by the converter, so changing it calls for a re-conversion, not a new recording
    if canon.get("packs"):
        baked["packs"] = {n: {k: v for k, v in p.items() if k != "downsample"} for n, p in canon["packs"].items()}
    blob = json.dumps(_baked(baked), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
