"""Where the robot holds the box it carries: the box fixed to one of its links, the arms at a pose that holds it.

The place is meta/scenarios.json's ``carry`` (S1e, S1f, S1g) -- the project's own choice (self-set), none of it a
standard value:

* ``asset`` -- the box's meta/assets.json key; ``link`` -- the robot link the box is fixed to, one of the H1's links
  (the links of generation/robots/unitree_h1_joints.json).
* ``bottom_m`` -- the middle of the box's bottom face in that link's frame (m). The box's centre is half its height
  above it, along the box's own up (the link's +z turned by ``yaw_deg``, which is the link's +z).
* ``yaw_deg`` -- the box's turn about the link's +z. The box's own axes (generation/scene/carried_box.py) are its
  width (x), its depth (y, the top label's up) and its height (z); at -90 its depth runs along the link's +x.
* ``arm_hold_rad`` -- the arm joints' angles that hold it, by the joint's name without its side: each one is set
  on both arms (``left_<name>`` and ``right_<name>`` of the joint map).
* ``release`` -- when the box leaves the robot (``first_human_contact``: the step the person is first touched).

carry_problems() checks a place against the robot at the hold pose (generation/robots/kinematics.py and the link
boxes of generation/robots/unitree_h1_link_boxes.json), for every box size of a range: the forearm tips under the
box's bottom edge, one beside each side face; no robot camera inside the box; the box's corners within the stop
test's turning reach; the box clear of its link's box. The hold pose is all it sees: the arms sway while the robot
walks and turns and the box is filtered from every robot link, so a camera on an arm (moving_cameras) can pass into
the box -- the run measures how near those come and counts the steps they spend inside. Pure numpy, no Isaac.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from generation import cameras, visibility
from generation.robots import kinematics

CARRY_KEY = "carry"
CARRY_KEYS = ("asset", "link", "bottom_m", "yaw_deg", "arm_hold_rad", "release", "determined_by", "note")
RELEASES = ("first_human_contact",)
DETERMINED_BY = "project"
SIDES = kinematics.SIDES
# carry_problems' bands (the project's own values, m):
# a forearm tip this far outside its side face of the box -- 0 touches it, more leaves a gap the box could slip in;
TIP_SIDE_GAP_M = (0.0, 0.06)
# a forearm tip this far above (+) or below (-) the box's bottom face -- the forearms hold its lower edge;
TIP_BOTTOM_M = (-0.01, 0.03)
# the box's corners may reach this much past the stop test's turning reach (rounding of the reach's own value);
REACH_SLACK_M = 0.005
# the box may go this deep into its link's box -- a link box wraps the mesh, so it is larger than the link.
LINK_OVERLAP_M = 0.01
# Cross products shorter than this are parallel axes and add no separating axis.
_AXIS_EPS = 1e-9


@dataclass(frozen=True)
class Carry:
    """The canon's ``carry`` of one scenario, read and checked by load_carry."""
    asset: str
    link: str
    bottom_m: tuple[float, float, float]
    yaw_deg: float
    arm_hold_rad: Mapping[str, float]
    release: str


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _text(where: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} 가 비어 있지 않은 문자열이 아닙니다: {value!r}")
    return value


@functools.lru_cache(maxsize=1)
def _robot_links() -> frozenset[str]:
    """The H1's link names: the links of its joint frames (kinematics.load_joints)."""
    return frozenset(kinematics.load_joints()["rest"])


def load_carry(entry: Mapping) -> Carry | None:
    """The ``carry`` of a meta/scenarios.json scenario entry, or None when it has none. A block that is not the
    canon's form is refused with a ValueError that names ``<scenario_id>.carry.<key>``."""
    if CARRY_KEY not in entry:
        return None
    where = f"{entry.get('scenario_id')}.{CARRY_KEY}"
    node = entry[CARRY_KEY]
    if not isinstance(node, Mapping):
        raise ValueError(f"{where} 가 객체가 아닙니다: {node!r}")
    for key in CARRY_KEYS:
        if key not in node:
            raise ValueError(f"{where}.{key} 가 없습니다")
    unknown = sorted(set(node) - set(CARRY_KEYS))
    if unknown:
        raise ValueError(f"{where} 에 모르는 키 {unknown} 가 있습니다 (키: {list(CARRY_KEYS)})")
    asset = _text(f"{where}.asset", node["asset"])
    link = _text(f"{where}.link", node["link"])
    if link not in _robot_links():
        raise ValueError(f"{where}.link {link!r} 이 H1 의 링크가 아닙니다 — 링크 이름은 "
                         f"generation/robots/unitree_h1_joints.json 의 rest 에서 옵니다: {sorted(_robot_links())}")
    bottom = node["bottom_m"]
    if not isinstance(bottom, (list, tuple)) or len(bottom) != 3 or not all(_number(v) for v in bottom):
        raise ValueError(f"{where}.bottom_m 이 유한한 수 3개의 목록이 아닙니다: {bottom!r}")
    if not _number(node["yaw_deg"]):
        raise ValueError(f"{where}.yaw_deg 이 유한한 수가 아닙니다: {node['yaw_deg']!r}")
    hold = node["arm_hold_rad"]
    if not isinstance(hold, Mapping) or not hold:
        raise ValueError(f"{where}.arm_hold_rad 이 비어 있지 않은 객체가 아닙니다: {hold!r}")
    angles = {}
    for name, angle in hold.items():
        if not isinstance(name, str) or not name or name.startswith(tuple(f"{s}_" for s in SIDES)):
            raise ValueError(f"{where}.arm_hold_rad.{name} 는 쪽을 뺀 이름이어야 합니다(양팔에 같은 값을 줍니다 — "
                             f"left_ · right_ 는 읽는 쪽이 붙임)")
        if not _number(angle):
            raise ValueError(f"{where}.arm_hold_rad.{name} 이 유한한 수가 아닙니다: {angle!r}")
        angles[name] = float(angle)
    if node["release"] not in RELEASES:
        raise ValueError(f"{where}.release 가 {list(RELEASES)} 중 하나가 아닙니다: {node['release']!r}")
    if node["determined_by"] != DETERMINED_BY:
        raise ValueError(f"{where}.determined_by 가 {DETERMINED_BY!r} 가 아닙니다 — 상자 자리는 자체 설정입니다: "
                         f"{node['determined_by']!r}")
    _text(f"{where}.note", node["note"])
    return Carry(asset=asset, link=link, bottom_m=tuple(float(v) for v in bottom), yaw_deg=float(node["yaw_deg"]),
                 arm_hold_rad=MappingProxyType(angles), release=node["release"])


def arm_hold_joints(carry: Carry, joint_names: Iterable[str]) -> dict[str, float]:
    """{joint name: angle} of the hold pose on both arms: ``left_<name>`` and ``right_<name>`` for every name of
    ``arm_hold_rad``, looked up in ``joint_names`` (the joint map's names). A joint either side lacks is refused."""
    names = set(joint_names)
    out, missing = {}, []
    for name, angle in carry.arm_hold_rad.items():
        for side in SIDES:
            joint = f"{side}_{name}"
            if joint in names:
                out[joint] = float(angle)
            else:
                missing.append(joint)
    if missing:
        raise ValueError(f"팔을 잡는 관절 {missing} 이 관절 이름에 없습니다 (arm_hold_rad {dict(carry.arm_hold_rad)})")
    return out


def moving_cameras(carry: Carry, cameras_canon: Mapping) -> tuple[str, ...]:
    """The robot's cameras (meta/cameras.json type "mounted") on a link other than the one the box is fixed to, in the
    canon's order -- the arms move them about the held box (the wrist cameras, for a box on the torso)."""
    return tuple(name for name in visibility.robot_cameras(cameras_canon)
                 if cameras_canon["cameras"][name].get("parent_prim") != carry.link)


def _size(size_m: object) -> np.ndarray:
    values = list(size_m) if isinstance(size_m, (list, tuple, np.ndarray)) else []
    if len(values) != 3 or not all(_number(v) and float(v) > 0 for v in values):
        raise ValueError(f"상자 크기는 양의 유한한 길이 셋(폭 · 깊이 · 높이 m)이어야 합니다: {size_m!r}")
    return np.array(values, dtype=float)


def box_pose(link_pos, link_quat_xyzw, carry: Carry, size_m) -> tuple[np.ndarray, np.ndarray]:
    """(centre, quaternion xyzw) of a box of ``size_m`` (width, depth, height) held at ``carry``, for its link at
    ``link_pos`` / ``link_quat_xyzw``: the centre half the height above the bottom's middle, the box turned by the
    link's orientation and then ``yaw_deg`` about the link's +z. The quaternion is unit with a non-negative real
    part."""
    size = _size(size_m)
    r_link = kinematics.quat_to_matrix(link_quat_xyzw)
    pos = np.asarray(link_pos, float)
    if pos.shape != (3,) or not np.isfinite(pos).all():
        raise ValueError(f"링크의 자리가 유한한 수 셋이 아닙니다: {link_pos!r}")
    r_box = r_link @ kinematics.axis_rotation("Z", math.radians(carry.yaw_deg))
    centre_link = np.asarray(carry.bottom_m, float) + np.array([0.0, 0.0, size[2] / 2.0])
    return pos + r_link @ centre_link, kinematics.quat_from_matrix(r_box)


def box_obb(link_pos, link_quat_xyzw, carry: Carry, size_m) -> visibility.Obb:
    """The held box as a turned box (visibility.Obb): box_pose's centre and orientation, half of ``size_m``."""
    centre, quat = box_pose(link_pos, link_quat_xyzw, carry, size_m)
    return visibility.Obb(center=centre, rot_xyzw=quat, half=_size(size_m) / 2.0)


def _overlap_depth(a: visibility.Obb, b: visibility.Obb) -> float:
    """How deep two turned boxes overlap: the least overlap of their projections over the separating axes (each
    box's three axes and their nine cross products), 0.0 when one axis separates them."""
    ra, rb = kinematics.quat_to_matrix(a.rot_xyzw), kinematics.quat_to_matrix(b.rot_xyzw)
    axes = [ra[:, i] for i in range(3)] + [rb[:, i] for i in range(3)]
    axes += [np.cross(ra[:, i], rb[:, j]) for i in range(3) for j in range(3)]
    between = b.center - a.center
    depth = math.inf
    for axis in axes:
        length = float(np.linalg.norm(axis))
        if length < _AXIS_EPS:
            continue
        u = axis / length
        reach = float(np.abs(ra.T @ u) @ a.half + np.abs(rb.T @ u) @ b.half)
        gap = reach - abs(float(between @ u))
        if gap <= 0.0:
            return 0.0
        depth = min(depth, gap)
    return depth


def _label(size: np.ndarray) -> str:
    return f"상자 {size[0]:.3f} × {size[1]:.3f} × {size[2]:.3f} m"


def carry_problems(carry: Carry, joints: Mapping, link_boxes: Mapping, cameras_canon: Mapping, *,
                   sizes: Sequence, reach_m: float) -> list[str]:
    """What is wrong with holding a box at ``carry`` with the arms at its hold pose, for every box size of
    ``sizes`` (width, depth, height): [] when nothing is. ``joints`` is kinematics.load_joints()'s document,
    ``link_boxes`` visibility.load_link_boxes()'s boxes, ``cameras_canon`` meta/cameras.json, ``reach_m`` the
    stop test's turning reach (meta/rules.json R-TS15066-SSM-02 stop_table.turning_reach_m). In the link's frame:

    1. each forearm tip (kinematics.forearm_tip) is between the box's back and front faces, outside its own side
       face by TIP_SIDE_GAP_M and within TIP_BOTTOM_M of its bottom face -- and the two tips on opposite sides;
    2. no camera on the robot (meta/cameras.json type "mounted") is inside the box;
    3. the box's corners are within ``reach_m`` + REACH_SLACK_M of the link's vertical axis;
    4. the box overlaps its link's box by at most LINK_OVERLAP_M.
    """
    if not isinstance(sizes, (list, tuple)) or not sizes:
        raise ValueError(f"sizes 가 상자 크기의 비어 있지 않은 목록이 아닙니다: {sizes!r}")
    checked = [_size(s) for s in sizes]
    if not _number(reach_m) or float(reach_m) <= 0:
        raise ValueError(f"reach_m 이 양의 유한한 길이가 아닙니다: {reach_m!r}")
    hold = arm_hold_joints(carry, joints["joints"])
    poses = kinematics.link_poses(joints, hold, frame=carry.link)
    link_pos, link_quat = poses[carry.link]
    tips = {side: kinematics.forearm_tip(link_boxes, poses, side) for side in SIDES}
    camera_at = {}
    for name in visibility.robot_cameras(cameras_canon):
        cam = cameras_canon["cameras"][name]
        parent_pos, parent_quat = poses[cam["parent_prim"]]
        camera_at[name] = cameras.mounted_pose(np.r_[parent_pos, parent_quat], cam["offset_pos"],
                                               cam["offset_rot_xyzw"])[0]
    out = []
    link_box = None
    if carry.link in link_boxes:
        centre, half = (np.asarray(v, float) for v in link_boxes[carry.link])
        link_box = visibility.Obb(center=link_pos + kinematics.quat_to_matrix(link_quat) @ centre,
                                  rot_xyzw=link_quat, half=half)
    else:
        out.append(f"링크 상자 파일에 {carry.link} 가 없습니다 — 상자와 몸통의 겹침을 잴 수 없습니다")
    for size in checked:
        label = _label(size)
        obb = box_obb(link_pos, link_quat, carry, size)
        to_box = kinematics.quat_to_matrix(obb.rot_xyzw).T
        half = obb.half
        across = {}
        for side, tip in tips.items():
            x, y, z = to_box @ (tip - obb.center)               # box frame: width x, depth y, height z
            across[side] = x
            if abs(y) > half[1]:
                out.append(f"{label}: {side} 아래팔 끝이 상자의 앞뒤 밖입니다 (뒷면에서 {y + half[1]:+.4f} m · "
                           f"앞면까지 {half[1] - y:+.4f} m)")
            gap = abs(x) - half[0]
            if not TIP_SIDE_GAP_M[0] <= gap <= TIP_SIDE_GAP_M[1]:
                out.append(f"{label}: {side} 아래팔 끝과 상자 옆면 사이가 {gap:+.4f} m 입니다 — "
                           f"{TIP_SIDE_GAP_M[0]} ~ {TIP_SIDE_GAP_M[1]} m 여야 합니다(음수면 끝이 상자 안)")
            lift = z + half[2]
            if not TIP_BOTTOM_M[0] <= lift <= TIP_BOTTOM_M[1]:
                out.append(f"{label}: {side} 아래팔 끝이 상자 밑면에서 {lift:+.4f} m 입니다 — "
                           f"{TIP_BOTTOM_M[0]:+} ~ {TIP_BOTTOM_M[1]:+} m 여야 합니다(아래팔이 밑면을 받침)")
        if across["left"] * across["right"] >= 0.0:
            out.append(f"{label}: 두 아래팔 끝이 상자의 같은 쪽에 있습니다 — 상자가 두 아래팔 사이에 있지 않습니다")
        for name, at in camera_at.items():
            local = to_box @ (at - obb.center)
            if (np.abs(local) < half).all():
                out.append(f"{label}: 카메라 {name} 가 상자 안에 있습니다 (상자 좌표 {np.round(local, 4).tolist()})")
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * half
        corners_link = (to_box.T @ corners.T).T + obb.center - link_pos
        radius = float(np.max(np.hypot(corners_link[:, 0], corners_link[:, 1])))
        if radius > float(reach_m) + REACH_SLACK_M:
            out.append(f"{label}: 상자 모서리의 수평 반지름 {radius:.4f} m 가 정지 시험의 돌기 반지름 {reach_m} m + "
                       f"{REACH_SLACK_M} m 를 넘습니다")
        if link_box is not None:
            depth = _overlap_depth(obb, link_box)
            if depth > LINK_OVERLAP_M:
                out.append(f"{label}: 상자가 몸통 링크({carry.link}) 상자와 {depth:.4f} m 겹칩니다 — "
                           f"{LINK_OVERLAP_M} m 까지(링크 상자는 메시를 감싸 실제보다 큼)")
    return out
