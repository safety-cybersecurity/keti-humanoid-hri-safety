"""Forward kinematics of the H1 without Isaac: where every link stands for a set of joint angles.

The frames come from generation/robots/unitree_h1_joints.json, which generation/robots/extract_joint_frames.py
reads from the pinned h1.usd. For a joint the physics engine places the child like this::

    child = parent  +  (pos0, rot0)  +  turn about the joint's axis by q  +  (pos1, rot1) inverted

("+" is composing poses: the right-hand pose is expressed in the left-hand one). A fixed joint is the same
without the turn. Only numpy is needed, so a placement worked out here can be tested and planned anywhere.

Poses are (position (3,), quaternion xyzw (4,)); a quaternion that comes out of this module is unit length with a
non-negative real part. Metres and radians throughout.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

JOINTS = Path(__file__).resolve().parent / "unitree_h1_joints.json"
DEFAULT_FRAME = "torso_link"
AXES = {"X": 0, "Y": 1, "Z": 2}
KINDS = ("revolute", "fixed")
SIDES = ("left", "right")
UNIT_TOL = 1e-6         # the file keeps float32 quaternions rounded to 9 decimals; they are normalised on use


# ---- rotations ----------------------------------------------------------------------------------

def quat_to_matrix(q) -> np.ndarray:
    """3x3 rotation matrix of a quaternion xyzw (normalised first, as the physics engine does)."""
    q = np.asarray(q, float)
    n = float(np.linalg.norm(q))
    if q.shape != (4,) or not math.isfinite(n) or n == 0.0:
        raise ValueError(f"사원수가 길이가 0 이 아닌 유한한 xyzw 네 수가 아닙니다: {q!r}")
    x, y, z, w = q / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_from_matrix(rot) -> np.ndarray:
    """Quaternion xyzw (unit, real part not negative) of a rotation matrix."""
    r = np.asarray(rot, float)
    if r.shape != (3, 3):
        raise ValueError(f"회전 행렬이 3x3 이 아닙니다: {r.shape}")
    trace = float(np.trace(r))
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = np.array([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4.0])
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(r[i, i] - r[j, j] - r[k, k] + 1.0)
        q = np.empty(4)
        q[i] = s / 4.0
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
        q[3] = (r[k, j] - r[j, k]) / s
    q = q / np.linalg.norm(q)
    return q if q[3] >= 0.0 else -q


def axis_rotation(axis: str, angle: float) -> np.ndarray:
    """3x3 rotation by ``angle`` (rad, right-handed) about the x, y or z axis ("X", "Y", "Z")."""
    c, s = math.cos(angle), math.sin(angle)
    if axis == "X":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "Y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "Z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"회전축이 X, Y, Z 중 하나가 아닙니다: {axis!r}")


# ---- the joint document -------------------------------------------------------------------------

def _numbers(value, count: int, where: str) -> np.ndarray:
    if (not isinstance(value, (list, tuple)) or len(value) != count
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)):
        raise ValueError(f"{where} 가 유한한 수 {count}개가 아닙니다: {value!r}")
    return np.array(value, float)


def _quaternion(value, where: str) -> np.ndarray:
    q = _numbers(value, 4, where)
    if abs(float(np.linalg.norm(q)) - 1.0) > UNIT_TOL:
        raise ValueError(f"{where} 의 사원수가 단위 길이가 아닙니다: {value!r}")
    return q


def joints_from_document(doc: Mapping) -> dict:
    """The joint document (what the file holds) checked and turned into arrays; a ValueError says what is wrong.

    Result: ``robot``, ``usd_sha256``, ``root_link``, ``joints`` (name -> the file's fields with the numbers as
    arrays), ``rest`` (link -> {"pos", "rot_xyzw"} as arrays), ``order`` (the joint names, a parent's joint
    before its children's) and ``note``. The joints must form a tree under ``root_link`` that reaches exactly the
    links ``rest`` lists."""
    if not isinstance(doc, Mapping):
        raise ValueError(f"관절 틀 문서가 객체가 아닙니다: {type(doc).__name__}")
    for key in ("robot", "usd_sha256", "root_link", "joints", "rest"):
        if key not in doc:
            raise ValueError(f"관절 틀 문서에 {key} 가 없습니다")
    root = doc["root_link"]
    raw_joints, raw_rest = doc["joints"], doc["rest"]
    if not isinstance(raw_joints, Mapping) or not isinstance(raw_rest, Mapping) or not raw_rest:
        raise ValueError("joints 와 rest 가 비어 있지 않은 객체가 아닙니다")
    if root not in raw_rest:
        raise ValueError(f"root_link {root!r} 이 rest 의 링크가 아닙니다: {sorted(raw_rest)}")
    rest = {}
    for link, entry in raw_rest.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"rest 의 {link} 가 객체가 아닙니다: {entry!r}")
        rest[link] = {"pos": _numbers(entry.get("pos"), 3, f"rest {link} 의 pos"),
                      "rot_xyzw": _quaternion(entry.get("rot_xyzw"), f"rest {link} 의 rot_xyzw")}
    joints = {}
    for name, entry in raw_joints.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"관절 {name} 이 객체가 아닙니다: {entry!r}")
        kind = entry.get("type")
        if kind not in KINDS:
            raise ValueError(f"관절 {name} 의 type 이 {KINDS} 중 하나가 아닙니다: {kind!r}")
        axis = entry.get("axis")
        if kind == "revolute" and axis not in AXES:
            raise ValueError(f"관절 {name} 의 axis 가 {tuple(AXES)} 중 하나가 아닙니다: {axis!r}")
        if kind == "fixed" and axis is not None:
            raise ValueError(f"고정 관절 {name} 에 axis 가 있습니다: {axis!r}")
        for link_key in ("parent", "child"):
            if entry.get(link_key) not in rest:
                raise ValueError(f"관절 {name} 의 {link_key} {entry.get(link_key)!r} 이 rest 의 링크가 아닙니다")
        joints[name] = {"type": kind, "axis": axis, "parent": entry["parent"], "child": entry["child"],
                        "pos0": _numbers(entry.get("pos0"), 3, f"관절 {name} 의 pos0"),
                        "rot0_xyzw": _quaternion(entry.get("rot0_xyzw"), f"관절 {name} 의 rot0_xyzw"),
                        "pos1": _numbers(entry.get("pos1"), 3, f"관절 {name} 의 pos1"),
                        "rot1_xyzw": _quaternion(entry.get("rot1_xyzw"), f"관절 {name} 의 rot1_xyzw")}
    return {"robot": doc["robot"], "usd_sha256": doc["usd_sha256"], "root_link": root, "joints": joints,
            "rest": rest, "order": _tree_order(root, joints, rest), "note": doc.get("note")}


def _tree_order(root: str, joints: Mapping, rest: Mapping) -> list[str]:
    """The joint names with every parent's joint before its children's; refuses anything that is not a tree
    from ``root`` over exactly the links ``rest`` lists."""
    owner = {}
    for name, joint in joints.items():
        child = joint["child"]
        if child == root:
            raise ValueError(f"관절 {name} 의 child 가 뿌리 링크 {root} 입니다")
        if child in owner:
            raise ValueError(f"링크 {child} 가 관절 둘({owner[child]}, {name})의 child 입니다")
        owner[child] = name
    by_parent: dict[str, list[str]] = {}
    for name, joint in joints.items():
        by_parent.setdefault(joint["parent"], []).append(name)
    order, seen, queue = [], {root}, [root]
    while queue:
        link = queue.pop(0)
        for name in by_parent.get(link, ()):
            order.append(name)
            seen.add(joints[name]["child"])
            queue.append(joints[name]["child"])
    missing = sorted(set(rest) - seen)
    if missing:
        raise ValueError(f"뿌리 링크 {root} 에서 관절로 닿지 않는 rest 의 링크: {missing} (닫힌 고리 포함)")
    return order


def load_joints(path: Path | None = None) -> dict:
    """The H1's joint document (``unitree_h1_joints.json`` unless another path is given), checked."""
    path = Path(path) if path is not None else JOINTS
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, Mapping):
        raise ValueError(f"관절 틀 파일이 객체가 아닙니다: {path}")
    return joints_from_document(doc)


# ---- the kinematics -----------------------------------------------------------------------------

def link_poses(joints: Mapping, q: Mapping[str, float], *,
               frame: str = DEFAULT_FRAME) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every link's (position, quaternion xyzw) in the frame of the link ``frame``, for joint angles ``q``
    (rad, by joint name; a joint left out is at 0). A name in ``q`` that is not a revolute joint of ``joints``
    is refused, as is an angle that is not a finite number and a ``frame`` that is not a link."""
    root = joints["root_link"]
    if frame not in joints["rest"]:
        raise ValueError(f"기준 링크 {frame!r} 이 없습니다: {sorted(joints['rest'])}")
    angles = {}
    for name, angle in q.items():
        joint = joints["joints"].get(name)
        if joint is None or joint["type"] != "revolute":
            raise ValueError(f"관절 {name!r} 은 각도를 가진 관절이 아닙니다 (회전 관절: "
                             f"{sorted(n for n, j in joints['joints'].items() if j['type'] == 'revolute')})")
        if isinstance(angle, bool) or not isinstance(angle, (int, float, np.floating, np.integer)) \
                or not math.isfinite(angle):
            raise ValueError(f"관절 {name} 의 각도가 유한한 수가 아닙니다: {angle!r}")
        angles[name] = float(angle)
    world = {root: (np.eye(3), np.zeros(3))}
    for name in joints["order"]:
        joint = joints["joints"][name]
        r_parent, p_parent = world[joint["parent"]]
        r0, r1 = quat_to_matrix(joint["rot0_xyzw"]), quat_to_matrix(joint["rot1_xyzw"])
        r_joint = r_parent @ r0                                  # the joint frame on the parent's side
        p_joint = p_parent + r_parent @ joint["pos0"]
        if joint["type"] == "revolute":
            r_joint = r_joint @ axis_rotation(joint["axis"], angles.get(name, 0.0))
        r_child = r_joint @ r1.T                                 # the child's own origin is pos1 behind that frame
        world[joint["child"]] = (r_child, p_joint - r_child @ joint["pos1"])
    r_frame, p_frame = world[frame]
    return {link: (r_frame.T @ (p - p_frame), quat_from_matrix(r_frame.T @ r)) for link, (r, p) in world.items()}


def forearm_tip(link_boxes: Mapping, poses: Mapping, side: str) -> np.ndarray:
    """The far end of a forearm in the frame ``poses`` is in: the +x end of the ``<side>_elbow_link`` box (the box
    centre plus its half size along x, the centre's y and z), placed by that link's pose. ``link_boxes`` is
    ``visibility.load_link_boxes()``'s {link: (centre, half sizes)}; ``poses`` is ``link_poses``'s result."""
    if side not in SIDES:
        raise ValueError(f"side 가 {SIDES} 중 하나가 아닙니다: {side!r}")
    link = f"{side}_elbow_link"
    if link not in link_boxes:
        raise ValueError(f"링크 상자에 {link} 가 없습니다")
    if link not in poses:
        raise ValueError(f"자세에 {link} 가 없습니다")
    centre, half = np.asarray(link_boxes[link][0], float), np.asarray(link_boxes[link][1], float)
    pos, quat = poses[link]
    end = centre + np.array([half[0], 0.0, 0.0])
    return np.asarray(pos, float) + quat_to_matrix(quat) @ end
