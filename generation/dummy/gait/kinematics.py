"""Rotations and the jointed dummy's leg kinematics. Pure numpy.

Frames are the dummy's body axes at rest: +x front, +y left, +z up, origin at the centre of the soles.
A joint's rotation is its child's frame in its parent's frame, the identity at the stand pose."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LegGeometry:
    hip: np.ndarray      # (3,) pivots at the stand pose
    knee: np.ndarray
    ankle: np.ndarray
    sole: np.ndarray     # (4, 3) the foot box's bottom corners at the stand pose


CONVENTIONS = ("rotvec", "xyz", "xzy", "yxz", "yzx", "zxy", "zyx")


def axis_rotation(axis: str, angle) -> np.ndarray:
    a = np.asarray(angle, float)
    c, s = np.cos(a), np.sin(a)
    one, zero = np.ones_like(a), np.zeros_like(a)
    if axis == "x":
        rows = [[one, zero, zero], [zero, c, -s], [zero, s, c]]
    elif axis == "y":
        rows = [[c, zero, s], [zero, one, zero], [-s, zero, c]]
    elif axis == "z":
        rows = [[c, -s, zero], [s, c, zero], [zero, zero, one]]
    else:
        raise ValueError(f"axis must be x, y or z, got {axis!r}")
    return np.moveaxis(np.array(rows), (0, 1), (-2, -1))


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = np.zeros_like(x)
    return np.stack([np.stack([o, -z, y], -1), np.stack([z, o, -x], -1), np.stack([-y, x, o], -1)], -2)


def rotvec_to_matrix(rv) -> np.ndarray:
    rv = np.asarray(rv, float)
    t = np.linalg.norm(rv, axis=-1)[..., None, None]
    small = t < 1e-8
    safe = np.where(small, 1.0, t)
    a = np.where(small, 1.0 - t ** 2 / 6.0, np.sin(t) / safe)
    b = np.where(small, 0.5 - t ** 2 / 24.0, (1.0 - np.cos(t)) / safe ** 2)
    K = _skew(rv)
    return np.eye(3) + a * K + b * (K @ K)


def matrix_to_rotvec(R) -> np.ndarray:
    R = np.asarray(R, float)
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos)
    if np.any(theta > np.pi - 1e-6):
        raise ValueError("a rotation of about 180 deg has no unique rotation vector")
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    small = theta < 1e-8
    s = np.where(small, 0.5 + theta ** 2 / 12.0, theta / (2.0 * np.sin(np.where(small, 1.0, theta))))
    return w * s[..., None]


def euler_to_matrix(order: str, angles) -> np.ndarray:
    """Intrinsic: R = R[order[0]] @ R[order[1]] @ R[order[2]]."""
    angles = np.asarray(angles, float)
    R = axis_rotation(order[0], angles[..., 0])
    for i, ax in enumerate(order[1:], 1):
        R = R @ axis_rotation(ax, angles[..., i])
    return R


def dofs_to_matrix(values, axes: str, convention: str) -> np.ndarray:
    """A joint's rotation from its positions; ``axes`` names its free axes in order, the others are 0."""
    values = np.asarray(values, float)
    full = np.zeros(values.shape[:-1] + (3,))
    for i, ax in enumerate(axes):
        full[..., "xyz".index(ax)] = values[..., i]
    if convention == "rotvec":
        return rotvec_to_matrix(full)
    if convention in CONVENTIONS:
        return euler_to_matrix(convention, full[..., ["xyz".index(c) for c in convention]])
    raise ValueError(f"unknown joint-position convention {convention!r}")


def matrix_to_dofs(R, axes: str, convention: str) -> np.ndarray:
    """The free components of R's joint positions: for rotvec, the rotation vector's ``axes`` components."""
    if convention != "rotvec":
        raise NotImplementedError("only the rotvec convention is decomposed")
    return matrix_to_rotvec(R)[..., ["xyz".index(a) for a in axes]]


def swing_rotvec(R) -> np.ndarray:
    """The rotation vector of R's swing about z: the smallest rotation that takes z where R takes it, so R's
    twist about its own z is dropped. Its z component is exactly 0 -- for a joint whose z is held at zero this
    is the nearest pose that keeps the child's z axis (the jointed dummy's ankle keeps the sole's normal)."""
    R = np.asarray(R, float)
    v = R[..., :, 2]
    s = np.hypot(v[..., 0], v[..., 1])                 # |z x v|
    angle = np.arctan2(s, v[..., 2])
    if np.any(angle > np.pi - 1e-6):
        raise ValueError("z turned upside down has no unique swing")
    small = s < 1e-12
    k = np.where(small, 0.0, angle / np.where(small, 1.0, s))
    return np.stack([-v[..., 1] * k, v[..., 0] * k, np.zeros_like(s)], -1)


def rotation_between(u, v) -> np.ndarray:
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    axis = np.cross(u, v)
    s, c = float(np.linalg.norm(axis)), float(np.dot(u, v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        raise ValueError("opposite vectors have no unique minimal rotation")
    return rotvec_to_matrix(axis / s * math.atan2(s, c))


def leg_fk(geom: LegGeometry, R_hip, knee: float, R_ankle):
    """Ankle position, sole corners and foot rotation in the body frame with the pelvis at its stand pose."""
    knee_pos = geom.hip + R_hip @ (geom.knee - geom.hip)
    R_shank = R_hip @ axis_rotation("y", knee)
    ankle_pos = knee_pos + R_shank @ (geom.ankle - geom.knee)
    R_foot = R_shank @ R_ankle
    return ankle_pos, ankle_pos + (geom.sole - geom.ankle) @ R_foot.T, R_foot
