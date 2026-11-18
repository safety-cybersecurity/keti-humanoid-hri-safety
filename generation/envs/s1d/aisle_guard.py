"""Aisle-width guard. Pure: no torch, no Isaac.

The dummy waits beside the robot's lane at wait_y, which reset_dummy_entry
derives from both bodies' measured reach (entry_geometry.derive_wait_y). The
front rack's inner face has to lie beyond that offset plus the dummy's own
half width plus a margin, or the dummy is parked inside a rack. The check runs
as a reset term after reset_dummy_entry, so it reads the offset that reset
actually drew instead of re-adding derive_wait_y's terms here.

Since 2026-10-01 the dummy stands turned to its own walk, a different turn every episode, so the
width it takes toward the rack is its horizontal radius (dummy_horizontal_radius): the unturned
half width (dummy_half_width) is what a dummy at 0 or 180 deg takes.
"""
from __future__ import annotations

import math


def _lateral_extent(region: dict) -> float:
    shape = region["shape"]
    if shape == "box":
        return region["size"][1] / 2.0
    if shape == "capsule":
        if str(region.get("axis", "z")).lower() == "y":
            return region["height"] / 2.0 + region["radius"]
        return region["radius"]
    if shape == "sphere":
        return region["radius"]
    raise ValueError(f"dummy region shape {shape!r} has no lateral extent rule")


def dummy_half_width(anthropometry: dict) -> float:
    """Widest |y| of the dummy's surface in its own (unyawed) frame — the pose it waits in."""
    return max(abs(r["center"][1]) + _lateral_extent(r) for r in anthropometry["regions"].values())


def _root_xy(anthropometry: dict) -> tuple[float, float]:
    """Where the root link stands in the spec's frame: the one region without a parent
    (build_dummy.py makes that region the articulation's root link, at the region's centre)."""
    root = _root(anthropometry)
    return float(root["center"][0]), float(root["center"][1])


def _horizontal_reach(region: dict, root_xy: tuple[float, float]) -> float:
    sides = (1.0, -1.0) if region.get("mirror_y") else (1.0,)
    return max(_reach_from(region, abs(float(region["center"][0]) - root_xy[0]),
                           abs(side * float(region["center"][1]) - root_xy[1])) for side in sides)


def _reach_from(region: dict, cx: float, cy: float) -> float:
    shape = region["shape"]
    if shape == "box":
        return math.hypot(cx + region["size"][0] / 2.0, cy + region["size"][1] / 2.0)
    if shape == "sphere":
        return math.hypot(cx, cy) + region["radius"]
    if shape == "capsule":
        axis = str(region.get("axis", "z")).lower()
        half = region["height"] / 2.0
        if axis == "z":
            return math.hypot(cx, cy) + region["radius"]
        if axis == "x":
            return math.hypot(cx + half, cy) + region["radius"]
        if axis == "y":
            return math.hypot(cx, cy + half) + region["radius"]
        raise ValueError(f"dummy capsule axis {axis!r} has no horizontal reach rule")
    raise ValueError(f"dummy region shape {shape!r} has no horizontal reach rule")


def dummy_horizontal_radius(anthropometry: dict) -> float:
    """Farthest horizontal distance from the root link's vertical axis to the dummy's surface. However
    the dummy is turned about that axis, its |y| surface stays within this of the root -- and the offset
    the guard checks is the root's. The spec's own origin is between the feet, 3 cm in front of the root."""
    root_xy = _root_xy(anthropometry)
    return max(_horizontal_reach(r, root_xy) for r in anthropometry["regions"].values())


def _root(anthropometry: dict) -> dict:
    roots = [r for r in anthropometry["regions"].values() if r.get("parent") is None]
    if len(roots) != 1:
        raise ValueError(f"the dummy needs exactly one region with parent null (its root link), found {len(roots)}")
    return roots[0]


def _lowest_z(region: dict) -> float:
    cz = float(region["center"][2])
    shape = region["shape"]
    if shape == "box":
        return cz - region["size"][2] / 2.0
    if shape == "sphere":
        return cz - region["radius"]
    if shape == "capsule":
        axis = str(region.get("axis", "z")).lower()
        if axis == "z":
            return cz - region["height"] / 2.0 - region["radius"]
        if axis in ("x", "y"):
            return cz - region["radius"]
        raise ValueError(f"dummy capsule axis {axis!r} has no lowest-point rule")
    raise ValueError(f"dummy region shape {shape!r} has no lowest-point rule")


def standing_root_height(anthropometry: dict) -> float:
    """How high the root link's centre is above the floor when the dummy stands on its lowest surface:
    the root region's centre height less the lowest point of any region. The hold servo carries the S1d
    dummy this high plus a clearance (commands._advance_dummy, 2026-10-02), so that its feet do not drag."""
    return float(_root(anthropometry)["center"][2]) - min(_lowest_z(r) for r in anthropometry["regions"].values())


def required_half_width(wait_y: float, dummy_half_width: float, margin: float) -> float:
    return wait_y + dummy_half_width + margin


def check(rack_inner_y: float, wait_y: float, dummy_half_width: float, margin: float) -> None:
    need = required_half_width(wait_y, dummy_half_width, margin)
    if rack_inner_y < need:
        raise ValueError(
            f"aisle too narrow: rack inner face {rack_inner_y:.3f} m < waiting offset {wait_y:.3f} + dummy half "
            f"width {dummy_half_width:.3f} + margin {margin:.3f} = {need:.3f} m")


def check_after_reset(env, env_ids, rack_inner_y: float, margin: float, dummy_half_width: float) -> None:
    """Reset term. Defined after reset_dummy_entry, so env.s1a_entry holds this reset's target."""
    offset = env.s1a_entry["target_xy"][env_ids, 1] - env.scene.env_origins[env_ids, 1]
    check(rack_inner_y, float(abs(offset).max()), dummy_half_width, margin)
