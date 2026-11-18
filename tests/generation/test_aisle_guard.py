"""Aisle-width guard: the front rack's inner face must lie beyond the
dummy's derived waiting offset + the dummy's half width + a margin. Isaac-free — the
reset term is exercised here with a stand-in env holding numpy arrays."""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from generation.envs.s1d import aisle_guard as g
from generation.scene import layout

REPO = Path(__file__).resolve().parents[2]


def _regions(**regions):
    return {"regions": regions}


# Hand-derived lateral surfaces: |centre y| + that shape's extent along y.
BOX = {"shape": "box", "size": [0.9, 0.2, 0.1], "center": [0.7, -0.15, 1.0]}                      # 0.15 + 0.10 = 0.25
Z_CAPSULE = {"shape": "capsule", "radius": 0.04, "height": 0.5, "axis": "z", "center": [0.0, 0.2, 1.0]}  # 0.20 + 0.04 = 0.24
Y_CAPSULE = {"shape": "capsule", "radius": 0.03, "height": 0.3, "axis": "y", "center": [0.0, 0.05, 1.0]}  # 0.05 + 0.15 + 0.03 = 0.23
SPHERE = {"shape": "sphere", "radius": 0.02, "center": [0.3, -0.2, 1.0]}                          # 0.20 + 0.02 = 0.22


@pytest.mark.parametrize("region,want", [(BOX, 0.25), (Z_CAPSULE, 0.24), (Y_CAPSULE, 0.23), (SPHERE, 0.22)])
def test_dummy_half_width_reaches_each_shape_s_lateral_surface(region, want):
    assert g.dummy_half_width(_regions(A=region)) == pytest.approx(want)


def test_dummy_half_width_is_the_widest_region():
    assert g.dummy_half_width(_regions(a=SPHERE, b=BOX, c=Y_CAPSULE, d=Z_CAPSULE)) == pytest.approx(0.25)


def test_dummy_half_width_refuses_a_shape_it_does_not_know():
    with pytest.raises(ValueError, match="cone"):
        g.dummy_half_width(_regions(A={"shape": "cone", "radius": 0.1, "center": [0.0, 0.0, 1.0]}))


def test_dummy_half_width_stops_on_a_missing_dimension():
    with pytest.raises(KeyError):          # not a silent 0
        g.dummy_half_width(_regions(A={"shape": "sphere", "center": [0.0, 0.3, 1.0]}))


# -- the dummy turned to its walk: its width toward the rack depends on the turn --
# The dummy turns about its root link -- the one region with no parent -- and the guard's offset is that
# root's. Measured from the spec's own origin instead (between the feet) the real dummy's radius came out
# 3.4 mm short: its root sits 3 cm behind that origin.
ROOT_XY = (-0.03, 0.0)


def _rooted(**regions):
    root = {"shape": "sphere", "radius": 0.001, "center": [*ROOT_XY, 1.4], "parent": None}
    return {"regions": {"ROOT": root, **{k: {**r, "parent": "ROOT"} for k, r in regions.items()}}}


# Hand-derived: from the root's vertical axis to the farthest point of each surface (x offsets are +0.03).
X_CAPSULE = {"shape": "capsule", "radius": 0.03, "height": 0.3, "axis": "x", "center": [0.1, 0.05, 1.0]}
RADII = [(BOX, math.hypot(0.73 + 0.45, 0.15 + 0.10)),          # the far corner
         (Z_CAPSULE, math.hypot(0.03, 0.2) + 0.04),
         (Y_CAPSULE, math.hypot(0.03, 0.05 + 0.15) + 0.03),    # the far end
         (X_CAPSULE, math.hypot(0.13 + 0.15, 0.05) + 0.03),
         (SPHERE, math.hypot(0.33, 0.2) + 0.02)]


@pytest.mark.parametrize("region,want", RADII)
def test_dummy_horizontal_radius_reaches_each_shape_s_farthest_point_from_the_root(region, want):
    assert g.dummy_horizontal_radius(_rooted(A=region)) == pytest.approx(want)


def test_dummy_horizontal_radius_is_the_farthest_region():
    assert g.dummy_horizontal_radius(_rooted(a=SPHERE, b=Z_CAPSULE, c=Y_CAPSULE)) == pytest.approx(RADII[4][1])


def test_a_mirrored_region_is_measured_on_its_far_side_too():
    spec = _rooted(A=dict(Z_CAPSULE, mirror_y=True))
    spec["regions"]["ROOT"]["center"] = [0.0, 0.05, 1.4]            # a root off the mid-plane
    assert g.dummy_horizontal_radius(spec) == pytest.approx(0.2 + 0.05 + 0.04)       # the -y twin is the far one
    spec["regions"]["A"]["mirror_y"] = False
    assert g.dummy_horizontal_radius(spec) == pytest.approx(0.2 - 0.05 + 0.04)


def _lateral_surface(spec, yaw_deg):
    """|y| of the farthest surface point once the body is turned yaw_deg about its root's vertical axis
    (boxes by their corners, upright capsules and spheres by their centre plus radius)."""
    regions = spec["regions"]
    rx, ry = next(r["center"][:2] for r in regions.values() if r.get("parent") is None)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    out = 0.0
    for r in regions.values():
        for side in ((1, -1) if r.get("mirror_y") else (1,)):
            x, y = r["center"][0] - rx, side * r["center"][1] - ry
            if r["shape"] == "box":
                hx, hy = r["size"][0] / 2, r["size"][1] / 2
                out = max(out, *(abs((x + ex * hx) * s + (y + ey * hy) * c) for ex in (-1, 1) for ey in (-1, 1)))
            else:
                assert r["shape"] == "sphere" or r.get("axis", "z") == "z"
                out = max(out, abs(x * s + y * c) + r["radius"])
    return out


def test_no_turn_puts_a_surface_beyond_the_horizontal_radius():
    # The property the guard needs: whichever way the dummy is turned, its |y| surface stays inside.
    spec = _rooted(A=BOX)
    radius = g.dummy_horizontal_radius(spec)
    widths = [_lateral_surface(spec, yaw) for yaw in np.arange(-180.0, 180.0, 0.25)]
    assert max(widths) <= radius + 1e-12
    assert max(widths) == pytest.approx(radius, rel=1e-4)      # and some turn does reach it: not a loose bound
    assert g.dummy_half_width(spec) < radius                   # the unturned width alone would let that turn through


def test_the_real_dummy_stays_inside_its_radius_at_every_turn():
    import yaml

    spec = yaml.safe_load((REPO / "generation/dummy/anthropometry.yaml").read_text(encoding="utf-8"))
    radius = g.dummy_horizontal_radius(spec)
    widths = [_lateral_surface(spec, yaw) for yaw in np.arange(-180.0, 180.0, 0.05)]
    assert max(widths) <= radius + 1e-12 and max(widths) == pytest.approx(radius, rel=1e-5)
    assert _lateral_surface(spec, 0.0) == pytest.approx(g.dummy_half_width(spec))
    assert radius > g.dummy_half_width(spec) > 0


def test_dummy_horizontal_radius_refuses_a_shape_or_axis_it_does_not_know():
    with pytest.raises(ValueError, match="cone"):
        g.dummy_horizontal_radius(_rooted(A={"shape": "cone", "radius": 0.1, "center": [0.0, 0.0, 1.0]}))
    with pytest.raises(ValueError, match="axis"):
        g.dummy_horizontal_radius(_rooted(A=dict(Z_CAPSULE, axis="w")))
    with pytest.raises(KeyError):          # not a silent 0
        g.dummy_horizontal_radius(_rooted(A={"shape": "sphere", "center": [0.0, 0.3, 1.0]}))


def test_dummy_horizontal_radius_needs_exactly_one_root():
    with pytest.raises(ValueError, match="parent"):
        g.dummy_horizontal_radius({"regions": {"A": dict(BOX, parent="B"), "B": dict(SPHERE, parent="A")}})
    with pytest.raises(ValueError, match="parent"):
        g.dummy_horizontal_radius({"regions": {"A": dict(BOX, parent=None), "B": dict(SPHERE, parent=None)}})


# -- the height the dummy stands at: the hold servo carries it a little above this --
# From the root region's centre down to the lowest point of any surface.
LOWEST = [(BOX, 1.0 - 0.05), (Z_CAPSULE, 1.0 - 0.25 - 0.04), (Y_CAPSULE, 1.0 - 0.03), (X_CAPSULE, 1.0 - 0.03),
          (SPHERE, 1.0 - 0.02)]


@pytest.mark.parametrize("region,lowest", LOWEST)
def test_standing_root_height_is_the_root_above_the_lowest_surface(region, lowest):
    # _rooted puts the root (a 1 mm sphere) at z 1.4
    assert g.standing_root_height(_rooted(A=region)) == pytest.approx(1.4 - lowest)


def test_standing_root_height_takes_the_lowest_of_all_regions():
    assert g.standing_root_height(_rooted(a=SPHERE, b=Z_CAPSULE, c=BOX)) == pytest.approx(1.4 - (1.0 - 0.25 - 0.04))
    # the root's own surface counts: alone, it stands on itself
    assert g.standing_root_height(_rooted()) == pytest.approx(0.001)


def test_the_real_dummy_stands_with_its_root_where_the_spec_puts_it():
    import yaml

    spec = yaml.safe_load((REPO / "generation/dummy/anthropometry.yaml").read_text(encoding="utf-8"))
    root = next(r for r in spec["regions"].values() if r.get("parent") is None)
    height = g.standing_root_height(spec)
    assert 0.5 * spec["stature"] < height <= root["center"][2]        # between the hips and its authored height
    lowest = root["center"][2] - height
    assert 0.0 <= lowest < 0.1                                         # the spec's feet end close to its floor


def test_standing_root_height_refuses_what_the_radius_refuses():
    with pytest.raises(ValueError, match="cone"):
        g.standing_root_height(_rooted(A={"shape": "cone", "radius": 0.1, "center": [0.0, 0.0, 1.0]}))
    with pytest.raises(ValueError, match="axis"):
        g.standing_root_height(_rooted(A=dict(Z_CAPSULE, axis="w")))
    with pytest.raises(ValueError, match="parent"):
        g.standing_root_height({"regions": {"A": dict(BOX, parent=None), "B": dict(SPHERE, parent=None)}})


def test_check_rejects_a_rack_inside_the_required_width():
    with pytest.raises(ValueError, match="aisle"):
        g.check(rack_inner_y=4.0, wait_y=3.7, dummy_half_width=0.3, margin=1.0)


def test_check_accepts_a_rack_at_or_beyond_the_required_width():
    g.check(rack_inner_y=5.0, wait_y=3.7, dummy_half_width=0.3, margin=1.0)   # 3.7 + 0.3 + 1.0 = 5.0
    g.check(rack_inner_y=6.0, wait_y=3.7, dummy_half_width=0.3, margin=1.0)


def _env(wait_y, origin_y):
    # What reset_dummy_entry leaves behind: target_xy in the world frame, env origins added.
    return SimpleNamespace(s1a_entry={"target_xy": np.array([[12.0, origin_y + wait_y]])},
                           scene=SimpleNamespace(env_origins=np.array([[10.0, origin_y, 0.0]])))


def test_check_after_reset_measures_the_offset_from_the_env_origin():
    env = _env(wait_y=3.7, origin_y=50.0)   # read without the origin it would be 53.7 and always refuse
    g.check_after_reset(env, np.array([0]), rack_inner_y=5.1, margin=1.0, dummy_half_width=0.3)
    with pytest.raises(ValueError, match="aisle"):
        g.check_after_reset(env, np.array([0]), rack_inner_y=4.9, margin=1.0, dummy_half_width=0.3)


def test_check_after_reset_guards_the_negative_side_too():
    with pytest.raises(ValueError, match="aisle"):
        g.check_after_reset(_env(wait_y=-3.7, origin_y=0.0), np.array([0]),
                            rack_inner_y=4.9, margin=1.0, dummy_half_width=0.3)


def test_the_spec_carries_its_guard_margin():
    spec = layout.load_spec(REPO / "generation/scene/logistics_aisle.yaml")
    assert spec["aisle"]["guard_margin_m"] > 0
