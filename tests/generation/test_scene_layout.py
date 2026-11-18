"""generation/scene/layout.py — 씬 명세(yaml) → 박스 목록이 명세가 말한 대로 놓이는가.

순수 numpy 라 CI 에서 돈다. USD 는 보지 않는다 — 산출물 검사는
generation/scene/check_scene_build.py (Isaac python) 의 몫이다.
"""

import copy
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from generation.scene import layout

SPEC = Path(__file__).resolve().parents[2] / "generation/scene/logistics_aisle.yaml"


@pytest.fixture(scope="module")
def spec():
    return layout.load_spec(SPEC)


@pytest.fixture(scope="module")
def boxes(spec):
    return layout.build_layout(spec)


def test_rack_inner_face_is_the_aisle_half_width(spec):
    assert layout.rack_inner_face_y(spec) == pytest.approx(spec["aisle"]["half_width"])


def test_every_rack_and_box_lies_outside_the_aisle(spec, boxes):
    # 랙 안쪽 면이 통로 반폭 안으로 들어오면 더미가 랙을 뚫고 나온다.
    half = spec["aisle"]["half_width"]
    racky = [b for b in boxes if b.semantic == "rack"]
    assert racky
    for b in racky:
        inner = abs(b.center[1]) - b.size[1] / 2
        assert inner >= half - 1e-9, b.name


def test_the_front_row_touches_the_aisle_exactly(spec, boxes):
    # 통로 폭이 명세보다 넓게 지어지면 「유도값 10 m」가 거짓이 된다.
    inner = min(abs(b.center[1]) - b.size[1] / 2 for b in boxes if b.name.startswith("rack/upright"))
    assert inner == pytest.approx(spec["aisle"]["half_width"])


def test_rows_cover_the_robot_run(spec, boxes):
    xs = [(b.center[0] - b.size[0] / 2, b.center[0] + b.size[0] / 2)
          for b in boxes if b.name.startswith("rack/upright")]
    assert min(x0 for x0, _ in xs) <= spec["aisle"]["x_min"]
    assert max(x1 for _, x1 in xs) >= spec["aisle"]["x_max"]


def test_both_sides_and_every_row_are_built(spec, boxes):
    tags = {b.name.split("/")[2] for b in boxes if b.name.startswith("rack/upright")}
    rows = spec["rack"]["rows_per_side"]
    assert tags == {f"s{s}r{r}" for s in "pn" for r in range(rows)}


def test_layout_is_deterministic(spec):
    a = layout.build_layout(spec)
    b = layout.build_layout(spec)
    assert [(x.name, x.center, x.size) for x in a] == [(x.name, x.center, x.size) for x in b]


def test_a_different_seed_moves_the_boxes(spec):
    other = dict(spec, boxes=dict(spec["boxes"], seed=spec["boxes"]["seed"] + 1))
    a = [b for b in layout.build_layout(spec) if b.name.startswith("box/")]
    b = [b for b in layout.build_layout(other) if b.name.startswith("box/")]
    assert a and [(x.name, x.size) for x in a] != [(x.name, x.size) for x in b]


def test_boxes_sit_on_shelves_and_stay_under_the_next_beam(spec, boxes):
    r = spec["rack"]
    shelves = {b.name.split("/", 2)[2]: b for b in boxes if b.name.startswith("rack/shelf/")}
    box_list = [b for b in boxes if b.name.startswith("box/")]
    assert box_list, "no boxes generated"
    heights = sorted(r["beam_heights"])
    for b in box_list:
        shelf_key = "/".join(b.name.split("/")[1:3])   # box/<row tag>/<bay,level>/s<j>l<L>
        shelf = shelves[shelf_key]
        shelf_top = shelf.center[2] + shelf.size[2] / 2
        bottom = b.center[2] - b.size[2] / 2
        top = b.center[2] + b.size[2] / 2
        assert bottom >= shelf_top - 1e-9, b.name
        assert b.semantic == "rack" and b.collider, b.name
        # under the next beam (or the top clearance for the top shelf)
        h = shelf.center[2] - r["beam_size"] - r["shelf_thickness"] / 2
        above = [x for x in heights if x > h + 1e-6]
        ceiling = above[0] if above else h + spec["boxes"]["top_shelf_clearance"]
        assert top <= ceiling + 1e-9, b.name
        # inside the bay in x and inside the shelf in y
        assert abs(b.center[0] - shelf.center[0]) + b.size[0] / 2 <= shelf.size[0] / 2 + 1e-9, b.name
        assert abs(b.center[1] - shelf.center[1]) + b.size[1] / 2 <= shelf.size[1] / 2 + 1e-9, b.name


def test_boxes_on_one_shelf_do_not_overlap(boxes):
    by_shelf: dict[str, list] = {}
    for b in boxes:
        if b.name.startswith("box/"):
            by_shelf.setdefault("/".join(b.name.split("/")[1:3]), []).append(b)
    for group in by_shelf.values():
        for a, b in combinations(group, 2):
            gap = [abs(a.center[i] - b.center[i]) - (a.size[i] + b.size[i]) / 2 for i in range(3)]
            assert max(gap) >= -1e-9, (a.name, b.name)


def test_floor_and_ceiling_have_no_collider_and_no_class(boxes):
    # 지면 물리는 Isaac Lab 의 평면이 맡는다. 겹치면 떨린다.
    named = {b.name: b for b in boxes}
    for n in ("floor", "ceiling"):
        assert n in named
        assert named[n].collider is False and named[n].semantic is None


def test_every_material_the_layout_uses_is_declared(spec, boxes):
    used = {b.material for b in boxes if b.material != "light"}
    assert used <= set(spec["materials"]), sorted(used - set(spec["materials"]))


# ---- an enclosed warehouse — walls, dock doors, roof trusses, fixtures, floor lines ----
# Everything added is dressing: it must not collide (the physics stays what the control runs measured),
# and the assertions below are about how the pieces meet each other, read off the built boxes.

def _aabb(b):
    """(lo, hi) corners of a box; one turned about x (rot_x_deg) spans its turned extent."""
    import math
    th = math.radians(b.rot_x_deg)
    c, s = abs(math.cos(th)), abs(math.sin(th))
    h = (b.size[0] / 2, c * b.size[1] / 2 + s * b.size[2] / 2, s * b.size[1] / 2 + c * b.size[2] / 2)
    return tuple(b.center[i] - h[i] for i in range(3)), tuple(b.center[i] + h[i] for i in range(3))


def _named(boxes, name):
    return next(b for b in boxes if b.name == name)


def _floor_edges(boxes):
    """x0, x1, y0, y1 of the floor slab and the height of its top face."""
    lo, hi = _aabb(_named(boxes, "floor"))
    return lo[0], hi[0], lo[1], hi[1], hi[2]


def _rack_top(boxes):
    return max(_aabb(b)[1][2] for b in boxes if b.semantic == "rack")


def _aisles(boxes):
    """(lo, hi) in y of every strip of floor no rack row stands on: the main aisle, the gaps
    between rows and the strips along the side walls -- from the racks the layout built."""
    _, _, y0, y1, _ = _floor_edges(boxes)
    rows: dict[str, list[float]] = {}
    for b in boxes:
        if b.name.startswith("rack/"):
            lo, hi = _aabb(b)
            span = rows.setdefault(b.name.split("/")[2], [lo[1], hi[1]])
            span[0], span[1] = min(span[0], lo[1]), max(span[1], hi[1])
    edges = [y0] + [v for span in sorted(rows.values()) for v in span] + [y1]
    return [(a, b) for a, b in zip(edges[0::2], edges[1::2]) if b - a > 1e-9]


def test_only_racks_and_their_boxes_collide_or_carry_a_class(boxes):
    # 벽·지붕·등기구·바닥 선은 보이기만 한다. 콜라이더가 하나라도 생기면 물리가 대조군과 갈라진다.
    assert {b.name.split("/")[0] for b in boxes if b.collider} == {"rack", "box"}
    assert {b.name.split("/")[0] for b in boxes if b.semantic} == {"rack", "box"}


def test_the_dressing_comes_after_every_collider(boxes):
    # Colliders keep the prim order (and paths) the control runs were measured with.
    last_collider = max(i for i, b in enumerate(boxes) if b.collider)
    first_dressing = min(i for i, b in enumerate(boxes) if b.name.split("/")[0] in
                         ("wall", "door", "roof", "fixture", "light", "marking"))
    assert first_dressing > last_collider


def test_box_and_prim_names_are_unique(boxes):
    names = [b.name for b in boxes]
    assert len(names) == len(set(names))
    assert len({n.replace("/", "_") for n in names}) == len(names)   # the builder's prim names


def test_walls_close_every_floor_edge_from_the_floor_to_the_ceiling(boxes):
    x0, x1, y0, y1, _ = _floor_edges(boxes)
    ceiling_top = _aabb(_named(boxes, "ceiling"))[1][2]
    walls = [(b, *_aabb(b)) for b in boxes if b.name.startswith("wall/")]
    for axis, edge, outward in ((0, x0, -1), (0, x1, 1), (1, y0, -1), (1, y1, 1)):
        on = [(lo, hi) for _, lo, hi in walls if abs((lo[axis] if outward > 0 else hi[axis]) - edge) < 1e-9]
        assert on, (axis, edge)
        across = 1 - axis
        span = (x0, x1) if across == 0 else (y0, y1)
        for lo, hi in on:
            assert lo[across] <= span[0] + 1e-9 and hi[across] >= span[1] - 1e-9, (axis, edge)
        zs = sorted((lo[2], hi[2]) for lo, hi in on)
        assert zs[0][0] <= 1e-9, (axis, edge, "starts above the floor")
        reach = zs[0][1]
        for z_lo, z_hi in zs[1:]:
            assert z_lo <= reach + 1e-9, (axis, edge, "a gap between wall bands")
            reach = max(reach, z_hi)
        assert reach >= ceiling_top - 1e-9, (axis, edge, "stops below the ceiling")
    # corners: the end walls run across the side walls' ends
    side_y = [(lo[1], hi[1]) for _, lo, hi in walls if lo[1] >= y1 - 1e-9 or hi[1] <= y0 + 1e-9]
    for _, lo, hi in walls:
        if hi[0] <= x0 + 1e-9 or lo[0] >= x1 - 1e-9:
            assert lo[1] <= min(a for a, _ in side_y) + 1e-9 and hi[1] >= max(b for _, b in side_y) - 1e-9


def test_dock_doors_stand_against_both_end_walls(spec, boxes):
    d = spec["walls"]["dock_doors"]
    x0, x1, y0, y1, _ = _floor_edges(boxes)
    doors = [(b, *_aabb(b)) for b in boxes if b.name.startswith("door/")]
    assert len(doors) == 2 * d["per_end_wall"]
    for end, face in ((x0, 0), (x1, 1)):
        on = sorted((b.center[1], lo, hi) for b, lo, hi in doors
                    if abs((lo[0] if face == 0 else hi[0]) - end) < 1e-9)
        assert len(on) == d["per_end_wall"], end
        for _, lo, hi in on:
            assert lo[2] <= 1e-9 and hi[2] == pytest.approx(d["height"])
            assert hi[1] - lo[1] == pytest.approx(d["width"])
            assert lo[1] >= y0 - 1e-9 and hi[1] <= y1 + 1e-9
        ys = [y for y, _, _ in on]
        assert ys == pytest.approx([-y for y in reversed(ys)])          # centred on the main aisle
        assert all(b - a == pytest.approx(d["width"] + d["gap"]) for a, b in zip(ys, ys[1:]))


def test_the_roof_structure_sits_between_the_racks_and_the_ceiling(boxes):
    underside = _aabb(_named(boxes, "ceiling"))[0][2]
    top = _rack_top(boxes)
    roof = [b for b in boxes if b.name.startswith("roof/")]
    assert roof
    for b in roof:
        lo, hi = _aabb(b)
        assert lo[2] > top and hi[2] <= underside + 1e-9, b.name


def test_trusses_cross_the_building_at_an_even_pitch(spec, boxes):
    x0, x1, y0, y1, _ = _floor_edges(boxes)
    chords = [b for b in boxes if b.name.startswith("roof/truss") and b.name.endswith(("/top", "/bottom"))]
    for b in chords:
        lo, hi = _aabb(b)
        assert lo[1] <= y0 + 1e-9 and hi[1] >= y1 - 1e-9, b.name      # wall to wall
    xs = sorted({round(b.center[0], 9) for b in chords})
    assert xs
    pitches = np.diff([x0, *xs, x1])
    assert np.allclose(pitches, pitches[0])
    target = spec["roof"]["truss_spacing_m"]
    assert abs(pitches[0] - target) <= target / 2
    for x in xs:
        at = {b.name.rsplit("/", 1)[1]: b for b in chords if abs(b.center[0] - x) < 1e-9}
        assert set(at) == {"top", "bottom"}
        assert at["top"].center[2] - at["bottom"].center[2] == pytest.approx(spec["roof"]["truss_depth"])


def test_web_members_zigzag_between_the_chords_from_wall_to_wall(boxes):
    import math
    _, _, y0, y1, _ = _floor_edges(boxes)
    tops = [b for b in boxes if b.name.startswith("roof/truss") and b.name.endswith("/top")]
    assert tops
    for top in tops:
        truss = top.name.rsplit("/", 1)[0]
        bottom = _named(boxes, truss + "/bottom")
        webs = sorted((b for b in boxes if b.name.startswith(truss + "/web")), key=lambda b: b.center[1])
        assert len(webs) >= 2, truss
        ends = []
        for w in webs:
            assert w.center[0] == pytest.approx(top.center[0]), w.name
            th, half = math.radians(w.rot_x_deg), w.size[1] / 2
            a = (w.center[1] - half * math.cos(th), w.center[2] - half * math.sin(th))
            b = (w.center[1] + half * math.cos(th), w.center[2] + half * math.sin(th))
            assert sorted([a[1], b[1]]) == pytest.approx([bottom.center[2], top.center[2]]), w.name
            ends.append(sorted([a, b]))
        for (_, far), (near, _) in zip(ends, ends[1:]):
            assert far == pytest.approx(near), truss                    # one continuous zig-zag
        assert ends[0][0][0] == pytest.approx(y0) and ends[-1][1][0] == pytest.approx(y1)


def test_every_aisle_has_a_row_of_fixtures_in_the_bays_between_trusses(boxes):
    fixtures = [b for b in boxes if b.name.startswith("fixture/")]
    assert fixtures
    rows = sorted({round(b.center[1], 9) for b in fixtures})
    assert rows == pytest.approx(sorted((a + b) / 2 for a, b in _aisles(boxes)))
    bottoms = [b for b in boxes if b.name.startswith("roof/truss") and b.name.endswith("/bottom")]
    truss_bottom = min(_aabb(b)[0][2] for b in bottoms)
    top = _rack_top(boxes)
    per_row: dict[float, list[float]] = {}
    for f in fixtures:
        lo, hi = _aabb(f)
        assert lo[2] > top and hi[2] < truss_bottom, f.name              # over the boxes, under the trusses
        for c in bottoms:
            clo, chi = _aabb(c)
            assert hi[0] <= clo[0] or lo[0] >= chi[0], (f.name, c.name)  # never under a truss
        per_row.setdefault(round(f.center[1], 9), []).append(round(f.center[0], 9))
    assert len({tuple(sorted(xs)) for xs in per_row.values()}) == 1      # every row, the same bays
    xs = sorted(next(iter(per_row.values())))
    truss_xs = sorted({round(b.center[0], 9) for b in bottoms})
    assert len(xs) == len(truss_xs) + 1                                  # one per bay


def test_each_fixture_has_its_light_just_below_it(boxes):
    fixtures = {b.name.split("/", 1)[1]: b for b in boxes if b.name.startswith("fixture/")}
    lights = {b.name.split("/", 1)[1]: b for b in boxes if b.name.startswith("light/")}
    assert fixtures and set(fixtures) == set(lights)
    for key, f in fixtures.items():
        light = lights[key]
        assert light.material == "light" and not light.collider
        assert light.center[:2] == pytest.approx(f.center[:2])
        assert light.size[:2] == pytest.approx(f.size[:2])               # the fixture's lower face
        gap = _aabb(f)[0][2] - light.center[2]
        assert 0 < gap < 0.05, key                                        # below it: the fixture cannot shade it


def test_the_dressing_takes_its_dimensions_from_the_spec(spec, boxes):
    # The relations above held with every purlin gone, or with the walls
    # built to other numbers than the spec's. Each knob is read back off the boxes here.
    w, roof, lg, d = spec["walls"], spec["roof"], spec["lights"], spec["walls"]["dock_doors"]
    x0, x1, y0, y1, _ = _floor_edges(boxes)
    for tag, axis in (("x0", 0), ("x1", 0), ("y0", 1), ("y1", 1)):
        base, upper = _named(boxes, f"wall/{tag}/base"), _named(boxes, f"wall/{tag}/upper")
        assert base.size[axis] == pytest.approx(w["thickness"]) and upper.size[axis] == pytest.approx(w["thickness"])
        assert _aabb(base)[1][2] == pytest.approx(w["base_height"]) and _aabb(upper)[0][2] == pytest.approx(w["base_height"])
    for b in boxes:
        if b.name.startswith("door/"):
            assert b.size[0] == pytest.approx(d["thickness"]), b.name
        elif b.name.startswith("fixture/"):
            assert b.size == pytest.approx(tuple(lg["fixture_size"])), b.name
        elif b.name.startswith("roof/truss") and b.name.endswith(("/top", "/bottom")):
            assert (b.size[0], b.size[2]) == pytest.approx((roof["chord_size"],) * 2), b.name
        elif b.name.startswith("roof/truss"):
            assert (b.size[0], b.size[2]) == pytest.approx((roof["web_size"],) * 2), b.name
    webs = sorted(b.center[1] for b in boxes if b.name.startswith("roof/truss0/web"))
    panel = np.diff(webs)
    assert np.allclose(panel, panel[0]) and abs(panel[0] - roof["panel_m"]) <= roof["panel_m"] / 2
    purlins = [b for b in boxes if b.name.startswith("roof/purlin")]
    assert purlins
    chord_top = max(_aabb(b)[1][2] for b in boxes if b.name.startswith("roof/truss") and b.name.endswith("/top"))
    underside = _aabb(_named(boxes, "ceiling"))[0][2]
    for b in purlins:
        lo, hi = _aabb(b)
        assert lo[0] == pytest.approx(x0) and hi[0] == pytest.approx(x1), b.name              # wall to wall
        assert (b.size[1], b.size[2]) == pytest.approx(tuple(roof["purlin_size"])), b.name
        assert lo[2] == pytest.approx(chord_top) and hi[2] == pytest.approx(underside), b.name  # on the chords
    ys = sorted(b.center[1] for b in purlins)
    pitch = np.diff([y0, *ys, y1])[1:-1] if len(ys) > 1 else np.array([y1 - y0])
    assert np.allclose(pitch, pitch[0]) and abs(pitch[0] - roof["purlin_spacing_m"]) <= roof["purlin_spacing_m"] / 2
    assert ys[0] - y0 == pytest.approx(pitch[0] / 2) and y1 - ys[-1] == pytest.approx(pitch[0] / 2)


def test_each_bay_holds_one_fixture_per_row_in_its_middle(boxes):
    # Seven fixtures stacked in the first bay passed the per-row checks.
    x0, x1, _, _, _ = _floor_edges(boxes)
    edges = [x0, *sorted({round(b.center[0], 9) for b in boxes if b.name.startswith("roof/truss")}), x1]
    rows: dict[float, list[float]] = {}
    for f in boxes:
        if f.name.startswith("fixture/"):
            rows.setdefault(round(f.center[1], 9), []).append(f.center[0])
    assert rows and len(edges) >= 3
    for y, xs in rows.items():
        for lo, hi in zip(edges, edges[1:]):
            inside = [x for x in xs if lo < x < hi]
            assert len(inside) == 1, (y, lo, hi, inside)
            assert inside[0] == pytest.approx((lo + hi) / 2), (y, inside)


def test_the_racks_stand_inside_the_walls_and_everything_inside_the_hall(spec, boxes):
    # With floor.margin 0 the end walls cut through the racks' ends.
    x0, x1, y0, y1, _ = _floor_edges(boxes)
    t = spec["walls"]["thickness"]
    top = _aabb(_named(boxes, "ceiling"))[1][2]
    for b in boxes:
        lo, hi = _aabb(b)
        if b.collider:
            assert x0 - 1e-9 <= lo[0] and hi[0] <= x1 + 1e-9 and y0 - 1e-9 <= lo[1] and hi[1] <= y1 + 1e-9, b.name
        assert x0 - t - 1e-9 <= lo[0] and hi[0] <= x1 + t + 1e-9, b.name
        assert y0 - t - 1e-9 <= lo[1] and hi[1] <= y1 + t + 1e-9, b.name
        assert -1e-9 <= lo[2] and hi[2] <= top + 1e-9, b.name


@pytest.mark.parametrize("bad,why", [
    (layout.Box("wall/x/stray", (60.0, 0.0, 1.0), (1.0, 1.0, 1.0), "wall_cladding", False, None), "outside the hall"),
    (layout.Box("rack/upright/x", (34.0, 8.0, 1.0), (0.1, 0.1, 2.0), "steel_blue", True, "rack"), "floor.margin"),
    (layout.Box("roof/truss9/top", (0.0, 0.0, 5.0), (0.15, 1.0, 0.15), "roof_steel", False, None), "roof member"),
    (layout.Box("fixture/r9/0", (0.0, 0.0, 7.0), (1.2, 0.4, 0.1), "fixture_lens", False, None), "not between"),
])
def test_validate_layout_refuses_a_box_outside_its_space(spec, boxes, bad, why):
    with pytest.raises(ValueError, match=why):
        layout.validate_layout(spec, boxes + [bad])


def test_yellow_lines_edge_the_main_aisle_on_the_floor(spec, boxes):
    m = spec["floor_markings"]
    half = spec["aisle"]["half_width"]
    x0, x1, _, _, floor_top = _floor_edges(boxes)
    marks = [b for b in boxes if b.name.startswith("marking/")]
    assert len(marks) == 2
    assert sorted(b.center[1] for b in marks)[0] == pytest.approx(-sorted(b.center[1] for b in marks)[1])
    for b in marks:
        lo, hi = _aabb(b)
        assert hi[1] - lo[1] == pytest.approx(m["width"])
        assert abs(b.center[1]) + m["width"] / 2 == pytest.approx(half - m["inset"])
        assert lo[0] == pytest.approx(x0) and hi[0] == pytest.approx(x1)
        assert lo[2] == pytest.approx(floor_top + m["lift"])                # on the floor, not in it
        assert hi[2] == pytest.approx(floor_top + m["lift"] + m["thickness"])


def test_load_spec_rejects_a_spec_without_an_aisle(tmp_path):
    bad = tmp_path / "s.yaml"
    bad.write_text("scene_version: '0.1'\nfloor: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="aisle"):
        layout.load_spec(bad)


# ---- the loader must refuse specs that build invalid colliders ----

def _mutated(spec, path, value):
    import copy
    s = copy.deepcopy(spec)
    node = s
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return s


@pytest.mark.parametrize("path,value,why", [
    (("aisle", "half_width"), float("nan"), "finite"),
    (("aisle", "half_width"), float("inf"), "finite"),
    (("aisle", "guard_margin_m"), 0.0, "guard_margin_m"),      # the aisle guard's margin must be a real margin
    (("aisle", "guard_margin_m"), -1.0, "guard_margin_m"),
    (("rack", "depth"), 0.05, "depth"),                       # thinner than two uprights: faces cross, sizes go negative
    (("rack", "beam_heights"), [1.5, 1.6, 4.5], "beam_heights"),  # 0.10 apart < beam + shelf: beams overlap
    (("rack", "beam_heights"), [1.5, 1.5, 4.5], "beam_heights"),
    (("rack", "beam_heights"), [-0.5, 1.5, 4.5], "beam_heights"),
    (("rack", "bay_width"), 0.05, "bay_width"),               # narrower than an upright
    (("rack", "rows_per_side"), 0, "rows_per_side"),
    (("boxes", "size_min"), [0.7, 0.35, 0.30], "size_min"),   # min above max
    (("boxes", "size_max"), [0.6, 0.6, -0.1], "positive"),
    (("boxes", "stacks_per_bay"), [3, 1], "stacks_per_bay"),
    (("boxes", "layers"), [0, 3], "layers"),
    (("boxes", "edge_margin"), 2.0, "edge_margin"),           # no usable width left in a 2.7 m bay
    (("boxes", "top_shelf_clearance"), 0.0, "top_shelf_clearance"),
    (("ceiling", "z"), 3.0, "ceiling"),                       # below the top beam
    (("materials", "cardboard", "tile_m"), 0.0, "tile_m"),
    (("materials", "cardboard", "texture"), "", "texture"),
    (("materials", "steel_orange", "tint"), [1.0, 0.5], "tint"),
    # the enclosure
    (("walls", "thickness"), 0.0, "walls.thickness"),
    (("walls", "base_height"), 9.0, "walls.base_height"),     # above the ceiling
    (("walls", "base_height"), -0.1, "walls.base_height"),
    (("walls", "dock_doors", "height"), 8.5, "dock_doors.height"),   # taller than the wall
    (("walls", "dock_doors", "width"), 20.0, "dock_doors"),   # two 20 m doors do not fit a 39 m end wall
    (("walls", "dock_doors", "per_end_wall"), -1, "per_end_wall"),
    (("walls", "dock_doors", "gap"), -0.5, "dock_doors.gap"),
    (("roof", "truss_spacing_m"), 0.0, "truss_spacing_m"),
    (("roof", "truss_spacing_m"), 1.0, "fixture_size"),       # 1 m bays: a 1.2 m fixture runs into a truss
    (("roof", "truss_depth"), 0.1, "truss_depth"),            # no deeper than a chord
    (("roof", "truss_depth"), 3.0, "roof"),                   # the bottom chord comes down into the fixtures
    (("roof", "panel_m"), 0.0, "panel_m"),
    (("roof", "purlin_size"), [0.1], "purlin_size"),
    (("lights", "z"), 5.0, "lights.z"),                       # fixtures among the boxes on the top shelf
    (("lights", "fixture_size"), [1.2, 0.4], "fixture_size"),
    (("lights", "intensity"), 0.0, "lights.intensity"),
    (("lights", "color"), [1.0, -0.1, 1.0], "lights.color"),
    (("lights", "fill", "intensity"), -1.0, "fill"),
    (("floor_markings", "width"), 0.0, "floor_markings.width"),
    (("floor_markings", "inset"), 5.0, "floor_markings"),     # the line would leave the aisle
    (("floor_markings", "lift"), -0.001, "floor_markings.lift"),
    (("materials", "marking_yellow", "color"), [1.0, 0.5], "color"),
    (("materials", "roof_steel", "roughness"), 1.5, "roughness"),
    (("materials", "fixture_lens", "emissive_intensity"), 0.0, "emissive_intensity"),
    (("materials", "roof_steel", "texture"), "corrugated_iron_02", "exactly one"),   # both texture and color
    (("materials", "roof_steel", "shine"), 1.0, "unknown"),
    # the enclosure: pieces that would not fit one another
    (("roof", "truss_spacing_m"), 40.0, "one bay"),           # 44 m in one bay: no truss at all
    (("roof", "panel_m"), 1e-320, "panel_m"),                 # finite and positive, but no member fits in it
    (("roof", "panel_m"), 0.05, "panel_m"),                   # narrower than a web member
    (("roof", "purlin_spacing_m"), 0.05, "purlin_spacing_m"),  # purlins would overlap
    (("roof", "web_size"), 2.0, "web_size"),                  # thicker than the chords: pokes through the ceiling
    (("roof", "web_size"), 0.2, "must not exceed roof.chord_size"),   # only this rule sees it: panels still wider
    (("walls", "dock_doors", "thickness"), 50.0, "dock_doors.thickness"),   # across the whole hall
    (("walls", "dock_doors", "width"), 5.0, "dock_doors"),    # 2 x 5 m + 1 m does not face the 10 m main aisle
    (("lights", "fixture_size"), [1.2, 50.0, 0.1], "fixture_size"),   # through the side walls
    (("lights", "fixture_size"), [1.2, 3.0, 0.1], "fixture_size"),    # wider than the 2.5 m gap between rows
    (("floor", "z"), -1.0, "floor.z"),                        # under the ground plane the walls stand on
    (("floor", "margin"), 0.0, "floor.margin"),               # the end walls cut through the racks' ends
    # the bottom tier's loads
    (("rack", "floor_loads", "per_bay"), 3, "rack.floor_loads"),               # 3 x 1.2 + 2 x 0.1 > the bay's 2.61 m
    (("rack", "floor_loads", "per_bay"), -1, "rack.floor_loads"),
    (("rack", "floor_loads", "per_bay"), 1.5, "rack.floor_loads"),
    (("rack", "floor_loads", "pallet", "size"), [1.2, 1.1, 0.15], "rack.floor_loads"),   # deeper than the 1.0 m row
    (("rack", "floor_loads", "carton", "size"), [1.2, 0.9, 1.4], "rack.floor_loads"),    # 0.15 + 1.4 touches the first beam
    (("rack", "floor_loads", "carton", "size"), [1.2, 0.9, 1.3], "rack.floor_loads"),    # clears the beam by 0.05 m, not the 0.15 m of air
    (("rack", "floor_loads", "carton", "size"), [1.2, 0.9, -1.2], "rack.floor_loads"),
    (("rack", "floor_loads", "gap_m"), 0.3, "rack.floor_loads"),                # 2 x 1.2 + 0.3 > 2.61: only the gap overflows the bay
    (("rack", "floor_loads", "pallet", "size"), [1.2, 0.9], "rack.floor_loads"),
    (("rack", "floor_loads", "gap_m"), -0.1, "rack.floor_loads"),
    (("rack", "floor_loads", "carton", "material"), "pallet_blue", "rack.floor_loads"),   # a name 'materials' does not declare
])
def test_validate_spec_rejects_geometry_that_cannot_be_built(spec, path, value, why):
    with pytest.raises(ValueError, match=why):
        layout.validate_spec(_mutated(spec, path, value))


def test_validate_spec_accepts_the_canon(spec):
    layout.validate_spec(spec)   # must not raise


def test_rack_top_counts_the_top_shelf_board(spec):
    # With a tiny clearance the top shelf's board is the highest point,
    # and lights.z was checked against a top 3 cm too low.
    s = _mutated(spec, ("boxes", "top_shelf_clearance"), 0.01)
    shelf_top = max(b.center[2] + b.size[2] / 2 for b in layout.build_layout(s) if b.semantic == "rack")
    assert layout.rack_top(s) == pytest.approx(shelf_top)
    with pytest.raises(ValueError, match="lights.z"):
        layout.validate_spec(_mutated(s, ("lights", "z"), 4.685))   # fixture bottom 4.635 under the board


@pytest.mark.parametrize("knob,value,material", [
    (("walls", "base_height"), 0.0, "concrete_wall"),
    (("walls", "dock_doors", "per_end_wall"), 0, "dock_door"),
])
def test_a_part_that_is_not_built_does_not_need_its_material(spec, knob, value, material):
    s = _mutated(spec, knob, value)
    del s["materials"][material]
    layout.validate_spec(s)                                    # must not raise
    assert material not in {b.material for b in layout.build_layout(s)}


@pytest.mark.parametrize("path,why", [(("walls", "dock_doors"), "dock_doors"), (("lights", "fill"), "fill"),
                                      (("roof", "chord_size"), "chord_size")])
def test_a_missing_key_is_a_value_error_naming_it(spec, path, why):
    import copy
    s = copy.deepcopy(spec)
    node = s
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    with pytest.raises(ValueError, match=why):
        layout.validate_spec(s)


def test_validate_spec_requires_the_guard_margin(spec):
    import copy
    s = copy.deepcopy(spec)
    del s["aisle"]["guard_margin_m"]
    with pytest.raises(ValueError, match="guard_margin_m"):
        layout.validate_spec(s)


def test_build_layout_refuses_a_layout_that_reaches_into_the_aisle(spec):
    """Belt and braces: even if a future rule slips past validate_spec, the
    layout itself is checked before anything reads it."""
    boxes = layout.build_layout(spec)
    bad = [layout.Box("rack/upright/x", (0.0, 4.0, 1.0), (0.1, 0.1, 2.0), "steel_blue", True, "rack")]
    with pytest.raises(ValueError, match="aisle"):
        layout.validate_layout(spec, boxes + bad)
    with pytest.raises(ValueError, match="finite"):
        layout.validate_layout(spec, boxes + [layout.Box("box/x", (float("nan"), 6.0, 1.0), (0.3, 0.3, 0.3), "cardboard", True, "rack")])
    layout.validate_layout(spec, boxes)   # the real layout passes


# ---- the bottom tier: a pallet and a large carton, two to a bay ----

def _loads(boxes):
    return [b for b in boxes if b.name.startswith(layout.FLOOR_LOAD_PREFIX)]


def test_every_bay_holds_two_floor_loads(spec, boxes):
    # the count of loads is derived from the spec's per_bay and the bays the racks build -- 14 bays x 6 rows x 2 = 168.
    loads = _loads(boxes)
    rows = 2 * spec["rack"]["rows_per_side"]
    bays = layout.rack_bays(spec)
    per_bay = spec["rack"]["floor_loads"]["per_bay"]
    assert sum(b.name.endswith("/pallet") for b in loads) == per_bay * bays * rows
    assert sum(b.name.endswith("/carton") for b in loads) == per_bay * bays * rows
    assert len(loads) == 2 * per_bay * bays * rows
    assert per_bay * bays * rows == 168     # this canon's count (14 bays, 6 rows, 2 a bay)


def test_floor_loads_sit_on_the_floor_inside_their_bay_below_the_first_beam(spec, boxes):
    r, fl = spec["rack"], spec["rack"]["floor_loads"]
    pallet_h, carton_h = fl["pallet"]["size"][2], fl["carton"]["size"][2]
    first_beam = min(r["beam_heights"])
    uprights = [b for b in boxes if b.name.startswith("rack/upright/")]
    for b in _loads(boxes):
        lo, hi = _aabb(b)
        bay = int(b.name.split("/")[3][1:])
        bay_lo = spec["aisle"]["x_min"] + bay * r["bay_width"]
        tag = b.name.split("/")[2]
        row_centre = next(s.center[1] for s in boxes if s.name.startswith(f"rack/shelf/{tag}/b{bay}l0"))
        if b.name.endswith("/pallet"):
            assert lo[2] == pytest.approx(0.0) and hi[2] == pytest.approx(pallet_h), b.name
        else:
            assert lo[2] == pytest.approx(pallet_h) and hi[2] == pytest.approx(pallet_h + carton_h), b.name
            assert hi[2] + 0.15 <= first_beam + 1e-9, b.name           # a gap under the first beam
        # inside the bay (between its uprights) and inside the row's depth, centred on it
        assert lo[0] >= bay_lo + r["upright_size"] / 2 - 1e-9 and hi[0] <= bay_lo + r["bay_width"] - r["upright_size"] / 2 + 1e-9, b.name
        assert b.center[1] == pytest.approx(row_centre), b.name
        assert hi[1] - lo[1] <= r["depth"] + 1e-9, b.name
        for u in uprights:
            ulo, uhi = _aabb(u)
            assert not all(lo[i] < uhi[i] - 1e-9 and hi[i] > ulo[i] + 1e-9 for i in range(3)), (b.name, u.name)


def test_the_two_loads_of_a_bay_are_side_by_side_with_the_gap_between_them(spec, boxes):
    fl = spec["rack"]["floor_loads"]
    by_slot: dict[str, list] = {}
    for b in _loads(boxes):
        by_slot.setdefault("/".join(b.name.split("/")[2:4]), []).append(b)       # row tag / bay
    assert by_slot
    for key, group in by_slot.items():
        pallets = sorted((b for b in group if b.name.endswith("/pallet")), key=lambda b: b.center[0])
        cartons = sorted((b for b in group if b.name.endswith("/carton")), key=lambda b: b.center[0])
        assert len(pallets) == len(cartons) == fl["per_bay"], key
        for p, c in zip(pallets, cartons):
            assert c.center[0] == pytest.approx(p.center[0]) and c.center[1] == pytest.approx(p.center[1]), key
            assert p.name.rsplit("/", 2)[1] == c.name.rsplit("/", 2)[1]                        # the same slot number
        for a, b in zip(pallets, pallets[1:]):
            assert b.center[0] - a.center[0] - fl["pallet"]["size"][0] == pytest.approx(fl["gap_m"]), key
        assert sum(p.center[0] for p in pallets) / len(pallets) == pytest.approx(
            spec["aisle"]["x_min"] + (int(key.split("/")[1][1:]) + 0.5) * spec["rack"]["bay_width"]), key


def test_floor_loads_leave_every_other_box_as_it_was(spec, boxes):
    # a change in any shelf box's draw, name or place would unsettle the comparison with S1d -- the loads draw no
    # random numbers and come after the shelf boxes.
    none = copy.deepcopy(spec)
    none["rack"]["floor_loads"]["per_bay"] = 0
    bare = layout.build_layout(none)
    assert not _loads(bare)
    assert [b for b in boxes if not b.name.startswith(layout.FLOOR_LOAD_PREFIX)] == bare


def test_floor_loads_come_after_the_boxes_and_before_the_dressing(boxes):
    names = [b.name.split("/")[0] + ("/load" if b.name.startswith(layout.FLOOR_LOAD_PREFIX) else "") for b in boxes]
    last_box = max(i for i, n in enumerate(names) if n == "box")
    loads = [i for i, n in enumerate(names) if n == "rack/load"]
    first_dressing = min(i for i, n in enumerate(names) if n in ("wall", "door", "roof", "fixture", "light", "marking"))
    assert last_box < min(loads) and max(loads) < first_dressing
    assert loads == list(range(min(loads), max(loads) + 1))                      # one block


def test_floor_loads_collide_read_as_rack_and_use_their_materials(spec, boxes):
    fl = spec["rack"]["floor_loads"]
    loads = _loads(boxes)
    assert loads
    for b in loads:
        assert b.collider and b.semantic == layout.SEMANTIC_RACK, b.name
        part = fl["pallet"] if b.name.endswith("/pallet") else fl["carton"]
        assert b.material == part["material"], b.name
        assert b.size == pytest.approx(tuple(part["size"])), b.name
    # this scene's choice: a plain grey pallet, a large carton in the shelf boxes' cardboard
    assert spec["materials"][fl["pallet"]["material"]].keys() <= layout.PLAIN_KEYS
    assert spec["materials"][fl["carton"]["material"]]["texture"] == spec["materials"][spec["boxes"]["material"]]["texture"]
    assert spec["materials"][fl["carton"]["material"]]["tile_m"] == pytest.approx(1.2)


def test_the_loads_do_not_touch_the_aisle_or_each_other(spec, boxes):
    loads = _loads(boxes)
    half = spec["aisle"]["half_width"]
    for b in loads:
        assert abs(b.center[1]) - b.size[1] / 2 >= half - 1e-9, b.name
    by_row: dict[str, list] = {}
    for b in loads:
        by_row.setdefault(b.name.split("/")[2], []).append(b)
    for group in by_row.values():
        for a, b in combinations(group, 2):
            gap = [abs(a.center[i] - b.center[i]) - (a.size[i] + b.size[i]) / 2 for i in range(3)]
            assert max(gap) >= -1e-9, (a.name, b.name)



# ---- the box labels: a white card and the KETI CI on the two y faces of every box ----

# The CI picture's width / height. The pinned PNG is not tracked (fetch_keti_ci.py fetches it), so the layout is
# asked with an aspect of the test's choosing; the builder reads the real one from the picture's header.
ASPECT = 4.0


@pytest.fixture(scope="module")
def plates(spec, boxes):
    return layout.box_labels(spec, boxes, ASPECT)


def _labelled(boxes):
    """Counted from the layout's names, not from label_kind: every box on a shelf and every bottom-tier carton."""
    shelf = [b for b in boxes if b.name.startswith("box/")]
    cartons = [b for b in boxes if b.name.startswith(layout.FLOOR_LOAD_PREFIX) and b.name.endswith("/carton")]
    return shelf, cartons


def test_every_shelf_box_and_carton_carries_a_card_and_a_ci_on_both_y_faces(boxes, plates):
    shelf, cartons = _labelled(boxes)
    assert shelf and cartons
    for kind in layout.LABEL_PLATES:
        mine = [p for p in plates if p.kind == kind]
        assert len(mine) == 2 * (len(shelf) + len(cartons)), kind
    on = {}
    for p in plates:
        on.setdefault(p.box, []).append((p.face, p.kind))
    assert set(on) == {b.name for b in shelf + cartons}
    # +y then -y, the card before its CI -- the order the builder writes the meshes in
    assert all(v == [("+y", "base"), ("+y", "ci"), ("-y", "base"), ("-y", "ci")] for v in on.values())
    # in the layout's order of the boxes
    order = [b.name for b in boxes if b.name in on]
    assert list(dict.fromkeys(p.box for p in plates)) == order


def test_nothing_but_shelf_boxes_and_cartons_is_labelled(boxes, plates):
    labelled = {p.box for p in plates}
    for b in boxes:
        if b.name.endswith("/pallet") or b.name.startswith(("rack/upright", "rack/beam", "rack/shelf", "wall/",
                                                            "door/", "roof/", "fixture/", "light/", "marking/")):
            assert b.name not in labelled, b.name
            assert layout.label_kind(b) is None, b.name


def test_each_plate_lies_centred_on_its_y_face_lifted_by_the_spec(spec, boxes, plates):
    lb = spec["labels"]
    by_name = {b.name: b for b in boxes}
    for p in plates:
        b = by_name[p.box]
        sign = layout.LABEL_FACES[p.face][1]
        lift = lb["lift_m"] if p.kind == "base" else lb["ci_lift_m"]
        assert p.normal == layout.LABEL_FACES[p.face]
        assert p.center[0] == pytest.approx(b.center[0]) and p.center[2] == pytest.approx(b.center[2]), p
        assert p.center[1] == pytest.approx(b.center[1] + sign * (b.size[1] / 2 + lift)), p
        c = p.corners()
        assert np.allclose(c[:, 1], p.center[1])                                   # flat, parallel to the face
        assert c[:, 0].min() >= b.center[0] - b.size[0] / 2 - 1e-9 and c[:, 0].max() <= b.center[0] + b.size[0] / 2 + 1e-9
        assert c[:, 2].min() >= b.center[2] - b.size[2] / 2 - 1e-9 and c[:, 2].max() <= b.center[2] + b.size[2] / 2 + 1e-9


def test_each_plate_takes_its_size_from_the_spec(spec, boxes, plates):
    lb = spec["labels"]
    by_name = {b.name: b for b in boxes}
    for p in plates:
        which = layout.label_kind(by_name[p.box])
        if p.kind == "base":
            assert (p.width, p.height) == pytest.approx(tuple(lb[which]["size"])), p
        else:
            assert (p.width, p.height) == pytest.approx((lb[which]["ci_width"], lb[which]["ci_width"] / ASPECT)), p
        c = p.corners()
        assert np.ptp(c[:, 0]) == pytest.approx(p.width) and np.ptp(c[:, 2]) == pytest.approx(p.height)


def test_no_plate_reaches_into_an_aisle(spec, plates):
    strips = layout.aisle_strips(spec)
    for p in plates:
        assert not any(lo + 1e-9 < p.center[1] < hi - 1e-9 for lo, hi in strips), p


def test_every_plate_lies_within_its_boxs_rack_row(spec, boxes, plates):
    by_name = {b.name: b for b in boxes}
    bands = layout.rack_row_bands(spec)
    for p in plates:
        lo, hi = _band(bands, by_name[p.box])
        assert lo <= p.center[1] <= hi, p


def test_the_rack_row_bands_are_the_rows_the_layout_builds(spec, boxes):
    """Each band runs from one depth face of a row to the other: every rack member of the row lies inside it,
    and the floor between bands is the aisle strips."""
    bands = layout.rack_row_bands(spec)
    assert len(bands) == 2 * spec["rack"]["rows_per_side"]
    for b in boxes:
        if b.name.startswith(("rack/upright", "rack/beam", "rack/shelf")):
            lo, hi = _aabb(b)[0][1], _aabb(b)[1][1]
            assert any(blo - 1e-9 <= lo and hi <= bhi + 1e-9 for blo, bhi in bands), b.name
    assert all(hi - lo == pytest.approx(spec["rack"]["depth"]) for lo, hi in bands)
    edges = sorted(v for band in bands for v in band)
    strips = layout.aisle_strips(spec)
    inner = sorted(v for strip in strips for v in strip)[1:-1]          # without the two walls
    assert edges == pytest.approx(inner)


def _band(bands, b):
    return next((lo, hi) for lo, hi in bands if lo < b.center[1] < hi)


def _clearances(spec, boxes):
    """Labelled boxes in layout order with the room between each y face and its rack row's face (the smaller)."""
    bands = layout.rack_row_bands(spec)
    out = []
    for b in boxes:
        if layout.label_kind(b) is None:
            continue
        lo, hi = _band(bands, b)
        out.append((b, min(hi - (b.center[1] + b.size[1] / 2), (b.center[1] - b.size[1] / 2) - lo)))
    return out


def _labels(spec, lift, ci_lift):
    import copy
    s = copy.deepcopy(spec)
    s["labels"]["lift_m"], s["labels"]["ci_lift_m"] = lift, ci_lift
    return s


def test_a_plate_lifted_out_of_the_hall_and_a_card_into_an_aisle_are_refused(spec, boxes):
    """A card 0.1 m off its face (past a carton's face, into the aisle) and a CI 40 m off it (past the wall):
    the CI of the first labelled box is the first plate out of its row, and nothing is built."""
    s = _labels(spec, 0.1, 40.0)
    first = _clearances(spec, boxes)[0][0]
    with pytest.raises(ValueError, match=f"^{re.escape(first.name)}: the CI on its \\+y face would lie .* outside its rack row"):
        layout.validate_spec(s)
    with pytest.raises(ValueError, match="outside its rack row"):
        layout.box_labels(s, boxes, ASPECT)


def test_a_white_card_past_its_rows_face_is_refused_though_its_ci_is_not(spec, boxes):
    """The card checked on its own: a card lifted past the narrowest room while the CI stays inside the row.
    validate_spec refuses that spec first for its lifts' order, so label_fit is asked directly."""
    room = min(c for _, c in _clearances(spec, boxes))
    s = _labels(spec, room + 0.01, room / 2)
    first = next(b for b, c in _clearances(spec, boxes) if c < room + 0.01)
    with pytest.raises(ValueError, match=f"^{re.escape(first.name)}: the white card on its .y face would lie .* outside its rack row"):
        layout.label_fit(s, boxes)
    layout.label_fit(_labels(spec, room / 4, room / 2), boxes)          # the card back in: nothing else is wrong


def test_a_ci_past_its_rows_face_is_refused_though_its_card_is_not(spec, boxes):
    room = min(c for _, c in _clearances(spec, boxes))
    s = _labels(spec, room / 2, room + 0.01)
    first = next(b for b, c in _clearances(spec, boxes) if c < room + 0.01)
    with pytest.raises(ValueError, match=f"^{re.escape(first.name)}: the CI on its .y face would lie .* outside its rack row"):
        layout.validate_spec(s)
    layout.validate_spec(_labels(spec, room / 2, room * 0.9))            # the CI back in: accepted


def test_a_labelled_box_in_no_rack_row_is_refused(spec):
    stray = layout.Box("box/spr0/b0l0/s0l0", (0.0, 0.0, 2.0), (0.4, 0.4, 0.3), "cardboard", True, "rack")
    with pytest.raises(ValueError, match="box/spr0/b0l0/s0l0: its centre .* is in no rack row"):
        layout.label_fit(spec, [stray])


def test_the_labels_leave_the_layouts_box_list_as_it_was(spec, boxes, plates):
    """The colliders, the occlusion geometry, the aisle guard and the measuring tools read build_layout's list:
    the labels are not in it, and the list does not depend on them."""
    import copy
    bare = copy.deepcopy(spec)
    del bare["labels"]
    assert layout.build_layout(bare) == boxes
    other = copy.deepcopy(spec)
    other["labels"]["lift_m"], other["labels"]["ci_lift_m"] = 0.01, 0.02
    other["labels"]["shelf_box"]["size"] = [0.1, 0.05]
    assert layout.build_layout(other) == boxes
    names = {b.name for b in boxes}
    assert len(names) == len(boxes) and not any(isinstance(b, layout.LabelPlate) for b in boxes)
    assert {p.box for p in plates} <= names


def test_a_plates_corners_wind_as_the_carried_boxs_so_the_ci_reads_alike():
    """The CI's UVs are (0,0) (1,0) (1,1) (0,1) on the corners in this order on both assets: the same corners
    put the picture the same way round on a warehouse box as on the carried one."""
    from generation.scene import carried_box as cb

    for face, normal in layout.LABEL_FACES.items():
        p = layout.LabelPlate("box/x", face, "ci", (1.0, 2.0, 3.0), normal, 0.18, 0.045)
        rect = cb.LabelRect(face, np.array(p.center), np.array(normal), np.array(layout.LABEL_UP), p.width, p.height)
        assert np.allclose(p.corners() - np.array(p.center), cb.plate_corners(rect)), face
        # the first corner to the second runs along the viewer's right, the first to the fourth up
        c = p.corners()
        assert np.allclose((c[1] - c[0]) / p.width, np.cross(layout.LABEL_UP, normal))
        assert np.allclose((c[3] - c[0]) / p.height, layout.LABEL_UP)


def test_a_shelf_label_looks_like_the_carried_boxs(spec):
    """The same card, CI and look as the carried box's labels (carried_box.py holds them as constants): the size,
    the CI's width, the lifts, the white and the roughness. A bottom-tier carton's card and CI are twice as large."""
    from generation.scene import carried_box as cb

    lb, mats = spec["labels"], spec["materials"]
    assert tuple(lb["shelf_box"]["size"]) == pytest.approx(cb.LABEL_M)
    assert lb["shelf_box"]["ci_width"] == pytest.approx(cb.CI_WIDTH_M)
    assert (lb["lift_m"], lb["ci_lift_m"]) == pytest.approx((cb.LABEL_LIFT_M, cb.CI_LIFT_M))
    assert tuple(mats[lb["material"]]["color"]) == pytest.approx((cb.LABEL_WHITE,) * 3)
    assert mats[lb["material"]]["roughness"] == pytest.approx(cb.LABEL_ROUGHNESS)
    assert set(mats[lb["material"]]) == {"color", "roughness"}
    assert mats[lb["ci_material"]] == {"picture": "keti_ci", "roughness": cb.LABEL_ROUGHNESS}
    assert tuple(lb["floor_carton"]["size"]) == pytest.approx(tuple(2 * v for v in lb["shelf_box"]["size"]))
    assert lb["floor_carton"]["ci_width"] == pytest.approx(2 * lb["shelf_box"]["ci_width"])


def test_box_labels_refuses_a_ci_taller_than_its_card_or_no_aspect(spec, boxes):
    tall = spec["labels"]["shelf_box"]["ci_width"] / spec["labels"]["shelf_box"]["size"][1] * 0.99
    with pytest.raises(ValueError, match="taller than its card"):
        layout.box_labels(spec, boxes, tall)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="width / height"):
            layout.box_labels(spec, boxes, bad)


def test_label_fit_refuses_a_box_whose_face_does_not_hold_its_card(spec):
    w, h = spec["labels"]["shelf_box"]["size"]
    narrow = layout.Box("box/spr0/b0l0/s0l0", (0.0, 5.5, 2.0), (w - 0.01, 0.4, 0.3), "cardboard", True, "rack")
    with pytest.raises(ValueError, match="box/spr0/b0l0/s0l0: its y face"):
        layout.label_fit(spec, [narrow])
    low = layout.Box("box/spr0/b0l0/s0l0", (0.0, 5.5, 2.0), (0.4, 0.4, h - 0.01), "cardboard", True, "rack")
    with pytest.raises(ValueError, match="its y face"):
        layout.label_fit(spec, [low])


@pytest.mark.parametrize("path,value,why", [
    (("labels", "shelf_box", "size"), [0.36, 0.07], "labels.shelf_box.size"),       # wider than size_min's x (0.35)
    (("labels", "shelf_box", "size"), [0.20, 0.31], "labels.shelf_box.size"),       # taller than size_min's z (0.30)
    (("labels", "floor_carton", "size"), [1.3, 0.14], "labels.floor_carton.size"),  # wider than the 1.2 m carton
    (("labels", "shelf_box", "size"), [0.20], "labels.shelf_box.size"),
    (("labels", "shelf_box", "ci_width"), 0.21, "ci_width"),                        # wider than its card
    (("labels", "shelf_box", "ci_width"), 0.0, "ci_width"),
    (("labels", "floor_carton", "ci_width"), -0.1, "ci_width"),
    (("labels", "lift_m"), 0.0, "lift_m"),                                          # a card on its face shares it
    (("labels", "ci_lift_m"), 0.001, "ci_lift_m"),                                  # the CI not above its card
    (("labels", "ci_lift_m"), 0.06, "CI on its .y face would lie .* outside its rack row"),   # a carton's face is 0.05 m in
    (("labels", "material"), "keti_ci", "labels.material must be a color material"),
    (("labels", "ci_material"), "label_white", "labels.ci_material must be a picture material"),
    (("labels", "ci_material"), "keti_logo", "labels.ci_material 'keti_logo' is not declared"),
    (("labels", "colour"), "red", "labels: unknown keys"),
    (("labels", "shelf_box", "depth"), 0.1, "labels.shelf_box must be a mapping of exactly"),
    (("materials", "keti_ci", "picture"), "", "materials.keti_ci.picture"),
    (("materials", "keti_ci", "roughness"), 1.5, "materials.keti_ci.roughness"),
    (("materials", "keti_ci", "color"), [1.0, 1.0, 1.0], "exactly one of"),
    (("materials", "keti_ci", "tile_m"), 0.5, "unknown keys"),
    (("boxes", "material"), "keti_ci", "is a picture"),                             # a cube has no UVs
])
def test_validate_spec_rejects_labels_that_cannot_be_built(spec, path, value, why):
    with pytest.raises(ValueError, match=why):
        layout.validate_spec(_mutated(spec, path, value))


def test_validate_spec_requires_the_labels(spec):
    import copy
    s = copy.deepcopy(spec)
    del s["labels"]
    with pytest.raises(ValueError, match="labels"):
        layout.validate_spec(s)


def test_no_floor_loads_need_no_carton_label_to_fit(spec):
    s = _mutated(spec, ("rack", "floor_loads", "per_bay"), 0)
    s["labels"]["floor_carton"]["size"] = [5.0, 5.0]
    s["labels"]["floor_carton"]["ci_width"] = 1.0
    layout.validate_spec(s)                                    # must not raise: no carton is built
    assert not [p for p in layout.box_labels(s, layout.build_layout(s), ASPECT) if "carton" in p.box]
