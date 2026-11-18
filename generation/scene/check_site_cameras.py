#!/usr/bin/env python3
"""Do the third-person and CCTV cameras of each scenario's site see what they are placed for?

    .venv/bin/python generation/scene/check_site_cameras.py

Isaac-free: sight lines through the warehouse layout (generation/scene/layout.py) by the geometry of
generation/visibility.py, for every scenario of meta/scenarios.json that declares a site (scene.site --
generation/sites.py), with the cameras placed by meta/cameras.json's ``sites`` the way the run places them
(generation/cameras.py fixed_pose and follow_pose). Prints one row per criterion and site and exits 1 when
a placement misses one. A site that holds a camera's placement per side the person comes from (meta/cameras.json
by_approach_side) is checked once per side, its rows named by the side ("S1g · 서쪽에서"), the robot's path turned on
the spot the way the conditions table turns it for a row from that side (side_site).

The criteria are the project's own, for checking where the cameras stand (not part of any judgement):

- CCTV: a person standing at each corner of the person's region and at its centre has at least
  CCTV_MIN_PERSON_FRACTION of the body's sample points in view and unoccluded, and a full-body height of
  at least CCTV_MIN_HEIGHT_PX in the picture; and wherever the robot's path comes within CCTV_ROBOT_NEAR_M
  of the region, at least CCTV_MIN_ROBOT_FRACTION of the robot's points are seen.
- Third person: at every PATH_STEP_M along the path, at least FOLLOW_MIN_ROBOT_FRACTION of the robot's
  points are seen; and while the robot is within FOLLOW_PERSON_NEAR_M of the region's centre, a person
  standing at that centre has at least FOLLOW_MIN_PERSON_FRACTION of the body seen.

The robot is a ROBOT_BOX_M box standing on its path, turned with it: its points are a lattice inside the box,
and in the third-person criterion the box hides the person behind it. A path row where any of the person's
sample points lies inside the robot's box is left out of that criterion: there the robot occupies the person's
place -- those rows are the contact itself, and the camera's view of it is checked by eye in the recorded
videos (the table lists them as 겹침). The CCTV's person is measured with no robot in the way.
The person is the seeing rule's sample points standing upright (visibility.standing_part_poses); a person's
fraction is the least over PERSON_FACINGS_DEG, so the facing the scenario gives the person does not decide it.
The person hides nothing (the body does not occlude) and the floor occludes nothing.

visibility.occluded ignores a box round the camera -- right for a camera inside the robot's own link boxes,
wrong for a camera standing inside a rack: its picture would be the inside of the rack while the fractions
read as if the rack were not there. So a fractions-only pass is not enough: the CCTV's eye and the
third-person eye -- at places along its own path no more than EYE_STEP_M apart (eye_rows), turns included --
must stand outside every layout box the sight lines are tested against, or the site has a problem. Reported
only, not judged: the third-person eye's least distance
from any layout box along the path, and the length of path during which it stands inside a rack row's
footprint (seen from above) -- between the boxes, but among the rack's parts.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation import cameras, conditions, hri_geometry, sites, visibility  # noqa: E402
from generation.dummy import joints_spec  # noqa: E402
from generation.dummy.joints_spec import JointedSpec  # noqa: E402
from generation.scene import layout  # noqa: E402

# The criteria (the project's own).
CCTV_MIN_PERSON_FRACTION = 0.8
CCTV_MIN_HEIGHT_PX = 60.0
CCTV_ROBOT_NEAR_M = 1.5
CCTV_MIN_ROBOT_FRACTION = 0.8
PATH_STEP_M = 0.25
FOLLOW_MIN_ROBOT_FRACTION = 0.8
FOLLOW_PERSON_NEAR_M = 2.0
FOLLOW_MIN_PERSON_FRACTION = 0.5
# The robot as this check sees it: a box along its heading, across it and up from the floor (m).
ROBOT_BOX_M = (0.5, 0.3, 1.75)
# Cells of the lattice inside the robot box along each of its axes; the cells' centres are the robot's points.
ROBOT_LATTICE = (5, 3, 9)
# The robot root's height while it walks: the H1's spawn height (Isaac Lab's H1 configuration), which the
# follow camera's offsets are added to.
ROBOT_ROOT_Z_M = 1.05
PERSON_FACINGS_DEG = (0.0, 90.0, 180.0, 270.0)
# The most the third-person eye moves between two of the places it is held out of the layout's boxes at
# (eye_rows): finer than PATH_STEP_M, so that a camera crossing a rack row (0.9 m deep) between two places is
# still found.
EYE_STEP_M = 0.05
FOLLOW_TYPE, FIXED_TYPE = "follow", "fixed"
# The follow frame whose camera turns with the robot's heading (cameras.follow_pose).
HEADING_FRAME = "heading"
# How a row names the side the person comes from, for a site checked per side.
SIDE_LABELS = {"west": "서쪽에서", "east": "동쪽에서"}


def _camera_of(canon: Mapping, kind: str) -> str:
    names = [n for n, c in canon["cameras"].items() if c.get("type") == kind]
    if len(names) != 1:
        raise ValueError(f"meta/cameras.json 에 {kind} 카메라가 하나가 아니라 {len(names)} 대입니다: {names}")
    return names[0]


def camera_sides(canon: Mapping, scenario_id: str) -> tuple[str | None, ...]:
    """The sides the person comes from that the scenario's site is checked for: every side a third-person or CCTV
    camera holds a placement for there (cameras.site_sides), in the canon's order; (None,) when each holds one."""
    sides = [side for c in canon["cameras"].values() if c.get("type") in (FOLLOW_TYPE, FIXED_TYPE)
             for side in cameras.site_sides(c, scenario_id)]
    return tuple(side for side in cameras.APPROACH_SIDES if side in sides) or (None,)


def side_site(site: sites.Site, side: str | None) -> sites.Site:
    """The site as an episode whose person comes from ``side`` has it: a turn on the spot turned the way the
    conditions table turns it for a row from that side (conditions.row_site -- away from the person; the line the
    person walks, which row_site also reads, is not taken), the rest of the site the canon's. The site itself when
    ``side`` is None or its path has no turn on the spot."""
    if side is None or site.path["kind"] != "turn_then_walk":
        return site
    row = {"approach_side": side, "lateral_offset_m": 0.0}
    return dataclasses.replace(site, path=dict(conditions.row_site(site, row).path))


def _side_label(scenario_id: str, side: str | None) -> str:
    return scenario_id if side is None else f"{scenario_id} · {SIDE_LABELS.get(side, side)}"


def _label(report: Mapping) -> str:
    return _side_label(report["scenario_id"], report.get("approach_side"))


def _cells(text: str) -> int:
    """How many terminal cells ``text`` takes: a wide character (Hangul) two, any other one."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    """``text`` padded with spaces to ``width`` terminal cells."""
    return text + " " * max(0, width - _cells(text))


def robot_box(pose) -> visibility.Obb:
    """The robot's box at a path row (x, y, yaw rad): standing on the floor, turned with the heading."""
    x, y, yaw = (float(v) for v in pose)
    length, width, height = ROBOT_BOX_M
    return visibility.Obb(center=np.array([x, y, height / 2.0]),
                          rot_xyzw=np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]),
                          half=np.array([length, width, height]) / 2.0)


def robot_points(pose) -> np.ndarray:
    """(N, 3) points of the robot at a path row: the centres of the ROBOT_LATTICE cells of its box."""
    box = robot_box(pose)
    axes = [(np.arange(n) + 0.5) / n * 2.0 - 1.0 for n in ROBOT_LATTICE]
    local = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3) * box.half
    return box.center + cameras._q_rot(box.rot_xyzw, local)


def standing_person(body_points: Mapping[str, np.ndarray], xy, yaw_rad: float,
                    spec: JointedSpec | None = None) -> dict[str, np.ndarray]:
    """The person standing at ``xy`` facing ``yaw_rad``: {"points": the body's sample points in the world,
    "root": the pelvis's centre -- the person's root (hri_geometry.pelvis_body, the body observation.human.root_pose
    records), the point the full-body height is measured to}."""
    spec = joints_spec.load() if spec is None else spec
    poses = visibility.standing_part_poses(spec, xy, yaw_rad)
    return {"points": visibility.points_world(body_points, poses),
            "root": poses[hri_geometry.pelvis_body(spec)][:3].copy()}


def _inside_any(point, lo: np.ndarray, hi: np.ndarray) -> bool:
    """Whether the point is in one of the boxes (lo, hi), faces included."""
    return bool(np.all((lo <= point) & (point <= hi), axis=1).any())


def _inside_box(box: visibility.Obb, points: np.ndarray) -> int:
    """How many of the points are inside the turned box."""
    to_box = np.array([-box.rot_xyzw[0], -box.rot_xyzw[1], -box.rot_xyzw[2], box.rot_xyzw[3]])
    local = cameras._q_rot(to_box, np.asarray(points, float) - box.center)
    return int(np.all(np.abs(local) <= box.half, axis=1).sum())


def overlaps(box: visibility.Obb, people: Sequence[np.ndarray]) -> bool:
    """Whether the robot's box holds any of the person's sample points (at any of the facings): the robot stands
    in the person's place -- the contact itself."""
    return any(_inside_box(box, p) for p in people)


def _gap(point, lo: np.ndarray, hi: np.ndarray) -> float:
    """The least distance from the point to the boxes (lo, hi); 0 inside one."""
    out = np.maximum(np.maximum(lo - point, point - hi), 0.0)
    return float(np.linalg.norm(out, axis=1).min()) if len(lo) else math.inf


def rack_row_footprints(boxes: Sequence[layout.Box]) -> dict[str, tuple[float, float, float, float]]:
    """{row tag: (x0, x1, y0, y1)} -- each rack row's footprint seen from above, the bounds of the colliders named
    rack/<part>/<row tag>/... (uprights, beams, shelves and the bottom tier's loads)."""
    out: dict[str, list[float]] = {}
    for b in boxes:
        parts = b.name.split("/")
        if not b.collider or parts[0] != "rack" or len(parts) < 3:
            continue
        lo, hi = layout.bounds(b)
        r = out.setdefault(parts[2], [math.inf, -math.inf, math.inf, -math.inf])
        r[0], r[1], r[2], r[3] = min(r[0], lo[0]), max(r[1], hi[0]), min(r[2], lo[1]), max(r[3], hi[1])
    return {tag: tuple(r) for tag, r in out.items()}


def eye_rows(site: sites.Site, placement: Mapping) -> np.ndarray:
    """(N, 3) rows of the robot's root x, y and yaw (rad) at which the third-person eye is held out of the layout's
    boxes: the path's rows at EYE_STEP_M (sites.nominal_path), each span between two of them cut into equal parts
    -- root x, y and the continuous yaw linear between the two rows -- so that consecutive eyes are at most
    EYE_STEP_M apart. Over a span the eye moves at most the root's displacement plus, for a camera that turns with
    the heading, the turn times the eye's horizontal reach from the root: the span is cut by that bound. On a
    tight U-turn or a turn on the spot the eye sweeps far between two path rows while the root hardly moves. The
    path rows are all kept, in order."""
    path = sites.nominal_path(site, step_m=EYE_STEP_M)
    reach = math.hypot(*placement["eye_offset"][:2]) if placement["frame"] == HEADING_FRAME else 0.0
    out = [path[:1]]
    for a, b in zip(path[:-1], path[1:]):
        moved = math.hypot(b[0] - a[0], b[1] - a[1]) + reach * abs(b[2] - a[2])
        n = max(1, math.ceil(moved / EYE_STEP_M - sites.COUNT_SLACK))
        t = np.arange(1, n + 1)[:, None] / n
        out.append(a * (1.0 - t) + b * t)
    return np.concatenate(out)


def _region_distance(site: sites.Site, xy: np.ndarray) -> np.ndarray:
    lo = np.array([site.human_x_m[0], site.human_y_m[0]])
    hi = np.array([site.human_x_m[1], site.human_y_m[1]])
    gap = np.maximum(np.maximum(lo - xy, xy - hi), 0.0)
    return np.hypot(gap[:, 0], gap[:, 1])


def _region_centre(site: sites.Site) -> np.ndarray:
    return np.array([sum(site.human_x_m) / 2.0, sum(site.human_y_m) / 2.0])


def _station_points(site: sites.Site) -> np.ndarray:
    """The region's four corners and its centre (a region of no width along an axis has its corners twice)."""
    (x0, x1), (y0, y1) = site.human_x_m, site.human_y_m
    return np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1], [(x0 + x1) / 2.0, (y0 + y1) / 2.0]])


def _least(rows: list[tuple[float, list[float]]]) -> dict:
    if not rows:
        return {"rows": 0, "fraction_min": None, "at": None}
    fraction, at = min(rows, key=lambda r: r[0])
    return {"rows": len(rows), "fraction_min": fraction, "at": at}


def coverage(site: sites.Site, canon: Mapping, boxes: Sequence[layout.Box], body_points: Mapping[str, np.ndarray],
             *, spec: JointedSpec | None = None, scene_spec: Mapping | None = None,
             approach_side: str | None = None) -> dict:
    """What the site's CCTV and third-person cameras see on ``boxes`` (a layout): per camera the fractions the
    criteria read, each the least over what it is measured on, with where that least was. ``body_points`` is
    visibility.body_sample_points of the dummy ``spec`` (joints_spec.load() when not given); ``scene_spec`` is
    read only by a CCTV placed by the rack-face rule. ``approach_side`` picks a camera's placement where the site
    holds one per side the person comes from (and is refused there when None); pass side_site(site, side) as
    ``site`` for that side's path."""
    spec = joints_spec.load() if spec is None else spec
    sid = site.scenario_id
    lo, hi = visibility.layout_occluders(boxes)
    path = sites.nominal_path(site, step_m=PATH_STEP_M)
    facings = [math.radians(d) for d in PERSON_FACINGS_DEG]

    cctv = _camera_of(canon, FIXED_TYPE)
    cam = canon["cameras"][cctv]
    scene_spec = layout.load_spec(layout.SCENE_SPEC) if scene_spec is None else scene_spec
    pos, quat = cameras.fixed_pose(cam, dict(scene_spec), sid, approach_side=approach_side)
    person_rows = []
    for xy in _station_points(site):
        fractions, heights = [], []
        for yaw in facings:
            person = standing_person(body_points, xy, yaw, spec)
            fractions.append(visibility.visible_fraction(cam, pos, quat, person["points"], lo, hi))
            heights.append(visibility.height_px(cam, pos, quat, person["root"], spec.anthropometry["stature"]))
        person_rows.append({"xy": [float(v) for v in xy], "fraction": min(fractions), "height_px": min(heights)})
    near = path[_region_distance(site, path[:, :2]) < CCTV_ROBOT_NEAR_M]
    robot_rows = [(visibility.visible_fraction(cam, pos, quat, robot_points(row), lo, hi), [float(v) for v in row])
                  for row in near]
    report = {"scenario_id": sid, "approach_side": approach_side,
              "cctv": {"camera": cctv, "eye_m": [float(v) for v in pos], "eye_inside": _inside_any(pos, lo, hi),
                       "person": person_rows, "robot": _least(robot_rows)}}

    follow = _camera_of(canon, FOLLOW_TYPE)
    cam = canon["cameras"][follow]
    placement = cameras.site_placement(cam, sid, name=follow, approach_side=approach_side)
    first_yaw = float(path[0, 2])
    centre = _region_centre(site)
    people = [standing_person(body_points, centre, yaw, spec)["points"] for yaw in facings]
    robot_rows, person_rows, overlap_rows = [], [], []
    for row in path:
        root = np.array([row[0], row[1], ROBOT_ROOT_Z_M])
        eye, target = cameras.follow_pose(placement, root, float(row[2]), first_yaw)
        pos, quat = cameras.look_at_pose(eye, target)
        at = [float(v) for v in row]
        robot_rows.append((visibility.visible_fraction(cam, pos, quat, robot_points(row), lo, hi), at))
        if math.dist(row[:2], centre) < FOLLOW_PERSON_NEAR_M:
            box = robot_box(row)
            fraction = min(visibility.visible_fraction(cam, pos, quat, p, lo, hi, (box,)) for p in people)
            if overlaps(box, people):
                overlap_rows.append({"at": at, "fraction": fraction})        # the contact itself: not judged
            else:
                person_rows.append((fraction, at))
    footprints = list(rack_row_footprints(boxes).values())
    inside, gap_min, in_rack_m, previous = [], math.inf, 0.0, None
    samples = eye_rows(site, placement)
    for row in samples:
        eye, _ = cameras.follow_pose(placement, np.array([row[0], row[1], ROBOT_ROOT_Z_M]), float(row[2]), first_yaw)
        if _inside_any(eye, lo, hi):
            inside.append([float(v) for v in row])
        gap_min = min(gap_min, _gap(eye, lo, hi))
        in_rack = any(x0 <= eye[0] <= x1 and y0 <= eye[1] <= y1 for x0, x1, y0, y1 in footprints)
        if previous is not None and previous[1] and in_rack:
            in_rack_m += math.dist(previous[0][:2], row[:2])
        previous = (row, in_rack)
    person = _least(person_rows)
    person["overlap_rows"] = overlap_rows
    report["follow"] = {"camera": follow, "frame": placement["frame"], "person_xy": [float(v) for v in centre],
                        "eye_samples": len(samples),
                        "eye_inside_rows": len(inside), "eye_inside_first_at": inside[0] if inside else None,
                        "eye_gap_min_m": gap_min, "eye_in_rack_footprint_path_m": in_rack_m,
                        "robot": _least(robot_rows), "person": person}
    return report


def _rows(report: Mapping) -> list[tuple[str, str, str, str, float | None, float]]:
    """(key, camera, criterion, where, measured, threshold) per criterion -- measured None when nothing was
    measured."""
    c, f = report["cctv"], report["follow"]
    people = c["person"]
    worst_person = min(people, key=lambda p: p["fraction"])
    lowest = min(people, key=lambda p: p["height_px"])
    return [
        ("cctv_person", c["camera"], f"사람 몫 — 사람 자리의 모서리 넷 · 가운데 {len(people)} 곳", _xy(worst_person["xy"]),
         worst_person["fraction"], CCTV_MIN_PERSON_FRACTION),
        ("cctv_height", c["camera"], "사람 키 높이 px — 같은 다섯 곳", _xy(lowest["xy"]), lowest["height_px"], CCTV_MIN_HEIGHT_PX),
        ("cctv_robot", c["camera"], f"로봇 몫 — 사람 자리 {CCTV_ROBOT_NEAR_M:g} m 안의 길 {c['robot']['rows']} 점", _xy(c["robot"]["at"]),
         c["robot"]["fraction_min"], CCTV_MIN_ROBOT_FRACTION),
        ("follow_robot", f["camera"], f"로봇 몫 — 길 {PATH_STEP_M:g} m 마다 {f['robot']['rows']} 점", _xy(f["robot"]["at"]),
         f["robot"]["fraction_min"], FOLLOW_MIN_ROBOT_FRACTION),
        ("follow_person", f["camera"],
         f"사람 몫 — 로봇이 사람 자리 가운데 {FOLLOW_PERSON_NEAR_M:g} m 안인 {f['person']['rows']} 점(겹침 {len(f['person']['overlap_rows'])} 점 뺌)",
         _xy(f["person"]["at"]), f["person"]["fraction_min"], FOLLOW_MIN_PERSON_FRACTION),
    ]


def _xy(at) -> str:
    return "-" if at is None else f"({at[0]:.2f}, {at[1]:.2f})"


def coverage_problems(report: Mapping) -> list[str]:
    """One "<scenario> <camera>: <why>" line per criterion the report misses, and one per camera whose eye stands
    inside a layout box; [] when every one holds. A criterion with nothing to measure on (no path row near enough)
    is a problem too: it says nothing. The third-person rows where the robot's box holds the person (overlap) are
    not measured, so they are never a problem."""
    sid, c, f = _label(report), report["cctv"], report["follow"]
    out = []
    if c["eye_inside"]:
        out.append(f"{sid} {c['camera']}: 눈 {_xy(c['eye_m'])} 이 레이아웃 상자 안입니다 — 그림은 그 상자 속입니다")
    if f["eye_inside_rows"]:
        out.append(f"{sid} {f['camera']}: 눈이 레이아웃 상자 안입니다 — {EYE_STEP_M:g} m 를 넘지 않는 간격으로 짚은 눈 "
                   f"{f['eye_samples']} 점 가운데 {f['eye_inside_rows']} 점 · 처음 로봇 {_xy(f['eye_inside_first_at'])} "
                   "· 거기서 잰 몫은 그 상자를 빼고 잰 것입니다")
    for _, camera, criterion, where, measured, threshold in _rows(report):
        head = f"{sid} {camera}: {criterion}"
        if measured is None:
            out.append(f"{head} — 잴 점이 없습니다")
        elif measured < threshold:
            out.append(f"{head} — 가장 낮은 값 {measured:.3f} 이 {threshold:g} 아래입니다 ({where})")
    return out


def _notes(report: Mapping, width: int) -> list[str]:
    """The table's report-only lines of a site: the overlap rows left out of the third-person person criterion, and
    the third-person eye's distance from the layout's boxes and its path inside rack rows. ``width``: the site
    column's, in terminal cells."""
    sid, f = _pad(_label(report), width), report["follow"]
    out = [f"  {sid} {f['camera']:<17} 겹침 — 로봇 {_xy(o['at'])} 의 상자가 사람 표본점을 품음 · 그 자리의 사람 몫 {o['fraction']:.3f} "
           "— 접촉 그 자체라 기준에서 뺌" for o in f["person"]["overlap_rows"]]
    out.append(f"  {sid} {f['camera']:<17} 눈 — 레이아웃 상자까지 최소 {f['eye_gap_min_m']:.3f} m · 랙 줄 발자국 안인 길 "
               f"{f['eye_in_rack_footprint_path_m']:.2f} m (보고만)")
    return out


def _site_scenarios(doc: Mapping) -> list[Mapping]:
    return [s for s in doc["scenarios"] if isinstance(s.get("scene"), Mapping) and "site" in s["scene"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", help="자리마다의 보고를 이 JSON 파일에 씀")
    args = p.parse_args(argv)
    canon = cameras.load(REPO / "meta" / "cameras.json")
    scenarios = json.loads((REPO / "meta" / "scenarios.json").read_text(encoding="utf-8"))
    scene_spec = layout.load_spec(layout.SCENE_SPEC)
    boxes = layout.build_layout(scene_spec)
    spec = joints_spec.load()
    body = visibility.body_sample_points(spec, visibility.seeing_rule(canon)["sample_points"])
    reports, problems = [], []
    checked = [(sites.load_site(scenario), side) for scenario in _site_scenarios(scenarios)
               for side in camera_sides(canon, scenario["scenario_id"])]
    # The site column is as wide as its widest label, in terminal cells (a side's label holds Hangul).
    width = max(_cells(text) for text in ["자리", *(_side_label(site.scenario_id, side) for site, side in checked)])
    print(f"{_pad('자리', width)} {'카메라':<17} {'기준':<44} {'가장 낮은 곳':<16} {'잰 값':>8} {'문턱':>6}  판정")
    for site, side in checked:
        report = coverage(side_site(site, side), canon, boxes, body, spec=spec, scene_spec=scene_spec,
                          approach_side=side)
        reports.append(report)
        for _, camera, criterion, where, measured, threshold in _rows(report):
            verdict = "잴 점 없음" if measured is None else ("통과" if measured >= threshold else "미달")
            value = "-" if measured is None else f"{measured:.3f}"
            print(f"{_pad(_label(report), width)} {camera:<17} {criterion:<44} {where:<16} {value:>8} "
                  f"{threshold:>6g}  {verdict}")
        for line in _notes(report, width):
            print(line)
        problems += coverage_problems(report)
    for line in problems:
        print(f"PROBLEM {line}")
    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=1, ensure_ascii=False), encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
