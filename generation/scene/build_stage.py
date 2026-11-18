#!/usr/bin/env python3
"""The Stage: one USD composing the warehouse, the vested dummy, the carried box and H1, with the robot's
five cameras on their links, the third-person and CCTV cameras of each site (one of each per side the person comes
from where the site is placed per side), and visible-only marks of each site's start, goal and the person's region.

    .venv/bin/python generation/robots/fetch_robot.py --robot unitree_h1 --nucleus <dir>   # H1 is measured
    PYTHONPATH= ~/isaacsim/python.sh generation/scene/build_stage.py    # -> build/keti_hri_stage_v1.usd
    PYTHONPATH= ~/isaacsim/python.sh generation/scene/check_stage.py    # opens it inside a release's assets/

Every asset is referenced by the name it ships under (generation.assets.shipped_asset_files), relative to the
Stage at assets/: the Stage opens inside a release's assets/, where what it composes sits beside it, and not
in the build directories. A release carries it only with everything it composes (generation.assets.
composed_shipped -- the canon entry's ``composes``, which COMPONENTS here must equal). Opened in Isaac Sim it
shows the scene; running a scenario -- its script, conditions, judgement and recording -- is the code
repository's commands. The Stage authors no physics and references no texture of its own: the lights are the
warehouse's, the marks a display colour.

stage_plan() lays out every prim from the canon without Isaac (the sites and S1e's ``carry`` of
meta/scenarios.json, the camera canon, the asset canon, H1's joint frames); robot_rest(), write_stage(), build() and
main() need pxr. The robot stands at S1e's start on its feet: its root height comes from the lowest point of H1's
own geometry (robot_rest). Its arms hold the carry pose -- the eight arm links are overs that set each link's
transform to where generation/robots/kinematics.py puts it at those joint angles, the legs stay as the robot's USD
has them -- and the box sits on the torso where the carry block puts it (generation/carry.py box_pose). Every
placement number here is the project's own choice (self-set) -- none is a standard value.

The canon's Stage entry keeps, beside the Stage's hash, what it was built from in a form read without pxr:
the robot's rest numbers the build measured (robot_rest) and the hash of the plan it wrote (plan_sha256 --
plan_digest). main() prints both for the entry. pin_problems() lays the plan out again from today's canon
with the kept rest numbers and holds it to plan_sha256, so a site, a camera or a placement changed and the
Stage not built again is caught where no USD can be opened (the tests in CI).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation import cameras as cam_canon  # noqa: E402
from generation import carry as carry_canon  # noqa: E402
from generation import sites  # noqa: E402
from generation.assets import (ASSETS_DIR, COMPOSES_KEY, TEXTURE_SUBDIR, check_shipped_sources,  # noqa: E402
                               load_assets, robot_ship_files, shipped_asset_files)
from generation.robots import kinematics  # noqa: E402
from generation.scene import layout  # noqa: E402
from generation.scene.carried_box import NOMINAL_SIZE_M  # noqa: E402

STAGE_KEY = "keti_hri_stage_v1"
OUT = REPO / "generation/scene/build/keti_hri_stage_v1.usd"
BUILT_BY = "generation/scene/build_stage.py"
STAGE_VERSION = "1"
ROOT = "/World"
# The prim under /World each composed asset is referenced at -> its meta/assets.json key.
COMPONENTS = (("Warehouse", "keti_logistics_aisle_v1"), ("Robot", "unitree_h1"),
              ("Dummy", "keti_dummy_ts15066_v2_vest"), ("CarriedBox", "keti_carried_box_v1"))
ROBOT_KEY = "unitree_h1"
# The site the robot, the dummy and the box are placed at -- the box and the arms by that scenario's carry block.
PLACED_AT = "S1e"
# S1e's person stands on a line in front of the rack face north of it, and faces it (+y).
DUMMY_FACING_YAW_DEG = 90.0
# The robot USD's links: each a child of its root prim with this one transform op, which a posed over sets.
LINK_OP = "xformOp:transform"
# How far the root link of the robot USD may be turned relative to its root prim (on unit vectors): the plan places
# the arm links in the root prim's frame from the root link's position alone.
ROOT_LINK_TURN_TOL = 1e-6
# The floor the simulation stands everything on (Isaac Lab's ground plane).
FLOOR_Z_M = 0.0
# The scope under a site that holds the cameras of one side the person comes from, where the site places them per
# side (meta/cameras.json by_approach_side).
SIDE_SCOPE = "from_{side}"
# The marks: thin plates lying above the painted floor lines, visible only (no collider, no material).
MARK_THICKNESS_M = 0.002
MARK_RADIUS_M = 0.25                     # the start and goal discs
REGION_MARGIN_M = 0.15                   # the person's region drawn this much wider each side (a line has no width)
MARK_COLORS = {"robot_start": (0.10, 0.65, 0.25), "robot_goal": (0.15, 0.35, 0.85),
               "human_region": (0.95, 0.45, 0.10)}
# The marks lie one on another, each a thickness above the one before: where a disc reaches into the person's region
# (S1g's start and goal both do) no two faces share a plane for the renderer to flicker between.
MARK_LAYERS = ("robot_start", "human_region", "robot_goal")
# The Stage entry's keys for what the Stage was built from: the robot's rest numbers (rest_record) and the
# plan's hash (plan_digest).
REST_KEY = "robot_rest"
PLAN_KEY = "plan_sha256"
# Decimal places the plan's numbers keep before they are hashed: a micrometre (a millionth of a unit vector) --
# far below anything the Stage shows, far above the last bits two numpy builds may give the same arithmetic.
PLAN_DIGITS = 6


@dataclass(frozen=True)
class RobotRest:
    """The robot's USD at its rest pose, in the frame of its root prim: the lowest point of its rendered
    geometry (m) and where its root link -- the articulation root the simulation reports the robot's pose
    by -- sits."""
    lowest_z_m: float
    root_link_m: tuple[float, float, float]


@dataclass(frozen=True)
class PlannedPrim:
    """One prim of the Stage. kind: "xform" · "scope" · "reference" (an Xform referencing ``ref``) · "over"
    (a prim the reference above it brings -- ``posed``: the over also sets that link's transform op, LINK_OP, to
    translate and orient) · "camera" · "disc" · "plate". Pose: translate and orient (xyzw) relative to the parent,
    scale for the marks; camera: focal_length_mm · horizontal_aperture_mm · vertical_aperture_mm ·
    clipping_range_m; color: a mark's display colour."""
    kind: str
    ref: str | None = None
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orient_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] | None = None
    camera: Mapping[str, object] | None = None
    color: tuple[float, float, float] | None = None
    posed: bool = False


def _yaw_xyzw(yaw_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(float(yaw_deg)) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def usd_camera_xyzw(ros_xyzw) -> tuple[float, float, float, float]:
    """A USD camera's orientation (xyzw) from the canon's ROS optical frame: a USD camera looks along its -z
    with +y up, a ROS optical frame along +z with +y down -- the same frame turned half a turn about its x,
    q * (1, 0, 0, 0)."""
    x, y, z, w = (float(v) for v in ros_xyzw)
    return (w, z, -y, -x)


def camera_lens(cam: Mapping) -> dict[str, object]:
    """The USD camera's lens for a canon camera: the focal length and apertures the simulation's camera
    gets (the Isaac Lab pinhole spawn: focal length from the canon fov, Isaac Sim's default horizontal
    aperture, the vertical one by the image's aspect), and the clipping range."""
    horizontal = cam_canon.HORIZONTAL_APERTURE_MM
    return {"focal_length_mm": cam_canon.focal_length_mm(cam["fov_h_deg"], horizontal),
            "horizontal_aperture_mm": horizontal,
            "vertical_aperture_mm": horizontal * cam["height"] / cam["width"],
            "clipping_range_m": tuple(float(v) for v in cam["clipping_range_m"])}


def composes_problems(assets: Mapping) -> list[str]:
    """What keeps the canon's Stage entry from saying it composes what this builder composes: no entry, or a
    ``composes`` other than COMPONENTS' keys. A release ships the Stage by the canon's list, so a builder
    that composes something the list leaves out would reference a file the release may lack."""
    entry = (assets.get("scene") or {}).get(STAGE_KEY)
    if not isinstance(entry, Mapping):
        return [f"meta/assets.json 의 scene 에 {STAGE_KEY} 가 없습니다"]
    want = [key for _, key in COMPONENTS]
    got = entry.get(COMPOSES_KEY)
    if not isinstance(got, list) or sorted(map(str, got)) != sorted(want) or len(got) != len(want):
        return [f"{STAGE_KEY}.composes {got!r} 가 이 빌더가 엮는 자산 {want} 와 다릅니다"]
    return []


def component_refs(assets: Mapping) -> dict[str, str]:
    """meta/assets.json key -> the reference the Stage writes for it: the path its USD ships under
    (shipped_asset_files for a batch that uses all of them), relative to assets/ where the Stage ships. Read
    from the canon without its composed entries: what the Stage references does not depend on whether the
    Stage itself is pinned. Raises ValueError for an asset the canon does not let a release carry."""
    plain = {group: {key: entry for key, entry in (assets.get(group) or {}).items()
                     if not (isinstance(entry, Mapping) and COMPOSES_KEY in entry)}
             for group in ("robots", "scene")}
    by_key = {key: entry for group in plain.values() for key, entry in group.items()}
    keys = [key for _, key in COMPONENTS]
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise ValueError(f"Stage 가 엮는 자산 {missing} 가 meta/assets.json 에 없습니다")
    shipped = shipped_asset_files(plain, [by_key[key]["usd_path"] for key in keys], REPO)
    out = {}
    for key in keys:
        name = Path(str(by_key[key]["usd_path"])).name
        found = [rel for rel, f in shipped.items()
                 if f.asset == key and PurePosixPath(rel).name == name
                 and PurePosixPath(rel).parent.name != TEXTURE_SUBDIR]
        if len(found) != 1:
            raise ValueError(f"{key}: 배포본에서 USD {name} 가 실리는 자리를 찾지 못했습니다 ({found}) — 배포본에 "
                             "실리지 않는 자산은 Stage 가 엮을 수 없습니다")
        out[key] = "./" + PurePosixPath(found[0]).relative_to(ASSETS_DIR).as_posix()
    return out


def _mark_z(spec: Mapping) -> float:
    """The marks' centre height: above the floor slab and the painted lines on it, as far again as the
    lines are lifted off the slab (logistics_aisle.yaml floor and floor_markings)."""
    paint = spec["floor_markings"]
    top = (float(spec["floor"]["z"]) + layout.SLAB / 2 + float(paint["lift"]) + float(paint["thickness"]))
    return top + float(paint["lift"]) + MARK_THICKNESS_M / 2


def _root_link_world(site: sites.Site, robot: RobotRest) -> tuple[float, float, float]:
    start = site.robot_start
    yaw = math.radians(start.yaw_deg)
    x, y, z = robot.root_link_m
    return (start.x + math.cos(yaw) * x - math.sin(yaw) * y,
            start.y + math.sin(yaw) * x + math.cos(yaw) * y,
            FLOOR_Z_M - robot.lowest_z_m + z)


def held_links(joints: Mapping, hold: Mapping[str, float]) -> list[str]:
    """The links the hold pose moves: the child of every held joint and everything below it in the joint tree, a
    parent before its children."""
    moved: list[str] = []
    for name in joints["order"]:
        joint = joints["joints"][name]
        if name in hold or joint["parent"] in moved:
            moved.append(joint["child"])
    return moved


def carry_of(scenarios: Mapping, scenario_id: str = PLACED_AT) -> carry_canon.Carry:
    """The carry block of the scenario the Stage is placed at; refused when it has none or it carries another asset
    than the box the Stage composes."""
    entry = next((s for s in scenarios["scenarios"] if s.get("scenario_id") == scenario_id), None)
    held = carry_canon.load_carry(entry) if isinstance(entry, Mapping) else None
    if held is None:
        raise ValueError(f"meta/scenarios.json 의 {scenario_id} 에 carry 가 없습니다 — 로봇이 상자를 드는 자리와 "
                         "팔 자세를 알 수 없습니다")
    box_key = dict((prim, key) for prim, key in COMPONENTS)["CarriedBox"]
    if held.asset != box_key:
        raise ValueError(f"{scenario_id}.carry.asset {held.asset!r} 이 Stage 가 엮는 상자 {box_key!r} 가 아닙니다")
    return held


def _robot_frame_poses(robot: RobotRest, joints: Mapping, hold: Mapping[str, float],
                       ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every link at the hold pose in the robot's root prim frame: the root link where robot_rest measured it, not
    turned (robot_rest refuses a USD whose root link is), the others where kinematics puts them relative to it."""
    if joints["root_link"] != root_link_name():
        raise ValueError(f"관절 틀의 뿌리 링크 {joints['root_link']!r} 이 관절 매핑의 아티큘레이션 뿌리 "
                         f"{root_link_name()!r} 가 아닙니다 — robot_rest 가 잰 뿌리 링크와 다릅니다")
    base = np.asarray(robot.root_link_m, float)
    return {link: (base + pos, quat)
            for link, (pos, quat) in kinematics.link_poses(joints, hold, frame=joints["root_link"]).items()}


def _in_world(site: sites.Site, robot: RobotRest, pose) -> tuple[np.ndarray, np.ndarray]:
    """A pose in the robot's root prim frame placed in the world, for the robot at the site's start on the floor."""
    start = site.robot_start
    turn = kinematics.axis_rotation("Z", math.radians(start.yaw_deg))
    pos, quat = pose
    return (np.array([start.x, start.y, FLOOR_Z_M - robot.lowest_z_m]) + turn @ np.asarray(pos, float),
            kinematics.quat_from_matrix(turn @ kinematics.quat_to_matrix(quat)))


def stage_plan(scenarios: Mapping, cameras_canon: Mapping, assets: Mapping, *,
               robot: RobotRest) -> dict[str, PlannedPrim]:
    """Prim path -> PlannedPrim for the whole Stage, parents before children. ``scenarios`` is
    meta/scenarios.json, ``cameras_canon`` meta/cameras.json, ``assets`` meta/assets.json; ``robot`` is what
    robot_rest() measured on the robot's USD (the plan cannot read a USD).

    The robot stands at PLACED_AT's start with its lowest point on the floor, its arms at that scenario's carry
    pose (posed overs of the links held_links names, in the robot's root prim frame); the dummy at the middle of
    that site's person's region facing DUMMY_FACING_YAW_DEG; the box of the nominal size on the torso where the
    carry block puts it. The mounted cameras sit
    under their parent links at their canon offsets (/World/Robot/<link>/<camera>); every site's follow and
    fixed cameras are placed for the robot at that site's start, beside the site's marks
    (/World/Sites/<scenario>/...) -- where a site holds a camera's placement per side the person comes from
    (meta/cameras.json by_approach_side), one camera per side, each under that side's scope
    (/World/Sites/<scenario>/from_<side>/<camera>). Raises ValueError when the canon cannot say: an asset the
    release cannot carry, a site the scenarios do not declare, a camera without a placement there, a scenario
    whose scene is not an object, or whose scene has a site key that holds no object (a scenario with no site key
    has no site on the Stage; one whose site is not an object would silently lose that site's cameras and marks),
    a placed scenario without a carry block (carry_of)."""
    refs = component_refs(assets)
    spec = layout.load_spec(layout.SCENE_SPEC)
    sited = []
    for s in scenarios["scenarios"]:
        scene = s.get("scene")
        if not isinstance(scene, Mapping):
            raise ValueError(f"{s.get('scenario_id')}.scene 이 객체가 아닙니다: {scene!r} — 그 시나리오에 자리가 있는지 "
                             "알 수 없습니다")
        if "site" not in scene:
            continue
        if not isinstance(scene["site"], Mapping):
            raise ValueError(f"{s.get('scenario_id')}.scene.site 가 객체가 아닙니다: {scene['site']!r} — 선언한 자리의 "
                             "카메라와 표시를 놓을 수 없습니다")
        sited.append(s)
    sited.sort(key=lambda s: s["scenario_id"])
    by_id = {s["scenario_id"]: sites.load_site(s) for s in sited}
    if PLACED_AT not in by_id:
        raise ValueError(f"meta/scenarios.json 에 {PLACED_AT} 의 scene.site 가 없습니다 — 로봇 · 인형 · 상자를 놓을 자리가 "
                         "없습니다")
    here = by_id[PLACED_AT]
    held = carry_of(scenarios)
    joints = kinematics.load_joints()
    hold = carry_canon.arm_hold_joints(held, joints["joints"])
    in_robot = _robot_frame_poses(robot, joints, hold)
    plan: dict[str, PlannedPrim] = {ROOT: PlannedPrim("xform")}
    for prim, key in COMPONENTS:
        if prim == "Robot":
            start = here.robot_start
            plan[f"{ROOT}/{prim}"] = PlannedPrim("reference", refs[key],
                                                 (start.x, start.y, FLOOR_Z_M - robot.lowest_z_m),
                                                 _yaw_xyzw(start.yaw_deg))
            for link in held_links(joints, hold):
                pos, quat = in_robot[link]
                plan[f"{ROOT}/{prim}/{link}"] = PlannedPrim("over", translate=tuple(float(v) for v in pos),
                                                            orient_xyzw=tuple(float(v) for v in quat), posed=True)
            for name, cam in cameras_canon["cameras"].items():
                if cam["type"] != "mounted":
                    continue
                link = f"{ROOT}/{prim}/{cam['parent_prim']}"
                plan.setdefault(link, PlannedPrim("over"))
                plan[f"{link}/{name}"] = PlannedPrim(
                    "camera", translate=tuple(float(v) for v in cam["offset_pos"]),
                    orient_xyzw=usd_camera_xyzw(cam["offset_rot_xyzw"]), camera=camera_lens(cam))
        elif prim == "Dummy":
            centre = (sum(here.human_x_m) / 2.0, sum(here.human_y_m) / 2.0, FLOOR_Z_M)
            plan[f"{ROOT}/{prim}"] = PlannedPrim("reference", refs[key], centre, _yaw_xyzw(DUMMY_FACING_YAW_DEG))
        elif prim == "CarriedBox":
            centre, quat = carry_canon.box_pose(*_in_world(here, robot, in_robot[held.link]), held, NOMINAL_SIZE_M)
            plan[f"{ROOT}/{prim}"] = PlannedPrim("reference", refs[key], tuple(float(v) for v in centre),
                                                 tuple(float(v) for v in quat))
        else:
            plan[f"{ROOT}/{prim}"] = PlannedPrim("reference", refs[key])
    plan[f"{ROOT}/Sites"] = PlannedPrim("scope")
    mark_z = _mark_z(spec)
    for sid, site in by_id.items():
        base = f"{ROOT}/Sites/{sid}"
        plan[base] = PlannedPrim("scope")
        root_link = _root_link_world(site, robot)
        yaw = math.radians(site.robot_start.yaw_deg)
        for name, cam in cameras_canon["cameras"].items():
            if cam["type"] not in ("follow", "fixed"):
                continue
            for side in cam_canon.site_sides(cam, sid, name=name) or (None,):
                under = base if side is None else f"{base}/{SIDE_SCOPE.format(side=side)}"
                plan.setdefault(under, PlannedPrim("scope"))
                if cam["type"] == "follow":
                    placement = cam_canon.site_placement(cam, sid, name=name, approach_side=side)
                    eye, target = cam_canon.follow_pose(placement, root_link, yaw, yaw)
                    pos, ros = cam_canon.look_at_pose(eye, target)
                else:
                    pos, ros = cam_canon.fixed_pose(cam, spec, sid, approach_side=side)
                plan[f"{under}/{name}"] = PlannedPrim("camera", translate=tuple(float(v) for v in pos),
                                                      orient_xyzw=usd_camera_xyzw(ros), camera=camera_lens(cam))
        for mark, pose in (("robot_start", site.robot_start), ("robot_goal", site.robot_goal)):
            plan[f"{base}/{mark}"] = PlannedPrim(
                "disc", translate=(pose.x, pose.y, mark_z + MARK_LAYERS.index(mark) * MARK_THICKNESS_M),
                scale=(2 * MARK_RADIUS_M, 2 * MARK_RADIUS_M, MARK_THICKNESS_M), color=MARK_COLORS[mark])
        (x0, x1), (y0, y1) = site.human_x_m, site.human_y_m
        plan[f"{base}/human_region"] = PlannedPrim(
            "plate", translate=((x0 + x1) / 2, (y0 + y1) / 2,
                                mark_z + MARK_LAYERS.index("human_region") * MARK_THICKNESS_M),
            scale=(x1 - x0 + 2 * REGION_MARGIN_M, y1 - y0 + 2 * REGION_MARGIN_M, MARK_THICKNESS_M),
            color=MARK_COLORS["human_region"])
    return plan


def rest_record(robot: RobotRest) -> dict[str, object]:
    """What the Stage entry keeps under REST_KEY: the numbers robot_rest() measured, as they are."""
    return {"lowest_z_m": float(robot.lowest_z_m), "root_link_m": [float(v) for v in robot.root_link_m]}


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def pinned_rest(assets: Mapping) -> RobotRest:
    """The rest numbers the canon's Stage entry keeps (rest_record). Raises ValueError when it keeps none, or
    not a lowest point and a root link position of finite numbers."""
    entry = (assets.get("scene") or {}).get(STAGE_KEY)
    rest = entry.get(REST_KEY) if isinstance(entry, Mapping) else None
    link = rest.get("root_link_m") if isinstance(rest, Mapping) else None
    if not (isinstance(rest, Mapping) and set(rest) == {"lowest_z_m", "root_link_m"}
            and _finite(rest["lowest_z_m"]) and isinstance(link, list) and len(link) == 3
            and all(_finite(v) for v in link)):
        raise ValueError(f"meta/assets.json 의 {STAGE_KEY}.{REST_KEY} 가 없거나 lowest_z_m(수 하나) · "
                         f"root_link_m(수 셋)이 아닙니다: {rest!r} — build_stage.py 가 지으며 찍는 값을 적으십시오")
    return RobotRest(float(rest["lowest_z_m"]), tuple(float(v) for v in link))


def _plain(value: object) -> object:
    """A plan value as JSON, every number rounded to PLAN_DIGITS (-0.0 as 0.0)."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    return round(float(value), PLAN_DIGITS) + 0.0


def _one_sign(q) -> list[float]:
    """A quaternion rounded, its sign chosen so its first non-zero component is positive: q and -q are one
    turn, and which of the two a computation lands on can hang on a last bit."""
    r = _plain(q)
    lead = next((v for v in r if v != 0.0), 1.0)
    return r if lead > 0 else [-v + 0.0 for v in r]


def plan_record(plan: Mapping[str, PlannedPrim], robot: RobotRest) -> dict[str, object]:
    """The plan as JSON: every prim in its order (the order write_stage authors them in) with each field of its
    PlannedPrim -- numbers rounded to PLAN_DIGITS, orientations of one sign -- beside the rest numbers it was
    laid out for (rounded alike) and STAGE_VERSION."""
    prims = []
    for path, p in plan.items():
        row = {f.name: _plain(getattr(p, f.name)) for f in dataclasses.fields(p)}
        row["orient_xyzw"] = _one_sign(p.orient_xyzw)
        prims.append([path, row])
    return {"stage_version": STAGE_VERSION, REST_KEY: _plain(rest_record(robot)), "prims": prims}


def plan_digest(plan: Mapping[str, PlannedPrim], robot: RobotRest) -> str:
    """SHA-256 of plan_record as compact JSON with sorted keys -- what the Stage entry pins as PLAN_KEY."""
    blob = json.dumps(plan_record(plan, robot), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def pin_problems(scenarios: Mapping, cameras_canon: Mapping, assets: Mapping) -> list[str]:
    """Whether the canon's Stage entry still pins the Stage the canon lays out today, without pxr: stage_plan()
    of today's canon -- the robot at the rest numbers the entry keeps -- hashed (plan_digest) against the
    entry's PLAN_KEY. [] when they agree. What shows: a site, the carry block (the box's place and the arms' pose), a
    camera, the name an asset ships under, the scene spec the marks and the CCTVs are placed by, H1's joint frames,
    a placement number here, changed since the Stage was built.
    What does not: how write_stage turns a plan into USD (check_stage.py builds again and compares the bytes)
    and the rest numbers themselves (check_stage.py measures h1.usd again)."""
    entry = (assets.get("scene") or {}).get(STAGE_KEY)
    if not isinstance(entry, Mapping):
        return [f"meta/assets.json 의 scene 에 {STAGE_KEY} 가 없습니다"]
    pinned = entry.get(PLAN_KEY)
    if not (isinstance(pinned, str) and re.fullmatch(r"[0-9a-f]{64}", pinned)):
        return [f"meta/assets.json 의 {STAGE_KEY}.{PLAN_KEY} 가 없거나 SHA-256 이 아닙니다: {pinned!r} — "
                "build_stage.py 가 지으며 찍는 값을 적으십시오"]
    try:
        rest = pinned_rest(assets)
        got = plan_digest(stage_plan(scenarios, cameras_canon, assets, robot=rest), rest)
    except ValueError as exc:
        return [str(exc)]
    if got != pinned:
        return [f"{STAGE_KEY}: 지금 정본으로 놓은 Stage 의 배치(해시 {got})가 고정한 {PLAN_KEY} {pinned} 와 "
                "다릅니다 — 시나리오의 자리 · 든 상자의 자리와 팔 자세(carry) · 카메라 정본 · 엮는 자산이 실리는 이름 · "
                "창고 명세 · H1 의 관절 틀 · build_stage.py 의 놓는 값 가운데 무엇이 Stage 를 지은 뒤에 바뀌었습니다. "
                "build_stage.py 로 다시 지어 usd_sha256 · "
                "usd_checked · plan_sha256 · robot_rest 를 고치고 check_stage.py 로 확인하십시오"]
    return []


def root_link_name(repo_root: Path = REPO, robot_key: str = ROBOT_KEY) -> str:
    """The robot's articulation root link -- the body the simulation reports the robot's pose by -- as the
    joint map extracted from the asset records it (provenance.articulation_root)."""
    doc = yaml.safe_load((Path(repo_root) / "meta" / "joint_map" / f"{robot_key}.yaml").read_text(encoding="utf-8"))
    root = str(((doc or {}).get("provenance") or {}).get("articulation_root") or "")
    if not root.strip("/"):
        raise ValueError(f"meta/joint_map/{robot_key}.yaml 에 provenance.articulation_root 가 없습니다")
    return PurePosixPath(root).name


# ---- the USD (pxr) ----------------------------------------------------------------------------

def robot_rest(usd: Path, root_link: str) -> RobotRest:
    """Measure the robot's USD at its rest pose: the lowest point of its rendered geometry (default and
    render purpose -- the collision shapes are guides) and its root link's position, both in its root
    prim's frame. Refuses a USD the plan's posed links cannot be authored on as they are: a rigid link that is not
    a child of the root prim or whose transform ops are not LINK_OP alone, a root link turned in its root prim."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd))
    if not stage:
        raise ValueError(f"{usd}: USD 를 열 수 없습니다")
    root = stage.GetDefaultPrim()
    link = stage.GetPrimAtPath(root.GetPath().AppendChild(root_link)) if root else None
    if not root or not link:
        raise ValueError(f"{usd}: 기본 프림이나 그 아래의 뿌리 링크 {root_link} 가 없습니다")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                              useExtentsHint=False)
    lowest = cache.ComputeUntransformedBound(root).ComputeAlignedRange().GetMin()[2]
    t = Usd.TimeCode.Default()
    rel = (UsdGeom.Xformable(link).ComputeLocalToWorldTransform(t)
           * UsdGeom.Xformable(root).ComputeLocalToWorldTransform(t).GetInverse())
    turn = np.array([[rel[i][j] for j in range(3)] for i in range(3)])
    if np.max(np.abs(turn - np.eye(3))) > ROOT_LINK_TURN_TOL:
        raise ValueError(f"{usd}: 뿌리 링크 {root_link} 가 뿌리 프림에 대해 돌아 있습니다 — Stage 의 팔 링크는 뿌리 "
                         "링크의 자리만으로 뿌리 프림 기준에 놓습니다")
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        ops = UsdGeom.Xformable(prim).GetXformOpOrderAttr().Get()
        if prim.GetParent() != root or list(ops or []) != [LINK_OP]:
            raise ValueError(f"{usd}: 링크 {prim.GetPath()} 가 뿌리 프림 바로 아래에 변환 {LINK_OP} 하나로 놓인 프림이 "
                             f"아닙니다(부모 {prim.GetParent().GetPath()} · 변환 {list(ops or [])}) — Stage 의 팔 "
                             "링크는 그 변환을 뿌리 프림 기준으로 덮어씁니다")
    return RobotRest(float(lowest), tuple(float(v) for v in rel.ExtractTranslation()))


def _matrix4d(p: PlannedPrim):
    """The USD matrix (row vectors, the translation in the last row) of a planned prim's translate and orient."""
    from pxr import Gf

    rot = kinematics.quat_to_matrix(p.orient_xyzw)
    rows = [[*map(float, rot[:, i]), 0.0] for i in range(3)] + [[*map(float, p.translate), 1.0]]
    return Gf.Matrix4d(*rows)


def _author(path: Path, plan: Mapping[str, PlannedPrim]) -> None:
    """Every prim but the references, on a stage that composes nothing yet."""
    from pxr import Gf, Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    layer = stage.GetRootLayer()
    for prim_path, p in plan.items():
        if p.kind == "over":
            spec = Sdf.CreatePrimInLayer(layer, prim_path)     # an over: the link comes from the robot's USD
            if p.posed:                                        # its one transform op, in the robot's root prim
                Sdf.AttributeSpec(spec, LINK_OP, Sdf.ValueTypeNames.Matrix4d).default = _matrix4d(p)
            continue
        if p.kind == "camera":
            # Authored as a spec, not with Define: Define would turn the over above it into a def
            spec = Sdf.CreatePrimInLayer(layer, prim_path)
            spec.specifier = Sdf.SpecifierDef
            spec.typeName = "Camera"
            prim = stage.GetPrimAtPath(prim_path)
            cam = UsdGeom.Camera(prim)
            lens = p.camera
            cam.CreateFocalLengthAttr(float(lens["focal_length_mm"]))
            cam.CreateHorizontalApertureAttr(float(lens["horizontal_aperture_mm"]))
            cam.CreateVerticalApertureAttr(float(lens["vertical_aperture_mm"]))
            cam.CreateClippingRangeAttr(Gf.Vec2f(*map(float, lens["clipping_range_m"])))
        elif p.kind == "scope":
            UsdGeom.Scope.Define(stage, prim_path)
            continue
        elif p.kind in ("xform", "reference"):
            prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
        elif p.kind == "disc":
            disc = UsdGeom.Cylinder.Define(stage, prim_path)
            disc.CreateRadiusAttr(0.5)
            disc.CreateHeightAttr(1.0)
            disc.CreateAxisAttr(UsdGeom.Tokens.z)
            prim = disc.GetPrim()
        elif p.kind == "plate":
            plate = UsdGeom.Cube.Define(stage, prim_path)
            plate.CreateSizeAttr(1.0)
            prim = plate.GetPrim()
        else:
            raise ValueError(f"{prim_path}: 모르는 종류 {p.kind!r}")
        if p.kind in ("disc", "plate"):
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
            gprim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*map(float, p.color))])
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, p.translate)))
        x, y, z, w = (float(v) for v in p.orient_xyzw)
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        if p.scale is not None:
            xf.AddScaleOp().Set(Gf.Vec3f(*map(float, p.scale)))
    world = stage.GetPrimAtPath(ROOT)
    stage.SetDefaultPrim(world)
    world.SetCustomDataByKey("keti:builtBy", BUILT_BY)
    world.SetCustomDataByKey("keti:stageVersion", STAGE_VERSION)
    layer.Save()


def write_stage(path: Path, plan: Mapping[str, PlannedPrim]) -> None:
    """Write ``plan`` as the USD at ``path`` (usdc, z up, metres, default prim /World). The references are
    added to the saved layer after the stage that authored it is gone, so nothing tries to open them here --
    they resolve inside a release's assets/, not beside the build. Same plan, same bytes."""
    from pxr import Sdf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    _author(path, plan)
    layer = Sdf.Layer.FindOrOpen(str(path))
    for prim_path, p in plan.items():
        if p.kind == "reference":
            layer.GetPrimAtPath(prim_path).referenceList.prependedItems.append(Sdf.Reference(p.ref))
    layer.Save()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def robot_usd(repo_root: Path = REPO) -> Path:
    """The fetched copy of the robot's USD (generation/robots/fetch_robot.py), its folder at the bytes the
    canon pins -- what the root height is measured on."""
    assets = load_assets(repo_root)
    entry = assets["robots"][ROBOT_KEY]
    files = robot_ship_files(ROBOT_KEY, entry, repo_root)
    check_shipped_sources(files, repo_root)
    name = Path(entry["usd_path"]).name
    return next(f.src for rel, f in files.items() if PurePosixPath(rel).name == name)


def canon_inputs(repo_root: Path = REPO) -> tuple[dict, dict, dict]:
    """stage_plan()'s canon in ``repo_root``: meta/scenarios.json, meta/cameras.json (cameras.load) and
    meta/assets.json."""
    root = Path(repo_root)
    scenarios = json.loads((root / "meta" / "scenarios.json").read_text(encoding="utf-8"))
    return scenarios, cam_canon.load(root / "meta" / "cameras.json"), load_assets(root)


def canon_rest(repo_root: Path = REPO) -> RobotRest:
    """robot_rest() of the robot's fetched copy, at the root link its joint map names."""
    return robot_rest(robot_usd(repo_root), root_link_name(repo_root))


def canon_plan(rest: RobotRest | None = None) -> dict[str, PlannedPrim]:
    """stage_plan() of the repository's canon, the robot measured on its fetched copy (or at ``rest``)."""
    return stage_plan(*canon_inputs(REPO), robot=canon_rest(REPO) if rest is None else rest)


def build(out: Path = OUT) -> Path:
    write_stage(out, canon_plan())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    rest = canon_rest(REPO)
    plan = canon_plan(rest)
    write_stage(a.out, plan)
    print(f"{a.out}  sha256 {_sha256(a.out)}")
    # What the Stage entry pins beside usd_sha256 -- pin_problems() reads them back without pxr.
    print(f"{REST_KEY} {json.dumps(rest_record(rest))}")
    print(f"{PLAN_KEY} {plan_digest(plan, rest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
