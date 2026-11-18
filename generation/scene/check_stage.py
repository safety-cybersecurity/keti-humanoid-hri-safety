#!/usr/bin/env python3
"""Open the built Stage the way a release holds it, and say what is wrong. Run with Isaac's python (pxr):

    PYTHONPATH= ~/isaacsim/python.sh generation/scene/check_stage.py

lay_release() puts in a temporary directory every file generation.assets.shipped_asset_files lays in the
assets/ of a release whose episodes use everything the Stage composes -- the Stage itself, the warehouse,
the vested dummy, the carried box, the robot's folder and the textures -- each from the tree at the bytes
the canon pins. check() opens the Stage there and asks:

- the canon's Stage entry composes what build_stage composes (build_stage.composes_problems);
- every layer and every asset the Stage needs (UsdUtils.ComputeAllDependencies) is a file of that assets/,
  and the only thing left unresolved is the renderer's own material library (OmniPBR.mdl);
- every prim of the plan is there, defined, of its kind -- a link a camera hangs under comes from the
  robot's USD, not from the Stage;
- each robot camera sits on its link at the canon offset, looking along the canon's optical axis, and each
  site camera where its placement puts it for the robot at that site's start (each side's under that side's scope
  where the site places it per side the person comes from) -- the robot's root link read from the composed Stage,
  not from the builder's measurement;
- each arm link the plan poses sits where the plan puts it and where generation/robots/kinematics.py puts it at
  the carry pose from the composed root link, and the box where generation/carry.py puts it on the composed torso
  (CARRY_TOL) -- a wrist camera moves with its forearm, the camera check above reads the composed link; the links
  so compared are the links the hold pose moves (build_stage.held_links), each once;
- the robot and the dummy stand on the floor (their lowest rendered point within FLOOR_TOLERANCE_M);
- the rest numbers the canon's Stage entry keeps are those measured on the fetched h1.usd (within REST_TOL_M),
  and its plan_sha256 is the plan the canon lays out today (build_stage.pin_problems -- what the tests check
  without pxr);
- building again gives the same bytes.

Exit 0 when nothing is wrong, 1 otherwise.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation import cameras as cam_canon  # noqa: E402
from generation import carry as carry_canon  # noqa: E402
from generation import sites  # noqa: E402
from generation.assets import (ASSETS_DIR, COMPOSES_KEY, check_shipped_sources, load_assets,  # noqa: E402
                               shipped_asset_files)
from generation.robots import kinematics  # noqa: E402
from generation.scene import build_stage, layout  # noqa: E402
from generation.scene.carried_box import NOMINAL_SIZE_M  # noqa: E402

# What a USD of the release may leave unresolved: a bare MDL name the renderer finds on its own search path.
RENDERER_LIBRARY = {"OmniPBR.mdl"}
# How far the lowest rendered point of the robot or the dummy may be from the floor.
FLOOR_TOLERANCE_M = 0.05
# Pose agreement between the composed Stage and the canon (m, and on unit vectors).
POSE_TOL = 1e-5
# How far the rest numbers the canon keeps may be from those measured on the fetched h1.usd (m).
REST_TOL_M = 1e-9
# How far a posed arm link or the box may be from the plan and from the kinematics (m, and on unit vectors). The
# joint frames are float32 in the robot's USD: a link the kinematics places is within 2e-7 of where the USD has it.
CARRY_TOL = 1e-6
KINDS = {"camera": "Camera", "disc": "Cylinder", "plate": "Cube", "scope": "Scope", "xform": "Xform",
         "reference": "Xform"}


def lay_release(repo_root: Path, dest: Path) -> Path:
    """Copy into ``dest`` what a release that ships the Stage holds under assets/ (shipped_asset_files for a
    batch whose episodes reference everything the canon's Stage entry composes), every file first checked
    against its pin (check_shipped_sources). Returns the Stage's path there. Raises ValueError /
    FileNotFoundError when the canon does not ship the Stage or a file is missing or not its pinned bytes."""
    assets = load_assets(repo_root)
    entry = (assets.get("scene") or {}).get(build_stage.STAGE_KEY)
    if not isinstance(entry, dict) or COMPOSES_KEY not in entry:
        raise ValueError(f"meta/assets.json 의 scene 에 composes 가 있는 {build_stage.STAGE_KEY} 가 없습니다")
    by_key = {k: e for g in ("robots", "scene") for k, e in (assets.get(g) or {}).items()}
    referenced = [by_key[key]["usd_path"] for key in entry[COMPOSES_KEY] if key in by_key]
    files = shipped_asset_files(assets, referenced, repo_root)
    stage_rel = f"{ASSETS_DIR}/{Path(entry['usd_path']).name}"
    if stage_rel not in files:
        raise ValueError(f"그것이 엮는 자산을 모두 싣는 배포본에 {stage_rel} 가 실리지 않습니다 — 정본의 재배포 표시를 보십시오")
    check_shipped_sources(files, repo_root)
    for rel, f in sorted(files.items()):
        out = Path(dest) / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f.src, out)
    return Path(dest) / stage_rel


def _matrix(prim):
    from pxr import Usd, UsdGeom

    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), float)


def _camera_axes(prim) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(position, viewing direction, image up) of a USD camera prim in the world: it looks along its -z."""
    m = _matrix(prim)                                   # row vectors: rows 0-2 the axes, row 3 the origin
    return m[3, :3], -m[2, :3] / np.linalg.norm(m[2, :3]), m[1, :3] / np.linalg.norm(m[1, :3])


def _ros_axes(pos, quat_xyzw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(position, +z, -y) of a ROS optical frame: where it looks and its image up."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return np.asarray(pos, float), r[:, 2], -r[:, 1]


def _link_pose(prim) -> np.ndarray:
    """pos3 + quat xyzw of a link prim in the world (mounted_pose's parent pose)."""
    from pxr import Gf

    m = _matrix(prim)
    q = Gf.Matrix4d(*m.ravel().tolist()).ExtractRotationQuat()
    im = q.GetImaginary()
    return np.array([*m[3, :3], im[0], im[1], im[2], q.GetReal()])


def _lowest_z(stage, path: str) -> float:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                              useExtentsHint=False)
    return float(cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange().GetMin()[2])


def rest_problems(assets, measured: build_stage.RobotRest) -> list[str]:
    """The canon's Stage entry against the rest numbers measured on the fetched robot: [] when they are within
    REST_TOL_M -- the numbers pin_problems lays the plan out with are then the robot's."""
    try:
        kept = build_stage.pinned_rest(assets)
    except ValueError as exc:
        return [str(exc)]
    got = (measured.lowest_z_m, *measured.root_link_m)
    want = (kept.lowest_z_m, *kept.root_link_m)
    if max(abs(a - b) for a, b in zip(got, want)) > REST_TOL_M:
        return [f"{build_stage.STAGE_KEY}.{build_stage.REST_KEY} {build_stage.rest_record(kept)} 가 받아 둔 "
                f"h1.usd 에서 잰 값 {build_stage.rest_record(measured)} 와 다릅니다 — build_stage.py 로 다시 지어 "
                "찍히는 값으로 고치십시오"]
    return []


def _close(a, b) -> bool:
    return bool(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))) <= POSE_TOL)


def _dependency_problems(stage_path: Path) -> list[str]:
    from pxr import UsdUtils

    base = stage_path.parent.resolve()
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(stage_path))
    out = []
    for layer in layers:
        real = Path(layer.realPath).resolve()
        if not real.is_relative_to(base):
            out.append(f"레이어 {layer.identifier} 가 배포본의 assets/ 밖에서 풀립니다: {real}")
    for path in assets:
        if not Path(path).resolve().is_relative_to(base):
            out.append(f"자산 {path} 가 배포본의 assets/ 밖에서 풀립니다")
    out += [f"풀리지 않는 의존: {u}" for u in sorted(set(unresolved) - RENDERER_LIBRARY)]
    names = sorted(Path(layer.realPath).resolve().relative_to(base).as_posix()
                   for layer in layers if Path(layer.realPath).resolve().is_relative_to(base))
    print(f"layers {names}")
    print(f"assets {len(assets)} · unresolved {sorted(unresolved)}")
    return out


def _prim_problems(stage, plan) -> list[str]:
    out = []
    for path, p in plan.items():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            out.append(f"{path}: 프림이 없습니다")
            continue
        if not prim.IsDefined():
            why = "로봇 USD 가 이 링크를 가져오지 않습니다" if p.kind == "over" else "정의되지 않았습니다"
            out.append(f"{path}: {why}")
            continue
        want = KINDS.get(p.kind)
        if want is not None and prim.GetTypeName() != want:
            out.append(f"{path}: 종류가 {prim.GetTypeName()!r} — {want!r} 여야 합니다")
        if p.kind == "reference" and not prim.GetChildren():
            out.append(f"{path}: 참조 {p.ref} 가 아무것도 가져오지 않았습니다")
    return out


def _axes_problems(path: str, got, want, what_want: str) -> list[str]:
    return [f"{path}: {what} {np.round(a, 6).tolist()} — {what_want} {np.round(b, 6).tolist()}"
            for what, a, b in zip(("자리", "보는 쪽", "그림의 위"), got, want) if not _close(a, b)]


def _camera_problems(stage, cameras_canon, scenarios) -> list[str]:
    root = build_stage.ROOT
    out = []
    for name, cam in cameras_canon["cameras"].items():
        if cam["type"] != "mounted":
            continue
        cam_path = f"{build_stage.ROOT}/Robot/{cam['parent_prim']}/{name}"
        link = stage.GetPrimAtPath(f"{build_stage.ROOT}/Robot/{cam['parent_prim']}")
        prim = stage.GetPrimAtPath(cam_path)
        if not link or not prim:
            continue                                     # said by _prim_problems
        pos, ros = cam_canon.mounted_pose(_link_pose(link), cam["offset_pos"], cam["offset_rot_xyzw"])
        out += _axes_problems(cam_path, _camera_axes(prim), _ros_axes(pos, ros), "정본은")
    # The site cameras are placed for the robot at each site's start: its root link where the composed Stage
    # has it relative to the robot's prim, turned to that start.
    robot = stage.GetPrimAtPath(f"{root}/Robot")
    link = stage.GetPrimAtPath(f"{root}/Robot/{build_stage.root_link_name()}")
    if not robot or not link:
        return out + [f"{root}/Robot 이나 그 뿌리 링크가 없어 자리 카메라를 대조하지 못합니다"]
    m_robot = _matrix(robot)
    offset = (_matrix(link) @ np.linalg.inv(m_robot))[3, :3]          # the root link in the robot's frame
    spec = layout.load_spec(layout.SCENE_SPEC)
    for scenario in scenarios["scenarios"]:
        if not isinstance((scenario.get("scene") or {}).get("site"), dict):
            continue
        site = sites.load_site(scenario)
        sid, yaw = site.scenario_id, np.radians(site.robot_start.yaw_deg)
        c, s = np.cos(yaw), np.sin(yaw)
        root_link = (site.robot_start.x + c * offset[0] - s * offset[1],
                     site.robot_start.y + s * offset[0] + c * offset[1], m_robot[3, 2] + offset[2])
        for name, cam in cameras_canon["cameras"].items():
            if cam["type"] not in ("follow", "fixed"):
                continue
            # one camera per side the person comes from where the site places it per side, under that side's scope
            for side in cam_canon.site_sides(cam, sid, name=name) or (None,):
                if cam["type"] == "follow":
                    placement = cam_canon.site_placement(cam, sid, name=name, approach_side=side)
                    eye, target = cam_canon.follow_pose(placement, root_link, yaw, yaw)
                    pos, ros = cam_canon.look_at_pose(eye, target)
                else:
                    pos, ros = cam_canon.fixed_pose(cam, spec, sid, approach_side=side)
                under = f"{root}/Sites/{sid}" if side is None else \
                    f"{root}/Sites/{sid}/{build_stage.SIDE_SCOPE.format(side=side)}"
                path = f"{under}/{name}"
                prim = stage.GetPrimAtPath(path)
                if not prim:
                    continue                             # said by _prim_problems
                got = _camera_axes(prim)
                out += _axes_problems(path, got, _ros_axes(pos, ros), "정본은")
                if got[2][2] <= 0:
                    out.append(f"{path}: 그림의 위가 아래를 향합니다 {np.round(got[2], 3).tolist()}")
    return out


def _pose_of(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(position, rotation matrix acting on column vectors) of a USD matrix (row vectors)."""
    return m[3, :3], m[:3, :3].T


def _pose_gap(a, b) -> float:
    """The larger of the position gap (m) and the largest rotation matrix entry gap of two (position, rotation)."""
    return max(float(np.max(np.abs(np.asarray(a[0]) - np.asarray(b[0])))),
               float(np.max(np.abs(np.asarray(a[1]) - np.asarray(b[1])))))


def held_link_problems(checked, held) -> list[str]:
    """The arm links the Stage check compared (``checked``, by link name) against the links the hold pose moves
    (``held`` -- build_stage.held_links): each of them once and nothing else. A comparison of none is a problem too --
    every held link is missing from it."""
    checked, held = list(checked), list(held)
    missing = [name for name in held if name not in checked]
    extra = sorted(set(checked) - set(held))
    twice = sorted({name for name in checked if checked.count(name) > 1})
    out = []
    if missing or extra:
        out.append(f"팔 자세를 대조한 링크 {len(checked)} 개가 잡은 자세가 움직이는 링크 {len(held)} 개와 다릅니다 — "
                   f"빠진 것 {missing} · 그 밖의 것 {extra}")
    if twice:
        out.append(f"팔 자세를 두 번 대조한 링크가 있습니다: {twice}")
    return out


def _carry_problems(stage, plan, scenarios) -> list[str]:
    """The arm links the plan poses and the box, as the Stage composes them, against the plan and against the
    kinematics: the links from the composed root link at the carry pose, the box from the composed torso. The links
    compared are the links the hold pose moves (held_link_problems)."""
    root = build_stage.ROOT
    robot = stage.GetPrimAtPath(f"{root}/Robot")
    held = build_stage.carry_of(scenarios)
    joints = kinematics.load_joints()
    hold = carry_canon.arm_hold_joints(held, joints["joints"])
    ref = kinematics.link_poses(joints, hold, frame=joints["root_link"])
    root_link = stage.GetPrimAtPath(f"{root}/Robot/{joints['root_link']}")
    if not robot or not root_link:
        return [f"{root}/Robot 이나 그 뿌리 링크가 없어 팔 자세를 대조하지 못합니다"]
    to_robot = np.linalg.inv(_matrix(robot))
    base_pos, base_rot = _pose_of(_matrix(root_link) @ to_robot)
    out, worst, checked = [], 0.0, []
    for path, p in plan.items():
        prim = stage.GetPrimAtPath(path)
        if not p.posed or not prim:
            continue                                     # a missing prim is said by _prim_problems
        checked.append(path.rsplit("/", 1)[1])
        got = _pose_of(_matrix(prim) @ to_robot)
        pos, quat = ref[checked[-1]]
        for what, want in (("계획", (p.translate, kinematics.quat_to_matrix(p.orient_xyzw))),
                           ("순기구학", (base_pos + base_rot @ pos, base_rot @ kinematics.quat_to_matrix(quat)))):
            gap = _pose_gap(got, want)
            worst = max(worst, gap)
            if gap > CARRY_TOL:
                out.append(f"{path}: 팔 링크의 자리가 {what}과 {gap:.2e} 다릅니다(허용 {CARRY_TOL})")
    out += held_link_problems(checked, build_stage.held_links(joints, hold))
    box = stage.GetPrimAtPath(f"{root}/CarriedBox")
    torso = stage.GetPrimAtPath(f"{root}/Robot/{held.link}")
    if not box or not torso:
        return out + [f"{root}/CarriedBox 나 {held.link} 가 없어 상자 자리를 대조하지 못합니다"]
    got = _pose_of(_matrix(box))
    planned = plan[f"{root}/CarriedBox"]
    torso_pos, torso_rot = _pose_of(_matrix(torso))
    centre, quat = carry_canon.box_pose(torso_pos, kinematics.quat_from_matrix(torso_rot), held, NOMINAL_SIZE_M)
    box_worst = 0.0
    for what, want in (("계획", (planned.translate, kinematics.quat_to_matrix(planned.orient_xyzw))),
                       ("몸통에 carry 를 더한 자리", (centre, kinematics.quat_to_matrix(quat)))):
        gap = _pose_gap(got, want)
        box_worst = max(box_worst, gap)
        if gap > CARRY_TOL:
            out.append(f"{root}/CarriedBox: 상자의 자리가 {what}와 {gap:.2e} 다릅니다(허용 {CARRY_TOL})")
    print(f"arm links at the carry pose: {len(checked)} (largest gap {worst:.1e}) · box on {held.link} "
          f"(largest gap {box_worst:.1e}) · box centre {np.round(got[0], 4).tolist()}")
    return out


def check(repo_root: Path = REPO) -> list[str]:
    """Everything wrong with the built Stage as a release holds it ([] when nothing is)."""
    from pxr import Usd

    repo_root = Path(repo_root)
    assets = load_assets(repo_root)
    problems = build_stage.composes_problems(assets)
    if problems:
        return problems
    built = repo_root / assets["scene"][build_stage.STAGE_KEY]["usd_path"]
    with tempfile.TemporaryDirectory(prefix="stage-check-") as tmp:
        tmp = Path(tmp)
        try:
            stage_path = lay_release(repo_root, tmp / "release")
        except (ValueError, FileNotFoundError) as exc:
            return [str(exc)]
        print(f"laid {sum(1 for q in (tmp / 'release').rglob('*') if q.is_file())} files under {ASSETS_DIR}/")
        problems += _dependency_problems(stage_path)
        stage = Usd.Stage.Open(str(stage_path))
        if not stage:
            return problems + [f"{stage_path.name} 를 열 수 없습니다"]
        rest = build_stage.canon_rest(repo_root)
        print(f"robot rest measured {build_stage.rest_record(rest)}")
        problems += rest_problems(assets, rest)
        plan = build_stage.canon_plan(rest)
        problems += _prim_problems(stage, plan)
        cameras_canon = cam_canon.load(repo_root / "meta" / "cameras.json")
        scenarios = json.loads((repo_root / "meta" / "scenarios.json").read_text(encoding="utf-8"))
        problems += _camera_problems(stage, cameras_canon, scenarios)
        problems += _carry_problems(stage, plan, scenarios)
        pin = build_stage.pin_problems(scenarios, cameras_canon, assets)
        problems += pin
        if not pin:
            print(f"{build_stage.PLAN_KEY} is the plan of today's canon "
                  f"({assets['scene'][build_stage.STAGE_KEY][build_stage.PLAN_KEY]})")
        for path in (f"{build_stage.ROOT}/Robot", f"{build_stage.ROOT}/Dummy"):
            if not stage.GetPrimAtPath(path):
                continue
            low = _lowest_z(stage, path)
            print(f"{path}: lowest rendered point z {low:.4f} m")
            if abs(low - build_stage.FLOOR_Z_M) > FLOOR_TOLERANCE_M:
                problems.append(f"{path}: 가장 낮은 점 z {low:.4f} m 가 바닥에서 {FLOOR_TOLERANCE_M} m 밖입니다")
        cams = sum(1 for p in plan.values() if p.kind == "camera")
        print(f"prims {len(plan)} · cameras {cams}")
        again = build_stage.build(tmp / "rebuild" / built.name)
        if again.read_bytes() != built.read_bytes():
            problems.append(f"다시 빌드한 {built.name} 의 바이트가 다릅니다")
        else:
            print(f"rebuild byte-identical ({built.stat().st_size} bytes)")
    return problems


def main() -> int:
    problems = check()
    for p in problems:
        print(f"PROBLEM {p}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
