#!/usr/bin/env python3
"""The H1's joint frames, read once from the pinned h1.usd (~/isaacsim/python.sh).

    ~/isaacsim/python.sh generation/robots/extract_joint_frames.py           # compare with the file
    ~/isaacsim/python.sh generation/robots/extract_joint_frames.py --write   # write the file

generation/robots/kinematics.py places every link of the H1 for a set of joint angles without Isaac. It needs,
for each joint, the two frames the physics engine pins together (a pose on the parent link and one on the child
link) and the axis the child turns about, and for each link the pose the asset stores it at. They are read here,
once, from the copy the simulation spawns -- the BSD-3 h1.usd that extract_link_boxes.py measures, opened the same
way -- and written to generation/robots/unitree_h1_joints.json, never typed by hand. Only USD stages are opened --
no environment, no physics.

A joint is a prim of the USD physics joint schema. A revolute joint records its axis, a fixed joint records none;
a joint of any other kind (prismatic, spherical, D6) is a problem, because the kinematics cannot place its child.
``rest`` holds the links' stored poses relative to the root link, the one link no joint has as its child. They are
what the asset itself says, so a check of the joint frames at all angles zero against them does not rest on the
frames: the extraction makes it, and tests/generation/test_kinematics.py makes it again from the written file.

The joint names have to be the joint map's (meta/joint_map/<robot>.yaml); the file records the SHA-256 of the h1.usd
it was read from, and that has to be the one meta/assets.json pins -- the script stops otherwise.

Check mode (the default) exits 1 when the file differs from a fresh reading by more than CHECK_TOL in any number,
has other joints or links, or records another h1.usd. --write writes the file only when the reading has no
problem, as json.dumps(indent=2, ensure_ascii=False) writes it, joints and links sorted by name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from generation.envs.s1d import extract_camera_mounts  # noqa: E402
from generation.robots import extract_link_boxes, kinematics  # noqa: E402

JOINTS_FILE = kinematics.JOINTS
JOINT_MAP_DIR = REPO / "meta" / "joint_map"
ASSETS = extract_link_boxes.ASSETS
ROBOT = extract_link_boxes.ROBOT
NOTE = ("H1 의 시뮬레이션이 띄우는 사본(BSD-3 h1.usd)에서 관절마다 읽은 두 틀(부모 링크 쪽 pos0 · rot0 — 자식 링크 쪽 pos1 · rot1)과 "
        "회전축, 링크마다 저장된 자세를 뿌리 링크 기준으로 한 번 읽은 값 — 미터, 쿼터니언은 xyzw, 소수 9자리로 반올림. 자식 링크의 자세는 "
        "부모 ⊕ (pos0, rot0) ⊕ 축 둘레 회전 ⊕ (pos1, rot1)의 역입니다. 손으로 고치지 않고 "
        "generation/robots/extract_joint_frames.py --write 로 씁니다.")

DECIMALS = 9
CHECK_TOL = 10.0 ** -DECIMALS       # file vs reading: what rounding to 9 decimals can move
REST_TOL = 1e-6                     # the joint frames at all angles zero against the stored link poses (m and rad)


# ---- the document (numpy only) ------------------------------------------------------------------

def _rounded(values) -> list[float]:
    """A vector rounded to DECIMALS places as plain floats (no -0.0)."""
    return [float(v) + 0.0 for v in np.round(np.asarray(values, float), DECIMALS)]


def canonical_xyzw(q) -> np.ndarray:
    """The unit quaternion xyzw of ``q`` with a non-negative real part (the first non-zero part positive when the
    real part is zero), so the same rotation always reads as the same numbers."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    pivot = q[3] if abs(q[3]) > 1e-12 else next((v for v in q[:3] if abs(v) > 1e-12), 1.0)
    return -q if pivot < 0.0 else q


def joint_record(kind: str, parent: str, child: str, axis: str | None, pos0, rot0_xyzw, pos1, rot1_xyzw) -> dict:
    return {"type": kind, "parent": parent, "child": child, "axis": axis,
            "pos0": _rounded(pos0), "rot0_xyzw": _rounded(rot0_xyzw),
            "pos1": _rounded(pos1), "rot1_xyzw": _rounded(rot1_xyzw)}


def rest_record(pos, rot_matrix) -> dict:
    return {"pos": _rounded(pos), "rot_xyzw": _rounded(canonical_xyzw(kinematics.quat_from_matrix(rot_matrix)))}


def document(usd_sha256: str, root: str, joints: Mapping[str, dict], rest: Mapping[str, dict]) -> dict:
    """The file's content, joints and links sorted by name."""
    return {"robot": ROBOT, "usd_sha256": usd_sha256, "root_link": root, "note": NOTE,
            "joints": {name: joints[name] for name in sorted(joints)},
            "rest": {link: rest[link] for link in sorted(rest)}}


def serialize(doc: Mapping) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def joint_map_names(robot: str = ROBOT) -> list[str]:
    """The joint names of the robot's joint map, in the map's order."""
    path = JOINT_MAP_DIR / f"{robot}.yaml"
    return list(yaml.safe_load(path.read_text(encoding="utf-8"))["joint_names"])


def rotation_gap(qa, qb) -> float:
    """The angle (rad) of the turn that takes one unit quaternion xyzw to the other, whatever their signs."""
    ra, rb = kinematics.quat_to_matrix(qa), kinematics.quat_to_matrix(qb)
    return float(np.arccos(np.clip((np.trace(ra.T @ rb) - 1.0) / 2.0, -1.0, 1.0)))


def document_problems(doc: Mapping, names: list[str]) -> list[str]:
    """What is wrong with a reading: joint names that are not the joint map's, a document that is not a tree of
    frames, and joint frames that at all angles zero do not put the links where the asset stores them."""
    problems = []
    revolute = sorted(n for n, j in doc["joints"].items() if j["type"] == "revolute")
    problems += [f"관절 매핑에 있는데 USD 에는 없는 회전 관절: {n}" for n in names if n not in revolute]
    problems += [f"USD 에는 있는데 관절 매핑에 없는 회전 관절: {n}" for n in revolute if n not in names]
    try:
        joints = kinematics.joints_from_document(doc)
    except ValueError as exc:
        return problems + [f"관절 틀이 링크의 나무가 아닙니다: {exc}"]
    poses = kinematics.link_poses(joints, {}, frame=doc["root_link"])
    for link, stored in joints["rest"].items():
        pos, quat = poses[link]
        shift, turn = float(np.max(np.abs(pos - stored["pos"]))), rotation_gap(quat, stored["rot_xyzw"])
        if shift > REST_TOL or turn > REST_TOL:
            problems.append(f"링크 {link}: 모든 관절 0 에서 관절 틀이 낸 자세가 USD 에 저장된 자세와 위치 {shift:.2e} m · "
                            f"회전 {turn:.2e} rad 다릅니다")
    return problems


def compare(canon: Mapping, fresh: Mapping) -> list[str]:
    """What differs between the file (``canon``) and a fresh reading: the robot, the h1.usd, the root link, the
    sets of joints and links, a joint's kind, parent, child or axis, and any number by more than CHECK_TOL.
    The note is prose and is not compared."""
    problems = []
    for key in ("robot", "usd_sha256", "root_link"):
        if canon.get(key) != fresh.get(key):
            problems.append(f"{key}: 파일 {canon.get(key)!r} · 읽은 값 {fresh.get(key)!r}")
    for section, numbers in (("joints", ("pos0", "rot0_xyzw", "pos1", "rot1_xyzw")), ("rest", ("pos", "rot_xyzw"))):
        have, want = canon.get(section) or {}, fresh[section]
        what = "관절" if section == "joints" else "링크"
        problems += [f"{what} {n}: 파일에 있는데 읽은 값에는 없습니다" for n in have if n not in want]
        problems += [f"{what} {n}: 읽은 값에는 있는데 파일에 없습니다" for n in want if n not in have]
        for name in want:
            if name not in have:
                continue
            for key in (("type", "parent", "child", "axis") if section == "joints" else ()):
                if have[name].get(key) != want[name][key]:
                    problems.append(f"{what} {name}: {key} 가 파일 {have[name].get(key)!r} · 읽은 값 {want[name][key]!r}")
            for key in numbers:
                try:
                    gap = float(np.max(np.abs(np.asarray(have[name][key], float) - np.asarray(want[name][key], float))))
                except (KeyError, TypeError, ValueError):
                    problems.append(f"{what} {name}: 파일의 {key} 를 읽을 수 없습니다")
                    continue
                if round(gap, 12) > CHECK_TOL:
                    problems.append(f"{what} {name}: {key} 가 읽은 값과 {gap:.2e} 다릅니다 (파일 {have[name][key]} · "
                                    f"읽은 값 {want[name][key]})")
    return problems


# ---- the asset (pxr, inside Kit) -----------------------------------------------------------------

def _xyzw(quat) -> list[float]:
    """A pxr quaternion (real, imaginary) as xyzw."""
    im = quat.GetImaginary()
    return [float(im[0]), float(im[1]), float(im[2]), float(quat.GetReal())]


def read_joints(stage, links: list[str]) -> tuple[dict[str, dict], list[str]]:
    """({joint name: record}, the things the reading refuses) of every physics joint prim of a stage."""
    from pxr import UsdPhysics

    joints, problems = {}, []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        name = prim.GetName()
        if name in joints:
            problems.append(f"관절 {name}: 이름이 같은 프림이 둘입니다")
            continue
        joint = UsdPhysics.Joint(prim)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            kind, axis = "revolute", str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())
        elif prim.IsA(UsdPhysics.FixedJoint):
            kind, axis = "fixed", None
        else:
            problems.append(f"관절 {name}: {prim.GetTypeName()} 는 회전도 고정도 아니라 순기구학이 자식 링크를 놓지 못합니다")
            continue
        ends = []
        for side, rel in (("body0", joint.GetBody0Rel()), ("body1", joint.GetBody1Rel())):
            targets = [Path(str(t)).name for t in rel.GetTargets()]
            if len(targets) != 1 or targets[0] not in links:
                problems.append(f"관절 {name}: {side} 가 강체 링크 하나가 아닙니다: {[str(t) for t in rel.GetTargets()]}")
            ends.append(targets[0] if targets else None)
        if None in ends or any(e not in links for e in ends):
            continue
        joints[name] = joint_record(kind, ends[0], ends[1], axis,
                                    joint.GetLocalPos0Attr().Get(), _xyzw(joint.GetLocalRot0Attr().Get()),
                                    joint.GetLocalPos1Attr().Get(), _xyzw(joint.GetLocalRot1Attr().Get()))
    return joints, problems


def read_rest(stage, root: str, links: list[str]) -> tuple[dict[str, dict], list[str]]:
    """({link: stored pose relative to the root link}, the links whose stored pose is not a rigid one)."""
    rest, problems = {}, []
    for link in links:
        try:
            pos, rot = extract_camera_mounts.pose_in(stage, link, root)
        except ValueError as exc:
            problems.append(f"링크 {link}: {exc}")
            continue
        rest[link] = rest_record(pos, rot)
    return rest, problems


def root_link_of(joints: Mapping[str, dict], links: list[str]) -> tuple[str | None, list[str]]:
    """The one link that is no joint's child, or None and what is wrong."""
    children = {j["child"] for j in joints.values()}
    roots = [link for link in links if link not in children]
    if len(roots) != 1:
        return None, [f"뿌리 링크가 하나가 아닙니다 (어떤 관절의 child 도 아닌 링크: {roots})"]
    return roots[0], []


def read(stage) -> tuple[dict | None, list[str]]:
    """(the file's content for a stage, the problems of the reading). None when it cannot be built at all."""
    links = extract_link_boxes.rigid_links(stage)
    joints, problems = read_joints(stage, links)
    root, more = root_link_of(joints, links)
    problems += more
    if root is None:
        return None, problems
    rest, more = read_rest(stage, root, links)
    problems += more
    if problems:
        return None, problems
    return {"root_link": root, "joints": joints, "rest": rest}, []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="write the joint frames into the file")
    args = ap.parse_args()

    from generation.isaac_boot import launch_app

    simulation_app = launch_app({"headless": True})[1]
    code = 1
    try:
        from pxr import Usd

        from generation.envs.s1d.s1d_env_cfg import S1dEnvCfg

        url = S1dEnvCfg(use_cameras=False, use_skin=False).scene.robot.spawn.usd_path
        print(f"simulated copy: {url}")
        sha = hashlib.sha256(extract_link_boxes.opened_bytes(url)).hexdigest()
        pin = extract_link_boxes.pinned_sha256()
        print(f"h1.usd sha256 {sha}")
        if sha != pin:
            print(f"PROBLEM 열어 본 h1.usd 의 해시가 {ASSETS.name} 의 고정값과 다릅니다: {pin}\n1 problem(s)")
            raise SystemExit(1)
        stage = Usd.Stage.Open(url)
        if not stage:
            raise RuntimeError(f"could not open the simulated copy: {url}")
        names = joint_map_names()
        body, problems = read(stage)
        fresh = None
        if body is not None:
            fresh = document(sha, body["root_link"], body["joints"], body["rest"])
            for name, joint in fresh["joints"].items():
                print(f"{name}: {joint['type']} {joint['axis']} {joint['parent']} -> {joint['child']}")
            print(f"{len(fresh['joints'])} joints, {len(fresh['rest'])} links, root link {fresh['root_link']}")
            problems += document_problems(fresh, names)
        if fresh is not None and not args.write:
            if JOINTS_FILE.is_file():
                problems += compare(json.loads(JOINTS_FILE.read_text(encoding="utf-8")), fresh)
            else:
                problems.append(f"{JOINTS_FILE} 가 없습니다 — --write 로 쓰십시오")
        elif fresh is not None and not problems:        # a reading with a problem writes nothing
            JOINTS_FILE.write_text(serialize(fresh), encoding="utf-8")
            print(f"wrote {JOINTS_FILE} ({len(fresh['joints'])} joints, {len(fresh['rest'])} links)")
        for p in problems:
            print(f"PROBLEM {p}")
        print("OK" if not problems else f"{len(problems)} problem(s)")
        code = 0 if not problems else 1
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        simulation_app.close(exit_code=code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
