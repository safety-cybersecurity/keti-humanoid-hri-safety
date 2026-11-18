#!/usr/bin/env python3
"""The H1's own body as one box per link, measured once from the rendered meshes (~/isaacsim/python.sh).

    ~/isaacsim/python.sh generation/robots/extract_link_boxes.py           # compare with the file
    ~/isaacsim/python.sh generation/robots/extract_link_boxes.py --write   # write the file

generation/visibility.py lets the robot's own limbs hide the person from its cameras. It needs the body
as geometry it can place without Isaac, so each rigid link's rendered meshes are measured here, once, as
the axis-aligned box that bounds them in the link's own frame (centre and half sizes, metres, rounded to the
micrometre) and written to generation/robots/unitree_h1_link_boxes.json. The file is canon-like data, never
typed by hand; visibility.robot_obbs turns a link's pose into the box's place in the world.

The copy measured is the one the simulation spawns -- the BSD-3 h1.usd that generation/envs/s1d/
extract_camera_mounts.py calls "simulated", opened the same way -- and what renders is what the camera
sees: render-purpose, visible meshes, instance proxies included (extract_camera_mounts.rendered_meshes). The
file records the SHA-256 of the h1.usd it measured, and that has to be the one meta/assets.json pins; the
script stops otherwise. Only USD stages are opened -- no environment, no physics.

A link is a prim with the rigid body API. One with no rendered mesh gets no box (it is said, not an
error); one whose meshes are flat along an axis does not give a usable box and is a problem.

Check mode (the default) exits 1 when the file differs from a fresh measurement by more than CHECK_TOL_M
in any link's box, has other links, or records another h1.usd. --write writes the file only when the
measurement has no problem, as json.dumps(indent=2, ensure_ascii=False) writes it, links sorted by name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from generation.envs.s1d import extract_camera_mounts  # noqa: E402

LINK_BOXES = Path(__file__).resolve().parent / "unitree_h1_link_boxes.json"
ASSETS = REPO / "meta" / "assets.json"
ROBOT = "unitree_h1"
FRAME = "link"
NOTE = ("H1 의 렌더되는 사본(BSD-3 h1.usd)의 메시를 링크마다 링크 좌표의 축 정렬 상자로 한 번 잰 값 — 가운데와 반 크기, "
        "미터, 마이크로미터로 반올림. 손으로 고치지 않고 generation/robots/extract_link_boxes.py --write 로 씁니다.")

DECIMALS = 6                        # the micrometre
CHECK_TOL_M = 10.0 ** -DECIMALS     # file vs measurement: what rounding to the micrometre can move


# ---- geometry (numpy only) ----------------------------------------------------------------------

def link_box(points) -> tuple[np.ndarray, np.ndarray]:
    """(centre, half sizes) of the axis-aligned box that bounds ``points`` (N, 3) in whatever frame they are
    in, each rounded to the micrometre -- so the box can fall short of a point by at most that. No -0.0."""
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 3 or not len(pts):
        raise ValueError(f"점이 (N, 3) 배열(N ≥ 1)이 아닙니다: {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("점에 유한하지 않은 값이 있습니다")
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return np.round((lo + hi) / 2.0, DECIMALS) + 0.0, np.round((hi - lo) / 2.0, DECIMALS) + 0.0


def boxes_from_meshes(meshes_by_link: Mapping[str, Sequence[tuple]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{link: (centre, half)} of every link's meshes taken together; each mesh is
    (points in the link's frame, face vertex counts, face vertex indices), as rendered_meshes hands it over."""
    return {link: link_box(np.vstack([np.asarray(m[0], float).reshape(-1, 3) for m in meshes]))
            for link, meshes in meshes_by_link.items()}


# ---- the file ------------------------------------------------------------------------------------

def document(usd_sha256: str, boxes: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """The file's content, links sorted by name."""
    return {"robot": ROBOT, "usd_sha256": usd_sha256, "frame": FRAME, "note": NOTE,
            "links": {link: {"center": [float(v) for v in centre], "half_extents": [float(v) for v in half]}
                      for link, (centre, half) in sorted(boxes.items())}}


def serialize(doc: Mapping) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def box_problems(boxes: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> list[str]:
    """Links whose box the loader would refuse: a half size that is not above zero."""
    return [f"{link}: 반 크기 {half.tolist()} 가 모두 0 보다 크지 않습니다 — 한 축으로 납작한 메시는 가림 상자가 되지 못합니다"
            for link, (_, half) in boxes.items() if not (half > 0).all()]


def compare(canon: Mapping, fresh: Mapping) -> list[str]:
    """What differs between the file (``canon``) and a fresh measurement: the robot, the h1.usd, the frame, the
    set of links, and any centre or half size by more than CHECK_TOL_M. The note is prose and is not compared."""
    problems = []
    for key in ("robot", "usd_sha256", "frame"):
        if canon.get(key) != fresh.get(key):
            problems.append(f"{key}: 파일 {canon.get(key)!r} · 측정 {fresh.get(key)!r}")
    have, want = canon.get("links") or {}, fresh["links"]
    problems += [f"링크 {link}: 파일에 있는데 측정에는 없습니다" for link in have if link not in want]
    problems += [f"링크 {link}: 측정에는 있는데 파일에 없습니다" for link in want if link not in have]
    for link in want:
        if link not in have:
            continue
        for key in ("center", "half_extents"):
            try:
                gap = float(np.max(np.abs(np.asarray(have[link][key], float) - np.asarray(want[link][key], float))))
            except (KeyError, TypeError, ValueError):
                problems.append(f"링크 {link}: 파일의 {key} 를 읽을 수 없습니다")
                continue
            if round(gap, 9) > CHECK_TOL_M:
                problems.append(f"링크 {link}: {key} 가 측정과 {gap:.2e} m 다릅니다 (파일 {have[link][key]} · "
                                f"측정 {want[link][key]})")
    return problems


# ---- the asset (pxr, inside Kit) -----------------------------------------------------------------

def rigid_links(stage) -> list[str]:
    """The names of the rigid bodies of the stage, sorted."""
    from pxr import UsdPhysics

    return sorted({p.GetName() for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)})


def measure(stage) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """({link: (centre, half)}, the links with no rendered mesh) of a stage."""
    meshes, bare = {}, []
    for link in rigid_links(stage):
        try:
            meshes[link] = extract_camera_mounts.rendered_meshes(stage, link)
        except ValueError as exc:
            if not str(exc).startswith("no rendered mesh"):
                raise
            bare.append(link)
    return boxes_from_meshes(meshes), bare


def opened_bytes(url: str) -> bytes:
    """The bytes a stage open reads from ``url``: through omni.client, as USD's resolver does, for a remote
    address (pathlib would fold the // of https://), straight from disk for a path."""
    if "://" not in url:
        return Path(url).read_bytes()
    import omni.client

    result, _, content = omni.client.read_file(url)
    if result != omni.client.Result.OK:
        raise RuntimeError(f"could not read {url}: {result}")
    return bytes(memoryview(content))


def pinned_sha256() -> str:
    return json.loads(ASSETS.read_text(encoding="utf-8"))["robots"][ROBOT]["usd_sha256"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="write the measured boxes into the file")
    args = ap.parse_args()

    from generation.isaac_boot import launch_app

    simulation_app = launch_app({"headless": True})[1]
    code = 1
    try:
        from pxr import Usd

        from generation.envs.s1d.s1d_env_cfg import S1dEnvCfg

        url = S1dEnvCfg(use_cameras=False, use_skin=False).scene.robot.spawn.usd_path
        print(f"simulated copy: {url}")
        sha = hashlib.sha256(opened_bytes(url)).hexdigest()
        pin = pinned_sha256()
        print(f"h1.usd sha256 {sha}")
        if sha != pin:
            print(f"PROBLEM 열어 본 h1.usd 의 해시가 {ASSETS.name} 의 고정값과 다릅니다: {pin}\n1 problem(s)")
            raise SystemExit(1)
        stage = Usd.Stage.Open(url)
        if not stage:
            raise RuntimeError(f"could not open the simulated copy: {url}")
        boxes, bare = measure(stage)
        for link, (centre, half) in sorted(boxes.items()):
            print(f"{link}: centre {centre.tolist()} half {half.tolist()}")
        print(f"{len(boxes)} links with a rendered mesh; without one: {bare or 'none'}")
        problems = box_problems(boxes)
        fresh = document(sha, boxes)
        if not args.write:
            if LINK_BOXES.is_file():
                problems += compare(json.loads(LINK_BOXES.read_text(encoding="utf-8")), fresh)
            else:
                problems.append(f"{LINK_BOXES} 가 없습니다 — --write 로 쓰십시오")
        elif not problems:                  # a measurement with a problem writes nothing
            LINK_BOXES.write_text(serialize(fresh), encoding="utf-8")
            print(f"wrote {LINK_BOXES} ({len(boxes)} links)")
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
