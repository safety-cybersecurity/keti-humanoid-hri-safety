#!/usr/bin/env python3
"""옆에서 본 막대 인형 영상 — 걸음 표의 인형과 그 표를 만든 배우를 나란히.

    python generation/dummy/gait/review_video.py --archive ~/Downloads/100STYLE.zip \
        --table generation/dummy/build/keti_gait_100style.json --out gait-stick.mp4 --speeds 0.4 1.0 1.6

왼쪽은 배우 — 표가 쓰는 곧게 걷는 주기 가운데 그 속도에 가장 가까운 것(인형 다리로 잰 속도)의 관절을, 그 주기의
평균 방향이 앞(+x)이 되게 돌려서. 오른쪽은 인형 — 표를
그 속도로 읽어 다리는 순기구학, 골반은 그 속도로 앞으로, 높이는 가장 낮은 발바닥 모서리가 바닥에 닿게 — 발은
발목 · 뒤꿈치 · 발끝의 세 점으로 그린다(뒤꿈치로 딛는 순간이 보이게). 속도마다 세 주기.
물리 없이 걸음 모양만 본다(matplotlib · ffmpeg)."""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

from generation.dummy import joints_spec as js  # noqa: E402
from generation.dummy.gait import bvh, cycles, kinematics as kin, retarget as rt  # noqa: E402
from generation.dummy.gait.build_gait import SPEC_PATH, _clip_cycles, _member  # noqa: E402
from generation.dummy.gait.gait_table import GaitTable  # noqa: E402

FPS = 50
FONT = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
ACTOR_BONES = [("Hips", "Chest4"), ("Chest4", "Head"), ("Chest4", "LeftShoulder"), ("LeftShoulder", "LeftElbow"),
               ("LeftElbow", "LeftWrist"), ("Chest4", "RightShoulder"), ("RightShoulder", "RightElbow"),
               ("RightElbow", "RightWrist"), ("Hips", "LeftHip"), ("LeftHip", "LeftKnee"), ("LeftKnee", "LeftAnkle"),
               ("LeftAnkle", "LeftToe"), ("Hips", "RightHip"), ("RightHip", "RightKnee"), ("RightKnee", "RightAnkle"),
               ("RightAnkle", "RightToe")]
DUMMY_BONES = [("neck", "pelvis")] + [(a + s, b + s) for s in ("_left", "_right") for a, b in (
    ("pelvis", "hip"), ("hip", "knee"), ("knee", "ankle"), ("ankle", "toe"), ("ankle", "heel"), ("heel", "toe"),
    ("neck", "shoulder"),
    ("shoulder", "elbow"), ("elbow", "wrist"))]


def dummy_points(row, spec) -> dict:
    cols = js.columns(spec)
    regions = spec.anthropometry["regions"]
    pts = {"neck": np.array(regions["BR-03"]["center"], float)}
    hips = []
    for side in rt.SIDES:
        g = js.leg_geometry(spec, side)
        R_h, kn, R_a = rt.leg_matrices(row, spec, side)
        ankle, sole, _ = kin.leg_fk(g, R_h, kn, R_a)
        pts[f"hip_{side}"] = g.hip
        pts[f"knee_{side}"] = g.hip + R_h @ (g.knee - g.hip)
        pts[f"ankle_{side}"] = ankle
        pts[f"toe_{side}"] = sole[int(np.argmax(sole[:, 0]))]
        pts[f"heel_{side}"] = sole[int(np.argmin(sole[:, 0]))]
        pts[f"low_{side}"] = float(sole[:, 2].min())
        sh, el = js.joint(spec, f"shoulder_{side}"), js.joint(spec, f"elbow_{side}")
        S = kin.dofs_to_matrix(row[cols[sh.name]], js.axes(sh), spec.d6_convention)
        sign = 1.0 if side == "left" else -1.0
        wrist_rest = js.capsule_ends(regions["BR-09"], sign)[1]
        elbow = np.array(sh.pivot) + S @ (np.array(el.pivot) - np.array(sh.pivot))
        pts[f"shoulder_{side}"] = np.array(sh.pivot)
        pts[f"elbow_{side}"] = elbow
        pts[f"wrist_{side}"] = elbow + S @ kin.axis_rotation("y", float(row[cols[el.name]][0])) @ (wrist_rest - np.array(el.pivot))
        pts[f"pelvis_{side}"] = (g.hip * [1, 0, 1])
        pts[f"neck_{side}"] = pts["neck"]
        hips.append(g.hip)
    pts["pelvis"] = np.mean(hips, axis=0)
    return pts


def actor_cycles(archive: Path, spec, gs):
    """The walk's straight cycles as the build takes them (_clip_cycles: their speed on the dummy's legs), each with the
    actor's joint positions over it, turned so that the cycle's mean heading is the front (+x)."""
    with zipfile.ZipFile(archive) as z:
        motion = bvh.parse_bvh(z.read(_member(z, gs["clips"]["walk"])).decode("utf-8"))
    sk = motion.skeleton
    pos, rot = bvh.forward_kinematics(motion, scale=cycles.unit_scale(sk))
    C = cycles.body_axes(sk)
    heading = cycles.pelvis_heading_deg(rot, sk, C)
    out = []
    for c in _clip_cycles(motion, spec, gs)[0]:
        if abs(c["turn_deg"]) > gs["cycles"]["max_turn_deg"]:
            continue                                      # the table leaves the turning cycles out
        a, b = c["frames"]
        face = kin.axis_rotation("z", -np.radians(np.mean(heading[a:b + 1])))   # the cycle's mean heading onto +x
        out.append({"speed": c["speed"], "pos": pos[a:b + 1] @ C.T @ face.T, "names": sk.names})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="걸음 표 검토 영상(막대 인형)")
    p.add_argument("--archive", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--speeds", type=float, nargs="+", default=[0.4, 1.0, 1.6])
    p.add_argument("--cycles", type=int, default=3)
    a = p.parse_args()
    if Path(FONT).is_file():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(FONT)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
        plt.rcParams["axes.unicode_minus"] = False       # the Korean font has no U+2212 minus sign
    spec = js.load()
    gs = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    table = GaitTable.from_json(json.loads(Path(a.table).read_text(encoding="utf-8")))
    actors = actor_cycles(Path(a.archive).expanduser(), spec, gs)
    fig, (axa, axd) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
    with writer.saving(fig, a.out, dpi=100):
        for v in a.speeds:
            stride = float(table.stride_at(np.array([v]))[0])
            T = stride / v
            act = min(actors, key=lambda c: abs(c["speed"] - v))
            n_act = len(act["pos"]) - 1
            step = act["pos"][-1, act["names"].index("Hips")] - act["pos"][0, act["names"].index("Hips")]
            for i in range(int(a.cycles * T * FPS)):
                t = i / FPS
                phase = (t / T) % 1.0
                k = int(t / T)
                row = table.at(np.array([v]), np.array([phase]))[0].astype(float)
                pts = dummy_points(row, spec)
                low = min(pts["low_left"], pts["low_right"])            # the lowest sole corner on the floor
                shift = np.array([v * t, 0.0, -low])
                f = act["pos"][int(round(phase * n_act))] + k * step * [1, 1, 0]
                for ax, title in ((axa, f"배우 — 가장 가까운 주기 {act['speed']:.2f} m/s"), (axd, f"관절 인형 — 걸음 표 {v:.2f} m/s")):
                    ax.cla()
                    ax.set_aspect("equal")
                    ax.set_ylim(-0.05, 1.9)
                    ax.axhline(0.0, color="0.6", lw=1)
                    ax.set_title(title)
                cx = f[act["names"].index("Hips"), 0]
                axa.set_xlim(cx - 1.2, cx + 1.2)
                for u, w in ACTOR_BONES:
                    pu, pw = f[act["names"].index(u)], f[act["names"].index(w)]
                    axa.plot([pu[0], pw[0]], [pu[2], pw[2]], color="tab:orange" if "Left" in u + w else "tab:blue", lw=3)
                axd.set_xlim(v * t - 1.2, v * t + 1.2)
                for u, w in DUMMY_BONES:
                    pu, pw = pts[u] + shift, pts[w] + shift
                    axd.plot([pu[0], pw[0]], [pu[2], pw[2]], color="tab:orange" if "left" in u + w else "tab:blue", lw=3)
                writer.grab_frame()
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
