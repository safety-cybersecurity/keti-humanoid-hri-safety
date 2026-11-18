#!/usr/bin/env python3
"""S1d 슬라이스를 잰다.

    .venv-lerobot/bin/python generation/scene/measure_slice.py --episodes <생성 출력 디렉터리> \\
        --log <그 실행의 로그> --dataset <변환한 배포본> --out <out.json>

Isaac 을 부팅하지 않습니다 — 생성 로그 · 사이드카 · npz · 무손실 mkv 와(주면) 변환한 배포본만 읽습니다.
배포하지 않는 측정 도구입니다.

에피소드마다 재는 것:
  (a) 벽시계 시간 — 생성 로그의 「에피소드 N … X 초 + 저장 Y 초」 줄
  (b) 크기 — 카메라별 무손실 mkv · npz, ``--dataset`` 을 주면 배포본 AV1 의 그 에피소드 몫과 data parquet
  (c) 랙 여유 — 랙 안쪽 면(씬 명세)에서 로봇 루트 |y| 의 최댓값을 뺀 값 (기대 1 m 이상)
  (d) ``bbox.camera`` 에 사람이 보인 스텝 비율 (bbox 가 무효값이 아닌 행)
  (e) bbox 면적 — 중앙값과, 검출률(detect_humans.py)이 쓰는 32×32 px 이상인 스텝의 비율
  (f) 최근접 순간 로봇 루트가 고정 카메라(CCTV) 화면 안인가 — 녹화된 카메라 자세로 투영
  (g) 진입 — 사이드카의 실현 ÷ 뽑은 진입 속도와, 인형의 방향이 뽑힌 각도의 진행 방향에서 벗어난 각.
      C9 · ``--compare-with`` 는 배열만 대조하므로 이 둘은 여기서만 봅니다
  (h) 서 있는 스텝의 부위 높이 — 사이드카의 ``contact_region_names`` 로 읽은 ``region_z`` 가 인체 치수 명세의 부위 중심
      높이와, 몸 전체의 공통 오프셋을 빼고 맞는가. 다른 부위의 높이에 서 있는 부위는 더미의 접촉 센서와 관절체의 바디를
      잘못 짝지은 것입니다(C9 는 같은 짝짓기로 다시 생성한 것과 대조하므로 이것을 못 봅니다)

(g) 에서 방향이 진행 방향과 다르거나 진입했는데 실현 속도가 없으면, (h) 에서 부위 높이가 어긋나면 보고서는 쓰되 종료 코드 1 입니다.

``--dataset`` 의 AV1 몫은 PyAV 로 패킷을 읽으므로 ``.venv-lerobot`` 이 필요합니다. 나머지는 ``.venv`` 로 돕니다.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation import cameras  # noqa: E402
from generation.envs.s1a.mdp.entry_geometry import travel_facing_yaw_deg  # noqa: E402
from generation.envs.s1d.jointed_walk_metrics import standing_height_problems, standing_region_heights  # noqa: E402
from generation.scene import layout  # noqa: E402

DETECTABLE_MIN_AREA_PX = 32 * 32  # detect_humans.py's MIN_AREA_PX: frames the detector is scored on
# (g) the sidecar's facing is the float64 the reset derived from the same recorded angle: equal but for
# the degrees <-> radians round trip.
FACING_TOL_DEG = 1e-6

_EPISODE_LINE = re.compile(
    r"에피소드 (?P<seq>\d+) \(seed (?P<seed>\d+)\): 기록 (?P<steps>\d+) 스텝 "
    r"\(env 스텝 (?P<env>\d+), (?P<step_s>[\d.]+) 초 \+ 저장 (?P<save_s>[\d.]+) 초\)")


# (h) a standing step's region heights against the dimensions (jointed_walk_metrics.standing_height_problems): the
# recorder pairs the dummy contact sensor's bodies with the articulation's by name, and a region standing at another
# region's height says that pairing went wrong on the engine. A few steps in, past the reset's settling and before
# the walk can start (the dummy waits until the robot comes near).
STANDING_STEP = 5


def episode_times(log_text: str) -> dict[int, dict]:
    """episode_seq -> its step/save seconds, as run_s1a.py prints them once per episode."""
    out = {}
    for m in _EPISODE_LINE.finditer(log_text):
        step_s, save_s = float(m["step_s"]), float(m["save_s"])
        out[int(m["seq"])] = {"seed": int(m["seed"]), "steps": int(m["steps"]), "env_steps": int(m["env"]),
                              "step_s": step_s, "save_s": save_s, "total_s": step_s + save_s}
    return out


def rack_clearance(root_pose: np.ndarray, face_y: float) -> float:
    """Smallest gap between the robot root's lateral offset and the racks' inner face, over the episode."""
    return float(face_y - np.max(np.abs(np.asarray(root_pose, float)[:, 1])))


def bbox_stats(bbox: np.ndarray, invalid, *, min_area_px: int) -> dict:
    """Visible share, inclusive pixel areas of the visible rows, their median and the share of all steps
    whose box is at least ``min_area_px``."""
    b = np.asarray(bbox, np.int64)
    visible = ~np.all(b == np.asarray(invalid, np.int64), axis=1)
    areas = ((b[visible, 2] - b[visible, 0] + 1) * (b[visible, 3] - b[visible, 1] + 1)).tolist()
    n = len(b)
    return {"visible_fraction": float(visible.sum()) / n if n else 0.0,
            "areas_px": [int(a) for a in areas],
            "area_median_px": float(np.median(areas)) if areas else None,
            "detectable_fraction": sum(a >= min_area_px for a in areas) / n if n else 0.0}


def closest_approach(min_distance: np.ndarray, root_pose: np.ndarray) -> dict | None:
    """The step of the smallest defined robot-human distance and the robot root there."""
    d = np.asarray(min_distance, float)
    if not np.isfinite(d).any():
        return None
    k = int(np.nanargmin(np.where(np.isfinite(d), d, np.nan)))
    return {"step": k, "distance_m": float(d[k]), "root_xyz": [float(v) for v in np.asarray(root_pose)[k, :3]]}


def in_picture(point_w, pose_record, cam: dict) -> dict:
    """Pixel of a world point seen through a recorded pose field (pos3 + quat wxyz) and its distance to the
    nearest picture edge; negative or -inf when outside (-inf: behind the camera)."""
    r = np.asarray(pose_record, float)
    q_xyzw = np.array([r[4], r[5], r[6], r[3]])
    u, v, z = (float(t) for t in cameras.project(point_w, r[:3], q_xyzw, cam))
    edge = min(u, cam["width"] - u, v, cam["height"] - v) if z > 0 else -math.inf
    return {"u": u, "v": v, "z_m": z, "edge_px": edge, "inside": bool(edge >= 0)}


def entry_facts(randomization: dict) -> dict:
    """(g) one episode's entry from its sidecar: realized over drawn speed, and how far the recorded
    facing is from the travel direction of the drawn angle (0 for a dummy that faces its walk)."""
    need = ("human_entry_angle_rad_drawn", "human_facing_yaw_deg", "human_entry_speed_drawn",
            "entry_triggered_realized")
    gone = [k for k in need if k not in randomization]
    if gone:
        raise ValueError(f"사이드카의 randomization 에 진입 기록이 없습니다 — {gone}")
    # json carries NaN and Infinity as they are, and neither compares: a NaN facing is not "off its walk",
    # a NaN speed is not None. Rows that are not usable numbers are named and left out of the arithmetic.
    drawn, realized = randomization["human_entry_speed_drawn"], randomization.get("human_entry_speed_realized")
    unusable = [k for k in ("human_entry_angle_rad_drawn", "human_facing_yaw_deg", "human_entry_speed_drawn")
                if not _finite(randomization[k])]
    if "human_entry_speed_drawn" not in unusable and not drawn > 0:
        unusable.append("human_entry_speed_drawn")
    if realized is not None and not _finite(realized):
        unusable.append("human_entry_speed_realized")
    angle_deg = facing = off = ratio = None
    if not {"human_entry_angle_rad_drawn", "human_facing_yaw_deg"} & set(unusable):
        angle_deg = math.degrees(randomization["human_entry_angle_rad_drawn"])
        facing = float(randomization["human_facing_yaw_deg"])
        off = abs((facing - travel_facing_yaw_deg(angle_deg) + 180.0) % 360.0 - 180.0)
    if realized is not None and not {"human_entry_speed_drawn", "human_entry_speed_realized"} & set(unusable):
        ratio = realized / drawn
    return {"triggered": bool(randomization["entry_triggered_realized"]), "angle_deg": angle_deg,
            "facing_yaw_deg": facing, "facing_off_travel_deg": off, "speed_drawn": drawn,
            "speed_realized": realized, "speed_ratio": ratio, "unusable": unusable}


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def entry_problems(rows: list[dict]) -> list[str]:
    out = []
    for r in rows:
        e = r["entry"]
        if e["unusable"]:
            out.append(f"{r['episode_id']}: 사이드카의 진입 기록이 쓸 수 있는 수가 아닙니다 — {e['unusable']}")
        if e["facing_off_travel_deg"] is not None and e["facing_off_travel_deg"] > FACING_TOL_DEG:
            out.append(f"{r['episode_id']}: 인형의 방향 {e['facing_yaw_deg']:.3f}° 가 진입 각도 {e['angle_deg']:.3f}° 의 "
                       f"진행 방향에서 {e['facing_off_travel_deg']:.3f}° 벗어나 있습니다")
        if e["triggered"] and e["speed_realized"] is None:
            out.append(f"{r['episode_id']}: 진입했는데 실현 속도가 없습니다 — 속도 창이 열리기 전에 닫혔는가 "
                       "(리셋이 walk_ended 를 지우는가), 아니면 진입한 스텝에 에피소드가 끝났는가")
    return out


def spread(xs, rng) -> dict:
    """n, min, p5, median, p95, max of ``xs`` and the share inside the closed range ``rng``."""
    a = np.asarray(xs, float)
    lo, hi = sorted(float(v) for v in rng)
    return {"n": int(a.size), "min": float(a.min()), "p5": float(np.percentile(a, 5)),
            "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max()),
            "range": [lo, hi], "inside_fraction": float(np.mean((a >= lo) & (a <= hi)))}


def share_by_spans(packets, spans) -> tuple[dict[int, int], dict[int, int], int]:
    """Bytes and packets per episode of one video file: each (timestamp, size) packet goes to the episode
    whose [from, to) span holds it (the converter's spans); the bytes no span holds are returned apart."""
    share = {ep: 0 for ep, _, _ in spans}
    frames = {ep: 0 for ep, _, _ in spans}
    unassigned = 0
    for t, size in packets:
        for ep, a, b in spans:
            if a - 1e-6 <= t < b - 1e-6:
                share[ep] += size
                frames[ep] += 1
                break
        else:
            unassigned += size
    return share, frames, unassigned


def check_complete_spans(name: str, frames: dict[int, int], lengths: dict[int, int]) -> None:
    """Every episode a file's spans name holds one packet per frame -- a file cut short inside a span
    would otherwise be counted as the whole episode."""
    short = {ep: (n, lengths[ep]) for ep, n in frames.items() if n != lengths[ep]}
    if short:
        raise ValueError(f"{name}: 구간 안의 패킷 수가 에피소드 길이와 다릅니다 — "
                         + ", ".join(f"에피소드 {ep} 패킷 {n} · 길이 {length}" for ep, (n, length) in sorted(short.items())))


def check_same_episodes(sides: list[dict], dataset_episodes: dict[int, dict]) -> None:
    """The converted dataset holds the very episodes measured here: the same ids with the same lengths,
    each of them once -- counted, since a set folds an episode listed twice into one."""
    want = Counter((s["episode_id"], s["steps"]) for s in sides)
    got = Counter((e["episode_id"], e["length"]) for e in dataset_episodes.values())
    if want != got:
        raise ValueError(f"배포본이 이 생성본의 변환이 아닙니다 — 생성에만 {sorted((want - got).elements())} · "
                         f"배포본에만 {sorted((got - want).elements())}")
    twice = sorted(eid for eid, n in Counter(eid for eid, _ in want.elements()).items() if n > 1)
    if twice:
        raise ValueError(f"같은 에피소드가 생성본과 배포본에 두 번씩 있습니다 — {twice}")


def _sidecars(episodes_dir: Path) -> list[tuple[Path, dict]]:
    out = []
    for p in sorted(episodes_dir.glob("*.json")):
        if p.name.endswith(".labels.json"):
            continue
        out.append((p, json.loads(p.read_text(encoding="utf-8"))))
    return sorted(out, key=lambda t: t[1]["episode_seq"])


def measure(episodes_dir, *, log_text: str, cameras_canon: dict, scene_spec: dict, anthropometry: dict | None = None) -> dict:
    episodes_dir = Path(episodes_dir)
    if anthropometry is None:
        import yaml

        anthropometry = yaml.safe_load((REPO / "generation/dummy/anthropometry.yaml").read_text(encoding="utf-8"))
    standing = standing_region_heights(anthropometry)
    sides = _sidecars(episodes_dir)
    if not sides:
        raise ValueError(f"{episodes_dir} 에 에피소드 사이드카가 없습니다")
    times = episode_times(log_text)
    missing = [s["episode_seq"] for _, s in sides if s["episode_seq"] not in times]
    if missing:
        raise ValueError(f"생성 로그에 에피소드 {missing} 의 시간 줄이 없습니다 — 다른 실행의 로그인가")
    # Joined on the sequence number, so the seed and the recorded steps have to agree too.
    other = [s["episode_id"] for _, s in sides
             if (times[s["episode_seq"]]["seed"], times[s["episode_seq"]]["steps"]) != (s["seed"], s["steps"])]
    if other:
        raise ValueError(f"생성 로그의 시드·기록 스텝이 사이드카와 다릅니다: {other} — 다른 실행의 로그인가")

    face_y = layout.rack_inner_face_y(scene_spec)
    bbox_canon = cameras_canon["bbox"]
    fixed = {n: c for n, c in cameras_canon["cameras"].items() if c["type"] == "fixed"}
    rows, pooled_areas = [], []
    for path, side in sides:
        media = side.get("media") or {}
        # media_writer's row for a camera without a frame is {"file": None, "frames": 0}.
        gone = [n for n in cameras_canon["cameras"]
                if not isinstance(media.get(n), dict) or not media[n].get("file")
                or not (episodes_dir / media[n]["file"]).is_file()]
        if gone:
            raise ValueError(f"{side['episode_id']}: 정본 카메라의 녹화가 없습니다 — {gone}")
        # The other way round: a file under a name the canon has no camera for (another canon's recording,
        # or a pack, which this tool does not size) would join the episode's total but no camera's mean.
        extra = sorted(n for n, v in media.items()
                       if isinstance(v, dict) and "file" in v and n not in cameras_canon["cameras"])
        if extra:
            raise ValueError(f"{side['episode_id']}: 정본 카메라가 아닌 녹화 파일이 사이드카에 있습니다 — {extra}")
        z = np.load(path.with_suffix(".npz"))
        seq = side["episode_seq"]
        labels_path = path.with_name(path.stem + ".labels.json")
        labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
        bb = bbox_stats(z["bbox_human"], bbox_canon["invalid"], min_area_px=DETECTABLE_MIN_AREA_PX)
        pooled_areas += bb["areas_px"]
        close = closest_approach(z["min_distance_human"], z["root_pose"])
        if close is not None:
            close["in_picture"] = {n: in_picture(close["root_xyz"], z[f"camera_pose_{n}"][close["step"]], cam)
                                   for n, cam in fixed.items()}
        rows.append({
            "episode_id": side["episode_id"], "episode_seq": seq, "seed": side["seed"], "steps": side["steps"],
            "wall_s": times[seq]["total_s"], "step_s": times[seq]["step_s"], "save_s": times[seq]["save_s"],
            "termination": list(side.get("termination", {}).get("reason", [])),
            "outcome": labels.get("outcome"), "severity": labels.get("severity"),
            "npz_bytes": path.with_suffix(".npz").stat().st_size,
            "mkv_bytes": {n: (episodes_dir / media[n]["file"]).stat().st_size for n in cameras_canon["cameras"]},
            "rack_clearance_m": rack_clearance(z["root_pose"], face_y),
            "entry": entry_facts(side.get("randomization") or {}),
            "hold_release_step": side.get("hold_release_step"),
            "bbox": {k: v for k, v in bb.items() if k != "areas_px"},
            "closest": close,
            "standing": standing_height_problems(side["contact_region_names"],
                                                 z["region_z"][min(STANDING_STEP, len(z["region_z"]) - 1)], standing),
        })

    closes = [r["closest"] for r in rows if r["closest"] is not None]
    ratios = [r["entry"]["speed_ratio"] for r in rows if r["entry"]["speed_ratio"] is not None]
    # held to the end of the walk: the servo carried the dummy the whole way, nothing pushed it
    held = [r["entry"]["speed_ratio"] for r in rows
            if r["entry"]["speed_ratio"] is not None and r["hold_release_step"] == -1]
    summary = {
        "episodes": len(rows),
        "wall_s": spread([r["wall_s"] for r in rows], (0, math.inf)),
        "mkv_bytes_per_episode": spread([sum(r["mkv_bytes"].values()) for r in rows], (0, math.inf)),
        "mkv_bytes_per_camera_mean": {n: float(np.mean([r["mkv_bytes"][n] for r in rows if n in r["mkv_bytes"]]))
                                      for n in cameras_canon["cameras"] if any(n in r["mkv_bytes"] for r in rows)},
        "rack_face_y_m": face_y,
        "rack_clearance_min_m": min(r["rack_clearance_m"] for r in rows),
        "termination": dict(sorted(Counter(x for r in rows for x in r["termination"]).items())),
        "outcome": dict(sorted(Counter(r["outcome"] for r in rows if r["outcome"]).items())),
        "bbox_camera": bbox_canon["camera"],
        "bbox_visible_fraction": spread([r["bbox"]["visible_fraction"] for r in rows], (0, 1)),
        "bbox_area_median_px": float(np.median(pooled_areas)) if pooled_areas else None,
        "bbox_detectable_fraction_mean": float(np.mean([r["bbox"]["detectable_fraction"] for r in rows])),
        "entry_triggered": sum(r["entry"]["triggered"] for r in rows),
        "entry_speed_ratio": spread(ratios, (0, math.inf)) if ratios else None,
        "entry_speed_ratio_held": spread(held, (0, math.inf)) if held else None,
        "facing_off_travel_deg_max": max((r["entry"]["facing_off_travel_deg"] for r in rows
                                          if r["entry"]["facing_off_travel_deg"] is not None), default=None),
        "closest_defined": len(closes),
        "closest_distance_m": spread([c["distance_m"] for c in closes], (0, math.inf)) if closes else None,
        "closest_x_m": {n: spread([c["root_xyz"][0] for c in closes],
                                  cameras.site_placement(c_, "S1d", name=n)["mount"]["aim_x_range_m"])
                        for n, c_ in fixed.items()} if closes else {},
        "closest_in_picture_fraction": {n: float(np.mean([c["in_picture"][n]["inside"] for c in closes]))
                                        for n in fixed} if closes else {},
        "closest_edge_px_min": {n: min(c["in_picture"][n]["edge_px"] for c in closes) for n in fixed}
                               if closes else {},
    }
    summary["standing_episodes_off"] = sum(bool(r["standing"]) for r in rows)
    problems = [f"(g) {line}" for line in entry_problems(rows)]
    problems += [f"(h) {r['episode_id']}: {line}" for r in rows for line in r["standing"]]
    return {"episodes": rows, "summary": summary, "problems": problems}


def _read_packets(path: Path) -> tuple[list[tuple[float, int]], str]:
    """(timestamp, size) of every packet of a video file's first stream, and its codec. Needs PyAV
    (.venv-lerobot) -- the one line of av1_shares the tests swap out."""
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        packets = [(float(p.pts * s.time_base), p.size) for p in c.demux(s) if p.size and p.pts is not None]
        return packets, s.codec_context.name


def av1_shares(dataset_root) -> dict:
    """Per video key: bytes per episode of the shipped AV1 (packets in the episode's span), plus each file's
    size and the bytes no span holds."""
    import pandas as pd

    root = Path(dataset_root)
    eps = pd.concat([pd.read_parquet(f) for f in sorted(root.glob("meta/episodes/*/*.parquet"))])
    # Everything below is keyed by episode_index: two rows of one index would fold into one.
    twice = sorted({int(i) for i in eps["episode_index"][eps["episode_index"].duplicated()]})
    if twice:
        raise ValueError(f"{root}: meta/episodes 에 같은 episode_index 가 두 번 있습니다 — {twice}")
    keys = sorted({c.split("/")[1] for c in eps.columns if c.startswith("videos/")})
    lengths = {int(r["episode_index"]): int(r["length"]) for _, r in eps.iterrows()}
    out = {"per_episode": {}, "files": {},
           "episodes_meta": {int(r["episode_index"]): {"episode_id": str(r["episode_id"]), "length": int(r["length"])}
                             for _, r in eps.iterrows()}}
    for key in keys:
        files: dict[tuple[int, int], list] = {}
        for _, row in eps.iterrows():
            ci, fi = int(row[f"videos/{key}/chunk_index"]), int(row[f"videos/{key}/file_index"])
            files.setdefault((ci, fi), []).append((int(row["episode_index"]), float(row[f"videos/{key}/from_timestamp"]),
                                                   float(row[f"videos/{key}/to_timestamp"])))
        for (ci, fi), spans in files.items():
            path = root / f"videos/{key}/chunk-{ci:03d}/file-{fi:03d}.mp4"
            packets, codec = _read_packets(path)
            share, frames, unassigned = share_by_spans(packets, spans)
            check_complete_spans(f"{key}/{path.name}", frames, {ep: lengths[ep] for ep in frames})
            out["files"][f"{key}/{path.name}"] = {"bytes": path.stat().st_size, "codec": codec,
                                                   "unassigned_packet_bytes": unassigned}
            for ep, b in share.items():
                out["per_episode"].setdefault(ep, {})[key] = b
    data = sorted(root.glob("data/*/*.parquet"))
    out["data_parquet_bytes"] = sum(p.stat().st_size for p in data)
    out["episodes"] = len(eps)
    return out


def _mb(b) -> str:
    return f"{b / 1e6:.2f}"


def _print(r: dict) -> None:
    s = r["summary"]
    print(f"에피소드 {s['episodes']}건")
    w = s["wall_s"]
    print(f"(a) 벽시계 시간 — 평균 {np.mean([e['wall_s'] for e in r['episodes']]):.1f} s · "
          f"최소 {w['min']:.1f} · 최대 {w['max']:.1f}")
    m = s["mkv_bytes_per_episode"]
    print(f"(b) 무손실 mkv / 에피소드 — 중앙 {_mb(m['median'])} MB · 최소 {_mb(m['min'])} · 최대 {_mb(m['max'])}; 카메라별 평균 "
          + " · ".join(f"{n} {_mb(b)}" for n, b in s["mkv_bytes_per_camera_mean"].items()))
    if "av1" in r:
        a = r["av1"]
        per_ep = [sum(v.values()) for v in a["per_episode"].values()]
        keys = sorted({k for v in a["per_episode"].values() for k in v})
        print(f"    배포 AV1 / 에피소드 — 평균 {_mb(np.mean(per_ep))} MB · 최소 {_mb(min(per_ep))} · 최대 {_mb(max(per_ep))}; "
              "카메라별 평균 " + " · ".join(
                  f"{k.split('.')[-1]} {_mb(np.mean([v.get(k, 0) for v in a['per_episode'].values()]))}" for k in keys))
        print(f"    data parquet 합 {_mb(a['data_parquet_bytes'])} MB ({_mb(a['data_parquet_bytes'] / a['episodes'])} / 에피소드)")
    print(f"(c) 랙 여유 최소 {s['rack_clearance_min_m']:.3f} m (안쪽 면 y {s['rack_face_y_m']} m) · 종료 사유 {s['termination']}")
    v = s["bbox_visible_fraction"]
    print(f"(d) {s['bbox_camera']} 에 사람이 보인 스텝 — 에피소드별 중앙 {v['median']:.3f} · 최소 {v['min']:.3f} · 최대 {v['max']:.3f}")
    print(f"(e) bbox 면적 중앙값(보인 프레임 전체) {s['bbox_area_median_px']} px · "
          f"{DETECTABLE_MIN_AREA_PX} px 이상 스텝 비율 평균 {s['bbox_detectable_fraction_mean']:.3f}")
    for n, x in s["closest_x_m"].items():
        print(f"(f) {n}: 최근접 {s['closest_defined']}건 — 화면 안 {s['closest_in_picture_fraction'][n]:.3f} "
              f"(가장자리 최소 {s['closest_edge_px_min'][n]:.1f} px) · 로봇 x p5 {x['p5']:.2f} · p95 {x['p95']:.2f} · "
              f"최소 {x['min']:.2f} · 최대 {x['max']:.2f} · 조준 구간 {x['range']} 안 {x['inside_fraction']:.3f}")
    e, h = s["entry_speed_ratio"], s["entry_speed_ratio_held"]
    print(f"(g) 진입 {s['entry_triggered']}건 — 실현 ÷ 뽑은 속도 "
          + (f"중앙 {e['median']:.4f} · 최소 {e['min']:.4f} · 최대 {e['max']:.4f}" if e else "없음")
          + (f"; 붙잡기가 안 풀린 것만 최소 {h['min']:.4f} · 최대 {h['max']:.4f}" if h else "")
          + (f" · 방향이 진행 방향에서 벗어난 각 최대 {s['facing_off_travel_deg_max']:.2e}°"
             if s["facing_off_travel_deg_max"] is not None else " · 방향을 잴 수 있는 에피소드 없음"))
    print(f"(h) 서 있는 스텝({STANDING_STEP})의 부위 높이가 치수와 어긋난 에피소드 {s['standing_episodes_off']}건 / {s['episodes']}")
    print(f"라벨 {s['outcome']}")
    for line in r["problems"]:
        print(f"PROBLEM {line}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", required=True, help="생성 출력 디렉터리 (예: .tmp/episodes/S1d)")
    p.add_argument("--log", required=True, help="그 실행의 run_s1a.py 로그")
    p.add_argument("--dataset", help="변환한 배포본 루트 — 주면 AV1 · parquet 크기도 잼 (.venv-lerobot)")
    p.add_argument("--out", help="결과 JSON")
    args = p.parse_args(argv)

    r = measure(args.episodes, log_text=Path(args.log).read_text(encoding="utf-8", errors="replace"),
                cameras_canon=cameras.load(REPO / "meta/cameras.json"),
                scene_spec=layout.load_spec(REPO / "generation/scene/logistics_aisle.yaml"))
    if args.dataset:
        r["av1"] = av1_shares(args.dataset)
        check_same_episodes(r["episodes"], r["av1"]["episodes_meta"])
    _print(r)
    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    return 1 if r["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
