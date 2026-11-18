"""generation/scene/measure_slice.py — 슬라이스 측정이 재는 양을 맞게 재는가.

측정 도구라 배포하지 않지만, 여기서 나온 수가 결정(키프레임 간격 · CCTV 조준 구간 ·
통로 폭)의 근거로 쓰입니다. 영상 패킷을 읽는 PyAV 는 `.venv` 에 없으므로 그 한 줄(`_read_packets`)만
갈아 끼우고, 나머지 — 패킷을 에피소드에 나누는 규칙 · 거부하는 입력 · `av1_shares` · `main` — 는 실제 코드로 돕니다.
도우미만 부르는 시험은 호출부가 빠져도 초록입니다.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from generation import cameras
from generation.envs.s1a.mdp.entry_geometry import travel_facing_yaw_deg
from generation.scene import layout, measure_slice as ms

REPO = Path(__file__).resolve().parents[2]
CANON = cameras.load(REPO / "meta/cameras.json")
SPEC = layout.load_spec(REPO / "generation/scene/logistics_aisle.yaml")
STANDING = {code: r["center"][2] for code, r in
            yaml.safe_load((REPO / "generation/dummy/anthropometry.yaml").read_text(encoding="utf-8"))["regions"].items()}

LOG = """\
[INFO]: RTX streaming completed in 0.02 s.
에피소드 0 (seed 2912205018): 기록 999 스텝 (env 스텝 1000, 93.2 초 + 저장 7.2 초) → .tmp/episodes/S1d/S1d_0000_seed2912205018.npz
some other line
에피소드 1 (seed 460152974): 기록 412 스텝 (env 스텝 413, 40.0 초 + 저장 3.5 초) → .tmp/episodes/S1d/S1d_0001_seed460152974.npz
"""


def test_episode_times_reads_each_episode_line_of_the_run_log():
    t = ms.episode_times(LOG)
    assert sorted(t) == [0, 1]
    assert t[0] == {"seed": 2912205018, "steps": 999, "env_steps": 1000,
                    "step_s": 93.2, "save_s": 7.2, "total_s": pytest.approx(100.4)}
    assert t[1]["steps"] == 412 and t[1]["total_s"] == pytest.approx(43.5)


def test_rack_clearance_is_the_face_minus_the_widest_lateral_offset():
    root = np.zeros((3, 7), np.float32)
    root[:, 1] = [0.1, -0.7, 0.3]
    assert ms.rack_clearance(root, 5.0) == pytest.approx(4.3)


def test_bbox_stats_leaves_invalid_rows_out_and_counts_extents_inclusively():
    invalid = CANON["bbox"]["invalid"]
    bbox = np.array([invalid, [10, 20, 19, 29], [0, 0, 39, 39], invalid], np.int32)
    s = ms.bbox_stats(bbox, invalid, min_area_px=32 * 32)
    assert s["visible_fraction"] == pytest.approx(0.5)
    assert s["areas_px"] == [100, 1600]
    assert s["area_median_px"] == pytest.approx(850.0)
    assert s["detectable_fraction"] == pytest.approx(0.25)


def test_bbox_stats_of_an_episode_that_never_sees_the_human():
    invalid = CANON["bbox"]["invalid"]
    s = ms.bbox_stats(np.array([invalid] * 3, np.int32), invalid, min_area_px=1024)
    assert s["visible_fraction"] == 0.0 and s["areas_px"] == []
    assert s["area_median_px"] is None and s["detectable_fraction"] == 0.0


def test_closest_approach_skips_undefined_distances_and_reports_the_robot_root():
    d = np.array([np.nan, 3.0, 0.8, np.nan, 1.2])
    root = np.zeros((5, 7), np.float32)
    root[:, 0] = np.arange(5) + 4.0
    root[:, 2] = 1.04
    c = ms.closest_approach(d, root)
    assert c["step"] == 2 and c["distance_m"] == pytest.approx(0.8)
    assert c["root_xyz"] == pytest.approx([6.0, 0.0, 1.04])


def test_closest_approach_of_an_episode_without_a_distance_is_none():
    assert ms.closest_approach(np.full(4, np.nan), np.zeros((4, 7), np.float32)) is None


def _fixed():
    names = [n for n, c in CANON["cameras"].items() if c["type"] == "fixed"]
    assert names, "the canon declares no fixed camera"
    cam = CANON["cameras"][names[0]]
    return cam, cameras.pose_record(*cameras.fixed_pose(cam, SPEC, "S1d"))


def test_the_aim_point_projects_to_the_picture_centre_through_the_recorded_pose():
    # The recorded pose is wxyz (pose_record); reading it as xyzw turns the camera away.
    cam, record = _fixed()
    m = cameras.site_placement(cam, "S1d")["mount"]
    aim = [sum(m["aim_x_range_m"]) / 2.0, m["aim_y_m"], m["aim_z_m"]]
    p = ms.in_picture(aim, record, cam)
    assert p["inside"] is True
    assert p["u"] == pytest.approx(cam["width"] / 2.0, abs=1e-3)
    assert p["v"] == pytest.approx(cam["height"] / 2.0, abs=1e-3)
    assert p["edge_px"] == pytest.approx(min(cam["width"], cam["height"]) / 2.0, abs=1e-3)


def test_a_point_behind_the_camera_is_not_in_the_picture():
    cam, record = _fixed()
    pos = np.asarray(record[:3], float)
    behind = pos - 10.0 * (np.array([8.55, 0.3, 1.0]) - pos)
    p = ms.in_picture(behind, record, cam)
    assert p["inside"] is False and p["edge_px"] == -np.inf


def test_a_point_far_along_the_aisle_falls_off_the_picture():
    cam, record = _fixed()
    assert ms.in_picture([60.0, 0.0, 1.0], record, cam)["inside"] is False


def test_spread_reports_percentiles_and_the_share_inside_a_range():
    xs = [0.4, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 15.1]
    s = ms.spread(xs, (5.3, 11.8))
    assert s["n"] == 10 and s["min"] == 0.4 and s["max"] == 15.1
    assert s["p5"] == pytest.approx(float(np.percentile(xs, 5)))
    assert s["p95"] == pytest.approx(float(np.percentile(xs, 95)))
    assert s["inside_fraction"] == pytest.approx(0.6)


def test_share_by_spans_gives_each_packet_to_the_episode_whose_span_holds_it():
    # The same half-open rule the converter's spans mean: [from, to).
    packets = [(0.0, 10), (0.02, 5), (19.98, 7), (20.0, 3), (40.0, 99)]
    spans = [(0, 0.0, 20.0), (1, 20.0, 40.0)]
    share, frames, unassigned = ms.share_by_spans(packets, spans)
    assert share == {0: 22, 1: 3}
    assert frames == {0: 3, 1: 1}
    assert unassigned == 99


def test_a_span_with_fewer_packets_than_its_episode_has_frames_is_refused():
    # A file cut short inside an episode's span was counted as the whole episode.
    with pytest.raises(ValueError, match="cam/file-000.mp4.*에피소드 1.*2.*4"):
        ms.check_complete_spans("cam/file-000.mp4", {0: 4, 1: 2}, {0: 4, 1: 4})
    ms.check_complete_spans("cam/file-000.mp4", {0: 4, 1: 4}, {0: 4, 1: 4})


def test_a_dataset_of_another_run_is_refused_even_with_the_same_episode_count():
    sides = [{"episode_id": "S1d_0000_seed1", "steps": 999}, {"episode_id": "S1d_0001_seed2", "steps": 504}]
    ms.check_same_episodes(sides, {0: {"episode_id": "S1d_0000_seed1", "length": 999},
                                   1: {"episode_id": "S1d_0001_seed2", "length": 504}})
    with pytest.raises(ValueError, match="S1d_0001_seed9"):
        ms.check_same_episodes(sides, {0: {"episode_id": "S1d_0000_seed1", "length": 999},
                                       1: {"episode_id": "S1d_0001_seed9", "length": 504}})
    with pytest.raises(ValueError, match="S1d_0001_seed2"):
        ms.check_same_episodes(sides, {0: {"episode_id": "S1d_0000_seed1", "length": 999},
                                       1: {"episode_id": "S1d_0001_seed2", "length": 999}})


def test_an_episode_listed_twice_is_refused_though_the_ids_match_as_a_set():
    # Compared as sets, generated A, B against shipped A, B, A passed --
    # an input the episode-count check this comparison replaced used to refuse.
    a, b = {"episode_id": "S1d_0000_seed1", "steps": 999}, {"episode_id": "S1d_0001_seed2", "steps": 504}
    shipped = {0: {"episode_id": "S1d_0000_seed1", "length": 999}, 1: {"episode_id": "S1d_0001_seed2", "length": 504}}
    twice = {**shipped, 2: shipped[0]}
    with pytest.raises(ValueError, match="배포본에만.*S1d_0000_seed1"):
        ms.check_same_episodes([a, b], twice)
    with pytest.raises(ValueError, match="생성에만.*S1d_0000_seed1.*배포본에만 \\[\\]"):
        ms.check_same_episodes([a, b, a], shipped)
    with pytest.raises(ValueError, match="두 번.*S1d_0000_seed1"):
        ms.check_same_episodes([a, b, a], twice)


def _entry(angle_deg=118.6, facing="travel", drawn=0.8, realized=0.804, triggered=True) -> dict:
    """The entry rows the recorder writes into a sidecar's randomization (S1d: the facing is the walk's)."""
    return {"human_entry_speed_drawn": drawn, "human_entry_speed_realized": realized,
            "human_entry_angle_rad_drawn": math.radians(angle_deg), "entry_triggered_realized": triggered,
            "human_facing_yaw_deg": travel_facing_yaw_deg(angle_deg) if facing == "travel" else facing}


def _standing_heights(swap=None, offset=-0.023):
    """The recorder's region heights at a standing step: each region's centre in the dimensions, the whole body
    a little lower; swap=(a, b) records region a at b's height and b at a's."""
    z = {c: h + offset for c, h in STANDING.items()}
    if swap:
        a, b = swap
        z[a], z[b] = z[b], z[a]
    return [z[c] for c in sorted(z)]


def _write_episode(d: Path, seq: int, seed: int, steps: int, *, y: float, see: bool, closest_x: float,
                   reason: str = "time_out", outcome: str = "near_miss", entry: dict | None = None,
                   region_z=None) -> str:
    eid = f"S1d_{seq:04d}_seed{seed}"
    root = np.zeros((steps, 7), np.float32)
    root[:, 0] = np.linspace(0.4, 12.0, steps)
    root[:, 1] = y
    root[:, 2] = 1.04
    dist = np.linspace(4.0, 2.0, steps)
    k = steps // 2
    root[k, 0] = closest_x
    dist[k] = 0.5
    invalid = CANON["bbox"]["invalid"]
    bbox = np.array([[100, 100, 199, 299] if see and i % 2 else invalid for i in range(steps)], np.int32)
    arrays = {"root_pose": root, "min_distance_human": dist, "bbox_human": bbox,
              "region_z": np.tile(np.asarray(_standing_heights() if region_z is None else region_z), (steps, 1))}
    for name, cam in CANON["cameras"].items():
        rec = (cameras.pose_record(*cameras.fixed_pose(cam, SPEC, "S1d")) if cam["type"] == "fixed"
               else np.zeros(7, np.float32))
        arrays[f"camera_pose_{name}"] = np.tile(rec, (steps, 1))
    np.savez(d / f"{eid}.npz", **arrays)
    (d / "media").mkdir(exist_ok=True)
    media = {}
    for i, name in enumerate(CANON["cameras"]):
        f = f"media/{eid}.{name}.mkv"
        (d / f).write_bytes(b"\0" * (1000 * (i + 1)))
        media[name] = {"file": f, "frames": steps}
    # The rows media_writer adds beside the cameras': none of them names a file.
    media.update({"rerendered_steps": [], "blank_frames": {n: 0 for n in CANON["cameras"]}, "canon_fingerprint": "f" * 64})
    side = {"episode_id": eid, "scenario_id": "S1d", "episode_seq": seq, "seed": seed, "steps": steps,
            "termination": {"terminated": reason != "time_out", "truncated": reason == "time_out",
                            "reason": [reason]},
            "hold_release_step": -1, "randomization": _entry() if entry is None else entry,
            "media": media, "contact_region_names": sorted(STANDING)}
    (d / f"{eid}.json").write_text(json.dumps(side))
    (d / f"{eid}.labels.json").write_text(json.dumps({"outcome": outcome, "severity": "S1"}))
    return eid


def _slice(tmp_path, **second):
    d = tmp_path / "S1d"
    d.mkdir()
    _write_episode(d, 0, 2912205018, 40, y=0.2, see=True, closest_x=9.0)
    _write_episode(d, 1, 460152974, 20, y=-0.9, see=False, closest_x=40.0, reason="base_contact",
                   outcome="fail", **second)
    log = (f"에피소드 0 (seed 2912205018): 기록 40 스텝 (env 스텝 41, 4.0 초 + 저장 1.0 초) → x\n"
           f"에피소드 1 (seed 460152974): 기록 20 스텝 (env 스텝 21, 2.0 초 + 저장 0.5 초) → x\n")
    return d, log


def test_measure_reports_every_episode_of_a_slice(tmp_path):
    d, log = _slice(tmp_path)
    r = ms.measure(d, log_text=log, cameras_canon=CANON, scene_spec=SPEC)
    rows = {e["episode_seq"]: e for e in r["episodes"]}
    assert sorted(rows) == [0, 1]
    assert rows[0]["wall_s"] == pytest.approx(5.0) and rows[1]["wall_s"] == pytest.approx(2.5)
    assert rows[0]["rack_clearance_m"] == pytest.approx(layout.rack_inner_face_y(SPEC) - 0.2, abs=1e-6)
    assert rows[1]["rack_clearance_m"] == pytest.approx(layout.rack_inner_face_y(SPEC) - 0.9, abs=1e-6)
    assert rows[0]["bbox"]["visible_fraction"] == pytest.approx(0.5)
    assert rows[1]["bbox"]["visible_fraction"] == 0.0
    assert rows[0]["mkv_bytes"] == {n: 1000 * (i + 1) for i, n in enumerate(CANON["cameras"])}
    fixed = [n for n, c in CANON["cameras"].items() if c["type"] == "fixed"]
    assert rows[0]["closest"]["in_picture"][fixed[0]]["inside"] is True
    assert rows[1]["closest"]["in_picture"][fixed[0]]["inside"] is False
    assert rows[1]["termination"] == ["base_contact"] and rows[1]["outcome"] == "fail"

    s = r["summary"]
    assert s["episodes"] == 2
    assert s["rack_clearance_min_m"] == pytest.approx(rows[1]["rack_clearance_m"])
    assert s["termination"] == {"base_contact": 1, "time_out": 1}
    assert s["outcome"] == {"fail": 1, "near_miss": 1}
    assert s["closest_in_picture_fraction"][fixed[0]] == pytest.approx(0.5)
    assert s["bbox_area_median_px"] == pytest.approx(100 * 200)


# -- (g) the dummy's entry as the sidecar records it --
# Nothing compared a generated episode's realized entry speed with the drawn one, or its facing with its
# walk: the arrays C9 and --compare-with look at hold neither.

def test_entry_facts_reads_the_walk_and_the_facing():
    f = ms.entry_facts(_entry(angle_deg=118.6, drawn=0.8, realized=0.804))
    assert f["triggered"] is True
    assert f["speed_ratio"] == pytest.approx(1.005)
    assert f["angle_deg"] == pytest.approx(118.6)
    assert f["facing_yaw_deg"] == pytest.approx(-61.4)
    assert f["facing_off_travel_deg"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("angle_deg, facing, off", [(118.6, 180.0, 118.6), (118.6, 0.0, 61.4), (30.0, 150.0, 60.0),
                                                    (0.0, -180.0, 0.0)])
def test_a_dummy_that_does_not_face_its_walk_is_measured_off_it(angle_deg, facing, off):
    # 180: the fixed facing S1d once stood at. 150 at 30 deg: the mirrored walk. -180 is 180.
    assert ms.entry_facts(_entry(angle_deg=angle_deg, facing=facing))["facing_off_travel_deg"] == pytest.approx(off)


def test_an_entry_that_never_triggered_has_no_speed_ratio():
    f = ms.entry_facts(_entry(realized=None, triggered=False))
    assert f["triggered"] is False and f["speed_ratio"] is None


def test_a_sidecar_without_the_entry_rows_is_refused():
    for gone in ("human_entry_angle_rad_drawn", "human_facing_yaw_deg", "human_entry_speed_drawn",
                 "entry_triggered_realized"):
        with pytest.raises(ValueError, match=gone):
            ms.entry_facts({k: v for k, v in _entry().items() if k != gone})


def test_a_clean_slice_has_no_entry_problem(tmp_path):
    d, log = _slice(tmp_path)
    r = ms.measure(d, log_text=log, cameras_canon=CANON, scene_spec=SPEC)
    assert r["problems"] == []
    assert [e["entry"]["speed_ratio"] for e in r["episodes"]] == pytest.approx([1.005, 1.005])
    assert r["summary"]["entry_speed_ratio"]["min"] == pytest.approx(1.005)
    assert r["summary"]["facing_off_travel_deg_max"] == pytest.approx(0.0, abs=1e-9)


def test_an_episode_facing_away_from_its_walk_is_a_problem(tmp_path):
    d, log = _slice(tmp_path, entry=_entry(facing=180.0))
    r = ms.measure(d, log_text=log, cameras_canon=CANON, scene_spec=SPEC)
    assert len(r["problems"]) == 1 and "S1d_0001_seed460152974" in r["problems"][0] and "방향" in r["problems"][0]
    assert r["summary"]["facing_off_travel_deg_max"] == pytest.approx(118.6)


def test_a_triggered_entry_without_a_realized_speed_is_a_problem(tmp_path):
    # the window never opened or was closed before the walk: the reset left walk_ended set
    d, log = _slice(tmp_path, entry=_entry(realized=None, triggered=True))
    r = ms.measure(d, log_text=log, cameras_canon=CANON, scene_spec=SPEC)
    assert len(r["problems"]) == 1 and "실현 속도" in r["problems"][0]


@pytest.mark.parametrize("row, bad", [("human_facing_yaw_deg", float("nan")), ("human_facing_yaw_deg", float("inf")),
                                      ("human_entry_angle_rad_drawn", float("nan")),
                                      ("human_entry_speed_realized", float("nan")),
                                      ("human_entry_speed_realized", float("inf")),
                                      ("human_entry_speed_drawn", float("nan")), ("human_entry_speed_drawn", 0.0)])
def test_an_entry_row_that_is_not_a_usable_number_is_a_problem(tmp_path, row, bad):
    # NaN > tolerance is False and NaN is not None, so a sidecar with a NaN facing or
    # speed measured as a clean slice (problems [], exit 0). json writes NaN and Infinity as they are.
    d, log = _slice(tmp_path, entry={**_entry(), row: bad})
    r = ms.measure(d, log_text=log, cameras_canon=CANON, scene_spec=SPEC)
    assert len(r["problems"]) == 1 and "S1d_0001_seed460152974" in r["problems"][0] and row in r["problems"][0]
    # the row is left out of the arithmetic it would poison, not carried into the spread as NaN
    speed_row = "speed" in row
    assert r["summary"]["entry_speed_ratio"]["n"] == (1 if speed_row else 2)
    assert r["summary"]["facing_off_travel_deg_max"] == pytest.approx(0.0, abs=1e-9)
    assert r["episodes"][1]["entry"]["facing_off_travel_deg" if not speed_row else "speed_ratio"] is None


def test_main_exits_1_on_an_entry_problem_and_still_writes_the_report(tmp_path, capsys):
    d, log = _slice(tmp_path, entry=_entry(facing=180.0))
    (tmp_path / "run.log").write_text(log, encoding="utf-8")
    assert ms.main(["--episodes", str(d), "--log", str(tmp_path / "run.log"), "--out", str(tmp_path / "r.json")]) == 1
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["problems"]
    assert "PROBLEM" in capsys.readouterr().out


def test_a_log_line_of_another_run_is_refused(tmp_path):
    # The log was once joined on the sequence number alone.
    d, log = _slice(tmp_path)
    other_seed = log.replace("seed 460152974", "seed 0")
    with pytest.raises(ValueError, match="S1d_0001_seed460152974"):
        ms.measure(d, log_text=other_seed, cameras_canon=CANON, scene_spec=SPEC)
    other_steps = log.replace("기록 20 스텝", "기록 1 스텝")
    with pytest.raises(ValueError, match="S1d_0001_seed460152974"):
        ms.measure(d, log_text=other_steps, cameras_canon=CANON, scene_spec=SPEC)


def test_an_episode_missing_a_canon_camera_is_refused(tmp_path):
    # A camera absent from one sidecar was left out of that episode's total.
    d, log = _slice(tmp_path)
    side_path = next(d.glob("S1d_0001_*[0-9].json"))
    side = json.loads(side_path.read_text())
    gone = list(CANON["cameras"])[-1]
    del side["media"][gone]
    side_path.write_text(json.dumps(side))
    with pytest.raises(ValueError, match=f"S1d_0001_seed460152974.*{gone}"):
        ms.measure(d, log_text=log, cameras_canon=CANON, scene_spec=SPEC)


def test_an_episode_whose_video_file_is_missing_is_refused(tmp_path):
    d, log = _slice(tmp_path)
    next((d / "media").glob("S1d_0000_*.mkv")).unlink()
    with pytest.raises(ValueError, match="S1d_0000_seed2912205018"):
        ms.measure(d, log_text=log, cameras_canon=CANON, scene_spec=SPEC)


def test_a_recording_the_canon_does_not_list_is_refused(tmp_path):
    # A file under a name the canon has no camera for was added to the
    # episode's total and left out of the per-camera means, so the report did not add up.
    d, log = _slice(tmp_path)
    side_path = next(d.glob("S1d_0000_*[0-9].json"))
    side = json.loads(side_path.read_text())
    (d / "media/extra.mkv").write_bytes(b"\0" * 1_000_000)
    side["media"]["side_rgb"] = {"file": "media/extra.mkv", "frames": 40}
    side_path.write_text(json.dumps(side))
    with pytest.raises(ValueError, match="S1d_0000_seed2912205018.*side_rgb"):
        ms.measure(d, log_text=log, cameras_canon=CANON, scene_spec=SPEC)


def test_a_camera_that_recorded_no_frame_is_refused(tmp_path):
    # media_writer's row for a camera without a frame names no file: {"file": None, "frames": 0}.
    d, log = _slice(tmp_path)
    side_path = next(d.glob("S1d_0001_*[0-9].json"))
    side = json.loads(side_path.read_text())
    empty = list(CANON["cameras"])[0]
    side["media"][empty] = {"file": None, "frames": 0}
    side_path.write_text(json.dumps(side))
    with pytest.raises(ValueError, match=f"S1d_0001_seed460152974.*{empty}"):
        ms.measure(d, log_text=log, cameras_canon=CANON, scene_spec=SPEC)


# -- the converted dataset (``--dataset``): av1_shares and main, with only the PyAV read swapped out --
# The helper tests above call check_complete_spans and check_same_episodes directly; with both call sites
# deleted they stayed green, and a 999-frame episode holding 998 packets went into the report.

FPS = 50
KEYS = ("observation.images.head_rgb", "observation.images.cctv_rgb")
SLICE = [("S1d_0000_seed2912205018", 40), ("S1d_0001_seed460152974", 20)]  # what _slice() writes


def _dataset(tmp_path, episodes, *, indices=None):
    """A converted dataset as av1_shares reads it: the episodes table, one video file per key holding
    every episode back to back, and a data parquet. The files hold no video -- _packets stands in."""
    import pandas as pd
    root = tmp_path / "dataset"
    rows, t = [], 0.0
    for i, (eid, length) in enumerate(episodes):
        row = {"episode_index": i if indices is None else indices[i], "episode_id": eid, "length": length}
        for k in KEYS:
            row.update({f"videos/{k}/chunk_index": 0, f"videos/{k}/file_index": 0,
                        f"videos/{k}/from_timestamp": t, f"videos/{k}/to_timestamp": t + length / FPS})
        rows.append(row)
        t += length / FPS
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    for k in KEYS:
        (root / f"videos/{k}/chunk-000").mkdir(parents=True)
        (root / f"videos/{k}/chunk-000/file-000.mp4").write_bytes(b"\0" * 10)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000/file-000.parquet").write_bytes(b"\0" * 7)
    return root


def _packets(frames: int, *, drop=()):
    """One 100-byte packet per frame at the dataset's fps, as _read_packets returns a file's."""
    return lambda path: ([(i / FPS, 100) for i in range(frames) if i not in drop], "av1")


def test_av1_shares_gives_each_episode_the_packets_of_its_span(tmp_path, monkeypatch):
    root = _dataset(tmp_path, SLICE)
    monkeypatch.setattr(ms, "_read_packets", _packets(60))
    a = ms.av1_shares(root)
    assert a["per_episode"] == {0: {k: 4000 for k in KEYS}, 1: {k: 2000 for k in KEYS}}
    assert a["episodes_meta"] == {i: {"episode_id": eid, "length": n} for i, (eid, n) in enumerate(SLICE)}
    assert a["episodes"] == 2 and a["data_parquet_bytes"] == 7
    assert all(f["unassigned_packet_bytes"] == 0 and f["codec"] == "av1" for f in a["files"].values())


def test_av1_shares_refuses_a_video_cut_short_inside_an_episode(tmp_path, monkeypatch):
    root = _dataset(tmp_path, SLICE)
    monkeypatch.setattr(ms, "_read_packets", _packets(60, drop={59}))
    with pytest.raises(ValueError, match="file-000.mp4.*에피소드 1 패킷 19 · 길이 20"):
        ms.av1_shares(root)


def test_av1_shares_refuses_an_episode_index_listed_twice(tmp_path, monkeypatch):
    # Keyed by episode_index, two rows of one index fold into one and the second episode vanishes.
    root = _dataset(tmp_path, SLICE, indices=[0, 0])
    monkeypatch.setattr(ms, "_read_packets", _packets(60))
    with pytest.raises(ValueError, match="episode_index.*0"):
        ms.av1_shares(root)


def _main(tmp_path, monkeypatch, shipped):
    d, log = _slice(tmp_path)
    (tmp_path / "run.log").write_text(log, encoding="utf-8")
    root = _dataset(tmp_path, shipped)
    monkeypatch.setattr(ms, "_read_packets", _packets(sum(n for _, n in shipped)))
    return ms.main(["--episodes", str(d), "--log", str(tmp_path / "run.log"), "--dataset", str(root),
                    "--out", str(tmp_path / "report.json")])


def test_main_measures_a_slice_against_its_own_conversion(tmp_path, monkeypatch, capsys):
    assert _main(tmp_path, monkeypatch, SLICE) == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["episodes"] == 2 and report["av1"]["episodes"] == 2
    assert "배포 AV1 / 에피소드" in capsys.readouterr().out


def test_main_refuses_a_dataset_converted_from_another_run(tmp_path, monkeypatch):
    other = [SLICE[0], ("S1d_0001_seed9", 20)]
    with pytest.raises(ValueError, match="S1d_0001_seed9"):
        _main(tmp_path, monkeypatch, other)
    assert not (tmp_path / "report.json").exists()


def test_main_refuses_a_dataset_that_ships_an_episode_twice(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="배포본에만.*S1d_0000_seed2912205018"):
        _main(tmp_path, monkeypatch, [*SLICE, SLICE[0]])
    assert not (tmp_path / "report.json").exists()


def test_a_log_that_misses_an_episode_is_refused(tmp_path):
    d, log = _slice(tmp_path)
    with pytest.raises(ValueError, match="1"):
        ms.measure(d, log_text=log.splitlines()[0], cameras_canon=CANON, scene_spec=SPEC)


def test_an_empty_directory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="에피소드"):
        ms.measure(tmp_path, log_text="", cameras_canon=CANON, scene_spec=SPEC)


# -- (h) a standing step's region heights against the dimensions --
# The sensor and the articulation list the dummy's bodies in different orders; an episode whose regions stand at each
# other's heights was recorded through a pairing gone wrong, whatever the arrays C9 compares say.


def test_a_slice_standing_at_its_dimensions_has_no_height_problem(tmp_path):
    d, log = _slice(tmp_path)
    r = ms.measure(d, log_text=log, cameras_canon=CANON, scene_spec=SPEC)
    assert r["problems"] == [] and all(e["standing"] == [] for e in r["episodes"])


def test_regions_standing_at_each_others_height_are_a_problem(tmp_path):
    d, log = _slice(tmp_path, region_z=_standing_heights(swap=("BR-01", "BR-06")))
    r = ms.measure(d, log_text=log, cameras_canon=CANON, scene_spec=SPEC)
    assert len(r["problems"]) == 2
    assert all(p.startswith("(h) S1d_0001_seed460152974: ") for p in r["problems"])
    assert "BR-01" in r["problems"][0] and "BR-06" in r["problems"][1]


def test_the_heights_are_read_through_the_sidecars_region_names(tmp_path):
    # the same columns under names in another order would be other regions' heights
    d, log = _slice(tmp_path)
    side = sorted(p for p in d.glob("S1d_0000_*.json") if "labels" not in p.name)[0]
    meta = json.loads(side.read_text())
    meta["contact_region_names"] = list(reversed(meta["contact_region_names"]))
    side.write_text(json.dumps(meta))
    r = ms.measure(d, log_text=log, cameras_canon=CANON, scene_spec=SPEC)
    assert r["problems"] and all(p.startswith("(h) S1d_0000_") for p in r["problems"])


def test_an_entry_problem_keeps_its_own_tag(tmp_path):
    d, log = _slice(tmp_path, entry=_entry(facing=180.0))
    r = ms.measure(d, log_text=log, cameras_canon=CANON, scene_spec=SPEC)
    assert len(r["problems"]) == 1 and r["problems"][0].startswith("(g) ")
