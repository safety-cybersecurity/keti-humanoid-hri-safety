"""배포 parquet 컬럼 복원 · C9 대조(compare_published.main). lerobot 도 Isaac 도 없이 돈다.

이 함수의 강제 캐스팅이 실제 값 차이 네 종류를
가렸습니다. 그 넷이 여기 회귀 시험으로 있습니다.

카메라 배포본(meta/cameras.json)의 팩 대조와, 훅(rerun_seed.sh)을 가짜 Isaac 으로
끝까지 돌리는 시험도 여기 있습니다 — 둘 다 같은 작은 배포본을 씁니다.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from generation import cameras as camera_canon
from generation.compare_published import (LABEL_NOTES, NOT_COMPARED, RUN_ENVIRONMENT,
                                          RUN_ENVIRONMENT_NOTES, column_to_array,
                                          compared_episode_fields, main, published_conditions,
                                          published_pairs)
from generation.compare_repro import comparable_step_fields, compare_arrays
from generation.convert_lerobot import (PACK_ARRAYS, build_episode_meta, episode_column,
                                        video_cameras)
from generation.field_map import CAMERAS, EPISODE_FIELDS, STEP_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[2]

PAIR = [("f", "x")]


def _same(ref, pub_column):
    got = column_to_array(pub_column, ref)
    return compare_arrays({"x": ref}, {"x": got}, PAIR) == []


def test_an_integer_that_overflows_the_reference_width_is_caught():
    assert not _same(np.array([1], dtype=np.int8), np.array([257], dtype=np.int64))


def test_a_float_difference_below_the_reference_precision_is_caught():
    assert not _same(np.array([1.0], dtype=np.float32),
                     np.array([1.0 + 2 ** -30], dtype=np.float64))


def test_a_non_canonical_boolean_is_caught():
    assert not _same(np.array([True]), np.array([2], dtype=np.int64))


def test_a_string_longer_than_the_reference_width_is_caught():
    ref = np.array([["none", "none"]], dtype="<U4")
    pub = np.array([json.dumps(["noneX", "none"])])
    assert not _same(ref, pub)


def _object_column(rows):
    """A pandas list-column comes back as a 1-D object array of ndarrays."""
    col = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        col[i] = row
    return col


def test_matching_columns_still_compare_equal():
    for ref, pub in (
        (np.zeros((3, 2), dtype=np.float64), _object_column([np.zeros(2)] * 3)),
        (np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)),
        (np.array([True, False]), np.array([True, False])),
    ):
        assert _same(ref, pub), (ref.dtype, column_to_array(pub, ref).dtype)


def test_a_vector_column_that_narrowed_in_the_round_trip_is_caught():
    # 예전에는 reference dtype 으로 캐스팅해 되돌려 놓아 못 잡았다.
    ref = np.zeros((2, 3), dtype=np.float64)
    pub = _object_column([np.zeros(3, dtype=np.float32)] * 2)
    assert not _same(ref, pub)


def test_a_json_string_column_of_the_same_values_compares_equal():
    ref = np.array([["none", "transient"], ["none", "none"]], dtype="<U11")
    pub = np.array([json.dumps(list(row)) for row in ref])
    assert _same(ref, pub)


def test_string_width_alone_is_not_a_mismatch():
    # <U11 npz 와 파싱으로 얻은 <U9 는 값이 같으면 같다. 폭은 인코딩 산물이다.
    ref = np.array(["transient"], dtype="<U11")
    got = np.array(["transient"])
    assert got.dtype != ref.dtype
    assert compare_arrays({"x": ref}, {"x": got}, PAIR) == []


def test_a_numeric_dtype_difference_is_reported_as_such():
    out = compare_arrays({"x": np.ones(2, dtype=np.float32)},
                         {"x": np.ones(2, dtype=np.float64)}, PAIR)
    assert len(out) == 1 and "dtype" in out[0]


def test_an_empty_column_does_not_crash():
    assert column_to_array(np.array([]), np.zeros(0)).size == 0


# ── Which columns C9 compares: meta/cameras.json switches the camera arrays on ──

EPISODE = "S1a_0000_seed1"
CAMERA_ARRAYS = {f.schema: f.npz for f in STEP_FIELDS if f.condition == CAMERAS and f.kind == "array"}


def _tiny_cameras(downsample=None, pack=True):
    """The repository's camera canon at a few pixels: what a distribution ships, small enough
    to write here. Its pack, with ``pack``, is the fixture's literal (make_fixture.canon_with_pack
    -- name, camera, dtypes) at the identity downsample unless ``downsample`` says otherwise; never
    the repository canon's own, which the canon switches off."""
    from validator.make_fixture import canon_with_pack
    doc = json.loads((REPO_ROOT / "meta" / "cameras.json").read_text(encoding="utf-8"))
    doc.pop("packs", None)
    for cam in doc["cameras"].values():
        cam["width"], cam["height"] = 8, 6
    if not pack:
        return doc
    return canon_with_pack(doc, downsample or {"width": 8, "height": 6, "every_n_steps": 1})


def _pack_name(cameras=None):
    return next(iter((cameras or _tiny_cameras())["packs"]))


def _recorded_pack(cameras, n, seed):
    """A pack as media_writer records one: every step, at the pack camera's full size."""
    pack = cameras["packs"][_pack_name(cameras)]
    cam = cameras["cameras"][pack["camera"]]
    rng = np.random.default_rng(seed)
    size = (n, cam["height"], cam["width"])
    return {"depth": rng.integers(1, 5000, size).astype(pack["depth"]["dtype"]),
            "mask": rng.integers(0, 15, size).astype(pack["mask"]["dtype"]),
            "step_index": np.arange(n, dtype=np.int32)}


def _packless_cameras():
    """_tiny_cameras() with the packs switched off -- what a distribution ships
    once the repository canon declares none."""
    return _tiny_cameras(pack=False)


def _write_media(regen, episode, n, pack, cameras):
    """The sidecar's media, laid out as media_writer lays it out under <out>/media/ -- the
    rerun's recorder names its pack after its own canon (here the fixture pack's name), whatever
    a distribution ships; ``pack`` None records none. The videos are placeholder bytes: nothing
    on the C9 path decodes them. The stamp is ``cameras``' fingerprint: a rerun recorded under
    the canon the distribution ships, unless a test says otherwise."""
    (regen / "media").mkdir()
    media = {}
    for cam in sorted(set(video_cameras().values())):
        rel = f"media/{episode}.{cam}.mkv"
        (regen / rel).write_bytes(b"placeholder -- C9 never decodes the video")
        media[cam] = {"file": rel, "frames": n}
    if pack is not None:
        rel = f"media/{episode}.{_pack_name()}.pack.npz"
        np.savez_compressed(regen / rel, **pack)
        media["pack"] = {"file": rel, "frames": n}
    media["rerendered_steps"] = []
    media["blank_frames"] = {cam: 0 for cam in sorted(set(video_cameras().values()))}   # as the recorder counts
    # ... and stamps the canon it recorded with (media_files reads it as a member that is no stream)
    media["canon_fingerprint"] = camera_canon.recording_fingerprint(cameras)
    return media


def _camera_dist(tmp_path, cameras_json):
    """``cameras_json``: False ships none; True ships _tiny_cameras(); a dict ships that
    document and a str those bytes as they are."""
    root = tmp_path / "published"
    (root / "meta").mkdir(parents=True)
    if cameras_json is not False:
        doc = _tiny_cameras() if cameras_json is True else cameras_json
        (root / "meta" / "cameras.json").write_text(
            doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
    return root


def test_a_distribution_without_cameras_json_compares_what_it_always_did(tmp_path):
    root = _camera_dist(tmp_path, cameras_json=False)
    assert published_conditions(root) == ()
    assert published_pairs(root) == comparable_step_fields()
    assert not set(CAMERA_ARRAYS) & {s for s, _ in published_pairs(root)}


def test_a_distribution_with_cameras_json_adds_the_box_and_the_poses(tmp_path):
    root = _camera_dist(tmp_path, cameras_json=True)
    assert published_conditions(root) == (CAMERAS,)
    compared = {s for s, _ in published_pairs(root)}
    assert set(CAMERA_ARRAYS) <= compared
    assert not any(s.startswith("observation.images.") for s in compared)


NUCLEUS_H1 = "{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd"
DUMMY = "generation/dummy/build/keti_dummy_ts15066_v1.usd"
# Just enough canon for build_episode_meta to make a record: the fields C9 compares come out of
# the sidecar, never out of this.
_SCENARIO = {"domain": "baseline", "task_archetype": "locomotion", "hazard_class": "collision",
             "safety_function": "ssm", "rules": [], "scene": {"human_model": "keti_dummy_ts15066_v1"}}
_CANON = dict(
    scenarios={"scenarios": [dict(_SCENARIO, scenario_id=s) for s in ("S1a", "S1d")]},
    joint_map={"robot_model": "unitree_h1", "dof": 2, "joint_names": ["a", "b"]},
    assets={"robots": {"unitree_h1": {"usd_path": NUCLEUS_H1, "license": "BSD-3-Clause",
                                      "redistributed": False, "embodiment": "biped"}},
            "scene": {"keti_dummy_ts15066_v1": {"usd_path": DUMMY, "license": "CC-BY-4.0",
                                                "redistributed": True}}},
    rules={"rules": []})
_LABEL = {"outcome": "nominal", "severity": "S0", "triggered_rules": [], "not_evaluated": [],
          "contact_observed": False, "contact_detection_limits": {"min_force_n": 0.5}}


def _sidecar(episode, n):
    """The episode json as the recorder writes one: every key the converter publishes out of it,
    with a value of each kind -- an object, a list, a float, a bool, an integer, a null, a sentence."""
    seq = int(episode.split("_")[1])
    return {
        "episode_id": episode, "scenario_id": episode.split("_")[0], "episode_seq": seq,
        "run_start_seq": 0, "seed": int(episode.rsplit("seed", 1)[1]), "steps": n,
        "degenerate": False, "robot_model": "unitree_h1", "action_space": "position_rad",
        "asset_paths": {"robot": NUCLEUS_H1, "dummy": DUMMY},
        "asset_paths_resolved": {"robot": "https://example/h1.usd", "dummy": DUMMY},
        "imu_link": "pelvis",
        "termination": {"terminated": False, "truncated": True, "reason": ["time_out"]},
        "sim_backend": {"isaacsim": "6.0.1", "isaaclab": "v3.0.0", "physics": "physx",
                        "lerobot": "0.6.1", "generator_commit": "a" * 40, "generator_dirty": False},
        "provisional": True,
        "ssm_params": {"human_speed": 1.6, "robot_stop_time": 0.34, "position_uncertainty_robot": 0.0},
        "human_region_radius": 0.1, "hold_release_step": -1, "hold_contaminated_steps": 0,
        "contact_onset_unresolved_regions": ["BR-12"],
        "contact_onset_unresolved_reason": "발이 정강이를 치는 접촉은 잡지 못함",
        "contact_type_unresolved_reason": None,
        "degenerate_reason": None, "box_released_step": 312,
        "camera_in_box_steps": {"head_rgb": 0, "left_wrist_rgb": 17},
        "task_stages": [{"name": "enter", "completed": True, "end_step": 140, "duration_s": 2.8},
                        {"name": "turn", "completed": False, "end_step": -1, "duration_s": None}],
        "human_stature_m": 1.74, "task_time_limit_s": 20.0,
        "randomization": {"walk_speed_drawn": 0.3278476297855377, "sensor_delay_steps": 4,
                          "human_entry_speed_realized": 1.1861921548843384,
                          "heading_correction_ang_vel_z_range": [-1.0, 1.0]},
    }


def _published_episode(sidecar, episode_index):
    """The episode's row as the converter publishes it: build_episode_meta, then episode_column."""
    record = build_episode_meta(sidecar, _LABEL, step_dt=0.02, **_CANON)
    return {"episode_index": episode_index, **{k: episode_column(v) for k, v in record.items()}}


def _write_episode(tmp_path, *, cameras_json, regenerated_cameras, published_cameras,
                   tamper=None, ship_pack=None, episode=EPISODE, episode_index=0,
                   regenerated_pack=None):
    """A one-episode regeneration beside a distribution holding that episode, just enough for
    compare_published.main.

    The regeneration: the npz and sidecar, and when recorded with cameras the sidecar's media
    (placeholder videos and the pack, _write_media). The distribution: the two parquet tables,
    every compared column carrying the npz's rows; for a camera distribution also its
    meta/cameras.json and, when that declares a pack, this episode's pack at
    packs/<name>/episode_<index>.npz, which ``ship_pack`` makes out of the recorded pack
    (default: a copy -- the identity downsample). ``regenerated_pack`` says whether the rerun
    recorded a pack (default: when the shipped canon declares one; the pack's shape is then
    _tiny_cameras()'s). With ``episode_index`` above 0 the distribution also lists a decoy
    episode at index 0, with a pack of its own."""
    n = 3
    arrays = {npz: np.arange(n * 2, dtype=np.float64).reshape(n, 2)
              for _, npz in comparable_step_fields()}
    arrays["timestamp"] = np.arange(n) * 0.02
    # A pose per camera of the canon (the map's camera arrays), each its own value so that a mix-up shows
    poses = [npz for schema, npz in CAMERA_ARRAYS.items() if schema.startswith("observation.camera_pose.")]
    camera = {"bbox_human": np.tile(np.array([1, 2, 3, 4], np.int32), (n, 1)),
              **{npz: np.full((n, 7), j, np.float32) for j, npz in enumerate(poses)}}
    assert set(camera) == set(CAMERA_ARRAYS.values())
    cameras = cameras_json if isinstance(cameras_json, dict) else _tiny_cameras()
    declares = bool(cameras.get("packs"))
    if regenerated_pack is None:
        regenerated_pack = declares
    # A recorded pack only where one is recorded or shipped: shaped by the shipped canon's pack,
    # or -- a rerun that records one the distribution does not declare -- by the fixture pack.
    recorded = (_recorded_pack(cameras if declares else _tiny_cameras(), n, seed=1)
                if declares or regenerated_pack else None)
    regen = tmp_path / "regenerated"
    regen.mkdir()
    np.savez(regen / f"{episode}.npz", **arrays, **(camera if regenerated_cameras else {}))
    sidecar = _sidecar(episode, n)
    if regenerated_cameras:
        sidecar["media"] = _write_media(regen, episode, n, recorded if regenerated_pack else None, cameras)
    (regen / f"{episode}.json").write_text(json.dumps(sidecar), encoding="utf-8")

    root = _camera_dist(tmp_path, cameras_json)
    (root / "meta" / "episodes").mkdir()
    listed = [(episode, episode_index)] + ([("S1a_0009_seed9", 0)] if episode_index else [])
    pd.DataFrame([_published_episode(_sidecar(e, n), i) for e, i in listed]).to_parquet(
        root / "meta" / "episodes" / "file-000.parquet")
    columns = {"episode_index": [episode_index] * n, "frame_index": list(range(n))}
    published = dict(arrays, **(camera if published_cameras else {}))
    if tamper:
        published[tamper] = published[tamper].copy()
        published[tamper][-1, -1] += 1
    for schema, npz in comparable_step_fields((CAMERAS,)):
        if npz in published:
            columns[schema] = list(published[npz])
    (root / "data").mkdir()
    pd.DataFrame(columns).to_parquet(root / "data" / "file-000.parquet")
    if cameras_json is not False and declares:
        packs = root / "packs" / _pack_name(cameras)
        packs.mkdir(parents=True)
        np.savez_compressed(packs / f"episode_{episode_index:06d}.npz",
                            **(ship_pack or dict)(recorded))
        if episode_index:
            np.savez_compressed(packs / "episode_000000.npz", **_recorded_pack(cameras, n, seed=9))
    return regen, root


def _run(monkeypatch, capsys, regen, root, episode=EPISODE, run_start=None):
    argv = ["compare_published.py", "--episode-id", episode, "--seed", "1",
            "--regenerated", str(regen), "--published", str(root)]
    if run_start is not None:
        argv += ["--run-start-seq", str(run_start)]
    monkeypatch.setattr("sys.argv", argv)
    code = main()
    return code, capsys.readouterr().out


def test_main_counts_the_rerun_from_the_runs_first_seq(tmp_path, monkeypatch, capsys):
    # 13번을 10번에서 시작한 실행으로 다시 만들면 네 건이 나온다. 0번부터 세면 14건을 기대하게 된다.
    episode = "S1a_0013_seed1"
    regen, root = _write_episode(tmp_path, cameras_json=False, regenerated_cameras=False,
                                 published_cameras=False, episode=episode)
    for seq in (10, 11, 12):                      # the run's earlier episodes; only the last is compared
        (regen / f"S1a_{seq:04d}_seed9.npz").write_bytes((regen / f"{episode}.npz").read_bytes())
    code, out = _run(monkeypatch, capsys, regen, root, episode=episode, run_start=10)
    assert code == 0, out
    code, out = _run(monkeypatch, capsys, regen, root, episode=episode)          # as a run from 0
    assert code == 1 and "재생성 에피소드 수가 다릅니다: 4 != 14" in out, out


def test_main_refuses_a_start_past_the_episode(tmp_path, monkeypatch, capsys):
    regen, root = _write_episode(tmp_path, cameras_json=False, regenerated_cameras=False,
                                 published_cameras=False)
    code, out = _run(monkeypatch, capsys, regen, root, run_start=3)
    assert code == 1 and "run_start_seq 3 이 에피소드 번호 0 보다 큽니다" in out, out


def _publish_backend(root, backend):
    path = root / "meta" / "episodes" / "file-000.parquet"
    eps = pd.read_parquet(path)
    eps["sim_backend"] = [json.dumps(backend)] * len(eps)
    eps.to_parquet(path)


RECORDED = "a" * 40
HEAD = "b" * 40


@pytest.mark.parametrize("now, expect", [
    ({"generator_commit": HEAD, "generator_dirty": False}, f"기록된 생성 커밋 {RECORDED[:12]} ≠ 지금 HEAD {HEAD[:12]}"),
    ({"generator_commit": RECORDED, "generator_dirty": True}, "지금 작업 트리에 커밋되지 않은 변경이 있습니다"),
    ({"generator_commit": RECORDED, "generator_dirty": False}, None),
    ({"generator_commit": None, "generator_dirty": None}, None),
], ids=["another commit", "dirty tree", "the same code", "git cannot say"])
def test_a_failed_reproduction_says_when_the_rerun_was_other_code(tmp_path, monkeypatch, capsys, now, expect):
    """C9 는 기록된 커밋이 아니라 지금 작업 트리로 다시 돌린다. 실패했을 때 첫 질문이 「같은 코드였나」다."""
    import generation.compare_published as cp
    regen, root = _write_episode(tmp_path, cameras_json=False, regenerated_cameras=False,
                                 published_cameras=False, tamper="joint_pos")
    _publish_backend(root, {"generator_commit": RECORDED, "generator_dirty": False})
    monkeypatch.setattr(cp.provenance, "read", lambda repo_root: now)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "재현 불일치" in out, out
    assert ("참고:" in out) == (expect is not None), out
    if expect:
        assert expect in out and "재현은 지금 작업 트리의 코드로 돌았습니다" in out, out


@pytest.mark.parametrize("damage, phrase", [
    pytest.param("frames", "프레임 수가 다릅니다", id="frame count"),
    pytest.param("column", "배포본에 컬럼이 없습니다", id="published column missing"),
    pytest.param("array", "재생성 에피소드에 배열이 없습니다", id="regenerated array missing"),
    pytest.param("meta", "에피소드 메타 notes.hold_release_step", id="episode metadata"),
])
def test_every_failure_after_the_published_row_is_known_says_when_the_rerun_was_other_code(
        tmp_path, monkeypatch, capsys, damage, phrase):
    """When the code differs the first thing to go wrong is usually the length of the run, and that return
    had no note -- the note was attached to the array mismatch alone."""
    import generation.compare_published as cp
    regen, root = _write_episode(tmp_path, cameras_json=False, regenerated_cameras=False,
                                 published_cameras=False)
    _publish_backend(root, {"generator_commit": RECORDED, "generator_dirty": False})
    monkeypatch.setattr(cp.provenance, "read",
                        lambda repo_root: {"generator_commit": HEAD, "generator_dirty": False})
    if damage == "frames":
        sidecar = regen / f"{EPISODE}.json"
        sidecar.write_text(json.dumps(dict(json.loads(sidecar.read_text(encoding="utf-8")), steps=99)),
                           encoding="utf-8")
    elif damage == "meta":
        _rewrite(regen / f"{EPISODE}.json", _set("hold_release_step", 7))
    elif damage == "column":
        data = root / "data" / "file-000.parquet"
        pd.read_parquet(data).drop(columns=["observation.state.joint_pos"]).to_parquet(data)
    else:
        arrays = dict(np.load(regen / f"{EPISODE}.npz"))
        arrays.pop("joint_pos")
        np.savez(regen / f"{EPISODE}.npz", **arrays)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and phrase in out, out
    assert f"기록된 생성 커밋 {RECORDED[:12]} ≠ 지금 HEAD {HEAD[:12]}" in out, out


def test_a_failure_before_the_published_row_is_known_has_no_note_to_give(tmp_path, monkeypatch, capsys):
    import generation.compare_published as cp
    regen, root = _write_episode(tmp_path, cameras_json=False, regenerated_cameras=False,
                                 published_cameras=False)
    monkeypatch.setattr(cp.provenance, "read",
                        lambda repo_root: {"generator_commit": HEAD, "generator_dirty": False})
    code, out = _run(monkeypatch, capsys, regen, root, run_start=3)
    assert code == 1 and "참고:" not in out, out


def test_a_reproduction_that_holds_says_nothing_about_the_commit(tmp_path, monkeypatch, capsys):
    import generation.compare_published as cp
    regen, root = _write_episode(tmp_path, cameras_json=False, regenerated_cameras=False,
                                 published_cameras=False)
    _publish_backend(root, {"generator_commit": RECORDED, "generator_dirty": False})
    monkeypatch.setattr(cp.provenance, "read", lambda repo_root: {"generator_commit": HEAD, "generator_dirty": False})
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0 and "참고:" not in out, out


def test_main_without_cameras_json_compares_the_unconditional_columns(tmp_path, monkeypatch, capsys):
    # Camera columns present on both sides are still not compared: no cameras.json, no
    # camera condition -- the v0.1 comparison, column for column. The regeneration's pack
    # is there too and is not read either.
    regen, root = _write_episode(tmp_path, cameras_json=False, regenerated_cameras=True,
                                 published_cameras=True, tamper="bbox_human")
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0, out
    assert f"· {len(comparable_step_fields())} 컬럼" in out
    assert "팩" not in out and NOT_COMPARED not in out, out


def test_main_without_cameras_json_says_columns_frames_and_metadata_alone(
        tmp_path, monkeypatch, capsys):
    # The v0.1 path, as a v0.1 regeneration arrives: no media in the sidecar, no packs/ in
    # the distribution. Nothing looks for either, and the line says nothing of a pack or of RGB.
    regen, root = _write_episode(tmp_path, cameras_json=False, regenerated_cameras=False,
                                 published_cameras=False)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0, out
    assert out == (f"재현 확인: {EPISODE} · {len(comparable_step_fields())} 컬럼 · 3 프레임 · "
                   f"에피소드 메타 {len(compared_episode_fields())} 필드 · "
                   f"notes 멤버 {len(_compared_members(root))}\n")


def test_main_with_cameras_json_compares_the_camera_columns_too(tmp_path, monkeypatch, capsys):
    regen, root = _write_episode(tmp_path, cameras_json=True, regenerated_cameras=True,
                                 published_cameras=True)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0, out
    assert f"· {len(comparable_step_fields((CAMERAS,)))} 컬럼" in out


def test_main_with_cameras_json_fails_on_a_differing_box(tmp_path, monkeypatch, capsys):
    regen, root = _write_episode(tmp_path, cameras_json=True, regenerated_cameras=True,
                                 published_cameras=True, tamper="bbox_human")
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "observation.bbox2d.human" in out, out


def test_main_with_cameras_json_fails_on_a_missing_camera_column(tmp_path, monkeypatch, capsys):
    regen, root = _write_episode(tmp_path, cameras_json=True, regenerated_cameras=True,
                                 published_cameras=False)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "배포본에 컬럼이 없습니다" in out, out


def test_main_with_cameras_json_fails_when_the_regeneration_has_no_camera_arrays(
        tmp_path, monkeypatch, capsys):
    # e.g. a camera episode regenerated without cameras: a named failure, not a KeyError.
    regen, root = _write_episode(tmp_path, cameras_json=True, regenerated_cameras=False,
                                 published_cameras=True)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "재생성 에피소드에 배열이 없습니다" in out, out


# ── A distribution that leaves out the step columns its schema calls optional ──
#
# Schema 0.2 lets a release leave out the optional step columns (meta/safety_schema.json optional_step_fields): a
# simulator records them, a release may still not carry them. C9 compares a regenerated episode against the columns the
# release carries, and a column the release's own schema lets it leave out is not owed -- a required or conditional
# column is, and an optional one that is there is compared as any other.

def _ship_schema(root, change=None):
    """The repository's meta/safety_schema.json in the distribution, ``change`` applied to it first."""
    doc = json.loads((REPO_ROOT / "meta" / "safety_schema.json").read_text(encoding="utf-8"))
    if change:
        change(doc)
    (root / "meta" / "safety_schema.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return doc


def _drop_step_columns(root, columns):
    path = root / "data" / "file-000.parquet"
    pd.read_parquet(path).drop(columns=list(columns)).to_parquet(path)


def _optional_compared():
    optional = set(json.loads((REPO_ROOT / "meta" / "safety_schema.json").read_text(encoding="utf-8"))
                   ["optional_step_fields"])
    return [s for s, _ in comparable_step_fields() if s in optional]


def test_a_distribution_without_its_optional_columns_reproduces_and_says_how_many_were_left_out(
        tmp_path, monkeypatch, capsys):
    regen, root = _plain(tmp_path)
    _ship_schema(root)
    left_out = _optional_compared()
    assert left_out
    _drop_step_columns(root, left_out)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0, out
    assert f"· {len(comparable_step_fields()) - len(left_out)} 컬럼 ·" in out, out
    assert out.rstrip("\n").endswith(f" · 배포본이 싣지 않은 선택 열 {len(left_out)}개는 대조하지 않음"), out


def test_a_distribution_that_carries_every_column_says_nothing_of_columns_left_out(tmp_path, monkeypatch, capsys):
    regen, root = _plain(tmp_path)
    _ship_schema(root)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0 and f"· {len(comparable_step_fields())} 컬럼 ·" in out and "선택 열" not in out, out


@pytest.mark.parametrize("column", ["safety.min_distance_human", "observation.root_pose"])
def test_a_distribution_without_a_required_column_still_fails(tmp_path, monkeypatch, capsys, column):
    regen, root = _plain(tmp_path)
    schema = _ship_schema(root)
    assert column in schema["required_step_fields"] and column not in schema["optional_step_fields"]
    _drop_step_columns(root, [column, *_optional_compared()])
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and f"배포본에 컬럼이 없습니다: {column}" in out, out


def test_an_optional_column_that_is_there_and_differs_fails(tmp_path, monkeypatch, capsys):
    regen, root = _write_episode(tmp_path, cameras_json=False, regenerated_cameras=False,
                                 published_cameras=False, tamper="joint_pos")
    _ship_schema(root)
    assert "observation.state.joint_pos" in _optional_compared()
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "재현 불일치" in out and "observation.state.joint_pos" in out, out


def test_an_optional_column_the_distributions_scenario_requires_is_left_out_like_any_other(
        tmp_path, monkeypatch, capsys):
    """The comparison takes the columns that are there. Whether the distribution owes an optional column -- its
    scenario names it in required_fields -- is the validator's C8 to say
    (tests/validator/test_optional_step_fields.py), and the success line counts the column as left out."""
    regen, root = _plain(tmp_path)
    _ship_schema(root)
    scenarios = json.loads((REPO_ROOT / "meta" / "scenarios.json").read_text(encoding="utf-8"))
    scenarios["scenarios"] = [s for s in scenarios["scenarios"] if s["scenario_id"] == EPISODE.split("_")[0]]
    (required,) = [s["required_fields"] for s in scenarios["scenarios"]]
    (root / "meta" / "scenarios.json").write_text(json.dumps(scenarios, ensure_ascii=False), encoding="utf-8")
    owed = [c for c in _optional_compared() if c in required]
    assert owed
    _drop_step_columns(root, owed)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0, out
    assert out.rstrip("\n").endswith(f" · 배포본이 싣지 않은 선택 열 {len(owed)}개는 대조하지 않음"), out


@pytest.mark.parametrize("schema", ["none shipped", "no optional_step_fields", "not a list", "not JSON"])
def test_a_distribution_whose_schema_names_no_optional_column_owes_every_column(tmp_path, monkeypatch, capsys,
                                                                                schema):
    """A release before 0.2 (no optional list in its schema), or one whose schema does not read, owes every column the
    comparison takes, as the comparison always did."""
    regen, root = _plain(tmp_path)
    if schema == "no optional_step_fields":
        _ship_schema(root, lambda doc: doc.pop("optional_step_fields"))
    elif schema == "not a list":
        _ship_schema(root, lambda doc: doc.update(optional_step_fields="observation.state.joint_pos"))
    elif schema == "not JSON":
        (root / "meta" / "safety_schema.json").write_text("{", encoding="utf-8")
    _drop_step_columns(root, ["observation.state.joint_pos"])
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "배포본에 컬럼이 없습니다: observation.state.joint_pos" in out, out


# ── The episode's metadata: what the converter copies out of the sidecar ──
#
# The step columns were all C9 compared. The episode row carries more out of the same recording --
# the randomization the run drew and realized, why it ended, the step the hold let go (a label
# input), the SSM parameters behind separation_required -- and a rerun that gave every column back
# bit for bit could still have written other values there: a speed measured over the wrong window
# does exactly that behind identical physics.

def _rewrite(path, change):
    """Apply ``change`` to the JSON document at ``path`` (a sidecar) and write it back."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _republish(root, column, change):
    """Apply ``change`` to the published row's ``column`` -- JSON text decoded, changed, encoded."""
    path = root / "meta" / "episodes" / "file-000.parquet"
    eps = pd.read_parquet(path)
    value = eps.at[0, column]
    if isinstance(value, str) and value[:1] in "{[":
        doc = json.loads(value)
        change(doc)
        eps.at[0, column] = json.dumps(doc, ensure_ascii=False)
    else:
        eps.at[0, column] = change(value)
    eps.to_parquet(path)


def _plain(tmp_path):
    return _write_episode(tmp_path, cameras_json=False, regenerated_cameras=False,
                          published_cameras=False)


def _compared_members(root):
    notes = json.loads(pd.read_parquet(root / "meta" / "episodes" / "file-000.parquet").at[0, "notes"])
    return sorted(set(notes) - set(LABEL_NOTES) - set(RUN_ENVIRONMENT_NOTES))


def _set(*path_and_value):
    *path, value = path_and_value

    def change(doc):
        for key in path[:-1]:
            doc = doc[key]
        doc[path[-1]] = value
    return change


def _drop(*path):
    def change(doc):
        for key in path[:-1]:
            doc = doc[key]
        del doc[path[-1]]
    return change


def test_main_says_how_much_of_the_episode_row_it_compared(tmp_path, monkeypatch, capsys):
    regen, root = _plain(tmp_path)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0, out
    members = _compared_members(root)
    assert len(members) == 17, members        # the converter's sidecar members, less the run's asset root
    assert {"degenerate_reason", "box_released_step", "camera_in_box_steps",
            "task_stages", "human_stature_m", "task_time_limit_s"} <= set(members), members
    assert f" · 에피소드 메타 {len(compared_episode_fields())} 필드 · notes 멤버 {len(members)}" in out, out


@pytest.mark.parametrize("change, where", [
    pytest.param(_set("randomization", "human_entry_speed_realized", 1.21),
                 "notes.randomization.human_entry_speed_realized", id="a realized speed"),
    pytest.param(_set("hold_release_step", 120), "notes.hold_release_step", id="the hold's release"),
    pytest.param(_set("ssm_params", "robot_stop_time", 0.42), "notes.ssm_params.robot_stop_time",
                 id="an SSM parameter"),
    pytest.param(_set("contact_type_unresolved_reason", "다른 문장"), "notes.contact_type_unresolved_reason",
                 id="a null that became a sentence"),
    pytest.param(_set("degenerate_reason", "사람이 첫 접촉 전에 넘어짐"), "notes.degenerate_reason",
                 id="a null reason that became a sentence"),
    pytest.param(_set("box_released_step", 313), "notes.box_released_step", id="the step the box was let go"),
    pytest.param(_set("camera_in_box_steps", "left_wrist_rgb", 18), "notes.camera_in_box_steps.left_wrist_rgb",
                 id="the steps a camera sat inside the box"),
    pytest.param(_set("task_stages", 1, "end_step", 141), "notes.task_stages[1].end_step",
                 id="the step a stage ended at"),
    pytest.param(_set("human_stature_m", 1.75), "notes.human_stature_m", id="the person's stature"),
    pytest.param(_set("task_time_limit_s", 25.0), "notes.task_time_limit_s", id="the task's time limit"),
    pytest.param(_set("contact_onset_unresolved_regions", ["BR-12", "BR-11"]),
                 "notes.contact_onset_unresolved_regions[1]", id="a list one longer"),
    pytest.param(_drop("randomization", "walk_speed_drawn"), "notes.randomization.walk_speed_drawn",
                 id="a drawn value the rerun did not write"),
    pytest.param(_set("randomization", "sensor_delay_steps", 4.0), "notes.randomization.sensor_delay_steps",
                 id="an integer that became a float"),
    pytest.param(_set("provisional", 1), "notes.provisional", id="true that became 1"),
    pytest.param(_set("human_region_radius", 0.1000001), "notes.human_region_radius", id="a radius"),
    pytest.param(_set("imu_link", "torso_link"), "notes.imu_link", id="the IMU link"),
    pytest.param(_set("termination", "reason", ["fall_orientation"]), "termination.reason[0]",
                 id="why the episode ended"),
    pytest.param(_set("degenerate", True), "degenerate", id="degenerate"),
    pytest.param(_set("action_space", "torque_nm"), "action_space", id="the action space"),
    pytest.param(_set("sim_backend", "physics", "other"), "measurement_method", id="the physics engine"),
    pytest.param(_set("robot_model", "unitree_g1"), "robot_model", id="the robot"),
    pytest.param(_set("scenario_id", "S1d"), "scenario_id", id="the scenario"),
])
def test_main_fails_on_a_rerun_that_wrote_other_episode_metadata(tmp_path, monkeypatch, capsys,
                                                                 change, where):
    regen, root = _plain(tmp_path)
    _rewrite(regen / f"{EPISODE}.json", change)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "재현 불일치 1 건" in out, out
    assert f"에피소드 메타 {where}:" in out, out
    assert "재현 확인" not in out, out


def test_main_shows_both_values_where_the_metadata_parts(tmp_path, monkeypatch, capsys):
    regen, root = _plain(tmp_path)
    _rewrite(regen / f"{EPISODE}.json", _set("randomization", "human_entry_speed_realized", 1.21))
    code, out = _run(monkeypatch, capsys, regen, root)
    assert ("에피소드 메타 notes.randomization.human_entry_speed_realized: "
            "배포 1.1861921548843384 · 재생성 1.21") in out, out


def test_main_fails_on_a_published_row_that_says_other_than_the_recording(tmp_path, monkeypatch, capsys):
    # The other side of the same comparison: the rerun is the recording, the row was changed.
    regen, root = _plain(tmp_path)
    _republish(root, "notes", _set("randomization", "walk_speed_drawn", 0.5))
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "에피소드 메타 notes.randomization.walk_speed_drawn: 배포 0.5" in out, out


@pytest.mark.parametrize("column, value, expect", [
    ("run_start_seq", 3, "에피소드 메타 run_start_seq: 배포 3 · 재생성 0"),
    ("randomization_seed", 2, "에피소드 메타 randomization_seed: 배포 2 · 재생성 1"),
])
def test_main_fails_on_a_published_number_the_rerun_does_not_give_back(tmp_path, monkeypatch, capsys,
                                                                       column, value, expect):
    # Fields main() cannot be made to differ from the rerun's side alone: the hook is handed the
    # published seed and start, and a rerun of another seed is "another episode" before any of this.
    regen, root = _plain(tmp_path)
    _republish(root, column, lambda v: value)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and expect in out, out


def test_c9_leaves_out_the_run_environment_alone():
    # Held as written, not derived: widening either tuple quietly stops comparing a seed output, and
    # every test above derives what it expects from these two.
    assert RUN_ENVIRONMENT == ("sim_backend",)
    assert RUN_ENVIRONMENT_NOTES == ("asset_paths_resolved",)


def _termination_as_text(doc):
    doc["termination"] = json.dumps(doc["termination"], ensure_ascii=False)     # as episode_column writes it


@pytest.mark.parametrize("change, why", [
    pytest.param(_termination_as_text, "termination 의 형이 dict 이 아닙니다",
                 id="termination as the JSON text the row holds"),
    pytest.param(_set("termination", "reason", []), "termination.reason 이 비어 있지 않은 문자열 목록이 아닙니다",
                 id="no reason"),
    pytest.param(_set("degenerate", "false"), "degenerate 의 형이 bool 이 아닙니다", id="degenerate as text"),
    pytest.param(_set("run_start_seq", 1), "run_start_seq=1 가 0 과 자기 번호 0 사이가 아닙니다",
                 id="a start past the episode"),
    pytest.param(_drop("episode_seq"), "사이드카에 'episode_seq' 가 없음", id="no episode_seq"),
])
def test_main_refuses_a_rerun_sidecar_the_converter_would_refuse(tmp_path, monkeypatch, capsys,
                                                                 change, why):
    regen, root = _plain(tmp_path)
    _rewrite(regen / f"{EPISODE}.json", change)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "에피소드 메타: 재생성한 사이드카를 변환기가 받지 않음 — " in out, out
    assert why in out, out


def test_a_termination_written_as_the_rows_text_differs_in_nothing_but_its_type(tmp_path, monkeypatch,
                                                                                 capsys):
    # The published row holds termination as JSON text, so a rerun that wrote that text in place of the
    # object reads as the same string on both sides: the converter's refusal is the only finding.
    regen, root = _plain(tmp_path)
    _rewrite(regen / f"{EPISODE}.json", _termination_as_text)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "재현 불일치 1 건" in out and "termination 의 형이 dict 이 아닙니다" in out, out


@pytest.mark.parametrize("change", [
    pytest.param(_set("sim_backend", "generator_commit", "b" * 40), id="another commit"),
    pytest.param(_set("sim_backend", "generator_dirty", True), id="a dirty tree"),
    pytest.param(_set("sim_backend", "isaacsim", "6.0.2"), id="another Isaac Sim build"),
    pytest.param(_set("asset_paths_resolved", "robot", "omniverse://localhost/h1.usd"), id="another asset root"),
])
def test_main_takes_a_rerun_in_another_environment_that_gave_every_value_back(tmp_path, monkeypatch,
                                                                              capsys, change):
    # Where and with what the rerun ran, not what its seed made: the arrays and the rest of the row
    # came back, so the episode reproduced. commit_note names the commit when it does not.
    regen, root = _plain(tmp_path)
    _rewrite(regen / f"{EPISODE}.json", change)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0 and "재현 확인" in out, out


POLICY = {"source": "isaaclab-published:rsl_rl/Isaac-Velocity-Flat-H1-v0", "sha256": "c" * 64,
          "behavior": "generation.behavior:PassThrough"}


def _with_policy(regen, root, *, published=POLICY, regenerated=POLICY):
    """The recorded policy on either side (sim_backend.policy), as a table scenario's run writes it; None leaves it out."""
    if published is not None:
        _republish(root, "sim_backend", _set("policy", published))
    if regenerated is not None:
        _rewrite(regen / f"{EPISODE}.json", _set("sim_backend", "policy", regenerated))


@pytest.mark.parametrize("published, regenerated, change, where", [
    pytest.param(POLICY, dict(POLICY, behavior="mod:Other"), None, "sim_backend.policy.behavior", id="another behaviour"),
    pytest.param(POLICY, dict(POLICY, sha256="d" * 64), None, "sim_backend.policy.sha256", id="another checkpoint"),
    pytest.param(POLICY, dict(POLICY, source="file:ckpt/a.pt"), None, "sim_backend.policy.source", id="another source"),
    pytest.param(POLICY, None, None, "sim_backend.policy", id="a rerun that recorded none"),
    pytest.param(None, POLICY, None, "sim_backend.policy", id="a row that records none"),
    pytest.param(POLICY, POLICY, _set("sim_backend", "isaacsim", "6.0.2"), None, id="another Isaac Sim build"),
    pytest.param(POLICY, POLICY, _set("sim_backend", "generator_commit", "b" * 40), None, id="another commit"),
    pytest.param(None, None, None, None, id="neither records one"),
])
def test_c9_compares_the_policy_and_nothing_else_of_sim_backend(tmp_path, monkeypatch, capsys, published,
                                                                regenerated, change, where):
    regen, root = _plain(tmp_path)
    _with_policy(regen, root, published=published, regenerated=regenerated)
    if change:
        _rewrite(regen / f"{EPISODE}.json", change)
    code, out = _run(monkeypatch, capsys, regen, root)
    if where is None:
        assert code == 0 and "재현 확인" in out, out
        assert (" · 정책" in out) == (regenerated is not None), out      # said only when there was one to compare
    else:
        assert code == 1 and "재현 불일치 1 건" in out, out
        assert f"에피소드 메타 {where}: 배포 " in out, out


def test_c9_shows_both_policies_where_they_part(tmp_path, monkeypatch, capsys):
    regen, root = _plain(tmp_path)
    _with_policy(regen, root, regenerated=dict(POLICY, behavior="mod:Other"))
    code, out = _run(monkeypatch, capsys, regen, root)
    assert ('에피소드 메타 sim_backend.policy.behavior: 배포 "generation.behavior:PassThrough" · 재생성 "mod:Other"'
            in out), out


@pytest.mark.parametrize("text, phrase", [
    ("{not json", "에피소드 메타 sim_backend: 배포본의 값이 JSON 으로 읽히지 않음"),
    ("[1, 2]", "에피소드 메타 sim_backend: 배포본의 값이 JSON 객체가 아님"),
])
def test_c9_names_a_published_sim_backend_it_cannot_read(tmp_path, monkeypatch, capsys, text, phrase):
    regen, root = _plain(tmp_path)
    _republish_raw(root, "sim_backend", text)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and phrase in out, out


@pytest.mark.parametrize("member, value", [("not_evaluated", [{"rule_id": "R-X", "reason": "x"}]),
                                           ("contact_observed", True),
                                           ("contact_detection_limits", None)])
def test_main_leaves_the_labellers_notes_to_the_label_check(tmp_path, monkeypatch, capsys, member, value):
    # A rerun makes no labels. They are re-derived from the published columns and notes instead
    # (check_release_labels), which C9 has just shown came out of the seed.
    regen, root = _plain(tmp_path)
    _republish(root, "notes", _set(member, value))
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0 and "재현 확인" in out, out


def test_main_takes_nan_where_both_sides_hold_it(tmp_path, monkeypatch, capsys):
    regen, root = _plain(tmp_path)
    _rewrite(regen / f"{EPISODE}.json", _set("randomization", "walk_speed_drawn", float("nan")))
    _republish(root, "notes", _set("randomization", "walk_speed_drawn", float("nan")))
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0, out


@pytest.mark.parametrize("damage, phrase", [
    pytest.param(lambda root, regen: _republish_raw(root, "notes", "{not json"),
                 "에피소드 메타 notes: 배포본의 값이 JSON 으로 읽히지 않음", id="notes not JSON"),
    pytest.param(lambda root, regen: _republish_raw(root, "notes", "[1, 2]"),
                 "에피소드 메타 notes: 배포본의 값이 JSON 객체가 아님", id="notes not an object"),
    pytest.param(lambda root, regen: _republish_raw(root, "termination", "{not json"),
                 "에피소드 메타 termination: 배포본의 값이 JSON 으로 읽히지 않음", id="a field not JSON"),
    pytest.param(lambda root, regen: _republish_raw(root, "notes", "[" * 100_000 + "]" * 100_000),
                 "에피소드 메타 notes: 배포본의 값이 JSON 으로 읽히지 않음", id="notes nested past the decoder"),
    pytest.param(lambda root, regen: _republish_raw(root, "termination", "1" * 5000),
                 "에피소드 메타 termination: 배포본의 값이 JSON 으로 읽히지 않음", id="an integer past the digit limit"),
    pytest.param(lambda root, regen: _drop_column(root, "termination"),
                 "에피소드 메타 termination: 배포본에 없음", id="a field not published"),
    pytest.param(lambda root, regen: _drop_column(root, "notes"),
                 "에피소드 메타 notes: 배포본에 없음", id="notes not published"),
    pytest.param(lambda root, regen: _rewrite(regen / f"{EPISODE}.json", _drop("termination")),
                 "에피소드 메타 termination: 재생성한 사이드카에 termination 이 없음", id="a key the rerun did not write"),
    pytest.param(lambda root, regen: _rewrite(regen / f"{EPISODE}.json", _drop("sim_backend", "physics")),
                 "에피소드 메타 measurement_method: 재생성한 사이드카에 sim_backend.physics 이 없음",
                 id="a nested key the rerun did not write"),
])
def test_main_names_metadata_it_cannot_compare(tmp_path, monkeypatch, capsys, damage, phrase):
    regen, root = _plain(tmp_path)
    damage(root, regen)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and phrase in out, out


def _republish_raw(root, column, text):
    path = root / "meta" / "episodes" / "file-000.parquet"
    eps = pd.read_parquet(path)
    eps.at[0, column] = text
    eps.to_parquet(path)


def test_main_names_a_difference_too_deep_to_walk(tmp_path, monkeypatch, capsys):
    # Both sides nested deeper than the walk's stack and alike all the way down: the walk, not the
    # decoder, runs out. Since Python 3.12 the C decoder counts its own depth apart from
    # sys.setrecursionlimit, which bounds the walk alone -- so 900 levels decode and do not walk.
    regen, root = _plain(tmp_path)
    deep = "[" * 900 + "]" * 900
    _rewrite(regen / f"{EPISODE}.json", _set("randomization", json.loads(deep)))
    notes = json.loads(pd.read_parquet(root / "meta" / "episodes" / "file-000.parquet").at[0, "notes"])
    text = json.dumps(dict(notes, randomization="DEEP")).replace('"DEEP"', deep)
    _republish_raw(root, "notes", text)
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(600)
    try:
        code, out = _run(monkeypatch, capsys, regen, root)
    finally:
        sys.setrecursionlimit(old)
    assert code == 1 and "에피소드 메타 notes.randomization: 너무 깊이 중첩되어 대조하지 못함" in out, out


def test_main_shows_a_published_value_no_json_holds(tmp_path, monkeypatch, capsys):
    # A cell pandas hands over as something no JSON value is -- here bytes -- is one more difference.
    regen, root = _plain(tmp_path)
    path = root / "meta" / "episodes" / "file-000.parquet"
    eps = pd.read_parquet(path)
    eps["action_space"] = [b"position_rad"]
    eps.to_parquet(path)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "에피소드 메타 action_space: 배포 \"b'position_rad'\" · 재생성 \"position_rad\"" in out, out


def _drop_column(root, column):
    path = root / "meta" / "episodes" / "file-000.parquet"
    pd.read_parquet(path).drop(columns=[column]).to_parquet(path)


def test_the_metadata_c9_compares_is_what_the_converter_copies_out_of_the_sidecar():
    # compared_episode_fields comes out of field_map, LABEL_NOTES is written out here: both are
    # held against what build_episode_meta actually does. Every sidecar key and every label member
    # carries a value of its own, so where each published value came from can be read off it.
    sidecar = _sidecar(EPISODE, 3)
    for member in (*LABEL_NOTES, "provisional", "hold_contaminated_steps", "hold_release_step",
                   "ssm_params", "human_region_radius", "contact_onset_unresolved_regions",
                   "contact_onset_unresolved_reason", "contact_type_unresolved_reason",
                   "randomization", "imu_link", "asset_paths_resolved", "task_completed_step",
                   "degenerate_reason", "box_released_step", "camera_in_box_steps",
                   "task_stages", "human_stature_m", "task_time_limit_s"):
        sidecar[member] = f"sidecar:{member}"
    label = dict(_LABEL, **{m: f"label:{m}" for m in LABEL_NOTES})
    record = build_episode_meta(sidecar, label, step_dt=0.02, **_CANON)
    notes = json.loads(record["notes"])
    from_label = {m for m, v in notes.items() if v == f"label:{m}"}
    from_sidecar = {m for m, v in notes.items() if v == f"sidecar:{m}"}
    assert from_label == set(LABEL_NOTES)
    assert from_sidecar == set(notes) - set(LABEL_NOTES), sorted(set(notes) - from_label - from_sidecar)
    assert set(RUN_ENVIRONMENT_NOTES) <= from_sidecar
    sidecar_fields = {f.schema for f in EPISODE_FIELDS if f.source == "sidecar"}
    assert set(RUN_ENVIRONMENT) <= sidecar_fields
    assert {s for s, _ in compared_episode_fields()} == sidecar_fields - set(RUN_ENVIRONMENT)
    for schema, key in compared_episode_fields():
        value = sidecar
        for part in key.split("."):
            value = value[part]
        assert record[schema] == value, (schema, key)


# ── The pack: a camera distribution's depth, mask and step_index, bit for bit ──
#
# Two runs of one S1d episode were measured to agree on the pack and the box/pose
# arrays but never on an RGB frame. So C9 compares the pack and says it did not compare RGB.

def _camera_episode(tmp_path, cameras_json=True, **kw):
    return _write_episode(tmp_path, cameras_json=cameras_json, regenerated_cameras=True,
                          published_cameras=True, **kw)


def _tampered(member, change):
    """A ship_pack that publishes the recorded pack with ``change`` applied to one member."""
    def ship(pack):
        out = {k: v.copy() for k, v in pack.items()}
        if change is None:
            del out[member]
        else:
            change(out[member])
        return out
    return ship


def _one_depth_pixel(a):
    a[-1, 2, 3] += 1


def _one_mask_pixel(a):
    a[0, 1, 1] ^= 1


def _shifted_by_one(a):
    a += 1


def test_main_with_cameras_json_compares_the_pack_bit_for_bit(tmp_path, monkeypatch, capsys):
    regen, root = _camera_episode(tmp_path)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0, out
    assert f"팩 {_pack_name()} ({' · '.join(PACK_ARRAYS)}) 3 행" in out, out
    assert NOT_COMPARED in out, "성공 줄이 RGB 를 대조하지 않았다고 말하지 않습니다"


@pytest.mark.parametrize("member, change", [
    ("depth", _one_depth_pixel), ("mask", _one_mask_pixel),
    ("step_index", _shifted_by_one), ("step_index", None)],
    ids=["one depth pixel", "one mask pixel", "step_index shifted", "step_index missing"])
def test_main_fails_on_a_pack_that_differs_in_one_place(tmp_path, monkeypatch, capsys,
                                                        member, change):
    regen, root = _camera_episode(tmp_path, ship_pack=_tampered(member, change))
    code, out = _run(monkeypatch, capsys, regen, root)
    said = "배열 없음" if change is None else "값 불일치"
    assert code == 1 and f"팩 {_pack_name()}.{member}: {said}" in out, out
    assert NOT_COMPARED not in out, out


@pytest.mark.parametrize("gone", ["file", "entry", "media"])
def test_main_fails_without_the_regenerated_pack(tmp_path, monkeypatch, capsys, gone):
    # The rerun left no pack: its file, its sidecar entry, or the sidecar's whole media.
    regen, root = _camera_episode(tmp_path)
    sidecar_path = regen / f"{EPISODE}.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if gone == "file":
        (regen / sidecar["media"]["pack"]["file"]).unlink()
    elif gone == "entry":
        del sidecar["media"]["pack"]
    else:
        del sidecar["media"]
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "재생성한 에피소드의 매체(사이드카 media)를 읽지 못했습니다" in out, out


@pytest.mark.parametrize("fault", ["video file", "video frames"])
def test_main_names_regenerated_media_the_converter_would_refuse(tmp_path, monkeypatch, capsys,
                                                                fault):
    # The regenerated sidecar is read with the converter's own reader (media_files): a video it
    # would refuse is refused here too, and the sentence says media, not "pack".
    regen, root = _camera_episode(tmp_path)
    sidecar_path = regen / f"{EPISODE}.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    cam = sorted(set(video_cameras().values()))[0]
    if fault == "video file":
        (regen / sidecar["media"][cam]["file"]).unlink()
    else:
        sidecar["media"][cam]["frames"] -= 1
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "재생성한 에피소드의 매체(사이드카 media)를 읽지 못했습니다" in out, out
    assert f"media.{cam}" in out, out


def test_main_fails_without_the_published_pack(tmp_path, monkeypatch, capsys):
    regen, root = _camera_episode(tmp_path)
    rel = f"packs/{_pack_name()}/episode_000000.npz"       # schema §4's path, spelled out here
    (root / rel).unlink()
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and f"배포본에 이 에피소드의 팩이 없습니다: {rel}" in out, out


def test_main_compares_the_pack_of_the_episode_it_reproduces(tmp_path, monkeypatch, capsys):
    # This episode is episode_index 1; a decoy with a pack of its own sits at 0.
    regen, root = _camera_episode(tmp_path, episode_index=1)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0, out


def test_main_cuts_the_regenerated_pack_as_the_shipped_canon_says(tmp_path, monkeypatch, capsys):
    # The distribution's own meta/cameras.json decides, not the repository's: 8x6 here, cut to
    # every 2nd step, rows 3 apart and columns 2 apart -- unequal strides, so a swap cannot
    # pass. The published pack is cut by slicing, not by the function under test.
    cameras = _tiny_cameras({"width": 4, "height": 2, "every_n_steps": 2})
    regen, root = _camera_episode(
        tmp_path, cameras_json=cameras,
        ship_pack=lambda p: {"depth": p["depth"][::2, ::3, ::2], "mask": p["mask"][::2, ::3, ::2],
                             "step_index": p["step_index"][::2]})
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0, out
    assert f"({' · '.join(PACK_ARRAYS)}) 2 행" in out, out


def test_main_fails_when_the_shipped_cameras_json_is_not_a_camera_canon(
        tmp_path, monkeypatch, capsys):
    # Present but empty: still a camera distribution (published_conditions), so its camera canon
    # is still owed -- a named failure, not a quiet column-only pass and not a traceback.
    regen, root = _camera_episode(tmp_path, cameras_json="{}")
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "배포본 meta/cameras.json 을 카메라 정본으로 읽지 못합니다" in out, out


def test_main_names_a_packless_canon_it_cannot_read_as_what_it_is(tmp_path, monkeypatch, capsys):
    # A bbox.camera defect in a canon that declares no pack is not a pack's to describe: the
    # sentence says the camera canon could not be read, and the loader's own reason follows.
    doc = _packless_cameras()
    doc["bbox"]["camera"] = "spare_rgb"
    regen, root = _camera_episode(tmp_path, cameras_json=doc)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "배포본 meta/cameras.json 을 카메라 정본으로 읽지 못합니다" in out, out
    assert "bbox.camera 'spare_rgb'" in out and "팩을 설명하지 못합니다" not in out, out


@pytest.mark.parametrize("break_it", ["downsample not an object", "camera height null"])
def test_main_names_a_shipped_downsample_it_cannot_read(tmp_path, monkeypatch, capsys, break_it):
    # Read with the pack's other facts, before anything is compared -- these two used to reach
    # the cut and crash with AttributeError / TypeError instead.
    regen, root = _camera_episode(tmp_path)
    doc = json.loads((root / "meta" / "cameras.json").read_text(encoding="utf-8"))
    pack = doc["packs"][_pack_name(doc)]
    if break_it == "downsample not an object":
        pack["downsample"] = "every step"
    else:
        doc["cameras"][pack["camera"]]["height"] = None
    (root / "meta" / "cameras.json").write_text(json.dumps(doc), encoding="utf-8")
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "배포본 meta/cameras.json 이 팩을 설명하지 못합니다" in out, out


def test_main_does_not_read_a_pack_outside_the_distribution(tmp_path, monkeypatch, capsys):
    # The pack's name comes out of the distribution. Named "../../outside", its path leaves
    # the distribution for tmp_path/outside -- where _write_episode has put a matching pack,
    # so reading it would pass.
    cameras = _tiny_cameras()
    cameras["packs"] = {"../../outside": cameras["packs"][_pack_name(cameras)]}
    regen, root = _camera_episode(tmp_path, cameras_json=cameras)
    assert (tmp_path / "outside" / "episode_000000.npz").is_file()
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "팩 이름이 배포본 밖을 가리킵니다" in out, out


@pytest.mark.parametrize("side", ["published", "regenerated"])
def test_main_names_a_pack_it_cannot_read(tmp_path, monkeypatch, capsys, side):
    # Bit rot or a truncated download: one finding with the file's side named, not a crash.
    regen, root = _camera_episode(tmp_path)
    if side == "published":
        path = root / f"packs/{_pack_name()}/episode_000000.npz"
        said = "배포본 팩을 읽지 못함"
    else:
        sidecar = json.loads((regen / f"{EPISODE}.json").read_text(encoding="utf-8"))
        path = regen / sidecar["media"]["pack"]["file"]
        said = "재생성한 팩을 읽지 못함"
    path.write_bytes(b"not an npz")
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and f"팩 {_pack_name()}: {said}" in out, out


def test_main_fails_on_a_regenerated_pack_the_shipped_canon_does_not_describe(
        tmp_path, monkeypatch, capsys):
    # The rerun rendered at another size than the distribution says its camera has (the
    # repository's camera canon changed since): the pack cannot be cut as the published one was.
    cameras = _tiny_cameras()
    regen, root = _camera_episode(tmp_path, cameras_json=cameras)
    sidecar = json.loads((regen / f"{EPISODE}.json").read_text(encoding="utf-8"))
    wider = json.loads(json.dumps(cameras))
    for cam in wider["cameras"].values():
        cam["width"] += 2
    np.savez_compressed(regen / sidecar["media"]["pack"]["file"], **_recorded_pack(wider, 3, seed=1))
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "meta/cameras.json 의 축소로 줄이지 못함" in out, out


# ── A camera distribution without packs ──
#
# The canon switches the packs off. The columns are compared as before, there is no pack to
# compare, and the success line still says RGB was not compared -- it used to ride on the pack
# phrase, and without a pack the one sentence that keeps a green C9 from reading as covering
# the video would have gone with it.

def test_main_with_a_canon_without_packs_compares_the_columns_and_says_rgb_was_not(
        tmp_path, monkeypatch, capsys):
    regen, root = _camera_episode(tmp_path, cameras_json=_packless_cameras())
    assert not (root / "packs").exists()
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0, out
    assert out == (f"재현 확인: {EPISODE} · {len(comparable_step_fields((CAMERAS,)))} 컬럼 · 3 프레임 · "
                   f"에피소드 메타 {len(compared_episode_fields())} 필드 · "
                   f"notes 멤버 {len(_compared_members(root))} · {NOT_COMPARED}\n")


def test_main_refuses_a_regenerated_pack_the_shipped_canon_does_not_declare(tmp_path, monkeypatch,
                                                                             capsys):
    # The rerun recorded a pack while the distribution's canon declares none: the recording and
    # the canon the distribution was converted with differ -- named as the converter names it.
    regen, root = _camera_episode(tmp_path, cameras_json=_packless_cameras(), regenerated_pack=True)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "재생성한 에피소드의 매체(사이드카 media)를 읽지 못했습니다" in out, out
    assert "녹화 때와 정본이 다름" in out, out


@pytest.mark.parametrize("fault", ["video file", "video frames"])
def test_main_reads_the_regenerated_media_without_a_pack_too(tmp_path, monkeypatch, capsys, fault):
    # The reader of the regenerated sidecar is not switched off with the pack: a video the
    # converter would refuse is refused here too.
    regen, root = _camera_episode(tmp_path, cameras_json=_packless_cameras())
    sidecar_path = regen / f"{EPISODE}.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert "pack" not in sidecar["media"]
    cam = sorted(set(video_cameras().values()))[0]
    if fault == "video file":
        (regen / sidecar["media"][cam]["file"]).unlink()
    else:
        sidecar["media"][cam]["frames"] -= 1
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "재생성한 에피소드의 매체(사이드카 media)를 읽지 못했습니다" in out, out
    assert f"media.{cam}" in out, out


# ── The rerun's camera canon against the shipped one ──
#
# The hook records with the repository's camera canon; the distribution ships its own. A value
# that reaches RGB alone -- a field of view, the clipping, the warm renders -- leaves the box and
# pose columns (and a pack) as they were, and RGB is not compared: without the stamp, a rerun made
# under another canon than the shipped one was reported as reproduced.

def _restamp(regen, fingerprint, episode=EPISODE):
    """Replace the regenerated sidecar's media.canon_fingerprint -- or, with None, remove it."""
    path = regen / f"{episode}.json"
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    if fingerprint is None:
        del sidecar["media"]["canon_fingerprint"]
    else:
        sidecar["media"]["canon_fingerprint"] = fingerprint
    path.write_text(json.dumps(sidecar), encoding="utf-8")


MISMATCH = "재생성한 녹화 때 카메라 정본과 배포본 meta/cameras.json 이 다릅니다"


@pytest.mark.parametrize("pack", [True, False], ids=["with a pack", "without packs"])
def test_main_refuses_a_rerun_stamped_with_another_canon(tmp_path, monkeypatch, capsys, pack):
    regen, root = _camera_episode(tmp_path, cameras_json=_tiny_cameras() if pack else _packless_cameras())
    _restamp(regen, "0" * 64)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and MISMATCH in out, out
    assert "재현 확인" not in out, out


def test_main_refuses_a_rerun_whose_canon_differs_from_the_shipped_one_in_rgb_alone(
        tmp_path, monkeypatch, capsys):
    # With the real fingerprint: the repository canon the rerun recorded with
    # differs from the shipped one in one camera's field of view. Box, poses and pack all agree.
    shipped = _tiny_cameras()
    regen, root = _camera_episode(tmp_path, cameras_json=shipped)
    repository = json.loads(json.dumps(shipped))
    camera = [c for c in repository["cameras"] if c != repository["bbox"]["camera"]][-1]
    repository["cameras"][camera]["fov_h_deg"] += 5
    _restamp(regen, camera_canon.recording_fingerprint(repository))
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and MISMATCH in out, out


@pytest.mark.parametrize("stamp", [None, "abc", 5], ids=["missing", "not a sha256", "not a string"])
def test_main_refuses_a_rerun_without_a_readable_stamp(tmp_path, monkeypatch, capsys, stamp):
    # Named as the converter names it, not compared as if it were "another canon".
    regen, root = _camera_episode(tmp_path)
    _restamp(regen, stamp)
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "재생성한 녹화의 카메라 정본 지문을 읽지 못했습니다" in out, out
    assert ("지문을 남기지 않았습니다" if stamp is None else "sha256") in out and MISMATCH not in out, out


# ── A rerun the converter would refuse for its black frames is not a reproduction ──
#
# The converter refuses a camera whose every recorded frame is black, and a recording that did not count them
# (convert_lerobot.check_blank_frames). RGB is not compared, and the box and pose columns of a camera buried in
# some geometry are what they are -- so C9 said 「재현 확인」 of a rerun the converter would not take.

def _reblank(regen, change, episode=EPISODE):
    """Apply ``change`` to the regenerated sidecar's media.blank_frames -- or, with None, remove it."""
    path = regen / f"{episode}.json"
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    if change is None:
        del sidecar["media"]["blank_frames"]
    else:
        change(sidecar["media"]["blank_frames"], sidecar["steps"])
    path.write_text(json.dumps(sidecar), encoding="utf-8")


def _all_black(camera):
    def change(blank, steps):
        blank[camera] = steps
    return change


@pytest.mark.parametrize("pack", [True, False], ids=["with a pack", "without packs"])
def test_main_refuses_a_rerun_with_a_camera_black_in_every_frame(tmp_path, monkeypatch, capsys, pack):
    regen, root = _camera_episode(tmp_path, cameras_json=_tiny_cameras() if pack else _packless_cameras())
    camera = sorted(set(video_cameras().values()))[-1]
    _reblank(regen, _all_black(camera))
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and f"카메라 {camera} 의 프레임이 모두 검습니다(3/3)" in out, out
    assert "재현 확인" not in out, out


def test_main_takes_a_rerun_with_a_camera_black_in_some_frames(tmp_path, monkeypatch, capsys):
    # As the converter does: black for some of its frames goes through.
    regen, root = _camera_episode(tmp_path, cameras_json=_packless_cameras())
    _reblank(regen, lambda blank, steps: blank.update({cam: steps - 1 for cam in blank}))
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 0, out


@pytest.mark.parametrize("broken", [None, "not a count"], ids=["missing", "not counts"])
def test_main_refuses_a_rerun_that_did_not_count_its_black_frames(tmp_path, monkeypatch, capsys, broken):
    regen, root = _camera_episode(tmp_path, cameras_json=_packless_cameras())
    camera = sorted(set(video_cameras().values()))[0]
    _reblank(regen, None if broken is None else (lambda blank, steps: blank.update({camera: "3"})))
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1, out
    assert ("media.blank_frames 가 없습니다" if broken is None else "정수가 아닙니다") in out, out


# ── The hook end to end, with a stand-in for Isaac ──
#
# The stand-in records the argv rerun_seed.sh builds and "regenerates" by copying a prepared
# regeneration into --out; the comparison is the real compare_published.py. Nothing here
# boots a simulator.

def _fake_isaac(tmp_path, staged):
    log = tmp_path / "isaac-argv.jsonl"
    fake = tmp_path / "fake-isaac"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, shutil, sys\n"
        f"open({str(log)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "out = sys.argv[sys.argv.index('--out') + 1]\n"
        f"shutil.copytree({str(staged)!r}, out, dirs_exist_ok=True)\n",
        encoding="utf-8")
    fake.chmod(0o755)
    return fake, log


def _run_hook(episode, root, fake):
    return subprocess.run(
        [str(REPO_ROOT / "generation" / "rerun_seed.sh"), episode, "1"],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "C9_PUBLISHED_ROOT": str(root), "C9_SEQ_PYTHON": sys.executable,
             "C9_ISAAC_PYTHON": str(fake), "C9_COMPARE_PYTHON": sys.executable})


@pytest.mark.parametrize("episode, cameras", [("S1a_0000_seed1", False), ("S1d_0000_seed1", True)],
                         ids=["S1a without cameras", "S1d with cameras"])
def test_the_hook_regenerates_the_episode_s_own_scenario_and_compares_it(tmp_path, episode,
                                                                          cameras):
    regen, root = _write_episode(tmp_path, cameras_json=cameras, regenerated_cameras=cameras,
                                 published_cameras=cameras, episode=episode)
    fake, log = _fake_isaac(tmp_path, regen)
    r = _run_hook(episode, root, fake)
    assert r.returncode == 0, r.stdout + r.stderr[-800:]
    (argv,) = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert argv[argv.index("--scenario") + 1] == episode.split("_")[0], argv
    assert ("팩" in r.stdout) == cameras and (NOT_COMPARED in r.stdout) == cameras, r.stdout


def test_the_hook_fails_on_one_differing_pack_pixel(tmp_path):
    episode = "S1d_0000_seed1"
    regen, root = _camera_episode(tmp_path, episode=episode,
                                  ship_pack=_tampered("depth", _one_depth_pixel))
    fake, _ = _fake_isaac(tmp_path, regen)
    r = _run_hook(episode, root, fake)
    assert r.returncode != 0, r.stdout
    assert f"팩 {_pack_name()}.depth: 값 불일치" in r.stdout, r.stdout + r.stderr[-800:]


def _validator_c9(tmp_path, episode, root, fake):
    """C9 as the validator runs it -- real subprocess, its own environment (C9_* stripped,
    fixed PATH) -- on the repository hook. The validator does not forward the hook's test
    doors, so a wrapper opens them itself and hands over to rerun_seed.sh."""
    import shlex

    import validator.validate_dataset as vd
    wrapper = tmp_path / "hook"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        f"export C9_SEQ_PYTHON={shlex.quote(sys.executable)} C9_ISAAC_PYTHON={shlex.quote(str(fake))} "
        f"C9_COMPARE_PYTHON={shlex.quote(sys.executable)}\n"
        f"exec {shlex.quote(str(REPO_ROOT / 'generation' / 'rerun_seed.sh'))} \"$@\"\n",
        encoding="utf-8")
    wrapper.chmod(0o755)
    rep = vd.Report(root)
    eps = pd.DataFrame([{"episode_id": episode, "randomization_seed": 1, "source": "sim"}])
    vd.c9_seed_repro(vd.Dataset(root, rep), eps, str(wrapper), 1, rep)
    (finding,) = [f for f in rep.findings if f.check == "C9"]
    return finding


def test_the_validator_reports_what_the_comparison_compared(tmp_path):
    # The success line -- columns, frames, the pack, and that RGB was not compared -- reaches the
    # validation report (console and JSON), not only the hook's stdout.
    episode = "S1d_0000_seed1"
    regen, root = _camera_episode(tmp_path, episode=episode)
    fake, _ = _fake_isaac(tmp_path, regen)
    f = _validator_c9(tmp_path, episode, root, fake)
    assert f.severity == "INFO", f.message
    assert f.message.startswith(f"시드 재현 확인 — 재현 확인: {episode} · "), f.message
    assert f"팩 {_pack_name()} ({' · '.join(PACK_ARRAYS)}) 3 행" in f.message and NOT_COMPARED in f.message
    assert f"에피소드 메타 {len(compared_episode_fields())} 필드" in f.message, f.message


def test_the_validator_reports_what_a_packless_comparison_compared(tmp_path):
    # The same report for a distribution whose canon declares no pack: no pack in it, and still
    # the sentence that RGB was not compared.
    episode = "S1d_0000_seed1"
    regen, root = _camera_episode(tmp_path, cameras_json=_packless_cameras(), episode=episode)
    fake, _ = _fake_isaac(tmp_path, regen)
    f = _validator_c9(tmp_path, episode, root, fake)
    assert f.severity == "INFO", f.message
    assert f.message.startswith(f"시드 재현 확인 — 재현 확인: {episode} · "), f.message
    assert NOT_COMPARED in f.message and "팩" not in f.message, f.message


def test_the_validator_reports_why_the_comparison_failed(tmp_path):
    # One depth pixel: the reason is the comparison's sentence, then the hook's stderr (which
    # names the distribution compared against). It used to be the stderr alone.
    episode = "S1d_0000_seed1"
    regen, root = _camera_episode(tmp_path, episode=episode,
                                  ship_pack=_tampered("depth", _one_depth_pixel))
    fake, _ = _fake_isaac(tmp_path, regen)
    f = _validator_c9(tmp_path, episode, root, fake)
    assert f.severity == "ERROR", f.message
    reason = f.message.index(f"팩 {_pack_name()}.depth: 값 불일치")
    assert reason < f.message.index("훅 stderr 끝") < f.message.index("대조 대상 배포본"), f.message


_DIRECTED_SCENARIOS = {"scenarios": [{"scenario_id": "S1e",
                                      "rules": ["R-TS15066-A2-*", "R-TS15066-SSM-02", "R-HRI-CONTACT-01"]}]}


def test_a_distribution_whose_scenario_declares_the_directed_rule_compares_the_hri_arrays(tmp_path):
    from generation.field_map import HRI, STEP_FIELDS
    root = _camera_dist(tmp_path, cameras_json=False)
    (root / "meta" / "scenarios.json").write_text(json.dumps(_DIRECTED_SCENARIOS), encoding="utf-8")
    assert published_conditions(root) == (HRI,)
    compared = {s for s, _ in published_pairs(root)}
    assert {f.schema for f in STEP_FIELDS if f.condition == HRI and f.kind == "array"} <= compared


def test_an_old_scenario_compares_no_hri_array(tmp_path):
    root = _camera_dist(tmp_path, cameras_json=True)
    (root / "meta" / "scenarios.json").write_text(json.dumps({"scenarios": [
        {"scenario_id": "S1d", "rules": ["R-TS15066-A2-*", "R-TS15066-SSM-01", "R-10218-STOP-01"]}]}),
        encoding="utf-8")
    assert published_conditions(root) == (CAMERAS,)


def test_an_unreadable_scenarios_json_is_refused_rather_than_compared_less(tmp_path):
    root = _camera_dist(tmp_path, cameras_json=False)
    (root / "meta" / "scenarios.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="scenarios.json"):
        published_conditions(root)


@pytest.mark.parametrize("text", ["[]", "{}", '{"scenarios": 5}', '{"scenarios": null}'])
def test_a_scenarios_json_without_its_scenario_list_is_refused_too(tmp_path, text):
    root = _camera_dist(tmp_path, cameras_json=False)
    (root / "meta" / "scenarios.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="scenarios.json"):
        published_conditions(root)


def test_main_reports_an_unreadable_scenarios_json_as_a_failed_reproduction(
        tmp_path, monkeypatch, capsys):
    regen, root = _write_episode(tmp_path, cameras_json=False, regenerated_cameras=False,
                                 published_cameras=False)
    (root / "meta" / "scenarios.json").write_text("{", encoding="utf-8")
    code, out = _run(monkeypatch, capsys, regen, root)
    assert code == 1 and "meta/scenarios.json" in out
