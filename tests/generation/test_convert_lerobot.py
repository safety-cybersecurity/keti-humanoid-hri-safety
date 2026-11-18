"""변환기가 스키마 §5·§6 을 실제로 채우는지. lerobot 없이 순수함수만 시험한다."""

import json
from pathlib import Path

import numpy as np
import pytest

import generation.convert_lerobot as cl
from generation import cameras as camera_canon
from generation.convert_lerobot import build_episode_meta, step_frames
from generation.field_map import EPISODE_FIELDS, STEP_FIELDS, episode_fields

NUCLEUS_H1 = "{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd"
DUMMY = "generation/dummy/build/keti_dummy_ts15066_v1.usd"
TERRAIN = "isaaclab://TerrainImporter/plane"
SKIN = "generation/dummy/build/keti_dummy_ts15066_v1_skin.usd"
AISLE = "generation/scene/build/keti_logistics_aisle_v1.usd"
# What a v0.1 (S1a) batch's episodes reference, and what an S1d batch's do (their sidecars'
# asset_paths).
V01_ASSETS = (NUCLEUS_H1, DUMMY, TERRAIN)
S1D_ASSETS = (NUCLEUS_H1, SKIN, TERRAIN, AISLE)

REPO_ROOT = Path(__file__).resolve().parents[2]
# The camera canon's cameras, in its order: what a camera recording names per camera.
CANON_CAMERAS = camera_canon.camera_names(camera_canon.load(REPO_ROOT / "meta" / "cameras.json"))


def _meta():
    return {
        "episode_id": "S1a_0000_seed1", "scenario_id": "S1a", "episode_seq": 0, "run_start_seq": 0,
        "seed": 1, "steps": 3, "degenerate": False,
        "contact_region_names": ["BR-01", "BR-02"],
        "safety_event_names": ["ssm_violation", "protective_stop_triggered", "contact_onset"],
        "stance_foot_names": ["left_ankle_link", "right_ankle_link"],
        "randomization": {}, "hold_release_step": -1, "hold_contaminated_steps": 0,
        "provisional": True, "ssm_params": {}, "global_time_offset_s": 0.02,
        "robot_model": "unitree_h1",
        "asset_paths": {"robot": NUCLEUS_H1, "dummy": DUMMY},
        "asset_paths_resolved": {"robot": "https://example/h1.usd", "dummy": DUMMY},
        "imu_link": "pelvis",
        "action_space": "position_rad",
        "termination": {"terminated": False, "truncated": True, "reason": ["time_out"]},
        "degenerate": False,
        "sim_backend": {"isaacsim": "6.0.1", "isaaclab": "v3.0.0", "physics": "physx",
                        "lerobot": "0.6.1",
                        "generator_commit": "0123456789abcdef0123456789abcdef01234567",
                        "generator_dirty": False},
    }


def _arrays(n=3, dof=2, n_region=2):
    return {
        "timestamp": np.arange(n) * 0.02,
        "joint_pos": np.zeros((n, dof)), "joint_vel": np.zeros((n, dof)),
        "joint_torque": np.zeros((n, dof)), "action": np.zeros((n, dof)),
        "root_pose": np.tile([0, 0, 1, 1, 0, 0, 0], (n, 1)).astype(float),
        "root_vel": np.zeros((n, 6)), "imu": np.zeros((n, 10)),
        "contact_force": np.zeros((n, n_region)),
        "contact_v_rel": np.zeros((n, n_region)),
        "contact_type": np.full((n, n_region), "none"),
        "com": np.zeros((n, 3)), "zmp": np.zeros((n, 2)),
        "stability_margin": np.zeros(n), "min_distance_human": np.ones(n),
        "separation_required": np.ones(n) * 0.5,
        "separation_violation": np.zeros(n, dtype=bool),
        "safety_event": np.zeros((n, 3), dtype=bool),
    }


def _canon():
    scenarios = {"scenarios": [{"scenario_id": "S1a", "domain": "baseline",
                                "task_archetype": "locomotion", "hazard_class": "collision",
                                "safety_function": "ssm",
                                "rules": ["R-10218-STOP-01"],
                                "scene": {"human_model": "keti_dummy_ts15066_v1"}}]}
    joint_map = {"robot_model": "unitree_h1", "dof": 2, "joint_names": ["a", "b"]}
    assets = {
        "robots": {"unitree_h1": {"usd_path": NUCLEUS_H1, "license": "BSD-3-Clause",
                                  "redistributed": False, "embodiment": "biped"}},
        "scene": {"keti_dummy_ts15066_v1": {"usd_path": DUMMY, "license": "CC-BY-4.0",
                                            "redistributed": True}},
    }
    rules = {"rules": [{"rule_id": "R-10218-STOP-01", "standard": "KS B ISO 10218-1"}]}
    return scenarios, joint_map, assets, rules


def _label():
    return {"outcome": "nominal", "triggered_rules": [], "severity": "S0",
            "not_evaluated": [], "contact_observed": False}


def _build(meta=None, dt=0.02):
    scenarios, joint_map, assets, rules = _canon()
    return build_episode_meta(meta or _meta(), _label(), scenarios, joint_map,
                              assets, rules, step_dt=dt)


# 스키마 §5: "triggered_rules — 빈 배열이면 nominal". 비어 있는 것이 정상값이다.
MAY_BE_EMPTY = {"triggered_rules"}


def test_every_required_episode_field_is_present_and_non_empty():
    # The unconditional fields, of an S1a record; the table fields have their own test, of an S1e one.
    out = _build()
    for f in episode_fields(False):
        assert f.schema in out, f"필수 에피소드 필드 없음: {f.schema}"
        if f.schema in MAY_BE_EMPTY:
            continue
        assert out[f.schema] not in (None, "", []), f"{f.schema} 가 비었음"


def test_a_triggered_label_carries_its_rules_through():
    scenarios, joint_map, assets, rules = _canon()
    label = dict(_label(), outcome="fail", severity="S2",
                 triggered_rules=["R-10218-STOP-01"])
    out = build_episode_meta(_meta(), label, scenarios, joint_map, assets, rules,
                             step_dt=0.02)
    assert out["triggered_rules"] == ["R-10218-STOP-01"]
    assert out["outcome"] == "fail" and out["severity"] == "S2"


def test_the_runs_first_seq_reaches_the_episode_record():
    out = _build(dict(_meta(), episode_id="S1a_0013_seed1", episode_seq=13, run_start_seq=10))
    assert out["run_start_seq"] == 10


@pytest.mark.parametrize("change, phrase", [
    (lambda m: m.pop("run_start_seq"), "run_start_seq 가 없습니다"),
    (lambda m: m.update(run_start_seq=True), "run_start_seq 의 형이 int 이 아닙니다"),
    (lambda m: m.update(run_start_seq="0"), "run_start_seq 의 형이 int 이 아닙니다"),
    (lambda m: m.update(run_start_seq=-1), "0 과 자기 번호 0 사이가 아닙니다"),
    (lambda m: m.update(run_start_seq=1), "0 과 자기 번호 0 사이가 아닙니다"),
])
def test_a_run_start_the_episode_could_not_have_is_refused(change, phrase):
    meta = _meta()
    change(meta)
    with pytest.raises(ValueError, match=phrase):
        cl.require_run_start(meta)


@pytest.mark.parametrize("change, phrase", [
    (lambda b: b.update(generator_dirty=True), "generator_dirty 가 false 가 아닙니다: True"),
    (lambda b: b.update(generator_dirty="false"), "generator_dirty 가 false 가 아닙니다: 'false'"),
    (lambda b: b.pop("generator_dirty"), "generator_dirty 가 false 가 아닙니다: None"),
    (lambda b: b.update(generator_commit="9ab48ae"), "40자 16진 커밋 해시가 아닙니다"),
    (lambda b: b.update(generator_commit=None), "40자 16진 커밋 해시가 아닙니다"),
    (lambda b: b.pop("generator_commit"), "40자 16진 커밋 해시가 아닙니다"),
])
def test_a_recording_that_does_not_name_committed_code_is_refused(change, phrase):
    meta = _meta()
    change(meta["sim_backend"])
    with pytest.raises(ValueError, match=phrase) as e:
        cl.check_generator_provenance(meta)
    assert "S1a_0000_seed1" in str(e.value)                 # which episode
    assert "커밋된 코드로 만든 녹화만" in str(e.value)        # and what to do about it


def test_fps_comes_from_the_step_period_not_a_constant():
    assert _build(dt=0.02)["fps"] == 50
    assert _build(dt=0.01)["fps"] == 100


def test_duration_is_steps_times_the_step_period():
    assert _build(dt=0.02)["duration_s"] == pytest.approx(3 * 0.02)


def test_scene_assets_carry_path_license_and_redistribution():
    out = _build()
    assert out["scene_assets"]
    for a in out["scene_assets"]:
        assert {"path", "license", "redistributed"} <= set(a)


def test_an_asset_missing_from_the_canon_stops_the_conversion():
    # 라이선스를 모르는 자산이 배포본에 들어가는 것보다 멈추는 편이 낫다.
    meta = _meta()
    meta["asset_paths"] = dict(meta["asset_paths"], robot="/somewhere/unknown.usd")
    with pytest.raises(ValueError, match="unknown.usd"):
        _build(meta)


def test_the_published_path_is_the_canonical_one_not_the_resolved_url():
    # asset_paths_resolved 는 출처 보존용이지 배포 메타의 값이 아니다.
    paths = {a["path"] for a in _build()["scene_assets"]}
    assert paths == {NUCLEUS_H1, DUMMY}


def test_notes_carry_the_unevaluated_rules_and_the_contact_flag():
    import json
    scenarios, joint_map, assets, rules = _canon()
    label = dict(_label(), contact_observed=True,
                 not_evaluated=[{"rule_id": "R-TS15066-A2-01", "reason": "한계값 없음"}])
    out = build_episode_meta(_meta(), label, scenarios, joint_map, assets, rules,
                             step_dt=0.02)
    notes = json.loads(out["notes"])
    assert notes["contact_observed"] is True
    assert notes["not_evaluated"][0]["rule_id"] == "R-TS15066-A2-01"


def test_a_dof_mismatch_between_joint_map_and_data_stops_the_conversion():
    _, joint_map, _, _ = _canon()
    joint_map = dict(joint_map, dof=19, joint_names=[f"j{i}" for i in range(19)])
    with pytest.raises(ValueError, match="dof"):
        step_frames(_arrays(), _meta(), joint_map)


def test_every_required_step_field_is_present_in_each_frame():
    # Without media in the sidecar (every v0.1 episode) a frame holds exactly the
    # unconditional fields -- the keys it held before camera fields existed -- and those
    # are exactly the schema's unconditionally required ones and its optional ones but the
    # measured contact area, so neither side can shrink alone.
    import validator.validate_dataset as v
    _, joint_map, _, _ = _canon()
    frames = step_frames(_arrays(), _meta(), joint_map)
    assert len(frames) == 3
    expected = {f.schema for f in STEP_FIELDS if f.condition is None}
    assert expected == (set(v.REQUIRED_STEP_FIELDS)
                        | (set(v.OPTIONAL_STEP_FIELDS) - {"observation.contact.pressure"}))
    for i, frame in enumerate(frames):
        assert set(frame) == expected, f"프레임 {i}: {sorted(set(frame) ^ expected)}"


def _camera_poses():
    """The map's camera pose fields, in its order: one per camera of the canon."""
    return [f for f in STEP_FIELDS if f.kind == "array" and f.schema.startswith("observation.camera_pose.")]


def _with_cameras(meta, arrays, canon=None):
    """A camera episode, as the recorder saves one: the sidecar names a stream per camera of the
    map (the canon's cameras) and the pack, with the recorder's two counts (no frame rendered again,
    no black frame) and the fingerprint of the camera canon it was recorded with (``canon``, the
    repository's by default), and the npz carries the box and a pose per camera in
    media_writer.split_media's types -- the j-th pose filled with j, so a mix-up shows."""
    n = len(arrays["timestamp"])
    media = {cam: {"file": f"media/S1a_0000_seed1.{cam}.mkv", "frames": n}
             for cam in cl.video_cameras().values()}
    media.update(pack={"file": "media/S1a_0000_seed1.head.pack.npz", "frames": n}, rerendered_steps=[],
                 blank_frames={cam: 0 for cam in cl.video_cameras().values()},
                 canon_fingerprint=camera_canon.recording_fingerprint(canon or _camera_canon()))
    meta = dict(meta, media=media)
    arrays = dict(arrays, bbox_human=np.tile(np.array([1, 2, 3, 4], np.int32), (n, 1)),
                  **{f.npz: np.full((n, 7), j, np.float32) for j, f in enumerate(_camera_poses())})
    return meta, arrays


def test_media_in_the_sidecar_adds_the_camera_arrays_and_nothing_else():
    # Videos go to LeRobot separately and the pack is copied; a frame holds
    # neither.
    from generation.field_map import CAMERAS
    _, joint_map, _, _ = _canon()
    meta, arrays = _with_cameras(_meta(), _arrays())
    frame = step_frames(arrays, meta, joint_map)[0]
    expected = {f.schema for f in STEP_FIELDS
                if f.condition is None or (f.condition == CAMERAS and f.kind == "array")}
    assert set(frame) == expected
    assert frame["observation.bbox2d.human"] == [1, 2, 3, 4]
    for j, f in enumerate(_camera_poses()):
        assert frame[f.schema] == [float(j)] * 7, f.schema


def test_media_without_a_camera_array_stops_rather_than_writing_a_hole():
    _, joint_map, _, _ = _canon()
    meta, arrays = _with_cameras(_meta(), _arrays())
    del arrays["bbox_human"]
    with pytest.raises(KeyError, match="bbox_human"):
        step_frames(arrays, meta, joint_map)


def test_camera_arrays_without_media_stop_the_conversion():
    # The recorder writes these arrays only together with media; an episode that has one
    # half did not come out of one save, and the converter does not guess which to trust.
    _, joint_map, _, _ = _canon()
    _, arrays = _with_cameras(_meta(), _arrays())
    with pytest.raises(ValueError, match="media"):
        step_frames(arrays, _meta(), joint_map)


@pytest.mark.parametrize("media", [None, {}, ["media/x.mkv"]], ids=["null", "empty", "list"])
def test_a_media_entry_that_names_nothing_stops_the_conversion(media):
    _, joint_map, _, _ = _canon()
    meta, arrays = _with_cameras(_meta(), _arrays())
    with pytest.raises(ValueError, match="media"):
        step_frames(arrays, dict(meta, media=media), joint_map)


def test_the_camera_arrays_keep_their_recorded_dtypes_into_the_frame():
    # Widening the int32 box to float64 would read as a reproduction failure in C9.
    from generation.convert_lerobot import recorded_dtypes, to_lerobot_frame
    _, joint_map, _, _ = _canon()
    meta, arrays = _with_cameras(_meta(), _arrays())
    dtypes = recorded_dtypes(arrays, meta)
    out = to_lerobot_frame(step_frames(arrays, meta, joint_map)[0], dtypes)
    assert out["observation.bbox2d.human"].dtype == np.int32
    assert out["observation.camera_pose.head_rgb"].dtype == np.float32


def test_recorded_dtypes_of_a_camera_less_episode_are_the_unconditional_arrays():
    from generation.convert_lerobot import recorded_dtypes
    assert set(recorded_dtypes(_arrays(), _meta())) == {
        f.schema for f in STEP_FIELDS if f.kind == "array" and f.condition is None}


def test_frame_index_counts_from_zero():
    _, joint_map, _, _ = _canon()
    frames = step_frames(_arrays(), _meta(), joint_map)
    assert [f["frame_index"] for f in frames] == [0, 1, 2]


def test_safety_event_becomes_names_not_booleans():
    _, joint_map, _, _ = _canon()
    arrays = _arrays()
    arrays["safety_event"][1, 2] = True
    frames = step_frames(arrays, _meta(), joint_map)
    assert frames[0]["safety.event"] == []
    assert frames[1]["safety.event"] == ["contact_onset"]


def test_region_names_are_broadcast_to_every_frame():
    _, joint_map, _, _ = _canon()
    for f in step_frames(_arrays(), _meta(), joint_map):
        assert f["observation.contact.region_names"] == ["BR-01", "BR-02"]


def test_scalar_fields_are_scalars_and_vector_fields_are_lists():
    _, joint_map, _, _ = _canon()
    frame = step_frames(_arrays(), _meta(), joint_map)[0]
    assert isinstance(frame["safety.stability_margin"], float)
    assert isinstance(frame["safety.separation_violation"], bool)
    assert frame["observation.root_pose"] == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert len(frame["safety.zmp"]) == 2


def test_a_missing_npz_array_stops_rather_than_writing_a_hole():
    _, joint_map, _, _ = _canon()
    arrays = _arrays()
    del arrays["zmp"]
    with pytest.raises(KeyError, match="zmp"):
        step_frames(arrays, _meta(), joint_map)


def test_an_array_longer_than_the_timeline_stops_rather_than_dropping_rows():
    # 예전에는 timestamp 길이만큼만 돌며 여분 행을 조용히 버렸다.
    _, joint_map, _, _ = _canon()
    arrays = _arrays()
    arrays["zmp"] = np.zeros((4, 2))
    with pytest.raises(ValueError, match="스텝 수"):
        step_frames(arrays, _meta(), joint_map)


def test_an_array_shorter_than_the_timeline_stops_before_indexerror():
    _, joint_map, _, _ = _canon()
    arrays = _arrays()
    arrays["com"] = np.zeros((2, 3))
    with pytest.raises(ValueError, match="스텝 수"):
        step_frames(arrays, _meta(), joint_map)


def test_a_steps_count_that_disagrees_with_the_arrays_stops():
    _, joint_map, _, _ = _canon()
    with pytest.raises(ValueError, match="스텝 수"):
        step_frames(_arrays(), dict(_meta(), steps=99), joint_map)


def test_a_non_boolean_event_matrix_stops_the_conversion():
    _, joint_map, _, _ = _canon()
    arrays = _arrays()
    arrays["safety_event"] = np.zeros((3, 3))
    with pytest.raises(ValueError, match="safety_event"):
        step_frames(arrays, _meta(), joint_map)


def test_a_scenario_without_safety_function_says_what_is_missing():
    # 나머지 11개 시나리오가 처음 변환될 때 만나는 경로. KeyError 가 아니라
    # 무엇을 어디에 채워야 하는지가 나와야 한다.
    scenarios, joint_map, assets, rules = _canon()
    del scenarios["scenarios"][0]["safety_function"]
    with pytest.raises(ValueError, match="safety_function"):
        build_episode_meta(_meta(), _label(), scenarios, joint_map, assets, rules,
                           step_dt=0.02)


def test_an_unknown_scenario_id_says_so():
    scenarios, joint_map, assets, rules = _canon()
    with pytest.raises(ValueError, match="S9z"):
        build_episode_meta(dict(_meta(), scenario_id="S9z"), _label(), scenarios,
                           joint_map, assets, rules, step_dt=0.02)


# ── 배포본 조립 ──────────────────────────────────────────────────

def _repo_root_with_redistributables(tmp_path):
    """변환기가 읽는 것만 담고, 선언된 재배포 자산이 반드시 존재하는 저장소 나무.

    더미 USD 는 Isaac 으로 빌드하는 산출물이고 `.gitignore` 의 `*.usd` 에 걸려
    저장소에 들어가지 않습니다. CI 에는 Isaac 이 없어 그 파일이 없고, 변환기는
    「redistributed=true 인데 파일이 없습니다」로 거부합니다 — 정본이 옮겨지는지만
    보는 아래 시험 넷이 자산과 무관한데도 CI 에서만 죽던 이유입니다.

    **저장소를 통째로 비추지 않습니다.** 최상위 전부를 심링크로 걸면
    `.git`·`.venv`·`.tmp` 까지 원본을 가리켜 **쓰기 경로에서 재사용되는 순간
    원본이 바뀝니다** — 그 심링크를 따라가 원본 자산을 0바이트로 자를 수
    있습니다. 그래서 정본을 실물로 복사합니다 — 나무 안에서
    무엇을 해도 저장소에 닿지 않고, 심링크가 없으니 따라갈 링크도 없습니다.

    복사 목록은 **변환기의 상수에서 끌어옵니다.** 손으로 관리하지 않으므로 정본이
    늘어도 따라오고, 변환기가 새로 읽는 것이 생기면 없는 파일로 **빨간불**이 납니다 —
    조용히 덜 시험하는 쪽으로 어긋나지 않습니다.

    로컬에 실물 자산이 있든 없든 **같은 경로를 돕니다.** 있을 때만 실물을 쓰면 이
    도우미가 CI 에서만 돌아, 여기서 틀려도 로컬에서는 드러나지 않습니다. 자산의
    내용은 이 시험들의 대상이 아니며, 존재를 요구하는 가드 자체는
    `test_the_converter_refuses_an_asset_the_canon_declares_but_the_tree_lacks`
    가 따로 시험합니다.

    **스텁은 정본이 고정한 해시와 맞아야 합니다.** 변환기는 싣는 파일마다
    정본의 SHA-256(USD 는 `usd_sha256`, 텍스처는 `bundle.files`)과 대조하고, 텍스처
    행은 lock 과도 대조합니다. 그래서 스텁마다 내용을 정하고 **이 나무의** 정본과 lock
    의 해시를 그 내용으로 바꿔 씁니다 — 저장소의 것은 건드리지 않습니다.

    **lock 하나를 두 자산이 함께 쓸 수 있습니다**(입힌 인형 v1 과 조끼 입은 관절 인형이
    같은 피부 lock 의 같은 행을, 창고와 들고 다니는 상자가 같은 KETI CI 고정 파일의 행을
    싣습니다). 그래서 lock 은 한 번만 옮겨 오고, 텍스처 스텁은 파일 자리마다 하나라 같은
    행을 가진 자산들이 같은 스텁 해시를 받습니다. **묶음 하나가 lock 여럿을 적을 수
    있으므로**(창고 — CC0 텍스처와 KETI CI) 행의 해시는 그 행을 가진 lock 에서 바꿉니다.

    **폴더째 싣는 로봇(`ship`)은 행마다 스텁을 받아 둔 자리(`source_dir`)에 놓고**, 행의
    해시와 로봇의 `usd_sha256`(USD 의 행과 같아야 합니다)을 그 스텁의 것으로 바꿔 씁니다 —
    로봇의 `usd_path` 는 Nucleus 토큰이라 그 자리에 놓을 파일이 아닙니다.
    """
    import json
    import shutil

    from generation.assets import bundle_per_lock
    from generation.convert_lerobot import (CANON_DIRS, CANON_FILES, CANON_RENAMES,
                                            NARROWED_CANON)

    root = tmp_path / "repo"
    wanted = [
        *NARROWED_CANON,          # 좁혀 쓰기 전에 원본을 읽는다
        *CANON_FILES,
        *CANON_RENAMES,           # 키가 저장소 쪽 경로다
        *CANON_DIRS,
        "meta/joint_map",         # 이 배치가 쓴 로봇의 것만 골라 싣는다
        # 배포되지 않지만 카드가 이것과 다른지 보는 시험이 읽는다.
        "README.md",
    ]
    for rel in wanted:
        src = REPO_ROOT / rel
        assert src.exists(), f"정본이 저장소에 없습니다: {rel}"
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    assets = json.loads((root / "meta" / "assets.json").read_text(encoding="utf-8"))
    declared = {name: e for group in ("robots", "scene")
                for name, e in (assets.get(group) or {}).items() if e.get("redistributed")}
    assert declared, "정본이 재배포 자산을 하나도 선언하지 않습니다 — 이 도우미의 전제가 깨졌습니다"
    stubbed: dict[Path, tuple[str, str]] = {}          # texture path -> (canon sha256, stub sha256)
    copied: set[str] = set()
    for name, entry in declared.items():
        if "ship" in entry:
            usd_row = Path(entry["usd_path"]).name
            for row in entry["ship"]["files"]:
                row["sha256"] = _stub_at(root, Path(entry["ship"]["source_dir"]) / row["file"],
                                         f"robot stub {name} {row['file']}\n".encode())
                if row["file"] == usd_row:
                    entry["usd_sha256"] = row["sha256"]
            continue
        entry["usd_sha256"] = _stub_at(root, Path(entry["usd_path"]),
                                       f"#usda 1.0\n# stub {name}\n".encode())
        bundle = entry.get("bundle")
        if not bundle:
            continue
        # The locks and the spec the bundle's readers parse (generation.assets.LOCK_READERS),
        # with each texture row's hash swapped for its stub's in the one lock that holds it -- rows
        # and locks stay one statement. A lock two bundles share is copied once: copying it again
        # would undo the first one's swaps.
        locks = bundle_per_lock(bundle, "textures_lock")
        for rel in (*locks, entry.get("spec")):
            if rel and rel not in copied:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO_ROOT / rel, root / rel)
                copied.add(rel)
        texts = {rel: (root / rel).read_text(encoding="utf-8") for rel in locks}
        for row in bundle["files"]:
            where = Path(bundle["textures_dir"]) / row["file"]
            if where not in stubbed:
                digest = _stub_at(root, where, f"texture stub {where}\n".encode())
                holders = [rel for rel, text in texts.items() if row["sha256"] in text]
                assert len(holders) == 1 and texts[holders[0]].count(row["sha256"]) == 1, (name, row["file"])
                texts[holders[0]] = texts[holders[0]].replace(row["sha256"], digest)
                stubbed[where] = (row["sha256"], digest)
            canon_sha, digest = stubbed[where]
            assert row["sha256"] in (canon_sha, digest), (name, row["file"], "two hashes for one file")
            row["sha256"] = digest
        for rel, text in texts.items():
            (root / rel).write_text(text, encoding="utf-8")
    (root / "meta" / "assets.json").write_text(
        json.dumps(assets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root


def _stub_at(root, rel, data=b""):
    """나무 **안에** 스텁을 놓고 그 SHA-256 을 돌려준다. 나무 밖은 만들지도 지우지도 않는다.

    앞 판은 만들고 지운 **뒤에** 나무 안인지 단언했습니다. `..` 조각이나 절대 경로가
    오면 바깥 파일을 0바이트로 자른 다음 단언이 나므로, 검사가 손해를 막지 못합니다
    (9바이트 sentinel 이 0바이트가 된 뒤 AssertionError).
    그래서 **아무것도 건드리기 전에** 거부합니다. `assert` 가 아니라 예외인 것은
    `python -O` 가 assert 를 지우기 때문이며, 안전 조건을 지워지는 문장에 두지
    않습니다.
    """
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"나무 밖을 가리키는 자산 경로입니다: {rel}")
    dst = root / rel
    if not dst.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"자산 경로가 나무 밖으로 나갑니다: {dst.parent.resolve()}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _lerobot_output_stub(tmp_path):
    """assemble_dataset 은 lerobot 이 이미 써 둔 meta/info.json 위에서 돈다."""
    import json

    out = tmp_path / "ds"
    (out / "meta").mkdir(parents=True)
    (out / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "fps": 50}), encoding="utf-8")
    return out


def test_the_conversion_takes_the_recordings_in_the_order_of_their_file_names(tmp_path, monkeypatch):
    # A chunk's rows can be asked in any order; the converter writes its episodes by the names of their
    # recordings, and whoever holds a converted distribution to the episodes it was asked for compares in that
    # order (conversion_order -- the assembly does).
    from generation import assemble_batch, convert_lerobot

    asked = ["S1e_0021_seed5", "S1e_0020_seed9", "S1e_0100_seed1", "S1e_0003_seed2"]
    for eid in asked:
        (tmp_path / f"{eid}.npz").write_bytes(b"")
    by_name = sorted(p.name for p in tmp_path.glob("*.npz"))
    assert [p.name for p in convert_lerobot.recordings(tmp_path)] == by_name
    assert [f"{eid}.npz" for eid in convert_lerobot.conversion_order(asked)] == by_name
    assert assemble_batch.conversion_order is convert_lerobot.conversion_order
    seen = []
    monkeypatch.setattr(convert_lerobot, "recordings", lambda episodes: seen.append(Path(episodes)) or [])
    assert convert_lerobot.convert(tmp_path, tmp_path / "out", "test/x", repo_root=REPO_ROOT,
                                   create_dataset=None) == 1
    assert seen == [tmp_path], "변환기가 에피소드를 recordings 로 고르지 않습니다"


def test_the_dataset_card_lands_as_the_readme_not_the_repository_one(tmp_path):
    """HF 에서 데이터셋 루트의 README.md 는 데이터셋 카드 그 자체다.

    저장소 README 를 그대로 올리면 front matter 가 없어 Hub 이 라이선스·태그·뷰어를
    구성하지 못하고, 내용도 데이터가 아니라 저장소를 설명한다. 실제로 그렇게 올라가고
    있었으므로 회귀로 못박는다.
    """
    import json
    from generation.convert_lerobot import CANON_FILES, CANON_RENAMES, assemble_dataset

    assert "README.md" not in CANON_FILES
    assert CANON_RENAMES["docs/dataset-card.md"] == "README.md"

    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    out = _lerobot_output_stub(tmp_path)
    copied = assemble_dataset(repo_root, out, assets, ["S1a"], referenced=V01_ASSETS)

    assert "README.md" in copied
    card = (out / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\n"), "데이터셋 카드에 YAML front matter 가 없습니다"
    assert "license: cc-by-4.0" in card.split("---\n", 2)[1]
    assert card == (repo_root / "docs" / "dataset-card.md").read_text(encoding="utf-8")
    assert card != (repo_root / "README.md").read_text(encoding="utf-8")


def test_the_card_points_the_hub_at_the_step_tables():
    """Which files the Hub's dataset viewer and ``datasets.load_dataset(<repo id>)`` read is the card's
    ``configs`` block. Without it the Hub guesses from the files present, and a release holds hundreds of
    camera videos against a dozen step tables, so the guess is a folder of videos -- 82 GB of them for a
    first ``load_dataset``. LeRobot writes this block into the cards it makes; this card is written by
    hand, so the block is held here: one config, the step tables, named by the path a release writes them
    under, and matching nothing else a release ships. The Hub matches the pattern one path part at a time
    (``*`` does not cross ``/``), and so does ``selected`` below.
    """
    import fnmatch
    import re

    import yaml
    from validator import make_fixture, validate_dataset

    # The front matter ends at the first line that is only "---" (trailing blanks allowed), as the Hub
    # reads it -- not at the first "---" anywhere, which a YAML comment inside the block can hold.
    lines = (REPO_ROOT / "docs" / "dataset-card.md").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "---"
    end = next(i for i, line in enumerate(lines[1:], 1) if line.rstrip(" \t") == "---")
    front = yaml.safe_load("\n".join(lines[1:end]))
    steps = re.sub(r"[^/{]*\{[^}]*\}", "*", validate_dataset.DATA_PATH)
    assert steps == "data/*/*.parquet"
    assert front.get("configs") == [{"config_name": "default", "data_files": steps}], (
        "카드 머리말의 configs 가 스텝 표를 가리키지 않습니다 — 없으면 Hub 이 배포본을 영상 폴더로 읽습니다")

    def selected(path: str) -> bool:
        want, got = steps.split("/"), path.split("/")
        return len(want) == len(got) and all(fnmatch.fnmatchcase(g, w) for g, w in zip(got, want))

    tables = [validate_dataset.DATA_PATH.format(chunk_index=c, file_index=f)
              for c, f in ((0, 0), (0, 12), (3, 999))]
    others = [
        make_fixture.VIDEO_PATH.format(video_key="observation.images.head_rgb", chunk_index=0, file_index=0),
        make_fixture._eps_path(Path(".")).as_posix(),
        validate_dataset.TASKS_PATH,
        "meta/info.json", "meta/stats.json", "meta/joint_map/unitree_h1.yaml",
        "assets/unitree_h1/h1.usd", "assets/textures/shoes01_diffuse.png",
        "README.md", "NOTICE", "LICENSE",
    ]
    assert [p for p in tables if not selected(p)] == []
    assert [p for p in others if selected(p)] == []


def test_the_assembly_finds_a_joint_map_where_the_converter_reads_and_lays_it(tmp_path):
    """assemble_batch.check_canon holds the joint map a chunk shipped to the working tree's, byte for
    byte, and finds both by a path rule of its own. This holds that rule to what the converter does:
    a rule pointing elsewhere would compare nothing, or two files that do not ship."""
    import json
    from generation import assemble_batch
    from generation.convert_lerobot import assemble_dataset

    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    out = _lerobot_output_stub(tmp_path)
    copied = assemble_dataset(repo_root, out, assets, ["S1a"], robot_models=["unitree_h1"],
                              referenced=V01_ASSETS)

    laid = [rel for rel in copied if rel.startswith("meta/joint_map/")]
    assert laid == ["meta/joint_map/unitree_h1.yaml"]
    assert assemble_batch.joint_map_of(out, "unitree_h1") == out / laid[0]
    assert sorted(p.name for p in (out / "meta" / "joint_map").iterdir()) == ["unitree_h1.yaml"]
    read_from = assemble_batch.joint_map_of(repo_root, "unitree_h1")
    assert read_from == repo_root / "meta" / "joint_map" / "unitree_h1.yaml"
    assert (out / laid[0]).read_bytes() == read_from.read_bytes()
    # The file the converter read is the one named: with it gone there is nothing to lay.
    read_from.unlink()
    with pytest.raises(FileNotFoundError, match="관절 정본"):
        assemble_dataset(repo_root, _lerobot_output_stub(tmp_path / "again"), assets, ["S1a"],
                         robot_models=["unitree_h1"], referenced=V01_ASSETS)


def test_the_assembly_reads_a_chunks_asset_files_where_the_converter_lays_them(tmp_path):
    """assemble_batch.check_canon holds the asset files a chunk sealed to the hashes the canon pins for
    the files the converter would lay for that chunk's episodes. Which entries of a seal are asset
    files, and which assets a table of episodes references, are readings of its own. This holds them
    to what the converter does: a reading that looked elsewhere would hold nothing to anything."""
    import hashlib
    import json
    import shutil

    from generation import assemble_batch
    from generation.batch_layout import DATASET_DIR, EPISODES_DIR, sealed_files
    from generation.convert_lerobot import assemble_dataset

    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    chunk = tmp_path / "chunk"
    (chunk / EPISODES_DIR).mkdir(parents=True)
    out = chunk / DATASET_DIR
    shutil.move(_lerobot_output_stub(tmp_path), out)
    copied = assemble_dataset(repo_root, out, assets, ["S1d"], referenced=S1D_ASSETS)

    # What the converter laid for an S1d batch: the two USDs that ship, the textures of both and the
    # robot's folder -- files beneath a folder of their own, as the textures are.
    used = [e for e in assets["scene"].values() if e["usd_path"] in S1D_ASSETS and e["redistributed"]]
    textures = {row["file"] for e in used for row in e["bundle"]["files"]}
    robot = assets["robots"]["unitree_h1"]["ship"]
    laid = {rel: hashlib.sha256((out / rel).read_bytes()).hexdigest() for rel in copied if rel.startswith("assets/")}
    assert len(used) == 2 and len(laid) == len(used) + len(textures) + len(robot["files"]) > len(used)
    assert {rel for rel in laid if rel.startswith(f"assets/{robot['dir']}/")} == {
        f"assets/{robot['dir']}/{row['file']}" for row in robot["files"]}
    assert sorted(laid) == sorted(p.relative_to(out).as_posix() for p in (out / "assets").rglob("*") if p.is_file())
    # The seal of a chunk holding that distribution: the reading finds those files, and no other.
    assert len(sealed_files(chunk)) > len(laid)
    assert assemble_batch.sealed_assets({"files": sealed_files(chunk)}) == laid

    # The episode table as the converter leaves it: the records it builds for two episodes, joined into
    # LeRobot's table by its own writer, and read back the way the assembly reads a chunk's.
    rows = _meta_root(tmp_path / "rows", [[{"episode_index": 0}, {"episode_index": 1}]])
    cl.merge_episode_metadata(rows, [_build(), _build(dict(_meta(), episode_id="S1a_0001_seed2"))])
    table = assemble_batch.episodes_table(rows)
    assert len(table) == 2
    assert assemble_batch.referenced_assets(table) == sorted(_meta()["asset_paths"].values())


def test_the_assembly_lays_the_asset_files_its_merged_episodes_reference(tmp_path):
    """assemble_batch.lay_canon hands the converter the assets the merged episode table references,
    read as check_canon reads a chunk's. The converter lays what it is handed and nothing else: with
    another list it lays other asset files, or none, and the assembly's own checks read no asset."""
    import json
    import shutil

    from generation import assemble_batch
    from generation.assets import asset_entry, shipped_asset_files

    repo_root = _repo_root_with_redistributables(tmp_path)
    shutil.copy2(REPO_ROOT / "meta" / "cameras.json", repo_root / "meta" / "cameras.json")
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    out = _lerobot_output_stub(tmp_path)
    # Two episodes that reference different assets, in the table the converter's own writer leaves
    # (the entries are the ones it builds for a loaded asset): what is laid is what either references.
    _meta_root(out, [[{"episode_index": 0}, {"episode_index": 1}]])
    cl.merge_episode_metadata(out, [
        {"episode_id": f"S1d_{i:04d}_seed{i}", "scenario_id": "S1d", "robot_model": "unitree_h1",
         "triggered_rules": [], "scene_assets": [asset_entry(assets, path) for path in part]}
        for i, part in enumerate((S1D_ASSETS[:2], S1D_ASSETS[2:]))])

    copied = assemble_batch.lay_canon(repo_root, out, [])
    laid = sorted(rel for rel in copied if rel.startswith("assets/"))
    assert laid == sorted(shipped_asset_files(assets, S1D_ASSETS, repo_root)) and len(laid) > 2
    assert all(shipped_asset_files(assets, part, repo_root) for part in (S1D_ASSETS[:2], S1D_ASSETS[2:])), \
        "두 에피소드가 저마다 싣는 자산을 참조해야 이 시험이 합집합을 봅니다"
    assert laid == sorted(p.relative_to(out).as_posix() for p in (out / "assets").rglob("*") if p.is_file())


def test_the_machine_readable_schema_travels_with_the_data(tmp_path):
    import json
    from generation.convert_lerobot import assemble_dataset

    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    out = _lerobot_output_stub(tmp_path)
    assemble_dataset(repo_root, out, assets, ["S1a"], referenced=V01_ASSETS)

    canon = json.loads((out / "meta" / "safety_schema.json").read_text(encoding="utf-8"))
    assert set(canon["body_regions"]) == {f"BR-{i:02d}" for i in range(1, 13)}
    # 임계값은 이 파일이 아니라 rules.json 이 나른다
    assert "contact_limits" not in canon


# ── JSON 문자열로 실리는 필드 ────────────────────────────────────

def _canon_schema():
    import json
    return json.loads((REPO_ROOT / "meta" / "safety_schema.json").read_text(encoding="utf-8"))


def test_the_canon_lists_exactly_the_step_fields_the_converter_stringifies():
    from generation.convert_lerobot import JSON_STRING_FIELDS

    declared = _canon_schema()["json_encoded_fields"]["step"]
    assert list(declared) == list(JSON_STRING_FIELDS)


def test_the_canon_lists_exactly_the_episode_fields_that_land_as_json():
    """episode_column 이 무엇을 문자열로 바꾸는지는 값의 모양이 정한다.

    목록을 손으로 적으면 §5 에 중첩 필드가 하나 늘 때 조용히 틀린다. 실제로
    한 에피소드를 만들어 어느 열이 JSON 이 되는지 세서 정본과 맞춘다.
    """
    import json

    from generation.convert_lerobot import build_episode_meta, episode_column

    from tests.generation.release_fixture import table_recording

    def stringified(record):
        out = []
        for key, value in record.items():
            col = episode_column(value)
            if isinstance(col, str) and not isinstance(value, str):
                out.append(key)
            elif isinstance(value, str) and col is value:
                # 원래 문자열인 필드는 JSON 인코딩이 아니다. notes 만 예외로,
                # build_episode_meta 가 이미 json.dumps 해서 넣는다.
                try:
                    json.loads(value)
                except (ValueError, TypeError):
                    continue
                if key == "notes":
                    out.append(key)
        return sorted(out)

    scenarios, joint_map, assets, rules = _canon()
    record = build_episode_meta(_meta(), _label(), scenarios, joint_map, assets,
                               rules, step_dt=0.02)
    declared = _canon_schema()["json_encoded_fields"]["episode"]
    # The unconditional fields by an S1a record, the table fields by an S1e one (which has both).
    assert stringified(record) == sorted(set(declared) - set(_table_fields()))
    assert stringified(_table_record(table_recording(0, 0))[0]) == sorted(declared)
    assert set(declared) & set(_table_fields()) == {"episode_spec"}


def test_the_assembled_dataset_carries_only_the_rules_its_scenarios_declare(tmp_path):
    """정본을 그대로 복사하지 않고 좁혀 쓰는지 — 헬퍼가 아니라 **조립 경로**를 잰다.

    release_canon 만 시험하면 조립이 그것을 부르지 않아도 초록입니다. 좁히기가
    빠지면 배포본은 적용한 적 없는 규칙 넷을 싣고, 그 미기재 임계값 4건 때문에
    C1 이 다시 빨강이 됩니다.
    """
    import json

    from generation.convert_lerobot import assemble_dataset
    from validator.validate_dataset import threshold_problems

    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    out = _lerobot_output_stub(tmp_path)
    copied = assemble_dataset(repo_root, out, assets, ["S1a"], ["R-TS15066-A2-12"],
                              referenced=V01_ASSETS)

    assert "meta/rules.json" in copied and "meta/scenarios.json" in copied

    shipped_rules = json.loads((out / "meta" / "rules.json").read_text(encoding="utf-8"))
    ids = {r["rule_id"] for r in shipped_rules["rules"]}
    assert ids.isdisjoint({"R-25785-STAB-01", "R-25785-STAB-02",
                           "R-22440-CTRL-01", "R-OBJ-DROP-01"})
    assert threshold_problems(shipped_rules) == [], "배포본에 미기재 임계값이 남았습니다"

    shipped_scenarios = json.loads((out / "meta" / "scenarios.json").read_text(encoding="utf-8"))
    assert [s["scenario_id"] for s in shipped_scenarios["scenarios"]] == ["S1a"]

    # 좁혔다는 사실이 배포본 안에 남아야 한다 — 규격서 목록과 다른 이유를 읽는 쪽이
    # 알 수 있어야 하기 때문이다.
    assert shipped_rules["excluded_rules"]["rule_ids"]
    assert shipped_scenarios["excluded_scenarios"]["scenario_ids"]


def test_the_assembled_dataset_refers_to_no_file_it_does_not_carry(tmp_path):
    """배포본 안의 정본이 배포본에 없는 경로를 가리키면 안 된다.

    `spec_document` 가 `docs/schema/…md` 를 가리킨 적이 있는데
    그 문서는 동봉되지 않는다. 내려받은 쪽이 따라가면 없는 파일이다. 앞으로
    다른 키가 같은 실수를 해도 여기서 걸리게 한다.
    """
    import json

    from generation.convert_lerobot import assemble_dataset

    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    out = _lerobot_output_stub(tmp_path)
    assemble_dataset(repo_root, out, assets, ["S1a"], referenced=V01_ASSETS)

    canon = json.loads((out / "meta" / "safety_schema.json").read_text(encoding="utf-8"))

    ref = canon["spec_document"]
    if ref.get("bundled_in_dataset"):
        assert (out / ref["path_in_repository"]).exists()
    else:
        # 동봉하지 않는다고 선언했으면, 사람이 읽을 것은 배포본에 있어야 한다.
        assert (out / "README.md").exists()

    def strings(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for v in node.values():
                yield from strings(v)
        elif isinstance(node, list):
            for v in node:
                yield from strings(v)

    declared_absent = {ref["path_in_repository"]}
    looks_like_path = [
        s for s in strings(canon)
        if "/" in s and s.rsplit(".", 1)[-1] in {"md", "json", "yaml", "yml", "usd", "parquet"}
    ]
    dangling = [s for s in looks_like_path
                if s not in declared_absent and not (out / s).exists()]
    assert dangling == [], f"배포본에 없는 경로를 가리킴: {dangling}"


# ── 변환기의 폭·주기 검증 ────────────────

def test_every_joint_width_column_is_checked_not_only_joint_pos():
    """넷이 일관되게 잘못된 배치는 대조 하나로는 안 잡힙니다."""
    import numpy as np
    import pytest

    from generation.convert_lerobot import step_frames

    _, joint_map, _, _ = _canon()
    arrays = _arrays(n=3)
    # joint_pos 는 정본대로 두고 나머지만 좁힌다 — 옛 검사는 통과했습니다.
    for name in ("joint_vel", "joint_torque", "action"):
        arrays[name] = arrays[name][:, :-1]
    with pytest.raises(ValueError, match="dof 불일치"):
        step_frames(arrays, _meta(), joint_map)


def test_a_rate_that_no_integer_fps_expresses_is_refused():
    """모든 에피소드가 같아도 정수 fps 로 표현 못 하면 timestamp 가 어긋납니다.

    앞의 시험은 이름만 「혼합 주기 거부」였고 메타 두 개의 fps 가 다른 것만 확인해,
    **검사 코드를 지워도 초록**이었습니다.
    """
    import pytest

    from generation.convert_lerobot import check_record_rate

    check_record_rate(0.02, 50)                       # 표현 가능 — 통과
    with pytest.raises(ValueError, match="정수 fps"):
        check_record_rate(0.0201, 50)                 # round(1/0.0201) == 50


def test_a_second_episode_with_a_different_rate_is_refused():
    import pytest

    from generation.convert_lerobot import check_record_rate

    check_record_rate(0.02, 50, 0.02, "S1a_0001_seed1")
    with pytest.raises(ValueError, match="다릅니다"):
        check_record_rate(0.01, 50, 0.02, "S1a_0001_seed1")


def test_main_actually_calls_the_rate_check():
    """헬퍼만 시험하면 호출부를 지워도 초록입니다.

    이 저장소에서 이미 두 번 나온 형태입니다 — 헬퍼는 옳은데 아무도 부르지 않으면
    결함은 그대로입니다. 변환 본체는 convert() 이고
    main() 은 LeRobot 을 넘겨 그것을 부를 뿐이므로, 호출이 둘 다에 실재하는지 봅니다.
    """
    import inspect

    import generation.convert_lerobot as conv

    assert "convert(" in inspect.getsource(conv.main)
    body = inspect.getsource(conv.convert)
    assert body.count("check_record_rate(") >= 2, (
        "main() 이 기록 주기 검사를 부르지 않습니다 — 첫 에피소드의 표현 가능성과 "
        "뒤 에피소드의 일치, 두 곳에서 불려야 합니다")


def _meta_root(tmp_path, frames):
    """LeRobot 이 쪼개 쓴 meta/episodes/ 를 흉내 냅니다."""
    import pandas as pd

    d = tmp_path / "meta" / "episodes" / "chunk-000"
    d.mkdir(parents=True)
    for i, df in enumerate(frames):
        pd.DataFrame(df).to_parquet(d / f"file-{i:03d}.parquet", index=False)
    return tmp_path


def test_episode_metadata_joins_across_split_files(tmp_path):
    """400 에피소드에서 이 표는 더 이상 파일 하나가 아닙니다.

    LeRobot 은 meta/episodes 를 크기로 쪼갭니다. 이어 붙이기(concat)는 파일 안에서만
    성립하므로, 쪼개진 뒤에도 위치로 붙이면 **에피소드마다 남의 라벨이 실립니다.**
    """
    import pandas as pd

    from generation.convert_lerobot import merge_episode_metadata

    # 두 번째 파일이 먼저 오는 순서가 아니라, 파일 안의 순서가 전역 순서와 다른
    # 상황을 만듭니다 — 위치로 붙이면 여기서 어긋납니다.
    root = _meta_root(tmp_path, [
        [{"episode_index": 0, "length": 10}, {"episode_index": 2, "length": 30}],
        [{"episode_index": 1, "length": 20}],
    ])
    records = [{"episode_id": f"S1a_{i:04d}_seed{i}", "outcome": oc}
               for i, oc in enumerate(["fail", "near_miss", "nominal"])]

    merge_episode_metadata(root, records)

    merged = pd.concat([pd.read_parquet(p)
                        for p in sorted(root.rglob("*.parquet"))])
    got = dict(zip(merged["episode_index"], merged["episode_id"]))
    assert got == {0: "S1a_0000_seed0", 1: "S1a_0001_seed1", 2: "S1a_0002_seed2"}


def test_an_episode_without_a_conversion_record_is_refused(tmp_path):
    import pytest

    from generation.convert_lerobot import merge_episode_metadata

    root = _meta_root(tmp_path, [[{"episode_index": 0}, {"episode_index": 7}]])
    with pytest.raises(ValueError, match="인덱스 불일치"):
        merge_episode_metadata(root, [{"episode_id": "S1a_0000_seed0"},
                                      {"episode_id": "S1a_0001_seed1"}])


def test_the_same_episode_in_two_shards_is_refused(tmp_path):
    """샤드마다 one_to_one 을 걸어도 샤드를 가로지르는 중복은 못 봅니다.

    두 샤드가 각각 episode_index 0 을 담고 1 을 빠뜨리면 합계가 records 와 같아
    통과했고, 0 번이 두 번 실리고 1 번이 사라졌습니다.
    """
    import pytest

    from generation.convert_lerobot import merge_episode_metadata

    root = _meta_root(tmp_path, [[{"episode_index": 0}], [{"episode_index": 0}]])
    with pytest.raises(ValueError, match="두 파일에 있습니다"):
        merge_episode_metadata(root, [{"episode_id": "S1a_0000_seed0"},
                                      {"episode_id": "S1a_0001_seed1"}])


def test_duplicate_episode_ids_are_refused(tmp_path):
    import pytest

    from generation.convert_lerobot import merge_episode_metadata

    root = _meta_root(tmp_path, [[{"episode_index": 0}, {"episode_index": 1}]])
    with pytest.raises(ValueError, match="중복된 episode_id"):
        merge_episode_metadata(root, [{"episode_id": "dup"}, {"episode_id": "dup"}])


def test_nothing_is_written_when_a_later_shard_is_wrong(tmp_path):
    """실패가 원자적이어야 재실행이 됩니다.

    앞 샤드에 열을 붙인 뒤 뒤에서 실패하면, 재실행이 「열 이름 겹침」으로 다시
    거부되어 손으로 치워야 했습니다.
    """
    import pandas as pd
    import pytest

    from generation.convert_lerobot import merge_episode_metadata

    root = _meta_root(tmp_path, [[{"episode_index": 0}], [{"episode_index": 9}]])
    before = [pd.read_parquet(p) for p in sorted(root.rglob("*.parquet"))]
    with pytest.raises(ValueError):
        merge_episode_metadata(root, [{"episode_id": "S1a_0000_seed0"},
                                      {"episode_id": "S1a_0001_seed1"}])
    after = [pd.read_parquet(p) for p in sorted(root.rglob("*.parquet"))]
    for b, a in zip(before, after):
        assert list(b.columns) == list(a.columns), "실패했는데 샤드가 이미 수정됐습니다"


def test_a_mistyped_flag_is_refused_not_coerced():
    """`bool("false")` 는 True 입니다.

    사이드카가 문자열을 담으면 변환기가 조용히 뒤집힌 값을 배포본에 싣습니다.
    강제 변환 대신 거부해야 합니다.
    """
    import pytest

    from generation.convert_lerobot import require_type

    assert require_type({"degenerate": True}, "degenerate", bool) is True
    for bad in ("false", 1, 0, None):
        with pytest.raises(ValueError, match="형이 bool"):
            require_type({"degenerate": bad}, "degenerate", bool)
    with pytest.raises(ValueError, match="형이 dict"):
        require_type({"termination": '{"terminated": true}'}, "termination", dict)


def test_main_actually_type_checks_the_two_new_fields():
    """헬퍼만 시험하면 호출부를 지워도 초록입니다 — 이 저장소에서 세 번 나온 형태."""
    import inspect

    import generation.convert_lerobot as conv

    body = inspect.getsource(conv.build_episode_meta)
    assert 'require_type(meta, "degenerate", bool)' in body
    assert "require_termination(meta)" in body



def test_termination_members_are_checked_not_just_the_container():
    """빈 객체도 dict 입니다.

    바깥 그릇만 보면 `{}` 와 `{"terminated": "yes", "reason": "fall"}` 가 모두
    통과해, 「왜 끝났는가」가 손상된 채 배포됩니다.
    """
    import pytest

    from generation.convert_lerobot import require_termination

    good = {"terminated": True, "truncated": False, "reason": ["fall_orientation"]}
    assert require_termination({"termination": good}) == good
    for bad in ({}, {"terminated": "yes", "truncated": 0, "reason": "fall"},
                {"terminated": True, "truncated": False, "reason": []},
                {"terminated": True, "truncated": False, "reason": [1]},
                {"terminated": True, "reason": ["x"]}):
        with pytest.raises(ValueError):
            require_termination({"termination": bad})


def test_the_validator_reports_the_same_termination_problems():
    """변환기와 검증기가 같은 계약을 봐야 합니다.

    한쪽만 조이면 배포본은 통과하는데 생성기가 거부하거나, 그 반대가 됩니다.
    """
    from validator.validate_dataset import termination_problems

    assert termination_problems(
        {"terminated": True, "truncated": False, "reason": ["time_out"]}) == []
    assert termination_problems({})
    assert termination_problems({"terminated": "yes", "truncated": 0, "reason": "fall"})
    assert termination_problems({"terminated": True, "truncated": False, "reason": []})


def test_the_converter_refuses_an_asset_the_canon_declares_but_the_tree_lacks(tmp_path):
    """`redistributed=true` 인데 파일이 없으면 프레임을 쓰기 전에 멈춘다.

    이 가드에는 시험이 없었습니다. 있던 것은 정본을 옮기는 시험 넷이 **우연히**
    자산을 요구한다는 사실뿐이었고, 그래서 파일이 없는 환경에서는 가드가 도는 것이
    아니라 **그 시험 넷이 죽었습니다**. 우연한 요구는 시험이 아니므로 여기서 의도적으로
    겁니다.
    """
    import json

    from generation.convert_lerobot import assemble_dataset

    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    declared = [e for e in (assets.get("scene") or {}).values() if e.get("redistributed")]
    assert declared, "정본이 재배포 자산을 선언하지 않습니다 — 이 시험의 전제가 깨졌습니다"

    victim = repo_root / declared[0]["usd_path"]
    assert victim.exists(), "도우미가 자산을 놓지 않았습니다"
    victim.unlink()

    out = _lerobot_output_stub(tmp_path)
    with pytest.raises(FileNotFoundError, match="redistributed=true") as raised:
        assemble_dataset(repo_root, out, assets, ["S1a"], referenced=V01_ASSETS)

    assert declared[0]["usd_path"] in str(raised.value), \
        "어느 자산이 없는지 말하지 않으면 받는 쪽이 정본을 뒤져야 합니다"
    assert not (out / "assets" / victim.name).exists(), \
        "없는 자산의 자리를 배포본에 만들어 두었습니다"
    # 「`assets/` 자체가 없다」는 여기서 단언하지 않는다 — 자산이 하나뿐인 배치라 우연히
    # 참이다. 여럿일 때 아무것도 복사하기 전에 멈추는 것은
    # test_a_missing_texture_stops_before_any_asset_is_copied 가 따로 겁니다.


# ── 배치가 쓰는 자산만 싣는다 ─────────────────────────────

def _assets_tree(out):
    root = out / "assets"
    return sorted(str(q.relative_to(out)) for q in root.rglob("*") if q.is_file()) \
        if root.is_dir() else []


def _expected_files(assets, referenced):
    """What should ship, read off the canon: redistributable, referenced USDs and the rows of
    their bundles -- a robot with a ship block as its folder, every row. Written out here rather
    than by calling the function under test."""
    want = set()
    for group in ("robots", "scene"):
        for entry in (assets.get(group) or {}).values():
            if entry.get("redistributed") and entry["usd_path"] in referenced and "ship" in entry:
                want |= {f"assets/{entry['ship']['dir']}/{row['file']}" for row in entry["ship"]["files"]}
            elif entry.get("redistributed") and entry["usd_path"] in referenced:
                want.add("assets/" + Path(entry["usd_path"]).name)
                for row in (entry.get("bundle") or {}).get("files", []):
                    want.add("assets/textures/" + row["file"])
    return sorted(want)


def test_a_v0_1_batch_ships_the_v1_dummy_alone(tmp_path):
    """v0.1 불변 — S1a 에피소드가 쓰는 자체 제작 재배포 자산은 v1 더미 하나다(로봇은 모든 배치가
    그렇듯 제 폴더째 함께 실린다). 정본이 S1d 의 씬·입힌 더미를 재배포한다고 적어도 이 배치는
    그것을 싣지도 텍스처 디렉터리를 만들지도 않는다."""
    import json

    from generation.convert_lerobot import assemble_dataset

    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    unused = [e["usd_path"] for e in assets["scene"].values()
              if e.get("redistributed") and e["usd_path"] not in V01_ASSETS]
    assert unused, "정본에 이 배치가 쓰지 않는 재배포 자산이 없습니다 — 이 시험이 볼 것이 없습니다"

    out = _lerobot_output_stub(tmp_path)
    copied = assemble_dataset(repo_root, out, assets, ["S1a"], referenced=V01_ASSETS)
    assert _assets_tree(out) == _expected_files(assets, V01_ASSETS)
    robot = assets["robots"]["unitree_h1"]["ship"]
    assert [p for p in _assets_tree(out) if not p.startswith(f"assets/{robot['dir']}/")] == [
        "assets/" + Path(DUMMY).name]
    assert len(_assets_tree(out)) == 1 + len(robot["files"])
    assert not (out / "assets" / "textures").exists()
    assert [c for c in copied if c.startswith("assets/")] == _assets_tree(out)


def test_an_s1d_batch_ships_its_usds_and_their_textures(tmp_path):
    import json

    from generation.convert_lerobot import assemble_dataset

    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    out = _lerobot_output_stub(tmp_path)
    assemble_dataset(repo_root, out, assets, ["S1d"], referenced=S1D_ASSETS)

    want = _expected_files(assets, S1D_ASSETS)
    assert [w for w in want if w.startswith("assets/textures/")], "정본에 텍스처 묶음이 없습니다"
    assert _assets_tree(out) == want
    assert "assets/" + Path(DUMMY).name not in want, "S1d 에피소드는 v1 더미를 쓰지 않습니다"
    # The bytes are the tree's, file for file -- a USD beside the textures its ./textures/ names.
    for entry in assets["scene"].values():
        if entry["usd_path"] not in S1D_ASSETS or not entry.get("redistributed"):
            continue
        usd = repo_root / entry["usd_path"]
        assert (out / "assets" / usd.name).read_bytes() == usd.read_bytes()
        for row in (entry.get("bundle") or {}).get("files", []):
            src = repo_root / entry["bundle"]["textures_dir"] / row["file"]
            assert (out / "assets" / "textures" / row["file"]).read_bytes() == src.read_bytes()


def _victim_texture(repo_root, assets):
    entry = next(e for e in assets["scene"].values()
                 if e.get("bundle") and e["usd_path"] in S1D_ASSETS)
    row = entry["bundle"]["files"][-1]
    return repo_root / entry["bundle"]["textures_dir"] / row["file"]


def test_a_missing_texture_stops_before_any_asset_is_copied(tmp_path):
    import json

    from generation.convert_lerobot import assemble_dataset

    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    victim = _victim_texture(repo_root, assets)
    victim.unlink()
    out = _lerobot_output_stub(tmp_path)
    with pytest.raises(FileNotFoundError, match=victim.name):
        assemble_dataset(repo_root, out, assets, ["S1d"], referenced=S1D_ASSETS)
    assert not (out / "assets").exists(), "자산 하나가 없는데 나머지를 먼저 실어 두었습니다"


def test_a_texture_that_is_not_the_pinned_bytes_is_refused(tmp_path):
    import json

    from generation.convert_lerobot import assemble_dataset

    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    victim = _victim_texture(repo_root, assets)
    victim.write_bytes(victim.read_bytes() + b"x")
    out = _lerobot_output_stub(tmp_path)
    with pytest.raises(ValueError, match=f"{victim.name} 의 해시가 정본과 다릅니다"):
        assemble_dataset(repo_root, out, assets, ["S1d"], referenced=S1D_ASSETS)
    assert not (out / "assets").exists()


def test_bundle_rows_that_are_not_the_lock_are_refused(tmp_path):
    """The rows are the lock's, moved into meta/assets.json so they travel; a row that says
    something the lock does not is a provenance nobody pinned."""
    import json

    from generation.assets import shipped_asset_files

    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    entry = next(e for e in assets["scene"].values() if e.get("bundle"))
    entry["bundle"]["files"][0]["url"] = "https://example.invalid/elsewhere.png"
    with pytest.raises(ValueError, match="bundle.files 가 .*textures.lock 과 다릅니다"):
        shipped_asset_files(assets, [entry["usd_path"]], repo_root)


def _share_a_texture_name(repo_root, assets, same_bytes):
    """Give the skin bundle a row -- in its lock and in the canon, so the two still agree --
    named like one of the scene's textures, with the scene's hash or another one."""
    from generation.assets import bundle_per_lock

    # by the lock's full path: the KETI CI pin (generation/scene/keti_ci.json) is a scene lock too, and
    # a bundle may name several locks
    def naming(lock):
        return next(e for e in assets["scene"].values()
                    if e.get("bundle") and lock in bundle_per_lock(e["bundle"], "textures_lock"))

    scene = naming("generation/scene/textures.lock")
    skin = naming("generation/dummy/skin/textures.lock")
    taken = scene["bundle"]["files"][0]
    sha = taken["sha256"] if same_bytes else "0" * 64
    lock = repo_root / skin["bundle"]["textures_lock"]
    lock.write_text(lock.read_text(encoding="utf-8")
                    + f"{taken['file']} {sha} 1 pack:x/{taken['file']} CC0-1.0\n", encoding="utf-8")
    url = skin["bundle"]["files"][0]["url"]
    skin["bundle"]["files"].append({"file": taken["file"], "sha256": sha, "license": "CC0-1.0",
                                    "url": url, "member": f"x/{taken['file']}"})
    return taken["file"]


def test_one_texture_name_with_two_different_hashes_is_refused(tmp_path):
    import json

    from generation.assets import shipped_asset_files

    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    name = _share_a_texture_name(repo_root, assets, same_bytes=False)
    with pytest.raises(ValueError, match=f"이름이 같은데 바이트가 다릅니다: {name}"):
        shipped_asset_files(assets, S1D_ASSETS, repo_root)


def test_one_texture_both_bundles_pin_alike_ships_once(tmp_path):
    import json

    from generation.assets import shipped_asset_files

    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    name = _share_a_texture_name(repo_root, assets, same_bytes=True)
    shipped = shipped_asset_files(assets, S1D_ASSETS, repo_root)
    assert [k for k in shipped if k.endswith("/" + name)] == [f"assets/textures/{name}"]


@pytest.mark.parametrize("kind", ["relative-dotdot", "absolute"])
def test_the_stub_helper_refuses_a_path_that_leaves_the_tree(tmp_path, kind):
    """검사는 손해가 나기 **전에** 서야 검사입니다.

    앞 판은 만들고 지운 뒤에 단언했습니다. 재현하면 나무 밖 9바이트 파일이 0바이트가
    된 **다음에** AssertionError 가 났습니다 — 「심링크를
    따라 원본을 잘랐다」와 결과가 같습니다.

    두 형태 모두 **같은 sentinel 을 겨눕니다.** `root / "../outside.usd"` 와
    `root / <절대경로>` 는 둘 다 그 파일로 떨어지므로, 막지 못하면 내용이 비워집니다.

    지금 자산 경로는 저장소 상대 경로 하나뿐이라 이 길로 들어올 일이 없습니다.
    그러나 `usd_path` 는 사람이 고치는 정본 값이고 막는 비용이 한 줄입니다.
    """
    from pathlib import Path as P

    root = tmp_path / "tree"
    root.mkdir()
    outside = tmp_path / "outside.usd"
    outside.write_bytes(b"MUST-STAY")

    target = P("../outside.usd") if kind == "relative-dotdot" else outside
    assert (root / target).resolve() == outside.resolve(), "시험이 sentinel 을 겨누지 않습니다"

    with pytest.raises(ValueError, match="나무 밖"):
        _stub_at(root, target)

    assert outside.read_bytes() == b"MUST-STAY", "거부하기 전에 나무 밖 파일을 건드렸습니다"


def _release_with_non_finite(tmp_path):
    """스텝 parquet 에 `-inf` 가 있고 통계는 무너진 배포본 흉내."""
    import json

    import pandas as pd

    root = tmp_path / "dist"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    pd.DataFrame({
        "safety.stability_margin": [-0.5, -0.2, float("-inf"), 0.1],
        "observation.state.joint_pos": [[0.1, 0.2]] * 4,
    }).to_parquet(root / "data" / "chunk-000" / "file-000.parquet", index=False)

    # LeRobot 이 내는 모양 — 분위수가 전부 최댓값 부근으로 무너져 **유한한 채로** 틀리다.
    (root / "meta" / "stats.json").write_text(json.dumps({
        "safety.stability_margin": {
            "min": [float("-inf")], "max": [0.1], "mean": [float("nan")],
            "std": [float("nan")], "count": [4],
            "q01": [0.1], "q10": [0.1], "q50": [0.1], "q90": [0.1], "q99": [0.1],
        },
        "observation.state.joint_pos": {
            "min": [0.1, 0.2], "max": [0.1, 0.2], "mean": [0.1, 0.2],
            "std": [0.0, 0.0], "count": [4],
            "q01": [0.1, 0.2], "q50": [0.1, 0.2], "q99": [0.1, 0.2],
        },
    }), encoding="utf-8")
    return root


def test_stats_of_a_field_with_non_finite_values_are_not_shipped(tmp_path):
    """비유한값이 있는 필드의 통계는 비운다 — 분위수가 **유한한 채로** 틀리기 때문.

    2026-08-29 실측: 배포본의 `safety.stability_margin` 이 q01~q99 전부 0.1235 로
    실려 있었고 유한값의 실제 중앙값은 -0.1302 였다. 「음수면 지지다각형 밖」인
    필드의 중앙값이 부호가 뒤집힌 채 나가고 있었다. NaN 은 시끄럽게 실패하지만
    무너진 분위수는 조용히 통과한다 — 앞선 `NaN → null` 청소가 그것을 못 봤다.
    """
    import json

    from generation.convert_lerobot import normalise_stats

    root = _release_with_non_finite(tmp_path)
    normalise_stats(root)
    doc = json.loads((root / "meta" / "stats.json").read_text(encoding="utf-8"))

    broken = doc["safety.stability_margin"]
    assert broken["count"] == [4], "표본 수까지 지우면 몇 개에서 나온 통계인지도 사라집니다"
    assert all(v is None for k, v in broken.items()
               if k not in ("count", "undefined_reason")), (
        f"믿을 수 없는 통계가 남았습니다: {broken}")
    assert "undefined_reason" in broken, "왜 비었는지 적혀 있지 않으면 결측으로 읽힙니다"

    clean = doc["observation.state.joint_pos"]
    assert clean["q50"] == [0.1, 0.2], "비유한값이 없는 필드는 건드리지 않아야 합니다"


def test_the_blanking_reads_the_data_not_a_hand_written_list(tmp_path):
    """어느 필드인지는 배포본의 parquet 에서 유도한다.

    목록을 코드에 적어 두면 새 필드가 `±inf` 를 담게 될 때 조용히 빠진다.
    """
    import json

    from generation.convert_lerobot import features_with_non_finite, normalise_stats

    root = _release_with_non_finite(tmp_path)
    assert features_with_non_finite(root) == {"safety.stability_margin"}

    # 멀쩡하던 필드에 NaN 이 생기면 그 필드도 저절로 대상이 된다.
    import pandas as pd
    p = root / "data" / "chunk-000" / "file-000.parquet"
    frame = pd.read_parquet(p)
    frame["observation.state.joint_pos"] = [[0.1, float("nan")]] * len(frame)
    frame.to_parquet(p, index=False)
    assert features_with_non_finite(root) == {"safety.stability_margin",
                                              "observation.state.joint_pos"}
    normalise_stats(root)
    doc = json.loads((root / "meta" / "stats.json").read_text(encoding="utf-8"))
    assert doc["observation.state.joint_pos"]["q50"] is None


# ── 카메라 매체 ───────────────────────────────────────
#
# lerobot 과 av 없이 변환 본체 convert() 를 끝까지 돌린다. LeRobotDataset.create 대신
# 받은 프레임을 적어 두는 가짜를, PyAV 대신 메모리의 그림을 내는 디코더를 넘긴다.
# 카메라 정본은 저장소의 것을 읽으므로 해상도를 손으로 적지 않는다. 팩은 저장소 정본의 것을
# 쓰지 않는다 — 정본은 팩을 끄고 팩이 필요한 시험은 픽스처의 팩
# (validator/make_fixture.canon_with_pack, 글자로 적힌 모양)을 명시적으로 얹는다.

def _camera_canon():
    from generation.cameras import load
    return load(REPO_ROOT / "meta" / "cameras.json")


def _packless(doc):
    """The camera canon with its packs switched off -- the shape of the repository canon."""
    import copy
    out = copy.deepcopy(doc)
    out.pop("packs", None)
    return out


def _with_a_pack(doc, downsample=None):
    """``doc`` declaring the fixture's pack (make_fixture.canon_with_pack: a literal on
    bbox.camera) in place of any it has -- never the repository canon's own. At the identity
    downsample of the pack camera unless ``downsample`` says otherwise."""
    from validator.make_fixture import canon_with_pack
    cam = doc["cameras"][doc["bbox"]["camera"]]
    return canon_with_pack(_packless(doc), downsample or {"width": cam["width"], "height": cam["height"],
                                                          "every_n_steps": 1})


class _FakeDataset:
    """LeRobotDataset 의 쓰기 쪽 넷. 프레임은 LeRobot 의 validate_frame 처럼 피처와
    키가 같아야 받고, finalize 는 뒤 단계가 읽는 meta 두 가지를 쓴다.

    영상 프레임은 LeRobot 처럼 images/<키>/episode-XXXXXX/frame-XXXXXX.png 에 내려 두고,
    save_episode 가 인코딩한 셈 치고 그 에피소드 디렉터리를 지운다 — 키 디렉터리는 빈 채
    남는다(0.6.1 실측). ``keep_staged`` 면 지우지 않는다: 인코딩되지 않은 프레임."""

    def __init__(self, root, features, fps, keep_staged=False):
        self.root, self.features, self.fps = Path(root), features, fps
        self.episodes, self.current = [], []
        self.keep_staged, self.staged = keep_staged, False
        self.root.mkdir(parents=True)

    def _episode_dir(self, key):
        return self.root / "images" / key / f"episode-{len(self.episodes):06d}"

    def add_frame(self, frame):
        assert set(frame) - {"task"} == set(self.features), sorted(set(frame) ^ set(self.features))
        for key, ft in self.features.items():
            if ft["dtype"] == "video":
                assert frame[key].shape == ft["shape"] and frame[key].dtype == np.uint8
                d = self._episode_dir(key)
                d.mkdir(parents=True, exist_ok=True)
                (d / f"frame-{len(self.current):06d}.png").write_bytes(b"png")
                self.staged = True
        self.current.append(frame)

    def save_episode(self):
        import shutil
        if not self.keep_staged:
            for key, ft in self.features.items():
                if ft["dtype"] == "video":
                    shutil.rmtree(self._episode_dir(key))
        self.episodes.append(self.current)
        self.current = []

    def finalize(self):
        import json

        import pandas as pd
        (self.root / "meta").mkdir(exist_ok=True)
        (self.root / "meta" / "info.json").write_text(json.dumps({"fps": self.fps}), encoding="utf-8")
        d = self.root / "meta" / "episodes" / "chunk-000"
        d.mkdir(parents=True)
        pd.DataFrame({"episode_index": range(len(self.episodes))}).to_parquet(
            d / "file-000.parquet", index=False)


class _Harness:
    """에피소드 디렉터리 + 저장소 나무 + 가짜 LeRobot/PyAV. ``pictures`` 는 영상 경로 →
    그림 목록이고 디코더와 프로버가 둘 다 여기서 읽는다(프로버만 따로 속일 수 있게
    ``probe_frames`` 로 덮을 수 있다).

    카메라 정본은 저장소의 것에서 팩을 뺀 것이 기본이다(정본이 팩을 끈 뒤의 꼴). ``pack`` 이면
    픽스처의 팩을 항등 축소로 얹는다(_with_a_pack) — 저장소 정본이 팩을 선언하든 말든 같다."""

    def __init__(self, tmp_path, pack=False):
        import json
        self.tmp = tmp_path
        self.repo = _repo_root_with_redistributables(tmp_path)
        canon = json.loads((REPO_ROOT / "meta" / "cameras.json").read_text(encoding="utf-8"))
        self.write_cameras(_with_a_pack(canon) if pack else _packless(canon))
        self.dof = int(__import__("yaml").safe_load(
            (self.repo / "meta/joint_map/unitree_h1.yaml").read_text(encoding="utf-8"))["dof"])
        self.episodes = tmp_path / "episodes"
        self.episodes.mkdir()
        self.out = tmp_path / "out"
        self.pictures: dict[Path, list] = {}
        self.probe_frames: dict[Path, int] = {}
        self.created: list[dict] = []
        self.decoded: list[Path] = []
        self.keep_staged = False

    def write_cameras(self, doc):
        import json
        self.cameras = doc
        (self.repo / "meta" / "cameras.json").write_text(json.dumps(doc), encoding="utf-8")

    def add_episode(self, seq, n=3, media=True, pictures=None, stem=None, assets=S1D_ASSETS,
                    scenario="S1d", period=None, pack=None):
        """``stem`` names the episode's three files when they should not be named after its id;
        ``assets`` are the paths its sidecar says were loaded (a real S1d sidecar's four, or
        V01_ASSETS with scenario="S1a"). A camera episode is recorded at the canon's fps as it
        is when added -- edit the canon afterwards to model an edit after recording -- unless
        ``period`` (seconds per step) says otherwise; and with the canon's pack when it declares
        one, unless ``pack`` says otherwise (False: no ``media.pack``; True: a pack, which needs a
        canon that declares one to take its camera from)."""
        import json
        # Read off the canon here, not through the_pack: the harness builds the input the code
        # under test is then held to.
        declared = self.cameras.get("packs") or {}
        if pack is None:
            pack = bool(declared)
        eid = f"{scenario}_{seq:04d}_seed{seq + 1}"
        meta = dict(_meta(), episode_id=eid, scenario_id=scenario, episode_seq=seq, seed=seq + 1,
                    steps=n, asset_paths={f"a{i}": path for i, path in enumerate(assets)})
        arrays = _arrays(n=n, dof=self.dof)
        if media:
            arrays["timestamp"] = np.arange(n) * (period or 1.0 / self.cameras["fps"])
            # recorded under the canon as it is when the episode is added, as its fps is
            meta, arrays = _with_cameras(meta, arrays, self.cameras)
            meta["media"] = {k: (dict(v, file=v["file"].replace("S1a_0000_seed1", eid))
                                 if isinstance(v, dict) and "file" in v else v)
                             for k, v in meta["media"].items()}
            (self.episodes / "media").mkdir(exist_ok=True)
            for cam in cl.video_cameras().values():
                spec = self.cameras["cameras"][cam]
                path = self.episodes / meta["media"][cam]["file"]
                path.write_bytes(b"placeholder")
                # A value per camera (its place in the canon) and per step, never 0: streams stay apart
                k = list(self.cameras["cameras"]).index(cam)
                self.pictures[path] = (pictures or {}).get(cam) or [
                    np.full((spec["height"], spec["width"], 3), (i * 7 + 32 * k + 1) % 256, np.uint8)
                    for i in range(n)]
            if not pack:
                del meta["media"]["pack"]
            else:
                assert len(declared) == 1, "pack=True needs a canon that declares one pack"
                (camera,) = [p["camera"] for p in declared.values()]
                spec = self.cameras["cameras"][camera]
                rng = np.random.default_rng(seq)
                np.savez_compressed(
                    self.episodes / meta["media"]["pack"]["file"],
                    depth=rng.integers(0, 5000, (n, spec["height"], spec["width"])).astype(np.uint16),
                    mask=rng.integers(0, 15, (n, spec["height"], spec["width"])).astype(np.uint8),
                    step_index=np.arange(n, dtype=np.int32))
        stem = stem or eid
        np.savez(self.episodes / f"{stem}.npz", **arrays)
        (self.episodes / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")
        (self.episodes / f"{stem}.labels.json").write_text(json.dumps(_label()), encoding="utf-8")
        return eid, meta

    def create(self, repo_id, fps, root, features, **kw):
        self.created.append(dict(kw, repo_id=repo_id, fps=fps, root=root, features=features))
        self.dataset = _FakeDataset(root, features, fps, keep_staged=self.keep_staged)
        return self.dataset

    def decode(self, path):
        self.decoded.append(Path(path))
        return iter(self.pictures[Path(path)])

    def probe(self, path):
        pics = self.pictures[Path(path)]
        h, w = pics[0].shape[:2]
        return w, h, self.probe_frames.get(Path(path), len(pics))

    def run(self):
        from generation.convert_lerobot import convert
        return convert(self.episodes, self.out, "test/s1d", repo_root=self.repo,
                       create_dataset=self.create, decode_video=self.decode, probe_video=self.probe)


def test_a_camera_less_sidecar_gives_the_v0_1_feature_list():
    """매체 없는 사이드카는 카메라 조건이 켜지기 전과 같은 피처 목록을 낸다 — v0.1 재변환이
    영상 피처나 카메라 배열 열을 얻으면 배포본 모양이 바뀐다."""
    from generation.convert_lerobot import (LEROBOT_OWNED, dataset_features, lerobot_features,
                                            recorded_dtypes, to_lerobot_frame)
    _, joint_map, _, _ = _canon()
    meta, arrays = _meta(), _arrays()
    frame = to_lerobot_frame(step_frames(arrays, meta, joint_map)[0], recorded_dtypes(arrays, meta))
    features = dataset_features(frame, None)
    assert features == lerobot_features(frame)
    assert set(features) == {f.schema for f in STEP_FIELDS if f.condition is None} - set(LEROBOT_OWNED)
    assert not [k for k, ft in features.items() if ft["dtype"] == "video"]


def test_the_camera_streams_are_declared_video_features_at_the_canon_resolution():
    # Derived from a frame, an (H, W, 3) uint8 array would become a parquet column.
    from generation.convert_lerobot import (dataset_features, recorded_dtypes, to_lerobot_frame,
                                            video_cameras)
    cameras = _camera_canon()
    _, joint_map, _, _ = _canon()
    meta, arrays = _with_cameras(_meta(), _arrays())
    frame = to_lerobot_frame(step_frames(arrays, meta, joint_map)[0], recorded_dtypes(arrays, meta))
    features = dataset_features(frame, cameras)
    videos = {f.schema for f in STEP_FIELDS if f.kind == "video"}
    assert videos and set(video_cameras()) == videos
    for schema, cam in video_cameras().items():
        spec = cameras["cameras"][cam]
        assert features[schema] == {"dtype": "video", "shape": (spec["height"], spec["width"], 3),
                                    "names": ["height", "width", "channels"]}
    assert features["observation.bbox2d.human"]["dtype"] == "int32"
    # Depth and masks are never features, pack or no pack: a pack is copied beside the data.
    assert not [k for k in features if k.startswith(("observation.depth.", "observation.segmentation."))]


def test_a_camera_batch_converts_with_videos_packs_and_the_camera_canon(tmp_path):
    import hashlib
    import json

    from generation.cameras import intrinsics
    h = _Harness(tmp_path, pack=True)
    h.add_episode(0, n=3)
    h.add_episode(1, n=4)
    assert h.run() == 0

    # Every picture reached LeRobot, in step order, under its video field.
    for ep, n in zip(h.dataset.episodes, (3, 4)):
        assert len(ep) == n
    first = sorted(h.pictures)[0]
    cam_field = "observation.images." + first.name.split(".")[-2]
    assert [f[cam_field][0, 0, 0] for f in h.dataset.episodes[0]] == \
        [p[0, 0, 0] for p in h.pictures[first]]
    assert h.created[0]["features"][cam_field]["dtype"] == "video"

    # The packs, copied byte for byte at the identity downsample, and their index.
    index = json.loads((h.out / "meta" / "packs.json").read_text(encoding="utf-8"))
    name = next(iter(h.cameras["packs"]))
    assert set(index) == {"0", "1"}
    for i, n in ((0, 3), (1, 4)):
        entry = index[str(i)]
        assert entry["file"] == f"packs/{name}/episode_{i:06d}.npz" and entry["frames"] == n
        shipped = (h.out / entry["file"]).read_bytes()
        assert hashlib.sha256(shipped).hexdigest() == entry["sha256"]
        src = sorted((h.episodes / "media").glob("*.pack.npz"))[i]
        assert shipped == src.read_bytes()

    # meta/cameras.json ships, with the derived intrinsics the repository canon leaves out.
    shipped_cams = json.loads((h.out / "meta" / "cameras.json").read_text(encoding="utf-8"))
    for cam, spec in h.cameras["cameras"].items():
        assert "fx" not in spec
        fx, fy, cx, cy = intrinsics(spec)
        got = shipped_cams["cameras"][cam]
        assert (got["fx"], got["fy"], got["cx"], got["cy"]) == (fx, fy, cx, cy)
        assert {k: v for k, v in got.items() if k not in ("fx", "fy", "cx", "cy")} == spec

    # The recorded streams are the ground for a re-conversion; the converter leaves them.
    assert all(p.is_file() for p in h.pictures)


# ── Packs only when the canon declares one ──────────
#
# The pack code stays and the canon switches it off: a meta/cameras.json without ``packs`` is
# recorded, converted and shipped without them, as a batch without meta/cameras.json is without
# any camera media.

def test_a_canon_without_packs_has_no_pack():
    assert cl.the_pack({"cameras": {"head_rgb": {}}}) is None


def test_a_canon_with_two_packs_is_refused():
    with pytest.raises(ValueError, match=r"packs 는 많아야 하나여야 합니다: \['a', 'b'\]"):
        cl.the_pack({"packs": {"a": {}, "b": {}}})


@pytest.mark.parametrize("packs,want", [(None, None), ({}, None), ({"head": {"camera": "head_rgb"}}, "head")],
                         ids=["null", "empty", "one"])
def test_the_pack_is_the_one_the_canon_declares(packs, want):
    assert cl.the_pack({"cameras": {"head_rgb": {}}, "packs": packs}) == want


def test_a_packs_block_that_is_not_an_object_is_refused():
    # [] would read as "no pack" through `or {}`, and a list of packs as nothing at all.
    with pytest.raises(ValueError, match="packs 가 객체가 아닙니다"):
        cl.the_pack({"packs": [{"camera": "head_rgb"}]})


def test_a_camera_batch_without_packs_converts_videos_only(tmp_path):
    import json
    h = _Harness(tmp_path)
    assert "packs" not in h.cameras, "the harness's default canon is the packless one"
    metas = [h.add_episode(0, n=3)[1], h.add_episode(1, n=4)[1]]
    assert all("pack" not in m["media"] for m in metas), "the recording carried a pack after all"
    assert h.run() == 0

    # The videos reached LeRobot as before; nothing about a pack was written or shipped.
    for ep, n in zip(h.dataset.episodes, (3, 4)):
        assert len(ep) == n
    videos = {k for k, ft in h.created[0]["features"].items() if ft["dtype"] == "video"}
    assert videos == set(cl.video_cameras())
    assert not (h.out / "packs").exists()
    assert not (h.out / "meta" / "packs.json").exists()
    shipped = json.loads((h.out / "meta" / "cameras.json").read_text(encoding="utf-8"))
    assert "packs" not in shipped and set(shipped["cameras"]) == set(h.cameras["cameras"])


def _rewrite_sidecar(h, eid, change):
    import json
    path = h.episodes / f"{eid}.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    change(meta)
    path.write_text(json.dumps(meta), encoding="utf-8")


def test_the_converter_refuses_a_recording_that_does_not_name_committed_code(tmp_path):
    """첫 프레임 전에 멈춘다 — 둘째 에피소드에서 알면 첫 에피소드가 이미 쓰였다."""
    h = _Harness(tmp_path)
    h.add_episode(0)
    eid, _ = h.add_episode(1)
    _rewrite_sidecar(h, eid, lambda m: m["sim_backend"].update(generator_dirty=True))
    with pytest.raises(ValueError, match="generator_dirty 가 false 가 아닙니다: True"):
        h.run()
    assert h.created == [], "거부하기 전에 배포본을 만들기 시작했습니다"


def test_the_converter_refuses_a_recording_made_where_git_could_not_say(tmp_path):
    h = _Harness(tmp_path)
    eid, _ = h.add_episode(0)
    _rewrite_sidecar(h, eid, lambda m: m["sim_backend"].update(generator_commit=None, generator_dirty=None))
    with pytest.raises(ValueError, match="40자 16진 커밋 해시가 아닙니다: None"):
        h.run()
    assert h.created == []


def test_the_converter_refuses_a_recording_without_its_runs_first_seq(tmp_path):
    h = _Harness(tmp_path)
    eid, _ = h.add_episode(0)
    _rewrite_sidecar(h, eid, lambda m: m.pop("run_start_seq"))
    with pytest.raises(ValueError, match="run_start_seq 가 없습니다"):
        h.run()
    assert h.created == []


def test_the_converter_refuses_a_later_episode_whose_run_start_the_episode_could_not_have_had(tmp_path):
    """The first episode's record is built before the dataset is created, so a bad run start
    there is refused anyway; only a later episode's depends on the pre-check loop."""
    h = _Harness(tmp_path)
    h.add_episode(0)
    eid, _ = h.add_episode(1)
    _rewrite_sidecar(h, eid, lambda m: m.update(run_start_seq=2))
    with pytest.raises(ValueError, match="0 과 자기 번호 1 사이가 아닙니다"):
        h.run()
    assert h.created == [], "거부하기 전에 배포본을 만들기 시작했습니다"


def test_a_recording_that_names_a_pack_the_canon_does_not_declare_is_refused(tmp_path):
    """Recorded while the canon declared a pack, converted after it was switched off (or an old
    recording under the new canon): the pack would be dropped without a word. Refused before any
    frame, as a canon fps edited after recording is."""
    h = _Harness(tmp_path, pack=True)
    eid, meta = h.add_episode(0, n=2)
    assert "pack" in meta["media"]
    h.write_cameras(_packless(h.cameras))
    with pytest.raises(ValueError, match=f"{eid}: .*media.pack.* 녹화 때와 정본이 다름"):
        h.run()
    assert not h.created and not h.out.exists()


def test_a_recording_without_the_pack_the_canon_declares_is_refused(tmp_path):
    # The other side: the canon declares a pack and the sidecar names none -- still required.
    h = _Harness(tmp_path, pack=True)
    h.add_episode(0, n=2, pack=False)
    with pytest.raises(ValueError, match=r"빠짐 \['pack'\]"):
        h.run()
    assert not h.created


def test_a_pack_step_without_a_declared_pack_says_so(tmp_path):
    """check_pack_layout and write_pack are reached only for a declared pack (preflight_media,
    convert); a caller that gets there without one is told so, not handed a KeyError."""
    from generation.convert_lerobot import check_pack_layout, write_pack
    canon = _packless(_camera_canon())
    with pytest.raises(ValueError, match="팩 단계는 정본이 팩을 선언할 때만"):
        write_pack(tmp_path / "recorded.pack.npz", tmp_path / "out", 0, canon, steps=3)
    with pytest.raises(ValueError, match="팩 단계는 정본이 팩을 선언할 때만"):
        check_pack_layout({}, canon, steps=3)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("give", ["packs_without_cameras", "packs_for_a_packless_canon",
                                  "no_packs_for_a_pack_canon"])
def test_the_pack_index_goes_with_a_canon_that_declares_a_pack(tmp_path, give):
    """assemble_dataset writes meta/packs.json exactly when the camera canon it ships declares a
    pack: an index without a declaration, or a declaration without an index, is what C1 refuses."""
    import json

    from generation.convert_lerobot import assemble_dataset
    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    canon = _with_a_pack(_camera_canon())
    only_with_cameras = "packs 는 cameras 가 있을 때만 줍니다"
    exactly_when_declared = r"packs 는 정본\(meta/cameras.json\)이 팩을 선언할 때만, 그리고 그때는 반드시 줍니다"
    cameras, packs, said = {
        "packs_without_cameras": (None, {}, only_with_cameras),
        "packs_for_a_packless_canon": (_packless(canon), {}, exactly_when_declared),
        "no_packs_for_a_pack_canon": (canon, None, exactly_when_declared)}[give]
    out = _lerobot_output_stub(tmp_path)
    with pytest.raises(ValueError, match=said):
        assemble_dataset(repo_root, out, assets, ["S1d"], cameras=cameras, packs=packs,
                         referenced=S1D_ASSETS)
    assert not (out / "meta" / "cameras.json").exists() and not (out / "meta" / "packs.json").exists()


def test_the_emptied_frame_staging_does_not_ship(tmp_path):
    """LeRobot stages every picture as a PNG under images/ and removes each episode's directory
    once it is encoded, but leaves images/<video key>/ behind, empty. A real S1d conversion's output
    carried two such directories, and the fixture copies that layout."""
    h = _Harness(tmp_path)
    h.add_episode(0, n=2)
    h.add_episode(1, n=2)
    assert h.run() == 0
    assert h.dataset.staged, "the fake staged nothing -- this test would prove nothing"
    assert not (h.out / "images").exists()


def test_frames_lerobot_left_unencoded_stop_the_conversion(tmp_path):
    # A picture still under images/ after finalize() never reached a video: name it and stop.
    h = _Harness(tmp_path)
    h.keep_staged = True
    h.add_episode(0, n=2)
    left = 2 * len(cl.video_cameras())               # two steps of every camera's stream
    with pytest.raises(ValueError, match=rf"인코딩하지 않은 프레임 {left}개.*frame-000000\.png"):
        h.run()
    assert (h.out / "images").is_dir(), "what is left has to stay for someone to look at"


def test_the_conversion_ships_the_assets_its_episodes_use(tmp_path):
    """Through convert(): the referenced set is read off the episodes the batch actually
    converted, and assets/ holds exactly what the canon says those need."""
    import json
    h = _Harness(tmp_path)
    h.add_episode(0, n=2)
    h.add_episode(1, n=2)
    assert h.run() == 0
    canon = json.loads((h.repo / "meta" / "assets.json").read_text(encoding="utf-8"))
    assert _assets_tree(h.out) == _expected_files(canon, S1D_ASSETS)
    assert (h.out / "assets" / "textures").is_dir()


def test_a_v0_1_conversion_ships_no_textures(tmp_path):
    import json
    h = _Harness(tmp_path)
    h.add_episode(0, n=2, media=False, assets=V01_ASSETS, scenario="S1a")
    assert h.run() == 0
    canon = json.loads((h.repo / "meta" / "assets.json").read_text(encoding="utf-8"))
    assert _assets_tree(h.out) == _expected_files(canon, V01_ASSETS)
    robot = f"assets/{canon['robots']['unitree_h1']['ship']['dir']}/"
    assert [p for p in _assets_tree(h.out) if not p.startswith(robot)] == ["assets/" + Path(DUMMY).name]
    assert not (h.out / "assets" / "textures").exists()


def test_the_pre_flight_refuses_a_missing_texture_before_any_frame(tmp_path):
    """assemble_dataset would find it too -- after the last episode, with the conversion
    already paid for. The pre-flight is where the scenario and media checks already stand."""
    import json
    h = _Harness(tmp_path)
    h.add_episode(0, n=2)
    canon = json.loads((h.repo / "meta" / "assets.json").read_text(encoding="utf-8"))
    victim = _victim_texture(h.repo, canon)
    victim.unlink()
    with pytest.raises(FileNotFoundError, match=victim.name):
        h.run()
    assert not h.created and not h.out.exists()


def test_the_pre_flight_refuses_an_asset_the_canon_does_not_know(tmp_path):
    """build_episode_meta refuses it as well, but halfway through the batch -- so the unknown
    asset is in the second episode: refused there, the first would already have been written."""
    h = _Harness(tmp_path)
    h.add_episode(0, n=2, media=False, assets=V01_ASSETS, scenario="S1a")
    h.add_episode(1, n=2, media=False, assets=V01_ASSETS + ("generation/nowhere/unknown.usd",),
                  scenario="S1a")
    with pytest.raises(ValueError, match="meta/assets.json 에 없는 자산"):
        h.run()
    assert not h.created


@pytest.mark.parametrize("left_out", [AISLE, SKIN, NUCLEUS_H1, TERRAIN])
def test_a_sidecar_that_leaves_out_an_asset_its_scenario_used_is_refused(tmp_path, left_out):
    """What ships is drawn from scene_assets, so a sidecar missing the S1d aisle would take the
    aisle and its textures out of the distribution -- and the aisle reaches asset_paths only
    through getattr(scene cfg, "aisle", None) (run_s1a). The canon says what S1d used
    (scene.assets, human_model) and the sidecar's robot_model names its robot: refused before
    any frame, naming the episode and the path. The robot and the ground neither ship: the
    record of what was loaded, and under which licence, is what is at stake (both once
    passed, the ground because S1d's scene.assets had dropped it)."""
    import re
    h = _Harness(tmp_path)
    h.add_episode(0, n=2)
    h.add_episode(1, n=2, assets=tuple(a for a in S1D_ASSETS if a != left_out))
    with pytest.raises(ValueError, match=r"사이드카 1건의 asset_paths 가 .*S1d_0001_seed2: \['"
                                         + re.escape(left_out) + r"'\]"):
        h.run()
    assert not h.created and not h.out.exists()


def test_the_robot_in_an_s1a_sidecar_is_required_through_its_robot_model(tmp_path):
    # The v0.1 shape: no scenario names a robot; the sidecar's robot_model is a key of
    # meta/assets.json robots.
    import re
    h = _Harness(tmp_path)
    h.add_episode(0, n=2, media=False, assets=tuple(a for a in V01_ASSETS if a != NUCLEUS_H1),
                  scenario="S1a")
    with pytest.raises(ValueError, match=r"S1a_0000_seed1: \['" + re.escape(NUCLEUS_H1) + r"'\]"):
        h.run()
    assert not h.created


def test_the_v1_dummy_in_an_s1a_sidecar_is_required_through_the_human_model(tmp_path):
    # S1a's scene.assets lists the plane only; the dummy is required because the scenario's
    # human_model is the v1 dummy (meta/assets.json scene key).
    h = _Harness(tmp_path)
    h.add_episode(0, n=2, media=False, assets=tuple(a for a in V01_ASSETS if a != DUMMY),
                  scenario="S1a")
    with pytest.raises(ValueError, match=f"S1a_0000_seed1: \\['{DUMMY}'\\]"):
        h.run()
    assert not h.created


def test_a_copy_that_is_not_the_pinned_bytes_stops_the_assembly(tmp_path, monkeypatch):
    """The sources were hashed before copying; the copy is hashed again -- what ships is what
    the canon pins, not merely what was read a moment ago."""
    import json
    import shutil

    from generation.convert_lerobot import assemble_dataset

    repo_root = _repo_root_with_redistributables(tmp_path)
    assets = json.loads((repo_root / "meta" / "assets.json").read_text(encoding="utf-8"))
    real_copy = shutil.copy2

    def corrupting_copy(src, dst, *a, **k):
        out = real_copy(src, dst, *a, **k)
        if "/assets/" in str(dst):
            Path(dst).write_bytes(Path(dst).read_bytes() + b"flipped")
        return out
    monkeypatch.setattr(shutil, "copy2", corrupting_copy)
    out = _lerobot_output_stub(tmp_path)
    with pytest.raises(ValueError, match="복사한 파일의 해시가 정본과 다릅니다"):
        assemble_dataset(repo_root, out, assets, ["S1a"], referenced=V01_ASSETS)


def test_a_camera_less_batch_ships_no_camera_canon_and_no_packs(tmp_path):
    # A cameras.json in a v0.1 re-conversion would make C1 demand videos.
    h = _Harness(tmp_path)
    h.add_episode(0, media=False)
    assert h.run() == 0
    assert not (h.out / "meta" / "cameras.json").exists()
    assert not (h.out / "meta" / "packs.json").exists()
    assert not (h.out / "packs").exists()
    assert "image_writer_threads" not in h.created[0]
    assert not h.decoded


def test_a_mixed_batch_is_refused_before_any_frame_is_written(tmp_path):
    # LeRobot keeps one feature list per dataset.
    h = _Harness(tmp_path)
    h.add_episode(0, media=True)
    h.add_episode(1, media=False)
    with pytest.raises(ValueError, match="조건이 다릅니다"):
        h.run()
    assert not h.created and not h.out.exists()


def test_a_mixed_batch_is_refused_even_when_two_files_carry_one_episode_id(tmp_path):
    """The check is about files, not ids: keyed by episode_id, the camera-less copy below
    overwrote the camera episode's entry, the batch read as camera-less, and the conversion went
    on to write."""
    h = _Harness(tmp_path)
    eid, _ = h.add_episode(0, media=True)
    h.add_episode(0, media=False, stem=f"{eid}_copy")
    with pytest.raises(ValueError, match=f"조건이 다릅니다.*{eid}_copy\\.npz"):
        h.run()
    assert not h.created


@pytest.mark.parametrize("where", ["probe", "decode_short", "decode_long"])
def test_a_frame_count_that_is_not_the_step_count_stops(tmp_path, where):
    """프레임이 남거나 모자라는 것은 기록기의 결함이다 — 채우거나 자르지 않는다. 프로버가
    보면 아무것도 쓰기 전에, 디코드에서야 드러나면 에피소드를 저장하기 전에 멈춘다."""
    h = _Harness(tmp_path)
    h.add_episode(0, n=3)
    path = sorted(h.pictures)[0]
    if where == "probe":
        h.probe_frames[path] = 4
    elif where == "decode_short":
        h.pictures[path] = h.pictures[path][:2]
        h.probe_frames[path] = 3
    else:
        h.pictures[path] = h.pictures[path] + [h.pictures[path][0]]
        h.probe_frames[path] = 3
    with pytest.raises(ValueError, match="스텝"):
        h.run()
    if where == "probe":
        assert not h.created
    else:
        assert h.created and not h.dataset.episodes, "모자라거나 남는 영상으로 에피소드가 저장됐습니다"


def test_a_video_whose_size_is_not_the_canon_stops_before_writing(tmp_path):
    h = _Harness(tmp_path)
    h.add_episode(0, n=2)
    path = sorted(h.pictures)[0]
    h.pictures[path] = [p[:, :-2] for p in h.pictures[path]]
    with pytest.raises(ValueError, match="영상 크기"):
        h.run()
    assert not h.created


def test_a_canon_fps_edited_after_recording_stops_before_writing(tmp_path):
    """The recorder holds the canon's fps to the step period (camera_io.Rig); an edit made
    afterwards is seen by nothing else in the conversion -- the dataset's rate comes from the
    timestamps -- so the converter ran to the end and shipped a meta/cameras.json whose fps is
    not info.json's, which C1 refuses. Refused in the pre-flight instead, with nothing created.

    Not looked at: the videos' own container rate (probe_video returns size and frame count), and
    a camera-less batch (the canon is not read at all -- the last test below)."""
    import copy

    from generation.field_map import step_dt_from_timestamps
    h = _Harness(tmp_path)
    eid, _ = h.add_episode(0, n=3)                       # recorded at the canon's fps
    with np.load(h.episodes / f"{eid}.npz") as z:
        recorded = round(1.0 / step_dt_from_timestamps(z["timestamp"]))
    assert recorded == h.cameras["fps"], "the fixture no longer records at the canon's fps"
    doc = copy.deepcopy(h.cameras)
    doc["fps"] = recorded * 2                            # edited after recording
    h.write_cameras(doc)
    with pytest.raises(ValueError, match=rf"{eid}: 카메라 정본.*fps {doc['fps']} .*{recorded} fps"):
        h.run()
    assert not h.created and not h.out.exists()


def test_every_episode_is_held_to_the_canon_rate_not_just_the_first(tmp_path):
    """The second episode is the one recorded at another rate: the main loop would find it only
    after the first was written, and the message has to name the episode."""
    h = _Harness(tmp_path)
    h.add_episode(0, n=3)
    eid, _ = h.add_episode(1, n=3, period=2.0 / h.cameras["fps"])
    with pytest.raises(ValueError, match=rf"{eid}: 카메라 정본.*fps"):
        h.run()
    assert not h.created and not h.out.exists()


@pytest.mark.parametrize("bad", [0, -50, "50", True, None], ids=["zero", "negative", "text", "bool", "absent"])
def test_a_canon_fps_that_is_not_a_positive_number_stops_before_writing(tmp_path, bad):
    """Nothing to compare the recording with: refused by name rather than by a TypeError, and
    not shipped for C1 to refuse (a bool passes for the number 1 in Python, and 1 < 0.02 s)."""
    import copy
    h = _Harness(tmp_path)
    h.add_episode(0, n=3)
    doc = copy.deepcopy(h.cameras)
    if bad is None:
        del doc["fps"]
    else:
        doc["fps"] = bad
    h.write_cameras(doc)
    with pytest.raises(ValueError, match="카메라 정본.*fps 가 양수가 아닙니다"):
        h.run()
    assert not h.created and not h.out.exists()


def test_a_camera_less_batch_never_reads_the_canon_rate(tmp_path):
    # v0.1 conversions carry no camera canon; a broken fps in the repository's copy is not theirs.
    import copy
    h = _Harness(tmp_path)
    doc = copy.deepcopy(h.cameras)
    doc["fps"] = "not a rate"
    h.write_cameras(doc)
    h.add_episode(0, media=False)
    assert h.run() == 0
    assert not (h.out / "meta" / "cameras.json").exists()


@pytest.mark.parametrize("breakage", ["missing_stream", "unknown_member", "escapes",
                                      "frames", "no_file", "null_file"])
def test_a_malformed_media_entry_stops_before_writing(tmp_path, breakage):
    import json
    h = _Harness(tmp_path, pack=True)
    eid, meta = h.add_episode(0, n=2)
    media = meta["media"]
    if breakage == "missing_stream":
        del media["third_person_rgb"]
    elif breakage == "unknown_member":
        media["depth_video"] = {"file": media["pack"]["file"], "frames": 2}
    elif breakage == "escapes":
        # The file is there, so only the path check can refuse it.
        outside = h.tmp / media["pack"]["file"]
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_bytes((h.episodes / media["pack"]["file"]).read_bytes())
        media["pack"] = dict(media["pack"], file="../" + media["pack"]["file"])
    elif breakage == "frames":
        media["head_rgb"] = dict(media["head_rgb"], frames=3)
    elif breakage == "no_file":
        (h.episodes / media["pack"]["file"]).unlink()
    else:
        media["head_rgb"] = {"file": None, "frames": 0}
    (h.episodes / f"{eid}.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError), match="media"):
        h.run()
    assert not h.created


# ── Black frames ──────────────────────────────
#
# A camera inside the robot's own body renders black: the 2026-09-30 preview's wrist camera, placed
# inside the forearm, did. The recorder counts per camera the frames whose every pixel is 0
# (media.blank_frames); a camera that recorded nothing else stops the conversion before a frame is
# written, and one that was black for some frames goes through.

def _rewrite(h, eid, meta):
    import json
    (h.episodes / f"{eid}.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.mark.parametrize("cam", CANON_CAMERAS)
def test_a_camera_that_recorded_only_black_frames_stops_before_writing(tmp_path, cam):
    h = _Harness(tmp_path)
    eid, meta = h.add_episode(0, n=3)
    spec = h.cameras["cameras"][cam]
    h.pictures[h.episodes / meta["media"][cam]["file"]] = [
        np.zeros((spec["height"], spec["width"], 3), np.uint8) for _ in range(3)]
    meta["media"]["blank_frames"][cam] = 3
    _rewrite(h, eid, meta)
    # Any camera type: the message points at the canon's mount rules, not at the robot's body alone.
    with pytest.raises(ValueError, match=f"{eid}: 카메라 {cam} 의 프레임이 모두 검습니다.*meta/cameras.json 의 장착 규칙"):
        h.run()
    assert not h.created and not h.out.exists()


def test_cameras_black_in_some_frames_convert(tmp_path):
    h = _Harness(tmp_path)
    eid, meta = h.add_episode(0, n=3)
    meta["media"]["blank_frames"] = {cam: 2 for cam in meta["media"]["blank_frames"]}
    _rewrite(h, eid, meta)
    assert h.run() == 0
    assert len(h.dataset.episodes) == 1


@pytest.mark.parametrize("breakage", ["absent", "not_an_object", "a_camera_left_out", "an_unknown_camera",
                                      "bool", "negative", "more_than_the_frames", "float"])
def test_a_black_frame_count_that_cannot_be_read_stops_before_writing(tmp_path, breakage):
    """The count is what the refusal reads: without it -- a recording that did not count, made before
    the recorder counted black frames per camera -- or with one that is not a count per camera, the
    refusal could not run, and a camera left out of it would never be looked at."""
    h = _Harness(tmp_path)
    eid, meta = h.add_episode(0, n=3)
    blank = meta["media"]["blank_frames"]
    first = next(iter(blank))
    if breakage == "absent":
        del meta["media"]["blank_frames"]
    elif breakage == "not_an_object":
        meta["media"]["blank_frames"] = list(blank.values())
    elif breakage == "a_camera_left_out":
        del blank[first]
    elif breakage == "an_unknown_camera":
        blank["pelvis_rgb"] = 0
    else:
        blank[first] = {"bool": True, "negative": -1, "more_than_the_frames": 4, "float": 3.0}[breakage]
    _rewrite(h, eid, meta)
    with pytest.raises(ValueError, match=f"{eid}: .*media.blank_frames"):
        h.run()
    assert not h.created


# ── The recording's canon fingerprint ─────────────────────────
#
# The pixels of a recording were baked from the camera canon as it was (field of view, mount,
# resolution, channels, the box's camera). The recorder stamps media.canon_fingerprint -- what
# camera_canon.recording_fingerprint counts of that canon -- and the converter holds it to the
# canon it converts with, before a frame is written: a canon edited after recording would ship as
# the description of pictures it did not produce.

def _fingerprint_of(canon):
    return camera_canon.recording_fingerprint(canon)


def test_a_recording_made_with_another_camera_canon_is_refused(tmp_path):
    h = _Harness(tmp_path)
    eid, meta = h.add_episode(0, n=3)
    assert meta["media"]["canon_fingerprint"] == _fingerprint_of(h.cameras)     # the harness records honestly
    meta["media"]["canon_fingerprint"] = "0" * 64
    _rewrite(h, eid, meta)
    with pytest.raises(ValueError, match=f"{eid}: 녹화 때 카메라 정본과 지금 meta/cameras.json 이 다릅니다.*SOP §10.1"):
        h.run()
    assert not h.created and not h.out.exists()


def test_a_camera_recording_without_a_fingerprint_is_refused(tmp_path):
    # Made before the recorder stamped one: nothing to hold to the canon, so nothing
    # says the pictures belong to it. The old S1d recordings are these.
    h = _Harness(tmp_path)
    eid, meta = h.add_episode(0, n=3)
    del meta["media"]["canon_fingerprint"]
    _rewrite(h, eid, meta)
    with pytest.raises(ValueError, match=f"{eid}: 녹화가 지문을 남기지 않았습니다.*지문을 남기기 전의 녹화.*재생성"):
        h.run()
    assert not h.created and not h.out.exists()


@pytest.mark.parametrize("bad", [None, 0, True, ["0" * 64], {"sha256": "0" * 64}, "", "0" * 63, "0" * 65,
                                 "A" * 64, "g" * 64],
                         ids=["null", "zero", "bool", "list", "object", "empty", "short", "long", "upper", "not_hex"])
def test_a_fingerprint_that_is_not_a_sha256_is_refused_not_compared(tmp_path, bad):
    # Fail closed: a value the recorder cannot have written is a broken sidecar, not "another canon".
    h = _Harness(tmp_path)
    eid, meta = h.add_episode(0, n=3)
    meta["media"]["canon_fingerprint"] = bad
    _rewrite(h, eid, meta)
    with pytest.raises(ValueError, match=f"{eid}: media.canon_fingerprint 가 sha256"):
        h.run()
    assert not h.created and not h.out.exists()


def test_a_note_edited_after_recording_does_not_stop_the_conversion(tmp_path):
    # Prose and the converter's derived numbers are not what the pixels were baked from
    # (camera_canon.recording_fingerprint): a wording fix in meta/cameras.json is no reason to record
    # 400 episodes again.
    import copy
    h = _Harness(tmp_path)
    h.add_episode(0, n=3)
    doc = copy.deepcopy(h.cameras)
    doc["description"] += " (reworded)"
    doc["intrinsics_note"] += " (reworded)"
    doc["cameras"]["head_rgb"]["mount_note"] += " (reworded)"
    doc["bbox"]["source"] += " (reworded)"
    assert _fingerprint_of(doc) == _fingerprint_of(h.cameras)       # the premise: only prose changed
    assert doc != h.cameras
    h.write_cameras(doc)
    assert h.run() == 0
    assert len(h.dataset.episodes) == 1


@pytest.mark.parametrize("path,edit", [
    (("cameras", "head_rgb", "fov_h_deg"), lambda v: v + 5.0),
    (("cameras", "head_rgb", "width"), lambda v: v + 16),
    (("cameras", "left_wrist_rgb", "offset_pos"), lambda v: [v[0] + 0.01, *v[1:]]),
    (("cameras", "cctv_rgb", "sites", "S1d", "mount", "height_m"), lambda v: v + 0.5),
    (("cameras", "cctv_rgb", "sites", "S1g", "by_approach_side", "west", "aim_m"), lambda v: [v[0] + 0.5, *v[1:]]),
    (("cameras", "third_person_rgb", "sites", "S1d", "eye_offset"), lambda v: [v[0] - 1.0, *v[1:]]),
    (("cameras", "third_person_rgb", "sites", "S1e", "frame"), lambda v: "world"),
    (("cameras", "cctv_rgb", "channels"), lambda v: v + ["semantic_segmentation"]),
    (("bbox", "classes_union"), lambda v: v[:-1]),
    (("render", "reset_warm_renders"), lambda v: v + 1),
], ids=lambda x: ".".join(x) if isinstance(x, tuple) else "edit")
def test_a_pixel_value_edited_after_recording_stops_before_writing(tmp_path, path, edit):
    """The field of view, the mount, the resolution and the box's camera edited in
    meta/cameras.json after the episodes were recorded. Nothing else in the conversion reads most of
    these (a mount is read by the recorder alone; a wider resolution the pre-flight would meet as a
    stream of another size, and now names as the canon having changed), so the fingerprint is what
    stops it -- at the pre-flight, with no dataset created. (An fps edit is check_canon_rate's, which
    speaks first: test_a_canon_fps_edited_after_recording_stops_before_writing.)"""
    import copy
    h = _Harness(tmp_path)
    eid, _ = h.add_episode(0, n=3)
    doc = copy.deepcopy(h.cameras)
    node = doc
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = edit(node[path[-1]])
    assert _fingerprint_of(doc) != _fingerprint_of(h.cameras)
    h.write_cameras(doc)
    with pytest.raises(ValueError, match=f"{eid}: 녹화 때 카메라 정본과 지금 meta/cameras.json 이 다릅니다"):
        h.run()
    assert not h.created and not h.out.exists()


def test_the_box_camera_moved_after_recording_stops_before_writing(tmp_path):
    """bbox.camera is the camera whose segmentation the box was computed from: moved to another (one
    that records a segmentation, so the canon still loads) the recorded box is another camera's picture
    of the dummy, and only the fingerprint says so."""
    import copy
    h = _Harness(tmp_path)
    eid, _ = h.add_episode(0, n=3)
    doc = copy.deepcopy(h.cameras)
    doc["cameras"]["cctv_rgb"]["channels"] = ["rgb", "semantic_segmentation"]
    doc["bbox"]["camera"] = "cctv_rgb"
    h.write_cameras(doc)
    with pytest.raises(ValueError, match=f"{eid}: 녹화 때 카메라 정본과 지금 meta/cameras.json 이 다릅니다"):
        h.run()
    assert not h.created and not h.out.exists()


@pytest.mark.parametrize("breakage", ["dtype", "extra_array", "resolution"])
def test_a_pack_that_is_not_the_canon_format_stops_before_writing(tmp_path, breakage):
    h = _Harness(tmp_path, pack=True)
    _, meta = h.add_episode(0, n=2)
    path = h.episodes / meta["media"]["pack"]["file"]
    with np.load(path) as z:
        pack = {k: z[k] for k in z.files}
    if breakage == "dtype":
        pack["depth"] = pack["depth"].astype(np.float32)
    elif breakage == "extra_array":
        pack["rgb"] = pack["mask"]
    else:
        pack["depth"], pack["mask"] = pack["depth"][:, ::2], pack["mask"][:, ::2]
    np.savez_compressed(path, **pack)
    with pytest.raises(ValueError, match="팩"):
        h.run()
    assert not h.created


@pytest.mark.parametrize("step_index", [[0, 0, 2], [1, 2, 3]], ids=["repeated", "offset"])
def test_a_pack_whose_rows_are_not_the_recorded_steps_stops_before_writing(tmp_path, step_index):
    """Pack row i is parquet row i. The layout (length, integer dtype) is right here and only the
    values are wrong, so only the alignment check can refuse it -- in the pre-flight, not after an
    episode has been saved (a bad pack in the last episode of a long batch cost the whole run)."""
    h = _Harness(tmp_path, pack=True)
    _, meta = h.add_episode(0, n=3)
    path = h.episodes / meta["media"]["pack"]["file"]
    with np.load(path) as z:
        pack = {k: z[k] for k in z.files}
    pack["step_index"] = np.asarray(step_index, dtype=pack["step_index"].dtype)
    np.savez_compressed(path, **pack)
    with pytest.raises(ValueError, match="step_index 가 기록된 스텝"):
        h.run()
    assert not h.created


def test_a_canon_downsample_that_is_not_a_whole_fraction_stops_before_writing(tmp_path):
    # Refused in the pre-flight; left to write_pack, it would come after the first episode's save.
    import copy
    h = _Harness(tmp_path, pack=True)
    doc = copy.deepcopy(h.cameras)
    name = next(iter(doc["packs"]))
    cam = doc["cameras"][doc["packs"][name]["camera"]]
    doc["packs"][name]["downsample"] = {"width": cam["width"] // 2 + 1, "height": cam["height"],
                                        "every_n_steps": 1}
    h.write_cameras(doc)
    h.add_episode(0, n=2)
    with pytest.raises(ValueError, match="downsample.width"):
        h.run()
    assert not h.created


def test_an_identity_pack_is_copied_without_being_inflated(tmp_path):
    """At the identity downsample the pre-flight has already checked the pack, and write_pack only
    copies and hashes it -- inflating ~0.9 GB per episode to copy the same bytes cost 1.6 s each.
    Bytes that are not an npz at all prove nothing reads them."""
    import hashlib

    from generation.convert_lerobot import write_pack
    cameras = _with_a_pack(_camera_canon())
    name = next(iter(cameras["packs"]))
    cam = cameras["cameras"][cameras["packs"][name]["camera"]]
    cameras["packs"][name]["downsample"] = {"width": cam["width"], "height": cam["height"],
                                            "every_n_steps": 1}
    src = tmp_path / "recorded.pack.npz"
    src.write_bytes(b"not an npz -- only copied")
    entry = write_pack(src, tmp_path / "out", 7, cameras, steps=5)
    shipped = tmp_path / "out" / entry["file"]
    assert entry["file"] == f"packs/{name}/episode_000007.npz"
    assert shipped.read_bytes() == src.read_bytes()
    assert entry == {"file": entry["file"], "frames": 5,
                     "sha256": hashlib.sha256(src.read_bytes()).hexdigest()}


def test_the_pack_downsample_picks_pixels_and_keeps_the_recorded_step_numbers():
    from generation.convert_lerobot import downsample_pack
    camera = {"width": 8, "height": 6}
    n = 5
    depth = np.arange(n * 6 * 8, dtype=np.uint16).reshape(n, 6, 8)
    mask = (depth % 15).astype(np.uint8)
    pack = {"depth": depth, "mask": mask, "step_index": np.arange(n, dtype=np.int32)}

    same = downsample_pack(pack, camera, {"width": 8, "height": 6, "every_n_steps": 1})
    for k in pack:
        assert np.array_equal(same[k], pack[k]) and same[k].dtype == pack[k].dtype

    half = downsample_pack(pack, camera, {"width": 4, "height": 3, "every_n_steps": 2})
    assert half["depth"].shape == (3, 3, 4) and half["depth"].dtype == np.uint16
    assert half["step_index"].tolist() == [0, 2, 4]          # arange(0, n, k), not renumbered
    assert np.array_equal(half["depth"], depth[[0, 2, 4]][:, ::2, ::2])   # picked, not averaged
    assert np.array_equal(half["mask"], mask[[0, 2, 4]][:, ::2, ::2])


def test_the_pack_downsample_keeps_rows_and_columns_apart():
    """Unequal strides: height 6 -> 2 keeps every third row, width 8 -> 4 every second column.
    With 2x2 the two strides are interchangeable and a swap passes unseen."""
    from generation.convert_lerobot import downsample_pack
    camera = {"width": 8, "height": 6}
    n = 4
    depth = np.arange(n * 6 * 8, dtype=np.uint16).reshape(n, 6, 8)
    mask = (depth % 15).astype(np.uint8)
    pack = {"depth": depth, "mask": mask, "step_index": np.arange(n, dtype=np.int32)}
    out = downsample_pack(pack, camera, {"width": 4, "height": 2, "every_n_steps": 3})
    assert out["depth"].shape == (2, 2, 4) and out["mask"].shape == (2, 2, 4)
    assert np.array_equal(out["depth"], depth[[0, 3]][:, ::3, ::2])
    assert np.array_equal(out["mask"], mask[[0, 3]][:, ::3, ::2])
    assert out["step_index"].tolist() == [0, 3]


def test_a_downsample_that_lands_on_another_shape_is_refused(monkeypatch):
    # The postcondition holds even if the factors come back in the wrong order.
    import generation.convert_lerobot as conv
    monkeypatch.setattr(conv, "downsample_factors", lambda camera, downsample: (1, 2, 3))
    pack = {"depth": np.zeros((2, 6, 8), np.uint16), "mask": np.zeros((2, 6, 8), np.uint8),
            "step_index": np.arange(2, dtype=np.int32)}
    with pytest.raises(ValueError, match="축소한 팩 depth 의 모양"):
        conv.downsample_pack(pack, {"width": 8, "height": 6},
                             {"width": 4, "height": 2, "every_n_steps": 1})


@pytest.mark.parametrize("ds", [{"width": 5, "height": 6, "every_n_steps": 1},
                                {"width": 8, "height": 4, "every_n_steps": 1},
                                {"width": 8, "height": 6, "every_n_steps": 0},
                                {"width": 8, "height": 6, "every_n_steps": 1.5}],
                         ids=["width", "height", "zero_step", "fractional_step"])
def test_a_downsample_that_is_not_a_whole_fraction_is_refused(ds):
    from generation.convert_lerobot import downsample_pack
    pack = {"depth": np.zeros((2, 6, 8), np.uint16), "mask": np.zeros((2, 6, 8), np.uint8),
            "step_index": np.arange(2, dtype=np.int32)}
    with pytest.raises(ValueError, match="downsample"):
        downsample_pack(pack, {"width": 8, "height": 6}, ds)


def test_the_canon_downsample_is_applied_to_the_shipped_pack(tmp_path):
    import copy
    import json
    h = _Harness(tmp_path, pack=True)
    doc = copy.deepcopy(h.cameras)
    name = next(iter(doc["packs"]))
    cam = doc["cameras"][doc["packs"][name]["camera"]]
    # Unequal strides (rows 4, columns 2), so a swap cannot pass.
    doc["packs"][name]["downsample"] = {"width": cam["width"] // 2, "height": cam["height"] // 4,
                                        "every_n_steps": 2}
    h.write_cameras(doc)
    _, meta = h.add_episode(0, n=5)
    assert h.run() == 0
    with np.load(h.episodes / meta["media"]["pack"]["file"]) as z:
        src = {k: z[k] for k in z.files}
    entry = json.loads((h.out / "meta" / "packs.json").read_text(encoding="utf-8"))["0"]
    assert entry["frames"] == 3
    with np.load(h.out / entry["file"]) as z:
        assert z["step_index"].tolist() == [0, 2, 4]
        assert z["depth"].shape == (3, cam["height"] // 4, cam["width"] // 2)
        assert np.array_equal(z["depth"], src["depth"][::2, ::4, ::2])
        assert np.array_equal(z["mask"], src["mask"][::2, ::4, ::2])


def test_the_converter_writes_every_notes_member_of_the_canon():
    # A member the schema declares and the converter leaves out is a C5 ERROR on every episode of
    # a real conversion; the fixture writes its notes by hand and would not show it.
    from generation.labeling.label_episodes import label_episode
    from tests.generation.release_fixture import (ASSETS, FPS, JOINT_MAP, SCENARIO, SCENARIOS,
                                                  canon_rules, recording)
    arrays, meta = recording(0)
    rules = canon_rules()
    record = cl.build_episode_meta(meta, label_episode(arrays, meta, rules, SCENARIO), SCENARIOS,
                                   JOINT_MAP, ASSETS, rules, step_dt=1.0 / FPS)
    schema = json.loads((Path(__file__).resolve().parents[2] / "meta" / "safety_schema.json")
                        .read_text(encoding="utf-8"))
    assert set(json.loads(record["notes"])) == set(schema["notes_members"])


_HRI_SIDECAR_NOTES = ("degenerate_reason", "box_released_step", "camera_in_box_steps",
                      "task_stages", "human_stature_m", "task_time_limit_s")


def _notes_of(arrays, meta, scenario):
    from generation.labeling.label_episodes import label_episode
    from tests.generation.release_fixture import ASSETS, FPS, JOINT_MAP, canon_rules
    rules = canon_rules()
    record = cl.build_episode_meta(meta, label_episode(arrays, meta, rules, scenario), {"scenarios": [scenario]},
                                   JOINT_MAP, ASSETS, rules, step_dt=1.0 / FPS)
    return json.loads(record["notes"])


def test_hri_notes_carry_the_three_sidecar_values():
    """The hri recorder writes three values to the sidecar that only the sidecar held: why the person was
    unusable, the step the held box was let go, and how long each robot camera sat inside it. The published
    notes carry each as the sidecar has it."""
    from tests.generation.release_fixture import HRI_SCENARIO, SCENARIO, hri_recording, recording
    arrays, meta = hri_recording(0, touch=(120, "BR-05", 100.0, "transient"))
    # what the fixture's own sidecar says reaches the notes ...
    notes = _notes_of(arrays, meta, HRI_SCENARIO)
    assert {name: notes[name] for name in _HRI_SIDECAR_NOTES} == {name: meta[name] for name in _HRI_SIDECAR_NOTES}
    # ... and so does a sidecar with a value of its own for each (a copy that fell back to a default would show)
    cameras = {cam: i for i, cam in enumerate(meta["camera_in_box_steps"])}
    own = dict(meta, degenerate_reason="사람이 첫 접촉 전에 넘어짐", box_released_step=312, camera_in_box_steps=cameras)
    notes = _notes_of(arrays, own, HRI_SCENARIO)
    assert notes["degenerate_reason"] == "사람이 첫 접촉 전에 넘어짐"
    assert notes["box_released_step"] == 312
    assert notes["camera_in_box_steps"] == cameras and len(cameras) == 5 and any(cameras.values())
    stages = [{"name": "enter", "completed": True, "end_step": 60, "duration_s": 1.2},
              {"name": "turn", "completed": False, "end_step": -1, "duration_s": None}]
    notes = _notes_of(arrays, dict(own, task_stages=stages, human_stature_m=1.81, task_time_limit_s=17.5),
                      HRI_SCENARIO)
    assert notes["task_stages"] == stages
    assert notes["human_stature_m"] == 1.81 and notes["task_time_limit_s"] == 17.5
    # a recording of a scenario with no script keeps all three null
    arrays, meta = recording(0)
    notes = _notes_of(arrays, meta, SCENARIO)
    assert {name: notes[name] for name in _HRI_SIDECAR_NOTES} == dict.fromkeys(_HRI_SIDECAR_NOTES)


def _hri_arrays(n=3):
    from generation.field_map import HRI
    shapes = json.loads((Path(__file__).resolve().parents[2] / "meta" / "safety_schema.json")
                        .read_text(encoding="utf-8"))["fixed_shapes"]
    return {f.npz: (np.zeros((n, shapes[f.schema])) if f.schema in shapes else np.zeros(n))
            for f in cl.STEP_FIELDS if f.condition == HRI}


def test_an_episode_with_the_hri_arrays_carries_the_hri_fields():
    from generation.field_map import HRI
    arrays = {**_arrays(), **_hri_arrays()}
    assert cl.active_conditions(arrays, _meta()) == (HRI,)
    frame = cl.step_frames(arrays, _meta(), {"dof": 2})[0]
    assert {f.schema for f in cl.STEP_FIELDS if f.condition == HRI} <= set(frame)
    assert len(frame["observation.human.root_pose"]) == 7 and frame["safety.approach_speed_robot"] == 0.0


def test_the_walking_command_of_an_old_recording_switches_nothing_on():
    assert cl.active_conditions({**_arrays(), "command_velocity": np.zeros((3, 3))}, _meta()) == ()


def test_part_of_the_hri_arrays_is_refused_before_a_frame():
    arrays = {**_arrays(), **_hri_arrays()}
    del arrays["human_root_vel"]
    with pytest.raises(ValueError, match="human_root_vel"):
        cl.active_conditions(arrays, _meta())


def test_a_batch_mixing_hri_and_plain_episodes_is_refused():
    from generation.field_map import HRI
    with pytest.raises(ValueError, match="조건 없음"):
        cl.batch_conditions({Path("a.npz"): (HRI,), Path("b.npz"): ()})


_HRI_SCENARIO = {"scenario_id": "S1e", "rules": ["R-TS15066-A2-*", "R-TS15066-SSM-02", "R-HRI-CONTACT-01"]}
_OLD_SCENARIO = {"scenario_id": "S1a", "rules": ["R-TS15066-A2-*", "R-TS15066-SSM-01", "R-10218-STOP-01"]}


def _recorded(tmp_path, name, arrays, scenario_id, **sidecar):
    np.savez(tmp_path / f"{name}.npz", **arrays)
    (tmp_path / f"{name}.json").write_text(
        json.dumps(dict(_meta(), scenario_id=scenario_id, **sidecar)), encoding="utf-8")
    return tmp_path / f"{name}.npz"


def test_check_hri_declared_passes_a_batch_whose_recordings_match_their_scenarios(tmp_path):
    paths = [_recorded(tmp_path, "a", {**_arrays(), **_hri_arrays()}, "S1e"),
             _recorded(tmp_path, "b", _arrays(), "S1a")]
    assert cl.check_hri_declared(paths, {"S1e": _HRI_SCENARIO, "S1a": _OLD_SCENARIO}) is None


@pytest.mark.parametrize("hri, sid, says", [
    (False, "S1e", "선언하는데 npz 에 hri 배열이 없습니다"),
    (True, "S1a", "선언하지 않는데 npz 에 hri 배열이 있습니다"),
])
def test_check_hri_declared_refuses_a_recording_its_scenario_does_not_match(tmp_path, hri, sid, says):
    path = _recorded(tmp_path, "e", {**_arrays(), **(_hri_arrays() if hri else {})}, sid)
    with pytest.raises(ValueError, match=says):
        cl.check_hri_declared([path], {"S1e": _HRI_SCENARIO, "S1a": _OLD_SCENARIO})


def test_check_hri_declared_names_the_file_and_the_scenario_of_half_an_hri_recording(tmp_path):
    """Half the hri arrays did not come out of one save: the refusal says which recording and of which scenario, so
    the episode can be found among the batch's."""
    arrays = {**_arrays(), **_hri_arrays()}
    del arrays["approach_speed_human"]
    paths = [_recorded(tmp_path, "good", {**_arrays(), **_hri_arrays()}, "S1e"),
             _recorded(tmp_path, "half", arrays, "S1f")]
    with pytest.raises(ValueError, match=r"half\.npz: 시나리오 S1f — .*일부만.*approach_speed_human"):
        cl.check_hri_declared(paths, {"S1e": _HRI_SCENARIO, "S1f": dict(_HRI_SCENARIO, scenario_id="S1f")})


# What the hri recorder's sidecar says of the task besides the step it was done at.
_TASK_RECORD = {"task_stages": [{"name": "enter", "completed": True, "end_step": 1, "duration_s": 0.02}],
                "human_stature_m": 1.74, "task_time_limit_s": 20.0}


@pytest.mark.parametrize("value", [-1, 0, 2])
def test_check_hri_completion_passes_a_recorded_step_or_the_not_done_mark(tmp_path, value):
    # _arrays() records three steps: 0, 1 and 2
    path = _recorded(tmp_path, "e", {**_arrays(), **_hri_arrays()}, "S1e", task_completed_step=value,
                     **_TASK_RECORD)
    assert cl.check_hri_completion([path], {"S1e": _HRI_SCENARIO, "S1a": _OLD_SCENARIO}) is None


@pytest.mark.parametrize("value", [3, 150])
def test_check_hri_completion_refuses_a_step_the_recording_does_not_have(tmp_path, value):
    """A step past the recording's last would count as a completion the episode never reached."""
    path = _recorded(tmp_path, "e", {**_arrays(), **_hri_arrays()}, "S1e", task_completed_step=value,
                     **_TASK_RECORD)
    with pytest.raises(ValueError, match=f"e.npz.*{value}.*3개"):
        cl.check_hri_completion([path], {"S1e": _HRI_SCENARIO, "S1a": _OLD_SCENARIO})


@pytest.mark.parametrize("sidecar", [{}, {"task_completed_step": None}, {"task_completed_step": 7}])
def test_check_hri_completion_leaves_a_scenario_without_the_rule_alone(tmp_path, sidecar):
    """The plain scenarios' recordings carry no completion criterion; null stays legal there."""
    path = _recorded(tmp_path, "e", _arrays(), "S1a", **sidecar)
    assert cl.check_hri_completion([path], {"S1e": _HRI_SCENARIO, "S1a": _OLD_SCENARIO}) is None


_ABSENT = object()


@pytest.mark.parametrize("value", [_ABSENT, None, True, 3.0, "5", [5], -2],
                         ids=["absent", "null", "bool", "float", "string", "list", "below the mark"])
def test_check_hri_completion_refuses_a_recording_that_names_no_completion_step(tmp_path, value):
    """Every scenario that declares the directed rule has a completion criterion: a null (or no
    member at all) would ship as 'no criterion' and the scorecard would leave the episode out of
    the completion rate."""
    sidecar = {} if value is _ABSENT else {"task_completed_step": value}
    path = _recorded(tmp_path, "e", {**_arrays(), **_hri_arrays()}, "S1e", **sidecar)
    with pytest.raises(ValueError, match="e.npz.*S1e.*task_completed_step"):
        cl.check_hri_completion([path], {"S1e": _HRI_SCENARIO, "S1a": _OLD_SCENARIO})


@pytest.mark.parametrize("key", list(_TASK_RECORD))
@pytest.mark.parametrize("how", ["absent", "null"])
def test_check_hri_completion_refuses_a_recording_that_names_no_task_record(tmp_path, key, how):
    """The stages, the person's stature and the time limit belong to the same criterion as the step: a sidecar
    that lacks one (or holds null) would ship it as "no script", and the refusal names the file and the key."""
    record = {k: v for k, v in _TASK_RECORD.items() if not (k == key and how == "absent")}
    if how == "null":
        record[key] = None
    path = _recorded(tmp_path, "e", {**_arrays(), **_hri_arrays()}, "S1e", task_completed_step=2, **record)
    with pytest.raises(ValueError, match=f"e.npz.*S1e.*{key}"):
        cl.check_hri_completion([path], {"S1e": _HRI_SCENARIO, "S1a": _OLD_SCENARIO})


def test_the_conversion_refuses_an_hri_recording_without_a_completion_step_before_a_frame(tmp_path):
    h = _Harness(tmp_path)
    eid, _ = h.add_episode(0, media=False, scenario="S1e", assets=V01_ASSETS)
    _give_hri_arrays(h, eid)
    with pytest.raises(ValueError, match="task_completed_step"):
        h.run()
    assert h.created == [], "거부하기 전에 배포본을 만들기 시작했습니다"


def _give_hri_arrays(h, eid):
    path = h.episodes / f"{eid}.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    np.savez(path, **arrays, **_hri_arrays(n=len(arrays["timestamp"])))


@pytest.mark.parametrize("scenario, record_hri, says", [
    ("S1e", False, "선언하는데 npz 에 hri 배열이 없습니다"),
    ("S1a", True, "선언하지 않는데 npz 에 hri 배열이 있습니다"),
])
def test_the_conversion_refuses_a_recording_its_scenario_does_not_match_before_a_frame(
        tmp_path, scenario, record_hri, says):
    h = _Harness(tmp_path)
    eid, _ = h.add_episode(0, media=False, scenario=scenario, assets=V01_ASSETS)
    if record_hri:
        _give_hri_arrays(h, eid)
    with pytest.raises(ValueError, match=says):
        h.run()
    assert h.created == [], "거부하기 전에 배포본을 만들기 시작했습니다"


# ── a scenario with a conditions table: its spec, its evidence and its place in the table (schema §5) ──────────────

def _table_fields():
    from generation.field_map import TABLE
    return [f.schema for f in EPISODE_FIELDS if f.condition == TABLE]


def _table_record(recording, cameras=None):
    """build_episode_meta of one table-scenario recording (release_fixture.table_recording) and its label."""
    from generation.labeling.label_episodes import label_episode
    from tests.generation.release_fixture import FPS, JOINT_MAP, TABLE_ASSETS, TABLE_SCENARIO, canon_rules
    arrays, meta = recording
    rules = canon_rules()
    label = label_episode(arrays, meta, rules, TABLE_SCENARIO, cameras)
    record = cl.build_episode_meta(meta, label, {"scenarios": [TABLE_SCENARIO]}, JOINT_MAP, TABLE_ASSETS, rules,
                                   step_dt=1.0 / FPS)
    return record, meta, label


def test_a_plain_scenarios_record_has_none_of_the_table_fields():
    from tests.generation.release_fixture import HRI_SCENARIO, hri_recording
    assert not set(_build()) & set(_table_fields())
    # An hri scenario without a conditions table has none either.
    record = _notes_record(*hri_recording(0), HRI_SCENARIO)
    assert not set(record) & set(_table_fields())


def _notes_record(arrays, meta, scenario):
    from generation.labeling.label_episodes import label_episode
    from tests.generation.release_fixture import ASSETS, FPS, JOINT_MAP, canon_rules
    rules = canon_rules()
    return cl.build_episode_meta(meta, label_episode(arrays, meta, rules, scenario), {"scenarios": [scenario]},
                                 JOINT_MAP, ASSETS, rules, step_dt=1.0 / FPS)


def test_every_table_field_is_in_a_table_scenarios_record_and_none_is_none():
    import math

    from generation.episode_spec import SPEC_MEMBERS
    from generation.labeling import evidence
    from tests.generation.release_fixture import table_recording
    for recording in (table_recording(0, 0), table_recording(1, 57, touch=(120, "BR-05", 300.0, "transient"))):
        record, _, label = _table_record(recording)
        assert [f for f in _table_fields() if f in record] == _table_fields()
        assert [f for f in _table_fields() if record[f] is None] == []
        assert tuple(record["episode_spec"]) == SPEC_MEMBERS
        for name in evidence.NAMES:
            got, want = record[evidence.PREFIX + name], label[evidence.EVIDENCE_KEY][name]
            assert type(got) is float and (math.isnan(got) if want is None else got == want), (name, got, want)
    untouched, _, _ = _table_record(table_recording(0, 0))
    assert math.isnan(untouched["evidence.first_contact_s"])


def test_the_place_in_the_table_is_the_condition_records():
    from generation import conditions
    from tests.generation.release_fixture import TABLE_SCENARIO, table_recording
    for row, want in ((0, ("random", -1, -1)), (57, ("grid", 0, 1)), (98, ("grid", 6, 6))):
        record, meta, _ = _table_record(table_recording(row, row))
        got = (record["condition_design"], record["grid_cell_robot"], record["grid_cell_human"])
        assert got == want
        design, cell = conditions.row_design(TABLE_SCENARIO, row)
        condition = json.loads(record["notes"])["randomization"]["condition"]
        assert (condition["design"], condition["grid_cell"]) == (design, None if cell is None else list(cell))
        assert got == (design, *(cell if cell is not None else (cl.GRID_NOT_A_CELL, cl.GRID_NOT_A_CELL)))
        assert all(type(v) is int for v in got[1:])


def test_the_designs_a_table_has_are_the_canon_enum():
    from generation import conditions
    from tests.generation.release_fixture import TABLE_SCENARIO
    spec = conditions.load_conditions(TABLE_SCENARIO)
    designs = {conditions.row_design(TABLE_SCENARIO, i)[0] for i in range(spec.rows + spec.grid.cells ** 2)}
    assert designs == set(_canon_schema()["enums"]["condition_design"])


def test_the_validator_holds_the_converters_cell_of_a_random_row():
    """The validator does not import generation/: it holds a copy of the cell number a random row's two columns
    carry, and C5 refuses a random row whose cells are not that."""
    from validator import validate_dataset
    assert validate_dataset.GRID_NOT_A_CELL == cl.GRID_NOT_A_CELL


def test_the_spec_column_is_the_episode_spec_of_the_sidecar():
    from generation.episode_spec import build_episode_spec
    from tests.generation.release_fixture import JOINT_MAP, TABLE_ASSETS, TABLE_SCENARIO, table_recording
    record, meta, _ = _table_record(table_recording(3, 50))
    assert record["episode_spec"] == build_episode_spec(meta, TABLE_SCENARIO, TABLE_ASSETS, dof=JOINT_MAP["dof"])
    assert json.loads(cl.episode_column(record["episode_spec"])) == record["episode_spec"]


def _written_table(root, recordings):
    """The episode table merge_episode_metadata writes for ``recordings``: one LeRobot file to join onto."""
    records = [_table_record(r)[0] for r in recordings]
    _meta_root(root, [[{"episode_index": i, "length": 200} for i in range(len(records))]])
    return cl.merge_episode_metadata(root, records)


def test_a_random_rows_table_and_a_grid_rows_table_type_the_table_columns_alike(tmp_path):
    """Two chunks' episode tables, one of random rows only and one of grid rows only, written as the converter writes
    them: the table columns are the same Arrow type in both -- a column whose every row is empty included -- or the
    two cannot be one release. A value that is not there is a NaN, which pandas writes as a double's null; written as
    None, a column of none of them would be Arrow's null type."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tests.generation.release_fixture import table_recording
    random = _written_table(tmp_path / "random", [table_recording(0, 0), table_recording(1, 1)])
    grid = _written_table(tmp_path / "grid", [table_recording(2, 50, touch=(120, "BR-05", 300.0, "transient")),
                                              table_recording(3, 57, touch=(120, "BR-01", 3.0, "transient"))])
    types = {name: {f: str(pq.read_schema(path).field(f).type) for f in _table_fields()}
             for name, path in (("random", random), ("grid", grid))}
    assert types["random"] == types["grid"]
    assert not [f for f in _table_fields() if pa.types.is_null(pq.read_schema(random).field(f).type)]
    assert pq.read_schema(random).field("evidence.first_contact_s").type == pa.float64()
    untouched = pq.read_table(random).column("evidence.first_contact_s").to_pylist()
    assert untouched == [None, None]
    assert pq.read_table(grid).column("evidence.force_ratio.BR-01").to_pylist() == [0.0, float("inf")]


def _table_on_disk(tmp_path, recording, label=None, name=None):
    """Write one recording as the converter reads it: <id>.npz, <id>.json and <id>.labels.json."""
    from generation.labeling.label_episodes import label_episode
    from tests.generation.release_fixture import TABLE_SCENARIO, canon_rules
    arrays, meta = recording
    stem = name or meta["episode_id"]
    np.savez(tmp_path / f"{stem}.npz", **arrays)
    (tmp_path / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")
    label = label if label is not None else label_episode(arrays, meta, canon_rules(), TABLE_SCENARIO)
    (tmp_path / f"{stem}.labels.json").write_text(json.dumps(label), encoding="utf-8")
    return tmp_path / f"{stem}.npz"


def _by_sid():
    from tests.generation.release_fixture import SCENARIO, TABLE_SCENARIO
    return {"S1e": TABLE_SCENARIO, "S1d": SCENARIO}


def _check_table_recordings(paths):
    """check_table_recordings with the canon the conversion reads: the repository's assets and joint maps."""
    from tests.generation.release_fixture import TABLE_ASSETS
    return cl.check_table_recordings(paths, _by_sid(), TABLE_ASSETS, REPO_ROOT)


def test_check_table_recordings_passes_a_table_batch_and_leaves_a_plain_one_alone(tmp_path):
    from tests.generation.release_fixture import recording, table_recording
    paths = [_table_on_disk(tmp_path, table_recording(0, 0)), _table_on_disk(tmp_path, table_recording(1, 57))]
    arrays, meta = recording(2)
    paths.append(_table_on_disk(tmp_path, (arrays, meta), label=_label()))       # S1d: no table, no evidence
    assert _check_table_recordings(paths) is None


def _without(mapping, *path):
    node = mapping
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]


@pytest.mark.parametrize("edit, said", [
    (lambda meta, label: _without(label, "evidence"), "evidence"),
    (lambda meta, label: _without(label, "evidence", "first_contact_s"), "first_contact_s"),
    (lambda meta, label: label["evidence"].update(extra=1.0), "extra"),
    (lambda meta, label: label["evidence"].update(first_contact_s=float("nan")), "first_contact_s"),
    (lambda meta, label: label["evidence"].update(first_contact_s="0.5"), "first_contact_s"),
    (lambda meta, label: _without(meta, "randomization", "condition", "design"), "design"),
    (lambda meta, label: _without(meta, "randomization", "condition"), "condition"),
    (lambda meta, label: meta["randomization"]["condition"].update(design="both"), "design"),
    (lambda meta, label: meta["randomization"]["condition"].update(grid_cell=[0, 1]), "grid_cell"),
    (lambda meta, label: _without(meta, "randomization", "condition", "grid_cell"), "grid_cell"),
], ids=["no-evidence", "a-name-short", "a-name-more", "nan", "text", "no-design", "no-condition",
        "another-design", "a-random-row-with-a-cell", "no-cell"])
def test_check_table_recordings_refuses_a_recording_or_label_without_what_the_columns_come_from(tmp_path, edit, said):
    from generation.labeling.label_episodes import label_episode
    from tests.generation.release_fixture import TABLE_SCENARIO, canon_rules, table_recording
    good = _table_on_disk(tmp_path, table_recording(0, 0))
    arrays, meta = table_recording(1, 1)
    label = label_episode(arrays, meta, canon_rules(), TABLE_SCENARIO)
    edit(meta, label)
    bad = _table_on_disk(tmp_path, (arrays, meta), label=label)
    with pytest.raises(ValueError, match=rf"{bad.name}.*{said}") as e:
        _check_table_recordings([good, bad])
    assert "0.2" in str(e.value) and good.name not in str(e.value)


@pytest.mark.parametrize("cell, said", [([0], "grid_cell"), ([0, -1], "grid_cell"), ([0, True], "grid_cell"),
                                        (None, "grid_cell")])
def test_check_table_recordings_refuses_a_grid_row_without_a_cell_of_two_counts(tmp_path, cell, said):
    from tests.generation.release_fixture import table_recording
    arrays, meta = table_recording(0, 57)
    meta["randomization"]["condition"]["grid_cell"] = cell
    path = _table_on_disk(tmp_path, (arrays, meta))
    with pytest.raises(ValueError, match=said):
        _check_table_recordings([path])


def _stages_named(*names):
    return [{"name": name, "completed": False, "end_step": -1, "duration_s": None} for name in names]


@pytest.mark.parametrize("stages", [[], _stages_named("enter", "exit")], ids=["no stage", "other stage names"])
def test_check_table_recordings_refuses_a_recording_whose_spec_cannot_be_built(tmp_path, stages):
    """What the episode spec refuses is said before a frame, by the file: task_stages that are not the stages the row's
    path has (none, or other names). Built in the conversion's loop, it would surface after the frames of the episodes
    before it were written."""
    from tests.generation.release_fixture import table_recording
    good = _table_on_disk(tmp_path, table_recording(0, 0))
    arrays, meta = table_recording(1, 1)
    meta["task_stages"] = stages
    bad = _table_on_disk(tmp_path, (arrays, meta))
    with pytest.raises(ValueError, match=rf"{bad.name}.*task_stages") as e:
        _check_table_recordings([good, bad])
    assert good.name not in str(e.value) and "0.2" not in str(e.value)


def test_check_table_recordings_refuses_an_asset_path_the_canon_does_not_know(tmp_path):
    from tests.generation.release_fixture import table_recording
    arrays, meta = table_recording(0, 0)
    meta["asset_paths"] = dict(meta["asset_paths"], extra="/not/in/the/canon.usd")
    path = _table_on_disk(tmp_path, (arrays, meta))
    with pytest.raises(ValueError, match=rf"{path.name}.*/not/in/the/canon\.usd"):
        _check_table_recordings([path])


def test_the_conversion_refuses_a_table_recording_whose_spec_cannot_be_built_before_a_frame(tmp_path):
    from tests.generation.release_fixture import table_recording
    h = _Harness(tmp_path)
    arrays, meta = table_recording(0, 0)
    meta["task_stages"] = _stages_named("walk")
    path = _table_on_disk(h.episodes, (arrays, meta))
    with pytest.raises(ValueError, match=rf"{path.name}.*task_stages"):
        h.run()
    assert h.created == [], "거부하기 전에 배포본을 만들기 시작했습니다"


def test_the_conversion_refuses_a_table_recording_without_its_condition_record_before_a_frame(tmp_path):
    h = _Harness(tmp_path)
    eid, _ = h.add_episode(0, media=False, scenario="S1e", assets=V01_ASSETS)
    _give_hri_arrays(h, eid)
    _rewrite_sidecar(h, eid, lambda m: m.update(task_completed_step=2, **_TASK_RECORD))
    with pytest.raises(ValueError, match=rf"{eid}\.npz.*condition"):
        h.run()
    assert h.created == [], "거부하기 전에 배포본을 만들기 시작했습니다"
