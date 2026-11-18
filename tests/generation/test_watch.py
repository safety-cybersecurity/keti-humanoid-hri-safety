"""화면에서 보면서 녹화하는 경로. Isaac 없이 도는 부분만 여기서 시험한다.

Isaac 이 있어야 도는 부분(카메라 추적 콜백, Kit 캡처)은 이 기계에서 실제로 돌려
확인하고 그 결과를 generation/README.md 에 적는다. 여기 있는 것은 그 경로가
읽는 값과 순서 — 런처 인자, 출력 위치, 프레임 전달, 페이싱, 배치 대조 — 이다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from generation.compare_repro import comparable_step_fields
from generation.envs.s1a import watch
from generation.field_map import CAMERAS, STEP_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_S1A = REPO_ROOT / "generation/envs/run_s1a.py"
NOW = datetime(2026, 9, 3, 15, 4, 5)


# --- 런처 인자 ---------------------------------------------------------------

def test_without_flags_the_launcher_gets_exactly_the_headless_dict_it_always_got():
    # 400 에피소드 전부 {"headless": True} 로 부팅해 만들었다. 여기서 다른 값이
    # 나오면 보기 기능이 아니라 생산 경로가 바뀐 것이다.
    assert watch.launcher_args(gui=False, video=False) == {"headless": True}


def test_gui_asks_for_the_kit_visualizer_not_just_headless_false():
    # 이 Isaac Lab 판은 Kit 시각화기를 청하지 않으면 headless=False 를 다시
    # headless 로 되돌린다 (app_launcher._resolve_headless_settings).
    args = watch.launcher_args(gui=True, video=False)
    assert args["headless"] is False
    assert args["visualizer"] == ["kit"]
    assert "enable_cameras" not in args


def test_video_turns_on_cameras_and_the_video_flag():
    args = watch.launcher_args(gui=True, video=True)
    assert args["visualizer"] == ["kit"] and args["headless"] is False
    assert args["enable_cameras"] is True and args["video"] is True


def test_video_without_gui_records_headless():
    args = watch.launcher_args(gui=False, video=True)
    assert args["headless"] is True
    assert args["enable_cameras"] is True and args["video"] is True
    assert "visualizer" not in args


def test_cameras_ask_for_the_camera_renderer_and_nothing_else():
    # S1d 의 카메라는 enable_cameras 가 있어야 렌더된다. 없는 S1d(--no-cameras)와 S1a 는
    # 이 인자를 받지 않는다 — 대조군 ⓑ 가 S1a 와 같은 조건으로 부팅해야 ⓑ↔ⓒ 가 카메라만 가른다.
    assert watch.launcher_args(gui=False, video=False, cameras=True) == {"headless": True, "enable_cameras": True}
    assert watch.launcher_args(gui=False, video=False, cameras=False) == {"headless": True}
    both = watch.launcher_args(gui=True, video=True, cameras=True)
    assert both == {**watch.launcher_args(gui=True, video=True), "enable_cameras": True}


# --- 출력 위치 ---------------------------------------------------------------

def test_the_batch_directory_stays_the_default_when_nothing_is_being_watched():
    assert watch.BATCH_OUT_DIR == ".tmp/episodes/S1a"
    assert watch.resolve_out_dir(None, gui=False, video=False, now=NOW) == watch.BATCH_OUT_DIR


def test_a_watched_run_defaults_to_a_fresh_timestamped_directory():
    # 정본 배치는 다시 만들지 않는 생성분이고 recorder 는 같은 이름의 파일을 덮어쓴다.
    out = watch.resolve_out_dir(None, gui=True, video=False, now=NOW)
    assert out == ".tmp/episodes/S1a-watch-20260903-150405"


def test_video_alone_also_stays_out_of_the_batch_directory():
    out = watch.resolve_out_dir(None, gui=False, video=True, now=NOW)
    assert out.startswith(".tmp/episodes/S1a-watch-") and out != watch.BATCH_OUT_DIR


def test_each_scenario_has_its_own_batch_and_watch_directories():
    # S1d 가 S1a 의 정본 배치 디렉터리에 쓰면 recorder 가 같은 seq 를 덮지는 않아도 두 배치가 섞인다.
    assert watch.resolve_out_dir(None, gui=False, video=False, now=NOW, scenario="S1a") == watch.BATCH_OUT_DIR
    assert watch.resolve_out_dir(None, gui=False, video=False, now=NOW, scenario="S1d") == ".tmp/episodes/S1d"
    assert watch.resolve_out_dir(None, gui=True, video=False, now=NOW, scenario="S1d") == ".tmp/episodes/S1d-watch-20260903-150405"


def test_an_explicit_out_wins_in_every_mode():
    for gui, video in ((False, False), (True, False), (True, True), (False, True)):
        assert watch.resolve_out_dir("/somewhere", gui=gui, video=video, now=NOW) == "/somewhere"


# --- 따라가는 카메라 -----------------------------------------------------------

def test_the_follow_camera_sits_behind_and_above_the_robot_looking_ahead():
    cam = watch.follow_camera()
    assert cam["asset_name"] == "robot"          # run_s1a 가 로봇을 읽는 씬 키
    ex, ey, ez = cam["eye"]
    lx, ly, lz = cam["lookat"]
    assert ex < 0 and ez > 0                      # 로봇은 +x 로 걷는다: 뒤·위에서
    assert lx > 0 and lz < 0                      # 앞을, 골반보다 낮게 본다


class FakeController:
    def __init__(self):
        self.calls = []

    def update_view_to_asset_root(self, asset_name):
        self.calls.append(("root", asset_name))

    def update_view_location(self, eye=None, lookat=None):
        self.calls.append(("location", tuple(eye), tuple(lookat)))


def test_tracking_binds_the_viewport_to_the_robot_root_then_installs_the_offsets():
    # 순서가 계약이다: update_view_location 은 그 시점의 origin 에 오프셋을 더하므로
    # origin 을 로봇으로 옮긴 뒤에 불러야 한다.
    ctrl = FakeController()
    cam = watch.follow_camera()
    watch.track_robot(ctrl, cam)
    assert ctrl.calls == [("root", "robot"), ("location", cam["eye"], cam["lookat"])]


def test_the_video_camera_is_moved_to_the_robot_before_every_capture(monkeypatch):
    # Kit 캡처는 첫 프레임에 한 번만 카메라를 놓는다. 로봇은 20 m 를 걸어가므로
    # 프레임마다 다시 놓아야 화면에 남는다. Isaac 모듈은 가짜로 세운다.
    #
    # 캡처 시점의 카메라 위치를 기록한다 — 마지막 좌표만 보면 「캡처 뒤에 옮기는」
    # 변형도 「처음 한 번만 옮기는」 변형도 통과한다.
    placed = []
    root = np.array([[10.0, 2.0, 1.0]])

    class FakeVideoRecorder:
        def __init__(self):
            self._capture = types.SimpleNamespace(cfg=types.SimpleNamespace(camera_prim_path="/Persp"))
            self._scene = {"robot": types.SimpleNamespace(
                data=types.SimpleNamespace(root_pos_w=types.SimpleNamespace(torch=root)))}
            self.seen_at_capture = []

        def render_rgb_array(self):
            self.seen_at_capture.append(placed[-1] if placed else None)
            return "frame"

    class FakeViewportManager:
        @staticmethod
        def set_camera_view(prim_path, eye, target):
            placed.append((prim_path, tuple(eye), tuple(target)))

    vr = types.ModuleType("isaaclab.envs.utils.video_recorder")
    vr.VideoRecorder = FakeVideoRecorder
    rm = types.ModuleType("isaacsim.core.rendering_manager")
    rm.ViewportManager = FakeViewportManager
    for name, mod in {
        "isaaclab": types.ModuleType("isaaclab"), "isaaclab.envs": types.ModuleType("isaaclab.envs"),
        "isaaclab.envs.utils": types.ModuleType("isaaclab.envs.utils"),
        "isaaclab.envs.utils.video_recorder": vr,
        "isaacsim": types.ModuleType("isaacsim"), "isaacsim.core": types.ModuleType("isaacsim.core"),
        "isaacsim.core.rendering_manager": rm,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    cam = watch.follow_camera()
    recorder = watch.following_video_recorder(cam)()
    ex, ey, ez = cam["eye"]
    lx, ly, lz = cam["lookat"]
    assert recorder.render_rgb_array() == "frame"
    root[0] = (15.0, 2.5, 1.0)                    # the robot walked on
    assert recorder.render_rgb_array() == "frame"
    first = ("/Persp", (10.0 + ex, 2.0 + ey, 1.0 + ez), (10.0 + lx, 2.0 + ly, 1.0 + lz))
    second = ("/Persp", (15.0 + ex, 2.5 + ey, 1.0 + ez), (15.0 + lx, 2.5 + ly, 1.0 + lz))
    assert placed == [first, second]
    assert recorder.seen_at_capture == [first, second]   # placed *before* each capture


# --- 실시간 페이싱 -------------------------------------------------------------

def test_a_usable_speed_is_zero_or_a_finite_number_not_too_small():
    # nan 은 페이싱을 조용히 끄고, 1e-320 은 부팅 뒤 첫 대기에서 OverflowError 를
    # 낸다 ("timestamp out of range for platform time_t").
    for bad in (float("nan"), float("inf"), -1.0, 1e-320, 1e-3):
        assert watch.validate_speed(bad) is not None, bad
    for good in (0.0, watch.MIN_SPEED, 1.0, 2.0, 50.0):
        assert watch.validate_speed(good) is None, good
    assert "0 이상" in watch.validate_speed(-1.0)

class FakeClock:
    def __init__(self):
        self.t = 100.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


def test_speed_zero_never_sleeps():
    c = FakeClock()
    p = watch.Pacer(step_dt=0.02, speed=0.0, clock=c.now, sleep=c.sleep)
    p.start()
    c.t += 0.001
    p.wait()
    assert c.slept == []


def test_a_fast_step_sleeps_for_the_rest_of_the_period():
    c = FakeClock()
    p = watch.Pacer(step_dt=0.02, speed=1.0, clock=c.now, sleep=c.sleep)
    p.start()
    c.t += 0.005
    p.wait()
    assert c.slept == pytest.approx([0.015])


def test_a_slow_step_does_not_sleep_and_does_not_try_to_catch_up():
    # 늦은 스텝 뒤에 빚을 갚으려 몰아 돌리면 화면이 튄다. 다음 마감은 늦은
    # 시점에서 다시 센다.
    c = FakeClock()
    p = watch.Pacer(step_dt=0.02, speed=1.0, clock=c.now, sleep=c.sleep)
    p.start()
    c.t += 0.050
    p.wait()
    assert c.slept == []
    c.t += 0.005
    p.wait()
    assert c.slept == pytest.approx([0.015])


def test_speed_two_halves_the_period():
    c = FakeClock()
    p = watch.Pacer(step_dt=0.02, speed=2.0, clock=c.now, sleep=c.sleep)
    p.start()
    c.t += 0.002
    p.wait()
    assert c.slept == pytest.approx([0.008])


# --- mp4 기록 ------------------------------------------------------------------

class FakeSink:
    def __init__(self):
        self.chunks = []
        self.closed = False

    def send(self, chunk):
        self.chunks.append(bytes(chunk))

    def close(self):
        self.closed = True


def test_frames_reach_the_sink_in_order_as_contiguous_rgb_bytes(tmp_path):
    sink = FakeSink()
    w = watch.Mp4Writer(tmp_path / "x.mp4", width=32, height=16, fps=50,
                        open_sink=lambda path, size, fps: sink, count_frames=lambda p: 2)
    rgba = np.zeros((16, 32, 4), dtype=np.uint8)
    rgba[..., 0] = 7
    rgba[..., 3] = 255                 # 알파는 파일에 새면 안 된다
    w.write(rgba[:, :, :3])            # Kit 캡처가 돌려주는 것과 같은 비연속 뷰
    w.write(np.ones((16, 32, 3), dtype=np.uint8))
    assert w.close() == 2
    assert sink.closed
    assert [len(c) for c in sink.chunks] == [16 * 32 * 3, 16 * 32 * 3]
    assert sink.chunks[0][:3] == bytes([7, 0, 0])
    assert sink.chunks[1] == bytes([1] * (16 * 32 * 3))


def test_a_frame_of_the_wrong_shape_or_dtype_is_refused_not_resized(tmp_path):
    w = watch.Mp4Writer(tmp_path / "x.mp4", width=32, height=16, fps=50,
                        open_sink=lambda *a: FakeSink())
    with pytest.raises(ValueError):
        w.write(np.zeros((32, 16, 3), dtype=np.uint8))     # transposed
    with pytest.raises(ValueError):
        w.write(np.zeros((16, 32, 3), dtype=np.float32))


def test_the_sink_is_opened_on_the_first_frame_with_path_size_and_fps(tmp_path):
    # 프레임이 하나도 없는 에피소드(퇴화)에 빈 mp4 를 남기지 않는다.
    opened = {}

    def open_sink(path, size, fps):
        opened.update(path=path, size=size, fps=fps)
        return FakeSink()

    w = watch.Mp4Writer(tmp_path / "x.mp4", width=1280, height=720, fps=50, open_sink=open_sink)
    assert opened == {}
    assert w.close() == 0
    w = watch.Mp4Writer(tmp_path / "x.mp4", width=1280, height=720, fps=50, open_sink=open_sink)
    w.write(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert opened == {"path": str(tmp_path / "x.mp4"), "size": (1280, 720), "fps": 50}


def test_a_size_the_encoder_would_rescale_is_refused_up_front(tmp_path):
    # imageio_ffmpeg 는 16 의 배수가 아닌 크기에 -vf scale 을 붙여 영상 크기를 바꾼다
    # (66x34 → scale=80:48). 형상 검사만 통과시키면 파일의
    # 크기가 입력과 달라진다.
    with pytest.raises(ValueError):
        watch.Mp4Writer(tmp_path / "x.mp4", width=66, height=34, fps=50, open_sink=lambda *a: FakeSink())
    watch.Mp4Writer(tmp_path / "x.mp4", width=1280, height=720, fps=50, open_sink=lambda *a: FakeSink())


def test_close_verifies_the_file_holds_every_frame_that_was_written(tmp_path):
    # 인코더가 종료코드 1 로 죽어도 imageio_ffmpeg 의 close 는 조용히 돌아온다.
    # 파일에서 프레임을 다시 세는 것이 유일한 확인이다.
    counted = []

    def count_frames(path):
        counted.append(path)
        return 1

    w = watch.Mp4Writer(tmp_path / "x.mp4", width=16, height=16, fps=50,
                        open_sink=lambda *a: FakeSink(), count_frames=count_frames)
    w.write(np.zeros((16, 16, 3), dtype=np.uint8))
    w.write(np.zeros((16, 16, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError):
        w.close()
    assert counted == [str(tmp_path / "x.mp4")]


def test_close_does_not_count_a_file_that_was_never_opened(tmp_path):
    w = watch.Mp4Writer(tmp_path / "x.mp4", width=16, height=16, fps=50,
                        open_sink=lambda *a: FakeSink(), count_frames=lambda p: 99)
    assert w.close() == 0


def test_finish_video_puts_the_mp4_next_to_the_npz(tmp_path):
    w = watch.Mp4Writer(tmp_path / ".episode-0000.part.mp4", width=16, height=16, fps=50,
                        open_sink=lambda *a: FakeSink(), count_frames=lambda p: 1)
    w.write(np.zeros((16, 16, 3), dtype=np.uint8))
    (tmp_path / ".episode-0000.part.mp4").write_bytes(b"x")   # what the sink would have left
    final = watch.finish_video(w, tmp_path / "S1a_0000_seed1.npz")
    assert final == tmp_path / "S1a_0000_seed1.mp4" and final.exists()
    assert not (tmp_path / ".episode-0000.part.mp4").exists()


def test_finish_video_removes_a_stale_mp4_when_this_run_recorded_no_frames(tmp_path):
    # 같은 출력 위치에 같은 seq 를 다시 돌리면 recorder 는 npz 를 덮어쓴다. 프레임이
    # 없는 실행이 옆의 옛 mp4 를 남기면 새 npz 와 짝이 맞지 않는다.
    stale = tmp_path / "S1a_0000_seed1.mp4"
    stale.write_bytes(b"old")
    w = watch.Mp4Writer(tmp_path / ".episode-0000.part.mp4", width=16, height=16, fps=50,
                        open_sink=lambda *a: FakeSink(), count_frames=lambda p: 0)
    assert watch.finish_video(w, tmp_path / "S1a_0000_seed1.npz") is None
    assert not stale.exists()


def test_the_real_encoder_writes_one_frame_per_write(tmp_path):
    # imageio_ffmpeg 는 Isaac 파이썬에만 있다. .venv 에서는 건너뛴다 — 여기서
    # 돌려 보려면 ~/isaacsim/python.sh -m pytest 로 이 파일을 돌린다.
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    path = tmp_path / "real.mp4"
    w = watch.Mp4Writer(path, width=64, height=32, fps=50)
    for i in range(3):
        w.write(np.full((32, 64, 3), i * 40, dtype=np.uint8))
    assert w.close() == 3
    frames, _secs = ffmpeg.count_frames_and_secs(str(path))
    assert frames == 3


# --- 배치와 대조 ---------------------------------------------------------------

N_UNCONDITIONAL = len(comparable_step_fields())
N_WITH_CAMERAS = len(comparable_step_fields((CAMERAS,)))
CAMERA_NPZ = [f.npz for f in STEP_FIELDS if f.condition == CAMERAS and f.kind == "array"]


def _write_npz(path: Path, value: float, timestamp: float = 0.0, media: bool = False,
               box: float | None = None, fingerprint: str = "0" * 64):
    """An episode as the recorder leaves it: the npz and its sidecar. ``media`` makes it
    a camera episode (sidecar names media -- with the recorder's canon fingerprint, ``fingerprint`` --
    and the npz carries the camera arrays); ``box`` sets those arrays' value apart from the rest."""
    arrays = {npz: np.full((3, 2), value) for _, npz in comparable_step_fields()}
    arrays["timestamp"] = np.full(3, timestamp)
    sidecar: dict = {"episode_id": path.stem}
    if media:
        arrays.update({npz: np.full((3, 4), value if box is None else box) for npz in CAMERA_NPZ})
        sidecar["media"] = {"pack": {"file": f"media/{path.stem}.head.pack.npz", "frames": 3},
                            "canon_fingerprint": fingerprint}
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")


def test_an_episode_identical_to_the_batch_reports_no_problems(tmp_path):
    out, batch = tmp_path / "out", tmp_path / "batch"
    out.mkdir(), batch.mkdir()
    _write_npz(out / "S1a_0000_seed1.npz", 1.0)
    _write_npz(batch / "S1a_0000_seed1.npz", 1.0)
    assert watch.compare_with_batch(out, batch) == [("S1a_0000_seed1", [], N_UNCONDITIONAL)]


def test_a_differing_episode_names_the_field(tmp_path):
    out, batch = tmp_path / "out", tmp_path / "batch"
    out.mkdir(), batch.mkdir()
    _write_npz(out / "S1a_0000_seed1.npz", 1.0)
    _write_npz(batch / "S1a_0000_seed1.npz", 2.0)
    [(episode_id, problems, _)] = watch.compare_with_batch(out, batch)
    assert episode_id == "S1a_0000_seed1"
    assert problems and all("값 불일치" in p for p in problems)


def test_an_episode_the_batch_does_not_have_is_reported_not_skipped(tmp_path):
    out, batch = tmp_path / "out", tmp_path / "batch"
    out.mkdir(), batch.mkdir()
    _write_npz(out / "S1a_0400_seed9.npz", 1.0)
    [(episode_id, problems, compared)] = watch.compare_with_batch(out, batch)
    assert episode_id == "S1a_0400_seed9" and len(problems) == 1 and compared == 0


def test_only_the_arrays_the_c9_hook_compares_are_compared(tmp_path):
    # timestamp 는 C9 도 비교하지 않는다. 여기서 다르게 세면 훅과 다른 말을 한다.
    out, batch = tmp_path / "out", tmp_path / "batch"
    out.mkdir(), batch.mkdir()
    _write_npz(out / "S1a_0000_seed1.npz", 1.0, timestamp=0.0)
    _write_npz(batch / "S1a_0000_seed1.npz", 1.0, timestamp=5.0)
    assert watch.compare_with_batch(out, batch) == [("S1a_0000_seed1", [], N_UNCONDITIONAL)]


def test_camera_arrays_are_compared_when_both_sidecars_name_media(tmp_path):
    out, batch = tmp_path / "out", tmp_path / "batch"
    out.mkdir(), batch.mkdir()
    _write_npz(out / "S1d_0000_seed1.npz", 1.0, media=True, box=1.0)
    _write_npz(batch / "S1d_0000_seed1.npz", 1.0, media=True, box=2.0)
    [(_, problems, compared)] = watch.compare_with_batch(out, batch)
    assert compared == N_WITH_CAMERAS > N_UNCONDITIONAL
    assert len(problems) == len(CAMERA_NPZ) and all("값 불일치" in p for p in problems)


def test_the_canon_fingerprint_in_the_sidecar_is_no_part_of_the_comparison(tmp_path):
    # The recorder's stamp (media.canon_fingerprint) is the converter's business: this
    # compares the arrays, and only asks the sidecars whether they name media. Two arms with the same
    # arrays compare equal whatever they were stamped with (arms recorded under different canons show
    # as differing pose arrays, not here).
    out, batch = tmp_path / "out", tmp_path / "batch"
    out.mkdir(), batch.mkdir()
    _write_npz(out / "S1d_0000_seed1.npz", 1.0, media=True, fingerprint="a" * 64)
    _write_npz(batch / "S1d_0000_seed1.npz", 1.0, media=True, fingerprint="b" * 64)
    assert watch.compare_with_batch(out, batch) == [("S1d_0000_seed1", [], N_WITH_CAMERAS)]


@pytest.mark.parametrize("media_side", ["out", "batch"])
def test_a_one_sided_camera_run_compares_only_the_unconditional_arrays(tmp_path, media_side):
    # The camera-less control arm against a camera arm: the camera arrays exist on one
    # side only, so they are not a mismatch -- the comparison is the 16 it always was.
    out, batch = tmp_path / "out", tmp_path / "batch"
    out.mkdir(), batch.mkdir()
    _write_npz(out / "S1d_0000_seed1.npz", 1.0, media=media_side == "out")
    _write_npz(batch / "S1d_0000_seed1.npz", 1.0, media=media_side == "batch")
    assert watch.compare_with_batch(out, batch) == [("S1d_0000_seed1", [], N_UNCONDITIONAL)]


@pytest.mark.parametrize("missing_side", ["out", "batch"])
def test_an_episode_without_its_sidecar_is_reported_not_compared(tmp_path, missing_side):
    # Whether the camera arrays are compared is read from the sidecars; without one there
    # is nothing to read it from, and guessing "no media" would quietly compare fewer.
    out, batch = tmp_path / "out", tmp_path / "batch"
    out.mkdir(), batch.mkdir()
    _write_npz(out / "S1d_0000_seed1.npz", 1.0, media=True)
    _write_npz(batch / "S1d_0000_seed1.npz", 1.0, media=True)
    (tmp_path / missing_side / "S1d_0000_seed1.json").unlink()
    [(_, problems, compared)] = watch.compare_with_batch(out, batch)
    assert compared == 0 and len(problems) == 1 and "사이드카" in problems[0]


def test_comparing_a_directory_with_itself_is_refused(tmp_path):
    # --out 과 --compare-with 가 같으면 자기 자신과 비교해 늘 「동일」이 된다.
    # 심볼릭 링크로 같은 곳을 가리켜도 마찬가지다.
    out = tmp_path / "out"
    out.mkdir()
    _write_npz(out / "S1a_0000_seed1.npz", 1.0)
    alias = tmp_path / "alias"
    alias.symlink_to(out)
    with pytest.raises(ValueError):
        watch.compare_with_batch(out, out)
    with pytest.raises(ValueError):
        watch.compare_with_batch(out, alias)


def test_the_verdict_fails_when_nothing_was_compared():
    # 비어 있는 출력은 대조를 하나도 안 하고도 「통과」가 될 수 있었다.
    ok, lines = watch.comparison_verdict([])
    assert ok is False and lines


def test_the_verdict_fails_on_any_mismatch_and_passes_only_when_all_match():
    ok, lines = watch.comparison_verdict([("a", [], 16), ("b", ["x: 값 불일치"], 16)])
    assert ok is False and any("b" in ln and "불일치" in ln for ln in lines)
    ok, lines = watch.comparison_verdict([("a", [], 16), ("b", [], 16)])
    assert ok is True and len(lines) == 2


def test_the_verdict_says_what_was_compared_not_bit_identical():
    # +0.0 과 -0.0 은 바이트가 다른데 같다고 판정되고 timestamp 는 아예 비교하지
    # 않는다. 「비트 단위 동일」은 실제 비교보다 강한 말이다.
    ok, [line] = watch.comparison_verdict([("S1a_0000_seed1", [], N_UNCONDITIONAL)])
    assert ok and "비트" not in line and f"{N_UNCONDITIONAL}개" in line


def test_the_verdict_counts_what_each_episode_actually_compared():
    # A camera pair compares more arrays than a one-sided one; each line says its own.
    ok, lines = watch.comparison_verdict([("a", [], N_WITH_CAMERAS), ("b", [], N_UNCONDITIONAL)])
    assert ok and f"{N_WITH_CAMERAS}개" in lines[0] and f"{N_UNCONDITIONAL}개" in lines[1]


def test_a_fresh_output_directory_is_claimed_not_shared(tmp_path):
    # 같은 초에 시작한 두 실행이 같은 경로를 받으면 .part.mp4 까지 공유한다.
    # 디렉터리를 만들며 잡고, 이미 있으면 번호를 붙인다.
    first = watch.claim_out_dir(str(tmp_path / "S1a-watch-20260906-120000"))
    second = watch.claim_out_dir(str(tmp_path / "S1a-watch-20260906-120000"))
    assert first == str(tmp_path / "S1a-watch-20260906-120000")
    assert second == str(tmp_path / "S1a-watch-20260906-120000-2")
    assert Path(first).is_dir() and Path(second).is_dir()


# --- run_s1a.py 배선 -----------------------------------------------------------

def _run_s1a_capturing_launcher(tmp_path: Path, *flags: str):
    """run_s1a.py 를 가짜 isaac_boot 아래에서 돌려, 런처가 받는 인자를 잡는다.

    launch_app 이 인자를 적고 SystemExit(0) 을 던지므로 Isaac import 앞에서 멈춘다.

    **--out 은 tmp_path 안을 가리키게 주십시오.** 없이 --gui/--video 를 주면
    run_s1a.py 가 부팅 전에 저장소의 실제 .tmp/episodes/S1a-watch-<시각>/ 를 만들어
    잡는데(claim_out_dir), 시험은 런처 인자만 보고 끝나므로 빈 디렉터리가 남는다 —
    make check 한 번에 하나씩 쌓인다(45개까지 쌓인 적이 있다). 아래에서 실행 전후를
    세어 새로 생겼으면 실패시킨다. 다음 시험도 같은 자리에 빠지기 때문이다.
    """
    episodes = REPO_ROOT / ".tmp" / "episodes"
    before = set(episodes.iterdir()) if episodes.is_dir() else set()
    captured = tmp_path / "launcher.json"
    driver = tmp_path / "driver.py"
    driver.write_text(textwrap.dedent(f"""
        import json, runpy, sys, types
        boot = types.ModuleType("generation.isaac_boot")
        def launch_app(launcher_args=None):
            open({str(captured)!r}, "w").write(json.dumps(launcher_args))
            raise SystemExit(0)
        boot.launch_app = launch_app
        sys.modules["generation.isaac_boot"] = boot
        sys.argv = [{str(RUN_S1A)!r}, *{list(flags)!r}]
        runpy.run_path({str(RUN_S1A)!r}, run_name="__main__")
        """), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run([sys.executable, str(driver)], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=120)
    args = json.loads(captured.read_text(encoding="utf-8")) if captured.exists() else None
    leaked = (set(episodes.iterdir()) if episodes.is_dir() else set()) - before
    assert not leaked, (
        "시험이 저장소에 디렉터리를 남겼습니다: "
        + ", ".join(sorted(x.name for x in leaked))
        + " — run_s1a.py 는 --out 이 없으면 부팅 전에 출력 디렉터리를 만들어"
          " 잡습니다. tmp_path 안을 가리키는 --out 을 함께 주십시오."
    )
    return proc, args


def test_run_s1a_boots_headless_exactly_as_before_when_no_watch_flag_is_given(tmp_path):
    proc, args = _run_s1a_capturing_launcher(tmp_path)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert args == {"headless": True}


def test_run_s1a_passes_the_watch_launcher_args_through(tmp_path):
    # 유도해 놓고 안 쓰면 유도한 적 없는 것과 같다.
    # --out 은 런처 인자를 바꾸지 않는다. 없으면 저장소에 빈 디렉터리가 남는다
    # (_run_s1a_capturing_launcher 독스트링).
    proc, args = _run_s1a_capturing_launcher(
        tmp_path, "--gui", "--video", "--out", str(tmp_path / "out"))
    assert proc.returncode == 0, proc.stderr[-800:]
    assert args == watch.launcher_args(gui=True, video=True)


def test_s1d_boots_with_the_camera_renderer(tmp_path):
    proc, args = _run_s1a_capturing_launcher(tmp_path, "--scenario", "S1d")
    assert proc.returncode == 0, proc.stderr[-800:]
    assert args == watch.launcher_args(gui=False, video=False, cameras=True)


def test_s1d_without_cameras_boots_exactly_like_s1a(tmp_path):
    proc, args = _run_s1a_capturing_launcher(tmp_path, "--scenario", "S1d", "--no-cameras", "--no-skin",
                                             "--out", str(tmp_path / "ctl"))
    assert proc.returncode == 0, proc.stderr[-800:]
    assert args == {"headless": True}


def test_control_flags_without_an_out_are_rejected_before_booting(tmp_path):
    # Episode ids carry only scenario and seq, so a control arm written to the S1d batch directory
    # would overwrite the batch's episodes of the same seq.
    for flags in (("--no-cameras",), ("--no-skin", "--no-cameras")):
        proc, args = _run_s1a_capturing_launcher(tmp_path, "--scenario", "S1d", *flags)
        assert proc.returncode == 2 and args is None, flags
        assert "--out" in proc.stderr


def test_s1d_only_flags_are_rejected_for_s1a_before_booting(tmp_path):
    for flag in ("--no-cameras", "--no-skin"):
        proc, args = _run_s1a_capturing_launcher(tmp_path, flag)
        assert proc.returncode == 2 and args is None, flag
        assert "--scenario S1d" in proc.stderr


def test_free_camera_without_gui_is_rejected_before_booting(tmp_path):
    proc, args = _run_s1a_capturing_launcher(tmp_path, "--free-camera")
    # argparse 가 모르는 플래그도 2 로 끝나므로, 검증이 낸 문장까지 본다.
    assert proc.returncode == 2 and args is None
    assert "--gui 와 함께" in proc.stderr


def test_a_negative_speed_is_rejected_before_booting(tmp_path):
    proc, args = _run_s1a_capturing_launcher(tmp_path, "--gui", "--speed", "-1")
    assert proc.returncode == 2 and args is None
    assert "0 이상" in proc.stderr


def test_a_non_finite_or_tiny_speed_is_rejected_before_booting(tmp_path):
    for bad in ("nan", "inf", "1e-320"):
        proc, args = _run_s1a_capturing_launcher(tmp_path, "--gui", "--speed", bad)
        assert proc.returncode == 2 and args is None, bad


def test_comparing_the_output_directory_with_itself_is_rejected_before_booting(tmp_path):
    out = tmp_path / "ref"
    out.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(out)
    for target in (out, alias):
        proc, args = _run_s1a_capturing_launcher(
            tmp_path, "--video", "--out", str(out), "--compare-with", str(target))
        assert proc.returncode == 2 and args is None, target
        assert "자신" in proc.stderr


def test_the_wiring_after_boot_is_pinned_in_the_source():
    # 부팅 뒤의 배선은 Isaac 없이 실행할 수 없다. 이 저장소가 호출부를 원문으로
    # 확인하는 시험을 두는 이유와 같다 — 이 줄들을 지워도 다른 시험은 초록이다.
    # 실측은 generation/README.md 에 있다.
    src = RUN_S1A.read_text(encoding="utf-8")
    for needle in (
        'render_mode="rgb_array" if args.video else None',
        "env_cfg.video_recorder.class_type = watch.following_video_recorder(watch.follow_camera())",
        "watch.track_robot(base_env.viewport_camera_controller, watch.follow_camera())",
        "watch.finish_video(writer, path)",
        "ok, verdict_lines = watch.comparison_verdict(",
        "code = 0 if compare_ok else 3",
    ):
        assert needle in src, needle


def test_run_s1a_help_lists_the_watch_flags():
    proc = subprocess.run([sys.executable, str(RUN_S1A), "--help"], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-800:]
    for flag in ("--gui", "--video", "--speed", "--free-camera", "--compare-with"):
        assert flag in proc.stdout


def test_run_s1a_puts_media_in_place_only_after_the_episode_is_saved():
    # run_s1a cannot run in CI, so its order is read from the source: close() stages, save() writes the
    # npz and sidecar, commit() replaces the media, and a failed save discards what was staged
    # (a save that failed after the media were replaced left them beside
    # the earlier run's npz and sidecar).
    import ast
    tree = ast.parse(RUN_S1A.read_text(encoding="utf-8"))
    calls = [(n.lineno, n.func.attr) for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr in ("close", "save", "commit", "discard")
             and isinstance(n.func.value, ast.Name) and n.func.value.id in ("media", "recorder")]
    order = [name for _, name in sorted(calls)]
    assert order.index("close") < order.index("save") < order.index("commit")
    handlers = [h for h in ast.walk(tree) if isinstance(h, ast.ExceptHandler)]
    assert any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "discard"
               for h in handlers for c in ast.walk(h))


def _add_hri(npz: Path, value: float, leave_out: str | None = None) -> None:
    from generation.field_map import HRI
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    for f in STEP_FIELDS:
        if f.condition == HRI and f.kind == "array" and f.npz != leave_out:
            arrays[f.npz] = np.full(3, value)
    np.savez(npz, **arrays)


def test_the_hri_arrays_join_the_comparison_when_both_episodes_carry_them(tmp_path):
    from generation.field_map import HRI
    out, batch = tmp_path / "out", tmp_path / "batch"
    out.mkdir()
    batch.mkdir()
    for d, value in ((out, 0.5), (batch, 0.7)):
        _write_npz(d / "S1e_0000_seed1.npz", 1.0)
        _add_hri(d / "S1e_0000_seed1.npz", value)
    [(_, problems, n)] = watch.compare_with_batch(out, batch)
    assert n == len(comparable_step_fields((HRI,)))
    assert any(p.startswith("safety.approach_speed_robot") for p in problems)


def test_half_an_hri_recording_is_a_reported_problem(tmp_path):
    out, batch = tmp_path / "out", tmp_path / "batch"
    out.mkdir()
    batch.mkdir()
    _write_npz(out / "S1e_0000_seed1.npz", 1.0)
    _write_npz(batch / "S1e_0000_seed1.npz", 1.0)
    _add_hri(out / "S1e_0000_seed1.npz", 0.5, leave_out="human_root_vel")
    [(_, problems, n)] = watch.compare_with_batch(out, batch)
    assert n == 0 and any("일부만" in p for p in problems)


def test_half_an_hri_recording_on_the_batch_side_is_a_reported_problem_too(tmp_path):
    # The check must not short-circuit on the output side: a complete hri recording against a half one
    # is refused, whichever side is the half.
    out, batch = tmp_path / "out", tmp_path / "batch"
    out.mkdir()
    batch.mkdir()
    _write_npz(out / "S1e_0000_seed1.npz", 1.0)
    _write_npz(batch / "S1e_0000_seed1.npz", 1.0)
    _add_hri(batch / "S1e_0000_seed1.npz", 0.5, leave_out="human_root_vel")
    [(_, problems, n)] = watch.compare_with_batch(out, batch)
    assert n == 0 and any("일부만" in p for p in problems)


@pytest.mark.parametrize("hri_side", ["out", "batch"])
def test_a_one_sided_hri_run_compares_only_the_unconditional_arrays(tmp_path, hri_side):
    # The hri arrays exist on one side only (a run against a control recorded without them), so they
    # are not a mismatch -- the comparison is of the unconditional arrays alone.
    out, batch = tmp_path / "out", tmp_path / "batch"
    out.mkdir()
    batch.mkdir()
    for d in (out, batch):
        _write_npz(d / "S1e_0000_seed1.npz", 1.0)
    _add_hri(tmp_path / hri_side / "S1e_0000_seed1.npz", 0.5)
    assert watch.compare_with_batch(out, batch) == [("S1e_0000_seed1", [], N_UNCONDITIONAL)]
