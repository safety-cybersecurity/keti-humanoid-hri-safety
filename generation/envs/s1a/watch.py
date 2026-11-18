"""Watch an S1a run in the Kit window and record it as mp4.

Everything the batch was generated with stays as it was: without ``--gui`` or
``--video`` the launcher gets the same ``{"headless": True}`` it always got, and
the output directory is still the batch. The functions here only decide what
changes when one of those flags is on.

Isaac-free at import. Every Isaac import lives inside the function that needs
it, so tests/generation/test_watch.py runs in .venv and in CI.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from generation.compare_repro import comparable_step_fields, compare_arrays
from generation.field_map import CAMERAS, HRI, hri_recorded

# run_s1a.py's default --out: the v0.1 batch, generated 2026-08-25 and never
# regenerated (.tmp/README.md). EpisodeRecorder.save() overwrites a same-named
# file without asking, so a watched run never defaults into this directory.
BATCH_OUT_DIR = ".tmp/episodes/S1a"
WATCH_OUT_DIR_PREFIX = f"{BATCH_OUT_DIR}-watch-"      # resolve_out_dir builds it the same way per scenario

# Camera offsets from the robot root (the pelvis, about 1 m up), in metres.
# The robot walks along +x and the dummy comes in from the side, so the camera
# sits behind, off to one side and above, looking ahead and down at the patch
# of ground the two meet on. Both offsets are added to the live root position
# every frame; a fixed world-frame camera loses a robot that walks 20 m.
FOLLOW_EYE = (-3.0, -3.0, 1.2)
FOLLOW_LOOKAT = (1.5, 0.0, -0.4)

# Slowest pacing accepted. Below this the period grows past what time.sleep
# can take (1e-320 gave "timestamp out of range for platform time_t" after a
# full boot, 2026-09-06); nan and inf would silently disable pacing.
MIN_SPEED = 0.01

# The encoder aligns frame size to this and *rescales* anything else with
# -vf scale (66x34 came out 80x48). Refuse such sizes instead of writing a
# file whose frames are not the frames that were handed in.
MACRO_BLOCK = 16


def launcher_args(gui: bool, video: bool, cameras: bool = False) -> dict:
    """What isaac_boot.launch_app gets.

    On this Isaac Lab (v3.0 beta) ``headless=False`` on its own is forced back
    to headless unless a Kit visualizer is requested explicitly
    (app_launcher._resolve_headless_settings), hence ``visualizer=["kit"]``.
    ``--video`` follows Isaac Lab's own play.py: recording needs
    ``enable_cameras`` for the replicator render product, and ``video`` keeps
    the viewport alive when recording headless. ``cameras`` (S1d's sensor
    cameras) needs ``enable_cameras`` and nothing else.
    """
    if not gui and not video:
        return {"headless": True, "enable_cameras": True} if cameras else {"headless": True}
    args: dict = {"headless": not gui}
    if gui:
        args["visualizer"] = ["kit"]
    if video or cameras:
        args["enable_cameras"] = True
    if video:
        args["video"] = True
    return args


def resolve_out_dir(out: str | None, gui: bool, video: bool,
                    now: datetime | None = None, scenario: str = "S1a") -> str:
    """An explicit --out wins; otherwise the scenario's batch, or a fresh directory per watched run."""
    if out is not None:
        return out
    batch = BATCH_OUT_DIR if scenario == "S1a" else f".tmp/episodes/{scenario}"
    if not (gui or video):
        return batch
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{batch}-watch-{stamp}"


def claim_out_dir(path: str) -> str:
    """Create ``path`` for this run, or a numbered sibling if it already exists.

    Two watched runs started in the same second get the same timestamp; left
    at that, they would share the directory and the temporary mp4 names in
    it. mkdir(exist_ok=False) is atomic, so the claim is too.
    """
    base = Path(path)
    for n in range(1, 1000):
        candidate = base if n == 1 else base.with_name(f"{base.name}-{n}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return str(candidate)
    raise RuntimeError(f"출력 디렉터리를 잡지 못했습니다: {path}")


def validate_speed(speed: float) -> str | None:
    """None if --speed is usable, otherwise the reason it is not."""
    if not math.isfinite(speed):
        return "--speed 는 유한한 수여야 합니다"
    if speed < 0:
        return "--speed 는 0 이상이어야 합니다"
    if 0 < speed < MIN_SPEED:
        return f"--speed 는 0 이거나 {MIN_SPEED} 이상이어야 합니다"
    return None


def follow_camera() -> dict:
    """The viewport's frame of reference: the robot root plus the two offsets."""
    return {"asset_name": "robot", "eye": FOLLOW_EYE, "lookat": FOLLOW_LOOKAT}


def track_robot(controller, cam: dict) -> None:
    """Make the on-screen viewport follow the robot.

    ``controller`` is the env's ViewportCameraController. Order matters:
    update_view_location adds the offsets to whatever origin the controller
    holds at that moment, so the origin is moved to the robot root first. From
    then on the controller's own post-update callback re-applies both every
    render.
    """
    controller.update_view_to_asset_root(cam["asset_name"])
    controller.update_view_location(eye=cam["eye"], lookat=cam["lookat"])


def following_video_recorder(cam: dict) -> type:
    """A VideoRecorder whose capture camera follows the robot.

    Isaac Lab's Kit capture places ``/OmniverseKit_Persp`` once, on the first
    frame, from a world-frame eye/lookat. The robot then walks out of it. This
    subclass moves that same camera to the live robot root before every
    capture, so the mp4 stays framed whether or not a window is open.
    """
    from isaaclab.envs.utils.video_recorder import VideoRecorder

    class FollowingVideoRecorder(VideoRecorder):
        def render_rgb_array(self):
            if self._capture is not None:
                from isaacsim.core.rendering_manager import ViewportManager

                root = self._scene[cam["asset_name"]].data.root_pos_w.torch[0]
                root = [float(v) for v in root]
                eye = [r + o for r, o in zip(root, cam["eye"])]
                lookat = [r + o for r, o in zip(root, cam["lookat"])]
                ViewportManager.set_camera_view(
                    self._capture.cfg.camera_prim_path, eye=eye, target=lookat
                )
            return super().render_rgb_array()

    return FollowingVideoRecorder


class Pacer:
    """Hold the loop to wall-clock time so the window plays at ``speed`` x real time.

    ``speed`` 0 disables pacing. A step that overran its period is not paid
    back by rushing the next ones -- that makes the picture jump -- the next
    deadline is simply measured from the late step.
    """

    def __init__(self, step_dt: float, speed: float,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.period = step_dt / speed if speed > 0 else 0.0
        self._clock = clock
        self._sleep = sleep
        self.start()

    def start(self) -> None:
        self._deadline = self._clock() + self.period

    def wait(self) -> None:
        if self.period <= 0:
            return
        now = self._clock()
        if now < self._deadline:
            self._sleep(self._deadline - now)
            self._deadline += self.period
        else:
            self._deadline = now + self.period


def _ffmpeg_sink(path: str, size: tuple[int, int], fps: int):
    """The real encoder: H.264 through imageio_ffmpeg's bundled ffmpeg, frame by frame."""
    import imageio_ffmpeg

    gen = imageio_ffmpeg.write_frames(path, size, fps=fps, codec="libx264", pix_fmt_in="rgb24")
    gen.send(None)  # seed the generator; it now waits for frame bytes
    return gen


def _ffmpeg_count_frames(path: str) -> int:
    """How many frames the finished file actually holds, read back from the file."""
    import imageio_ffmpeg

    frames, _seconds = imageio_ffmpeg.count_frames_and_secs(path)
    return int(frames)


class Mp4Writer:
    """Stream RGB frames into an mp4 as they arrive.

    Frames go out one at a time rather than being kept: a 20 s episode at
    50 fps and 1280x720 would otherwise hold 2.7 GB before anything is
    written. The sink is opened on the first frame, so an episode with no
    frames leaves no file behind. ``open_sink`` is only for tests.
    """

    def __init__(self, path, width: int, height: int, fps: int,
                 open_sink=None, count_frames=None):
        if width <= 0 or height <= 0 or width % MACRO_BLOCK or height % MACRO_BLOCK:
            raise ValueError(
                f"영상 크기는 {MACRO_BLOCK} 의 배수여야 합니다 (인코더가 아니면 크기를 "
                f"바꿉니다): {width}x{height}"
            )
        self.path = Path(path)
        self.size = (width, height)
        self.fps = fps
        self.frames = 0
        self._open_sink = open_sink or _ffmpeg_sink
        self._count_frames = count_frames or _ffmpeg_count_frames
        self._sink = None

    def write(self, frame) -> None:
        frame = np.asarray(frame)
        expected = (self.size[1], self.size[0], 3)
        if frame.dtype != np.uint8 or frame.shape != expected:
            raise ValueError(
                f"프레임이 {expected} uint8 이어야 합니다: {frame.shape} {frame.dtype}"
            )
        if self._sink is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._sink = self._open_sink(str(self.path), self.size, self.fps)
        # The Kit capture hands back rgb[:, :, :3] of an RGBA buffer -- a
        # non-contiguous view. tobytes() on a contiguous copy is what the
        # encoder's stdin expects: exactly width*height*3 bytes per frame.
        self._sink.send(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self) -> int:
        """Finish the file and check it. Returns the number of frames written.

        The encoder's close returns normally even when ffmpeg exited nonzero
        (measured 2026-09-06, exit code 1 injected), so a broken encode would be
        reported as success. Counting the frames back out of the file is the
        one check that does not trust the encoder's word.
        """
        if self._sink is not None:
            self._sink.close()
            self._sink = None
        if self.frames:
            found = self._count_frames(str(self.path))
            if found != self.frames:
                raise RuntimeError(
                    f"영상 프레임 수가 맞지 않습니다: 썼음 {self.frames}, 파일 {found} — {self.path}"
                )
        return self.frames


def finish_video(writer: Mp4Writer, npz_path) -> Path | None:
    """Close the writer and put the mp4 next to the npz under the npz's name.

    A run that recorded no frames leaves no mp4 -- and removes a stale one
    from an earlier run of the same seq, so an npz never sits next to a video
    of a different run.
    """
    frames = writer.close()
    final = Path(npz_path).with_suffix(".mp4")
    if frames:
        writer.path.replace(final)
        return final
    final.unlink(missing_ok=True)
    return None


def comparison_verdict(report: list[tuple[str, list[str], int]]) -> tuple[bool, list[str]]:
    """(passed, lines to print) for a compare_with_batch report.

    Nothing compared is a failure, not a pass -- an empty output directory
    must not read as "all identical". The wording names what was compared:
    the C9 arrays, with NaN equal to NaN and timestamp left out, which is
    weaker than "bit-identical" (+0.0 and -0.0 compare equal). The count is
    each episode's own: a camera pair compares more arrays than a one-sided one.
    """
    if not report:
        return False, ["대조한 에피소드가 없습니다"]
    ok = True
    lines = []
    for episode_id, problems, n_arrays in report:
        if problems:
            ok = False
            lines.append(f"{episode_id}: 불일치: " + "; ".join(problems))
        else:
            lines.append(f"{episode_id}: 동일 (C9 대조 배열 {n_arrays}개)")
    return ok, lines


def _names_media(npz: Path) -> bool:
    """Whether this episode's sidecar names media, i.e. it was recorded with cameras."""
    return "media" in json.loads(npz.with_suffix(".json").read_text(encoding="utf-8"))


def _records_hri(npz: Path) -> bool:
    """Whether this episode's npz carries the hri arrays (field_map.hri_recorded)."""
    with np.load(npz) as arrays:
        return hri_recorded(arrays.files)


def compare_with_batch(out_dir, batch_dir) -> list[tuple[str, list[str], int]]:
    """Per episode in ``out_dir``: (episode, mismatches against the same-named npz in
    ``batch_dir``, number of arrays compared).

    Compares exactly what the C9 hook compares (compare_repro), so "equal"
    here means the same thing it means there. The camera arrays join the
    comparison only when both sidecars name media: a camera-less control arm
    against a camera arm compares the unconditional arrays, as it always did.
    The hri arrays join when both npz carry them; half of them, on either side, is a reported
    problem.
    An episode the batch lacks, or one without its sidecar to say which, is a
    reported problem, not a skip.
    """
    out_dir, batch_dir = Path(out_dir), Path(batch_dir)
    if out_dir.resolve() == batch_dir.resolve():
        # Comparing a directory with itself passes every time and proves
        # nothing; symlinks and relative spellings resolve to the same place.
        raise ValueError(f"대조 대상이 출력 디렉터리 자신입니다: {out_dir}")
    report: list[tuple[str, list[str], int]] = []
    for npz in sorted(out_dir.glob("*.npz")):
        other = batch_dir / npz.name
        if not other.exists():
            report.append((npz.stem, ["배치에 같은 에피소드가 없습니다"], 0))
            continue
        no_sidecar = [str(p.with_suffix(".json")) for p in (npz, other)
                      if not p.with_suffix(".json").exists()]
        if no_sidecar:
            report.append((npz.stem, [f"사이드카가 없습니다: {no_sidecar}"], 0))
            continue
        both_cameras = _names_media(npz) and _names_media(other)
        try:
            # Both sides are read before they are combined: `and` would never look at the batch's
            # recording once the output's carries no hri arrays.
            out_hri, other_hri = _records_hri(npz), _records_hri(other)
        except ValueError as e:
            report.append((npz.stem, [str(e)], 0))
            continue
        both_hri = out_hri and other_hri
        active = tuple(c for c, on in ((CAMERAS, both_cameras), (HRI, both_hri)) if on)
        pairs = comparable_step_fields(active)
        with np.load(npz) as left, np.load(other) as right:
            problems = compare_arrays(dict(left), dict(right), pairs)
        report.append((npz.stem, problems, len(pairs)))
    return report
