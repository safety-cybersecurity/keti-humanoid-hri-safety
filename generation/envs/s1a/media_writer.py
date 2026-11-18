"""Camera frames -> lossless intermediate video; the dummy's box and the camera poses -> per-step
arrays; and, only when meta/cameras.json declares a pack, that camera's depth and mask -> the pack.
The canon declares none since 2026-09-30 (the pack code stays, the canon switches it off): the
segmentation is then read for the box alone and not kept.

Isaac-free: the run loop reads the cameras and hands this plain numpy, once per recorded step,
so "frame count == recorded steps" is a check (step_index has to follow on), not a hope. Videos
go through watch.Mp4Writer's streaming writer with a lossless sink -- the converter's encode is
the only lossy one -- and Mp4Writer counts the finished file's frames back rather than trusting
the encoder's exit status.

RGB is not bit-reproducible across runs (the renderer's temporal accumulation, measured
2026-09-23); depth and masks are, and so is the box computed from a mask. The box -- and a declared
pack -- carries what can be compared; the video is what was rendered once.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from generation import cameras as camera_canon
from generation.convert_lerobot import the_pack
from generation.envs.s1a.watch import Mp4Writer

# What this writer writes. meta/cameras.json ships as the reader's description of it, so the writer
# refuses a canon that describes anything else (CONTRIBUTING.md 「함께 고치는 곳」). The box always; a
# pack's formats only when the canon declares one, under packs.<the name the canon gives it>.
WRITTEN_FORMAT = {("bbox", "invalid"): [-1, -1, -1, -1]}
WRITTEN_PACK_FORMAT = {("depth", "dtype"): "uint16", ("depth", "unit"): "mm", ("depth", "invalid"): 0,
                       ("mask", "dtype"): "uint8"}
# Media go in a subdirectory: the labeller, the converter and the other episode readers take every
# *.npz in the episode directory for an episode, and the pack is an npz.
MEDIA_DIR = "media"
_DEPTH_MAX_MM = int(np.iinfo(np.uint16).max)


def _lossless_sink(path: str, size: tuple[int, int], fps: int):
    """libx264rgb at -qp 0: RGB in, the same RGB out."""
    import imageio_ffmpeg

    gen = imageio_ffmpeg.write_frames(path, size, fps=fps, codec="libx264rgb", pix_fmt_in="rgb24",
                                      pix_fmt_out="rgb24", quality=None, output_params=["-qp", "0"])
    gen.send(None)  # seed the generator; it now waits for frame bytes
    return gen


def decode_labels(raw: dict) -> dict[int, str]:
    """idToLabels -> {segmentation id: label}. The keys are "(r, g, b, a)" strings; the int32
    image holds that colour as r | g<<8 | b<<16 | a<<24 read as a signed 32-bit number."""
    out: dict[int, str] = {}
    for key, value in raw.items():
        r, g, b, a = (int(x) for x in re.findall(r"-?\d+", str(key)))
        packed = r | (g << 8) | (b << 16) | (a << 24)
        out[packed - (1 << 32) if packed >= (1 << 31) else packed] = (
            value.get("class", "") if isinstance(value, dict) else str(value))
    return out


def remap_mask(seg: np.ndarray, labels: dict[int, str], classes: dict[str, str]) -> np.ndarray:
    """Segmentation ids -> uint8 class ids of ``classes`` ({"id": name}: a declared pack's mask
    classes, or with no pack the box's classes numbered from 1). Names match case-insensitively
    (Replicator lower-cases labels); of several comma-joined labels the first one the classes know
    wins. Everything else is background, 0."""
    by_name = {name.lower(): int(i) for i, name in classes.items()}
    out = np.zeros(np.shape(seg), np.uint8)
    for sid, label in labels.items():
        for part in str(label).split(","):
            cid = by_name.get(part.strip().lower())
            if cid is not None:
                out[seg == sid] = cid
                break
    return out


def union_bbox(mask: np.ndarray, class_ids) -> np.ndarray:
    """[x_min, y_min, x_max, y_max] of the pixels in class_ids as pixel indices, both ends
    included; [-1, -1, -1, -1] when none is visible."""
    rows, cols = np.nonzero(np.isin(mask, list(class_ids)))
    if rows.size == 0:
        return np.full(4, -1, np.int32)
    return np.array([cols.min(), rows.min(), cols.max(), rows.max()], np.int32)


def is_black(frame) -> bool:
    """Every pixel 0: what a camera buried in some geometry, or mounted in the wrong place, renders. One
    definition for the writer's count and for camera_io.Rig, which never takes such a frame for a discarded
    render -- all zeros repeat whether drawn again or not."""
    return bool(np.asarray(frame).max() == 0)


def depth_to_mm(depth: np.ndarray) -> np.ndarray:
    """Metres -> uint16 millimetres. Non-finite (no hit) is 0, the canon's invalid value; beyond
    65.535 m saturates."""
    d = np.asarray(depth, np.float64)
    mm = np.where(np.isfinite(d), np.rint(d * 1000.0), 0.0)
    return np.clip(mm, 0, _DEPTH_MAX_MM).astype(np.uint16)


class MediaWriter:
    """One episode's camera media. step() per recorded step, close() once; close() returns what
    the recorder stores: file entries, two counts and the canon's fingerprint for the sidecar (paths
    relative to the episode directory, under media/; the steps rendered again, each camera's black
    frames; the fingerprint of the camera canon the recording is made with) and the bbox/pose arrays
    for the npz. ``pack`` is the name of the pack the canon declares, or None -- then no pack is
    written and the sidecar names none."""

    def __init__(self, out_dir, episode_id: str, cameras_canon: dict, open_sink=None, count_frames=None):
        # the_pack: the converter's reading of the canon -- None, or the one pack (two are refused:
        # the sidecar has a single media.pack), so the recording is what the conversion will expect.
        self.pack = the_pack(cameras_canon)
        written = dict(WRITTEN_FORMAT)
        if self.pack is not None:
            written.update({("packs", self.pack, *path): value for path, value in WRITTEN_PACK_FORMAT.items()})
        for path, value in written.items():
            node = cameras_canon
            for key in path:
                node = node[key]
            if node != value:
                raise ValueError(f"meta/cameras.json {'.'.join(path)} is {node!r}; this writer writes {value!r}")
        self.media_dir = Path(out_dir) / MEDIA_DIR
        self.episode_id = episode_id
        # Of the canon this writer was handed -- the one whose format was just checked, and the one the
        # pixels are baked from. The converter holds it to the canon it converts with
        # (convert_lerobot.check_canon_fingerprint), so a recording made under another canon stops there.
        self._fingerprint = camera_canon.recording_fingerprint(cameras_canon)
        self._cams = dict(cameras_canon["cameras"])
        box = cameras_canon["bbox"]
        # The box is computed from bbox.camera's segmentation, at that camera's size.
        self._seg_cam = box["camera"]
        self._seg_shape = (self._cams[self._seg_cam]["height"], self._cams[self._seg_cam]["width"])
        union = list(box["classes_union"])
        self._pack_path = self._pack_part = None
        if self.pack is None:
            # No mask to keep: the box's classes alone, numbered from 1. Everything else -- robot,
            # rack, unlabelled -- is background, as it is to the box in a pack's mask for a prim with
            # one label. Not for a prim with several (remap_mask takes the first the classes know): a
            # "rack,br-04" prim is the rack in a pack's table, outside the box, and here BR-04, inside
            # it. Every label measured so far is single (2026-09-22/23).
            self._classes = {str(i): name for i, name in enumerate(union, start=1)}
        else:
            pack = cameras_canon["packs"][self.pack]
            # A pack's depth is its camera's (cameras.load puts the pack on bbox.camera: the mask
            # stored beside it is the box's segmentation).
            self._depth_cam = pack["camera"]
            self._depth_shape = (self._cams[self._depth_cam]["height"], self._cams[self._depth_cam]["width"])
            self._classes = pack["mask"]["classes"]
            unknown = set(union) - set(self._classes.values())
            if unknown:
                raise ValueError(f"bbox.classes_union names classes the {self.pack} mask does not have: "
                                 f"{sorted(unknown)}")
            self._pack_path = self.media_dir / f"{episode_id}.{self.pack}.pack.npz"
            self._pack_part = self.media_dir / f".{episode_id}.{self.pack}.pack.part.npz"
        self._bbox_ids = [int(i) for i, name in self._classes.items() if name in union]
        self._videos = {
            cam: Mp4Writer(self.media_dir / f".{episode_id}.{cam}.part.mkv", width=c["width"], height=c["height"],
                           fps=cameras_canon["fps"], open_sink=open_sink or _lossless_sink,
                           count_frames=count_frames)
            for cam, c in self._cams.items()}
        self._depth: list[np.ndarray] = []
        self._mask: list[np.ndarray] = []
        self._bbox: list[np.ndarray] = []
        self._poses: dict[str, list[np.ndarray]] = {cam: [] for cam in self._cams}
        self._rerendered: list[int] = []
        self._blank: dict[str, int] = {cam: 0 for cam in self._cams}
        self._counts: dict[str, int] | None = None
        self.frames = 0

    def step(self, frames: dict, seg: np.ndarray, id_to_labels: dict, step_index: int, poses: dict,
             rerendered: int = 0, depth: np.ndarray | None = None) -> None:
        """``seg`` is bbox.camera's segmentation; ``depth`` the pack camera's, exactly when the canon
        declares a pack (camera_io.Rig reads it only then)."""
        # Everything is checked before anything is written, so a refused step leaves no partial frame.
        if step_index != self.frames:
            raise ValueError(f"step_index {step_index} does not follow on from {self.frames} frames written "
                             "-- one step() per recorded step")
        for what, got in (("frame", frames), ("pose", poses)):
            missing = [cam for cam in self._cams if cam not in got]
            if missing:
                raise ValueError(f"step {step_index}: no {what} for {missing}")
        for cam, c in self._cams.items():
            f = np.asarray(frames[cam])
            if f.dtype != np.uint8 or f.shape != (c["height"], c["width"], 3):
                raise ValueError(f"step {step_index}: {cam} frame is {f.shape} {f.dtype}, "
                                 f"expected ({c['height']}, {c['width']}, 3) uint8")
        if np.shape(seg) != self._seg_shape:
            raise ValueError(f"step {step_index}: seg is {np.shape(seg)}, "
                             f"bbox.camera {self._seg_cam} is {self._seg_shape}")
        if self.pack is None:
            if depth is not None:
                # Dropping it would hide that whoever read it (camera_io.Rig) read another canon.
                raise ValueError(f"step {step_index}: a depth was handed over, but meta/cameras.json "
                                 "declares no pack to put it in")
        elif depth is None:
            raise ValueError(f"step {step_index}: no depth for the {self.pack} pack")
        elif np.shape(depth) != self._depth_shape:
            raise ValueError(f"step {step_index}: depth is {np.shape(depth)}, "
                             f"the {self.pack} pack's camera {self._depth_cam} is {self._depth_shape}")
        pose_rows = {}
        for cam in self._cams:
            p = np.asarray(poses[cam], np.float32)
            if p.shape != (7,) or not np.isfinite(p).all():
                raise ValueError(f"step {step_index}: the {cam} pose must be 7 finite numbers, got {p!r}")
            pose_rows[cam] = p

        mask = remap_mask(np.asarray(seg), decode_labels(id_to_labels), self._classes)
        for cam in self._cams:
            self._videos[cam].write(frames[cam])
            # Counted, not refused: the converter refuses a camera that recorded nothing else.
            if is_black(frames[cam]):
                self._blank[cam] += 1
        if self.pack is not None:
            self._depth.append(depth_to_mm(depth))
            self._mask.append(mask)
        self._bbox.append(union_bbox(mask, self._bbox_ids))
        for cam, p in pose_rows.items():
            self._poses[cam].append(p)
        if rerendered:
            self._rerendered.append(step_index)     # its first frame came back unchanged (camera_io.Rig)
        self.frames += 1

    def close(self) -> dict:
        """Finish every stream and stage it; replace nothing yet. Returns what the recorder stores
        (file entries named as they will be once committed, the steps rendered again, per camera the
        frames whose every pixel is 0, the canon's fingerprint, the bbox and pose arrays). The pack's
        entry is there only when the canon declares a pack.

        The episode's own files are written between close() and commit() (run_s1a): media put in
        place first would sit beside an earlier run's npz and sidecar if that save failed.
        A stream that fails here -- its file's frame count comes back
        wrong, the pack cannot be written -- removes this run's unfinished files and raises."""
        counts: dict[str, int] = {}
        failure: BaseException | None = None
        for cam, video in self._videos.items():      # close every sink, even after one has failed
            try:
                counts[cam] = video.close()
            except BaseException as exc:              # noqa: BLE001 -- re-raised below
                failure = failure or exc
        if failure is None and self.frames and self.pack is not None:
            try:
                self.media_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(self._pack_part, depth=np.stack(self._depth), mask=np.stack(self._mask),
                                    step_index=np.arange(self.frames, dtype=np.int32))
            except BaseException as exc:              # noqa: BLE001 -- re-raised below
                failure = exc
        if failure is not None:
            self.discard()
            raise failure
        self._counts = counts

        media: dict = {}
        for cam in self._videos:
            media[cam] = ({"file": f"{MEDIA_DIR}/{self._final(cam).name}", "frames": counts[cam]} if counts[cam]
                          else {"file": None, "frames": 0})
        if self.pack is not None:
            media["pack"] = ({"file": f"{MEDIA_DIR}/{self._pack_path.name}", "frames": self.frames} if self.frames
                             else {"file": None, "frames": 0})
        media["rerendered_steps"] = list(self._rerendered)
        media["blank_frames"] = dict(self._blank)          # camera -> frames whose every pixel is 0
        media["canon_fingerprint"] = self._fingerprint     # sha256 of what the pixels were baked from
        media["bbox"] = np.array(self._bbox, np.int32).reshape(-1, 4)
        media["poses"] = {cam: np.array(rows, np.float32).reshape(-1, 7) for cam, rows in self._poses.items()}
        return media

    def commit(self) -> None:
        """Put the staged streams in place, and remove what an earlier run left under this episode id
        for any stream this run did not record. Call after the episode's own files are written."""
        if self._counts is None:
            raise RuntimeError("commit() before close()")
        for cam, video in self._videos.items():
            if self._counts[cam]:
                video.path.replace(self._final(cam))
            else:
                self._final(cam).unlink(missing_ok=True)
        if self.pack is None:
            # This run records no pack; one an earlier run left under this id (recorded while the
            # canon declared it) would sit beside this run's videos, named by nothing.
            for stale in self.media_dir.glob(f"{self.episode_id}.*.pack.npz"):
                stale.unlink()
        elif self.frames:
            self._pack_part.replace(self._pack_path)
        else:
            self._pack_path.unlink(missing_ok=True)

    def discard(self) -> None:
        """Remove this run's staged or unfinished files; an earlier run's stay as they are."""
        for video in self._videos.values():
            video.path.unlink(missing_ok=True)
        if self._pack_part is not None:
            self._pack_part.unlink(missing_ok=True)

    def _final(self, cam: str) -> Path:
        return self.media_dir / f"{self.episode_id}.{cam}.mkv"


def clear_media(out_dir, episode_id: str) -> None:
    """Remove one episode's media -- videos, pack and unfinished part files -- so an episode saved
    without cameras never sits beside an earlier run's. Other episodes' files stay."""
    media = Path(out_dir) / MEDIA_DIR
    if not media.is_dir():
        return
    for path in media.iterdir():
        if path.name.startswith(f"{episode_id}.") or path.name.startswith(f".{episode_id}."):
            path.unlink()


def split_media(media: dict) -> tuple[dict[str, np.ndarray], dict]:
    """close()'s result -> (npz arrays, sidecar entry). The per-step arrays go into the episode
    npz beside the other step fields, as bbox_human and camera_pose_<cam>; the sidecar keeps the
    rest -- the entries that name files, the recorder's two counts (rerendered_steps, blank_frames)
    and the canon's fingerprint (canon_fingerprint)."""
    rest = dict(media)
    arrays = {"bbox_human": np.asarray(rest.pop("bbox"), np.int32)}
    for cam, rows in rest.pop("poses").items():
        arrays[f"camera_pose_{cam}"] = np.asarray(rows, np.float32)
    return arrays, rest
