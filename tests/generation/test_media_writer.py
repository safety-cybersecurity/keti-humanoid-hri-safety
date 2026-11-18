"""generation/envs/s1a/media_writer.py — camera frames to lossless video, the dummy's box and the
camera poses to arrays, and -- only when the camera canon declares one -- the pack camera's depth and
mask to a pack. The repository canon declares none (the pack code stays, the canon
switches it off), so the writer is tested both ways: _canon() declares no pack,
_canon(with_pack=True) today's pack shape. Isaac-free: the encoder is a stand-in with
watch.Mp4Writer's protocol (send bytes, close, count the file's frames back)."""
from pathlib import Path

import numpy as np
import pytest

from generation.envs.s1a import media_writer as mw

CLASSES = {"0": "background", "1": "BR-01", "4": "BR-04", "13": "robot", "14": "rack"}
# The shape Isaac Lab hands back in camera.data.info["semantic_segmentation"]["idToLabels"]
# (measured 2026-09-22/23): RGBA-string keys, {"class": ...} values, lower-case labels.
RAW = {"(0, 0, 0, 0)": {"class": "BACKGROUND"}, "(0, 0, 0, 255)": {"class": "UNLABELLED"},
       "(33, 243, 3, 255)": {"class": "br-04"}, "(240, 4, 111, 255)": {"class": "rack"}}
BR04 = -16518367       # (33, 243, 3, 255) packed as the int32 image stores it


def test_decode_labels_packs_the_colour_key_like_the_int32_image():
    assert mw.decode_labels(RAW) == {0: "BACKGROUND", -16777216: "UNLABELLED", BR04: "br-04", -9501456: "rack"}


def test_remap_mask_matches_classes_case_insensitively():
    seg = np.array([[BR04, -9501456], [0, -16777216]], dtype=np.int32)
    out = mw.remap_mask(seg, mw.decode_labels(RAW), CLASSES)
    assert out.tolist() == [[4, 14], [0, 0]] and out.dtype == np.uint8


def test_remap_mask_ignores_case_on_the_label_side_too():
    # the canon's names are upper-case and this Replicator lower-cases labels; neither side's case may matter
    seg = np.array([[7, 8]], dtype=np.int32)
    assert mw.remap_mask(seg, {7: "BR-01", 8: "RaCk"}, CLASSES).tolist() == [[1, 14]]


def test_remap_mask_takes_the_first_label_the_classes_know():
    # one prim can carry several labels; Replicator joins them with commas
    seg = np.array([[5, 6]], dtype=np.int32)
    assert mw.remap_mask(seg, {5: "probe,br-01", 6: "rack,br-04"}, CLASSES).tolist() == [[1, 14]]


def test_remap_mask_leaves_an_id_the_frame_does_not_list_as_background():
    seg = np.array([[12345]], dtype=np.int32)
    assert mw.remap_mask(seg, mw.decode_labels(RAW), CLASSES).tolist() == [[0]]


def test_union_bbox_comes_from_the_mask():
    mask = np.zeros((8, 10), np.uint8)
    mask[2:5, 3:6] = 1
    mask[6, 8] = 4
    mask[0, 0] = 14
    box = mw.union_bbox(mask, [1, 4])
    assert box.tolist() == [3, 2, 8, 6] and box.dtype == np.int32    # pixel indices, both ends in (0 ≤ x < width)
    assert mw.union_bbox(np.zeros((8, 10), np.uint8), [1]).tolist() == [-1, -1, -1, -1]


def test_depth_to_mm_clips_and_zeroes_invalid():
    d = np.array([[0.5, np.inf], [70.0, np.nan]], dtype=np.float32)
    out = mw.depth_to_mm(d)
    assert out.tolist() == [[500, 0], [65535, 0]] and out.dtype == np.uint16


# ---- the writer ----------------------------------------------------------------------------

def _canon(with_pack: bool = False):
    # 16-pixel multiples: watch.Mp4Writer refuses sizes the encoder would resize. The box is computed
    # from bbox.camera's segmentation; a declared pack sits on that same camera (cameras.load).
    canon = {"fps": 50,
             "cameras": {"head_rgb": {"width": 16, "height": 16}, "third_person_rgb": {"width": 32, "height": 16}},
             "bbox": {"camera": "head_rgb", "classes_union": ["BR-01", "BR-04"], "invalid": [-1, -1, -1, -1]}}
    if with_pack:
        canon["packs"] = {"head": {"camera": "head_rgb",
                                   "depth": {"dtype": "uint16", "unit": "mm", "invalid": 0,
                                             "source": "distance_to_image_plane"},
                                   "mask": {"dtype": "uint8", "classes": CLASSES}}}
    return canon


@pytest.mark.parametrize("path,value", [
    (("packs", "head", "depth", "dtype"), "float32"),
    (("packs", "head", "depth", "unit"), "m"),
    (("packs", "head", "depth", "invalid"), 65535),
    (("packs", "head", "mask", "dtype"), "uint16"),
    (("bbox", "invalid"), [0, 0, 0, 0]),
])
def test_writer_refuses_a_canon_that_describes_another_format(tmp_path, path, value):
    # meta/cameras.json ships as the reader's description of the pack and the box; it has to be
    # what the writer writes (CONTRIBUTING.md 「함께 고치는 곳」)
    canon = _canon(with_pack=True)
    node = canon
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match=path[-2] if path[-1] in ("dtype", "unit") else path[-1]):
        mw.MediaWriter(tmp_path, "S1d_0000_seed1", canon)


def test_a_writer_without_a_pack_still_refuses_another_box_format(tmp_path):
    # The box is written with or without a pack: only the pack's formats wait for a pack to be declared.
    canon = _canon()
    canon["bbox"]["invalid"] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="bbox.invalid"):
        mw.MediaWriter(tmp_path, "S1d_0000_seed1", canon)


def test_writer_accepts_the_real_canon(tmp_path):
    from generation import cameras
    repo = Path(__file__).resolve().parents[2]
    mw.MediaWriter(tmp_path, "S1d_0000_seed1", cameras.load(repo / "meta/cameras.json"))


def test_writer_accepts_the_real_canon_with_the_fixture_s_pack(tmp_path):
    # The pack code stays. The pack the fixture and the consumer tests declare
    # (make_fixture.canon_with_pack, a literal) has to be one this writer writes, or the pack side
    # the validator and the converter are tested on would describe a pack nobody records. The
    # fixture needs pandas: this runs in the .venv (CI) and skips in Isaac python, which runs this
    # file for the encoder test below.
    fixture = pytest.importorskip("validator.make_fixture")
    from generation import cameras
    repo = Path(__file__).resolve().parents[2]
    w = mw.MediaWriter(tmp_path, "S1d_0000_seed1", fixture.canon_with_pack(cameras.load(repo / "meta/cameras.json")))
    assert w.pack == fixture.PACK_NAME


class FakeSink:
    def __init__(self, path):
        Path(path).write_bytes(b"")            # the encoder creates its file
        self.frames = []

    def send(self, data):
        self.frames.append(len(data))

    def close(self):
        pass


def _writer(tmp_path, sinks, with_pack=False, canon=None):
    def open_sink(path, size, fps):
        sinks[Path(path).name] = FakeSink(path)
        return sinks[Path(path).name]

    return mw.MediaWriter(tmp_path, "S1d_0000_seed1", canon if canon is not None else _canon(with_pack),
                          open_sink=open_sink, count_frames=lambda path: len(sinks[Path(path).name].frames))


def _frames():
    return {"head_rgb": np.zeros((16, 16, 3), np.uint8), "third_person_rgb": np.zeros((16, 32, 3), np.uint8)}


def _poses(i):
    return {"head_rgb": np.arange(7, dtype=np.float32) + i, "third_person_rgb": np.zeros(7, np.float32)}


def _step(w, i, seg_value=0, frames=None, **over):
    # What camera_io.Rig hands over with a pack declared; a step without one passes depth=None.
    kw = dict(depth=np.full((16, 16), 1.5, np.float32), seg=np.full((16, 16), seg_value, np.int32),
              id_to_labels=RAW, step_index=i, poses=_poses(i))
    kw.update(over)
    w.step(frames if frames is not None else _frames(), **kw)


def test_writer_writes_one_frame_per_step_to_every_camera_and_the_pack(tmp_path):
    sinks = {}
    w = _writer(tmp_path, sinks, with_pack=True)
    for i in range(3):
        _step(w, i, seg_value=BR04 if i == 1 else 0)
    media = w.close()
    w.commit()
    assert media["head_rgb"] == {"file": "media/S1d_0000_seed1.head_rgb.mkv", "frames": 3}
    assert media["third_person_rgb"] == {"file": "media/S1d_0000_seed1.third_person_rgb.mkv", "frames": 3}
    assert media["pack"] == {"file": "media/S1d_0000_seed1.head.pack.npz", "frames": 3}
    assert sorted(p.name for p in (tmp_path / "media").iterdir()) == [
        "S1d_0000_seed1.head.pack.npz", "S1d_0000_seed1.head_rgb.mkv", "S1d_0000_seed1.third_person_rgb.mkv"]
    assert [len(s.frames) for s in sinks.values()] == [3, 3]
    pack = np.load(tmp_path / media["pack"]["file"])
    assert pack["step_index"].tolist() == [0, 1, 2]
    assert pack["depth"].shape == (3, 16, 16) and pack["depth"].dtype == np.uint16 and int(pack["depth"][0, 0, 0]) == 1500
    assert pack["mask"].dtype == np.uint8 and int(pack["mask"][1, 0, 0]) == 4 and int(pack["mask"][0, 0, 0]) == 0
    assert media["bbox"].dtype == np.int32
    assert media["bbox"].tolist() == [[-1, -1, -1, -1], [0, 0, 15, 15], [-1, -1, -1, -1]]
    assert media["poses"]["head_rgb"].dtype == np.float32
    assert media["poses"]["head_rgb"].tolist() == [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8]]


def test_nothing_the_writer_leaves_beside_the_episodes_reads_as_an_episode(tmp_path):
    # The labeller, the converter, compare_with_batch and the other episode readers take every *.npz
    # in the episode directory for an episode; a pack there would be read as one.
    w = _writer(tmp_path, {}, with_pack=True)
    _step(w, 0)
    w.close()
    w.commit()
    assert not list(tmp_path.glob("*.npz")) and list((tmp_path / "media").glob("*.npz"))


def test_writer_lists_the_steps_that_had_to_be_rendered_again(tmp_path):
    # camera_io.Rig re-renders a step whose frame came back unchanged; the episode says which steps
    w = _writer(tmp_path, {}, with_pack=True)
    for i in range(4):
        _step(w, i, rerendered=2 if i == 2 else 0)
    assert w.close()["rerendered_steps"] == [2]


def test_writer_refuses_a_skipped_step(tmp_path):
    w = _writer(tmp_path, {}, with_pack=True)
    _step(w, 0)
    with pytest.raises(ValueError, match="step_index"):
        _step(w, 2)


def test_writer_refuses_a_step_missing_a_camera_pose(tmp_path):
    w = _writer(tmp_path, {}, with_pack=True)
    with pytest.raises(ValueError, match="third_person_rgb"):
        _step(w, 0, poses={"head_rgb": np.zeros(7, np.float32)})


def test_writer_refuses_a_step_missing_a_camera_frame(tmp_path):
    w = _writer(tmp_path, {}, with_pack=True)
    with pytest.raises(ValueError, match="third_person_rgb"):
        _step(w, 0, frames={"head_rgb": np.zeros((16, 16, 3), np.uint8)})


def test_writer_refuses_depth_that_is_not_the_pack_camera_s_size(tmp_path):
    w = _writer(tmp_path, {}, with_pack=True)
    with pytest.raises(ValueError, match="depth"):
        _step(w, 0, depth=np.ones((8, 8), np.float32))


def test_a_failed_close_leaves_an_earlier_run_s_files_as_they_were(tmp_path):
    # close() once replaced the head video before the third-person frame count could fail, leaving
    # an episode id with one new and two old files. Nothing is replaced until
    # every stream is complete, and nothing half-written is left behind.
    media = tmp_path / "media"
    media.mkdir()
    old = {"S1d_0000_seed1.head_rgb.mkv", "S1d_0000_seed1.third_person_rgb.mkv", "S1d_0000_seed1.head.pack.npz"}
    for name in old:
        (media / name).write_bytes(b"old")
    sinks = {}

    def open_sink(path, size, fps):
        sinks[Path(path).name] = FakeSink(path)
        return sinks[Path(path).name]

    def count_frames(path):             # the third-person file comes back one frame short
        n = len(sinks[Path(path).name].frames)
        return n - 1 if "third_person" in path else n

    w = mw.MediaWriter(tmp_path, "S1d_0000_seed1", _canon(with_pack=True), open_sink=open_sink,
                       count_frames=count_frames)
    for i in range(2):
        _step(w, i)
    with pytest.raises(RuntimeError):
        w.close()
    assert {p.name for p in media.iterdir()} == old
    assert all((media / name).read_bytes() == b"old" for name in old)


def _earlier_run(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    old = {"S1d_0000_seed1.head_rgb.mkv", "S1d_0000_seed1.third_person_rgb.mkv", "S1d_0000_seed1.head.pack.npz"}
    for name in old:
        (media / name).write_bytes(b"old")
    return media, old


def test_close_replaces_nothing_until_commit(tmp_path):
    # run_s1a saves the episode's npz and sidecar between close() and commit(); if that save fails,
    # the media must still be the earlier run's
    media, old = _earlier_run(tmp_path)
    w = _writer(tmp_path, {}, with_pack=True)
    for i in range(2):
        _step(w, i)
    w.close()
    assert all((media / name).read_bytes() == b"old" for name in old)
    w.commit()
    assert {p.name for p in media.iterdir()} == old
    assert all((media / name).read_bytes() != b"old" for name in old)


def test_discard_leaves_the_earlier_run_s_files_and_no_parts(tmp_path):
    media, old = _earlier_run(tmp_path)
    w = _writer(tmp_path, {}, with_pack=True)
    _step(w, 0)
    w.close()
    w.discard()
    assert {p.name for p in media.iterdir()} == old
    assert all((media / name).read_bytes() == b"old" for name in old)


def test_writer_with_no_steps_leaves_no_files(tmp_path):
    w = _writer(tmp_path, {}, with_pack=True)
    media = w.close()
    w.commit()
    assert media["head_rgb"] == {"file": None, "frames": 0} and media["pack"] == {"file": None, "frames": 0}
    assert list(tmp_path.iterdir()) == []
    assert media["bbox"].shape == (0, 4) and media["poses"]["head_rgb"].shape == (0, 7)


def test_writer_with_no_steps_removes_an_earlier_run_s_media(tmp_path):
    # an episode that recorded nothing must not sit beside a previous run's video or pack
    stale = ["S1d_0000_seed1.head_rgb.mkv", "S1d_0000_seed1.third_person_rgb.mkv", "S1d_0000_seed1.head.pack.npz"]
    (tmp_path / "media").mkdir()
    for name in stale:
        (tmp_path / "media" / name).write_bytes(b"old")
    w = _writer(tmp_path, {}, with_pack=True)
    w.close()
    w.commit()
    assert list((tmp_path / "media").iterdir()) == []


# ---- no pack: the repository canon ------------------------------------------

def test_a_writer_without_a_pack_writes_the_videos_and_the_box_and_no_pack(tmp_path):
    sinks = {}
    w = _writer(tmp_path, sinks)
    assert w.pack is None
    for i in range(3):
        _step(w, i, seg_value=BR04 if i == 1 else 0, depth=None)
    media = w.close()
    assert not list((tmp_path / "media").glob("*.npz"))          # not even a part file to commit
    w.commit()
    assert "pack" not in media
    assert media["head_rgb"] == {"file": "media/S1d_0000_seed1.head_rgb.mkv", "frames": 3}
    assert media["third_person_rgb"] == {"file": "media/S1d_0000_seed1.third_person_rgb.mkv", "frames": 3}
    assert sorted(p.name for p in (tmp_path / "media").iterdir()) == [
        "S1d_0000_seed1.head_rgb.mkv", "S1d_0000_seed1.third_person_rgb.mkv"]
    assert [len(s.frames) for s in sinks.values()] == [3, 3]
    # the box is still computed, from the segmentation, as it was beside a pack
    assert media["bbox"].dtype == np.int32
    assert media["bbox"].tolist() == [[-1, -1, -1, -1], [0, 0, 15, 15], [-1, -1, -1, -1]]
    assert media["poses"]["head_rgb"].tolist() == [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8]]
    # what recorder.save() puts in the sidecar: the streams, the two counts, the canon's fingerprint and no
    # pack entry
    arrays, sidecar = mw.split_media(media)
    assert sorted(sidecar) == ["blank_frames", "canon_fingerprint", "head_rgb", "rerendered_steps",
                               "third_person_rgb"]
    assert sorted(arrays) == ["bbox_human", "camera_pose_head_rgb", "camera_pose_third_person_rgb"]


@pytest.mark.parametrize("dark", list(_canon()["cameras"]))
def test_the_writer_counts_each_camera_s_frames_whose_every_pixel_is_0(tmp_path, dark):
    # A camera inside the robot's own body renders black -- the 2026-09-30
    # preview's wrist camera inside the forearm did. The sidecar says per camera how many frames had every
    # pixel 0, for the converter to refuse a camera that recorded nothing else. One lit pixel is not black.
    canon = _canon()
    w = _writer(tmp_path, {})
    for i in range(3):
        frames = {cam: np.full((c["height"], c["width"], 3), 9, np.uint8) for cam, c in canon["cameras"].items()}
        frames[dark] = np.zeros_like(frames[dark])
        if i == 1:
            frames[dark][-1, -1, 2] = 1
        _step(w, i, frames=frames, depth=None)
    media = w.close()
    assert media["blank_frames"] == {cam: 2 if cam == dark else 0 for cam in canon["cameras"]}
    assert all(type(n) is int for n in media["blank_frames"].values())     # the sidecar is JSON


def test_the_sidecar_carries_the_fingerprint_of_the_canon_it_recorded_with(tmp_path):
    # What the pixels were baked from travels with the recording, so the converter can
    # tell a recording made under another camera canon from one made under the canon it converts with.
    # The repository canon, one step: what run_s1a hands the writer, seven cameras and all.
    from generation import cameras
    repo = Path(__file__).resolve().parents[2]
    canon = cameras.load(repo / "meta/cameras.json")
    w = _writer(tmp_path, {}, canon=canon)
    box = canon["cameras"][canon["bbox"]["camera"]]
    w.step({cam: np.zeros((c["height"], c["width"], 3), np.uint8) for cam, c in canon["cameras"].items()},
           seg=np.zeros((box["height"], box["width"]), np.int32), id_to_labels=RAW, step_index=0,
           poses={cam: np.zeros(7, np.float32) for cam in canon["cameras"]})
    media = w.close()
    assert media["canon_fingerprint"] == cameras.recording_fingerprint(canon)
    assert isinstance(media["canon_fingerprint"], str) and len(media["canon_fingerprint"]) == 64
    # a string for the sidecar, not an array for the npz
    arrays, sidecar = mw.split_media(media)
    assert sidecar["canon_fingerprint"] == media["canon_fingerprint"]
    assert not any("fingerprint" in name for name in arrays)


def test_the_fingerprint_is_the_writer_s_own_canon_s_and_follows_a_pixel_edit(tmp_path):
    # Not a constant and not the repository's: the canon the writer was handed. An edit the pixels depend
    # on (a camera's field of view) gives another fingerprint; a note does not.
    import copy

    from generation import cameras

    def fingerprint(canon):
        w = _writer(tmp_path, {}, canon=canon)
        w.step(_frames(), depth=None, seg=np.zeros((16, 16), np.int32), id_to_labels=RAW, step_index=0,
               poses=_poses(0))
        return w.close()["canon_fingerprint"]

    canon = _canon()
    canon["cameras"]["head_rgb"]["fov_h_deg"] = 69.4
    base = fingerprint(canon)
    assert base == cameras.recording_fingerprint(canon)
    noted = copy.deepcopy(canon)
    noted["cameras"]["head_rgb"]["mount_note"] = "prose"
    assert fingerprint(noted) == base
    wider = copy.deepcopy(canon)
    wider["cameras"]["head_rgb"]["fov_h_deg"] = 90.0
    assert fingerprint(wider) != base


def test_without_a_pack_the_box_comes_from_the_bbox_camera_s_segmentation(tmp_path):
    # bbox.camera names the camera whose segmentation the box is computed from: moved here to the
    # third-person camera, 32 wide, so a box past column 15 can only be read in its picture.
    canon = _canon()
    canon["bbox"]["camera"] = "third_person_rgb"
    w = _writer(tmp_path, {}, canon=canon)
    seg = np.zeros((16, 32), np.int32)
    seg[3:5, 20:26] = BR04
    seg[0, 0] = -9501456                     # the rack: background to the box, as in the pack's mask
    _step(w, 0, seg=seg, depth=None)
    with pytest.raises(ValueError, match="bbox.camera third_person_rgb"):
        _step(w, 1, seg=np.zeros((16, 16), np.int32), depth=None)       # the head camera's size
    assert w.close()["bbox"].tolist() == [[20, 3, 25, 4]]


def test_a_writer_without_a_pack_refuses_depth(tmp_path):
    # Depth handed to a writer with no pack to put it in: the rig and the writer read different
    # canons. Dropping it would hide that.
    w = _writer(tmp_path, {})
    with pytest.raises(ValueError, match="no pack"):
        _step(w, 0)                                                    # _step hands a depth by default
    assert w.frames == 0


def test_a_writer_with_a_pack_refuses_a_step_without_depth(tmp_path):
    # depth became optional in the signature; with a pack declared it is still required
    w = _writer(tmp_path, {}, with_pack=True)
    with pytest.raises(ValueError, match="depth"):
        _step(w, 0, depth=None)
    assert w.frames == 0


def test_a_writer_without_a_pack_and_no_steps_leaves_no_files(tmp_path):
    w = _writer(tmp_path, {})
    media = w.close()
    w.commit()
    assert "pack" not in media and media["head_rgb"] == {"file": None, "frames": 0}
    assert list(tmp_path.iterdir()) == []


def test_a_writer_without_a_pack_removes_an_earlier_run_s_pack_on_commit(tmp_path):
    # An earlier run under this episode id recorded while the canon still declared the pack. This run
    # records no pack stream, and commit() removes what an earlier run left for a stream this run did
    # not record -- the pack too, not only the videos. Other episodes' packs stay, and so does one
    # whose id only extends this one.
    media, old = _earlier_run(tmp_path)
    others = ["S1d_0001_seed2.head.pack.npz", "S1d_0000_seed12.head.pack.npz"]
    for name in others:
        (media / name).write_bytes(b"other")
    w = _writer(tmp_path, {})
    _step(w, 0, depth=None)
    w.close()
    assert (media / "S1d_0000_seed1.head.pack.npz").read_bytes() == b"old"      # nothing before commit
    w.commit()
    assert sorted(p.name for p in media.iterdir()) == sorted(
        ["S1d_0000_seed1.head_rgb.mkv", "S1d_0000_seed1.third_person_rgb.mkv", *others])


def test_clear_media_removes_one_episode_s_media_and_nothing_else(tmp_path):
    # An episode saved without cameras must not sit beside an earlier run's videos and pack
    media = tmp_path / "media"
    media.mkdir()
    mine = ["S1d_0000_seed1.head_rgb.mkv", "S1d_0000_seed1.head.pack.npz", ".S1d_0000_seed1.head_rgb.part.mkv"]
    other = ["S1d_0001_seed2.head_rgb.mkv", "S1d_00000_seed1.head_rgb.mkv",
             "S1d_0000_seed12.head_rgb.mkv", ".S1d_0000_seed12.head_rgb.part.mkv"]   # an id that extends this one
    for name in mine + other:
        (media / name).write_bytes(b"x")
    mw.clear_media(tmp_path, "S1d_0000_seed1")
    assert sorted(p.name for p in media.iterdir()) == sorted(other)
    mw.clear_media(tmp_path / "nowhere", "S1d_0000_seed1")          # no media directory: nothing to do


def test_the_default_encoder_is_lossless(tmp_path):
    # The intermediate video is lossless so the converter's encode is the only lossy one.
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    rng = np.random.default_rng(0)
    canon = _canon(with_pack=True)
    w = mw.MediaWriter(tmp_path, "S1d_0000_seed1", canon)
    sent = []
    for i in range(3):
        frames = {"head_rgb": rng.integers(0, 256, (16, 16, 3), dtype=np.uint8),
                  "third_person_rgb": rng.integers(0, 256, (16, 32, 3), dtype=np.uint8)}
        sent.append(frames["head_rgb"])
        _step(w, i, frames=frames)
    media = w.close()
    w.commit()
    reader = ffmpeg.read_frames(str(tmp_path / media["head_rgb"]["file"]), pix_fmt="rgb24")
    meta = next(reader)
    got = [np.frombuffer(f, np.uint8).reshape(meta["size"][1], meta["size"][0], 3) for f in reader]
    assert len(got) == 3 and all(np.array_equal(a, b) for a, b in zip(got, sent))


# ---- what the recorder stores (recorder.py imports Isaac, so the split is tested here) -------

def test_split_media_writes_the_npz_names_the_field_map_expects():
    # The converter and C9 read the camera arrays by generation/field_map.py's npz names;
    # recorder.save(media=...) stores them through this split. One list, not two -- kept
    # here rather than in test_field_map.py so that file imports nothing under
    # generation/envs/ (it has to run where that directory is left out).
    import json
    from generation.field_map import CAMERAS, STEP_FIELDS
    cams = json.loads((Path(__file__).resolve().parents[2] / "meta" / "cameras.json")
                      .read_text(encoding="utf-8"))
    media = {"bbox": np.full((2, len(cams["bbox"]["invalid"])), -1, np.int32),
             "poses": {cam: np.zeros((2, 7), np.float32) for cam in cams["cameras"]}}
    arrays, _ = mw.split_media(media)
    assert set(arrays) == {f.npz for f in STEP_FIELDS if f.condition == CAMERAS and f.kind == "array"}


def test_split_media_moves_the_arrays_to_the_npz_and_keeps_files_for_the_sidecar():
    media = {"head_rgb": {"file": "e.head_rgb.mkv", "frames": 2},
             "third_person_rgb": {"file": "e.third_person_rgb.mkv", "frames": 2},
             "pack": {"file": "e.head.pack.npz", "frames": 2},
             "blank_frames": {"head_rgb": 0, "third_person_rgb": 1},
             "canon_fingerprint": "ab" * 32,
             "bbox": np.array([[1, 2, 3, 4], [-1, -1, -1, -1]], np.int32),
             "poses": {"head_rgb": np.ones((2, 7), np.float32), "third_person_rgb": np.zeros((2, 7), np.float32)}}
    arrays, sidecar = mw.split_media(media)
    assert sorted(arrays) == ["bbox_human", "camera_pose_head_rgb", "camera_pose_third_person_rgb"]
    assert arrays["bbox_human"].tolist() == [[1, 2, 3, 4], [-1, -1, -1, -1]]
    assert arrays["camera_pose_head_rgb"].shape == (2, 7)
    assert sidecar == {"head_rgb": {"file": "e.head_rgb.mkv", "frames": 2},
                       "third_person_rgb": {"file": "e.third_person_rgb.mkv", "frames": 2},
                       "pack": {"file": "e.head.pack.npz", "frames": 2},
                       "blank_frames": {"head_rgb": 0, "third_person_rgb": 1},
                       "canon_fingerprint": "ab" * 32}
