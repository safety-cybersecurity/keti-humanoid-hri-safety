"""Camera media: what the validator requires when a distribution ships meta/cameras.json.

The camera fields are conditional (schema §6 「조건부 필드 — 카메라」): a distribution that
ships meta/cameras.json must carry all of them, one that does not carries none. These tests
pin the validator's side of that contract, starting with the shipped schema's own statement
of it.

The registered variants (broken_media_*) are run through the CLI by
test_fixture_detection_reason.py. Here each error branch of the camera checks gets its own
in-process case on a copy of a clean camera fixture, asserting the one sentence it exists to
say -- one variant per check would leave the branches beside it deletable with make check green.

There are three clean camera fixtures. clean_media declares no pack, as the repository canon
does: the packs are switched off in the canon.
clean_media_pack declares one: the pack code stays, so the side of every pack check that passes
a good pack stays tested too -- with the canon's pack gone, nothing else would run it.
clean_hri_media carries the camera media on the hri distribution, the shape of a release of the new
scenarios: every camera beside a visible-fraction column per camera on the robot.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.validator.test_fixture_detection_reason import EXPECTED_CHECK, EXPECTED_PHRASE
from validator import make_fixture as F
from validator import validate_dataset as V

REPO_ROOT = Path(__file__).resolve().parents[2]
CANON_SCHEMA = REPO_ROOT / "meta" / "safety_schema.json"
KEY = "conditional_required_step_fields"
CAMERA_FIELDS = json.loads(CANON_SCHEMA.read_text(encoding="utf-8"))[KEY]["cameras"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The four clean fixtures, built once in-process: camera-less, camera, camera with a pack, and the hri
    distribution with cameras."""
    root = tmp_path_factory.mktemp("fixtures")
    F.build(root / "clean")
    F.build(root / F.MEDIA_CLEAN, with_media=True)
    F.build(root / F.MEDIA_PACK_CLEAN, with_media=True, with_pack=True)
    F.build(root / F.HRI_MEDIA_CLEAN, with_media=True, hri=True)
    return root


CAMERA_BASES = [F.MEDIA_CLEAN, F.MEDIA_PACK_CLEAN, F.HRI_MEDIA_CLEAN]


def _episodes(root: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p)
                      for p in sorted((root / "meta" / "episodes").rglob("*.parquet"))],
                     ignore_index=True)


def _data_columns(root: Path) -> set[str]:
    return {c for p in sorted((root / "data").rglob("*.parquet"))
            for c in pd.read_parquet(p).columns}


def test_a_build_without_media_carries_no_camera_media(built):
    # The default fixture is the camera-less distribution (v0.1's shape); the camera checks'
    # "no lines at all without meta/cameras.json" is only a claim if this stays true.
    root = built / "clean"
    for rel in ("meta/cameras.json", "meta/packs.json", "videos", "packs"):
        assert not (root / rel).exists(), rel
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    # Its features are the step columns', and none of those is a camera's.
    assert "video_path" not in info
    assert not [name for name, feature in info["features"].items()
                if feature["dtype"] == "video" or name in CAMERA_FIELDS]
    assert not (_data_columns(root) & set(CAMERA_FIELDS))
    assert not [c for c in _episodes(root).columns if c.startswith("videos/")]


@pytest.mark.parametrize("base", CAMERA_BASES)
def test_the_media_fixture_carries_the_layout_a_release_does(built, base):
    """The layout the converter really writes (spec §4), derived from the canon.

    Streams: a LeRobot video feature per camera, its files under videos/<field>/, the
    episode's span in four meta/episodes columns. Packs, only where the canon declares one:
    packs/<name>/episode_XXXXXX.npz per episode, listed in meta/packs.json --
    clean_media declares none and ships neither. Box and poses: parquet columns.
    """
    root = built / base
    cameras = json.loads((root / "meta" / "cameras.json").read_text(encoding="utf-8"))
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    eps = _episodes(root)
    videos = [f for f in CAMERA_FIELDS if f.startswith(F.VIDEO_FIELD_PREFIX)]
    assert videos, "정본에 영상 필드가 없습니다 — 이 픽스처가 시험할 것이 없습니다"
    for key in videos:
        spec = cameras["cameras"][key.rsplit(".", 1)[-1]]
        feature = info["features"][key]
        assert feature["dtype"] == "video"
        assert feature["shape"] == [spec["height"], spec["width"], 3]
        for col in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            assert f"videos/{key}/{col}" in eps.columns, (key, col)
        for ci, fi in zip(eps[f"videos/{key}/chunk_index"], eps[f"videos/{key}/file_index"]):
            path = root / info["video_path"].format(video_key=key, chunk_index=ci, file_index=fi)
            assert path.is_file() and path.stat().st_size > 0, path
        # Episodes run back to back in more than one file, as a release rolls over.
        assert eps[f"videos/{key}/file_index"].nunique() >= 2
    if base == F.MEDIA_PACK_CLEAN:
        listed = json.loads((root / "meta" / "packs.json").read_text(encoding="utf-8"))
        (name, _), = cameras["packs"].items()
        assert set(listed) == {str(i) for i in eps["episode_index"]}
        for idx, entry in listed.items():
            assert entry["file"] == f"packs/{name}/episode_{int(idx):06d}.npz"
            assert (root / entry["file"]).is_file()
    else:
        assert "packs" not in cameras
        assert not (root / "packs").exists() and not (root / "meta" / "packs.json").exists()
    assert set(CAMERA_FIELDS) - set(videos) <= _data_columns(root)


def test_the_media_fixture_keeps_the_pack_and_the_pictures_apart(built):
    # The fixture's reason to shrink the pictures unevenly: at the canon's identity downsample
    # and two equal cameras, "which size" and every_n_steps would be invisible to any test.
    cameras = json.loads((built / F.MEDIA_PACK_CLEAN / "meta" / "cameras.json").read_text(encoding="utf-8"))
    (_, pack), = cameras["packs"].items()
    native = cameras["cameras"][pack["camera"]]
    down = pack["downsample"]
    assert down["every_n_steps"] > 1
    assert (down["width"], down["height"]) != (native["width"], native["height"])
    sizes = {(c["width"], c["height"]) for c in cameras["cameras"].values()}
    assert len(sizes) == len(cameras["cameras"]) > 1


def test_the_media_fixture_videos_are_placeholders_behind_an_mp4_signature(built):
    """Still not a video -- nothing here can vouch for a picture. The bytes open with an
    ISO-BMFF ftyp box header, the MP4 signature C1 reads (and the only bytes of a stream it
    reads), and then say what they are."""
    paths = sorted((built / F.MEDIA_CLEAN / "videos").rglob("*.mp4"))
    assert paths
    for path in paths:
        data = path.read_bytes()
        assert data[4:8] == b"ftyp", path
        assert b"not a video" in data, path


def test_the_hri_camera_fixture_carries_a_visible_fraction_per_camera_on_the_robot(built):
    # Both conditions at once: the hri columns beside the camera ones, a fraction for each camera the shipped
    # camera canon puts on the robot (type mounted) and for no other.
    root = built / F.HRI_MEDIA_CLEAN
    schema = json.loads(CANON_SCHEMA.read_text(encoding="utf-8"))[KEY]
    assert set(schema["hri"]) <= _data_columns(root)
    cameras = json.loads((root / "meta" / "cameras.json").read_text(encoding="utf-8"))["cameras"]
    fractions = {c[len(V.VISIBLE_FRACTION_PREFIX):] for c in _data_columns(root)
                 if c.startswith(V.VISIBLE_FRACTION_PREFIX)}
    assert fractions == {name for name, cam in cameras.items() if cam["type"] == "mounted"}
    assert len(fractions) < len(cameras)


@pytest.mark.parametrize("base", CAMERA_BASES)
def test_the_media_fixture_passes_the_validator(built, base):
    rep = V.run(built / base, set(V.ALL_CHECKS), None, 3, None)
    errors = [f.line() for f in rep.findings if f.severity == V.ERROR]
    assert not errors, errors


def test_the_fixture_pack_is_one_the_camera_loader_accepts(tmp_path):
    """The pack is written into the fixture, not read from the repository canon (which declares
    none). It has to stay a pack the generator's loader would take: on the box's
    camera, its depth from a channel that camera records."""
    from generation import cameras as camera_canon
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(F.media_canon(with_pack=True)), encoding="utf-8")
    canon = camera_canon.load(path)
    (_, pack), = canon["packs"].items()
    assert pack["camera"] == canon["bbox"]["camera"]
    assert pack["depth"]["source"] in canon["cameras"][pack["camera"]]["channels"]
    # The class table the pack carried: background, the region codes (their canon), robot, rack.
    regions = sorted(json.loads(CANON_SCHEMA.read_text(encoding="utf-8"))["body_regions"])
    classes = pack["mask"]["classes"]
    assert [classes[str(i + 1)] for i in range(len(regions))] == regions
    assert classes["0"] == "background" and len(classes) == len(regions) + 3
    assert [classes[str(len(regions) + 1)], classes[str(len(regions) + 2)]] == ["robot", "rack"]


def test_the_default_media_canon_declares_no_pack():
    # With or without a pack in the repository canon, the default fixture canon has none.
    assert "packs" not in F.media_canon()


def test_a_pack_is_added_to_a_copy_not_to_the_canon_passed_in():
    """canon_with_pack gives the box's camera the pack's depth channel when the canon lacks it (as
    the repository canon does) -- on the copy it returns, never on its
    argument: generation tests pass it their own canon and go on using it."""
    import copy
    canon = F.media_canon()
    camera = canon["bbox"]["camera"]
    canon["cameras"][camera]["channels"] = [c for c in canon["cameras"][camera]["channels"]
                                            if c != F.PACK_DEPTH["source"]]
    before = copy.deepcopy(canon)
    packed = F.canon_with_pack(canon)
    assert canon == before
    assert F.PACK_DEPTH["source"] in packed["cameras"][camera]["channels"]
    assert packed["packs"][F.PACK_NAME]["camera"] == camera


def test_a_pack_fixture_without_media_is_refused(tmp_path):
    # A pack lives in a camera distribution; asked for alone, the build would quietly be clean.
    with pytest.raises(ValueError, match="with_pack 는 with_media 와 함께만"):
        F.build(tmp_path / "x", with_pack=True)
    assert not (tmp_path / "x").exists()


def _schema_findings(tmp_path: Path, mutate) -> list[V.Finding]:
    """C1's shipped-schema comparison on the repo canon after ``mutate``, as findings
    that mention the conditional step fields."""
    schema = json.loads(CANON_SCHEMA.read_text(encoding="utf-8"))
    mutate(schema)
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "safety_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False), encoding="utf-8")
    rep = V.Report(root=str(tmp_path))
    V._c1_safety_schema_sound(V.Dataset(tmp_path, rep), rep)
    return [f for f in rep.findings if KEY in f.message]


def test_the_canon_passes_the_conditional_step_comparison(tmp_path):
    # The control for the cases below: the same path on the unmutated canon says nothing.
    assert _schema_findings(tmp_path, lambda s: None) == []


# The comparison has three error branches. broken_canon_conditional_steps covers the set
# mismatch; these are the other two -- delete either branch and the shipped schema could drop
# the camera contract, or carry it in a form a consumer would read letter by letter.
@pytest.mark.parametrize("mutate", [
    pytest.param(lambda s: s.pop(KEY), id="absent"),
    pytest.param(lambda s: s.__setitem__(KEY, s[KEY]["cameras"]), id="list_not_object"),
    pytest.param(lambda s: s.__setitem__(KEY, None), id="null"),
])
def test_a_shipped_schema_without_the_conditional_object_is_refused(tmp_path, mutate):
    found = _schema_findings(tmp_path, mutate)
    assert [f.severity for f in found] == [V.ERROR], found
    assert f"규격 정의에 {KEY} 가 없거나 객체가 아님" in found[0].message
    assert found[0].check == "C1"


@pytest.mark.parametrize("mutate", [
    # A bare string would set() into its characters -- the one shape that silently
    # "matches" nothing while looking like a field name.
    pytest.param(lambda s: s[KEY].__setitem__("cameras", "observation.bbox2d.human"),
                 id="bare_string"),
    pytest.param(lambda s: s[KEY].__setitem__("cameras", s[KEY]["cameras"] + [7]),
                 id="non_string_member"),
    pytest.param(lambda s: s[KEY].__setitem__("cameras", {"observation.bbox2d.human": True}),
                 id="object_value"),
])
def test_a_conditional_value_that_is_not_a_string_list_is_refused(tmp_path, mutate):
    found = _schema_findings(tmp_path, mutate)
    assert [f.severity for f in found] == [V.ERROR], found
    assert f"규격 정의의 {KEY} 의 ['cameras'] 가 문자열 목록이 아님" in found[0].message


# ── The camera checks (C1 · C7 · C8) ────────────────────────────────────────

def test_the_validators_camera_constants_follow_the_canon():
    """The validator holds the box field, its marker, the pack's picture members and the
    video-field prefix as constants (it never imports generation code); each has to be what
    meta/cameras.json and the shipped schema say (CONTRIBUTING.md 「함께 고치는 곳」). The pack's
    members are the fixture pack's -- the one pack shape left, the repository canon having
    switched its pack off; the canon is not asked for a pack it need not declare."""
    cams = json.loads((REPO_ROOT / "meta" / "cameras.json").read_text(encoding="utf-8"))
    assert V.CAMERAS_CONDITION in V.CONDITIONAL_REQUIRED_STEP_FIELDS
    assert cams["bbox"]["field"] == V.BBOX_FIELD
    assert cams["bbox"]["invalid"] == list(V.BBOX_INVALID)
    assert V.BBOX_FIELD in V.camera_column_fields()
    (name, pack), = F.media_canon(with_pack=True)["packs"].items()
    assert set(V.PACK_PICTURES) == {k for k, v in pack.items()
                                    if isinstance(v, dict) and "dtype" in v}, name
    rgb = {f"{V.VIDEO_FIELD_PREFIX}{n}" for n, c in cams["cameras"].items() if "rgb" in c["channels"]}
    assert set(V.camera_video_fields()) == rgb
    assert F.VIDEO_FIELD_PREFIX == V.VIDEO_FIELD_PREFIX


CAMERA_WORDS = ("cameras.json", "packs.json", "packs/", "videos/", "영상", "팩",
                "observation.images", "bbox2d", "camera_pose", "카메라")


def test_a_camera_less_distribution_gets_no_camera_lines(built):
    """Without meta/cameras.json the camera checks say nothing at all -- not even SKIP.

    A release without cameras has to report as one did before these checks existed: the camera
    checks add no line to it, not even a SKIP.
    """
    rep = V.run(built / "clean", set(V.ALL_CHECKS), None, 3, None)
    said = [f.line() for f in rep.findings if any(w in f.line() for w in CAMERA_WORDS)]
    assert not said, said
    assert {f.check for f in rep.findings if f.severity == V.SKIP} <= {"C9"}
    assert "camera_media" not in rep.stats


@pytest.mark.parametrize("base", CAMERA_BASES)
def test_the_camera_checks_count_what_they_looked_at(built, base):
    """ERROR 0 on a clean camera fixture is only worth something if the checks looked at
    everything: every stream file, every pack (hashed and aligned -- none where the canon
    declares none), every span, every camera file's layout of spans, every box row."""
    root = built / base
    rep = V.run(root, set(V.ALL_CHECKS), None, 3, None)
    assert not [f.line() for f in rep.findings if f.severity == V.ERROR]
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    eps = _episodes(root)
    videos = [f for f in CAMERA_FIELDS if f.startswith(F.VIDEO_FIELD_PREFIX)]
    files = {info["video_path"].format(video_key=k, chunk_index=c, file_index=f)
             for k in videos
             for c, f in zip(eps[f"videos/{k}/chunk_index"], eps[f"videos/{k}/file_index"])}
    layouts = {(k, c, f) for k in videos
               for c, f in zip(eps[f"videos/{k}/chunk_index"], eps[f"videos/{k}/file_index"])}
    packs = _read_json(root, "meta/packs.json") if base == F.MEDIA_PACK_CLEAN else {}
    rows = sum(len(pd.read_parquet(p)) for p in (root / "data").rglob("*.parquet"))
    assert rep.stats["camera_media"] == {
        "video_files": len(files), "pack_files": len(packs),
        "video_spans": len(videos) * len(eps), "video_file_layouts": len(layouts),
        "packs_aligned": len(packs), "bbox_rows": rows}


def test_the_media_fixture_spans_carry_float_noise(built):
    # The C7 tolerance is exercised only if the fixture's spans are not exact: a validator
    # comparing frame counts with == has to fail on clean_media, as it would on S1d-p4.
    root = built / F.MEDIA_CLEAN
    fps = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))["fps"]
    eps = _episodes(root)
    key = [f for f in CAMERA_FIELDS if f.startswith(F.VIDEO_FIELD_PREFIX)][0]
    spans = (eps[f"videos/{key}/to_timestamp"] - eps[f"videos/{key}/from_timestamp"]) * fps
    assert (spans != float(int(round(spans.iloc[0])))).any()


@pytest.mark.parametrize("variant", sorted(v for v in F.VARIANTS
                                           if v.startswith(F.MEDIA_VARIANT_PREFIX)))
def test_each_media_variant_fails_only_on_its_defect(built, tmp_path, variant):
    """The base is green and the variant red with exactly the checks it is registered for --
    so the red comes from the defect it injects, not from the fixture around it."""
    base = built / F.variant_base(variant)
    assert not _errors(base), f"{base.name} 에 오류가 있습니다 — 변종의 빨간불이 결함 때문인지 알 수 없습니다"
    root = _copy(base, tmp_path)
    F.VARIANTS[variant](root)
    errors = _errors(root)
    want = EXPECTED_CHECK[variant]
    assert {f.check for f in errors} == ({want} if isinstance(want, str) else set(want)), \
        [f.line() for f in errors]
    phrases = EXPECTED_PHRASE[variant]
    for phrase in (phrases,) if isinstance(phrases, str) else phrases:
        assert any(phrase in f.message for f in errors), (phrase, [f.line() for f in errors])


def _copy(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    return dst


def _errors(root: Path, checks=V.ALL_CHECKS) -> list[V.Finding]:
    rep = V.run(root, set(checks), None, 3, None)
    return [f for f in rep.findings if f.severity == V.ERROR]


def _read_json(root: Path, rel: str):
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _edit_json(rel: str, fn):
    def mutate(root: Path) -> None:
        doc = _read_json(root, rel)
        fn(doc)
        (root / rel).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    return mutate


def _write_table(df: pd.DataFrame, path: Path) -> None:
    """A step table (under data/) keeps its NaN as a distribution's does -- df.to_parquet would write a null,
    which C7 reports on its own and the loader dies on -- so a case that plants a NaN tests the NaN alone.
    The other tables (episodes) have no NaN to keep."""
    if "data" in path.parts:
        F.write_steps(df, path)
    else:
        df.to_parquet(path, index=False)


def _edit_parquet(rel: str, fn):
    def mutate(root: Path) -> None:
        df = pd.read_parquet(root / rel)
        fn(df)
        _write_table(df, root / rel)
    return mutate


EPISODES = "meta/episodes/chunk-000/file-000.parquet"
DATA_0, DATA_1 = "data/chunk-000/file-000.parquet", "data/chunk-000/file-001.parquet"
VIDEOS = [f for f in CAMERA_FIELDS if f.startswith(F.VIDEO_FIELD_PREFIX)]
HEAD, LAST = VIDEOS[0], VIDEOS[-1]
POSES = [f for f in CAMERA_FIELDS if f not in VIDEOS and f != V.BBOX_FIELD]


def _rewrite_pack(idx: int, fn):
    """Change one pack's arrays and keep meta/packs.json's SHA-256 true to the new bytes,
    so the case is about alignment (C7) and not about the hash (C1)."""
    def mutate(root: Path) -> None:
        listed = _read_json(root, "meta/packs.json")
        path = root / listed[str(idx)]["file"]
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        fn(arrays)
        np.savez_compressed(path, **arrays)
        listed[str(idx)]["sha256"] = F.sha256_of(path)
        (root / "meta" / "packs.json").write_text(json.dumps(listed, indent=2), encoding="utf-8")
    return mutate


def _replace_pack_bytes(idx: int, data: bytes):
    def mutate(root: Path) -> None:
        listed = _read_json(root, "meta/packs.json")
        path = root / listed[str(idx)]["file"]
        path.write_bytes(data)
        listed[str(idx)]["sha256"] = F.sha256_of(path)
        (root / "meta" / "packs.json").write_text(json.dumps(listed, indent=2), encoding="utf-8")
    return mutate


def _duplicate_pack_key(root: Path) -> None:
    listed = list(_read_json(root, "meta/packs.json").items())
    text = ", ".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in listed + [listed[2]])
    (root / "meta" / "packs.json").write_text("{" + text + "}", encoding="utf-8")


def _both(*mutations):
    def mutate(root: Path) -> None:
        for m in mutations:
            m(root)
    return mutate


def _unlink(rel: str):
    return lambda root: (root / rel).unlink()


def _rmtree(rel: str):
    return lambda root: shutil.rmtree(root / rel)


def _set_box(row: int, box: list[int], rel: str = DATA_1):
    def fn(df):
        vals = [np.asarray(v, dtype=np.int32) for v in df[V.BBOX_FIELD]]
        vals[row] = np.asarray(box, dtype=np.int32)
        df[V.BBOX_FIELD] = vals
    return _edit_parquet(rel, fn)


def _second_camera(cams):
    (name, spec), = list(cams["cameras"].items())[:1]
    cams["cameras"]["spare_rgb"] = dict(spec)


def _swap_shape(info):
    f = info["features"][HEAD]
    f["shape"] = [f["shape"][1], f["shape"][0], 3]


def _eps_row(df, idx):
    return df.index[df["episode_index"] == idx][0]


# The fixture's pack and its camera's native size, for the downsample cases' sentences.
(PACK_NAME, _PACK), = F.media_canon(with_pack=True)["packs"].items()
PACK_CAMERA = _PACK["camera"]
PACK_NATIVE = {axis: F.media_canon(with_pack=True)["cameras"][PACK_CAMERA][axis]
               for axis in ("width", "height")}
# The camera the box is in (bbox.camera), and one that is not it.
BOX_CAMERA = F.media_canon()["bbox"]["camera"]
OTHER_CAMERA = next(n for n in F.media_canon()["cameras"] if n != BOX_CAMERA)
assert min(PACK_NATIVE.values()) >= 3, "native - 1 must be a size that does not divide native"


def _pack_downsample(axis: str, want: int):
    return _edit_json("meta/cameras.json",
                      lambda c: c["packs"][PACK_NAME]["downsample"].__setitem__(axis, want))


def _not_a_fraction(axis: str, want: int) -> str:
    return (f"팩 {PACK_NAME} 의 downsample.{axis} {want} 가 카메라 {PACK_CAMERA} 의 원 해상도 "
            f"{PACK_NATIVE[axis]} 의 정수 분의 일이 아님")


SPAN_COLUMNS = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")


def _copy_spans(src: int, dst: int):
    """Episode ``dst`` given episode ``src``'s place in every camera's files: both then point
    at the same pictures."""
    def fn(df):
        s, d = _eps_row(df, src), _eps_row(df, dst)
        for key in VIDEOS:
            for col in SPAN_COLUMNS:
                df.loc[d, f"videos/{key}/{col}"] = df.loc[s, f"videos/{key}/{col}"]
    return _edit_parquet(EPISODES, fn)


def _shift_spans(key: str, episodes, frames: float):
    """The spans of ``episodes`` in ``key``'s files moved by ``frames`` frames, lengths kept."""
    def fn(df):
        rows = df["episode_index"].isin(list(episodes))
        for end in ("from_timestamp", "to_timestamp"):
            col = f"videos/{key}/{end}"
            df.loc[rows, col] = df.loc[rows, col] + frames / F.FPS
    return _edit_parquet(EPISODES, fn)


def _swap_spans(key: str, a: int, b: int):
    def fn(df):
        ra, rb = _eps_row(df, a), _eps_row(df, b)
        for end in ("from_timestamp", "to_timestamp"):
            col = f"videos/{key}/{end}"
            df.loc[ra, col], df.loc[rb, col] = df.loc[rb, col], df.loc[ra, col]
    return _edit_parquet(EPISODES, fn)


CASES = [
    # C1 -- the condition file and the camera canon
    ("cameras_json_lost", _unlink("meta/cameras.json"), "C1",
     "meta/cameras.json 이 없는데 카메라 매체가 있음"),
    ("cameras_json_unreadable", lambda r: (r / "meta/cameras.json").write_text("{"), "C1",
     "JSON 파싱 실패"),
    ("cameras_json_unreadable_c7", lambda r: (r / "meta/cameras.json").write_text("{"), "C7",
     "meta/cameras.json 을 읽지 못해 팩의 스텝 간격"),
    ("cameras_json_unreadable_c8", lambda r: (r / "meta/cameras.json").write_text("{"), "C8",
     f"meta/cameras.json 을 읽지 못해 {V.BBOX_FIELD} 의 범위"),
    ("cameras_json_list", lambda r: (r / "meta/cameras.json").write_text("[]"), "C1",
     "meta/cameras.json 의 최상위가 객체가 아님: list"),
    # A canon that cannot say whether it declares a pack: said, and not read as "none" -- that
    # would call a real pack a stray one, or pass packs/ that nothing looks at.
    ("cameras_json_unreadable_packs", lambda r: (r / "meta/cameras.json").write_text("{"), "C1",
     "meta/cameras.json 을 읽지 못해 팩을 선언했는지 알 수 없음"),
    ("cameras_json_list_packs", lambda r: (r / "meta/cameras.json").write_text("[]"), "C1",
     "meta/cameras.json 을 읽지 못해 팩을 선언했는지 알 수 없음"),
    ("cameras_fps_zero", _edit_json("meta/cameras.json", lambda c: c.__setitem__("fps", 0)), "C1",
     "meta/cameras.json 의 fps 가 양수가 아님: 0"),
    ("cameras_empty", _edit_json("meta/cameras.json", lambda c: c.__setitem__("cameras", {})), "C1",
     "meta/cameras.json 의 cameras 가 비었거나 객체가 아님"),
    ("camera_unknown", _edit_json("meta/cameras.json", _second_camera), "C1",
     "가 규격의 영상 필드가 말하는 카메라"),
    ("camera_size", _edit_json("meta/cameras.json",
                               lambda c: next(iter(c["cameras"].values())).pop("width")), "C1",
     "의 width·height 가 양의 정수가 아님"),
    ("cameras_fps", _edit_json("meta/cameras.json", lambda c: c.__setitem__("fps", c["fps"] * 2)),
     "C1", "meta/cameras.json 의 fps"),
    # bbox.camera names the camera the box is in; C8 measures the box against it.
    ("bbox_camera_missing", _edit_json("meta/cameras.json", lambda c: c["bbox"].pop("camera")), "C1",
     "meta/cameras.json 의 bbox.camera None 가 cameras 의 카메라가 아님"),
    ("bbox_camera_unknown", _edit_json("meta/cameras.json",
                                       lambda c: c["bbox"].__setitem__("camera", "spare_rgb")), "C1",
     "meta/cameras.json 의 bbox.camera 'spare_rgb' 가 cameras 의 카메라가 아님"),
    # Unhashable, like pack_camera_list below: `in cameras` alone would raise TypeError (exit 3).
    ("bbox_camera_list", _edit_json("meta/cameras.json",
                                    lambda c: c["bbox"].__setitem__("camera", [BOX_CAMERA])), "C1",
     f"meta/cameras.json 의 bbox.camera {[BOX_CAMERA]!r} 가 cameras 의 카메라가 아님"),
    # Said, not guessed: without the box's camera C8 has no size to hold the box to.
    ("bbox_camera_missing_c8", _edit_json("meta/cameras.json", lambda c: c["bbox"].pop("camera")), "C8",
     f"meta/cameras.json 을 읽지 못해 {V.BBOX_FIELD} 의 범위를 대조하지 못함"),
    ("box_field", _edit_json("meta/cameras.json",
                             lambda c: c["bbox"].__setitem__("field", "observation.bbox2d.dummy")),
     "C1", f"bbox 의 field·invalid 가 규격의 {V.BBOX_FIELD}"),
    ("box_marker", _edit_json("meta/cameras.json",
                              lambda c: c["bbox"].__setitem__("invalid", [0, 0, 0, 0])),
     "C1", f"bbox 의 field·invalid 가 규격의 {V.BBOX_FIELD}"),
    # C1 -- the video features
    ("video_feature_missing", _edit_json("meta/info.json", lambda i: i["features"].pop(HEAD)),
     "C1", f"info.json 에 영상 피처 {HEAD} (dtype video) 가 없음"),
    ("video_feature_extra", _edit_json(
        "meta/info.json", lambda i: i["features"].__setitem__("observation.images.spare_rgb",
                                                              i["features"][HEAD])),
     "C1", "info.json 에 규격에 없는 영상 피처 ['observation.images.spare_rgb']"),
    ("video_shape", _edit_json("meta/info.json", _swap_shape), "C1",
     f"영상 피처 {HEAD} 의 shape"),
    ("video_info_missing", _edit_json("meta/info.json", lambda i: i["features"][LAST].pop("info")),
     "C1", f"영상 피처 {LAST} 에 info 가 없어"),
    ("video_fps", _edit_json("meta/info.json", lambda i: i["features"][LAST]["info"].__setitem__(
        "video.fps", i["fps"] / 2)), "C1", f"영상 피처 {LAST} 의 video.fps"),
    ("video_encoded_size", _edit_json("meta/info.json", lambda i: i["features"][HEAD]["info"].__setitem__(
        "video.width", i["features"][HEAD]["info"]["video.width"] + 2)), "C1",
     f"영상 피처 {HEAD} 의 video.width·height"),
    ("info_features_missing", _edit_json("meta/info.json", lambda i: i.pop("features")), "C1",
     "info.json 에 features 가 없어"),
    ("video_path_missing", _edit_json("meta/info.json", lambda i: i.pop("video_path")), "C1",
     "info.json 에 video_path 가 없어"),
    ("video_path_template", _edit_json("meta/info.json", lambda i: i.__setitem__(
        "video_path", "videos/{video_key}/{nope}.mp4")), "C1", "를 채울 수 없음: KeyError"),
    ("video_path_attribute", _edit_json("meta/info.json", lambda i: i.__setitem__(
        "video_path", "videos/{video_key.nope}/x.mp4")), "C1", "를 채울 수 없음: AttributeError"),
    ("video_path_subscript", _edit_json("meta/info.json", lambda i: i.__setitem__(
        "video_path", "videos/{chunk_index[0]}/x.mp4")), "C1", "를 채울 수 없음: TypeError"),
    ("video_path_outside", _edit_json("meta/info.json", lambda i: i.__setitem__(
        "video_path", "../" + i["video_path"])), "C1", "영상 경로가 배포본 밖을 가리킴"),
    # C1 -- the stream files
    ("video_dir_missing", _rmtree(f"videos/{HEAD}"), "C1", "영상 디렉터리 없음"),
    ("video_file_missing", _unlink(f"videos/{LAST}/chunk-000/file-001.mp4"), "C1",
     "영상 파일이 없거나 비어 있음 — 에피소드 5건"),
    ("video_file_empty", lambda r: (r / f"videos/{HEAD}/chunk-000/file-000.mp4").write_bytes(b""),
     "C1", "영상 파일이 없거나 비어 있음"),
    ("video_index_columns", _edit_parquet(EPISODES, lambda d: d.pop(f"videos/{HEAD}/file_index")), "C1",
     f"['videos/{HEAD}/file_index'] 열이 없어"),
    ("video_index_value", _edit_parquet(EPISODES, lambda d: d.__setitem__(
        f"videos/{LAST}/file_index", d[f"videos/{LAST}/file_index"].astype(float).where(
            d["episode_index"] != 4, 0.5))), "C1", "의 chunk_index·file_index 가 0 이상의 정수가 아님"),
    # Any non-empty file inside the distribution passed as a stream --
    # video_path "meta/info.json" made that JSON file both cameras' video and C1 said nothing.
    # A stream is an MP4 under its own camera's videos/<field>/, and two file numbers are two
    # files (a template that drops one sends the episodes of the other file to this one).
    ("video_path_not_a_stream", _edit_json("meta/info.json", lambda i: i.__setitem__(
        "video_path", "meta/info.json")), "C1",
     f"{HEAD} 의 영상 경로 meta/info.json 가 이 카메라의 디렉터리 videos/{HEAD}/ 밖을 가리킴"),
    ("video_path_other_camera", _edit_json("meta/info.json", lambda i: i.__setitem__(
        "video_path", i["video_path"].replace("{video_key}", HEAD))), "C1",
     f"{LAST} 의 영상 경로 videos/{HEAD}/chunk-000/file-000.mp4 가 이 카메라의 디렉터리 "
     f"videos/{LAST}/ 밖을 가리킴"),
    ("video_file_not_mp4", lambda r: (r / f"videos/{LAST}/chunk-000/file-001.mp4").write_bytes(
        b'{"not": "a video"}\n'), "C1", "영상 파일이 MP4 가 아님"),
    # The signature is the box type at offset 4, after the box size -- not "ftyp" anywhere.
    ("video_file_ftyp_misplaced", lambda r: (r / f"videos/{HEAD}/chunk-000/file-000.mp4").write_bytes(
        b"ftypisom\x00\x00\x02\x00 not a box header\n"), "C1", "영상 파일이 MP4 가 아님"),
    ("video_file_shorter_than_a_box_header", lambda r: (
        r / f"videos/{HEAD}/chunk-000/file-001.mp4").write_bytes(b"\x00\x00\x00\x10fty"), "C1",
     "영상 파일이 MP4 가 아님"),
    ("video_path_ignores_file_index", _both(
        _edit_json("meta/info.json", lambda i: i.__setitem__(
            "video_path", i["video_path"].replace("{file_index:03d}", "000"))),
        _unlink(f"videos/{HEAD}/chunk-000/file-001.mp4")), "C1",
     f"{HEAD} 의 파일 번호 (chunk_index, file_index) [(0, 0), (0, 1)] 가 모두 한 파일 "
     f"videos/{HEAD}/chunk-000/file-000.mp4 로 풀림"),
    # C1 -- packs the canon does not declare: work done on top of an
    # old conversion's copy. Each kind alone, and both as the registered variant leaves them.
    ("stray_pack", F.break_media_stray_pack, "C1", "정본이 팩을 선언하지 않았는데"),
    ("stray_packs_dir", lambda r: (r / "packs" / PACK_NAME).mkdir(parents=True), "C1",
     "정본이 팩을 선언하지 않았는데 팩이 있음: ['packs/']"),
    ("stray_packs_json", lambda r: (r / "meta/packs.json").write_text("{}"), "C1",
     "정본이 팩을 선언하지 않았는데 팩이 있음: ['meta/packs.json']"),
    # C7 -- spans
    ("span_one_frame_long", _edit_parquet(EPISODES, lambda d: d.__setitem__(
        f"videos/{HEAD}/to_timestamp", d[f"videos/{HEAD}/to_timestamp"] + 0.01)), "C7",
     "401.000 프레임인데 이 에피소드의 스텝은 400개"),
    ("span_columns", _edit_parquet(EPISODES, lambda d: d.pop(f"videos/{HEAD}/from_timestamp")), "C7",
     f"영상 구간 열 ['videos/{HEAD}/from_timestamp'] 이 없어"),
    ("span_value", _edit_parquet(EPISODES, lambda d: d.__setitem__(
        f"videos/{LAST}/to_timestamp", d[f"videos/{LAST}/to_timestamp"].where(
            d["episode_index"] != 6, np.nan))), "C7", f"{LAST} 의 영상 구간이 수치가 아님"),
    ("info_fps_missing", _edit_json("meta/info.json", lambda i: i.pop("fps")), "C7",
     "info.json 에 fps 가 없어 영상 구간을 프레임 수로"),
    # Each span was checked alone. LeRobot writes a camera file's
    # episodes back to back in episode order from 0 s (dataset_writer._save_episode_video: an
    # episode starts where the one before it in that file ended) -- anything else points an
    # episode at pictures that are not its own. Episode 1 given episode 0's spans passed C7.
    ("span_copied_from_another_episode", _copy_spans(0, 1), "C7",
     "의 구간 0~4 s 와 400.000 프레임 겹침 — 같은 파일(chunk_index 0 · file_index 0)이라 두 "
     "에피소드가 같은 그림을 가리킨다"),
    ("span_gap", _shift_spans(HEAD, (2, 3, 4), 1), "C7",
     "의 구간이 끝난 8 s 에서 시작하지 않음 (+1.000 프레임)"),
    ("span_not_from_zero", _shift_spans(LAST, range(F.EPISODES_PER_VIDEO_FILE,
                                                    2 * F.EPISODES_PER_VIDEO_FILE), 1), "C7",
     "가 0 s 에서 시작하지 않음 (+1.000 프레임) — 파일(chunk_index 0 · file_index 1)의 첫 에피소드다"),
    ("span_order_swapped", _swap_spans(HEAD, 1, 2), "C7",
     "의 구간이 끝난 4 s 에서 시작하지 않음 (+400.000 프레임)"),
    # The tolerance is float noise, well under a frame: half a frame is a gap.
    ("span_gap_half_frame", _shift_spans(HEAD, (3, 4), 0.5), "C7",
     "의 구간이 끝난 12 s 에서 시작하지 않음 (+0.500 프레임)"),
    ("episode_index_missing", _edit_parquet(EPISODES, lambda d: d.pop(
        "episode_index")), "C7", "meta/episodes 에 episode_index 가 없어"),
    ("episode_index_value", _edit_parquet(EPISODES, lambda d: d.__setitem__(
        "episode_index", d["episode_index"].where(d["episode_index"] != 9, -9))), "C7",
     "episode_index 가 0 이상의 정수가 아님: -9"),
    # A float episode_index holding inf survives dropna and int(inf) raised OverflowError; the
    # row now joins no episode, so episode 5 is one step short of its streams.
    ("data_episode_index_inf", _edit_parquet(DATA_1, lambda d: d.__setitem__(
        "episode_index", d["episode_index"].astype(float).where(d.index != 3, np.inf))), "C7",
     "400.000 프레임인데 이 에피소드의 스텝은 399개"),
    # C8 -- the camera columns
    ("box_column_missing", _edit_parquet(DATA_1, lambda d: d.pop(V.BBOX_FIELD)),
     "C8", f"조건부 필수 스텝 필드 없음: {V.BBOX_FIELD}"),
    ("pose_column_missing", _edit_parquet(DATA_0, lambda d: d.pop(POSES[-1])),
     "C8", f"조건부 필수 스텝 필드 없음: {POSES[-1]}"),
    ("pose_nan", _edit_parquet(DATA_1, lambda d: d.__setitem__(POSES[0], [
        np.where(np.arange(7) == 2, np.nan, v).astype(np.float32) if i == 17 else v
        for i, v in enumerate(d[POSES[0]])])), "C8", f"{POSES[0]} 에 유한하지 않은 값 1개"),
    ("box_reversed", _set_box(9, [5, 1, 2, 3]), "C8", "영상 범위 밖이거나 뒤집힌 행 1개 (예: [5, 1, 2, 3])"),
    ("box_below_zero", _set_box(9, [-1, 0, 3, 3]), "C8", "영상 범위 밖이거나 뒤집힌 행 1개 (예: [-1, 0, 3, 3])"),
    ("box_past_bottom", _set_box(9, [0, 0, 1, 6]), "C8", "영상 범위 밖이거나 뒤집힌 행 1개 (예: [0, 0, 1, 6])"),
    # One case per range condition: each is caught by that one condition alone (y_min < 0 and
    # y_min > y_max had none, so deleting either term stayed green).
    ("box_above_top", _set_box(9, [0, -1, 3, 3]), "C8", "영상 범위 밖이거나 뒤집힌 행 1개 (예: [0, -1, 3, 3])"),
    ("box_upside_down", _set_box(9, [0, 3, 3, 1]), "C8", "영상 범위 밖이거나 뒤집힌 행 1개 (예: [0, 3, 3, 1])"),
    # Four numbers as a JSON string pass C8's length check (elem_len parses them) and were
    # skipped by the range check: an out-of-range box printed OK before the fix.
    ("box_json_strings", _edit_parquet(DATA_1, lambda d: d.__setitem__(
        V.BBOX_FIELD, [json.dumps([0, 0, 99, 99])] * len(d))), "C8",
     "가 수 4개의 배열이 아닌 행 2000개 (예: '[0, 0, 99, 99]')"),
    ("box_float", _edit_parquet(DATA_0, lambda d: d.__setitem__(V.BBOX_FIELD, [
        np.asarray(v, dtype=np.float64) for v in d[V.BBOX_FIELD]])), "C8", "가 정수 픽셀 번호가 아님"),
]


def _pack_on_the_other_camera(cams):
    """The pack moved to a camera that is not bbox.camera, with a downsample that divides that
    camera's size -- so the one rule that can refuse it is "the pack is on the box's camera"."""
    pack = cams["packs"][PACK_NAME]
    pack["camera"] = OTHER_CAMERA
    pack["downsample"] = dict(pack["downsample"], width=cams["cameras"][OTHER_CAMERA]["width"] // 2,
                              height=cams["cameras"][OTHER_CAMERA]["height"] // 2)


# The same table's cases about a pack: they run on clean_media_pack, the one clean fixture whose
# canon declares a pack (the pack code stays in the tree, switched off in the canon).
PACK_CASES = [
    # C1 -- the pack in the camera canon
    ("pack_not_object", _edit_json("meta/cameras.json", lambda c: c["packs"].__setitem__("head", 5)),
     "C1", "meta/cameras.json 의 팩 head 이 객체가 아님"),
    ("pack_count", _edit_json("meta/cameras.json",
                              lambda c: c["packs"].__setitem__("rear", c["packs"]["head"])),
     "C1", "packs 가 팩 하나를 담은 객체가 아님"),
    ("pack_camera", _edit_json("meta/cameras.json",
                               lambda c: c["packs"]["head"].__setitem__("camera", "spare_rgb")),
     "C1", "의 camera 'spare_rgb' 가 크기를 가진 카메라가 아님"),
    # The box and the pack's mask come from one camera's segmentation.
    ("pack_camera_not_the_bbox_camera", _edit_json("meta/cameras.json", _pack_on_the_other_camera),
     "C1", f"meta/cameras.json 의 팩 {PACK_NAME} 의 camera {OTHER_CAMERA!r} 가 bbox.camera "
           f"{BOX_CAMERA!r} 와 다름"),
    ("pack_downsample", _edit_json(
        "meta/cameras.json", lambda c: c["packs"]["head"]["downsample"].__setitem__("every_n_steps", 0)),
     "C1", "의 downsample 이 width·height·every_n_steps 양의 정수가 아님"),
    # The converter keeps every k-th pixel (downsample_factors), so a
    # size that does not divide the camera's is one it refuses. The validator took it, and C7
    # then held the shipped pack to a size no conversion can produce.
    ("pack_downsample_not_a_fraction", _pack_downsample("height", PACK_NATIVE["height"] - 1),
     "C1", _not_a_fraction("height", PACK_NATIVE["height"] - 1)),
    ("pack_downsample_larger_than_the_camera", _pack_downsample("width", PACK_NATIVE["width"] + 1),
     "C1", _not_a_fraction("width", PACK_NATIVE["width"] + 1)),
    ("pack_downsample_not_a_fraction_c7", _pack_downsample("width", PACK_NATIVE["width"] - 1),
     "C7", "meta/cameras.json 을 읽지 못해 팩의 스텝 간격"),
    # A list is unhashable: `camera in sizes` raised TypeError (exit 3) before the fix.
    ("pack_camera_list", _edit_json("meta/cameras.json",
                                    lambda c: c["packs"]["head"].__setitem__("camera", ["head_rgb"])),
     "C1", "의 camera ['head_rgb'] 가 크기를 가진 카메라가 아님"),
    ("pack_dtype", _edit_json("meta/cameras.json",
                              lambda c: c["packs"]["head"]["depth"].__setitem__("dtype", "uint17")),
     "C1", "의 depth.dtype 이 numpy 형 이름이 아님"),
    # numpy's own parse errors besides TypeError: "," raised SyntaxError, "0f=65:" ValueError.
    ("pack_dtype_syntax", _edit_json("meta/cameras.json",
                                     lambda c: c["packs"]["head"]["mask"].__setitem__("dtype", ",")),
     "C1", "의 mask.dtype 이 numpy 형 이름이 아님: ','"),
    ("pack_dtype_value", _edit_json("meta/cameras.json",
                                    lambda c: c["packs"]["head"]["depth"].__setitem__("dtype", "0f=65:")),
     "C1", "의 depth.dtype 이 numpy 형 이름이 아님: '0f=65:'"),
    # C1 -- the packs
    ("packs_dir_missing", _rmtree("packs"), "C1", "packs/ 없음"),
    ("packs_json_missing", _unlink("meta/packs.json"), "C1", "meta/packs.json 없음"),
    ("packs_json_unreadable", lambda r: (r / "meta/packs.json").write_text("["), "C1",
     "meta/packs.json 의 JSON 파싱 실패"),
    ("packs_json_list", lambda r: (r / "meta/packs.json").write_text("[]"), "C1",
     "meta/packs.json 최상위가 객체가 아님"),
    ("packs_duplicate_key", _duplicate_pack_key, "C1", "같은 키가 두 번 이상 나옴: ['2']"),
    # str.isdigit accepts "²" and int() refuses it: ValueError (exit 3) before the fix.
    ("packs_key_superscript", _edit_json("meta/packs.json", lambda p: p.__setitem__("²", p.pop("3"))),
     "C1", "키 '²' 가 에피소드 번호(episode_index)가 아님"),
    ("packs_json_deep", lambda r: (r / "meta/packs.json").write_text("[" * 100_000), "C1",
     "meta/packs.json 의 JSON 파싱 실패: RecursionError"),
    ("packs_key", _edit_json("meta/packs.json", lambda p: p.__setitem__("03", p.pop("3"))), "C1",
     "키 '03' 가 에피소드 번호(episode_index)가 아님"),
    ("pack_entry_type", _edit_json("meta/packs.json", lambda p: p.__setitem__("3", [])), "C1",
     "항목이 객체가 아님"),
    ("pack_path_other_episode", _edit_json("meta/packs.json", lambda p: p["3"].__setitem__(
        "file", p["2"]["file"])), "C1", "규격 경로 packs/head/episode_000003.npz 가 아닌"),
    # Without a usable camera canon the spec path is unknown -- the listed path still has to
    # stay inside the distribution before anything opens it.
    ("pack_path_outside", _both(_edit_json("meta/cameras.json", lambda c: c.__setitem__("fps", 0)),
                                _edit_json("meta/packs.json", lambda p: p["3"].__setitem__(
                                    "file", "../elsewhere.npz"))), "C1",
     "file 이 배포본 안의 상대 경로가 아님: '../elsewhere.npz'"),
    ("pack_sha", _edit_json("meta/packs.json", lambda p: p["4"].__setitem__("sha256", "0" * 64)),
     "C1", "팩 packs/head/episode_000004.npz 의 SHA-256 이 meta/packs.json 과 다름"),
    ("pack_sha_form", _edit_json("meta/packs.json", lambda p: p["4"].__setitem__("sha256", "ABC")),
     "C1", "sha256 이 64자리 소문자 16진수가 아님"),
    ("pack_unlisted", lambda r: shutil.copy(r / "packs/head/episode_000000.npz",
                                            r / "packs/head/episode_000099.npz"), "C1",
     "packs/ 에 meta/packs.json 이 적지 않은 파일 1개: ['packs/head/episode_000099.npz']"),
    # C7 -- packs
    ("packs_json_unreadable_c7", lambda r: (r / "meta/packs.json").write_text("["), "C7",
     "meta/packs.json 을 읽지 못해 팩 정렬을 대조하지 못함"),
    ("pack_entry_missing", _edit_json("meta/packs.json", lambda p: p.pop("6")), "C7",
     "meta/packs.json 에 이 에피소드(episode_index=6)의 팩이 없음"),
    ("pack_entry_extra", _edit_json("meta/packs.json", lambda p: p.__setitem__("42", dict(p["0"]))),
     "C7", "meta/packs.json 에 에피소드 메타에 없는 팩 1개: ['42']"),
    ("pack_frames", _edit_json("meta/packs.json", lambda p: p["2"].__setitem__(
        "frames", p["2"]["frames"] + 1)), "C7", "의 frames 201 가 스텝 400개를 2 스텝마다 남긴 200 와 다름"),
    ("pack_frames_type", _edit_json("meta/packs.json", lambda p: p["2"].__setitem__("frames", "200")),
     "C7", "meta/packs.json 의 frames 가 정수가 아님"),
    ("pack_step_index_renumbered", _rewrite_pack(5, lambda a: a.__setitem__(
        "step_index", np.arange(len(a["step_index"]), dtype=np.int32))), "C7",
     "packs/head/episode_000005.npz 의 step_index 가 arange(0, 400, 2) 가 아님"),
    ("pack_step_index_float", _rewrite_pack(5, lambda a: a.__setitem__(
        "step_index", a["step_index"].astype(np.float64))), "C7",
     "packs/head/episode_000005.npz 의 step_index 가 arange(0, 400, 2) 가 아님"),
    ("pack_depth_rows", _rewrite_pack(1, lambda a: a.__setitem__(
        "depth", np.concatenate([a["depth"], a["depth"][:1]]))), "C7",
     "packs/head/episode_000001.npz 의 depth 가"),
    ("pack_mask_dtype", _rewrite_pack(1, lambda a: a.__setitem__("mask", a["mask"].astype(np.uint16))),
     "C7", "packs/head/episode_000001.npz 의 mask 가"),
    ("pack_members", _rewrite_pack(8, lambda a: a.__setitem__("rgb", a["mask"])), "C7",
     "packs/head/episode_000008.npz 의 배열이"),
    # A --checks C7 run printed OK over these two before the fix (packs_aligned 9 of 10).
    ("pack_path_outside_c7", _edit_json("meta/packs.json", lambda p: p["3"].__setitem__(
        "file", "../elsewhere.npz")), "C7",
     "팩 경로 '../elsewhere.npz' 가 배포본 안의 상대 경로가 아니어서 스텝 정렬을 대조하지 못함 (사유는 C1)"),
    ("pack_entry_type_c7", _edit_json("meta/packs.json", lambda p: p.__setitem__("3", [])), "C7",
     "meta/packs.json 의 이 에피소드 항목이 객체가 아니어서(list) 팩 정렬을 대조하지 못함 (사유는 C1)"),
    ("pack_unreadable", _replace_pack_bytes(8, b"not a zip archive"), "C7",
     "팩 packs/head/episode_000008.npz 을 열지 못해"),
]


@pytest.mark.parametrize("base,mutate,check,phrase",
                         [pytest.param(F.MEDIA_CLEAN, m, c, p, id=i) for i, m, c, p in CASES]
                         + [pytest.param(F.MEDIA_PACK_CLEAN, m, c, p, id=i) for i, m, c, p in PACK_CASES])
def test_a_camera_defect_is_reported_by_its_sentence(built, tmp_path, base, mutate, check, phrase):
    root = _copy(built / base, tmp_path)
    mutate(root)
    errors = _errors(root, {check})
    assert any(f.check == check and phrase in f.message for f in errors), \
        (phrase, [f.line() for f in errors])


@pytest.mark.parametrize("down", [
    pytest.param({"width": 640, "height": 480, "every_n_steps": 1}, id="identity"),
    pytest.param({"width": 320, "height": 240, "every_n_steps": 2}, id="half"),
    pytest.param({"width": 1, "height": 1, "every_n_steps": 1}, id="one_pixel"),
    pytest.param({"width": 300, "height": 480, "every_n_steps": 1}, id="width_not_a_fraction"),
    pytest.param({"width": 640, "height": 250, "every_n_steps": 1}, id="height_not_a_fraction"),
    pytest.param({"width": 1280, "height": 480, "every_n_steps": 1}, id="larger_than_the_camera"),
    pytest.param({"width": 0, "height": 480, "every_n_steps": 1}, id="zero"),
    pytest.param({"width": 640, "height": 480, "every_n_steps": 0}, id="every_zero"),
    pytest.param({"width": True, "height": 480, "every_n_steps": 1}, id="bool"),
    pytest.param({"width": 640.0, "height": 480, "every_n_steps": 1}, id="float"),
    pytest.param({"width": "640", "height": 480, "every_n_steps": 1}, id="string"),
    pytest.param({"width": 640, "height": 480}, id="every_absent"),
])
def test_the_validator_and_the_converter_agree_on_a_pack_downsample(down):
    """One rule, two copies: the validator does not import generation,
    so read_camera_setup restates convert_lerobot.downsample_factors -- a whole fraction of the
    pack camera's native size on each axis, every_n_steps an integer >= 1. What the converter
    refuses the validator reports, and what the validator accepts the converter can produce.
    The camera is the repo canon's (640 x 480), as in the real distribution; the pack is the
    fixture's (media_canon(with_pack=True)) -- the repository canon declares none."""
    from generation.convert_lerobot import downsample_factors
    cams = json.loads((REPO_ROOT / "meta" / "cameras.json").read_text(encoding="utf-8"))
    cams["packs"] = F.media_canon(with_pack=True)["packs"]
    (name, pack), = cams["packs"].items()
    pack["downsample"] = down
    setup, problems = V.read_camera_setup(cams)
    try:
        downsample_factors(cams["cameras"][pack["camera"]], down)
    except ValueError:
        assert setup is None and any(f"팩 {name} 의 downsample" in p for p in problems), problems
    else:
        assert setup is not None and not problems, problems


@pytest.mark.parametrize("packs", [
    pytest.param("absent", id="absent"), pytest.param(None, id="null"), pytest.param({}, id="empty"),
    pytest.param("one", id="one"), pytest.param("two", id="two"), pytest.param([], id="list"),
    pytest.param("head", id="string"),
])
def test_the_validator_and_the_converter_agree_on_how_many_packs(packs, tmp_path):
    """Packs are optional: none, or exactly one -- the sidecar has one media.pack.
    The converter's the_pack and the validator's read_camera_setup are two readers of one rule;
    what one takes as "no pack" the other must not refuse, or C1 fails a conversion's output.
    Two more readers are held to the same answer: the publisher's pack index check
    (publish_dataset.shipped_packs_match_index, which asks the_pack) and this fixture's own
    (the_one_pack) -- a copy that read [] as "no pack" would build a distribution C1 refuses."""
    from generation.convert_lerobot import the_pack
    from generation.publish_dataset import shipped_packs_match_index
    cams = F.media_canon(with_pack=True)
    one = cams["packs"]
    if packs == "absent":
        del cams["packs"]
    elif packs == "one":
        cams["packs"] = one
    elif packs == "two":
        cams["packs"] = {**one, "rear": next(iter(one.values()))}
    else:
        cams["packs"] = packs
    setup, problems = V.read_camera_setup(cams)
    # A distribution shipping this canon and no pack files at all.
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "cameras.json").write_text(json.dumps(cams), encoding="utf-8")
    published = shipped_packs_match_index(tmp_path)
    try:
        name = the_pack(cams)
    except ValueError:
        assert setup is None and any("packs 가 팩 하나를 담은 객체가 아님" in p for p in problems), problems
        assert len(published) == 1 and "팩을 선언했는지 알 수 없습니다" in published[0], published
        with pytest.raises(SystemExit, match="meta/cameras.json 의 packs"):
            F.the_one_pack(cams)
    else:
        assert setup is not None and not problems, problems
        assert setup.pack == name and (name is None) == (packs in ("absent", None, {}))
        assert (F.the_one_pack(cams) or (None,))[0] == name
        if name is None:
            assert published == [], published
        else:
            assert len(published) == 1 and "팩을 선언했는데 meta/packs.json 이 없습니다" in published[0], published


LEFTOVERS = [
    # One kind of camera media at a time, in the camera-less distribution: the guard has to
    # see each, or dropping cameras.json from a release that carries that kind goes quiet.
    ("video_feature", _edit_json("meta/info.json", lambda i: i.setdefault("features", {}).__setitem__(
        "observation.images.head_rgb", {"dtype": "video", "shape": [6, 8, 3]})),
     "observation.images.head_rgb"),
    ("videos_dir", lambda r: (r / "videos").mkdir(), "videos/"),
    ("packs_dir", lambda r: (r / "packs").mkdir(), "packs/"),
    ("packs_json", lambda r: (r / "meta/packs.json").write_text("{}"), "meta/packs.json"),
    ("box_column", _edit_parquet(DATA_1, lambda d: d.__setitem__(
        V.BBOX_FIELD, [np.asarray(V.BBOX_INVALID, dtype=np.int32)] * len(d))), V.BBOX_FIELD),
]


@pytest.mark.parametrize("mutate,named", [pytest.param(m, n, id=i) for i, m, n in LEFTOVERS])
def test_camera_media_without_the_canon_is_refused(built, tmp_path, mutate, named):
    root = _copy(built / "clean", tmp_path)
    mutate(root)
    errors = _errors(root, {"C1"})
    assert [f.message for f in errors if "meta/cameras.json 이 없는데 카메라 매체가 있음" in f.message
            and named in f.message], [f.line() for f in errors]


def test_a_partial_read_still_aligns_the_unread_episodes(built, tmp_path):
    """--file-limit reads only the first data file; an episode in an unread file is aligned
    against meta/episodes length instead of being skipped -- a quick check that silently
    looked at half the streams would read as a clean one."""
    root = _copy(built / F.MEDIA_CLEAN, tmp_path)
    late = F.EPISODES_PER_DATA_FILE + 1                   # an episode in the second data file
    _edit_parquet(EPISODES, lambda d: d.__setitem__(f"videos/{HEAD}/to_timestamp", d[
        f"videos/{HEAD}/to_timestamp"].where(d["episode_index"] != late,
                                             d[f"videos/{HEAD}/to_timestamp"] + 0.01)))(root)
    rep = V.run(root, {"C7"}, None, 3, 1)
    assert any("프레임인데 이 에피소드의 스텝은" in f.message
               for f in rep.findings if f.severity == V.ERROR), [f.line() for f in rep.findings]
    assert rep.stats["camera_media"]["video_spans"] == len(VIDEOS) * len(_episodes(root))


def test_a_span_off_by_float_noise_is_not_a_frame(built, tmp_path):
    # The other side of span_one_frame_long: a microsecond (1e-4 frame at the fixture's
    # rate) is noise, not a frame, and must not turn C7 red.
    root = _copy(built / F.MEDIA_CLEAN, tmp_path)
    _edit_parquet(EPISODES, lambda d: d.__setitem__(
        f"videos/{HEAD}/to_timestamp", d[f"videos/{HEAD}/to_timestamp"] + 1e-6))(root)
    assert not _errors(root, {"C7"})


def test_a_file_laid_out_with_float_noise_is_not_a_gap(built, tmp_path):
    # The other side of span_gap and span_not_from_zero: every start a microsecond late (1e-4
    # frame at the fixture's rate) is noise, not a gap -- LeRobot's own spans carry ~1e-13 s.
    root = _copy(built / F.MEDIA_CLEAN, tmp_path)
    _edit_parquet(EPISODES, lambda d: d.__setitem__(
        f"videos/{HEAD}/from_timestamp", d[f"videos/{HEAD}/from_timestamp"] + 1e-6))(root)
    assert not _errors(root, {"C7"})


def test_a_box_in_the_pack_cameras_far_corner_is_inside(built):
    # clean_media_pack carries boxes at the pack camera's last column and row, past the shipped
    # pack's (downsampled) size: green there means the range is the native picture.
    root = built / F.MEDIA_PACK_CLEAN
    cams = _read_json(root, "meta/cameras.json")
    (_, pack), = cams["packs"].items()
    width, height = cams["cameras"][pack["camera"]]["width"], cams["cameras"][pack["camera"]]["height"]
    boxes = np.stack([np.asarray(v) for p in sorted((root / "data").rglob("*.parquet"))
                      for v in pd.read_parquet(p)[V.BBOX_FIELD]])
    assert ((boxes[:, 2] == width - 1) & (boxes[:, 3] == height - 1)).any()
    assert (boxes[:, 2] >= pack["downsample"]["width"]).any()
    assert ((boxes[:, 0] == boxes[:, 2]) & (boxes[:, 0] >= 0)).any()        # one-pixel boxes
    assert not _errors(root, {"C8"})


def test_a_box_in_the_bbox_cameras_far_corner_is_inside(built):
    # Without a pack the range is bbox.camera's own picture: its last column and row are inside,
    # and one past them is not (box_past_bottom, broken_media_bbox_out_of_range).
    root = built / F.MEDIA_CLEAN
    cams = _read_json(root, "meta/cameras.json")
    assert "packs" not in cams
    camera = cams["cameras"][cams["bbox"]["camera"]]
    boxes = np.stack([np.asarray(v) for p in sorted((root / "data").rglob("*.parquet"))
                      for v in pd.read_parquet(p)[V.BBOX_FIELD]])
    assert ((boxes[:, 2] == camera["width"] - 1) & (boxes[:, 3] == camera["height"] - 1)).any()
    assert not _errors(root, {"C8"})


# The first camera in the canon's order, and the next one: bbox.camera is moved to the latter.
FIRST_CAMERA = next(iter(F.media_canon()["cameras"]))
NOT_FIRST_CAMERA = next(n for n in F.media_canon()["cameras"] if n != FIRST_CAMERA)


def _box_on_a_camera_that_is_not_the_first(cams):
    """bbox.camera moved off the first camera, to a camera whose picture is made wider and
    shorter than the first one's: neither picture then holds the other, so a box can fit exactly
    one. The validator reads no channels, so for C8 this canon is as good as the real one."""
    first = cams["cameras"][FIRST_CAMERA]
    cams["bbox"]["camera"] = NOT_FIRST_CAMERA
    moved = cams["cameras"][NOT_FIRST_CAMERA]
    moved["width"], moved["height"] = first["width"] + 2, first["height"] - 2


def _only_box(box):
    """Every row's box not visible, except one in the middle of the second data file."""
    def mutate(root: Path) -> None:
        for rel in (DATA_0, DATA_1):
            df = pd.read_parquet(root / rel)
            vals = [np.asarray(V.BBOX_INVALID, dtype=np.int32)] * len(df)
            if rel == DATA_1:
                vals[len(vals) // 2] = np.asarray(box, dtype=np.int32)
            df[V.BBOX_FIELD] = vals
            _write_table(df, root / rel)
    return mutate


def test_a_nan_planted_in_a_pose_is_the_only_thing_wrong_with_the_file(built, tmp_path):
    """pose_nan exists to say "a NaN in a pose" -- C8's sentence. Written through to_parquet its NaN became
    a null as well, and C7 fired on it too: the case no longer tested the NaN alone."""
    root = _copy(built / F.MEDIA_CLEAN, tmp_path)
    (mutate,) = [m for case, m, _, _ in CASES if case == "pose_nan"]
    mutate(root)
    errors = _errors(root)
    assert {f.check for f in errors} == {"C8"}, [f.line() for f in errors]
    assert any(f"{POSES[0]} 에 유한하지 않은 값 1개" in f.message for f in errors), [f.line() for f in errors]


@pytest.mark.parametrize("fits", ["bbox_camera", "first_camera"])
def test_the_box_is_held_to_bbox_camera_not_to_the_first_camera(built, tmp_path, fits):
    """C8 measures the box against bbox.camera's picture. In the clean fixtures the box camera is
    also the first camera, so nothing there tells the two apart: here bbox.camera is another one,
    and a box that fits only its picture is accepted, one that fits only the first camera's is
    refused -- a C8 that took the first camera's size would get both the wrong way round."""
    root = _copy(built / F.MEDIA_CLEAN, tmp_path)
    _edit_json("meta/cameras.json", _box_on_a_camera_that_is_not_the_first)(root)
    cams = _read_json(root, "meta/cameras.json")
    assert cams["bbox"]["camera"] == NOT_FIRST_CAMERA != next(iter(cams["cameras"]))
    size = cams["cameras"][NOT_FIRST_CAMERA if fits == "bbox_camera" else FIRST_CAMERA]
    box = [0, 0, size["width"] - 1, size["height"] - 1]
    _only_box(box)(root)
    errors = [f.message for f in _errors(root, {"C8"})]
    if fits == "bbox_camera":
        assert not errors, errors
    else:
        assert any(f"영상 범위 밖이거나 뒤집힌 행 1개 (예: {box})" in m
                   and f"({NOT_FIRST_CAMERA} 원 해상도)" in m for m in errors), errors


# ── Damaged packs: a finding, never a crash ─────────────────────────────────
# A crash (exit 3) loses the whole report -- the C1 SHA-256 line that names the damage
# included. The damage below is built from the zip structure, not from byte positions, so it
# means the same thing whatever bytes numpy and zlib write.

DAMAGED = 4                                      # the episode whose pack is damaged


def _member(data: bytes, name: str) -> tuple[int, int]:
    """(central-directory record offset, local data offset) of one member."""
    import struct
    central = local = -1
    i = data.find(b"PK\x01\x02")
    while i >= 0:
        n, = struct.unpack("<H", data[i + 28:i + 30])
        if data[i + 46:i + 46 + n] == name.encode():
            central = i
            offset, = struct.unpack("<I", data[i + 42:i + 46])
            ln, le = struct.unpack("<HH", data[offset + 26:offset + 30])
            local = offset + 30 + ln + le
            break
        i = data.find(b"PK\x01\x02", i + 4)
    assert central >= 0, name
    return central, local


def _damage(fn, keep_sha: bool = True):
    """Rewrite episode DAMAGED's pack bytes with ``fn``. keep_sha leaves meta/packs.json's
    SHA-256 as it was -- damage after conversion, which the hash is there to catch."""
    def mutate(root: Path) -> None:
        listed = _read_json(root, "meta/packs.json")
        path = root / listed[str(DAMAGED)]["file"]
        path.write_bytes(fn(bytearray(path.read_bytes())))
        if not keep_sha:
            listed[str(DAMAGED)]["sha256"] = F.sha256_of(path)
            (root / "meta" / "packs.json").write_text(json.dumps(listed, indent=2), encoding="utf-8")
    return mutate


def _deflate_garbage(data: bytearray) -> bytes:
    _, local = _member(bytes(data), "depth.npy")
    for i in range(local, local + 8):
        data[i] ^= 0xFF
    return bytes(data)


def _encrypted_flag(data: bytearray) -> bytes:
    central, _ = _member(bytes(data), "mask.npy")
    data[central + 8] |= 0x01                    # general purpose flag bit 0: encrypted
    return bytes(data)


def _unknown_method(data: bytearray) -> bytes:
    central, _ = _member(bytes(data), "depth.npy")
    data[central + 10:central + 12] = (99).to_bytes(2, "little")
    return bytes(data)


def _stored_pack(step_index_npy: bytes):
    """A pack rebuilt uncompressed, with the step_index member's .npy bytes given."""
    def fn(data: bytearray) -> bytes:
        import io
        import zipfile
        with zipfile.ZipFile(io.BytesIO(bytes(data))) as zf:
            members = {i.filename: zf.read(i) for i in zf.infolist()}
        members["step_index.npy"] = step_index_npy
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
            for name, blob in members.items():
                zf.writestr(name, blob)
        return out.getvalue()
    return fn


def _npy(header: str) -> bytes:
    """A version-1.0 .npy with this header dict text and no data."""
    text = header.encode("latin1")
    pad = 64 - (10 + len(text) + 1) % 64
    text += b" " * pad + b"\n"
    return b"\x93NUMPY\x01\x00" + len(text).to_bytes(2, "little") + text


CONTROL_CHAR_HEADER = _npy("{'descr': '<i4', 'fortran_order': False, 'shape': (200,), \x15}")
HUGE_HEADER = _npy("{'descr': '<i4', 'fortran_order': False, 'shape': (1000000000000,), }")
CANT_OPEN = f"팩 packs/head/episode_{DAMAGED:06d}.npz 을 열지 못해 스텝 정렬을 대조하지 못함"
SHA_LINE = f"팩 packs/head/episode_{DAMAGED:06d}.npz 의 SHA-256 이 meta/packs.json 과 다름"


@pytest.mark.parametrize("mutate,phrases", [
    # zlib.error on this pack ("invalid distance too far back")
    pytest.param(_damage(_deflate_garbage), (CANT_OPEN, SHA_LINE), id="deflate_stream"),
    # zipfile raises RuntimeError: "is encrypted, password required"
    pytest.param(_damage(_encrypted_flag), (CANT_OPEN, SHA_LINE), id="encrypted_flag"),
    # zipfile raises NotImplementedError: "That compression method is not supported"
    pytest.param(_damage(_unknown_method), (CANT_OPEN, SHA_LINE), id="compression_method"),
    # numpy's header parse (tokenize.TokenError on Python 3.12); the hash is kept true here
    pytest.param(_damage(_stored_pack(CONTROL_CHAR_HEADER), keep_sha=False), (CANT_OPEN,),
                 id="header_text"),
    # a header claiming 10^12 rows is refused from the header, before any allocation
    pytest.param(_damage(_stored_pack(HUGE_HEADER), keep_sha=False),
                 (f"packs/head/episode_{DAMAGED:06d}.npz 의 step_index 가 arange(0, 400, 2) 가 아님",),
                 id="header_claims_1e12_rows"),
])
def test_a_damaged_pack_is_a_finding_not_a_crash(built, tmp_path, mutate, phrases):
    root = _copy(built / F.MEDIA_PACK_CLEAN, tmp_path)
    mutate(root)
    errors = _errors(root)                        # all checks: a crash raises out of V.run
    for phrase in phrases:
        assert any(phrase in f.message for f in errors), (phrase, [f.line() for f in errors])


def test_no_single_byte_corruption_of_a_pack_escapes_c7(built, tmp_path):
    """Every byte of one pack flipped in turn: C7 either reads the pack or says it could not.

    The property behind the cases above -- the exception types belong to zipfile, zlib and
    numpy and differ between their versions, so no list of them is pinned here."""
    root = _copy(built / F.MEDIA_PACK_CLEAN, tmp_path)
    cams = _read_json(root, "meta/cameras.json")
    setup, problems = V.read_camera_setup(cams)
    assert setup is not None, problems
    entry = _read_json(root, "meta/packs.json")["0"]
    path = root / entry["file"]
    steps = len(pd.read_parquet(root / DATA_0).query("episode_index == 0"))
    original = path.read_bytes()
    unreadable = 0
    for pos in range(len(original)):
        data = bytearray(original)
        data[pos] ^= 0xFF
        path.write_bytes(bytes(data))
        rep = V.Report(root=str(root))
        read = V._c7_pack(V.Dataset(root, rep), setup, entry, steps, "ep0", rep)
        said = [f for f in rep.findings if f.check == "C7" and f.severity == V.ERROR]
        if not read:
            unreadable += 1
            assert said, f"byte {pos}: the pack was not read and nothing was said"
    assert unreadable, "no corruption made the pack unreadable -- the loop tested nothing"


def test_an_episode_whose_steps_are_unknown_is_said(built, tmp_path):
    """--file-limit reads the first data file only; without meta/episodes length the other
    file's episodes have no step count. C7 printed OK over their streams and packs."""
    root = _copy(built / F.MEDIA_CLEAN, tmp_path)
    _edit_parquet(EPISODES, lambda d: d.pop("length"))(root)
    rep = V.run(root, {"C7"}, None, 3, 1)
    said = [f.location for f in rep.findings if f.severity == V.ERROR
            and "스텝 수를 알 수 없어" in f.message]
    unread = _episodes(root).query(f"episode_index >= {F.EPISODES_PER_DATA_FILE}")
    assert sorted(said) == sorted(unread["episode_id"]), said
    assert rep.stats["camera_media"]["video_spans"] == len(VIDEOS) * F.EPISODES_PER_DATA_FILE


def test_bbox_rows_counts_the_rows_compared_not_the_rows_present(built, tmp_path):
    root = _copy(built / F.MEDIA_CLEAN, tmp_path)
    _edit_parquet(DATA_1, lambda d: d.__setitem__(
        V.BBOX_FIELD, [json.dumps([0, 0, 1, 1])] * len(d)))(root)
    rep = V.run(root, {"C8"}, None, 3, None)
    assert rep.stats["camera_media"]["bbox_rows"] == len(pd.read_parquet(root / DATA_0))
