"""The box the robot carries (generation/scene/carried_box.py) and the KETI CI on its labels
(generation/scene/fetch_keti_ci.py, generation/scene/keti_ci.json).

The label layout, the resize overrides, the PNG header and the CI fetch are Isaac-free and run
everywhere. The two USD tests need pxr: they skip in CI and run under the Isaac python

    PYTHONPATH= ~/isaacsim/python.sh -B -m pytest -q -p no:launch_testing -p no:launch_ros \\
        tests/generation/test_carried_box.py
"""

import hashlib
import json
import re
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest

from generation.scene import carried_box as cb
from generation.scene import fetch_keti_ci as fk

REPO = Path(__file__).resolve().parents[2]

# Hand-worked: (distance of the face from the centre, normal, up, face extent along the label's
# width, face extent along the label's height) for a 0.40 x 0.25 x 0.20 m box. A label's width
# runs along up x normal (the viewer's right when facing the side).
SMALLEST = (0.40, 0.25, 0.20)
FACES_SMALLEST = {
    "+x": (0.20, (1, 0, 0), (0, 0, 1), 0.25, 0.20),
    "-x": (0.20, (-1, 0, 0), (0, 0, 1), 0.25, 0.20),
    "+y": (0.125, (0, 1, 0), (0, 0, 1), 0.40, 0.20),
    "-y": (0.125, (0, -1, 0), (0, 0, 1), 0.40, 0.20),
    "+z": (0.10, (0, 0, 1), (0, 1, 0), 0.40, 0.25),
}
LARGEST = (0.40, 0.30, 0.30)
FACES_LARGEST = {
    "+x": (0.20, (1, 0, 0), (0, 0, 1), 0.30, 0.30),
    "-x": (0.20, (-1, 0, 0), (0, 0, 1), 0.30, 0.30),
    "+y": (0.15, (0, 1, 0), (0, 0, 1), 0.40, 0.30),
    "-y": (0.15, (0, -1, 0), (0, 0, 1), 0.40, 0.30),
    "+z": (0.15, (0, 0, 1), (0, 1, 0), 0.40, 0.30),
}
CI_ASPECT = 584 / 147                               # the official PNG's width / height in pixels


def _png(path: Path, width: int, height: int) -> Path:
    """A valid RGBA PNG of that size, every pixel transparent."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    raw = b"".join(b"\x00" + bytes(width * 4) for _ in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


# ---- the labels -------------------------------------------------------------------------------

@pytest.mark.parametrize("size, faces", [(SMALLEST, FACES_SMALLEST), (LARGEST, FACES_LARGEST)])
def test_labels_sit_centred_on_five_faces_and_fit_the_smallest_box(size, faces):
    rects = cb.label_rects(size, CI_ASPECT)
    # two per face, in face order, the white label first and the CI on it second
    assert [r.face for r in rects] == ["+x", "+x", "-x", "-x", "+y", "+y", "-y", "-y", "+z", "+z"]
    for i, r in enumerate(rects):
        half, normal, up, room_w, room_h = faces[r.face]
        lift = 0.001 if i % 2 == 0 else 0.002       # the label 1 mm off the face, the CI 2 mm
        assert np.allclose(r.normal, normal) and np.allclose(r.up, up)
        assert np.allclose(r.center, np.array(normal) * (half + lift))
        if i % 2 == 0:
            assert (r.width, r.height) == pytest.approx((0.20, 0.07))
        else:
            assert (r.width, r.height) == pytest.approx((0.18, 0.18 * 147 / 584))
        assert r.width <= room_w and r.height <= room_h, (r.face, r.width, r.height)


def test_a_label_corner_winds_outward():
    """The plates are one-sided quads: counter-clockwise seen from outside, so their normal
    points away from the box."""
    for r in cb.label_rects(cb.NOMINAL_SIZE_M, CI_ASPECT):
        p = cb.plate_corners(r)
        assert p.shape == (4, 3)
        n = np.cross(p[1] - p[0], p[3] - p[0])
        assert np.allclose(n / np.linalg.norm(n), r.normal), r.face
        assert np.ptp(p @ r.normal) == pytest.approx(0.0)          # flat, in the face's plane
        assert np.allclose(p.mean(axis=0), 0.0)                      # centred on the rect
        assert not np.signbit(p[p == 0]).any(), p                    # no -0 written into the USD


@pytest.mark.parametrize("size, face", [((0.40, 0.15, 0.20), "+x"), ((0.40, 0.25, 0.05), "+x"),
                                        ((0.15, 0.25, 0.20), "+y"), ((0.40, 0.06, 0.20), "+x")])
def test_a_box_too_small_for_its_label_is_refused(size, face):
    with pytest.raises(ValueError, match=re.escape(f"{face} 면")):
        cb.label_rects(size, CI_ASPECT)


@pytest.mark.parametrize("size", [(0.4, 0.3), (0.4, 0.3, 0.0), (0.4, -0.3, 0.3), (0.4, float("nan"), 0.3),
                                  (0.4, True, 0.3), ("0.4", 0.3, 0.3)])
def test_a_size_that_is_not_three_positive_lengths_is_refused(size):
    with pytest.raises(ValueError, match="상자 크기"):
        cb.label_rects(size, CI_ASPECT)


@pytest.mark.parametrize("aspect", [0.0, -4.0, float("inf"), float("nan"), 2.0])
def test_a_ci_that_would_not_fit_its_label_is_refused(aspect):
    # 2.0: an 0.18 m wide CI would be 0.09 m tall, taller than the 0.07 m label
    with pytest.raises(ValueError, match="CI"):
        cb.label_rects(cb.NOMINAL_SIZE_M, aspect)


def test_resizing_moves_labels_and_never_stretches_them():
    nominal = cb.box_overrides(cb.NOMINAL_SIZE_M)
    bigger = cb.box_overrides(LARGEST)
    faces = ("pos_x", "neg_x", "pos_y", "neg_y", "pos_z")
    plates = {f"labels/{f}/{k}" for f in faces for k in ("base", "ci")}
    assert set(nominal) == set(bigger) == {"body"} | plates
    # the body is the only thing scaled -- to the box size, about the box centre
    assert np.allclose(nominal["body"][0], 0.0) and np.allclose(nominal["body"][1], (0.40, 0.275, 0.25))
    assert np.allclose(bigger["body"][1], (0.40, 0.30, 0.30))
    for p in plates:
        assert np.allclose(nominal[p][1], 1.0) and np.allclose(bigger[p][1], 1.0), p
    # labels move with their face: half the new depth or height, plus the lift
    assert np.allclose(nominal["labels/pos_y/base"][0], (0, 0.1385, 0))
    assert np.allclose(bigger["labels/pos_y/base"][0], (0, 0.151, 0))
    assert np.allclose(nominal["labels/pos_z/ci"][0], (0, 0, 0.127))
    assert np.allclose(bigger["labels/pos_z/ci"][0], (0, 0, 0.152))
    assert np.allclose(nominal["labels/neg_x/base"][0], (-0.201, 0, 0))
    assert np.allclose(bigger["labels/neg_x/base"][0], (-0.201, 0, 0))


def test_the_nominal_box_is_the_middle_of_the_carried_range():
    # width fixed; depth 0.25-0.30 and height 0.20-0.30 -- the nominal is their middle
    assert cb.NOMINAL_SIZE_M == (0.40, 0.275, 0.25)
    assert cb.LABEL_M == (0.20, 0.07) and cb.CI_WIDTH_M == 0.18


def test_the_ci_keeps_its_aspect(tmp_path):
    png = _png(tmp_path / "ci.png", 584, 147)
    assert cb.png_size(png) == (584, 147)
    w, h = cb.png_size(png)
    ci = cb.label_rects(cb.NOMINAL_SIZE_M, w / h)[1]
    assert ci.width / ci.height == pytest.approx(584 / 147)
    assert ci.width == pytest.approx(0.18)


@pytest.mark.parametrize("data", [b"", b"GIF89a" + bytes(30), b"\x89PNG\r\n\x1a\n" + bytes(16),
                                  b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + struct.pack(">II", 0, 147)])
def test_png_size_refuses_what_is_not_a_png_header(tmp_path, data):
    f = tmp_path / "x.png"
    f.write_bytes(data)
    with pytest.raises(ValueError, match="PNG"):
        cb.png_size(f)


# ---- the CI: pinned by hash, picked from the archive by hash ----------------------------------

MEMBER = "CI-가로타입#1.png"                           # the official archive's own member name


def _archive(tmp_path: Path, png: bytes) -> Path:
    z = tmp_path / "KETI_CI.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("CI 가로타입#1.ai", b"%PDF-1.4 vector\n")
        zf.writestr("keti_ci_en.png", b"a decoy with the output's name\n")
        zf.writestr(MEMBER, png)
    return z


def _lock(archive: Path, png: bytes) -> dict:
    return {"url": "https://example.invalid/KETI_CI.zip?a=1",
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "member": MEMBER, "member_sha256": hashlib.sha256(png).hexdigest(),
            "file": "keti_ci_en.png", "license": "LicenseRef-KETI-CI"}


def test_fetch_finds_the_member_by_its_hash(tmp_path):
    png = _png(tmp_path / "src.png", 584, 147).read_bytes()
    archive = _archive(tmp_path, png)
    lock = _lock(archive, png)
    out = tmp_path / "textures"
    got = fk.fetch(lock, archive, out)
    assert got == out / "keti_ci_en.png" and got.read_bytes() == png      # not the decoy
    # already the pinned bytes: left as it is, the archive not needed
    archive.unlink()
    assert fk.fetch(lock, None, out) == got and got.read_bytes() == png


def test_fetch_refuses_an_archive_with_another_hash(tmp_path):
    png = _png(tmp_path / "src.png", 584, 147).read_bytes()
    archive = _archive(tmp_path, png)
    lock = dict(_lock(archive, png), archive_sha256="0" * 64)
    out = tmp_path / "textures"
    with pytest.raises(ValueError, match="압축 파일의 SHA-256"):
        fk.fetch(lock, archive, out)
    assert not (out / "keti_ci_en.png").exists()


def test_fetch_refuses_an_archive_without_the_pinned_member(tmp_path):
    png = _png(tmp_path / "src.png", 584, 147).read_bytes()
    archive = _archive(tmp_path, png)
    lock = dict(_lock(archive, png), member_sha256="1" * 64)
    with pytest.raises(ValueError, match="SHA-256 이 1111"):
        fk.fetch(lock, archive, tmp_path / "textures")
    assert not (tmp_path / "textures" / "keti_ci_en.png").exists()


def test_fetch_replaces_a_file_that_is_not_the_pinned_one(tmp_path):
    png = _png(tmp_path / "src.png", 584, 147).read_bytes()
    archive = _archive(tmp_path, png)
    out = tmp_path / "textures"
    out.mkdir()
    (out / "keti_ci_en.png").write_bytes(b"something else\n")
    assert fk.fetch(_lock(archive, png), archive, out).read_bytes() == png


def test_the_command_takes_a_downloaded_archive(tmp_path, capsys):
    png = _png(tmp_path / "src.png", 584, 147).read_bytes()
    archive = _archive(tmp_path, png)
    lock_file = tmp_path / "keti_ci.json"
    lock_file.write_text(json.dumps(_lock(archive, png)), encoding="utf-8")
    out = tmp_path / "textures"
    assert fk.main(["--lock", str(lock_file), "--archive", str(archive), "--out", str(out)]) == 0
    assert (out / "keti_ci_en.png").read_bytes() == png
    assert "keti_ci_en.png" in capsys.readouterr().out


@pytest.mark.parametrize("edit, phrase", [
    (lambda d: d.pop("member_sha256"), "member_sha256"),
    (lambda d: d.update(archive_sha256="ABC"), "archive_sha256"),
    (lambda d: d.update(member_sha256="A" * 64), "member_sha256"),
    (lambda d: d.update(file="textures/keti_ci_en.png"), "file"),
    (lambda d: d.update(url="http://example.invalid/KETI_CI.zip"), "url"),
    (lambda d: d.update(license="CC-BY-4.0"), "LicenseRef-KETI-CI"),
    (lambda d: d.update(licence="x"), "모르는 키"),
    (lambda d: d.update(member=""), "member"),
])
def test_a_ci_lock_that_does_not_pin_the_file_is_refused(tmp_path, edit, phrase):
    lock = _lock(_archive(tmp_path, b"png"), b"png")
    edit(lock)
    f = tmp_path / "keti_ci.json"
    f.write_text(json.dumps(lock, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match=phrase):
        fk.load_ci_lock(f)


def test_the_ci_lock_pins_the_official_archive():
    lock = fk.load_ci_lock()
    assert fk.CI_LOCK == REPO / "generation/scene/keti_ci.json"
    assert lock["url"].startswith("https://keti.re.kr/") and lock["license"] == "LicenseRef-KETI-CI"
    assert lock["file"] == "keti_ci_en.png"


# ---- the canon entry --------------------------------------------------------------------------

def _canon() -> dict:
    return json.loads((REPO / "meta/assets.json").read_text(encoding="utf-8"))


def test_the_ci_lock_row_is_what_the_bundle_says():
    from generation.assets import _ci_lock_rows
    e = _canon()["scene"]["keti_carried_box_v1"]
    b = e["bundle"]
    assert b["textures_lock"] == "generation/scene/keti_ci.json"
    assert _ci_lock_rows(REPO, e, b["textures_lock"]) == b["files"]
    assert b["textures_license"] == fk.CI_LICENSE == "LicenseRef-KETI-CI"
    assert hashlib.sha256((REPO / b["textures_lock"]).read_bytes()).hexdigest() == b["textures_lock_sha256"]


def test_the_box_ships_its_usd_and_the_ci_from_where_the_build_puts_them():
    from generation.assets import shipped_asset_files
    canon = _canon()
    e = canon["scene"]["keti_carried_box_v1"]
    assert (REPO / e["usd_path"], e["built_by"]) == (cb.OUT, cb.BUILT_BY)
    assert e["license"] == "CC-BY-4.0" and e["redistributed"] is True
    assert e["usd_checked"]["sha256"] == e["usd_sha256"]
    got = shipped_asset_files(canon, [e["usd_path"]], REPO)
    assert set(got) == {"assets/keti_carried_box_v1.usd", "assets/textures/keti_ci_en.png"}
    ci = got["assets/textures/keti_ci_en.png"]
    # the shipping code takes the CI where the box build copies it, beside the USD
    assert ci.src == cb.OUT.parent / cb.TEXTURE_SUBDIR / "keti_ci_en.png"
    assert ci.sha256 == fk.load_ci_lock()["member_sha256"]


def test_the_ci_ships_with_the_line_that_no_trademark_is_granted():
    e = _canon()["scene"]["keti_carried_box_v1"]
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    inventory = (REPO / "docs/assets/asset-license-inventory-v0.1.md").read_text(encoding="utf-8")
    for where, text in (("bundle.note", e["bundle"]["note"]), ("NOTICE", notice), ("inventory", inventory)):
        assert "LicenseRef-KETI-CI" in text and "§2(b)(2)" in text, where
    lock = fk.load_ci_lock()
    assert lock["page"] in notice and lock["url"] in notice


# ---- the USD (pxr) ----------------------------------------------------------------------------

def _bbox(stage, path):
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    r = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.array(r.GetMin()), np.array(r.GetMax())


def _plate_extent_in_plane(lo, hi, normal):
    """(width-ish, height-ish): the two extents of a flat plate's box that are not along its normal."""
    span = hi - lo
    return sorted(float(s) for i, s in enumerate(span) if abs(normal[i]) < 0.5)


def test_the_box_usd_has_a_body_five_labels_and_no_mass(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    png = _png(tmp_path / "keti_ci_en.png", 584, 147)
    out = tmp_path / "build" / "box.usd"
    cb.write_box_usd(out, cb.NOMINAL_SIZE_M, png)
    st = Usd.Stage.Open(str(out))
    assert st.GetDefaultPrim().GetPath() == Sdf.Path("/CarriedBox")
    body = st.GetPrimAtPath("/CarriedBox/body")
    assert body.IsA(UsdGeom.Mesh) and body.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(body).GetApproximationAttr().Get() == UsdPhysics.Tokens.boundingCube
    lo, hi = _bbox(st, "/CarriedBox/body")
    assert np.allclose(hi - lo, cb.NOMINAL_SIZE_M) and np.allclose((hi + lo) / 2, 0.0)
    # mass and the rigid body belong to whoever carries the box
    for p in st.Traverse():
        assert not p.HasAPI(UsdPhysics.RigidBodyAPI) and not p.HasAPI(UsdPhysics.MassAPI), p.GetPath()
        assert not p.GetAttribute("physics:mass").HasAuthoredValue(), p.GetPath()
        if p.GetPath() != body.GetPath():
            assert not p.HasAPI(UsdPhysics.CollisionAPI), p.GetPath()
    faces = [c.GetName() for c in st.GetPrimAtPath("/CarriedBox/labels").GetChildren()]
    assert sorted(faces) == ["neg_x", "neg_y", "pos_x", "pos_y", "pos_z"]
    rects = cb.label_rects(cb.NOMINAL_SIZE_M, 584 / 147)
    for r, (face, kind) in zip(rects, [(f, k) for f in ("pos_x", "neg_x", "pos_y", "neg_y", "pos_z")
                                        for k in ("base", "ci")]):
        path = f"/CarriedBox/labels/{face}/{kind}"
        assert st.GetPrimAtPath(path).IsA(UsdGeom.Mesh), path
        lo, hi = _bbox(st, path)
        assert np.allclose((hi + lo) / 2, r.center, atol=1e-7), path
        assert _plate_extent_in_plane(lo, hi, r.normal) == pytest.approx(sorted((r.width, r.height))), path
    # the CI is the PNG beside the USD, cut out by its alpha; one layer, nothing else referenced
    ci = UsdShade.MaterialBindingAPI(st.GetPrimAtPath("/CarriedBox/labels/pos_x/ci")).ComputeBoundMaterial()[0]
    shader = UsdShade.Shader(ci.GetPrim().GetChild("Shader"))
    assert shader.GetInput("diffuse_texture").Get().path == "./textures/keti_ci_en.png"
    assert shader.GetInput("opacity_texture").Get().path == "./textures/keti_ci_en.png"
    assert (out.parent / "textures" / "keti_ci_en.png").read_bytes() == png.read_bytes()
    assert [Path(lay.realPath).resolve() for lay in st.GetUsedLayers() if not lay.anonymous] == [out.resolve()]
    # deterministic: the same inputs build the same bytes
    again = tmp_path / "again" / "box.usd"
    cb.write_box_usd(again, cb.NOMINAL_SIZE_M, png)
    assert again.read_bytes() == out.read_bytes()


def test_apply_box_size_keeps_the_labels_unstretched(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    png = _png(tmp_path / "keti_ci_en.png", 584, 147)
    box = tmp_path / "box.usd"
    cb.write_box_usd(box, cb.NOMINAL_SIZE_M, png)
    # the way a stage composes it: a reference under another prim, resized there
    st = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(st, "/World")
    holder = st.DefinePrim("/World/Box")
    holder.GetReferences().AddReference(str(box))
    UsdGeom.Xformable(holder).AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 0.8))
    cb.apply_box_size(st, "/World/Box", SMALLEST)
    lo, hi = _bbox(st, "/World/Box/body")
    assert np.allclose(hi - lo, SMALLEST) and np.allclose((hi + lo) / 2, (1.0, 2.0, 0.8))
    for r, (face, kind) in zip(cb.label_rects(SMALLEST, 584 / 147),
                               [(f, k) for f in ("pos_x", "neg_x", "pos_y", "neg_y", "pos_z")
                                for k in ("base", "ci")]):
        path = f"/World/Box/labels/{face}/{kind}"
        lo, hi = _bbox(st, path)
        assert np.allclose((hi + lo) / 2, np.array((1.0, 2.0, 0.8)) + r.center, atol=1e-7), path
        assert _plate_extent_in_plane(lo, hi, r.normal) == pytest.approx(sorted((r.width, r.height))), path
        scale = [op for op in UsdGeom.Xformable(st.GetPrimAtPath(path)).GetOrderedXformOps()
                 if op.GetOpType() == UsdGeom.XformOp.TypeScale]
        assert all(np.allclose(tuple(op.Get()), 1.0) for op in scale), path
    # the box file itself is untouched: the resize is the composing stage's opinion
    lo, hi = _bbox(Usd.Stage.Open(str(box)), "/CarriedBox/body")
    assert np.allclose(hi - lo, cb.NOMINAL_SIZE_M)
    with pytest.raises(ValueError, match="/World/Nowhere/body"):
        cb.apply_box_size(st, "/World/Nowhere", SMALLEST)
