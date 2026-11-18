"""generation/scene/textures.lock — 씬 텍스처의 출처·라이선스·해시 정본.

재배포하는 자산이므로 CC0 가 아니거나 해시가 없는 줄은 들어올 수 없다.
"""

import hashlib
from pathlib import Path

import pytest

from generation.scene import fetch_textures as ft
from generation.scene import layout

HERE = Path(__file__).resolve().parents[2] / "generation/scene"
LOCK = HERE / "textures.lock"
MAPS = {"diffuse", "normal_gl", "roughness"}


def test_lock_covers_every_texture_set_the_spec_names():
    spec = layout.load_spec(HERE / "logistics_aisle.yaml")
    used = {b.material for b in layout.build_layout(spec) if b.material != "light"}
    # a plain-colour material names no texture set and carries no file (layout._validate_material)
    needed = {spec["materials"][m]["texture"] for m in used if "texture" in spec["materials"][m]}
    assert needed
    have = {e.role for e in ft.parse_lock(LOCK)}
    assert needed <= have, f"texture sets without a lock entry: {sorted(needed - have)}"


def test_every_texture_set_has_the_three_maps():
    by_role: dict[str, set[str]] = {}
    for e in ft.parse_lock(LOCK):
        by_role.setdefault(e.role, set()).add(e.map)
    for role, maps in by_role.items():
        assert MAPS <= maps, f"{role}: missing {sorted(MAPS - maps)}"


def test_every_entry_is_cc0_with_a_sha256_and_a_source():
    for e in ft.parse_lock(LOCK):
        assert e.license == "CC0", e.name
        assert len(e.sha256) == 64 and int(e.sha256, 16), e.name
        assert e.source in ft.KNOWN_SOURCES, e.name
        if e.map == "archive" or not e.url.startswith("zip:"):
            assert e.url.startswith("https://"), e.name


def test_zip_members_point_at_an_archive_of_the_same_set():
    entries = ft.parse_lock(LOCK)
    archives = {e.role for e in entries if e.map == "archive"}
    for e in entries:
        if e.url.startswith("zip:"):
            assert e.role in archives, e.name


def test_filenames_are_unique_so_the_flat_texture_dir_cannot_collide():
    names = [e.filename for e in ft.parse_lock(LOCK) if e.map != "archive"]
    assert len(names) == len(set(names)), names


def test_parse_rejects_a_line_with_the_wrong_number_of_columns(tmp_path):
    bad = tmp_path / "t.lock"
    bad.write_text("a floor diffuse CC0 polyhaven https://x/a.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        ft.parse_lock(bad)


def test_parse_rejects_a_non_cc0_licence(tmp_path):
    bad = tmp_path / "t.lock"
    bad.write_text("a floor diffuse CC-BY polyhaven https://x/a.jpg " + "0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CC0"):
        ft.parse_lock(bad)


def test_verify_detects_a_changed_file(tmp_path):
    f = tmp_path / "t.jpg"
    f.write_bytes(b"abc")
    e = ft.TextureEntry("t", "floor", "diffuse", "CC0", "polyhaven", "https://x/t.jpg",
                        hashlib.sha256(b"abc").hexdigest())
    assert ft.verify(e, f)
    f.write_bytes(b"abd")
    assert not ft.verify(e, f)
    assert not ft.verify(e, tmp_path / "missing.jpg")


def test_files_for_role_returns_one_path_per_map(tmp_path):
    entries = ft.parse_lock(LOCK)
    role = entries[0].role
    files = ft.files_for_role(entries, role, tmp_path)
    assert set(files) == MAPS
    for p in files.values():
        assert p.parent == tmp_path


# ---- a lock with nothing in it, and a material whose texture set is unknown or incomplete ----

def test_parse_rejects_an_empty_or_comment_only_lock(tmp_path):
    """A truncated lock must not verify as 'all files ok' — there would be nothing to check."""
    for body in ("", "# only a comment\n", "\n\n"):
        f = tmp_path / "t.lock"
        f.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match="no entries"):
            ft.parse_lock(f)


def test_check_only_fails_on_an_empty_lock(tmp_path, capsys):
    f = tmp_path / "t.lock"
    f.write_text("# nothing\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        ft.main(["--check-only", "--lock", str(f), "--out", str(tmp_path)])


def test_files_for_material_requires_every_map_of_a_known_set(tmp_path):
    entries = ft.parse_lock(LOCK)
    role = entries[0].role
    files = ft.files_for_material(entries, role, tmp_path)
    assert set(files) == MAPS
    with pytest.raises(ValueError, match="no texture set"):
        ft.files_for_material(entries, role + "_typo", tmp_path)
    partial = [e for e in entries if not (e.role == role and e.map == "roughness")]
    with pytest.raises(ValueError, match="roughness"):
        ft.files_for_material(partial, role, tmp_path)
