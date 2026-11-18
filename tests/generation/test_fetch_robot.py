"""Fetching a robot that ships as its own folder: the canon's ship rows, a nucleus root (an address or a
copy of it on disk), each file held to its row's hash before it lands. Isaac-free."""

import hashlib
import json
from pathlib import Path

import pytest

from generation.robots import fetch_robot as fr

FILES = {"H1/h1.usd": b"#usda 1.0\n# h1\n", "H1/instanceable_meshes.usd": b"#usda 1.0\n# meshes\n",
         "LICENSE.txt": b"BSD 3-Clause License"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _nucleus(tmp_path, files=FILES) -> Path:
    """A copy of the Isaac Lab asset root holding the robot under Robots/Unitree/."""
    root = tmp_path / "nucleus"
    for rel, data in files.items():
        (root / "Robots/Unitree" / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / "Robots/Unitree" / rel).write_bytes(data)
    return root


def _entry() -> dict:
    rows = [{"file": Path(rel).name, "sha256": _sha(data), "license": "BSD-3-Clause",
             "url": f"{{ISAACLAB_NUCLEUS_DIR}}/Robots/Unitree/{rel}"} for rel, data in FILES.items()]
    return {"usd_path": "{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd", "license": "BSD-3-Clause",
            "redistributed": True, "usd_sha256": _sha(FILES["H1/h1.usd"]),
            "ship": {"dir": "unitree_h1", "source_dir": "build/unitree_h1", "license_file": "LICENSE.txt",
                     "files": rows}}


def test_the_folder_is_fetched_from_a_copy_on_disk(tmp_path):
    repo = tmp_path / "repo"
    got = fr.fetch(_entry(), str(_nucleus(tmp_path)), repo, name="unitree_h1")
    assert got == [repo / "build/unitree_h1" / Path(rel).name for rel in FILES]
    assert {p.name: p.read_bytes() for p in got} == {Path(rel).name: d for rel, d in FILES.items()}
    # Already the pinned bytes: left as they are, even with no source to fetch from.
    assert fr.fetch(_entry(), str(tmp_path / "nowhere"), repo) == got


def test_a_file_already_there_that_is_not_the_pinned_bytes_is_fetched_again(tmp_path):
    repo = tmp_path / "repo"
    stale = repo / "build/unitree_h1/LICENSE.txt"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"an older licence")
    fr.fetch(_entry(), str(_nucleus(tmp_path)), repo)
    assert stale.read_bytes() == FILES["LICENSE.txt"]


def test_a_source_that_is_not_the_pinned_bytes_lands_nothing(tmp_path):
    repo = tmp_path / "repo"
    nucleus = _nucleus(tmp_path, dict(FILES, **{"LICENSE.txt": b"another licence"}))
    with pytest.raises(ValueError, match="LICENSE.txt 의 SHA-256 이 정본과 다릅니다"):
        fr.fetch(_entry(), str(nucleus), repo)
    # Every file is held to its row before any lands: the two that matched are not left half a folder.
    landed = sorted(p.name for p in (repo / "build/unitree_h1").iterdir()) \
        if (repo / "build/unitree_h1").is_dir() else []
    assert landed == []


def _where(nucleus: Path, kind: str) -> str:
    # The nucleus as a directory, or as an address (a file: URL goes through the download path, without a network).
    return str(nucleus) if kind == "copy" else nucleus.as_uri()


@pytest.mark.parametrize("kind", ["copy", "address"])
def test_a_file_named_like_another_file_being_fetched_lands_as_itself(tmp_path, kind):
    # Two rows whose names differ by a suffix a fetcher might use for its temporary file (h1.usd.part beside
    # h1.usd): each lands with its own bytes, and nothing of the fetch is left beside the folder.
    data = {"h1.usd.part": b"BSD licence text", "h1.usd": b"#usda 1.0\n"}
    nucleus = tmp_path / "nucleus"
    nucleus.mkdir()
    for name, content in data.items():
        (nucleus / name).write_bytes(content)
    rows = [{"file": n, "sha256": _sha(c), "license": "BSD-3-Clause", "url": f"{{ISAACLAB_NUCLEUS_DIR}}/{n}"}
            for n, c in data.items()]
    entry = {"usd_path": "{ISAACLAB_NUCLEUS_DIR}/h1.usd", "redistributed": True, "usd_sha256": _sha(data["h1.usd"]),
             "license": "BSD-3-Clause",
             "ship": {"dir": "h1", "source_dir": "build/h1", "license_file": "h1.usd.part", "files": rows}}
    repo = tmp_path / "repo"
    got = fr.fetch(entry, _where(nucleus, kind), repo)
    assert {p.name: p.read_bytes() for p in (repo / "build/h1").iterdir()} == data
    assert got == [repo / "build/h1" / name for name in data]
    assert [p.name for p in (repo / "build").iterdir()] == ["h1"]


@pytest.mark.parametrize("kind", ["copy", "address"])
def test_a_failed_fetch_leaves_nothing_beside_the_folder(tmp_path, kind):
    repo = tmp_path / "repo"
    nucleus = _nucleus(tmp_path, dict(FILES, **{"LICENSE.txt": b"another licence"}))
    with pytest.raises(ValueError, match="LICENSE.txt 의 SHA-256 이 정본과 다릅니다"):
        fr.fetch(_entry(), _where(nucleus, kind), repo)
    assert [p.name for p in (repo / "build").iterdir()] == ["unitree_h1"]
    assert list((repo / "build/unitree_h1").iterdir()) == []


def test_a_missing_source_is_said(tmp_path):
    nucleus = _nucleus(tmp_path)
    (nucleus / "Robots/Unitree/H1/instanceable_meshes.usd").unlink()
    with pytest.raises(FileNotFoundError, match="Robots/Unitree/H1/instanceable_meshes.usd"):
        fr.fetch(_entry(), str(nucleus), tmp_path / "repo")


def test_a_row_outside_the_nucleus_root_is_refused(tmp_path):
    entry = _entry()
    entry["ship"]["files"][0]["url"] = "https://example.invalid/h1.usd"
    with pytest.raises(ValueError, match="url 이 \\{ISAACLAB_NUCLEUS_DIR\\} 아래가 아닙니다"):
        fr.fetch(entry, str(_nucleus(tmp_path)), tmp_path / "repo")


def test_a_ship_block_the_shipping_rule_refuses_is_refused_before_anything_lands(tmp_path):
    entry = _entry()
    entry["usd_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="usd_sha256 과 다릅니다"):
        fr.fetch(entry, str(_nucleus(tmp_path)), tmp_path / "repo")
    assert not (tmp_path / "repo").exists()


def test_the_main_reads_the_robot_from_the_canon(tmp_path, capsys):
    repo = tmp_path / "repo"
    (repo / "meta").mkdir(parents=True)
    (repo / "meta/assets.json").write_text(json.dumps({"robots": {"unitree_h1": _entry()}, "scene": {}}),
                                           encoding="utf-8")
    assert fr.main(["--robot", "unitree_h1", "--nucleus", str(_nucleus(tmp_path)), "--repo", str(repo)]) == 0
    assert (repo / "build/unitree_h1/h1.usd").read_bytes() == FILES["H1/h1.usd"]
    assert "unitree_h1" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="robots 에 'nobody' 가 없습니다"):
        fr.main(["--robot", "nobody", "--nucleus", str(_nucleus(tmp_path)), "--repo", str(repo)])


# ---- the canon ----------------------------------------------------------------------------------

def test_the_canon_robot_fetches_into_the_ignored_build_folder():
    """The H1's ship rows come from the BSD-3 Isaac Lab path (the USD beside the layer it
    references, the licence one folder up), and the folder they land in is not tracked."""
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    entry = json.loads((repo / "meta/assets.json").read_text(encoding="utf-8"))["robots"]["unitree_h1"]
    ship = entry["ship"]
    assert ship["source_dir"] == f"{fr.BUILD}/{ship['dir']}"
    assert [r["url"] for r in ship["files"]] == [
        "{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
        "{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/instanceable_meshes.usd",
        "{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/LICENSE.txt"]
    # the licence file: the USDs are ignored by their suffix wherever they are
    ignored = subprocess.run(["git", "check-ignore", "-q", f"{ship['source_dir']}/LICENSE.txt"], cwd=repo)
    assert ignored.returncode == 0, f"{ship['source_dir']} 가 git 이 무시하는 자리가 아닙니다"
