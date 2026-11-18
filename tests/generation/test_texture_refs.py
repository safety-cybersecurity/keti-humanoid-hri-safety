"""A shipped USD's ./textures/ references are exactly its bundle.files in meta/assets.json.

The converter ships what bundle.files lists, the publisher and C6 check the distribution
against it -- none of them can open a USD (no pxr where they run), so none of them can say
the list is what the USD actually needs. This does, on the built USDs. Needs pxr: runs under
the Isaac python, skips in CI; an asset that is not built here skips on its own (*.usd is
not tracked).

    PYTHONPATH= ~/isaacsim/python.sh -B -m pytest -q -p no:launch_testing -p no:launch_ros \\
        tests/generation/test_texture_refs.py
"""
import json
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Sdf, Usd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# Both builders reference a texture as ./textures/<file> (TEXTURE_SUBDIR); the distribution
# keeps that layout (generation/assets.py TEXTURE_SUBDIR).
PREFIX = "./textures/"


def _redistributed() -> dict[str, dict]:
    """The redistributable USDs of the repository that are one layer each. An entry that composes others (the
    Stage) references their files by design -- it is opened inside a release (the last test here)."""
    canon = json.loads((REPO / "meta" / "assets.json").read_text(encoding="utf-8"))
    return {k: e for group in ("robots", "scene") for k, e in (canon.get(group) or {}).items()
            if e.get("redistributed") and "composes" not in e
            and not str(e["usd_path"]).startswith(("{", "isaaclab://"))}


def _asset_references(usd: Path) -> set[str]:
    stage = Usd.Stage.Open(str(usd))
    refs = set()
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
                value = attr.Get()
                if value is not None and value.path:
                    refs.add(value.path)
    return refs


@pytest.mark.parametrize("key", sorted(_redistributed()))
def test_a_redistributed_usd_references_exactly_its_bundle(key):
    entry = _redistributed()[key]
    usd = REPO / entry["usd_path"]
    if not usd.is_file():
        pytest.skip(f"{entry['usd_path']} 가 빌드되어 있지 않습니다")
    refs = _asset_references(usd)
    textures = {r[len(PREFIX):] for r in refs if r.startswith(PREFIX)}
    listed = {row["file"] for row in (entry.get("bundle") or {}).get("files", [])}
    assert textures == listed, (
        f"{key}: USD 가 참조하는 텍스처와 meta/assets.json bundle.files 가 다릅니다 — "
        f"USD 에만 {sorted(textures - listed)}, 목록에만 {sorted(listed - textures)}")
    # Anything else it names is not a file of the asset: a bare MDL name the renderer resolves
    # from its own library (OmniPBR.mdl). A relative path here would be a file assets/ lacks.
    others = refs - {PREFIX + t for t in textures}
    assert all("/" not in r and r.endswith(".mdl") for r in others), sorted(others)
    # One layer: no sublayer, reference or payload pulls in another file.
    stage = Usd.Stage.Open(str(usd))
    files = [layer.realPath for layer in stage.GetUsedLayers() if not layer.anonymous]
    assert [Path(f).resolve() for f in files] == [usd.resolve()], files


# ---- a robot that ships as its own folder ------------------------------------------------------

# What a robot folder's USDs may leave unresolved: the renderer's own material library, a bare MDL name
# resolved from the renderer's search path -- not a file of the asset (plain pxr cannot find it).
RENDERER_LIBRARY = {"OmniPBR.mdl"}


def _shipping_robots() -> dict[str, dict]:
    canon = json.loads((REPO / "meta" / "assets.json").read_text(encoding="utf-8"))
    return {k: e for k, e in (canon.get("robots") or {}).items() if e.get("redistributed") and "ship" in e}


@pytest.mark.parametrize("key", sorted(_shipping_robots()))
def test_a_robot_folder_holds_every_layer_its_usd_needs(key):
    """The folder ships as fetched: the USD's layers are all files of the folder (rows of ship.files), each
    reached by a relative path, so they resolve beside it under assets/<dir>/ as they did where they were
    fetched. Only where the robot was fetched (generation/robots/fetch_robot.py)."""
    from pxr import UsdUtils

    entry = _shipping_robots()[key]
    folder = (REPO / entry["ship"]["source_dir"]).resolve()
    usd = folder / Path(entry["usd_path"]).name
    if not usd.is_file():
        pytest.skip(f"{key}: 로봇을 받아 두지 않았습니다 ({entry['ship']['source_dir']}) — fetch_robot.py")
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(usd))
    files = sorted(Path(layer.realPath).resolve().relative_to(folder).as_posix() for layer in layers)
    rows = sorted(row["file"] for row in entry["ship"]["files"])
    assert usd.name in files and set(files) <= set(rows), (files, rows)
    assert len(files) > 1, "참조하는 레이어가 없는 로봇 USD 입니다 — 폴더째 실을 까닭을 다시 보십시오"
    # Any other file a layer names (a texture) is a file of the folder too.
    others = [Path(a).resolve() for a in assets]
    assert all(o.is_relative_to(folder) and o.relative_to(folder).as_posix() in rows for o in others), assets
    assert set(unresolved) <= RENDERER_LIBRARY, unresolved
    for layer in layers:
        for ref in layer.GetExternalReferences():
            if ref in RENDERER_LIBRARY:
                continue
            assert not Path(ref).is_absolute() and "://" not in ref and not ref.startswith("{"), (layer.realPath, ref)


# ---- the Stage, opened inside a release ------------------------------------------------------------------

def test_the_stage_resolves_inside_a_release(tmp_path):
    """The Stage references what it composes by the names those ship under: laid out as shipped_asset_files lays
    a release's assets/ for a batch that uses all of them, every layer and every file it needs is one of that
    assets/, and nothing is left unresolved but the renderer's own material library. Only where the Stage, what
    it composes and the robot's folder are built and fetched."""
    import shutil

    from pxr import UsdUtils

    from generation.assets import check_shipped_sources, load_assets, shipped_asset_files

    canon = load_assets(REPO)
    stages = {k: e for k, e in canon["scene"].items() if "composes" in e}
    assert stages, "엮은 자산(composes)이 정본에 없습니다"
    by_key = {k: e for g in ("robots", "scene") for k, e in canon[g].items()}
    for key, entry in sorted(stages.items()):
        files = shipped_asset_files(canon, [by_key[k]["usd_path"] for k in entry["composes"]], REPO)
        missing = sorted(str(f.src.relative_to(REPO)) for f in files.values() if not f.src.is_file())
        if missing:
            pytest.skip(f"빌드 · 받기되지 않은 파일: {missing}")
        check_shipped_sources(files, REPO)
        release = tmp_path / key
        for rel, f in files.items():
            (release / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f.src, release / rel)
        stage = release / "assets" / Path(entry["usd_path"]).name
        assert stage.is_file(), sorted(files)
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(stage))
        base = (release / "assets").resolve()
        reals = [Path(layer.realPath).resolve() for layer in layers]
        assert all(r.is_relative_to(base) for r in reals), reals
        names = sorted(r.relative_to(base).as_posix() for r in reals)
        composed = sorted(rel[len("assets/"):] for rel, f in files.items()
                          if f.asset in entry["composes"] and rel.endswith(".usd"))
        assert set(composed) <= set(names), (composed, names)
        assert all(Path(a).resolve().is_relative_to(base) for a in assets), assets
        assert set(unresolved) <= RENDERER_LIBRARY, unresolved
