"""The built jointed dummy with the vest: every body sits, at rest, with its frame's origin at its part's primitive
centre (generation/dummy/anthropometry.yaml, the feet of generation/dummy/joints.yaml) and unturned, and its
collider is the primitive generation/hri_geometry.py measures distances to -- at the body's origin, its sizes the
spec's. The sample points (generation/visibility.py points_world) and the distances lean on both. Needs pxr and the
built USD (generation/dummy/build_dummy.py, not tracked): runs under the Isaac python, skips elsewhere."""
import json
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from generation import hri_geometry as hri  # noqa: E402
from generation.dummy import joints_spec  # noqa: E402
from generation.dummy.skin import region_split  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ASSET = "keti_dummy_ts15066_v2_vest"
ROOT = "/KetiDummy"
TOL = 1e-6


@pytest.fixture(scope="module")
def stage():
    entry = json.loads((REPO / "meta" / "assets.json").read_text(encoding="utf-8"))["scene"][ASSET]
    usd = REPO / entry["usd_path"]
    if not usd.is_file():
        pytest.skip(f"빌드한 인형 USD 가 없습니다: {entry['usd_path']} (generation/dummy/build_dummy.py 로 지음)")
    return Usd.Stage.Open(str(usd))


def _primitives(spec):
    """{body prim name: primitive} -- the spec's parts and feet, read as the builder reads them."""
    prims = region_split.primitives_from_anthropometry(spec.anthropometry) + joints_spec.foot_primitives(spec)
    return {p.name: p for p in prims}


def test_vest_body_frames_sit_at_their_primitives(stage):
    spec = joints_spec.load()
    primitives = _primitives(spec)
    shapes = hri.human_shapes(spec)
    bodies = {p.GetName(): p for p in stage.GetPrimAtPath(ROOT).GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    assert set(bodies) == set(shapes) == set(primitives)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for name, prim in bodies.items():
        frame = cache.GetLocalToWorldTransform(prim)
        assert list(frame.ExtractTranslation()) == pytest.approx(list(primitives[name].center), abs=TOL), name
        turn = frame.ExtractRotationQuat()
        assert abs(turn.GetReal()) == pytest.approx(1.0, abs=TOL), name

        collider = prim.GetChild("collider")
        local = UsdGeom.Xformable(collider).GetLocalTransformation()
        assert list(local.ExtractTranslation()) == pytest.approx([0.0, 0.0, 0.0], abs=TOL), name
        shape = shapes[name]
        if shape.kind == "sphere":
            assert collider.IsA(UsdGeom.Sphere), name
            assert collider.GetAttribute("radius").Get() == pytest.approx(shape.half[0], abs=TOL), name
        elif shape.kind == "capsule":
            assert collider.IsA(UsdGeom.Capsule), name
            assert collider.GetAttribute("radius").Get() == pytest.approx(shape.half[0], abs=TOL), name
            assert collider.GetAttribute("height").Get() == pytest.approx(2.0 * shape.half[1], abs=TOL), name
            assert str(collider.GetAttribute("axis").Get()).lower() == shape.axis, name
        else:
            assert collider.IsA(UsdGeom.Cube), name
            size = collider.GetAttribute("size").Get()
            scale = [local.GetRow(i).GetLength() for i in range(3)]
            assert [size * s / 2.0 for s in scale] == pytest.approx(list(shape.half), abs=TOL), name
