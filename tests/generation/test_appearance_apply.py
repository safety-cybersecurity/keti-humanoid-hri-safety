"""Applying a drawn look to an opened scene touches the scene in memory only.

Needs pxr: runs under the Isaac python, skips in CI; a USD that is not built here skips on its own.

    PYTHONPATH= ~/isaacsim/python.sh -B -m pytest -q -p no:launch_testing -p no:launch_ros \\
        tests/generation/test_appearance_apply.py
"""
import hashlib
from pathlib import Path

import pytest

pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdLux, UsdShade  # noqa: E402

from generation import appearance  # noqa: E402
from generation.dummy import build_dummy  # noqa: E402
from generation.envs.hri import appearance_apply  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
USDS = {
    "aisle": (REPO / "generation/scene/build/keti_logistics_aisle_v1.usd", "/KetiLogisticsAisle"),
    "dummy": (REPO / "generation/dummy/build/keti_dummy_ts15066_v2_jointed_skin_vest.usd", "/KetiDummy"),
    "box": (REPO / "generation/scene/build/keti_carried_box_v1.usd", "/CarriedBox"),
}
ROOTS = {"aisle": "/World/envs/env_0/Aisle", "dummy": "/World/envs/env_0/Dummy", "box": "/World/envs/env_0/Box"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def scene():
    for path, _ in USDS.values():
        if not path.exists():
            pytest.skip(f"{path.name} is not built here")
    before = {k: _sha(p) for k, (p, _) in USDS.items()}
    stage = Usd.Stage.CreateInMemory()
    for key, (path, root) in USDS.items():
        prim = stage.DefinePrim(ROOTS[key], "Xform")
        prim.GetReferences().AddReference(str(path), root)
    yield stage
    assert {k: _sha(p) for k, (p, _) in USDS.items()} == before


def _input(stage, key, shader, name):
    attr = stage.GetPrimAtPath(f"{ROOTS[key]}/{shader}").GetAttribute(f"inputs:{name}")
    return attr.Get() if attr else None


def _lights(stage):
    return [p for p in Usd.PrimRange(stage.GetPrimAtPath(ROOTS["aisle"])) if p.HasAPI(UsdLux.LightAPI)]


def _all_lights(stage):
    return [p for p in Usd.PrimRange(stage.GetPrimAtPath(ROOTS["aisle"]))
            if p.GetTypeName().endswith("Light")]


def _sample_edits():
    E = appearance.ShaderEdit
    return [
        E("dummy", "Looks/vest_yellow/Shader", "diffuse_color_constant", "set", (1.0, 0.35, 0.0)),
        E("dummy", f"Looks/{build_dummy.SHIRT_MATERIAL}/Shader", "diffuse_tint", "set", (0.12, 0.16, 0.27)),
        E("dummy", f"Looks/{build_dummy.TROUSERS_MATERIAL}/Shader", "albedo_desaturation", "set", 1.0),
        E("box", "Looks/cardboard/Shader", "diffuse_color_constant", "set", (0.6, 0.6, 0.6)),
        E("aisle", "Looks/steel_blue/Shader", "diffuse_tint", "set", (0.45, 0.85, 0.5)),
        E("aisle", "Looks/concrete_wall/Shader", "diffuse_tint", "scale", 1.1),
        E("aisle", "Looks/wall_cladding/Shader", "diffuse_tint", "scale", 0.9),
        E("aisle", "Looks/fixture_lens/Shader", "emissive_intensity", "scale", 0.5),
    ]


def test_the_edits_land_in_the_stage_and_scale_multiplies_the_built_value(scene):
    wall0 = tuple(_input(scene, "aisle", "Looks/concrete_wall/Shader", "diffuse_tint") or (1, 1, 1))
    clad0 = tuple(_input(scene, "aisle", "Looks/wall_cladding/Shader", "diffuse_tint"))
    lens0 = _input(scene, "aisle", "Looks/fixture_lens/Shader", "emissive_intensity")
    assert clad0 == pytest.approx((1.35, 1.38, 1.42))
    appearance_apply.apply(scene, ROOTS, _sample_edits(), appearance.LightEdit(scale=1.0, temperature_k=5000.0))
    assert tuple(_input(scene, "dummy", "Looks/vest_yellow/Shader", "diffuse_color_constant")) == pytest.approx((1.0, 0.35, 0.0))
    assert tuple(_input(scene, "dummy", f"Looks/{build_dummy.SHIRT_MATERIAL}/Shader", "diffuse_tint")) == pytest.approx((0.12, 0.16, 0.27))
    assert _input(scene, "dummy", f"Looks/{build_dummy.TROUSERS_MATERIAL}/Shader", "albedo_desaturation") == 1.0
    assert tuple(_input(scene, "box", "Looks/cardboard/Shader", "diffuse_color_constant")) == pytest.approx((0.6, 0.6, 0.6))
    assert tuple(_input(scene, "aisle", "Looks/steel_blue/Shader", "diffuse_tint")) == pytest.approx((0.45, 0.85, 0.5))
    # A tint that was not built is the MDL default (1, 1, 1) times the factor; a built one keeps its value.
    assert tuple(_input(scene, "aisle", "Looks/concrete_wall/Shader", "diffuse_tint")) == pytest.approx(
        tuple(c * 1.1 for c in wall0))
    assert tuple(_input(scene, "aisle", "Looks/wall_cladding/Shader", "diffuse_tint")) == pytest.approx(
        tuple(c * 0.9 for c in clad0))
    assert _input(scene, "aisle", "Looks/fixture_lens/Shader", "emissive_intensity") == pytest.approx(lens0 * 0.5)


def test_an_absent_input_is_created_with_the_mdl_type(scene):
    appearance_apply.apply(scene, ROOTS, _sample_edits(), appearance.LightEdit(scale=1.0, temperature_k=5000.0))
    prim = scene.GetPrimAtPath(f"{ROOTS['dummy']}/Looks/{build_dummy.TROUSERS_MATERIAL}/Shader")
    assert prim.GetAttribute("inputs:albedo_desaturation").GetTypeName() == Sdf.ValueTypeNames.Float
    prim = scene.GetPrimAtPath(f"{ROOTS['dummy']}/Looks/{build_dummy.SHIRT_MATERIAL}/Shader")
    assert prim.GetAttribute("inputs:diffuse_tint").GetTypeName() == Sdf.ValueTypeNames.Color3f


def test_every_light_is_scaled_and_given_the_temperature(scene):
    lights = _all_lights(scene)
    assert len(lights) == 50 and sum(p.GetTypeName() == "RectLight" for p in lights) == 49
    before = [(p.GetPath(), p.GetAttribute("inputs:intensity").Get(),
               p.GetAttribute("inputs:color").Get()) for p in lights]
    appearance_apply.apply(scene, ROOTS, [], appearance.LightEdit(scale=0.7, temperature_k=4200.0))
    for (path, intensity, color) in before:
        p = scene.GetPrimAtPath(path)
        assert p.GetAttribute("inputs:intensity").Get() == pytest.approx(intensity * 0.7)
        assert p.GetAttribute("inputs:enableColorTemperature").Get() is True
        assert p.GetAttribute("inputs:colorTemperature").Get() == pytest.approx(4200.0)
        assert p.GetAttribute("inputs:color").Get() == color


def test_nothing_outside_the_opened_stage_is_dirtied(scene):
    appearance_apply.apply(scene, ROOTS, _sample_edits(), appearance.LightEdit(scale=1.2, temperature_k=6000.0))
    for path, _ in USDS.values():
        layer = Sdf.Layer.Find(str(path))
        assert layer is not None and not layer.dirty
    assert scene.GetEditTarget().GetLayer() == scene.GetRootLayer()


def test_a_shader_that_is_not_there_is_refused(scene):
    bad = [appearance.ShaderEdit("aisle", "Looks/no_such/Shader", "diffuse_tint", "set", (1.0, 1.0, 1.0))]
    with pytest.raises(ValueError, match="no_such"):
        appearance_apply.apply(scene, ROOTS, bad, appearance.LightEdit(scale=1.0, temperature_k=5000.0))


def test_an_aisle_without_a_light_is_refused():
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim(ROOTS["aisle"], "Xform")
    with pytest.raises(ValueError, match="빛"):
        appearance_apply.apply(stage, ROOTS, [], appearance.LightEdit(scale=1.0, temperature_k=5000.0))
