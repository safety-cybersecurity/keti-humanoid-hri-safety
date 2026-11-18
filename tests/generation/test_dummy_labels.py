"""generation/envs/s1d/dummy_labels.py — S1d labels the v1 dummy's regions for the head camera
when it runs without the skin (the capsule arm the detection-rate comparison needs). Needs pxr:
runs under the Isaac python, skips in CI."""
import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdSemantics, Sdf  # noqa: E402

from generation.envs.s1d import dummy_labels  # noqa: E402


def _stage():
    st = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(st, "/Dummy")
    for name, code in (("BR_01", "BR-01"), ("BR_08_left", "BR-08")):
        body = UsdGeom.Xform.Define(st, f"/Dummy/{name}").GetPrim()
        body.CreateAttribute("keti:bodyRegion", Sdf.ValueTypeNames.String).Set(code)
        UsdGeom.Cube.Define(st, f"/Dummy/{name}/geom")
    UsdGeom.Xform.Define(st, "/Dummy/joints")
    return st


def _labels(prim):
    attr = UsdSemantics.LabelsAPI(prim, "class").GetLabelsAttr()
    return list(attr.Get()) if attr and attr.HasAuthoredValue() else []


def test_each_region_is_labelled_with_its_own_code():
    st = _stage()
    assert dummy_labels.label_regions(st.GetPrimAtPath("/Dummy")) == 2
    assert _labels(st.GetPrimAtPath("/Dummy/BR_01")) == ["BR-01"]
    assert _labels(st.GetPrimAtPath("/Dummy/BR_08_left")) == ["BR-08"]
    assert _labels(st.GetPrimAtPath("/Dummy/joints")) == []


def test_a_spawn_that_yields_no_regions_is_refused():
    st = Usd.Stage.CreateInMemory()
    empty = UsdGeom.Xform.Define(st, "/Nothing").GetPrim()
    spawn = dummy_labels.labelled(lambda prim_path, cfg, *a, **k: empty)
    with pytest.raises(ValueError, match="keti:bodyRegion"):
        spawn("/Nothing", None)


def test_the_jointed_dummys_feet_take_the_class_the_skin_gives_them():
    # The bare v2 feet carry keti:groundContact and no region; left unlabelled they would drop out of the
    # person mask and box when the jointed dummy runs without its skin. The skinned v2 labels its feet with
    # the spec's foot class -- the bare one takes the same, and a body with the flag off stays unlabelled.
    from generation.dummy import joints_spec as js

    st = _stage()
    for name, on in (("FOOT_left", True), ("FOOT_right", True), ("prop", False)):
        body = UsdGeom.Xform.Define(st, f"/Dummy/{name}").GetPrim()
        body.CreateAttribute("keti:groundContact", Sdf.ValueTypeNames.Bool).Set(on)
    assert dummy_labels.label_regions(st.GetPrimAtPath("/Dummy")) == 4
    foot = js.load().semantic_class_feet
    assert _labels(st.GetPrimAtPath("/Dummy/FOOT_left")) == [foot] == _labels(st.GetPrimAtPath("/Dummy/FOOT_right"))
    assert _labels(st.GetPrimAtPath("/Dummy/prop")) == []
    assert _labels(st.GetPrimAtPath("/Dummy/BR_01")) == ["BR-01"]
