"""build_dummy.py --jointed -- the v2 dummy: v1 rebuilt to the byte, ten joints where v1 had fixed ones,
two feet without a body region, densities from the spec and no physics:mass. Needs pxr."""
import argparse
import hashlib
import json
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdPhysics  # noqa: E402

from generation.dummy import build_dummy as bd  # noqa: E402
from generation.dummy import joints_spec as js  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CANON = json.loads((REPO / "meta/assets.json").read_text(encoding="utf-8"))["scene"]


def _build(tmp, **kw):
    out = tmp / kw.pop("name")
    args = argparse.Namespace(out=str(out), geometry_only=False, root="dynamic", skin=None, jointed=False, gait=None)
    for k, v in kw.items():
        setattr(args, k, v)
    assert bd.build(args) == 0
    return out


def test_v1_is_rebuilt_to_the_byte(tmp_path):
    out = _build(tmp_path, name="v1.usd")
    assert hashlib.sha256(out.read_bytes()).hexdigest() == CANON["keti_dummy_ts15066_v1"]["usd_sha256"]


def test_v2_has_the_spec_joints_and_feet(tmp_path):
    out = _build(tmp_path, name="v2.usd", jointed=True)
    st = Usd.Stage.Open(str(out))
    spec = js.load()
    kinds = {}
    for p in st.Traverse():
        if p.IsA(UsdPhysics.Joint):
            kinds.setdefault(p.GetTypeName(), []).append(p.GetName())
    assert sorted(kinds["PhysicsRevoluteJoint"]) == ["elbow_left", "elbow_right", "knee_left", "knee_right"]
    assert sorted(kinds["PhysicsJoint"]) == sorted(j.name for j in spec.joints if j.kind == "d6")
    # hips, knees, shoulders and elbows replace eight of v1's sixteen fixed joints; the ankles are new
    assert len(kinds["PhysicsFixedJoint"]) == 16 - 8
    for side in ("left", "right"):
        foot = st.GetPrimAtPath(f"/KetiDummy/FOOT_{side}")
        assert foot.HasAPI(UsdPhysics.RigidBodyAPI)
        assert foot.GetAttribute("keti:groundContact").Get() is True
        assert not foot.GetAttribute("keti:bodyRegion")


def test_v2_writes_density_not_mass(tmp_path):
    out = _build(tmp_path, name="v2.usd", jointed=True)
    st = Usd.Stage.Open(str(out))
    spec = js.load()
    seen = {}
    for p in st.Traverse():
        assert not p.GetAttribute("physics:mass").HasAuthoredValue(), p.GetPath()
        d = p.GetAttribute("physics:density")
        if d and d.HasAuthoredValue():
            body = p if p.HasAPI(UsdPhysics.RigidBodyAPI) else p.GetParent()
            seen[body.GetName()] = d.Get()
    assert seen.keys() == spec.density.keys()
    for name, rho in spec.density.items():
        assert seen[name] == pytest.approx(rho, rel=1e-6)


def test_v2_skin_puts_the_feet_skin_on_the_feet(tmp_path):
    skin = REPO / "generation/dummy/skin/build/keti_skin.obj"
    if not skin.is_file():
        pytest.skip("피부 OBJ 가 빌드되어 있지 않습니다")
    out = _build(tmp_path, name="v2_skin.usd", jointed=True, skin=str(skin))
    st = Usd.Stage.Open(str(out))
    from pxr import UsdSemantics
    for side in ("left", "right"):
        m = st.GetPrimAtPath(f"/KetiDummy/FOOT_{side}/skin")
        assert m and list(UsdSemantics.LabelsAPI(m, "class").GetLabelsAttr().Get()) == ["BR-12"]


def test_the_physics_check_reads_the_density_on_the_shape(tmp_path):
    from generation.dummy import check_skin_physics as csp

    out = _build(tmp_path, name="v2.usd", jointed=True)
    desc, unsupported = csp.describe(out)
    assert unsupported == []
    for name, rho in js.load().density.items():
        mass = desc["bodies"][name]["shape"]["mass"]
        densities = [float(m.split("=", 1)[1]) for m in mass if m.startswith("physics:density=")]
        assert "api PhysicsMassAPI" in mass and densities == [pytest.approx(rho, rel=1e-6)]


def test_the_physics_check_refuses_a_mass_beside_the_density(tmp_path):
    from generation.dummy import check_skin_physics as csp

    out = _build(tmp_path, name="v2.usd", jointed=True)
    st = Usd.Stage.Open(str(out))
    UsdPhysics.MassAPI(st.GetPrimAtPath("/KetiDummy/BR_04")).CreateMassAttr(30.0)
    st.GetRootLayer().Save()
    _, unsupported = csp.describe(out)
    assert any("MassAPI" in u and "physics:mass" in u for u in unsupported), unsupported


def test_v2_carries_the_gait_table_it_was_given(tmp_path):
    from generation.dummy import jointed
    from generation.dummy.gait import gait_table as gt
    import numpy as np

    names = tuple(js.dof_names(js.load()))
    table = gt.GaitTable(names, np.array([0.0, 0.5, 1.5]), np.array([0.0, 0.8, 1.6]),
                         np.zeros((3, 4, len(names)), np.float32))
    table = gt.GaitTable(table.dof_names, table.speeds, table.stride,
                         np.concatenate([table.targets[:1], np.full((2, 4, len(names)), 0.1, np.float32)]))
    g = tmp_path / "gait.json"
    g.write_text(json.dumps(table.to_json()), encoding="utf-8")
    out = _build(tmp_path, name="v2g.usd", jointed=True, gait=str(g))
    back = jointed.read_gait(out)
    assert back.dof_names == table.dof_names
    assert (back.targets == table.targets).all()


def test_body_codes_read_feet_as_ground(tmp_path):
    from generation.dummy import jointed
    out = _build(tmp_path, name="v2c.usd", jointed=True)
    codes = jointed.read_body_codes(str(out), ["BR_04", "FOOT_left", "BR_12_right"])
    assert codes == ["BR-04", None, "BR-12"]


def test_a_body_without_either_tag_is_refused(tmp_path):
    from pxr import Usd
    from generation.dummy import jointed
    out = _build(tmp_path, name="v2d.usd", jointed=True)
    st = Usd.Stage.Open(str(out))
    st.GetPrimAtPath("/KetiDummy/FOOT_left").GetAttribute("keti:groundContact").Set(False)
    st.GetRootLayer().Save()
    with pytest.raises(ValueError, match="FOOT_left"):
        jointed.read_body_codes(str(out), ["FOOT_left"])


def _recorder_ground_bodies():
    """recorder.ground_contact_body_names as the recorder's source has it (the recorder imports Isaac, so the
    function is compiled from its syntax tree with the names it reads)."""
    import ast

    from generation.dummy import jointed
    from generation.envs.s1a.mdp.safety import GROUND_REGION_CODE, dummy_body_masks

    src = REPO / "generation/envs/s1a/recorder.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)
           and n.name in ("read_region_codes", "ground_contact_body_names")]
    ns = {"Usd": Usd, "_GROUND_REGION_CODE": GROUND_REGION_CODE, "jointed": jointed,
          "dummy_body_masks": dummy_body_masks}
    exec(compile(ast.fix_missing_locations(ast.Module(body=fns, type_ignores=[])), str(src), "exec"), ns)
    return ns["ground_contact_body_names"]


def test_the_check_scripts_ground_bodies_are_the_jointed_dummys_feet(tmp_path):
    # The check scripts gate ground reactions on these bodies: on v2 the feet carry the weight, not the shins.
    out = _build(tmp_path, name="v2e.usd", jointed=True)
    names = ["BR_04", "BR_12_left", "BR_12_right", "FOOT_left", "FOOT_right"]
    assert _recorder_ground_bodies()(str(out), names) == {"FOOT_left", "FOOT_right"}


def test_the_check_scripts_ground_bodies_are_the_rigid_dummys_shins(tmp_path):
    out = _build(tmp_path, name="v1e.usd")
    names = ["BR_04", "BR_12_left", "BR_12_right"]
    assert _recorder_ground_bodies()(str(out), names) == {"BR_12_left", "BR_12_right"}


# ---- the vest (build_dummy.py --vest) ------------------------------------------------------

SKIN_OBJ = REPO / "generation/dummy/skin/build/keti_skin.obj"


@pytest.fixture(scope="module")
def vested(tmp_path_factory):
    """The jointed dummy in its vest, built once. The build itself compares it with the bare v2 as the
    engine reads them (verify_engine_identity) and fails if they differ -- so a built file already says
    the vest added nothing physical."""
    if not SKIN_OBJ.is_file():
        pytest.skip("피부 OBJ 가 빌드되어 있지 않습니다")
    from generation.dummy import vest
    return _build(tmp_path_factory.mktemp("vest"), name="v2_vest.usd", jointed=True, skin=str(SKIN_OBJ),
                  vest=str(vest.VEST_SPEC))


def _vest_meshes(stage):
    return {p.GetParent().GetName(): p for p in stage.Traverse() if p.GetName() == bd.VEST_CHILD}


def test_the_vest_hangs_meshes_under_the_trunk_bodies_it_covers_only(vested):
    from pxr import UsdGeom
    from generation.dummy import vest
    from generation.dummy.skin import region_split as rs

    st = Usd.Stage.Open(str(vested))
    spec = vest.load_vest_spec()
    regions = js.load().anthropometry["regions"]
    want = {name for code in spec.regions for name, _ in rs.prim_names(code, regions[code])}
    got = _vest_meshes(st)
    assert set(got) == want == {"BR_04", "BR_05", "BR_06", "BR_07"}
    for name, mesh in got.items():
        assert mesh.IsA(UsdGeom.Mesh)
        assert mesh.GetParent().GetAttribute("keti:bodyRegion").Get() in spec.regions
        assert len(UsdGeom.Mesh(mesh).GetFaceVertexCountsAttr().Get()) > 0


def test_the_vest_carries_its_part_class_and_no_physics(vested):
    from pxr import UsdGeom, UsdSemantics, UsdShade
    from generation.dummy import vest

    st = Usd.Stage.Open(str(vested))
    spec = vest.load_vest_spec()
    shaders = {}
    for name, mesh in _vest_meshes(st).items():
        code = mesh.GetParent().GetAttribute("keti:bodyRegion").Get()
        assert list(UsdSemantics.LabelsAPI(mesh, "class").GetLabelsAttr().Get()) == [code]
        assert mesh.GetAttribute("keti:visualOnly").Get() is True
        assert UsdGeom.Imageable(mesh).ComputePurpose() == UsdGeom.Tokens.default_
        assert not any(s.startswith(("Physics", "Physx")) for s in mesh.GetAppliedSchemas()), name
        assert not any(a.GetName().startswith("physics:") for a in mesh.GetAttributes()), name
        holders = [s.GetPrim() for s in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(mesh))] or [mesh]
        for holder in holders:
            material = UsdShade.MaterialBindingAPI(holder).ComputeBoundMaterial()[0]
            assert material, holder.GetPath()
            shaders[material.GetPath().name] = UsdShade.Shader(material.GetSurfaceOutput("mdl").GetConnectedSource()[0])
    assert set(shaders) == set(vest.MATERIALS)
    for name, sh in shaders.items():
        m = spec.materials[name]
        assert tuple(sh.GetInput("diffuse_color_constant").Get()) == pytest.approx(m["color"])
        assert sh.GetInput("metallic_constant").Get() == pytest.approx(m["metallic"])
        assert sh.GetInput("reflection_roughness_constant").Get() == pytest.approx(m["roughness"])
        assert not sh.GetInput("diffuse_texture"), f"{name}: 조끼 재질이 텍스처를 참조합니다"
    # the asset names the spec its vest was cut by
    assert st.GetPrimAtPath("/KetiDummy").GetCustomDataByKey("keti:vest")["sourceObject"] == spec.source_object


def test_a_vest_without_the_jointed_skin_is_refused(tmp_path):
    from generation.dummy import vest

    for kw in ({"jointed": True}, {"skin": str(SKIN_OBJ)}):
        with pytest.raises(SystemExit) as e:
            _build(tmp_path, name="refused.usd", vest=str(vest.VEST_SPEC), **kw)
        assert e.value.code == 3
        assert not (tmp_path / "refused.usd").exists()


def test_v1_skin_still_builds_to_its_pin(tmp_path):
    """The skin material takes a metallic input for the vest's band; at its default the v1 skinned dummy
    is the same bytes the canon pins."""
    if not SKIN_OBJ.is_file():
        pytest.skip("피부 OBJ 가 빌드되어 있지 않습니다")
    entry = CANON["keti_dummy_ts15066_v1_skin"]
    out = _build(tmp_path, name=Path(entry["usd_path"]).name, skin=str(SKIN_OBJ))
    assert hashlib.sha256(out.read_bytes()).hexdigest() == entry["usd_sha256"]


def _canon_build_kwargs(key):
    """The build_dummy arguments meta/assets.json gives for an entry (build_args), as _build keywords. Skips where an
    input it names is not built; an argument this parser does not know fails."""
    import shlex

    words = shlex.split(CANON[key]["build_args"])
    kw = {}
    while words:
        flag = words.pop(0)
        if flag in ("--jointed", "--split-worksuit"):
            kw[flag[2:].replace("-", "_")] = True
        elif flag in ("--gait", "--skin", "--vest", "--seams"):
            path = REPO / words.pop(0)
            if not path.is_file():
                pytest.skip(f"{path.name} 가 빌드되어 있지 않습니다")
            kw[flag[2:]] = str(path)
        else:
            raise AssertionError(f"{key}: build_args 에 이 시험이 모르는 인자가 있습니다: {flag}")
    return kw


@pytest.fixture(scope="module")
def canon_built(tmp_path_factory):
    """Each jointed dummy built once from its canon arguments, on first use."""
    built = {}

    def get(key):
        if key not in built:
            kw = _canon_build_kwargs(key)
            built[key] = _build(tmp_path_factory.mktemp(key), name=Path(CANON[key]["usd_path"]).name, **kw)
        return built[key]
    return get


@pytest.mark.parametrize("key", ["keti_dummy_ts15066_v2_jointed", "keti_dummy_ts15066_v2_vest"])
def test_the_jointed_dummies_rebuild_from_their_canon_arguments_to_their_pins(canon_built, key):
    """meta/assets.json says how each jointed dummy is built (build_args) and pins its bytes. Building
    with exactly those arguments gives those bytes: the arguments are the real ones and the build is
    deterministic. Skips where the gait table or the skin export is not built."""
    out = canon_built(key)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == CANON[key]["usd_sha256"]


def test_the_canon_argument_parser_knows_the_split_and_still_refuses_a_stranger(monkeypatch):
    monkeypatch.setitem(CANON, "probe", {"build_args": "--jointed --split-worksuit"})
    assert _canon_build_kwargs("probe") == {"jointed": True, "split_worksuit": True}
    monkeypatch.setitem(CANON, "probe", {"build_args": "--jointed --tint-worksuit"})
    with pytest.raises(AssertionError, match="--tint-worksuit"):
        _canon_build_kwargs("probe")


# ---- the skin seams (build_dummy.py --seams) ------------------------------------------------

@pytest.fixture(scope="module")
def seamed(tmp_path_factory):
    """The jointed dummy in its vest with the skin seams, built once (the build checks the blend itself:
    verify_skin_usd's _seam_problems)."""
    if not SKIN_OBJ.is_file():
        pytest.skip("피부 OBJ 가 빌드되어 있지 않습니다")
    from generation.dummy import skin_seams, vest
    return _build(tmp_path_factory.mktemp("seams"), name="v2_vest_seams.usd", jointed=True, skin=str(SKIN_OBJ),
                  vest=str(vest.VEST_SPEC), seams=str(skin_seams.SPEC_PATH))


def _skins(stage):
    return {p.GetParent().GetName(): p for p in stage.Traverse() if p.GetName() == bd.SKIN_CHILD}


def test_the_seamed_skin_carries_a_blend_on_the_jointed_pieces_and_names_its_spec(seamed):
    from generation.dummy import skin_seams

    st = Usd.Stage.Open(str(seamed))
    skins = _skins(st)
    carrying = {name for name, mesh in skins.items() if mesh.GetAttribute(skin_seams.ATTR_POINTS)}
    # every body a joint turns, and the trunk pieces the hips and shoulders meet
    assert {j.child for j in js.load().joints} <= carrying
    assert {"BR_04", "BR_07"} <= carrying
    bodies = {p.GetName() for p in st.GetPrimAtPath("/KetiDummy").GetChildren()}
    for name in carrying:
        problems, held = bd._seam_problems(skins[name], bodies)
        assert held == 1 and problems == [], (name, problems)
    data = st.GetPrimAtPath("/KetiDummy").GetCustomDataByKey(skin_seams.CUSTOM_DATA_KEY)
    assert data["specSha256"] == hashlib.sha256(skin_seams.SPEC_PATH.read_bytes()).hexdigest()
    assert data["movedPatches"] > 0 and data["blendedPoints"] > 1000


def test_the_seams_leave_the_vest_as_it_was(seamed, vested):
    from pxr import UsdGeom

    a, b = Usd.Stage.Open(str(seamed)), Usd.Stage.Open(str(vested))
    for name, mesh in _vest_meshes(a).items():
        other = _vest_meshes(b)[name]
        for attr in ("points", "faceVertexIndices", "normals"):
            assert list(mesh.GetAttribute(attr).Get()) == list(other.GetAttribute(attr).Get()), (name, attr)
    assert not any(m.GetAttribute("keti:seamBlend:points") for m in _skins(b).values())
    assert len(_skins(a)) == len(_skins(b))
    assert sum(len(UsdGeom.Mesh(m).GetFaceVertexCountsAttr().Get()) for m in _skins(a).values()) == \
        sum(len(UsdGeom.Mesh(m).GetFaceVertexCountsAttr().Get()) for m in _skins(b).values())


def test_seams_without_the_jointed_skin_are_refused(tmp_path):
    from generation.dummy import skin_seams

    for kw in ({"jointed": True}, {"skin": str(SKIN_OBJ)}):
        with pytest.raises(SystemExit) as e:
            _build(tmp_path, name="refused.usd", seams=str(skin_seams.SPEC_PATH), **kw)
        assert e.value.code == 3
        assert not (tmp_path / "refused.usd").exists()


def test_a_blend_that_does_not_add_up_is_named(tmp_path):
    from pxr import Sdf, UsdGeom, Vt
    from generation.dummy import skin_seams

    st = Usd.Stage.CreateInMemory()
    m = UsdGeom.Mesh.Define(st, "/KetiDummy/BR_11_left/skin")
    m.CreatePointsAttr(Vt.Vec3fArray([(0, 0, 0)] * 4))
    prim = m.GetPrim()
    prim.CreateAttribute(skin_seams.ATTR_POINTS, Sdf.ValueTypeNames.IntArray, custom=True).Set(Vt.IntArray([0, 1]))
    prim.CreateAttribute(skin_seams.ATTR_BODIES, Sdf.ValueTypeNames.TokenArray, custom=True).Set(
        Vt.TokenArray(["BR_07", "BR_11_left"]))
    weights = prim.CreateAttribute(skin_seams.ATTR_WEIGHTS, Sdf.ValueTypeNames.FloatArray, custom=True)
    bodies = {"BR_07", "BR_11_left"}
    weights.Set(Vt.FloatArray([0.5, 0.5, 0.25, 0.75]))
    assert bd._seam_problems(prim, bodies) == ([], 1)
    weights.Set(Vt.FloatArray([0.5, 0.5, 0.25, 0.5]))
    assert "합이 1" in bd._seam_problems(prim, bodies)[0][0]
    weights.Set(Vt.FloatArray([1.0, 0.0, 0.25, 0.75]))
    assert "둘 이상" in bd._seam_problems(prim, bodies)[0][0]
    weights.Set(Vt.FloatArray([0.5, 0.5, 0.25]))
    assert "정점 2 × 열 2" in bd._seam_problems(prim, bodies)[0][0]
    weights.Set(Vt.FloatArray([0.5, 0.5, 0.25, 0.75]))
    assert "몸이 아닙니다" in bd._seam_problems(prim, {"BR_07"})[0][0]
    prim.GetAttribute(skin_seams.ATTR_POINTS).Set(Vt.IntArray([0, 4]))
    assert "범위 밖" in bd._seam_problems(prim, bodies)[0][0]


def _materials_on(mesh):
    from pxr import UsdGeom, UsdShade

    holders = [s.GetPrim() for s in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(mesh))] or [mesh]
    bound = [UsdShade.MaterialBindingAPI(h).ComputeBoundMaterial()[0] for h in holders]
    assert all(bound), mesh.GetPath()
    return {m.GetPath().name for m in bound}


def test_the_seamed_hands_carry_no_trousers_and_the_unseamed_ones_do(seamed, vested):
    """The stray patches move with the skin's split: without --seams the trousers beside a hand ride on it."""
    for path, want in ((seamed, False), (vested, True)):
        stage = Usd.Stage.Open(str(path))       # held: its prims expire with it
        skins = _skins(stage)
        for side in ("left", "right"):
            on_hand = "Human_male_worksuit01" in _materials_on(skins[f"BR_10_{side}"])
            assert on_hand is want, (path.name, side)


# ---- the work shirt and trousers (build_dummy.py --split-worksuit) ---------------------------------------

def _faces_by_material(mesh):
    """{material name: the mesh's face indices bound to it}, through the materialBind subsets or the mesh's own binding."""
    from pxr import UsdGeom, UsdShade

    subsets = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(mesh))
    if not subsets:
        n = len(UsdGeom.Mesh(mesh).GetFaceVertexCountsAttr().Get())
        return {UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0].GetPath().name: set(range(n))}
    out = {}
    for sub in subsets:
        name = UsdShade.MaterialBindingAPI(sub.GetPrim()).ComputeBoundMaterial()[0].GetPath().name
        out.setdefault(name, set()).update(int(i) for i in sub.GetIndicesAttr().Get())
    return out


def _shader_inputs(stage, name):
    """A material's shader as authored: its inputs (an asset by the path it names -- the two builds resolve it beside
    different files) and its MDL source."""
    from pxr import Sdf, UsdShade

    def authored(value):
        return value.path if isinstance(value, Sdf.AssetPath) else value

    material = UsdShade.Material(stage.GetPrimAtPath(f"{bd.LOOKS}/{name}"))
    assert material, name
    sh = UsdShade.Shader(material.GetSurfaceOutput("mdl").GetConnectedSource()[0])
    return ({i.GetBaseName(): (str(i.GetTypeName()), authored(i.Get())) for i in sh.GetInputs()},
            authored(sh.GetSourceAsset("mdl")), sh.GetSourceAssetSubIdentifier("mdl"))


def test_the_vest_dummy_paints_shirt_and_trousers_apart(canon_built, seamed):
    """The canon vest dummy binds the worksuit's faces to a shirt and a trousers material and nothing to the worksuit
    material; each carries the inputs the worksuit material carries in the same build without the split (seamed --
    the gait only adds the table), so the built look is the same. The pieces keep their points, faces, normals, UVs
    and class; only the worksuit's faces change hands. The shirt faces are the ones shirt_faces picks, and every one
    of them samples a white texel at its UV centroid while no other worksuit face does."""
    import numpy as np
    from pxr import UsdSemantics
    from PIL import Image

    from generation.dummy.skin import region_split as rs

    split, plain = Usd.Stage.Open(str(canon_built("keti_dummy_ts15066_v2_vest"))), Usd.Stage.Open(str(seamed))
    a, b = _skins(split), _skins(plain)
    assert set(a) == set(b)
    suit = bd.WORKSUIT_MATERIAL
    parts = (bd.SHIRT_MATERIAL, bd.TROUSERS_MATERIAL)
    counts = dict.fromkeys(parts, 0)
    for name, mesh in a.items():
        other = b[name]
        for attr in ("points", "faceVertexIndices", "faceVertexCounts", "normals", "primvars:st"):
            assert list(mesh.GetAttribute(attr).Get()) == list(other.GetAttribute(attr).Get()), (name, attr)
        assert list(UsdSemantics.LabelsAPI(mesh, "class").GetLabelsAttr().Get()) == \
            list(UsdSemantics.LabelsAPI(other, "class").GetLabelsAttr().Get())
        got, was = _faces_by_material(mesh), _faces_by_material(other)
        assert suit not in got, name
        assert not set(parts) & set(was), name
        assert {k: v for k, v in got.items() if k not in parts} == {k: v for k, v in was.items() if k != suit}, name
        assert set().union(*(got.get(p, set()) for p in parts)) == was.get(suit, set()), name
        for p in parts:
            counts[p] += len(got.get(p, ()))
    old = _shader_inputs(plain, suit)
    for p in parts:
        assert _shader_inputs(split, p) == old, p
    assert not split.GetPrimAtPath(f"{bd.LOOKS}/{suit}")

    mesh = rs.read_obj(SKIN_OBJ)
    mtl = rs.read_mtl(SKIN_OBJ.parent / mesh.mtllib)
    texture = SKIN_OBJ.parent / mtl["Human.male_worksuit01"].map_kd
    shirt = bd.shirt_faces(mesh, texture)
    worksuit = np.nonzero(mesh.FM == mesh.materials.index("Human.male_worksuit01"))[0]
    assert counts == {bd.SHIRT_MATERIAL: len(shirt), bd.TROUSERS_MATERIAL: len(worksuit) - len(shirt)}
    assert 0 < len(shirt) < len(worksuit)
    img = np.asarray(Image.open(texture).convert("RGB"), float) / 255.0
    h, w = img.shape[:2]
    centre = mesh.VT[mesh.FT[worksuit]].mean(axis=1)
    texel = img[np.clip(((1.0 - centre[:, 1]) * h).astype(int), 0, h - 1), np.clip((centre[:, 0] * w).astype(int), 0, w - 1)]
    on_shirt = np.isin(worksuit, shirt)
    assert texel[on_shirt].min() >= 0.95
    assert (texel[~on_shirt] @ np.array([0.2126, 0.7152, 0.0722])).max() < 0.9


def test_a_worksuit_split_without_the_skin_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        _build(tmp_path, name="refused.usd", jointed=True, split_worksuit=True)
    assert e.value.code == 3
    assert not (tmp_path / "refused.usd").exists()


def test_the_build_check_counts_the_shirt_and_trousers_faces(canon_built):
    """verify_skin_usd reads the split back from the file: the faces bound to each part are the faces the split
    made, and a file without the split may bind none to either. A shirt piece rebound to the trousers is named, and
    so is the split file checked as if it had been built without the flag."""
    import shutil

    from pxr import UsdGeom, UsdShade

    from generation.dummy.skin import region_split as rs

    built = canon_built("keti_dummy_ts15066_v2_vest")
    copy = built.parent / "checked.usd"               # beside the build's textures/, which the check resolves
    shutil.copyfile(built, copy)
    mesh = rs.read_obj(SKIN_OBJ)
    shirt = len(bd.shirt_faces(mesh, SKIN_OBJ.parent / rs.read_mtl(SKIN_OBJ.parent / mesh.mtllib)[
        "Human.male_worksuit01"].map_kd))
    suit = int((mesh.FM == mesh.materials.index("Human.male_worksuit01")).sum())
    split = {bd.SHIRT_MATERIAL: shirt, bd.TROUSERS_MATERIAL: suit - shirt}
    st = Usd.Stage.Open(str(copy))
    vest = {name: len(UsdGeom.Mesh(m).GetFaceVertexCountsAttr().Get()) for name, m in _vest_meshes(st).items()}

    def check(**kw):
        return bd.verify_skin_usd(copy, len(mesh.F), vest=vest, seams=True, **kw)

    assert check(worksuit=split) == []
    assert any("--split-worksuit 없이" in p for p in check())
    held = [s for m in _skins(st).values() for s in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(m))
            if s.GetPrim().GetName() == bd.SHIRT_MATERIAL]
    assert held, "상의 면을 묶은 서브셋이 없습니다"
    trousers = UsdShade.Material(st.GetPrimAtPath(f"{bd.LOOKS}/{bd.TROUSERS_MATERIAL}"))
    UsdShade.MaterialBindingAPI(held[0].GetPrim()).Bind(trousers)
    st.GetRootLayer().Save()
    moved = len(held[0].GetIndicesAttr().Get())
    problems = check(worksuit=split)
    assert f"{bd.SHIRT_MATERIAL} 에 묶인 피부 면 {shirt - moved} ≠ 가른 면 {shirt}" in problems, problems
