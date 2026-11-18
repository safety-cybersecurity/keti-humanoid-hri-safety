"""generation/envs/s1d/skin_blend.py -- the run's side of the skin seams: the physics view's quaternions, the render
the blend hangs on, and one update written back to the skin. The arithmetic itself is skin_seams.place, tested in
tests/generation/test_skin_seams.py; the Isaac side (reading the env's prims) runs in the scenario runs.

The update test needs pxr (Vt) and skips without it."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from generation.dummy import skin_seams as ss
from generation.envs.s1d import skin_blend as sb

REPO = Path(__file__).resolve().parents[2]


def quat_xyzw(axis, deg):
    a = math.radians(deg) / 2
    v = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.array([*(v * math.sin(a)), math.cos(a)])


def test_the_view_quaternions_turn_as_their_axis_and_angle_say():
    R = sb.rotations_xyzw(np.array([quat_xyzw([0, 0, 1], 90), quat_xyzw([1, 0, 0], 180), [0, 0, 0, 1]]))
    assert np.allclose(R[0] @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(R[1] @ [0, 1, 0], [0, -1, 0])
    assert np.allclose(R[2], np.eye(3))
    q = quat_xyzw([1, 2, 3], 37)
    assert np.allclose(sb.rotations_xyzw(q[None])[0] @ sb.rotations_xyzw(q[None])[0].T, np.eye(3))


class FakeSim:
    def __init__(self, log):
        self.log = log

    def render(self, *a, **k):
        self.log.append("render")


class FakeEnv:
    def __init__(self, log):
        self.sim = FakeSim(log)


def test_attach_blends_before_every_render_and_only_once():
    log = []
    env = FakeEnv(log)
    blend = sb.SkinBlend(env, view=None, pieces=[], links=0)
    blend.update = lambda: log.append("blend")
    blend.attach()
    env.sim.render()
    env.sim.render(skip_app_pumping=True)
    assert log == ["blend", "render", "blend", "render"]
    with pytest.raises(RuntimeError, match="이미"):
        blend.attach()


def test_the_run_attaches_the_blend_whenever_it_renders():
    src = (REPO / "generation/envs/run_s1a.py").read_text(encoding="utf-8")
    block = re.search(r"\n    if CAMERAS_ON or args\.video or args\.gui:\n(.*?)\n    if SCENARIO_ID", src, re.S)
    assert block, "run_s1a 가 렌더하는 실행에서 피부 이음새를 걸지 않습니다"
    assert "SkinBlend.from_env(base_env)" in block.group(1) and ".attach()" in block.group(1)
    # after the camera rig is built (the rig renders in its warm-up) and before the first episode
    assert src.index("rig = Rig(base_env") < block.start() < src.index("for offset in range(args.episodes)")


class FakeAttr:
    def __init__(self):
        self.value = None

    def Set(self, value):
        self.value = np.asarray(value)


class FakeView:
    def __init__(self, tf):
        self.tf = tf

    def get_link_transforms(self):
        return self.tf


def test_one_update_writes_the_blended_points():
    pytest.importorskip("pxr")
    # two links: A at rest, B turned 90 degrees about z through its rest point
    rest = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    local = np.array([[0.5, 0, 0], [0.2, 0, 0]], np.float32)       # A's piece: a seam point and an inner one
    piece = sb._Piece(points_attr=FakeAttr(), extent_attr=FakeAttr(), own=0, own_rest=rest[0], local=local, rows=np.array([0]),
                      cols=np.array([0, 1]), col_rests=rest, weights=np.array([[0.5, 0.5]], np.float32))
    tf = np.array([[0, 0, 0, 0, 0, 0, 1], [1, 0, 0, *quat_xyzw([0, 0, 1], 90)]], np.float32)
    blend = sb.SkinBlend(env=None, view=FakeView(tf), pieces=[piece], links=2)
    blend.update()
    points = piece.points_attr.value
    # B carries the seam point (0.5, 0, 0) to (1, -0.5, 0); half of that and half of where A leaves it
    assert np.allclose(points[0], [0.75, -0.25, 0.0], atol=1e-6)
    assert np.array_equal(points[1], local[1])
    # the authored bounds follow the moved point
    assert np.array_equal(piece.extent_attr.value, np.stack([points.min(axis=0), points.max(axis=0)]))
    assert blend.updates == 1
    want = ss.place(local, rest[0], piece.rows, piece.weights,
                    [(np.eye(3), rest[0]), (sb.rotations_xyzw(tf[1:, 3:])[0], rest[1])], rest, (np.eye(3), rest[0]))
    assert np.array_equal(points, want)


# ---- from_env with stand-ins for the simulator's modules ---------------------------------------

class Prim:
    def __init__(self, name, local=None, world=None, attrs=None, children=None):
        self.name, self.local, self.world = name, local, world
        self.attrs, self.children = attrs or {}, children or {}

    def __bool__(self):
        return True

    def GetChild(self, name):
        return self.children.get(name)

    def GetAttribute(self, name):
        return self.attrs.get(name)

    def GetPath(self):
        return self.name


class StoredAttr(FakeAttr):
    def __init__(self, value):
        self.value = value

    def __bool__(self):
        return self.value is not None

    def HasAuthoredValue(self):
        return self.value is not None

    def Get(self):
        return self.value


def matrix(pos, scale=1.0):
    m = np.diag([scale, scale, scale, 1.0])
    m[3, :3] = pos
    return m


@pytest.fixture
def standins(monkeypatch):
    """omni.usd and pxr replaced by the few calls from_env and update make (a dict of what the stage holds)."""
    import types

    held = {}

    class Xformable:
        def __init__(self, prim):
            self.prim = prim

        def GetLocalTransformation(self):
            return self.prim.local

        def ComputeLocalToWorldTransform(self, _time):
            return self.prim.world if self.prim.world is not None else np.eye(4)

    class Mesh:
        def __init__(self, prim):
            self.prim = prim

        def GetPointsAttr(self):
            return self.prim.GetAttribute("points")

        def GetExtentAttr(self):
            return self.prim.GetAttribute("extent")

    omni = types.ModuleType("omni")
    usd = types.ModuleType("omni.usd")
    usd.get_context = lambda: types.SimpleNamespace(get_stage=lambda: types.SimpleNamespace(
        GetPrimAtPath=lambda path: held["root"]))
    omni.usd = usd
    pxr = types.ModuleType("pxr")
    pxr.UsdGeom = types.SimpleNamespace(Xformable=Xformable, Mesh=Mesh)
    pxr.Usd = types.SimpleNamespace(TimeCode=types.SimpleNamespace(Default=lambda: None))
    pxr.Vt = types.SimpleNamespace(Vec3fArray=types.SimpleNamespace(FromNumpy=lambda a: np.array(a)))
    for name, module in (("omni", omni), ("omni.usd", usd), ("pxr", pxr)):
        monkeypatch.setitem(sys.modules, name, module)

    def env(root, names, tf=None):
        held["root"] = root
        dummy = types.SimpleNamespace(body_names=names,
                                      root_physx_view=types.SimpleNamespace(get_link_transforms=lambda: tf))

        class Scene:
            cfg = types.SimpleNamespace(dummy=types.SimpleNamespace(prim_path="{ENV_REGEX_NS}/Dummy"))
            env_regex_ns = "/World/envs/env_.*"
            env_prim_paths = ["/World/envs/env_0"]

            def __getitem__(self, key):
                return dummy

        return types.SimpleNamespace(num_envs=1, scene=Scene())

    return env


def seamed_piece():
    return Prim("A/skin", attrs={
        "points": StoredAttr(np.array([[0.5, 0, 0], [0.1, 0, 0]], np.float32)), "extent": StoredAttr(None),
        ss.ATTR_POINTS: StoredAttr([0]), ss.ATTR_BODIES: StoredAttr(["A", "B"]), ss.ATTR_WEIGHTS: StoredAttr([0.5, 0.5])})


def test_a_dummy_without_the_blend_is_left_alone_whatever_its_bodies(standins):
    # the unskinned dummy's bodies are its shapes, scaled boxes: nothing of them is read
    root = Prim("Dummy", children={"BR_04": Prim("BR_04", local=matrix([0, 0, 1.43], scale=0.2))})
    assert sb.SkinBlend.from_env(standins(root, ["BR_04"])) is None


def test_a_scaled_dummy_or_a_turned_body_is_refused(standins):
    def root(world_scale=1.0, b_scale=1.0):
        return Prim("Dummy", world=matrix([0, 0, 0], world_scale), children={
            "A": Prim("A", local=matrix([0, 0, 0]), children={"skin": seamed_piece()}),
            "B": Prim("B", local=matrix([1, 0, 0], b_scale))})

    assert sb.SkinBlend.from_env(standins(root(), ["A", "B"])).points == 1
    with pytest.raises(ValueError, match="늘어나"):
        sb.SkinBlend.from_env(standins(root(world_scale=2.0), ["A", "B"]))
    with pytest.raises(ValueError, match="쉬는 자세"):
        sb.SkinBlend.from_env(standins(root(b_scale=1.5), ["A", "B"]))


def test_from_env_reads_the_rest_frames_the_points_were_built_in(standins):
    piece = seamed_piece()
    root = Prim("Dummy", children={"A": Prim("A", local=matrix([0, 0, 0]), children={"skin": piece}),
                                   "B": Prim("B", local=matrix([1, 0, 0]))})
    tf = np.array([[0, 0, 0, 0, 0, 0, 1], [1, 0, 0, *quat_xyzw([0, 0, 1], 90)]], np.float32)
    blend = sb.SkinBlend.from_env(standins(root, ["A", "B"], tf))
    blend.update()
    assert np.allclose(piece.attrs["points"].value[0], [0.75, -0.25, 0.0], atol=1e-6)
    assert np.array_equal(piece.attrs["points"].value[1], np.float32([0.1, 0, 0]))
    points = piece.attrs["points"].value
    assert np.array_equal(piece.attrs["extent"].value, np.stack([points.min(axis=0), points.max(axis=0)]))
