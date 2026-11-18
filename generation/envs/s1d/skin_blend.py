"""The jointed dummy's skin, blended at its joints before every render.

The skinned jointed dummy hangs one skin piece under each body, and a piece's edge parts from its neighbour's when
the joint between them bends. build_dummy.py --seams puts on each piece the weights its points near a joint blend by
(generation/dummy/skin_seams.py); before every render this moves those points to the weighted mean of where their
bodies carry them now, so a bent knee or hip shows cloth instead of the gap between two pieces. Only the picture
changes: the skin has no physics, and the poses are read, never written. A dummy without the weights (v1, the bare
v2, the rigid v1 skin) is left as it is built. The face-corner normals stay as built: rewritten at run time they
made the renderer draw the whole piece see-through, while the points alone render solid.

The poses come from the articulation's physics view (get_link_transforms), current right after a physics step --
the env renders inside its step, before the scene's data buffers are refreshed. Renders go through env.sim.render
(the env's own step, the camera rig, the video writer), so attach() wraps that one method of the env's simulation
context. Torch-free at import; the arithmetic is skin_seams.place, which CI tests.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from generation.dummy import skin_seams

ENTITY = "dummy"
SKIN_CHILD = "skin"      # build_dummy.SKIN_CHILD: the visual child a body carries


@dataclass
class _Piece:
    points_attr: object          # the skin mesh's points attribute
    extent_attr: object          # its authored bounds, kept round the moved points
    own: int                     # the body's index among the articulation's links
    own_rest: np.ndarray         # its rest translation in the dummy's frame
    local: np.ndarray            # the points as built (float32)
    rows: np.ndarray             # the points that blend
    cols: np.ndarray             # the link index of each weight column
    col_rests: list              # each column body's rest translation
    weights: np.ndarray          # len(rows) x len(cols)


def _np(value) -> np.ndarray:
    t = getattr(value, "torch", value)
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return t.numpy() if hasattr(t, "numpy") else np.asarray(t)


def rotations_xyzw(q: np.ndarray) -> np.ndarray:
    """(n, 4) unit quaternions x y z w (the physics view's order) -> (n, 3, 3) rotation matrices."""
    q = np.asarray(q, np.float64)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def _env_prim_path(env, entity: str) -> str:
    """The prim of a scene entity in the first env: its cfg's prim path with the env part put on that env."""
    path = env.scene.cfg.__dict__[entity].prim_path
    for ns in ("{ENV_REGEX_NS}", env.scene.env_regex_ns):
        if path.startswith(ns + "/"):
            return env.scene.env_prim_paths[0] + path[len(ns):]
    raise ValueError(f"{entity} 의 프림 경로 {path!r} 가 환경 아래에 있지 않습니다")


class SkinBlend:
    """The dummy's blended skin pieces in an env; update() places them for the bodies' poses now."""

    def __init__(self, env, view, pieces: list[_Piece], links: int):
        self._env, self._view, self._pieces, self._links = env, view, pieces, links
        self.updates = 0

    @property
    def pieces(self) -> int:
        return len(self._pieces)

    @property
    def points(self) -> int:
        return int(sum(len(p.rows) for p in self._pieces))

    @classmethod
    def from_env(cls, env) -> "SkinBlend | None":
        """The blend of ``env``'s dummy, or None when no skin piece carries one (a v1 dummy's bodies are the shapes
        themselves, scaled -- nothing of them is read then). Refuses more than one env -- the pieces are read from
        the first env's prims and the poses of its links -- a dummy whose prim is scaled in the world (the physics
        view's positions would not be in the skin's units), and a body the blend names that does not rest unrotated
        at a translation, the frame the skin was built in."""
        import omni.usd
        from pxr import Usd, UsdGeom

        dummy = env.scene[ENTITY]
        names = list(dummy.body_names)
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(_env_prim_path(env, ENTITY))
        if not root:
            raise ValueError(f"인형 프림이 없습니다: {_env_prim_path(env, ENTITY)}")
        carrying = []
        for name in names:
            body = root.GetChild(name)
            mesh = body.GetChild(SKIN_CHILD) if body else None
            attrs = [mesh.GetAttribute(a) if mesh else None
                     for a in (skin_seams.ATTR_POINTS, skin_seams.ATTR_BODIES, skin_seams.ATTR_WEIGHTS)]
            if all(a and a.HasAuthoredValue() for a in attrs):
                carrying.append((name, mesh, attrs))
        if not carrying:
            return None
        if env.num_envs != 1:
            raise ValueError(f"피부 이음새는 환경 하나에서만 섞습니다: num_envs {env.num_envs}")
        world = np.array(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), np.float64)
        if not np.allclose(world[:3, :3] @ world[:3, :3].T, np.eye(3), atol=1e-6):
            raise ValueError(f"{root.GetPath()}: 인형 프림이 월드에서 늘어나 있습니다 — 물리 뷰의 자리와 피부의 좌표가 맞지 않습니다")
        index = {n: k for k, n in enumerate(names)}
        rest = {}

        def rest_of(name):
            if name not in rest:
                body = root.GetChild(name)
                # the bodies rest unrotated, at a translation in the dummy's frame -- what the skin was built against
                m = np.array(UsdGeom.Xformable(body).GetLocalTransformation(), np.float64)
                if not np.allclose(m[:3, :3], np.eye(3), atol=1e-6):
                    raise ValueError(f"{body.GetPath()}: 쉬는 자세가 돌았거나 늘어나 있습니다 — 피부를 지은 기준과 다릅니다")
                rest[name] = m[3, :3].copy()
            return rest[name]

        pieces = []
        for name, mesh, attrs in carrying:
            rows = np.asarray(attrs[0].Get(), np.int64)
            cols = [str(c) for c in attrs[1].Get()]
            unknown = sorted(set(cols) - set(index))
            if unknown:
                raise ValueError(f"{mesh.GetPath()}: 섞는 비율의 열이 관절체의 링크가 아닙니다 {unknown}")
            weights = np.asarray(attrs[2].Get(), np.float32).reshape(len(rows), len(cols))
            geom = UsdGeom.Mesh(mesh)
            local = np.asarray(geom.GetPointsAttr().Get(), np.float32)
            if len(rows) and (rows.min() < 0 or rows.max() >= len(local)):
                raise ValueError(f"{mesh.GetPath()}: 섞는 정점의 번호가 점의 범위 밖입니다")
            pieces.append(_Piece(
                points_attr=geom.GetPointsAttr(), extent_attr=geom.GetExtentAttr(), own=index[name], own_rest=rest_of(name), local=local, rows=rows,
                cols=np.array([index[c] for c in cols], np.int64), col_rests=[rest_of(c) for c in cols],
                weights=weights))
        return cls(env, dummy.root_physx_view, pieces, len(names))

    def update(self) -> None:
        """Place every blended piece for the links' poses now (the first env)."""
        from pxr import Vt

        tf = _np(self._view.get_link_transforms()).reshape(-1, self._links, 7)[0]
        x = tf[:, :3].astype(np.float64)
        R = rotations_xyzw(tf[:, 3:7])
        for p in self._pieces:
            own = (R[p.own], x[p.own])
            poses = [(R[c], x[c]) for c in p.cols]
            points = skin_seams.place(p.local, p.own_rest, p.rows, p.weights, poses, p.col_rests, own)
            p.points_attr.Set(Vt.Vec3fArray.FromNumpy(points))
            p.extent_attr.Set(Vt.Vec3fArray.FromNumpy(np.stack([points.min(axis=0), points.max(axis=0)])))
        self.updates += 1

    def attach(self) -> None:
        """Blend before every render of the env: wraps env.sim.render, once."""
        sim = self._env.sim
        if getattr(sim.render, "_skin_blend", None) is not None:
            raise RuntimeError("피부 이음새 섞기가 이미 렌더에 걸려 있습니다")
        original = sim.render

        @functools.wraps(original)
        def render(*args, **kwargs):
            self.update()
            return original(*args, **kwargs)

        render._skin_blend = self
        sim.render = render
