"""The box the robot carries in S1e, S1f and S1g: a rigid body of the row's size and mass, held on the carry link
(meta/scenarios.json ``carry`` -- generation/carry.py) by a fixed joint, put back on the link at every reset and let
go by turning the joint off; and the arm hold that keeps the arms at the pose the carry names.

How it is held (measured in Isaac Sim 6.0.1 / Isaac Lab 3.0.0 on 2026-10-20, the published H1 walking policy with a
2-3 kg box walking at 0.8 m/s and turning at 0.5 rad/s: the box within 0.05 mm and 0.006 deg of its carry pose, no
force between it and the robot, the walk within 0.3 % of the same mass lumped into the link, no fall):

* The box USD (generation/scene/carried_box.py) is spawned under the env and resized on the live stage
  (apply_box_size). Its rigid-body, mass and contact-report APIs are applied on the live stage before the simulation
  starts -- never written into the built file, which ships without a ``physics:mass``.
* Collisions between the box and every rigid link of the robot are filtered (FilteredPairsAPI), so only the joint holds
  it; unfiltered, the arms and the torso pressed on it with up to 0.7 kN.
* A FixedJoint, kept out of the articulation, holds it at the carry pose of the link. It is authored where the spawned
  stage holds the link -- which is not where the robot stands after its reset (the asset's links sit above the spawned
  root) -- and at every reset the box is put on the link's pose computed from the root pose the reset just wrote
  (CarryRig.reset). Authored at the post-reset pose instead, the joint pulled the box about 1.1 m up during start-up
  and back down after the reset, and tipped the robot further.
* Released by ``physics:jointEnabled`` False on the live stage: the box falls from the next env step on.

The authoring is an event term of mode "prestartup" (Isaac Lab allows one only with ``scene.replicate_physics``
False) for a single env: the contact filters each name one prim. Isaac is imported inside the functions, so the
arithmetic here (arm_hold_columns, pose_error, box_usd_path) loads and is tested without it.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml

from generation import carry as carry_canon
from generation import conditions
from generation.assets import load_assets

REPO = Path(__file__).resolve().parents[3]
BOX_ENTITY = "box"                    # the scene entity of the box, and its prim under the env
BOX_PRIM = "Box"
BOX_CONTACT = "box_contact"           # its contact sensor
JOINT_PRIM = "BoxHold"                # the fixed joint, under the env
RIG_ATTR = "carry_rig"                # where the environment keeps its CarryRig
# The hold pose has to be the robot's default joint position (the policy's zero action): a hold that differs by more
# than this is refused, since the walking policy was never trained with the arms elsewhere.
HOLD_DEFAULT_TOL_RAD = 1e-6
# The robot's root link -- what the articulation's root pose is the pose of -- from the joint map's provenance.
ROOT_LINK = yaml.safe_load((REPO / "meta/joint_map/unitree_h1.yaml").read_text(encoding="utf-8")
                           )["provenance"]["articulation_root"].rsplit("/", 1)[-1]


def box_usd_path(carry: carry_canon.Carry, repo_root: Path = REPO) -> Path:
    """The built USD of the box ``carry`` names: its meta/assets.json scene entry's ``usd_path``."""
    entry = (load_assets(repo_root).get("scene") or {}).get(carry.asset)
    if not isinstance(entry, Mapping) or not isinstance(entry.get("usd_path"), str):
        raise ValueError(f"meta/assets.json scene 에 들고 다니는 상자 {carry.asset!r} 의 usd_path 가 없습니다")
    return Path(repo_root) / entry["usd_path"]


def arm_hold_columns(terms: Sequence[tuple[Sequence[str], Sequence[float], Sequence[float]]],
                     hold: Mapping[str, float], default: Mapping[str, float],
                     tol: float = HOLD_DEFAULT_TOL_RAD) -> tuple[list[int], list[float]]:
    """(columns, values) of the policy's action that hold the joints of ``hold`` at their angle.

    ``terms`` are the action manager's terms in order, each (joint names, offset, scale) per column -- a joint
    action's command is offset + scale * action, so the column holds (angle - offset) / scale. The columns are found
    by the joints' names, never by position, and come back in column order. ValueError when a joint of ``hold`` is in
    no term, or its angle is not the robot's ``default`` joint position (within ``tol``)."""
    off = sorted(name for name, angle in hold.items()
                 if name not in default or not abs(float(angle) - float(default[name])) <= tol)
    if off:
        raise ValueError(f"팔을 잡는 자세가 로봇의 기본 관절 자세와 다릅니다: "
                         f"{ {n: (hold[n], default.get(n)) for n in off} } — 보행 정책은 기본 자세의 팔로만 배웠습니다")
    found: dict[str, tuple[int, float]] = {}
    base = 0
    for names, offset, scale in terms:
        for k, name in enumerate(names):
            if name in hold:
                found[name] = (base + k, (float(hold[name]) - float(offset[k])) / float(scale[k]))
        base += len(names)
    missing = sorted(set(hold) - set(found))
    if missing:
        raise ValueError(f"팔을 잡는 관절 {missing} 을 움직이는 행동 항이 없습니다")
    ordered = sorted(found.values())
    return [c for c, _ in ordered], [v for _, v in ordered]


def pose_error(pos_a, quat_a_xyzw, pos_b, quat_b_xyzw) -> tuple[float, float]:
    """(distance m, angle deg) between two poses; a quaternion and its negative are the same turn."""
    d = float(np.linalg.norm(np.asarray(pos_b, float) - np.asarray(pos_a, float)))
    qa = np.asarray(quat_a_xyzw, float)
    qb = np.asarray(quat_b_xyzw, float)
    dot = abs(float(qa @ qb) / (float(np.linalg.norm(qa)) * float(np.linalg.norm(qb))))
    return d, math.degrees(2.0 * math.acos(min(1.0, dot)))


def _t(x):
    """A warp-backed Isaac Lab buffer as a torch tensor."""
    return x.torch if hasattr(x, "torch") else x


def _world_pose(stage, path: str) -> tuple[np.ndarray, np.ndarray]:
    """(pos (3,), quat xyzw (4,)) of a prim from the stage's xforms."""
    from pxr import UsdGeom

    m = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    t, r = m.ExtractTranslation(), m.ExtractRotationQuat()
    im = r.GetImaginary()
    q = np.array([im[0], im[1], im[2], r.GetReal()], float)
    return np.array([t[0], t[1], t[2]], float), q / np.linalg.norm(q)


def _relative_pose(stage, path: str, frame: str) -> tuple[list[float], list[float]]:
    """(pos, quat xyzw) of the prim at ``path`` in the frame of the prim at ``frame``, as the stage holds them."""
    from pxr import UsdGeom

    cache = UsdGeom.XformCache()
    rel = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)) * cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(frame)).GetInverse()                       # USD row vectors: world = local * parent
    t, r = rel.ExtractTranslation(), rel.ExtractRotationQuat()
    im = r.GetImaginary()
    q = np.array([im[0], im[1], im[2], r.GetReal()], float)
    return [float(t[0]), float(t[1]), float(t[2])], (q / np.linalg.norm(q)).tolist()


def _env_prim(env, entity: str, env_path: str) -> str:
    """The prim of a scene entity in the env at ``env_path``: its cfg's prim path, whose env part the scene has made
    the regex of every env (or still holds as the ``{ENV_REGEX_NS}`` token), put on that env."""
    path = env.scene.cfg.__dict__[entity].prim_path
    for ns in ("{ENV_REGEX_NS}", env.scene.env_regex_ns):
        if path.startswith(ns + "/"):
            return env_path + path[len(ns):]
    raise ValueError(f"{entity} 의 프림 경로 {path!r} 가 환경 아래에 있지 않습니다")


def _set_local_pose(prim, pos, quat_xyzw) -> None:
    from pxr import Gf, UsdGeom

    ops = {op.GetOpType(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
    x, y, z, w = (float(v) for v in quat_xyzw)
    if UsdGeom.XformOp.TypeTranslate in ops and UsdGeom.XformOp.TypeOrient in ops:
        ops[UsdGeom.XformOp.TypeTranslate].Set(Gf.Vec3d(*map(float, pos)))
        orient = ops[UsdGeom.XformOp.TypeOrient]
        orient.Set(Gf.Quatf(w, x, y, z) if orient.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
                   else Gf.Quatd(w, x, y, z))
    else:
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))


class CarryRig:
    """The held box of the environment's single env. Made on the live stage before the simulation starts (the
    "prestartup" term author_carried_box); reset() after the robot's own reset terms puts the box on the link."""

    def __init__(self, env, carry: carry_canon.Carry, size_m: Sequence[float], mass_kg: float, *,
                 robot: str = "robot", box: str = BOX_ENTITY):
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdPhysics

        from generation.scene.carried_box import apply_box_size

        if env.num_envs != 1:
            raise ValueError(f"들고 다니는 상자는 환경 하나에서만 짓습니다(접촉 필터가 프림 하나씩을 가리킴): num_envs {env.num_envs}")
        if not (isinstance(mass_kg, (int, float)) and math.isfinite(float(mass_kg)) and float(mass_kg) > 0):
            raise ValueError(f"상자 질량이 양의 유한한 수가 아닙니다: {mass_kg!r}")
        self.carry = carry
        self.size = tuple(float(v) for v in size_m)
        self.mass = float(mass_kg)
        self.released = False
        self._env, self._robot_name, self._box_name = env, robot, box
        stage = env.scene.stage
        env_path = env.scene.env_prim_paths[0]
        self._robot_path = _env_prim(env, robot, env_path)
        self._box_path = _env_prim(env, box, env_path)
        self._joint_path = f"{env_path}/{JOINT_PRIM}"
        prim = stage.GetPrimAtPath(self._box_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"{self._box_path} 가 없습니다 — 상자를 씬에 띄운 뒤에 지어야 합니다")
        link_path = f"{self._robot_path}/{carry.link}"
        if not stage.GetPrimAtPath(link_path).IsValid():
            raise ValueError(f"상자를 붙일 링크 {link_path} 가 없습니다")
        apply_box_size(stage, self._box_path, self.size)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(self.mass)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
        # At attach PhysX takes the link where the spawned stage holds it: the box goes there, so the joint holds it
        # without an error at the start; reset() moves it with the robot.
        env_pos, _ = _world_pose(stage, env_path)
        centre, quat = carry_canon.box_pose(*_world_pose(stage, link_path), carry, self.size)
        _set_local_pose(prim, centre - env_pos, quat)
        bodies = [p.GetPath() for p in Usd.PrimRange(stage.GetPrimAtPath(self._robot_path))
                  if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if not bodies:
            raise ValueError(f"{self._robot_path} 아래에 강체 링크가 없습니다")
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets([Sdf.Path(b) for b in bodies])
        self.filtered_links = len(bodies)
        joint = UsdPhysics.FixedJoint.Define(stage, self._joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(link_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(self._box_path)])
        half = math.radians(carry.yaw_deg) / 2.0
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, self._centre_in_link())))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(math.cos(half), 0.0, 0.0, math.sin(half)))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateExcludeFromArticulationAttr().Set(True)
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateJointEnabledAttr().Set(True)
        # The link in the root link's frame as the asset authors it -- the robot's reset writes the root, and the
        # joint from the root to the carry link stands at the default it is reset to.
        self._link_in_root = _relative_pose(stage, link_path, f"{self._robot_path}/{ROOT_LINK}")

    def _centre_in_link(self) -> np.ndarray:
        return np.asarray(self.carry.bottom_m, float) + np.array([0.0, 0.0, self.size[2] / 2.0])

    def _set_joint(self, enabled: bool) -> None:
        self._env.scene.stage.GetPrimAtPath(self._joint_path).GetAttribute("physics:jointEnabled").Set(bool(enabled))

    def reset(self, env_ids=None) -> None:
        """Put the box at the carry pose of the link, at rest, for the root pose the robot's reset just wrote (the
        link's pose is not read: the links are not where the root was written until the next physics step). A box
        that was let go is held again."""
        import torch

        import isaaclab.utils.math as mu

        robot, box = self._env.scene[self._robot_name], self._env.scene[self._box_name]
        rp = _t(robot.data.root_link_pos_w)
        dev = rp.device
        ids = (torch.arange(self._env.num_envs, device=dev) if env_ids is None
               else torch.as_tensor(env_ids, dtype=torch.long, device=dev))
        rp, rq = rp[ids], _t(robot.data.root_link_quat_w)[ids]
        n = len(ids)
        lp = torch.tensor(self._link_in_root[0], dtype=rp.dtype, device=dev).expand(n, 3)
        lq = torch.tensor(self._link_in_root[1], dtype=rp.dtype, device=dev).expand(n, 4)
        tp, tq = rp + mu.quat_apply(rq, lp), mu.quat_mul(rq, lq)
        centre = torch.tensor(self._centre_in_link(), dtype=rp.dtype, device=dev).expand(n, 3)
        half = math.radians(self.carry.yaw_deg) / 2.0
        q_yaw = torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)], dtype=rp.dtype, device=dev).expand(n, 4)
        pose = torch.cat([tp + mu.quat_apply(tq, centre), mu.quat_mul(tq, q_yaw)], dim=-1)
        box.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        box.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, dtype=rp.dtype, device=dev), env_ids=ids)
        if self.released:
            self._set_joint(True)
            self.released = False

    def release(self) -> None:
        """Let the box go: the joint is turned off on the live stage and the box falls from the next env step on."""
        if not self.released:
            self._set_joint(False)
            self.released = True

    def box_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(centre (3,), quat xyzw (4,)) of the box, world frame."""
        box = self._env.scene[self._box_name]
        return (_t(box.data.root_pos_w)[0].detach().cpu().numpy().astype(float),
                _t(box.data.root_quat_w)[0].detach().cpu().numpy().astype(float))

    def box_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """(linear (3,), angular (3,)) velocity of the box, world frame."""
        box = self._env.scene[self._box_name]
        return (_t(box.data.root_lin_vel_w)[0].detach().cpu().numpy().astype(float),
                _t(box.data.root_ang_vel_w)[0].detach().cpu().numpy().astype(float))

    def carry_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the carry puts the box for the link as it is now (carry.box_pose), world frame."""
        robot = self._env.scene[self._robot_name]
        k = list(robot.body_names).index(self.carry.link)
        return carry_canon.box_pose(_t(robot.data.body_pos_w)[0, k].detach().cpu().numpy().astype(float),
                                    _t(robot.data.body_quat_w)[0, k].detach().cpu().numpy().astype(float),
                                    self.carry, self.size)

    def carry_error(self) -> tuple[float, float]:
        """(m, deg) between the box and its carry pose (pose_error)."""
        return pose_error(*self.carry_pose(), *self.box_pose())


class ArmHold:
    """Holds the arm joints the carry names at its hold pose: before every env step the policy's action columns of
    those joints are set to (hold - offset) / scale (arm_hold_columns over the action manager's terms -- the next
    observation's last action is the held one, as in the stop test). Refuses a hold that is not the robot's default
    joint position."""

    def __init__(self, env, carry: carry_canon.Carry, *, robot: str = "robot"):
        import torch

        asset = env.scene[robot]
        names = list(asset.joint_names)
        self.hold = carry_canon.arm_hold_joints(carry, names)
        default_row = _t(asset.data.default_joint_pos)[0].detach().cpu().tolist()
        default = dict(zip(names, default_row))
        manager = env.action_manager
        terms = []
        for term_name in manager.active_terms:
            term = manager.get_term(term_name)
            term_names = list(getattr(term, "_joint_names", [None] * term.action_dim))
            terms.append((term_names, _row(getattr(term, "_offset", 0.0), term.action_dim),
                          _row(getattr(term, "_scale", 1.0), term.action_dim)))
        cols, values = arm_hold_columns(terms, self.hold, default)
        dev = env.device
        self._asset = asset
        self._cols = torch.tensor(cols, dtype=torch.long, device=dev)
        self._values = torch.tensor(values, dtype=torch.float32, device=dev)
        self._joint_ids = torch.tensor([names.index(n) for n in self.hold], dtype=torch.long, device=dev)
        self._target = torch.tensor([self.hold[n] for n in self.hold], dtype=torch.float32, device=dev)

    def apply(self, actions) -> None:
        """Overwrite the held columns of ``actions`` (num_envs, action dim) in place."""
        actions[:, self._cols] = self._values.to(actions.dtype)

    def error(self) -> float:
        """The largest |joint - hold| over the held joints now (rad)."""
        q = _t(self._asset.data.joint_pos)[0, self._joint_ids]
        return float((q - self._target.to(q.dtype)).abs().max())


def _row(value, dim: int) -> list[float]:
    """An action term's offset or scale for the first env, as ``dim`` floats (a float, or a (num_envs, dim) tensor)."""
    if isinstance(value, (int, float)):
        return [float(value)] * dim
    return [float(v) for v in _t(value)[0].detach().cpu().tolist()]


def author_carried_box(env, env_ids, scenario_id: str, size_m: tuple, mass_kg: float) -> None:
    """The "prestartup" event term: the scenario's carry block, a CarryRig for it kept on the environment."""
    held = carry_canon.load_carry(conditions.scenario_entry(scenario_id))
    if held is None:
        raise ValueError(f"{scenario_id} 에 carry 블록이 없습니다 — 상자를 들지 않는 시나리오입니다")
    setattr(env, RIG_ATTR, CarryRig(env, held, size_m, mass_kg))


def place_carried_box(env, env_ids) -> None:
    """The "reset" event term, after the robot's reset terms: the box back on the link (CarryRig.reset)."""
    getattr(env, RIG_ATTR).reset(env_ids)


def spawn_carried_box(cfg, scenario_id: str, size_m: Sequence[float], mass_kg: float) -> None:
    """Add the box of ``size_m`` (width, depth, height m) and ``mass_kg`` that ``scenario_id``'s carry block holds to
    an environment config: the scene entity and its contact sensor, the authoring term and the reset term (defined
    last, so after the robot's reset terms -- terms run in definition order). Turns ``replicate_physics`` off."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab_physx.sensors import ContactSensorCfg

    held = carry_canon.load_carry(conditions.scenario_entry(scenario_id))
    if held is None:
        raise ValueError(f"{scenario_id} 에 carry 블록이 없습니다 — 상자를 들지 않는 시나리오입니다")
    size = tuple(float(v) for v in size_m)
    carry_canon.box_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), held, size)      # refuses a size that is not one
    cfg.scene.replicate_physics = False
    prim = "{ENV_REGEX_NS}/" + BOX_PRIM
    setattr(cfg.scene, BOX_ENTITY, RigidObjectCfg(prim_path=prim,
                                                  spawn=sim_utils.UsdFileCfg(usd_path=str(box_usd_path(held)))))
    setattr(cfg.scene, BOX_CONTACT, ContactSensorCfg(prim_path=prim, history_length=1))
    cfg.events.carried_box = EventTerm(func=author_carried_box, mode="prestartup",
                                       params={"scenario_id": scenario_id, "size_m": size, "mass_kg": float(mass_kg)})
    cfg.events.carried_box_place = EventTerm(func=place_carried_box, mode="reset", params={})
