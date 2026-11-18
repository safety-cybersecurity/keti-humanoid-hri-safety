"""Collect one episode into a neutral file pair.

The generation env has no lerobot and the conversion venv has no Isaac Sim,
so the two cannot share a format. npz + json is what both can read.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import warp as wp
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.utils.math as math_utils

from generation import conditions
from generation.envs.hri.record_logic import (  # noqa: F401 -- the HRI scenarios' pure half, re-exported
    CONTACT_ONSET_GAP_REASON,
    CONTACT_TYPE_GAP_REASON,
    DUMMY_ROW_KEYS,
    HRI_EVENT_NAMES,
    attribute_contacts,
    camera_in_box_counts,
    condition_record,
    count_cameras_in_box,
    dummy_degenerate_reason,
    hri_ssm_params,
    hri_step_columns,
    hri_zero_step_keys,
    rack_contact_edge,
    rack_residual,
    records_hri,
    region_contacts,
    step_or_none,
    task_stages_record,
    touched,
    zero_step_arrays,
)

from . import substep_contact
from .media_writer import split_media
from .mdp.entry_geometry import EntrySpeedWindow
from .mdp.safety import (
    ANTHROPOMETRY_PATH,
    GROUND_REGION_CODE,
    aggregate_contact_by_region,
    contact_onset_gap,
    contact_relative_speed,
    dummy_body_masks,
    dummy_lever_cap,
    non_foot_mask,
    reduce_region_peaks,
    center_of_mass,
    classify_contact_type,
    stability_margin,
    zero_moment_point,
)

# Observed: a robot that falls at the very end of one episode can land the
# *next* episode's reset already in (near-)collapse, so the fall terminations
# fire again within the episode's first step or two -- before any SSM/entry
# interaction has had time to unfold. An episode that short carries no usable
# data; flag it rather than let it silently dilute episode-length and outcome
# statistics downstream. 5 gives a small margin over the observed 1-step
# failure while staying far below what any genuine interaction needs (entry
# alone takes dozens of steps at minimum to reach the robot's lane, even at
# the catalogue's fastest human speed).
MIN_VALID_STEPS = 5

# The three S1a-relevant safety.event values (catalogue §4.6). The schema's
# full enum also has pfl_limit_exceeded, fall_detected, etc. -- deliberately
# not produced here. pfl_limit_exceeded specifically would require comparing
# contact_force against Table A.2, and contact magnitude was measured not to
# be physically trustworthy yet (the dummy is airborne at the moment of contact
# in 4 of 5 measured steps, hanging off a hold servo at ~4x body weight; the
# hold/contact ratio swings 12x step to step). See mdp/commands.py's
# _advance_dummy docstring for the measurement.
EVENT_NAMES = ("ssm_violation", "protective_stop_triggered", "contact_onset")

# Per-prim contact below this magnitude counts as "no contact" for
# contact_onset edge-detection. A noise guard, not a Table A.2 comparison
# (out of scope -- see EVENT_NAMES above): PhysX can report a vanishingly
# small nonzero residual between bodies that are not meaningfully touching.
_CONTACT_EPS_N = 1e-6
# Public alias: run_s1a.py hands the same value to the substep accumulator so the
# recorder and the accumulator cannot drift apart on what counts as contact.
CONTACT_EPS_N = _CONTACT_EPS_N

# Whether a foot counts as load-bearing for the support polygon (below).
# Measured, not guessed -- a 400-step rollout
# (seed 0) of robot_contact's own per-ankle force is cleanly bimodal: 395 of
# 800 (both feet x 400 steps) readings are exactly 0.0 (PhysX reports a
# clean zero when a foot is not in contact, no meaningful sensor noise to
# float above), and every other reading is either a single-step heel-strike/
# toe-off transition (4 readings, 14-36 N) or already >=50 N and climbing
# into the hundreds (stance, mean ~260/234 N left/right, up to 2860 N peak).
# 10 N sits in the empty gap between the "exactly 0" swing cluster and the
# transition readings -- comfortably clear of both, and still counts a
# just-touching-down foot as bearing *some* weight rather than none.
_STANCE_FORCE_THRESHOLD_N = 10.0

# Height at or below which a loaded region counts as pressed against the
# floor rather than free to recede -- the operational half of the schema's
# transient/quasistatic distinction (§8.1). Measured the same way
# _STANCE_FORCE_THRESHOLD_N was, by finding the empty gap between two
# clusters rather than by picking a plausible number.
#
# Four 1000-step episodes, per-region body-origin height, split by whether
# the dummy was still being held upright or had been released and settled:
#
#   standing      n=43800   minimum 0.2422 m
#   settled fallen n=2352   maximum 0.1640 m
#
# The two never overlap: the gap [0.1640, 0.2422] is 0.078 m wide and
# contains no sample of either kind. 0.20 m sits inside it with ~0.036 m of
# margin below and ~0.042 m above, so no standing region is ever read as
# pinned and every settled fallen one is.
#
# The transient of the fall itself passes through the gap, and that is
# correct: a body still falling can recede, so its contacts are transient.
_PIN_HEIGHT_M = 0.20

# 이 사이드카가 반드시 담아야 하는 키. **데이터셋 스키마의 필드 목록이 아닙니다** —
# 이것은 Isaac 이 뱉는 중간 산출물이고, 공개 스키마(§5)의 필드는 7단계의 LeRobot
# 변환이 만듭니다. 둘을 같은 것으로 읽으면 "필수 22개 중 11개뿐" 같은 잘못된 대조가
# 나옵니다. 여기 목록은 그 변환이 입력으로 기대하는 최소 계약입니다.
SIDECAR_REQUIRED_KEYS = (
    "episode_id", "scenario_id", "episode_seq", "run_start_seq", "seed", "steps", "degenerate",
    "sim_backend", "contact_region_names", "safety_event_names", "stance_foot_names",
    "randomization", "hold_release_step", "hold_contaminated_steps",
    "provisional", "ssm_params", "global_time_offset_s",
    "robot_model", "asset_paths", "asset_paths_resolved", "imu_link",
    "substep_contact_samples", "action_space", "termination",
)


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Isaac Lab reports (x, y, z, w); dataset schema §6 wants (w, x, y, z).

    Reordered here rather than at conversion time so the npz and the parquet
    agree on one convention -- a consumer reading the npz directly would
    otherwise be silently rotated.
    """
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=q.dtype)


def _pose_xyzw_to_wxyz(pose7: np.ndarray) -> np.ndarray:
    """(pos3, quat_xyzw) -> (pos3, quat_wxyz)."""
    return np.concatenate([pose7[:3], _quat_xyzw_to_wxyz(pose7[3:7])])

# BR-12 (the shins) is the dummy's only ground-contact region -- the same
# structural fact check_scene.py / check_fall_termination.py /
# check_dummy_hold.py all use (anthropometry.yaml: only regions low enough to
# reach z=0). Read through region_codes (asset-derived) rather than the
# prim-name set those scripts hard-code, per CONTRIBUTING.md 「관절 순서」. The jointed dummy stands on its
# feet instead -- bodies with no region code (safety.dummy_body_masks).
_GROUND_REGION_CODE = GROUND_REGION_CODE


# The joint command the schema publishes: what the robot was told to do, in the
# units the schema names. The action manager's own `action` is the policy's raw
# input, which for a normalised policy is not radians and not torque.
ACTION_SPACE = "position_rad"


def processed_action(env):
    """The joint targets this step, concatenated across action terms, in rad."""
    manager = env.action_manager
    parts = [manager._terms[name].processed_actions[0] for name in manager.active_terms]
    if len(parts) == 1:
        return parts[0].cpu().numpy()
    import torch

    return torch.cat(parts, dim=-1).cpu().numpy()


def read_region_codes(usd_path: str, body_names: list[str]) -> list[str]:
    """Map contact-sensor body order onto BR codes using the asset's own tags.

    The code lives on the prim as keti:bodyRegion. Deriving it from the prim
    name instead would break silently the first time a name changes -- the
    prim name (e.g. "BR_08_left") looks like the code but is not one 1:1
    (paired regions don't even share a name).
    """
    stage = Usd.Stage.Open(usd_path)
    by_name = {}
    for prim in stage.Traverse():
        attr = prim.GetAttribute("keti:bodyRegion")
        if attr and attr.Get():
            by_name[prim.GetName()] = attr.Get()
    missing = [n for n in body_names if n not in by_name]
    if missing:
        raise ValueError(f"keti:bodyRegion 이 없는 바디: {missing}")
    return [by_name[n] for n in body_names]


def ground_contact_body_names(usd_path: str, body_names: list[str]) -> set[str]:
    """Which of body_names carry the dummy's standing load: the ground-contact region (BR-12) of the rigid
    dummy, the feet (keti:groundContact, no region) of the jointed one -- safety.dummy_body_masks.

    Several check_*.py scripts need "is this body the shin/ground region" (a
    self-collision leak check, ground-reaction-force gating, foot-slide
    diagnostics) and used to hand-type {"BR_12_left", "BR_12_right"} for it --
    a prim-name literal, which CONTRIBUTING.md 「관절 순서」 says is not the
    canon (prim names are not guaranteed 1:1 with BR codes in general; that
    BR-12 happens to be named "_left"/"_right" on this asset build is not
    something a caller should assume). Reads the asset's own tags instead
    (jointed.read_body_codes -- a body with neither tag is refused), so a rename
    or a future asset revision with a different ground-body prim layout is still
    caught correctly rather than silently stopping.
    """
    from generation.dummy import jointed

    _, ground = dummy_body_masks(jointed.read_body_codes(usd_path, body_names))
    return {name for name, g in zip(body_names, ground) if g}


def foot_link_names(robot) -> list[str]:
    """Which of robot's own body links are its feet, structurally.

    H1RoughEnvCfg's own feet_air_time/feet_slide reward terms already use
    ".*ankle_link" as *the* definition of "this is a foot" for this asset,
    and check_fall_termination.py's robot_ground_bodies() uses the same
    precedent for the identical reason -- a biped alternates load between
    feet, so this has to be a structural fact about the asset, not something
    re-derived from a short window of readings.

    A single module-level definition, not duplicated between run_s1a.py
    (which needs the names to look up footprint geometry before the episode
    loop starts) and EpisodeRecorder (which needs the same names' indices
    every step) -- the point of CONTRIBUTING.md 「관절 순서」 about a
    hand-typed copy drifting out of sync applies to this list exactly as
    much as to a BR code list.
    """
    names = [n for n in robot.body_names if n.endswith("ankle_link")]
    if not names:
        raise ValueError(f"발 링크를 찾지 못했습니다: {robot.body_names}")
    return names


def read_foot_footprint(usd_path: str, link_names: Sequence[str]) -> dict[str, np.ndarray]:
    """Read each foot link's ground-contact footprint from its own collision mesh.

    A foot's single link origin is not enough of a "support point" to hand
    stability_margin. With only 2 points, one per foot,
    mdp.safety.stability_margin's own _convex_hull always returns fewer than
    3 points, and that branch of the function is defined to return a
    *negated* distance-to-segment -- i.e. it cannot read positive, ever,
    regardless of how the robot is actually standing (see that function's
    docstring). All 999 and all 427 steps measured that way came back
    negative for exactly this structural reason, not because ordinary gait
    sways the CoM outside a thin line segment.

    So stability_margin gets each foot's actual ground-contact
    rectangle (4 corners) instead of its single origin point -- 8 points
    total, enough for a real polygon. The corners come from the H1 asset's
    own collision mesh, not a typed-in foot length/width: the "measure from
    the asset, don't hand-type" of CONTRIBUTING.md 「관절 순서」 applies to
    geometry here just as much as to joint order, and the actual mesh (not
    an eyeballed footprint) is what PhysX's contact solver reasons about.
    Each ankle link's collision mesh is a USD-instanced Mesh (~13k points on
    this H1 build, confirmed by direct inspection) reached via
    Usd.TraverseInstanceProxies() -- a plain stage.GetPrimAtPath() on the
    same string path does not resolve through the instance boundary and
    silently returns an unbounded/empty box from UsdGeom.BBoxCache, which is
    why this reads the mesh's own `points` attribute directly and transforms
    it by hand rather than asking BBoxCache for a world bound.

    Reduced to the (x, y) footprint at the mesh's lowest z (the sole) -- a
    flat rectangle, not the mesh's true (slightly domed) contact surface,
    which is the same level of approximation the rest of this module's
    geometry already sits at.

    Returns a dict from link name to a (4, 3) array of corner positions in
    that link's own local frame. Callers rotate/translate these by the
    link's live world pose every step -- this function only runs once
    (before the episode loop), since the geometry itself never changes.
    """
    stage = Usd.Stage.Open(usd_path)
    out: dict[str, np.ndarray] = {}
    for name in link_names:
        link_prim = next((p for p in stage.Traverse() if p.GetName() == name), None)
        if link_prim is None:
            raise ValueError(f"{name} 프림을 찾지 못했습니다: {usd_path}")
        mesh_prim = next(
            (
                c for c in Usd.PrimRange(link_prim, Usd.TraverseInstanceProxies())
                if c.HasAPI(UsdPhysics.CollisionAPI) and c.IsA(UsdGeom.Mesh)
            ),
            None,
        )
        if mesh_prim is None:
            raise ValueError(f"{name} 아래에 충돌 메시가 없습니다: {usd_path}")
        points = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
        if not points:
            raise ValueError(f"{name} 충돌 메시에 points 가 없습니다: {usd_path}")
        mesh_to_world = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        link_to_world = UsdGeom.Xformable(link_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        mesh_to_link = mesh_to_world * link_to_world.GetInverse()
        pts_in_link = np.array([mesh_to_link.Transform(Gf.Vec3d(*p)) for p in points])
        xmin, ymin, zmin = pts_in_link.min(axis=0)
        xmax, ymax, _zmax = pts_in_link.max(axis=0)
        out[name] = np.array([
            [xmin, ymin, zmin], [xmin, ymax, zmin],
            [xmax, ymin, zmin], [xmax, ymax, zmin],
        ])
    return out


# Default of EpisodeRecorder(appearance=): the scenario declares no look, the sidecar names none.
_NO_APPEARANCE = object()


class EpisodeRecorder:
    """Accumulates one episode's per-step measurements and writes them to disk.

    Call .step(env) once for every env.step() call whose done flag came back
    False. A step where done=True has *already* been reset internally by the
    time env.step() returns (ManagerBasedRLEnv._reset_idx() runs inside the
    same call -- confirmed in manager_based_rl_env.py, and independently
    relied on by check_fall_termination.py's last_z/last_angle), so every
    asset/command/entry read on a done=True step
    already belongs to the *next* episode's post-reset state, not this one's
    last step. run_s1a.py enforces the ordering; this class does not
    re-check it.
    """

    def __init__(self, out_dir, scenario_id, episode_seq, seed, provisional, region_codes,
                 sim_backend, foot_footprints, robot_model, asset_paths,
                 asset_paths_resolved, *, run_start_seq, appearance=_NO_APPEARANCE):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.scenario_id = scenario_id
        # The scenarios that declare R-TS15066-SSM-02 (S1e, S1f, S1g -- meta/scenarios.json) record the hri columns,
        # and their contact columns carry only what the robot's links and the carried box gave the person (_hri_step).
        self.hri = records_hri(scenario_id)
        # meta/scenarios.json's seed_rule: seed = hash(scenario_id,
        # episode_seq), one seed per episode. generation/seeding.py derives it
        # and run_s1a.py re-applies it immediately before this episode's
        # reset, so it is genuinely this episode's seed rather than a per-run
        # label -- every randomisation value recorded below is a function of
        # it.
        #
        # It does not yet follow that the episode can be reproduced on its
        # own: measured 2026-08-20, a standalone run agrees on every recorded
        # randomisation value and still produces a different trajectory.
        # Whole-run reproduction holds bit for bit. See run_s1a.py's own
        # comment at the seeding call and generation/README.md.
        self.episode_seq = episode_seq
        # The first seq of the run this episode is recorded in (run_s1a.py --seq). The episode
        # cannot be reproduced on its own, so the command that reproduces it starts there:
        # --seq run_start_seq --episodes (episode_seq - run_start_seq + 1). Schema §3.
        self.run_start_seq = run_start_seq
        # The look the row drew (appearance.draw_appearance): a dict (also when the run renders nothing), None only
        # under --no-appearance, or the default when the scenario declares none -- then sidecar randomization has
        # no ``appearance`` key.
        self.appearance = appearance
        self.seed = seed
        self.provisional = provisional
        self.region_codes = region_codes
        self.sim_backend = sim_backend
        # {link_name: (4, 3) local-frame corners} from read_foot_footprint(),
        # read once by run_s1a.py before the episode loop (geometry is fixed
        # for the asset's lifetime) and handed to every episode's recorder.
        self.foot_footprints = foot_footprints
        # The meta/joint_map and meta/assets.json key this episode's robot is
        # indexed by, and the paths actually resolved and loaded. Both become
        # schema §5 fields at conversion time (robot_model, scene_assets); a
        # hand-typed path there could name a different H1 variant than the one
        # in the running scene.
        self.robot_model = robot_model
        # Canonical (meta/assets.json join keys) and, separately, what the
        # simulator literally resolved -- the second is provenance (which S3
        # build of h1.usd) and is never used to look a licence up.
        self.asset_paths = asset_paths
        self.asset_paths_resolved = asset_paths_resolved
        self.steps: list[dict] = []
        # A body with no region code is a ground-contact body (the jointed dummy's feet): not a region, and
        # then the standing load's channel; without one, BR-12 is both a region and the ground channel.
        self._codes = sorted({c for c in region_codes if c is not None})
        self._region_mask, self._ground_mask = dummy_body_masks(region_codes)
        self.randomization: dict[str, object] = {}
        # Filled in by capture_randomization(). Top-level meta fields, not
        # nested under randomization -- these are the resolved SSM
        # parameters and the human-region sphere radius the command term
        # actually used to compute every step's separation_required /
        # min_distance_human, not a per-episode draw. Without this, once
        # meta/rules.json's provisional values are replaced by the real
        # standard copy, a published episode's separation_required could no
        # longer be reproduced from the repository -- the "provisional" flag
        # alone said *that* a placeholder was used, never *which* six
        # numbers it was.
        self.ssm_params: dict[str, float] = {}
        self.human_region_radius: float | None = None
        # Filled in by record_termination(), called by run_s1a.py right
        # after env.step() returns done=True and before the reset that same
        # call already performed overwrites the termination_manager's
        # per-term buffers. None for an episode that never actually
        # terminated (e.g. the run exhausted --max-steps first -- run_s1a.py
        # already warns on this and it is not the common path).
        self.termination: dict[str, object] | None = None

        # Cached on first step() call -- body_names/body order do not change
        # across an episode, so these structural lookups only need doing once.
        self._foot_idx: list[int] | None = None
        self._foot_names: list[str] | None = None
        self._robot_non_foot_mask: np.ndarray | None = None
        # Indices into robot_contact.body_names for the same two feet
        # self._foot_idx names, in the same order -- a *different* index
        # space than self._foot_idx (that one indexes robot.body_names, the
        # articulation's own order, for body_pos_w/body_quat_w). Used to
        # decide which foot's footprint corners actually belong in the
        # support polygon this step (see _STANCE_FORCE_THRESHOLD_N).
        self._robot_ankle_sensor_idx: list[int] | None = None
        # Indices into the dummy articulation's bodies, in the dummy contact sensor's body order
        # (substep_contact.sensor_order): the sensor lists them in the USD's order, the articulation along its
        # joint tree, and the region codes and masks are the sensor's.
        self._dummy_order: np.ndarray | None = None
        # Global common_step_counter (and step_dt to convert it to seconds)
        # at this episode's first recorded step -- lets a global sim-clock
        # time be reconstructed from the per-step (episode-relative)
        # timestamp if ever needed: global_time =
        # global_time_offset_s + timestamp.
        self._global_start_step: int | None = None
        self._step_dt: float | None = None
        # Read off the Imu sensor's own prim_path on the first step rather
        # than named here -- s1a_env_cfg.ROBOT_ROOT_LINK already derives it
        # from the joint map's provenance, and a second copy would drift.
        self._imu_link: str | None = None
        self._imu_body_idx: int | None = None
        # From the running sim, not a constant: zero_moment_point takes
        # gravity as a required argument for the same reason.
        self._gravity: float | None = None

        # How many contact samples the last step() folded into its peak. 1
        # means the substep hook is not firing and the data is back to one
        # snapshot per record step -- the defect substep_contact.py exists to
        # prevent, and one that looks like nothing at all in the numbers.
        self._substep_samples = 0
        # Set on the first step() alongside the other structural lookups.
        self._lever_cap_m = 0.0

        # Edge-detection state for safety.event (see EVENT_NAMES).
        self._prev_stop_step = -1
        self._prev_contact_any = False
        # -1 until _update_hold_release latches (see save()'s metadata note).
        self.hold_release_step = -1
        # Steps where a contact was present while the hold was still applying
        # force -- see step(). 0 or 1 is expected; more means the release
        # failed to fire ahead of a sustained contact.
        self.hold_contaminated_steps = 0

        # Realized-vs-drawn tracking for the entry axes. Read live inside
        # step() rather than once at episode end, for the same reset-inside-
        # step reason the class docstring gives: by the time the outer loop
        # could look back at env/command after a *terminated* episode, both
        # already belong to the next episode's reset.
        self._entry_ever_triggered = False
        self._entry_trigger_distance_realized: float | None = None
        # Onset to the scripted walk's last move -- not to the last triggered step once S1d's walk stops
        # (entry_geometry.EntrySpeedWindow).
        self._entry_speed = EntrySpeedWindow()

        # The hri scenarios only. The command term clears its per-episode fields at the reset inside the episode's
        # last env.step, so they are taken on every recorded step, never read back after the loop.
        self.task_started_step = -1       # the step the robot's task came at
        self.task_completed_step = -1     # task_path.CompletionTracker's step: the last of the held stretch
        self.task_stage_names: tuple[str, ...] = ()   # task_path.StageTracker's stages and the step each ended at
        self.task_stage_end_steps: list[int] = []
        self.human_stature_m: float | None = None     # the person's stature the seeing rule takes
        self.task_time_limit_s: float | None = None   # the environment's horizon
        self._task_fps: int | None = None             # the steps a second the stages' durations are counted in
        self.box_released_step = -1       # the step after whose physics the carried box was let go
        self.first_touch_step = -1        # the first step the robot or the box gave the person force
        # Per robot camera, the recorded steps it sat inside the held box (its fraction then leaves the box out).
        self.camera_in_box_steps = camera_in_box_counts()
        self.degenerate_reason: str | None = None
        self._prev_rack = False
        self._box_let_go = False          # let go after an earlier step: the box was not held this step
        self._condition: tuple | None = None
        self._dummy_rows: dict[str, list] = {key: [] for key in DUMMY_ROW_KEYS}
        self._head_body: int | None = None
        self._region_bodies: np.ndarray | None = None
        self._standing_heights: dict[str, float] | None = None
        self._pelvis_body: int | None = None

    def step(self, env) -> None:
        command = env.command_manager.get_term("base_velocity")
        robot = env.scene["robot"]
        dummy = env.scene["dummy"]
        contact = env.scene["dummy_contact"]
        # The robot's own inherited sensor (velocity_env_cfg.py). Its
        # body_names lists all 20 of the robot's bodies.
        #
        # All 20 carry a collider on h1.usd, which the env has loaded since
        # 2026-08-20 (s1a_env_cfg.py overrides spawn.usd_path); forced
        # overlap reports contact on 13. On h1_minimal.usd only 3 of the 20
        # do, and the collider counts in the contact_onset comment below
        # were measured on that build.
        robot_contact = env.scene["contact_forces"]

        if self._foot_idx is None:
            self._foot_names = foot_link_names(robot)
            self._foot_idx = [robot.body_names.index(n) for n in self._foot_names]
            # A *different* index space than self._foot_idx: this one is
            # into robot_contact.body_names (the sensor's own order), used
            # below to read each foot's live *load*, not its pose.
            self._robot_ankle_sensor_idx = [
                robot_contact.body_names.index(n) for n in self._foot_names
            ]
            self._robot_non_foot_mask = non_foot_mask(robot_contact.body_names)
            # Geometry, so it is fixed for the asset's lifetime; read once.
            self._lever_cap_m = dummy_lever_cap(ANTHROPOMETRY_PATH)
        if self._dummy_order is None:
            self._dummy_order = substep_contact.sensor_order(contact.body_names, dummy.body_names)
        imu_sensor = env.scene["robot_imu"]
        if self._imu_body_idx is None:
            # The sensor's own prim_path is the single source for which link
            # observation.imu / observation.root_pose are expressed in.
            self._imu_link = imu_sensor.cfg.prim_path.rsplit("/", 1)[-1]
            self._imu_body_idx = robot.body_names.index(self._imu_link)
        if self._global_start_step is None:
            self._global_start_step = int(env.common_step_counter)
            self._step_dt = float(env.step_dt)
            self._gravity = abs(float(env.sim.cfg.gravity[2]))

        # net_forces_w/body_pos_w/joint_pos/... are ProxyArray (warp-backed)
        # on this Isaac Lab version -- .torch is the explicit, non-deprecated
        # accessor onto the underlying torch view (isaaclab.utils.warp.
        # proxy_array.ProxyArray; implicit indexing works too but emits a
        # DeprecationWarning). command/entry buffers below need no such
        # accessor: S1aStraightWalkCommand builds them directly with
        # torch.zeros/torch.full in __init__, so they are already plain
        # torch.Tensor.
        # Peak over this env step's physics substeps, not the value at the
        # record instant. A contact born and gone between two record steps used
        # to be written as exactly 0 and vanish; substep_contact.py explains
        # why that is a sampling problem rather than an fps one, and carries
        # the measurement. drain() also resets the batch, so it is called once
        # per step and its result shared by every consumer below.
        peak = substep_contact.get(env).drain()
        self._substep_samples = peak["samples"]
        forces = peak["dummy_force"]

        # Force, height and robot-contact collapse onto regions together, by
        # argmax rather than independently: the body carrying a region's peak
        # hands over the height and the flag *it* had at the substep that peak
        # occurred in. Reduced separately they can come from different bodies and
        # different substeps, and then a region gets typed against a Table A.2
        # column chosen from a moment its measured force never happened in.
        # Everything here is in the dummy contact sensor's body order, which is
        # what region_codes indexes: substep_contact.install reads the heights
        # into it, and the v_rel further down reads the articulation's arrays
        # into it through self._dummy_order.
        rm = self._region_mask
        by_region, by_region_z, by_region_robot = reduce_region_peaks(
            [c for c in self.region_codes if c is not None], forces[rm], peak["dummy_z_at_peak"][rm],
            peak["robot_gate_at_peak"][rm], contact_eps_n=_CONTACT_EPS_N)
        region_z = np.array([by_region_z[c] for c in self._codes])
        region_robot_contact = np.array([by_region_robot[c] for c in self._codes])
        contact_force = np.array([by_region[c] for c in self._codes])
        # The robot sensor is read two ways because its two consumers want
        # different things. The support polygon describes which feet bear load
        # *at the record instant*, alongside the CoM and ZMP of that same
        # instant, so it takes the snapshot -- a substep peak would let a foot
        # that touched briefly inflate the polygon. The contact gate below
        # asks whether the robot touched the human at all during the step, so
        # it takes the peak, for the same reason the dummy side does.
        robot_forces_now = torch.linalg.norm(
            robot_contact.data.net_forces_w.torch[0], dim=-1).cpu().numpy()
        robot_forces_peak = peak["robot_force"]

        # Support polygon: only the feet bearing weight this step. Handed
        # each foot's 4 measured ground-contact corners regardless of load,
        # stability_margin sees a foot mid-swing (measured up to ~0.32 m off
        # the ground) still contributing its corners, inflating the polygon
        # and reading positive on 99% of steps even during ordinary
        # single-support gait, when the CoM legitimately sits outside the
        # one loaded foot's footprint 94% of the time (dynamic walking is
        # controlled falling, not a static stance). So each foot's corners
        # are gated on its own *live* load: ankle_forces >
        # _STANCE_FORCE_THRESHOLD_N, read from the same robot_contact sensor
        # as contact_onset (the ankle links carry a collider on either H1
        # build -- see the sensor comment above -- so this reading is
        # trustworthy, unlike net_forces_w on a body with no collider). 0,
        # 4, or 8 points depending on how many feet are loaded;
        # stability_margin's own contract returns -inf for 0 (flight phase
        # -- correct: no measured support at all that step).
        ankle_forces = robot_forces_now[self._robot_ankle_sensor_idx]
        stance = ankle_forces > _STANCE_FORCE_THRESHOLD_N
        support_xy = []
        for idx, name, is_stance in zip(self._foot_idx, self._foot_names, stance):
            if not is_stance:
                continue
            pos = robot.data.body_pos_w.torch[0, idx]
            quat = robot.data.body_quat_w.torch[0, idx]
            local_corners = torch.as_tensor(
                self.foot_footprints[name], dtype=pos.dtype, device=pos.device
            )
            world_corners = pos.unsqueeze(0) + math_utils.quat_apply(
                quat.unsqueeze(0).expand(4, -1), local_corners
            )
            support_xy.append(world_corners[:, :2].cpu().numpy())
        support_xy = np.concatenate(support_xy, axis=0) if support_xy else np.zeros((0, 2))
        # Mass-weighted centre of mass, not the root link position. The
        # schema publishes safety.com under that name so it has to be one,
        # and the old approximation also moved stability_margin by however
        # far root and CoM happened to be apart on a given step. Every number
        # here comes off the asset -- no mass is written down.
        body_mass = robot.data.body_mass.torch[0].cpu().numpy()
        body_com_pos = robot.data.body_com_pose_w.torch[0, :, :3].cpu().numpy()
        # (B, 6) = [lin_acc, ang_acc] and [lin_vel, ang_vel], world frame. The
        # angular velocity is not decoration: dH/dt carries an omega x (I omega)
        # term, and dropping it mis-states the ZMP by millimetres to
        # centimetres while the robot turns.
        body_acc = robot.data.body_com_acc_w.torch[0].cpu().numpy()
        body_vel = robot.data.body_com_vel_w.torch[0].cpu().numpy()
        # observation.contact.v_rel (schema §6): relative speed at the contact
        # point, which Table A.4's transferred-energy limit and equation (A.6)'s
        # velocity limit both need. PhysX contact points are not available here
        # -- track_contact_points wants a filter expression, and all three
        # variants of that failed (check_contact_filter.py: two attach but
        # report nothing over 700 steps, the third PhysX rejects) -- so this is
        # an approximation, and the schema says so.
        #
        # The arithmetic and the reason the lever arm is capped live in
        # contact_relative_speed(), which is pure numpy and tested without Isaac.
        # Regions collapse by maximum, the same reduction the force takes and for
        # the same reason.
        #
        # This is the record instant, not the substep peak the force above
        # takes. The two would have to be sampled together to feed an energy
        # comparison; v0.1 does not make one (rules.json marks the energy axis
        # unused_in_v0_1) and labels stay on the force axis.
        by_region_v = aggregate_contact_by_region(
            [c for c in self.region_codes if c is not None],
            contact_relative_speed(
                dummy.data.body_com_pose_w.torch[0, :, :3].cpu().numpy()[self._dummy_order],
                dummy.data.body_com_vel_w.torch[0].cpu().numpy()[self._dummy_order],
                body_com_pos,
                body_vel,
                lever_cap_m=self._lever_cap_m,
            )[self._region_mask],
        )
        contact_v_rel = np.array([by_region_v[c] for c in self._codes])

        # Documented world-frame and flattened (9,) on this Isaac Lab version,
        # so it needs reshaping but no body-to-world rotation.
        body_inertia = robot.data.body_inertia.torch[0].cpu().numpy().reshape(-1, 3, 3)

        com3 = center_of_mass(body_mass, body_com_pos)
        com = com3[:2]
        zmp = zero_moment_point(
            body_mass, body_com_pos, body_acc[:, :3], body_vel[:, 3:],
            body_acc[:, 3:], body_inertia, gravity=self._gravity,
        )
        robot_x = float(robot.data.root_pos_w.torch[0, 0])

        min_distance = float(command.min_distance[0])
        required = float(command.separation_required[0])
        separation_violation = bool(min_distance < required)

        if command.dummy_lin_force_applied is None:
            # Should not be reachable: env.s1a_entry is created at the very
            # first reset (mdp.events.reset_dummy_entry, mode="reset", also
            # fires at env construction -- s1a_env_cfg.py), and
            # _advance_dummy runs unconditionally once that entry exists,
            # before command_manager.compute() ever returns control from
            # env.step(). A step() call only ever happens after at least one
            # such env.step() has already returned. Kept as a loud failure
            # rather than a silent NaN if that ordering is ever violated.
            raise RuntimeError(
                "command.dummy_lin_force_applied 가 비어 있습니다 — _advance_dummy 가 "
                "아직 한 번도 돌지 않았습니다 (reset 이전에 step() 이 불렸을 가능성)."
            )
        # Audit trio: hold force, ground reaction force and
        # the dummy's realized yaw, so a future re-run after the hold is
        # redesigned can tell which episodes were recorded under the old,
        # contact-magnitude-untrustworthy hold.
        # hold_force_in_effect, not dummy_lin_force_applied: the latter is the
        # wrench queued for the *next* step (CommandManager.compute() runs
        # after the physics loop), so reading it here labelled every row with
        # the following step's force -- and read zero on exactly the step the
        # contact backstop fires. See S1aStraightWalkCommand._update_command.
        hold_force = float(torch.linalg.norm(command.hold_force_in_effect[0]))
        hold_torque = float(torch.linalg.norm(command.hold_torque_in_effect[0]))
        # The hold is a wrench. Watching the force alone would call a step
        # "ran free" while the attitude torque was still acting, so both the
        # release boundary and the contamination count use the pair.
        hold_applied = hold_force > 0.0 or hold_torque > 0.0
        # Sum of the two feet, not max (check_dummy_hold.py's
        # dummy_ground_force_n()): standing load
        # legitimately shifts between the two feet over time, and a
        # single-foot max under-reads whenever the load happens to be near
        # an even split. Same ambiguity as contact_force's own BR-12 column
        # (below): this is "force registered at the shins", ground-standing
        # load and a robot-vs-shin knock both included, not a value already
        # cleaned to isolate one cause -- audit telemetry, not a judgement.
        ground_reaction_force = float(forces[self._ground_mask].sum())
        _, _, yaw = math_utils.euler_xyz_from_quat(dummy.data.root_quat_w.torch[0:1])
        dummy_yaw_realized = float(yaw[0])

        events = np.zeros(len(EVENT_NAMES), dtype=bool)
        # ssm_violation: a level, mirroring separation_violation every step
        # it holds -- unlike the other two, its schema name has no "_onset"/
        # "_triggered" suffix implying an edge.
        events[0] = separation_violation
        # protective_stop_triggered: edge-triggered, firing only on the step
        # the stop first latches (the schema's own naming is the moment of
        # onset, not "currently stopped").
        stop_step_now = int(command.protective_stop_step[0])
        events[1] = stop_step_now >= 0 and self._prev_stop_step < 0
        self._prev_stop_step = stop_step_now

        # Episode-local, not the command term's own value. That one is
        # common_step_counter, which never resets, so writing it into a
        # sidecar next to episode-length arrays would hand a consumer an
        # index outside the arrays from the second episode of a run onward.
        # len(self.steps) is the index this step is about to take.
        # The first recorded step whose physics actually ran with no hold --
        # not the step the latch flipped on. The wrench for that step was
        # already queued before the release decision, so the dummy is only
        # free from the following step onward.
        if self.hold_release_step < 0 and int(command.hold_release_step[0]) >= 0 \
                and not hold_applied:
            self.hold_release_step = len(self.steps)

        # contact_onset: the dummy's own sensor cannot tell robot-contact
        # apart from the dummy simply standing on the ground, because
        # net_forces_w is a net force per *prim*, not per contact pair
        # (s1a_env_cfg.py's own comment on why track_contact_points was
        # dropped). BR-12 (the shins, the dummy's only ground-contact
        # region) swings between 0 and ~m*g every time the PD hold's
        # airborne/grounded oscillation crosses zero -- measured on a
        # robot-contact-free episode (seed 0, all 11 other regions exactly 0
        # for all 999 steps): 123 rising edges on BR-12 alone, every one of
        # which an any-region edge reads as contact_onset. Simply excluding
        # BR-12 is not the fix either: the shin is also a plausible first
        # thing a walking robot's own foot hits, so excluding it would drop
        # that real contact instead of the ground noise.
        #
        # The fix ORs two channels, each keyed on a body identity that
        # structurally cannot be explained by routine ground contact:
        #   - dummy-side: any of the *other* 11 dummy regions (never touch
        #     the ground -- anthropometry.yaml) showing force. Catches the
        #     robot touching any dummy region except the shin.
        #   - robot-side: any of the robot's own *non-foot* bodies showing
        #     force on its own sensor. The mask is built from the asset's
        #     bodies, not from a list of names, so it covers the bodies that
        #     carry a collider and no others: all 18 on h1.usd, which this
        #     env loads, and torso_link alone on h1_minimal.usd, where the
        #     other 17 have none and read 0 whatever touches them (see the
        #     sensor comment above).
        #
        # On either build neither channel catches foot<->shin -- the
        # dummy-side one because it structurally excludes BR-12, the
        # robot-side one because the feet are excluded from "non-foot".
        # Standing-pose height ranges measured directly: robot ankles
        # 0-0.081 m, dummy BR-12 (shins) 0.045-0.535 m above ground -- the
        # feet can barely reach anything else, so foot<->shin is not a
        # narrow theoretical edge case.
        #
        # Reading BR-12 for an excess over its own baseline (the idea: a
        # robot foot strike should add on top of whatever the dummy's own
        # standing/PD-hold dynamics already produce) does not close that
        # gap. Measured first: BR-12's own baseline, on episodes with *zero*
        # robot contact (seed 0 and both seed-30 episodes), already spans 0
        # to 10416/9620/11983 N and puts 9.7-15.9% of all steps above 2000 N
        # purely from the hold servo's own airborne/grounded oscillation --
        # there is no baseline stable enough to read an "excess" against,
        # and no confirmed real foot-vs-shin example in the measured data to
        # calibrate a threshold on even if there were. Abandoned as
        # unreliable rather than shipped as a guess; contact_onset_
        # unresolved_regions (meta, save() below) records this gap in the
        # file itself rather than only in a comment.
        robot_side_contact = bool(
            (robot_forces_peak[self._robot_non_foot_mask] > _CONTACT_EPS_N).any())
        dummy_side_contact = bool((forces[~self._ground_mask] > _CONTACT_EPS_N).any())
        contact_any = robot_side_contact or dummy_side_contact
        if self.hri:
            # The force matrix, not the net forces: what the robot's links and the box gave each body of the person.
            hri = self._hri_step(env, command, dummy, contact)
            contact_force, region_z, contact_any = hri["contact_force"], hri["region_z"], hri["touched"]

        # observation.contact.type (schema §8.1) -- which Table A.2 column a
        # contact is measured against. Gated on the robot-side channel, so a
        # region carrying only the dummy's own weight on the ground is not
        # typed as a robot-human contact at all.
        contact_type = classify_contact_type(
            contact_force,
            region_z,
            robot_contact=region_robot_contact,
            contact_eps_n=_CONTACT_EPS_N,
            pin_height_m=_PIN_HEIGHT_M,
        )
        events[2] = contact_any and not self._prev_contact_any
        self._prev_contact_any = contact_any
        if self.hri:
            # Typed by each region's own share of the robot's and the box's force (record_logic.region_contacts), and
            # the events of HRI_EVENT_NAMES: the directed S_p's breach, the touch's edge, the rack contact's edge.
            contact_type = hri["contact_type"]
            events = np.array([hri["columns"]["separation_violation"], events[2], hri["rack_edge"]], dtype=bool)

        # The whole point of releasing the dummy's hold is that a contact
        # magnitude is not standing next to a servo force. That property is
        # asserted in the command term's docstring; this counts it, so a
        # reader can check it in the data instead of trusting the claim. The
        # release gate is best-effort (min_distance_human is not a
        # surface-to-surface distance -- schema v0.1), and the contact
        # backstop bounds the damage to the step that first sees contact, so
        # this should be 0 or 1. Anything larger means the hold ran through a
        # contact and those steps' contact_force values are not usable.
        if contact_any and hold_applied:
            self.hold_contaminated_steps += 1

        # Episode-relative, not the global sim clock. env.episode_length_buf
        # is what ManagerBasedRLEnv itself zeroes on every reset
        # (manager_based_rl_env.py) and increments once per step -- exactly
        # the LeRobotDataset v3.0 convention (timestamp restarts at 0 per
        # episode). common_step_counter never resets within a process, so on
        # it a second episode's timestamps would start at wherever the first
        # one left off (e.g. 20.02s), which is not what the LeRobot
        # conversion expects and has no marker in the sidecar saying so.
        # global_time_offset_s (meta, below) keeps the global time
        # reconstructable for anyone who does want it: global = offset + this.
        #
        # -1 because episode_length_buf is already incremented for *this*
        # step by the time step() reads it (manager_based_rl_env.py
        # increments it near the top of step(), before returning), so
        # without it the first recorded sample of an episode reads
        # timestamp=step_dt (0.02s), not 0.0 -- LeRobotDataset v3.0
        # expects frame_index (and so timestamp) to start at exactly 0. This
        # single -1 also fixes global_time_offset_s's matching one-step
        # over-count: that field is captured the same already-incremented
        # way (by design, so it equals the *true* absolute time of the first
        # recorded sample -- see its own comment below), so restoring
        # timestamp to 0-indexed is what makes global_time_offset_s +
        # timestamp equal the true absolute time again, rather than
        # double-adding that one step's dt.
        timestamp = float(env.episode_length_buf[0] - 1) * float(env.step_dt)

        entry = getattr(env, "s1a_entry", None)
        if entry is not None and not self.hri:
            triggered_now = bool(entry["triggered"][0])
            dummy_xy = dummy.data.root_pos_w.torch[0, :2].cpu().numpy().copy()
            if triggered_now and not self._entry_ever_triggered:
                self._entry_ever_triggered = True
                # Mirrors the along-track check _advance_dummy itself makes
                # (mdp/commands.py: dist_to_crossing = entry["x_cross"] -
                # robot_x) -- the realized counterpart to the drawn
                # trigger_distance threshold, generally a little smaller
                # than it because the trigger is only checked once per
                # (discrete) step.
                self._entry_trigger_distance_realized = float(entry["x_cross"][0]) - robot_x
            # The window reads the entry's flags itself (triggered, and walk_ended where the walk can
            # stop), so that read is the one the tests step -- this module cannot be imported without Isaac.
            self._entry_speed.observe(entry, xy=dummy_xy, t=timestamp)

        self.steps.append({
            "timestamp": timestamp,
            "joint_pos": robot.data.joint_pos.torch[0].cpu().numpy(),
            "joint_vel": robot.data.joint_vel.torch[0].cpu().numpy(),
            "joint_torque": robot.data.applied_torque.torch[0].cpu().numpy(),
            "root_pose": _pose_xyzw_to_wxyz(
                robot.data.root_link_pose_w.torch[0].cpu().numpy()
            ),
            # quat from the same link the Imu sensor sits on (self._imu_link),
            # then the sensor's own body-frame angular velocity and proper
            # linear acceleration -- schema §6 observation.imu is (10,).
            "imu": np.concatenate([
                _quat_xyzw_to_wxyz(
                    robot.data.body_link_pose_w.torch[0, self._imu_body_idx, 3:7]
                    .cpu().numpy()
                ),
                imu_sensor.data.ang_vel_b.torch[0].cpu().numpy(),
                imu_sensor.data.lin_acc_b.torch[0].cpu().numpy(),
            ]),
            "com": com3,
            "zmp": zmp,
            "root_vel": robot.data.root_vel_w.torch[0].cpu().numpy(),
            # The joint position target in radians, not the policy's raw output.
            # ActionManager.action is what was *sent in* -- for this task a
            # normalised vector that the term turns into a target with
            # `raw * 0.5 + default_joint_pos` (JointPositionActionCfg, scale=0.5,
            # use_default_offset=True). Publishing that under a column the schema
            # documents as radians states a unit the numbers do not have.
            # processed_actions is the target itself, and with
            # joint_names=[".*"] the term indexes the articulation's own
            # order -- the same order joint_names publishes (joint_actions.py:76).
            "action": processed_action(env),
            # [lin_vel_x, lin_vel_y, ang_vel_z] -- the velocity command the
            # command term actually issued this step (command.command, the
            # exact tensor the policy reads as its own "velocity_commands"
            # observation), not the drawn walk_speed alone. lin_vel_x is
            # walk_speed scaled by the SSM ramp (mdp/safety.py's
            # command_scale) -- the one signal that tells "the supervisor
            # acted" apart from "the policy slowed on its own". ang_vel_z is
            # S1aStraightWalkCommand._heading_correction()'s output exactly
            # (mdp/commands.py sets self._command[:, 2] to it directly) --
            # an active controller intervention on the robot's heading that
            # was otherwise invisible in the recorded data. lin_vel_y is
            # always 0 for this command term but recorded anyway so the
            # column is the same 3-vector the policy's observation is.
            "command_velocity": command.command[0].cpu().numpy(),
            "contact_force": contact_force,
            "contact_v_rel": contact_v_rel,
            "min_distance_human": min_distance,
            "separation_required": required,
            "separation_violation": separation_violation,
            "stability_margin": stability_margin(support_xy, com),
            "hold_force": hold_force,
            "hold_torque": hold_torque,
            "ground_reaction_force": ground_reaction_force,
            "region_z": region_z,
            "contact_type": np.array(contact_type),
            "dummy_yaw_realized": dummy_yaw_realized,
            "safety_event": events,
            # Which foot (if any) actually contributed to the support
            # polygon this step, in self._foot_names order (so the
            # load-bearing-only gating above can be audited) -- meta's
            # stance_foot_names (save(), below) documents the column order.
            "stance_foot": stance,
        })
        if self.hri:
            self.steps[-1].update(hri["columns"])

    def _hri_step(self, env, command, dummy, contact) -> dict:
        """One recorded step's hri columns, read off the command term (generation/envs/hri/commands.py) as it stands
        after this step's physics, and its contact columns off the dummy sensor's force matrix.

        The contact columns count only what the robot's links and the box (let go or not) gave the person, peaked
        over the step's substeps (substep_contact.install_hri): a region is typed by its own share, and the touch --
        contact_onset's edge -- is any of the person's bodies getting force, the feet included; the box is let go at
        the first such step. The rack contact is the step's largest force from something that is not the person on a
        robot link that is not a foot, or on the box while it was held (record_logic.rack_residual). The command
        term's per-episode fields are cleared at the reset inside the episode's last env.step, so the task's start,
        its completion and the step each of its stages ended at are taken here, step by step."""
        index = len(self.steps)
        touch = command.touch
        if touch is None or touch.samples != env.cfg.decimation:
            raise RuntimeError(
                f"명령 항의 접촉 창이 이 스텝의 서브스텝 {env.cfg.decimation}개를 다 읽지 않았습니다 "
                f"({None if touch is None else touch.samples}개) — 상자를 놓는 접촉과 기록하는 접촉이 다른 창이 됩니다")
        if command.hri is None:
            raise RuntimeError("명령 항의 hri 가 비었습니다 — 이 스텝의 사람과 로봇의 기하가 셈해지지 않았습니다")
        peak = substep_contact.get_hri(env).drain()
        human = peak["human_force"]
        _, force, z, types = region_contacts(human, peak["human_z_at_peak"], self.region_codes,
                                             contact_eps_n=_CONTACT_EPS_N, pin_height_m=_PIN_HEIGHT_M)
        now = touched(human, _CONTACT_EPS_N)
        if now and self.first_touch_step < 0:
            self.first_touch_step = index
        residual = rack_residual(peak["link_residual"], self._robot_non_foot_mask, peak["box_residual"],
                                 box_held=not self._box_let_go)
        self._prev_rack, rack_edge = rack_contact_edge(self._prev_rack, residual)
        if int(command.box_released_step) >= 0 and self.box_released_step < 0:
            self.box_released_step = int(command.box_released_step)
            self._box_let_go = True
        if int(command.task_started_step) >= 0 and self.task_started_step < 0:
            self.task_started_step = int(command.task_started_step)
        if int(command.completion.step) >= 0 and self.task_completed_step < 0:
            self.task_completed_step = int(command.completion.step)
        self._take_stages(command)
        self.camera_in_box_steps = count_cameras_in_box(self.camera_in_box_steps, command.cameras_in_box)

        if self._pelvis_body is None:
            import yaml

            from generation import hri_geometry
            from generation.dummy import joints_spec
            from generation.envs.s1d import jointed_walk_metrics

            # The person's root is the pelvis's body (hri_geometry.pelvis_body); the head is the body tagged BR-01 and
            # each region's height is its first body's -- read through the sensor's order into the articulation's.
            self._pelvis_body = list(dummy.body_names).index(hri_geometry.pelvis_body(joints_spec.load()))
            codes = list(self.region_codes)
            self._head_body = int(self._dummy_order[codes.index("BR-01")])
            self._region_bodies = np.array([self._dummy_order[codes.index(c)] for c in self._codes])
            self._standing_heights = jointed_walk_metrics.standing_region_heights(
                yaml.safe_load(Path(ANTHROPOMETRY_PATH).read_text(encoding="utf-8")))
        pose = dummy.data.body_link_pose_w.torch[0, self._pelvis_body].cpu().numpy()
        vel = dummy.data.body_link_vel_w.torch[0, self._pelvis_body].cpu().numpy()
        columns = hri_step_columns(command.hri, pelvis_pose_xyzw=pose, pelvis_vel=vel,
                                   fractions=command.camera_fractions)

        # What dummy_degenerate_reason reads of the dummy at this step.
        entry = env.s1a_entry
        x, y = (float(v) for v in dummy.data.root_quat_w.torch[0, :2])
        up_z = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
        rows = self._dummy_rows
        rows["head_z"].append(float(dummy.data.body_pos_w.torch[0, self._head_body, 2]))
        rows["tilt"].append(float(np.degrees(np.arccos(up_z))))
        rows["released"].append(bool(command.hold_released[0]))
        rows["triggered"].append(bool(entry["triggered"][0]))
        rows["walk_ended"].append(bool(entry["walk_ended"][0]))
        rows["region_heights"].append(dummy.data.body_pos_w.torch[0, :, 2].cpu().numpy()[self._region_bodies].copy())
        return {"contact_force": force, "region_z": z, "contact_type": types, "touched": now, "rack_edge": rack_edge,
                "columns": columns}

    def capture_randomization(self, env, command) -> None:
        """Record what this episode drew, before any step has run.

        Only the *_drawn values and the static controller settings are
        knowable at this point. The *_realized counterparts are filled in
        by _finalize_randomization(), from state step() accumulated live —
        never by re-reading env/command after the step loop ends, which (for
        an episode that ended in termination rather than exhausting its step
        budget) would already belong to the next episode's reset. Same
        hazard the class docstring describes for step() itself.
        """
        entry = getattr(env, "s1a_entry", None)
        self.randomization = {
            "walk_speed_drawn": float(command._walk_speed[0]),
            # The command term only exposes the discretized step count
            # (self._delay_steps, floor(delay_s / step_dt)) -- the
            # continuous millisecond value _resample_command draws is a
            # local variable there and never stored, so there is no
            # separate "drawn" value to recover; this already *is* the
            # operational one.
            "sensor_delay_steps": int(command._delay_steps[0]),
            # Controller settings, not judgement thresholds (those live in
            # meta/rules.json) -- recorded because they move the outcome
            # distribution directly: without them, two
            # episodes with the same seed but a different slowdown_factor
            # would look inexplicably different.
            "slowdown_factor": float(command.cfg.slowdown_factor),
            "stop_band_factor": float(command.cfg.stop_band_factor),
            "heading_correction_gain": float(command.cfg.heading_correction_gain),
            "heading_correction_ang_vel_z_range": list(command.cfg.heading_correction_ang_vel_z_range),
        }
        if self.hri:
            # The row is these scenarios' draw (randomization.condition, written at save once the task's step is
            # known) and the reset draws nothing. The entry holds the script's values under S1d's names -- seconds
            # for a time trigger, the walk's direction, the person's start -- and S1d's speed scaling and heading
            # correction do not act: neither is recorded as if it were S1d's.
            self.randomization = {k: self.randomization[k] for k in ("walk_speed_drawn", "sensor_delay_steps")}
            # A controller setting: the share of the dummy's weight the hold carries while it holds (hold_force).
            self.randomization["human_hold_support_share"] = float(command.cfg.dummy_hold_support_share or 0.0)
            self._condition = (dict(command.row), command.site, command.plan)
            # The task's stages as the episode starts (all open), so a recording without a step still names them;
            # _hri_step takes them again every step. The person's stature, the task's time limit (the environment's
            # horizon) and the step rate the stages' durations are counted in hold for the episode.
            self._take_stages(command)
            self.human_stature_m = float(command.stature_m)
            self.task_time_limit_s = float(env.max_episode_length_s)
            self._task_fps = round(1.0 / float(env.step_dt))
            entry = None
        if self.appearance is not _NO_APPEARANCE:
            self.randomization["appearance"] = None if self.appearance is None else dict(self.appearance)
        if entry is not None:
            self.randomization.update({
                "human_entry_speed_drawn": float(entry["speed"][0]),
                "human_entry_angle_rad_drawn": float(entry["angle"][0]),
                "entry_trigger_distance_drawn": float(entry["trigger_distance"][0]),
                # Not drawn: which way the dummy faces, deg about +z. 0 = +x, the robot sees its
                # back -- S1a, a scenario setting like slowdown_factor above. S1d derives it from the
                # drawn entry angle (entry_geometry.travel_facing_yaw_deg): the dummy faces its walk.
                # Recorded because it moves which body regions a contact lands on.
                "human_facing_yaw_deg": float(entry["facing_yaw_deg"][0]),
            })
            if command.cfg.dummy_hold_clearance_m is not None:
                # A controller setting (S1d): how far above its standing height the hold servo carries the
                # dummy. Its presence says the recorded hold_force includes the dummy's weight while held.
                self.randomization["human_hold_clearance_m"] = float(command.cfg.dummy_hold_clearance_m)

        # The six resolved SSM parameters the command term actually used for
        # every separation_required call this episode (mdp/safety.py never
        # stores them -- it only takes params in and a distance out).
        # command.ssm_params is already the standard copy once meta/rules.json
        # is filled and the provisional fallback until then; the "provisional"
        # flag below says *that* a placeholder was used, this says *which*
        # six numbers -- without this, a published episode's
        # separation_required stops being reproducible from the repository
        # the moment the provisionals are replaced by the real standard copy.
        self.ssm_params = dict(command.ssm_params)
        if self.hri:
            # R-TS15066-SSM-02's five parameters and its stop table: what the S_p column was computed with.
            self.ssm_params = hri_ssm_params(command._ssm, command._stop_table)
        # Same reason: min_distance_human's sphere-approximation radius
        # (schema §6 "min_distance_human 산출 방법") is a cfg default, not a
        # standard value, but it still has to be known to reproduce the
        # recorded distance.
        self.human_region_radius = float(command.cfg.human_region_radius)

        # Friction and mass as the scene has them, because a declared range
        # and the scene have differed here: floor friction is not
        # randomized at all (the inherited event targets the robot's own
        # bodies, not the terrain, with a zero-width range), and robot mass
        # deviation is roughly ±25% multiplicative log-uniform on the robot's
        # base body, where ±5% had been declared. With these a consumer can
        # see such a discrepancy from the data itself: read from the live scene
        # (not the declared range), so this is what randomize_rigid_body_
        # material/randomize_rigid_body_mass (isaaclab.envs.mdp.events)
        # actually wrote in, not a restatement of their config.
        robot = env.scene["robot"]
        # root_view.get_material_properties() is the same PhysX tensor-API
        # call isaaclab's own randomize_rigid_body_material reads before
        # overwriting (isaaclab/envs/mdp/events.py) -- (num_envs, num_shapes,
        # 3) = (static_friction, dynamic_friction, restitution) per shape.
        # Every shape gets the same bucket-sampled value here (S1aEnvCfg
        # never narrows body_names off the robot's ".*"), so the per-shape
        # mean and the per-shape value agree; the mean is taken anyway
        # rather than assuming shape 0 is representative.
        material = wp.to_torch(robot.root_view.get_material_properties())[0]  # (num_shapes, 3)
        self.randomization["robot_static_friction_realized"] = float(material[:, 0].mean())
        self.randomization["robot_dynamic_friction_realized"] = float(material[:, 1].mean())
        # Which body index "base" (H1RoughEnvCfg's add_base_mass asset_cfg)
        # resolved to is read from the event term's own already-resolved
        # SceneEntityCfg, not re-derived by name here -- the "don't hand-type
        # what the asset/config already resolved" of CONTRIBUTING.md
        # 「관절 순서」 applies to this index exactly as much as to a joint
        # index.
        # A scenario that does not randomise the base mass (the conditions-table scenarios turn the term off) has
        # no realized draw to record.
        if any("add_base_mass" in names for names in env.event_manager.active_terms.values()):
            mass_body_ids = env.event_manager.get_term_cfg("add_base_mass").params["asset_cfg"].body_ids
            self.randomization["robot_base_mass_kg_realized"] = float(
                robot.data.body_mass.torch[0, mass_body_ids].sum()
            )

    def _finalize_randomization(self) -> None:
        """Fill in the *_realized values from what step() actually observed."""
        if self.hri:
            if self._condition is None:
                raise RuntimeError("capture_randomization() 이 불리지 않아 조건표의 행을 적을 수 없습니다")
            row, site, plan = self._condition
            design, cell = conditions.row_design(conditions.scenario_entry(self.scenario_id), self.episode_seq)
            self.randomization["condition"] = condition_record(self.episode_seq, row, site, plan,
                                                              task_started_step=self.task_started_step,
                                                              design=design, grid_cell=cell)
            return
        if self.steps:
            # World-frame x-velocity, averaged over every recorded step --
            # a reasonable stand-in for "how fast it actually walked" only
            # because heading correction (mdp/commands.py) actively holds
            # yaw near 0 for this whole scenario, so world +x tracks the
            # robot's own forward direction throughout.
            self.randomization["walk_speed_realized"] = float(
                np.mean([s["root_vel"][0] for s in self.steps])
            )
        else:
            self.randomization["walk_speed_realized"] = None

        self.randomization["entry_triggered_realized"] = self._entry_ever_triggered
        self.randomization["entry_trigger_distance_realized"] = self._entry_trigger_distance_realized

        # Net displacement over elapsed time between trigger onset and the
        # walk's last move -- the dummy's *measured* physical speed (PD hold
        # tracking a ramping target), not a restatement of the scripted drawn
        # speed. None when the window spans fewer than two steps.
        self.randomization["human_entry_speed_realized"] = self._entry_speed.speed()

    def _take_stages(self, command) -> None:
        """The task's stages and the step each ended at (-1: open), copied off the command term's tracker
        (task_path.StageTracker -- a new one at every reset, so they are copied while it is this episode's)."""
        self.task_stage_names = tuple(command.stages.names)
        self.task_stage_end_steps = [int(s) for s in command.stages.end_steps]

    def record_termination(self, env) -> None:
        """Record which termination condition ended this episode, right after
        env.step() returns done=True and before the next episode's own step()
        or reset overwrites the same state.

        No terminated/truncated/which-term-fired field existed before this:
        step 5 has to label `fall` outcomes and would otherwise have to
        re-derive them from stability_margin alone, and the recorder already
        skips the terminal step itself (the class docstring's reset-inside-
        step() hazard) -- the one step that would have shown the fall in the
        time series is exactly the step whose termination reason this method
        captures instead.

        Safe to call after the reset that same env.step() already performed
        internally: TerminationManager.compute() (called before _reset_idx())
        is what sets .terminated/.time_outs/the per-term buffers this reads,
        and _reset_idx() does not touch any of them afterwards (confirmed
        against isaaclab's termination_manager.py/manager_based_rl_env.py) --
        the same ordering guarantee the class docstring already relies on for
        why step() itself must be skipped on this exact step, not a new one.
        """
        tm = env.termination_manager
        fired = [name for name in tm.active_terms if bool(tm.get_term(name)[0])]
        self.termination = {
            "terminated": bool(tm.terminated[0]),
            "truncated": bool(tm.time_outs[0]),
            # Usually exactly one name (fall_height, fall_orientation or
            # time_out for S1a -- base_contact is disabled, see
            # S1aEnvCfg) but a list because more than one term can fire on
            # the same step (e.g. root height and orientation crossing their
            # thresholds together).
            "reason": fired,
        }

    @property
    def episode_id(self) -> str:
        """The episode's name. save() writes under it; S1d's media writer names its files by it too."""
        return f"{self.scenario_id}_{self.episode_seq:04d}_seed{self.seed}"

    def save(self, media: dict | None = None) -> Path:
        """Write the npz and the sidecar. ``media`` is MediaWriter.close()'s result (S1d with
        cameras): its per-step arrays join the npz, its file entries become the sidecar's
        ``media``. Without it the output is what it always was -- S1a's is byte-identical."""
        self._finalize_randomization()
        episode_id = self.episode_id
        degenerate = len(self.steps) < MIN_VALID_STEPS
        if self.hri:
            # Too short, or the dummy was not usable before it was first touched (record_logic.dummy_degenerate_reason
            # -- it fell, tipped while held, or stood off its height). The robot is not looked at: its fall is an outcome.
            if degenerate:
                self.degenerate_reason = f"기록 스텝 {len(self.steps)}개 — {MIN_VALID_STEPS}개 미만"
            else:
                self.degenerate_reason = dummy_degenerate_reason(
                    {k: np.asarray(v) for k, v in self._dummy_rows.items()}, self.first_touch_step,
                    codes=self._codes, standing=self._standing_heights)
            degenerate = self.degenerate_reason is not None

        if self.steps:
            arrays = {
                key: np.stack([s[key] for s in self.steps])
                for key in self.steps[0]
                if key != "timestamp"
            }
            arrays["timestamp"] = np.array([s["timestamp"] for s in self.steps])
        else:
            # Degenerate case: nothing to stack (see MIN_VALID_STEPS). Shape
            # (0,) for every field keeps the file's key set predictable for
            # any downstream reader that just checks `for k in expected: k in
            # d.files` -- per-field trailing dims (dof, n_region, ...) cannot
            # be known without a live read, and there is nothing useful in
            # this file for them to matter for.
            # An hri recording carries every hri key as well, or the converter reads it as half of one.
            arrays = zero_step_arrays(hri=self.hri)
        media_files = None
        if media is not None:
            media_arrays, media_files = split_media(media)
            arrays.update(media_arrays)
        np.savez_compressed(self.out_dir / f"{episode_id}.npz", **arrays)

        onset_gap_regions, onset_gap_reason = contact_onset_gap(self.region_codes)
        meta = {
            "episode_id": episode_id,
            "scenario_id": self.scenario_id,
            # episode_seq is the reproduction key and seed is what it derives
            # to; both are written so a reader can check the derivation
            # without importing anything. seed is the value that becomes the
            # schema's randomization_seed at LeRobot conversion time.
            "episode_seq": self.episode_seq,
            "run_start_seq": self.run_start_seq,
            "seed": self.seed,
            "steps": len(self.steps),
            "degenerate": degenerate,
            "robot_model": self.robot_model,
            "asset_paths": self.asset_paths,
            "asset_paths_resolved": self.asset_paths_resolved,
            # Which link observation.imu / observation.root_pose are in.
            "imu_link": self._imu_link,
            # Evidence that the substep hook ran. The converter does not read
            # it; a human or a check script comparing it against decimation+1
            # does, and an episode recorded without the hook says 1 here.
            "substep_contact_samples": int(self._substep_samples),
            # Which units `action` is in. The schema listed this as required and
            # nothing produced it, so a consumer could not tell a position
            # command from a torque one. Which of the two the canon should
            # be is still open.
            "action_space": ACTION_SPACE,
            "contact_region_names": self._codes,
            "safety_event_names": list(HRI_EVENT_NAMES if self.hri else EVENT_NAMES),
            # Column order for the stance_foot per-step field -- same pattern
            # as contact_region_names/safety_event_names above. None (not an
            # empty list) in the rare case step() never ran once to discover
            # the foot names.
            "stance_foot_names": self._foot_names,
            # An explicit, in-file record of what contact_onset cannot see,
            # rather than only a code comment
            # -- see that block's docstring in step() for the full
            # measurement (BR-12's own baseline already spans 0-12000 N on
            # episodes with zero robot contact, so an excess-over-baseline
            # detector was tried and abandoned as unreliable, not simply
            # skipped).
            # Which bodies are the ground channel decides what goes unseen -- the rigid dummy's shins,
            # the jointed dummy's feet (safety.contact_onset_gap).
            "contact_onset_unresolved_regions": onset_gap_regions,
            "contact_onset_unresolved_reason": onset_gap_reason,
            # global = global_time_offset_s + a step's (episode-relative)
            # timestamp. None in the (rare, degenerate) case
            # step() never ran once to capture it.
            "global_time_offset_s": (
                self._global_start_step * self._step_dt
                if self._global_start_step is not None
                else None
            ),
            # The episode-local step at which the dummy's hold servo was
            # switched off and the dummy became a free body (-1 if it never
            # was). See S1aStraightWalkCommand._update_hold_release.
            #
            # This marks the free-body boundary. Which ISO/TS 15066 column
            # applies is a separate, per-region question answered by
            # contact_type (schema §8.1) -- not by this field.
            "hold_release_step": self.hold_release_step,
            # 0 or 1 expected -- see step(). Larger means the hold ran through
            # a contact and those steps' contact_force is not usable.
            "hold_contaminated_steps": self.hold_contaminated_steps,
            # observation.contact.type is gated on the robot-side channel,
            # which masks out the robot's own feet -- they carry ground load
            # every step, so leaving them in would type every walking step as
            # a human contact. The cost is that the robot's *foot* on a fallen
            # dummy is invisible to that gate, and that is exactly the
            # clamping case the quasistatic value exists for. The dummy-side
            # channel cannot fill the gap either: once the dummy is down, its
            # non-BR-12 regions carry ground load too, so force on them no
            # longer implies robot contact.
            #
            # Recorded here rather than left in a comment, the same way
            # contact_onset_unresolved_* is. Closing it needs a signal that
            # separates "foot on the floor" from "foot on a body" -- foot
            # height above the floor while loaded is the cheapest candidate
            # and has not been tried.
            "contact_type_unresolved_reason": (
                "로봇 발이 넘어진 더미를 밟는 접촉은 유형이 매겨지지 않는다. "
                "로봇 쪽 채널이 발을 제외하기 때문이며(발은 매 스텝 지면 하중을 받는다), "
                "더미 쪽 채널도 넘어진 뒤에는 지면 하중과 구분되지 않는다. "
                "협착 사례 자체가 이 경로로 들어오므로 구조적 사각지대다."
            ),
            "randomization": self.randomization,
            "sim_backend": self.sim_backend,
            "provisional": self.provisional,
            # The six resolved SSM parameters and the human-region radius
            # the command term actually used this episode (see
            # capture_randomization()) -- what "provisional" above says was
            # used, not just that something was. Empty dict / None only in
            # the (should-not-happen) case capture_randomization() was never
            # called before save().
            "ssm_params": self.ssm_params,
            "human_region_radius": self.human_region_radius,
            # Which termination condition ended the episode (see
            # record_termination()) -- None if the run never actually
            # terminated this episode (e.g. --max-steps exhausted first;
            # run_s1a.py already prints a warning for that case).
            "termination": self.termination,
        }
        if self.hri:
            # The force matrix sees every body of the person, the feet included, and nothing but the robot and the box.
            meta["contact_onset_unresolved_regions"] = []
            meta["contact_onset_unresolved_reason"] = CONTACT_ONSET_GAP_REASON
            meta["contact_type_unresolved_reason"] = CONTACT_TYPE_GAP_REASON
            # The step the task was done at (-1: not in time), the box's release, and why the episode is degenerate.
            meta["task_completed_step"] = step_or_none(self.task_completed_step)
            meta["box_released_step"] = step_or_none(self.box_released_step)
            meta["degenerate_reason"] = self.degenerate_reason
            # Each of the task's stages: whether it ended, the step it ended at and the seconds it took from the end of
            # the one before (task_path.StageTracker); the person's stature the seeing rule takes; the task's time
            # limit -- the environment's horizon.
            meta["task_stages"] = task_stages_record(self.task_stage_names, self.task_stage_end_steps,
                                                     self.task_started_step, self._task_fps)
            meta["human_stature_m"] = self.human_stature_m
            meta["task_time_limit_s"] = self.task_time_limit_s
            # How many recorded steps each robot camera spent inside the held box: on those its visible fraction leaves
            # the box out while its picture shows the box's inside (visibility.robot_camera_fractions).
            meta["camera_in_box_steps"] = dict(self.camera_in_box_steps)
        if media_files is not None:
            # Which files hold this episode's camera streams (and a pack, only when meta/cameras.json
            # declares one -- none since 2026-09-30), with the recorder's counts and the canon fingerprint.
            meta["media"] = media_files
        absent = [k for k in SIDECAR_REQUIRED_KEYS if k not in meta]
        if absent:
            # Written, not warned: a sidecar the conversion cannot read is not
            # a partial result, it is an unusable one, and finding that out at
            # step 7 costs a whole batch.
            raise ValueError(
                f"{episode_id} 사이드카에 필수 키가 없습니다: {absent} — "
                "SIDECAR_REQUIRED_KEYS 참고"
            )
        (self.out_dir / f"{episode_id}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.out_dir / f"{episode_id}.npz"
