"""S1a scene: the H1 flat-terrain velocity env with a human dummy in it.

Inherits H1FlatEnvCfg so the observation layout the published checkpoint was
trained on stays untouched. Only the scene grows.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg
from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ImuCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg

REPO_ROOT = Path(__file__).resolve().parents[3]
DUMMY_USD = REPO_ROOT / "generation/dummy/build/keti_dummy_ts15066_v1.usd"

# The link observation.imu and observation.root_pose describe. Read from the
# joint map's provenance (dump_joint_names.py wrote it off the live
# articulation) rather than typed here -- "pelvis" is an asset fact, and a
# second hand-written copy is what CONTRIBUTING.md 「관절 순서」 is about.
# recorder.py does not copy it either: it reads the leaf back off this
# sensor's own prim_path and writes it into the sidecar as imu_link, so the
# frame the field is expressed in is never a guess downstream.
ROBOT_ROOT_LINK = yaml.safe_load(
    (REPO_ROOT / "meta/joint_map/unitree_h1.yaml").read_text(encoding="utf-8")
)["provenance"]["articulation_root"].rsplit("/", 1)[-1]


@configclass
class S1aEnvCfg(H1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # A single env for the vertical slice: the goal is one clean traversal,
        # not throughput.
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0

        # Load h1.usd, not the h1_minimal.usd the inherited task picks.
        #
        # H1RoughEnvCfg uses H1_MINIMAL_CFG, whose docstring says it "removes
        # most collision meshes to speed up simulation". Measured on the two
        # USD files: h1.usd carries a collisions scope on 20/20 bodies,
        # h1_minimal.usd on 3/20 (both ankles and the torso). With the minimal
        # asset the robot's arms and legs cannot report a contact no matter
        # what they touch -- and per-region contact is the reason this dataset
        # exists. The asset licence inventory (docs/assets/) also pins h1.usd
        # as *the* BSD-3 path, so this is what the licence record already
        # says we load.
        #
        # The one thing that blocked the swap was whether the published
        # pretrained policy (Isaac-Velocity-Flat-H1-v0) was trained on the
        # minimal asset. Measured (check_asset_swap.py, 2026-08-20): the
        # walk is unchanged -- 2/2 episodes completed on both, mean forward
        # speed +0.961 vs +0.972 m/s, travel +5.72 vs +5.79 m -- and a forced
        # overlap reports 13/20 bodies against the minimal asset's 3/20, the
        # eleven extra being both elbows, both knees, all four hip links, the
        # pelvis and the shoulders' chain. Nothing else differs: upstream's
        # H1_MINIMAL_CFG is H1_CFG.copy() with only spawn.usd_path changed,
        # and the two files agree on all 19 joint names and their order and on
        # all 20 body names and masses.
        self.scene.robot.spawn.usd_path = (
            f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd"
        )

        # Assigning an attribute that InteractiveSceneCfg's base class does not
        # declare is the supported way to add a scene entity outside __init__:
        # InteractiveScene._add_entities_from_cfg() walks self.cfg.__dict__ and
        # picks up anything not in InteractiveSceneCfg.__dataclass_fields__, so
        # a plain instance attribute set here is discovered the same way the H1
        # scene cfg's own declared fields (robot, contact_forces, ...) are.
        self.scene.dummy = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Dummy",
            # The asset is deliberately physics-engine neutral (build_dummy.py
            # bakes in keti:springConstant as a plain attribute, not a PhysX
            # schema, for the same reason) — it never applies PhysxContactReportAPI
            # itself. activate_contact_sensors=True is what makes the spawner
            # apply it to every rigid body under the dummy at spawn time;
            # without it, ContactSensor._initialize_impl() finds zero bodies
            # with the API and raises (confirmed: "could not find any bodies
            # with contact reporter API").
            spawn=sim_utils.UsdFileCfg(usd_path=str(DUMMY_USD), activate_contact_sensors=True),
            # InitialStateCfg defaults joint_pos/joint_vel to {".*": 0.0} —
            # a regex meant to match every joint. With zero joints, PhysX's
            # articulation init calls resolve_matching_names_values(strict=True)
            # against an empty joint list, and ".*" matching nothing is
            # treated as an unmatched pattern (ValueError), not a no-op.
            # An empty dict has nothing to match and passes vacuously.
            #
            # This pose only matters for the instant between spawn and the
            # first reset — mdp.events.reset_dummy_entry (mode="reset", so it
            # also runs at env creation) overwrites it immediately after, to a
            # waiting point it derives from the robot's own reach and the SSM
            # stop band (computed below, only available once the robot asset
            # exists — so it cannot be folded into this literal). Placed well
            # off to the side purely so nothing overlaps the robot during the
            # single frame before that first reset event fires.
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(20.0, 20.0, 0.0), joint_pos={}, joint_vel={}
            ),
            # ArticulationCfg.actuators has no default (MISSING) and cfg
            # validation rejects it unset. The dummy has zero real joints —
            # all 16 are fixed — so there is nothing to actuate; an empty
            # dict is the correct value, not a placeholder for one to fill in.
            actuators={},
        )
        # This wires up the sensor itself — it does NOT map reported bodies to
        # BR codes. ContactSensor.body_names is a prim path leaf name (e.g.
        # "BR_08_left"), not the BR-08 code the schema uses, and for paired
        # regions the two prims don't even share a name. The actual BR code
        # lives in the keti:bodyRegion USD attribute on each prim; reading it
        # is the recorder's work (recorder.read_region_codes), not this
        # config's. See generation/dummy/check_contact_resolution.py for the
        # precedent on how to read it.
        # track_contact_points=True requires a non-empty
        # filter_prim_paths_expr (confirmed: ContactSensor raises
        # otherwise). Filtering against the robot's 21 bodies (glob
        # "{ENV_REGEX_NS}/Robot/.*") then hit a lower-level PhysX tensor-view
        # rejection — "Filter pattern ... did not match the correct number of
        # entries (expected 17, found 21)" — which reads like this Isaac Lab
        # version's filtered rigid-contact view wants a 1:1 shape pairing
        # with the sensor bodies, not an arbitrary M:N set. Per-contact-point
        # geometry also isn't what this dataset needs: aggregate_contact_by_region()
        # in mdp/safety.py takes one scalar magnitude per prim, which
        # is exactly the plain net_forces_w a contact sensor already reports
        # without track_contact_points — the same shape the robot's own
        # `contact_forces` sensor in velocity_env_cfg.py uses (history_length
        # only, no track_contact_points, no filter). Matching that pattern
        # here instead of chasing the tensor-view constraint.
        self.scene.dummy_contact = PhysXContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Dummy/.*",
            history_length=3,
        )

        # observation.imu (schema §6) is a 10-vector: orientation quaternion,
        # body-frame angular velocity, body-frame proper linear acceleration.
        # The Imu sensor supplies the last two; the quaternion comes from the
        # same link's pose in recorder.step(), so both halves of the field
        # describe one frame rather than two.
        self.scene.robot_imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + ROBOT_ROOT_LINK,
        )

        # The asset only declares that self-collision has to go
        # (keti:requiresSelfCollisionDisabled); turning it off is the scene's
        # job. Adjacent regions touch by construction, and left on they
        # saturate the per-region contact array on every single step.
        #
        # `isaaclab.sim.ArticulationRootPropertiesCfg` still resolves (a
        # forwarding shim keeps the name alive) but is a deprecated subclass
        # as of this Isaac Lab version: `enabled_self_collisions` moved off
        # the solver-common base onto the PhysX-specific class because the
        # field is dual-namespace (physxArticulation:enabledSelfCollisions
        # today, a future newton:* sibling later). physx is the pinned
        # backend for this env, so import the concrete class instead of the
        # alias.
        self.scene.dummy.spawn.articulation_props = PhysxArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        )

        from .mdp.commands import S1aStraightWalkCommandCfg

        # Replaces the inherited UniformVelocityCommandCfg. mdp.generated_commands
        # reads term.command for the "velocity_commands" observation, so the shape
        # this term produces — (num_envs, 3), [lin_vel_x, lin_vel_y, ang_vel_z] in
        # the robot base frame — has to match it exactly; the policy's observation
        # layout is otherwise silently misaligned. See S1aStraightWalkCommand.command.
        #
        # Provisional numbers: each is used only while meta/rules.json has no
        # value for that name (fill_missing_only).
        # S1aStraightWalkCommand.provisional says whether any was used, and
        # EpisodeRecorder writes that flag to the episode metadata.
        self.commands.base_velocity = S1aStraightWalkCommandCfg(
            rules_path=str(REPO_ROOT / "meta/rules.json"),
            provisional_ssm_params={
                "human_speed": 1.6,
                "robot_reaction_time": 0.1,
                "robot_stop_time": 0.5,
                "intrusion_distance": 0.2,
                "position_uncertainty_robot": 0.05,
                "position_uncertainty_human": 0.05,
            },
        )

        # The inherited reset_base event (LocomotionVelocityRoughEnvCfg.EventsCfg)
        # spawns the robot facing a uniformly random heading (yaw in
        # [-3.14, 3.14] rad) with a random initial yaw-rate kick on top. Left
        # alone, S1aStraightWalkCommand's heading correction (mdp/commands.py)
        # would spend the episode's first several steps turning
        # a large random spawn error back toward 0 rather than holding a
        # clean straight line from the start — the human entry geometry
        # (mdp/events.py) places the dummy relative to the fixed world
        # direction the correction targets (+x), and the catalogue's own
        # robot_task description ("직선 경로", scenario-catalog-v0.1.md §4.3)
        # implies one well-defined path per episode, not a turning transient.
        # Heading is not in §4.5's list of eight randomized axes either.
        # Pinning yaw (pose and initial rate) to zero removes that startup
        # transient; every other reset_base axis (x/y spawn offset, linear
        # velocity kick, roll/pitch rate kick) is untouched.
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["yaw"] = (0.0, 0.0)

        # Two more disturbances from the same inherited EventsCfg, disabled
        # for the same reason: neither the trained
        # policy nor S1aStraightWalkCommand's heading correction has any
        # *lateral-position* feedback (only a heading loop), so a lateral
        # velocity kick becomes a permanent lateral offset instead of
        # something that gets corrected out — exactly the kind of unforced
        # drift entry geometry (mdp/events.py) is trying to budget a margin
        # against. Both exist upstream for locomotion *training* robustness
        # (a policy that never gets shoved learns a more brittle gait) and
        # neither is one of catalogue §4.5's eight S1a-specific randomized
        # axes — the same precedent already used for yaw above.
        #
        # - reset_base's own y velocity_range: leave x/z linear and
        #   roll/pitch angular kicks alone (yaw's is already zero above);
        #   only lateral (y) directly produces the uncorrected drift.
        # - push_robot: a periodic (10-15 s interval) lateral velocity kick
        #   during the episode, not just at reset — same failure mode, and
        #   with no position feedback to undo it, worse (it can land at any
        #   point in a 20 s episode, including right as the human enters).
        self.events.reset_base.params["velocity_range"]["y"] = (0.0, 0.0)
        self.events.push_robot = None

        # Physics randomisation moves from mode="startup" to mode="reset".
        #
        # Inherited from LocomotionVelocityRoughEnvCfg, both of these were
        # drawn once when the scene was built, so every episode of a run
        # shared one robot and the catalogue's §6 "전 시나리오 공통" physics
        # randomisation did not actually vary anything within a dataset. It
        # was also the reason the construction seed leaked into the data: seed
        # it from the first episode number and a batch starting at episode 2
        # built a *different* robot than the same episode inside a batch
        # starting at 0 (measured: base mass 16.596 kg vs 15.167 kg).
        #
        # Drawing them at reset makes both a function of the episode's own
        # seed, which is what meta/scenarios.json's seed_rule implies. Neither
        # term accumulates across resets -- both randomise from the values
        # they cached on their first call, which Isaac Lab documents and
        # implements explicitly (randomize_rigid_body_mass: "the randomization
        # is applied on the default values and not the previously randomized
        # values"; randomize_rigid_body_com: "randomizes from those defaults
        # on each call, ensuring repeatable randomization across resets").
        #
        # physics_material stays at startup: its ranges are degenerate
        # (static 0.8-0.8, dynamic 0.6-0.6, restitution 0-0), so it draws a
        # constant and moving it would buy nothing. That friction is not
        # randomised at all is a separate, already-recorded gap (카탈로그 §6).
        self.events.add_base_mass.mode = "reset"
        # base_com is the other startup-mode randomisation in the inherited
        # config, but H1's own rough_env_cfg.py sets it to None (it also
        # retargets add_base_mass from "base" to "torso_link"), so on this
        # robot there is nothing to move. Guarded rather than assumed, so a
        # future robot that re-enables it does not quietly keep drawing its
        # CoM once per process. It arrives as a preset() node here --
        # __post_init__ runs before resolve_presets -- so the mode goes on the
        # branch resolution will pick (default = PhysX, what this scenario
        # pins).
        base_com = self.events.base_com
        if base_com is not None:
            target = getattr(base_com, "default", base_com)
            if target is not None:
                target.mode = "reset"

        from isaaclab.managers import EventTermCfg as EventTerm

        from .mdp.events import reset_dummy_entry
        from .mdp.safety import fill_missing_only, load_ssm_parameters, separation_required

        # How far off the robot's path the dummy has to wait so that merely
        # standing there never trips a protective stop — separation_required
        # at the robot's fastest walk speed, scaled by the same
        # stop_band_factor the command term ramps to zero at, at exactly the
        # stop band and not the wider slowdown radius (a waiting human
        # merely slowing the robot down is correct, unflagged SSM behavior,
        # not something to also design the corridor around). Read from the
        # command cfg just built above, not a second literal — a rules_path
        # re-typed here could silently diverge from the one the command term
        # itself reads. Both bodies' own reach past their roots are added
        # at reset time (mdp/events.py._lateral_reach) since they need the
        # live assets, not just the config.
        ssm_params, ssm_missing = load_ssm_parameters(self.commands.base_velocity.rules_path)
        if ssm_missing:
            ssm_params = fill_missing_only(
                ssm_params, self.commands.base_velocity.provisional_ssm_params, ssm_missing)
        stop_band_clearance = separation_required(
            ssm_params,
            human_speed=float(ssm_params["human_speed"]),
            robot_speed=float(self.commands.base_velocity.walk_speed_range[1]),
        ) * float(self.commands.base_velocity.stop_band_factor)

        # The robot's own spawn-position randomization (reset_base above,
        # untouched apart from yaw) — read back rather than repeated as a
        # second "0.5" literal that could silently drift out of sync with it.
        spawn_margin_x = max(abs(v) for v in self.events.reset_base.params["pose_range"]["x"])
        spawn_margin_y = max(abs(v) for v in self.events.reset_base.params["pose_range"]["y"])

        # Small, testing-justified — not an empirical margin of metres (the
        # 4.0 m an uncorrected heading needed). Measured under three
        # settings, each of which can move the residual either way: the
        # heading-correction gain at the upstream value (0.5, not 2.0), the
        # wait_x delay term at 1x trigger_distance_max (to remove step-0
        # triggers — see mdp/entry_geometry.py), and the two sources of
        # unforced lateral drift disabled just above (reset_base's y
        # velocity_range, push_robot). With the margin at 0.6 m the worst
        # pretrigger shortfall measured -0.094 m (walk_speed 1.15 m/s, angle
        # 120 deg, trigger_distance 1.8 m; -0.352 m without those three) —
        # smaller, but still not exactly 0. 0.8 m (below) leaves ~0.2 m of
        # margin over that measured worst case — re-run check_entry.py --force-*
        # with the same draw if the walking policy, episode length, or
        # heading-correction gain ever change again.
        residual_drift_margin = 0.8

        self.events.reset_dummy_entry = EventTerm(
            func=reset_dummy_entry,
            mode="reset",
            params={
                "speed_range": (0.4, 1.6),              # catalogue §4.5 human_entry_speed
                "angle_range_deg": (30.0, 150.0),       # catalogue §4.5 human_entry_angle_deg
                "trigger_distance_range": (1.0, 4.0),   # catalogue §4.5 entry_trigger_distance
                "stop_band_clearance": stop_band_clearance,
                "spawn_margin_x": spawn_margin_x,
                "spawn_margin_y": spawn_margin_y,
                # min_distance_human reports a surface distance — read from
                # the command cfg, the same source _measure_min_distance
                # itself uses, not a second literal.
                "human_region_radius": self.commands.base_velocity.human_region_radius,
                "residual_drift_margin": residual_drift_margin,
            },
        )

        # No dummy_foot_friction EventTerm here (measured 2026-08-20). One
        # was tried, lowering BR_12_left/right (the two ground-contact regions) from
        # this scene's inherited 1.0 static+dynamic friction to 0.4, to test
        # whether the two feet's differing vertical load was turning into an
        # asymmetric friction reaction and a net yaw moment during tracking.
        #
        # What it bought: in the single window it was tuned against (seed
        # 2), worst-case tracking yaw fell from a 29.4deg baseline to
        # 13.7deg, clearing an ad hoc 20deg gate in check_dummy_hold.py.
        #
        # What it cost: it did not survive a three-seed sweep. With the
        # velocity feedforward of _advance_dummy in place,
        # mu=1.0 (i.e. this term absent) gives yaw
        # 37.4 / 30.9 / 39.0deg at seeds 0/1/3 — a spread that *brackets*
        # the 29.4deg baseline mu=0.4 was declared better than, not
        # consistently above it. mu=0.4 itself does no better: it fails the
        # same 20deg gate at seed 3 (27.9deg) and at check_dummy_hold.py's
        # own default arguments (20.6deg). What *is* robust across all
        # three seeds is the cost side: lowering the feet's friction shifts
        # more of the standing load from the ground onto the PD-hold servo,
        # raising servo (hold) force 2-3x and the airborne-step fraction by
        # 11-46 percentage points relative to mu=1.0 — worsening the very
        # confound (a servo-dominated, not contact-dominated, force
        # reading; see generation/README.md's "접촉력 — 지금 쓸 수 있는
        # 것과 없는 것" section) the data has to disclose rather than
        # paper over with a single-seed tuning win.
        #
        # Do not re-add this without a seed sweep that holds up — a change
        # that only helps in the one window it was fitted to is not a fix.
        # The dummy's feet are back on the scene's inherited material (1.0
        # static+dynamic friction, from velocity_env_cfg.py's terrain
        # physics_material, the scene-wide fallback for any body — the
        # dummy included — that does not set its own).

        # The inherited base_contact termination
        # (LocomotionVelocityRoughEnvCfg.TerminationsCfg, body_names set to
        # ".*torso_link" by H1RoughEnvCfg) is mdp.illegal_contact on the
        # robot's own contact_forces sensor — it does not distinguish "torso
        # hit the ground" from "torso touched the dummy". For a locomotion
        # *training* env that distinction never mattered (nothing else was
        # ever in the scene to touch); for S1a it is exactly backwards.
        # Robot-human contact is the event this dataset exists to record —
        # ending the episode on first contact would cut off the reaction
        # (and the contact force itself) at the one moment safety.event
        # needs it, and the "1-step episode right after a fall" pattern
        # traces to precisely this: torso
        # touches ground -> base_contact fires -> reset -> the next episode
        # can start already in (near-)contact -> fires again next step.
        #
        # Falls still have to end the episode (there is nothing useful left
        # to record once the robot is on the ground), so base_contact is
        # replaced rather than just dropped — with a termination keyed on
        # the robot's own pose, not on what it touched. That also lines up
        # with this project's own fall rule (meta/rules.json
        # R-25785-STAB-01: "stability_margin < 0 이 min_duration 이상 지속 →
        # 전도" — stability, not contact, is what the schema calls a fall).
        # That exact rule cannot be wired in as a termination yet — min_duration
        # is still null and safety.stability_margin needs a
        # support polygon from ground-contact points, which is the recorder's
        # instrumentation, not this config's. Root height + upright angle are a
        # deliberately cruder proxy for the same idea (a robot standing on
        # its feet has a high root and a small tilt; a robot down on the
        # ground has neither) — cheap, contact-independent, and the two
        # existing upstream termination functions used for exactly this
        # purpose elsewhere in isaaclab_tasks (root_height_below_minimum:
        # classic/humanoid/humanoid_env_cfg.py's "torso_height" term;
        # bad_orientation: locomotion/velocity/config/digit/rough_env_cfg.py's
        # "base_orientation" term, at limit_angle=0.7) rather than a new
        # hand-rolled check.
        #
        # Thresholds measured, not guessed — generation/envs/s1a/
        # check_fall_thresholds.py (committed) is the measurement: real
        # published policy, 6 episodes across 2 seeds, base_contact disabled
        # so falls play out instead of being cut off at the old termination.
        #   - Normal walking (including SSM slowdowns/stops and entries that
        #     never reach a fall) never dropped root height below 0.873 m or
        #     raised the upright angle above 13.23 deg, across every episode
        #     that did not fall.
        #   - The one draw that did fall collapsed to 0.070 m and 86.29 deg
        #     within the episode and stayed there (nothing to recover to
        #     once actually on the ground).
        # minimum_height=0.6 sits ~0.27 m below the worst normal reading and
        # ~0.53 m above the collapsed reading; limit_angle=0.5 rad (28.65 deg)
        # sits ~15 deg above the worst normal reading and ~58 deg below the
        # collapsed one — both catch a fall well before the robot is fully
        # flat, with a wide margin against tripping on ordinary gait or a
        # protective stop. Numbers will drift a little run to run (GPU
        # simulation, not bit-exact) — re-run check_fall_thresholds.py rather
        # than trust these as permanently exact. See check_fall_termination.py
        # for the "contact does not end the episode, a fall still does"
        # demonstration built on top of these thresholds.
        self.terminations.base_contact = None
        self.terminations.fall_height = DoneTerm(
            func=base_mdp.root_height_below_minimum,
            params={"minimum_height": 0.6},
        )
        self.terminations.fall_orientation = DoneTerm(
            func=base_mdp.bad_orientation,
            params={"limit_angle": 0.5},
        )
