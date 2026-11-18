"""S1d: S1a's entry scenario inside the self-built logistics aisle, with the
skinned dummy and the cameras meta/cameras.json declares (seven),
each placed by its type. The physics is S1a's but for three settings: scene entities
are added (the skinned dummy is engine-identical to v1 — generation/dummy/check_skin_physics.py),
the dummy is stood up before reset_dummy_entry measures its reach (dummy_reset), an aisle
guard runs at every reset, and without the skin the v1 dummy's regions get the class
labels the cameras need (dummy_labels). Control arms showed the
recorded physics identical with and without each of these. The first setting that differs is
the dummy's facing: it stands turned to its own walk (face_travel, 2026-10-01), a facing per
episode derived from the drawn entry angle, where S1a's shows the robot its back -- so contacts
land on the side the robot meets. And once across, the dummy stops at the waiting offset mirrored to
the far side (2026-10-01) instead of walking on into the far rack. And while the servo holds the
dummy it carries it HOLD_CLEARANCE_M above its standing height (dummy_hold_clearance_m, 2026-10-02)
instead of dragging it over the floor, which also changes the inertia and gravity terms of the
hold (mdp/commands.py, _advance_dummy)."""
from __future__ import annotations

from pathlib import Path

import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from generation import cameras as cam_canon
from generation.envs.s1a.s1a_env_cfg import S1aEnvCfg
from generation.envs.s1d import aisle_guard, dummy_labels, dummy_reset
from generation.scene import layout
# The aisle spec: layout's, which camera_io reads too (the fixed camera's recorded pose). Kept importable
# from here -- check_s1d_scene takes it from this module.
from generation.scene.layout import SCENE_SPEC

REPO = Path(__file__).resolve().parents[3]          # absolute: a relative USD path loses its textures
SCENE_USD = REPO / "generation/scene/build/keti_logistics_aisle_v1.usd"
SKIN_USD = REPO / "generation/dummy/build/keti_dummy_ts15066_v1_skin.usd"
# The jointed dummy (use_jointed): joints, feet and the gait table inside the asset (build_dummy.py --jointed --gait).
V2_USD = REPO / "generation/dummy/build/keti_dummy_ts15066_v2_jointed.usd"
V2_SKIN_USD = REPO / "generation/dummy/build/keti_dummy_ts15066_v2_jointed_skin.usd"
# The jointed dummy in the skin and the high-visibility vest (build_dummy.py --jointed --gait --skin --vest): the human
# of the scenarios with a conditions table (meta/assets.json keti_dummy_ts15066_v2_vest; generation/envs/hri/).
V2_VEST_USD = REPO / "generation/dummy/build/keti_dummy_ts15066_v2_jointed_skin_vest.usd"
GAIT_SPEC = REPO / "generation/dummy/gait/gait_spec.yaml"
ANTHROPOMETRY = REPO / "generation/dummy/anthropometry.yaml"
CAMERAS = REPO / "meta/cameras.json"
# How far above its standing height the hold servo carries the dummy (2026-10-02). Small:
# it is the drop when the hold is released. The servo holds the height within a millimetre.
HOLD_CLEARANCE_M = 0.005
# The share of the jointed dummy's weight the hold servo carries while it holds it (on its legs alone the
# dummy tips off them): the feet carry the rest. A controller setting of this
# project, from the walking experiments: at 0.3 and 0.5 every walk leaned past 10 degrees; at 0.7 the walks at
# 1.0 and 1.6 m/s stayed under 10 degrees until their stop.
HOLD_SUPPORT_SHARE = 0.7
# The hold's natural frequency across the plane for the jointed dummy (Hz): lower than the hold's own 2 Hz so the
# walking legs move the body more and the hold less -- at 1 Hz the median horizontal assist fell to 0.06-0.16 of
# the body's weight from 0.12-0.24 at 2 Hz (1.0-1.6 m/s), the other measures much as they were.
HOLD_PLANE_FREQ_HZ = 1.0
ROBOT_CLASS = "robot"           # the robot's semantic class; check_s1d_scene counts its pixels by it


def camera_cfg(name: str, cam: dict, spec: dict, scenario_id: str, approach_side: str | None = None) -> CameraCfg:
    """One canon camera as the scene entity that renders it, placed by its type -- the follow and fixed ones at
    ``scenario_id``'s site (the canon's ``sites``; a scenario without a placement is refused before anything is
    spawned), of ``approach_side`` where the site holds a placement per side the person comes from (refused, naming
    the camera, without one):

    - mounted: a child of its parent link, at the canon offset (a ROS optical frame, xyzw).
    - follow: a world prim the run loop moves onto the robot before every step (camera_io.Rig.follow).
    - fixed: a world prim under the env namespace, at cameras.fixed_pose of the site's placement (the aisle spec
      for the rack-face rule) -- the env frame: the aisle is spawned at the env origin too. The rig records the
      same pose (camera_io).

    Every camera's pose is recorded every frame (2026-09-23). A mounted one's is derived from its link
    (cameras.mounted_pose), not read back: for a camera under an articulation link the engine keeps
    reporting the spawn pose while the render follows the link (measured 2026-09-23), so
    update_latest_camera_pose stays off there. A follow camera is moved by the loop itself and its engine
    pose is the rendered one, so it is on there. A fixed camera's pose is derived once. Rendering does not
    depend on the flag (the RTX renderer's update_camera is a no-op)."""
    kw = dict(spawn=sim_utils.PinholeCameraCfg(focal_length=cam_canon.focal_length_mm(cam["fov_h_deg"]),
                                               horizontal_aperture=cam_canon.HORIZONTAL_APERTURE_MM,
                                               clipping_range=tuple(cam["clipping_range_m"])),
              data_types=list(cam["channels"]), width=cam["width"], height=cam["height"])
    if "semantic_segmentation" in cam["channels"]:
        kw["colorize_semantic_segmentation"] = False        # int32 ids, decoded through idToLabels
    kind = cam["type"]
    if kind == "mounted":
        return CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/" + cam["parent_prim"] + "/" + name,
                         offset=CameraCfg.OffsetCfg(pos=tuple(cam["offset_pos"]), rot=tuple(cam["offset_rot_xyzw"]),
                                                    convention=cam["convention"]), **kw)
    if kind in ("follow", "fixed"):
        # refused here, naming the camera, rather than by the rig later
        cam_canon.site_placement(cam, scenario_id, name=name, approach_side=approach_side)
    if kind == "follow":
        return CameraCfg(prim_path="{ENV_REGEX_NS}/" + name, update_latest_camera_pose=True, **kw)
    if kind == "fixed":
        pos, rot = cam_canon.fixed_pose(cam, spec, scenario_id, approach_side=approach_side)
        return CameraCfg(prim_path="{ENV_REGEX_NS}/" + name,
                         offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in pos), rot=tuple(float(v) for v in rot),
                                                    convention="ros"), **kw)
    raise ValueError(f"{name}: no scene entity for a camera of type {kind!r}")


@configclass
class S1dEnvCfg(S1aEnvCfg):
    use_cameras: bool = True
    use_skin: bool = True
    # The scenario whose site places the third-person and CCTV cameras (meta/cameras.json sites).
    scenario_id: str = "S1d"
    # The side the person comes from (the conditions table's approach_side), where the site holds a placement of the
    # third-person and CCTV cameras per side (meta/cameras.json by_approach_side); None for a site with one.
    camera_approach_side: str | None = None
    # The jointed dummy walking its gait (generation/dummy/jointed.py); off, the rigid v1 dummy.
    use_jointed: bool = False

    def __post_init__(self):
        super().__post_init__()
        spec = layout.load_spec(SCENE_SPEC)
        anthropometry = yaml.safe_load(ANTHROPOMETRY.read_text(encoding="utf-8"))
        # The dummy is stood up before its reach is measured: an auto-reset otherwise measures a
        # fallen dummy and inflates the waiting offset the guard below checks (dummy_reset docstring).
        self.events.reset_dummy_entry.func = dummy_reset.stand_up_first(self.events.reset_dummy_entry.func)
        # The dummy faces the way it walks (2026-10-01): facing the robot at a fixed 180 deg
        # (2026-09-29) it crossed the aisle sideways. The facing is each episode's entry angle plus a half
        # turn (entry_geometry.travel_facing_yaw_deg) -- set here only, S1a keeps 0 -- and the park and
        # the hold servo read it back per env. It moves which regions contacts land on.
        self.events.reset_dummy_entry.params["face_travel"] = True
        # The walk stops at the far side's waiting offset, mirrored across the robot's lane (2026-10-01):
        # S1a's dummy walks on into empty space, S1d's walked into the far rack with its
        # hold servo still on -- 7 of the 20-episode slice ran it through that contact for 83-432 steps.
        self.events.reset_dummy_entry.params["stop_at_far_side"] = True
        # The hold servo carries the dummy (2026-10-02): standing on its feet while it was
        # pulled at its centre of mass, friction at the feet turned it 12-24 deg about the vertical in the
        # entries from about 1 m/s. The standing height is the spec's, not a number typed here.
        self.commands.base_velocity.dummy_hold_clearance_m = HOLD_CLEARANCE_M
        self.commands.base_velocity.dummy_hold_stand_height_m = aisle_guard.standing_root_height(anthropometry)
        # wait_y is only known once reset_dummy_entry has measured both bodies' reach, so the
        # guard is a reset term defined after it (terms run in definition order).
        self.events.aisle_guard = EventTerm(
            func=aisle_guard.check_after_reset, mode="reset",
            params={"rack_inner_y": layout.rack_inner_face_y(spec),
                    "margin": spec["aisle"]["guard_margin_m"],
                    # the width at any turn: the dummy stands at a different yaw every episode
                    "dummy_half_width": aisle_guard.dummy_horizontal_radius(anthropometry)})
        self.scene.aisle = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Aisle",
                                        spawn=sim_utils.UsdFileCfg(usd_path=str(SCENE_USD)))
        # The aisle is an enclosed hall that brings its own lights (fixtures and a shadowless
        # fill, build_logistics_scene.py). The inherited sky dome would add a light from outside
        # and read an HDRI from Nucleus the release does not account for (inventory §5 risk 2).
        # A light is not physics: the control arms' arrays do not depend on it.
        self.scene.sky_light = None
        if self.use_skin:
            self.scene.dummy.spawn.usd_path = str(SKIN_USD)
        elif self.use_cameras:
            # v1 has no semantic labels; the capsule arm still needs its dummy in the mask and box.
            from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
            from isaaclab.sim.utils.prims import clone

            self.scene.dummy.spawn.func = clone(dummy_labels.labelled(spawn_from_usd.__wrapped__))
        if self.use_jointed:
            self._use_jointed_dummy()
        if self.use_cameras:
            cams = cam_canon.load(CAMERAS)
            # Labels have to be on the asset before spawn: labelling after the physics views
            # exist invalidates them. One tag on the root reaches every robot link.
            self.scene.robot.spawn.semantic_tags = [("class", ROBOT_CLASS)]
            # Every camera of the canon, in its order, under the canon's name (camera_io.Rig reads them
            # back by those names).
            for name in cam_canon.camera_names(cams):
                setattr(self.scene, name, camera_cfg(name, cams["cameras"][name], spec, self.scenario_id,
                                                     approach_side=self.camera_approach_side))

    def _use_jointed_dummy(self) -> None:
        """The jointed dummy (v2): its asset, one implicit actuator per joint group at the walking gains, the
        gait on, a share of its weight carried toward its stand height with the feet (not the carried hold), the
        inertia of the body above the hips, and the aisle guard's width with the feet."""
        from isaaclab.actuators import ImplicitActuatorCfg

        from generation.dummy import joints_spec as js

        spec = js.load()
        names = js.dof_names(spec)
        by_group = {}
        for j in spec.joints:
            start = js.columns(spec)[j.name].start
            by_group.setdefault(j.group, []).extend(names[start:start + len(j.dofs)])
        self.scene.dummy.spawn.usd_path = str(V2_SKIN_USD if self.use_skin else V2_USD)
        self.scene.dummy.init_state.joint_pos = {".*": 0.0}
        self.scene.dummy.init_state.joint_vel = {".*": 0.0}
        self.scene.dummy.actuators = {
            g: ImplicitActuatorCfg(joint_names_expr=[f"^{n}$" for n in dofs],
                                   stiffness=spec.drives["walking"][g][0], damping=spec.drives["walking"][g][1])
            for g, dofs in by_group.items()}
        gains = {mode: {n: spec.drives[mode][g] for g, dofs in by_group.items() for n in dofs}
                 for mode in ("walking", "tone")}
        ramp = yaml.safe_load(GAIT_SPEC.read_text(encoding="utf-8"))["ramp"]
        cmd = self.commands.base_velocity
        cmd.dummy_hold_clearance_m = None
        cmd.dummy_hold_support_share = HOLD_SUPPORT_SHARE
        cmd.dummy_hold_plane_freq_hz = HOLD_PLANE_FREQ_HZ
        cmd.dummy_hold_stand_height_m = aisle_guard.standing_root_height(js.geometry_with_feet(spec))
        cmd.dummy_gait = True
        cmd.dummy_gait_accel_time_s = float(ramp["accel_time_s"])
        cmd.dummy_gait_decel_time_s = float(ramp["decel_time_s"])
        cmd.dummy_hold_inertia_model = "upper_body_hips"
        cmd.dummy_upper_body_prims = tuple(js.upper_body_prims(spec))
        cmd.dummy_region_prims = tuple(js.region_prims(spec))
        cmd.dummy_hip_above_root_m = js.hip_above_root(spec)
        cmd.dummy_walk_gains = gains["walking"]
        cmd.dummy_tone_gains = gains["tone"]
        self.events.aisle_guard.params["dummy_half_width"] = aisle_guard.dummy_horizontal_radius(
            js.geometry_with_feet(spec))
