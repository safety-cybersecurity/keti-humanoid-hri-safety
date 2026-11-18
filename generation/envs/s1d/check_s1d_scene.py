#!/usr/bin/env python3
"""Does S1dEnvCfg build the scene it says, and do its cameras record what they claim? (~/isaacsim/python.sh)

    ~/isaacsim/python.sh generation/envs/s1d/check_s1d_scene.py
    ~/isaacsim/python.sh generation/envs/s1d/check_s1d_scene.py --cameras-only
    ~/isaacsim/python.sh generation/envs/s1d/check_s1d_scene.py --narrow-guard

Reads the running env, not the config. The cameras are driven and read the way a recording does it,
through camera_io.Rig (the follow cameras moved before each step, every camera warmed after the reset,
all of them read after each step), so the frames and poses checked are the ones a recording stores.
Every camera meta/cameras.json declares is checked, by its type:

(a) the aisle is spawned with as many Cube/RectLight prims as the layout has boxes, and every
    visible light on the stage is the aisle's own: no light outside it (the sky dome S1dEnvCfg
    drops), one dome (the aisle's fill), every layout light visible, none reading a texture file
(b) each camera's outputs have the canon's width and height and, per canon channel, that channel's
    depth (CHANNEL_DEPTH: rgb 3, semantic_segmentation 1); and none of a camera's recorded frames over
    the camera steps is black -- every pixel 0: what a camera buried in its own link records (a wrist
    camera placed by the URDF mesh did, on 2026-09-30)
(c) with the published policy driving, the robot keeps clear of the front racks: the
    minimum over steps of (rack inner face - largest |y| of any robot body) must stay above
    the spec's guard margin
(e) the spawned scene and dummy USDs are the files meta/assets.json pins (SHA-256), and the
    dummy's runtime total mass is v1's (87.6038 kg, measured 2026-09-23 --
    check_skin_physics.py is the real guard; this is the spawned file saying the same)
(f) every camera's intrinsics as cameras.intrinsics() derives them are the engine's intrinsic_matrices
(g) the pose a camera records puts the world where the renderer drew it. Collision-free probe spheres
    stand ahead of the robot and behind it (this check only, not in S1d); PROBES says which camera must see
    which. A link camera no probe covers -- the wrists, whose picture is floor and legs -- sees a
    marker on its own link instead (MARKER_CAM): a visual-only sphere that rides the link like the
    camera, so where it is drawn says where the camera renders from (every link camera is
    checked against the render). At every step where a target's outline -- the ellipse the sphere
    projects to through the recorded pose and the derived intrinsics -- lies whole inside a camera's
    picture, the centroid of its pixels must be within PX_TOL of that ellipse's centre: pixels from the
    segmentation on bbox.camera, from the target's colour elsewhere -- looked for in a window around that
    ellipse (roi), so that two targets of one colour can share a picture (a step where another one's
    outline reaches the window is not compared). A camera the probes check sees at least two of them,
    whose lines of sight are MIN_TARGET_SEP_DEG apart at half its steps or more: one target cannot see a
    recorded pose turned about the camera's line of sight to it -- the fixed camera, whose one probe sat
    still in its picture, passed a 30 deg turn at 0.2 px. The recorded pose is by type:
    mounted = its parent link composed with the canon offset (cameras.mounted_pose), fixed =
    cameras.fixed_pose plus the env origin, follow = the engine's -- which must also be the eye it was
    moved to, looking at its target. And between steps, every mounted camera's recorded pose must
    undergo its parent link's own world-frame rigid motion (the link read from the engine here, not
    through the Rig)
(h) the dummy faces the way it walks: parked
    at the drawn entry angle plus a half turn (entry_geometry.travel_facing_yaw_deg -- from the drawn
    angle, not from the cfg), and held there -- upright, not turning -- for as long as the hold servo
    holds it. Its front, as parked, points along the way it then actually moves while held. The servo
    once took its centre of mass as if the dummy faced +x; turned, that misplaces the CoM and the
    parasitic torque shows up as yaw drift and tilt. And the servo's per-axis inertia is this episode's:
    it equals the inertia recomputed from the live link positions, in this episode and in a second one
    reset at another entry angle. --entry-angle-deg pins the first episode's angle (the oblique ends are
    where a facing shows); --facing-yaw-deg runs the fixed-facing control.
    And the servo carries the dummy: once settled, its root stays at the height the spec says
    it stands at plus S1dEnvCfg's HOLD_CLEARANCE_M, and no link centre comes lower than it would standing
    plus that clearance -- its feet are off the floor. The inertia compared above is a free body's, about
    its centre of mass: a carried dummy does not tip about its feet
(i) bbox.camera, the camera the person box is computed from, does not look at the robot: over the
    first OWN_VIEW_STEPS steps, robot-class pixels stay under OWN_VIEW_MAX of its picture -- the add-on
    head module sits clear of its own head. The robot has to carry the class on the
    stage (S1dEnvCfg's ROBOT_CLASS), or its pixels would count as nothing
(j) a fixed camera sees the robot where it is aimed: a pelvis (the robot's spawn height) at either end
    of the mount's aim_x_range_m, on aim_y_m, projects inside its picture; and at every step with the
    robot's x (env frame) inside that range, the robot's own pelvis does. Geometry only (the recorded
    pose and the intrinsics); for the steps it keeps stepping past the camera checks, with (h), until
    the robot has passed the range
(d) --narrow-guard: with the rack face moved in to 3.0 m the aisle guard stops the first
    reset (the guard's revert experiment; run on its own because the env cannot be rebuilt
    inside one process)

--cameras-only runs (a)-(g), (i) and (j)'s two ends over --steps and stops there: (h) and (j)'s steps
need the steps after it and are not judged. Exit 0 only if everything asked for holds.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--steps", type=int, default=200, help="policy steps for (b), (c), (g) and (i)")
parser.add_argument("--narrow-guard", action="store_true", help="run (d) only")
parser.add_argument("--cameras-only", action="store_true",
                    help="stop after --steps: (h) and (j) are not judged")
parser.add_argument("--hold-steps", type=int, default=1000,
                    help="(h), (j) keep stepping after the camera checks until the hold ends and the robot has "
                         "passed every fixed camera's aim range, or the episode ends, or this many steps")
parser.add_argument("--facing-yaw-deg", type=float, default=None,
                    help="(h) control: park and hold the dummy at this fixed yaw instead of turning it to its walk")
parser.add_argument("--entry-angle-deg", type=float, default=None,
                    help="(h) draw the entry angle from this one value instead of the scenario's range")
parser.add_argument("--report", default=str(REPO / ".tmp/s1d_check/check_s1d_scene.json"))
args = parser.parse_args()

from generation.isaac_boot import launch_app  # noqa: E402

simulation_app = launch_app({"headless": True, "enable_cameras": True})[1]

V1_DUMMY_MASS_KG = 87.6038     # runtime body_mass sum of v1, measured 2026-09-23
# The scenario whose camera sites (meta/cameras.json sites) place the third-person and fixed cameras here: the
# env's cfg is built for it, and the camera rig refuses any other.
SITE = "S1d"
# label -> (centre in the env frame, radius, colour, the cameras that must see it). The colours are
# pairwise apart for the colour test (colour_mask): each camera that finds its target by colour finds
# nothing else there.
# - probe: ahead at eye height, in view of the level head camera, the third-person camera and the fixed
#   camera through the camera steps. 4.5-6.6 m from the head camera, where an error along its optical
#   axis moves the probe little -- hence the near one.
# - near_probe: off the head camera's axis and nearer (3.0 m ahead at the reset, 1.3 m when it leaves the
#   picture; 0.9 m aside, off the robot's path and clear of racks and floor): an axial pose error of 1 cm moves it up to 2.8 px (0.7 px at the
#   median step), where the far probe moves under 0.3 px -- the old depth comparison's job. Whole in the
#   picture for 140 of the 200 steps (2026-09-30 poses). Found through the segmentation, so its colour only
#   has to be none the others test for; on the +y side, away from the third-person and fixed cameras (-y),
#   so it cannot stand between them and the probe they test.
# - floor_probe: for the camera that looks down at 50.8 deg: on the robot's path, whole in that picture from
#   the reset (robot at x 0.41) to step 175 of 200 (the camera at x 2.6, 2026-09-30), and clear of the floor
#   -- a sphere cut by the floor has an outline the renderer does not draw.
# Every camera the probes check sees a second one, its line of sight 11-55 deg from the first's
# (placed from the camera poses a six-camera recording of this seed stored -- whole in the picture at
# the steps below, no outline reaching another target's, off the robot's lane and on the cameras' side of it):
# - floor_probe_2: the down-looking camera's second, 0.6 m to the side of the lane (0.45 m clear of it): whole
#   at 193 of 200 steps, 11-22 deg from floor_probe. Green like floor_probe: the windows keep them apart.
# - side_probe: the third-person and fixed cameras' second, low and 2 m toward them from the lane, so the
#   robot never stands between: whole at all 200 steps in both, 13-19 deg from probe in the third-person
#   camera and 29 deg in the fixed one, where probe sits near the picture's centre and a turn about the
#   optical axis (an image-up error) hardly moves it. Magenta like probe; the windows keep them apart.
# - rear_probe, rear_probe_2: for the camera on the back of the head, which looks straight back -- 1.5 and 2.5 m
#   behind it at the reset, one on the robot's left at 1.6 m and one on its right at 1.2 m, so that neither stands
#   behind the other. Worked out from the head camera's poses that recording stored, through the canon offsets: as
#   the robot walks away from them, both whole in that picture at all 200 steps, 15-32 deg apart, and in neither the
#   third-person nor the fixed camera's picture. Magenta like probe, which this camera never sees.
PROBES = {
    "probe": ((7.0, -0.6, 1.5), 0.15, (1.0, 0.0, 1.0), ("head_rgb", "third_person_rgb", "cctv_rgb")),
    "near_probe": ((3.5, 0.9, 1.3), 0.10, (1.0, 0.5, 0.0), ("head_rgb",)),
    "floor_probe": ((3.0, 0.0, 0.20), 0.15, (0.0, 1.0, 0.0), ("head_down_rgb",)),
    "floor_probe_2": ((3.2, 0.6, 0.20), 0.15, (0.0, 1.0, 0.0), ("head_down_rgb",)),
    "side_probe": ((4.0, -2.0, 0.30), 0.15, (1.0, 0.0, 1.0), ("third_person_rgb", "cctv_rgb")),
    "rear_probe": ((-1.1, 0.5, 1.6), 0.15, (1.0, 0.0, 1.0), ("rear_rgb",)),
    "rear_probe_2": ((-2.1, -0.6, 1.2), 0.15, (1.0, 0.0, 1.0), ("rear_rgb",)),
}
# A link camera no probe covers gets a marker on its own link: a sphere with no collider and no mass (the
# arm's physics is untouched) at MARKER_CAM in the camera's optical frame -- right, down, forward -- put on
# the link through the canon offset. 10 cm ahead, 2 cm up and aside: past the fist's tip (the simulated
# forearm mesh ends at link x 0.316, the marker spans 0.352-0.372), about 34 px in radius, and off the optical
# axis so that an offset along it moves the marker too (about 9 px per cm).
MARKER_CAM = (0.02, -0.02, 0.10)
MARKER_RADIUS = 0.01
MARKER_COLOUR = (0.0, 1.0, 1.0)   # cyan: none of the probes' colours
# Target pixels' centroid vs the outline's centre. Found by colour, the centroid moves with frame-to-frame
# render noise (DLSS): the worst frame of each of seven runs on 2026-09-30 was 0.43-0.84 px for the floor
# probe in head_down_rgb (mean 0.56, sd 0.14 -- mean + 3 sd is 0.99, so expect an occasional stray red there:
# run again; a pose error is red every run), up to 0.36 px for the third-person probe and 0.24 px for the
# fixed camera's, 0.49-0.51 px for the wrist markers; through the segmentation (head_rgb) 0.19-0.23 px. The
# errors it is for are far larger: a 1 cm offset of a wrist camera's render moves its marker about 8 px, a
# composition error ~150-190 px.
PX_TOL = 1.0
# An outline this close to the picture's edge is not compared: the drawn rim is blended over about a pixel
# (anti-aliasing, DLSS), and part of it past the edge would be lost and pull the centroid inward. A pixel for
# that blend and one to spare -- the colour masks otherwise match the outline, e.g. a wrist marker's
# 3662-3757 px against the ellipse's ~3750 (2026-09-30).
EDGE_MARGIN_PX = 2.0
MIN_PROBE_PX = 50              # fewer pixels than this is not the target (stray colour, a sliver)
# A target found by colour is looked for in a window around its outline: the outline's box grown by half its
# size and this many pixels. A pose error larger than that leaves the window short of pixels: missed, red.
ROI_PAD_PX = 8
# Two targets of one camera count as a pair where their lines of sight are this far apart: a 1 deg turn about
# either one's line of sight then moves the other about PX_TOL even in the widest-angle camera (the fixed one,
# fx 320 px: 320 * sin(10 deg) * pi / 180 = 0.97 px).
MIN_TARGET_SEP_DEG = 10.0
# A probe's lit channels are above this, its unlit ones under half of the dimmest lit one. Low enough to take
# the probe's shaded underside: at 120 (the two-camera check's value) the lights overhead left the bottom rim
# out, and the centroid sat above the outline's centre -- floor probe 0.56 px on average, 2.34 px at worst,
# third-person 0.32 px, along v only; at 60, 0.04 and 0.03 px (2026-09-30, 176 and 200 frames).
COLOUR_ON_MIN = 60
CHANNEL_DEPTH = {"rgb": 3, "semantic_segmentation": 1}
OWN_VIEW_STEPS = 20            # (i)
OWN_VIEW_MAX = 0.01
# (g) continuity: a mounted camera's step-to-step motion vs its link's. Both come from float32 poses (the
# recorded field, the engine's link pose); rounding near 10 m from the origin is ~1e-6 m.
CONTINUITY_TOL_M = 1e-5
CONTINUITY_TOL_RAD = 1e-5
PARK_YAW_TOL_DEG = 0.5         # (h) the pose reset_dummy_entry writes
TRAVEL_GAP_TOL_DEG = 2.0       # (h) the parked front vs the way the held dummy then moves
MIN_TRAVEL_M = 0.5             # (h) a shorter move does not say which way it went
# (h) the servo's inertia vs the live links'. Turning the dummy between the walk's two ends moves a tipping
# axis' inertia by a few parts in a thousand (measured below, reported): this is well inside that and well
# outside float32's ~1e-5.
INERTIA_REL_TOL = 5e-4
# (h) the carried hold: the root's height against the spec's standing height plus the clearance, and the
# lowest link centre against where it would be standing plus the clearance. The servo was measured to hold
# the height within 1 mm; the first second after the reset is its settling from the
# spawn height.
CARRY_TOL_M = 2e-3
CARRY_SETTLE_STEPS = 50
COM_GAP_TOL_M = 1e-3          # (h) the servo's push point vs the dummy's centre of mass: equal but for rounding
# Gross bounds only (the push-point gap is the precise check): the moving hold already lets the
# dummy turn ~22-29 deg and tilt ~8-9 deg at either facing, S1a's known behaviour.
HOLD_YAW_TOL_DEG = 45.0
HOLD_TILT_TOL_DEG = 15.0

code = 1
try:
    import importlib.metadata as metadata

    import numpy as np
    import torch
    from pxr import Usd
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    from isaaclab_physx.sim.schemas import CollisionPropertiesCfg, RigidBodyPropertiesCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    import isaaclab_tasks  # noqa: F401
    import generation.envs.register  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_tasks.utils.hydra import resolve_presets

    from generation import cameras
    from generation.assets import asset_entry, canonical_usd_path, load_assets
    from generation.envs.s1a.media_writer import decode_labels, is_black
    from generation.envs.s1d.camera_io import Rig
    from generation.envs.s1a.mdp.entry_geometry import relative_gap_max, travel_facing_yaw_deg
    import yaml

    from generation.envs.s1d import aisle_guard
    from generation.envs.s1d.s1d_env_cfg import ANTHROPOMETRY, CAMERAS, HOLD_CLEARANCE_M, ROBOT_CLASS, SCENE_SPEC, S1dEnvCfg
    from generation.scene import layout
    from generation.seeding import episode_seed, scenario_startup_seed

    problems: list[str] = []
    report: dict = {}
    canon = cameras.load(CAMERAS)
    spec = layout.load_spec(SCENE_SPEC)
    names = cameras.camera_names(canon)
    ccs = canon["cameras"]
    # The link cameras no probe covers, and where each one's marker sits on its link: MARKER_CAM through
    # the canon offset -- the offset the recorded pose is composed with.
    probed = {n for *_, cams_of in PROBES.values() for n in cams_of}
    marked = [n for n in names if ccs[n]["type"] == "mounted" and n not in probed]
    marker_on_link = {n: np.asarray(ccs[n]["offset_pos"], float) + cameras._q_rot(ccs[n]["offset_rot_xyzw"], MARKER_CAM)
                      for n in marked}

    # Two looks, each measured on 2026-09-30. A probe is lit, a plain diffuse colour: emitted, the green floor
    # probe's glow on the floor beside it joined its colour mask (300-400 px more) and pulled the centroid
    # 1.85 px on average, 3.10 px at worst, against 0.56 / 0.84 lit. A marker emits its colour: its picture
    # then does not change with how the swinging arm faces the lights -- worst 0.49-0.51 px emitted against
    # 0.59-0.84 px lit (its mask grew by ~50 px, not by a glow's hundreds).
    def lit(colour):
        return sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(colour))

    def emitted(colour):
        return sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), emissive_color=tuple(colour))

    cfg = resolve_presets(S1dEnvCfg(scenario_id=SITE), selected=("physx",))
    cfg.seed = scenario_startup_seed("S1d")
    # (h) The facing S1d promises is each episode's entry angle plus a half turn. It is worked out below
    # from the angle the reset drew -- not read back from the cfg it is meant to check: a cfg that dropped
    # the setting parks the dummy at 0 and fails on the park yaw.
    entry_params = cfg.events.reset_dummy_entry.params
    if args.entry_angle_deg is not None:
        entry_params["angle_range_deg"] = (args.entry_angle_deg, args.entry_angle_deg)
    if args.facing_yaw_deg is not None:
        entry_params["face_travel"] = False
        entry_params["facing_yaw_deg"] = args.facing_yaw_deg
    if args.narrow_guard:
        cfg.events.aisle_guard.params["rack_inner_y"] = 3.0
    else:
        for label, (centre, radius, colour, _) in PROBES.items():
            setattr(cfg.scene, label, RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + "".join(part.capitalize() for part in label.split("_")),
                spawn=sim_utils.SphereCfg(
                    radius=radius,
                    rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=CollisionPropertiesCfg(collision_enabled=False),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    visual_material=lit(colour),
                    semantic_tags=[("class", label)]),
                init_state=RigidObjectCfg.InitialStateCfg(pos=centre)))
        # Under the camera's own link, spawned after the robot (scene entities spawn in cfg order). A plain
        # asset with no rigid body, collider or mass: the link's mass and collision shapes stay as they are.
        for n in marked:
            setattr(cfg.scene, n + "_marker", AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + ccs[n]["parent_prim"] + "/" + n + "_marker",
                spawn=sim_utils.SphereCfg(radius=MARKER_RADIUS, visual_material=emitted(MARKER_COLOUR)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(float(v) for v in marker_on_link[n]))))

    def build_and_reset():
        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg))
        env.seed(episode_seed("S1d", 0))
        with torch.inference_mode():
            obs, _ = env.reset()
        return env, obs

    if args.narrow_guard:
        try:
            build_and_reset()
            problems.append("(d) the env reset with the rack face at 3.0 m — the aisle guard did not stop it")
        except ValueError as exc:
            if "aisle too narrow" not in str(exc):
                raise
            print(f"(d) guard stopped the reset: {exc}")
        for p in problems:
            print(f"PROBLEM {p}")
        print("OK" if not problems else f"{len(problems)} problem(s)")
        raise SystemExit(0 if not problems else 1)

    env, obs = build_and_reset()
    base = env.unwrapped
    stage = base.sim.stage if hasattr(base.sim, "stage") else sim_utils.get_current_stage()

    # (a) the aisle, from the stage
    aisle_root = stage.GetPrimAtPath("/World/envs/env_0/Aisle")
    boxes = layout.build_layout(spec)
    n_prims = sum(1 for p in Usd.PrimRange(aisle_root) if p.GetTypeName() in ("Cube", "RectLight")) if aisle_root else 0
    report["aisle_prims"] = n_prims
    if n_prims != len(boxes):
        problems.append(f"(a) aisle prims {n_prims} != layout boxes {len(boxes)}")
    # An invisible light does not light the render: the ground plane Isaac Lab spawns brings a
    # SphereLight it makes invisible itself (spawn_ground_plane), so only visible ones count.
    from pxr import UsdGeom
    every_light = [p for p in stage.Traverse() if p.GetTypeName().endswith("Light")]
    lights = [p for p in every_light if UsdGeom.Imageable(p).ComputeVisibility() != UsdGeom.Tokens.invisible]
    outside = [str(p.GetPath()) for p in lights if not p.GetPath().HasPrefix(aisle_root.GetPath())] if aisle_root else []
    domes = [str(p.GetPath()) for p in lights if p.GetTypeName() == "DomeLight"]
    textured = [str(p.GetPath()) for p in lights
                if any(a.GetName().endswith("texture:file") and a.HasAuthoredValue() for a in p.GetAttributes())]
    report["lights"] = {"visible": len(lights), "outside_the_aisle": outside, "domes": domes, "textured": textured,
                        "invisible": [str(p.GetPath()) for p in every_light if p not in lights]}
    if outside:
        problems.append(f"(a) lights outside the aisle {outside[:3]} — the hall brings its own (S1dEnvCfg drops the sky)")
    if len(domes) != 1 or textured:
        problems.append(f"(a) domes {domes}, textured lights {textured} — expected the aisle's one untextured fill")
    # ... and all of them: the fill alone passed the two lines above
    want_rect = sum(1 for b in boxes if b.material == "light")
    rects = sum(1 for p in lights if p.GetTypeName() == "RectLight")
    report["lights"]["rect_visible"] = rects
    if rects != want_rect:
        problems.append(f"(a) {rects} visible RectLights, the layout has {want_rect} — a fixture that lights nothing")

    robot = base.scene["robot"]
    dummy = base.scene["dummy"]

    def np_(v):
        t = getattr(v, "torch", v)
        return t.detach().cpu().numpy()

    origin = np_(base.scene.env_origins)[0].astype(float)

    # (e) the files that were spawned
    assets = load_assets(REPO)
    for key, path in (("aisle", cfg.scene.aisle.spawn.usd_path), ("dummy", cfg.scene.dummy.spawn.usd_path)):
        rec = asset_entry(assets, canonical_usd_path(path, ISAACLAB_NUCLEUS_DIR, REPO))   # raises on an unknown asset
        entry = assets["scene"][rec["key"]]
        got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        report[f"{key}_sha256"] = got
        if got != entry["usd_sha256"]:
            problems.append(f"(e) {key}: spawned {path} is {got[:12]}…, meta/assets.json pins {entry['usd_sha256'][:12]}…")
    mass = float(dummy.data.body_mass.torch[0].sum())
    report["dummy_total_mass_kg"] = mass
    if round(mass, 4) != V1_DUMMY_MASS_KG:
        problems.append(f"(e) dummy total mass {mass:.6f} kg != v1 {V1_DUMMY_MASS_KG} kg")

    # Per camera, everything below; printed one camera to a line.
    cams: dict[str, dict] = {name: {"type": ccs[name]["type"]} for name in names}
    report["cameras"] = cams

    # (f) intrinsics
    for name in names:
        k = np_(base.scene[name].data.intrinsic_matrices)[0]
        got = (float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))
        want = cameras.intrinsics(ccs[name])
        cams[name]["intrinsics"] = {"engine": got, "derived": want}
        if max(abs(g - w) for g, w in zip(got, want)) > 1e-3:
            problems.append(f"(f) {name}: engine fx,fy,cx,cy {got} != derived {want}")

    # (g) what each camera must see: the probes PROBES gives it, or -- a link camera no probe covers -- the
    # marker on its link. target: label -> (radius, colour); probe_of: camera -> its targets
    target = {label: (radius, colour) for label, (_, radius, colour, _) in PROBES.items()}
    target.update({n + "_marker": (MARKER_RADIUS, MARKER_COLOUR) for n in marked})
    probe_of: dict[str, list[str]] = {n: [n + "_marker"] for n in marked}
    for label, (*_, cams_of) in PROBES.items():
        for name in cams_of:
            if name not in ccs:
                problems.append(f"(g) PROBES has {label} seen by {name}, not a camera of meta/cameras.json")
            else:
                probe_of.setdefault(name, []).append(label)
    mounted = [n for n in names if ccs[n]["type"] == "mounted"]
    follow = [n for n in names if ccs[n]["type"] == "follow"]
    fixed = [n for n in names if ccs[n]["type"] == "fixed"]
    site = {n: cameras.site_placement(ccs[n], SITE, name=n) for n in follow + fixed}
    unchecked = [n for n in names if n not in probe_of]
    if unchecked:
        problems.append(f"(g) {unchecked}: no probe in PROBES and not on a link to carry a marker — nothing checks "
                        "where they look")
    probe_w = {label: np.asarray(centre, float) + origin for label, (centre, *_) in PROBES.items()}
    # The parent links, looked up here rather than through the Rig: the continuity check is the Rig's
    # recorded pose against the engine's link.
    link_idx: dict[str, int] = {}
    for n in mounted:
        ids, _ = robot.find_bodies(ccs[n]["parent_prim"])
        if len(ids) != 1:
            raise ValueError(f"{n}'s parent {ccs[n]['parent_prim']!r} matched {len(ids)} robot bodies")
        link_idx[n] = ids[0]
    box_cam = canon["bbox"]["camera"]

    # (i) the robot has to carry its class, or no pixel of it would count
    robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    robot_classes = [str(v).lower() for v in (sim_utils.get_labels(robot_prim).get("class", []) if robot_prim else [])]
    if ROBOT_CLASS not in robot_classes:
        problems.append(f"(i) the robot prim carries class labels {robot_classes}, not {ROBOT_CLASS!r} — its pixels "
                        f"in {box_cam} would count as nothing")

    agent_cfg = handle_deprecated_rsl_rl_cfg(
        load_cfg_from_registry("Keti-S1a-v0", "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Flat-H1-v0"))
    policy = runner.get_inference_policy(device=base.device)

    def conj(q) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]], float)

    def pose_of(record) -> tuple[np.ndarray, np.ndarray]:
        """A recorded pose (pos3 + quat4 wxyz) -> (pos, quat xyzw)."""
        r = np.asarray(record, float)
        return r[:3], np.array([r[4], r[5], r[6], r[3]])

    def motion(a, b) -> tuple[np.ndarray, np.ndarray]:
        """The world-frame rigid motion taking pose a to pose b (b composed with a's inverse) -- one and
        the same for every frame fixed to a body."""
        (pa, qa), (pb, qb) = a, b
        dq = cameras._q_mul(qb, conj(qa))
        return pb - cameras._q_rot(dq, pa), dq

    def angle(q) -> float:
        return 2.0 * math.atan2(float(np.linalg.norm(q[:3])), abs(float(q[3])))

    def labelled(labels: dict[int, str], name: str) -> list[int]:
        """The segmentation ids whose label -- of several comma-joined, any -- is name (case-insensitive:
        Replicator lower-cases labels, meta/cameras.json bbox.segmentation_note)."""
        return [sid for sid, lab in labels.items() if name in (p.strip().lower() for p in str(lab).split(","))]

    def colour_mask(rgb: np.ndarray, colour) -> np.ndarray:
        """A probe's pixels by colour: its lit channels above COLOUR_ON_MIN, the others under half of the
        dimmest lit one (the old magenta rule, R and B over G, for any primary or secondary colour)."""
        px = rgb.astype(np.int16)
        on = [i for i, c in enumerate(colour) if c > 0.5]
        off = [i for i, c in enumerate(colour) if c <= 0.5]
        lo = px[..., on].min(-1)
        return (lo > COLOUR_ON_MIN) & (px[..., off].max(-1) * 2 < lo)

    def centroid(mask) -> tuple[float, float]:
        rows, cols = np.nonzero(mask)
        return float(cols.mean() + 0.5), float(rows.mean() + 0.5)     # pixel centres; cx = W/2 convention

    def outline(centre_w, radius, pos, q, cc):
        """The ellipse a sphere of that radius projects to: (centre (u, v), extent (u_min, v_min, u_max, v_max)) in pixels, or
        None unless the whole sphere is in front of the camera. The ellipse's centre is not the projection
        of the sphere's centre off the optical axis -- 1-2 px for the floor probe 1.5 m from the camera
        that looks down at it -- so it is solved for: the rays touching the sphere are the cone
        x.(s s^T - (|s|^2 - r^2) I).x = 0 about its centre s in the optical frame, cut by the image plane."""
        fx, fy, cx, cy = cameras.intrinsics(cc)
        s = cameras._q_rot(conj(q), np.asarray(centre_w, float) - pos)
        d = float(np.linalg.norm(s))
        if d <= radius:
            return None
        axis = s / d
        e1 = np.cross(axis, (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        sin_a = radius / d
        th = np.linspace(0.0, 2.0 * math.pi, 256, endpoint=False)
        rays = (math.sqrt(1.0 - sin_a * sin_a) * axis[None]
                + sin_a * (np.cos(th)[:, None] * e1[None] + np.sin(th)[:, None] * e2[None]))
        if (rays[:, 2] <= 1e-9).any():
            return None
        u, v = fx * rays[:, 0] / rays[:, 2] + cx, fy * rays[:, 1] / rays[:, 2] + cy
        m = np.outer(s, s) - (d * d - radius * radius) * np.eye(3)
        c = -np.linalg.solve(m[:2, :2], m[:2, 2])
        return (fx * c[0] + cx, fy * c[1] + cy), (float(u.min()), float(v.min()), float(u.max()), float(v.max()))

    def visibility(centre_w: dict, poses: dict, n: str, lbl: str):
        """(in view, whole in the picture, outline) of target lbl in camera n: its centre this step through
        the camera's recorded pose."""
        cc = ccs[n]
        o = outline(centre_w[lbl], target[lbl][0], *poses[n], cc)
        if o is None:
            return False, False, None
        (u0, v0, u1, v1) = o[1]
        return (u1 > 0 and v1 > 0 and u0 < cc["width"] and v0 < cc["height"],
                u0 >= EDGE_MARGIN_PX and v0 >= EDGE_MARGIN_PX
                and u1 <= cc["width"] - EDGE_MARGIN_PX and v1 <= cc["height"] - EDGE_MARGIN_PX, o)

    def roi(o, cc) -> tuple[int, int, int, int]:
        """The window a target found by colour is looked for in: its outline's box grown by half its size and
        ROI_PAD_PX, clipped to the picture -- (u0, v0, u1, v1), u1 and v1 exclusive."""
        u0, v0, u1, v1 = o[1]
        pad = ROI_PAD_PX + 0.5 * max(u1 - u0, v1 - v0)
        return (max(0, math.floor(u0 - pad)), max(0, math.floor(v0 - pad)),
                min(int(cc["width"]), math.ceil(u1 + pad)), min(int(cc["height"]), math.ceil(v1 + pad)))

    def meets(win, extent) -> bool:
        """Does an outline's box (u_min, v_min, u_max, v_max) reach into the window?"""
        return not (extent[2] < win[0] or extent[0] >= win[2] or extent[3] < win[1] or extent[1] >= win[3])

    def angle_deg(a, b) -> float:
        a, b = np.asarray(a, float), np.asarray(b, float)
        return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))))))

    rack = layout.rack_inner_face_y(spec)
    margin = spec["aisle"]["guard_margin_m"]
    clear_body, clear_pelvis, dones = math.inf, math.inf, 0
    command = base.command_manager.get_term("base_velocity")

    def yaw_tilt_deg(q) -> tuple[float, float]:
        x, y, z, w = (float(v) for v in q)          # xyzw, the engine's order
        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        up_z = 1 - 2 * (x * x + y * y)               # the body's +z, seen in world z
        return yaw, math.degrees(math.acos(max(-1.0, min(1.0, up_z))))

    def wrap(deg: float) -> float:
        return (deg + 180.0) % 360.0 - 180.0

    def com_gap_m() -> float:
        """How far the servo's push point sits from the dummy's centre of mass (the same mass-weighted
        link origins the servo captures). The servo keeps that offset in the body frame and turns it
        with the dummy every step; captured in the wrong frame it is off by twice the CoM's x/y offset."""
        m, p = np_(dummy.data.body_mass)[0], np_(dummy.data.body_pos_w)[0]
        x, y, z, w = (float(v) for v in np_(dummy.data.root_quat_w)[0])
        u, v = np.array([x, y, z]), np_(command._dummy_com_offset_body)[0].astype(np.float64)
        tv = 2 * np.cross(u, v)
        push = np_(dummy.data.root_pos_w)[0] + v + w * tv + np.cross(u, tv)
        return float(np.linalg.norm((m[:, None] * p).sum(0) / m.sum() - push))

    def asset_standing_height(usd_path: str) -> float:
        """The root body's origin above the lowest point of any collision shape, in the asset's authored
        pose. A body's shape is its ``collider`` child (the skinned asset) or the body prim itself (v1)."""
        from pxr import UsdPhysics

        asset = Usd.Stage.Open(str(usd_path))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
        bodies = [p for p in asset.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        roots = [p for p in bodies if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(roots) != 1 or not bodies:
            raise ValueError(f"{usd_path}: {len(bodies)} rigid bodies, {len(roots)} articulation roots")
        root_z = UsdGeom.Xformable(roots[0]).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()[2]
        lows = []
        for body in bodies:
            shape = body.GetChild("collider") or body
            lows.append(cache.ComputeWorldBound(shape).ComputeAlignedRange().GetMin()[2])
        return float(root_z - min(lows))

    def axis_inertia() -> dict:
        """The servo's per-axis inertia against the same model worked out here from the live links: point
        masses about the world axes through the centre of mass. Read one step after a reset, while the dummy
        still stands as parked."""
        m = np_(dummy.data.body_mass)[0].astype(np.float64)
        r = np_(dummy.data.body_pos_w)[0].astype(np.float64)
        com = (m[:, None] * r).sum(0) / m.sum()
        r = r - com
        # no ground-pivot term m*h^2: the servo carries the dummy, which then tips about its own centre of mass
        live = [float((m * (r[:, 1] ** 2 + r[:, 2] ** 2)).sum()),
                float((m * (r[:, 0] ** 2 + r[:, 2] ** 2)).sum()),
                float((m * (r[:, 0] ** 2 + r[:, 1] ** 2)).sum())]
        servo = [float(v) for v in np_(command._dummy_moment_of_inertia)[0]]
        # relative_gap_max: infinite unless all six are positive finite numbers -- a NaN axis must not pass
        return {"entry_angle_deg": math.degrees(float(base.s1a_entry["angle"][0])), "servo_kgm2": servo,
                "live_kgm2": live, "rel_gap_max": relative_gap_max(servo, live)}

    def track_hold() -> None:
        yaw, tilt = yaw_tilt_deg(np_(dummy.data.root_quat_w)[0])
        hold["steps"] += 1
        if bool(base.s1a_entry["triggered"][0]):
            xy = np_(dummy.data.root_pos_w)[0, :2].astype(np.float64)
            hold.setdefault("moved_from_xy", xy)
            hold["moved_to_xy"] = xy
        if hold["steps"] == 1:
            hold["inertia"] = [axis_inertia()]
        if hold["steps"] > CARRY_SETTLE_STEPS:
            carry["steps"] += 1
            root_z, lowest = float(np_(dummy.data.root_pos_w)[0, 2]), float(np_(dummy.data.body_pos_w)[0, :, 2].min())
            if math.isfinite(root_z) and math.isfinite(lowest):
                carry["root_gap_max_m"] = max(carry["root_gap_max_m"], abs(root_z - carry["root_goal_z_m"]))
                carry["lowest_link_min_m"] = min(carry["lowest_link_min_m"], lowest)
            else:
                # max() and min() keep the earlier value when handed a NaN: counted, or the step would
                # pass as one that was checked
                carry["non_finite_steps"] += 1
        hold["moving_steps"] += int(bool(base.s1a_entry["triggered"][0]))
        hold["yaw_err_max_deg"] = max(hold["yaw_err_max_deg"], abs(wrap(yaw - facing_deg)))
        hold["tilt_max_deg"] = max(hold["tilt_max_deg"], tilt)
        hold["com_gap_max_m"] = max(hold["com_gap_max_m"], com_gap_m())

    # (h) the carried hold's two heights, from the spec and the constant -- not from the cfg being checked
    anthropometry = yaml.safe_load(ANTHROPOMETRY.read_text(encoding="utf-8"))
    stand_height = aisle_guard.standing_root_height(anthropometry)
    spec_root_z = next(float(r["center"][2]) for r in anthropometry["regions"].values() if r.get("parent") is None)
    spec_lowest_centre_z = min(float(r["center"][2]) for r in anthropometry["regions"].values())
    carry = {"clearance_m": HOLD_CLEARANCE_M, "stand_height_m": stand_height, "steps": 0, "root_gap_max_m": 0.0,
             "root_goal_z_m": float(origin[2]) + stand_height + HOLD_CLEARANCE_M,
             "lowest_link_goal_m": float(origin[2]) + spec_lowest_centre_z - (spec_root_z - stand_height) + HOLD_CLEARANCE_M,
             "lowest_link_min_m": math.inf, "non_finite_steps": 0}
    # The spec's standing height against the asset that was spawned: the same height read off the USD's own
    # collision shapes in their authored pose. The servo and the two heights above all come from the spec;
    # a spec edited without rebuilding the asset would move them together and pass.
    carry["asset_stand_height_m"] = asset_standing_height(cfg.scene.dummy.spawn.usd_path)
    drawn_angle_deg = math.degrees(float(base.s1a_entry["angle"][0]))
    facing_deg = (float(args.facing_yaw_deg) if args.facing_yaw_deg is not None
                  else travel_facing_yaw_deg(drawn_angle_deg))
    park_yaw, _ = yaw_tilt_deg(np_(dummy.data.root_quat_w)[0])
    # Reported, not judged: the waiting offset reset_dummy_entry derived, and the dummy's reach it was
    # derived from -- its links' horizontal distance from the root, the same at every turn.
    park_links = np_(dummy.data.body_pos_w)[0, :, :2] - np_(dummy.data.root_pos_w)[0, :2]
    hold = {"entry_angle_deg": drawn_angle_deg, "park_yaw_deg": park_yaw,
            "wait_y_m": abs(float(np_(base.s1a_entry["target_xy"])[0, 1]) - float(origin[1])),
            "dummy_reach_m": float(np.linalg.norm(park_links, axis=1).max()), "steps": 0, "moving_steps": 0, "yaw_err_max_deg": 0.0, "tilt_max_deg": 0.0, "com_gap_max_m": 0.0,
            "park_yaw_err_deg": abs(wrap(park_yaw - facing_deg)), "facing_deg": facing_deg}

    # (j) the fixed cameras' aim, from the pose they record (constant: taken at the first read)
    aim_x = {n: sorted(float(v) for v in site[n]["mount"]["aim_x_range_m"]) for n in fixed}
    aim = {n: {"aim_x_range_m": aim_x[n], "steps": 0, "outside": 0, "edge_px_min": math.inf,
               "x_covered_m": [math.inf, -math.inf], "first_outside": None} for n in fixed}
    fixed_pose: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def robot_x() -> float:
        return float(np_(robot.data.root_pos_w)[0, 0]) - float(origin[0])

    pelvis_z = float(cfg.scene.robot.init_state.pos[2])

    def check_aim_ends(n: str) -> None:
        """A pelvis at either end of the aim range, on the aim line, at the robot's spawn height -- however far
        the robot happens to walk in this episode."""
        ends = []
        for x in aim_x[n]:
            point = np.array([x, float(site[n]["mount"]["aim_y_m"]), pelvis_z]) + origin
            u, v, z = (float(t) for t in cameras.project(point, *fixed_pose[n], ccs[n]))
            edge = min(u, ccs[n]["width"] - u, v, ccs[n]["height"] - v) if z > 0 else -math.inf
            ends.append({"x_m": x, "u": u, "v": v, "z_m": z, "edge_px": edge})
        aim[n]["ends"] = ends
        aim[n]["pelvis_z_m"] = pelvis_z

    def check_aim(step: int) -> None:
        pelvis = np_(robot.data.root_pos_w)[0].astype(float)
        x = float(pelvis[0] - origin[0])
        for n in fixed:
            lo, hi = aim_x[n]
            if not lo <= x <= hi:
                continue
            pos, q = fixed_pose[n]
            u, v, z = (float(t) for t in cameras.project(pelvis, pos, q, ccs[n]))
            a = aim[n]
            a["steps"] += 1
            a["x_covered_m"] = [min(a["x_covered_m"][0], x), max(a["x_covered_m"][1], x)]
            edge = min(u, ccs[n]["width"] - u, v, ccs[n]["height"] - v) if z > 0 else -math.inf
            a["edge_px_min"] = min(a["edge_px_min"], edge)
            if edge < 0:
                a["outside"] += 1
                if a["first_outside"] is None:
                    a["first_outside"] = {"step": step, "x_m": x, "u": u, "v": v, "z_m": z}

    # The camera steps: the Rig warms and reads every camera the way run_s1a does after a reset.
    rig = Rig(base, canon, SITE)
    rig.reset(base)
    first_yaw = cameras.heading_yaw(np_(robot.data.root_quat_w)[0])
    rig.follow()
    rig.warm(canon["render"]["reset_warm_renders"])
    for name in names:
        cams[name]["frames"] = 0
        cams[name]["black"] = {"frames": 0, "first_at_step": None}
        cams[name]["engine_pos_gap_m_max"] = 0.0          # informational: a link camera's engine pose is its spawn's
    probe_stats = {(n, lbl): {"steps": 0, "whole": 0, "compared": 0, "missed": 0, "seen_off_picture": 0,
                              "crowded": 0, "worst_px": 0.0, "sum_px": 0.0, "first_missed": None,
                              "px_range": [math.inf, 0]}
                   for n, lbls in probe_of.items() for lbl in lbls}
    view = {n: {"eye_m": 0.0, "axis_rad": 0.0, "image_up_z_min": math.inf} for n in follow}
    # The cameras the probes check (a marker camera has its one marker): steps with a pair of targets compared
    probe_checked = sorted({n for *_, cams_of in PROBES.values() for n in cams_of if n in ccs})
    pairs = {n: {"steps": 0, "sep_deg_range": [math.inf, 0.0]} for n in probe_checked}
    cont = {n: {"steps": 0, "worst_m": 0.0, "worst_rad": 0.0, "moved_max_m": 0.0, "turned_max_rad": 0.0}
            for n in mounted}
    own = {"camera": box_cam, "steps": 0, "robot_frac_max": 0.0, "robot_frac_max_all_steps": 0.0,
           "robot_classes_on_stage": robot_classes}
    stepped = 0
    x_max = -math.inf               # the furthest the robot got (a done resets it to the start)
    prev_cam: dict[str, tuple] = {}
    prev_link: dict[str, tuple] = {}
    first_episode_over = False
    shapes_checked = False
    labels_seen: list[str] = []
    for step in range(args.steps):
        root = np_(robot.data.root_pos_w)[0]
        yaw = cameras.heading_yaw(np_(robot.data.root_quat_w)[0])
        eyes = {n: cameras.follow_pose(site[n], root, yaw, first_yaw) for n in follow}
        rig.follow()   # before the step, like the recorder: the frame read after it shows the camera here
        with torch.inference_mode():
            obs, _, done, _ = env.step(policy(obs))
        stepped += 1
        dones += int(bool(done[0]))
        if not bool(done[0]):
            x_max = max(x_max, robot_x())
        got = rig.read()

        # (h) while the servo still holds the dummy (the hold ends at release, on contact) -- and only in the
        # episode the expected facing was worked out for. After a done the env has reset itself into another
        # episode, with another entry angle and so another facing: tracked on, its yaw would be compared
        # with the first episode's and the reset's teleport would join the travel.
        first_episode_over = first_episode_over or bool(done[0])
        if not bool(command.hold_released[0]) and not first_episode_over:
            track_hold()

        # (b) the outputs' shapes, once; black frames, every step
        if not shapes_checked:
            shapes_checked = True
            for name in names:
                cc, out = ccs[name], base.scene[name].data.output
                cams[name]["shapes"] = {}
                for ch in cc["channels"]:
                    if ch not in CHANNEL_DEPTH:
                        problems.append(f"(b) {name}.{ch}: a channel this check does not know the depth of")
                        continue
                    shape = tuple(np_(out[ch]).shape)
                    cams[name]["shapes"][ch] = shape
                    if shape != (1, cc["height"], cc["width"], CHANNEL_DEPTH[ch]):
                        problems.append(f"(b) {name}.{ch} shape {shape} != "
                                        f"(1, {cc['height']}, {cc['width']}, {CHANNEL_DEPTH[ch]})")
        for name in names:
            black = cams[name]["black"]
            cams[name]["frames"] += 1
            if is_black(got["frames"][name]):
                black["frames"] += 1
                if black["first_at_step"] is None:
                    black["first_at_step"] = step

        # (c) clearance to the front racks
        body_y = np.abs(np_(robot.data.body_pos_w)[0, :, 1])
        clear_body = min(clear_body, rack - float(body_y.max()))
        clear_pelvis = min(clear_pelvis, rack - abs(float(np_(robot.data.root_pos_w)[0, 1])))

        poses = {n: pose_of(got["poses"][n]) for n in names}
        for n in names:
            gap = float(np.linalg.norm(np_(base.scene[n].data.pos_w)[0] - poses[n][0]))
            cams[n]["engine_pos_gap_m_max"] = max(cams[n]["engine_pos_gap_m_max"], gap)
        labels = decode_labels(got["id_to_labels"])
        labels_seen = sorted({str(v) for v in labels.values()})

        # (i) how much of bbox.camera's picture is the robot
        frac = float(np.isin(got["seg"], labelled(labels, ROBOT_CLASS)).mean())
        own["robot_frac_max_all_steps"] = max(own["robot_frac_max_all_steps"], frac)
        if step < OWN_VIEW_STEPS:
            own["steps"] += 1
            own["robot_frac_max"] = max(own["robot_frac_max"], frac)

        # (g) probes and markers: the target's pixels against its outline through the recorded pose. A
        # marker's centre is where its link puts it this step -- the pixel it should be drawn at is then its
        # point in the camera frame through the canon offset, whatever the link does.
        links = np_(robot.data.body_link_pose_w)[0].astype(float)
        centre_w = dict(probe_w)
        for n in marked:
            lp = links[link_idx[n]]
            centre_w[n + "_marker"] = lp[:3] + cameras._q_rot(lp[3:7], marker_on_link[n])
        sight = functools.partial(visibility, centre_w, poses)


        compared_now: dict[str, list[str]] = {}
        for (n, lbl), pr in probe_stats.items():
            colour = target[lbl][1]
            pr["steps"] += 1
            in_view, whole, o = sight(n, lbl)
            if n == box_cam:
                mask = np.isin(got["seg"], labelled(labels, lbl))
            else:
                mask = colour_mask(got["frames"][n], colour)
                # The others of this colour in view (the two wrist markers share one, as do the two probes of
                # the third-person and fixed cameras, and of the down-looking one): a step where one's outline
                # reaches this target's window is not compared -- its pixels would join. Elsewhere they are left
                # out: the window when this one is whole, their own windows when it is looked for off the picture.
                same = [so[2] for other, (_, c) in target.items() if other != lbl and c == colour
                        for so in (sight(n, other),) if so[0]]
                if whole:
                    u0, v0, u1, v1 = roi(o, ccs[n])
                    if any(meets((u0, v0, u1, v1), x[1]) for x in same):
                        pr["crowded"] += 1
                        continue
                    inside = np.zeros_like(mask)
                    inside[v0:v1, u0:u1] = True
                    mask &= inside
                else:
                    for x in same:
                        w0, w1, w2, w3 = roi(x, ccs[n])
                        mask[w1:w3, w0:w2] = False
            n_px = int(mask.sum())
            if whole:
                pr["whole"] += 1
                if n_px < MIN_PROBE_PX:
                    pr["missed"] += 1
                    if pr["first_missed"] is None:
                        pr["first_missed"] = {"step": step, "pixels": n_px, "outline_centre": o[0]}
                else:
                    cu, cv = centroid(mask)
                    err = math.hypot(cu - o[0][0], cv - o[0][1])
                    pr["compared"] += 1
                    pr["worst_px"] = max(pr["worst_px"], err)
                    pr["sum_px"] += err
                    pr["px_range"] = [min(pr["px_range"][0], n_px), max(pr["px_range"][1], n_px)]
                    compared_now.setdefault(n, []).append(lbl)
            elif not in_view and n_px >= MIN_PROBE_PX:
                pr["seen_off_picture"] += 1
        # ... and a camera's compared targets as a pair: two lines of sight MIN_TARGET_SEP_DEG apart or more
        for n, lbls in compared_now.items():
            sep = max((angle_deg(centre_w[a] - poses[n][0], centre_w[b] - poses[n][0])
                       for i, a in enumerate(lbls) for b in lbls[i + 1:]), default=0.0)
            if sep:
                lo, hi = pairs[n]["sep_deg_range"]
                pairs[n]["sep_deg_range"] = [min(lo, sep), max(hi, sep)]
            if sep >= MIN_TARGET_SEP_DEG:
                pairs[n]["steps"] += 1

        # (g) a follow camera's pose is the eye it was moved to, looking at its target
        for n in follow:
            eye, lookat = eyes[n]
            pos, q = poses[n]
            w = view[n]
            w["eye_m"] = max(w["eye_m"], float(np.linalg.norm(pos - eye)))
            look = (lookat - eye) / np.linalg.norm(lookat - eye)
            axis = cameras._q_rot(q, [0, 0, 1])
            w["axis_rad"] = max(w["axis_rad"], math.acos(max(-1.0, min(1.0, float(np.dot(axis, look))))))
            w["image_up_z_min"] = min(w["image_up_z_min"], float(cameras._q_rot(q, [0, -1, 0])[2]))  # ROS image-up is -y

        # (g) a mounted camera moves with its link
        for n in mounted:
            cam_now = poses[n]
            link_now = (links[link_idx[n], :3], links[link_idx[n], 3:7])
            if n in prev_cam:
                (ct, cq), (lt, lq) = motion(prev_cam[n], cam_now), motion(prev_link[n], link_now)
                c = cont[n]
                c["steps"] += 1
                c["worst_m"] = max(c["worst_m"], float(np.linalg.norm(ct - lt)))
                c["worst_rad"] = max(c["worst_rad"], angle(cameras._q_mul(cq, conj(lq))))
                c["moved_max_m"] = max(c["moved_max_m"], float(np.linalg.norm(cam_now[0] - prev_cam[n][0])))
                c["turned_max_rad"] = max(c["turned_max_rad"], angle(cq))
            prev_cam[n], prev_link[n] = cam_now, link_now

        # (j) from the first read on
        if step == 0:
            for n in fixed:
                fixed_pose[n] = poses[n]
                check_aim_ends(n)
        check_aim(step)

    # (h), (j) past the camera checks: waiting costs the servo almost nothing (no pull, so no lever arm for
    # a misplaced centre of mass); the crossing is where it pulls hardest. Keep stepping until the hold
    # ends (release on contact, or the episode) so both phases are measured, and until the robot has
    # walked past every fixed camera's aim range.
    if not args.cameras_only:
        for step in range(args.steps, args.hold_steps):
            holding = not bool(command.hold_released[0]) and not first_episode_over
            if not holding and all(robot_x() > aim_x[n][1] for n in fixed):
                break
            with torch.inference_mode():
                obs, _, done, _ = env.step(policy(obs))
            stepped += 1
            if bool(done[0]):
                break
            x_max = max(x_max, robot_x())
            if holding:
                track_hold()
            check_aim(step)
    report["steps_run"] = stepped
    report["robot_x_max_m"] = x_max

    # (h) a second episode at another entry angle, one step: the servo's inertia has to be that episode's.
    # Half a turn away the tipping axes read the same, so the other angle is chosen a quarter of the way.
    if not args.cameras_only and args.facing_yaw_deg is None and hold.get("inertia"):
        other = 90.0 if abs(drawn_angle_deg - 90.0) >= 30.0 else 30.0
        base.event_manager.get_term_cfg("reset_dummy_entry").params["angle_range_deg"] = (other, other)
        with torch.inference_mode():
            obs, _ = env.reset()
            env.step(policy(obs))
        hold["inertia"].append(axis_inertia())

    # --- verdicts ---
    for (n, lbl), pr in probe_stats.items():
        pr["mean_px"] = pr.pop("sum_px") / pr["compared"] if pr["compared"] else None
        cams[n].setdefault("targets", {})[lbl] = pr
        if pr["compared"] < pr["steps"] // 2:
            problems.append(f"(g) {n}: the {lbl} was compared in only {pr['compared']}/{pr['steps']} steps "
                            f"(whole in the picture at {pr['whole']}, another of its colour in view at {pr['crowded']})")
        if pr["missed"]:
            problems.append(f"(g) {n}: the {lbl} lies whole in the picture at {pr['missed']} steps where fewer than "
                            f"{MIN_PROBE_PX} of its pixels were drawn (first {pr['first_missed']})")
        if pr["seen_off_picture"]:
            problems.append(f"(g) {n}: the {lbl} was drawn at {pr['seen_off_picture']} steps where the recorded pose "
                            "puts it outside the picture")
        if pr["worst_px"] > PX_TOL:
            problems.append(f"(g) {n}: the {lbl} drawn up to {pr['worst_px']:.2f} px from where the recorded pose "
                            f"and the derived intrinsics put it (tolerance {PX_TOL} px)")
    for n, pair in pairs.items():
        cams[n]["target_pairs"] = pair
        if pair["steps"] < args.steps // 2:
            problems.append(f"(g) {n}: two of its targets, their lines of sight {MIN_TARGET_SEP_DEG:g} deg apart or more, "
                            f"were compared at only {pair['steps']}/{args.steps} steps -- a recorded pose turned about its "
                            "line of sight to one target would go unseen")
    for n, w in view.items():
        cams[n]["view"] = w
        if w["eye_m"] > 1e-4 or w["axis_rad"] > 1e-3 or w["image_up_z_min"] <= 0:
            problems.append(f"(g) {n}: eye {w['eye_m']:.2e} m from where it was moved, axis {w['axis_rad']:.2e} rad "
                            f"off its target, image-up z {w['image_up_z_min']:.3f}")
    for n, c in cont.items():
        cams[n]["continuity"] = c
        if c["worst_m"] > CONTINUITY_TOL_M or c["worst_rad"] > CONTINUITY_TOL_RAD:
            problems.append(f"(g) {n}: between steps it moved up to {c['worst_m']:.2e} m / {c['worst_rad']:.2e} rad "
                            f"otherwise than its link {ccs[n]['parent_prim']}")
        if c["steps"] and c["moved_max_m"] <= CONTINUITY_TOL_M:
            problems.append(f"(g) {n}: never moved more than {c['moved_max_m']:.2e} m between steps — its "
                            "continuity says nothing")
    for n in names:
        black = cams[n]["black"]
        if black["frames"]:
            problems.append(f"(b) {n}: {black['frames']} of its {cams[n]['frames']} frames are black (every pixel 0), the "
                            f"first at step {black['first_at_step']} — the camera sits inside something")
    own["robot_frac_max_pct"] = 100.0 * own["robot_frac_max"]
    report["own_view"] = own
    if own["robot_frac_max"] >= OWN_VIEW_MAX:
        problems.append(f"(i) {box_cam}: the robot fills up to {own['robot_frac_max_pct']:.2f} % of its picture in the "
                        f"first {own['steps']} steps (limit {100 * OWN_VIEW_MAX:.0f} %) — it looks at its own body")
    report["labels_seen_last_frame"] = labels_seen
    report.update({"steps": args.steps, "dones": dones, "rack_inner_y": rack, "guard_margin_m": margin,
                   "clearance_any_body_m": clear_body, "clearance_pelvis_m": clear_pelvis})
    if clear_body < margin:
        problems.append(f"(c) a robot body came within {clear_body:.3f} m of the rack face (< margin {margin} m)")
    report["aim"] = aim
    for n, a in aim.items():
        out = [e for e in a.get("ends", []) if e["edge_px"] < 0]
        if out:
            problems.append(f"(j) {n}: a pelvis at the end of the aim range x {a['aim_x_range_m']} m projects outside "
                            f"the picture: {out}")
    if args.cameras_only:
        report["not_judged"] = "(h) and (j)'s steps: --cameras-only"
    else:
        report["hold"] = hold
        if hold["park_yaw_err_deg"] > PARK_YAW_TOL_DEG:
            problems.append(f"(h) the dummy was parked {hold['park_yaw_err_deg']:.2f} deg off its facing {facing_deg:.2f} "
                            f"(entry angle {drawn_angle_deg:.2f} deg)")
        moved = hold.pop("moved_to_xy", np.zeros(2)) - hold.pop("moved_from_xy", np.zeros(2))
        hold["travel_m"] = float(np.linalg.norm(moved))
        if hold["travel_m"] >= MIN_TRAVEL_M:
            hold["travel_dir_deg"] = math.degrees(math.atan2(moved[1], moved[0]))
            hold["travel_gap_deg"] = abs(wrap(hold["travel_dir_deg"] - park_yaw))
        if args.facing_yaw_deg is None:
            if "travel_gap_deg" not in hold:
                problems.append(f"(h) the held dummy moved {hold['travel_m']:.2f} m (< {MIN_TRAVEL_M}) — which way it "
                                "walks against the way it faces was not checked")
            elif hold["travel_gap_deg"] > TRAVEL_GAP_TOL_DEG:
                problems.append(f"(h) the dummy was parked facing {park_yaw:.2f} deg and then moved toward "
                                f"{hold['travel_dir_deg']:.2f} deg — {hold['travel_gap_deg']:.2f} deg apart "
                                f"(tolerance {TRAVEL_GAP_TOL_DEG})")
            if len(hold.get("inertia", [])) < 2:
                problems.append("(h) the servo's inertia was not read in two episodes")
            for i, a in enumerate(hold.get("inertia", [])):
                if a["rel_gap_max"] > INERTIA_REL_TOL:
                    problems.append(f"(h) episode {i} (entry angle {a['entry_angle_deg']:.1f} deg): the servo's per-axis "
                                    f"inertia {a['servo_kgm2']} is not the live links' {a['live_kgm2']} "
                                    f"(off by {a['rel_gap_max']:.2e}, tolerance {INERTIA_REL_TOL:.0e}) — taken at "
                                    "another episode's facing")
        if hold["steps"] == 0 or hold["moving_steps"] == 0:
            problems.append(f"(h) the servo held the dummy for {hold['steps']} steps, {hold['moving_steps']} of them "
                            "moving — the crossing, where a misplaced centre of mass shows, was not checked")
        if hold["com_gap_max_m"] > COM_GAP_TOL_M:
            problems.append(f"(h) the hold servo pushes {hold['com_gap_max_m'] * 1000:.1f} mm from the dummy's centre "
                            "of mass — its body-frame offset was captured for another facing")
        report["carry"] = carry
        if abs(carry["asset_stand_height_m"] - stand_height) > 1e-4:
            problems.append(f"(h) the spawned dummy's collision shapes put its root {carry['asset_stand_height_m']:.4f} m "
                            f"above its lowest surface, the spec {stand_height:.4f} m — the asset was not built from "
                            "this spec, and the clearance it is carried at is off by the difference")
        if carry["non_finite_steps"]:
            problems.append(f"(h) the held dummy's height was not a finite number at {carry['non_finite_steps']} steps")
        if carry["steps"] == 0:
            problems.append(f"(h) the servo did not hold the dummy past its first {CARRY_SETTLE_STEPS} steps — the height "
                            "it carries it at was not checked")
        else:
            if carry["root_gap_max_m"] > CARRY_TOL_M:
                problems.append(f"(h) the held dummy's root was up to {carry['root_gap_max_m'] * 1000:.1f} mm from "
                                f"{carry['root_goal_z_m']:.4f} m, its standing height {stand_height:.4f} plus the clearance "
                                f"{HOLD_CLEARANCE_M} (tolerance {CARRY_TOL_M * 1000:.0f} mm) — the servo does not carry it there")
            if carry["lowest_link_min_m"] < carry["lowest_link_goal_m"] - CARRY_TOL_M:
                problems.append(f"(h) the held dummy's lowest link centre came down to {carry['lowest_link_min_m']:.4f} m, "
                                f"below {carry['lowest_link_goal_m']:.4f} m where it is standing plus the clearance — its "
                                "feet are on the floor")
        if hold["steps"] and (hold["yaw_err_max_deg"] > HOLD_YAW_TOL_DEG or hold["tilt_max_deg"] > HOLD_TILT_TOL_DEG):
            problems.append(f"(h) held dummy turned {hold['yaw_err_max_deg']:.2f} deg / tilted "
                            f"{hold['tilt_max_deg']:.2f} deg (tolerance {HOLD_YAW_TOL_DEG} / {HOLD_TILT_TOL_DEG}) "
                            f"over {hold['steps']} steps")
        for n, a in aim.items():
            if a["steps"] == 0:
                problems.append(f"(j) {n}: the robot never reached the aim range x {a['aim_x_range_m']} m "
                                f"(got to x {x_max:.2f} m) — nothing checked")
            elif a["outside"]:
                problems.append(f"(j) {n}: the robot's pelvis was outside the picture at {a['outside']} of "
                                f"{a['steps']} steps inside the aim range (first {a['first_outside']})")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    for k, v in report.items():
        if k == "cameras":
            for n, c in v.items():
                print(f"  camera {n}: {c}")
        else:
            print(f"  {k}: {v}")
    for p in problems:
        print(f"PROBLEM {p}")
    if problems:
        print(f"{len(problems)} problem(s)")
    else:
        print("OK (cameras only: (h) and (j)'s steps not judged)" if args.cameras_only else "OK")
    env.close()
    code = 0 if not problems else 1
except SystemExit as exc:
    code = int(exc.code or 0)
except Exception:
    import traceback
    traceback.print_exc()
    code = 1
finally:
    simulation_app.close(exit_code=code)
raise SystemExit(code)
