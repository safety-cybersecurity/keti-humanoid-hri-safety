"""What the S1d run loop does with the cameras meta/cameras.json declares (seven):
place each one by its type, move the ones that follow the robot, warm them all after a reset, and
read them each step for the media writer -- every camera's frame and pose, bbox.camera's segmentation for
the box, and a depth only for a pack the canon declares (none since 2026-09-30). Torch-free
at import -- the engine hands tensors (read through .torch), the tests numpy -- so this runs in CI.

Each camera's pose is recorded every frame, by its type:

- mounted: derived from its own parent link (cameras.mounted_pose), not read back -- for a camera under
  an articulation link the engine keeps reporting the spawn pose while the render follows the link
  (measured 2026-09-23). Every camera looks its own link up: the two wrist cameras hang on two elbows.
- follow: moved here before each step to its offsets from the robot root, so the engine's pose for it is
  the rendered one. The offsets are turned by the frame of the scenario's placement (cameras.follow_pose):
  not at all, by the robot's heading, or by the heading reset() took right after the reset.
- fixed: where s1d_env_cfg puts its prim -- cameras.fixed_pose of the scenario's placement, in the env frame,
  under the env origin. It never moves, so it is computed once.

The rig is built for one scenario (the canon's ``sites``): a scenario a follow or fixed camera has no placement
for is refused when the rig is built, and so is one other than the scenario the env's cfg names (its
``scenario_id`` -- the site s1d_env_cfg put the fixed cameras' prims at), when the cfg names one. Where the site
holds a placement per side the person comes from (``by_approach_side``), the rig takes the side the env's cfg names
(its ``camera_approach_side`` -- the side s1d_env_cfg put those prims for); a cfg that names none, or a side the
site holds no placement for, is refused there.

A frame bit-identical to the one the same camera gave at the previous read may not have been drawn
again: the renderer discards frames, and per camera -- seen once in 4,995 frames on 2026-09-23, where step
448 repeated step 447 in the head camera's RGB, depth and mask while the third-person camera's frame was new,
the frame count still right. A fresh render of a lit picture differs, the
renderer's temporal noise sees to that, so a repeat on any camera renders the state again -- every camera,
all read again -- up to MAX_RERENDERS times, and the step counts in ``rerendered``. What still repeats
after that:

- a moving camera's frame stops the run rather than being recorded as the wrong step;
- a fixed camera's frame is recorded: with nothing moving in its view (the robot not yet in it, the dummy
  waiting) a still picture can be real. In the 2026-09-30 smoke run the
  fixed camera's frame changed at every step, by render noise alone (mean absolute difference 0.14-0.69
  between steps, against 1.2 and more for every moving camera).

A black frame -- every pixel 0 -- is never taken for a repeat: all zeros repeat whether drawn again or not,
so bit-equality cannot tell a discarded render from a camera that sees nothing (buried in some geometry,
or mounted in the wrong place). It is recorded like any other: media_writer counts it per camera, the
converter refuses a camera whose every frame is black, and the scene check looks for one
(generation/envs/s1d/check_s1d_scene.py (b)).
"""
from __future__ import annotations

import numpy as np

from generation import cameras
from generation.convert_lerobot import check_record_rate, the_pack
from generation.envs.s1a.media_writer import is_black
from generation.scene import layout

MAX_RERENDERS = 3      # renders of one state before a repeated frame is judged
MOVING = ("mounted", "follow")          # the types whose frame still repeating stops the run
PLACED = (*MOVING, "fixed")             # the types this rig places (cameras.load refuses any other)


def _np(value) -> np.ndarray:
    t = getattr(value, "torch", value)
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


class Rig:
    def __init__(self, env, canon: dict, scenario_id: str):
        # s1d_env_cfg places each fixed camera's prim at its cfg's scenario's site, and this rig records the pose
        # of the site it is given: two ids would record one site's pose for a camera standing at another's.
        built_for = getattr(getattr(env, "cfg", None), "scenario_id", None)
        if built_for is not None and built_for != scenario_id:
            raise ValueError(f"리그의 시나리오 {scenario_id!r} 가 환경 설정의 scenario_id {built_for!r} 와 다릅니다 — "
                             "고정 카메라는 환경 설정의 자리에 놓였으므로 다른 자리의 자세를 기록하게 됩니다")
        self._env = env
        self._robot = env.scene["robot"]
        self._cfg = {name: canon["cameras"][name] for name in cameras.camera_names(canon)}
        unplaced = {name: cfg["type"] for name, cfg in self._cfg.items() if cfg["type"] not in PLACED}
        if unplaced:
            raise ValueError(f"cameras whose type this rig does not place: {unplaced} (types {list(PLACED)})")
        # The side the person comes from that s1d_env_cfg placed the scene cameras for, where the site holds a
        # placement per side (None where the cfg names none -- a site with one placement does not read it).
        side = getattr(getattr(env, "cfg", None), "camera_approach_side", None)
        # The follow and fixed cameras' placements at this scenario's site; a scenario without one is refused here.
        self._site = {name: cameras.site_placement(cfg, scenario_id, name=name, approach_side=side)
                      for name, cfg in self._cfg.items() if cfg["type"] != "mounted"}
        self._initial_yaw: float | None = None
        self._cams = {name: env.scene[name] for name in self._cfg}
        self._moving = [name for name, cfg in self._cfg.items() if cfg["type"] in MOVING]
        self._follow = [name for name, cfg in self._cfg.items() if cfg["type"] == "follow"]
        # The box is computed from the segmentation and labels of the camera meta/cameras.json names
        # for it (bbox.camera). A depth only for a declared pack: the channel the pack names, on the
        # pack's camera -- bbox.camera, cameras.load sees to that, so the mask the writer keeps
        # beside it is that camera's segmentation.
        self._seg = env.scene[canon["bbox"]["camera"]]
        pack = the_pack(canon)
        self._depth = None if pack is None else (env.scene[canon["packs"][pack]["camera"]],
                                                  canon["packs"][pack]["depth"]["source"])
        # One frame per recorded step: the videos' frame period has to be the step period itself, not
        # just round to the same rate (0.0201 s at 50 fps drifts 99.9 ms by frame 999). Same rule as
        # the converter's check on the published timestamps.
        check_record_rate(float(env.step_dt), canon["fps"])
        # Each mounted camera's own parent link.
        self._parent: dict[str, int] = {}
        for name, cfg in self._cfg.items():
            if cfg["type"] == "mounted":
                ids, _ = self._robot.find_bodies(cfg["parent_prim"])
                if len(ids) != 1:
                    raise ValueError(f"{name}'s parent {cfg['parent_prim']!r} matched {len(ids)} robot bodies")
                self._parent[name] = ids[0]
        # A fixed camera's world pose, once: the env-frame pose s1d_env_cfg gives its prim (the same
        # fixed_pose of the same placement and spec) plus the origin of the env its prim sits under.
        self._fixed: dict[str, np.ndarray] = {}
        fixed = [name for name, cfg in self._cfg.items() if cfg["type"] == "fixed"]
        if fixed:
            spec = layout.load_spec(layout.SCENE_SPEC)
            origin = _np(env.scene.env_origins)[0].astype(float)
            for name in fixed:
                pos, quat = cameras.fixed_pose(self._cfg[name], spec, scenario_id, approach_side=side)
                self._fixed[name] = cameras.pose_record(np.asarray(pos, float) + origin, quat)
        self._last: dict[str, np.ndarray] | None = None

    def reset(self, env) -> None:
        """Take the robot's heading in ``env`` -- the env this rig was built on, just reset -- as the episode's first:
        a follow camera placed in the ``initial_heading`` frame keeps its offsets turned by it until the next
        reset. Call after each reset, before follow() and warm()."""
        self._initial_yaw = cameras.heading_yaw(_np(env.scene["robot"].data.root_quat_w)[0])

    def follow(self) -> None:
        """Put every follow camera at its offsets from the robot root, turned by its placement's frame. Call
        before a step."""
        root = _np(self._robot.data.root_pos_w)[0]
        yaw = None
        for name in self._follow:
            if self._site[name]["frame"] == "heading" and yaw is None:
                yaw = cameras.heading_yaw(_np(self._robot.data.root_quat_w)[0])
            eye, target = cameras.follow_pose(self._site[name], root, yaw, self._initial_yaw)
            self._cams[name].set_world_poses_from_view(eye[None].astype(np.float32), target[None].astype(np.float32))

    def warm(self, n: int) -> None:
        """Render the current state n times without recording it (a reset's first frame otherwise
        carries the previous episode -- meta/cameras.json render.reset_warm_renders). The last of
        them becomes what the next read is compared with, so a step-0 frame the renderer discards
        is caught like any other."""
        self._render(n)
        if n:
            self._last = {name: self._rgb(name) for name in self._cams}

    def _render(self, n: int) -> None:
        for _ in range(n):
            self._env.sim.render()
            for cam in self._cams.values():
                cam.update(0.0, force_recompute=True)

    def read(self) -> dict:
        """This step's frames, every camera's pose (pos3 + quat4 wxyz), bbox.camera's segmentation and
        label table, and how many times the state had to be rendered again for fresh frames; with a pack
        declared, its depth too -- without one there is no ``depth`` key."""
        got = self._read()
        rerendered = 0
        while True:
            stale = self._repeated(got["frames"])
            if not stale:
                break
            if rerendered == MAX_RERENDERS:
                stuck = [name for name in stale if name in self._moving]
                if stuck:
                    raise RuntimeError(f"{', '.join(stuck)}: still the previous read's frame after "
                                       f"{MAX_RERENDERS} renders -- not recording it as this step's")
                break               # only fixed cameras still repeat: a still view, recorded as it is
            self._render(1)
            rerendered += 1
            got = self._read()
        self._last = {name: frame.copy() for name, frame in got["frames"].items()}
        got["rerendered"] = rerendered
        return got

    def _repeated(self, frames: dict) -> list[str]:
        """The cameras whose frame is, bit for bit, the one they gave at the previous read -- a black frame
        aside, which repeats whether drawn again or not (is_black)."""
        if self._last is None:
            return []
        return [name for name, frame in frames.items()
                if not is_black(frame) and np.array_equal(frame, self._last[name])]

    def _rgb(self, name: str) -> np.ndarray:
        # A copy, not a view: a buffer the engine refills must not change a frame already handed out.
        return np.array(_np(self._cams[name].data.output["rgb"])[0, ..., :3], order="C")

    def _pose(self, name: str, links: np.ndarray | None) -> np.ndarray:
        cfg = self._cfg[name]
        if cfg["type"] == "mounted":
            pos, quat = cameras.mounted_pose(links[self._parent[name]], cfg["offset_pos"], cfg["offset_rot_xyzw"])
            return cameras.pose_record(pos, quat)
        if cfg["type"] == "follow":
            data = self._cams[name].data
            return cameras.pose_record(_np(data.pos_w)[0], _np(data.quat_w_ros)[0])
        return self._fixed[name].copy()

    def _read(self) -> dict:
        out_s = self._seg.data.output
        info = self._seg.data.info or {}
        links = _np(self._robot.data.body_link_pose_w)[0] if self._parent else None
        got = {"frames": {name: self._rgb(name) for name in self._cams},
               "seg": np.array(_np(out_s["semantic_segmentation"])[0, ..., 0]),
               "id_to_labels": (info.get("semantic_segmentation") or {}).get("idToLabels") or {},
               "poses": {name: self._pose(name, links) for name in self._cams}}
        if self._depth is not None:
            cam, channel = self._depth
            got["depth"] = np.array(_np(cam.data.output[channel])[0, ..., 0])
        return got
