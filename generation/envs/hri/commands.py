"""The command term of the scenarios with a conditions table (S1e, S1f, S1g): the row's task through the behaviour
layer, the person's script, the HRI geometry of every step, what the robot's cameras see, and the carried box let go
at the first touch.

HriTaskCommand keeps S1d's command term for the person -- the hold servo, the gait, the hold's release -- and changes
three things in it: the walk is the row's script (human_script.plan_for, parked by events.reset_dummy_scripted --
its trigger and stop are read through _trigger_latched and _stop_distance, and a plan that turns on the spot between
the trigger and the walk (S1f) turns there: the facing the hold servo aims at goes step by step from the plan's facing
to its walk while the legs mark time through the gait drive, _gait_drive, and the hold's yaw damper reads the dummy's
rate against the facing's, _facing_turn_torque), the hold has no distance gate (the
environment sets dummy_release_gap_m to None), and the contact the hold is released by is what the robot's links and
the held box gave the person, read from the dummy contact sensor's force matrix (command_logic.body_force -- the
robot's own net force counts the racks and the floor), above the threshold the box is let go at
(release_threshold_n, not S1d's robot_contact_threshold_n). The hold, the box and the recorder's first contact_onset
therefore go at one step, and through a near miss that never touches the dummy stays held, walking with the hold's
assist, to the end. The robot's command is its own: S1d's speed scaling, protective stop and heading correction toward
yaw 0 are not used.

Every policy step, after the physics of the step:

1. the step's touch -- the peak over its substeps of what the robot and the box gave each of the person's bodies
   (command_logic.TouchWindow on the action term's substep hook);
2. the HRI geometry (``hri``) and what each robot camera sees of the person (``camera_fractions``, ``seen``), with the
   box as it was held during the step -- a camera the swaying arm has brought inside the box sees past it
   (visibility.robot_camera_fractions), and each camera's clearance from the box and the cameras inside it are kept
   (``camera_box_clearance``, ``cameras_in_box``);
3. the hold's release and the box let go, both at the first step with any touch (release_threshold_n -- the noise
   floor the recorder's contact_onset uses; no distance gate), and the person's script moved on;
4. the task: S1e's and S1f's come at the first step, S1g's once the person is within the row's trigger distance along
   their walk (command_logic.task_starts); the path follower's command (task_path.PathFollower), the perceived person
   delayed by the row's sensor delay (behavior.DelayLine), the behaviour's answer -- [forward speed, 0, yaw rate] is
   the policy's command for the next step -- the goal's completion (task_path.CompletionTracker) and the step each
   of the task's stages ended at (task_path.StageTracker).

The recorder reads ``hri``, ``camera_fractions``, ``cameras_in_box``, ``completion``, ``stages``, ``stature_m``,
``box_released_step`` and ``task_started_step``; the run prints the least ``camera_box_clearance`` of the
``moving_cameras``. One env: the contact filters name one prim each.
"""
from __future__ import annotations

import json
import math
from dataclasses import MISSING
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils.configclass import configclass

from generation import behavior as behavior_layer
from generation import cameras as camera_canon
from generation import carry as carry_canon
from generation import conditions, directed_separation, hri_geometry, human_script, sites, task_path, visibility
from generation.dummy import joints_spec
from generation.envs.hri import command_logic as logic
from generation.envs.hri.carry_rig import RIG_ATTR
from generation.envs.s1a.mdp.commands import S1aStraightWalkCommand, S1aStraightWalkCommandCfg
from generation.envs.s1a.mdp.entry_geometry import facing_quat_xyzw
from generation.envs.s1a.recorder import CONTACT_EPS_N
from generation.scene import layout

REPO = Path(__file__).resolve().parents[3]
RULES = REPO / "meta/rules.json"
CAMERAS = REPO / "meta/cameras.json"
DUMMY_SENSOR = "dummy_contact"


def _t(x):
    """A warp-backed Isaac Lab buffer as a torch tensor."""
    return x.torch if hasattr(x, "torch") else x


def _row(x) -> np.ndarray:
    """Env 0's row of a buffer, as float64 numpy."""
    return _t(x)[0].detach().cpu().numpy().astype(float)


class HriTaskCommand(S1aStraightWalkCommand):
    """The row's task through the behaviour layer, the person's script and the box let go at the first touch."""

    cfg: "HriTaskCommandCfg"

    def __init__(self, cfg: "HriTaskCommandCfg", env):
        super().__init__(cfg, env)
        if self.num_envs != 1:
            raise ValueError(f"조건표 시나리오의 명령 항은 환경 하나에서만 돕니다(접촉 필터가 프림 하나씩을 가리킴): {self.num_envs}")
        entry = conditions.scenario_entry(cfg.scenario_id)
        self.row = dict(cfg.row)
        self.site = conditions.row_site(sites.load_site(entry), self.row)
        self.plan = human_script.plan_for(entry, self.row)
        self.carry = carry_canon.load_carry(entry)
        self._box_mass_kg = float(self.row["box_mass_kg"]) if self.carry is not None else 0.0
        self._start_distance = (conditions.s1g_trigger_distance(self.row)
                                if self.site.path["kind"] == logic.WAITING_KIND else None)
        # The canon, read once.
        self._ssm, self._stop_table = directed_separation.load_directed_ssm(
            json.loads(RULES.read_text(encoding="utf-8")))
        self._link_boxes = visibility.load_link_boxes()
        self._cameras = camera_canon.load(CAMERAS)
        self._rule = visibility.seeing_rule(self._cameras)
        self._robot_cameras = visibility.robot_cameras(self._cameras)
        spec = joints_spec.load()
        self._shapes = hri_geometry.human_shapes(spec)
        self._pelvis = hri_geometry.pelvis_body(spec)
        self._points = visibility.body_sample_points(spec, self._rule["sample_points"])
        # The person's stature (m): the seeing rule's height on the picture takes it, and the sidecar records it.
        self.stature_m = float(spec.anthropometry["stature"])
        self._occluders = visibility.layout_occluders(layout.build_layout(layout.load_spec(layout.SCENE_SPEC)))
        self.behavior = behavior_layer.PassThrough()
        # The cameras on a link the box is not fixed to -- the wrist cameras, which the arms carry about the box.
        self.moving_cameras = carry_canon.moving_cameras(self.carry, self._cameras) if self.carry is not None else ()
        # Read by the recorder.
        self.hri: hri_geometry.HriStep | None = None
        self.camera_fractions: dict[str, float] = {}
        self.seen = False
        self.camera_box_clearance: dict[str, float] = {}
        self.cameras_in_box: tuple[str, ...] = ()
        self.touch: logic.Touch | None = None
        self.box_released_step = -1
        self.task_started_step = -1
        self.follower: task_path.PathFollower | None = None
        self.completion: task_path.CompletionTracker | None = None
        self.stages: task_path.StageTracker | None = None
        self._delay: behavior_layer.DelayLine | None = None
        self._last_direction: np.ndarray | None = None
        self._frame: dict | None = None
        self._indices = None
        self._window: logic.TouchWindow | None = None
        # The dummy sensor's bodies and the filters its force matrix is read against (_ensure_window).
        self.touch_bodies: list[str] = []
        self.touch_sources: list[str] = []
        self._new_episode()

    # ---- the behaviour ----

    def set_behavior(self, behavior) -> None:
        """The behaviour layer between the task and the walking policy (behavior.load_behavior's)."""
        if not callable(behavior):
            raise ValueError(f"행동 층이 호출할 수 있는 것이 아닙니다: {behavior!r}")
        self.behavior = behavior

    # ---- per episode ----

    def _new_episode(self) -> None:
        self.follower = task_path.PathFollower(self.site, self.row, dt=self._env.step_dt)
        self.completion = task_path.CompletionTracker(self.site.robot_goal)
        self.stages = task_path.StageTracker(self.site, self.row)
        self._delay = behavior_layer.DelayLine(
            behavior_layer.delay_steps(float(self.row["sensor_delay_ms"]), self._env.step_dt))
        self.hri = None
        self.camera_fractions = {}
        self.seen = False
        self.camera_box_clearance = {}
        self.cameras_in_box = ()
        self.touch = None
        self.box_released_step = -1
        self.task_started_step = -1
        # The turn on the spot: the episode step the trigger latched at, and how many steps into the turn this step is
        # (-1 outside it).
        self._turn_from_step = -1
        self._turn_k = -1

    def _resample_command(self, env_ids):
        # The first step's approach line is the robot's root to the pelvis (command_logic.first_direction): the
        # distance the inherited reset measures is taken with it.
        self._last_direction = None
        super()._resample_command(env_ids)
        self._command[env_ids] = 0.0
        # What the sidecar records of the drawn controller settings: the row's delay and the task's walking speed.
        self._delay_steps[env_ids] = behavior_layer.delay_steps(float(self.row["sensor_delay_ms"]), self._env.step_dt)
        self._walk_speed[env_ids] = float(self.row.get("robot_speed_mps", self.row.get("walk_speed_mps", 0.0)))
        self._new_episode()
        self._ensure_window()
        if self._window is not None:
            self._window.reset()

    # ---- the scene, read by name ----

    def _ensure_indices(self) -> None:
        if self._indices is not None:
            return
        robot, dummy = self._env.scene["robot"], self._env.scene["dummy"]
        robot_names, dummy_names = list(robot.body_names), list(dummy.body_names)
        missing = sorted(set(self._link_boxes) - set(robot_names))
        if missing:
            raise ValueError(f"링크 상자의 링크 {missing} 가 로봇의 바디({robot_names})에 없습니다")
        missing = sorted(set(self._shapes) - set(dummy_names))
        if missing:
            raise ValueError(f"사람 부위 {missing} 가 인형의 바디({dummy_names})에 없습니다")
        self._indices = {
            "links": [robot_names.index(n) for n in self._link_boxes],
            "link_names": list(self._link_boxes),
            "parts": [dummy_names.index(n) for n in self._shapes],
            "part_names": list(self._shapes),
            "pelvis": dummy_names.index(self._pelvis),
        }

    def _ensure_window(self) -> None:
        """The touch window on the action term's substep hook, once the action manager exists."""
        if self._window is not None:
            return
        manager = getattr(self._env, "action_manager", None)
        if manager is None:
            return
        sensor = self._env.scene[DUMMY_SENSOR]
        filters = list(sensor.cfg.filter_prim_paths_expr or [])
        if not filters:
            raise ValueError("인형의 접촉 센서에 로봇 링크 · 상자 필터가 없습니다 — HriEnvCfg 가 걸어야 합니다")
        count = int(sensor.contact_view.filter_count)
        if count != len(filters):
            raise ValueError(f"인형의 접촉 센서가 필터 {count} 개를 봅니다 — 설정은 {len(filters)} 개입니다")
        self.touch_sources = logic.filter_names(filters)
        self.touch_bodies = list(sensor.body_names)

        def read():
            return _t(sensor.data.force_matrix_w)[0].detach().cpu().numpy()

        self._window = logic.TouchWindow(read)
        term = manager._terms[manager.active_terms[0]]
        original = term.apply_actions

        def apply_and_sample():
            original()
            self._window.sample()

        term.apply_actions = apply_and_sample

    def _rig(self):
        return getattr(self._env, RIG_ATTR, None) if self.carry is not None else None

    # ---- the person's script ----

    def _episode_step(self) -> int:
        """This env step's index in the episode (the recorder's): the episode length has been counted up already."""
        return int(self._env.episode_length_buf[0]) - 1

    def _trigger_latched(self, entry, robot) -> torch.Tensor:
        """The row's script: never, the robot behind the person by less than the row's distance, or a time -- and
        which envs walk this step. A plan that turns on the spot (command_logic.turn_steps > 0) spends its first n
        steps from the trigger's own turning while held: the facing the hold servo aims at goes from the plan's facing
        to its walk (command_logic.turn_yaw_deg), the gait marks time (_gait_drive) and no env walks, so the base term
        keeps the target where it is and the walk's speed at rest. The step after those n the walk sets off at the
        slowest gait (human_script.turn_gait_speed_mps), as the plan's walk does. A release ends the turn in
        _advance_dummy (the base term returns before this method once the person is released); the ``held`` branch
        here covers a direct call, where a released person's facing is left where it was and the latch is returned."""
        step = max(0, self._episode_step())
        t = step * self._env.step_dt
        due = logic.trigger_due(entry["trigger_kind"], float(entry["trigger_distance"][0]), t=t,
                                person_x=float(entry["x_cross"][0]), robot_x=float(_t(robot.data.root_pos_w)[0, 0]))
        if due:
            entry["triggered"][:] = True
        n = logic.turn_steps(self.plan.turn_time_s, self._env.step_dt)
        if n <= 0 or not bool(entry["triggered"][0]):
            return entry["triggered"]
        if self._turn_from_step < 0:
            self._turn_from_step = step
        k = step - self._turn_from_step
        held = not bool(self.hold_released[0])
        if held and k < n:
            yaw = logic.turn_yaw_deg(self.plan.yaw_deg, human_script.walk_yaw_deg(self.plan), k + 1, n)
            quat = entry["facing_quat"]
            quat[:] = torch.tensor(facing_quat_xyzw(yaw), dtype=quat.dtype, device=quat.device)
            self._turn_k = k
            return torch.zeros_like(entry["triggered"])
        self._turn_k = -1
        if held and k == n:
            # The base term's ramp goes on from here: the walk after the turn sets off at the slowest gait.
            self.gait_speed[:] = human_script.turn_gait_speed_mps()
        return entry["triggered"]

    def _gait_drive(self, target_dxy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """While the person turns on the spot (_turn_k >= 0, set by _trigger_latched this step) the legs mark time:
        the gait is driven at command_logic.turn_gait_speed and its phase advances by that speed times the step,
        while the target stays put and nothing is slowing. Otherwise the base term's drive."""
        if self._turn_k < 0:
            return super()._gait_drive(target_dxy)
        dt = self._env.step_dt
        v = logic.turn_gait_speed(self._turn_k, human_script.turn_gait_speed_mps(), dt)
        return np.array([v]), np.array([False]), np.array([v * dt])

    def _facing_turn_torque(self, ang_damping: torch.Tensor) -> torch.Tensor | None:
        """While the person turns on the spot (_turn_k >= 0, set by _trigger_latched this step) the facing the hold aims
        at -- the turn's k + 1, written this step -- turns about +z at command_logic.turn_yaw_rate_deg_s: the hold's yaw
        damper times that rate, on z alone, so the damper pulls the dummy toward the facing's rate instead of toward
        standing still. Measured in Isaac Sim 6.0.1 / Isaac Lab 3.0.0 on 2026-11-13 (S1f rows 0, 1, 98): the dummy
        at the turn's last step 13.2 / 1.8 / 0.3 deg from the walk's direction with it, 23.7 / 39.1 / 30.7 deg
        without. None otherwise: the hold's spring and damper alone."""
        if self._turn_k < 0:
            return None
        dt = self._env.step_dt
        n = logic.turn_steps(self.plan.turn_time_s, dt)
        rate = logic.turn_yaw_rate_deg_s(self.plan.yaw_deg, human_script.walk_yaw_deg(self.plan), self._turn_k + 1, n,
                                         dt)
        torque = torch.zeros_like(ang_damping)
        torque[:, 2] = ang_damping[:, 2] * math.radians(rate)
        return torque

    def _stop_distance(self, entry):
        """Metres the script's walk has left -- its length less how far the target has come along it -- or None for a
        script without a stop (the person who stands)."""
        remaining = logic.walk_remaining(entry["target_xy"][0].cpu().numpy(), entry["start_xy"][0].cpu().numpy(),
                                         entry["walk_dir"][0].cpu().numpy(), float(entry["stop_after_m"][0]))
        return None if math.isinf(remaining) else np.array([remaining])

    def _advance_dummy(self):
        if bool(self.hold_released.all()):
            # Released, the base term returns before the script (_trigger_latched): the turn ends here too.
            self._turn_k = -1
        super()._advance_dummy()
        entry = getattr(self._env, "s1a_entry", None)
        if entry is None or not bool(entry["triggered"][0]):
            return
        remaining = self._stop_distance(entry)
        if remaining is not None and float(remaining[0]) <= 0.0:
            entry["walk_ended"][:] = True

    # ---- the geometry ----

    def _read_frame(self) -> dict:
        """The step's poses and velocities, as the geometry and the cameras take them (world frame)."""
        self._ensure_indices()
        robot, dummy = self._env.scene["robot"], self._env.scene["dummy"]
        idx = self._indices
        r_pose, r_vel = _row(robot.data.body_link_pose_w), _row(robot.data.body_link_vel_w)
        d_pose, d_vel = _row(dummy.data.body_link_pose_w), _row(dummy.data.body_link_vel_w)
        link_poses = {n: r_pose[i] for n, i in zip(idx["link_names"], idx["links"])}
        part_poses = {n: d_pose[i] for n, i in zip(idx["part_names"], idx["parts"])}
        # The root's velocity as observation.root_vel records it (the root body's centre of mass): S_p, the task's
        # completion and the behaviour read the speed a consumer can read again from the shipped columns.
        root_pose, root_vel = _row(robot.data.root_link_pose_w), _row(robot.data.root_vel_w)
        rig = self._rig()
        box_obb = box_state = None
        released = False
        if rig is not None:
            pos, quat = rig.box_pose()
            lin, ang = rig.box_velocity()
            box_obb = visibility.Obb(center=pos, rot_xyzw=quat / np.linalg.norm(quat), half=np.asarray(rig.size) / 2.0)
            box_state = (pos, lin, ang)
            released = bool(rig.released)
        origin = _row(self._env.scene.env_origins)
        return {
            "link_poses": link_poses,
            "link_states": [(r_pose[i, :3], r_vel[i, :3], r_vel[i, 3:]) for i in idx["links"]],
            "parts": [(self._shapes[n], d_pose[i, :3], d_pose[i, 3:7]) for n, i in zip(idx["part_names"], idx["parts"])],
            "part_states": [(d_pose[i, :3], d_vel[i, :3], d_vel[i, 3:]) for i in idx["parts"]],
            "part_poses": part_poses,
            "pelvis": d_pose[idx["pelvis"], :3], "pelvis_vel": d_vel[idx["pelvis"], :3],
            "root": root_pose[:3], "root_vel": root_vel,
            "yaw": float(_row(robot.data.heading_w)),
            "box_obb": box_obb, "box_state": box_state, "released": released,
            "origin": origin,
        }

    def _geometry(self, frame: dict) -> hri_geometry.HriStep:
        last = self._last_direction
        if last is None:
            last = logic.first_direction(frame["root"], frame["pelvis"])
        robot_obbs = visibility.robot_obbs(self._link_boxes, frame["link_poses"])
        v = frame["root_vel"]
        return hri_geometry.hri_step(
            robot_obbs=robot_obbs, robot_states=frame["link_states"], box_obb=frame["box_obb"],
            box_state=frame["box_state"], released=frame["released"], human=frame["parts"],
            human_states=frame["part_states"], robot_speed=float(math.hypot(v[0], v[1])), yaw_rate=float(v[5]),
            box_mass_kg=self._box_mass_kg, params=self._ssm, table=self._stop_table, last_direction=last)

    def _measure_min_distance(self) -> torch.Tensor:
        """The surface distance of the HRI geometry (hri_geometry.hri_step), as the inherited reset measures it. The
        hold's release does not read a distance here: dummy_release_gap_m is None."""
        distance = self._geometry(self._read_frame()).distance
        return torch.full((self.num_envs,), distance, device=self.device)

    def _robot_contact_present(self) -> torch.Tensor:
        """Whether the robot's links and the held box gave the person anything this step: the touch above
        release_threshold_n, the box's threshold, so the hold goes on the step the box does."""
        over = self.touch is not None and self.touch.over(self.cfg.release_threshold_n)
        return torch.full((self.num_envs,), over, dtype=torch.bool, device=self.device)

    def _camera_view(self, frame: dict) -> tuple[dict[str, float], bool]:
        """{robot camera: visible fraction} and whether one of them sees the person (visibility.seen). Keeps each robot
        camera's clearance from the held box and the cameras inside it ({} and () when no box is held)."""
        points = visibility.points_world(self._points, frame["part_poses"])
        held_box = frame["box_obb"] if not frame["released"] else None
        self.camera_box_clearance = {} if held_box is None else visibility.robot_camera_box_clearances(
            self._cameras, frame["link_poses"], held_box)
        self.cameras_in_box = visibility.cameras_in_box(self.camera_box_clearance)
        lo, hi = self._occluders
        fractions = visibility.robot_camera_fractions(self._cameras, frame["link_poses"], self._link_boxes, held_box,
                                                      points, lo, hi)
        heights = {}
        for name in self._robot_cameras:
            cam = self._cameras["cameras"][name]
            pos, quat = camera_canon.mounted_pose(frame["link_poses"][cam["parent_prim"]], cam["offset_pos"],
                                                  cam["offset_rot_xyzw"])
            heights[name] = visibility.height_px(cam, pos, quat, frame["pelvis"], self.stature_m)
        return fractions, logic.any_seen(fractions, heights, self._rule)

    # ---- the step ----

    def _update_command(self):
        # The wrench the physics that just ran used, before the release or the servo replaces it (as S1d's term).
        self.hold_force_in_effect = (torch.zeros(self.num_envs, 3, device=self.device)
                                     if self.dummy_lin_force_applied is None else self.dummy_lin_force_applied.clone())
        self.hold_torque_in_effect = (torch.zeros(self.num_envs, 3, device=self.device)
                                      if self.dummy_torque_applied is None else self.dummy_torque_applied.clone())
        self._ensure_window()
        self.touch = self._window.drain()
        step = self._episode_step()
        t = max(0, step) * self._env.step_dt
        frame = self._read_frame()
        self._frame = frame
        self.hri = self._geometry(frame)
        self._last_direction = self.hri.direction
        self.camera_fractions, self.seen = self._camera_view(frame)
        measured = torch.full((self.num_envs,), self.hri.distance, device=self.device)
        self._update_hold_release(measured)
        rig = self._rig()
        if rig is not None and not rig.released and self.touch.over(self.cfg.release_threshold_n):
            rig.release()
            self.box_released_step = step
        self._advance_dummy()
        self.min_distance = measured
        self.separation_required = torch.full((self.num_envs,), self.hri.s_p, device=self.device)

        # The task, in the site's frame (the env's).
        origin = frame["origin"]
        xy = frame["root"][:2] - origin[:2]
        pelvis_xy = frame["pelvis"][:2] - origin[:2]
        v = frame["root_vel"]
        if step >= 0 and self.follower.started_step < 0 and logic.task_starts(
                self.site.path["kind"], self.plan, pelvis_xy=pelvis_xy, robot_xy=xy, distance=self._start_distance):
            self.follower.start(step)
            self.task_started_step = step
        task_cmd = self.follower.command(t, xy, frame["yaw"])
        self.completion.update(step, xy, frame["yaw"], float(math.hypot(v[0], v[1])))
        self.stages.update(step, self.follower, self.completion.step)
        now = logic.perceived(self.seen, pelvis_xy=pelvis_xy, pelvis_vel_xy=frame["pelvis_vel"][:2], robot_xy=xy,
                              robot_vel_xy=v[:2])
        inp = behavior_layer.BehaviorInput(
            t=float(t), robot_xy=(float(xy[0]), float(xy[1])), robot_yaw=frame["yaw"],
            robot_vel_xy=(float(v[0]), float(v[1])), yaw_rate=float(v[5]),
            task_command=(float(task_cmd[0]), float(task_cmd[1])), task_phase=self.follower.phase,
            perceived=self._delay.push(now))
        forward, yaw_rate = logic.behavior_command(self.behavior(inp))
        self._command[:, 0] = forward
        self._command[:, 1] = 0.0
        self._command[:, 2] = yaw_rate


@configclass
class HriTaskCommandCfg(S1aStraightWalkCommandCfg):
    class_type: type = HriTaskCommand

    # The scenario and its conditions-table row (conditions.draw_row).
    scenario_id: str = MISSING
    row: dict = MISSING
    # The box and the dummy's hold are let go at the first step whose touch is above this: the noise floor the
    # recorder's contact_onset uses (recorder.CONTACT_EPS_N), so the two go at the episode's first contact_onset.
    release_threshold_n: float = CONTACT_EPS_N
