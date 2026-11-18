"""The scenarios with a conditions table (S1e, S1f, S1g) in the S1d warehouse, one table row per environment.

HriEnvCfg is S1dEnvCfg -- the warehouse aisle, the cameras at the scenario's site (meta/cameras.json ``sites`` -- at
S1g the side's own, camera_approach_side from the row's approach_side), the jointed dummy walking its gait -- with:

* the jointed dummy in the skin and the high-visibility vest (V2_VEST_USD: the canon's human model of these
  scenarios, keti_dummy_ts15066_v2_vest);
* the robot reset to the row's start (conditions.row_site): its x, y and heading; its height the inherited default
  root height; no random offset and no velocity kick (reset_robot_to_site in place of the inherited reset_base);
* no base-mass randomisation (add_base_mass off): the stop test the directed S_p reads was measured on the robot as
  it is;
* the row guard in place of S1d's aisle guard: after every reset the row is checked against the layout once more
  (conditions.row_problems) and the run stops on any problem;
* in a scenario with a ``carry`` block (S1e, S1f, S1g), the box of the row's size and mass held on the torso
  (generation/envs/hri/carry_rig.py);
* the dummy created where the row's script parks it (command_logic.spawn_pose -- created elsewhere, the first reset
  moved it across the floor and the first physics step threw it), placed and walked by the script
  (human_script.plan_for -- events.reset_dummy_scripted in place of S1d's drawn entry), and its contact sensor filtered against each robot link and the held box
  (command_logic.contact_filters -- the force matrix the hold's release, the box's release and the recorder read);
* the command term HriTaskCommand (generation/envs/hri/commands.py): the row's task through the behaviour layer, with
  no speed scaling, protective stop or heading correction, and the dummy's hold let go at the first touch only (no
  distance gate). Until its task comes (S1g) the robot is sent a zero command,
  which is not a hold: the walking policy keeps stepping in place and creeps -- measured over 3 s at the three starts
  on 2026-10-20, 0.07-0.17 m from the start (the most with the box) and 3-5 deg of heading.

The row is ``row`` (conditions.draw_row of the table -- run_s1a.py draws ``--seq``'s); left None it is row 0, which is
what the gym ids register (S1eEnvCfg, S1fEnvCfg, S1gEnvCfg).
"""
from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.configclass import configclass

from generation import carry as carry_canon
from generation import conditions, human_script, sites, visibility
from generation.envs.hri import carry_rig, command_logic
from generation.envs.s1a.mdp.events import reset_dummy_scripted
from generation.envs.s1d import dummy_reset
from generation.envs.s1d.s1d_env_cfg import V2_VEST_USD, S1dEnvCfg
from generation.scene import layout


def layout_boxes():
    """The colliders of the warehouse layout the aisle USD is built from (the scene spec)."""
    return layout.build_layout(layout.load_spec(layout.SCENE_SPEC))


def reset_robot_to_site(env, env_ids, x_m: float, y_m: float, yaw_deg: float, asset_name: str = "robot") -> None:
    """The robot's root at (x_m, y_m) of the env frame, turned yaw_deg (counter-clockwise from +x) about the vertical
    from its default root orientation, at its default root height, at rest."""
    import torch

    import isaaclab.utils.math as math_utils

    asset = env.scene[asset_name]
    pose = asset.data.default_root_pose.torch[env_ids].clone()
    n = len(env_ids)
    origins = env.scene.env_origins[env_ids]
    pose[:, 0] = origins[:, 0] + float(x_m)
    pose[:, 1] = origins[:, 1] + float(y_m)
    pose[:, 2] = pose[:, 2] + origins[:, 2]
    zero = torch.zeros(n, dtype=pose.dtype, device=pose.device)
    yaw = torch.full((n,), math.radians(float(yaw_deg)), dtype=pose.dtype, device=pose.device)
    pose[:, 3:7] = math_utils.quat_mul(pose[:, 3:7], math_utils.quat_from_euler_xyz(zero, zero, yaw))
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, dtype=pose.dtype, device=pose.device),
                                           env_ids=env_ids)


def check_row_after_reset(env, env_ids, scenario_id: str, row: dict) -> None:
    """The row guard once more, after the reset (conditions.row_problems on the layout): a row handed to the
    environment that the guard refuses stops the run before a step is recorded."""
    problems = conditions.row_problems(conditions.scenario_entry(scenario_id), row, layout_boxes())
    if problems:
        more = f" (외 {len(problems) - 1}건)" if len(problems) > 1 else ""
        raise RuntimeError(f"{scenario_id} 조건표의 행이 가드에 걸립니다 — {problems[0]}{more}")


@configclass
class HriEnvCfg(S1dEnvCfg):
    use_jointed: bool = True
    # The conditions-table row of this environment (conditions.draw_row); None is row 0.
    row: dict | None = None

    def __post_init__(self):
        if self.scenario_id not in conditions.table_scenarios():
            raise ValueError(f"HriEnvCfg 는 조건표가 있는 시나리오({list(conditions.table_scenarios())})의 환경입니다: "
                             f"{self.scenario_id!r}")
        if not (self.use_jointed and self.use_skin):
            raise ValueError("조건표 시나리오의 사람은 조끼를 입힌 관절 인형입니다 — use_jointed · use_skin 을 끄지 않습니다")
        entry = conditions.scenario_entry(self.scenario_id)
        if self.row is None:
            self.row = conditions.draw_row(entry, 0, layout_boxes())
        # The third-person and CCTV cameras face the person from the side it walks toward where the site holds a
        # placement per side it comes from (meta/cameras.json by_approach_side): the row's side, on the cfg before
        # the inherited part builds the cameras.
        self.camera_approach_side = self.row.get("approach_side")
        super().__post_init__()
        start = conditions.row_site(sites.load_site(entry), self.row).robot_start
        self.scene.dummy.spawn.usd_path = str(V2_VEST_USD)
        # Same place in the event order as the inherited term, so the joint reset still follows it.
        self.events.reset_base = EventTerm(func=reset_robot_to_site, mode="reset",
                                           params={"x_m": start.x, "y_m": start.y, "yaw_deg": start.yaw_deg})
        self.events.add_base_mass = None
        self.events.aisle_guard = None
        self.events.row_guard = EventTerm(func=check_row_after_reset, mode="reset",
                                          params={"scenario_id": self.scenario_id, "row": dict(self.row)})
        # The person's script, in the place of S1d's drawn entry (same place in the event order).
        plan = human_script.plan_for(entry, self.row)
        self.events.reset_dummy_entry = EventTerm(func=dummy_reset.stand_up_first(reset_dummy_scripted), mode="reset",
                                                  params={"plan": human_script.plan_record(plan)})
        # Created where the script parks it, so that the first reset does not move it across the floor it stood on
        # while the simulator started (command_logic.spawn_pose).
        self.scene.dummy.init_state.pos, self.scene.dummy.init_state.rot = command_logic.spawn_pose(plan)
        carried = carry_canon.load_carry(entry) is not None
        self.scene.dummy_contact.filter_prim_paths_expr = command_logic.contact_filters(
            list(visibility.load_link_boxes()), carry_rig.BOX_PRIM if carried else None)
        self._use_task_command()
        if carried:
            carry_rig.spawn_carried_box(self, self.scenario_id, conditions.box_size(self.row),
                                        float(self.row["box_mass_kg"]))

    def _use_task_command(self) -> None:
        """The inherited command term's settings (the dummy's hold, gait and release) under HriTaskCommandCfg, with no
        walk speed to draw and no heading correction -- the sidecar records both as they are -- and no distance gate on
        the hold: it goes at the first touch (HriTaskCommand._robot_contact_present)."""
        import dataclasses

        from generation.envs.hri.commands import HriTaskCommandCfg

        old = self.commands.base_velocity
        kept = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init and f.name != "class_type"}
        kept.update(walk_speed_range=(0.0, 0.0), heading_correction_gain=0.0, dummy_release_gap_m=None)
        self.commands.base_velocity = HriTaskCommandCfg(**kept, scenario_id=self.scenario_id, row=dict(self.row))


# One class per scenario with a conditions table, for the gym ids (generation/envs/register.py).
@configclass
class S1eEnvCfg(HriEnvCfg):
    scenario_id: str = "S1e"


@configclass
class S1fEnvCfg(HriEnvCfg):
    scenario_id: str = "S1f"


@configclass
class S1gEnvCfg(HriEnvCfg):
    scenario_id: str = "S1g"
