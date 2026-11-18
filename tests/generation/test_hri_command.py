"""The Isaac-free half of the conditions-table scenarios' command term (generation/envs/hri/command_logic.py) and the
scripted person's entry angle (entry_geometry.walk_angle_rad): the dummy sensor's filters and how its force matrix is
read, the peak over an env step's substeps, the person's trigger and walk, when the robot's task comes, the first
approach line and what a behaviour is told. The Isaac side (commands.py, hri_env_cfg.py, events.py) imports Isaac and
is pinned from its source; the expectations here are matrices and walks written out by hand, the canon and its rows."""

import ast
import copy
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from generation import conditions, human_script, visibility
from generation.behavior import Perceived
from generation.envs.hri import command_logic as logic
from generation.envs.s1a.mdp.entry_geometry import walk_angle_rad
from generation.scene import layout

REPO = Path(__file__).resolve().parents[2]
DT = 0.02
COMMANDS = REPO / "generation/envs/hri/commands.py"
HRI_CFG = REPO / "generation/envs/hri/hri_env_cfg.py"
EVENTS = REPO / "generation/envs/s1a/mdp/events.py"
BASE = REPO / "generation/envs/s1a/mdp/commands.py"


def _entry(sid):
    canon = json.loads((REPO / "meta" / "scenarios.json").read_text(encoding="utf-8"))
    return copy.deepcopy(next(s for s in canon["scenarios"] if s["scenario_id"] == sid))


@pytest.fixture(scope="module")
def boxes():
    return layout.build_layout(layout.load_spec(layout.SCENE_SPEC))


# ---- the filters and the force matrix ----

def test_the_filters_are_one_per_link_then_the_box():
    links = list(visibility.load_link_boxes())
    filters = logic.contact_filters(links, "Box")
    assert len(links) == 20 and len(filters) == 21
    assert filters[:20] == [f"{{ENV_REGEX_NS}}/Robot/{n}" for n in links]
    assert filters[20] == "{ENV_REGEX_NS}/Box"
    assert logic.filter_names(filters) == links + ["Box"]
    assert logic.contact_filters(links, None) == filters[:20]          # a scenario without a box
    # the columns follow the names handed over, in that order
    assert logic.filter_names(logic.contact_filters(["z_link", "a_link"], "Box")) == ["z_link", "a_link", "Box"]


@pytest.mark.parametrize("links", ["torso_link", [], ["a", "a"], ["a", ""]])
def test_filters_need_distinct_link_names(links):
    with pytest.raises(ValueError):
        logic.contact_filters(links, None)


def _matrix():
    """Three of the person's bodies, filters (link a, link b, box): body 0 pressed by a and the box, body 1 by b alone,
    body 2 pinched between a and b pushing against each other."""
    m = np.zeros((3, 3, 3))
    m[0, 0] = (3.0, 0.0, 0.0)
    m[0, 2] = (0.0, 4.0, 0.0)
    m[1, 1] = (0.0, 0.0, -2.0)
    m[2, 0] = (5.0, 0.0, 0.0)
    m[2, 1] = (-5.0, 0.0, 0.0)
    return m


def test_a_body_gets_the_sum_of_what_the_filters_give_it_or_its_strongest_part_whichever_is_larger():
    # body 0: |(3, 4, 0)| = 5 > its strongest part 4; body 1: one part; body 2: the pinch keeps one side's 5 N
    assert logic.body_force(_matrix()) == pytest.approx([5.0, 2.0, 5.0])


def _one_body(*forces):
    """One of the person's bodies, one filter per force."""
    return np.array([[list(f) for f in forces]], float)


def test_a_pinch_between_two_parts_is_one_sides_push():
    # two robot parts pressing one body from opposite sides -- a hand between the held box and a forearm
    assert logic.body_force(_one_body((10.0, 0, 0), (-10.0, 0, 0))) == pytest.approx([10.0])


def test_parts_pushing_the_same_way_add_up():
    assert logic.body_force(_one_body((5.0, 0, 0), (5.0, 0, 0))) == pytest.approx([10.0])


def test_a_single_part_is_its_own_push():
    assert logic.body_force(_one_body((3.0, 4.0, 0.0))) == pytest.approx([5.0])
    assert logic.body_force(_one_body((3.0, 4.0, 0.0), (0.0, 0.0, 0.0))) == pytest.approx([5.0])


@pytest.mark.parametrize("forces, expected", [
    (((10.0, 0, 0), (-10.0, 0, 0), (0, 3.0, 0)), 10.0),         # a pinch beside a light push: the pinch's side
    (((6.0, 0, 0), (6.0, 0, 0), (-4.0, 0, 0)), 8.0),           # two the same way, one against: the sum
    (((3.0, 0, 0), (0, 4.0, 0), (0, 0, 12.0)), 13.0),           # three across: the sum
    (((7.0, 0, 0), (-7.0, 0, 0), (0, 0, 9.0)), 9.0),            # the pinch cancels to 9 N, the lone 9 N part stays
], ids=["pinch-and-a-push", "two-with-one-against", "three-across", "pinch-beside-a-larger"])
def test_three_parts_on_one_body(forces, expected):
    assert logic.body_force(_one_body(*forces)) == pytest.approx([expected])


def test_a_body_with_no_filter_gets_nothing():
    assert logic.body_force(np.zeros((2, 0, 3))) == pytest.approx([0.0, 0.0])


def test_a_filter_gives_the_person_the_sum_over_the_bodies():
    assert logic.source_force(_matrix()) == pytest.approx([8.0, math.hypot(5.0, 2.0), 4.0])


@pytest.mark.parametrize("bad", [np.zeros((3, 3)), np.zeros((3, 3, 2)), np.full((1, 1, 3), np.nan)])
def test_a_matrix_of_the_wrong_shape_is_refused(bad):
    with pytest.raises(ValueError):
        logic.body_force(bad)


class _Sensor:
    """The matrices the hook and the drain read in turn: the previous step's last substep, then this step's four."""

    def __init__(self, forces):
        self.queue = [np.zeros((2, 2, 3)) for _ in forces]
        for m, f in zip(self.queue, forces):
            m[0, 0, 0] = f

    def __call__(self):
        return self.queue.pop(0)


def _step(window, forces):
    sensor = _Sensor(forces)
    window._read = sensor
    for _ in range(4):                     # the hook, before each of the four substeps
        window.sample()
    return window.drain()                  # after the physics: the step's last substep


def test_the_window_keeps_the_steps_own_four_substeps():
    window = logic.TouchWindow(lambda: None)
    # 99 N lived in the previous step's last substep: read again by this step's first hook call, and dropped
    touch = _step(window, [99.0, 1.0, 7.0, 2.0, 3.0])
    assert touch.samples == 4
    assert touch.peak_n == pytest.approx(7.0)
    assert touch.body_n == pytest.approx([7.0, 0.0])
    assert touch.source_n == pytest.approx([7.0, 0.0])


def test_a_touch_in_the_last_substep_is_read_by_the_drain():
    window = logic.TouchWindow(lambda: None)
    assert _step(window, [0.0, 0.0, 0.0, 0.0, 12.0]).peak_n == pytest.approx(12.0)


def test_the_next_step_starts_a_new_batch():
    window = logic.TouchWindow(lambda: None)
    _step(window, [0.0, 50.0, 0.0, 0.0, 0.0])
    assert _step(window, [0.0, 0.0, 0.0, 0.0, 0.0]).peak_n == 0.0


def test_a_reset_drops_a_partial_batch():
    window = logic.TouchWindow(lambda: None)
    window._read = _Sensor([0.0, 80.0, 80.0])
    window.sample()
    window.sample()
    window.sample()
    window.reset()
    assert _step(window, [80.0, 0.0, 0.0, 0.0, 0.0]).peak_n == 0.0      # the first read after the reset is dropped


def test_a_pinch_lets_the_box_and_the_hold_go():
    # the step's touch of a pinch: neither the box's gate nor the hold's misses it
    pinch = np.array([[[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    touch = logic.TouchWindow(lambda: pinch).drain()
    assert touch.body_n == pytest.approx([10.0, 0.0])
    assert touch.over(1e-6)                                            # the box's and the hold's threshold


def test_the_two_gates():
    touch = logic.Touch(body_n=np.array([0.0, 10.0]), source_n=np.zeros(3), samples=4)
    assert not touch.over(10.0)                                        # the hold's release: above 10 N
    assert logic.Touch(body_n=np.array([10.001]), source_n=np.zeros(1), samples=4).over(10.0)
    assert logic.Touch(body_n=np.array([2e-6]), source_n=np.zeros(1), samples=4).over(1e-6)   # the box's: any touch
    assert not logic.Touch(body_n=np.zeros(2), source_n=np.zeros(1), samples=4).over(1e-6)
    assert logic.Touch(body_n=np.zeros(0), source_n=np.zeros(0), samples=1).peak_n == 0.0


# ---- the person's trigger and walk ----

def test_the_triggers():
    assert not logic.trigger_due("never", 0.0, t=10.0, person_x=0.0, robot_x=0.0)
    assert logic.trigger_due("time", 0.0, t=0.0, person_x=0.0, robot_x=0.0)
    assert not logic.trigger_due("time", 1.5, t=1.48, person_x=0.0, robot_x=0.0)
    # robot_behind: the robot behind the person along +x by less than the distance, and not past them
    assert not logic.trigger_due("robot_behind", 1.0, t=0.0, person_x=2.0, robot_x=0.0)
    assert not logic.trigger_due("robot_behind", 1.0, t=0.0, person_x=2.0, robot_x=1.0)        # exactly 1.0 behind
    assert logic.trigger_due("robot_behind", 1.0, t=0.0, person_x=2.0, robot_x=1.01)
    assert logic.trigger_due("robot_behind", 1.0, t=0.0, person_x=2.0, robot_x=2.0)
    assert not logic.trigger_due("robot_behind", 1.0, t=0.0, person_x=2.0, robot_x=2.01)       # walked past
    with pytest.raises(ValueError):
        logic.trigger_due("soon", 0.0, t=0.0, person_x=0.0, robot_x=0.0)


def test_the_s1f_trigger_comes_at_the_rows_distance(boxes):
    entry = _entry("S1f")
    row = conditions.draw_row(entry, 0, boxes)
    plan = human_script.plan_for(entry, row)
    start_x = conditions.row_site(human_script.sites.load_site(entry), row).robot_start.x
    xs = start_x + np.arange(0.0, 12.0, 0.02)          # the robot walking +x in 2 cm steps
    first = next(x for x in xs if logic.trigger_due(plan.trigger.kind, plan.trigger.value, t=0.0,
                                                    person_x=plan.start_xy[0], robot_x=x))
    distance = conditions.s1f_trigger_distance(row)          # trigger_lead_s at the row's robot speed
    assert distance == pytest.approx(row["trigger_lead_s"] * row["robot_speed_mps"])
    assert plan.trigger.kind == "robot_behind" and plan.trigger.value == pytest.approx(distance)
    assert plan.start_xy[0] - first == pytest.approx(distance, abs=0.02)


def test_the_walk_has_its_length_left_and_none_past_it():
    d = np.array([0.6, -0.8])
    assert logic.walk_remaining([1.0, 2.0], [1.0, 2.0], d, 3.0) == pytest.approx(3.0)
    assert logic.walk_remaining(np.array([1.0, 2.0]) + 1.25 * d, [1.0, 2.0], d, 3.0) == pytest.approx(1.75)
    assert logic.walk_remaining(np.array([1.0, 2.0]) + 3.5 * d, [1.0, 2.0], d, 3.0) == 0.0
    assert logic.walk_remaining([9.0, 9.0], [1.0, 2.0], d, None) == math.inf
    assert logic.walk_remaining([9.0, 9.0], [1.0, 2.0], d, math.inf) == math.inf


@pytest.mark.parametrize("walk", [(1.0, 0.0), (-1.0, 0.0), (0.6, -0.8), (math.cos(1.2), -math.sin(1.2))])
def test_the_entry_angle_moves_the_target_along_the_walk(walk):
    # the command term moves the target by -(cos, sin)(angle) per metre
    a = walk_angle_rad(walk)
    assert (-math.cos(a), -math.sin(a)) == pytest.approx(walk, abs=1e-12)
    assert walk_angle_rad((3.0 * walk[0], 3.0 * walk[1])) == pytest.approx(a)


@pytest.mark.parametrize("bad", [(0.0, 0.0), (math.nan, 1.0)])
def test_an_entry_angle_needs_a_direction(bad):
    with pytest.raises(ValueError):
        walk_angle_rad(bad)


# ---- the person's turn on the spot (S1f) ----

def test_the_turn_takes_the_short_way_and_eases_in_and_out():
    assert logic.turn_steps(1.0, 0.02) == 50 and logic.turn_steps(0.0, 0.02) == 0 and logic.turn_steps(0.009, 0.02) == 0
    assert logic.turn_yaw_deg(90.0, -30.0, 0, 50) == 90.0 and logic.turn_yaw_deg(90.0, -30.0, 50, 50) == pytest.approx(-30.0)
    assert logic.turn_yaw_deg(90.0, -30.0, 25, 50) == pytest.approx(30.0)        # half way, through +x: the robot's heading
    assert logic.turn_yaw_deg(90.0, -150.0, 25, 50) == pytest.approx(150.0)      # the short way is through 180
    first = logic.turn_yaw_deg(90.0, -30.0, 1, 50) - 90.0
    middle = logic.turn_yaw_deg(90.0, -30.0, 26, 50) - logic.turn_yaw_deg(90.0, -30.0, 25, 50)
    assert abs(first) < 0.2 * abs(middle)                                        # slow at the ends
    assert logic.turn_yaw_deg(90.0, -30.0, 60, 50) == pytest.approx(-30.0)


def test_the_legs_mark_time_at_the_slowest_gait_after_a_short_rise():
    assert logic.turn_gait_speed(0, 0.3, 0.02) == pytest.approx(0.3 * 0.02 / 0.4)
    assert logic.turn_gait_speed(19, 0.3, 0.02) == pytest.approx(0.3) and logic.turn_gait_speed(40, 0.3, 0.02) == 0.3


@pytest.mark.parametrize("to", [-30.0, -150.0, 170.0, 60.0])
def test_the_facings_rate_is_the_derivative_of_its_eased_arc(to):
    # Against a central difference of turn_yaw_deg itself, at every step inside the turn (the difference's own error is
    # about (pi / n)^2 / 6 of the rate: 0.13 deg/s at the 188 deg/s peak of a 120 deg turn in 50 steps). The difference
    # is taken the short way: at k >= n the facing is to_deg itself, a full turn from the arc's last value if the arc
    # passes 180.
    n = 50
    yaws = [logic.turn_yaw_deg(90.0, to, k, n) for k in range(n + 1)]
    for k in range(1, n):
        central = ((yaws[k + 1] - yaws[k - 1] + 180.0) % 360.0 - 180.0) / (2.0 * DT)
        assert logic.turn_yaw_rate_deg_s(90.0, to, k, n, DT) == pytest.approx(central, abs=0.2), k


def test_the_facings_rate_takes_the_short_way_and_is_zero_outside_the_turn():
    n = 50
    # half way: the shorter arc times pi / (2 n dt), its sign the arc's -- clockwise to -30, through 180 to -150
    assert logic.turn_yaw_rate_deg_s(90.0, -30.0, 25, n, DT) == pytest.approx(-120.0 * math.pi / (2 * n * DT))
    assert logic.turn_yaw_rate_deg_s(90.0, -150.0, 25, n, DT) == pytest.approx(120.0 * math.pi / (2 * n * DT))
    assert all(logic.turn_yaw_rate_deg_s(90.0, -30.0, k, n, DT) < 0.0 for k in range(1, n))
    for k in (-3, 0, n, n + 7):
        assert logic.turn_yaw_rate_deg_s(90.0, -30.0, k, n, DT) == 0.0
    with pytest.raises(ValueError):
        logic.turn_yaw_rate_deg_s(90.0, -30.0, 1, 0, DT)


def test_the_plan_and_the_command_count_the_turn_with_one_integer(boxes):
    # A turn shorter than half a step is no turn for the command (turn_steps 0) and none in the plan's walk either: the
    # walk sets off from rest at the trigger, not from the slowest gait.
    still = human_script.HumanPlan((0.0, 0.0), 90.0, (1.0, 0.0), 1.2, human_script.Trigger("time", 0.0), 10.0)
    from_rest = [human_script.walked(still, k * DT) for k in range(1, 80)]
    for turn_s in (0.0, 0.009):
        plan = dataclasses.replace(still, turn_time_s=turn_s)
        assert logic.turn_steps(plan.turn_time_s, DT) == 0
        assert [human_script.walked(plan, k * DT) for k in range(1, 80)] == from_rest
    # every canon S1f row turns: its walk waits the command's steps and sets off on the next
    entry = _entry("S1f")
    for index in range(8):
        plan = human_script.plan_for(entry, conditions.draw_row(entry, index, boxes))
        n = logic.turn_steps(plan.turn_time_s, DT)
        assert n == round(plan.turn_time_s / DT) > 0
        assert human_script.walked(plan, n * DT) == 0.0 < human_script.walked(plan, (n + 1) * DT)


# ---- when the task comes ----

@pytest.mark.parametrize("sid", ["S1e", "S1f"])
def test_the_task_comes_at_once_on_a_path_that_does_not_wait(sid, boxes):
    entry = _entry(sid)
    row = conditions.draw_row(entry, 0, boxes)
    plan = human_script.plan_for(entry, row)
    kind = human_script.sites.load_site(entry).path["kind"]
    assert logic.task_starts(kind, plan, pelvis_xy=(50.0, 50.0), robot_xy=(0.0, 0.0), distance=None)


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_s1g_task_comes_when_the_person_is_within_the_rows_distance(index, boxes):
    # The person walked by the gait drive from its start (human_script.walked): the task comes run_up_s after they
    # set off, with the person the row's trigger distance from the robot along the aisle.
    entry = _entry("S1g")
    row = conditions.draw_row(entry, index, boxes)
    plan = human_script.plan_for(entry, row)
    site = conditions.row_site(human_script.sites.load_site(entry), row)
    robot = (site.robot_start.x, site.robot_start.y)
    distance = conditions.s1g_trigger_distance(row)
    start = np.asarray(plan.start_xy)
    walk = np.asarray(plan.walk_dir)
    first = None
    for k in range(1, 2000):
        pelvis = start + human_script.walked(plan, k * DT) * walk
        if logic.task_starts(site.path["kind"], plan, pelvis_xy=pelvis, robot_xy=robot, distance=distance):
            first = k
            break
    assert first is not None
    assert first * DT == pytest.approx(row["run_up_s"], abs=DT + 1e-9)
    along = float(np.dot(np.asarray(robot) - pelvis, walk))
    assert distance - row["human_speed_mps"] * DT <= along <= distance


def test_s1g_task_needs_its_distance(boxes):
    entry = _entry("S1g")
    row = conditions.draw_row(entry, 0, boxes)
    with pytest.raises(ValueError):
        logic.task_starts("turn_then_walk", human_script.plan_for(entry, row), pelvis_xy=(0.0, 0.0),
                          robot_xy=(0.0, 0.0), distance=None)


# ---- the first approach line, the seeing and the behaviour ----

def test_the_first_approach_line_is_the_root_to_the_pelvis():
    assert logic.first_direction([1.0, 2.0, 1.0], [4.0, 6.0, 1.0]) == pytest.approx([0.6, 0.8, 0.0])
    with pytest.raises(ValueError):
        logic.first_direction([1.0, 2.0, 1.0], [1.0, 2.0, 1.0])


def test_seen_needs_one_camera_with_both_the_share_and_the_height():
    rule = {"min_visible_fraction": 0.2, "min_height_px": 50.0, "sample_points": 64}
    assert not logic.any_seen({"a": 0.9, "b": 0.1}, {"a": 40.0, "b": 200.0}, rule)    # each lacks one of the two
    assert logic.any_seen({"a": 0.9, "b": 0.2}, {"a": 40.0, "b": 50.0}, rule)
    with pytest.raises(ValueError):
        logic.any_seen({"a": 0.9}, {"b": 60.0}, rule)


def test_the_perceived_person_is_relative_to_the_robot():
    assert logic.perceived(False, pelvis_xy=(1, 2), pelvis_vel_xy=(0, 0), robot_xy=(0, 0), robot_vel_xy=(0, 0)) is None
    p = logic.perceived(True, pelvis_xy=(3.0, 1.0), pelvis_vel_xy=(-1.0, 0.5), robot_xy=(1.0, 2.0),
                        robot_vel_xy=(0.5, 0.0))
    assert p == Perceived(rel_xy=(2.0, -1.0), rel_vel_xy=(-1.5, 0.5))


def test_a_behaviour_answers_two_finite_numbers():
    assert logic.behavior_command((0.5, -0.25)) == (0.5, -0.25)
    assert logic.behavior_command([np.float32(0.5), 0]) == (0.5, 0.0)
    for bad in [(1.0,), (1.0, 2.0, 3.0), (True, 0.0), (math.nan, 0.0), "ab", None]:
        with pytest.raises(ValueError):
            logic.behavior_command(bad)


# ---- the Isaac side, from its source ----

def test_s1d_keeps_its_trigger_and_stop_in_the_base_term():
    src = BASE.read_text(encoding="utf-8")
    trigger = src[src.index("    def _trigger_latched"):src.index("    def _stop_distance")]
    assert ("should_trigger = (dist_to_crossing >= 0.0) & (dist_to_crossing < entry[\"trigger_distance\"])"
            in trigger)
    assert "dist_to_crossing = entry[\"x_cross\"] - robot_x" in trigger
    stop = src[src.index("    def _stop_distance"):src.index("    def _plane_gains")]
    assert 'if entry.get("stop_y") is None:' in stop and "gt.remaining_to_stop(" in stop
    assert "        moving = self._trigger_latched(entry, robot)\n" in src
    assert "        remaining = self._stop_distance(entry)\n" in src


def test_the_scripted_entry_has_no_far_side_stop_line():
    src = EVENTS.read_text(encoding="utf-8")
    block = src[src.index("def reset_dummy_scripted"):]
    assert '"stop_y"' not in block                  # the base term would clamp the walk on y
    assert 'entry["face_travel"] = True' in block
    assert "walk_angle_rad(plan.walk_dir)" in block


def test_the_environment_filters_the_dummy_sensor_and_uses_the_task_command():
    src = HRI_CFG.read_text(encoding="utf-8")
    assert ("self.scene.dummy_contact.filter_prim_paths_expr = command_logic.contact_filters(\n"
            "            list(visibility.load_link_boxes()), carry_rig.BOX_PRIM if carried else None)") in src
    assert "func=dummy_reset.stand_up_first(reset_dummy_scripted)" in src
    assert "params={\"plan\": human_script.plan_record(" in src
    assert "HriTaskCommandCfg(**kept, scenario_id=self.scenario_id, row=dict(self.row))" in src


def test_the_command_term_gates_on_the_matrix_and_drops_s1ds_supervisor():
    tree = ast.parse(COMMANDS.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HriTaskCommand")
    methods = {n.name: ast.get_source_segment(COMMANDS.read_text(encoding="utf-8"), n)
               for n in cls.body if isinstance(n, ast.FunctionDef)}
    update = methods["_update_command"]
    # no S1d speed scaling, protective stop or heading correction
    for s1d in ("command_scale", "protective_stop_step", "_heading_correction", "separation_required("):
        assert s1d not in update
    # the hold's gate reads the touch, never the robot's net force, at the box's threshold -- not S1d's backstop
    assert "self.touch.over(self.cfg.release_threshold_n)" in methods["_robot_contact_present"]
    assert "contact_forces" not in methods["_robot_contact_present"]
    assert "robot_contact_threshold_n" not in methods["_robot_contact_present"]
    # the box is let go at the first touch, after the geometry of the step it was held in
    i_geometry = update.index("self.hri = self._geometry(frame)")
    i_release = update.index("rig.release()")
    assert i_geometry < i_release
    assert "self.touch.over(self.cfg.release_threshold_n)" in update


def test_each_episode_starts_without_a_turn_and_a_release_ends_one():
    src = COMMANDS.read_text(encoding="utf-8")
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "HriTaskCommand")
    methods = {n.name: ast.get_source_segment(src, n) for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert "self._turn_from_step = -1" in methods["_new_episode"] and "self._turn_k = -1" in methods["_new_episode"]
    # released, the base term returns before the script -- _trigger_latched is not called -- so the turn ends here
    advance = methods["_advance_dummy"]
    assert advance.index("self._turn_k = -1") < advance.index("super()._advance_dummy()")
    base = BASE.read_text(encoding="utf-8")
    block = base[base.index("    def _advance_dummy"):base.index("    def _trigger_latched")]
    assert block.index("        if bool(self.hold_released.all()):\n") < block.index(
        "        moving = self._trigger_latched(entry, robot)\n")


def _class_defaults(path: Path, cls_name: str) -> dict:
    """{field: its default's source} of a config class, read from the syntax tree (the module imports Isaac)."""
    src = path.read_text(encoding="utf-8")
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return {n.target.id: ast.get_source_segment(src, n.value) for n in cls.body
            if isinstance(n, ast.AnnAssign) and n.value is not None}


def test_the_hold_is_let_go_at_the_first_touch_only():
    # no distance gate in the conditions-table scenarios: the environment sets the copied S1d gap to None ...
    assert ("kept.update(walk_speed_range=(0.0, 0.0), heading_correction_gain=0.0, dummy_release_gap_m=None)"
            in HRI_CFG.read_text(encoding="utf-8"))
    # ... the base term reads None as no gate, and its contact backstop is the HRI term's touch
    base = BASE.read_text(encoding="utf-8")
    hold = base[base.index("    def _update_hold_release"):base.index("    def _robot_contact_present")]
    assert "near = (measured <= gap) if gap is not None else torch.zeros_like(self.hold_released)" in hold
    assert "newly = (near | self._robot_contact_present()) & (~self.hold_released)" in hold
    # the hold, the box and the first contact_onset go at one step: the same threshold, the recorder's noise floor
    assert _class_defaults(COMMANDS, "HriTaskCommandCfg")["release_threshold_n"] == "CONTACT_EPS_N"
    assert "from generation.envs.s1a.recorder import CONTACT_EPS_N" in COMMANDS.read_text(encoding="utf-8")


def test_s1ds_gate_and_backstop_keep_their_defaults():
    defaults = _class_defaults(BASE, "S1aStraightWalkCommandCfg")
    assert defaults["dummy_release_gap_m"] == "0.10"
    assert defaults["robot_contact_threshold_n"] == "10.0"


@pytest.mark.parametrize("sid", ["S1e", "S1f", "S1g"])
def test_a_plan_travels_as_plain_data_and_comes_back_the_same(sid, boxes):
    # An environment config copies its event parameters and sets their fields, which a frozen plan refuses.
    entry = _entry(sid)
    plan = human_script.plan_for(entry, conditions.draw_row(entry, 3, boxes))
    record = human_script.plan_record(plan)
    assert isinstance(record, dict) and isinstance(record["trigger"], dict)
    assert human_script.plan_from_record(record) == plan
    assert human_script.plan_from_record(json.loads(json.dumps(record))) == plan       # lists for tuples too


# ---- where the dummy is created ----
# A dummy that stands at a spawn place of its own while the simulator starts, and is moved by the first reset of a
# process across the warehouse to the row's start, is thrown by the first physics substep (S1g row 37: 10 kN under
# one foot, 14 steps in the air, 20 deg of tilt; created at the start, 0.48/0.49 kN and 4 deg, measured
# 2026-10-20). The dummy is created where its script parks it.

@pytest.mark.parametrize("sid, index", [("S1e", 20), ("S1f", 12), ("S1g", 37), ("S1g", 2)])
def test_the_dummy_is_created_where_its_script_parks_it(sid, index):
    from generation.envs.s1a.mdp.entry_geometry import facing_quat_xyzw

    entry = conditions.scenario_entry(sid)
    plan = human_script.plan_for(entry, conditions.draw_row(entry, index, layout.build_layout(
        layout.load_spec(layout.SCENE_SPEC))))
    pos, rot = logic.spawn_pose(plan)
    assert pos == (float(plan.start_xy[0]), float(plan.start_xy[1]), 0.0)
    assert rot == tuple(float(v) for v in facing_quat_xyzw(plan.yaw_deg))
    x, y, z, w = rot                                  # +x of the body turned to the facing
    forward = (1 - 2 * (y * y + z * z), 2 * (x * y + z * w))
    yaw = math.radians(plan.yaw_deg)
    assert forward == pytest.approx((math.cos(yaw), math.sin(yaw)), abs=1e-6)


def test_the_environment_creates_the_dummy_at_the_rows_start():
    src = HRI_CFG.read_text(encoding="utf-8")
    assert "self.scene.dummy.init_state.pos, self.scene.dummy.init_state.rot = command_logic.spawn_pose(plan)" in src
    assert src.index("plan = human_script.plan_for(entry, self.row)") < src.index("command_logic.spawn_pose(plan)")


def _hri_method(name):
    src = COMMANDS.read_text(encoding="utf-8")
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "HriTaskCommand")
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_a_person_who_is_not_turning_gets_the_base_terms_gait_drive():
    # Without the turn the command's gait drive is the base term's: the early return on `_turn_k < 0` is the method's
    # first statement after its docstring, and nothing else returns before it.
    body = _hri_method("_gait_drive").body
    first = body[1]
    assert isinstance(body[0], ast.Expr) and isinstance(first, ast.If)
    assert ast.unparse(first.test) == "self._turn_k < 0"
    assert [ast.unparse(n) for n in first.body] == ["return super()._gait_drive(target_dxy)"]
    assert not first.orelse


def test_a_person_who_is_not_turning_adds_nothing_to_the_holds_torque():
    # Without the turn the hold's torque is its spring and damper alone: the early return of None on `_turn_k < 0` is
    # the method's first statement after its docstring, and the turning torque is written about z alone.
    method = _hri_method("_facing_turn_torque")
    first = method.body[1]
    assert isinstance(method.body[0], ast.Expr) and isinstance(first, ast.If)
    assert ast.unparse(first.test) == "self._turn_k < 0"
    assert [ast.unparse(n) for n in first.body] == ["return None"]
    assert not first.orelse
    src = ast.unparse(method)
    assert "self._turn_k + 1" in src           # the facing _trigger_latched wrote this step: the turn's k + 1
    assert "torque[:, 2] = ang_damping[:, 2] * math.radians(" in src


def test_a_plan_without_a_turn_writes_neither_the_facing_nor_the_gait_speed():
    # A plan whose turn has no steps returns the latch before the first write to facing_quat / gait_speed.
    method = _hri_method("_trigger_latched")
    guard = ast.unparse(ast.parse('n <= 0 or not bool(entry["triggered"][0])', mode="eval"))
    at = next(i for i, n in enumerate(method.body) if isinstance(n, ast.If) and ast.unparse(n.test) == guard)
    assert [ast.unparse(n) for n in method.body[at].body] == [ast.unparse(ast.parse('return entry["triggered"]'))]
    assert ast.unparse(method.body[at - 1]) == "n = logic.turn_steps(self.plan.turn_time_s, self._env.step_dt)"
    for name in ("facing_quat", "gait_speed", "_turn_k"):
        assert not any(name in ast.unparse(n) for n in method.body[1:at + 1]), name   # [0] is the docstring


# ---- the task's stages, through one step of the command term ----

def _task_term(sid, boxes):
    """HriTaskCommand's _new_episode and _update_command compiled from their source onto stand-ins for the rest of the
    term, at the episode's first step: the robot at the row's start, the person at the script's, nothing touched and
    nobody seen. Needs torch (the step writes the command into a tensor)."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace as NS

    from generation import behavior, sites, task_path

    src = COMMANDS.read_text(encoding="utf-8")
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "HriTaskCommand")
    methods = [ast.get_source_segment(src, n, padded=True) for n in cls.body
               if isinstance(n, ast.FunctionDef) and n.name in ("_new_episode", "_update_command")]
    assert len(methods) == 2

    class Base:
        num_envs, device = 1, "cpu"
        dummy_lin_force_applied = dummy_torque_applied = None

        def _ensure_window(self):
            pass

        def _episode_step(self):
            return 0

        def _read_frame(self):
            return self.frame

        def _geometry(self, frame):
            return NS(direction=np.array([1.0, 0.0]), distance=5.0, s_p=1.0)

        def _camera_view(self, frame):
            return {}, False

        def _update_hold_release(self, measured):
            pass

        def _rig(self):
            return None

        def _advance_dummy(self):
            pass

    ns = {"Base": Base, "torch": torch, "math": math, "np": np, "task_path": task_path, "logic": logic,
          "behavior_layer": behavior}
    exec(compile("class Task(Base):\n" + "\n\n".join(methods) + "\n", str(COMMANDS), "exec"), ns)
    entry = _entry(sid)
    row = conditions.draw_row(entry, 0, boxes)
    site = conditions.row_site(sites.load_site(entry), row)
    plan = human_script.plan_for(entry, row)
    t = ns["Task"]()
    t.__dict__.update(
        site=site, row=row, plan=plan, behavior=behavior.PassThrough(), cfg=NS(release_threshold_n=1e-6),
        _env=NS(step_dt=DT), _command=torch.zeros(1, 3),
        _start_distance=conditions.s1g_trigger_distance(row) if site.path["kind"] == logic.WAITING_KIND else None,
        _window=NS(drain=lambda: NS(over=lambda threshold: False)),
        frame={"origin": np.zeros(3), "root": np.array([site.robot_start.x, site.robot_start.y, 1.0]),
               "pelvis": np.array([*plan.start_xy, 1.0]), "root_vel": np.zeros(6),
               "yaw": math.radians(site.robot_start.yaw_deg), "pelvis_vel": np.zeros(6)})
    t._new_episode()
    return t, task_path


@pytest.mark.parametrize("sid", ["S1e", "S1f", "S1g"])
def test_after_one_step_the_command_term_holds_the_scenarios_stages(sid, boxes):
    t, task_path = _task_term(sid, boxes)
    kind = t.site.path["kind"]
    first = t.stages
    t._update_command()
    assert t.stages is first and t.stages.names == task_path.STAGE_NAMES[kind]
    assert t.stages.end_steps == [-1] * len(task_path.STAGE_NAMES[kind])       # nothing has ended at the first step
    # S1e's and S1f's task comes at the first step; S1g's waits for the person
    assert t.follower.started_step == (-1 if kind == logic.WAITING_KIND else 0)
    t._new_episode()
    assert t.stages is not first and t.stages.names == first.names            # a new tracker every episode
