"""The recorder of the scenarios that declare R-TS15066-SSM-02 (S1e, S1f, S1g): which contact columns count, the
rack-contact event, the step's hri columns, the sidecar's members and the dummy's degenerate verdict. The arithmetic
lives in generation/envs/hri/record_logic.py and the per-substep window in generation/envs/s1a/substep_contact.py; the
recorder and the command term import Isaac, so the lines that wire them are pinned from the source. Isaac-free."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from generation import conditions, directed_separation, hri_geometry, human_script, sites, visibility
from generation.envs.hri import record_logic as rl
from generation.envs.s1a.substep_contact import HriContactAccumulator
from generation.field_map import HRI, active_step_fields, hri_recorded
from generation.scene import layout

ROOT = Path(__file__).resolve().parents[2]
RECORDER = (ROOT / "generation/envs/s1a/recorder.py").read_text(encoding="utf-8")
COMMANDS = (ROOT / "generation/envs/hri/commands.py").read_text(encoding="utf-8")
RUN = (ROOT / "generation/envs/run_s1a.py").read_text(encoding="utf-8")
SCENARIOS = json.loads((ROOT / "meta/scenarios.json").read_text(encoding="utf-8"))
RULES = json.loads((ROOT / "meta/rules.json").read_text(encoding="utf-8"))
SCHEMA = json.loads((ROOT / "meta/safety_schema.json").read_text(encoding="utf-8"))
EPS = 1e-6
PIN = 0.20


def _boxes():
    return layout.build_layout(layout.load_spec(layout.SCENE_SPEC))


# ---- a recording with no step ----

def test_a_zero_step_hri_recording_carries_every_hri_key(tmp_path):
    from generation import convert_lerobot as cl

    expected = [f.npz for f in active_step_fields((HRI,)) if f.condition == HRI and f.kind == "array"]
    assert list(rl.hri_zero_step_keys()) == expected and len(expected) == 10
    hri = rl.zero_step_arrays(hri=True)
    plain = rl.zero_step_arrays(hri=False)
    assert set(expected) <= set(hri) and set(plain) <= set(hri)
    assert all(a.shape == (0,) for a in hri.values())
    assert hri_recorded(list(hri)) is True and hri_recorded(list(plain)) is False
    # the keys S1a and S1d write, in their order
    assert list(plain) == list(rl.PLAIN_ZERO_STEP_KEYS)
    # the converter reads the empty hri recording as hri, not half of one, for a scenario that declares the rule
    np.savez(tmp_path / "e.npz", **hri)
    (tmp_path / "e.json").write_text(json.dumps({"episode_id": "S1e_0000_seed1", "scenario_id": "S1e"}))
    by_sid = {s["scenario_id"]: s for s in SCENARIOS["scenarios"]}
    assert cl.check_hri_declared([tmp_path / "e.npz"], by_sid) is None


def test_the_recorder_saves_the_empty_arrays_of_its_mode():
    assert "arrays = zero_step_arrays(hri=self.hri)" in RECORDER
    assert "self.hri = records_hri(scenario_id)" in RECORDER


# ---- the force matrix: what the robot and the box gave the person ----

# Columns: two robot links, a floor and a rack (were the sensor filtered against them), the box.
LINK_A, LINK_B, FLOOR, RACK, BOX = range(5)


def _matrix(*entries, bodies=4, filters=5):
    m = np.zeros((bodies, filters, 3))
    for body, col, force in entries:
        m[body, col] = force
    return m


def test_contact_force_counts_robot_and_box_only():
    m = _matrix((0, LINK_A, (30.0, 0.0, 0.0)), (0, FLOOR, (0.0, 0.0, 700.0)),
                (1, RACK, (0.0, 50.0, 0.0)), (2, BOX, (0.0, 0.0, -12.0)), (2, LINK_B, (5.0, 0.0, 0.0)),
                (3, FLOOR, (0.0, 0.0, 400.0)))
    got = rl.attribute_contacts(m, [LINK_A, LINK_B], BOX)
    assert got == pytest.approx([30.0, 0.0, np.hypot(12.0, 5.0), 0.0])
    # opposite forces from the robot and the box on one body -- a pinch -- do not cancel: one side's push
    assert rl.attribute_contacts(_matrix((0, LINK_A, (8.0, 0, 0)), (0, BOX, (-8.0, 0, 0))), [LINK_A], BOX)[0] == 8.0
    # nor do two robot links; the floor's column is still left out
    pinch = _matrix((0, LINK_A, (10.0, 0, 0)), (0, LINK_B, (-10.0, 0, 0)), (0, FLOOR, (0.0, 0.0, 700.0)))
    assert rl.attribute_contacts(pinch, [LINK_A, LINK_B], BOX) == pytest.approx([10.0, 0.0, 0.0, 0.0])
    assert rl.touched(rl.attribute_contacts(pinch, [LINK_A, LINK_B], BOX), EPS)
    # no box: its column is not read
    assert rl.attribute_contacts(m, [LINK_A, LINK_B], None) == pytest.approx([30.0, 0.0, 5.0, 0.0])
    with pytest.raises(ValueError, match="열"):
        rl.attribute_contacts(m, [LINK_A, LINK_A], BOX)
    with pytest.raises(ValueError, match="열"):
        rl.attribute_contacts(m, [LINK_A], 7)


def test_the_recorder_and_the_command_term_read_the_matrix_with_one_function(monkeypatch):
    # the box and the hold are let go on the command term's touch and contact_onset is the recorder's: one function
    # (command_logic.body_force) reads the matrix for both, so the two cannot disagree on what a body got
    from generation.envs.hri import command_logic

    calls = []

    def spy(matrix):
        calls.append(np.asarray(matrix).shape)
        return np.full(np.asarray(matrix).shape[0], 42.0)

    monkeypatch.setattr(command_logic, "body_force", spy)
    m = _matrix((0, LINK_A, (1.0, 0, 0)))
    assert rl.attribute_contacts(m, [LINK_A, LINK_B], BOX) == pytest.approx([42.0] * 4)
    assert command_logic.TouchWindow(lambda: m).drain().body_n == pytest.approx([42.0] * 4)
    assert calls == [(4, 3, 3), (4, 5, 3)]          # the recorder's robot and box columns; the term's every filter


def test_a_pinch_is_recorded_as_one_sides_push_in_the_step_window():
    pinch = _matrix((1, LINK_A, (0.0, 10.0, 0.0)), (1, LINK_B, (0.0, -10.0, 0.0)))
    zero = _matrix()
    acc = _accumulator([_reading(zero), _reading(zero), _reading(pinch), _reading(zero), _reading(zero)])
    for _ in range(4):
        acc.sample()
    out = acc.drain()
    assert out["human_force"] == pytest.approx([0.0, 10.0, 0.0, 0.0])
    assert rl.touched(out["human_force"], EPS)


def test_only_the_robots_and_the_boxs_force_is_typed():
    # bodies: head, pelvis, a foot, the shin; the shin lies on the floor (z 0.1) carrying only the floor's force
    codes = ["BR-01", "BR-07", None, "BR-12"]
    floor_only = np.array([0.0, 0.0, 0.0, 0.0])
    regions, force, z, types = rl.region_contacts(floor_only, [1.6, 0.9, 0.05, 0.1], codes,
                                                  contact_eps_n=EPS, pin_height_m=PIN)
    assert regions == ["BR-01", "BR-07", "BR-12"] and types == ["none"] * 3
    assert not rl.touched(floor_only, EPS)
    # the robot presses the fallen shin on the floor: quasistatic; a pelvis knock standing: transient
    regions, force, z, types = rl.region_contacts([0.0, 40.0, 0.0, 220.0], [1.6, 0.9, 0.05, 0.1], codes,
                                                  contact_eps_n=EPS, pin_height_m=PIN)
    assert dict(zip(regions, types)) == {"BR-01": "none", "BR-07": "transient", "BR-12": "quasistatic"}
    assert force == pytest.approx([0.0, 40.0, 220.0])
    # a foot touched alone is a touch with no region and no type
    regions, force, z, types = rl.region_contacts([0.0, 0.0, 15.0, 0.0], [1.6, 0.9, 0.05, 0.4], codes,
                                                  contact_eps_n=EPS, pin_height_m=PIN)
    assert types == ["none"] * 3 and rl.touched([0.0, 0.0, 15.0, 0.0], EPS)


def _accumulator(readings, *, box=True, expected=4):
    """readings: one (matrix, z, robot_net, box_net) per hook call, the last one read by drain()."""
    it = iter(readings)
    cur = {}

    def matrix():
        cur["r"] = next(it)
        return cur["r"][0]

    return HriContactAccumulator(matrix, lambda: cur["r"][1], lambda: cur["r"][2],
                                 (lambda: cur["r"][3]) if box else None, robot_cols=[LINK_A, LINK_B],
                                 box_col=BOX if box else None, link_cols=[LINK_A, LINK_B, None],
                                 expected_samples=expected)


def _reading(m, z=(1.6, 0.9, 0.05, 0.4), robot=((0, 0, 0), (0, 0, 0), (0, 0, 500.0)), box=(0, 0, 0)):
    return (m, np.asarray(z, float), np.asarray(robot, float), np.asarray(box, float))


def test_the_matrix_window_is_this_steps_four_substeps_with_the_height_at_each_peak():
    zero = _matrix()
    knock = _matrix((1, LINK_A, (0.0, 90.0, 0.0)))
    readings = [_reading(knock),                       # the previous step's last substep: dropped
                _reading(zero), _reading(_matrix((1, LINK_A, (0.0, 30.0, 0.0))), z=(1.6, 0.7, 0.05, 0.4)),
                _reading(knock, z=(1.6, 0.5, 0.05, 0.4)), _reading(zero)]
    acc = _accumulator(readings)
    for _ in range(4):
        acc.sample()
    out = acc.drain()
    assert out["samples"] == 4
    assert out["human_force"] == pytest.approx([0.0, 90.0, 0.0, 0.0])
    assert out["human_z_at_peak"][1] == pytest.approx(0.5)


def test_a_window_with_the_wrong_number_of_substeps_is_refused():
    acc = _accumulator([_reading(_matrix())] * 4)
    for _ in range(3):
        acc.sample()
    with pytest.raises(RuntimeError, match="표본이 3개"):
        acc.drain()


def test_what_a_robot_link_takes_from_something_else_is_its_net_less_the_persons_share():
    # link A pushes the pelvis with 60 N along +y (the person pushes back: -60 on the link) and a rack pushes the link
    # with 25 N along -x; link B touches nothing; the foot (no column) stands on the floor
    m = _matrix((1, LINK_A, (0.0, 60.0, 0.0)))
    robot = ((-25.0, -60.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 500.0))
    links = rl.link_residuals(robot, m, [LINK_A, LINK_B, None])
    assert links == pytest.approx([25.0, 0.0, 500.0])
    # the box: pressed on the person (the person's force on it is the opposite of M) and on nothing else
    m_box = _matrix((0, BOX, (0.0, 0.0, -12.0)))
    assert rl.box_residual((0.0, 0.0, 12.0), m_box, BOX) == pytest.approx(0.0)
    assert rl.box_residual((0.0, 0.0, 30.0), m_box, BOX) == pytest.approx(18.0)


def test_rack_contact_is_an_edge():
    non_foot = [True, True, False]
    feet_on_floor = [0.0, 0.0, 800.0]
    # a foot carrying the floor is not a rack contact; a link is; the box only while held
    assert rl.rack_residual(feet_on_floor, non_foot, None, box_held=False) == 0.0
    assert rl.rack_residual([0.0, 6.5, 800.0], non_foot, None, box_held=False) == 6.5
    assert rl.rack_residual(feet_on_floor, non_foot, 300.0, box_held=True) == 300.0
    assert rl.rack_residual(feet_on_floor, non_foot, 300.0, box_held=False) == 0.0
    assert rl.RACK_CONTACT_THRESHOLD_N == 5.0
    prev, edges = False, []
    for residual in [0.0, 6.0, 7.0, 2.0, 5.0, 8.0, 9.0]:
        prev, edge = rl.rack_contact_edge(prev, residual)
        edges.append(edge)
    assert edges == [False, True, False, False, False, True, False]     # 5.0 is not over 5 N


def test_the_matrix_columns_are_matched_by_name():
    filters = ["{ENV_REGEX_NS}/Robot/torso_link", "{ENV_REGEX_NS}/Robot/left_ankle_link",
               "{ENV_REGEX_NS}/Robot/pelvis", "{ENV_REGEX_NS}/Box"]
    robot_cols, box_col, link_cols = rl.matrix_columns(
        filters, ["pelvis", "left_ankle_link", "torso_link"], "Box", [True, False, True])
    assert link_cols == [2, 1, 0] and robot_cols == [0, 1, 2] and box_col == 3
    with pytest.raises(ValueError, match="torso_link"):
        rl.matrix_columns(filters[1:], ["pelvis", "left_ankle_link", "torso_link"], "Box", [True, False, True])
    with pytest.raises(ValueError, match="상자"):
        rl.matrix_columns(filters[:3], ["pelvis", "left_ankle_link", "torso_link"], "Box", [True, False, True])
    # a foot without a column is allowed: its net force is the floor's
    assert rl.matrix_columns(filters[::2], ["pelvis", "left_ankle_link", "torso_link"], None,
                             [True, False, True])[2] == [1, None, 0]


# ---- the typed steps and the first onset ----

def _episode(forces, codes):
    """(per step: typed, onset edge) as the recorder computes them from each body's peak force."""
    prev, rows = False, []
    for f in forces:
        _, _, _, types = rl.region_contacts(f, np.full(len(codes), 1.0), codes, contact_eps_n=EPS, pin_height_m=PIN)
        now = rl.touched(f, EPS)
        rows.append((any(t != "none" for t in types), now and not prev))
        prev = now
    return rows


def test_typed_steps_follow_the_first_onset():
    codes = ["BR-01", "BR-07", "BR-09", "BR-09", None, None]
    forces = [np.zeros(6), np.array([0, 0, 0, 0, 20.0, 0]), np.array([0, 0, 35.0, 0, 20.0, 0]), np.zeros(6),
              np.array([0, 0, 0, 12.0, 0, 0])]
    rows = _episode(forces, codes)
    assert [edge for _, edge in rows] == [False, True, False, False, True]      # a foot first, then again later
    assert [typed for typed, _ in rows] == [False, False, True, False, True]
    rng = np.random.default_rng(7)
    for _ in range(200):
        seq = [np.where(rng.random(6) < 0.15, rng.uniform(0.0, 50.0, 6), 0.0) for _ in range(30)]
        rows = _episode(seq, codes)
        onsets = [k for k, (_, edge) in enumerate(rows) if edge]
        typed = [k for k, (t, _) in enumerate(rows) if t]
        assert not typed or (onsets and min(typed) >= onsets[0])


# ---- the step's hri columns and S_p from the shipped columns ----

def _sphere(r):
    return hri_geometry.Shape("sphere", (r,), None)


def _obb(centre, half):
    return visibility.Obb(center=np.asarray(centre, float), rot_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                          half=np.asarray(half, float))


def _s1e_row():
    entry = conditions.scenario_entry("S1e")
    row = conditions.draw_row(entry, 0, _boxes())
    return entry, row


def test_sp_can_be_recomputed_from_the_shipped_columns():
    params, table = directed_separation.load_directed_ssm(RULES)
    entry, row = _s1e_row()
    mass = float(row["box_mass_kg"])
    site = conditions.row_site(sites.load_site(entry), row)
    condition = json.loads(json.dumps(rl.condition_record(0, row, site, human_script.plan_for(entry, row),
                                                          task_started_step=0)))
    ssm = json.loads(json.dumps(rl.hri_ssm_params(params, table)))
    dt, v, w = 0.02, 0.8, 0.15
    fractions = {cam: 0.5 for cam in rl.fraction_cameras()}
    columns, root_vel, onsets = [], [], []
    last, released, prev_touch = None, False, False
    for k in range(80):
        x_r, x_h = v * k * dt, 1.6 - 0.5 * k * dt
        link = _obb((x_r, 0.0, 1.0), (0.1, 0.2, 0.4))
        box = _obb((x_r + 0.35, 0.0, 1.0), (0.2, 0.14, 0.1))
        states = [((x_r, 0.0, 1.0), (v, 0.0, 0.0), (0.0, 0.0, w))]
        box_state = ((x_r + 0.35, 0.0, 1.0), (v, 0.0, 0.0), (0.0, 0.0, w))
        human = [(_sphere(0.1), np.array([x_h, 0.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0]))]
        human_states = [((x_h, 0.0, 1.0), (-0.5, 0.0, 0.0), (0.0, 0.0, 0.0))]
        rv = np.array([v, 0.0, 0.0, 0.0, 0.0, w])
        if last is None:
            last = np.array([1.0, 0.0, 0.0])
        h = hri_geometry.hri_step(robot_obbs=[link], robot_states=states, box_obb=box, box_state=box_state,
                                  released=released, human=human, human_states=human_states,
                                  robot_speed=float(np.hypot(rv[0], rv[1])), yaw_rate=float(rv[5]), box_mass_kg=mass,
                                  params=params, table=table, last_direction=last)
        last = h.direction
        touch = h.distance <= hri_geometry.TOUCHING_M
        onsets.append(touch and not prev_touch)
        prev_touch = touch
        if touch and not released:
            released = True          # let go after this step's geometry, as the command term does
        columns.append(rl.hri_step_columns(h, pelvis_pose_xyzw=[x_h, 0, 1, 0, 0, 0, 1],
                                           pelvis_vel=[-0.5, 0, 0, 0, 0, 0], fractions=fractions))
        root_vel.append(rv)
    first = onsets.index(True)
    assert 5 < first < 70

    def recompute(state_of):
        out = []
        for k, (col, rv) in enumerate(zip(columns, root_vel)):
            out.append(directed_separation.separation_distance(
                {n: ssm[n] for n in directed_separation.DIRECTED_SSM_PARAMETERS}, ssm["stop_table"],
                state=state_of(k), robot_speed=float(np.hypot(rv[0], rv[1])), yaw_rate=float(rv[5]),
                v_r=col["approach_speed_robot"], v_h=col["approach_speed_human"]))
        return np.array(out)

    held = directed_separation.carry_state(ssm["stop_table"], condition["box_mass_kg"])
    shipped = np.array([c["separation_required"] for c in columns])
    again = recompute(lambda k: held if k <= first else "empty")
    assert np.abs(again - shipped).max() < 1e-6
    # the state switch is read: holding the box for the whole episode does not give the shipped column
    assert np.abs(recompute(lambda k: held) - shipped).max() > 1e-3


def test_the_hri_columns_carry_the_step_as_the_command_term_computed_it():
    h = hri_geometry.HriStep(distance=0.42, v_r=0.7, v_h=0.3, s_p=0.9, violated=True,
                             robot_point=np.zeros(3), human_point=np.ones(3), direction=np.array([1.0, 0, 0]))
    fr = {cam: 0.25 for cam in rl.fraction_cameras()}
    out = rl.hri_step_columns(h, pelvis_pose_xyzw=[1, 2, 0.9, 0.1, 0.2, 0.3, 0.9], pelvis_vel=[0.5, 0, 0, 0, 0, 0.1],
                              fractions=fr)
    assert (out["min_distance_human"], out["separation_required"], out["separation_violation"]) == (0.42, 0.9, True)
    assert (out["approach_speed_robot"], out["approach_speed_human"]) == (0.7, 0.3)
    assert list(out["human_root_pose"]) == [1, 2, 0.9, 0.9, 0.1, 0.2, 0.3]           # wxyz
    assert {k for k in out if k.startswith(rl.FRACTION_PREFIX)} == {k for k in rl.hri_zero_step_keys()
                                                                     if k.startswith(rl.FRACTION_PREFIX)}
    with pytest.raises(ValueError, match="카메라"):
        rl.hri_step_columns(h, pelvis_pose_xyzw=[0] * 6 + [1], pelvis_vel=[0] * 6, fractions={"head_rgb": 0.1})


# ---- the sidecar ----

def test_the_sidecar_carries_ssm02_and_the_task_step(tmp_path):
    import validator.validate_dataset as vd
    from generation import convert_lerobot as cl

    params, table = directed_separation.load_directed_ssm(RULES)
    ssm = json.loads(json.dumps(rl.hri_ssm_params(params, table)))
    assert set(ssm) == {*directed_separation.DIRECTED_SSM_PARAMETERS, "stop_table"}
    done = rl.step_or_none(np.int64(593))
    assert done == 593 and type(done) is int
    for bad in (True, -2, 3.0, "5"):
        with pytest.raises(ValueError):
            rl.step_or_none(bad)
    record = {"task_stages": [{"name": "walk", "completed": True, "end_step": done, "duration_s": 11.86}],
              "human_stature_m": 1.74, "task_time_limit_s": 20.0}
    notes = {"ssm_params": ssm, "task_completed_step": done, **record}
    assert vd.notes_problems(notes, vd.BODY_REGIONS, steps=999, carried=params, ssm_names=vd.DIRECTED_SSM_IN_NOTES,
                             stop_table=table, completion_required=True) == []
    with pytest.raises(ValueError, match="파라미터"):
        rl.hri_ssm_params({**params, "human_speed": 1.6}, table)
    # the converter takes the step
    np.savez(tmp_path / "e.npz", timestamp=np.arange(999) * 0.02)
    (tmp_path / "e.json").write_text(json.dumps({
        "scenario_id": "S1e", "task_completed_step": done, **record}))
    by_sid = {s["scenario_id"]: s for s in SCENARIOS["scenarios"]}
    assert cl.check_hri_completion([tmp_path / "e.npz"], by_sid) is None
    # the events: no protective stop (the baseline robot runs no supervisor), contact_onset for the judgement
    assert set(rl.HRI_EVENT_NAMES) <= set(SCHEMA["safety_events"])
    assert "protective_stop_triggered" not in rl.HRI_EVENT_NAMES and "contact_onset" in rl.HRI_EVENT_NAMES


def test_the_sidecar_counts_the_steps_each_camera_sat_inside_the_held_box():
    counts = rl.camera_in_box_counts()
    # every robot camera with a visible-fraction column, 0 before any step
    assert counts == {cam: 0 for cam in rl.fraction_cameras()} and len(counts) == 5
    for inside in [("left_wrist_rgb",), (), ("left_wrist_rgb", "right_wrist_rgb"), ("left_wrist_rgb",)]:
        counts = rl.count_cameras_in_box(counts, inside)
    assert counts == {**{cam: 0 for cam in rl.fraction_cameras()}, "left_wrist_rgb": 3, "right_wrist_rgb": 1}
    assert json.loads(json.dumps(counts)) == counts and all(type(v) is int for v in counts.values())
    with pytest.raises(ValueError, match="cctv_rgb"):
        rl.count_cameras_in_box(counts, ("cctv_rgb",))
    with pytest.raises(ValueError, match="두 번"):
        rl.count_cameras_in_box(counts, ("left_wrist_rgb", "left_wrist_rgb"))


def test_a_stage_runs_from_the_end_of_the_one_before():
    assert rl.task_stages_record(("enter", "turn", "exit"), (120, 400, 610), 0, 50) == [
        {"name": "enter", "completed": True, "end_step": 120, "duration_s": 2.4},
        {"name": "turn", "completed": True, "end_step": 400, "duration_s": 5.6},
        {"name": "exit", "completed": True, "end_step": 610, "duration_s": 4.2}]


def test_an_open_stage_has_no_end_and_no_duration():
    got = rl.task_stages_record(("turn", "walk"), (150, -1), 30, 50)
    assert got[0] == {"name": "turn", "completed": True, "end_step": 150, "duration_s": 2.4}
    assert got[1] == {"name": "walk", "completed": False, "end_step": -1, "duration_s": None}
    assert all(not s["completed"] for s in rl.task_stages_record(("turn", "walk"), (-1, -1), -1, 50))
    # plain JSON: the steps are ints, the durations floats or null
    record = rl.task_stages_record(("turn", "walk"), (np.int64(150), np.int64(-1)), np.int64(30), 50)
    assert json.loads(json.dumps(record)) == got
    assert type(record[0]["end_step"]) is int and type(record[0]["duration_s"]) is float
    # a stage ending at the step the one before ended takes no time
    assert rl.task_stages_record(("a", "b"), (40, 40), 10, 50)[1]["duration_s"] == 0.0


@pytest.mark.parametrize("names, ends, started, fps", [
    (("turn", "walk"), (150,), 30, 50),            # a step for each stage
    (("turn", "walk"), (-1, 200), 30, 50),         # a stage ends only after the one before
    (("turn", "walk"), (150, 120), 30, 50),        # nor before it
    (("turn",), (20,), 30, 50),                    # nor before the task came
    (("turn",), (150,), -1, 50),                   # nor when it never came
    (("turn",), (-2,), 30, 50),                    # a step is an int from -1
    (("turn",), (150.0,), 30, 50),
    (("turn",), (150,), -2, 50),
    (("turn",), (150,), 30, 0),                    # the step rate is a positive int
    (("turn",), (150,), 30, 50.0),
    (("turn",), (150,), 30, True),
])
def test_stages_that_cannot_have_ended_so_are_refused(names, ends, started, fps):
    with pytest.raises(ValueError):
        rl.task_stages_record(names, ends, started, fps)


def test_the_recorder_takes_the_stages_on_every_step_and_writes_them_with_the_stature_and_the_time_limit():
    import ast

    flat = lambda text: " ".join(text.split())                                  # noqa: E731
    # one helper copies the stages off the command term, on every step and once before the first step, so a
    # recording without a step still names its stages, all open
    helper = RECORDER[RECORDER.index("    def _take_stages("):RECORDER.index("    def ", RECORDER.index("    def _take_stages(") + 10)]
    for line in ("self.task_stage_names = tuple(command.stages.names)",
                 "self.task_stage_end_steps = [int(s) for s in command.stages.end_steps]"):
        assert line in helper and RECORDER.count(line) == 1, line
    step = RECORDER[RECORDER.index("    def _hri_step("):RECORDER.index("    def capture_randomization")]
    assert "self._take_stages(command)" in step
    capture = RECORDER[RECORDER.index("    def capture_randomization"):RECORDER.index("    def _finalize_randomization")]
    hri_capture = capture[capture.index("if self.hri:"):capture.index("entry = None")]
    assert RECORDER.count("self._take_stages(command)") == 2
    for line in ("self._take_stages(command)", "self.human_stature_m = float(command.stature_m)",
                 "self.task_time_limit_s = float(env.max_episode_length_s)",
                 "self._task_fps = round(1.0 / float(env.step_dt))"):
        assert line in hri_capture, line
    save = RECORDER[RECORDER.index("    def save("):]
    hri_block = save[save.index("        if self.hri:\n            # The force matrix"):save.index("        if media_files is not None:")]
    for line in ('meta["task_stages"] = task_stages_record(self.task_stage_names, self.task_stage_end_steps, '
                 'self.task_started_step, self._task_fps)',
                 'meta["human_stature_m"] = self.human_stature_m',
                 'meta["task_time_limit_s"] = self.task_time_limit_s'):
        assert line in flat(hri_block), line
    required = RECORDER[RECORDER.index("SIDECAR_REQUIRED_KEYS = ("):RECORDER.index(")\n", RECORDER.index("SIDECAR_REQUIRED_KEYS = ("))]
    assert not any(key in required for key in ("task_stages", "human_stature_m", "task_time_limit_s"))
    # the command term: a new tracker every episode, moved on right after the completion, and the stature the seeing
    # rule takes is the one the recorder reads
    tree = ast.parse(COMMANDS)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HriTaskCommand")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert "self.stages = task_path.StageTracker(self.site, self.row)" in [
        ast.unparse(n) for n in methods["_new_episode"].body]
    update = [ast.unparse(n) for n in methods["_update_command"].body]
    at = next(i for i, s in enumerate(update) if s.startswith("self.completion.update("))
    assert update[at + 1] == "self.stages.update(step, self.follower, self.completion.step)"
    assert sum("self.stages.update(" in s for s in update) == 1
    assert 'self.stature_m = float(spec.anthropometry["stature"])' in COMMANDS
    assert 'visibility.height_px(cam, pos, quat, frame["pelvis"], self.stature_m)' in COMMANDS
    assert "_stature_m" not in COMMANDS.replace("self.stature_m", "")


def test_the_recorder_takes_the_cameras_in_the_box_on_every_step_and_writes_them():
    init = RECORDER[RECORDER.index("    def __init__("):RECORDER.index("    def step(self, env)")]
    assert "self.camera_in_box_steps = camera_in_box_counts()" in init
    step = RECORDER[RECORDER.index("    def _hri_step("):RECORDER.index("    def capture_randomization")]
    assert "self.camera_in_box_steps = count_cameras_in_box(self.camera_in_box_steps, command.cameras_in_box)" in step
    save = RECORDER[RECORDER.index("    def save("):]
    start = save.index("        if self.hri:\n            # The force matrix")
    hri_block = save[start:save.index("        if media_files is not None:")]
    assert 'meta["camera_in_box_steps"] = dict(self.camera_in_box_steps)' in hri_block
    # the command term: the held box is left out of the fractions of a camera inside it, and those cameras are named
    view = COMMANDS[COMMANDS.index("    def _camera_view("):COMMANDS.index("    # ---- the step ----")]
    assert ('{} if held_box is None else visibility.robot_camera_box_clearances(\n'
            '            self._cameras, frame["link_poses"], held_box)') in view
    assert "self.cameras_in_box = visibility.cameras_in_box(self.camera_box_clearance)" in view
    # the run prints the least clearance of the cameras that move about the box, beside the arms' hold error
    assert "command.moving_cameras" in RUN and "command.camera_box_clearance" in RUN


def test_the_condition_names_the_row_its_places_and_the_task_step():
    entry, row = _s1e_row()
    site = conditions.row_site(sites.load_site(entry), row)
    plan = human_script.plan_for(entry, row)
    record = json.loads(json.dumps(rl.condition_record(7, row, site, plan, task_started_step=0)))
    assert record["row"] == 7 and record["task_started_step"] == 0
    assert {k: record[k] for k in row} == json.loads(json.dumps(dict(row)))
    assert record["robot_start"] == {"x": site.robot_start.x, "y": site.robot_start.y,
                                     "yaw_deg": site.robot_start.yaw_deg}
    assert human_script.plan_from_record(record["human_plan"]) == plan
    with pytest.raises(ValueError, match="겹칩니다"):
        rl.condition_record(7, {**row, "robot_goal": 1.0}, site, plan, task_started_step=0)
    # a random row says so, and has no grid cell
    assert record["design"] == "random" and record["grid_cell"] is None


def test_a_grid_row_names_its_cell_and_the_names_are_not_a_columns():
    entry, row = _s1e_row()
    site = conditions.row_site(sites.load_site(entry), row)
    plan = human_script.plan_for(entry, row)
    record = json.loads(json.dumps(rl.condition_record(7, row, site, plan, task_started_step=0,
                                                       design="grid", grid_cell=(3, 5))))
    assert record["design"] == "grid" and record["grid_cell"] == [3, 5]
    for name in ("design", "grid_cell"):
        with pytest.raises(ValueError, match="겹칩니다"):
            rl.condition_record(7, {**row, name: 1}, site, plan, task_started_step=0)


def test_the_recorder_classifies_its_row_by_the_episodes_own_number():
    finalize = RECORDER[RECORDER.index("    def _finalize_randomization"):RECORDER.index("    def ", RECORDER.index("    def _finalize_randomization") + 10)]
    assert "conditions.row_design(conditions.scenario_entry(self.scenario_id), self.episode_seq)" in finalize
    assert "design=design, grid_cell=cell" in finalize


def test_the_recorder_writes_the_sidecar_of_its_mode():
    assert "self.ssm_params = hri_ssm_params(command._ssm, command._stop_table)" in RECORDER
    assert '"safety_event_names": list(HRI_EVENT_NAMES if self.hri else EVENT_NAMES),' in RECORDER
    assert 'meta["task_completed_step"] = step_or_none(self.task_completed_step)' in RECORDER
    assert 'meta["degenerate_reason"] = self.degenerate_reason' in RECORDER
    # the entry's keys hold other things in these scenarios (a time trigger, a walking direction): not under S1d's names
    capture = RECORDER[RECORDER.index("    def capture_randomization"):RECORDER.index("    def _finalize_randomization")]
    assert capture.index("if self.hri:") < capture.index('"entry_trigger_distance_drawn"')


def test_the_recorder_reads_the_command_terms_fields_on_every_step():
    step = RECORDER[RECORDER.index("    def _hri_step("):RECORDER.index("    def capture_randomization")]
    for line in ("touch = command.touch", "touch.samples != env.cfg.decimation", "command.hri", "command.camera_fractions",
                 "command.completion.step", "command.task_started_step", "command.box_released_step",
                 "substep_contact.get_hri(env).drain()"):
        assert line in step, line


def test_the_recorders_hri_contact_columns_replace_the_net_force_ones():
    # the net-force columns are computed first for every scenario; in hri mode the matrix's force, height, type and
    # touch take their place before the step is stored, and the events are the three of HRI_EVENT_NAMES
    step = RECORDER[RECORDER.index("    def step(self, env)"):RECORDER.index("    def _hri_step(")]
    swap = 'contact_force, region_z, contact_any = hri["contact_force"], hri["region_z"], hri["touched"]'
    assert step.index("contact_any = robot_side_contact or dummy_side_contact") < step.index(swap)
    assert step.index("contact_type = classify_contact_type(") < step.index('contact_type = hri["contact_type"]')
    assert step.index('contact_type = hri["contact_type"]') < step.index("self.steps.append(")
    assert 'events = np.array([hri["columns"]["separation_violation"], events[2], hri["rack_edge"]], dtype=bool)' in step
    assert step.index("if contact_any and hold_applied:") > step.index(swap)


def test_the_robots_velocity_the_command_term_reads_is_the_recorded_root_velocity():
    # S_p takes the root's horizontal speed and yaw rate; read off the velocity observation.root_vel records, a
    # consumer can compute it again from the shipped columns
    assert '"root_vel": robot.data.root_vel_w.torch[0].cpu().numpy(),' in RECORDER
    assert "root_pose, root_vel = _row(robot.data.root_link_pose_w), _row(robot.data.root_vel_w)" in COMMANDS


def test_run_s1a_installs_and_resets_the_matrix_window():
    assert "hri_contact = substep_contact.install_hri(base_env) if SCENARIO_ID in TABLE_SCENARIOS else None" in RUN
    assert RUN.count("if hri_contact is not None:\n") >= 2
    assert RUN.count("hri_contact.reset()") >= 2


# ---- the dummy ----

STANDING = {"BR-01": 1.66, "BR-07": 0.98, "BR-12": 0.29}
CODES = list(STANDING)


def _rows(n=100, *, released_at=60, triggered_at=10):
    """An upright dummy held until ``released_at`` that sets off walking at ``triggered_at``."""
    return {"head_z": np.full(n, 1.66), "tilt": np.full(n, 2.0), "released": np.arange(n) >= released_at,
            "triggered": np.arange(n) >= triggered_at, "walk_ended": np.zeros(n, bool),
            "region_heights": np.tile([STANDING[c] for c in CODES], (n, 1))}


def _verdict(rows, release_step):
    return rl.dummy_degenerate_reason(rows, release_step, codes=CODES, standing=STANDING)


def test_a_dummy_that_falls_before_contact_is_degenerate_and_a_robot_fall_is_not():
    assert _verdict(_rows(), 70) is None
    fallen = _rows()
    fallen["head_z"][65:] = 0.45                              # down after the hold let go, before the touch
    assert "스텝 65" in _verdict(fallen, 70)
    assert _verdict(fallen, 65) is None                       # it fell as it was hit: the outcome, not a fault
    assert "스텝 65" in _verdict(fallen, -1)                   # never hit and down
    tipped = _rows()
    tipped["tilt"][62:] = 50.0                                # not held any more, but past the fall's tilt
    assert "기울기 50.0" in _verdict(tipped, 70)
    # A robot that falls ends the episode at its fall, here at step 40 -- before the hold let go (60) and with no
    # touch: the rows stop there with the dummy upright, held and walking. The verdict reads the dummy only, so the
    # robot's fall is an outcome; the same short episode with the dummy down is broken.
    robot_fell = {k: v[:40] for k, v in _rows().items()}
    assert _verdict(robot_fell, -1) is None
    both_fell = {k: v.copy() for k, v in robot_fell.items()}
    both_fell["head_z"][30:] = 0.45
    assert "스텝 30" in _verdict(both_fell, -1)


def test_a_held_dummy_that_tips_to_the_walks_tilt_mark_before_contact_is_degenerate():
    from generation.envs.s1d.jointed_walk_metrics import TILT_MAX_DEG

    at_step_7 = _rows(triggered_at=0)
    at_step_7["tilt"][7] = 20.2                               # held from step 0, walking from step 0
    assert "스텝 7" in _verdict(at_step_7, 70) and "20.2" in _verdict(at_step_7, 70)
    edge = _rows()
    edge["tilt"][30] = TILT_MAX_DEG                           # the mark passes below it
    assert _verdict(edge, 70) is not None
    edge["tilt"][30] = TILT_MAX_DEG - 0.01
    assert _verdict(edge, 70) is None
    loose = _rows()
    loose["tilt"][62:] = 20.0                                 # let go at 60: no longer held, short of a fall
    assert _verdict(loose, 70) is None
    assert _verdict(at_step_7, 7) is None                     # touched at 7: step 7 is not before it


def _off(rows, step, by, code="BR-12"):
    """``rows`` with one region ``by`` metres off its standing height at ``step`` (the median offset stays 0)."""
    rows["region_heights"][step, CODES.index(code)] += by
    return rows


def test_a_dummy_before_setting_off_off_its_height_is_degenerate_and_a_walking_one_is_not_judged_on_it():
    from generation.envs.s1d.jointed_walk_metrics import STANDING_TOL_M

    low = _off(_rows(), 5, 2 * STANDING_TOL_M)                # standing, held, before setting off at 10
    assert "스텝 5" in _verdict(low, 70) and "BR-12" in _verdict(low, 70)
    stride = _off(_rows(), 30, 0.03)                          # a walking step lifts the shin
    assert _verdict(stride, 70) is None
    sway = _rows()
    sway["region_heights"][5] += 0.04                         # the whole body higher: the shared offset comes off
    assert _verdict(sway, 70) is None


def test_the_tolerances_are_the_decided_ones():
    # self-set numbers: widening either one has to be a decision, not a drive-by
    from generation.envs.s1d import jointed_walk_metrics as m

    assert m.STANDING_TOL_M == 0.015
    assert m.STOPPED_STANDING_TOL_M == 0.10


def test_a_dummy_that_walked_and_stopped_stands_within_the_wider_tolerance():
    from generation.envs.s1d.jointed_walk_metrics import STANDING_TOL_M, STOPPED_STANDING_TOL_M

    stopped = _rows()
    stopped["walk_ended"][40:] = True
    just_in = _off({k: v.copy() for k, v in stopped.items()}, 45, STOPPED_STANDING_TOL_M - 0.0005)
    assert _verdict(just_in, 70) is None                      # inside the stopped tolerance
    # the same limb off by a distance between the two tolerances is judged only by the one that applies to the step
    between = (STANDING_TOL_M + STOPPED_STANDING_TOL_M) / 2
    assert STANDING_TOL_M < between < STOPPED_STANDING_TOL_M
    assert _verdict(_off({k: v.copy() for k, v in stopped.items()}, 45, between), 70) is None
    just_out = _off({k: v.copy() for k, v in stopped.items()}, 45, STOPPED_STANDING_TOL_M + 0.0005)
    reason = _verdict(just_out, 70)
    assert "스텝 45" in reason and "BR-12" in reason and "멈춘 뒤" in reason
    assert f"{STOPPED_STANDING_TOL_M * 100:g} cm" in reason


def test_a_dummy_before_setting_off_keeps_the_narrow_tolerance():
    from generation.envs.s1d.jointed_walk_metrics import STANDING_TOL_M, STOPPED_STANDING_TOL_M

    between = (STANDING_TOL_M + STOPPED_STANDING_TOL_M) / 2
    assert _verdict(_off(_rows(), 5, STANDING_TOL_M - 0.0005), 70) is None
    reason = _verdict(_off(_rows(), 5, between), 70)          # a limb the stopped tolerance would let pass
    assert "스텝 5" in reason and "BR-12" in reason and "출발 전" in reason
    assert f"{STANDING_TOL_M * 100:g} cm" in reason
    assert _verdict(_off(_rows(), 5, STANDING_TOL_M + 0.0005), 70) is not None


def test_a_step_that_has_not_set_off_and_has_stopped_is_judged_before_setting_off():
    from generation.envs.s1d.jointed_walk_metrics import STANDING_TOL_M, STOPPED_STANDING_TOL_M

    both = _rows()
    both["walk_ended"][5] = True                              # the flags disagree: the strict tolerance applies
    reason = _verdict(_off(both, 5, (STANDING_TOL_M + STOPPED_STANDING_TOL_M) / 2), 70)
    assert "스텝 5" in reason and "출발 전" in reason


def test_a_released_dummy_is_not_judged_on_its_height_after_the_stop():
    # let go at 60 (the first touch): the steps from there on are the contact's, whatever the height
    from generation.envs.s1d.jointed_walk_metrics import STOPPED_STANDING_TOL_M

    late = _rows(released_at=60)
    late["walk_ended"][40:] = True
    assert _verdict(_off(late, 65, 2 * STOPPED_STANDING_TOL_M), 70) is None


def test_the_walks_assist_marks_are_not_part_of_the_verdict():
    # the hand's force across the plane, the tilt torque and the feet's share were set on long walks; a short window
    # misses them at every start -- the rule reads tilt and standing height only
    assert set(rl.DUMMY_ROW_KEYS) == {"head_z", "tilt", "released", "triggered", "walk_ended", "region_heights"}
    with pytest.raises(ValueError, match="head_z"):
        rl.dummy_degenerate_reason({k: v for k, v in _rows().items() if k != "head_z"}, 70, codes=CODES,
                                   standing=STANDING)
    # a standing-height check that cannot run is not a pass
    with pytest.raises(ValueError, match="BR-12"):
        rl.dummy_degenerate_reason(_rows(), 70, codes=CODES, standing={"BR-01": 1.66, "BR-07": 0.98})
    with pytest.raises(ValueError, match="부위 수"):
        rl.dummy_degenerate_reason(_rows(), 70, codes=CODES[:2], standing=STANDING)


def test_the_recorder_judges_the_dummy_and_not_the_robot():
    save = RECORDER[RECORDER.index("    def save("):]
    block = save[save.index("if self.hri:"):save.index("if self.steps:")]
    assert "dummy_degenerate_reason(" in block and "self.first_touch_step" in block
    assert "codes=self._codes, standing=self._standing_heights)" in block
    assert "standing_region_heights(" in RECORDER and "ANTHROPOMETRY_PATH" in RECORDER
    assert "termination" not in block


# ---- the drawn look in the sidecar ----

def test_the_sidecar_carries_the_drawn_look_in_every_state():
    from generation import appearance

    capture = RECORDER[RECORDER.index("    def capture_randomization"):RECORDER.index("    def _finalize_randomization")]
    # set after the hri cut-down replaces the dict, so it survives it; not conditioned on the scenario
    assert capture.index("if self.hri:") < capture.index('self.randomization["appearance"]')
    assert 'self.randomization["appearance"] = None if self.appearance is None else dict(self.appearance)' in capture
    assert "if self.appearance is not _NO_APPEARANCE:" in capture
    init = RECORDER[RECORDER.index("    def __init__("):RECORDER.index("    def step(self, env)")]
    assert "appearance=_NO_APPEARANCE" in init and "self.appearance = appearance" in init
    # the drawn dict goes through json whole: every float is the float that was drawn
    for sid in conditions.table_scenarios():
        entry = conditions.scenario_entry(sid)
        for row in range(40):
            drawn = appearance.draw_appearance(entry, row)
            recorded = dict(drawn)
            assert json.loads(json.dumps({"appearance": recorded})) == {"appearance": drawn}
    # S1d declares no block: the run passes no draw, so the sidecar has no key
    s1d = next(e for e in SCENARIOS["scenarios"] if e["scenario_id"] == "S1d")
    assert appearance.draw_appearance(s1d, 0) is None
