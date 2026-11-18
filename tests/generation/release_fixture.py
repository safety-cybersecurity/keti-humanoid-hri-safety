"""A small distribution whose labels the labeller made, for the tests that read one back.

The release is built with the production path's own functions -- the step columns by the
converter's step_frames and to_lerobot_frame, the episode columns by build_episode_meta and
episode_column, the labels by label_episode. What is written by hand is LeRobot's side: where
the files lie, the index columns and the Arrow types, copied from the schema of the assembly of
2026-10-04 (``.tmp/batches/S1d-400/dataset``) -- vectors as fixed_size_list, a length of one as
a scalar, timestamp as float32. The episode table's floats go in as the converter's pandas writes them: a NaN
is a null.

Shared by the tests that read a release back (the label re-derivation, the assembly, the scorecard) and by the
converter's tests of the table scenario's columns; not a test module.
"""

import functools
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import generation.convert_lerobot as cl
from generation import assets as assets_lib
from generation import cameras as camera_canon
from generation import conditions, human_script, sites, visibility
from generation.directed_separation import CONTACT_RULE_ID, DIRECTED_SSM_RULE_ID
from generation.envs.hri import record_logic
from generation.field_map import HRI, STEP_FIELDS
from generation.labeling.label_episodes import label_episode
from generation.scene import layout

REPO_ROOT = Path(__file__).resolve().parents[2]

FPS = 50
N = 200                               # 4 s: float32 timestamps are no longer evenly spaced to 1e-9
REGIONS = ["BR-01", "BR-05"]          # 접촉 금지 부위 하나 + 허용 부위 하나
EVENTS = ["ssm_violation", "protective_stop_triggered", "contact_onset"]
ROBOT_USD = "IsaacLab/Robots/Unitree/H1/h1.usd"
JOINT_MAP = {"robot_model": "unitree_h1", "dof": 2, "joint_names": ["a", "b"]}
ASSETS = {"robots": {"unitree_h1": {"usd_path": ROBOT_USD, "license": "BSD-3-Clause",
                                    "redistributed": False, "embodiment": "biped"}}}
SCENARIO = {"scenario_id": "S1d", "domain": "logistics", "task_archetype": "locomotion",
            "hazard_class": "collision", "safety_function": "ssm",
            "rules": ["R-TS15066-A2-*", "R-TS15066-SSM-01", "R-10218-STOP-01"],
            "scene": {"human_model": "keti_dummy_ts15066_v1_skin"}}
SCENARIOS = {"scenarios": [SCENARIO]}
HRI_EVENTS = ["ssm_violation", "contact_onset", "rack_contact"]
HRI_SCENARIO = {"scenario_id": "S1e", "domain": "logistics", "task_archetype": "transport",
                "hazard_class": "collision", "safety_function": "none",
                "rules": ["R-TS15066-A2-*", DIRECTED_SSM_RULE_ID, CONTACT_RULE_ID],
                "scene": {"human_model": None}}
FRACTION_PREFIX = "observation.human_visible_fraction."
HRI_CAMERAS = tuple(f.schema[len(FRACTION_PREFIX):] for f in STEP_FIELDS
                    if f.condition == HRI and f.schema.startswith(FRACTION_PREFIX))
APPROACH = 1.0      # m/s, the robot's approach speed while it walks
CLEAR = 0.5         # m, min_distance_human - separation_required on a step without a breach


def _canon_rule(rule_id):
    doc = json.loads((REPO_ROOT / "meta" / "rules.json").read_text(encoding="utf-8"))
    return next(r for r in doc["rules"] if r["rule_id"] == rule_id)


def limit(value, unit="N", **extra):
    return {"value": value, "unit": unit, "determined_by": "standard",
            "standard_ref": "KS B ISO TS 15066:2016 표 A.2", **extra}


def canon_rules():
    """정본과 같은 모양, 합성 값. BR-05 는 준정적 140 N · 동적 2배, BR-01 은 접촉 금지.
    SSM-02 · CONTACT-01 은 정본의 항목 그대로(표준 값이 없어 합성할 것이 없음)."""
    def region(force, permitted):
        r = {"name_ko": "합성", "name_en": "synthetic", "contact_permitted": permitted,
             "limits": {"force_quasistatic": limit(force),
                        "pressure_quasistatic": limit(None, "N/cm^2"),
                        "energy_transient": limit(None, "J")},
             "biomechanical": {"effective_mass": limit(None, "kg"),
                               "spring_constant": limit(None, "N/mm")}}
        r["transient_multiplier"] = (limit(2, "1") if permitted else
                                     limit(None, "1", not_applicable=True, reason="접촉 금지"))
        return r

    return {
        "rules": [
            {"rule_id": "R-TS15066-SSM-01", "outcome": "near_miss",
             "standard": "KS B ISO/TS 15066 (SSM)", "parameters": {}},
            {"rule_id": "R-10218-STOP-01", "outcome": "fail", "standard": "KS B ISO 10218-1",
             "parameters": {"max_stop_time": {"value": 0.34, "unit": "s", "determined_by": "project"},
                            "stop_speed_threshold": {"value": 0.15, "unit": "m/s",
                                                     "determined_by": "project"}}},
            {"rule_id": "R-TS15066-A2-01", "outcome": "fail", "body_region": "BR-01",
             "standard": "KS B ISO TS 15066 표 A.2", "parameters": {},
             "limits_ref": "contact_limits.BR-01"},
            {"rule_id": "R-TS15066-A2-05", "outcome": "fail", "body_region": "BR-05",
             "standard": "KS B ISO TS 15066 표 A.2", "parameters": {},
             "limits_ref": "contact_limits.BR-05"},
            _canon_rule(DIRECTED_SSM_RULE_ID), _canon_rule(CONTACT_RULE_ID),
        ],
        "contact_limits": {"BR-01": region(130, False), "BR-05": region(140, True)},
    }


def recording(seq, *, violation=(), stops=(), slow_from=None, touch=None, release=0,
               reason=("time_out",)):
    """One episode as the recorder leaves it: the npz arrays and the sidecar.

    touch: (step, region, newtons, contact type). stops: the steps a protective stop is
    triggered at -- the robot walks at 1 m/s and, with ``slow_from``, at 0.1 m/s from there."""
    events = np.zeros((N, len(EVENTS)), dtype=bool)
    events[list(violation), 0] = True
    events[list(stops), 1] = True
    force = np.zeros((N, len(REGIONS)))
    kind = np.full((N, len(REGIONS)), "none", dtype="<U11")
    if touch is not None:
        step, region, newtons, contact_type = touch
        force[step, REGIONS.index(region)] = newtons
        kind[step, REGIONS.index(region)] = contact_type
        events[step, 2] = True
    root_vel = np.zeros((N, 6), dtype=np.float32)
    root_vel[:, 0] = 1.0
    if slow_from is not None:
        root_vel[slow_from:, 0] = 0.1
    arrays = {
        "timestamp": np.arange(N) * (1.0 / FPS),
        "joint_pos": np.zeros((N, 2), np.float32), "joint_vel": np.zeros((N, 2), np.float32),
        "joint_torque": np.zeros((N, 2), np.float32), "action": np.zeros((N, 2), np.float32),
        "root_pose": np.tile(np.float32([0, 0, 1, 1, 0, 0, 0]), (N, 1)),
        "root_vel": root_vel, "imu": np.zeros((N, 10), np.float32),
        "contact_force": force, "contact_v_rel": np.zeros((N, len(REGIONS))),
        "contact_type": kind,
        "com": np.zeros((N, 3)), "zmp": np.zeros((N, 2)), "stability_margin": np.zeros(N),
        "min_distance_human": np.ones(N), "separation_required": np.full(N, 0.5),
        "separation_violation": events[:, 0].copy(),
        "safety_event": events,
    }
    meta = {
        "episode_id": f"S1d_{seq:04d}_seed{seq + 1}", "scenario_id": "S1d", "episode_seq": seq,
        "run_start_seq": 0, "seed": seq + 1, "steps": N, "degenerate": False,
        "contact_region_names": list(REGIONS), "safety_event_names": list(EVENTS),
        "hold_release_step": release, "hold_contaminated_steps": 0, "provisional": True,
        "ssm_params": {}, "randomization": {}, "robot_model": "unitree_h1",
        "asset_paths": {"robot": ROBOT_USD}, "asset_paths_resolved": {"robot": ROBOT_USD},
        "imu_link": "pelvis", "action_space": "position_rad",
        "contact_onset_unresolved_reason": "발이 넘어진 더미를 밟는 접촉은 잡히지 않는다",
        "termination": {"terminated": reason != ("time_out",), "truncated": reason == ("time_out",),
                        "reason": list(reason)},
        "sim_backend": {"isaacsim": "6.0.1", "isaaclab": "v3.0.0", "physics": "physx",
                        "generator_commit": "0123456789abcdef0123456789abcdef01234567",
                        "generator_dirty": False},
    }
    return arrays, meta


def eight():
    """Eight episodes, one per way a label comes about. What each one's label is, by hand:

    0 nothing happens                                        nominal   S0
    1 separation breached, no contact                        near_miss S1  SSM-01
    2 a protective stop the robot never slows after          fail      S2  STOP-01
    3 300 N transient on BR-05 after release (limit 2x140)   fail      S3  A2-05
    4 the same contact, but before the hold let go           nominal   S0  (contact observed)
    5 500.5 N on BR-05 that never got a contact type         nominal   S0  (not evaluated)
    6 ended in a fall, no rule fired                         nominal   S0  (not evaluated)
    7 a stop reached exactly on the budget (17 frames)       nominal   S0
    """
    return [
        recording(0),
        recording(1, violation=range(50, 61)),
        recording(2, stops=[20]),
        recording(3, touch=(120, "BR-05", 300.0, "transient"), release=100),
        recording(4, touch=(90, "BR-05", 300.0, "transient"), release=100),
        recording(5, touch=(150, "BR-05", 500.5, "none")),
        recording(6, reason=("fall_orientation",)),
        recording(7, stops=[30], slow_from=47),
    ]


def _task_stages(completed):
    """The U-turn's three stages of a task done at step ``completed`` (-1 or None: not done, so the
    first stage ends at step 20 and the others stay open)."""
    done = completed is not None and completed >= 0
    ends = [completed // 4, completed // 2, completed] if done else [20, -1, -1]
    start, stages = 0, []
    for name, end in zip(("enter", "turn", "exit"), ends):
        stages.append({"name": name, "completed": end >= 0, "end_step": end,
                       "duration_s": (end - start) / FPS if end >= 0 else None})
        start = end if end >= 0 else start
    return stages


def hri_recording(seq, *, violation=(), touch=None, release=0, reason=("time_out",), completed=150,
                  standing=False, rack=(), degenerate=False, distance_knots=None, human_approach=0.0,
                  condition=None):
    """One episode of the hri scenario as the recorder will leave it: recording()'s arrays and
    sidecar with the hri arrays, the hri events and the scenario's notes. The robot walks toward
    the person at APPROACH -- with ``standing`` at 0, so a contact is one with a standing robot.
    The distance to the person is 1 m on every step unless ``distance_knots`` (step, metres pairs,
    joined by straight lines) say otherwise; a breach puts it 0.1 m inside S_p and a contact step
    puts it at 0. The stored flag agrees with the columns: a moving robot in contact is inside S_p
    and approaching, so that step is a breach too. The person approaches at ``human_approach`` on every
    step, and ``condition`` is the episode's condition-table row (randomization.condition)."""
    arrays, meta = recording(seq, release=release, reason=reason)
    events = np.zeros((N, len(HRI_EVENTS)), dtype=bool)
    breach = set(violation) | ({touch[0]} if touch is not None and not standing else set())
    events[sorted(breach), 0] = True
    events[list(rack), 2] = True
    force = np.zeros((N, len(REGIONS)))
    kind = np.full((N, len(REGIONS)), "none", dtype="<U11")
    if touch is not None:
        step, region, newtons, contact_type = touch
        force[step, REGIONS.index(region)] = newtons
        kind[step, REGIONS.index(region)] = contact_type
        events[step, 1] = True
    separation = np.full(N, 1.0 - CLEAR)
    distance = np.ones(N)
    if distance_knots is not None:
        knots = list(distance_knots)
        distance = np.interp(np.arange(N), [k for k, _ in knots], [d for _, d in knots])
    distance[list(violation)] = separation[list(violation)] - 0.1
    if touch is not None:
        distance[touch[0]] = 0.0
    speed = 0.0 if standing else APPROACH
    shapes = json.loads((REPO_ROOT / "meta" / "safety_schema.json").read_text(encoding="utf-8"))["fixed_shapes"]
    hri = {f.npz: (np.zeros((N, shapes[f.schema])) if f.schema in shapes else np.zeros(N))
           for f in STEP_FIELDS if f.condition == HRI}
    hri["command_velocity"][:, 0] = speed
    hri["approach_speed_robot"][:] = speed
    hri["approach_speed_human"][:] = human_approach
    arrays["root_vel"][:, 0] = speed
    arrays.update(hri, contact_force=force, contact_type=kind, min_distance_human=distance,
                  separation_required=separation, separation_violation=events[:, 0].copy(),
                  safety_event=events)
    # The values only the hri recorder's sidecar holds: why the episode is degenerate, the step the held box
    # was let go (the first contact; -1 when none came) and the steps each robot camera sat inside it.
    meta.update(episode_id=f"S1e_{seq:04d}_seed{seq + 1}", scenario_id="S1e",
                safety_event_names=list(HRI_EVENTS), degenerate=degenerate,
                task_completed_step=completed,
                degenerate_reason="합성 픽스처의 사유" if degenerate else None,
                box_released_step=touch[0] if touch is not None else -1,
                camera_in_box_steps={cam: 0 for cam in HRI_CAMERAS},
                task_stages=_task_stages(completed), human_stature_m=1.74, task_time_limit_s=N / FPS)
    if condition is not None:
        meta["randomization"] = {**meta["randomization"], "condition": condition}
    return arrays, meta


def hri_cases():
    """Nine hri episodes, one per way a label comes about under the hri judgement. By hand:

    0 nothing happens; the task is done at step 150                       nominal   S0
    1 breached while approaching, no contact                               near_miss S1  SSM-02
    2 100 N transient on BR-05 after release (limit 2 x 140)               fail      S2  CONTACT-01
    3 300 N transient on BR-05 after release                               fail      S3  CONTACT-01 A2-05
    4 50 N quasistatic on BR-01 (limit 130) -- the head                    fail      S3  CONTACT-01
    5 150 N on BR-05 that never got a contact type (stricter limit 140)    fail      S3  CONTACT-01 A2-05
    6 300 N transient on BR-05 on the first contact step, whose physics
      ran with the hold (let go at its end) -- judged all the same        fail      S3  CONTACT-01 A2-05
    7 a standing robot walked into, 100 N transient on BR-05; it fell and
      had touched a rack                                                   fail      S2  CONTACT-01
    8 the robot fell with no contact; the distance to the person wanders
      (1.0 m, up to 1.3 m at step 60, down to 0.6 m at step 140, 0.8 m on
      the last step) and never comes inside S_p -- the task was not done   nominal   S0  (not evaluated)
    """
    return [
        hri_recording(0),
        hri_recording(1, violation=range(50, 61)),
        hri_recording(2, touch=(120, "BR-05", 100.0, "transient"), completed=-1),
        hri_recording(3, touch=(120, "BR-05", 300.0, "transient"), completed=-1),
        hri_recording(4, touch=(120, "BR-01", 50.0, "quasistatic"), completed=-1),
        hri_recording(5, touch=(120, "BR-05", 150.0, "none"), completed=-1),
        hri_recording(6, touch=(90, "BR-05", 300.0, "transient"), release=91, completed=-1),
        hri_recording(7, touch=(120, "BR-05", 100.0, "transient"), standing=True,
                      reason=("fall_orientation",), rack=(30,), completed=-1),
        hri_recording(8, reason=("fall_orientation",), completed=-1,
                      distance_knots=[(0, 1.0), (60, 1.3), (140, 0.6), (N - 1, 0.8)]),
    ]


# A scenario with a conditions table: the canon's S1e entry -- its table, its site and the box its robot carries --
# whose episodes carry the spec, the evidence and the grid place (schema §5). TABLE_ASSETS is the asset canon its
# sidecars name. The cameras canon is the repository's, for the releases that ship one.
TABLE_SCENARIO = conditions.scenario_entry("S1e")
TABLE_ASSETS = assets_lib.load_assets(REPO_ROOT)
CAMERAS = camera_canon.load(REPO_ROOT / "meta" / "cameras.json")
POLICY = {"source": "isaaclab-published:rsl_rl/Isaac-Velocity-Flat-H1-v0", "sha256": "ab" * 32,
          "behavior": "task_follow"}
APPEARANCE = {"vest_color": "orange", "light_scale": 1.0}
SEEN_FRACTION = 0.5     # what head_rgb records of the person on a step it sees them (the rule's threshold is lower)
SEEN_DEPTH_M = 3.0      # how far ahead of the robot cameras the person's root stands on such a step


@functools.cache
def _boxes():
    return layout.build_layout(layout.load_spec(layout.SCENE_SPEC))


def _table_asset_paths():
    scene = TABLE_ASSETS["scene"]
    return {"robot": TABLE_ASSETS["robots"]["unitree_h1"]["usd_path"],
            "dummy": scene[TABLE_SCENARIO["scene"]["human_model"]]["usd_path"],
            "terrain": scene["isaaclab_terrain_plane"]["usd_path"],
            "aisle": scene["keti_logistics_aisle_v1"]["usd_path"],
            "box": scene[TABLE_SCENARIO["carry"]["asset"]]["usd_path"]}


def table_recording(seq, row, *, seen=None, **kw):
    """An episode of TABLE_SCENARIO at conditions-table row ``row`` as the recorder leaves it: hri_recording()'s
    arrays and sidecar (``kw`` goes to it) with what the episode spec is built from, written by the recorder's own
    helpers -- the condition record (the row's values, its places, the person's script, the design and the grid cell),
    the canon's asset paths, the walking policy and the look drawn.

    ``seen`` (a slice of steps) records the cameras too, in the recorder's dtypes (poses float32, the box int32): every
    robot camera at the origin looking along world +x and the person's root SEEN_DEPTH_M ahead, head_rgb recording
    SEEN_FRACTION of the person on those steps and the others nothing."""
    arrays, meta = hri_recording(seq, **kw)
    drawn = conditions.draw_row(TABLE_SCENARIO, row, _boxes())
    site = conditions.row_site(sites.load_site(TABLE_SCENARIO), drawn)
    design, cell = conditions.row_design(TABLE_SCENARIO, row)
    record = record_logic.condition_record(row, drawn, site, human_script.plan_for(TABLE_SCENARIO, drawn),
                                           task_started_step=3, design=design, grid_cell=cell)
    meta.update(asset_paths=_table_asset_paths(), asset_paths_resolved=_table_asset_paths(),
                sim_backend={**meta["sim_backend"], "policy": dict(POLICY)},
                randomization={"condition": record, "appearance": dict(APPEARANCE)})
    if seen is not None:
        pose = camera_canon.pose_record(np.zeros(3), camera_canon.optical_frame_xyzw((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        for name in CAMERAS["cameras"]:
            arrays[f"camera_pose_{name}"] = np.tile(pose, (N, 1)).astype(np.float32)
        arrays["bbox_human"] = np.full((N, 4), -1, dtype=np.int32)
        for name in visibility.robot_cameras(CAMERAS):
            arrays[f"human_visible_fraction_{name}"] = np.zeros(N)
        arrays["human_visible_fraction_head_rgb"][seen] = SEEN_FRACTION
        arrays["human_root_pose"] = np.tile([SEEN_DEPTH_M, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (N, 1))
        meta["media"] = {name: {"file": f"media/{meta['episode_id']}.{name}.mkv"} for name in CAMERAS["cameras"]}
    return arrays, meta


def table_cases():
    """Six episodes of the table scenario, three random rows and three grid rows:

    0 row 0  nothing happens                                    first contact null
    1 row 1  300 N transient on BR-05 at step 120               BR-05 300 / (2 x 140)
    2 row 2  50 N quasistatic on BR-01 at step 120 (the head)   BR-01 50 / 130
    3 row 50 (grid cell 0, 0) a breach, no contact
    4 row 57 (grid cell 0, 1) 3 N transient on BR-01            BR-01 inf (no transient contact permitted)
    5 row 98 (grid cell 6, 6) nothing happens
    """
    return [
        table_recording(0, 0),
        table_recording(1, 1, touch=(120, "BR-05", 300.0, "transient"), completed=-1),
        table_recording(2, 2, touch=(120, "BR-01", 50.0, "quasistatic"), completed=-1),
        table_recording(3, 50, violation=range(50, 61)),
        table_recording(4, 57, touch=(120, "BR-01", 3.0, "transient"), completed=-1),
        table_recording(5, 98),
    ]


def column(values, fixed=True):
    """An Arrow column typed the way LeRobot's writer types it -- or, with ``fixed`` off, with
    its vectors as plain lists, the other way a writer may spell the same feature."""
    first = values[0]
    if isinstance(first, str):
        return pa.array(values, pa.string())
    if isinstance(first, np.ndarray):
        flat = pa.array(np.concatenate(values))
        if first.shape == (1,):
            return flat
        return pa.FixedSizeListArray.from_arrays(flat, first.shape[0]) if fixed else pa.array(values)
    if isinstance(first, np.float32):
        return pa.array(np.asarray(values, dtype=np.float32))
    return pa.array(np.asarray(values, dtype=np.int64))


def episode_column(values):
    """As the converter's pandas writes an episode column: a NaN is a null (merge_episode_metadata)."""
    if any(isinstance(v, list) for v in values):
        return pa.array(values, pa.list_(pa.string()))
    return pa.array(values, from_pandas=True)


def write_release(root, recordings, labels=None, rules=None, per_file=3, fixed=True,
                  scenario=SCENARIO, assets=ASSETS, cameras=None):
    """Write ``recordings`` as a distribution under ``root``. ``labels`` replaces the labeller's
    own verdicts (one per episode; None keeps the labeller's). ``cameras`` is the cameras canon the
    labeller reads and the release ships (meta/cameras.json, as the converter ships it)."""
    rules = rules or canon_rules()
    labels = list(labels or [None] * len(recordings))
    step_rows, episode_rows, start = [], [], 0
    for e, (arrays, meta) in enumerate(recordings):
        label = labels[e] or label_episode(arrays, meta, rules, scenario, cameras)
        dtypes = cl.recorded_dtypes(arrays, meta)
        frames = [cl.to_lerobot_frame(f, dtypes) for f in cl.step_frames(arrays, meta, JOINT_MAP)]
        for i, frame in enumerate(frames):
            step_rows.append((e // per_file, {
                **frame, "timestamp": np.float32(i / FPS), "frame_index": i, "episode_index": e,
                "index": start + i, "task_index": 0}))
        record = cl.build_episode_meta(meta, label, {"scenarios": [scenario]}, JOINT_MAP, assets, rules,
                                       step_dt=1.0 / FPS)
        episode_rows.append({
            "episode_index": e, "length": len(frames), "data/chunk_index": 0,
            "data/file_index": e // per_file, "dataset_from_index": start,
            "dataset_to_index": start + len(frames),
            **{k: cl.episode_column(v) for k, v in record.items()}})
        start += len(frames)

    for file_index in sorted({f for f, _ in step_rows}):
        rows = [row for f, row in step_rows if f == file_index]
        path = root / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({k: column([r[k] for r in rows], fixed) for k in rows[0]}), path)

    path = episodes_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if episode_rows:
        pq.write_table(pa.table({k: episode_column([r[k] for r in episode_rows])
                                 for k in episode_rows[0]}), path)
    else:
        pq.write_table(pa.table({"episode_index": pa.array([], pa.int64())}), path)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": FPS}), encoding="utf-8")
    (root / "meta" / "rules.json").write_text(json.dumps(rules, ensure_ascii=False), encoding="utf-8")
    (root / "meta" / "scenarios.json").write_text(
        json.dumps({"scenarios": [scenario]}, ensure_ascii=False), encoding="utf-8")
    (root / "meta" / "safety_schema.json").write_bytes(
        (REPO_ROOT / "meta" / "safety_schema.json").read_bytes())
    if cameras is not None:
        (root / "meta" / "cameras.json").write_text(
            json.dumps(cl.shipped_cameras(cameras), ensure_ascii=False), encoding="utf-8")
    return root


def episodes_path(root):
    return root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"


def edit_episode(root, e, **columns):
    table = pq.read_table(episodes_path(root))
    rows = table.to_pylist()
    rows[e].update(columns)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), episodes_path(root))


def edit_notes(root, e, **members):
    notes = json.loads(pq.read_table(episodes_path(root)).to_pylist()[e]["notes"])
    notes.update(members)
    edit_episode(root, e, notes=json.dumps(notes, ensure_ascii=False))


def edit_steps(root, column, change):
    """Rewrite one step column of the first data file, as Python values."""
    path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    values = table.column(column).to_pylist()
    change(values)
    at = table.schema.get_field_index(column)
    pq.write_table(table.set_column(at, column, pa.array(values, table.schema.field(column).type)),
                   path)


def edit_json(path, change):
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc)
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
