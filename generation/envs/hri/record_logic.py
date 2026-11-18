"""The Isaac-free half of the recorder for the scenarios that declare R-TS15066-SSM-02 (S1e, S1f, S1g --
generation/envs/s1a/recorder.py): which contact columns count, the step's hri columns, the rack-contact event, the
sidecar's members, and whether the dummy was usable.

The contact columns of these scenarios carry only what the robot's links and the box it carries (after it is let go
too) gave the person. The dummy's contact sensor is filtered against each robot link and the box, one prim per
expression (generation/envs/hri/command_logic.contact_filters), so its force matrix (B, F, 3) is the force on each of
the person's B bodies from each of the F filters -- measured in Isaac Sim 6.0.1 / Isaac Lab 3.0.0 on 2026-10-20: the
dummy's feet standing on the floor, a fallen dummy lying on it and a rack the robot pushed into gave the matrix
nothing, and a body's row summed to its net force whenever only the robot or the box touched it. The force a body
gets is therefore read from the robot and box columns of the matrix (attribute_contacts), never the body's net force,
which counts the floor: the larger of |sum over those columns of M[b, f]| and the largest |M[b, f]| among them
(command_logic.body_force -- the function the command term lets the box and the hold go by), so two robot parts
pressing one body from opposite sides -- a pinch -- record one side's push instead of cancelling. The matrix holds
normal forces only, as the net force does.

What a robot link or the held box takes from anything that is not the person -- a rack, or the floor once the robot
falls -- is the rest of its net force: |net + sum over b of M[b, col]| (link_residuals, box_residual). The rack-contact
event (rack_contact_edge) is the step such a force on a robot link that is not a foot, or on the box while it is
held, starts to exceed RACK_CONTACT_THRESHOLD_N. After the box is let go its own contact sensor counts the floor it
lands on, so only the held box raises it. It does not change an episode's outcome.

Pure numpy; the recorder and generation/envs/s1a/substep_contact.py call these, and the tests call them on arrays
written by hand.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from generation import human_script, sites
from generation.directed_separation import DIRECTED_SSM_PARAMETERS, scenario_declares_directed_ssm
from generation.envs.hri import command_logic
from generation.envs.s1a.mdp.safety import classify_contact_type, reduce_region_peaks
from generation.envs.s1d import jointed_walk_metrics as walk_metrics
from generation.field_map import HRI, STEP_FIELDS

REPO = Path(__file__).resolve().parents[3]
SCENARIOS_JSON = REPO / "meta/scenarios.json"

# safety.event of these scenarios. The protective stop is the project's own SSM supervisor's, which the baseline robot
# recorded here does not run.
HRI_EVENT_NAMES = ("ssm_violation", "contact_onset", "rack_contact")
# A robot link that is not a foot, or the held box, taking more than this from something that is not the person.
# Self-set: above the matrix's noise (0.000 N on robot links walking free of racks, 2026-10-20) and well below a
# rack push (2.4 kN).
RACK_CONTACT_THRESHOLD_N = 5.0
# The npz keys of the visible fractions start with this; the camera follows (field_map).
FRACTION_PREFIX = "human_visible_fraction_"

# The npz keys of a recording that has no step (EpisodeRecorder.save): each a (0,) array, so a reader that checks key
# by key finds the file it expects.
PLAIN_ZERO_STEP_KEYS = (
    "timestamp", "joint_pos", "joint_vel", "joint_torque",
    "root_pose", "imu", "com", "zmp", "root_vel", "action",
    "command_velocity", "contact_force", "min_distance_human",
    "separation_required", "separation_violation", "stability_margin",
    "hold_force", "hold_torque", "ground_reaction_force", "region_z",
    "contact_v_rel", "contact_type",
    "dummy_yaw_realized",
    "safety_event", "stance_foot",
)

# What the episode metadata says the contact columns of these scenarios cannot see (contact_onset_unresolved_* and
# contact_type_unresolved_reason): the robot's and the box's force on every body of the person is read from the
# matrix, the dummy's feet included, so no robot-to-person path is left unseen by contact_onset or by the type.
CONTACT_ONSET_GAP_REASON = (
    "로봇 링크와 든 상자가 인형의 바디마다 준 힘을 접촉 센서의 거른 행렬에서 읽으므로 인형의 발을 포함한 모든 바디의 "
    "접촉을 잡음 — 랙 · 바닥과의 힘은 행렬에 들지 않음. 인형의 발은 12부위 밖이라 부위별 접촉력에는 기록하지 않음. "
    "행렬은 법선력만 담음(마찰력 없음).")
CONTACT_TYPE_GAP_REASON = (
    "접촉 유형은 부위마다 로봇 링크와 든 상자가 준 힘과 그 서브스텝의 부위 높이로 매기므로 로봇 발이 넘어진 인형을 "
    "누르는 접촉도 매겨짐. 인형의 발에 닿은 접촉은 12부위 밖이라 유형이 없음.")

# The rows dummy_degenerate_reason reads, one entry per recorded step: the head's height (m), the torso's tilt (deg),
# whether the hold had let go, whether the script had set off and had stopped walking, and each region's height (m,
# the first body of each code, in the order of the codes handed with them).
DUMMY_ROW_KEYS = ("head_z", "tilt", "released", "triggered", "walk_ended", "region_heights")


def records_hri(scenario_id: str, doc: Mapping | None = None) -> bool:
    """Whether an episode of ``scenario_id`` records the hri columns: its meta/scenarios.json (or ``doc``) entry declares
    R-TS15066-SSM-02 -- the converter refuses a recording that disagrees (check_hri_declared)."""
    doc = json.loads(SCENARIOS_JSON.read_text(encoding="utf-8")) if doc is None else doc
    entry = next((e for e in doc.get("scenarios", []) if e.get("scenario_id") == scenario_id), None)
    if entry is None:
        raise ValueError(f"meta/scenarios.json 에 {scenario_id!r} 가 없습니다")
    return scenario_declares_directed_ssm(entry)


def hri_zero_step_keys() -> tuple[str, ...]:
    """The npz key of every hri array field of the field map, in its order."""
    return tuple(f.npz for f in STEP_FIELDS if f.condition == HRI and f.kind == "array")


def zero_step_arrays(*, hri: bool) -> dict[str, np.ndarray]:
    """The arrays of a recording that has no step: PLAIN_ZERO_STEP_KEYS, and with ``hri`` every hri key as well, so the
    converter reads it as an hri recording rather than half of one."""
    keys = list(PLAIN_ZERO_STEP_KEYS)
    if hri:
        keys += [k for k in hri_zero_step_keys() if k not in keys]
    return {key: np.zeros((0,)) for key in keys}


# ---- what the robot and the box gave the person ----

def _matrix(force_matrix) -> np.ndarray:
    m = np.asarray(force_matrix, float)
    if m.ndim != 3 or m.shape[2] != 3 or not np.isfinite(m).all():
        raise ValueError(f"힘 행렬이 유한한 수의 (사람 바디, 필터, 3) 이 아닙니다: {m.shape}")
    return m


def _columns(robot_cols: Sequence[int], box_col: int | None, count: int) -> list[int]:
    cols = [int(c) for c in robot_cols] + ([] if box_col is None else [int(box_col)])
    if len(set(cols)) != len(cols) or any(not 0 <= c < count for c in cols):
        raise ValueError(f"로봇 · 상자 열이 서로 다른 0..{count - 1} 의 열이 아닙니다: robot {list(robot_cols)} · box {box_col}")
    return sorted(cols)


def attribute_contacts(force_matrix, robot_cols: Sequence[int], box_col: int | None) -> np.ndarray:
    """(B,) the force each of the person's bodies got from the robot's links (``robot_cols``) and the box (``box_col``,
    None without one): command_logic.body_force of those columns -- the larger of |sum over them of M[b, f]| and the
    largest |M[b, f]| among them (N), so a pinch between two of them is one side's push. Every other column of the
    matrix -- a floor or a rack, if the sensor were filtered against one -- is left out."""
    m = _matrix(force_matrix)
    cols = _columns(robot_cols, box_col, m.shape[1])
    return command_logic.body_force(m[:, cols, :])


def link_residuals(robot_net, force_matrix, link_cols: Sequence[int | None]) -> np.ndarray:
    """(L,) what each robot link took from anything that is not the person (N): |robot_net[l] + sum over b of
    M[b, link_cols[l]]| -- the link's net force less the person's share of it (the person's force on the link is minus
    the link's on the person). A link without a column (None) is its whole net force."""
    net = np.asarray(robot_net, float)
    m = _matrix(force_matrix)
    if net.ndim != 2 or net.shape[1] != 3 or len(link_cols) != net.shape[0] or not np.isfinite(net).all():
        raise ValueError(f"로봇 링크의 알짜 힘 {net.shape} 이 링크 열 {len(link_cols)} 개의 유한한 (링크, 3) 이 아닙니다")
    share = m.sum(axis=0)                                   # (F, 3): what each filter gave the person
    out = net.copy()
    for k, col in enumerate(link_cols):
        if col is not None:
            if not 0 <= int(col) < m.shape[1]:
                raise ValueError(f"링크 {k} 의 열 {col} 이 행렬의 열 0..{m.shape[1] - 1} 밖입니다")
            out[k] = net[k] + share[int(col)]
    return np.linalg.norm(out, axis=-1)


def box_residual(box_net, force_matrix, box_col: int) -> float:
    """What the box took from anything that is not the person (N): |box_net + sum over b of M[b, box_col]|."""
    net = np.asarray(box_net, float).reshape(-1)
    m = _matrix(force_matrix)
    if net.shape != (3,) or not np.isfinite(net).all() or not 0 <= int(box_col) < m.shape[1]:
        raise ValueError(f"상자의 알짜 힘 {net.shape} 또는 상자 열 {box_col} 이 맞지 않습니다")
    return float(np.linalg.norm(net + m[:, int(box_col), :].sum(axis=0)))


def rack_residual(link_residual, non_foot, box_residual_n: float | None, *, box_held: bool) -> float:
    """The step's largest force from something that is not the person on a robot link that is not a foot or, while it
    was held during the step, on the box (N). The feet carry the floor every step; a box let go lies on it."""
    links = np.asarray(link_residual, float)
    mask = np.asarray(non_foot, bool)
    if links.shape != mask.shape:
        raise ValueError(f"링크의 힘 {links.shape} 와 발 아닌 링크 표시 {mask.shape} 의 길이가 다릅니다")
    out = float(links[mask].max()) if mask.any() else 0.0
    if box_held and box_residual_n is not None:
        out = max(out, float(box_residual_n))
    return out


def rack_contact_edge(prev: bool, residual_n: float, threshold_n: float = RACK_CONTACT_THRESHOLD_N) -> tuple[bool, bool]:
    """(the force from something that is not the person is over the threshold now, and that started this step)."""
    now = float(residual_n) > float(threshold_n)
    return now, now and not bool(prev)


def matrix_columns(filters: Sequence[str], link_names: Sequence[str], box_prim: str | None,
                   non_foot: Sequence[bool]) -> tuple[list[int], int | None, list[int | None]]:
    """(the robot link columns, the box column, each robot link's column) of a force matrix whose columns are the
    prims ``filters`` name (their last path parts), for the robot links ``link_names`` (in the robot contact sensor's
    order) and the box ``box_prim`` (None without one). Matched by name. Every link that is not a foot needs a column
    -- its share of the person would otherwise count as a rack contact."""
    names = [str(f).rsplit("/", 1)[-1] for f in filters]
    if len(set(names)) != len(names):
        raise ValueError(f"필터의 프림 이름이 겹칩니다: {names}")
    where = {n: k for k, n in enumerate(names)}
    links = list(link_names)
    mask = list(non_foot)
    if len(mask) != len(links):
        raise ValueError(f"발 아닌 링크 표시 {len(mask)} 개가 링크 {len(links)} 개와 다릅니다")
    missing = [n for n, keep in zip(links, mask) if keep and n not in where]
    if missing:
        raise ValueError(f"발 아닌 로봇 링크 {missing} 의 필터가 인형의 접촉 센서에 없습니다")
    link_cols = [where.get(n) for n in links]
    robot_cols = sorted(c for c in link_cols if c is not None)
    box_col = None
    if box_prim is not None:
        if box_prim not in where:
            raise ValueError(f"든 상자 {box_prim!r} 의 필터가 인형의 접촉 센서에 없습니다: {names}")
        box_col = where[box_prim]
    return robot_cols, box_col, link_cols


def region_contacts(human_force, human_z, codes: Sequence[str | None], *, contact_eps_n: float,
                    pin_height_m: float) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """(the region codes in order, each region's force (N), its height (m), its contact type) of one step, from each
    of the person's bodies' peak force from the robot and the box and its height at that substep (sensor order;
    ``codes`` None for a foot, which is no region). A region is typed only where the robot or the box gave it force,
    so a typed step is always a step something touched the person."""
    force = np.asarray(human_force, float)
    z = np.asarray(human_z, float)
    codes = list(codes)
    if not (force.shape == z.shape == (len(codes),)):
        raise ValueError(f"바디의 힘{force.shape} · 높이{z.shape} · 부위 코드 {len(codes)} 개의 길이가 다릅니다")
    region = np.array([c is not None for c in codes])
    named = [c for c in codes if c is not None]
    by_f, by_z, by_gate = reduce_region_peaks(named, force[region], z[region], force[region] > contact_eps_n,
                                              contact_eps_n=contact_eps_n)
    order = sorted(set(named))
    f = np.array([by_f[c] for c in order])
    zz = np.array([by_z[c] for c in order])
    gate = np.array([by_gate[c] for c in order])
    types = classify_contact_type(f, zz, robot_contact=gate, contact_eps_n=contact_eps_n, pin_height_m=pin_height_m)
    return order, f, zz, types


def touched(human_force, contact_eps_n: float) -> bool:
    """Whether the robot or the box gave any of the person's bodies, the feet included, force this step -- the box is
    let go at the first such step, and contact_onset is its edge."""
    return bool((np.asarray(human_force, float) > float(contact_eps_n)).any())


# ---- the step's hri columns ----

def fraction_cameras() -> tuple[str, ...]:
    """The robot cameras whose visible fraction is a step column, in the field map's order."""
    return tuple(k[len(FRACTION_PREFIX):] for k in hri_zero_step_keys() if k.startswith(FRACTION_PREFIX))


def _quat_xyzw_to_wxyz(pose7: np.ndarray) -> np.ndarray:
    return np.concatenate([pose7[:3], [pose7[6]], pose7[3:6]])


def hri_step_columns(hri, *, pelvis_pose_xyzw, pelvis_vel, fractions: Mapping[str, float]) -> dict[str, object]:
    """The step's npz entries an hri recording adds or sets: the surface distance, S_p, its breach and the two approach
    speeds of ``hri`` (hri_geometry.HriStep -- the command term's step), the pelvis's pose (pos3 + quat wxyz, from the
    engine's xyzw) and velocity (lin3 + ang3), and each robot camera's visible fraction."""
    pose = np.asarray(pelvis_pose_xyzw, float).reshape(-1)
    vel = np.asarray(pelvis_vel, float).reshape(-1)
    if pose.shape != (7,) or vel.shape != (6,) or not (np.isfinite(pose).all() and np.isfinite(vel).all()):
        raise ValueError(f"골반의 자세 {pose.shape} · 속도 {vel.shape} 가 유한한 7 · 6 개가 아닙니다")
    cams = fraction_cameras()
    if set(fractions) != set(cams):
        raise ValueError(f"보이는 몫의 카메라 {sorted(fractions)} 가 열의 카메라 {sorted(cams)} 와 다릅니다")
    out: dict[str, object] = {
        "min_distance_human": float(hri.distance),
        "separation_required": float(hri.s_p),
        "separation_violation": bool(hri.violated),
        "approach_speed_robot": float(hri.v_r),
        "approach_speed_human": float(hri.v_h),
        "human_root_pose": _quat_xyzw_to_wxyz(pose),
        "human_root_vel": vel.copy(),
    }
    for cam in cams:
        value = float(fractions[cam])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{cam} 의 보이는 몫이 0..1 이 아닙니다: {value!r}")
        out[f"{FRACTION_PREFIX}{cam}"] = value
    return out


# ---- the sidecar ----

def _plain(value: object) -> object:
    """A JSON value of ``value``: tuples as lists, numpy numbers as Python ones."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def condition_record(index: int, row: Mapping, site: sites.Site, plan: human_script.HumanPlan, *,
                     task_started_step: int, design: str = "random",
                     grid_cell: tuple[int, int] | None = None) -> dict[str, object]:
    """randomization.condition of an episode: its conditions-table row number, whether the row is a random row or a
    grid row (``design``) and which cell of the grid (``grid_cell`` -- robot cell, human cell; null for a random
    row), the row's columns and fixed values as drawn, the places they put the robot and the person (the row's site -- conditions.row_site -- and the person's
    script -- human_script.plan_for), and the step the robot's task came at (-1: never)."""
    out: dict[str, object] = {"row": int(index)}
    out.update(_plain(dict(row)))
    derived = {
        "robot_start": {"x": site.robot_start.x, "y": site.robot_start.y, "yaw_deg": site.robot_start.yaw_deg},
        "robot_goal": {"x": site.robot_goal.x, "y": site.robot_goal.y, "yaw_deg": site.robot_goal.yaw_deg},
        "robot_path": _plain(dict(site.path)),
        "human_plan": _plain(human_script.plan_record(plan)),
        "task_started_step": int(task_started_step),
        "design": design,
        "grid_cell": None if grid_cell is None else [int(grid_cell[0]), int(grid_cell[1])],
    }
    clash = sorted(set(derived) & set(out))
    if clash:
        raise ValueError(f"조건표의 열 이름이 유도한 자리의 이름과 겹칩니다: {clash}")
    out.update(_plain(derived))
    return out


def hri_ssm_params(params: Mapping[str, float], table: Mapping) -> dict[str, object]:
    """The sidecar's ssm_params of these scenarios: R-TS15066-SSM-02's five parameters, as the command term read them,
    and its stop table under ``stop_table`` -- the episode's S_p column was computed with them."""
    names = set(params)
    if names != set(DIRECTED_SSM_PARAMETERS):
        raise ValueError(f"S_p 의 파라미터가 {list(DIRECTED_SSM_PARAMETERS)} 가 아닙니다: {sorted(names)}")
    if not isinstance(table, Mapping):
        raise ValueError(f"정지 시험 표가 객체가 아닙니다: {type(table).__name__}")
    out: dict[str, object] = {name: float(params[name]) for name in DIRECTED_SSM_PARAMETERS}
    out["stop_table"] = _plain(copy.deepcopy(dict(table)))
    return out


def camera_in_box_counts() -> dict[str, int]:
    """The sidecar's camera_in_box_steps before the first step: every robot camera with a visible-fraction column, 0."""
    return {cam: 0 for cam in fraction_cameras()}


def count_cameras_in_box(counts: Mapping[str, int], inside: Sequence[str]) -> dict[str, int]:
    """``counts`` with one more step for each camera of ``inside`` -- the robot cameras that sat inside the held box at
    this step (the command term's cameras_in_box). On such a step that camera's visible fraction leaves the box out
    (visibility.robot_camera_fractions) while its picture shows the box's inside; the count says how often. A camera
    that is not one of ``counts``, or one named twice, is refused."""
    unknown = sorted(set(inside) - set(counts))
    if unknown:
        raise ValueError(f"상자 안의 카메라 {unknown} 가 보이는 몫의 카메라 {sorted(counts)} 가 아닙니다")
    if len(set(inside)) != len(inside):
        raise ValueError(f"상자 안의 카메라를 두 번 셌습니다: {list(inside)}")
    out = {cam: int(n) for cam, n in counts.items()}
    for cam in inside:
        out[cam] += 1
    return out


def step_or_none(value: object) -> int:
    """A step the sidecar records (task_completed_step, box_released_step): an int from -1 (never)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < -1:
        raise ValueError(f"스텝이 -1 이상의 정수가 아닙니다: {value!r}")
    return int(value)


def task_stages_record(names: Sequence[str], end_steps: Sequence[int], started_step: int,
                       fps: int) -> list[dict]:
    """The sidecar's task_stages: for each of the task's stages (task_path.StageTracker's ``names`` and
    ``end_steps``), its name, whether it ended, the step it ended at (-1: open) and the seconds it took -- stage k runs
    from the end of stage k - 1 (from ``started_step``, the step the task came at, for the first) to its own end,
    (end - start) / ``fps``; None while open. Refused: a step for each stage missing, a step that is not an int from
    -1, a stage that ended after an open one, before the one before it or the task, or a step rate that is not a
    positive int."""
    names, ends = list(names), [step_or_none(s) for s in end_steps]
    if len(names) != len(ends) or not all(isinstance(n, str) for n in names):
        raise ValueError(f"단계 이름 {names} 과 끝난 스텝 {ends} 이 단계마다 하나씩이 아닙니다")
    if isinstance(fps, (bool, np.bool_)) or not isinstance(fps, (int, np.integer)) or fps <= 0:
        raise ValueError(f"스텝의 빠르기(fps)가 양의 정수가 아닙니다: {fps!r}")
    start: int | None = step_or_none(started_step)
    out = []
    for name, end in zip(names, ends):
        if end < 0:
            out.append({"name": name, "completed": False, "end_step": -1, "duration_s": None})
            start = None                                 # the stages after an open one are open too
            continue
        if start is None or start < 0 or end < start:
            before = "열려 있습니다" if start is None or start < 0 else f"스텝 {start} 에 끝났습니다"
            raise ValueError(f"단계 {name!r} 가 스텝 {end} 에 끝났는데 그 앞 단계(첫 단계는 과제가 온 때)가 {before}")
        out.append({"name": name, "completed": True, "end_step": end, "duration_s": (end - start) / int(fps)})
        start = end
    return out


# ---- the dummy ----

def dummy_degenerate_reason(rows: Mapping[str, Sequence], release_step: int, *, codes: Sequence[str],
                            standing: Mapping[str, float]) -> str | None:
    """Why the episode's dummy was not usable before the robot or the box first touched it, or None.

    ``rows`` holds DUMMY_ROW_KEYS, one entry per recorded step; ``release_step`` is the first step the person was
    touched (the episode's first contact_onset, the box let go at it), -1 for never. From step 0 to the step before it:

    * the dummy fell -- the head under jointed_walk_metrics.HEAD_FALL_M or the torso tilted past TILT_FALL_DEG;
    * while held (``released`` False), the torso tilted to jointed_walk_metrics.TILT_MAX_DEG or more -- the walk's
      tilt mark;
    * while held and standing, a region stood off its standing height (jointed_walk_metrics.standing_height_problems
      against ``standing``, the regions' heights in the dimensions, for the ``codes`` of region_heights) -- by more
      than STANDING_TOL_M before the script set off, or by more than STOPPED_STANDING_TOL_M after it had stopped
      (a step that is both is judged before setting off). A walking step lifts the legs and is not a standing pose.

    The walk's assist marks (the hand's force across the plane, the tilt torque, the feet's share) are not part of it:
    they were set on long walks and a window of a few walking steps misses them at every start. The robot is not
    looked at: a robot that falls is a recorded outcome, not a broken episode."""
    missing = [k for k in DUMMY_ROW_KEYS if k not in rows]
    if missing:
        raise ValueError(f"인형의 행에 {missing} 가 없습니다")
    n = len(rows["head_z"])
    if any(len(rows[k]) != n for k in DUMMY_ROW_KEYS):
        raise ValueError(f"인형의 행 길이가 다릅니다: {[(k, len(rows[k])) for k in DUMMY_ROW_KEYS]}")
    unknown = [c for c in codes if c not in standing]
    if unknown or len(codes) != np.asarray(rows["region_heights"]).reshape(n, -1).shape[1]:
        raise ValueError(f"부위 높이의 코드 {list(codes)} 에 서 있는 높이가 없거나({unknown}) 행의 부위 수와 다릅니다")
    end = n if int(release_step) < 0 else min(int(release_step), n)
    if end <= 0:
        return None
    head = np.asarray(rows["head_z"], float)[:end]
    tilt = np.asarray(rows["tilt"], float)[:end]
    down = np.flatnonzero((head < walk_metrics.HEAD_FALL_M) | (tilt > walk_metrics.TILT_FALL_DEG))
    if down.size:
        k = int(down[0])
        return (f"사람(인형)이 첫 접촉 전에 넘어짐 — 스텝 {k}: 머리 높이 {head[k]:.3f} m · 기울기 {tilt[k]:.1f}° "
                f"(머리 {walk_metrics.HEAD_FALL_M} m 아래 또는 기울기 {walk_metrics.TILT_FALL_DEG}° 넘음)")
    held = ~np.asarray(rows["released"], bool)[:end]
    tipped = np.flatnonzero(held & (tilt >= walk_metrics.TILT_MAX_DEG))
    if tipped.size:
        k = int(tipped[np.argmax(tilt[tipped])])
        return (f"붙잡힌 인형이 첫 접촉 전에 기울어짐 — 스텝 {k}: 기울기 {tilt[k]:.1f}° "
                f"({walk_metrics.TILT_MAX_DEG}° 이상)")
    before_start = held & ~np.asarray(rows["triggered"], bool)[:end]
    after_stop = held & np.asarray(rows["walk_ended"], bool)[:end] & ~before_start
    heights = np.asarray(rows["region_heights"], float)[:end]
    for k in np.flatnonzero(before_start | after_stop):
        tol = walk_metrics.STANDING_TOL_M if before_start[k] else walk_metrics.STOPPED_STANDING_TOL_M
        problems = walk_metrics.standing_height_problems(list(codes), heights[k], standing, tol)
        if problems:
            when = "출발 전" if before_start[k] else "멈춘 뒤"
            return (f"서 있는 인형의 높이가 첫 접촉 전에 맞지 않음 — 스텝 {int(k)}({when}, 허용 {tol * 100:g} cm): "
                    f"{problems[0]}")
    return None
