"""npz/sidecar <-> published schema field mapping. The canonical copy.

Two consumers read this: generation/convert_lerobot.py, which builds the
published parquet from it, and the C9 reproduction hook, which uses it to
decide which published column each regenerated npz array has to match.

validator/ deliberately does NOT read it. A validator that trusts the
generator's own map is not validating -- the same reason the region codes
are hand-synced copies (CONTRIBUTING.md 「함께 고치는 곳」) rather than one
shared import.
tests/generation/test_field_map.py is where the two sides are checked
against each other, so an omission fails in CI rather than in a published
dataset.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StepField:
    """One schema §6 step field.

    kind:
      "array"   -- copied straight out of an npz array (npz names it)
      "derived" -- computed by the conversion (frame_index, event names)
      "sidecar" -- broadcast from an episode-level sidecar value
      "video"   -- a camera stream the sidecar's ``media`` names; LeRobot keeps it
                   as a video feature, not a parquet column
      "pack"    -- a member of the per-episode depth/mask pack; not a LeRobot
                   feature at all (the converter copies the pack beside the data).
                   No field is one since 2026-09-30: the camera canon declares no
                   pack and spec §6 marks its depth and segmentation
                   rows as not shipped. The pack code stays, switched off by the canon

    condition: None when every episode carries the field; otherwise the name of
    the condition under which it is carried (CAMERAS, HRI). The name is the key of
    meta/safety_schema.json conditional_required_step_fields -- the same string.
    """

    schema: str
    npz: str | None
    kind: str
    note: str = ""
    condition: str | None = None


# The camera condition. An episode meets it when its sidecar names ``media``
# (recorder.save(media=...)); a distribution declares it by shipping
# meta/cameras.json. Each consumer decides for itself which of the two it
# reads and passes the active conditions to the functions below -- with none,
# they return exactly what they returned before camera fields existed.
CAMERAS = "cameras"
# The scenario declares R-TS15066-SSM-02 (the same key in meta/safety_schema.json): the step fields
# of the hri scenarios -- the walking command, the person's pelvis, what the robot's cameras see of
# the person, and the two approach speeds the directed S_p is computed from.
HRI = "hri"


@dataclass(frozen=True)
class EpisodeField:
    """One schema §5 episode field.

    source:
      "sidecar"   -- generation/envs/s1a/recorder.py's episode json
      "scenarios" -- meta/scenarios.json, keyed by scenario_id
      "joint_map" -- meta/joint_map/{robot_model}.yaml
      "assets"    -- meta/assets.json
      "label"     -- generation/labeling's *.labels.json
      "const"     -- fixed by the schema itself
      "derived"   -- computed from another field

    condition: None when every episode carries the field; TABLE when only the episodes of a
    scenario that declares a conditions table do. The name is the key of meta/safety_schema.json
    scenario_conditional_episode_fields -- the same string.
    """

    schema: str
    source: str
    key: str | None
    note: str = ""
    condition: str | None = None


# The episode condition: the scenario's entry in meta/scenarios.json declares a conditions table (the
# same key). Its episodes carry the spec they were recorded under, the evidence of their label and their
# place in the table.
TABLE = "conditions"

# The evidence columns, in generation/labeling/evidence.py's order (its EVIDENCE_COLUMNS --
# tests/generation/test_field_map.py holds the two together). Spelled here because that module imports
# the labeller, which imports this map.
_EVIDENCE_NAMES = ("min_separation_margin_m", "first_contact_s",
                   *(f"force_ratio.BR-{n:02d}" for n in range(1, 13)), "force_ratio_max",
                   "human_first_seen_s", "human_last_seen_s")


# observation.contact.pressure is absent on purpose: the schema makes it an
# optional field (§6 「선택 필드」), carried only where contact area is measured, and
# PhysX reports force and not area.
STEP_FIELDS: tuple[StepField, ...] = (
    StepField("timestamp", "timestamp", "array"),
    StepField("frame_index", None, "derived", "0..N-1"),
    StepField("observation.state.joint_pos", "joint_pos", "array"),
    StepField("observation.state.joint_vel", "joint_vel", "array"),
    StepField("observation.state.joint_torque", "joint_torque", "array"),
    StepField("observation.root_pose", "root_pose", "array", "pos3 + quat4 (wxyz)"),
    StepField("observation.root_vel", "root_vel", "array"),
    StepField("observation.imu", "imu", "array", "quat4 (wxyz) + ang_vel_b3 + lin_acc_b3"),
    StepField("observation.contact.force", "contact_force", "array"),
    StepField("observation.contact.v_rel", "contact_v_rel", "array"),
    StepField("observation.contact.region_names", None, "sidecar", "contact_region_names"),
    StepField("observation.contact.type", "contact_type", "array"),
    StepField("action", "action", "array"),
    StepField("safety.com", "com", "array"),
    StepField("safety.zmp", "zmp", "array"),
    StepField("safety.stability_margin", "stability_margin", "array"),
    StepField("safety.min_distance_human", "min_distance_human", "array"),
    StepField("safety.separation_required", "separation_required", "array"),
    StepField("safety.separation_violation", "separation_violation", "array"),
    StepField("safety.event", None, "derived", "safety_event (bool columns) -> names"),
    # Camera fields (schema §6 「조건부 필드 — 카메라」). Last, so that with no condition
    # active the map reads exactly as it did before them. One stream and one pose per camera of
    # meta/cameras.json, in its camera order (seven) -- tests/generation/test_field_map.py
    # and tests/validator/test_safety_schema_canon.py derive the list from that file.
    StepField("observation.images.head_rgb", None, "video", "sidecar media.head_rgb",
              condition=CAMERAS),
    StepField("observation.images.head_down_rgb", None, "video", "sidecar media.head_down_rgb",
              condition=CAMERAS),
    StepField("observation.images.left_wrist_rgb", None, "video", "sidecar media.left_wrist_rgb",
              condition=CAMERAS),
    StepField("observation.images.right_wrist_rgb", None, "video", "sidecar media.right_wrist_rgb",
              condition=CAMERAS),
    StepField("observation.images.rear_rgb", None, "video", "sidecar media.rear_rgb",
              condition=CAMERAS),
    StepField("observation.images.third_person_rgb", None, "video",
              "sidecar media.third_person_rgb", condition=CAMERAS),
    StepField("observation.images.cctv_rgb", None, "video", "sidecar media.cctv_rgb",
              condition=CAMERAS),
    StepField("observation.bbox2d.human", "bbox_human", "array", condition=CAMERAS),
    StepField("observation.camera_pose.head_rgb", "camera_pose_head_rgb", "array",
              "pos3 + quat4 (wxyz), world, ROS optical", condition=CAMERAS),
    StepField("observation.camera_pose.head_down_rgb", "camera_pose_head_down_rgb", "array",
              "pos3 + quat4 (wxyz), world, ROS optical", condition=CAMERAS),
    StepField("observation.camera_pose.left_wrist_rgb", "camera_pose_left_wrist_rgb", "array",
              "pos3 + quat4 (wxyz), world, ROS optical", condition=CAMERAS),
    StepField("observation.camera_pose.right_wrist_rgb", "camera_pose_right_wrist_rgb", "array",
              "pos3 + quat4 (wxyz), world, ROS optical", condition=CAMERAS),
    StepField("observation.camera_pose.rear_rgb", "camera_pose_rear_rgb", "array",
              "pos3 + quat4 (wxyz), world, ROS optical", condition=CAMERAS),
    StepField("observation.camera_pose.third_person_rgb", "camera_pose_third_person_rgb",
              "array", "pos3 + quat4 (wxyz), world, ROS optical", condition=CAMERAS),
    StepField("observation.camera_pose.cctv_rgb", "camera_pose_cctv_rgb", "array",
              "pos3 + quat4 (wxyz), world, ROS optical", condition=CAMERAS),
    # hri fields (schema §6 「조건부 필드 — 사람과 로봇」), after the camera fields for the same reason.
    # The visible fractions follow the robot's own cameras of meta/cameras.json (type mounted) --
    # tests/validator/test_safety_schema_canon.py derives the list from that file.
    StepField("observation.command_velocity", "command_velocity", "array",
              "vx, vy, yaw rate -- the walking policy's command, robot base frame", condition=HRI),
    StepField("observation.human.root_pose", "human_root_pose", "array",
              "pos3 + quat4 (wxyz), world -- the dummy's pelvis", condition=HRI),
    StepField("observation.human.root_vel", "human_root_vel", "array",
              "lin3 + ang3, world -- the dummy's pelvis", condition=HRI),
    *(StepField(f"observation.human_visible_fraction.{cam}", f"human_visible_fraction_{cam}", "array",
                "share of the body's sample points in view and unoccluded, 0..1", condition=HRI)
      for cam in ("head_rgb", "head_down_rgb", "left_wrist_rgb", "right_wrist_rgb", "rear_rgb")),
    StepField("safety.approach_speed_robot", "approach_speed_robot", "array",
              "the robot's closest point (box included) toward the person, m/s, >= 0", condition=HRI),
    StepField("safety.approach_speed_human", "approach_speed_human", "array",
              "the person's closest point toward the robot, m/s, >= 0", condition=HRI),
    # Spec §6's observation.depth.* and observation.segmentation.* rows are not mapped: not
    # shipped since 2026-09-30, the segmentation read for the box alone.
)

# Every condition some step field carries; an "active" set may name only these.
CONDITIONS = frozenset(f.condition for f in STEP_FIELDS if f.condition is not None)

EPISODE_FIELDS: tuple[EpisodeField, ...] = (
    EpisodeField("action_space", "sidecar", "action_space"),
    EpisodeField("termination", "sidecar", "termination"),
    EpisodeField("degenerate", "sidecar", "degenerate"),
    # 사이드카와 라벨 여러 곳에서 모은 단서 묶음. 변환기가 조립한다
    # (convert_lerobot.build_episode_meta 의 notes 딕셔너리).
    EpisodeField("notes", "derived", None),
    EpisodeField("episode_id", "sidecar", "episode_id"),
    EpisodeField("schema_version", "const", None, "0.2"),
    EpisodeField("domain", "scenarios", "domain"),
    EpisodeField("source", "const", None, "sim"),
    EpisodeField("robot_model", "sidecar", "robot_model"),
    EpisodeField("robot_embodiment", "assets", "robots.{robot_model}.embodiment"),
    EpisodeField("dof", "joint_map", "dof"),
    EpisodeField("joint_names", "joint_map", "joint_names"),
    EpisodeField("scenario_id", "sidecar", "scenario_id"),
    EpisodeField("task_archetype", "scenarios", "task_archetype"),
    EpisodeField("hazard_class", "scenarios", "hazard_class"),
    EpisodeField("outcome", "label", "outcome"),
    EpisodeField("severity", "label", "severity"),
    EpisodeField("triggered_rules", "label", "triggered_rules"),
    EpisodeField("safety_function", "scenarios", "safety_function"),
    EpisodeField("standard_refs", "derived", None, "scenarios.rules -> rules.json standard"),
    EpisodeField("measurement_method", "sidecar", "sim_backend.physics"),
    EpisodeField("scene_assets", "assets", "scene + robots.{robot_model}"),
    EpisodeField("fps", "derived", None, "round(1 / step_dt_from_timestamps(timestamp))"),
    EpisodeField("duration_s", "derived", None, "steps * step_dt_from_timestamps(timestamp)"),
    # Required of a simulated episode only (schema §5 「sim 필수」; the validator's
    # CONDITIONAL_REQUIRED_EPISODE_FIELDS). Every episode this pipeline converts is one.
    EpisodeField("randomization_seed", "sidecar", "seed"),
    EpisodeField("sim_backend", "sidecar", "sim_backend"),
    EpisodeField("run_start_seq", "sidecar", "run_start_seq"),
    # Required of an episode whose scenario declares a conditions table (schema §5 「조건표 필수」; the
    # validator's SCENARIO_CONDITIONAL_EPISODE_FIELDS), in the canon's order. The converter builds them
    # (convert_lerobot.table_columns): none is a sidecar value as it stands, so C9 compares none of them.
    EpisodeField("episode_spec", "derived", None, "the sidecar and the canon -> episode_spec.build_episode_spec",
                 condition=TABLE),
    *(EpisodeField(f"evidence.{name}", "label", f"evidence.{name}", "null (NaN) where there is no value",
                   condition=TABLE) for name in _EVIDENCE_NAMES),
    EpisodeField("condition_design", "derived", None, "randomization.condition.design", condition=TABLE),
    EpisodeField("grid_cell_robot", "derived", None, "randomization.condition.grid_cell[0]; -1 on a random row",
                 condition=TABLE),
    EpisodeField("grid_cell_human", "derived", None, "randomization.condition.grid_cell[1]; -1 on a random row",
                 condition=TABLE),
)


def episode_fields(table: bool) -> tuple[EpisodeField, ...]:
    """The episode fields an episode carries: every unconditional one, plus the TABLE ones when its
    scenario declares a conditions table (``table``), in map order."""
    return tuple(f for f in EPISODE_FIELDS if f.condition is None or (table and f.condition == TABLE))


def active_step_fields(active: Iterable[str] = ()) -> tuple[StepField, ...]:
    """The step fields an episode carries when the conditions in ``active`` hold:
    every unconditional field, plus those whose condition is active, in map order.

    A bare string is refused rather than iterated into letters, and an unknown
    condition name rather than ignored -- either way the conditional fields
    would drop out without a word.
    """
    if isinstance(active, str):
        raise TypeError(f"active 는 조건 이름의 모음이어야 합니다 (문자열 하나가 아니라): {active!r}")
    on = frozenset(active)
    unknown = sorted(on - CONDITIONS)
    if unknown:
        raise ValueError(f"대응표에 없는 조건입니다: {unknown} (있는 것: {sorted(CONDITIONS)})")
    return tuple(f for f in STEP_FIELDS if f.condition is None or f.condition in on)


def npz_backed_step_fields(active: Iterable[str] = ()) -> tuple[tuple[str, str], ...]:
    """(schema field, npz array) for every field copied straight across.

    The C9 hook compares exactly these columns between a regenerated npz and
    the published parquet; anything derived is excluded because a mismatch
    there would be a conversion bug, not a reproduction failure, and folding
    the two together makes the hook's red light mean two different things.
    Videos and packs are not npz arrays and never appear here.

    ``active``: the conditions that hold (CAMERAS, HRI). With none -- every v0.1
    caller -- the result is the unconditional arrays, as it always was.
    """
    return tuple((f.schema, f.npz) for f in active_step_fields(active) if f.kind == "array")


# The walking command is in every recording, those of S1a and S1d as well, so it says nothing about
# the condition; the other hri arrays do.
_HRI_LONG_RECORDED = ("command_velocity",)


def hri_recorded(names: Iterable[str]) -> bool:
    """Whether a recording carries the hri arrays: True with all of them, False with none (a v0.1
    or S1d recording). Some but not all, or all without the walking command, did not come out of
    one save -- refused rather than half-converted."""
    present = set(names)
    markers = {f.npz for f in STEP_FIELDS if f.condition == HRI and f.kind == "array"}
    markers -= set(_HRI_LONG_RECORDED)
    found = markers & present
    if not found:
        return False
    if found != markers:
        raise ValueError(f"npz 에 사람과 로봇(hri) 배열이 일부만 있습니다 — 없는 것: {sorted(markers - found)}")
    missing = [n for n in _HRI_LONG_RECORDED if n not in present]
    if missing:
        raise ValueError(f"npz 에 hri 배열이 있는데 {missing} 가 없습니다")
    return True


def step_dt_from_timestamps(timestamps) -> float:
    """The recording period, in seconds, from an episode's own timestamp array.

    NOT from the sidecar's ``global_time_offset_s``. That field is the global
    sim-clock time at which the episode *started* (episode k of a run begins
    near 20k seconds), and it only looks like a step period for episode 0,
    where the value happens to be one step. Reading it as dt gives a
    plausible 50 fps for the first episode of every batch and nonsense for
    the rest -- measured on the 2026-08-22 batch, where it ran 0.02, 20.02,
    40.02, ... across the ten episodes.

    recorder.step() writes ``(episode_length_buf - 1) * step_dt``, so the
    differences are exactly constant; an array where they are not did not
    come from that recorder, and is rejected rather than averaged.
    """
    ts = np.asarray(timestamps, dtype=float)
    # Not reshape(-1): a (2, 2) array flattened into [0, .02, .04, .06] and
    # read as a perfectly regular 50 Hz timeline.
    if ts.ndim != 1:
        raise ValueError(f"timestamp 는 1차원이어야 합니다: {ts.shape}")
    if ts.size < 2:
        raise ValueError(f"기록 주기를 잴 수 없습니다 — 스텝 {ts.size}개")
    diffs = np.diff(ts)
    if not np.all(np.isfinite(diffs)) or np.any(diffs <= 0):
        raise ValueError("timestamp 가 단조 증가하지 않습니다")
    dt = float(diffs[0])
    if not np.allclose(diffs, dt, rtol=0, atol=1e-9):
        raise ValueError(
            f"기록 주기가 일정하지 않습니다: {float(diffs.min())!r}~{float(diffs.max())!r}"
        )
    return dt
