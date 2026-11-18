"""대응표가 스키마의 필수 필드를 하나도 빠뜨리지 않는지.

generation/ 은 런타임에 validator/ 를 import 하지 않는다 — 검증기가 생성기를
신뢰하면 검증이 아니게 되고, 그 반대도 같다. 두 쪽을 일부러 따로 두는 대신
어긋남은 이 시험이 잡는다 (설계문서 §6.2).
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from generation.field_map import (
    CAMERAS,
    EPISODE_FIELDS,
    HRI,
    STEP_FIELDS,
    TABLE,
    active_step_fields,
    episode_fields,
    hri_recorded,
    npz_backed_step_fields,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "docs" / "schema" / "dataset-schema-v0.2.md"

# C3 가 따로 강제하므로 REQUIRED_STEP_FIELDS 에는 없지만 필수인 필드.
C3_ENFORCED = {"observation.contact.type"}
# The optional step field no simulator records: the measured contact area (spec §6). The converter
# carries every other optional field.
NOT_RECORDED = {"observation.contact.pressure"}

# How spec §6 marks a row it keeps for a field that is not shipped: the depth and segmentation
# rows, members of the pack the camera canon no longer declares.
NOT_SHIPPED_MARK = "싣지 않음"
# Spec §6's names for them: observation.depth.{cam} and observation.segmentation.{cam}.
PACK_MEMBER_PREFIXES = ("observation.depth.", "observation.segmentation.")


def _validator():
    import validator.validate_dataset as v
    return v


def _carried_unconditionally(v) -> set[str]:
    """The step fields every episode of this pipeline carries: the required ones and the optional ones
    a simulator records."""
    return set(v.REQUIRED_STEP_FIELDS) | (set(v.OPTIONAL_STEP_FIELDS) - NOT_RECORDED)


def _canon_conditions() -> dict:
    schema = json.loads((REPO_ROOT / "meta" / "safety_schema.json").read_text(encoding="utf-8"))
    return schema["conditional_required_step_fields"]


def _spec_step_rows() -> dict[str, str]:
    """Spec §6's step field table: field name -> the rest of its row."""
    section = SPEC.read_text(encoding="utf-8").split("## 6. 스텝 단위 시계열")[1].split("\n## 7.")[0]
    rows = dict(re.findall(r"^\| `([^`]+)` \|(.*)$", section, re.M))
    assert rows, "규격서 §6 표를 파싱하지 못했습니다"
    return rows


def test_every_required_step_field_has_a_source():
    # The optional fields the simulator records too: the converter carries them in every release.
    v = _validator()
    mapped = {f.schema for f in STEP_FIELDS}
    missing = sorted(_carried_unconditionally(v) - mapped)
    assert not missing, f"대응표에 없는 필수 스텝 필드: {missing}"


def test_contact_type_is_mapped_even_though_c8_does_not_list_it():
    # 검사 위치가 다를 뿐 필수이므로 대응표에도 있어야 한다.
    assert C3_ENFORCED <= {f.schema for f in STEP_FIELDS}


def test_every_required_episode_field_has_a_source():
    v = _validator()
    mapped = {f.schema for f in EPISODE_FIELDS}
    missing = sorted(set(v.REQUIRED_EPISODE_FIELDS) - mapped)
    assert not missing, f"대응표에 없는 필수 에피소드 필드: {missing}"


def test_every_conditionally_required_episode_field_has_a_source():
    """`randomization_seed` 와 `sim_backend` 는 변환기가 실으면서도 이 표에 없었다 — 무조건 필수만
    대조하던 위 시험이 그 빈칸을 못 봤다. 이 파이프라인이 변환하는 것은 전부 sim 이다."""
    v = _validator()
    mapped = {f.schema for f in EPISODE_FIELDS}
    owed = {f for fields in v.CONDITIONAL_REQUIRED_EPISODE_FIELDS.values() for f in fields}
    assert not sorted(owed - mapped), sorted(owed - mapped)


def test_the_map_invents_no_step_field_the_schema_does_not_have():
    # Every mapped field is one the schema names: required, optional, or required under a condition. The
    # pack members used to be the exception (spec §6 rows, not parquet columns); they are not
    # shipped and not mapped.
    v = _validator()
    conditional = {f for fields in v.CONDITIONAL_REQUIRED_STEP_FIELDS.values() for f in fields}
    known = set(v.REQUIRED_STEP_FIELDS) | set(v.OPTIONAL_STEP_FIELDS) | C3_ENFORCED | conditional
    extra = sorted({f.schema for f in STEP_FIELDS} - known)
    assert not extra, f"스키마에 없는 필드가 대응표에 있음: {extra}"


def test_every_conditionally_required_step_field_is_mapped_under_its_condition():
    v = _validator()
    by_name = {f.schema: f for f in STEP_FIELDS}
    for cond, fields in v.CONDITIONAL_REQUIRED_STEP_FIELDS.items():
        for name in fields:
            assert name in by_name, f"대응표에 없는 조건부 스텝 필드: {name}"
            assert by_name[name].condition == cond, (name, by_name[name].condition, cond)


def test_the_map_conditions_are_the_canon_keys():
    # The same string on both sides: the canon's key is what a consumer reads, the
    # map's is what the converter and C9 switch on.
    used = {f.condition for f in STEP_FIELDS if f.condition is not None}
    assert used == set(_canon_conditions())
    assert CAMERAS in used


def test_the_unconditional_fields_are_exactly_the_unconditionally_required():
    # Guards the v0.1 invariant below from the other side: marking a v0.1 field as
    # conditional would drop it from a v0.1 conversion, and every test derived from
    # "the unconditional fields" would shrink along with it.
    v = _validator()
    unconditional = {f.schema for f in STEP_FIELDS if f.condition is None}
    assert unconditional == _carried_unconditionally(v) | C3_ENFORCED


def test_the_conditional_columns_and_videos_are_the_canon_list():
    # Packs aside, what the map switches on under a condition is exactly what the
    # canon says that condition requires.
    for cond, fields in _canon_conditions().items():
        mapped = {f.schema for f in STEP_FIELDS
                  if f.condition == cond and f.kind in ("array", "video")}
        assert mapped == set(fields), cond


def test_every_mapped_step_field_has_a_row_in_spec_section_6():
    rows = _spec_step_rows()
    missing = sorted(f.schema for f in STEP_FIELDS if f.schema not in rows)
    assert not missing, f"규격서 §6 에 행이 없는 대응표 필드: {missing}"


def test_the_map_carries_no_pack_member():
    # The canon declares no pack. The pack code stays, but no step
    # field stands for a member of one. Its own assertion, not a set compared with the spec's: two
    # empty sets would agree about nothing.
    assert [f.schema for f in STEP_FIELDS if f.kind == "pack"] == []


def test_the_spec_keeps_the_depth_and_segmentation_rows_marked_not_shipped():
    rows = _spec_step_rows()
    members = {name: rest for name, rest in rows.items() if name.startswith(PACK_MEMBER_PREFIXES)}
    found = {p for p in PACK_MEMBER_PREFIXES for name in members if name.startswith(p)}
    assert found == set(PACK_MEMBER_PREFIXES), (
        f"규격서 §6 에서 깊이·분할 행을 찾지 못했습니다 — 찾은 것 {sorted(members)}")
    unmarked = sorted(name for name, rest in members.items() if NOT_SHIPPED_MARK not in rest)
    assert not unmarked, f"규격서 §6 의 깊이·분할 행에 「{NOT_SHIPPED_MARK}」 표시가 없습니다: {unmarked}"
    mapped = sorted({name for name, rest in rows.items() if NOT_SHIPPED_MARK in rest}
                    & {f.schema for f in STEP_FIELDS})
    assert not mapped, f"규격서가 싣지 않는다는 필드가 대응표에 있습니다: {mapped}"


@pytest.mark.parametrize("field", [f for f in STEP_FIELDS if f.kind in ("video", "pack")],
                         ids=lambda f: f.schema)
def test_video_and_pack_fields_are_conditional(field):
    # v0.1 has no media; an unconditional video or pack field would be demanded of it.
    assert field.condition is not None


def test_pressure_is_absent_now_that_it_is_conditional():
    assert "observation.contact.pressure" not in {f.schema for f in STEP_FIELDS}


def test_no_schema_field_is_mapped_twice():
    names = [f.schema for f in STEP_FIELDS]
    assert len(set(names)) == len(names), "중복 매핑"


@pytest.mark.parametrize("active", [(), (CAMERAS,)], ids=["none", "cameras"])
def test_npz_backed_pairs_are_unique_on_both_sides(active):
    pairs = npz_backed_step_fields(active)
    assert len({s for s, _ in pairs}) == len(pairs)
    assert len({n for _, n in pairs}) == len(pairs)


def test_without_conditions_the_npz_backed_fields_are_the_unconditional_arrays():
    # What npz_backed_step_fields() returned before camera fields existed, derived rather
    # than counted -- the converter and C9 of a v0.1 batch read exactly this.
    expected = tuple((f.schema, f.npz) for f in STEP_FIELDS
                     if f.kind == "array" and f.condition is None)
    assert npz_backed_step_fields() == expected


def test_the_camera_condition_adds_exactly_its_arrays_after_the_rest():
    base = npz_backed_step_fields()
    with_cameras = npz_backed_step_fields((CAMERAS,))
    added = tuple((f.schema, f.npz) for f in STEP_FIELDS
                  if f.kind == "array" and f.condition == CAMERAS)
    assert added and with_cameras == base + added


def test_active_step_fields_keeps_the_map_order():
    on = active_step_fields((CAMERAS,))
    assert on == tuple(f for f in STEP_FIELDS if f.condition in (None, CAMERAS))
    assert active_step_fields() == tuple(f for f in STEP_FIELDS if f.condition is None)


@pytest.mark.parametrize("bad, error", [
    (CAMERAS, TypeError),          # a bare string iterates into letters and matches nothing
    (("camera",), ValueError),     # an unknown name would switch nothing on, silently
])
def test_active_conditions_that_would_quietly_drop_fields_are_refused(bad, error):
    with pytest.raises(error):
        npz_backed_step_fields(bad)


@pytest.mark.parametrize("field", STEP_FIELDS, ids=lambda f: f.schema)
def test_array_fields_name_an_npz_array_and_others_do_not(field):
    if field.kind == "array":
        assert field.npz, f"{field.schema}: kind=array 인데 npz 이름이 없음"
    else:
        assert field.npz is None, f"{field.schema}: kind={field.kind} 인데 npz 이름이 있음"


@pytest.mark.parametrize("field", STEP_FIELDS, ids=lambda f: f.schema)
def test_step_field_kinds_are_from_the_declared_set(field):
    assert field.kind in {"array", "derived", "sidecar", "video", "pack"}


@pytest.mark.parametrize("field", EPISODE_FIELDS, ids=lambda f: f.schema)
def test_episode_field_sources_are_from_the_declared_set(field):
    assert field.source in {"sidecar", "scenarios", "joint_map", "assets",
                            "label", "const", "derived"}


def _canon_table_fields() -> dict:
    schema = json.loads((REPO_ROOT / "meta" / "safety_schema.json").read_text(encoding="utf-8"))
    return schema["scenario_conditional_episode_fields"]


def test_the_table_fields_are_the_canons_in_order():
    canon = _canon_table_fields()
    assert set(canon) == {TABLE}
    assert canon[TABLE] == [f.schema for f in EPISODE_FIELDS if f.condition == TABLE]
    assert {f.condition for f in EPISODE_FIELDS} == {None, TABLE}


def test_the_table_label_fields_are_the_evidence_columns():
    from generation.labeling import evidence
    label = [f for f in EPISODE_FIELDS if f.condition == TABLE and f.source == "label"]
    assert tuple(f.schema for f in label) == evidence.EVIDENCE_COLUMNS
    # The key is the member of the label the column carries: evidence.<name>.
    assert [f.key for f in label] == [f"{evidence.EVIDENCE_KEY}.{name}" for name in evidence.NAMES]
    others = {f.source for f in EPISODE_FIELDS if f.condition == TABLE and f.source != "label"}
    assert others == {"derived"}


def test_the_label_columns_read_twice_are_the_unconditional_label_fields():
    # The re-derivation reads these again from the numbers a loader hands out (float32); the evidence is not
    # among them -- a ratio worked out again from narrowed forces differs in its last digits.
    from generation.labeling import check_release_labels
    assert check_release_labels.LABEL_COLUMNS == ("outcome", "severity", "triggered_rules")


def test_episode_fields_adds_the_table_fields_only_when_asked():
    assert episode_fields(False) == tuple(f for f in EPISODE_FIELDS if f.condition is None)
    assert episode_fields(True) == tuple(f for f in EPISODE_FIELDS if f.condition in (None, TABLE))
    assert len(episode_fields(True)) == len(episode_fields(False)) + 21


def test_step_dt_rejects_a_two_dimensional_timestamp_array():
    # reshape(-1) 이 (2,2) 를 [0,.02,.04,.06] 으로 펴서 정상 50 Hz 로 읽었다.
    from generation.field_map import step_dt_from_timestamps
    with pytest.raises(ValueError, match="1차원"):
        step_dt_from_timestamps(np.array([[0.0, 0.02], [0.04, 0.06]]))


def test_the_hri_condition_is_a_canon_key():
    assert HRI in {f.condition for f in STEP_FIELDS if f.condition is not None}
    assert HRI in _canon_conditions()


@pytest.mark.parametrize("active", [(HRI,), (CAMERAS, HRI)], ids=["hri", "both"])
def test_npz_backed_pairs_stay_unique_with_hri(active):
    pairs = npz_backed_step_fields(active)
    assert len({s for s, _ in pairs}) == len(pairs)
    assert len({n for _, n in pairs}) == len(pairs)


def test_the_hri_condition_adds_exactly_its_arrays_after_the_rest():
    added = tuple((f.schema, f.npz) for f in STEP_FIELDS if f.kind == "array" and f.condition == HRI)
    assert added and npz_backed_step_fields((HRI,)) == npz_backed_step_fields() + added


def _hri_names():
    return [f.npz for f in STEP_FIELDS if f.condition == HRI]


def test_a_recording_with_every_hri_array_is_hri():
    assert hri_recorded(["timestamp", *_hri_names()]) is True


def test_the_walking_command_alone_is_not_hri():
    # S1a's and S1d's recordings carry command_velocity too.
    assert hri_recorded(["timestamp", "command_velocity"]) is False
    assert hri_recorded(["timestamp"]) is False


def test_part_of_the_hri_arrays_is_refused():
    with pytest.raises(ValueError, match="approach_speed_human"):
        hri_recorded([n for n in _hri_names() if n != "approach_speed_human"])


def test_the_hri_arrays_without_the_walking_command_are_refused():
    with pytest.raises(ValueError, match="command_velocity"):
        hri_recorded([n for n in _hri_names() if n != "command_velocity"])
