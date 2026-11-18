"""meta/safety_schema.json 이 검증기의 상수와 같은 것을 말하는지.

이 파일은 배포본에 동봉되는 「규격서의 기계가독 정의」(스키마 §4)이고, 검증기는
자기 상수로 같은 것을 검사한다. 둘이 어긋나면 배포본을 받은 사람이 읽는 규격과
우리가 실제로 검사하는 규격이 달라지는데, 그 상태로도 CI 는 통과한다.
CONTRIBUTING.md 「함께 고치는 곳」이 말하는 함정이 정확히 이것이므로 여기서 묶어 둔다.

검증기 쪽을 고치면 이 시험이 깨진다. 그때 정본을 함께 고치는 것이 의도된 동작이다.
"""

import json
from pathlib import Path

import pytest

from validator import validate_dataset as v

REPO_ROOT = Path(__file__).resolve().parents[2]
CANON = REPO_ROOT / "meta" / "safety_schema.json"
RULES = REPO_ROOT / "meta" / "rules.json"


@pytest.fixture(scope="module")
def canon():
    assert CANON.exists(), f"정본이 없습니다: {CANON.relative_to(REPO_ROOT)}"
    return json.loads(CANON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rules():
    return json.loads(RULES.read_text(encoding="utf-8"))


def test_schema_version_matches(canon):
    assert canon["schema_version"] == v.SCHEMA_VERSION


def test_body_region_codes_match(canon):
    assert set(canon["body_regions"]) == v.BODY_REGIONS


def test_every_body_region_is_named(canon):
    for code, entry in canon["body_regions"].items():
        assert entry.get("name_ko"), f"{code}: name_ko 없음"
        assert entry.get("name_en"), f"{code}: name_en 없음"


def test_contact_types_match(canon):
    assert set(canon["contact_types"]) == v.CONTACT_TYPES


def test_contact_types_agree_with_rules_canon(canon, rules):
    """meta/rules.json 도 contact_types 를 들고 있다 — 셋이 같은 것을 말해야 한다.

    rules.json 쪽은 한계가 갈리는 두 유형(transient·quasistatic)에만 뜻풀이를
    달아 두므로 `none` 이 없다. 그 차이만 빼고 일치해야 한다.
    """
    assert set(rules["contact_types"]) | {"none"} == set(canon["contact_types"])


def test_safety_events_match(canon):
    assert set(canon["safety_events"]) == v.SAFETY_EVENTS


def test_enums_match(canon):
    assert set(canon["enums"]) == set(v.ENUMS)
    for name, allowed in canon["enums"].items():
        assert set(allowed) == v.ENUMS[name], f"enum {name} 불일치"


def test_required_episode_fields_match(canon):
    # 순서까지 본다. 규격서 §5 의 표와 같은 순서로 읽히는 편이 낫고, 순서가
    # 흔들리면 어느 쪽이 옮겨 적히다 말았다는 뜻이다.
    assert canon["required_episode_fields"] == v.REQUIRED_EPISODE_FIELDS


def test_required_step_fields_match(canon):
    assert canon["required_step_fields"] == v.REQUIRED_STEP_FIELDS


def test_optional_step_fields_match(canon):
    # The order too, as for the required list: the two are read side by side (spec §6).
    assert canon["optional_step_fields"] == list(v.OPTIONAL_STEP_FIELDS)


def test_an_optional_step_field_is_neither_required_nor_conditional(canon):
    # One field in two lists gives two answers to "may a release leave it out".
    optional = set(canon["optional_step_fields"])
    assert not optional & set(canon["required_step_fields"])
    assert not optional & _conditional_step_fields(canon)


def test_fixed_shapes_match(canon):
    assert canon["fixed_shapes"] == v.FIXED_SHAPES


def test_safety_fields_are_the_safety_prefixed_step_fields(canon):
    # 파생값이지만 배포본 소비자가 가장 먼저 찾는 목록이라 따로 싣는다.
    # 손으로 적으면 §6 이 늘 때 여기만 남으므로 검증기에서 유도해 대조한다.
    # 필수와 선택의 safety. 열 — 집합으로 본다(두 목록에 갈라져 있어 한 순서가 없다).
    derived = [f for f in v.REQUIRED_STEP_FIELDS + list(v.OPTIONAL_STEP_FIELDS) if f.startswith("safety.")]
    assert len(canon["safety_fields"]) == len(set(canon["safety_fields"]))
    assert set(canon["safety_fields"]) == set(derived)


def test_thresholds_are_not_duplicated_here(canon):
    """임계값의 정본은 meta/rules.json 하나다 (CONTRIBUTING.md 「작업 스타일」).

    이 파일은 '무엇이 허용되는가' 만 담고 '그 값이 얼마인가' 는 담지 않는다.
    """
    assert "contact_limits" not in canon
    assert "rules" not in canon
    assert "thresholds" not in canon


def test_json_encoded_fields_are_declared(canon):
    """규격서가 리스트·객체로 정의하는데 parquet 에는 문자열로 실리는 필드들.

    LeRobot v3.0 의 string dtype 이 스칼라 전용이라 생기는 제약이고, 배포본을
    받은 쪽은 json.loads 로 풀어야 한다. 규격서만 읽으면 리스트를 기대하게
    되므로 기계가독 정본이 말해 주어야 한다 (2026-08-24, 8단계 실측).
    """
    block = canon["json_encoded_fields"]
    assert block["step"] and block["episode"]
    # 스텝 필드는 전부 필수이므로 여기서 오타를 잡을 수 있다. 에피소드 쪽은
    # sim_backend 가 조건부(시뮬레이션 필수)라 무조건 필수 목록만으로는 대조할 수
    # 없고, 조건부까지 합쳐서 본다. notes 는 무조건 필수다(규격서 §5). episode_spec 은
    # 조건표가 있는 시나리오의 열이다.
    assert set(block["step"]) <= set(canon["required_step_fields"])
    episode_known = set(canon["required_episode_fields"]) | {
        f for v in canon["conditional_required_episode_fields"].values() for f in v} | _table_fields(canon)
    assert set(block["episode"]) <= episode_known, (
        "json_encoded_fields.episode 가 필수·조건부 어디에도 없는 필드를 가리킵니다")
    assert "episode_spec" in block["episode"]


def _table_fields(canon) -> set[str]:
    return {f for fields in canon["scenario_conditional_episode_fields"].values() for f in fields}


def test_the_table_scenarios_episode_fields_match(canon):
    # The order too, as for the required lists: the spec's §5 rows and the field map read in it.
    declared = canon["scenario_conditional_episode_fields"]
    assert declared == {key: list(fields) for key, fields in v.SCENARIO_CONDITIONAL_EPISODE_FIELDS.items()}
    assert len(declared["conditions"]) == 21 == len(set(declared["conditions"]))


def test_a_table_scenarios_fields_are_neither_required_nor_source_conditional(canon):
    # One field in two lists gives two answers to "when is it required".
    table = _table_fields(canon)
    assert not table & set(canon["required_episode_fields"])
    assert not table & {f for fields in canon["conditional_required_episode_fields"].values() for f in fields}


def test_the_table_condition_names_a_key_of_the_scenarios_canon():
    # The key is read off a scenario entry of the release's meta/scenarios.json: a name no entry carries
    # would switch the columns on for no scenario.
    scenarios = json.loads((REPO_ROOT / "meta" / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    for key in v.SCENARIO_CONDITIONAL_EPISODE_FIELDS:
        assert [s["scenario_id"] for s in scenarios if key in s] == ["S1e", "S1f", "S1g"], key


def test_the_episode_spec_members_match(canon):
    assert canon["episode_spec_members"] == list(v.EPISODE_SPEC_MEMBERS)
    assert canon["episode_json_shapes"]["episode_spec"] == "object"


def test_the_condition_design_enum_is_random_or_grid(canon):
    assert set(canon["enums"]["condition_design"]) == {"random", "grid"} == v.ENUMS["condition_design"]


def _c1_errors(tmp_path, schema) -> list[str]:
    """C1's errors about a release whose meta/safety_schema.json is ``schema`` (nothing else is needed)."""
    (tmp_path / "meta").mkdir(parents=True, exist_ok=True)
    (tmp_path / "meta" / "safety_schema.json").write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
    rep = v.Report(root=str(tmp_path))
    v._c1_safety_schema_sound(v.Dataset(tmp_path, rep), rep)
    return [f.message for f in rep.findings if f.check == "C1" and f.severity == v.ERROR]


def test_c1_takes_the_canon_as_it_stands(canon, tmp_path):
    assert _c1_errors(tmp_path, canon) == []


@pytest.mark.parametrize("edit, said", [
    (lambda s: s["scenario_conditional_episode_fields"]["conditions"].remove("evidence.first_contact_s"),
     "규격 정의의 scenario_conditional_episode_fields 가 검증기와 다름"),
    (lambda s: s["scenario_conditional_episode_fields"]["conditions"].reverse(),
     "규격 정의의 scenario_conditional_episode_fields 가 검증기와 다름"),
    (lambda s: s["scenario_conditional_episode_fields"].update(grid=["condition_design"]),
     "규격 정의의 scenario_conditional_episode_fields 가 검증기와 다름"),
    (lambda s: s.pop("scenario_conditional_episode_fields"), "규격 정의에 scenario_conditional_episode_fields 가 없음"),
    (lambda s: s.update(scenario_conditional_episode_fields=["episode_spec"]),
     "규격 정의에 scenario_conditional_episode_fields 가 없음"),
    (lambda s: s["episode_spec_members"].remove("repro"), "규격 정의의 episode_spec_members 가 검증기와 다름"),
    (lambda s: s.pop("episode_spec_members"), "규격 정의에 episode_spec_members 가 없음"),
    (lambda s: s["json_encoded_fields"]["episode"].append("not_a_field"),
     "규격 정의의 json_encoded_fields.episode 가 실재하지 않는 필드를 가리킴"),
    (lambda s: s["enums"]["condition_design"].remove("grid"), "규격 정의의 enum condition_design 이 검증기와 다름"),
    (lambda s: s["episode_json_shapes"].pop("episode_spec"), "규격 정의의 episode_json_shapes 가 검증기와 다름"),
], ids=["one-short", "reordered", "another-key", "absent", "a-list", "a-member-short", "members-absent",
        "a-json-field-unknown", "enum-short", "shape-absent"])
def test_c1_holds_the_shipped_table_columns_to_the_validators(canon, tmp_path, edit, said):
    schema = json.loads(json.dumps(canon))
    edit(schema)
    errors = _c1_errors(tmp_path, schema)
    assert any(m.startswith(said) for m in errors), errors


def test_json_encoded_step_fields_are_not_claimed_as_fixed_shape(canon):
    # fixed_shapes 는 고정 길이 벡터의 계약이다. 문자열 한 칸인 필드가 거기
    # 걸려 있으면 두 계약이 같은 필드를 두고 다른 말을 하게 된다.
    assert not (set(canon["json_encoded_fields"]["step"]) & set(canon["fixed_shapes"]))


def test_body_region_names_come_from_the_rules_canon(rules, canon):
    """부위 표시명의 정본은 meta/rules.json 이다 (부위별 한계값이 사는 곳).

    여기에 이름을 새로 지어 넣었더니 영문명 7개가
    rules.json 과 달랐다(`skull / forehead` vs `skull and forehead`). 표준 용어에
    가까운 쪽은 rules.json 이고, 두 벌을 두면 한쪽만 고쳐도 CI 가 통과한다.
    """
    limits = rules["contact_limits"]
    for code, entry in canon["body_regions"].items():
        assert entry["name_ko"] == limits[code]["name_ko"], code
        assert entry["name_en"] == limits[code]["name_en"], code


def test_conditional_required_fields_match(canon):
    declared = canon["conditional_required_episode_fields"]
    expected = {f"source={src}": sorted(fields)
                for src, fields in v.CONDITIONAL_REQUIRED_EPISODE_FIELDS.items()}
    assert declared == expected


def test_forbidden_measurement_methods_match(canon):
    declared = canon["forbidden_measurement_methods"]
    expected = {f"source={src}": sorted(mm)
                for src, mm in v.FORBIDDEN_MEASUREMENT_METHODS.items()}
    assert declared == expected


def test_conditional_fields_are_not_also_unconditionally_required(canon):
    # 두 목록이 같은 필드를 담으면 「언제 필수인가」에 두 답이 생긴다.
    conditional = {f for fields in canon["conditional_required_episode_fields"].values()
                   for f in fields}
    assert not (conditional & set(canon["required_episode_fields"]))


def _conditional_step_fields(canon) -> set[str]:
    return {f for fields in canon["conditional_required_step_fields"].values() for f in fields}


def test_conditional_required_step_fields_match(canon):
    # The step-side twin of test_conditional_required_fields_match. The key is a bare
    # condition name ("cameras": the distribution ships meta/cameras.json), not
    # "source=<value>", so it is compared as written; the order is compared too, as for
    # required_step_fields.
    declared = canon["conditional_required_step_fields"]
    expected = {cond: list(fields)
                for cond, fields in v.CONDITIONAL_REQUIRED_STEP_FIELDS.items()}
    assert declared == expected


def test_conditional_step_fields_are_not_also_unconditionally_required(canon):
    # Same reason as the episode side: two lists holding one field give two answers to
    # "when is it required" -- and a v0.1 distribution would be told it lacks a column.
    assert not (_conditional_step_fields(canon) & set(canon["required_step_fields"]))


def test_fixed_shapes_name_only_declared_step_fields(canon):
    # C8 checks a length only where the column exists, so a misspelt key here would sit
    # in the shipped schema promising a length for a column no distribution carries.
    declared = (set(canon["required_step_fields"]) | _conditional_step_fields(canon)
                | set(canon["optional_step_fields"]))
    assert set(canon["fixed_shapes"]) <= declared, sorted(set(canon["fixed_shapes"]) - declared)


def test_the_camera_step_fields_follow_the_camera_canon(canon):
    # Derived from meta/cameras.json, not written down a second time: a camera added
    # there has to bring its pose (and, if it records rgb, its image) field here, and the
    # box field is the one that file names.
    cams = json.loads((REPO_ROOT / "meta" / "cameras.json").read_text(encoding="utf-8"))
    expected = {cams["bbox"]["field"]}
    for name, cam in cams["cameras"].items():
        expected.add(f"observation.camera_pose.{name}")
        if "rgb" in cam["channels"]:
            expected.add(f"observation.images.{name}")
    assert set(canon["conditional_required_step_fields"]["cameras"]) == expected


def test_the_camera_field_lengths_follow_the_canon_they_describe(canon):
    # The box has as many members as its "not visible" marker; a camera pose is the
    # 7-element form of observation.root_pose (meta/cameras.json pose.axes_note).
    cams = json.loads((REPO_ROOT / "meta" / "cameras.json").read_text(encoding="utf-8"))
    shapes = canon["fixed_shapes"]
    assert shapes[cams["bbox"]["field"]] == len(cams["bbox"]["invalid"])
    for name in cams["cameras"]:
        assert shapes[f"observation.camera_pose.{name}"] == shapes["observation.root_pose"]


def test_the_spec_document_reference_is_honest(canon):
    """배포본 안에서 따라갈 수 없는 경로를 가리키지 않아야 한다.

    `spec_document` 가 `docs/schema/…md` 를 가리키는데 그 문서는 배포본에 동봉되지
    않는다. 내려받은 쪽이 따라가면 없는 파일이다.
    """
    ref = canon["spec_document"]
    assert isinstance(ref, dict), "문자열 경로만 두면 배포본 안에서 깨진다"
    assert ref["bundled_in_dataset"] is False
    assert (REPO_ROOT / ref["path_in_repository"]).exists()


def test_the_canon_is_schema_0_2(canon):
    assert canon["schema_version"] == "0.2"
    assert canon["spec_document"]["path_in_repository"] == "docs/schema/dataset-schema-v0.2.md"


def test_every_schema_version_copy_agrees():
    """schema_version 의 사본은 코드 상수와 정본 다섯이다. 한 곳만 올리면 조용히 어긋난다.

    검증기의 버전 불일치는 WARN 이라 CI 를 빨갛게 만들지 못한다.
    """
    from generation import convert_lerobot
    from validator import make_fixture

    copies = {
        "validator/validate_dataset.py": v.SCHEMA_VERSION,
        "generation/convert_lerobot.py": convert_lerobot.SCHEMA_VERSION,
        "validator/make_fixture.py": make_fixture.SCHEMA_VERSION,
    }
    for name in ("safety_schema", "rules", "scenarios", "assets", "cameras"):
        path = REPO_ROOT / "meta" / f"{name}.json"
        copies[f"meta/{name}.json"] = json.loads(
            path.read_text(encoding="utf-8"))["schema_version"]

    assert len(set(copies.values())) == 1, f"불일치: {copies}"


def _source_of(key: str) -> str:
    assert key.startswith("source="), f"키 형식이 아닙니다: {key}"
    return key.split("=", 1)[1]


def test_conditional_keys_name_a_real_source_value(canon):
    """`source=<값>` 의 <값> 이 enum 안에 있어야 한다.

    검증기 상수와 정본에서 `sim` 을 동시에
    `sm` 으로 바꾸면 두 목록은 여전히 일치하므로 모든 대조 시험이 통과하는데,
    C5 는 `.get(source, ())` 로 찾으므로 실제 `source="sim"` 행에는 아무 검사도
    걸리지 않는다. 이름을 맞춘 것과 이름이 실재하는 것은 다른 문제다.
    """
    allowed = set(canon["enums"]["source"])
    for key in canon["conditional_required_episode_fields"]:
        assert _source_of(key) in allowed, key
    for key in canon["forbidden_measurement_methods"]:
        assert _source_of(key) in allowed, key


def test_validator_keys_name_a_real_source_value():
    # 정본만 고쳐도 못 잡으므로 검증기 상수 쪽도 같은 것을 본다.
    allowed = v.ENUMS["source"]
    assert set(v.CONDITIONAL_REQUIRED_EPISODE_FIELDS) <= allowed
    assert set(v.FORBIDDEN_MEASUREMENT_METHODS) <= allowed


def test_forbidden_methods_are_real_measurement_methods(canon):
    # enum 에 없는 값을 금지 목록에 적으면 아무것도 금지하지 못한다.
    allowed = set(canon["enums"]["measurement_method"])
    for key, methods in canon["forbidden_measurement_methods"].items():
        assert set(methods) <= allowed, f"{key}: {sorted(set(methods) - allowed)}"


def test_no_source_forbids_every_measurement_method(canon):
    # 한 source 가 전부를 금지하면 그 source 로는 어떤 에피소드도 유효하지 않다.
    allowed = set(canon["enums"]["measurement_method"])
    for key, methods in canon["forbidden_measurement_methods"].items():
        assert set(methods) != allowed, key


def test_every_source_value_has_an_entry_in_the_forbidden_map(canon):
    """enum 에만 값을 더해도 조용히 fail-open 이 되는 것을 막는다.

    정본과 검증기 enum **양쪽에** `"sm"` 을
    더하기만 하면 기존 대조 시험이 전부 통과하는데, C5 의 `.get(source, ())` 는
    그 source 에 아무 조건도 걸지 않는다. 키가 enum 의 부분집합인지만 보면
    이 방향은 못 잡으므로, 반대 방향(모든 enum 값이 선언되었는지)도 본다.
    """
    declared = {_source_of(k) for k in canon["forbidden_measurement_methods"]}
    assert set(canon["enums"]["source"]) <= declared


def test_the_validator_declares_every_source_it_accepts():
    assert v.ENUMS["source"] <= set(v.FORBIDDEN_MEASUREMENT_METHODS)


@pytest.mark.parametrize("key, validator_table", [
    ("episode_json_shapes", v.EPISODE_JSON_SHAPES),
    ("notes_members", v.NOTES_MEMBERS),
    ("notes_member_shapes", v.NOTES_MEMBER_SHAPES),
    ("notes_bounds", v.NOTES_BOUNDS),
])
def test_the_json_member_tables_match(canon, key, validator_table):
    # The four tables C5 reads the episode's JSON columns by. C1 compares them whole at run time;
    # this holds the canon file and the constants to the same reading before a release is checked.
    assert canon[key] == validator_table


@pytest.mark.parametrize("member, declared", [
    ("degenerate_reason", "string|null"),
    ("box_released_step", "integer|null"),
    ("camera_in_box_steps", "object|null"),
    ("task_stages", "array|null"),
    ("human_stature_m", "number|null"),
    ("task_time_limit_s", "number|null"),
])
def test_the_members_only_a_scripted_scenario_fills_are_declared_with_null(canon, member, declared):
    # C5 asks every episode for every member of notes_members. These six come from the hri
    # recorder's sidecar; a scenario without a script ships them as null, so each one's
    # declaration has to admit null -- and says the type the recorder writes otherwise.
    assert canon["notes_members"][member] == declared
    assert "null" in canon["notes_members"][member].split("|")


def test_a_task_stage_is_declared_with_the_shape_the_recorder_writes(canon):
    assert canon["notes_member_shapes"]["task_stages[]"] == {
        "name": "string", "completed": "boolean", "end_step": "integer>=-1", "duration_s": "number|null"}


def test_step_field_layers_match(canon):
    assert canon["step_field_layers"] == v.STEP_FIELD_LAYERS


def test_every_step_field_has_one_layer(canon):
    # Required, conditional and optional: every column a distribution may carry is read as raw or
    # processed, and nothing else is in the table.
    fields = (set(canon["required_step_fields"]) | _conditional_step_fields(canon)
              | set(canon["optional_step_fields"]))
    assert set(canon["step_field_layers"]) == fields
    assert set(canon["step_field_layers"].values()) <= set(v.LAYERS)


def test_the_hri_visible_fractions_follow_the_robot_cameras(canon):
    # Derived from meta/cameras.json: a camera mounted on the robot brings its fraction field here.
    cams = json.loads((REPO_ROOT / "meta" / "cameras.json").read_text(encoding="utf-8"))["cameras"]
    robot = {name for name, cam in cams.items() if cam.get("type") == "mounted"}
    hri = canon["conditional_required_step_fields"]["hri"]
    assert {f.rsplit(".", 1)[1] for f in hri if f.startswith(v.VISIBLE_FRACTION_PREFIX)} == robot
    assert {"observation.command_velocity", "observation.human.root_pose", "observation.human.root_vel",
            "safety.approach_speed_robot", "safety.approach_speed_human"} <= set(hri)
