"""규격서와 시나리오 카탈로그가 정본에 대해 하는 진술을 정본에서 유도해 대조한다.

**왜 여기에 시험이 없었나.** 배포본까지 이어지는 사슬이 문서마다 다르다.

    docs/dataset-card.md          publish_dataset.preflight 가 배포본과 대조 (표식)
    docs/assets/…inventory…md     배포본에 동봉되어 함께 나간다
    docs/schema/…md               ← 아무것도 없었다
    docs/scenarios/…md            ← 아무것도 없었다

카드도 사슬이 닿지만 **표식은 배치 통계 몇 개만** 가드한다
(표식은 숫자를 지키지 산문을 지키지 않는다).

그런데 배포본의 `meta/safety_schema.json` 이 `spec_document` 로 규격서를 가리킨다.
받은 쪽이 규격을 찾아가면 도착하는 문서이고, 공개 저장소에서는 이용자가
읽는 규격이다. 그 문서가 정본과 반대를 말한 자리가 셋
있었다 — 압력 한계가 "모두 비어 있다"(12/12 실려 있음), 생체역학 파라미터의 출처가
표 A.2(표 A.3이며 같은 문서 §9.2 가 그렇게 적고 있음), 배포본에 값이 "실리지
않는다"(§9.3 이 정반대를 적고 있음).

**이 시험이 보지 못하는 것 (먼저 적는다).**

  1. **산문.** 위 셋은 전부 산문이었고 이 시험은 그중 하나(채움 상태)만 잡는다.
     열거·계수 가능한 진술만 대조하며, 자유 문장이 정본과 어긋나는 것은 못 본다.
  2. **다른 낱말로 같은 말.** 아래 채움 상태·출처 표 검사는 문장을 정규식으로
     찾는다. 문장을 고쳐 쓰면 **찾지 못하고, 찾지 못하면 실패한다** — 통과가
     아니다. 확인할 수 없는 것을 통과시키지 않기 위해서다.
  3. **코드 거동에 대한 진술.** "빌더가 이 값을 쓴다" 류는 여기서 못 본다.
  4. **표시명 이외의 부위 속성**, **규칙 조건문의 내용**, **환경이 없는 시나리오의
     랜덤화 범위** 등 정본이 담지 않는 서술. (환경이 있는 S1a · S1d 의 랜덤화는
     정본에 있고 카탈로그 §4.5 를 아래에서 대조한다. 그 정본이
     환경과 맞는지는 여기서 못 본다 — generation/randomization.py.)

**단언을 손으로 적지 않는다.** 아래 어느
숫자도 이 파일에 상수로 없다. 방침이 뒤집히면 문서를 고치라고 말해야지, 시험이
배포본과 반대를 주장해서는 안 된다.
"""

import json
import re

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs/schema/dataset-schema-v0.2.md"
CATALOG = REPO / "docs/scenarios/scenario-catalog-v0.1.md"
CARD = REPO / "docs/dataset-card.md"
INVENTORY = REPO / "docs/assets/asset-license-inventory-v0.1.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def canon(name: str) -> dict:
    return json.loads((REPO / "meta" / name).read_text(encoding="utf-8"))


def only(pattern: str, text: str, where: str) -> re.Match:
    """정확히 한 번 맞아야 한다. 0 이면 문장이 바뀐 것이고 그때도 실패한다."""
    found = list(re.finditer(pattern, text))
    assert len(found) == 1, (
        f"{where}: 대조할 문장을 {len(found)}건 찾았습니다 (1건이어야 합니다).\n"
        f"  패턴: {pattern}\n"
        "  문장을 고쳐 썼다면 이 시험의 패턴도 함께 고치십시오 — 찾지 못한 것을 "
        "통과시키면 그 자리는 다시 아무도 안 보는 자리가 됩니다.")
    return found[0]


def paragraphs(text: str) -> list[tuple[int, str]]:
    """빈 줄로 끊어 문단 단위로 돌려준다 — (시작 줄 번호, 한 줄로 이은 본문).

    줄 단위로 훑으면 **접힌 문장을 못 본다.** 되돌리기 실험에서 실제로 났다:
    §8 의 「유효질량·유효탄성계수의」와 「출처는 표 A.2」가 서로 다른 줄에 있어,
    표 귀속을 표 A.2 로 되돌려도 검사가 초록이었다. 마크다운은 문장을 접으므로
    「이 줄에 낱말과 표 번호가 함께 있는가」는 우연에 기댄 질문이다.
    """
    out, start, buf = [], 1, []
    for line_no, line in enumerate(text.splitlines() + [""], 1):
        if line.strip():
            if not buf:
                start = line_no
            buf.append(line.strip())
        elif buf:
            out.append((start, " ".join(buf)))
            buf = []
    return out


# KETI 개조식의 항목 표지. 한 항목이 하나의 진술이다.
_BULLET = re.compile(r"^\s*(?:ㅇ|[-–]|:)\s")


def statements(text: str) -> list[tuple[int, str]]:
    """문단을 다시 **항목 단위**로 쪼갠다 — 접힌 줄은 항목에 붙인 채로.

    개조식은 `ㅇ` 항목 바로 아래에 `-` 하위 항목이 빈 줄 없이 붙으므로, 문단
    단위로 보면 **머리 항목의 주장과 하위 항목의 부정이 한 덩어리**가 된다.
    문단 단위로 보면 머리 항목을 틀린 문장으로 바꿔도 바로 아래 하위 항목의
    「쓰이지 않음」 이 같은 문단에 있어 초록이 된다. 훑는 사람이 읽는 것은 머리
    항목이므로 항목마다 따로 본다.
    """
    out = []
    for start, para in [(s, p) for s, p in _raw_lines(text)]:
        out.append((start, para))
    return out


def _raw_lines(text: str) -> list[tuple[int, str]]:
    out, start, buf = [], 1, []

    def flush():
        if buf:
            out.append((start, " ".join(buf)))

    for line_no, line in enumerate(text.splitlines() + [""], 1):
        stripped = line.strip()
        if not stripped:
            flush()
            buf.clear()
            continue
        if _BULLET.match(line) or not buf:
            flush()
            buf.clear()
            start = line_no
        buf.append(stripped)
    flush()
    return out


def bundled_paths() -> set[str]:
    """변환기가 배포본에 담는 정본 파일. 배포본이 아니라 **담기로 정한 곳**에서 읽는다."""
    from generation.convert_lerobot import (CANON_FILES, CANON_RENAMES,
                                            NARROWED_CANON)
    return set(CANON_FILES) | set(NARROWED_CANON) | set(CANON_RENAMES.values())


# ─────────────────────────────────────────────────────────────────────
# 열거를 복제한 자리
# ─────────────────────────────────────────────────────────────────────

def test_body_region_codes_and_names_match_canon():
    """§8 의 부위 코드 블록 ↔ `rules.json` `contact_limits`.

    표시명의 정본은 `contact_limits` 다 (CONTRIBUTING.md 「함께 고치는 곳」).
    `safety_schema.json` 이 영문명을 새로 지었을 때 12개 중 7개가 달랐다.
    """
    limits = canon("rules.json")["contact_limits"]
    block = only(r"```\n(BR-01[\s\S]*?)\n```", read(SPEC), "규격서 §8 부위 코드 블록")
    pairs = dict(re.findall(r"(BR-\d\d)\s+(\S+)", block.group(1)))

    assert set(pairs) == set(limits), (
        f"부위 코드 집합이 다릅니다 — 문서만: {sorted(set(pairs) - set(limits))}, "
        f"정본만: {sorted(set(limits) - set(pairs))}")
    wrong = {c: (n, limits[c]["name_ko"])
             for c, n in pairs.items() if n != limits[c]["name_ko"]}
    assert not wrong, f"표시명이 정본과 다릅니다 (문서, 정본): {wrong}"


def test_rule_list_matches_canon():
    """§9.1 의 총수와 규칙 ID ↔ `rules.json` `rules`.

    A2 계열 12개는 문서가 `-01` … `-12` 로 줄여 쓰므로 총수로만 대조하고,
    나머지는 ID 로 대조한다.
    """
    rules = canon("rules.json")["rules"]
    ids = {r["rule_id"] for r in rules}
    spec = read(SPEC)

    declared = int(only(r"### 9\.1 규칙 목록 \(정본 총 (\d+)개\)", spec,
                        "규격서 §9.1 제목").group(1))
    assert declared == len(ids), f"§9.1 제목은 {declared}개, 정본은 {len(ids)}개"

    wildcarded = {i for i in ids if i.startswith("R-TS15066-A2-")}
    mentioned = set(re.findall(r"`(R-[A-Z0-9]+-[A-Z]+-\d\d)`", spec))
    missing = (ids - wildcarded) - mentioned
    assert not missing, f"§9 에 없는 정본 규칙: {sorted(missing)}"

    invented = {m for m in mentioned if m.startswith("R-")} - ids
    assert not invented, f"정본에 없는 규칙을 문서가 부릅니다: {sorted(invented)}"


def _outcome_share_section() -> str:
    spec = read(SPEC)
    start = only(r"### 시나리오별 구성 비율 \(목표\)\n", spec, "규격서 §9 시나리오별 구성 비율").end()
    return spec[start:spec.index("\n## 10.", start)]


def test_the_outcome_targets_the_spec_names_are_the_canons():
    """§9 「시나리오별 구성 비율」 ↔ `scenarios.json` `outcome_target`: the default shares of its table and the
    scenarios its sentence names with other shares. Every scenario the sentence names has the shares it says, and
    every scenario it does not name has the table's -- so a target changed in the canon, a scenario added with a
    target of its own, or a share edited on the page is a red test."""
    section = _outcome_share_section()
    default = {name: int(pct) for name, pct in
               re.findall(r"^\| `(fail|near_miss|nominal)` \| (\d+)% \|", section, re.MULTILINE)}
    assert sorted(default) == ["fail", "near_miss", "nominal"], default
    [sentence] = [p for _, p in paragraphs(section) if "다른 비율을 두는 시나리오는" in p]
    tail = sentence[sentence.index("다른 비율을 두는 시나리오는"):]
    named = {sid: {"fail": int(f), "near_miss": int(n), "nominal": int(o)} for sid, f, n, o in re.findall(
        r"`(S\w+)` 는 `fail` (\d+)% · `near_miss` (\d+)% · `nominal` (\d+)%", tail)}
    assert named, f"§9 의 시나리오별 목표 문장을 읽지 못했습니다: {tail[:120]}"

    def shares(entry: dict) -> dict:
        return {k: round(float(v) * 100) for k, v in entry["outcome_target"].items()}
    by_id = {s["scenario_id"]: s for s in canon("scenarios.json")["scenarios"]}
    assert set(named) <= set(by_id), sorted(set(named) - set(by_id))
    assert {sid: shares(s) for sid, s in by_id.items() if sid in named} == named
    others = {sid: shares(s) for sid, s in by_id.items() if sid not in named and shares(s) != default}
    assert not others, f"§9 가 기본 비율이라 말하는데 정본의 목표가 다른 시나리오: {others}"


def test_the_spec_states_the_floors_c4_warns_below():
    """§9 「시나리오별 구성 비율」's floors ↔ `validate_dataset.OUTCOME_FLOOR` (the validator holds no target)."""
    from validator import validate_dataset

    m = only(r"`fail` \*\*(\d+)%\*\*\s*·\s*`near_miss` \*\*(\d+)%\*\*", _outcome_share_section(), "규격서 §9 C4 하한")
    assert {"fail": int(m.group(1)), "near_miss": int(m.group(2))} == {
        k: round(v * 100) for k, v in validate_dataset.OUTCOME_FLOOR.items()}


# How §5 marks a column required of the episodes of a scenario that declares a conditions table.
TABLE_MARK = "조건표 필수"


def test_episode_field_table_matches_canon():
    """§5 의 필수 표시(✔) · 조건표 표시와 enum 허용값 ↔ `safety_schema.json`."""
    schema = canon("safety_schema.json")
    section = read(SPEC).split("## 5. 에피소드 단위 메타데이터")[1].split("\n## 6.")[0]
    rows = re.findall(r"^\| `([^`]+)` \|([^|]*)\|([^|]*)\|([^|]*)\|", section, re.M)
    assert rows, "§5 표를 파싱하지 못했습니다 — 표 형식이 바뀌었습니까?"

    marked = {name for name, _, _, req in rows if "✔" in req}
    required = set(schema["required_episode_fields"])
    conditional = {f for fs in schema["conditional_required_episode_fields"].values()
                   for f in fs}
    assert marked - conditional == required - conditional, (
        f"필수 표시가 정본과 다릅니다 — 문서만 ✔: {sorted(marked - required)}, "
        f"정본만 필수: {sorted(required - marked)}")

    # The columns a scenario with a conditions table carries: every row, marked so, in the canon's order.
    table = [f for fs in schema["scenario_conditional_episode_fields"].values() for f in fs]
    in_table = [name for name, _, _, req in rows if req.strip() == TABLE_MARK]
    assert in_table == table, (
        f"「{TABLE_MARK}」 표시가 정본과 다릅니다 — 문서만: {sorted(set(in_table) - set(table))}, "
        f"정본만: {sorted(set(table) - set(in_table))}")

    allowed = {name: {v.strip(" `") for v in values.split("/")}
               for name, kind, values, _ in rows if kind.strip() == "enum"}
    unseen = set(schema["enums"]) - set(allowed)
    assert not unseen, (
        f"정본 enum 중 §5 표에서 찾지 못한 것: {sorted(unseen)} — "
        "행이 빠졌거나 타입 칸이 `enum` 이 아닙니다. 못 찾은 것을 통과시키지 않습니다")
    wrong = {k: (allowed[k], set(v)) for k, v in schema["enums"].items()
             if allowed[k] != set(v)}
    assert not wrong, f"enum 허용값이 정본과 다릅니다 (문서, 정본): {wrong}"


def test_step_field_table_matches_canon():
    """§6's required marks (✔), optional marks (선택), conditional marks and fixed shapes <-> `safety_schema.json`.

    §5 is compared above, but §6 itself was not compared with anything. The camera
    fields (conditional, with fixed lengths) put it in the same position as §5, so it
    gets the same test.
    """
    schema = canon("safety_schema.json")
    section = read(SPEC).split("## 6. 스텝 단위 시계열 (Parquet)")[1].split("\n## 7.")[0]
    rows = re.findall(r"^\| `([^`]+)` \|([^|]*)\|[^|]*\|([^|]*)\|", section, re.M)
    parsed = {name: (shape.strip(), req.strip()) for name, shape, req in rows}
    assert parsed, "§6 표를 파싱하지 못했습니다 — 표 형식이 바뀌었습니까?"

    marked = {name for name, (_, req) in parsed.items() if req == "✔"}
    required = set(schema["required_step_fields"])
    assert marked == required, (
        f"필수 표시가 정본과 다릅니다 — 문서만 ✔: {sorted(marked - required)}, "
        f"정본만 필수: {sorted(required - marked)}")

    marked_optional = {name for name, (_, req) in parsed.items() if req == "선택"}
    optional = set(schema["optional_step_fields"])
    assert marked_optional == optional, (
        f"선택 표시가 정본과 다릅니다 — 문서만 선택: {sorted(marked_optional - optional)}, "
        f"정본만 선택: {sorted(optional - marked_optional)}")

    for cond, fields in schema["conditional_required_step_fields"].items():
        not_conditional = {f: parsed.get(f, ("", "(행 없음)"))[1] for f in fields
                           if parsed.get(f, ("", ""))[1] != "조건부"}
        assert not not_conditional, (
            f"정본이 `{cond}` 조건부라는 필드가 §6 에서 조건부로 표시되지 않았습니다: "
            f"{not_conditional}")

    shapes = schema["fixed_shapes"]
    assert shapes, "정본에 fixed_shapes 가 없습니다 — 대조할 것이 사라졌습니다"
    wrong = {}
    for field, size in shapes.items():
        drawn = parsed.get(field, ("(없음)", ""))[0]
        if re.findall(r"\d+", drawn) != [str(size)]:
            wrong[field] = (drawn, size)
    assert not wrong, f"고정 형상이 정본과 다릅니다 (문서, 정본): {wrong}"


def test_the_layer_table_matches_canon():
    """§6 「층 — 로우와 가공」 ↔ `safety_schema.json` `step_field_layers` — every field, both ways."""
    layers = canon("safety_schema.json")["step_field_layers"]
    section = read(SPEC).split("### 층 — 로우와 가공")[1].split("\n### ")[0]
    rows = dict(re.findall(r"^\| (로우|가공) \|[^|]*\|(.*)\|$", section, re.M))
    assert set(rows) == {"로우", "가공"}, "§6 「층」 표를 파싱하지 못했습니다 — 표 형식이 바뀌었습니까?"
    drawn = {name: layer for word, layer in (("로우", "raw"), ("가공", "processed"))
             for name in re.findall(r"`([^`]+)`", rows[word])}
    assert drawn == layers, (
        f"층 표가 정본과 다릅니다 — 문서만: {sorted(set(drawn) - set(layers))}, "
        f"정본만: {sorted(set(layers) - set(drawn))}, "
        f"층이 다름: {sorted(k for k in set(drawn) & set(layers) if drawn[k] != layers[k])}")


def test_scenario_summary_table_matches_canon():
    """카탈로그 §3 총괄표 ↔ `scenarios.json` (hazard · 에피소드 수 · 우선순위)."""
    scenarios = {s["scenario_id"]: s for s in canon("scenarios.json")["scenarios"]}
    rows = re.findall(
        r"^\|\s*\*{0,2}(S\d[a-z])\*{0,2}\s*\|\s*(\S+)\s*\|[^|]*\|[^|]*\|"
        r"\s*\*{0,2}([\d,]+)\*{0,2}\s*\|\s*\*{0,2}(\d)",
        read(CATALOG), re.M)
    parsed = {sid: (hz, int(ep.replace(",", "")), int(pr)) for sid, hz, ep, pr in rows}

    assert set(parsed) == set(scenarios), (
        f"총괄표의 시나리오 집합이 다릅니다 — 문서만: {sorted(set(parsed) - set(scenarios))}, "
        f"정본만: {sorted(set(scenarios) - set(parsed))}")
    wrong = {sid: (parsed[sid], (s["hazard_class"], s["episodes"], s["priority"]))
             for sid, s in scenarios.items()
             if parsed[sid] != (s["hazard_class"], s["episodes"], s["priority"])}
    assert not wrong, f"총괄표가 정본과 다릅니다 (문서, 정본): {wrong}"

    total = sum(s["episodes"] for s in scenarios.values())
    assert f"{total:,}" in read(CATALOG), f"총괄표 합계 {total:,} 를 찾지 못했습니다"
    must_have = sum(s["episodes"] for s in scenarios.values() if s["priority"] <= 2)
    assert f"우선순위 1~2({must_have:,}개)" in read(CATALOG), (
        f"「우선순위 1~2」 확보분이 정본 합({must_have:,})과 다릅니다")


# A row of a condition table: (scenario, column, window). The catalog and the card (both halves) write the same rows.
CONDITION_ROW = re.compile(r"^\| `(S\d[a-z])` \| `(\w+)` \| ([^|]+?) \|", re.M)


def conditions_rows(scenario_ids, fixed_word: str) -> set[tuple[str, str, str]]:
    """조건표가 ``scenario_ids`` 에 대해 보여야 하는 행 — (시나리오, 열, 창) — 을 정본 `conditions` 에서 만든다.

    창은 `range` 면 「하한 ~ 상한」, `choice` 면 백틱 선택지를 ` · ` 로 이은 것이고, `fixed` 의 값은 「값 (``fixed_word``)」 —
    카탈로그와 국문 카드는 「고정」, 영문 카드는 「fixed」. 정본에 조건표가 없는 시나리오를 주면 실패한다.
    """
    by_id = {s["scenario_id"]: s for s in canon("scenarios.json")["scenarios"]}
    want = set()
    for sid in scenario_ids:
        assert "conditions" in by_id.get(sid, {}), f"{sid} 는 정본에 조건표(conditions)가 없습니다"
        block = by_id[sid]["conditions"]
        for name, col in block["columns"].items():
            window = (f"{col['range'][0]:g} ~ {col['range'][1]:g}" if "range" in col
                      else " · ".join(f"`{c}`" for c in col["choice"]))
            want.add((sid, name, window))
        want |= {(sid, name, f"{value:g} ({fixed_word})") for name, value in block["fixed"].items()}
    return want


def test_the_conditions_tables_match_canon():
    """카탈로그 §5 의 조건표 ↔ `scenarios.json` 의 `conditions`(열 · 창 · 고정값), 그리고 그 위 문단의 가드 · 멈춤 거리 ↔
    `generation/conditions.py` 의 상수. 창이 바뀌면 이 표가 가장 먼저 낡는다."""
    import yaml

    from generation import conditions, sites

    with_table = [s["scenario_id"] for s in canon("scenarios.json")["scenarios"] if "conditions" in s]
    assert with_table, "정본에 conditions 를 가진 시나리오가 없습니다 — 카탈로그 §5 의 조건표와 이 시험을 함께 고치십시오"
    want = conditions_rows(with_table, "고정")
    said = set(CONDITION_ROW.findall(read(CATALOG)))
    assert said == want, (f"카탈로그 §5 의 조건표가 정본과 다릅니다 — 문서만: {sorted(said - want)}, "
                          f"정본만: {sorted(want - said)}")

    para = [text for _, text in paragraphs(read(CATALOG)) if text.startswith("조건표 — ")]
    assert len(para) == 1, f"조건표 문단을 {len(para)}개 찾았습니다 (1개여야 합니다)"
    lo, hi = sites.OBSTACLE_Z_M
    accel = yaml.safe_load(conditions.GAIT_SPEC.read_text(encoding="utf-8"))["ramp"]["accel_time_s"]
    for phrase in (f"{conditions.MAX_ATTEMPTS}번이 모두 걸리면",
                   f"로봇 몸(반폭 {conditions.ROBOT_HALF_WIDTH_M:g} m)",
                   f"높이 {lo:g}~{hi:g} m 의 충돌체에서 {conditions.ROBOT_CLEARANCE_M:g} m 밖",
                   f"원이 충돌체에서 {conditions.ROBOT_CLEARANCE_M:g} m 밖",
                   "`S1g` 는 사람이 반대쪽에서 와도 그 길이 가드를 넘어야 합니다",
                   f"충돌체에서 {conditions.HUMAN_CLEARANCE_M:g} m 밖이고 건물 벽에서 {conditions.WALL_INSET_M:g} m 안쪽",
                   f"건너편 랙 면 {conditions.FAR_FACE_STOP_M:g} m 앞에서",
                   f"로봇의 x 를 {conditions.WALK_PAST_M:g} m 지나서",
                   f"정지에서 {accel:g} s 에 걸쳐"):
        assert phrase in para[0], f"카탈로그 §5 의 조건표 문단에 「{phrase}」가 없습니다 — 코드와 문단이 어긋났습니다"


def test_the_hazard_classes_left_after_deferral_are_counted_from_canon():
    """카탈로그 §3 — 우선순위 3~4 를 이월하면 `hazard_class` 가 몇 종, 무엇이 남는가.

    이 문장은 「이 경우에도 `hazard_class` 6종이 모두 포함」이라고 적혀 있었고, 정본으로 세면
    넷이었다 — 위 시험이 총괄표의 행은 정본과 대조하면서, 그 표를 요약한 바로 아래 문장은
    아무도 세지 않았다. 종 수도 이름도 여기 적지 않고 정본에서 센다.
    """
    scenarios = canon("scenarios.json")["scenarios"]
    every = {s["hazard_class"] for s in scenarios}
    kept = {s["hazard_class"] for s in scenarios if s["priority"] <= 2}
    assert kept < every, (
        "우선순위 1~2 가 hazard_class 를 모두 담게 되었습니다 — 「…은 우선순위 3~4 에만」이라는 "
        "문장 자체가 틀렸으니 카탈로그 §3 과 이 시험의 패턴을 함께 고치십시오")

    said = only(r"남는 `hazard_class` 는 총괄표의 (\d+)종 가운데 (\d+)종\(([^)]*)\)이며, "
                r"([^.]*?)은 우선순위 3~4 에만", " ".join(read(CATALOG).split()), "카탈로그 §3")

    def names(listed: str) -> set[str]:
        return {name.strip() for name in listed.split("·")}

    assert (int(said[1]), int(said[2])) == (len(every), len(kept)), (
        f"카탈로그 §3 은 {said[1]}종 가운데 {said[2]}종이 남는다고 적는데 정본은 "
        f"{len(every)}종 가운데 {len(kept)}종입니다")
    assert names(said[3]) == kept, f"남는 종 — 문서 {sorted(names(said[3]))}, 정본 {sorted(kept)}"
    assert names(said[4]) == every - kept, (
        f"이월되는 종 — 문서 {sorted(names(said[4]))}, 정본 {sorted(every - kept)}")


_RANDOMIZATION_BLOCKS = ("randomization", "randomization_fixed", "policy_observation_noise")

# 겉모습 표의 머리 — 축이 첫 칸이다(시나리오 id 로 시작하는 줄은 조건표 행으로 읽힌다)
APPEARANCE_HEADS = {"ko": "| 축 | 고르는 값 | 입히는 곳 |", "en": "| Axis | Values | Applied to |"}


def is_appearance_table(header: str) -> bool:
    return header.strip() in APPEARANCE_HEADS.values()


def randomization_tables(body: str, where: str) -> dict[str, dict[str, list[float]]]:
    """문서의 한 절(``body``)에 있는 랜덤화 표들을 정본 블록 이름별로 — {블록: {정본 키: [수, …]}}.

    카탈로그 §4.5 와 데이터셋 카드(영 · 국)가 같은 모양의 표 셋을 싣는다. ``where`` 는 오류
    문장에 적을 절의 이름이다.

    표가 어느 블록의 것인지는 **표 바로 앞 산문이 마지막으로 든 블록 이름**으로 정한다(백틱 안).
    행은 둘째 칸의 정본 키로 찾고, 키 없는 데이터 행은 **거부한다** — 버리면 「바닥 마찰계수 0.3 – 1.0」
    같은 행을 더해도 아무것도 대조되지 않는다. 유니코드 빼기 부호(U+2212)는
    빼기로 읽는다 — 안 읽으면 「−0.4」가 0.4 로 대조된다.

    **조건표는 이 표가 아니다.** 데이터 행의 첫 칸이 백틱 시나리오 id(`` `S1e` ``)인 표는 조건표이고 건너뛴다 — 그 행들은
    `conditions_rows` 가 정본과 대조한다(카드의 「에피소드마다 달라지는 것」은 조건표와 랜덤화 표를 한 절에 싣는다). 한 표에
    조건표 행과 그렇지 않은 행이 섞이면 거부한다.

    **못 보는 것 하나를 더 적는다 — 첫 칸의 이름.** 「사람의 진입 속도」를 「바닥 마찰계수」로 바꾸고
    키와 값을 그대로 두면 통과한다. 이름은 문서 셋 · 언어 둘에 따로 적힌 산문이라
    대조할 정본이 없다 — 이름 목록을 하나 더 손으로 두면 그것이 또 하나의 사본이 된다. 독자가 보는
    행에는 정본 키가 이름 바로 옆에 있다.
    """
    tables: dict[str, dict[str, list[float]]] = {}
    prose: list[str] = []
    rows: list[str] = []

    def close():
        if not rows:
            return
        if is_appearance_table(rows[0]):                        # 겉모습 표 — `appearance_rows` 가 정본과 대조한다
            prose.clear()
            rows.clear()
            return
        scenario_first = [bool(re.match(r"\|\s*`S\d[a-z]`\s*\|", row.strip())) for row in rows[2:]]
        if any(scenario_first):
            assert all(scenario_first), f"{where} 의 한 표에 조건표 행과 다른 행이 섞였습니다 — 표 머리: {rows[0]}"
            prose.clear()
            rows.clear()
            return
        named = re.findall(r"`(%s)`" % "|".join(_RANDOMIZATION_BLOCKS), " ".join(prose))
        assert named, f"{where} 의 표 앞 산문이 정본 블록 이름을 들지 않습니다 — 표 머리: {rows[0]}"
        block = named[-1]
        assert block not in tables, f"{where} 에 `{block}` 의 표가 둘입니다"
        parsed: dict[str, list[float]] = {}
        for row in rows[2:]:                                    # header, rule, then data
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            key = re.fullmatch(r"`([a-z_]+)`", cells[1]) if len(cells) >= 3 else None
            assert key, f"{where} `{block}` 표에 정본 키 없는 행이 있습니다: {row}"
            assert key.group(1) not in parsed, (
                f"{where} `{block}` 표에 같은 정본 키가 두 번 나옵니다: {key.group(1)}")
            # 값 칸은 정해진 꼴만 받는다 — 뽑는 축은 「하한 – 상한 [단위]」, 나머지는 「값 [단위]」.
            # 숫자만 골라내면 「0.4 ± 1.6」도 [0.4, 1.6] 으로 읽혀 통과한다.
            cell = cells[2].replace("\u2212", "-")
            number = r"(-?\d+(?:\.\d+)?)"
            unit = r"(?:\s*[^\d\s–±~\-].*)?"
            shape = (number + r"\s*–\s*" + number + unit) if block == "randomization" else (number + unit)
            value = re.fullmatch(shape, cell)
            assert value, (
                f"{where} `{block}` 표의 `{key.group(1)}` 값 칸을 읽지 못했습니다: {cells[2]!r} — "
                + ("「하한 – 상한 [단위]」" if block == "randomization" else "「값 [단위]」") + " 로 적습니다")
            parsed[key.group(1)] = [float(n) for n in value.groups()]
        tables[block] = parsed
        prose.clear()
        rows.clear()

    for line in body.splitlines():
        if line.lstrip().startswith("|"):
            rows.append(line)
        else:
            close()
            prose.append(line)
    close()
    return tables


def catalog_randomization_section(text: str) -> str:
    return text[text.index("### 4.5 "):text.index("### 4.6 ")]


def randomization_differences(body: str, entry: dict, where: str = "카탈로그 §4.5") -> list[str]:
    """그 절의 표와 정본 항목이 다른 자리. 같으면 빈 목록.

    정본에서 비지 않은 블록마다 표가 있어야 한다. **정본에서 빈 블록은 표를 요구하지 않는다** — 조건표가 있는 시나리오는
    `randomization` 이 비어 있다(에피소드마다 바뀌는 것은 조건표의 행). 빈 블록의 표가 있으면 그 표의 키가 「문서만」으로 나온다.
    """
    tables = randomization_tables(body, where)
    out = []
    wanted = {block for block in _RANDOMIZATION_BLOCKS if entry[block]}
    if not wanted <= set(tables):
        out.append(f"표가 없는 블록 {sorted(wanted - set(tables))} — 정본에서 비지 않은 블록마다 표가 있어야 합니다")
    for block in _RANDOMIZATION_BLOCKS:
        canon_block = entry[block]
        expected = {k: ([float(x) for x in v] if isinstance(v, list) else [float(v)])
                    for k, v in canon_block.items()}
        parsed = tables.get(block, {})
        if set(parsed) != set(expected):
            out.append(f"`{block}` — 문서만: {sorted(set(parsed) - set(expected))}, "
                       f"정본만: {sorted(set(expected) - set(parsed))}")
        out += [f"`{block}`.{k} — 문서 {parsed[k]} · 정본 {expected[k]}"
                for k in expected if k in parsed and parsed[k] != expected[k]]
    return out


def test_s1a_randomization_tables_match_canon():
    """카탈로그 §4.5 의 세 표 ↔ `scenarios.json` S1a 의 랜덤화 블록 셋.

    이 절과 정본이 **같은 틀린 값**(바닥 마찰 0.3–1.0 · 질량 ±5%)을 적고 있던 적이
    있다. 둘 다 계획을 옮긴 것이었고 그 값으로 만든 에피소드는 없었다. 정본이
    환경과 맞는지는 generation/randomization.py 가 보고, 여기서는 문서 ↔ 정본만 본다.

    표마다 그 표의 블록과 대조하고(키 집합이 같아야 한다), 정본 키 없는 행은 거부한다.
    못 보는 것: 셋째 칸의 단위, 넷째 칸의 서술(분포 · 어느 링크인가), 표 밖의 산문 —
    「계획했으나 없는 것」 목록이 그 산문이다.
    """
    s1a = next(s for s in canon("scenarios.json")["scenarios"] if s["scenario_id"] == "S1a")
    assert randomization_differences(catalog_randomization_section(read(CATALOG)), s1a) == []


def _moved_to_the_fixed_table(text: str) -> str:
    row = "| 관절 위치 | `joint_pos` | 0.01 rad |\n"
    anchor = "| 지면 재질 — 동적 마찰 | `terrain_dynamic_friction` | 1.0 |\n"
    assert text.count(row) == 1 and text.count(anchor) == 1
    return text.replace(row, "").replace(anchor, anchor + row)


def _a_row_without_a_key(text: str) -> str:
    anchor = "| 로봇 시작 피치 각속도 | `robot_start_pitch_rate` | -0.5 – 0.5 rad/s | 위와 같음 |\n"
    assert text.count(anchor) == 1
    return text.replace(anchor, anchor + "| 바닥 마찰계수 | — | 0.3 – 1.0 | 에피소드마다 뽑음 |\n")


def _a_unicode_minus(text: str) -> str:
    cell = "| `human_entry_speed` | 0.4 – 1.6 m/s |"
    assert text.count(cell) == 1
    return text.replace(cell, "| `human_entry_speed` | \u22120.4 – 1.6 m/s |")


def _a_range_written_as_centre_and_half_width(text: str) -> str:
    cell = "| `human_entry_speed` | 0.4 – 1.6 m/s |"
    assert text.count(cell) == 1
    return text.replace(cell, "| `human_entry_speed` | 0.4 ± 1.6 m/s |")


def _a_table_nobody_named(text: str) -> str:
    lead = "**랜덤화하지 않고 고정한 값** (정본 `randomization_fixed`):"
    assert text.count(lead) == 1
    return text.replace(lead, "**랜덤화하지 않고 고정한 값**:")


@pytest.mark.parametrize("edit", [_moved_to_the_fixed_table, _a_row_without_a_key, _a_unicode_minus,
                                  _a_table_nobody_named, _a_range_written_as_centre_and_half_width])
def test_the_randomization_tables_check_refuses_an_edit_that_changes_what_the_page_says(edit):
    """위 대조가 거부해야 하는 편집 다섯 — 앞의 셋은 고치기 전의 대조를 실제로 통과했다.

    행을 다른 표로 옮김 · 정본 키 없는 행을 더함 · 빼기 부호를 U+2212 로 적음 · 표가 어느
    블록의 것인지 말하는 문장을 지움 · 범위를 「가운데 ± 반폭」으로 적음. 편집은 메모리에서만 하고,
    대조가 그것을 거부하는지 본다.
    """
    s1a = next(s for s in canon("scenarios.json")["scenarios"] if s["scenario_id"] == "S1a")
    edited = edit(read(CATALOG))
    try:
        found = randomization_differences(catalog_randomization_section(edited), s1a)
    except AssertionError:
        return
    assert found, "편집한 문서가 정본과 같다고 나왔습니다"


# 카드의 「에피소드마다 달라지는 것」 — (절의 머리, 그 절이 끝나는 다음 머리)
CARD_RANDOMIZATION_SECTIONS = {
    "en": ("### What varies from episode to episode\n", "\n## Loading\n"),
    "ko": ("### 에피소드마다 달라지는 것\n", "\n## 내려받아 여는 법\n"),
}
# 조건표의 고정값에 붙는 낱말 — 「0.5 (fixed)」 · 「0.5 (고정)」
CARD_FIXED_WORD = {"en": "fixed", "ko": "고정"}


def card_section(start: str, ends=("\n## ", "\n### ")) -> str:
    """카드에서 ``start`` 머리로 시작하는 절 — ``ends`` 가운데 먼저 오는 것까지. 그 머리는 카드에 정확히 하나여야 한다."""
    text = read(CARD)
    assert text.count(start) == 1, f"카드에 「{start.strip()}」 절이 {text.count(start)}개입니다 (1개여야 합니다)"
    body = text[text.index(start) + len(start):]
    cuts = [body.find(end) for end in ends if end in body]
    return body[:min(cuts)] if cuts else body


@pytest.mark.parametrize("language", sorted(CARD_RANDOMIZATION_SECTIONS))
def test_the_card_randomization_tables_match_canon(language):
    """카드(영 · 국)의 랜덤화 표 ↔ 정본 — 카드가 말하는 시나리오(`card_scenarios`)마다. 카드는 배포본의 README 라 받은 쪽이
    처음 읽는 자리다.

    표 한 벌이 카드의 모든 시나리오에 맞아야 한다 — 그 시나리오들의 블록이 서로 같은 동안만 참이고, 갈라지면 여기가
    빨개진다(그때는 카드가 시나리오마다 표를 보여야 한다). 정본에서 빈 블록은 표를 요구하지 않는다(`randomization_differences`).
    못 보는 것은 카탈로그 쪽과 같다: 단위 · 넷째 칸 · 표 밖의 산문.
    """
    start, end = CARD_RANDOMIZATION_SECTIONS[language]
    body = card_section(start, (end,))
    by_id = {s["scenario_id"]: s for s in canon("scenarios.json")["scenarios"]}
    for sid in card_scenarios():
        assert randomization_differences(body, by_id[sid], f"카드({language}) {sid}") == []


@pytest.mark.parametrize("language", sorted(CARD_RANDOMIZATION_SECTIONS))
def test_the_card_conditions_tables_match_canon(language):
    """카드(영 · 국)의 「에피소드마다 달라지는 것」의 조건표 ↔ 정본 `conditions` — 카드가 말하는 시나리오의 행 전부, 양쪽으로.

    조건표의 시나리오에서 에피소드마다 바뀌는 것은 그 행이 전부다. 카드에만 있는 행 · 정본에만 있는 행 · 창이 다른 행을 모두
    거부하고, 같은 행이 두 번 나와도 거부한다. 못 보는 것: 넷째 칸(뜻)과 표 밖의 산문.
    """
    start, end = CARD_RANDOMIZATION_SECTIONS[language]
    rows = CONDITION_ROW.findall(card_section(start, (end,)))
    twice = sorted({row for row in rows if rows.count(row) > 1})
    assert not twice, f"카드({language})의 조건표에 같은 행이 두 번 나옵니다: {twice}"
    said = set(rows)
    want = conditions_rows(card_scenarios(), CARD_FIXED_WORD[language])
    assert said == want, (f"카드({language})의 조건표가 정본과 다릅니다 — 문서만: {sorted(said - want)}, "
                          f"정본만: {sorted(want - said)}")


def test_s1d_is_said_to_draw_what_s1a_draws_only_while_the_canon_agrees():
    """카탈로그 §5 「랜덤화 축은 S1a 와 같고」 ↔ 두 항목의 랜덤화 블록이 실제로 같은가."""
    only(r"랜덤화 축은 `S1a`와\s+같고", read(CATALOG), "카탈로그 §5 S1d")
    by_id = {s["scenario_id"]: s for s in canon("scenarios.json")["scenarios"]}
    keys = ("randomization", "randomization_distribution", "randomization_fixed", "policy_observation_noise")
    differ = [k for k in keys if by_id["S1a"].get(k) != by_id["S1d"].get(k)]
    assert not differ, f"S1d 의 랜덤화 블록이 S1a 와 다릅니다: {differ} — 카탈로그 §5 의 문장을 고치십시오"


def test_the_box_script_and_task_paragraphs_match_the_code():
    """카탈로그 §5 의 「든 상자」 · 「사람 대본」 · 「과제와 끝」 문단의 수 ↔ 그 수를 쓰는 코드.

    상자의 폭(`generation/scene/carried_box.py`) · 걸음의 가속 · 감속 시간과 돈 뒤 걷기 시작하는 빠르기(걸음 명세) · 세워
    두는 제어기가 풀리는 때(거리 게이트 없음 · 상자와 같은 문턱의 첫 접촉 — 환경 설정과 명령 항 설정의 원문에서 읽음, Isaac
    없이는 import 하지 못함) · 길을 따르는 규칙(`generation/task_path.py`) · 15 s 규칙(과제 경로 시험의 창). 못 보는 것: 문단의
    나머지 산문, 최대 20 s(Isaac Lab 보행 설정에서 물려받는 값).
    """
    import yaml

    from generation import task_path
    from generation.scene.carried_box import NOMINAL_SIZE_M

    def para(head: str) -> str:
        found = [text for _, text in paragraphs(read(CATALOG)) if text.startswith(head)]
        assert len(found) == 1, f"카탈로그 §5 에서 「{head}」 문단을 {len(found)}개 찾았습니다 (1개여야 합니다)"
        return found[0]

    gait = yaml.safe_load(read(REPO / "generation/dummy/gait/gait_spec.yaml"))
    ramp = gait["ramp"]
    window = re.findall(r"^WINDOW_S = (\S+)$", read(REPO / "tests/generation/test_task_path.py"), re.M)
    assert len(window) == 1, "과제 경로 시험의 창(WINDOW_S)을 찾지 못했습니다"
    # The hold's release: no distance gate, and the box's threshold -- the recorder's contact_onset.
    assert "dummy_release_gap_m=None)" in read(REPO / "generation/envs/hri/hri_env_cfg.py"), (
        "조건표 시나리오의 환경이 붙잡기의 거리 게이트를 끄지 않습니다 — 카탈로그 §5 「사람 대본」 문단과 함께 고치십시오")
    assert re.search(r"^    release_threshold_n: float = CONTACT_EPS_N$", read(REPO / "generation/envs/hri/commands.py"),
                     re.M), "붙잡기와 상자를 놓는 문턱이 접촉 시작의 잡음 문턱이 아닙니다 — 카탈로그 §5 와 함께 고치십시오"
    expected = {
        "든 상자 — ": [f"폭은 {NOMINAL_SIZE_M[0]:g} m 로 고정"],
        "사람 대본 — ": [f"출발에서 {ramp['accel_time_s']:g} s 에 걸쳐 오르고 멈출 자리 앞에서 {ramp['decel_time_s']:g} s 에 걸쳐",
                       f"걸음 표의 가장 느린 빠르기 {gait['speeds']['grid_mps'][0]:g} m/s 에서",
                       "처음 힘을 준 스텝 — 그 에피소드의 첫 `contact_onset` 이자 상자를 놓는 스텝 — 에 풀리고",
                       "거리로 미리 풀지 않음"],
        "과제와 끝 — ": [f"길에서 {task_path.LOOKAHEAD_M:g} m 앞의 점",
                       f"끝 {task_path.SLOWDOWN_M:g} m 에서는 남은 거리에 비례해",
                       f"요 오차 {task_path.TURN_DONE_DEG:g}° 안",
                       f"{float(window[0]):g} s 안에 목표에 서도록"],
    }
    for head, phrases in expected.items():
        text = para(head)
        for phrase in phrases:
            assert phrase in text, f"카탈로그 §5 「{head}」 문단에 「{phrase}」가 없습니다 — 코드와 문단이 어긋났습니다"


def test_the_specs_turn_stage_ends_at_the_followers_own_threshold():
    """규격서 §5 `notes` 행 「제자리에서 도는 단계는 목표 방향까지 남은 요가 N° 안에 든 첫 스텝」 ↔
    `generation/task_path.py` `TURN_DONE_DEG` — 단계가 끝나는 문턱은 길을 따르는 쪽이 돌기를 마치고 걷기를 시작하는
    문턱 그 하나다(`StageTracker` 가 그 단계를 길을 따르는 쪽의 `turn` 이 끝난 첫 스텝에 닫음)."""
    from generation import task_path
    only(re.escape(f"제자리에서 도는 단계는 목표 방향까지 남은 요가 {task_path.TURN_DONE_DEG:g}° 안에 든 첫 스텝"),
         read(SPEC), "규격서 §5 notes 행의 task_stages")


def test_the_table_scenarios_are_said_to_vary_by_the_row_only_while_the_canon_agrees():
    """카탈로그 §5 「에피소드마다 뽑는 축은 비어 있고 … `S1d` 와 같습니다」 ↔ 조건표가 있는 항목의 네 블록.

    네 블록은 환경 설정에서 읽어 쓴다(generation/envs/s1a/check_randomization.py). 여기서는 그 블록이 문장이
    말하는 그대로인지 — 뽑는 축 · 분포가 비어 있고, 고정한 값 · 정책 관측의 잡음이 `S1d` 의 것과 같은지만 본다.
    """
    text = read(CATALOG)
    only(r"에피소드마다 뽑는 축\(`randomization`\)과\s+균등이 아닌 분포\(`randomization_distribution`\)는\s+비어\s+있고",
         text, "카탈로그 §5 S1e · S1f · S1g")
    only(r"고정한 값\(`randomization_fixed`\)과\s+보행 정책 관측의 잡음\(`policy_observation_noise`\)은\s+`S1d`\s*와\s+같습니다",
         text, "카탈로그 §5 S1e · S1f · S1g")
    by_id = {s["scenario_id"]: s for s in canon("scenarios.json")["scenarios"]}
    tables = [sid for sid, s in by_id.items() if "conditions" in s]
    assert tables, "정본에 conditions 를 가진 시나리오가 없습니다 — 카탈로그 §5 의 문장과 이 시험을 함께 고치십시오"
    for sid in tables:
        entry = by_id[sid]
        assert entry.get("randomization") == {} and entry.get("randomization_distribution") == {}, (
            f"{sid} 가 조건표 밖에서 뽑는 축을 선언합니다 — 카탈로그 §5 의 문장을 고치십시오")
        differ = [k for k in ("randomization_fixed", "policy_observation_noise") if entry.get(k) != by_id["S1d"].get(k)]
        assert not differ, f"{sid} 의 {differ} 가 S1d 와 다릅니다 — 카탈로그 §5 의 문장을 고치십시오"


# ─────────────────────────────────────────────────────────────────────
# 정본의 채움 상태를 말하는 자리
# ─────────────────────────────────────────────────────────────────────

def standard_thresholds(scope: str | None = None, root: dict | None = None) -> dict[str, int]:
    """`determined_by` 를 가진 항목을 층위에 상관없이 모아 상태별로 센다.

    `root` 를 주면 정본 대신 그 문서를 센다 — 배포본이 싣는 `rules.json`(`card_release_rules`)처럼
    정본을 좁힌 문서의 채움 상태를 말하는 문장은 그 문서로 세어야 한다.

    `value` 를 가진 객체만 모으면 부위 수준 플래그를 놓친다 — `contact_permitted` 가
    정확히 그 형태다.

    `scope` 를 주면 정본의 그 하위 트리만 센다. **문장이 범위를 못박고 있으면 세는
    범위도 그 문장과 같아야 한다** — 규격서는 「부속서 A 의 N개」, 인벤토리는
    「`contact_limits` 의 임계값 N개」라고 적는다. `contact_limits` **밖**에도 값을
    실은 임계값(`human_speed`)이 있어, 정본 전체를 세면 검사가 **옳은 문장을 거부**한다.
    """
    def walk(node):
        if isinstance(node, dict):
            if "determined_by" in node:
                yield node
            for v in node.values():
                yield from walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v)

    root = canon("rules.json") if root is None else root
    if scope is not None:
        assert scope in root, f"정본에 {scope} 트리가 없습니다"
        root = root[scope]

    counts = {"value": 0, "reference_only": 0, "not_applicable": 0}
    for entry in walk(root):
        if entry.get("determined_by") != "standard":
            continue
        if entry.get("not_applicable"):
            counts["not_applicable"] += 1
        elif entry.get("value") is not None:
            counts["value"] += 1
        else:
            counts["reference_only"] += 1
    return counts


# (문서, 「값이 실린 것」을 말하는 패턴이 못박은 범위, 그 패턴, 「조항 참조만」 패턴)
# 「조항 참조만」 쪽은 어느 문서에서도 범위를 못박지 않으므로 정본 전체로 센다. 카드는 여기 없다 — 카드는
# 배포본이 싣는 규칙의 채움 상태를 말하므로 `test_the_card_states_the_fill_state_of_what_it_ships` 가 본다.
FILL_STATE_CLAIMS = [
    (SPEC, "contact_limits",
           r"부속서 A 의 (\d+)개를 `meta/rules\.json` 으로 옮겼습니다",
           r"값 없이 조항 참조만 실린 항목이 (\d+)개 있습니다"),
    (INVENTORY, "contact_limits",
                r"`contact_limits` 의 \*\*임계값 (\d+)개\*\*",
                r"조항 참조만 실린 임계값이 (\d+)개 있습니다"),
]


@pytest.mark.parametrize("path,scope,valued,ref_only",
                         FILL_STATE_CLAIMS,
                         ids=[p.name for p, _, _, _ in FILL_STATE_CLAIMS])
def test_documents_state_the_same_fill_state_as_canon(path, scope, valued, ref_only):
    """「값이 몇 개 실렸는가」를 말하는 문장을 정본에서 유도한 수와 대조한다.

    채움 상태를 말하는 문장은 정본이 값을 실어도 그대로 남아 낡는다. 그래서 그
    자리를 §9.3 하나로 모으고, 그 하나를 여기서 정본에 묶는다. 인벤토리는 배포본에 함께 나가므로 같이 묶는다.
    """
    text = read(path)
    if valued:
        scoped = standard_thresholds(scope)
        got = int(only(valued, text, f"{path.name} · 값이 실린 개수").group(1))
        assert got == scoped["value"], (
            f"{path.name} 은 {got}개라 하고 정본의 {scope} 는 {scoped['value']}개입니다")
    got = int(only(ref_only, text, f"{path.name} · 조항 참조만").group(1))
    whole = standard_thresholds()
    assert got == whole["reference_only"], (
        f"{path.name} 은 {got}개라 하고 정본은 {whole['reference_only']}개입니다")


def test_a_standard_value_outside_contact_limits_is_named_in_the_spec():
    """`contact_limits` 밖에서 값이 실린 표준 임계값은 규격서가 이름으로 밝힌다.

    위 검사가 「부속서 A 의 N개」를 `contact_limits` 범위로 세도록 좁히면서, **그
    밖에서** 값이 늘어도 세 문장이 모두 초록인 자리가 생겼다. 좁히기가 만든 그 구멍을
    여기서 막는다 — 개수가 아니라 **이름**을 요구한다. 개수를 또 세면 어느 문장이
    그 수를 말하는지 정해야 하는데, 이 항목들은 접촉 한계와 달리 문서에서 개별로
    설명되지 하나의 수로 세어지지 않기 때문이다.

    `human_speed` 가 첫 사례다 — 그 값은 KS B ISO TS 15066
    부속서 A 가 아니라 KS B ISO 13855 에서 온다.
    """
    rules = canon("rules.json")

    def walk(node, path=()):
        if isinstance(node, dict):
            if node.get("determined_by") == "standard" and node.get("value") is not None:
                yield path
            for k, v in node.items():
                yield from walk(v, path + (k,))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from walk(v, path + (str(i),))

    outside = [p for p in walk(rules) if p and p[0] != "contact_limits"]

    # **문서 전체에서 이름을 찾으면 아무것도 지키지 않는다.** 필드 표에 한 번,
    # 부록에 한 번 나오는 것만으로 통과하므로 §9.3 이 그 항목을 설명하지 않아도
    # 초록이다 (설명을 전부 지우고 코드 블록에 이름만 둬도 통과).
    # 채움 상태를 말하는 자리를 §9.3 하나로 모은 것이 이 파일의 방침이므로,
    # **그 절의 산문 안에** 있는지를 묻는다.
    spec = read(SPEC)
    cut = re.search(r"^### 9\.3 .*?(?=^### 9\.4 )", spec, re.S | re.M)
    assert cut, "규격서에서 §9.3 절을 찾지 못했습니다 — 제목을 고쳤다면 여기도 고치십시오"
    prose = " ".join(para for _, para in
                     paragraphs(re.sub(r"```.*?```", " ", cut.group(0), flags=re.S)))
    missing = [p for p in outside if p[-1] not in prose]
    assert not missing, (
        "`contact_limits` 밖에서 값이 실린 표준 임계값을 규격서 §9.3 이 "
        "설명하지 않습니다: " + ", ".join(".".join(p) for p in missing))


def withholding_claims(paths) -> list[str]:
    """Paragraphs of ``paths`` that say the release does not carry the standard values."""
    forbidden = re.compile(
        r"배포본에[^\n]{0,12}(그 )?값이 (실리지|싣지) 않"
        r"|배포본은[^\n]{0,20}수치를 (싣지|담지) 않"
        r"|standard 임계값[^\n]{0,20}값 없이 조항 참조만[^\n]{0,10}싣")
    hits = []
    for path in paths:
        # 문단 단위로 본다 — 줄 단위면 접힌 문장을 놓친다 (`paragraphs` 참고).
        for line_no, para in paragraphs(read(path)):
            if forbidden.search(para):
                hits.append(f"{path.relative_to(REPO)}:{line_no}: {para[:110]}")
    return hits


def test_no_document_claims_the_release_withholds_the_values():
    """배포본이 표준 수치를 싣는다는 사실과 반대되는 진술이 남아 있지 않은지.

    **약한 검사다** — 활용형 목록을 열거하는 방식이라 다른 표현은 못 잡는다
    (grep 은 같은 뜻의 다른 표현을 못 찾는다). 그래도 두는 이유는
    실제로 나온 형태가 정확히 이것이었고, 이 문장들이 되살아나면
    배포본과 반대를 말하기 때문이다. 정본이 값을 싣지 않게 되면 이 검사부터
    지우십시오.
    """
    if standard_thresholds()["value"] == 0:
        pytest.skip("정본이 표준 값을 싣지 않습니다 — 이 검사는 그때 의미가 없습니다")

    hits = withholding_claims([SPEC, CATALOG, CARD, INVENTORY])
    assert not hits, "배포본이 값을 싣는데 반대로 적힌 자리:\n  " + "\n  ".join(hits)


def test_biomechanical_values_are_attributed_to_the_table_canon_names():
    """유효질량·유효탄성계수를 말하는 문장의 표 번호를 정본의 `standard_ref` 에서 유도한다.

    규격서 §8 이 이 둘을 표 A.2 로 돌리고 있었다 — 힘·압력의 표다. 같은 문서
    §9.2 가 「표 A.3 이다. 표 A.2 가 아니다」라고 적고 있었는데도 셋 섹션 위에서
    반대를 말했다.
    """
    limits = canon("rules.json")["contact_limits"]
    expected = set()
    for entry in limits.values():
        for item in (entry.get("biomechanical") or {}).values():
            found = re.search(r"표 (A\.\d)", str(item.get("standard_ref", "")))
            assert found, f"정본의 biomechanical.standard_ref 에 표 번호가 없습니다: {item}"
            expected.add(found.group(1))
    assert len(expected) == 1, f"정본이 여러 표를 가리킵니다: {sorted(expected)}"
    table = expected.pop()

    term = re.compile(r"유효\s?질량|유효\s?탄성\s?계수|스프링\s?상수|effective_mass|spring_constant")
    wrong = []
    for path in (SPEC, CATALOG, REPO / "generation/dummy/anthropometry.yaml",
                 REPO / "generation/dummy/build_dummy.py"):
        for line_no, line in paragraphs(read(path)):
            if not term.search(line):
                continue
            cited = {a or b for a, b in
                     re.findall(r"[Tt]able (A\.\d)|표 (A\.\d)", line)}
            # 「정본의 표만 부를 것」이 아니라 **「부른다면 그 안에 있을 것」** 이다.
            # 집합이 같기를 요구하면 정확히 맞게 쓴 두 줄이 걸린다 —
            # §9.2 의 「**표 A.3**. 표 A.2 가 아니다」와 빌더의 「부위 구분은 표
            # A.1/A.2, 유효질량은 **표 A.3**」. 옳은 문장은 대비하려고 다른 표를
            # 함께 부른다.
            # 잡으려는 것은 **다른 표 하나에만** 돌리는 형태이고 그건 그대로 걸린다.
            if cited and table not in cited:
                wrong.append(f"{path.relative_to(REPO)}:{line_no}: {line.strip()}")
    assert not wrong, (
        f"정본은 유효질량·탄성계수를 표 {table} 로 돌립니다. 그 표를 부르지 않는 줄:\n  "
        + "\n  ".join(wrong))


# ─────────────────────────────────────────────────────────────────────
# 더미와 생체역학 값 — 문서가 하지 말아야 하는 진술
# ─────────────────────────────────────────────────────────────────────


def embodiment_claims(paths) -> list[str]:
    """Statements of ``paths`` that read as: the dummy was built from the biomechanical values."""
    # 「더미」와 「생체역학 값」이 한 문단에서 만드는·따르는·구성하는 으로 묶이는 자리
    dummy = re.compile(r"더미|dummy|인체 모델")
    values = re.compile(r"유효\s?질량|유효\s?탄성\s?계수|스프링\s?상수|생체역학 파라미터")
    builds = re.compile(r"로 구성한|으로 구성한|를 따른|을 따른|로 만든|으로 만든|"
                        r"를 반영한|을 반영한|로 제작한|으로 제작한")
    negated = re.compile(r"쓰이지 않|쓰지 않|들어가지 않|아님|아니다|읽지 않|않습니다")

    hits = []
    for path in paths:
        for line_no, para in statements(read(path)):
            if dummy.search(para) and values.search(para) and builds.search(para) \
                    and not negated.search(para):
                hits.append(f"{path.relative_to(REPO)}:{line_no}: {para[:120]}")
    return hits


def test_no_document_says_the_dummy_embodies_the_biomechanical_values():
    """더미가 표준의 유효질량·탄성계수를 「따라 만들어졌다」고 말하는 자리가 없는지.

    **같은 주장이 문서마다 다른 낱말로 나올 수 있다**(규격서 §8 등). 어느 것이든
    「기록된 접촉력을 표준 생체역학 모형의 산출로 읽어도 된다」는 뜻이 되므로, 이
    데이터셋이 해서는 안 되는 진술이다.

    **먼저 사실을 유도한다.** 빌더가 `contact_limits` 에서 실제로 꺼내 쓰는 항목이
    표시명 하나뿐인지 본다. 그 값을 쓰게 되면 이 검사는 근거를 잃으므로 **함께
    빨간불이 나서** 문서를 다시 보게 만든다.

    **약한 검사다** — 아래 낱말 조합으로 훑으므로 또 다른 표현은 못 잡는다
    (grep 은 같은 뜻의 다른 표현을 못 찾는다). 되풀이되기 쉬운 형태라 둔다.
    """
    builder = read(REPO / "generation/dummy/build_dummy.py")
    consumed = set(re.findall(r"bio\[code\]\[\"(\w+)\"\]", builder))
    assert consumed == {"name_ko"}, (
        f"빌더가 정본에서 꺼내 쓰는 항목이 바뀌었습니다: {sorted(consumed)} — "
        "생체역학 값을 실제로 쓰게 되었다면 이 검사와 함께 규격서 §8 · 카탈로그 · "
        "카드 · 인벤토리의 서술을 다시 보십시오")

    hits = embodiment_claims([SPEC, CATALOG, CARD, INVENTORY, REPO / "generation/dummy/build_dummy.py"])
    assert not hits, (
        "빌더는 생체역학 값을 읽지 않는데 더미가 그것으로 만들어졌다고 읽히는 자리:\n  "
        + "\n  ".join(hits))


# ─────────────────────────────────────────────────────────────────────
# 데이터셋 카드 — 배포본의 README.md 이고 국·영 두 벌이다
# ─────────────────────────────────────────────────────────────────────

# 카드 두 벌의 제목 — 국문 반쪽이 시작하는 자리
CARD_KOREAN_TITLE = "# AI 휴머노이드 HRI 안전성 데이터셋"
# 카드의 「한눈에」 — (절의 머리, 시나리오 행의 첫 칸, 카메라 수를 말하는 낱말의 틀)
CARD_GLANCE = {
    "en": ("## At a glance\n", "Scenarios", "{} RGB cameras"),
    "ko": ("## 한눈에\n", "시나리오", "카메라 {} 대"),
}
# 수를 말하는 낱말 — (영, 국)
COUNT_WORDS = {5: ("five", "다섯"), 6: ("six", "여섯"), 7: ("seven", "일곱"),
               8: ("eight", "여덟"), 9: ("nine", "아홉")}


def card_scenarios() -> list[str]:
    """카드가 말하는 시나리오 — 「한눈에」 표의 시나리오 행(영 `| Scenarios |` · 국 `| 시나리오 |`)이 백틱으로 든
    시나리오 id 를 처음 나온 차례대로.

    카드의 표를 정본과 대조하는 시험들이 어느 시나리오의 정본과 대조할지를 여기서 읽는다 — 시나리오 이름을 시험에 적어 두면
    카드가 다른 배포본을 말하게 되었을 때 시험은 옛 시나리오를 대조하며 초록으로 남는다. 두 벌이 같고, 비지 않고, 모두 정본에
    있어야 한다.
    """
    said = {}
    for language, (head, label, _) in CARD_GLANCE.items():
        row = only(r"(?m)^\| %s \|(.*)\|$" % re.escape(label), card_section(head),
                   f"카드({language}) 「한눈에」의 시나리오 행")
        ids = list(dict.fromkeys(re.findall(r"`(S\d[a-z])`", row.group(1))))
        assert ids, f"카드({language}) 「한눈에」의 시나리오 행에 백틱 시나리오 id 가 없습니다"
        said[language] = ids
    assert said["en"] == said["ko"], f"카드 두 벌이 다른 시나리오를 말합니다 — 영 {said['en']} · 국 {said['ko']}"
    known = [s["scenario_id"] for s in canon("scenarios.json")["scenarios"]]
    unknown = [sid for sid in said["en"] if sid not in known]
    assert not unknown, f"카드가 정본에 없는 시나리오를 말합니다: {unknown}"
    return said["en"]


def test_the_card_names_its_scenarios_in_both_halves():
    """카드의 「한눈에」가 말하는 시나리오 ↔ 정본에서 조건표(`conditions`)를 가진 시나리오, 정본의 차례로.

    공개 배포본은 조건표가 있는 시나리오로 만든다 — 기록 데이터는 조건표의 행마다 에피소드 하나이고, 조건표가 없는
    시나리오의 배치는 공개하지 않는다. 그래서 기대값을 정본에서 읽는다. 카드가 실린 배포본의 배치를 말하는지는 게시 때
    카드의 표식이 배포본과 대조된다(`publish_dataset.card_matches_batch`) — 여기서는 카드와 정본만 본다.
    """
    expected = [s["scenario_id"] for s in canon("scenarios.json")["scenarios"] if "conditions" in s]
    assert expected, "정본에 conditions 를 가진 시나리오가 없습니다 — 이 시험과 카드를 함께 고치십시오"
    assert card_scenarios() == expected, f"카드는 {card_scenarios()} 를 말하고 정본의 조건표 시나리오는 {expected} 입니다"


# 카드의 「카메라」 절의 머리
CARD_CAMERA_SECTIONS = {"en": "### Cameras\n", "ko": "### 카메라\n"}


@pytest.mark.parametrize("language", sorted(CARD_CAMERA_SECTIONS))
def test_the_card_camera_table_matches_canon(language):
    """카드(영 · 국)의 「카메라」 표 ↔ `meta/cameras.json` — 카메라마다 한 행이고 그 수평 시야각, 그리고 「한눈에」가 말하는
    카메라 수.

    행의 집합과 카메라 수를 함께 대조해 카드와 정본이 다른 카메라를 말할 수 없게 한다. 표의 행은 영상 필드 이름
    (`observation.images.<카메라>`)으로 읽으므로 행이 빠지거나 더해지거나 시야각이 다르면 여기서 걸린다. 못 보는 것: 둘째 칸
    (카메라가 있는 곳)의 산문 — 장착 규칙과 자리는 정본의 `mount` · `sites` 이고 카드는 그것을 실물 말로 옮긴다.
    """
    cameras = canon("cameras.json")["cameras"]
    rows = re.findall(r"^\| `observation\.images\.(\w+)` \|[^|]*\| ([\d.]+)° \|",
                      card_section(CARD_CAMERA_SECTIONS[language]), re.M)
    names = [name for name, _ in rows]
    assert len(names) == len(set(names)), f"카드({language})의 카메라 표에 같은 카메라가 두 번 나옵니다: {names}"
    said = {name: float(fov) for name, fov in rows}
    want = {name: float(cam["fov_h_deg"]) for name, cam in cameras.items()}
    assert said == want, f"카드({language})의 카메라 표가 정본과 다릅니다 — 카드 {said} · 정본 {want}"

    assert len(cameras) in COUNT_WORDS, f"정본의 카메라가 {len(cameras)}대라 셀 낱말이 없습니다"
    english, korean = COUNT_WORDS[len(cameras)]
    head, _, template = CARD_GLANCE[language]
    phrase = template.format(english.capitalize() if language == "en" else korean)
    assert phrase in card_section(head), (
        f"정본의 카메라는 {len(cameras)}대인데 카드({language})의 「한눈에」에 「{phrase}」가 없습니다")


# 카드의 「스텝 열 — 로우와 가공」 — (절의 머리, {표의 첫 칸: 정본의 층})
CARD_LAYER_SECTIONS = {
    "en": ("### Step columns, raw and derived\n", {"Raw": "raw", "Derived": "processed"}),
    "ko": ("### 스텝 열 — 로우와 가공\n", {"로우": "raw", "가공": "processed"}),
}


@pytest.mark.parametrize("language", sorted(CARD_LAYER_SECTIONS))
def test_the_card_layer_table_matches_canon(language):
    """카드(영 · 국)의 층 표 ↔ `safety_schema.json` `step_field_layers` — 배포본에 실리는 열로 좁혀, 양쪽으로.

    좁히는 것은 변환기가 싣는 열에서 유도한다 — `generation.field_map.STEP_FIELDS` 의 열 전부(조건 없는 열과
    조건부 열). 선택 열도 시뮬레이터가 기록하는 것은 실리고, 기록하지 않는 열(`observation.contact.pressure` —
    접촉 면적을 실측할 때만)은 대응표에 없으므로 저절로 빠진다. 한 열이 두 번 나오거나 두 층에 나와도 거부한다.
    """
    from generation.field_map import STEP_FIELDS

    schema = canon("safety_schema.json")
    layers = schema["step_field_layers"]
    carried = {field.schema for field in STEP_FIELDS}
    assert carried <= set(layers), f"층이 없는 열: {sorted(carried - set(layers))}"
    want = {name: layer for name, layer in layers.items() if name in carried}

    head, words = CARD_LAYER_SECTIONS[language]
    rows = re.findall(r"^\| (%s) \|[^|]*\|(.*)\|$" % "|".join(map(re.escape, words)), card_section(head), re.M)
    assert sorted(word for word, _ in rows) == sorted(words), (
        f"카드({language})의 층 표를 파싱하지 못했습니다 — 층마다 한 행이어야 합니다: {[w for w, _ in rows]}")
    drawn = {}
    for word, cells in rows:
        for name in re.findall(r"`([^`]+)`", cells):
            assert name not in drawn, f"카드({language})의 층 표에 `{name}` 이 두 번 나옵니다"
            drawn[name] = words[word]
    assert drawn == want, (
        f"카드({language})의 층 표가 정본과 다릅니다 — 문서만: {sorted(set(drawn) - set(want))}, "
        f"정본만: {sorted(set(want) - set(drawn))}, "
        f"층이 다름: {sorted(k for k in set(drawn) & set(want) if drawn[k] != want[k])}")


def card_release_rules() -> dict:
    """카드가 말하는 배포본의 `rules.json` — 정본을 카드의 「한눈에」가 말하는 시나리오(`card_scenarios`)로 좁힌 것.

    변환기 · 조립이 배포본에 싣는 것과 같은 함수(`generation.release_canon.release_canon`)로 좁힌다. 카드가 「이 배포본은
    무엇을 싣는가」를 말하는 자리는 이 문서와 대조해야 한다 — 정본 전체로 세면 빠진 규칙의 항목까지 세어, 배포본에 없는
    것을 카드가 말하라고 요구하게 된다.
    """
    from generation.release_canon import release_canon

    return release_canon(canon("scenarios.json"), canon("rules.json"), card_scenarios())[1]


# 카드가 실린 표준 임계값의 채움 상태를 말하는 문장 — 언어마다 한 번
CARD_FILL_STATE = {
    "en": r"Of the (\d+) thresholds the standard sets, (\d+) carry only a citation",
    "ko": r"표준이 정한 임계값 (\d+)개 가운데 조항 참조만 실린 것은 (\d+)개",
}


def card_fill_state(root: dict) -> tuple[int, int]:
    """카드의 채움 상태 문장이 말해야 할 (N, M) — 「표준이 정한 임계값 N개 가운데 조항 참조만 실린 것은 M개」.

    N 은 표준이 정한 임계값 전부, 곧 값이 실린 것과 조항 참조만인 것의 합이고 M 은 그 가운데 조항 참조만인 것이다
    (`standard_thresholds` 로 센다). `not_applicable` 인 칸은 표준이 값을 정하지 않은 칸이라 N 에 들지 않는다 — 카드는
    그 칸을 같은 문단에서 따로 말한다. 값이 실린 수만 N 으로 쓰면, 항목 하나가 조항 참조만으로 바뀔 때 N 이 함께 줄어
    「N개 가운데 M개」가 표준이 정한 수를 말하지 않게 된다.
    """
    counts = standard_thresholds(root=root)
    return counts["value"] + counts["reference_only"], counts["reference_only"]


@pytest.mark.parametrize("language", sorted(CARD_FILL_STATE))
def test_the_card_states_the_fill_state_of_what_it_ships(language):
    """카드(영 · 국)의 「표준이 정한 임계값 N개 가운데 조항 참조만 실린 것은 M개」 ↔ 배포본이 싣는 `rules.json`
    (`card_release_rules`)을 `card_fill_state` 로 센 값 — N 은 값이 실린 것과 조항 참조만인 것의 합, M 은 조항 참조만.

    카드는 배포본의 README 이므로 정본 전체가 아니라 실린 문서를 센다 — 정본에서만 조항 참조인 항목(빠진 규칙의
    파라미터)은 받은 쪽에게 없다. 문장을 찾지 못하거나 두 번 찾으면 실패한다(`only`).
    """
    said = only(CARD_FILL_STATE[language], read(CARD), f"카드({language}) · 채움 상태")
    got = (int(said.group(1)), int(said.group(2)))
    want = card_fill_state(card_release_rules())
    assert got == want, (f"카드({language})는 표준 임계값 {got[0]}개 · 조항 참조만 {got[1]}개라 하고, 배포본이 싣는 "
                         f"rules.json 은 표준 임계값 {want[0]}개 · 조항 참조만 {want[1]}개입니다")


# ─────────────────────────────────────────────────────────────────────
# The fill state said again outside §9.3
# ─────────────────────────────────────────────────────────────────────
#
# §9.3, the inventory's two sentences and the card's one are where the fill state is said and counted above. Other
# sentences say it again — how many contact limits carry a value, which cells are not applicable, which entry is
# still a clause reference only, how many rules the release ships — and grow stale the same way when the canon
# fills or empties an entry. The tables below pin each of them to a count derived from the canon or from what the
# card's release ships (`card_release_rules`). A sentence that is reworded is not found, and that fails.

VALIDATOR_README = REPO / "validator/README.md"
GENERATION_README = REPO / "generation/README.md"
CONTRIBUTING = REPO / "CONTRIBUTING.md"

# Words the documents use in place of a numeral.
COUNT_TOKENS = {"하나": 1, "한": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "여섯": 6, "열네": 14,
                "one": 1, "two": 2, "three": 3, "six": 6, "fourteen": 14, "twenty": 20}


def count_token(token: str) -> int:
    if token.isdigit():
        return int(token)
    assert token in COUNT_TOKENS, f"수를 말하는 낱말 「{token}」 을 모릅니다 — COUNT_TOKENS 에 더하십시오"
    return COUNT_TOKENS[token]


def fill_state_facts() -> dict[str, int]:
    """What a sentence about the fill state can count, from the canon and from the card's release.

    `limit …` is the canon's `contact_limits`, `shipped …` the `rules.json` the card's release carries.
    """
    shipped = card_release_rules()
    limits = standard_thresholds("contact_limits")
    shipped_limits = standard_thresholds("contact_limits", root=shipped)
    shipped_whole = standard_thresholds(root=shipped)

    def valued(*where: str) -> int:
        """Regions whose entry at ``where`` (keys below the region — where the labeler and the builder read it)
        carries a value."""
        count = 0
        for region in canon("rules.json")["contact_limits"].values():
            entry = region
            for key in where:
                entry = entry.get(key) if isinstance(entry, dict) else None
            count += isinstance(entry, dict) and entry.get("value") is not None
        return count

    force, multipliers = valued("limits", "force_quasistatic"), valued("transient_multiplier")
    return {
        "limit values": limits["value"],
        "force limit values": force,
        "multiplier values": multipliers,
        "force and multiplier values": force + multipliers,
        "other limit values": limits["value"] - force - multipliers,
        "pressure limit values": valued("limits", "pressure_quasistatic"),
        "biomechanical values": valued("biomechanical", "effective_mass") + valued("biomechanical", "spring_constant"),
        "limit cells not applicable": limits["not_applicable"],
        "shipped limit values": shipped_limits["value"],
        "shipped limit cells not applicable": shipped_limits["not_applicable"],
        "reference only": standard_thresholds()["reference_only"],
        "shipped values": shipped_whole["value"],
        "shipped reference only": shipped_whole["reference_only"],
        "rules": len(canon("rules.json")["rules"]),
        "shipped rules": len(shipped["rules"]),
        "excluded rules": len(shipped["excluded_rules"]["rule_ids"]),
    }


def reference_only_entries(root: dict | None = None) -> list[tuple[str, ...]]:
    """The path of every entry `standard_thresholds` counts as `reference_only` — a rule is named by its `rule_id`,
    so a rule parameter reads `("rules", <rule_id>, "parameters", <name>)`."""
    def walk(node, path):
        if isinstance(node, dict):
            if (node.get("determined_by") == "standard" and not node.get("not_applicable")
                    and node.get("value") is None):
                yield path
            for key, value in node.items():
                yield from walk(value, path + (key,))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                step = value.get("rule_id", str(i)) if isinstance(value, dict) else str(i)
                yield from walk(value, path + (step,))

    return list(walk(canon("rules.json") if root is None else root, ()))


def said_text(source) -> tuple[str, str]:
    """(where, text) — a document read paragraph by paragraph with its folded lines joined, or a canon note."""
    if isinstance(source, Path):
        return str(source.relative_to(REPO)), "\n\n".join(body for _, body in paragraphs(read(source)))
    name, key = source
    return f"meta/{name}#{key}", canon(name)[key]


def check_fill_state_said(source, pattern: str, groups: tuple[str, ...], implied: dict[str, int],
                          times: int) -> None:
    """``pattern`` is found exactly ``times`` times in ``source``. Each group is a count that must equal the fact it
    is named for (`a&b` — both facts), and ``implied`` holds the facts the sentence states without a numeral."""
    where, text = said_text(source)
    found = list(re.finditer(pattern, text))
    assert len(found) == times, (
        f"{where}: 대조할 문장을 {len(found)}건 찾았습니다 ({times}건이어야 합니다).\n  패턴: {pattern}\n"
        "  문장을 고쳐 썼다면 이 시험의 패턴도 함께 고치십시오")
    facts = fill_state_facts()
    for match in found:
        assert len(match.groups()) == len(groups), f"{where}: 패턴의 묶음 {len(match.groups())}개 · 이름 {len(groups)}개"
        for i in range(1, len(groups) + 1):
            # A count word matched at the tail of a longer one (「두」 in 「열두」, 「two」 in 「twenty-two」) is a
            # different number — the count must start a word.
            # 「1,020」 · 「26.20」 are other numbers too.
            before = text[match.start(i) - 1:match.start(i)]
            assert not re.match(r"[\w.,-]", before), (
                f"{where}: 「{text[max(0, match.start(i) - 12):match.end(i)]}」 — 수가 낱말의 중간에서 잡혔습니다")
        for token, names in zip(match.groups(), groups):
            for name in names.split("&"):
                assert count_token(token) == facts[name], (
                    f"{where}: 「{match.group(0)[:90]}」 는 {token} 이라 하고 정본의 {name} 은 {facts[name]} 입니다")
        for name, value in implied.items():
            assert facts[name] == value, (
                f"{where}: 「{match.group(0)[:90]}」 는 {name} 이 {value} 라 하고 정본은 {facts[name]} 입니다")


def check_reference_only_named(source, pattern: str, times: int, rule_shipped: bool | None) -> None:
    """The sentence names the one entry that is still a clause reference only — group `name` is its parameter and
    group `rule` the rule it belongs to. ``rule_shipped`` is what the sentence says of that rule in the card's
    release (`None` — it says nothing)."""
    where, text = said_text(source)
    found = list(re.finditer(pattern, text))
    assert len(found) == times, (
        f"{where}: 대조할 문장을 {len(found)}건 찾았습니다 ({times}건이어야 합니다).\n  패턴: {pattern}")
    entries = reference_only_entries()
    assert len(entries) == 1, f"정본에서 조항 참조만 실린 항목이 하나가 아닙니다: {entries} — {where} 의 문장을 고치십시오"
    entry = entries[0]
    excluded = card_release_rules()["excluded_rules"]["rule_ids"]
    for match in found:
        said = match.groupdict()
        if said.get("name") is not None:
            assert said["name"] == entry[-1], f"{where}: {said['name']} 라 하고 정본에서는 {'.'.join(entry)} 입니다"
        if said.get("rule") is not None:
            assert entry[:2] == ("rules", said["rule"]), (
                f"{where}: {said['rule']} 의 것이라 하고 정본에서는 {'.'.join(entry)} 입니다")
    if rule_shipped is not None:
        assert entry[0] == "rules", f"{where}: 규칙을 말하는데 정본의 항목은 규칙의 것이 아닙니다: {'.'.join(entry)}"
        assert (entry[1] not in excluded) == rule_shipped, (
            f"{where}: {entry[1]} 를 배포본이 {'싣는다' if rule_shipped else '싣지 않는다'}고 하는데 배포본의 "
            f"excluded_rules 는 {excluded} 입니다")


# (where, the sentence, the facts its count groups say, the facts it says without a numeral, how many times it is said)
FILL_STATE_SAID_AGAIN = [
    (INVENTORY, r"판정에 쓰는 \*\*표준 임계값 (\d+)개\*\*", ("shipped values",), {}, 1),
    (INVENTORY, r"이 배포본의 `meta/rules\.json` 에는 조항 참조만 실린 임계값이 없고, 값이 실린 표준 임계값은 "
                r"위 「대상」의 (\d+)개가 전부입니다", ("shipped values&limit values",),
     {"shipped reference only": 0}, 1),
    (INVENTORY, r"위 표의 「대상」이 접촉 한계 (\d+)개인 것과 별개이며", ("limit values",), {}, 1),
    (SPEC, r"\*\*배포본은 이 (\d+)개를 전부 싣지 않습니다\.\*\*", ("rules",), {}, 1),
    (SPEC, r"(두) 부위의 `transient_multiplier` 는 수치가 아니라 `not_applicable`", ("limit cells not applicable",),
     {}, 1),
    (CARD, r"(두) 칸 — 두개골·이마와 얼굴의 동적 배수 — 은 위의 까닭으로 `not_applicable`",
     ("shipped limit cells not applicable",), {}, 1),
    (CARD, r"(two) further cells, the transient multipliers of the skull and forehead and of the face, are "
           r"`not_applicable`", ("shipped limit cells not applicable",), {}, 1),
    (VALIDATOR_README, r"조항 참조만 실린 임계값\(정본에서는 `\w+` (\d+)건\)", ("reference only",), {}, 1),
    (VALIDATOR_README, r"26\.11 배포본은 그 규칙을 싣지 않아 WARN 0", (), {"shipped reference only": 0}, 1),
    (VALIDATOR_README, r"표준이 상수로 주지 않는 임계값 (\d+)건\(`\w+`\)이 정본에 조항 참조로만 있고",
     ("reference only",), {}, 1),
    (CONTRIBUTING, r"`meta/rules\.json` `contact_limits`\(값 (\d+)개\)", ("limit values",), {}, 1),
    (("rules.json", "contact_limits_note"), r"(\d+)개 항목이 KS B ISO TS 15066", ("limit values",), {}, 1),
    (CARD, r"Release 26\.11 ships (fourteen): the twelve contact-limit rules", ("shipped rules",), {}, 1),
    (CARD, r"The other (six) belong to scenarios", ("excluded rules",), {}, 1),
    (CARD, r"The full (twenty)-rule catalogue", ("rules",), {}, 1),
    (CARD, r"26\.11 배포본은 (열네) 개를 싣습니다", ("shipped rules",), {}, 1),
    (CARD, r"나머지 (여섯)은 여기 실린 어떤 에피소드도", ("excluded rules",), {}, 1),
    (CARD, r"(\d+)개 전체 목록은 데이터 스키마 정의서", ("rules",), {}, 1),
    (SPEC, r"`meta/rules\.json` 의 압력 한계 (\d+)개에는 값이 실려 있습니다", ("pressure limit values",), {}, 1),
    (GENERATION_README, r"유효질량 · 스프링상수 (\d+)건\(표 A\.3\)은 `meta/rules\.json` 의 "
                        r"`contact_limits\.\*\.biomechanical` 에 값과 조항 참조가 함께 실려", ("biomechanical values",),
     {}, 1),
    (INVENTORY, r"v0\.1 라벨이 실제로 계산에 쓰는 것은 (\d+)개\*\*입니다 — 부위별 준정적 힘 (\d+)개와 적용 가능한 동적 "
                r"배수 (\d+)개\. 나머지 (\d+)개\(압력·전달 에너지·유효질량·탄성계수\)",
     ("force and multiplier values", "force limit values", "multiplier values", "other limit values"), {}, 1),
]

def not_applicable_cells() -> list[tuple[str, ...]]:
    """The path of every `not_applicable` entry under the canon's `contact_limits`."""
    return [("contact_limits", region, field) for region, body in canon("rules.json")["contact_limits"].items()
            if isinstance(body, dict)
            for field, entry in [*body.get("limits", {}).items(), *body.items()]
            if isinstance(entry, dict) and entry.get("not_applicable")]


def check_not_applicable_named(source, pattern: str, times: int) -> None:
    """The sentence names the cells that are `not_applicable` — group `cells` names each of them by its region's code
    or display name (`name_ko` · `name_en`), and they are all transient multipliers, which is what every such
    sentence calls them. The count is pinned by `check_fill_state_said`; this pins which cells."""
    where, text = said_text(source)
    found = list(re.finditer(pattern, text))
    assert len(found) == times, (
        f"{where}: 대조할 문장을 {len(found)}건 찾았습니다 ({times}건이어야 합니다).\n  패턴: {pattern}")
    cells = not_applicable_cells()
    assert cells and all(cell[-1] == "transient_multiplier" for cell in cells), (
        f"정본의 해당 없음 칸이 동적 배수만이 아닙니다: {cells} — {where} 의 문장을 고치십시오")
    regions = canon("rules.json")["contact_limits"]

    def flat(s: str) -> str:
        return re.sub(r"\s+", "", s).replace("/", "·").lower()

    for match in found:
        named = flat(match.group("cells"))
        # Containment only shows each cell is named. Where the sentence names regions by code, the codes are exactly
        # the cells' — a display name can sit inside other words, a code cannot.
        codes = set(re.findall(r"BR-\d\d", match.group("cells")))
        assert not codes or codes == {region for _, region, _ in cells}, (
            f"{where}: 「{match.group('cells')}」 의 부위 {sorted(codes)} ↔ 정본의 해당 없음 칸 {[c[1] for c in cells]}")
        for _, region, _ in cells:
            forms = {region, regions[region].get("name_ko", ""), regions[region].get("name_en", "")} - {""}
            assert any(flat(form) in named for form in forms), (
                f"{where}: 「{match.group('cells')}」 이 정본의 해당 없음 칸 {region}({'/'.join(sorted(forms))}) 을 "
                "말하지 않습니다")


# (where, the sentence naming the entry that is still a clause reference only, how many times it is said,
#  what it says of that entry's rule in the card's release)
REFERENCE_ONLY_NAMED = [
    (CONTRIBUTING, r"\*\*`(?P<name>\w+)` 만 조항 참조뿐입니다\*\*", 1, None),
    (VALIDATOR_README, r"정본에서는 `(?P<name>\w+)` \d+건\)을 싣는 배포본에서는", 1, None),
    (VALIDATOR_README, r"임계값 \d+건\(`(?P<name>\w+)`\)이 정본에 조항 참조로만 있고, 그 규칙\(`(?P<rule>[\w-]+)`\)을 "
                       r"싣는 배포본에서는 그것이 WARN 을 만들므로 `--strict` 는 정상 배포본에서도 실패합니다\(26\.11 "
                       r"배포본은 그 규칙을 싣지 않습니다\)", 1, False),
    (INVENTORY, r"그 임계값은 `(?P<rule>[\w-]+)` 의 것이고, 26\.11 배포본은 그 규칙을 싣지 않습니다", 1, False),
    (SPEC, r"\*\*아직 값 없이 조항 참조만 실린 항목이 1개 있습니다\*\* — `(?P<name>\w+)` 입니다", 1, None),
    (CARD, r"`(?P<rule>[\w-]+)` — its `(?P<name>\w+)` — has no value in the code repository's `meta/rules\.json`\. "
           r"That rule is not in this release", 1, False),
    (CARD, r"`(?P<rule>[\w-]+)` 의 파라미터 하나\(`(?P<name>\w+)`\)가 코드 저장소의 `meta/rules\.json` 에서 값이 없다는 "
           r"뜻입니다\. 그 규칙은 이 배포본에 없고", 1, False),
]

# (where, the sentence naming the `not_applicable` cells, how many times it is said)
NOT_APPLICABLE_NAMED = [
    (SPEC, r"(?P<cells>`BR-\d\d`\([^)]*\)·`BR-\d\d`\([^)]*\))은 표 A\.2 의 동적 칸이", 1),
    (CARD, r"두 칸 — (?P<cells>[^—]+)의 동적 배수 — 은 위의 까닭으로 `not_applicable`", 1),
    (CARD, r"the transient multipliers (?P<cells>.+?), are `not_applicable`", 1),
]


@pytest.mark.parametrize("source,pattern,groups,implied,times", FILL_STATE_SAID_AGAIN,
                         ids=[f"{said_text(s)[0]}-{i}" for i, (s, *_) in enumerate(FILL_STATE_SAID_AGAIN)])
def test_the_fill_state_said_again_matches_canon(source, pattern, groups, implied, times):
    """Each sentence that says the fill state again — a count, or a fact without a numeral — against the canon or
    the card's release (`fill_state_facts`)."""
    check_fill_state_said(source, pattern, groups, implied, times)


@pytest.mark.parametrize("source,pattern,times,rule_shipped", REFERENCE_ONLY_NAMED,
                         ids=[f"{said_text(s)[0]}-{i}" for i, (s, *_) in enumerate(REFERENCE_ONLY_NAMED)])
def test_the_entry_named_as_reference_only_is_the_canons(source, pattern, times, rule_shipped):
    """A sentence that names the entry still carrying only a clause reference names the canon's one entry, and
    says of its rule what the card's release does."""
    check_reference_only_named(source, pattern, times, rule_shipped)


@pytest.mark.parametrize("source,pattern,times", NOT_APPLICABLE_NAMED,
                         ids=[f"{said_text(s)[0]}-{i}" for i, (s, *_) in enumerate(NOT_APPLICABLE_NAMED)])
def test_the_cells_named_as_not_applicable_are_the_canons(source, pattern, times):
    """A sentence that names the `not_applicable` cells names each of the canon's."""
    check_not_applicable_named(source, pattern, times)


# 카드의 「자체 설정 값」 절의 머리
CARD_SELF_SET_SECTIONS = {"en": "### Values this project set\n", "ko": "### 자체 설정 값\n"}
# 그 절의 표 — 첫 칸(뜻) · 둘째 칸의 백틱 키 · 셋째 칸의 수로 시작하는 값
SELF_SET_ROW = re.compile(r"^\| [^|]* \| `(\w+)` \| (-?[\d.]+)", re.M)


def self_set_values() -> dict[str, float]:
    """카드가 「자체 설정 값」 표로 보일 값 — 배포본이 싣는 규칙의 `parameters` 가운데 `determined_by: project` 인 항목
    전부와, 「봄」의 두 문턱(`meta/cameras.json` `seeing` 의 `min_visible_fraction` · `min_height_px`).

    같은 이름이 두 자리에서 나오면 표 한 행으로 어느 값을 말하는지 정할 수 없으므로 실패한다.
    """
    out: dict[str, float] = {}

    def put(name: str, value, where: str) -> None:
        assert name not in out, f"자체 설정 값의 이름 {name!r} 이 두 자리에 있습니다 — 카드의 표 한 행이 어느 값인지 정할 수 없습니다"
        assert isinstance(value, (int, float)) and not isinstance(value, bool), f"{where} 의 값이 수가 아닙니다: {value!r}"
        out[name] = float(value)

    for rule in card_release_rules()["rules"]:
        for name, entry in (rule.get("parameters") or {}).items():
            if isinstance(entry, dict) and entry.get("determined_by") == "project":
                put(name, entry.get("value"), f"{rule['rule_id']}.parameters.{name}")
    seeing = canon("cameras.json")["seeing"]
    assert seeing.get("determined_by") == "project", "「봄」의 문턱이 자체 설정이 아닙니다 — 카드의 「자체 설정 값」과 함께 고치십시오"
    for name in ("min_visible_fraction", "min_height_px"):
        put(name, seeing[name], f"cameras.json seeing.{name}")
    return out


@pytest.mark.parametrize("language", sorted(CARD_SELF_SET_SECTIONS))
def test_the_card_self_set_table_matches_canon(language):
    """카드(영 · 국)의 「자체 설정 값」 표 ↔ 배포본이 싣는 규칙의 project 파라미터와 「봄」의 두 문턱 — 이름과 값, 양쪽으로.

    표에만 있는 행 · 정본에만 있는 값 · 값이 다른 행 · 두 번 나온 행을 모두 거부한다. 못 보는 것: 첫 칸(뜻) · 넷째
    칸(근거)과 표 밖의 산문(정지 시험 표 · 조건표의 창 · 목표 비율 같은 자체 설정은 산문으로 적는다).
    """
    rows = SELF_SET_ROW.findall(card_section(CARD_SELF_SET_SECTIONS[language]))
    names = [name for name, _ in rows]
    twice = sorted({name for name in names if names.count(name) > 1})
    assert not twice, f"카드({language})의 「자체 설정 값」 표에 같은 행이 두 번 나옵니다: {twice}"
    said = {name: float(value) for name, value in rows}
    want = self_set_values()
    assert said == want, (f"카드({language})의 「자체 설정 값」 표가 정본과 다릅니다 — 문서만: "
                          f"{sorted(set(said) - set(want))}, 정본만: {sorted(set(want) - set(said))}, "
                          f"값이 다름: {sorted(k for k in set(said) & set(want) if said[k] != want[k])}")


def test_the_card_has_the_same_sections_in_both_languages():
    """카드는 국·영 두 벌이며 절 구성이 어긋나면 한쪽 독자만 못 읽는 내용이 생긴다.

    **국문에 「인용」 절이 빠진 적이 있다**(영문 7 · 국문 6). 인용 절이 한쪽에만
    있으면 다른 쪽 독자는 이 데이터셋을 인용하는 법을 모른다.
    「국문은 고쳤는데 영문만 반대로 남는」 형태의 거울이다.

    **제목은 언어가 달라 짝지을 수 없으므로 개수만 본다.** 어느 절이 빠졌는지는
    사람이 봐야 하지만, 빠졌다는 사실은 여기서 걸린다.
    """
    text = read(CARD)
    split = text.index(CARD_KOREAN_TITLE)
    # `^## ` 는 `### ` 의 앞 두 글자에도 맞는다 — 하위 절까지 세면 무엇을 비교하는지
    # 흐려진다. 최상위 절만 본다.
    english = re.findall(r"^## ([^#].*)$", text[:split], re.M)
    korean = re.findall(r"^## ([^#].*)$", text[split:], re.M)

    assert english and korean, "카드에서 절 제목을 찾지 못했습니다 — 구조가 바뀌었습니까?"
    assert len(english) == len(korean), (
        f"카드 두 벌의 절 수가 다릅니다 — 영문 {len(english)}개 {english}, "
        f"국문 {len(korean)}개 {korean}. 한쪽에만 있는 내용은 그 언어 독자만 읽습니다")


def test_the_card_tree_names_every_bundled_canon_file():
    """카드의 구성 트리는 배포본에 동봉되는 정본 파일을 모두 그린다.

    `meta/assets.json` 은 자산 경로·체크섬·라이선스를 담은 파일이고 배포본에는 있다 —
    라이선스 검토자가 `NOTICE` 다음으로 볼 파일이다. 트리에서 빠지면 받은 쪽이 그
    파일을 찾을 길잡이가 없다.
    """
    tree = re.search(r"## What is in it\n\n```\n(.*?)```", read(CARD), re.S)
    assert tree, "카드의 구성 트리를 찾지 못했습니다"
    drawn = tree.group(1)
    missing = sorted(p for p in bundled_paths() if p.rsplit("/", 1)[-1] not in drawn)
    assert not missing, (
        f"변환기가 동봉하는데 카드 트리가 그리지 않는 파일: {missing}")


def test_both_trees_count_their_own_extension_marks():
    """「ext 여섯」·「확장 표시 여섯」이 트리에 실제로 찍힌 표시 수와 같은가.

    카드에서 `meta/assets.json` 이 빠져 있는 동안 **국·영 두 벌 모두 「다섯」**이라고
    적고 있었다 — 빠진 줄을 세지 않았으니 산문도 따라 틀렸다. 수를 트리에서 유도하면
    줄이 늘거나 줄 때 산문이 따라오지 않는 것을 잡는다.
    """
    card = read(CARD)
    tree = re.search(r"## What is in it\n\n```\n(.*?)```", card, re.S).group(1)
    marks = len(re.findall(r"\bext\b", tree))
    assert marks in COUNT_WORDS, f"트리의 ext 표시가 {marks}개라 셀 낱말이 없습니다"
    english, korean = COUNT_WORDS[marks]
    assert f"The {english} `ext` entries" in card, (
        f"트리에 ext 표시가 {marks}개인데 영문 산문이 그렇게 말하지 않습니다")
    assert f"`ext` 표시 **{korean}**" in card, (
        f"트리에 ext 표시가 {marks}개인데 국문 산문이 그렇게 말하지 않습니다")


def test_the_spec_counts_joints_from_the_joint_maps():
    """규격서 §7.3 의 DOF 합계와 §7.3.1 의 「이 데이터셋이 쓰는 H1 사본」 문장 ↔ `meta/joint_map/`.

    §7.3.1 은 H1 사본 둘을 말한다 — 이 데이터셋이 쓰는 사본(매핑 파일의 `provenance.usd_path`)은 관절 19개에
    고정관절이 없고, 쓰지 않는 `Robots/Unitree` 사본은 관절 24개 가운데 5개가 고정관절이다(자산 라이선스 인벤토리
    §2.2). 두 사본의 경로가 하나는 다른 하나의 접미사라 섞어 적기 쉬우므로, 쓰는 사본의 문장 — 경로 · 관절 수 ·
    「모두 DOF」 — 을 매핑 파일에서 유도해 대조한다. 못 보는 것: 다른 사본에 대한 문장(인벤토리의 실측을 옮긴 것이라
    정본이 없음)과 절의 나머지 산문.
    """
    import yaml

    def joint_map(model: str) -> dict:
        return yaml.safe_load((REPO / "meta/joint_map" / f"{model}.yaml").read_text(encoding="utf-8"))

    h1, g1 = joint_map("unitree_h1"), joint_map("unitree_g1_29dof")
    spec = read(SPEC)

    total = only(r"\| 합계 \| \*\*(\d+)\*\* \| \*\*(\d+)\*\* \|", spec, "규격서 §7.3 합계")
    assert (int(total.group(1)), int(total.group(2))) == (len(h1["joint_names"]), len(g1["joint_names"])), (
        f"§7.3 합계 {total.group(1)} · {total.group(2)} ↔ 매핑 파일의 DOF "
        f"{len(h1['joint_names'])} · {len(g1['joint_names'])}")

    section = spec.split("### 7.3.1 ", 1)[1].split("\n### ", 1)[0]
    used = only(r"이 데이터셋이 쓰는 H1 사본\(`([^`]+)`\)은 관절 (\d+)개가 모두 DOF", " ".join(section.split()),
                "규격서 §7.3.1 의 H1 사본 문장")
    detail = h1["articulation_detail"]
    # Anchored at the Isaac asset root: the other copy's path is a suffix of this one's without it.
    assert h1["provenance"]["usd_path"].endswith("/Isaac/" + used.group(1)), (
        f"§7.3.1 이 말하는 사본 {used.group(1)} ↔ 매핑 파일의 자산 {h1['provenance']['usd_path']}")
    assert int(used.group(2)) == detail["num_joints"] == len(h1["joint_names"]), (
        f"§7.3.1 의 관절 {used.group(2)}개 ↔ 매핑 파일의 관절 {detail['num_joints']}개 · DOF {len(h1['joint_names'])}개")
    assert detail["non_dof_joints"] == [], (
        f"매핑 파일에 고정관절이 있습니다({detail['non_dof_joints']}) — §7.3.1 의 「모두 DOF」 문장을 고치십시오")


APPEARANCE_ROW = re.compile(r"^\| `(\w+)` \| ([^|]+?) \| [^|]+ \|$", re.M)
AS_BUILT_WORD = {"en": "as built", "ko": "지은 그대로"}


def appearance_rows(language: str) -> set[tuple[str, str]]:
    """겉모습 표가 보여야 하는 (축, 고르는 값) — 정본 `appearance.axes` 에서 만든다.

    색 축은 팔레트 이름을 백틱으로 ` · ` 로 이은 것(수치 `null` — 지은 그대로 — 인 이름 뒤에 「(지은 그대로)」 / 「(as built)」),
    범위 축은 「하한 ~ 상한」(`:g`). 세 시나리오의 블록이 같아야 한다.
    """
    by_id = {s["scenario_id"]: s for s in canon("scenarios.json")["scenarios"]}
    blocks = [by_id[sid]["appearance"] for sid in card_scenarios()]
    assert all(b == blocks[0] for b in blocks), "카드가 말하는 시나리오의 `appearance` 블록이 서로 다릅니다"
    want = set()
    for axis, spec in blocks[0]["axes"].items():
        if "range" in spec:
            want.add((axis, f"{spec['range'][0]:g} ~ {spec['range'][1]:g}"))
        else:
            names = [f"`{n}`" + (f" ({AS_BUILT_WORD[language]})" if v is None else "")
                     for n, v in spec["choice"].items()]
            want.add((axis, " · ".join(names)))
    return want


@pytest.mark.parametrize("language", sorted(CARD_RANDOMIZATION_SECTIONS))
def test_the_card_appearance_tables_match_canon(language):
    """카드(영 · 국)의 겉모습 표 ↔ 정본 `appearance.axes` — 축과 고르는 값(팔레트 이름 · 범위), 양쪽으로, 중복 없이.

    못 보는 것: 셋째 칸(입히는 곳)과 표 밖의 산문, 색의 수치(표에 적지 않는다).
    """
    start, end = CARD_RANDOMIZATION_SECTIONS[language]
    body = card_section(start, (end,))
    lines = body.splitlines()
    assert sum(1 for l in lines if l.strip() == APPEARANCE_HEADS[language]) == 1, (
        f"카드({language})에 겉모습 표 머리 「{APPEARANCE_HEADS[language]}」 가 정확히 하나여야 합니다")
    head = lines.index(APPEARANCE_HEADS[language])
    table = []
    for line in lines[head + 2:]:
        if not line.startswith("|"):
            break
        table.append(line)
    rows = [(m.group(1), m.group(2)) for m in (APPEARANCE_ROW.match(l) for l in table)]
    assert len(rows) == len(table), f"카드({language}) 겉모습 표에 읽지 못한 줄이 있습니다"
    twice = sorted({r for r in rows if rows.count(r) > 1})
    assert not twice, f"카드({language}) 겉모습 표에 같은 행이 두 번 나옵니다: {twice}"
    said, want = set(rows), appearance_rows(language)
    assert said == want, (f"카드({language}) 겉모습 표가 정본과 다릅니다 — 문서만: {sorted(said - want)}, "
                          f"정본만: {sorted(want - said)}")


@pytest.mark.parametrize("language", sorted(CARD_RANDOMIZATION_SECTIONS))
def test_the_appearance_table_is_not_read_as_a_conditions_or_randomization_table(language):
    """겉모습 표가 카드 절에 있어도 조건표 행(`CONDITION_ROW`)으로도 랜덤화 표로도 읽히지 않는다 — 축이 첫 칸이다."""
    start, end = CARD_RANDOMIZATION_SECTIONS[language]
    body = card_section(start, (end,))
    axes = {a for a, _ in appearance_rows(language)}
    assert not [r for r in CONDITION_ROW.findall(body) if r[1] in axes]
    assert not (set(randomization_tables(body, f"카드({language})")) - set(_RANDOMIZATION_BLOCKS))


def test_the_catalog_names_the_appearance_block():
    """카탈로그가 정본의 `appearance` 블록과 그 축 이름을 모두 부른다 — 축을 더하고 카탈로그를 두면 빨강."""
    text = read(CATALOG)
    assert "`appearance`" in text, "카탈로그가 `appearance` 블록을 부르지 않습니다"
    by_id = {s["scenario_id"]: s for s in canon("scenarios.json")["scenarios"]}
    for sid in card_scenarios():
        missing = [a for a in by_id[sid]["appearance"]["axes"] if f"`{a}`" not in text]
        assert not missing, f"카탈로그가 {sid} 의 겉모습 축을 부르지 않습니다: {missing}"


# 카드의 「조건표 시나리오의 에피소드 열」 — (절의 머리, 열 표의 머리, 멤버 표의 머리)
CARD_EPISODE_COLUMN_SECTIONS = {
    "en": ("### Columns of an episode in a table scenario\n", "| Column | Meaning |", "| Member | What it holds |"),
    "ko": ("### 조건표 시나리오의 에피소드 열\n", "| 열 | 뜻 |", "| 멤버 | 담는 것 |"),
}
REGION_RATIO_ROW = "evidence.force_ratio.<%s>"
CARD_REGION_WORD = {"en": "region code", "ko": "부위 코드"}


def card_table(body: str, head: str, where: str) -> list[list[str]]:
    """``body`` 에서 머리가 ``head`` 인 표의 행 — 칸 목록. 그 머리는 정확히 하나여야 하고 표는 빈 줄까지."""
    lines = body.splitlines()
    assert sum(1 for l in lines if l.strip() == head) == 1, f"{where}: 표 머리 「{head}」 가 정확히 하나여야 합니다"
    rows = []
    for line in lines[lines.index(head) + 2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def names_in(cell: str) -> list[str]:
    return re.findall(r"`([^`]+)`", cell)


@pytest.mark.parametrize("language", sorted(CARD_EPISODE_COLUMN_SECTIONS))
def test_the_card_table_episode_columns_match_canon(language):
    """카드(영 · 국)의 「조건표 시나리오의 에피소드 열」 표 ↔ `scenario_conditional_episode_fields` — 열 이름을 양쪽으로, 중복 없이.

    부위 비율 열두 개는 한 줄 `evidence.force_ratio.<부위 코드>` 로 적고, 이 시험이 정본의 `body_regions` 코드로 풀어 대조한다.
    못 보는 것: 둘째 칸(뜻)의 산문.
    """
    start, columns_head, _ = CARD_EPISODE_COLUMN_SECTIONS[language]
    rows = card_table(card_section(start), columns_head, f"카드({language})")
    codes = list(canon("safety_schema.json")["body_regions"])
    ratio_row = REGION_RATIO_ROW % CARD_REGION_WORD[language]
    said = []
    for cells in rows:
        assert len(cells) == 2, f"카드({language}) 열 표에 읽지 못한 줄이 있습니다: {cells}"
        for name in names_in(cells[0]):
            said.extend(f"evidence.force_ratio.{code}" for code in codes) if name == ratio_row else said.append(name)
    twice = sorted({n for n in said if said.count(n) > 1})
    assert not twice, f"카드({language}) 열 표에 같은 열이 두 번 나옵니다: {twice}"
    want = canon("safety_schema.json")["scenario_conditional_episode_fields"]["conditions"]
    assert len(want) == len(set(want))
    assert set(said) == set(want), (f"카드({language}) 열 표가 정본과 다릅니다 — 문서만: {sorted(set(said) - set(want))}, "
                                    f"정본만: {sorted(set(want) - set(said))}")


@pytest.mark.parametrize("language", sorted(CARD_EPISODE_COLUMN_SECTIONS))
def test_the_card_spec_members_match_canon(language):
    """카드(영 · 국)의 `episode_spec` 멤버 표 ↔ `episode_spec_members` — 이름과 순서, 양쪽으로, 중복 없이."""
    start, _, members_head = CARD_EPISODE_COLUMN_SECTIONS[language]
    rows = card_table(card_section(start), members_head, f"카드({language})")
    said = [name for cells in rows for name in names_in(cells[0])]
    assert len(said) == len(rows), f"카드({language}) 멤버 표에 읽지 못한 줄이 있습니다"
    want = canon("safety_schema.json")["episode_spec_members"]
    assert said == want, f"카드({language}) 멤버 표가 정본과 다릅니다 — 문서: {said}, 정본: {want}"


@pytest.mark.parametrize("language", sorted(CARD_EPISODE_COLUMN_SECTIONS))
def test_the_card_names_the_optional_step_columns(language):
    """카드(영 · 국)의 층 표 아래 선택 열 문장이 정본의 `optional_step_fields` 를 모두, 그것만 부른다."""
    head = CARD_LAYER_SECTIONS[language][0]
    para = [text for _, text in paragraphs(card_section(head)) if "optional_step_fields" in text]
    assert len(para) == 1, f"카드({language})에 선택 열 문장이 {len(para)}개입니다 (1개여야 합니다)"
    want = canon("safety_schema.json")["optional_step_fields"]
    said = [n for n in names_in(para[0]) if n in set(canon("safety_schema.json")["step_field_layers"])]
    # 관절 상태 셋은 한 낱말 묶음(「관절 상태」 / 「the joint state」)으로 적는다 — 그 낱말이 있어야 셋을 센다
    joint_state = [f for f in want if f.startswith("observation.state.")]
    assert ("관절 상태" if language == "ko" else "the joint state") in para[0]
    said += joint_state
    assert sorted(said) == sorted(want), f"선택 열 문장이 정본과 다릅니다 — 문서: {sorted(said)}, 정본: {sorted(want)}"


@pytest.mark.parametrize("language", sorted(CARD_EPISODE_COLUMN_SECTIONS))
def test_the_episode_column_tables_are_not_read_as_conditions_randomization_or_appearance_tables(language):
    """새 표 둘은 조건표 · 랜덤화 · 겉모습 표를 읽는 절(`CARD_RANDOMIZATION_SECTIONS`) 밖에 있고, 조건표 행 꼴
    (첫 칸이 시나리오 id)이나 겉모습 표 머리가 아니다 — 이웃 시험이 새 표 때문에 다른 말을 하지 않는다."""
    start, _, _ = CARD_EPISODE_COLUMN_SECTIONS[language]
    body = card_section(start)
    varies, _ = CARD_RANDOMIZATION_SECTIONS[language]
    assert varies not in body and start not in card_section(varies, (CARD_RANDOMIZATION_SECTIONS[language][1],)), (
        f"카드({language})의 새 절이 「에피소드마다 달라지는 것」 절과 겹칩니다")
    assert not CONDITION_ROW.findall(body)
    assert not [l for l in body.splitlines() if is_appearance_table(l)]


# 카드의 목표 비율 표 — (절의 머리, 표의 머리). 표는 「Scenarios」 · 「시나리오」 절 안에 하나
CARD_TARGET_TABLES = {
    "en": ("## Scenarios\n", "| Scenario | `fail` | `near_miss` | `nominal` |"),
    "ko": ("## 시나리오\n", "| 시나리오 | `fail` | `near_miss` | `nominal` |"),
}


@pytest.mark.parametrize("language", sorted(CARD_TARGET_TABLES))
def test_the_card_target_table_matches_canon(language):
    """The card's target-mix table (en · ko, in *Scenarios*) ↔ `scenarios.json` `outcome_target`, per scenario.

    One row per scenario the card speaks of (`card_scenarios`), in that order and none twice; the header names every
    outcome of the canon's targets, and each cell is the canon's share as a whole percent (`round(share × 100)` and
    `%`). A target changed in the canon, a scenario added to or dropped from the card, a row left out or a share edited
    on the page is a red test. What it cannot see: the prose around the table -- that the labels do not follow the
    targets, that the mix counts the random rows only -- and the other places that name a target in words."""
    start, head = CARD_TARGET_TABLES[language]
    where = f"카드({language}) 「{start.strip()}」"
    rows = card_table(card_section(start, ("\n## ",)), head, where)
    outcomes = names_in(head)
    unread = [cells for cells in rows if len(cells) != 1 + len(outcomes) or len(names_in(cells[0])) != 1]
    assert rows and not unread, f"{where}: 목표 표에 읽지 못한 줄이 있습니다: {unread or '표가 비었습니다'}"
    said = [names_in(cells[0])[0] for cells in rows]
    twice = sorted({sid for sid in said if said.count(sid) > 1})
    assert not twice, f"{where}: 목표 표에 같은 시나리오가 두 번 나옵니다: {twice}"
    assert said == card_scenarios(), f"{where}: 목표 표의 시나리오 {said} ↔ 카드가 말하는 시나리오 {card_scenarios()}"
    by_id = {s["scenario_id"]: s for s in canon("scenarios.json")["scenarios"]}
    for sid, cells in zip(said, rows):
        target = by_id[sid]["outcome_target"]
        assert sorted(outcomes) == sorted(target), f"{where}: 표 머리의 결과 {outcomes} ↔ 정본 {sid} 의 {sorted(target)}"
        want = [f"{round(float(target[o]) * 100)}%" for o in outcomes]
        assert cells[1:] == want, f"{where}: {sid} 의 목표 {cells[1:]} ↔ 정본 {want}"


# The card's failure maps -- (the section's heading, the word for the person's axis, the word for the robot's axis).
# The words follow the two quantities in the header of each map table. The scorecard command's own output uses the
# Korean pair; the English half of the card writes the same header with the English pair.
CARD_MAP_SECTIONS = {
    "en": ("### Failure map\n", "down", "across"),
    "ko": ("### 실패 지도\n", "세로", "가로"),
}
# A cell of a map as the scorecard command prints it: "<outcome> <severity>", "degenerate", or – where no row is.
MAP_CELL = re.compile(r"(?:fail|near_miss|nominal) S[0-3]|degenerate|–")


@pytest.mark.parametrize("language", sorted(CARD_MAP_SECTIONS))
def test_the_card_failure_maps_are_the_canons_grids(language):
    """The card's failure-map tables (en · ko, in *Failure map*) ↔ `scenarios.json` `conditions.grid`, per scenario.

    For each scenario the card speaks of (`card_scenarios`) the section holds exactly one table whose header names the
    grid's person axis (down) and robot axis (across) and the robot axis's `cells` bins, each written as the scorecard
    command writes it (`generation.scorecard._edge` of the range cut into equal parts); the table has `cells` rows; the
    first cell of each row is a bin of the person axis, from the high end at the top; every other cell reads
    `<outcome> <severity>`, `degenerate` or `–`. The table belongs to its caption: the nearest non-blank line above
    its header names that scenario's id in backticks and no other scenario the card speaks of -- the header finds a
    table by its grid wherever it stands, so two captions exchanged would put one scenario's map under another's
    name while every header still matched. The section holds no other map table: the lines shaped like a map header (two backticked
    quantities followed by the language's two axis words) number exactly as many as the scenarios. A grid changed in
    the canon -- an axis renamed, a range moved, another number of cells -- or a table edited on the page (a bin, a
    row left out, the axes swapped, a caption exchanged with another map's or missing, a map left over or added) is a
    red test.

    What it cannot see: which outcome stands in which cell. That is the release's, not the canon's -- the cells are
    compared with the scorecard counted from the assembled release, outside this test. Nor the prose around the
    tables: what the axes mean, how a cell is run, what the map shows -- nor the rest of the caption's words beside
    the scenario's id. Nor a table outside the section, or one whose header is not shaped like a map header."""
    from generation.scorecard import _edge

    start, down, across = CARD_MAP_SECTIONS[language]
    where = f"카드({language}) 「{start.strip()}」"
    body = card_section(start, ("\n## ", "\n### "))
    shaped = re.findall(r"(?m)^\| `[^`|]+` \(%s\) · `[^`|]+` \(%s\) \|" % (re.escape(down), re.escape(across)), body)
    assert len(shaped) == len(card_scenarios()), (
        f"{where}: 지도 머리 꼴의 줄이 {len(shaped)}개입니다 (카드가 말하는 시나리오는 {len(card_scenarios())}개)")
    by_id = {s["scenario_id"]: s for s in canon("scenarios.json")["scenarios"]}
    for sid in card_scenarios():
        grid = by_id[sid]["conditions"]["grid"]
        n = grid["cells"]

        def bins(axis):
            lo, hi = (float(v) for v in axis["range"])
            return [f"{_edge(lo + (hi - lo) * k / n)}~{_edge(lo + (hi - lo) * (k + 1) / n)}" for k in range(n)]

        person, robot = grid["human_axis"], grid["robot_axis"]
        head = (f"| `{person['quantity']}` ({down}) · `{robot['quantity']}` ({across}) | "
                + " | ".join(bins(robot)) + " |")
        rows = card_table(body, head, f"{where} {sid}")
        lines = body.splitlines()
        caption = next((line for line in reversed(lines[:lines.index(head)]) if line.strip()), "")
        named = [s for s in card_scenarios() if f"`{s}`" in caption]
        assert named == [sid], (f"{where}: {sid} 의 지도 바로 위 줄이 시나리오 {named} 를 말합니다 "
                                f"({sid} 하나여야 합니다): 「{caption}」")
        assert len(rows) == n, f"{where}: {sid} 의 지도가 {len(rows)}줄입니다 (정본의 격자는 {n}줄)"
        unread = [cells for cells in rows if len(cells) != 1 + n]
        assert not unread, f"{where}: {sid} 의 지도에 칸 수가 {1 + n} 이 아닌 줄이 있습니다: {unread}"
        said = [cells[0] for cells in rows]
        want = list(reversed(bins(person)))
        assert said == want, f"{where}: {sid} 의 지도의 첫 칸 {said} ↔ 정본의 사람 축 구간(높은 쪽부터) {want}"
        odd = sorted({cell for cells in rows for cell in cells[1:] if not MAP_CELL.fullmatch(cell)})
        assert not odd, f"{where}: {sid} 의 지도에 읽지 못한 칸이 있습니다: {odd}"
