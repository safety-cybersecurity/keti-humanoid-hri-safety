"""The scorecard of a release: per scenario, four columns -- no combined score, no ranking.

For the scenarios that declare R-TS15066-SSM-02 (schema §6 「조건부 필드 — 사람과 로봇」 · §9.1), a
batch run on the condition table sums up the same way whatever robot and policy ran it, so a run
of another robot can be set beside the baseline's:

    안전        outcome and severity counts, and the distribution of each episode's smallest
                stopping margin (min over its steps of min_distance_human - separation_required)
    과제        completion rate and median completion time (notes.task_completed_step)
    접촉의 성격  contacts made by an approaching robot / with a robot not approaching -- the robot's
                approach speed at the first contact_onset at most the rule's approach_speed_threshold
                (a robot walking past or turning counts here too) -- and, of each, those where the
                person approached faster than the same threshold
    이상 사건    robot falls (of which without contact, and the distance to the person at the last
                step), rack and floor contacts (rack_contact)

The four columns count the scenario's random rows only. Its grid rows (the condition record's design is "grid" and
names a cell [robot cell, human cell]) are left out of them, counted in grid_episodes, and drawn as the failure map: one
cell per grid row, the row's outcome and severity, over the two axes the release's own meta/scenarios.json
(conditions.grid) names -- the task's quantity the robot is given across, the person's down. A cell without a row
is empty, a degenerate row's cell says so, and a scenario entry with grid rows but no grid block (or the block
without grid rows) gets a map that says it is absent, not none. A condition record with no design is a random row.

Degenerate episodes are left out of all four, and counted; on the map they stand as degenerate. Only the release is read: meta/info.json,
meta/rules.json, meta/scenarios.json and the two tables, one directory down as LeRobot lays them out. The tables are read
the way LeRobot and generation/labeling/check_release_labels.py read them -- in the order they are
stored, each episode's rows where its pointers say -- and a release that loader would not read as
it is written (rows out of place, an episode whose rows are another's, a path with glob characters)
is refused, not re-sorted into a release it is not. So is one the card cannot count as written: a
distance, separation or approach speed that is not a finite number, a completion step past the
episode's last, and a null completion step in a scenario that declares R-TS15066-SSM-02 (which has
a completion criterion).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Run as a script, sys.path[0] is this file's directory, not the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generation.directed_separation import (  # noqa: E402
    DIRECTED_SSM_RULE_ID, load_directed_ssm, robot_was_standing, scenario_declares_directed_ssm)
from generation.labeling.check_release_labels import (  # noqa: E402
    GLOB_CHARACTERS, LOADED, ReleaseError, check_spans, read_episodes, read_steps)

SCORECARD_VERSION = 3
# The step columns of numbers the shared reader does not carry: the distance to the person, the
# separation that was required, and the speeds the robot and the person approached at.
NUMBER_COLUMNS = ("safety.min_distance_human", "safety.separation_required",
                  "safety.approach_speed_robot", "safety.approach_speed_human")
OUTCOMES = ("fail", "near_miss", "nominal")
SEVERITIES = ("S0", "S1", "S2", "S3")
QUANTILES = (5, 25, 50, 75, 95)

class ScorecardError(ValueError):
    """The directory cannot be scored -- nothing was summed."""


def _json_file(root: Path, rel: str) -> dict:
    try:
        doc = json.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ScorecardError(f"{rel} 을 읽지 못했습니다: {e}") from e
    if not isinstance(doc, dict):
        raise ScorecardError(f"{rel} 이 JSON 객체가 아닙니다")
    return doc


def _stored(files: list[Path], root: Path, name: str) -> pa.Array:
    """One column of the given files, all together in the order they are stored."""
    chunks: list[pa.Array] = []
    for f in files:
        if name not in pq.read_schema(f).names:
            raise ScorecardError(f"{f.relative_to(root)} 에 열이 없습니다: {name} — 점수표는 "
                                 f"{DIRECTED_SSM_RULE_ID} 를 선언한 시나리오의 배포본에서 셉니다")
        chunks += pq.read_table(f, columns=[name]).column(name).chunks
    return pa.concat_arrays(chunks)


def _numbers(column: pa.Array, name: str) -> np.ndarray:
    if column.null_count:
        raise ScorecardError(f"{name} 에 null 이 {column.null_count}개 있습니다")
    if not (pa.types.is_floating(column.type) or pa.types.is_integer(column.type)):
        raise ScorecardError(f"{name} 이 스텝마다 수 하나인 열이 아닙니다: {column.type}")
    values = column.to_numpy(zero_copy_only=False).astype(float)
    # A NaN approach speed would read as a moving robot (NaN <= threshold is false), a NaN distance
    # would drop out of the least margin.
    bad = int((~np.isfinite(values)).sum())
    if bad:
        raise ScorecardError(f"{name} 에 유한하지 않은 값(NaN · ±inf)이 {bad}개 있습니다")
    return values


def approach_threshold(rules: dict) -> float:
    """R-TS15066-SSM-02's approach_speed_threshold from the release's meta/rules.json, read by
    the one reader the generator uses (it checks the other parameters and the stop table too)."""
    try:
        params, _ = load_directed_ssm(rules)
    except ValueError as e:
        raise ScorecardError(f"meta/rules.json 에서 {DIRECTED_SSM_RULE_ID} 의 approach_speed_threshold 를 "
                             f"읽지 못했습니다 ({e}) — 점수표는 그 규칙을 선언한 시나리오의 배포본에서 "
                             "셉니다") from e
    return params["approach_speed_threshold"]


def summary(values) -> dict | None:
    """n, min, p05 ... p95, max of the finite values; None when there are none."""
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if not arr.size:
        return None
    out = {"n": int(arr.size), "min": float(arr.min())}
    out.update({f"p{q:02d}": float(np.percentile(arr, q)) for q in QUANTILES})
    out["max"] = float(arr.max())
    return out


def episode_facts(distance, separation, approach, approach_human, events, threshold: float) -> dict:
    """What one episode adds to the card, from its step rows in frame order. The person approaches at the
    first contact on the robot's boundary: at the threshold or below is not approaching."""
    d = np.asarray(distance, dtype=float)
    margin = d - np.asarray(separation, dtype=float)
    onsets = [i for i, names in enumerate(events) if "contact_onset" in names]
    first = onsets[0] if onsets else None
    return {
        "min_margin": float(margin.min()) if margin.size else None,
        "contact": first is not None,
        "robot_not_approaching": first is not None and robot_was_standing(float(approach[first]), threshold),
        "person_approaching": first is not None and not robot_was_standing(float(approach_human[first]), threshold),
        "rack_contacts": sum("rack_contact" in names for names in events),
        "last_distance": float(d[-1]) if d.size else None,
    }


def robot_fell(reasons) -> bool:
    """A termination reason that begins with "fall" is a robot fall -- the labeller's own reading of
    termination.reason (it notes the stability rule as not evaluated for such an episode)."""
    return any(str(x).startswith("fall") for x in reasons)


def score_scenario(rows: list[dict], facts: dict[int, dict], fps: float) -> dict:
    """The four columns of one scenario's random rows; degenerate ones counted and left out."""
    kept = [r for r in rows if not r["degenerate"]]
    n = len(kept)
    outcome = {o: sum(r["outcome"] == o for r in kept) for o in OUTCOMES}
    with_criterion = [r for r in kept if r["completed_step"] is not None]
    done = [r for r in with_criterion if r["completed_step"] >= 0]
    fell = [r for r in kept if robot_fell(r["reason"])]
    contact = [r for r in kept if facts[r["episode_index"]]["contact"]]
    still = [r for r in contact if facts[r["episode_index"]]["robot_not_approaching"]]
    moving = [r for r in contact if not facts[r["episode_index"]]["robot_not_approaching"]]

    def person(group):
        return sum(facts[r["episode_index"]]["person_approaching"] for r in group)
    racks = [facts[r["episode_index"]]["rack_contacts"] for r in kept]
    return {
        "episodes": len(rows),
        "degenerate_excluded": len(rows) - n,
        "safety": {
            "outcome": outcome,
            "outcome_ratio": {o: (c / n if n else None) for o, c in outcome.items()},
            "severity": {s: sum(r["severity"] == s for r in kept) for s in SEVERITIES},
            "min_stopping_margin_m": summary(facts[r["episode_index"]]["min_margin"] for r in kept),
        },
        "task": {
            "with_criterion": len(with_criterion),
            "completed": len(done),
            "completion_rate": len(done) / len(with_criterion) if with_criterion else None,
            "median_time_s": float(np.median([r["completed_step"] / fps for r in done])) if done else None,
        },
        "contact": {"episodes_with_contact": len(contact),
                    "approaching_robot": len(moving),
                    "robot_not_approaching": len(still),
                    "person_approaching": {"approaching_robot": person(moving),
                                           "robot_not_approaching": person(still)}},
        "anomalies": {
            "robot_falls": len(fell),
            "falls_without_contact": sum(not facts[r["episode_index"]]["contact"] for r in fell),
            "distance_at_fall_m": summary(facts[r["episode_index"]]["last_distance"] for r in fell),
            "rack_contact_episodes": sum(1 for c in racks if c),
            "rack_contact_events": sum(racks),
        },
    }


def _axis(sid: str, where: str, node: object) -> dict:
    """{"quantity", "range"} of one grid axis; the range is two increasing finite numbers."""
    quantity = node.get("quantity") if isinstance(node, dict) else None
    window = node.get("range") if isinstance(node, dict) else None
    if (not isinstance(quantity, str) or not quantity or not isinstance(window, list) or len(window) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in window)
            or not window[0] < window[1]):
        raise ScorecardError(f"meta/scenarios.json 의 {sid} conditions.grid.{where} 가 quantity(비지 않은 문자열)와 "
                             f"커지는 유한한 수 둘의 range 를 가진 객체가 아닙니다: {node!r}"[:300])
    return {"quantity": quantity, "range": [float(window[0]), float(window[1])]}


def _grid_block(entry: dict | None, sid: str) -> dict | None:
    """The scenario entry's conditions.grid block as {"cells", "x", "y"}; None when it has none."""
    conditions = entry.get("conditions") if isinstance(entry, dict) else None
    block = conditions.get("grid") if isinstance(conditions, dict) else None
    if block is None:
        return None
    cells = block.get("cells") if isinstance(block, dict) else None
    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 1:
        raise ScorecardError(f"meta/scenarios.json 의 {sid} conditions.grid.cells 가 1 이상의 정수가 아닙니다: {cells!r}")
    return {"cells": cells, "x": _axis(sid, "robot_axis", block.get("robot_axis")),
            "y": _axis(sid, "human_axis", block.get("human_axis"))}


def failure_map(rows: list[dict], entry: dict | None, scenario_id: str) -> dict | None:
    """The scenario's failure map (see the module docstring); ``rows`` are its grid rows. None when there are no grid
    rows and the entry has no conditions.grid, {"absent": why} when only one of the two is there, else the cells --
    outcome[j][i] and severity[j][i] are the row of human cell j and robot cell i (None where no row is)."""
    block = _grid_block(entry, scenario_id)
    if block is None and not rows:
        return None
    if block is None:
        return {"absent": f"meta/scenarios.json 의 {scenario_id} 항목에 격자 블록(conditions.grid)이 없습니다 — 격자 행 "
                          f"{len(rows)}개의 칸을 놓을 축을 알 수 없습니다"}
    if not rows:
        return {"absent": f"{scenario_id} 에 격자 행이 없습니다 — conditions.grid 는 있으나 칸에 놓을 에피소드가 없습니다"}
    n = block["cells"]
    outcome: list[list] = [[None] * n for _ in range(n)]
    severity: list[list] = [[None] * n for _ in range(n)]
    seen: dict[tuple[int, int], int] = {}
    for r in rows:
        i, j = r["grid_cell"]
        if not (0 <= i < n and 0 <= j < n):
            raise ScorecardError(f"에피소드 {r['episode_index']} 의 grid_cell {[i, j]} 이 {n} × {n} 격자 밖입니다")
        if (i, j) in seen:
            raise ScorecardError(f"격자의 칸 {[i, j]} 에 에피소드 {seen[(i, j)]} 와 {r['episode_index']} 두 행이 있습니다 — "
                                 "칸마다 행은 하나입니다")
        seen[(i, j)] = r["episode_index"]
        outcome[j][i] = "degenerate" if r["degenerate"] else r["outcome"]
        severity[j][i] = None if r["degenerate"] else r["severity"]
    return {"cells": n, "x": block["x"], "y": block["y"], "outcome": outcome, "severity": severity}


def _design(index: int, condition: object) -> tuple[str, list[int] | None]:
    """("random", None) or ("grid", [robot cell, human cell]) from an episode's condition record. A record with no
    design, or no condition at all, is a random row."""
    if not isinstance(condition, dict) or "design" not in condition:
        return "random", None
    design = condition["design"]
    if design not in ("random", "grid"):
        raise ScorecardError(f"에피소드 {index} 의 notes.randomization.condition.design 이 random 이나 grid 가 아닙니다: "
                             f"{design!r}")
    if design == "random":
        return "random", None
    cell = condition.get("grid_cell")
    if (not isinstance(cell, list) or len(cell) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in cell)):
        raise ScorecardError(f"에피소드 {index} 는 격자 행인데 notes.randomization.condition.grid_cell 이 정수 둘의 목록이 "
                             f"아닙니다: {cell!r}")
    return "grid", [int(cell[0]), int(cell[1])]


def _episode_rows(episodes: list[dict], degenerate: list, with_criterion: set[str]) -> list[dict]:
    """``with_criterion``: the scenario ids that declare R-TS15066-SSM-02 and so have a completion
    criterion -- their episodes name a step (or -1), never null."""
    rows = []
    for row, flag in zip(episodes, degenerate):
        index = row["episode_index"]
        try:
            notes = json.loads(row["notes"])
            termination = json.loads(row["termination"])
        except (TypeError, ValueError) as e:
            raise ScorecardError(f"에피소드 {index} 의 notes · termination 이 JSON 이 아닙니다: {e}") from e
        if not isinstance(notes, dict):
            raise ScorecardError(f"에피소드 {index} 의 notes 가 JSON 객체가 아닙니다")
        completed = notes.get("task_completed_step")
        if completed is not None and (isinstance(completed, bool) or not isinstance(completed, int)
                                      or completed < -1):
            raise ScorecardError(f"에피소드 {index} 의 notes.task_completed_step 이 -1 이상의 정수나 null 이 "
                                 f"아닙니다: {completed!r}")
        if completed is None and str(row["scenario_id"]) in with_criterion:
            raise ScorecardError(f"에피소드 {index} 의 notes.task_completed_step 이 null 입니다 — 시나리오 "
                                 f"{row['scenario_id']} 는 {DIRECTED_SSM_RULE_ID} 를 선언해 완료 기준이 있으므로 "
                                 "완료한 스텝이나 -1 이어야 합니다")
        reason = termination.get("reason") if isinstance(termination, dict) else None
        if not isinstance(reason, list):
            raise ScorecardError(f"에피소드 {index} 의 termination.reason 이 목록이 아닙니다")
        if row["outcome"] not in OUTCOMES or row["severity"] not in SEVERITIES:
            raise ScorecardError(f"에피소드 {index} 의 outcome · severity 가 정본의 값이 아닙니다: "
                                 f"{row['outcome']!r} · {row['severity']!r}")
        randomization = notes.get("randomization")
        condition = randomization.get("condition") if isinstance(randomization, dict) else None
        design, grid_cell = _design(index, condition)
        rows.append({"episode_index": index, "scenario_id": str(row["scenario_id"]),
                     "outcome": row["outcome"], "severity": row["severity"],
                     "degenerate": flag, "reason": reason, "completed_step": completed,
                     "design": design, "grid_cell": grid_cell})
    return rows


def scorecard(root: Path) -> dict:
    """The scorecard of the release at ``root``."""
    root = Path(root)
    # What a loader given this root passes on is the path joined to the working directory; a glob
    # character in it makes `datasets` read the neighbours of the files (see GLOB_CHARACTERS).
    handed = os.path.join(os.getcwd(), os.fspath(root))
    magic = sorted(set(GLOB_CHARACTERS) & set(handed))
    if magic:
        raise ScorecardError(f"배포본의 경로에 glob 문자 {magic} 가 있습니다 — 이 경로를 받은 적재기(datasets)는 파일 "
                             f"경로를 패턴으로 풀어 이웃 디렉터리의 파일을 읽습니다: {handed}")
    info = _json_file(root, "meta/info.json")
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not (math.isfinite(fps) and fps > 0):
        raise ScorecardError(f"meta/info.json 의 fps 가 유한한 양수가 아닙니다: {fps!r}")
    threshold = approach_threshold(_json_file(root, "meta/rules.json"))
    entries = _json_file(root, "meta/scenarios.json").get("scenarios")
    if not isinstance(entries, list):
        raise ScorecardError("meta/scenarios.json 의 scenarios 가 목록이 아닙니다")
    # Every entry is an object with a scenario_id that is a non-empty string, once: an entry skipped or
    # read twice would drop or change its scenario's completion criterion and map without a word.
    entry_of: dict[str, dict] = {}
    for e in entries:
        sid = e.get("scenario_id") if isinstance(e, dict) else None
        if not isinstance(sid, str) or not sid:
            raise ScorecardError(f"meta/scenarios.json 의 scenarios 에 scenario_id 가 비지 않은 문자열인 객체가 아닌 "
                                 f"항목이 있습니다: {e!r}"[:300])
        if sid in entry_of:
            raise ScorecardError(f"meta/scenarios.json 에 시나리오 {sid} 의 항목이 두 번 있습니다 — 어느 창으로 실패 "
                                 "지도를 셀지 정할 수 없습니다 (scenario_id 가 겹침)")
        entry_of[sid] = e
    with_criterion = {sid for sid, e in entry_of.items() if scenario_declares_directed_ssm(e)}
    try:
        episodes = read_episodes(root)
        steps = read_steps(root)
        check_spans(root, info, episodes, steps)
        degenerate = _stored(sorted((root / "meta" / "episodes").glob(LOADED)), root, "degenerate")
        numbers = {c: _numbers(_stored(steps["files"], root, c), c) for c in NUMBER_COLUMNS}
    except ReleaseError as e:
        raise ScorecardError(f"배포본을 적재기가 읽는 대로 읽을 수 없습니다 — {e}") from e
    except (pa.ArrowException, OSError) as e:
        raise ScorecardError(f"parquet 을 읽지 못했습니다: {e}") from e
    if not pa.types.is_boolean(degenerate.type) or degenerate.null_count:
        raise ScorecardError(f"에피소드 표의 degenerate 가 null 없는 bool 열이 아닙니다: {degenerate.type}")
    rows = _episode_rows(episodes, degenerate.to_pylist(), with_criterion)
    facts: dict[int, dict] = {}
    for r, episode in zip(rows, episodes):
        a, b = episode["dataset_from_index"], episode["dataset_to_index"]
        if r["completed_step"] is not None and r["completed_step"] >= b - a:
            raise ScorecardError(f"에피소드 {r['episode_index']} 의 notes.task_completed_step {r['completed_step']} 이 "
                                 f"그 에피소드의 스텝({b - a}개 — 0..{b - a - 1}) 안이 아닙니다")
        names = [steps["events"][int(i)] for i in steps["event_of"][a:b]]
        facts[r["episode_index"]] = episode_facts(
            numbers["safety.min_distance_human"][a:b], numbers["safety.separation_required"][a:b],
            numbers["safety.approach_speed_robot"][a:b], numbers["safety.approach_speed_human"][a:b],
            names, threshold)
    scenarios = {}
    for sid in sorted({r["scenario_id"] for r in rows}):
        mine = [r for r in rows if r["scenario_id"] == sid]
        random_rows = [r for r in mine if r["design"] == "random"]
        grid_rows = [r for r in mine if r["design"] == "grid"]
        scenarios[sid] = {**score_scenario(random_rows, facts, float(fps)), "grid_episodes": len(grid_rows),
                          "failure_map": failure_map(grid_rows, entry_of.get(sid), sid)}
    return {"scorecard_version": SCORECARD_VERSION, "approach_speed_threshold_mps": threshold,
            "fps": fps, "scenarios": scenarios}


def _count(c: int, n: int) -> str:
    return f"{c} ({100 * c / n:.1f} %)" if n else f"{c}"


def _metres(s: dict | None, key: str) -> str:
    return "–" if not s else f"{s[key]:.3f}"


def _edge(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def _map_lines(m: dict | None) -> list[str]:
    """The failure map as a table of cells: the person's cell down (its high end on top), the task's across; a cell
    reads "<outcome> <severity>", "degenerate", or – where no row is."""
    if m is None:
        return []
    if "absent" in m:
        return [f"실패 지도 없음 — {m['absent']}", ""]
    n = m["cells"]

    def labels(axis):
        lo, hi = axis["range"]
        edges = [lo + (hi - lo) * k / n for k in range(n + 1)]
        return [f"{_edge(edges[k])}~{_edge(edges[k + 1])}" for k in range(n)]

    def text(j, i):
        outcome, severity = m["outcome"][j][i], m["severity"][j][i]
        return "–" if outcome is None else outcome if severity is None else f"{outcome} {severity}"

    across, down = labels(m["x"]), labels(m["y"])
    lines = ["실패 지도 — 격자 행마다 한 칸 (결과 · 심각도)", "",
             f"| `{m['y']['quantity']}` (세로) · `{m['x']['quantity']}` (가로) | " + " | ".join(across) + " |",
             "|---" * (n + 1) + "|"]
    for j in reversed(range(n)):
        lines.append(f"| {down[j]} | " + " | ".join(text(j, i) for i in range(n)) + " |")
    return lines + [""]


def markdown(card: dict) -> str:
    """One table of four columns per scenario."""
    lines = ["# 점수표", "",
             f"다가감 문턱 {card['approach_speed_threshold_mps']:g} m/s (자체 설정) · 합친 점수와 순위는 두지 않습니다.", ""]
    for sid, s in card["scenarios"].items():
        n = s["episodes"] - s["degenerate_excluded"]
        safety, task, contact, anomalies = s["safety"], s["task"], s["contact"], s["anomalies"]
        margin, fall = safety["min_stopping_margin_m"], anomalies["distance_at_fall_m"]
        columns = [
            [f"실패 {_count(safety['outcome']['fail'], n)}",
             f"근접 {_count(safety['outcome']['near_miss'], n)}",
             f"정상 {_count(safety['outcome']['nominal'], n)}",
             " · ".join(f"{k} {v}" for k, v in safety["severity"].items()),
             f"정지거리 여유 최솟값 p5 {_metres(margin, 'p05')} · 중앙 {_metres(margin, 'p50')} · "
             f"p95 {_metres(margin, 'p95')} m"],
            [f"완료 {_count(task['completed'], task['with_criterion'])}" if task["with_criterion"]
             else "완료 기준 없음",
             f"걸린 시간 중앙 {task['median_time_s']:.2f} s" if task["median_time_s"] is not None
             else "걸린 시간 –"],
            [f"접촉 {contact['episodes_with_contact']}",
             f"다가가던 로봇 {contact['approaching_robot']} "
             f"(사람도 다가가던 것 {contact['person_approaching']['approaching_robot']})",
             f"다가가지 않던 로봇 {contact['robot_not_approaching']} "
             f"(사람이 다가가던 것 {contact['person_approaching']['robot_not_approaching']})"],
            [f"넘어짐 {anomalies['robot_falls']} (접촉 없이 {anomalies['falls_without_contact']})",
             f"넘어질 때 거리 중앙 {_metres(fall, 'p50')} m",
             f"랙 · 바닥 접촉 {anomalies['rack_contact_episodes']} 에피소드 · {anomalies['rack_contact_events']} 번"],
        ]
        grid = f" · 격자 행 {s['grid_episodes']}" if s["grid_episodes"] else ""
        lines += [f"## {sid} — 무작위 행 {s['episodes']} (degenerate 제외 {s['degenerate_excluded']}){grid}", "",
                  "| 안전 | 과제 | 접촉의 성격 | 이상 사건 |", "|---|---|---|---|"]
        for i in range(max(len(c) for c in columns)):
            lines.append("| " + " | ".join(c[i] if i < len(c) else "" for c in columns) + " |")
        lines.append("")
        lines += _map_lines(s.get("failure_map"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="배포본 하나의 점수표 — 시나리오마다 네 칸, 합친 점수 없음")
    ap.add_argument("--dataset", required=True, type=Path, help="LeRobot 배포본 디렉터리")
    ap.add_argument("--out", type=Path, help="점수표 JSON 을 쓸 곳")
    ap.add_argument("--markdown", type=Path, help="점수표 Markdown 을 쓸 곳")
    a = ap.parse_args(argv)
    try:
        card = scorecard(a.dataset)
    except ScorecardError as e:
        print(f"점수표를 셀 수 없습니다: {e}", file=sys.stderr)
        return 2
    text = markdown(card)
    if a.out:
        a.out.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if a.markdown:
        a.markdown.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
