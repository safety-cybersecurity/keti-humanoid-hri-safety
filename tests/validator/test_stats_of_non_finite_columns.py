"""비유한값이 있는 열의 정규화 통계를 C8 이 보는가.

**결함 픽스처로는 못 잡습니다** — 합성 픽스처에는 `meta/stats.json` 이 없습니다.
LeRobot 이 쓰는 파일이라 픽스처 생성기가 만들지 않고, 그러면 이 검사는 픽스처
회귀에서 **한 번도 돌지 않습니다**. 그래서 여기서 직접 부릅니다
(이 저장소에서 「돌리지 않은 검사가 OK 로 찍혔다」가 여러 번 났습니다).

배경: LeRobot 의 통계 경로는 `-inf` 가 하나만 있어도 그 필드의 q01~q99 를 전부
최댓값 부근으로 무너뜨립니다. 값이 **유한하므로** 앞선 `NaN → null` 청소를 그대로
통과합니다. 배포본 400건 실측(2026-08-29)에서 `safety.stability_margin` 의 다섯
분위수가 전부 +0.1235 였고 유한값의 실제 중앙값은 -0.1302 였습니다 — 「음수면
지지다각형 밖」인 필드의 중앙값이 부호가 뒤집힌 채 실려 있었습니다.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from validator.validate_dataset import (Dataset, Report,
                                        check_stats_of_non_finite_columns)

COLUMN = "safety.stability_margin"


def run(tmp_path, stats_block, values=(-0.5, float("-inf"), 0.1), eps=None):
    root = tmp_path / "dist"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "stats.json").write_text(
        json.dumps({COLUMN: stats_block}), encoding="utf-8")
    rep = Report(root=str(root))
    frames = [(root / "data" / "f.parquet", pd.DataFrame({COLUMN: list(values)}))]
    check_stats_of_non_finite_columns(Dataset(root, rep), frames, eps, rep)
    return rep.findings


def test_finite_but_collapsed_quantiles_are_an_error(tmp_path):
    """무너진 분위수는 유한하다 — 그래서 조용하다. 그것이 이 검사가 있는 이유다."""
    findings = run(tmp_path, {"min": [None], "max": [0.1], "count": [3],
                              "q50": [0.1], "q99": [0.1]})
    errors = [f for f in findings if f.severity == "ERROR"]
    assert errors, f"무너진 분위수를 통과시켰습니다: {findings}"
    assert "q50" in errors[0].message and "max" in errors[0].message, errors[0].message


def test_a_block_of_all_null_lists_counts_as_blank(tmp_path):
    """`[null, null]` 은 값이 아니다.

    처음 판은 `v is not None` 으로만 봐서 **이미 비운 블록을 「값 있음」으로** 셌고,
    실배포본의 `safety.zmp` 가 그렇게 잘못 걸렸습니다 (2026-08-29 자가 확인).
    """
    findings = run(tmp_path, {"min": [None, None], "q50": [None, None],
                              "count": [3], "undefined_reason": "정의되지 않음"})
    assert not findings, f"비어 있는 블록을 걸었습니다: {findings}"


def test_blank_without_a_reason_is_a_warning(tmp_path):
    """비운 것과 빠뜨린 것은 받는 쪽에서 구별되지 않는다 — 사유가 그 차이다."""
    findings = run(tmp_path, {"min": [None], "q50": [None], "count": [3]})
    assert [f.severity for f in findings] == ["WARN"], findings
    assert "사유가 없습니다" in findings[0].message


def test_a_column_without_non_finite_values_is_left_alone(tmp_path):
    """멀쩡한 열의 통계를 요구하면 검사가 자기 생산자를 거부한다."""
    findings = run(tmp_path, {"min": [-0.5], "max": [0.1], "q50": [0.0], "count": [3]},
                   values=(-0.5, 0.0, 0.1))
    assert not findings, findings


def test_the_column_list_is_derived_from_the_data(tmp_path):
    """대상 열을 코드에 적어 두지 않는다 — 새 필드가 비유한값을 담으면 저절로 걸린다."""
    root = tmp_path / "dist"
    (root / "meta").mkdir(parents=True)
    other = "observation.state.joint_pos"
    (root / "meta" / "stats.json").write_text(
        json.dumps({other: {"min": [0.1], "q50": [0.1], "count": [2]}}), encoding="utf-8")
    rep = Report(root=str(root))
    frames = [(root / "data" / "f.parquet",
               pd.DataFrame({other: [[0.1, 0.2], [0.1, float("nan")]]}))]
    check_stats_of_non_finite_columns(Dataset(root, rep), frames, None, rep)
    assert [f.severity for f in rep.findings] == ["ERROR"], rep.findings
    assert other in rep.findings[0].message


def test_c8_actually_calls_it(tmp_path):
    """헬퍼만 시험하면 호출부를 지워도 초록입니다 — 이 저장소에서 여러 번 난 형태."""
    import inspect

    import validator.validate_dataset as v

    assert "check_stats_of_non_finite_columns(" in inspect.getsource(v.c8_values), (
        "C8 이 이 검사를 부르지 않습니다")


BLANK = {"min": [None], "q50": [None], "count": [3], "undefined_reason": "정의되지 않음"}


def test_the_per_episode_table_is_checked_too(tmp_path):
    """전역 통계만 비우면 **같은 틀린 값이 옆 파일에 남는다.**

    무너지는 것은 에피소드 단계다 — 배포본 400건 전부의 `stats/…/q50` 이 자기
    최댓값 부근이었고, 전역 통계는 그것을 합친 결과였다 (2026-08-29).
    """
    eps = pd.DataFrame({
        f"stats/{COLUMN}/q50": [[0.1], [0.1]],
        f"stats/{COLUMN}/count": [[3], [3]],
    })
    findings = run(tmp_path, BLANK, eps=eps)
    errors = [f for f in findings if f.severity == "ERROR"]
    assert errors, f"에피소드별 통계의 무너진 값을 통과시켰습니다: {findings}"
    assert "q50" in errors[0].message and "count" not in errors[0].message, errors[0].message


def test_a_blanked_per_episode_table_passes(tmp_path):
    """비운 뒤에는 조용해야 한다 — 아니면 검사가 자기 생산자를 거부한다."""
    eps = pd.DataFrame({
        f"stats/{COLUMN}/q50": [[float("nan")], [float("nan")]],
        f"stats/{COLUMN}/count": [[3], [3]],
    })
    assert not run(tmp_path, BLANK, eps=eps)
