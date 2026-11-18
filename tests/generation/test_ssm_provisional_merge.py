"""잠정 SSM 파라미터가 정본을 덮지 않는지.

`dict(loaded, **provisional)` 은 "빈 칸을 채운다" 처럼 읽히지만 실제로는
provisional 이 가진 **모든** 키를 덮어씁니다. provisional_ssm_params 는 여섯 개를
전부 갖고 있고 무언가는 늘 비어 있었으므로, 병합은 매 실행 발동했고 실측값이
자리표시자로 되돌아갔습니다 — 2026-08-22 에 잰 robot_stop_time 0.31 s 가 0.5 s 로.

정본과 코드가 다른 값을 쓰는데 CI 는 어느 쪽이든 초록입니다 (CONTRIBUTING.md 「함께 고치는 곳」).
그래서 여기서 묶습니다.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RULES = REPO_ROOT / "meta" / "rules.json"

# generation.envs.s1a.mdp.safety imports Isaac Lab at module scope on some
# paths; the helper itself is pure, so pull it in defensively.
safety = pytest.importorskip("generation.envs.s1a.mdp.safety",
                             reason="Isaac Lab 없이 import 되지 않는 환경")
fill_missing_only = safety.fill_missing_only


PROVISIONAL = {
    "human_speed": 1.6, "robot_reaction_time": 0.1, "robot_stop_time": 0.5,
    "intrusion_distance": 0.2,
    "position_uncertainty_robot": 0.05, "position_uncertainty_human": 0.05,
}


def test_a_value_the_canon_has_survives_the_merge():
    loaded = {"robot_stop_time": 0.31, "human_speed": None}
    out = fill_missing_only(loaded, PROVISIONAL, ["human_speed"])
    assert out["robot_stop_time"] == 0.31, "실측값이 자리표시자로 되돌아갔습니다"
    assert out["human_speed"] == 1.6


def test_zero_is_a_value_not_an_absence():
    # 이격거리·불확실도는 0 이 정직한 값일 수 있다 — 시뮬레이터는 상태를 그대로
    # 읽으므로 측정 허용 오차가 구조적으로 0 이다.
    loaded = {"position_uncertainty_robot": 0.0, "human_speed": None}
    out = fill_missing_only(loaded, PROVISIONAL, ["human_speed"])
    assert out["position_uncertainty_robot"] == 0.0


def test_the_canon_values_are_the_ones_the_generator_would_use():
    """정본에 값이 있는 파라미터는 전부 그대로 쓰여야 한다."""
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    ssm = next(r for r in rules["rules"] if r["rule_id"] == "R-TS15066-SSM-01")
    canon = {k: v.get("value") for k, v in ssm["parameters"].items()}

    loaded = {k: (v if isinstance(v, (int, float)) else None) for k, v in canon.items()}
    missing = [k for k, v in loaded.items() if v is None]
    merged = fill_missing_only(loaded, PROVISIONAL, missing)

    overwritten = {k: (loaded[k], merged[k]) for k in loaded
                   if loaded[k] is not None and merged[k] != loaded[k]}
    assert not overwritten, f"정본 값이 잠정값으로 덮였습니다: {overwritten}"


# --- 호출부 -----------------------------------------------------------------
# 위 시험들은 헬퍼만 직접 부릅니다. 호출부 두 곳만 옛 방식으로 되돌리면 전 시험이
# 통과했습니다. 헬퍼가 옳아도 아무도 그것을 쓰지 않으면
# 결함은 그대로입니다. Isaac 없이는 두 모듈을 import 할 수 없으므로 원문을 봅니다.

CALL_SITES = (
    REPO_ROOT / "generation" / "envs" / "s1a" / "mdp" / "commands.py",
    REPO_ROOT / "generation" / "envs" / "s1a" / "s1a_env_cfg.py",
)


@pytest.mark.parametrize("path", CALL_SITES, ids=lambda p: p.name)
def test_the_call_site_uses_the_gap_filling_merge(path):
    src = path.read_text(encoding="utf-8")
    assert "fill_missing_only(" in src, (
        f"{path.name} 이 fill_missing_only 를 쓰지 않습니다 — 헬퍼가 옳아도 "
        "호출부가 옛 방식이면 실측값이 다시 자리표시자로 덮입니다")


@pytest.mark.parametrize("path", CALL_SITES, ids=lambda p: p.name)
def test_the_call_site_does_not_overwrite_with_provisional(path):
    src = path.read_text(encoding="utf-8")
    offenders = [ln.strip() for ln in src.splitlines()
                 if "provisional_ssm_params" in ln and "dict(" in ln and "**" in ln]
    assert not offenders, (
        f"{path.name} 이 provisional 로 덮어쓰는 병합을 씁니다: {offenders} — "
        "`dict(a, **b)` 는 b 가 가진 모든 키를 덮습니다")
