"""C4's outcome shares: counted over the random rows, warned on below the floors alone.

A grid row of a conditions table (condition_design = grid) is a cell of the failure map, set by the grid and not
drawn, so the shares C4 holds to its floors leave it out; a row without the column, or with a null design, is
counted. The targets are each scenario's own (meta/scenarios.json outcome_target) and the validator holds none: the
warning names the floor. The label's two-way checks (a nominal row with evidence, a fail without) are not shares
and still run over every row.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from validator import make_fixture as F
from validator import validate_dataset as V


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixtures")
    F.build(root / F.HRI_CLEAN, hri=True)
    F.build(root / "clean")
    return root


def _copy(built: Path, base: str, tmp_path: Path) -> Path:
    root = tmp_path / base
    shutil.copytree(built / base, root)
    return root


def _edit_episodes(root: Path, change) -> pd.DataFrame:
    df = pd.read_parquet(F._eps_path(root))
    change(df)
    df.to_parquet(F._eps_path(root), index=False)
    return df


def _outcome_as_grid_rows(outcome: str):
    """Every row of that outcome made a grid row, each on a cell of its own -- in the flat columns and the condition
    record alike, as a release says it."""
    def edit(df: pd.DataFrame) -> None:
        for k, i in enumerate(df.index[df["outcome"] == outcome]):
            F.as_grid_row(df, i, k, k)
    return edit


def _floor_warnings(rep: V.Report, outcome: str) -> list[V.Finding]:
    return [f for f in rep.findings if f.check == "C4" and f.severity == V.WARN and f.message.startswith(outcome)]


def test_the_validator_holds_no_outcome_target():
    assert not hasattr(V, "OUTCOME_TARGET")


@pytest.mark.parametrize("outcome", sorted(V.OUTCOME_FLOOR))
def test_grid_rows_are_not_counted(built, tmp_path, outcome):
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    df = _edit_episodes(root, _outcome_as_grid_rows(outcome))
    grid = int((df["outcome"] == outcome).sum())
    assert grid
    rep = V.run(root, set(V.ALL_CHECKS), None, 0, None)
    assert not [f.line() for f in rep.findings if f.severity == V.ERROR]
    assert outcome not in rep.stats["outcome_distribution"]
    assert sum(rep.stats["outcome_distribution"].values()) == len(df) - grid
    assert rep.stats["grid_episodes_not_counted"] == grid
    assert len(_floor_warnings(rep, outcome)) == 1


@pytest.mark.parametrize("outcome", sorted(V.OUTCOME_FLOOR))
def test_the_floor_warning_names_the_floor_and_no_target(built, tmp_path, outcome):
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    _edit_episodes(root, _outcome_as_grid_rows(outcome))
    [warning] = _floor_warnings(V.run(root, {"C4"}, None, 0, None), outcome)
    assert f"경고 하한 {V.OUTCOME_FLOOR[outcome]:.0%}" in warning.message
    assert "목표" not in warning.message


def test_a_release_without_the_design_column_counts_every_row(built):
    root = built / "clean"
    eps = pd.read_parquet(F._eps_path(root))
    assert "condition_design" not in eps.columns
    rep = V.run(root, {"C4"}, None, 0, None)
    assert sum(rep.stats["outcome_distribution"].values()) == len(eps)
    assert rep.stats["grid_episodes_not_counted"] == 0


def test_a_row_whose_design_is_null_is_counted(built, tmp_path):
    root = _copy(built, F.HRI_CLEAN, tmp_path)
    df = _edit_episodes(root, lambda d: d.__setitem__("condition_design", None))
    rep = V.run(root, {"C4"}, None, 0, None)
    assert sum(rep.stats["outcome_distribution"].values()) == len(df)
    assert rep.stats["grid_episodes_not_counted"] == 0


def test_a_grid_row_is_still_held_to_its_label(built, tmp_path):
    """A grid row leaves the shares, not the label checks: a fail with no rule behind it is an ERROR there too."""
    root = _copy(built, F.HRI_CLEAN, tmp_path)

    def edit(df: pd.DataFrame) -> None:
        _outcome_as_grid_rows("fail")(df)
        rules = list(df["triggered_rules"])
        rules[int(df.index[df["outcome"] == "fail"][0])] = []
        df["triggered_rules"] = rules
    df = _edit_episodes(root, edit)
    rep = V.run(root, {"C4"}, None, 0, None)
    eid = df.loc[df["outcome"] == "fail", "episode_id"].iloc[0]
    assert [(f.message, f.location) for f in rep.findings if f.severity == V.ERROR] == \
        [("outcome=fail 인데 판정 근거 없음", eid)]
