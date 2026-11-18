"""Bit-level comparison behind the C9 reproduction hook.

Compares only the columns field_map calls "array": a derived column that
disagreed would be a conversion bug, not a reproduction failure, and folding
the two together makes the hook's red light mean two different things.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from generation.field_map import npz_backed_step_fields

_EPISODE_ID = re.compile(r"^(?P<scenario>[^_]+)_(?P<seq>\d{4})_seed(?P<seed>\d+)$")

# Deliberately not compared. LeRobot generates timestamp itself from
# frame_index and fps and stores it as float32; checking it would measure
# LeRobot's arithmetic rather than the simulation's reproduction, and would
# fail on precision alone. Named differently from convert_lerobot's
# LEROBOT_OWNED on purpose -- that one also drops frame_index, which never
# reaches this comparison because field_map calls it derived, and two
# same-named tuples with different contents would invite one to be "fixed"
# to match the other.
EXCLUDED_FROM_COMPARISON = ("timestamp",)


def _parts(episode_id: str) -> re.Match:
    m = _EPISODE_ID.match(episode_id)
    if m is None:
        raise ValueError(f"episode_id 형식이 아닙니다: {episode_id!r}")
    return m


def episode_seq_of(episode_id: str) -> int:
    """The episode_seq encoded in an episode_id (schema §3).

    Raises on anything that does not match. Falling back to 0 would make the
    hook silently re-check the first episode of every batch and always pass.
    """
    return int(_parts(episode_id).group("seq"))


def episode_scenario_of(episode_id: str) -> str:
    """The scenario_id encoded in an episode_id (schema §3) -- the scenario the hook
    has to regenerate. run_s1a.py defaults to S1a, so without it an S1d episode came
    back as S1a and C9 could only report a different episode.

    Raises on anything that does not match, as episode_seq_of does: a fallback would
    rerun the generator's default scenario for any id."""
    return _parts(episode_id).group("scenario")


def comparable_step_fields(active: Iterable[str] = ()) -> tuple[tuple[str, str], ...]:
    """(schema field, npz array) pairs the hook actually compares.

    ``active``: the step-field conditions that hold on both sides (field_map.CAMERAS);
    the caller decides how it knows. With none, the pairs every v0.1 comparison used.
    """
    return tuple((s, n) for s, n in npz_backed_step_fields(active)
                 if s not in EXCLUDED_FROM_COMPARISON)


def compare_arrays(left: dict, right: dict, pairs) -> list[str]:
    """Mismatch descriptions per (schema field, npz array) pair. [] means equal."""
    problems: list[str] = []
    for schema_field, npz_name in pairs:
        if npz_name not in left or npz_name not in right:
            problems.append(f"{schema_field}: 배열 없음 ({npz_name})")
            continue
        a, b = np.asarray(left[npz_name]), np.asarray(right[npz_name])
        if a.shape != b.shape:
            problems.append(f"{schema_field}: shape {a.shape} != {b.shape}")
            continue
        # Numeric dtype is part of what reproduction means -- a float32 column
        # standing in for float64 is a narrowing, not a match. String width is
        # not: "<U11" in the npz and "<U9" out of a JSON parse hold the same
        # values, and numpy compares those by value anyway.
        string_pair = a.dtype.kind in "US" and b.dtype.kind in "US"
        if not string_pair and a.dtype != b.dtype:
            problems.append(f"{schema_field}: dtype {a.dtype} != {b.dtype}")
            continue
        # equal_nan matters: safety.zmp is nan through every flight phase, and
        # plain equality would report each of those steps as a failure.
        if a.dtype.kind in "fc" and b.dtype.kind in "fc":
            same = np.array_equal(a, b, equal_nan=True)
        else:
            same = np.array_equal(a, b)
        if not same:
            problems.append(f"{schema_field}: 값 불일치 ({npz_name})")
    return problems
