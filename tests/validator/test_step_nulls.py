"""스텝 데이터에 null 이 없어야 한다.

규격의 비유한값은 NaN · ±inf 로 실린다(safety.zmp · safety.stability_margin). null 은 결측이다 —
LeRobotDataset 은 값이나 목록이 통째로 null 인 스텝을 그 열에 None 을 단 채 내주고, 목록 안에 null 이
든 스텝은 그 항목을 읽다 실패한다(실측, lerobot 0.6.1 — validator/README.md). pandas 로 읽으면 둘이
같아 보이므로 Arrow 로 읽는다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import validator.validate_dataset as vd


REL = Path("data/chunk-000/file-000.parquet")


def _steps(root: Path, zmp, margin, torque=None):
    n = len(margin)
    table = pa.table({
        "index": pa.array(range(n), pa.int64()),
        "safety.zmp": pa.array(zmp, pa.list_(pa.float64(), 2)),            # fixed_size_list, as LeRobot writes it
        "safety.stability_margin": pa.array(margin, pa.float64()),
        "observation.state.joint_torque": pa.array(torque or [[0.0, 0.0]] * n, pa.list_(pa.float64())),
    })
    (root / REL).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, root / REL)
    # As load_frames hands them over: the path relative to the distribution's root. The check has
    # to open the file under that root -- from the working directory there is no such file.
    return [(REL, pd.read_parquet(root / REL))]


def _findings(root, frames):
    rep = vd.Report(root=str(root))
    vd.c7_step_nulls(vd.Dataset(root, rep), frames, rep)
    return [f for f in rep.findings if f.check == "C7"]


def test_the_non_finite_values_of_the_spec_are_not_nulls(tmp_path):
    frames = _steps(tmp_path, zmp=[[0.1, 0.2], [float("nan"), float("nan")], [0.3, 0.4]],
                    margin=[0.1, float("-inf"), float("nan")])
    assert _findings(tmp_path, frames) == []


def test_a_null_scalar_is_reported_with_its_column_and_count(tmp_path):
    frames = _steps(tmp_path, zmp=[[0.1, 0.2]] * 3, margin=[0.1, None, None])
    (f,) = _findings(tmp_path, frames)
    assert f.severity == "ERROR" and "safety.stability_margin 에 null 2개" in f.message, f.message


def test_a_null_inside_a_list_is_reported(tmp_path):
    # NaN 을 pandas 로 다시 쓰면 여기에 생긴다: 행은 있고 원소가 null.
    frames = _steps(tmp_path, zmp=[[0.1, 0.2], [None, None], [0.3, 0.4]], margin=[0.1, 0.2, 0.3])
    (f,) = _findings(tmp_path, frames)
    assert "safety.zmp 에 null 2개" in f.message, f.message


def test_a_null_inside_a_variable_length_list_is_reported(tmp_path):
    frames = _steps(tmp_path, zmp=[[0.1, 0.2]] * 2, margin=[0.1, 0.2], torque=[[0.0, None], [0.0, 0.0]])
    (f,) = _findings(tmp_path, frames)
    assert "observation.state.joint_torque 에 null 1개" in f.message, f.message


def test_a_missing_list_is_reported(tmp_path):
    # In the variable-length column: parquet cannot hold a missing fixed-size list at all.
    frames = _steps(tmp_path, zmp=[[0.1, 0.2]] * 2, margin=[0.1, 0.2], torque=[[0.0, 0.0], None])
    (f,) = _findings(tmp_path, frames)
    assert "observation.state.joint_torque 에 null 1개" in f.message, f.message


def test_a_step_file_that_cannot_be_read_is_a_finding_not_a_crash(tmp_path):
    frames = _steps(tmp_path, zmp=[[0.1, 0.2]], margin=[0.1])
    (tmp_path / REL).write_bytes(b"not a parquet file")
    (f,) = _findings(tmp_path, frames)
    assert f.severity == "ERROR" and "Arrow 로 읽지 못함" in f.message, f.message


def test_pandas_alone_cannot_tell_the_two_apart(tmp_path):
    """이 검사가 Arrow 로 읽는 이유를 못박는다 — pandas 가 구분하게 되면 이 시험이 알려 준다."""
    frames = _steps(tmp_path, zmp=[[0.1, 0.2]] * 2, margin=[0.1, None])
    (_, df), = frames
    assert np.isnan(df["safety.stability_margin"].iloc[1])
