"""검증기가 예외로 죽을 때(exit 3) 트레이스백을 stderr 에 남기는지 본다.

**왜 필요한가.** exit 3 은 「데이터셋이 결함」이 아니라 「검증기 자신이 죽었다」는 뜻이다.
그런데 픽스처 회귀에서 그것이 세 번 났고(2026-08-28 · 09-01 SIGSEGV, 09-07 IndexError)
단독 재실행은 전부 초록이라, 메시지 한 줄만으로는 어디서 죽었는지 알 길이 없었다.
트레이스백이 없으면 간헐 크래시는 영원히 「재현 안 됨」으로 남는다.

**이 시험이 보지 못하는 것.** SIGSEGV 는 파이썬 예외가 아니라 여기서 안 잡힌다 —
그쪽은 `main()` 이 켜는 `faulthandler` 가 담당이며(죽는 순간의 파이썬 스택을 stderr 에
쓴다), 프로세스를 실제로 죽여야 확인되므로 이 시험의 범위 밖이다.
"""

from __future__ import annotations

import pytest

import validator.validate_dataset as vd


def test_exit_3_carries_the_traceback(monkeypatch, capsys, tmp_path):
    def boom(*_args, **_kwargs):
        raise RuntimeError("synthetic failure for the traceback test")

    monkeypatch.setattr(vd, "run", boom)

    rc = vd.main([str(tmp_path)])

    err = capsys.readouterr().err
    assert rc == 3
    assert "검증기 실행 실패: RuntimeError: synthetic failure" in err
    assert "Traceback (most recent call last)" in err, (
        "exit 3 인데 트레이스백이 없다 — 간헐 크래시를 다시 조사할 수 없게 된다")
    assert "boom" in err, "트레이스백이 실제로 죽은 프레임을 가리켜야 한다"


def test_a_clean_return_prints_no_traceback(monkeypatch, capsys, tmp_path):
    """트레이스백은 죽었을 때만. 정상 경로의 stderr 는 그대로 비어 있어야 한다."""
    class _Report:
        def __init__(self):
            self.items = []
            self.stats = {}
            self.check_status = {}

        def count(self, _level):
            return 0

    if not hasattr(vd, "print_report"):
        pytest.skip("print_report 가 없는 검증기 판")
    monkeypatch.setattr(vd, "run", lambda *a, **k: _Report())
    monkeypatch.setattr(vd, "print_report", lambda *a, **k: None)

    rc = vd.main([str(tmp_path)])

    assert rc == 0
    assert "Traceback" not in capsys.readouterr().err
