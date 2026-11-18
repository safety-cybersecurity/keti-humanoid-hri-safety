"""픽스처가 스텝 열을 선언하는 방식이 변환기와 같은가.

배포본의 `info.json` `features` 는 변환기가 첫 프레임에서 읽어 선언한 것이고(`convert_lerobot.lerobot_features`),
픽스처의 것은 픽스처 생성기가 첫 스텝에서 읽어 선언한 것이다(`make_fixture.step_features`). 검증기는
`generation/` 을 import 하지 않으므로 생성기가 그 방식을 따로 구현했다. 둘이 갈라지면 픽스처는 배포본에 없는
꼴의 선언으로 C7(`c7_step_features`)의 통과 쪽을 시험하게 된다 — 같은 값에서 같은 선언이 나오는지 본다.
"""

import json

import numpy as np
import pandas as pd
import pytest

import generation.convert_lerobot as cl
from validator.make_fixture import build


def _as_the_converter_holds_it(step: pd.Series) -> dict:
    """A stored step as the frame the converter hands LeRobot: a string as it is, an array as it
    is, a number as an array of one (convert_lerobot.step_frames)."""
    return {name: value if isinstance(value, (str, np.ndarray)) else np.asarray([value])
            for name, value in step.items()}


@pytest.mark.parametrize("with_media", [False, True])
def test_the_fixture_declares_its_steps_as_the_converter_would(tmp_path, with_media):
    root = tmp_path / "clean"
    build(root, with_media=with_media)
    features = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))["features"]
    declared = {name: feature for name, feature in features.items() if feature["dtype"] != "video"}
    first = pd.read_parquet(sorted((root / "data").glob("*/*.parquet"))[0]).iloc[0]

    expected = {name: {**feature, "shape": list(feature["shape"])}        # info.json holds a shape as a list
                for name, feature in cl.lerobot_features(_as_the_converter_holds_it(first)).items()}
    assert declared == expected
    # Both kinds of column are among them, and with cameras the ones the recorder types narrower.
    assert {tuple(f["shape"]) == (1,) for f in declared.values()} == {True, False}
    assert ({"int32", "float32"} <= {f["dtype"] for f in declared.values()}) == with_media
