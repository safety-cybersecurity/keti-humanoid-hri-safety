"""generation/scene/detect_humans.py — 검출률이 무엇을 세는가.

검출기(torchvision)는 `.venv` 에 없으므로 시험하지 않습니다. 시험하는 것은 채점 규칙입니다 — 어느 프레임을
채점하는가(정답 상자가 있고 32×32 px 이상), 정답 상자를 어떤 좌표로 바꾸는가(정본 `bbox.coords` 의 양끝 포함
→ 픽셀 가장자리), 어떤 검출을 맞은 것으로 치는가(사람 · 점수 문턱 · IoU 0.5).
"""

from pathlib import Path

import numpy as np
import pytest

from generation import cameras
from generation.scene import detect_humans as dh

REPO = Path(__file__).resolve().parents[2]
CANON = cameras.load(REPO / "meta/cameras.json")
INVALID = CANON["bbox"]["invalid"]


def test_the_inclusive_canon_box_becomes_a_pixel_edge_box():
    # bbox.coords: x_max, y_max are the last pixel column/row; a detector's box ends one past them.
    assert dh.edge_box([10, 20, 19, 29]) == (10.0, 20.0, 20.0, 30.0)


def test_iou_of_identical_disjoint_and_half_overlapping_boxes():
    a = (0.0, 0.0, 10.0, 10.0)
    assert dh.iou(a, a) == pytest.approx(1.0)
    assert dh.iou(a, (20.0, 20.0, 30.0, 30.0)) == 0.0
    assert dh.iou(a, (5.0, 0.0, 15.0, 10.0)) == pytest.approx(50 / 150)


def test_scorable_frames_have_a_box_of_at_least_the_minimum_area():
    bbox = np.array([INVALID, [0, 0, 31, 31], [0, 0, 30, 31], [5, 5, 104, 204]], np.int32)
    assert dh.scorable(bbox, INVALID, min_area_px=32 * 32).tolist() == [False, True, False, True]


def test_a_person_box_over_the_threshold_that_overlaps_enough_is_a_hit():
    gt = (100.0, 100.0, 200.0, 300.0)
    boxes = [(102.0, 98.0, 198.0, 305.0)]
    assert dh.is_hit(gt, boxes, labels=[dh.COCO_PERSON], scores=[0.9], score_min=0.5, iou_min=0.5)


@pytest.mark.parametrize("label, score, box", [
    (3, 0.9, (102.0, 98.0, 198.0, 305.0)),               # a car, not a person
    (None, 0.3, (102.0, 98.0, 198.0, 305.0)),            # below the score threshold
    (None, 0.9, (100.0, 100.0, 150.0, 180.0)),           # a person, but a part of the box
])
def test_a_wrong_class_a_weak_score_or_a_small_overlap_is_a_miss(label, score, box):
    gt = (100.0, 100.0, 200.0, 300.0)
    label = dh.COCO_PERSON if label is None else label
    assert not dh.is_hit(gt, [box], labels=[label], scores=[score], score_min=0.5, iou_min=0.5)


def test_any_one_matching_box_among_several_is_enough():
    gt = (100.0, 100.0, 200.0, 300.0)
    boxes = [(400.0, 0.0, 500.0, 100.0), (101.0, 101.0, 199.0, 299.0)]
    assert dh.is_hit(gt, boxes, labels=[dh.COCO_PERSON] * 2, scores=[0.99, 0.6], score_min=0.5, iou_min=0.5)


def test_frames_to_score_takes_every_stride_th_scorable_frame():
    bbox = np.array([INVALID] * 3 + [[0, 0, 99, 99]] * 7, np.int32)
    assert dh.frames_to_score(bbox, INVALID, min_area_px=1024, stride=3) == [3, 6, 9]


def test_rate_reports_hits_over_scored_frames_and_refuses_none_scored():
    assert dh.rate([True, False, True, True]) == {"frames": 4, "hits": 3, "rate": 0.75}
    with pytest.raises(ValueError):
        dh.rate([])
