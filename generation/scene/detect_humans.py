#!/usr/bin/env python3
"""범용 사람 검출기가 S1d 의 더미를 사람으로 잡는가.

    .venv-lerobot/bin/python generation/scene/detect_humans.py \\
        --arm skin=<스킨 녹화 디렉터리> --arm capsule=<캡슐 녹화 디렉터리> --out <out.json>

팔(arm)마다 ``bbox.camera``(정본 — 지금 ``head_rgb``)의 무손실 녹화를 읽어, 정답 상자가 있고 32×32 px 이상인
프레임을 ``--stride`` 간격으로 골라 torchvision 의 COCO 사전학습 Faster R-CNN(``fasterrcnn_resnet50_fpn``,
BSD-3)에 넣습니다. 점수 ``--score-min`` 이상인 「person」 상자 하나가 정답 상자와 IoU ``--iou-min`` 이상이면
맞은 프레임입니다. 정답 상자는 녹화 때 분할에서 유도한 값(정본 ``bbox.source``)이고 양끝을 포함하므로 픽셀
가장자리 좌표로 바꿔 잽니다(``bbox.coords``).

배포하지 않는 측정 도구입니다. 검출기 가중치는 처음 실행 때 torchvision 이 내려받습니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation import cameras  # noqa: E402

COCO_PERSON = 1        # torchvision's COCO category index of "person" (checked against the weights at run time)
MIN_AREA_PX = 32 * 32  # a ground-truth box smaller than this is not scored


def edge_box(b) -> tuple[float, float, float, float]:
    """The canon's inclusive [x_min, y_min, x_max, y_max] as a pixel-edge box, the form a detector returns."""
    x0, y0, x1, y1 = (float(v) for v in b)
    return x0, y0, x1 + 1.0, y1 + 1.0


def iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def scorable(bbox, invalid, *, min_area_px: int) -> np.ndarray:
    b = np.asarray(bbox, np.int64)
    valid = ~np.all(b == np.asarray(invalid, np.int64), axis=1)
    area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    return valid & (area >= min_area_px)


def frames_to_score(bbox, invalid, *, min_area_px: int, stride: int) -> list[int]:
    return [int(i) for i in np.flatnonzero(scorable(bbox, invalid, min_area_px=min_area_px))[::stride]]


def is_hit(gt, boxes, *, labels, scores, score_min: float, iou_min: float) -> bool:
    return any(lab == COCO_PERSON and s >= score_min and iou(gt, box) >= iou_min
               for box, lab, s in zip(boxes, labels, scores))


def rate(hits) -> dict:
    hits = list(hits)
    if not hits:
        raise ValueError("채점한 프레임이 없습니다")
    return {"frames": len(hits), "hits": int(sum(hits)), "rate": sum(hits) / len(hits)}


def _detector(device):
    import torch
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    if weights.meta["categories"][COCO_PERSON] != "person":
        raise RuntimeError(f"가중치의 범주 {COCO_PERSON} 가 person 이 아닙니다: {weights.meta['categories'][COCO_PERSON]}")
    model = fasterrcnn_resnet50_fpn(weights=weights).eval().to(device)
    return model, weights.transforms(), str(weights), torch


def _frames(path: Path, wanted: set[int]):
    import av

    with av.open(str(path)) as c:
        for i, f in enumerate(c.decode(c.streams.video[0])):
            if i in wanted:
                yield i, f.to_ndarray(format="rgb24")


def score_arm(episodes_dir: Path, canon: dict, *, stride: int, score_min: float, iou_min: float,
              detector, batch: int = 16) -> dict:
    model, preprocess, _, torch = detector
    device = next(model.parameters()).device
    cam = canon["bbox"]["camera"]
    invalid = canon["bbox"]["invalid"]
    episodes, all_hits = {}, []
    sides = sorted(p for p in episodes_dir.glob("*.json") if not p.name.endswith(".labels.json"))
    if not sides:
        raise ValueError(f"{episodes_dir} 에 에피소드 사이드카가 없습니다")
    for side_path in sides:
        side = json.loads(side_path.read_text(encoding="utf-8"))
        bbox = np.load(side_path.with_suffix(".npz"))["bbox_human"]
        picks = frames_to_score(bbox, invalid, min_area_px=MIN_AREA_PX, stride=stride)
        hits = []
        pending: list[tuple[int, np.ndarray]] = []

        def flush():
            if not pending:
                return
            with torch.no_grad():
                imgs = [preprocess(torch.from_numpy(a).permute(2, 0, 1)).to(device) for _, a in pending]
                outs = model(imgs)
            for (k, _), o in zip(pending, outs):
                hits.append(is_hit(edge_box(bbox[k]), o["boxes"].cpu().tolist(), labels=o["labels"].cpu().tolist(),
                                   scores=o["scores"].cpu().tolist(), score_min=score_min, iou_min=iou_min))
            pending.clear()

        for k, a in _frames(episodes_dir / side["media"][cam]["file"], set(picks)):
            pending.append((k, a))
            if len(pending) >= batch:
                flush()
        flush()
        if len(hits) != len(picks):
            raise RuntimeError(f"{side['episode_id']}: 고른 프레임 {len(picks)} 중 {len(hits)} 만 읽었습니다")
        episodes[side["episode_id"]] = {"seed": side["seed"], **(rate(hits) if hits else {"frames": 0, "hits": 0})}
        all_hits += hits
        print(f"  {side['episode_id']}: {episodes[side['episode_id']]}", flush=True)
    return {"camera": cam, "episodes": episodes, **rate(all_hits)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=DIR",
                   help="팔 이름과 생성 출력 디렉터리 (여러 번)")
    p.add_argument("--stride", type=int, default=5, help="채점할 프레임 간격 (기본 5)")
    p.add_argument("--score-min", type=float, default=0.5)
    p.add_argument("--iou-min", type=float, default=0.5)
    p.add_argument("--out", help="결과 JSON")
    args = p.parse_args(argv)

    arms = dict(a.split("=", 1) for a in args.arm)
    canon = cameras.load(REPO / "meta/cameras.json")
    import torch

    detector = _detector("cuda" if torch.cuda.is_available() else "cpu")
    result = {"detector": detector[2], "stride": args.stride, "score_min": args.score_min, "iou_min": args.iou_min,
              "min_area_px": MIN_AREA_PX, "arms": {}}
    for name, d in arms.items():
        print(f"{name}: {d}", flush=True)
        result["arms"][name] = score_arm(Path(d), canon, stride=args.stride, score_min=args.score_min,
                                         iou_min=args.iou_min, detector=detector)
    for name, r in result["arms"].items():
        print(f"{name}: 검출률 {r['rate']:.3f} ({r['hits']}/{r['frames']} 프레임, {r['camera']})")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
