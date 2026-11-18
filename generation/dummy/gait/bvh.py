"""Read BVH motion capture: the hierarchy, the channels and the frames, and the joints' global poses.
Pure numpy. Units and axes are the file's own -- the caller decides what they mean (gait.cycles)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generation.dummy.gait.kinematics import axis_rotation


@dataclass(frozen=True)
class Skeleton:
    names: tuple[str, ...]
    parents: tuple[int, ...]
    offsets: np.ndarray
    channels: tuple[tuple[str, ...], ...]

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"BVH 에 관절 {name!r} 이 없습니다") from None


@dataclass(frozen=True)
class Motion:
    skeleton: Skeleton
    frame_time: float
    frames: np.ndarray


def parse_bvh(text: str) -> Motion:
    tok = text.split()
    pos = 0

    def take(expect: str | None = None) -> str:
        nonlocal pos
        if pos >= len(tok):
            raise ValueError("BVH: 파일이 중간에 끝났습니다")
        t = tok[pos]
        pos += 1
        if expect is not None and t != expect:
            raise ValueError(f"BVH: {expect!r} 자리에 {t!r}")
        return t

    names, parents, offsets, channels = [], [], [], []

    def joint(parent: int) -> None:
        name = take()
        take("{")
        take("OFFSET")
        off = [float(take()) for _ in range(3)]
        take("CHANNELS")
        ch = tuple(take() for _ in range(int(take())))
        me = len(names)
        names.append(name)
        parents.append(parent)
        offsets.append(off)
        channels.append(ch)
        while True:
            t = take()
            if t == "JOINT":
                joint(me)
            elif t == "End":
                take("Site")
                take("{")
                take("OFFSET")
                for _ in range(3):
                    take()
                take("}")
            elif t == "}":
                return
            else:
                raise ValueError(f"BVH: 예상하지 못한 토큰 {t!r}")

    take("HIERARCHY")
    take("ROOT")
    joint(-1)
    take("MOTION")
    take("Frames:")
    n = int(take())
    take("Frame")
    take("Time:")
    frame_time = float(take())
    width = sum(len(c) for c in channels)
    values = tok[pos:]
    if len(values) != n * width:
        raise ValueError(f"BVH: 프레임 값이 {len(values)}개 — {n} 프레임 × {width} 채널이어야 합니다")
    return Motion(Skeleton(tuple(names), tuple(parents), np.array(offsets, float), tuple(channels)),
                  frame_time, np.array(values, float).reshape(n, width))


def read_bvh(path) -> Motion:
    return parse_bvh(Path(path).read_text(encoding="utf-8"))


def forward_kinematics(motion: Motion, scale: float = 1.0):
    """Global positions (F, J, 3) and rotations (F, J, 3, 3) in the file's axes, lengths times ``scale``.
    Rotation channels compose in the order listed (intrinsic); a position channel replaces that axis of the
    joint's offset."""
    sk = motion.skeleton
    F, J = motion.frames.shape[0], len(sk.names)
    pos = np.zeros((F, J, 3))
    rot = np.zeros((F, J, 3, 3))
    col = 0
    for j in range(J):
        R = np.broadcast_to(np.eye(3), (F, 3, 3)).copy()
        local = np.broadcast_to(sk.offsets[j] * scale, (F, 3)).copy()
        for ch in sk.channels[j]:
            v = motion.frames[:, col]
            col += 1
            kind = ch[1:].lower()
            if kind == "position":
                local[:, "XYZ".index(ch[0])] = v * scale
            elif kind == "rotation":
                R = R @ axis_rotation(ch[0].lower(), np.deg2rad(v))
            else:
                raise ValueError(f"BVH: 모르는 채널 {ch!r}")
        p = sk.parents[j]
        if p < 0:
            pos[:, j], rot[:, j] = local, R
        else:
            pos[:, j] = pos[:, p] + np.einsum("fij,fj->fi", rot[:, p], local)
            rot[:, j] = rot[:, p] @ R
    return pos, rot
