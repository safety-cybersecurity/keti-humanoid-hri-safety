"""Find the gait cycles of a walking clip: its units, its body axes, when each foot is down, and the
cycles from one right heel strike to the next. Pure numpy."""
from __future__ import annotations

import numpy as np

from generation.dummy.gait.bvh import Skeleton


def unit_scale(skeleton: Skeleton) -> float:
    """Metres per file unit, from the rest leg's length: 1.0 for metres, 0.01 for centimetres."""
    k, a = skeleton.index("LeftKnee"), skeleton.index("LeftAnkle")
    length = float(np.linalg.norm(skeleton.offsets[k]) + np.linalg.norm(skeleton.offsets[a]))
    if 0.5 < length < 1.5:
        return 1.0
    if 50.0 < length < 150.0:
        return 0.01
    raise ValueError(f"다리 길이 {length:.3f} 로는 단위를 알 수 없습니다(m 도 cm 도 아님)")


def body_axes(skeleton: Skeleton) -> np.ndarray:
    """Rows: the rest skeleton's front, left and up in the file's axes. Up runs from the hips to the head
    along the spine; left from the right hip to the left; front = left x up."""
    j, up = skeleton.index("Head"), np.zeros(3)
    while skeleton.names[j] != "Hips":
        up = up + skeleton.offsets[j]
        j = skeleton.parents[j]
        if j < 0:
            raise ValueError("Head 가 Hips 아래에 있지 않습니다")
    up = up / np.linalg.norm(up)
    left = skeleton.offsets[skeleton.index("LeftHip")] - skeleton.offsets[skeleton.index("RightHip")]
    left = left - np.dot(left, up) * up
    left = left / np.linalg.norm(left)
    return np.stack([np.cross(left, up), left, up])


def pelvis_heading_deg(rot: np.ndarray, skeleton: Skeleton, C: np.ndarray) -> np.ndarray:
    """The pelvis's heading per frame in degrees, unwrapped: the angle of its front axis in the floor plane, from
    the rest front toward the left (a left turn is positive)."""
    front = np.einsum("fij,j->fi", rot[:, skeleton.index("Hips")], C[0])
    return np.degrees(np.unwrap(np.arctan2(front @ C[1], front @ C[0])))


def foot_signals(pos: np.ndarray, skeleton: Skeleton, C: np.ndarray, side: str, fps: float):
    """(height above the clip's floor, horizontal speed) of a foot -- the lower of its ankle and toe."""
    s = "Left" if side == "left" else "Right"
    pts = pos[:, [skeleton.index(f"{s}Ankle"), skeleton.index(f"{s}Toe")]] @ C.T      # (F, 2, 3) body axes
    height = pts[..., 2].min(axis=1)
    horiz = pts[:, 1, :2]
    speed = np.zeros(len(horiz))
    speed[1:] = np.linalg.norm(np.diff(horiz, axis=0), axis=1) * fps
    speed[0] = speed[1] if len(speed) > 1 else 0.0
    return height - np.percentile(height, 5.0), speed


def _drop_short(mask: np.ndarray, value: bool, n: int) -> np.ndarray:
    out = mask.copy()
    i = 0
    while i < len(out):
        if out[i] == value:
            j = i
            while j < len(out) and out[j] == value:
                j += 1
            if j - i < n and i > 0 and j < len(out):
                out[i:j] = not value
            i = j
        else:
            i += 1
    return out


def foot_contacts(height, horiz_speed, *, height_tol: float, speed_tol: float, min_frames: int) -> np.ndarray:
    raw = (np.asarray(height) < height_tol) & (np.asarray(horiz_speed) < speed_tol)
    return _drop_short(_drop_short(raw, False, min_frames), True, min_frames)


def heel_strikes(contact) -> np.ndarray:
    c = np.asarray(contact, bool)
    return np.nonzero(c[1:] & ~c[:-1])[0] + 1


def gait_cycles(right, left, fps: float, *, min_s: float, max_s: float) -> list[tuple[int, int]]:
    right, left = np.asarray(right), np.asarray(left)
    out = []
    for a, b in zip(right[:-1], right[1:]):
        if min_s <= (b - a) / fps <= max_s and np.count_nonzero((left > a) & (left < b)) == 1:
            out.append((int(a), int(b)))
    return out


def resample_cycle(values, start: int, end: int, samples: int) -> np.ndarray:
    """values[start..end] (end is the next cycle's first frame) at `samples` even phases in [0, 1)."""
    v = np.asarray(values, float)[start:end + 1]
    t = np.linspace(0.0, 1.0, len(v))
    q = np.arange(samples) / samples
    flat = v.reshape(len(v), -1)
    out = np.stack([np.interp(q, t, flat[:, k]) for k in range(flat.shape[1])], -1)
    return out.reshape((samples,) + v.shape[1:])


def make_periodic(cycle, drift) -> np.ndarray:
    """A resampled cycle (samples at phases i / P) with its end-minus-start `drift` taken out in proportion to phase,
    so that it ends where it began. A real cycle does not, and a mean of such cycles jumps where it wraps round."""
    c = np.asarray(cycle, float)
    q = (np.arange(len(c)) / len(c)).reshape((-1,) + (1,) * (c.ndim - 1))
    return c - q * np.asarray(drift, float)
