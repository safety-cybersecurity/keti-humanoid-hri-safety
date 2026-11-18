"""The gait table at run time: joint targets for a speed and a phase, the stride that advances the phase,
and the entry's speed profile. Pure numpy -- the environment converts at its call sites.

Row 0 is standing (speed 0, stride 0, every target 0). Starting from rest, between standing and the slowest
walking row the gait shrinks by sqrt(v / v1), targets and stride by the same factor: the cadence rises as
sqrt(v) and the stride-to-cadence ratio of a walk holds. Slowing to a stop it does not shrink: the legs keep
the slowest row's steps, ever slower, so a foot that is down stays where it landed until the body is at rest,
and the stance the walk ends in is set by the caller (support.rest_stance). (Shrunk to a stop, the targets
would go toward the stand pose with the feet planted a stride apart, and the body would tip over them.) Above the
fastest row the fastest row is used."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaitTable:
    dof_names: tuple
    speeds: np.ndarray
    stride: np.ndarray
    targets: np.ndarray

    def __post_init__(self):
        K, P, J = self.targets.shape
        if len(self.dof_names) != J or len(set(self.dof_names)) != J:
            raise ValueError("dof_names must name every column once")
        if self.speeds.shape != (K,) or self.stride.shape != (K,) or K < 3:
            raise ValueError("speeds and stride need one entry per row, and at least two walking rows")
        # Before the order checks: a NaN fails every comparison and would pass them.
        if not (np.all(np.isfinite(self.speeds)) and np.all(np.isfinite(self.stride))):
            raise ValueError("speeds and stride must be finite")
        if self.speeds[0] != 0.0 or np.any(np.diff(self.speeds) <= 0.0):
            raise ValueError("speeds must start at 0 and increase")
        if self.stride[0] != 0.0 or np.any(self.stride[1:] <= 0.0):
            raise ValueError("the stand row has no stride and every walking row has one")
        if np.any(self.targets[0] != 0.0) or not np.all(np.isfinite(self.targets)):
            raise ValueError("row 0 must be the stand pose and every target finite")

    @classmethod
    def from_json(cls, d: dict) -> "GaitTable":
        K, P, J = (int(v) for v in d["shape"])
        return cls(tuple(d["dof_names"]), np.asarray(d["speeds"], float), np.asarray(d["stride"], float),
                   np.asarray(d["targets"], np.float32).reshape(K, P, J))

    def to_json(self) -> dict:
        return {"dof_names": list(self.dof_names), "speeds": [float(v) for v in self.speeds],
                "stride": [float(v) for v in self.stride], "shape": list(self.targets.shape),
                "targets": [float(v) for v in self.targets.ravel()]}

    def _blend(self, speed, slowing=False):
        v = np.clip(np.asarray(speed, float), 0.0, self.speeds[-1])
        k = np.clip(np.searchsorted(self.speeds, v, side="right") - 1, 1, len(self.speeds) - 2)
        lo, hi = self.speeds[k], self.speeds[k + 1]
        w = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
        amp = np.where((v < self.speeds[1]) & ~np.asarray(slowing, bool), np.sqrt(v / self.speeds[1]), 1.0)
        return k, w, amp

    def stride_at(self, speed, slowing=False) -> np.ndarray:
        k, w, amp = self._blend(speed, slowing)
        return amp * ((1.0 - w) * self.stride[k] + w * self.stride[k + 1])

    def at(self, speed, phase, slowing=False) -> np.ndarray:
        k, w, amp = self._blend(speed, slowing)
        P = self.targets.shape[1]
        x = (np.asarray(phase, float) % 1.0) * P
        i0 = np.floor(x).astype(int) % P
        i1 = (i0 + 1) % P
        f = (x - np.floor(x))[..., None]

        def row(kk):
            return (1.0 - f) * self.targets[kk, i0] + f * self.targets[kk, i1]

        out = (1.0 - w[..., None]) * row(k) + w[..., None] * row(k + 1)
        return (amp[..., None] * out).astype(np.float32)


def advance_phase(phase, distance, stride) -> np.ndarray:
    s = np.asarray(stride, float)
    ok = s > 1e-9
    step = np.where(ok, np.asarray(distance, float) / np.where(ok, s, 1.0), 0.0)
    return (np.asarray(phase, float) + step) % 1.0


def ramp_speed(v_now, v_target, dt: float, accel_time_s: float, decel_time_s: float, remaining=None):
    if not (accel_time_s > 0.0 and decel_time_s > 0.0):
        raise ValueError("ramp times must be positive")
    v_t = np.asarray(v_target, float)
    v = np.minimum(v_t, np.asarray(v_now, float) + v_t / accel_time_s * dt)
    if remaining is not None:
        v = np.minimum(v, np.sqrt(np.maximum(0.0, 2.0 * v_t / decel_time_s * np.asarray(remaining, float))))
    return v


def remaining_to_stop(target_y, stop_y, angle) -> np.ndarray:
    """Metres left along the entry line to the far-side stop line (the target moves -y, sin(angle) per metre)."""
    return np.maximum(0.0, (np.asarray(target_y, float) - np.asarray(stop_y, float)) / np.sin(np.asarray(angle, float)))
