"""A synthetic walking BVH with 100STYLE's joint names (Y up, centimetres, ZXY rotation channels) for the
gait build's tests. The stance foot does not slide: the root is placed from the foot that is down."""
from __future__ import annotations

import math

import numpy as np

# (name, parent, offset in cm, channels)
SKELETON = [
    ("Hips", None, (0, 0, 0), 6), ("Chest", "Hips", (0, 10, 0), 3), ("Chest2", "Chest", (0, 10, 0), 3),
    ("Chest3", "Chest2", (0, 10, 0), 3), ("Chest4", "Chest3", (0, 10, 0), 3), ("Neck", "Chest4", (0, 15, 0), 3),
    ("Head", "Neck", (0, 10, 0), 3),
    ("LeftCollar", "Chest4", (5, 10, 0), 3), ("LeftShoulder", "LeftCollar", (12, 0, 0), 3),
    ("LeftElbow", "LeftShoulder", (28, 0, 0), 3), ("LeftWrist", "LeftElbow", (25, 0, 0), 3),
    ("RightCollar", "Chest4", (-5, 10, 0), 3), ("RightShoulder", "RightCollar", (-12, 0, 0), 3),
    ("RightElbow", "RightShoulder", (-28, 0, 0), 3), ("RightWrist", "RightElbow", (-25, 0, 0), 3),
    ("LeftHip", "Hips", (9, 0, 0), 3), ("LeftKnee", "LeftHip", (0, -42, 0), 3),
    ("LeftAnkle", "LeftKnee", (0, -42, 0), 3), ("LeftToe", "LeftAnkle", (0, -7, 14), 3),
    ("RightHip", "Hips", (-9, 0, 0), 3), ("RightKnee", "RightHip", (0, -42, 0), 3),
    ("RightAnkle", "RightKnee", (0, -42, 0), 3), ("RightToe", "RightAnkle", (0, -7, 14), 3),
]


def _hierarchy() -> str:
    children = {}
    for name, parent, _, _ in SKELETON:
        children.setdefault(parent, []).append(name)
    spec = {n: (o, c) for n, _, o, c in SKELETON}

    def block(name, depth, root=False):
        pad = "  " * depth
        off, nch = spec[name]
        ch = "Xposition Yposition Zposition Zrotation Xrotation Yrotation" if nch == 6 else "Zrotation Xrotation Yrotation"
        lines = [f"{pad}{'ROOT' if root else 'JOINT'} {name}", f"{pad}{{",
                 f"{pad}  OFFSET {off[0]:.4f} {off[1]:.4f} {off[2]:.4f}", f"{pad}  CHANNELS {nch} {ch}"]
        kids = children.get(name, [])
        for k in kids:
            lines += block(k, depth + 1)
        if not kids:
            lines += [f"{pad}  End Site", f"{pad}  {{", f"{pad}    OFFSET 0.0 1.0 0.0", f"{pad}  }}"]
        lines.append(f"{pad}}}")
        return lines

    return "\n".join(["HIERARCHY"] + block("Hips", 0, root=True))


def synthetic_walk_bvh(speed: float = 1.2, seconds: float = 6.0, fps: int = 60, cm: bool = True,
                       drift_deg_per_s: float = 0.0) -> str:
    """A straight-legged-stance walk: each hip sweeps from -amp to +amp about the lateral axis (BVH X; negative
    is flexion, the foot ahead) over its stance half and swings back with the knee bent; the legs are half a
    cycle apart, the ankles keep the feet level and the arms swing against the legs. The root follows the
    stance foot, so that foot stays where it landed. Stride = 4 * leg * sin(amp); the cadence makes `speed`.
    With a drift, the left hip turns sideways (BVH Z) at that rate through the clip -- a walk whose cycles do
    not end where they began."""
    return synthetic_walk_segments_bvh([(speed, seconds, 0.0)], fps=fps, cm=cm, drift_deg_per_s=drift_deg_per_s)


def synthetic_walk_segments_bvh(segments, fps: int = 60, cm: bool = True, drift_deg_per_s: float = 0.0) -> str:
    """The same walk through segments of (speed m/s, seconds, turn deg/s). The gait's phase carries over from one
    segment to the next; while a segment turns, the root yaws at its rate (BVH Y rotation) and travels along its
    heading, so the stance foot slides a little (yaw rate times its offset from the root)."""
    leg, amp, knee_max = 0.84, math.radians(18.0), math.radians(45.0)
    n, scale = sum(int(seconds * fps) for _, seconds, _ in segments), (100.0 if cm else 1.0)

    def angles(phi):
        """(hip, knee) of a leg whose stance starts at phi = 0."""
        phi %= 1.0
        if phi < 0.5:
            return -amp + 4.0 * amp * phi, 0.0
        return amp - 4.0 * amp * (phi - 0.5), knee_max * math.sin(2.0 * math.pi * (phi - 0.5))

    rows, foot, stance_prev, root_z = [], 0.0, None, 0.0
    phi0, yaw0, x, z, turned = 0.0, 0.0, 0.0, 0.0, False
    frames = []
    for speed, seconds, turn in segments:
        period = 4.0 * leg * math.sin(amp) / speed
        m = int(seconds * fps)
        frames += [(phi0, i, period, yaw0 + math.radians(turn) * i / fps) for i in range(m)]
        phi0, yaw0 = phi0 + m / fps / period, yaw0 + math.radians(turn) * m / fps
    for k, (phi_start, i, period, yaw) in enumerate(frames):
        phi = (phi_start + i / fps / period) % 1.0
        hr, kr = angles(phi)
        hl, kl = angles(phi + 0.5)
        stance = "R" if phi < 0.5 else "L"
        hip = hr if stance == "R" else hl
        if stance != stance_prev:
            foot = root_z + leg * math.sin(-hip)        # the new stance foot lands ahead of the root
            stance_prev = stance
        before, root_z = root_z, foot - leg * math.sin(-hip)
        root_y = leg * math.cos(hip) + 0.07             # ankle 7 cm above the floor, sole flat
        if turned or yaw != 0.0:                        # from the first turn on, travel along the heading
            turned = True
            x, z = x + math.sin(yaw) * (root_z - before), z + math.cos(yaw) * (root_z - before)
        else:
            x, z = 0.0, root_z
        vals = [x * scale, root_y * scale, z * scale, 0.0, 0.0, math.degrees(yaw)]
        for name, _, _, _ in SKELETON[1:]:
            zr = xr = yr = 0.0
            if name == "LeftHip":
                xr, zr = math.degrees(hl), drift_deg_per_s * k / fps
            elif name == "RightHip":
                xr = math.degrees(hr)
            elif name == "LeftKnee":
                xr = math.degrees(kl)
            elif name == "RightKnee":
                xr = math.degrees(kr)
            elif name == "LeftAnkle":
                xr = -math.degrees(hl + kl)              # keep the left foot level
            elif name == "RightAnkle":
                xr = -math.degrees(hr + kr)
            elif name == "LeftShoulder":
                zr, xr = -80.0, -0.6 * math.degrees(hl)  # T-pose arm brought down, swinging against the leg
            elif name == "RightShoulder":
                zr, xr = 80.0, -0.6 * math.degrees(hr)
            elif name == "LeftElbow":
                yr = 15.0
            elif name == "RightElbow":
                yr = -15.0
            vals += [zr, xr, yr]
        rows.append(" ".join(f"{v:.5f}" for v in vals))
    return "\n".join([_hierarchy(), "MOTION", f"Frames: {n}", f"Frame Time: {1.0 / fps:.8f}"] + rows) + "\n"
