"""What the engine does with a D6 joint, decided from d6_convention_measured.json (written by
generation/dummy/probe_d6.py): which axis each degree of freedom turns about, the quaternion order of
its body poses and the convention of its joint positions -- and joints.yaml holding the same answer.

The probe's joints: "tri" opens all three rotation axes; "ankle" too, with its third axis in a narrow range
and driven to zero, as the dummy's ankle; "hinge" is a revolute joint; "duo" is a D6 with rotZ locked,
kept as the measurement of why every multi-axis joint of the dummy opens all three axes."""
import json
import math
from pathlib import Path

import numpy as np

from generation.dummy import joints_spec as js
from generation.dummy.gait import kinematics as kin

MEASURED = json.loads((Path(__file__).parent / "d6_convention_measured.json").read_text(encoding="utf-8"))
ORDERS = ("wxyz", "xyzw")
JOINT_AXES = {"tri": "xyz", "ankle": "xyz", "hinge": "y"}     # the joints whose positions follow a convention
THREE_AXIS = ("tri", "ankle")
FIT = 1e-4            # rad: a convention fits a case when it reproduces the measured rotation this closely
HELD = 1e-3           # rad: how close to its target a driven or limited axis counts as held
JOINT_OF = {n: c["joint"] for c in MEASURED["cases"] for n in c["joint_pos"]}


def _matrix(q, order):
    w, x, y, z = q if order == "wxyz" else (q[3], q[0], q[1], q[2])
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _angle(R):
    # from the trace: the wrong quaternion order can read as a half turn, which has no rotation vector
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def _cases(kind, joints):
    return [c for c in MEASURED["cases"] if c["kind"] == kind and c["joint"] in joints]


def _order():
    """The order in which a single degree of freedom's rotation angle equals its joint position."""
    fits = []
    for order in ORDERS:
        if all(abs(_angle(_matrix(c["relative_quat"], order)) - abs(c["joint_pos"][n])) < FIT
               for c in _cases("single", JOINT_AXES) for n in c["targets"]):
            fits.append(order)
    assert len(fits) == 1, fits
    return fits[0]


def _axes(joints=tuple(JOINT_AXES), clean=True):
    """{dof name: (axis, sign)} -- the body axis a degree of freedom turns about, and its sense. ``clean``
    requires the single case to turn about that axis alone."""
    order, out = _order(), {}
    for c in _cases("single", joints):
        (name, value), = c["targets"].items()
        rv = kin.matrix_to_rotvec(_matrix(c["relative_quat"], order))
        k = int(np.argmax(np.abs(rv)))
        assert not clean or abs(rv[k]) > 0.999 * np.linalg.norm(rv), c
        out[name] = ("xyz"[k], float(np.sign(rv[k]) * np.sign(value)))
    return out


def _worst(convention, cases, axes):
    """The largest angle between a measured rotation and the one ``convention`` makes of its positions."""
    order, worst = _order(), 0.0
    for c in cases:
        by_axis = {axes[n][0]: axes[n][1] * v for n, v in c["joint_pos"].items()}
        letters = "".join(a for a in "xyz" if a in by_axis)
        want = kin.dofs_to_matrix([by_axis[a] for a in letters], letters, convention)
        worst = max(worst, _angle(_matrix(c["relative_quat"], order).T @ want))
    return worst


def _named(joint, axis):
    return next(n for n, (a, _) in _axes().items() if JOINT_OF[n] == joint and a == axis)


def test_every_degree_of_freedom_turns_positively_about_its_own_axis():
    axes = _axes()
    assert all(sign == 1.0 for _, sign in axes.values()), axes
    for joint, letters in JOINT_AXES.items():
        assert sorted(a for n, (a, _) in axes.items() if JOINT_OF[n] == joint) == sorted(letters)


def test_exactly_one_convention_fits_every_three_axis_case():
    cases = _cases("combined", THREE_AXIS) + _cases("past_limit", THREE_AXIS)
    assert {c["joint"] for c in cases} == set(THREE_AXIS)
    errors = {c: _worst(c, cases, _axes()) for c in kin.CONVENTIONS}
    assert sum(e < FIT for e in errors.values()) == 1, errors


def test_the_ankles_third_axis_stays_at_zero_while_the_others_move():
    z, order = _named("ankle", "z"), _order()
    cases = _cases("combined", ("ankle",))
    assert len(cases) == 3
    for c in cases:
        assert c["targets"][z] == 0.0
        assert all(abs(c["joint_pos"][n] - v) < HELD for n, v in c["targets"].items()), c
        assert abs(kin.matrix_to_rotvec(_matrix(c["relative_quat"], order))[2]) < HELD    # no turn about the shank


def test_a_narrow_axis_stops_at_its_limit_in_the_same_coordinate():
    """The range bounds the same rotation-vector component the position reports. Pressed against it, the
    narrow axis also pushes the other two off their targets (0.04 and 0.06 rad here, with the drive aiming
    0.33 rad past the limit at 1e5 N m/rad) -- the dummy's ankle meets its z limit only under a large yaw
    torque, and this test does not hold the engine to that amount."""
    z, axes = _named("ankle", "z"), _axes()
    lower, upper = MEASURED["joint_pos_limits"][z]
    assert math.isclose(lower, -upper) and 0.0 < upper < math.radians(30.0)
    cases = _cases("past_limit", ("ankle",))
    assert len(cases) == 1 and cases[0]["targets"][z] > upper
    assert abs(cases[0]["joint_pos"][z] - upper) < HELD, cases[0]
    assert _worst(js.load().d6_convention, cases, axes) < FIT


def test_a_two_axis_d6_fits_no_convention():
    """Why the dummy's ankle opens all three axes: with rotZ locked the link still turns about z, and no
    convention reproduces its rotation from the two positions the engine reports."""
    cases = _cases("combined", ("duo",))
    assert len(cases) == 2
    axes = _axes(("duo",), clean=False)
    assert sorted(a for a, _ in axes.values()) == ["x", "y"]
    errors = {c: _worst(c, cases, axes) for c in kin.CONVENTIONS}
    assert min(errors.values()) > FIT, errors


def test_joints_yaml_holds_the_measured_convention():
    cases = _cases("combined", THREE_AXIS)
    assert _worst(js.load().d6_convention, cases, _axes()) < FIT


def test_joints_yaml_names_the_axes_as_the_engine_does():
    spec, axes = js.load(), _axes()
    for name, (axis, _) in axes.items():
        joint = JOINT_OF[name]
        if joint == "hinge":
            assert name == "hinge"                       # a revolute joint's degree of freedom is the joint's name
            continue
        index = [n for n in MEASURED["joint_names"] if JOINT_OF.get(n) == joint].index(name)
        assert index == "xyz".index(axis)                # a three-axis joint reports x, y, z in that order
        assert name == spec.dof_name.format(joint=joint, axis=axis, index=index)
