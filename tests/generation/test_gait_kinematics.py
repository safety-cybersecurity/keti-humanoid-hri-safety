"""generation/dummy/gait/kinematics.py — rotations and leg kinematics. Pure; runs in CI."""
import math

import numpy as np
import pytest

from generation.dummy.gait import kinematics as kin


def test_axis_rotation_sends_a_hanging_limb_backward_about_plus_y():
    down = np.array([0.0, 0.0, -1.0])
    assert kin.axis_rotation("y", math.radians(30)) @ down == pytest.approx([-0.5, 0.0, -math.sqrt(3) / 2])


def test_axis_rotation_broadcasts():
    R = kin.axis_rotation("z", np.array([0.0, math.pi / 2]))
    assert R.shape == (2, 3, 3)
    assert R[1] @ np.array([1.0, 0, 0]) == pytest.approx([0.0, 1.0, 0.0])


@pytest.mark.parametrize("rv", [[0, 0, 0], [1e-10, 0, 0], [0.3, -0.2, 0.5], [0, 2.5, 0], [-1.0, 1.0, 1.0]])
def test_rotvec_round_trip(rv):
    rv = np.array(rv, float)
    R = kin.rotvec_to_matrix(rv)
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert kin.matrix_to_rotvec(R) == pytest.approx(rv, abs=1e-9)


def test_rotvec_about_one_axis_is_that_axis_rotation():
    assert kin.rotvec_to_matrix([0.0, 0.7, 0.0]) == pytest.approx(kin.axis_rotation("y", 0.7))


def test_matrix_to_rotvec_refuses_a_half_turn():
    with pytest.raises(ValueError, match="180"):
        kin.matrix_to_rotvec(kin.axis_rotation("x", math.pi))


def test_euler_is_intrinsic_in_the_order_given():
    a = [0.1, 0.2, 0.3]
    R = kin.euler_to_matrix("xyz", a)
    assert R == pytest.approx(kin.axis_rotation("x", 0.1) @ kin.axis_rotation("y", 0.2) @ kin.axis_rotation("z", 0.3))


def test_dofs_fill_missing_axes_with_zero():
    R = kin.dofs_to_matrix([0.2, -0.4], "xy", "rotvec")
    assert R == pytest.approx(kin.rotvec_to_matrix([0.2, -0.4, 0.0]))
    assert kin.matrix_to_dofs(R, "xy", "rotvec") == pytest.approx([0.2, -0.4])


def test_unknown_convention_is_refused():
    with pytest.raises(ValueError):
        kin.dofs_to_matrix([0.1], "y", "quaternion")


def test_rotation_between_takes_u_to_v():
    u, v = np.array([0.0, 0.0, -1.0]), np.array([0.3, 0.2, -0.9])
    R = kin.rotation_between(u, v)
    assert R @ u == pytest.approx(v / np.linalg.norm(v))
    assert kin.rotation_between(u, u) == pytest.approx(np.eye(3))



def test_swing_keeps_the_axis_image_and_drops_the_twist():
    swing = np.array([0.2, -0.3, 0.0])
    R = kin.rotvec_to_matrix(swing) @ kin.axis_rotation("z", 0.4)
    out = kin.swing_rotvec(R)
    assert out == pytest.approx(swing, abs=1e-12)
    assert out[2] == 0.0                                       # exactly: the twist axis's component
    assert kin.rotvec_to_matrix(out)[:, 2] == pytest.approx(R[:, 2], abs=1e-12)
    assert kin.swing_rotvec(kin.axis_rotation("z", 0.7)) == pytest.approx(np.zeros(3), abs=1e-15)


def test_swing_broadcasts():
    R = np.stack([kin.rotvec_to_matrix([0.1, 0.0, 0.0]), kin.rotvec_to_matrix([0.0, 0.3, 0.2])])
    out = kin.swing_rotvec(R)
    assert out.shape == (2, 3) and (out[:, 2] == 0.0).all()
    assert out[0] == pytest.approx([0.1, 0.0, 0.0])

def _geom():
    sole = np.array([[0.2, 0.14, 0.0], [0.2, 0.04, 0.0], [-0.06, 0.14, 0.0], [-0.06, 0.04, 0.0]])
    return kin.LegGeometry(np.array([0, 0.09, 0.89]), np.array([0, 0.09, 0.495]), np.array([0, 0.09, 0.10]), sole)


def test_leg_fk_at_rest_is_the_stand_pose():
    g = _geom()
    ankle, sole, R_foot = kin.leg_fk(g, np.eye(3), 0.0, np.eye(3))
    assert ankle == pytest.approx(g.ankle)
    assert sole == pytest.approx(g.sole)
    assert R_foot == pytest.approx(np.eye(3))


def test_leg_fk_bent_knee_lifts_the_foot_backward():
    g = _geom()
    ankle, sole, _ = kin.leg_fk(g, np.eye(3), math.radians(90), np.eye(3))
    assert ankle == pytest.approx([-0.395, 0.09, 0.495])
    # the foot turns with the shank: its toe corners (0.2 m ahead of the ankle at rest) point straight down
    assert sole[:, 2].min() == pytest.approx(0.495 - 0.2)
    assert sole[:, 0].max() < 0.0
