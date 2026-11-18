"""generation/dummy/gait/support.py -- the height the hold servo's partial weight support aims the jointed dummy's
root at: the pelvis height under the joint targets the gait sends, the formula the gait build smooths
(retarget.pelvis_path), with the targets taken by their degree-of-freedom names. Pure; runs in CI."""
import numpy as np
import pytest

from generation.dummy import joints_spec as js
from generation.dummy.gait import retarget as rt
from generation.dummy.gait import support

SPEC = js.load()
NAMES = js.dof_names(SPEC)


def _crouch(angle_rad: float) -> np.ndarray:
    """Thighs forward by the angle, shanks back by twice it, feet back to flat: both soles stay level and the pelvis
    comes down (+y about the standing body's axes sends a hanging limb backward -- joints.yaml)."""
    row = np.zeros(len(NAMES))
    for side in ("left", "right"):
        row[NAMES.index(f"hip_{side}:1")] = -angle_rad
        row[NAMES.index(f"knee_{side}")] = 2.0 * angle_rad
        row[NAMES.index(f"ankle_{side}:1")] = -angle_rad
    return row


def test_the_stand_pose_rides_at_the_stand_height():
    assert support.pelvis_rise(SPEC, NAMES, np.zeros((1, len(NAMES)))) == pytest.approx([0.0], abs=1e-12)


def test_a_crouch_lowers_the_pelvis_by_the_builds_own_path():
    rows = np.stack([_crouch(0.2), _crouch(0.4)])
    got = support.pelvis_rise(SPEC, NAMES, rows)
    assert np.all(got < 0.0) and got[1] < got[0]
    np.testing.assert_allclose(got, rt.pelvis_path(rows, SPEC), rtol=0, atol=0)


def test_the_targets_are_read_by_name_whatever_their_column_order():
    rows = np.stack([_crouch(0.3)])
    rows[0, NAMES.index("hip_left:0")] = 0.1     # left and right differ, so a swapped column would show
    order = np.random.default_rng(0).permutation(len(NAMES))
    shuffled = [NAMES[i] for i in order]
    np.testing.assert_allclose(support.pelvis_rise(SPEC, shuffled, rows[:, order]),
                               support.pelvis_rise(SPEC, NAMES, rows), rtol=0, atol=0)


@pytest.mark.parametrize("names", [NAMES[:-1], NAMES[:-1] + ["not_a_joint"], NAMES[:-1] + [NAMES[0]]])
def test_names_that_are_not_the_specs_degrees_of_freedom_are_refused(names):
    with pytest.raises(ValueError, match="명세의 DOF"):
        support.pelvis_rise(SPEC, names, np.zeros((1, len(names))))


def test_a_targets_width_that_is_not_the_names_is_refused():
    with pytest.raises(ValueError):
        support.pelvis_rise(SPEC, NAMES, np.zeros((1, len(NAMES) - 1)))


# --- the stance a walk ends in ---------------------------------------------------------------------------------------

from generation.dummy.gait import kinematics as kin  # noqa: E402

GEOMS = {s: js.leg_geometry(SPEC, s) for s in ("left", "right")}


def _soles(row):
    return {s: kin.leg_fk(GEOMS[s], *rt.leg_matrices(np.asarray(row, float), SPEC, s))[1] for s in GEOMS}


def _mid_step() -> np.ndarray:
    """A walking pose: the left leg forward with its heel down and toe up, the right behind with its heel off the floor
    and the foot lifted by a bent knee; an arm swung."""
    row = np.zeros(len(NAMES))
    row[NAMES.index("hip_left:1")] = -0.25
    row[NAMES.index("knee_left")] = 0.1
    row[NAMES.index("ankle_left:1")] = 0.1
    row[NAMES.index("hip_right:1")] = 0.15
    row[NAMES.index("knee_right")] = 0.4
    row[NAMES.index("ankle_right:1")] = -0.15
    row[NAMES.index("hip_left:0")] = 0.05
    row[NAMES.index("shoulder_left:1")] = 0.3
    return row


def test_the_stance_at_rest_puts_the_foot_in_the_air_down_to_the_other():
    row = _mid_step()
    before, after = _soles(row), _soles(support.rest_stance(SPEC, NAMES, row[None])[0])
    down, up = sorted(GEOMS, key=lambda s: before[s][:, 2].min())
    assert before[up][:, 2].min() - before[down][:, 2].min() > 0.01                   # one foot is in the air
    assert after[up][:, 2].min() == pytest.approx(after[down][:, 2].min(), abs=1e-3)
    # the leg in the air is straight: it cannot reach down so far, so the foot that is down comes up to meet it
    assert after[down][:, 2].min() > before[down][:, 2].min() + 0.005
    np.testing.assert_allclose(after[down].mean(axis=0)[:2], before[down].mean(axis=0)[:2], atol=0.02)
    # the foot comes down where it is (its orientation kept), so the feet stay a stride apart
    np.testing.assert_allclose(after[up].mean(axis=0)[:2], before[up].mean(axis=0)[:2], atol=0.02)
    assert abs(after["left"].mean(axis=0)[0] - after["right"].mean(axis=0)[0]) > 0.2


def test_the_stance_at_rest_brings_the_pelvis_down_by_what_the_feet_met_at():
    row = _mid_step()
    rest = support.rest_stance(SPEC, NAMES, row[None])
    low = min(_soles(rest[0])[s][:, 2].min() for s in GEOMS)
    assert support.pelvis_rise(SPEC, NAMES, rest)[0] == pytest.approx(-low, abs=1e-9)
    assert support.pelvis_rise(SPEC, NAMES, rest)[0] < support.pelvis_rise(SPEC, NAMES, row[None])[0]


def test_a_foot_in_the_air_that_the_leg_can_reach_comes_down_and_the_other_stays():
    # both legs crouched with flat feet, the right more: its foot is in the air, and the leg can reach down
    row = np.zeros(len(NAMES))
    for side, a in (("left", 0.2), ("right", 0.35)):
        row[NAMES.index(f"hip_{side}:1")] = -a
        row[NAMES.index(f"knee_{side}")] = 2 * a
        row[NAMES.index(f"ankle_{side}:1")] = -a
    before, after = _soles(row), _soles(support.rest_stance(SPEC, NAMES, row[None])[0])
    assert before["right"][:, 2].min() - before["left"][:, 2].min() > 0.01
    np.testing.assert_allclose(after["left"], before["left"], atol=1e-9)
    assert after["right"][:, 2].min() == pytest.approx(before["left"][:, 2].min(), abs=1e-3)


def test_the_stance_at_rest_leaves_what_is_not_a_leg_and_reads_by_name():
    row = _mid_step()
    rest = support.rest_stance(SPEC, NAMES, row[None])[0]
    legs = [i for i, n in enumerate(NAMES) if n.split(":")[0].rsplit("_", 1)[0] in ("hip", "knee", "ankle")]
    others = [i for i in range(len(NAMES)) if i not in legs]
    assert np.array_equal(rest[others], row[others])
    order = np.random.default_rng(1).permutation(len(NAMES))
    shuffled = support.rest_stance(SPEC, [NAMES[i] for i in order], row[None][:, order])[0]
    np.testing.assert_allclose(shuffled, rest[order], atol=1e-12)


def test_the_stance_at_rest_stays_in_the_joint_ranges():
    rest = support.rest_stance(SPEC, NAMES, _mid_step()[None])
    assert not rt.clamp(rest, SPEC)[1]


def test_the_stand_pose_rests_as_it_is():
    np.testing.assert_allclose(support.rest_stance(SPEC, NAMES, np.zeros((1, len(NAMES)))), 0.0, atol=1e-9)
