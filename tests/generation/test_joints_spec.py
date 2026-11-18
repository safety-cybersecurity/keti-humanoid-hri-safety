"""generation/dummy/joints_spec.py — the jointed dummy's spec: pivots from the capsules, feet from the
stature, masses from the segment fractions, names for the degrees of freedom. Pure; runs in CI."""
import copy
import math

import numpy as np
import pytest
import yaml

from generation.dummy import joints_spec as js


@pytest.fixture(scope="module")
def spec():
    return js.load()


def test_twenty_two_dofs_ten_joints_left_then_right(spec):
    assert [j.name for j in spec.joints] == [
        "hip_left", "hip_right", "knee_left", "knee_right", "ankle_left", "ankle_right",
        "shoulder_left", "shoulder_right", "elbow_left", "elbow_right"]
    assert sum(len(j.dofs) for j in spec.joints) == 22
    assert [j.kind for j in spec.joints[:6]] == ["d6", "d6", "revolute", "revolute", "d6", "d6"]


def test_the_ankle_turns_about_three_axes_with_a_narrow_third(spec):
    for side in ("left", "right"):
        ankle = js.joint(spec, f"ankle_{side}")
        assert js.axes(ankle) == "xyz"
        by = {d.axis: d for d in ankle.dofs}
        assert by["z"].lower < 0.0 < by["z"].upper
        assert by["z"].upper - by["z"].lower < min(by[a].upper - by[a].lower for a in "xy")


def test_a_two_axis_joint_is_refused(tmp_path):
    raw = yaml.safe_load(js.SPEC_PATH.read_text(encoding="utf-8"))
    del raw["joints"]["ankle"]["dofs"]["z"]
    p = tmp_path / "joints.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="세 축"):
        js.load(p)


def test_pivots_are_the_capsule_ends(spec):
    hip = js.joint(spec, "hip_left")
    assert hip.pivot == pytest.approx((0.0, 0.09, 0.89))          # BR-11: centre 0.70, height 0.38
    knee = js.joint(spec, "knee_right")
    assert knee.pivot == pytest.approx((0.0, -0.09, 0.495))       # (0.51 + 0.48) / 2
    assert js.joint(spec, "ankle_left").pivot == pytest.approx((0.0, 0.09, 0.10))
    assert js.joint(spec, "shoulder_right").pivot == pytest.approx((0.0, -0.21, 1.42))
    # BR-08 ends at z 1.16 (centre 1.29, height 0.26), BR-09 starts at 1.15 (centre 1.03, height 0.24)
    assert js.joint(spec, "elbow_left").pivot == pytest.approx((0.0, 0.2125, 1.155))


def test_hip_above_root_is_the_hip_pivot_less_the_root_centre(spec):
    assert js.hip_above_root(spec) == pytest.approx(0.89 - 1.43)


def test_bodies_on_each_side(spec):
    assert js.joint(spec, "hip_left").parent == "BR_07"
    assert js.joint(spec, "hip_left").child == "BR_11_left"
    assert js.joint(spec, "ankle_right").child == "FOOT_right"
    assert js.joint(spec, "elbow_right").parent == "BR_08_right"


def test_right_ranges_mirror_the_left(spec):
    left, right = js.joint(spec, "hip_left"), js.joint(spec, "hip_right")
    by = {d.axis: d for d in right.dofs}
    lby = {d.axis: d for d in left.dofs}
    assert (by["x"].lower, by["x"].upper) == pytest.approx((-lby["x"].upper, -lby["x"].lower))
    assert (by["y"].lower, by["y"].upper) == pytest.approx((lby["y"].lower, lby["y"].upper))
    assert lby["y"].lower == pytest.approx(math.radians(-120))


def test_feet_sit_under_the_ankle_on_the_floor(spec):
    foot = next(f for f in spec.feet if f.side == "left")
    L, W, H = foot.size
    assert L == pytest.approx(0.152 * 1.74) and W == pytest.approx(0.055 * 1.74) and H == pytest.approx(0.039 * 1.74)
    assert foot.center[2] - H / 2 == pytest.approx(0.0)            # sole on the spec's floor
    ankle = js.joint(spec, "ankle_left").pivot
    assert foot.center[0] - L / 2 == pytest.approx(ankle[0] - 0.25 * L)   # heel behind the ankle
    assert H < ankle[2]


def test_masses_follow_the_fractions(spec):
    assert sum(spec.mass.values()) == pytest.approx(75.0)
    assert spec.mass["BR_11_left"] == pytest.approx(75.0 * 0.1416)
    assert spec.mass["FOOT_right"] == pytest.approx(75.0 * 0.0137)
    head = spec.mass["BR_01"] + spec.mass["BR_02"] + spec.mass["BR_03"]
    assert head == pytest.approx(75.0 * 0.0694)
    assert spec.density["BR_01"] == pytest.approx(spec.density["BR_03"])   # one density per group
    assert len(spec.density) == 19                                         # 17 region prims + 2 feet


def test_percentages_must_add_up(tmp_path):
    raw = yaml.safe_load(js.SPEC_PATH.read_text(encoding="utf-8"))
    raw["mass"]["groups"]["thigh"]["percent"] = 15.0
    p = tmp_path / "joints.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="100"):
        js.load(p)


def test_an_unknown_pivot_rule_is_refused(tmp_path):
    raw = yaml.safe_load(js.SPEC_PATH.read_text(encoding="utf-8"))
    raw["joints"]["knee"]["pivot"] = {"middle": "BR-11"}
    p = tmp_path / "joints.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="피벗"):
        js.load(p)


def test_dof_names_use_the_measured_pattern(spec):
    names = js.dof_names(spec)
    assert names[:4] == ["hip_left:0", "hip_left:1", "hip_left:2", "hip_right:0"]
    assert "knee_left" in names and "elbow_right" in names and "ankle_right:2" in names
    assert len(names) == len(set(names)) == 22


def test_drives_cover_every_group_in_both_modes(spec):
    for mode in ("walking", "tone"):
        assert set(spec.drives[mode]) == {"hip", "knee", "ankle", "shoulder", "elbow"}
        assert all(k > 0 and d > 0 for k, d in spec.drives[mode].values())


def test_geometry_with_feet_stands_on_the_feet(spec):
    from generation.envs.s1d import aisle_guard
    g = js.geometry_with_feet(spec)
    assert aisle_guard.standing_root_height(g) == pytest.approx(1.43)
    assert aisle_guard.standing_root_height(spec.anthropometry) == pytest.approx(1.385)  # v1 unchanged
    assert js.FOOT in g["regions"] and js.FOOT not in spec.anthropometry["regions"]


def test_foot_primitives_carry_the_segmentation_class(spec):
    prims = js.foot_primitives(spec)
    assert [p.name for p in prims] == ["FOOT_left", "FOOT_right"]
    assert {p.code for p in prims} == {"BR-12"}


def test_upper_body_excludes_legs_and_feet(spec):
    upper = set(js.upper_body_prims(spec))
    assert "BR_04" in upper and "BR_08_left" in upper
    assert not upper & {"BR_11_left", "BR_12_right", "FOOT_left"}
    assert len(js.region_prims(spec)) == 17
