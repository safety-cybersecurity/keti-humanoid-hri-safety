"""generation/envs/s1a/recorder.py and run_s1a.py with the jointed dummy's feet. The recorder imports Isaac, so the
lines that take the feet out of the body regions and make them the standing load's channel are pinned from the
source (safety.dummy_body_masks itself is tested in test_dummy_body_masks). Isaac-free."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDER = (ROOT / "generation/envs/s1a/recorder.py").read_text(encoding="utf-8")
RUN = (ROOT / "generation/envs/run_s1a.py").read_text(encoding="utf-8")


def test_the_recorder_reduces_regions_over_region_bodies_only():
    assert "self._region_mask, self._ground_mask = dummy_body_masks(region_codes)" in RECORDER
    assert "self._codes = sorted({c for c in region_codes if c is not None})" in RECORDER
    assert ("        by_region, by_region_z, by_region_robot = reduce_region_peaks(\n"
            "            [c for c in self.region_codes if c is not None], forces[rm], peak[\"dummy_z_at_peak\"][rm],\n"
            "            peak[\"robot_gate_at_peak\"][rm], contact_eps_n=_CONTACT_EPS_N)\n") in RECORDER


def test_the_ground_code_is_the_safety_modules():
    assert "_GROUND_REGION_CODE = GROUND_REGION_CODE" in RECORDER and '_GROUND_REGION_CODE = "BR-12"' not in RECORDER


def test_run_s1a_reads_feet_as_ground_bodies():
    assert 'codes = jointed.read_body_codes(dummy_usd, list(base_env.scene["dummy_contact"].body_names))' in RUN
    assert "read_region_codes(dummy_usd" not in RUN


# The dummy's contact sensor (forces, region codes, masks) and its articulation (centre-of-mass poses and
# velocities) list the bodies in different orders; the recorder reads the articulation's arrays into the sensor's
# order by name before reducing them over the sensor's region codes.


def test_the_recorder_pairs_the_dummys_bodies_by_name_once():
    assert "self._dummy_order: np.ndarray | None = None" in RECORDER
    assert ("        if self._dummy_order is None:\n"
            "            self._dummy_order = substep_contact.sensor_order(contact.body_names, dummy.body_names)\n") in RECORDER


def test_the_relative_speed_is_taken_in_the_sensors_body_order():
    assert ("            contact_relative_speed(\n"
            "                dummy.data.body_com_pose_w.torch[0, :, :3].cpu().numpy()[self._dummy_order],\n"
            "                dummy.data.body_com_vel_w.torch[0].cpu().numpy()[self._dummy_order],\n") in RECORDER
    assert RECORDER.index("self._dummy_order = substep_contact.sensor_order(") < RECORDER.index("contact_relative_speed(\n")


def test_no_comment_says_the_two_orders_are_the_same():
    assert "body order is the articulation's own" not in RECORDER
    assert "refuses a dummy" not in RECORDER
