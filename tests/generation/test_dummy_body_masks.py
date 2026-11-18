"""safety.dummy_body_masks -- which of the dummy's contact-sensor bodies are body regions and which carry
its standing load. With no foot bodies the answer is the rigid dummy's: BR-12 stands on the floor."""
import numpy as np

from generation.envs.s1a.mdp.safety import dummy_body_masks

V1 = ["BR-04", "BR-03", "BR-01", "BR-02", "BR-05", "BR-06", "BR-07", "BR-08", "BR-08", "BR-09", "BR-09",
      "BR-10", "BR-10", "BR-11", "BR-11", "BR-12", "BR-12"]


def test_rigid_dummy_masks_are_the_old_ones():
    region, ground = dummy_body_masks(V1)
    assert region.all()
    assert (ground == np.array([c == "BR-12" for c in V1])).all()


def test_feet_are_ground_and_not_regions():
    codes = V1 + [None, None]
    region, ground = dummy_body_masks(codes)
    assert (region == np.array([c is not None for c in codes])).all()
    assert (ground == np.array([c is None for c in codes])).all()


def test_masking_with_all_true_leaves_the_arrays_bit_identical():
    rng = np.random.default_rng(0)
    forces = rng.random(len(V1))
    region, _ = dummy_body_masks(V1)
    assert forces[region].tobytes() == forces.tobytes()


# What contact_onset cannot see goes into each episode's metadata. With the rigid dummy the shins (BR-12) are
# its ground channel and the text is the one the S1d batch of 400 carries, to the byte. With the jointed
# dummy the feet are -- the shins are then in the dummy-side gate, and what goes unseen is a foot on a foot.

# SHA-256 of the reason every episode of the S1d batch of 400 carries (all 400 read alike).
V1_REASON_SHA256 = "3fc9fba73287e59d91e509905de0f5efb2d07a338b5582d57df6df1e03f43fc6"


def test_the_rigid_dummys_onset_gap_is_the_s1d_batchs_to_the_byte():
    from generation.envs.s1a.mdp.safety import contact_onset_gap

    import hashlib

    regions, reason = contact_onset_gap(V1)
    assert regions == ["BR-12"] and hashlib.sha256(reason.encode("utf-8")).hexdigest() == V1_REASON_SHA256


def test_the_jointed_dummys_onset_gap_is_its_feet_not_a_region():
    from generation.envs.s1a.mdp.safety import contact_onset_gap

    regions, reason = contact_onset_gap(V1 + [None, None])
    assert regions == []
    assert "발" in reason and "BR-12" in reason and reason != contact_onset_gap(V1)[1]


def test_the_recorder_writes_the_gap_of_its_own_dummy():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "generation/envs/s1a/recorder.py").read_text(encoding="utf-8")
    assert "contact_onset_gap(self.region_codes)" in src
    assert '"contact_onset_unresolved_regions": ["BR-12"]' not in src
