"""Per-episode seed derivation. No Isaac Sim.

meta/scenarios.json's seed_rule is the canon: `seed = hash(scenario_id,
episode_seq)`, one seed per episode. A seed drawn per process instead would
make the same episode index carry a different number in every run, so nothing
downstream could line the two up.

What this does NOT establish is that a lone episode reproduces. It does not
(generation/README.md), and the canon no longer claims it -- simulator state
crosses the episode boundary. The seed is per episode regardless.
"""

import subprocess
import sys
from pathlib import Path

from generation.seeding import episode_seed, scenario_startup_seed

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_same_episode_always_derives_the_same_seed():
    assert episode_seed("S1a", 7) == episode_seed("S1a", 7)


def test_the_seed_survives_a_fresh_interpreter():
    # Python's built-in hash() is salted per process, so a seed built on it
    # would differ between the run that produced an episode and the run that
    # tries to reproduce it -- silently, since both look like plausible ints.
    code = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});"
        "from generation.seeding import episode_seed;"
        "print(episode_seed('S1a', 7))"
    )
    first = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    second = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert first.stdout.strip() == second.stdout.strip()
    assert int(first.stdout.strip()) == episode_seed("S1a", 7)


def test_neighbouring_episodes_do_not_share_a_seed():
    seeds = {episode_seed("S1a", seq) for seq in range(64)}
    assert len(seeds) == 64


def test_the_same_sequence_number_in_another_scenario_is_another_seed():
    # scenario_id is half the key precisely so that S1a's episode 3 and S2a's
    # episode 3 are not the same draw.
    assert episode_seed("S1a", 3) != episode_seed("S2a", 3)


def test_the_seed_fits_the_range_the_generators_accept():
    # numpy's legacy seeding refuses anything >= 2**32, and Isaac Lab's
    # seeding path feeds the same value to numpy and torch alike.
    for seq in range(256):
        seed = episode_seed("S1a", seq)
        assert isinstance(seed, int)
        assert 0 <= seed < 2**32


def test_the_seed_is_derived_rather_than_the_sequence_number_itself():
    # A run-seed-plus-index scheme would make episode_seed("S1a", 0) == 0 and
    # reintroduce exactly the contract this replaces.
    assert episode_seed("S1a", 0) != 0


def test_the_startup_seed_does_not_move_with_the_first_episode_number():
    # Isaac Lab's mode="startup" randomisation (static friction, base mass)
    # is drawn once when the scene is built, not at each reset, so it cannot
    # belong to an episode. Deriving it from whichever episode a batch
    # happened to start at made `--seq 0 --episodes 3` and `--seq 2
    # --episodes 1` build two different robots -- measured 2026-08-20:
    # robot_base_mass_kg_realized 16.596 vs 15.167 for the same episode.
    assert scenario_startup_seed("S1a") == scenario_startup_seed("S1a")


def test_the_startup_seed_is_a_property_of_the_scenario():
    assert scenario_startup_seed("S1a") != scenario_startup_seed("S2a")


def test_the_startup_seed_is_not_any_episode_seed():
    # Sharing a value with an episode seed would make the startup draws and
    # that episode's reset draws come off the same stream position.
    startup = scenario_startup_seed("S1a")
    assert startup not in {episode_seed("S1a", seq) for seq in range(1024)}


def test_the_startup_seed_fits_the_range_the_generators_accept():
    for scenario in ("S1a", "S1b", "S2a", "S3a"):
        assert 0 <= scenario_startup_seed(scenario) < 2**32
