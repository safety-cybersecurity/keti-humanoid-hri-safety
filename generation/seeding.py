"""Per-episode seed derivation.

``meta/scenarios.json`` declares the contract::

    seed = hash(scenario_id, episode_seq).

Two properties follow from it and neither is optional:

* the seed belongs to an **episode**, not to a run, so episode seven of a
  ten-episode run carries the same number whether the run stops at seven or
  goes on to ten;
* the derivation is **stable across processes**, so the run that produced an
  episode and the run that reproduces it agree on the number.

It does not follow that an episode can be rerun on its own:
measured on 2026-08-20, a lone ``--seq 7
--episodes 1`` diverges at step zero from the same episode inside a longer run,
with every recorded randomisation input identical. Reproduction holds per run;
the C9 hook replays the episode's own run from its first seq -- which the episode
carries as run_start_seq (schema §3) -- up to the episode. That is a property of
simulator state crossing the episode boundary, not of this function -- the seed
here is per episode either way.

Python's built-in ``hash()`` has the second property only for ints. For
strings it is salted per process (PYTHONHASHSEED), so a seed built on it
would differ between those two runs -- silently, because both values look
like perfectly ordinary integers. SHA-256 of the pair is what the canon's
"hash" has to mean here.
"""

from __future__ import annotations

import hashlib

# numpy's legacy seeding refuses anything >= 2**32 and Isaac Lab hands the
# same value to numpy and torch alike, so the digest is truncated to fit.
_SEED_MODULUS = 2**32


def episode_seed(scenario_id: str, episode_seq: int) -> int:
    """The seed for one episode of one scenario. Same inputs, same number, always."""
    if episode_seq < 0:
        raise ValueError(f"episode_seq 는 0 이상이어야 합니다: {episode_seq}")
    digest = hashlib.sha256(f"{scenario_id}:{episode_seq}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % _SEED_MODULUS


def scenario_startup_seed(scenario_id: str) -> int:
    """Seed for the draws made once when the scene is built, not per episode.

    Isaac Lab's ``mode="startup"`` events -- static friction and the robot's
    base mass, inherited from ``LocomotionVelocityRoughEnvCfg`` -- run once at
    construction and never again, so their realisations belong to the run, not
    to any episode. Deriving the construction seed from whichever episode a
    batch happened to start at therefore built a *different robot* depending
    on where the batch began: measured 2026-08-20, the same episode came out
    with ``robot_base_mass_kg_realized`` 16.596 kg inside ``--seq 0 --episodes
    3`` and 15.167 kg from ``--seq 2 --episodes 1``, and every recorded array
    diverged from step 0 as a result.

    A constant per scenario costs nothing -- those values were already fixed
    for a whole run either way -- and it takes where a batch started out of
    what gets built. That is not enough to rerun a lone episode: see the
    module docstring.

    Whether a quantity varies per episode is a separate matter, settled by its
    event's mode and not by this seed: S1aEnvCfg sets the base mass event to
    ``mode="reset"``, so the mass is drawn per episode from the episode's own
    seed, and static friction stays at startup with a degenerate range (it
    draws a constant).
    """
    digest = hashlib.sha256(f"{scenario_id}:startup".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % _SEED_MODULUS
