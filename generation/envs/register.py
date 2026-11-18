"""Gym registration for the KETI safety scenarios.

Imported explicitly by the entry point, never from __init__.py: the unit
tests import this package under a venv that has neither gymnasium nor
Isaac Lab.
"""

import gymnasium as gym

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1 import agents

from generation import conditions

# The published checkpoint is an H1 flat-velocity policy, so the agent config
# has to be the one it was trained with -- for every scenario.
AGENT_CFG_ENTRY_POINT = f"{agents.__name__}.rsl_rl_ppo_cfg:H1FlatPPORunnerCfg"

gym.register(
    id="Keti-S1a-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "generation.envs.s1a.s1a_env_cfg:S1aEnvCfg",
        "rsl_rl_cfg_entry_point": AGENT_CFG_ENTRY_POINT,
    },
)

# The scenarios with a conditions table (meta/scenarios.json): one config class per scenario in
# generation/envs/hri/hri_env_cfg.py, at the table's row 0 (run_s1a.py builds any other row itself).
for _scenario in conditions.table_scenarios():
    gym.register(
        id=f"Keti-{_scenario}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"generation.envs.hri.hri_env_cfg:{_scenario}EnvCfg",
            "rsl_rl_cfg_entry_point": AGENT_CFG_ENTRY_POINT,
        },
    )
