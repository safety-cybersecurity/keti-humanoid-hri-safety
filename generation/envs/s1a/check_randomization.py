#!/usr/bin/env python3
"""What each scenario with an environment randomises, read off its resolved config (~/isaacsim/python.sh).

    ~/isaacsim/python.sh generation/envs/s1a/check_randomization.py           # check the canon
    ~/isaacsim/python.sh generation/envs/s1a/check_randomization.py --write   # write the blocks

meta/scenarios.json says, for a scenario that has an environment, which axes are drawn per episode
(randomization), which of them are not drawn uniformly (randomization_distribution), which values a
reader may expect to vary are held fixed (randomization_fixed), and what noise the walking policy's
observations get (policy_observation_noise). Those blocks are this script's output, not typed by hand:
from August to 2026-10-03 the canon declared a floor friction of 0.3-1.0 and a mass deviation of +-5 %
that no episode was ever generated with (generation/randomization.py).

The scenarios with a conditions table (S1e, S1f, S1g) are built at row 0, the way run_s1a.py builds a row's
config. What a row sets is not a draw and is not read (generation/randomization.py FROM_THE_ROW), so the row
chosen does not change the blocks.

The configs are built the way run_s1a.py builds them (resolve_presets, physx) and read; no environment
is created and nothing is simulated (about 20 s, most of it the boot). Run it after touching an
environment config, a command config or the Isaac Lab checkout -- most of what is read here is
inherited from Isaac Lab's locomotion config, which this repository does not hold.

Without --write the canon is compared and nothing is written; exit 1 on any difference. With --write
the blocks are written and the file is left as json.dumps(indent=2) leaves it, which is how it is
kept. The notes beside the blocks (randomization_note, randomization_wiring_note) are prose and are
not written: read them again after a --write that changed something.

What is baked into the trajectories is the config, not the canon. Changing a range in a config is a
regeneration (SOP section 10.1); changing the canon to say what the config already does is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation import conditions, randomization  # noqa: E402

SCENARIOS = REPO / "meta" / "scenarios.json"
TABLE_SCENARIOS = conditions.table_scenarios()
IMPLEMENTED = ("S1a", "S1d", *TABLE_SCENARIOS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="write the blocks into meta/scenarios.json")
    ap.add_argument("--scenario", choices=IMPLEMENTED, action="append",
                    help="check only this scenario (repeatable; default: every implemented one)")
    args = ap.parse_args()
    wanted = tuple(dict.fromkeys(args.scenario or IMPLEMENTED))     # --scenario given twice is one scenario

    from generation.isaac_boot import launch_app

    simulation_app = launch_app({"headless": True})[1]
    code = 1
    try:
        import isaaclab_tasks  # noqa: F401  -- registers the H1 tasks the configs inherit from
        from isaaclab_tasks.utils.hydra import resolve_presets

        from generation.envs.hri.hri_env_cfg import HriEnvCfg
        from generation.envs.s1a.s1a_env_cfg import S1aEnvCfg
        from generation.envs.s1d.s1d_env_cfg import S1dEnvCfg

        # The same construction as run_s1a.py; S1d with the cameras and the skin it is recorded with, a scenario
        # with a conditions table at row 0 (HriEnvCfg draws row 0 when it is given none).
        build = {
            "S1a": lambda: S1aEnvCfg(),
            "S1d": lambda: S1dEnvCfg(use_cameras=True, use_skin=True),
            **{s: (lambda s=s: HriEnvCfg(use_cameras=True, scenario_id=s)) for s in TABLE_SCENARIOS},
        }

        text = SCENARIOS.read_text(encoding="utf-8")
        canon = json.loads(text)
        entries, problems = randomization.entries(canon)
        changed: list[str] = []
        for scenario in wanted:
            entry = entries.get(scenario)
            if entry is None:
                problems.append(f"{scenario}: meta/scenarios.json 에서 한 번 나오는 항목이 아닙니다")
                continue
            cfg = resolve_presets(build[scenario](), selected=("physx",))
            blocks = randomization.read(cfg)[0]
            mass = getattr(cfg.events, "add_base_mass", None)
            body = getattr(mass.params.get("asset_cfg"), "body_names", None) if mass is not None else None
            print(f"== {scenario} — resolved config (base mass body: {body!r})")
            for block in randomization.BLOCKS:
                print(f"{block}: {json.dumps(blocks[block], ensure_ascii=False)}")
            if args.write:
                # Nothing is written from a config that cannot be read in full, or into a result that
                # would not pass the comparison (generation/randomization.rewritten).
                after, refused = randomization.rewritten(entry, cfg)
                if after is None:
                    problems += [f"{scenario}: {p}" for p in refused]
                    continue
                if json.dumps(after, ensure_ascii=False) != json.dumps(entry, ensure_ascii=False):
                    changed.append(scenario)
                    at = next(i for i, e in enumerate(canon["scenarios"]) if e is entry)
                    canon["scenarios"][at] = after
            else:
                problems += [f"{scenario}: {p}" for p in randomization.problems(entry, cfg)]

        if args.write and not problems:
            rewritten = json.dumps(canon, indent=2, ensure_ascii=False) + "\n"
            if rewritten != text:
                SCENARIOS.write_text(rewritten, encoding="utf-8")
            print(f"{SCENARIOS.relative_to(REPO)}: {len(changed)} scenario(s) changed"
                  + (f" — {', '.join(changed)}" if changed else ""))
        for p in problems:
            print(f"PROBLEM {p}")
        print("OK" if not problems else f"{len(problems)} problem(s)")
        code = 0 if not problems else 1
    except SystemExit as exc:
        code = int(exc.code or 0)
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        simulation_app.close(exit_code=code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
