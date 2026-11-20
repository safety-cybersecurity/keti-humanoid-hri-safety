---
license: cc-by-4.0
language:
  - en
  - ko
pretty_name: KETI Humanoid HRI Safety Dataset
task_categories:
  - robotics
tags:
  - robotics
  - humanoid
  - safety
  - near-miss
  - human-robot-interaction
  - LeRobot
  - isaac-sim
configs:
  - config_name: default
    data_files: data/*/*.parquet
---

<div align="center">

# KETI Humanoid HRI Safety Dataset

**A humanoid robot meets a person in a warehouse — failures and near-misses, labelled against ISO/TS 15066**

[![English](https://img.shields.io/badge/English-1f4e79?style=for-the-badge)](https://huggingface.co/datasets/KETI-IRRC/keti-humanoid-hri-safety)
[![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-e6e9ee?style=for-the-badge)](README.ko.md)

[![License](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey)](https://huggingface.co/datasets/KETI-IRRC/keti-humanoid-hri-safety/blob/main/LICENSE)
[![Format](https://img.shields.io/badge/LeRobot-v3.0-blue)](https://github.com/huggingface/lerobot)
[![Release](https://img.shields.io/badge/Release-26.11-green)](https://huggingface.co/datasets/KETI-IRRC/keti-humanoid-hri-safety/tree/v26.11.1)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/safety-cybersecurity/keti-humanoid-hri-safety)
[![Report](https://img.shields.io/badge/Technical_report-PDF-orange)](https://github.com/safety-cybersecurity/keti-humanoid-hri-safety/blob/v26.11.2/docs/tech-report/humanoid-dataset-technical-report-v0.1.pdf)

![The three scenarios](docs/media/teaser.png)

*`S1e` a corner hidden by a rack · `S1f` stepping in from the side · `S1g` turning round with a box*

</div>

## Overview

A public dataset for **evaluating the safety of humanoid robots that work around people**. In a simulated logistics warehouse (NVIDIA Isaac Sim), a Unitree H1 carrying a box meets a person in three places where it does not see them in time. The recorded robot is a baseline that does not react to the person, so what the dataset holds is **failures, near-misses and boundary cases**.

Labels are **computed, not annotated**. The decision rules ship with the data: they compare the contact force on twelve body regions with the limits of ISO/TS 15066 Annex A, and the distance to the person with the robot's protective separation distance. Every label names the rule that produced it and the clause that rule rests on.

Built by the **Korea Electronics Technology Institute (KETI)**.

| | |
|---|---|
| Episodes | **297** — three scenarios × 99 conditions · 247,181 steps at 50 fps |
| Robot | Unitree H1 (19 DOF) carrying a 1–3 kg box, driven by the published `Isaac-Velocity-Flat-H1-v0` walking policy. **It does not react to the person** |
| Person | A jointed dummy on the ISO/TS 15066 body-region breakdown (12 regions), walking by a gait table |
| Cameras | 7 RGB, 640×480, AV1 video — five on the robot, one following it, one fixed in the warehouse |
| Labels | `outcome` (`fail` / `near_miss` / `nominal`) · `severity` (`S0`–`S3`) · `triggered_rules` |
| Format | LeRobotDataset v3.0 + `safety.*` extension (schema `0.2`) |
| Source | Simulation only — **not** recorded on physical hardware |
| Release | `26.11` — tag `v26.11.1` |

## Scenarios

All three take place in one warehouse — tall racks on both sides of a main aisle, narrow aisles between the rows — each at a place where the robot does not see the person in time.

| Scenario | The robot | The person | Where the person is |
|---|---|---|---|
| **`S1e`** — a corner hidden by a rack | walks down a narrow aisle and turns round the end of the rack row | stands close to the corner, facing the rack | ahead, behind the rack |
| **`S1f`** — stepping in from the side | walks straight down a narrow aisle | faces a rack with their back to the robot, then turns and steps into the aisle | beside |
| **`S1g`** — turning round with a box | stands at one rack, turns half a turn and walks to the facing rack | walks along the aisle, passing the robot | behind |

Each scenario has a condition table of 99 rows, one episode per row: **50 random rows** drawn from the scenario's ranges, and **49 grid rows** that sweep two quantities over a 7 × 7 grid — the *failure map*. Places, ranges and the grid are this project's own settings (`meta/scenarios.json`).

## Get started

```bash
pip install "lerobot[dataset]==0.6.1"   # Python 3.12 or newer
```

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("KETI-IRRC/keti-humanoid-hri-safety", revision="v26.11.1")
print(ds.num_episodes, ds.num_frames, ds.fps)
```

- `revision="v26.11.1"` pins this release. The `v3.0` tag is LeRobot's format version and moves with every publish.
- Every file is plain parquet and JSON, so the tables can be read without LeRobot.
- `lerobot-dataset-viz` opens the release as shipped. `lerobot-train` needs a training copy, which `scripts/make_train_copy.py` in the code repository makes.

## Dataset structure

```
README.md · README.ko.md          this card, in English and in Korean
data/chunk-000/*.parquet          step time series
videos/observation.images.*/      RGB video, one directory per camera
meta/info.json                    LeRobot standard (features, fps, version)
meta/episodes/                    per-episode index and metadata
meta/tasks.parquet                task definitions — one robot task per scenario
meta/stats.json                   normalisation statistics
meta/safety_schema.json    ext    machine-readable definition of the schema
meta/scenarios.json        ext    scenario definitions, condition tables, and the rules each applies
meta/rules.json            ext    decision rules, thresholds, and the robot's stop-test table
meta/assets.json           ext    asset paths, checksums and licences
meta/cameras.json          ext    camera intrinsics, mounts and placements
meta/joint_map/*.yaml      ext    joint names and order, extracted from the asset
assets/                    ext    the warehouse, the dummy, the box, the composed stage, the Unitree H1 (USD)
docs/dataset-reference.md         the full reference
docs/dataset-reference.ko.md      the full reference, in Korean
docs/assets/asset-license-inventory-v0.1.md
                                  the asset licence findings NOTICE points at
NOTICE                            provenance and licence of included assets
LICENSE · LICENSE-CODE            CC BY 4.0 for the data · Apache-2.0 for code quoted here
```

The `ext` entries are this project's extension. A standard LeRobot loader ignores them and still works.

<details>
<summary><b>Step table</b> — raw and derived columns</summary>

Every column belongs to one of two layers, so a model can take its inputs from the raw layer alone.

| Layer | Columns |
|---|---|
| Raw — what the simulator and its sensors produced | `timestamp` · `frame_index` · `observation.state.joint_pos` · `observation.state.joint_vel` · `observation.state.joint_torque` · `observation.root_pose` · `observation.root_vel` · `observation.imu` · `action` · `observation.command_velocity` · `observation.contact.force` · `observation.contact.region_names` · `observation.human.root_pose` · `observation.human.root_vel` · `observation.images.<camera>` (7) · `observation.camera_pose.<camera>` (7) |
| Derived — what the judge computed from the raw layer | `safety.min_distance_human` · `safety.separation_required` · `safety.separation_violation` · `safety.approach_speed_robot` · `safety.approach_speed_human` · `safety.event` · `safety.com` · `safety.zmp` · `safety.stability_margin` · `observation.contact.type` · `observation.contact.v_rel` · `observation.bbox2d.human` · `observation.human_visible_fraction.<camera>` (5) |

`observation.contact.force` is the force the robot and the box it carries put on each of the person's twelve body regions, the peak over each step's physics substeps.

</details>

<details>
<summary><b>Episode table</b> — conditions, labels and evidence</summary>

| Group | Columns |
|---|---|
| Identity | `episode_id` · `scenario_id` · `randomization_seed` · `robot_model` · `sim_backend` |
| Condition | `condition_design` (a random row or a grid row) · `grid_cell_robot` · `grid_cell_human` · `episode_spec` |
| Labels | `outcome` · `severity` · `triggered_rules` · `standard_refs` |
| Evidence | `evidence.first_contact_s` · `evidence.min_separation_margin_m` · `evidence.force_ratio.<region>` (12) · `evidence.force_ratio_max` · `evidence.human_first_seen_s` · `evidence.human_last_seen_s` |
| Notes | `notes` — the drawn condition row, why a rule was not evaluated, task stages |

</details>

## Labels

| `outcome` | When | `severity` |
|---|---|---|
| `fail` | The robot, or the box it carries, touched the person | `S3` if a contact force exceeded its ISO/TS 15066 Annex A limit, or the head or face took any force · otherwise `S2` |
| `near_miss` | No contact, but the robot moved toward the person faster than 0.05 m/s (a self-set threshold) while closer than its protective separation distance (ISO/TS 15066, 5.5.4.2.3) | `S1` |
| `nominal` | Neither | `S0` |

`nominal` means that none of the rules this release applies held. It is not a guarantee that the episode was safe.

## Release 26.11

What the baseline robot did, by scenario:

| Scenario | Random rows (50) | Grid rows (49) |
|---|---|---|
| `S1e` | `fail` 9 · `near_miss` 41 | `fail` 15 · `near_miss` 34 |
| `S1f` | `fail` 37 · `near_miss` 13 | `fail` 22 · `near_miss` 27 |
| `S1g` | `fail` 50 · `near_miss` 0 | `fail` 38 · `near_miss` 11 |

Of the 150 random rows, 96 ended in contact, and in 86 of those a contact force exceeded its standard limit. No episode is `nominal`: a robot that does not react to the person either touches them or comes closer than its stopping distance.

<!-- batch-stats: episodes=297 grid_episodes=147 fail=96 near_miss=54 nominal=0 contact_exceeded=86 contact_prohibited=2 -->
<!-- generator-commit: 26a81accbe69d7a8c63f4ed1f7cf434ed0074a66 -->

## Limitations

- **Simulation only.** No sim-to-real validation has been performed.
- **One controller.** Every episode comes from one published walking policy that does not react to the person.
- **The person is a dummy.** It walks by a gait table while a hold carries part of its weight, and becomes a passive body at the first contact.
- **The forces are this dummy's, in this physics engine.** They are not measurements on a human body.
- **"Seen" is geometry.** It is computed from the cameras' views, not from a detector.

The full list is in the dataset reference.

## Documentation

| | |
|---|---|
| [Dataset reference](docs/dataset-reference.md) | Every column, the condition tables, how each label is computed, the scorecard and failure maps, the full list of limitations |
| [Technical report](https://github.com/safety-cybersecurity/keti-humanoid-hri-safety/blob/v26.11.2/docs/tech-report/humanoid-dataset-technical-report-v0.1.pdf) | Step-by-step use, in Korean — validating, rebuilding the labels, running another controller |
| [Code repository](https://github.com/safety-cybersecurity/keti-humanoid-hri-safety) | Generation pipeline, validator, labeller, scorecard |

## License

The dataset and its documentation are released under **CC BY 4.0**; the pipeline code is Apache-2.0. The contact limits in `meta/rules.json` are values of KS B ISO TS 15066:2016 Annex A, redistributed with the permission of the Korean Agency for Technology and Standards. The Unitree H1 files under `assets/unitree_h1/` keep their BSD 3-Clause licence. The KETI logo on the box labels is KETI's own mark: it may be redistributed unchanged as part of this dataset, and no right is granted to use it as a trademark or apart from the dataset. `NOTICE` gives the source and licence of every included asset.

## Citation

```bibtex
@misc{keti_humanoid_hri_safety_dataset,
  title  = {KETI Humanoid HRI Safety Dataset},
  author = {Korea Electronics Technology Institute},
  year   = {2026},
  note   = {Version 26.11.1. LeRobotDataset v3.0 with safety extensions},
  howpublished = {Hugging Face Hub}
}
```
