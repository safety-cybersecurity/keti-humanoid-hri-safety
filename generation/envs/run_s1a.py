#!/usr/bin/env python3
"""S1a 에피소드를 돌린다.

    ~/isaacsim/python.sh generation/envs/run_s1a.py --episodes 1

보행 정책은 학습하지 않고 Isaac Lab 이 게시한 H1 평지 체크포인트를 내려받아 씁니다.
그 정책의 입력은 H1FlatEnvCfg 의 관찰 배열이므로, 관찰 항을 건드리면 조용히 이상하게
걷습니다. 씬만 늘리고 관찰은 상속받은 그대로 둡니다.

화면에서 보면서 녹화하려면:

    ~/isaacsim/python.sh generation/envs/run_s1a.py --gui --video --seq 0 --episodes 1

--gui 는 Kit 창을 열고 카메라가 로봇을 따라갑니다(--free-camera 로 끄면 마우스로
돌립니다). --video 는 에피소드마다 npz 옆에 같은 이름의 mp4 를 남깁니다. 둘 중 하나라도
켜면 --out 이 없을 때 정본 배치가 아니라 .tmp/episodes/S1a-watch-<시각>/ 에 씁니다.
플래그가 없으면 배치를 만든 헤드리스 경로 그대로입니다 — generation/envs/s1a/watch.py.

S1d(물류 통로 씬 · 입힌 더미 · meta/cameras.json 의 카메라 일곱)도
같은 진입점입니다. 카메라마다 영상·자세와 사람 상자가 에피소드 옆에 함께 남습니다(깊이/마스크 팩은 정본이 선언할 때만):

    ~/isaacsim/python.sh generation/envs/run_s1a.py --scenario S1d --seq 0 --episodes 1

--no-cameras · --no-skin 은 대조군용입니다. 둘 다 주면 S1d 씬에 v1 캡슐 더미뿐입니다.

조건표가 있는 시나리오(S1e · S1f · S1g — meta/scenarios.json 의 conditions)는 한 프로세스가 조건표의 한 행이고
행 번호가 --seq 입니다(--episodes 1 만). 같은 창고에 조끼를 입힌 관절 인형, 로봇은 그 행의 출발 자리에서 시작하고
S1e · S1f · S1g 의 로봇은 몸통에 고정한 상자를 팔을 정책의 기본 자세로 잡은 채 듭니다:

    ~/isaacsim/python.sh generation/envs/run_s1a.py --scenario S1e --seq 0

--policy 는 H1 형식 보행 정책(rsl_rl 체크포인트 — 기본은 Isaac Lab 이 게시한 것), --behavior 는 과제 명령과
보행 정책 사이의 행동 층(모듈:이름 — 기본은 과제 명령을 그대로 넘기는 generation.behavior:PassThrough)이고,
둘 다 에피소드의 sim_backend.policy 에 남습니다.
"""

import argparse
import sys

sys.stdout.reconfigure(line_buffering=True)

# `~/isaacsim/python.sh <script>` puts only the script's own directory on
# sys.path, not the repository root, so `generation` (an implicit namespace
# package) is not importable without this. Same fix as check_scene.py.
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Isaac-free; imported before the parser so the help can name the batch dir
# from one place instead of a second literal.
from generation import appearance as appearance_canon
from generation import behavior as behavior_layer
from generation import conditions
from generation import provenance
from generation.envs.s1a import watch

# The scenarios with a conditions table (meta/scenarios.json): one process is one row of it. They and S1d run in the
# warehouse with the cameras of meta/cameras.json.
TABLE_SCENARIOS = conditions.table_scenarios()
CAMERA_SCENARIOS = ("S1d", *TABLE_SCENARIOS)

parser = argparse.ArgumentParser(description="S1a · S1d · " + " · ".join(TABLE_SCENARIOS) + " 에피소드 생성")
parser.add_argument("--scenario", choices=("S1a", "S1d", *TABLE_SCENARIOS), default="S1a",
                    help="S1d: S1a 의 진입 시나리오를 물류 통로에서, 입힌 더미와 카메라 일곱으로 · "
                         + " · ".join(TABLE_SCENARIOS) + ": 같은 창고에서 조건표의 한 행(--seq) (기본 S1a)")
parser.add_argument("--no-cameras", action="store_true",
                    help="S1d · " + " · ".join(TABLE_SCENARIOS)
                         + " 에서 카메라를 만들지 않는다 (대조군 — 부팅 인자도 S1a 와 같아진다)")
parser.add_argument("--no-skin", action="store_true", help="S1d 에서 v1 캡슐 더미를 쓴다 (대조군)")
parser.add_argument("--no-appearance", action="store_true",
                    help="조건표 시나리오에서 뽑은 겉모습을 장면에 입히지 않는다 (대조군 — 물리 배열은 입혔을 때와 같고 "
                         "사이드카의 appearance 는 null)")
parser.add_argument("--episodes", type=int, default=1, help="생성할 에피소드 수")
parser.add_argument("--seq", type=int, default=0,
                    help="첫 에피소드의 시퀀스 번호. 시드는 여기서 파생된다 "
                         "(meta/scenarios.json seed_rule)")
parser.add_argument("--out", default=None,
                    help=f"에피소드 출력 위치 (기본 {watch.BATCH_OUT_DIR}, S1d 는 .tmp/episodes/S1d; "
                         f"--gui/--video 이면 {watch.WATCH_OUT_DIR_PREFIX}<시각>)")
parser.add_argument("--max-steps", type=int, default=1000, help="에피소드당 최대 스텝")
parser.add_argument("--gui", action="store_true",
                    help="Kit 창을 열어 보면서 돌린다. 카메라는 로봇을 따라간다")
parser.add_argument("--video", action="store_true",
                    help="에피소드마다 npz 와 같은 이름의 mp4 를 남긴다 (--gui 없이도 됨)")
parser.add_argument("--speed", type=float, default=1.0,
                    help="--gui 에서 실시간 대비 배속의 상한 (기본 1.0). 한 스텝의 계산·렌더가 "
                         "그보다 느리면 그만큼 느려진다. 0 이면 기다리지 않는다")
parser.add_argument("--free-camera", action="store_true",
                    help="--gui 에서 카메라 추적을 끄고 마우스로 돌린다. 영상도 그 시점을 담는다")
parser.add_argument("--compare-with", default=None, metavar="DIR",
                    help="끝난 뒤 생성된 에피소드를 이 배치의 같은 이름 npz 와 비트 단위로 대조한다")
parser.add_argument("--policy", default=None, metavar="CHECKPOINT",
                    help="조건표 시나리오의 보행 정책 — H1 형식 rsl_rl 체크포인트 파일 (기본: Isaac Lab 이 게시한 "
                         f"{behavior_layer.PUBLISHED_POLICY_TASK} 체크포인트)")
parser.add_argument("--behavior", default=None, metavar="MODULE:NAME",
                    help=f"조건표 시나리오의 행동 층 (기본 {behavior_layer.DEFAULT_BEHAVIOR} — 과제 명령 그대로)")
args = parser.parse_args()
if args.scenario == "S1a" and (args.no_cameras or args.no_skin):
    parser.error("--no-cameras · --no-skin 은 --scenario S1d 와 함께 씁니다 (--no-cameras 는 "
                 + " · ".join(TABLE_SCENARIOS) + " 에도)")
if (args.no_cameras or args.no_skin or args.no_appearance) and args.out is None:
    # Episode names carry scenario and seq, not the variant: a control arm left to the default
    # directory would overwrite the batch's episodes of the same seq.
    parser.error("--no-cameras · --no-skin · --no-appearance 는 대조군 플래그라 --out 을 함께 줍니다 — 없으면 그 시나리오 배치의 같은 에피소드를 덮어씁니다")
ROW = None
APPEARANCE_SPEC = None   # the scenario's appearance block (None: it declares none)
APPEARANCE_DRAWN = None  # the row's look, drawn before the boot (None: no block, or --no-appearance)
if args.no_appearance and args.scenario not in TABLE_SCENARIOS:
    parser.error("--no-appearance 는 조건표 시나리오(" + " · ".join(TABLE_SCENARIOS) + ")에서 씁니다")
if args.scenario in TABLE_SCENARIOS:
    if args.no_skin:
        parser.error("--no-skin 은 S1d 의 대조군 플래그입니다 — 조건표 시나리오의 사람은 조끼를 입힌 관절 인형입니다")
    if args.episodes != 1:
        parser.error(f"{args.scenario} 는 한 프로세스가 조건표의 한 행입니다 — --episodes 1 로 돌리고 행 번호는 --seq 로 줍니다")
    if args.policy is not None and not Path(args.policy).is_file():
        parser.error(f"--policy 의 체크포인트 파일이 없습니다: {args.policy}")
    try:
        behavior_layer.parse_spec(args.behavior or behavior_layer.DEFAULT_BEHAVIOR)
        # The row before the minute-long boot: a row the table lacks, or one the guard refuses, ends here.
        from generation.scene import layout as scene_layout

        ROW = conditions.draw_row(conditions.scenario_entry(args.scenario), args.seq,
                                  scene_layout.build_layout(scene_layout.load_spec(scene_layout.SCENE_SPEC)))
        # The look of the same row, from its own stream: a block the canon gets wrong ends here too, before the boot.
        APPEARANCE_SPEC = appearance_canon.load_appearance(conditions.scenario_entry(args.scenario))
        if APPEARANCE_SPEC is not None and not args.no_appearance:
            APPEARANCE_DRAWN = appearance_canon.draw_appearance(conditions.scenario_entry(args.scenario), args.seq)
            appearance_canon.appearance_edits(APPEARANCE_SPEC, APPEARANCE_DRAWN)
    except ValueError as error:
        parser.error(str(error))
elif args.policy is not None or args.behavior is not None:
    parser.error("--policy · --behavior 는 조건표 시나리오(" + " · ".join(TABLE_SCENARIOS) + ")에서 씁니다")
if args.free_camera and not args.gui:
    parser.error("--free-camera 는 --gui 와 함께 씁니다")
speed_problem = watch.validate_speed(args.speed)
if speed_problem:
    parser.error(speed_problem)
# Half of the seed key (meta/scenarios.json seed_rule).
SCENARIO_ID = args.scenario
CAMERAS_ON = SCENARIO_ID in CAMERA_SCENARIOS and not args.no_cameras
out_dir = watch.resolve_out_dir(args.out, gui=args.gui, video=args.video, scenario=SCENARIO_ID)
if args.compare_with and Path(out_dir).resolve() == Path(args.compare_with).resolve():
    parser.error(f"--compare-with 가 출력 디렉터리 자신입니다: {out_dir}")
if args.out is None and (args.gui or args.video):
    # Claim the fresh directory now, before the minute-long boot, so a second
    # run started in the same second lands in a numbered sibling instead.
    out_dir = watch.claim_out_dir(out_dir)

from generation.isaac_boot import launch_app

# Booting through the helper is what makes a boot failure exit nonzero: the
# try/finally below only arms once simulation_app exists, and until it does,
# Kit's own atexit handler would close with its default exit_code=0 and hand
# back a green process. See generation/isaac_boot.py for the measurement.
# Without --gui/--video the argument is the {"headless": True} the batch was
# generated with (tests/generation/test_watch.py pins that).
simulation_app = launch_app(watch.launcher_args(gui=args.gui, video=args.video, cameras=CAMERAS_ON))[1]

try:
    import importlib.metadata as metadata
    import math
    import time

    import torch
    import yaml
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    import isaaclab_tasks  # noqa: F401  — registers the H1 tasks the agent cfg lives in
    import generation.envs.register  # noqa: F401  — registers Keti-S1a-v0

    from isaaclab_tasks.utils import load_cfg_from_registry
    from generation.envs.s1a import substep_contact
    from generation.envs.s1a.recorder import (
        CONTACT_EPS_N,
        EpisodeRecorder,
        MIN_VALID_STEPS,
        foot_link_names,
        read_foot_footprint,
    )
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    from generation.assets import asset_entry, canonical_usd_path, load_assets
    from generation.dummy import jointed
    from generation.envs.s1a.s1a_env_cfg import S1aEnvCfg
    from generation.seeding import episode_seed, scenario_startup_seed

    from isaaclab_tasks.utils.hydra import resolve_presets

    # H1FlatEnvCfg wraps sim.physics and scene.contact_forces in Hydra presets.
    # Instantiating the cfg directly leaves them unresolved — the sim context
    # then gets a PresetCfg where it expects a PhysxCfg, and the contact sensor
    # never registers. parse_env_cfg resolves them for the CLI path; this is
    # the same call, pinned to physx (see check_scene.py).
    if SCENARIO_ID == "S1d":
        from generation.envs.s1d.s1d_env_cfg import S1dEnvCfg

        env_cfg = resolve_presets(S1dEnvCfg(use_cameras=CAMERAS_ON, use_skin=not args.no_skin,
                                            scenario_id=SCENARIO_ID),
                                  selected=("physx",))
    elif SCENARIO_ID in TABLE_SCENARIOS:
        from generation.envs.hri.hri_env_cfg import HriEnvCfg

        env_cfg = resolve_presets(HriEnvCfg(use_cameras=CAMERAS_ON, scenario_id=SCENARIO_ID, row=dict(ROW)),
                                  selected=("physx",))
        print(f"조건표 {SCENARIO_ID} 행 {args.seq}: {ROW}")
    else:
        env_cfg = resolve_presets(S1aEnvCfg(), selected=("physx",))
    # Construction draws whatever randomisation is still mode="startup" once
    # for the whole process -- the friction material, a constant here; the
    # base mass is drawn at each reset (S1aEnvCfg). Seeding that from --seq
    # would build a different robot depending on where a batch started, so it
    # gets a constant of the scenario's own instead -- see scenario_startup_
    # seed's docstring for the measurement. Per-episode seeding happens at
    # each reset in the loop below.
    env_cfg.seed = scenario_startup_seed(SCENARIO_ID)
    agent_cfg = load_cfg_from_registry("Keti-S1a-v0", "rsl_rl_cfg_entry_point")
    # H1RoughPPORunnerCfg (the base H1FlatPPORunnerCfg inherits from) still sets
    # its actor/critic through the pre-5.0 `distribution_cfg` field but leaves
    # the now-removed `stochastic`/`init_noise_std` fields on the dataclass at
    # their MISSING default. rsl-rl-lib 5.0.1's MLPModel.__init__ no longer
    # accepts `stochastic` at all, so passing agent_cfg.to_dict() straight
    # through raises `TypeError: unexpected keyword argument 'stochastic'`
    # (confirmed by running this script). Isaac Lab's own play.py calls this
    # exact function first to strip the deprecated fields before constructing
    # OnPolicyRunner; same fix here.
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # --video: Isaac Lab's own VideoRecorder captures the viewport camera when
    # render_mode is "rgb_array"; the subclass moves that camera to the robot
    # before every frame (see watch.following_video_recorder). With
    # --free-camera the stock recorder is kept, so the mp4 shows whatever the
    # viewport shows.
    if args.video and not args.free_camera:
        env_cfg.video_recorder.class_type = watch.following_video_recorder(watch.follow_camera())
    env = RslRlVecEnvWrapper(
        ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    )
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # The checkpoint is published against the training task, not against ours.
    checkpoint = args.policy or get_published_pretrained_checkpoint("rsl_rl", behavior_layer.PUBLISHED_POLICY_TASK)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # sim_backend is captured, never hand-typed — the repo-root
    # environment.lock is the single source of truth for it.
    lock = yaml.safe_load((REPO_ROOT / "environment.lock").read_text(encoding="utf-8"))
    # The lock cannot hold the commit it is committed in. The two keys are read here, once, so
    # every episode of the run names the tree the run started on (generation/provenance.py).
    sim_backend = dict(lock["sim_backend"], **provenance.read(REPO_ROOT))
    arm_hold = carried = None
    if SCENARIO_ID in TABLE_SCENARIOS:
        behavior_spec = args.behavior or behavior_layer.DEFAULT_BEHAVIOR
        # Loaded before the first episode, so a behaviour that cannot be made stops the run before anything is
        # recorded.
        behavior = behavior_layer.load_behavior(behavior_spec)
        sim_backend["policy"] = behavior_layer.policy_record(Path(checkpoint), published=args.policy is None,
                                                             behavior_spec=behavior_spec, repo_root=REPO_ROOT)
        print(f"정책: {sim_backend['policy']}")

    base_env = env.unwrapped
    if args.gui and not args.free_camera:
        # The controller only exists when a Kit viewport does; --gui made one.
        watch.track_robot(base_env.viewport_camera_controller, watch.follow_camera())
    # Real-time pacing only matters with a window to look at.
    pacer = watch.Pacer(base_env.step_dt, args.speed if args.gui else 0.0)
    video_size = (env_cfg.video_recorder.window_width, env_cfg.video_recorder.window_height)
    video_fps = round(1.0 / base_env.step_dt)
    # Make both contact sensors compute every physics substep, so a contact
    # that lives and dies between two record steps is recorded instead of
    # vanishing. Installed once on the env; EpisodeRecorder finds it through
    # substep_contact.get() and refuses to record without it. See that module.
    substep = substep_contact.install(base_env, contact_eps_n=CONTACT_EPS_N)
    # The scenarios with a conditions table read the dummy sensor's force matrix every substep as well: what the
    # robot's links and the carried box gave the person -- their contact columns -- and what the links and the box
    # took from anything else -- the rack contact (substep_contact.install_hri).
    hri_contact = substep_contact.install_hri(base_env) if SCENARIO_ID in TABLE_SCENARIOS else None
    # The dummy that was spawned, like the robot below -- not a module constant, which would
    # keep naming v1 after S1d spawned the skinned dummy.
    dummy_usd = str(base_env.scene["dummy"].cfg.spawn.usd_path)
    # Region codes per contact-sensor body, None for a ground-contact body (the jointed dummy's feet);
    # the rigid dummy's asset gives the same list as read_region_codes.
    codes = jointed.read_body_codes(dummy_usd, list(base_env.scene["dummy_contact"].body_names))
    robot = base_env.scene["robot"]
    # robot.cfg.spawn.usd_path is whatever Isaac Lab actually resolved and
    # loaded (a Nucleus/S3 URL on this build, not a local path) -- read from
    # the live asset rather than a second, hand-typed copy of the path that
    # could silently point at a different H1 variant than the one in the
    # running scene.
    foot_footprints = read_foot_footprint(str(robot.cfg.spawn.usd_path), foot_link_names(robot))
    # The key meta/joint_map/ and meta/assets.json are both indexed by. Read
    # from the map rather than typed here, so a rename cannot leave three
    # copies disagreeing (CONTRIBUTING.md 「함께 고치는 곳」).
    robot_model = yaml.safe_load(
        (REPO_ROOT / "meta/joint_map/unitree_h1.yaml").read_text(encoding="utf-8")
    )["robot_model"]
    # What was actually loaded, off the live assets -- then normalised to the
    # form meta/assets.json indexes by. Isaac Lab expands ISAACLAB_NUCLEUS_DIR
    # into a versioned S3 URL and hands back the dummy as an absolute path, so
    # the raw strings join against nothing and the absolute one would put this
    # machine's home directory into episode metadata.
    # The ground is a third asset even though it is not a file: Isaac Lab's
    # TerrainImporter builds it procedurally, and an episode whose
    # scene_assets omit it is not describing the scene it ran in. Read from
    # the live config so the canonical id follows terrain_type rather than
    # being typed here.
    terrain_type = str(base_env.scene.cfg.terrain.terrain_type)
    resolved = {
        "robot": str(robot.cfg.spawn.usd_path),
        "dummy": dummy_usd,
        "terrain": f"isaaclab://TerrainImporter/{terrain_type}",
    }
    # S1d's logistics aisle is a fourth asset the episode ran in; S1a has none.
    aisle_cfg = getattr(base_env.scene.cfg, "aisle", None)
    if aisle_cfg is not None:
        resolved["aisle"] = str(aisle_cfg.spawn.usd_path)
    # The box the robot carries (S1e, S1f, S1g) is one more.
    box_cfg = getattr(base_env.scene.cfg, "box", None)
    if box_cfg is not None:
        resolved["box"] = str(box_cfg.spawn.usd_path)
    asset_paths = {
        k: canonical_usd_path(v, ISAACLAB_NUCLEUS_DIR, REPO_ROOT)
        for k, v in resolved.items()
    }
    asset_paths_resolved = {
        k: canonical_usd_path(v, "", REPO_ROOT) for k, v in resolved.items()
    }
    # Check the join before spending a batch. Left to conversion time, an
    # unknown path costs every episode that was generated in between.
    assets_canon = load_assets(REPO_ROOT)
    for key, path in asset_paths.items():
        asset_entry(assets_canon, path)   # raises on an asset the canon lacks
    print(f"자산 정본 조인 확인: {list(asset_paths.values())}")
    command = base_env.command_manager.get_term("base_velocity")
    if SCENARIO_ID in TABLE_SCENARIOS:
        # The behaviour layer between the row's task and the walking policy (generation/envs/hri/commands.py).
        command.set_behavior(behavior)
    rig = None
    if CAMERAS_ON:
        from generation import cameras as camera_canon
        from generation.envs.s1a.media_writer import MediaWriter
        from generation.envs.s1d.camera_io import Rig
        from generation.envs.s1d.s1d_env_cfg import CAMERAS

        cameras_canon = camera_canon.load(CAMERAS)
        rig = Rig(base_env, cameras_canon, SCENARIO_ID)
    if CAMERAS_ON or args.video or args.gui:
        # A jointed dummy's skin pieces part at a bent joint: blend them at their seams before every render
        # (generation/envs/s1d/skin_blend.py). A dummy built without the weights renders as it is built.
        from generation.envs.s1d.skin_blend import SkinBlend

        skin_blend = SkinBlend.from_env(base_env)
        if skin_blend is not None:
            skin_blend.attach()
            print(f"피부 이음새 섞기: 조각 {skin_blend.pieces} · 정점 {skin_blend.points} (렌더할 때마다)")
    if SCENARIO_ID in TABLE_SCENARIOS:
        from generation import carry as carry_canon
        from generation.envs.hri.carry_rig import RIG_ATTR, ArmHold

        held = carry_canon.load_carry(conditions.scenario_entry(SCENARIO_ID))
        if held is not None:
            # The arms at the carry's hold pose -- refused unless it is the robot's default joint position.
            arm_hold = ArmHold(base_env, held)
            carried = getattr(base_env, RIG_ATTR)

    if (CAMERAS_ON or args.video or args.gui) and APPEARANCE_DRAWN is not None:
        # The row's look goes into the opened stage once, here: before the episode loop's reset and the first recorded
        # frame, and not in the episode loop -- the edits compound (scales multiply what is there). The built USD files are not written.
        import omni.usd

        from generation.envs.hri.appearance_apply import apply as apply_appearance
        from generation.envs.s1d.skin_blend import _env_prim_path

        appearance_edits, appearance_light = appearance_canon.appearance_edits(APPEARANCE_SPEC, APPEARANCE_DRAWN)
        appearance_canon.claim_single_apply()
        apply_appearance(omni.usd.get_context().get_stage(),
                         {"aisle": _env_prim_path(base_env, "aisle"), "dummy": _env_prim_path(base_env, "dummy"),
                          "box": _env_prim_path(base_env, "box")},
                         appearance_edits, appearance_light)
        print(f"겉모습: {APPEARANCE_DRAWN}")

    for offset in range(args.episodes):
        episode_seq = args.seq + offset
        seed = episode_seed(SCENARIO_ID, episode_seq)
        # Seed first, reset second: the reset events draw the robot's pose, the
        # dummy's entry timing and the physics randomisation from the RNG this
        # seed() call just re-based, so every *recorded input* of the episode
        # is a function of episode_seq alone.
        #
        # That is not yet the same as the episode being reproducible on its
        # own. Measured 2026-08-20: `--seq 2 --episodes 1` and episode 2 of
        # `--seq 0 --episodes 3` agree on every randomisation value written to
        # the episode's .json -- robot base mass to the last digit -- and
        # still diverge from step 0. Whole-run reproduction does hold, bit for
        # bit. What carries over between episodes has not been identified;
        # env.reset() not clearing everything is a hypothesis, and policy
        # state and GPU nondeterminism have not been ruled out. See
        # generation/README.md "재현은 실행 단위로만 됩니다".
        #
        # RslRlVecEnvWrapper.reset() returns (TensorDict, extras) -- a
        # 2-tuple, unlike get_observations()'s bare TensorDict (confirmed
        # against the installed rsl-rl-lib 5.0.1,
        # isaaclab_rl/rsl_rl/vecenv_wrapper.py).
        env.seed(seed)
        # inference_mode, like every env.step() below: the command manager's
        # metric buffers were first allocated inside one of those blocks, and
        # reset() writes to them in place. Outside the block that raises
        # "Inplace update to inference tensor outside InferenceMode is not
        # allowed" on the second episode -- measured, not guessed.
        with torch.inference_mode():
            obs, _ = env.reset()
        # The reset ran physics, so the accumulator holds substeps that belong
        # to no episode. Drop them rather than fold them into step 0.
        substep.reset()
        if hri_contact is not None:
            hri_contact.reset()
        if SCENARIO_ID in TABLE_SCENARIOS:
            from generation import sites

            start = conditions.row_site(sites.load_site(conditions.scenario_entry(SCENARIO_ID)), ROW).robot_start
            root_xy = robot.data.root_link_pos_w.torch[0, :2].detach().cpu().tolist()
            x, y, z, w = robot.data.root_link_quat_w.torch[0].detach().cpu().tolist()
            root_yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
            print(f"  출발 — 행 ({start.x:.4f}, {start.y:.4f}, {start.yaw_deg:.2f}°) · 리셋 뒤 뿌리 "
                  f"({root_xy[0]:.4f}, {root_xy[1]:.4f}, {root_yaw:.2f}°)")
        # The largest error of the held box from its carry pose (m, deg) and of the held arms from their hold (rad), and
        # the least clearance of a camera the arms move about the box from the held box (m, camera; negative: inside).
        held_box_err, held_arm_err = (0.0, 0.0), 0.0
        wrist_clear: tuple[float, str] | None = None
        recorder = EpisodeRecorder(
            out_dir, SCENARIO_ID, episode_seq, seed, command.provisional, codes,
            sim_backend, foot_footprints,
            robot_model=robot_model, asset_paths=asset_paths,
            asset_paths_resolved=asset_paths_resolved,
            # Where this run began: the reproduction of any of its episodes reruns from here
            # (meta/scenarios.json seed_rule -- reproduction holds per run, not per episode).
            run_start_seq=args.seq,
            # What the row drew, whether or not this run renders it; null under --no-appearance; no key at all for a
            # scenario without an appearance block.
            **({} if APPEARANCE_SPEC is None else {"appearance": APPEARANCE_DRAWN}),
        )
        recorder.capture_randomization(base_env, command)
        media = None
        if rig is not None:
            # The renderer's temporal history still holds the last episode: render the new state a
            # few times, from where the third-person camera will be, before anything is recorded
            # (meta/cameras.json render). Nothing is read for recording between reset and step 0 --
            # it would be the last frame before the reset. The rig first takes the robot's heading after the
            # reset, which a third-person camera placed in the initial_heading frame keeps for the episode.
            rig.reset(base_env)
            rig.follow()
            rig.warm(cameras_canon["render"]["reset_warm_renders"])
            media = MediaWriter(out_dir, recorder.episode_id, cameras_canon)
        # The episode id is the recorder's to make (save() builds it); the mp4
        # is written under a temporary name and renamed next to the npz once
        # save() has said what that name is, so the two cannot disagree.
        writer = None
        if args.video:
            writer = watch.Mp4Writer(
                Path(out_dir) / f".episode-{episode_seq:04d}.part.mp4",
                width=video_size[0], height=video_size[1], fps=video_fps,
            )
        pacer.start()
        episode_started = time.monotonic()
        step = -1
        done = False
        for step in range(args.max_steps):
            if rig is not None:
                rig.follow()   # before the step: the frame read after it shows the camera here
            with torch.inference_mode():
                actions = policy(obs)
                if arm_hold is not None:
                    arm_hold.apply(actions)
                obs, _, dones, _ = env.step(actions)
            done = bool(dones[0])
            if done:
                # The env reset itself inside that step(), so whatever the hook
                # sampled straddles two episodes. run_s1a happens to reset again
                # at the top of the next episode, but relying on that leaves the
                # accumulator correct only by accident.
                substep.reset()
                if hri_contact is not None:
                    hri_contact.reset()
            # ManagerBasedRLEnv._reset_idx() runs *inside* this same step()
            # call, before it returns (manager_based_rl_env.py: physics step
            # -> compute reset_buf -> if any done, _reset_idx() -> command_
            # manager.compute() -> THEN recompute obs -> return). By the
            # time this call has returned with done=True, every asset/
            # command/entry read recorder.step() would take already belongs
            # to the *next* episode's post-reset state, not this one's last
            # step — the same hazard check_fall_termination.py's
            # last_z/last_angle comment documents. Skip recording exactly that
            # one contaminated step rather than let it silently corrupt this
            # episode's tail.
            if not done:
                recorder.step(base_env)
                if carried is not None and not carried.released:
                    box_m, box_deg = carried.carry_error()
                    held_box_err = (max(held_box_err[0], box_m), max(held_box_err[1], box_deg))
                if arm_hold is not None:
                    held_arm_err = max(held_arm_err, arm_hold.error())
                if carried is not None and command.camera_box_clearance:
                    for cam in command.moving_cameras:
                        gap = command.camera_box_clearance[cam]
                        if wrist_clear is None or gap < wrist_clear[0]:
                            wrist_clear = (gap, cam)
                if media is not None:
                    # Reading renders (this Isaac Lab renders a camera when its data is fetched), so
                    # every recorded step has exactly one frame of the state it recorded.
                    media.step(**rig.read(), step_index=len(recorder.steps) - 1)
                if writer is not None:
                    # One frame per recorded step, taken at the same instant,
                    # so frame count == recorded steps is a check, not a hope.
                    writer.write(base_env.render())
            # After the step's own work, so the sleep only holds the loop back.
            pacer.wait()
            if done:
                # Same instant recorder.step() would have to skip (the
                # reset already happened inside this env.step() call) is
                # exactly when TerminationManager's per-term buffers still
                # hold *this* step's computation — record_termination()
                # relies on that ordering (see its own docstring). Reading
                # it any later (e.g. after the loop, or on the next
                # iteration) would already see the next episode's state.
                recorder.record_termination(base_env)
                break
        if not done:
            # --max-steps below the env's own horizon (env.max_episode_length)
            # means the inner loop can exhaust without the env ever hitting a
            # termination. The explicit reset at the top of the next iteration
            # keeps that from bleeding into the next episode (it used to: two
            # files with byte-identical randomization for one episode cut in
            # half), but this episode is still truncated by the step budget
            # rather than by a termination, so it is not a normal sample.
            # check_entry.py warns on the same condition.
            print(
                f"  [주의] 에피소드 {episode_seq}: {args.max_steps} 스텝 안에 dones 가 "
                f"서지 않음 (비정상) — 이 에피소드는 종료 조건이 아니라 스텝 예산으로 "
                f"잘렸습니다. --max-steps 를 늘리십시오."
            )
        episode_wall = time.monotonic() - episode_started
        # close() finishes and stages the media; they replace an earlier run's only once the
        # episode's own npz and sidecar are written, so a failed save cannot pair them with old files.
        closed = media.close() if media is not None else None
        try:
            path = recorder.save(media=closed)
        except BaseException:
            if media is not None:
                media.discard()
            raise
        if media is not None:
            media.commit()
        elif SCENARIO_ID in CAMERA_SCENARIOS:
            # Saved without cameras: an earlier run's videos and pack under this name would be orphans.
            from generation.envs.s1a.media_writer import clear_media

            clear_media(out_dir, recorder.episode_id)
        # Saving is its own cost once there is media (the lossless videos are closed here, and a pack
        # the canon declares is compressed), so it is reported beside the episode's stepping time
        # rather than folded into it.
        save_wall = time.monotonic() - episode_started - episode_wall
        degenerate_note = " [퇴화 — 유효 스텝 미달]" if len(recorder.steps) < MIN_VALID_STEPS else ""
        if recorder.degenerate_reason is not None:
            degenerate_note = f" [퇴화 — {recorder.degenerate_reason}]"
        print(
            f"에피소드 {episode_seq} (seed {seed}): 기록 {len(recorder.steps)} 스텝 "
            f"(env 스텝 {step + 1}, {episode_wall:.1f} 초 + 저장 {save_wall:.1f} 초) → {path}{degenerate_note}"
        )
        if recorder.hri:
            # The command term's steps as the recorder took them, and the rack contacts' starts.
            racks = sum(bool(s["safety_event"][2]) for s in recorder.steps)
            print(f"  사람과 로봇 — 과제 시작 {recorder.task_started_step} · 첫 접촉 {recorder.first_touch_step} · "
                  f"상자 놓음 {recorder.box_released_step} · 과제 완료 {recorder.task_completed_step} · 랙 접촉 {racks}번")
        if carried is not None:
            # The held box against its carry pose and the held arms against their hold, over the recorded steps
            # while the box was held -- a box that slips or arms that sag show here.
            # The wrist cameras ride on the swaying arms: how near they came to the held box (below 0: inside it), and
            # the steps each robot camera spent inside it -- its visible fraction then leaves the box out.
            near = "-" if wrist_clear is None else f"{wrist_clear[0] * 1000.0:+.1f} mm ({wrist_clear[1]})"
            inside = {cam: n for cam, n in recorder.camera_in_box_steps.items() if n}
            print(f"  든 상자 — 붙은 자세와의 차이 최대 {held_box_err[0] * 1000.0:.3f} mm · {held_box_err[1]:.4f}° · "
                  f"팔 — 잡은 자세와의 차이 최대 {held_arm_err:.4f} rad · 손목 카메라와 상자 면 사이 최소 {near} · "
                  f"상자 안 카메라 스텝 {inside or 0} (기록 {len(recorder.steps)} 스텝)")
        if closed is not None and closed["rerendered_steps"]:
            # A frame came back unchanged and was drawn again (camera_io.Rig); rare -- worth seeing.
            print(f"  카메라 프레임을 다시 렌더한 스텝 {len(closed['rerendered_steps'])}개: {closed['rerendered_steps'][:10]}")
        if closed is not None and any(closed["blank_frames"].values()):
            # Frames whose every pixel is 0 (media_writer): a camera buried in some geometry or mounted wrong.
            # Seen here, at generation time; the converter refuses a camera black in every frame.
            dark = {cam: n for cam, n in closed["blank_frames"].items() if n}
            print(f"  [주의] 모든 화소가 0 인(검은) 프레임 — 카메라별 {dark} / 기록 {len(recorder.steps)} 스텝")
        if args.gui and episode_wall > 0:
            # --speed is a ceiling; say what the loop actually achieved.
            achieved = (step + 1) * base_env.step_dt / episode_wall
            print(f"  실제 배속 {achieved:.2f}x (상한 {args.speed}x, {episode_wall:.1f} 초)")
        if writer is not None:
            video_path = watch.finish_video(writer, path)
            if video_path is not None:
                print(f"  영상 {writer.frames} 프레임 → {video_path}")
            else:
                print("  영상 없음 (기록된 스텝이 없어 프레임도 없음)")
    env.close()
    compare_ok = True
    if args.compare_with:
        # Does what you watched match the batch? Rendering is not supposed to
        # touch physics, but reading the substep sensor did change trajectories
        # once (generation/README.md), so this is measured, not assumed. A
        # mismatch -- or nothing to compare -- is exit code 3, so a caller can
        # tell it from a run that simply finished.
        print(f"배치 대조: {args.compare_with}")
        report = watch.compare_with_batch(out_dir, args.compare_with)
        compare_ok, verdict_lines = watch.comparison_verdict(report)
        for line in verdict_lines:
            print(f"  {line}")
        print("  대조 통과" if compare_ok else "  대조 실패 (exit 3)")
    code = 0 if compare_ok else 3
except Exception:
    import traceback

    print("S1a 실행 실패:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    code = 1
finally:
    # A bare close() exits the process with 0 before any later line runs — a
    # failing run would report success. Pass the code into close() instead.
    simulation_app.close(exit_code=code)

# Belt-and-suspenders, matching check_scene.py's ending: close(exit_code=code)
# has been observed to hard-exit via os._exit() before this line on
# this build, but that is Kit's fast-shutdown path, not a documented
# guarantee. If a future Isaac Sim build ever returns control to Python
# instead, this is what keeps a caught failure (code = 1) from falling off
# the end of the script and exiting 0 by default.
raise SystemExit(code)
