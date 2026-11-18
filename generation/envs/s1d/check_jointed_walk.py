#!/usr/bin/env python3
"""관절 인형(v2)의 걷기 실험 — 정한 진입 속도 · 각도에서 서고 걷고 멈추는지, 놓인 뒤 어떻게 되는지를 잰다.
에피소드를 기록하지 않는다(기록기 · 변환기를 거치지 않음).

    PYTHONPATH= ~/isaacsim/python.sh generation/envs/s1d/check_jointed_walk.py \
        --cases 0.4:30 1.0:90 1.6:150 --seqs 0 3 7 --out <DIR> [--cameras] [--rigid-dummy]
        [--inertia-model whole_body_feet] [--walk-gain-scale 0.3] [--tone-scale 3.0] [--support-shares 0.3 0.5 0.7]
        [--walk-gain-scales 1.0 0.3] [--plane-freqs 2.0 1.0] [--tone-scales 0.3 1.0 3.0]

--cases 의 i 번째 사례는 S1d 의 i 번 시드로(속도:각도:시드 로 적으면 그 시드로) 리셋하되 진입 속도와 각도를 그 값 하나로
고정한다. --seqs 는 그 번호의 시드가 뽑은 그대로다. 사례마다 <DIR>/<이름>.npz · .json(카메라를 켜면 .mp4), 끝에 summary.json 과 표.
--rigid-dummy 는 v1 인형(들고 가는 붙잡기)을 같은 방식으로 잰다(비교용). --support-shares 는 붙잡는 동안 보조가 드는
몸무게의 몫을 사례마다 그 값들로 바꿔 돌린다(사례 × 몫 — 이름 끝에 -s<몫>). 없으면 환경 설정의 몫 하나.
--walk-gain-scales 는 걷는 관절 강성의 배(배율을 곱하지 않은 값에), --plane-freqs 는 보조의 평면 고유 진동수(Hz)를 같은
방식으로 사례마다 바꾼다(-w<배> · -p<Hz>). --tone-scales 는 놓인 뒤의 근육 긴장 강성의 배(--tone-scale 을 곱한 값에 —
-t<배>)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser()
parser.add_argument("--cases", nargs="*", default=[])
parser.add_argument("--seqs", nargs="*", type=int, default=[])
parser.add_argument("--out", required=True)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--cameras", action="store_true")
parser.add_argument("--rigid-dummy", action="store_true")
parser.add_argument("--inertia-model", default=None, choices=("whole_body_feet", "upper_body_hips"))
parser.add_argument("--walk-gain-scale", type=float, default=1.0)
parser.add_argument("--tone-scale", type=float, default=1.0)
parser.add_argument("--support-shares", nargs="*", type=float, default=[])
parser.add_argument("--walk-gain-scales", nargs="*", type=float, default=[])
parser.add_argument("--plane-freqs", nargs="*", type=float, default=[])
parser.add_argument("--tone-scales", nargs="*", type=float, default=[])
args = parser.parse_args()

from generation.isaac_boot import launch_app  # noqa: E402

simulation_app = launch_app({"headless": True, "enable_cameras": bool(args.cameras)})[1]
code = 1
try:
    import importlib.metadata as metadata

    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    import generation.envs.register  # noqa: F401
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_tasks.utils.hydra import resolve_presets

    import yaml

    from generation.dummy import jointed
    from generation.envs.s1a import substep_contact
    from generation.envs.s1a.recorder import CONTACT_EPS_N
    from generation.envs.s1a.mdp.safety import dummy_body_masks, non_foot_mask
    from generation.envs.s1d import jointed_walk_metrics as metrics
    from generation.envs.s1d.s1d_env_cfg import S1dEnvCfg
    from generation.scene.measure_slice import STANDING_STEP
    from generation.seeding import episode_seed, scenario_startup_seed

    def np_(v):
        return getattr(v, "torch", v).detach().cpu().numpy()

    cfg = resolve_presets(S1dEnvCfg(use_cameras=args.cameras, use_skin=True, use_jointed=not args.rigid_dummy),
                          selected=("physx",))
    cfg.seed = scenario_startup_seed("S1d")
    cmd = cfg.commands.base_velocity
    if not args.rigid_dummy:
        if args.inertia_model:
            cmd.dummy_hold_inertia_model = args.inertia_model
        for act in cfg.scene.dummy.actuators.values():
            act.stiffness *= args.walk_gain_scale
            act.damping *= args.walk_gain_scale
        cmd.dummy_walk_gains = {n: (k * args.walk_gain_scale, d * args.walk_gain_scale) for n, (k, d) in cmd.dummy_walk_gains.items()}
        cmd.dummy_tone_gains = {n: (k * args.tone_scale, d * args.tone_scale) for n, (k, d) in cmd.dummy_tone_gains.items()}
    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg))
    base = env.unwrapped
    agent_cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry("Keti-S1a-v0", "rsl_rl_cfg_entry_point"),
                                             metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Flat-H1-v0"))
    policy = runner.get_inference_policy(device=base.device)

    dummy, command = base.scene["dummy"], base.command_manager.get_term("base_velocity")
    sensor, robot_sensor = base.scene["dummy_contact"], base.scene["contact_forces"]
    names = list(sensor.body_names)
    _, ground = dummy_body_masks(jointed.read_body_codes(str(dummy.cfg.spawn.usd_path), names))
    ground_names = [n for n, g in zip(names, ground) if g]
    g_sensor = [names.index(n) for n in ground_names]
    g_body = [list(dummy.body_names).index(n) for n in ground_names]
    # The head is the body tagged BR-01 in the asset, not a prim name typed here.
    head = jointed.read_body_codes(str(dummy.cfg.spawn.usd_path), list(dummy.body_names)).index("BR-01")
    # The recorder pairs the contact sensor's per-body arrays with the articulation's by body name
    # (substep_contact.sensor_order); say whether the two orders are one.
    orders = {"sensor": names, "articulation": list(dummy.body_names), "same": names == list(dummy.body_names)}
    # The heights as run_s1a's recorder reads them -- through the sampler it installs -- checked at a standing step
    # against the dimensions: a region standing at another region's height is a pairing gone wrong on the engine.
    acc = substep_contact.install(base, contact_eps_n=CONTACT_EPS_N)
    codes = jointed.read_body_codes(str(dummy.cfg.spawn.usd_path), names)
    first_body = {c: codes.index(c) for c in sorted({c for c in codes if c is not None})}
    standing = metrics.standing_region_heights(
        yaml.safe_load((REPO / "generation/dummy/anthropometry.yaml").read_text(encoding="utf-8")))
    print(f"바디 순서 — 센서와 관절체가 {'같음' if orders['same'] else '다름'}: 센서 {names} · 관절체 {orders['articulation']}")
    robot_mask = non_foot_mask(robot_sensor.body_names)
    gravity = abs(float(base.sim.cfg.gravity[2]))
    weight = float(np_(dummy.data.body_mass)[0].sum()) * gravity
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    term = base.event_manager.get_term_cfg("reset_dummy_entry")
    drawn = {k: tuple(term.params[k]) for k in ("speed_range", "angle_range_deg")}

    import isaaclab.utils.math as mu

    def axis(v) -> np.ndarray:
        """One of the torso's own axes in the world -- through the library's own quaternion order."""
        return mu.quat_apply(dummy.data.root_quat_w.torch[0:1], torch.tensor([v], device=base.device))[0].cpu().numpy()

    def tilt_deg() -> float:
        """Angle between the torso's up and the world's."""
        return math.degrees(math.acos(max(-1.0, min(1.0, float(axis([0.0, 0.0, 1.0])[2])))))

    rig, cams = None, ("third_person_rgb", "cctv_rgb", "head_rgb")
    if args.cameras:
        import imageio_ffmpeg

        from generation import cameras as cam_canon
        from generation.envs.s1d.camera_io import Rig
        from generation.envs.s1d.s1d_env_cfg import CAMERAS

        canon = cam_canon.load(CAMERAS)
        rig = Rig(base, canon, cfg.scenario_id)          # the site the env placed its cameras at
        width = sum(int(canon["cameras"][n]["width"]) for n in cams)
        height = int(canon["cameras"][cams[0]]["height"])

    runs = []
    for i, c in enumerate(args.cases):
        parts = c.split(":")
        if len(parts) not in (2, 3):
            raise SystemExit(f"--cases 의 {c!r} 는 속도:각도 또는 속도:각도:시드 가 아닙니다")
        seq = int(parts[2]) if len(parts) == 3 else i
        runs.append((f"case{seq}-{parts[0]}_{parts[1]}", seq, (float(parts[0]), float(parts[1]))))
    runs += [(f"seq{s}", s, None) for s in args.seqs]
    if args.support_shares and args.rigid_dummy:
        raise SystemExit("--support-shares 는 관절 인형의 보조입니다 — --rigid-dummy 와 함께 쓰지 않습니다")
    if (args.walk_gain_scales or args.plane_freqs) and args.rigid_dummy:
        raise SystemExit("--walk-gain-scales · --plane-freqs 는 관절 인형의 설정입니다 — --rigid-dummy 와 함께 쓰지 않습니다")
    if args.tone_scales and args.rigid_dummy:
        raise SystemExit("--tone-scales 는 관절 인형의 설정입니다 — --rigid-dummy 와 함께 쓰지 않습니다")
    shares = args.support_shares or [command.cfg.dummy_hold_support_share]
    base_walk = dict(command.cfg.dummy_walk_gains or {})
    base_tone = dict(command.cfg.dummy_tone_gains or {})

    def suffix(share, wscale, pfreq, tscale) -> str:
        return ((f"-s{share:g}" if args.support_shares else "") + (f"-w{wscale:g}" if args.walk_gain_scales else "")
                + (f"-p{pfreq:g}" if args.plane_freqs else "") + (f"-t{tscale:g}" if args.tone_scales else ""))

    runs = [(name + suffix(share, wscale, pfreq, tscale), seq, fixed, share, wscale, pfreq, tscale)
            for share in shares for wscale in (args.walk_gain_scales or [None]) for pfreq in (args.plane_freqs or [None])
            for tscale in (args.tone_scales or [None]) for name, seq, fixed in runs]
    summary = {}
    for name, seq, fixed, share, wscale, pfreq, tscale in runs:
        command.cfg.dummy_hold_support_share = share
        command.cfg.dummy_hold_plane_freq_hz = pfreq
        if wscale is not None:
            command.cfg.dummy_walk_gains = {n: (k * wscale, d * wscale) for n, (k, d) in base_walk.items()}
        if tscale is not None:
            command.cfg.dummy_tone_gains = {n: (k * tscale, d * tscale) for n, (k, d) in base_tone.items()}
        term.params["speed_range"] = (fixed[0], fixed[0]) if fixed else drawn["speed_range"]
        term.params["angle_range_deg"] = (fixed[1], fixed[1]) if fixed else drawn["angle_range_deg"]
        base.event_manager.set_term_cfg("reset_dummy_entry", term)
        env.seed(episode_seed("S1d", seq))
        with torch.inference_mode():
            obs, _ = env.reset()
        # The reset ran physics: drop the substeps that belong to no case (run_s1a does the same).
        acc.reset()
        if wscale is not None:
            # The term writes the walking gains on a reset only once it has read its gait: write this case's here.
            command._ensure_gait(dummy)
            command._write_gains(torch.arange(base.num_envs, device=base.device), command.cfg.dummy_walk_gains)
        standing_problems = ["서 있는 스텝까지 가지 못함"]
        rows = {k: [] for k in ("hold_xy", "hold_torque", "feet_force", "feet_load", "feet_pos", "tilt", "root", "head_z",
                                "released", "triggered", "walk_ended", "gait_phase", "gait_speed", "robot_contact",
                                "foot_robot", "joint_pos", "up", "fwd", "com", "hold_force_vec", "hold_torque_vec",
                                "body_force_z")}
        entry = base.s1a_entry
        # the case's own draw: the env resets itself when the episode's time is up and draws the next case's
        case_speed = float(entry["speed"][0])
        case_angle_deg = math.degrees(float(entry["angle"][0]))
        writer = None
        if rig is not None:
            # As run_s1a: the renderer's history still holds the last case -- render the new state a few
            # times from where the third-person camera will be before any frame is kept.
            rig.reset(base)
            rig.follow()
            rig.warm(canon["render"]["reset_warm_renders"])
            writer = imageio_ffmpeg.write_frames(str(out / f"{name}.mp4"), (width, height), fps=int(canon["fps"]),
                                                 codec="libx264", pix_fmt_in="rgb24",
                                                 output_params=["-crf", "20", "-pix_fmt", "yuv420p"])
            writer.send(None)
        for step in range(args.steps):
            if rig is not None:
                rig.follow()
            with torch.inference_mode():
                obs, _, done, _ = env.step(policy(obs))
            if bool(done[0]):
                break
            peak = acc.drain()
            if step == STANDING_STEP:
                z = peak["dummy_z_at_peak"]
                standing_problems = metrics.standing_height_problems(
                    list(first_body), [z[k] for k in first_body.values()], standing)
            if writer is not None:
                got = rig.read()
                frame = np.concatenate([np.asarray(got["frames"][n]).reshape(height, -1, 3) for n in cams], axis=1)
                writer.send(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            f = np_(sensor.data.net_forces_w)[0]
            load = f[g_sensor, 2]
            rforce = np.linalg.norm(np_(robot_sensor.data.net_forces_w)[0], axis=-1)
            robot_touch = bool((rforce[robot_mask] > 1.0).any())
            feet_pos = np_(dummy.data.body_pos_w)[0][g_body]
            robot_pos = np_(base.scene["robot"].data.body_pos_w)[0]
            near = min(np.linalg.norm(robot_pos[:, :2] - p[:2], axis=1).min() for p in feet_pos)
            rows["hold_xy"].append(float(np.linalg.norm(np_(command.hold_force_in_effect)[0, :2])))
            rows["hold_torque"].append(float(np.linalg.norm(np_(command.hold_torque_in_effect)[0])))
            rows["feet_force"].append(float(load.sum()))
            rows["feet_load"].append(load.copy())
            rows["feet_pos"].append(feet_pos.copy())
            rows["tilt"].append(tilt_deg())
            rows["root"].append(np_(dummy.data.root_pos_w)[0].copy())
            rows["head_z"].append(float(np_(dummy.data.body_pos_w)[0, head, 2]))
            rows["released"].append(bool(command.hold_released[0]))
            rows["triggered"].append(bool(entry["triggered"][0]))
            rows["walk_ended"].append(bool(entry.get("walk_ended", torch.zeros(1, dtype=torch.bool))[0]))
            rows["gait_phase"].append(float(getattr(command, "gait_phase", torch.zeros(1))[0]))
            rows["gait_speed"].append(float(getattr(command, "gait_speed", torch.zeros(1))[0]))
            rows["robot_contact"].append(robot_touch)
            # an approximation: a foot loaded past 1.5x its standing share with a robot link within 0.3 m of it
            rows["foot_robot"].append(bool((load > 0.75 * weight).any() and near < 0.3))
            rows["joint_pos"].append(np_(dummy.data.joint_pos)[0].copy() if len(dummy.joint_names) else np.zeros(0))
            # what tips the body: the torso's axes, the centre of mass, the assist as vectors, and the floor-normal
            # contact force on every dummy body (the contact sensor's order -- names in the json)
            rows["up"].append(axis([0.0, 0.0, 1.0]))
            rows["fwd"].append(axis([1.0, 0.0, 0.0]))
            mass = np_(dummy.data.body_mass)[0]
            rows["com"].append((mass[:, None] * np_(dummy.data.body_pos_w)[0]).sum(axis=0) / mass.sum())
            rows["hold_force_vec"].append(np_(command.hold_force_in_effect)[0].copy())
            rows["hold_torque_vec"].append(np_(command.hold_torque_in_effect)[0].copy())
            rows["body_force_z"].append(f[:, 2].copy())
        if writer is not None:
            writer.close()
        arrays = {k: np.asarray(v) for k, v in rows.items()}
        np.savez(out / f"{name}.npz", **arrays)
        s = metrics.summarize(arrays, weight, base.step_dt, command.cfg.dummy_hold_support_share or 0.0)
        e = metrics.evaluate(s)
        meta = {"seq": seq, "fixed": fixed, "support_share": command.cfg.dummy_hold_support_share,
                "walk_gain_scale": wscale, "plane_freq_hz": command.cfg.dummy_hold_plane_freq_hz,
                "tone_scale": args.tone_scale * (tscale if tscale is not None else 1.0),
                "speed": case_speed,
                "angle_deg": case_angle_deg, "summary": s, "marks": e,
                "joint_names": list(dummy.joint_names), "foot_robot_note": "근사 — 발 하중과 로봇 링크 거리로 셈",
                "standing_problems": standing_problems, "body_force_names": names}
        (out / f"{name}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        summary[name] = meta
        print(f"{name:26s} v {meta['speed']:.2f} a {meta['angle_deg']:5.1f} | "
              + " ".join(f"{k}={'-' if v is None else 'O' if v else 'X'}" for k, v in e.items())
              + f" standing={'X' if standing_problems else 'O'}"
              + f" | hold {s['hold_xy_walk_median_frac']:.3f} grf {s['grf_walk_median_frac']:.3f}"
              f" flight {s['flight_frac']:.3f} torque {s['tilt_torque_walk_median_nm']:.0f}"
              f" slip {s['max_stance_slip_m']:.3f} tilt {s['max_tilt_deg']:.1f}", flush=True)
    summary["body_orders"] = orders
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    env.close()
    code = 0
except Exception:
    import traceback
    traceback.print_exc()
    code = 1
finally:
    simulation_app.close(exit_code=code)
raise SystemExit(code)
