#!/usr/bin/env python3
"""자산에서 관절 이름·순서를 추출해 `meta/joint_map/{robot_model}.yaml` 을 생성한다.

스키마 정의서 v0.1 §7.1 의 절대 원칙 — **관절 순서를 손으로 적지 않는다** — 을 실행하는
도구다. 이 스크립트의 출력이 관절 이름·순서의 유일한 정본이며, 규격서 §7.3 의 구성표와
픽스처의 자리표시 값은 여기서 나온 값으로 교체한다.

    ~/isaacsim/python.sh generation/dump_joint_names.py \
        --usd  ~/isaacsim_assets/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/Unitree/H1/h1.usd \
        --robot-model unitree_h1

**왜 시뮬레이터를 띄우는가.** USD 파일을 파싱하는 것만으로는 부족하다. Isaac 은 물리엔진이
아티큘레이션을 파싱한 뒤의 순서를 상태 배열에 쓰고, 그 순서는 USD 순회 순서와 다를 수 있다.
Isaac Sim 자신도 두 순서가 어긋나면 경고를 남긴다 (`articulation.py` 의 "Joint names mismatch").
따라서 타임라인을 재생해 물리 텐서 뷰가 초기화된 뒤에 읽어야 한다.

**joint 와 DOF 는 다르다.** `Articulation.joint_names` 는 고정관절을 포함한 전체 관절이고,
`observation.state.*` 의 `(dof,)` 배열이 따르는 것은 `Articulation.dof_names` 다. 규격서 §5 의
`joint_names` 필드는 후자를 가리키므로 이 스크립트는 후자를 `joint_names` 로 기록하고,
전체 관절 목록은 `articulation_detail` 에 참고용으로 남긴다.

종료 코드: 0 정상 / 1 추출 실패 / 3 인자 오류.
다만 `python.sh` 는 0 이 아닌 코드를 모두 1 로 접으므로(`error_exit`), 런처를 거치면
성공(0)과 실패(비 0)만 구별됩니다. 세부 코드가 필요하면 번들 파이썬을 직접 부르십시오.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# `~/isaacsim/python.sh <script>` puts only the script's own directory on
# sys.path, so `generation` is not importable without this.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.isaac_boot import launch_simulation_app  # noqa: E402

# §7.2 의미 키 명명 규칙. 이 어휘에 없는 토큰이 나오면 사람이 확인해야 한다.
SIDES = {"left": "left", "l": "left", "right": "right", "r": "right"}
CHAINS = ("hip", "knee", "ankle", "waist", "torso", "shoulder", "elbow", "wrist")
AXES = ("roll", "pitch", "yaw")

# 이름에 붙지만 의미를 담지 않는 토큰
NOISE = {"joint", "jnt", "link", "unitree"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="자산에서 관절 이름·순서를 추출해 매핑 파일을 생성한다",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--usd", required=True, help="로봇 USD 경로")
    p.add_argument("--robot-model", required=True,
                   help="스키마 §5 의 robot_model 값. 예: unitree_h1")
    p.add_argument("--out", default=None,
                   help="출력 경로 (기본: meta/joint_map/{robot_model}.yaml)")
    p.add_argument("--variant", action="append", default=[], metavar="SET=VALUE",
                   help="USD 배리언트 선택. 반복 지정 가능. 예: --variant Hands=None")
    p.add_argument("--missing-vs", default=None, metavar="YAML",
                   help="다른 로봇의 매핑 파일과 대조해 `missing` 을 채운다")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="물리 연산 장치 (기본 cpu — 메타데이터 조회에는 GPU 가 필요 없다)")
    p.add_argument("--warmup-steps", type=int, default=4,
                   help="텐서 뷰 초기화를 기다리는 업데이트 횟수")
    return p.parse_args()


# ── 의미 키 유도 ──────────────────────────────────────────────────────────────

def to_canonical(native: str) -> tuple[str | None, str | None]:
    """네이티브 관절 이름에서 §7.2 의미 키를 유도한다.

    반환: (의미 키 또는 None, 확인이 필요한 사유 또는 None)

    이름에서 규칙적으로 유도할 수 있는 것만 유도한다. 축 토큰이 없으면 축을 지어내지 않고
    사유를 붙여 돌려보낸다 — 여기서 추측하면 §7.1 이 금지하는 수기 기재와 같아진다.
    """
    tokens = [t for t in re.split(r"[_\-.]+", native.lower()) if t and t not in NOISE]

    side = chain = axis = None
    for t in tokens:
        if side is None and t in SIDES:
            side = SIDES[t]
        elif chain is None and t in CHAINS:
            chain = t
        elif axis is None and t in AXES:
            axis = t

    if chain is None:
        return None, "운동사슬 토큰 없음"
    if side is None:
        side = "center"

    key = f"{side}_{chain}" + (f"_{axis}" if axis else "")
    return key, None if axis else "축 토큰 없음 — 회전축 확인 필요"


def build_mapping(dof_names: list[str]) -> dict:
    """전체 DOF 에 대해 의미 키를 유도하고 이상 항목을 분류한다."""
    canonical_map: dict[str, str] = {}
    unmapped: list[str] = []
    needs_review: list[str] = []

    for name in dof_names:
        key, reason = to_canonical(name)
        if key is None:
            unmapped.append(f"{name}  # {reason}")
            continue
        canonical_map[name] = key
        if reason:
            needs_review.append(f"{name} -> {key}  # {reason}")

    # 두 관절이 같은 의미 키로 떨어지면 매핑이 깨진 것이다
    seen: dict[str, str] = {}
    conflicts: list[str] = []
    for native, key in canonical_map.items():
        if key in seen:
            conflicts.append(f"{key}  # {seen[key]}, {native}")
        else:
            seen[key] = native

    return {"canonical_map": canonical_map, "unmapped": unmapped,
            "needs_review": needs_review, "conflicts": conflicts}


# ── YAML 출력 ────────────────────────────────────────────────────────────────
# pyyaml 로는 주석을 남길 수 없고, 이 파일은 사람이 읽는 규격 산출물이라 주석이 값을 한다.
# 출력 구조가 단순하므로(스칼라 · 문자열 리스트 · 문자열 매핑) 직접 쓰되, 쓰기 직전에
# 역파싱해 값이 보존되는지 확인한다.

# 인용 없이 내보내도 안전한 문자열: 글자로 시작하는 식별자꼴.
# 숫자꼴(13.10 → 13.1)이나 YAML 예약 스칼라(null, no, on …)는 여기 걸리지 않으므로
# 아래 _RESERVED 검사와 함께 반드시 인용된다.
_PLAIN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.+/-]*$")
_RESERVED = {"null", "true", "false", "yes", "no", "on", "off", "y", "n", "~"}


def _scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    if _PLAIN.match(s) and s.lower() not in _RESERVED:
        return s
    # JSON 문자열은 YAML 1.2 의 double-quoted 스칼라와 문법이 같다. 따옴표·개행·제어문자를
    # 직접 처리하다 틀리는 것보다 표준 인코더에 맡기는 편이 안전하다.
    return json.dumps(s, ensure_ascii=False)


def _emit_list(key: str, items: list, indent: str = "") -> list[str]:
    if not items:
        return [f"{indent}{key}: []"]
    out = [f"{indent}{key}:"]
    for it in items:
        # "값  # 주석" 형태는 주석을 살려서 내보낸다
        if isinstance(it, str) and "  # " in it:
            val, comment = it.split("  # ", 1)
            out.append(f"{indent}  - {_scalar(val)}  # {comment}")
        else:
            out.append(f"{indent}  - {_scalar(it)}")
    return out


def _emit_map(key: str, mapping: dict, indent: str = "") -> list[str]:
    if not mapping:
        return [f"{indent}{key}: {{}}"]
    out = [f"{indent}{key}:"]
    for k, v in mapping.items():
        out.append(f"{indent}  {_scalar(k)}: {_scalar(v)}")
    return out


def render_yaml(d: dict) -> str:
    L: list[str] = [
        "# 자동 생성 파일 — 손으로 고치지 마십시오.",
        "# 생성: generation/dump_joint_names.py  (스키마 정의서 v0.1 §7.4)",
        "#",
        "# 관절 이름과 순서의 정본입니다. 규격서 §7.3 의 구성표와 검증기 픽스처는",
        "# 이 파일을 따라갑니다. 자산이 바뀌면 이 스크립트를 다시 돌리십시오.",
        "",
        f"robot_model: {_scalar(d['robot_model'])}",
        f"dof: {d['dof']}",
        "",
        "# 정본 — observation.state.* 의 (dof,) 배열이 따르는 순서.",
        "# Articulation.dof_names, 즉 물리엔진이 보고하는 DOF 순서입니다.",
        "# 고정관절은 상태 배열에 들어가지 않으므로 여기 없습니다 (articulation_detail 참조).",
    ]
    L += _emit_list("joint_names", d["joint_names"])
    L += [
        "",
        "# 네이티브 이름 → 의미 키 (§7.2). 이름에서 규칙적으로 유도한 값입니다.",
    ]
    L += _emit_map("canonical_map", d["canonical_map"])
    L += [
        "",
        "# 의미 키로 해석하지 못한 관절. 비어 있어야 정상입니다.",
    ]
    L += _emit_list("unmapped", d["unmapped"])
    L += [
        "",
        "# 이름만으로 회전축을 판별하지 못한 관절. 자산의 회전축을 보고 확정해야 합니다.",
    ]
    L += _emit_list("needs_review", d["needs_review"])
    L += [
        "",
        "# 서로 다른 관절이 같은 의미 키로 떨어진 경우. 비어 있어야 정상입니다.",
    ]
    L += _emit_list("conflicts", d["conflicts"])
    L += [
        "",
        "# 다른 로봇에는 있고 이 로봇에는 없는 의미 키 (--missing-vs 로 계산).",
        "# 2차년도에 H1·G1 데이터를 함께 학습할 때 결손을 패딩이 아니라 마스크로 처리하는 근거입니다.",
    ]
    L += _emit_list("missing", d["missing"])
    L += ["", "provenance:"]
    for k, v in d["provenance"].items():
        if isinstance(v, dict):
            L += _emit_map(k, v, indent="  ")
        else:
            L.append(f"  {k}: {_scalar(v)}")
    L += [
        "",
        "# 참고 — DOF 가 아닌 관절 전체. 고정관절은 상태 배열에 포함되지 않습니다.",
        "articulation_detail:",
        f"  num_joints: {d['articulation_detail']['num_joints']}",
    ]
    L += _emit_list("joint_names_all", d["articulation_detail"]["joint_names_all"], indent="  ")
    L += _emit_list("non_dof_joints", d["articulation_detail"]["non_dof_joints"], indent="  ")
    L += _emit_map("dof_types", d["articulation_detail"]["dof_types"], indent="  ")
    L += [
        "",
        "# 관절 한계 [하한, 상한]. 회전 관절은 rad, 병진 관절은 m.",
        "# 검증기 C8 의 관절 범위 검사가 참조할 근거값입니다.",
        "dof_limits:",
    ]
    for name, (lo, hi) in d["dof_limits"].items():
        L.append(f"  {_scalar(name)}: [{lo!r}, {hi!r}]")

    return "\n".join(L) + "\n"


# ── 본체 ─────────────────────────────────────────────────────────────────────

def tilde(path: Path) -> str:
    """홈 디렉터리를 ~ 로 접는다 — 매핑 파일은 공개 대상이므로 계정명을 남기지 않는다."""
    s, home = str(path), str(Path.home())
    return "~" + s[len(home):] if s == home or s.startswith(home + "/") else s


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_missing_vs(path: Path, own_keys: set[str]) -> list[str]:
    """다른 로봇의 매핑 파일을 읽어 이 로봇에 없는 의미 키를 추린다."""
    keys: set[str] = set()
    in_map = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("canonical_map:"):
            in_map = True
            continue
        if in_map:
            if line.startswith((" ", "\t")):
                if ":" in line:
                    keys.add(line.split(":", 1)[1].split("#")[0].strip())
            elif line.strip():
                break
    return sorted(keys - own_keys)


def main() -> int:
    args = parse_args()

    usd_path = Path(args.usd).expanduser().resolve()
    if not usd_path.is_file():
        print(f"USD 파일을 찾을 수 없습니다: {usd_path}", file=sys.stderr)
        return 3

    variants = []
    for spec in args.variant:
        if "=" not in spec:
            print(f"--variant 형식은 SET=VALUE 입니다: {spec}", file=sys.stderr)
            return 3
        k, v = spec.split("=", 1)
        variants.append((k.strip(), v.strip()))

    out_path = (Path(args.out).expanduser().resolve() if args.out
                else REPO_ROOT / "meta" / "joint_map" / f"{args.robot_model}.yaml")

    # SimulationApp 은 Omniverse 계열 import 보다 먼저 생성되어야 한다 (확장 로딩 순서).
    # 부팅은 헬퍼를 거친다 — 아래 try/finally 는 simulation_app 이 생긴 뒤에야
    # 걸리고, 그전에 실패하면 Kit 의 atexit 가 종료 코드 0 으로 덮는다
    # (generation/isaac_boot.py 참고).
    simulation_app = launch_simulation_app()

    import isaacsim.core.experimental.utils.stage as stage_utils
    import omni.timeline
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.simulation_manager import SimulationManager

    # SimulationApp.close() 는 Kit 의 fast shutdown 경로로 프로세스를 끝내므로, 예외를
    # 그냥 두면 트레이스백이 출력되지 않고 종료 코드도 0 으로 덮인다. 여기서 직접 잡아
    # 출력하고, 종료 코드를 close() 에 넘겨 보존한다.
    code = 0
    try:
        SimulationManager.set_physics_sim_device(args.device)
        simulation_app.update()

        stage_utils.create_new_stage()
        robot_prim = stage_utils.add_reference_to_stage(
            usd_path=str(usd_path), path="/World/robot",
            variants=variants if variants else None,
        )
        while stage_utils.is_stage_loading():
            simulation_app.update()

        # 요청한 배리언트가 아니라 **실제로 해석된** 선택을 기록한다. 자산의 기본 배리언트가
        # 바뀌면 DOF 구성이 달라지므로, 재현에 필요한 것은 요청값이 아니라 결과값이다.
        # 조회 실패와 "배리언트가 없는 자산" 은 결과가 똑같이 비어 보인다. provenance 에서
        # 구분되지 않으면 배리언트가 바뀌어 DOF 구성이 달라져도 알 수 없으므로 사유를 남긴다.
        resolved_variants: dict[str, str] = {}
        variants_query_error: str | None = None
        try:
            vsets = robot_prim.GetVariantSets()
            for vname in vsets.GetNames():
                resolved_variants[vname] = vsets.GetVariantSet(vname).GetVariantSelection()
        except Exception as e:
            variants_query_error = f"{e.__class__.__name__}: {e}"
            print(f"경고: 배리언트 조회 실패 ({variants_query_error})", file=sys.stderr)

        roots = Articulation.fetch_articulation_root_api_prim_paths("/World/robot")
        root = next((r for r in roots if r), None)
        if root is None:
            print("아티큘레이션 루트를 찾지 못했습니다 — 물리 스키마가 없는 자산일 수 있습니다.",
                  file=sys.stderr)
            raise SystemExit(1)

        robot = Articulation(root)

        # 타임라인을 재생해야 물리 텐서 뷰가 초기화되고, 그때의 순서가 상태 배열의 순서다.
        omni.timeline.get_timeline_interface().play()
        for _ in range(max(1, args.warmup_steps)):
            simulation_app.update()

        dof_names = list(robot.dof_names)
        joint_names_all = list(robot.joint_names)
        num_dofs = int(robot.num_dofs)
        num_joints = int(robot.num_joints)

        if len(dof_names) != num_dofs:
            # 이름 배열과 DOF 수가 다르면 상태 배열을 해석할 근거가 깨진 것이다.
            # 경고만 남기고 파일을 쓰면 모순된 매핑이 정본으로 굳는다.
            print(f"오류: dof_names {len(dof_names)} != num_dofs {num_dofs} "
                  "— 관절 맵을 생성하지 않습니다", file=sys.stderr)
            raise SystemExit(1)

        dof_types = {n: str(t).split(".")[-1].split(":")[0].strip()
                     for n, t in zip(dof_names, robot.dof_types)}

        # get_dof_limits() 는 (하한, 상한) 두 배열을 돌려준다. 각각 shape (N, D).
        lower, upper = robot.get_dof_limits()
        lo, hi = lower.numpy()[0], upper.numpy()[0]
        dof_limits = {n: (round(float(lo[i]), 6), round(float(hi[i]), 6))
                      for i, n in enumerate(dof_names)}

        mapping = build_mapping(dof_names)

        missing: list[str] = []
        if args.missing_vs:
            ref = Path(args.missing_vs).expanduser().resolve()
            if not ref.is_file():
                print(f"--missing-vs 파일 없음: {ref}", file=sys.stderr)
                # return 으로 빠져나가면 finally 의 code 가 0 이라 close(exit_code=0) 이
                # 호출되고 인자 오류가 성공으로 끝난다. 예외로 올려 code 에 반영시킨다.
                raise SystemExit(3)
            missing = load_missing_vs(ref, set(mapping["canonical_map"].values()))

        from isaacsim.core.version import get_version
        version = get_version()
        isaacsim_version = version[0] if isinstance(version, (list, tuple)) else str(version)

        doc = {
            "robot_model": args.robot_model,
            "dof": num_dofs,
            "joint_names": dof_names,
            **mapping,
            "missing": missing,
            "provenance": {
                "source": "isaacsim_physx_tensor_dof_order",
                "generated_by": "generation/dump_joint_names.py",
                "isaacsim_version": isaacsim_version,
                "physics_engine": str(SimulationManager.get_active_physics_engine()),
                "physics_device": args.device,
                "usd_path": tilde(usd_path),
                "usd_sha256": sha256_of(usd_path),
                "articulation_root": root,
                "variants_requested": dict(variants),
                "variants_resolved": resolved_variants,
                "variants_query_error": variants_query_error,
            },
            "articulation_detail": {
                "num_joints": num_joints,
                "joint_names_all": joint_names_all,
                "non_dof_joints": [j for j in joint_names_all if j not in set(dof_names)],
                "dof_types": dof_types,
            },
            "dof_limits": dof_limits,
        }

        text = render_yaml(doc)

        # 수기 emitter 는 인용 하나만 틀려도 조용히 다른 값이 된다. 쓰기 전에 역파싱해
        # 정본이 원래 값으로 돌아오는지 확인한다. pyyaml 이 없는 환경에서는 건너뛴다.
        try:
            import yaml
        except ImportError:
            print("참고: pyyaml 없음 — 생성물 역파싱 검증을 건너뜁니다", file=sys.stderr)
        else:
            back = yaml.safe_load(text)
            if not isinstance(back, dict):
                print("생성물이 매핑으로 파싱되지 않습니다", file=sys.stderr)
                raise SystemExit(1)
            mismatch = [k for k in ("robot_model", "dof", "joint_names", "canonical_map")
                        if back.get(k) != doc[k]]
            if mismatch:
                print(f"생성물이 원래 값으로 역파싱되지 않습니다: {mismatch}", file=sys.stderr)
                raise SystemExit(1)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

        rel = out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path
        print(f"\n생성: {rel}")
        print(f"  robot_model  {args.robot_model}")
        print(f"  dof          {num_dofs}   (전체 관절 {num_joints})")
        print(f"  매핑         {len(mapping['canonical_map'])} / {num_dofs}")
        if mapping["unmapped"]:
            print(f"  미매핑       {len(mapping['unmapped'])}  ← 확인 필요")
        if mapping["needs_review"]:
            print(f"  축 미확정    {len(mapping['needs_review'])}  ← 확인 필요")
        if mapping["conflicts"]:
            print(f"  키 충돌      {len(mapping['conflicts'])}  ← 확인 필요")

    except SystemExit as e:
        code = int(e.code or 0)
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close(exit_code=code)

    return code


if __name__ == "__main__":
    raise SystemExit(main())
