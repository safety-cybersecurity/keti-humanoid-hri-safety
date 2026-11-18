#!/usr/bin/env python3
"""실행 환경의 버전을 수집해 `environment.lock` 을 생성한다.

재현성은 공개 데이터셋의 핵심 심사 포인트입니다. 스키마 §5 의 `sim_backend`
필드(`{isaacsim, isaaclab, physics, lerobot}`)가 에피소드마다 기록되는데, 그 값의 정본이
이 파일입니다.

**버전을 손으로 적지 않습니다.** 관절 이름과 같은 원칙입니다 — 수기로 적은 버전 문자열은
환경이 바뀌어도 그대로 남아 아무도 알아채지 못합니다.

    python3 generation/capture_environment.py            # 수집 → environment.lock
    python3 generation/capture_environment.py --check    # 현재 환경과 lock 대조

Isaac Sim 을 **부팅하지 않습니다.** 설치 트리의 VERSION 파일과 시스템 명령만 읽으므로
일반 python3 로 실행합니다. 데이터 생성 직전에 `--check` 로 드리프트를 확인하십시오.

종료 코드: 0 정상 / 1 --check 에서 불일치 / 3 실행 실패
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCK = REPO_ROOT / "environment.lock"

# 정식 릴리스가 아닌 버전을 가려내기 위한 표지
PRERELEASE = re.compile(r"(rc|alpha|beta|dev|nightly)", re.I)


def tilde(path: str | Path | None) -> str | None:
    """홈 디렉터리를 ~ 로 접는다.

    lock 파일은 공개 저장소에도 실린다. 절대 경로를
    그대로 두면 사용자 계정명이 공개물에 남으므로, 의미를 잃지 않는 선에서 접어 둔다.
    """
    if path is None:
        return None
    s = str(path)
    home = str(Path.home())
    return "~" + s[len(home):] if s == home or s.startswith(home + "/") else s


def run(cmd: list[str], timeout: int = 20) -> str | None:
    """명령을 실행해 stdout 첫 줄을 돌려준다. 실패하면 None."""
    if not shutil.which(cmd[0]):
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError):
        return None
    if r.returncode != 0:
        return None
    out = r.stdout.strip()
    return out or None


def isaacsim_version(root: Path) -> str | None:
    f = root / "VERSION"
    return f.read_text(encoding="utf-8").strip() if f.is_file() else None


def isaaclab_version(root: Path | None) -> tuple[str | None, str | None]:
    """(버전, git 커밋) 을 돌려준다. Isaac Lab 은 PyPI 가 아니라 저장소에서 설치한다.

    정본은 **git 태그**입니다. `config/extension.toml` 의 `version` 은 확장 버전이라
    릴리스 버전(v3.0.0-beta2.patch1)과 전혀 다른 값(6.1.14)이 나옵니다 — 이용자가
    체크아웃하는 식별자는 태그이므로 태그를 우선합니다. 태그를 못 얻는 배포본(tarball 등)
    에서만 extension.toml 로 물러납니다.
    """
    if root is None or not root.is_dir():
        return None, None

    version = commit = None
    if (root / ".git").exists():
        # --dirty 가 없으면 로컬에서 고친 Isaac Lab 이 태그만으로 깨끗해 보인다.
        version = run(["git", "-C", str(root), "describe", "--tags", "--always", "--dirty"])
        commit = run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"])

    if version is None:
        f = root / "source" / "isaaclab" / "config" / "extension.toml"
        if f.is_file():
            m = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']',
                          f.read_text(encoding="utf-8"), re.M)
            if m:
                version = f"extension-{m.group(1)}"  # 태그가 아님을 값에 남긴다
    return version, commit


def package_version(python_exe: Path | None, pkg: str) -> str | None:
    """지정한 인터프리터에 설치된 패키지의 버전을 읽는다."""
    if python_exe is None or not python_exe.exists():
        return None
    code = (f"import importlib.metadata as m;"
            f"print(m.version({pkg!r}))")
    return run([str(python_exe), "-c", code], timeout=60)


def gpu_info() -> dict[str, str | None]:
    line = run(["nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits"])
    if not line:
        return {"name": None, "driver": None, "memory_mib": None}
    parts = [p.strip() for p in line.split(",")]
    while len(parts) < 3:
        parts.append(None)
    return {"name": parts[0], "driver": parts[1], "memory_mib": parts[2]}


def cuda_version() -> str | None:
    out = run(["nvcc", "--version"])
    if out:
        m = re.search(r"release\s+([\d.]+)", out)
        if m:
            return m.group(1)
    # nvcc 가 없어도 드라이버가 보고하는 CUDA 는 있다
    out = run(["nvidia-smi"])
    if out:
        m = re.search(r"CUDA Version:\s*([\d.]+)", out)
        if m:
            return f"{m.group(1)} (driver)"
    return None


def os_name() -> str:
    f = Path("/etc/os-release")
    if f.is_file():
        m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', f.read_text(encoding="utf-8"), re.M)
        if m:
            return m.group(1)
    return platform.platform()


def collect(args: argparse.Namespace) -> dict:
    sim_root = Path(args.isaacsim_path).expanduser()
    lab_root = Path(args.isaaclab_path).expanduser() if args.isaaclab_path else None
    sim_python = sim_root / "kit" / "python" / "bin" / "python3"

    # 변환(lerobot)은 Isaac Sim 이 필요 없어 별도 환경에 둔다. 생성 환경에 의존성 222개를
    # 부으면 Isaac Lab 설치가 깨질 수 있기 때문이다. 어디에 있든 버전은 기록해야 하므로
    # 경로를 인자로 받고, 지정이 없으면 저장소의 .venv-lerobot → Isaac 순으로 찾는다.
    # 존재만 보고 고르면, 껍데기만 남은 venv 하나가 정상 설치를 가린다. 실제로 lerobot 을
    # 보고하는 인터프리터를 고른다. 명시 지정이 있으면 그것만 시도한다.
    if args.lerobot_python:
        candidates = [Path(args.lerobot_python).expanduser()]
    else:
        candidates = [REPO_ROOT / ".venv-lerobot" / "bin" / "python", sim_python]

    lerobot_python, lerobot_ver = candidates[-1], None
    for c in candidates:
        v = package_version(c, "lerobot")
        if v:
            lerobot_python, lerobot_ver = c, v
            break

    sim_ver = isaacsim_version(sim_root)
    lab_ver, lab_commit = isaaclab_version(lab_root)

    gpu = gpu_info()
    backend = {
        "isaacsim": sim_ver,
        "isaaclab": lab_ver,
        "physics": args.physics,
        "lerobot": lerobot_ver,
    }
    prerelease = sorted(k for k, v in backend.items() if v and PRERELEASE.search(v))
    missing = sorted(k for k, v in backend.items() if not v)

    return {
        "sim_backend": backend,
        "status": "complete" if not missing else "incomplete",
        "missing": missing,
        "prerelease": prerelease,
        "isaaclab_commit": lab_commit,
        "host": {
            "os": os_name(),
            "kernel": platform.release(),
            "arch": platform.machine(),
            "gpu": gpu["name"],
            "gpu_memory_mib": gpu["memory_mib"],
            "nvidia_driver": gpu["driver"],
            "cuda": cuda_version(),
            "python_isaacsim": run([str(sim_python), "-c",
                                    "import platform;print(platform.python_version())"]),
            "python_host": platform.python_version(),
        },
        "paths": {
            "isaacsim": tilde(sim_root) if sim_root.is_dir() else None,
            "isaacsim_assets": tilde(Path(args.assets_path).expanduser()) if args.assets_path else None,
            "isaaclab": tilde(lab_root) if lab_root and lab_root.is_dir() else None,
            "lerobot_python": tilde(lerobot_python) if lerobot_ver else None,
        },
    }


# 인용 없이 내보내도 안전한 문자열: 글자로 시작하는 식별자꼴.
# 숫자꼴(13.10 → 13.1)과 YAML 예약 스칼라(no, on …)는 걸러져 반드시 인용된다.
_PLAIN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.+/-]*$")
_RESERVED = {"null", "true", "false", "yes", "no", "on", "off", "y", "n", "~"}


def _v(x) -> str:
    if x is None:
        return "null"
    s = str(x)
    if _PLAIN.match(s) and s.lower() not in _RESERVED:
        return s
    # JSON 문자열은 YAML 1.2 double-quoted 스칼라와 문법이 같다. 직접 따옴표를 escape 하다
    # 틀리면(작은따옴표 하나로 lock 전체가 파싱 불가가 된다) 조용히 깨지므로 표준 인코더에 맡긴다.
    return json.dumps(s, ensure_ascii=False)


def render(d: dict, generated_at: str) -> str:
    L = [
        "# 자동 생성 파일 — 손으로 고치지 마십시오.",
        "# 생성: generation/capture_environment.py",
        "#",
        "# 데이터 생성 환경의 버전 고정본입니다. 스키마 §5 의 sim_backend 필드는",
        "# 이 파일의 sim_backend 를 그대로 씁니다. 생성 직전에 --check 로 대조하십시오.",
        "",
        f"captured_at: {_v(generated_at)}",
        f"status: {d['status']}",
        "",
        "# 에피소드 메타의 sim_backend 에 그대로 들어가는 값",
        "sim_backend:",
    ]
    for k, v in d["sim_backend"].items():
        L.append(f"  {k}: {_v(v)}")

    L += ["", "# 아직 확정되지 않은 구성요소 (설치 후 재수집 필요)"]
    L.append("missing: []" if not d["missing"]
             else "missing:\n" + "\n".join(f"  - {m}" for m in d["missing"]))

    L += ["",
          "# 정식 릴리스가 아닌 구성요소. 재현성 근거가 약하므로 기술문서에 명시합니다.",
          "#   — 정식 릴리스로 교체하면 이 스크립트를 다시 돌리십시오."]
    L.append("prerelease: []" if not d["prerelease"]
             else "prerelease:\n" + "\n".join(f"  - {p}" for p in d["prerelease"]))

    L += ["", f"isaaclab_commit: {_v(d['isaaclab_commit'])}", "", "host:"]
    for k, v in d["host"].items():
        L.append(f"  {k}: {_v(v)}")
    L += ["", "# 이 기기에서의 설치 경로. 재현에 필수는 아니며 참고용입니다.", "paths:"]
    for k, v in d["paths"].items():
        L.append(f"  {k}: {_v(v)}")
    return "\n".join(L) + "\n"


def check(lock_path: Path, current: dict) -> int:
    try:
        import yaml
    except ImportError:
        print("--check 에는 PyYAML 이 필요합니다 (pip install pyyaml)", file=sys.stderr)
        return 3
    if not lock_path.is_file():
        print(f"lock 파일이 없습니다: {lock_path}", file=sys.stderr)
        return 3

    locked = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}

    # YAML 은 13.1 을 float 으로, 24564 를 int 로 읽는다. 수집값은 전부 문자열이므로
    # 문자열로 맞춰 비교하지 않으면 버전이 같은데도 드리프트로 잡힌다.
    def norm(x):
        return None if x is None else str(x)

    drift = []
    # kernel·gpu_memory_mib·python 버전은 기록만 하고 비교하지 않는다. 커널은 보안 업데이트로
    # 수시로 바뀌고, 나머지는 비교 대상(isaacsim·gpu)이 같으면 따라오는 값이라 대조하면
    # 사소한 변화로 생성이 막힌다.
    for section, keys in (("sim_backend", ("isaacsim", "isaaclab", "physics", "lerobot")),
                          ("host", ("gpu", "nvidia_driver", "cuda", "os", "arch"))):
        for key in keys:
            want = norm((locked.get(section) or {}).get(key))
            got = norm(current[section][key])
            if want != got:
                drift.append(f"  {section}.{key}: lock={want!r} 현재={got!r}")

    # 태그는 옮겨 붙을 수 있지만 커밋은 그렇지 않다. Isaac Lab 코드의 실제 신원이다.
    if norm(locked.get("isaaclab_commit")) != norm(current["isaaclab_commit"]):
        drift.append(f"  isaaclab_commit: lock={locked.get('isaaclab_commit')!r} "
                     f"현재={current['isaaclab_commit']!r}")

    if drift:
        print("환경이 lock 과 다릅니다 — 데이터를 생성하면 재현성 근거가 깨집니다:")
        print("\n".join(drift))
        print("\n의도한 변경이면 --check 없이 다시 실행해 lock 을 갱신하십시오.")
        return 1

    print(f"환경이 {lock_path.name} 과 일치합니다.")
    if current["missing"]:
        print(f"참고: 미확정 구성요소 {current['missing']} — status={current['status']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="데이터 생성 환경 버전 수집")
    ap.add_argument("--isaacsim-path", default=os.environ.get("ISAACSIM_PATH", "~/isaacsim"))
    ap.add_argument("--isaaclab-path", default=os.environ.get("ISAACLAB_PATH"))
    ap.add_argument("--assets-path", default=os.environ.get("ISAAC_ASSETS_PATH"))
    ap.add_argument("--lerobot-python", default=os.environ.get("LEROBOT_PYTHON"),
                    help="lerobot 이 설치된 파이썬 (기본: .venv-lerobot → Isaac Sim 순으로 탐색)")
    ap.add_argument("--physics", default="physx", choices=["physx", "newton"],
                    help="사용 물리엔진 (스키마 §5 sim_backend.physics)")
    ap.add_argument("--out", default=str(DEFAULT_LOCK))
    ap.add_argument("--check", action="store_true", help="생성하지 않고 현재 환경과 대조")
    a = ap.parse_args()

    current = collect(a)
    lock_path = Path(a.out).expanduser()

    if a.check:
        return check(lock_path, current)

    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = render(current, stamp)

    # 손으로 쓴 emitter 는 인용 하나만 틀려도 조용히 다른 값이 되거나 파일 전체를
    # 파싱 불가로 만든다. 쓰기 전에 역파싱해 확인한다.
    try:
        import yaml
    except ImportError:
        print("참고: pyyaml 없음 — 생성물 역파싱 검증을 건너뜁니다", file=sys.stderr)
    else:
        parsed = yaml.safe_load(text)
        bad = [f"{k}: {v!r} -> {parsed.get('sim_backend', {}).get(k)!r}"
               for k, v in current["sim_backend"].items()
               if v is not None and str(parsed.get("sim_backend", {}).get(k)) != str(v)]
        if parsed is None or bad:
            print(f"생성물이 원래 값으로 역파싱되지 않습니다: {bad or 'parse=None'}",
                  file=sys.stderr)
            return 3

    lock_path.write_text(text, encoding="utf-8")

    rel = lock_path.relative_to(REPO_ROOT) if lock_path.is_relative_to(REPO_ROOT) else lock_path
    print(f"생성: {rel}")
    for k, v in current["sim_backend"].items():
        print(f"  {k:<10} {v if v else '— 미설치'}")
    if current["prerelease"]:
        print(f"\n  주의: 정식 릴리스가 아닌 구성요소 → {', '.join(current['prerelease'])}")
    if current["missing"]:
        print(f"  미확정: {', '.join(current['missing'])} (status=incomplete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
