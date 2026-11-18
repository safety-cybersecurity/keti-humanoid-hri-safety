#!/usr/bin/env python3
"""KS B ISO TS 15066 부속서 A 기반 인체 더미 USD 를 생성한다.

부위 구분은 표 A.1/A.2 에서, 유효질량·유효탄성계수는 **표 A.3** 에서 옵니다.

    ~/isaacsim/python.sh generation/dummy/build_dummy.py

부위별 유효질량·스프링상수의 정본은 `meta/rules.json` 의 `contact_limits` 이고, 형상 치수의
정본은 `generation/dummy/anthropometry.yaml` 이다. 이 스크립트는 어느 값도 스스로 정하지
않는다. 관절 이름을 손으로 적지 않는 것과 같은 이유다 — 코드에 박힌 수치는 정본이 바뀌어도
그대로 남고 아무도 알아채지 못한다.

**형상을 만드는 데 쓰는 것은 치수뿐이다.** 유효질량·유효탄성계수는 **채워져 있을 것을
요구할 뿐 값을 읽지 않는다** — 조립에 들어가는 `contact_limits` 항목은 `name_ko` 하나다.
그 둘은 두 물체 충돌 모형(식 A.1)의 파라미터이지 자유 물체의 질량이 아니어서, 강체 질량으로
쓰면 12부위 합이 인체가 아닌 크기가 된다. 그래서 강체 질량을
명시하지 않고 물리 엔진이 형상과 기본 밀도로 계산한다.

**그래도 비어 있으면 만들지 않는다.** 요구를 남겨 두는 이유는 정본이 완전한 상태에서만
자산이 나가게 하기 위해서다. 값이 빠진 채로 만든 더미의 접촉 하중은 표준과 아무 관계가
없는데도 관계가 있는 것처럼 보인다. 그래서 기본 동작은 실패이며, 형상과 접촉 판정 해상도만
확인하려면 `--geometry-only` 를 명시해야 한다. 그 경우 산출물에 잠정 표시가 박힌다.
(2026-08-27 국가기술표준원 이용 허락으로 `contact_limits` 의 값 70개가 정본에 실려 있다.)

**부위와 프림은 1:1 이 아니다.** Table A.2 는 좌우를 구분하지 않으므로(예: "상완/팔꿈치"는
한 부위), 좌우 한 쌍인 부위는 프림 두 개가 같은 BR 코드를 갖는다. 스텝 필드
`observation.contact.force` 는 부위당 한 값(12개)이므로, 라벨링 모듈이 같은 코드의 프림들을
합산이 아니라 **최대값**으로 모아야 한다. 합산하면 양팔에 동시에 스친 접촉이 한쪽에 세게
맞은 것처럼 보인다.

산출물은 `*.usd` 라 저장소에 담지 않는다. 정본은 이 스크립트와 두 데이터 파일이다.

종료 코드: 0 정상 / 1 생성 실패 / 2 임계값 미확보 / 3 인자 오류.
다만 `python.sh` 는 0 이 아닌 코드를 모두 1 로 접으므로, 세부 코드가 필요하면 번들 파이썬을
직접 부르십시오.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RULES = REPO_ROOT / "meta" / "rules.json"
ANTHRO = Path(__file__).resolve().parent / "anthropometry.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from generation.appearance import SHIRT_MATERIAL, TROUSERS_MATERIAL  # noqa: E402,F401 — pure numpy
from generation.dummy.skin import region_split as rs  # noqa: E402 — pure numpy

# 좌우 한 쌍인 부위가 만드는 프림 접미사와 이름 규칙. 정본은 region_split 이다 — 시각 메시를
# 부위별로 가르는 쪽이 같은 이름을 써야 서브메시가 자기 강체 밑에 붙는다.
SIDES = rs.SIDES
prim_names = rs.prim_names

# --skin 이 붙이는 자식 프림 이름과 재질 스코프. 그때 콜라이더도 자식으로 내려간다.
SKIN_CHILD = "skin"
COLLIDER_CHILD = "collider"
# --vest hangs one more visual child beside `skin` on each trunk body the vest covers
VEST_CHILD = "vest"
LOOKS = "/KetiDummy/Looks"
TEXTURE_SUBDIR = "textures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ISO/TS 15066 부위 구분을 따르는 인체 더미 USD 를 생성한다",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", default=None,
                   help="출력 USD 경로 (기본: generation/dummy/build/keti_dummy_ts15066_v1.usd, "
                        "--jointed 면 keti_dummy_ts15066_v2_jointed.usd)")
    p.add_argument("--geometry-only", action="store_true",
                   help="임계값 없이 형상만 만든다. 접촉 판정 해상도 확인 전용이며 "
                        "산출물에 잠정 표시가 박힌다")
    p.add_argument("--root", choices=["dynamic", "kinematic"], default="dynamic",
                   help="루트 강체의 구동 방식 (기본 dynamic). kinematic 은 대본대로 움직이지만 "
                        "무한 질량으로 취급되어 유효질량이 접촉력에 반영되지 않는다")
    p.add_argument("--skin", default=None, metavar="OBJ",
                   help="시각 메시 OBJ (make_skin.py 산출물). 주면 부위 프림마다 콜라이더 없는 "
                        "자식 메시를 붙이고 프리미티브는 purpose=guide 로 숨긴다. 물리는 그대로다. "
                        "출력 기본값이 keti_dummy_ts15066_v1_skin.usd 가 되고 textures/ 가 옆에 생긴다")
    p.add_argument("--jointed", action="store_true",
                   help="관절 인형(v2)을 만든다 — generation/dummy/joints.yaml 의 관절 · 발 · 부위 질량. "
                        "출력 기본값이 keti_dummy_ts15066_v2_jointed{,_skin}.usd 가 된다")
    p.add_argument("--gait", default=None, metavar="JSON",
                   help="--jointed 에 실을 걸음 표(build_gait.py 산출물)")
    p.add_argument("--vest", default=None, metavar="YAML",
                   help="조끼 명세(generation/dummy/vest.yaml). --jointed · --skin 과 함께만 — 몸통 옷 위에 덮은 매끈한 "
                        "형광 조끼를 그것이 덮는 몸통 · 골반 부위 몸마다 보이기만 하는 자식(vest)으로 붙인다. 물리는 그대로다. "
                        "출력 기본값이 keti_dummy_ts15066_v2_jointed_skin_vest.usd 가 된다")
    p.add_argument("--seams", default=None, metavar="YAML",
                   help="피부 이음새 명세(generation/dummy/skin_seams.yaml). --jointed · --skin 과 함께만 — 엉뚱한 몸에 든 피부 "
                        "조각을 둘레의 몸으로 옮겨 가르고, 관절 이음새 근처의 피부 정점마다 이웃 몸과 섞는 비율을 그 피부 메시에 싣는다. "
                        "물리는 그대로다")
    p.add_argument("--split-worksuit", action="store_true",
                   help="--skin 과 함께만 — 작업복 재질의 면을 흰 티셔츠의 UV 섬과 나머지(바지)로 갈라 입력이 같은 재질 둘"
                        f"({SHIRT_MATERIAL} · {TROUSERS_MATERIAL})에 묶는다. 지은 겉모습은 그대로이고 상의와 바지의 색을 "
                        "따로 바꿀 자리만 생긴다. 물리는 그대로다")
    return p.parse_args()


def load_limits(geometry_only: bool) -> tuple[dict, list[str]]:
    """부위별 생체역학 파라미터를 정본에서 읽는다. 비어 있으면 그 목록을 함께 돌려준다.

    `meta/rules.json` 은 standard 임계값을 **값과 조항 참조를 함께** 싣습니다 —
    2026-08-27 국가기술표준원 이용 허락으로 부속서 A 의 값이 정본으로 옮겨졌습니다
    (스키마 §9.3). 그래서 이 파일만 읽어도 질량·강성 24건이 채워집니다.
    표준 전사본(공개 저장소에는 없음) 병합은 라벨러와 같은 방식으로 남겨 두되 **이미 채워진
    값을 덮지 않으므로**, 전사본이 없는 저장소도 같은 더미를 만듭니다.
    """
    if not RULES.exists():
        raise SystemExit(f"판정 규칙 파일이 없습니다: {RULES.relative_to(REPO_ROOT)}")
    d = json.loads(RULES.read_text(encoding="utf-8"))

    # 라벨러와 같은 병합. 두 곳이 다른 방식으로 읽으면 더미와 판정이 서로 다른
    # 수치를 쓰게 됩니다 (CONTRIBUTING.md 「함께 고치는 곳」).
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from generation.labeling.label_episodes import (load_standard_values,
                                                        merge_standard_values)
        values = load_standard_values()
        if values:
            merge_standard_values(d, values)
    except ImportError:                      # 표준 전사본이 없는 저장소
        pass
    limits = d.get("contact_limits")
    if not isinstance(limits, dict) or not limits:
        raise SystemExit("rules.json 에 contact_limits 가 없습니다")

    out: dict[str, dict] = {}
    missing: list[str] = []
    for code, entry in limits.items():
        bio = (entry or {}).get("biomechanical") or {}
        vals = {}
        for key in ("effective_mass", "spring_constant"):
            item = bio.get(key) or {}
            if not isinstance(item, dict) or "value" not in item:
                raise SystemExit(f"rules.json 의 {code}.biomechanical.{key} 형식이 예상과 다릅니다")
            v = item.get("value")
            # None 만 걸러서는 부족하다. 0 이나 음수, NaN 을 자리표시로 넣어 두면
            # "채워졌다" 로 통과해 질량 0 짜리 더미가 만들어지고, 그 접촉력은 표준과
            # 아무 관계가 없는데도 관계가 있는 것처럼 보인다.
            if v is None:
                missing.append(f"{code}.{key} (미기재)")
            elif not isinstance(v, (int, float)) or isinstance(v, bool):
                missing.append(f"{code}.{key} (수치 아님: {v!r})")
            elif not math.isfinite(float(v)) or float(v) <= 0:
                missing.append(f"{code}.{key} (양수여야 함: {v!r})")
            vals[key] = v
            vals[f"{key}_unit"] = item.get("unit")
        vals["name_ko"] = (entry or {}).get("name_ko")
        vals["name_en"] = (entry or {}).get("name_en")
        out[code] = vals

    if missing and not geometry_only:
        print(f"임계값 {len(missing)}건이 쓸 수 없는 상태라 더미를 만들지 않습니다.",
              file=sys.stderr)
        print(f"  예: {', '.join(missing[:4])}", file=sys.stderr)
        print("  meta/rules.json 의 contact_limits 를 표준 사본에서 채우십시오.", file=sys.stderr)
        print("  형상과 접촉 판정 해상도만 확인하려면 --geometry-only 를 붙이십시오.",
              file=sys.stderr)
        raise SystemExit(2)
    return out, missing


def load_anthropometry() -> dict:
    import yaml                                    # Isaac 번들에 포함되어 있다
    if not ANTHRO.exists():
        raise SystemExit(f"치수 파일이 없습니다: {ANTHRO.relative_to(REPO_ROOT)}")
    d = yaml.safe_load(ANTHRO.read_text(encoding="utf-8"))
    regions = d.get("regions")
    if not isinstance(regions, dict) or not regions:
        raise SystemExit("anthropometry.yaml 에 regions 가 없습니다")
    return d


# ---- 시각 메시 (--skin) ------------------------------------------------------------------

def engine_anchor(local_in_scaled_frame, scale) -> tuple[float, float, float]:
    """A joint anchor as the physics engine reads it off v1.

    v1 stores the anchor in its box body's *scaled* frame as float32, and the engine
    multiplies it back by the box scale in float32. The skinned asset's body is an
    unscaled Xform, so it has to author that product itself. The plain unscaled offset
    is the same point to 1e-8 m but not the same float32: 4 of the 16 anchors differ by
    one ulp, and a closed-loop episode grows that into a different outcome (measured
    2026-09-23 — S1a episode 0 fell at step 699 with v1 and timed out at step 999 with
    the skin). generation/dummy/check_skin_physics.py compares the assets in these terms.
    """
    import numpy as np

    product = np.asarray(local_in_scaled_frame, dtype=np.float32) * np.asarray(scale, dtype=np.float32)
    return tuple(float(x) for x in product)


def verify_engine_identity(args: argparse.Namespace, skinned: Path) -> list[str]:
    """Build the bare dummy of the same version (v1, or v2 with --jointed) from the same inputs into a
    scratch file and compare the two as the engine reads them (generation/dummy/check_skin_physics.py).
    Empty means identical."""
    import contextlib
    import io
    import tempfile

    from generation.dummy import check_skin_physics

    with tempfile.TemporaryDirectory() as tmp:
        stem = "v2" if getattr(args, "jointed", False) else "v1"
        bare = Path(tmp) / f"keti_dummy_{stem}.usd"
        with contextlib.redirect_stdout(io.StringIO()):
            build(argparse.Namespace(out=str(bare), geometry_only=args.geometry_only, root=args.root, skin=None,
                                     jointed=getattr(args, "jointed", False), gait=getattr(args, "gait", None)))
        return check_skin_physics.compare_assets(bare, skinned)


def _scale_op(prim) -> tuple[float, float, float]:
    """The prim's own xformOp:scale, or unit scale if it has none (spheres, capsules)."""
    from pxr import UsdGeom

    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpName() == "xformOp:scale":
            return tuple(float(v) for v in op.Get())
    return (1.0, 1.0, 1.0)


def _png_has_alpha(path: Path) -> bool:
    """IHDR 의 colour type: 4 (grey+alpha) 와 6 (RGBA) 만 알파를 갖는다. PIL 없이 읽는다."""
    with open(path, "rb") as fh:
        head = fh.read(26)
    return head[:8] == b"\x89PNG\r\n\x1a\n" and len(head) >= 26 and head[25] in (4, 6)


def _prim_token(name: str) -> str:
    """OBJ 의 재질·객체 이름("Human.body")을 USD 프림 이름으로."""
    import re
    tok = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return tok if tok and not tok[0].isdigit() else f"_{tok}"


def _material_tokens(names) -> dict[str, str]:
    """재질 이름 → 프림 이름. 서로 다른 이름이 같은 프림 이름으로 접히면 뒤의 재질이 앞의
    것을 덮어쓰고 서브셋도 충돌한다 (예: "Human.a-b" 와 "Human.a_b") —
    그런 입력은 거부한다."""
    tokens: dict[str, str] = {}
    owners: dict[str, str] = {}
    for name in names:
        tok = _prim_token(name)
        if tok in owners and owners[tok] != name:
            raise ValueError(f"재질 이름 {owners[tok]!r} 와 {name!r} 이 같은 프림 이름 {tok!r} 으로 접힙니다")
        owners[tok] = name
        tokens[name] = tok
    return tokens


def _sha256_of(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def skin_face_prims(mesh: "rs.ObjMesh", prims):
    """The body prim each face of the skin lands on: every vertex to its nearest primitive, every face by
    the majority of its three vertices (a three-way split by its centroid's nearest primitive). These are
    the labels attach_skin splits the skin by (region_split.split_faces groups the faces by the same
    face_labels call) and the vest takes its clothes by. Pure numpy."""
    import numpy as np

    names = np.array([p.name for p in prims])
    return rs.face_labels(mesh.F, names[rs.assign_primitives(mesh.V, prims)], V=mesh.V, prims=prims,
                          label_of=lambda p: p.name)


# OmniPBR's base::mono_mode enum, as Kit reads it from an int input.
_MONO_ALPHA = 0


def _skin_material(stage, token: str, mat: "rs.MtlMaterial", tex_dir: Path, metallic: float = 0.0):
    """MTL 재질 하나 → OmniPBR (MDL). 확산 맵(+알파 컷아웃)과 법선 맵을 옆의 textures/ 로 참조한다.
    ``metallic`` 은 조끼의 반사 띠가 쓴다 — 피부 재질은 기본값 0 이라 v1 입힌 인형의 바이트가 그대로다.

    씬 빌더와 같은 OmniPBR 이라 배포본의 재질이 한 종류다. 메시의 `primvars:st` 를 UV 세트 0
    으로 읽는다. **상대 경로는 스폰하는 쪽이 USD 를 절대 경로로 참조할 때만 풀린다** —
    `UsdFileCfg(usd_path=<상대 경로>)` 로 스폰하면 Kit 이 레이어를 상대 식별자로 열어
    `./textures/…` 가 "can not be found" 가 되고 전신이 회색(UsdPreviewSurface 면 검정)으로
    나온다 (2026-09-22 실측, 두 셰이더 다 같음). 환경 설정은 `REPO / …` 절대 경로를 쓴다.
    """
    from pxr import Gf, Sdf, UsdShade

    path = f"{LOOKS}/{token}"
    material = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(
        float(mat.pr if mat.pr is not None else 0.5))
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if mat.map_kd:
        diffuse = f"./{TEXTURE_SUBDIR}/{Path(mat.map_kd).name}"
        sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(diffuse)
        # hair, eyebrows and eyes are cards: their diffuse alpha cuts the card out
        if _png_has_alpha(tex_dir / Path(mat.map_kd).name):
            sh.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
            sh.CreateInput("enable_opacity_texture", Sdf.ValueTypeNames.Bool).Set(True)
            sh.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(diffuse)
            sh.CreateInput("opacity_mode", Sdf.ValueTypeNames.Int).Set(_MONO_ALPHA)
            sh.CreateInput("opacity_threshold", Sdf.ValueTypeNames.Float).Set(0.5)
    else:
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*[float(c) for c in mat.kd]))
    if mat.map_bump:
        sh.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
            f"./{TEXTURE_SUBDIR}/{Path(mat.map_bump).name}")
        sh.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(float(mat.bump_strength))
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(out)
    material.CreateDisplacementOutput("mdl").ConnectToSource(out)
    material.CreateVolumeOutput("mdl").ConnectToSource(out)
    return material


# ---- the work shirt and trousers (--split-worksuit) ---------------------------------------
# The worksuit is one MTL material on one texture: a white T-shirt under denim bib overalls. Its UV
# islands tell them apart -- the shirt is the island the texture paints white -- so the split needs no
# change to the skin export (shared with the v1 skinned dummy, whose pin it would break).
WORKSUIT_MATERIAL = "Human_male_worksuit01"            # the MTL material the split applies to (its prim token)
SHIRT_UV = (0.44, 0.80)                                # a UV point inside the white T-shirt's island
# SHIRT_MATERIAL / TROUSERS_MATERIAL (Looks/<name> in the built USD) are defined once in generation.appearance.
# mean relative luminance (linear light) of the texels an island covers: the shirt's at least this,
# every other worksuit island's at most the second
SHIRT_WHITE_MIN = 0.85
TROUSERS_WHITE_MAX = 0.6


def worksuit_islands(uv_faces) -> list:
    """The UV islands of a set of faces: (m, 3) UV indices in, one array of row indices per island out, each
    ascending and the islands ordered by their first row. Two faces are in one island when a chain of faces
    sharing UV indices links them; faces whose UVs only sit at the same coordinates under other indices are
    islands of their own. Pure numpy."""
    import numpy as np

    uv_faces = np.asarray(uv_faces, dtype=np.int64).reshape(-1, 3)
    if not len(uv_faces):
        return []
    ids, corner = np.unique(uv_faces, return_inverse=True)
    corner = corner.reshape(-1, 3)
    parent = list(range(len(ids)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in corner.tolist():
        ra = root(a)
        for other in (b, c):
            ro = root(other)
            if ro != ra:
                parent[ro] = ra
    islands: dict[int, list[int]] = {}
    for row, a in enumerate(corner[:, 0].tolist()):
        islands.setdefault(root(a), []).append(row)
    return sorted((np.asarray(rows, dtype=np.int64) for rows in islands.values()), key=lambda rows: rows[0])


def _texture_rgb(path: Path):
    """An image as (h, w, 3) sRGB values in [0, 1], row 0 at the image's top. PIL, as the Isaac python ships it."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def _relative_luminance(rgb):
    """Relative luminance of sRGB values: decoded to linear light, then the Rec. 709 weights."""
    import numpy as np

    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear @ np.array([0.2126, 0.7152, 0.0722])


def _covered_texels(triangles, shape):
    """The texels of an image of ``shape`` (h, w) that UV triangles (n, 3, 2) cover: those whose centre lies in a
    triangle, and the one under each triangle's centroid so a triangle smaller than a texel still counts. v runs up
    the image: v = 1 is row 0."""
    import numpy as np

    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    xs = triangles[..., 0] * w
    ys = (1.0 - triangles[..., 1]) * h
    for x, y in zip(xs, ys):
        x0, x1 = max(int(np.floor(x.min())), 0), min(int(np.ceil(x.max())), w - 1)
        y0, y1 = max(int(np.floor(y.min())), 0), min(int(np.ceil(y.max())), h - 1)
        if x0 <= x1 and y0 <= y1:
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            e = [(x[(k + 1) % 3] - x[k]) * (gy - y[k]) - (y[(k + 1) % 3] - y[k]) * (gx - x[k]) for k in range(3)]
            inside = ((e[0] >= 0) & (e[1] >= 0) & (e[2] >= 0)) | ((e[0] <= 0) & (e[1] <= 0) & (e[2] <= 0))
            mask[y0:y1 + 1, x0:x1 + 1] |= inside
        mask[min(max(int(y.mean()), 0), h - 1), min(max(int(x.mean()), 0), w - 1)] = True
    return mask


def _holds(triangles, point):
    """Per UV triangle (n, 3, 2): whether it holds ``point``, its edges included."""
    import numpy as np

    p = np.asarray(point, dtype=np.float64)
    e = [(triangles[:, (k + 1) % 3, 0] - triangles[:, k, 0]) * (p[1] - triangles[:, k, 1])
         - (triangles[:, (k + 1) % 3, 1] - triangles[:, k, 1]) * (p[0] - triangles[:, k, 0]) for k in range(3)]
    return ((e[0] >= 0) & (e[1] >= 0) & (e[2] >= 0)) | ((e[0] <= 0) & (e[1] <= 0) & (e[2] <= 0))


def shirt_faces(mesh: "rs.ObjMesh", texture_path: Path):
    """The worksuit faces of the white T-shirt, as ascending indices into the mesh's faces: the UV island
    (worksuit_islands) whose triangles hold SHIRT_UV. ValueError unless the mesh has the worksuit material with UVs
    in [0, 1] on every face, exactly one island holds the point, the texels that island covers on ``texture_path``
    average a relative luminance of at least SHIRT_WHITE_MIN and every other worksuit island's at most
    TROUSERS_WHITE_MAX -- the split follows what the texture paints white, so another export, another texture or
    a v axis read the wrong way up is refused instead of colouring the wrong cloth."""
    import numpy as np

    suit = [i for i, name in enumerate(mesh.materials) if _prim_token(name) == WORKSUIT_MATERIAL]
    if len(suit) != 1:
        raise ValueError(f"메시에 작업복 재질({WORKSUIT_MATERIAL})이 하나가 아니라 {len(suit)}개입니다")
    faces = np.nonzero(mesh.FM == suit[0])[0]
    if not len(faces):
        raise ValueError(f"작업복 재질({WORKSUIT_MATERIAL})의 면이 없습니다")
    uv_faces = mesh.FT[faces]
    if (uv_faces < 0).any():
        raise ValueError("UV 가 없는 작업복 면이 있습니다")
    triangles = mesh.VT[uv_faces]
    if not np.isfinite(triangles).all() or (triangles < 0).any() or (triangles > 1).any():
        raise ValueError("작업복 면의 UV 가 [0, 1] 밖에 있습니다")
    islands = worksuit_islands(uv_faces)
    holding = [k for k, rows in enumerate(islands) if _holds(triangles[rows], SHIRT_UV).any()]
    if len(holding) != 1:
        raise ValueError(f"SHIRT_UV {SHIRT_UV} 를 담은 작업복 섬이 하나가 아니라 {len(holding)}개입니다")
    luminance = _relative_luminance(_texture_rgb(texture_path))
    means = [float(luminance[_covered_texels(triangles[rows], luminance.shape)].mean()) for rows in islands]
    shirt = holding[0]
    if means[shirt] < SHIRT_WHITE_MIN:
        raise ValueError(f"SHIRT_UV {SHIRT_UV} 를 담은 섬(면 {len(islands[shirt])}개)의 평균 상대 휘도 "
                         f"{means[shirt]:.3f} < {SHIRT_WHITE_MIN} — 희지 않습니다")
    white = [(len(islands[k]), round(means[k], 3)) for k in range(len(islands))
             if k != shirt and means[k] > TROUSERS_WHITE_MAX]
    if white:
        raise ValueError(f"상의 밖의 작업복 섬 {len(white)}개(면 수 · 평균 상대 휘도 {white})가 "
                         f"{TROUSERS_WHITE_MAX} 보다 밝습니다 — 흰 섬이 상의 하나가 아닙니다")
    return faces[islands[shirt]]


def attach_skin(stage, skin_obj: Path, anthro: dict, out: Path, extra_prims=(), seams=None,
                split_worksuit: bool = False) -> dict:
    """부위 프림마다 시각 메시의 서브메시를 콜라이더 없는 자식으로 붙인다.

    ``seams`` 는 --seams 가 주는 (SeamSpec, body_links, 명세 경로)다. 주면 엉뚱한 몸에 든 조각을 둘레의 몸으로
    옮겨 가르고(skin_seams.clean_labels), 관절 이음새 근처의 정점마다 섞는 비율을 그 피부 메시의 속성 셋에 싣는다
    (skin_seams.ATTR_*). 조끼가 덮을 옷은 옮기기 전의 라벨로 고른다(``face_prims``) — 조끼는 그대로다.

    ``split_worksuit`` 면 작업복 재질의 면을 흰 티셔츠의 UV 섬(shirt_faces)과 나머지로 갈라 두 재질 SHIRT_MATERIAL ·
    TROUSERS_MATERIAL 에 묶는다. 두 재질은 작업복 재질과 입력이 같아(같은 텍스처 · 법선 지도 · 거칠기 · 금속성) 지은
    겉모습은 그대로이고, 상의와 바지의 색을 따로 바꿀 자리만 생긴다. 메시의 점 · 면 · 법선 · UV · 라벨은 그대로다.

    정점은 가장 가까운 프리미티브(anthropometry.yaml 의 형상)로, 면은 세 정점 다수결로 가른다
    (region_split). 자식은 부모 강체의 로컬 프레임에 쓰므로 물리가 부위를 옮기면 함께 움직이고,
    물리 API 는 하나도 붙이지 않아 접촉·질량·조인트가 v1 과 같다. 프리미티브는 purpose=guide 로
    숨기고 자식은 default 를 명시한다 — 자식의 명시된 purpose 가 부모의 guide 를 이긴다
    (pxr 실측 2026-09-22). 시맨틱 라벨 class=BR-xx 는 자식 메시에 굽는다 — 스폰 뒤에 붙이면
    PhysX 텐서 뷰가 무효화된다 (실측).
    """
    import hashlib
    import shutil

    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdSemantics, UsdShade, Vt

    if not skin_obj.is_file():
        raise SystemExit(f"시각 메시가 없습니다: {skin_obj}\n"
                         "  .venv/bin/python generation/dummy/skin/make_skin.py")
    mesh = rs.read_obj(skin_obj)
    if mesh.mtllib is None:
        raise SystemExit("OBJ 에 mtllib 가 없습니다 — 재질 없이 온 메시입니다")
    mtl_path = skin_obj.parent / mesh.mtllib
    if not mtl_path.is_file():
        raise SystemExit(f"OBJ 가 가리키는 MTL 이 없습니다: {mtl_path}")
    mtl = rs.read_mtl(mtl_path)
    try:
        mesh.validate()
    except ValueError as e:
        raise SystemExit(f"시각 메시가 깨져 있습니다 ({skin_obj.name}): {e}")
    for what, arr in (("재질", mesh.FM), ("UV", mesh.FT), ("법선", mesh.FN)):
        if len(arr) == 0 or (arr < 0).any():
            raise SystemExit(f"{what} 없는 면이 있습니다 — make_skin.py 는 전부 내보냅니다")
    unknown = sorted(set(mesh.materials) - set(mtl))
    if unknown:
        raise SystemExit(f"OBJ 가 쓰는 재질이 MTL 에 없습니다: {unknown}")
    try:
        tokens = _material_tokens(mtl)
    except ValueError as e:
        raise SystemExit(str(e))

    # The OBJ has to be the one make_skin.py finished and checked — a run that failed
    # its fit, frame or lock check must not leave a consumable export behind (it cleans
    # up), and a stale export must not pass as current. The
    # manifest pins the files' hashes and the anthropometry the mesh was fitted to;
    # the tracked textures.lock pins the textures independently of the manifest.
    manifest_path = skin_obj.parent / "skin_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{manifest_path.name} 이 없습니다 — make_skin.py 가 끝까지 통과한 산출물이 아닙니다")
    info = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = info.get("files") or {}
    for key, path in (("obj", skin_obj), ("mtl", mtl_path)):
        if (files.get(key) or {}).get("sha256") != _sha256_of(path):
            raise SystemExit(f"{path.name} 이 매니페스트와 다릅니다 — make_skin.py 를 다시 돌리십시오")
    if (info.get("anthropometry") or {}).get("sha256") != _sha256_of(ANTHRO):
        raise SystemExit("시각 메시가 맞춘 anthropometry.yaml 이 지금 것과 다릅니다 — make_skin.py 를 다시 돌리십시오")
    from generation.dummy.skin import make_skin
    # the spec too: a pose, macro or fit change must invalidate an export made before it
    if (info.get("spec") or {}).get("sha256") != _sha256_of(make_skin.SPEC):
        raise SystemExit(f"시각 메시를 만든 명세가 지금의 {make_skin.SPEC.relative_to(REPO_ROOT)} 와 다릅니다 — "
                         "make_skin.py 를 다시 돌리십시오")
    all_rows = make_skin.parse_lock(make_skin.HERE / "textures.lock")
    lock_rows = {r["name"]: r for r in all_rows if make_skin.is_texture_row(r)}
    lock_pack = {r["name"]: r for r in all_rows if not make_skin.is_texture_row(r)}
    for what, lock_side, manifest_side in (("텍스처", lock_rows, files.get("textures") or []),
                                           ("팩 파일", lock_pack, files.get("pack_files") or [])):
        manifest_rows = {t["name"]: t for t in manifest_side}
        if set(lock_side) != set(manifest_rows) or any(
                lock_side[n]["sha256"] != manifest_rows[n].get("sha256") for n in lock_side):
            raise SystemExit(f"매니페스트의 {what} 목록이 textures.lock 과 다릅니다 — make_skin.py 를 다시 돌리십시오")

    prims = rs.primitives_from_anthropometry(anthro) + list(extra_prims)
    face_prims = skin_face_prims(mesh, prims)
    split, moves, weights = face_prims, [], {}
    if seams is not None:
        from generation.dummy import skin_seams

        seam_spec, links, seam_path = seams
        roots = skin_seams.rigid_roots(links)
        strangers = sorted(set(face_prims.tolist()) - set(roots))
        if strangers:
            raise SystemExit(f"피부가 든 몸이 관절 인형의 몸이 아닙니다: {strangers}")
        split, moves = skin_seams.clean_labels(mesh.F, face_prims, roots, seam_spec.stray_share)
        weights = skin_seams.blend_weights(
            mesh.V, mesh.F, split, roots,
            lambda a, b: skin_seams.pair_radius(a, b, links, seam_spec.radius_m))
    parts = rs.split_by_face(mesh.F, split)
    absent = [p.name for p in prims if p.name not in parts]
    if absent:
        raise SystemExit(f"시각 메시가 닿지 않는 부위 프림: {absent} — 세그멘테이션에 구멍이 납니다")
    code_of = {p.name: p.code for p in prims}

    # textures beside the USD, referenced relatively: build/ ships as a whole (like the scene)
    tex_dir = out.parent / TEXTURE_SUBDIR
    tex_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for m in mtl.values():
        for ref in (m.map_kd, m.map_bump):
            if not ref:
                continue
            src = skin_obj.parent / ref
            if not src.is_file():
                raise SystemExit(f"MTL 이 가리키는 텍스처가 없습니다: {src}")
            if src.name in copied and copied[src.name] != src:
                raise SystemExit(f"이름이 같은 텍스처가 둘입니다: {copied[src.name]} / {src}")
            if src.name not in copied:
                row = lock_rows.get(src.name)
                if row is None:
                    raise SystemExit(f"MTL 이 쓰는 텍스처 {src.name} 이 textures.lock 에 없습니다")
                if _sha256_of(src) != row["sha256"]:
                    raise SystemExit(f"텍스처 {src.name} 이 textures.lock 의 해시와 다릅니다")
                shutil.copyfile(src, tex_dir / src.name)
                copied[src.name] = src
    shirt = None
    if split_worksuit:
        suit = [name for name in mtl if tokens[name] == WORKSUIT_MATERIAL]
        if len(suit) != 1 or not mtl[suit[0]].map_kd:
            raise SystemExit(f"작업복을 상의 · 바지로 가를 수 없습니다: MTL 에 확산 텍스처가 있는 {WORKSUIT_MATERIAL} 이 없습니다")
        if {SHIRT_MATERIAL, TROUSERS_MATERIAL} & set(tokens.values()):
            raise SystemExit(f"MTL 재질이 상의 · 바지 재질의 이름({SHIRT_MATERIAL} · {TROUSERS_MATERIAL})을 씁니다")
        try:
            shirt = np.zeros(len(mesh.F), dtype=bool)
            shirt[shirt_faces(mesh, skin_obj.parent / mtl[suit[0]].map_kd)] = True
        except ValueError as e:
            raise SystemExit(f"작업복을 상의 · 바지로 가를 수 없습니다: {e}")
    UsdGeom.Scope.Define(stage, LOOKS)
    # one material per MTL material, keyed by its prim token; with the split the worksuit's inputs go to two
    looks = {}
    for name, m in mtl.items():
        for token in ((SHIRT_MATERIAL, TROUSERS_MATERIAL) if shirt is not None and tokens[name] == WORKSUIT_MATERIAL
                      else (tokens[name],)):
            looks[token] = _skin_material(stage, token, m, tex_dir)

    stats: dict[str, dict] = {}
    for name, (faces, old, fid) in parts.items():
        body = stage.GetPrimAtPath(f"/KetiDummy/{name}")
        if not body:
            raise SystemExit(f"부위 프림이 없습니다: /KetiDummy/{name}")
        # Into the body's local frame. Box bodies carry a scale op, so the child's points
        # must be divided by it (and its normals multiplied: n_local = n_world · Lᵀ for
        # row vectors) or the renderer applies the scale a second time.
        xf = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        inv = xf.GetInverse()
        L = np.array([[xf[i][j] for j in range(3)] for i in range(3)], float)
        pts = [inv.Transform(Gf.Vec3d(*[float(c) for c in p])) for p in mesh.V[old]]
        n_world = mesh.VN[mesh.FN[fid]].reshape(-1, 3)
        n_local = n_world @ L.T
        n_local /= np.maximum(np.linalg.norm(n_local, axis=1, keepdims=True), 1e-12)
        st = mesh.VT[mesh.FT[fid]].reshape(-1, 2)

        m = UsdGeom.Mesh.Define(stage, f"/KetiDummy/{name}/{SKIN_CHILD}")
        points = Vt.Vec3fArray([Gf.Vec3f(*[float(c) for c in p]) for p in pts])
        m.CreatePointsAttr(points)
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in faces.ravel()]))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
        m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(c) for c in n]) for n in n_local]))
        m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                  UsdGeom.Tokens.faceVarying)
        pv.Set(Vt.Vec2fArray([Gf.Vec2f(*[float(c) for c in uv]) for uv in st]))
        UsdGeom.Imageable(m.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.default_)
        UsdSemantics.LabelsAPI.Apply(m.GetPrim(), "class").CreateLabelsAttr([code_of[name]])
        # Not keti:bodyRegion: the recorder and the checks collect bodies by that attribute
        # and this child is not a body. The label above carries the code for segmentation.
        m.GetPrim().CreateAttribute("keti:visualOnly", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
        blended = 0
        if seams is not None:
            rows, cols, M = skin_seams.piece_blend(old, weights)
            if len(rows):
                prim = m.GetPrim()
                prim.CreateAttribute(skin_seams.ATTR_POINTS, Sdf.ValueTypeNames.IntArray, custom=True).Set(
                    Vt.IntArray([int(i) for i in rows]))
                prim.CreateAttribute(skin_seams.ATTR_BODIES, Sdf.ValueTypeNames.TokenArray, custom=True).Set(
                    Vt.TokenArray(list(cols)))
                prim.CreateAttribute(skin_seams.ATTR_WEIGHTS, Sdf.ValueTypeNames.FloatArray, custom=True).Set(
                    Vt.FloatArray([float(w) for w in M.ravel()]))
                blended = int(len(rows))
        if not stage.GetPrimAtPath(f"/KetiDummy/{name}/{COLLIDER_CHILD}"):
            raise SystemExit(f"/KetiDummy/{name}: 콜라이더 자식이 없습니다 — split_body 없이 attach_skin 을 불렀습니다")

        # the piece's faces by the material they are bound to, in the MTL materials' order (the split puts the
        # shirt's before the trousers'): one material binds the mesh, more bind a materialBind subset each
        fm = mesh.FM[fid]
        groups = []
        for mi in (int(i) for i in np.unique(fm)):
            sel = np.nonzero(fm == mi)[0]
            token = tokens[mesh.materials[mi]]
            if shirt is not None and token == WORKSUIT_MATERIAL:
                on_shirt = shirt[fid[sel]]
                groups += [(t, s) for t, s in ((SHIRT_MATERIAL, sel[on_shirt]), (TROUSERS_MATERIAL, sel[~on_shirt]))
                           if len(s)]
            else:
                groups.append((token, sel))
        if len(groups) == 1:
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(looks[groups[0][0]])
        else:
            UsdGeom.Subset.SetFamilyType(m, "materialBind", UsdGeom.Tokens.partition)
            for token, sel in groups:
                sub = UsdGeom.Subset.CreateGeomSubset(m, token, UsdGeom.Tokens.face,
                                                      Vt.IntArray([int(i) for i in sel]), "materialBind")
                UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(looks[token])
        stats[name] = {"vertices": int(len(old)), "faces": int(len(faces)), "materials": len(groups), "blended": blended}

    # what the visual came from, on the asset itself
    manifest = skin_obj.parent / "skin_manifest.json"
    provenance = {"source": "generation/dummy/skin/make_skin.py",
                  "obj": skin_obj.name, "objSha256": hashlib.sha256(skin_obj.read_bytes()).hexdigest(),
                  "objects": list(mesh.objects), "textures": sorted(copied)}
    if manifest.is_file():
        info = json.loads(manifest.read_text(encoding="utf-8"))
        provenance["skinVersion"] = str(info.get("skin_version"))
        provenance["generatedWith"] = {k: str(v) for k, v in (info.get("generated_with") or {}).items()}
        provenance["statureM"] = float((info.get("stature") or {}).get("dressed_body_m", float("nan")))
    root = stage.GetPrimAtPath("/KetiDummy")
    root.SetCustomDataByKey("keti:skin", provenance)
    if seams is not None:
        root.SetCustomDataByKey(skin_seams.CUSTOM_DATA_KEY, {
            "spec": seam_path.name, "specSha256": _sha256_of(seam_path), "strayShare": float(seam_spec.stray_share),
            "radiusM": {g: float(r) for g, r in seam_spec.radius_m.items()},
            "movedPatches": len(moves), "movedFaces": int(sum(mv["faces"] for mv in moves)),
            "blendedPoints": int(sum(st["blended"] for st in stats.values()))})
    worksuit = None
    if shirt is not None:
        suit_faces = int((mesh.FM == mesh.materials.index(suit[0])).sum())
        worksuit = {SHIRT_MATERIAL: int(shirt.sum()), TROUSERS_MATERIAL: suit_faces - int(shirt.sum())}
    return {"parts": stats, "materials": len(looks), "textures": sorted(copied), "worksuit": worksuit,
            "faces": int(len(mesh.F)), "vertices": int(len(mesh.V)), "moves": moves,
            # what --vest makes the vest over: the mesh, each face's body prim, the primitives it split by
            "mesh": mesh, "face_prims": face_prims, "prims": prims}


def attach_vest(stage, pieces, spec, *, spec_path: Path) -> dict:
    """조끼 조각(generation/dummy/vest.py)마다 그 부위 몸 아래에 보이기만 하는 자식 메시 vest 를 붙인다.

    조각은 인형 좌표로 오고 피부처럼 몸의 로컬 프레임으로 옮겨 쓴다 — 물리가 몸을 옮기면 함께 움직이고,
    물리 API 는 하나도 붙이지 않아 엔진이 읽는 몸은 같은 판의 맨몸과 같다(빌더가 그 대조를 돌린다).
    분할 라벨 class 는 그 몸의 부위 코드(keti:bodyRegion)라 조끼의 화소는 그 부위의 사람 화소로 셈된다.
    재질은 텍스처 없는 OmniPBR 둘이고 피부 재질과 같은 함수로 짓는다 — 명세의 sRGB 색을 MTL 의 Kd 처럼
    diffuse_color_constant 에 그대로 쓴다. 재질이 둘인 조각은 면 서브셋으로 묶는다.
    """
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdSemantics, UsdShade, Vt

    from generation.dummy import vest as vest_mod

    UsdGeom.Scope.Define(stage, LOOKS)
    materials = {}
    for name in vest_mod.MATERIALS:
        if stage.GetPrimAtPath(f"{LOOKS}/{name}"):
            raise SystemExit(f"조끼 재질의 자리에 이미 프림이 있습니다: {LOOKS}/{name}")
        m = spec.materials[name]
        materials[name] = _skin_material(
            stage, name, rs.MtlMaterial(name, kd=tuple(float(c) for c in m["color"]), pr=float(m["roughness"])),
            Path("."), metallic=float(m["metallic"]))

    stats: dict[str, dict] = {}
    for name, piece in pieces.items():
        body = stage.GetPrimAtPath(f"/KetiDummy/{name}")
        region = body.GetAttribute("keti:bodyRegion") if body else None
        if not body or not region or not region.Get():
            raise SystemExit(f"조끼 조각을 붙일 부위 몸이 없습니다: /KetiDummy/{name}")
        if not stage.GetPrimAtPath(f"/KetiDummy/{name}/{SKIN_CHILD}"):
            raise SystemExit(f"/KetiDummy/{name}: 피부 자식이 없습니다 — attach_skin 없이 attach_vest 를 불렀습니다")
        # into the body's local frame, as attach_skin does for the skin piece beside it
        xf = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        inv = xf.GetInverse()
        L = np.array([[xf[i][j] for j in range(3)] for i in range(3)], float)
        pts = [inv.Transform(Gf.Vec3d(*[float(c) for c in p])) for p in piece.V]
        n_local = piece.N @ L.T
        n_local /= np.maximum(np.linalg.norm(n_local, axis=1, keepdims=True), 1e-12)

        m = UsdGeom.Mesh.Define(stage, f"/KetiDummy/{name}/{VEST_CHILD}")
        points = Vt.Vec3fArray([Gf.Vec3f(*[float(c) for c in p]) for p in pts])
        m.CreatePointsAttr(points)
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(piece.F)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in piece.F.ravel()]))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
        # one normal per face corner, like the skin: the vest's normal at each corner
        m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(c) for c in n]) for n in n_local[piece.F].reshape(-1, 3)]))
        m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        UsdGeom.Imageable(m.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.default_)
        UsdSemantics.LabelsAPI.Apply(m.GetPrim(), "class").CreateLabelsAttr([str(region.Get())])
        m.GetPrim().CreateAttribute("keti:visualOnly", Sdf.ValueTypeNames.Bool, custom=True).Set(True)

        here = [mat for mat in vest_mod.MATERIALS if (piece.material == mat).any()]
        if len(here) == 1:
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(materials[here[0]])
        else:
            UsdGeom.Subset.SetFamilyType(m, "materialBind", UsdGeom.Tokens.partition)
            for mat in here:
                sel = Vt.IntArray([int(i) for i in np.nonzero(piece.material == mat)[0]])
                sub = UsdGeom.Subset.CreateGeomSubset(m, mat, UsdGeom.Tokens.face, sel, "materialBind")
                UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(materials[mat])
        stats[name] = {"vertices": int(len(piece.V)), "faces": int(len(piece.F)),
                       "band_faces": int((piece.material == vest_mod.BAND).sum())}

    stage.GetPrimAtPath("/KetiDummy").SetCustomDataByKey("keti:vest", {
        "spec": spec_path.name, "specSha256": _sha256_of(spec_path), "sourceObject": spec.source_object,
        "regions": list(spec.regions), "pieces": sorted(stats)})
    return {"parts": stats}


def _foot_class() -> str:
    """The segmentation class a foot's skin carries: a person's pixels, not a contact region."""
    from generation.dummy import joints_spec as js
    return js.load().semantic_class_feet


def _visual_mesh_problems(mesh_prim, code: str, *, needs_uv: bool) -> tuple[list[str], int, set[str]]:
    """One visual child (skin or vest) as attach_skin / attach_vest promise it: rendered, no physics, the
    body's class, a sound triangle mesh with a normal per corner, an extent and a bound material on every
    face. Returns the problems, the face count and the bound materials' paths."""
    from pxr import UsdGeom, UsdPhysics, UsdSemantics, UsdShade

    problems: list[str] = []
    bound: set[str] = set()
    where = str(mesh_prim.GetPath())
    if UsdGeom.Imageable(mesh_prim).ComputePurpose() != UsdGeom.Tokens.default_:
        problems.append(f"{where}: purpose 가 default 가 아니라 렌더되지 않습니다")
    if mesh_prim.HasAPI(UsdPhysics.CollisionAPI) or mesh_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        problems.append(f"{where}: 시각 메시에 물리 API 가 붙었습니다")
    if any(s.startswith(("Physics", "Physx")) for s in mesh_prim.GetAppliedSchemas()):
        problems.append(f"{where}: 시각 메시에 물리 스키마가 붙었습니다")
    if any(a.GetName().startswith("physics:") for a in mesh_prim.GetAttributes()):
        problems.append(f"{where}: physics: 속성이 있습니다")
    labels = UsdSemantics.LabelsAPI(mesh_prim, "class")
    got = list(labels.GetLabelsAttr().Get() or []) if labels else []
    if got != [code]:
        problems.append(f"{where}: 시맨틱 라벨이 {code} 가 아니라 {got} 입니다")
    # topology: counting faces is not enough — empty points with out-of-range
    # indices counted the same
    mesh_api = UsdGeom.Mesh(mesh_prim)
    pts = mesh_api.GetPointsAttr().Get() or []
    counts = list(mesh_api.GetFaceVertexCountsAttr().Get() or [])
    idx = list(mesh_api.GetFaceVertexIndicesAttr().Get() or [])
    corners = sum(counts)
    if not pts:
        problems.append(f"{where}: 점이 없습니다")
    elif not all(math.isfinite(c) for p in pts for c in p):
        problems.append(f"{where}: 비유한 좌표가 있습니다")
    if not counts or any(c != 3 for c in counts):
        problems.append(f"{where}: 삼각형이 아닌 면이 있습니다")
    if len(idx) != corners or (idx and (min(idx) < 0 or max(idx) >= len(pts))):
        problems.append(f"{where}: 면 인덱스가 점의 범위 밖입니다")
    normals = mesh_api.GetNormalsAttr().Get() or []
    if len(normals) != corners or mesh_api.GetNormalsInterpolation() != UsdGeom.Tokens.faceVarying:
        problems.append(f"{where}: 법선이 면 모서리마다 하나가 아닙니다")
    elif not all(math.isfinite(c) for n in normals for c in n):
        problems.append(f"{where}: 비유한 법선이 있습니다")
    if needs_uv:
        # the flattened primvar is what a renderer reads: an indexed primvar with bad
        # indices has the right raw length and no usable values
        st = UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("st")
        flat = st.ComputeFlattened() if st else None
        if not st or st.GetInterpolation() != UsdGeom.Tokens.faceVarying or flat is None or len(flat) != corners:
            problems.append(f"{where}: primvars:st 가 면 모서리마다 하나로 풀리지 않습니다")
        elif not all(math.isfinite(c) for uv in flat for c in uv):
            problems.append(f"{where}: 비유한 UV 가 있습니다")
    if not mesh_api.GetExtentAttr().HasAuthoredValue():
        problems.append(f"{where}: extent 가 없습니다")
    # bindings: a subset is not a binding — every subset must bind a material and
    # the subsets must cover every face exactly once
    subsets = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(mesh_prim))
    if subsets:
        covered: list[int] = []
        for sub in subsets:
            mat = UsdShade.MaterialBindingAPI(sub.GetPrim()).ComputeBoundMaterial()[0]
            if not mat:
                problems.append(f"{sub.GetPath()}: 재질이 묶여 있지 않습니다")
            else:
                bound.add(str(mat.GetPath()))
            if sub.GetFamilyNameAttr().Get() != "materialBind":
                problems.append(f"{sub.GetPath()}: materialBind 패밀리가 아닙니다")
            if sub.GetElementTypeAttr().Get() != UsdGeom.Tokens.face:
                problems.append(f"{sub.GetPath()}: 면 서브셋이 아닙니다 (elementType)")
            covered.extend(int(i) for i in (sub.GetIndicesAttr().Get() or []))
        if sorted(covered) != list(range(len(counts))):
            problems.append(f"{where}: 서브셋이 면을 정확히 한 번씩 덮지 않습니다")
    else:
        mat = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()[0]
        if not mat:
            problems.append(f"{where}: 재질이 없습니다")
        else:
            bound.add(str(mat.GetPath()))
    return problems, len(counts), bound


def _faces_by_material(mesh_prim) -> dict[str, int]:
    """{bound material path: number of the mesh's faces bound to it}, through its materialBind subsets or, without
    subsets, its own binding. A face or subset without a material counts for nothing (_visual_mesh_problems names
    it)."""
    from pxr import UsdGeom, UsdShade

    out: dict[str, int] = {}
    subsets = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(mesh_prim))
    holders = [(s.GetPrim(), len(s.GetIndicesAttr().Get() or [])) for s in subsets] or \
        [(mesh_prim, len(UsdGeom.Mesh(mesh_prim).GetFaceVertexCountsAttr().Get() or []))]
    for holder, n in holders:
        material = UsdShade.MaterialBindingAPI(holder).ComputeBoundMaterial()[0]
        if material:
            out[str(material.GetPath())] = out.get(str(material.GetPath()), 0) + n
    return out


def _seam_problems(mesh_prim, bodies: set[str]) -> tuple[list[str], int]:
    """One skin mesh's seam blend (skin_seams.ATTR_*), as build_dummy writes and skin_blend reads it: none of the
    three, or all three with distinct in-range point indices, columns that are bodies of the dummy, and a row of
    finite weights in [0, 1] summing to 1 for each point, at least two of them above 0. Returns (problems, 1 if the
    mesh carries a blend else 0)."""
    import numpy as np

    from generation.dummy import skin_seams

    where = str(mesh_prim.GetPath())
    attrs = [mesh_prim.GetAttribute(a) for a in (skin_seams.ATTR_POINTS, skin_seams.ATTR_BODIES, skin_seams.ATTR_WEIGHTS)]
    held = [bool(a) and a.HasAuthoredValue() for a in attrs]
    if not any(held):
        return [], 0
    if not all(held):
        return [f"{where}: 섞는 비율의 속성 셋 가운데 일부만 있습니다"], 1
    points, cols, weights = (list(a.Get()) for a in attrs)
    n = len(mesh_prim.GetAttribute("points").Get() or [])
    problems = []
    if not points or len(set(points)) != len(points) or not all(0 <= int(i) < n for i in points):
        problems.append(f"{where}: 섞는 정점의 번호가 비었거나 겹치거나 점의 범위 밖입니다")
    if not cols or len(set(cols)) != len(cols) or set(cols) - bodies:
        problems.append(f"{where}: 섞는 비율의 열이 인형의 몸이 아닙니다 ({list(cols)})")
    if len(weights) != len(points) * len(cols):
        problems.append(f"{where}: 비율 {len(weights)} 개 ≠ 정점 {len(points)} × 열 {len(cols)}")
        return problems, 1
    M = np.asarray(weights, float).reshape(len(points), max(len(cols), 1))
    if not np.isfinite(M).all() or (M < 0).any() or (M > 1).any():
        problems.append(f"{where}: 비율이 0 과 1 사이의 유한한 수가 아닙니다")
    elif not np.allclose(M.sum(axis=1), 1.0, atol=1e-5) or ((M > 0).sum(axis=1) < 2).any():
        problems.append(f"{where}: 비율의 합이 1 이 아니거나 둘 이상의 열에 걸치지 않은 정점이 있습니다")
    return problems, 1


def verify_skin_usd(out: Path, expected_faces: int, vest: dict[str, int] | None = None,
                    seams: bool = False, worksuit: dict[str, int] | None = None) -> list[str]:
    """산출물을 다시 열어 attach_skin 이 약속한 형태인지 본다 — 소스가 아니라 파일을.

    ``vest`` 는 --vest 가 붙인 조각의 {부위 몸: 면 수}다. 주면 그 몸마다 vest 자식이 같은 면 수로 있고
    피부와 같은 꼴(보이기만 · 그 몸의 라벨 · 물리 없음)인지 보고, 그 밖의 몸에 vest 자식이 있으면 문제로 센다.
    주지 않으면 어느 몸에도 vest 자식이 없어야 한다. ``seams`` 면 섞는 비율을 실은 피부가 하나 이상이고 그 셋이
    맞는 꼴인지(_seam_problems), 아니면 어느 피부에도 그 속성이 없어야 한다. ``worksuit`` 는 --split-worksuit 가 가른
    {상의 · 바지 재질: 면 수}다. 주면 피부의 면 가운데 그 재질에 묶인 수가 그대로이고 작업복 재질에 묶인 면이 없어야
    하고, 주지 않으면 어느 피부의 면도 상의 · 바지 재질에 묶여 있지 않아야 한다."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    problems: list[str] = []
    stage = Usd.Stage.Open(str(out))
    bodies = [p for p in stage.Traverse()
              if (p.GetAttribute("keti:bodyRegion") and p.GetAttribute("keti:bodyRegion").Get())
              or (p.GetAttribute("keti:groundContact") and p.GetAttribute("keti:groundContact").Get())]
    if not bodies:
        return ["부위 프림이 없습니다"]
    faces = 0
    blended_pieces = 0
    bound_materials: set[str] = set()
    skin_faces: dict[str, int] = {}
    vested: set[str] = set()
    for body in bodies:
        attr = body.GetAttribute("keti:bodyRegion")
        code = attr.Get() if attr and attr.Get() else _foot_class()
        if not body.HasAPI(UsdPhysics.RigidBodyAPI) or body.HasAPI(UsdPhysics.CollisionAPI):
            problems.append(f"{body.GetPath()}: 강체이되 형상은 아니어야 합니다 (콜라이더는 자식)")
        colliders = [c for c in body.GetChildren() if c.HasAPI(UsdPhysics.CollisionAPI)]
        if len(colliders) != 1 or colliders[0].GetName() != COLLIDER_CHILD:
            problems.append(f"{body.GetPath()}: 콜라이더 자식이 정확히 하나({COLLIDER_CHILD})여야 합니다")
        elif UsdGeom.Imageable(colliders[0]).ComputePurpose() != UsdGeom.Tokens.guide:
            problems.append(f"{colliders[0].GetPath()}: 프리미티브가 숨겨지지 않았습니다")
        skin = stage.GetPrimAtPath(f"{body.GetPath()}/{SKIN_CHILD}")
        if not skin or not skin.IsA(UsdGeom.Mesh):
            problems.append(f"{body.GetPath()}: skin 자식 메시가 없습니다")
            continue
        found, n, bound = _visual_mesh_problems(skin, code, needs_uv=True)
        problems += found
        faces += n
        bound_materials |= bound
        for path, count in _faces_by_material(skin).items():
            skin_faces[path] = skin_faces.get(path, 0) + count
        found, carries = _seam_problems(skin, {b.GetName() for b in bodies})
        problems += found
        blended_pieces += carries
        piece = stage.GetPrimAtPath(f"{body.GetPath()}/{VEST_CHILD}")
        want = (vest or {}).get(body.GetName())
        if want is None:
            if piece:
                problems.append(f"{piece.GetPath()}: 조끼 조각이 없어야 할 몸에 vest 자식이 있습니다")
        elif not piece or not piece.IsA(UsdGeom.Mesh):
            problems.append(f"{body.GetPath()}: vest 자식 메시가 없습니다")
        else:
            found, n, bound = _visual_mesh_problems(piece, code, needs_uv=False)
            problems += found
            bound_materials |= bound
            if n != want:
                problems.append(f"{piece.GetPath()}: 면 {n} ≠ 조끼 조각의 면 {want}")
            vested.add(body.GetName())
    if faces != expected_faces:
        problems.append(f"자식 메시의 면 합 {faces} ≠ OBJ 의 면 {expected_faces}")
    if vest and set(vest) - vested:
        problems.append(f"조끼 조각이 붙지 않은 몸: {sorted(set(vest) - vested)}")
    if seams and not blended_pieces:
        problems.append("--seams 로 지었는데 섞는 비율을 실은 피부가 하나도 없습니다")
    if not seams and blended_pieces:
        problems.append(f"--seams 없이 지었는데 섞는 비율을 실은 피부가 {blended_pieces}개 있습니다")
    parts = {f"{LOOKS}/{name}": name for name in (SHIRT_MATERIAL, TROUSERS_MATERIAL)}
    if worksuit is not None:
        if skin_faces.get(f"{LOOKS}/{WORKSUIT_MATERIAL}"):
            problems.append(f"--split-worksuit 로 지었는데 작업복 재질 {WORKSUIT_MATERIAL} 에 묶인 피부 면이 "
                            f"{skin_faces[f'{LOOKS}/{WORKSUIT_MATERIAL}']}개 있습니다")
        for path, name in parts.items():
            if skin_faces.get(path, 0) != worksuit.get(name):
                problems.append(f"{name} 에 묶인 피부 면 {skin_faces.get(path, 0)} ≠ 가른 면 {worksuit.get(name)}")
    else:
        stray = {name: skin_faces[path] for path, name in parts.items() if skin_faces.get(path)}
        if stray:
            problems.append(f"--split-worksuit 없이 지었는데 상의 · 바지 재질에 묶인 피부 면이 있습니다: {stray}")
    # every material actually bound has one shader with a diffuse, and every texture
    # it references resolves beside the USD (a texture on an unused shader counts for nothing)
    textures = 0
    for path in sorted(bound_materials):
        material = UsdShade.Material(stage.GetPrimAtPath(path))
        # the shader the surface output is actually connected to — a child shader that
        # nothing connects renders as no material
        surface = material.GetSurfaceOutput("mdl")
        source = surface.GetConnectedSource() if surface else None
        if not source or not source[0]:
            problems.append(f"{path}: mdl surface 출력이 셰이더에 연결되지 않았습니다")
            continue
        sh = UsdShade.Shader(source[0].GetPrim())
        if not sh or sh.GetPrim().GetParent() != material.GetPrim():
            problems.append(f"{path}: surface 가 이 재질의 셰이더가 아닌 곳에 연결됐습니다")
            continue
        if not (sh.GetInput("diffuse_texture") or sh.GetInput("diffuse_color_constant")):
            problems.append(f"{path}: 확산 입력이 없습니다")
        for inp in sh.GetInputs():
            value = inp.Get()
            if inp.GetTypeName() == Sdf.ValueTypeNames.Asset:
                if not value:
                    continue
                rel = value.path
                textures += 1
                if not (out.parent / rel).is_file():
                    problems.append(f"{path}.{inp.GetBaseName()}: 텍스처가 없습니다 {rel}")
            elif isinstance(value, float) and not math.isfinite(value):
                problems.append(f"{path}.{inp.GetBaseName()}: 비유한 값입니다")
            elif hasattr(value, "__len__") and not isinstance(value, str) and \
                    any(isinstance(c, float) and not math.isfinite(c) for c in value):
                problems.append(f"{path}.{inp.GetBaseName()}: 비유한 값이 있습니다")
    if not bound_materials:
        problems.append("묶인 재질이 하나도 없습니다")
    elif textures == 0:
        problems.append("묶인 재질 어느 것도 텍스처를 참조하지 않습니다")
    return problems


def build(args: argparse.Namespace) -> int:
    from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

    bio, missing = load_limits(args.geometry_only)
    anthro = load_anthropometry()
    regions = anthro["regions"]

    unknown = set(regions) - set(bio)
    if unknown:
        raise SystemExit(f"rules.json 에 없는 부위 코드가 치수 파일에 있습니다: {sorted(unknown)}")
    absent = set(bio) - set(regions)
    if absent:
        raise SystemExit(f"치수 파일에 빠진 부위가 있습니다: {sorted(absent)}")

    vest_path = getattr(args, "vest", None)
    if vest_path and not (args.skin and getattr(args, "jointed", False)):
        print("--vest 는 --jointed · --skin 과 함께만 씁니다 — 조끼는 관절 인형의 입힌 판에 붙습니다", file=sys.stderr)
        raise SystemExit(3)

    seams_path = getattr(args, "seams", None)
    if seams_path and not (args.skin and getattr(args, "jointed", False)):
        print("--seams 는 --jointed · --skin 과 함께만 씁니다 — 이음새는 관절 인형의 입힌 판에서 섞입니다", file=sys.stderr)
        raise SystemExit(3)

    split_worksuit = bool(getattr(args, "split_worksuit", False))
    if split_worksuit and not args.skin:
        print("--split-worksuit 는 --skin 과 함께만 씁니다 — 작업복은 입힌 판의 피부에 있습니다", file=sys.stderr)
        raise SystemExit(3)

    stem = "keti_dummy_ts15066_v2_jointed" if getattr(args, "jointed", False) else "keti_dummy_ts15066_v1"
    default_name = f"{stem}_skin_vest.usd" if vest_path else f"{stem}_skin.usd" if args.skin else f"{stem}.usd"
    out = Path(args.out) if args.out else REPO_ROOT / "generation/dummy/build" / default_name
    out.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(out)) if not out.exists() else Usd.Stage.CreateNew(str(out), load=Usd.Stage.LoadNone)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/KetiDummy")
    stage.SetDefaultPrim(root.GetPrim())

    # 어느 정본에서 나온 산출물인지 자산 안에 남긴다. 파일만 보고 재현할 수 있어야 한다.
    root.GetPrim().SetCustomDataByKey("keti:schemaVersion", "0.1")
    # 부위 구분은 표 A.1/A.2, **질량과 강성은 표 A.3** 에서 온다. 하나로 뭉뚱그리면
    # 산출된 USD 만 보고 A.3 값을 A.2 값으로 오인한다.
    root.GetPrim().SetCustomDataByKey(
        "keti:standard",
        "KS B ISO TS 15066:2016 (2022 확인) — 부위 구분 표 A.1/A.2, "
        "유효질량·유효탄성계수 표 A.3")
    root.GetPrim().SetCustomDataByKey("keti:anthropometryVersion",
                                      str(anthro.get("anthropometry_version")))
    root.GetPrim().SetCustomDataByKey("keti:limitsStatus",
                                      "pending" if missing else "complete")
    if missing:
        root.GetPrim().SetCustomDataByKey("keti:provisional", True)
        root.GetPrim().SetCustomDataByKey(
            "keti:provisionalReason",
            f"contact_limits {len(missing)}건 미확보 — 질량·강성이 실제 값이 아님")

    # With --skin the body prim is an Xform and the shape hangs under it as `collider`
    # (purpose guide) beside the visual `skin` child — the layout Isaac's own robot assets
    # use. Hiding the shape any other way hides the skin with it: purpose and visibility
    # are pruned per subtree by Hydra, so a guide gprim's children never render even when
    # UsdGeom.ComputePurpose says the child is `default` (measured 2026-09-22). Without
    # --skin the shape is the body itself, byte-identical to the v0.1 asset.
    split_body = bool(args.skin)

    made: dict[str, list[str]] = {}
    centers: dict[str, tuple[float, float, float]] = {}
    for code, spec in regions.items():
        shape = spec.get("shape")
        cx, cy, cz = spec["center"]
        for name, sign in prim_names(code, spec):
            path = f"/KetiDummy/{name}"
            shape_path = f"{path}/{COLLIDER_CHILD}" if split_body else path
            if shape == "sphere":
                geom = UsdGeom.Sphere.Define(stage, shape_path)
                geom.CreateRadiusAttr(float(spec["radius"]))
            elif shape == "capsule":
                geom = UsdGeom.Capsule.Define(stage, shape_path)
                geom.CreateRadiusAttr(float(spec["radius"]))
                geom.CreateHeightAttr(float(spec["height"]))
                geom.CreateAxisAttr(str(spec.get("axis", "z")).upper())
            elif shape == "box":
                geom = UsdGeom.Cube.Define(stage, shape_path)
                geom.CreateSizeAttr(1.0)            # 크기는 scale 로 준다
            else:
                raise SystemExit(f"{code}: 알 수 없는 shape '{shape}'")

            # 이동을 먼저 넣고 크기를 나중에 넣는다. 순서를 뒤집으면 이동량까지 배율이
            # 곱해져, 몸통 상자들이 원래 높이의 size 배 위치로 내려앉는다.
            translate = Gf.Vec3d(float(cx), float(cy) * sign, float(cz))
            if split_body:
                body = UsdGeom.Xform.Define(stage, path)
                body.AddTranslateOp().Set(translate)
                UsdGeom.Imageable(geom.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)
            else:
                geom.AddTranslateOp().Set(translate)
            if shape == "box":
                sx, sy, sz = spec["size"]
                geom.AddScaleOp().Set(Gf.Vec3f(float(sx), float(sy), float(sz)))

            prim = stage.GetPrimAtPath(path)
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(prim)

            # 접촉 보고를 부위 코드로 되돌리는 근거. 프림 경로 규칙에 의존하면 이름을
            # 바꾸는 순간 조용히 어긋나므로 코드 자체를 자산에 담는다.
            prim.CreateAttribute("keti:bodyRegion", Sdf.ValueTypeNames.String,
                                 custom=True).Set(code)
            prim.CreateAttribute("keti:bodyRegionNameKo", Sdf.ValueTypeNames.String,
                                 custom=True).Set(str(bio[code]["name_ko"]))

            # **표 A.3 의 값을 이 파일에 쓰지 않는다.** 두 가지 이유가 겹친다.
            #
            # 1) 유효질량을 강체 질량으로 쓰면 안 된다 — 그것은 두 물체 충돌
            #    모형(식 A.1~A.4)의 파라미터이지 자유물체의 질량이 아니다. 강체
            #    질량으로 쓰면 12부위 합이 481 kg 이 되어 중력·지면반력·낙상
            #    동역학이 인체가 아닌 질량으로 계산된다.
            # 2) 그렇다고 주석 속성으로 싣는 것도 안 된다 — **이 USD 는 배포본에
            #    복사되는 재배포 자산**이고, 배포본은 CC BY 4.0 이다. 표준 수치를
            #    싣는 자리는 조항 참조와 함께인 `meta/rules.json` 하나이고, 여기에
            #    쓰면 유료 표준의 값이 그 형태 밖으로 나간다(빌더가 34개 값을
            #    실제로 싣고 있었다).
            #
            # 값은 `meta/rules.json` 의 `contact_limits` 에 조항 참조와 함께 실려 있다.
            # 자산은 형상·충돌체·부위 코드만 싣는다.

            made.setdefault(code, []).append(name)
            centers[name] = (float(cx), float(cy) * sign, float(cz))

    # 부위를 서로 붙인다. 부모가 null 인 부위가 root LINK(아티큘레이션 기저 강체)가 된다.
    root_code = [c for c, s in regions.items() if s.get("parent") is None]
    if len(root_code) != 1:
        raise SystemExit(f"parent 가 null 인 부위는 정확히 하나여야 합니다: {root_code}")
    root_prim_name = made[root_code[0]][0]
    # ArticulationRootAPI 는 root LINK 프림(BR-04) 자체에 준다.
    # 대안도 시험했다: 마커를 /KetiDummy(Xform 조상)로 한 단계
    # 옮기고 root LINK 에만 kinematicEnabled 를 주는 구성(Isaac Sim 로봇 자산들이
    # 흔히 쓰는 배치). PhysX 가 "같은 프림에 ArticulationRootAPI 와 kinematicEnabled
    # 를 동시에 못 준다"는 제약은 피했지만 더 근본적인 다른
    # 제약에 걸렸다 — "Articulations with kinematic bodies are not supported,
    # please exclude joint (.../BR_03/fixed_to_BR_04) from articulation." 등, 루트가
    # 참여하는 고정조인트마다 하나씩. 즉 PhysX 아티큘레이션은 마커 위치와 무관하게
    # kinematic 인 링크를 아예 허용하지 않는다. 이 17-프림 단일
    # 아티큘레이션 구조에서는 kinematic 루트가 근본적으로 불가능하다(mdp/events.py
    # 모듈 docstring 에 근거 전문). 그래서 이 배치(같은 프림)를 그대로 유지한다 —
    # 조상으로 옮기는 것 자체는 동작했지만(원래 제약은 피함) 대안이 되지 못했고,
    # 옮기면 check_contact_resolution.py 등 root LINK 프림 자체에서 BR 코드와
    # ArticulationRootAPI 를 함께 찾는 도구들이 조용히 깨진다.
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(f"/KetiDummy/{root_prim_name}"))
    if args.root == "kinematic":
        UsdPhysics.RigidBodyAPI(
            stage.GetPrimAtPath(f"/KetiDummy/{root_prim_name}")).CreateKinematicEnabledAttr(True)

    joints = 0
    for code, spec in regions.items():
        parent = spec.get("parent")
        if parent is None:
            continue
        for idx, name in enumerate(made[code]):
            # 좌우 한 쌍이 한 쌍인 부모에 붙으면 같은 쪽끼리, 아니면 단일 부모에 붙는다.
            pnames = made[parent]
            pname = pnames[idx] if len(pnames) == len(made[code]) else pnames[0]
            j = UsdPhysics.FixedJoint.Define(stage, f"/KetiDummy/{name}/fixed_to_{pname}")
            j.CreateBody0Rel().SetTargets([Sdf.Path(f"/KetiDummy/{pname}")])
            j.CreateBody1Rel().SetTargets([Sdf.Path(f"/KetiDummy/{name}")])
            # 로컬 프레임을 주지 않으면 양쪽 다 원점으로 해석되어, 물리엔진이 두 부위의
            # 원점을 겹치려 든다. 더미가 한 점으로 무너지므로 반드시 현재 상대 위치를 준다.
            #
            # 그런데 이 값은 월드 좌표가 아니라 **body0 의 로컬 프레임** 값이다. 몸통 부위는
            # 상자를 scale 로 키워 만들었으므로 그 로컬 프레임에는 배율이 들어 있고, 월드
            # 기준 상대 위치를 그대로 적으면 물리엔진이 배율만큼 줄여 읽는다. 실측(2026-08-19):
            # BR-05 를 몸통 아래 0.12 m 로 적었더니 0.0192 m 에 놓였고(배율 0.16), 첫 스텝에
            # 아티큘레이션이 몸통 안으로 접혀 부위 대부분이 저작 위치에서 사라졌다.
            # 부모의 월드 변환 역으로 옮겨 적는다 — 회전이 붙어도 같은 식이 성립한다.
            # With --skin the scaled shape is the parent's `collider` child, whose world
            # transform is exactly v1's body transform — so this is v1's anchor either way.
            parent_shape = stage.GetPrimAtPath(
                f"/KetiDummy/{pname}/{COLLIDER_CHILD}" if split_body else f"/KetiDummy/{pname}")
            parent_xf = UsdGeom.Xformable(parent_shape).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            local0 = parent_xf.GetInverse().Transform(Gf.Vec3d(*centers[name]))
            local0 = Gf.Vec3f(*[float(v) for v in local0])
            if split_body:
                # The skinned body is an unscaled Xform: author the anchor the engine reads
                # off v1, not the plain offset (one ulp apart in 4 of 16 anchors, enough to
                # change an episode's outcome — see engine_anchor).
                local0 = Gf.Vec3f(*engine_anchor(local0, _scale_op(parent_shape)))
            j.CreateLocalPos0Attr(local0)
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            joints += 1

    # 부위끼리는 서로 충돌시키지 않는다. 인접 부위는 형상이 맞닿아 있고 고정조인트로
    # 묶여 있어, 이대로 두면 매 스텝 자가 접촉이 발생해 부위별 접촉 배열이 상시 포화되고,
    # 접촉이 없는 프레임에도 A2 규칙이 성립해 라벨이 전부 fail 로 나온다.
    #
    # 그런데 이것은 자산이 아니라 런타임에서 꺼야 한다. CollisionGroup 이 자기 자신을
    # 거르게 했더니 그룹 구성원이 **아무것과도** 충돌하지 않아 로봇이 더미를 통과했고,
    # 아티큘레이션 플래그는 PhysxSchema 로 적용해야 먹는데 그 스키마는 시뮬레이터를
    # 띄워야 열린다. 자산은 물리엔진 중립으로 두고, 씬을 구성하는 쪽이 아티큘레이션의
    # self-collision 을 끈다.
    root_prim = stage.GetPrimAtPath(f"/KetiDummy/{root_prim_name}")
    root_prim.CreateAttribute("keti:requiresSelfCollisionDisabled",
                              Sdf.ValueTypeNames.Bool, custom=True).Set(True)

    spec, jointed_summary = None, None
    if getattr(args, "jointed", False):
        from generation.dummy import joints_spec as js
        from generation.dummy import jointed

        spec = js.load()
        jointed_summary = jointed.apply(stage, spec, split_body=split_body)
        if getattr(args, "gait", None):
            gait_path = Path(args.gait)
            jointed.embed_gait(stage, json.loads(gait_path.read_text(encoding="utf-8")),
                               hashlib.sha256(gait_path.read_bytes()).hexdigest())

    # Everything above is byte-identical with and without --skin (the v1 hash in
    # meta/assets.json pins it); the skin only adds children, materials and purposes.
    seams = None
    if seams_path:
        from generation.dummy import skin_seams

        try:
            seams = (skin_seams.load_seam_spec(Path(seams_path)), skin_seams.body_links(spec), Path(seams_path))
        except ValueError as e:
            raise SystemExit(f"피부 이음새 명세를 쓸 수 없습니다: {e}")
    skin = (attach_skin(stage, Path(args.skin), anthro, out, extra_prims=js.foot_primitives(spec) if spec else (),
                        seams=seams, split_worksuit=split_worksuit)
            if args.skin else None)
    vest = None
    if vest_path:
        from generation.dummy import vest as vest_mod

        try:
            vest_spec = vest_mod.load_vest_spec(Path(vest_path))
            pieces = vest_mod.vest_pieces(skin["mesh"], skin["face_prims"], skin["prims"], vest_spec)
        except ValueError as e:
            raise SystemExit(f"조끼를 만들 수 없습니다: {e}")
        vest = attach_vest(stage, pieces, vest_spec, spec_path=Path(vest_path))

    stage.GetRootLayer().Save()

    if skin is not None:
        problems = verify_skin_usd(out, skin["faces"],
                                   vest={k: v["faces"] for k, v in vest["parts"].items()} if vest else None,
                                   seams=seams is not None, worksuit=skin["worksuit"])
        if problems:
            for pr in problems:
                print(f"  {pr}", file=sys.stderr)
            raise SystemExit(f"시각 메시 산출물 검사 실패 {len(problems)}건: {out}")
        # The skin is only a visual variant if the engine reads the same body as v1. Checked
        # here, on every skinned build, so an asset that differs never survives the build.
        problems = verify_engine_identity(args, out)
        if problems:
            out.unlink(missing_ok=True)
            for pr in problems:
                print(f"  {pr}", file=sys.stderr)
            raise SystemExit(f"입힌 더미가 같은 입력의 v1 과 물리엔진이 읽는 값으로 같지 않습니다 {len(problems)}건 "
                             f"— 산출물을 지웠습니다: {out}")

    total = sum(len(v) for v in made.values())
    try:
        shown = out.relative_to(REPO_ROOT)
    except ValueError:
        shown = out                          # --out 이 저장소 밖이어도 실패하지 않는다
    print(f"생성: {shown}")
    fixed = joints - (jointed_summary["replaced_fixed_joints"] if jointed_summary else 0)
    print(f"  부위 {len(made)}종 · 프림 {total}개 · 고정조인트 {fixed}개 · 루트 {root_prim_name}"
          f" ({args.root})")
    if jointed_summary:
        print(f"  관절 인형: 관절 {len(jointed_summary['joints'])}개 · 자유도 {jointed_summary['dofs']} · "
              f"발 {len(jointed_summary['feet'])} · 질량 {jointed_summary['total_mass_kg']:.1f} kg (형상마다 밀도로)")
    pairs = [c for c, v in made.items() if len(v) > 1]
    print("  자가 충돌: 씬 구성 시 꺼야 함 (keti:requiresSelfCollisionDisabled)")
    print(f"  좌우 한 쌍인 부위 {len(pairs)}종: {', '.join(pairs)}")
    print("    → 같은 BR 코드의 프림이 둘이므로 라벨링은 합산이 아니라 최대값으로 모을 것")
    if skin is not None:
        print(f"  시각 메시: 자식 메시 {len(skin['parts'])}개 · 정점 {skin['vertices']} · 면 {skin['faces']}"
              f" · 재질 {skin['materials']} · 텍스처 {len(skin['textures'])} ({TEXTURE_SUBDIR}/)")
        for name, s in skin["parts"].items():
            print(f"    {name:14s} 정점 {s['vertices']:5d} · 면 {s['faces']:5d} · 재질 {s['materials']}")
        print("    프리미티브는 purpose=guide 로 숨김 · 콜라이더·조인트·질량은 "
              + ("같은 판의 맨몸과" if jointed_summary else "v1 과")
              + " 동일 · 물리 API 없음 (엔진 값 대조: check_skin_physics.py)")
        if seams is not None:
            moved = skin["moves"]
            print(f"  피부 이음새: 엉뚱한 몸에서 옮긴 조각 {len(moved)}개 · 면 {sum(mv['faces'] for mv in moved)}"
                  f" · 섞는 정점 {sum(s['blended'] for s in skin['parts'].values())} (조각마다 그 피부 메시의 속성으로)")
            for mv in moved:
                print(f"    {mv['from']:12s} → {mv['to']:12s} 면 {mv['faces']}")
        if skin["worksuit"] is not None:
            print("  작업복: " + " · ".join(f"{name} 면 {n}" for name, n in skin["worksuit"].items())
                  + " (입력은 작업복 재질과 같음 — 겉모습 그대로)")
    if vest is not None:
        print(f"  조끼: 조각 {len(vest['parts'])}개 (보이기만 · 물리 없음 · 재질 둘, 텍스처 없음)")
        for name, s in vest["parts"].items():
            print(f"    {name:14s} 정점 {s['vertices']:5d} · 면 {s['faces']:5d} · 띠 면 {s['band_faces']}")
    if missing:
        print(f"  경고: 잠정 산출물입니다. contact_limits {len(missing)}건이 비어 질량·강성이"
              " 실제 값이 아닙니다.")
        print("        접촉 판정 해상도 확인 외의 용도로 쓰지 마십시오.")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return build(args)
    except SystemExit:
        raise
    except Exception as e:                          # noqa: BLE001 — 원인을 그대로 보여준다
        print(f"더미 생성 실패: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
