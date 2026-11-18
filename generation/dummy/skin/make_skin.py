#!/usr/bin/env python3
"""Generate the dummy's visual mesh from skin_spec.yaml with MakeHuman (MPFB2) inside Blender.

    .venv/bin/python generation/dummy/skin/make_skin.py            # -> generation/dummy/skin/build/
    .venv/bin/python generation/dummy/skin/make_skin.py --update-lock   # textures changed on purpose

One file, two stages. Run under an ordinary Python it is the *driver*: it reads
the yaml spec and anthropometry.yaml (Blender's bundled Python has numpy but no
PyYAML), validates them, resolves the target stature, writes a JSON copy of the
spec and re-launches itself inside Blender:

    blender -b --python-exit-code 1 --python make_skin.py -- --resolved <json> --out <dir>

Inside Blender it creates a human from the macros, bisects the `height` macro
until the *exported* mesh's stature matches the anthropometry canon (the
basemesh's vertex coordinates are the un-morphed basis and must not be
measured), dresses it, turns it to face +x, exports OBJ + MTL + textures with
`GAMEENGINE` materials (the only mode whose textures survive the exporter),
checks the export stands in the dummy frame, and writes:

    build/keti_skin.obj · keti_skin.mtl · textures/<map>.png
    build/skin_manifest.json      what was made, from which assets, with SHA256s
    textures.lock (tracked)       SHA256 · size · pack member of every exported texture
                                  and of every pack file the figure was built from

The lock is the canon for "which textures did we ship" and "which pack files made
the figure": a later run that differs stops unless --update-lock says the change is
intended, and the lock is only ever written after every row was matched byte for
byte against the pinned pack zip (--pack). Being installed in MPFB's asset
directory is not proof of provenance.

Exit codes: 0 ok / 1 generation or check failed / 2 MPFB or the system asset
pack missing / 3 argument or environment error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPEC = HERE / "skin_spec.yaml"
ANTHRO = HERE.parent / "anthropometry.yaml"          # the file build_dummy.py reads
OUT = HERE / "build"

# HumanService.add_mhclo_asset asset_type values that have been exercised. Others may
# exist in MPFB; add them here once they have been seen to work.
ASSET_TYPES = ("Clothes", "Hair", "Eyes", "Eyebrow")
ASSET_SUBDIRS = ("clothes", "hair", "eyes", "eyebrows")
# Macro keys TargetService.get_default_macro_info_dict() returns (MPFB 2.0.17);
# the Blender stage checks the live dict again.
MACRO_KEYS = ("gender", "age", "muscle", "weight", "proportions", "cupsize", "firmness", "race")
RACE_KEYS = ("asian", "caucasian", "african")
FRAME = {"front": "+x", "up": "+z", "left": "+y"}
LOCK_COLUMNS = ("name", "sha256", "bytes", "source", "license")
MPFB_MODULE = "bl_ext.user_default.mpfb"
# The bisection stops when the exported stature is this close to the target; the
# spec's stature_tolerance is the acceptance test on the final mesh.
STATURE_CONVERGENCE_M = 0.002


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- spec (driver side; also imported by the tests) -----------------------------------

def _unit(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: macro value must be a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or not 0.0 <= v <= 1.0:
        raise ValueError(f"{where}: macro value must lie in [0, 1], got {value!r}")
    return v


def validate_spec(spec: dict) -> dict:
    """Refuse a spec the generator could not honour, before Blender is launched."""
    for key in ("skin_version", "source", "stature_from", "stature_tolerance", "macro",
                "height_search", "skin", "clothes", "pose", "fit", "material_type", "frame", "export"):
        if key not in spec:
            raise ValueError(f"spec lacks {key!r}")
    src = spec["source"]
    if src.get("license_of_output") != "CC0-1.0":
        raise ValueError("source.license_of_output must be CC0-1.0 — that is the licence the "
                         "asset entry and SKIN-LICENSE.md claim for the output")
    pack = src.get("system_assets") or {}
    if not (str(pack.get("url", "")).startswith("https://") and len(str(pack.get("sha256", ""))) == 64):
        raise ValueError("source.system_assets needs an https url and a sha256 — it is what "
                         "the generator tells a machine without the pack to fetch")
    if (REPO_ROOT / str(spec["stature_from"])).resolve() != ANTHRO.resolve():
        raise ValueError(f"stature_from must be the anthropometry canon {ANTHRO.relative_to(REPO_ROOT)}, "
                         f"got {spec['stature_from']!r}")
    tol = spec["stature_tolerance"]
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not 0 < tol <= 0.05:
        raise ValueError(f"stature_tolerance must be in (0, 0.05], got {tol!r}")
    macro = spec["macro"]
    if not isinstance(macro, dict) or not macro:
        raise ValueError("macro must be a non-empty mapping")
    if "height" in macro:
        raise ValueError("macro.height is searched from stature_from, not set")
    for k, v in macro.items():
        if k not in MACRO_KEYS:
            raise ValueError(f"macro.{k}: unknown MPFB macro (known: {MACRO_KEYS})")
        if k == "race":
            if not isinstance(v, dict) or set(v) - set(RACE_KEYS):
                raise ValueError(f"macro.race must map {RACE_KEYS}")
            for rk, rv in v.items():
                _unit(rv, f"macro.race.{rk}")
        else:
            _unit(v, f"macro.{k}")
    hs = spec["height_search"]
    try:
        lo, hi, n = float(hs["lo"]), float(hs["hi"]), int(hs["max_iter"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"height_search needs lo, hi, max_iter: {e}") from e
    if not (0.0 <= lo < hi <= 1.0) or n < 1:
        raise ValueError(f"height_search must satisfy 0 <= lo < hi <= 1 and max_iter >= 1, got {hs}")
    skin = spec["skin"]
    if not isinstance(skin, dict) or skin.get("subdir") != "skins" or not skin.get("label"):
        raise ValueError("skin must be {subdir: skins, label: <mhmat label>}")
    clothes = spec["clothes"]
    if not isinstance(clothes, list) or not clothes:
        raise ValueError("clothes must be a non-empty list — a naked dummy is not the design")
    for a in clothes:
        if not isinstance(a, dict) or not {"subdir", "label", "type"} <= set(a):
            raise ValueError(f"clothes entry needs subdir, label, type: {a!r}")
        if a["subdir"] not in ASSET_SUBDIRS or a["type"] not in ASSET_TYPES or not a["label"]:
            raise ValueError(f"clothes entry: subdir in {ASSET_SUBDIRS}, type in {ASSET_TYPES}: {a!r}")
    pose = spec.get("pose")
    if not isinstance(pose, dict) or not str(pose.get("rig", "")).strip():
        raise ValueError("pose must name a rig — MakeHuman's default A pose puts the arms 0.4 m "
                         "from the arm colliders (measured 2026-09-22)")
    aims = pose.get("aim")
    if not isinstance(aims, list) or not aims:
        raise ValueError("pose.aim must list the bones to aim")
    for a in aims:
        if not isinstance(a, dict) or not a.get("bone") or not isinstance(a.get("direction"), list):
            raise ValueError(f"pose.aim entry needs bone and direction: {a!r}")
        d = a["direction"]
        if len(d) != 3 or any(x is not None and (isinstance(x, bool) or not isinstance(x, (int, float))
                                                 or not math.isfinite(x)) for x in d):
            raise ValueError(f"pose.aim {a['bone']}: direction must be three finite numbers (null keeps a component)")
        given = [x for x in d if x is not None]
        if not given or math.hypot(*given) < 1e-6:
            raise ValueError(f"pose.aim {a['bone']}: direction must not be zero — nothing to aim at")
    fit = spec.get("fit")
    if not isinstance(fit, dict):
        raise ValueError("fit must be a mapping")
    lat = fit.get("max_lateral_offset_m")
    if isinstance(lat, bool) or not isinstance(lat, (int, float)) or not 0 < lat < 0.3:
        raise ValueError(f"fit.max_lateral_offset_m must be in (0, 0.3), got {lat!r}")
    if spec["material_type"] != "GAMEENGINE":
        raise ValueError("material_type must be GAMEENGINE — other MPFB material modes are node groups "
                         "the exporter cannot carry (measured: no map_Kd, 5 of 11 materials)")
    if spec["frame"] != FRAME:
        raise ValueError(f"frame must be the dummy frame {FRAME} (anthropometry.yaml), got {spec['frame']!r}")
    exp = spec["export"]
    for key in ("obj", "textures_dir", "manifest", "textures_lock", "yaw_deg"):
        if key not in exp:
            raise ValueError(f"export lacks {key!r}")
    if exp.get("format") != "obj":
        raise ValueError("export.format must be obj — region_split.read_obj is what consumes it")
    yaw = exp["yaw_deg"]
    if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(float(yaw)):
        raise ValueError(f"export.yaw_deg must be a finite number, got {yaw!r}")
    return spec


def load_spec(path: Path = SPEC) -> dict:
    import yaml
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"{path}: not a mapping")
    return validate_spec(spec)


def resolve(spec: dict, spec_path: Path = SPEC) -> dict:
    """The spec plus what the Blender stage cannot read for itself."""
    import yaml
    anthro = yaml.safe_load(ANTHRO.read_text(encoding="utf-8"))
    stature = anthro.get("stature")
    if isinstance(stature, bool) or not isinstance(stature, (int, float)) or not 1.0 < stature < 2.5:
        raise ValueError(f"{ANTHRO.relative_to(REPO_ROOT)}: stature must be a height in metres, got {stature!r}")
    out = dict(spec)
    out["stature_target"] = float(stature)
    out["anthropometry_regions"] = anthro["regions"]      # the fit check needs the primitives
    out["anthropometry_version"] = str(anthro.get("anthropometry_version"))
    out["anthropometry_sha256"] = sha256_of(ANTHRO)
    resolved_spec = Path(spec_path).resolve()
    try:
        out["spec_path"] = str(resolved_spec.relative_to(REPO_ROOT))
    except ValueError:                       # a spec outside the repo (experiments) is fine
        out["spec_path"] = str(resolved_spec)
    out["spec_sha256"] = sha256_of(Path(spec_path))
    return out


# ---- textures.lock ----------------------------------------------------------------------

def parse_lock(path: Path) -> list[dict]:
    rows = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) != len(LOCK_COLUMNS):
            raise ValueError(f"{path}:{lineno}: expected {len(LOCK_COLUMNS)} columns {LOCK_COLUMNS}, got {parts}")
        row = dict(zip(LOCK_COLUMNS, parts))
        if len(row["sha256"]) != 64:
            raise ValueError(f"{path}:{lineno}: sha256 must be 64 hex chars")
        int(row["sha256"], 16)
        row["bytes"] = int(row["bytes"])
        if row["license"] != "CC0-1.0":
            raise ValueError(f"{path}:{lineno}: {row['name']}: licence must be CC0-1.0, got {row['license']!r}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no entries — an empty lock cannot pin anything")
    names = [r["name"] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate texture names")
    return rows


def format_lock(rows: list[dict]) -> str:
    head = ("# generation/dummy/skin/textures.lock — make_skin.py 가 내보낸 텍스처와, 사람을 만든 팩 파일의 정본. 파일마다 한 줄.\n"
            "# 다음 실행에서 이 표와 다르면 생성기가 멈춘다 (--update-lock 이 의도된 변경). 쓰거나 갱신할 때는\n"
            "# --pack <zip> 으로 팩과 바이트를 대조한 뒤에만 쓴다 — 자산 디렉터리에 있다는 것은 출처의 증명이 아니다.\n"
            "#\n"
            f"# 열: {' '.join(LOCK_COLUMNS)}\n"
            "#   name    내보낸 텍스처는 파일 이름, 팩 파일(.mhclo · .obj · .mhmat)은 팩 안의 경로\n"
            "#   source  pack:<MakeHuman 시스템 자산 팩 안의 경로> — 팩은 skin_spec.yaml source.system_assets\n"
            "#\n")
    body = "".join(f"{r['name']} {r['sha256']} {r['bytes']} {r['source']} {r['license']}\n" for r in rows)
    return head + body


def mhclo_refs(mhclo: Path) -> list[Path]:
    """The files an .mhclo builds its asset from: the geometry (obj_file) and the material.
    Both are resolved beside the .mhclo and must exist — MPFB reads exactly these."""
    out = []
    for line in mhclo.read_text(encoding="utf-8", errors="ignore").splitlines():
        t = line.split(None, 1)
        if len(t) == 2 and t[0] in ("obj_file", "material"):
            ref = (mhclo.parent / t[1].strip()).resolve()
            if not ref.is_file():
                raise ValueError(f"{mhclo.name}: {t[0]} {t[1].strip()!r} does not exist beside it")
            out.append(ref)
    if not any(f.suffix == ".obj" for f in out):
        raise ValueError(f"{mhclo.name}: no obj_file line")
    return out


def verify_against_pack(pack: Path, expected_sha256: str, rows: list[dict]) -> list[str]:
    """Is every `pack:<member>` row byte-identical to that member of the pinned pack zip?

    Living in MPFB's asset directory proves nothing about where a file came from — a
    swapped geometry file would keep its label and still be attributed to the CC0 pack.
    The zip is the pinned source, so a lock is written only after each row's bytes
    were matched against it."""
    problems: list[str] = []
    got = sha256_of(pack)
    if got != expected_sha256:
        return [f"{pack.name}: sha256 {got[:12]}… is not the pinned pack {expected_sha256[:12]}…"]
    with zipfile.ZipFile(pack) as z:
        names = set(z.namelist())
        for r in rows:
            if not r["source"].startswith("pack:"):
                problems.append(f"{r['name']}: source {r['source']!r} is not a pack member")
                continue
            member = r["source"][len("pack:"):]
            if member not in names:
                problems.append(f"{r['name']}: {member} is not in the pack")
                continue
            if hashlib.sha256(z.read(member)).hexdigest() != r["sha256"]:
                problems.append(f"{r['name']}: bytes differ from the pack's {member}")
    return problems


def is_texture_row(row: dict) -> bool:
    """Exported textures are named by file name; pack files by their pack path."""
    return "/" not in row["name"]


# ---- Blender stage ------------------------------------------------------------------------

def _blender_run(argv: list[str]) -> int:
    import bpy                                     # noqa: F401 — only here
    import importlib
    import numpy as np

    from generation.dummy.skin import region_split as rs

    p = argparse.ArgumentParser()
    p.add_argument("--resolved", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--update-lock", action="store_true")
    p.add_argument("--pack", default=None)
    args = p.parse_args(argv)
    spec = json.loads(Path(args.resolved).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tex_dir = out / "textures"
    lock_path = HERE / spec["export"]["textures_lock"]

    def say(*a):
        print("SKIN", *a, flush=True)

    # -- MPFB ----------------------------------------------------------------------------
    if MPFB_MODULE not in [a.module for a in bpy.context.preferences.addons]:
        try:
            bpy.ops.preferences.addon_enable(module=MPFB_MODULE)
        except Exception as e:                     # noqa: BLE001
            say(f"MPFB2 is not installed in this Blender ({bpy.app.version_string}): {e}")
            say(f"  get {spec['source']['mpfb_download']['url']}")
            say(f"  sha256 {spec['source']['mpfb_download']['sha256']}")
            say("  bpy.ops.extensions.package_install_files(filepath=..., repo='user_default'); "
                "bpy.ops.preferences.addon_enable(module='bl_ext.user_default.mpfb'); bpy.ops.wm.save_userpref()")
            return 2
    mpfb = importlib.import_module(MPFB_MODULE)
    S = importlib.import_module(MPFB_MODULE + ".services")
    H, A, T, O, L = S.HumanService, S.AssetService, S.TargetService, S.ObjectService, S.LocationService
    mpfb_version = ".".join(str(x) for x in getattr(mpfb, "VERSION", ()))
    if mpfb_version != spec["source"]["mpfb_version"]:
        say(f"MPFB {mpfb_version} is installed but the spec was written against {spec['source']['mpfb_version']} — "
            "update source.mpfb_version after checking the export still matches")
        return 2
    data_dir = Path(L.get_user_data()).resolve()

    # -- system assets -------------------------------------------------------------------
    wanted = [(spec["skin"]["subdir"], "mhmat", spec["skin"]["label"])] + \
             [(a["subdir"], "mhclo", a["label"]) for a in spec["clothes"]]
    missing = []
    paths: dict[tuple[str, str], str] = {}
    for subdir, kind, label in wanted:
        lst = A.get_asset_list(subdir, kind)
        if label not in lst:
            missing.append(f"{subdir}/{label}")
        else:
            paths[(subdir, label)] = lst[label]["full_path"]
    if missing:
        pack = spec["source"]["system_assets"]
        say(f"system assets missing: {missing}")
        say(f"  install the MakeHuman system asset pack ({pack['license']}):")
        say(f"  {pack['url']}")
        say(f"  sha256 {pack['sha256']} · {pack['bytes']} bytes")
        say("  AssetService.fix_and_extract_asset_pack_zip(<zip>, LocationService.get_user_data())")
        return 2

    # -- macros ------------------------------------------------------------------------------
    defaults = T.get_default_macro_info_dict()
    unknown = set(spec["macro"]) - set(defaults)
    if unknown:
        say(f"macro keys MPFB {mpfb_version} does not know: {sorted(unknown)} (known {sorted(defaults)})")
        return 1

    def macro_for(height: float) -> dict:
        m = json.loads(json.dumps(defaults))
        for k, v in spec["macro"].items():
            if isinstance(v, dict):
                m[k].update(v)
            else:
                m[k] = v
        m["height"] = float(height)
        return m

    def select_only(objs):
        bpy.ops.object.select_all(action="DESELECT")
        for ob in objs:
            ob.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]

    # -- stature: bisect the height macro on the exported bare mesh -------------------------
    target = float(spec["stature_target"])
    scratch = out / "_bare.obj"

    def stature_at(height: float) -> float:
        bm = H.create_human(macro_detail_dict=macro_for(height), feet_on_ground=True)
        select_only([bm])
        bpy.ops.wm.obj_export(filepath=str(scratch), export_selected_objects=True, export_materials=False,
                              export_uv=False, export_normals=False, apply_modifiers=True,
                              forward_axis="Y", up_axis="Z", global_scale=1.0)
        st = rs.stature_of(rs.read_obj(scratch).V)
        O.delete_object(bm)
        return st

    hs = spec["height_search"]
    lo, hi = float(hs["lo"]), float(hs["hi"])
    rows = [(lo, stature_at(lo)), (hi, stature_at(hi))]
    if not rows[0][1] < target < rows[1][1]:
        say(f"stature {target} m is outside height_search [{lo}, {hi}] -> [{rows[0][1]:.3f}, {rows[1][1]:.3f}] m")
        scratch.unlink(missing_ok=True)
        return 1
    best = None
    for _ in range(int(hs["max_iter"])):
        mid = (lo + hi) / 2
        st = stature_at(mid)
        rows.append((mid, st))
        if best is None or abs(st - target) < abs(best[1] - target):
            best = (mid, st)
        if abs(st - target) <= STATURE_CONVERGENCE_M:
            break
        if st < target:
            lo = mid
        else:
            hi = mid
    scratch.unlink(missing_ok=True)
    height, stature_bare = best
    tol_m = target * float(spec["stature_tolerance"])
    if abs(stature_bare - target) > tol_m:
        say(f"bisection did not reach {target} m within {tol_m*1000:.0f} mm: best height {height:.5f} -> {stature_bare:.4f} m")
        return 1
    say(f"height macro {height:.5f} -> stature {stature_bare:.4f} m (target {target} m, {len(rows)} exports)")

    # -- the dressed, posed human ----------------------------------------------------------
    bm = H.create_human(macro_detail_dict=macro_for(height), feet_on_ground=True)
    # the rig first: clothes added afterwards get armature modifiers with interpolated
    # weights (set_up_rigging), so the pose below moves them with the body
    rig = H.add_builtin_rig(bm, spec["pose"]["rig"], import_weights=True)
    if rig is None:
        say(f"MPFB has no builtin rig {spec['pose']['rig']!r}")
        return 1
    H.set_character_skin(paths[(spec["skin"]["subdir"], spec["skin"]["label"])], bm,
                         skin_type=spec["material_type"], material_instances=True)
    for a in spec["clothes"]:
        H.add_mhclo_asset(paths[(a["subdir"], a["label"])], bm, asset_type=a["type"],
                          subdiv_levels=0, material_type=spec["material_type"])
    # Aim each bone (head -> tail) along the spec's direction in the rig's frame, one at a
    # time down the chain so a child's current direction already includes its parent's turn.
    # PoseBone.matrix is settable in armature space with the parent's pose accounted for.
    from mathutils import Matrix, Vector
    for a in spec["pose"]["aim"]:
        bpy.context.view_layer.update()
        pb = rig.pose.bones.get(a["bone"])
        if pb is None:
            say(f"rig {spec['pose']['rig']!r} has no bone {a['bone']!r}")
            return 1
        head = pb.head.copy()
        current = (pb.tail - pb.head).normalized()
        # a null component keeps the bone's current one (the legs keep their fore-aft tilt)
        want = Vector([current[i] if x is None else float(x) for i, x in enumerate(a["direction"])])
        if want.length < 1e-6:
            say(f"pose.aim {a['bone']}: the target direction collapsed to zero")
            return 1
        want.normalize()
        turn = current.rotation_difference(want).to_matrix().to_4x4()
        pb.matrix = Matrix.Translation(head) @ turn @ Matrix.Translation(-head) @ pb.matrix
    bpy.context.view_layer.update()
    # MakeHuman faces -y in Blender; the dummy frame faces +x. The rig is the top of the
    # hierarchy once added (the clothes hang under it, not under the basemesh), so it is
    # the object to turn and the root to collect the meshes from — the exporter writes
    # world-space coordinates, so the children follow.
    top = bm
    while top.parent is not None:
        top = top.parent
    top.rotation_euler = (0.0, 0.0, math.radians(float(spec["export"]["yaw_deg"])))
    objs = [ob for ob in (top, *top.children_recursive) if ob.type == "MESH"]
    if len(objs) != 1 + len(spec["clothes"]):
        say(f"expected the body and {len(spec['clothes'])} assets under {top.name!r}, found {[o.name for o in objs]}")
        return 1

    # provenance of every image the materials reference, before the exporter copies them
    images: dict[str, Path] = {}
    for ob in objs:
        if ob.type != "MESH":
            continue
        for mat in ob.data.materials:
            if not mat or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                img = getattr(node, "image", None)
                if img is None:
                    continue
                src = Path(bpy.path.abspath(img.filepath_from_user())).resolve()
                prev = images.get(src.name)
                if prev is not None and prev != src:
                    say(f"two different images share the basename {src.name}: {prev} / {src}")
                    return 1
                images[src.name] = src

    obj_path = out / Path(spec["export"]["obj"]).name
    mtl_path = obj_path.with_suffix(".mtl")
    manifest_path = out / Path(spec["export"]["manifest"]).name

    def discard_export():
        # A failed run must leave nothing build_dummy.py could consume: the export is
        # written before the checks, so on any failure from here on it goes, and so
        # does any manifest from an earlier run.
        for stale in (obj_path, mtl_path, manifest_path):
            stale.unlink(missing_ok=True)
        shutil.rmtree(tex_dir, ignore_errors=True)

    discard_export()
    select_only(objs)
    bpy.ops.wm.obj_export(filepath=str(obj_path), export_selected_objects=True, export_materials=True,
                          export_pbr_extensions=True, export_uv=True, export_normals=True,
                          export_triangulated_mesh=True, apply_modifiers=True, path_mode="COPY",
                          forward_axis="Y", up_axis="Z", global_scale=1.0)
    mtl_path = obj_path.with_suffix(".mtl")
    if not mtl_path.is_file():
        say("the exporter wrote no MTL — materials did not survive")
        return 1

    # move the copied textures next to the OBJ into textures/ and point the MTL there
    tex_dir.mkdir()
    lines = mtl_path.read_text(encoding="utf-8").splitlines()
    rewritten = []
    exported: dict[str, Path] = {}
    for line in lines:
        t = line.split()
        if t and t[0] in ("map_Kd", "map_Bump", "map_Ks", "map_d", "map_Pr", "map_Pm", "map_Ns", "norm", "bump"):
            src = out / t[-1]
            dst = tex_dir / src.name
            # the same file is referenced twice (map_Kd and map_d for its alpha): move it
            # on the first reference, rewrite the path on every one
            if src.is_file() and not dst.exists():
                shutil.move(str(src), str(dst))
            if dst.is_file():
                exported[src.name] = dst
                t[-1] = f"{tex_dir.name}/{src.name}"
                line = " ".join(t)
        rewritten.append(line)
    mtl_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    if not exported:
        say("the MTL references no textures — check material_type (GAMEENGINE) and the asset pack")
        return 1

    # -- checks on what was written ------------------------------------------------------------
    mesh = rs.read_obj(obj_path)
    if mesh.mtllib != mtl_path.name:
        say(f"OBJ references {mesh.mtllib!r}, expected {mtl_path.name}")
        return 1
    if bm.name not in mesh.objects:
        say(f"the body object {bm.name!r} is not in the export: {mesh.objects}")
        return 1
    body = mesh.FO == mesh.objects.index(bm.name)
    body_v = mesh.V[np.unique(mesh.F[body])]
    # The clothes' delete groups strip the body under them (legs, feet), so the dressed
    # body's z-range is not a stature — its crown is. The soles are on z = 0 by
    # feet_on_ground, which the frame check below confirms on the whole mesh.
    stature_dressed = float(body_v[:, 2].max())
    if abs(stature_dressed - target) > tol_m:
        say(f"dressed body crown at {stature_dressed:.4f} m is off the target {target} m by more than {tol_m*1000:.0f} mm")
        return 1
    orient = rs.orientation_report(mesh.V)
    y_half = max(abs(orient["y_span"][0]), abs(orient["y_span"][1]))
    problems = []
    if not orient["faces_plus_x"]:
        problems.append(f"the figure does not face +x (foot front {orient['foot_front_m']:.3f} m, back {orient['foot_back_m']:.3f} m)")
    if abs(orient["y_mean"]) > 0.02 or abs(orient["y_span"][0] + orient["y_span"][1]) > 0.05 * y_half:
        problems.append(f"not symmetric in y: mean {orient['y_mean']:.4f}, span {orient['y_span']}")
    if abs(orient["z_min"]) > 0.02:
        problems.append(f"the soles are not on z=0: z_min {orient['z_min']:.4f}")
    if problems:
        for pr in problems:
            say("frame check failed:", pr)
        return 1
    mats = rs.read_mtl(mtl_path)
    unresolved = [(m.name, f) for m in mats.values() for f in (m.map_kd, m.map_bump)
                  if f and not (out / f).is_file()]
    if unresolved:
        say(f"MTL maps that do not resolve beside the OBJ: {unresolved}")
        return 1

    # -- fit against the anthropometry primitives ------------------------------------------
    # The same assignment build_dummy.py --skin will make. A region whose vertices sit far
    # from its primitive would render a limb where no collider is.
    anthro_regions = json.loads(json.dumps(spec["anthropometry_regions"]))
    prims = rs.primitives_from_anthropometry({"regions": anthro_regions})
    idx = rs.assign_primitives(mesh.V, prims)
    # The anthropometry has no foot region (Table A.2 stops at the lower legs), so the
    # shoes below the lowest collider have nothing to fit to; they are assigned to BR-12
    # by build_dummy all the same but must not pull its centroid off the shin here.
    floor = min(rs.bottom_of(p) for p in prims)
    unbacked = int((mesh.V[:, 2] < floor).sum())
    fit_rows = []
    worst = []
    for i, prim in enumerate(prims):
        mine = idx == i
        # the centroid is compared where the collider is: alongside the primitive's own
        # z-extent, so a hand hanging past the hand box or a shoe under the shin does
        # not stand in for the limb the collider represents
        alongside = mine & (mesh.V[:, 2] >= rs.bottom_of(prim)) & (mesh.V[:, 2] <= rs.top_of(prim))
        v = mesh.V[alongside]
        if len(v) == 0:
            worst.append(f"{prim.name}: no vertices alongside it")
            continue
        off = v.mean(axis=0) - prim.center
        d = rs.distance(mesh.V[mine], prim)
        row = {"prim": prim.name, "code": prim.code, "vertices": int(mine.sum()), "vertices_alongside": int(len(v)),
               "centroid_offset_m": [round(float(x), 4) for x in off],
               "lateral_offset_m": round(float(math.hypot(off[0], off[1])), 4),
               "off_surface_p50_m": round(float(np.percentile(d, 50)), 4),
               "off_surface_p95_m": round(float(np.percentile(d, 95)), 4),
               "off_surface_max_m": round(float(d.max()), 4)}
        fit_rows.append(row)
        if row["lateral_offset_m"] > float(spec["fit"]["max_lateral_offset_m"]):
            worst.append(f"{prim.name}: centroid {row['lateral_offset_m']:.3f} m off its primitive laterally")
    if worst:
        say(f"fit check failed (limit {spec['fit']['max_lateral_offset_m']} m): " + "; ".join(worst))
        for r in fit_rows:
            say(f"  {r['prim']:14s} n={r['vertices']:5d} alongside={r['vertices_alongside']:5d} "
                f"centroid {r['centroid_offset_m']} lateral {r['lateral_offset_m']:.3f} off-surface p95 {r['off_surface_p95_m']:.3f}")
        return 1
    say("fit: worst lateral centroid offset {:.3f} m · off-surface p95 max {:.3f} m · {} vertices below the "
        "lowest collider ({:.3f} m) have no region".format(
            max(r["lateral_offset_m"] for r in fit_rows), max(r["off_surface_p95_m"] for r in fit_rows),
            unbacked, floor))

    # -- lock ---------------------------------------------------------------------------------
    rows_lock = []
    for name, dst in sorted(exported.items()):
        src = images.get(name)
        if src is None:
            say(f"exported texture {name} was not referenced by any material image node — cannot state its source")
            return 1
        try:
            rel = src.relative_to(data_dir)
            source = f"pack:{rel.as_posix()}"
        except ValueError:
            say(f"texture {name} comes from outside the system asset pack ({src}) — its licence is not the pack's")
            return 1
        rows_lock.append({"name": name, "sha256": sha256_of(dst), "bytes": dst.stat().st_size,
                          "source": source, "license": "CC0-1.0"})
    # the pack files the figure was built from: each asset's .mhclo/.mhmat, the geometry
    # and material the .mhclo names — a swapped geometry keeps its label
    pack_files: list[Path] = []
    for (subdir, label), full in paths.items():
        f = Path(full).resolve()
        pack_files.append(f)
        if f.suffix == ".mhclo":
            try:
                pack_files.extend(mhclo_refs(f))
            except ValueError as e:
                say(f"asset {subdir}/{label}: {e}")
                return 1
    seen: set[Path] = set()
    for f in pack_files:
        if f in seen:
            continue
        seen.add(f)
        try:
            rel = f.relative_to(data_dir).as_posix()
        except ValueError:
            say(f"asset file {f} lies outside the system asset directory — its licence is not the pack's")
            return 1
        rows_lock.append({"name": rel, "sha256": sha256_of(f), "bytes": f.stat().st_size,
                          "source": f"pack:{rel}", "license": "CC0-1.0"})
    n_tex = sum(1 for r in rows_lock if is_texture_row(r))
    n_pack = len(rows_lock) - n_tex

    def write_lock(verb: str) -> int:
        # only after every row was matched byte for byte against the pinned pack
        if not args.pack:
            say(f"{lock_path.relative_to(REPO_ROOT)} would be {verb} — pass --pack <makehuman_system_assets_cc0.zip> "
                "so every row is verified against the pinned pack first")
            say(f"  {spec['source']['system_assets']['url']}")
            return 3
        pack = Path(args.pack)
        if not pack.is_file():
            say(f"pack zip not found: {pack}")
            return 3
        problems = verify_against_pack(pack, spec["source"]["system_assets"]["sha256"], rows_lock)
        if problems:
            say(f"{len(problems)} lock rows do not match the pack {pack.name}:")
            for pr in problems:
                say("  ", pr)
            return 1
        lock_path.write_text(format_lock(rows_lock), encoding="utf-8")
        say(f"{verb} {lock_path.relative_to(REPO_ROOT)} ({n_tex} textures · {n_pack} pack files, all matched "
            f"against {pack.name}) — commit it")
        return 0

    key = lambda r: (r["name"], r["sha256"], r["bytes"], r["source"])  # noqa: E731
    if lock_path.is_file():
        old = parse_lock(lock_path)
        if sorted(map(key, old)) != sorted(map(key, rows_lock)):
            if not args.update_lock:
                say(f"textures or pack files differ from {lock_path.relative_to(REPO_ROOT)} — re-run with "
                    "--update-lock --pack <zip> if intended")
                for r in sorted(set(map(key, old)) ^ set(map(key, rows_lock))):
                    say("  ", r)
                return 1
            rc = write_lock("updated")
            if rc:
                return rc
        else:
            say(f"textures and pack files match {lock_path.relative_to(REPO_ROOT)} ({n_tex} textures · {n_pack} pack files)")
    else:
        rc = write_lock("wrote")
        if rc:
            return rc

    # -- manifest -----------------------------------------------------------------------------
    per_object = []
    for i, name in enumerate(mesh.objects):
        sel = mesh.FO == i
        per_object.append({"name": name, "faces": int(sel.sum()),
                           "vertices": int(len(np.unique(mesh.F[sel]))),
                           "materials": sorted({mesh.materials[m] for m in np.unique(mesh.FM[sel]) if m >= 0})})
    manifest = {
        "skin_version": spec["skin_version"],
        "spec": {"path": spec["spec_path"], "sha256": spec["spec_sha256"]},
        "anthropometry": {"path": str(ANTHRO.relative_to(REPO_ROOT)), "version": spec["anthropometry_version"],
                          "sha256": spec["anthropometry_sha256"]},
        "generated_with": {"tool": spec["source"]["tool"], "mpfb": mpfb_version,
                           "blender": bpy.app.version_string, "material_type": spec["material_type"]},
        "macro": macro_for(height),
        "stature": {"target_m": target, "tolerance_m": tol_m, "bare_m": stature_bare, "dressed_body_m": stature_dressed,
                    "height_macro": height, "search": [{"height": h, "stature_m": s} for h, s in rows]},
        "frame": {**spec["frame"], "yaw_deg": spec["export"]["yaw_deg"], **orient},
        "pose": spec["pose"],
        "fit": {"limit": spec["fit"], "lowest_collider_m": floor, "vertices_below_lowest_collider": unbacked,
                "regions": fit_rows},
        "objects": per_object,
        "totals": {"vertices": int(len(mesh.V)), "faces": int(len(mesh.F))},
        "assets": [{"subdir": s, "label": l, "type": next((a["type"] for a in spec["clothes"] if a["label"] == l), "Skin"),
                    "file": f"pack:{Path(paths[(s, l)]).resolve().relative_to(data_dir).as_posix()}", "license": "CC0-1.0"}
                   for (s, l) in paths],
        "files": {"obj": {"path": obj_path.name, "sha256": sha256_of(obj_path), "bytes": obj_path.stat().st_size},
                  "mtl": {"path": mtl_path.name, "sha256": sha256_of(mtl_path), "bytes": mtl_path.stat().st_size},
                  "textures": [r for r in rows_lock if is_texture_row(r)],
                  "pack_files": [r for r in rows_lock if not is_texture_row(r)]},
    }
    manifest_path = out / Path(spec["export"]["manifest"]).name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    say(f"wrote {obj_path.name} ({manifest['totals']['vertices']} vertices · {manifest['totals']['faces']} faces · "
        f"{len(mesh.objects)} objects · {n_tex} textures · {n_pack} pack files)")
    say(f"manifest {manifest_path}")
    return 0


def _blender_stage(argv: list[str]) -> int:
    """Run the Blender stage; on any failure discard the export so build_dummy.py has
    nothing stale or half-checked to consume (a run that failed its lock check left
    the previous manifest beside a new OBJ)."""
    try:
        rc = _blender_run(argv)
    except BaseException as e:                     # noqa: BLE001 — discard, then let it propagate
        # An exception before the in-run discard (asset loading, the height search) would
        # otherwise leave the previous export consumable.
        print(f"SKIN Blender stage raised {type(e).__name__}: {e}", flush=True)
        rc = 1
        _discard_export(argv)
        raise
    if rc != 0:
        _discard_export(argv)
    return rc


def _discard_export(argv: list[str]) -> None:
    if True:
        p = argparse.ArgumentParser()
        p.add_argument("--resolved", required=True)
        p.add_argument("--out", required=True)
        known, _ = p.parse_known_args(argv)
        try:
            spec = json.loads(Path(known.resolved).read_text(encoding="utf-8"))
            out = Path(known.out)
            for name in (Path(spec["export"]["obj"]).name, Path(spec["export"]["obj"]).with_suffix(".mtl").name,
                         Path(spec["export"]["manifest"]).name):
                (out / name).unlink(missing_ok=True)
            shutil.rmtree(out / "textures", ignore_errors=True)
            print("SKIN export discarded — nothing for build_dummy.py to consume until a run passes", flush=True)
        except (OSError, ValueError, KeyError) as e:      # noqa: BLE001 — never mask the real failure
            print(f"SKIN could not discard the export: {e}", flush=True)


# ---- driver ---------------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MakeHuman(MPFB2) 으로 더미 시각 메시를 만든다 (Blender 헤드리스)")
    p.add_argument("--spec", default=str(SPEC), help="시각 메시 정본 (기본: skin_spec.yaml)")
    p.add_argument("--out", default=str(OUT), help="산출 디렉터리 (기본: generation/dummy/skin/build)")
    p.add_argument("--blender", default=os.environ.get("BLENDER", "blender"), help="Blender 실행 파일")
    p.add_argument("--update-lock", action="store_true", help="textures.lock 을 이번 산출물로 갱신한다 (--pack 필요)")
    p.add_argument("--pack", default=os.environ.get("MAKEHUMAN_ASSET_PACK"),
                   help="MakeHuman 시스템 자산 팩 zip. lock 을 처음 쓰거나 갱신할 때 모든 행을 이 팩과 바이트로 "
                        "대조한다 (기본: 환경변수 MAKEHUMAN_ASSET_PACK)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        spec = load_spec(Path(args.spec))
        resolved = resolve(spec, Path(args.spec))
    except (ValueError, OSError) as e:
        print(f"명세 오류: {e}", file=sys.stderr)
        return 3
    blender = shutil.which(args.blender)
    if blender is None:
        print(f"Blender 를 찾을 수 없습니다: {args.blender!r} (--blender 또는 BLENDER=)", file=sys.stderr)
        return 3
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    resolved_path = out / "resolved_spec.json"
    resolved_path.write_text(json.dumps(resolved, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    cmd = [blender, "-b", "--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--",
           "--resolved", str(resolved_path), "--out", str(out)]
    if args.update_lock:
        cmd.append("--update-lock")
    if args.pack:
        cmd += ["--pack", str(Path(args.pack).resolve())]
    log = out / "make_skin.log"
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", cwd=str(REPO_ROOT))
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            if line.startswith("SKIN "):
                print(line[5:], end="", flush=True)
        rc = proc.wait()
    if rc != 0:
        print(f"Blender 단계 실패 (exit {rc}) — 전체 로그: {log}", file=sys.stderr)
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        print("\n".join(tail), file=sys.stderr)
        return rc if rc in (1, 2, 3) else 1
    manifest = out / Path(spec["export"]["manifest"]).name
    if not manifest.is_file():
        print(f"Blender 단계가 0 으로 끝났는데 매니페스트가 없습니다: {manifest}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if "--resolved" in sys.argv and "bpy" in sys.modules:
        raise SystemExit(_blender_stage(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]))
    raise SystemExit(main())
