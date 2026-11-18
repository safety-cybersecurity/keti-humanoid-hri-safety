#!/usr/bin/env python3
"""Look at the Stage before anything runs: one picture from each camera it holds. Run with Isaac's python:

    PYTHONPATH= ~/isaacsim/python.sh generation/scene/look_stage.py --out <dir> [--sheet <jpg>]

check_stage.lay_release lays in a temporary directory what a release that ships the Stage holds under assets/,
and the Stage opens there headless -- the same files, at the same pinned bytes, a reader of the release opens.
Every camera of build_stage's plan (the robot's cameras on their links and each site's third-person and CCTV
cameras) gets a WIDTH x HEIGHT render product with an RGB annotator; after WARMUP_STEPS renders each picture is
saved as <dir>/<prim path with "/" as "__">.png and all of them are tiled, each labelled by its prim path, into
one JPEG (--sheet, default <dir>/stage-cameras.jpg).

It is a tool to look with, not a check. It says only that the Stage opened, that the cameras under its root are
the plan's, and that no picture is all black (every pixel 0, as the recorder counts a blank frame).
Exit 0 when so, 1 otherwise.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generation.isaac_boot import launch_simulation_app  # noqa: E402

WIDTH, HEIGHT = 640, 480
# Renders before the pictures are read: the RTX renderer accumulates over frames and streams textures in.
WARMUP_STEPS = 8
RT_SUBFRAMES = 4
# The sheet: tiles at this fraction of a picture, this many to a row.
TILE_SCALE = 0.5
TILE_COLUMNS = 4
LABEL_HEIGHT = 18


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="directory for one PNG per camera")
    ap.add_argument("--sheet", type=Path, default=None, help="the tiled JPEG (default <out>/stage-cameras.jpg)")
    return ap.parse_args(argv)


def png_name(prim_path: str) -> str:
    return prim_path.strip("/").replace("/", "__") + ".png"


def tile(pictures: dict[str, "np.ndarray"], sheet: Path) -> None:
    """Tile the pictures in plan order, each labelled by its prim path, into one JPEG."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    w, h = int(WIDTH * TILE_SCALE), int(HEIGHT * TILE_SCALE)
    rows = math.ceil(len(pictures) / TILE_COLUMNS)
    canvas = Image.new("RGB", (w * TILE_COLUMNS, (h + LABEL_HEIGHT) * rows), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=13)
    for i, (path, rgb) in enumerate(pictures.items()):
        x, y = (i % TILE_COLUMNS) * w, (i // TILE_COLUMNS) * (h + LABEL_HEIGHT)
        canvas.paste(Image.fromarray(np.asarray(rgb, np.uint8)).resize((w, h), Image.LANCZOS), (x, y + LABEL_HEIGHT))
        draw.text((x + 4, y + 2), path, fill=(255, 230, 90), font=font)
    sheet.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(sheet, quality=90)


def look(out: Path, sheet: Path) -> list[str]:
    """Open the Stage inside a laid-out release, render every planned camera once, save the pictures; the
    problems found ([] when none)."""
    import numpy as np
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import UsdGeom

    from generation.scene import build_stage, check_stage

    app = omni.kit.app.get_app()
    plan = build_stage.canon_plan()
    planned = [path for path, p in plan.items() if p.kind == "camera"]
    with tempfile.TemporaryDirectory(prefix="stage-look-") as tmp:
        stage_path = check_stage.lay_release(REPO, Path(tmp) / "release")
        context = omni.usd.get_context()
        if not context.open_stage(str(stage_path)):
            return [f"{stage_path.name} 를 열 수 없습니다"]
        while context.get_stage_loading_status()[2] > 0:
            app.update()
        stage = context.get_stage()
        held = [str(p.GetPath()) for p in stage.Traverse()
                if p.IsA(UsdGeom.Camera) and str(p.GetPath()).startswith(build_stage.ROOT + "/")]
        problems = []
        if sorted(held) != sorted(planned):
            problems.append(f"Stage 의 카메라 {sorted(held)} — 계획은 {sorted(planned)}")
        print(f"cameras: {len(held)} in the Stage, {len(planned)} planned")
        rep.orchestrator.set_capture_on_play(False)
        annotators = {}
        for path in planned:
            if not stage.GetPrimAtPath(path).IsA(UsdGeom.Camera):
                continue                                  # said above
            product = rep.create.render_product(path, (WIDTH, HEIGHT))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([product])
            annotators[path] = annotator
        for _ in range(WARMUP_STEPS):
            rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
        out.mkdir(parents=True, exist_ok=True)
        pictures = {}
        for path, annotator in annotators.items():
            data = np.asarray(annotator.get_data(), np.uint8)
            if data.size == 0 or data.size % (HEIGHT * WIDTH):
                problems.append(f"{path}: 그림이 나오지 않았습니다 (값 {data.size} 개)")
                continue
            rgb = data.reshape(HEIGHT, WIDTH, -1)[:, :, :3]
            pictures[path] = rgb
            Image.fromarray(rgb).save(out / png_name(path))
            luma = float((rgb @ np.array([0.2126, 0.7152, 0.0722])).mean())
            print(f"{path}: mean luma {luma:.1f} · max {int(rgb.max())}")
            if int(rgb.max()) == 0:
                problems.append(f"{path}: 그림이 모두 검습니다")
        rep.orchestrator.wait_until_complete()
        for annotator in annotators.values():
            annotator.detach()
        context.close_stage()
    tile(pictures, sheet)
    print(f"wrote {len(pictures)} pictures under {out} and the sheet {sheet}")
    return problems


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    sheet = args.sheet or args.out / "stage-cameras.jpg"
    # Boot through the helper: the try/finally below arms only once the app exists (generation/isaac_boot.py).
    app = launch_simulation_app({"headless": True})
    code = 1
    try:
        problems = look(args.out, sheet)
        for p in problems:
            print(f"PROBLEM {p}")
        print("OK" if not problems else f"{len(problems)} problem(s)")
        code = 0 if not problems else 1
    except Exception:  # noqa: BLE001 -- printed, and the exit code says it
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        app.close(exit_code=code)
    return code


if __name__ == "__main__":
    sys.exit(main())
