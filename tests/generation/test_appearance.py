"""The look axes of the new scenarios: the canon block, its checks and the per-row draw."""
import copy
import hashlib

import numpy as np
import pytest

from generation import appearance, conditions

NEW = ("S1e", "S1f", "S1g")


def test_the_seed_is_the_rows_own_hash():
    digest = hashlib.sha256(b"S1f:appearance:7").digest()
    assert appearance.appearance_seed("S1f", 7) == int.from_bytes(digest[:8], "big")


@pytest.mark.parametrize("sid", NEW)
def test_a_row_draws_every_axis_in_the_written_order(sid):
    entry = conditions.scenario_entry(sid)
    drawn = appearance.draw_appearance(entry, 3)
    assert list(drawn) == list(entry["appearance"]["axes"])
    rng = np.random.default_rng(appearance.appearance_seed(sid, 3))      # the draw, step by step
    for name, node in entry["appearance"]["axes"].items():
        if "choice" in node:
            assert drawn[name] == list(node["choice"])[int(rng.integers(len(node["choice"])))]
        else:
            assert drawn[name] == float(rng.uniform(*node["range"]))


def test_the_draw_does_not_depend_on_the_conditions_table():
    entry = copy.deepcopy(conditions.scenario_entry("S1e"))
    before = appearance.draw_appearance(entry, 60)
    entry["conditions"]["rows"] = 400
    entry["conditions"].pop("grid")
    entry["episodes"] = 400
    entry["conditions"]["columns"]["turn_radius_m"]["range"] = [1.5, 1.75]
    assert appearance.draw_appearance(entry, 60) == before


def test_the_three_scenarios_declare_the_same_axes():
    blocks = [conditions.scenario_entry(s)["appearance"] for s in NEW]
    assert blocks[0]["axes"] == blocks[1]["axes"] == blocks[2]["axes"]


def test_every_value_turns_up_over_the_tables_rows():
    entry = conditions.scenario_entry("S1g")
    draws = [appearance.draw_appearance(entry, row) for row in range(99)]
    for name, node in entry["appearance"]["axes"].items():
        if "choice" in node:
            assert {d[name] for d in draws} == set(node["choice"])
        else:
            lo, hi = node["range"]
            assert all(lo <= d[name] < hi for d in draws)


def test_a_scenario_without_a_block_draws_nothing():
    # S1d has no conditions table, so scenario_entry refuses it; take its entry from the document.
    doc = conditions._scenarios_doc()
    entry = next(e for e in doc["scenarios"] if e["scenario_id"] == "S1d")
    assert appearance.draw_appearance(entry, 0) is None


@pytest.mark.parametrize("change, message", [
    (lambda a: a["axes"].pop("vest_color"), "vest_color"),
    (lambda a: a["axes"].update(hair_color={"choice": {"x": None}}), "hair_color"),
    (lambda a: a["axes"]["shirt_color"].update(choice={}), "shirt_color"),
    (lambda a: a["axes"]["box_color"]["choice"].update(odd=[1.0, 0.5]), "box_color"),
    (lambda a: a["axes"]["box_color"]["choice"].update(odd=[True, 0.5, 0.5]), "box_color"),
    (lambda a: a["axes"]["beam_color"]["choice"].update(odd=[-0.1, 0.5, 0.5]), "beam_color"),
    (lambda a: a["axes"]["light_scale"].update(range=[1.3, 0.6]), "light_scale"),
    (lambda a: a["axes"]["floor_wall_scale"].update(range=[0.0, 1.1]), "floor_wall_scale"),
    (lambda a: a["axes"]["light_temperature_k"].update(range=[500.0, 6500.0]), "light_temperature_k"),
    (lambda a: a["axes"]["light_temperature_k"].update(range=[3500.0, 12000.0]), "light_temperature_k"),
    (lambda a: a.update(extra=1), "extra"),
    (lambda a: a.update(determined_by="standard"), "determined_by"),
])
def test_a_bad_block_is_refused_naming_the_axis(change, message):
    entry = copy.deepcopy(conditions.scenario_entry("S1e"))
    change(entry["appearance"])
    with pytest.raises(ValueError, match=message):
        appearance.load_appearance(entry)


def test_a_block_on_a_scenario_without_a_conditions_table_is_refused_naming_it():
    entry = copy.deepcopy(conditions.scenario_entry("S1e"))
    entry.pop("conditions")
    with pytest.raises(ValueError, match="S1e appearance: 조건표가 없는"):
        appearance.load_appearance(entry)


# --- the mapping from a draw to shader and light edits -------------------------------------------------------

def _edits_for(sid, drawn):
    spec = appearance.load_appearance(conditions.scenario_entry(sid))
    return appearance.appearance_edits(spec, drawn)


def _drawn_with(sid, **chosen):
    """A draw whose every colour is 'as built' (None) except the named axes; range axes at their low end."""
    spec = appearance.load_appearance(conditions.scenario_entry(sid))
    out = {}
    for axis in spec.axes:
        if axis.choices is not None:
            out[axis.name] = next(n for n, v in axis.choices.items() if v is None)
        else:
            out[axis.name] = axis.lo
    out.update(chosen)
    return out


def _the_non_null(sid, axis_name):
    spec = appearance.load_appearance(conditions.scenario_entry(sid))
    axis = next(a for a in spec.axes if a.name == axis_name)
    return next((n, v) for n, v in axis.choices.items() if v is not None)


def _select(edits, shader):
    return [e for e in edits if e.shader == shader]


@pytest.mark.parametrize("sid", NEW)
def test_a_colour_drawn_as_built_edits_nothing(sid):
    edits, light = _edits_for(sid, _drawn_with(sid))
    # Only the range axes edit: the floor and wall tints, the lens emission and the light.
    assert {e.shader for e in edits} == {"Looks/concrete_floor/Shader", "Looks/concrete_wall/Shader",
                                         "Looks/wall_cladding/Shader", "Looks/fixture_lens/Shader"}
    assert all(e.op == "scale" for e in edits)


@pytest.mark.parametrize("sid", NEW)
def test_the_colour_axes_edit_the_shaders_the_table_names(sid):
    from generation.dummy import build_dummy
    table = {
        "vest_color": ("dummy", ["Looks/vest_yellow/Shader"], [("diffuse_color_constant", None)]),
        "shirt_color": ("dummy", [f"Looks/{build_dummy.SHIRT_MATERIAL}/Shader"], [("diffuse_tint", None)]),
        "trousers_color": ("dummy", [f"Looks/{build_dummy.TROUSERS_MATERIAL}/Shader"],
                           [("albedo_desaturation", 1.0), ("diffuse_tint", None)]),
        "box_color": ("box", ["Looks/cardboard/Shader"], [("diffuse_color_constant", None)]),
        "upright_color": ("aisle", ["Looks/steel_blue/Shader"],
                          [("albedo_desaturation", 1.0), ("diffuse_tint", None)]),
        "beam_color": ("aisle", ["Looks/steel_orange/Shader"],
                       [("albedo_desaturation", 1.0), ("diffuse_tint", None)]),
    }
    for axis_name, (target, shaders, inputs) in table.items():
        name, rgb = _the_non_null(sid, axis_name)
        edits, _ = _edits_for(sid, _drawn_with(sid, **{axis_name: name}))
        got = _select(edits, shaders[0])
        assert [(e.name, e.op, e.target) for e in got] == [(n, "set", target) for n, _ in inputs]
        for e, (_, fixed) in zip(got, inputs):
            assert e.value == (fixed if fixed is not None else rgb)


@pytest.mark.parametrize("sid", NEW)
def test_the_range_axes_become_scales_and_the_light_edit(sid):
    entry = conditions.scenario_entry(sid)
    lo_ls, hi_ls = entry["appearance"]["axes"]["light_scale"]["range"]
    ls, temp, fw = (lo_ls + hi_ls) / 2, 5000.0, 1.0
    edits, light = _edits_for(sid, _drawn_with(sid, light_scale=ls, light_temperature_k=temp, floor_wall_scale=fw))
    assert light == appearance.LightEdit(scale=ls, temperature_k=temp)
    for shader in ("Looks/concrete_floor/Shader", "Looks/concrete_wall/Shader", "Looks/wall_cladding/Shader"):
        (e,) = _select(edits, shader)
        assert (e.target, e.name, e.op, e.value) == ("aisle", "diffuse_tint", "scale", fw)
    (lens,) = _select(edits, "Looks/fixture_lens/Shader")
    assert (lens.target, lens.name, lens.op, lens.value) == ("aisle", "emissive_intensity", "scale", ls)


def test_the_same_draw_gives_the_same_ordered_edits():
    entry = conditions.scenario_entry("S1f")
    spec = appearance.load_appearance(entry)
    drawn = appearance.draw_appearance(entry, 11)
    assert appearance.appearance_edits(spec, drawn) == appearance.appearance_edits(spec, dict(drawn))
    shaders = [e.shader for e in appearance.appearance_edits(spec, drawn)[0]]
    # The order follows the canon's written axes, then the floor/wall and lens scales.
    assert shaders[-4:] == [*appearance._FLOOR_WALL_SHADERS, appearance._LENS_SHADER]
    again = [e.shader for e in appearance.appearance_edits(spec, appearance.draw_appearance(entry, 11))[0]]
    assert shaders == again


def test_a_draw_that_does_not_match_the_spec_is_refused():
    spec = appearance.load_appearance(conditions.scenario_entry("S1e"))
    drawn = appearance.draw_appearance(conditions.scenario_entry("S1e"), 0)
    with pytest.raises(ValueError, match="vest_color"):
        appearance.appearance_edits(spec, {**drawn, "vest_color": "not_a_colour"})
    broken = dict(drawn)
    broken.pop("beam_color")
    with pytest.raises(ValueError, match="beam_color"):
        appearance.appearance_edits(spec, broken)


# ---- the run wires the look: draw before the boot, apply once when rendering ----

from pathlib import Path as _Path

RUN_SOURCE = (_Path(__file__).resolve().parents[2] / "generation/envs/run_s1a.py").read_text(encoding="utf-8")
RENDER_GATE = "    if (CAMERAS_ON or args.video or args.gui) and APPEARANCE_DRAWN is not None:\n"


def test_the_look_is_drawn_before_the_boot_from_the_runs_row():
    draw = RUN_SOURCE.index("appearance_canon.draw_appearance(")
    assert draw < RUN_SOURCE.index("launch_app(")
    assert "draw_appearance(conditions.scenario_entry(args.scenario), args.seq)" in RUN_SOURCE
    assert RUN_SOURCE.index("conditions.draw_row(") < draw


def test_the_look_is_applied_under_the_render_condition_after_the_env_exists():
    gate = RUN_SOURCE.index(RENDER_GATE)
    assert gate > RUN_SOURCE.index("skin_blend = SkinBlend.from_env(base_env)")
    assert gate > RUN_SOURCE.index("base_env = env.unwrapped")
    assert "apply_appearance(" in RUN_SOURCE[gate:]
    assert RUN_SOURCE.count("apply_appearance(") == 1


def test_the_look_is_applied_before_the_episode_loop_and_not_inside_it():
    loop = RUN_SOURCE.index("    for offset in range(args.episodes):")
    assert RUN_SOURCE.index("apply_appearance(") < loop < RUN_SOURCE.index("env.reset()")
    assert "apply_appearance" not in RUN_SOURCE[loop:]
    assert "claim_single_apply" not in RUN_SOURCE[loop:]


def test_every_control_flag_needs_an_out_directory():
    guard = RUN_SOURCE.index("if (args.no_cameras or args.no_skin or args.no_appearance) and args.out is None:")
    message = RUN_SOURCE[guard:guard + 400]
    assert "--no-cameras · --no-skin · --no-appearance 는 대조군 플래그라 --out" in message


def test_no_appearance_draws_nothing_to_apply_and_records_null():
    assert '"--no-appearance"' in RUN_SOURCE
    assert "if APPEARANCE_SPEC is not None and not args.no_appearance:" in RUN_SOURCE
    # the apply is gated on the draw, which --no-appearance leaves None; the recorder then gets appearance=None
    assert RENDER_GATE in RUN_SOURCE
    assert '**({} if APPEARANCE_SPEC is None else {"appearance": APPEARANCE_DRAWN})' in RUN_SOURCE


def test_a_second_apply_in_one_process_is_refused():
    claim = RUN_SOURCE.index("appearance_canon.claim_single_apply()")
    assert claim < RUN_SOURCE.index("apply_appearance(", claim)
    code = (_Path(__file__).resolve().parents[2] / "generation/appearance.py")
    import importlib
    import sys
    spec = importlib.util.spec_from_file_location("appearance_fresh", code)
    fresh = importlib.util.module_from_spec(spec)
    sys.modules["appearance_fresh"] = fresh
    try:
        spec.loader.exec_module(fresh)
        fresh.claim_single_apply()
        with pytest.raises(RuntimeError, match="한 번만"):
            fresh.claim_single_apply()
    finally:
        del sys.modules["appearance_fresh"]


# ---- the names the edits target exist in the builders' sources (pxr-free) ----

import yaml as _yaml

_ROOT = _Path(__file__).resolve().parents[2]


def _targeted_materials(target):
    names = {}
    for axis, (tgt, shader, _inputs) in appearance._COLOR_EDITS.items():
        if tgt == target:
            names[axis] = shader.split("/")[1]
    return names


def test_the_material_names_the_edits_target_exist_in_the_builders():
    aisle = _yaml.safe_load((_ROOT / "generation/scene/logistics_aisle.yaml").read_text(encoding="utf-8"))
    aisle_materials = set(aisle["materials"])
    targeted = set(_targeted_materials("aisle").values())
    targeted |= {s.split("/")[1] for s in (*appearance._FLOOR_WALL_SHADERS, appearance._LENS_SHADER)}
    assert targeted == {"steel_blue", "steel_orange", "concrete_floor", "concrete_wall", "wall_cladding",
                        "fixture_lens"}
    assert targeted <= aisle_materials, sorted(targeted - aisle_materials)
    assert aisle["lights"]["fixture_material"] == "fixture_lens"       # the lens the light scale reaches

    vest = _yaml.safe_load((_ROOT / "generation/dummy/vest.yaml").read_text(encoding="utf-8"))
    assert _targeted_materials("dummy")["vest_color"] in vest["materials"]

    box_source = (_ROOT / "generation/scene/carried_box.py").read_text(encoding="utf-8")
    assert f'_omnipbr(stage, "{_targeted_materials("box")["box_color"]}"' in box_source

    # the shirt and trousers materials are named once, here, and the dummy build splits the worksuit into them
    build_source = (_ROOT / "generation/dummy/build_dummy.py").read_text(encoding="utf-8")
    assert "SHIRT_MATERIAL" in build_source and "TROUSERS_MATERIAL" in build_source
    assert _targeted_materials("dummy")["shirt_color"] == appearance.SHIRT_MATERIAL
    assert _targeted_materials("dummy")["trousers_color"] == appearance.TROUSERS_MATERIAL


def test_the_light_scale_reaches_only_lights_under_the_warehouse_root():
    # an inherited sky dome would sit outside the warehouse and keep its brightness
    cfg = (_ROOT / "generation/envs/s1d/s1d_env_cfg.py").read_text(encoding="utf-8")
    assert "self.scene.sky_light = None" in cfg
