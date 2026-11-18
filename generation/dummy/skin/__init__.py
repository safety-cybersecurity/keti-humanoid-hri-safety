"""Visual mesh for the ISO/TS 15066 dummy: spec (yaml) -> MakeHuman via MPFB2 -> OBJ + textures.

`region_split` is pure numpy and runs in CI; `make_skin.py` needs Blender with
the MPFB2 extension (it drives Blender headless from an ordinary Python), and
the USD assembly lives in `generation/dummy/build_dummy.py --skin` (Isaac python).
"""
