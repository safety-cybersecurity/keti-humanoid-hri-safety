"""Label the bare dummy's body regions for the head camera when S1d runs it without the skin (and the
jointed dummy's feet, which are outside the regions, with the class its skin gives them).

The skinned dummy carries a class label per region on its skin meshes; v1 carries none, and its
published bytes stay as they are. Unlabelled, a capsule dummy is background in the head mask and
never gets a box -- yet the capsule arm is what the detection-rate comparison needs.
Each region prim already names its code in keti:bodyRegion, so the spawn
wrapper copies that into a class label as the asset is referenced into the stage, before any
physics view exists (labelling later invalidates the views). pxr is imported inside,
so the module loads without Isaac.
"""
from __future__ import annotations

import functools


def label_regions(root) -> int:
    """Give every prim under root that carries keti:bodyRegion that code as its class label, and every
    ground-contact body (keti:groundContact -- the jointed dummy's feet, which have no region) the class
    the skin gives its feet (the joint spec's semantic_class_feet). The rigid dummy has no such body."""
    from pxr import Usd, UsdSemantics

    labelled, foot_class = 0, None
    for prim in Usd.PrimRange(root):
        attr = prim.GetAttribute("keti:bodyRegion")
        if attr and attr.HasAuthoredValue():
            label = str(attr.Get())
        else:
            flag = prim.GetAttribute("keti:groundContact")
            if not (flag and flag.HasAuthoredValue() and flag.Get()):
                continue
            if foot_class is None:
                from generation.dummy import joints_spec as js

                foot_class = js.load().semantic_class_feet
            label = foot_class
        UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr().Set([label])
        labelled += 1
    return labelled


def labelled(spawn):
    """Wrap an undecorated spawn function (the one Isaac Lab's clone decorator wraps) so the
    spawned dummy's regions are labelled before any copy is made of it."""

    @functools.wraps(spawn)
    def wrapped(prim_path, cfg, *args, **kwargs):
        prim = spawn(prim_path, cfg, *args, **kwargs)
        if label_regions(prim) == 0:
            raise ValueError(f"{prim_path}: no keti:bodyRegion prims to label")
        return prim

    return wrapped
