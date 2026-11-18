"""meta/cameras.json and its loader. Isaac-free: the derivations are checked here,
the engine's own reading of them in generation/envs/s1d/check_s1d_scene.py."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from generation import cameras, conditions, convert_lerobot, sites
from generation.scene import layout
from validator.make_fixture import PACK_NAME, canon_with_pack, media_canon

REPO = Path(__file__).resolve().parents[2]
CAMERAS = REPO / "meta/cameras.json"
# generation/envs/s1d/s1d_env_cfg.py SCENE_SPEC -- that module imports Isaac, so the path is spelt here
# as tests/generation/test_aisle_guard.py spells it.
SCENE_SPEC = REPO / "generation/scene/logistics_aisle.yaml"
# Read-only here: a test that edits a canon loads its own copy.
CANON = cameras.load(CAMERAS)


def _body_regions() -> list[str]:
    # The region codes' canon; the camera canon copies them into its class table.
    schema = json.loads((REPO / "meta/safety_schema.json").read_text(encoding="utf-8"))
    return sorted(schema["body_regions"])


def _rotate_xyzw(q, v):
    # v' = v + 2w(u×v) + 2u×(u×v), u = (x, y, z) — written out here, not taken from the code under test
    x, y, z, w = q
    u = np.array([x, y, z], float)
    v = np.asarray(v, float)
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


def test_focal_length_reproduces_the_fov():
    f = cameras.focal_length_mm(69.4)
    assert 2 * math.degrees(math.atan(20.955 / (2 * f))) == pytest.approx(69.4)


def test_intrinsics_are_square_pixels_centred():
    # Every camera's, from its own size and fov in the canon; the formula is written out here, not
    # taken from cameras.intrinsics.
    for name, cam in cameras.load(CAMERAS)["cameras"].items():
        fx, fy, cx, cy = cameras.intrinsics(cam)
        assert fx == pytest.approx(fy) and cx == cam["width"] / 2 and cy == cam["height"] / 2, name
        assert fx == pytest.approx(cam["width"] / (2 * math.tan(math.radians(cam["fov_h_deg"]) / 2))), name


def test_follow_eye_target_adds_the_offsets_to_the_root():
    eye, target = cameras.follow_eye_target(np.array([10.0, 0.0, 1.0]), [-3.0, -3.0, 1.2], [1.5, 0.0, -0.4])
    assert eye == pytest.approx([7.0, -3.0, 2.2]) and target == pytest.approx([11.5, 0.0, 0.6])


def test_watch_follows_the_s1d_site():
    # S1d's third-person placement is the one generation/envs/s1a/watch.py's recordings use: world offsets
    # added to the robot root.
    from generation.envs.s1a import watch
    s1d = cameras.site_placement(cameras.load(CAMERAS)["cameras"]["third_person_rgb"], "S1d")
    assert s1d["frame"] == "world"
    assert tuple(s1d["eye_offset"]) == tuple(watch.FOLLOW_EYE)
    assert tuple(s1d["lookat_offset"]) == tuple(watch.FOLLOW_LOOKAT)


def test_no_channel_the_camera_refuses():
    # bounding_box_* raise in this Isaac Lab's Camera (UNSUPPORTED_TYPES) — the box comes from the mask
    cams = cameras.load(CAMERAS)
    assert all(not c.startswith("bounding_box") for cam in cams["cameras"].values() for c in cam["channels"])


def test_the_repository_canon_declares_no_pack():
    """Depth and masks are not shipped. The pack code stays and the canon
    switches it off by declaring none. With no pack there is nowhere for a depth channel to go, so
    no camera renders one: every camera records rgb, and bbox.camera the segmentation the box is
    computed from -- nothing a recording would render only to drop."""
    canon = cameras.load(CAMERAS)
    assert "packs" not in canon
    box = canon["bbox"]["camera"]
    renders = {name: sorted(cam["channels"]) for name, cam in canon["cameras"].items()}
    assert renders == {name: sorted({"rgb", "semantic_segmentation"} if name == box else {"rgb"})
                       for name in canon["cameras"]}


# The pack shape, kept for a canon that declares one: the fixture's literal (make_fixture.fixture_pack
# via media_canon(with_pack=True)), which the consumer tests and clean_media_pack use -- the
# repository canon's own pack is gone.

def test_pack_classes_cover_regions_robot_and_rack():
    classes = media_canon(with_pack=True)["packs"][PACK_NAME]["mask"]["classes"]
    regions = _body_regions()
    assert [classes[str(i)] for i in range(1, len(regions) + 1)] == regions
    assert classes["0"] == "background"
    assert classes[str(len(regions) + 1)] == "robot" and classes[str(len(regions) + 2)] == "rack"
    assert len(classes) == len(regions) + 3


def test_bbox_union_is_every_body_region():
    # The box's classes: the region codes' canon, all of them (CONTRIBUTING.md 「함께 고치는 곳」, the BR
    # row). With no pack this is the only place the camera canon names the regions.
    assert cameras.load(CAMERAS)["bbox"]["classes_union"] == _body_regions()


def test_head_offset_looks_along_the_torso_front_and_level():
    # head_rgb's optical axis is torso +x turned down by its mount rule's tilt, and the
    # rule says level (0). In the ROS optical frame the axis is +z and image-down is +y, so those two
    # land on the parent's (cos t, 0, -sin t) and (-sin t, 0, -cos t) -- at t = 0, +x and -z.
    head = cameras.load(CAMERAS)["cameras"]["head_rgb"]
    assert head["convention"] == "ros" and head["mount"]["tilt_down_deg"] == 0
    t = math.radians(head["mount"]["tilt_down_deg"])
    q = head["offset_rot_xyzw"]
    assert _rotate_xyzw(q, [0, 0, 1]) == pytest.approx([math.cos(t), 0, -math.sin(t)], abs=1e-12)
    assert _rotate_xyzw(q, [0, 1, 0]) == pytest.approx([-math.sin(t), 0, -math.cos(t)], abs=1e-12)


def test_load_refuses_a_camera_missing_a_required_key(tmp_path):
    canon = json.loads(CAMERAS.read_text(encoding="utf-8"))
    del canon["cameras"]["head_rgb"]["fov_h_deg"]
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(canon), encoding="utf-8")
    with pytest.raises(ValueError, match="head_rgb lacks fov_h_deg"):
        cameras.load(path)


# ---- bbox.camera: the segmentation camera the box is computed from ----------
# The box camera used to be the pack camera by implication; with the packs switched off in the
# canon it is named. Each case below is refused by one rule alone: the canon
# is built here rather than taken whole from the repository's, which declares no pack.

def _write(tmp_path, canon):
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(canon), encoding="utf-8")
    return path


def _without_packs():
    canon = json.loads(CAMERAS.read_text(encoding="utf-8"))
    canon.pop("packs", None)
    return canon


def _a_pack_on(camera: str) -> dict:
    # The pack shape the canon used to declare, written out: the repository canon has none.
    return {"camera": camera,
            "depth": {"dtype": "uint16", "unit": "mm", "invalid": 0, "source": "distance_to_image_plane"},
            "mask": {"dtype": "uint8", "classes": {"0": "background"}},
            "downsample": {"width": 640, "height": 480, "every_n_steps": 1}}


def test_a_canon_without_packs_loads(tmp_path):
    loaded = cameras.load(_write(tmp_path, _without_packs()))
    assert "packs" not in loaded and loaded["bbox"]["camera"] in loaded["cameras"]


@pytest.mark.parametrize("value", [None, "no_such_rgb", ["head_rgb"]], ids=["absent", "unknown", "list"])
def test_a_bbox_camera_that_is_not_a_camera_is_refused(tmp_path, value):
    canon = _without_packs()
    if value is None:
        del canon["bbox"]["camera"]
    else:
        canon["bbox"]["camera"] = value
    with pytest.raises(ValueError, match=r"bbox\.camera .* is not one of the cameras"):
        cameras.load(_write(tmp_path, canon))


def test_a_bbox_camera_without_semantic_segmentation_is_refused(tmp_path):
    canon = _without_packs()
    rgb_only = next(n for n, c in canon["cameras"].items() if "semantic_segmentation" not in c["channels"])
    canon["bbox"]["camera"] = rgb_only
    with pytest.raises(ValueError, match=f"bbox.camera {rgb_only} does not record semantic_segmentation"):
        cameras.load(_write(tmp_path, canon))


def test_a_bbox_camera_that_is_not_the_pack_camera_is_refused(tmp_path):
    """The box and a pack's mask come from one camera's segmentation. The pack stays on head_rgb and
    the box moves to the third-person camera, which is given a segmentation channel -- so the
    pack-camera rule is the only one that can refuse this."""
    canon = _without_packs()
    canon["packs"] = {"head": _a_pack_on("head_rgb")}
    canon["cameras"]["third_person_rgb"]["channels"] = ["rgb", "semantic_segmentation"]
    canon["bbox"]["camera"] = "third_person_rgb"
    with pytest.raises(ValueError, match="pack head's camera 'head_rgb' is not bbox.camera 'third_person_rgb'"):
        cameras.load(_write(tmp_path, canon))


def test_a_pack_on_the_bbox_camera_loads(tmp_path):
    # The control for the case above: the same explicit pack on the box's own camera.
    canon = _without_packs()
    canon["packs"] = {"head": _a_pack_on(canon["bbox"]["camera"])}
    assert cameras.load(_write(tmp_path, canon))["packs"]["head"]["camera"] == canon["bbox"]["camera"]


def test_a_packs_block_that_is_not_an_object_is_refused(tmp_path):
    canon = _without_packs()
    canon["packs"] = [_a_pack_on("head_rgb")]
    with pytest.raises(ValueError, match="packs is not an object"):
        cameras.load(_write(tmp_path, canon))


# ---- camera poses (recorded every frame) ----------------------------------------------------
# The head camera's pose is derived from its parent link, not read from the engine: the
# engine's reported pose for a camera under an articulation link stays at the spawn pose
# (measured), while the render follows the link.

YAW_90 = [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]     # xyzw, +90° about world z


def test_mounted_pose_carries_the_offset_with_the_parent():
    # parent at (1, 2, 3) turned +90° about z: its +x is world +y and its +y world -x, so the canon's
    # head offset (x, y, z) lands at (1 - y, 2 + x, 3 + z), and the optical axis and image-down its
    # rule's tilt gives in the parent turn with it
    head = cameras.load(CAMERAS)["cameras"]["head_rgb"]
    x, y, z = head["offset_pos"]
    t = math.radians(head["mount"]["tilt_down_deg"])
    pos, q = cameras.mounted_pose(np.array([1.0, 2.0, 3.0, *YAW_90]), head["offset_pos"], head["offset_rot_xyzw"])
    assert pos == pytest.approx([1.0 - y, 2.0 + x, 3.0 + z])
    assert _rotate_xyzw(q, [0, 0, 1]) == pytest.approx([0, math.cos(t), -math.sin(t)], abs=1e-12)   # parent +x → world +y
    assert _rotate_xyzw(q, [0, 1, 0]) == pytest.approx([0, -math.sin(t), -math.cos(t)], abs=1e-12)  # image down: world down


# A ROS optical frame at the origin whose axis is world +x: image right is world -y, image down world -z
LOOK_ALONG_X = [-0.5, 0.5, -0.5, 0.5]
# The head camera's pinhole, from its size and fov in the canon: hand formula, not cameras.intrinsics
_HEAD = CANON["cameras"]["head_rgb"]
FX = _HEAD["width"] / (2 * math.tan(math.radians(_HEAD["fov_h_deg"]) / 2))
CX, CY = _HEAD["width"] / 2, _HEAD["height"] / 2


@pytest.mark.parametrize("point,want", [
    ([5.0, 0.0, 0.0], (CX, CY, 5.0)),
    ([5.0, -1.0, 0.0], (CX + FX / 5.0, CY, 5.0)),     # world -y is image right
    ([5.0, 0.0, 1.0], (CX, CY - FX / 5.0, 5.0)),      # world up is image up
    ([4.0, 0.0, 0.0], (CX, CY, 4.0)),
])
def test_project_follows_the_ros_optical_frame(point, want):
    head = cameras.load(CAMERAS)["cameras"]["head_rgb"]
    u, v, depth = cameras.project(point, np.zeros(3), LOOK_ALONG_X, head)
    assert (u, v, depth) == pytest.approx(want)


def test_project_uses_the_camera_position():
    head = cameras.load(CAMERAS)["cameras"]["head_rgb"]
    assert cameras.project([6.0, 0.0, 1.0], np.array([1.0, 0.0, 1.0]), LOOK_ALONG_X, head) == pytest.approx(
        (head["width"] / 2, head["height"] / 2, 5.0))


def test_pose_record_is_the_root_pose_layout():
    # observation.root_pose is pos3 + quat4 (wxyz); the camera pose fields use the same seven numbers
    rec = cameras.pose_record(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3, 0.9]))
    assert rec.dtype == np.float32 and rec.tolist() == pytest.approx([1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3])
    assert cameras.load(CAMERAS)["pose"]["layout"] == "pos3 + quat4 (wxyz)"


def test_the_pack_depth_comes_from_a_channel_its_camera_records(tmp_path):
    # A declared pack's depth is a channel its camera renders (canon_with_pack adds it to the copy),
    # and the loader takes that canon.
    canon = media_canon(with_pack=True)
    pack = canon["packs"][PACK_NAME]
    assert pack["depth"]["source"] in canon["cameras"][pack["camera"]]["channels"]
    assert cameras.load(_write(tmp_path, canon))["packs"][PACK_NAME]["camera"] == canon["bbox"]["camera"]


# ---- new camera helpers: optical frames, fixed poses, fingerprint, names ----

HEAD_OFFSET = (-0.5, 0.5, -0.5, 0.5)            # the canon's horizontal head (optical +z along torso +x)


def _same_rotation(a, b):
    return all(np.allclose(cameras._q_rot(a, v), cameras._q_rot(b, v)) for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))


def test_the_optical_frame_looks_forward_with_the_image_down_away_from_up():
    q = cameras.optical_frame_xyzw((1, 0, 0), (0, 0, 1))
    assert np.allclose(cameras._q_rot(q, (0, 0, 1)), (1, 0, 0))     # optical z = forward
    assert np.allclose(cameras._q_rot(q, (0, 1, 0)), (0, 0, -1))    # image down = -up
    assert np.allclose(cameras._q_rot(q, (1, 0, 0)), (0, -1, 0))    # image right


def test_an_optical_frame_along_its_own_up_is_refused():
    with pytest.raises(ValueError, match="parallel"):
        cameras.optical_frame_xyzw((0, 0, 2), (0, 0, 1))


def test_a_zero_length_forward_is_refused():
    with pytest.raises(ValueError, match="zero-length"):
        cameras.optical_frame_xyzw((0, 0, 0), (0, 0, 1))


def test_no_tilt_is_the_horizontal_head_offset():
    assert _same_rotation(cameras.tilted_along_x_xyzw(0.0), HEAD_OFFSET)


def test_a_tilt_turns_the_optical_axis_toward_minus_z():
    t = math.radians(15.0)
    q = cameras.tilted_along_x_xyzw(15.0)
    assert np.allclose(cameras._q_rot(q, (0, 0, 1)), (math.cos(t), 0.0, -math.sin(t)))


SPEC = {"aisle": {"half_width": 5.0}}
MOUNT = {"side": "-y", "inset_from_rack_face_m": 0.1, "height_m": 5.0,
         "aim_x_range_m": [5.3, 11.8], "aim_y_m": 0.3, "aim_z_m": 1.0}
CCTV = {"type": "fixed", "width": 640, "height": 480, "fov_h_deg": 90.0, "sites": {"S1d": {"mount": MOUNT}}}


def _cctv(**mount):
    """CCTV with its S1d mount rule changed as given."""
    return {**CCTV, "sites": {"S1d": {"mount": {**MOUNT, **mount}}}}


def test_a_fixed_camera_stands_at_the_rack_face_and_looks_at_the_middle_of_its_aim():
    pos, q = cameras.fixed_pose(CCTV, SPEC, "S1d")
    assert np.allclose(pos, (8.55, -4.9, 5.0))
    aim = np.array([8.55, 0.3, 1.0]) - pos
    assert np.allclose(cameras._q_rot(q, (0, 0, 1)), aim / np.linalg.norm(aim))


def test_a_fixed_camera_sees_both_ends_of_its_aim_range():
    pos, q = cameras.fixed_pose(CCTV, SPEC, "S1d")
    for x in MOUNT["aim_x_range_m"]:
        u, v, z = cameras.project((x, 0.3, 1.0), pos, q, CCTV)
        assert z > 0 and 0 <= u < CCTV["width"] and 0 <= v < CCTV["height"]


def test_a_fixed_camera_on_an_unknown_side_is_refused():
    with pytest.raises(ValueError, match="side"):
        cameras.fixed_pose(_cctv(side="z"), SPEC, "S1d")


def test_a_fixed_camera_on_plus_y_side_sees_its_aim():
    cctv_plus_y = _cctv(side="+y")
    pos, q = cameras.fixed_pose(cctv_plus_y, SPEC, "S1d")
    # At +y side: y = +(half_width - inset) = +(5.0 - 0.1) = 4.9
    assert np.allclose(pos, (8.55, 4.9, 5.0))
    for x in MOUNT["aim_x_range_m"]:
        u, v, z = cameras.project((x, 0.3, 1.0), pos, q, cctv_plus_y)
        assert z > 0 and 0 <= u < CCTV["width"] and 0 <= v < CCTV["height"]


def test_fixed_pose_uses_rack_inner_face_y_from_layout(monkeypatch):
    monkeypatch.setattr(layout, "rack_inner_face_y", lambda spec: 7.0)
    pos, q = cameras.fixed_pose(CCTV, SPEC, "S1d")
    # Mocked rack_inner_face_y returns 7.0; y = -(7.0 - 0.1) = -6.9
    assert np.allclose(pos, (8.55, -6.9, 5.0))


def test_the_fingerprint_ignores_prose_and_the_pack_downsample_but_not_pixels():
    canon = {"fps": 50, "cameras": {"a": {"fov_h_deg": 69.4, "fov_note": "x"}},
             "packs": {"p": {"camera": "a", "downsample": {"width": 1}}}}
    f = cameras.recording_fingerprint(canon)
    canon["cameras"]["a"]["fov_note"] = "y"
    canon["packs"]["p"]["downsample"]["width"] = 2
    assert cameras.recording_fingerprint(canon) == f
    canon["cameras"]["a"]["fov_h_deg"] = 70.0
    assert cameras.recording_fingerprint(canon) != f


def test_the_fingerprint_ignores_what_the_converter_derives_for_the_shipped_copy():
    canon = cameras.load(CAMERAS)
    # shipped_cameras adds derived keys (fx, fy, cx, cy) the fingerprint must ignore
    shipped = convert_lerobot.shipped_cameras(canon)
    assert cameras.recording_fingerprint(canon) == cameras.recording_fingerprint(shipped)


def test_camera_names_keep_the_canon_order():
    assert cameras.camera_names({"cameras": {"b": {}, "a": {}}}) == ["b", "a"]


@pytest.mark.parametrize("edit_path,should_change", [
    # Fields nobody reads do not change fingerprint
    (("bbox", "source"), False),
    (("bbox", "coords"), False),
    (("pose", "layout"), False),
    (("description",), False),
    (("cameras", "head_rgb", "mount_note"), False),
    (("cameras", "cctv_rgb", "mount_note"), False),
    (("cameras", "cctv_rgb", "sites_note"), False),
    (("cameras", "third_person_rgb", "sites_note"), False),
    (("bbox", "segmentation_note"), False),
    # Fields recording paths read do change fingerprint
    (("fps",), True),
    (("cameras", "head_rgb", "width"), True),
    (("cameras", "head_rgb", "offset_pos"), True),
    (("cameras", "cctv_rgb", "sites", "S1d", "mount", "height_m"), True),     # fixed_pose places the camera from it
    (("cameras", "cctv_rgb", "sites", "S1f", "eye_m"), True),
    (("cameras", "third_person_rgb", "sites", "S1e", "frame"), True),         # follow_pose turns the offsets by it
    # a placement per side the person comes from: each side's placement is read by the episodes from that side
    (("cameras", "cctv_rgb", "sites", "S1g", "by_approach_side", "west", "eye_m"), True),
    (("cameras", "third_person_rgb", "sites", "S1g", "by_approach_side", "east", "eye_offset"), True),
    (("bbox", "invalid"), True),
    (("bbox", "classes_union"), True),
    (("bbox", "camera"), True),
    (("render", "reset_warm_renders"), True),
    (("seeing", "min_visible_fraction"), True),       # the seeing rule goes in whole, its notes aside
    (("packs", "head", "depth", "source"), True),
])
def test_the_fingerprint_counts_only_what_recording_reads(edit_path, should_change):
    """Fingerprint is stable under edits to fields nobody reads (prose, descriptive strings);
    changes under edits to fields recording code actually consumes. A pack row needs a canon that
    declares one: the fixture's (canon_with_pack), the repository canon has none."""
    canon = cameras.load(CAMERAS)
    if edit_path[0] == "packs":
        canon = canon_with_pack(canon)
    original_fp = cameras.recording_fingerprint(canon)

    # Make a deep copy and edit the specified path
    canon_copy = copy.deepcopy(canon)
    node = canon_copy
    for key in edit_path[:-1]:
        node = node[key]
    # Change the target: for bools toggle; for lists/numbers alter; for strings change
    target_val = node[edit_path[-1]]
    if isinstance(target_val, bool):
        node[edit_path[-1]] = not target_val
    elif isinstance(target_val, list):
        node[edit_path[-1]] = target_val + [0]  # extend
    elif isinstance(target_val, (int, float)):
        node[edit_path[-1]] = target_val + 1
    else:
        node[edit_path[-1]] = target_val + "_edited"

    new_fp = cameras.recording_fingerprint(canon_copy)
    if should_change:
        assert new_fp != original_fp, f"Expected {'.'.join(edit_path)} to change fingerprint but it did not"
    else:
        assert new_fp == original_fp, f"Expected {'.'.join(edit_path)} not to change fingerprint but it did"


def test_mat_to_xyzw_roundtrip_preserves_rotation():
    """_mat_to_xyzw converts rotation matrices to xyzw quaternions; verify it preserves
    rotations over random matrices covering all branches (trace>0, i=0, i=1, i=2)."""
    rng = np.random.default_rng(42)
    # Generate uniform random rotations from random unit quaternions
    branch_hits = {"trace>0": 0, "i=0": 0, "i=1": 0, "i=2": 0}
    for _ in range(100):
        # Sample uniform random quaternion via normal distribution normalization
        q_random = rng.normal(0, 1, 4)
        q_random /= np.linalg.norm(q_random)
        # Convert to rotation matrix: the basis vectors rotated by _q_rot
        R = np.column_stack([cameras._q_rot(q_random, v) for v in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]])

        # Record which branch _mat_to_xyzw will take
        t = float(np.trace(R))
        if t > 0.0:
            branch_hits["trace>0"] += 1
        else:
            i = int(np.argmax(np.diag(R)))
            branch_hits[f"i={i}"] += 1

        # Convert back and verify round-trip
        q = cameras._mat_to_xyzw(R)
        assert q[3] >= 0, "Quaternion should have w >= 0"

        # Check that rotating the three basis vectors recovers the matrix columns
        for i, v in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
            rotated = cameras._q_rot(q, v)
            expected = R[:, i]
            assert np.allclose(rotated, expected, rtol=0, atol=1e-12), \
                f"Basis {v} rotated by {q} should be {expected}, got {rotated}"

    # Verify all branches were hit
    assert all(branch_hits.values()), \
        f"Not all branches covered: {branch_hits}. trace>0: {branch_hits['trace>0']}, " \
        f"i=0: {branch_hits['i=0']}, i=1: {branch_hits['i=1']}, i=2: {branch_hits['i=2']}"


# ---- the seven cameras ----------------------------------------------
# The mounted cameras' offsets are extracted from the H1 assets (generation/envs/s1d/
# extract_camera_mounts.py, Isaac python); what is checked here is what the canon says about them.

def test_the_canon_names_seven_cameras_in_field_order():
    assert cameras.camera_names(CANON) == ["head_rgb", "head_down_rgb", "left_wrist_rgb", "right_wrist_rgb",
                                           "rear_rgb", "third_person_rgb", "cctv_rgb"]


def test_the_head_down_camera_looks_about_fifty_one_degrees_down():
    # The real D435's optical axis, 50.8 deg below the torso's horizontal (the official URDF)
    f = cameras._q_rot(CANON["cameras"]["head_down_rgb"]["offset_rot_xyzw"], (0, 0, 1))
    assert abs(math.degrees(math.atan2(-f[2], math.hypot(f[0], f[1]))) - 50.8) < 0.2


def test_the_head_camera_sits_above_the_d435_and_looks_level():
    head = CANON["cameras"]["head_rgb"]
    rule = head["mount"]
    down = CANON["cameras"][rule["above"]]
    assert rule["above"] == "head_down_rgb" and head["parent_prim"] == down["parent_prim"]
    assert abs(head["offset_pos"][2] - down["offset_pos"][2] - rule["up_m"]) < 1e-6
    assert head["offset_pos"][1] == down["offset_pos"][1]                  # the D435's left-right line
    # in front of the D435's lens. That the module clears both head meshes is checked where the meshes are,
    # in generation/envs/s1d/extract_camera_mounts.py (Isaac python)
    assert head["offset_pos"][0] > down["offset_pos"][0]
    assert _same_rotation(head["offset_rot_xyzw"], cameras.tilted_along_x_xyzw(rule["tilt_down_deg"]))


def test_the_rear_camera_sits_behind_the_head_camera_and_looks_straight_back():
    rear = CANON["cameras"]["rear_rgb"]
    rule = rear["mount"]
    front = CANON["cameras"][rule["behind"]]
    assert rule["behind"] == "head_rgb" and rear["parent_prim"] == front["parent_prim"]
    assert rear["offset_pos"][1:] == front["offset_pos"][1:]          # the head camera's height and left-right line
    # behind the torso's origin, the head between the two. That the module clears both head meshes is checked where
    # the meshes are, in generation/envs/s1d/extract_camera_mounts.py (Isaac python)
    assert rear["offset_pos"][0] < 0.0 < front["offset_pos"][0]
    t = math.radians(rule["tilt_down_deg"])
    q = rear["offset_rot_xyzw"]
    assert _rotate_xyzw(q, (0, 0, 1)) == pytest.approx((-math.cos(t), 0.0, -math.sin(t)), abs=1e-9)   # straight back
    assert _rotate_xyzw(q, (1, 0, 0)) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)     # image right: the robot's left


def test_the_wrist_cameras_are_mirror_mounts_tilted_as_their_rule_says():
    left, right = CANON["cameras"]["left_wrist_rgb"], CANON["cameras"]["right_wrist_rgb"]
    assert left["offset_pos"] == right["offset_pos"] and left["mount"] == right["mount"]
    assert left["offset_rot_xyzw"] == right["offset_rot_xyzw"]
    assert left["parent_prim"].replace("left", "right") == right["parent_prim"] != left["parent_prim"]
    assert _same_rotation(left["offset_rot_xyzw"], cameras.tilted_along_x_xyzw(left["mount"]["tilt_down_deg"]))


def test_the_cctv_stands_inside_the_hall_below_the_lights_and_sees_its_aim():
    spec = layout.load_spec(SCENE_SPEC)
    cam = CANON["cameras"]["cctv_rgb"]
    pos, q = cameras.fixed_pose(cam, spec, "S1d")
    mount = cameras.site_placement(cam, "S1d")["mount"]
    assert pos[2] < spec["lights"]["z"] and abs(pos[1]) < spec["aisle"]["half_width"]
    for x in mount["aim_x_range_m"]:
        u, v, z = cameras.project((x, mount["aim_y_m"], mount["aim_z_m"]), pos, q, cam)
        assert z > 0 and 0 <= u < cam["width"] and 0 <= v < cam["height"]


def test_the_new_cctvs_stand_inside_the_hall_below_the_rack_uprights():
    # Every placement of the new scenarios' CCTV, one per side the person comes from where a site has two: inside the
    # building, out of every collider and below every rack upright's top.
    spec = layout.load_spec(SCENE_SPEC)
    boxes = layout.build_layout(spec)
    floor = next(b for b in boxes if b.name == "floor")
    (x0, y0, _), (x1, y1, _) = layout.bounds(floor)
    top = min(layout.bounds(b)[1][2] for b in boxes if b.name.startswith("rack/upright/"))
    cam = CANON["cameras"]["cctv_rgb"]
    for sid, side in _placed("cctv_rgb", ("S1e", "S1f", "S1g")):
        pos, _ = cameras.fixed_pose(cam, spec, sid, approach_side=side)
        assert x0 < pos[0] < x1 and y0 < pos[1] < y1 and pos[2] < top, (sid, side)
        for b in boxes:
            if b.collider:
                lo, hi = layout.bounds(b)
                assert not all(lo[i] <= pos[i] <= hi[i] for i in range(3)), (sid, side, b.name)


def _placed(name, sids):
    """(scenario, side) for each placement of the canon camera ``name`` at those scenarios: side None for a site with
    one placement, each side the person comes from for a site with one per side."""
    cam = CANON["cameras"][name]
    return [(sid, side) for sid in sids for side in (cameras.site_sides(cam, sid) or (None,))]


def _set(camera, **values):
    return lambda c: c["cameras"][camera].update(values)


def _pop(camera, key):
    return lambda c: c["cameras"][camera].pop(key)


def _site(camera, sid, block=..., **values):
    """Replace one site's placement with ``block``, or update it with ``values``."""
    def edit(c):
        if block is not ...:
            c["cameras"][camera]["sites"][sid] = block
        else:
            c["cameras"][camera]["sites"][sid].update(values)
    return edit


def _site_pop(camera, sid, key):
    return lambda c: c["cameras"][camera]["sites"][sid].pop(key)


# One case per rule load() holds a camera to by its type, each breaking that rule alone.
@pytest.mark.parametrize("breakage, match", [
    pytest.param(_set("cctv_rgb", type="orbit"), r"cctv_rgb's type 'orbit' is not one of", id="unknown-type"),
    pytest.param(_pop("head_rgb", "parent_prim"), r"head_rgb \(mounted\) lacks parent_prim", id="mounted-parent"),
    pytest.param(_pop("head_rgb", "offset_pos"), r"head_rgb \(mounted\) lacks offset_pos", id="mounted-pos-absent"),
    pytest.param(_set("head_down_rgb", offset_pos=[0.1, float("nan"), 0.7]),
                 r"head_down_rgb\.offset_pos is not 3 finite numbers", id="mounted-pos-nan"),
    pytest.param(_set("left_wrist_rgb", offset_rot_xyzw=[0, 0, 1]),
                 r"left_wrist_rgb\.offset_rot_xyzw is not 4 finite numbers", id="mounted-rot-short"),
    pytest.param(_set("left_wrist_rgb", offset_rot_xyzw=[0, 0, 0, 2]),
                 r"left_wrist_rgb\.offset_rot_xyzw .* is not a unit quaternion", id="mounted-rot-unit"),
    pytest.param(_set("right_wrist_rgb", convention="opengl"),
                 r"right_wrist_rgb\.convention is 'opengl', not 'ros'", id="mounted-convention"),
    pytest.param(_set("head_rgb", sites={"S1d": {}}), r"head_rgb \(mounted\) has sites", id="mounted-sites"),
    pytest.param(_pop("third_person_rgb", "sites"), r"third_person_rgb \(follow\) lacks sites", id="follow-sites"),
    pytest.param(_set("third_person_rgb", sites={}), r"third_person_rgb \(follow\) lacks sites", id="follow-no-site"),
    pytest.param(_site("third_person_rgb", "S1e", None), r"third_person_rgb\.sites\.S1e is not a placement",
                 id="follow-site-empty"),
    pytest.param(_site_pop("third_person_rgb", "S1d", "eye_offset"), r"third_person_rgb\.sites\.S1d lacks eye_offset",
                 id="follow-eye"),
    pytest.param(_site_pop("third_person_rgb", "S1f", "frame"),
                 r"third_person_rgb\.sites\.S1f\.frame None is not one of", id="follow-frame-absent"),
    pytest.param(_site("third_person_rgb", "S1d", lookat_offset=[1.5, 0.0]),
                 r"third_person_rgb\.sites\.S1d\.lookat_offset is not 3 finite numbers", id="follow-lookat"),
    pytest.param(_site("third_person_rgb", "S1f", eye_offset=[-3.0, "0", 1.5]),
                 r"third_person_rgb\.sites\.S1f\.eye_offset is not 3 finite numbers", id="follow-eye-string"),
    pytest.param(_site("third_person_rgb", "S1e", eye_offset=[1.5, 0.0, -0.4], lookat_offset=[1.5, 0.0, -0.4]),
                 r"third_person_rgb\.sites\.S1e: the eye .* is the point it looks at", id="follow-eye-at-target"),
    pytest.param(_site("third_person_rgb", "S1e", eye_offset=[1.5, 0.0, 2.0], lookat_offset=[1.5, 0.0, -0.4]),
                 r"third_person_rgb\.sites\.S1e: .* straight up or down", id="follow-vertical"),
    pytest.param(_pop("cctv_rgb", "sites"), r"cctv_rgb \(fixed\) lacks sites", id="fixed-sites"),
    pytest.param(_site("cctv_rgb", "S1e", None), r"cctv_rgb\.sites\.S1e is not a placement", id="fixed-site-empty"),
    pytest.param(_site("cctv_rgb", "S1e", {"aim": [0, 0, 0]}), r"cctv_rgb\.sites\.S1e has neither a mount rule nor",
                 id="fixed-neither"),
    pytest.param(_site_pop("cctv_rgb", "S1f", "aim_m"), r"cctv_rgb\.sites\.S1f lacks aim_m", id="fixed-aim-absent"),
    pytest.param(_site("cctv_rgb", "S1f", eye_m=[-5.9, float("nan"), 4.5]),
                 r"cctv_rgb\.sites\.S1f\.eye_m is not 3 finite numbers", id="fixed-eye-nan"),
    pytest.param(_site("cctv_rgb", "S1f", aim_m=[True, -6.9, 0.9]),
                 r"cctv_rgb\.sites\.S1f\.aim_m is not 3 finite numbers", id="fixed-aim-bool"),
    pytest.param(_site("cctv_rgb", "S1f", eye_m=[-5.9, -6.6, 4.5], aim_m=[-5.9, -6.6, 4.5]),
                 r"cctv_rgb\.sites\.S1f: the eye .* is the point it looks at", id="fixed-eye-at-aim"),
    pytest.param(_site("cctv_rgb", "S1f", eye_m=[-5.9, -6.6, 4.5], aim_m=[-5.9, -6.6, 0.9]),
                 r"cctv_rgb\.sites\.S1f: .* straight up or down", id="fixed-vertical"),
    pytest.param(_site("cctv_rgb", "S1d", mount=None), r"cctv_rgb\.sites\.S1d\.mount is not an object", id="fixed-mount"),
])
def test_load_refuses_a_camera_its_type_cannot_place(tmp_path, breakage, match):
    canon = json.loads(CAMERAS.read_text(encoding="utf-8"))
    breakage(canon)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(canon), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        cameras.load(p)


def test_load_refuses_a_fixed_camera_whose_mount_lacks_a_key(tmp_path):
    # fixed_pose reads six keys: the side, the inset from the rack face, the height, and the aim's x range,
    # y and z. Each is required, one at a time.
    keys = list(cameras.site_placement(CANON["cameras"]["cctv_rgb"], "S1d")["mount"])
    assert len(keys) == 6, keys
    for key in keys:
        canon = json.loads(CAMERAS.read_text(encoding="utf-8"))
        del canon["cameras"]["cctv_rgb"]["sites"]["S1d"]["mount"][key]
        with pytest.raises(ValueError, match=rf"cctv_rgb\.sites\.S1d\.mount lacks \['{key}'\]"):
            cameras.load(_write(tmp_path, canon))


def _mount(**values):
    return lambda c: c["cameras"]["cctv_rgb"]["sites"]["S1d"]["mount"].update(values)


# One rule per value fixed_pose reads from a fixed camera's mount, each case breaking that rule alone.
# If load() saw only that the six keys were there, fixed_pose would meet a height of "5" or a NaN
# later, with a bare TypeError or ValueError instead of a sentence naming the canon.
@pytest.mark.parametrize("breakage, match", [
    pytest.param(_mount(side="y"), r"cctv_rgb\.sites\.S1d\.mount\.side 'y' is not one of \['\+y', '-y'\]", id="side"),
    pytest.param(_mount(height_m="5"), r"cctv_rgb\.sites\.S1d\.mount\.height_m is not a finite number: '5'", id="number-string"),
    pytest.param(_mount(inset_from_rack_face_m=True),
                 r"cctv_rgb\.sites\.S1d\.mount\.inset_from_rack_face_m is not a finite number: True", id="number-bool"),
    pytest.param(_mount(aim_z_m=float("inf")), r"cctv_rgb\.sites\.S1d\.mount\.aim_z_m is not a finite number: inf",
                 id="number-infinite"),
    pytest.param(_mount(aim_y_m=None), r"cctv_rgb\.sites\.S1d\.mount\.aim_y_m is not a finite number: None", id="number-null"),
    pytest.param(_mount(aim_x_range_m=[5.3]), r"cctv_rgb\.sites\.S1d\.mount\.aim_x_range_m is not 2 finite numbers: \[5\.3\]",
                 id="range-length"),
    pytest.param(_mount(aim_x_range_m=[5.3, float("nan")]),
                 r"cctv_rgb\.sites\.S1d\.mount\.aim_x_range_m is not 2 finite numbers: \[5\.3, nan\]", id="range-nan"),
    pytest.param(_mount(aim_x_range_m="5.3-11.8"),
                 r"cctv_rgb\.sites\.S1d\.mount\.aim_x_range_m is not 2 finite numbers: '5\.3-11\.8'", id="range-string"),
    pytest.param(_mount(aim_x_range_m=[11.8, 5.3]),
                 r"cctv_rgb\.sites\.S1d\.mount\.aim_x_range_m \[11\.8, 5\.3\] is not a range \(lo < hi\)", id="range-reversed"),
    pytest.param(_mount(aim_x_range_m=[5.3, 5.3]),
                 r"cctv_rgb\.sites\.S1d\.mount\.aim_x_range_m \[5\.3, 5\.3\] is not a range \(lo < hi\)", id="range-empty"),
])
def test_load_refuses_a_fixed_camera_whose_mount_value_fixed_pose_cannot_read(tmp_path, breakage, match):
    canon = json.loads(CAMERAS.read_text(encoding="utf-8"))
    breakage(canon)
    with pytest.raises(ValueError, match=match):
        cameras.load(_write(tmp_path, canon))


def test_load_takes_a_fixed_camera_on_either_side_with_integer_values(tmp_path):
    # The rules refuse what fixed_pose cannot read, not what it can: +y and whole numbers are fine.
    canon = json.loads(CAMERAS.read_text(encoding="utf-8"))
    canon["cameras"]["cctv_rgb"]["sites"]["S1d"]["mount"].update(side="+y", height_m=5, aim_x_range_m=[5, 12])
    loaded = cameras.load(_write(tmp_path, canon))
    assert cameras.site_placement(loaded["cameras"]["cctv_rgb"], "S1d")["mount"]["side"] == "+y"


# ---- the third-person and CCTV placements per scenario site -----------------------------------------
# A camera on the robot rides its link wherever the robot is; the third-person and CCTV cameras are placed
# per scenario (meta/cameras.json ``sites``): S1d's are the ones its recordings have, each other site has its own.

def _scenarios() -> list[dict]:
    return json.loads((REPO / "meta/scenarios.json").read_text(encoding="utf-8"))["scenarios"]


def test_every_camera_scenario_has_a_third_person_and_a_cctv_placement():
    ids = {s["scenario_id"] for s in _scenarios()}
    with_site = {s["scenario_id"] for s in _scenarios() if "site" in (s.get("scene") or {})}
    for name in ("third_person_rgb", "cctv_rgb"):
        sites = set(CANON["cameras"][name]["sites"])
        assert sites == {"S1d", "S1e", "S1f", "S1g"} == {"S1d"} | with_site, name
        assert sites <= ids, name
    assert all("sites" not in c for c in CANON["cameras"].values() if c["type"] == "mounted")


def test_s1d_keeps_its_recorded_cameras():
    # S1d's placements, as its recordings have them: the CCTV's rack-face rule and the third-person camera's
    # world offsets.
    pos, _ = cameras.fixed_pose(CANON["cameras"]["cctv_rgb"], layout.load_spec(SCENE_SPEC), "S1d")
    assert np.allclose(pos, (8.55, -4.9, 5.0), rtol=0.0, atol=1e-9)
    s1d = cameras.site_placement(CANON["cameras"]["third_person_rgb"], "S1d")
    root = np.array([4.0, -1.0, 1.05])
    want = cameras.follow_eye_target(root, (-3.0, -3.0, 1.2), (1.5, 0.0, -0.4))
    for yaw, first in ((0.0, 0.0), (1.3, -2.0)):          # a world placement does not turn with the robot
        eye, target = cameras.follow_pose(s1d, root, yaw, first)
        assert eye.tolist() == want[0].tolist() and target.tolist() == want[1].tolist()


def test_heading_follow_turns_with_the_robot():
    # S1e's and S1f's cameras follow the heading; the turn is worked on round numbers here, not on their values.
    assert all(cameras.site_placement(CANON["cameras"]["third_person_rgb"], sid)["frame"] == "heading"
               for sid in ("S1e", "S1f"))
    s1e = {"frame": "heading", "eye_offset": [-3.0, 0.0, 1.5], "lookat_offset": [1.5, 0.0, -0.4]}
    root = np.array([1.0, 2.0, 1.0])
    # A quarter turn left takes (x, y) to (-y, x): the eye (-3, 0) to (0, -3), the target (1.5, 0) to (0, 1.5)
    eye, target = cameras.follow_pose(s1e, root, math.pi / 2, 0.0)
    assert eye == pytest.approx(root + [0.0, -3.0, 1.5], abs=1e-12)
    assert target == pytest.approx(root + [0.0, 1.5, -0.4], abs=1e-12)
    eye, _ = cameras.follow_pose(s1e, root, 0.0, math.pi / 2)       # the first yaw is not what it turns by
    assert eye == pytest.approx(root + [-3.0, 0.0, 1.5], abs=1e-12)


def test_initial_heading_follow_keeps_its_first_yaw():
    s1g = cameras.site_placement(CANON["cameras"]["third_person_rgb"], "S1g", approach_side="east")
    assert s1g["frame"] == "initial_heading"
    start = _s1g_site().robot_start
    first = math.radians(start.yaw_deg)
    assert first == pytest.approx(-math.pi / 2, abs=1e-12)
    root = np.array([start.x, start.y, 1.05])
    # S1g's robot starts facing -y: the eye (0, -3.5) turned a quarter right is (-3.5, 0) -- west of it, along the
    # aisle, the side a person coming from the east walks toward -- and stays there while the robot turns about
    for yaw in (first, 0.0, math.pi / 2, math.pi, -0.4):
        eye, target = cameras.follow_pose(s1g, root, yaw, first)
        assert eye == pytest.approx(root + [-3.5, 0.0, 1.2], abs=1e-12), yaw
        assert target == pytest.approx(root + [0.0, 0.0, -0.2], abs=1e-12), yaw


def test_an_initial_heading_follow_needs_its_first_yaw():
    s1g = cameras.site_placement(CANON["cameras"]["third_person_rgb"], "S1g", approach_side="east")
    with pytest.raises(ValueError, match="처음 요"):
        cameras.follow_pose(s1g, np.zeros(3), 0.0, None)


def test_a_point_cctv_looks_at_its_aim():
    spec = layout.load_spec(SCENE_SPEC)
    cam = CANON["cameras"]["cctv_rgb"]
    for sid, side in _placed("cctv_rgb", ("S1e", "S1f", "S1g")):
        site = cameras.site_placement(cam, sid, approach_side=side)
        pos, q = cameras.fixed_pose(cam, spec, sid, approach_side=side)
        assert pos.tolist() == site["eye_m"], (sid, side)
        u, v, z = cameras.project(site["aim_m"], pos, q, cam)
        assert z > 0 and (u, v) == pytest.approx((cam["width"] / 2, cam["height"] / 2), abs=1e-6), (sid, side)
        assert abs(cameras._q_rot(q, (1, 0, 0))[2]) < 1e-12, (sid, side)  # a level picture: its rows are horizontal


def test_a_placement_for_an_unknown_scenario_is_refused():
    spec = layout.load_spec(SCENE_SPEC)
    cctv, tp = CANON["cameras"]["cctv_rgb"], CANON["cameras"]["third_person_rgb"]
    for sid in ("S1a", "S1x"):
        for cam in (cctv, tp):
            with pytest.raises(ValueError, match=f"{sid} 의 자리가 없음"):
                cameras.site_placement(cam, sid)
        with pytest.raises(ValueError, match=f"cctv_rgb 에 {sid} 의 자리가 없음"):
            cameras.site_placement(cctv, sid, name="cctv_rgb")
        with pytest.raises(ValueError, match=f"{sid} 의 자리가 없음"):
            cameras.fixed_pose(cctv, spec, sid)
    with pytest.raises(ValueError, match="자리"):
        cameras.site_placement(CANON["cameras"]["head_rgb"], "S1d")      # a camera on the robot has none


def test_a_follow_frame_nobody_knows_is_refused(tmp_path):
    canon = json.loads(CAMERAS.read_text(encoding="utf-8"))
    canon["cameras"]["third_person_rgb"]["sites"]["S1e"]["frame"] = "orbit"
    with pytest.raises(ValueError, match=r"third_person_rgb\.sites\.S1e\.frame 'orbit' is not one of "
                                         r"\['world', 'heading', 'initial_heading'\]"):
        cameras.load(_write(tmp_path, canon))
    with pytest.raises(ValueError, match="orbit"):
        cameras.follow_pose(canon["cameras"]["third_person_rgb"]["sites"]["S1e"], np.zeros(3), 0.0, 0.0)
    assert cameras.FOLLOW_FRAMES == ("world", "heading", "initial_heading")


def test_a_fixed_site_with_both_forms_is_refused(tmp_path):
    canon = json.loads(CAMERAS.read_text(encoding="utf-8"))
    sites = canon["cameras"]["cctv_rgb"]["sites"]
    sites["S1d"].update(eye_m=[8.55, -4.9, 5.0], aim_m=[8.55, 0.3, 1.0])
    with pytest.raises(ValueError, match=r"cctv_rgb\.sites\.S1d has both a mount rule and eye_m/aim_m"):
        cameras.load(_write(tmp_path, canon))
    with pytest.raises(ValueError, match="both"):
        cameras.fixed_pose(canon["cameras"]["cctv_rgb"], layout.load_spec(SCENE_SPEC), "S1d")


def test_the_fingerprint_sees_a_site_change():
    canon = copy.deepcopy(CANON)
    before = cameras.recording_fingerprint(canon)
    canon["cameras"]["cctv_rgb"]["sites"]["S1g"]["by_approach_side"]["west"]["aim_m"][0] += 0.5
    assert cameras.recording_fingerprint(canon) != before


# ---- a placement per side the person comes from (S1g) ------------------------------------------------
# A site may hold its placement as ``by_approach_side``: one placement for each side the person comes from (the
# conditions table's approach_side), the episode taking the one of its row's side.

PLAIN = {"eye_m": [0.0, -6.0, 5.0], "aim_m": [0.0, 0.0, 1.0]}
SIDED = {"by_approach_side": {"west": {"eye_m": [1.0, 2.0, 3.0], "aim_m": [0.0, 0.0, 1.0]},
                              "east": {"eye_m": [-1.0, 2.0, 3.0], "aim_m": [0.0, 0.0, 1.0]}}}


def _two_site_cctv():
    return {"type": "fixed", "sites": {"S9": copy.deepcopy(SIDED), "S8": copy.deepcopy(PLAIN)}}


def test_the_camera_canon_s_sides_are_the_conditions_tables():
    assert cameras.APPROACH_SIDES == conditions.APPROACH_SIDES


def test_a_placement_per_side_hands_out_the_side_asked_for():
    cam = _two_site_cctv()
    for side in ("west", "east"):
        assert cameras.site_placement(cam, "S9", approach_side=side) == SIDED["by_approach_side"][side]
        pos, _ = cameras.fixed_pose(cam, SPEC, "S9", approach_side=side)
        assert pos.tolist() == SIDED["by_approach_side"][side]["eye_m"]
    assert cameras.site_sides(cam, "S9") == ("west", "east") and cameras.site_sides(cam, "S8") == ()


@pytest.mark.parametrize("side", [None, "north", "West"])
def test_a_placement_per_side_refuses_a_side_it_holds_none_for(side):
    cam = _two_site_cctv()
    with pytest.raises(ValueError, match=rf"cctv_rgb 의 S9 자리는 사람이 오는 쪽마다 하나입니다 — {side!r} 의 자리가 없음 "
                                         r"\(고를 수 있는 쪽: \['west', 'east'\]"):
        cameras.site_placement(cam, "S9", name="cctv_rgb", approach_side=side)
    with pytest.raises(ValueError, match="사람이 오는 쪽마다"):
        cameras.fixed_pose(cam, SPEC, "S9", approach_side=side)


def test_a_plain_placement_does_not_read_the_side():
    cam = _two_site_cctv()
    for side in (None, "west", "east", "north"):
        assert cameras.site_placement(cam, "S8", approach_side=side) == PLAIN
    # The canon's other sites: the same placement and the same pose, to the bit, whichever side is handed in.
    spec = layout.load_spec(SCENE_SPEC)
    for name in ("cctv_rgb", "third_person_rgb"):
        cam = CANON["cameras"][name]
        for sid in ("S1d", "S1e", "S1f"):
            assert cameras.site_sides(cam, sid) == ()
            base = cameras.site_placement(cam, sid)
            assert all(cameras.site_placement(cam, sid, approach_side=s) == base for s in cameras.APPROACH_SIDES)
            if cam["type"] == "fixed":
                pos, q = cameras.fixed_pose(cam, spec, sid)
                for s in cameras.APPROACH_SIDES:
                    p2, q2 = cameras.fixed_pose(cam, spec, sid, approach_side=s)
                    assert p2.tobytes() == pos.tobytes() and q2.tobytes() == q.tobytes(), (sid, s)


def _sides_of(canon, camera):
    return canon["cameras"][camera]["sites"]["S1g"]["by_approach_side"]


# One case per rule load() holds a placement per side to, each breaking that rule alone.
@pytest.mark.parametrize("breakage, match", [
    pytest.param(lambda c: _sides_of(c, "cctv_rgb").pop("east"),
                 r"cctv_rgb\.sites\.S1g\.by_approach_side lacks a placement for \['east'\]", id="side-missing"),
    pytest.param(lambda c: _sides_of(c, "third_person_rgb").update(north=dict(_sides_of(c, "third_person_rgb")["west"])),
                 r"third_person_rgb\.sites\.S1g\.by_approach_side has \['north'\], not a side the person comes from",
                 id="side-unknown"),
    pytest.param(lambda c: c["cameras"]["cctv_rgb"]["sites"]["S1g"].update(eye_m=[-5.4, 11.4, 4.5]),
                 r"cctv_rgb\.sites\.S1g has \['eye_m'\] beside by_approach_side", id="key-beside"),
    pytest.param(lambda c: c["cameras"]["cctv_rgb"]["sites"]["S1g"].update(by_approach_side=[PLAIN, PLAIN]),
                 r"cctv_rgb\.sites\.S1g\.by_approach_side is not an object", id="not-an-object"),
    pytest.param(lambda c: _sides_of(c, "cctv_rgb").update(west=copy.deepcopy(SIDED)),
                 r"cctv_rgb\.sites\.S1g\.by_approach_side\.west has a by_approach_side of its own", id="nested"),
    pytest.param(lambda c: _sides_of(c, "cctv_rgb").update(west={}),
                 r"cctv_rgb\.sites\.S1g\.by_approach_side\.west is not a placement", id="inner-empty"),
    pytest.param(lambda c: _sides_of(c, "cctv_rgb")["west"].update(eye_m=[9.4, float("nan"), 4.5]),
                 r"cctv_rgb\.sites\.S1g\.by_approach_side\.west\.eye_m is not 3 finite numbers", id="inner-fixed-nan"),
    pytest.param(lambda c: _sides_of(c, "cctv_rgb")["east"].update(mount=dict(MOUNT)),
                 r"cctv_rgb\.sites\.S1g\.by_approach_side\.east has both a mount rule and eye_m/aim_m",
                 id="inner-fixed-both"),
    pytest.param(lambda c: _sides_of(c, "third_person_rgb")["west"].update(frame="orbit"),
                 r"third_person_rgb\.sites\.S1g\.by_approach_side\.west\.frame 'orbit' is not one of", id="inner-frame"),
    pytest.param(lambda c: _sides_of(c, "third_person_rgb")["east"].update(eye_offset=[0.0, 0.0, -0.2]),
                 r"third_person_rgb\.sites\.S1g\.by_approach_side\.east: the eye .* is the point it looks at",
                 id="inner-follow-at-target"),
])
def test_load_refuses_a_placement_per_side_it_cannot_read(tmp_path, breakage, match):
    canon = json.loads(CAMERAS.read_text(encoding="utf-8"))
    breakage(canon)
    with pytest.raises(ValueError, match=match):
        cameras.load(_write(tmp_path, canon))


def _s1g_site():
    return sites.load_site(next(s for s in _scenarios() if s["scenario_id"] == "S1g"))


def _first_s1g_row_from_each_side():
    """The first row of S1g's conditions table from each side the person comes from."""
    entry = conditions.scenario_entry("S1g")
    boxes = layout.build_layout(layout.load_spec(SCENE_SPEC))
    found = {}
    for i in range(conditions.load_conditions(entry).rows):
        row = conditions.draw_row(entry, i, boxes)
        found.setdefault(row["approach_side"], row)
        if len(found) == len(conditions.APPROACH_SIDES):
            return found
    raise AssertionError(f"S1g's table lacks a side: {sorted(found)}")


def test_the_s1g_scene_cameras_from_the_two_sides_are_mirrors_about_the_robot_s_start():
    # The CCTV: the west placement's eye and aim are the east one's turned over the robot's start x (y, z alike). The
    # third-person camera at the robot's start (its root there, its heading the start's): the two eyes are mirrors
    # about that x and look at one point.
    start = _s1g_site().robot_start
    x0, yaw = start.x, math.radians(start.yaw_deg)
    west, east = (cameras.site_placement(CANON["cameras"]["cctv_rgb"], "S1g", approach_side=s) for s in ("west", "east"))
    for key in ("eye_m", "aim_m"):
        assert west[key] == pytest.approx([2.0 * x0 - east[key][0], *east[key][1:]], abs=1e-12), key
    root = np.array([start.x, start.y, 1.05])
    (w_eye, w_at), (e_eye, e_at) = (
        cameras.follow_pose(cameras.site_placement(CANON["cameras"]["third_person_rgb"], "S1g", approach_side=s),
                            root, yaw, yaw) for s in ("west", "east"))
    assert w_eye == pytest.approx([2.0 * x0 - e_eye[0], e_eye[1], e_eye[2]], abs=1e-12)
    assert w_at == pytest.approx(e_at, abs=1e-12)


def test_each_s1g_scene_camera_stands_on_the_side_the_person_walks_toward():
    # The reason for one placement per side: the person, walking from the side the row names, comes toward the
    # cameras and is not between them and the robot. Read from real rows of the table and the walk the table gives
    # them (conditions.human_walk), the cameras for the robot at the row's start.
    site = _s1g_site()
    spec = layout.load_spec(SCENE_SPEC)
    rows = _first_s1g_row_from_each_side()
    assert sorted(rows) == sorted(cameras.APPROACH_SIDES)
    for side, row in rows.items():
        (from_x, _), (to_x, _) = conditions.human_walk(site, row)
        toward = math.copysign(1.0, to_x - from_x)
        start = conditions.row_site(site, row).robot_start
        pos, _ = cameras.fixed_pose(CANON["cameras"]["cctv_rgb"], spec, "S1g", approach_side=row["approach_side"])
        assert (pos[0] - start.x) * toward > 0.0, (side, pos)
        placement = cameras.site_placement(CANON["cameras"]["third_person_rgb"], "S1g", approach_side=row["approach_side"])
        eye, _ = cameras.follow_pose(placement, (start.x, start.y, 1.05), math.radians(start.yaw_deg),
                                     math.radians(start.yaw_deg))
        assert (eye[0] - start.x) * toward > 0.0, (side, eye)
