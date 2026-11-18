"""generation/envs/s1d/camera_io.py — how the run loop places, moves, warms and reads the canon's cameras
each step. Isaac-free: a stand-in env holds numpy arrays where the engine has tensors (camera_io reads
.torch when there is one).

Every camera comes out of meta/cameras.json, and the tests take the names from it by type
(cameras.camera_names), never as literals: seven of them, each placed by its type --
mounted on its own robot link, following the robot, or fixed in the hall. The rig is built for one
scenario's site (the canon's ``sites``): S1d unless a test says otherwise.

Depth is read only for a pack the canon declares, and the repository canon declares none: the
pack cases use the fixture's literal pack (make_fixture.canon_with_pack),
never the repository canon's."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from generation import cameras
from generation.envs.s1d import camera_io
from generation.scene import layout
from validator.make_fixture import PACK_NAME, canon_with_pack

REPO = Path(__file__).resolve().parents[2]
CANON = cameras.load(REPO / "meta/cameras.json")
# The aisle spec the S1d hall is built from (generation/scene/layout.py SCENE_SPEC) -- spelt here as
# tests/generation/test_cameras.py spells it, not read from the code under test.
SCENE_SPEC = REPO / "generation/scene/logistics_aisle.yaml"
NAMES = cameras.camera_names(CANON)
S1D = "S1d"                                         # the site most tests build the rig for
BOX = CANON["bbox"]["camera"]                       # the camera whose segmentation the box comes from
OTHER = next(name for name in NAMES if name != BOX)


def _of_type(*kinds):
    return [name for name in NAMES if CANON["cameras"][name]["type"] in kinds]


MOUNTED, FOLLOW, FIXED = _of_type("mounted"), _of_type("follow"), _of_type("fixed")
MOVING = _of_type("mounted", "follow")
# Every mounted camera's parent link. Each gets a pose of its own below, so a camera placed off another
# camera's link shows.
LINKS = sorted({CANON["cameras"][name]["parent_prim"] for name in MOUNTED})
VALUE = {name: 7 + k for k, name in enumerate(NAMES)}   # each camera's picture: its own value
STALE = [0.108, 0.032, 2.843]          # what the engine kept reporting for the head camera (2026-09-23)
BR04 = -16518367                       # (33, 243, 3, 255) packed as the int32 image stores it


def _rot(q_xyzw, v):
    # v' = v + 2w(u×v) + 2u×(u×v), u = (x, y, z) — written out here, not taken from the code under test
    x, y, z, w = q_xyzw
    u, v = np.array([x, y, z], float), np.asarray(v, float)
    return v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v))


def _turn(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    h = math.radians(deg) / 2
    return [*(a * math.sin(h)), math.cos(h)]


def _link_poses(shift=0.0):
    """The robot's body_link_pose_w row for env 0: body 0 is one no camera hangs on (the pelvis), link i of
    LINKS is body i + 1, each at its own place and turned its own way. ``shift`` walks them all along x."""
    axes = ((0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 1))
    poses = np.zeros((len(LINKS) + 1, 7), np.float32)
    poses[0] = [shift, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    for i in range(len(LINKS)):
        poses[i + 1] = [1.0 + 3 * i + shift, 2.0 - i, 3.0 + 0.5 * i, *_turn(axes[i % len(axes)], 90.0 - 35.0 * i)]
    return poses


class Cam:
    def __init__(self, output, info=None, pos=STALE, quat=(0.0, 0.0, 0.0, 1.0)):
        self.data = SimpleNamespace(output=output, info=info, pos_w=np.array([pos], np.float32),
                                    quat_w_ros=np.array([quat], np.float32))
        self.calls = []

    def update(self, dt, force_recompute=False):
        self.calls.append(("update", dt, force_recompute))

    def set_world_poses_from_view(self, eyes, targets):
        self.calls.append(("view", np.asarray(eyes), np.asarray(targets)))


class Scene(dict):
    """env.scene: the entities by name, and the env origins (num_envs x 3)."""

    def __init__(self, entities, origins):
        super().__init__(entities)
        self.env_origins = np.asarray(origins, np.float32)


def _env(root=(10.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0), follow_pose=((5.0, 6.0, 7.0), (0.0, 0.0, 0.0, 1.0)),
         root_quat=(0.0, 0.0, 0.0, 1.0)):
    """A stand-in env with every camera of the canon: its own picture (VALUE), the box camera's depth,
    segmentation and labels, the engine's pose (STALE but for a follow camera, which the loop moves), and a
    robot whose root is at ``root`` turned by ``root_quat`` (xyzw) and whose links are at _link_poses()."""
    cams = {}
    for name in NAMES:
        c = CANON["cameras"][name]
        rgba = np.zeros((1, c["height"], c["width"], 4), np.uint8)
        rgba[..., 0] = VALUE[name]
        output, info = {"rgb": rgba[..., :3]}, None          # the engine's rgb is a strided view of rgba
        if name == BOX:
            output["distance_to_image_plane"] = np.full((1, c["height"], c["width"], 1), 2.5, np.float32)
            output["semantic_segmentation"] = np.full((1, c["height"], c["width"], 1), BR04, np.int32)
            info = {"semantic_segmentation": {"idToLabels": {"(33, 243, 3, 255)": {"class": "br-04"}}}}
        pos, quat = follow_pose if c["type"] == "follow" else (STALE, (0.0, 0.0, 0.0, 1.0))
        cams[name] = Cam(output, info, pos=pos, quat=quat)
    robot = SimpleNamespace(find_bodies=lambda n: ([LINKS.index(n) + 1], [n]) if n in LINKS else ([], []),
                            data=SimpleNamespace(root_pos_w=np.array([root], np.float32),
                                                 root_quat_w=np.array([root_quat], np.float32),
                                                 body_link_pose_w=_link_poses()[None]))
    log = []
    env = SimpleNamespace(scene=Scene({**cams, "robot": robot}, [origin]),
                          sim=SimpleNamespace(render=lambda: log.append("render")), step_dt=0.02)
    return env, cams, log


def _draw(cams, value, names=NAMES):
    """The engine drew a new picture for these cameras."""
    for name in names:
        cams[name].data.output["rgb"][...] = value


def test_the_canon_has_a_camera_of_every_type_the_rig_places():
    # The tests below go through the cameras of each type; with none of a type they would pass vacuously.
    assert MOUNTED and FOLLOW and FIXED and len(LINKS) > 1
    assert set(NAMES) == set(MOUNTED) | set(FOLLOW) | set(FIXED)


# ---- poses, by type ---------------------------------------------------------------------------

def test_each_mounted_camera_s_pose_comes_from_its_own_parent_link():
    # The wrists hang on two different elbows: every link here is at a pose of its own, and each
    # mounted camera's record is its own link's pose composed with its canon offset -- not another camera's
    # link, and not the pose the engine reports for a camera under an articulation link (STALE).
    env, cams, _ = _env()
    poses = camera_io.Rig(env, CANON, S1D).read()["poses"]
    links = env.scene["robot"].data.body_link_pose_w[0]
    for name in MOUNTED:
        c = CANON["cameras"][name]
        link = links[LINKS.index(c["parent_prim"]) + 1]
        rec = poses[name]
        q = [rec[4], rec[5], rec[6], rec[3]]                               # the record is wxyz
        assert rec[:3] == pytest.approx(link[:3] + _rot(link[3:], c["offset_pos"]), abs=1e-5), name
        for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):                     # the optical frame turned by the link
            assert _rot(q, axis) == pytest.approx(_rot(link[3:], _rot(c["offset_rot_xyzw"], axis)), abs=1e-5), name
        pos, quat = cameras.mounted_pose(link, c["offset_pos"], c["offset_rot_xyzw"])
        assert rec.tolist() == pytest.approx(cameras.pose_record(pos, quat).tolist(), abs=1e-6), name


def test_a_follow_camera_s_pose_is_the_engine_s():
    env, *_ = _env(follow_pose=((5.0, 6.0, 7.0), (0.1, 0.2, 0.3, 0.9)))
    poses = camera_io.Rig(env, CANON, S1D).read()["poses"]
    for name in FOLLOW:
        assert poses[name].tolist() == pytest.approx([5.0, 6.0, 7.0, 0.9, 0.1, 0.2, 0.3]), name


def test_a_fixed_camera_s_pose_is_its_derived_pose_plus_the_env_origin_at_every_step(monkeypatch):
    # Where s1d_env_cfg puts its prim: cameras.fixed_pose from the aisle spec (env frame) under the env
    # origin. It never moves -- the robot walking changes nothing -- and it is derived once, not per step.
    derived = []
    fixed_pose = cameras.fixed_pose
    monkeypatch.setattr(cameras, "fixed_pose",
                        lambda cam, spec, sid, approach_side=None: derived.append((cam, sid, approach_side))
                        or fixed_pose(cam, spec, sid, approach_side=approach_side))
    origin = (10.0, -20.0, 0.5)
    env, cams, _ = _env(origin=origin)
    rig = camera_io.Rig(env, CANON, S1D)
    reads = []
    for step in range(3):
        env.scene["robot"].data.body_link_pose_w = _link_poses(shift=float(step))[None]
        _draw(cams, 40 + step)
        reads.append(rig.read()["poses"])
    spec = layout.load_spec(SCENE_SPEC)
    for name in FIXED:
        pos, q = fixed_pose(CANON["cameras"][name], spec, S1D)
        want = cameras.pose_record(np.asarray(pos) + np.asarray(origin), q)
        assert all(np.array_equal(r[name], want) for r in reads), name
    assert len(derived) == len(FIXED) and all(sid == S1D and side is None for _, sid, side in derived)


# ---- moving, warming, reading -----------------------------------------------------------------

def test_follow_moves_each_follow_camera_to_its_root_offsets_and_no_other_camera():
    root = (10.0, 0.0, 1.0)
    env, cams, _ = _env(root=root)
    camera_io.Rig(env, CANON, S1D).follow()
    for name in NAMES:
        if name not in FOLLOW:
            assert cams[name].calls == [], name        # a mounted camera rides its link; a fixed one stays
            continue
        site = cameras.site_placement(CANON["cameras"][name], S1D)
        assert site["frame"] == "world"
        (_, eyes, targets), = cams[name].calls
        assert eyes.shape == (1, 3) and eyes[0].tolist() == pytest.approx(np.add(root, site["eye_offset"]).tolist())
        assert targets[0].tolist() == pytest.approx(np.add(root, site["lookat_offset"]).tolist())


# ---- the follow camera's frame at the other sites: the robot's heading, or the one after the reset ------

def _yaw(deg):
    return _turn((0, 0, 1), deg)


def _quarter_left(v):
    # (x, y, z) turned a quarter left about z: (-y, x, z) -- written out here
    return [-v[1], v[0], v[2]]


def _quarter_right(v):
    return [v[1], -v[0], v[2]]


@pytest.mark.parametrize("tilt_deg", [0.0, 12.0])
def test_a_heading_follow_camera_turns_its_eye_with_the_robot(tilt_deg):
    # S1e's third-person camera keeps behind the robot: its offsets turn with the robot's heading -- the way its
    # x axis points, seen from above, so a robot leaning forward has the heading it walks along.
    root = (1.0, 2.0, 1.0)
    quat = cameras._q_mul(_yaw(90.0), _turn((0, 1, 0), tilt_deg))
    env, cams, _ = _env(root=root, root_quat=quat)
    rig = camera_io.Rig(env, CANON, "S1e")
    rig.reset(env)
    rig.follow()
    for name in FOLLOW:
        site = cameras.site_placement(CANON["cameras"][name], "S1e")
        assert site["frame"] == "heading"
        (_, eyes, targets), = cams[name].calls
        assert eyes[0].tolist() == pytest.approx(np.add(root, _quarter_left(site["eye_offset"])).tolist(), abs=1e-5)
        assert targets[0].tolist() == pytest.approx(np.add(root, _quarter_left(site["lookat_offset"])).tolist(),
                                                    abs=1e-5)


def _s1g_env(side, **kw):
    """A stand-in env built for S1g with the person coming from ``side``: its cfg names both, as HriEnvCfg's does."""
    env, cams, log = _env(**kw)
    env.cfg = SimpleNamespace(scenario_id="S1g", camera_approach_side=side)
    return env, cams, log


@pytest.mark.parametrize("side", ["west", "east"])
def test_an_initial_heading_follow_camera_keeps_the_yaw_the_reset_left(side):
    # S1g's robot turns about on the spot; its third-person camera stays where the robot's first heading puts
    # it, until the next reset takes the heading the robot then has.
    root = (2.0, 10.75, 1.05)
    env, cams, _ = _s1g_env(side, root=root, root_quat=_yaw(90.0))
    rig = camera_io.Rig(env, CANON, "S1g")
    rig.reset(env)
    env.scene["robot"].data.root_quat_w = np.array([_yaw(-90.0)], np.float32)       # turned about since
    rig.follow()
    for name in FOLLOW:
        site = cameras.site_placement(CANON["cameras"][name], "S1g", approach_side=side)
        assert site["frame"] == "initial_heading"
        (_, eyes, _), = cams[name].calls
        assert eyes[0].tolist() == pytest.approx(np.add(root, _quarter_left(site["eye_offset"])).tolist(), abs=1e-5)
        cams[name].calls.clear()
    rig.reset(env)
    rig.follow()
    for name in FOLLOW:
        site = cameras.site_placement(CANON["cameras"][name], "S1g", approach_side=side)
        (_, eyes, _), = cams[name].calls
        assert eyes[0].tolist() == pytest.approx(np.add(root, _quarter_right(site["eye_offset"])).tolist(), abs=1e-5)


def test_an_initial_heading_follow_camera_before_any_reset_is_refused():
    env, *_ = _s1g_env("west")
    rig = camera_io.Rig(env, CANON, "S1g")
    with pytest.raises(ValueError, match="처음 요"):
        rig.follow()


def test_a_fixed_camera_at_a_point_site_records_its_eye_under_the_env_origin():
    origin = (10.0, -20.0, 0.5)
    env, *_ = _env(origin=origin)
    poses = camera_io.Rig(env, CANON, "S1f").read()["poses"]
    for name in FIXED:
        site = cameras.site_placement(CANON["cameras"][name], "S1f")
        assert poses[name][:3].tolist() == pytest.approx(np.add(site["eye_m"], origin).tolist(), abs=1e-5), name


@pytest.mark.parametrize("sid", ["S1a", "S1x"])
def test_rig_refuses_a_scenario_without_a_site(sid):
    env, *_ = _env()
    with pytest.raises(ValueError, match=f"{sid} 의 자리가 없음"):
        camera_io.Rig(env, CANON, sid)


def test_rig_refuses_a_scenario_other_than_the_one_its_env_was_built_for():
    # s1d_env_cfg puts the CCTV's prim at the site of its cfg's scenario_id; a rig built for another site would
    # record that other site's camera pose for every frame the CCTV renders from where it stands.
    env, *_ = _env()
    env.cfg = SimpleNamespace(scenario_id="S1e")
    with pytest.raises(ValueError, match="리그의 시나리오 'S1f' 가 환경 설정의 scenario_id 'S1e' 와 다릅니다"):
        camera_io.Rig(env, CANON, "S1f")


def test_rig_built_for_its_env_s_own_scenario_records_that_site():
    origin = (10.0, -20.0, 0.5)
    env, *_ = _env(origin=origin)
    env.cfg = SimpleNamespace(scenario_id="S1f")
    poses = camera_io.Rig(env, CANON, "S1f").read()["poses"]
    for name in FIXED:
        site = cameras.site_placement(CANON["cameras"][name], "S1f")
        assert poses[name][:3].tolist() == pytest.approx(np.add(site["eye_m"], origin).tolist(), abs=1e-5), name


@pytest.mark.parametrize("cfg", [SimpleNamespace(), None], ids=["cfg-without-scenario_id", "no-cfg"])
def test_rig_on_an_env_whose_cfg_names_no_scenario_takes_the_one_it_is_given(cfg):
    # S1a's cfg has no scenario_id (it places no site camera); the stand-in env of the other tests has no cfg at all.
    env, *_ = _env()
    if cfg is not None:
        env.cfg = cfg
    poses = camera_io.Rig(env, CANON, "S1f").read()["poses"]
    for name in FIXED:
        site = cameras.site_placement(CANON["cameras"][name], "S1f")
        assert poses[name][:3].tolist() == pytest.approx(site["eye_m"], abs=1e-5), name


# ---- a site placed per side the person comes from (S1g): the side the env's cfg was built for ---------------

@pytest.mark.parametrize("side", ["west", "east"])
def test_rig_records_the_side_its_env_s_cfg_placed_the_scene_cameras_for(side):
    # s1d_env_cfg puts the CCTV's prim at the placement of the cfg's camera_approach_side (HriEnvCfg fills it from
    # its row); the rig records that same placement's pose, and moves the third-person camera to the same side's
    # offsets -- not the other side's.
    origin = (10.0, -20.0, 0.5)
    root = (2.0, 10.3, 1.05)
    env, cams, _ = _s1g_env(side, origin=origin, root=root, root_quat=_yaw(-90.0))
    rig = camera_io.Rig(env, CANON, "S1g")
    poses = rig.read()["poses"]
    other = next(s for s in cameras.APPROACH_SIDES if s != side)
    for name in FIXED:
        site = cameras.site_placement(CANON["cameras"][name], "S1g", approach_side=side)
        assert poses[name][:3].tolist() == pytest.approx(np.add(site["eye_m"], origin).tolist(), abs=1e-5), name
        far = cameras.site_placement(CANON["cameras"][name], "S1g", approach_side=other)
        assert far["eye_m"] != site["eye_m"], name                      # the two sides are two places
    rig.reset(env)
    rig.follow()
    for name in FOLLOW:
        site = cameras.site_placement(CANON["cameras"][name], "S1g", approach_side=side)
        (_, eyes, _), = cams[name].calls
        assert eyes[0].tolist() == pytest.approx(np.add(root, _quarter_right(site["eye_offset"])).tolist(), abs=1e-5)


@pytest.mark.parametrize("cfg", [SimpleNamespace(scenario_id="S1g"), SimpleNamespace(scenario_id="S1g",
                                                                                       camera_approach_side=None),
                                 SimpleNamespace(scenario_id="S1g", camera_approach_side="north"), None],
                         ids=["cfg-without-side", "side-none", "side-unknown", "no-cfg"])
def test_rig_for_a_site_placed_per_side_refuses_an_env_that_names_no_side_it_holds(cfg):
    env, *_ = _env()
    if cfg is not None:
        env.cfg = cfg
    with pytest.raises(ValueError, match=r"의 S1g 자리는 사람이 오는 쪽마다 하나입니다 .*\['west', 'east'\]"):
        camera_io.Rig(env, CANON, "S1g")


def test_warm_renders_and_recomputes_every_camera_n_times():
    env, cams, log = _env()
    camera_io.Rig(env, CANON, S1D).warm(3)
    assert log == ["render"] * 3
    assert {name: cam.calls for name, cam in cams.items()} == {name: [("update", 0.0, True)] * 3 for name in NAMES}


def test_read_hands_the_writer_plain_frames_depth_seg_and_labels():
    env, *_ = _env()
    got = camera_io.Rig(env, canon_with_pack(CANON), S1D).read()
    assert list(got["frames"]) == NAMES and list(got["poses"]) == NAMES
    for name in NAMES:
        c, frame, pose = CANON["cameras"][name], got["frames"][name], got["poses"][name]
        assert frame.shape == (c["height"], c["width"], 3) and frame.dtype == np.uint8, name
        assert frame.flags["C_CONTIGUOUS"] and int(frame[0, 0, 0]) == VALUE[name], name   # its own picture
        assert pose.shape == (7,) and pose.dtype == np.float32, name
    box = CANON["cameras"][BOX]
    assert got["depth"].shape == (box["height"], box["width"]) and float(got["depth"][0, 0]) == 2.5
    assert got["seg"].shape == (box["height"], box["width"]) and int(got["seg"][0, 0]) == BR04
    assert got["id_to_labels"] == {"(33, 243, 3, 255)": {"class": "br-04"}}


def _packless(canon):
    # A copy declaring no pack whatever the canon it starts from does.
    out = copy.deepcopy(canon)
    out.pop("packs", None)
    return out


def test_without_a_pack_read_hands_the_writer_no_depth():
    # No pack declared: nothing to put a depth in, so none is read -- the writer refuses one it
    # would have to drop. The frames, the box's segmentation and its labels are read as before.
    env, *_ = _env()
    got = camera_io.Rig(env, _packless(CANON), S1D).read()
    assert "depth" not in got
    for name in NAMES:
        assert got["frames"][name].shape == (CANON["cameras"][name]["height"], CANON["cameras"][name]["width"], 3)
    box = CANON["cameras"][BOX]
    assert got["seg"].shape == (box["height"], box["width"]) and int(got["seg"][0, 0]) == BR04
    assert got["id_to_labels"] == {"(33, 243, 3, 255)": {"class": "br-04"}}


def _box_on(canon, name):
    """``canon`` with bbox.camera moved to ``name``, which then records the segmentation -- the channel the box
    is computed from, and one cameras.load requires of bbox.camera."""
    canon["bbox"]["camera"] = name
    canon["cameras"][name]["channels"] = [*canon["cameras"][name]["channels"], "semantic_segmentation"]
    return canon


def test_without_a_pack_the_segmentation_is_the_bbox_camera_s():
    # bbox.camera names the camera whose segmentation and labels the box is computed from; moved to
    # another camera, its picture is the one handed over, not the first box camera's.
    env, cams, _ = _env()
    c = CANON["cameras"][OTHER]
    cams[OTHER].data.output["semantic_segmentation"] = np.full((1, c["height"], c["width"], 1), 3, np.int32)
    cams[OTHER].data.info = {"semantic_segmentation": {"idToLabels": {"(3, 0, 0, 0)": {"class": "rack"}}}}
    got = camera_io.Rig(env, _box_on(_packless(CANON), OTHER), S1D).read()
    assert "depth" not in got
    assert got["seg"].shape == (c["height"], c["width"]) and int(got["seg"][0, 0]) == 3
    assert got["id_to_labels"] == {"(3, 0, 0, 0)": {"class": "rack"}}


# ---- stale frames (2026-09-23: the renderer once discarded a head frame, and step 448 of a
# control run repeated step 447 bit for bit while the frame count stayed right) ---------------

class Screen:
    """A camera whose picture changes only when it is rendered again (update with force_recompute)."""

    def __init__(self, cam, shape):
        self.cam, self.value = cam, 1
        self.rgb = np.full(shape, 1, np.uint8)          # contiguous on purpose: a Rig that kept a view
        cam.data.output["rgb"] = self.rgb[None]          # of it would see every frame as unchanged
        original = cam.update

        def update(dt, force_recompute=False):
            original(dt, force_recompute)
            if force_recompute:
                self.value += 1
                self.rgb[...] = self.value

        cam.update = update


def _screens(env, names):
    return {name: Screen(env.scene[name], (CANON["cameras"][name]["height"], CANON["cameras"][name]["width"], 3))
            for name in names}


def _stale_named(err):
    # The message leads with the unchanged cameras: "<a>, <b>: still the previous read's frame ..."
    return str(err.value).split(": ", 1)[0].split(", ")


def test_fresh_frames_are_not_rendered_again():
    env, _, log = _env()
    screens = _screens(env, NAMES)
    rig = camera_io.Rig(env, CANON, S1D)
    for value in (5, 6, 7):                              # the engine draws a new picture every step
        for screen in screens.values():
            screen.rgb[...] = value
        assert rig.read()["rerendered"] == 0
    assert log == []


def test_a_frame_identical_to_the_last_one_is_rendered_again():
    env, _, log = _env()
    screens = _screens(env, NAMES)
    rig = camera_io.Rig(env, CANON, S1D)
    first = rig.read()
    again = rig.read()                                   # nothing drew a new picture in between
    assert again["rerendered"] == 1 and log == ["render"]
    # The state is rendered again for every camera, and every camera is read again -- a fixed one too.
    for name in NAMES:
        assert int(again["frames"][name][0, 0, 0]) == screens[name].value != int(first["frames"][name][0, 0, 0]), name


def test_a_frame_that_stays_stale_stops_the_run():
    env, _, log = _env()
    rig = camera_io.Rig(env, CANON, S1D)                      # no Screen: rendering again changes nothing
    rig.read()
    with pytest.raises(RuntimeError, match="still the previous read's frame") as err:
        rig.read()
    assert _stale_named(err) == MOVING                   # every moving camera; a fixed one's repeat is recorded
    assert log == ["render"] * camera_io.MAX_RERENDERS


@pytest.mark.parametrize("still", MOVING)
def test_one_moving_camera_that_stays_stale_stops_the_run(still):
    # Every other camera draws a fresh picture at each render; this one keeps giving the last read's.
    env, _, log = _env()
    _screens(env, [name for name in NAMES if name != still])
    rig = camera_io.Rig(env, CANON, S1D)
    rig.read()
    with pytest.raises(RuntimeError, match="still the previous read's frame") as err:
        rig.read()
    assert _stale_named(err) == [still]
    assert log == ["render"] * camera_io.MAX_RERENDERS


@pytest.mark.parametrize("still", MOVING)
def test_a_moving_camera_that_repeats_a_frame_with_one_lit_pixel_still_stops_the_run(still):
    # Black frames are exempt from the stale rule (below) -- all zeros, not dark ones: one lit pixel is a
    # picture a fresh render would change, so its repeat is a discarded render as before.
    env, cams, log = _env()
    _screens(env, [name for name in NAMES if name != still])
    cams[still].data.output["rgb"][...] = 0
    cams[still].data.output["rgb"][0, -1, -1, 2] = 1
    rig = camera_io.Rig(env, CANON, S1D)
    rig.read()
    with pytest.raises(RuntimeError, match="still the previous read's frame") as err:
        rig.read()
    assert _stale_named(err) == [still]
    assert log == ["render"] * camera_io.MAX_RERENDERS


@pytest.mark.parametrize("dark", NAMES)
def test_a_camera_that_renders_black_frames_is_recorded_without_rendering_again(dark):
    # A black frame -- every pixel 0 -- is never stale: all zeros repeat whether the renderer drew them again
    # or not, so bit-equality cannot tell a discarded render from a camera that sees nothing (one buried in
    # some geometry, or mounted in the wrong place). The frames are recorded for the writer to count and the
    # converter to refuse a camera that recorded nothing else -- not taken for the renderer's fault, and not
    # a reason to stop the run (a black wrist camera once stopped it at step 0).
    env, cams, log = _env()
    others = [name for name in NAMES if name != dark]
    cams[dark].data.output["rgb"][...] = 0
    rig = camera_io.Rig(env, CANON, S1D)
    rig.warm(2)                                          # black in the warm-up renders already
    log.clear()
    for step, value in enumerate((0, 0, 9, 0, 0)):       # black from the reset, lit once, black again
        _draw(cams, 40 + step, others)                   # every other camera: a new picture
        cams[dark].data.output["rgb"][...] = value
        got = rig.read()
        assert got["rerendered"] == 0, step
        assert int(got["frames"][dark].max()) == value, step
    assert log == []


def test_a_fixed_camera_s_repeated_frame_is_rendered_again_and_counted():
    # A discarded render is per camera -- in a measured run the head camera's frame repeated while
    # the third-person camera's was new -- so fresh moving frames do not prove a fixed camera's frame is fresh.
    # Its repeat is rendered again like any other camera's, and the step counts in rerendered.
    assert FIXED
    env, cams, log = _env()
    screens = _screens(env, NAMES)
    rig = camera_io.Rig(env, CANON, S1D)
    rig.read()
    for name in MOVING:
        screens[name].rgb[...] = 50                      # the moving cameras drew a new picture; the fixed did not
    got = rig.read()
    assert got["rerendered"] == 1 and log == ["render"]
    for name in FIXED:
        assert int(got["frames"][name][0, 0, 0]) == screens[name].value, name      # the picture drawn again


def test_a_fixed_camera_that_repeats_through_every_render_is_recorded_and_the_run_goes_on():
    # Nothing in a fixed camera's view may move for a while (the robot not yet
    # in it, the dummy waiting), and its picture may then repeat for real. After MAX_RERENDERS a fixed
    # camera's repeat is taken for a still view and recorded -- a moving camera's stops the run (above).
    assert FIXED
    env, cams, log = _env()
    screens = _screens(env, MOVING)                      # the fixed cameras never change
    rig = camera_io.Rig(env, CANON, S1D)
    rig.read()
    for step in range(2):
        for name in MOVING:
            screens[name].rgb[...] = 60 + step           # the moving cameras are fresh at every step
        got = rig.read()
        assert got["rerendered"] == camera_io.MAX_RERENDERS, step
        assert all(int(got["frames"][name][0, 0, 0]) == VALUE[name] for name in FIXED), step
    assert log == ["render"] * (2 * camera_io.MAX_RERENDERS)


def test_a_step_zero_frame_that_repeats_the_warm_up_is_rendered_again():
    # warm() renders the reset state; if the renderer then discards step 0's frame, the read returns
    # the last warm-up render -- that must count as unchanged, not as step 0.
    env, _, log = _env()
    _screens(env, NAMES)
    rig = camera_io.Rig(env, CANON, S1D)
    rig.warm(2)
    log.clear()
    assert rig.read()["rerendered"] == 1 and log == ["render"]


# ---- what the rig refuses ---------------------------------------------------------------------

@pytest.mark.parametrize("step_dt,fps", [(0.02, 30), (0.0201, 50)])
def test_rig_refuses_a_canon_frame_rate_that_is_not_the_step_period(step_dt, fps):
    # one frame per recorded step: the videos' frame period has to be the env's step period. 0.0201 s
    # against 50 fps rounds to the same rate but drifts 99.9 ms by frame 999.
    env, *_ = _env()
    env.step_dt = step_dt
    canon = copy.deepcopy(CANON)
    canon["fps"] = fps
    with pytest.raises(ValueError):
        camera_io.Rig(env, canon, S1D)


def test_rig_refuses_a_mounted_camera_whose_parent_is_not_one_robot_body():
    env, *_ = _env()
    canon = copy.deepcopy(CANON)
    name = MOUNTED[-1]
    canon["cameras"][name]["parent_prim"] = "no_such_link"
    with pytest.raises(ValueError, match=f"{name}'s parent 'no_such_link' matched 0 robot bodies"):
        camera_io.Rig(env, canon, S1D)


def test_rig_refuses_a_camera_type_it_cannot_place():
    env, *_ = _env()
    canon = copy.deepcopy(CANON)
    canon["cameras"][NAMES[-1]]["type"] = "orbit"
    with pytest.raises(ValueError, match=f"{NAMES[-1]}.*'orbit'"):
        camera_io.Rig(env, canon, S1D)


# ---- a declared pack --------------------------------------------------------------------------

def test_rig_reads_the_pack_from_the_camera_the_canon_names():
    # The canon decides whose depth, mask and labels the pack holds: the pack sits on
    # bbox.camera (cameras.load), here another camera, and everything is read off it.
    env, cams, _ = _env()
    c = CANON["cameras"][OTHER]
    cams[OTHER].data.output["distance_to_image_plane"] = np.full((1, c["height"], c["width"], 1), 7.5, np.float32)
    cams[OTHER].data.output["semantic_segmentation"] = np.full((1, c["height"], c["width"], 1), 3, np.int32)
    cams[OTHER].data.info = {"semantic_segmentation": {"idToLabels": {"(3, 0, 0, 0)": {"class": "rack"}}}}
    canon = canon_with_pack(_box_on(_packless(CANON), OTHER))
    assert canon["packs"][PACK_NAME]["camera"] == OTHER
    got = camera_io.Rig(env, canon, S1D).read()
    assert float(got["depth"][0, 0]) == 7.5 and int(got["seg"][0, 0]) == 3
    assert got["id_to_labels"] == {"(3, 0, 0, 0)": {"class": "rack"}}


def test_rig_reads_depth_from_the_channel_the_canon_names():
    env, cams, _ = _env()
    canon = canon_with_pack(CANON)
    canon["packs"][PACK_NAME]["depth"]["source"] = "depth_elsewhere"
    c = CANON["cameras"][BOX]
    cams[BOX].data.output["depth_elsewhere"] = np.full((1, c["height"], c["width"], 1), 7.5, np.float32)
    assert float(camera_io.Rig(env, canon, S1D).read()["depth"][0, 0]) == 7.5
