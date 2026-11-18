"""generation/dummy/gait/bvh.py -- BVH parsing and forward kinematics. Pure; runs in CI."""
import math

import numpy as np
import pytest

from generation.dummy.gait import bvh
from tests.generation.gait_fixture import SKELETON, synthetic_walk_bvh

TINY = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Knee
  {
    OFFSET 0 -10 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 -10 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.5
1 2 3 0 0 0 0 0 0
1 2 3 90 0 0 0 0 0
"""


def test_parse_reads_hierarchy_channels_and_frames():
    m = bvh.parse_bvh(TINY)
    assert m.skeleton.names == ("Hips", "Knee")
    assert m.skeleton.parents == (-1, 0)
    assert m.skeleton.channels[1] == ("Zrotation", "Xrotation", "Yrotation")
    assert m.frame_time == 0.5 and m.frames.shape == (2, 9)


def test_forward_kinematics_turns_the_child_with_the_root():
    pos, rot = bvh.forward_kinematics(bvh.parse_bvh(TINY), scale=0.01)
    assert pos[0, 1] == pytest.approx([0.01, -0.08, 0.03])          # root (1,2,3) cm + knee offset (0,-10,0)
    assert pos[1, 1] == pytest.approx([0.01 + 0.10, 0.02, 0.03])    # 90 deg about Z sends -Y to +X


def test_a_truncated_motion_is_refused():
    with pytest.raises(ValueError, match="프레임"):
        bvh.parse_bvh(TINY.replace("1 2 3 90 0 0 0 0 0\n", "1 2 3\n"))


def test_unknown_joint_names_say_which():
    m = bvh.parse_bvh(TINY)
    with pytest.raises(KeyError, match="Toe"):
        m.skeleton.index("Toe")


def test_the_fixture_parses_with_every_joint():
    m = bvh.parse_bvh(synthetic_walk_bvh(seconds=1.0))
    assert m.skeleton.names == tuple(n for n, *_ in SKELETON)
    assert m.frames.shape == (60, 6 + 3 * (len(SKELETON) - 1))
