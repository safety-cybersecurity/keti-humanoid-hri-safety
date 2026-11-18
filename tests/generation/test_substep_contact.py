"""서브스텝 접촉 누적기. Isaac Sim 없이 돕니다.

**이 파일이 없던 동안 훅을 무는 시험이 하나도 없었습니다.**
누적기를 스냅샷 한 번 읽는 옛 방식으로 되돌려도 전 시험이 초록이었고, 점검 스크립트는
표본이 1이어도 exit 0 이었습니다. 여기서 계약을 못박습니다.

훅이 부르는 순서는 실측으로 확정돼 있습니다 — 훅은 각 서브스텝의 `sim.step()` **앞**에서
돌므로 읽는 값이 직전 서브스텝 결과입니다. 그래서 한 env 스텝의 호출 순서는

    sample() x 4   (첫 번째는 직전 스텝의 마지막 서브스텝, 나머지가 이 스텝의 0·1·2)
    drain()        (이 스텝의 서브스텝 3)

이고, 누적기는 **첫 번째를 버려야** 창이 정확히 이 스텝의 넷이 됩니다.
"""

from pathlib import Path

import numpy as np
import pytest

from generation.envs.s1a.substep_contact import SubstepContactAccumulator

EPS = 1.0


def make(readings):
    """readings: [(dummy_force, dummy_z, robot_force), ...] 을 순서대로 내보낸다."""
    it = iter(readings)
    box = {}

    def advance():
        box["cur"] = next(it)

    def dummy_force():
        advance()
        return np.asarray(box["cur"][0], float)

    def dummy_z():
        return np.asarray(box["cur"][1], float)

    def robot_force():
        return np.asarray(box["cur"][2], float)

    return SubstepContactAccumulator(dummy_force, dummy_z, robot_force,
                                     contact_eps_n=EPS, expected_samples=4)


def step(acc, hook_calls=4):
    for _ in range(hook_calls):
        acc.sample()
    return acc.drain()


def test_a_normal_step_folds_four_substeps_not_five():
    acc = make([([0.0], [1.0], [0.0])] * 5)
    out = step(acc)
    assert out["samples"] == 4, "첫 표본을 버리지 않으면 5가 됩니다"
    assert acc.last_samples == 4


def test_the_peak_is_the_maximum_over_this_step_substeps():
    # 버려지는 첫 표본 · 이 스텝의 0·1·2 · drain 이 읽는 3
    acc = make([([0.0], [1.0], [0.0]),
                ([5.0], [1.0], [0.0]),
                ([90.0], [1.0], [0.0]),
                ([7.0], [1.0], [0.0]),
                ([3.0], [1.0], [0.0])])
    assert step(acc)["dummy_force"] == pytest.approx([90.0])


def test_a_contact_living_only_in_an_inner_substep_survives():
    """스냅샷이 잃던 바로 그 접촉. 마지막 표본만 보면 0 입니다."""
    acc = make([([0.0], [1.0], [0.0]),
                ([0.0], [1.0], [0.0]),
                ([812.0], [1.0], [50.0]),   # 안쪽 서브스텝에서만 존재
                ([0.0], [1.0], [0.0]),
                ([0.0], [1.0], [0.0])])
    assert step(acc)["dummy_force"] == pytest.approx([812.0])


def test_the_previous_step_last_substep_does_not_bleed_into_this_one():
    """되돌리기 대상. 첫 표본을 버리지 않으면 이 접촉이 두 스텝에 기록됩니다."""
    tail = ([600.0], [1.0], [50.0])          # 스텝 t-1 의 마지막 서브스텝
    quiet = ([0.0], [1.0], [0.0])
    acc = make([quiet, quiet, quiet, quiet, tail,      # 스텝 t-1: drain 이 tail 을 읽음
                tail, quiet, quiet, quiet, quiet])     # 스텝 t: 첫 표본이 같은 tail
    first = step(acc)
    second = step(acc)
    assert first["dummy_force"] == pytest.approx([600.0]), "직전 스텝이 접촉을 놓쳤습니다"
    assert second["dummy_force"] == pytest.approx([0.0]), \
        "직전 스텝의 마지막 서브스텝이 이 스텝 최댓값에 섞였습니다"


def test_the_height_comes_from_the_substep_where_the_force_peaked():
    """유형 판정이 힘과 높이를 다른 시점에서 가져오면 표 A.2 열 선택 근거가 무너집니다."""
    acc = make([([0.0], [9.0], [0.0]),
                ([10.0], [0.05], [50.0]),    # 낮은 자세에서 작은 힘
                ([900.0], [1.40], [50.0]),   # 최댓값은 서 있는 높이에서
                ([20.0], [0.05], [50.0]),
                ([5.0], [0.05], [50.0])])    # 마지막 표본은 다시 낮음
    out = step(acc)
    assert out["dummy_force"] == pytest.approx([900.0])
    assert out["dummy_z_at_peak"] == pytest.approx([1.40]), \
        "기록 시점 높이를 쓰면 자유 충돌이 협착으로 뒤집힙니다"


def test_the_robot_gate_comes_from_the_same_substep_as_the_peak():
    acc = make([([0.0], [1.0], [0.0]),
                ([700.0], [1.0], [0.0]),    # 최댓값 순간에는 로봇 접촉 없음 (지면 하중)
                ([10.0], [1.0], [90.0]),    # 로봇 접촉은 다른 서브스텝
                ([10.0], [1.0], [0.0]),
                ([10.0], [1.0], [0.0])])
    out = step(acc)
    assert out["dummy_force"] == pytest.approx([700.0])
    assert out["robot_gate_at_peak"].tolist() == [False], \
        "다른 서브스텝의 로봇 접촉이 이 최댓값을 로봇-인체 접촉으로 만들었습니다"


def test_the_hold_release_backstop_sees_an_inner_substep():
    """홀드 해제가 스냅샷을 보면 안쪽 서브스텝 접촉을 놓쳐 오염 스텝이 생깁니다."""
    acc = make([([0.0], [1.0], [0.0]),
                ([0.0], [1.0], [0.0]),
                ([0.0], [1.0], [77.0]),     # 안쪽에서만 로봇 접촉
                ([0.0], [1.0], [0.0]),
                ([0.0], [1.0], [0.0])])     # robot_contact_seen 이 확정하는 마지막 서브스텝
    for _ in range(4):
        acc.sample()
    assert acc.robot_contact_seen() is True


def test_the_backstop_also_sees_a_contact_that_lives_only_in_the_last_substep():
    """직전 수정이 연 구멍. 백스톱이 서브스텝 0~2 만 보면 마지막 것을 놓칩니다.

    그러면 데이터에는 접촉이 남는데 홀드는 걸린 채라 오염 스텝이 생깁니다.
    """
    acc = make([([0.0], [1.0], [0.0]),
                ([0.0], [1.0], [0.0]),
                ([0.0], [1.0], [0.0]),
                ([0.0], [1.0], [0.0]),
                ([0.0], [1.0], [88.0])])    # 마지막 서브스텝에만 접촉
    for _ in range(4):
        acc.sample()
    assert acc.robot_contact_seen() is True, "마지막 서브스텝을 확정하지 않았습니다"


def test_the_backstop_and_the_recorder_read_one_frozen_batch():
    """확정이 멱등하지 않으면 표본이 하나 더 세어져 기대 수와 어긋납니다."""
    acc = make([([0.0], [1.0], [0.0])] * 5)
    for _ in range(4):
        acc.sample()
    acc.robot_contact_seen()
    acc.robot_contact_seen()
    out = acc.drain()
    assert out["samples"] == 4


def test_reset_drops_a_partial_batch():
    """에피소드 경계에서 부분 batch 를 버린다. 운영 순서(리셋 → 스텝)를 그대로 따른다."""
    acc = make([([5.0], [1.0], [0.0])] * 2 + [([0.0], [1.0], [0.0])] * 5)
    acc.sample(); acc.sample()
    acc.reset()
    out = step(acc)
    assert out["dummy_force"] == pytest.approx([0.0]), "리셋 전 최댓값이 남았습니다"
    assert out["samples"] == 4


def test_a_height_array_of_the_wrong_length_is_refused():
    acc = make([([0.0, 0.0], [1.0], [0.0])] * 2)
    # 버려지는 첫 표본에서도 읽기는 일어나므로 거기서 바로 걸립니다.
    with pytest.raises(ValueError, match="길이가 다릅니다"):
        acc.sample()


def test_draining_without_any_sample_is_an_error_not_a_silent_zero():
    """훅이 안 걸린 상태를 0 으로 넘기면 스냅샷 시절로 조용히 돌아갑니다."""
    acc = SubstepContactAccumulator(
        lambda: (_ for _ in ()).throw(RuntimeError("센서 없음")),
        lambda: np.zeros(1), lambda: np.zeros(1), contact_eps_n=EPS)
    with pytest.raises(RuntimeError):
        acc.drain()


# ── 발은 매 스텝 지면 하중을 받는다 ────────────────────────────────

def make_masked(readings, mask):
    acc = make(readings)
    acc._non_foot = np.asarray(mask, bool)
    return acc


def test_foot_load_alone_does_not_count_as_robot_contact():
    """이 마스크가 빠진 채 돌린 배치에서 유형이 매겨진 스텝이 229 → 41,990 이었습니다.

    발은 걷는 내내 지면 하중을 받으므로, 발까지 세면 로봇이 늘 인체를 만지고 있다는
    답이 나오고 힘이 조금이라도 있는 모든 부위가 접촉 유형을 얻습니다.
    """
    # 바디 순서 [ankle, ankle, torso] — 발만 하중을 받는 상황
    mask = [False, False, True]
    acc = make_masked([([500.0], [1.0], [800.0, 800.0, 0.0])] * 5, mask)
    for _ in range(4):
        acc.sample()
    assert acc.robot_contact_seen() is False, "발의 지면 하중으로 홀드가 풀립니다"
    out = acc.drain()
    assert out["dummy_force"] == pytest.approx([500.0])
    assert out["robot_gate_at_peak"].tolist() == [False], \
        "발의 지면 하중을 로봇-인체 접촉으로 셌습니다"


def test_a_non_foot_body_does_count():
    mask = [False, False, True]
    acc = make_masked([([500.0], [1.0], [800.0, 800.0, 45.0])] * 5, mask)
    assert step(acc)["robot_gate_at_peak"].tolist() == [True]


def test_the_foot_rule_is_read_off_the_body_names_not_hardcoded():
    from generation.envs.s1a.mdp.safety import non_foot_mask

    names = ["left_ankle_link", "torso_link", "right_ankle_link", "left_elbow_link"]
    assert non_foot_mask(names).tolist() == [False, True, False, True]


# ── 훅이 죽으면 조용히 스냅샷을 기록하지 않는다 ──────────────────

def test_a_batch_with_too_few_samples_is_refused_not_recorded():
    """훅이 안 걸리면 표본이 1이고, 그대로 기록하면 스냅샷 시절로 조용히 돌아갑니다.

    drain() 이 스스로 표본을 하나 만들기 때문에 「최댓값이 비었는가」 검사는 이 경우를
    영영 잡지 못했습니다.
    """
    acc = make([([0.0], [1.0], [0.0])] * 3)
    with pytest.raises(RuntimeError, match="표본이"):
        acc.drain()          # 훅 호출 0회 → 표본 1개


def test_the_expected_count_is_not_enforced_when_it_was_not_given():
    """점검 스크립트처럼 기대 수를 모르는 호출부는 그대로 돌아야 합니다."""
    acc = SubstepContactAccumulator(
        lambda: np.zeros(1), lambda: np.zeros(1), lambda: np.zeros(1),
        contact_eps_n=EPS)
    assert acc.drain()["samples"] == 1


# The dummy's contact sensor lists its bodies in the USD's order and its articulation along the joint tree, and the
# two differ on both dummies. The recorder pairs the sensor's per-body forces and region codes with the
# articulation's per-body heights (here) and velocities (in the recorder), so the articulation's arrays are read
# into the sensor's order by body name -- once, refusing names that do not pair one to one.


def test_the_same_order_maps_onto_itself():
    from generation.envs.s1a.substep_contact import sensor_order

    assert sensor_order(["BR_04", "BR_12_left", "FOOT_left"], ("BR_04", "BR_12_left", "FOOT_left")).tolist() == [0, 1, 2]


def test_a_joint_tree_order_is_read_back_in_the_sensors_order():
    from generation.envs.s1a.substep_contact import sensor_order

    sensor = ["BR_01", "BR_02", "BR_03", "BR_04", "FOOT_left"]
    articulation = ["BR_04", "BR_03", "FOOT_left", "BR_01", "BR_02"]
    heights = np.array([1.43, 1.52, 0.05, 1.66, 1.61])          # the articulation's, in its order
    order = sensor_order(sensor, articulation)
    assert heights[order].tolist() == [1.66, 1.61, 1.52, 1.43, 0.05]
    assert [articulation[k] for k in order] == sensor


@pytest.mark.parametrize("sensor, articulation", [
    (["BR_04", "BR_12_left", "FOOT_left"], ["BR_04", "BR_12_left"]),                       # one missing
    (["BR_04", "BR_12_left"], ["BR_04", "BR_12_left", "FOOT_left"]),                       # one extra
    (["BR_04", "BR_12_left", "FOOT_left"], ["BR_04", "BR_12_left", "FOOT_right"]),         # one renamed
    (["BR_04", "BR_04", "FOOT_left"], ["BR_04", "FOOT_left", "BR_04"]),                    # a name twice
    (["BR_04", "BR_12_left", "BR_04"], ["BR_04", "BR_12_left", "FOOT_left"]),              # twice on one side
    (["BR_04", "BR_04", "FOOT_left"], ["BR_04", "FOOT_left"]),                             # twice in the sensor only
    (["BR_04", "FOOT_left"], ["BR_04", "FOOT_left", "BR_04"]),                             # twice in the articulation only
])
def test_bodies_that_do_not_pair_one_to_one_are_refused(sensor, articulation):
    from generation.envs.s1a.substep_contact import sensor_order

    with pytest.raises(ValueError, match="바디"):
        sensor_order(sensor, articulation)


SUBSTEP_SRC = (Path(__file__).resolve().parents[2] / "generation/envs/s1a/substep_contact.py").read_text(encoding="utf-8")


def test_install_reads_the_heights_in_the_sensors_order():
    block = SUBSTEP_SRC[SUBSTEP_SRC.index("\ndef install("):SUBSTEP_SRC.index("\ndef get(")]
    assert "order = sensor_order(base_env.scene[dummy_sensor].body_names, base_env.scene[dummy_asset].body_names)" in block
    assert "return base_env.scene[dummy_asset].data.body_pos_w.torch[0, :, 2].cpu().numpy()[order]" in block
    assert block.index("order = sensor_order(") < block.index("acc = SubstepContactAccumulator(")
    assert "require_one_body_order" not in SUBSTEP_SRC


def _scene_with(sensor_names, articulation_names, heights):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace as NS

    def sensor(names, force):
        forces = torch.zeros(1, len(names), 3)
        forces[0, :, 2] = force
        return NS(body_names=list(names), data=NS(net_forces_w=NS(torch=forces)))

    pos = torch.zeros(1, len(articulation_names), 3)
    pos[0, :, 2] = torch.tensor(heights)
    scene = {"dummy_contact": sensor(sensor_names, 50.0), "contact_forces": sensor(["pelvis", "left_ankle_link"], 0.0),
             "dummy": NS(body_names=list(articulation_names), data=NS(body_pos_w=NS(torch=pos)))}
    term = NS(apply_actions=lambda: None)
    return NS(scene=scene, action_manager=NS(_terms={"joint_pos": term}, active_terms=["joint_pos"]),
              cfg=NS(decimation=1)), term


def test_installed_sampling_hands_each_sensor_body_its_own_height():
    from generation.envs.s1a.substep_contact import install

    env, term = _scene_with(["BR_01", "BR_02", "BR_03", "BR_04", "FOOT_left"],
                            ["BR_04", "BR_03", "FOOT_left", "BR_01", "BR_02"], [1.43, 1.52, 0.05, 1.66, 1.61])
    acc = install(env, contact_eps_n=EPS)
    term.apply_actions()
    z = acc.drain()["dummy_z_at_peak"]
    assert np.allclose(z, [1.66, 1.61, 1.52, 1.43, 0.05])


def test_install_refuses_a_dummy_whose_bodies_do_not_pair():
    from generation.envs.s1a.substep_contact import install

    env, _ = _scene_with(["BR_01", "BR_02"], ["BR_01", "FOOT_left"], [1.66, 0.05])
    with pytest.raises(ValueError, match="바디"):
        install(env, contact_eps_n=EPS)
