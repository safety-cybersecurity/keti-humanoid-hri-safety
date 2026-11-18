"""Per-substep contact sampling, because the record-step snapshot loses contacts.

``EpisodeRecorder.step()`` used to read ``contact.data.net_forces_w`` once per
env step. That is the value at one instant, and an env step is four physics
substeps: a contact born and gone inside one of the other three is not recorded
short, it is recorded as exactly zero and disappears. For a dataset whose reason
to exist is near misses and failures, that is the wrong thing to lose.

Raising fps would work and cost storage. It is not needed. Isaac Lab's sensors
are lazily evaluated -- ``SensorBase.update()`` only marks the buffers outdated,
and ``_update_buffers_impl`` runs when ``.data`` is touched -- so on the PhysX
path, where ``handles_decimation()`` is False and ``ManagerBasedRLEnv.step()``
runs the substep loop itself, touching ``.data`` once per substep makes the
sensor compute once per substep. ``action_manager.apply_action()`` is the hook:
the substep loop calls it every substep and nothing else it calls there touches
sensor data.

**The window is the four substeps of this env step, and getting that right took
a measurement.** The hook fires before each substep's ``sim.step()``, so a read
inside it returns the *previous* substep's result. Written naively the batch
therefore held five samples spanning ``[t-1 substep 3, t substep 0..2]`` plus one
more from ``drain()``, and a contact living only in the previous step's last
substep was recorded again here -- inflating duration by a step and firing
``contact_onset`` a step early. Probed on 2026-08-25: the first sample of a step
equals the last sample of the previous step in **119 of 119 steps (100.0%)**, so
the first one is dropped from the peak and the batch is exactly this step's four
substeps. Within a step, adjacent samples differ 85.8% of the time -- the
resolution is real, not an artefact of reading one buffer repeatedly.

**Each peak carries the state that goes with it.** Force alone cannot choose a
Table A.2 column: the type also depends on whether the region was pressed
against the floor and whether the robot was touching anything at that moment.
Taking those from whichever substep each happened to be extreme in would let one
substep's robot contact type another substep's peak force. So the region height
and the robot-contact gate are captured *at the substep where that region's force
peaked*.

**The gate is global, not per region, and the array does not make it otherwise.**
It answers "was the robot touching anything that is not the ground at that
substep", because per-pair attribution is not available on this pipeline -- the
filtered-sensor route was tried and failed (schema §8.1). Two regions peaking in
the same substep therefore get the same answer even if only one of them is what
the robot touched. That limit is the pre-existing one §8.1 records; narrowing the
time axis does not narrow the space axis, and the field name says gate rather
than contact so the array is not read as an attribution it cannot make.

Everything here is numpy and the env is reached only through reader callables
built in ``install()``, which is what lets the accumulator be tested without
Isaac. The readers convert off the GPU per substep; against a 30 s episode that
cost is not visible.
"""

from __future__ import annotations

import numpy as np

_ATTR = "_keti_substep_contact"


class SubstepContactAccumulator:
    """Per-substep peaks for one env step, with the state that goes with each."""

    def __init__(self, read_dummy_force, read_dummy_z, read_robot_force, *,
                 contact_eps_n: float, non_foot_mask=None, expected_samples=None):
        self._read_dummy_force = read_dummy_force
        self._read_dummy_z = read_dummy_z
        self._read_robot_force = read_robot_force
        self._eps = float(contact_eps_n)
        # Which robot bodies count as "touching something that is not the
        # ground". Without it the feet answer yes on every step and every region
        # carrying any force gets a contact type -- measured at 180x too many.
        self._non_foot = None if non_foot_mask is None else np.asarray(non_foot_mask, bool)
        # How many samples a complete batch must hold: one per physics substep.
        # Without it drain()'s own sample makes the "did the hook fire" guard
        # unreachable -- the peak is never empty, so a dead hook records a
        # snapshot per step and says nothing. That is the exact failure this
        # module exists to prevent, left fail-open.
        self._expected = None if expected_samples is None else int(expected_samples)
        self.installed = False
        # How many samples went into the batch drain() last returned. 4 is
        # normal, one per physics substep. 1 means the hook is not firing and
        # the data is back to one snapshot per record step -- the defect this
        # module exists to prevent, and one that looks like nothing at all.
        self.last_samples = 0
        self._reset_batch()

    # ── batch state ──────────────────────────────────────────────────────
    def finalize(self) -> None:
        """Take this step's last substep, once. Idempotent.

        The hook fires *before* each substep, so after the physics loop the batch
        holds substeps 0..2 and the fourth has not been read. Whoever asks first
        must therefore complete it, or two consumers of the same step answer from
        different windows: the hold-release backstop runs during command compute
        and saw 0..2, while the recorder's drain() later saw 0..3, so a contact
        living only in the final substep reached the data with the hold still
        applied.

        Idempotent because both call it and the sample must not be counted twice.
        """
        if self._finalized:
            return
        self.sample(force=True)
        self._finalized = True

    def _reset_batch(self) -> None:
        self._dummy_peak: np.ndarray | None = None
        self._dummy_z: np.ndarray | None = None
        self._dummy_robot_contact: np.ndarray | None = None
        self._robot_peak: np.ndarray | None = None
        self._samples = 0
        self._finalized = False
        # The next sample is the previous batch's final one (measured 100.0%);
        # folding it in again would report that contact in two steps.
        self._skip_next = True

    def reset(self) -> None:
        """Drop a partial batch. Called at an episode boundary, not per step."""
        self._reset_batch()

    # ── sampling ─────────────────────────────────────────────────────────
    def sample(self, *, force: bool = False) -> None:
        # Read first and unconditionally. The read *is* the mechanism -- touching
        # .data is what makes the lazily-evaluated sensor compute this substep --
        # so a sample dropped from the peak must still be taken. Skipping the
        # read as well would leave the sensor uncomputed at that substep and take
        # the resolution back out.
        dummy_f = np.asarray(self._read_dummy_force(), dtype=float)
        dummy_z = np.asarray(self._read_dummy_z(), dtype=float)
        robot_f = np.asarray(self._read_robot_force(), dtype=float)
        if dummy_f.shape != dummy_z.shape:
            raise ValueError(
                f"인체 접촉력{dummy_f.shape} 와 높이{dummy_z.shape} 의 길이가 다릅니다")

        if self._skip_next and not force:
            self._skip_next = False
            return
        self._skip_next = False

        robot_contact_now = self._robot_touching(robot_f, self._eps)
        if self._dummy_peak is None:
            self._dummy_peak = dummy_f.copy()
            self._dummy_z = dummy_z.copy()
            self._dummy_robot_contact = np.full(dummy_f.shape, robot_contact_now)
            self._robot_peak = robot_f.copy()
        else:
            # Strictly greater: the earliest substep reaching a value owns it,
            # which keeps the captured height and contact flag deterministic.
            newer = dummy_f > self._dummy_peak
            self._dummy_peak = np.where(newer, dummy_f, self._dummy_peak)
            self._dummy_z = np.where(newer, dummy_z, self._dummy_z)
            self._dummy_robot_contact = np.where(
                newer, robot_contact_now, self._dummy_robot_contact)
            self._robot_peak = np.maximum(self._robot_peak, robot_f)
        self._samples += 1

    def _robot_touching(self, robot_f: np.ndarray, limit: float) -> bool:
        gated = robot_f if self._non_foot is None else robot_f[self._non_foot]
        return bool((gated > limit).any())

    def robot_contact_seen(self, threshold_n=None) -> bool:
        """Whether any substep of this batch so far showed robot contact.

        The dummy's hold-release backstop reads this rather than a snapshot. It
        used to read the instantaneous value while the recorder judged contact
        off the peak, so a contact living in an inner substep was written to the
        data while the hold stayed on -- the two have to ask the same question or
        the contamination counter is measuring the asymmetry rather than the
        physics.

        The threshold stays the caller's. The release backstop gates at 10 N and
        the recorder at a noise-floor epsilon; sharing the peak should not merge
        two different questions into one.
        """
        self.finalize()
        if self._robot_peak is None:
            return False
        return self._robot_touching(
            self._robot_peak, self._eps if threshold_n is None else float(threshold_n))

    # ── draining ─────────────────────────────────────────────────────────
    def drain(self) -> dict:
        """This env step's peaks, with the state that goes with each. Resets."""
        self.finalize()
        if self._dummy_peak is None:
            raise RuntimeError("접촉 표본이 하나도 없습니다")
        if self._expected is not None and self._samples != self._expected:
            raise RuntimeError(
                f"이 스텝의 접촉 표본이 {self._samples}개입니다 (기대 {self._expected}). "
                "훅이 서브스텝마다 걸리지 않았다는 뜻이고, 그대로 기록하면 두 기록 스텝 "
                "사이의 접촉이 데이터에서 통째로 빠집니다")
        out = {
            "dummy_force": self._dummy_peak,
            "dummy_z_at_peak": self._dummy_z,
            # Named "gate", not "contact": it is the global robot-touching signal
            # sampled at each region's peak substep, not an attribution of the
            # contact to that region. See the module docstring and schema §8.1.
            "robot_gate_at_peak": self._dummy_robot_contact.astype(bool),
            "robot_force": self._robot_peak,
            "samples": self._samples,
        }
        self.last_samples = self._samples
        self._reset_batch()
        return out

    # ── wiring ───────────────────────────────────────────────────────────
    def install(self, base_env) -> "SubstepContactAccumulator":
        manager = base_env.action_manager
        term = manager._terms[manager.active_terms[0]]
        original = term.apply_actions

        def apply_and_sample():
            original()
            self.sample()

        term.apply_actions = apply_and_sample
        self.installed = True
        setattr(base_env, _ATTR, self)
        return self


def sensor_order(sensor_body_names, articulation_body_names) -> np.ndarray:
    """Indices into the articulation's bodies, listed in the contact sensor's body order.

    The sensor lists the dummy's bodies in the USD's order and the articulation along its joint tree, and the two
    differ. The sensor's per-body forces and region codes are paired with the articulation's per-body heights here
    and with its per-body velocities in the recorder, so the articulation's arrays are read as ``array[order]``.
    Paired by body name, once; names that do not pair one to one are refused rather than guessed."""
    sensor, bodies = list(sensor_body_names), list(articulation_body_names)
    if len(set(sensor)) != len(sensor) or len(set(bodies)) != len(bodies) or set(sensor) != set(bodies):
        raise ValueError(f"더미의 접촉 센서와 관절체의 바디가 이름으로 하나씩 짝지어지지 않습니다 — 센서 {sensor} · 관절체 {bodies}")
    index = {name: k for k, name in enumerate(bodies)}
    return np.array([index[name] for name in sensor], dtype=int)


def install(base_env, *, contact_eps_n: float,
            robot_sensor: str = "contact_forces", dummy_sensor: str = "dummy_contact",
            dummy_asset: str = "dummy") -> SubstepContactAccumulator:
    """Wrap the live action term so both contact sensors compute every substep."""
    import torch

    def magnitudes(name):
        def read():
            sensor = base_env.scene[name]
            return torch.linalg.norm(
                sensor.data.net_forces_w.torch[0], dim=-1).cpu().numpy()
        return read

    # The articulation's heights in the sensor's body order -- sample() pairs them with the sensor's forces.
    order = sensor_order(base_env.scene[dummy_sensor].body_names, base_env.scene[dummy_asset].body_names)

    def heights():
        return base_env.scene[dummy_asset].data.body_pos_w.torch[0, :, 2].cpu().numpy()[order]

    from generation.envs.s1a.mdp.safety import non_foot_mask

    acc = SubstepContactAccumulator(
        magnitudes(dummy_sensor), heights, magnitudes(robot_sensor),
        contact_eps_n=contact_eps_n,
        non_foot_mask=non_foot_mask(base_env.scene[robot_sensor].body_names),
        # One sample per physics substep. Read off the env rather than written
        # down, so raising decimation cannot silently leave the guard behind.
        expected_samples=base_env.cfg.decimation)
    return acc.install(base_env)


def get(base_env) -> SubstepContactAccumulator:
    """The accumulator installed on this env.

    Absent means the recorder would silently fall back to one snapshot per env
    step, which is the defect. Raising is the point.
    """
    acc = getattr(base_env, _ATTR, None)
    if acc is None:
        raise RuntimeError(
            "서브스텝 접촉 누적기가 설치되지 않았습니다 — "
            "generation.envs.s1a.substep_contact.install(env) 을 먼저 부르십시오. "
            "없이 기록하면 두 기록 스텝 사이의 접촉이 데이터에서 통째로 빠집니다")
    return acc


# ── the scenarios that declare R-TS15066-SSM-02 ─────────────────────────────

_HRI_ATTR = "_keti_hri_contact"


class HriContactAccumulator:
    """Per-substep peaks, over one env step, of what the robot's links and the carried box gave each of the person's
    bodies -- read from the dummy contact sensor's force matrix, so the floor and the racks are left out
    (generation/envs/hri/record_logic.attribute_contacts) -- with the body's height at the substep its force peaked,
    and of what each robot link and the box took from anything that is not the person (link_residuals,
    box_residual). The window is SubstepContactAccumulator's: the hook reads the substep before it, so the first
    sample of a step is dropped and drain() reads the step's last substep."""

    def __init__(self, read_matrix, read_dummy_z, read_robot_net, read_box_net=None, *, robot_cols, box_col,
                 link_cols, expected_samples=None):
        self._read_matrix = read_matrix
        self._read_dummy_z = read_dummy_z
        self._read_robot_net = read_robot_net
        self._read_box_net = read_box_net
        if (read_box_net is None) != (box_col is None):
            raise ValueError("든 상자의 접촉 센서와 상자 열은 함께 있거나 함께 없습니다")
        self._robot_cols = list(robot_cols)
        self._box_col = box_col
        self._link_cols = list(link_cols)
        self._expected = None if expected_samples is None else int(expected_samples)
        self.installed = False
        self.last_samples = 0
        self._reset_batch()

    def _reset_batch(self) -> None:
        self._human: np.ndarray | None = None
        self._human_z: np.ndarray | None = None
        self._links: np.ndarray | None = None
        self._box: float | None = None
        self._samples = 0
        self._skip_next = True

    def reset(self) -> None:
        """Drop a partial batch. Called at an episode boundary, not per step."""
        self._reset_batch()

    def sample(self, *, force: bool = False) -> None:
        from generation.envs.hri import record_logic

        # Read first and unconditionally: the read is what makes the lazily evaluated sensors compute this substep.
        matrix = np.asarray(self._read_matrix(), dtype=float)
        z = np.asarray(self._read_dummy_z(), dtype=float)
        robot_net = np.asarray(self._read_robot_net(), dtype=float)
        box_net = None if self._read_box_net is None else np.asarray(self._read_box_net(), dtype=float)
        if self._skip_next and not force:
            self._skip_next = False
            return
        self._skip_next = False
        human = record_logic.attribute_contacts(matrix, self._robot_cols, self._box_col)
        if human.shape != z.shape:
            raise ValueError(f"사람 바디의 힘{human.shape} 과 높이{z.shape} 의 길이가 다릅니다")
        links = record_logic.link_residuals(robot_net, matrix, self._link_cols)
        box = None if box_net is None else record_logic.box_residual(box_net, matrix, self._box_col)
        if self._human is None:
            self._human, self._human_z, self._links, self._box = human.copy(), z.copy(), links.copy(), box
        else:
            # Strictly greater: the earliest substep reaching a value owns it (as SubstepContactAccumulator).
            newer = human > self._human
            self._human = np.where(newer, human, self._human)
            self._human_z = np.where(newer, z, self._human_z)
            self._links = np.maximum(self._links, links)
            if box is not None:
                self._box = max(self._box, box)
        self._samples += 1

    def drain(self) -> dict:
        """This env step's peaks. Resets."""
        self.sample(force=True)
        if self._expected is not None and self._samples != self._expected:
            raise RuntimeError(
                f"이 스텝의 사람 접촉 행렬 표본이 {self._samples}개입니다 (기대 {self._expected}). "
                "훅이 서브스텝마다 걸리지 않았다는 뜻이고, 그대로 기록하면 두 기록 스텝 사이의 접촉이 데이터에서 빠집니다")
        out = {"human_force": self._human, "human_z_at_peak": self._human_z, "link_residual": self._links,
               "box_residual": self._box, "samples": self._samples}
        self.last_samples = self._samples
        self._reset_batch()
        return out

    def install(self, base_env) -> "HriContactAccumulator":
        manager = base_env.action_manager
        term = manager._terms[manager.active_terms[0]]
        original = term.apply_actions

        def apply_and_sample():
            original()
            self.sample()

        term.apply_actions = apply_and_sample
        self.installed = True
        setattr(base_env, _HRI_ATTR, self)
        return self


def install_hri(base_env, *, robot_sensor: str = "contact_forces", dummy_sensor: str = "dummy_contact",
                dummy_asset: str = "dummy", box_sensor: str = "box_contact",
                box_prim: str | None = None) -> HriContactAccumulator:
    """Wrap the live action term so the dummy sensor's force matrix, the robot's and the box's net forces are read
    every substep. The matrix's columns are matched to the robot's links and the box by name from the sensor's filter
    expressions; ``box_prim`` None reads the box's prim name from the environment when it carries one."""
    from generation.envs.hri import record_logic
    from generation.envs.s1a.mdp.safety import non_foot_mask

    dummy = base_env.scene[dummy_sensor]
    robot = base_env.scene[robot_sensor]
    filters = list(dummy.cfg.filter_prim_paths_expr or [])
    if not filters:
        raise ValueError("인형의 접촉 센서에 로봇 링크 · 상자 필터가 없습니다 — 거른 행렬 없이는 로봇과 상자의 힘을 가를 수 없습니다")
    count = int(dummy.contact_view.filter_count)
    if count != len(filters):
        raise ValueError(f"인형의 접촉 센서가 필터 {count} 개를 봅니다 — 설정은 {len(filters)} 개입니다")
    carried = box_sensor in base_env.scene.keys()
    if box_prim is None and carried:
        from generation.envs.hri.carry_rig import BOX_PRIM

        box_prim = BOX_PRIM
    robot_cols, box_col, link_cols = record_logic.matrix_columns(
        filters, list(robot.body_names), box_prim if carried else None, non_foot_mask(robot.body_names))
    order = sensor_order(dummy.body_names, base_env.scene[dummy_asset].body_names)

    def read_matrix():
        matrix = dummy.data.force_matrix_w
        return (matrix.torch if hasattr(matrix, "torch") else matrix)[0].detach().cpu().numpy()

    def read_z():
        return base_env.scene[dummy_asset].data.body_pos_w.torch[0, :, 2].cpu().numpy()[order]

    def read_robot():
        return robot.data.net_forces_w.torch[0].detach().cpu().numpy()

    read_box = None
    if carried:
        box = base_env.scene[box_sensor]

        def read_box():
            return box.data.net_forces_w.torch[0, 0].detach().cpu().numpy()

    acc = HriContactAccumulator(read_matrix, read_z, read_robot, read_box, robot_cols=robot_cols, box_col=box_col,
                                link_cols=link_cols, expected_samples=base_env.cfg.decimation)
    return acc.install(base_env)


def get_hri(base_env) -> HriContactAccumulator:
    """The force-matrix accumulator installed on this env (install_hri); raising when there is none, as get() does."""
    acc = getattr(base_env, _HRI_ATTR, None)
    if acc is None:
        raise RuntimeError(
            "사람 접촉 행렬의 서브스텝 누적기가 설치되지 않았습니다 — "
            "generation.envs.s1a.substep_contact.install_hri(env) 을 먼저 부르십시오")
    return acc
