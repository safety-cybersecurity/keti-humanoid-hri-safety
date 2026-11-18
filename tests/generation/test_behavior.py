"""The behaviour layer (generation/behavior.py): a pass-through any policy can be swapped for, a delay
line for what the robot perceives, and the loader for ``module:name``."""

import json

import pytest

from generation import behavior


def _input(command=(0.7, -0.2)):
    return behavior.BehaviorInput(1.0, (0.0, 0.0), 0.0, (0.7, 0.0), 0.0, command, "walk",
                                  behavior.Perceived((1.0, 0.0), (-0.5, 0.0)))


def test_pass_through_is_the_identity():
    assert behavior.PassThrough()(_input()) == (0.7, -0.2)
    assert behavior.PassThrough()(_input((0.0, 0.0))) == (0.0, 0.0)


def test_delay_line_starts_empty():
    line = behavior.DelayLine(2)
    assert [line.push(x) for x in "abcd"] == [None, None, "a", "b"]
    assert behavior.DelayLine(0).push("z") == "z"
    with pytest.raises(ValueError):
        behavior.DelayLine(-1)


def test_delay_steps_round_to_the_nearest_step():
    assert behavior.delay_steps(0.0, 0.02) == 0
    assert behavior.delay_steps(100.0, 0.02) == 5
    assert behavior.delay_steps(50.0, 0.02) == 3              # a half rounds up
    assert behavior.delay_steps(49.0, 0.02) == 2
    assert behavior.delay_steps(29.0, 0.02) == 1
    with pytest.raises(ValueError):
        behavior.delay_steps(-1.0, 0.02)


def test_load_behavior_finds_a_class_and_a_function():
    assert isinstance(behavior.load_behavior("generation.behavior:PassThrough"), behavior.PassThrough)
    assert behavior.load_behavior("os.path:basename") is __import__("os").path.basename


@pytest.mark.parametrize("spec", ["", "nocolon", "a:b:c", ":name", "generation.behavior:", "no_such_module_xyz:Thing",
                                  "generation.behavior:NoSuchName", "generation.behavior:Perceived.__doc__", 3])
def test_load_behavior_refuses_what_it_cannot_load(spec):
    with pytest.raises(ValueError):
        behavior.load_behavior(spec)


def test_load_behavior_refuses_a_class_that_needs_arguments():
    # Perceived is a class, but not one that can be made without arguments: the loader says so instead of letting
    # the constructor's TypeError escape.
    with pytest.raises(ValueError, match="generation.behavior:Perceived"):
        behavior.load_behavior("generation.behavior:Perceived")


def test_load_behavior_refuses_a_module_that_fails_while_importing(tmp_path, monkeypatch):
    (tmp_path / "broken_behaviour_xyz.py").write_text("raise RuntimeError('broken at import')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match="broken_behaviour_xyz:Thing"):
        behavior.load_behavior("broken_behaviour_xyz:Thing")


def test_load_behavior_refuses_a_class_whose_constructor_fails(tmp_path, monkeypatch):
    (tmp_path / "failing_init_xyz.py").write_text(
        "class Layer:\n    def __init__(self):\n        raise RuntimeError('no')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match="failing_init_xyz:Layer"):
        behavior.load_behavior("failing_init_xyz:Layer")


def test_parse_spec_splits_a_module_and_a_name():
    assert behavior.parse_spec("generation.behavior:PassThrough") == ("generation.behavior", "PassThrough")
    assert behavior.parse_spec(behavior.DEFAULT_BEHAVIOR) == ("generation.behavior", "PassThrough")
    for spec in ("", "nocolon", "a:b:c", ":name", "module:", 3, None):
        with pytest.raises(ValueError):
            behavior.parse_spec(spec)


def test_policy_record_names_the_checkpoint_without_the_machines_paths(tmp_path):
    import hashlib

    blob = b"weights"
    digest = hashlib.sha256(blob).hexdigest()
    published = tmp_path / "cache" / "model_1000.pt"
    published.parent.mkdir()
    published.write_bytes(blob)
    record = behavior.policy_record(published, published=True, behavior_spec=behavior.DEFAULT_BEHAVIOR,
                                    repo_root=tmp_path / "repo")
    assert record == {"source": f"isaaclab-published:rsl_rl/{behavior.PUBLISHED_POLICY_TASK}", "sha256": digest,
                      "behavior": behavior.DEFAULT_BEHAVIOR}
    inside = tmp_path / "repo" / "policies" / "mine.pt"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(blob)
    assert behavior.policy_record(inside, published=False, behavior_spec="m:B",
                                  repo_root=tmp_path / "repo")["source"] == "file:policies/mine.pt"
    outside = tmp_path / "elsewhere" / "theirs.pt"
    outside.parent.mkdir()
    outside.write_bytes(blob)
    record = behavior.policy_record(outside, published=False, behavior_spec="m:B", repo_root=tmp_path / "repo")
    assert record["source"] == "file:theirs.pt" and str(tmp_path) not in json.dumps(record)
