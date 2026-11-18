import numpy as np

from generation.dummy.gait.held import held_update


def test_only_held_environments_move():
    old = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    new = np.zeros((3, 2))
    out = held_update(old, new, np.array([True, False, True]))
    assert (out == np.array([[0, 0], [2, 2], [0, 0]])).all()


def test_vectors_too():
    assert (held_update(np.array([0.3, 0.6]), np.array([0.4, 0.7]), np.array([False, True])) == [0.3, 0.7]).all()
